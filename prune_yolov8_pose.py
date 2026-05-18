"""Minimal iterative pruning for YOLOv8 detection / pose models using fasterai.

The core idea is simple: prune a fraction of channels, fine-tune to recover,
repeat. Everything else is boilerplate required by the ultralytics framework.

What the script DOES:
    1. Swap C2f blocks for a pruning-friendly variant (torch-pruning can't
       trace C2f's .chunk() operation — this is the only model surgery needed).
    2. Ask torch-pruning's dependency graph which channels to remove, except
       for (a) the 3-channel stem (RGB input is fixed) and (b) the head's
       output convs (output dims are fixed by the task).
    3. Iterate: prune → fine-tune with ultralytics' default training loop.

What the script does NOT do:
    - Knowledge distillation (orthogonal; write a separate script if needed).
    - Hardcode layer indices per architecture.
    - Tweak augmentation schedules.
    - Pretty-print per-step recovery tables.

Usage:
    python prune_yolov8_pose.py --data coco8-pose.yaml
    python prune_yolov8_pose.py --model yolov8s-pose.pt --steps 5 --ratio 0.1
"""

import argparse
import json
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, C2f, Conv

from fasterai.prune.all import Pruner, Schedule, large_final, movement, updating_movement, sched_onecycle


# Tier thresholds and per-layer ratios for sensitivity-driven asymmetric pruning.
# Layers are bucketed by their pose_drop @ ratio=0.30 sensitivity probe; the
# global weighted-by-filter-count ratio lands at ~12.5% with the values below.
SENSITIVITY_TIERS = (
    # (pose_drop_threshold, per_layer_ratio)
    (0.100, 0.00),  # PROTECT
    (0.050, 0.03),  # MILD
    (0.020, 0.06),  # LIGHT
    (0.010, 0.10),  # MODERATE
    (0.005, 0.15),  # STRONG
    (0.000, 0.22),  # AGGRESSIVE (default — covers everything below 0.005)
)


def load_asymmetric_ratios(
    jsonl_path: Path | None = None,
    scale: float = 1.0,
) -> dict[str, float]:
    """Load sensitivity_results.jsonl and bucket each layer's pose_drop into
    a tiered per-layer pruning ratio. `scale` multiplies all ratios uniformly,
    so scale<1 reduces total compression and scale>1 increases it. PROTECT
    tier (ratio=0) stays at 0 regardless of scale. Returns name → ratio in [0, 1]."""
    if jsonl_path is None:
        jsonl_path = Path(__file__).resolve().parent / "sensitivity_results.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Asymmetric ratios require sensitivity_results.jsonl. "
            f"Run: python sensitivity.py --data <data> --ratio 0.30"
        )
    ratios = {}
    for line in jsonl_path.read_text().strip().split("\n"):
        rec = json.loads(line)
        if "pose_drop" not in rec:
            continue
        for thresh, ratio in SENSITIVITY_TIERS:
            if rec["pose_drop"] >= thresh:
                ratios[rec["name"]] = min(1.0, max(0.0, ratio * scale))
                break
    return ratios


# Criteria registry — must include any criterion exposed via --criterion.
# `large_final` reads only current weights; `movement` and `updating_movement`
# need `_init_weights` (and `_old_weights`) buffers registered before Pruner runs.
CRITERIA = {
    "magnitude": large_final,            # default — |w|
    "movement": movement,                # |w_now - w_init|, init = pretrained
    "updating_movement": updating_movement,  # |w_now - w_prev_step|
}


# ---------------------------------------------------------------------------
# C2f_v2: the one required architectural change
# ---------------------------------------------------------------------------
# YOLOv8's C2f block uses .chunk() to split a tensor in half. torch-pruning's
# dependency tracer can't follow .chunk(), which means it refuses to prune
# any C2f block. C2f_v2 does the same math with two explicit 1x1 convs — and
# is traceable.

class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def replace_c2f_in_place(module: nn.Module) -> None:
    """Replace every C2f child with C2f_v2, copying weights."""
    for name, child in module.named_children():
        if isinstance(child, C2f):
            shortcut = hasattr(child.m[0], "add") and child.m[0].add
            new = C2f_v2(
                child.cv1.conv.in_channels,
                child.cv2.conv.out_channels,
                n=len(child.m),
                shortcut=shortcut,
                g=child.m[0].cv2.conv.groups,
                e=child.c / child.cv2.conv.out_channels,
            )
            # cv1 in the original holds 2*c output channels; split into cv0/cv1 halves.
            w = child.cv1.conv.weight.data
            half = w.shape[0] // 2
            new.cv0.conv.weight.data.copy_(w[:half])
            new.cv1.conv.weight.data.copy_(w[half:])
            for key in ("weight", "bias", "running_mean", "running_var"):
                old = getattr(child.cv1.bn, key)
                getattr(new.cv0.bn, key).data.copy_(old[:half])
                getattr(new.cv1.bn, key).data.copy_(old[half:])
            # Sync BN hyperparameters (eps, momentum). Ultralytics trains with
            # eps=1e-3 but fresh nn.BatchNorm2d defaults to 1e-5 — without this,
            # BN normalization drifts ~100x and compounds through the network,
            # silently degrading accuracy by a couple of mAP points.
            new.cv0.bn.eps = child.cv1.bn.eps
            new.cv1.bn.eps = child.cv1.bn.eps
            new.cv0.bn.momentum = child.cv1.bn.momentum
            new.cv1.bn.momentum = child.cv1.bn.momentum
            new.cv2 = child.cv2
            new.m = child.m
            # Copy ultralytics graph-metadata attributes (f, i, type, n, np, ...).
            # The top-level model's forward pass uses m.f to route inputs between
            # layers; without this, the pruned model errors out at runtime.
            for attr_name in dir(child):
                if attr_name.startswith("_") or callable(getattr(child, attr_name)):
                    continue
                if not hasattr(new, attr_name):
                    setattr(new, attr_name, getattr(child, attr_name))
            setattr(module, name, new)
        else:
            replace_c2f_in_place(child)


# ---------------------------------------------------------------------------
# Protected layers (detected, not hardcoded)
# ---------------------------------------------------------------------------
# Two kinds of layer whose output channel count is fixed by the task:
#   - The stem Conv2d (3 RGB inputs — can't reduce input channels)
#   - The head's final output convs (box, class, keypoints — each a fixed dim)
# torch-pruning handles all intermediate dependencies automatically.

def find_protected_layers(model: nn.Module) -> list[nn.Module]:
    protected: list[nn.Module] = []

    # Stem: the first Conv2d that sees 3-channel input.
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            protected.append(m)
            break

    # Head: the last child of the backbone is the detection/pose head.
    head = model.model[-1]
    # cv2 (box regression) and cv3 (classification) are over-provisioned internally —
    # protect only their final Conv2d (fixed output dim). Their intermediates prune.
    for attr in ("cv2", "cv3"):
        if hasattr(head, attr):
            for seq in getattr(head, attr):
                last_conv = next((m for m in reversed(list(seq.modules()))
                                  if isinstance(m, nn.Conv2d)), None)
                if last_conv is not None:
                    protected.append(last_conv)
    # cv4 (keypoints, pose-only) is already minimal — 51ch = 17 keypoints x 3.
    # Pruning its intermediates collapses pose mAP even with fine-tuning.
    # Protect the entire branch.
    if hasattr(head, "cv4"):
        for seq in head.cv4:
            protected.append(seq)
    if hasattr(head, "dfl"):
        protected.append(head.dfl)

    # Empirically pose-fragile layers (sensitivity.py probe @ ratio=0.30):
    #   model.1.conv         (32f, early backbone) — pose drop 0.153
    #   model.2.cv2.conv     (32f, first C2f output proj) — pose drop 0.492
    # These are tiny in absolute params but every later layer reads their
    # features; pruning them even slightly cascades into mAP collapse.
    sensitive_paths = ["model.1.conv", "model.2.cv2.conv"]
    name_to_module = dict(model.named_modules())
    for path in sensitive_paths:
        if path in name_to_module:
            protected.append(name_to_module[path])

    return protected


# ---------------------------------------------------------------------------
# Ultralytics trainer bypass
# ---------------------------------------------------------------------------
# By default YOLO.train() rebuilds the model from its YAML config, which
# breaks after pruning (the shapes no longer match the YAML). We need one
# small wrapper that keeps the in-memory model intact.

def fine_tune(yolo: YOLO, **train_kwargs) -> nn.Module:
    """Run YOLO.train without letting it rebuild the model from YAML.

    Defaults aggressive augmentations OFF (mosaic, randaugment, erasing, scale).
    These are calibrated for from-scratch 300-epoch ImageNet-style training and
    corrupt BN running stats during short fine-tunes from a converged checkpoint.
    Override per-call if a different recovery profile is needed.
    """
    fine_tune_safe_defaults = {
        "mosaic": 0.0, "close_mosaic": 0,
        "scale": 0.0, "auto_augment": "", "erasing": 0.0,
        "warmup_epochs": 0.5, "lrf": 0.1,
        # Disable bias warmup. Ultralytics applies warmup_bias_lr=0.1 to ALL
        # bias parameters (including BN beta) during the first warmup_epochs,
        # independent of lr0. In iterative pruning this fires once per
        # `yolo.train()` call (5× per run) and produces ~40% BN-beta drift
        # per step (verified in diagnose3.py). Each restart's spike costs
        # ~0.005-0.010 pose mAP that subsequent epochs don't fully recover.
        # Setting it to 0 eliminates the spike entirely; one-epoch tests
        # showed no harm and the cumulative multi-step benefit is meaningful.
        "warmup_bias_lr": 0.0,
    }
    # User kwargs take precedence over the safe defaults.
    overrides = {**yolo.overrides, **fine_tune_safe_defaults, **train_kwargs, "mode": "train"}
    if "data" not in overrides:
        raise ValueError("fine_tune requires data='<dataset>.yaml'")
    trainer_cls = yolo.task_map[yolo.task]["trainer"]
    trainer = trainer_cls(overrides=overrides, _callbacks=yolo.callbacks)
    trainer.pruning = True
    trainer.model = yolo.model
    # Save full model (not state_dict) so pruned shapes survive serialization.
    trainer.save_model = lambda: torch.save(
        {"model": deepcopy(trainer.model), "train_args": vars(trainer.args)},
        trainer.last,
    )
    trainer.final_eval = lambda: None  # skip the "load best and re-validate" step
    trainer.train()
    yolo.model = trainer.model
    # Expose the saved checkpoint path so callers can find/report it.
    # `trainer.last` is a pathlib.Path to runs/<task>/train<N>/weights/last.pt.
    yolo._last_pt_path = str(trainer.last)
    return trainer.model


# ---------------------------------------------------------------------------
# End-of-training summary helper
# ---------------------------------------------------------------------------

def print_pruning_summary(
    yolo_pruned,
    baseline_model_path: str,
    last_pt_path: str | None = None,
) -> None:
    """Print a side-by-side weight-shape comparison between the baseline
    checkpoint and the pruned model, plus the saved-checkpoint path.

    Only compares layer paths that survive the C2→C2_v2 / C2f→C2f_v2
    surgery: plain `Conv` layers (stem, downsamplers) and the `cv2`
    output-projection inside each CSP block (the surgery reuses `cv2`
    unchanged). `cv1` / `cv0` paths diverge across the surgery and aren't
    meaningfully comparable.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 100)
    print(f"Pruning summary — baseline ({baseline_model_path}) vs pruned")
    print("=" * 100)

    yolo_base = YOLO(baseline_model_path)
    base_mods = dict(yolo_base.model.named_modules())
    prune_mods = dict(yolo_pruned.model.named_modules())

    candidates = [
        ("model.0.conv",       "stem (3-ch input — PROTECTED)"),
        ("model.1.conv",       "first backbone downsample"),
        ("model.3.conv",       "backbone downsample"),
        ("model.5.conv",       "backbone downsample"),
        ("model.7.conv",       "deeper backbone downsample"),
        ("model.9.conv",       "deepest backbone Conv (P6 only)"),
        ("model.2.cv2.conv",   "first CSP block — output projection"),
        ("model.4.cv2.conv",   "CSP block — output projection"),
        ("model.6.cv2.conv",   "CSP block — output projection"),
        ("model.8.cv2.conv",   "CSP block — output projection"),
    ]

    print(f"{'Layer':<24} {'Baseline shape':<22} {'Pruned shape':<22} "
          f"{'Δ':<22}  Description")
    print("-" * 100)
    for path, desc in candidates:
        b, p = base_mods.get(path), prune_mods.get(path)
        if b is None or p is None or not hasattr(b, "weight") or not hasattr(p, "weight"):
            continue
        bs, ps = tuple(b.weight.shape), tuple(p.weight.shape)
        if bs == ps:
            change = "unchanged"
        else:
            parts = []
            if bs[0] != ps[0]:
                parts.append(f"out −{(1-ps[0]/bs[0])*100:.1f}%")
            if bs[1] != ps[1]:
                parts.append(f"in −{(1-ps[1]/bs[1])*100:.1f}%")
            change = ", ".join(parts) if parts else "unchanged"
        print(f"{path:<24} {str(bs):<22} {str(ps):<22} {change:<22}  {desc}")

    # Overall totals
    base_params = sum(p.numel() for p in yolo_base.model.parameters())
    prune_params = sum(p.numel() for p in yolo_pruned.model.parameters())
    print("-" * 100)
    print(f"Total params: {base_params/1e6:.3f} M → {prune_params/1e6:.3f} M "
          f"({(1 - prune_params/base_params)*100:.2f}% reduction)")

    if last_pt_path is None:
        last_pt_path = getattr(yolo_pruned, "_last_pt_path", None)
    if last_pt_path:
        print(f"\nSaved checkpoint: {last_pt_path}")
        print(f"To load: `from prune_yolov8_pose import C2f_v2; "
              f"yolo = YOLO('{last_pt_path}')`")
    print("=" * 100 + "\n")
    del yolo_base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _validate_final(yolo: YOLO, data: str, imgsz: int, batch: int) -> dict:
    """Run ultralytics val() on the current model, return box/pose mAPs as a dict."""
    try:
        metrics = yolo.val(data=data, imgsz=imgsz, batch=batch, verbose=False, plots=False)
        return {
            "box_map50": float(metrics.box.map50),
            "box_map5095": float(metrics.box.map),
            "pose_map50": float(metrics.pose.map50),
            "pose_map5095": float(metrics.pose.map),
        }
    except Exception as e:
        print(f"Final validation failed ({type(e).__name__}): {e}")
        return {}


def _log_experiment(args: argparse.Namespace, baseline: dict, final: dict, wall_time_s: float) -> None:
    """Append a JSONL line to experiments.jsonl — one run per line."""
    import datetime
    from pathlib import Path
    import json as _json

    log_path = Path(__file__).resolve().parent / "experiments.jsonl"
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": f"{args.steps}step_{args.epochs}ep_r{args.ratio:.2f}",
        "command": f"python prune_yolov8_pose.py --data {args.data} --steps {args.steps} "
                   f"--epochs {args.epochs} --imgsz {args.imgsz} --batch {args.batch} "
                   f"--ratio {args.ratio}",
        "config": {
            "model": args.model, "data": args.data,
            "imgsz": args.imgsz, "batch": args.batch,
            "steps": args.steps, "epochs": args.epochs, "ratio": args.ratio,
        },
        "baseline": baseline,
        "final": final,
        "wall_time_s": round(wall_time_s, 1),
        "notes": "",
    }
    with open(log_path, "a") as f:
        f.write(_json.dumps(record) + "\n")
    print(f"\nLogged experiment to {log_path}")


def prune_continuous(args: argparse.Namespace) -> None:
    """Single-train-call iterative pruning, using ultralytics callbacks.

    Replaces the multi-train architecture (5 separate `yolo.train()` calls,
    each re-running warmup_bias_lr=0.1 → cumulative BN-beta drift) with a
    SINGLE train() call covering `steps × epochs` total epochs. Pruning is
    injected via an `on_train_epoch_start` callback at epoch boundaries.

    Benefits:
      - Only ONE optimizer warmup at the start, not 5
      - LR cosine decay continues uninterrupted across prune steps
      - BN running stats keep accumulating without trainer-restart spikes
      - Optimizer rebuild after prune-induced shape changes is the only
        per-step state perturbation

    Caveat: later steps prune at lower LR (cosine has decayed further),
    which slightly changes the recovery dynamics. The trade-off is fewer
    restart-tax penalties for slightly less aggressive late-step recovery.
    """
    import time as _time
    t0 = _time.time()

    yolo = YOLO(args.model)
    model = yolo.model.train()
    replace_c2f_in_place(model)

    device = next(model.parameters()).device
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Baseline: {base_macs/1e9:.2f} GMACs, {base_params/1e6:.2f}M params")

    protected = find_protected_layers(model)
    print(f"Protecting {len(protected)} layer(s) from pruning")

    criterion = CRITERIA[args.criterion]
    if args.criterion in ("movement", "updating_movement"):
        for m in model.modules():
            if hasattr(m, "weight") and m.weight is not None:
                m.register_buffer("_init_weights", m.weight.detach().clone())

    if args.asymmetric:
        ratio_arg = load_asymmetric_ratios(scale=args.asymmetric_scale)
        print(f"Asymmetric pruning: {len(ratio_arg)} per-layer ratios (scale={args.asymmetric_scale})")
    else:
        ratio_arg = args.ratio

    pruner = Pruner(
        model, ratio_arg, "local", criterion,
        ignored_layers=protected,
        iterative_steps=args.steps,
        schedule=Schedule(partial(sched_onecycle, α=10, β=4)),
    )

    total_epochs = args.steps * args.epochs
    prune_epochs = set(i * args.epochs for i in range(args.steps))
    print(f"Continuous training: {total_epochs} epochs total, "
          f"prune at epochs {sorted(prune_epochs)}")

    state = {"prune_count": 0}

    def prune_callback(trainer):
        epoch = trainer.epoch  # 0-indexed
        if epoch not in prune_epochs or state["prune_count"] >= args.steps:
            return
        print(f"\n[CB] Pruning at epoch {epoch} "
              f"(step {state['prune_count']+1}/{args.steps})")
        pruner.prune_model()
        state["prune_count"] += 1

        # Compute MAC/param stats BEFORE the device-and-dtype enforcement.
        # count_ops_and_params runs a tracing forward that can leave the
        # model in a state where some buffers are CPU-bound; we enforce
        # afterwards.
        macs, params = tp.utils.count_ops_and_params(
            trainer.model.to(device), example_inputs
        )

        # Re-snapshot init buffers for movement criteria.
        if args.criterion in ("movement", "updating_movement"):
            for m in trainer.model.modules():
                if hasattr(m, "weight") and m.weight is not None:
                    m._init_weights = m.weight.detach().clone()
                    if hasattr(m, "_old_weights"):
                        m._old_weights = m.weight.detach().clone()

        # AGGRESSIVE re-attach: walk every parameter and buffer, force them
        # to the training device + float32. torch-pruning may create new
        # Parameter objects via direct module attribute assignment that
        # bypass register_parameter, leaving them stranded on CPU when
        # model.to(device) is called.
        for n, p in trainer.model.named_parameters():
            p.data = p.data.to(device=device, dtype=torch.float32)
        for n, b in trainer.model.named_buffers():
            if b.dtype.is_floating_point:
                b.data = b.data.to(device=device, dtype=torch.float32)
            else:
                b.data = b.data.to(device=device)

        # Rebuild optimizer to match new parameter shapes. AFTER device fix
        # so the optimizer captures on-device parameter tensors.
        old_opt = trainer.optimizer
        old_groups = old_opt.param_groups
        new_optimizer = type(old_opt)(trainer.model.parameters(),
                                      lr=old_groups[0].get("lr", args.lr))
        for new_g, old_g in zip(new_optimizer.param_groups, old_groups):
            for k, v in old_g.items():
                if k != "params":
                    new_g[k] = v
        trainer.optimizer = new_optimizer
        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            trainer.scheduler.optimizer = new_optimizer

        print(f"[CB] After step {state['prune_count']}: "
              f"{macs/1e9:.2f} GMACs, {params/1e6:.2f}M params "
              f"({params/base_params*100:.1f}%)")

        # Final guarantee: call .cuda() at the very end. Some torch internal
        # caches only refresh on this top-level Module call. Ditto .train()
        # so dropout/BN behavior is restored to training mode.
        if device.type == "cuda":
            trainer.model.cuda()
        trainer.model.train()
        # Sanity-check: any param still on CPU is a bug we'd want surfaced.
        bad = [n for n, p in trainer.model.named_parameters() if p.device != device]
        if bad:
            print(f"[CB] WARNING — {len(bad)} params still on CPU after fix: {bad[:3]}...")

    yolo.add_callback("on_train_epoch_start", prune_callback)

    # AMP=False is required in continuous mode. AMP creates a shadow model
    # copy with cached fp16 weights; after structural pruning resizes the
    # original weights, the shadow's references go stale and produce a
    # CPU/CUDA + fp16/fp32 mismatch on the next forward pass. Disabling AMP
    # forces the trainer to use the live fp32 weights directly.
    lr_kwargs = {"lr0": args.lr, "optimizer": "AdamW"} if args.lr is not None else {}
    model = fine_tune(yolo, data=args.data, epochs=total_epochs,
                      imgsz=args.imgsz, batch=args.batch, verbose=False,
                      amp=False, **lr_kwargs)

    # Final state + log.
    macs, params = tp.utils.count_ops_and_params(model.to(device), example_inputs)
    metrics = _validate_final(yolo, args.data, args.imgsz, args.batch)
    final = {
        **metrics,
        "params_m": round(params / 1e6, 3),
        "macs_g": round(macs / 1e9, 3),
        "param_reduction_pct": round(100 * (1 - params / base_params), 2),
        "mac_reduction_pct": round(100 * (1 - macs / base_macs), 2),
    }
    baseline = {
        "params_m": round(base_params / 1e6, 3),
        "macs_g": round(base_macs / 1e9, 3),
    }
    _log_experiment(args, baseline, final, wall_time_s=_time.time() - t0)


def prune(args: argparse.Namespace) -> None:
    import time as _time
    t0 = _time.time()

    yolo = YOLO(args.model)
    model = yolo.model.train()
    replace_c2f_in_place(model)

    device = next(model.parameters()).device
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Baseline: {base_macs/1e9:.2f} GMACs, {base_params/1e6:.2f}M params")

    protected = find_protected_layers(model)
    print(f"Protecting {len(protected)} layer(s) from pruning")

    # Movement-based criteria need `_init_weights` registered on EVERY module
    # that has a weight, before Pruner is constructed. We can't skip protected
    # layers — torch-pruning's dependency-group importance scoring still calls
    # the criterion on them (only the ratio is zero), and the criterion
    # accesses `m._init_weights` directly without hasattr-guarding.
    criterion = CRITERIA[args.criterion]
    if args.criterion in ("movement", "updating_movement"):
        n_registered = 0
        for m in model.modules():
            if hasattr(m, "weight") and m.weight is not None:
                m.register_buffer("_init_weights", m.weight.detach().clone())
                n_registered += 1
        print(f"Registered _init_weights on {n_registered} weighted modules "
              f"(criterion={args.criterion})")

    # Asymmetric ratios: tiered per-layer dict from sensitivity probe.
    # Falls back to scalar args.ratio when --asymmetric is off.
    if args.asymmetric:
        asym_ratios = load_asymmetric_ratios(scale=args.asymmetric_scale)
        weighted_avg = (
            sum(r * 1 for r in asym_ratios.values()) / max(len(asym_ratios), 1)
        )
        print(f"Asymmetric pruning enabled: {len(asym_ratios)} per-layer ratios "
              f"(scale={args.asymmetric_scale}, unweighted mean = {weighted_avg:.3f})")
        ratio_arg = asym_ratios
    else:
        ratio_arg = args.ratio

    pruner = Pruner(
        model, ratio_arg, "local", criterion,
        ignored_layers=protected,
        iterative_steps=args.steps,
        schedule=Schedule(partial(sched_onecycle, α=10, β=4)),
    )

    for i in range(args.steps):
        pruner.prune_model()
        macs, params = tp.utils.count_ops_and_params(model.to(device), example_inputs)
        print(f"\nStep {i+1}/{args.steps}: "
              f"{macs/1e9:.2f} GMACs ({base_macs/macs:.2f}x), "
              f"{params/1e6:.2f}M params ({params/base_params*100:.1f}%)")

        # Re-snapshot init buffers after pruning. torch-pruning reshapes
        # `weight` but does not propagate the cuts to our `_init_weights`
        # buffer, so by step 2 the buffer would mismatch the new shape.
        # Re-snapshotting at this point converts `movement` into "change
        # since the previous prune step" — functionally equivalent to
        # `updating_movement` after the first step. The alternative (slicing
        # the original init by torch-pruning's pruned indexes) would be more
        # faithful to the classical movement criterion but requires deeper
        # integration with the dependency-group pruning history.
        if args.criterion in ("movement", "updating_movement"):
            for m in model.modules():
                if hasattr(m, "weight") and m.weight is not None:
                    m._init_weights = m.weight.detach().clone()
                    if hasattr(m, "_old_weights"):
                        m._old_weights = m.weight.detach().clone()

        # Re-enable gradients on all parameters before fine-tuning.
        # torch-pruning's protection + ultralytics' fused checkpoint leaves some
        # params with requires_grad=False; the trainer warns once per param if
        # we skip this. Pre-setting them here keeps the console quiet.
        for p in model.parameters():
            p.requires_grad = True

        yolo.model = model
        # Pass lr0 + force AdamW only when --lr explicit. Without this, ultralytics'
        # auto-optimizer picks lr=2e-3 which catastrophically retrains pretrained
        # weights — the actual cause of accuracy regression in pose runs.
        lr_kwargs = {"lr0": args.lr, "optimizer": "AdamW"} if args.lr is not None else {}
        model = fine_tune(yolo, data=args.data, epochs=args.epochs,
                          imgsz=args.imgsz, batch=args.batch, verbose=False, **lr_kwargs)

    # Export directly via torch.onnx — ultralytics' yolo.export() subscripts
    # self.model.args["imgsz"] which breaks after our custom trainer path
    # (args becomes an IterableSimpleNamespace rather than a dict).
    #
    # Notes on why we go CPU + dynamo=False:
    #   1. YOLOv8's Detect head caches anchor grids on first forward at a
    #      device; after CUDA fine-tuning some of those buffers lag on CPU
    #      and the device-mismatch trips the new torch.export path.
    #   2. PyTorch 2.9's default `dynamo=True` is strict about traced graphs
    #      and rejects any dynamic control flow the pose head uses.
    # Moving everything to CPU and using the legacy tracing exporter
    # (dynamo=False) sidesteps both.
    onnx_path = "pruned_model.onnx"
    model.cpu().eval()
    sample_cpu = torch.randn(1, 3, args.imgsz, args.imgsz)
    try:
        torch.onnx.export(
            model,
            sample_cpu,
            onnx_path,
            opset_version=13,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,  # use the legacy tracing exporter
        )
        print(f"\nExported to {onnx_path}")
    except Exception as e:
        # Fallback: save the pruned weights as a .pt checkpoint instead.
        # Customers can export via their own runtime later (TRT, OpenVINO, etc.).
        pt_path = "pruned_model.pt"
        torch.save({"model": model}, pt_path)
        print(f"\nONNX export failed ({type(e).__name__}): {str(e)[:120]}")
        print(f"Saved .pt checkpoint instead: {pt_path}")

    # Measure final metrics + log the experiment.
    # macs/params come from the last pruning step (variable `macs`, `params`).
    final = {
        "params_m": round(params / 1e6, 3),
        "macs_g": round(macs / 1e9, 3),
        "param_reduction_pct": round((1 - params / base_params) * 100, 2),
        "mac_reduction_pct": round((1 - macs / base_macs) * 100, 2),
    }
    print("\n=== Running final validation for experiment log ===")
    yolo.model = model
    final.update(_validate_final(yolo, args.data, args.imgsz, args.batch))

    baseline = {
        "params_m": round(base_params / 1e6, 3),
        "macs_g": round(base_macs / 1e9, 3),
    }
    _log_experiment(args, baseline, final, wall_time_s=_time.time() - t0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterative channel pruning for YOLOv8.")
    p.add_argument("--model", default="yolov8n-pose.pt")
    p.add_argument("--data", default="coco8-pose.yaml")
    p.add_argument("--steps", type=int, default=3, help="Prune+fine-tune cycles")
    p.add_argument("--epochs", type=int, default=3, help="Fine-tune epochs per cycle")
    p.add_argument("--ratio", type=float, default=0.15, help="Channel removal rate per step")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Initial LR (lr0). Default 1e-5 is calibrated for fine-tuning "
                        "converged pretrained checkpoints. ultralytics' auto picks "
                        "~2e-3 (good for from-scratch, catastrophic for fine-tune). "
                        "Pass None to use ultralytics' auto.")
    p.add_argument("--criterion", choices=list(CRITERIA.keys()), default="magnitude",
                   help="Pruning importance criterion. 'magnitude' (default) keeps "
                        "filters with largest |w|. 'movement' keeps filters that moved "
                        "most from pretrained values during fine-tuning. "
                        "'updating_movement' uses change since the previous step only.")
    p.add_argument("--asymmetric", action="store_true",
                   help="Use per-layer pruning ratios derived from "
                        "sensitivity_results.jsonl instead of the uniform --ratio. "
                        "Sensitive layers get ratio=0; redundant layers up to 0.22. "
                        "Global weighted ratio ≈ 0.125 at default scale. Requires "
                        "having run sensitivity.py first.")
    p.add_argument("--asymmetric-scale", type=float, default=1.0,
                   help="Multiply all asymmetric tier ratios by this factor. "
                        "0.7 → ~8.75%% global compression (gentler), "
                        "1.0 → ~12.5%% (default), 1.4 → ~17.5%% (more aggressive). "
                        "PROTECT tier stays at 0 regardless.")
    p.add_argument("--continuous", action="store_true",
                   help="EXPERIMENTAL — single train() call for steps×epochs "
                        "total, with pruning injected via on_train_epoch_start "
                        "callback. Currently fails on the post-prune forward "
                        "pass due to ultralytics' ModelEMA shadow not following "
                        "structural changes. Use default iterative mode (now "
                        "with warmup_bias_lr=0 to mitigate the restart-tax).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.continuous:
        prune_continuous(args)
    else:
        prune(args)
