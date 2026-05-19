"""Universal pruning script for YOLOv8 pose/detection, including P6 variants.

Same structure as prune_yolov8_pose.py, but handles BOTH C2f AND C2 CSP blocks.
YOLOv8 n/s/m/l/x use C2f throughout; the P6 variants (e.g. yolov8x-pose-p6)
mix C2f in the neck with C2 in the backbone. Both blocks use .chunk() which
torch-pruning can't trace, so we swap each with a traceable _v2 variant that
uses two explicit 1x1 convs.

Usage:
    # Full run on the P6 variant (long — ~10h at imgsz=960 on a 5090):
    python prune_yolov8_pose_p6.py --data coco-pose.yaml \\
        --model yolov8x-pose-p6.pt --imgsz 960 --batch 8 \\
        --steps 5 --epochs 5 --ratio 0.12

    # Smoke test: verify replacement + one prune step, no training:
    python prune_yolov8_pose_p6.py --test-only --model yolov8x-pose-p6.pt
"""

import argparse
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, C2, C2f, Conv

from fasterai.prune.all import Pruner, Schedule, large_final, sched_onecycle


# ---------------------------------------------------------------------------
# Pruning-friendly replacements for C2f and C2
# ---------------------------------------------------------------------------

class C2f_v2(nn.Module):
    """Pruning-friendly C2f (replaces .chunk() with two explicit 1x1 convs)."""
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


class C2_v2(nn.Module):
    """Pruning-friendly C2 (simpler than C2f — no accumulated chunk history).

    C2 splits cv1 output into (a, b), runs a through a Sequential bottleneck
    stack, passes b as a pure skip, concatenates both, and projects with cv2.
    We just replace the split with two explicit 1x1 convs.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)  # the "a" branch (fed through m)
        self.cv1 = Conv(c1, self.c, 1, 1)  # the "b" branch (pure skip)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(*(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        ))

    def forward(self, x):
        return self.cv2(torch.cat((self.m(self.cv0(x)), self.cv1(x)), 1))


def _copy_meta_attrs(src: nn.Module, dst: nn.Module) -> None:
    """Copy ultralytics graph-metadata attrs (f, i, type, n, np) from src to dst."""
    for attr_name in dir(src):
        if attr_name.startswith("_") or callable(getattr(src, attr_name)):
            continue
        if not hasattr(dst, attr_name):
            setattr(dst, attr_name, getattr(src, attr_name))


def _split_cv1_weights(old_cv1, new_cv0, new_cv1) -> None:
    """Transfer weights from old_cv1 (2*c out) into new cv0 + cv1 (c each).

    Critically also copies BN HYPERPARAMETERS (eps, momentum): ultralytics
    trains with eps=0.001 but fresh nn.BatchNorm2d defaults to 1e-5. Skipping
    this causes BN normalization to diverge ~100x and compounds through the
    network — a bug that only manifests in production models, not in toy
    fresh-init tests. Buffers (running_*) use copy_() for version-stable
    element-wise assignment.
    """
    w = old_cv1.conv.weight.data
    half = w.shape[0] // 2
    new_cv0.conv.weight.data.copy_(w[:half])
    new_cv1.conv.weight.data.copy_(w[half:])
    for key in ("weight", "bias", "running_mean", "running_var"):
        old_val = getattr(old_cv1.bn, key)
        getattr(new_cv0.bn, key).data.copy_(old_val[:half])
        getattr(new_cv1.bn, key).data.copy_(old_val[half:])
    if hasattr(old_cv1.bn, "num_batches_tracked"):
        new_cv0.bn.num_batches_tracked.data.copy_(old_cv1.bn.num_batches_tracked.data)
        new_cv1.bn.num_batches_tracked.data.copy_(old_cv1.bn.num_batches_tracked.data)
    # Sync BN hyperparameters (eps, momentum). Without this, production
    # ultralytics weights trained with eps=1e-3 get normalized using eps=1e-5
    # from fresh-init, corrupting all subsequent layers.
    new_cv0.bn.eps = old_cv1.bn.eps
    new_cv1.bn.eps = old_cv1.bn.eps
    new_cv0.bn.momentum = old_cv1.bn.momentum
    new_cv1.bn.momentum = old_cv1.bn.momentum


def replace_csp_blocks_in_place(module: nn.Module) -> dict:
    """Replace every C2 / C2f child with the corresponding _v2 variant.

    Returns {"C2": n, "C2f": n} count per block type — useful for diagnostics.
    """
    counts = {"C2": 0, "C2f": 0}

    def _walk(mod: nn.Module) -> None:
        for name, child in mod.named_children():
            # Check C2f FIRST — it's a subclass dance in some ultralytics versions
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
                _split_cv1_weights(child.cv1, new.cv0, new.cv1)
                new.cv2 = child.cv2
                new.m = child.m
                _copy_meta_attrs(child, new)
                setattr(mod, name, new)
                counts["C2f"] += 1
            elif isinstance(child, C2):
                shortcut = hasattr(child.m[0], "add") and child.m[0].add
                new = C2_v2(
                    child.cv1.conv.in_channels,
                    child.cv2.conv.out_channels,
                    n=len(child.m),
                    shortcut=shortcut,
                    g=child.m[0].cv2.conv.groups,
                    e=child.c / child.cv2.conv.out_channels,
                )
                _split_cv1_weights(child.cv1, new.cv0, new.cv1)
                new.cv2 = child.cv2
                new.m = child.m
                _copy_meta_attrs(child, new)
                setattr(mod, name, new)
                counts["C2"] += 1
            else:
                _walk(child)

    _walk(module)
    return counts


def replace_c2f_in_place(module: nn.Module) -> None:
    """Backward-compat alias — calls the universal replacer."""
    replace_csp_blocks_in_place(module)


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

    # Second-to-last Conv2d of the LARGEST-SCALE class head. The cv3
    # protection above only shields the FINAL Conv of each branch; the
    # intermediate Conv of the largest-scale branch (where large objects
    # — and therefore most pose-relevant detections — fire) is also
    # load-bearing per the n-pose sensitivity probe (pose drop 0.148).
    # The P6 model hasn't been probed yet, but the same logical pathway
    # exists, so we apply the same protection by symmetry. Variant-
    # agnostic: cv3[-1] works for both 3-scale (non-P6) and 4-scale (P6).
    if hasattr(head, "cv3") and len(head.cv3) >= 1:
        cv3_largest = head.cv3[-1]
        convs = [m for m in cv3_largest.modules() if isinstance(m, nn.Conv2d)]
        if len(convs) >= 2:
            protected.append(convs[-2])

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
    These are calibrated for from-scratch 300-epoch training and corrupt BN
    running stats during short fine-tunes from a converged checkpoint.
    """
    fine_tune_safe_defaults = {
        "mosaic": 0.0, "close_mosaic": 0,
        "scale": 0.0, "auto_augment": "", "erasing": 0.0,
        "warmup_epochs": 0.5, "lrf": 0.1,
    }
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
    yolo._last_pt_path = str(trainer.last)
    return trainer.model


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


def prune(args: argparse.Namespace) -> None:
    import time as _time
    t0 = _time.time()

    yolo = YOLO(args.model)
    # NOTE: use eval() during replacement + pruner graph-build. P6 variants'
    # Detect/Pose head produces spatially-misaligned Concat inputs in train mode
    # at non-standard imgsz, which crashes torch-pruning's tracer. fine_tune()
    # switches the model to train mode internally, so this doesn't hurt training.
    model = yolo.model.eval()
    replace_c2f_in_place(model)

    device = next(model.parameters()).device
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Baseline: {base_macs/1e9:.2f} GMACs, {base_params/1e6:.2f}M params")

    protected = find_protected_layers(model)
    print(f"Protecting {len(protected)} layer(s) from pruning")

    pruner = Pruner(
        model, args.ratio, "local", large_final,
        ignored_layers=protected,
        iterative_steps=args.steps,
        schedule=Schedule(partial(sched_onecycle, α=10, β=4)),
        example_inputs=example_inputs,   # P6-compatible: use our imgsz, not fasterai's 224 default
    )

    for i in range(args.steps):
        pruner.prune_model()
        macs, params = tp.utils.count_ops_and_params(model.to(device), example_inputs)
        print(f"\nStep {i+1}/{args.steps}: "
              f"{macs/1e9:.2f} GMACs ({base_macs/macs:.2f}x), "
              f"{params/1e6:.2f}M params ({params/base_params*100:.1f}%)")

        # Re-enable gradients on all parameters before fine-tuning.
        # torch-pruning's protection + ultralytics' fused checkpoint leaves some
        # params with requires_grad=False; the trainer warns once per param if
        # we skip this. Pre-setting them here keeps the console quiet.
        for p in model.parameters():
            p.requires_grad = True

        yolo.model = model
        # Pass lr0 + force AdamW only when --lr explicit. ultralytics' auto-optimizer
        # picks lr=2e-3 which over-trains pretrained weights — use 1e-4 or lower
        # for fine-tuning a converged checkpoint.
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


def test_only(args: argparse.Namespace) -> None:
    """Smoke-test: replace CSP blocks, compare forward outputs, run 1 prune step."""
    from collections import Counter
    print(f"=== Loading {args.model} ===")
    yolo = YOLO(args.model)
    model = yolo.model.eval()

    types_before = Counter(type(m).__name__ for m in model.model)
    params_before = sum(p.numel() for p in model.parameters())
    print(f"Block types before: {dict(types_before)}")
    print(f"Params before:      {params_before:,}")

    # Forward-pass output before replacement (on a small input for speed)
    probe = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        out_before = model(probe)

    print("\n=== Replacing C2 + C2f blocks with _v2 variants ===")
    counts = replace_csp_blocks_in_place(model)
    types_after = Counter(type(m).__name__ for m in model.model)
    params_after = sum(p.numel() for p in model.parameters())
    print(f"Replaced: {counts}")
    print(f"Block types after: {dict(types_after)}")
    print(f"Params after:      {params_after:,} (expected same as before)")
    assert params_before == params_after, "replacement changed param count — weight transfer is wrong"

    # Forward-pass output after replacement — should match pre-replacement
    model.eval()
    with torch.no_grad():
        out_after = model(probe)

    # Both outputs are tuples of tensors; compare element-wise
    def _flatten(x):
        if isinstance(x, torch.Tensor): return [x]
        out = []
        for e in x: out.extend(_flatten(e))
        return out
    before_flat = _flatten(out_before)
    after_flat = _flatten(out_after)
    assert len(before_flat) == len(after_flat), \
        f"output shape structure differs: {len(before_flat)} vs {len(after_flat)}"
    max_diff = max((a - b).abs().max().item() for a, b in zip(before_flat, after_flat))
    print(f"\nMax |output diff| before-vs-after replacement: {max_diff:.2e}")
    if max_diff > 1e-3:
        print("  WARNING: replacement changed output math — weight transfer likely wrong")
    else:
        print("  OK: replacement preserves forward pass (within fp32 precision)")

    # Now try one prune step to verify torch-pruning can trace the new blocks.
    # IMPORTANT: trace in eval mode — ultralytics' Pose head routes differently in
    # train mode, and for P6 variants some Concat layers hit spatial misalignment.
    # eval() gives a stable forward graph for torch-pruning to analyze.
    print("\n=== Running 1 prune step (ratio=0.10, steps=1) ===")
    model.eval()
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Baseline: {base_macs/1e9:.2f} GMACs, {base_params/1e6:.2f}M params")

    for p in model.parameters():
        p.requires_grad = True
    protected = find_protected_layers(model)
    print(f"Protecting {len(protected)} layer(s)")

    # IMPORTANT for P6: fasterai.Pruner defaults example_inputs to a 224x224
    # tensor, which is too small for P6's /64 feature pyramid. Pass our own.
    pruner = Pruner(
        model, 0.10, "local", large_final,
        ignored_layers=protected,
        iterative_steps=1,
        schedule=Schedule(partial(sched_onecycle, α=10, β=4)),
        example_inputs=example_inputs,
    )
    pruner.prune_model()
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"After prune: {pruned_macs/1e9:.2f} GMACs ({base_macs/pruned_macs:.2f}x), "
          f"{pruned_params/1e6:.2f}M params ({pruned_params/base_params*100:.1f}%)")

    # Verify pruned model still produces a finite forward pass
    model.eval()
    with torch.no_grad():
        out_pruned = model(probe)
    finite = all(torch.isfinite(t).all().item() for t in _flatten(out_pruned))
    print(f"Pruned-model forward finite: {finite}")
    print("\n=== test-only: PASS ===" if finite else "\n=== test-only: FAIL (NaN/Inf) ===")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterative channel pruning for YOLOv8 incl. P6 variants.")
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
    p.add_argument("--test-only", action="store_true",
                   help="Smoke-test: replace blocks, verify forward equivalence, run 1 prune step, exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test_only:
        test_only(args)
    else:
        prune(args)
