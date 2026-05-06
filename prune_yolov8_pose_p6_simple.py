"""Pruning YOLOv8-pose-p6 by hooking into ultralytics' real training loop.

This is the P6 sibling of `prune_yolov8_pose_simple.py`. Same callback-based
architecture (single `yolo.train()` call + `on_train_epoch_start` pruning
callback + ModelEMA rebuild after each prune), but adapted for P6 variants:

  - **Two CSP block types**: P6 mixes C2 (in the backbone) with C2f (in the
    neck), where the n/s/m/l/x non-P6 variants are pure C2f. Both block
    types use `.chunk()` which torch-pruning can't trace, so we replace
    both with `_v2` variants in-place. `replace_csp_blocks_in_place` from
    `prune_yolov8_pose_p6` handles both.

  - **Eval-mode graph build**: P6's Detect head produces spatially-misaligned
    Concat inputs in train mode at non-standard imgsz (e.g., 960), which
    crashes torch-pruning's tracer. We construct the Pruner with the model
    in eval mode; `fine_tune()` flips it back to train mode internally.

  - **Larger default imgsz=960**: P6 is calibrated for high-resolution input
    and needs imgsz=960 (or higher) at inference. Smaller imgsz produces
    geometric mismatches that crash the head.

Usage (full run, ~10h on RTX 5090):
    python prune_yolov8_pose_p6_simple.py \
        --model yolov8x-pose-p6.pt \
        --data coco-pose.yaml \
        --epochs 50 --imgsz 960 --batch 8 \
        --lr 5e-5 --ratio 0.12

Usage (smoke test on coco8, no real training):
    python prune_yolov8_pose_p6_simple.py \
        --model yolov8x-pose-p6.pt \
        --data coco8-pose.yaml \
        --epochs 2 --imgsz 960 --batch 2 --ratio 0.05
"""

import argparse
import sys
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.utils.torch_utils import ModelEMA

sys.path.insert(0, ".")
# P6-specific helpers: dual-block (C2 + C2f) surgery, head-protection logic,
# and the shared fine_tune() wrapper that suppresses augmentation defaults.
from prune_yolov8_pose_p6 import (
    replace_csp_blocks_in_place,
    find_protected_layers,
    fine_tune,
    _validate_final,
    _log_experiment,
)
# Criteria registry comes from the n-pose script (same set works for P6).
from prune_yolov8_pose import CRITERIA
from fasterai.prune.all import (
    Pruner, Schedule,
    sched_onecycle, sched_agp, sched_oneshot, sched_iterative,
)


PRUNE_SCHEDULES = {
    "onecycle":  sched_onecycle,
    "agp":       sched_agp,
    "oneshot":   sched_oneshot,
    "iterative": sched_iterative,
}


def make_prune_callback(pruner: Pruner, schedule: Schedule, args,
                         base_macs: float, base_params: float, device):
    """Build the on_train_epoch_start callback that drives pruning,
    plus an on_fit_epoch_end callback that prints sparsity+mAP every
    epoch. Returns (start_cb, end_cb, state_dict)."""
    state = {
        "prunes_made": 0,
        "total_epochs": args.epochs,
        "pruned_pct": 0.0,
        "macs_reduction_pct": 0.0,
    }
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    def cb(trainer):
        epoch = trainer.epoch
        # (epoch+1)/total so pct_train hits 1.0 at the last epoch — otherwise
        # the final prune step never fires (sched_onecycle's progress(0.98)
        # is still < 1.0).
        pct_train = (epoch + 1) / max(state["total_epochs"], 1)
        target_steps = int(schedule.sched_func(0, 1, pct_train) * args.steps)
        target_steps = min(target_steps, args.steps)

        while state["prunes_made"] < target_steps:
            state["prunes_made"] += 1
            params_before = sum(p.numel() for p in trainer.model.parameters())
            print(f"\n[CB epoch {epoch} (pct_train={pct_train:.3f})] "
                  f"Pruning step {state['prunes_made']}/{args.steps}")
            pruner.prune_model()
            params_after = sum(p.numel() for p in trainer.model.parameters())
            if params_after == params_before:
                print(f"   → no-op (ratio=0 for all layers); skipping rebuild")
                continue

            # Move newly-resized parameters back to device + float32. P6's
            # bigger model (yolov8x: 99M params) makes this more important
            # because more torch-pruning re-allocations can leave more
            # tensors stranded on CPU.
            trainer.model.to(device)
            for p in trainer.model.parameters():
                p.data = p.data.to(device=device, dtype=torch.float32)

            # Rebuild ModelEMA — it shadow-copies the model and crashes on
            # shape mismatch after structural pruning.
            if hasattr(trainer, "ema") and trainer.ema is not None:
                trainer.ema = ModelEMA(trainer.model)

            # Rebuild optimizer with new parameter shapes; preserve LR.
            if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                old_opt = trainer.optimizer
                old_groups = old_opt.param_groups
                new_optimizer = type(old_opt)(
                    trainer.model.parameters(),
                    lr=old_groups[0].get("lr", args.lr),
                )
                for new_g, old_g in zip(new_optimizer.param_groups, old_groups):
                    for k, v in old_g.items():
                        if k != "params":
                            new_g[k] = v
                trainer.optimizer = new_optimizer
                if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
                    trainer.scheduler.optimizer = new_optimizer

            # Re-snapshot init buffers for movement criteria.
            if args.criterion in ("movement", "updating_movement"):
                for m in trainer.model.modules():
                    if hasattr(m, "weight") and m.weight is not None:
                        m._init_weights = m.weight.detach().clone()
                        if hasattr(m, "_old_weights"):
                            m._old_weights = m.weight.detach().clone()

            macs, params = tp.utils.count_ops_and_params(
                trainer.model, example_inputs
            )
            state["pruned_pct"] = 100.0 * (1.0 - params / base_params)
            state["macs_reduction_pct"] = 100.0 * (1.0 - macs / base_macs)
            print(f"   → {macs/1e9:.2f} GMACs, {params/1e6:.2f}M params "
                  f"(sparsity {state['pruned_pct']:.1f}%, "
                  f"MACs −{state['macs_reduction_pct']:.1f}%)")

    def end_cb(trainer):
        epoch = trainer.epoch + 1
        msg = (f"\n[Epoch {epoch}/{state['total_epochs']}] "
               f"sparsity={state['pruned_pct']:.1f}%  "
               f"MACs−{state['macs_reduction_pct']:.1f}%  "
               f"prunes={state['prunes_made']}/{args.steps}")
        m = getattr(trainer, "metrics", None) or {}
        for k in ("metrics/mAP50-95(P)", "metrics/mAP50(P)",
                  "metrics/mAP50-95(B)", "metrics/mAP50(B)"):
            if k in m:
                short = k.replace("metrics/", "").replace("(", "_").replace(")", "")
                msg += f"  {short}={m[k]:.4f}"
        print(msg)

    return cb, end_cb, state


def main():
    p = argparse.ArgumentParser(description="Single-loop P6 pruning for YOLOv8.")
    p.add_argument("--model", default="yolov8x-pose-p6.pt",
                   help="P6 variant (e.g., yolov8x-pose-p6.pt). Non-P6 models "
                        "should use prune_yolov8_pose_simple.py instead.")
    p.add_argument("--data", default="coco-pose.yaml")
    p.add_argument("--epochs", type=int, default=50,
                   help="Total training epochs (single yolo.train() call).")
    p.add_argument("--steps", type=int, default=None,
                   help="Number of prune events. Default = epochs (gradual).")
    p.add_argument("--imgsz", type=int, default=960,
                   help="P6 default 960. Smaller can crash the Detect head's "
                        "Concat geometry.")
    p.add_argument("--batch", type=int, default=8,
                   help="P6 has ~30× more params than n-pose, so reduce batch.")
    p.add_argument("--ratio", type=float, default=0.12)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--criterion", choices=list(CRITERIA.keys()), default="magnitude")
    p.add_argument("--prune-schedule", choices=list(PRUNE_SCHEDULES.keys()),
                   default="onecycle")
    p.add_argument("--onecycle-alpha", type=float, default=14)
    p.add_argument("--onecycle-beta", type=float, default=6)
    args = p.parse_args()
    if args.steps is None:
        args.steps = args.epochs

    t0 = time.time()

    yolo = YOLO(args.model)
    # P6 graph-build needs eval mode: train mode at non-standard imgsz
    # produces a Concat shape mismatch in the Detect head that crashes
    # torch-pruning's tracer. fine_tune() switches back to train mode.
    model = yolo.model.eval()
    block_counts = replace_csp_blocks_in_place(model)
    print(f"Replaced CSP blocks: {block_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # P6 needs explicit example_inputs at the actual imgsz — fasterai's
    # default 224×224 fails on the Detect head's spatial slicing.
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    base_macs, base_params = tp.utils.count_ops_and_params(
        model.to(device), example_inputs
    )
    print(f"Baseline: {base_macs/1e9:.2f} GMACs, {base_params/1e6:.2f}M params")

    protected = find_protected_layers(model)
    print(f"Protecting {len(protected)} layer(s)")

    criterion_fn = CRITERIA[args.criterion]
    if args.criterion in ("movement", "updating_movement"):
        for m in model.modules():
            if hasattr(m, "weight") and m.weight is not None:
                m.register_buffer("_init_weights", m.weight.detach().clone())

    sched_func = PRUNE_SCHEDULES[args.prune_schedule]
    if args.prune_schedule == "onecycle" and (args.onecycle_alpha != 14 or
                                              args.onecycle_beta != 6):
        sched_func = partial(sched_onecycle,
                             α=args.onecycle_alpha,
                             β=args.onecycle_beta)
        print(f"Using sched_onecycle with α={args.onecycle_alpha}, "
              f"β={args.onecycle_beta} (non-default)")
    schedule = Schedule(sched_func)
    pruner = Pruner(
        model, args.ratio, "local", criterion_fn,
        ignored_layers=protected,
        iterative_steps=args.steps,
        schedule=schedule,
        # Pass example_inputs explicitly; P6's default 224×224 graph-build
        # fails on Concat shape misalignment.
        example_inputs=example_inputs,
    )

    prune_cb, summary_cb, _state = make_prune_callback(
        pruner, schedule, args, base_macs, base_params, device
    )
    yolo.add_callback("on_train_epoch_start", prune_cb)
    yolo.add_callback("on_fit_epoch_end", summary_cb)

    print(f"\nLaunching ultralytics yolo.train() for {args.epochs} epochs, "
          f"{args.steps} prune events driven by '{args.prune_schedule}'.")

    lr_kwargs = {"lr0": args.lr, "optimizer": "AdamW"} if args.lr is not None else {}
    fine_tune(yolo, data=args.data, epochs=args.epochs,
              imgsz=args.imgsz, batch=args.batch, verbose=True, **lr_kwargs)

    metrics = _validate_final(yolo, args.data, args.imgsz, args.batch)
    macs, params = tp.utils.count_ops_and_params(
        yolo.model.to(device), example_inputs
    )
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
    _log_experiment(args, baseline, final, wall_time_s=time.time() - t0)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
