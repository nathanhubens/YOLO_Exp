"""Pruning YOLOv8-pose by hooking into ultralytics' actual training loop.

The previous attempt (prune_yolov8_pose_custom.py) reimplemented the
training loop in plain PyTorch and got significantly worse mAP because
it skipped ultralytics' battle-tested machinery: gradient accumulation
(nbs=64 → effective batch from batch=16), AMP+GradScaler, ModelEMA,
custom 3-group optimizer (decay/no-decay/biases), warmup ramps, etc.

This script flips the design: USE `yolo.train()` directly — full
ultralytics pipeline — and inject pruning at epoch boundaries via
the `on_train_epoch_start` callback. The only state we have to manage
ourselves is rebuilding ModelEMA after each prune (its shadow-copy
of the model has stale references after structural channel removal).

Usage:
  python prune_yolov8_pose_simple.py --data coco-pose.yaml \
      --epochs 50 --imgsz 640 --batch 16 --lr 5e-5 \
      --asymmetric --asymmetric-scale 0.7
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
from prune_yolov8_pose import (
    replace_c2f_in_place,
    find_protected_layers,
    fine_tune,
    CRITERIA,
    load_asymmetric_ratios,
    _validate_final,
    _log_experiment,
)
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
        "pruned_pct": 0.0,        # cumulative param reduction
        "macs_reduction_pct": 0.0,
    }
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    def cb(trainer):
        epoch = trainer.epoch  # 0-indexed
        # Schedule-driven trigger. Use (epoch+1)/total so pct_train hits
        # 1.0 at the last epoch — otherwise the final prune step never
        # fires (sched_onecycle's progress(0.98) is still < 1.0).
        pct_train = (epoch + 1) / max(state["total_epochs"], 1)
        target_steps = int(schedule.sched_func(0, 1, pct_train) * args.steps)
        target_steps = min(target_steps, args.steps)

        while state["prunes_made"] < target_steps:
            state["prunes_made"] += 1
            # Snapshot param count BEFORE the prune to detect no-op cases.
            # When asymmetric_scale=0 or args.ratio=0, pruner.prune_model()
            # is a no-op (no channels removed) but its caller would still
            # discard Adam's m/v buffers via the optimizer rebuild — which
            # is itself a trainer-tax source. Skip the rebuild when the
            # prune produced zero structural change.
            params_before = sum(p.numel() for p in trainer.model.parameters())
            print(f"\n[CB epoch {epoch} (pct_train={pct_train:.3f})] "
                  f"Pruning step {state['prunes_made']}/{args.steps}")
            pruner.prune_model()
            params_after = sum(p.numel() for p in trainer.model.parameters())
            if params_after == params_before:
                print(f"   → no-op (ratio=0 for all layers); skipping rebuild")
                continue

            # Move new params back to device + float32. torch-pruning's
            # parameter resize can leave new Parameter objects on CPU.
            trainer.model.to(device)
            for p in trainer.model.parameters():
                p.data = p.data.to(device=device, dtype=torch.float32)

            # Rebuild ModelEMA. This is the critical step that breaks
            # without explicit handling: EMA holds a deep copy of the
            # model's parameter tensors, and after structural pruning
            # those tensors mismatch the live model's shapes. ema.update()
            # then crashes with shape errors. Reconstructing EMA from
            # the live (just-pruned) model resyncs everything.
            if hasattr(trainer, "ema") and trainer.ema is not None:
                trainer.ema = ModelEMA(trainer.model)

            # Rebuild optimizer to track new parameter shapes. Preserve
            # current LR + group hyperparameters so warmup/decay still
            # apply correctly.
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
                # Scheduler holds a stale reference to the old optimizer.
                if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
                    trainer.scheduler.optimizer = new_optimizer

            # Re-snapshot init buffers for movement criteria. Mirrors
            # the iterative script's behavior — converts movement into
            # "change since previous prune step" which is what works
            # across multi-step pruning without shape mismatches.
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
        """Per-epoch summary after train+val complete. Pulls val mAP from
        trainer.metrics (populated by ultralytics after validation) and
        prints alongside the cumulative sparsity state. Runs every epoch
        regardless of whether pruning fired."""
        epoch = trainer.epoch + 1  # 1-indexed for display
        msg = (f"\n[Epoch {epoch}/{state['total_epochs']}] "
               f"sparsity={state['pruned_pct']:.1f}%  "
               f"MACs−{state['macs_reduction_pct']:.1f}%  "
               f"prunes={state['prunes_made']}/{args.steps}")
        # Pull the val mAP that ultralytics just computed.
        m = getattr(trainer, "metrics", None) or {}
        # ultralytics stores keys like 'metrics/mAP50-95(P)' for pose.
        for k in ("metrics/mAP50-95(P)", "metrics/mAP50(P)",
                  "metrics/mAP50-95(B)", "metrics/mAP50(B)"):
            if k in m:
                short = k.replace("metrics/", "").replace("(", "_").replace(")", "")
                msg += f"  {short}={m[k]:.4f}"
        print(msg)

    return cb, end_cb, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n-pose.pt")
    p.add_argument("--data", default="coco-pose.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps", type=int, default=None,
                   help="Number of prune events. Default = epochs (gradual).")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--ratio", type=float, default=0.12)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--criterion", choices=list(CRITERIA.keys()), default="magnitude")
    p.add_argument("--asymmetric", action="store_true")
    p.add_argument("--asymmetric-scale", type=float, default=1.0)
    p.add_argument("--prune-schedule", choices=list(PRUNE_SCHEDULES.keys()),
                   default="onecycle")
    p.add_argument("--onecycle-alpha", type=float, default=14,
                   help="α (steepness) for sched_onecycle. Default 14 = sharp "
                        "logistic that saturates at pct_train≈0.65. Try 8 for a "
                        "gentler curve that spreads prunes through epochs 30-45.")
    p.add_argument("--onecycle-beta", type=float, default=6,
                   help="β (offset) for sched_onecycle. Default 6 places the "
                        "inflection at pct_train≈0.43. Smaller pushes prunes "
                        "earlier; larger pushes them later.")
    args = p.parse_args()
    if args.steps is None:
        args.steps = args.epochs

    t0 = time.time()

    yolo = YOLO(args.model)
    model = yolo.model.train()
    replace_c2f_in_place(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if args.asymmetric:
        ratio_arg = load_asymmetric_ratios(scale=args.asymmetric_scale)
        print(f"Asymmetric: {len(ratio_arg)} per-layer ratios "
              f"(scale={args.asymmetric_scale})")
    else:
        ratio_arg = args.ratio

    sched_func = PRUNE_SCHEDULES[args.prune_schedule]
    # Allow softening sched_onecycle's logistic via CLI. Default (α=14, β=6)
    # saturates by pct_train≈0.65 — most pruning done in the middle third
    # of training. Smaller α + smaller β stretches the curve so prunes
    # happen later: e.g. α=8, β=4 saturates by pct_train≈0.85.
    if args.prune_schedule == "onecycle" and (args.onecycle_alpha != 14 or
                                              args.onecycle_beta != 6):
        sched_func = partial(sched_onecycle,
                             α=args.onecycle_alpha,
                             β=args.onecycle_beta)
        print(f"Using sched_onecycle with α={args.onecycle_alpha}, "
              f"β={args.onecycle_beta} (non-default)")
    schedule = Schedule(sched_func)
    pruner = Pruner(
        model, ratio_arg, "local", criterion_fn,
        ignored_layers=protected,
        iterative_steps=args.steps,
        schedule=schedule,
    )

    # Register callbacks:
    #   on_train_epoch_start → pruning trigger (schedule-driven)
    #   on_fit_epoch_end     → per-epoch sparsity + mAP summary
    prune_cb, summary_cb, _state = make_prune_callback(
        pruner, schedule, args, base_macs, base_params, device
    )
    yolo.add_callback("on_train_epoch_start", prune_cb)
    yolo.add_callback("on_fit_epoch_end", summary_cb)

    print(f"\nLaunching ultralytics yolo.train() for {args.epochs} epochs, "
          f"{args.steps} prune events driven by '{args.prune_schedule}'.")
    print("Pruning hooks via on_train_epoch_start; otherwise stock ultralytics "
          "training (AMP+GradScaler+EMA+accumulate+warmup all preserved).")

    # Single yolo.train() call — full ultralytics machinery.
    # `fine_tune` wraps it with our safe-defaults (warmup_bias_lr=0, aug off).
    lr_kwargs = {"lr0": args.lr, "optimizer": "AdamW"} if args.lr is not None else {}
    fine_tune(yolo, data=args.data, epochs=args.epochs,
              imgsz=args.imgsz, batch=args.batch, verbose=True, **lr_kwargs)

    # Final val + log.
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
