# YOLO_Exp — Iterative Pruning of YOLOv8-Pose

Compression-aware fine-tuning of YOLOv8-pose models via structured channel
pruning, with the pruning loop riding on top of ultralytics' real training
machinery (AMP, EMA, gradient accumulation, 3-group optimizer, warmup) rather
than a hand-rolled training loop.

## Result

Best run on `coco-pose` with `yolov8n-pose.pt`:

| Metric | Baseline | Pruned (champion config) |
|--------|----------|--------------------------|
| Pose mAP50-95 | 0.5116 | **0.4853** (94.9% retained) |
| Pose mAP50 | 0.7995 | 0.7840 |
| Box mAP50-95 | 0.7002 | 0.6824 |
| Box mAP50 | 0.9118 | 0.9109 |
| Params | 3.30 M | 2.81 M (**−14.86%**) |
| MACs | 4.62 G | 3.88 G (**−16.07%**) |
| Wall time | — | ~2 h on RTX 5090 |

## Setup

```bash
pip install -r requirements.txt
```

Pin versions: `torch ≥ 2.9`, `ultralytics 8.3.162`, `torch-pruning 1.5.3`,
`fasterai 0.3.0`. Different versions of any of these may shift results slightly
(particularly `ultralytics`, whose default `LambdaLR + lrf=0.1` schedule is
load-bearing for the result above).

`coco-pose` dataset is downloaded automatically by ultralytics on first run
(~1.6 GB). `yolov8n-pose.pt` checkpoint also auto-downloads.

## Quick Start

### Champion config — single training loop, asymmetric pruning, gradual schedule

```bash
python prune_yolov8_pose_simple.py \
    --data coco-pose.yaml \
    --epochs 50 --imgsz 640 --batch 16 \
    --lr 5e-5 \
    --asymmetric --asymmetric-scale 0.7
```

Reproduces the 0.4853 / 14.86% result above. Auto-logs to
`experiments.jsonl` on completion.

### Trainer-tax baseline (no pruning, same training recipe)

```bash
python prune_yolov8_pose_simple.py \
    --data coco-pose.yaml \
    --epochs 50 --imgsz 640 --batch 16 \
    --lr 5e-5 \
    --asymmetric --asymmetric-scale 0.0
```

Measures how much pose mAP the fine-tuning recipe alone destroys on the
pretrained checkpoint, with zero structural change. Useful for decomposing
"compression damage" vs "trainer drift."

### P6 variants (yolov8x-pose-p6, etc.)

```bash
python prune_yolov8_pose_p6_simple.py \
    --model yolov8x-pose-p6.pt \
    --data coco-pose.yaml \
    --epochs 50 --imgsz 960 --batch 8 \
    --lr 5e-5 --ratio 0.12
```

P6-specific notes:
- **Default `--imgsz 960`** (vs 640 for non-P6). Smaller imgsz crashes the
  Detect head's Concat geometry.
- **Default `--batch 8`** (vs 16). The yolov8x-pose-p6 model has ~99 M params
  (~30× larger than yolov8n-pose) so batch must be reduced for memory.
- **Wall time ≈ 10 h on RTX 5090** for `--epochs 50`. Use `--epochs 5` first
  for a quick sanity check.
- The script handles both C2 (backbone) and C2f (neck) block replacements
  in the same pass via `replace_csp_blocks_in_place`. The graph-build runs
  in eval mode because P6's Detect head produces spatially-misaligned Concat
  inputs in train mode at non-standard imgsz.

## Scripts

| File | Role |
|------|------|
| `prune_yolov8_pose_simple.py` | **Recommended (n/s/m/l/x variants).** Single `yolo.train()` call + pruning callback. Inherits ultralytics' full training pipeline. |
| `prune_yolov8_pose_p6_simple.py` | **Recommended for P6 variants** (`yolov8x-pose-p6.pt` etc.). Same architecture as the simple script, but handles both C2 and C2f blocks and uses eval-mode graph build. |
| `prune_yolov8_pose.py` | Legacy iterative variant for non-P6 models (5 train calls × 10 epochs each). Kept for A/B comparison and as the source of helper functions imported by `_simple.py`. |
| `prune_yolov8_pose_p6.py` | Legacy iterative variant for P6 models. Source of P6-specific helpers (`replace_csp_blocks_in_place`, P6-aware `find_protected_layers`) imported by the P6 simple script. |
| `sensitivity.py` | Per-layer sensitivity probe: zeros bottom-N filters by L2 norm, runs val, ranks layers by pose mAP drop. Output: `sensitivity_results.jsonl`. |
| `sensitivity_results.jsonl` | Pre-computed sensitivity ranking at ratio=0.30 for `yolov8n-pose`. Required by `--asymmetric` mode in the n-pose pruning scripts. P6 doesn't ship with one — regenerate via `sensitivity.py --model yolov8x-pose-p6.pt --imgsz 960` if needed. |
| `experiments.jsonl` | Auto-appended log of all runs. One line per run with full config + final metrics. |

## How `--asymmetric` Works

The `--asymmetric` flag opts into per-layer pruning ratios derived from
`sensitivity_results.jsonl`. Layers are bucketed into 6 tiers based on their
sensitivity probe pose-drop, and each tier gets its own per-layer ratio:

| Tier (pose_drop) | Per-layer ratio | Layer count |
|------------------|-----------------|-------------|
| ≥ 0.10 (PROTECT) | 0.00 | 7 |
| 0.05–0.10 (MILD) | 0.03 | 5 |
| 0.02–0.05 (LIGHT) | 0.06 | 15 |
| 0.01–0.02 (MODERATE) | 0.10 | 15 |
| 0.005–0.01 (STRONG) | 0.15 | 12 |
| < 0.005 (AGGRESSIVE) | 0.22 | 19 |

`--asymmetric-scale 0.7` multiplies all tier ratios by 0.7 → ~12.5% global
weighted compression. Scale 1.0 → ~17.8%. Scale 0.0 → no pruning (used as
trainer-tax baseline).

## Recomputing Sensitivity for a Different Model

`sensitivity_results.jsonl` is specific to `yolov8n-pose.pt`. For other YOLO
checkpoints, regenerate via:

```bash
python sensitivity.py --model yolov8s-pose.pt --data coco-pose.yaml --ratio 0.30
```

~30 minutes on RTX 5090 (one validation pass per prunable Conv2d layer).

## Architecture

`prune_yolov8_pose_simple.py` is the cleanest implementation. The pattern:

```
ultralytics' yolo.train()  ← unchanged training loop
    └── on_train_epoch_start callback
        └── if schedule says we should prune more:
            ├── pruner.prune_model()
            ├── trainer.model.to(device); float32
            ├── trainer.ema = ModelEMA(trainer.model)   ← critical
            ├── rebuild trainer.optimizer with new params
            └── repoint trainer.scheduler at new optimizer
    └── on_fit_epoch_end callback
        └── print sparsity + mAP per epoch
```

Key insights:

- Ultralytics' `ModelEMA` shadow-copies the model and crashes on shape
  mismatch after structural pruning. Reconstructing it after each prune
  is the load-bearing fix that makes single-loop training viable.
- The `Pruner` is built once with `iterative_steps=N`. The schedule
  (default `sched_onecycle`) drives *when* `prune_model()` fires; the
  schedule's pre-computed ratios drive *what* each call removes.
- ultralytics' default `LambdaLR + lrf=0.1` (single linear decay over
  total epochs) is the right LR schedule for fine-tuning a converged
  pose checkpoint. Warm restarts and per-step cosine cycles were tested
  and regressed pose mAP by ~0.016.

## CLI Reference (`prune_yolov8_pose_simple.py`)

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `yolov8n-pose.pt` | Any ultralytics pose checkpoint |
| `--data` | `coco-pose.yaml` | Use `coco8-pose.yaml` for smoke tests |
| `--epochs` | 50 | Total training epochs |
| `--steps` | = `--epochs` | Number of prune events. Default = epochs (one per epoch, gradual). Set lower (e.g. 5) for chunked. |
| `--imgsz` | 640 | |
| `--batch` | 16 | |
| `--ratio` | 0.12 | Used only when `--asymmetric` is off |
| `--lr` | 5e-5 | Calibrated for fine-tune. Ultralytics' auto picks ~2e-3 which destroys pretrained weights. |
| `--criterion` | `magnitude` | Or `movement`, `updating_movement` |
| `--asymmetric` | off | Use per-layer ratios from `sensitivity_results.jsonl` |
| `--asymmetric-scale` | 1.0 | Multiplier on tier ratios. 0.7 = champion config; 0.0 = no pruning |
| `--prune-schedule` | `onecycle` | Or `agp`, `oneshot`, `iterative` |
| `--onecycle-alpha` / `--onecycle-beta` | 14 / 6 | Logistic shape; defaults preserve early/late epochs |

## Decomposition of Pose mAP Loss

Empirically validated on coco-pose:

| Source | Pose mAP cost |
|--------|---------------|
| Pretrained baseline | 0.5116 |
| 50 epochs of fine-tuning, **no pruning** | 0.5116 → ~0.4863 (**−0.025**) |
| 50 epochs + 14.86% pruning at scale=0.7 | 0.5116 → 0.4853 (**−0.026**) |

So at this compression level, **pure compression damage is ~0.001 pose mAP**;
the rest of the loss is "trainer tax" from any fine-tune of a converged
checkpoint with a recipe different from the original 500-epoch SGD+mosaic
training. The compression itself is essentially free.

## License

MIT.
