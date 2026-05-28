# Run Provenance

Ground-truth mapping of `runs/pose/train*` dirs to the **actual** command/config
that produced them. Needed because `_log_experiment()` currently writes a
template command string (`prune_yolov8_pose.py --ratio 0.12`) into
`experiments.jsonl` instead of the real `sys.argv`, and the logged `config`
dict omits `--asymmetric`, `--protect-sppf`, `--enable-extreme-tier`, and the
onecycle α/β. Until that's fixed, this file is the source of truth.

| run dir | date | actual flags | schedule | final pose50-95 | param ↓ |
|---------|------|--------------|----------|-----------------|---------|
| train28 | 2026-05-26 14:58 | `--asymmetric --asymmetric-scale 1.0 --protect-sppf --enable-extreme-tier` | onecycle **default** (α=14, β=6) | 0.4797 | 21.46% |
| train29 | 2026-05-26 15:01 | `--asymmetric --asymmetric-scale 1.0 --protect-sppf --enable-extreme-tier --onecycle-alpha 8 --onecycle-beta 4` | onecycle **softened** (α=8, β=4) | 0.4796 | 21.46% |

## A/B intent

train28 vs train29 isolate **schedule shape only** — identical per-layer ratios,
protected layers, and extreme tier, so the final architecture (~2.588M params,
~21.5% reduction) is the same. The single variable is *when* pruning fires:

- **train28 (default α=14/β=6)**: bursts up to 4 prune steps/epoch around epochs
  19-21; mAP dipped to 0.4550 @ epoch 22, recovered to 0.4797.
- **train29 (softened α=8/β=4)**: max 2 prune steps/epoch, spread evenly.
  Hypothesis: shallower dip → less to recover → final pose ≥ 0.480.

### A/B verdict (2026-05-26)

Hypothesis **half-confirmed**: dip got shallower (0.4550 → 0.4636) and moved
later (epoch 22 → 34), but **final accuracy was identical** (0.4797 vs 0.4796).
Conclusion: **schedule shape is a training-stability knob, not an accuracy knob**
— given enough post-prune fine-tuning epochs, the model converges to the same
place regardless of pruning pace. Negative result worth reporting in the paper.
Next lever to beat 0.4797 @ 21.5%: **analyzer-guided ratio reallocation** (ease
the 16 AT_LIMIT layers, push the 12 HAS_SLACK layers to 0.30) at fixed global
compression.

## `--reallocate` mode (added 2026-05-26)

Post-training-sensitivity reallocation. Buckets per-layer ratios by
`post_train_pose_drop` (from `impact_analysis_train22.jsonl`) instead of the
pre-training probe, EXTREME tiers × `REALLOC_SCALE=1.15`. Self-contained —
supersedes `--asymmetric/--protect-sppf/--enable-extreme-tier`. Dry-run verified:
**21.38% global** param reduction (vs 21.44% for the train30 recipe). Eases the
7 over-pruned FPN AT_LIMIT convs (0.06→0.034), auto-protects `model.9.cv1`
(SPPF) and `model.4.cv1` (→0), pushes empirically-free layers (`model.8.cv0/cv2`,
box-head `cv2.*`) to 0.345. Launch:

```bash
python prune_yolov8_pose_simple.py --data coco-pose.yaml \
  --epochs 50 --imgsz 640 --batch 16 --lr 5e-5 \
  --reallocate --onecycle-alpha 8 --onecycle-beta 4
```

## `--augment` flag (added 2026-05-26)

Re-enables recovery augmentation (`mosaic=1.0`, `scale=0.5`, `fliplr=0.5`,
`close_mosaic=epochs//5`) over `fine_tune()`'s aug-off defaults. Single-phase
(aug is on during pruning too; mosaic auto-closes for the final ~20% epochs so
BN stats settle). Purpose: test whether regularization breaks the ~0.48
fine-tuning plateau. Validated: overrides safe defaults correctly, `--help` clean.

## `--prune-epochs`, best.pt, `--close-mosaic` (added 2026-05-27)

Two-phase prune-then-finetune + best-checkpoint capture (smoke-tested on coco8):
- `--prune-epochs N` — confines all pruning to the first N epochs (schedule pct
  normalized to N, not total). Epochs N..total are a **pure fine-tuning tail at
  the final frozen architecture**. Default = epochs (no tail; legacy behavior).
- **best.pt** — saved only once pruning is complete (`prunes_made == steps`),
  tracking val pose mAP, saving the EMA weights when improved. A high-mAP epoch
  *during* pruning is at a larger/less-pruned arch and is correctly NOT saved.
  (Rationale: train34's per-epoch peak 0.4974 was at ep17 / only 7% pruned —
  invalid as a 21.5% result; last.pt's 0.4895 was the correct number.)
- `--close-mosaic N` — clean-image finish length. Default now a **fixed 15**
  (was epochs//5), so longer runs get proportionally longer augmentation;
  override per run. Augmented phase = epochs − close_mosaic.

Recommended long run (prune 0-100, fine-tune 100-150, best.pt from the tail):
```bash
python prune_yolov8_pose_simple.py --data coco-pose.yaml \
  --epochs 150 --prune-epochs 100 --imgsz 640 --batch 16 --lr 5e-5 \
  --reallocate --onecycle-alpha 8 --onecycle-beta 4 --augment
```

## Planned ablation chain (one variable per step)

| step | flags added | isolates | run dir | pose50-95 |
|------|-------------|----------|---------|-----------|
| train29 | `--asymmetric --asymmetric-scale 1.0 --protect-sppf --enable-extreme-tier` (softened sched) | baseline | train29 | 0.4796 |
| +reallocate | `--reallocate` (replaces the 3 asym flags) | ratio reallocation | **train31** | **0.4812** (+0.16pp, 21.53%) |
| +augment | `--augment` | recovery augmentation | **train32** | **0.4831** (+0.19pp, 21.53%) |
| +epochs | `--epochs 100` | aug × epochs compounding | **train34** | **0.4895** (+0.64pp; crosses champion 0.4846) |

## Pareto frontier so far (yolov8n-pose, coco-pose, imgsz 640)

| param ↓ | pose50-95 | run | notes |
|---------|-----------|-----|-------|
| 0%      | 0.5116    | baseline | unpruned yolov8n-pose |
| 14.86%  | 0.4846    | champion | asymmetric, default onecycle |
| 21.46%  | 0.4797    | train28  | + protect-sppf + extreme tier |
| 21.46%  | 0.4796    | train29  | + softened schedule |
| 21.46%  | ~0.48     | train30  | 150-epoch (plateaued) |
| 21.53%  | 0.4812    | train31  | `--reallocate` |
| 21.53%  | 0.4831    | train32  | `--reallocate --augment`, 50ep |
| 21.53%  | 0.4895    | train34  | `--reallocate --augment`, 100ep (last.pt) — crosses champion |
| 21.53%  | **0.4911**| **train35** | **150ep req / 118 done (EarlyStop), `--prune-epochs 100`, best.pt from ep118 post-prune tail — NEW BEST, +0.65pp vs champion. Weights file accidentally deleted 2026-05-28; figure cited from prior validation, rerun needed to recover deployable .pt** |
