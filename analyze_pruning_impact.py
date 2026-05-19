"""Post-training per-layer impact analysis on a pruned checkpoint.

For each prunable Conv2d in the TRAINED pruned model, runs the same probe
that `sensitivity.py` runs on the pretrained baseline: zero the bottom-N
filters by L2 norm, run val, restore. Differences vs the pre-training
probe expose three regimes:

  1. **AT LIMIT**: layer's post-training sensitivity is HIGH → applied
     ratio was about right; further pruning would cost real accuracy.
  2. **HAS SLACK**: layer's post-training sensitivity is LOW → applied
     ratio was conservative; could prune more for free.
  3. **REGRESSED**: post-training sensitivity grew vs pre-training →
     fine-tuning made this layer more sensitive (unusual; flag for review).

The script also produces a "headroom" estimate per layer: how much MORE
each layer could theoretically be pruned without exceeding the same
sensitivity threshold the original protection used.

Usage:
    python analyze_pruning_impact.py \
        --pruned runs/pose/train42/weights/last.pt \
        --data coco-pose.yaml \
        --probe-ratio 0.30

Wall time: comparable to running sensitivity.py once (~25-30 min on a
yolov8n-pose checkpoint at imgsz=640; longer for larger models).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

sys.path.insert(0, ".")
# C2f_v2 needs to be importable for the pickle loader.
from prune_yolov8_pose import C2f_v2, find_protected_layers, load_asymmetric_ratios
from sensitivity import find_sibling_bn, list_prunable_convs, zero_bottom_filters, restore_filters


def fmt_val(v):
    return (f"pose50-95={v['pose50_95']:.4f}  pose50={v['pose50']:.4f}  "
            f"box50-95={v['box50_95']:.4f}  box50={v['box50']:.4f}")


def quick_val(yolo, data, imgsz, batch):
    """Val on a deepcopy to avoid mutating the training model via fuse()."""
    from copy import deepcopy
    saved = yolo.model
    yolo.model = deepcopy(saved)
    try:
        m = yolo.val(data=data, imgsz=imgsz, batch=batch, verbose=False, plots=False)
        return {
            "pose50_95": float(m.pose.map), "pose50": float(m.pose.map50),
            "box50_95": float(m.box.map),  "box50":   float(m.box.map50),
        }
    finally:
        yolo.model = saved


def main():
    p = argparse.ArgumentParser(description="Per-layer impact analysis on trained pruned model.")
    p.add_argument("--pruned", required=True, help="Path to trained pruned .pt")
    p.add_argument("--data", default="coco-pose.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--probe-ratio", type=float, default=0.30,
                   help="Fraction of each layer's CURRENT (post-prune) filters "
                        "to zero per probe. Same default as sensitivity.py.")
    p.add_argument("--sensitivity-ref", default="sensitivity_results.jsonl",
                   help="Original pre-training sensitivity for comparison.")
    p.add_argument("--applied-ratios", default=None,
                   help="JSON file mapping layer name → applied ratio. If "
                        "omitted, infers from --asymmetric-scale assuming the "
                        "default SENSITIVITY_TIERS were used during pruning.")
    p.add_argument("--asymmetric-scale", type=float, default=1.0,
                   help="Used to infer applied ratios if --applied-ratios is "
                        "not provided.")
    p.add_argument("--out", default="impact_analysis.jsonl",
                   help="JSONL output: one record per probed layer.")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test mode: probe first N layers only.")
    args = p.parse_args()

    # Load pre-training sensitivity (the original ranking)
    pre_train = {}
    if Path(args.sensitivity_ref).exists():
        with open(args.sensitivity_ref) as f:
            for line in f:
                rec = json.loads(line)
                if "pose_drop" in rec:
                    pre_train[rec["name"]] = rec["pose_drop"]

    # Infer applied ratios (what each layer was pruned at)
    applied = {}
    if args.applied_ratios:
        applied = json.loads(Path(args.applied_ratios).read_text())
    else:
        try:
            applied = load_asymmetric_ratios(scale=args.asymmetric_scale)
        except FileNotFoundError:
            pass  # Uniform pruning; applied will be empty and we'll annotate "uniform"

    print(f"Loading pruned checkpoint: {args.pruned}")
    yolo = YOLO(args.pruned)
    yolo.model.eval()

    print(f"Baseline val on trained pruned model...")
    baseline = quick_val(yolo, args.data, args.imgsz, args.batch)
    print(f"  {fmt_val(baseline)}")
    base_pose = baseline["pose50_95"]

    protected = find_protected_layers(yolo.model)
    prunable = list_prunable_convs(yolo.model, protected)
    if args.limit:
        prunable = prunable[: args.limit]
    print(f"Probing {len(prunable)} prunable layers at ratio={args.probe_ratio:.2f}\n")

    results = []
    out_path = Path(args.out)
    t0 = time.time()
    with open(out_path, "w") as f:
        f.write(json.dumps({"baseline": baseline, "args": vars(args)}) + "\n")
        for i, (name, conv) in enumerate(prunable):
            t_layer = time.time()
            bn = find_sibling_bn(yolo.model, conv)
            saved = zero_bottom_filters(conv, bn, args.probe_ratio)
            n_zero = saved.get("n_zero", 0)
            n_total = conv.weight.shape[0]
            if n_zero == 0:
                print(f"[{i+1}/{len(prunable)}] {name}  SKIPPED (too few filters)")
                continue
            print(f"[{i+1}/{len(prunable)}] {name}  "
                  f"({n_zero}/{n_total} filters zeroed)", flush=True)
            try:
                v = quick_val(yolo, args.data, args.imgsz, args.batch)
                post_drop = base_pose - v["pose50_95"]
                pre_drop = pre_train.get(name)
                ratio_used = applied.get(name)
                shift = (post_drop - pre_drop) if pre_drop is not None else None
                r = {
                    "name": name,
                    "shape": list(conv.weight.shape),
                    "n_filters": n_total,
                    "post_train_pose_drop": post_drop,
                    "pre_train_pose_drop": pre_drop,
                    "applied_ratio": ratio_used,
                    "sensitivity_shift": shift,
                }
                print(f"   post-train Δ pose={post_drop:+.4f}  "
                      f"(pre-train was {pre_drop:+.4f if pre_drop is not None else 0.0})  "
                      f"applied_ratio={ratio_used}  "
                      f"({time.time()-t_layer:.0f}s)")
            except Exception as e:
                r = {"name": name, "error": f"{type(e).__name__}: {e}"}
                print(f"   ERROR: {r['error']}")
            results.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            restore_filters(conv, bn, saved)

    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")

    # Final summary table
    valid = [r for r in results if "post_train_pose_drop" in r]
    valid.sort(key=lambda r: r["post_train_pose_drop"], reverse=True)

    print("\n" + "=" * 110)
    print("PER-LAYER IMPACT ANALYSIS")
    print("=" * 110)
    print(f"{'rank':>4}  {'post-train':>10}  {'pre-train':>10}  {'shift':>10}  "
          f"{'applied':>8}  {'verdict':<16}  layer")
    print("-" * 110)
    for rank, r in enumerate(valid, 1):
        pre = r.get("pre_train_pose_drop")
        post = r["post_train_pose_drop"]
        shift = r.get("sensitivity_shift")
        ratio = r.get("applied_ratio")
        # Heuristic verdict
        if post >= 0.05:
            verdict = "AT_LIMIT"
        elif post >= 0.02:
            verdict = "tight"
        elif post >= 0.005:
            verdict = "ok"
        else:
            verdict = "HAS_SLACK"
        if shift is not None and shift > 0.05:
            verdict += "*"  # sensitivity grew during training
        pre_s = f"{pre:+.4f}" if pre is not None else "  —"
        shift_s = f"{shift:+.4f}" if shift is not None else "  —"
        ratio_s = f"{ratio:.3f}" if ratio is not None else "  —"
        print(f"{rank:>4}  {post:>+10.4f}  {pre_s:>10}  {shift_s:>10}  "
              f"{ratio_s:>8}  {verdict:<16}  {r['name']}")

    n_slack = sum(1 for r in valid if r["post_train_pose_drop"] < 0.005)
    n_limit = sum(1 for r in valid if r["post_train_pose_drop"] >= 0.05)
    print(f"\nSummary: {n_limit} layers AT_LIMIT (further pruning would hurt), "
          f"{n_slack} layers HAS_SLACK (could prune more)")
    print(f"Verdict '*' suffix = sensitivity grew during training (rare; review).")
    print(f"Full per-layer data in {out_path}")


if __name__ == "__main__":
    main()
