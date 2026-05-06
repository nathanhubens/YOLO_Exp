"""Per-layer pruning sensitivity analysis for YOLOv8-pose.

For each prunable Conv2d:
  1. Find its sibling BatchNorm2d (the standard yolov8 Conv block: conv + bn + act).
  2. Rank the layer's filters by L2 norm over (in_channels, kH, kW).
  3. Zero out the bottom-`ratio` filters: conv weights, conv bias (if any), AND
     the BN gamma/beta for those channels (so their output is exactly 0,
     simulating structural channel removal without rebuilding the graph).
  4. Run yolo.val() WITHOUT fine-tuning.
  5. Restore the saved tensors.

Why zero-out instead of torch-pruning structural removal:
  torch-pruning's dependency graph refuses to prune a layer whose outputs feed
  into ignored layers — so "isolate one layer" via ignored_layers makes most
  layers un-prunable. Zero-out doesn't have this constraint: the channels
  produce hard zeros, downstream consumers just see those zeros.

Why this is a faithful sensitivity proxy:
  After fine-tuning, magnitude pruning would remove approximately the same
  filters that L2 norm flags as smallest. So the "what if these were dead"
  signal is what magnitude-based filter pruning will inflict.

Wall time: ~25-30 min (one val per layer, ~25-30s each, no per-iteration reload).

Usage:
    python sensitivity.py --data coco-pose.yaml --ratio 0.30
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

import sys
sys.path.insert(0, ".")
from prune_yolov8_pose import replace_c2f_in_place, find_protected_layers


def find_sibling_bn(model: nn.Module, conv: nn.Conv2d) -> nn.BatchNorm2d | None:
    """Find the BatchNorm2d that sits in the same parent module as `conv`,
    by scanning the tree once. Returns None if no sibling BN exists."""
    target_id = id(conv)
    for parent in model.modules():
        children = list(parent.children())
        for i, c in enumerate(children):
            if id(c) == target_id:
                # Look for a BN sibling in the same parent.
                for sib in children:
                    if isinstance(sib, nn.BatchNorm2d):
                        return sib
                return None
    return None


def list_prunable_convs(model: nn.Module, protected: list[nn.Module]
                        ) -> list[tuple[str, nn.Conv2d]]:
    protected_ids = {id(m) for m in protected}
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and id(m) not in protected_ids]


def zero_bottom_filters(conv: nn.Conv2d, bn: nn.BatchNorm2d | None,
                        ratio: float) -> dict:
    """Zero the bottom `ratio` filters by L2 norm. Return a dict of saved
    original tensors (cloned, on CPU) so the caller can restore."""
    w = conv.weight
    n_filters = w.shape[0]
    n_zero = int(n_filters * ratio)
    if n_zero == 0:
        return {"n_zero": 0}

    # Filter L2 norms.
    norms = w.detach().view(n_filters, -1).norm(dim=1)
    _, bottom = norms.topk(n_zero, largest=False)

    saved = {
        "n_zero": n_zero,
        "indices": bottom.cpu().clone(),
        "conv_weight_rows": w.detach()[bottom].cpu().clone(),
        "conv_bias_rows": (conv.bias.detach()[bottom].cpu().clone()
                           if conv.bias is not None else None),
        "bn_weight_rows": (bn.weight.detach()[bottom].cpu().clone()
                           if bn is not None else None),
        "bn_bias_rows": (bn.bias.detach()[bottom].cpu().clone()
                         if bn is not None else None),
    }

    with torch.no_grad():
        w[bottom] = 0
        if conv.bias is not None:
            conv.bias[bottom] = 0
        if bn is not None:
            bn.weight[bottom] = 0
            bn.bias[bottom] = 0
    return saved


def restore_filters(conv: nn.Conv2d, bn: nn.BatchNorm2d | None, saved: dict):
    if saved["n_zero"] == 0:
        return
    idx = saved["indices"].to(conv.weight.device)
    with torch.no_grad():
        conv.weight[idx] = saved["conv_weight_rows"].to(conv.weight.device)
        if saved["conv_bias_rows"] is not None and conv.bias is not None:
            conv.bias[idx] = saved["conv_bias_rows"].to(conv.bias.device)
        if bn is not None and saved["bn_weight_rows"] is not None:
            bn.weight[idx] = saved["bn_weight_rows"].to(bn.weight.device)
            bn.bias[idx] = saved["bn_bias_rows"].to(bn.bias.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n-pose.pt")
    p.add_argument("--data", default="coco-pose.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--ratio", type=float, default=0.30,
                   help="Per-layer filter zero-out ratio (0..1)")
    p.add_argument("--out", default="sensitivity_results.jsonl")
    p.add_argument("--limit", type=int, default=None,
                   help="Only probe first N layers (smoke test)")
    args = p.parse_args()

    print("=" * 72)
    print(f"Sensitivity probe @ ratio={args.ratio:.2f}")
    print("=" * 72)

    # Baseline: load fresh, run val.
    yolo = YOLO(args.model)
    replace_c2f_in_place(yolo.model)
    yolo.model.eval()
    print("Baseline validation (no zero-out)...")
    bm = yolo.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                  verbose=False, plots=False)
    baseline = {
        "pose_map5095": float(bm.pose.map),
        "pose_map50": float(bm.pose.map50),
        "box_map5095": float(bm.box.map),
        "box_map50": float(bm.box.map50),
    }
    print(f"  Baseline pose_map50-95={baseline['pose_map5095']:.4f}, "
          f"box_map50-95={baseline['box_map5095']:.4f}")

    # Reload to get a fresh model that's not yet tainted by val()'s fused weights.
    yolo = YOLO(args.model)
    replace_c2f_in_place(yolo.model)
    protected = find_protected_layers(yolo.model)
    prunable_names = [n for n, _ in list_prunable_convs(yolo.model, protected)]
    if args.limit:
        prunable_names = prunable_names[: args.limit]
    del yolo
    print(f"\nProbing {len(prunable_names)} prunable Conv2d layers...")

    out_path = Path(args.out)
    results = []
    t0 = time.time()
    with open(out_path, "w") as f:
        f.write(json.dumps({"baseline": baseline, "args": vars(args)}) + "\n")
        for i, name in enumerate(prunable_names):
            t_layer = time.time()
            # Fresh load each iteration: avoids inference-tensor taint from prior val().
            yolo = YOLO(args.model)
            replace_c2f_in_place(yolo.model)
            modules = dict(yolo.model.named_modules())
            conv = modules[name]
            bn = find_sibling_bn(yolo.model, conv)
            saved = zero_bottom_filters(conv, bn, args.ratio)
            n_zero = saved.get("n_zero", 0)
            n_total = conv.weight.shape[0]
            print(f"\n[{i+1}/{len(prunable_names)}] {name}  "
                  f"({n_zero}/{n_total} filters zeroed, BN={'yes' if bn else 'no'})",
                  flush=True)
            if n_zero == 0:
                r = {"name": name, "skipped": "ratio rounded to 0 filters",
                     "n_filters": n_total}
                results.append(r)
                f.write(json.dumps(r) + "\n"); f.flush()
                del yolo
                continue
            try:
                yolo.model.eval()
                m = yolo.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                             verbose=False, plots=False)
                r = {
                    "name": name,
                    "n_filters": n_total,
                    "n_zeroed": n_zero,
                    "pose_map5095": float(m.pose.map),
                    "pose_map50": float(m.pose.map50),
                    "box_map5095": float(m.box.map),
                    "box_map50": float(m.box.map50),
                    "pose_drop": baseline["pose_map5095"] - float(m.pose.map),
                    "box_drop": baseline["box_map5095"] - float(m.box.map),
                }
                print(f"   pose Δ = {r['pose_drop']:+.4f}  "
                      f"box Δ = {r['box_drop']:+.4f}  "
                      f"({time.time()-t_layer:.0f}s)")
            except Exception as e:
                r = {"name": name, "error": f"{type(e).__name__}: {e}"}
                print(f"   ERROR: {r['error']}")
            results.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            del yolo
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")

    valid = [r for r in results if "pose_drop" in r]
    valid.sort(key=lambda r: r["pose_drop"], reverse=True)

    print("\n" + "=" * 88)
    print(f"{'rank':>4}  {'pose_drop':>10}  {'box_drop':>10}  {'n_filters':>9}  layer")
    print("-" * 88)
    for rank, r in enumerate(valid, 1):
        print(f"{rank:>4}  {r['pose_drop']:>+10.4f}  {r['box_drop']:>+10.4f}  "
              f"{r['n_filters']:>9d}  {r['name']}")

    print("\nMost pose-sensitive layers should be added to find_protected_layers() "
          "or given low per-layer ratios in the Pruner dict.")


if __name__ == "__main__":
    main()
