import argparse
import glob
import os
import re
import time
import pandas as pd
import torch
import torch_pruning as tp
from ultralytics import YOLO

def parse_ratio_from_dir(dir_name):
    """
    Extracts the ratio from folder names like 'train_ratio012'.
    Assumes format where '012' means 0.12 (12%).
    """
    match = re.search(r'ratio(\d+)', dir_name)
    if match:
        val = int(match.group(1))
        # e.g., '012' -> 12 -> 0.12. '05' -> 5 -> 0.05
        # If your naming convention differs (e.g. 125 -> 0.125), adjust this math
        divisor = 10 ** len(match.group(1)) if val < 10 else 100
        return float(val) / divisor
    return None

def main():
    parser = argparse.ArgumentParser(description="Analyze YOLOv8-pose pruning runs.")
    parser.add_argument("--model-name", required=True, help="Model folder name (e.g., 'yolov8x-pose-p6' or 'yolov8n-pose')")
    parser.add_argument("--baseline-model", default="yolov8x-pose-p6.pt", help="Original unpruned model for base metrics.")
    parser.add_argument("--data", default="coco-pose.yaml", help="Dataset yaml for validation.")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size for P6 model evaluation.")
    parser.add_argument("--fast", action="store_true", help="Read mAP from results.csv instead of re-running yolo.val().")
    args = parser.parse_args()
    
    # Build the model-specific runs directory
    model_runs_dir = os.path.join("./runs/pose", args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    # Verify model directory exists
    if not os.path.isdir(model_runs_dir):
        print(f"Error: Model directory '{model_runs_dir}' not found.")
        return

    print(f"=== Establishing Baseline ({args.baseline_model}) ===")
    base_yolo = YOLO(args.baseline_model)
    base_model = base_yolo.model.eval().to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(base_model, example_inputs)
    print(f"Baseline: {base_macs/1e9:.2f} GMACs, {base_params/1e6:.2f}M params\n")

    # Find all directories matching *train_ratio* within the model-specific directory
    run_dirs = glob.glob(os.path.join(model_runs_dir, "*train_ratio*"))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    
    if not run_dirs:
        print(f"No directories matching '*train_ratio*' found in {model_runs_dir}")
        return

    results = []

    for run_dir in sorted(run_dirs):
        ratio = parse_ratio_from_dir(os.path.basename(run_dir))
        weights_path = os.path.join(run_dir, "weights", "last.pt")
        results_csv = os.path.join(run_dir, "results.csv")

        if not os.path.exists(weights_path):
            print(f"Skipping {run_dir} - last.pt not found.")
            continue

        print(f"=== Analyzing {os.path.basename(run_dir)} (Ratio: {ratio}) ===")
        yolo = YOLO(weights_path)
        
        # 1. Calculate MACs and Params using torch_pruning
        model = yolo.model.eval().to(device)
        macs, params = tp.utils.count_ops_and_params(model, example_inputs)

        # 2. Get mAP Metrics
        if args.fast and os.path.exists(results_csv):
            # FAST PATH: Read the best/last epoch directly from Ultralytics logs
            print("  -> Fast mode: Reading mAP from results.csv")
            df_res = pd.read_csv(results_csv)
            df_res.columns = df_res.columns.str.strip() # Clean Ultralytics whitespace
            last_row = df_res.iloc[-1]
            box_map50 = float(last_row.get("metrics/mAP50(B)", 0))
            box_map5095 = float(last_row.get("metrics/mAP50-95(B)", 0))
            pose_map50 = float(last_row.get("metrics/mAP50(P)", 0))
            pose_map5095 = float(last_row.get("metrics/mAP50-95(P)", 0))
        else:
            # SLOW PATH: Re-run full validation (Matches your snippet exactly)
            print("  -> Running full yolo.val() ... this may take a while.")
            metrics = yolo.val(data=args.data, imgsz=args.imgsz, verbose=False, plots=False)
            box_map50 = float(metrics.box.map50)
            box_map5095 = float(metrics.box.map)
            pose_map50 = float(metrics.pose.map50)
            pose_map5095 = float(metrics.pose.map)

        # 3. Assemble the requested dictionary
        final = {
            "run_name": os.path.basename(run_dir),
            "ratio_arg": ratio,
            "box_map50": round(box_map50, 4),
            "box_map5095": round(box_map5095, 4),
            "pose_map50": round(pose_map50, 4),
            "pose_map5095": round(pose_map5095, 4),
            "params_m": round(params / 1e6, 3),
            "macs_g": round(macs / 1e9, 3),
            "param_reduction_pct": round(100 * (1 - params / base_params), 2),
            "mac_reduction_pct": round(100 * (1 - macs / base_macs), 2),
        }
        
        results.append(final)
        print(f"  -> Params: {final['params_m']}M (-{final['param_reduction_pct']}%) | MACs: {final['macs_g']}G (-{final['mac_reduction_pct']}%)")
        print(f"  -> Pose mAP50-95: {final['pose_map5095']}\n")

    # Output formatting
    if results:
        df = pd.DataFrame(results).sort_values("ratio_arg")
        print("\n" + "="*80)
        print("FINAL COMPARISON TABLE")
        print("="*80)
        # Drop run_name if it clutters the terminal, but keep in CSV
        display_cols = ["ratio_arg", "param_reduction_pct", "mac_reduction_pct", 
                        "pose_map50", "pose_map5095", "box_map50", "box_map5095"]
        print(df[display_cols].to_string(index=False))
        
        csv_path = "pruning_analysis_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()