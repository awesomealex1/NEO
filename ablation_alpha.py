import argparse
import subprocess
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# --- Configuration ---
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
ALGORITHM = 'neo_cont' # Target for alpha ablation
VIT_TYPE = 'base'
LEVEL = 5
SAMPLES = 512
SEEDS = [2020]

def get_args():
    parser = argparse.ArgumentParser(description='Ablation Study: Alpha (Learning Rate)')
    parser.add_argument('--alphas', type=float, nargs='+', default=ALPHAS, help='List of alphas/LRs to test')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS, help='List of seeds')
    parser.add_argument('--config', type=str, default='config_example.yaml', help='Path to config file')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Skip if result file exists')
    return parser.parse_args()

def run_experiment(alpha, seed, corruption, args):
    cmd = [
        "python", "main.py",
        "--config", args.config,
        "--vit_type", VIT_TYPE,
        "--level", str(LEVEL),
        "--adapt_num_samples", str(SAMPLES),
        "--algorithm", ALGORITHM,
        "--seed", str(seed),
        "--corruption", corruption,
        "--learning_rate", str(alpha)
    ]
    
    # Same filename issue as batch size.
    # main.py filename: f"{algo}-imagenet-{vit}-{seed}-{corruption}"
    # It does NOT include learning_rate.
    
    base_filename = f"results/{ALGORITHM}-imagenet-{VIT_TYPE}-{seed}-{corruption}.json"
    target_filename = f"results/{ALGORITHM}-imagenet-{VIT_TYPE}-{seed}-{corruption}-alpha{alpha}.json"
    
    if args.skip_existing and os.path.exists(target_filename):
        print(f"Skipping existing: {target_filename}")
        return True

    print(f"Running Alpha={alpha}: {' '.join(cmd)}")
    
    if args.dry_run:
        return True
        
    try:
        subprocess.run(cmd, check=True)
        
        if os.path.exists(base_filename):
            os.rename(base_filename, target_filename)
            return True
        else:
            print(f"Error: Result file {base_filename} not found.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error running Alpha={alpha}: {e}")
        return False

def aggregate_results(args):
    results = []
    
    for alpha in args.alphas:
        for corruption in CORRUPTIONS:
            accuracies = []
            for seed in args.seeds:
                filename = f"results/{ALGORITHM}-imagenet-{VIT_TYPE}-{seed}-{corruption}-alpha{alpha}.json"
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        if corruption in data:
                             acc = data[corruption].get('avg_adapt_accuracy1')
                             if acc is not None:
                                 accuracies.append(acc)
            
            if accuracies:
                results.append({
                    'Alpha': alpha,
                    'Corruption': corruption,
                    'Mean Acc': np.mean(accuracies),
                    'Std Dev': np.std(accuracies),
                    'Samples': len(accuracies)
                })
    
    return pd.DataFrame(results)

def main():
    args = get_args()
    
    print(f"--- Ablation: Alpha (LR) ---")
    print(f"Algorithm: {ALGORITHM}")
    print(f"Alphas: {args.alphas}")
    
    for alpha in args.alphas:
        for corruption in CORRUPTIONS:
            for seed in args.seeds:
                run_experiment(alpha, seed, corruption, args)
                
    if not args.dry_run:
        print("\n--- Aggregating ---")
        df = aggregate_results(args)
        if not df.empty:
            pivot = df.pivot(index='Corruption', columns='Alpha', values='Mean Acc')
            print("\nMean Adaptation Accuracy (Top-1) by Alpha:")
            print(pivot.to_markdown(floatfmt=".2f"))
            
            print("\nAverage Across All Corruptions:")
            print(pivot.mean().to_markdown(floatfmt=".2f"))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_csv(f"ablation_alpha_results_{timestamp}.csv", index=False)
            print(f"\nSaved to ablation_alpha_results_{timestamp}.csv")

if __name__ == "__main__":
    main()
