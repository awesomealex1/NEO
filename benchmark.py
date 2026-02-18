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

ALGORITHMS = ['no_adapt', 'tent', 'sar', 'cotta', 'neo']
SEEDS = [2020, 2021, 2022]

def get_args():
    parser = argparse.ArgumentParser(description='Benchmark ResNet-50 Adaptation')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS, help='List of seeds to run')
    parser.add_argument('--algorithms', type=str, nargs='+', default=ALGORITHMS, help='List of algorithms to run')
    parser.add_argument('--num_samples', type=int, default=512, help='Number of adaptation samples')
    parser.add_argument('--config', type=str, default='config_example.yaml', help='Path to config file for dataset paths')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Skip if result file exists')
    return parser.parse_args()

def run_experiment(algo, seed, corruption, args):
    cmd = [
        "python", "main.py",
        "--config", args.config,
        "--vit_type", "resnet50",
        "--level", "5",
        "--adapt_num_samples", str(args.num_samples),
        "--algorithm", algo,
        "--seed", str(seed),
        "--corruption", corruption
    ]
    
    # Construct expected result filename to check for existence
    # Logic from main.py: name=f"{args.algorithm}-{args.dataset_name}-{args.vit_type}-{args.seed}-{corruption}"
    # Assuming dataset_name is 'imagenet' (default)
    result_filename = f"results/{algo}-imagenet-resnet50-{seed}-{corruption}.json"

    if args.skip_existing and os.path.exists(result_filename):
        print(f"Skipping existing: {result_filename}")
        return True

    print(f"Running: {' '.join(cmd)}")
    if not args.dry_run:
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running {algo} with seed {seed} on {corruption}: {e}")
            return False
    return True

def aggregate_results(args):
    results = []
    
    for algo in args.algorithms:
        for corruption in CORRUPTIONS:
            accuracies = []
            for seed in args.seeds:
                filename = f"results/{algo}-imagenet-resnet50-{seed}-{corruption}.json"
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        # We want accuracy WHILE adapting
                        # main.py saves 'avg_adapt_accuracy1'
                        # Note: main.py uses 'avg_adapt_accuracy1' which is a scalar mean
                        if corruption in data:
                             acc = data[corruption].get('avg_adapt_accuracy1')
                             if acc is not None:
                                 accuracies.append(acc)
                else:
                    if not args.dry_run:
                        print(f"Warning: Missing result file {filename}")

            if accuracies:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                results.append({
                    'Algorithm': algo,
                    'Corruption': corruption,
                    'Mean Acc': mean_acc,
                    'Std Dev': std_acc,
                    'Samples': len(accuracies)
                })
    
    return pd.DataFrame(results)

def main():
    args = get_args()
    
    # 1. Run Experiments
    print(f"--- Starting Benchmark ---")
    print(f"Algorithms: {args.algorithms}")
    print(f"Seeds: {args.seeds}")
    
    for algo in args.algorithms:
        for corruption in CORRUPTIONS:
            for seed in args.seeds:
                run_experiment(algo, seed, corruption, args)
                
    # 2. Aggregate Results
    if not args.dry_run:
        print("\n--- Aggregating Results ---")
        df = aggregate_results(args)
        
        if not df.empty:
            # Pivot table for better viewing
            pivot_df = df.pivot(index='Corruption', columns='Algorithm', values='Mean Acc')
            print("\nMean Adaptation Accuracy (Top-1):")
            print(pivot_df.to_markdown(floatfmt=".2f"))
            
            # Save raw results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_csv(f"benchmark_results_{timestamp}.csv", index=False)
            print(f"\nDetailed results saved to benchmark_results_{timestamp}.csv")
        else:
            print("No results found to aggregate.")

if __name__ == "__main__":
    main()
