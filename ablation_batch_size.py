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

BATCH_SIZES = [1, 4, 8, 16, 32, 64]
ALGORITHM = 'neo' # Default target for batch size ablation
VIT_TYPE = 'base'
LEVEL = 5
SAMPLES = 512
SEEDS = [2020] # Default to single seed for speed, can be overridden

def get_args():
    parser = argparse.ArgumentParser(description='Ablation Study: Batch Size')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=BATCH_SIZES, help='List of batch sizes to test')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS, help='List of seeds')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Skip if result file exists')
    return parser.parse_args()

def run_experiment(batch_size, seed, corruption, args):
    cmd = [
        "python", "-u", "main.py",
        "--config", args.config,
        "--vit_type", VIT_TYPE,
        "--level", str(LEVEL),
        "--adapt_num_samples", str(SAMPLES),
        "--algorithm", ALGORITHM,
        "--seed", str(seed),
        "--corruption", corruption,
        "--batch_size", str(batch_size)
    ]
    
    # Result filename convention from main.py
    # name=f"{args.algorithm}-{args.dataset_name}-{args.vit_type}-{args.seed}-{corruption}"
    # BUT main.py does NOT include batch_size in the filename by default!
    # This is a problem. If we run multiple batch sizes, they will overwrite each other:
    # "neo-imagenet-base-2020-gaussian_noise.json"
    
    # WORKAROUND: We need to modify main.py or rely on a different output directory/tag.
    # main.py has "--tag" which is appended to output directory, but the filename is fixed.
    # Actually, looking at main.py:
    # file_path = Path(f"results/{name}.json")
    
    # We must ensure unique filenames.
    # We can use the "--tag" argument if main.py uses it in the filename, but it doesn't seem to.
    # Wait, main.py lines 38-48 define 'name'. It DOES NOT include batch_size.
    #
    # However, for this ablation, we can use the `tag` argument IF we modify main.py to use it, 
    # OR we can just save the results to a separate directory? 
    # main.py writes to "results/{name}.json".
    
    # Let's check main.py again.
    # 212: parser.add_argument('--output', default='./outputs', help='the output directory of this experiment')
    # But line 50: file_path = Path(f"results/{name}.json") -> hardcoded "results/" dir?
    
    # Line 35: if not os.path.exists("results"): os.makedirs("results")
    
    # Issue: We cannot easily change the output filename structure without modifying main.py.
    # Modification Plan:
    # We will modify main.py to include a generic "--suffix" or allow "--tag" to be part of the filename.
    # Or, we can just run the experiment, read the result, and rename it immediately.
    
    # Let's go with the rename approach for safety without changing main.py logic too much.
    
    base_filename = f"results/{ALGORITHM}-imagenet-{VIT_TYPE}-{seed}-{corruption}.json"
    target_filename = f"results/{ALGORITHM}-imagenet-{VIT_TYPE}-{seed}-{corruption}-bs{batch_size}.json"
    
    if args.skip_existing and os.path.exists(target_filename):
        print(f"Skipping existing: {target_filename}")
        return True

    print(f"Running BS={batch_size}: {' '.join(cmd)}")
    
    if args.dry_run:
        return True
        
    try:
        # Run
        subprocess.run(cmd, check=True)
        
        # Rename result to include batch size
        if os.path.exists(base_filename):
            os.rename(base_filename, target_filename)
            return True
        else:
            print(f"Error: Result file {base_filename} not found after run.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error running BS={batch_size} seed={seed} corruption={corruption}: {e}")
        return False

def aggregate_results(args):
    results = []
    
    for bs in args.batch_sizes:
        for corruption in CORRUPTIONS:
            accuracies = []
            for seed in args.seeds:
                filename = f"results/{ALGORITHM}-imagenet-{VIT_TYPE}-{seed}-{corruption}-bs{bs}.json"
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        if corruption in data:
                             acc = data[corruption].get('avg_adapt_accuracy1')
                             if acc is not None:
                                 accuracies.append(acc)
            
            if accuracies:
                results.append({
                    'Batch Size': bs,
                    'Corruption': corruption,
                    'Mean Acc': np.mean(accuracies),
                    'Std Dev': np.std(accuracies),
                    'Samples': len(accuracies)
                })
    
    return pd.DataFrame(results)

def main():
    args = get_args()
    
    print(f"--- Ablation: Batch Size ---")
    print(f"Algorithm: {ALGORITHM}")
    print(f"Batch Sizes: {args.batch_sizes}")
    
    for bs in args.batch_sizes:
        for corruption in CORRUPTIONS:
            for seed in args.seeds:
                run_experiment(bs, seed, corruption, args)
                
    if not args.dry_run:
        print("\n--- Aggregating ---")
        df = aggregate_results(args)
        if not df.empty:
            pivot = df.pivot(index='Corruption', columns='Batch Size', values='Mean Acc')
            print("\nMean Adaptation Accuracy (Top-1) by Batch Size:")
            print(pivot.to_markdown(floatfmt=".2f"))
            
            # Also average across corruptions
            print("\nAverage Across All Corruptions:")
            print(pivot.mean().to_markdown(floatfmt=".2f"))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_csv(f"ablation_bs_results_{timestamp}.csv", index=False)
            print(f"\nSaved to ablation_bs_results_{timestamp}.csv")

if __name__ == "__main__":
    main()
