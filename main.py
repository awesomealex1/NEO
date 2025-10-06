import os
import sys
sys.path.append(os.getcwd())

import time
import argparse
import random
import math
from importlib import reload, import_module
from evaluator import run
from utils.utils import get_logger, get_device, calculate_entropy, get_args, set_reproducible, init_wandb
from utils.cli_utils import *
import wandb
from factories import init_adapt_model, init_corruptions_dataset, init_data_loaders, init_model
import torch    
import numpy as np
import json
from pathlib import Path


def main():
    args = get_args()
    set_reproducible(args.seed)
    args.corruptions, args.dataset_name, args.num_classes = init_corruptions_dataset(args)
    net = init_model(args)
    adapt_model = init_adapt_model(args, net)
    init_wandb(args)
    
    avg_val_acc1, avg_val_acc5, avg_val_entropy, avg_val_ece = [], [], [], []
    avg_adapt_acc1, avg_adapt_acc5, avg_adapt_entropy, avg_adapt_ece = [], [], [], []

    if args.continual:
        random.shuffle(args.corruptions)
    
    if not os.path.exists("results"):
            os.makedirs("results")
    
    for i, corruption in enumerate(args.corruptions):
        all_results = {}
        args.corruption = corruption
        if args.continual:
            name=f"continual-{args.algorithm}-{args.dataset_name}-{args.vit_type}-{args.seed}-{corruption}-{i}"
        elif args.corrupt_center_path != '':
            name=f"load-{args.algorithm}-{args.dataset_name}-{args.vit_type}-{args.seed}-{corruption}-{args.corrupt_center_path.split('/')[-1].split('.')[0]}"
        elif args.adapt_label_count > 0:
            name=f"{args.algorithm}-{args.dataset_name}-{args.vit_type}-{args.seed}-{corruption}-{args.adapt_label_count}"
        else:
            name=f"{args.algorithm}-{args.dataset_name}-{args.vit_type}-{args.seed}-{corruption}"

        file_path = Path(f"results/{name}.json")

        

        if file_path.is_file():
            print(f"Skipping {name}, results already exist.")
            continue
        
        adapt_loader, val_loader = init_data_loaders(args)
        top1_adapt_vals, top5_adapt_vals, entropy_adapt_vals, ece_adapt_vals = [], [], [], []
        top1_val_vals, top5_val_vals, entropy_val_vals, ece_val_vals =  [], [], [], []
        top1_adapt_avg, top5_adapt_avg, entropy_adapt_avg, ece_adapt_avg = [-1] * 4
        top1_val_avg, top5_val_avg, entropy_val_avg, ece_val_avg =  [-1] * 4
        
        # --- Phase 1: Adaptation ---
        if adapt_loader != None:
            top1_adapt, top5_adapt, entropy_adapt, ece_adapt = run(loader=adapt_loader, model=adapt_model, args=args, adapt=True)
            top1_adapt_avg, top5_adapt_avg, entropy_adapt_avg, ece_adapt_avg = top1_adapt.avg.item(), top5_adapt.avg.item(), entropy_adapt.avg, ece_adapt.avg
            top1_adapt_vals, top5_adapt_vals, entropy_adapt_vals, ece_adapt_vals = top1_adapt.vals, top5_adapt.vals, entropy_adapt.vals, ece_adapt.vals
        
            print(f"\nAdaptation Results for \033[94m{args.corruption}\033[0m with algorithm \033[93m{args.algorithm}\033[0m:")
            print(f"  Adapt Validation Top-1 Accuracy: {top1_adapt_avg:.6f}")
            print(f"  Adapt Validation Top-5 Accuracy: {top5_adapt_avg:.6f}")
            print(f"  Adapt Validation Entropy: {entropy_adapt_avg:.6f}")
            print(f"  Adapt ECE Avg: {ece_adapt_avg:.6f}\n")

            if args.save_shift and "neo" in args.algorithm:
                if args.corrupt_center_save_path != '':
                    torch.save(adapt_model.corrupt_class_center, args.corrupt_center_save_path)
                else:
                    torch.save(adapt_model.corrupt_class_center, f"results/{name}_corrupt_center.pth")
        
        # --- Phase 2: Validation (Do not continue adapting at this point) --- 
        if val_loader != None and args.eval:
            top1_val, top5_val, entropy_val, ece_val = run(loader=val_loader, model=adapt_model, args=args, adapt=False)
            top1_val_avg, top5_val_avg, entropy_val_avg, ece_val_avg = top1_val.avg.item(), top5_val.avg.item(), entropy_val.avg, ece_val.avg
            top1_val_vals, top5_val_vals, entropy_val_vals, ece_val_vals = top1_val.vals, top5_val.vals, entropy_val.vals, ece_val.vals

            print(f"\Validation Results for \033[94m{args.corruption}\033[0m with algorithm \033[93m{args.algorithm}\033[0m:")
            print(f"  Validation Top-1 Accuracy: {top1_val_avg:.6f}")
            print(f"  Validation Top-5 Accuracy: {top5_val_avg:.6f}")
            print(f"  Validation Entropy: {entropy_val_avg:.6f}")
            print(f"  Validation ECE Avg: {ece_val_avg:.6f}\n")
        
        # Store results for the current corruption
        corruption_results = {
            'avg_val_accuracy1': top1_val_avg,
            'avg_val_accuracy5': top5_val_avg,
            'avg_val_entropy': entropy_val_avg,
            'avg_val_ece': ece_val_avg,
            'avg_adapt_accuracy1': top1_adapt_avg,
            'avg_adapt_accuracy5': top5_adapt_avg,
            'avg_adapt_entropy': entropy_adapt_avg,
            'avg_adapt_ece': ece_adapt_avg,
            'val_accuracy1': [val.item() for val in top1_val_vals],
            'val_accuracy5': [val.item() for val in top5_val_vals],
            'val_entropy': [val for val in  entropy_val_vals],
            'val_ece': [val for val in  ece_val_vals],
            'adapt_accuracy1': [val.item() for val in top1_adapt_vals],
            'adapt_accuracy5': [val.item() for val in top5_adapt_vals],
            'adapt_entropy': [val for val in entropy_adapt_vals],
            'adapt_ece': [val for val in ece_adapt_vals],
        }
        all_results[corruption] = corruption_results
        
        avg_val_acc1.append(top1_val_avg)
        avg_val_acc5.append(top5_val_avg)
        avg_val_entropy.append(entropy_val_avg)
        avg_val_ece.append(ece_val_avg)
        avg_adapt_acc1.append(top1_adapt_avg)
        avg_adapt_acc5.append(top5_adapt_avg)
        avg_adapt_entropy.append(entropy_adapt_avg)
        avg_adapt_ece.append(ece_adapt_avg)
        
        torch.cuda.empty_cache()
        if not args.continual:
            adapt_model.reset()
    
        wandb.log(all_results)
        with open(f"results/{name}.json", "w") as outfile:
            json.dump(all_results, outfile, indent=4)
    
    print("--- Overall Summary ---")
    mean_val_acc1 = np.mean(avg_val_acc1)
    mean_val_acc5 = np.mean(avg_val_acc5)
    mean_val_entropy = np.mean(avg_val_entropy)
    mean_val_ece = np.mean(avg_val_ece)

    mean_adapt_acc1 = np.mean(avg_adapt_acc1)
    mean_adapt_acc5 = np.mean(avg_adapt_acc5)
    mean_adapt_entropy = np.mean(avg_adapt_entropy)
    mean_adapt_ece = np.mean(avg_adapt_ece)
    
    print(f'Mean Top-1 Adaptation Accuracy across all corruptions: {mean_adapt_acc1:.4f}')
    print(f'Mean Top-5 Adaptation Accuracy across all corruptions: {mean_adapt_acc5:.4f}')
    print(f'Mean Adaptation Entropy across all corruptions: {mean_adapt_entropy:.4f}')
    print(f'Mean Adaptation ECE across all corruptions: {mean_adapt_ece:.4f}')

    print(f'Mean Top-1 Validation Accuracy across all corruptions: {mean_val_acc1:.4f}')
    print(f'Mean Top-5 Validation Accuracy across all corruptions: {mean_val_acc5:.4f}')
    print(f'Mean Validation Entropy across all corruptions: {mean_val_entropy:.4f}')
    print(f'Mean Validation ECE across all corruptions: {mean_val_ece:.4f}')

if __name__ == '__main__':
    main()