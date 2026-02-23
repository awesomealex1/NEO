import os
import sys
sys.path.append(os.getcwd())

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb

from utils.utils import get_args, set_reproducible, init_wandb, get_device
from factories import init_corruptions_dataset, init_model, init_data_loaders
from tta_library.neo import get_feature_extractor

import neural_collapse as nc
from neural_collapse.accumulate import MeanAccumulator, VarNormAccumulator, DecAccumulator
from neural_collapse.measure import variability_cdnv, simplex_etf_error, self_duality_error, clf_ncc_agreement

def extract_classifier_weights(net):
    if hasattr(net, 'head') and isinstance(net.head, torch.nn.Linear):
        return net.head.weight.data
    elif hasattr(net, 'classifier') and isinstance(net.classifier, torch.nn.Linear):
        return net.classifier.weight.data
    elif hasattr(net, 'fc') and isinstance(net.fc, torch.nn.Linear):
        return net.fc.weight.data
    
    # Fallback to try and automatically find the last linear layer
    last_linear = None
    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module
            
    if last_linear is not None:
        return last_linear.weight.data
        
    raise ValueError("Could not find classifier weights.")

def main():
    args = get_args()
    set_reproducible(args.seed)
    args.corruptions, args.dataset_name, args.num_classes = init_corruptions_dataset(args)
    
    # Force evaluation mode for feature extraction context
    args.eval = True 
    
    # We want to measure NC on the full corrupted dataset without artificial adaptation splits
    args.adapt_num_samples = -1 
    
    net = init_model(args)
    weights = extract_classifier_weights(net).to(get_device()) # shape: [num_classes, feature_dim]
    num_classes, feature_dim = weights.shape
    
    feature_extractor = get_feature_extractor(net)
    
    init_wandb(args)
    
    if not os.path.exists("results_nc"):
        os.makedirs("results_nc")
        
    all_nc_results = {}
    
    for i, corruption in enumerate(args.corruptions):
        args.corruption = corruption
        name = f"nc-{args.dataset_name}-{args.vit_type}-{args.seed}-{corruption}"
        file_path = Path(f"results_nc/{name}.json")
        
        if file_path.is_file():
            print(f"Skipping {name}, results already exist.")
            # Load existing to contribute to summary
            with open(file_path, "r") as f:
                all_nc_results[corruption] = json.load(f)
            continue
            
        print(f"\n--- Processing Corruption: {corruption} ---")
        
        adapt_loader, val_loader = init_data_loaders(args)
        
        # We use adapt_loader since prepare_test_data maps adapt_dataset to all samples if adapt_num_samples == -1
        loader = adapt_loader if adapt_loader is not None else val_loader
        if loader is None:
             print("No data loader found.")
             continue
             
        # Cache features
        all_features = []
        all_labels = []
        
        feature_extractor.eval()
        with torch.no_grad():
            for images, target in tqdm(loader, desc="Extracting Features"):
                images = images.to(get_device())
                target = target.to(get_device())
                
                features = feature_extractor(images)
                
                # Squeeze in case ViT features have an extra seq dimension like [B, 1, D]
                if features.dim() == 3:
                     features = features.squeeze(1)
                     
                all_features.append(features.cpu())
                all_labels.append(target.cpu())
                
        all_features = torch.cat(all_features, dim=0).to(get_device())
        all_labels = torch.cat(all_labels, dim=0).to(get_device())
        
        print(f"Extracted features shape: {all_features.shape}")
        
        # --- NC Measurements ---
        try:
            # 1. Means
            mean_accum = MeanAccumulator(num_classes, feature_dim, get_device())
            mean_accum.accumulate(all_features, all_labels)
            means, mG = mean_accum.compute()
            
            # 2. Variances & Decision Agreement (using means)
            var_norms_accum = VarNormAccumulator(num_classes, feature_dim, get_device(), M=means)
            var_norms_accum.accumulate(all_features, all_labels, means)
            var_norms, _ = var_norms_accum.compute()
            
            dec_accum = DecAccumulator(num_classes, feature_dim, get_device(), M=means, W=weights)
            dec_accum.accumulate(all_features, all_labels, weights, means)
            
            # 3. Calculate Results
            results = {
                "nc1_cdnv": variability_cdnv(var_norms, means).item(),
                "nc2_etf_err": simplex_etf_error(means, mG).item(),
                "nc3_dual_err": self_duality_error(weights, means, mG).item(),
                "nc4_agree": clf_ncc_agreement(dec_accum).item(),
            }
            
            print(f"Results for {corruption}:")
            for k, v in results.items():
                print(f"  {k}: {v:.6f}")
                
            all_nc_results[corruption] = results
            wandb.log({f"nc/{corruption}/{k}": v for k, v in results.items()})
            
            with open(file_path, "w") as outfile:
                json.dump(results, outfile, indent=4)
                
        except Exception as e:
            print(f"Error computing NC stats for {corruption}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n--- Overall NC Summary ---")
    if len(all_nc_results) > 0:
        agg_results = {k: np.mean([res[k] for res in all_nc_results.values() if k in res]) for k in list(all_nc_results.values())[0].keys()}
        for k, v in agg_results.items():
            print(f"Mean {k}: {v:.4f}")
        
        with open("results_nc/summary.json", "w") as f:
            json.dump(agg_results, f, indent=4)

if __name__ == '__main__':
    main()
