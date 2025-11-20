import os
import sys
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import wandb

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def calculate_entropy(logits, dim=1):
    """
    Calculate entropy of model outputs
    Args:
        logits: model outputs (can be logits or probabilities) - shape [batch_size, num_classes]
        dim: dimension along which to calculate entropy (1 for class dimension)
    Returns:
        entropy: entropy values - shape [batch_size]
    """
    # Better check: see if values sum to ~1 along the class dimension
    sums = torch.sum(logits, dim=dim, keepdim=True)
    is_probably_probs = torch.allclose(sums, torch.ones_like(sums), rtol=1e-3)
    
    if is_probably_probs and logits.min() >= 0.0:
        # Already probabilities
        probs = logits
    else:
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=dim)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    probs = torch.clamp(probs, eps, 1.0)
    
    # Calculate entropy: H(p) = -sum(p * log(p)) across class dimension
    entropy = -torch.sum(probs * torch.log(probs), dim=dim)
    return entropy

def mean(items):
    return sum(items)/len(items)


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger
    

def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_flops(module: nn.Module, size, skip_pattern, device):
    # print(module._auxiliary)
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("init hool for", name)
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size).to(device))
        module.train(mode=training)
        # print(f"training={training}")
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops

def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param

def get_device() -> torch.device:
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/data/imagenet', help='path to dataset')
    parser.add_argument('--data_v2', default='/data/imagenet-v2', help='path to dataset')
    parser.add_argument('--data_sketch', default='/data/sketch', help='path to dataset')
    parser.add_argument('--data_corruption', default='/data/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--data_rendition', default='/data/imagenet-r', help='path to corruption dataset')
    parser.add_argument('--data_cifar_10_c', default='/data/cifar-10-c', help='path to corruption dataset')
    parser.add_argument('--cifar_10_original', default='/data/cifar-10', help='path to CIFAR-10 original dataset')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')
    parser.add_argument('--num_classes', default=1000, type=int, help='number of classes in dataset')
    parser.add_argument('--batch_count', default=-1, type=int, help='number of batches to perform validation on (default: unlimited)')
    parser.add_argument('--adapt_label_count', default=0, type=int, help='number of labels to only use for adaptation (default: 0)')
    parser.add_argument('--adapt_batch_count', default=-1, type=int, help='for how many batches to adapt, if even label distributions (default: unlimited)')
    parser.add_argument('--vit_type', default="base", type=str, help='type of ViT model (small, base, large)')
    parser.add_argument('--save_shift', default=False, action='store_true', help='whether to save the shift vector after adaptation')

    # algorithm selection
    parser.add_argument('--algorithm', default='no_adapt', type=str, help='supporting foa, sar, cotta and etc.')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='all', type=str, help='corruption type of test(val) set. (rendition, sketch, v2, or from all)')
    parser.add_argument('--cifar_10', default=False, action='store_true', help='whether to use CIFAR-10-C dataset')

    # model settings
    parser.add_argument('--quant', default=False, action='store_true', help='whether to use quantized model in the experiment')

    # foa settings
    parser.add_argument('--num_prompts', default=2, type=int, help='number of inserted prompts for test-time adaptation.')    
    parser.add_argument('--fitness_lambda', default=0.4, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA')    
    parser.add_argument('--lambda_bp', default=30, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA-BP')    

    # compared method settings
    parser.add_argument('--margin_e0', default=0.4*math.log(1000), type=float, help='the entropy margin for sar')    

    # neo settings
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate for neo direction updates')

    # output settings
    parser.add_argument('--output', default='./outputs', help='the output directory of this experiment')
    parser.add_argument('--tag', default='_first_experiment', type=str, help='the tag of experiment')
    parser.add_argument('--wandb', default=False, action='store_true', help='whether to log to online wandb')
    parser.add_argument('--adapt_num_samples', type=int, default=-1, help='The exact number of samples to use for adaptation.')
    parser.add_argument('--continual', default=False, action='store_true', help='whether to use continual test-time adaptation (default: False)')
    parser.add_argument('--eval', default=False, action='store_true', help='whether to evaluate after adaptation (default: False)')
    parser.add_argument('--corrupt_center_path', default='', type=str, help='path to pre-saved corrupt class center (for NEO)')
    parser.add_argument('--corrupt_center_save_path', default='', type=str, help='path to save corrupt class center after adaptation (for NEO)')

    return parser.parse_args()


def init_wandb(args):
    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"
    
    wandb.init(
        project="neo",
        entity="entity",    # Replace with your wandb entity
        name=f"{args.algorithm}-{args.dataset_name}-{args.vit_type}",
        config=args
    )
