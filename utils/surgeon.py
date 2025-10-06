# KATANA: Simple Post-Training Robustness Using Test Time Augmentations
# https://arxiv.org/pdf/2109.08191v1.pdf
import PIL
import torch
import torchvision.transforms.functional as F_vis
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random
"""AutoFreeze FC"""
import torch
from torch import nn, Tensor
from torch.nn import Linear
from typing import Optional, Any
from collections import defaultdict

from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states, detach_variable, checkpoint
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

import collections
from itertools import repeat
import logging

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = None
        self.sum = 0
        self.count = 0
        self.values = []
        self.update_cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.values.append(val)
        self.update_cnt += 1

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else None

    @property
    def max(self):
        return np.max(self.values) if self.count > 0 else None

    @property
    def step_avg(self):
        return np.mean(self.values)

    @property
    def step_std(self):
        return np.std(self.values)

    def __str__(self):
        if self.count > 0:
            fmtstr = '{name} {val' + self.fmt + '} (avg={avg' + self.fmt + '})'
            return fmtstr.format(name=self.name, val=self.val, avg=self.avg)
        else:
            return f'{self.name}: N/A'

def split_results_by_domain(domain_dict, data, predictions):
    """
    Separate the labels and predictions by domain
    :param domain_dict: dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
    :param data: list containing [images, labels, domains, ...]
    :param predictions: tensor containing the predictions of the model
    :return: updated result dict
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict

def eval_domain_dict(domain_dict, domain_seq=None):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    :param domain_dict: dictionary containing the labels and predictions for each domain
    :param domain_seq: if specified and the domains are contained in the domain dict, the results will be printed in this order
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    dom_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting up the results by domain...")
    for key in dom_names:
        content = np.array(domain_dict[key])
        correct.append((content[:, 0] == content[:, 1]).sum())
        num_samples.append(content.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")

def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 device: torch.device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    cache_mt = AverageMeter('Cache', ':6.3f')
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            predictions = output.argmax(1)
            correct += (predictions == labels.to(device)).float().sum()
            mem = get_mem(model)
            cache_mt.update(mem)
    accuracy = correct.item() / len(data_loader.dataset)
    return accuracy, domain_dict, cache_mt

def get_mem(model: torch.nn.Module):
    """Get cache memory costs of each layer."""
    FC_cache_size = 0
    for mod_name, target_mod in model.named_modules():
        # Cache size of FC layers
        if isinstance(target_mod, nn.Linear):
            FC_cache_size = FC_cache_size + target_mod.back_cache_size

    # return (BN_cache_size + Conv_cache_size + FC_cache_size) * 4 / (2 ** 20) # Total backward cache size
    return (FC_cache_size) * 4 / (2 ** 20) * 2 # Total backward cache size
    # Note!: If the loss includes Consistency Regularization (CR), the total backward cache size doubles


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)

class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F_vis.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F_vis.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F_vis.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F_vis.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F_vis.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F_vis.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F_vis.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F_vis.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F_vis.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                                            # after contrast
                img = F_vis.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string

def get_tta_transforms(dataset, gaussian_std: float=0.005, soft=False, padding_mode='edge', cotta_augs=True):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    tta_transforms = [
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode=padding_mode),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=0
        )
    ]
    if cotta_augs:
        tta_transforms += [transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
                           transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5),
                           GaussianNoise(0, gaussian_std),
                           Clip(0.0, 1.0)]
    else:
        tta_transforms += [transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5),
                           Clip(0.0, 1.0)]

    return transforms.Compose(tta_transforms)



def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")

class AutoFreezeFC(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name=None, num=0, BN_only=False):
        super(AutoFreezeFC, self).__init__(in_features, out_features, True if bias is not None else False)
        self.name = name
        self.num = num
        self.clip_ratio = 0 # 0-without pruning, 1-pruning all activations
        self.sparsity_signal = False
        self.back_cache_size = 0
        self.activation_size = 0
        self.forward_compute = 0
        self.backward_compute = 0
        self.BN_only = BN_only # False-updating all layers, True-updating BN layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return AutoFreezeFCFunction.apply(self, True, x, self.weight, self.bias)
    
    def forward_autofreeze(self, input, weight, bias) -> torch.Tensor:
        return F.linear(input, weight, bias)

def forward_compute(in_features, out_features):
    # Compute the computational cost of FC forward
    return in_features * out_features * 2

def backward_compute(in_features, out_features):
    # Compute the computational cost of gradients for W and x
    dL_dW_flops = in_features * out_features * 2
    dL_dX_flops = out_features * in_features * 2
    return dL_dW_flops + dL_dX_flops
    # return dL_dX_flops

dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}

def clip_tensor(input_tensor, clip_ratio, mode='random'):
    """
    Args:
        input_tensor (torch.Tensor), size: [n, w, h]
        clip_ratio (float): range: [0, 1]
        mode (str): 'min_abs'
    Returns:
        torch.Tensor
    """
    input_tensor = input_tensor.clone()

    n, c, w, h = input_tensor.shape

    num_to_clip = int(clip_ratio * n * c * w * h)

    if mode == 'min_abs':
        abs_tensor = torch.abs(input_tensor)
        min_abs_indices = abs_tensor.view(-1).argsort()[:num_to_clip]
        zero_indices = min_abs_indices

    # Create a mask tensor with ones everywhere except at zero_indices
    mask_tensor = torch.ones_like(input_tensor)
    mask_tensor.view(-1)[zero_indices] = 0

    return mask_tensor

class Clipper(object):
    def __init__(self, clip_ratio, mode):
        self.clip_ratio = clip_ratio
        self.mode = mode
        self.clip = getattr(self, f"clip_{mode}")
        self.reshape = getattr(self, f"reshape_{mode}")

    def clip_min_abs(self, x: torch.Tensor, ctx=None):
        ctx.x_shape = x.shape
        numel = x.numel()
        # Use reshape() instead of view() to handle non-contiguous tensors
        x_flat = x.reshape(-1) 
        idxs = x_flat.abs().topk(int(numel * (1 - self.clip_ratio)), sorted=False)[1]
        x_clipped = x_flat[idxs]
        ctx.idxs = idxs.to(torch.int32)
        ctx.numel = numel
        return x_clipped

    def reshape_min_abs(self, x, ctx=None):
        idxs = ctx.idxs.to(torch.int64)
        del ctx.idxs
        return torch.zeros(
            ctx.numel, device=x.device, dtype=x.dtype
        ).scatter_(0, idxs, x)


class AutoFreezeFCFunction(torch.autograd.Function):
    """Will stochastically cache data for backwarding."""

    @staticmethod
    def forward(ctx, autofreeze_fc: AutoFreezeFC, preserve_rng_state, x, weight, bias):
        check_backward_validity([x, weight, bias])
        # print(f"### x req grad: {x.requires_grad}")
        ctx.autofreeze_fc = autofreeze_fc
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(x)

        with torch.no_grad():
            y = autofreeze_fc.forward_autofreeze(x, weight, bias)

            # Computation costs
            autofreeze_fc.forward_compute = forward_compute(autofreeze_fc.in_features, autofreeze_fc.out_features)
            autofreeze_fc.backward_compute = forward_compute(autofreeze_fc.in_features, autofreeze_fc.out_features)

            mode = "min_abs"
            # Dynamic Activation Sparsity
            if autofreeze_fc.sparsity_signal: # DAS
                clip_strategy = Clipper(autofreeze_fc.clip_ratio, mode)
                autofreeze_fc.back_cache_size = int(x.numel() * (1 - autofreeze_fc.clip_ratio))
            else:
                clip_strategy = Clipper(0, mode)
                autofreeze_fc.activation_size = int(x.numel())
            ctx.clip_strategy = clip_strategy
            clipped_x = clip_strategy.clip(x, ctx)

            ctx.save_for_backward(clipped_x)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        autofreeze_fc = ctx.autofreeze_fc
        clip_strategy = ctx.clip_strategy
        x = ctx.saved_tensors
        # sparsity_signal = autofreeze_fc.sparsity_signal
        BN_only = autofreeze_fc.BN_only

        # Stash the surrounding rng state, and mimic the state that was present at this time during forward. Restore the surrounding state when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)

            # grad computing
            # grad_b = torch.sum(grad_out, list(range(grad_out.dim()-1))) if autofreeze_fc.bias is not None else None
            ic, oc = autofreeze_fc.weight.shape
            clipped_x = clip_strategy.reshape(x[0], ctx)
            grad_out_reshaped = grad_out.reshape(-1, ic)
            clipped_x_reshaped = clipped_x.reshape(-1, oc)
            grad_w = grad_out_reshaped.T.mm(clipped_x_reshaped)
            grad_x = torch.matmul(grad_out, autofreeze_fc.weight, out=clipped_x.view(ctx.x_shape))

            del clip_strategy, autofreeze_fc

        if BN_only: # BN-only mode
            return None, None, grad_x, None, None
        else:
            return None, None, grad_x, grad_w, None