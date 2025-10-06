from tta_library.sam import SAM
from tta_library.t3a import T3A
from tta_library.foa import FOA
from tta_library.lame import LAME
from tta_library.no_adapt import NoAdapt
from tta_library.neo import NEO
import tta_library.tent as tent
import tta_library.sar as sar
import tta_library.cotta as cotta
from tta_library.neo_cont import NEO_Cont
from tta_library.surgeon import Surgeon

import copy
import timm
from foa_models.vpt import PromptViT
from foa_models.vpt_cifar import PromptViTForImageClassification
from transformers import ViTForImageClassification, AutoModelForImageClassification
from dataset.selectedRotateImageFolder import prepare_test_data
from utils.utils import get_device
import wandb
import torch
import math
from dataset.ImageNetMask import imagenet_r_mask

def init_data_loaders(args):
    adapt_dataset, adapt_loader, val_dataset, val_loader = prepare_test_data(args)
    return adapt_loader, val_loader

def obtain_train_loader(args):
    tmp = copy.deepcopy(args.corruption)
    tmp_samples = copy.deepcopy(args.adapt_num_samples)
    args.corruption = "original"
    args.adapt_num_samples = -1
    train_dataset, train_loader, _, _ = prepare_test_data(args)
    args.corruption = tmp
    args.adapt_num_samples = tmp_samples
    return train_dataset, train_loader

all_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def init_corruptions_dataset(args):
    if args.cifar_10:
        dataset_name = 'cifar10'
    elif args.corruption == 'v2':
        dataset_name = 'imagenet_v2'
    elif args.corruption == 'sketch':
        dataset_name = 'sketch'
    elif args.corruption == 'rendition':
        dataset_name = 'imagenet_r'
    else:
        dataset_name = 'imagenet'
    
    if args.corruption == 'all':
        corruptions = all_corruptions
    else:
        corruptions = [args.corruption]
    
    if args.corruption == 'rendition':
        num_classes = 200
    elif args.cifar_10:
        num_classes = 10
    else:
        num_classes = 1000

    return corruptions, dataset_name, num_classes

def init_model(args):
    if args.cifar_10:
        if args.vit_type == "small":
            net = ViTForImageClassification.from_pretrained('MF21377197/vit-small-patch16-224-finetuned-Cifar10')
        elif args.vit_type == "base":
            net = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
        elif args.vit_type == "large":
            net = ViTForImageClassification.from_pretrained('tzhao3/vit-L-CIFAR10')
    else:
        if args.vit_type == "small":
            net = timm.create_model('vit_small_patch16_224', pretrained=True)
        elif args.vit_type == "base":
            net = timm.create_model('vit_base_patch16_224', pretrained=True)
        elif args.vit_type == "large":
            net = timm.create_model('vit_large_patch16_224', pretrained=True)
    
    net = net.to(get_device())
    net.eval()
    net.requires_grad_(False)

    return net

def init_adapt_model(args, net):
    if args.algorithm == 'tent':
        net = tent.configure_model(net)
        params, _ = tent.collect_params(net)
        tent_lr = 0.00025
        optimizer = torch.optim.Adam(params, tent_lr)
        adapt_model = tent.Tent(net, optimizer)
    elif args.algorithm == 'foa':
        if args.cifar_10:
            net = PromptViTForImageClassification(net, args.num_prompts).to(get_device())
        else:
            net = PromptViT(net, args.num_prompts).to(get_device())
        adapt_model = FOA(net, args.fitness_lambda, population_size=2).to(get_device())
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 't3a':
        adapt_model = T3A(net, args.num_classes, 20).to(get_device())
    elif args.algorithm == 'sar':
        net = sar.configure_model(net)
        params, _ = sar.collect_params(net)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=0.001, momentum=0.9)
        args.margin_e0 = 0.4 * math.log(args.num_classes)
        adapt_model = sar.SAR(net, optimizer, margin_e0=args.margin_e0)
    elif args.algorithm == 'cotta':
        net = cotta.configure_model(net)
        params, _ = cotta.collect_params(net)
        if args.cifar_10:
            optimizer = torch.optim.Adam(params, lr=0.001)
        else:
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)
        adapt_model = cotta.CoTTA(net, optimizer, steps=1, episodic=False)
    elif args.algorithm == 'lame':
        adapt_model = LAME(net)
    elif args.algorithm == 'no_adapt':
        adapt_model = NoAdapt(net)
    elif args.algorithm == "neo":
        adapt_model = NEO(net, args.num_classes).to(get_device())
        if args.corrupt_center_path != '':
            center = torch.load(args.corrupt_center_path, map_location=get_device())
            adapt_model.set_corrupt_center(center)
    elif args.algorithm == "neo_cont":
        adapt_model = NEO_Cont(net, args.num_classes, args.learning_rate).to(get_device())
    elif args.algorithm == "surgeon":
        adapt_model = Surgeon(net, args.num_classes)
    else:
        assert False, NotImplementedError

    if args.corruption == 'rendition':
        adapt_model.imagenet_mask = imagenet_r_mask
    else:
        adapt_model.imagenet_mask = None

    return adapt_model

