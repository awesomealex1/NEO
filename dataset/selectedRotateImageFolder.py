import os
import random
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .ImagenetV2 import ImageNetV2Dataset
from transformers import ViTImageProcessor
import numpy as np
from torch.utils.data import Subset
from collections import defaultdict


# ========================= Global Transforms =========================

# NOTE: The global `normalize` is used for ImageNet but overridden by the processor for CIFAR-10.
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet Stats
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes to [-1, 1]

tr_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor(),
    normalize
])

te_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

te_transforms_imageC = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

rotation_tr_transforms = tr_transforms
rotation_te_transforms = te_transforms

common_corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression'
]

# ========================= Helper Functions for Rotation =========================

def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)

def rotate_single_with_label(img, label):
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    return img

# ========================= Custom Dataset Class =========================

class SelectedRotateImageFolder(datasets.ImageFolder):
    def __init__(self, root, train_transform, original=True, rotation=True, rotation_transform=None, processor=None):
        super(SelectedRotateImageFolder, self).__init__(root, train_transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform
        self.original_samples = self.samples
        self.processor = processor
        random.shuffle(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_input = self.loader(path)

        if self.processor is not None:
            # The processor returns a dict, we need the pixel_values, and remove the batch dim.
            processed_output = self.processor(images=img_input, return_tensors="pt")
            img = processed_output["pixel_values"][0]
        elif self.transform is not None:
            img = self.transform(img_input)
        else:
            img = img_input # Fallback if no processor or transform

        results = []
        if self.original:
            results.append(img)
            results.append(target)
        if self.rotation:
            img_for_rotation = self.rotation_transform(img_input) if self.rotation_transform else img
            target_ssh = np.random.randint(0, 4, 1)[0]
            img_ssh = rotate_single_with_label(img_for_rotation, target_ssh)
            results.append(img_ssh)
            results.append(target_ssh)
        return results

    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation

    def set_target_class_dataset(self, target_class_index, logger=None):
        self.target_class_index = target_class_index
        self.samples = [(path, idx) for (path, idx) in self.original_samples if idx in self.target_class_index]
        self.targets = [s[1] for s in self.samples]

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = [self.samples[i] for i in indices[:subset_size]]
        self.targets = [self.targets[i] for i in indices[:subset_size]]
        return len(self.targets)

    def set_specific_subset(self, indices):
        self.samples = [self.original_samples[i] for i in indices]
        self.targets = [s[1] for s in self.samples]

# ========================= Data Loading Helper Functions =========================

def _get_test_transforms(corruption_type: str, use_transforms: bool):
    """Determines the appropriate transforms based on the corruption type."""
    if not use_transforms:
        return None
    if corruption_type in common_corruptions:
        return te_transforms_imageC
    elif corruption_type in ['original', 'rendition', 'v2', 'sketch']:
        return te_transforms
    else:
        raise NotImplementedError(f"Transforms for corruption '{corruption_type}' not defined.")

def _create_dataset(args, transforms_to_apply, train=False):
    """Factory function to create the appropriate test dataset."""
    corruption = args.corruption
    
    # Handle CIFAR-10 datasets
    if args.cifar_10:
        print(f'Loading CIFAR-10 data for corruption: \033[94m{corruption}\033[0m, level: {args.level}')
        if corruption == 'original':
            path = os.path.join(args.cifar_10_original)
        else:
            base_path = '../foa-original/FOA/data/cifar-10-c-train' if train else args.data_cifar_10_c
            path = os.path.join(base_path, corruption, str(args.level))
        
        # 1. Load a generic processor, e.g., for ViT-Base. Use the model you trained with.
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        dataset = SelectedRotateImageFolder(path, None, original=True, rotation=False, processor=processor)
        return dataset

    # Handle ImageNet and its variants
    if corruption == 'original':
        path = os.path.join(args.data, 'val')
        dataset = SelectedRotateImageFolder(path, transforms_to_apply, original=True, rotation=False)
    elif corruption in common_corruptions:
        print(f'Loading ImageNet-C data for corruption: {corruption}, level: {args.level}')
        path = os.path.join(args.data_corruption, corruption, str(args.level))
        dataset = SelectedRotateImageFolder(path, transforms_to_apply, original=True, rotation=False)
    elif corruption == 'rendition':
        dataset = datasets.ImageFolder(args.data_rendition, transforms_to_apply)
    elif corruption == 'sketch':
        dataset = datasets.ImageFolder(args.data_sketch, transforms_to_apply)
    elif corruption == 'v2':
        dataset = ImageNetV2Dataset(transform=transforms_to_apply, location=args.data_v2)
    else:
        raise ValueError(f"Unknown corruption type: '{corruption}'")
        
    return dataset

def _split_dataset_for_adaptation(dataset, num_adapt_classes, seed):
    """Splits a dataset into adaptation and validation subsets by class."""
    if hasattr(dataset, 'targets'):
        all_targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        all_targets = dataset.labels
    else:
        print("Extracting labels by iterating through the dataset...")
        all_targets = [label for _, label in dataset]
    
    all_targets = torch.tensor(all_targets)
    
    unique_classes = torch.unique(all_targets).tolist()
    random.seed(seed)
    adapt_classes = set(random.sample(unique_classes, k=num_adapt_classes))
    
    print(f"Splitting dataset: {len(adapt_classes)} classes for adaptation, {len(unique_classes) - len(adapt_classes)} for validation.")
    
    adapt_indices = [i for i, target in enumerate(all_targets) if target.item() in adapt_classes]
    val_indices = [i for i, target in enumerate(all_targets) if target.item() not in adapt_classes]

    print(f"Adaptation samples: {len(adapt_indices)}, Validation samples: {len(val_indices)}")
    
    adapt_subset = torch.utils.data.Subset(dataset, adapt_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    return adapt_subset, val_subset

def _create_dataloader(dataset, args):
    """Creates a DataLoader with standardized parameters."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.if_shuffle,
        num_workers=getattr(args, 'workers', 1),
        pin_memory=True
    )


def prepare_test_data(args, use_transforms=True, semi=False, train=False):
    """
    Prepares and loads the test data.

    If adapt_label_count > 0, it creates an adaptation set of exactly 50 samples,
    drawn as evenly as possible from the specified number of classes.
    Otherwise, it splits randomly based on a fixed number of samples.
    """
    transforms = _get_test_transforms(args.corruption, use_transforms)
    full_dataset = _create_dataset(args, transforms, train)

    # --- NEW LOGIC FOR CLASS-BASED SPLIT (FIXED 50 SAMPLES) ---
    if hasattr(args, 'adapt_label_count') and args.adapt_label_count > 0:
        # --- Configuration ---
        ADAPT_TOTAL_SAMPLES = 50
        label_count = args.adapt_label_count
        
        print(f"Creating adaptation set with {ADAPT_TOTAL_SAMPLES} samples from {label_count} classes.")

        if not hasattr(full_dataset, 'targets'):
            raise AttributeError("The dataset must have a 'targets' attribute for class-based splitting.")
        
        targets = np.array(full_dataset.targets)
        unique_classes = np.unique(targets)

        if label_count > len(unique_classes):
            raise ValueError(f"adapt_label_count ({label_count}) cannot be greater than the number of unique classes ({len(unique_classes)}).")

        # Use a seeded random number generator for reproducibility
        rng = np.random.default_rng()
        
        # 1. Randomly choose which classes to use for adaptation
        adapt_classes = rng.choice(unique_classes, size=label_count, replace=False)
        print(f"Selected adaptation classes: {adapt_classes}")

        # 2. Pre-calculate indices for all samples of the chosen classes
        class_to_indices = defaultdict(list)
        for i, target in enumerate(targets):
            if target in adapt_classes:
                class_to_indices[target].append(i)

        # 3. Determine how many samples to draw from each class
        base_samples = ADAPT_TOTAL_SAMPLES // label_count
        remainder = ADAPT_TOTAL_SAMPLES % label_count
        
        adapt_indices = []
        for i, cls in enumerate(adapt_classes):
            # The first 'remainder' classes get one extra sample
            num_samples_to_draw = base_samples + 1 if i < remainder else base_samples
            
            available_indices = class_to_indices[cls]
            if len(available_indices) < num_samples_to_draw:
                raise ValueError(f"Cannot draw {num_samples_to_draw} samples for class {cls}. "
                                 f"Only {len(available_indices)} available.")
            
            # Randomly select the indices for this class and add to our list
            chosen_indices = rng.choice(available_indices, size=num_samples_to_draw, replace=False)
            adapt_indices.extend(chosen_indices)
        
        # 4. Create the validation set from all other samples
        all_indices_set = set(range(len(full_dataset)))
        adapt_indices_set = set(adapt_indices)
        potential_val_indices = list(all_indices_set - adapt_indices_set)

        # 2. Check if there are enough samples for the validation set
        VAL_TOTAL_SAMPLES = 64*64
        if len(potential_val_indices) < VAL_TOTAL_SAMPLES:
            raise ValueError(f"Not enough samples for validation. "
                             f"Required: {VAL_TOTAL_SAMPLES}, Available: {len(potential_val_indices)}")
        
        # 3. Randomly sample 1000 indices from the potential pool
        val_indices = rng.choice(potential_val_indices, size=VAL_TOTAL_SAMPLES, replace=False)
        
        # Create final dataset subsets
        adapt_dataset = Subset(full_dataset, adapt_indices)
        val_dataset = Subset(full_dataset, sorted(val_indices))

    # --- ORIGINAL LOGIC FOR RANDOM SPLIT ---
    else:
        total_size = len(full_dataset)
        adapt_size = args.adapt_num_samples

        if adapt_size == -1:
            adapt_size = total_size
        
        val_size = total_size - adapt_size
        
        print(f"Splitting dataset of size {total_size} into {adapt_size} for adaptation and {val_size} for validation.")

        generator = torch.Generator().manual_seed(getattr(args, 'seed', 42))
        adapt_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [adapt_size, val_size], generator=generator
        )

    # --- COMMON LOGIC FOR CREATING DATALOADERS ---
    adapt_size = len(adapt_dataset)
    val_size = len(val_dataset)
    print(f"Final split: {adapt_size} adaptation samples, {val_size} validation samples.")

    if adapt_size == 0:
        adapt_loader = None
    else:
        adapt_loader = _create_dataloader(adapt_dataset, args)
    
    if val_size == 0 or not args.eval:
        val_loader = None
    else:
        val_loader = _create_dataloader(val_dataset, args)

    return adapt_dataset, adapt_loader, val_dataset, val_loader