import torch
from timm.models.vision_transformer import VisionTransformer
from dataset.ImageNetMask import imagenet_r_mask
import torch.nn as nn
from utils.utils import get_device

def get_vit_featurer(network: VisionTransformer):
    if hasattr(network, 'head'):
        network.head = nn.Identity()
        if hasattr(network, 'head_dist'):
            network.head_dist = None
        return network
    elif hasattr(network, 'classifier'):
        return lambda x: network.vit(x)[0][:, 0, :]

class T3A(torch.nn.Module):
    """
    Test Time Template Adjustments (T3A) - Optimized Version
    """
    def __init__(self, model: VisionTransformer, num_classes, filter_K):
        super().__init__()
        if hasattr(model, 'head'):
            self.classifier = model.head
        elif hasattr(model, 'classifier'):
            self.classifier = model.classifier
        self.featurizer = get_vit_featurer(model)
        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        if num_classes == 200:  # for rendition
            warmup_prob = warmup_prob[:, imagenet_r_mask]
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        self.filter_K = filter_K
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)
        self.imagenet_mask = imagenet_r_mask if num_classes == 200 else None

    def forward(self, x, adapt=True):
        z = self.featurizer(x)

        # online adaptation
        p = self.classifier(z)
        if self.imagenet_mask is not None:
            p = p[:, self.imagenet_mask]  # Fixed: was using undefined 'outputs'
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)
        
        # Get classes in current batch
        current_batch_classes = torch.unique(yhat.argmax(dim=1))
        
        # prediction
        if adapt:
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        supports, labels = self.select_supports(current_batch_classes)
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)


    def select_supports(self, current_batch_classes=None):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to(get_device())
        else:
            # OPTIMIZATION: Only re-select supports for classes in current batch
            if current_batch_classes is not None:
                classes_to_update = current_batch_classes
            else:
                # Fallback: get classes from the most recent samples (last batch_size samples)
                # Assumes the last samples added are from the current batch
                recent_labels = self.labels[-len(current_batch_classes):] if current_batch_classes is not None else self.labels
                classes_to_update = torch.unique(recent_labels.argmax(dim=1))
            
            # Start with all current indices
            indices = torch.LongTensor(list(range(len(ent_s)))).to(get_device())
            keep_mask = torch.ones(len(indices), dtype=torch.bool, device=indices.device)
            
            # For each class that needs updating, remove old samples and add new top-K
            for class_idx in classes_to_update:
                class_mask = y_hat == class_idx
                if class_mask.sum() > 0:  # Only process if class has samples
                    class_indices = indices[class_mask]
                    class_entropies = ent_s[class_mask]
                    
                    # Remove all samples of this class from keep_mask
                    keep_mask[class_mask] = False
                    
                    # Sort by entropy and take top filter_K
                    _, sorted_indices = torch.sort(class_entropies)
                    selected_indices = class_indices[sorted_indices[:filter_K]]
                    
                    # Add back the selected samples
                    keep_mask[selected_indices] = True
            
            indices = indices[keep_mask]
        
        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)