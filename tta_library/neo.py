import torch
from timm.models.vision_transformer import VisionTransformer
from dataset.ImageNetMask import imagenet_r_mask
import torch.nn as nn
from utils.utils import get_device

def get_vit_feature_extractor(network):
    # Different ways depending on the ViT implementation
    if hasattr(network, 'head'):
        network.head = nn.Identity()
        if hasattr(network, 'head_dist'):
            network.head_dist = None
        return network
    elif hasattr(network, 'classifier'):
        return lambda x: network.vit(x)[0][:, 0, :]

class NEO(torch.nn.Module):

    def __init__(self, model, num_classes):
        super().__init__()
        if hasattr(model, 'head'):
            self.classifier = model.head
        elif hasattr(model, 'classifier'):
            self.classifier = model.classifier
        self.feature_extractor = get_vit_feature_extractor(model)
        self.adapt_sample_count = 0
        self.corrupt_class_center = torch.zeros(self.classifier.weight.data.shape[1]).to(get_device())

    def forward(self, x, adapt=True):
        z = self.feature_extractor(x)

        if adapt:
            self.adapt_sample_count += z.size(0)
            self.corrupt_class_center = (self.adapt_sample_count - z.size(0))/self.adapt_sample_count * self.corrupt_class_center + z.size(0)/(self.adapt_sample_count) * torch.mean(z, dim=0)
        
        z_centered = z - self.corrupt_class_center
        output = self.classifier(z_centered)
        
        if self.imagenet_mask is not None:  # Only keep predictions for the ImageNet-R classes
            output = output[:, self.imagenet_mask]
        return output
    
    def reset(self):
        """Reset to initial state"""
        self.adapt_sample_count = 0
        self.corrupt_class_center = torch.zeros(self.classifier.weight.data.shape[1]).to(get_device())