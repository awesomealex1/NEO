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

class NEO_Cont(torch.nn.Module):

    def __init__(self, model, num_classes, learning_rate=0.1):
        super().__init__()
        if hasattr(model, 'head'):
            self.classifier = model.head
        elif hasattr(model, 'classifier'):
            self.classifier = model.classifier
        self.feature_extractor = get_vit_feature_extractor(model)
        self.corrupt_class_center = torch.zeros(self.classifier.weight.data.shape[1], device=get_device())
        self.learning_rate = learning_rate

    def forward(self, x, adapt=True):
        z = self.featurizer(x)

        if adapt:
            self.corrupt_class_center = (1 - self.learning_rate) * self.corrupt_class_center + self.learning_rate * torch.mean(z, dim=0) 
        
        z_centered = z - self.corrupt_class_center
        output = self.classifier(z_centered)

        if self.imagenet_mask is not None: # Only keep predictions for the ImageNet-R classes
            output = output[:, self.imagenet_mask]
        return output
    
    def reset(self):
        """Reset to initial state"""
        self.corrupt_class_center = torch.zeros(self.classifier.weight.data.shape[1], device=get_device())