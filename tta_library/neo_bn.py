import torch
import torch.nn as nn
from tta_library.neo import NEO

class NEO_BN(NEO):
    def __init__(self, model, num_classes):
        super().__init__(model, num_classes)
        self.configure_bn()

    def configure_bn(self):
        """
        Set Batch Normalization layers to training mode to update running statistics
        using the test batches (BN Adaptation).
        """
        # Ensure the feature extractor is in eval mode first (to handle Dropout etc.)
        self.feature_extractor.eval()
        
        # Set only BN layers to train mode
        for m in self.feature_extractor.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
                # Ensure standard momentum (usually 0.1) is used, or force it if needed.
                # Default is usually fine.
                # m.track_running_stats is usually True.
    
    def forward(self, x, adapt=True):
        # We need to ensure BN layers stay in train mode during forward pass
        # In case something external sets the whole model to eval()
        if adapt:
            self.configure_bn()
            
        return super().forward(x, adapt)
