class NoAdapt:
    """Wrapper for no adaptation that supports imagenet_mask attribute"""
    def __init__(self, model):
        self.model = model
        self.imagenet_mask = None
    
    def __call__(self, x):
        output = self.model(x)
        # Apply ImageNet-R mask if available
        if self.imagenet_mask is not None:
            # Only keep predictions for the ImageNet-R classes
            output = output[:, self.imagenet_mask]
        return output
    
    def forward(self, x, adapt=False):
        output = self.model(x)
        # Apply ImageNet-R mask if available
        if self.imagenet_mask is not None:
            # Only keep predictions for the ImageNet-R classes
            output = output[:, self.imagenet_mask]
        return output
    
    def reset(self):
        # No-op for no adaptation
        pass