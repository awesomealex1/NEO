import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from collections import defaultdict
from utils.surgeon import AutoFreezeFC, get_tta_transforms
import numpy as np
import math
import logging
from torch.nn.utils.weight_norm import WeightNorm
from copy import deepcopy
import math

logger = logging.getLogger(__name__)

class TTAMethod(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.episodic = False
        self.dataset_name = "imagenet_c"
        self.steps = 1
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"

        # configure model and optimizer
        self.configure_model()
        self.params, self.param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def forward(self, x, adapt=True):
        if self.episodic:
            self.reset()

        x = x if isinstance(x, list) else [x]

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)

        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        """
        raise NotImplementedError

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        base_params = []
        for param in self.params:
            base_params.append({"params": param, "lr": 1e-5, "betas": (0.9, 0.999), "weight_decay": 0.0})
        return torch.optim.Adam(base_params,
                                lr=1e-5,
                                betas=(0.9, 0.999),
                                weight_decay=0.0)

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @staticmethod
    def copy_model(model):
        return deepcopy(model)

def softmax(weights_dict):
    weights = np.array(list(weights_dict.values()))
    exp_weights = np.exp(weights)
    softmax_weights = exp_weights / np.sum(exp_weights)
    softmax_dict = {k: v for k, v in zip(weights_dict.keys(), softmax_weights)}
    return softmax_dict

def log_norm(weights_dict, epsilon=1e-8):
    weights = np.array(list(weights_dict.values()))
    log_weights = np.log(weights + epsilon)
    min_log = np.min(log_weights)
    max_log = np.max(log_weights)
    log_norm_weights = (log_weights - min_log) / (max_log - min_log)
    log_norm_dict = {k: v for k, v in zip(weights_dict.keys(), log_norm_weights)}
    return log_norm_dict

class Surgeon(TTAMethod):
    def __init__(self, model, num_classes):
        super().__init__(model, num_classes)
        self.base_lr = self.optimizer.param_groups[0]['lr']
        self.betas = self.optimizer.param_groups[0]['betas']
        self.weight_decay = self.optimizer.param_groups[0]['weight_decay']
        self.transforms = get_tta_transforms(self.dataset_name)
        self.eps = 1e-8        
        self.grad_weight = defaultdict(lambda: 0.0)
        self.trainable_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}
        self.tau = 1.0
        self.high_margin = math.log(num_classes) * 0.40

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        1. Get layer-wise Gradient Importance and Memory Importance via an additional forward-backward process.
        2. Calculate layer-wise activation pruning ratios.
        3. Forward and adapt models via dynamic activation sparsity.
        """

        # 1-Setting the state of the additional forward-backward process
        self.Activation_Sparsity_Deactivate(self.model)
        # 1-An additional forward-backward process with random sampling
        imgs = x[0]
        random_indices = torch.randperm(imgs.size()[0])[:10] # Randomly select 10 samples for gradient importance calculation
        logits = self.model(imgs[random_indices])
        if hasattr(logits, "logits"):
            logits = logits.logits
        if self.imagenet_mask is not None:
            logits = logits[:, self.imagenet_mask]  
        loss = softmax_entropy(logits)
        loss.backward(retain_graph=True)
        # 1-Get layer-wise Gradient Importance and Memory Importance
        layer_names = [n for n, param in self.model.named_parameters() if param.grad is not None]
        grad_xent = [param.grad for param in self.model.parameters() if param.grad is not None]
        param_xent = [param for param in self.model.parameters() if param.grad is not None]
        memories = self.memory_layer_cal(self.model)
        metrics = defaultdict(list)
        average_metrics = defaultdict(float)
        xent_grads = []
        xent_grads.append([g.detach() for g in grad_xent])
        for xent_grad in xent_grads:
            xent_grad_metrics = get_tgi(param_xent, xent_grad, layer_names, memories) # Combination of Importance Metrics
            for k, v in xent_grad_metrics.items():
                metrics[k].append(v)
        for k, v in metrics.items():
            average_metrics[k] = np.array(v).mean(0)

        # 2-Calculate layer-wise activation pruning ratios
        weights = average_metrics
        lr_weights_standard = defaultdict(type(weights.default_factory))
        lr_weights_standard.update(weights)
        max_weight = max(lr_weights_standard.values())
        for k, v in weights.items():
            lr_weights_standard[k] = v / max_weight
        del layer_names, grad_xent, param_xent, metrics, average_metrics, xent_grads, weights
        
        # 3-Setting the state of the normal forward-adapt process
        self.Activation_Sparsity_Activate(self.model, lr_weights_standard)
        # 3-Forward propagation
        logits = self.model(imgs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        if self.imagenet_mask is not None:
            logits = logits[:, self.imagenet_mask]  
        logits_aug = self.model(self.transforms(imgs))
        if hasattr(logits_aug, "logits"):
            logits_aug = logits_aug.logits
        if self.imagenet_mask is not None:
            logits_aug = logits_aug[:, self.imagenet_mask]  
        self.optimizer.zero_grad()
        # 3-Loss calculation
        # Original loss
        # loss = softmax_entropy(logits).mean(0)
        entropys = softmax_entropy_sample(logits)
        filter_ids_1 = torch.where(entropys < self.high_margin)
        entropys = entropys[filter_ids_1]
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.high_margin))
        entropys = entropys.mul(coeff)
        # Loss with Certainty-based Sample Selection (CSS) and Consistency Regularization (CR)
        loss = entropys.mean(0) + 0.01*logits.shape[1]*consistency(logits, logits_aug)
        # 3-Model Adaptation
        loss.backward()
        self.optimizer.step()

        return logits

    def memory_layer_cal(self, net):
        layer_memories = defaultdict(list)
        memory_sum = 0
        for mod_name, target_mod in net.named_modules():
            mod_name = mod_name + ".weight"
            if isinstance(target_mod, nn.Linear):
                layer_memories[mod_name] = target_mod.activation_size
                memory_sum = memory_sum + target_mod.activation_size
        return [layer_memories, memory_sum]

    def Activation_Sparsity_Activate(self, net, lr_weights_standard):
        """Setting the state of the normal forward-adapt process."""
        for mod_name, target_mod in net.named_modules():
            mod_name = mod_name + ".weight"
            if isinstance(target_mod, nn.Linear): # Setting FC
                target_mod.sparsity_signal = True
                if mod_name in lr_weights_standard:
                    target_mod.clip_ratio = 1 - lr_weights_standard[mod_name]
                else:
                    target_mod.clip_ratio = 1

    def Activation_Sparsity_Deactivate(self, net):
        """Setting the state of the additional forward-backward process."""
        for _, target_mod in net.named_modules():
            if isinstance(target_mod, nn.Linear):  # Setting FC
                target_mod.sparsity_signal = False

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with das. load → (train_conf → optimizer)."""
        self.model = self.Auto_freeze_module(self.model)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()
        self.model.requires_grad_(True)
        self.params_origin = self.origin_params_collect(self.model)

    def Auto_freeze_module(self, net):
        """Load Autofreeze modules."""
        # Replace FC as AutoFC
        f_replaced = self.replace_fc(net, "Standard", False)
        f_fc = self.count_fc(net)
        if f_replaced != f_fc:
            print(f"Replaced {f_replaced} FCs but actually have {f_fc}. Need to update `Auto_freeze_module`.")
        mf_cnt = 0
        for m in net.modules():
            if isinstance(m, AutoFreezeFC):
                mf_cnt += 1
        
        assert f_replaced == mf_cnt, f"Replaced {f_replaced} FCs but actually inserted {mf_cnt} AutoFreezeFC."

        print("Successfully insert %d AutoFreeze_FC layers,", f_fc)

        return net

    def replace_fc(self, model, name, BN_only, f_replaced=0):
        copy_keys = []

        for mod_name, target_mod in model.named_children():
            if isinstance(target_mod, nn.Linear):
                print(f" Insert AutoFreeze-FC to ", name + '.' + mod_name)
                f_replaced += 1

                new_mod = AutoFreezeFC(
                    target_mod.in_features,
                    target_mod.out_features,
                    target_mod.bias,
                    **{k: getattr(target_mod, k) for k in copy_keys},
                    name=f'{name}.{mod_name}',
                    num=f_replaced,
                    BN_only=BN_only)
                new_mod.load_state_dict(target_mod.state_dict())
                setattr(model, mod_name, new_mod)
            else:
                f_replaced = self.replace_fc(
                    target_mod, name + '.' + mod_name, BN_only, f_replaced=f_replaced)
        return f_replaced

    def count_fc(self, model: nn.Module):
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                cnt += 1
        return cnt

    @staticmethod
    def check_model(model):
        """Check model for compatability with law."""
        is_training = model.training
        assert is_training, "law needs train mode: call model.train()"

    def origin_params_collect(self, model):
        params_origin = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                params_origin[name] = param.data.clone().detach()
        return params_origin

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()

@torch.jit.script
def softmax_entropy_sample(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """Consistency loss between two softmax distributions."""
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()

def get_grad_norms(params, grads, layer_names):
    _metrics = defaultdict(list)
    for (layer_name, param, grad) in zip(layer_names, params, grads):
        _metrics[layer_name] = torch.norm(grad).item() / torch.norm(param).item()
    return _metrics

def get_tgi(params, grads, layer_names, memories):
    """Combination of Gradient Importance and Memory Importance."""
    layer_memories = memories[0].values()
    memory_sum = memories[1]
    _metrics = defaultdict(list)
    for (layer_name, param, grad, memory) in zip(layer_names, params, grads, layer_memories):
        grad_norm = torch.norm(grad).item()
        _metrics[layer_name] = grad_norm / grad.numel()**0.5 * math.log(memory_sum / memory)
    return _metrics
