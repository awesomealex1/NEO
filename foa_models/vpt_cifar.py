import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import math
from typing import Optional

class PromptViTForImageClassification(nn.Module):
    """
    A wrapper for a Hugging Face ViTForImageClassification model that incorporates
    learnable prompt tokens into the input sequence for the transformer encoder.
    
    This allows for prompt-based tuning of Vision Transformers.
    """
    def __init__(
        self,
        vit_model: ViTForImageClassification,
        num_prompts: int = 1
    ):
        """
        Initializes the prompt-enabled Vision Transformer.

        Args:
            vit_model (ViTForImageClassification): An instance of the Hugging Face
                ViT model for image classification.
            num_prompts (int): The number of learnable prompt tokens to add.
        """
        super().__init__()
        
        if not isinstance(vit_model, ViTForImageClassification):
            raise TypeError("vit_model must be an instance of transformers.ViTForImageClassification")

        self.vit_model = vit_model
        self.num_prompts = num_prompts
        self.prompt_dim = vit_model.config.hidden_size

        if self.num_prompts > 0:
            # Initialize the prompt parameters
            self._initialize_prompts()

    @property
    def vit(self):
        """
        A property to directly access the underlying ViTModel, but with an
        added 'head' attribute that points to the main model's classifier.
        This provides compatibility with code that expects a 'model.vit.head' structure.
        """
        # The main transformer block
        vit_module = self.vit_model.vit
        # Dynamically attach the classifier from the parent ViTForImageClassification model
        # to mimic the structure of a timm ViT, where the head is part of the main block.
        vit_module.head = self.vit_model.classifier
        return vit_module

    def _initialize_prompts(self):
        """
        Initializes the prompt embeddings using Xavier uniform initialization,
        matching the logic from the original implementation.
        """
        # In Hugging Face, config.patch_size is an integer for square patches
        patch_size = self.vit_model.config.patch_size
        num_patch_pixels = patch_size * patch_size
        
        # Calculation from the paper "Visual Prompt Tuning" (https://arxiv.org/abs/2203.12119)
        # val = sqrt(6. / ( (num_channels * patch_size^2) + hidden_dim ))
        val = math.sqrt(6. / float(3 * num_patch_pixels + self.prompt_dim))
        
        # Create and initialize the prompts parameter
        self.prompts = nn.Parameter(torch.zeros(1, self.num_prompts, self.prompt_dim))
        nn.init.uniform_(self.prompts.data, -val, val)

    def reset(self):
        """
        Allows for re-initializing the prompts if needed during training or experimentation.
        """
        if self.num_prompts > 0:
            self._initialize_prompts()
            print("Prompt parameters have been reset.")

    def _inject_prompts(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Injects the learnable prompts into the sequence of embeddings.
        The prompts are inserted after the [CLS] token.

        Args:
            embeddings (torch.Tensor): The patch and [CLS] token embeddings from the ViT
                                       embedding layer. Shape: (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Embeddings with prompts inserted.
                          Shape: (batch_size, sequence_length + num_prompts, hidden_size).
        """
        if self.num_prompts <= 0:
            return embeddings

        # The [CLS] token is always at the beginning of the sequence
        cls_token_embedding = embeddings[:, :1, :]
        patch_embeddings = embeddings[:, 1:, :]
        
        # Expand prompts to match the current batch size
        prompts_expanded = self.prompts.expand(embeddings.shape[0], -1, -1)
        
        # Concatenate in the order: [CLS], [PROMPTS], [PATCHES]
        embeddings_with_prompts = torch.cat(
            (cls_token_embedding, prompts_expanded, patch_embeddings),
            dim=1
        )
        return embeddings_with_prompts

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Performs the forward pass with prompt injection.
        The method signature is compatible with the original ViTForImageClassification,
        allowing for seamless integration with the Hugging Face Trainer and other tools.
        """
        # Use the return_dict behavior from the model's config
        return_dict = return_dict if return_dict is not None else self.vit_model.config.use_return_dict
        
        # 1. Get initial embeddings from the ViT model.
        # This module handles patch extraction, adding the [CLS] token, and position embeddings.
        embeddings = self.vit_model.vit.embeddings(pixel_values)
        
        # 2. Inject the learnable prompts into the embedding sequence. This is our modification.
        embeddings_with_prompts = self._inject_prompts(embeddings)
        
        # 3. Pass the modified embeddings through the transformer encoder.
        encoder_outputs = self.vit_model.vit.encoder(
            embeddings_with_prompts,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 4. Extract the final hidden state of the sequence.
        sequence_output = encoder_outputs[0]
        
        # 5. Apply LayerNorm to the [CLS] token's representation.
        # This follows the standard ViT architecture.
        cls_token_representation = self.vit_model.vit.layernorm(sequence_output[:, 0, :])
        
        # 6. Pass the [CLS] token representation through the classifier head.
        logits = self.vit_model.classifier(cls_token_representation)

        # 7. Calculate loss if labels are provided.
        loss = None
        if labels is not None:
            # The model's config holds information about the problem type (e.g., multi-class)
            if self.vit_model.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.vit_model.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.vit_model.config.num_labels), labels.view(-1))
            elif self.vit_model.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 8. Return a standard SequenceClassifierOutput object, making it compatible with HF tools.
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # --- ADDED METHODS TO FIX THE ATTRIBUTEERROR ---

    def _collect_cls_features_per_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        A helper function to iterate through the transformer encoder layers and
        collect the [CLS] token's representation from each one.
        
        Args:
            hidden_states (torch.Tensor): The initial embeddings (including [CLS] token
                                          and potentially prompts) to be fed into the encoder.

        Returns:
            torch.Tensor: A concatenated tensor of [CLS] token features from all layers.
        """
        all_cls_features = []
        # The encoder in Hugging Face ViT has a 'layer' attribute which is a ModuleList
        for layer_module in self.vit_model.vit.encoder.layer:
            # The output of a ViTLayer is a tuple, where the first element is the hidden states
            hidden_states = layer_module(hidden_states)[0]
            # The [CLS] token is the first token in the sequence
            cls_feature = hidden_states[:, 0]
            all_cls_features.append(cls_feature)

        # In the original ViT, a final LayerNorm is applied to the CLS token before the classifier.
        # We apply it to the last collected feature to be consistent.
        if all_cls_features:
            last_cls_feature_normed = self.vit_model.vit.layernorm(all_cls_features[-1])
            all_cls_features[-1] = last_cls_feature_normed

        # Concatenate features from all layers along the feature dimension
        return torch.cat(all_cls_features, dim=1)

    def layers_cls_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extracts the [CLS] token features from each transformer layer, without using prompts.
        This method is restored from the original timm-based implementation.
        
        Args:
            pixel_values (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: Concatenated [CLS] token features from all layers.
        """
        # Get patch and position embeddings, including the [CLS] token
        embeddings = self.vit_model.vit.embeddings(pixel_values)
        return self._collect_cls_features_per_layer(embeddings)

    def layers_cls_features_with_prompts(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extracts the [CLS] token features from each transformer layer after injecting prompts.
        This method is restored from the original timm-based implementation.
        
        Args:
            pixel_values (torch.Tensor): The input image tensor.
            
        Returns:
            torch.Tensor: Concatenated [CLS] token features from all layers.
        """
        # Get patch and position embeddings, including the [CLS] token
        embeddings = self.vit_model.vit.embeddings(pixel_values)
        # Inject the learnable prompts
        embeddings_with_prompts = self._inject_prompts(embeddings)
        return self._collect_cls_features_per_layer(embeddings_with_prompts)
