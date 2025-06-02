"""
Contrastive Language Image Pretraining

References:
1. ViT: https://arxiv.org/pdf/2010.11929
2. CLIP: https://arxiv.org/pdf/2103.00020
3. SigLIP: https://arxiv.org/pdf/2303.15343
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class VisionConfig:
    """Configs for vision transformer and using sigmoid loss on clip model"""

    # Transformer encoder
    hidden_size: int = 768
    feedforward_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    # Vit input
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16

    # Vit output
    num_image_tokens: int = None


class VisionEmbeddings(nn.Module):
    """ViT: https://arxiv.org/pdf/2010.11929"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        # input
        self.image_size = config.image_size
        self.channels = config.num_channels

        # intermediate
        self.patch_size = config.patch_size

        # output
        num_patches_row = self.image_size // self.patch_size
        num_patches_col = self.image_size // self.patch_size
        self.num_patches = num_patches_col * num_patches_row
        self.embed_size = config.hidden_size

        # extract image patch embeddings
        self.patch_embedding = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.embed_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",  # no padding is added
        )

        # patch positional embeddings
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(
            num_embeddings=self.num_positions, embedding_dim=self.embed_size
        )

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand(1, -1),
            persistent=False,
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Get images embeddings from convolution layer and position encoding layer"""
        _, _, height, width = imgs.shape

        # [batch_size, 3, height, width] ==>
        # [batch_size, embed_size, num_patches_row, num_patches_col]
        patch_embeds = self.patch_embedding(imgs)

        # [batch_size, embed_size, num_patches_row, num_patches_col]
        # ==> [batch_size, embed_size, num_patches]
        # ==> [batch_size, num_patches, embed_size]
        embeddings = patch_embeds.flatten(2).transpose(2, 1)

        # [batch_size, num_patches, embed_size] +
        # [batch_size, num_patches, embed_size]
        # ==> [batch_size, num_patches, embed_size]
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class EncoderLayer(nn.Module):
    """Encoder layer of transformer

    Reference:
        Vit: https://arxiv.org/pdf/2010.11929
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = MHAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Standard transformer encoder layer"""
        # [batch_size, num_patches, embed_size]
        residual = hidden_states

        # Norm
        hidden_states = self.layer_norm1(hidden_states)
        # Multi-Head Attention
        hidden_states, _ = self.self_attn(hidden_states)

        # Add
        hidden_states = residual + hidden_states
        residual = hidden_states

        # Norm
        hidden_states = self.layer_norm2(hidden_states)
        # FeedForward
        hidden_states = self.mlp(hidden_states)

        # Add
        hidden_states = residual + hidden_states

        # [batch_size, num_patches, embed_size]
        return hidden_states


class VisionTransformer(nn.Module):
    """ViT: https://arxiv.org/pdf/2010.11929"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = VisionEmbeddings(config)
        self.encoder = TransformerEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Transform images into number of patches"""
        hidden_states = self.embeddings(imgs)
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        last_hidden_state = self.layernorm(last_hidden_state)
        return last_hidden_state


class VisionModel(nn.Module):
    """Model used to extract token embedding for image"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.vit = VisionTransformer(config)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Transform images into number of patches"""
        # [batch_size, channels, height, width] ==>
        # [batch_size, num_patches, embed_size]
        return self.vit(imgs)
