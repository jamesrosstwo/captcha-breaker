import torch.nn as nn
from utils.utils import get_device
from src.architecture.backbone.backbone import CaptchaBackbone
import torch.nn.functional as F
import torch

device = get_device()

class CrossAttentionFusionHeadModel(nn.Module):
    def __init__(self):
        super(CrossAttentionFusionHeadModel, self).__init__()
        self.text_projection = nn.Linear(768, 448)
        self.c_attention = nn.MultiheadAttention(embed_dim=448, num_heads=2, batch_first=True)
        self.conv1 = nn.Conv2d(in_channels=448, out_channels=224, kernel_size=3, padding=1)
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, im_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        # Project text embedding to match the number of channels in the image embedding
        projected_text = self.text_projection(text_embed)

        # Reshape projected text to have a spatial structure
        batch_size, channels = projected_text.shape
        height = width = 7
        projected_text = projected_text.view(batch_size, channels, 1, 1).repeat(1, 1, height, width)

        # Reshape image embedding and projected text for attention
        batch_size, seq_length, channels = im_embed.shape
        im_embed_reshaped = im_embed.view(batch_size, seq_length, channels)
        projected_text_reshaped = projected_text.view(batch_size, -1, channels)

        # Apply attention using reshaped projected text as query and reshaped image embedding as key and value
        attn_output, attn_weights = self.c_attention(query=projected_text_reshaped,
                                                    key=im_embed_reshaped,
                                                    value=im_embed_reshaped)

        # Reshape attended image features back to [batch, channels, height, width]
        attn_output = attn_output.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        # Apply convolution and process the output of the attention module
        conv_output = F.relu(self.conv1(attn_output))
        output = self.output_layer(conv_output)
        return output


class CrossAttentionFusionHead(CaptchaBackbone):
    def __init__(self):
        super().__init__()
        self.model = CrossAttentionFusionHeadModel().to(device)

    def forward(self, x):
        image_embeddings, text_embeddings = x
        return self.model(image_embeddings, text_embeddings)