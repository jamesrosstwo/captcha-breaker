import torch.nn as nn
from utils.utils import get_device
from src.architecture.backbone.backbone import CaptchaBackbone

device = get_device()

class CrossAttentionFusionHeadModel(nn.Module):
    def __init__(self):
        super(CrossAttentionFusionHeadModel, self).__init__()


        # Expands the image embeddings to shape of text embeddings
        self.adaptation_layer = nn.Sequential(
            nn.Linear(448, 768),
            nn.ReLU()  # TODO RELU or Tanh?
        )

        self.output = nn.Sequential(
            nn.Linear(768),
            nn.Sigmoid()
        )

        self.c_attention = nn.MultiheadAttention(embed_dim=768, num_heads=2)
        

    def forward(self, embeds):
        # Split the input 'x' into image and text embeddings
        # Assume 'x' is of shape [batch_size, 448 + 768]
        im_embed = embeds[0]  # First 448 dims are image embeddings
        text_embed = embeds[1]  # Next 768 dims are text embeddings

        # Adapt image embeddings to match the dimensionality of text embeddings
        expanded_im = self.adaptation_layer(im_embed)

        # Reshape for attention module: from (batch_size, 768) to (1, batch_size, 768) to add a sequence length dimension
        expanded_im = expanded_im.unsqueeze(0)
        text_embed = text_embed.unsqueeze(0)

        # Text to image cross attention:
        # text as queries, image as keys and values
        attn_output, attn_weights = self.c_attention(query=text_embed, key=expanded_im, value=expanded_im)

        # Remove the sequence length dimension, back to (batch_size, 768)
        attn_output = attn_output.squeeze(0)

        # Pass through the final output layer
        output = self.output(attn_output)

        return output


class CrossAttentionFusionHead(CaptchaBackbone):
    def __init__(self):
        super().__init__()
        self.model = CrossAttentionFusionHeadModel().to(device)

    def forward(self, x):
        return self.model(x)
