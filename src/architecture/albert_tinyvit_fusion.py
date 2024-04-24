from src.architecture.architecture import CaptchaArchitecture
from utils.utils import get_device
import torch 

device = get_device()

class AlbertTinyViTFusion(CaptchaArchitecture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Freeze the weights of the backbones
        for param in self._backbone.parameters():
            param.requires_grad = False
        
        for param in self._text_backbone.parameters():
            param.requires_grad = False


    def forward(self, x_text, x_img):
        # Get text and image embeddings using their respective models
        img_embed = self._backbone(x_img)
        text_embed = self._text_backbone(x_text)

        # Concatenate the embeddings
        # If running with challenge_size > 1, text_embeddings will need to be duplicated along the questions
        combined_embed = torch.cat((img_embed, text_embed), dim=1)

        # Forward pass through the fusion head
        out = self._fusion_head(combined_embed)
        return out
