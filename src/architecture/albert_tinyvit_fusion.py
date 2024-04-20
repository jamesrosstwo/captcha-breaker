from src.architecture.architecture import CaptchaArchitecture
from utils.utils import get_device
import torch 

device = get_device()

class AlbertTinyViTFusion(CaptchaArchitecture):
    # def forward(self, questions, challenges):
    #     # TODO: Move the architecture forward pass from AlbertTinyViTFusion.py to here
    #     pass

    def forward(self, x_text, x_img):
        # Get text and image embeddings using their respective models
        img_embed = self._backbone(x_img)
        text_embed = self._text_backbone(x_text)

        # Concatenate the embeddings
        combined_embed = torch.cat((img_embed, text_embed), dim=1)

        # Forward pass through the fusion head
        out = self._fusion_head(combined_embed)
        return out
