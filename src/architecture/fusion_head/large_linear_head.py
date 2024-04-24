import ipdb
import torch
import torch.nn as nn
from utils.utils import get_device
from src.architecture.backbone.backbone import CaptchaBackbone

device = get_device()

class LargeLinearFusionHeadModel(nn.Module):
    def __init__(self):
        super(LargeLinearFusionHeadModel, self).__init__()
        
        self.sequential = nn.Sequential(
            nn.Linear(448+768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequential(x)

class LargeLinearFusionHead(CaptchaBackbone):
    def __init__(self):
        super().__init__()
        self.layernorm = nn.LayerNorm(448)
        self.lin_model = LargeLinearFusionHeadModel().to(device)

    def forward(self, x):
        im_embed, text_embed = x
        im_embed = torch.mean(im_embed, dim=1)  # collapse channels
        im_embed = self.layernorm(im_embed)
        x = torch.cat([im_embed, text_embed], dim=1)
        return self.lin_model(x)
