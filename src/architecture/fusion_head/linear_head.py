import torch.nn as nn
from utils.utils import get_device
from src.architecture.backbone.backbone import CaptchaBackbone

device = get_device()

class LinearFusionHeadModel(nn.Module):
    def __init__(self):
        super(LinearFusionHeadModel, self).__init__()

        # TODO experiment with this
        self.sequential = nn.Sequential(
            nn.Linear(448+768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequential(x)

class LinearFusionHead(CaptchaBackbone):
    def __init__(self):
        super().__init__()
        self.lin_model = LinearFusionHeadModel().to(device)

    def forward(self, x):
        return self.lin_model(x)
