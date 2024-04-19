from torchvision import models

from src.architecture.backbone.backbone import CaptchaBackbone


class Resnet50(CaptchaBackbone):
    def __init__(self):
        super().__init__()
        self._resnet = models.resnet50(pretrained=True)

    def forward(self, x):
        return self._resnet.forward(x)
