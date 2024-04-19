from src.architecture.backbone.backbone import CaptchaBackbone


class TinyViT(CaptchaBackbone):
    def __init__(self):
        # TODO: wrap the model here
        self._vit = None

    def forward(self, x):
        return self._vit(x)
