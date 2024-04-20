from src.architecture.backbone.backbone import CaptchaBackbone
from src.architecture.backbone.tiny_vit_model.tiny_vit import tiny_vit_11m_224
from src.utils.utils import get_device

class TinyViTBackbone(CaptchaBackbone):
    def __init__(self):
        super().__init__()
        device = get_device()
        self._vit = tiny_vit_11m_224(pretrained=True).to(device)


    def forward(self, x):
        return self._vit(x)
