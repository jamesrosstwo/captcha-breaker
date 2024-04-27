import torch
from src.architecture.architecture import CaptchaArchitecture


class CaptchaRandom(CaptchaArchitecture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def is_trainable(self):
        return False

    @property
    def name(self) -> str:
        return "Random"

    def forward(self, questions, challenges):
        return torch.rand((len(questions), 1), device=challenges.device)
