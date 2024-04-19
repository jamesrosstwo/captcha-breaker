import torch

from src.architecture.architecture import CaptchaArchitecture


class CaptchaClustering(CaptchaArchitecture):
    def forward(self, questions, challenges):
        img_features = self._backbone(challenges.flatten(start_dim=0, end_dim=1))
        return torch.zeros(challenges.shape[:2]).to(challenges.device)
