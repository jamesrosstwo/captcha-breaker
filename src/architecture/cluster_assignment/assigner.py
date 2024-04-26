from abc import ABC, abstractmethod

import torch

class CaptchaAssigner(ABC):
    @abstractmethod
    def assign_clusters(self, labels: torch.tensor, centroids: torch.tensor) -> torch.tensor:
        raise NotImplementedError()
