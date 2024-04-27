from abc import ABC, abstractmethod
from typing import Tuple

import torch


class CaptchaClusterer(ABC):
    @abstractmethod
    def fit_predict(self, challenge_features: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        :param im_features: One feature for each challenge image. <challenge_size, feat_size>
        :return: Returns the unassigned labels and centroids for each image in the challenge.
        """
        raise NotImplementedError()