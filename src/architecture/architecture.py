from abc import ABC, abstractmethod

import torch
from torch import nn

from src.architecture.backbone.backbone import CaptchaBackbone


class CaptchaArchitecture(nn.Module, ABC):
    def __init__(self, backbone: CaptchaBackbone, text_backbone: CaptchaBackbone = None, fusion_head: CaptchaBackbone = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backbone = backbone
        self._text_backbone = text_backbone
        self._fusion_head = fusion_head

    @abstractmethod
    def forward(self, questions, challenges):
        """
        Generating predictions with the captcha breaking architecture.
        :param questions: The text for each question in the batch
        :param challenges: The image tensor for each image in each challenge. <batch_size, challenge_size, *img_dims>
        :return: A set of binary predictions for each image in each challenge. <batch_size, challenge_size>
        """
        raise NotImplementedError("Forward pass has not been implemented for {}".format(self.__class__.__name__))
