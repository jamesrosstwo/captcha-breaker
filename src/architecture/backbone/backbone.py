from abc import ABC, abstractmethod

from torch import nn


class CaptchaBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Forward pass has not been implemented for {}".format(self.__class__.__name__))
