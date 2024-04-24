from abc import ABC
from functools import partial
from typing import Tuple, Optional

import ipdb
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader


def _captcha_collate(data, should_flatten: bool):
    questions = [x[0] for x in data]
    images = torch.stack([x[1] for x in data])
    selections = torch.stack([x[2] for x in data])
    # When flattening, assume that the images are evenly distributed across the questions.
    if should_flatten:
        return questions, images.flatten(start_dim=0, end_dim=1), selections.flatten(start_dim=0, end_dim=1)

    return questions, images, selections


class CaptchaDataset(Dataset, ABC):
    def __init__(self, should_flatten: bool = True, loader: Optional[DictConfig] = None):
        self._should_flatten = should_flatten
        self._loader_kwargs = loader if loader else dict()

    @property
    def _collate_fn(self):
        return partial(_captcha_collate, should_flatten=self.should_flatten)

    @property
    def should_flatten(self):
        return self._should_flatten

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Tuple["CaptchaDataset", "CaptchaDataset"]:
        return instantiate(cfg.train), instantiate(cfg.val)

    def construct_loader(self, *loader_args, **loader_kwargs):
        base_kwargs = dict(
            collate_fn=self._collate_fn
        )
        base_kwargs.update(self._loader_kwargs)
        base_kwargs.update(loader_kwargs)
        return DataLoader(self, *loader_args, **base_kwargs)
