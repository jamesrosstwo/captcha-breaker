from abc import ABC
from typing import Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader


def _captcha_collate(data):
    questions = [x[0] for x in data]
    images = torch.stack([x[1] for x in data])
    selections = torch.stack([x[2] for x in data])
    return questions, images, selections

class CaptchaDataset(Dataset, ABC):
    @classmethod
    def from_config(cls, cfg: DictConfig) -> Tuple["CaptchaDataset", "CaptchaDataset"]:
        return instantiate(cfg.train), instantiate(cfg.val)

    def construct_loader(self, *loader_args, **loader_kwargs):
        base_kwargs = dict(
            collate_fn=_captcha_collate
        )

        base_kwargs.update(loader_kwargs)
        return DataLoader(self, *loader_args, **base_kwargs)
