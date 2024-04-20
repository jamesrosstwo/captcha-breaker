import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture.architecture import CaptchaArchitecture
from src.dataset.captcha import CaptchaDataset
from utils.utils import get_device

device = get_device()

class CaptchaExperiment:
    _train_dataset: CaptchaDataset
    _val_dataset: CaptchaDataset

    def __init__(
            self,
            dataset: DictConfig,
            architecture: DictConfig,
            epochs,
            learning_rate,
            loader: DictConfig = None,

    ):
        self._train_dataset, self._val_dataset = CaptchaDataset.from_config(dataset)
        self._architecture: CaptchaArchitecture = instantiate(architecture)
        self._train_epochs = epochs
        self._loader_args: DictConfig = loader
        self._loss_fn = F.binary_cross_entropy
        self._optimizer = torch.optim.Adam(self._architecture.parameters(), learning_rate)

    def _train_epoch(self, epoch: int):
        loader = self._train_dataset.construct_loader(**self._loader_args)
        for batch_idx, (questions, challenges, selections) in enumerate(loader):
            challenges = challenges.to(device)
            selections = selections.to(device)
            preds = self._architecture(questions, challenges)
            loss = self._loss_fn(preds, selections)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

    def _train_architecture(self):
        for epoch in tqdm(range(self._train_epochs)):
            self._train_epoch(epoch)

    def _evaluate_architecture(self):
        pass

    def run(self):
        self._train_architecture()
