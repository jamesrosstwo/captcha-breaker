import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch.nn.functional as F
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
            epochs: int,
            learning_rate: float,
            device: str = None,
            loader: DictConfig = None,
    ):
        if not device:
            self._device = get_device()
        else:
            self._device = torch.device(device)
        self._train_dataset, self._val_dataset = CaptchaDataset.from_config(dataset)
        self._architecture: CaptchaArchitecture = instantiate(architecture).to(self._device)
        self._train_epochs = epochs
        self._loader_args: DictConfig = loader
        self._loss_fn = F.binary_cross_entropy
        self._optimizer = torch.optim.Adam(self._architecture.parameters(), learning_rate)
        # Initialize wandb

    @property
    def device(self):
        return self._device

    def _train_epoch(self, epoch: int):
        loader = self._train_dataset.construct_loader(**self._loader_args)
        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        for batch_idx, (questions, challenges, selections) in tqdm(enumerate(loader)):
            challenges = challenges.to(device)
            selections = selections.to(device)
            preds = self._architecture(questions, challenges)
            loss = self._loss_fn(preds, selections)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            total_loss += loss.item() * selections.size(0)
            total_accuracy += ((preds.round() == selections).sum().item())
            total_samples += selections.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples

        print(f"Epoch {epoch+1}/{self._train_epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")

        return avg_loss, avg_accuracy

        # TODO: Return some kind of aggregated loss
        # Or upload directly to wandb?

    def _train_architecture(self):
        for epoch in range(self._train_epochs):
            print("Training epoch {}".format(epoch))
            train_loss, train_accuracy = self._train_epoch(epoch)
            
            # TODO: Log train_loss and train_accuracy to wandb
            val_loss, val_accuracy = self._evaluate_architecture()


    def _evaluate_architecture(self):
        loader = self._val_dataset.construct_loader(**self._loader_args)
        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        self._architecture.eval()
        with torch.no_grad():
            for batch_idx, (questions, challenges, selections) in tqdm(enumerate(loader)):
                challenges = challenges.to(device)
                selections = selections.to(device)

                preds = self._architecture(questions, challenges)
                loss = self._loss_fn(preds, selections)

                total_loss += loss.item() * selections.size(0)
                total_accuracy += ((preds.round() == selections).sum().item())
                total_samples += selections.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples

        print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")

        return avg_loss, avg_accuracy

    def run(self):
        self._train_architecture()
