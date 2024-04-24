import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
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
            visualizer: DictConfig,
            epochs: int,
            learning_rate: float,
            device: str = None,
    ):
        if not device:
            self._device = get_device()
        else:
            self._device = torch.device(device)
        self._train_dataset, self._val_dataset = CaptchaDataset.from_config(dataset)
        self._architecture: CaptchaArchitecture = instantiate(architecture).to(self._device)
        self._n_train_epochs = epochs
        self._loss_fn = F.binary_cross_entropy
        self._optimizer = torch.optim.Adam(self._architecture.parameters(), learning_rate)
        self._out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir).resolve()
        assert self._out_dir.exists() and self._out_dir.is_dir()

        self._eval_dir = self._out_dir / "vis"
        self._eval_dir.mkdir()
        # Initialize wandb
        self._visualizer = instantiate(visualizer, out_path=self._out_dir)

    @property
    def device(self):
        return self._device

    def _train_epoch(self, epoch: int):
        loader = self._train_dataset.construct_loader()
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

        print(f"Epoch {epoch+1}/{self._n_train_epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")

        return avg_loss, avg_accuracy

        # TODO: Return some kind of aggregated loss
        # Or upload directly to wandb?

    def _train_architecture(self):
        for epoch in range(self._n_train_epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            self._evaluate_architecture(self._val_dataset, "validation", "epoch_{}".format(epoch))

    def _evaluate_architecture(self, dataset, file_name: str, eval_name: str):

        print("Evaluating {}".format(dataset))
        loader = dataset.construct_loader()

        evaluation_metrics = defaultdict(list)
        for batch_idx, (questions, challenges, gt_selections) in tqdm(enumerate(loader)):
            challenges = challenges.to(device)
            gt_selections = gt_selections.to(device)
            preds = self._architecture(questions, challenges)
            selections = self._architecture.get_selections(preds)
            correct = selections == gt_selections
            evaluation_metrics["accuracy"].append((correct.sum() / selections.numel()).detach().cpu().item())
            evaluation_metrics["loss"].append(self._loss_fn(preds, gt_selections).detach().cpu().item())

        metrics_path = self._eval_dir / "{}.json".format(file_name)

        results = []
        if metrics_path.exists():
            with open(metrics_path, "r") as metrics_file:
                results = json.load(metrics_file)

        agg_eval = {eval_name: {"mean_{}".format(k): np.mean(v) for k, v in evaluation_metrics.items()}}
        results.append(agg_eval)
        print(agg_eval)
        with open(metrics_path, "w") as metrics_file:
            json.dump(results, metrics_file)

        return metrics_path

    def run(self):
        if self._architecture.is_trainable:
            self._train_architecture()
        # todo: replace with test set
        self._evaluate_architecture(self._val_dataset, "final_validation", "val_set")

