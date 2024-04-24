import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import ipdb
import wandb
import hydra
import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch.nn.functional as F
from tqdm import tqdm, trange

from src.architecture.architecture import CaptchaArchitecture
from src.dataset.captcha import CaptchaDataset
from utils.utils import get_device

device = get_device()


class CaptchaExperiment:
    _train_dataset: CaptchaDataset
    _val_dataset: CaptchaDataset
    _test_dataset: CaptchaDataset

    def __init__(
            self,
            dataset: DictConfig,
            architecture: DictConfig,
            visualizer: DictConfig,
            epochs: int,
            learning_rate: float,
            use_wandb: bool,
            device: str = None,
    ):
        if not device:
            self._device = get_device()
        else:
            self._device = torch.device(device)
        self._train_dataset, self._val_dataset, self._test_dataset = CaptchaDataset.from_config(dataset)
        self._architecture: CaptchaArchitecture = instantiate(architecture).to(self._device)
        self._n_train_epochs = epochs
        self._loss_fn = F.binary_cross_entropy
        self._optimizer = torch.optim.Adam(self._architecture.parameters(), learning_rate)
        self.cfg = hydra.core.hydra_config.HydraConfig.get()
        self.use_wandb = use_wandb

        self._out_dir = Path(self.cfg.runtime.output_dir).resolve()
        assert self._out_dir.exists() and self._out_dir.is_dir()
        self.head_name = self.cfg.runtime.choices['architecture/fusion_head']
        self._eval_dir = self._out_dir / "vis"
        self._eval_dir.mkdir()
        
        # Initialize wandb
        self.run_id = ""
        self._visualizer = instantiate(visualizer, out_path=self._out_dir)
        timestamp = self.cfg.runtime.output_dir.split("outputs/")[-1] 
        if self.use_wandb:
            self.run_id = wandb.util.generate_id()
            print(f"creating new run id: {self.run_id}")
            wandb.init(id=self.run_id,
                    project= "CaptchaExperiment", name=self.head_name+f"_{timestamp}",
                    config={
                    "timestamp": timestamp,
                    "epochs": epochs,
                    "lr": learning_rate,
                    "head": self.head_name,
                    }
                    )

    @property
    def device(self):
        return self._device

    def _train_epoch(self, epoch: int):
        loader = self._train_dataset.construct_loader()
        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        for batch_idx, (questions, challenges, selections) in tqdm(enumerate(loader)):
            if batch_idx!=0: 
                continue

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

        if self.use_wandb:
            wandb.log({ "avg_loss": avg_loss,
                            "avg_accuracy": avg_accuracy,
                            "epoch": epoch+1,
                            })
        
        return avg_loss, avg_accuracy

        # TODO: Return some kind of aggregated loss
        # Or upload directly to wandb?

    def _train_architecture(self):
        self.best_avg_val_cc = float('-inf')
        for epoch in trange(self._n_train_epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            self._evaluate_architecture(self._val_dataset, "validation", "epoch_{}".format(epoch), i_epoch=epoch)

    def _save_best_model(self, model, optimizer, path, global_epoch, run_id="", best_acc=None):
        '''Save all model state dict'''
        dict_ = {'global_epoch': global_epoch, 
                    f'net_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'run_id': run_id
                    }
        if best_acc != None:
            dict_['best_acc'] = best_acc
        model_path = f"{path}/checkpt_{global_epoch:06d}.pt"
        torch.save(dict_, model_path)
        print('Saved weights at', model_path)
    
    def _load_model(self, checkpt_path, model, optimizer=None, global_epoch=None, get_run_id=False, best_acc=float("inf")):
        checkpoint = torch.load(checkpt_path)
        if get_run_id:
            return checkpoint['run_id']
        
        model.load_state_dict(checkpoint["net_state_dict"])
        if optimizer != None: # to continue training
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_epoch = checkpoint['global_epoch']
        if "best_acc" in checkpoint:
            best_acc = checkpoint['best_acc']
        print(f"Reloaded weights at {checkpt_path} \n")
        return model, optimizer, global_epoch, best_acc


    def _evaluate_architecture(self, dataset, file_name: str, eval_name: str, i_epoch=None):
        print("Evaluating {}".format(file_name))
        loader = dataset.construct_loader()

        evaluation_metrics = defaultdict(list)
        for batch_idx, (questions, challenges, gt_selections) in tqdm(enumerate(loader)):
            challenges = challenges.to(device)
            gt_selections = gt_selections.to(device)
            preds = self._architecture(questions, challenges)
            selections = self._architecture.get_selections(preds)
            correct = selections == gt_selections
            val_acc = (correct.sum() / selections.numel()).detach().cpu().item()
            val_loss = self._loss_fn(preds, gt_selections).detach().cpu().item()
            evaluation_metrics["accuracy"].append(val_acc)
            evaluation_metrics["loss"].append(val_loss)

        metrics_path = self._eval_dir / "{}.json".format(file_name)
        results = []
        if metrics_path.exists():
            with open(metrics_path, "r") as metrics_file:
                results = json.load(metrics_file)

        agg_eval = {eval_name: {"mean_{}".format(k): np.mean(v) for k, v in evaluation_metrics.items()}}
        results.append(agg_eval)
        print(agg_eval)

        # save best model
        if agg_eval[eval_name]['mean_accuracy'] > self.best_avg_val_cc:
            self.best_avg_val_cc = agg_eval[eval_name]['mean_accuracy']
            self._save_best_model(self._architecture, self._optimizer, self._out_dir, global_epoch=i_epoch, run_id=self.run_id, best_acc=self.best_avg_val_cc)

        with open(metrics_path, "w") as metrics_file:
            json.dump(results, metrics_file)

        return metrics_path

    def run(self):
        if self._architecture.is_trainable:
            self._train_architecture()
        self._evaluate_architecture(self._test_dataset, "test", "Final Evaluation")

