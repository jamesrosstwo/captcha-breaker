from abc import ABC
from pathlib import Path
from typing import Dict

from hydra.utils import instantiate


class Visualizer(ABC):
    def __init__(self):
        pass

    def on_evaluated(self, evaluation_metrics):
        pass


class VisualizationCollection:
    def __init__(self, out_path: Path, **visualizers):
        self._visualizers: Dict[str, Visualizer] = visualizers

    def on_evaluated(self, evaluation_metrics):
        for k, v in self._visualizers.items():
            v.on_evaluated(evaluation_metrics)
