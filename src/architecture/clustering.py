import torch

from src.architecture.cluster_assignment.assigner import CaptchaAssigner
from src.architecture.cluster.clusterer import CaptchaClusterer
from src.architecture.architecture import CaptchaArchitecture


class CaptchaClustering(CaptchaArchitecture):
    def __init__(self, clusterer: CaptchaClusterer, assigner: CaptchaAssigner, **kwargs):
        super().__init__(**kwargs)
        self._clusterer = clusterer
        self._assigner = assigner

    @property
    def is_trainable(self):
        return False

    @property
    def name(self) -> str:
        return str(self._clusterer.__class__.__name__) + str(self._assigner.__class__.__name__)

    def forward(self, questions, challenges):
        img_features = self._backbone(challenges)
        labels, centroids = self._clusterer.fit_predict(img_features)
        return self._assigner.assign_clusters(labels, centroids).unsqueeze(1).to(torch.float)
