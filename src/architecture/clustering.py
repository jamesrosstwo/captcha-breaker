from enum import Enum, auto

import torch

from architecture.models.kmeans import KMeans
from src.architecture.architecture import CaptchaArchitecture


def fit_kmeans(im_features):
    """
    :param im_features: One feature for each challenge image. <challenge_size, feat_size>
    :return: Returns the unassigned clusters for each image in the challenge.
    """

    kmeans = KMeans(
        n_clusters=2,
        max_iter=100,
        tolerance=1e-4,
        distance="euclidean",
        sub_sampling=None,
    )

    centroids, features = kmeans.fit_predict(im_features)


class _ClusteringMethod(Enum):
    KMEANS = auto()
    # LINEAR = auto()


_clustering_fn_map = {
    _ClusteringMethod.KMEANS: fit_kmeans
}


class CaptchaClustering(CaptchaArchitecture):
    def __init__(self, method: str, **kwargs):
        super().__init__(kwargs)

        valid_keys = list(_ClusteringMethod.__members__.keys())
        if method.upper() not in valid_keys:
            e = "The given method {} for captcha clustering is not implemented. Please select from the available categories {}"
            raise KeyError(e.format(method.upper(), valid_keys))

        self._method: _ClusteringMethod = _ClusteringMethod[method.upper()]
        self._clustering_fn = _clustering_fn_map[self._method]

    def forward(self, questions, challenges):
        img_features = torch.stack([self._backbone(challenge) for challenge in challenges])
        cluster_assignments = self._clustering_fn(img_features)
        return torch.zeros(challenges.shape[:2], device=challenges.device)
