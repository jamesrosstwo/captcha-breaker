import torch

from architecture.cluster.clusterer import CaptchaClusterer
from architecture.models.kmeans import KMeans


class CaptchaKMeans(CaptchaClusterer):
    def __init__(self, normalize_features: bool = True, **kmeans_kwargs):
        self._normalize_feats = normalize_features
        self._kmeans_kwargs = kmeans_kwargs

    def fit_predict(self, im_features):
        if self._normalize_feats:
            im_features = torch.nn.functional.normalize(im_features, p=2, dim=0)
        kmeans = KMeans(**self._kmeans_kwargs)
        return kmeans.fit_predict(im_features)
