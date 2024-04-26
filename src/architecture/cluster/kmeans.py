from architecture.cluster.clusterer import CaptchaClusterer
from architecture.models.kmeans import KMeans


class CaptchaKMeans(CaptchaClusterer):
    def __init__(self, **kmeans_kwargs):
        self._kmeans_kwargs = kmeans_kwargs

    def fit_predict(self, im_features):
        kmeans = KMeans(**self._kmeans_kwargs)
        return kmeans.fit_predict(im_features)
