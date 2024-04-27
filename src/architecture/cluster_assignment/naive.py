import torch

from architecture.cluster_assignment.assigner import CaptchaAssigner


class NaiveAssigner(CaptchaAssigner):
    def assign_clusters(self, labels: torch.tensor, centroids: torch.tensor) -> torch.tensor:
        return labels