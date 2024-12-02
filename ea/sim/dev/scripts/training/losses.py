from abc import ABC, abstractmethod

import torch
from pytorch_metric_learning.losses import NTXentLoss, CircleLoss as CircleLossPTM
from torch import nn

from ea.sim.main.methods.neural.similarity import CosineSimilarity
from ea.sim.main.utils import device


class PointwiseLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PairLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        """
        Loss on embedding pairs.
        :param anchors: Anchor embeddings, shape (batch_size, embedding_dim)
        :param positives: Positive embeddings, shape (batch_size, embedding_dim)
        :return: loss
        """
        raise NotImplementedError


class TripletLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """
        Loss on embedding triplets.
        :param anchors: Anchor embeddings, shape (batch_size, embedding_dim)
        :param positives: Positive embeddings, shape (batch_size, embedding_dim)
        :param negatives: Negative embeddings, shape (batch_size, embedding_dim)
        :return: loss
        """
        raise NotImplementedError


class RankNetLoss(TripletLoss):
    def __init__(self, scale: bool = False, start_temp: float = 0, learn_temp: bool = False, reduction: str = "mean"):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self._scale = scale
        self._t = nn.Parameter(torch.tensor([start_temp], device=device, dtype=torch.float32), requires_grad=learn_temp)
        self._similarity = CosineSimilarity()

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        positive_scores = self._similarity(anchors, positives)
        negative_scores = self._similarity(anchors, negatives)
        margins = positive_scores - negative_scores
        if self._scale:
            margins *= self._t.exp()
        target = torch.tensor([1.] * len(positive_scores), device=anchors.device)
        return self._loss(margins.view(-1), target)

    @property
    def temp(self) -> tuple[torch.Tensor, ...]:
        return self._t, self._t.exp()


class CircleLoss(TripletLoss):
    def __init__(self, m: float = 0.1, gamma: float = 1):
        super().__init__()
        self._loss = CircleLossPTM(m=m, gamma=gamma)
        self._similarity = CosineSimilarity()

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        embeddings = torch.cat((anchors, positives, negatives), dim=0)
        positive_labels = torch.zeros(len(anchors) + len(positives), device=anchors.device)
        negative_labels = torch.arange(len(negatives), device=anchors.device)
        labels = torch.cat((positive_labels, negative_labels))
        return self._loss(embeddings, labels)


class InfoNCEPairs(PairLoss):
    # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179#issuecomment-674320695
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self._loss = NTXentLoss(temp)

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        assert anchors.shape == positives.shape
        embeddings = torch.cat((anchors, positives), dim=0)
        batch_size = embeddings.shape[0]
        labels = torch.arange(batch_size // 2, device=anchors.device)
        labels = torch.cat((labels, labels))  # [1, 2, ..., n - 1, n, 1, 2, 3, ..., n - 1, n]
        return self._loss(embeddings, labels)


class InfoNCEOneVersusAll(TripletLoss):
    # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179#issuecomment-674320695
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self._loss = NTXentLoss(temp)

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        assert anchors.shape[1] == positives.shape[1] == negatives.shape[1]
        assert anchors.shape[0] == positives.shape[0] == 1
        assert negatives.shape[0] > 1
        embeddings = torch.cat((anchors, positives, negatives))
        batch_size = embeddings.shape[0]
        labels = torch.arange(batch_size, device=anchors.device)
        labels[1] = labels[0]  # [0, 0, 2, 3, 4, ..., n - 1, n]
        return self._loss(embeddings, labels)


class CircleLossUsingScores(nn.Module):
    def __init__(self, m: float = 0.1, gamma: float = 1):
        super().__init__()
        self._m = m
        self._gamma = gamma
        self._soft_plus = nn.Softplus()

    def forward(self, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for CircleLoss model.
        Implementation from https://github.com/TinyZeaMays/CircleLoss/blob/master/circle_loss.py
        :param positives: 
        :param negatives: 
        :return: loss
        """
        ap = torch.clamp_min(-positives.detach() + 1 + self._m, min=0.)
        an = torch.clamp_min(negatives.detach() + self._m, min=0.)

        delta_p = 1 - self._m
        delta_n = self._m

        logit_p = -ap * (positives - delta_p) * self._gamma
        logit_n = an * (negatives - delta_n) * self._gamma

        loss = self._soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
