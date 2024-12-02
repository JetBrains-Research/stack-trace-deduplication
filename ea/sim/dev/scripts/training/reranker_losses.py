import torch
from torch import nn

from ea.sim.main.utils import device

class RerankerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def temp(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        raise NotImplementedError

class BCELoss(RerankerLoss):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCELoss()
        self._eps = 1e-6

    def forward(self, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        positives = (positives + 1) / 2
        negatives = (negatives + 1) / 2

        positives = positives.clamp(self._eps, 1 - self._eps)
        negatives = negatives.clamp(self._eps, 1 - self._eps)

        target = torch.tensor([1.] * len(positives)).to(device)
        loss = self._loss(positives, target) + self._loss(negatives, 1 - target)
        return loss

    @property
    def temp(self) -> tuple[torch.Tensor, ...]:
        return torch.tensor([0], device=device), torch.tensor([0], device=device)

    @property
    def name(self) -> str:
        return "bce"
    

class BCELossWithLogits(RerankerLoss):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss()

    def forward(self, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        target = torch.tensor([1.] * len(positives)).to(device)
        loss = self._loss(positives, target) + self._loss(negatives, 1 - target)
        return loss

    @property
    def temp(self) -> tuple[torch.Tensor, ...]:
        return torch.tensor([0], device=device), torch.tensor([0], device=device)
    
    @property
    def name(self) -> str:
        return "bce_with_logits"


class RankNetRerankerLoss(RerankerLoss):
    def __init__(self, t=1):
        super().__init__()
        self._t = 1
    
    def forward(self, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        # - log (σ (t · (sp − sn)))
        return -torch.log(torch.sigmoid(self._t * (positives - negatives))).mean()
    
    @property
    def temp(self) -> tuple[torch.Tensor, ...]:
        return torch.tensor([0], device=device), torch.tensor([0], device=device)

    @property
    def name(self) -> str:
        return "ranknet"


class PairwiseSoftmaxCrossEntropyLoss(RerankerLoss):
    def __init__(self):
        super().__init__()

    def forward(self, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        score_diff = negatives - positives
        loss = torch.log(1 + torch.exp(score_diff)).mean()  
        return loss
    
    @property
    def temp(self) -> tuple[torch.Tensor, ...]:
        return torch.tensor([0], device=device), torch.tensor([0], device=device)
    
    @property
    def name(self) -> str:
        return "pairwise_softmax_ce"