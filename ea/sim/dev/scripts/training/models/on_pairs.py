from typing import Any
import lightning as L
import torch
from torch import optim
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy

from ea.sim.dev.scripts.training.losses import PairLoss
from ea.sim.main.methods.neural.encoders.objects import Item
from ea.sim.main.methods.neural.encoders.texts import Encoder
from ea.sim.main.methods.neural.similarity import Similarity


class ModelOnPairs(L.LightningModule):
    def __init__(
            self,
            encoder: Encoder,
            similarity: Similarity,
            loss: PairLoss,
            train_size: int,
            batch_size: int
    ):
        super().__init__()
        # Modules.
        self.encoder = encoder
        self.similarity = similarity
        self.loss = loss

        # Statistics.
        self._seen_tokens = set()

        # Metrics.
        self._acc_metric = BinaryAccuracy()
        self._loss_metric = MeanMetric()
        self._positive_scores_metric = MeanMetric()
        self._negative_scores_metric = MeanMetric()
        self._diff_scores_metric = MeanMetric()

        # For Scheduler.
        self._train_size = train_size
        self._batch_size = batch_size

    def forward(self, xs: list[Item]) -> torch.Tensor:
        return self.encoder(xs)

    def log_train_metrics(self, positive_scores: torch.Tensor, loss: torch.Tensor) -> None:
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/pos_scores", self._positive_scores_metric(positive_scores), prog_bar=False)
        self.log("train/seen_tokens", len(self._seen_tokens), prog_bar=False)
        self.log(f"lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

    def log_val_or_test_metrics(
            self,
            mode: str,
            positive_scores: torch.Tensor,
            negative_scores: torch.Tensor,
            loss: torch.Tensor
    ):
        assert len(positive_scores) == len(negative_scores)
        batch_size = len(positive_scores)
        preds = (positive_scores >= negative_scores)
        target = torch.ones(len(positive_scores), device=positive_scores.device)
        self.log(f"{mode}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{mode}/acc", self._acc_metric(preds, target),
                 prog_bar=True, batch_size=batch_size)
        self.log(f"{mode}/pos_scores", self._positive_scores_metric(positive_scores),
                 prog_bar=False, batch_size=batch_size)
        self.log(f"{mode}/neg_scores", self._negative_scores_metric(negative_scores),
                 prog_bar=False, batch_size=batch_size)
        self.log(f"{mode}/diff_scores", self._diff_scores_metric(positive_scores - negative_scores),
                 prog_bar=False, batch_size=batch_size)

    def update_seen_tokens(self, items: list[Item]):
        for item in items:
            self._seen_tokens.update(item.all_ids)

    def training_step(self, batch: list[tuple[Item, Item]], batch_index: int) -> torch.Tensor:
        # Uses (anchor, positive) pairs for training.
        anchor_items, doc_items = tuple(map(list, zip(*batch)))
        anchor_embeddings = self(anchor_items)
        positive_embeddings = self(doc_items)
        loss = self.loss(anchor_embeddings, positive_embeddings)
        self.update_seen_tokens(anchor_items + doc_items)
        self.log_train_metrics(
            positive_scores=self.similarity(anchor_embeddings, positive_embeddings),
            loss=loss
        )
        return loss

    def validation_or_test_step(self, mode: str, batch: list[tuple[Item, Item, Item]], batch_index: int) -> None:
        anchor_items, positive_items, negative_items = tuple(map(list, zip(*batch)))
        anchor_embeddings = self(anchor_items)
        positive_embeddings = self(positive_items)
        negative_embeddings = self(negative_items)
        loss = self.loss(anchor_embeddings, positive_embeddings)
        self.log_val_or_test_metrics(
            mode=mode,
            positive_scores=self.similarity(anchor_embeddings, positive_embeddings),
            negative_scores=self.similarity(anchor_embeddings, negative_embeddings),
            loss=loss
        )

    def validation_step(self, batch: list[tuple[Item, Item, Item]], batch_idx: int) -> None:
        # Uses (anchor, positive, negative) triplets for validation.
        self.validation_or_test_step("val", batch, batch_idx)

    def test_step(self, batch: list[tuple[Item, Item, Item]], batch_idx: int) -> None:
        # User (anchor, positive, negative) triplets for test.
        self.validation_or_test_step("test", batch, batch_idx)

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-4)
        # scheduler = optim.lr_scheduler.CyclicLR
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=1e-3,
            epochs=10,
            steps_per_epoch=self._train_size // self._batch_size + 1
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

