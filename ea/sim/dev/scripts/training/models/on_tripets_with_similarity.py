import lightning as L
import torch
from torch import optim
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy

from ea.sim.dev.scripts.training.losses import TripletLoss
from ea.sim.main.methods.neural.encoders.objects import Item
from ea.sim.main.methods.neural.encoders.texts import Encoder
from ea.sim.main.methods.neural.similarity import Similarity


class ModelOnTripletsWithSimilarity(L.LightningModule):
    def __init__(self, encoder: Encoder, similarity: Similarity, loss):
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

    def forward(self, xs: list[Item]) -> torch.Tensor:
        return self.encoder(xs)

    def log_train_or_val_or_test_metrics(
            self,
            mode: str,
            positive_scores: torch.Tensor,
            negative_scores: torch.Tensor,
            loss: torch.Tensor
    ):
        preds = (positive_scores >= negative_scores)
        batch_size = len(positive_scores)
        target = torch.ones(len(positive_scores), device=positive_scores.device)
        self.log(f"{mode}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{mode}/acc", self._acc_metric(preds, target), prog_bar=True, batch_size=batch_size)
        self.log(f"{mode}/pos_scores", self._positive_scores_metric(positive_scores), prog_bar=False, batch_size=batch_size)
        self.log(f"{mode}/neg_scores", self._negative_scores_metric(negative_scores), prog_bar=False, batch_size=batch_size)
        self.log(f"{mode}/diff_scores", self._diff_scores_metric(positive_scores - negative_scores), prog_bar=False, batch_size=batch_size)
        if mode == "train":
            self.log(f"{mode}/seen_tokens", len(self._seen_tokens))

    def update_seen_tokens(self, items: list[Item]):
        for item in items:
            self._seen_tokens.update(item.all_ids)

    def training_step(self, batch: list[tuple[Item, Item, Item]], batch_idx: int) -> torch.Tensor:
        anchor_items, positive_items, negative_items = tuple(map(list, zip(*batch)))
        anchor_embeddings = self(anchor_items)
        positive_embeddings = self(positive_items)
        negative_embeddings = self(negative_items)
        positive_scores = self.similarity(anchor_embeddings, positive_embeddings)
        negative_scores = self.similarity(anchor_embeddings, negative_embeddings)
        loss = self.loss(positive_scores, negative_scores)
        self.update_seen_tokens(anchor_items + positive_items + negative_items)
        self.log_train_or_val_or_test_metrics(
            mode="train",
            positive_scores=self.similarity(anchor_embeddings, positive_embeddings),
            negative_scores=self.similarity(anchor_embeddings, negative_embeddings),
            loss=loss
        )
        return loss

    def validation_or_test_step(self, mode: str, batch: list[tuple[Item, Item, Item]], batch_idx: int):
        anchor_items, positive_items, negative_items = tuple(map(list, zip(*batch)))
        anchor_embeddings = self(anchor_items)
        positive_embeddings = self(positive_items)
        negative_embeddings = self(negative_items)
        positive_scores = self.similarity(anchor_embeddings, positive_embeddings)
        negative_scores = self.similarity(anchor_embeddings, negative_embeddings)
        loss = self.loss(positive_scores, negative_scores)
        self.log_train_or_val_or_test_metrics(
            mode=mode,
            positive_scores=self.similarity(anchor_embeddings, positive_embeddings),
            negative_scores=self.similarity(anchor_embeddings, negative_embeddings),
            loss=loss
        )

    def validation_step(self, batch: list[tuple[Item, Item, Item]], batch_idx: int):
        self.validation_or_test_step("val", batch, batch_idx)

    def test_step(self, batch: list[tuple[Item, Item, Item]], batch_idx: int):
        self.validation_or_test_step("test", batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
