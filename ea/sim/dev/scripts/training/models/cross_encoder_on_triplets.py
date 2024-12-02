import lightning as L
import torch
from torch import optim
from torchmetrics.classification import BinaryAccuracy

from ea.sim.dev.scripts.training.reranker_losses import BCELossWithLogits, RerankerLoss
from ea.sim.main.methods.neural.cross_encoders.base import CrossEncoder
from ea.sim.main.utils import Scope, device
from ea.sim.main.methods.neural.encoders.objects import Item


class CrossEncoderModel(L.LightningModule):
    def __init__(self, encoder: CrossEncoder, scope: Scope, loss: RerankerLoss, train_size: int, batch_size: int, epochs: int):
        super().__init__()
        self.encoder = encoder
        self.acc = BinaryAccuracy()
        self._seen_tokens = set()
        self._loss = loss

        self._scope = scope
        self._train_size = train_size
        self._batch_size = batch_size
        self._epochs = epochs


    def loss(self, batch: list[tuple[Item, Item, Item]]) -> tuple[torch.Tensor, ...]:
        positive_scores = self.compute_positive_scores(batch)
        negative_scores = self.compute_negative_scores(batch)
        return self._loss(positive_scores, negative_scores), positive_scores, negative_scores
    
    def forward(self, x: list[Item], y: list[Item]) -> torch.Tensor:
        return self.encoder(x, y)
    
    def compute_positive_scores(self, batch: list[tuple[Item, Item, Item]]) -> torch.Tensor:
        return self([anchor for (anchor, _, _) in batch], [positive for (_, positive, _) in batch])

    def compute_negative_scores(self, batch: list[tuple[Item, Item, Item]]) -> torch.Tensor:
        return self([anchor for (anchor, _, _) in batch], [negative for (_, _, negative) in batch])

    def log_metrics(
            self,
            mode: str,
            positive_scores: torch.Tensor, negative_scores: torch.Tensor, loss: torch.Tensor,
            batch_size: int
    ):
        preds = (positive_scores >= negative_scores).float()
        target = torch.tensor([1] * len(positive_scores)).to(device)
        self.log(f"{mode}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{mode}/acc", self.acc(preds, target), prog_bar=True, batch_size=batch_size)
        self.log(f"{mode}/diff", (positive_scores - negative_scores).mean(), prog_bar=False, batch_size=batch_size)
        self.log(f"{mode}/pos_scores", positive_scores.mean(), prog_bar=False, batch_size=batch_size)
        self.log(f"{mode}/neg_scores", negative_scores.mean(), prog_bar=False, batch_size=batch_size)
        self.log("param/temp", self._loss.temp[1], prog_bar=False, batch_size=batch_size)

        optimizer = self.optimizers()
        self.log("lr", optimizer.param_groups[0]['lr'], prog_bar=False, batch_size=batch_size)

    def update_seen_tokens(self, batch: list[tuple[Item, Item, Item]]):
        for anchor, positive, negative in batch:
            self._seen_tokens.update(anchor.all_ids + positive.all_ids + negative.all_ids)

    def training_step(self, batch: list[tuple[Item, Item, Item]], batch_idx: int) -> torch.Tensor:
        loss, positive_scores, negative_scores = self.loss(batch)
        self.log_metrics("train", positive_scores, negative_scores, loss, batch_size=len(batch))
        self.log("train/seen_tokens", len(self._seen_tokens))
        self.update_seen_tokens(batch)
        return loss

    def validation_step(self, batch: list[tuple[Item, Item, Item]], batch_idx: int):
        loss, positive_scores, negative_scores = self.loss(batch)
        self.log_metrics("val", positive_scores, negative_scores, loss, batch_size=len(batch))

    def test_step(self, batch: list[tuple[Item, Item, Item]], batch_idx: int):
        loss, positive_scores, negative_scores = self.loss(batch)
        self.log_metrics("test", positive_scores, negative_scores, loss, batch_size=len(batch))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-3)
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer,
        #     max_lr=1e-3,
        #     epochs=10,
        #     steps_per_epoch=self._train_size // self._batch_size + 1
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._epochs * self._train_size // self._batch_size)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
