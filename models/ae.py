import torch
from torch import nn
import pytorch_lightning as pl


class LitAutoencoder(pl.LightningModule):

    def __init__(self, autoencoder: nn.Module, criterion: nn.Module = nn.MSELoss()):
        super(LitAutoencoder, self).__init__()
        self._autoencoder = autoencoder
        self._criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._autoencoder(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out = self._autoencoder(x)
        loss = self._criterion(out, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out = self._autoencoder(x)
        loss = self._criterion(out, x)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @property
    def autoencoder(self):
        return self._autoencoder
