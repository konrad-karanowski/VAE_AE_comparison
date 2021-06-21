import torch
from torch import nn
import pytorch_lightning as pl

from models import VariationalAutoencoder


class VAELoss(nn.Module):

    def __init__(self, alpha: float = 1000):
        super(VAELoss, self).__init__()
        self._alpha = alpha
        self._reconstruction_criterion = nn.MSELoss()

    def forward(self, pred, target, mi, log_var) -> (torch.Tensor, float, float):
        rc_loss = self._reconstruction_criterion(pred, target)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mi ** 2 - log_var.exp(), dim=1), dim=0)
        return rc_loss * self._alpha + kl_loss, rc_loss.item(), kl_loss.item()


class LitVariationalAutoencoder(pl.LightningModule):

    def __init__(self, autoencoder: VariationalAutoencoder, alpha: float = 1000, lr: float = 1e-3):
        super(LitVariationalAutoencoder, self).__init__()
        self._lr = lr
        self._autoencoder = autoencoder
        self._criterion = VAELoss(alpha)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self._autoencoder(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out, mi, log_var = self._autoencoder(x)
        loss, base_loss, kl_loss = self._criterion(out, x, mi, log_var)
        self.log('train_loss', loss)
        self.log('train_reconstruction_loss', base_loss)
        self.log('train_kl_loss', kl_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out, mi, log_var = self._autoencoder(x)
        loss, base_loss, kl_loss = self._criterion(out, x, mi, log_var)
        self.log('val_loss', loss)
        self.log('val_reconstruction_loss', base_loss)
        self.log('val_kl_loss', kl_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    @property
    def autoencoder(self) -> nn.Module:
        return self._autoencoder
