import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, z_size: int):
        super(Encoder, self).__init__()
        self._conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self._conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self._conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self._flatten = nn.Flatten()
        self._latent = nn.Linear(3136, z_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        encoded = self._flatten(x)
        return self._latent(encoded)


class Decoder(nn.Module):

    def __init__(self, z_size: int):
        super(Decoder, self).__init__()
        self._latent = nn.Sequential(
            nn.Linear(z_size, 3136),
            nn.ReLU()
        )
        self._convup1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self._convup2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self._convup3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self._convup4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._latent(x).view(-1, 64, 7, 7)
        x = self._convup1(x)
        x = self._convup2(x)
        x = self._convup3(x)
        x = self._convup4(x)[:, :, :28, :28]
        return x


class Autoencoder(nn.Module):

    def __init__(self, z_size: int):
        super(Autoencoder, self).__init__()
        self._encoder = Encoder(z_size)
        self._decoder = Decoder(z_size)

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decoder(self._encoder(x))


class VEncoder(nn.Module):

    def __init__(self, z_size):
        super(VEncoder, self).__init__()
        self._conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self._conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self._conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self._flatten = nn.Flatten()
        self._mi = nn.Linear(3136, z_size)
        self._log_var = nn.Linear(3136, z_size)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._flatten(x)
        # encode
        mi = self._mi(x)
        log_var = self._log_var(x)
        epsilon = torch.randn(mi.shape).to(mi.get_device())
        latent = mi + epsilon * torch.exp(log_var / 2)
        return latent, mi, log_var


class VariationalAutoencoder(nn.Module):

    def __init__(self, z_size):
        super(VariationalAutoencoder, self).__init__()
        self._encoder = VEncoder(z_size)
        self._decoder = Decoder(z_size)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        latent, mi, log_var = self._encoder(x)
        return self._decoder(latent), mi, log_var

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder
