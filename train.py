import argparse

import torch
import pytorch_lightning as pl

from dataset import get_datasets
from models import LitAutoencoder, Autoencoder, VariationalAutoencoder, LitVariationalAutoencoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--max_epochs',
        type=int,
        help='Maximal number of epochs',
        dest='epochs'
    )
    parser.add_argument(
        '-k',
        '--kind',
        type=str,
        help='vae or ae',
        dest='kind',
        default='ae'
    )
    parser.add_argument(
        '-z',
        '--z_size',
        type=int,
        help='Latent vector size',
        dest='z_size',
        default=10
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='Batch size',
        dest='bs',
        default=10
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        help='Seed for torch random',
        dest='seed',
        default=2137
    )
    parser.add_argument(
        '-g',
        '--gpus',
        type=int,
        help='Number of GPU',
        dest='gpus',
        default=1
    )
    parser.add_argument(
        '-lr',
        '--lr',
        type=float,
        help='Learning rate',
        dest='lr',
        default=1e-4
    )
    parser.add_argument(
        '-a',
        '--alpha',
        type=float,
        help='Alpha for balance KLD loss / RC Loss (VAE only)',
        dest='alpha',
        default=1000

    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.kind == 'ae':
        autoencoder = Autoencoder(args.z_size)
        model = LitAutoencoder(autoencoder, lr=args.lr)
    elif args.kind == 'vae':
        autoencoder = VariationalAutoencoder(args.z_size)
        model = LitVariationalAutoencoder(autoencoder, alpha=args.alpha, lr=args.lr)
    else:
        raise KeyError('Kind must be "vae" or "ae"')

    # get data
    train_data, val_data, test_data = get_datasets(
        batch_size=args.bs
    )

    # setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        callbacks=[pl.callbacks.EarlyStopping('val_loss')]
    )
    trainer.fit(model, train_data, val_data)


if __name__ == '__main__':
    main()
