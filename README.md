# VAE_AE_Comparison
The repository contains the basic versions of autoencoders: regular autoencoder and variational auotencoder. The algorithms were compared in terms of image reconstruction and the possibility of generating based on the [FasionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The repository was made in order to learn the [PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) framework and the basics of generative modeling.

# Usage
To train models use `train.py` script:
```
  -h, --help            show this help message and exit
  -e EPOCHS, --max_epochs EPOCHS
                        Maximal number of epochs
  -k KIND, --kind KIND  vae or ae
  -z Z_SIZE, --z_size Z_SIZE
                        Latent vector size
  -b BS, --batch_size BS
                        Batch size
  -s SEED, --seed SEED  Seed for torch random
  -g GPUS, --gpus GPUS  Number of GPU
  -lr LR, --lr LR       Learning rate
  -a ALPHA, --alpha ALPHA
                        Alpha for balance KLD loss / RC Loss (VAE only)

```

In order to see logs of training use tensorboard:
```
tensorboard --logdir lightning_logs
```

In order to run experiments in Jupyter Notebook change `PATH_VAE` and` PATH_AE` in the first line to checkpoints of your model. If you use a different z_size, you must also change `Z_SIZE_VAE` and` Z_SIZE_AE`.
