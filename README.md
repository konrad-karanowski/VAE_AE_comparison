# VAE_AE_Comparison
The repository contains the basic versions of autoencoders: regular autoencoder and variational auotencoder. The algorithms were compared in terms of image reconstruction and the possibility of generating based on the [FasionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The repository was made in order to learn the [PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) framework and the basics of generative modeling.

# Usage
To train models use `train.py` script:
```

```

In order to see logs of training use tensorboard:
```
tensorboard --logdir lightning_logs
```

In order to run experiments in Jupyter Notebook change `PATH_VAE` and `PATH_AE` in first line to checkpoints of your model. If you use different z_size, you must also change `Z_SIZE_VAE` and `Z_SIZE_AE`.
