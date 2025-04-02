import json
import sys


def get_config(path):
    # Default values
    default_config = {
        "results_dir": "results/",
        "name": "",
        "epoch_vae": 1,
        "epoch_rna": 1,
        "latent_dim": 10,
        "batch_size_vae": 64,
        "batch_size_rna": 32,
        "kl_loss": 1e-4,
        "r_loss": 1.0,
        "vae_normalization": 1,
        "log": False
    }

    # Load configuration file
    with open(path, 'r') as f:
        user_config = json.load(f)

    # Merge user configuration with default values
    config = {**default_config, **user_config}
    return config