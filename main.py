import sys
from src.model import ModelSelector
from src.dataloader import DataLoader, DataLoaderFCI
from pathlib import Path
from src.config_vae import get_config
import sys
import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib
import shutil
import tensorflow as tf
from src.trainer import Trainer_FCI
from keras import losses
matplotlib.use('Agg')
cmap = matplotlib.colormaps['viridis']  

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    # config_path = sys.argv[1]
    config_path = "conf.json"
    config = get_config(config_path)
    
    # # Access parameters
    # dataset_path = config["dataset_path"]
    # results_dir = config["results_dir"]
    # name = config["name"]
    # epoch_vae = config["epoch_vae"]
    # latent_dim = config["latent_dim"]
    # batch_size_vae = config["batch_size_vae"]
    # kl_loss = config["kl_loss"]
    # filtered = config["filter"]
    
    # Load dataset and preprocessing
    fci_dataset = DataLoaderFCI(config)
    # fci_dataset.pipeline(batch_size=batch_size_vae, shuffle=True, split = 0.8, filter = filtered)
        
    # Get VAE model
    model = ModelSelector()
    model.select(vae = "1D-FCI", gain = '12MLP')
    
    trainer = Trainer_FCI(model, fci_dataset, config)
    trainer.train()
