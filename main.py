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
from keras import losses
matplotlib.use('Agg')
cmap = matplotlib.colormaps['viridis']  



if __name__ == '__main__':
    config_path = sys.argv[1]
    config = get_config(config_path)
    
    # Access parameters
    dataset_path = config["dataset_path"]
    results_dir = config["results_dir"]
    name = config["name"]
    epoch_vae = config["epoch_vae"]
    latent_dim = config["latent_dim"]
    batch_size_vae = config["batch_size_vae"]
    kl_loss = config["kl_loss"]
    filtered = config["filter"]
    
    # Load dataset and preprocessing
    fci_dataset = DataLoaderFCI(dataset_path)
    dataset = fci_dataset.pipeline(batch_size=batch_size_vae, shuffle=True, split = 0.8,filter = filtered)

    input_shape = fci_dataset.get_shape(1)
    latent_dim = latent_dim
    r_loss = 1.
    k_loss = kl_loss 
    gain_loss = 0.
        
    # Get VAE model
    model = ModelSelector("1D")
    autoencoder, encoder, decoder = model.get_model(
        input_shape = (input_shape,1), 
        latent_dim=latent_dim,
        r_loss = r_loss,
        k_loss=k_loss,
        gain_loss=gain_loss
    )
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=4000,
                decay_rate=0.9,
                staircase=False
            )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)   
    
    autoencoder.compile(optimizer=optimizer)
    

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    folder_name = f"std_{name}_{fci_dataset.get_shape(0)}_latent_{int(latent_dim)}_kl_{k_loss}_{batch_size_vae}"
    res_folder = results_path / folder_name

    
    log_dir = res_folder / "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir
    )

    callbacks=[
        tensorboard_callback
    ]
    
    # Train VAE model
    history = autoencoder.fit(
            dataset["train_x"],
            epochs=epoch_vae, 
            validation_data=dataset["val_x"],
            callbacks=callbacks,
            verbose = 2
    )

    autoencoder.save(res_folder / "model.keras")
    encoder.save(res_folder / 'encoder_model.keras')
    decoder.save(res_folder / 'decoder_model.keras')
    
    # Saving latent space
    batch_size = 128
    dataset_batched = fci_dataset.pipeline(batch_size=batch_size,shuffle=False,split=0,filter = filtered)
    _, _, z = encoder.predict(dataset_batched["train_x"])
    np.savetxt(res_folder / 'latent_z.txt',z)
