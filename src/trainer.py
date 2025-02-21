from src.dataloader import DataLoader
from src.model import ModelSelector
import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt 
from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init__(self,model : ModelSelector, data_loader : DataLoader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.results_path = None
        self.res_folder = None 
        self._prepare_data()

    @abstractmethod
    def train(self):
        """Train routine.
        """
        pass
        
    def _prepare_data(self):
        self.data_loader.pipeline()
        
        
    def _create_folder(self):
        results_dir = self.config["results_dir"]
        name = self.config["name"]
        latent_dim = self.config["latent_dim"]
        kl_loss = self.config["kl_loss"]
        batch_size_vae = self.config["batch_size_vae"]
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        folder_name = f"std_{name}_{self.data_loader.get_shape()[0]}_latent_{int(latent_dim)}_kl_{kl_loss}_{batch_size_vae}"
        self.res_folder = results_path / folder_name
        os.makedirs(os.path.dirname(self.res_folder / 'conf.json'), exist_ok=True)
        self.config["dataset_path"] = os.path.abspath(self.config["dataset_path"])
        with open(self.res_folder / 'conf.json', "w") as file:
            json.dump(self.config, file, indent=4)
    
    def _train_vae(self,x,y,models):
        
        self._create_folder()
        epoch_vae = self.config["epoch_vae"]
        autoencoder, encoder, decoder = models["vae"]

        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
                    decay_steps=4000,
                    decay_rate=0.9,
                    staircase=False
                )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)   
        
        autoencoder.compile(optimizer=optimizer)
        
        log_dir = self.res_folder / "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        callbacks=[
            tensorboard_callback
        ]
        
        # Train VAE model
        history = autoencoder.fit(
                x,
                epochs=epoch_vae, 
                validation_data=y,
                callbacks=callbacks,
                verbose = 2
        )

        autoencoder.save(self.res_folder / "model.keras")
        encoder.save(self.res_folder / 'encoder_model.keras')
        decoder.save(self.res_folder / 'decoder_model.keras')
        
        models["vae"] = (autoencoder, encoder, decoder)
        
        data_train = history.history['loss']
        data_val = history.history['val_loss']

        np.savetxt(self.res_folder / "losses.txt",[data_train,data_val])
        
        plt.figure()
        plt.grid(True,which="both")
        plt.semilogy(data_train,label="Données d'entraînement")
        plt.semilogy(data_val,label="Données de validation")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.tick_params(axis='both', which='both', direction='in')
        plt.legend(frameon=True)
        plt.savefig(self.res_folder / "losses.png")
        plt.close()
                
        return history
    
    def _train_mlp(self,x,y,models,res_folder_n=Path('./')):
        epoch_rna = self.config["epoch_rna"]
        latent_gain = models["mlp"]

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.0005,
                decay_steps=500,
                decay_rate=0.95,
                staircase=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)   
        latent_gain.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),metrics=['MAPE'])

        log_dir = res_folder_n / "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        callbacks=[tensorboard_callback]

        history = latent_gain.fit(x,
            epochs=epoch_rna, 
            validation_data=y,
            callbacks=callbacks,
            verbose = 2)

        latent_gain.save(res_folder_n / "model.keras")

        models["mlp"] = latent_gain

        return history

    
    
class TrainerFCI(Trainer):
    def __init__(self,model : ModelSelector, data_loader : DataLoader, config):
        super().__init__(model, data_loader, config)
        self.res_folder = None
        
    def train(self):
        gain_only = self.config["reprise"]["gain_only"]
        training = self.config['training']
        
        if not gain_only:
            self.train_vae()
        for key in training:
            self.train_gain(key)
    
    def train_vae(self):
        config = self.config
        kl_loss = config["kl_loss"]
        latent_dim =  config["latent_dim"]
        
        dataset = self.data_loader.get_tf_dataset()

        input_shape = self.data_loader.get_shape()
            
        # Get VAE model
        models = self.model.get_model(
            input_shape = input_shape, 
            latent_dim  = latent_dim,
            k_loss      = kl_loss
        )
        
        history = self._train_vae(dataset["train_x"],dataset["val_x"],models)
        _, encoder, decoder = models["vae"]
    
        # Saving latent space
        batch_size = 256
        dataset_batched, _ = self.data_loader.to_dataset(batch_size=batch_size, shuffle=False, split=0)
        _, _, z = encoder.predict(dataset_batched)
        tilde_laser = decoder.predict(z)
        data, label = self.data_loader.get_x_y()

        np.savetxt(self.res_folder / 'latent_z.txt',z)



        error = []
        for i in range(len(data)):
            error.append( np.max(np.abs(data[i] - tilde_laser[i])) / np.max(np.abs(data[i])) )

        plt.figure()            
        plt.hist(np.array(error) * 100 ,bins=30)
        plt.title("Erreur de reconstruction")
        plt.savefig(self.res_folder / "hist_error.png")
        plt.close()
        
        plt.figure()            
        plt.hist(label ,bins=30)
        plt.title("Distribution des gains")
        plt.yscale("log")
        plt.savefig(self.res_folder / "hist_gain.png")
        plt.close()

        return history

        
    def train_gain(self,var_name):
        config = self.config
        # Access parameters
        results_dir = config["results_dir"]
        name = config["name"]
        latent_dim = (config["latent_dim"])
        batch_size_rna = config["batch_size_rna"]
        kl_loss = (config["kl_loss"])
        gain_only = config["reprise"]["gain_only"]
        batch_size_vae = config["batch_size_vae"]
        
        if gain_only:
            res_folder = Path(config["reprise"]["result_folder"])
            self.res_folder = res_folder
            z = np.loadtxt(res_folder / 'latent_z.txt')

        else:
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            folder_name = f"std_{name}_{self.data_loader.get_shape()[0]}_latent_{int(latent_dim)}_kl_{kl_loss}_{batch_size_vae}"
            res_folder = results_path / folder_name
            z = np.loadtxt(res_folder / 'latent_z.txt')

        _, gain = self.data_loader.get_x_y(var_name)
        gain_dataset = self.data_loader.to_tensorflow_dataset((z,gain))
        res_folder_n = res_folder / "values" / var_name 
        gain_batched_train_dataset,gain_batched_validation_dataset = self.data_loader.to_dataset(
            batch_size=batch_size_rna,
            shuffle=True,
            split=0.8,
            dataset=gain_dataset
        )
        
        
        models = self.model.get_model(
            latent_dim=latent_dim
        )

        history = self._train_mlp(gain_batched_train_dataset,gain_batched_validation_dataset,models,res_folder_n=res_folder_n)

        return history
    
    
class TrainerMNIST(Trainer):
    def __init__(self,model : ModelSelector, data_loader : DataLoader, config):
        super().__init__(model, data_loader, config)
        
    def train(self):
        self.train_vae()

    
    def train_vae(self):
        config = self.config
        kl_loss = config["kl_loss"]
        latent_dim =  config["latent_dim"]
        
        dataset = self.data_loader.get_tf_dataset()

        input_shape = self.data_loader.get_shape()
            
        # Get VAE model
        models = self.model.get_model(
            input_shape = input_shape, 
            latent_dim  = latent_dim,
            k_loss      = kl_loss
        )
        
        history = self._train_vae(dataset["train_x"],dataset["val_x"],models)
        _, encoder, _ = models["vae"]
    
        # Saving latent space
        batch_size = 256
        dataset_batched, _ = self.data_loader.to_dataset(batch_size=batch_size, shuffle=False, split=0)
        _, _, z = encoder.predict(dataset_batched)
        np.savetxt(self.res_folder / 'latent_z.txt',z)
        return history
