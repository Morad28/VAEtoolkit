from src.dataloader import DataLoader
from src.model import ModelSelector
import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt 
from abc import ABC, abstractmethod
import copy


class Trainer(ABC):
    def __init__(self,model_selector : ModelSelector, data_loader : DataLoader, config):
        self.model_selector = model_selector
        self.data_loader = data_loader
        self.config = config
        self.results_path = None
        self.res_folder = None 
        self.models = {}
        self.history = {}
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
        epoch_vae = self.config["epoch_vae"]
        name = self.config["name"]
        latent_dim = self.config["latent_dim"]
        kl_loss = self.config["kl_loss"]
        num_components = self.config["num_components"]
        batch_size_vae = self.config["batch_size_vae"]
        model = self.config["Model"]["vae"]
        physical_penalty_weight = self.config["physical_penalty_weight"]
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        folder_name = f"std_{name}_{self.data_loader.get_shape()[0]}_latent_{int(latent_dim)}_kl_{kl_loss}_{batch_size_vae}_{model}"
        if model == "2D-MNIST-MoG":
            folder_name += f"_gaussians_{num_components}"
        if model == "1D-COILS-GAIN" or model == "COILS-MULTI" or model == "COILS-MULTI-OUT":
            gain_weight = self.config["gain_weight"]
            folder_name += f"_gw_{gain_weight}"
            if self.config["sep_loss"] or model == "COILS-MULTI-OUT":
                gain_loss = self.config["gain_loss"]
                folder_name += f"_gl_{gain_loss}"
                r_loss = self.config["r_loss"]
                folder_name += f"_rl_{r_loss}"
        if model == "COILS-MULTI" or model == "COILS-MULTI-OUT":
            values = self.config["values"]
            for value in values:
                folder_name += f"_{value}"
        folder_name += f"_phys_{physical_penalty_weight}"
        folder_name += f"_epochs_{epoch_vae}"
        self.res_folder = results_path / folder_name
        os.makedirs(os.path.dirname(self.res_folder / 'conf.json'), exist_ok=True)
        self.config["dataset_path"] = os.path.abspath(self.config["dataset_path"])
        with open(self.res_folder / 'conf.json', "w") as file:
            json.dump(self.config, file, indent=4)
    
    def _train_vae(self,x,y,models):
        
        self._create_folder()
        epoch_vae = self.config["epoch_vae"]
        # Handle multiple decoders
        if len(models["vae"]) == 3:
            autoencoder, encoder, decoder = models["vae"]
            multi_decoder = False
        elif len(models["vae"]) == 4:
            autoencoder, encoder, decoder_cnn, decoder_mlp = models["vae"]
            multi_decoder = True

        
        '''lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
                    decay_steps=4000,
                    decay_rate=0.9,
                    staircase=False
                )''' # Learning rate schedule for FCI
        
        # do a lr_schedule for MNIST
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=2500,
            decay_rate=0.95,
            staircase=True
        ) # Learning rate schedule for MNIST

        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)   
        
        autoencoder.compile(optimizer=optimizer)
        
        log_dir = self.res_folder / "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        callbacks=[
            tensorboard_callback
        ]
        
        # Train VAE model_selector
        history = autoencoder.fit(
                x,
                epochs=epoch_vae, 
                validation_data=y,
                callbacks=callbacks,
                verbose = 2
        )
        
        self.history["vae"] = history

        autoencoder.save(self.res_folder / "model.keras")
        encoder.save(self.res_folder / 'encoder_model.keras')
        if multi_decoder:
            decoder_cnn.save(self.res_folder / 'decoder_cnn_model.keras')
            decoder_mlp.save(self.res_folder / 'decoder_mlp_model.keras')
            self.models["vae"] = (autoencoder, encoder, decoder_cnn, decoder_mlp)
        else:
            decoder.save(self.res_folder / 'decoder_model.keras')
            self.models["vae"] = (autoencoder, encoder, decoder)
        
        return history
    
    def _train_mlp(self,x,y,models,res_folder_n=Path('./')):
        epoch_rna = self.config["epoch_rna"]
        latent_gain = copy.deepcopy(models["mlp"])

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

        return history, latent_gain

    
    
class TrainerFCI(Trainer):
    def __init__(self,model_selector : ModelSelector, data_loader : DataLoader, config):
        super().__init__(model_selector, data_loader, config)
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
        physical_penalty_weight = config["physical_penalty_weight"]
        
        dataset = self.data_loader.get_tf_dataset()

        input_shape = self.data_loader.get_shape()
            
        # Get VAE model_selector
        models = self.model_selector.get_model(
            input_shape = input_shape, 
            latent_dim  = latent_dim,
            k_loss      = kl_loss,
            physical_penalty_weight=physical_penalty_weight,
        )
        
        history = self._train_vae(dataset["train_x"],dataset["val_x"],models)
        _, encoder, decoder = models["vae"]
    
        # Saving latent space
        batch_size = 256
        dataset_batched, _ = self.data_loader.to_dataset(batch_size=batch_size, shuffle=False, split=0)
        _, _, z = encoder.predict(dataset_batched)

        np.savetxt(self.res_folder / 'latent_z.txt',z)

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
            res_folder = self.res_folder
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
        
        
        models = self.model_selector.get_model(
            latent_dim=latent_dim
        )

        history, latent_gain = self._train_mlp(gain_batched_train_dataset,gain_batched_validation_dataset,models,res_folder_n=res_folder_n)

        self.history[var_name] = history
        self.models[var_name] = latent_gain


        return history
    
    
class TrainerMNIST(Trainer):
    def __init__(self,model_selector : ModelSelector, data_loader : DataLoader, config):
        super().__init__(model_selector, data_loader, config)
        
    def train(self):
        self.train_vae()

    
    def train_vae(self):

        config = self.config
        kl_loss = config["kl_loss"]
        r_loss = config["r_loss"]
        latent_dim =  config["latent_dim"]
        num_components = config["num_components"]
        
        dataset = self.data_loader.get_tf_dataset()

        input_shape = self.data_loader.get_shape()
            
        # Get VAE model_selector
        models = self.model_selector.get_model(
            input_shape = input_shape, 
            latent_dim  = latent_dim,
            num_components = num_components,
            k_loss      = kl_loss,
            r_loss      = r_loss,
        )
        
        history = self._train_vae(dataset["train_x"],dataset["val_x"],models)
        _, encoder, _ = models["vae"]
        
        # Saving latent space
        batch_size = 256
        dataset_batched, _ = self.data_loader.to_dataset(batch_size=batch_size, shuffle=False, split=0)
        z = encoder.predict(dataset_batched)[-1]
        np.savetxt(self.res_folder / 'latent_z.txt',z)
        return history


class TrainerGain(Trainer):
    def __init__(self,model_selector : ModelSelector, data_loader : DataLoader, config):
        super().__init__(model_selector, data_loader, config)
        
    def train(self):
        self.train_vae()
    
    def train_vae(self):
        config = self.config
        kl_loss = config["kl_loss"]
        gain_loss = config["gain_loss"]
        latent_dim =  config["latent_dim"]
        num_components = config["num_components"]
        r_loss = config["r_loss"]
        physical_penalty_weight = config["physical_penalty_weight"]
        
        dataset = self.data_loader.get_tf_dataset()

        input_shape = self.data_loader.get_shape()
        print(f"\nInput shape: {input_shape}")
            
        # Get VAE model_selector
        models = self.model_selector.get_model(
            input_shape = input_shape,
            latent_dim  = latent_dim,
            num_components = num_components,
            k_loss      = kl_loss,
            gain_loss   = gain_loss,
            r_loss = r_loss,
            config = config,
            dataloader = self.data_loader,
            physical_penalty_weight = physical_penalty_weight,
        )
        
        history = self._train_vae(dataset["train_x"],dataset["val_x"],models)
        _, encoder, *decoder = models["vae"]
        
        # Saving latent space
        batch_size = 256
        dataset_batched, _ = self.data_loader.to_dataset(batch_size=batch_size, shuffle=False, split=0)
        z = encoder.predict(dataset_batched)[-1]
        np.savetxt(self.res_folder / 'latent_z.txt',z)
        return history



    def __init__(self,model_selector : ModelSelector, data_loader : DataLoader, config):
        super().__init__(model_selector, data_loader, config)
        
    def train(self):
        self.train_vae()
    
    def train_vae(self):
        config = self.config
        kl_loss = config["kl_loss"]
        gain_loss = config["gain_loss"]
        latent_dim =  config["latent_dim"]
        num_components = config["num_components"]
        r_loss = config["r_loss"]
        physical_penalty_weight = config["physical_penalty_weight"]
        
        dataset = self.data_loader.get_tf_dataset()

        input_shape = self.data_loader.get_shape()
        print(f"\nInput shape: {input_shape}")
            
        # Get VAE model_selector
        models = self.model_selector.get_model(
            input_shape = input_shape,
            latent_dim  = latent_dim,
            num_components = num_components,
            k_loss      = kl_loss,
            gain_loss   = gain_loss,
            r_loss = r_loss,
            config = config,
            dataloader = self.data_loader,
            physical_penalty_weight = physical_penalty_weight,
        )
        
        history = self._train_vae(dataset["train_x"],dataset["val_x"],models)
        _, encoder, *decoder = models["vae"]
        
        # Saving latent space
        batch_size = 256
        dataset_batched, _ = self.data_loader.to_dataset(batch_size=batch_size, shuffle=False, split=0)
        z = encoder.predict(dataset_batched)[-1]
        np.savetxt(self.res_folder / 'latent_z.txt',z)
        return history