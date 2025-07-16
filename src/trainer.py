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
        folder_name = f"{self.data_loader.get_shape()[0]}_lt_{int(latent_dim)}_kl_{kl_loss}_{model.split('-')[-1]}"
        if model == "2D-MNIST-MoG":
            folder_name += f"_gaussians_{num_components}"
        if model == "1D-COILS-GAIN" or model == "COILS-MULTI" or model == "COILS-MULTI-OUT" or model == "COILS-MULTI-OUT-DUO" or model == "COILS-MULTI-OUT-DUO-FOCUS":
            gain_weight = self.config["gain_weight"]
            folder_name += f"_gw{gain_weight}"
            if self.config["sep_loss"] or model == "COILS-MULTI-OUT" or model == "COILS-MULTI-OUT-DUO" or model == "COILS-MULTI-OUT-DUO-FOCUS":
                gain_loss = self.config["gain_loss"]
                folder_name += f"_gl{gain_loss}"
                r_loss = self.config["r_loss"]
                folder_name += f"_rl{r_loss}"
            if model == "COILS-MULTI-OUT" or model == "COILS-MULTI-OUT-DUO" or model == "COILS-MULTI-OUT-DUO-FOCUS":
                if self.config["predict_z_mean"]:
                    folder_name += "_mean"
        if model == "COILS-MULTI" or model == "COILS-MULTI-OUT" or model == "COILS-MULTI-OUT-DUO" or model == "COILS-MULTI-OUT-DUO-FOCUS":
            values = self.config["values"]
            for value in values:
                folder_name += f"_{value}"
        for key in self.config["filter"]:
            folder_name += f"_{key}min{self.config['filter'][key]}"
        if physical_penalty_weight > 0:
            folder_name += f"_phy{physical_penalty_weight}"
        folder_name += f"_epc{epoch_vae}"
        if model == "COILS-MULTI-OUT-DUO" or model == "COILS-MULTI-OUT-DUO-FOCUS":
            folder_name += f"_spls{self.config['sep_loss']}"
            folder_name += f"_smth{self.config['smooth']}"
            folder_name += f"_gdm{self.config['gain_latent_size']}"
            if self.config["kl_annealing"]:
                folder_name += f"_kla{self.config['kl_annealing'][0]}"
                if self.config["kl_annealing"] == "cyclical":
                    folder_name += f"_wrmp{self.config['warmup_steps']}"
                    folder_name += f"_cc{self.config['cycle_length']}"
        if model == "COILS-MULTI-OUT-DUO-FOCUS":
            folder_name += f"_klpr{self.config['kl_loss_profile']}"
        if self.config["finetuning"]:
            folder_name = self.config["res_folder"].split("/")[-1]  # Use the provided res_folder name
            folder_name += f"_epc{epoch_vae}"
            folder_name += "ft"
        if self.config["transfer_learning"]:
            folder_name = self.config["res_folder"].split("/")[-1]  # Use the provided res_folder name
            folder_name += f"_epc{epoch_vae}"
            folder_name += "tl"
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
            multi_encoder = False
        elif len(models["vae"]) == 4:
            autoencoder, encoder, decoder_cnn, decoder_mlp = models["vae"]
            multi_decoder = True
            multi_encoder = False
        elif len(models["vae"]) == 6:
            autoencoder, encoder_cnn, encoder_mlp, encoder_latent, decoder_cnn, decoder_mlp = models["vae"]
            multi_decoder = True
            multi_encoder = True
        
        """if self.config["finetuning"]:
            # load the keras models 
            autoenc"""

        
        '''lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
                    decay_steps=4000,
                    decay_rate=0.9,
                    staircase=False
                )''' # Learning rate schedule for FCI
        
        # do a lr_schedule for MNIST
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config["learning_rate"],
            decay_steps=2500,
            decay_rate=0.95,
            staircase=True
        ) # Learning rate schedule for MNIST

        if self.config["finetuning"]:
            if not multi_encoder or not multi_decoder:
                raise ValueError("Transfer learning is only supported for multi-encoder and multi-decoder models.")
            res_folder_old = self.config["res_folder"]
            # load the weights from the previous model
            encoder_cnn.load_weights(res_folder_old + "/encoder_cnn.weights.h5")
            encoder_mlp.load_weights(res_folder_old + "/encoder_mlp.weights.h5")
            encoder_latent.load_weights(res_folder_old + "/encoder_latent.weights.h5")
            decoder_cnn.load_weights(res_folder_old + "/decoder_cnn.weights.h5")
            decoder_mlp.load_weights(res_folder_old + "/decoder_mlp.weights.h5")
        
        if self.config["transfer_learning"]:
            if not multi_encoder or not multi_decoder:
                raise ValueError("Transfer learning is only supported for multi-encoder and multi-decoder models.")
            res_folder_old = self.config["res_folder"]
            # load the weights from the previous model
            encoder_cnn.load_weights(res_folder_old + "/encoder_cnn.weights.h5")
            encoder_mlp.load_weights(res_folder_old + "/encoder_mlp.weights.h5")
            encoder_latent.load_weights(res_folder_old + "/encoder_latent.weights.h5")
            decoder_cnn.load_weights(res_folder_old + "/decoder_cnn.weights.h5")
            decoder_mlp.load_weights(res_folder_old + "/decoder_mlp.weights.h5")
            # freeze all layers and add a trainable layer to the decoders
            encoder_cnn.trainable = False
            encoder_mlp.trainable = False
            encoder_latent.trainable = False
            decoder_cnn.trainable = True
            decoder_mlp.trainable = True
            #this will be to adapt depending on the quality of the transfer learning dataset
            """#reset the weights of the decoders
            decoder_cnn.set_weights([np.random.rand(*w.shape)-0.5 for w in decoder_cnn.get_weights()])
            decoder_mlp.set_weights([np.random.rand(*w.shape)-0.5 for w in decoder_mlp.get_weights()])"""
            """ # only last layer not enough for training apparently
            # replace the last layer of the decoders (not the reshape layers) with the same layer but trainable and with random weights
            decoder_cnn_last_layer = decoder_cnn.layers[-3]
            decoder_mlp_last_layer = decoder_mlp.layers[-2]
            print(f"Decoder CNN last layer: {decoder_cnn_last_layer}"
                  f"\nDecoder MLP last layer: {decoder_mlp_last_layer}")
            decoder_cnn_last_layer.trainable = True
            decoder_mlp_last_layer.trainable = True
            # For each weight in the layer, generate a random array of the same shape
            random_weights = [np.random.rand(*w.shape)-0.5 for w in decoder_cnn_last_layer.get_weights()]
            decoder_cnn_last_layer.set_weights(random_weights)
            random_weights = [np.random.rand(*w.shape)-0.5 for w in decoder_mlp_last_layer.get_weights()]
            decoder_mlp_last_layer.set_weights(random_weights)
            decoder_cnn.layers[-3] = decoder_cnn_last_layer
            decoder_mlp.layers[-2] = decoder_mlp_last_layer"""
        



        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
        autoencoder.compile(optimizer=optimizer)
        log_dir = self.res_folder / "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # --- SubmoduleWeightsSaver integration ---
        submodules = {}
        if multi_encoder:
            submodules["encoder_cnn"] = encoder_cnn
            submodules["encoder_mlp"] = encoder_mlp
            submodules["encoder_latent"] = encoder_latent
            if multi_decoder:
                submodules["decoder_cnn"] = decoder_cnn
                submodules["decoder_mlp"] = decoder_mlp
        else:
            submodules["encoder"] = encoder
            if multi_decoder:
                submodules["decoder_cnn"] = decoder_cnn
                submodules["decoder_mlp"] = decoder_mlp
            else:
                submodules["decoder"] = decoder
        submodule_weights_saver = SubmoduleWeightsSaver(submodules, self.res_folder, self.config, freq=1)  # Save every epoch
        # --- End integration ---

        callbacks=[
            tensorboard_callback,
            submodule_weights_saver
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
        super(type(autoencoder), autoencoder).save(self.res_folder / "model.keras")
        if multi_encoder:
            encoder_cnn.save(self.res_folder / 'encoder_cnn_model.keras')
            encoder_mlp.save(self.res_folder / 'encoder_mlp_model.keras')
            encoder_latent.save(self.res_folder / 'encoder_latent_model.keras')
            if multi_decoder:
                decoder_cnn.save(self.res_folder / 'decoder_cnn_model.keras')
                decoder_mlp.save(self.res_folder / 'decoder_mlp_model.keras')
                self.models["vae"] = (autoencoder, encoder_cnn, encoder_mlp, encoder_latent, decoder_cnn, decoder_mlp)
        else:
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
    """
    Trainer for MNIST dataset.
    """
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
    """
    Trainer for all coil models
    """
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
        if self.config["Model"]["vae"] == "COILS-MULTI-OUT-DUO" or self.config["Model"]["vae"] == "COILS-MULTI-OUT-DUO-FOCUS":
            _, encoder_cnn, encoder_mlp, encoder_latent, decoder_cnn, decoder_mlp = models["vae"]
            
            # Combine the three encoders
            cnn_output = encoder_cnn.output
            if self.config["Model"]["vae"] == "COILS-MULTI-OUT-DUO-FOCUS":
                cnn_output = cnn_output[2]  # Extract the latent representation from the CNN output
            mlp_output = encoder_mlp.output
            concatenated = tf.keras.layers.Concatenate()([cnn_output, mlp_output])
            z_mean, z_log_var, z = encoder_latent(concatenated)
            encoder = tf.keras.Model(
                inputs=[encoder_cnn.input, encoder_mlp.input],
                outputs=[z_mean, z_log_var, z]
            )
        else:
            _, encoder, *decoder = models["vae"]

        # Saving latent space
        batch_size = 256
        dataset_batched, _ = self.data_loader.to_dataset(batch_size=batch_size, shuffle=False, split=0)

        if self.config["Model"]["vae"] == "COILS-MULTI-OUT-DUO" or self.config["Model"]["vae"] == "COILS-MULTI-OUT-DUO-FOCUS":
            # Initialize lists to store profile and vals
            profiles = []
            vals = []
            
            # Iterate over the dataset to extract profile and vals
            len_values = len(self.config["values"])
            for batch in dataset_batched:
                # Unpack the batch tuple (assuming the first element is the input tensor)
                inputs = batch[0]  # Adjust this if your dataset structure is different
                profiles.append(inputs[:, :-len_values])
                vals.append(inputs[:, -len_values:])
            
            # Convert lists to NumPy arrays
            profile = np.concatenate(profiles, axis=0)
            vals = np.concatenate(vals, axis=0)
            
            # Use the combined encoder for prediction
            z_mean, z_log_var, z = encoder.predict([profile, vals])
        else:
            z = encoder.predict(dataset_batched)[-1]

        np.savetxt(self.res_folder / 'latent_z.txt', z)
        return history
class SubmoduleWeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, submodules, save_dir, config, freq=1):
        super().__init__()
        self.submodules = submodules  # dict: name -> model
        self.save_dir = Path(save_dir)
        self.freq = freq
        self.config = config
        print(f"Submodules: {list(self.submodules.keys())}")

    def on_epoch_end(self, epoch, logs=None):
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        if (epoch+1) >= self.config["epoch_vae"]:
            for name, model in self.submodules.items():
                path = self.save_dir / f"{name}.weights.h5"
                model.save_weights(path)