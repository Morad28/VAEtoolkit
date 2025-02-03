from src.dataloader import DataLoader
from src.model import ModelSelector
import numpy as np
import tensorflow as tf
from pathlib import Path
import copy

class Trainer:
    def __init__(self,model : ModelSelector, data_loader : DataLoader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.results_path = None
        self.res_folder = None 
        
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
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
        )

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
        
        models["vae"] = [autoencoder, encoder, decoder]
                
        return history
    
    def train(self):
        pass
    
    
class Trainer_FCI(Trainer):
    def __init__(self,model : ModelSelector, data_loader : DataLoader, config):
        super().__init__(model, data_loader, config)
        self.res_folder = None
        self.dataset = None
        self._raw_data = copy.deepcopy(self.data_loader)
        
    def train(self):
        self.train_vae()
        self.train_gain('gain')
    
    def train_vae(self):
        config = self.config
        # Access parameters
        filtered = config["filter"]
        kl_loss = config["kl_loss"]
        latent_dim =  config["latent_dim"]
        batch_size_vae = config["batch_size_vae"]
        
        self.data_loader.pipeline(
            batch_size = batch_size_vae,
            filter = filtered,
            shuffle = True,
            split = 0.8
        )
        
        dataset = self.data_loader.get_tf_dataset()
        self.dataset = dataset


        input_shape = self.data_loader.get_shape()[1]
        r_loss = 1.
        k_loss = kl_loss 
        gain_loss = 0.
            
        # Get VAE model
        models = self.model.get_model(
            input_shape = (input_shape,1), 
            latent_dim=latent_dim,
            r_loss = r_loss,
            k_loss=k_loss,
            gain_loss=gain_loss
        )
        
        history = self._train_vae(dataset["train_x"],dataset["val_x"],models)
        _, encoder, _ = models["vae"]
    
        # Saving latent space
        batch_size = 256
        self._raw_data.pipeline(batch_size=batch_size, shuffle=False, split=0, filter = filtered)
        dataset_batched = self._raw_data.get_tf_dataset()
        _, _, z = encoder.predict(dataset_batched["train_x"])
        np.savetxt(self.res_folder / 'latent_z.txt',z)

        
    def train_gain(self,var_name):
        config = self.config
        # Access parameters
        dataset_path = config["dataset_path"]
        results_dir = config["results_dir"]
        name = config["name"]
        epoch_vae = config["epoch_vae"]
        epoch_rna = (config["epoch_rna"])
        latent_dim = (config["latent_dim"])
        batch_size_vae = config["batch_size_vae"]
        batch_size_rna = config["batch_size_rna"]
        kl_loss = (config["kl_loss"])
        log = config["log"]
        values = config["training"]
        gain_only = config["reprise"]["gain_only"]
        filtered = config["filter"]
        
        if gain_only:
            res_folder = config["reprise"]["result_folder"]
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
        
        gain_batched_train_dataset,gain_batched_validation_dataset = self.data_loader.to_dataset(dataset=gain_dataset)
        
        
        models = self.model.get_model(
            latent_dim=latent_dim
        )
        
        latent_gain = models["gain"]

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.0005,
                        decay_steps=500,
                        decay_rate=0.95,
                        staircase=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)   
        latent_gain.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),metrics=['MAPE'])

        log_dir = res_folder_n / "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)


        callbacks=[
            # callback,
            tensorboard_callback]


        epoch = epoch_rna


        history = latent_gain.fit(gain_batched_train_dataset,
            epochs=epoch, 
            validation_data=gain_batched_validation_dataset,
            callbacks=callbacks,
            verbose = 2)

        latent_gain.save(res_folder_n / "model.keras")