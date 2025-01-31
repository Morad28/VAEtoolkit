from src.dataloader import DataLoader
from src.model import ModelSelector
import numpy as np
import tensorflow as tf
from pathlib import Path

class Trainer:
    def __init__(self,model : ModelSelector, data_loader : DataLoader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
    
    def train(self):
        pass
    
    
class Trainer_FCI(Trainer):
    def __init__(self,model : ModelSelector, data_loader : DataLoader, config):
        super().__init__(model, data_loader, config)
        self.res_folder = None
        self.dataset = None
        
    def train(self):
        self.train_vae()
    
    def train_vae(self):
        config = self.config
        # Access parameters
        dataset_path = config["dataset_path"]
        results_dir = config["results_dir"]
        name = config["name"]
        epoch_vae = config["epoch_vae"]
        latent_dim = config["latent_dim"]
        batch_size_vae = config["batch_size_vae"]
        kl_loss = config["kl_loss"]
        filtered = config["filter"]
        
        dataset = self.data_loader.get_tf_dataset()

        input_shape = self.data_loader.get_shape(1)
        latent_dim = latent_dim
        r_loss = 1.
        k_loss = kl_loss 
        gain_loss = 0.
            
        # Get VAE model
        autoencoder, encoder, decoder = self.model.get_model(
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
        folder_name = f"std_{name}_{self.data_loader.get_shape(0)}_latent_{int(latent_dim)}_kl_{k_loss}_{batch_size_vae}"
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
        self.data_loader.pipeline(batch_size=batch_size,shuffle=False,split=0,filter = filtered)
        dataset_batched = self.data_loader.get_tf_dataset()
        _, _, z = encoder.predict(dataset_batched["train_x"])
        np.savetxt(res_folder / 'latent_z.txt',z)

        
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
            z = np.loadtxt(res_folder+'latent_z.txt')
            self.data_loader = DataLoaderFCI(dataset_path)
            self.data_loader.apply_mask(filtered)
            loaded_dataset = self.data_loader.get_data()

        else:
            res_folder = self.res_folder
            z = np.loadtxt(res_folder+'latent_z.txt')
            loaded_dataset = self.dataset

        np_gain = np.array(loaded_dataset['values']["gain"]) /  np.max(loaded_dataset['values']["gain"])
            
        if log:
            norm_gain = np_gain
            norm_gain = np.log(norm_gain)
        else:
            norm_gain = np_gain
            
        res_folder_n = res_folder / "values" / var_name 
        
        gain_batched_train_dataset,gain_batched_validation_dataset = self.data_loader.to_dataset((z,norm_gain),batch_size_rna,shuffle=True, split = 0.8)
        ModelSelector = ModelSelector('1D')
        latent_gain = ModelSelector.get_model(latent_dim)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.0005,
                        decay_steps=500,
                        decay_rate=0.95,
                        staircase=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)   
        latent_gain.compile(optimizer=optimizer, loss=losses.MeanSquaredError(),metrics=['MAPE'])

        log_dir = res_folder_n + "logs"
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

        latent_gain.save(res_folder_n + "model.keras")