import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from src.vae_class import Sampling, SamplingLayer
import numpy as np


class DataLoader:
    def __init__(self, dataset_path, result_folder=None):
        
        self.dataset_path = dataset_path
        self.result_folder = result_folder
        self.dataset = None
        self.model = None
        self._load_data()
        if result_folder is not None: self._load_model()
                
    def get_data(self):
        """Get data.
        """
        return self.dataset
    
    def get_model(self):
        return self.model
    
    def to_dataset(self,data,batch_size,shuffle=True,split=0.8):
        """ shuflle, split and Convert to tf.data.Dataset object.
        """
        dataset = data
        if shuffle:
            dataset = tf.random.shuffle(dataset)
            
        train_dataset = dataset[:int(len(dataset)*split)]
        test_dataset = dataset[int(len(dataset)*split):]
        train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
           
        return train_dataset, test_dataset
    
    def _load_data(self):
        """Load dataset.
        """
        pass
    
    def _load_model(self):
        """Load model.
        """
        pass


class DataLoaderFCI(DataLoader):
    """
    1D data loader to load FCI data from .npy file 
    """
    
    def __init__(self, dataset_path, result_folder=None):
        super().__init__(dataset_path, result_folder)
        
    def apply_mask(self,filter):
        key = list(filter.keys())[0]
        gain_val = np.array(self.dataset['values'][key])
        mask = gain_val >= filter[key]

        for key in self.dataset['values'].keys():
            self.dataset['values'][key] = np.array(self.dataset['values'][key])[mask]
        
        self.dataset['data'] = np.array(self.dataset['data'])[mask]
        self.dataset['name'] = np.array(self.dataset['name'])[mask]

        return None

    def _load_data(self) -> dict:
        """Load dataset from .npy file.
        """

        loaded_dataset = np.load(self.dataset_path, allow_pickle=True).item()
        
        self.vae_norm = np.max(loaded_dataset["data"])
        
        self.gain_norm = {}
        for key in loaded_dataset["values"]:
            self.gain_norm[key] = np.max(loaded_dataset["values"][key])
        
        self.dataset = loaded_dataset
    

    def _load_model(self) -> dict:
        """Load model.
        returns:
            encoder (tf.keras.models): Encoder model.
            decoder (tf.keras.models): Decoder model.
            latent_gain (dict(str, tf.keras.models)): Dictionary of the gain ANNs for prediction.
            latent_space (np.array): Latent space.
        """
        def below_10_percent(y_true, y_pred):
            absolute_error = tf.abs(y_true - y_pred)
            relative_error = absolute_error / tf.maximum(tf.abs(y_true), 1e-7) # Add a small epsilon to prevent division by zero
            below_10_percent = tf.reduce_mean(tf.cast(relative_error < 0.1, tf.float32)) * 100
            return below_10_percent
        
        encoder = tf.keras.models.load_model(os.path.join( self.result_folder, "model-encoder.keras"),
                                        custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})

        decoder = tf.keras.models.load_model(os.path.join( self.result_folder, "model-decoder.keras"),
                                        custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})
        
        latent_gain = {}
        l_dir = [item for item in os.listdir(self.result_folder+'/values') if not item.startswith('.')]
        for gain in l_dir:
            try:
                latent_gain[gain] = tf.keras.models.load_model(os.path.join(self.result_folder+'/values' , 
                                                                            f"{gain}/model.keras"),custom_objects={'below_10_percent':below_10_percent})
            except:
                print(f'The folder {gain} is not a valid directory. Skipped')

        latent_space = np.loadtxt(os.path.join(self.result_folder,"latent_z.txt"))
        
        self.model = {"encoder": encoder, 
             "decoder": decoder,
             "latent_gain": latent_gain,
             "latent_space": latent_space}
