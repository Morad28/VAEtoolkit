import numpy as np
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
        self.tf_dataset = None
        self._load_data()
        if result_folder is not None: self._load_model()
        
    def pipeline(self,**kwargs):
        """Pipeline for data loading and preprocessing.
        """
        pass
                
    def get_data(self):
        """Get data.
        """
        return self.dataset
    
    def get_tf_dataset(self):
        return self.tf_dataset
    
    def get_model(self):
        return self.model
    
    def to_dataset(self, batch_size, shuffle=True, split=0.8):
        """ Shuffle, split, and convert to tf.data.Dataset object.
        
        Args:
            data: A single array or a tuple (x, y) where x is the features and y is the labels.
            batch_size: The batch size for the dataset.
            shuffle: Whether to shuffle the data before splitting.
            split: The proportion of the dataset to use for training (default is 0.8).
        
        Returns:
            train_dataset: A tf.data.Dataset object for training.
            test_dataset: A tf.data.Dataset object for testing.
        """
        pass
            
    
    def _to_tensorflow_dataset(self,data):
        if isinstance(data, tuple):
            x, y = data
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(data)
        return(dataset)
    
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
        
    def pipeline(self,**kwargs):
        """Apply preprocessing and transform dataset to tensorflow dataset.
        """
        batch_size = kwargs.get("batch_size", 128)
        shuffle = kwargs.get("shuffle", True)
        split = kwargs.get("kwargs", 0.8)
        filter = kwargs.get("filter", None)
        proprecessing = kwargs.get("proprecessing", True)
        
        if proprecessing:
            if filter is not None: self.apply_mask(filter)   
            self._normalize_data()
        
        train_dataset, test_dataset = self.to_dataset(batch_size, shuffle=shuffle, split=split)
        
        self.tf_dataset = {
            "train_x"   : train_dataset.map(lambda x,y:x),
            "val_x"     : test_dataset.map(lambda x,y:x),
            "train_y"   : train_dataset.map(lambda x,y:y),
            "val_y"     : test_dataset.map(lambda x,y:y)
        }
        
        
    def to_dataset(self, batch_size = 128, shuffle=True, split=0.8):
        """ Shuffle, split, and convert to tf.data.Dataset object.
        
        Args:
            data: A single array or a tuple (x, y) where x is the features and y is the labels.
            batch_size: The batch size for the dataset.
            shuffle: Whether to shuffle the data before splitting.
            split: The proportion of the dataset to use for training (default is 0.8).
        
        Returns:
            train_dataset: A tf.data.Dataset object for training.
            test_dataset: A tf.data.Dataset object for testing.
        """
        x = self.dataset['data']
        y = self.dataset['values']['gain']
        
        dataset = self._to_tensorflow_dataset((x,y))
        

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(dataset))
        
        # Split the dataset into training and testing
        if split > 0:
            dataset_size = len(dataset)
            train_size = int(dataset_size * split)
            
            train_dataset = dataset.take(train_size)
            test_dataset = dataset.skip(train_size)
            
            # Batch and prefetch the datasets
            train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            
        if split == 0:
            # Batch and prefetch the datasets
            train_dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            test_dataset = None
        
        return train_dataset, test_dataset
    
          
    def _normalize_data(self):
        """Normalize data to be in the range [-1, 1].
        
        Returns:
            normalized_data: The normalized data.
        """
        # Normalize data to be in the range [-1, 1]
        self.dataset['data'] = np.array(self.dataset['data'])
        self.dataset['values']['gain'] = np.array(self.dataset['values']['gain'])
        
        self.dataset['data'] = self.dataset['data'] / np.max(self.dataset['data'])
        for key in self.dataset['values'].keys():
            self.dataset['values'][key] = np.array(self.dataset['values'][key]) / (np.max(self.dataset['values'][key]))
        
    def apply_mask(self,filter):
        key = list(filter.keys())[0]
        gain_val = np.array(self.dataset['values'][key])
        mask = gain_val >= filter[key]

        for key in self.dataset['values'].keys():
            self.dataset['values'][key] = np.array(self.dataset['values'][key])[mask]
        
        self.dataset['data'] = np.array(self.dataset['data'])[mask]
        self.dataset['name'] = np.array(self.dataset['name'])[mask]

        return None     

    def get_shape(self,i):
        return self.dataset['data'].shape[i]

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
            encoder (tf.keras.Model): Encoder model.
            decoder (tf.keras.Model): Decoder model.
            latent_gain (dict(str, tf.keras.Model)): Dictionary of the gain ANNs for prediction.
            latent_space (np.array): Latent space.
        """
        def below_10_percent(y_true, y_pred):
            absolute_error = tf.abs(y_true - y_pred)
            relative_error = absolute_error / tf.maximum(tf.abs(y_true), 1e-7) # Add a small epsilon to prevent division by zero
            below_10_percent = tf.reduce_mean(tf.cast(relative_error < 0.1, tf.float32)) * 100
            return below_10_percent
        
        encoder = tf.keras.Model.load_model(os.path.join( self.result_folder, "model-encoder.keras"),
                                        custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})

        decoder = tf.keras.Model.load_model(os.path.join( self.result_folder, "model-decoder.keras"),
                                        custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})
        
        latent_gain = {}
        l_dir = [item for item in os.listdir(self.result_folder+'/values') if not item.startswith('.')]
        for gain in l_dir:
            try:
                latent_gain[gain] = tf.keras.Model.load_model(os.path.join(self.result_folder+'/values' , 
                                                                            f"{gain}/model.keras"),custom_objects={'below_10_percent':below_10_percent})
            except:
                print(f'The folder {gain} is not a valid directory. Skipped')

        latent_space = np.loadtxt(os.path.join(self.result_folder,"latent_z.txt"))
        
        self.model = {"encoder": encoder, 
             "decoder": decoder,
             "latent_gain": latent_gain,
             "latent_space": latent_space}
