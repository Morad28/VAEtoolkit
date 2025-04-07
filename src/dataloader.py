import numpy as np
import os
import tensorflow as tf
from src.vae_class import Sampling, SamplingLayer
import numpy as np

class DataLoader:
    """
    Abstract class for loading and preprocessing data.
    
    """
    def __init__(self, config, result_folder=None):
        self.config = config
        self.dataset_path = config["dataset_path"]
        self.result_folder = result_folder
        self.dataset = None
        self.model = None
        self.tf_dataset = None
        self.latent_space = None
        self.vae_norm = 1.
        self.dataset = self._load_data()
        if result_folder is not None: 
            self.model = self._load_model()

    def _load_data(self) -> dict:
        """
        Load dataset.
        """
        raise NotImplementedError("This is an abstract class. Please use a concrete implementation.")
    
    def _load_model(self) -> dict:
        """
        Load model. 
        """
        raise NotImplementedError("This is an abstract class. Please use a concrete implementation.")

    def get_x_y(self):
        """Get x data and y data (gain for FCI, labels, ...)
        """
        raise NotImplementedError("This is an abstract class. Please use a concrete implementation.")
    
    def preprocessing(self):
        """preprocessing of data. Normalization or other stuff.
        """
        pass
    
    def pipeline(self):
        """Apply preprocessing and transform dataset to tensorflow dataset. 
        Stores dataset in self.tf_dataset for vae training.
        """
        batch_size      = self.config.get("batch_size", 128)
        shuffle         = self.config.get("shuffle", True)
        split           = self.config.get("split", 0.8)
        
        self.preprocessing()
        
        train_dataset, test_dataset = self.to_dataset(batch_size, shuffle=shuffle, split=split)
        
        self.tf_dataset = {
            "train_x"   : train_dataset.map(lambda x,y:x),
            "val_x"     : test_dataset.map(lambda x,y:x),
            "train_y"   : train_dataset.map(lambda x,y:y) if 0 < split < 1 else None,
            "val_y"     : test_dataset.map(lambda x,y:y) if 0 < split < 1 else None
        }

            
    def get_data(self):
        """Get data.
        """
        return self.dataset
    
    def get_tf_dataset(self):
        return self.tf_dataset
    
    def get_model(self):
        return self.model
    
    
    def to_dataset(self, batch_size = 128, shuffle=True, split=0.8, dataset = None):
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
        
        if dataset is None:
            x, y = self.get_x_y()
            dataset = self.to_tensorflow_dataset((x,y))
        
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
            
    
    def to_tensorflow_dataset(self,data):
        if isinstance(data, tuple):
            x, y = data
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(data)
        return(dataset)
    


class DataLoaderFCI(DataLoader):
    """
    1D data loader to load FCI data from .npy file 
    """
    
    def __init__(self, config, result_folder=None):
        self._preprocessed = False
        self.gain_norm = {}
        self.vae_norm = 1.
        super().__init__(config, result_folder)
        
    def preprocessing(self):
        filter          = self.config.get("filter", None)
        if filter is not None: self.apply_mask(filter)   
        if not self._preprocessed:
            self._normalize_data()
            self._preprocessed = True
        
        
    def get_x_y(self, values = 'gain'):
        x = self.dataset['data']
        y = self.dataset['values'][values]

        return(x,y)
        
    def _normalize_data(self):
        """Normalize data to be in the range [0, 1].
        
        Returns:
            normalized_data: The normalized data.
        """
        # Normalize data to be in the range [-1, 1]
        self.dataset['data'] = np.array(self.dataset['data'])
        self.dataset['values']['gain'] = np.array(self.dataset['values']['gain'])
        
        self.dataset['data'] = self.dataset['data'] / self.vae_norm
        
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
        
        if self.dataset['data'].shape[0] == 0:
            raise ValueError(f"No data left after applying filter. {gain_val} might be to high.")


    def get_shape(self):
        if len(self.dataset['data'].shape[1:]) == 1:
            return (self.dataset['data'].shape[1], 1)
        return self.dataset['data'].shape[1:]

    def _load_data(self) -> dict:
        """Load dataset from .npy file.
        """

        loaded_dataset = np.load(self.dataset_path, allow_pickle=True).item()
        
        loaded_dataset['data'] = np.array(loaded_dataset['data'])
        loaded_dataset['values']['gain'] = np.array(loaded_dataset['values']['gain'])
        
        if len(loaded_dataset["data"].shape) == 2:
            self.vae_norm  =  np.max(loaded_dataset["data"])
        elif len(loaded_dataset["data"].shape) == 3:
            self.vae_norm  =  np.max(loaded_dataset["data"], axis=(0, 1), keepdims=True)
        
        for key in loaded_dataset["values"]:
            self.gain_norm[key] = np.max(loaded_dataset["values"][key])
        
        return(loaded_dataset)
    
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
        
        model = {
            "encoder": encoder, 
            "decoder": decoder,
            "latent_space": latent_space,
            "latent_gain": latent_gain # Gain network
        }
        
        return(model)


class DataLoaderMNIST(DataLoader):
    def __init__(self, config, result_folder = None, take = -1):
        self.take = take
        self.vae_norm = 255.
        super().__init__(config, result_folder)
        
    
    def _load_data(self) -> dict:
        """Load dataset from mnist database.
        """
        filter = self.config.get("filter", None)
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        
        x_train = x_train[:self.take]
        y_train = y_train[:self.take]
        
        if filter is not None:
            mask = np.isin(y_train, filter)
            x_train = x_train[mask]
            y_train = y_train[mask]
        
        # normalize data
        ''' Z-score normalization
        mean = np.mean(x_train)
        std = np.std(x_train)
        print(f"Mean: {mean}, Std: {std}")
        x_train = (x_train - mean) / std'''

        # Min-max normalization
        x_train = x_train / np.max(x_train)
        
        dataset = {
            "data": x_train,
            "labels": y_train
        }

        return(dataset)
    
    def _load_model(self):
        """Load model.
        
        returns:
            encoder (tf.keras.Model): Encoder model.
            decoder (tf.keras.Model): Decoder model.
            latent_gain (dict(str, tf.keras.Model)): Dictionary of the gain ANNs for prediction.
        """
        encoder = tf.keras.models.load_model(os.path.join( self.result_folder, "model-encoder.keras"),
                                        custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})

        decoder = tf.keras.models.load_model(os.path.join( self.result_folder, "model-decoder.keras"),
                                        custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})
        
        latent_space = np.loadtxt(os.path.join(self.result_folder,"latent_z.txt"))
        
        model = {
            "encoder": encoder, 
            "decoder": decoder,
            "latent_space": latent_space,
        }

        return(model)
    
    def get_x_y(self):
        x = self.dataset['data']
        y = self.dataset['labels']

        return(x,y)
    
    def preprocessing(self):
        self.dataset["data"] = np.expand_dims(self.dataset["data"], axis=-1) / self.vae_norm
        
    def get_shape(self):
        return self.dataset["data"].shape[1:] 
        
        
        
