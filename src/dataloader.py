import numpy as np
import os
import tensorflow as tf
from src.vae_class import Sampling, SamplingLayer
import numpy as np

class DataLoader:
    """
    Abstract base class for loading and preprocessing data for Variational Autoencoder (VAE) training.

    This class provides the framework for data loading, preprocessing, and conversion to TensorFlow datasets.
    Concrete implementations should override the abstract methods for specific data types.

    Attributes:
        config (dict): Configuration dictionary containing dataset and training parameters.
        dataset_path (str): Path to the dataset file.
        result_folder (str): Path to the folder containing trained model results (for visualization).
        dataset (dict): Loaded dataset in dictionary format.
        model (dict): Loaded model components (encoder, decoder, etc.).
        tf_dataset (dict): Processed data in TensorFlow dataset format.
        latent_space (np.array): Latent space representation of the data.
        vae_norm (float): Normalization factor for VAE input data.
    """
    
    def __init__(self, config, result_folder=None):
        """Initialize the DataLoader with configuration and optional result folder.

        Args:
            config (dict): Configuration dictionary containing dataset and training parameters.
            result_folder (str, optional): Path to folder containing trained model results. 
                Defaults to None for training mode.
        """
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
        """Load dataset from source.
        
        Returns:
            dict: Dictionary containing the loaded dataset.
            
        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError("This is an abstract class. Please use a concrete implementation.")
    
    def _load_model(self) -> dict:
        """Load trained model components.
        
        Returns:
            dict: Dictionary containing model components (encoder, decoder, etc.).
            
        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError("This is an abstract class. Please use a concrete implementation.")

    def get_x_y(self):
        """Get input features and target values/labels.
        
        Returns:
            tuple: (x, y) where x is the input features and y is the target values/labels.
            
        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError("This is an abstract class. Please use a concrete implementation.")
    
    def preprocessing(self):
        """Preprocess the loaded data (normalization, filtering, etc.).
        
        This base implementation does nothing. Subclasses should override for specific preprocessing.
        """
        pass
    
    def pipeline(self):
        """Apply preprocessing and transform dataset to TensorFlow dataset format.
        
        Stores processed dataset in self.tf_dataset with keys:
        - "train_x": training features
        - "val_x": validation features
        - "train_y": training targets (if split < 1)
        - "val_y": validation targets (if split < 1)
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
        """Get the loaded dataset.

        Returns:
            dict: The loaded dataset dictionary.
        """
        return self.dataset
    
    def get_tf_dataset(self):
        """Get the processed TensorFlow dataset.
        
        Returns:
            dict: Dictionary containing TensorFlow datasets for training and validation.
        """
        return self.tf_dataset
    
    def get_model(self):
        """Get the loaded model components.
        
        Returns:
            dict: Dictionary containing model components (encoder, decoder, etc.).
        """
        return self.model
    
    
    def to_dataset(self, batch_size = 128, shuffle=True, split=0.8, dataset = None):
        """Convert data to TensorFlow Dataset object with optional shuffling and splitting.
        
        Args:
            batch_size (int, optional): Batch size for the dataset. Defaults to 128.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            split (float, optional): Proportion of data to use for training (0-1). Defaults to 0.8.
            dataset (tf.data.Dataset, optional): Existing dataset to process. If None, uses get_x_y().
            
        Returns:
            tuple: (train_dataset, test_dataset) TensorFlow Dataset objects.
                   If split=0, test_dataset will be None.
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
        """Convert numpy arrays to TensorFlow Dataset.
        
        Args:
            data: Either a tuple (x, y) of features and labels, or a single array of features.
            
        Returns:
            tf.data.Dataset: The converted TensorFlow Dataset.
        """
        if isinstance(data, tuple):
            x, y = data
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(data)
        return(dataset)
    


class DataLoaderFCI(DataLoader):
    """
    Data loader for 1D FCI (Fusion Confinement Index) data from .npy files.
    
    Inherits from DataLoader and implements specific methods for FCI data handling.
    
    Attributes:
        _preprocessed (bool): Flag indicating if data has been preprocessed.
        gain_norm (dict): Dictionary containing normalization factors for gain values.
        vae_norm (float): Normalization factor for VAE input data.
    """
    
    def __init__(self, config, result_folder=None):
        """Initialize the FCI DataLoader.
        
        Args:
            config (dict): Configuration dictionary.
            result_folder (str, optional): Path to folder containing trained model results.
        """
        self._preprocessed = False
        self.gain_norm = {}
        self.vae_norm = 1.
        super().__init__(config, result_folder)
        
    def preprocessing(self):
        """Apply preprocessing steps to FCI data.
        
        Includes optional filtering and normalization.
        """
        filter          = self.config.get("filter", None)
        if filter is not None: self.apply_mask(filter)   
        if not self._preprocessed:
            self._normalize_data()
            self._preprocessed = True
        
        
    def get_x_y(self, values = 'gain'):
        """Get input features and specified target values.
        
        Args:
            values (str, optional): Key specifying which target values to return. Defaults to 'gain'.
            
        Returns:
            tuple: (x, y) where x is the input features and y is the target values.
        """
        x = self.dataset['data']
        y = self.dataset['values'][values]

        return(x,y)
        
    def _normalize_data(self):
        """Normalize data to be in the range [0, 1] for both input features and target values."""
        # Normalize data to be in the range [-1, 1]
        self.dataset['data'] = np.array(self.dataset['data'])
        self.dataset['values']['gain'] = np.array(self.dataset['values']['gain'])
        
        self.dataset['data'] = self.dataset['data'] / self.vae_norm
        
        for key in self.dataset['values'].keys():
            self.dataset['values'][key] = np.array(self.dataset['values'][key]) / (np.max(self.dataset['values'][key]))
        
    def apply_mask(self,filter):
        """Apply filter mask to dataset based on threshold values.
        
        Args:
            filter (dict): Dictionary with key specifying which value to filter on and value as threshold.
            
        Raises:
            ValueError: If no data remains after filtering.
        """
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
        """Get the shape of the input data.
        
        Returns:
            tuple: Shape of the input data with channel dimension added if needed.
        """
        if len(self.dataset['data'].shape[1:]) == 1:
            return (self.dataset['data'].shape[1], 1)
        return self.dataset['data'].shape[1:]

    def _load_data(self) -> dict:
        """Load FCI dataset from .npy file and compute normalization factors.
        
        Returns:
            dict: Loaded dataset dictionary.
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
        """Load trained VAE model components from result folder.
        
        Returns:
            dict: Dictionary containing:
                - "encoder": Encoder model
                - "decoder": Decoder model
                - "latent_space": Latent space representation
                - "latent_gain": Dictionary of trained gain prediction models
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

class DataLoaderGain(DataLoader):
    """
    Incorporates the gain in the data to predict the gain in the VAE model.
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
        
        # Z-scoring normalization
        mean = self.vae_norm["mean"]
        std = self.vae_norm["std"]
        mean_gain = self.vae_norm["mean_gain"]
        std_gain = self.vae_norm["std_gain"]
        self.dataset['data'][:,:-1] = (self.dataset['data'][:,:-1] - mean) / std
        self.dataset['data'][:,-1] = (self.dataset['data'][:,-1] - mean_gain) / std_gain
        gain_weight = self.config["gain_weight"]
        self.dataset['data'][:, -1] = self.dataset['data'][:, -1] * gain_weight


        
    def apply_mask(self,filter):
        key = list(filter.keys())[0]
        gain_val = np.array(self.dataset['values'][key])
        mask = gain_val >= filter[key]
        
        for key in self.dataset['values'].keys():
            self.dataset['values'][key] = np.array(self.dataset['values'][key])[mask]
            
        self.dataset['data'] = np.array(self.dataset['data'])[mask]
        self.dataset['name'] = np.array(self.dataset['name'])[mask]

        print("Numbers of samples that will get filtered out: ", len(self.dataset['data']) - len(mask[mask == True]))
        print("Numbers of samples that will be used: ", len(mask[mask == True]))
        
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
        
        std = np.std(loaded_dataset["data"])
        mean = np.mean(loaded_dataset["data"])
        std_gain = np.std(loaded_dataset["values"]["gain"])
        mean_gain = np.mean(loaded_dataset["values"]["gain"])
        self.vae_norm = {"mean": mean, "std": std, "mean_gain": mean_gain, "std_gain": std_gain}

        for key in loaded_dataset["values"]:
            self.gain_norm[key] = np.max(loaded_dataset["values"][key])
        
        # incorporate the gain in the data
        gain = loaded_dataset['values']['gain']
        gain = np.expand_dims(gain, axis=-1)
        loaded_dataset['data'] = np.concatenate((loaded_dataset['data'], gain), axis=-1)
        
        return(loaded_dataset)
    
    def _load_model(self) -> dict:
        """Load model.
        returns:
            encoder (tf.keras.models): Encoder model.
            decoder (tf.keras.models): Decoder model.
            latent_space (np.array): Latent space.
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



class DataLoaderCoilsMulti(DataLoader):
    """
    Incorporates the gain in the data to predict the gain in the VAE model.
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
        
        
    def get_x_y(self, value = 'cutoff'):
        x = self.dataset['data']
        values = self.config["values"]
        y = []
        for value in values:
            if value not in self.dataset['values'].keys():
                raise ValueError(f"The key {value} is not in the dataset. Please check the filter.")
            else:
                y.append(self.dataset['values'][value])
        y = np.array(y).T
        return(x,y)
        
    def _normalize_data(self):
        """Normalize data to be in the range [0, 1].
        
        Returns:
            normalized_data: The normalized data.
        """
        # Normalize data to be in the range [-1, 1]
        self.dataset['data'] = np.array(self.dataset['data'])
        
        # Z-scoring normalization
        length_values = len(self.dataset['values'].keys())


        self.dataset['data'][:,:-length_values] = (self.dataset['data'][:,:-length_values] - self.vae_norm["profile"]["mean"]) / self.vae_norm["profile"]["std"]
        for i, key in enumerate(self.dataset['values'].keys()):
            self.dataset['data'][:,-length_values+i] = (self.dataset['data'][:,-length_values+i] - self.vae_norm[key]["mean"]) / self.vae_norm[key]["std"]

        values_weight = self.config["gain_weight"]
        self.dataset['data'][:, -length_values:] = self.dataset['data'][:, -length_values:] * values_weight

        
    def apply_mask(self,filter):
        keys = list(filter.keys())
        for key in keys:
            if key not in self.dataset['values'].keys():
                raise ValueError(f"The key {key} is not in the dataset. Please check the filter.")
            val_to_mask = np.array(self.dataset['values'][key])
            mask = val_to_mask >= filter[key]
        
            self.dataset['values'][key] = np.array(self.dataset['values'][key])[mask]
            
            self.dataset['data'] = np.array(self.dataset['data'])[mask]
            self.dataset['name'] = np.array(self.dataset['name'])[mask]

            print("\nNumbers of samples that will get filtered out: ", len(self.dataset['data']) - len(mask[mask == True]))
            print("\nNumbers of samples that will be used: ", len(mask[mask == True]))
        
        if self.dataset['data'].shape[0] == 0:
            raise ValueError(f"No data left after applying filter. Filters might be too high.")


    def get_shape(self):
        if len(self.dataset['data'].shape[1:]) == 1:
            return (self.dataset['data'].shape[1], 1)
        return self.dataset['data'].shape[1:]

    def _load_data(self) -> dict:
        """Load dataset from .npy file.
        """

        loaded_dataset = np.load(self.dataset_path, allow_pickle=True).item()

        self.values = self.config["values"]
        
        loaded_dataset['data'] = np.array(loaded_dataset['data'])
        std = np.std(loaded_dataset["data"])
        mean = np.mean(loaded_dataset["data"])
        self.vae_norm = {"profile": {"mean": mean, "std": std}}
        
        for value in self.values:
            loaded_dataset['values'][value] = np.array(loaded_dataset['values'][value])
            std = np.std(loaded_dataset['values'][value])
            mean = np.mean(loaded_dataset['values'][value])
            self.vae_norm[value] = {"mean": mean, "std": std}
        
        # incorporate the values in the data
        for value in self.values:
            val = loaded_dataset['values'][value]
            val = np.expand_dims(val, axis=-1)
            loaded_dataset['data'] = np.concatenate((loaded_dataset['data'], val), axis=-1)
        
        # remove keys that are not in the values list
        values = list(loaded_dataset['values'].keys())
        for key in values:
            if key not in self.values:
                del loaded_dataset['values'][key]
        
        return(loaded_dataset)
    
    def _load_model(self) -> dict:
        """Load model.
        returns:
            encoder (tf.keras.models): Encoder model.
            decoder (tf.keras.models): Decoder model.
            latent_space (np.array): Latent space.
        """    
        encoder = tf.keras.models.load_model(os.path.join( self.result_folder, "model-encoder.keras"),
                                        custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})

        if self.config["Model"]["vae"] == "COILS-MULTI-OUT":
            decoder0 = tf.keras.models.load_model(os.path.join( self.result_folder, "model-decoder-0.keras"),
                                            custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})
            decoder1 = tf.keras.models.load_model(os.path.join( self.result_folder, "model-decoder-1.keras"),
                                            custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})

            latent_space = np.loadtxt(os.path.join(self.result_folder,"latent_z.txt"))
            
            model = {
                "encoder": encoder, 
                "decoder_cnn": decoder0,
                "decoder_mlp": decoder1,
                "latent_space": latent_space,
            }
        else:
            decoder = tf.keras.models.load_model(os.path.join( self.result_folder, "model-decoder.keras"),
                                            custom_objects={'SamplingLayer': SamplingLayer,'Sampling':Sampling})
            
            latent_space = np.loadtxt(os.path.join(self.result_folder,"latent_z.txt"))
            
            model = {
                "encoder": encoder, 
                "decoder": decoder,
                "latent_space": latent_space,
            }

        
        return(model)