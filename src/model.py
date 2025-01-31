import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, Model
from keras.layers import Input, Dense, Conv1D, Conv1DTranspose, Flatten, Reshape, MaxPooling1D
from src.vae_class import VAE, Sampling



class ModelSelector:
    def __init__(self):
        self.model_name = "1D"
    
    def select(self, model_name):
        self.model_name = model_name
        
    def get_model(self,input_shape=(512,1), latent_dim=5,r_loss=0., k_loss=1., gain_loss=0.):
        if self.model_name == '1D':
            return self._get_1d_vae(input_shape=input_shape, latent_dim=latent_dim,r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss)
        elif self.model_name == '2D':
            return self._get_2d_vae(input_shape=input_shape, latent_dim=latent_dim,r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss)
        elif self.model_name == 'gain_nn':
            return self._get_gain_network(latent_dim)
        else:
            raise ValueError('Model name not recognized')
        
    
    def _get_1d_vae(self, input_shape=(512,1), latent_dim=5,r_loss=0., k_loss=1., gain_loss=0.):
        """For training on 1D FCI target 

        Args:
            input_shape (int, optional): _description_. Defaults to 512.
            latent_dim (int, optional): _description_. Defaults to 5.
            r_loss (_type_, optional): _description_. Defaults to 0..
            k_loss (_type_, optional): _description_. Defaults to 1..
            gain_loss (_type_, optional): _description_. Defaults to 0..

        Returns:
            Model: autoencoder
        """
        inputs = Input(shape=input_shape)
        x = Conv1D(32, 3, activation='leaky_relu', padding='same', strides=2)(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Flatten()(x)

        z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z         = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.compile()

        inputs = Input(shape=(latent_dim,))
        x = Dense   (32*128,  activation='leaky_relu')(inputs)
        x = Reshape((32,128))(x)
        x = Conv1DTranspose(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(32, 3, activation='leaky_relu', padding='same', strides=2)(x)
        decoded = Conv1DTranspose(1, 1, padding='same')(x)
        decoded = Reshape((512,))(decoded)
        decoder = tf.keras.Model(inputs,  decoded, name='decoder')
        decoder.compile()

        print(encoder.summary())
        print(decoder.summary())

        autoencoder = VAE(encoder,decoder, [r_loss,k_loss,gain_loss])
        
        return autoencoder, encoder, decoder

    def _get_2d_vae(self, input_shape=512, latent_dim=5, r_loss=0., k_loss=1., gain_loss=0.):
        """For training on 2D FCI target (non tested yet)

        Args:
            input_shape (int, optional): _description_. Defaults to 512.
            latent_dim (int, optional): _description_. Defaults to 5.
            r_loss (_type_, optional): _description_. Defaults to 0..
            k_loss (_type_, optional): _description_. Defaults to 1..
            gain_loss (_type_, optional): _description_. Defaults to 0..

        Returns:
            Model: autoencoder
        """
        inputs = Input(shape=(input_shape, 2))  # Adjusted for 2 input channels
        x = Conv1D(32, 3, activation='leaky_relu', padding='same', strides=2)(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Flatten()(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.compile()

        inputs = Input(shape=(latent_dim,))
        x = Dense(32 * 128, activation='leaky_relu')(inputs)
        x = Reshape((32, 128))(x)
        x = Conv1DTranspose(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(32, 3, activation='leaky_relu', padding='same', strides=2)(x)
        decoded = Conv1DTranspose(2, 1, padding='same')(x)  # Adjusted output channels to 2
        decoded = Reshape((512, 2))(decoded)
        decoder = tf.keras.Model(inputs, decoded, name='decoder')
        decoder.compile()

        print(encoder.summary())
        print(decoder.summary())

        autoencoder = VAE(encoder, decoder, [r_loss, k_loss, gain_loss])

        return autoencoder, encoder, decoder
    
    def _get_gain_network(self,input_shape):

        inputs = Input(shape=(input_shape,))
        x = Dense(64, activation='leaky_relu')(inputs)
        x = Dense(64, activation='leaky_relu')(x)
        x = Dense(128, activation='leaky_relu')(x)
        x = Dense(128, activation='leaky_relu')(x)
        x = Dense(256, activation='leaky_relu')(x)
        x = Dense(256, activation='leaky_relu')(x)
        x = Dense(256, activation='leaky_relu')(x)
        x = Dense(256, activation='leaky_relu')(x)
        x = Dense(128, activation='leaky_relu')(x)
        x = Dense(128, activation='leaky_relu')(x)
        x = Dense(64, activation='leaky_relu')(x)
        x = Dense(64, activation='leaky_relu')(x)
        decoded = Dense(1)(x)
            
        model = tf.keras.Model(inputs,  decoded, name='decoder')
        return(model) 