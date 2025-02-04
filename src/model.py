import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, Model
from keras.layers import Input, Dense, Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose, MaxPooling2D, Flatten, Reshape, MaxPooling1D
from src.vae_class import VAE, Sampling



class ModelSelector:
    def __init__(self):
        self.model_name = None
    
    def select(self, **kwargs):
        self.vae = kwargs.get('vae', None)
        self.gain = kwargs.get('gain', None)
        
    def get_model(self,input_shape=(512,1), latent_dim=5,r_loss=0., k_loss=1., gain_loss=0.):
        s = {}
        if self.vae == '1D-FCI':
            s["vae"] = self._get_1d_vae(input_shape=input_shape, latent_dim=latent_dim,r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss)
        if self.vae == '2D-MNIST':
            s["vae"] = self._get_2d_vae(input_shape=input_shape, latent_dim=latent_dim,r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss)
        if self.gain == '12MLP':
            s["mlp"] = self._get_gain_network_12_mlp(latent_dim)
        if not bool(s):
            raise ValueError('Model name not recognized')
        return s
        
    
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

    def _get_1d_vae_2_channel(self, input_shape=(512,2), latent_dim=5, r_loss=0., k_loss=1., gain_loss=0.):
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
        inputs = Input(shape=input_shape)  # Adjusted for 2 input channels
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
   
    def _get_2d_vae(self,input_shape=(28, 28, 1), latent_dim=5, r_loss=0., k_loss=1., gain_loss=0.):
        """For training on 2D image input (28x28)

        Args:
            input_shape (tuple, optional): Input shape of the image. Defaults to (28, 28, 1).
            latent_dim (int, optional): Dimension of the latent space. Defaults to 5.
            r_loss (float, optional): Weight for reconstruction loss. Defaults to 0..
            k_loss (float, optional): Weight for KL divergence loss. Defaults to 1..
            gain_loss (float, optional): Weight for gain loss. Defaults to 0..

        Returns:
            Model: autoencoder
        """
        inputs = Input(shape=input_shape)
        x         = layers.Conv2D(32, 3, strides=1, padding="same", activation="relu")(inputs)
        x         = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x         = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x         = layers.Conv2D(64, 3, strides=1, padding="same", activation="relu")(x)
        x         = layers.Flatten()(x)
        x         = layers.Dense(16, activation="relu")(x)

        z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z         = Sampling()([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.compile()

        latent_inputs = Input(shape=(latent_dim,))
        x       = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x       = layers.Reshape((7, 7, 64))(x)
        x       = layers.Conv2DTranspose(64, 3, strides=1, padding="same", activation="relu")(x)
        x       = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x       = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        decoded = layers.Conv2DTranspose(1,  3, padding="same", activation="sigmoid")(x)
        decoder = Model(latent_inputs, decoded, name='decoder')
        decoder.compile()

        print(encoder.summary())
        print(decoder.summary())

        autoencoder = VAE(encoder, decoder, [r_loss, k_loss, gain_loss])
        return autoencoder, encoder, decoder 
    
    def _get_gain_network_12_mlp(self,input_shape, output_shape = 1):

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
        decoded = Dense(output_shape)(x)
            
        model = tf.keras.Model(inputs,  decoded, name='12MLP')
        return(model) 