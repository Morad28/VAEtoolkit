import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, Model
from keras.layers import Input, Dense, Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose, MaxPooling2D, Flatten, Reshape, MaxPooling1D, UpSampling1D, Concatenate, UpSampling2D, ZeroPadding2D
from src.vae_class import VAE, VAE_MoG, Sampling, SamplingMoG, VAE_multi_decoder, VAE_multi_decoder_encoder, VAE_singleval


class ModelSelector:
    def __init__(self):
        self.model_name = None
    
    def select(self, kwargs):
        self.vae = kwargs.get('vae', None)
        self.gain = kwargs.get('gain', None)
        
    def get_model(self,input_shape=(512,1), latent_dim=5, num_components=3, r_loss=1., k_loss=1.,
                   gain_loss=0., config=None, dataloader=None, physical_penalty_weight=1):
        s = {}
        if self.vae == '1D-FCI':
            s["vae"] = self._get_1d_vae(input_shape=input_shape, latent_dim=latent_dim,r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss)
        if self.vae == '2D-FCI':
            s["vae"] = self._get_2d_vae_fci(input_shape=input_shape, latent_dim=latent_dim,r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss)
        if self.vae == '2D-MNIST' or self.vae == '2D-MNIST-MoG':
            s["vae"] = self._get_2d_vae(input_shape=input_shape, latent_dim=latent_dim, num_components=num_components, r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss)
        if self.vae == '1D-COILS':
            s["vae"] = self._get_1d_vae_coils(input_shape=input_shape, latent_dim=latent_dim,r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss, physical_penalty_weight=physical_penalty_weight)
        if self.vae == '1D-COILS-GAIN':
            s["vae"] = self._get_1d_vae_coils_gain(input_shape=input_shape, latent_dim=latent_dim,
                                                   r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss, config=config,
                                                     dataloader=dataloader, physical_penalty_weight=physical_penalty_weight)
        if self.vae == 'COILS-MULTI':
            s["vae"] = self._get_1d_vae_coils_gain_multi(input_shape=input_shape, latent_dim=latent_dim,
                                                       r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss, config=config,
                                                     dataloader=dataloader, physical_penalty_weight=physical_penalty_weight)
        if self.vae == 'COILS-MULTI-OUT':
            s["vae"] = self._get_1d_vae_coils_gain_multi_out(input_shape=input_shape, latent_dim=latent_dim,
                                                       r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss, config=config,
                                                     dataloader=dataloader, physical_penalty_weight=physical_penalty_weight)
        if self.vae == 'COILS-MULTI-OUT-DUO':
            s["vae"] = self._get_1d_vae_coils_gain_multi_out_duo(input_shape=input_shape, latent_dim=latent_dim,
                                                       r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss, config=config,
                                                     dataloader=dataloader, physical_penalty_weight=physical_penalty_weight)
        if self.vae == 'COILS-MULTI-SINGLEVAL':
            s["vae"] = self._get_1d_vae_coils_gain_multi_out_singleval(input_shape=input_shape, latent_dim=latent_dim,
                                                       r_loss=r_loss, k_loss=k_loss, gain_loss=gain_loss, config=config,
                                                     dataloader=dataloader, physical_penalty_weight=physical_penalty_weight)
        if self.gain == '12MLP':
            s["mlp"] = self._get_gain_network_12_mlp(latent_dim)
        if not bool(s):
            raise ValueError('Model name not recognized')
        return s
    
    def _get_2d_vae_fci(self, input_shape=(512,2), latent_dim=5, r_loss=0., k_loss=1., gain_loss=0.):
        """For training on 1D FCI target 

        Args:
            input_shape (tuple, optional): Input shape of the data. Defaults to (512,2).
            latent_dim (int, optional): Dimensionality of the latent space. Defaults to 5.
            r_loss (float, optional): Reconstruction loss weight. Defaults to 0..
            k_loss (float, optional): KL divergence loss weight. Defaults to 1..
            gain_loss (float, optional): Additional loss weight. Defaults to 0..

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

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.compile()

        # Decoder
        inputs = Input(shape=(latent_dim,))
        x = Dense(32 * 128, activation='leaky_relu')(inputs)
        x = Reshape((32, 128))(x)
        x = Conv1DTranspose(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(32, 3, activation='leaky_relu', padding='same', strides=2)(x)
        
        # Change output channels from 1 to 2 to match input shape
        decoded = Conv1DTranspose(2, 1, padding='same')(x)
        decoded = Reshape((512, 2))(decoded)
        
        decoder = tf.keras.Model(inputs, decoded, name='decoder')
        decoder.compile()

        print(encoder.summary())
        print(decoder.summary())

        autoencoder = VAE(encoder, decoder, [r_loss, k_loss, gain_loss])

        return autoencoder, encoder, decoder
        
    
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
    
    def _get_1d_vae_coils(self, input_shape=(40,1), latent_dim=5,r_loss=0., k_loss=1., gain_loss=0., physical_penalty_weight=1.):
        """For training on 1D coils 

        Args:
            input_shape (int, optional): _description_. Defaults to 40.
            latent_dim (int, optional): _description_. Defaults to 5.
            r_loss (_type_, optional): _description_. Defaults to 0..
            k_loss (_type_, optional): _description_. Defaults to 1..
            gain_loss (_type_, optional): _description_. Defaults to 0..

        Returns:
            Model: autoencoder
        """
        inputs = Input(shape=input_shape)
        
        # Encoder
        x = Conv1D(32, 3, activation='leaky_relu', padding='same', strides=1)(inputs)
        x = MaxPooling1D(pool_size=2)(x)  # Output: (20, 32)
        x = Conv1D(64, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = MaxPooling1D(pool_size=2)(x)  # Output: (10, 64)
        x = Conv1D(128, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = Flatten()(x)  # Output: (1280,)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.compile()

        # Decoder
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(10 * 128, activation='leaky_relu')(latent_inputs)  # Match flattened size
        x = Reshape((10, 128))(x)  # Output: (10, 128)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = UpSampling1D(size=2)(x)  # Output: (20, 64)
        x = Conv1DTranspose(32, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = UpSampling1D(size=2)(x)  # Output: (40, 32)
        decoded = Conv1DTranspose(1, 1, padding='same')(x)  # Output: (40, 1)
        decoded = Reshape((40,))(decoded)
        decoder = tf.keras.Model(latent_inputs, decoded, name='decoder')
        decoder.compile()

        print(encoder.summary())
        print(decoder.summary())

        autoencoder = VAE(encoder,decoder, [r_loss,k_loss,gain_loss], physical_penalty_weight=physical_penalty_weight)
        
        return autoencoder, encoder, decoder
    
    def _get_1d_vae_coils_gain(self, input_shape=(41,1), latent_dim=5,r_loss=0., k_loss=1., gain_loss=0.,
                                physical_penalty_weight=1, config=None, dataloader=None):
        """For training on 1D coils with gain integrated in the VAE
        Obsolete, use _get_1d_vae_coils_gain_multi instead (with only "cutoff" values to replicate the behaviour of this model)

        Args:
            input_shape (int, optional): _description_. Defaults to 41.
            latent_dim (int, optional): _description_. Defaults to 5.
            r_loss (_type_, optional): _description_. Defaults to 0..
            k_loss (_type_, optional): _description_. Defaults to 1..
            gain_loss (_type_, optional): _description_. Defaults to 0..

        Returns:
            Model: autoencoder
        """
        inputs = Input(shape=input_shape)
        
        # Encoder
        x = Conv1D(32, 3, activation='leaky_relu', padding='same', strides=1)(inputs)
        x = MaxPooling1D(pool_size=2)(x)  # Output: (20, 32)
        x = Conv1D(64, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = MaxPooling1D(pool_size=2)(x)  # Output: (10, 64)
        x = Conv1D(128, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = Flatten()(x)  # Output: (1280,)

        x = Dense(128, activation='leaky_relu')(x)  # Add a dense layer before the latent space

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.compile()

        # Decoder
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(11 * 128, activation='leaky_relu')(latent_inputs)  # Match flattened size
        x = Reshape((11, 128))(x)  # Output: (11, 128)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = UpSampling1D(size=2)(x)  # Output: (22, 64)
        x = Conv1DTranspose(32, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = UpSampling1D(size=2)(x)  # Output: (44, 32)
        x = Conv1DTranspose(1, 3, activation='linear', padding='same', strides=1)(x)  # Output: (44, 1)
        decoded = x[:, :41, :]  # Slice to ensure the output shape is (41, 1)
        decoded = Reshape((41,))(decoded)
        decoder = tf.keras.Model(latent_inputs, decoded, name='decoder')
        decoder.compile()

        print(encoder.summary())
        print(decoder.summary())
        
        std = dataloader.vae_norm["std"]
        mean = dataloader.vae_norm["mean"]

        minimum_value = config["min_value"] if config is not None else 0.3
        # normalize to make it correspond to the data
        minimum_value = minimum_value * std + mean

        autoencoder = VAE(encoder,decoder, [r_loss,k_loss,gain_loss], config=config, min_value = minimum_value, physical_penalty_weight=physical_penalty_weight)

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
   
    def _get_2d_vae(self,input_shape=(28, 28, 1), latent_dim=5, num_components=3, r_loss=0., k_loss=1., gain_loss=0.):
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
        x         = layers.BatchNormalization()(x)
        x         = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x         = layers.BatchNormalization()(x)
        x         = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x         = layers.BatchNormalization()(x)
        x         = layers.Conv2D(128, 3, strides=1, padding="same", activation="relu")(x)
        x         = layers.BatchNormalization()(x)
        x         = layers.Flatten()(x)

        if self.vae == '2D-MNIST':
            z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            z         = Sampling()([z_mean, z_log_var])
            encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
            encoder.compile()
        
        if self.vae == '2D-MNIST-MoG':
            # Define z_means, z_log_vars, and z_pis for each component
            z_means = layers.Dense(num_components * latent_dim, name="z_means")(x)
            z_means = layers.Reshape((num_components, latent_dim))(z_means)

            z_log_vars = layers.Dense(num_components * latent_dim, name="z_log_vars")(x)
            z_log_vars = layers.Reshape((num_components, latent_dim))(z_log_vars)

            z_pis = layers.Dense(num_components, activation="softmax", name="z_pis")(x)

            # Sampling from the Mixture of Gaussians
            z = SamplingMoG()([z_means, z_log_vars, z_pis])

            encoder = Model(inputs, [z_means, z_log_vars, z_pis, z], name="encoder")
            encoder.compile()

        latent_inputs = Input(shape=(latent_dim,))
        x       = layers.Dense(7 * 7 * 128, activation="relu")(latent_inputs)
        x       = layers.Reshape((7, 7, 128))(x)
        x       = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
        x       = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x       = layers.Conv2DTranspose(32, 3, strides=1, padding="same", activation="relu")(x)
        decoded = layers.Conv2DTranspose(1,  3, padding="same", activation="sigmoid")(x)
        decoder = Model(latent_inputs, decoded, name='decoder')
        decoder.compile()

        print(encoder.summary())
        print(decoder.summary())
        
        if self.vae == '2D-MNIST':
            autoencoder = VAE(encoder, decoder, [r_loss, k_loss, gain_loss])
        elif self.vae == '2D-MNIST-MoG':
            autoencoder = VAE_MoG(encoder, decoder, [r_loss, k_loss, gain_loss])
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

    def _get_1d_vae_coils_gain_multi(self, input_shape=(41,1), latent_dim=5,r_loss=0., k_loss=1., gain_loss=0.,
                                physical_penalty_weight=1, config=None, dataloader=None):
        """For training on 1D coils with multiple scalar values associated with the profile, using the CNN layers

        Args:
            input_shape (int, optional): _description_. Defaults to 41.
            latent_dim (int, optional): _description_. Defaults to 5.
            r_loss (_type_, optional): _description_. Defaults to 0..
            k_loss (_type_, optional): _description_. Defaults to 1..
            gain_loss (_type_, optional): _description_. Defaults to 0..

        Returns:
            Model: autoencoder
        """
        inputs = Input(shape=input_shape)

        out_shape = input_shape[0]
        
        # Encoder
        x = Conv1D(32, 3, activation='leaky_relu', padding='same', strides=1)(inputs)
        x = MaxPooling1D(pool_size=2)(x)  # Output: (20, 32)
        x = Conv1D(64, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = MaxPooling1D(pool_size=2)(x)  # Output: (10, 64)
        x = Conv1D(128, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = Flatten()(x)  # Output: (1280,)

        x = Dense(128, activation='leaky_relu')(x)  # Add a dense layer before the latent space

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.compile()

        # Decoder
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(11 * 128, activation='leaky_relu')(latent_inputs)  # Match flattened size
        x = Reshape((11, 128))(x)  # Output: (11, 128)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = UpSampling1D(size=2)(x)  # Output: (22, 64)
        x = Conv1DTranspose(32, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = UpSampling1D(size=2)(x)  # Output: (44, 32)
        x = Conv1DTranspose(1, 3, activation='linear', padding='same', strides=1)(x)  # Output: (44, 1)
        decoded = x[:, :out_shape, :]  # Slice to ensure the output shape is (out_shape, 1)
        decoded = Reshape((out_shape,))(decoded)
        decoder = tf.keras.Model(latent_inputs, decoded, name='decoder')
        decoder.compile()

        print(encoder.summary())
        print(decoder.summary())
        
        std = dataloader.vae_norm["profile"]["std"]
        mean = dataloader.vae_norm["profile"]["mean"]

        minimum_value = config["min_value"] if config is not None else 0.
        # normalize to make it correspond to the data
        minimum_value = minimum_value * std + mean

        autoencoder = VAE(encoder,decoder, [r_loss,k_loss,gain_loss], config=config, min_value = minimum_value, physical_penalty_weight=physical_penalty_weight)

        return autoencoder, encoder, decoder

    def _get_1d_vae_coils_gain_multi_out(self, input_shape=(41,1), latent_dim=5,r_loss=0., k_loss=1., gain_loss=0.,
                                physical_penalty_weight=1, config=None, dataloader=None):
        """For training on 1D coils with multiple scalar values associated with the profile, not using the CNN layers for reconstruction of the scalar values

        Args:
            input_shape (int, optional): _description_. Defaults to 41.
            latent_dim (int, optional): _description_. Defaults to 5.
            r_loss (_type_, optional): _description_. Defaults to 0..
            k_loss (_type_, optional): _description_. Defaults to 1..
            gain_loss (_type_, optional): _description_. Defaults to 0..

        Returns:
            Model: autoencoder, with two decoders: one for the profile and one for the scalar values
        """
        inputs = Input(shape=input_shape)
        len_values = len(config["values"])

        profile = inputs[:,:-len_values,:]
        vals = inputs[:,-len_values:,:]
        vals = Flatten()(vals)

        out_shape = input_shape[0] - len_values
        
        # Encoder
        x = Conv1D(32, 3, activation='leaky_relu', padding='same', strides=1)(profile)
        x = MaxPooling1D(pool_size=2)(x)  # Output: (20, 32)
        x = Conv1D(64, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = MaxPooling1D(pool_size=2)(x)  # Output: (10, 64)
        x = Conv1D(128, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = Flatten()(x)  # Output: (1280,)
        #x = Concatenate()([x,vals])

        x = Dense(128, activation='leaky_relu')(x)  # Add a dense layer before the latent space

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.compile()

        # Decoder CNN (profile)
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(10 * 128, activation='leaky_relu')(latent_inputs)  # Match flattened size
        # remove the values from the output
        x = Reshape((10, 128))(x)  # Output: (10, 128)
        x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = UpSampling1D(size=2)(x)  # Output: (20, 64)
        x = Conv1DTranspose(32, 3, activation='leaky_relu', padding='same', strides=1)(x)
        x = UpSampling1D(size=2)(x)  # Output: (40, 32)
        x = Conv1DTranspose(1, 3, activation='linear', padding='same', strides=1)(x)  # Output: (40, 1)
        decoded = Reshape((out_shape,))(x)
        # Concatenate the values to the output
        decoder_cnn = tf.keras.Model(latent_inputs, decoded, name='decoder')
        decoder_cnn.compile()

        x2 = Dense(64, activation='leaky_relu')(latent_inputs)
        x2 = Dense(128, activation='leaky_relu')(x2)
        predictions = Dense(len_values, activation='linear')(x2)
        predictions = Reshape((len_values,))(predictions)
        decoder_mlp = tf.keras.Model(latent_inputs, predictions, name='decoder2')
        decoder_mlp.compile()

        print(encoder.summary())
        print(decoder_cnn.summary())
        print(decoder_mlp.summary())
        
        std = dataloader.vae_norm["profile"]["std"]
        mean = dataloader.vae_norm["profile"]["mean"]

        minimum_value = config["min_value"] if config else 0.
        # normalize to make it correspond to the data
        minimum_value = minimum_value * std + mean

        autoencoder = VAE_multi_decoder(encoder, [decoder_cnn, decoder_mlp], [r_loss,k_loss,gain_loss], config=config, min_value = minimum_value, physical_penalty_weight=physical_penalty_weight)

        return autoencoder, encoder, decoder_cnn, decoder_mlp
    

    def _get_1d_vae_coils_gain_multi_out_duo(self, input_shape=(41,1), latent_dim=5,r_loss=0., k_loss=1., gain_loss=0.,
                                physical_penalty_weight=1, config=None, dataloader=None):
        """For training on 1D coils with multiple scalar values associated with the profile, not using the CNN layers for the scalar values for both
        reconstruction and encoding

        Args:
            input_shape (int, optional): _description_. Defaults to 41.
            latent_dim (int, optional): _description_. Defaults to 5.
            r_loss (_type_, optional): _description_. Defaults to 0..
            k_loss (_type_, optional): _description_. Defaults to 1..
            gain_loss (_type_, optional): _description_. Defaults to 0..

        Returns:
            Model: autoencoder, with two decoders: one for the profile and one for the scalar values,
            and two encoders: one for the profile and one for the scalar values
        """
        inputs = Input(shape=input_shape)
        len_values = len(config["values"])

        profile = inputs[:,:-len_values,:]
        vals = inputs[:,-len_values:,:]
        vals = Flatten()(vals)

        if config["profile_types"] == 2:
            out_shape = (input_shape[0] - len_values) // 2
            # Encoder CNN
            # reshape the input to have 1 channel
            x = Reshape((out_shape, 2, 1))(profile) # batch size, height, width, channels (,100,2,1)
            # the kernels are applied to all channels and the results are summed
            # add two columns of padding to the input
            x = ZeroPadding2D(padding=(1, 1))(x) # (,102,4,1)
            # this kernel size will allow for the creation of 3 columns, one that focuses on the left profile,
            # one on the right profile and one the combination of both
            x = Conv2D(32, (3,2), activation='leaky_relu', padding='valid', strides=1)(x)  # (,100,3,32)
            x = MaxPooling2D(pool_size=(2,1))(x) # (,50,3,32)
            x = Conv2D(64, (3,2), activation='leaky_relu', padding='valid', strides=1)(x) # (,48,2,64)
            x = MaxPooling2D(pool_size=(2,1))(x) # (,24,2,64)
            x = Conv2D(128, (3,2), activation='leaky_relu', padding='valid', strides=1)(x) # (,22,1,128)
            x = Flatten()(x) # (,2816,)
        else:
            out_shape = input_shape[0] - len_values
            # Encoder CNN
            x = Conv1D(32, 3, activation='leaky_relu', padding='same', strides=1)(profile)
            x = MaxPooling1D(pool_size=2)(x)  # Output: (20, 32)
            x = Conv1D(64, 3, activation='leaky_relu', padding='same', strides=1)(x)
            x = MaxPooling1D(pool_size=2)(x)  # Output: (10, 64)
            x = Conv1D(128, 3, activation='leaky_relu', padding='same', strides=1)(x)
            x = Flatten()(x)  # Output: (1280,)


        encoder_cnn = keras.Model(profile, x, name="encoder_cnn")
        encoder_cnn.compile()

        # Encoder MLP (values)
        x2 = Dense(64, activation='leaky_relu')(vals)
        x2 = Dense(config["gain_latent_size"], activation='leaky_relu')(x2)
        x2 = Flatten()(x2)  # Flatten the values
        encoder_mlp = keras.Model(vals, x2, name="encoder_mlp")
        encoder_mlp.compile()

        input_latent = Concatenate()([x,x2])
        concat = Dense(128, activation='leaky_relu')(input_latent)  # Add a dense layer before the latent space
        z_mean = layers.Dense(latent_dim, name="z_mean")(concat)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(concat)
        z = Sampling()([z_mean, z_log_var])
        encoder_latent = keras.Model(input_latent, [z_mean, z_log_var, z], name="encoder_latent")
        encoder_latent.compile()

        # Decoder CNN (profile)
        latent_inputs = Input(shape=(latent_dim,))
        # remove the values from the output
        if config["profile_types"] == 2:
            x = Dense(2816, activation='leaky_relu')(latent_inputs)  # Match flattened size
            x = Reshape((22, 1, 128))(x)
            x = Conv2DTranspose(64, (3,2), activation='leaky_relu', padding='valid', strides=1)(x)  # (,24,2,64)
            x = UpSampling2D(size=(2,1))(x)  # Output: (48,2,64)
            x = ZeroPadding2D(padding=(1, 0))(x) # (,50,2,64)
            x = Conv2DTranspose(32, (3,2), activation='leaky_relu', padding='same', strides=1)(x) # (,50,2,32)
            x = UpSampling2D(size=(2,1))(x)  # Output: (100,2,32)
            x = Conv2DTranspose(1, (3,2), activation='linear', padding='same', strides=1)(x) # (,100,2,1)
            x = Reshape((100, 2))(x)
            decoded = Reshape((out_shape*2,))(x)
        else:
            x = Dense(3200, activation='leaky_relu')(latent_inputs)  # Match flattened size
            x = Reshape((25, 128))(x)  # Output: (25, 128)
            x = Conv1DTranspose(64, 3, activation='leaky_relu', padding='same', strides=1)(x) # Output: (25, 64)
            x = UpSampling1D(size=2)(x)  # Output: (50, 64)
            x = Conv1DTranspose(32, 3, activation='leaky_relu', padding='same', strides=1)(x) # Output: (100, 32)
            x = UpSampling1D(size=2)(x)  # Output: (100, 32)
            x = Conv1DTranspose(1, 3, activation='linear', padding='same', strides=1)(x)  # Output: (100, 1)
            decoded = Reshape((out_shape,))(x)
            # Concatenate the values to the output
        decoder_cnn = tf.keras.Model(latent_inputs, decoded, name='decoder')
        decoder_cnn.compile()

        # Decoder MLP (values)
        x2 = Dense(64, activation='leaky_relu')(latent_inputs)
        x2 = Dense(128, activation='leaky_relu')(x2)
        predictions = Dense(len_values, activation='linear')(x2)
        predictions = Reshape((len_values,))(predictions)
        decoder_mlp = tf.keras.Model(latent_inputs, predictions, name='decoder2')
        decoder_mlp.compile()

        print(encoder_cnn.summary())
        print(encoder_mlp.summary())
        print(encoder_latent.summary())
        print(decoder_cnn.summary())
        print(decoder_mlp.summary())
        
        std = dataloader.vae_norm["profile"]["std"]
        mean = dataloader.vae_norm["profile"]["mean"]

        minimum_value = config["min_value"] if config else 0.
        # normalize to make it correspond to the data
        minimum_value = minimum_value * std + mean

        autoencoder = VAE_multi_decoder_encoder([encoder_cnn, encoder_mlp, encoder_latent], [decoder_cnn, decoder_mlp], [r_loss,k_loss,gain_loss], config=config, min_value = minimum_value, physical_penalty_weight=physical_penalty_weight)

        return autoencoder, encoder_cnn, encoder_mlp, encoder_latent, decoder_cnn, decoder_mlp



    def _get_1d_vae_coils_gain_multi_out_singleval(self, input_shape=(3,1), latent_dim=5,r_loss=0., k_loss=1., gain_loss=0.,
                                physical_penalty_weight=1, config=None, dataloader=None):
        """For training on 1D coils not using CNN layers, with coil profiles having constant values (e.g. pitch, radius) and scalar values associated with the profile

        Args:
            input_shape (int, optional): _description_. Defaults to 3.
            latent_dim (int, optional): _description_. Defaults to 5.
            r_loss (_type_, optional): _description_. Defaults to 0..
            k_loss (_type_, optional): _description_. Defaults to 1..
            gain_loss (_type_, optional): _description_. Defaults to 0..

        Returns:
            Model: autoencoder
        """
        inputs = Input(shape=input_shape)
        inputs_zero = inputs
        len_values = len(config["values"])

        profile = inputs[:,:-len_values,:]
        vals = inputs[:,-len_values:,:]
        vals = Flatten()(vals)

        if config["profile_types"] == 2:
            len_profile = (input_shape[0] - len_values) //2
            pitch_val = profile[:,0,:]
            radius_val = profile[:,len_profile,:]
            inputs = Concatenate()([pitch_val, radius_val, vals])
        else:
            profile_val = profile[:,0,:]
            inputs = Concatenate()([profile_val, vals])

        x = Dense(64, activation='leaky_relu')(inputs)
        x = Dense(128, activation='leaky_relu')(x)
        x = Flatten()(x)

    
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs_zero, [z_mean, z_log_var, z], name="encoder")
        encoder.compile()

        
        latent_inputs = Input(shape=(latent_dim,))

        # Decoder MLP (values)
        x = Dense(64, activation='leaky_relu')(latent_inputs)
        x = Dense(128, activation='leaky_relu')(x)

        length_predictions = len_values + config["profile_types"]
        predictions = Dense(length_predictions, activation='linear')(x)
        predictions = Reshape((length_predictions,))(predictions)
        decoder = tf.keras.Model(latent_inputs, predictions, name='decoder')
        decoder.compile()

        print(encoder.summary())
        print(decoder.summary())
        
        std = dataloader.vae_norm["profile"]["std"]
        mean = dataloader.vae_norm["profile"]["mean"]

        minimum_value = config["min_value"] if config else 0.
        # normalize to make it correspond to the data
        minimum_value = minimum_value * std + mean

        autoencoder = VAE_singleval(encoder, decoder, [r_loss,k_loss,gain_loss], config=config, min_value = minimum_value, physical_penalty_weight=physical_penalty_weight)

        return autoencoder, encoder, decoder