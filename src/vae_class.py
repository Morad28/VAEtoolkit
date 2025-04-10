import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.saving import register_keras_serializable
from keras import layers, losses, metrics
import os

# VAE Class from FIDLE
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
@register_keras_serializable(package="Custom")
class SamplingMoG(layers.Layer):
    """Sampling layer for Mixture of Gaussians."""
    def call(self, inputs):
        z_means, z_log_vars, z_pis = inputs
        batch = tf.shape(z_means)[0]
        num_components = tf.shape(z_means)[1]
        latent_dim = tf.shape(z_means)[2]

        # Sample a component index based on z_pis
        z_pis = tf.nn.softmax(z_pis, axis=-1)  # Ensure mixture coefficients are valid probabilities
        component_indices = tf.random.categorical(tf.math.log(z_pis), 1)  # Sample component index
        component_indices = tf.squeeze(component_indices, axis=-1)  # Shape: (batch,)

        # Gather the mean and log variance for the selected component
        selected_means = tf.gather(z_means, component_indices, batch_dims=1)
        selected_log_vars = tf.gather(z_log_vars, component_indices, batch_dims=1)

        # Sample from the selected Gaussian
        epsilon = tf.random.normal(shape=(batch, latent_dim))
        z = selected_means + tf.exp(0.5 * selected_log_vars) * epsilon
        return z
    
class SamplingLayer(keras.layers.Layer):
    '''A custom layer that receive (z_mean, z_var) and sample a z vector'''

    def call(self, inputs):
        
        z_mean, z_log_var = inputs
        
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        return z

class SamplingMoGLayer(keras.layers.Layer):
    '''A custom layer that receive (z_mean, z_var) and sample a z vector for Mixture of Gaussians'''

    def call(self, inputs):
        
        z_means, z_log_vars, z_pis = inputs
        
        batch_size = tf.shape(z_means)[0]
        num_components = tf.shape(z_means)[1]
        latent_dim = tf.shape(z_means)[2]

        # Sample a component index based on z_pis
        z_pis = tf.nn.softmax(z_pis, axis=-1)  # Ensure mixture coefficients are valid probabilities
        component_indices = tf.random.categorical(tf.math.log(z_pis), 1)  # Sample component index
        component_indices = tf.squeeze(component_indices, axis=-1)  # Shape: (batch,)

        # Gather the mean and log variance for the selected component
        selected_means = tf.gather(z_means, component_indices, batch_dims=1)
        selected_log_vars = tf.gather(z_log_vars, component_indices, batch_dims=1)

        # Sample from the selected Gaussian
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))
        z = selected_means + tf.exp(0.5 * selected_log_vars) * epsilon
        
        return z

class VAE(keras.Model):
    '''
    A VAE model, built from given encoder and decoder
    '''

    version = '1.4'

    def __init__(self, encoder=None, decoder=None, loss_weights=[1,1,1], config=None, **kwargs):
        '''
        VAE instantiation with encoder, decoder and r_loss_factor
        args :
            encoder : Encoder model
            decoder : Decoder model
            loss_weights : Weight of the loss functions: reconstruction_loss and kl_loss
            r_loss_factor : Proportion of reconstruction loss for global loss (0.3)
        return:
            None
        '''
        super(VAE, self).__init__(**kwargs)
        self.config = config
        self.encoder      = encoder
        self.decoder      = decoder
        self.loss_weights = loss_weights
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="r_loss"
        )
        self.gain_loss_tracker = keras.metrics.Mean(name="gain_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.gain_loss_tracker,
        ]
       
    @tf.function
    def call(self, inputs):
        '''
        Model forward pass, when we use our model
        args:
            inputs : Model inputs
        return:
            output : Output of the model 
        '''
        z_mean, z_log_var, z = self.encoder(inputs)
        output               = self.decoder(z)
        return output
                
    @tf.function
    def train_step(self, data):
        '''
        Implementation of the training update.
        Receive an input, compute loss, get gradient, update weights and return metrics.
        Here, our metrics are loss.
        args:
            inputs : Model inputs
        return:
            loss    : Total loss
            r_loss  : Reconstruction loss
            kl_loss : KL loss
        '''

        input = data
        # ---- Get the input we need, specified in the .fit()
        #
        # if isinstance(input, tuple):
        #     input = input[0]
        
        # r_loss, k_loss, gain_loss
        k1,k2,k3 = self.loss_weights
        
        # ---- Forward pass
        #      Run the forward pass and record 
        #      operations on the GradientTape.
        #
        input32              = tf.cast(input,dtype=tf.float32)
        if self.config != None and self.config["DataType"] == "1DFCI-GAIN" and self.config["sep_loss"]:
            print("Separating gain and data loss")
            # check the shape of input32
            input_data = input32[:,:-1]
            gain = input32[:,-1]

        with tf.GradientTape() as tape:
            
            # ---- Get encoder outputs
            #
            z_mean, z_log_var, z = self.encoder(input)
            
            # ---- Get reconstruction from decoder
            #
            reconstruction       = self.decoder(z)
            if self.config != None and self.config["DataType"] == "1DFCI-GAIN" and self.config["sep_loss"]:
                reconstruction_data = reconstruction[:,:-1]
                reconstruction_gain = reconstruction[:,-1]
         
                # ---- Compute loss
                #      Reconstruction loss, KL loss and Total loss

                # gain32        = tf.cast(gain,dtype=tf.float32)
                # gain_constraint_loss = tf.reduce_mean(tf.square(z_mean[:, 0] - gain32))

                len_input_data = tf.cast(tf.shape(input_data)[1], dtype=tf.float32)
                len_gain = 1.
                len_input32 = tf.cast(tf.shape(input32)[1], dtype=tf.float32)
                # show the values of the tensors


                reconstruction_loss_data  = k1 * tf.reduce_mean(tf.square(input_data - reconstruction_data)) * len_input_data / len_input32
                reconstruction_loss_gain  = k3 * tf.reduce_mean(tf.square(gain - reconstruction_gain)) * len_gain / len_input32
            
                reconstruction_loss  = reconstruction_loss_data + reconstruction_loss_gain
            else:
                # ---- Compute loss
                #      Reconstruction loss, KL loss and Total loss

                # gain32        = tf.cast(gain,dtype=tf.float32)
                # gain_constraint_loss = tf.reduce_mean(tf.square(z_mean[:, 0] - gain32))            

                reconstruction_loss  = k1 * tf.reduce_mean(tf.square(input32 - reconstruction))

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = -tf.reduce_mean(kl_loss) * k2

            total_loss = reconstruction_loss + kl_loss

        # ---- Retrieve gradients from gradient_tape
        #      and run one step of gradient descent
        #      to optimize trainable weights
        #

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        if self.config != None and self.config["DataType"] == "1DFCI-GAIN" and self.config["sep_loss"]:
            self.gain_loss_tracker.update_state(reconstruction_loss_gain)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss_data)
            return {
                "loss":     self.total_loss_tracker.result(),
                "r_loss":   self.reconstruction_loss_tracker.result(),
                "kl_loss":  self.kl_loss_tracker.result(),
                "gain_loss" : self.gain_loss_tracker.result()
            }
        else: 
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            return {
                "loss":     self.total_loss_tracker.result(),
                "r_loss":   self.reconstruction_loss_tracker.result(),
                "kl_loss":  self.kl_loss_tracker.result()
                # "gain_loss" : gain_constraint_loss,
            }
    
    @tf.function
    def test_step(self,val_data):
        input = val_data
        z_mean, z_log_var, z = self.encoder(input)

        k1,k2,k3 = self.loss_weights

            
            # ---- Get reconstruction from decoder
            #
        reconstruction       = self.decoder(z)
        input32              = tf.cast(input,dtype=tf.float32)

        if self.config != None and self.config["DataType"] == "1DFCI-GAIN" and self.config["sep_loss"]:
            reconstruction_data = reconstruction[:,:-1]
            reconstruction_gain = reconstruction[:,-1]
            gain = input32[:,-1]
            input_data = input32[:,:-1]

            len_input_data = tf.cast(tf.shape(input_data)[1], dtype=tf.float32)
            len_gain = 1.
            len_input32 = tf.cast(tf.shape(input32)[1], dtype=tf.float32)
            reconstruction_loss_data  = k1 * tf.reduce_mean(tf.square(input_data - reconstruction_data)) * len_input_data / len_input32
            reconstruction_loss_gain  = k3 * tf.reduce_mean(tf.square(gain - reconstruction_gain)) * len_gain / len_input32
        
            reconstruction_loss  = reconstruction_loss_data + reconstruction_loss_gain
         
            # ---- Compute loss
            #      Reconstruction loss, KL loss and Total loss

        # gain32        = tf.cast(gain,dtype=tf.float32)
        # gain_constraint_loss = tf.reduce_mean(tf.square(z_mean[:, 0] - gain32))            

        else:
            reconstruction_loss  = k1 * tf.reduce_mean(tf.square(input32 - reconstruction) )
        # reconstruction_loss  = k1 * tf.keras.losses.binary_crossentropy(input32,reconstruction)

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -tf.reduce_mean(kl_loss) * k2

        total_loss = reconstruction_loss + kl_loss #+ gain_constraint_loss * k3 
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        if self.config != None and self.config["DataType"] == "1DFCI-GAIN" and self.config["sep_loss"]:
            self.gain_loss_tracker.update_state(reconstruction_loss_gain)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss_data)
            return {
                "loss":     self.total_loss_tracker.result(),
                "r_loss":   self.reconstruction_loss_tracker.result(),
                "kl_loss":  self.kl_loss_tracker.result(),
                "gain_loss" : self.gain_loss_tracker.result()
            }
        else:
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            return {
                "loss":     self.total_loss_tracker.result(),
                "r_loss":   self.reconstruction_loss_tracker.result(),
                "kl_loss":  self.kl_loss_tracker.result()
                # "gain_loss" : gain_constraint_loss,
            }
    
    def predict(self,inputs):
        '''Our predict function...'''
        z_mean, z_var, z  = self.encoder.predict(inputs)
        outputs           = self.decoder.predict(z)
        return outputs

    def save(self,filename):
        '''Save model in 2 part'''
        filename, extension = os.path.splitext(filename)
        self.encoder.save(f'{filename}-encoder.keras')
        self.decoder.save(f'{filename}-decoder.keras')

    def reload(self,filename):
        '''Reload a 2 part saved model.'''
        filename, extension = os.path.splitext(filename)
        self.encoder = keras.ModelSelector.load_model(f'{filename}-encoder.keras', custom_objects={'SamplingLayer': SamplingLayer})
        self.decoder = keras.ModelSelector.load_model(f'{filename}-decoder.keras')
        print('Reloaded.')

class VAE_MoG(keras.Model):
    '''
    A VAE model, built from given encoder and decoder
    '''

    version = '1.4'

    def __init__(self, encoder=None, decoder=None, loss_weights=[1,1,1], **kwargs):
        '''
        VAE instantiation with encoder, decoder and r_loss_factor, for a Mixture of Gaussians latent space
        args :
            encoder : Encoder model
            decoder : Decoder model
            loss_weights : Weight of the loss functions: reconstruction_loss and kl_loss
            r_loss_factor : Proportion of reconstruction loss for global loss (0.3)
        return:
            None
        '''
        super(VAE_MoG, self).__init__(**kwargs)
        self.encoder      = encoder
        self.decoder      = decoder
        self.loss_weights = loss_weights
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="r_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
       
    @tf.function
    def call(self, inputs):
        '''
        Model forward pass, when we use our model
        args:
            inputs : Model inputs
        return:
            output : Output of the model 
        '''
        z_mean, z_log_var, z_pis, z = self.encoder(inputs)
        output               = self.decoder(z)
        return output
                
    @tf.function
    def train_step(self, data):
        '''
        Implementation of the training update.
        Receive an input, compute loss, get gradient, update weights and return metrics.
        Here, our metrics are loss.
        args:
            inputs : Model inputs
        return:
            loss    : Total loss
            r_loss  : Reconstruction loss
            kl_loss : KL loss
        '''

        input = data    
        # ---- Get the input we need, specified in the .fit()
        #
        # if isinstance(input, tuple):
        #     input = input[0]
            
        k1,k2,k3 = self.loss_weights
        
        # ---- Forward pass
        #      Run the forward pass and record 
        #      operations on the GradientTape.
        #
        input32              = tf.cast(input,dtype=tf.float32)
        with tf.GradientTape() as tape:
            
            # ---- Get encoder outputs
            #
            z_means, z_log_vars, z_pis, z = self.encoder(input)
            
            # ---- Get reconstruction from decoder
            #
            reconstruction       = self.decoder(z)
         
            # ---- Compute loss
            #      Reconstruction loss, KL loss and Total loss

            # gain32        = tf.cast(gain,dtype=tf.float32)
            # gain_constraint_loss = tf.reduce_mean(tf.square(z_mean[:, 0] - gain32))            

            reconstruction_loss  = k1 * tf.reduce_mean(tf.square(input32 - reconstruction))
            # reconstruction_loss  = k1 * tf.keras.losses.binary_crossentropy(input32,reconstruction)

            z_pis = tf.nn.softmax(z_pis, axis=-1)  # Ensure mixture coefficients are valid probabilities
            z_pis_expanded = tf.expand_dims(z_pis, axis=-1)  # Shape: [batch_size, num_components, 1]

            kl_loss = -0.5 * tf.reduce_sum(
                z_pis_expanded * (1 + z_log_vars - tf.square(z_means) - tf.exp(z_log_vars)),
                axis=[1, 2]
            )
            kl_loss = tf.reduce_mean(kl_loss) * k2

            total_loss = reconstruction_loss + kl_loss # + gain_constraint_loss * k3 

        # ---- Retrieve gradients from gradient_tape
        #      and run one step of gradient descent
        #      to optimize trainable weights
        #

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss":     self.total_loss_tracker.result(),
            "r_loss":   self.reconstruction_loss_tracker.result(),
            "kl_loss":  self.kl_loss_tracker.result()
            # "gain_loss" : gain_constraint_loss,
        }
    
    @tf.function
    def test_step(self,val_data):
        input = val_data
        z_means, z_log_vars, z_pis, z = self.encoder(input)

        k1,k2,k3 = self.loss_weights

            
            # ---- Get reconstruction from decoder
            #
        reconstruction       = self.decoder(z)
        input32              = tf.cast(input,dtype=tf.float32)


         
            # ---- Compute loss
            #      Reconstruction loss, KL loss and Total loss

        # gain32        = tf.cast(gain,dtype=tf.float32)
        # gain_constraint_loss = tf.reduce_mean(tf.square(z_mean[:, 0] - gain32))            

        reconstruction_loss  = k1 * tf.reduce_mean(tf.square(input32 - reconstruction) )
        # reconstruction_loss  = k1 * tf.keras.losses.binary_crossentropy(input32,reconstruction)

        z_pis = tf.nn.softmax(z_pis, axis=-1)
        z_pis_expanded = tf.expand_dims(z_pis, axis=-1)

        kl_loss = -0.5 * tf.reduce_sum(
            z_pis_expanded * (1 + z_log_vars - tf.square(z_means) - tf.exp(z_log_vars)),
            axis=[1, 2]
        )
        kl_loss = tf.reduce_mean(kl_loss) * k2

        total_loss = reconstruction_loss + kl_loss #+ gain_constraint_loss * k3 
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss":     self.total_loss_tracker.result(),
            "r_loss":   self.reconstruction_loss_tracker.result(),
            "kl_loss":  self.kl_loss_tracker.result()
            # "gain_loss" : gain_constraint_loss,
        }
    
    def predict(self,inputs):
        '''Our predict function...'''
        z_means, z_vars, z_pis, z  = self.encoder.predict(inputs)
        outputs           = self.decoder.predict(z)
        return outputs

    def save(self,filename):
        '''Save model in 2 part'''
        filename, extension = os.path.splitext(filename)
        self.encoder.save(f'{filename}-encoder.keras')
        self.decoder.save(f'{filename}-decoder.keras')

    def reload(self, filename):
        '''Reload a 2-part saved model.'''
        filename, extension = os.path.splitext(filename)
        self.encoder = keras.models.load_model(
            f"{filename}-encoder.keras",
            custom_objects={"SamplingMoG": SamplingMoG}
        )
        self.decoder = keras.models.load_model(f"{filename}-decoder.keras")
        print("Reloaded.")