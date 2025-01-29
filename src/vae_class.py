import tensorflow as tf 
from tensorflow import keras
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
    
class SamplingLayer(keras.layers.Layer):
    '''A custom layer that receive (z_mean, z_var) and sample a z vector'''

    def call(self, inputs):
        
        z_mean, z_log_var = inputs
        
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        return z

class VAE(keras.Model):
    '''
    A VAE model, built from given encoder and decoder
    '''

    version = '1.4'

    def __init__(self, encoder=None, decoder=None, loss_weights=[1,1,1], **kwargs):
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
            
        k1,k2,k3 = self.loss_weights
        
        # ---- Forward pass
        #      Run the forward pass and record 
        #      operations on the GradientTape.
        #
        input32              = tf.cast(input,dtype=tf.float32)
        with tf.GradientTape() as tape:
            
            # ---- Get encoder outputs
            #
            z_mean, z_log_var, z = self.encoder(input)
            
            # ---- Get reconstruction from decoder
            #
            reconstruction       = self.decoder(z)
         
            # ---- Compute loss
            #      Reconstruction loss, KL loss and Total loss

            # gain32        = tf.cast(gain,dtype=tf.float32)
            # gain_constraint_loss = tf.reduce_mean(tf.square(z_mean[:, 0] - gain32))            

            reconstruction_loss  = k1 * tf.reduce_mean(tf.square(input32 - reconstruction))
            # reconstruction_loss  = k1 * tf.keras.losses.binary_crossentropy(input32,reconstruction)

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = -tf.reduce_mean(kl_loss) * k2

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
        z_mean, z_log_var, z = self.encoder(input)

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

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -tf.reduce_mean(kl_loss) * k2

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
        self.encoder = keras.models.load_model(f'{filename}-encoder.keras', custom_objects={'SamplingLayer': SamplingLayer})
        self.decoder = keras.models.load_model(f'{filename}-decoder.keras')
        print('Reloaded.')
