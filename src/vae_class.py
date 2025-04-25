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

    def __init__(self, encoder=None, decoder=None, loss_weights=[1,1,1], config=None, min_value=0.3, physical_penalty_weight=1., **kwargs):
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
        self.physical_penalty_weight = physical_penalty_weight
        self.config = config
        self.min_value = min_value
        self.encoder      = encoder
        self.decoder      = decoder
        self.loss_weights = loss_weights
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="r_loss"
        )
        self.gain_loss_tracker = keras.metrics.Mean(name="gain_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.physical_loss_tracker = keras.metrics.Mean(name="physical_penalty")
        


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
        if self.config != None and (self.config["DataType"] == "1DFCI-GAIN" or self.config["DataType"]=="COILS-MULTI") and self.config["sep_loss"]:
            print("Separating gain and data loss")
            # check the shape of input32
            len_values = len(self.config["values"])
            input_data = input32[:,:-len_values]
            gain = input32[:,-len_values:]

        with tf.GradientTape() as tape:
            
            # ---- Get encoder outputs
            #
            z_mean, z_log_var, z = self.encoder(input)
            
            # ---- Get reconstruction from decoder
            #
            reconstruction       = self.decoder(z)
            if self.config != None and (self.config["DataType"] == "1DFCI-GAIN" or self.config["DataType"]=="COILS-MULTI") and self.config["sep_loss"]:
                len_values = len(self.config["values"])
                reconstruction_data = reconstruction[:,:-len_values]
                reconstruction_values = reconstruction[:,-len_values:]

                # add an additional penalty to the reconstructed data loss if the reconstructed data gets below the minimum value
                # the penalty is calculated as the mean over the samples of the sum of the squared difference between the reconstructed
                # data and the minimum value
                min_value = self.min_value
                penalty = reconstruction_data - min_value
                penalty = tf.where(penalty > 0, penalty, 0)
                penalty = tf.reduce_mean(penalty, axis=0)
                penalty = tf.reduce_sum(tf.square(penalty))
                penalty = penalty * self.physical_penalty_weight
         
                # ---- Compute loss
                #      Reconstruction loss, KL loss and Total loss

                # gain32        = tf.cast(gain,dtype=tf.float32)
                # gain_constraint_loss = tf.reduce_mean(tf.square(z_mean[:, 0] - gain32))

                len_input_data = tf.cast(tf.shape(input_data)[1], dtype=tf.float32)
                len_input32 = tf.cast(tf.shape(input32)[1], dtype=tf.float32)
                # show the values of the tensors


                reconstruction_loss_data  = k1 * tf.reduce_mean(tf.square(input_data - reconstruction_data)) * len_input_data / len_input32
                reconstruction_loss_gain  = k3 * tf.reduce_mean(tf.square(gain - reconstruction_values)) * len_values / len_input32
            
                reconstruction_loss  = reconstruction_loss_data + reconstruction_loss_gain + penalty
            else:
                # ---- Compute loss
                #      Reconstruction loss, KL loss and Total loss

                # gain32        = tf.cast(gain,dtype=tf.float32)
                # gain_constraint_loss = tf.reduce_mean(tf.square(z_mean[:, 0] - gain32))

                # add an additional penalty to the reconstructed data loss if the reconstructed data gets below the minimum value
                # the penalty is calculated as the mean over the samples of the sum of the squared difference between the reconstructed
                # data and the minimum value
                if self.config != None and (self.config["DataType"] == "1DFCI-GAIN" or self.config["DataType"]=="COILS-MULTI"):
                    len_values = len(self.config["values"])
                    reconstruction_data = reconstruction[:,:-len_values]
                    min_value = self.min_value
                    penalty = reconstruction_data - min_value
                    penalty = tf.where(penalty > 0, penalty, 0)
                    penalty = tf.reduce_mean(penalty, axis=0)
                    penalty = tf.reduce_sum(tf.square(penalty))
                    penalty = penalty * self.physical_penalty_weight       
                else:
                    min_value = self.min_value
                    penalty = reconstruction - min_value
                    penalty = tf.where(penalty > 0, penalty, 0)
                    penalty = tf.reduce_mean(penalty, axis=0)
                    penalty = tf.reduce_sum(tf.square(penalty))
                    penalty = penalty * self.physical_penalty_weight     

                reconstruction_loss_r  = k1 * tf.reduce_mean(tf.square(input32 - reconstruction))
                reconstruction_loss  = reconstruction_loss_r + penalty

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
        self.physical_loss_tracker.update_state(penalty)
        if self.config != None and (self.config["DataType"] == "1DFCI-GAIN" or self.config["DataType"]=="COILS-MULTI") and self.config["sep_loss"]:
            self.gain_loss_tracker.update_state(reconstruction_loss_gain)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss_data)
            return {
                "loss":     self.total_loss_tracker.result(),
                "r_loss":   self.reconstruction_loss_tracker.result(),
                "kl_loss":  self.kl_loss_tracker.result(),
                "gain_loss" : self.gain_loss_tracker.result(),
                "physical_penalty" : self.physical_loss_tracker.result()
            }
        else: 
            self.reconstruction_loss_tracker.update_state(reconstruction_loss_r)
            return {
                "loss":     self.total_loss_tracker.result(),
                "r_loss":   self.reconstruction_loss_tracker.result(),
                "kl_loss":  self.kl_loss_tracker.result(),
                "physical_penalty" : self.physical_loss_tracker.result()
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

        if self.config != None and (self.config["DataType"] == "1DFCI-GAIN" or self.config["DataType"]=="COILS-MULTI") and self.config["sep_loss"]:
            len_values = len(self.config["values"])
            reconstruction_data = reconstruction[:,:-len_values]
            reconstruction_gain = reconstruction[:,-len_values:]
            gain = input32[:,-len_values:]
            input_data = input32[:,:-len_values]

            # add an additional penalty to the reconstructed data loss if the reconstructed data gets below the minimum value
            # the penalty is calculated as the mean over the samples of the sum of the squared difference between the reconstructed
            # data and the minimum value
            min_value = self.min_value
            penalty = reconstruction_data - min_value
            penalty = tf.where(penalty > 0, penalty, 0)
            penalty = tf.reduce_mean(penalty, axis=0)
            penalty = tf.reduce_sum(tf.square(penalty))
            penalty = penalty * self.physical_penalty_weight

            len_input_data = tf.cast(tf.shape(input_data)[1], dtype=tf.float32)
            len_input32 = tf.cast(tf.shape(input32)[1], dtype=tf.float32)
            reconstruction_loss_data  = k1 * tf.reduce_mean(tf.square(input_data - reconstruction_data)) * len_input_data / len_input32
            reconstruction_loss_gain  = k3 * tf.reduce_mean(tf.square(gain - reconstruction_gain)) * len_values / len_input32
        
            reconstruction_loss  = reconstruction_loss_data + reconstruction_loss_gain + penalty
         
            # ---- Compute loss
            #      Reconstruction loss, KL loss and Total loss

        # gain32        = tf.cast(gain,dtype=tf.float32)
        # gain_constraint_loss = tf.reduce_mean(tf.square(z_mean[:, 0] - gain32))            

        else:
            # add an additional penalty to the reconstructed data loss if the reconstructed data gets below the minimum value
            # the penalty is calculated as the mean over the samples of the sum of the squared difference between the reconstructed
            # data and the minimum value
            if self.config != None and (self.config["DataType"] == "1DFCI-GAIN" or self.config["DataType"]=="COILS-MULTI"):
                len_values = len(self.config["values"])
                reconstruction_data = reconstruction[:,:-len_values]
                min_value = self.min_value
                penalty = reconstruction_data - min_value
                penalty = tf.where(penalty > 0, penalty, 0)
                penalty = tf.reduce_mean(penalty, axis=0)
                penalty = tf.reduce_sum(tf.square(penalty))
                penalty = penalty * self.physical_penalty_weight
            else:
                min_value = self.min_value
                penalty = reconstruction - min_value
                penalty = tf.where(penalty > 0, penalty, 0)
                penalty = tf.reduce_mean(penalty, axis=0)
                penalty = tf.reduce_sum(tf.square(penalty))
                penalty = penalty * self.physical_penalty_weight

            reconstruction_loss_r  = k1 * tf.reduce_mean(tf.square(input32 - reconstruction) )
            reconstruction_loss  = reconstruction_loss_r + penalty

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -tf.reduce_mean(kl_loss) * k2

        total_loss = reconstruction_loss + kl_loss #+ gain_constraint_loss * k3 
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.physical_loss_tracker.update_state(penalty)
        if self.config != None and (self.config["DataType"] == "1DFCI-GAIN" or self.config["DataType"]=="COILS-MULTI") and self.config["sep_loss"]:
            self.gain_loss_tracker.update_state(reconstruction_loss_gain)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss_data)
            return {
                "loss":     self.total_loss_tracker.result(),
                "r_loss":   self.reconstruction_loss_tracker.result(),
                "kl_loss":  self.kl_loss_tracker.result(),
                "gain_loss" : self.gain_loss_tracker.result(),
                "physical_penalty" : self.physical_loss_tracker.result()
            }
        else:
            self.reconstruction_loss_tracker.update_state(reconstruction_loss_r)
            return {
                "loss":     self.total_loss_tracker.result(),
                "r_loss":   self.reconstruction_loss_tracker.result(),
                "kl_loss":  self.kl_loss_tracker.result(),
                "physical_penalty" : self.physical_loss_tracker.result()
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

class VAE_multi_decoder(keras.Model):
    '''
    A VAE model, built from given encoder and decoder
    '''

    version = '1.4'

    def __init__(self, encoder=None, decoders=None, loss_weights=[1,1,1], config=None, min_value=0.3, physical_penalty_weight=1., **kwargs):
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
        super(VAE_multi_decoder, self).__init__(**kwargs)
        self.physical_penalty_weight = physical_penalty_weight
        self.config = config
        self.min_value = min_value
        self.encoder      = encoder
        self.decoders      = decoders
        self.loss_weights = loss_weights
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="r_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.values_loss_tracker = keras.metrics.Mean(name="values_loss")
        self.physical_loss_tracker = keras.metrics.Mean(name="physical_penalty")

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=2500,
            decay_rate=0.95,
            staircase=True
        ) # Learning rate schedule for MNIST

        self.optimizer_cnn = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
        self.optimizer_mlp = tf.keras.optimizers.AdamW(
                                    learning_rate=0.001,
                                    weight_decay=0.004,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-07,
                                    amsgrad=False,
                                    clipnorm=None,
                                    clipvalue=None,
                                    global_clipnorm=None,
                                    use_ema=False,
                                    ema_momentum=0.99,
                                    ema_overwrite_frequency=None,
                                    loss_scale_factor=None,
                                    gradient_accumulation_steps=None,
                                    name='adamw',
                                    **kwargs
                                )
        


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.values_loss_tracker,
            self.physical_loss_tracker
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
        outputs = [decoder(z) for decoder in self.decoders]
        return outputs
                
    @tf.function
    def train_step(self, data):
        '''
        Implementation of the training update for multiple decoders.
        args:
            data : Model inputs
        return:
            Dictionary of losses
        '''
        input = data
        k1, k2, k3 = self.loss_weights
        len_values = len(self.config["values"])

        with tf.GradientTape(persistent=True) as tape:
            # Encoder forward pass
            z_mean, z_log_var, z = self.encoder(input)

            # Decoder forward passes
            reconstructions = [decoder(z) for decoder in self.decoders]

            # switch to float32
            input = tf.cast(input, dtype=tf.float32)
            reconstructions = [tf.cast(reconstruction, dtype=tf.float32) for reconstruction in reconstructions]

            # Compute reconstruction losses
            reconstruction_loss_profile = k1 * tf.reduce_mean(tf.square(input[:, :-len_values] - reconstructions[0]))
            reconstruction_loss_values = k3 * tf.reduce_mean(tf.square(input[:, -len_values:] - reconstructions[1]))


            # KL divergence loss
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = -tf.reduce_mean(kl_loss) * k2

            # Total loss
            reconstruction_loss_profile = reconstruction_loss_profile + kl_loss
            total_loss = reconstruction_loss_profile + reconstruction_loss_values

        # Backpropagation
        grads_encoder = tape.gradient(reconstruction_loss_profile, self.encoder.trainable_weights)
        grads_cnn = tape.gradient(reconstruction_loss_profile, self.decoders[0].trainable_weights)
        grads_mlp = tape.gradient(reconstruction_loss_values, self.decoders[1].trainable_weights)

        self.optimizer.apply_gradients(zip(grads_encoder, self.encoder.trainable_weights))
        self.optimizer_cnn.apply_gradients(zip(grads_cnn, self.decoders[0].trainable_weights))
        self.optimizer_mlp.apply_gradients(zip(grads_mlp, self.decoders[1].trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss_profile)
        self.values_loss_tracker.update_state(reconstruction_loss_values)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "r_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "values_loss": self.values_loss_tracker.result(),
        }
    
    @tf.function
    def test_step(self, data):
        '''
        Implementation of the evaluation step for multiple decoders.
        args:
            data : Validation inputs
        return:
            Dictionary of losses
        '''
        input = data
        k1, k2, k3 = self.loss_weights
        len_values = len(self.config["values"])

        # Encoder forward pass
        z_mean, z_log_var, z = self.encoder(input)

        # Decoder forward passes
        reconstructions = [decoder(z) for decoder in self.decoders]

        # switch to float32
        input = tf.cast(input, dtype=tf.float32)
        reconstructions = [tf.cast(reconstruction, dtype=tf.float32) for reconstruction in reconstructions]

        # Compute reconstruction losses
        reconstruction_loss_profile = k1 * tf.reduce_mean(tf.square(input[:, :-len_values] - reconstructions[0]))
        reconstruction_loss_values = k3 * tf.reduce_mean(tf.square(input[:, -len_values:] - reconstructions[1]))

        # Combine reconstruction losses
        reconstruction_loss = reconstruction_loss_profile + reconstruction_loss_values

        # KL divergence loss
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -tf.reduce_mean(kl_loss) * k2

        # Total loss
        total_loss = reconstruction_loss + kl_loss

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss_profile)
        self.values_loss_tracker.update_state(reconstruction_loss_values)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "r_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "values_loss": self.values_loss_tracker.result(),
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
        if isinstance(self.decoders, list):
            for i, decoder in enumerate(self.decoders):
                decoder.save(f'{filename}-decoder-{i}.keras')
        else:
            self.decoders.save(f'{filename}-decoder.keras')

    def reload(self,filename):
        '''Reload a 2 part saved model.'''
        filename, extension = os.path.splitext(filename)
        self.encoder = keras.ModelSelector.load_model(f'{filename}-encoder.keras', custom_objects={'SamplingLayer': SamplingLayer})
        self.decoder = keras.ModelSelector.load_model(f'{filename}-decoder.keras')
        print('Reloaded.')


