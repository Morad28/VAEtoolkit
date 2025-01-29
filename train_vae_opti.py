import sys
from src.vae_class import VAE, Sampling
from src.config_vae import get_config
import sys
import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib
import random
import shutil
from src.dataloader import DataLoader, DataLoaderFCI

# To remove
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    matplotlib.use('Agg')
    cmap = matplotlib.colormaps['viridis']  

    config_path = sys.argv[1]
    config = get_config(config_path)

    # Access parameters
    dataset_path = config["dataset_path"]
    results_dir = config["results_dir"]
    name = config["name"]
    epoch_vae = config["epoch_vae"]
    epoch_rna = (config["epoch_rna"])
    latent_dim = (config["latent_dim"])
    batch_size_vae = config["batch_size_vae"]
    batch_size_rna = config["batch_size_rna"]
    kl_loss = (config["kl_loss"])
    log = config["log"]
    values = config["training"]
    gain_only = config["reprise"]["gain_only"]
    filtered = config["filter"]
    
    loaded_dataset = np.load(dataset_path, allow_pickle=True).item()
        
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, losses
    from keras.layers import Input, Dense, Conv1D, Conv1DTranspose, Flatten, Reshape, MaxPooling1D
    

    """
    DATA PROCESSING 
    """
    
    fci_dataset = DataLoaderFCI(dataset_path)
    fci_dataset.apply_mask(filtered)
    loaded_dataset = fci_dataset.get_data()

    np_gain = np.array(loaded_dataset['values']["gain"]) /  np.max(loaded_dataset['values']["gain"])
    np_data = np.array(loaded_dataset['data'])
    print(np_data.shape)

    """
    PREPROCESSING OF DATA (normalization,...)
    """

    # np_data = np.array(data)[:,:] 
    np_data = np_data / np.max(np_data)

    # Shuffle data and split
    combined = list(zip(np_data,np_gain))
    random.shuffle(combined)

    np_data_shuffled, np_gain_shuffled = zip(*combined)
    np_data_shuffled = np.array(np_data_shuffled)
    np_gain_shuffled = np.array(np_gain_shuffled)

    train_size = int(0.8 * np_data.shape[0])

    np_data_train = np_data_shuffled[:train_size]
    np_gain_train = np_gain_shuffled[:train_size]

    np_data_val = np_data_shuffled[train_size:]
    np_gain_val = np_gain_shuffled[train_size:]

    def below_10_percent(y_true, y_pred):
        y_true = tf.cast(y_true,tf.float32)
        absolute_error = tf.abs( y_true- y_pred)
        relative_error = absolute_error / tf.maximum(tf.abs(y_true), 1e-7) # Add a small epsilon to prevent division by zero
        below_10_percent = tf.reduce_mean(tf.cast(relative_error < 0.1, tf.float32)) * 100
        return below_10_percent

    if not gain_only:
        """
        DEFINITION OF DEEP LEARNING MODEL
        """



        def std_conv_ae(input_shape=250,latent_dim = 64, r_loss = 1, k_loss = 1, gain_loss = 1):
            
            inputs = Input(shape=(input_shape,1))
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
            print(encoder.summary())

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

            print(decoder.summary())

            autoencoder = VAE(encoder,decoder, [r_loss,k_loss,gain_loss])


            return(autoencoder, encoder, decoder)

        """
        PREPARE DATASET FOR TRAINING
        """
        # Batch size
        batch_size = batch_size_vae
        # train_datasettt = tf.data.Dataset.from_tensor_slices((np_data_train)).batch(batch_size)
        # val_datasettt = tf.data.Dataset.from_tensor_slices((np_data_val)).batch(batch_size)
        
        train_datasettt, val_datasettt = fci_dataset.to_dataset(np_data, batch_size=batch_size_vae, shuffle=True, split = 0.8)
        # val_datasettt = fci_dataset.to_dataset(np_data_val, batch_size=batch_size_vae, shuffle=True)

        """
        Preparation of training  
        """

        # Parameters (nothing to be modified here all is avaible in command line)
        input_shape = np_data.shape[1]
        latent_dim = latent_dim
        r_loss = 1.
        k_loss = kl_loss 
        gain_loss = 0.
        autoencoder, encoder, decoder = std_conv_ae(input_shape,latent_dim,r_loss,k_loss,gain_loss) # ref
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.001,
                        decay_steps=4000,
                        decay_rate=0.9,
                        staircase=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)   
        autoencoder.compile(optimizer=optimizer,metrics = [below_10_percent])

        res_name = f'{results_dir}'
        if not os.path.isdir(f'./{res_name}'):
            os.mkdir(f'./{res_name}')
            
        res_folder = f"./{res_name}/std_{name}_{np_data.shape[0]}_latent_{int(latent_dim)}_kl_{k_loss}_{batch_size}/"
        print(res_folder)

        if not os.path.isdir(res_folder):
            os.mkdir(res_folder)

        shutil.copy(config_path,res_folder + f'conf.json')

        # Saving all data
        plt.figure()
        for i in range(0,np_data.shape[0]):
            plt.plot(np_data[i])
        plt.savefig(res_folder + "data.png")

        # Tensorboard to track training loss and metrics
        log_dir = res_folder + "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

        callbacks=[
            # callback,
            tensorboard_callback]

        epochs = epoch_vae

        """
        TRAINING
        """

        history = autoencoder.fit(
            train_datasettt,
            epochs=epochs, 
            validation_data=val_datasettt,
            callbacks=callbacks,
            verbose = 2
            )

        """
        All that follows is for plotting and saving the results. 
        It can be ignored if you just want to use the model.
        """

        dataset = tf.data.Dataset.from_tensor_slices((np_data,np_gain))
        batch_size = 128
        dataset_batched = dataset.batch(batch_size)


        data_train = history.history['loss']
        data_val = history.history['val_loss']

        np.savetxt(res_folder + "losses.txt",[data_train,data_val])
        plt.figure()

        plt.grid(True,which="both")
        plt.semilogy(data_train,label="Données d'entraînement")
        plt.semilogy(data_val,label="Données de validation")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.tick_params(axis='both', which='both', direction='in')
        plt.legend(frameon=True)
        plt.savefig(res_folder+"losses.png")
        plt.close()

        autoencoder.save(res_folder + "model.keras")
        encoder.save(res_folder+'encoder_model.keras')
        decoder.save(res_folder+'decoder_model.keras')


        latent_space = encoder.predict(dataset_batched)[-1]
        tilde_laser = decoder.predict(latent_space)

        if not os.path.isdir(res_folder+"img"):
            os.mkdir(res_folder+"img")

        # Takes random data to plot 
        list_i = np.arange(len(np_data))
        choices = np.random.choice(list_i,25) # 25 random data
        for j in choices:
            plt.figure()
            plt.plot(np_data[j])
            plt.plot(tilde_laser[j])
            plt.title(f"{np_gain[j]}")
            plt.savefig(res_folder+f"img/sample_{j}.png")
            plt.close()

        error = []
        i=0
        for i in range(len(np_data)):
            err = np.max(np.abs(np_data[i]- tilde_laser[i])) / (np.max(np.abs(np_data[i])))
            error.append(err)

        error = np.array(error)
        error_l2 = np.linalg.norm(np_data- tilde_laser,axis=1,ord = 2)

        plt.figure()
        plt.hist(error*100, bins=30, edgecolor='black') 

        # Add labels and title
        plt.xlabel('Erreur relative (\%)$')
        plt.ylabel('Fréquence des données')
        plt.yscale("log")
        plt.title(f'{np.round(np.mean(error<0.1)*100,2)}% of data below 10% of error')
        plt.savefig(res_folder+"hist_vae.png")
        plt.close()


        plt.figure()
        plt.hist(error_l2, bins=30, edgecolor='black') 
        # Add labels and title
        plt.xlabel('Erreur L2$')
        plt.ylabel('Fréquence des données')
        plt.yscale("log")
        plt.title(f'L2')
        plt.savefig(res_folder+"hist_vae_l2.png")
        plt.close()


        plt.figure()
        index = np.argmax(error)
        plt.plot(np_data[index])
        plt.plot(tilde_laser[index])
        plt.title(f"gain {np_gain[index]}")
        plt.savefig(res_folder+"error_max.png")
        plt.show();plt.close()
        plt.close()

        plt.figure()
        index = np.argmax(error_l2)
        plt.plot(np_data[index])
        plt.plot(tilde_laser[index])
        plt.title(f"gain {np_gain[index]} error = {error_l2[index]}")
        plt.savefig(res_folder+"error_max_l2.png")
        plt.close()

        latent_space = encoder.predict(dataset_batched)[-1]

        z = latent_space
        z_mean, z_var, z = encoder.predict(dataset_batched)
        np.savetxt(res_folder+'latent_z.txt',z)

        for key in loaded_dataset['values'].keys():
            plt.hist(np.array(loaded_dataset['values'][key]))
            plt.savefig(res_folder+f'hist_{key}.png')
            plt.close()

    """ 
    GAIN NETWORK
    """

    if gain_only:
        res_folder = config["reprise"]["result_folder"]
        z = np.loadtxt(res_folder+'latent_z.txt')


    if not os.path.isdir(f"{res_folder}values/"):
        os.mkdir(f"{res_folder}values/")
        
    def train_gain(np_gain, var_name):
                
        res_folder_n = f"{res_folder}values/{var_name}/"

        if not os.path.isdir(res_folder_n):
            os.mkdir(res_folder_n)
        
        GG_values = np_gain
        min_GG = min(GG_values)
        max_GG = max(GG_values)
        normalized_GG = [(g - min_GG) / (max_GG - min_GG) for g in GG_values]
        color = cmap(normalized_GG)
        
        latent_space = z
        list_index = [(x1,x2) for x2 in range(latent_dim) for x1 in range(x2)]
        for x1,x2 in tqdm(list_index):
            plt.figure()
            color = cmap(normalized_GG)
            plt.scatter(latent_space[:,x1],latent_space[:,x2],s=5,color=color)
            sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_GG, max_GG))
            sm.set_array([])
            cbar = plt.colorbar(sm,ax=plt.gca())
            cbar.set_label('Valeur de gain')
            plt.title(f"{(x1,x2)}")
            plt.savefig(res_folder_n+f"latent_{x1}_{x2}.png")
            plt.close()

        # 2D PCA
        n = 2
        pca = PCA(n_components=n)
        pca_latent = pca.fit_transform(latent_space[:,:])
        plt.figure()
        plt.scatter(pca_latent[:,0],pca_latent[:,1],s=5,color=color)
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_GG, max_GG))
        sm.set_array([])
        cbar = plt.colorbar(sm,ax=plt.gca())
        cbar.set_label(f'Valeur de gain {pca.explained_variance_ratio_}')
        plt.savefig(res_folder_n+f'{n}_pca.png')
        plt.close()

        # 2D T-SNE vis
        tsne = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=3).fit_transform(latent_space)
        plt.figure()
        plt.scatter(tsne[:,0],tsne[:,1],s=5,color=color)
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_GG, max_GG))
        sm.set_array([])
        cbar = plt.colorbar(sm,ax=plt.gca())
        cbar.set_label('Valeur de gain')
        plt.savefig(res_folder_n+f'{n}_tsne.png')
        plt.close()


        if log:
            norm_gain = np_gain
            # norm_gain[norm_gain<0.5] = 0.5
            norm_gain = np.log(norm_gain)
        else:
            norm_gain = np_gain
            # norm_gain[norm_gain<0.5] = 0.5

        plt.figure()
        plt.hist(norm_gain)
        plt.savefig(res_folder+f'norm_hist_{var_name}.png')
        plt.close()

        dataset_gain = tf.data.Dataset.from_tensor_slices((z,(norm_gain)))

        validation_size = int(0.2 * np_gain.shape[0])

        gain_shuffled_dataset    = dataset_gain.shuffle(buffer_size=5000)
        gain_train_dataset       = gain_shuffled_dataset.skip(validation_size)
        gain_validation_dataset  = gain_shuffled_dataset.take(validation_size)

        batch_size = batch_size_rna
        gain_batched_train_dataset = gain_train_dataset.batch(batch_size)
        gain_batched_validation_dataset = gain_validation_dataset.batch(batch_size)

            
        def standard_ae_r(input_shape):

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


        latent_gain = standard_ae_r(latent_dim)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.0005,
                        decay_steps=500,
                        decay_rate=0.95,
                        staircase=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)   
        latent_gain.compile(optimizer=optimizer, loss=losses.MeanSquaredError(),metrics=['MAPE',below_10_percent])



        log_dir = res_folder_n + "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)


        callbacks=[
            # callback,
            tensorboard_callback]


        epoch = epoch_rna


        history = latent_gain.fit(gain_batched_train_dataset,
            epochs=epoch, 
            validation_data=gain_batched_validation_dataset,
            callbacks=callbacks,
            verbose = 2)

        latent_gain.save(res_folder_n + "model.keras")


        btch_data = dataset_gain.batch(32)
        predict_gain = (latent_gain.predict(btch_data)) 

        data_train = history.history['loss']
        data_val = history.history['val_loss']

        np.savetxt(res_folder_n + "losses.txt",[data_train,data_val])
        plt.figure()

        plt.grid(True,which="both")
        plt.semilogy(data_train,label="Données d'entraînement")
        plt.semilogy(data_val,label="Données de validation")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.tick_params(axis='both', which='both', direction='in')
        plt.legend(frameon=True)
        plt.savefig(res_folder_n+"losses.png")
        plt.close()


        start = 0
        epoch = history.epoch[start:]
        data_train = history.history['MAPE'][start:]
        data_val = history.history['val_MAPE'][start:]

        np.savetxt(res_folder_n + "MAPE.txt",[data_train,data_val])



        plt.figure()
        plt.grid(True,which="both")
        plt.plot(epoch,data_train,label="Données d'entraînement")
        plt.plot(epoch,data_val,label="Données de validation")
        plt.ylabel("MAPE (\%)")
        plt.xlabel("Epochs")
        plt.tick_params(axis='both', which='both', direction='in')
        plt.legend(frameon=True)
        plt.savefig(res_folder_n+"MAPE.png")
        plt.close()


        error_gain = []
        i=0
        if log:
            error_gain = np.abs(np_gain-np.exp(np.squeeze(predict_gain))) / np_gain
        else:
            error_gain = np.abs(np_gain-(np.squeeze(predict_gain))) / np_gain

        plt.figure()
        plt.hist(error_gain * 100, bins=50, edgecolor='black') 
        plt.xlabel('Erreur relative (%)')
        plt.ylabel('Fréquence des données')
        plt.title(f'{np.round(np.mean(error_gain<0.1)*100,2)}% of data below 10% of error')
        plt.legend()
        plt.savefig(res_folder_n+"hist_gain.png")
        plt.close()

        # index = np.argmax(error_gain)

        # GG_values = np_gain
        # min_GG = min(GG_values)
        # max_GG = max(GG_values)
        # normalized_GG = [(g - min_GG) / (max_GG - min_GG) for g in GG_values]

        color = cmap(normalized_GG)

        n = 2
        pca = PCA(n_components=n)
        pca_latent = pca.fit_transform(latent_space[:,:])

        plt.figure()
        plt.scatter(pca_latent[:,0],pca_latent[:,1],s=5,color=color)
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_GG, max_GG))
        sm.set_array([])
        cbar = plt.colorbar(sm,ax=plt.gca())
        cbar.set_label(f'Valeur de gain {pca.explained_variance_ratio_}')
        plt.legend()
        plt.savefig(res_folder+f'{n}_pca_gain_nom.png')
        plt.close()

        try:
            data = {
                'gain': np_gain,
                'error': error_gain * 100
            }

            df = pd.DataFrame(data)

            # Binning gain values
            bins = np.linspace(0, np.ceil(max_GG), 15)  # 10 bins from 0 to 100
            df['gain_bin'] = pd.cut(df['gain'], bins, labels=False, include_lowest=True)

            error_ranges = [0, 10, 20, 50, 100, np.inf]
            error_labels = ['0-10', '10-20','20-50', '50-100', '100+']

            df['error_range'] = pd.cut(df['error'], bins=error_ranges, labels=error_labels, include_lowest=True)
            pivot_table = df.pivot_table(index='gain_bin', columns='error_range', aggfunc='size', fill_value=0)

            # Recalcule des bins et ajustement des labels
            all_bins = range(len(bins) - 1)
            pivot_table = pivot_table.reindex(index=all_bins, fill_value=0)
            bar_positions = np.arange(len(pivot_table))

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get the bar positions and width
            bar_width = 0.7

            # Plot stacked bars
            bottom = np.zeros(len(pivot_table))
            colors = plt.cm.viridis(np.linspace(0, 1, len(error_labels)))

            for error_label, color in zip(pivot_table.columns, colors):
                ax.bar(bar_positions, pivot_table[error_label], width=bar_width, bottom=bottom, label=error_label, color=color)
                bottom += pivot_table[error_label]

            # Customize the plot
            ax.set_xlabel('Gain Value Bins')
            ax.set_ylabel('Frequency')
            ax.set_title('Histogram of Gain Values Split by Error Ranges')
            # Ajuster les labels pour correspondre aux ticks
            ax.set_xticks(bar_positions)
            ax.set_xticklabels([f'{np.round(bins[i])}-{np.round(bins[i+1])}' for i in all_bins], rotation=45)
            ax.legend(title='Error Ranges')
            plt.yscale('log')
            plt.savefig(res_folder_n+f'hist_complex.png')
            plt.close()
        except Exception as e: 
            print(e)            
            print("Error on histogram but the training will continue")


    
    for key in values:
        if key in loaded_dataset['values'].keys():
            gain_val = np.array(loaded_dataset['values'][key])
            train_gain(gain_val / np.max(gain_val), key)


if __name__ == "__main__":
    main()
    
