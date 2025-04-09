import numpy as np 
import matplotlib.pyplot as plt 
from .trainer import Trainer
from .dataloader import DataLoader

class Diagnostics():
    def __init__(self, config, trainer : Trainer):
        self.config = config
        self.trainer = trainer 
        self.res_folder = self.trainer.res_folder
        
    def run_diagnostics(self):
        history = self.trainer.history
        history_vae = history.get("vae", None)
        training = self.config['training']
        gain_only = self.config["reprise"]["gain_only"]

        
        if not gain_only:
            self.save_loss(history_vae, self.trainer.res_folder)

        for key in training:
            self.save_loss(history.get(key,None), self.trainer.res_folder / "values" /key)
            
        self.save_errors(self.trainer.data_loader)
                
    def save_loss(self, history, res_folder):
        if history is None:
            print(f"Could not save loss in {res_folder}.")
            return
        
        data_train = history.history['loss']
        data_val = history.history['val_loss']
        
        # np.savetxt(res_folder / "losses.txt",[data_train,data_val])
        
        plt.figure()
        plt.grid(True,which="both")
        plt.semilogy(data_train,label="Données d'entraînement")
        plt.semilogy(data_val,label="Données de validation")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.tick_params(axis='both', which='both', direction='in')
        plt.legend(frameon=True)
        plt.savefig(res_folder / "losses.png")
        plt.close()
        
    def save_errors(self, data_loader : DataLoader):
        autoencoder, encoder, decoder =  self.trainer.models["vae"]
        batch_size = 256
        dataset_batched, _ = data_loader.to_dataset(batch_size=batch_size, shuffle=False, split=0)
        z = encoder.predict(dataset_batched)[-1]
        tilde_laser = decoder.predict(z)
        data, label = data_loader.get_x_y()
        
        np.savetxt(self.res_folder / 'latent_z.txt',z)

        if self.config["DataType"] == "MNIST":
            # erreur mse : somme des carrés des différences entre l'image d'origine et l'image reconstruite
            error = []
            for i in range(len(data)):
                error.append( np.sum((data[i].reshape(28,28) - tilde_laser[i].reshape(28,28))**2) / np.sum((data[i].reshape(28,28))**2) )
            plt.figure()            
            plt.hist(np.array(error) * 100 ,bins=30)
            plt.title("Erreur de reconstruction mse")
            plt.savefig(self.res_folder / "hist_mse.png")
            plt.close()

            # erreur max : plus grand écart entre un pixel de l'image d'origine et de l'image reconstruite
            error2 = []
            for i in range(len(data)):
                error2.append( np.max(np.abs(data[i].reshape(28,28) - tilde_laser[i].reshape(28,28))) / np.max(np.abs(data[i].reshape(28,28))) )
            plt.figure()
            plt.hist(np.array(error2) * 100 ,bins=30)
            plt.title("Erreur de reconstruction absolue")
            plt.savefig(self.res_folder / "hist_abs.png")
            plt.close()

            # erreur mae : somme des différences entre l'image d'origine et l'image reconstruite
            error3 = []
            for i in range(len(data)):
                error3.append( np.sum(np.abs(data[i].reshape(28,28) - tilde_laser[i].reshape(28,28))) / np.sum(np.abs(data[i].reshape(28,28))) )
            plt.figure()
            plt.hist(np.array(error3) * 100 ,bins=30)
            plt.title("Erreur de reconstruction MAE")
            plt.savefig(self.res_folder / "hist_mae.png")
            plt.close()

        elif self.config["DataType"] == "1DFCI-GAIN":
            vae_norm = data_loader.vae_norm
            predicted_gain = tilde_laser[:,-1] / self.config["gain_weight"] * vae_norm["std"] + vae_norm["mean"]
            predicted_laser = tilde_laser[:,:-1] * vae_norm["std"] + vae_norm["mean"]
            gain = data[:,-1] / self.config["gain_weight"] * vae_norm["std"] + vae_norm["mean"]
            laser = data[:,:-1] * vae_norm["std"] + vae_norm["mean"]

            error_gain = []
            for i in range(len(data)):
                error_gain.append( np.abs(gain[i] - predicted_gain[i]) / np.abs(gain[i]))
            
            error_laser = []
            for i in range(len(data)):
                error_laser.append( np.max(np.abs(laser[i] - predicted_laser[i]) / np.abs(laser[i])) )
            
            plt.figure()
            plt.hist(np.array(error_gain) * 100 ,bins=30)
            plt.title("Erreur de reconstruction du cutoff")
            plt.savefig(self.res_folder / "hist_error_cutoff.png")
            plt.close()

            plt.figure()
            plt.hist(np.array(error_laser) * 100 ,bins=30)
            plt.title("Erreur de reconstruction du profil de pas")
            plt.savefig(self.res_folder / "hist_error_pas.png")
            plt.close()

        else:
            error = []
            for i in range(len(data)):
                error.append( np.max(np.abs(data[i] - tilde_laser[i]) / np.abs(data[i])) )
            plt.figure()            
            plt.hist(np.array(error) * 100 ,bins=30)
            plt.title("Erreur de reconstruction du laser")
            plt.savefig(self.res_folder / "hist_error_laser.png")
            plt.close()

            predicted_gain = self.trainer.models["gain"].predict(z) * data_loader.gain_norm["gain"]
            gain = label * data_loader.gain_norm["gain"]
            gain_errors = []
            for i in range(len(data)):
                gain_errors.append( np.abs(gain[i] - predicted_gain[i]) / np.abs(gain[i]))
            
            plt.figure()
            plt.hist(np.array(gain_errors) * 100 ,bins=30)
            plt.title("Erreur de reconstruction du gain")
            plt.savefig(self.res_folder / "hist_error_gain.png")
            plt.close()
        

        if self.config["DataType"] == "MNIST":
            index_max_mse = np.argmax(np.array(error))
            index_max_max = np.argmax(np.array(error2))
            index_max_mae = np.argmax(np.array(error3))
            index_min_mse = np.argmin(np.array(error))
            index_min_max = np.argmin(np.array(error2))
            index_min_mae = np.argmin(np.array(error3))

            # plot the 3 couples of images with the max error and the 3 couples of images with the min error in the same figure
            plt.figure(figsize=(12, 8))
            plt.subplot(3, 2, 1)
            plt.imshow(data[index_max_mse].reshape(28,28), cmap='gray')
            plt.title("Image d'origine mse")
            plt.subplot(3, 2, 2)
            plt.imshow(tilde_laser[index_max_mse].reshape(28,28), cmap='gray')
            plt.title("Image reconstruite mse")
            plt.subplot(3, 2, 3)
            plt.imshow(data[index_max_max].reshape(28,28), cmap='gray')
            plt.title("Image d'origine abs")
            plt.subplot(3, 2, 4)
            plt.imshow(tilde_laser[index_max_max].reshape(28,28), cmap='gray')
            plt.title("Image reconstruite abs")
            plt.subplot(3, 2, 5)
            plt.imshow(data[index_max_mae].reshape(28,28), cmap='gray')
            plt.title("Image d'origine mae")
            plt.subplot(3, 2, 6)
            plt.imshow(tilde_laser[index_max_mae].reshape(28,28), cmap='gray')
            plt.title("Image reconstruite mae")
            plt.savefig(self.res_folder / "max_error.png")
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.subplot(3, 2, 1)
            plt.imshow(data[index_min_mse].reshape(28,28), cmap='gray')
            plt.title("Image d'origine mse")
            plt.subplot(3, 2, 2)
            plt.imshow(tilde_laser[index_min_mse].reshape(28,28), cmap='gray')
            plt.title("Image reconstruite mse")
            plt.subplot(3, 2, 3)
            plt.imshow(data[index_min_max].reshape(28,28), cmap='gray')
            plt.title("Image d'origine abs")
            plt.subplot(3, 2, 4)
            plt.imshow(tilde_laser[index_min_max].reshape(28,28), cmap='gray')
            plt.title("Image reconstruite abs")
            plt.subplot(3, 2, 5)
            plt.imshow(data[index_min_mae].reshape(28,28), cmap='gray')
            plt.title("Image d'origine mae")
            plt.subplot(3, 2, 6)
            plt.imshow(tilde_laser[index_min_mae].reshape(28,28), cmap='gray')
            plt.title("Image reconstruite mae")
            plt.savefig(self.res_folder / "min_error.png")
            plt.close()

        elif self.config["DataType"] == "1DFCI-GAIN":
            index_max_gain = np.argmax(np.array(error_gain))
            index_max_laser = np.argmax(np.array(error_laser))
            index_min_gain = np.argmin(np.array(error_gain))
            index_min_laser = np.argmin(np.array(error_laser))

            plt.figure()
            plt.plot(predicted_laser[index_max_laser], label=f"Reconstruction, gain = {predicted_gain[index_max_laser]} ", c = 'r')
            plt.plot(laser[index_max_laser], label=f"True, gain = {gain[index_max_laser]} ", c = 'g')
            plt.legend()
            plt.savefig(self.res_folder / "max_error_laser.png")
            plt.close()

            plt.figure()
            plt.plot(predicted_laser[index_max_gain], label=f"Reconstruction, gain = {predicted_gain[index_max_gain]} ", c = 'r')
            plt.plot(laser[index_max_gain], label=f"True, gain = {gain[index_max_gain]} ", c = 'g')
            plt.legend()
            plt.savefig(self.res_folder / "max_error_gain.png")
            plt.close()

            plt.figure()
            plt.plot(predicted_laser[index_min_laser], label=f"Reconstruction, gain = {predicted_gain[index_min_laser]} ", c = 'r')
            plt.plot(laser[index_min_laser], label=f"True, gain = {gain[index_min_laser]} ", c = 'g')
            plt.legend()
            plt.savefig(self.res_folder / "min_error_laser.png")
            plt.close()

            plt.figure()
            plt.plot(predicted_laser[index_min_gain], label=f"Reconstruction, gain = {predicted_gain[index_min_gain]} ", c = 'r')
            plt.plot(laser[index_min_gain], label=f"True, gain = {gain[index_min_gain]} ", c = 'g')
            plt.legend()
            plt.savefig(self.res_folder / "min_error_gain.png")
            plt.close()

            plt.figure()
            plt.hist(gain ,bins=30)
            plt.title("Distribution des gains")
            plt.xlabel("Gain")
            plt.ylabel("Fréquence")
            plt.yscale("log")
            plt.savefig(self.res_folder / f"hist_gain.png")
            plt.close()



        else:
            index_max = np.argmax(np.array(error))
            index_min = np.argmin(np.array(error))

            plt.figure()
            plt.plot(tilde_laser[index_max], label="Reconstruction", c = 'r')
            plt.plot(data[index_max], label="True", c = 'g')
            plt.legend()
            plt.savefig(self.res_folder / "max_error.png")
            plt.close()

            plt.figure()
            plt.plot(tilde_laser[index_min], label="Reconstruction", c = 'r')
            plt.plot(data[index_min], label="True", c = 'g')
            plt.legend()
            plt.savefig(self.res_folder / "min_error.png")
            plt.close()

        
        training = self.config['training']

        for key in training:
            l = label * data_loader.gain_norm[key]
            plt.figure()            
            plt.hist(l ,bins=30)
            plt.title("Distribution des gains")
            plt.yscale("log")
            plt.savefig(self.res_folder / f"hist_{key}.png")
            plt.close()
