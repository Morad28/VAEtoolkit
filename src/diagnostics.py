import numpy as np 
import matplotlib.pyplot as plt 
from .trainer import Trainer
from .dataloader import DataLoader

class Diagnostics():
    def __init__(self, config, trainer : Trainer):
        self.config = config
        self.trainer = trainer 
        
    def run_diagnonstics(self):
        history = self.trainer.history
        history_vae = history.get("vae", None)
        training = self.config['training']

        self.save_loss(history_vae, self.trainer.res_folder)

        for key in training:
            self.save_loss(history.get(key,None))
            
        self.save_errors(self.trainer.data_loader)
                
    def save_loss(self, history, res_folder):
        if history is None:
            print(f"Could not save loss in {res_folder}.")
            return
        
        data_train = history.history['loss']
        data_val = history.history['val_loss']
        
        np.savetxt(res_folder / "losses.txt",[data_train,data_val])
        
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
        autoencoder, encoder, decoder =  self.trainer.models
        batch_size = 256
        dataset_batched, _ = data_loader.to_dataset(batch_size=batch_size, shuffle=False, split=0)
        _, _, z = encoder.predict(dataset_batched)
        tilde_laser = decoder.predict(z)
        data, label = self.data_loader.get_x_y()
        
        np.savetxt(self.res_folder / 'latent_z.txt',z)

        error = []
        for i in range(len(data)):
            error.append( np.max(np.abs(data[i] - tilde_laser[i])) / np.max(np.abs(data[i])) )

        plt.figure()            
        plt.hist(np.array(error) * 100 ,bins=30)
        plt.title("Erreur de reconstruction")
        plt.savefig(self.res_folder / "hist_error.png")
        plt.close()
        
        plt.figure()            
        plt.hist(label ,bins=30)
        plt.title("Distribution des gains")
        plt.yscale("log")
        plt.savefig(self.res_folder / "hist_gain.png")
        plt.close()