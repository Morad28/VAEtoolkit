import sys
from src.model import ModelSelector
from src.dataloader import DataLoader, DataLoaderFCI, DataLoaderMNIST
from src.config_vae import get_config
import sys
import os
import matplotlib
from src.trainer import TrainerFCI, TrainerMNIST
matplotlib.use('Agg')
cmap = matplotlib.colormaps['viridis']  

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    # config_path = sys.argv[1]
    config_path = "conf.json"
    config = get_config(config_path)
    
    # Load dataset and preprocessing
    fci_dataset = DataLoaderMNIST(config)
        
    # Get VAE model
    model = ModelSelector()
    # model.select(vae = "1D-FCI", gain = '12MLP')
    model.select(vae = "2D-MNIST")
    
    trainer = TrainerMNIST(model, fci_dataset, config)
    trainer.train()
