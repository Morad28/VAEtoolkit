import sys
import os 
from src.model import ModelSelector
from src.dataloader import DataLoader, DataLoaderFCI, DataLoaderMNIST
from src.config_vae import get_config
from src.trainer import TrainerFCI, TrainerMNIST


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# def main():
#     config_path = sys.argv[1]
#     config = get_config(config_path)
    
#     # Load dataset and preprocessing
#     fci_dataset = DataLoaderFCI(config)
        
#     # Get VAE model
#     model = ModelSelector()
#     model.select(vae = "1D-FCI", gain = '12MLP')
    
#     trainer = TrainerFCI(model, fci_dataset, config)
#     trainer.train()
    
def main():
    config_path = sys.argv[1]
    config = get_config(config_path)
    
    # Load dataset and preprocessing
    fci_dataset = DataLoaderMNIST(config)
        
    # Get VAE model
    model = ModelSelector()
    model.select(vae = "2D-MNIST")
    
    trainer = TrainerMNIST(model, fci_dataset, config)
    trainer.train()
    
if __name__ == '__main__':
    main()
    
