import sys
import os 
from src.model import ModelSelector
from src.dataloader import (DataLoaderFCI, 
                            DataLoaderMNIST)
from src.config_vae import get_config
from src.trainer import (TrainerFCI, 
                         TrainerMNIST)


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


DATA_LOADER_MAP = {
    '1DFCI': DataLoaderFCI,
    'MNIST': DataLoaderMNIST,
}

TRAINER_MAP = {
    '1DFCI': TrainerFCI,
    'MNIST': TrainerMNIST,
}

def loader(config):

    # Retrieve the class name for the data loader
    loader_class_name = config.get('DataType', '1DFCI')  # Default to 'DataLoader' if not present
    print(loader_class_name)
    # Get the class from the dictionary, default to DataLoader if not found
    loader_class = DATA_LOADER_MAP.get(loader_class_name, None)
    trainer_class = TRAINER_MAP.get(loader_class_name, None)
    
    print(loader_class)
    

    return loader_class, trainer_class
    
def main():
    config_path = sys.argv[1]
    config = get_config(config_path)
    data_type = config.get('DataType', '1DFCI')
    
    loader_class, trainer_class = loader(config)
    
    # Load dataset and preprocessing
    fci_dataset = loader_class(config)
        
    # Get VAE model
    model = ModelSelector()
    if data_type == "1DFCI":
        model.select(vae='1D-FCI', gain = '12MLP')
    elif data_type == "MNIST":
        model.select(vae = "2D-MNIST")
    
    trainer = trainer_class(model, fci_dataset, config)
    trainer.train()
    
    
if __name__ == '__main__':
    main()
    
