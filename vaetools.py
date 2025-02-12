import argparse
import os 
from src.config_vae import get_config
from src.model import ModelSelector
from src.dataloader import (DataLoaderFCI, 
                            DataLoaderMNIST)
from src.trainer import (TrainerFCI, 
                         TrainerMNIST)

from src.latent_postprocessing import (PostprocessingFCI, 
                                       PostprocessingMNSIT)


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


DATA_LOADER_MAP = {
    '1DFCI': DataLoaderFCI,
    'MNIST': DataLoaderMNIST,
}

TRAINER_MAP = {
    '1DFCI': TrainerFCI,
    'MNIST': TrainerMNIST,
}

POSTPRO_MAP = {
    '1DFCI': PostprocessingFCI, 
    'MNIST': PostprocessingMNSIT
}

def loader(config):
    """Get correct class according to configuration file.

    Args:
        config (dict): Config file.

    Returns:
        _type_: Classes.
    """

    # Retrieve the class name for the data loader
    loader_class_name = config.get('DataType', '1DFCI')  # Default to 'DataLoader' if not present
    
    # Get the class from the dictionary, default to DataLoader if not found
    loader_class = DATA_LOADER_MAP.get(loader_class_name, None)
    trainer_class = TRAINER_MAP.get(loader_class_name, None)
    postpro_class = POSTPRO_MAP.get(loader_class_name, None)
    
    return loader_class, trainer_class, postpro_class
    
def main():
    """Training part of VAE model.
    """
    parser = argparse.ArgumentParser(description="This is a small python module to train a VAE model and to visualize the results.")
    parser.add_argument("mode", type=str, default="train", choices=["train", "visu"], help="Choices are: train or visu.")
    parser.add_argument("path", help="Path to configuration file or result folder to visualize.", default=None)
    args = parser.parse_args()    
    
    
    if args.mode == 'train':
        config = get_config(args.path)
        data_type = config.get('DataType', '1DFCI')
        loader_class, trainer_class, postpro_class = loader(config)
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
    
    if args.mode == 'visu':
        config = get_config(args.path + "/conf.json")
        loader_class, trainer_class, postpro_class = loader(config)

        data = loader_class(
            config,
            result_folder=args.path
        )

        root = tk.Tk()
        vis = postpro_class(root, data)
        root.mainloop()    
    

if __name__ == '__main__':
    main()
    
