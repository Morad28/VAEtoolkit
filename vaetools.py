import argparse
import os 
import tkinter as tk
from src.config_vae import get_config
from src.model import ModelSelector
from src.dataloader import (DataLoaderFCI, 
                            DataLoaderMNIST)
from src.trainer import (TrainerFCI, 
                         TrainerMNIST)

from src.latent_postprocessing import (PostprocessingFCI, 
                                       PostProcessingMNIST,
                                       PostprocessingFCI2D)
from src.diagnostics import Diagnostics

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


_DATA_TYPE = {
    "1DFCI": {
        "loader": DataLoaderFCI,
        "trainer": TrainerFCI,
        "postprocessing": PostprocessingFCI,
    },
    "MNIST": {
        "loader": DataLoaderMNIST,
        "trainer": TrainerMNIST,
        "postprocessing": PostProcessingMNIST,
    },
    "2DFCI": {
        "loader": DataLoaderFCI,
        "trainer": TrainerFCI,
        "postprocessing": PostprocessingFCI2D,
    },
}


def loader(config):
    """Get correct classes according to configuration file.

    Args:
        config (dict): Config file.

    Returns:
        tuple: (DataLoader class, Trainer class, Postprocessing class)
    """
    data_type = config.get("DataType", "1DFCI")  # Default to "1DFCI" if missing
    classes = _DATA_TYPE.get(data_type, _DATA_TYPE["1DFCI"])  # Default if not found
    
    return classes["loader"], classes["trainer"], classes["postprocessing"]
    
def main():
    """Training part of VAE model.
    """
    parser = argparse.ArgumentParser(description="This is a small python module to train a VAE model and to visualize the results.")
    parser.add_argument("path", help="Path to configuration file or result folder to visualize.", default=None)
    args = parser.parse_args() 
    
    if os.path.isfile(args.path) and args.path.endswith(".json"):
        mode = 'train'
    elif os.path.isdir(args.path):
        mode = 'visu'    

    if mode == 'train':
        config = get_config(args.path)
        data_type = config.get('DataType', '1DFCI')
        model_select = config.get('Model', {"vae" : "1D-FCI", "gain": "12MLP"})
        loader_class, trainer_class, postpro_class = loader(config)
        
        # Load dataset and preprocessing
        fci_dataset = loader_class(config)
            
        # Get VAE model
        model = ModelSelector()
        model.select(model_select)
        
        trainer = trainer_class(model, fci_dataset, config)
        trainer.train()

        diag = Diagnostics(config, trainer)
        diag.run_diagnostics()
    
    elif mode == 'visu':
        config = get_config(args.path + "/conf.json")
        loader_class, trainer_class, postpro_class = loader(config)

        data = loader_class(
            config,
            result_folder=args.path
        )

        root = tk.Tk()
        vis = postpro_class(root, data)
        root.mainloop()    
        
    else:
        print("Invalid file or foler.")
    

if __name__ == '__main__':
    main()
    
