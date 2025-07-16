import argparse
import os
import tkinter as tk
from src.config_vae import get_config
from src.model import ModelSelector
from src.dataloader import DataLoaderFCI, DataLoaderMNIST, DataLoaderGain, DataLoaderCoilsMulti
from src.trainer import TrainerFCI, TrainerMNIST, TrainerGain
from src.latent_postprocessing import (
    PostprocessingFCI,
    PostProcessingMNIST,
    PostProcessingFCI2D,
    PostprocessingGain,
    PostprocessingCoilsMulti,
)
from src.diagnostics import Diagnostics

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DATA_TYPE_MAP = {
    "1DFCI": {
        "loader": DataLoaderFCI,
        "trainer": TrainerFCI,
        "postprocessing": PostprocessingFCI,
    },
    "1DFCI-GAIN": {
        "loader": DataLoaderGain,
        "trainer": TrainerGain,
        "postprocessing": PostprocessingGain,
    },
    "COILS-MULTI": {
        "loader": DataLoaderCoilsMulti,
        "trainer": TrainerGain,
        "postprocessing": PostprocessingCoilsMulti,
    },
    "MNIST": {
        "loader": DataLoaderMNIST,
        "trainer": TrainerMNIST,
        "postprocessing": PostProcessingMNIST,
    },
    "2DFCI": {
        "loader": DataLoaderFCI,
        "trainer": TrainerFCI,
        "postprocessing": PostProcessingFCI2D,
    },
}

def get_classes(config):
    """Retrieve appropriate classes based on configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: DataLoader class, Trainer class, Postprocessing class.
    """
    data_type = config.get("DataType", "1DFCI")
    classes = DATA_TYPE_MAP.get(data_type, DATA_TYPE_MAP["1DFCI"])
    return classes["loader"], classes["trainer"], classes["postprocessing"]

def main():
    """Main function for training or visualizing the VAE model."""
    parser = argparse.ArgumentParser(
        description="Train a VAE model or visualize the results."
    )
    parser.add_argument(
        "path",
        help="Path to the configuration file or result folder for visualization.",
        default=None,
    )
    args = parser.parse_args()

    if os.path.isfile(args.path) and args.path.endswith(".json"):
        mode = "train"
    elif os.path.isdir(args.path):
        mode = "visualize"
    else:
        print(f"Invalid file or folder: {args.path}")
        return

    if mode == "train":
        config = get_config(args.path)
        model_config = config.get("Model", {"vae": "1D-FCI", "gain": "12MLP"})
        loader_class, trainer_class, postprocessing_class = get_classes(config)

        dataset = loader_class(config)

        model_selector = ModelSelector()
        model_selector.select(model_config)

        trainer = trainer_class(model_selector, dataset, config)
        trainer.train()

        diagnostics = Diagnostics(config, trainer)
        diagnostics.run_diagnostics()

    elif mode == "visualize":
        config = get_config(os.path.join(args.path, "conf.json"))
        loader_class, _, postprocessing_class = get_classes(config)

        dataset = loader_class(config, result_folder=args.path)

        root = tk.Tk()
        visualization = postprocessing_class(root, dataset)
        root.mainloop()

if __name__ == "__main__":
    main()