import tkinter as tk
from src.latent_postprocessing import PostprocessingFCI
import numpy as np
import matplotlib.pyplot as plt 
from src.dataloader import DataLoaderFCI
import argparse
from src.config_vae import get_config
import os

# import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        

def main():    
    parser = argparse.ArgumentParser(description="Analyze VAE latent space.")
    parser.add_argument("dataset_path", help="Path to the dataset.")
    parser.add_argument("result_folder", help="Directory containing VAE results.")
    args = parser.parse_args()

    data = DataLoaderFCI(
        get_config("./conf.json"),
        result_folder=args.result_folder
    )

    root = tk.Tk()
    vis = PostprocessingFCI(root, data)
    root.mainloop()
    
if __name__ == '__main__':
    main()


