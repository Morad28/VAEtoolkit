import tkinter as tk
from src.latent_postprocessing import PostprocessingFCI, PostprocessingMNSIT
import numpy as np
import matplotlib.pyplot as plt 
from src.dataloader import DataLoaderFCI, DataLoaderMNIST
import argparse
from src.config_vae import get_config
import os

# import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        

# def main():    
#     parser = argparse.ArgumentParser(description="Analyze VAE latent space.")
#     parser.add_argument("result_folder", help="Directory containing VAE results.")
#     args = parser.parse_args()

#     data = DataLoaderFCI(
#         get_config(args.result_folder + "/conf.json"),
#         result_folder=args.result_folder
#     )

#     root = tk.Tk()
#     vis = PostprocessingFCI(root, data)
#     root.mainloop()
    
def main():    
    parser = argparse.ArgumentParser(description="Analyze VAE latent space.")
    parser.add_argument("result_folder", help="Directory containing VAE results.")
    args = parser.parse_args()

    data = DataLoaderMNIST(
        get_config(args.result_folder + "/conf.json"),
        result_folder=args.result_folder
    )

    root = tk.Tk()
    vis = PostprocessingMNSIT(root, data)
    root.mainloop()
    
if __name__ == '__main__':
    main()


