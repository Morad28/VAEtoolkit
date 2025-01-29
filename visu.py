import tkinter as tk
from src.latent_postprocessing import PostprocessingVisualizer
import numpy as np
import matplotlib.pyplot as plt 
from src.dataloader import DataLoaderFCI
import argparse

# import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        

def main():    
    parser = argparse.ArgumentParser(description="Analyze VAE latent space.")
    parser.add_argument("dataset_path", help="Path to the dataset.")
    parser.add_argument("result_folder", help="Directory containing VAE results.")
    args = parser.parse_args()

    data = DataLoaderFCI(
        dataset_path=args.dataset_path,
        result_folder=args.result_folder
    )


    root = tk.Tk()
    vis = PostprocessingVisualizer(root, data)
    root.mainloop()
    
if __name__ == '__main__':
    main()


