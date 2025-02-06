import tkinter as tk
from src.latent_postprocessing import PostprocessingFCI
import numpy as np
import matplotlib.pyplot as plt 
from src.dataloader import DataLoaderFCI
        

data = DataLoaderFCI(
    dataset_path='../datasets/smooth_data_testing.npy',
    result_folder='../testing/std_test_9435_latent_5_kl_5e-05_256/'
)

latent_space = data.latent_space
gain = data.dataset["values"]


root = tk.Tk()
vis = PostprocessingFCI(root, data)
root.mainloop()
