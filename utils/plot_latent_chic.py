import sys

import tkinter as tk
from src.latent_postprocessing import PostprocessingFCI
import numpy as np
import matplotlib.pyplot as plt 
from src.dataloader import DataLoaderFCI
        

data = DataLoaderFCI(
    dataset_path='./datasets/deconding_chic.npy',
    result_folder='./testing/std_test_9435_latent_5_kl_5e-05_256/'
)

latent_space = data.latent_space
gain = data.dataset["values"]
names = data.dataset["name"]

x_arr = []
y_arr = []
for n in names:
    n_s = n.split('_')
    x, y = n_s[1], n_s[2]
    x_arr.append(float(x))
    y_arr.append(float(y))

x_arr = np.array(x_arr)
y_arr = np.array(y_arr)

mesh = np.meshgrid(x_arr, y_arr)
# print(gain["gain"])
plt.scatter(x_arr.reshape((50, 50)), y_arr.reshape((50, 50)), c= np.array(gain["yield"]).reshape((50,50)))
plt.colorbar()
plt.show()

