import sys
sys.path.append('./src')

from latent_postprocessing import visu_latent
import numpy as np
import matplotlib.pyplot as plt 
        
    
prepro = [lambda x: np.exp(x) / 1e7, lambda x: np.exp(x) * 1e11 ]



vis = visu_latent()

# Load the training dataset 
# data, gain, time = vis.load_dataset('./data_dyn_shell.npy')
# data, gain, time = vis.load_dataset('./datasets/smooth_data.npy')
data, gain, time = vis.load_dataset('./datasets/dataset_2185.npy')

# Load the result folder you want to analyze 
# vis.load_model('./test/std_testJL_3931_latent_5_kl_5e-05_128')
# vis.load_model('./test/std_test_3931_latent_5_kl_5e-05_128')
# vis.load_model('./test/std_test_3931_latent_5_kl_5e-05_128') # TO CHIC
# vis.load_model('./test/std_yield_3931_latent_5_kl_5e-05_128')

vis.load_model('./results/std_full_2185_latent_5_kl_5e-05_256') # TO CHIC

# It will run the visualization routine.

vis.run(0,1,enablePca = True,prepro=prepro)
