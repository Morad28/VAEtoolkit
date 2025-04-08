import numpy as np

data_path = "./testing_coils\std_dyn-shell_41_latent_5_kl_1e-05_256_1D-COILS-GAIN_epochs_1000\laser_shot.dat"
save_path = "./pitch_profiles/pitch1.txt"

# load the data from the .dat file
data = np.loadtxt(data_path, skiprows=1)

# transpose the data
data = data.T

# multiply by 1e-3 the second row
data[1] = data[1] * 1e-3

# save to the .txt file
np.savetxt(save_path, data, fmt='%f', delimiter=' ')
