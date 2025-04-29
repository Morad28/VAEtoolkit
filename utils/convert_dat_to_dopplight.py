import numpy as np

data_path = "testing_coils_multi_6000/std_dyn-shell_42_latent_5_kl_1e-05_256_COILS-MULTI_gw_10_e99_e95_phys_0_epochs_2000/laser_shot_e38.dat"
save_path = "pitch_profiles\DB2_profile\laser_shot_e38.txt"

# load the data from the .dat file
data = np.loadtxt(data_path, skiprows=1)

# transpose the data
data = data.T

# save to the .txt file
np.savetxt(save_path, data, fmt='%f', delimiter=' ')
