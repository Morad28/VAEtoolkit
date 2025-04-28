import numpy as np

data_path = "testing_coils_multi_6000\std_dyn-shell_41_latent_10_kl_0.001_256_COILS-MULTI-OUT_gw_1_gl_1_rl_1_e99_phys_0_epochs_2000\laser_shot_e31.dat"
save_path = "pitch_profiles\DB2_profile\laser_shot_e31.txt"

# load the data from the .dat file
data = np.loadtxt(data_path, skiprows=1)

# transpose the data
data = data.T

# save to the .txt file
np.savetxt(save_path, data, fmt='%f', delimiter=' ')
