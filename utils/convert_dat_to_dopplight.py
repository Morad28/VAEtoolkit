import numpy as np

data_path = "db_june_1_small/201_latent_10_kl_1e-05_256_COILS-MULTI-OUT-DUO-FOCUS_gw_0.05_gl_1_rl_1_e99_e99min_1e-06_phys_0_epch_1000_seploss_1_smooth_0_gaindim_10\laser_shot_e87.dat"
save_path = "pitch_profiles\DB2_profile\laser_shot_e87.txt"

# load the data from the .dat file
data = np.loadtxt(data_path, skiprows=1)

# transpose the data
data = data.T

# save to the .txt file
np.savetxt(save_path, data, fmt='%f', delimiter=' ')
