import numpy as np

data_path = "db_june_2/201_latent_5_kl_0_256_COILS-MULTI-OUT-DUO-FOCUS_gw0.05_gl1_rl1_e99_e99min1e-06_epch5000_seploss1_smth0_gaindim5_klac_wrmp700_cyc1000_klpr0\laser_shot_45.dat"
save_path = "pitch_profiles\DB2_profile\laser_shot_e45.txt"

# load the data from the .dat file
data = np.loadtxt(data_path, skiprows=1)

# transpose the data
data = data.T

# save to the .txt file
np.savetxt(save_path, data, fmt='%f', delimiter=' ')
