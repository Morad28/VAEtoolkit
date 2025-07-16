import numpy as np

data_path = "\
db_june_2/201_latent_16_kl_1e-05_256_COILS-MULTI-OUT-DUO-FOCUS_gw0.05_gl1_rl1_e99_e99min1e-06_epch5000_seploss1_smth0_gaindim5_klpr1e-05\laser_shot_e133dat"
save_path = "pitch_profiles/DB2_profile/laser_shot_e133.txt"

# load the data from the .dat file
data = np.loadtxt(data_path, skiprows=1)

# transpose the data
data = data.T

# save to the .txt file
np.savetxt(save_path, data, fmt='%f', delimiter=' ')
