import numpy as np

data_path = "pitch_profiles\DB2_profile\laser_shot_56_e28.dat"
save_path = "pitch_profiles\DB2_profile\laser_shot_56_e28.txt"

# load the data from the .dat file
data = np.loadtxt(data_path, skiprows=1)

# transpose the data
data = data.T

# save to the .txt file
np.savetxt(save_path, data, fmt='%f', delimiter=' ')
