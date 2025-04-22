import numpy as np

data_path = "pitch_profiles\DB2_profile\laser_shot_61_e22.dat"
save_path = "pitch_profiles\DB2_profile\laser_shot_61_e22.txt"

# load the data from the .dat file
data = np.loadtxt(data_path, skiprows=1)

# transpose the data
data = data.T

# multiply by 1e-3 the second row to get the pitch in meters
data[1] = data[1] * 1e-3

# save to the .txt file
np.savetxt(save_path, data, fmt='%f', delimiter=' ')
