import numpy as np
import matplotlib.pyplot as plt 
import sys 


path = './datasets/smooth_data.npy'

loaded_dataset = np.load(path, allow_pickle=True)

np_gain = np.array(loaded_dataset[0]['gain'])[:,0] 
np_yield = np.array(loaded_dataset[0]['gain'])[:,1] 
np_data = np.array(loaded_dataset[0]['data'])
np_time = np.array(loaded_dataset[0]['time'])

# Define weights for x and y (for equal importance, use 0.5 each)
w_x = 0.5
w_y = 0.5

# Calculate the weighted sum for each point

weighted_sum = w_x * np_gain * 1e7 + w_y * np_yield /1e11

weighted_sum.sort()

# Find the index of the point with the maximum weighted sum
best_index = np.argsort(-weighted_sum)[:5]

# Get the best compromise point

for p in best_index:

    best_point = np_data[p]

    plt.plot(np_time,best_point,label=f'Best Compromise Point x={np_gain[p]}, y={np_yield[p] }')
    plt.legend()
    plt.show()
