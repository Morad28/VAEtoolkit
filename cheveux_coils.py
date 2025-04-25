# pitch generation around a specific pitch profile

import numpy as np
import matplotlib.pyplot as plt
import os

data_path = "pitch_profiles\DB2_profile\laser_shot_56_e28.txt"

# Load the data from the text file
data = np.loadtxt(data_path)

# Create sinusoidal variations within the envelope
data_copies = []
for i in range(100):
    # Create a copy of the data
    data_copy = np.zeros((2, 100))

    # upscale the data from 40 points to 100 points
    x = np.linspace(data[0, 0], data[0, -1], 100)
    y = np.interp(x, data[0], data[1])
    data_copy[0] = x
    data_copy[1] = y

    sin_amplitude = np.random.uniform(0.000005, 0.00003)  # Random amplitude for sinusoidal variation
    sin_frequency = np.random.uniform(30, 500)  # Random frequency for sinusoidal variation

    # Add sinusoidal variations
    sinusoidal_variation = sin_amplitude * np.sin(2 * np.pi * sin_frequency * data_copy[0])
    data_copy[1] = data_copy[1] + sinusoidal_variation

    # Append the modified copy to the list
    data_copies.append(data_copy)

# Plot the original and modified profiles
plt.figure()
plt.plot(data[0], data[1], label='Original Data', color='blue')
for i in range(5):  # Plot a few modified profiles
    plt.plot(data_copies[i][0], data_copies[i][1], label=f'Modified Data {i}', alpha=0.7)
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Data with Sinusoidal Variations')
plt.legend()
plt.grid()
plt.show()

# Save the modified profiles
folder_path = "pitch_profiles/modified_data_with_sinusoidal_variations"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for i, data_copy in enumerate(data_copies):
    file_path = os.path.join(folder_path, f"coil_h_profile_{i}.txt")
    np.savetxt(file_path, data_copy)
    print(f"Saved modified data to {file_path}")