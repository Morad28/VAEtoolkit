import os
import glob
import numpy as np
import sys
from tqdm import tqdm

path = sys.argv[1]
npy_path = sys.argv[2]


dat_files = list(glob.glob(f"{path}/*.dat"))
dat_files.sort()


# Create a list to store the data along with gain
dataset = []

datas = []
gains = []
yields = []
time_step = []
names = []

for file in dat_files[:1]:
    time = np.loadtxt(file)
    time_old = time[:,0]
    time = np.linspace(time_old[0],time_old[-1],512)
    # print(time)

# Loop through each file, extract gain, and load the data
for file in tqdm(list(dat_files)):
    # Open the file in read mode


    # Load the data from the file
    # print(file)
    data = np.loadtxt(file)  # Adjust depending on how the data is structured
    data_f = np.interp(time,data[:,0],data[:,1])
    
    gain = float(os.path.basename(file).split('_')[0])
    
    datas.append(data_f)
    gains.append(gain)
    names.append(os.path.basename(file))


    # Store the gain and data in a dictionary
dataset = {'values': {"gain": gains}, 'data': datas, 'time': time, 'name': names}

if os.path.exists(npy_path):
  print("WARNING : The File %s already exists" % npy_path)  
  x = input("Type y to continue or any other key to exit : ")
else:
    x = 'y' 
    
if x != 'y':
    sys.exit("Ciao!")

# Save the dataset as a .npy file
np.save(npy_path, dataset)
