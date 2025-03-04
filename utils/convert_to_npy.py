import os
import glob
import numpy as np
import sys
from tqdm import tqdm

path = sys.argv[1]
npy_path = sys.argv[2]

# List all .dat files in the folder
dat_files = list(glob.glob(f"{path}/*/shot_*.dat"))
txt_files = list(glob.glob(f"{path}/*/gain.txt"))
base_files = list(glob.glob(f"{path}/*")) 

print(base_files[0])


# Create a list to store the data along with gain
dataset = []

datas = []
gains = []
yields = []
adiabat = []
energy = []
time_step = []
names = []

for file in dat_files[:1]:
    time = np.loadtxt(file)
    time_old = time[:,0]
    time = np.linspace(time_old[0],time_old[-1],512)
    # print(time)

# Loop through each file, extract gain, and load the data
for b, file, txt in tqdm(list(zip(base_files, dat_files, txt_files))):
    # Open the file in read mode
    with open(txt, 'r') as f:
        lines = f.readlines()
        
    # print("Base File:", b)
    # print("DAT File:", file)
    # print("TXT File:", txt)
    # print("---------------------")
    # Initialize variables to store gain and yield
    gain = None
    yield_value = None
    adiabat_value = None
    energy_value = None
    # Iterate through each line to find the gain and yield values
    for line in lines:
        if "Thermonuclear gain" in line:
            gain = float(line.split(':')[1].strip())
        elif "Yield" in line:
            yield_value = float(line.split(':')[1].strip())   
        elif "adiabat" in line:
            adiabat_value = float(line.split(':')[1].strip())   
        elif "Incident energy laser (kJ)" in line:
            energy_value = float(line.split(':')[1].strip())   
    # Load the data from the file
    # print(file)
    data = np.loadtxt(file)  # Adjust depending on how the data is structured
    data_f = np.interp(time,data[:,0],data[:,1])
    
    datas.append(data_f)
    gains.append(gain)
    adiabat.append(adiabat_value)
    yields.append(yield_value)
    energy.append(energy_value)
    names.append(os.path.basename(b))

print(adiabat_value)
# Store the gain and data in a dictionary
dataset = {'values': {"gain": np.array(gains), 
                      "yield": np.array(yields), 
                      "adiabat": np.array(adiabat), 
                      "energy" : np.array(energy)}, 
           'data': np.array(datas), 
           'time': np.array(time), 
           'name': names
        }

if os.path.exists(npy_path):
  print("WARNING : The File %s already exists" % npy_path)  
  x = input("Type y to continue or any other key to exit : ")
else:
    x = 'y' 
    
if x != 'y':
    sys.exit("Ciao!")

# Save the dataset as a .npy file
np.save(npy_path, dataset)
