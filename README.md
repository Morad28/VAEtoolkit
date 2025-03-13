# Installation

First create a new conda environment with Python 3.11.9:

```bash
conda create -n tf-env python=3.11.9
conda activate tf-env
```

From the root directory of this repository, install the package using pip:

```bash
pip install .
```

## Git commands

To update your local repository with the latest version of the remote repository:
```bash
git pull origin master
```

If you have modified a file, you can come back to your previous version with:

```bash
git checkout -- <file>
```
or if you have modified multiple files:
```bash
git checkout .
```

You can update by running the following command, be careful, it will try to erase all your modifications:

```bash
source update_master.sh
```

# Usage

To run the training script, you will have to create a configuration file in .json format. For example:

You can find an example of dataset under datasets/ folder that you can copy.

```json
{
    "dataset_path": "path/to/dataset/folder",
    "DataType": "1DFCI", 
    "Model": {
        "vae": "1D-FCI",
        "gain": "12MLP"
    },
    "results_dir": "path/to/results/folder",
    "name": "name_of_your_experiment",
    "epoch_vae": 5,
    "epoch_rna": 3,
    "latent_dim": 5,
    "batch_size_vae": 256,
    "batch_size_rna": 128,
    "kl_loss": 1e-05,
    "training": [
        "gain",
        "yield"
    ],
    "reprise": {
        "gain_only": 0,
        "result_folder": "path/to/results/folder"
    },
    "filter": {
        "gain": 2e-06
    }
}

```
The **training** option is here to specify what quantities you want to train. In this example, we want to train a network both for the gain and the yield.

Then you can start the training with 

```bash
vaetools conf_files/conf_FCI.json
```

or

```bash
python vaetools.py conf_files/conf_FCI.json
```

Once the training is over you can use the visualisation tool with:

```bash
vaetools path/to/results/folder
```
or if you are developping:

```bash
python vaetools.py path/to/results/folder
```

# Datasets

For FCI application, you can create a new dataset using the script provided as guide in **utils/convert_to_npy.py**. In this version of the script it will scan all folders from CHIC results and get the .dat (laser pulse) and .txt files (thermonuclear gain, yields and more can be added).

## 1D FCI dataset
It will create the .npy dataset using this dictionnary structure:
```python
{
    {
    'values': {"gain": gains, "yield": yields}, # You can add more if you want
    'data': datas,                              # Laser numpy arrays
    'time': time,                               # Time numpy arrays
    'name': names                               # Name of the folder 
    }
}
```
This structure is mandatory for the code to work.

## 2D FCI dataset
It is the same as 1D but data and time are numpy arrays of shape (512, 2), so you will have to stack them together:

```python
datas = np.column_stack((laser_pulse, target_density))
time = np.column_stack((time, x_mm))
```


# Important note

When you modify sources, you will have to perform : 

```bash
pip install .
```