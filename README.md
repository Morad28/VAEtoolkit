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

## Coils datasets
For coils datasets, best practice is to have a dataset with this format, although some keys may not be explored by certain configurations:

```python
{
    'values': {"cutoff": cutoffs, "qsup": qsups, "e99", e99s},      # You can add more scalar values if you want
    'data': {"pitch": pitch_profiles, "radius": radius_profiles},   # Coil data numpy arrays
    'time': coil_z_axis,                                            # Only one z axis for all coils
    'name': names                                                   # Names of the coils, never used, but here for consistency
}
```

For the coils datasets, certain AI models may not be compatible with the data structure, so you will have to adapt the code to your needs. This is usually caused by the length of the profiles (mostly 40 or 100 points), which is hard coded in order to be able to use CNNs correctly.

# Specific configurations
The conf_coils.json file leads to a training both a VAE, and a MLP to predict the gain from the latent space. It is the most basic configuration, and it is used to train on coils datasets. Most settings and specific options are unused, but it is the first configuration used for coils training.

The conf_coils_gain.json leads to a training using only a VAE, and incorporating the gain as a scalar value at the end of the profile.

The conf_coils_multi_values.json leads to a training using a VAE, and incorporating multiple values (cutoff, qsup, e99) to the latent space. There are many options for the model vae to choose from, check out model.py for details.

The conf_coils_multi_values_singleval.json leads to a training using a VAE, and incorporating the gain as a scalar value in the latent space, while using multiple values (cutoff, qsup, e99) as inputs.

In model.py, you can find the different models available for training. You can also create your own model by inheriting from the base class.
For training on coils, the COILS-MULTI-OUT-DUO model is the best to create a new model from, as it is the first one to take into account the possibility of training on both pitch and radius profiles at the same time, and uses a double encoder, double decoder structure, which was found to be most efficient.

# Important note

When you modify sources, you will have to perform : 

```bash
pip install .
```
