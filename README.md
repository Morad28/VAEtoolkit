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

# Usage

To run the training script, you will have to create a configuration file in .json format. For example:

You can find an example of dataset under datasets/ folder that you can copy.

```json
{
    "DataType" : "1DFCI",
    "dataset_path": "datasets/smooth_data_testing.npy",
    "results_dir": "testing/",
    "name": "dyn-shell",
    "epoch_vae": 5,
    "epoch_rna": 3,
    "latent_dim": 5,
    "batch_size_vae": 256,
    "batch_size_rna": 128,
    "kl_loss": 1e-5,
    "training" : ["gain", "yield"],
    "reprise" : {
        "gain_only" : 0,
        "result_folder" : "./testing/std_dyn_shell_330_latent_5_kl_1e-05_256/"
    },
    "filter" : {
        "gain" : 2e-6
    }
}
```
The **training** option is here to specify what quantities you want to train. In this example, we want to train a network both for the gain and the yield.

Then you can start the training with 

```bash
vaetools train conf_files/conf_FCI.json
```

or

```bash
python vaetools.py train conf_files/conf_FCI.json
```

Once the training is over you can use the visualisation tool with:

```bash
vaetools visu path/to/results/folder
```
or if you are developping:

```bash
python vaetools.py visu path/to/results/folder
```

# Datasets

For FCI application, you can create a new dataset using the script provided as guide in **utils/convert_to_npy.py**. In this version of the script it will scan all folders from CHIC results and get the .dat (laser pulse) and .txt files (thermonuclear gain, yields and more can be added).

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

# Important note

When you modify sources, you will have to perform : 

```bash
pip install .
```