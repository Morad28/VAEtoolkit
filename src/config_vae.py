import json
import sys

def get_config(path):
    # Default values
    default_config = {
        "results_dir": "results/",                # Directory to save results
        "name": "",                              # Experiment name
        "epoch_vae": 1,                          # Number of epochs for VAE training
        "epoch_rna": 1,                          # Number of epochs for RNA training
        "latent_dim": 10,                        # Latent space dimension
        "batch_size_vae": 64,                    # Batch size for VAE
        "batch_size_rna": 32,                    # Batch size for RNA
        "kl_loss": 1e-4,                         # Weight for KL divergence loss
        "r_loss": 1.0,                           # Weight for reconstruction loss
        "vae_normalization": 1,                  # Normalization factor of the VAE input, should be unused as the normalization is now done in the dataset as standardization
        "log": False,                            # Enable logging

        # Additional parameters from JSON config
        "DataType": "",                          # Type of data (e.g., COILS-MULTI)
        "Model": {},                             # Model configuration dictionary, e.g., {"vae": vae type}
        "dataset_path": "",                      # Path to dataset file
        "sep_loss": 0,                           # Whether or not to separate the loss for the profile and the values
        "smooth": 0,                             # Whether or not to add a smoothing penalty for training
        "min_value": 0.0,                        # Minimum value for data filtering, used in the physical penalty
        "gain_loss": 0,                          # Gain loss weight
        "radius_loss": 0,                        # Radius loss weight (if multiple channels in profile)
        "pitch_loss": 0,                         # Pitch loss weight (if multiple channels in profile)
        "smooth_loss": 0,                        # Smooth loss weight
        "physical_penalty_weight": 0,            # Weight for physical penalty
        "gain_weight": 0.0,                      # Weight for the values
        "gain_latent_size": 0,                   # Latent size for gain (size of the vector to be concatenated with the profile vector when separating the encoders)
        "training": [],                          # Training configuration list
        "values": [],                            # List of values to use
        "predict_z_mean": 0,                     # Whether or not to predict not from the latent space but just an encoding vector (VAE --> simple AE)
        "num_components": 1,                     # Number of components for the MoG (mixture of gaussians), if MoG is used as the latent distribution
        "profile_types": 1,                      # Number of profile types
        "heatmap": "",                           # Value to plot the heatmap for in postprocessing
        "reprise": {},                           # Reprise configuration dictionary
        "filter": {},                            # Filter configuration dictionary
    }

    # Load configuration file
    with open(path, 'r') as f:
        user_config = json.load(f)

    # Merge user configuration with default values
    config = {**default_config, **user_config}
    return config