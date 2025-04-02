import argparse 


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="No description provided."
    )

    # Add positional arguments
    parser.add_argument(
        'dataset',
        type=str,
        help='Dataset path'
    )
    
    parser.add_argument(
    'results_folder',
    type=str,
    help='Path to the saved results : all folder will go here'
    )


    # Add optional arguments
    parser.add_argument(
        '--name',
        type=str,
        default='model',
        help='Name of the specific result folder'
    )

    parser.add_argument(
        '--log',
        type=bool,
        choices=[True, False],
        default=True,
        help='Activate/Desactivate log normalisation for gain'
    )

    parser.add_argument(
        '--epoch_vae',
        type=int,
        default=100,
        help='The number of epochs for training VAE'
    )

    parser.add_argument(
        '--epoch_rna',
        type=int,
        default=100,
        help='The number of epochs for training gain network'
    )

    parser.add_argument(
        '--latent_dim',
        type=int,
        default=5,
        help='Dimension of latent space'
    )

    parser.add_argument(
        '--batch_size_vae',
        type=int,
        default=64,
        help='Dimension of latent space'
    )

    parser.add_argument(
        '--batch_size_rna',
        type=int,
        default=64,
        help='Dimension of latent space'
    )

    parser.add_argument(
        '--kl_loss',
        type=float,
        default=1e-5,
        help='kl_loss weight'
    )

    parser.add_argument(
        '--r-loss',
        type=float,
        default=1.,
        help='r_loss weight'
    )

    parser.add_argument(
        '--gain_only',
        type=bool,
        choices=[True, False],
        default=False,
        help='Train only gain network'
    )

    parser.add_argument(
        '--vae_only',
        type=bool,
        choices=[True, False],
        default=False,
        help='Train only vae network'
    )

    parser.add_argument(
        '--vae_norm',
        type=float,
        default=420,
        help='Normalize with another value'
    )

    parser.add_argument(
        '--gain_threshold',
        type=float,
        default=0.,
        help='Take only gain above this threshold'
    )

    # Parse the arguments
    args = parser.parse_args()
    
    return args

