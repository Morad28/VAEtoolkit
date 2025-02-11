from setuptools import setup, find_packages

setup(
    name="vaetools",                 # Your package name
    version="0.1.0",                   # Initial version
    packages=find_packages(),          # Automatically find submodules
    description="This package is for VAE training and latent space visualization.",
    author="Morad BEN TAYEB",
    author_email="morad.ben-tayeb@u-bordeaux.fr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",      # Choose your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",           # Minimum Python version
    
    py_modules=["vaetools"],  # `main.py` must be in the same directory as `setup.py`

    entry_points={
        "console_scripts": [
            "vaetools=vaetools:main", 
        ],
    },
    
    install_requires=[
        "keras>=3.3.2",
        "matplotlib>=3.8.4",
        "numpy",
        "pandas",
        "scikit_learn",
        "tensorflow>=2.16.1",
        "tqdm>=4.66.2",
    ],
)