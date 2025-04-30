import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Vaetools'
author = 'Morad BEN TAYEB'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
