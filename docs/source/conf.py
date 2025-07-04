import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'Enhanced SHAP'
copyright = '2025, Niyang Bai'
author = 'Niyang Bai'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
    'sphinx_autodoc_typehints',
]

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True,
}

templates_path = ['_templates']
exclude_patterns = []

html_theme = "furo"
html_static_path = ['_static']

html_theme_options = {
    "light_logo": "fau_wiso_light.png",
    "dark_logo": "fau_wiso_dark.png",
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}