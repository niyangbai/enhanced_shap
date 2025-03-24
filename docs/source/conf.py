import os
import sys

project = 'ESHAP'
copyright = '2025, Niyang Bai'
author = 'Niyang Bai'

sys.path.insert(0, os.path.abspath('../../src'))

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

# Optional: Theme
html_theme = "furo"

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
