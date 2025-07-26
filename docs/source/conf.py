import os
import sys
import warnings
import logging

# Suppress all warnings during documentation build
warnings.filterwarnings('ignore')

# Also suppress specific Sphinx/autodoc logging
logging.getLogger('sphinx').setLevel(logging.ERROR)

# Add source path for documentation build
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../'))

project = 'Enhanced SHAP'
copyright = '2025, Niyang Bai'
author = 'Niyang Bai'
release = '0.0.1a3'
version = '0.0.1a3'

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

# Configure autodoc type hints to reduce warnings  
autodoc_typehints = 'none'  # Disable type hints to eliminate torch warnings
autodoc_typehints_description_target = 'documented'

# Mock only the most problematic imports for documentation
autodoc_mock_imports = ['torch']

# Suppress specific warnings categories
suppress_warnings = [
    'ref.python',
    'myst.xref_missing', 
    'autodoc.import_object',
    'autodoc.*',
]

def suppress_torch_warnings(app, what, name, obj, options, lines):
    """Remove torch-related warnings from documentation output."""
    if lines and any('torch' in line for line in lines):
        # Filter out lines mentioning torch issues
        filtered_lines = [line for line in lines if 'Cannot resolve forward reference' not in line]
        lines[:] = filtered_lines

def setup(app):
    """Sphinx extension setup to suppress torch warnings."""
    app.connect('autodoc-process-docstring', suppress_torch_warnings)

# Additional configuration to reduce warnings
autodoc_inherit_docstrings = False
autodoc_typehints_format = 'short'
autodoc_preserve_defaults = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

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