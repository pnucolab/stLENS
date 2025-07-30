# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

project = 'stLENS'
copyright = '2025, khyeonm'
author = 'pnucolab'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'nbsphinx', 
    'jupyter_sphinx',
    'sphinx_design',
    'sphinx.ext.autodoc', 
    'sphinx.ext.autosummary',   
    'sphinx.ext.napoleon',   
    'sphinx.ext.viewcode',
    'autoapi.extension'
]

autoapi_dirs = ['../../src']

autoapi_options = [
    "members",
    "undoc-members",   
    "show-inheritance",
]
autoapi_ignore = ["**/PCA.py", "**/calc.py", "*/__init__.py", "**/PCA/*", "**/calc/*"]


templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "collapse_navigation": False,  
    "navigation_depth": 2,        
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'myst',
}

nbsphinx_execute = "never"

autodoc_default_options = {
    'members': True,
    'undoc-members': False,   
    'show-inheritance': True
}

autodoc_mock_imports = ["multiprocess", "scanpy"]

autoapi_keep_files = True
autoapi_add_toctree_entry = False
autosummary_generate = True

napoleon_google_docstring = False  
napoleon_numpy_docstring = True    
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
