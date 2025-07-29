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
    'jupyter_sphinx',
    'sphinx_design',
    'sphinx.ext.autodoc',    
    'sphinx.ext.napoleon',   
    'sphinx.ext.viewcode',
    'autoapi.extension'
]

autoapi_dirs = ['../../src']

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'myst',
}

autodoc_default_options = {
    'members': True,
    'undoc-members': False,   
    'show-inheritance': True
}

def autodoc_skip_member(app, what, name, obj, skip, options):
    if not hasattr(obj, "__doc__"):
        return skip

    if obj.__doc__ is None or obj.__doc__.strip() == "":
        return True

    return skip

def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)

autodoc_mock_imports = ["multiprocess"]

autoapi_keep_files = True
autoapi_add_toctree_entry = True

napoleon_google_docstring = False  
napoleon_numpy_docstring = True    
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
