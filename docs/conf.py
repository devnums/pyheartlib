# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "pyecg"
copyright = "2023, Sadegh Mohammadi"
author = "Sadegh Mohammadi"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autoapi_dirs = ["../src"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/archs", "**/extra"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# -- MyST-NB -------------------------------------------------
nb_execution_timeout = 180


# -- Custom code -----------------------------------------------------
print("\n***************\ndoc is running\n***************\n")
import shutil

files = [
    ["../model/data_preparation.py", "examples/model/"],
    ["../model/train.py", "examples/model/"],
    ["../model/inference.py", "examples/model/"],
    ["../model/result.txt", "examples/model/"],
    ["../model/plots/mis.png", "examples/model/plots/mis.png"],
]
for f in files:
    shutil.copy(f[0], f[1])
