# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information --------------------------------------------------
from importlib.metadata import version

project_version = version("pyheartlib")

project = "pyheartlib"
copyright = "2023, devnums"
author = "devnums"

print("Project version:" + str(project_version))
# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_togglebutton",
]
autoapi_dirs = ["../src"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/archs", "**/extra"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
}

myst_enable_extensions = [
    "fieldlist",
    "colon_fence",
    "substitution",
]
myst_all_links_external = True
myst_heading_anchors = 7

myst_substitutions = {"versionkey": str(project_version)}

# -- Options for HTML output -----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# -- MyST-NB ---------------------------------------------------------------
nb_execution_timeout = 900
nb_execution_mode = "off"


# -- Custom code -----------------------------------------------------------
print("\n***************\ndoc is running\n***************\n")

files = [
    ["../examples/dataset/heartbeat.ipynb", "examples/"],
    ["../examples/dataset/arrhythmia.ipynb", "examples/"],
    ["../examples/dataset/rpeak.ipynb", "examples/"],
    ["../examples/model/rpeak_detection.ipynb", "examples/"],
]

import shutil  # noqa: E402

for f in files:
    shutil.copy(f[0], f[1])
