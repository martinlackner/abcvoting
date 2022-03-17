# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "abcvoting"
copyright = "MIT license"
author = "Martin Lackner, Peter Regner, Benjamin Krenn, Stefan Forster, and others"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    # "sphinx.ext.coverage",
    # "sphinx.ext.napoleon",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_codeautolink",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

doctest_path = [".."]

numpydoc_validation_checks = {"all", "SA01", "EX01", "ES01", "RT03"}
numpydoc_show_class_members = False

codeautolink_custom_blocks = {
    "python3": None,
    "pycon3": "sphinx_codeautolink.clean_pycon",
}
codeautolink_autodoc_inject = True
codeautolink_global_preface = (
    "from abcvoting.abcrules import *;"
    "from abcvoting.misc import *;"
    "from abcvoting.generate import *;"
    "from abcvoting.preferences import *;"
    "from abcvoting import abcrules, misc, generate, preferences, output, properties"
)
