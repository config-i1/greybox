"""Sphinx configuration for greybox documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "greybox"
copyright = "2025, Ivan Svetunkov"
author = "Ivan Svetunkov"

release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

htmlhelp_basename = "greyboxdoc"

texinfo_documents = [
    (
        "index",
        "greybox",
        "Greybox Documentation",
        "Ivan Svetunkov",
        "greybox",
        "Toolbox for model building and forecasting.",
        "Scientific/Engineering",
    ),
]

epub_titles = [
    (
        "index",
        "Greybox Documentation",
        "Ivan Svetunkov",
    ),
]

epub_exclude_files = ["search.html"]

language = "en"

master_doc = "index"

