# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# If the package is not installed, point Sphinx at the source tree so autoapi
# can discover the modules without needing an editable install.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "DPG"
copyright = "2024, Sylvio Barbon Junior, Leonardo Arrighi"
author = "Sylvio Barbon Junior, Leonardo Arrighi"
release = "0.1.5"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    # Auto-generate API reference pages from docstrings — works without importing
    # the package (safer for CI environments that may lack heavy ML deps).
    "autoapi.extension",
    # Google / NumPy docstring support inside napoleon.
    "sphinx.ext.napoleon",
    # Cross-link to NumPy, pandas, scikit-learn, matplotlib docs.
    "sphinx.ext.intersphinx",
    # "View Source" link on every auto-generated page.
    "sphinx.ext.viewcode",
    # Render Markdown files (index, quickstart, etc.) via MyST.
    "myst_parser",
    # Copy-button on all code blocks.
    "sphinx_copybutton",
    # Grid cards, tabs, badges — required for ::::{grid} in index.md.
    "sphinx_design",
]

# Let MyST parse both .md and .rst files.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress the "Unknown type: placeholder" warning that autoapi emits when it
# encounters type annotations referencing external objects it cannot resolve
# (e.g. numpy arrays, sklearn estimators). This is a known autoapi limitation.
suppress_warnings = ["autoapi"]

# ---------------------------------------------------------------------------
# sphinx-autoapi
# ---------------------------------------------------------------------------
autoapi_dirs = ["../dpg", "../metrics"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
# Don't re-document members imported from other modules (avoids duplicates
# when e.g. DecisionPredicateGraph is in both dpg.core and dpg.__init__).
autoapi_keep_files = False
# Put the auto-generated pages under /api/
autoapi_root = "api"
# Don't add autoapi to the TOC automatically — we control placement in index.
autoapi_add_toctree_entry = False
# Render docstrings using napoleon (Google-style).
autoapi_python_use_implicit_namespaces = False

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# ---------------------------------------------------------------------------
# Intersphinx — cross-link to external project docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
}

# ---------------------------------------------------------------------------
# MyST parser options
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# ---------------------------------------------------------------------------
# HTML output — pydata-sphinx-theme (same look as NumPy / pandas / sklearn)
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/Meta-Group/DPG",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/dpg/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": [],
}

html_context = {
    "github_user": "Meta-Group",
    "github_repo": "DPG",
    "github_version": "main",
    "doc_path": "docs",
}

html_title = "DPG"
html_short_title = "DPG"
html_css_files = ["custom.css"]
