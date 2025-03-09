"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../"))
project = "pandahandler"
copyright = "2025, Zach Kurtz"
author = "Zach Kurtz"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
    "dev.py",
    "api/modules.rst",
]
# Show inherited members in class documentation
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "private-members": False,  # Don't include private members (leading underscore)
    "special-members": False,  # Don't include special methods (__init__, etc.)
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,  # Show deeper nesting in the sidebar
    "collapse_navigation": False,  # Don't collapse navigation entries
    "sticky_navigation": True,  # Make navigation scrollable with content
    "titles_only": False,  # Show full titles in navigation
}


def linkcode_resolve(domain, info):
    """Utility function to generate GitHub source links."""
    if domain != "py":
        return None
    if not info["module"]:
        return None

    # Your GitHub username and repository name
    github_user = "zkurtz"
    github_repo = "pandahandler"

    # Get the branch or tag (typically 'main' or 'master')
    github_branch = "main"  # Change to your default branch

    filename = info["module"].replace(".", "/")
    return f"https://github.com/{github_user}/{github_repo}/blob/{github_branch}/{filename}.py"


# Try importing the package to verify it's findable
try:
    import pandahandler

    print(f"Successfully imported pandahandler from {pandahandler.__file__}")
except ImportError as e:
    print(f"ERROR: Failed to import pandahandler: {e}")
