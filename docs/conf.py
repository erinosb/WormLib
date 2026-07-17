# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'WormLib'
copyright = '2026, Naly Torres, Luis de Lira Aguilera, Karissa Coleman, Richard Bruno, Brian Munsky, Erin Osborne Nishimura'
author = 'Naly Torres, Luis de Lira Aguilera, Karissa Coleman, Richard Bruno, Brian Munsky, Erin Osborne Nishimura'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # pull docstrings from your code
    'sphinx.ext.napoleon',      # support Google/NumPy style docstrings
    'sphinx.ext.viewcode',      # add links to highlighted source code
    'sphinx.ext.intersphinx',   # link out to other projects' docs (e.g. numpy)
    'sphinx.ext.autosummary',   # auto-generate summary tables
    'sphinx_copybutton',        # adds a "copy" button to code blocks (pip install sphinx-copybutton)
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'
highlight_language = 'python'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'style_nav_header_background': '#2c3e50',
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'titles_only': False,
}

# Optional: add a logo/favicon if you have image files in _static/
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

# Optional custom CSS
html_css_files = [
    'custom.css',
]