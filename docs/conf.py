# -*- coding: utf-8 -*-
#
# PyDy Package documentation build configuration file, created by
# sphinx-quickstart on Fri Jul  5 12:03:22 2013.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
import datetime
import shutil

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))

import pydy

COPYRIGHT_YEARS = '2009-{}'.format(datetime.datetime.now().year)

# Extract stuff from the readme
with open('../README.rst') as f:
    readme_text = f.read()

headings = ['PyDy',  # 1
            'Installation',  # 2
            'Usage',  # 3
            'Documentation',  # 4
            'Modules and Packages',  # 5
            'Development Environment',  # 6
            'Benchmark',  # 7
            'Citation',  # 8
            'Questions, Bugs, Feature Requests',  # 9
            'Related Packages']  # 10

headings_with_lines = [heading + '\n' + '='*len(heading)
                       for heading in headings]

for h in headings_with_lines:
    readme_text = readme_text.replace(h, 'SPLIT HERE!!')

parts = readme_text.split('SPLIT HERE!!')

with open('index-opening.txt', 'w') as f:
    opening = parts[1] + '\n' + parts[8] + '\n' + parts[9]
    f.write(opening)

with open('install.rst', 'w') as f:
    bar = len(headings[1])*'='
    f.write(bar + '\n' + headings[1] + '\n' + bar + parts[2])

with open('usage.rst', 'w') as f:
    bar = len(headings[2])*'='
    f.write(bar + '\n' + headings[2] + '\n' + bar + parts[3])

shutil.copyfile('../CHANGELOG.rst', 'changelog.rst')
shutil.copyfile('../readme-msd-result.png', 'readme-msd-result.png')

# -- General configuration ------------------------------------------------

import numpydoc

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'numpydoc',
]

# NOTE : jupyter_sphinx is only available on Python 3.5+, make it an optional
# dependency and don't build the examples if not installed.
# TODO : The examples could be built when jupyter_sphinx is not available if
# the .. jupyter-execute:: directives were dynamically swapped out with ..
# code:: directives.
try:
    import jupyter_sphinx
except ImportError:
    def setup(app):
        app.add_config_value('INCLUDE_EXAMPLES', False, 'env')
    exclude_patterns = ['*/examples/*.rst']
else:
    del jupyter_sphinx

    def setup(app):
        app.add_config_value('INCLUDE_EXAMPLES', True, 'env')
    extensions.append('jupyter_sphinx')
    # NOTE : The default order causes SymPy output to show the png math images
    # instead of MathJax so I moved the LaTeX above the images.
    jupyter_execute_data_priority = [
        'application/vnd.jupyter.widget-view+json',
        'text/html',
        'text/latex',
        'image/svg+xml',
        'image/png',
        'image/jpeg',
        'text/plain',
    ]

    # NOTE: This is required so that when jupyter_sphinx executes it picks up
    # the uninstalled pydy package.
    package_path = os.path.abspath('..')
    os.environ['PYTHONPATH'] = ':'.join((package_path,
                                         os.environ.get('PYTHONPATH', '')))

numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'PyDy Distribution'
copyright = u'{}, PyDy Authors'.format(COPYRIGHT_YEARS)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = pydy.__version__
# The full version, including alpha/beta/rc tags.
release = pydy.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#html_theme = 'default'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'PyDyDistributiondoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'PyDyDistribution.tex', u'PyDy Distribution Documentation',
   u'PyDy Authors', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'pydydistribution', u'PyDy Distribution Documentation',
     [u'PyDy Authors'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'PyDyDistribution', u'PyDy Distribution Documentation',
   u'PyDy Authors', 'PyDyDistribution', 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False


# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = u'PyDy Distribution'
epub_author = u'PyDy Authors'
epub_publisher = u'PyDy Authors'
epub_copyright = u'{}, PyDy Authors'.format(COPYRIGHT_YEARS)

# The language of the text. It defaults to the language option
# or en if the language is not set.
#epub_language = ''

# The scheme of the identifier. Typical schemes are ISBN or URL.
#epub_scheme = ''

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#epub_identifier = ''

# A unique identification for the text.
#epub_uid = ''

# A tuple containing the cover image and cover page html template filenames.
#epub_cover = ()

# A sequence of (type, uri, title) tuples for the guide element of content.opf.
#epub_guide = ()

# HTML files that should be inserted before the pages created by sphinx.
# The format is a list of tuples containing the path and title.
#epub_pre_files = []

# HTML files shat should be inserted after the pages created by sphinx.
# The format is a list of tuples containing the path and title.
#epub_post_files = []

# A list of files that should not be packed into the epub file.
#epub_exclude_files = []

# The depth of the table of contents in toc.ncx.
#epub_tocdepth = 3

# Allow duplicate toc entries.
#epub_tocdup = True

# Choose between 'default' and 'includehidden'.
#epub_tocscope = 'default'

# Fix unsupported image types using the PIL.
#epub_fix_images = False

# Scale large images.
#epub_max_image_width = 0

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#epub_show_urls = 'inline'

# If false, no index is generated.
#epub_use_index = True


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None}
