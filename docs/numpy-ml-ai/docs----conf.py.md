# `numpy-ml\docs\conf.py`

```py
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# 如果需要在其他目录中使用扩展（或要使用autodoc进行文档化的模块），在这里添加这些目录。
# 如果目录相对于文档根目录，请使用os.path.abspath使其绝对化，如下所示。
#
import os
import sys
import inspect

sys.path.insert(0, os.path.abspath(".."))


gh_url = "https://github.com/ddbourgin/numpy-ml"

# -- Project information -----------------------------------------------------

project = "numpy-ml"
copyright = "2022, David Bourgin"
author = "David Bourgin"

# The short X.Y version
version = "0.1"
# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# 如果文档需要最低的Sphinx版本，请在这里声明。
#
# needs_sphinx = '1.0'

# 在这里添加任何Sphinx扩展模块名称，作为字符串。它们可以是
# 与Sphinx一起提供的扩展（命名为'sphinx.ext.*'）或您自定义的扩展。
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode"
    #  "numpydoc",
]

# 为了避免在read-the-docs构建过程中出现内存错误
autodoc_mock_imports = ["tensorflow", "torch", "gym"]

# 尝试在GitHub上链接到源代码
def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    module = info.get("module", None)
    fullname = info.get("fullname", None)

    if not module or not fullname:
        return None
    # 获取指定模块的对象
    obj = sys.modules.get(module, None)
    # 如果对象不存在，则返回 None
    if obj is None:
        return None

    # 根据完整的模块名进行循环，逐级获取属性
    for part in fullname.split("."):
        obj = getattr(obj, part)
        # 如果属性是 property 类型，则获取其 fget 方法
        if isinstance(obj, property):
            obj = obj.fget

    # 尝试获取对象对应的源文件路径
    try:
        file = inspect.getsourcefile(obj)
        # 如果源文件路径为空，则返回 None
        if file is None:
            return None
    except:
        return None

    # 将文件路径转换为相对路径
    file = os.path.relpath(file, start=os.path.abspath(".."))
    # 获取对象对应源代码的起始行号和结束行号
    source, line_start = inspect.getsourcelines(obj)
    line_end = line_start + len(source) - 1
    # 构建包含文件名和行号范围的字符串
    filename = f"{file}#L{line_start}-L{line_end}"
    # 返回包含 GitHub URL 的完整链接
    return f"{gh_url}/blob/master/{filename}"
# 设置 Napoleon 插件的配置选项
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_keyword = True

# 添加包含模板的路径，相对于当前目录
templates_path = ["_templates"]

# 源文件名的后缀
source_suffix = ".rst"

# 主目录文档
master_doc = "index"

# 自动生成内容的语言
language = None

# 忽略查找源文件时匹配的文件和目录的模式
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# 使用的 Pygments（语法高亮）样式
pygments_style = "friendly"

# 类文档的内容
autoclass_content = "both"

# HTML 输出选项
# 使用的主题
html_theme = "alabaster"

# 主题选项用于自定义主题的外观
# html_theme_options = {}
html_css_files = ["css/custom.css"]
# 添加包含自定义静态文件（如样式表）的路径，相对于当前目录
# 这些文件会在内置静态文件之后复制，因此名为"default.css"的文件将覆盖内置的"default.css"
html_static_path = ["_static"]

# 自定义侧边栏模板，必须是一个将文档名称映射到模板名称的字典
#
# 默认侧边栏（不匹配任何模式的文档）由主题自己定义。内置主题默认使用这些模板：['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']
#
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "donate.html",
    ]
}

html_theme_options = {
    "github_user": "ddbourgin",
    "github_repo": "numpy-ml",
    "description": "Machine learning, in NumPy",
    "github_button": True,
    "show_powered_by": False,
    "fixed_sidebar": True,
    "analytics_id": "UA-65839510-3",
    #  'logo': 'logo.png',
}

# -- 用于 HTML 帮助输出的选项 ---------------------------------------------

# HTML 帮助生成器的输出文件基本名称
htmlhelp_basename = "numpy-mldoc"

# -- 用于 LaTeX 输出的选项 --------------------------------------------

latex_elements = {
    # 纸张大小（'letterpaper' 或 'a4paper'）
    #
    # 'papersize': 'letterpaper',
    # 字体大小（'10pt'、'11pt' 或 '12pt'）
    #
    # 'pointsize': '10pt',
    # LaTeX 导言中的额外内容
    #
    # 'preamble': '',
    # LaTeX 图片（浮动）对齐
    #
    # 'figure_align': 'htbp',
}

# 将文档树分组为 LaTeX 文件。元组列表
# （源起始文件，目标名称，标题，
# 作者，文档类[howto、manual 或自定义类]）。
latex_documents = [
    (master_doc, "numpy-ml.tex", "numpy-ml Documentation", "David Bourgin", "manual")
]

# -- 用于手册页输出的选项 ------------------------------------------
# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "numpy-ml", "numpy-ml Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "numpy-ml",
        "numpy-ml Documentation",
        author,
        "numpy-ml",
        "Machine learning, in NumPy.",
        "Miscellaneous",
    )
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# Set the order in which members are documented in the autodoc output
autodoc_member_order = "bysource"

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for numpydocs extension -----------------------------------------
# https://numpydoc.readthedocs.io/en/latest/install.html

# Whether to produce plot:: directives for Examples sections that contain
# import matplotlib or from matplotlib import.
numpydoc_use_plots = True
# 是否自动显示类的所有成员在方法和属性部分，默认为True
numpydoc_show_class_members = True

# 是否自动显示类的所有继承成员在方法和属性部分，默认为True。如果为false，则不显示继承成员。默认为True。
numpydoc_show_inherited_class_members = True

# 是否为类方法和属性列表创建Sphinx目录。如果创建了目录，Sphinx期望每个条目有单独的页面。默认为False。
numpydoc_class_members_toctree = False

# 匹配引用的正则表达式，应该被修改以避免在文档中重复。默认为[\w-]+。
numpydoc_citation_re = r"[\w-]+"

# 在版本0.8之前，参数定义显示为块引用，而不是在定义列表中。如果您的样式需要块引用，请将此配置选项切换为True。此选项将在版本0.10中删除。
numpydoc_use_blockquotes = False

# 是否以与参数部分相同的方式格式化类页面的属性部分。如果为False，则属性部分将以使用autosummary表的方法部分格式化。默认为True。
numpydoc_attributes_as_param_list = False

# 是否为docstring的Parameters、Other Parameters、Returns和Yields部分中的参数类型创建交叉引用。默认为False。
numpydoc_xref_param_type = False

# 完全限定路径的映射（或正确的ReST引用），用于指定参数类型时使用的别名/快捷方式。键不应包含任何空格。与intersphinx扩展一起使用，可以映射到任何文档中的链接。默认为空字典。此选项取决于numpydoc_xref_param_type选项为True。
numpydoc_xref_aliases = {}
# 定义一个集合，用于存储不需要交叉引用的单词。这些单词通常是在参数类型描述中使用的常见单词，可能会与相同名称的类混淆。例如：{'type'，'optional'，'default'}。默认为空集。
numpydoc_xref_ignore = set([])

# 是否在文档字符串后插入编辑链接。已弃用自版本编辑：您的HTML模板。
numpydoc_edit_link: bool
```