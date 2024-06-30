# `D:\src\scipysrc\seaborn\doc\conf.py`

```
# Sphinx 文档生成器的配置文件。

# -- Path setup --------------------------------------------------------------
# 如果扩展（或要与自动文档一起使用的模块）位于另一个目录中，
# 在这里添加这些目录到 sys.path。如果目录相对于文档根目录，
# 使用 os.path.abspath 将其转换为绝对路径，如下所示。
import os
import sys
import time
import seaborn
from seaborn._core.properties import PROPERTIES

sys.path.insert(0, os.path.abspath('sphinxext'))


# -- Project information -----------------------------------------------------
# 项目信息设定
project = 'seaborn'  # 项目名称
copyright = f'2012-{time.strftime("%Y")}'  # 版权
author = 'Michael Waskom'  # 作者
version = release = seaborn.__version__  # 版本


# -- General configuration ---------------------------------------------------
# 一般配置设定

# 添加任何 Sphinx 扩展模块名称到这里，作为字符串。它们可以是与 Sphinx 一起提供的扩展（如 'sphinx.ext.*'）或自定义的扩展。
extensions = [
    'sphinx.ext.autodoc',  # 自动文档生成
    'sphinx.ext.doctest',  # doctest 测试
    'sphinx.ext.coverage',  # 测试覆盖率
    'sphinx.ext.mathjax',  # 数学公式支持
    'sphinx.ext.autosummary',  # 自动生成摘要
    'sphinx.ext.intersphinx',  # 其他文档的交叉引用
    'matplotlib.sphinxext.plot_directive',  # Matplotlib 绘图指令
    'gallery_generator',  # 图库生成器
    'tutorial_builder',  # 教程生成器
    'numpydoc',  # NumPy 风格的文档支持
    'sphinx_copybutton',  # 复制按钮
    'sphinx_issues',  # GitHub 问题支持
    'sphinx_design',  # 设计元素支持
]

# 模板路径，相对于当前目录
templates_path = ['_templates']

# 根文档
root_doc = 'index'

# 忽略的文件和目录模式，相对于源目录
exclude_patterns = ['_build', 'docstrings', 'nextgen', 'Thumbs.db', '.DS_Store']

# reST 默认角色（用于文本标记），用于所有文档的默认角色
default_role = 'literal'

# 构建时生成 API 文档
autosummary_generate = True

# 不显示类成员
numpydoc_show_class_members = False


# Sphinx-issues 配置
issues_github_path = 'mwaskom/seaborn'


# 包含 API 文档中绘图示例的源代码
plot_include_source = True

# 绘图格式设定
plot_formats = [('png', 90)]

# 不在 HTML 中显示绘图格式选项
plot_html_show_formats = False

# 不在 HTML 中显示源代码链接
plot_html_show_source_link = False


# 侧边栏中不显示源代码链接
html_show_sourcelink = False

# 控制类型提示的外观
autodoc_typehints = "none"
autodoc_typehints_format = "short"

# 允许使用主函数接口的简写引用
rst_prolog = """
.. currentmodule:: seaborn
"""

# 定义替换（用于 WhatsNew 的条目）
rst_epilog = r"""
.. role:: raw-html(raw)
   :format: html

.. role:: raw-latex(raw)
   :format: latex

.. |API| replace:: :raw-html:`<span class="badge badge-api">API</span>` :raw-latex:`{\small\sc [API]}`
"""  # noqa

# 将各项属性的文档字符串添加到 reStructuredText 文档的末尾
rst_epilog += "\n".join([
    # 为每个属性生成一个 reStructuredText 的替换指令，链接到相应属性的类文档
    f".. |{key}| replace:: :ref:`{key} <{val.__class__.__name__.lower()}_property>`"
    for key, val in PROPERTIES.items()
])

# -- Options for HTML output -------------------------------------------------

# HTML 输出的主题设置为 'pydata_sphinx_theme'
html_theme = 'pydata_sphinx_theme'

# 添加包含自定义静态文件（如样式表）的路径，相对于当前目录，这些文件会复制到内置静态文件之后
html_static_path = ['_static', 'example_thumbs']

# 如果路径不存在，则创建路径
for path in html_static_path:
    if not os.path.exists(path):
        os.makedirs(path)

# 定义一个列表，包含自定义的 CSS 文件，版本号基于 seaborn 库的版本
html_css_files = [f'css/custom.css?v={seaborn.__version__}']

# 指定 HTML 页面的 logo 和 favicon 文件路径
html_logo = "_static/logo-wide-lightbg.svg"
html_favicon = "_static/favicon.ico"

# 配置 HTML 主题选项
html_theme_options = {
    # 配置顶部图标链接
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mwaskom/seaborn",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "StackOverflow",
            "url": "https://stackoverflow.com/tags/seaborn",
            "icon": "fab fa-stack-overflow",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/michaelwaskom",
            "icon": "fab fa-twitter",
            "type": "fontawesome",
        },
    ],
    # 禁用默认的“上一页”和“下一页”链接
    "show_prev_next": False,
    # 导航栏左侧显示 logo
    "navbar_start": ["navbar-logo"],
    # 导航栏右侧显示图标链接
    "navbar_end": ["navbar-icon-links"],
    # 在下拉菜单之前显示的顶部链接数
    "header_links_before_dropdown": 8,
}

# 定义 HTML 页面的上下文变量，设置默认模式为亮色模式
html_context = {
    "default_mode": "light",
}

# 定义 HTML 页面的侧边栏设置
html_sidebars = {
    # 针对不同页面设置不同的侧边栏，通配符 "**" 匹配所有页面
    "index": [],
    "examples/index": [],
    "**": ["sidebar-nav-bs.html"],
}

# -- Intersphinx ------------------------------------------------

# 配置 intersphinx 映射，提供外部库的链接和说明
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None)
}
```