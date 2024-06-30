# `D:\src\scipysrc\sympy\doc\src\conf.py`

```
# 导入必要的模块和库

import sys  # 导入sys模块，用于系统相关操作
import inspect  # 导入inspect模块，用于检查活动对象的内部信息
import os  # 导入os模块，用于操作系统相关功能
import subprocess  # 导入subprocess模块，用于创建子进程，执行外部命令
from datetime import datetime  # 从datetime模块中导入datetime类，用于处理日期和时间

# 确保从Git中导入sympy库
sys.path.insert(0, os.path.abspath('../..'))

import sympy  # 导入sympy库，进行符号计算

# 如果您的扩展在另一个目录中，请在这里添加路径
sys.path = ['ext'] + sys.path

# 总体配置
# ---------------------

# 在这里添加任何Sphinx扩展模块的名称，作为字符串。它们可以是随Sphinx提供的扩展（命名为'sphinx.addons.*'）或您自定义的扩展。
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.linkcode',
              'sphinx_math_dollar', 'sphinx.ext.mathjax', 'numpydoc',
              'sphinx_reredirects', 'sphinx_copybutton',
              'sphinx.ext.graphviz', 'sphinxcontrib.jquery',
              'matplotlib.sphinxext.plot_directive', 'myst_parser',
              'convert-svg-to-pdf', 'sphinx.ext.intersphinx', ]

# 添加重定向规则。当现有发布文档中的页面移动到其他位置时，应添加重定向，以防止URL断裂。
# 格式如下：

# "旧页面/路径/不包含/扩展名": "../新页面/相对路径_with.html"

# 注意，html路径是相对于重定向页面的。始终手动测试重定向（它们不会自动测试）。
# 参见 https://documatt.gitlab.io/sphinx-reredirects/usage.html

redirects = {
    "guides/getting_started/install": "../../install.html",
    "documentation-style-guide": "contributing/documentation-style-guide.html",
    "gotchas": "explanation/gotchas.html",
    "special_topics/classification": "../explanation/classification.html",
    "special_topics/finite_diff_derivatives": "../explanation/finite_diff_derivatives.html",
    "special_topics/intro": "../explanation/index.html",
    "special_topics/index": "../explanation/index.html",
    "modules/index": "../reference/index.html",
    "modules/physics/index": "../../reference/public/physics/index.html",

    "guides/contributing/index": "../../contributing/index.html",
    "guides/contributing/dev-setup": "../../contributing/dev-setup.html",
    "guides/contributing/dependencies": "../../contributing/dependencies.html",
    "guides/contributing/build-docs": "../../contributing/new-contributors-guide/build-docs.html",
    "guides/contributing/debug": "../../contributing/debug.html",
    "guides/contributing/docstring": "../../contributing/docstring.html",
    "guides/documentation-style-guide": "../../contributing/contributing/documentation-style-guide.html",
    # 定义一个字典，将每个键（路径别名）映射到其对应的文件路径
    path_mapping = {
        "guides/make-a-contribution": "../../contributing/make-a-contribution.html",
        "guides/contributing/deprecations": "../../contributing/deprecations.html",
    
        "tutorial/preliminaries": "../tutorials/intro-tutorial/preliminaries.html",
        "tutorial/intro": "../tutorials/intro-tutorial/intro.html",
        "tutorial/index": "../tutorials/intro-tutorial/index.html",
        "tutorial/gotchas": "../tutorials/intro-tutorial/gotchas.html",
        "tutorial/features": "../tutorials/intro-tutorial/features.html",
        "tutorial/next": "../tutorials/intro-tutorial/next.html",
        "tutorial/basic_operations": "../tutorials/intro-tutorial/basic_operations.html",
        "tutorial/printing": "../tutorials/intro-tutorial/printing.html",
        "tutorial/simplification": "../tutorials/intro-tutorial/simplification.html",
        "tutorial/calculus": "../tutorials/intro-tutorial/calculus.html",
        "tutorial/solvers": "../tutorials/intro-tutorial/solvers.html",
        "tutorial/matrices": "../tutorials/intro-tutorial/matrices.html",
        "tutorial/manipulation": "../tutorials/intro-tutorial/manipulation.html",
    }
}

# 在这里结束了前面的配置段落，下面开始定义全局变量和设置项

# Sphinx文档生成工具需要使用的基础HTML链接地址
html_baseurl = "https://docs.sympy.org/latest/"

# 配置Sphinx复制按钮的提示文本和正则表达式设置
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# 使用pngmath代替MathJax的配置选项，目前被注释掉
#extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.pngmath', ]

# 启用所有坏交叉引用的警告，这些会在Makefile的-W标志下变成错误
nitpicky = True

# 忽略特定的nitpick错误，这里忽略了('py:class', 'sympy.logic.boolalg.Boolean')的交叉引用错误
nitpick_ignore = [
    ('py:class', 'sympy.logic.boolalg.Boolean')
]

# 停止文档字符串的继承
autodoc_inherit_docstrings = False

# 配置MathJax3的选项，用于LaTeX数学公式的显示设置
mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

# Myst解析器的配置，用于处理.md文件，包括Dollarmath、链接处理和任务列表等扩展
myst_enable_extensions = ["dollarmath", "linkify", "tasklist"]
myst_heading_anchors = 6
# 使任务列表中的复选框可以勾选，需要特定的解析器支持
# myst_enable_checkboxes = True
# myst_update_mathjax = False

# 不将链接化为链接，除非以"https://"开头，以避免处理.py作为顶级域名
myst_linkify_fuzzy_links = False

# 添加包含模板的路径，相对于当前目录
templates_path = ['_templates']

# 源文件的后缀名
source_suffix = '.rst'

# 主索引文件
master_doc = 'index'

# 抑制特定的警告信息
suppress_warnings = ['ref.citation', 'ref.footnote']

# 通用的替换项
project = 'SymPy'
copyright = '{} SymPy Development Team'.format(datetime.utcnow().year)

# 替换|version|和|release|的默认值，用于各处的版本信息显示
version = sympy.__version__
release = version

# 用于替换|today|的格式
today_fmt = '%B %d, %Y'

# 不包含在构建中的文档列表
# unused_docs = []

# 如果为True，对于函数等交叉引用文本，将追加'()'括号
# add_function_parentheses = True

# 如果为True，将当前模块名添加到所有描述单元标题的前面
# add_module_names = True

# 如果为True，sectionauthor和moduleauthor指令将在输出中显示
# show_authors = False

# 指定Pygments（语法高亮）使用的样式
sys.path.append(os.path.abspath("./_pygments"))
pygments_style = 'styles.SphinxHighContrastStyle'
pygments_dark_style = 'styles.NativeHighContrastStyle'

# 使用matplotlib绘图指令时，不显示源代码的超链接
plot_html_show_source_link = False
# Options for HTML output
# -----------------------

# 定义静态文件路径，这里包含了自定义的静态文件目录，相对于当前目录
html_static_path = ['_static']

# 如果非空字符串，将在每个页面底部插入“Last updated on:”时间戳，使用给定的 strftime 格式
html_last_updated_fmt = '%b %d, %Y'

# 定义 HTML 主题。之前是经典主题，现在改为了 "furo" 主题
html_theme = "furo"

# 调整侧边栏使其整体可以滚动
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",   # 滚动侧边栏开始的 HTML 文件
        "sidebar/brand.html",          # 侧边栏的品牌信息 HTML 文件
        "sidebar/search.html",         # 侧边栏的搜索功能 HTML 文件
        "sidebar/navigation.html",     # 侧边栏的导航 HTML 文件
        "sidebar/versions.html",       # 侧边栏的版本信息 HTML 文件
        "sidebar/scroll-end.html",     # 滚动侧边栏结束的 HTML 文件
    ],
}

common_theme_variables = {
    # 主题的主要颜色设置，例如 "SymPy green" 的两种颜色
    "color-brand-primary": "#52833A",
    "color-brand-content": "#307748",

    # 左侧边栏的背景颜色设置
    "color-sidebar-background": "#3B5526",
    "color-sidebar-background-border": "var(--color-background-primary)",
    "color-sidebar-link-text": "#FFFFFF",
    "color-sidebar-brand-text": "var(--color-sidebar-link-text--top-level)",
    "color-sidebar-link-text--top-level": "#FFFFFF",
    "color-sidebar-item-background--hover": "var(--color-brand-primary)",
    "color-sidebar-item-expander-background--hover": "var(--color-brand-primary)",

    # 链接的下划线颜色设置（鼠标悬停时）
    "color-link-underline--hover": "var(--color-link)",
    "color-api-keyword": "#000000bd",
    "color-api-name": "var(--color-brand-content)",
    "color-api-pre-name": "var(--color-brand-content)",
    "api-font-size": "var(--font-size--normal)",
    "color-foreground-secondary": "#53555B",

    # 不同类型的警告信息框颜色设置
    "color-admonition-title-background--seealso": "#CCCCCC",
    "color-admonition-title--seealso": "black",
    "color-admonition-title-background--note": "#CCCCCC",
    "color-admonition-title--note": "black",
    "color-admonition-title-background--warning": "var(--color-problematic)",
    "color-admonition-title--warning": "white",
    "admonition-font-size": "var(--font-size--normal)",
    "admonition-title-font-size": "var(--font-size--normal)",

    # 注意：这个设置当前不生效，如果需要改变，必须在 custom.css 中设置 .highlight 背景
    "color-code-background": "hsl(80deg 100% 95%)",

    # 代码区域的字体大小设置
    "code-font-size": "var(--font-size--small)",

    # 等宽字体的字体堆栈设置
    "font-stack--monospace": 'DejaVu Sans Mono,"SFMono-Regular",Menlo,Consolas,Monaco,Liberation Mono,Lucida Console,monospace;'
}

html_theme_options = {
    # 使用定义好的公共主题变量设置亮色主题的 CSS 变量
    "light_css_variables": common_theme_variables,
}
    # The dark variables automatically inherit values from the light variables
    # dark_css_variables contains CSS variable definitions for a dark theme,
    # inheriting some values from common_theme_variables.
    "dark_css_variables": {
        **common_theme_variables,  # Inherit values from common_theme_variables
        "color-brand-primary": "#33CB33",  # Define primary brand color
        "color-brand-content": "#1DBD1D",  # Define content brand color

        "color-api-keyword": "#FFFFFFbd",  # Define color for API keywords
        "color-api-overall": "#FFFFFF90",  # Define overall API color
        "color-api-paren": "#FFFFFF90",  # Define parentheses color in API

        "color-sidebar-item-background--hover": "#52833A",  # Define sidebar item background color on hover
        "color-sidebar-item-expander-background--hover": "#52833A",  # Define sidebar item expander background color on hover
        # This is the color of the text in the right sidebar
        "color-foreground-secondary": "#9DA1AC",  # Define secondary foreground text color

        "color-admonition-title-background--seealso": "#555555",  # Define background color for 'seealso' admonition title
        "color-admonition-title-background--note": "#555555",  # Define background color for 'note' admonition title
        "color-problematic": "#B30000",  # Define problematic color
    },

    # See https://pradyunsg.me/furo/customisation/footer/
    # footer_icons is a list containing details of icons for the footer section
    "footer_icons": [
        {
            "name": "GitHub",  # Name of the icon
            "url": "https://github.com/sympy/sympy",  # URL link associated with the icon
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # SVG icon code
            "class": "",  # CSS class associated with the icon
        },
    ],
# Add a header for PR preview builds. See the Circle CI configuration.
if os.environ.get("CIRCLECI") == "true":
    # 获取 Circle CI 环境变量中的 PR 编号
    PR_NUMBER = os.environ.get('CIRCLE_PR_NUMBER')
    # 获取 Circle CI 环境变量中的提交 SHA1 值
    SHA1 = os.environ.get('CIRCLE_SHA1')
    # 在 HTML 主题选项中添加预览构建的公告信息，包括 PR 链接和提交 SHA1 的链接
    html_theme_options['announcement'] = f"""This is a preview build from
SymPy pull request <a href="https://github.com/sympy/sympy/pull/{PR_NUMBER}">
#{PR_NUMBER}</a>. It was built against <a
href="https://github.com/sympy/sympy/pull/{PR_NUMBER}/commits/{SHA1}">{SHA1[:7]}</a>.
If you aren't looking for a PR preview, go to <a
href="https://docs.sympy.org/">the main SymPy documentation</a>. """

# custom.css contains changes that aren't possible with the above because they
# aren't specified in the Furo theme as CSS variables
# 将 custom.css 文件添加到 HTML 的 CSS 文件列表中
html_css_files = ['custom.css']

# html_js_files = []

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Content template for the index page.
#html_index = ''

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_use_modindex = True

# 在 HTML 文档中生成模块索引
html_domain_indices = ['py-modindex']

# If true, the reST sources are included in the HTML build as _sources/<name>.
# html_copy_source = True

# Output file base name for HTML help builder.
# 设置生成的 HTML 帮助文件的基本名称
htmlhelp_basename = 'SymPydoc'

language = 'en'

# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual], toctree_only).
# toctree_only is set to True so that the start file document itself is not included in the
# output, only the documents referenced by it via TOC trees.  The extra stuff in the master
# document is intended to show up in the HTML, but doesn't really belong in the LaTeX output.
# 定义生成 LaTeX 文件的配置，包括文档结构、文档类别等
latex_documents = [('index', 'sympy-%s.tex' % release, 'SymPy Documentation',
                    'SymPy Development Team', 'manual', True)]

# Additional stuff for the LaTeX preamble.
# Tweaked to work with XeTeX.
# LaTeX 渲染引擎特定的配置，设置字体、语言等
latex_elements = {
    'babel':     '',
    'fontenc': r'''
% Define version of \LaTeX that is usable in math mode
\let\OldLaTeX\LaTeX
\renewcommand{\LaTeX}{\text{\OldLaTeX}}

\usepackage{bm}
\usepackage{amssymb}
\usepackage{fontspec}
\usepackage[english]{babel}
\defaultfontfeatures{Mapping=tex-text}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
''',
    'fontpkg':   '',
    'inputenc':  '',
    'utf8extra': '',
    'preamble':  r'''
'''
}

# SymPy logo on title page
# 设置生成的 HTML 页面的 Logo 和 Favicon
html_logo = '_static/sympylogo.png'
latex_logo = '_static/sympylogo_big.png'
html_favicon = '../_build/logo/sympy-notailtext-favicon.ico'
# 设置 LaTeX 输出中追加附录文档的选项，默认为空列表
latex_appendices = []

# 在内部引用旁边显示页码
latex_show_pagerefs = True

# 禁止生成模块索引，否则会重复生成
latex_use_modindex = False

# 默认角色设为 'math'，用于数学表达式
default_role = 'math'

# PNGMath 的参数设置，应用于生成 PNG 数学表达式
pngmath_divpng_args = ['-gamma 1.5', '-D 110']

# 注意：MathJax 扩展会忽略此设置
# 扩展 LaTeX 的前导声明，包括一些数学相关的 LaTeX 宏包和设置
pngmath_latex_preamble = '\\usepackage{amsmath}\n' \
    '\\usepackage{bm}\n' \
    '\\usepackage{amsfonts}\n' \
    '\\usepackage{amssymb}\n' \
    '\\setlength{\\parindent}{0pt}\n'

# Texinfo 文档的配置信息，指定文档的标题、作者等内容
texinfo_documents = [
    (master_doc, 'sympy', 'SymPy Documentation', 'SymPy Development Team',
   'SymPy', 'Computer algebra system (CAS) in Python', 'Programming', 1),
]

# 设置 Graphviz 的输出格式为 SVG
graphviz_output_format = 'svg'

# 允许链接到其他包的文档
intersphinx_mapping = {
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'mpmath': ('https://mpmath.org/doc/current/', None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# 禁用 intersphinx 的引用类型，防止意外链接到 matplotlib 等包的文档
intersphinx_disabled_reftypes = ['*']

# 从外部文件中获取提交哈希值
commit_hash_filepath = '../commit_hash.txt'
commit_hash = None

# 如果存在提交哈希文件，则读取其中的哈希值
if os.path.isfile(commit_hash_filepath):
    with open(commit_hash_filepath) as f:
        commit_hash = f.readline()

# 如果未能从文件中获取哈希值，则尝试通过 git 命令获取当前的提交哈希值
if not commit_hash:
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        commit_hash = commit_hash.decode('ascii')
        commit_hash = commit_hash.rstrip()
    except:
        import warnings
        warnings.warn(
            "Failed to get the git commit hash as the command " \
            "'git rev-parse HEAD' is not working. The commit hash will be " \
            "assumed as the SymPy master, but the lines may be misleading " \
            "or nonexistent as it is not the correct branch the doc is " \
            "built with. Check your installation of 'git' if you want to " \
            "resolve this warning.")
        commit_hash = 'master'

# GitHub 仓库和分支路径的格式化字符串，用于生成链接
fork = 'sympy'
blobpath = \
    "https://github.com/{}/sympy/blob/{}/sympy/".format(fork, commit_hash)

def linkcode_resolve(domain, info):
    """确定 Python 对象对应的 URL."""
    if domain != 'py':
        return

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return
    # 捕获任何异常，将行号设置为 None
    except Exception:
        lineno = None

    # 如果行号存在（非 None），则生成以行号开始和结束的字符串表示
    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        # 如果行号不存在，设置行号字符串为空字符串
        linespec = ""

    # 获取相对于 sympy 模块所在目录的相对路径
    fn = os.path.relpath(fn, start=os.path.dirname(sympy.__file__))
    # 返回构建的文件路径，包括 blobpath、相对路径和行号信息
    return blobpath + fn + linespec
```