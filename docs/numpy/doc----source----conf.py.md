# `.\numpy\doc\source\conf.py`

```
# 导入必要的标准库和第三方库
import os            # 操作系统接口
import re            # 正则表达式
import sys           # 系统特定的参数和函数
import importlib     # 实现动态加载模块和包
from docutils import nodes                   # 文档处理工具
from docutils.parsers.rst import Directive  # reStructuredText 的指令

# Minimum version, enforced by sphinx
# 最低版本要求，由 Sphinx 强制执行
needs_sphinx = '4.3'

# This is a nasty hack to use platform-agnostic names for types in the
# documentation.
# 这是一个不好的 hack，用于在文档中使用与平台无关的类型名称。

# must be kept alive to hold the patched names
# 必须保持存活状态以保存打补丁的名称
_name_cache = {}

def replace_scalar_type_names():
    """ Rename numpy types to use the canonical names to make sphinx behave """
    """ 重命名 numpy 类型以使用规范名称，使 sphinx 正常工作 """
    import ctypes

    Py_ssize_t = ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_int32

    class PyObject(ctypes.Structure):
        pass

    class PyTypeObject(ctypes.Structure):
        pass

    PyObject._fields_ = [
        ('ob_refcnt', Py_ssize_t),
        ('ob_type', ctypes.POINTER(PyTypeObject)),
    ]

    PyTypeObject._fields_ = [
        # varhead
        ('ob_base', PyObject),
        ('ob_size', Py_ssize_t),
        # declaration
        ('tp_name', ctypes.c_char_p),
    ]

    # prevent numpy attaching docstrings to the scalar types
    # 防止 numpy 将文档字符串附加到标量类型上
    assert 'numpy._core._add_newdocs_scalars' not in sys.modules
    sys.modules['numpy._core._add_newdocs_scalars'] = object()

    import numpy

    # change the __name__ of the scalar types
    # 更改标量类型的 __name__
    for name in [
        'byte', 'short', 'intc', 'int_', 'longlong',
        'ubyte', 'ushort', 'uintc', 'uint', 'ulonglong',
        'half', 'single', 'double', 'longdouble',
        'half', 'csingle', 'cdouble', 'clongdouble',
    ]:
        typ = getattr(numpy, name)
        c_typ = PyTypeObject.from_address(id(typ))
        c_typ.tp_name = _name_cache[typ] = b"numpy." + name.encode('utf8')

    # now generate the docstrings as usual
    # 现在像往常一样生成文档字符串
    del sys.modules['numpy._core._add_newdocs_scalars']
    import numpy._core._add_newdocs_scalars

replace_scalar_type_names()


# As of NumPy 1.25, a deprecation of `str`/`bytes` attributes happens.
# For some reasons, the doc build accesses these, so ignore them.
# 从 NumPy 1.25 开始，将会弃用 `str`/`bytes` 属性。
# 由于某些原因，文档生成访问了这些属性，因此忽略它们。
import warnings
warnings.filterwarnings("ignore", "In the future.*NumPy scalar", FutureWarning)


# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
# 在这里添加任何 Sphinx 扩展模块名称，作为字符串。它们可以是随 Sphinx 一起提供的扩展（如 'sphinx.ext.*'）或您自己的扩展。

sys.path.insert(0, os.path.abspath('../sphinxext'))

extensions = [
    'sphinx.ext.autodoc',                       # 自动生成 API 文档
    'numpydoc',                                 # 支持 NumPy 风格的文档
    'sphinx.ext.intersphinx',                   # 支持链接外部文档
    'sphinx.ext.coverage',                      # 测试覆盖率相关
    'sphinx.ext.doctest',                       # 运行文档中的示例，并验证结果的正确性
    'sphinx.ext.autosummary',                   # 自动生成摘要
    'sphinx.ext.graphviz',                      # 生成图形
    'sphinx.ext.ifconfig',                      # 条件设置配置
    'matplotlib.sphinxext.plot_directive',      # 绘图指令
    'IPython.sphinxext.ipython_console_highlighting',  # IPython 控制台高亮
    'IPython.sphinxext.ipython_directive',      # IPython 指令
    'sphinx.ext.mathjax',                       # 数学公式支持
    'sphinx_design',                            # Sphinx 主题设计扩展
]

skippable_extensions = [
    ('breathe', 'skip generating C/C++ API from comment blocks.'),
]
for ext, warn in skippable_extensions:
    # 检查指定的 Sphinx 扩展是否存在于当前环境中
    ext_exist = importlib.util.find_spec(ext) is not None
    
    # 如果找到了指定的 Sphinx 扩展，则将其添加到列表中
    if ext_exist:
        extensions.append(ext)
    # 如果未找到指定的 Sphinx 扩展，则打印警告信息
    else:
        print(f"Unable to find Sphinx extension '{ext}', {warn}.")
# Add any paths that contain templates here, relative to this directory.
# 添加包含模板的路径列表，相对于当前目录。
templates_path = ['_templates']

# The suffix of source filenames.
# 源文件名的后缀。
source_suffix = '.rst'

# General substitutions.
# 一般的替换项。
project = 'NumPy'
copyright = '2008-2024, NumPy Developers'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
# 默认的 |version| 和 |release| 替换值，在构建文档中的多处使用。
import numpy
# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
# 短版本 X.Y（包括 .devXXXX、rcX、b1 后缀，如果存在的话）
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', numpy.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)
# The full version, including alpha/beta/rc tags.
# 完整版本，包括 alpha/beta/rc 标签。
release = numpy.__version__
print("%s %s" % (version, release))

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
# 有两种方式替换 |today|：一种是将 today 设置为某个非假值，然后它会被使用；
# 另一种是使用 today_fmt 作为 strftime 调用的格式。
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
# 不应包含在构建中的文档列表。
#unused_docs = []

# The reST default role (used for this markup: `text`) to use for all documents.
# 所有文档使用的 reST 默认角色（用于此标记：`text`）。
default_role = "autolink"

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
# 不应搜索源文件的目录列表，相对于源目录。
exclude_dirs = []

exclude_patterns = []
if sys.version_info[:2] >= (3, 12):
    exclude_patterns += ["reference/distutils.rst"]

# If true, '()' will be appended to :func: etc. cross-reference text.
# 如果为 True，则在 :func: 等交叉引用文本后附加 '()'。
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# 如果为 True，则当前模块名将前置于所有描述单元标题之前（例如 .. function::）。
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# 如果为 True，则输出中将显示 sectionauthor 和 moduleauthor 指令。它们默认情况下被忽略。
#show_authors = False

class LegacyDirective(Directive):
    """
    Adapted from docutils/parsers/rst/directives/admonitions.py

    Uses a default text if the directive does not have contents. If it does,
    the default text is concatenated to the contents.

    See also the same implementation in SciPy's conf.py.
    """
    # 来自 docutils/parsers/rst/directives/admonitions.py 的适应版本

    # 如果指令没有内容，则使用默认文本。如果有内容，则将默认文本连接到内容中。
    has_content = True
    node_class = nodes.admonition
    optional_arguments = 1
    def run(self):
        try:
            # 尝试获取第一个参数作为对象名称
            obj = self.arguments[0]
        except IndexError:
            # 没有参数传入时，默认使用文本 "submodule"
            obj = "submodule"
        
        # 创建包含说明文本的字符串
        text = (f"This {obj} is considered legacy and will no longer receive "
                "updates. This could also mean it will be removed in future "
                "NumPy versions.")
        
        try:
            # 尝试将文本添加到现有内容列表的第一个位置
            self.content[0] = text + " " + self.content[0]
        except IndexError:
            # 如果内容列表为空，则创建新的内容条目
            source, lineno = self.state_machine.get_source_and_line(
                self.lineno
            )
            self.content.append(
                text,
                source=source,
                offset=lineno
            )
        
        # 将内容列表转换为单个文本字符串
        text = '\n'.join(self.content)
        
        # 创建警告提示节点，稍后由 `nested_parse` 填充内容
        admonition_node = self.node_class(rawsource=text)
        
        # 设置自定义标题
        title_text = "Legacy"
        textnodes, _ = self.state.inline_text(title_text, self.lineno)
        
        # 创建标题节点
        title = nodes.title(title_text, '', *textnodes)
        admonition_node += title
        
        # 设置警告提示节点的 CSS 类
        admonition_node['classes'] = ['admonition-legacy']
        
        # 解析指令内容并填充到警告提示节点中
        self.state.nested_parse(self.content, self.content_offset,
                                admonition_node)
        
        # 返回最终的警告提示节点列表
        return [admonition_node]
# 为 Sphinx 应用程序设置函数，用于配置各种选项和插件
def setup(app):
    # 添加一个配置值，用于 `ifconfig` 指令
    app.add_config_value('python_version_major', str(sys.version_info.major), 'env')
    # 添加一个词法分析器，将 'NumPyC' 识别为 NumPyLexer 类型
    app.add_lexer('NumPyC', NumPyLexer)
    # 添加一个自定义指令 'legacy'，使用 LegacyDirective 处理
    app.add_directive("legacy", LegacyDirective)


# 将 'numpy.char' 的模块别名设定为 numpy.char
# 虽然对象类型是 `module`，但名称是模块的别名，用于使 Sphinx 可以识别这些别名作为真实模块
sys.modules['numpy.char'] = numpy.char

# -----------------------------------------------------------------------------
# HTML 输出配置
# -----------------------------------------------------------------------------

# 设置 HTML 主题
html_theme = 'pydata_sphinx_theme'

# 设置网站图标路径
html_favicon = '_static/favicon/favicon.ico'

# 设置版本切换器，versions.json 存储在文档仓库中
if os.environ.get('CIRCLE_JOB', False) and \
        os.environ.get('CIRCLE_BRANCH', '') != 'main':
    # 对于 PR，将版本名称设置为其引用
    switcher_version = os.environ['CIRCLE_BRANCH']
elif ".dev" in version:
    switcher_version = "devdocs"
else:
    switcher_version = f"{version}"

# 设置 HTML 主题选项
html_theme_options = {
    "logo": {
        "image_light": "_static/numpylogo.svg",
        "image_dark": "_static/numpylogo_dark.svg",
    },
    "github_url": "https://github.com/numpy/numpy",
    "collapse_navigation": True,
    "external_links": [
        {"name": "Learn", "url": "https://numpy.org/numpy-tutorials/"},
        {"name": "NEPs", "url": "https://numpy.org/neps"},
    ],
    "header_links_before_dropdown": 6,
    # 添加亮色/暗色模式和文档版本切换器
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "version-switcher",
        "navbar-icon-links"
    ],
    "navbar_persistent": [],
    "switcher": {
        "version_match": switcher_version,
        "json_url": "https://numpy.org/doc/_static/versions.json",
    },
    "show_version_warning_banner": True,
}

# 设置 HTML 页面标题
html_title = "%s v%s Manual" % (project, version)
# 设置静态文件路径
html_static_path = ['_static']
# 设置最后更新时间格式
html_last_updated_fmt = '%b %d, %Y'
# 设置 HTML 使用的 CSS 文件
html_css_files = ["numpy.css"]
# 设置 HTML 上下文
html_context = {"default_mode": "light"}
# 禁用模块索引
html_use_modindex = True
# 不复制源文件
html_copy_source = False
# 禁用文档索引
html_domain_indices = False
# 设置 HTML 文件后缀
html_file_suffix = '.html'

# 设置 HTML 帮助文档的基本名称
htmlhelp_basename = 'numpy'

# 如果扩展列表中包含 'sphinx.ext.pngmath'
if 'sphinx.ext.pngmath' in extensions:
    # 启用 PNG 数学公式预览
    pngmath_use_preview = True
    # 设置 dvipng 参数
    pngmath_dvipng_args = ['-gamma', '1.5', '-D', '96', '-bg', 'Transparent']

# -----------------------------------------------------------------------------
# LaTeX 输出配置
# -----------------------------------------------------------------------------

# 设置纸张大小（'letter' 或 'a4'）
#latex_paper_size = 'letter'

# 设置字体大小（'10pt', '11pt' 或 '12pt'）
#latex_font_size = '10pt'
# 设置 LaTeX 引擎为 XeLaTeX，以支持更好的 Unicode 字符处理能力
latex_engine = 'xelatex'

# 将文档树分组为 LaTeX 文件的列表。每个元组包含：
# （源文件的起始点，目标文件名，标题，作者，文档类别[如何指南/手册]）
_stdauthor = 'Written by the NumPy community'
latex_documents = [
  ('reference/index', 'numpy-ref.tex', 'NumPy Reference',
   _stdauthor, 'manual'),
  ('user/index', 'numpy-user.tex', 'NumPy User Guide',
   _stdauthor, 'manual'),
]

# 标题页顶部放置的图像文件名（相对于当前目录）
#latex_logo = None

# 对于“手册”文档，如果为 True，则顶级标题为部分而不是章节
#latex_use_parts = False

latex_elements = {
}

# LaTeX 导言区的附加内容
latex_elements['preamble'] = r'''
\newfontfamily\FontForChinese{FandolSong-Regular}[Extension=.otf]
\catcode`琴\active\protected\def琴{{\FontForChinese\string琴}}
\catcode`春\active\protected\def春{{\FontForChinese\string春}}
\catcode`鈴\active\protected\def鈴{{\FontForChinese\string鈴}}
\catcode`猫\active\protected\def猫{{\FontForChinese\string猫}}
\catcode`傅\active\protected\def傅{{\FontForChinese\string傅}}
\catcode`立\active\protected\def立{{\FontForChinese\string立}}
\catcode`业\active\protected\def业{{\FontForChinese\string业}}
\catcode`（\active\protected\def（{{\FontForChinese\string（}}
\catcode`）\active\protected\def）{{\FontForChinese\string）}}

% 在参数部分的标题后面放置一个换行。这在 Sphinx 5.0.0+ 中是默认行为，因此不再需要旧的 hack。
% 不幸的是，sphinx.sty 5.0.0 没有更新其版本日期，因此我们检查 sphinxpackagefootnote.sty（自 Sphinx 4.0.0 起存在）。
\makeatletter
\@ifpackagelater{sphinxpackagefootnote}{2022/02/12}
    {}% Sphinx >= 5.0.0，无需操作
    {%
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}
% 修复 expdlist 旧 LaTeX 包的问题：
% 1) 移除额外的空格
\usepackage{etoolbox}
\patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
% 2) 修复 expdlist 在长标签后换行的 bug
\def\breaklabel{%
    \def\@breaklabel{%
        \leavevmode\par
        % 现在是一个 hack，因为 Sphinx 在术语节点之后插入 \leavevmode
        \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
    }%
}
    }% Sphinx < 5.0.0（假设 >= 4.0.0）
\makeatother

% 使示例等部分的标题更小更紧凑
\makeatletter
\titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
            {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
\titlespacing*{\paragraph}{0pt}{1ex}{0pt}
\makeatother

% 修复页眉页脚
\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}
'''

# 将文档附加为所有手册的附录
#latex_appendices = []

# 如果为 False，则不生成模块索引
latex_use_modindex = False
# -----------------------------------------------------------------------------
# Texinfo output
# -----------------------------------------------------------------------------

# 定义生成 Texinfo 格式文档的配置参数
texinfo_documents = [
  ("index", 'numpy', 'NumPy Documentation', _stdauthor, 'NumPy',
   "NumPy: array processing for numbers, strings, records, and objects.",
   'Programming',
   1),
]

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------

# 定义用于 intersphinx 的映射配置，指定外部文档的链接
intersphinx_mapping = {
    'neps': ('https://numpy.org/neps', None),
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'imageio': ('https://imageio.readthedocs.io/en/stable', None),
    'skimage': ('https://scikit-image.org/docs/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'scipy-lecture-notes': ('https://scipy-lectures.org', None),
    'pytest': ('https://docs.pytest.org/en/stable', None),
    'numpy-tutorials': ('https://numpy.org/numpy-tutorials', None),
    'numpydoc': ('https://numpydoc.readthedocs.io/en/latest', None),
    'dlpack': ('https://dmlc.github.io/dlpack/latest', None)
}

# -----------------------------------------------------------------------------
# NumPy extensions
# -----------------------------------------------------------------------------

# 定义 NumPy 扩展功能的配置选项

# 指定进行虚拟导入的 XML 文件
phantom_import_file = 'dump.xml'

# 设置 numpydoc 是否生成示例部分的图表
numpydoc_use_plots = True

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

# 自动生成摘要页面的配置开关
autosummary_generate = True

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------

# 覆盖率检查工具的配置选项

# 忽略的模块列表
coverage_ignore_modules = r"""
    """.split()

# 忽略的函数列表的正则表达式
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()

# 忽略的类列表
coverage_ignore_classes = r"""
    """.split()

# C 语言覆盖检查路径
coverage_c_path = []
# C 语言覆盖检查正则表达式
coverage_c_regexes = {}
# 忽略的 C 语言项
coverage_ignore_c_items = {}

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

# 绘图配置选项

# 绘图前的预处理代码
plot_pre_code = """
import numpy as np
np.random.seed(0)
"""

# 是否包含绘图源代码
plot_include_source = True

# 绘图格式及其分辨率设置
plot_formats = [('png', 100), 'pdf']

# 绘图参数配置
import math
phi = (math.sqrt(5) + 1)/2

plot_rcparams = {
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3*phi, 3),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
}
    # 设置图形的顶部子图位置为0.85
    'figure.subplot.top': 0.85,
    # 设置子图之间的水平间距为0.4
    'figure.subplot.wspace': 0.4,
    # 禁用使用LaTeX渲染文本
    'text.usetex': False,
}

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------

import inspect  # 导入 inspect 模块，用于检查和分析 Python 对象
from os.path import relpath, dirname  # 从 os.path 模块中导入 relpath 和 dirname 函数

# 循环遍历包含字符串列表，尝试导入指定的模块，如果成功则将模块名添加到 extensions 列表中
for name in ['sphinx.ext.linkcode', 'numpydoc.linkcode']:
    try:
        __import__(name)
        extensions.append(name)  # 将成功导入的模块名添加到 extensions 列表
        break  # 如果成功导入模块，则结束循环
    except ImportError:
        pass
else:
    print("NOTE: linkcode extension not found -- no links to source generated")  # 如果未找到任何 linkcode 扩展模块，则打印提示信息

# 定义一个函数，根据输入的对象返回相应的 C 源文件路径
def _get_c_source_file(obj):
    if issubclass(obj, numpy.generic):
        return r"_core/src/multiarray/scalartypes.c.src"  # 如果输入对象是 numpy.generic 的子类，返回对应的 C 源文件路径
    elif obj is numpy.ndarray:
        return r"_core/src/multiarray/arrayobject.c"  # 如果输入对象是 numpy.ndarray 类型，则返回对应的 C 源文件路径
    else:
        # 如果输入对象不符合以上条件，返回 None，并提醒需要找到更好的生成方式
        return None


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None  # 如果域不是 'py'，返回 None

    modname = info['module']  # 获取模块名
    fullname = info['fullname']  # 获取完整的对象名（包括模块名和对象名）

    submod = sys.modules.get(modname)  # 获取模块对象
    if submod is None:
        return None  # 如果未找到模块对象，返回 None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)  # 逐级获取对象的属性
        except Exception:
            return None  # 如果获取属性时发生异常，返回 None

    # 尝试去掉装饰器，这可能是 inspect.getsourcefile 的一个问题
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    fn = None  # 初始化文件名变量为 None
    lineno = None  # 初始化行号变量为 None

    # 尝试链接 C 扩展类型
    if isinstance(obj, type) and obj.__module__ == 'numpy':
        fn = _get_c_source_file(obj)  # 获取对应的 C 源文件路径

    if fn is None:
        try:
            fn = inspect.getsourcefile(obj)  # 尝试获取对象的源文件路径
        except Exception:
            fn = None
        if not fn:
            return None  # 如果未获取到源文件路径，返回 None

        # 忽略重新导出的对象，因为它们的源文件不在 numpy 仓库内
        module = inspect.getmodule(obj)
        if module is not None and not module.__name__.startswith("numpy"):
            return None  # 如果模块不是以 'numpy' 开头，返回 None

        try:
            source, lineno = inspect.getsourcelines(obj)  # 获取对象的源码和起始行号
        except Exception:
            lineno = None

        fn = relpath(fn, start=dirname(numpy.__file__))  # 获取相对于 numpy 包的路径

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)  # 构造行号范围字符串
    else:
        linespec = ""

    if 'dev' in numpy.__version__:
        return "https://github.com/numpy/numpy/blob/main/numpy/%s%s" % (
           fn, linespec)  # 返回 GitHub 上对应源码的 URL
    else:
        return "https://github.com/numpy/numpy/blob/v%s/numpy/%s%s" % (
           numpy.__version__, fn, linespec)  # 返回 GitHub 上对应版本源码的 URL

from pygments.lexers import CLexer  # 从 pygments 库导入 CLexer 类
from pygments.lexer import inherit, bygroups  # 从 pygments 库导入 inherit 和 bygroups 函数
from pygments.token import Comment  # 从 pygments 库导入 Comment 类

class NumPyLexer(CLexer):
    name = 'NUMPYLEXER'  # 定义自定义的词法分析器名称为 'NUMPYLEXER'

    tokens = {
        'statements': [
            (r'@[a-zA-Z_]*@', Comment.Preproc, 'macro'),  # 定义预处理指令的词法规则
            inherit,  # 继承默认的词法规则
        ],
    }
# -----------------------------------------------------------------------------
# Breathe & Doxygen
# -----------------------------------------------------------------------------
# 定义一个字典，用于指定 Breathe 文档生成工具的项目配置，
# 将 "numpy" 项目的 XML 文档路径设置为相对路径 "../build/doxygen/xml"
breathe_projects = dict(numpy=os.path.join("..", "build", "doxygen", "xml"))

# 设置默认的 Breathe 项目为 "numpy"
breathe_default_project = "numpy"

# 设置默认的成员过滤器，包括 "members", "undoc-members", "protected-members"
breathe_default_members = ("members", "undoc-members", "protected-members")

# See https://github.com/breathe-doc/breathe/issues/696
# 忽略特定的 Doxygen 错误或警告，这里列出了需要忽略的标识符列表
nitpick_ignore = [
    ('c:identifier', 'FILE'),
    ('c:identifier', 'size_t'),
    ('c:identifier', 'PyHeapTypeObject'),
]
```