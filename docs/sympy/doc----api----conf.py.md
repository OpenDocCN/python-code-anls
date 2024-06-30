# `D:\src\scipysrc\sympy\doc\api\conf.py`

```
# 导入系统模块 sys 和 SymPy 模块
import sys
import sympy

# 如果你的扩展模块在另一个目录中，可以在这里添加该目录
#sys.path.append('some/directory')

# Sphinx 的常规配置
# ---------------------

# 添加任何 Sphinx 扩展模块的名称到这里，作为字符串。它们可以是随 Sphinx 一起提供的扩展
# （如 'sphinx.addons.*'）或你自定义的扩展。
extensions = ['sphinx.ext.autodoc']

# 添加包含模板的路径，相对于当前目录。
templates_path = ['.templates']

# 源文件名的后缀。
source_suffix = '.rst'

# 主 toctree 文档。
master_doc = 'index'

# 常规替换。
project = 'SymPy'
copyright = '2015, SymPy Development Team'

# |version| 和 |release| 的默认替换，也用于生成文档中的其他各处。
#
# 短的 X.Y 版本。
version = sympy.__version__
# 完整版本，包括 alpha/beta/rc 标签。
release = version

# 有两种选择替换 |today|：要么设置 today 为某个非假值，然后它被使用：
#today = ''
# 否则，today_fmt 用作 strftime 调用的格式。
today_fmt = '%B %d, %Y'

# 列出不应包含在构建中的文档。
#unused_docs = []

# 如果为真，则在 :func: 等交叉引用文本中添加 '()'。
#add_function_parentheses = True

# 如果为真，则当前模块名称将前置到所有描述单元标题中（例如 .. function::）。
#add_module_names = True

# 如果为真，则 sectionauthor 和 moduleauthor 指令将显示在输出中。它们默认被忽略。
#show_authors = False

# 要使用的 Pygments（语法高亮）样式的名称。
pygments_style = 'sphinx'


# HTML 输出选项
# -----------------------

# 用于 HTML 和 HTML 帮助页面的样式表。该名称的文件必须存在于 Sphinx 的 static/ 路径中，
# 或者在 html_static_path 中给出的自定义路径中之一。
html_style = 'default.css'

# 添加包含自定义静态文件（如样式表）的任何路径，
# 相对于此目录。它们会在内置静态文件之后复制，因此名为 "default.css" 的文件会覆盖内置的 "default.css"。
html_static_path = ['.static']

# 如果不是空字符串，则在每个页面底部插入一个 'Last updated on:' 时间戳，
# 使用给定的 strftime 格式。
html_last_updated_fmt = '%b %d, %Y'

# 如果为真，则使用 SmartyPants 将引号和破折号转换为排版正确的实体。
#html_use_smartypants = True

# 首页的内容模板。
# 设置生成的 HTML 帮助文件的基本文件名为 'SymPydoc'
htmlhelp_basename = 'SymPydoc'

# LaTeX 输出选项部分开始
# ------------------------

# 定义纸张大小为 'letter' 或 'a4'
#latex_paper_size = 'letter'

# 定义字体大小为 '10pt', '11pt' 或 '12pt'
#latex_font_size = '10pt'

# 将文档树分组成 LaTeX 文件。每个元组包含源文件名、目标文件名、标题、作者和文档类别 [howto/manual]。
latex_documents = [('index', 'sympy.tex', 'SymPy Documentation',
                    'SymPy Development Team', 'manual')]

# LaTeX 前文部分的额外内容
#latex_preamble = ''

# 将指定文档追加为所有手册的附录
#latex_appendices = []

# 如果为 False，则不生成模块索引
#latex_use_modindex = True
```