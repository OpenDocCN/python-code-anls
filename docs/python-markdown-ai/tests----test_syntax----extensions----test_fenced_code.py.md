# `markdown\tests\test_syntax\extensions\test_fenced_code.py`

```py

"""
Python Markdown

A Python implementation of John Gruber's Markdown.

Documentation: https://python-markdown.github.io/
GitHub: https://github.com/Python-Markdown/markdown/
PyPI: https://pypi.org/project/Markdown/

Started by Manfred Stienstra (http://www.dwerg.net/).
Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
Currently maintained by Waylan Limberg (https://github.com/waylan),
Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

Copyright 2007-2019 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""

# 导入需要的模块
from markdown.test_tools import TestCase
import markdown
import markdown.extensions.codehilite
import os

# 尝试导入 pygments 模块，如果导入失败则设置 has_pygments 为 False
try:
    import pygments  # noqa
    import pygments.formatters  # noqa
    has_pygments = True
except ImportError:
    has_pygments = False

# 测试所需的 Pygments 版本由 `pygments` tox 环境中指定和安装的版本确定。
# 在任何环境中，如果 `PYGMENTS_VERSION` 环境变量未定义或与安装的 Pygments 版本不匹配，
# 则所有依赖于 Pygments 的测试将被跳过。
required_pygments_version = os.environ.get('PYGMENTS_VERSION', '')

```