# `markdown\markdown\htmlparser.py`

```

# Python Markdown

# A Python implementation of John Gruber's Markdown.

# Documentation: https://python-markdown.github.io/
# GitHub: https://github.com/Python-Markdown/markdown/
# PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
This module imports a copy of [`html.parser.HTMLParser`][] and modifies it heavily through monkey-patches.
A copy is imported rather than the module being directly imported as this ensures that the user can import
and  use the unmodified library for their own needs.
"""

from __future__ import annotations

import re  # 导入正则表达式模块
import importlib.util  # 导入模块导入工具
import sys  # 导入系统模块
from typing import TYPE_CHECKING, Sequence  # 导入类型提示相关模块

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown  # 如果是类型检查，导入Markdown模块


# Import a copy of the html.parser lib as `htmlparser` so we can monkeypatch it.
# Users can still do `from html import parser` and get the default behavior.
spec = importlib.util.find_spec('html.parser')  # 查找html.parser模块的规范
htmlparser = importlib.util.module_from_spec(spec)  # 从规范中创建模块
spec.loader.exec_module(htmlparser)  # 执行模块
sys.modules['htmlparser'] = htmlparser  # 将模块添加到sys.modules中

# Monkeypatch `HTMLParser` to only accept `?>` to close Processing Instructions.
htmlparser.piclose = re.compile(r'\?>')  # 用正则表达式修改HTMLParser，只接受`?>`来关闭处理指令
# Monkeypatch `HTMLParser` to only recognize entity references with a closing semicolon.
htmlparser.entityref = re.compile(r'&([a-zA-Z][-.a-zA-Z0-9]*);')  # 用正则表达式修改HTMLParser，只识别带有分号的实体引用
# Monkeypatch `HTMLParser` to no longer support partial entities. We are always feeding a complete block,
# so the 'incomplete' functionality is unnecessary. As the `entityref` regex is run right before incomplete,
# and the two regex are the same, then incomplete will simply never match and we avoid the logic within.
htmlparser.incomplete = htmlparser.entityref  # 用正则表达式修改HTMLParser，不再支持部分实体
# Monkeypatch `HTMLParser` to not accept a backtick in a tag name, attribute name, or bare value.
htmlparser.locatestarttagend_tolerant = re.compile(r"""
  <[a-zA-Z][^`\t\n\r\f />\x00]*       # tag name <= added backtick here
  (?:[\s/]*                           # optional whitespace before attribute name
    (?:(?<=['"\s/])[^`\s/>][^\s/=>]*  # attribute name <= added backtick here
      (?:\s*=+\s*                     # value indicator
        (?:'[^']*'                    # LITA-enclosed value
          |"[^"]*"                    # LIT-enclosed value
          |(?!['"])[^`>\s]*           # bare value <= added backtick here
         )
         (?:\s*,)*                    # possibly followed by a comma
       )?(?:\s|/(?!>))*
     )*
   )?
  \s*                                 # trailing whitespace
""", re.VERBOSE)  # 用正则表达式修改HTMLParser，不再接受反引号

# Match a blank line at the start of a block of text (two newlines).
# The newlines may be preceded by additional whitespace.
blank_line_re = re.compile(r'^([ ]*\n){2}')  # 匹配文本块开头的空行（两个换行符）

```