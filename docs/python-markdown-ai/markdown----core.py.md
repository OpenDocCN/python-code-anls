# `markdown\markdown\core.py`

```py

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

# 导入未来的注释语法
from __future__ import annotations

# 导入所需的模块
import codecs
import sys
import logging
import importlib
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, ClassVar, Mapping, Sequence
from . import util
from .preprocessors import build_preprocessors
from .blockprocessors import build_block_parser
from .treeprocessors import build_treeprocessors
from .inlinepatterns import build_inlinepatterns
from .postprocessors import build_postprocessors
from .extensions import Extension
from .serializers import to_html_string, to_xhtml_string
from .util import BLOCK_LEVEL_ELEMENTS

# 如果是类型检查，则导入 Element 类
if TYPE_CHECKING:  # pragma: no cover
    from xml.etree.ElementTree import Element

# 定义导出的函数
__all__ = ['Markdown', 'markdown', 'markdownFromFile']


"""
EXPORTED FUNCTIONS
=============================================================================

Those are the two functions we really mean to export: `markdown()` and
`markdownFromFile()`.
"""

# 定义 markdown 函数，将 markdown 字符串转换为 HTML 并返回 Unicode 字符串
def markdown(text: str, **kwargs: Any) -> str:
    """
    Convert a markdown string to HTML and return HTML as a Unicode string.

    This is a shortcut function for [`Markdown`][markdown.Markdown] class to cover the most
    basic use case.  It initializes an instance of [`Markdown`][markdown.Markdown], loads the
    necessary extensions and runs the parser on the given text.

    Arguments:
        text: Markdown formatted text as Unicode or ASCII string.

    Keyword arguments:
        **kwargs: Any arguments accepted by the Markdown class.

    Returns:
        A string in the specified output format.

    """
    md = Markdown(**kwargs)
    return md.convert(text)

# 定义 markdownFromFile 函数，从文件中读取 Markdown 文本并将输出写入文件或流
def markdownFromFile(**kwargs: Any):
    """
    Read Markdown text from a file and write output to a file or a stream.

    This is a shortcut function which initializes an instance of [`Markdown`][markdown.Markdown],
    and calls the [`convertFile`][markdown.Markdown.convertFile] method rather than
    [`convert`][markdown.Markdown.convert].

    Keyword arguments:
        input (str | BinaryIO): A file name or readable object.
        output (str | BinaryIO): A file name or writable object.
        encoding (str): Encoding of input and output.
        **kwargs: Any arguments accepted by the `Markdown` class.

    """
    md = Markdown(**kwargs)
    md.convertFile(kwargs.get('input', None),
                   kwargs.get('output', None),
                   kwargs.get('encoding', None))

```