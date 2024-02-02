# `markdown\markdown\extensions\attr_list.py`

```py

# Attribute List Extension for Python-Markdown
# ============================================

# Adds attribute list syntax. Inspired by
# [Maruku](http://maruku.rubyforge.org/proposal.html#attribute_lists)'s
# feature of the same name.

# See https://Python-Markdown.github.io/extensions/attr_list
# for documentation.

# Original code Copyright 2011 [Waylan Limberg](http://achinghead.com/).

# All changes Copyright 2011-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
 Adds attribute list syntax. Inspired by
[Maruku](http://maruku.rubyforge.org/proposal.html#attribute_lists)'s
feature of the same name.

See the [documentation](https://Python-Markdown.github.io/extensions/attr_list)
for details.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from . import Extension
from ..treeprocessors import Treeprocessor
import re

if TYPE_CHECKING:  # pragma: no cover
    from xml.etree.ElementTree import Element


def _handle_double_quote(s, t):
    k, v = t.split('=', 1)
    return k, v.strip('"')


def _handle_single_quote(s, t):
    k, v = t.split('=', 1)
    return k, v.strip("'")


def _handle_key_value(s, t):
    return t.split('=', 1)


def _handle_word(s, t):
    if t.startswith('.'):
        return '.', t[1:]
    if t.startswith('#'):
        return 'id', t[1:]
    return t, t


_scanner = re.Scanner([
    (r'[^ =]+=".*?"', _handle_double_quote),  # 匹配双引号括起来的属性值
    (r"[^ =]+='.*?'", _handle_single_quote),  # 匹配单引号括起来的属性值
    (r'[^ =]+=[^ =]+', _handle_key_value),  # 匹配键值对
    (r'[^ =]+', _handle_word),  # 匹配单个属性
    (r' ', None)  # 匹配空格
])


def get_attrs(str: str) -> list[tuple[str, str]]:
    """ Parse attribute list and return a list of attribute tuples. """
    return _scanner.scan(str)[0]  # 解析属性列表并返回属性元组的列表


def isheader(elem: Element) -> bool:
    return elem.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']  # 判断元素是否为标题标签


class AttrListExtension(Extension):
    """ Attribute List extension for Python-Markdown """
    def extendMarkdown(self, md):
        md.treeprocessors.register(AttrListTreeprocessor(md), 'attr_list', 8)  # 注册属性列表处理器
        md.registerExtension(self)  # 注册扩展


def makeExtension(**kwargs):  # pragma: no cover
    return AttrListExtension(**kwargs)  # 创建属性列表扩展

```