# `markdown\markdown\extensions\admonition.py`

```

# Admonition extension for Python-Markdown
# ========================================

# Adds rST-style admonitions. Inspired by [rST][] feature with the same name.

# [rST]: http://docutils.sourceforge.net/docs/ref/rst/directives.html#specific-admonitions

# See https://Python-Markdown.github.io/extensions/admonition
# for documentation.

# Original code Copyright [Tiago Serafim](https://www.tiagoserafim.com/).

# All changes Copyright The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Adds rST-style admonitions. Inspired by [rST][] feature with the same name.

[rST]: http://docutils.sourceforge.net/docs/ref/rst/directives.html#specific-admonitions

See the [documentation](https://Python-Markdown.github.io/extensions/admonition)
for details.
"""

from __future__ import annotations

from . import Extension  # 导入自定义的 Extension 模块
from ..blockprocessors import BlockProcessor  # 导入 blockprocessors 模块中的 BlockProcessor 类
import xml.etree.ElementTree as etree  # 导入 xml.etree.ElementTree 模块并重命名为 etree
import re  # 导入 re 模块，用于正则表达式操作
from typing import TYPE_CHECKING  # 导入 TYPE_CHECKING 类型提示

if TYPE_CHECKING:  # 如果是类型检查模式
    from markdown import blockparser  # 导入 markdown 模块中的 blockparser 类


class AdmonitionExtension(Extension):
    """ Admonition extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Add Admonition to Markdown instance. """
        md.registerExtension(self)  # 注册扩展

        md.parser.blockprocessors.register(AdmonitionProcessor(md.parser), 'admonition', 105)  # 注册 AdmonitionProcessor 处理器


def makeExtension(**kwargs):  # 创建扩展的函数
    return AdmonitionExtension(**kwargs)  # 返回 AdmonitionExtension 实例

```