# `markdown\markdown\extensions\tables.py`

```py

# Tables Extension for Python-Markdown
# ====================================

# Added parsing of tables to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/tables
# for documentation.

# Original code Copyright 2009 [Waylan Limberg](http://achinghead.com)

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Added parsing of tables to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/tables)
for details.
"""

from __future__ import annotations

from . import Extension  # 导入 Extension 模块
from ..blockprocessors import BlockProcessor  # 导入 BlockProcessor 模块
import xml.etree.ElementTree as etree  # 导入 xml.etree.ElementTree 模块并重命名为 etree
import re  # 导入 re 模块
from typing import TYPE_CHECKING, Any, Sequence  # 导入类型提示相关的模块

if TYPE_CHECKING:  # 如果是类型检查
    from .. import blockparser  # 导入 blockparser 模块

PIPE_NONE = 0  # 定义 PIPE_NONE 常量为 0
PIPE_LEFT = 1  # 定义 PIPE_LEFT 常量为 1
PIPE_RIGHT = 2  # 定义 PIPE_RIGHT 常量为 2

class TableExtension(Extension):
    """ Add tables to Markdown. """

    def __init__(self, **kwargs):
        self.config = {
            'use_align_attribute': [False, 'True to use align attribute instead of style.'],
        }
        """ Default configuration options. """

        super().__init__(**kwargs)  # 调用父类的初始化方法

    def extendMarkdown(self, md):
        """ Add an instance of `TableProcessor` to `BlockParser`. """
        if '|' not in md.ESCAPED_CHARS:  # 如果 '|' 不在 md.ESCAPED_CHARS 中
            md.ESCAPED_CHARS.append('|')  # 将 '|' 添加到 md.ESCAPED_CHARS 中
        processor = TableProcessor(md.parser, self.getConfigs())  # 创建 TableProcessor 实例
        md.parser.blockprocessors.register(processor, 'table', 75)  # 将 processor 注册到 blockprocessors 中，类型为 'table'，优先级为 75

def makeExtension(**kwargs):  # 定义 makeExtension 函数，用于创建 TableExtension 实例
    return TableExtension(**kwargs)  # 返回 TableExtension 实例

```