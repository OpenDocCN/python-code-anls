# `markdown\markdown\extensions\def_list.py`

```

# Definition List Extension for Python-Markdown
# =============================================

# Adds parsing of Definition Lists to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/definition_lists
# for documentation.

# Original code Copyright 2008 [Waylan Limberg](http://achinghead.com)

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Adds parsing of Definition Lists to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/definition_lists)
for details.
"""

from __future__ import annotations

from . import Extension
from ..blockprocessors import BlockProcessor, ListIndentProcessor
import xml.etree.ElementTree as etree
import re

# 定义处理器类，用于处理定义列表
class DefListProcessor(BlockProcessor):
    """ Process Definition Lists. """

    # 定义正则表达式，用于匹配定义列表的语法
    RE = re.compile(r'(^|\n)[ ]{0,3}:[ ]{1,3}(.*?)(\n|$)')
    NO_INDENT_RE = re.compile(r'^[ ]{0,3}[^ :]')

    # 检测是否需要处理当前块
    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.search(block))

    # 处理当前块
    def run(self, parent: etree.Element, blocks: list[str]) -> bool | None:

        raw_block = blocks.pop(0)
        m = self.RE.search(raw_block)
        terms = [term.strip() for term in
                 raw_block[:m.start()].split('\n') if term.strip()]
        block = raw_block[m.end():]
        no_indent = self.NO_INDENT_RE.match(block)
        if no_indent:
            d, theRest = (block, None)
        else:
            d, theRest = self.detab(block)
        if d:
            d = '{}\n{}'.format(m.group(2), d)
        else:
            d = m.group(2)
        sibling = self.lastChild(parent)
        if not terms and sibling is None:
            # This is not a definition item. Most likely a paragraph that
            # starts with a colon at the beginning of a document or list.
            blocks.insert(0, raw_block)
            return False
        if not terms and sibling.tag == 'p':
            # The previous paragraph contains the terms
            state = 'looselist'
            terms = sibling.text.split('\n')
            parent.remove(sibling)
            # Acquire new sibling
            sibling = self.lastChild(parent)
        else:
            state = 'list'

        if sibling is not None and sibling.tag == 'dl':
            # This is another item on an existing list
            dl = sibling
            if not terms and len(dl) and dl[-1].tag == 'dd' and len(dl[-1]):
                state = 'looselist'
        else:
            # This is a new list
            dl = etree.SubElement(parent, 'dl')
        # Add terms
        for term in terms:
            dt = etree.SubElement(dl, 'dt')
            dt.text = term
        # Add definition
        self.parser.state.set(state)
        dd = etree.SubElement(dl, 'dd')
        self.parser.parseBlocks(dd, [d])
        self.parser.state.reset()

        if theRest:
            blocks.insert(0, theRest)

# 定义处理缩进的类，用于处理定义列表项的缩进
class DefListIndentProcessor(ListIndentProcessor):
    """ Process indented children of definition list items. """

    # 定义列表项类型和列表类型
    ITEM_TYPES = ['dd', 'li']
    """ Include `dd` in list item types. """
    LIST_TYPES = ['dl', 'ol', 'ul']
    """ Include `dl` is list types. """

    # 创建新的列表项
    def create_item(self, parent: etree.Element, block: str) -> None:
        """ Create a new `dd` or `li` (depending on parent) and parse the block with it as the parent. """

        dd = etree.SubElement(parent, 'dd')
        self.parser.parseBlocks(dd, [block])

# 定义扩展类，用于将定义列表添加到 Markdown 中
class DefListExtension(Extension):
    """ Add definition lists to Markdown. """

    # 扩展 Markdown 的功能
    def extendMarkdown(self, md):
        """ Add an instance of `DefListProcessor` to `BlockParser`. """
        md.parser.blockprocessors.register(DefListIndentProcessor(md.parser), 'defindent', 85)
        md.parser.blockprocessors.register(DefListProcessor(md.parser), 'deflist', 25)

# 创建扩展实例
def makeExtension(**kwargs):  # pragma: no cover
    return DefListExtension(**kwargs)

```