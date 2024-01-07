# `markdown\markdown\extensions\footnotes.py`

```

# Footnotes Extension for Python-Markdown
# =======================================

# Adds footnote handling to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/footnotes
# for documentation.

# Copyright The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Adds footnote handling to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/footnotes)
for details.
"""

from __future__ import annotations

from . import Extension  # 导入 Extension 模块
from ..blockprocessors import BlockProcessor  # 导入 BlockProcessor 模块
from ..inlinepatterns import InlineProcessor  # 导入 InlineProcessor 模块
from ..treeprocessors import Treeprocessor  # 导入 Treeprocessor 模块
from ..postprocessors import Postprocessor  # 导入 Postprocessor 模块
from .. import util  # 导入 util 模块
from collections import OrderedDict  # 导入 OrderedDict 模块
import re  # 导入 re 模块
import copy  # 导入 copy 模块
import xml.etree.ElementTree as etree  # 导入 etree 模块

FN_BACKLINK_TEXT = util.STX + "zz1337820767766393qq" + util.ETX  # 定义用于替换的文本
NBSP_PLACEHOLDER = util.STX + "qq3936677670287331zz" + util.ETX  # 定义用于替换的文本
RE_REF_ID = re.compile(r'(fnref)(\d+)')  # 编译正则表达式

class FootnoteBlockProcessor(BlockProcessor):
    """ Find all footnote references and store for later use. """

    RE = re.compile(r'^[ ]{0,3}\[\^([^\]]*)\]:[ ]*(.*)$', re.MULTILINE)  # 定义用于匹配的正则表达式

    def __init__(self, footnotes: FootnoteExtension):
        super().__init__(footnotes.parser)  # 调用父类的构造函数
        self.footnotes = footnotes  # 初始化属性

    def test(self, parent: etree.Element, block: str) -> bool:
        return True  # 测试函数，始终返回 True

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        """ Find, set, and remove footnote definitions. """
        block = blocks.pop(0)  # 弹出第一个块
        m = self.RE.search(block)  # 在块中搜索匹配的内容
        if m:
            # 处理匹配到的内容
            id = m.group(1)
            fn_blocks = [m.group(2)]

            # 处理剩余的块
            therest = block[m.end():].lstrip('\n')
            m2 = self.RE.search(therest)
            if m2:
                # 处理剩余的块中的另一个脚注
                before = therest[:m2.start()].rstrip('\n')
                fn_blocks[0] = '\n'.join([fn_blocks[0], self.detab(before)]).lstrip('\n')
                blocks.insert(0, therest[m2.start():])
            else:
                fn_blocks[0] = '\n'.join([fn_blocks[0], self.detab(therest)]).strip('\n')
                fn_blocks.extend(self.detectTabbed(blocks))

            footnote = "\n\n".join(fn_blocks)
            self.footnotes.setFootnote(id, footnote.rstrip())

            if block[:m.start()].strip():
                blocks.insert(0, block[:m.start()].rstrip('\n'))
            return True
        blocks.insert(0, block)
        return False

    def detectTabbed(self, blocks: list[str]) -> list[str]:
        """ Find indented text and remove indent before further processing. """
        fn_blocks = []
        while blocks:
            if blocks[0].startswith(' '*4):
                block = blocks.pop(0)
                m = self.RE.search(block)
                if m:
                    before = block[:m.start()].rstrip('\n')
                    fn_blocks.append(self.detab(before))
                    blocks.insert(0, block[m.start():])
                    break
                else:
                    fn_blocks.append(self.detab(block))
            else:
                break
        return fn_blocks

    def detab(self, block: str) -> str:
        """ Remove one level of indent from a block. """
        lines = block.split('\n')
        for i, line in enumerate(lines):
            if line.startswith(' '*4):
                lines[i] = line[4:]
        return '\n'.join(lines)


class FootnoteInlineProcessor(InlineProcessor):
    """ `InlineProcessor` for footnote markers in a document's body text. """

    def __init__(self, pattern: str, footnotes: FootnoteExtension):
        super().__init__(pattern)
        self.footnotes = footnotes

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        id = m.group(1)
        if id in self.footnotes.footnotes.keys():
            sup = etree.Element("sup")
            a = etree.SubElement(sup, "a")
            sup.set('id', self.footnotes.makeFootnoteRefId(id, found=True))
            a.set('href', '#' + self.footnotes.makeFootnoteId(id))
            a.set('class', 'footnote-ref')
            a.text = self.footnotes.getConfig("SUPERSCRIPT_TEXT").format(
                list(self.footnotes.footnotes.keys()).index(id) + 1
            )
            return sup, m.start(0), m.end(0)
        else:
            return None, None, None


class FootnotePostTreeprocessor(Treeprocessor):
    """ Amend footnote div with duplicates. """

    def __init__(self, footnotes: FootnoteExtension):
        self.footnotes = footnotes

    # 省略其他方法的注释

class FootnoteTreeprocessor(Treeprocessor):
    """ Build and append footnote div to end of document. """

    def __init__(self, footnotes: FootnoteExtension):
        self.footnotes = footnotes

    # 省略其他方法的注释

class FootnotePostprocessor(Postprocessor):
    """ Replace placeholders with html entities. """
    def __init__(self, footnotes: FootnoteExtension):
        self.footnotes = footnotes

    # 省略其他方法的注释

def makeExtension(**kwargs):  # pragma: no cover
    """ Return an instance of the `FootnoteExtension` """
    return FootnoteExtension(**kwargs)

```