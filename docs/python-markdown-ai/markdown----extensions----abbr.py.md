# `markdown\markdown\extensions\abbr.py`

```

# Abbreviation Extension for Python-Markdown
# ==========================================

# This extension adds abbreviation handling to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/abbreviations
# for documentation.

# Original code Copyright 2007-2008 [Waylan Limberg](http://achinghead.com/)
# and [Seemant Kulleen](http://www.kulleen.org/)

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
This extension adds abbreviation handling to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/abbreviations)
for details.
"""

from __future__ import annotations

from . import Extension  # 导入扩展模块
from ..blockprocessors import BlockProcessor  # 导入块处理器
from ..inlinepatterns import InlineProcessor  # 导入内联处理器
from ..util import AtomicString  # 导入原子字符串
import re  # 导入正则表达式模块
import xml.etree.ElementTree as etree  # 导入 XML 解析模块


class AbbrExtension(Extension):
    """ Abbreviation Extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Insert `AbbrPreprocessor` before `ReferencePreprocessor`. """
        md.parser.blockprocessors.register(AbbrPreprocessor(md.parser), 'abbr', 16)  # 在解析器中注册缩写预处理器


class AbbrPreprocessor(BlockProcessor):
    """ Abbreviation Preprocessor - parse text for abbr references. """

    RE = re.compile(r'^[*]\[(?P<abbr>[^\]]*)\][ ]?:[ ]*\n?[ ]*(?P<title>.*)$', re.MULTILINE)  # 定义正则表达式用于匹配缩写引用

    def test(self, parent: etree.Element, block: str) -> bool:
        return True  # 测试是否需要运行预处理器

    def run(self, parent: etree.Element, blocks: list[str]) -> bool:
        """
        Find and remove all Abbreviation references from the text.
        Each reference is set as a new `AbbrPattern` in the markdown instance.

        """
        block = blocks.pop(0)  # 弹出一个块
        m = self.RE.search(block)  # 使用正则表达式匹配块内容
        if m:
            abbr = m.group('abbr').strip()  # 获取缩写
            title = m.group('title').strip()  # 获取标题
            self.parser.md.inlinePatterns.register(
                AbbrInlineProcessor(self._generate_pattern(abbr), title), 'abbr-%s' % abbr, 2
            )  # 在解析器中注册内联处理器
            if block[m.end():].strip():
                # Add any content after match back to blocks as separate block
                blocks.insert(0, block[m.end():].lstrip('\n'))
            if block[:m.start()].strip():
                # Add any content before match back to blocks as separate block
                blocks.insert(0, block[:m.start()].rstrip('\n'))
            return True
        # No match. Restore block.
        blocks.insert(0, block)  # 恢复块内容
        return False

    def _generate_pattern(self, text: str) -> str:
        """
        Given a string, returns an regex pattern to match that string.

        'HTML' -> r'(?P<abbr>[H][T][M][L])'

        Note: we force each char as a literal match (in brackets) as we don't
        know what they will be beforehand.

        """
        chars = list(text)  # 将字符串转换为字符列表
        for i in range(len(chars)):
            chars[i] = r'[%s]' % chars[i]  # 将每个字符转换为正则表达式的字符类
        return r'(?P<abbr>\b%s\b)' % (r''.join(chars))  # 返回正则表达式模式

class AbbrInlineProcessor(InlineProcessor):
    """ Abbreviation inline pattern. """

    def __init__(self, pattern: str, title: str):
        super().__init__(pattern)
        self.title = title

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        abbr = etree.Element('abbr')  # 创建 XML 元素
        abbr.text = AtomicString(m.group('abbr'))  # 设置元素文本
        abbr.set('title', self.title)  # 设置元素属性
        return abbr, m.start(0), m.end(0)  # 返回处理结果

def makeExtension(**kwargs):  # pragma: no cover
    return AbbrExtension(**kwargs)  # 创建并返回缩写扩展

```