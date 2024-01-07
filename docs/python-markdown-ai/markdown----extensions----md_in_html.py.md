# `markdown\markdown\extensions\md_in_html.py`

```

# Python-Markdown Markdown in HTML Extension
# ===============================

# An implementation of [PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/)'s
# parsing of Markdown syntax in raw HTML.

# See https://Python-Markdown.github.io/extensions/raw_html
# for documentation.

# Copyright The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
An implementation of [PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/)'s
parsing of Markdown syntax in raw HTML.

See the [documentation](https://Python-Markdown.github.io/extensions/raw_html)
for details.
"""

# 导入必要的模块和类
from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Literal, Mapping

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown

# 定义一个预处理器类，用于移除文本中的 HTML 块并存储以供后续检索
class HtmlBlockPreprocessor(Preprocessor):
    def run(self, lines: list[str]) -> list[str]:
        source = '\n'.join(lines)
        parser = HTMLExtractorExtra(self.md)
        parser.feed(source)
        parser.close()
        return ''.join(parser.cleandoc).split('\n')

# 定义一个后处理器类，用于处理 Markdown 中的 HTML 内容
class MarkdownInHTMLPostprocessor(RawHtmlPostprocessor):
    def stash_to_string(self, text: str | etree.Element) -> str:
        """ Override default to handle any `etree` elements still in the stash. """
        if isinstance(text, etree.Element):
            return self.md.serializer(text)
        else:
            return str(text)

# 定义一个扩展类，用于向 Markdown 类添加 HTML 内容的解析
class MarkdownInHtmlExtension(Extension):
    def extendMarkdown(self, md):
        """ Register extension instances. """

        # 替换原始 HTML 预处理器
        md.preprocessors.register(HtmlBlockPreprocessor(md), 'html_block', 20)
        # 添加一个块处理器，用于处理 `etree` 元素的占位符
        md.parser.blockprocessors.register(
            MarkdownInHtmlProcessor(md.parser), 'markdown_block', 105
        )
        # 替换原始 HTML 后处理器
        md.postprocessors.register(MarkdownInHTMLPostprocessor(md), 'raw_html', 30)

# 创建并返回 MarkdownInHtmlExtension 实例的函数
def makeExtension(**kwargs):  # pragma: no cover
    return MarkdownInHtmlExtension(**kwargs)

```