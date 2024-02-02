# `markdown\markdown\extensions\codehilite.py`

```py

# CodeHilite Extension for Python-Markdown
# ========================================

# Adds code/syntax highlighting to standard Python-Markdown code blocks.

# See https://Python-Markdown.github.io/extensions/code_hilite
# for documentation.

# Original code Copyright 2006-2008 [Waylan Limberg](http://achinghead.com/).

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Adds code/syntax highlighting to standard Python-Markdown code blocks.

See the [documentation](https://Python-Markdown.github.io/extensions/code_hilite)
for details.
"""

from __future__ import annotations

from . import Extension  # 导入 Extension 模块
from ..treeprocessors import Treeprocessor  # 导入 Treeprocessor 模块
from ..util import parseBoolValue  # 导入 parseBoolValue 函数
from typing import TYPE_CHECKING, Callable, Any  # 导入类型提示相关的模块

if TYPE_CHECKING:  # pragma: no cover
    import xml.etree.ElementTree as etree  # 如果是类型检查，导入 xml.etree.ElementTree 模块

try:  # pragma: no cover
    from pygments import highlight  # 导入 highlight 函数
    from pygments.lexers import get_lexer_by_name, guess_lexer  # 导入 get_lexer_by_name 和 guess_lexer 函数
    from pygments.formatters import get_formatter_by_name  # 导入 get_formatter_by_name 函数
    from pygments.util import ClassNotFound  # 导入 ClassNotFound 类
    pygments = True  # 设置 pygments 变量为 True
except ImportError:  # pragma: no cover
    pygments = False  # 设置 pygments 变量为 False


def parse_hl_lines(expr: str) -> list[int]:
    """Support our syntax for emphasizing certain lines of code.

    `expr` should be like '1 2' to emphasize lines 1 and 2 of a code block.
    Returns a list of integers, the line numbers to emphasize.
    """
    if not expr:  # 如果 expr 为空
        return []  # 返回空列表

    try:
        return list(map(int, expr.split()))  # 尝试将 expr 按空格分割后转换为整数列表
    except ValueError:  # pragma: no cover
        return []  # 如果出现 ValueError，返回空列表


# ------------------ The Main CodeHilite Class ----------------------
# ------------------ The Markdown Extension -------------------------------


class HiliteTreeprocessor(Treeprocessor):
    """ Highlight source code in code blocks. """

    config: dict[str, Any]  # 定义 config 属性为字典类型

    def code_unescape(self, text: str) -> str:
        """Unescape code."""
        text = text.replace("&lt;", "<")  # 替换 &lt; 为 <
        text = text.replace("&gt;", ">")  # 替换 &gt; 为 >
        # Escaped '&' should be replaced at the end to avoid
        # conflicting with < and >.
        text = text.replace("&amp;", "&")  # 替换 &amp; 为 &
        return text  # 返回处理后的文本

    def run(self, root: etree.Element) -> None:
        """ Find code blocks and store in `htmlStash`. """
        blocks = root.iter('pre')  # 遍历 root 中的 pre 元素
        for block in blocks:  # 遍历每个 pre 元素
            if len(block) == 1 and block[0].tag == 'code':  # 如果 pre 元素只有一个子元素且为 code
                local_config = self.config.copy()  # 复制 config 属性
                text = block[0].text  # 获取 code 元素的文本内容
                if text is None:  # 如果文本内容为空
                    continue  # 继续下一次循环
                code = CodeHilite(  # 创建 CodeHilite 对象
                    self.code_unescape(text),  # 处理文本内容
                    tab_length=self.md.tab_length,  # 设置 tab_length 属性
                    style=local_config.pop('pygments_style', 'default'),  # 设置 style 属性
                    **local_config  # 使用剩余的配置参数
                )
                placeholder = self.md.htmlStash.store(code.hilite())  # 将高亮后的代码存储到 htmlStash 中
                # Clear code block in `etree` instance
                block.clear()  # 清空 block 元素
                # Change to `p` element which will later
                # be removed when inserting raw html
                block.tag = 'p'  # 将 block 元素的标签改为 p
                block.text = placeholder  # 设置 block 元素的文本内容为 placeholder


class CodeHiliteExtension(Extension):
    """ Add source code highlighting to markdown code blocks. """

    def __init__(self, **kwargs):
        # define default configs
        self.config = {  # 默认配置
            'linenums': [
                None, "Use lines numbers. True|table|inline=yes, False=no, None=auto. Default: `None`."
            ],
            'guess_lang': [
                True, "Automatic language detection - Default: `True`."
            ],
            'css_class': [
                "codehilite", "Set class name for wrapper <div> - Default: `codehilite`."
            ],
            'pygments_style': [
                'default', 'Pygments HTML Formatter Style (Colorscheme). Default: `default`.'
            ],
            'noclasses': [
                False, 'Use inline styles instead of CSS classes - Default `False`.'
            ],
            'use_pygments': [
                True, 'Highlight code blocks with pygments. Disable if using a JavaScript library. Default: `True`.'
            ],
            'lang_prefix': [
                'language-', 'Prefix prepended to the language when `use_pygments` is false. Default: `language-`.'
            ],
            'pygments_formatter': [
                'html', 'Use a specific formatter for Pygments highlighting. Default: `html`.'
            ],
        }
        """ Default configuration options. """

        for key, value in kwargs.items():  # 遍历传入的参数
            if key in self.config:  # 如果参数在默认配置中
                self.setConfig(key, value)  # 设置配置
            else:
                # manually set unknown keywords.
                if isinstance(value, str):  # 如果值是字符串
                    try:
                        # Attempt to parse `str` as a boolean value
                        value = parseBoolValue(value, preserve_none=True)  # 尝试将字符串解析为布尔值
                    except ValueError:
                        pass  # Assume it's not a boolean value. Use as-is.
                self.config[key] = [value, '']  # 设置配置

    def extendMarkdown(self, md):
        """ Add `HilitePostprocessor` to Markdown instance. """
        hiliter = HiliteTreeprocessor(md)  # 创建 HiliteTreeprocessor 实例
        hiliter.config = self.getConfigs()  # 设置配置
        md.treeprocessors.register(hiliter, 'hilite', 30)  # 注册 treeprocessor

        md.registerExtension(self)  # 注册扩展


def makeExtension(**kwargs):  # pragma: no cover
    return CodeHiliteExtension(**kwargs)  # 创建 CodeHiliteExtension 实例

```