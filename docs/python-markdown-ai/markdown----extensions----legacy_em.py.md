# `markdown\markdown\extensions\legacy_em.py`

```

# Legacy Em Extension for Python-Markdown
# =======================================

# This extension provides legacy behavior for _connected_words_.

# Copyright 2015-2018 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
This extension provides legacy behavior for _connected_words_.
"""

from __future__ import annotations  # 导入未来版本的注解特性

from . import Extension  # 导入当前目录下的 Extension 模块
from ..inlinepatterns import UnderscoreProcessor, EmStrongItem, EM_STRONG2_RE, STRONG_EM2_RE  # 导入相对路径下的模块和变量
import re  # 导入正则表达式模块

# _emphasis_
EMPHASIS_RE = r'(_)([^_]+)\1'  # 定义强调文本的正则表达式

# __strong__
STRONG_RE = r'(_{2})(.+?)\1'  # 定义加粗文本的正则表达式

# __strong_em___
STRONG_EM_RE = r'(_)\1(?!\1)([^_]+?)\1(?!\1)(.+?)\1{3}'  # 定义强调和加粗文本的正则表达式


class LegacyUnderscoreProcessor(UnderscoreProcessor):
    """Emphasis processor for handling strong and em matches inside underscores."""

    PATTERNS = [
        EmStrongItem(re.compile(EM_STRONG2_RE, re.DOTALL | re.UNICODE), 'double', 'strong,em'),  # 使用正则表达式创建 EmStrongItem 对象
        EmStrongItem(re.compile(STRONG_EM2_RE, re.DOTALL | re.UNICODE), 'double', 'em,strong'),  # 使用正则表达式创建 EmStrongItem 对象
        EmStrongItem(re.compile(STRONG_EM_RE, re.DOTALL | re.UNICODE), 'double2', 'strong,em'),  # 使用正则表达式创建 EmStrongItem 对象
        EmStrongItem(re.compile(STRONG_RE, re.DOTALL | re.UNICODE), 'single', 'strong'),  # 使用正则表达式创建 EmStrongItem 对象
        EmStrongItem(re.compile(EMPHASIS_RE, re.DOTALL | re.UNICODE), 'single', 'em')  # 使用正则表达式创建 EmStrongItem 对象
    ]


class LegacyEmExtension(Extension):
    """ Add legacy_em extension to Markdown class."""

    def extendMarkdown(self, md):
        """ Modify inline patterns. """
        md.inlinePatterns.register(LegacyUnderscoreProcessor(r'_'), 'em_strong2', 50)  # 注册 LegacyUnderscoreProcessor 对象到 Markdown 实例的 inlinePatterns 中


def makeExtension(**kwargs):  # pragma: no cover
    """ Return an instance of the `LegacyEmExtension` """
    return LegacyEmExtension(**kwargs)  # 返回 LegacyEmExtension 的实例

```