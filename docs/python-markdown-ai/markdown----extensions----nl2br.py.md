# `markdown\markdown\extensions\nl2br.py`

```
# `NL2BR` Extension
# ===============

# A Python-Markdown extension to treat newlines as hard breaks; like
# GitHub-flavored Markdown does.

# See https://Python-Markdown.github.io/extensions/nl2br
# for documentation.

# Original code Copyright 2011 [Brian Neal](https://deathofagremmie.com/)

# All changes Copyright 2011-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
A Python-Markdown extension to treat newlines as hard breaks; like
GitHub-flavored Markdown does.

See the [documentation](https://Python-Markdown.github.io/extensions/nl2br)
for details.
"""

from __future__ import annotations

from . import Extension  # 导入 Extension 模块
from ..inlinepatterns import SubstituteTagInlineProcessor  # 导入 SubstituteTagInlineProcessor 模块

BR_RE = r'\n'  # 定义换行符正则表达式

class Nl2BrExtension(Extension):  # 定义 Nl2BrExtension 类，继承自 Extension 类

    def extendMarkdown(self, md):  # 定义 extendMarkdown 方法
        """ Add a `SubstituteTagInlineProcessor` to Markdown. """
        br_tag = SubstituteTagInlineProcessor(BR_RE, 'br')  # 创建 SubstituteTagInlineProcessor 实例
        md.inlinePatterns.register(br_tag, 'nl', 5)  # 将 br_tag 注册为内联模式处理器，优先级为 5

def makeExtension(**kwargs):  # pragma: no cover
    return Nl2BrExtension(**kwargs)  # 创建并返回 Nl2BrExtension 实例
```