# `markdown\markdown\extensions\meta.py`

```

# Meta Data Extension for Python-Markdown
# =======================================

# This extension adds Meta Data handling to markdown.

# See https://Python-Markdown.github.io/extensions/meta_data
# for documentation.

# Original code Copyright 2007-2008 [Waylan Limberg](http://achinghead.com).

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
This extension adds Meta Data handling to markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/meta_data)
for details.
"""

from __future__ import annotations

from . import Extension  # 导入扩展模块
from ..preprocessors import Preprocessor  # 导入预处理器模块
import re  # 导入正则表达式模块
import logging  # 导入日志模块
from typing import Any  # 导入类型提示模块

log = logging.getLogger('MARKDOWN')  # 获取名为 'MARKDOWN' 的日志记录器

# Global Vars
META_RE = re.compile(r'^[ ]{0,3}(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)')  # 编译正则表达式，用于匹配元数据
META_MORE_RE = re.compile(r'^[ ]{4,}(?P<value>.*)')  # 编译正则表达式，用于匹配更多的元数据
BEGIN_RE = re.compile(r'^-{3}(\s.*)?')  # 编译正则表达式，用于匹配元数据开始
END_RE = re.compile(r'^(-{3}|\.{3})(\s.*)?')  # 编译正则表达式，用于匹配元数据结束


class MetaExtension (Extension):
    """ Meta-Data extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Add `MetaPreprocessor` to Markdown instance. """
        md.registerExtension(self)  # 注册扩展
        self.md = md  # 设置 Markdown 实例
        md.preprocessors.register(MetaPreprocessor(md), 'meta', 27)  # 注册预处理器

    def reset(self) -> None:
        self.md.Meta = {}  # 重置 Markdown 实例的元数据


class MetaPreprocessor(Preprocessor):
    """ Get Meta-Data. """

    def run(self, lines: list[str]) -> list[str]:
        """ Parse Meta-Data and store in Markdown.Meta. """
        meta: dict[str, Any] = {}  # 创建元数据字典
        key = None  # 初始化键为 None
        if lines and BEGIN_RE.match(lines[0]):  # 如果存在行且匹配到元数据开始
            lines.pop(0)  # 移除第一行
        while lines:
            line = lines.pop(0)  # 弹出第一行
            m1 = META_RE.match(line)  # 匹配元数据
            if line.strip() == '' or END_RE.match(line):  # 如果是空行或者匹配到元数据结束
                break  # 结束循环
            if m1:
                key = m1.group('key').lower().strip()  # 获取键
                value = m1.group('value').strip()  # 获取值
                try:
                    meta[key].append(value)  # 尝试将值添加到键对应的列表中
                except KeyError:
                    meta[key] = [value]  # 如果键不存在，则创建键值对
            else:
                m2 = META_MORE_RE.match(line)  # 匹配更多的元数据
                if m2 and key:
                    # Add another line to existing key
                    meta[key].append(m2.group('value').strip())  # 将另一行添加到现有键的值中
                else:
                    lines.insert(0, line)  # 将行插入到列表开头
                    break  # 结束循环
        self.md.Meta = meta  # 将解析后的元数据存储在 Markdown 实例中
        return lines  # 返回剩余的行


def makeExtension(**kwargs):  # pragma: no cover
    return MetaExtension(**kwargs)  # 创建并返回 MetaExtension 实例

```