# `markdown\markdown\extensions\extra.py`

```py

# Python-Markdown Extra Extension
# ===============================

# 一个包含各种 Python-Markdown 扩展的编译，模仿了 [PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/)。

# 请参阅 https://Python-Markdown.github.io/extensions/extra
# 以获取文档。

# 版权所有 Python Markdown 项目

# 许可证：[BSD](https://opensource.org/licenses/bsd-license.php)

"""
一个包含各种 Python-Markdown 扩展的编译，模仿了 [PHP Markdown Extra](http://michelf.com/projects/php-markdown/extra/)。

请注意，每个单独的扩展仍然需要在您的 `PYTHONPATH` 上可用。这个扩展只是将它们全部包装在一起，以便在初始化 Markdown 时只需要列出一个扩展。请参阅每个单独扩展的文档，了解有关该扩展的具体信息。

可能有其他扩展是与 Python-Markdown 一起分发的，但在 Extra 中没有包含在内。这些扩展不是 PHP Markdown Extra 的一部分，因此也不是 Python-Markdown Extra 的一部分。如果您真的希望 Extra 包括其他扩展，我们建议创建自己的 Extra 克隆，并赋予不同的名称。您也可以编辑下面定义的 `extensions` 全局变量，但请注意，这样的更改可能会在将来升级到任何 Python-Markdown 的版本时丢失。

请参阅[文档](https://Python-Markdown.github.io/extensions/extra)
获取详细信息。
"""

from __future__ import annotations

from . import Extension

extensions = [
    'fenced_code',
    'footnotes',
    'attr_list',
    'def_list',
    'tables',
    'abbr',
    'md_in_html'
]
""" 包含的扩展列表。 """


class ExtraExtension(Extension):
    """ 向 Markdown 类添加各种扩展。"""

    def __init__(self, **kwargs):
        """ `config` 是一个简单的持有者，稍后会传递给实际的扩展。 """
        self.config = kwargs

    def extendMarkdown(self, md):
        """ 注册扩展实例。 """
        md.registerExtensions(extensions, self.config)


def makeExtension(**kwargs):  # pragma: no cover
    return ExtraExtension(**kwargs)

```