# `D:\src\scipysrc\scikit-learn\doc\sphinxext\autoshortsummary.py`

```
from sphinx.ext.autodoc import ModuleLevelDocumenter

# 继承自 ModuleLevelDocumenter 类，用于自动文档生成，仅呈现对象的简短摘要信息
class ShortSummaryDocumenter(ModuleLevelDocumenter):
    """An autodocumenter that only renders the short summary of the object."""

    # 定义对象类型为 "shortsummary"
    objtype = "shortsummary"

    # 禁用内容缩进
    content_indent = ""

    # 优先级设定为 -99，避免成为某些对象的默认文档生成器
    priority = -99

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        """Allow documenting any object."""
        return True

    def get_object_members(self, want_all):
        """Document no members."""
        return (False, [])

    def add_directive_header(self, sig):
        """Override default behavior to add no directive header or options."""
        # 重写默认行为，不添加指令头或选项
        pass

    def add_content(self, more_content):
        """Override default behavior to add only the first line of the docstring.

        Modified based on the part of processing docstrings in the original
        implementation of this method.

        https://github.com/sphinx-doc/sphinx/blob/faa33a53a389f6f8bc1f6ae97d6015fa92393c4a/sphinx/ext/autodoc/__init__.py#L609-L622
        """
        # 获取源文件名和文档字符串
        sourcename = self.get_sourcename()
        docstrings = self.get_doc()

        if docstrings is not None:
            if not docstrings:
                docstrings.append([])
            # 获取处理后文档字符串的第一行非空内容；如果对象没有简短摘要行，可能导致意外结果
            short_summary = next(
                (s for s in self.process_doc(docstrings) if s), "<no summary>"
            )
            # 添加简短摘要到文档中
            self.add_line(short_summary, sourcename, 0)


def setup(app):
    # 向 Sphinx 应用程序添加 ShortSummaryDocumenter 自动文档生成器
    app.add_autodocumenter(ShortSummaryDocumenter)
```