# `.\DB-GPT-src\dbgpt\rag\knowledge\__init__.py`

```py
"""Module Of Knowledge."""

# 引入必要的类型和字典
from typing import Any, Dict
# 从 dbgpt.rag.knowledge.factory 模块中导入 KnowledgeFactory 类
from dbgpt.rag.knowledge.factory import KnowledgeFactory

# 全局变量，用于缓存已加载的模块
_MODULE_CACHE: Dict[str, Any] = {}

# 当通过属性访问时调用的特殊方法，实现延迟加载
def __getattr__(name: str):
    # 延迟加载机制，引入 importlib 模块
    import importlib

    # 如果请求的属性名已经在缓存中，则直接返回缓存中的对象
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    # 定义一个字典，将要加载的模块映射到其相对于当前模块的路径
    _LIBS = {
        "KnowledgeFactory": "factory",
        "Knowledge": "base",
        "KnowledgeType": "base",
        "ChunkStrategy": "base",
        "CSVKnowledge": "csv",
        "DatasourceKnowledge": "datasource",
        "DocxKnowledge": "docx",
        "HTMLKnowledge": "html",
        "MarkdownKnowledge": "markdown",
        "PDFKnowledge": "pdf",
        "PPTXKnowledge": "pptx",
        "StringKnowledge": "string",
        "TXTKnowledge": "txt",
        "URLKnowledge": "url",
        "ExcelKnowledge": "xlsx",
    }

    # 如果请求的属性名存在于 _LIBS 字典中，则加载对应模块
    if name in _LIBS:
        module_path = "." + _LIBS[name]
        module = importlib.import_module(module_path, __name__)
        attr = getattr(module, name)
        _MODULE_CACHE[name] = attr  # 将加载的对象缓存起来
        return attr

    # 如果请求的属性名不在 _LIBS 字典中，则抛出 AttributeError 异常
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 模块中允许被外部导入的所有属性列表
__all__ = [
    "KnowledgeFactory",
    "Knowledge",
    "KnowledgeType",
    "ChunkStrategy",
    "CSVKnowledge",
    "DatasourceKnowledge",
    "DocxKnowledge",
    "HTMLKnowledge",
    "MarkdownKnowledge",
    "PDFKnowledge",
    "PPTXKnowledge",
    "StringKnowledge",
    "TXTKnowledge",
    "URLKnowledge",
    "ExcelKnowledge",
]
```