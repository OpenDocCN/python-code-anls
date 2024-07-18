# `.\graphrag\graphrag\model\document.py`

```py
"""
A package containing the 'Document' model.
"""

from dataclasses import dataclass, field  # 导入必要的模块
from typing import Any  # 导入必要的模块

from .named import Named  # 导入自定义模块


@dataclass
class Document(Named):
    """A protocol for a document in the system."""
    
    type: str = "text"
    """Type of the document."""

    text_unit_ids: list[str] = field(default_factory=list)
    """list of text units in the document."""

    raw_content: str = ""
    """The raw text content of the document."""

    summary: str | None = None
    """Summary of the document (optional)."""

    summary_embedding: list[float] | None = None
    """The semantic embedding for the document summary (optional)."""

    raw_content_embedding: list[float] | None = None
    """The semantic embedding for the document raw content (optional)."""

    attributes: dict[str, Any] | None = None
    """A dictionary of structured attributes such as author, etc (optional)."""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "short_id",
        title_key: str = "title",
        type_key: str = "type",
        raw_content_key: str = "raw_content",
        summary_key: str = "summary",
        summary_embedding_key: str = "summary_embedding",
        raw_content_embedding_key: str = "raw_content_embedding",
        text_units_key: str = "text_units",
        attributes_key: str = "attributes",
    ) -> "Document":
        """Create a new document from the dict data."""
        return Document(
            id=d[id_key],  # 使用给定字典的'id_key'键的值作为文档的ID
            short_id=d.get(short_id_key),  # 获取字典中可选的'short_id_key'键的值
            title=d[title_key],  # 使用给定字典的'title_key'键的值作为文档的标题
            type=d.get(type_key, "text"),  # 获取字典中可选的'type_key'键的值，如果不存在则默认为"text"
            raw_content=d[raw_content_key],  # 使用给定字典的'raw_content_key'键的值作为文档的原始内容
            summary=d.get(summary_key),  # 获取字典中可选的'summary_key'键的值作为文档的摘要
            summary_embedding=d.get(summary_embedding_key),  # 获取字典中可选的'summary_embedding_key'键的值作为文档摘要的语义嵌入
            raw_content_embedding=d.get(raw_content_embedding_key),  # 获取字典中可选的'raw_content_embedding_key'键的值作为文档原始内容的语义嵌入
            text_unit_ids=d.get(text_units_key, []),  # 获取字典中可选的'text_units_key'键的值作为文档的文本单元ID列表，如果不存在则为空列表
            attributes=d.get(attributes_key),  # 获取字典中可选的'attributes_key'键的值作为结构化属性字典，如作者等
        )
```