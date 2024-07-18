# `.\graphrag\graphrag\model\text_unit.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing the 'TextUnit' model."""

from dataclasses import dataclass
from typing import Any

from .identified import Identified  # 导入Identified类


@dataclass
class TextUnit(Identified):
    """A protocol for a TextUnit item in a Document database."""
    
    text: str
    """The text of the unit."""

    text_embedding: list[float] | None = None
    """The text embedding for the text unit (optional)."""

    entity_ids: list[str] | None = None
    """List of entity IDs related to the text unit (optional)."""

    relationship_ids: list[str] | None = None
    """List of relationship IDs related to the text unit (optional)."""

    covariate_ids: dict[str, list[str]] | None = None
    "Dictionary of different types of covariates related to the text unit (optional)."

    n_tokens: int | None = None
    """The number of tokens in the text (optional)."""

    document_ids: list[str] | None = None
    """List of document IDs in which the text unit appears (optional)."""

    attributes: dict[str, Any] | None = None
    """A dictionary of additional attributes associated with the text unit (optional)."""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "short_id",
        text_key: str = "text",
        text_embedding_key: str = "text_embedding",
        entities_key: str = "entity_ids",
        relationships_key: str = "relationship_ids",
        covariates_key: str = "covariate_ids",
        n_tokens_key: str = "n_tokens",
        document_ids_key: str = "document_ids",
        attributes_key: str = "attributes",
    ) -> "TextUnit":
        """Create a new text unit from the dict data."""
        return TextUnit(
            id=d[id_key],  # 使用给定的id_key获取ID
            short_id=d.get(short_id_key),  # 使用给定的short_id_key获取短ID（可选）
            text=d[text_key],  # 使用给定的text_key获取文本内容
            text_embedding=d.get(text_embedding_key),  # 使用给定的text_embedding_key获取文本嵌入（可选）
            entity_ids=d.get(entities_key),  # 使用给定的entities_key获取实体ID列表（可选）
            relationship_ids=d.get(relationships_key),  # 使用给定的relationships_key获取关系ID列表（可选）
            covariate_ids=d.get(covariates_key),  # 使用给定的covariates_key获取协变量字典（可选）
            n_tokens=d.get(n_tokens_key),  # 使用给定的n_tokens_key获取文本中的标记数（可选）
            document_ids=d.get(document_ids_key),  # 使用给定的document_ids_key获取文档ID列表（可选）
            attributes=d.get(attributes_key),  # 使用给定的attributes_key获取与文本单元相关联的附加属性字典（可选）
        )
```