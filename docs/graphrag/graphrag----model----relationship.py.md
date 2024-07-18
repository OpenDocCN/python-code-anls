# `.\graphrag\graphrag\model\relationship.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing the 'Relationship' model."""

# 引入必要的库和模块
from dataclasses import dataclass
from typing import Any

# 引入自定义模块中的Identified类
from .identified import Identified


@dataclass
class Relationship(Identified):
    """A relationship between two entities. This is a generic relationship, and can be used to represent any type of relationship between any two entities."""

    source: str
    """The source entity name."""

    target: str
    """The target entity name."""

    weight: float | None = 1.0
    """The edge weight."""

    description: str | None = None
    """A description of the relationship (optional)."""

    description_embedding: list[float] | None = None
    """The semantic embedding for the relationship description (optional)."""

    text_unit_ids: list[str] | None = None
    """List of text unit IDs in which the relationship appears (optional)."""

    document_ids: list[str] | None = None
    """List of document IDs in which the relationship appears (optional)."""

    attributes: dict[str, Any] | None = None
    """Additional attributes associated with the relationship (optional). To be included in the search prompt"""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "short_id",
        source_key: str = "source",
        target_key: str = "target",
        description_key: str = "description",
        weight_key: str = "weight",
        text_unit_ids_key: str = "text_unit_ids",
        document_ids_key: str = "document_ids",
        attributes_key: str = "attributes",
    ) -> "Relationship":
        """Create a new relationship from the dict data."""
        # 使用提供的字典数据创建并返回一个新的Relationship对象
        return Relationship(
            id=d[id_key],  # 设置id属性
            short_id=d.get(short_id_key),  # 设置short_id属性，如果不存在则为None
            source=d[source_key],  # 设置source属性
            target=d[target_key],  # 设置target属性
            description=d.get(description_key),  # 设置description属性，如果不存在则为None
            weight=d.get(weight_key, 1.0),  # 设置weight属性，默认为1.0
            text_unit_ids=d.get(text_unit_ids_key),  # 设置text_unit_ids属性，如果不存在则为None
            document_ids=d.get(document_ids_key),  # 设置document_ids属性，如果不存在则为None
            attributes=d.get(attributes_key),  # 设置attributes属性，如果不存在则为None
        )
```