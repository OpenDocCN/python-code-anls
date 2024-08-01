# `.\DB-GPT-src\dbgpt\storage\vector_store\filters.py`

```py
"""Vector Store Meta data filters."""
# 导入必要的模块和类
from enum import Enum  # 导入枚举类型模块
from typing import List, Union  # 导入类型提示模块

from dbgpt._private.pydantic import BaseModel, Field  # 导入数据验证和模型定义相关的类


class FilterOperator(str, Enum):
    """Meta data filter operator."""
    # 定义元数据过滤器操作符的枚举类

    EQ = "=="  # 等于操作符
    GT = ">"    # 大于操作符
    LT = "<"    # 小于操作符
    NE = "!="   # 不等于操作符
    GTE = ">="  # 大于等于操作符
    LTE = "<="  # 小于等于操作符
    IN = "in"   # 包含操作符
    NIN = "nin"  # 不包含操作符
    EXISTS = "exists"  # 存在操作符


class FilterCondition(str, Enum):
    """Vector Store Meta data filter conditions."""
    # 定义向量存储元数据过滤条件的枚举类

    AND = "and"  # 与条件
    OR = "or"    # 或条件


class MetadataFilter(BaseModel):
    """Meta data filter."""
    # 定义元数据过滤器模型类

    key: str = Field(
        ...,
        description="The key of metadata to filter.",
    )  # 元数据的键，用于过滤，不能为空

    operator: FilterOperator = Field(
        default=FilterOperator.EQ,
        description="The operator of metadata filter.",
    )  # 元数据过滤器的操作符，默认为等于操作符

    value: Union[str, int, float, List[str], List[int], List[float]] = Field(
        ...,
        description="The value of metadata to filter.",
    )  # 元数据的值，可以是字符串、整数、浮点数或它们的列表，用于过滤，不能为空


class MetadataFilters(BaseModel):
    """Meta data filters."""
    # 定义元数据过滤器集合模型类

    condition: FilterCondition = Field(
        default=FilterCondition.AND,
        description="The condition of metadata filters.",
    )  # 元数据过滤器之间的条件，默认为与条件

    filters: List[MetadataFilter] = Field(
        ...,
        description="The metadata filters.",
    )  # 元数据过滤器的列表，不能为空
```