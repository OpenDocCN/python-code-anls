# `.\graphrag\graphrag\index\verbs\graph\merge\typing.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'BasicMergeOperation', 'StringOperation', 'NumericOperation' and 'DetailedAttributeMergeOperation' models."""

# 导入必要的模块和类
from dataclasses import dataclass
from enum import Enum


class BasicMergeOperation(str, Enum):
    """Basic Merge Operation class definition."""

    Replace = "replace"  # 替换操作，用于基本合并操作
    Skip = "skip"  # 跳过操作，用于基本合并操作


class StringOperation(str, Enum):
    """String Operation class definition."""

    Concat = "concat"  # 字符串连接操作
    Replace = "replace"  # 字符串替换操作
    Skip = "skip"  # 跳过操作，用于字符串操作


class NumericOperation(str, Enum):
    """Numeric Operation class definition."""

    Sum = "sum"  # 求和操作，用于数字操作
    Average = "average"  # 平均值操作，用于数字操作
    Max = "max"  # 最大值操作，用于数字操作
    Min = "min"  # 最小值操作，用于数字操作
    Multiply = "multiply"  # 乘法操作，用于数字操作
    Replace = "replace"  # 替换操作，用于数字操作
    Skip = "skip"  # 跳过操作，用于数字操作


@dataclass
class DetailedAttributeMergeOperation:
    """Detailed attribute merge operation class definition."""

    operation: str  # 操作类型，可以是 StringOperation 或 NumericOperation

    # concat
    separator: str | None = None  # 连接字符串时的分隔符，可选
    delimiter: str | None = None  # 连接字符串时的分隔符，可选
    distinct: bool = False  # 是否去重，默认为 False


AttributeMergeOperation = str | DetailedAttributeMergeOperation
```