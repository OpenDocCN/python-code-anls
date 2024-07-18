# `.\graphrag\graphrag\index\emit\types.py`

```py
# 版权声明，指明版权归属及许可协议
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 引入枚举类型Enum，用于定义枚举类
from enum import Enum

# 定义表格发射器类型的枚举类TableEmitterType，继承自str和Enum
class TableEmitterType(str, Enum):
    """表格发射器类型枚举类."""

    # Json类型的表格发射器
    Json = "json"
    # Parquet类型的表格发射器
    Parquet = "parquet"
    # CSV类型的表格发射器
    CSV = "csv"
```