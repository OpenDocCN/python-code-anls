# `.\graphrag\graphrag\config\input_models\chunking_config_input.py`

```py
# 版权声明和许可声明，指出版权归属和使用许可
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块和库
"""Parameterization settings for the default configuration."""
# 用于默认配置的参数化设置

# 导入类型提示模块中的特定类型和类型组合
from typing_extensions import NotRequired, TypedDict

# 定义一个 TypedDict 类型 ChunkingConfigInput，用于配置分块处理的参数
class ChunkingConfigInput(TypedDict):
    """Configuration section for chunking."""
    # 分块处理配置部分

    # chunking 大小参数，可以是 int、str 或 None 类型，可选
    size: NotRequired[int | str | None]
    # 分块重叠参数，可以是 int、str 或 None 类型，可选
    overlap: NotRequired[int | str | None]
    # 按列分组参数，可以是 list[str]、str 或 None 类型，可选
    group_by_columns: NotRequired[list[str] | str | None]
    # 策略参数，可以是 dict 或 None 类型，可选
    strategy: NotRequired[dict | None]
```