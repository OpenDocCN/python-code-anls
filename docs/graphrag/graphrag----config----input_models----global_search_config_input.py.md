# `.\graphrag\graphrag\config\input_models\global_search_config_input.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

# 从 typing_extensions 模块导入 NotRequired 和 TypedDict 类型
from typing_extensions import NotRequired, TypedDict

# 定义一个 TypedDict 类型 GlobalSearchConfigInput，用于描述全局搜索配置的输入参数
class GlobalSearchConfigInput(TypedDict):
    """The default configuration section for Cache."""

    # 最大令牌数，可选参数，可以是整数、字符串或 None
    max_tokens: NotRequired[int | str | None]
    # 数据最大令牌数，可选参数，可以是整数、字符串或 None
    data_max_tokens: NotRequired[int | str | None]
    # 映射最大令牌数，可选参数，可以是整数、字符串或 None
    map_max_tokens: NotRequired[int | str | None]
    # 减少最大令牌数，可选参数，可以是整数、字符串或 None
    reduce_max_tokens: NotRequired[int | str | None]
    # 并发性，可选参数，可以是整数、字符串或 None
    concurrency: NotRequired[int | str | None]
```