# `.\graphrag\graphrag\config\input_models\embed_graph_config_input.py`

```py
# 版权声明及许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块
"""Parameterization settings for the default configuration."""

# 导入类型相关的扩展，用于参数类型声明
from typing_extensions import NotRequired, TypedDict

# 定义一个类型字典类，用于描述 Node2Vec 的默认配置参数
class EmbedGraphConfigInput(TypedDict):
    """The default configuration section for Node2Vec."""
    
    # 是否启用 Node2Vec，默认可选值为布尔型、字符串或空值
    enabled: NotRequired[bool | str | None]
    # 随机漫步的数量，默认可选值为整数或字符串
    num_walks: NotRequired[int | str | None]
    # 每次随机漫步的步长，默认可选值为整数或字符串
    walk_length: NotRequired[int | str | None]
    # 窗口大小，用于生成节点上下文，默认可选值为整数或字符串
    window_size: NotRequired[int | str | None]
    # 迭代次数，默认可选值为整数或字符串
    iterations: NotRequired[int | str | None]
    # 随机数种子，默认可选值为整数或字符串
    random_seed: NotRequired[int | str | None]
    # Node2Vec 算法的策略配置，默认可选值为字典或空值
    strategy: NotRequired[dict | None]
```