# `.\graphrag\graphrag\config\input_models\cache_config_input.py`

```py
# 著作权声明和许可证信息，指明代码版权及授权方式
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块和类
"""Parameterization settings for the default configuration."""

# 从 typing_extensions 模块导入 NotRequired 类型提示和 TypedDict 类
from typing_extensions import NotRequired, TypedDict

# 从 graphrag.config.enums 模块导入 CacheType 枚举类型
from graphrag.config.enums import CacheType

# 定义一个 TypedDict 子类 CacheConfigInput，表示缓存配置的默认部分
class CacheConfigInput(TypedDict):
    """The default configuration section for Cache."""
    
    # type 字段，类型为 NotRequired[CacheType | str | None]
    type: NotRequired[CacheType | str | None]
    # base_dir 字段，类型为 NotRequired[str | None]
    base_dir: NotRequired[str | None]
    # connection_string 字段，类型为 NotRequired[str | None]
    connection_string: NotRequired[str | None]
    # container_name 字段，类型为 NotRequired[str | None]
    container_name: NotRequired[str | None]
    # storage_account_blob_url 字段，类型为 NotRequired[str | None]
    storage_account_blob_url: NotRequired[str | None]
```