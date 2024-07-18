# `.\graphrag\graphrag\config\input_models\storage_config_input.py`

```py
# 导入必要的模块和类
"""Parameterization settings for the default configuration."""

# 导入特定模块和类以供使用
from typing_extensions import NotRequired, TypedDict

# 导入 StorageType 枚举类型以便使用
from graphrag.config.enums import StorageType

# 定义一个 TypedDict 类型的子类，用于描述 StorageConfigInput 类型
class StorageConfigInput(TypedDict):
    """The default configuration section for Storage."""

    # 定义一个可选的 'type' 键，可以是 StorageType、str 或 None 类型
    type: NotRequired[StorageType | str | None]

    # 定义一个可选的 'base_dir' 键，可以是 str 或 None 类型
    base_dir: NotRequired[str | None]

    # 定义一个可选的 'connection_string' 键，可以是 str 或 None 类型
    connection_string: NotRequired[str | None]

    # 定义一个可选的 'container_name' 键，可以是 str 或 None 类型
    container_name: NotRequired[str | None]

    # 定义一个可选的 'storage_account_blob_url' 键，可以是 str 或 None 类型
    storage_account_blob_url: NotRequired[str | None]
```