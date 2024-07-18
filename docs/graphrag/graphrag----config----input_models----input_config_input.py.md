# `.\graphrag\graphrag\config\input_models\input_config_input.py`

```py
# 引入必要的模块和类型定义
"""Parameterization settings for the default configuration."""

from typing_extensions import NotRequired, TypedDict
# 导入所需的模块和类型定义

from graphrag.config.enums import InputFileType, InputType
# 从graphrag.config.enums模块导入InputFileType和InputType枚举类型


class InputConfigInput(TypedDict):
    """The default configuration section for Input."""
    # 定义TypedDict子类InputConfigInput，表示输入配置的默认部分

    type: NotRequired[InputType | str | None]
    # 输入的类型，可以是InputType枚举类型、字符串或None，可选项
    file_type: NotRequired[InputFileType | str | None]
    # 文件的类型，可以是InputFileType枚举类型、字符串或None，可选项
    base_dir: NotRequired[str | None]
    # 基本目录，字符串或None，可选项
    connection_string: NotRequired[str | None]
    # 连接字符串，字符串或None，可选项
    container_name: NotRequired[str | None]
    # 容器名，字符串或None，可选项
    file_encoding: NotRequired[str | None]
    # 文件编码，字符串或None，可选项
    file_pattern: NotRequired[str | None]
    # 文件模式，字符串或None，可选项
    source_column: NotRequired[str | None]
    # 源列，字符串或None，可选项
    timestamp_column: NotRequired[str | None]
    # 时间戳列，字符串或None，可选项
    timestamp_format: NotRequired[str | None]
    # 时间戳格式，字符串或None，可选项
    text_column: NotRequired[str | None]
    # 文本列，字符串或None，可选项
    title_column: NotRequired[str | None]
    # 标题列，字符串或None，可选项
    document_attribute_columns: NotRequired[list[str] | str | None]
    # 文档属性列，字符串列表、字符串或None，可选项
    storage_account_blob_url: NotRequired[str | None]
    # 存储账户Blob URL，字符串或None，可选项
```