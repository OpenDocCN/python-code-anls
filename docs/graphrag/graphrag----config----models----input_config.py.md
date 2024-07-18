# `.\graphrag\graphrag\config\models\input_config.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数设置。"""

# 导入 pydantic 库中的 BaseModel 和 Field 类
from pydantic import BaseModel, Field

# 导入 graphrag.config.defaults 中的默认设置
import graphrag.config.defaults as defs
# 导入 graphrag.config.enums 中的枚举类型 InputFileType 和 InputType
from graphrag.config.enums import InputFileType, InputType

# 定义输入配置的 BaseModel 子类 InputConfig
class InputConfig(BaseModel):
    """Input 的默认配置部分。"""

    # 输入类型，使用 Field 定义，默认值为 defs.INPUT_TYPE
    type: InputType = Field(
        description="要使用的输入类型。", default=defs.INPUT_TYPE
    )
    # 输入文件类型，使用 Field 定义，默认值为 defs.INPUT_FILE_TYPE
    file_type: InputFileType = Field(
        description="要使用的输入文件类型。", default=defs.INPUT_FILE_TYPE
    )
    # 输入的基础目录，使用 Field 定义，默认值为 defs.INPUT_BASE_DIR
    base_dir: str = Field(
        description="要使用的输入基础目录。", default=defs.INPUT_BASE_DIR
    )
    # Azure Blob 存储的连接字符串，可以为 None，使用 Field 定义，默认为 None
    connection_string: str | None = Field(
        description="要使用的 Azure Blob 存储连接字符串。", default=None
    )
    # 存储账户 Blob URL，可以为 None，使用 Field 定义，默认为 None
    storage_account_blob_url: str | None = Field(
        description="要使用的存储账户 Blob URL。", default=None
    )
    # Azure Blob 存储的容器名称，可以为 None，使用 Field 定义，默认为 None
    container_name: str | None = Field(
        description="要使用的 Azure Blob 存储容器名称。", default=None
    )
    # 输入文件的编码方式，可以为 None，使用 Field 定义，默认为 defs.INPUT_FILE_ENCODING
    encoding: str | None = Field(
        description="要使用的输入文件编码方式。",
        default=defs.INPUT_FILE_ENCODING,
    )
    # 输入文件的匹配模式，使用 Field 定义，默认值为 defs.INPUT_TEXT_PATTERN
    file_pattern: str = Field(
        description="要使用的输入文件匹配模式。", default=defs.INPUT_TEXT_PATTERN
    )
    # 输入文件的过滤器，可以为 None，使用 Field 定义，默认为 None
    file_filter: dict[str, str] | None = Field(
        description="输入文件的可选过滤器。", default=None
    )
    # 输入源列，可以为 None，使用 Field 定义，默认为 None
    source_column: str | None = Field(
        description="要使用的输入源列。", default=None
    )
    # 时间戳列，可以为 None，使用 Field 定义，默认为 None
    timestamp_column: str | None = Field(
        description="要使用的时间戳列。", default=None
    )
    # 时间戳格式，可以为 None，使用 Field 定义，默认为 None
    timestamp_format: str | None = Field(
        description="要使用的时间戳格式。", default=None
    )
    # 输入文本列，使用 Field 定义，默认为 defs.INPUT_TEXT_COLUMN
    text_column: str = Field(
        description="要使用的输入文本列。", default=defs.INPUT_TEXT_COLUMN
    )
    # 标题列，可以为 None，使用 Field 定义，默认为 None
    title_column: str | None = Field(
        description="要使用的标题列。", default=None
    )
    # 文档属性列的列表，使用 Field 定义，默认为空列表
    document_attribute_columns: list[str] = Field(
        description="要使用的文档属性列。", default=[]
    )
```