# `.\graphrag\graphrag\config\input_models\reporting_config_input.py`

```py
# 引入必要的模块和库
"""Parameterization settings for the default configuration."""

# 从 typing_extensions 模块中导入 NotRequired 和 TypedDict 类型
from typing_extensions import NotRequired, TypedDict

# 从 graphrag.config.enums 模块中导入 ReportingType 枚举类型
from graphrag.config.enums import ReportingType

# 定义一个名为 ReportingConfigInput 的字典类型，用于表示报告配置的默认部分
class ReportingConfigInput(TypedDict):
    """The default configuration section for Reporting."""

    # type 字段，可选，可以是 ReportingType 类型、str 类型、或者 None
    type: NotRequired[ReportingType | str | None]
    # base_dir 字段，可选，字符串类型或者 None
    base_dir: NotRequired[str | None]
    # connection_string 字段，可选，字符串类型或者 None
    connection_string: NotRequired[str | None]
    # container_name 字段，可选，字符串类型或者 None
    container_name: NotRequired[str | None]
    # storage_account_blob_url 字段，可选，字符串类型或者 None
    storage_account_blob_url: NotRequired[str | None]
```