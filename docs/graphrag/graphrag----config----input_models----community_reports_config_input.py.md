# `.\graphrag\graphrag\config\input_models\community_reports_config_input.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

# 从 typing_extensions 模块中导入 NotRequired 类型提示
from typing_extensions import NotRequired

# 从当前包中导入 LLMConfigInput 类
from .llm_config_input import LLMConfigInput

# 定义 CommunityReportsConfigInput 类，继承自 LLMConfigInput 类
class CommunityReportsConfigInput(LLMConfigInput):
    """Configuration section for community reports."""

    # prompt 属性，可选的字符串或 None 类型
    prompt: NotRequired[str | None]

    # max_length 属性，可选的整数或字符串或 None 类型
    max_length: NotRequired[int | str | None]

    # max_input_length 属性，可选的整数或字符串或 None 类型
    max_input_length: NotRequired[int | str | None]

    # strategy 属性，可选的字典或 None 类型
    strategy: NotRequired[dict | None]
```