# `.\graphrag\graphrag\config\input_models\llm_config_input.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

# 引入异步类型
from datashaper import AsyncType
# 引入类型字典的非必需扩展
from typing_extensions import NotRequired, TypedDict

# 引入本地模块中的输入参数类
from .llm_parameters_input import LLMParametersInput
from .parallelization_parameters_input import ParallelizationParametersInput

# 定义LLMConfigInput类，继承自TypedDict
class LLMConfigInput(TypedDict):
    """Base class for LLM-configured steps."""
    
    # 定义llm字段，可选，类型为LLMParametersInput或None
    llm: NotRequired[LLMParametersInput | None]
    # 定义parallelization字段，可选，类型为ParallelizationParametersInput或None
    parallelization: NotRequired[ParallelizationParametersInput | None]
    # 定义async_mode字段，可选，类型为AsyncType、str或None
    async_mode: NotRequired[AsyncType | str | None]
```