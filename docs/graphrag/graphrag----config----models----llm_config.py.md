# `.\graphrag\graphrag\config\models\llm_config.py`

```py
# 导入必要的模块和类
"""Parameterization settings for the default configuration."""

from datashaper import AsyncType  # 导入AsyncType类，用于定义异步操作类型
from pydantic import BaseModel, Field  # 导入BaseModel和Field类，用于定义数据模型和字段

import graphrag.config.defaults as defs  # 导入默认配置模块，别名为defs

from .llm_parameters import LLMParameters  # 导入LLMParameters类，定义LLM参数
from .parallelization_parameters import ParallelizationParameters  # 导入ParallelizationParameters类，定义并行化参数


class LLMConfig(BaseModel):
    """Base class for LLM-configured steps."""
    
    llm: LLMParameters = Field(
        description="The LLM configuration to use.", default=LLMParameters()
    )  # 定义llm字段，使用LLMParameters类作为数据类型，默认为LLMParameters的实例，描述为使用的LLM配置

    parallelization: ParallelizationParameters = Field(
        description="The parallelization configuration to use.",
        default=ParallelizationParameters(),
    )  # 定义parallelization字段，使用ParallelizationParameters类作为数据类型，默认为ParallelizationParameters的实例，描述为使用的并行化配置

    async_mode: AsyncType = Field(
        description="The async mode to use.", default=defs.ASYNC_MODE
    )  # 定义async_mode字段，使用AsyncType类作为数据类型，默认为defs.ASYNC_MODE的值，描述为使用的异步模式
```