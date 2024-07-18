# `.\graphrag\graphrag\config\models\parallelization_parameters.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入基础模型和字段描述符
"""LLM Parameters model."""
from pydantic import BaseModel, Field

# 导入默认配置模块
import graphrag.config.defaults as defs


# 并行化参数类，继承自 BaseModel
class ParallelizationParameters(BaseModel):
    """LLM Parameters model."""

    # LLM 服务使用的延迟时间，使用 Field 描述符设置，默认值从 defs.PARALLELIZATION_STAGGER 中获取
    stagger: float = Field(
        description="The stagger to use for the LLM service.",
        default=defs.PARALLELIZATION_STAGGER,
    )
    
    # LLM 服务使用的线程数，使用 Field 描述符设置，默认值从 defs.PARALLELIZATION_NUM_THREADS 中获取
    num_threads: int = Field(
        description="The number of threads to use for the LLM service.",
        default=defs.PARALLELIZATION_NUM_THREADS,
    )
```