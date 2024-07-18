# `.\graphrag\graphrag\config\models\global_search_config.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

# 导入 pydantic 的 BaseModel 和 Field 类
from pydantic import BaseModel, Field

# 导入默认配置模块
import graphrag.config.defaults as defs

# 定义全局搜索配置类 GlobalSearchConfig，继承自 pydantic 的 BaseModel
class GlobalSearchConfig(BaseModel):
    """The default configuration section for Cache."""

    # 温度参数，用于生成令牌，可以为 None
    temperature: float | None = Field(
        description="The temperature to use for token generation.",
        default=defs.GLOBAL_SEARCH_LLM_TEMPERATURE,
    )

    # top-p 值，用于生成令牌，可以为 None
    top_p: float | None = Field(
        description="The top-p value to use for token generation.",
        default=defs.GLOBAL_SEARCH_LLM_TOP_P,
    )

    # 要生成的完成数目
    n: int | None = Field(
        description="The number of completions to generate.",
        default=defs.GLOBAL_SEARCH_LLM_N,
    )

    # 令牌的最大上下文大小
    max_tokens: int = Field(
        description="The maximum context size in tokens.",
        default=defs.GLOBAL_SEARCH_MAX_TOKENS,
    )

    # 数据 llm 的最大令牌数目
    data_max_tokens: int = Field(
        description="The data llm maximum tokens.",
        default=defs.GLOBAL_SEARCH_DATA_MAX_TOKENS,
    )

    # 地图 llm 的最大令牌数目
    map_max_tokens: int = Field(
        description="The map llm maximum tokens.",
        default=defs.GLOBAL_SEARCH_MAP_MAX_TOKENS,
    )

    # 减少 llm 的最大令牌数目
    reduce_max_tokens: int = Field(
        description="The reduce llm maximum tokens.",
        default=defs.GLOBAL_SEARCH_REDUCE_MAX_TOKENS,
    )

    # 并发请求的数量
    concurrency: int = Field(
        description="The number of concurrent requests.",
        default=defs.GLOBAL_SEARCH_CONCURRENCY,
    )
```