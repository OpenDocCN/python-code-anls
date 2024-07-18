# `.\graphrag\graphrag\config\models\llm_parameters.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM Parameters model."""

# 导入必要的模块和类
from pydantic import BaseModel, ConfigDict, Field

# 导入默认配置和枚举类型
import graphrag.config.defaults as defs
from graphrag.config.enums import LLMType


# 定义 LLMParameters 类，表示语言模型的参数
class LLMParameters(BaseModel):
    """LLM Parameters model."""

    # 模型配置字典，允许额外字段，但保护命名空间
    model_config = ConfigDict(protected_namespaces=(), extra="allow")
    
    # LLM 服务的 API 密钥，可选，默认为 None
    api_key: str | None = Field(
        description="The API key to use for the LLM service.",
        default=None,
    )
    
    # LLM 模型的类型，使用 defs.LLM_TYPE 的默认值
    type: LLMType = Field(
        description="The type of LLM model to use.", default=defs.LLM_TYPE
    )
    
    # 要使用的 LLM 模型的名称，默认为 defs.LLM_MODEL
    model: str = Field(description="The LLM model to use.", default=defs.LLM_MODEL)
    
    # 生成的最大令牌数，默认为 defs.LLM_MAX_TOKENS
    max_tokens: int | None = Field(
        description="The maximum number of tokens to generate.",
        default=defs.LLM_MAX_TOKENS,
    )
    
    # 用于令牌生成的温度值，默认为 defs.LLM_TEMPERATURE
    temperature: float | None = Field(
        description="The temperature to use for token generation.",
        default=defs.LLM_TEMPERATURE,
    )
    
    # 用于令牌生成的 top-p 值，默认为 defs.LLM_TOP_P
    top_p: float | None = Field(
        description="The top-p value to use for token generation.",
        default=defs.LLM_TOP_P,
    )
    
    # 要生成的完成数目，默认为 defs.LLM_N
    n: int | None = Field(
        description="The number of completions to generate.",
        default=defs.LLM_N,
    )
    
    # 请求超时时间，默认为 defs.LLM_REQUEST_TIMEOUT
    request_timeout: float = Field(
        description="The request timeout to use.", default=defs.LLM_REQUEST_TIMEOUT
    )
    
    # LLM API 的基础 URL，默认为 None
    api_base: str | None = Field(
        description="The base URL for the LLM API.", default=None
    )
    
    # 要使用的 LLM API 的版本号，默认为 None
    api_version: str | None = Field(
        description="The version of the LLM API to use.", default=None
    )
    
    # 要用于 LLM 服务的组织名称，默认为 None
    organization: str | None = Field(
        description="The organization to use for the LLM service.", default=None
    )
    
    # 要用于 LLM 服务的代理设置，默认为 None
    proxy: str | None = Field(
        description="The proxy to use for the LLM service.", default=None
    )
    
    # 达到认知服务的终结点，默认为 None
    cognitive_services_endpoint: str | None = Field(
        description="The endpoint to reach cognitives services.", default=None
    )
    
    # 要用于 LLM 服务的部署名称，默认为 None
    deployment_name: str | None = Field(
        description="The deployment name to use for the LLM service.", default=None
    )
    
    # 模型是否支持 JSON 输出模式，默认为 None
    model_supports_json: bool | None = Field(
        description="Whether the model supports JSON output mode.", default=None
    )
    
    # 每分钟用于 LLM 服务的令牌数，默认为 defs.LLM_TOKENS_PER_MINUTE
    tokens_per_minute: int = Field(
        description="The number of tokens per minute to use for the LLM service.",
        default=defs.LLM_TOKENS_PER_MINUTE,
    )
    
    # 每分钟用于 LLM 服务的请求数，默认为 defs.LLM_REQUESTS_PER_MINUTE
    requests_per_minute: int = Field(
        description="The number of requests per minute to use for the LLM service.",
        default=defs.LLM_REQUESTS_PER_MINUTE,
    )
    
    # LLM 服务的最大重试次数，默认为 defs.LLM_MAX_RETRIES
    max_retries: int = Field(
        description="The maximum number of retries to use for the LLM service.",
        default=defs.LLM_MAX_RETRIES,
    )
    
    # LLM 服务的最大重试等待时间，默认为 defs.LLM_MAX_RETRY_WAIT
    max_retry_wait: float = Field(
        description="The maximum retry wait to use for the LLM service.",
        default=defs.LLM_MAX_RETRY_WAIT,
    )
    # 定义一个布尔型字段，用于控制是否在速率限制建议时进行休眠。
    sleep_on_rate_limit_recommendation: bool = Field(
        description="Whether to sleep on rate limit recommendations.",  # 描述字段用途：是否在速率限制建议时进行休眠
        default=defs.LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION,  # 默认值来自于常量 defs.LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION
    )
    
    # 定义一个整数字段，用于设置LLM服务的并发请求数量。
    concurrent_requests: int = Field(
        description="Whether to use concurrent requests for the LLM service.",  # 描述字段用途：是否为LLM服务使用并发请求
        default=defs.LLM_CONCURRENT_REQUESTS,  # 默认值来自于常量 defs.LLM_CONCURRENT_REQUESTS
    )
```