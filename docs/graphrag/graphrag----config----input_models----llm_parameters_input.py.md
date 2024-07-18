# `.\graphrag\graphrag\config\input_models\llm_parameters_input.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块
"""LLM Parameters model."""
from typing_extensions import NotRequired, TypedDict

# 导入枚举类型LLMType
from graphrag.config.enums import LLMType

# 定义LLMParametersInput类，继承自TypedDict
class LLMParametersInput(TypedDict):
    """LLM Parameters model."""

    # API密钥，可选的字符串或None
    api_key: NotRequired[str | None]
    # LLM类型，可选的LLMType枚举类型、字符串或None
    type: NotRequired[LLMType | str | None]
    # 模型名称，可选的字符串或None
    model: NotRequired[str | None]
    # 最大生成token数量，可选的整数或字符串或None
    max_tokens: NotRequired[int | str | None]
    # 请求超时时间，可选的浮点数或字符串或None
    request_timeout: NotRequired[float | str | None]
    # API基础地址，可选的字符串或None
    api_base: NotRequired[str | None]
    # API版本，可选的字符串或None
    api_version: NotRequired[str | None]
    # 组织名称，可选的字符串或None
    organization: NotRequired[str | None]
    # 代理设置，可选的字符串或None
    proxy: NotRequired[str | None]
    # 认知服务终端点，可选的字符串或None
    cognitive_services_endpoint: NotRequired[str | None]
    # 部署名称，可选的字符串或None
    deployment_name: NotRequired[str | None]
    # 模型是否支持JSON格式，可选的布尔值或字符串或None
    model_supports_json: NotRequired[bool | str | None]
    # 每分钟生成token数量，可选的整数或字符串或None
    tokens_per_minute: NotRequired[int | str | None]
    # 每分钟请求次数，可选的整数或字符串或None
    requests_per_minute: NotRequired[int | str | None]
    # 最大重试次数，可选的整数或字符串或None
    max_retries: NotRequired[int | str | None]
    # 最大重试等待时间，可选的浮点数或字符串或None
    max_retry_wait: NotRequired[float | str | None]
    # 在速率限制建议时休眠，可选的布尔值或字符串或None
    sleep_on_rate_limit_recommendation: NotRequired[bool | str | None]
    # 并发请求数，可选的整数或字符串或None
    concurrent_requests: NotRequired[int | str | None]
```