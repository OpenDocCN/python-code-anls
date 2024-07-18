# `.\graphrag\graphrag\llm\openai\openai_configuration.py`

```py
# 版权声明和许可证声明，指明版权归属和许可协议
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OpenAI Configuration class definition."""

# 导入所需模块和库
import json  # 导入处理 JSON 数据的模块
from collections.abc import Hashable  # 导入 Hashable 抽象基类
from typing import Any, cast  # 导入类型提示相关的库

from graphrag.llm.types import LLMConfig  # 导入 LLMConfig 类型


def _non_blank(value: str | None) -> str | None:
    # 如果值为 None，直接返回 None
    if value is None:
        return None
    # 去除字符串两端的空白字符
    stripped = value.strip()
    # 如果处理后的字符串为空字符串，则返回 None，否则返回原始值
    return None if stripped == "" else value


class OpenAIConfiguration(Hashable, LLMConfig):
    """OpenAI Configuration class definition."""

    # 核心配置项
    _api_key: str  # API 密钥
    _model: str  # 模型名称

    _api_base: str | None  # API 基础地址，可为 None
    _api_version: str | None  # API 版本号，可为 None
    _cognitive_services_endpoint: str | None  # 认知服务端点，可为 None
    _deployment_name: str | None  # 部署名称，可为 None
    _organization: str | None  # 组织名称，可为 None
    _proxy: str | None  # 代理设置，可为 None

    # 操作配置项
    _n: int | None  # 参数 n，可为 None
    _temperature: float | None  # 温度参数，可为 None
    _frequency_penalty: float | None  # 频率惩罚参数，可为 None
    _presence_penalty: float | None  # 存在惩罚参数，可为 None
    _top_p: float | None  # top-p 参数，可为 None
    _max_tokens: int | None  # 最大 token 数量，可为 None
    _response_format: str | None  # 响应格式，可为 None
    _logit_bias: dict[str, float] | None  # logit 偏置，可为 None
    _stop: list[str] | None  # 停止词列表，可为 None

    # 重试逻辑配置项
    _max_retries: int | None  # 最大重试次数，可为 None
    _max_retry_wait: float | None  # 最大重试等待时间，可为 None
    _request_timeout: float | None  # 请求超时时间，可为 None

    # 原始配置对象
    _raw_config: dict  # 原始配置对象的字典表示

    # 功能标志
    _model_supports_json: bool | None  # 模型是否支持 JSON，可为 None

    # 自定义配置项
    _tokens_per_minute: int | None  # 每分钟 token 数量，可为 None
    _requests_per_minute: int | None  # 每分钟请求次数，可为 None
    _concurrent_requests: int | None  # 并发请求数，可为 None
    _encoding_model: str | None  # 编码模型名称，可为 None
    _sleep_on_rate_limit_recommendation: bool | None  # 在速率限制建议时是否休眠，可为 None

    def __init__(
        self,
        config: dict,
        """Init method definition."""

        # 定义一个方法用于查找必需的配置项并返回字符串类型的值
        def lookup_required(key: str) -> str:
            return cast(str, config.get(key))

        # 定义一个方法用于查找字符串类型的配置项并返回字符串或None
        def lookup_str(key: str) -> str | None:
            return cast(str | None, config.get(key))

        # 定义一个方法用于查找整数类型的配置项并返回整数或None
        def lookup_int(key: str) -> int | None:
            result = config.get(key)
            if result is None:
                return None
            return int(cast(int, result))

        # 定义一个方法用于查找浮点数类型的配置项并返回浮点数或None
        def lookup_float(key: str) -> float | None:
            result = config.get(key)
            if result is None:
                return None
            return float(cast(float, result))

        # 定义一个方法用于查找字典类型的配置项并返回字典或None
        def lookup_dict(key: str) -> dict | None:
            return cast(dict | None, config.get(key))

        # 定义一个方法用于查找列表类型的配置项并返回列表或None
        def lookup_list(key: str) -> list | None:
            return cast(list | None, config.get(key))

        # 定义一个方法用于查找布尔类型的配置项并返回布尔值或None
        def lookup_bool(key: str) -> bool | None:
            value = config.get(key)
            if isinstance(value, str):
                return value.upper() == "TRUE"
            if isinstance(value, int):
                return value > 0
            return cast(bool | None, config.get(key))

        # 使用上述方法分别获取各个配置项的值，并赋给实例变量
        self._api_key = lookup_required("api_key")
        self._model = lookup_required("model")
        self._deployment_name = lookup_str("deployment_name")
        self._api_base = lookup_str("api_base")
        self._api_version = lookup_str("api_version")
        self._cognitive_services_endpoint = lookup_str("cognitive_services_endpoint")
        self._organization = lookup_str("organization")
        self._proxy = lookup_str("proxy")
        self._n = lookup_int("n")
        self._temperature = lookup_float("temperature")
        self._frequency_penalty = lookup_float("frequency_penalty")
        self._presence_penalty = lookup_float("presence_penalty")
        self._top_p = lookup_float("top_p")
        self._max_tokens = lookup_int("max_tokens")
        self._response_format = lookup_str("response_format")
        self._logit_bias = lookup_dict("logit_bias")
        self._stop = lookup_list("stop")
        self._max_retries = lookup_int("max_retries")
        self._request_timeout = lookup_float("request_timeout")
        self._model_supports_json = lookup_bool("model_supports_json")
        self._tokens_per_minute = lookup_int("tokens_per_minute")
        self._requests_per_minute = lookup_int("requests_per_minute")
        self._concurrent_requests = lookup_int("concurrent_requests")
        self._encoding_model = lookup_str("encoding_model")
        self._max_retry_wait = lookup_float("max_retry_wait")
        self._sleep_on_rate_limit_recommendation = lookup_bool(
            "sleep_on_rate_limit_recommendation"
        )
        self._raw_config = config
    def deployment_name(self) -> str | None:
        """Deployment name property definition."""
        # 返回非空的 _deployment_name 属性值
        return _non_blank(self._deployment_name)

    @property
    def api_base(self) -> str | None:
        """API base property definition."""
        # 返回非空的 _api_base 属性值，去除末尾的斜杠
        result = _non_blank(self._api_base)
        return result[:-1] if result and result.endswith("/") else result

    @property
    def api_version(self) -> str | None:
        """API version property definition."""
        # 返回非空的 _api_version 属性值
        return _non_blank(self._api_version)

    @property
    def cognitive_services_endpoint(self) -> str | None:
        """Cognitive services endpoint property definition."""
        # 返回非空的 _cognitive_services_endpoint 属性值
        return _non_blank(self._cognitive_services_endpoint)

    @property
    def organization(self) -> str | None:
        """Organization property definition."""
        # 返回非空的 _organization 属性值
        return _non_blank(self._organization)

    @property
    def proxy(self) -> str | None:
        """Proxy property definition."""
        # 返回非空的 _proxy 属性值
        return _non_blank(self._proxy)

    @property
    def n(self) -> int | None:
        """N property definition."""
        # 返回 _n 属性值
        return self._n

    @property
    def temperature(self) -> float | None:
        """Temperature property definition."""
        # 返回 _temperature 属性值
        return self._temperature

    @property
    def frequency_penalty(self) -> float | None:
        """Frequency penalty property definition."""
        # 返回 _frequency_penalty 属性值
        return self._frequency_penalty

    @property
    def presence_penalty(self) -> float | None:
        """Presence penalty property definition."""
        # 返回 _presence_penalty 属性值
        return self._presence_penalty

    @property
    def top_p(self) -> float | None:
        """Top p property definition."""
        # 返回 _top_p 属性值
        return self._top_p

    @property
    def max_tokens(self) -> int | None:
        """Max tokens property definition."""
        # 返回 _max_tokens 属性值
        return self._max_tokens

    @property
    def response_format(self) -> str | None:
        """Response format property definition."""
        # 返回非空的 _response_format 属性值
        return _non_blank(self._response_format)

    @property
    def logit_bias(self) -> dict[str, float] | None:
        """Logit bias property definition."""
        # 返回 _logit_bias 属性值
        return self._logit_bias

    @property
    def stop(self) -> list[str] | None:
        """Stop property definition."""
        # 返回 _stop 属性值
        return self._stop

    @property
    def max_retries(self) -> int | None:
        """Max retries property definition."""
        # 返回 _max_retries 属性值
        return self._max_retries

    @property
    def max_retry_wait(self) -> float | None:
        """Max retry wait property definition."""
        # 返回 _max_retry_wait 属性值
        return self._max_retry_wait

    @property
    def request_timeout(self) -> float | None:
        """Request timeout property definition."""
        # 返回 _request_timeout 属性值
        return self._request_timeout

    @property
    def model_supports_json(self) -> bool | None:
        """Model supports json property definition."""
        # 返回 _model_supports_json 属性值
        return self._model_supports_json

    @property
    def tokens_per_minute(self) -> int | None:
        """Tokens per minute property definition."""
        # 返回 _tokens_per_minute 属性值
        return self._tokens_per_minute
    # 返回每分钟的请求次数属性定义
    def requests_per_minute(self) -> int | None:
        """Requests per minute property definition."""
        return self._requests_per_minute

    # 返回并发请求的数量属性定义
    @property
    def concurrent_requests(self) -> int | None:
        """Concurrent requests property definition."""
        return self._concurrent_requests

    # 返回编码模型属性的定义
    @property
    def encoding_model(self) -> str | None:
        """Encoding model property definition."""
        return _non_blank(self._encoding_model)

    # 返回是否在收到 429 错误时建议休眠 <n> 秒（特定于 Azure）
    @property
    def sleep_on_rate_limit_recommendation(self) -> bool | None:
        """Whether to sleep for <n> seconds when recommended by 429 errors (azure-specific)."""
        return self._sleep_on_rate_limit_recommendation

    # 返回原始配置的字典表示
    @property
    def raw_config(self) -> dict:
        """Raw config method definition."""
        return self._raw_config

    # 查询指定名称的配置项，如不存在返回默认值
    def lookup(self, name: str, default_value: Any = None) -> Any:
        """Lookup method definition."""
        return self._raw_config.get(name, default_value)

    # 返回对象的 JSON 字符串表示形式
    def __str__(self) -> str:
        """Str method definition."""
        return json.dumps(self.raw_config, indent=4)

    # 返回对象的规范字符串表示形式
    def __repr__(self) -> str:
        """Repr method definition."""
        return f"OpenAIConfiguration({self._raw_config})"

    # 判断对象是否与另一个对象相等
    def __eq__(self, other: object) -> bool:
        """Eq method definition."""
        if not isinstance(other, OpenAIConfiguration):
            return False
        return self._raw_config == other._raw_config

    # 返回对象的哈希值
    def __hash__(self) -> int:
        """Hash method definition."""
        return hash(tuple(sorted(self._raw_config.items())))
```