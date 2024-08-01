# `.\DB-GPT-src\dbgpt\model\proxy\llms\deepseek.py`

```py
import os  # 导入标准库 os，用于处理操作系统相关功能
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast  # 导入类型提示相关模块

from dbgpt.core import ModelRequest, ModelRequestContext  # 导入自定义模块中的类
from dbgpt.model.proxy.llms.proxy_model import ProxyModel  # 导入自定义模块中的类

from .chatgpt import OpenAILLMClient  # 从当前包中导入自定义模块

if TYPE_CHECKING:
    from httpx._types import ProxiesTypes  # 如果是类型检查模式，则导入特定类型

    from openai import AsyncAzureOpenAI, AsyncOpenAI  # 如果是类型检查模式，则导入特定类型

    ClientType = Union[AsyncAzureOpenAI, AsyncOpenAI]  # 定义一个类型别名，表示两种异步客户端类型

# 默认模型名称为 "deepseek-chat"
_DEFAULT_MODEL = "deepseek-chat"

async def deepseek_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    # 强制类型转换，确保 model 是 DeepseekLLMClient 类型
    client: DeepseekLLMClient = cast(DeepseekLLMClient, model.proxy_llm_client)
    # 创建模型请求上下文对象，指定为流式请求，并传入用户名称参数
    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))
    # 构建模型请求，包括模型名称、消息、温度、上下文等参数
    request = ModelRequest.build_request(
        client.default_model,
        messages=params["messages"],
        temperature=params.get("temperature"),
        context=context,
        max_new_tokens=params.get("max_new_tokens"),
    )
    # 使用客户端对象生成流式响应，异步迭代返回每个生成的结果
    async for r in client.generate_stream(request):
        yield r

class DeepseekLLMClient(OpenAILLMClient):
    """Deepseek LLM Client.

    Deepseek's API is compatible with OpenAI's API, so we inherit from OpenAILLMClient.

    API Reference: https://platform.deepseek.com/api-docs/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
        model: Optional[str] = _DEFAULT_MODEL,
        proxies: Optional["ProxiesTypes"] = None,
        timeout: Optional[int] = 240,
        model_alias: Optional[str] = "deepseek_proxyllm",
        context_length: Optional[int] = None,
        openai_client: Optional["ClientType"] = None,
        openai_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # 获取 API 基础地址，默认为环境变量 DEEPSEEK_API_BASE 或默认地址
        api_base = (
            api_base or os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com/v1"
        )
        # 获取 API 密钥，默认为环境变量 DEEPSEEK_API_KEY
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        # 获取模型名称，默认为 _DEFAULT_MODEL
        model = model or _DEFAULT_MODEL
        # 如果未指定上下文长度，则根据模型名称设置不同的默认长度
        if not context_length:
            if "deepseek-chat" in model:
                context_length = 1024 * 32  # 对于 deepseek-chat 模型，上下文长度为 32KB
            elif "deepseek-coder" in model:
                context_length = 1024 * 16  # 对于 deepseek-coder 模型，上下文长度为 16KB
            else:
                context_length = 1024 * 8   # 对于其他模型，默认上下文长度为 8KB

        # 如果缺少 API 密钥，则抛出 ValueError 异常
        if not api_key:
            raise ValueError(
                "Deepseek API key is required, please set 'DEEPSEEK_API_KEY' in "
                "environment variable or pass it to the client."
            )
        
        # 调用父类的构造方法，初始化客户端对象
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            api_type=api_type,
            api_version=api_version,
            model=model,
            proxies=proxies,
            timeout=timeout,
            model_alias=model_alias,
            context_length=context_length,
            openai_client=openai_client,
            openai_kwargs=openai_kwargs,
            **kwargs,
        )
    # 检查 SDK 版本是否满足要求
    def check_sdk_version(self, version: str) -> None:
        # 如果版本号小于 "1.0"，则抛出数值错误异常
        if not version >= "1.0":
            raise ValueError(
                "Deepseek API requires openai>=1.0, please upgrade it by "
                "`pip install --upgrade 'openai>=1.0'`"
            )

    # 获取默认模型属性
    @property
    def default_model(self) -> str:
        # 获取当前对象的私有属性 _model
        model = self._model
        # 如果 _model 为空，则将其设为默认模型 _DEFAULT_MODEL
        if not model:
            model = _DEFAULT_MODEL
        # 返回模型名称
        return model
```