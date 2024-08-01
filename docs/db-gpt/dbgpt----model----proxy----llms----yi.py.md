# `.\DB-GPT-src\dbgpt\model\proxy\llms\yi.py`

```py
import os  # 导入标准库 os，用于操作操作系统相关功能
from typing import TYPE_CHECKING, Any, Dict, Optional, Union  # 导入类型提示相关的模块

from dbgpt.core import ModelRequest, ModelRequestContext  # 导入自定义模块
from dbgpt.model.proxy.llms.proxy_model import ProxyModel  # 导入自定义模块

from .chatgpt import OpenAILLMClient  # 从当前包中导入模块

if TYPE_CHECKING:
    from httpx._types import ProxiesTypes  # 类型检查，导入 httpx._types 中的 ProxiesTypes 类型
    from openai import AsyncAzureOpenAI, AsyncOpenAI  # 类型检查，导入 openai 中的 AsyncAzureOpenAI 和 AsyncOpenAI 类型

    ClientType = Union[AsyncAzureOpenAI, AsyncOpenAI]  # 定义 ClientType 类型别名，可以是 AsyncAzureOpenAI 或 AsyncOpenAI 的联合类型

_YI_DEFAULT_MODEL = "yi-34b-chat-0205"  # 定义默认模型名称

async def yi_generate_stream(  # 定义异步函数 yi_generate_stream，生成一个异步生成器
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    client: YiLLMClient = model.proxy_llm_client  # 获取 model 对象的 proxy_llm_client 属性作为 YiLLMClient 客户端
    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))  # 创建 ModelRequestContext 上下文对象
    request = ModelRequest.build_request(  # 构建 ModelRequest 请求对象
        client.default_model,  # 使用 client 的默认模型
        messages=params["messages"],  # 请求参数中的消息内容
        temperature=params.get("temperature"),  # 请求参数中的温度参数（可选）
        context=context,  # 使用上面创建的上下文对象
        max_new_tokens=params.get("max_new_tokens"),  # 请求参数中的最大新 token 数（可选）
    )
    async for r in client.generate_stream(request):  # 异步迭代 client 生成的流数据
        yield r  # 生成每一个异步结果

class YiLLMClient(OpenAILLMClient):  # 定义 YiLLMClient 类，继承自 OpenAILLMClient 类
    """Yi LLM Client.

    Yi' API is compatible with OpenAI's API, so we inherit from OpenAILLMClient.
    """

    def __init__(  # 构造函数，初始化 YiLLMClient 对象
        self,
        api_key: Optional[str] = None,  # API 密钥，可选参数，默认为 None
        api_base: Optional[str] = None,  # API 基础地址，可选参数，默认为 None
        api_type: Optional[str] = None,  # API 类型，可选参数，默认为 None
        api_version: Optional[str] = None,  # API 版本，可选参数，默认为 None
        model: Optional[str] = _YI_DEFAULT_MODEL,  # 模型名称，可选参数，默认为 _YI_DEFAULT_MODEL 定义的默认模型名称
        proxies: Optional["ProxiesTypes"] = None,  # 代理类型，可选参数，默认为 None
        timeout: Optional[int] = 240,  # 超时时间，可选参数，默认为 240 秒
        model_alias: Optional[str] = "yi_proxyllm",  # 模型别名，可选参数，默认为 "yi_proxyllm"
        context_length: Optional[int] = None,  # 上下文长度，可选参数，默认为 None
        openai_client: Optional["ClientType"] = None,  # OpenAI 客户端类型，可选参数，默认为 None
        openai_kwargs: Optional[Dict[str, Any]] = None,  # OpenAI 客户端的额外参数，可选参数，默认为 None
        **kwargs
    ):
        api_base = (  # 设置 API 基础地址，优先使用参数中的 api_base，其次是环境变量 YI_API_BASE，最后默认为 "https://api.lingyiwanwu.com/v1"
            api_base or os.getenv("YI_API_BASE") or "https://api.lingyiwanwu.com/v1"
        )
        api_key = api_key or os.getenv("YI_API_KEY")  # 获取 API 密钥，优先使用参数中的 api_key，其次是环境变量 YI_API_KEY
        model = model or _YI_DEFAULT_MODEL  # 设置模型名称，优先使用参数中的 model，其次是默认的 _YI_DEFAULT_MODEL

        if not context_length:  # 如果未指定上下文长度
            if "200k" in model:  # 如果模型名称包含 "200k"
                context_length = 200 * 1024  # 设置上下文长度为 200KB
            else:
                context_length = 4096  # 否则设置上下文长度为 4096

        if not api_key:  # 如果未提供 API 密钥
            raise ValueError(  # 抛出值错误异常
                "Yi API key is required, please set 'YI_API_KEY' in environment "
                "variable or pass it to the client."
            )
        
        super().__init__(  # 调用父类构造函数进行初始化
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
            **kwargs
        )

    @property
    def default_model(self) -> str:  # 定义属性 default_model，返回当前实例的模型名称
        model = self._model  # 获取实例的 _model 属性
        if not model:  # 如果 _model 为空
            model = _YI_DEFAULT_MODEL  # 使用默认的 _YI_DEFAULT_MODEL
        return model  # 返回模型名称
```