# `.\DB-GPT-src\dbgpt\model\proxy\llms\chatgpt.py`

```py
from __future__ import annotations
# 引入 Python 未来版本的 annotations 特性

import importlib.metadata as metadata
# 导入 importlib.metadata 库并重命名为 metadata

import logging
# 导入 logging 库

from concurrent.futures import Executor
# 从 concurrent.futures 中导入 Executor 类

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Union
# 导入 typing 库中的类型标注，包括 TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Union

from dbgpt.core import (
    MessageConverter,
    ModelMetadata,
    ModelOutput,
    ModelRequest,
    ModelRequestContext,
)
# 从 dbgpt.core 模块导入 MessageConverter, ModelMetadata, ModelOutput, ModelRequest, ModelRequestContext 类

from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource
# 从 dbgpt.core.awel.flow 模块导入 Parameter, ResourceCategory, register_resource

from dbgpt.model.parameter import ProxyModelParameters
# 从 dbgpt.model.parameter 模块导入 ProxyModelParameters 类

from dbgpt.model.proxy.base import ProxyLLMClient
# 从 dbgpt.model.proxy.base 模块导入 ProxyLLMClient 类

from dbgpt.model.proxy.llms.proxy_model import ProxyModel
# 从 dbgpt.model.proxy.llms.proxy_model 模块导入 ProxyModel 类

from dbgpt.model.utils.chatgpt_utils import OpenAIParameters
# 从 dbgpt.model.utils.chatgpt_utils 模块导入 OpenAIParameters 类

from dbgpt.util.i18n_utils import _
# 从 dbgpt.util.i18n_utils 模块导入 _ 函数（用于国际化字符串）

if TYPE_CHECKING:
    from httpx._types import ProxiesTypes
    # 如果是类型检查，从 httpx._types 导入 ProxiesTypes 类型

    from openai import AsyncAzureOpenAI, AsyncOpenAI
    # 如果是类型检查，从 openai 模块导入 AsyncAzureOpenAI, AsyncOpenAI 类型

    ClientType = Union[AsyncAzureOpenAI, AsyncOpenAI]
    # 如果是类型检查，定义 ClientType 类型别名为 AsyncAzureOpenAI 或 AsyncOpenAI

logger = logging.getLogger(__name__)
# 获取当前模块的 logger 对象


async def chatgpt_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    # 异步生成聊天内容的生成器函数，接受 ProxyModel 对象、tokenizer、params 字典、device 参数和可选的 context_len

    client: OpenAILLMClient = model.proxy_llm_client
    # 从传入的 model 参数获取 proxy_llm_client 属性，类型标注为 OpenAILLMClient

    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))
    # 创建 ModelRequestContext 对象，指定 stream=True 表示数据流式处理，从 params 参数获取用户名称

    request = ModelRequest.build_request(
        client.default_model,
        messages=params["messages"],
        temperature=params.get("temperature"),
        context=context,
        max_new_tokens=params.get("max_new_tokens"),
    )
    # 使用 client 的默认模型构建 ModelRequest 请求对象，包括 messages、temperature、context 和 max_new_tokens 参数

    async for r in client.generate_stream(request):
        yield r
    # 使用 client 的生成器方法 generate_stream 处理请求并异步迭代生成结果


@register_resource(
    label=_("OpenAI LLM Client"),
    name="openai_llm_client",
    category=ResourceCategory.LLM_CLIENT,
    parameters=[
        Parameter.build_from(
            label=_("OpenAI API Key"),
            name="apk_key",
            type=str,
            optional=True,
            default=None,
            description=_(
                "OpenAI API Key, not required if you have set OPENAI_API_KEY "
                "environment variable."
            ),
        ),
        Parameter.build_from(
            label=_("OpenAI API Base"),
            name="api_base",
            type=str,
            optional=True,
            default=None,
            description=_(
                "OpenAI API Base, not required if you have set OPENAI_API_BASE "
                "environment variable."
            ),
        ),
    ],
    documentation_url="https://github.com/openai/openai-python",
)
class OpenAILLMClient(ProxyLLMClient):
    # 定义 OpenAILLMClient 类，继承自 ProxyLLMClient 类

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
        model: Optional[str] = None,
        proxies: Optional["ProxiesTypes"] = None,
        timeout: Optional[int] = 240,
        model_alias: Optional[str] = "chatgpt_proxyllm",
        context_length: Optional[int] = 8192,
        openai_client: Optional["ClientType"] = None,
        openai_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # OpenAILLMClient 的初始化方法，接受多个可选参数和关键字参数
    ):
        # 尝试导入 openai 包，如果导入失败则抛出异常
        try:
            import openai
        except ImportError as exc:
            # 如果导入失败，抛出 ValueError 异常，提供安装 openai 包的提示信息
            raise ValueError(
                "Could not import python package: openai "
                "Please install openai by command `pip install openai"
            ) from exc
        
        # 获取当前安装的 openai 包的版本
        self._openai_version = metadata.version("openai")
        
        # 检查当前 openai 版本是否小于 1.0.0，返回布尔值
        self._openai_less_then_v1 = not self._openai_version >= "1.0.0"
        
        # 检查 SDK 版本是否符合要求
        self.check_sdk_version(self._openai_version)
        
        # 初始化 OpenAIParameters 对象，保存 API 相关参数
        self._init_params = OpenAIParameters(
            api_type=api_type,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            proxies=proxies,
            full_url=kwargs.get("full_url"),
        )
        
        # 保存模型名称
        self._model = model
        
        # 保存代理参数
        self._proxies = proxies
        
        # 设置超时时间
        self._timeout = timeout
        
        # 保存模型别名
        self._model_alias = model_alias
        
        # 设置上下文长度
        self._context_length = context_length
        
        # 保存 API 类型
        self._api_type = api_type
        
        # 保存 openai_client 对象
        self._client = openai_client
        
        # 设置 openai_kwargs 或使用空字典
        self._openai_kwargs = openai_kwargs or {}
        
        # 调用父类的构造函数，初始化模型名称和上下文长度
        super().__init__(model_names=[model_alias], context_length=context_length)
        
        # 如果当前 openai 版本小于 1.0.0，则执行初始化 openai 的特定函数
        if self._openai_less_then_v1:
            from dbgpt.model.utils.chatgpt_utils import _initialize_openai

            _initialize_openai(self._init_params)

    @classmethod
    def new_client(
        cls,
        model_params: ProxyModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "OpenAILLMClient":
        # 创建并返回一个新的 OpenAILLMClient 实例，使用给定的模型参数
        return cls(
            api_key=model_params.proxy_api_key,
            api_base=model_params.proxy_api_base,
            api_type=model_params.proxy_api_type,
            api_version=model_params.proxy_api_version,
            model=model_params.proxyllm_backend,
            proxies=model_params.http_proxy,
            model_alias=model_params.model_name,
            context_length=max(model_params.max_context_size, 8192),
            full_url=model_params.proxy_server_url,
        )

    def check_sdk_version(self, version: str) -> None:
        """Check the sdk version of the client.

        Raises:
            ValueError: If check failed.
        """
        # 检查传入的版本号是否符合要求，否则抛出 ValueError 异常
        pass

    @property
    def client(self) -> ClientType:
        # 获取或构建并返回当前的 openai_client 对象
        if self._openai_less_then_v1:
            raise ValueError(
                "Current model (Load by OpenAILLMClient) require openai.__version__>=1.0.0"
            )
        if self._client is None:
            from dbgpt.model.utils.chatgpt_utils import _build_openai_client

            # 如果 _client 为 None，则构建并保存一个新的 openai_client 对象
            self._api_type, self._client = _build_openai_client(
                init_params=self._init_params
            )
        return self._client

    @property
    def default_model(self) -> str:
        # 获取默认模型名称，如果未设置则根据 api_type 返回默认模型名称
        model = self._model
        if not model:
            model = "gpt-35-turbo" if self._api_type == "azure" else "gpt-3.5-turbo"
        return model

    def _build_request(
        self, request: ModelRequest, stream: Optional[bool] = False
        # 构建一个请求对象，使用给定的 ModelRequest 和 stream 参数
        ) -> Dict[str, Any]:
        # 构建请求的 payload 字典，包含流信息
        payload = {"stream": stream}
        # 根据请求中的 model 参数选择模型，若未指定则使用默认模型
        model = request.model or self.default_model
        # 如果 API 版本低于 v1，并且使用的是 Azure API，则设置 engine 参数
        if self._openai_less_then_v1 and self._api_type == "azure":
            payload["engine"] = model
        else:
            # 否则设置 model 参数
            payload["model"] = model
        # 应用额外的 openai kwargs 到 payload
        for k, v in self._openai_kwargs.items():
            payload[k] = v
        # 如果请求中包含 temperature 参数，则添加到 payload 中
        if request.temperature:
            payload["temperature"] = request.temperature
        # 如果请求中包含 max_new_tokens 参数，则添加到 payload 中
        if request.max_new_tokens:
            payload["max_tokens"] = request.max_new_tokens
        # 返回构建好的 payload 字典
        return payload

    async def generate(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelOutput:
        # 将请求消息转换为通用消息格式
        request = self.local_covert_message(request, message_converter)
        # 将请求转换为多条消息
        messages = request.to_common_messages()
        # 构建生成请求的 payload
        payload = self._build_request(request)
        # 记录日志，包括发送到 OpenAI 的请求版本和 payload 数据，以及消息内容
        logger.info(
            f"Send request to openai({self._openai_version}), payload: {payload}\n\n messages:\n{messages}"
        )
        try:
            # 根据 API 版本选择调用不同的生成方法
            if self._openai_less_then_v1:
                return await self.generate_less_then_v1(messages, payload)
            else:
                return await self.generate_v1(messages, payload)
        except Exception as e:
            # 如果发生异常，返回带有错误信息的 ModelOutput
            return ModelOutput(
                text=f"**LLMServer Generate Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )

    async def generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> AsyncIterator[ModelOutput]:
        # 将请求消息转换为通用消息格式
        request = self.local_covert_message(request, message_converter)
        # 将请求转换为多条消息
        messages = request.to_common_messages()
        # 构建流式生成请求的 payload
        payload = self._build_request(request, stream=True)
        # 记录日志，包括发送到 OpenAI 的请求版本和 payload 数据，以及消息内容
        logger.info(
            f"Send request to openai({self._openai_version}), payload: {payload}\n\n messages:\n{messages}"
        )
        # 根据 API 版本选择调用不同的流式生成方法
        if self._openai_less_then_v1:
            async for r in self.generate_stream_less_then_v1(messages, payload):
                yield r
        else:
            async for r in self.generate_stream_v1(messages, payload):
                yield r

    async def generate_v1(
        self, messages: List[Dict[str, Any]], payload: Dict[str, Any]
    ) -> ModelOutput:
        # 调用 OpenAI API 客户端的聊天生成接口，获取聊天完成的结果
        chat_completion = await self.client.chat.completions.create(
            messages=messages, **payload
        )
        # 从生成结果中获取第一条选择的消息文本
        text = chat_completion.choices[0].message.content
        # 获取生成的用法信息
        usage = chat_completion.usage.dict()
        # 返回 ModelOutput 对象，包括生成的文本、错误代码和用法信息
        return ModelOutput(text=text, error_code=0, usage=usage)

    async def generate_less_then_v1(
        self, messages: List[Dict[str, Any]], payload: Dict[str, Any]
    ) -> ModelOutput:
        # 在 OpenAI API 版本低于 v1 时调用的生成方法，实现类似 v1 的生成流程
        # 具体实现可能因 API 版本的不同而有所调整或限制
        # 实现方式应符合兼容低版本 API 的要求
        # 该方法的具体逻辑需根据实际 API 的响应和功能进行调整和扩展
        pass
    async def generate_stream_v1(
        self, messages: List[Dict[str, Any]], payload: Dict[str, Any]
    ) -> AsyncIterator[ModelOutput]:
        # 使用OpenAI的client进行聊天完成请求创建
        chat_completion = await self.client.chat.completions.create(
            messages=messages, **payload
        )
        text = ""
        # 异步迭代聊天完成对象
        async for r in chat_completion:
            if len(r.choices) == 0:
                continue
            if r.choices[0].delta.content is not None:
                content = r.choices[0].delta.content
                text += content
                yield ModelOutput(text=text, error_code=0)

    async def generate_stream_less_then_v1(
        self, messages: List[Dict[str, Any]], payload: Dict[str, Any]
    ) -> AsyncIterator[ModelOutput]:
        import openai
        
        # 使用OpenAI的ChatCompletion类进行聊天完成请求创建
        res = await openai.ChatCompletion.acreate(messages=messages, **payload)
        text = ""
        # 异步迭代聊天完成结果
        async for r in res:
            if not r.get("choices"):
                continue
            if r["choices"][0]["delta"].get("content") is not None:
                content = r["choices"][0]["delta"]["content"]
                text += content
                yield ModelOutput(text=text, error_code=0)

    async def models(self) -> List[ModelMetadata]:
        # 创建模型元数据对象
        model_metadata = ModelMetadata(
            model=self._model_alias,
            context_length=await self.get_context_length(),
        )
        return [model_metadata]

    async def get_context_length(self) -> int:
        """获取模型的上下文长度。

        Returns:
            int: 上下文长度。
        # TODO: 这是一个临时解决方案。我们应该有更好的方法从OpenAI API获取上下文长度。
            例如，从OpenAI API真实获取上下文长度。
        """
        return self._context_length
```