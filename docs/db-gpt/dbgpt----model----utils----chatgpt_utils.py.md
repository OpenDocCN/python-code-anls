# `.\DB-GPT-src\dbgpt\model\utils\chatgpt_utils.py`

```py
from __future__ import annotations

import importlib.metadata as metadata  # 导入 importlib.metadata 库，用于获取包的元数据信息
import logging  # 导入 logging 模块，用于记录日志信息
import os  # 导入 os 模块，提供操作系统相关的功能
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于定义数据类
from typing import (  # 导入 typing 模块中的类型提示工具
    TYPE_CHECKING,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
    Tuple,
    Union,
)

from dbgpt._private.pydantic import model_to_json  # 导入自定义模块 dbgpt._private.pydantic 中的 model_to_json 函数
from dbgpt.core.awel import TransformStreamAbsOperator  # 导入 dbgpt.core.awel 模块中的 TransformStreamAbsOperator 类
from dbgpt.core.awel.flow import (  # 导入 dbgpt.core.awel.flow 模块中的多个类和函数
    IOField,
    OperatorCategory,
    OperatorType,
    ViewMetadata,
)
from dbgpt.core.interface.llm import ModelOutput  # 导入 dbgpt.core.interface.llm 模块中的 ModelOutput 类
from dbgpt.core.operators import BaseLLM  # 导入 dbgpt.core.operators 模块中的 BaseLLM 类
from dbgpt.util.i18n_utils import _  # 导入自定义模块 dbgpt.util.i18n_utils 中的 _ 函数

if TYPE_CHECKING:
    import httpx  # 如果是类型检查，导入 httpx 库，用于 HTTP 客户端功能
    from httpx._types import ProxiesTypes  # 导入 httpx 库中的 ProxiesTypes 类型
    from openai import AsyncAzureOpenAI, AsyncOpenAI  # 导入 openai 库中的 AsyncAzureOpenAI 和 AsyncOpenAI 类型

    ClientType = Union[AsyncAzureOpenAI, AsyncOpenAI]  # 定义 ClientType 类型别名，可以是 AsyncAzureOpenAI 或 AsyncOpenAI

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


@dataclass
class OpenAIParameters:
    """A class to represent a LLM model."""

    api_type: str = "open_ai"  # LLM 模型的 API 类型，默认为 "open_ai"
    api_base: Optional[str] = None  # LLM 模型的 API 基础地址，可选参数，默认为 None
    api_key: Optional[str] = None  # LLM 模型的 API 密钥，可选参数，默认为 None
    api_version: Optional[str] = None  # LLM 模型的 API 版本，可选参数，默认为 None
    api_azure_deployment: Optional[str] = None  # LLM 模型的 Azure 部署标识，可选参数，默认为 None
    full_url: Optional[str] = None  # LLM 模型的完整 URL 地址，可选参数，默认为 None
    proxies: Optional["ProxiesTypes"] = None  # LLM 模型的代理设置，可选参数，默认为 None


def _initialize_openai_v1(init_params: OpenAIParameters):
    try:
        from openai import OpenAI  # 尝试导入 openai 库中的 OpenAI 类
    except ImportError as exc:
        raise ValueError(
            "Could not import python package: openai "
            "Please install openai by command `pip install openai"
        ) from exc  # 捕获 ImportError 异常，提醒用户安装 openai 库

    if not metadata.version("openai") >= "1.0.0":  # 检查 openai 库的版本是否大于等于 "1.0.0"
        raise ImportError("Please upgrade openai package to version 1.0.0 or above")  # 若版本低于 "1.0.0"，则抛出 ImportError

    api_type: Optional[str] = init_params.api_type  # 初始化 API 类型
    api_base: Optional[str] = init_params.api_base  # 初始化 API 基础地址
    api_key: Optional[str] = init_params.api_key  # 初始化 API 密钥
    api_version: Optional[str] = init_params.api_version  # 初始化 API 版本
    full_url: Optional[str] = init_params.full_url  # 初始化完整 URL 地址

    api_type = api_type or os.getenv("OPENAI_API_TYPE", "open_ai")  # 获取 API 类型，优先使用参数中的，然后环境变量，最后默认为 "open_ai"

    base_url = api_base or os.getenv(  # 获取 API 基础地址，优先使用参数中的，然后环境变量
        "OPENAI_API_BASE",
        os.getenv("AZURE_OPENAI_ENDPOINT") if api_type == "azure" else None,
    )
    api_key = api_key or os.getenv(  # 获取 API 密钥，优先使用参数中的，然后环境变量
        "OPENAI_API_KEY",
        os.getenv("AZURE_OPENAI_KEY") if api_type == "azure" else None,
    )
    api_version = api_version or os.getenv("OPENAI_API_VERSION")  # 获取 API 版本，优先使用参数中的，然后环境变量

    api_azure_deployment = init_params.api_azure_deployment or os.getenv(  # 获取 Azure 部署标识，优先使用参数中的，然后环境变量
        "API_AZURE_DEPLOYMENT"
    )
    if not base_url and full_url:
        base_url = full_url.split("/chat/completions")[0]  # 如果缺少 base_url 但提供了 full_url，则从 full_url 中截取 base_url

    if api_key is None:
        raise ValueError("api_key is required, please set OPENAI_API_KEY environment")  # 如果缺少 api_key，则抛出 ValueError
    if base_url is None:
        raise ValueError("base_url is required, please set OPENAI_BASE_URL environment")  # 如果缺少 base_url，则抛出 ValueError
    if base_url.endswith("/"):
        base_url = base_url[:-1]  # 如果 base_url 以 '/' 结尾，则去除最后的 '/'

    openai_params = {"api_key": api_key, "base_url": base_url}  # 构建包含 API 密钥和基础地址的参数字典
    return openai_params, api_type, api_version, api_azure_deployment  # 返回初始化后的参数信息


def _initialize_openai(params: OpenAIParameters):
    try:
        import openai  # 尝试导入 openai 库

        import openai  # 尝试导入 openai 库
    except ImportError as exc:
        raise ValueError(
            "Could not import python package: openai "
            "Please install openai by command `pip install openai"
        ) from exc  # 捕获 ImportError 异常，提醒用户安装 openai 库
    # 如果导入错误，将抛出 ValueError 异常，提示用户安装 openai 包
    except ImportError as exc:
        raise ValueError(
            "Could not import python package: openai "
            "Please install openai by command `pip install openai` "
        ) from exc

    # 设置 API 类型为参数中的 api_type 或者从环境变量 OPENAI_API_TYPE 获取，默认为 'open_ai'
    api_type = params.api_type or os.getenv("OPENAI_API_TYPE", "open_ai")

    # 设置 API 基础地址为参数中的 api_base 或者从环境变量 OPENAI_API_TYPE 获取
    # 如果 api_type 是 'azure'，则尝试从环境变量 AZURE_OPENAI_ENDPOINT 获取
    api_base = params.api_base or os.getenv(
        "OPENAI_API_TYPE",
        os.getenv("AZURE_OPENAI_ENDPOINT") if api_type == "azure" else None,
    )

    # 设置 API 密钥为参数中的 api_key 或者从环境变量 OPENAI_API_KEY 获取
    # 如果 api_type 是 'azure'，则尝试从环境变量 AZURE_OPENAI_KEY 获取
    api_key = params.api_key or os.getenv(
        "OPENAI_API_KEY",
        os.getenv("AZURE_OPENAI_KEY") if api_type == "azure" else None,
    )

    # 设置 API 版本为参数中的 api_version 或者从环境变量 OPENAI_API_VERSION 获取
    api_version = params.api_version or os.getenv("OPENAI_API_VERSION")

    # 如果没有设置 api_base 并且 params.full_url 为真，则从 params.full_url 推断出 api_base
    if not api_base and params.full_url:
        # 根据 full_url 配置适应以前的 proxy_server_url
        api_base = params.full_url.split("/chat/completions")[0]

    # 如果指定了 api_type，则将 openai.api_type 设置为指定的值
    if api_type:
        openai.api_type = api_type

    # 如果指定了 api_base，则将 openai.api_base 设置为指定的值
    if api_base:
        openai.api_base = api_base

    # 如果指定了 api_key，则将 openai.api_key 设置为指定的值
    if api_key:
        openai.api_key = api_key

    # 如果指定了 api_version，则将 openai.api_version 设置为指定的值
    if api_version:
        openai.api_version = api_version

    # 如果指定了 proxies，则将 openai.proxy 设置为参数中的 proxies
    if params.proxies:
        openai.proxy = params.proxies
# 定义一个函数 _build_openai_client，接受一个名为 init_params 的 OpenAIParameters 类型参数，并返回一个元组
# 包含字符串和 ClientType 类型的对象
def _build_openai_client(init_params: OpenAIParameters) -> Tuple[str, ClientType]:
    # 导入 httpx 模块，用于异步 HTTP 客户端操作
    import httpx

    # 调用 _initialize_openai_v1 函数，初始化 OpenAI 的相关参数
    openai_params, api_type, api_version, api_azure_deployment = _initialize_openai_v1(
        init_params
    )
    
    # 根据 API 类型决定使用哪个 OpenAI 的客户端类
    if api_type == "azure":
        # 如果是 Azure 类型，导入 AsyncAzureOpenAI 类并返回对应的实例化对象
        from openai import AsyncAzureOpenAI
        return api_type, AsyncAzureOpenAI(
            api_key=openai_params["api_key"],
            api_version=api_version,
            azure_deployment=api_azure_deployment,
            azure_endpoint=openai_params["base_url"],
            http_client=httpx.AsyncClient(proxies=init_params.proxies),
        )
    else:
        # 否则，导入 AsyncOpenAI 类并返回对应的实例化对象
        from openai import AsyncOpenAI
        return api_type, AsyncOpenAI(
            **openai_params, http_client=httpx.AsyncClient(proxies=init_params.proxies)
        )


# 定义一个名为 OpenAIStreamingOutputOperator 的类，继承自 TransformStreamAbsOperator[ModelOutput, str]
class OpenAIStreamingOutputOperator(TransformStreamAbsOperator[ModelOutput, str]):
    """Transform ModelOutput to openai stream format."""

    # 声明类属性 incremental_output 为 True，表示输出是增量的
    incremental_output = True
    # 声明类属性 output_format 为 "SSE"，表示输出格式为 Server-Sent Events
    output_format = "SSE"

    # 声明类属性 metadata 为 ViewMetadata 对象，包含了该操作符的元数据信息
    metadata = ViewMetadata(
        label=_("OpenAI Streaming Output Operator"),
        name="openai_streaming_output_operator",
        operator_type=OperatorType.TRANSFORM_STREAM,
        category=OperatorCategory.OUTPUT_PARSER,
        description=_("The OpenAI streaming LLM operator."),
        parameters=[],
        inputs=[
            IOField.build_from(
                _("Upstream Model Output"),
                "model_output",
                ModelOutput,
                is_list=True,
                description=_("The model output of upstream."),
            )
        ],
        outputs=[
            IOField.build_from(
                _("Model Output"),
                "model_output",
                str,
                is_list=True,
                description=_(
                    "The model output after transformed to openai stream format."
                ),
            )
        ],
    )

    # 声明异步方法 transform_stream，接受 AsyncIterator[ModelOutput] 类型的参数 model_output
    async def transform_stream(self, model_output: AsyncIterator[ModelOutput]):
        # 声明内部异步函数 model_caller，返回值类型为 str
        async def model_caller() -> str:
            """Read model name from share data.
            In streaming mode, this transform_stream function will be executed
            before parent operator(Streaming Operator is trigger by downstream Operator).
            """
            # 从当前的 DAG 上下文中获取共享数据中的模型名称
            return await self.current_dag_context.get_from_share_data(
                BaseLLM.SHARE_DATA_KEY_MODEL_NAME
            )

        # 使用 _to_openai_stream 函数将 model_output 转换为 OpenAI 流格式，并迭代生成输出
        async for output in _to_openai_stream(model_output, None, model_caller):
            yield output


# 声明一个异步函数 _to_openai_stream，接受 AsyncIterator[ModelOutput] 类型的参数 output_iter
# 可选参数 model 是字符串类型，model_caller 是一个可调用对象
async def _to_openai_stream(
    output_iter: AsyncIterator[ModelOutput],
    model: Optional[str] = None,
    model_caller: Callable[[], Union[Awaitable[str], str]] = None,
) -> AsyncIterator[str]:
    """Convert the output_iter to openai stream format."""
    # 实现将 output_iter 转换为 OpenAI 流格式的功能
    Args:
        output_iter (AsyncIterator[ModelOutput]): 异步迭代器，产生模型输出结果。
        model (Optional[str], optional): 模型名称。默认为 None。
        model_caller (Callable[[None], Union[Awaitable[str], str]], optional): 模型调用器。默认为 None。
    """
    import asyncio  # 导入 asyncio 模块，用于异步编程
    import json  # 导入 json 模块，用于处理 JSON 数据

    import shortuuid  # 导入 shortuuid 模块，用于生成短的唯一标识符

    from dbgpt.core.schema.api import (  # 导入特定模块中的 API 类和对象
        ChatCompletionResponseStreamChoice,  # 聊天完成响应流中的选择
        ChatCompletionStreamResponse,  # 聊天完成流响应
        DeltaMessage,  # 变化消息
    )

    id = f"chatcmpl-{shortuuid.random()}"  # 创建一个唯一的标识符，以字符串形式存储

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,  # 设置选择数据的索引为 0
        delta=DeltaMessage(role="assistant"),  # 设置变化消息的角色为“assistant”
        finish_reason=None,  # 设置完成原因为空
    )
    chunk = ChatCompletionStreamResponse(
        id=id,  # 设置响应块的 ID
        choices=[choice_data],  # 将选择数据添加到响应块中
        model=model or "",  # 设置模型名称，如果未提供则为空字符串
    )
    yield f"data: {model_to_json(chunk, exclude_unset=True, ensure_ascii=False)}\n\n"
    # 生成器函数，返回 JSON 格式化后的响应块数据

    previous_text = ""  # 初始化前一个文本为空字符串
    finish_stream_events = []  # 初始化完成流事件列表为空
    async for model_output in output_iter:  # 异步迭代模型输出结果
        if model_caller is not None:  # 如果模型调用器不为空
            if asyncio.iscoroutinefunction(model_caller):  # 如果模型调用器是协程函数
                model = await model_caller()  # 调用并等待模型调用器的结果
            else:
                model = model_caller()  # 否则直接调用模型调用器
        model_output: ModelOutput = model_output  # 将模型输出结果转换为 ModelOutput 类型
        if model_output.error_code != 0:  # 如果模型输出结果的错误代码不为 0
            yield f"data: {json.dumps(model_output.to_dict(), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"  # 生成器函数，返回错误信息和完成标记
            return  # 结束生成器函数的执行
        decoded_unicode = model_output.text.replace("\ufffd", "")  # 解码模型输出文本
        delta_text = decoded_unicode[len(previous_text):]  # 计算变化文本部分
        previous_text = (  # 更新前一个文本为较长的文本
            decoded_unicode
            if len(decoded_unicode) > len(previous_text)
            else previous_text
        )

        if len(delta_text) == 0:  # 如果变化文本长度为 0
            delta_text = None  # 将变化文本设置为 None
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,  # 设置选择数据的索引为 0
            delta=DeltaMessage(content=delta_text),  # 设置变化消息的内容为 delta_text
            finish_reason=model_output.finish_reason,  # 设置完成原因为模型输出的完成原因
        )
        chunk = ChatCompletionStreamResponse(id=id, choices=[choice_data], model=model)
        # 创建聊天完成流响应块

        if delta_text is None:  # 如果变化文本为 None
            if model_output.finish_reason is not None:  # 如果完成原因不为空
                finish_stream_events.append(chunk)  # 将完成块添加到完成流事件列表
            continue  # 继续下一次迭代

        yield f"data: {model_to_json(chunk, exclude_unset=True, ensure_ascii=False)}\n\n"
        # 生成器函数，返回 JSON 格式化后的响应块数据

    for finish_chunk in finish_stream_events:  # 遍历完成流事件列表
        yield f"data: {model_to_json(finish_chunk, exclude_none=True, ensure_ascii=False)}\n\n"
        # 生成器函数，返回 JSON 格式化后的完成块数据
    yield "data: [DONE]\n\n"  # 生成器函数，返回完成标记
```