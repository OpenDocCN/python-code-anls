# `.\DB-GPT-src\dbgpt\model\cluster\apiserver\api.py`

```py
# 创建一个用于提供 OpenAI 兼容的 RESTful API 的服务器。支持以下功能：
# - 聊天完成 (Chat Completions)。参考文档：https://platform.openai.com/docs/api-reference/chat
import asyncio  # 异步IO库
import json  # 处理JSON数据的库
import logging  # 日志记录库
from typing import Any, Dict, Generator, List, Optional  # 引入类型提示相关的库

import shortuuid  # 用于生成短UUID的库
from fastapi import APIRouter, Depends, HTTPException  # FastAPI相关组件：APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError  # FastAPI异常处理：RequestValidationError
from fastapi.middleware.cors import CORSMiddleware  # FastAPI中间件：CORSMiddleware，用于处理跨域请求
from fastapi.responses import JSONResponse, StreamingResponse  # FastAPI响应相关：JSONResponse, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer  # FastAPI安全认证相关：HTTPAuthorizationCredentials, HTTPBearer

# dbgpt库相关的模块和类的引入
from dbgpt._private.pydantic import BaseModel, model_to_dict, model_to_json
from dbgpt.component import BaseComponent, ComponentType, SystemApp
from dbgpt.core import ModelOutput
from dbgpt.core.interface.message import ModelMessage
from dbgpt.core.schema.api import (
    APIChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorCode,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    RelevanceRequest,
    RelevanceResponse,
    UsageInfo,
)
from dbgpt.model.base import ModelInstance  # dbgpt中模型实例基类的引入
from dbgpt.model.cluster.manager_base import WorkerManager, WorkerManagerFactory  # dbgpt中工作管理相关类的引入
from dbgpt.model.cluster.registry import ModelRegistry  # dbgpt中模型注册表的引入
from dbgpt.model.parameter import ModelAPIServerParameters, WorkerType  # dbgpt中模型API服务器参数和工作类型的引入
from dbgpt.util.fastapi import create_app  # dbgpt中用于创建FastAPI应用的函数的引入
from dbgpt.util.parameter_utils import EnvArgumentParser  # dbgpt中环境参数解析的工具函数的引入
from dbgpt.util.tracer import initialize_tracer, root_tracer  # dbgpt中用于追踪的初始化和根追踪器的引入
from dbgpt.util.utils import setup_logging  # dbgpt中用于设置日志的工具函数的引入

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class APIServerException(Exception):
    """API服务器异常类，继承自Python内置的Exception类。用于表示API服务器发生的异常。

    Args:
        code (int): 异常代码。
        message (str): 异常消息。
    """
    def __init__(self, code: int, message: str):
        self.code = code  # 异常代码
        self.message = message  # 异常消息


class APISettings(BaseModel):
    """API服务器的设置模型，继承自BaseModel。

    Attributes:
        api_keys (Optional[List[str]]): API密钥列表，默认为None。
        embedding_bach_size (int): 嵌入批处理大小，默认为4。
    """
    api_keys: Optional[List[str]] = None  # API密钥列表，默认为None
    embedding_bach_size: int = 4  # 嵌入批处理大小，默认为4


api_settings = APISettings()  # 创建API服务器的设置实例
get_bearer_token = HTTPBearer(auto_error=False)  # 创建一个获取HTTP Bearer Token的实例，自动处理错误情况


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    """检查API密钥的异步函数。

    Args:
        auth (Optional[HTTPAuthorizationCredentials], optional): HTTP授权凭据对象。默认为从get_bearer_token获取。

    Raises:
        HTTPException: 如果API密钥验证失败，抛出401 Unauthorized异常。

    Returns:
        str: 验证通过的API密钥。
    """
    if api_settings.api_keys:  # 如果API密钥列表不为空
        if auth is None or (token := auth.credentials) not in api_settings.api_keys:
            # 如果未提供认证信息或提供的token不在API密钥列表中，则抛出401 Unauthorized异常
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token  # 返回通过验证的API密钥
    else:
        # 如果API密钥列表为空，允许所有请求
        return None


def create_error_response(code: int, message: str) -> JSONResponse:
    """创建错误响应的函数。

    Args:
        code (int): 错误代码。
        message (str): 错误消息。

    Returns:
        JSONResponse: 包含错误信息的JSON响应对象。
    """
    """Copy from fastchat.serve.openai_api_server.check_requests
    从fastchat.serve.openai_api_server.check_requests复制过来的错误响应创建函数。
    """
    return JSONResponse(
        status_code=code,
        content={"error": {"code": code, "message": message}},
    )  # 返回带有指定状态码和错误消息的JSON响应对象
    # 由于 fastchat.serve.openai_api_server 依赖过多，我们无法使用它。
    """
    返回一个 JSONResponse 对象，其中包含一个 ErrorResponse 对象的字典表示。
    ErrorResponse 对象包含给定消息和错误代码。
    状态码设置为 400 表示请求错误。
    """
    return JSONResponse(
        model_to_dict(ErrorResponse(message=message, code=code)), status_code=400
    )
def check_requests(request) -> Optional[JSONResponse]:
    """Copy from fastchat.serve.openai_api_server.create_error_response

    We can't use fastchat.serve.openai_api_server because it has too many dependencies.
    """
    # 检查所有参数

    # 检查 max_tokens 参数是否存在且大于零
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )

    # 检查 n 参数是否存在且大于零
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )

    # 检查 temperature 参数是否存在且大于等于零
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )

    # 检查 temperature 参数是否存在且小于等于 2
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )

    # 检查 top_p 参数是否存在且大于等于零
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )

    # 检查 top_p 参数是否存在且小于等于 1
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )

    # 检查 top_k 参数是否存在且在指定范围外
    if request.top_k is not None and (request.top_k > -1 and request.top_k < 1):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_k} is out of Range. Either set top_k to -1 or >=1.",
        )

    # 检查 stop 参数是否存在且不符合预期的字符串或列表类型
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None


class APIServer(BaseComponent):
    name = ComponentType.MODEL_API_SERVER

    def init_app(self, system_app: SystemApp):
        self.system_app = system_app

    def get_worker_manager(self) -> WorkerManager:
        """Get the worker manager component instance

        Raises:
            APIServerException: If can't get worker manager component instance
        """
        # 获取 worker manager 组件实例
        worker_manager = self.system_app.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        
        # 如果未能获取到 worker manager 实例，则抛出异常
        if not worker_manager:
            raise APIServerException(
                ErrorCode.INTERNAL_ERROR,
                f"Could not get component {ComponentType.WORKER_MANAGER_FACTORY} from system_app",
            )
        
        return worker_manager
    # 获取模型注册表组件实例的方法
    def get_model_registry(self) -> ModelRegistry:
        """Get the model registry component instance
        
        Raises:
            APIServerException: 如果无法获取模型注册表组件实例时抛出异常
        """

        # 从系统应用中获取模型注册表组件实例
        controller = self.system_app.get_component(
            ComponentType.MODEL_REGISTRY, ModelRegistry
        )
        # 如果获取失败，抛出 API 服务器异常
        if not controller:
            raise APIServerException(
                ErrorCode.INTERNAL_ERROR,
                f"Could not get component {ComponentType.MODEL_REGISTRY} from system_app",
            )
        return controller

    # 获取特定模型名称的健康模型实例的方法
    async def get_model_instances_or_raise(
        self, model_name: str, worker_type: str = "llm"
    ) -> List[ModelInstance]:
        """Get healthy model instances with request model name
        
        Args:
            model_name (str): 模型名称
            worker_type (str, optional): 工作类型，默认为 "llm"。

        Raises:
            APIServerException: 如果无法获取请求模型名称的健康模型实例时抛出异常
        """
        # 获取模型注册表组件实例
        registry = self.get_model_registry()
        # 拼接模型名称和工作类型后缀
        suffix = f"@{worker_type}"
        registry_model_name = f"{model_name}{suffix}"
        # 获取所有健康的特定模型实例列表
        model_instances = await registry.get_all_instances(
            registry_model_name, healthy_only=True
        )
        # 如果没有找到符合条件的模型实例，处理异常情况
        if not model_instances:
            # 获取所有健康的模型实例列表
            all_instances = await registry.get_all_model_instances(healthy_only=True)
            # 提取所有以指定工作类型后缀结尾的模型名称
            models = [
                ins.model_name.split(suffix)[0]
                for ins in all_instances
                if ins.model_name.endswith(suffix)
            ]
            # 如果找到符合条件的模型，构建错误信息
            if models:
                models = "&&".join(models)
                message = f"Only {models} allowed now, your model {model_name}"
            else:
                message = f"No models allowed now, your model {model_name}"
            # 抛出 API 服务器异常
            raise APIServerException(ErrorCode.INVALID_MODEL, message)
        return model_instances

    # 返回可用模型列表的方法
    async def get_available_models(self) -> ModelList:
        """Return available models
        
        Just include LLM and embedding models.

        Returns:
            List[ModelList]: 模型列表。
        """
        # 获取模型注册表组件实例
        registry = self.get_model_registry()
        # 获取所有健康的模型实例列表
        model_instances = await registry.get_all_model_instances(healthy_only=True)
        model_name_set = set()
        # 遍历所有模型实例，筛选出语言模型和嵌入模型
        for inst in model_instances:
            name, worker_type = WorkerType.parse_worker_key(inst.model_name)
            if worker_type == WorkerType.LLM or worker_type == WorkerType.TEXT2VEC:
                model_name_set.add(name)
        # 将模型名称集合转换为排序后的列表
        models = list(model_name_set)
        models.sort()
        # TODO: 返回真实的模型权限详细信息
        model_cards = []
        # 为每个模型创建模型卡片对象
        for m in models:
            model_cards.append(
                ModelCard(
                    id=m, root=m, owned_by="DB-GPT", permission=[ModelPermission()]
                )
            )
        return ModelList(data=model_cards)
    # 异步生成器函数，用于生成聊天完成流

    # 获取工作管理器实例
    worker_manager = self.get_worker_manager()

    # 生成唯一的流 ID
    id = f"chatcmpl-{shortuuid.random()}"

    # 存储结束流事件的列表
    finish_stream_events = []

    # 循环生成指定数量的完成数据块
    for i in range(n):
        # 创建聊天完成响应流的第一个数据块
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name
        )

        # 将数据块转换为 JSON 格式字符串
        json_data = model_to_json(chunk, exclude_unset=True, ensure_ascii=False)

        # 通过异步生成器返回数据块
        yield f"data: {json_data}\n\n"

        # 初始化前一个文本片段为空字符串
        previous_text = ""

        # 异步迭代工作管理器生成的模型输出流
        async for model_output in worker_manager.generate_stream(params):
            # 确保模型输出是 ModelOutput 类型
            model_output: ModelOutput = model_output

            # 处理模型输出的错误情况
            if model_output.error_code != 0:
                # 将错误信息转换为 JSON 格式并返回
                yield f"data: {json.dumps(model_output.to_dict(), ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 解码模型输出的 Unicode 文本，替换掉无效字符
            decoded_unicode = model_output.text.replace("\ufffd", "")

            # 计算本次输出的增量文本
            delta_text = decoded_unicode[len(previous_text):]

            # 更新前一个文本片段为当前解码后的文本
            previous_text = (
                decoded_unicode
                if len(decoded_unicode) > len(previous_text)
                else previous_text
            )

            # 如果增量文本长度为 0，则设置 delta_text 为 None
            if len(delta_text) == 0:
                delta_text = None

            # 创建聊天完成响应流的数据块
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=delta_text),
                finish_reason=model_output.finish_reason,
            )
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=model_name
            )

            # 如果增量文本为 None，则检查是否有结束原因并添加到结束事件列表
            if delta_text is None:
                if model_output.finish_reason is not None:
                    finish_stream_events.append(chunk)
                continue

            # 将数据块转换为 JSON 格式字符串
            json_data = model_to_json(chunk, exclude_unset=True, ensure_ascii=False)

            # 通过异步生成器返回数据块
            yield f"data: {json_data}\n\n"

    # 遍历结束事件列表中的所有结束事件数据块
    for finish_chunk in finish_stream_events:
        # 将结束事件数据块转换为 JSON 格式字符串
        json_data = model_to_json(
            finish_chunk, exclude_unset=True, ensure_ascii=False
        )

        # 通过异步生成器返回数据块
        yield f"data: {json_data}\n\n"

    # 返回完成信号
    yield "data: [DONE]\n\n"
    async def chat_completion_generate(
        self, model_name: str, params: Dict[str, Any], n: int
    ) -> ChatCompletionResponse:
        """Generate completion
        Args:
            model_name (str): Model name
            params (Dict[str, Any]): The parameters passed to the model worker
            n (int): How many completions to generate for each prompt.
        """
        # 获取工作管理器实例
        worker_manager: WorkerManager = self.get_worker_manager()
        # 初始化空列表用于存储生成的聊天完成项
        choices = []
        chat_completions = []
        # 根据指定次数生成聊天完成项
        for i in range(n):
            # 异步创建任务，生成模型输出
            model_output = asyncio.create_task(worker_manager.generate(params))
            chat_completions.append(model_output)
        try:
            # 等待所有聊天完成项任务完成
            all_tasks = await asyncio.gather(*chat_completions)
        except Exception as e:
            # 捕获异常并创建相应的错误响应
            return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
        # 初始化使用信息对象
        usage = UsageInfo()
        # 处理每个生成的模型输出
        for i, model_output in enumerate(all_tasks):
            # 将模型输出解析为 ModelOutput 类型
            model_output: ModelOutput = model_output
            # 如果模型输出中存在错误，创建相应的错误响应
            if model_output.error_code != 0:
                return create_error_response(model_output.error_code, model_output.text)
            # 构建聊天完成响应项并添加到选择列表中
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=model_output.text),
                    finish_reason=model_output.finish_reason or "stop",
                )
            )
            # 如果模型输出包含使用信息，验证并更新使用信息对象
            if model_output.usage:
                task_usage = UsageInfo.model_validate(model_output.usage)
                for usage_key, usage_value in model_to_dict(task_usage).items():
                    setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        # 返回完整的聊天完成响应对象
        return ChatCompletionResponse(model=model_name, choices=choices, usage=usage)

    async def embeddings_generate(
        self,
        model: str,
        texts: List[str],
        span_id: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings

        Args:
            model (str): Model name
            texts (List[str]): Texts to embed
            span_id (Optional[str], optional): The span id. Defaults to None.

        Returns:
            List[List[float]]: The embeddings of texts
        """
        # 使用根跟踪器开始生成嵌入向量的操作
        with root_tracer.start_span(
            "dbgpt.model.apiserver.generate_embeddings",
            parent_span_id=span_id,
            metadata={
                "model": model,
            },
        ):
            # 获取工作管理器实例
            worker_manager: WorkerManager = self.get_worker_manager()
            # 准备生成嵌入向量的参数
            params = {
                "input": texts,
                "model": model,
            }
            # 调用工作管理器生成嵌入向量
            return await worker_manager.embeddings(params)

    async def relevance_generate(
        self, model: str, query: str, texts: List[str]
    ) -> List[float]:
        """Generate relevance scores

        Args:
            model (str): Model name
            query (str): Query text
            texts (List[str]): Texts to compare relevance against

        Returns:
            List[float]: List of relevance scores
        """
    ) -> List[float]:
        """Generate embeddings
        
        Args:
            model (str): Model name
            query (str): Query text
            texts (List[str]): Texts to embed
        
        Returns:
            List[List[float]]: The embeddings of texts
        """
        # 获取当前对象的 WorkerManager 实例
        worker_manager: WorkerManager = self.get_worker_manager()
        # 准备传递给 embeddings 方法的参数字典
        params = {
            "input": texts,  # 将 texts 参数作为输入数据
            "model": model,  # 将 model 参数作为模型名称
            "query": query,  # 将 query 参数作为查询文本
        }
        # 调用 WorkerManager 的 embeddings 方法获取嵌入向量
        scores = await worker_manager.embeddings(params)
        # 返回嵌入向量的第一个元素（假设 scores 是一个列表，每个元素是一个嵌入向量）
        return scores[0]
# 定义一个函数，用于获取全局系统应用中的模型 API 服务器实例
def get_api_server() -> APIServer:
    # 调用全局系统应用的方法获取模型 API 服务器组件实例，类型为 APIServer，如果未找到则返回 None
    api_server = global_system_app.get_component(
        ComponentType.MODEL_API_SERVER, APIServer, default_component=None
    )
    # 如果未获取到有效的 api_server 实例
    if not api_server:
        # 在全局系统应用中注册一个 APIServer 组件
        global_system_app.register(APIServer)
    # 再次获取模型 API 服务器组件实例，并返回
    return global_system_app.get_component(ComponentType.MODEL_API_SERVER, APIServer)


# 创建一个 APIRouter 实例
router = APIRouter()


# 定义一个异步函数处理 GET 请求，获取可用的模型列表，依赖于 get_api_server 函数返回的 APIServer 实例
@router.get("/v1/models", dependencies=[Depends(check_api_key)])
async def get_available_models(api_server: APIServer = Depends(get_api_server)):
    return await api_server.get_available_models()


# 定义一个异步函数处理 POST 请求，创建聊天完成的请求，依赖于 get_api_server 函数返回的 APIServer 实例
@router.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(
    request: APIChatCompletionRequest, api_server: APIServer = Depends(get_api_server)
):
    # 调用 API 服务器的方法，获取指定模型实例，或者抛出异常
    await api_server.get_model_instances_or_raise(request.model)
    
    # 检查请求的有效性，如果有错误则返回错误信息
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret
    
    # 准备请求参数字典，包括模型名称、消息列表转换后的字典、温度、top_p、最大生成token数、停止标识、用户标识等
    params = {
        "model": request.model,
        "messages": ModelMessage.to_dict_list(
            ModelMessage.from_openai_messages(request.messages)
        ),
        "echo": False,
    }
    if request.temperature:
        params["temperature"] = request.temperature
    if request.top_p:
        params["top_p"] = request.top_p
    if request.max_tokens:
        params["max_new_tokens"] = request.max_tokens
    if request.stop:
        params["stop"] = request.stop
    if request.user:
        params["user"] = request.user
    
    # 构建跟踪信息的关键字参数字典
    trace_kwargs = {
        "operation_name": "dbgpt.model.apiserver.create_chat_completion",
        "metadata": {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stop": request.stop,
            "user": request.user,
        },
    }
    
    # 如果请求中包含流标志，则调用 API 服务器的流生成器方法，并生成响应流对象
    if request.stream:
        generator = api_server.chat_completion_stream_generator(
            request.model, params, request.n
        )
        # 使用根追踪器包装异步流生成器，返回一个流响应对象
        trace_generator = root_tracer.wrapper_async_stream(generator, **trace_kwargs)
        return StreamingResponse(trace_generator, media_type="text/event-stream")
    else:
        # 否则，在根追踪器下开始一个新的跟踪 span，并等待聊天完成的生成过程完成
        with root_tracer.start_span(**trace_kwargs):
            return await api_server.chat_completion_generate(
                request.model, params, request.n
            )


# 定义一个异步函数处理 POST 请求，创建嵌入向量的请求，依赖于 get_api_server 函数返回的 APIServer 实例
@router.post("/v1/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings(
    request: EmbeddingsRequest, api_server: APIServer = Depends(get_api_server)
):
    # 调用 API 服务器的方法，获取指定模型实例或者抛出异常，指定工作类型为"text2vec"
    await api_server.get_model_instances_or_raise(request.model, worker_type="text2vec")
    
    # 处理输入文本，如果是字符串则转换为列表
    texts = request.input
    if isinstance(texts, str):
        texts = [texts]
    
    # 设置批处理大小为 API 设置中定义的嵌入向量批处理大小
    batch_size = api_settings.embedding_bach_size
    
    # 将文本分成多个批次，每个批次大小为批处理大小
    batches = [
        texts[i : min(i + batch_size, len(texts))]
        for i in range(0, len(texts), batch_size)
    ]
    
    # 初始化数据列表和异步任务列表
    data = []
    async_tasks = []
    # 对每个批次进行枚举，生成异步任务列表
    for num_batch, batch in enumerate(batches):
        async_tasks.append(
            # 调用 API 服务器生成嵌入向量的异步请求，传入模型和当前批次数据，
            # 使用根跟踪器获取当前跟踪 ID 作为请求的跨度 ID
            api_server.embeddings_generate(request.model, batch, span_id=root_tracer.get_current_span_id())
        )

    # 并行请求所有批次的嵌入向量
    batch_embeddings: List[List[List[float]]] = await asyncio.gather(*async_tasks)
    
    # 遍历每个批次的嵌入向量，组装成标准化的数据列表
    for num_batch, embeddings in enumerate(batch_embeddings):
        data += [
            {
                "object": "embedding",    # 对象类型为嵌入向量
                "embedding": emb,         # 嵌入向量数据
                "index": num_batch * batch_size + i,  # 索引位置
            }
            for i, emb in enumerate(embeddings)  # 遍历批次中的每个嵌入向量
        ]
    
    # 返回嵌入向量响应的模型字典表示形式，排除值为 None 的字段
    return model_to_dict(
        EmbeddingsResponse(data=data, model=request.model, usage=UsageInfo()),
        exclude_none=True,
    )
# 定义一个路由处理函数，用于处理 POST 请求，路径为 /v1/beta/relevance
# 依赖项包括 check_api_key 函数验证 API 密钥，响应模型为 RelevanceResponse
async def create_relevance(
    request: RelevanceRequest, api_server: APIServer = Depends(get_api_server)
):
    """Generate relevance scores for a query and a list of documents."""
    
    # 获取 API 服务器中与 request.model 相关的模型实例，确保存在
    await api_server.get_model_instances_or_raise(request.model, worker_type="text2vec")

    # 创建一个名为 "dbgpt.model.apiserver.generate_relevance" 的跟踪 span
    # 设置元数据包括模型名称和查询内容
    with root_tracer.start_span(
        "dbgpt.model.apiserver.generate_relevance",
        metadata={
            "model": request.model,
            "query": request.query,
        },
    ):
        # 调用 API 服务器的 relevance_generate 方法生成相关性分数
        scores = await api_server.relevance_generate(
            request.model, request.query, request.documents
        )
    
    # 将相关性分数、模型和使用信息构建成 RelevanceResponse 对象，并转换为字典格式返回
    return model_to_dict(
        RelevanceResponse(data=scores, model=request.model, usage=UsageInfo()),
        exclude_none=True,
    )


# 初始化所有组件函数，接受控制器地址和系统应用对象作为参数
def _initialize_all(controller_addr: str, system_app: SystemApp):
    from dbgpt.model.cluster.controller.controller import ModelRegistryClient
    from dbgpt.model.cluster.worker.manager import _DefaultWorkerManagerFactory
    from dbgpt.model.cluster.worker.remote_manager import RemoteWorkerManager

    # 如果系统应用中不存在 MODEL_REGISTRY 组件，则注册该组件
    if not system_app.get_component(
        ComponentType.MODEL_REGISTRY, ModelRegistry, default_component=None
    ):
        # 创建 ModelRegistryClient 对象，使用控制器地址初始化
        registry = ModelRegistryClient(controller_addr)
        registry.name = ComponentType.MODEL_REGISTRY.value
        # 将注册的实例添加到系统应用中
        system_app.register_instance(registry)

    # 获取 MODEL_REGISTRY 组件，如果不存在则为 None
    registry = system_app.get_component(
        ComponentType.MODEL_REGISTRY, ModelRegistry, default_component=None
    )
    # 创建 RemoteWorkerManager 对象，使用注册的模型注册表
    worker_manager = RemoteWorkerManager(registry)

    # 如果系统应用中不存在 WORKER_MANAGER_FACTORY 组件，则注册该组件
    system_app.get_component(
        ComponentType.WORKER_MANAGER_FACTORY,
        WorkerManagerFactory,
        or_register_component=_DefaultWorkerManagerFactory,
        worker_manager=worker_manager,
    )
    # 如果系统应用中不存在 MODEL_API_SERVER 组件，则注册该组件
    system_app.get_component(
        ComponentType.MODEL_API_SERVER, APIServer, or_register_component=APIServer
    )


# 初始化 API 服务器函数，接受控制器地址和可选的 ModelAPIServerParameters 参数
def initialize_apiserver(
    controller_addr: str,
    apiserver_params: Optional[ModelAPIServerParameters] = None,
    app=None,
    system_app: SystemApp = None,
    host: str = None,
    port: int = None,
    api_keys: List[str] = None,
    embedding_batch_size: Optional[int] = None,
):
    import os

    from dbgpt.configs.model_config import LOGDIR

    # 全局变量声明
    global global_system_app
    global api_settings
    
    # 初始化 embedded_mod 标志，如果 app 不存在，则置为 False 并创建一个新的应用对象
    embedded_mod = True
    if not app:
        embedded_mod = False
        app = create_app()

    # 如果系统应用对象为空，则创建一个新的 SystemApp 对象
    if not system_app:
        system_app = SystemApp(app)
    global_system_app = system_app
    # 如果传入了 apiserver_params，则初始化追踪器
    if apiserver_params:
        # 使用指定的追踪文件路径初始化追踪器，设定系统应用和根操作名称
        initialize_tracer(
            os.path.join(LOGDIR, apiserver_params.tracer_file),
            system_app=system_app,
            root_operation_name="DB-GPT-APIServer",
            tracer_storage_cls=apiserver_params.tracer_storage_cls,
            enable_open_telemetry=apiserver_params.tracer_to_open_telemetry,
            otlp_endpoint=apiserver_params.otel_exporter_otlp_traces_endpoint,
            otlp_insecure=apiserver_params.otel_exporter_otlp_traces_insecure,
            otlp_timeout=apiserver_params.otel_exporter_otlp_traces_timeout,
        )

    # 如果传入了 api_keys，则设置全局的 API 密钥配置
    if api_keys:
        api_settings.api_keys = api_keys

    # 如果传入了 embedding_batch_size，则设置全局的嵌入批量大小
    if embedding_batch_size:
        api_settings.embedding_bach_size = embedding_batch_size

    # 将路由器注册到应用的 '/api' 前缀下，并添加标签为 "APIServer"
    app.include_router(router, prefix="/api", tags=["APIServer"])

    # 异常处理器，处理 APIServerException 异常，返回相应的错误响应
    @app.exception_handler(APIServerException)
    async def validation_apiserver_exception_handler(request, exc: APIServerException):
        return create_error_response(exc.code, exc.message)

    # 异常处理器，处理 RequestValidationError 异常，返回相应的验证错误响应
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))

    # 初始化所有的控制器地址和系统应用
    _initialize_all(controller_addr, system_app)

    # 如果没有传入 embedded_mod，则引入 uvicorn 并运行应用
    if not embedded_mod:
        import uvicorn

        # 创建支持跨域请求的中间件，并运行带有跨域支持的应用
        # 参考：https://github.com/encode/starlette/issues/617
        cors_app = CORSMiddleware(
            app=app,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        uvicorn.run(cors_app, host=host, port=port, log_level="info")
# 定义一个函数来运行 API 服务器
def run_apiserver():
    # 创建一个环境参数解析器的实例
    parser = EnvArgumentParser()
    # 定义环境变量前缀
    env_prefix = "apiserver_"
    # 使用解析器解析参数并将其转换为指定的数据类 ModelAPIServerParameters
    apiserver_params: ModelAPIServerParameters = parser.parse_args_into_dataclass(
        ModelAPIServerParameters,
        env_prefixes=[env_prefix],
    )
    # 设置日志记录的配置，包括日志名称、日志级别和日志文件名
    setup_logging(
        "dbgpt",
        logging_level=apiserver_params.log_level,
        logger_filename=apiserver_params.log_file,
    )
    # 如果配置了 API 密钥，则将其拆分为列表形式
    api_keys = None
    if apiserver_params.api_keys:
        api_keys = apiserver_params.api_keys.strip().split(",")

    # 初始化 API 服务器，设置控制器地址、参数、主机、端口、API 密钥和嵌入批处理大小
    initialize_apiserver(
        apiserver_params.controller_addr,
        apiserver_params,
        host=apiserver_params.host,
        port=apiserver_params.port,
        api_keys=api_keys,
        embedding_batch_size=apiserver_params.embedding_batch_size,
    )


if __name__ == "__main__":
    # 主程序入口，运行 API 服务器
    run_apiserver()
```