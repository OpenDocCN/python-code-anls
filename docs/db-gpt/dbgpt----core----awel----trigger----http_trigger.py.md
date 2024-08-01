# `.\DB-GPT-src\dbgpt\core\awel\trigger\http_trigger.py`

```py
"""Http trigger for AWEL."""

# 导入所需的模块和库
import json  # 导入处理 JSON 格式数据的模块
import logging  # 导入日志记录模块
from enum import Enum  # 导入枚举类型的支持
from typing import (  # 导入类型提示相关的模块
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
    get_origin,
)

# 导入 Pydantic 相关模块和函数
from dbgpt._private.pydantic import (
    BaseModel,  # 导入基础模型类
    Field,  # 导入字段定义类
    field_is_required,  # 导入字段是否必需的函数
    field_outer_type,  # 导入字段外部类型函数
    model_fields,  # 导入模型字段函数
    model_to_dict,  # 导入模型转字典函数
)

# 导入国际化相关函数
from dbgpt.util.i18n_utils import _

# 导入追踪器模块
from dbgpt.util.tracer import root_tracer

# 导入 DAG 类型
from ..dag.base import DAG

# 导入流相关模块和类
from ..flow import (
    IOField,  # 输入输出字段类
    OperatorCategory,  # 运算符类别枚举
    OperatorType,  # 运算符类型枚举
    OptionValue,  # 选项值类
    Parameter,  # 参数类
    ResourceCategory,  # 资源类别枚举
    ResourceType,  # 资源类型枚举
    ViewMetadata,  # 视图元数据类
    register_resource,  # 注册资源函数
)

# 导入基础操作符相关模块和类
from ..operators.base import BaseOperator

# 导入常用操作符类
from ..operators.common_operator import MapOperator

# 导入类型工具函数
from ..util._typing_util import _parse_bool

# 导入 HTTP 相关工具函数
from ..util.http_util import join_paths

# 导入触发器基类和元数据
from .base import Trigger, TriggerMetadata

# 如果是类型检查模式，导入相应模块和类
if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI  # 导入 FastAPI 相关类
    from starlette.requests import Request  # 导入 Starlette 请求类

    from dbgpt.core.interface.llm import ModelRequestContext  # 导入模型请求上下文类

    # 定义通用请求体类型
    RequestBody = Union[Type[Request], Type[BaseModel], Type[Dict[str, Any]], Type[str]]
    CommonRequestType = Union[Request, BaseModel, Dict[str, Any], str, None]
    StreamingPredictFunc = Callable[[CommonRequestType], bool]

# 设置日志记录器
logger = logging.getLogger(__name__)


class AWELHttpError(RuntimeError):
    """AWEL Http Error."""

    def __init__(self, msg: str, code: Optional[str] = None):
        """Init the AWELHttpError."""
        super().__init__(msg)
        self.msg = msg  # 初始化错误消息
        self.code = code  # 初始化错误代码


def _default_streaming_predict_func(body: "CommonRequestType") -> bool:
    """默认的流预测函数。

    根据请求体的类型预测是否为流式处理请求。

    Args:
        body (CommonRequestType): 请求体。

    Returns:
        bool: 预测结果，True 表示是流式处理请求，False 表示不是。
    """
    if isinstance(body, BaseModel):
        body = model_to_dict(body)
    elif isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            return False
    elif not isinstance(body, dict):
        return False
    streaming = body.get("streaming") or body.get("stream")
    return _parse_bool(streaming)


class HttpTriggerMetadata(TriggerMetadata):
    """Http触发器元数据类。"""

    path: str = Field(..., description="触发器的路径")
    methods: List[str] = Field(..., description="触发器的方法列表")

    trigger_type: Optional[str] = Field(
        default="http", description="触发器的类型"
    )


class BaseHttpBody(BaseModel):
    """Http请求体或响应体基类。

    用于 HTTP 请求体或响应体。
    """

    @classmethod
    def get_body_class(cls) -> Type:
        """获取请求体类。

        Returns:
            Type: 请求体类。
        """
        return cls

    def get_body(self) -> Any:
        """获取请求体。

        Returns:
            Any: 请求体。
        """
        return self

    @classmethod
    def streaming_predict_func(cls) -> Optional["StreamingPredictFunc"]:
        """获取流预测函数。

        Returns:
            Optional[StreamingPredictFunc]: 流预测函数。
        """
        return _default_streaming_predict_func
    # 定义一个方法 `streaming_response`，用于判断响应是否是流式的
    def streaming_response(self) -> bool:
        """Whether the response is streaming.

        Returns:
            bool: Whether the response is streaming.
        """
        # 返回值为 False，表示响应不是流式的
        return False
@register_resource(
    label=_("Dict Http Body"),
    name="dict_http_body",
    category=ResourceCategory.HTTP_BODY,
    resource_type=ResourceType.CLASS,
    description=_("Parse the request body as a dict or response body as a dict"),
)
class DictHttpBody(BaseHttpBody):
    """Dict http body."""

    _default_body: Optional[Dict[str, Any]] = None

    @classmethod
    def get_body_class(cls) -> Type[Dict[str, Any]]:
        """Get body class.

        Just return Dict[str, Any] here.

        Returns:
            Type[Dict[str, Any]]: The body class.
        """
        return Dict[str, Any]

    def get_body(self) -> Dict[str, Any]:
        """Get the body."""
        if self._default_body is None:
            raise AWELHttpError("DictHttpBody is not set")
        return self._default_body


@register_resource(
    label=_("String Http Body"),
    name="string_http_body",
    category=ResourceCategory.HTTP_BODY,
    resource_type=ResourceType.CLASS,
    description=_("Parse the request body as a string or response body as string"),
)
class StringHttpBody(BaseHttpBody):
    """String http body."""

    _default_body: Optional[str] = None

    @classmethod
    def get_body_class(cls) -> Type[str]:
        """Get body class.

        Just return str here.

        Returns:
            Type[str]: The body class.
        """
        return str

    def get_body(self) -> str:
        """Get the body."""
        if self._default_body is None:
            raise AWELHttpError("StringHttpBody is not set")
        return self._default_body


@register_resource(
    label=_("Request Http Body"),
    name="request_http_body",
    category=ResourceCategory.HTTP_BODY,
    resource_type=ResourceType.CLASS,
    description=_("Parse the request body as a starlette Request"),
)
class RequestHttpBody(BaseHttpBody):
    """Http trigger body."""

    _default_body: Optional["Request"] = None

    @classmethod
    def get_body_class(cls) -> Type["Request"]:
        """Get the request body type.

        Import the Request class from starlette.requests module and return it.

        Returns:
            Type[Request]: The request body type.
        """
        from starlette.requests import Request

        return Request

    def get_body(self) -> "Request":
        """Get the body."""
        if self._default_body is None:
            raise AWELHttpError("RequestHttpBody is not set")
        return self._default_body


@register_resource(
    label=_("Common LLM Http Request Body"),
    name="common_llm_http_request_body",
    category=ResourceCategory.HTTP_BODY,
    resource_type=ResourceType.CLASS,
    description=_("Parse the request body as a common LLM http body"),
)
class CommonLLMHttpRequestBody(BaseHttpBody):
    """Common LLM http request body."""

    model: str = Field(
        ..., description="The model name", examples=["gpt-3.5-turbo", "proxyllm"]
    )
    messages: Union[str, List[str]] = Field(
        ..., description="User input messages", examples=["Hello", "How are you?"]
    )
    stream: bool = Field(default=False, description="Whether return stream")
    # 是否返回流式数据的标志，默认为 False

    temperature: Optional[float] = Field(
        default=None,
        description="What sampling temperature to use, between 0 and 2. Higher values "
        "like 0.8 will make the output more random, while lower values like 0.2 will "
        "make it more focused and deterministic.",
    )
    # 采样温度，介于 0 和 2 之间的浮点数。较高的值（如 0.8）会使输出更随机，而较低的值（如 0.2）会使其更集中和确定性。

    max_new_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens that can be generated in the chat "
        "completion.",
    )
    # 聊天完成时可以生成的最大令牌数。

    conv_uid: Optional[str] = Field(
        default=None, description="The conversation id of the model inference"
    )
    # 模型推断的对话 ID。

    span_id: Optional[str] = Field(
        default=None, description="The span id of the model inference"
    )
    # 模型推断的 span ID。

    chat_mode: Optional[str] = Field(
        default="chat_normal",
        description="The chat mode",
        examples=["chat_awel_flow", "chat_normal"],
    )
    # 聊天模式，默认为 "chat_normal"，可选值包括 "chat_awel_flow" 和 "chat_normal"。

    chat_param: Optional[str] = Field(
        default=None,
        description="The chat param of chat mode",
    )
    # 聊天模式的参数。

    user_name: Optional[str] = Field(
        default=None, description="The user name of the model inference"
    )
    # 模型推断的用户名称。

    sys_code: Optional[str] = Field(
        default=None, description="The system code of the model inference"
    )
    # 模型推断的系统代码。

    incremental: bool = Field(
        default=True,
        description="Used to control whether the content is returned incrementally "
        "or in full each time. "
        "If this parameter is not provided, the default is full return.",
    )
    # 是否增量返回内容的标志。默认为 True，即增量返回。

    enable_vis: bool = Field(
        default=True, description="response content whether to output vis label"
    )
    # 是否输出可视化标签的响应内容，默认为 True。

    extra: Optional[Dict[str, Any]] = Field(
        default=None, description="The extra info of the model inference"
    )
    # 模型推断的额外信息。

    @property
    def context(self) -> "ModelRequestContext":
        """Get the model request context."""
        from dbgpt.core.interface.llm import ModelRequestContext

        return ModelRequestContext(
            stream=self.stream,
            user_name=self.user_name,
            sys_code=self.sys_code,
            conv_uid=self.conv_uid,
            span_id=self.span_id,
            chat_mode=self.chat_mode,
            chat_param=self.chat_param,
            extra=self.extra,
        )
    # 返回模型请求的上下文对象，包括各个属性的值。
@register_resource(
    label=_("Common LLM Http Response Body"),  # 资源注册的标签，用于标识资源
    name="common_llm_http_response_body",  # 资源的名称
    category=ResourceCategory.HTTP_BODY,  # 资源的类别，这里是HTTP响应体
    resource_type=ResourceType.CLASS,  # 资源的类型，这里是一个类
    description=_("Parse the response body as a common LLM http body"),  # 资源的描述信息
)
class CommonLLMHttpResponseBody(BaseHttpBody):
    """Common LLM http response body."""

    text: str = Field(
        ...,  # 生成的文本内容，类型为字符串
        description="The generated text",  # 描述字段含义
        examples=["Hello", "How are you?"]  # 示例值
    )
    error_code: int = Field(
        default=0,  # 默认值为0，表示没有错误
        description="The error code, 0 means no error",  # 错误码的含义描述
        examples=[0, 1]  # 示例值
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,  # 默认为None
        description="The metrics of the model, like the number of tokens generated",  # 模型的度量信息描述
    )


class HttpTrigger(Trigger):
    """Http trigger for AWEL.

    Http trigger is used to trigger a DAG by http request.
    """
    metadata = ViewMetadata(
        # 设置视图元数据的标签为 "Http Trigger"
        label="Http Trigger",
        # 设置视图元数据的名称为 "http_trigger"
        name="http_trigger",
        # 将视图分类为触发器
        category=OperatorCategory.TRIGGER,
        # 操作类型设定为输入操作
        operator_type=OperatorType.INPUT,
        # 设定描述信息，说明这是通过 HTTP 请求触发工作流的操作
        description="Trigger your workflow by http request",
        # 定义输入参数为空列表
        inputs=[],
        # 定义输出参数为空列表
        outputs=[],
        # 定义一系列参数
        parameters=[
            # 创建名为 "API Endpoint" 的参数，类型为字符串，描述为 API 的端点
            Parameter.build_from(
                "API Endpoint", "endpoint", str, description="The API endpoint"
            ),
            # 创建名为 "Http Methods" 的参数，类型为字符串，可选参数，默认为 "GET"，描述为 API 端点支持的 HTTP 方法
            Parameter.build_from(
                "Http Methods",
                "methods",
                str,
                optional=True,
                default="GET",
                description="The methods of the API endpoint",
                # 设定选项列表，包括 GET、PUT、POST、DELETE 四种 HTTP 方法
                options=[
                    OptionValue(label="HTTP Method GET", name="http_get", value="GET"),
                    OptionValue(label="HTTP Method PUT", name="http_put", value="PUT"),
                    OptionValue(label="HTTP Method POST", name="http_post", value="POST"),
                    OptionValue(label="HTTP Method DELETE", name="http_delete", value="DELETE"),
                ],
            ),
            # 创建名为 "Http Request Trigger Body" 的参数，类型为 BaseHttpBody 类，可选参数，默认为 None，描述为 API 端点的请求体
            Parameter.build_from(
                "Http Request Trigger Body",
                "http_trigger_body",
                BaseHttpBody,
                optional=True,
                default=None,
                description="The request body of the API endpoint",
                resource_type=ResourceType.CLASS,
            ),
            # 创建名为 "Streaming Response" 的参数，类型为布尔值，可选参数，默认为 False，描述为是否响应是流式的
            Parameter.build_from(
                "Streaming Response",
                "streaming_response",
                bool,
                optional=True,
                default=False,
                description="Whether the response is streaming",
            ),
            # 创建名为 "Http Response Body" 的参数，类型为 BaseHttpBody 类，可选参数，默认为 None，描述为 API 端点的响应体
            Parameter.build_from(
                "Http Response Body",
                "http_response_body",
                BaseHttpBody,
                optional=True,
                default=None,
                description="The response body of the API endpoint",
                resource_type=ResourceType.CLASS,
            ),
            # 创建名为 "Response Media Type" 的参数，类型为字符串，可选参数，默认为 None，描述为响应的媒体类型
            Parameter.build_from(
                "Response Media Type",
                "response_media_type",
                str,
                optional=True,
                default=None,
                description="The response media type",
            ),
            # 创建名为 "Http Status Code" 的参数，类型为整数，可选参数，默认为 200，描述为 HTTP 状态码
            Parameter.build_from(
                "Http Status Code",
                "status_code",
                int,
                optional=True,
                default=200,
                description="The http status code",
            ),
        ],
    )
    def __init__(
        self,
        endpoint: str,
        methods: Optional[Union[str, List[str]]] = "GET",
        request_body: Optional["RequestBody"] = None,
        http_trigger_body: Optional[Type[BaseHttpBody]] = None,
        streaming_response: bool = False,
        streaming_predict_func: Optional["StreamingPredictFunc"] = None,
        http_response_body: Optional[Type[BaseHttpBody]] = None,
        response_model: Optional[Type] = None,
        response_headers: Optional[Dict[str, str]] = None,
        response_media_type: Optional[str] = None,
        status_code: Optional[int] = 200,
        router_tags: Optional[List[str | Enum]] = None,
        register_to_app: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a HttpTrigger."""
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        
        # 如果 endpoint 不以 '/' 开头，则添加 '/'
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        
        # 如果 request_body 为空但存在 http_trigger_body，则获取其 body 类，并设置 streaming_predict_func
        if not request_body and http_trigger_body:
            request_body = http_trigger_body.get_body_class()
            streaming_predict_func = http_trigger_body.streaming_predict_func()
        
        # 如果 response_model 为空但存在 http_response_body，则获取其 body 类
        if not response_model and http_response_body:
            response_model = http_response_body.get_body_class()
        
        # 设置对象的各个属性
        self._endpoint = endpoint
        self._methods = [methods] if isinstance(methods, str) else methods
        self._req_body = request_body
        self._streaming_response = _parse_bool(streaming_response)
        self._streaming_predict_func = streaming_predict_func
        self._response_model = response_model
        self._status_code = status_code
        self._router_tags = router_tags
        self._response_headers = response_headers
        self._response_media_type = response_media_type
        self._end_node: Optional[BaseOperator] = None
        self._register_to_app = register_to_app

    async def trigger(self, **kwargs) -> Any:
        """Trigger the DAG. Not used in HttpTrigger."""
        # 抛出未实现错误，因为 HttpTrigger 不支持直接触发
        raise NotImplementedError("HttpTrigger does not support trigger directly")

    def register_to_app(self) -> bool:
        """Register the trigger to a FastAPI app.

        Returns:
            bool: Whether register to app, if not register to app, will register to
                router.
        """
        # 返回是否将触发器注册到 FastAPI 应用程序
        return self._register_to_app

    def mount_to_router(
        self, router: "APIRouter", global_prefix: Optional[str] = None
    ):
        """Mount the HttpTrigger to a FastAPI router.

        Args:
            router (APIRouter): The FastAPI router to mount to.
            global_prefix (Optional[str], optional): Global prefix for the router. Defaults to None.
        """
        # 方法未完整展示，但通常会将触发器注册到 FastAPI 路由器中
        pass
    def mount_to_app(
        self, app: "FastAPI", global_prefix: Optional[str] = None
    ) -> HttpTriggerMetadata:
        """Mount the trigger to a FastAPI app.

        TODO: The performance of this method is not good, need to be optimized.

        Args:
            app (FastAPI): The FastAPI app.
            global_prefix (Optional[str], optional): The global prefix of the app.
                Defaults to None.
        """
        # 构建完整的路径，如果有全局前缀则加上全局前缀，否则直接使用端点路径
        path = (
            join_paths(global_prefix, self._endpoint)
            if global_prefix
            else self._endpoint
        )
        # 创建动态路由处理函数
        dynamic_route_function = self._create_route_func()
        # 将路由添加到 FastAPI 应用的路由器中
        router = cast(PriorityAPIRouter, app.router)
        router.add_api_route(
            path,
            dynamic_route_function,
            methods=self._methods,
            response_model=self._response_model,
            status_code=self._status_code,
            tags=self._router_tags,
            priority=10,  # 设置优先级为 10
        )
        # 清空 OpenAPI 模式，因为路由已经发生变化
        app.openapi_schema = None
        # 清空中间件栈，因为路由已经发生变化
        app.middleware_stack = None
        # 记录日志，表示成功挂载 HTTP 触发器到指定路径
        logger.info(f"Mount http trigger success, path: {path}")
        # 返回包含路径和方法的 HttpTriggerMetadata 对象
        return HttpTriggerMetadata(path=path, methods=self._methods)

    def remove_from_app(
        self, app: "FastAPI", global_prefix: Optional[str] = None
    ) -> None:
        """Remove the trigger from a FastAPI app.

        Args:
            app (FastAPI): The FastAPI app.
            global_prefix (Optional[str], optional): The global prefix of the app.
                Defaults to None.
        """
        # 构建完整的路径，如果有全局前缀则加上全局前缀，否则直接使用端点路径
        path = (
            join_paths(global_prefix, self._endpoint)
            if global_prefix
            else self._endpoint
        )
        # 强制转换应用程序的路由器为 APIRouter 类型
        app_router = cast(APIRouter, app.router)
        # 遍历应用程序路由器中的所有路由
        for i, r in enumerate(app_router.routes):
            # 如果路由的路径格式匹配当前路径
            if r.path_format == path:  # type: ignore
                # TODO，根据路径和方法删除路由
                del app_router.routes[i]
async def _trigger_dag(
    body: Any,
    dag: DAG,
    streaming_response: Optional[bool] = False,
    response_headers: Optional[Dict[str, str]] = None,
    response_media_type: Optional[str] = None,
) -> Any:
    # 导入必要的模块：BackgroundTasks 和 StreamingResponse
    from fastapi import BackgroundTasks
    from fastapi.responses import StreamingResponse

    # 从根跟踪器解析 span_id
    span_id = root_tracer._parse_span_id(body)

    # 获取 DAG 的叶节点
    leaf_nodes = dag.leaf_nodes
    # 如果叶节点数量不为1，抛出数值错误
    if len(leaf_nodes) != 1:
        raise ValueError("HttpTrigger just support one leaf node in dag")
    # 将叶节点转换为 BaseOperator 类型
    end_node = cast(BaseOperator, leaf_nodes[0])
    # 构建元数据字典，包含节点的 ID 和名称
    metadata = {
        "awel_node_id": end_node.node_id,
        "awel_node_name": end_node.node_name,
    }
    
    # 如果不需要流式响应
    if not streaming_response:
        # 使用根跟踪器启动一个 span，并在执行结束后返回结果
        with root_tracer.start_span(
            "dbgpt.core.trigger.http.run_dag", span_id, metadata=metadata
        ):
            return await end_node.call(call_data=body)
    else:
        # 设置响应头和媒体类型
        headers = response_headers
        media_type = response_media_type if response_media_type else "text/event-stream"
        # 如果没有指定响应头，设置默认响应头
        if not headers:
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            }
        # 调用节点的异步流方法，并使用跟踪器封装生成器
        _generator = await end_node.call_stream(call_data=body)
        trace_generator = root_tracer.wrapper_async_stream(
            _generator, "dbgpt.core.trigger.http.run_dag", span_id, metadata=metadata
        )

        # 定义在 DAG 执行结束后执行的异步函数
        async def _after_dag_end():
            await dag._after_dag_end(end_node.current_event_loop_task_id)

        # 创建 BackgroundTasks 对象，并添加 DAG 结束后执行的任务
        background_tasks = BackgroundTasks()
        background_tasks.add_task(_after_dag_end)
        # 返回流式响应对象
        return StreamingResponse(
            trace_generator,
            headers=headers,
            media_type=media_type,
            background=background_tasks,
        )


# 创建 API 端点的参数对象
_PARAMETER_ENDPOINT = Parameter.build_from(
    _("API Endpoint"), "endpoint", str, description=_("The API endpoint")
)
# 创建 HTTP POST 和 PUT 方法的参数对象
_PARAMETER_METHODS_POST_PUT = Parameter.build_from(
    _("Http Methods"),
    "methods",
    str,
    optional=True,
    default="POST",
    description=_("The methods of the API endpoint"),
    options=[
        OptionValue(label=_("HTTP Method PUT"), name="http_put", value="PUT"),
        OptionValue(label=_("HTTP Method POST"), name="http_post", value="POST"),
    ],
)
# 创建所有 HTTP 方法的参数对象
_PARAMETER_METHODS_ALL = Parameter.build_from(
    _("Http Methods"),
    "methods",
    str,
    optional=True,
    default="GET",
    description=_("The methods of the API endpoint"),
    options=[
        OptionValue(label=_("HTTP Method GET"), name="http_get", value="GET"),
        OptionValue(label=_("HTTP Method DELETE"), name="http_delete", value="DELETE"),
        OptionValue(label=_("HTTP Method PUT"), name="http_put", value="PUT"),
        OptionValue(label=_("HTTP Method POST"), name="http_post", value="POST"),
    ],
)
# 创建流式响应的参数对象
_PARAMETER_STREAMING_RESPONSE = Parameter.build_from(
    _("Streaming Response"),
    "streaming_response",
    Optional[bool],
    optional=True,
    default=False,
    description=_("Enable streaming response for HTTP trigger"),
)
    bool,  # 参数类型为布尔型
    optional=True,  # 参数可选，默认为True
    default=False,  # 参数的默认值为False
    description=_("Whether the response is streaming"),  # 参数的描述，指示响应是否为流式
# 定义一个命名的常量，表示参数的响应体
_PARAMETER_RESPONSE_BODY = Parameter.build_from(
    _("Http Response Body"),  # 参数的显示名称，国际化处理
    "http_response_body",  # 参数的名称
    BaseHttpBody,  # 参数的类型，基于基类BaseHttpBody
    optional=True,  # 参数是否可选
    default=None,  # 参数的默认值
    description=_("The response body of the API endpoint"),  # 参数的描述信息
    resource_type=ResourceType.CLASS,  # 参数的资源类型
)

# 定义一个命名的常量，表示参数的响应媒体类型
_PARAMETER_MEDIA_TYPE = Parameter.build_from(
    _("Response Media Type"),  # 参数的显示名称，国际化处理
    "response_media_type",  # 参数的名称
    str,  # 参数的类型为字符串
    optional=True,  # 参数是否可选
    default=None,  # 参数的默认值
    description=_("The response media type"),  # 参数的描述信息
)

# 定义一个命名的常量，表示参数的HTTP状态码
_PARAMETER_STATUS_CODE = Parameter.build_from(
    _("Http Status Code"),  # 参数的显示名称，国际化处理
    "status_code",  # 参数的名称
    int,  # 参数的类型为整数
    optional=True,  # 参数是否可选
    default=200,  # 参数的默认值为200
    description=_("The http status code"),  # 参数的描述信息
)

# 定义一个类DictHttpTrigger，继承自HttpTrigger类，用于处理字典类型的HTTP触发器
class DictHttpTrigger(HttpTrigger):
    """Dict http trigger for AWEL.

    Parse the request body as a dict.
    """

    # 类的元数据信息
    metadata = ViewMetadata(
        label=_("Dict Http Trigger"),  # 元数据的显示标签，国际化处理
        name="dict_http_trigger",  # 元数据的名称
        category=OperatorCategory.TRIGGER,  # 元数据的类别，操作符的触发器
        operator_type=OperatorType.INPUT,  # 操作符的类型，输入操作符
        description=_(
            "Trigger your workflow by http request, and parse the request body"
            " as a dict"
        ),  # 元数据的描述信息，描述HTTP请求触发工作流，并将请求体解析为字典类型
        inputs=[],  # 元数据的输入信息为空列表
        outputs=[
            IOField.build_from(
                _("Request Body"),  # 输出字段的显示名称，国际化处理
                "request_body",  # 输出字段的名称
                dict,  # 输出字段的类型为字典
                description=_("The request body of the API endpoint"),  # 输出字段的描述信息
            ),
        ],
        parameters=[  # 元数据的参数信息列表
            _PARAMETER_ENDPOINT.new(),  # 使用预定义的_ENDPOINT参数的实例
            _PARAMETER_METHODS_POST_PUT.new(),  # 使用预定义的METHODS_POST_PUT参数的实例
            _PARAMETER_STREAMING_RESPONSE.new(),  # 使用预定义的STREAMING_RESPONSE参数的实例
            _PARAMETER_RESPONSE_BODY.new(),  # 使用预定义的RESPONSE_BODY参数的实例
            _PARAMETER_MEDIA_TYPE.new(),  # 使用预定义的MEDIA_TYPE参数的实例
            _PARAMETER_STATUS_CODE.new(),  # 使用预定义的STATUS_CODE参数的实例
        ],
    )

    # 初始化方法，用于创建DictHttpTrigger类的实例
    def __init__(
        self,
        endpoint: str,  # HTTP端点的URL
        methods: Optional[Union[str, List[str]]] = "POST",  # HTTP方法，可选参数，默认为"POST"
        streaming_response: bool = False,  # 是否支持流式响应，可选参数，默认为False
        http_response_body: Optional[Type[BaseHttpBody]] = None,  # HTTP响应体的类型，可选参数，默认为None
        response_media_type: Optional[str] = None,  # 响应的媒体类型，可选参数，默认为None
        status_code: Optional[int] = 200,  # HTTP状态码，可选参数，默认为200
        router_tags: Optional[List[str | Enum]] = None,  # 路由标签列表，可选参数，默认为None
        **kwargs,  # 其他关键字参数
    ):
        """Initialize a DictHttpTrigger."""
        if not router_tags:  # 如果路由标签为空
            router_tags = ["AWEL DictHttpTrigger"]  # 设置默认的路由标签为"AWEL DictHttpTrigger"
        super().__init__(  # 调用父类(HttpTrigger)的初始化方法
            endpoint,  # 传入HTTP端点的URL
            methods,  # 传入HTTP方法
            streaming_response=streaming_response,  # 传入是否支持流式响应
            request_body=dict,  # 指定请求体的类型为字典
            http_response_body=http_response_body,  # 传入HTTP响应体的类型
            response_media_type=response_media_type,  # 传入响应的媒体类型
            status_code=status_code,  # 传入HTTP状态码
            router_tags=router_tags,  # 传入路由标签
            register_to_app=True,  # 将实例注册到应用程序中
            **kwargs,  # 传入其他关键字参数
        )


# 定义一个类StringHttpTrigger，继承自HttpTrigger类，用于处理字符串类型的HTTP触发器
class StringHttpTrigger(HttpTrigger):
    """String http trigger for AWEL."""
    metadata = ViewMetadata(
        label=_("String Http Trigger"),  # 设置视图标签为“String Http Trigger”
        name="string_http_trigger",  # 设置视图名称为"string_http_trigger"
        category=OperatorCategory.TRIGGER,  # 设置视图类别为触发器
        operator_type=OperatorType.INPUT,  # 设置操作类型为输入操作
        description=_(
            "Trigger your workflow by http request, and parse the request body"
            " as a string"
        ),  # 设置视图描述，描述触发器通过 HTTP 请求触发工作流，并将请求体解析为字符串
    
        # 设置视图输入为空列表
        inputs=[],
    
        # 设置视图输出，包含一个 IOField 对象，表示输出中的请求体
        outputs=[
            IOField.build_from(
                _("Request Body"),  # 输出字段名称为"Request Body"
                "request_body",  # 输出字段标识符为"request_body"
                str,  # 输出字段类型为字符串
                description=_(
                    "The request body of the API endpoint, parse as a json " "string"
                ),  # 输出字段描述，表示 API 端点的请求体，解析为 JSON 字符串
            ),
        ],
    
        # 设置视图参数列表，包括几个预定义的参数对象
        parameters=[
            _PARAMETER_ENDPOINT.new(),  # 添加新的端点参数对象
            _PARAMETER_METHODS_POST_PUT.new(),  # 添加新的请求方法参数对象
            _PARAMETER_STREAMING_RESPONSE.new(),  # 添加新的流式响应参数对象
            _PARAMETER_RESPONSE_BODY.new(),  # 添加新的 HTTP 响应体参数对象
            _PARAMETER_MEDIA_TYPE.new(),  # 添加新的媒体类型参数对象
            _PARAMETER_STATUS_CODE.new(),  # 添加新的状态码参数对象
        ],
    )
    
    def __init__(
        self,
        endpoint: str,
        methods: Optional[Union[str, List[str]]] = "POST",
        streaming_response: bool = False,
        http_response_body: Optional[Type[BaseHttpBody]] = None,
        response_media_type: Optional[str] = None,
        status_code: Optional[int] = 200,
        router_tags: Optional[List[str | Enum]] = None,
        **kwargs,
    ):
        """Initialize a StringHttpTrigger."""
        # 如果未提供 router_tags，则设置默认为 ["AWEL StringHttpTrigger"]
        if not router_tags:
            router_tags = ["AWEL StringHttpTrigger"]
        
        # 调用父类的初始化方法，初始化基类 Operator，并传入各种参数
        super().__init__(
            endpoint,
            methods,
            streaming_response=streaming_response,
            request_body=str,  # 设置请求体类型为字符串
            http_response_body=http_response_body,
            response_media_type=response_media_type,
            status_code=status_code,
            router_tags=router_tags,
            register_to_app=True,
            **kwargs,
        )
class CommonLLMHttpTrigger(HttpTrigger):
    """Common LLM http trigger for AWEL."""

    metadata = ViewMetadata(
        label=_("Common LLM Http Trigger"),
        name="common_llm_http_trigger",
        category=OperatorCategory.TRIGGER,
        operator_type=OperatorType.INPUT,
        description=_(
            "Trigger your workflow by http request, and parse the request body "
            "as a common LLM http body"
        ),
        inputs=[],
        outputs=[
            IOField.build_from(
                _("Request Body"),
                "request_body",
                CommonLLMHttpRequestBody,
                description=_(
                    "The request body of the API endpoint, parse as a common "
                    "LLM http body"
                ),
            ),
        ],
        parameters=[
            _PARAMETER_ENDPOINT.new(),
            _PARAMETER_METHODS_POST_PUT.new(),
            _PARAMETER_STREAMING_RESPONSE.new(),
            _PARAMETER_RESPONSE_BODY.new(),
            _PARAMETER_MEDIA_TYPE.new(),
            _PARAMETER_STATUS_CODE.new(),
        ],
    )

    def __init__(
        self,
        endpoint: str,
        methods: Optional[Union[str, List[str]]] = "POST",
        streaming_response: bool = False,
        http_response_body: Optional[Type[BaseHttpBody]] = None,
        response_media_type: Optional[str] = None,
        status_code: Optional[int] = 200,
        router_tags: Optional[List[str | Enum]] = None,
        **kwargs,
    ):
        """Initialize a CommonLLMHttpTrigger."""
        # 设置默认的路由标签，如果未提供则为 ["AWEL CommonLLMHttpTrigger"]
        if not router_tags:
            router_tags = ["AWEL CommonLLMHttpTrigger"]
        # 调用父类的初始化方法，传入必要的参数和关键字参数
        super().__init__(
            endpoint,
            methods,
            streaming_response=streaming_response,
            request_body=CommonLLMHttpRequestBody,
            http_response_body=http_response_body,
            response_media_type=response_media_type,
            status_code=status_code,
            router_tags=router_tags,
            register_to_app=True,
            **kwargs,
        )


@register_resource(
    label=_("Example Http Response"),
    name="example_http_response",
    category=ResourceCategory.HTTP_BODY,
    resource_type=ResourceType.CLASS,
    description=_("Example Http Request"),
)
class ExampleHttpResponse(BaseHttpBody):
    """Example Http Response.

    Just for test.
    Register as a resource.
    """

    # 服务器响应字段，来自 Operator
    server_res: str = Field(..., description="The server response from Operator")
    # HTTP 请求的请求体字段
    request_body: Dict[str, Any] = Field(
        ..., description="The request body from Http request"
    )


class ExampleHttpHelloOperator(MapOperator[dict, ExampleHttpResponse]):
    """Example Http Hello Operator.

    Just for test.
    """
    # 创建视图元数据对象 ViewMetadata，用于定义操作符的元数据信息
    metadata = ViewMetadata(
        # 设置操作符的标签，显示名称
        label=_("Example Http Hello Operator"),
        # 操作符的名称，用于标识操作符
        name="example_http_hello_operator",
        # 操作符所属的类别，这里是通用类别
        category=OperatorCategory.COMMON,
        # 操作符的参数列表为空
        parameters=[],
        # 输入字段列表，包括 HTTP 请求体作为输入
        inputs=[
            IOField.build_from(
                _("Http Request Body"),
                "request_body",
                dict,
                # 输入字段的描述，指明其为 API 端点的请求体（Dict[str, Any] 类型）
                description=_("The request body of the API endpoint (Dict[str, Any])"),
            )
        ],
        # 输出字段列表，包括响应体作为输出
        outputs=[
            IOField.build_from(
                _("Response Body"),
                "response_body",
                ExampleHttpResponse,
                # 输出字段的描述，指明其为 API 端点的响应体
                description=_("The response body of the API endpoint"),
            )
        ],
        # 操作符的描述信息
        description=_("Example Http Hello Operator"),
    )

    # 定义 ExampleHttpHelloOperator 类，并进行初始化
    def __int__(self, **kwargs):
        """Initialize a ExampleHttpHelloOperator."""
        # 调用父类的初始化方法
        super().__init__(**kwargs)

    # 定义异步方法 map，将请求体映射为响应体
    async def map(self, request_body: dict) -> ExampleHttpResponse:
        """Map the request body to response body."""
        # 打印接收到的输入数值
        print(f"Receive input value: {request_body}")
        # 从请求体中获取姓名和年龄
        name = request_body.get("name")
        age = request_body.get("age")
        # 构造服务器响应消息
        server_res = f"Hello, {name}, your age is {age}"
        # 返回 ExampleHttpResponse 对象，包含服务器响应和原始请求体
        return ExampleHttpResponse(server_res=server_res, request_body=request_body)
class RequestBodyToDictOperator(MapOperator[CommonLLMHttpRequestBody, Dict[str, Any]]):
    """Request body to dict operator."""

    metadata = ViewMetadata(
        label=_("Request Body To Dict Operator"),
        name="request_body_to_dict_operator",
        category=OperatorCategory.COMMON,
        parameters=[
            Parameter.build_from(
                _("Prefix Key"),
                "prefix_key",
                str,
                optional=True,
                default=None,
                description=_(
                    "The prefix key of the dict, like 'message' or 'extra.info'"
                ),
            )
        ],
        inputs=[
            IOField.build_from(
                _("Request Body"),
                "request_body",
                CommonLLMHttpRequestBody,
                description=_("The request body of the API endpoint"),
            )
        ],
        outputs=[
            IOField.build_from(
                _("Response Body"),
                "response_body",
                dict,
                description=_("The response body of the API endpoint"),
            )
        ],
        description="Request body to dict operator",
    )
    # 定义一个将请求体映射为字典的操作符

    def __init__(self, prefix_key: Optional[str] = None, **kwargs):
        """Initialize a RequestBodyToDictOperator."""
        super().__init__(**kwargs)
        self._key = prefix_key
        # 初始化方法，设置前缀键用于数据字典

    async def map(self, request_body: CommonLLMHttpRequestBody) -> Dict[str, Any]:
        """Map the request body to response body."""
        # 将请求体映射为响应体的方法
        dict_value = model_to_dict(request_body)
        # 调用 model_to_dict 函数将请求体转换为字典

        if not self._key:
            return dict_value
            # 如果没有设置前缀键，直接返回转换后的字典

        else:
            keys = self._key.split(".")
            # 如果设置了前缀键，则按点分割键
            for k in keys:
                dict_value = dict_value[k]
                # 逐级获取字典中对应的值

            if not isinstance(dict_value, dict):
                raise ValueError(
                    f"Prefix key {self._key} is not a valid key of the request body"
                )
                # 如果最终的值不是字典类型，则抛出数值错误异常

            return dict_value
            # 返回按前缀键提取后的字典值


class UserInputParsedOperator(MapOperator[CommonLLMHttpRequestBody, Dict[str, Any]]):
    """User input parsed operator."""
    # 用户输入解析操作符，尚未实现具体内容
    # 创建视图元数据对象，用于描述用户输入解析操作符
    metadata = ViewMetadata(
        # 标签为“User Input Parsed Operator”，用于界面显示
        label=_("User Input Parsed Operator"),
        # 名称为"user_input_parsed_operator"
        name="user_input_parsed_operator",
        # 类别为常见操作符
        category=OperatorCategory.COMMON,
        # 参数列表，包括一个键为"key"的可选参数，默认为"user_input"
        parameters=[
            Parameter.build_from(
                # 参数名称为"Key"
                _("Key"),
                # 参数标识为"key"
                "key",
                # 参数类型为字符串
                str,
                # 可选参数，缺省值为"user_input"
                optional=True,
                # 描述为"字典的键，链接到'user_input'"
                default="user_input",
                description=_("The key of the dict, link 'user_input'"),
            )
        ],
        # 输入字段列表，包括一个名为"request_body"的字段，类型为CommonLLMHttpRequestBody
        inputs=[
            IOField.build_from(
                # 字段名称为"Request Body"
                _("Request Body"),
                # 字段标识为"request_body"
                "request_body",
                # 字段类型为CommonLLMHttpRequestBody
                CommonLLMHttpRequestBody,
                # 描述为"API端点的请求体"
                description=_("The request body of the API endpoint"),
            )
        ],
        # 输出字段列表，包括一个名为"user_input_dict"的字段，类型为字典
        outputs=[
            IOField.build_from(
                # 字段名称为"User Input Dict"
                _("User Input Dict"),
                # 字段标识为"user_input_dict"
                "user_input_dict",
                # 字段类型为字典
                dict,
                # 描述为"API端点的用户输入字典"
                description=_("The user input dict of the API endpoint"),
            )
        ],
        # 描述为"用户输入解析操作符，解析API端点的请求体并作为字典返回"
        description=_(
            "User input parsed operator, parse the user input from request body"
            " and return as a dict"
        ),
    )

    # 定义UserInputParsedOperator类的初始化方法，接受一个名为key的字符串参数，默认为"user_input"
    def __init__(self, key: str = "user_input", **kwargs):
        """Initialize a UserInputParsedOperator."""
        # 设置实例变量_key为传入的参数key
        self._key = key
        # 调用父类的初始化方法，传入额外的关键字参数
        super().__init__(**kwargs)

    # 定义异步方法map，将请求体CommonLLMHttpRequestBody映射为字典作为返回值
    async def map(self, request_body: CommonLLMHttpRequestBody) -> Dict[str, Any]:
        """Map the request body to response body."""
        # 返回一个字典，包含键为self._key，值为request_body.messages的内容
        return {self._key: request_body.messages}
class RequestedParsedOperator(MapOperator[CommonLLMHttpRequestBody, str]):
    """User input parsed operator."""

    metadata = ViewMetadata(
        label=_("Request Body Parsed To String Operator"),
        name="request_body_to_str__parsed_operator",
        category=OperatorCategory.COMMON,
        parameters=[
            Parameter.build_from(
                _("Key"),
                "key",
                str,
                optional=True,
                default="messages",
                description=_("The key of the dict, link 'user_input'"),
            )
        ],
        inputs=[
            IOField.build_from(
                _("Request Body"),
                "request_body",
                CommonLLMHttpRequestBody,
                description=_("The request body of the API endpoint"),
            )
        ],
        outputs=[
            IOField.build_from(
                _("User Input String"),
                "user_input_str",
                str,
                description=_("The user input dict of the API endpoint"),
            )
        ],
        description=_(
            "User input parsed operator, parse the user input from request body and "
            "return as a string"
        ),
    )

    def __init__(self, key: str = "user_input", **kwargs):
        """Initialize a UserInputParsedOperator."""
        self._key = key  # 设置实例变量 _key 用于存储用户指定的键名，默认为 "user_input"
        super().__init__(**kwargs)  # 调用父类的初始化方法

    async def map(self, request_body: CommonLLMHttpRequestBody) -> str:
        """Map the request body to response body."""
        dict_value = model_to_dict(request_body)  # 将请求体转换为字典
        if not self._key or self._key not in dict_value:
            raise ValueError(
                f"Prefix key {self._key} is not a valid key of the request body"
            )  # 如果指定的键名不在字典中，抛出值错误异常
        return dict_value[self._key]  # 返回指定键名对应的值
```