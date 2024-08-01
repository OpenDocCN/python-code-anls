# `.\DB-GPT-src\dbgpt\core\interface\operators\llm_operator.py`

```py
"""The LLM operator."""

import dataclasses
from abc import ABC
from typing import Any, AsyncIterator, Dict, List, Optional, Union, cast

from dbgpt._private.pydantic import BaseModel  # 导入基础模型
from dbgpt.core.awel import (
    BaseOperator,  # 导入基础运算符
    BranchFunc,  # 分支函数
    BranchJoinOperator,  # 分支连接运算符
    BranchOperator,  # 分支运算符
    CommonLLMHttpRequestBody,  # 常见LLM HTTP请求体
    CommonLLMHttpResponseBody,  # 常见LLM HTTP响应体
    DAGContext,  # DAG上下文
    JoinOperator,  # 连接运算符
    MapOperator,  # 映射运算符
    StreamifyAbsOperator,  # 流转换抽象运算符
    TransformStreamAbsOperator,  # 流转换抽象运算符
)
from dbgpt.core.awel.flow import (
    IOField,  # 输入输出字段
    OperatorCategory,  # 运算符类别
    OperatorType,  # 运算符类型
    Parameter,  # 参数
    ViewMetadata,  # 视图元数据
)
from dbgpt.core.interface.llm import (
    LLMClient,  # LLM客户端
    ModelOutput,  # 模型输出
    ModelRequest,  # 模型请求
    ModelRequestContext,  # 模型请求上下文
)
from dbgpt.core.interface.message import ModelMessage  # 模型消息
from dbgpt.util.function_utils import rearrange_args_by_type  # 根据类型重新排列参数
from dbgpt.util.i18n_utils import _  # 国际化工具

RequestInput = Union[
    ModelRequest,  # 模型请求
    str,  # 字符串
    Dict[str, Any],  # 字典类型
    BaseModel,  # 基础模型
    ModelMessage,  # 模型消息
    List[ModelMessage],  # 模型消息列表
]


class RequestBuilderOperator(MapOperator[RequestInput, ModelRequest]):
    """
    Build the model request from the input value.
    
    This class represents an operator that transforms various types of inputs 
    into a ModelRequest object. It includes metadata about its purpose, input 
    and output specifications, and optional parameters.
    """

    metadata = ViewMetadata(
        label=_("Build Model Request"),  # 视图标签，构建模型请求
        name="request_builder_operator",  # 名称，请求构建运算符
        category=OperatorCategory.COMMON,  # 类别，通用类别
        description=_("Build the model request from the http request body."),  # 描述，从HTTP请求体构建模型请求
        parameters=[
            Parameter.build_from(
                _("Default Model Name"),  # 默认模型名称
                "model",  # 参数名，模型
                str,  # 参数类型，字符串
                optional=True,  # 可选参数
                default=None,  # 默认值为None
                description=_("The model name of the model request."),  # 描述，模型请求的模型名称
            ),
            Parameter.build_from(
                _("Temperature"),  # 温度
                "temperature",  # 参数名，温度
                float,  # 参数类型，浮点数
                optional=True,  # 可选参数
                default=None,  # 默认值为None
                description=_("The temperature of the model request."),  # 描述，模型请求的温度
            ),
            Parameter.build_from(
                _("Max New Tokens"),  # 最大新标记数
                "max_new_tokens",  # 参数名，最大新标记数
                int,  # 参数类型，整数
                optional=True,  # 可选参数
                default=None,  # 默认值为None
                description=_("The max new tokens of the model request."),  # 描述，模型请求的最大新标记数
            ),
            Parameter.build_from(
                _("Context Length"),  # 上下文长度
                "context_len",  # 参数名，上下文长度
                int,  # 参数类型，整数
                optional=True,  # 可选参数
                default=None,  # 默认值为None
                description=_("The context length of the model request."),  # 描述，模型请求的上下文长度
            ),
        ],
        inputs=[
            IOField.build_from(
                _("Request Body"),  # 请求体
                "input_value",  # 输入值名，输入值
                CommonLLMHttpRequestBody,  # 输入值类型，常见LLM HTTP请求体
                description=_("The input value of the operator."),  # 描述，运算符的输入值
            ),
        ],
        outputs=[
            IOField.build_from(
                _("Model Request"),  # 模型请求
                "output_value",  # 输出值名，输出值
                ModelRequest,  # 输出值类型，模型请求
                description=_("The output value of the operator."),  # 描述，运算符的输出值
            ),
        ],
    )
    # 初始化函数，用于创建一个请求构建器操作对象
    def __init__(
        self,
        model: Optional[str] = None,        # 模型名称，可选参数，默认为 None
        temperature: Optional[float] = None,    # 温度参数，控制生成文本的多样性，可选参数，默认为 None
        max_new_tokens: Optional[int] = None,   # 最大新标记数，限制生成的最大标记数量，可选参数，默认为 None
        context_len: Optional[int] = None,      # 上下文长度，控制模型生成文本时的上下文长度，可选参数，默认为 None
        **kwargs,    # 其他关键字参数，用于接收除上述参数外的其他参数
    ):
        """Create a new request builder operator."""
        self._model = model    # 将传入的模型名称赋值给对象的 _model 属性
        self._temperature = temperature    # 将传入的温度参数赋值给对象的 _temperature 属性
        self._max_new_tokens = max_new_tokens    # 将传入的最大新标记数赋值给对象的 _max_new_tokens 属性
        self._context_len = context_len    # 将传入的上下文长度赋值给对象的 _context_len 属性
        super().__init__(**kwargs)    # 调用父类的初始化方法，传入其他所有未明确列出的关键字参数
class MergedRequestBuilderOperator(JoinOperator[ModelRequest]):
    """Build the model request from the input value."""

    metadata = ViewMetadata(
        label=_("Merge Model Request Messages"),
        name="merged_request_builder_operator",
        category=OperatorCategory.COMMON,
        description=_("Merge the model request from the input value."),
        parameters=[],
        inputs=[
            IOField.build_from(
                _("Model Request"),
                "model_request",
                ModelRequest,
                description=_("The model request of upstream."),
            ),
            IOField.build_from(
                _("Model messages"),
                "messages",
                ModelMessage,
                description=_("The model messages of upstream."),
                is_list=True,
            ),
        ],
        outputs=[
            IOField.build_from(
                _("Model Request"),
                "output_value",
                ModelRequest,
                description=_("The output value of the operator."),
            ),
        ],
    )

    def __init__(self, **kwargs):
        """Create a new request builder operator."""
        # 调用父类初始化方法，并指定合并函数为 merge_func
        super().__init__(combine_function=self.merge_func, **kwargs)

    @rearrange_args_by_type
    def merge_func(
        self, model_request: ModelRequest, messages: List[ModelMessage]
    ) -> ModelRequest:
        """Merge the model request with the messages."""
        # 将输入的模型消息合并到模型请求中
        model_request.messages = messages
        return model_request


class BaseLLM:
    """The abstract operator for a LLM."""

    SHARE_DATA_KEY_MODEL_NAME = "share_data_key_model_name"
    SHARE_DATA_KEY_MODEL_OUTPUT = "share_data_key_model_output"

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Create a new LLM operator."""
        # 初始化 LLM 操作类，接受一个可选的 LLM 客户端对象
        self._llm_client = llm_client

    @property
    def llm_client(self) -> LLMClient:
        """Return the LLM client."""
        # 返回 LLM 客户端对象，如果未设置则引发 ValueError 异常
        if not self._llm_client:
            raise ValueError("llm_client is not set")
        return self._llm_client

    async def save_model_output(
        self, current_dag_context: DAGContext, model_output: ModelOutput
    ) -> None:
        """Save the model output to the share data."""
        # 异步保存模型输出到共享数据中
        await current_dag_context.save_to_share_data(
            self.SHARE_DATA_KEY_MODEL_OUTPUT, model_output
        )


class BaseLLMOperator(BaseLLM, MapOperator[ModelRequest, ModelOutput], ABC):
    """The operator for a LLM.

    Args:
        llm_client (LLMClient, optional): The LLM client. Defaults to None.

    This operator will generate a no streaming response.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        """Create a new LLM operator."""
        # 调用父类初始化方法，设置 LLM 客户端，并调用 MapOperator 初始化
        super().__init__(llm_client=llm_client)
        MapOperator.__init__(self, **kwargs)
    async def map(self, request: ModelRequest) -> ModelOutput:
        """Generate the model output.

        Args:
            request (ModelRequest): The model request object containing input data.

        Returns:
            ModelOutput: The generated model output based on the input request.
        """
        # 将请求中的模型名称保存到共享数据中
        await self.current_dag_context.save_to_share_data(
            self.SHARE_DATA_KEY_MODEL_NAME, request.model
        )
        # 使用LLM客户端生成模型输出
        model_output = await self.llm_client.generate(request)
        # 将模型输出保存到当前DAG上下文中
        await self.save_model_output(self.current_dag_context, model_output)
        # 返回生成的模型输出
        return model_output
class BaseStreamingLLMOperator(
    BaseLLM, StreamifyAbsOperator[ModelRequest, ModelOutput], ABC
):
    """The streaming operator for a LLM.

    Args:
        llm_client (LLMClient, optional): The LLM client. Defaults to None.

    This operator will generate streaming response.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        """Create a streaming operator for a LLM.

        Args:
            llm_client (LLMClient, optional): The LLM client. Defaults to None.
        """
        super().__init__(llm_client=llm_client)  # 调用父类 BaseLLM 的初始化方法，传入 llm_client 参数
        BaseOperator.__init__(self, **kwargs)  # 调用 BaseOperator 的初始化方法，传入其他参数

    async def streamify(  # type: ignore
        self, request: ModelRequest  # type: ignore
    ) -> AsyncIterator[ModelOutput]:  # type: ignore
        """Streamify the request."""
        await self.current_dag_context.save_to_share_data(
            self.SHARE_DATA_KEY_MODEL_NAME, request.model
        )  # 将请求中的模型名称保存到共享数据中
        model_output = None
        async for output in self.llm_client.generate_stream(request):  # type: ignore
            model_output = output  # 将生成的流输出存储到 model_output 变量中
            yield output  # 返回每个生成的流输出
        if model_output:
            await self.save_model_output(self.current_dag_context, model_output)  # 如果有生成的流输出，保存到当前 DAG 上下文中


class LLMBranchOperator(BranchOperator[ModelRequest, ModelRequest]):
    """Branch operator for LLM.

    This operator will branch the workflow based on
    the stream flag of the request.
    """

    metadata = ViewMetadata(
        label=_("LLM Branch Operator"),
        name="llm_branch_operator",
        category=OperatorCategory.LLM,
        operator_type=OperatorType.BRANCH,
        description=_("Branch the workflow based on the stream flag of the request."),
        parameters=[],
        inputs=[
            IOField.build_from(
                _("Model Request"),
                "input_value",
                ModelRequest,
                description=_("The input value of the operator."),
            ),
        ],
        outputs=[
            IOField.build_from(
                _("Streaming Model Request"),
                "streaming_request",
                ModelRequest,
                description=_("The streaming request, to streaming Operator."),
            ),
            IOField.build_from(
                _("Non-Streaming Model Request"),
                "no_streaming_request",
                ModelRequest,
                description=_("The non-streaming request, to non-streaming Operator."),
            ),
        ],
    )

    def __init__(
        self,
        stream_task_name: Optional[str] = None,
        no_stream_task_name: Optional[str] = None,
        **kwargs,
    ):
        """Create a new LLM branch operator.

        Args:
            stream_task_name (str): The name of the streaming task.
            no_stream_task_name (str): The name of the non-streaming task.
        """
        super().__init__(**kwargs)  # 调用父类 BranchOperator 的初始化方法，传入其他参数
        self._stream_task_name = stream_task_name  # 设置流任务的名称
        self._no_stream_task_name = no_stream_task_name  # 设置非流任务的名称
    async def branches(
        self,
    ) -> Dict[BranchFunc[ModelRequest], Union[BaseOperator, str]]:
        """Return a dict of branch function and task name.

        Returns:
            Dict[BranchFunc[ModelRequest], str]: A dictionary mapping branch functions
            to task names. The branch function is a predicate function that takes a
            ModelRequest object and returns a boolean indicating whether to run the
            corresponding task. The value associated with the branch function is the
            name of the task to be executed.
        """
        # 如果已经设置了流式任务名称和非流式任务名称，则使用已设置的名称
        if self._stream_task_name and self._no_stream_task_name:
            stream_task_name = self._stream_task_name
            no_stream_task_name = self._no_stream_task_name
        else:
            # 否则，初始化为空字符串
            stream_task_name = ""
            no_stream_task_name = ""
            # 遍历下游节点
            for node in self.downstream:
                # 将节点转换为基础操作器对象
                task = cast(BaseOperator, node)
                # 如果节点是流式操作器
                if task.streaming_operator:
                    # 设置流式任务名称
                    stream_task_name = node.node_name
                else:
                    # 设置非流式任务名称
                    no_stream_task_name = node.node_name

        async def check_stream_true(r: ModelRequest) -> bool:
            # 如果 ModelRequest 中 stream 属性为 True，则返回 True，表示运行流式任务
            # 否则返回 False，表示运行非流式任务
            return r.stream

        # 返回一个字典，将判断函数与对应的任务名称关联起来
        return {
            check_stream_true: stream_task_name,
            lambda x: not x.stream: no_stream_task_name,
        }
class ModelOutput2CommonResponseOperator(
    MapOperator[ModelOutput, CommonLLMHttpResponseBody]
):
    """Map the model output to the common response body."""

    metadata = ViewMetadata(
        label=_("Map Model Output to Common Response Body"),
        name="model_output_2_common_response_body_operator",
        category=OperatorCategory.COMMON,
        description=_("Map the model output to the common response body."),
        parameters=[],
        inputs=[
            IOField.build_from(
                _("Model Output"),
                "input_value",
                ModelOutput,
                description=_("The input value of the operator."),
            ),
        ],
        outputs=[
            IOField.build_from(
                _("Common Response Body"),
                "output_value",
                CommonLLMHttpResponseBody,
                description=_("The output value of the operator."),
            ),
        ],
    )

    def __int__(self, **kwargs):
        """Create a new operator."""
        super().__init__(**kwargs)

    async def map(self, input_value: ModelOutput) -> CommonLLMHttpResponseBody:
        """Map the model output to the common response body."""
        metrics = input_value.metrics.to_dict() if input_value.metrics else None
        return CommonLLMHttpResponseBody(
            text=input_value.text,
            error_code=input_value.error_code,
            metrics=metrics,
        )



class CommonStreamingOutputOperator(TransformStreamAbsOperator[ModelOutput, str]):
    """The Common Streaming Output Operator.

    Transform model output to the string output to show in DB-GPT chat flow page.
    """

    output_format = "SSE"

    metadata = ViewMetadata(
        label=_("Common Streaming Output Operator"),
        name="common_streaming_output_operator",
        operator_type=OperatorType.TRANSFORM_STREAM,
        category=OperatorCategory.OUTPUT_PARSER,
        description=_("The common streaming LLM operator, for chat flow."),
        parameters=[],
        inputs=[
            IOField.build_from(
                _("Upstream Model Output"),
                "output_iter",
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
                    "The model output after transform to common stream format"
                ),
            )
        ],
    )
    async def transform_stream(self, output_iter: AsyncIterator[ModelOutput]):
        """Transform upstream output iter to string foramt."""
        async for model_output in output_iter:
            # 检查模型输出的错误代码，如果不为0则生成包含错误信息的数据流
            if model_output.error_code != 0:
                error_msg = (
                    f"[ERROR](error_code: {model_output.error_code}): "
                    f"{model_output.text}"
                )
                yield f"data:{error_msg}"
                return  # 返回生成的错误消息后结束函数执行

            # 将模型输出的文本中的乱码字符替换为空字符
            decoded_unicode = model_output.text.replace("\ufffd", "")
            # 生成包含解码后文本的数据流，并包含两个换行符
            yield f"data:{decoded_unicode}\n\n"
class StringOutput2ModelOutputOperator(MapOperator[str, ModelOutput]):
    """Map String to ModelOutput."""

    # 视图元数据，定义了操作符的各种属性
    metadata = ViewMetadata(
        label=_("Map String to ModelOutput"),  # 操作符的显示名称
        name="string_2_model_output_operator",  # 操作符的内部名称
        category=OperatorCategory.COMMON,  # 操作符所属的类别
        description=_("Map String to ModelOutput."),  # 操作符的描述信息
        parameters=[],  # 操作符的参数列表为空
        inputs=[  # 输入字段的定义
            IOField.build_from(
                _("String"),  # 输入字段的显示名称
                "input_value",  # 输入字段的内部名称
                str,  # 输入字段的数据类型为字符串
                description=_("The input value of the operator."),  # 输入字段的描述信息
            ),
        ],
        outputs=[  # 输出字段的定义
            IOField.build_from(
                _("Model Output"),  # 输出字段的显示名称
                "input_value",  # 输出字段的内部名称
                ModelOutput,  # 输出字段的数据类型为ModelOutput类
                description=_("The input value of the operator."),  # 输出字段的描述信息
            ),
        ],
    )

    def __int__(self, **kwargs):
        """Create a new operator."""
        super().__init__(**kwargs)  # 调用父类构造函数初始化操作符

    async def map(self, input_value: str) -> ModelOutput:
        """Map the model output to the common response body."""
        return ModelOutput(
            text=input_value,  # 使用输入字符串创建ModelOutput对象的text字段
            error_code=500,  # 设置错误码为500
        )


class LLMBranchJoinOperator(BranchJoinOperator[ModelOutput]):
    """The LLM Branch Join Operator.

    Decide which output to keep(streaming or non-streaming).
    """

    streaming_operator = True  # 标识这是一个流操作符
    metadata = ViewMetadata(
        label=_("LLM Branch Join Operator"),  # 操作符的显示名称
        name="llm_branch_join_operator",  # 操作符的内部名称
        category=OperatorCategory.LLM,  # 操作符所属的类别
        operator_type=OperatorType.JOIN,  # 操作符的类型为JOIN
        description=_("Just keep the first non-empty output."),  # 操作符的描述信息
        parameters=[],  # 操作符的参数列表为空
        inputs=[  # 输入字段的定义
            IOField.build_from(
                _("Streaming Model Output"),  # 输入字段的显示名称
                "stream_output",  # 输入字段的内部名称
                ModelOutput,  # 输入字段的数据类型为ModelOutput类
                is_list=True,  # 输入字段是一个ModelOutput对象列表
                description=_("The streaming output."),  # 输入字段的描述信息
            ),
            IOField.build_from(
                _("Non-Streaming Model Output"),  # 输入字段的显示名称
                "not_stream_output",  # 输入字段的内部名称
                ModelOutput,  # 输入字段的数据类型为ModelOutput类
                description=_("The non-streaming output."),  # 输入字段的描述信息
            ),
        ],
        outputs=[  # 输出字段的定义
            IOField.build_from(
                _("Model Output"),  # 输出字段的显示名称
                "output_value",  # 输出字段的内部名称
                ModelOutput,  # 输出字段的数据类型为ModelOutput类
                is_list=True,  # 输出字段是一个ModelOutput对象列表
                description=_("The output value of the operator."),  # 输出字段的描述信息
            ),
        ],
    )

    def __init__(self, **kwargs):
        """Create a new LLM branch join operator."""
        super().__init__(**kwargs)  # 调用父类构造函数初始化操作符


class StringBranchJoinOperator(BranchJoinOperator[str]):
    """The String Branch Join Operator.

    Decide which output to keep(streaming or non-streaming).
    """

    streaming_operator = True  # 标识这是一个流操作符
    metadata = ViewMetadata(
        label=_("String Branch Join Operator"),  # 设置操作符的显示标签，用于界面显示
        name="string_branch_join_operator",  # 设置操作符的内部名称，用于标识和引用
        category=OperatorCategory.COMMON,  # 设置操作符所属的类别，这里是通用类别
        operator_type=OperatorType.JOIN,  # 设置操作符的类型，这是一个连接类型的操作符
        description=_("Just keep the first non-empty output."),  # 设置操作符的简要描述
        parameters=[],  # 设置操作符的参数列表为空
        inputs=[  # 设置操作符的输入字段列表
            IOField.build_from(
                _("Streaming String Output"),  # 输入字段的显示标签，流式字符串输出
                "stream_output",  # 输入字段的内部名称
                str,  # 输入字段的数据类型，这里是字符串
                is_list=True,  # 输入字段是否为列表形式
                description=_("The streaming output."),  # 输入字段的详细描述
            ),
            IOField.build_from(
                _("Non-Streaming String Output"),  # 输入字段的显示标签，非流式字符串输出
                "not_stream_output",  # 输入字段的内部名称
                str,  # 输入字段的数据类型，这里是字符串
                description=_("The non-streaming output."),  # 输入字段的详细描述
            ),
        ],
        outputs=[  # 设置操作符的输出字段列表
            IOField.build_from(
                _("String Output"),  # 输出字段的显示标签，字符串输出
                "output_value",  # 输出字段的内部名称
                str,  # 输出字段的数据类型，这里是字符串
                is_list=True,  # 输出字段是否为列表形式
                description=_("The output value of the operator."),  # 输出字段的详细描述
            ),
        ],
    )
    
    def __init__(self, **kwargs):
        """Create a new LLM branch join operator."""
        super().__init__(**kwargs)
```