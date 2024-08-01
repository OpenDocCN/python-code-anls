# `.\DB-GPT-src\dbgpt\core\awel\util\chat_util.py`

```py
"""The utility functions for chatting with the DAG task."""

import json  # 导入用于处理 JSON 格式数据的模块
import traceback  # 导入用于获取异常堆栈信息的模块
from typing import Any, AsyncIterator, Dict, Optional  # 导入类型提示相关的模块

from ...interface.llm import ModelInferenceMetrics, ModelOutput  # 导入自定义模块中的类和函数
from ...schema.api import ChatCompletionResponseStreamChoice  # 导入自定义模块中的类
from ..operators.base import BaseOperator  # 导入自定义模块中的类
from ..trigger.http_trigger import CommonLLMHttpResponseBody  # 导入自定义模块中的类


def is_chat_flow_type(output_obj: Any, is_class: bool = False) -> bool:
    """Check whether the output object is a chat flow type."""
    if is_class:
        return output_obj in (str, CommonLLMHttpResponseBody, ModelOutput)  # 如果 is_class 为 True，判断 output_obj 是否属于特定类型
    else:
        chat_types = (str, CommonLLMHttpResponseBody)
        return isinstance(output_obj, chat_types)  # 如果 is_class 为 False，判断 output_obj 是否是指定类型之一的实例


async def safe_chat_with_dag_task(
    task: BaseOperator, request: Any, covert_to_str: bool = False
) -> ModelOutput:
    """Chat with the DAG task.

    Args:
        task (BaseOperator): The DAG task to be executed.
        request (Any): The request to be passed to the DAG task.
        covert_to_str (bool, optional): Whether to convert the output to string.

    Returns:
        ModelOutput: The model output, the result is not incremental.
    """
    try:
        finish_reason = None  # 初始化变量，用于记录任务完成的原因
        usage = None  # 初始化变量，用于记录任务的使用情况
        metrics = None  # 初始化变量，用于记录任务的指标数据
        error_code = 0  # 初始化变量，记录错误码，默认为 0 表示无错误
        text = ""  # 初始化变量，记录任务输出的文本内容，默认为空字符串
        async for output in safe_chat_stream_with_dag_task(
            task, request, False, covert_to_str=covert_to_str
        ):
            finish_reason = output.finish_reason  # 获取任务完成原因
            usage = output.usage  # 获取任务使用情况
            metrics = output.metrics  # 获取任务指标数据
            error_code = output.error_code  # 获取任务的错误码
            text = output.text  # 获取任务输出的文本内容
        return ModelOutput(
            error_code=error_code,
            text=text,
            metrics=metrics,
            usage=usage,
            finish_reason=finish_reason,
        )  # 返回整合后的任务输出对象
    except Exception as e:
        return ModelOutput(error_code=1, text=str(e), incremental=False)  # 捕获异常并返回包含错误信息的任务输出对象


async def safe_chat_stream_with_dag_task(
    task: BaseOperator, request: Any, incremental: bool, covert_to_str: bool = False
) -> AsyncIterator[ModelOutput]:
    """Chat with the DAG task.

    This function is similar to `chat_stream_with_dag_task`, but it will catch the
    exception and return the error message.

    Args:
        task (BaseOperator): The DAG task to be executed.
        request (Any): The request to be passed to the DAG task.
        incremental (bool): Whether the output is incremental.
        covert_to_str (bool, optional): Whether to convert the output to string.

    Yields:
        ModelOutput: The model output.
    """
    try:
        async for output in chat_stream_with_dag_task(
            task, request, incremental, covert_to_str=covert_to_str
        ):
            yield output  # 通过异步迭代产生聊天任务的输出
    except Exception as e:
        simple_error_msg = str(e)  # 获取异常的简单错误消息
        if not simple_error_msg:
            simple_error_msg = traceback.format_exc()  # 如果没有简单错误消息，则获取完整的异常堆栈信息
        yield ModelOutput(error_code=1, text=simple_error_msg, incremental=incremental)
        # 返回包含异常信息的任务输出对象，标记不是增量输出
    # 最终执行块：无论如何都会执行的代码块，通常用于清理操作或确保资源释放
    finally:
        # 如果任务有流操作并且有 DAG（有向无环图）
        if task.streaming_operator and task.dag:
            # 等待 DAG 结束后的操作，调用当前事件循环任务的 ID
            await task.dag._after_dag_end(task.current_event_loop_task_id)
# 检查 DAG 任务是否为服务器推送事件输出
def _is_sse_output(task: BaseOperator) -> bool:
    """Check whether the DAG task is a server-sent event output.

    Args:
        task (BaseOperator): The DAG task.

    Returns:
        bool: Whether the DAG task is a server-sent event output.
    """
    # 返回条件：任务的输出格式不为空且为"SSE"
    return task.output_format is not None and task.output_format.upper() == "SSE"


# 与 DAG 任务进行交互流
async def chat_stream_with_dag_task(
    task: BaseOperator, request: Any, incremental: bool, covert_to_str: bool = False
) -> AsyncIterator[ModelOutput]:
    """Chat with the DAG task.

    Args:
        task (BaseOperator): The DAG task to be executed.
        request (Any): The request to be passed to the DAG task.
        incremental (bool): Whether the output is incremental.
        covert_to_str (bool, optional): Whether to convert the output to string.

    Yields:
        ModelOutput: The model output.
    """
    # 检查任务是否为服务器推送事件输出
    is_sse = _is_sse_output(task)
    
    # 如果任务不是流操作符
    if not task.streaming_operator:
        try:
            # 调用任务的异步方法，传入请求，并获取结果
            result = await task.call(request)
            # 解析单个输出结果
            model_output = parse_single_output(
                result, is_sse, covert_to_str=covert_to_str
            )
            # 设置增量输出标志
            model_output.incremental = incremental
            # 生成器产出模型输出
            yield model_output
        except Exception as e:
            # 处理异常情况，获取简单错误消息或者完整的异常堆栈信息
            simple_error_msg = str(e)
            if not simple_error_msg:
                simple_error_msg = traceback.format_exc()
            # 生成器产出包含错误信息的模型输出
            yield ModelOutput(
                error_code=1, text=simple_error_msg, incremental=incremental
            )
    else:
        # 导入 OpenAIStreamingOutputOperator 类
        from dbgpt.model.utils.chatgpt_utils import OpenAIStreamingOutputOperator
        
        # 检查 task 是否为 OpenAIStreamingOutputOperator 的实例
        if OpenAIStreamingOutputOperator and isinstance(
            task, OpenAIStreamingOutputOperator
        ):
            # 初始化 full_text 为空字符串
            full_text = ""
            
            # 使用异步循环处理流式输出
            async for output in await task.call_stream(request):
                # 解析 OpenAI 输出结果
                model_output = parse_openai_output(output)
                
                # OpenAI 流式 API 的输出是增量的
                full_text += model_output.text  # 累加增量文本到 full_text
                model_output.incremental = incremental
                model_output.text = model_output.text if incremental else full_text
                
                # 生成模型输出
                yield model_output
                
                # 如果输出不成功，跳出循环
                if not model_output.success:
                    break
        else:
            # 初始化 full_text 和 previous_text 为空字符串
            full_text = ""
            previous_text = ""
            
            # 使用异步循环处理流式输出
            async for output in await task.call_stream(request):
                # 解析单个输出结果
                model_output = parse_single_output(
                    output, is_sse, covert_to_str=covert_to_str
                )
                model_output.incremental = incremental
                
                # 如果任务的增量输出为真
                if task.incremental_output:
                    # 输出是增量的，追加文本
                    full_text += model_output.text
                else:
                    # 输出不是增量的，上一个输出是完整的文本
                    full_text = model_output.text
                
                # 如果不是增量输出
                if not incremental:
                    # 返回完整文本
                    model_output.text = full_text
                else:
                    # 返回增量文本
                    delta_text = full_text[len(previous_text):]
                    previous_text = (
                        full_text
                        if len(full_text) > len(previous_text)
                        else previous_text
                    )
                    model_output.text = delta_text
                
                # 生成模型输出
                yield model_output
                
                # 如果输出不成功，跳出循环
                if not model_output.success:
                    break
def parse_single_output(
    output: Any, is_sse: bool, covert_to_str: bool = False
) -> ModelOutput:
    """解析单个输出。

    Args:
        output (Any): 要解析的输出。
        is_sse (bool): 输出是否为 SSE 格式。
        covert_to_str (bool, optional): 是否将输出转换为字符串。默认为 False。

    Returns:
        ModelOutput: 解析后的输出对象。
    """
    # 初始化可选变量
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    metrics: Optional[ModelInferenceMetrics] = None

    # 根据输出的类型进行不同的解析逻辑
    if output is None:
        error_code = 1
        text = "The output is None!"
    elif isinstance(output, str):
        if is_sse:
            # 解析 SSE 数据
            sse_output = parse_sse_data(output)
            if sse_output is None:
                error_code = 1
                text = "The output is not a SSE format"
            else:
                error_code = 0
                text = sse_output
        else:
            error_code = 0
            text = output
    elif isinstance(output, ModelOutput):
        # 复用 ModelOutput 的属性
        error_code = output.error_code
        text = output.text
        finish_reason = output.finish_reason
        usage = output.usage
        metrics = output.metrics
    elif isinstance(output, CommonLLMHttpResponseBody):
        # 使用 CommonLLMHttpResponseBody 的属性
        error_code = output.error_code
        text = output.text
    elif isinstance(output, dict):
        # 转换字典为 JSON 字符串
        error_code = 0
        text = json.dumps(output, ensure_ascii=False)
    elif covert_to_str:
        # 转换输出为字符串
        error_code = 0
        text = str(output)
    else:
        # 未知类型的输出
        error_code = 1
        text = f"The output is not a valid format({type(output)})"

    # 返回解析后的 ModelOutput 对象
    return ModelOutput(
        error_code=error_code,
        text=text,
        finish_reason=finish_reason,
        usage=usage,
        metrics=metrics,
    )


def parse_openai_output(output: Any) -> ModelOutput:
    """解析 OpenAI 的输出。

    Args:
        output (Any): 要解析的输出。必须是流格式。

    Returns:
        ModelOutput: 解析后的输出对象。
    """
    text = ""

    # 检查输出是否为字符串类型
    if not isinstance(output, str):
        return ModelOutput(
            error_code=1,
            text="The output is not a stream format",
        )

    # 检查特定的结束标志或格式开始标志
    if output.strip() == "data: [DONE]" or output.strip() == "data:[DONE]":
        return ModelOutput(error_code=0, text="")

    if not output.startswith("data:"):
        return ModelOutput(
            error_code=1,
            text="The output is not a stream format",
        )

    # 解析 SSE 格式的数据
    sse_output = parse_sse_data(output)
    if sse_output is None:
        return ModelOutput(error_code=1, text="The output is not a SSE format")

    # 尝试解析 JSON 数据
    json_data = sse_output.strip()
    try:
        dict_data = json.loads(json_data)
    except Exception as e:
        return ModelOutput(
            error_code=1,
            text=f"Invalid JSON data: {json_data}, {e}",
        )
    # 如果字典数据中没有键"choices"
    if "choices" not in dict_data:
        # 返回带有错误码和错误文本的模型输出对象
        return ModelOutput(
            error_code=1,
            text=dict_data.get("text", "Unknown error"),
        )
    
    # 从字典数据中获取键"choices"对应的值
    choices = dict_data["choices"]
    
    # 初始化完成原因为 None 的可选字符串
    finish_reason: Optional[str] = None
    
    # 如果 choices 非空列表
    if choices:
        # 获取第一个选择项
        choice = choices[0]
        
        # 使用 ChatCompletionResponseStreamChoice 类初始化 delta_data 对象
        delta_data = ChatCompletionResponseStreamChoice(**choice)
        
        # 如果 delta_data 的 delta 属性有内容
        if delta_data.delta.content:
            # 将文本设为 delta_data 的 delta.content 属性的内容
            text = delta_data.delta.content
        
        # 将完成原因设为 delta_data 的 finish_reason 属性的内容
        finish_reason = delta_data.finish_reason
    
    # 返回模型输出对象，包括错误码、文本和完成原因
    return ModelOutput(error_code=0, text=text, finish_reason=finish_reason)
# 解析服务器发送事件（SSE）数据，提取数据部分并返回

def parse_sse_data(output: str) -> Optional[str]:
    r"""Parse the SSE data.

    Just keep the data part.

    Examples:
        .. code-block:: python

            from dbgpt.core.awel.util.chat_util import parse_sse_data

            assert parse_sse_data("data: [DONE]") == "[DONE]"
            assert parse_sse_data("data:[DONE]") == "[DONE]"
            assert parse_sse_data("data: Hello") == "Hello"
            assert parse_sse_data("data: Hello\n") == "Hello"
            assert parse_sse_data("data: Hello\r\n") == "Hello"
            assert parse_sse_data("data: Hi, what's up?") == "Hi, what's up?"

    Args:
        output (str): The output.

    Returns:
        Optional[str]: The parsed data.
    """
    # 检查输出是否以"data:"开头
    if output.startswith("data:"):
        # 去除字符串两端的空白字符
        output = output.strip()
        # 如果字符串以"data: "开头，去除"data: "部分
        if output.startswith("data: "):
            output = output[6:]
        else:
            # 否则，去除"data:"部分
            output = output[5:]

        # 返回提取出的数据部分
        return output
    else:
        # 如果输出不以"data:"开头，则返回None
        return None
```