# `.\DB-GPT-src\dbgpt\client\flow.py`

```py
"""this module contains the flow client functions."""

# 从 typing 模块导入需要的类型注解
from typing import Any, Callable, Dict, List

# 导入异步 HTTP 客户端 AsyncClient
from httpx import AsyncClient

# 导入 FlowPanel 类
from dbgpt.core.awel.flow.flow_factory import FlowPanel

# 导入 Result 类型
from dbgpt.core.schema.api import Result

# 导入本地的 Client 和 ClientException 类
from .client import Client, ClientException


async def create_flow(client: Client, flow: FlowPanel) -> FlowPanel:
    """Create a new flow.

    Args:
        client (Client): The dbgpt client.
        flow (FlowPanel): The flow panel.
    """
    try:
        # 发起 POST 请求以创建流程
        res = await client.get("/awel/flows", flow.to_dict())
        # 解析响应体为 Result 对象
        result: Result = res.json()
        if result["success"]:
            # 如果成功，返回包含新流程数据的 FlowPanel 对象
            return FlowPanel(**result["data"])
        else:
            # 如果失败，抛出 ClientException 异常
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获并重新抛出异常，添加自定义错误消息
        raise ClientException(f"Failed to create flow: {e}")


async def update_flow(client: Client, flow: FlowPanel) -> FlowPanel:
    """Update a flow.

    Args:
        client (Client): The dbgpt client.
        flow (FlowPanel): The flow panel.
    Returns:
        FlowPanel: The flow panel.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发起 PUT 请求以更新流程
        res = await client.put("/awel/flows", flow.to_dict())
        # 解析响应体为 Result 对象
        result: Result = res.json()
        if result["success"]:
            # 如果成功，返回包含更新后流程数据的 FlowPanel 对象
            return FlowPanel(**result["data"])
        else:
            # 如果失败，抛出 ClientException 异常
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获并重新抛出异常，添加自定义错误消息
        raise ClientException(f"Failed to update flow: {e}")


async def delete_flow(client: Client, flow_id: str) -> FlowPanel:
    """
    Delete a flow.

    Args:
        client (Client): The dbgpt client.
        flow_id (str): The flow id.
    Returns:
        FlowPanel: The flow panel.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发起 DELETE 请求以删除流程
        res = await client.delete("/awel/flows/" + flow_id)
        # 解析响应体为 Result 对象
        result: Result = res.json()
        if result["success"]:
            # 如果成功，返回包含被删除流程数据的 FlowPanel 对象
            return FlowPanel(**result["data"])
        else:
            # 如果失败，抛出 ClientException 异常
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获并重新抛出异常，添加自定义错误消息
        raise ClientException(f"Failed to delete flow: {e}")


async def get_flow(client: Client, flow_id: str) -> FlowPanel:
    """
    Get a flow.

    Args:
        client (Client): The dbgpt client.
        flow_id (str): The flow id.
    Returns:
        FlowPanel: The flow panel.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发起 GET 请求以获取流程
        res = await client.get("/awel/flows/" + flow_id)
        # 解析响应体为 Result 对象
        result: Result = res.json()
        if result["success"]:
            # 如果成功，返回包含请求的流程数据的 FlowPanel 对象
            return FlowPanel(**result["data"])
        else:
            # 如果失败，抛出 ClientException 异常
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获并重新抛出异常，添加自定义错误消息
        raise ClientException(f"Failed to get flow: {e}")


async def list_flow(
    client: Client, name: str | None = None, uid: str | None = None
) -> List[FlowPanel]:
    """
    List flows.

    """
    # 该函数尚未实现内容，保留了函数框架和参数列表
    pass
    Args:
        client (Client): The dbgpt client.  # 传入的客户端对象，用于发送请求
        name (str): The name of the flow.  # 流的名称，用于请求指定名称的流数据
        uid (str): The uid of the flow.  # 流的唯一标识符，用于请求指定UID的流数据
    Returns:
        List[FlowPanel]: The list of flow panels.  # 返回的结果是一个FlowPanel对象列表，表示流的面板信息
    Raises:
        ClientException: If the request failed.  # 如果请求失败，则抛出ClientException异常
    """
    try:
        # 使用客户端对象向指定URL发送GET请求，传入名称和UID作为参数
        res = await client.get("/awel/flows", **{"name": name, "uid": uid})
        # 解析响应内容为Result对象
        result: Result = res.json()
        # 检查请求是否成功
        if result["success"]:
            # 如果成功，将每个流项目转换为FlowPanel对象，并组成列表返回
            return [FlowPanel(**flow) for flow in result["data"]["items"]]
        else:
            # 如果请求不成功，抛出ClientException异常，传入错误代码和原因
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，如果发生异常则抛出ClientException异常，包含错误信息
        raise ClientException(f"Failed to list flows: {e}")
async def run_flow_cmd(
    client: Client,
    name: str | None = None,
    uid: str | None = None,
    data: Dict[str, Any] | None = None,
    non_streaming_callback: Callable[[str], None] | None = None,
    streaming_callback: Callable[[str], None] | None = None,
) -> None:
    """
    Run flows.

    Args:
        client (Client): The dbgpt client.
        name (str): The name of the flow.
        uid (str): The uid of the flow.
        data (Dict[str, Any]): The data to run the flow.
        non_streaming_callback (Callable[[str], None]): The non-streaming callback.
        streaming_callback (Callable[[str], None]): The streaming callback.
    Returns:
        None
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发起异步 GET 请求以获取流信息
        res = await client.get("/awel/flows", **{"name": name, "uid": uid})
        # 将响应内容解析为 JSON 格式
        result: Result = res.json()
        # 如果请求不成功，抛出客户端异常
        if not result["success"]:
            raise ClientException("Flow not found with the given name or uid")
        # 获取返回的流数据列表
        flows = result["data"]["items"]
        # 如果没有找到对应的流，抛出客户端异常
        if not flows:
            raise ClientException("Flow not found with the given name or uid")
        # 如果找到多个流，抛出客户端异常
        if len(flows) > 1:
            raise ClientException("More than one flow found")
        # 获取第一个流对象
        flow = flows[0]
        # 根据流对象创建流面板
        flow_panel = FlowPanel(**flow)
        # 获取流的元数据
        metadata = flow.get("metadata")
        # 调用内部函数以触发流操作
        await _run_flow_trigger(
            client,
            flow_panel,
            metadata,
            data,
            non_streaming_callback=non_streaming_callback,
            streaming_callback=streaming_callback,
        )
    except Exception as e:
        # 如果任何步骤出现异常，抛出客户端异常并附带错误信息
        raise ClientException(f"Failed to run flows: {e}")


async def _run_flow_trigger(
    client: Client,
    flow: FlowPanel,
    metadata: Dict[str, Any] | None = None,
    data: Dict[str, Any] | None = None,
    non_streaming_callback: Callable[[str], None] | None = None,
    streaming_callback: Callable[[str], None] | None = None,
):
    # 如果没有流的元数据，抛出客户端异常
    if not metadata:
        raise ClientException("No AWEL flow metadata found")
    # 如果元数据中没有触发器定义，抛出客户端异常
    if "triggers" not in metadata:
        raise ClientException("No triggers found in AWEL flow metadata")
    # 获取触发器列表
    triggers = metadata["triggers"]
    # 如果找到多个触发器，抛出客户端异常
    if len(triggers) > 1:
        raise ClientException("More than one trigger found")
    # 获取第一个触发器对象
    trigger = triggers[0]
    # 获取元数据中的 SSE 输出设置，默认为 False
    sse_output = metadata.get("sse_output", False)
    # 获取元数据中的流输出设置，默认为 False
    streaming_output = metadata.get("streaming_output", False)
    # 获取触发器类型
    trigger_type = trigger["trigger_type"]
    # 如果触发器类型为 "http"，则执行以下操作
    if trigger_type == "http":
        # 从触发器对象中获取支持的请求方法列表
        methods = trigger["methods"]
        # 如果方法列表为空，则默认使用 "GET" 方法
        if not methods:
            method = "GET"
        else:
            # 否则，使用方法列表中的第一个方法
            method = methods[0]
        # 获取触发器对象中的路径信息
        path = trigger["path"]
        # 获取客户端对象的基础 URL
        base_url = client._base_url()
        # 构建完整的请求 URL，结合基础 URL 和路径信息
        req_url = f"{base_url}{path}"
        
        # 如果设置了流式输出标志
        if streaming_output:
            # 调用异步流式请求处理函数，向服务器发起请求
            await _call_stream_request(
                client._http_client,
                method,
                req_url,
                sse_output,
                data,
                streaming_callback,
            )
        # 如果设置了非流式回调函数
        elif non_streaming_callback:
            # 调用异步非流式请求处理函数，向服务器发起请求
            await _call_non_stream_request(
                client._http_client, method, req_url, data, non_streaming_callback
            )
    # 如果触发器类型不是 "http"
    else:
        # 抛出客户端异常，指明无效的触发器类型
        raise ClientException(f"Invalid trigger type: {trigger_type}")
# 定义一个异步函数，用于执行非流式请求
async def _call_non_stream_request(
    http_client: AsyncClient,
    method: str,
    base_url: str,
    data: Dict[str, Any] | None = None,
    non_streaming_callback: Callable[[str], None] | None = None,
):
    import httpx  # 导入httpx库

    # 初始化请求参数字典
    kwargs: Dict[str, Any] = {"url": base_url, "method": method}
    # 根据请求方法选择适当的数据格式
    if method in ["POST", "PUT"]:
        kwargs["json"] = data
    else:
        kwargs["params"] = data
    # 发起异步请求
    response = await http_client.request(**kwargs)
    # 读取响应的字节流内容
    bytes_response_content = await response.aread()
    # 如果响应状态码不为200，处理错误信息
    if response.status_code != 200:
        str_error_message = ""
        error_message = await response.aread()
        if error_message:
            str_error_message = error_message.decode("utf-8")
        # 抛出自定义的请求错误异常
        raise httpx.RequestError(
            f"Request failed with status {response.status_code}, error_message: "
            f"{str_error_message}",
            request=response.request,
        )
    # 解码响应内容为UTF-8编码的字符串
    response_content = bytes_response_content.decode("utf-8")
    # 如果有非流式回调函数，调用该函数处理响应内容
    if non_streaming_callback:
        non_streaming_callback(response_content)
    # 返回解码后的响应内容字符串
    return response_content


# 定义一个异步函数，用于执行流式请求
async def _call_stream_request(
    http_client: AsyncClient,
    method: str,
    base_url: str,
    sse_output: bool,
    data: Dict[str, Any] | None = None,
    streaming_callback: Callable[[str], None] | None = None,
):
    # 初始化完整输出字符串
    full_out = ""
    # 异步迭代流式请求的输出
    async for out in _stream_request(http_client, method, base_url, sse_output, data):
        # 如果有流式回调函数，调用该函数处理每个输出部分
        if streaming_callback:
            streaming_callback(out)
        # 将每个输出部分添加到完整输出字符串中
        full_out += out
    # 返回完整的流式请求输出字符串
    return full_out


# 定义一个异步生成器函数，用于执行流式请求并逐行处理响应
async def _stream_request(
    http_client: AsyncClient,
    method: str,
    base_url: str,
    sse_output: bool,
    data: Dict[str, Any] | None = None,
):
    import json  # 导入json库
    from dbgpt.core.awel.util.chat_util import parse_openai_output  # 导入特定的函数模块

    # 初始化请求参数字典
    kwargs: Dict[str, Any] = {"url": base_url, "method": method}
    # 根据请求方法选择适当的数据格式
    if method in ["POST", "PUT"]:
        kwargs["json"] = data
    else:
        kwargs["params"] = data

    # 使用异步上下文管理器发起流式请求
    async with http_client.stream(**kwargs) as response:
        # 如果响应状态码为200，逐行读取响应内容并处理
        if response.status_code == 200:
            async for line in response.aiter_lines():
                if not line:
                    continue
                # 如果需要SSE输出，解析每行输出并返回文本内容
                if sse_output:
                    out = parse_openai_output(line)
                    if not out.success:
                        raise ClientException(f"Failed to parse output: {out.text}")
                    yield out.text
                else:
                    # 否则直接返回每行文本内容
                    yield line
        else:
            try:
                # 尝试读取响应的错误信息并解析为JSON对象返回
                error = await response.aread()
                yield json.loads(error)
            except Exception as e:
                # 发生异常时抛出该异常
                raise e
```