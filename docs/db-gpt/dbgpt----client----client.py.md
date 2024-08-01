# `.\DB-GPT-src\dbgpt\client\client.py`

```py
"""This module contains the client for the DB-GPT API."""

# 导入必要的模块和库
import atexit  # 用于注册退出函数的模块
import json  # 用于处理 JSON 数据的模块
import os  # 提供与操作系统交互的功能
from typing import Any, AsyncGenerator, Dict, List, Optional, Union  # 类型提示相关的模块
from urllib.parse import urlparse  # 解析 URL 的模块

import httpx  # 提供 HTTP 客户端功能的库

# 从 dbgpt._private.pydantic 模块中导入 model_to_dict 函数
from dbgpt._private.pydantic import model_to_dict
# 从 dbgpt.core.schema.api 模块中导入 ChatCompletionResponse 和 ChatCompletionStreamResponse 类
from dbgpt.core.schema.api import ChatCompletionResponse, ChatCompletionStreamResponse

# 从当前包的 schema 模块中导入 ChatCompletionRequestBody 类
from .schema import ChatCompletionRequestBody

# 定义常量 CLIENT_API_PATH 和 CLIENT_SERVE_PATH
CLIENT_API_PATH = "api"
CLIENT_SERVE_PATH = "serve"

# 定义自定义异常类 ClientException，继承自内建的 Exception 类
class ClientException(Exception):
    """ClientException is raised when an error occurs in the client."""

    def __init__(self, status=None, reason=None, http_resp=None):
        """Initialize the ClientException.

        Args:
            status: Optional[int], the HTTP status code.
            reason: Optional[str], the reason for the exception.
            http_resp: Optional[httpx.Response], the HTTP response object.
        """
        # 初始化异常对象，保存状态码、原因和 HTTP 响应对象等信息
        self.status = status
        self.reason = reason
        self.http_resp = http_resp
        self.headers = http_resp.headers if http_resp else None
        self.body = http_resp.text if http_resp else None

    def __str__(self):
        """Return the error message."""
        # 返回异常的详细错误信息，包括状态码、原因、HTTP 响应头和响应体等
        error_message = "({0})\n" "Reason: {1}\n".format(self.status, self.reason)
        if self.headers:
            error_message += "HTTP response headers: {0}\n".format(self.headers)

        if self.body:
            error_message += "HTTP response body: {0}\n".format(self.body)

        return error_message


"""Client API."""


# 定义 Client 类，用于操作 DB-GPT API
class Client:
    """The client for the DB-GPT API."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        version: str = "v2",
        timeout: Optional[httpx._types.TimeoutTypes] = 120,
    ):
        """
        Create the client.

        Args:
            api_base: Optional[str], a full URL for the DB-GPT API.
                Defaults to the `http://localhost:5670/api/v2`.
            api_key: Optional[str], The dbgpt api key to use for authentication.
                Defaults to None.
            timeout: Optional[httpx._types.TimeoutTypes]: The timeout to use.
                Defaults to None.
                In most cases, pass in a float number to specify the timeout in seconds.

        Returns:
            None

        Raise: ClientException

        Examples:
        --------
        .. code-block:: python

            from dbgpt.client import Client

            DBGPT_API_BASE = "http://localhost:5670/api/v2"
            DBGPT_API_KEY = "dbgpt"
            client = Client(api_base=DBGPT_API_BASE, api_key=DBGPT_API_KEY)
            client.chat(model="chatgpt_proxyllm", messages="Hello?")
        """
        if not api_base:
            # 如果未提供 api_base 参数，则尝试从环境变量 DBGPT_API_BASE 中获取，默认为本地地址
            api_base = os.getenv(
                "DBGPT_API_BASE", f"http://localhost:5670/{CLIENT_API_PATH}/{version}"
            )
        if not api_key:
            # 如果未提供 api_key 参数，则尝试从环境变量 DBGPT_API_KEY 中获取
            api_key = os.getenv("DBGPT_API_KEY")
        if api_base and is_valid_url(api_base):
            # 如果提供了有效的 api_base 参数且是有效的 URL，则设置 _api_url 属性
            self._api_url = api_base
        else:
            # 如果 api_base 参数无效或不可访问，则抛出 ValueError 异常
            raise ValueError(f"api url {api_base} does not exist or is not accessible.")
        # 设置客户端的 _api_key、_version 和 _timeout 属性
        self._api_key = api_key
        self._version = version
        self._timeout = timeout
        # 根据是否提供了 api_key 设置 HTTP 请求的 Authorization 头部
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        # 初始化异步 HTTP 客户端对象，设置请求头和超时时间
        self._http_client = httpx.AsyncClient(
            headers=headers, timeout=timeout if timeout else httpx.Timeout(None)
        )
        # 注册退出时关闭客户端的方法
        atexit.register(self.close)
    # 定义方法的输入和输出类型，此方法返回一个 ChatCompletionResponse 对象
    -> ChatCompletionResponse:
        """
        Chat Completion.

        Args:
            model: str, The model name. 模型名称，字符串类型
            messages: Union[str, List[str]], The user input messages. 用户输入的消息，可以是字符串或字符串列表
            temperature: Optional[float], What sampling temperature to use,between 0
                and 2. Higher values like 0.8 will make the output more random,
                while lower values like 0.2 will make it more focused and deterministic.
                可选参数，采样温度，范围在0到2之间，较高的值（如0.8）会使输出更随机，
                而较低的值（如0.2）会使其更集中和确定性更强
            max_new_tokens: Optional[int].The maximum number of tokens that can be
                generated in the chat completion.
                可选参数，生成的聊天完成中可以生成的最大标记数
            chat_mode: Optional[str], The chat mode. 聊天模式，字符串类型的可选参数
            chat_param: Optional[str], The chat param of chat mode. 聊天模式的参数，字符串类型的可选参数
            conv_uid: Optional[str], The conversation id of the model inference.
                模型推理的会话ID，字符串类型的可选参数
            user_name: Optional[str], The user name of the model inference.
                模型推理的用户名称，字符串类型的可选参数
            sys_code: Optional[str], The system code of the model inference.
                模型推理的系统代码，字符串类型的可选参数
            span_id: Optional[str], The span id of the model inference.
                模型推理的跨度ID，字符串类型的可选参数
            incremental: bool, Used to control whether the content is returned
                incrementally or in full each time. If this parameter is not provided,
                the default is full return.
                布尔类型参数，控制是否每次增量返回内容还是完整返回。如果未提供此参数，默认为完整返回
            enable_vis: bool, Response content whether to output vis label.
                布尔类型参数，控制响应内容是否输出可视化标签

        Returns:
            ChatCompletionResponse: The chat completion response.
                返回 ChatCompletionResponse 对象，表示聊天完成的响应结果

        Examples:
        --------
        .. code-block:: python

            from dbgpt.client import Client

            DBGPT_API_BASE = "http://localhost:5670/api/v2"
            DBGPT_API_KEY = "dbgpt"
            client = Client(api_base=DBGPT_API_BASE, api_key=DBGPT_API_KEY)
            res = await client.chat(model="chatgpt_proxyllm", messages="Hello?")
        """
        # 创建 ChatCompletionRequestBody 对象，包含了聊天完成所需的所有请求参数
        request = ChatCompletionRequestBody(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            chat_mode=chat_mode,
            chat_param=chat_param,
            conv_uid=conv_uid,
            user_name=user_name,
            sys_code=sys_code,
            span_id=span_id,
            incremental=incremental,
            enable_vis=enable_vis,
        )
        # 发起 HTTP POST 请求，将请求数据转换为字典形式并发送到指定的聊天完成 API 端点
        response = await self._http_client.post(
            self._api_url + "/chat/completions", json=model_to_dict(request)
        )
        # 检查响应状态码，如果为200，则解析响应内容并返回 ChatCompletionResponse 对象
        if response.status_code == 200:
            json_data = json.loads(response.text)
            chat_completion_response = ChatCompletionResponse(**json_data)
            return chat_completion_response
        else:
            # 如果响应状态码不是200，则解析响应内容并返回相应的 JSON 对象
            return json.loads(response.content)
    async def chat_stream(
        self,
        model: str,
        messages: Union[str, List[str]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        chat_mode: Optional[str] = None,
        chat_param: Optional[str] = None,
        conv_uid: Optional[str] = None,
        user_name: Optional[str] = None,
        sys_code: Optional[str] = None,
        span_id: Optional[str] = None,
        incremental: bool = True,
        enable_vis: bool = True,
        **kwargs,
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        """
        Chat Stream Completion.

        Args:
            model: str, The model name.
            messages: Union[str, List[str]], The user input messages.
            temperature: Optional[float], What sampling temperature to use, between 0
                and 2. Higher values like 0.8 will make the output more random, while lower
                values like 0.2 will make it more focused and deterministic.
            max_new_tokens: Optional[int], The maximum number of tokens that can be
                generated in the chat completion.
            chat_mode: Optional[str], The chat mode.
            chat_param: Optional[str], The chat param of chat mode.
            conv_uid: Optional[str], The conversation id of the model inference.
            user_name: Optional[str], The user name of the model inference.
            sys_code: Optional[str], The system code of the model inference.
            span_id: Optional[str], The span id of the model inference.
            incremental: bool, Used to control whether the content is returned
                incrementally or in full each time. If this parameter is not provided,
                the default is full return.
            enable_vis: bool, Response content whether to output vis label.
        Returns:
            ChatCompletionStreamResponse: The chat completion response.

        Examples:
        --------
        .. code-block:: python

            from dbgpt.client import Client

            DBGPT_API_BASE = "http://localhost:5670/api/v2"
            DBGPT_API_KEY = "dbgpt"
            client = Client(api_base=DBGPT_API_BASE, api_key=DBGPT_API_KEY)
            res = await client.chat_stream(model="chatgpt_proxyllm", messages="Hello?")
        """
        # 构建请求体对象，用于发起聊天流完成请求
        request = ChatCompletionRequestBody(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            chat_mode=chat_mode,
            chat_param=chat_param,
            conv_uid=conv_uid,
            user_name=user_name,
            sys_code=sys_code,
            span_id=span_id,
            incremental=incremental,
            enable_vis=enable_vis,
        )
        # 通过生成器异步迭代处理聊天流响应
        async for chat_completion_response in self._chat_stream(model_to_dict(request)):
            yield chat_completion_response

    async def _chat_stream(
        self, data: Dict[str, Any]
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        """Chat Stream Completion.

        Args:
            data: dict, The data to send to the API.
        Returns:
            AsyncGenerator[ChatCompletionStreamResponse, None]: Async generator yielding chat completion responses.
        """
        # 使用异步上下文管理器来发起 HTTP POST 请求，获取聊天完成的流式响应
        async with self._http_client.stream(
            method="POST",
            url=self._api_url + "/chat/completions",
            json=data,
            headers={},
        ) as response:
            # 如果响应状态码为 200
            if response.status_code == 200:
                # 初始化 SSE 数据为空字符串
                sse_data = ""
                # 异步迭代响应内容的每一行
                async for line in response.aiter_lines():
                    try:
                        # 如果当前行是 "data: [DONE]"，则停止迭代
                        if line.strip() == "data: [DONE]":
                            break
                        # 如果当前行以 "data:" 开头
                        if line.startswith("data:"):
                            # 处理不同格式的 SSE 数据行
                            if line.startswith("data: "):
                                sse_data = line[len("data: ") :]
                            else:
                                sse_data = line[len("data:") :]
                            # 解析 JSON 格式的 SSE 数据
                            json_data = json.loads(sse_data)
                            # 构建聊天完成响应对象
                            chat_completion_response = ChatCompletionStreamResponse(
                                **json_data
                            )
                            # 使用异步生成器返回聊天完成响应对象
                            yield chat_completion_response
                    except Exception as e:
                        # 如果解析 SSE 数据出错，则抛出异常
                        raise Exception(
                            f"Failed to parse SSE data: {e}, sse_data: {sse_data}"
                        )

            else:
                try:
                    # 如果响应状态码不是 200，则尝试读取错误响应并返回
                    error = await response.aread()
                    yield json.loads(error)
                except Exception as e:
                    # 如果处理错误响应出错，则抛出异常
                    raise e

    async def get(self, path: str, *args, **kwargs):
        """Get method.

        Args:
            path: str, The path to get.
            args: Any, The arguments to pass to the get method.
        """
        # 清理掉参数中值为 None 的项，发起 HTTP GET 请求
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        response = await self._http_client.get(
            f"{self._api_url}/{CLIENT_SERVE_PATH}{path}",
            *args,
            params=kwargs,
        )
        return response

    async def post(self, path: str, args):
        """Post method.

        Args:
            path: str, The path to post.
            args: Any, The arguments to pass to the post.
        """
        # 发起 HTTP POST 请求，传递 JSON 格式的参数
        return await self._http_client.post(
            f"{self._api_url}/{CLIENT_SERVE_PATH}{path}",
            json=args,
        )

    async def post_param(self, path: str, args):
        """Post method.

        Args:
            path: str, The path to post.
            args: Any, The arguments to pass to the post.
        """
        # 发起 HTTP POST 请求，传递作为参数的字典格式参数
        return await self._http_client.post(
            f"{self._api_url}/{CLIENT_SERVE_PATH}{path}",
            params=args,
        )
    async def patch(self, path: str, *args):
        """
        Patch method.

        Args:
            path: str, The path to patch.
            args: Any, The arguments to pass to the patch.
        """
        # 使用 HTTP 客户端进行 PATCH 请求，构建完整的 API URL
        return self._http_client.patch(
            f"{self._api_url}/{CLIENT_SERVE_PATH}{path}", *args
        )

    async def put(self, path: str, args):
        """
        Put method.

        Args:
            path: str, The path to put.
            args: Any, The arguments to pass to the put.
        """
        # 使用 HTTP 客户端进行 PUT 请求，将 args 转换为 JSON 发送，构建完整的 API URL
        return await self._http_client.put(
            f"{self._api_url}/{CLIENT_SERVE_PATH}{path}", json=args
        )

    async def delete(self, path: str, *args):
        """
        Delete method.

        Args:
            path: str, The path to delete.
            args: Any, The arguments to pass to delete.
        """
        # 使用 HTTP 客户端进行 DELETE 请求，构建完整的 API URL
        return await self._http_client.delete(
            f"{self._api_url}/{CLIENT_SERVE_PATH}{path}", *args
        )

    async def head(self, path: str, *args):
        """
        Head method.

        Args:
            path: str, The path to head.
            args: Any, The arguments to pass to the head
        """
        # 使用 HTTP 客户端进行 HEAD 请求，构建完整的 API URL
        return self._http_client.head(self._api_url + path, *args)

    def close(self):
        """
        Close the client.
        """
        # 如果 HTTP 客户端未关闭，则获取或创建事件循环，并异步关闭 HTTP 客户端
        from dbgpt.util import get_or_create_event_loop

        if not self._http_client.is_closed:
            loop = get_or_create_event_loop()
            loop.run_until_complete(self._http_client.aclose())

    async def aclose(self):
        """
        Close the client.
        """
        # 异步关闭 HTTP 客户端
        await self._http_client.aclose()
# 定义一个函数，用于检查给定的 URL 是否有效
def is_valid_url(api_url: Any) -> bool:
    """Check if the given URL is valid.

    Args:
        api_url: Any, The URL to check.
    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    # 如果 api_url 不是字符串类型，直接返回 False
    if not isinstance(api_url, str):
        return False
    # 使用 urlparse 函数解析 URL，得到其组成部分
    parsed = urlparse(api_url)
    # 检查解析后的 URL 是否包含有效的 scheme 和 netloc（即协议和主机部分）
    return parsed.scheme != "" and parsed.netloc != ""
```