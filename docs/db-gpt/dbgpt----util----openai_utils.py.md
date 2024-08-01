# `.\DB-GPT-src\dbgpt\util\openai_utils.py`

```py
# 导入必要的模块
import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional

import httpx

# 获取日志记录器
logger = logging.getLogger(__name__)
# 定义一个类型别名，表示一个异步函数，接受一个字符串参数并返回一个异步 None 对象
MessageCaller = Callable[[str], Awaitable[None]]


# 定义一个异步函数，用于处理聊天完成的操作
async def _do_chat_completion(
    url: str,
    chat_data: Dict[str, Any],
    client: httpx.AsyncClient,
    headers: Dict[str, Any] = {},
    timeout: int = 60,
    caller: Optional[MessageCaller] = None,
) -> Iterator[str]:
    # 使用异步 HTTP 客户端发送 POST 请求，获取响应流
    async with client.stream(
        "POST",
        url,
        headers=headers,
        json=chat_data,
        timeout=timeout,
    ) as res:
        # 如果响应状态码不是 200，则处理错误信息
        if res.status_code != 200:
            error_message = await res.aread()
            if error_message:
                error_message = error_message.decode("utf-8")
            logger.error(
                f"Request failed with status {res.status_code}. Error: {error_message}"
            )
            raise httpx.RequestError(
                f"Request failed with status {res.status_code}",
                request=res.request,
            )
        # 遍历响应流的每一行
        async for line in res.aiter_lines():
            if line:
                if not line.startswith("data: "):
                    # 如果行不以"data: "开头，则调用回调函数并返回该行
                    if caller:
                        await caller(line)
                    yield line
                else:
                    decoded_line = line.split("data: ", 1)[1]
                    if decoded_line.lower().strip() != "[DONE]".lower():
                        obj = json.loads(decoded_line)
                        if "error_code" in obj and obj["error_code"] != 0:
                            if caller:
                                await caller(obj.get("text"))
                            yield obj.get("text")
                        else:
                            if (
                                "choices" in obj
                                and obj["choices"][0]["delta"].get("content")
                                is not None
                            ):
                                text = obj["choices"][0]["delta"].get("content")
                                if caller:
                                    await caller(text)
                                yield text
            await asyncio.sleep(0.02)


# 定义一个异步函数，用于处理聊天完成的操作，并返回一个迭代器
async def chat_completion_stream(
    url: str,
    chat_data: Dict[str, Any],
    client: Optional[httpx.AsyncClient] = None,
    headers: Dict[str, Any] = {},
    timeout: int = 60,
    caller: Optional[MessageCaller] = None,
) -> Iterator[str]:
    # 如果提供了客户端，则调用_do_chat_completion函数处理
    if client:
        async for text in _do_chat_completion(
            url,
            chat_data,
            client=client,
            headers=headers,
            timeout=timeout,
            caller=caller,
        ):
            yield text
    else:
        # 使用异步 HTTP 客户端创建会话
        async with httpx.AsyncClient() as client:
            # 异步循环，从聊天完成函数中逐个获取文本结果
            async for text in _do_chat_completion(
                url,
                chat_data,
                client=client,
                headers=headers,
                timeout=timeout,
                caller=caller,
            ):
                # 生成器：产出每个文本结果
                yield text
# 异步函数，用于获取聊天内容的完整文本
async def chat_completion(
    url: str,  # 聊天接口的 URL
    chat_data: Dict[str, Any],  # 聊天数据，字典类型
    client: Optional[httpx.AsyncClient] = None,  # 异步 HTTP 客户端，可选参数，默认为 None
    headers: Dict[str, Any] = {},  # HTTP 请求头部信息，字典类型，默认为空字典
    timeout: int = 60,  # 超时时间，整数类型，默认为 60 秒
    caller: Optional[MessageCaller] = None,  # 消息调用者，可选参数，默认为 None
) -> str:  # 返回值为字符串类型
    full_text = ""  # 初始化完整文本为空字符串
    # 异步迭代聊天内容流，获取文本并拼接到完整文本中
    async for text in chat_completion_stream(
        url, chat_data, client, headers=headers, timeout=timeout, caller=caller
    ):
        full_text += text  # 将获取的文本拼接到完整文本中
    return full_text  # 返回完整文本
```