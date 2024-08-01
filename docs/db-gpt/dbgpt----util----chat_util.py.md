# `.\DB-GPT-src\dbgpt\util\chat_util.py`

```py
import asyncio  # 引入 asyncio 库，用于异步编程
from typing import Any, Coroutine, List  # 引入类型提示相关的库


async def llm_chat_response_nostream(chat_scene: str, **chat_param):
    """llm_chat_response_nostream

    引入 BaseChat 和 ChatFactory 类，从调试器中获取聊天实现对象，调用其获取 GPT 回应的方法并返回结果
    """
    from dbgpt.app.scene import BaseChat, ChatFactory  # 从调试器的 app.scene 模块导入类

    chat_factory = ChatFactory()  # 创建 ChatFactory 实例
    chat: BaseChat = chat_factory.get_implementation(chat_scene, **chat_param)  # 获取指定场景的 BaseChat 实例
    res = await chat.get_llm_response()  # 调用实例的获取 GPT 回应的异步方法
    return res  # 返回获取到的 GPT 回应


async def llm_chat_response(chat_scene: str, **chat_param):
    """llm_chat_response

    引入 BaseChat 和 ChatFactory 类，从调试器中获取聊天实现对象，调用其流式调用方法并返回结果
    """
    from dbgpt.app.scene import BaseChat, ChatFactory  # 从调试器的 app.scene 模块导入类

    chat_factory = ChatFactory()  # 创建 ChatFactory 实例
    chat: BaseChat = chat_factory.get_implementation(chat_scene, **chat_param)  # 获取指定场景的 BaseChat 实例
    return chat.stream_call()  # 调用实例的流式调用方法并返回结果


async def run_async_tasks(
    tasks: List[Coroutine],
    concurrency_limit: int = None,
) -> List[Any]:
    """Run a list of async tasks.

    执行一组异步任务，可以选择限制并发数量。
    """
    tasks_to_execute: List[Any] = tasks  # 初始化需要执行的任务列表

    async def _gather() -> List[Any]:
        if concurrency_limit:
            semaphore = asyncio.Semaphore(concurrency_limit)  # 如果有并发限制，创建信号量对象

            async def _execute_task(task):
                async with semaphore:  # 使用信号量限制并发执行
                    return await task

            # 使用信号量限制并发执行任务，并收集结果
            return await asyncio.gather(
                *[_execute_task(task) for task in tasks_to_execute]
            )
        else:
            return await asyncio.gather(*tasks_to_execute)  # 没有并发限制时，直接执行任务并收集结果

    # 调用内部异步函数执行任务并返回结果
    return await _gather()


def run_tasks(
    tasks: List[Coroutine],
) -> List[Any]:
    """Run a list of async tasks.

    执行一组异步任务，等待所有任务完成并返回结果。
    """
    tasks_to_execute: List[Any] = tasks  # 初始化需要执行的任务列表

    async def _gather() -> List[Any]:
        return await asyncio.gather(*tasks_to_execute)  # 执行所有任务并收集结果

    outputs: List[Any] = asyncio.run(_gather())  # 运行异步函数并获取结果
    return outputs  # 返回所有任务的结果
```