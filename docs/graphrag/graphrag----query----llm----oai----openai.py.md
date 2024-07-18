# `.\graphrag\graphrag\query\llm\oai\openai.py`

```py
# 从 logging 模块导入日志功能
import logging
# 从 typing 模块导入类型提示工具
from typing import Any

# 从 tenacity 库中导入异步重试相关的类和函数
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# 从 graphrag.query.llm.base 模块中导入 BaseLLMCallback 类
from graphrag.query.llm.base import BaseLLMCallback
# 从 graphrag.query.llm.oai.base 模块中导入 OpenAILLMImpl 类
from graphrag.query.llm.oai.base import OpenAILLMImpl
# 从 graphrag.query.llm.oai.typing 模块中导入相关类型
from graphrag.query.llm.oai.typing import (
    OPENAI_RETRY_ERROR_TYPES,
    OpenaiApiType,
)

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)

# 继承 OpenAILLMImpl 类，实现对 OpenAI Completion 模型的包装
class OpenAI(OpenAILLMImpl):
    """Wrapper for OpenAI Completion models."""

    def __init__(
        self,
        api_key: str,
        model: str,
        deployment_name: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: OpenaiApiType = OpenaiApiType.OpenAI,
        organization: str | None = None,
        max_retries: int = 10,
        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # type: ignore
    ):
        # 初始化 OpenAI 对象的属性
        self.api_key = api_key
        self.model = model
        self.deployment_name = deployment_name
        self.api_base = api_base
        self.api_version = api_version
        self.api_type = api_type
        self.organization = organization
        self.max_retries = max_retries
        self.retry_error_types = retry_error_types

    def generate(
        self,
        messages: str | list[str],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text."""
        try:
            # 创建 Retrying 实例，用于进行重试操作
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            # 开始重试过程
            for attempt in retryer:
                with attempt:
                    # 调用内部方法 _generate 来生成文本
                    return self._generate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError:
            # 如果达到最大重试次数仍然失败，记录异常日志并返回空字符串
            log.exception("RetryError at generate(): %s")
            return ""
        else:
            # TODO: 这种情况下为什么不直接抛出异常？
            return ""

    async def agenerate(
        self,
        messages: str | list[str],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ):
        # Async 方法的声明，具体实现需要进一步补充
        pass
    ) -> str:
        """生成文本的异步方式。"""
        try:
            # 设置重试策略
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            # 异步循环重试操作
            async for attempt in retryer:
                with attempt:
                    # 调用异步生成方法
                    return await self._agenerate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError:
            # 记录异常并返回空字符串
            log.exception("Error at agenerate()")
            return ""
        else:
            # 如果没有发生异常，为什么不直接抛出异常？
            return ""

    def _generate(
        self,
        messages: str | list[str],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        # 使用同步客户端创建文本生成请求
        response = self.sync_client.chat.completions.create(  # type: ignore
            model=self.model,
            messages=messages,  # type: ignore
            stream=streaming,
            **kwargs,
        )  # type: ignore
        if streaming:
            full_response = ""
            while True:
                try:
                    # 获取下一个响应块
                    chunk = response.__next__()  # type: ignore
                    # 如果块为空或没有选择项，则继续循环
                    if not chunk or not chunk.choices:
                        continue

                    # 获取响应块中的内容增量
                    delta = (
                        chunk.choices[0].delta.content
                        if chunk.choices[0].delta and chunk.choices[0].delta.content
                        else ""
                    )  # type: ignore

                    # 将增量添加到完整响应中
                    full_response += delta
                    # 触发回调处理新的生成标记
                    if callbacks:
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)
                    # 如果响应块指示完成原因为“stop”，则终止循环
                    if chunk.choices[0].finish_reason == "stop":  # type: ignore
                        break
                except StopIteration:
                    break
            return full_response
        # 对于非流式响应，返回第一个选择项的消息内容或空字符串
        return response.choices[0].message.content or ""  # type: ignore

    async def _agenerate(
        self,
        messages: str | list[str],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        # 异步方法，类似于_generate方法，但是异步处理
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=streaming,
            **kwargs,
        )
        if streaming:
            full_response = ""
            while True:
                try:
                    chunk = await response.__anext__()
                    if not chunk or not chunk.choices:
                        continue

                    delta = (
                        chunk.choices[0].delta.content
                        if chunk.choices[0].delta and chunk.choices[0].delta.content
                        else ""
                    )

                    full_response += delta
                    if callbacks:
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)
                    if chunk.choices[0].finish_reason == "stop":
                        break
                except StopAsyncIteration:
                    break
            return full_response
        return response.choices[0].message.content or ""
    ) -> str:
        # 调用异步客户端创建聊天完成请求，并忽略类型检查
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,  # 传递消息列表参数，忽略类型检查
            stream=streaming,
            **kwargs,  # 传递其它关键字参数
        )
        if streaming:
            # 如果是流式处理，初始化完整响应字符串
            full_response = ""
            while True:
                try:
                    # 异步迭代获取响应的下一个结果块，忽略类型检查
                    chunk = await response.__anext__()
                    if not chunk or not chunk.choices:
                        continue

                    # 提取第一个选择的内容增量，如果不存在则为空字符串，忽略类型检查
                    delta = (
                        chunk.choices[0].delta.content
                        if chunk.choices[0].delta and chunk.choices[0].delta.content
                        else ""
                    )

                    # 将增量添加到完整响应中
                    full_response += delta

                    # 如果有回调函数，对每个回调函数调用新令牌方法，传递增量
                    if callbacks:
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)

                    # 如果选择块的结束原因为"stop"，则终止循环
                    if chunk.choices[0].finish_reason == "stop":
                        break
                except StopIteration:
                    break
            return full_response  # 返回完整响应字符串
        return response.choices[0].message.content or ""  # 返回响应中第一个选择的消息内容，如果不存在则为空字符串，忽略类型检查
```