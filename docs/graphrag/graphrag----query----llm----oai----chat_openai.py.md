# `.\graphrag\graphrag\query\llm\oai\chat_openai.py`

```py
# 版权声明和许可声明，表明此代码版权归Microsoft Corporation所有，根据MIT许可证授权

# 引入必要的库和模块
from collections.abc import Callable  # 导入Callable类
from typing import Any  # 导入Any类型

# 导入tenacity库中所需的模块和函数
from tenacity import (
    AsyncRetrying,  # 异步重试装饰器
    RetryError,  # 重试错误异常类
    Retrying,  # 同步重试装饰器
    retry_if_exception_type,  # 根据异常类型决定是否重试的函数装饰器
    stop_after_attempt,  # 重试次数上限装饰器
    wait_exponential_jitter,  # 指数退避加随机延迟装饰器
)

# 导入本地自定义模块和类
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback  # 基础LLM和回调类
from graphrag.query.llm.oai.base import OpenAILLMImpl  # OpenAI基础LLM实现类
from graphrag.query.llm.oai.typing import (
    OPENAI_RETRY_ERROR_TYPES,  # 导入OpenAI的重试错误类型
    OpenaiApiType,  # OpenAI的API类型
)
from graphrag.query.progress import StatusReporter  # 状态报告器类

# 提示信息：必须提供模型
_MODEL_REQUIRED_MSG = "model is required"

# ChatOpenAI类，继承自BaseLLM和OpenAILLMImpl类
class ChatOpenAI(BaseLLM, OpenAILLMImpl):
    """Wrapper for OpenAI ChatCompletion models."""

    # 初始化方法
    def __init__(
        self,
        api_key: str | None = None,  # API密钥，可选
        model: str | None = None,  # 模型名称，可选
        azure_ad_token_provider: Callable | None = None,  # Azure AD令牌提供者，可选
        deployment_name: str | None = None,  # 部署名称，可选
        api_base: str | None = None,  # API基础路径，可选
        api_version: str | None = None,  # API版本，可选
        api_type: OpenaiApiType = OpenaiApiType.OpenAI,  # API类型，默认为OpenAI
        organization: str | None = None,  # 组织名称，可选
        max_retries: int = 10,  # 最大重试次数，默认为10
        request_timeout: float = 180.0,  # 请求超时时间，默认为180.0秒
        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # 重试错误类型，默认为OpenAI支持的错误类型
        reporter: StatusReporter | None = None,  # 状态报告器，可选
    ):
        # 调用父类初始化方法
        OpenAILLMImpl.__init__(
            self=self,  # 实例自身
            api_key=api_key,  # API密钥
            azure_ad_token_provider=azure_ad_token_provider,  # Azure AD令牌提供者
            deployment_name=deployment_name,  # 部署名称
            api_base=api_base,  # API基础路径
            api_version=api_version,  # API版本
            api_type=api_type,  # API类型
            organization=organization,  # 组织名称
            max_retries=max_retries,  # 最大重试次数
            request_timeout=request_timeout,  # 请求超时时间
            reporter=reporter,  # 状态报告器
        )
        # 设置模型名称
        self.model = model
        # 设置重试错误类型
        self.retry_error_types = retry_error_types

    # 生成文本的方法
    def generate(
        self,
        messages: str | list[Any],  # 消息文本或消息列表
        streaming: bool = True,  # 是否流式处理，默认为True
        callbacks: list[BaseLLMCallback] | None = None,  # 回调函数列表或空
        **kwargs: Any,  # 其他关键字参数
    ) -> str:
        """Generate text."""
        try:
            # 创建Retrying对象，用于重试逻辑
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),  # 设置重试次数上限
                wait=wait_exponential_jitter(max=10),  # 设置指数退避加随机延迟
                reraise=True,  # 异常重新抛出
                retry=retry_if_exception_type(self.retry_error_types),  # 根据异常类型决定是否重试
            )
            # 迭代重试器
            for attempt in retryer:
                with attempt:  # 使用当前尝试
                    return self._generate(  # 调用内部_generate方法生成文本
                        messages=messages,  # 消息文本或列表
                        streaming=streaming,  # 是否流式处理
                        callbacks=callbacks,  # 回调函数列表
                        **kwargs,  # 其他关键字参数
                    )
        except RetryError as e:
            # 报告错误到状态报告器
            self._reporter.error(
                message="Error at generate()",  # 错误消息
                details={self.__class__.__name__: str(e)},  # 错误详情
            )
            return ""  # 返回空字符串
        else:
            # TODO: 为什么在这种情况下不直接抛出异常？
            return ""  # 返回空字符串
    async def agenerate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text asynchronously."""
        try:
            # 设置异步重试器，指定重试条件和行为
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),  # type: ignore
            )
            # 异步迭代重试器
            async for attempt in retryer:
                with attempt:
                    # 调用内部异步生成方法进行生成
                    return await self._agenerate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError as e:
            # 记录异步生成过程中的重试错误
            self._reporter.error(f"Error at agenerate(): {e}")
            return ""
        else:
            # TODO: why not just throw in this case?
            # 在其他异常情况下返回空字符串
            return ""

    def _generate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        # 获取模型实例
        model = self.model
        # 如果模型不存在，则抛出数值错误异常
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        # 同步调用生成文本的 API
        response = self.sync_client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            stream=streaming,
            **kwargs,
        )  # type: ignore
        # 如果使用流式处理
        if streaming:
            # 初始化完整响应字符串
            full_response = ""
            # 持续处理响应流
            while True:
                try:
                    # 获取下一个响应块
                    chunk = response.__next__()  # type: ignore
                    # 如果响应为空或者没有选择项，则继续下一个循环
                    if not chunk or not chunk.choices:
                        continue

                    # 提取并添加响应块中的内容增量
                    delta = (
                        chunk.choices[0].delta.content
                        if chunk.choices[0].delta and chunk.choices[0].delta.content
                        else ""
                    )  # type: ignore

                    # 将增量添加到完整响应中
                    full_response += delta
                    # 如果存在回调函数列表，则对每个增量调用回调函数
                    if callbacks:
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)
                    # 如果响应块标记为结束，则退出循环
                    if chunk.choices[0].finish_reason == "stop":  # type: ignore
                        break
                except StopIteration:
                    break
            # 返回完整的响应结果
            return full_response
        # 如果非流式处理，则直接返回第一个选择项的消息内容或空字符串
        return response.choices[0].message.content or ""  # type: ignore

    async def _agenerate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        # 此方法内部实现了异步生成文本的逻辑，具体细节见 agenerate 方法的异步调用部分
        pass
    ) -> str:
        # 获取当前对象的模型
        model = self.model
        # 如果模型不存在，抛出数值错误异常
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        # 使用异步客户端创建聊天完成的请求
        response = await self.async_client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            stream=streaming,
            **kwargs,
        )
        # 如果是流式处理
        if streaming:
            # 初始化完整响应字符串
            full_response = ""
            # 持续获取响应数据
            while True:
                try:
                    # 获取下一个异步响应块
                    chunk = await response.__anext__()  # type: ignore
                    # 如果块为空或者选择项为空，则继续
                    if not chunk or not chunk.choices:
                        continue

                    # 获取第一个选择项的内容差异
                    delta = (
                        chunk.choices[0].delta.content
                        if chunk.choices[0].delta and chunk.choices[0].delta.content
                        else ""
                    )  # type: ignore

                    # 将差异内容添加到完整响应中
                    full_response += delta
                    # 如果存在回调函数，则逐个调用回调函数处理新的令牌
                    if callbacks:
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)
                    # 如果选择项的完成原因为“stop”，则跳出循环
                    if chunk.choices[0].finish_reason == "stop":  # type: ignore
                        break
                except StopIteration:
                    break
            # 返回完整的响应字符串
            return full_response

        # 如果不是流式处理，返回第一个选择项的消息内容或空字符串
        return response.choices[0].message.content or ""  # type: ignore
```