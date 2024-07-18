# `.\graphrag\graphrag\query\llm\oai\embedding.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OpenAI Embedding model implementation."""

# 引入 asyncio 库，支持异步编程
import asyncio
# 从 collections.abc 模块中引入 Callable 类型
from collections.abc import Callable
# 从 typing 模块中引入 Any 类型
from typing import Any

# 引入 numpy 库，用于数值计算
import numpy as np
# 引入 tiktoken 模块，用于获取编码器
import tiktoken
# 引入 tenacity 库，支持重试逻辑
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# 从 graphrag.query.llm.base 模块中引入 BaseTextEmbedding 类
from graphrag.query.llm.base import BaseTextEmbedding
# 从 graphrag.query.llm.oai.base 模块中引入 OpenAILLMImpl 类
from graphrag.query.llm.oai.base import OpenAILLMImpl
# 从 graphrag.query.llm.oai.typing 模块中引入相关类型
from graphrag.query.llm.oai.typing import (
    OPENAI_RETRY_ERROR_TYPES,
    OpenaiApiType,
)
# 从 graphrag.query.llm.text_utils 模块中引入 chunk_text 函数
from graphrag.query.llm.text_utils import chunk_text
# 从 graphrag.query.progress 模块中引入 StatusReporter 类
from graphrag.query.progress import StatusReporter

# 定义 OpenAIEmbedding 类，继承自 BaseTextEmbedding 和 OpenAILLMImpl
class OpenAIEmbedding(BaseTextEmbedding, OpenAILLMImpl):
    """Wrapper for OpenAI Embedding models."""

    # 初始化方法
    def __init__(
        self,
        api_key: str | None = None,  # API 密钥，可选
        azure_ad_token_provider: Callable | None = None,  # Azure AD 令牌提供者，可选
        model: str = "text-embedding-3-small",  # 模型名称，默认为 "text-embedding-3-small"
        deployment_name: str | None = None,  # 部署名称，可选
        api_base: str | None = None,  # API 基础地址，可选
        api_version: str | None = None,  # API 版本，可选
        api_type: OpenaiApiType = OpenaiApiType.OpenAI,  # API 类型，默认为 OpenAI
        organization: str | None = None,  # 组织名称，可选
        encoding_name: str = "cl100k_base",  # 编码器名称，默认为 "cl100k_base"
        max_tokens: int = 8191,  # 最大令牌数，默认为 8191
        max_retries: int = 10,  # 最大重试次数，默认为 10
        request_timeout: float = 180.0,  # 请求超时时间，默认为 180.0 秒
        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # 重试时的错误类型，默认为预定义的错误类型
        reporter: StatusReporter | None = None,  # 进度报告器，可选
    ):
        # 调用父类 OpenAILLMImpl 的初始化方法
        OpenAILLMImpl.__init__(
            self=self,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            deployment_name=deployment_name,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,  # 指定 API 类型
            organization=organization,
            max_retries=max_retries,
            request_timeout=request_timeout,
            reporter=reporter,
        )

        # 设置模型名称
        self.model = model
        # 设置编码器名称
        self.encoding_name = encoding_name
        # 设置最大令牌数
        self.max_tokens = max_tokens
        # 获取并设置指定编码器
        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
        # 设置重试时的错误类型
        self.retry_error_types = retry_error_types
    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's sync function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        Please refer to: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
        """
        # 将文本按照最大标记数分块处理
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        # 存储每个分块的嵌入和长度
        chunk_embeddings = []
        chunk_lens = []
        # 遍历每个分块
        for chunk in token_chunks:
            try:
                # 尝试嵌入当前分块的文本
                embedding, chunk_len = self._embed_with_retry(chunk, **kwargs)
                chunk_embeddings.append(embedding)
                chunk_lens.append(chunk_len)
            # 捕获所有异常，记录错误信息
            except Exception as e:  # noqa BLE001
                self._reporter.error(
                    message="Error embedding chunk",
                    details={self.__class__.__name__: str(e)},
                )

                continue
        # 计算所有分块嵌入的加权平均值
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        # 对结果进行归一化处理
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        # 将嵌入结果转换为列表并返回
        return chunk_embeddings.tolist()

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's async function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        """
        # 将文本按照最大标记数分块处理
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        # 存储每个分块的嵌入和长度
        chunk_embeddings = []
        chunk_lens = []
        # 使用异步方式嵌入每个分块的文本
        embedding_results = await asyncio.gather(*[
            self._aembed_with_retry(chunk, **kwargs) for chunk in token_chunks
        ])
        # 筛选出有效的嵌入结果并分别存储嵌入和长度
        embedding_results = [result for result in embedding_results if result[0]]
        chunk_embeddings = [result[0] for result in embedding_results]
        chunk_lens = [result[1] for result in embedding_results]
        # 计算所有分块嵌入的加权平均值
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)  # type: ignore
        # 对结果进行归一化处理
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        # 将嵌入结果转换为列表并返回
        return chunk_embeddings.tolist()

    def _embed_with_retry(
        self, text: str | tuple, **kwargs: Any
        # 使用重试机制嵌入给定的文本或元组
    ) -> tuple[list[float], int]:
        # 定义一个方法，接受一个文本输入和其他关键字参数，并返回一个包含嵌入向量和文本长度的元组

        try:
            # 创建一个重试器对象，用于异步操作，设定重试策略
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),  # 最大重试次数
                wait=wait_exponential_jitter(max=10),  # 指数退避的等待时间，最大10秒
                reraise=True,  # 发生异常时重新抛出
                retry=retry_if_exception_type(self.retry_error_types),  # 指定需要重试的异常类型
            )
            # 遍历重试器对象
            for attempt in retryer:
                # 在当前重试尝试下执行以下代码块
                with attempt:
                    # 使用同步客户端创建嵌入向量，输入为文本和模型名，kwargs表示其他关键字参数
                    embedding = (
                        self.sync_client.embeddings.create(  # type: ignore
                            input=text,
                            model=self.model,
                            **kwargs,  # type: ignore
                        )
                        .data[0]  # 获取第一个数据项
                        .embedding  # 获取嵌入向量
                        or []  # 如果为空则返回空列表
                    )
                    # 返回嵌入向量和文本的长度
                    return (embedding, len(text))
        except RetryError as e:
            # 捕获重试错误，并记录错误信息
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            # 返回空列表和0作为默认值
            return ([], 0)
        else:
            # 如果没有发生异常，则返回空列表和0作为默认值
            # TODO: 为什么在这种情况下不直接抛出异常？
            return ([], 0)

    async def _aembed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        # 异步方法定义，接受文本输入和其他关键字参数，并返回包含嵌入向量和文本长度的元组

        try:
            # 创建一个异步重试器对象，设定重试策略
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),  # 最大重试次数
                wait=wait_exponential_jitter(max=10),  # 指数退避的等待时间，最大10秒
                reraise=True,  # 发生异常时重新抛出
                retry=retry_if_exception_type(self.retry_error_types),  # 指定需要重试的异常类型
            )
            # 在异步重试器对象上进行异步迭代
            async for attempt in retryer:
                # 在当前异步重试尝试下执行以下代码块
                with attempt:
                    # 使用异步客户端创建嵌入向量，输入为文本和模型名，kwargs表示其他关键字参数
                    embedding = (
                        await self.async_client.embeddings.create(  # type: ignore
                            input=text,
                            model=self.model,
                            **kwargs,  # type: ignore
                        )
                    ).data[0].embedding or []  # 获取嵌入向量数据，如果为空则返回空列表
                    # 返回嵌入向量和文本的长度
                    return (embedding, len(text))
        except RetryError as e:
            # 捕获重试错误，并记录错误信息
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            # 返回空列表和0作为默认值
            return ([], 0)
        else:
            # 如果没有发生异常，则返回空列表和0作为默认值
            # TODO: 为什么在这种情况下不直接抛出异常？
            return ([], 0)
```