# `.\graphrag\graphrag\index\verbs\text\embed\strategies\openai.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run method definition."""

import asyncio  # 引入异步IO库
import logging  # 引入日志记录库
from typing import Any  # 引入类型提示

import numpy as np  # 引入数值计算库numpy
from datashaper import ProgressTicker, VerbCallbacks, progress_ticker  # 从datashaper模块中引入进度条和回调相关功能

import graphrag.config.defaults as defs  # 引入默认配置
from graphrag.index.cache import PipelineCache  # 从graphrag.index.cache模块引入PipelineCache类
from graphrag.index.llm import load_llm_embeddings  # 从graphrag.index.llm模块引入加载LLM嵌入模型的函数
from graphrag.index.text_splitting import TokenTextSplitter  # 从graphrag.index.text_splitting模块引入TokenTextSplitter类
from graphrag.index.utils import is_null  # 从graphrag.index.utils模块引入is_null函数
from graphrag.llm import EmbeddingLLM, OpenAIConfiguration  # 从graphrag.llm模块引入EmbeddingLLM类和OpenAIConfiguration类

from .typing import TextEmbeddingResult  # 从当前目录的typing模块引入TextEmbeddingResult类型

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


async def run(
    input: list[str],  # 输入参数为字符串列表
    callbacks: VerbCallbacks,  # 回调函数对象
    cache: PipelineCache,  # PipelineCache对象用于缓存
    args: dict[str, Any],  # 参数字典，键为字符串，值为任意类型
) -> TextEmbeddingResult:
    """Run the Claim extraction chain."""
    if is_null(input):  # 如果输入为空
        return TextEmbeddingResult(embeddings=None)  # 返回空的TextEmbeddingResult对象

    llm_config = args.get("llm", {})  # 获取参数中的LLM配置，若不存在则为空字典
    batch_size = args.get("batch_size", 16)  # 获取参数中的批处理大小，默认为16
    batch_max_tokens = args.get("batch_max_tokens", 8191)  # 获取参数中的最大令牌数，默认为8191
    oai_config = OpenAIConfiguration(llm_config)  # 使用LLM配置创建OpenAIConfiguration对象
    splitter = _get_splitter(oai_config, batch_max_tokens)  # 调用_get_splitter函数获取TokenTextSplitter对象
    llm = _get_llm(oai_config, callbacks, cache)  # 调用_get_llm函数获取EmbeddingLLM对象
    semaphore: asyncio.Semaphore = asyncio.Semaphore(args.get("num_threads", 4))  # 使用异步IO信号量，控制并发线程数量

    # Break up the input texts. The sizes here indicate how many snippets are in each input text
    texts, input_sizes = _prepare_embed_texts(input, splitter)  # 调用_prepare_embed_texts函数对输入文本进行处理，获取文本列表和大小列表
    text_batches = _create_text_batches(
        texts,
        batch_size,
        batch_max_tokens,
        splitter,
    )  # 调用_create_text_batches函数创建文本批次

    log.info(
        "embedding %d inputs via %d snippets using %d batches. max_batch_size=%d, max_tokens=%d",
        len(input),
        len(texts),
        len(text_batches),
        batch_size,
        batch_max_tokens,
    )  # 记录信息到日志，显示输入数量、文本片段数量和批次数量以及最大批处理大小和最大令牌数

    ticker = progress_ticker(callbacks.progress, len(text_batches))  # 使用进度条回调函数创建ticker对象

    # Embed each chunk of snippets
    embeddings = await _execute(llm, text_batches, ticker, semaphore)  # 调用_execute异步函数执行嵌入操作，获取嵌入结果列表
    embeddings = _reconstitute_embeddings(embeddings, input_sizes)  # 调用_reconstitute_embeddings函数重构嵌入结果

    return TextEmbeddingResult(embeddings=embeddings)  # 返回TextEmbeddingResult对象，包含嵌入结果


def _get_splitter(
    config: OpenAIConfiguration, batch_max_tokens: int
) -> TokenTextSplitter:
    return TokenTextSplitter(
        encoding_name=config.encoding_model or defs.ENCODING_MODEL,  # 使用配置中的编码模型或默认的编码模型
        chunk_size=batch_max_tokens,  # 设置分块大小为batch_max_tokens
    )


def _get_llm(
    config: OpenAIConfiguration,
    callbacks: VerbCallbacks,
    cache: PipelineCache,
) -> EmbeddingLLM:
    llm_type = config.lookup("type", "Unknown")  # 查找配置中的LLM类型，默认为"Unknown"
    return load_llm_embeddings(
        "text_embedding",  # 加载文本嵌入模型
        llm_type,
        callbacks,
        cache,
        config.raw_config,
    )


async def _execute(
    llm: EmbeddingLLM,
    chunks: list[list[str]],
    tick: ProgressTicker,
    semaphore: asyncio.Semaphore,
) -> list[list[float]]:
    # 省略部分内容，用于执行LLM嵌入操作，返回嵌入结果列表
    # 定义一个异步函数 embed，接受`
async def embed(chunk: list[str]):
    # 定义一个异步函数 embed，接收一个字符串列表 chunk 作为参数
    async with semaphore:
        # 使用异步上下文管理器 semaphore，控制并发访问
        chunk_embeddings = await llm(chunk)
        # 调用异步函数 llm，传入 chunk 参数，等待获取 chunk 中各元素的嵌入表示
        result = np.array(chunk_embeddings.output)
        # 将嵌入表示结果转换为 NumPy 数组
        tick(1)
        # 执行一个时间跟踪函数 tick，参数为 1
    return result
    # 返回异步函数 embed使用列表推导式创建 embed 函数的异步任务列表 futures，每个任务处理 chunks 中的一个 chunk
    futures = [embed(chunk) for chunk in chunks]
    # 使用 asyncio.gather 并发执行所有异步任务，并等待所有任务完成，将结果列表化
    results = await asyncio.gather(*futures)
    # 合并嵌套的结果列表，将结果展平为单个列表
    return [item for sublist in results for item in sublist]
# 创建文本批次以进行嵌入处理
def _create_text_batches(
    texts: list[str],            # 输入的文本列表
    max_batch_size: int,         # 最大批次大小限制
    max_batch_tokens: int,       # 单个批次的最大标记数限制
    splitter: TokenTextSplitter, # 分割器对象，用于文本分割
) -> list[list[str]]:
    """Create batches of texts to embed."""
    # Azure 文档指出，每次请求最多允许 16 个并发嵌入和 8191 个标记
    result = []                   # 存储最终批次的列表
    current_batch = []            # 当前正在构建的批次
    current_batch_tokens = 0      # 当前批次已使用的标记数

    for text in texts:
        token_count = splitter.num_tokens(text)  # 计算当前文本的标记数
        if (
            len(current_batch) >= max_batch_size
            or current_batch_tokens + token_count > max_batch_tokens
        ):
            result.append(current_batch)        # 如果达到批次大小或标记数限制，添加当前批次到结果列表
            current_batch = []                  # 重置当前批次
            current_batch_tokens = 0            # 重置已使用标记数

        current_batch.append(text)              # 将当前文本添加到当前批次
        current_batch_tokens += token_count     # 更新当前批次已使用的标记数

    if len(current_batch) > 0:
        result.append(current_batch)            # 添加最后一个未完成的批次到结果列表

    return result                              # 返回所有批次的列表


# 准备文本以进行嵌入处理
def _prepare_embed_texts(
    input: list[str],             # 输入的文本列表
    splitter: TokenTextSplitter,  # 分割器对象，用于文本分割
) -> tuple[list[str], list[int]]:
    sizes: list[int] = []         # 存储每个文本分割后的子文本数量列表
    snippets: list[str] = []      # 存储所有子文本的列表

    for text in input:
        # 分割输入文本，并过滤掉空内容
        split_texts = splitter.split_text(text)
        if split_texts is None:
            continue
        split_texts = [text for text in split_texts if len(text) > 0]

        sizes.append(len(split_texts))   # 记录当前文本分割后的子文本数量
        snippets.extend(split_texts)     # 将分割后的子文本添加到 snippets 列表中

    return snippets, sizes               # 返回所有子文本和它们的数量列表


# 将嵌入重新组合为原始输入文本
def _reconstitute_embeddings(
    raw_embeddings: list[list[float]],       # 原始嵌入向量列表
    sizes: list[int]                         # 每个文本对应的子文本数量列表
) -> list[list[float] | None]:
    """Reconstitute the embeddings into the original input texts."""
    embeddings: list[list[float] | None] = [] # 存储重新组合后的嵌入向量列表
    cursor = 0                               # 游标，用于追踪当前处理到的原始嵌入向量

    for size in sizes:
        if size == 0:
            embeddings.append(None)          # 如果当前文本没有子文本，添加 None 到嵌入向量列表
        elif size == 1:
            embedding = raw_embeddings[cursor]   # 如果当前文本只有一个子文本，则直接使用对应的原始嵌入向量
            embeddings.append(embedding)
            cursor += 1
        else:
            chunk = raw_embeddings[cursor : cursor + size]  # 否则，取出当前文本所有子文本对应的原始嵌入向量
            average = np.average(chunk, axis=0)             # 计算这些向量的平均值
            normalized = average / np.linalg.norm(average)  # 对平均值向量进行归一化处理
            embeddings.append(normalized.tolist())          # 将归一化后的向量添加到 embeddings 列表中
            cursor += size

    return embeddings                        # 返回所有文本的重新组合后的嵌入向量列表
```