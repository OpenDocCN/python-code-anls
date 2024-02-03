# `.\AutoGPT\autogpts\autogpt\autogpt\memory\vector\utils.py`

```py
import logging
from contextlib import suppress
from typing import Any, Sequence, overload

import numpy as np

from autogpt.config import Config
from autogpt.core.resource.model_providers import EmbeddingModelProvider

logger = logging.getLogger(__name__)

Embedding = list[float] | list[np.float32] | np.ndarray[Any, np.dtype[np.float32]]
"""Embedding vector"""

TText = Sequence[int]
"""Tokenized text"""


@overload
async def get_embedding(
    input: str | TText, config: Config, embedding_provider: EmbeddingModelProvider
) -> Embedding:
    ...


@overload
async def get_embedding(
    input: list[str] | list[TText],
    config: Config,
    embedding_provider: EmbeddingModelProvider,
) -> list[Embedding]:
    ...


async def get_embedding(
    input: str | TText | list[str] | list[TText],
    config: Config,
    embedding_provider: EmbeddingModelProvider,
) -> Embedding | list[Embedding]:
    """Get an embedding from the ada model.

    Args:
        input: Input text to get embeddings for, encoded as a string or array of tokens.
            Multiple inputs may be given as a list of strings or token arrays.
        embedding_provider: The provider to create embeddings.

    Returns:
        List[float]: The embedding.
    """
    multiple = isinstance(input, list) and all(not isinstance(i, int) for i in input)

    if isinstance(input, str):
        # 将换行符替换为空格
        input = input.replace("\n", " ")

        # 使用插件获取嵌入（如果实现了）
        with suppress(NotImplementedError):
            return _get_embedding_with_plugin(input, config)

    elif multiple and isinstance(input[0], str):
        # 将每个字符串中的换行符替换为空格
        input = [text.replace("\n", " ") for text in input]

        # 使用插件获取每个输入的嵌入（如果实现了）
        with suppress(NotImplementedError):
            return [_get_embedding_with_plugin(i, config) for i in input]

    # 获取配置中的嵌入模型
    model = config.embedding_model

    # 记录调试信息，显示正在使用的模型和输入数量
    logger.debug(
        f"Getting embedding{f's for {len(input)} inputs' if multiple else ''}"
        f" with model '{model}'"
    )
    # 如果不需要处理多个输入文本，则直接调用embedding_provider.create_embedding创建嵌入
    if not multiple:
        return (
            await embedding_provider.create_embedding(
                text=input,
                model_name=model,
                embedding_parser=lambda e: e,
            )
        ).embedding
    # 如果需要处理多个输入文本，则遍历每个文本，分别调用embedding_provider.create_embedding创建嵌入
    else:
        embeddings = []
        for text in input:
            result = await embedding_provider.create_embedding(
                text=text,
                model_name=model,
                embedding_parser=lambda e: e,
            )
            embeddings.append(result.embedding)
        # 返回所有文本的嵌入列表
        return embeddings
# 使用插件获取文本的嵌入向量
def _get_embedding_with_plugin(text: str, config: Config) -> Embedding:
    # 遍历配置中的插件列表
    for plugin in config.plugins:
        # 检查插件是否能处理文本的嵌入向量
        if plugin.can_handle_text_embedding(text):
            # 如果插件可以处理，则调用插件处理文本的嵌入向量
            embedding = plugin.handle_text_embedding(text)
            # 如果插件返回了嵌入向量，则直接返回该嵌入向量
            if embedding is not None:
                return embedding

    # 如果没有插件可以处理文本的嵌入向量，则抛出未实现错误
    raise NotImplementedError
```