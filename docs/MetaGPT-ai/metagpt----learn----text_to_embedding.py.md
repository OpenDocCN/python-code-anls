# `MetaGPT\metagpt\learn\text_to_embedding.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : text_to_embedding.py
@Desc    : Text-to-Embedding skill, which provides text-to-embedding functionality.
"""

# 从metagpt.config中导入CONFIG
from metagpt.config import CONFIG
# 从metagpt.tools.openai_text_to_embedding中导入oas3_openai_text_to_embedding
from metagpt.tools.openai_text_to_embedding import oas3_openai_text_to_embedding

# 异步函数，将文本转换为嵌入向量
async def text_to_embedding(text, model="text-embedding-ada-002", openai_api_key="", **kwargs):
    """Text to embedding

    :param text: The text used for embedding. 用于嵌入的文本
    :param model: One of ['text-embedding-ada-002'], ID of the model to use. For more details, checkout: `https://api.openai.com/v1/models`. 使用的模型ID
    :param openai_api_key: OpenAI API key, For more details, checkout: `https://platform.openai.com/account/api-keys` OpenAI API密钥
    :return: A json object of :class:`ResultEmbedding` class if successful, otherwise `{}`. 如果成功，则返回ResultEmbedding类的json对象，否则返回空字典。
    """
    # 如果CONFIG.OPENAI_API_KEY或openai_api_key存在
    if CONFIG.OPENAI_API_KEY or openai_api_key:
        # 调用oas3_openai_text_to_embedding函数
        return await oas3_openai_text_to_embedding(text, model=model, openai_api_key=openai_api_key)
    # 如果不存在，则抛出环境错误
    raise EnvironmentError

```