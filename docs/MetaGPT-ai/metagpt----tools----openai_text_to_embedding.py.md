# `MetaGPT\metagpt\tools\openai_text_to_embedding.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : openai_text_to_embedding.py
@Desc    : OpenAI Text-to-Embedding OAS3 api, which provides text-to-embedding functionality.
            For more details, checkout: `https://platform.openai.com/docs/api-reference/embeddings/object`
"""
# 导入所需的模块
from typing import List
import aiohttp
import requests
from pydantic import BaseModel, Field
from metagpt.config import CONFIG
from metagpt.logs import logger

# 定义 Embedding 类，表示由嵌入端点返回的嵌入向量
class Embedding(BaseModel):
    object: str  # The object type, which is always "embedding".
    embedding: List[float]  # The embedding vector, which is a list of floats. The length of vector depends on the model as listed in the embedding guide.
    index: int  # The index of the embedding in the list of embeddings.

# 定义 Usage 类，表示使用情况
class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0

# 定义 ResultEmbedding 类，表示结果嵌入
class ResultEmbedding(BaseModel):
    class Config:
        alias = {"object_": "object"}
    object_: str = ""
    data: List[Embedding] = []
    model: str = ""
    usage: Usage = Field(default_factory=Usage)

# 定义 OpenAIText2Embedding 类，提供文本到嵌入的功能
class OpenAIText2Embedding:
    def __init__(self, openai_api_key):
        """
        :param openai_api_key: OpenAI API key, For more details, checkout: `https://platform.openai.com/account/api-keys`
        """
        self.openai_api_key = openai_api_key or CONFIG.OPENAI_API_KEY

    async def text_2_embedding(self, text, model="text-embedding-ada-002"):
        """Text to embedding

        :param text: The text used for embedding.
        :param model: One of ['text-embedding-ada-002'], ID of the model to use. For more details, checkout: `https://api.openai.com/v1/models`.
        :return: A json object of :class:`ResultEmbedding` class if successful, otherwise `{}`.
        """
        # 设置代理
        proxies = {"proxy": CONFIG.openai_proxy} if CONFIG.openai_proxy else {}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.openai_api_key}"}
        data = {"input": text, "model": model}
        url = "https://api.openai.com/v1/embeddings"
        try:
            # 使用 aiohttp 发送异步请求
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, **proxies) as response:
                    data = await response.json()
                    return ResultEmbedding(**data)
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred:{e}")
        return ResultEmbedding()

# 导出函数，将文本转换为嵌入
async def oas3_openai_text_to_embedding(text, model="text-embedding-ada-002", openai_api_key=""):
    """Text to embedding

    :param text: The text used for embedding.
    :param model: One of ['text-embedding-ada-002'], ID of the model to use. For more details, checkout: `https://api.openai.com/v1/models`.
    :param openai_api_key: OpenAI API key, For more details, checkout: `https://platform.openai.com/account/api-keys`
    :return: A json object of :class:`ResultEmbedding` class if successful, otherwise `{}`.
    """
    if not text:
        return ""
    if not openai_api_key:
        openai_api_key = CONFIG.OPENAI_API_KEY
    return await OpenAIText2Embedding(openai_api_key).text_2_embedding(text, model=model)

```