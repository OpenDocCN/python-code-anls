# `MetaGPT\metagpt\provider\google_gemini_api.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : Google Gemini LLM from https://ai.google.dev/tutorials/python_quickstart
# 导入所需的模块和库

import google.generativeai as genai
from google.ai import generativelanguage as glm
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import content_types
from google.generativeai.types.generation_types import (
    AsyncGenerateContentResponse,
    GenerateContentResponse,
    GenerationConfig,
)
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
# 导入所需的模块和库

from metagpt.config import CONFIG, LLMProviderEnum
from metagpt.logs import log_llm_stream, logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.provider.openai_api import log_and_reraise
# 导入所需的模块和库

class GeminiGenerativeModel(GenerativeModel):
    """
    Due to `https://github.com/google/generative-ai-python/pull/123`, inherit a new class.
    Will use default GenerativeModel if it fixed.
    """
    # 创建 GeminiGenerativeModel 类，继承自 GenerativeModel 类

    def count_tokens(self, contents: content_types.ContentsType) -> glm.CountTokensResponse:
        contents = content_types.to_contents(contents)
        return self._client.count_tokens(model=self.model_name, contents=contents)
    # 定义 count_tokens 方法，用于计算内容中的标记数量

    async def count_tokens_async(self, contents: content_types.ContentsType) -> glm.CountTokensResponse:
        contents = content_types.to_contents(contents)
        return await self._async_client.count_tokens(model=self.model_name, contents=contents)
    # 定义 count_tokens_async 方法，用于异步计算内容中的标记数量

@register_provider(LLMProviderEnum.GEMINI)
# 注册 LLMProviderEnum.GEMINI 类型的提供者

```