# `.\DB-GPT-src\dbgpt\rag\transformer\llm_extractor.py`

```py
"""TripletExtractor class."""
# 导入日志模块
import logging
# 导入抽象基类和类型提示
from abc import ABC, abstractmethod
from typing import List, Optional

# 导入具体模块和类
from dbgpt.core import HumanPromptTemplate, LLMClient, ModelMessage, ModelRequest
from dbgpt.rag.transformer.base import ExtractorBase

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class LLMExtractor(ExtractorBase, ABC):
    """LLMExtractor class."""

    def __init__(self, llm_client: LLMClient, model_name: str, prompt_template: str):
        """Initialize the LLMExtractor."""
        # 初始化方法，设置私有属性
        self._llm_client = llm_client
        self._model_name = model_name
        self._prompt_template = prompt_template

    async def extract(self, text: str, limit: Optional[int] = None) -> List:
        """Extract by LLm."""
        # 使用 HumanPromptTemplate 对象解析模板
        template = HumanPromptTemplate.from_template(self._prompt_template)
        # 根据文本生成模型消息列表
        messages = template.format_messages(text=text)

        # 如果没有指定模型名称，则使用默认模型
        if not self._model_name:
            # 获取可用的模型列表
            models = await self._llm_client.models()
            # 如果没有可用模型则抛出异常
            if not models:
                raise Exception("No models available")
            # 设置模型名称为第一个可用模型
            self._model_name = models[0].model
            logger.info(f"Using model {self._model_name} to extract")

        # 根据模型消息创建请求对象
        model_messages = ModelMessage.from_base_messages(messages)
        request = ModelRequest(model=self._model_name, messages=model_messages)
        # 发送生成请求并获取响应
        response = await self._llm_client.generate(request=request)

        # 如果生成请求失败，则记录错误并返回空列表
        if not response.success:
            code = str(response.error_code)
            reason = response.text
            logger.error(f"request llm failed ({code}) {reason}")
            return []

        # 如果设置了限制并且限制小于1，则抛出值错误异常
        if limit and limit < 1:
            ValueError("optional argument limit >= 1")
        
        # 解析生成的文本响应，并返回结果列表
        return self._parse_response(response.text, limit)

    @abstractmethod
    def _parse_response(self, text: str, limit: Optional[int] = None) -> List:
        """Parse llm response."""
```