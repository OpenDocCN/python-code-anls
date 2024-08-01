# `.\DB-GPT-src\dbgpt\agent\util\llm\strategy\priority.py`

```py
"""Priority strategy for LLM."""

import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
from typing import List, Optional  # 导入类型提示模块

from ..llm import LLMStrategy, LLMStrategyType  # 导入 LLMStrategy 相关模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LLMStrategyPriority(LLMStrategy):
    """Priority strategy for llm model service."""

    @property
    def type(self) -> LLMStrategyType:
        """Return the strategy type."""
        return LLMStrategyType.Priority  # 返回当前策略的类型为 Priority

    async def next_llm(self, excluded_models: Optional[List[str]] = None) -> str:
        """Return next available llm model name."""
        try:
            if not excluded_models:
                excluded_models = []  # 如果排除模型列表为空，则初始化为空列表

            all_models = await self._llm_client.models()  # 获取所有可用的模型列表

            if not self._context:
                raise ValueError("No context provided for priority strategy!")  # 如果没有提供上下文，则抛出 ValueError

            priority: List[str] = json.loads(self._context)  # 解析上下文中的优先级列表

            can_uses = self._excluded_models(all_models, excluded_models, priority)  # 根据优先级和排除模型筛选可用模型

            if can_uses and len(can_uses) > 0:
                return can_uses[0].model  # 返回第一个可用模型的名称
            else:
                raise ValueError("No model service available!")  # 如果没有可用模型，则抛出 ValueError

        except Exception as e:
            logger.error(f"{self.type} get next llm failed!{str(e)}")  # 记录错误日志
            raise ValueError(f"Failed to allocate model service,{str(e)}!")  # 抛出带有详细错误信息的 ValueError
```