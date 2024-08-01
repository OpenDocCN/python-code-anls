# `.\DB-GPT-src\dbgpt\agent\util\llm\llm.py`

```py
"""LLM module."""
# 导入日志模块
import logging
# 导入默认字典模块
from collections import defaultdict
# 导入枚举模块
from enum import Enum
# 导入类型提示模块
from typing import Any, Dict, List, Optional, Type

# 导入基础模型、配置字典和字段模块
from dbgpt._private.pydantic import BaseModel, ConfigDict, Field
# 导入LLM客户端、模型元数据和模型请求模块
from dbgpt.core import LLMClient, ModelMetadata, ModelRequest

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def _build_model_request(input_value: Dict) -> ModelRequest:
    """Build model request from input value.

    Args:
        input_value(str or dict): input value

    Returns:
        ModelRequest: model request, pass to llm client
    """
    # 从输入值构建模型请求参数字典
    parm = {
        "model": input_value.get("model"),
        "messages": input_value.get("messages"),
        "temperature": input_value.get("temperature", None),
        "max_new_tokens": input_value.get("max_new_tokens", None),
        "stop": input_value.get("stop", None),
        "stop_token_ids": input_value.get("stop_token_ids", None),
        "context_len": input_value.get("context_len", None),
        "echo": input_value.get("echo", None),
        "span_id": input_value.get("span_id", None),
    }

    # 使用参数字典创建并返回模型请求对象
    return ModelRequest(**parm)


class LLMStrategyType(Enum):
    """LLM strategy type."""

    # LLM策略类型枚举定义
    Priority = "priority"
    Auto = "auto"
    Default = "default"


class LLMStrategy:
    """LLM strategy base class."""

    def __init__(self, llm_client: LLMClient, context: Optional[str] = None):
        """Create an LLMStrategy instance."""
        # 初始化LLM策略对象，包括LLM客户端和上下文信息
        self._llm_client = llm_client
        self._context = context

    @property
    def type(self) -> LLMStrategyType:
        """Return the strategy type."""
        # 返回LLM策略对象的类型
        return LLMStrategyType.Default

    def _excluded_models(
        self,
        all_models: List[ModelMetadata],
        excluded_models: List[str],
        need_uses: Optional[List[str]] = None,
    ):
        # 获取排除指定模型后可用模型列表的私有方法
        if not need_uses:
            need_uses = []
        can_uses = []
        for item in all_models:
            if item.model in need_uses and item.model not in excluded_models:
                can_uses.append(item)
        return can_uses

    async def next_llm(self, excluded_models: Optional[List[str]] = None):
        """Return next available llm model name.

        Args:
            excluded_models(List[str]): excluded models

        Returns:
            str: Next available llm model name
        """
        # 异步获取下一个可用LLM模型名称的方法
        if not excluded_models:
            excluded_models = []
        try:
            all_models = await self._llm_client.models()
            available_llms = self._excluded_models(all_models, excluded_models, None)
            if available_llms and len(available_llms) > 0:
                return available_llms[0].model
            else:
                raise ValueError("No model service available!")

        except Exception as e:
            # 记录错误日志并抛出异常
            logger.error(f"{self.type} get next llm failed!{str(e)}")
            raise ValueError(f"Failed to allocate model service,{str(e)}!")


# LLM策略类型到策略类列表的默认字典映射
llm_strategies: Dict[LLMStrategyType, List[Type[LLMStrategy]]] = defaultdict(list)
# 注册LLM策略函数，将指定LLM策略类型与策略类对象关联起来
def register_llm_strategy(
    llm_strategy_type: LLMStrategyType, strategy: Type[LLMStrategy]
):
    """Register llm strategy."""
    # 将策略类对象添加到对应LLM策略类型的策略列表中
    llm_strategies[llm_strategy_type].append(strategy)


class LLMConfig(BaseModel):
    """LLM configuration."""

    # 模型配置字典，允许任意类型的值
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # LLM客户端，可选，默认使用LLMClient的默认工厂值
    llm_client: Optional[LLMClient] = Field(default_factory=LLMClient)
    
    # LLM策略类型，默认为LLMStrategyType.Default
    llm_strategy: LLMStrategyType = Field(default=LLMStrategyType.Default)
    
    # 策略上下文，可选
    strategy_context: Optional[Any] = None
```