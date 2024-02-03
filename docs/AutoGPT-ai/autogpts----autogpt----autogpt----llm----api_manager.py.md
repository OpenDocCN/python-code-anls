# `.\AutoGPT\autogpts\autogpt\autogpt\llm\api_manager.py`

```py
from __future__ import annotations
# 导入未来版本的注解特性

import logging
# 导入日志模块
from typing import List, Optional
# 导入类型提示相关模块

from openai import OpenAI
# 从 openai 模块中导入 OpenAI 类
from openai.types import Model
# 从 openai.types 模块中导入 Model 类

from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_MODELS,
    OpenAICredentials,
)
# 从 autogpt.core.resource.model_providers.openai 模块中导入 OPEN_AI_MODELS 和 OpenAICredentials 类
from autogpt.core.resource.model_providers.schema import ChatModelInfo
# 从 autogpt.core.resource.model_providers.schema 模块中导入 ChatModelInfo 类
from autogpt.singleton import Singleton
# 从 autogpt.singleton 模块中导入 Singleton 类

logger = logging.getLogger(__name__)
# 获取当前模块的日志记录器

class ApiManager(metaclass=Singleton):
    # 定义 ApiManager 类，使用 Singleton 元类
    def __init__(self):
        # 初始化方法
        self.total_prompt_tokens = 0
        # 初始化总提示令牌数为 0
        self.total_completion_tokens = 0
        # 初始化总完成令牌数为 0
        self.total_cost = 0
        # 初始化总成本为 0
        self.total_budget = 0
        # 初始化总预算为 0
        self.models: Optional[list[Model]] = None
        # 初始化模型列表为 None

    def reset(self):
        # 重置方法
        self.total_prompt_tokens = 0
        # 重置总提示令牌数为 0
        self.total_completion_tokens = 0
        # 重置总完成令牌数为 0
        self.total_cost = 0
        # 重置总成本为 0
        self.total_budget = 0.0
        # 重置总预算为 0.0
        self.models = None
        # 重置模型列表为 None

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        # 更新成本、提示令牌数和完成令牌数的方法

        # the .model property in API responses can contain version suffixes like -v2
        model = model[:-3] if model.endswith("-v2") else model
        # 如果模型以 "-v2" 结尾，则去掉后三个字符，否则保持不变
        model_info = OPEN_AI_MODELS[model]
        # 获取模型信息

        self.total_prompt_tokens += prompt_tokens
        # 增加总提示令牌数
        self.total_completion_tokens += completion_tokens
        # 增加总完成令牌数
        self.total_cost += prompt_tokens * model_info.prompt_token_cost / 1000
        # 计算总成本，包括提示令牌的成本
        if isinstance(model_info, ChatModelInfo):
            self.total_cost += (
                completion_tokens * model_info.completion_token_cost / 1000
            )
        # 如果模型信息是 ChatModelInfo 类型，则计算完成令牌的成本

        logger.debug(f"Total running cost: ${self.total_cost:.3f}")
        # 记录总运行成本的日志信息
    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        total_budget (float): The total budget for API calls.
        """
        # 设置用户定义的 API 调用总预算
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        # 获取提示令牌的总数
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        # 获取完成令牌的总数
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        # 获取 API 调用的总成本
        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
        float: The total budget for API calls.
        """
        # 获取用户定义的 API 调用总预算
        return self.total_budget

    def get_models(self, openai_credentials: OpenAICredentials) -> List[Model]:
        """
        Get list of available GPT models.

        Returns:
            list[Model]: List of available GPT models.
        """
        # 如果模型列表为空，则获取所有可用的 GPT 模型
        if self.models is None:
            all_models = (
                OpenAI(**openai_credentials.get_api_access_kwargs()).models.list().data
            )
            # 从所有模型中筛选出包含"gpt"的模型
            self.models = [model for model in all_models if "gpt" in model.id]

        return self.models
```