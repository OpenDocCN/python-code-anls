# `MetaGPT\metagpt\provider\llm_provider_registry.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19 17:26
@Author  : alexanderwu
@File    : llm_provider_registry.py
"""
# 导入 LLMProviderEnum 枚举类型
from metagpt.config import LLMProviderEnum

# 定义 LLMProviderRegistry 类
class LLMProviderRegistry:
    def __init__(self):
        # 初始化 providers 字典
        self.providers = {}

    # 注册方法，将 provider_cls 与 key 绑定存储到 providers 字典中
    def register(self, key, provider_cls):
        self.providers[key] = provider_cls

    # 获取 provider 实例的方法，根据传入的 LLMProviderEnum 枚举类型返回对应的 provider 实例
    def get_provider(self, enum: LLMProviderEnum):
        """get provider instance according to the enum"""
        return self.providers[enum]()

# 创建 LLMProviderRegistry 的实例
LLM_REGISTRY = LLMProviderRegistry()

# 注册 provider 到 registry 的装饰器函数
def register_provider(key):
    """register provider to registry"""

    # 装饰器函数，将 cls 与 key 绑定存储到 LLM_REGISTRY 中
    def decorator(cls):
        LLM_REGISTRY.register(key, cls)
        return cls

    return decorator

```