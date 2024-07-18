# `.\graphrag\graphrag\query\structured_search\global_search\callbacks.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""GlobalSearch LLM Callbacks."""

# 导入基类 BaseLLMCallback 以及搜索结果类 SearchResult
from graphrag.query.llm.base import BaseLLMCallback
from graphrag.query.structured_search.base import SearchResult

# 定义 GlobalSearchLLMCallback 类，继承自 BaseLLMCallback
class GlobalSearchLLMCallback(BaseLLMCallback):
    """GlobalSearch LLM Callbacks."""

    # 初始化方法
    def __init__(self):
        super().__init__()
        # 初始化 map_response_contexts 为空列表
        self.map_response_contexts = []
        # 初始化 map_response_outputs 为空列表
        self.map_response_outputs = []

    # 处理 map 响应开始的方法，接收 map_response_contexts 参数
    def on_map_response_start(self, map_response_contexts: list[str]):
        """Handle the start of map response."""
        # 将传入的 map_response_contexts 赋值给实例变量 map_response_contexts
        self.map_response_contexts = map_response_contexts

    # 处理 map 响应结束的方法，接收 map_response_outputs 参数
    def on_map_response_end(self, map_response_outputs: list[SearchResult]):
        """Handle the end of map response."""
        # 将传入的 map_response_outputs 赋值给实例变量 map_response_outputs
        self.map_response_outputs = map_response_outputs
```