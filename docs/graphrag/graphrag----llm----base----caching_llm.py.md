# `.\graphrag\graphrag\llm\base\caching_llm.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A class to interact with the cache."""

# 导入 JSON 模块，用于序列化和反序列化 JSON 数据
import json
# 导入类型相关的模块
from typing import Any, Generic, TypeVar

# 导入类型扩展模块
from typing_extensions import Unpack

# 导入自定义类型
from graphrag.llm.types import LLM, LLMCache, LLMInput, LLMOutput, OnCacheActionFn

# 导入创建缓存键的方法
from ._create_cache_key import create_hash_key

# 定义缓存策略的版本号，用于处理缓存项变更时的兼容性
_cache_strategy_version = 2

# 定义类型变量
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

# 定义一个空操作的缓存函数
def _noop_cache_fn(_k: str, _v: str | None):
    pass

# 定义缓存化语言模型的类，继承自泛型类 LLM
class CachingLLM(LLM[TIn, TOut], Generic[TIn, TOut]):
    """A class to interact with the cache."""

    _cache: LLMCache  # 缓存对象
    _delegate: LLM[TIn, TOut]  # 委托的语言模型对象
    _operation: str  # 操作名称
    _llm_paramaters: dict  # 语言模型参数
    _on_cache_hit: OnCacheActionFn  # 缓存命中时执行的函数
    _on_cache_miss: OnCacheActionFn  # 缓存未命中时执行的函数

    # 初始化方法，接受委托对象、语言模型参数、操作名称和缓存对象作为参数
    def __init__(
        self,
        delegate: LLM[TIn, TOut],
        llm_parameters: dict,
        operation: str,
        cache: LLMCache,
    ):
        self._delegate = delegate  # 设置委托对象
        self._llm_paramaters = llm_parameters  # 设置语言模型参数
        self._cache = cache  # 设置缓存对象
        self._operation = operation  # 设置操作名称
        self._on_cache_hit = _noop_cache_fn  # 设置默认的缓存命中函数
        self._on_cache_miss = _noop_cache_fn  # 设置默认的缓存未命中函数

    # 设置缓存命中时执行的函数
    def on_cache_hit(self, fn: OnCacheActionFn | None) -> None:
        """Set the function to call when a cache hit occurs."""
        self._on_cache_hit = fn or _noop_cache_fn

    # 设置缓存未命中时执行的函数
    def on_cache_miss(self, fn: OnCacheActionFn | None) -> None:
        """Set the function to call when a cache miss occurs."""
        self._on_cache_miss = fn or _noop_cache_fn

    # 构建缓存键的方法，基于输入、名称和参数构建哈希键
    def _cache_key(self, input: TIn, name: str | None, args: dict) -> str:
        json_input = json.dumps(input)  # 将输入对象序列化为 JSON 字符串
        tag = (
            f"{name}-{self._operation}-v{_cache_strategy_version}"
            if name is not None
            else self._operation
        )
        return create_hash_key(tag, json_input, args)  # 调用创建哈希键的方法生成最终的缓存键

    # 异步方法，从缓存中读取值
    async def _cache_read(self, key: str) -> Any | None:
        """Read a value from the cache."""
        return await self._cache.get(key)  # 使用缓存对象异步获取指定键的值

    # 异步方法，向缓存中写入值
    async def _cache_write(
        self, key: str, input: TIn, result: TOut | None, args: dict
    ) -> None:
        """Write a value to the cache."""
        if result:
            await self._cache.set(
                key,
                result,
                {
                    "input": input,
                    "parameters": args,
                },
            )  # 如果结果不为空，使用缓存对象异步设置指定键的值

    # 实现可调用对象的方法，接受输入参数和附加参数
    async def __call__(
        self,
        input: TIn,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[TOut]:
        """Execute the LLM."""
        # 检查是否存在缓存项
        name = kwargs.get("name")
        llm_args = {**self._llm_paramaters, **(kwargs.get("model_parameters") or {})}
        # 生成缓存键
        cache_key = self._cache_key(input, name, llm_args)
        # 从缓存中读取结果
        cached_result = await self._cache_read(cache_key)
        if cached_result:
            # 如果命中缓存，执行缓存命中处理
            self._on_cache_hit(cache_key, name)
            # 返回命中缓存的结果
            return LLMOutput(output=cached_result)

        # 报告缓存未命中
        self._on_cache_miss(cache_key, name)

        # 计算新的结果
        result = await self._delegate(input, **kwargs)
        # 将新结果写入缓存
        await self._cache_write(cache_key, input, result.output, llm_args)
        # 返回最终结果
        return result
```