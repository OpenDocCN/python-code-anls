# `.\DB-GPT-src\dbgpt\storage\cache\llm_cache.py`

```py
"""Cache client for LLM."""

import hashlib  # 导入 hashlib 库，用于生成哈希值
from dataclasses import asdict, dataclass  # 导入 dataclasses 模块中的 asdict 和 dataclass 装饰器
from typing import Any, Dict, List, Optional, Union, cast  # 导入类型提示相关的模块

from dbgpt.core import ModelOutput  # 从 dbgpt.core 模块中导入 ModelOutput 类
from dbgpt.core.interface.cache import CacheClient, CacheConfig, CacheKey, CacheValue  # 从 dbgpt.core.interface.cache 模块导入缓存相关的类

from .manager import CacheManager  # 从当前包中的 manager 模块导入 CacheManager 类


@dataclass
class LLMCacheKeyData:
    """Cache key data for LLM."""
    
    prompt: str  # 字段：提示信息，字符串类型
    model_name: str  # 字段：模型名称，字符串类型
    temperature: Optional[float] = 0.7  # 字段：温度，可选的浮点数，默认为 0.7
    max_new_tokens: Optional[int] = None  # 字段：最大新标记数，可选的整数，默认为 None
    top_p: Optional[float] = 1.0  # 字段：top-p 参数，可选的浮点数，默认为 1.0
    # See dbgpt.model.base.ModelType
    model_type: Optional[str] = "huggingface"  # 字段：模型类型，可选的字符串，默认为 "huggingface"


CacheOutputType = Union[ModelOutput, List[ModelOutput]]  # 定义 CacheOutputType 类型别名，可以是 ModelOutput 或 ModelOutput 列表类型之一


@dataclass
class LLMCacheValueData:
    """Cache value data for LLM."""

    output: CacheOutputType  # 字段：输出结果，类型为 CacheOutputType
    user: Optional[str] = None  # 字段：用户，可选的字符串，默认为 None
    _is_list: bool = False  # 字段：是否为列表，布尔类型，默认为 False

    @staticmethod
    def from_dict(**kwargs) -> "LLMCacheValueData":
        """Create LLMCacheValueData object from dict."""
        output = kwargs.get("output")  # 从参数中获取键为 "output" 的值
        if not output:
            raise ValueError("Can't new LLMCacheValueData object, output is None")  # 如果 output 为空，则抛出 ValueError 异常
        if isinstance(output, dict):  # 如果 output 是字典类型
            output = ModelOutput(**output)  # 使用 output 字典创建 ModelOutput 对象
        elif isinstance(output, list):  # 如果 output 是列表类型
            kwargs["_is_list"] = True  # 设置 _is_list 属性为 True
            output_list = []
            for out in output:
                if isinstance(out, dict):
                    out = ModelOutput(**out)
                output_list.append(out)
            output = output_list
        kwargs["output"] = output
        return LLMCacheValueData(**kwargs)  # 返回根据参数创建的 LLMCacheValueData 对象

    def to_dict(self) -> Dict:
        """Convert to dict."""
        output = self.output  # 获取 output 属性
        is_list = False
        if isinstance(output, list):  # 如果 output 是列表类型
            output_list = []
            is_list = True
            for out in output:
                output_list.append(out.to_dict())  # 将列表中每个元素转换为字典并添加到 output_list 中
            output = output_list  # 设置 output 为转换后的列表
        else:
            output = output.to_dict()  # 否则将 output 转换为字典类型
        return {"output": output, "_is_list": is_list, "user": self.user}  # 返回包含转换后信息的字典

    @property
    def is_list(self) -> bool:
        """Return whether the output is a list."""
        return self._is_list  # 返回 _is_list 属性值

    def __str__(self) -> str:
        """Return string representation."""
        if not isinstance(self.output, list):
            return f"user: {self.user}, output: {self.output}"  # 如果 output 不是列表，则返回用户和输出信息
        else:
            return f"user: {self.user}, output(last two item): {self.output[-2:]}"  # 否则返回用户和最后两个输出项的信息


class LLMCacheKey(CacheKey[LLMCacheKeyData]):
    """Cache key for LLM."""

    def __init__(self, **kwargs) -> None:
        """Create a new instance of LLMCacheKey."""
        super().__init__()  # 调用父类的初始化方法
        self.config = LLMCacheKeyData(**kwargs)  # 根据传入的关键字参数创建 LLMCacheKeyData 对象并赋值给 config 属性

    def __hash__(self) -> int:
        """Return the hash value of the object."""
        serialize_bytes = self.serialize()  # 调用 serialize 方法获取序列化后的字节流
        return int(hashlib.sha256(serialize_bytes).hexdigest(), 16)  # 使用 SHA-256 算法计算哈希值并返回整数形式
    # 检查当前对象是否与另一个键相等
    def __eq__(self, other: Any) -> bool:
        """Check equality with another key."""
        # 如果另一个对象不是LLMCacheKey类型，则返回False
        if not isinstance(other, LLMCacheKey):
            return False
        # 返回当前配置与另一个对象配置是否相等的结果
        return self.config == other.config

    # 返回哈希值的字节数组
    def get_hash_bytes(self) -> bytes:
        """Return the byte array of hash value.

        Returns:
            bytes: The byte array of hash value.
        """
        # 序列化当前对象并计算SHA256哈希值的字节数组
        serialize_bytes = self.serialize()
        return hashlib.sha256(serialize_bytes).digest()

    # 将对象转换为字典
    def to_dict(self) -> Dict:
        """Convert to dict."""
        # 使用asdict函数将配置转换为字典并返回
        return asdict(self.config)

    # 返回当前缓存键的真实对象
    def get_value(self) -> LLMCacheKeyData:
        """Return the real object of current cache key."""
        # 返回当前配置对象
        return self.config
class LLMCacheValue(CacheValue[LLMCacheValueData]):
    """Cache value for LLM."""

    def __init__(self, **kwargs) -> None:
        """Create a new instance of LLMCacheValue."""
        # 调用父类的初始化方法
        super().__init__()
        # 根据传入的关键字参数创建LLMCacheValueData对象并赋值给实例变量value
        self.value = LLMCacheValueData.from_dict(**kwargs)

    def to_dict(self) -> Dict:
        """Convert to dict."""
        # 调用value对象的to_dict方法，将对象转换成字典并返回
        return self.value.to_dict()

    def get_value(self) -> LLMCacheValueData:
        """Return the underlying real value."""
        # 返回实例变量value，即LLMCacheValueData对象
        return self.value

    def __str__(self) -> str:
        """Return string representation."""
        # 返回LLMCacheValue对象的字符串表示形式，包含"value: "前缀
        return f"value: {str(self.value)}"


class LLMCacheClient(CacheClient[LLMCacheKeyData, LLMCacheValueData]):
    """Cache client for LLM."""

    def __init__(self, cache_manager: CacheManager) -> None:
        """Create a new instance of LLMCacheClient."""
        # 调用父类的初始化方法，传入缓存管理器对象
        super().__init__()
        # 设置实例变量_cache_manager，存储传入的缓存管理器对象
        self._cache_manager: CacheManager = cache_manager

    async def get(
        self,
        key: LLMCacheKey,  # type: ignore
        cache_config: Optional[CacheConfig] = None,
    ) -> Optional[LLMCacheValue]:
        """Retrieve a value from the cache using the provided key.

        Args:
            key (LLMCacheKey): The key to get cache
            cache_config (Optional[CacheConfig]): Cache config

        Returns:
            Optional[LLMCacheValue]: The value retrieved according to key. If cache key
                not exist, return None.
        """
        # 调用缓存管理器的get方法，根据key和LLMCacheValue类型获取缓存中的值
        return cast(
            LLMCacheValue,
            await self._cache_manager.get(key, LLMCacheValue, cache_config),
        )

    async def set(
        self,
        key: LLMCacheKey,  # type: ignore
        value: LLMCacheValue,  # type: ignore
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        """Set a value in the cache for the provided key."""
        # 调用缓存管理器的set方法，将key和value存入缓存
        return await self._cache_manager.set(key, value, cache_config)

    async def exists(
        self,
        key: LLMCacheKey,  # type: ignore
        cache_config: Optional[CacheConfig] = None,
    ) -> bool:
        """Check if a key exists in the cache."""
        # 调用get方法检查缓存中是否存在指定的key
        return await self.get(key, cache_config) is not None

    def new_key(self, **kwargs) -> LLMCacheKey:  # type: ignore
        """Create a cache key with params."""
        # 使用传入的关键字参数创建LLMCacheKey对象
        key = LLMCacheKey(**kwargs)
        # 设置key对象的序列化器为_cache_manager的序列化器
        key.set_serializer(self._cache_manager.serializer)
        return key

    def new_value(self, **kwargs) -> LLMCacheValue:  # type: ignore
        """Create a cache value with params."""
        # 使用传入的关键字参数创建LLMCacheValue对象
        value = LLMCacheValue(**kwargs)
        # 设置value对象的序列化器为_cache_manager的序列化器
        value.set_serializer(self._cache_manager.serializer)
        return value
```