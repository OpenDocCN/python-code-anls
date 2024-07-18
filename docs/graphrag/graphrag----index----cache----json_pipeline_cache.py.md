# `.\graphrag\graphrag\index\cache\json_pipeline_cache.py`

```py
# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 Any 类型，用于表示任意类型的数据
from typing import Any

# 从 graphrag.index.storage 中导入 PipelineStorage 类
from graphrag.index.storage import PipelineStorage
# 从当前目录下的 pipeline_cache.py 文件中导入 PipelineCache 类
from .pipeline_cache import PipelineCache


class JsonPipelineCache(PipelineCache):
    """File pipeline cache class definition."""

    # 类变量 _storage，类型为 PipelineStorage，用于存储管道数据
    _storage: PipelineStorage
    # 类变量 _encoding，用于指定编码格式，默认为 "utf-8"
    _encoding: str

    def __init__(self, storage: PipelineStorage, encoding="utf-8"):
        """Init method definition."""
        # 初始化方法，接受 PipelineStorage 实例和编码参数，存储到对应的实例变量中
        self._storage = storage
        self._encoding = encoding

    async def get(self, key: str) -> str | None:
        """Get method definition."""
        # 获取方法，根据键名从存储中获取数据
        if await self.has(key):
            try:
                # 尝试从存储中读取数据，并解析为 JSON 格式
                data = await self._storage.get(key, encoding=self._encoding)
                data = json.loads(data)
            except UnicodeDecodeError:
                # 如果解码错误，删除该键对应的数据并返回 None
                await self._storage.delete(key)
                return None
            except json.decoder.JSONDecodeError:
                # 如果 JSON 解析错误，删除该键对应的数据并返回 None
                await self._storage.delete(key)
                return None
            else:
                # 返回 JSON 数据中的 "result" 字段值
                return data.get("result")

        return None

    async def set(self, key: str, value: Any, debug_data: dict | None = None) -> None:
        """Set method definition."""
        # 设置方法，根据键名设置数据
        if value is None:
            return
        # 构建包含 "result" 字段和调试数据的字典
        data = {"result": value, **(debug_data or {})}
        # 将数据转换为 JSON 格式并存储到指定的键名位置
        await self._storage.set(key, json.dumps(data), encoding=self._encoding)

    async def has(self, key: str) -> bool:
        """Has method definition."""
        # 判断指定的键名是否存在于存储中
        return await self._storage.has(key)

    async def delete(self, key: str) -> None:
        """Delete method definition."""
        # 删除指定的键名及其对应的数据
        if await self.has(key):
            await self._storage.delete(key)

    async def clear(self) -> None:
        """Clear method definition."""
        # 清空所有存储的数据
        await self._storage.clear()

    def child(self, name: str) -> "JsonPipelineCache":
        """Child method definition."""
        # 创建一个新的 JsonPipelineCache 实例作为子实例，继承当前实例的存储和编码设置
        return JsonPipelineCache(self._storage.child(name), encoding=self._encoding)
```