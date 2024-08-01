# `.\DB-GPT-src\dbgpt\agent\resource\knowledge.py`

```py
"""Knowledge resource."""

import dataclasses  # 导入用于定义数据类的模块
from typing import TYPE_CHECKING, Any, List, Optional, Type  # 导入类型提示相关模块

import cachetools  # 导入缓存工具模块

from dbgpt.util.cache_utils import cached  # 导入自定义的缓存装饰器

from .base import Resource, ResourceParameters, ResourceType  # 导入相对路径的模块

if TYPE_CHECKING:
    from dbgpt.core import Chunk  # 条件导入，用于类型检查
    from dbgpt.rag.retriever.base import BaseRetriever  # 条件导入，用于类型检查
    from dbgpt.storage.vector_store.filters import MetadataFilters  # 条件导入，用于类型检查


@dataclasses.dataclass
class RetrieverResourceParameters(ResourceParameters):
    """Retriever resource parameters."""
    pass  # 数据类，用于存储检索器资源的参数


class RetrieverResource(Resource[ResourceParameters]):
    """Retriever resource.

    Retrieve knowledge chunks from a retriever.
    """

    def __init__(self, name: str, retriever: "BaseRetriever"):
        """Create a new RetrieverResource."""
        self._name = name  # 初始化资源名称
        self._retriever = retriever  # 初始化检索器对象

    @property
    def name(self) -> str:
        """Return the resource name."""
        return self._name  # 返回资源名称

    @property
    def retriever(self) -> "BaseRetriever":
        """Return the retriever."""
        return self._retriever  # 返回检索器对象

    @classmethod
    def type(cls) -> ResourceType:
        """Return the resource type."""
        return ResourceType.Knowledge  # 返回资源类型为知识类型

    @classmethod
    def resource_parameters_class(cls) -> Type[ResourceParameters]:
        """Return the resource parameters class."""
        return RetrieverResourceParameters  # 返回资源参数的数据类

    @cached(cachetools.TTLCache(maxsize=100, ttl=10))
    async def get_prompt(
        self,
        *,
        lang: str = "en",
        prompt_type: str = "default",
        question: Optional[str] = None,
        resource_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Get the prompt for the resource."""
        if not question:
            raise ValueError("Question is required for knowledge resource.")  # 如果没有提供问题，则抛出错误
        chunks = await self.retrieve(question)  # 调用retrieve方法获取知识块
        content = "\n".join([chunk.content for chunk in chunks])  # 将知识块的内容连接成字符串
        prompt_template = "known information: {content}"  # 英文提示模板
        prompt_template_zh = "已知信息: {content}"  # 中文提示模板
        if lang == "en":
            return prompt_template.format(content=content)  # 返回英文格式的提示
        return prompt_template_zh.format(content=content)  # 返回中文格式的提示

    async def async_execute(
        self, *args, resource_name: Optional[str] = None, **kwargs
    ) -> Any:
        """Execute the resource asynchronously."""
        return await self.retrieve(*args, **kwargs)  # 异步执行资源的检索操作

    async def retrieve(
        self, query: str, filters: Optional["MetadataFilters"] = None
    ) -> List["Chunk"]:
        """Retrieve knowledge chunks.

        Args:
            query (str): query text.
            filters: (Optional[MetadataFilters]) metadata filters.

        Returns:
            List[Chunk]: list of chunks
        """
        return await self.retriever.aretrieve(query, filters)  # 调用检索器的aretrieve方法，获取知识块列表
```