# `.\DB-GPT-src\dbgpt\serve\rag\retriever\knowledge_space.py`

```py
from typing import List, Optional  # 引入必要的类型提示

from dbgpt._private.config import Config  # 从私有模块导入配置类
from dbgpt.component import ComponentType  # 导入组件类型定义
from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG  # 导入嵌入模型配置
from dbgpt.core import Chunk  # 导入核心模块的 Chunk 类
from dbgpt.rag.embedding.embedding_factory import EmbeddingFactory  # 导入嵌入工厂类
from dbgpt.rag.retriever.base import BaseRetriever  # 导入基础检索器类
from dbgpt.serve.rag.connector import VectorStoreConnector  # 导入向量存储连接器类
from dbgpt.storage.vector_store.filters import MetadataFilters  # 导入元数据过滤器类
from dbgpt.util.executor_utils import ExecutorFactory, blocking_func_to_async  # 导入执行工具类和异步转换函数

CFG = Config()  # 创建一个配置对象

class KnowledgeSpaceRetriever(BaseRetriever):
    """Knowledge Space retriever."""

    def __init__(
        self,
        space_name: str = None,  # 初始化函数，接受知识空间名称和可选的 top k 参数
        top_k: Optional[int] = 4,
    ):
        """
        Args:
            space_name (str): knowledge space name 知识空间的名称
            top_k (Optional[int]): top k 可选的前 k 个结果数
        """
        if space_name is None:
            raise ValueError("space_name is required")  # 如果没有提供空间名称，则引发值错误
        self._space_name = space_name  # 将空间名称保存到实例变量中
        self._top_k = top_k  # 将 top k 参数保存到实例变量中
        embedding_factory = CFG.SYSTEM_APP.get_component(
            "embedding_factory", EmbeddingFactory
        )  # 从配置对象中获取嵌入工厂组件
        embedding_fn = embedding_factory.create(
            model_name=EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]
        )  # 根据配置中的嵌入模型名称创建嵌入函数
        from dbgpt.storage.vector_store.base import VectorStoreConfig

        config = VectorStoreConfig(name=self._space_name, embedding_fn=embedding_fn)  # 创建向量存储配置对象
        self._vector_store_connector = VectorStoreConnector(
            vector_store_type=CFG.VECTOR_STORE_TYPE,
            vector_store_config=config,
        )  # 创建向量存储连接器对象，传入存储类型和配置对象
        self._executor = CFG.SYSTEM_APP.get_component(
            ComponentType.EXECUTOR_DEFAULT, ExecutorFactory
        ).create()  # 从系统应用配置中获取默认执行器组件并创建实例

    def _retrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Retrieve knowledge chunks.

        Args:
            query (str): query text. 查询文本
            filters: (Optional[MetadataFilters]) metadata filters. 可选的元数据过滤器

        Return:
            List[Chunk]: list of chunks 返回块的列表
        """
        candidates = self._vector_store_connector.similar_search(
            doc=query, topk=self._top_k, filters=filters
        )  # 使用向量存储连接器执行相似搜索
        return candidates  # 返回搜索结果列表

    def _retrieve_with_score(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Retrieve knowledge chunks with score.

        Args:
            query (str): query text 查询文本
            score_threshold (float): score threshold 分数阈值
            filters: (Optional[MetadataFilters]) metadata filters 可选的元数据过滤器

        Return:
            List[Chunk]: list of chunks with score 返回带有分数的块列表
        """
        candidates_with_score = self._vector_store_connector.similar_search_with_scores(
            doc=query,
            topk=self._top_k,
            score_threshold=score_threshold,
            filters=filters,
        )  # 使用向量存储连接器执行带有分数的相似搜索
        return candidates_with_score  # 返回带有分数的搜索结果列表
    # 异步函数，用于检索知识块
    async def _aretrieve(
        self, query: str, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Retrieve knowledge chunks.

        Args:
            query (str): query text. 查询文本
            filters: (Optional[MetadataFilters]) metadata filters. 元数据过滤器

        Return:
            List[Chunk]: list of chunks 返回知识块列表
        """
        # 调用阻塞函数转换为异步函数，使用执行器和检索函数来获取候选结果
        candidates = await blocking_func_to_async(
            self._executor, self._retrieve, query, filters
        )
        return candidates

    # 异步函数，用于检索带有分数的知识块
    async def _aretrieve_with_score(
        self,
        query: str,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Retrieve knowledge chunks with score.

        Args:
            query (str): query text. 查询文本
            score_threshold (float): score threshold. 分数阈值
            filters: (Optional[MetadataFilters]) metadata filters. 元数据过滤器

        Return:
            List[Chunk]: list of chunks with score. 带有分数的知识块列表
        """
        # 调用阻塞函数转换为异步函数，使用执行器和带有分数的检索函数来获取候选结果
        candidates_with_score = await blocking_func_to_async(
            self._executor, self._retrieve_with_score, query, score_threshold, filters
        )
        return candidates_with_score
```