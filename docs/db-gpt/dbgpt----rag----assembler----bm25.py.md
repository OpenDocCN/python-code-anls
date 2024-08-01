# `.\DB-GPT-src\dbgpt\rag\assembler\bm25.py`

```py
"""BM25 Assembler."""
# 导入所需的模块
import json
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, List, Optional

# 导入自定义模块
from dbgpt.core import Chunk

# 导入相关模块
from ...storage.vector_store.elastic_store import ElasticsearchVectorConfig
from ...util.executor_utils import blocking_func_to_async
from ..assembler.base import BaseAssembler
from ..chunk_manager import ChunkParameters
from ..knowledge.base import Knowledge
from ..retriever.bm25 import BM25Retriever

# 定义 BM25Assembler 类，继承自 BaseAssembler 类
class BM25Assembler(BaseAssembler):
    """BM25 Assembler.

    refer https://www.elastic.co/guide/en/elasticsearch/reference/8.9/index-
    modules-similarity.html
    TF/IDF based similarity that has built-in tf normalization and is supposed to
    work better for short fields (like names). See Okapi_BM25 for more details.
    This similarity has the following options:

    Example:
    .. code-block:: python

        from dbgpt.rag.assembler import BM25Assembler

        pdf_path = "path/to/document.pdf"
        knowledge = KnowledgeFactory.from_file_path(pdf_path)
        assembler = BM25Assembler.load_from_knowledge(
            knowledge=knowledge,
            es_config=es_config,
            chunk_parameters=chunk_parameters,
        )
        assembler.persist()
        # get bm25 retriever
        retriever = assembler.as_retriever(3)
        chunks = retriever.retrieve_with_scores("what is awel talk about", 0.3)
        print(f"bm25 rag example results:{chunks}")
    """

    # 初始化 BM25Assembler 类的实例
    def __init__(
        self,
        knowledge: Knowledge,
        es_config: ElasticsearchVectorConfig,
        k1: Optional[float] = 2.0,
        b: Optional[float] = 0.75,
        chunk_parameters: Optional[ChunkParameters] = None,
        executor: Optional[Executor] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with BM25 Assembler arguments.

        Args:
            knowledge: (Knowledge) Knowledge datasource.
            es_config: (ElasticsearchVectorConfig) Elasticsearch config.
            k1 (Optional[float]): Controls non-linear term frequency normalization
            (saturation). The default value is 2.0.
            b (Optional[float]): Controls to what degree document length normalizes
            tf values. The default value is 0.75.
            chunk_parameters: (Optional[ChunkParameters]) ChunkManager to use for
                chunking.
        """
        # 导入 Elasticsearch 模块
        from elasticsearch import Elasticsearch

        # 设置 Elasticsearch 相关属性
        self._es_config = es_config
        self._es_url = es_config.uri
        self._es_port = es_config.port
        self._es_username = es_config.user
        self._es_password = es_config.password
        self._index_name = es_config.name
        self._k1 = k1
        self._b = b

        # 根据用户名和密码创建 Elasticsearch 客户端连接
        if self._es_username and self._es_password:
            self._es_client = Elasticsearch(
                hosts=[f"http://{self._es_url}:{self._es_port}"],
                basic_auth=(self._es_username, self._es_password),
            )
        else:
            # 创建没有认证信息的 Elasticsearch 客户端连接
            self._es_client = Elasticsearch(
                hosts=[f"http://{self._es_url}:{self._es_port}"],
            )

        # Elasticsearch 索引设置
        self._es_index_settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                }
            },
        }

        # Elasticsearch 映射设置
        self._es_mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",
                },
                "metadata": {
                    "type": "keyword",
                },
            }
        }

        # 初始化线程池执行器
        self._executor = executor or ThreadPoolExecutor()

        # 检查知识数据源是否为 None，若是则抛出 ValueError
        if knowledge is None:
            raise ValueError("knowledge datasource must be provided.")

        # 如果指定的索引名称在 Elasticsearch 中不存在，则创建该索引
        if not self._es_client.indices.exists(index=self._index_name):
            self._es_client.indices.create(
                index=self._index_name,
                mappings=self._es_mappings,
                settings=self._es_index_settings,
            )

        # 调用父类的初始化方法，传递知识数据源和其他参数
        super().__init__(
            knowledge=knowledge,
            chunk_parameters=chunk_parameters,
            **kwargs,
        )

    @classmethod
    def load_from_knowledge(
        cls,
        knowledge: Knowledge,
        es_config: ElasticsearchVectorConfig,
        k1: Optional[float] = 2.0,
        b: Optional[float] = 0.75,
        chunk_parameters: Optional[ChunkParameters] = None,
    ) -> "BM25Assembler":
        """返回一个BM25Assembler对象。

        加载文档全文到Elasticsearch中的路径。

        Args:
            knowledge: (Knowledge) 知识数据源。
            es_config: (ElasticsearchVectorConfig) Elasticsearch配置。
            k1: (Optional[float]) BM25参数k1。
            b: (Optional[float]) BM25参数b。
            chunk_parameters: (Optional[ChunkParameters]) 用于分块的ChunkManager。

        Returns:
             BM25Assembler对象
        """
        return cls(
            knowledge=knowledge,
            es_config=es_config,
            k1=k1,
            b=b,
            chunk_parameters=chunk_parameters,
        )

    @classmethod
    async def aload_from_knowledge(
        cls,
        knowledge: Knowledge,
        es_config: ElasticsearchVectorConfig,
        k1: Optional[float] = 2.0,
        b: Optional[float] = 0.75,
        chunk_parameters: Optional[ChunkParameters] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> "BM25Assembler":
        """从知识数据源加载文档全文到Elasticsearch中的路径。

        Args:
            knowledge: (Knowledge) 知识数据源。
            es_config: (ElasticsearchVectorConfig) Elasticsearch配置。
            k1: (Optional[float]) BM25参数k1。
            b: (Optional[float]) BM25参数b。
            chunk_parameters: (Optional[ChunkParameters]) 用于分块的ChunkManager。
            executor: (Optional[ThreadPoolExecutor]) 执行器。

        Returns:
             BM25Assembler对象
        """
        return await blocking_func_to_async(
            executor,
            cls,
            knowledge,
            es_config=es_config,
            k1=k1,
            b=b,
            chunk_parameters=chunk_parameters,
        )

    def persist(self, **kwargs) -> List[str]:
        """将分块持久化到Elasticsearch中。

        Returns:
            List[str]: 分块ID列表。
        """
        try:
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ValueError("请安装包 `pip install elasticsearch`.")
        es_requests = []
        ids = []
        contents = [chunk.content for chunk in self._chunks]
        metadatas = [json.dumps(chunk.metadata) for chunk in self._chunks]
        chunk_ids = [chunk.chunk_id for chunk in self._chunks]
        for i, content in enumerate(contents):
            es_request = {
                "_op_type": "index",
                "_index": self._index_name,
                "content": content,
                "metadata": metadatas[i],
                "_id": chunk_ids[i],
            }
            ids.append(chunk_ids[i])
            es_requests.append(es_request)
        bulk(self._es_client, es_requests)
        self._es_client.indices.refresh(index=self._index_name)
        return ids
    # 异步方法，将数据块持久化到 Elasticsearch 中。
    async def apersist(self, **kwargs) -> List[str]:
        """Persist chunks into elasticsearch.

        Returns:
            List[str]: List of chunk ids.
        """
        # 调用异步函数，将阻塞式函数转换为异步执行，使用执行器执行持久化操作
        return await blocking_func_to_async(self._executor, self.persist)

    # 提取数据块的信息，返回空列表
    def _extract_info(self, chunks) -> List[Chunk]:
        """Extract info from chunks."""
        return []

    # 将当前对象配置为 BM25Retriever 的检索器
    def as_retriever(self, top_k: int = 4, **kwargs) -> BM25Retriever:
        """Create a BM25Retriever.

        Args:
            top_k(int): default 4.

        Returns:
            BM25Retriever
        """
        # 创建一个 BM25Retriever 实例，使用给定的 top_k、索引名称和 Elasticsearch 客户端
        return BM25Retriever(
            top_k=top_k, es_index=self._index_name, es_client=self._es_client
        )
```