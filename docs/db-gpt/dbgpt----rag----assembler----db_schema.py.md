# `.\DB-GPT-src\dbgpt\rag\assembler\db_schema.py`

```py
"""DBSchemaAssembler."""
# 导入必要的模块和类
from typing import Any, List, Optional

from dbgpt.core import Chunk  # 导入Chunk类
from dbgpt.datasource.base import BaseConnector  # 导入BaseConnector类

from ..assembler.base import BaseAssembler  # 导入BaseAssembler类
from ..chunk_manager import ChunkParameters  # 导入ChunkParameters类
from ..index.base import IndexStoreBase  # 导入IndexStoreBase类
from ..knowledge.datasource import DatasourceKnowledge  # 导入DatasourceKnowledge类
from ..retriever.db_schema import DBSchemaRetriever  # 导入DBSchemaRetriever类

class DBSchemaAssembler(BaseAssembler):
    """DBSchemaAssembler.

    Example:
        .. code-block:: python

            from dbgpt.datasource.rdbms.conn_sqlite import SQLiteTempConnector
            from dbgpt.serve.rag.assembler.db_struct import DBSchemaAssembler
            from dbgpt.storage.vector_store.connector import VectorStoreConnector
            from dbgpt.storage.vector_store.chroma_store import ChromaVectorConfig

            connection = SQLiteTempConnector.create_temporary_db()
            assembler = DBSchemaAssembler.load_from_connection(
                connector=connection,
                embedding_model=embedding_model_path,
            )
            assembler.persist()
            # get db struct retriever
            retriever = assembler.as_retriever(top_k=3)
    """

    def __init__(
        self,
        connector: BaseConnector,
        index_store: IndexStoreBase,
        chunk_parameters: Optional[ChunkParameters] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with Embedding Assembler arguments.

        Args:
            connector: (BaseConnector) BaseConnector connection.
            index_store: (IndexStoreBase) IndexStoreBase to use.
            chunk_manager: (Optional[ChunkManager]) ChunkManager to use for chunking.
            embedding_model: (Optional[str]) Embedding model to use.
            embeddings: (Optional[Embeddings]) Embeddings to use.
        """
        knowledge = DatasourceKnowledge(connector)  # 使用connector创建DatasourceKnowledge对象
        self._connector = connector  # 初始化连接器
        self._index_store = index_store  # 初始化索引存储对象

        super().__init__(
            knowledge=knowledge,
            chunk_parameters=chunk_parameters,
            **kwargs,
        )

    @classmethod
    def load_from_connection(
        cls,
        connector: BaseConnector,
        index_store: IndexStoreBase,
        chunk_parameters: Optional[ChunkParameters] = None,
    ) -> "DBSchemaAssembler":
        """Load document embedding into vector store from path.

        Args:
            connector: (BaseConnector) BaseConnector connection.
            index_store: (IndexStoreBase) IndexStoreBase to use.
            chunk_parameters: (Optional[ChunkParameters]) ChunkManager to use for
                chunking.
        Returns:
             DBSchemaAssembler
        """
        return cls(
            connector=connector,
            index_store=index_store,
            chunk_parameters=chunk_parameters,
        )

    def get_chunks(self) -> List[Chunk]:
        """Return chunk ids."""
        return self._chunks  # 返回存储的Chunk对象列表
    # 将数据块持久化到向量存储中
    def persist(self, **kwargs: Any) -> List[str]:
        """Persist chunks into vector store.

        Returns:
            List[str]: List of chunk ids.
        """
        # 调用索引存储对象的方法，加载文档数据并返回文档的 chunk id 列表
        return self._index_store.load_document(self._chunks)

    # 从数据块中提取信息的私有方法
    def _extract_info(self, chunks) -> List[Chunk]:
        """Extract info from chunks."""
        # 返回一个空列表，表示从数据块中提取信息的功能尚未实现
        return []

    # 生成 DBSchemaRetriever 实例的方法
    def as_retriever(self, top_k: int = 4, **kwargs) -> DBSchemaRetriever:
        """Create DBSchemaRetriever.

        Args:
            top_k(int): default 4.

        Returns:
            DBSchemaRetriever
        """
        # 返回一个 DBSchemaRetriever 实例，配置包括 top_k 值、连接器对象、使用嵌入表示、索引存储对象
        return DBSchemaRetriever(
            top_k=top_k,
            connector=self._connector,
            is_embeddings=True,
            index_store=self._index_store,
        )
```