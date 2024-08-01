# `.\DB-GPT-src\dbgpt\rag\operators\db_schema.py`

```py
"""The DBSchema Retriever Operator."""

# 导入必要的模块和类
from typing import List, Optional
from dbgpt.core import Chunk  # 导入核心模块中的Chunk类
from dbgpt.core.interface.operators.retriever import RetrieverOperator  # 导入RetrieverOperator接口类
from dbgpt.datasource.base import BaseConnector  # 导入基础连接器类

from ..assembler.db_schema import DBSchemaAssembler  # 导入DBSchemaAssembler类
from ..chunk_manager import ChunkParameters  # 导入ChunkParameters类
from ..index.base import IndexStoreBase  # 导入IndexStoreBase类
from ..retriever.db_schema import DBSchemaRetriever  # 导入DBSchemaRetriever类
from .assembler import AssemblerOperator  # 导入AssemblerOperator类


class DBSchemaRetrieverOperator(RetrieverOperator[str, List[Chunk]]):
    """The DBSchema Retriever Operator.

    Args:
        connector (BaseConnector): The connection.
        top_k (int, optional): The top k. Defaults to 4.
        index_store (IndexStoreBase, optional): The vector store
        connector. Defaults to None.
    """

    def __init__(
        self,
        index_store: IndexStoreBase,
        top_k: int = 4,
        connector: Optional[BaseConnector] = None,
        **kwargs
    ):
        """Create a new DBSchemaRetrieverOperator."""
        super().__init__(**kwargs)
        # 初始化DBSchemaRetriever对象
        self._retriever = DBSchemaRetriever(
            top_k=top_k,
            connector=connector,
            index_store=index_store,
        )

    def retrieve(self, query: str) -> List[Chunk]:
        """Retrieve the table schemas.

        Args:
            query (str): The query.
        
        Returns:
            List[Chunk]: A list of Chunk objects representing retrieved schemas.
        """
        # 调用DBSchemaRetriever的retrieve方法，返回查询到的表模式信息
        return self._retriever.retrieve(query)


class DBSchemaAssemblerOperator(AssemblerOperator[BaseConnector, List[Chunk]]):
    """The DBSchema Assembler Operator."""

    def __init__(
        self,
        connector: BaseConnector,
        index_store: IndexStoreBase,
        chunk_parameters: Optional[ChunkParameters] = None,
        **kwargs
    ):
        """Create a new DBSchemaAssemblerOperator.

        Args:
            connector (BaseConnector): The connection.
            index_store (IndexStoreBase): The Storage IndexStoreBase.
            chunk_parameters (Optional[ChunkParameters], optional): The chunk
                parameters.
        """
        # 如果未提供chunk_parameters，则使用默认的ChunkParameters配置
        if not chunk_parameters:
            chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_SIZE")
        self._chunk_parameters = chunk_parameters
        self._index_store = index_store
        self._connector = connector
        super().__init__(**kwargs)

    def assemble(self, dummy_value) -> List[Chunk]:
        """Persist the database schema.

        Args:
            dummy_value: Dummy value, not used.

        Returns:
            List[Chunk]: The chunks representing persisted database schema.
        """
        # 从连接器和索引存储中加载DBSchemaAssembler实例
        assembler = DBSchemaAssembler.load_from_connection(
            connector=self._connector,
            chunk_parameters=self._chunk_parameters,
            index_store=self._index_store,
        )
        # 持久化数据库模式信息
        assembler.persist()
        # 返回持久化后的Chunk对象列表
        return assembler.get_chunks()
```