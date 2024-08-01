# `.\DB-GPT-src\dbgpt\rag\assembler\tests\test_embedding_assembler.py`

```py
from unittest.mock import MagicMock  # 导入 MagicMock 类，用于创建模拟对象

import pytest  # 导入 pytest 测试框架

from dbgpt.datasource.rdbms.conn_sqlite import SQLiteTempConnector  # 导入 SQLiteTempConnector 类
from dbgpt.rag.assembler.db_schema import DBSchemaAssembler  # 导入 DBSchemaAssembler 类
from dbgpt.rag.chunk_manager import ChunkParameters, SplitterType  # 导入 ChunkParameters 和 SplitterType 类
from dbgpt.rag.embedding.embedding_factory import EmbeddingFactory  # 导入 EmbeddingFactory 类
from dbgpt.rag.text_splitter.text_splitter import CharacterTextSplitter  # 导入 CharacterTextSplitter 类
from dbgpt.storage.vector_store.chroma_store import ChromaStore  # 导入 ChromaStore 类


@pytest.fixture
def mock_db_connection():
    """为测试创建临时数据库连接。"""
    connect = SQLiteTempConnector.create_temporary_db()  # 创建临时数据库连接对象
    connect.create_temp_tables(  # 创建临时表
        {
            "user": {
                "columns": {
                    "id": "INTEGER PRIMARY KEY",
                    "name": "TEXT",
                    "age": "INTEGER",
                },
                "data": [
                    (1, "Tom", 10),
                    (2, "Jerry", 16),
                    (3, "Jack", 18),
                    (4, "Alice", 20),
                    (5, "Bob", 22),
                ],
            }
        }
    )
    return connect  # 返回连接对象


@pytest.fixture
def mock_chunk_parameters():
    """创建一个模拟的 ChunkParameters 对象。"""
    return MagicMock(spec=ChunkParameters)  # 返回一个 ChunkParameters 的模拟对象


@pytest.fixture
def mock_embedding_factory():
    """创建一个模拟的 EmbeddingFactory 对象。"""
    return MagicMock(spec=EmbeddingFactory)  # 返回一个 EmbeddingFactory 的模拟对象


@pytest.fixture
def mock_vector_store_connector():
    """创建一个模拟的 ChromaStore 对象。"""
    return MagicMock(spec=ChromaStore)  # 返回一个 ChromaStore 的模拟对象


def test_load_knowledge(
    mock_db_connection,
    mock_chunk_parameters,
    mock_embedding_factory,
    mock_vector_store_connector,
):
    """测试加载知识的功能。"""
    mock_chunk_parameters.chunk_strategy = "CHUNK_BY_SIZE"  # 设置模拟的 ChunkParameters 对象的 chunk_strategy 属性
    mock_chunk_parameters.text_splitter = CharacterTextSplitter()  # 设置模拟的 ChunkParameters 对象的 text_splitter 属性为 CharacterTextSplitter 的实例
    mock_chunk_parameters.splitter_type = SplitterType.USER_DEFINE  # 设置模拟的 ChunkParameters 对象的 splitter_type 属性为 SplitterType.USER_DEFINE
    assembler = DBSchemaAssembler(
        connector=mock_db_connection,  # 使用模拟的数据库连接对象
        chunk_parameters=mock_chunk_parameters,  # 使用模拟的 ChunkParameters 对象
        embeddings=mock_embedding_factory.create(),  # 使用模拟的 EmbeddingFactory 对象创建的嵌入
        index_store=mock_vector_store_connector,  # 使用模拟的 ChromaStore 对象作为索引存储
    )
    assert len(assembler._chunks) == 1  # 断言装配器的 _chunks 属性的长度为 1
```