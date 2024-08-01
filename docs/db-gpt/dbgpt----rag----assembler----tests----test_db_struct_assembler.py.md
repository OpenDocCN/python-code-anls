# `.\DB-GPT-src\dbgpt\rag\assembler\tests\test_db_struct_assembler.py`

```py
# 导入需要的模块和类
from unittest.mock import MagicMock
import pytest
from dbgpt.datasource.rdbms.conn_sqlite import SQLiteTempConnector
from dbgpt.rag.assembler.embedding import EmbeddingAssembler
from dbgpt.rag.chunk_manager import ChunkParameters, SplitterType
from dbgpt.rag.embedding.embedding_factory import EmbeddingFactory
from dbgpt.rag.knowledge.base import Knowledge
from dbgpt.rag.text_splitter.text_splitter import CharacterTextSplitter
from dbgpt.storage.vector_store.chroma_store import ChromaStore


# 定义一个用于测试的装置：创建临时数据库连接
@pytest.fixture
def mock_db_connection():
    """Create a temporary database connection for testing."""
    # 创建一个临时的 SQLite 数据库连接对象
    connect = SQLiteTempConnector.create_temporary_db()
    # 在连接上创建临时表
    connect.create_temp_tables(
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
    # 返回创建的连接对象
    return connect


# 定义一个用于测试的装置：创建一个 Mock 对象，模拟 ChunkParameters 类的行为
@pytest.fixture
def mock_chunk_parameters():
    # 创建一个 MagicMock 对象，其行为被限定为 ChunkParameters 类的行为
    return MagicMock(spec=ChunkParameters)


# 定义一个用于测试的装置：创建一个 Mock 对象，模拟 EmbeddingFactory 类的行为
@pytest.fixture
def mock_embedding_factory():
    # 创建一个 MagicMock 对象，其行为被限定为 EmbeddingFactory 类的行为
    return MagicMock(spec=EmbeddingFactory)


# 定义一个用于测试的装置：创建一个 Mock 对象，模拟 ChromaStore 类的行为
@pytest.fixture
def mock_vector_store_connector():
    # 创建一个 MagicMock 对象，其行为被限定为 ChromaStore 类的行为
    return MagicMock(spec=ChromaStore)


# 定义一个用于测试的装置：创建一个 Mock 对象，模拟 Knowledge 类的行为
@pytest.fixture
def mock_knowledge():
    # 创建一个 MagicMock 对象，其行为被限定为 Knowledge 类的行为
    return MagicMock(spec=Knowledge)


# 定义一个测试函数：测试知识加载功能
def test_load_knowledge(
    mock_db_connection,
    mock_knowledge,
    mock_chunk_parameters,
    mock_embedding_factory,
    mock_vector_store_connector,
):
    # 设置 Mock 对象的属性
    mock_chunk_parameters.chunk_strategy = "CHUNK_BY_SIZE"
    mock_chunk_parameters.text_splitter = CharacterTextSplitter()
    mock_chunk_parameters.splitter_type = SplitterType.USER_DEFINE
    # 创建 EmbeddingAssembler 对象，传入 Mock 对象作为参数
    assembler = EmbeddingAssembler(
        knowledge=mock_knowledge,
        chunk_parameters=mock_chunk_parameters,
        embeddings=mock_embedding_factory.create(),
        index_store=mock_vector_store_connector,
    )
    # 调用装配器对象的加载知识方法
    assembler.load_knowledge(knowledge=mock_knowledge)
    # 断言装配器对象的分块列表长度为 0
    assert len(assembler._chunks) == 0
```