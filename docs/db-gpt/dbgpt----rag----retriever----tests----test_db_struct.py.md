# `.\DB-GPT-src\dbgpt\rag\retriever\tests\test_db_struct.py`

```py
# 从 typing 模块导入 List 类型
from typing import List
# 从 unittest.mock 模块导入 MagicMock 和 patch
from unittest.mock import MagicMock, patch

# 导入 pytest 测试框架
import pytest

# 导入 dbgpt 应用程序
import dbgpt
# 从 dbgpt.core 模块导入 Chunk 类
from dbgpt.core import Chunk
# 从 dbgpt.rag.retriever.db_schema 模块导入 DBSchemaRetriever 类
from dbgpt.rag.retriever.db_schema import DBSchemaRetriever
# 从 dbgpt.rag.summary.rdbms_db_summary 模块导入 _parse_db_summary 函数
from dbgpt.rag.summary.rdbms_db_summary import _parse_db_summary


# 定义 pytest fixture，返回一个 MagicMock 对象用于模拟数据库连接
@pytest.fixture
def mock_db_connection():
    return MagicMock()


# 定义 pytest fixture，返回一个 MagicMock 对象用于模拟向量存储连接器
@pytest.fixture
def mock_vector_store_connector():
    # 创建 MagicMock 对象并模拟 similar_search 方法的返回值
    mock_connector = MagicMock()
    mock_connector.similar_search.return_value = [Chunk(content="Table summary")] * 4
    return mock_connector


# 定义 pytest fixture，返回一个 DBSchemaRetriever 实例，使用模拟的数据库连接和向量存储连接器
@pytest.fixture
def db_struct_retriever(mock_db_connection, mock_vector_store_connector):
    return DBSchemaRetriever(
        connector=mock_db_connection,
        index_store=mock_vector_store_connector,
    )


# 定义用于模拟 _parse_db_summary 方法的函数，返回一个字符串列表
def mock_parse_db_summary(conn) -> List[str]:
    """Patch _parse_db_summary method."""
    return ["Table summary"]


# 使用 patch.object 装饰器，将 dbgpt.rag.summary.rdbms_db_summary._parse_db_summary 方法替换为 mock_parse_db_summary 函数
@patch.object(
    dbgpt.rag.summary.rdbms_db_summary, "_parse_db_summary", mock_parse_db_summary
)
# 定义测试函数 test_retrieve_with_mocked_summary，用于测试带有模拟摘要的数据检索功能
def test_retrieve_with_mocked_summary(db_struct_retriever):
    query = "Table summary"
    # 调用 DBSchemaRetriever 实例的 _retrieve 方法，获取查询结果
    chunks: List[Chunk] = db_struct_retriever._retrieve(query)
    # 断言第一个返回的 chunk 是 Chunk 类型的实例
    assert isinstance(chunks[0], Chunk)
    # 断言第一个返回的 chunk 的内容与预期的相符
    assert chunks[0].content == "Table summary"


# 使用 patch.object 装饰器，将 dbgpt.rag.summary.rdbms_db_summary._parse_db_summary 方法替换为 mock_parse_db_summary 函数
@patch.object(
    dbgpt.rag.summary.rdbms_db_summary, "_parse_db_summary", mock_parse_db_summary
)
# 定义异步测试函数 test_aretrieve_with_mocked_summary，用于测试带有模拟摘要的异步数据检索功能
async def test_aretrieve_with_mocked_summary(db_struct_retriever):
    query = "Table summary"
    # 调用 DBSchemaRetriever 实例的 _aretrieve 方法，获取异步查询结果
    chunks: List[Chunk] = await db_struct_retriever._aretrieve(query)
    # 断言第一个返回的 chunk 是 Chunk 类型的实例
    assert isinstance(chunks[0], Chunk)
    # 断言第一个返回的 chunk 的内容与预期的相符
    assert chunks[0].content == "Table summary"
```