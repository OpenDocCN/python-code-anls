# `.\DB-GPT-src\tests\intetration_tests\datasource\test_conn_tugraph.py`

```py
# 导入 pytest 模块，用于编写和运行测试
# 导入 TuGraphConnector 类，用于连接图数据库
import pytest
from dbgpt.datasource.conn_tugraph import TuGraphConnector

# 设置数据库连接参数
HOST = "localhost"
PORT = 7687
USER = "admin"
PWD = "73@TuGraph"
DB_NAME = "default"

# 定义一个 session 级别的 fixture，用于整个测试会话中创建 TuGraphConnector 对象
@pytest.fixture(scope="session")
def connector():
    """Create a TuGraphConnector for the entire test session."""
    # 使用给定的连接参数创建 TuGraphConnector 对象
    connector = TuGraphConnector.from_uri_db(HOST, PORT, USER, PWD, DB_NAME)
    yield connector  # 将 connector 对象提供给测试函数使用
    # 在所有测试完成后关闭连接
    connector.close()


def test_get_table_names(connector):
    """Test retrieving table names from the graph database."""
    # 调用 TuGraphConnector 的 get_table_names 方法获取表名信息
    table_names = connector.get_table_names()
    # 验证顶点表和边表的数量是否符合预期
    assert len(table_names["vertex_tables"]) == 5
    assert len(table_names["edge_tables"]) == 8


def test_get_columns(connector):
    """Test retrieving columns for a specific table."""
    # 获取名为 'person' 的顶点表的列信息
    columns = connector.get_columns("person", "vertex")
    assert len(columns) == 4
    # 验证列信息中是否包含名为 'id' 的列
    assert any(col["name"] == "id" for col in columns)


def test_get_indexes(connector):
    """Test retrieving indexes for a specific table."""
    # 获取名为 'person' 的顶点表的索引信息
    indexes = connector.get_indexes("person", "vertex")
    assert len(indexes) > 0
```