# `.\DB-GPT-src\tests\intetration_tests\datasource\test_conn_starrocks.py`

```py
"""
运行单元测试，使用命令 pytest dbgpt/datasource/rdbms/tests/test_conn_starrocks.py

启动 StarRocks Docker 容器：
docker run -p 9030:9030 -p 8030:8030 -p 8040:8040 -itd --name quickstart starrocks/allin1-ubuntu

使用 MySQL 客户端连接到 StarRocks 数据库：
mysql -P 9030 -h 127.0.0.1 -u root --prompt="StarRocks > "
"""

# 导入 pytest 测试框架
import pytest

# 导入 StarRocks 连接器类
from dbgpt.datasource.rdbms.conn_starrocks import StarRocksConnector

# 定义 pytest fixture，用于创建数据库连接
@pytest.fixture
def db():
    # 使用 StarRocks 连接器创建数据库连接对象
    conn = StarRocksConnector.from_uri_db("localhost", 9030, "root", "", "test")
    yield conn  # 返回数据库连接对象

# 测试函数：测试获取表名列表功能
def test_get_table_names(db):
    assert list(db.get_table_names()) == []

# 测试函数：测试获取空表信息功能
def test_get_table_info(db):
    assert db.get_table_info() == ""

# 测试函数：测试获取指定表信息功能
def test_get_table_info_with_table(db):
    # 在数据库中创建表 test
    db.run("create table test(id int)")
    print(db._sync_tables_from_db())  # 打印同步数据库表信息的结果
    table_info = db.get_table_info()  # 获取表信息
    assert "CREATE TABLE test" in table_info  # 断言表创建语句在表信息中存在

# 测试函数：测试执行错误 SQL 语句不抛出异常功能
def test_run_no_throw(db):
    assert db.run_no_throw("this is a error sql").startswith("Error:")

# 测试函数：测试获取空表索引功能
def test_get_index_empty(db):
    # 在数据库中创建表 test（如果不存在）
    db.run("create table if not exists test(id int)")
    assert db.get_indexes("test") == []
```