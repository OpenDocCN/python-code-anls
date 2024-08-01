# `.\DB-GPT-src\tests\intetration_tests\datasource\test_conn_mysql.py`

```py
"""
    Run unit test with command: pytest dbgpt/datasource/rdbms/tests/test_conn_mysql.py
    docker run -itd --name mysql-test -p 3307:3306 -e MYSQL_ROOT_PASSWORD=12345678 mysql:5.7
    mysql -h 127.0.0.1 -uroot -p -P3307
    Enter password:
    Welcome to the MySQL monitor.  Commands end with ; or \g.
    Your MySQL connection id is 2
    Server version: 5.7.41 MySQL Community Server (GPL)

    Copyright (c) 2000, 2023, Oracle and/or its affiliates.

    Oracle is a registered trademark of Oracle Corporation and/or its
    affiliates. Other names may be trademarks of their respective
    owners.

    Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.
    
    > create database test;
"""

import pytest

from dbgpt.datasource.rdbms.conn_mysql import MySQLConnector

# SQL语句：创建名为test的表，如果不存在则创建，使用InnoDB引擎，字符集为utf8mb4
_create_table_sql = """
            CREATE TABLE IF NOT EXISTS `test` (
                `id` int(11) DEFAULT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """


@pytest.fixture
def db():
    # 使用MySQL连接器从指定URI连接到数据库
    conn = MySQLConnector.from_uri_db(
        "localhost",
        3307,
        "root",
        "********",  # 这里应该是实际的数据库密码
        "test",
        engine_args={"connect_args": {"charset": "utf8mb4"}},
    )
    yield conn


def test_get_usable_table_names(db):
    # 运行SQL语句创建表
    db.run(_create_table_sql)
    # 打印数据库同步的表信息
    print(db._sync_tables_from_db())
    # 断言获取的可用表名列表为空
    assert list(db.get_usable_table_names()) == []


def test_get_table_info(db):
    # 断言包含指定表的创建语句
    assert "CREATE TABLE test" in db.get_table_info()


def test_get_table_info_with_table(db):
    # 运行SQL语句创建表
    db.run(_create_table_sql)
    # 打印数据库同步的表信息
    print(db._sync_tables_from_db())
    # 获取表信息并断言包含指定表的创建语句
    table_info = db.get_table_info()
    assert "CREATE TABLE test" in table_info


def test_run_no_throw(db):
    # 断言运行错误SQL语句不会抛出异常
    assert db.run_no_throw("this is a error sql").startswith("Error:")


def test_get_index_empty(db):
    # 运行SQL语句创建表
    db.run(_create_table_sql)
    # 断言获取指定表的索引为空列表
    assert db.get_indexes("test") == []


def test_get_fields(db):
    # 运行SQL语句创建表
    db.run(_create_table_sql)
    # 断言获取指定表的字段列表，第一个字段为"id"
    assert list(db.get_fields("test")[0])[0] == "id"


def test_get_charset(db):
    # 断言获取数据库的字符集为utf8mb4或latin1
    assert db.get_charset() == "utf8mb4" or db.get_charset() == "latin1"


def test_get_collation(db):
    # 断言获取数据库的排序规则为utf8mb4_general_ci或latin1_swedish_ci
    assert (
        db.get_collation() == "utf8mb4_general_ci"
        or db.get_collation() == "latin1_swedish_ci"
    )


def test_get_users(db):
    # 断言数据库中存在用户名为"root"，允许从任何主机连接
    assert ("root", "%") in db.get_users()


def test_get_database_lists(db):
    # 断言获取的数据库名列表包含单个元素"test"
    assert db.get_database_names() == ["test"]
```