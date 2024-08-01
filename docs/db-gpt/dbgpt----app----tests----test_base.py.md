# `.\DB-GPT-src\dbgpt\app\tests\test_base.py`

```py
# 导入所需的模块和类
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# 导入待测试的函数
from dbgpt.app.base import _create_mysql_database


# 使用 patch 修饰器，模拟 create_engine 和 logger
@patch("sqlalchemy.create_engine")
@patch("dbgpt.app.base.logger")
def test_database_already_exists(mock_logger, mock_create_engine):
    # 创建一个模拟的数据库连接对象
    mock_connection = MagicMock()
    # 设置 mock_create_engine 返回的模拟对象及其方法
    mock_create_engine.return_value.connect.return_value.__enter__.return_value = (
        mock_connection
    )

    # 调用 _create_mysql_database 函数测试已存在的数据库情况
    _create_mysql_database(
        "test_db", "mysql+pymysql://user:password@host/test_db", True
    )
    # 验证 logger 是否按预期调用了 info 方法
    mock_logger.info.assert_called_with("Database test_db already exists")
    # 验证数据库连接对象的 execute 方法未被调用
    mock_connection.execute.assert_not_called()


# 使用 patch 修饰器，模拟 create_engine 和 logger
@patch("sqlalchemy.create_engine")
@patch("dbgpt.app.base.logger")
def test_database_creation_success(mock_logger, mock_create_engine):
    # 模拟第一次连接失败，第二次连接成功的情况
    mock_create_engine.side_effect = [
        MagicMock(
            connect=MagicMock(
                side_effect=OperationalError("Unknown database", None, None)
            )
        ),
        MagicMock(),  # 第二次连接成功的模拟对象
    ]

    # 调用 _create_mysql_database 函数测试数据库创建成功的情况
    _create_mysql_database(
        "test_db", "mysql+pymysql://user:password@host/test_db", True
    )
    # 验证 logger 是否按预期调用了 info 方法
    mock_logger.info.assert_called_with("Database test_db successfully created")


# 使用 patch 修饰器，模拟 create_engine 和 logger
@patch("sqlalchemy.create_engine")
@patch("dbgpt.app.base.logger")
def test_database_creation_failure(mock_logger, mock_create_engine):
    # 模拟第一次连接失败，第二次连接也失败的情况（抛出 SQLAlchemyError）
    mock_create_engine.side_effect = [
        MagicMock(
            connect=MagicMock(
                side_effect=OperationalError("Unknown database", None, None)
            )
        ),
        MagicMock(connect=MagicMock(side_effect=SQLAlchemyError("Creation failed"))),
    ]

    # 使用 pytest.raises 检查是否抛出了 SQLAlchemyError 异常
    with pytest.raises(SQLAlchemyError):
        _create_mysql_database(
            "test_db", "mysql+pymysql://user:password@host/test_db", True
        )
    # 验证 logger 是否按预期调用了 error 方法
    mock_logger.error.assert_called_with(
        "Failed to create database test_db: Creation failed"
    )


# 使用 patch 修饰器，模拟 create_engine 和 logger
@patch("sqlalchemy.create_engine")
@patch("dbgpt.app.base.logger")
def test_skip_database_creation(mock_logger, mock_create_engine):
    # 调用 _create_mysql_database 函数测试跳过数据库创建的情况
    _create_mysql_database(
        "test_db", "mysql+pymysql://user:password@host/test_db", False
    )
    # 验证 logger 是否按预期调用了 info 方法
    mock_logger.info.assert_called_with("Skipping creation of database test_db")
    # 验证 create_engine 方法未被调用
    mock_create_engine.assert_not_called()
```