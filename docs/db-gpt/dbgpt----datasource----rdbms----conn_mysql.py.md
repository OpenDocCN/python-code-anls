# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_mysql.py`

```py
"""MySQL connector."""

# 导入基类 RDBMSConnector，该类定义了与关系型数据库的通用连接接口
from .base import RDBMSConnector

# 定义 MySQLConnector 类，继承自 RDBMSConnector 基类，用于连接 MySQL 数据库
class MySQLConnector(RDBMSConnector):
    """MySQL connector."""

    # 定义数据库类型为 MySQL
    db_type: str = "mysql"
    # 定义数据库方言为 MySQL
    db_dialect: str = "mysql"
    # 定义数据库驱动程序为 "mysql+pymysql"
    driver: str = "mysql+pymysql"

    # 定义默认数据库列表，包括系统默认的几个数据库
    default_db = ["information_schema", "performance_schema", "sys", "mysql"]
```