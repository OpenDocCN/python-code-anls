# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_hive.py`

```py
"""Hive Connector."""
# 导入所需的模块和类
from typing import Any, Optional, cast
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote

# 导入基类 RDBMSConnector
from .base import RDBMSConnector

# 定义 HiveConnector 类，继承自 RDBMSConnector
class HiveConnector(RDBMSConnector):
    """Hive connector."""

    # 数据库类型为 Hive
    db_type: str = "hive"
    # 数据库驱动为 Hive
    driver: str = "hive"
    # 数据库方言为 Hive
    dialect: str = "hive"

    @classmethod
    def from_uri_db(
        cls,
        host: str,
        port: int,
        user: str,
        pwd: str,
        db_name: str,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> "HiveConnector":
        """Create a new HiveConnector from host, port, user, pwd, db_name."""
        # 构建数据库连接 URL
        db_url: str = f"{cls.driver}://{host}:{str(port)}/{db_name}"
        # 如果提供了用户名和密码，则使用安全的方式构建 URL
        if user and pwd:
            db_url = (
                f"{cls.driver}://{quote(user)}:{urlquote(pwd)}@{host}:{str(port)}/"
                f"{db_name}"
            )
        # 调用父类方法创建 HiveConnector 实例并返回
        return cast(HiveConnector, cls.from_uri(db_url, engine_args, **kwargs))

    def table_simple_info(self):
        """Get table simple info."""
        # 返回空列表，获取表简要信息的方法暂未实现
        return []

    def get_users(self):
        """Get users."""
        # 返回空列表，获取用户信息的方法暂未实现
        return []

    def get_grants(self):
        """Get grants."""
        # 返回空列表，获取授权信息的方法暂未实现
        return []

    def get_collation(self):
        """Get collation."""
        # 返回当前数据库的排序规则为 UTF-8
        return "UTF-8"

    def get_charset(self):
        """Get character_set of current database."""
        # 返回当前数据库的字符集为 UTF-8
        return "UTF-8"
```