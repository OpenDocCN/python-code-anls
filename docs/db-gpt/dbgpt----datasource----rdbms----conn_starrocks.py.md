# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_starrocks.py`

```py
"""StarRocks connector."""
# 导入必要的类型声明和函数
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote

# 导入 SQL 相关功能
from sqlalchemy import text

# 导入基类 RDBMSConnector
from .base import RDBMSConnector

# 导入 StarRocks 的 SQLAlchemy 方言
from .dialect.starrocks.sqlalchemy import *  # noqa


class StarRocksConnector(RDBMSConnector):
    """StarRocks connector."""

    # 定义数据库驱动、类型和方言
    driver = "starrocks"
    db_type = "starrocks"
    db_dialect = "starrocks"

    @classmethod
    def from_uri_db(
        cls: Type["StarRocksConnector"],
        host: str,
        port: int,
        user: str,
        pwd: str,
        db_name: str,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> "StarRocksConnector":
        """Create a new StarRocksConnector from host, port, user, pwd, db_name."""
        # 构建连接字符串
        db_url: str = (
            f"{cls.driver}://{quote(user)}:{urlquote(pwd)}@{host}:{str(port)}/{db_name}"
        )
        # 调用父类方法创建连接对象
        return cast(StarRocksConnector, cls.from_uri(db_url, engine_args, **kwargs))

    def _sync_tables_from_db(self) -> Iterable[str]:
        # 获取当前数据库名称
        db_name = self.get_current_db_name()
        # 查询当前数据库中的所有表名
        table_results = self.session.execute(
            text(
                "SELECT TABLE_NAME FROM information_schema.tables where "
                f'TABLE_SCHEMA="{db_name}"'
            )
        )
        # 将查询结果转换为集合
        table_results = set(row[0] for row in table_results)  # noqa: C401
        # 更新对象的表名集合
        self._all_tables = table_results
        # 反射数据库结构到元数据对象
        self._metadata.reflect(bind=self._engine)
        # 返回所有表名集合
        return self._all_tables

    def get_grants(self):
        """Get grants."""
        # 获取数据库会话对象
        session = self._db_sessions()
        # 执行 SQL 查询获取授权信息
        cursor = session.execute(text("SHOW GRANTS"))
        grants = cursor.fetchall()
        # 如果结果为空，返回空列表
        if len(grants) == 0:
            return []
        # 根据结果格式提取授权信息
        if len(grants[0]) == 2:
            grants_list = [x[1] for x in grants]
        else:
            grants_list = [x[2] for x in grants]
        return grants_list

    def _get_current_version(self):
        """Get database current version."""
        # 查询并返回当前数据库版本号
        return int(self.session.execute(text("select current_version()")).scalar())

    def get_collation(self):
        """Get collation."""
        # StarRocks 的排序是表级别的，这里返回 None
        return None

    def get_users(self):
        """Get user info."""
        # 返回空列表，因为 StarRocks 不提供用户信息的方法
        return []
    def get_fields(self, table_name, db_name="database()") -> List[Tuple]:
        """Get column fields about specified table."""
        # 获取数据库会话对象
        session = self._db_sessions()
        # 如果指定了数据库名，则格式化为数据库引用字符串
        if db_name != "database()":
            db_name = f'"{db_name}"'
        # 执行 SQL 查询，获取表的字段信息：列名、列类型、默认值、是否可为空、列注释
        cursor = session.execute(
            text(
                "select COLUMN_NAME, COLUMN_TYPE, COLUMN_DEFAULT, IS_NULLABLE, "
                "COLUMN_COMMENT from information_schema.columns where "
                f'TABLE_NAME="{table_name}" and TABLE_SCHEMA = {db_name}'
            )
        )
        # 获取所有字段信息并以元组列表形式返回
        fields = cursor.fetchall()
        return [(field[0], field[1], field[2], field[3], field[4]) for field in fields]

    def get_charset(self):
        """Get character_set."""
        # 返回固定的字符集为 utf-8
        return "utf-8"

    def get_show_create_table(self, table_name: str):
        """Get show create table."""
        # 执行 SQL 查询，获取表的创建语句
        cur = self.session.execute(
            text(
                "SELECT TABLE_COMMENT FROM information_schema.tables where "
                f'TABLE_NAME="{table_name}" and TABLE_SCHEMA=database()'
            )
        )
        # 获取表的注释信息
        table = cur.fetchone()
        if table:
            return str(table[0])  # 返回表的注释内容
        else:
            return ""  # 如果没有找到表，则返回空字符串

    def get_table_comments(self, db_name=None):
        """Get table comments."""
        # 如果未指定数据库名，则获取当前数据库名
        if not db_name:
            db_name = self.get_current_db_name()
        # 执行 SQL 查询，获取指定数据库中所有表的注释信息
        cur = self.session.execute(
            text(
                "SELECT TABLE_NAME,TABLE_COMMENT FROM information_schema.tables "
                f'where TABLE_SCHEMA="{db_name}"'
            )
        )
        # 获取所有表的注释信息并以列表形式返回
        tables = cur.fetchall()
        return [(table[0], table[1]) for table in tables]

    def get_database_names(self):
        """Get database names."""
        # 获取数据库会话对象
        session = self._db_sessions()
        # 执行 SQL 查询，获取所有数据库的名称列表
        cursor = session.execute(text("SHOW DATABASES;"))
        results = cursor.fetchall()
        # 返回数据库名列表，过滤掉系统数据库和统计信息数据库
        return [
            d[0]
            for d in results
            if d[0] not in ["information_schema", "sys", "_statistics_", "dataease"]
        ]

    def get_current_db_name(self) -> str:
        """Get current database name."""
        # 执行 SQL 查询，获取当前数据库的名称
        return self.session.execute(text("select database()")).scalar()

    def table_simple_info(self):
        """Get table simple info."""
        # 构建查询语句，获取数据库中所有表的简要信息
        _sql = """
          SELECT concat(TABLE_NAME,"(",group_concat(COLUMN_NAME,","),");")
           FROM information_schema.columns where TABLE_SCHEMA=database()
            GROUP BY TABLE_NAME
            """
        cursor = self.session.execute(text(_sql))
        results = cursor.fetchall()
        # 返回每个表的简要信息的列表
        return [x[0] for x in results]
    # 定义一个方法，用于获取指定表的索引信息
    def get_indexes(self, table_name):
        """Get table indexes about specified table."""
        # 获取数据库会话对象
        session = self._db_sessions()
        # 执行 SQL 查询，获取指定表的索引信息
        cursor = session.execute(text(f"SHOW INDEX FROM {table_name}"))
        # 从游标中获取所有索引信息的元组列表
        indexes = cursor.fetchall()
        # 返回一个列表，其中每个元素是一个元组，包含索引名称和索引列名
        return [(index[2], index[4]) for index in indexes]
```