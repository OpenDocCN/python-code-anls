# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_duckdb.py`

```py
"""DuckDB connector."""
# 导入必要的模块和类
from typing import Any, Iterable, Optional

from sqlalchemy import create_engine, text

# 导入基类 RDBMSConnector
from .base import RDBMSConnector


class DuckDbConnector(RDBMSConnector):
    """DuckDB connector."""

    # 定义 DuckDB 数据库类型和方言
    db_type: str = "duckdb"
    db_dialect: str = "duckdb"

    @classmethod
    def from_file_path(
        cls, file_path: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> RDBMSConnector:
        """Construct a SQLAlchemy engine from URI."""
        # 处理传入的引擎参数
        _engine_args = engine_args or {}
        # 创建并返回 DuckDBConnector 实例，使用传入的文件路径作为连接 URI
        return cls(create_engine("duckdb:///" + file_path, **_engine_args), **kwargs)

    def get_users(self):
        """Get users."""
        # 执行查询以获取用户信息
        cursor = self.session.execute(
            text(
                "SELECT * FROM sqlite_master WHERE type = 'table' AND "
                "name = 'duckdb_sys_users';"
            )
        )
        users = cursor.fetchall()
        # 返回用户列表，每个用户是一个元组
        return [(user[0], user[1]) for user in users]

    def get_grants(self):
        """Get grants."""
        # 返回空列表，因为 DuckDB 不支持授权信息查询
        return []

    def get_collation(self):
        """Get collation."""
        # 返回当前数据库的字符排序规则
        return "UTF-8"

    def get_charset(self):
        """Get character_set of current database."""
        # 返回当前数据库的字符集
        return "UTF-8"

    def get_table_comments(self, db_name: str):
        """Get table comments."""
        # 查询表的名称和创建 SQL 语句
        cursor = self.session.execute(
            text(
                """
                SELECT name, sql FROM sqlite_master WHERE type='table'
                """
            )
        )
        table_comments = cursor.fetchall()
        # 返回表名和创建 SQL 语句组成的列表
        return [
            (table_comment[0], table_comment[1]) for table_comment in table_comments
        ]

    def table_simple_info(self) -> Iterable[str]:
        """Get table simple info."""
        # 查询所有表的名称
        _tables_sql = """
                SELECT name FROM sqlite_master WHERE type='table'
            """
        cursor = self.session.execute(text(_tables_sql))
        tables_results = cursor.fetchall()
        results = []
        # 遍历每个表名并查询其列信息
        for row in tables_results:
            table_name = row[0]
            # 构建查询表结构的 SQL 语句
            _sql = f"""
                PRAGMA  table_info({table_name})
            """
            cursor_colums = self.session.execute(text(_sql))
            colum_results = cursor_colums.fetchall()
            table_colums = []
            # 提取每个列的信息并组成列表
            for row_col in colum_results:
                field_info = list(row_col)
                table_colums.append(field_info[1])

            # 将表名和其列信息组成字符串添加到结果列表
            results.append(f"{table_name}({','.join(table_colums)});")
        # 返回所有表的简要信息列表
        return results
```