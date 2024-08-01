# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_doris.py`

```py
"""Doris connector."""
# 导入所需的模块和类
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote

# 导入 SQLAlchemy 中的 text 类
from sqlalchemy import text

# 导入基础数据库连接类 RDBMSConnector
from .base import RDBMSConnector


class DorisConnector(RDBMSConnector):
    """Doris connector."""

    # 定义类属性 driver、db_type 和 db_dialect
    driver = "doris"
    db_type = "doris"
    db_dialect = "doris"

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
    ) -> "DorisConnector":
        """Create a new DorisConnector from host, port, user, pwd, db_name."""
        # 构建数据库连接 URI 字符串
        db_url: str = (
            f"{cls.driver}://{quote(user)}:{urlquote(pwd)}@{host}:{str(port)}/{db_name}"
        )
        # 调用父类方法创建并返回 DorisConnector 对象
        return cast(DorisConnector, cls.from_uri(db_url, engine_args, **kwargs))

    def _sync_tables_from_db(self) -> Iterable[str]:
        """Sync tables from the database and return an iterable of table names."""
        # 查询数据库中所有表的表名
        table_results = self.get_session().execute(
            text(
                "SELECT TABLE_NAME FROM information_schema.tables where "
                "TABLE_SCHEMA=database()"
            )
        )
        # 将查询结果转换为集合，存储在对象属性 _all_tables 中
        table_results = set(row[0] for row in table_results)  # noqa: C401
        self._all_tables = table_results
        # 使用 _metadata 对象反射数据库结构到 _engine
        self._metadata.reflect(bind=self._engine)
        # 返回所有表名的集合
        return self._all_tables

    def get_grants(self):
        """Get grants from the database."""
        # 查询数据库的授权信息
        cursor = self.get_session().execute(text("SHOW GRANTS"))
        grants = cursor.fetchall()
        # 根据查询结果构建授权信息列表
        if len(grants) == 0:
            return []
        if len(grants[0]) == 2:
            grants_list = [x[1] for x in grants]
        else:
            grants_list = [x[2] for x in grants]
        return grants_list

    def _get_current_version(self):
        """Get the current version of the database."""
        # 查询数据库当前版本号并转换为整数返回
        return int(
            self.get_session().execute(text("select current_version()")).scalar()
        )

    def get_collation(self):
        """Get the collation settings of the database."""
        # 查询数据库的字符校对规则
        cursor = self.get_session().execute(text("SHOW COLLATION"))
        results = cursor.fetchall()
        # 返回查询结果的第一个字符校对规则，如果不存在结果则返回空字符串
        return "" if not results else results[0][0]

    def get_users(self):
        """Get information about the users in the database."""
        # 返回空列表，因为 Doris 数据库中暂不支持获取用户信息的操作
        return []
    def get_columns(self, table_name: str) -> List[Dict]:
        """Get columns of a specified table.

        Args:
            table_name (str): Name of the table.
        Returns:
            List[Dict]: A list of dictionaries, each containing information about a column,
                        including 'name', 'type', 'default', 'nullable', and 'comment'.
        """
        # Retrieve the fields (columns) information for the specified table
        fields = self.get_fields(table_name)
        # Construct a list of dictionaries representing columns with their attributes
        return [
            {
                "name": field[0],
                "type": field[1],
                "default": field[2],
                "nullable": field[3],
                "comment": field[4],
            }
            for field in fields
        ]

    def get_fields(self, table_name) -> List[Tuple]:
        """Retrieve column fields for a specified table."""
        # Execute SQL query to fetch column information from database schema
        cursor = self.get_session().execute(
            text(
                "select COLUMN_NAME, COLUMN_TYPE, COLUMN_DEFAULT, IS_NULLABLE, "
                "COLUMN_COMMENT from information_schema.columns "
                f'where TABLE_NAME="{table_name}" and TABLE_SCHEMA=database()'
            )
        )
        # Fetch all rows of the result set
        fields = cursor.fetchall()
        # Return a list of tuples containing column information
        return [(field[0], field[1], field[2], field[3], field[4]) for field in fields]

    def get_charset(self):
        """Get the character set used by the database."""
        return "utf-8"

    def get_show_create_table(self, table_name) -> str:
        """Retrieve the SQL statement used to create the specified table.

        Args:
            table_name (str): Name of the table.
        Returns:
            str: The SQL statement used to create the table.
        """
        # Query to retrieve the table creation SQL statement
        cur = self.get_session().execute(
            text(
                f"SELECT TABLE_COMMENT "
                f"FROM information_schema.tables "
                f'where TABLE_NAME="{table_name}" and TABLE_SCHEMA=database()'
            )
        )
        # Fetch the result (table comment) from the query
        table = cur.fetchone()
        # If table comment exists, return it as a string; otherwise, return an empty string
        if table:
            return str(table[0])
        else:
            return ""

    def get_table_comments(self, db_name=None):
        """Retrieve comments for all tables in the specified database.

        Args:
            db_name (str, optional): Name of the database. Defaults to the current database.
        Returns:
            List[Tuple]: A list of tuples, each containing (table_name, table_comment).
        """
        # Determine the database name to use in the query
        db_name = "database()" if not db_name else f"'{db_name}'"
        # Execute SQL query to fetch table names and their comments
        cursor = self.get_session().execute(
            text(
                f"SELECT TABLE_NAME,TABLE_COMMENT "
                f"FROM information_schema.tables "
                f"where TABLE_SCHEMA={db_name}"
            )
        )
        # Fetch all rows of the result set
        tables = cursor.fetchall()
        # Return a list of tuples containing table names and their comments
        return [(table[0], table[1]) for table in tables]
    def get_database_names(self):
        """获取数据库名称列表。"""
        # 获取数据库会话的游标对象
        cursor = self.get_session().execute(text("SHOW DATABASES"))
        # 获取所有查询结果
        results = cursor.fetchall()
        # 返回过滤掉特定系统数据库的数据库名称列表
        return [
            d[0]
            for d in results
            if d[0]
            not in [
                "information_schema",
                "sys",
                "_statistics_",
                "mysql",
                "__internal_schema",
                "doris_audit_db__",
            ]
        ]

    def get_current_db_name(self) -> str:
        """获取当前数据库名称。"""
        # 执行SQL语句以获取当前数据库名称并返回
        return self.get_session().execute(text("select database()")).scalar()

    def table_simple_info(self):
        """获取数据库中每张表的简要信息。"""
        # 获取数据库会话的游标对象
        cursor = self.get_session().execute(
            text(
                "SELECT concat(TABLE_NAME,'(',group_concat(COLUMN_NAME,','),');') "
                "FROM information_schema.columns "
                "where TABLE_SCHEMA=database() "
                "GROUP BY TABLE_NAME"
            )
        )
        # 获取所有查询结果
        results = cursor.fetchall()
        # 返回每张表的简要信息列表
        return [x[0] for x in results]

    def get_indexes(self, table_name):
        """获取指定表的索引信息。"""
        # 执行SQL语句以获取指定表的索引信息
        cursor = self.get_session().execute(text(f"SHOW INDEX FROM {table_name}"))
        # 获取所有索引信息
        indexes = cursor.fetchall()
        # 返回包含索引名和索引列的元组列表
        return [(index[2], index[4]) for index in indexes]
```