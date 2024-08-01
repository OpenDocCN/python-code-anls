# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_postgresql.py`

```py
"""PostgreSQL connector."""
# 引入日志记录模块
import logging
# 引入类型提示相关模块
from typing import Any, Iterable, List, Optional, Tuple, cast
# 引入 URL 编码相关模块
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote

# 引入 SQLAlchemy 的文本处理模块
from sqlalchemy import text

# 引入基础数据库连接器
from .base import RDBMSConnector

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class PostgreSQLConnector(RDBMSConnector):
    """PostgreSQL connector."""

    # 定义 PostgreSQL 驱动和数据库类型
    driver = "postgresql+psycopg2"
    db_type = "postgresql"
    db_dialect = "postgresql"

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
    ) -> "PostgreSQLConnector":
        """Create a new PostgreSQLConnector from host, port, user, pwd, db_name."""
        # 构建数据库连接 URL
        db_url: str = (
            f"{cls.driver}://{quote(user)}:{urlquote(pwd)}@{host}:{str(port)}/{db_name}"
        )
        # 调用父类方法创建 PostgreSQLConnector 实例并返回
        return cast(PostgreSQLConnector, cls.from_uri(db_url, engine_args, **kwargs))

    def _sync_tables_from_db(self) -> Iterable[str]:
        # 查询数据库中的表信息（排除系统表和信息架构表）
        table_results = self.session.execute(
            text(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE "
                "schemaname != 'pg_catalog' AND schemaname != 'information_schema'"
            )
        )
        # 查询数据库中的视图信息（排除系统视图和信息架构视图）
        view_results = self.session.execute(
            text(
                "SELECT viewname FROM pg_catalog.pg_views WHERE "
                "schemaname != 'pg_catalog' AND schemaname != 'information_schema'"
            )
        )
        # 提取查询结果中的表名和视图名，形成集合
        table_results = set(row[0] for row in table_results)  # noqa: C401
        view_results = set(row[0] for row in view_results)  # noqa: C401
        # 合并表名和视图名集合，更新对象的所有表信息
        self._all_tables = table_results.union(view_results)
        # 利用引擎对象反射更新元数据
        self._metadata.reflect(bind=self._engine)
        # 返回所有表名的可迭代对象
        return self._all_tables

    def get_grants(self):
        """Get grants."""
        # 获取数据库会话对象
        session = self._db_sessions()
        # 执行 SQL 查询，获取授予当前用户的权限信息
        cursor = session.execute(
            text(
                """
                SELECT DISTINCT grantee, privilege_type
                FROM information_schema.role_table_grants
                WHERE grantee = CURRENT_USER;"""
            )
        )
        # 获取查询结果
        grants = cursor.fetchall()
        # 返回权限信息列表
        return grants

    def get_collation(self):
        """Get collation."""
        try:
            # 尝试获取数据库会话对象
            session = self._db_sessions()
            # 执行 SQL 查询，获取当前数据库的排序规则信息
            cursor = session.execute(
                text(
                    "SELECT datcollate AS collation FROM pg_database WHERE "
                    "datname = current_database();"
                )
            )
            # 获取查询结果中的排序规则信息
            collation = cursor.fetchone()[0]
            # 返回排序规则信息
            return collation
        except Exception as e:
            # 捕获异常并记录警告日志
            logger.warning(f"postgresql get collation error: {str(e)}")
            # 返回空值
            return None
    def get_users(self):
        """Get user info."""
        try:
            # 执行 SQL 查询，获取所有非系统用户的角色名
            cursor = self.session.execute(
                text("SELECT rolname FROM pg_roles WHERE rolname NOT LIKE 'pg_%';")
            )
            users = cursor.fetchall()
            # 提取角色名并返回列表
            return [user[0] for user in users]
        except Exception as e:
            # 如果出现异常，记录警告日志并返回空列表
            logger.warning(f"postgresql get users error: {str(e)}")
            return []

    def get_fields(self, table_name) -> List[Tuple]:
        """Get column fields about specified table."""
        session = self._db_sessions()
        # 执行 SQL 查询，获取指定表的列信息
        cursor = session.execute(
            text(
                "SELECT column_name, data_type, column_default, is_nullable, "
                "column_name as column_comment \
                FROM information_schema.columns WHERE table_name = :table_name",
            ),
            {"table_name": table_name},
        )
        fields = cursor.fetchall()
        # 返回列信息的列表
        return [(field[0], field[1], field[2], field[3], field[4]) for field in fields]

    def get_charset(self):
        """Get character_set."""
        session = self._db_sessions()
        # 执行 SQL 查询，获取当前数据库的字符集编码
        cursor = session.execute(
            text(
                "SELECT pg_encoding_to_char(encoding) FROM pg_database WHERE "
                "datname = current_database();"
            )
        )
        character_set = cursor.fetchone()[0]
        # 返回字符集编码
        return character_set

    def get_show_create_table(self, table_name: str):
        """Return show create table."""
        # 执行 SQL 查询，获取指定表的创建语句
        cur = self.session.execute(
            text(
                f"""
            SELECT a.attname as column_name,
             pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type
            FROM pg_catalog.pg_attribute a
            WHERE a.attnum > 0 AND NOT a.attisdropped AND a.attnum <= (
                SELECT max(a.attnum)
                FROM pg_catalog.pg_attribute a
                WHERE a.attrelid = (SELECT oid FROM pg_catalog.pg_class
                    WHERE relname='{table_name}')
            ) AND a.attrelid = (SELECT oid FROM pg_catalog.pg_class
                 WHERE relname='{table_name}')
                """
            )
        )
        rows = cur.fetchall()

        create_table_query = f"CREATE TABLE {table_name} (\n"
        # 构建创建表的 SQL 查询语句
        for row in rows:
            create_table_query += f"    {row[0]} {row[1]},\n"
        create_table_query = create_table_query.rstrip(",\n") + "\n)"

        return create_table_query

    def get_table_comments(self, db_name=None):
        """Get table comments."""
        # 获取数据库中所有表的简单信息
        tablses = self.table_simple_info()
        comments = []
        # 遍历每张表，获取其创建表语句并存入列表
        for table in tablses:
            table_name = table[0]
            table_comment = self.get_show_create_table(table_name)
            comments.append((table_name, table_comment))
        # 返回包含表名和创建语句的列表
        return comments
    def get_database_names(self):
        """获取数据库名称列表。"""
        # 获取数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，选择所有数据库名，排除系统数据库
        cursor = session.execute(text("SELECT datname FROM pg_database;"))
        # 获取查询结果集合
        results = cursor.fetchall()
        # 返回除去系统数据库（template0, template1, postgres）的数据库名列表
        return [
            d[0] for d in results if d[0] not in ["template0", "template1", "postgres"]
        ]

    def get_current_db_name(self) -> str:
        """获取当前数据库名称。"""
        # 执行 SQL 查询，获取当前数据库名称
        return self.session.execute(text("SELECT current_database()")).scalar()

    def table_simple_info(self):
        """获取表的简要信息。"""
        # 定义 SQL 查询语句，获取表名及其字段信息
        _sql = """
            SELECT table_name, string_agg(column_name, ', ') AS schema_info
            FROM (
                SELECT c.relname AS table_name, a.attname AS column_name
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
                WHERE c.relkind = 'r'
                AND a.attnum > 0
                AND NOT a.attisdropped
                AND n.nspname NOT LIKE 'pg_%'
                AND n.nspname != 'information_schema'
                ORDER BY c.relname, a.attnum
            ) sub
            GROUP BY table_name;
            """
        # 执行 SQL 查询，获取结果集
        cursor = self.session.execute(text(_sql))
        # 获取查询结果
        results = cursor.fetchall()
        # 返回表名及其字段信息的列表
        return results

    def get_fields_wit_schema(self, table_name, schema_name="public"):
        """获取指定表的列字段信息。"""
        # 获取数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，获取指定表在指定模式下的列字段信息
        cursor = session.execute(
            text(
                f"""
                SELECT c.column_name, c.data_type, c.column_default, c.is_nullable,
                 d.description FROM information_schema.columns c
                 LEFT JOIN pg_catalog.pg_description d
                ON (c.table_schema || '.' || c.table_name)::regclass::oid = d.objoid
                 AND c.ordinal_position = d.objsubid
                 WHERE c.table_name='{table_name}' AND c.table_schema='{schema_name}'
                """
            )
        )
        # 获取查询结果集
        fields = cursor.fetchall()
        # 返回包含字段信息的列表
        return [(field[0], field[1], field[2], field[3], field[4]) for field in fields]

    def get_indexes(self, table_name):
        """获取指定表的索引信息。"""
        # 获取数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，获取指定表的索引信息
        cursor = session.execute(
            text(
                f"SELECT indexname, indexdef FROM pg_indexes WHERE "
                f"tablename = '{table_name}'"
            )
        )
        # 获取查询结果集
        indexes = cursor.fetchall()
        # 返回包含索引信息的列表
        return [(index[0], index[1]) for index in indexes]
```