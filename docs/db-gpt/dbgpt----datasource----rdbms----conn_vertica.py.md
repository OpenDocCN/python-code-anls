# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_vertica.py`

```py
"""Vertica connector."""
# 导入日志模块
import logging
# 导入类型提示模块
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
# 导入 URL 编码相关模块
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote

# 导入 SQLAlchemy 相关模块
from sqlalchemy import text
from sqlalchemy.dialects import registry

# 导入自定义的 RDBMS 连接器基类
from .base import RDBMSConnector

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)
# 注册 Vertica 的 SQLAlchemy 方言
registry.register(
    "vertica.vertica_python",
    "dbgpt.datasource.rdbms.dialect.vertica.dialect_vertica_python",
    "VerticaDialect",
)

# 定义 Vertica 连接器类，继承自 RDBMSConnector 基类
class VerticaConnector(RDBMSConnector):
    """Vertica connector."""

    # 驱动名称
    driver = "vertica+vertica_python"
    # 数据库类型
    db_type = "vertica"
    # 数据库方言
    db_dialect = "vertica"

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
    ) -> "VerticaConnector":
        """Create a new VerticaConnector from host, port, user, pwd, db_name."""
        # 构建数据库连接 URL
        db_url: str = (
            f"{cls.driver}://{quote(user)}:{urlquote(pwd)}@{host}:{str(port)}/{db_name}"
        )
        # 调用基类方法创建并返回 VerticaConnector 实例
        return cast(VerticaConnector, cls.from_uri(db_url, engine_args, **kwargs))

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        # 返回用于 Vertica 的 SQL 方言描述字符串
        return "Vertica sql, \
correct postgresql sql is the another option \
if you don't know much about Vertica. \
尤其要注意，表名称前面一定要带上模式名称！! \
Note， the most important requirement is that \
table name should keep its schema name in "

    def _sync_tables_from_db(self) -> Iterable[str]:
        # 查询数据库中的表和视图名称
        table_results = self.session.execute(
            text(
                """
                SELECT table_schema||'.'||table_name
                FROM v_catalog.tables
                WHERE table_schema NOT LIKE 'v\_%'
                UNION
                SELECT table_schema||'.'||table_name
                FROM v_catalog.views
                WHERE table_schema NOT LIKE 'v\_%';
                """
            )
        )
        # 将查询结果存储到对象的成员变量中
        self._all_tables = {row[0] for row in table_results}
        # 使用引擎反射元数据
        self._metadata.reflect(bind=self._engine)
        # 返回所有表和视图的名称集合
        return self._all_tables

    def get_grants(self):
        """Get grants."""
        # 返回空列表，暂未实现
        return []

    def get_collation(self):
        """Get collation."""
        # 返回 None，暂未实现
        return None

    def get_users(self):
        """Get user info."""
        try:
            # 查询 Vertica 数据库中的用户信息
            cursor = self.session.execute(text("SELECT name FROM v_internal.vs_users;"))
            users = cursor.fetchall()
            # 返回查询结果中的用户列表
            return [user[0] for user in users]
        except Exception as e:
            # 记录警告日志并返回空列表
            logger.warning(f"vertica get users error: {str(e)}")
            return []
    # 获取指定表格的字段信息列表
    def get_fields(self, table_name) -> List[Tuple]:
        """Get column fields about specified table."""
        # 创建数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，获取列信息
        cursor = session.execute(
            text(
                f"""
                SELECT column_name, data_type, column_default, is_nullable,
                  nvl(comment, column_name) as column_comment
                FROM v_catalog.columns c
                  LEFT JOIN v_internal.vs_sub_comments s ON c.table_id = s.objectoid
                    AND c.column_name = s.childobject
                WHERE table_schema||'.'||table_name = '{table_name}';
                """
            )
        )
        # 获取所有查询结果
        fields = cursor.fetchall()
        # 返回结果列表，每个元素为一个元组，包含列名、数据类型、默认值、是否可空、注释
        return [(field[0], field[1], field[2], field[3], field[4]) for field in fields]

    # 获取指定表格的列信息列表
    def get_columns(self, table_name: str) -> List[Dict]:
        """Get columns about specified table.

        Args:
            table_name (str): table name

        Returns:
            columns: List[Dict], which contains name: str, type: str,
                default_expression: str, is_in_primary_key: bool, comment: str
                eg:[{'name': 'id', 'type': 'int', 'default_expression': '',
                'is_in_primary_key': True, 'comment': 'id'}, ...]
        """
        # 创建数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，获取列信息
        cursor = session.execute(
            text(
                f"""
                SELECT c.column_name, data_type, column_default
                  , (p.column_name IS NOT NULL) is_in_primary_key
                  , nvl(comment, c.column_name) as column_comment
                FROM v_catalog.columns c
                  LEFT JOIN v_internal.vs_sub_comments s ON c.table_id = s.objectoid
                    AND c.column_name = s.childobject
                  LEFT JOIN v_catalog.primary_keys p ON c.table_schema = p.table_schema
                    AND c.table_name = p.table_name
                    AND c.column_name = p.column_name
                WHERE c.table_schema||'.'||c.table_name = '{table_name}';
                """
            )
        )
        # 获取所有查询结果
        fields = cursor.fetchall()
        # 返回结果列表，每个元素为一个字典，包含列名、数据类型、默认值、是否在主键中、注释
        return [
            {
                "name": field[0],
                "type": field[1],
                "default_expression": field[2],
                "is_in_primary_key": field[3],
                "comment": field[4],
            }
            for field in fields
        ]

    # 获取字符集信息
    def get_charset(self):
        """Get character_set."""
        # 返回固定的字符集为 UTF-8
        return "utf-8"
    def get_show_create_table(self, table_name: str):
        """Return show create table."""
        # 执行查询，获取指定表的列名和数据类型
        cur = self.session.execute(
            text(
                f"""
                SELECT column_name, data_type
                FROM v_catalog.columns
                WHERE table_schema||'.'||table_name = '{table_name}';
                """
            )
        )
        # 获取所有查询结果行
        rows = cur.fetchall()

        # 构造创建表的 SQL 查询语句
        create_table_query = f"CREATE TABLE {table_name} (\n"
        for row in rows:
            create_table_query += f"    {row[0]} {row[1]},\n"
        # 去除最后一个逗号和换行符，并添加右括号，完成创建表语句
        create_table_query = create_table_query.rstrip(",\n") + "\n)"

        # 返回创建表的 SQL 查询语句
        return create_table_query

    def get_table_comments(self, db_name=None):
        """Return table comments."""
        # 执行查询，获取指定数据库中所有表的注释
        cursor = self.session.execute(
            text(
                f"""
                SELECT table_schema||'.'||table_name
                  , nvl(comment, table_name) as column_comment
                FROM v_catalog.tables t
                  LEFT JOIN v_internal.vs_comments c ON t.table_id = c.objectoid
                WHERE table_schema = '{db_name}'
                """
            )
        )
        # 获取所有查询结果行
        table_comments = cursor.fetchall()
        # 返回结果为列表，每个元素为元组，包含表名和注释
        return [
            (table_comment[0], table_comment[1]) for table_comment in table_comments
        ]

    def get_table_comment(self, table_name: str) -> Dict:
        """Get table comments.

        Args:
            table_name (str): table name
        Returns:
            comment: Dict, which contains text: Optional[str], eg:["text": "comment"]
        """
        # 执行查询，获取指定表的注释
        cursor = self.session.execute(
            text(
                f"""
                SELECT nvl(comment, table_name) as column_comment
                FROM v_catalog.tables t
                  LEFT JOIN v_internal.vs_comments c ON t.table_id = c.objectoid
                WHERE table_schema||'.'||table_name = '{table_name}'
                """
            )
        )
        # 返回结果为字典，包含表注释
        return {"text": cursor.scalar()}

    def get_column_comments(self, db_name: str, table_name: str):
        """Return column comments."""
        # 执行查询，获取指定数据库和表中所有列的注释
        cursor = self.session.execute(
            text(
                f"""
                SELECT column_name, nvl(comment, column_name) as column_comment
                FROM v_catalog.columns c
                  LEFT JOIN v_internal.vs_sub_comments s ON c.table_id = s.objectoid
                    AND c.column_name = s.childobject
                WHERE table_schema = '{db_name}' AND table_name = '{table_name}'
                """
            )
        )
        # 获取所有查询结果行
        column_comments = cursor.fetchall()
        # 返回结果为列表，每个元素为元组，包含列名和注释
        return [
            (column_comment[0], column_comment[1]) for column_comment in column_comments
        ]
    # 获取数据库名称列表的方法
    def get_database_names(self):
        """Get database names."""
        # 获取数据库会话对象
        session = self._db_sessions()
        # 执行 SQL 查询，获取所有数据库模式名称
        cursor = session.execute(text("SELECT schema_name FROM v_catalog.schemata;"))
        # 从游标中获取所有结果
        results = cursor.fetchall()
        # 返回结果列表，过滤掉以 'v_' 开头的数据库模式名称
        return [d[0] for d in results if not d[0].startswith("v_")]

    # 获取当前数据库名称的方法
    def get_current_db_name(self) -> str:
        """Get current database name."""
        # 执行 SQL 查询，获取当前数据库模式名称
        return self.session.execute(text("SELECT current_schema()")).scalar()

    # 获取表格简单信息的方法
    def table_simple_info(self):
        """Get table simple info."""
        # 定义 SQL 查询语句
        _sql = """
            SELECT table_schema||'.'||table_name
              , listagg(column_name using parameters max_length=65000)
            FROM v_catalog.columns
            WHERE table_schema NOT LIKE 'v\_%'
            GROUP BY 1;
            """
        # 执行 SQL 查询，获取表格简单信息
        cursor = self.session.execute(text(_sql))
        # 从游标中获取所有结果
        results = cursor.fetchall()
        # 返回结果列表
        return results

    # 获取指定表格索引信息的方法
    def get_indexes(self, table_name):
        """Get table indexes about specified table."""
        # 返回空列表，因为当前方法尚未实现获取索引的逻辑
        return []
```