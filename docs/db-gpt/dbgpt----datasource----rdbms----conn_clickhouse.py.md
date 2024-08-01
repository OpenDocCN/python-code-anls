# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_clickhouse.py`

```py
"""Clickhouse connector."""
# 导入日志模块
import logging
# 导入正则表达式模块
import re
# 导入类型提示模块中的特定类型
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 导入 SQL 解析模块
import sqlparse
# 导入 SQLAlchemy 中的元数据和文本对象
from sqlalchemy import MetaData, text

# 导入自定义的数据库类型枚举
from dbgpt.storage.schema import DBType
# 导入基础的关系型数据库连接器
from .base import RDBMSConnector

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class ClickhouseConnector(RDBMSConnector):
    """Clickhouse connector."""

    """db type"""
    # 数据库类型为 Clickhouse
    db_type: str = "clickhouse"
    """db driver"""
    # 数据库驱动程序为 Clickhouse
    driver: str = "clickhouse"
    """db dialect"""
    # 数据库方言为 Clickhouse
    db_dialect: str = "clickhouse"

    # 客户端对象，初始化为 None
    client: Any = None

    def __init__(self, client, **kwargs):
        """Create a new ClickhouseConnector from client."""
        # 初始化方法，接受客户端对象和其他关键字参数
        self.client = client

        # 下面初始化了一系列实例变量，用于存储各种表和信息
        self._all_tables = set()
        self.view_support = False
        self._usable_tables = set()
        self._include_tables = set()
        self._ignore_tables = set()
        self._custom_table_info = set()
        self._indexes_in_table_info = set()
        self._usable_tables = set()  # 重复初始化
        self._usable_tables = set()  # 重复初始化
        self._sample_rows_in_table_info = set()

        # 创建 SQLAlchemy 元数据对象
        self._metadata = MetaData()

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
    ) -> "ClickhouseConnector":
        """Create a new ClickhouseConnector from host, port, user, pwd, db_name."""
        # 延迟导入
        import clickhouse_connect
        from clickhouse_connect.driver import httputil

        # 使用 httputil 获取一个大型连接池管理器
        big_pool_mgr = httputil.get_pool_manager(maxsize=16, num_pools=12)
        # 通过 clickhouse_connect 获取 Clickhouse 客户端对象
        client = clickhouse_connect.get_client(
            host=host,
            user=user,
            password=pwd,
            port=port,
            connect_timeout=15,
            database=db_name,
            settings={"distributed_ddl_task_timeout": 300},
            pool_mgr=big_pool_mgr,
        )

        # 将客户端对象设置为类属性
        cls.client = client
        # 返回新创建的 ClickhouseConnector 实例
        return cls(client, **kwargs)

    def get_table_names(self):
        """Get all table names."""
        # 获取客户端会话对象
        session = self.client

        # 使用查询行块流的方式获取所有表名
        with session.query_row_block_stream("SHOW TABLES") as stream:
            tables = [row[0] for block in stream for row in block]
            return tables
    def get_indexes(self, table_name: str) -> List[Dict]:
        """Get table indexes about specified table.

        Args:
            table_name (str): table name
        Returns:
            indexes: List[Dict], eg:[{'name': 'idx_key', 'column_names': ['id']}]
        """
        # 获取当前会话的客户端
        session = self.client

        # 构建查询 SQL 语句，查询指定表在系统表中的索引信息
        _query_sql = f"""
                    SELECT name AS table, primary_key, from system.tables where
                     database ='{self.client.database}' and table = '{table_name}'
                """
        
        # 使用查询 SQL 语句执行查询，并获取查询结果的流
        with session.query_row_block_stream(_query_sql) as stream:
            # 从流中获取索引信息块
            indexes = [block for block in stream]  # noqa
            # 构建并返回索引信息列表
            return [
                {"name": "primary_key", "column_names": column_names.split(",")}
                for table, column_names in indexes[0]
            ]

    @property
    def table_info(self) -> str:
        """Get table info."""
        # 返回获取表信息的方法结果
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        # TODO: 实现获取指定表信息的功能，当前方法返回空字符串
        return ""

    def get_show_create_table(self, table_name):
        """Get table show create table about specified table."""
        # 使用客户端执行显示创建表的命令，并获取结果
        result = self.client.command(text(f"SHOW CREATE TABLE  {table_name}"))

        # 对结果进行处理，移除引擎和字符集信息，并返回处理后的结果
        ans = result
        ans = re.sub(r"\s*ENGINE\s*=\s*MergeTree\s*", " ", ans, flags=re.IGNORECASE)
        ans = re.sub(
            r"\s*DEFAULT\s*CHARSET\s*=\s*\w+\s*", " ", ans, flags=re.IGNORECASE
        )
        ans = re.sub(r"\s*SETTINGS\s*\s*\w+\s*", " ", ans, flags=re.IGNORECASE)
        return ans

    def get_columns(self, table_name: str) -> List[Dict]:
        """Get columns.

        Args:
            table_name (str): str
        Returns:
            List[Dict], which contains name: str, type: str,
                default_expression: str, is_in_primary_key: bool, comment: str
                eg:[{'name': 'id', 'type': 'UInt64', 'default_expression': '',
                'is_in_primary_key': True, 'comment': 'id'}, ...]
        """
        # 获取指定表的字段信息列表
        fields = self.get_fields(table_name)
        
        # 构建并返回字段信息字典列表
        return [
            {"name": name, "comment": comment, "type": column_type}
            for name, column_type, _, _, comment in fields[0]
        ]

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        # 返回当前对象使用的方言字符串表示，当前方法返回空字符串
        return ""
    def get_fields(self, table_name) -> List[Tuple]:
        """Get column fields about specified table."""
        # 获取当前数据库会话
        session = self.client

        # 构建查询 SQL 语句，用于获取指定表的列信息
        _query_sql = f"""
            SELECT name, type, default_expression, is_in_primary_key, comment
                from system.columns where table='{table_name}'
        """.format(
            table_name
        )
        
        # 使用查询语句执行查询操作，返回字段信息列表
        with session.query_row_block_stream(_query_sql) as stream:
            fields = [block for block in stream]  # noqa
            return fields

    def get_users(self):
        """Get user info."""
        # 返回空列表，暂未实现获取用户信息的功能
        return []

    def get_grants(self):
        """Get grants."""
        # 返回空列表，暂未实现获取授权信息的功能
        return []

    def get_collation(self):
        """Get collation."""
        # 返回默认的字符集校对规则，此处为UTF-8
        return "UTF-8"

    def get_charset(self):
        """Get character_set."""
        # 返回默认的字符集，此处为UTF-8
        return "UTF-8"

    def get_database_names(self):
        """Get database names."""
        # 获取当前数据库会话
        session = self.client

        # 使用命令方式获取数据库列表
        with session.command("SHOW DATABASES") as stream:
            # 解析流数据，获取数据库名称列表，排除系统默认的数据库
            databases = [
                row[0]
                for block in stream
                for row in block
                if row[0]
                not in ("INFORMATION_SCHEMA", "system", "default", "information_schema")
            ]
            return databases

    def run(self, command: str, fetch: str = "all") -> List:
        """Execute sql command."""
        # TODO 需要实现

        # 记录日志，记录执行的 SQL 命令
        logger.info("SQL:" + command)

        # 如果命令为空或长度小于0，则返回空列表
        if not command or len(command) < 0:
            return []

        # 解析 SQL 命令，获取类型、SQL 类型和表名等信息
        _, ttype, sql_type, table_name = self.__sql_parse(command)

        # 如果是数据操作语言（DML）
        if ttype == sqlparse.tokens.DML:
            # 如果是 SELECT 语句，则执行查询操作
            if sql_type == "SELECT":
                return self._query(command, fetch)
            else:
                # 否则执行写操作，并转换成 SELECT 语句执行查询
                self._write(command)
                select_sql = self.convert_sql_write_to_select(command)
                logger.info(f"write result query:{select_sql}")
                return self._query(select_sql)
        else:
            # 否则为数据定义语言（DDL），记录日志
            logger.info(
                "DDL execution determines whether to enable through configuration "
            )

            # 执行命令并获取游标
            cursor = self.client.command(command)

            # 如果有写入行数，则获取结果行和字段名
            if cursor.written_rows:
                result = cursor.result_rows
                field_names = result.column_names

                result = list(result)
                result.insert(0, field_names)
                logger.info("DDL Result:" + str(result))
                if not result:
                    # 如果结果为空，则返回简单字段信息
                    return self.get_simple_fields(table_name)
                return result
            else:
                # 否则返回简单字段信息
                return self.get_simple_fields(table_name)

    def get_simple_fields(self, table_name):
        """Get column fields about specified table."""
        # 返回指定表的简单字段信息查询结果
        return self._query(f"SHOW COLUMNS FROM {table_name}")

    def get_current_db_name(self):
        """Get current database name."""
        # 返回当前数据库的名称
        return self.client.database
    def get_table_comments(self, db_name: str):
        """获取表格注释信息。

        Args:
            db_name (str): 数据库名称

        Returns:
            list: 包含表格名称和注释的列表，每个元素是一个字典 {"table": str, "comment": str}
        """
        session = self.client

        _query_sql = f"""
                SELECT table, comment FROM system.tables WHERE database = '{db_name}'
            """.format(
            db_name
        )

        with session.query_row_block_stream(_query_sql) as stream:
            table_comments = [row for block in stream for row in block]
            return table_comments

    def get_table_comment(self, table_name: str) -> Dict:
        """获取表格的注释信息。

        Args:
            table_name (str): 表格名称

        Returns:
            Dict: 包含表格注释的字典 {"text": str}
        """
        session = self.client

        _query_sql = f"""
                SELECT table, comment FROM system.tables WHERE
                 database = '{self.client.database}'and table = '{table_name}'
                 """.format(
            self.client.database
        )

        with session.query_row_block_stream(_query_sql) as stream:
            table_comments = [row for block in stream for row in block]
            return [{"text": comment} for table_name, comment in table_comments][0]

    def get_column_comments(self, db_name, table_name):
        """获取列的注释信息。

        Args:
            db_name (str): 数据库名称
            table_name (str): 表格名称

        Returns:
            list: 包含列名和注释的列表，每个元素是一个字典 {"column": str, "comment": str}
        """
        session = self.client
        _query_sql = f"""
            select name column, comment from  system.columns where database='{db_name}'
             and table='{table_name}'
        """.format(
            db_name, table_name
        )

        with session.query_row_block_stream(_query_sql) as stream:
            column_comments = [row for block in stream for row in block]
            return column_comments

    def table_simple_info(self):
        """获取表格的简单信息。

        Returns:
            list: 包含表格简要信息的列表，每个元素是一个字符串
        """
        # group_concat() 在 ClickHouse 中不支持，使用 arrayStringConcat + groupArray 代替，并且需要转义引号
        _sql = f"""
            SELECT concat(TABLE_NAME, '(', arrayStringConcat(
                groupArray(column_name), '-'), ')') AS schema_info
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = '{self.get_current_db_name()}'
            GROUP BY TABLE_NAME
        """
        with self.client.query_row_block_stream(_sql) as stream:
            return [row[0] for block in stream for row in block]

    def _write(self, write_sql: str):
        """执行写入 SQL。

        Args:
            write_sql (str): SQL 字符串
        """
        # TODO 需要实现
        logger.info(f"Write[{write_sql}]")
        result = self.client.command(write_sql)
        logger.info(f"SQL[{write_sql}], result:{result.written_rows}")
    def _query(self, query: str, fetch: str = "all"):
        """Query data from clickhouse.

        Args:
            query (str): sql string - SQL查询语句
            fetch (str, optional): "one" or "all". Defaults to "all". - 指定查询结果返回一条数据还是全部数据

        Raises:
            ValueError: Error - 当fetch参数既不是'one'也不是'all'时引发数值错误

        Returns:
            _type_: List<Result> - 返回一个包含查询结果的列表
        """
        # TODO need to be implemented
        logger.info(f"Query[{query}]")  # 记录查询日志

        if not query:
            return []

        cursor = self.client.query(query)  # 使用客户端执行查询
        if fetch == "all":
            result = cursor.result_rows  # 获取全部查询结果行
        elif fetch == "one":
            result = cursor.first_row  # 获取第一行查询结果
        else:
            raise ValueError("Fetch parameter must be either 'one' or 'all'")  # 如果fetch参数不合法，抛出数值错误异常

        field_names = cursor.column_names  # 获取查询结果的字段名
        result.insert(0, field_names)  # 将字段名插入到结果列表的开头
        return result  # 返回查询结果列表

    def __sql_parse(self, sql):
        sql = sql.strip()  # 去除SQL语句两端的空白字符
        parsed = sqlparse.parse(sql)[0]  # 使用sqlparse解析SQL语句
        sql_type = parsed.get_type()  # 获取SQL语句的类型
        if sql_type == "CREATE":
            table_name = self._extract_table_name_from_ddl(parsed)  # 如果是CREATE语句，从DDL中提取表名
        else:
            table_name = parsed.get_name()  # 否则直接获取SQL语句中的对象名称

        first_token = parsed.token_first(skip_ws=True, skip_cm=False)  # 获取SQL语句的第一个标记
        ttype = first_token.ttype  # 获取第一个标记的类型
        logger.info(
            f"SQL:{sql}, ttype:{ttype}, sql_type:{sql_type}, table:{table_name}"
        )  # 记录SQL解析相关信息的日志
        return parsed, ttype, sql_type, table_name  # 返回解析结果及相关信息

    def _sync_tables_from_db(self) -> Iterable[str]:
        """Read table information from database."""
        # TODO Use a background thread to refresh periodically
        # 从数据库中读取表信息

        # SQL will raise error with schema
        _schema = (
            None if self.db_type == DBType.SQLite.value() else self._engine.url.database
        )  # 根据数据库类型设置schema，SQLite不需要schema

        # including view support by adding the views as well as tables to the all
        # tables list if view_support is True
        self._all_tables = set(
            self._inspector.get_table_names(schema=_schema)
            + (
                self._inspector.get_view_names(schema=_schema)
                if self.view_support
                else []
            )  # 获取数据库中的所有表名和视图名
        )
        return self._all_tables  # 返回所有表和视图名的集合
```