# `.\DB-GPT-src\dbgpt\datasource\rdbms\base.py`

```py
"""Base class for RDBMS connectors."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote

import sqlalchemy
import sqlparse
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.schema import CreateTable

from dbgpt.datasource.base import BaseConnector
from dbgpt.storage.schema import DBType

logger = logging.getLogger(__name__)


def _format_index(index: sqlalchemy.engine.interfaces.ReflectedIndex) -> str:
    return (
        f'Name: {index["name"]}, Unique: {index["unique"]},'
        f' Columns: {str(index["column_names"])}'
    )


class RDBMSConnector(BaseConnector):
    """SQLAlchemy wrapper around a database."""

    def __init__(
        self,
        engine,
        schema: Optional[str] = None,
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        indexes_in_table_info: bool = False,
        custom_table_info: Optional[Dict[str, str]] = None,
        view_support: bool = False,
    ):
        """Create engine from database URI.

        Args:
           - engine: Engine sqlalchemy.engine
           - schema: Optional[str].
           - metadata: Optional[MetaData]
           - ignore_tables: Optional[List[str]]
           - include_tables: Optional[List[str]]
           - sample_rows_in_table_info: int default:3,
           - indexes_in_table_info: bool = False,
           - custom_table_info: Optional[dict] = None,
           - view_support: bool = False,
        """
        self._engine = engine  # 设置数据库引擎
        self._schema = schema  # 指定数据库模式

        # 检查是否同时指定了包含和忽略的表，如果是则抛出异常
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        # 初始化自定义表信息，如果未提供则为空字典
        if not custom_table_info:
            custom_table_info = {}

        # 使用引擎创建一个数据库检查对象
        self._inspector = inspect(engine)

        # 创建会话工厂和作用域会话管理器
        session_factory = sessionmaker(bind=engine)
        Session_Manages = scoped_session(session_factory)
        self._db_sessions = Session_Manages
        self.session = self.get_session()  # 获取当前会话

        # 设置是否支持视图
        self.view_support = view_support

        # 初始化内部使用的表集合和包含、忽略的表集合
        self._usable_tables: Set[str] = set()
        self._include_tables: Set[str] = set()
        self._ignore_tables: Set[str] = set()

        # 设置自定义表信息、表信息采样行数、是否包含索引信息
        self._custom_table_info = custom_table_info
        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = indexes_in_table_info

        # 反射元数据并从数据库同步所有表信息
        self._metadata = metadata or MetaData()
        self._metadata.reflect(bind=self._engine)
        self._all_tables: Set[str] = cast(Set[str], self._sync_tables_from_db())

    @classmethod
    # 构建从 URI 数据库连接的类方法
    def from_uri_db(
        cls,
        host: str,
        port: int,
        user: str,
        pwd: str,
        db_name: str,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> RDBMSConnector:
        """Construct a SQLAlchemy engine from uri database.

        Args:
            host (str): database host.
            port (int): database port.
            user (str): database user.
            pwd (str): database password.
            db_name (str): database name.
            engine_args (Optional[dict]): other engine_args.
        """
        # 构建数据库连接的 URI
        db_url: str = (
            f"{cls.driver}://{quote(user)}:{urlquote(pwd)}@{host}:{str(port)}/{db_name}"
        )
        # 调用另一个类方法，根据 URI 构建 SQLAlchemy 引擎并返回连接器实例
        return cls.from_uri(db_url, engine_args, **kwargs)

    @classmethod
    def from_uri(
        cls, database_uri: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> RDBMSConnector:
        """Construct a SQLAlchemy engine from URI."""
        _engine_args = engine_args or {}
        # 使用 SQLAlchemy 的 create_engine 方法构建引擎，并返回连接器实例
        return cls(create_engine(database_uri, **_engine_args), **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        # 返回当前引擎所使用的方言名称
        return self._engine.dialect.name

    def _sync_tables_from_db(self) -> Iterable[str]:
        """Read table information from database."""
        # TODO 使用后台线程定期刷新

        # 根据数据库类型判断是否需要指定模式（schema），SQLite 不支持
        _schema = (
            None if self.db_type == DBType.SQLite.value() else self._engine.url.database
        )
        # 获取数据库中所有表和视图的名称，并根据视图支持设置是否包含视图
        self._all_tables = set(
            self._inspector.get_table_names(schema=_schema)
            + (
                self._inspector.get_view_names(schema=_schema)
                if self.view_support
                else []
            )
        )
        return self._all_tables

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        # 如果设置了包含的表，则返回包含的表名，否则返回所有表名减去忽略的表名
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        # 返回可用表名，实际调用了 get_usable_table_names 方法
        return self.get_usable_table_names()

    def get_session(self):
        """Get session."""
        # 获取数据库会话对象
        session = self._db_sessions()
        return session

    def get_current_db_name(self) -> str:
        """Get current database name.

        Returns:
            str: database name
        """
        # 执行 SQL 查询获取当前数据库的名称并返回
        return self.session.execute(text("SELECT DATABASE()")).scalar()
    # 返回数据库中所有表的简要信息，包括表名及其列名拼接而成的字符串
    def table_simple_info(self):
        """Return table simple info."""
        # 构建 SQL 查询语句，获取当前数据库中每张表的表名及其列名拼接信息
        _sql = f"""
                select concat(table_name, "(" , group_concat(column_name), ")")
                as schema_info from information_schema.COLUMNS where
                table_schema="{self.get_current_db_name()}" group by TABLE_NAME;
            """
        # 执行 SQL 查询并获取游标
        cursor = self.session.execute(text(_sql))
        # 获取所有查询结果
        results = cursor.fetchall()
        # 返回查询结果
        return results

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        # 返回数据库中所有表的详细信息
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        # 获取当前数据库中可用的所有表名
        all_table_names = self.get_usable_table_names()
        
        # 如果指定了特定的表名列表，则检查是否存在于数据库中，若不存在则抛出异常
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        # 从 metadata 中过滤出与指定表名匹配的表对象
        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            # 如果用户提供了自定义的表信息，则使用该信息
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # 获取创建表的 SQL 命令并转换为字符串形式
            create_table = str(CreateTable(table).compile(self._engine))
            table_info = f"{create_table.rstrip()}"
            
            # 检查是否需要额外的表信息（索引或样本行信息）
            has_extra_info = (
                self._indexes_in_table_info or self._sample_rows_in_table_info
            )
            if has_extra_info:
                table_info += "\n\n/*"
            if self._indexes_in_table_info:
                # 添加表的索引信息
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self._sample_rows_in_table_info:
                # 添加表的样本行信息
                table_info += f"\n{self._get_sample_rows(table)}\n"
            if has_extra_info:
                table_info += "*/"
            
            # 将处理后的表信息添加到列表中
            tables.append(table_info)
        
        # 将所有表信息连接成一个字符串并返回
        final_str = "\n\n".join(tables)
        return final_str
    # 获取指定表的列信息
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
        return self._inspector.get_columns(table_name)

    # 获取表中的样本行
    def _get_sample_rows(self, table: Table) -> str:
        # 构建查询命令
        command = select(table).limit(self._sample_rows_in_table_info)

        # 保存列的字符串格式
        columns_str = "\t".join([col.name for col in table.columns])

        try:
            # 获取样本行
            with self._engine.connect() as connection:
                sample_rows_result: CursorResult = connection.execute(command)
                # 缩短样本行中的值
                sample_rows = list(
                    map(lambda ls: [str(i)[:100] for i in ls], sample_rows_result)
                )

            # 保存样本行的字符串格式
            sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

        # 在某些方言中，当表中没有行时会返回'ProgrammingError'
        except ProgrammingError:
            sample_rows_str = ""

        return (
            f"{self._sample_rows_in_table_info} rows from {table.name} table:\n"
            f"{columns_str}\n"
            f"{sample_rows_str}"
        )

    # 获取表的索引信息
    def _get_table_indexes(self, table: Table) -> str:
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(map(_format_index, indexes))
        return f"Table Indexes:\n{indexes_formatted}"

    # 获取指定表的信息，避免抛出异常
    def get_table_info_no_throw(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables."""
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    # 运行 SQL 写入命令并将结果作为元组列表返回
    def _write(self, write_sql: str):
        """Run a SQL write command and return the results as a list of tuples.

        Args:
            write_sql (str): SQL write command to run
        """
        logger.info(f"Write[{write_sql}]")
        db_cache = self._engine.url.database
        result = self.session.execute(text(write_sql))
        self.session.commit()
        # TODO  Subsequent optimization of dynamically specified database submission
        #  loss target problem
        self.session.execute(text(f"use `{db_cache}`"))
        logger.info(f"SQL[{write_sql}], result:{result.rowcount}")
        return result.rowcount
    def _query(self, query: str, fetch: str = "all"):
        """Run a SQL query and return the results as a list of tuples.

        Args:
            query (str): SQL query to run
            fetch (str): fetch type ("all" for all results, "one" for single result)

        Returns:
            List: list of tuples representing query results
        """
        result: List[Any] = []  # Initialize an empty list to store query results

        logger.info(f"Query[{query}]")  # Log the SQL query being executed
        if not query:
            return result  # If query is empty, return empty result list
        cursor = self.session.execute(text(query))  # Execute the SQL query using SQLAlchemy's text function

        if cursor.returns_rows:
            if fetch == "all":
                result = cursor.fetchall()  # Fetch all rows if fetch type is "all"
            elif fetch == "one":
                result = [cursor.fetchone()]  # Fetch a single row if fetch type is "one"
            else:
                raise ValueError("Fetch parameter must be either 'one' or 'all'")

            field_names = tuple(i[0:] for i in cursor.keys())  # Get field names from cursor

            result.insert(0, field_names)  # Insert field names as the first element in the result list
            return result  # Return the list of tuples representing query results

    def query_table_schema(self, table_name: str):
        """Query table schema.

        Args:
            table_name (str): table name

        Returns:
            List: list of tuples representing schema of the table
        """
        sql = f"select * from {table_name} limit 1"  # SQL query to fetch schema of the table
        return self._query(sql)  # Call _query method to execute the SQL query and return results

    def query_ex(self, query: str, fetch: str = "all"):
        """Execute a SQL command and return the results.

        Only for query command.

        Args:
            query (str): SQL query to run
            fetch (str): fetch type ("all" for all results, "one" for single result)

        Returns:
            Tuple: tuple containing field names (list) and query results (list of tuples)
        """
        logger.info(f"Query[{query}]")  # Log the SQL query being executed
        if not query:
            return [], None  # Return empty lists if query is empty
        cursor = self.session.execute(text(query))  # Execute the SQL query

        if cursor.returns_rows:
            if fetch == "all":
                result = cursor.fetchall()  # Fetch all rows if fetch type is "all"
            elif fetch == "one":
                result = cursor.fetchone()  # Fetch a single row if fetch type is "one"
            else:
                raise ValueError("Fetch parameter must be either 'one' or 'all'")

            field_names = list(cursor.keys())  # Get field names from cursor

            result = list(result)  # Convert result to a list
            return field_names, result  # Return tuple containing field names and query results

        return [], None  # Return empty lists if cursor does not return rows
    def run(self, command: str, fetch: str = "all") -> List:
        """Execute a SQL command and return a string representing the results."""
        # 记录日志，记录执行的 SQL 命令
        logger.info("SQL:" + command)
        
        # 检查命令是否为空或长度小于0，如果是则返回空列表
        if not command or len(command) < 0:
            return []
        
        # 解析 SQL 命令，获取命令类型、SQL 类型和表名等信息
        parsed, ttype, sql_type, table_name = self.__sql_parse(command)
        
        # 如果是数据操作语言（DML）
        if ttype == sqlparse.tokens.DML:
            # 如果是 SELECT 命令
            if sql_type == "SELECT":
                # 执行查询操作
                return self._query(command, fetch)
            else:
                # 执行写入操作
                self._write(command)
                # 将写入的命令转换为对应的 SELECT 查询
                select_sql = self.convert_sql_write_to_select(command)
                logger.info(f"write result query:{select_sql}")
                # 执行转换后的查询并返回结果
                return self._query(select_sql)

        else:
            # 如果是数据定义语言（DDL）
            logger.info(
                "DDL execution determines whether to enable through configuration "
            )
            # 执行 SQL 命令并提交事务
            cursor = self.session.execute(text(command))
            self.session.commit()
            
            # 如果返回结果集
            if cursor.returns_rows:
                # 获取所有查询结果
                result = cursor.fetchall()
                # 获取字段名列表
                field_names = tuple(i[0:] for i in cursor.keys())
                result = list(result)
                # 将字段名插入结果列表的首部
                result.insert(0, field_names)
                logger.info("DDL Result:" + str(result))
                
                # 如果结果为空
                if not result:
                    # 返回表的简单字段信息
                    return self.get_simple_fields(table_name)
                return result
            else:
                # 如果没有返回结果集，返回表的简单字段信息
                return self.get_simple_fields(table_name)


    def run_to_df(self, command: str, fetch: str = "all"):
        """Execute sql command and return result as dataframe."""
        import pandas as pd
        
        # 使用 pandas 将结果转换为 DataFrame
        # Pandas 依赖性过大，导入时间过长
        # TODO: 移除对 pandas 的依赖
        result_lst = self.run(command, fetch)
        colunms = result_lst[0]
        values = result_lst[1:]
        return pd.DataFrame(values, columns=colunms)


    def run_no_throw(self, command: str, fetch: str = "all") -> List:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            # 尝试执行 SQL 命令，如果有结果则返回结果列表
            return self.run(command, fetch)
        except SQLAlchemyError as e:
            # 捕获 SQLAlchemy 异常，记录警告日志并返回空列表
            logger.warning(f"Run SQL command failed: {e}")
            return []
    def convert_sql_write_to_select(self, write_sql: str) -> str:
        """Convert SQL write command to a SELECT command.

        SQL classification processing
        author:xiangh8

        Examples:
            .. code-block:: python

                write_sql = "insert into test(id) values (1)"
                select_sql = convert_sql_write_to_select(write_sql)
                print(select_sql)
                # SELECT * FROM test WHERE id=1
        Args:
            write_sql (str): SQL write command

        Returns:
            str: SELECT command corresponding to the write command
        """
        # Convert the SQL command to lowercase and split by space
        parts = write_sql.lower().split()
        # Get the command type (insert, delete, update)
        cmd_type = parts[0]

        # Handle according to command type
        if cmd_type == "insert":
            match = re.match(
                r"insert\s+into\s+(\w+)\s*\(([^)]+)\)\s*values\s*\(([^)]+)\)",
                write_sql.lower(),
            )
            if match:
                # Get the table name, columns, and values
                table_name, columns, values = match.groups()
                columns = columns.split(",")
                values = values.split(",")
                # Build the WHERE clause
                where_clause = " AND ".join(
                    [
                        f"{col.strip()}={val.strip()}"
                        for col, val in zip(columns, values)
                    ]
                )
                return f"SELECT * FROM {table_name} WHERE {where_clause}"
            else:
                raise ValueError(f"Unsupported SQL command: {write_sql}")

        elif cmd_type == "delete":
            table_name = parts[2]  # delete from <table_name> ...
            # Return a SELECT statement that selects all data from the table
            return f"SELECT * FROM {table_name} "

        elif cmd_type == "update":
            table_name = parts[1]
            set_idx = parts.index("set")
            where_idx = parts.index("where")
            # Get the field name in the `set` clause
            set_clause = parts[set_idx + 1 : where_idx][0].split("=")[0].strip()
            # Get the condition statement after the `where`
            where_clause = " ".join(parts[where_idx + 1 :])
            # Return a SELECT statement that selects the updated data
            return f"SELECT {set_clause} FROM {table_name} WHERE {where_clause}"
        else:
            raise ValueError(f"Unsupported SQL command type: {cmd_type}")
    def __sql_parse(self, sql):
        # 去除 SQL 语句两端的空白字符
        sql = sql.strip()
        # 使用 sqlparse 解析 SQL 语句，取得第一个解析结果
        parsed = sqlparse.parse(sql)[0]
        # 获取 SQL 语句的类型
        sql_type = parsed.get_type()
        # 根据 SQL 类型决定获取表名的方式
        if sql_type == "CREATE":
            # 如果是 CREATE 语句，从 DDL 中提取表名
            table_name = self._extract_table_name_from_ddl(parsed)
        else:
            # 否则直接获取 SQL 语句中的名称作为表名
            table_name = parsed.get_name()

        # 获取解析后 SQL 的第一个 token，并获取其类型
        first_token = parsed.token_first(skip_ws=True, skip_cm=False)
        ttype = first_token.ttype
        # 记录日志，包括 SQL 语句、token 类型、SQL 类型、表名等信息
        logger.info(
            f"SQL:{sql}, ttype:{ttype}, sql_type:{sql_type}, table:{table_name}"
        )
        # 返回解析后的结果、token 类型、SQL 类型和表名
        return parsed, ttype, sql_type, table_name

    def _extract_table_name_from_ddl(self, parsed):
        """Extract table name from CREATE TABLE statement.""" ""
        # 遍历解析后的 token，找到 Identifier 类型的 token，并返回其真实名称作为表名
        for token in parsed.tokens:
            if token.ttype is None and isinstance(token, sqlparse.sql.Identifier):
                return token.get_real_name()
        # 如果未找到合适的 token，则返回 None
        return None

    def get_indexes(self, table_name: str) -> List[Dict]:
        """Get table indexes about specified table.

        Args:
            table_name:(str) table name

        Returns:
            List[Dict]:eg:[{'name': 'idx_key', 'column_names': ['id']}]
        """
        # 使用 _inspector 对象获取指定表的索引信息，并返回结果
        return self._inspector.get_indexes(table_name)

    def get_show_create_table(self, table_name):
        """Get table show create table about specified table."""
        # 创建数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，获取指定表的创建语句
        cursor = session.execute(text(f"SHOW CREATE TABLE  {table_name}"))
        ans = cursor.fetchall()
        # 返回查询结果中第一行的第二列，即创建表的 SQL 语句
        return ans[0][1]

    def get_fields(self, table_name) -> List[Tuple]:
        """Get column fields about specified table."""
        # 创建数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，获取指定表的字段信息
        cursor = session.execute(
            text(
                "SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_DEFAULT, IS_NULLABLE, "
                "COLUMN_COMMENT  from information_schema.COLUMNS where "
                f"table_name='{table_name}'".format(table_name)
            )
        )
        # 获取查询结果的所有字段信息，以列表形式返回
        fields = cursor.fetchall()
        return [(field[0], field[1], field[2], field[3], field[4]) for field in fields]

    def get_simple_fields(self, table_name):
        """Get column fields about specified table."""
        # 调用 _query 方法执行 SQL 查询，获取指定表的字段信息
        return self._query(f"SHOW COLUMNS FROM {table_name}")

    def get_charset(self) -> str:
        """Get character_set."""
        # 创建数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，获取当前数据库的字符集
        cursor = session.execute(text("SELECT @@character_set_database"))
        # 获取查询结果的第一行第一列的值，即数据库的字符集
        character_set = cursor.fetchone()[0]  # type: ignore
        return character_set

    def get_collation(self):
        """Get collation."""
        # 创建数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，获取当前数据库的排序规则（collation）
        cursor = session.execute(text("SELECT @@collation_database"))
        # 获取查询结果的第一行第一列的值，即数据库的排序规则（collation）
        collation = cursor.fetchone()[0]
        return collation

    def get_grants(self):
        """Get grant info."""
        # 创建数据库会话
        session = self._db_sessions()
        # 执行 SQL 查询，获取数据库的授权信息（GRANTS）
        cursor = session.execute(text("SHOW GRANTS"))
        # 获取查询结果的所有行，即数据库的授权信息列表
        grants = cursor.fetchall()
        return grants
    def get_users(self):
        """获取用户信息。"""
        try:
            # 使用 SQL 查询语句获取 MySQL 用户表中的用户和主机信息
            cursor = self.session.execute(text("SELECT user, host FROM mysql.user"))
            # 获取所有查询结果
            users = cursor.fetchall()
            # 将结果转换为列表，每个元素为 (user, host) 的元组
            return [(user[0], user[1]) for user in users]
        except Exception:
            # 如果发生异常，返回空列表
            return []

    def get_table_comments(self, db_name: str):
        """返回表的注释。"""
        # 使用 SQL 查询语句获取指定数据库中所有表的名称和注释
        cursor = self.session.execute(
            text(
                f"""SELECT table_name, table_comment FROM information_schema.tables
                    WHERE table_schema = '{db_name}'"""
            )
        )
        # 获取所有查询结果
        table_comments = cursor.fetchall()
        # 将结果转换为列表，每个元素为 (table_name, table_comment) 的元组
        return [(table_comment[0], table_comment[1]) for table_comment in table_comments]

    def get_table_comment(self, table_name: str) -> Dict:
        """获取表的注释。

        Args:
            table_name (str): 表名
        Returns:
            Dict: 包含文本注释的字典，例如 {"text": "comment"}
        """
        # 调用 _inspector 对象的方法获取指定表的注释信息
        return self._inspector.get_table_comment(table_name)

    def get_column_comments(self, db_name: str, table_name: str):
        """返回列的注释。"""
        # 使用 SQL 查询语句获取指定数据库和表中所有列的名称和注释
        cursor = self.session.execute(
            text(
                f"""SELECT column_name, column_comment FROM information_schema.columns
                    WHERE table_schema = '{db_name}' and table_name = '{table_name}'"""
            )
        )
        # 获取所有查询结果
        column_comments = cursor.fetchall()
        # 将结果转换为列表，每个元素为 (column_name, column_comment) 的元组
        return [(column_comment[0], column_comment[1]) for column_comment in column_comments]

    def get_database_names(self) -> List[str]:
        """返回数据库中可用的数据库名称列表。"""
        # 获取数据库会话
        session = self._db_sessions()
        # 使用 SQL 查询语句获取所有数据库的名称
        cursor = session.execute(text("SHOW DATABASES;"))
        # 获取所有查询结果
        results = cursor.fetchall()
        # 将结果转换为列表，过滤掉系统数据库，只保留用户数据库名称
        return [
            d[0]
            for d in results
            if d[0] not in ["information_schema", "performance_schema", "sys", "mysql"]
        ]
```