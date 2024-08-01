# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_sqlite.py`

```py
# 导入日志模块和操作系统模块
import logging
import os
# 导入临时文件模块
import tempfile
# 导入类型提示相关的模块
from typing import Any, Iterable, List, Optional, Tuple

# 导入 SQLAlchemy 中的创建引擎和文本对象
from sqlalchemy import create_engine, text

# 导入 RDBMSConnector 基类
from .base import RDBMSConnector

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class SQLiteConnector(RDBMSConnector):
    """SQLite 数据库连接器。"""

    # 定义数据库类型和方言为 SQLite
    db_type: str = "sqlite"
    db_dialect: str = "sqlite"

    @classmethod
    def from_file_path(
        cls, file_path: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> "SQLiteConnector":
        """从文件路径创建一个新的 SQLiteConnector 实例。"""
        # 如果未提供 engine_args 参数，则设为空字典
        _engine_args = engine_args or {}
        # 设置 SQLite 连接参数，检查是否在同一线程中进行检查
        _engine_args["connect_args"] = {"check_same_thread": False}
        # 创建目录以存放 SQLite 数据库文件
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 使用文件路径创建 SQLite 连接，并返回 SQLiteConnector 实例
        return cls(create_engine("sqlite:///" + file_path, **_engine_args), **kwargs)

    def get_indexes(self, table_name):
        """获取指定表的索引信息。"""
        # 执行 SQL 查询，获取表的索引列表
        cursor = self.session.execute(text(f"PRAGMA index_list({table_name})"))
        indexes = cursor.fetchall()
        result = []
        # 遍历索引列表，获取每个索引的详细信息
        for idx in indexes:
            index_name = idx[1]
            cursor = self.session.execute(text(f"PRAGMA index_info({index_name})"))
            index_infos = cursor.fetchall()
            # 提取索引包含的列名，并存入结果列表
            column_names = [index_info[2] for index_info in index_infos]
            result.append({"name": index_name, "column_names": column_names})
        return result

    def get_show_create_table(self, table_name):
        """获取指定表的创建 SQL 语句。"""
        # 执行 SQL 查询，获取指定表的创建 SQL 语句
        cursor = self.session.execute(
            text(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                f"AND name='{table_name}'"
            )
        )
        ans = cursor.fetchall()
        return ans[0][0]

    def get_fields(self, table_name) -> List[Tuple]:
        """获取指定表的字段信息。"""
        # 执行 SQL 查询，获取指定表的字段信息
        cursor = self.session.execute(text(f"PRAGMA table_info('{table_name}')"))
        fields = cursor.fetchall()
        # 记录字段信息到日志中
        logger.info(fields)
        # 返回字段信息的列表，每个元素是字段的详细信息元组
        return [(field[1], field[2], field[3], field[4], field[5]) for field in fields]

    def get_simple_fields(self, table_name):
        """获取指定表的简化字段信息。"""
        # 调用 get_fields 方法获取指定表的字段信息，并直接返回
        return self.get_fields(table_name)

    def get_users(self):
        """获取用户信息。"""
        # SQLite 不支持用户管理，因此返回空列表
        return []

    def get_grants(self):
        """获取授权信息。"""
        # SQLite 不支持授权管理，因此返回空列表
        return []

    def get_collation(self):
        """获取数据库的排序规则。"""
        # 返回当前数据库的排序规则为 UTF-8
        return "UTF-8"

    def get_charset(self):
        """获取当前数据库的字符集。"""
        # 返回当前数据库的字符集为 UTF-8
        return "UTF-8"

    def get_database_names(self):
        """获取数据库名称列表。"""
        # SQLite 不支持多数据库操作，因此返回空列表
        return []
    def _sync_tables_from_db(self) -> Iterable[str]:
        # 从数据库中同步表和视图的名称
        table_results = self.session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        view_results = self.session.execute(
            text("SELECT name FROM sqlite_master WHERE type='view'")
        )
        # 将查询结果转换为集合，存储表和视图的名称
        table_results = set(row[0] for row in table_results)  # noqa
        view_results = set(row[0] for row in view_results)  # noqa
        # 合并所有表和视图的名称
        self._all_tables = table_results.union(view_results)
        # 使用数据库引擎反射元数据
        self._metadata.reflect(bind=self._engine)
        return self._all_tables

    def _write(self, write_sql):
        # 记录写操作的日志信息
        logger.info(f"Write[{write_sql}]")
        session = self.session
        # 执行写操作的 SQL
        result = session.execute(text(write_sql))
        # 提交事务
        session.commit()
        # 记录写操作执行后的结果行数
        logger.info(f"SQL[{write_sql}], result:{result.rowcount}")
        return result.rowcount

    def get_table_comments(self, db_name=None):
        """获取表的注释信息."""
        # 查询数据库中所有表的名称和创建 SQL
        cursor = self.session.execute(
            text(
                """
                SELECT name, sql FROM sqlite_master WHERE type='table'
                """
            )
        )
        # 获取所有表的注释信息并返回
        table_comments = cursor.fetchall()
        return [
            (table_comment[0], table_comment[1]) for table_comment in table_comments
        ]

    def get_current_db_name(self) -> str:
        """获取当前数据库的名称.

        Returns:
            str: 数据库名称
        """
        # 获取数据库连接的完整路径
        full_path = self._engine.url.database
        # 提取数据库名称并去掉扩展名 .db
        db_name = os.path.basename(full_path)
        if db_name.endswith(".db"):
            db_name = db_name[:-3]
        return db_name

    def table_simple_info(self) -> Iterable[str]:
        """获取表的简单信息."""
        # 查询数据库中所有表的名称
        _tables_sql = """
                SELECT name FROM sqlite_master WHERE type='table'
            """
        cursor = self.session.execute(text(_tables_sql))
        tables_results = cursor.fetchall()
        results = []
        # 遍历每个表，获取表的列信息
        for row in tables_results:
            table_name = row[0]
            _sql = f"""
                PRAGMA  table_info({table_name})
            """
            cursor_colums = self.session.execute(text(_sql))
            colum_results = cursor_colums.fetchall()
            table_columns = []
            # 遍历每个列，获取列的名称
            for row_col in colum_results:
                field_info = list(row_col)
                table_columns.append(field_info[1])

            # 将表名和列名拼接为字符串，存储在结果列表中
            results.append(f"{table_name}({','.join(table_columns)});")
        return results
class SQLiteTempConnector(SQLiteConnector):
    """A temporary SQLite database connection.

    The database file will be deleted when the connection is closed.
    """

    def __init__(self, engine, temp_file_path, *args, **kwargs):
        """Construct a temporary SQLite database connection."""
        super().__init__(engine, *args, **kwargs)
        self.temp_file_path = temp_file_path  # 设置临时数据库文件路径
        self._is_closed = False  # 初始化连接状态为未关闭

    @classmethod
    def create_temporary_db(
        cls, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> "SQLiteTempConnector":
        """Create a temporary SQLite database with a temporary file.

        Examples:
            .. code-block:: python

                with SQLiteTempConnect.create_temporary_db() as db:
                    db.run(db.session, "CREATE TABLE test (id INTEGER PRIMARY KEY);")
                    db.run(db.session, "insert into test(id) values (1)")
                    db.run(db.session, "insert into test(id) values (2)")
                    field_names, result = db.query_ex(db.session, "select * from test")
                    assert field_names == ["id"]
                    assert result == [(1,), (2,)]

        Args:
            engine_args (Optional[dict]): SQLAlchemy engine arguments.

        Returns:
            SQLiteTempConnector: A SQLiteTempConnect instance.
        """
        _engine_args = engine_args or {}  # 如果未提供引擎参数，则设为空字典
        _engine_args["connect_args"] = {"check_same_thread": False}  # 设置连接参数

        temp_file = tempfile.NamedTemporaryFile(delete=False)  # 创建一个命名临时文件
        temp_file_path = temp_file.name  # 获取临时文件的路径
        temp_file.close()  # 关闭临时文件

        engine = create_engine(f"sqlite:///{temp_file_path}", **_engine_args)  # 使用临时文件路径创建 SQLite 引擎
        return cls(engine, temp_file_path, **kwargs)  # 返回创建的 SQLiteTempConnector 实例

    def close(self):
        """Close the connection."""
        if not self._is_closed:  # 如果连接未关闭
            if self._engine:
                self._engine.dispose()  # 释放数据库引擎资源
            try:
                if os.path.exists(self.temp_file_path):  # 如果临时文件路径存在
                    os.remove(self.temp_file_path)  # 删除临时数据库文件
            except Exception as e:
                logger.error(f"Error removing temporary database file: {e}")  # 记录删除临时文件时的错误
            self._is_closed = True  # 标记连接已关闭
    def create_temp_tables(self, tables_info):
        """Create temporary tables with data.

        Examples:
            .. code-block:: python

                tables_info = {
                    "test": {
                        "columns": {
                            "id": "INTEGER PRIMARY KEY",
                            "name": "TEXT",
                            "age": "INTEGER",
                        },
                        "data": [
                            (1, "Tom", 20),
                            (2, "Jack", 21),
                            (3, "Alice", 22),
                        ],
                    },
                }
                with SQLiteTempConnector.create_temporary_db() as db:
                    db.create_temp_tables(tables_info)
                    field_names, result = db.query_ex(db.session, "select * from test")
                    assert field_names == ["id", "name", "age"]
                    assert result == [(1, "Tom", 20), (2, "Jack", 21), (3, "Alice", 22)]

        Args:
            tables_info (dict): A dictionary of table information.
        """
        # 遍历提供的表信息字典
        for table_name, table_data in tables_info.items():
            # 构建表的列信息字符串
            columns = ", ".join(
                [f"{col} {dtype}" for col, dtype in table_data["columns"].items()]
            )
            # 构建创建表的 SQL 语句
            create_sql = f"CREATE TABLE {table_name} ({columns});"
            # 执行 SQL 语句来创建表
            self.session.execute(text(create_sql))
            # 遍历表中的数据行，准备插入数据
            for row in table_data.get("data", []):
                # 构建插入数据的占位符字符串
                placeholders = ", ".join(
                    [":param" + str(index) for index, _ in enumerate(row)]
                )
                # 构建插入数据的 SQL 语句
                insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders});"

                # 准备数据插入时所需的参数字典
                param_dict = {
                    "param" + str(index): value for index, value in enumerate(row)
                }
                # 执行插入数据的 SQL 语句
                self.session.execute(text(insert_sql), param_dict)
            # 提交事务，确保数据被写入数据库
            self.session.commit()
        # 同步数据库中的表信息到对象
        self._sync_tables_from_db()

    def __enter__(self):
        """Return the connection when entering the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the connection when exiting the context manager."""
        # 在退出上下文管理器时关闭数据库连接
        self.close()

    def __del__(self):
        """Close the connection when the object is deleted."""
        # 在对象被删除时关闭数据库连接
        self.close()

    @classmethod
    def is_normal_type(cls) -> bool:
        """Return whether the connector is a normal type."""
        # 返回连接器是否为普通类型的布尔值结果
        return False
```