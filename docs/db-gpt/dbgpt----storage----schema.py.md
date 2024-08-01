# `.\DB-GPT-src\dbgpt\storage\schema.py`

```py
"""Database information class and database type enumeration."""
import os  # 导入操作系统模块
from enum import Enum  # 导入枚举类型
from typing import Optional  # 导入可选类型


class DbInfo:
    """Database information class."""

    def __init__(self, name, is_file_db: bool = False):
        """Create a new instance of DbInfo.

        Args:
            name (str): The name of the database.
            is_file_db (bool, optional): Whether the database is file-based. Defaults to False.
        """
        self.name = name  # 设置数据库名称
        self.is_file_db = is_file_db  # 设置数据库是否为文件数据库


class DBType(Enum):
    """Database type enumeration."""

    MySQL = DbInfo("mysql")  # MySQL数据库
    OceanBase = DbInfo("oceanbase")  # OceanBase数据库
    DuckDb = DbInfo("duckdb", True)  # DuckDB文件数据库
    SQLite = DbInfo("sqlite", True)  # SQLite文件数据库
    Oracle = DbInfo("oracle")  # Oracle数据库
    MSSQL = DbInfo("mssql")  # Microsoft SQL Server数据库
    Postgresql = DbInfo("postgresql")  # PostgreSQL数据库
    Vertica = DbInfo("vertica")  # Vertica数据库
    Clickhouse = DbInfo("clickhouse")  # ClickHouse数据库
    StarRocks = DbInfo("starrocks")  # StarRocks数据库
    Spark = DbInfo("spark", True)  # Spark文件数据库
    Doris = DbInfo("doris")  # Doris数据库
    Hive = DbInfo("hive")  # Hive数据库
    TuGraph = DbInfo("tugraph")  # TuGraph数据库

    def value(self) -> str:
        """Return the name of the database type."""
        return self._value_.name  # 返回数据库类型的名称

    def is_file_db(self) -> bool:
        """Return whether the database is a file database."""
        return self._value_.is_file_db  # 返回数据库是否为文件数据库的布尔值

    @staticmethod
    def of_db_type(db_type: str) -> Optional["DBType"]:
        """Return the database type of the given name.

        Args:
            db_type (str): The name of the database type.

        Returns:
            Optional[DBType]: The database type of the given name, or None if not found.
        """
        for item in DBType:
            if item.value() == db_type:
                return item  # 如果找到匹配的数据库类型名称，返回对应的枚举项
        return None  # 如果未找到匹配的数据库类型名称，返回None

    @staticmethod
    def parse_file_db_name_from_path(db_type: str, local_db_path: str):
        """Parse out the database name of the embedded database from the file path.

        Args:
            db_type (str): The name of the database type.
            local_db_path (str): The local file path containing the database.

        Returns:
            str: The parsed database name based on the file path.
        """
        base_name = os.path.basename(local_db_path)  # 获取路径中的基本文件名
        db_name = os.path.splitext(base_name)[0]  # 去除文件扩展名后的文件名
        if "." in db_name:
            db_name = os.path.splitext(db_name)[0]  # 再次去除可能存在的额外扩展名
        return db_type + "_" + db_name  # 返回组合后的数据库名称
```