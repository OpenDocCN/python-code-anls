# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_mssql.py`

```py
"""MSSQL connector."""
# 导入必要的模块和类
from typing import Iterable

from sqlalchemy import text

from .base import RDBMSConnector


class MSSQLConnector(RDBMSConnector):
    """MSSQL connector."""

    # 定义数据库类型和方言
    db_type: str = "mssql"
    db_dialect: str = "mssql"
    driver: str = "mssql+pymssql"

    # 定义默认的数据库名称列表
    default_db = ["master", "model", "msdb", "tempdb", "modeldb", "resource", "sys"]

    def table_simple_info(self) -> Iterable[str]:
        """Get table simple info."""
        # SQL 查询语句，获取所有基本表的表名
        _tables_sql = """
                SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE
                TABLE_TYPE='BASE TABLE'
            """
        # 执行 SQL 查询语句并获取游标
        cursor = self.session.execute(text(_tables_sql))
        # 获取所有表名查询结果
        tables_results = cursor.fetchall()
        results = []
        # 遍历每个表名
        for row in tables_results:
            table_name = row[0]
            # 构建查询表结构信息的 SQL 查询语句
            _sql = f"""
                SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE
                 TABLE_NAME='{table_name}'
            """
            # 执行查询表结构信息的 SQL 查询语句并获取游标
            cursor_columns = self.session.execute(text(_sql))
            # 获取查询结果中的所有列信息
            column_results = cursor_columns.fetchall()
            table_columns = []
            # 遍历每个列的信息
            for row_col in column_results:
                # 将列名和数据类型组成列表，并添加到表的列信息列表中
                field_info = list(row_col)
                table_columns.append(field_info[0])
            # 将表名及其列信息拼接成字符串，并添加到结果列表中
            results.append(f"{table_name}({','.join(table_columns)});")
        # 返回所有表的简单信息列表
        return results
```