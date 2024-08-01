# `.\DB-GPT-src\dbgpt\datasource\rdbms\dialect\starrocks\sqlalchemy\dialect.py`

```py
"""StarRocks dialect for SQLAlchemy."""
# 版权声明和许可信息，指明此代码的版权归属和使用许可
# Copyright 2021-present StarRocks, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版本授权使用此代码；
# you may not use this file except in compliance with the License.
# 除非符合许可证的规定，否则不得使用此文件。
# You may obtain a copy of the License at
# 你可以在以下网址获取许可证副本：
#
#     https:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，否则依据许可分发的软件是按"原样"分发的，不附带任何明示或暗示的担保或条件。
import logging
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import exc, log, text
from sqlalchemy.dialects.mysql.pymysql import MySQLDialect_pymysql
from sqlalchemy.engine import Connection

from dbgpt.datasource.rdbms.dialect.starrocks.sqlalchemy import datatype

# 设置日志记录器
logger = logging.getLogger(__name__)


@log.class_logger
class StarRocksDialect(MySQLDialect_pymysql):  # type: ignore
    """StarRocks dialect for SQLAlchemy."""

    # 缓存
    # 如果不显式设置此标志，则SQLAlchemy会生成警告，并且在启用之前需要进行测试
    supports_statement_cache = False

    # 方言名称
    name = "starrocks"

    def __init__(self, *args, **kw):
        """Create a new StarRocks dialect."""
        super(StarRocksDialect, self).__init__(*args, **kw)

    def has_table(self, connection, table_name, schema: Optional[str] = None, **kw):
        """Return True if the given table is present in the database."""
        # 确保连接具备查询表结构的能力
        self._ensure_has_table_connection(connection)

        if schema is None:
            schema = self.default_schema_name

        assert schema is not None

        quote = self.identifier_preparer.quote_identifier
        full_name = quote(table_name)
        if schema:
            full_name = "{}.{}".format(quote(schema), full_name)

        # 执行 DESCRIBE 查询以确定表是否存在
        res = connection.execute(text(f"DESCRIBE {full_name}"))
        return res.first() is not None

    def get_schema_names(self, connection, **kw):
        """Return a list of schema names available in the database."""
        # 执行 SHOW schemas 查询以获取数据库中所有模式的名称列表
        rp = connection.exec_driver_sql("SHOW schemas")
        return [r[0] for r in rp]

    def get_table_names(self, connection, schema: Optional[str] = None, **kw):
        """Return a Unicode SHOW TABLES from a given schema."""
        current_schema: str = cast(str, schema or self.default_schema_name)

        charset = self._connection_charset

        # 执行 SHOW FULL TABLES 查询以获取指定模式中所有表的名称
        rp = connection.exec_driver_sql(
            "SHOW FULL TABLES FROM %s"
            % self.identifier_preparer.quote_identifier(current_schema)
        )

        # 提取所有基本表（BASE TABLE）的名称并返回
        return [
            row[0]
            for row in self._compat_fetchall(rp, charset=charset)
            if row[1] == "BASE TABLE"
        ]
    def get_view_names(self, connection, schema: Optional[str] = None, **kw):
        """Return a Unicode SHOW TABLES from a given schema."""
        # 如果未提供模式，则使用默认模式名称
        if schema is None:
            schema = self.default_schema_name
        # 将模式名称转换为字符串类型
        current_schema = cast(str, schema)
        # 获取连接的字符集
        charset = self._connection_charset
        # 执行特定驱动程序的 SQL 查询，获取所有表的信息
        rp = connection.exec_driver_sql(
            "SHOW FULL TABLES FROM %s"
            % self.identifier_preparer.quote_identifier(current_schema)
        )
        # 返回所有视图的名称，过滤掉非视图类型的表
        return [
            row[0]
            for row in self._compat_fetchall(rp, charset=charset)
            if row[1] in ("VIEW", "SYSTEM VIEW")
        ]

    def get_columns(  # type: ignore
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw,
    ) -> List[Dict[str, Any]]:  # type: ignore
        """Return information about columns in `table_name`."""
        # 如果表不存在，则抛出 NoSuchTableError 异常
        if not self.has_table(connection, table_name, schema):
            raise exc.NoSuchTableError(f"schema={schema}, table={table_name}")
        # 如果未指定模式，则使用连接的默认模式
        schema = schema or self._get_default_schema_name(connection)

        # 准备标识符的引用函数
        quote = self.identifier_preparer.quote_identifier
        # 构建完整的表名，包括模式名
        full_name = quote(table_name)
        if schema:
            full_name = "{}.{}".format(quote(schema), full_name)

        # 执行 SQL 查询以获取表的列信息
        res = connection.execute(text(f"SHOW COLUMNS FROM {full_name}"))
        columns = []
        # 解析每一列的信息并添加到列表中
        for record in res:
            column = dict(
                name=record.Field,
                type=datatype.parse_sqltype(record.Type),
                nullable=record.Null == "YES",
                default=record.Default,
            )
            columns.append(column)
        return columns

    def get_pk_constraint(
        self, connection, table_name, schema: Optional[str] = None, **kw
    ):
        """Return information about the primary key constraint."""
        # 返回空的主键约束信息字典
        return {
            "name": None,
            "constrained_columns": [],
        }

    def get_unique_constraints(  # type: ignore
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw,
    ) -> List[Dict[str, Any]]:
        """Return information about unique constraints."""
        # 返回空的唯一约束列表
        return []

    def get_check_constraints(  # type: ignore
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw,
    ) -> List[Dict[str, Any]]:
        """Return information about check constraints."""
        # 返回空的检查约束列表
        return []

    def get_foreign_keys(  # type: ignore
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw,
    ) -> List[Dict[str, Any]]:
        """Return information about foreign keys."""
        # 返回空的外键约束列表
        return []

    def get_primary_keys(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw,
    ) -> List[Dict[str, Any]]:
        """Return information about primary keys."""
        # 返回空的主键列表
        return []
    ) -> List[str]:
        """Return the primary key columns of the given table."""
        # 获取给定表的主键约束
        pk = self.get_pk_constraint(connection, table_name, schema)
        # 返回主键约束中的受限制列
        return pk.get("constrained_columns")  # type: ignore

    def get_indexes(self, connection, table_name, schema: Optional[str] = None, **kw):
        """Get table indexes about specified table."""
        # 返回一个空列表，表示没有获取到指定表的索引信息
        return []

    def has_sequence(
        self,
        connection: Connection,
        sequence_name: str,
        schema: Optional[str] = None,
        **kw,
    ) -> bool:
        """Return True if the given sequence is present in the database."""
        # 始终返回 False，表示数据库中不存在指定的序列
        return False

    def get_sequence_names(
        self, connection: Connection, schema: Optional[str] = None, **kw
    ) -> List[str]:
        """Return a list of sequence names."""
        # 返回一个空列表，表示数据库中没有序列名
        return []

    def get_temp_view_names(
        self, connection: Connection, schema: Optional[str] = None, **kw
    ) -> List[str]:
        """Return a list of temporary view names."""
        # 返回一个空列表，表示数据库中没有临时视图名
        return []

    def get_temp_table_names(
        self, connection: Connection, schema: Optional[str] = None, **kw
    ) -> List[str]:
        """Return a list of temporary table names."""
        # 返回一个空列表，表示数据库中没有临时表名
        return []

    def get_table_options(
        self, connection, table_name, schema: Optional[str] = None, **kw
    ):
        """Return a dictionary of options specified when the table was created."""
        # 返回一个空字典，表示获取不到表创建时的选项信息
        return {}

    def get_table_comment(  # type: ignore
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw,
    ) -> Dict[str, Any]:
        """Return the comment for a table."""
        # 返回一个包含 "text" 键且值为 None 的字典，表示表没有注释信息
        return dict(text=None)
```