# `.\DB-GPT-src\dbgpt\rag\summary\gdbms_db_summary.py`

```py
"""Summary for rdbms database."""

# 引入必要的类型和模块
from typing import TYPE_CHECKING, Dict, List, Optional

# 从 dbgpt._private.config 模块中导入 Config 类
from dbgpt._private.config import Config

# 从 dbgpt.datasource 中导入 BaseConnector 类
from dbgpt.datasource import BaseConnector
# 从 dbgpt.datasource.conn_tugraph 中导入 TuGraphConnector 类
from dbgpt.datasource.conn_tugraph import TuGraphConnector
# 从 dbgpt.rag.summary.db_summary 中导入 DBSummary 类
from dbgpt.rag.summary.db_summary import DBSummary

# 如果 TYPE_CHECKING 为真，则从 dbgpt.datasource.manages 中导入 ConnectorManager 类
if TYPE_CHECKING:
    from dbgpt.datasource.manages import ConnectorManager

# 创建 Config 的实例 CFG
CFG = Config()

# 定义 GdbmsSummary 类，继承自 DBSummary 类
class GdbmsSummary(DBSummary):
    """Get graph db table summary template."""

    def __init__(
        self, name: str, type: str, manager: Optional["ConnectorManager"] = None
    ):
        """Create a new RdbmsSummary."""
        # 初始化对象属性 name 和 type
        self.name = name
        self.type = type
        # 定义表格摘要模板字符串
        self.summary_template = "{table_name}({columns})"
        # 初始化 tables 属性为空字典
        self.tables = {}

        # TODO: 不要使用全局变量。
        # 初始化 db_manager 变量，如果 manager 不为 None 则使用 manager，否则使用 CFG.local_db_manager
        db_manager = manager or CFG.local_db_manager
        # 如果 db_manager 仍为 None，则抛出 ValueError 异常
        if not db_manager:
            raise ValueError("Local db manage is not initialized.")
        # 使用 db_manager 获取与 name 相关的数据库连接器
        self.db = db_manager.get_connector(name)

        # 构建 metadata 属性，包含用户信息、授权信息、字符集和排序规则等信息
        self.metadata = """user info :{users}, grant info:{grant}, charset:{charset},
        collation:{collation}""".format(
            users=self.db.get_users(),
            grant=self.db.get_grants(),
            charset=self.db.get_charset(),
            collation=self.db.get_collation(),
        )

        # 获取数据库中的表名列表
        tables = self.db.get_table_names()
        # 构建 table_info_summaries 属性，包含顶点表和边表的摘要信息
        self.table_info_summaries = {
            "vertex_tables": [
                self.get_table_summary(table_name, "vertex")
                for table_name in tables["vertex_tables"]
            ],
            "edge_tables": [
                self.get_table_summary(table_name, "edge")
                for table_name in tables["edge_tables"]
            ],
        }

    def get_table_summary(self, table_name, table_type):
        """Get table summary for table.

        example:
            table_name(column1(column1 comment),column2(column2 comment),
            column3(column3 comment) and index keys, and table comment: {table_comment})
        """
        # 调用 _parse_table_summary 函数获取表格摘要信息
        return _parse_table_summary(
            self.db, self.summary_template, table_name, table_type
        )

    def table_summaries(self):
        """Get table summaries."""
        # 返回 table_info_summaries 属性，包含顶点表和边表的摘要信息
        return self.table_info_summaries


def _parse_db_summary(
    conn: BaseConnector, summary_template: str = "{table_name}({columns})"
) -> List[str]:
    """Get db summary for database."""
    # 此函数用于获取数据库的摘要信息，返回一个字符串列表
    table_info_summaries = None
    # 检查 conn 是否为 TuGraphConnector 类型的实例
    if isinstance(conn, TuGraphConnector):
        # 获取连接器 conn 中的表名字典
        table_names = conn.get_table_names()
        # 从表名字典中获取顶点表和边表的列表，如果没有则为空列表
        v_tables = table_names.get("vertex_tables", [])
        e_tables = table_names.get("edge_tables", [])
        # 使用列表推导式遍历顶点表和边表，分别解析表的摘要信息，形成摘要信息的列表
        table_info_summaries = [
            _parse_table_summary(conn, summary_template, table_name, "vertex")
            for table_name in v_tables
        ] + [
            _parse_table_summary(conn, summary_template, table_name, "edge")
            for table_name in e_tables
        ]
    else:
        # 如果 conn 不是 TuGraphConnector 类型的实例，则将 table_info_summaries 设置为空列表
        table_info_summaries = []

    # 返回解析后的表摘要信息列表
    return table_info_summaries
def _format_column(column: Dict) -> str:
    """格式化单个列的摘要。"""
    # 获取列的注释，如果没有则为空字符串
    comment = column.get("comment", "")
    # 如果列在主键中，则添加到注释中
    if column.get("is_in_primary_key"):
        comment += " Primary Key" if comment else "Primary Key"
    # 返回格式化后的列摘要，包括列名和注释（如果有的话）
    return f"{column['name']} ({comment})" if comment else column["name"]


def _format_indexes(indexes: List[Dict]) -> str:
    """格式化表摘要中的索引键。"""
    # 使用逗号连接每个索引的格式化字符串
    return ", ".join(
        f"{index['name']}(`{', '.join(index['column_names'])}`)" for index in indexes
    )


def _parse_table_summary(
    conn: TuGraphConnector, summary_template: str, table_name: str, table_type: str
) -> str:
    """增强型表摘要函数。"""
    # 获取表的所有列，并格式化每一列
    columns = [
        _format_column(column) for column in conn.get_columns(table_name, table_type)
    ]
    # 将格式化后的列信息连接成字符串
    column_str = ", ".join(columns)

    # 获取表的索引信息
    indexes = conn.get_indexes(table_name, table_type)
    # 格式化索引信息，如果有索引的话
    index_str = _format_indexes(indexes) if indexes else ""

    # 使用摘要模板格式化表名和列信息
    table_str = summary_template.format(table_name=table_name, columns=column_str)
    # 如果有索引信息，则添加到表摘要中
    if index_str:
        table_str += f", and index keys: {index_str}"

    # 尝试获取表的注释信息，如果失败则设为默认字典
    try:
        comment = conn.get_table_comment(table_name)
    except Exception:
        comment = dict(text=None)
    
    # 如果存在表注释，则添加到表摘要中；否则添加默认的类型信息
    if comment.get("text"):
        table_str += (
            f", and table comment: {comment.get('text')}, this is a {table_type} table"
        )
    else:
        table_str += f", and table comment: this is a {table_type} table"

    # 返回最终的表摘要字符串
    return table_str
```