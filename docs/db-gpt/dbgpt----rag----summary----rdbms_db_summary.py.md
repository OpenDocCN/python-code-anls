# `.\DB-GPT-src\dbgpt\rag\summary\rdbms_db_summary.py`

```py
"""Summary for rdbms database."""
# 导入所需的模块和类型声明
import re
from typing import TYPE_CHECKING, List, Optional

# 导入配置文件和基础连接器
from dbgpt._private.config import Config
from dbgpt.datasource import BaseConnector
from
    Examples:
        table_name(column1(column1 comment),column2(column2 comment),
        column3(column3 comment) and index keys, and table comment: {table_comment})
    """
    # 初始化一个空列表，用于存储表的列信息
    columns = []
    # 遍历数据库连接对象中指定表的所有列信息
    for column in conn.get_columns(table_name):
        # 检查每一列是否有注释，如果有则将列名和注释拼接成字符串添加到列列表中，否则只添加列名
        if column.get("comment"):
            columns.append(f"{column['name']} ({column.get('comment')})")
        else:
            columns.append(f"{column['name']}")

    # 将列列表转换成逗号分隔的字符串
    column_str = ", ".join(columns)
    
    # 获取索引信息
    index_keys = []
    raw_indexes = conn.get_indexes(table_name)
    # 遍历原始索引数据
    for index in raw_indexes:
        # 如果索引是元组类型，处理元组类型的索引信息
        if isinstance(index, tuple):
            index_name, index_creation_command = index
            # 使用正则表达式提取索引创建命令中的列名
            matched_columns = re.findall(r"\(([^)]+)\)", index_creation_command)
            if matched_columns:
                key_str = ", ".join(matched_columns)
                index_keys.append(f"{index_name}(`{key_str}`) ")
        else:
            # 处理字典类型的索引信息，获取其中的列名
            key_str = ", ".join(index["column_names"])
            index_keys.append(f"{index['name']}(`{key_str}`) ")

    # 使用给定的模板格式化表的摘要信息，包括表名和列信息
    table_str = summary_template.format(table_name=table_name, columns=column_str)
    
    # 如果存在索引信息，则将索引信息添加到表的摘要信息中
    if len(index_keys) > 0:
        index_key_str = ", ".join(index_keys)
        table_str += f", and index keys: {index_key_str}"
    
    # 尝试获取表的注释信息，如果获取失败则设置为空字典
    try:
        comment = conn.get_table_comment(table_name)
    except Exception:
        comment = dict(text=None)
    
    # 如果表有注释信息，则将注释信息添加到表的摘要信息中
    if comment.get("text"):
        table_str += f", and table comment: {comment.get('text')}"
    
    # 返回格式化后的表的摘要信息字符串
    return table_str
```