# `.\DB-GPT-src\dbgpt\app\scene\chat_data\chat_excel\excel_learning\verify_sql.py`

```py
import re  # 导入 re 模块，用于正则表达式操作

import sqlparse  # 导入 sqlparse 模块，用于 SQL 解析


def add_quotes(sql, column_names=[]):
    # 解析输入的 SQL 语句
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        for token in stmt.tokens:
            # 调用 deep_quotes 函数，为 SQL 中的标识符添加双引号
            deep_quotes(token, column_names)
    # 将解析后的 SQL 语句对象转换为字符串并返回
    return str(parsed[0])


def deep_quotes(token, column_names=[]):
    # 如果 token 对象有 tokens 属性，递归处理它的子 token
    if hasattr(token, "tokens"):
        for token_child in token.tokens:
            deep_quotes(token_child, column_names)
    else:
        # 如果 token 是一个标识符
        if token.ttype == sqlparse.tokens.Name:
            # 如果 column_names 非空，则只为列表中的标识符添加双引号
            if len(column_names) > 0:
                if token.value in column_names:
                    token.value = f'"{token.value}"'
            # 如果 column_names 为空，则所有标识符均添加双引号
            else:
                token.value = f'"{token.value}"'
```