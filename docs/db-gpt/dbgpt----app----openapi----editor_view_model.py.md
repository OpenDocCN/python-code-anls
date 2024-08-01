# `.\DB-GPT-src\dbgpt\app\openapi\editor_view_model.py`

```py
# 导入必要的模块和类
from typing import Any, List  # 引入类型提示需要的模块

from dbgpt._private.pydantic import BaseModel, Field  # 引入 Pydantic 模块中的 BaseModel 和 Field 类


# 数据库字段定义类，继承自 Pydantic 的 BaseModel 类
class DbField(BaseModel):
    colunm_name: str  # 列名，字符串类型
    type: str  # 类型，字符串类型
    colunm_len: str  # 列长度，字符串类型
    can_null: bool = True  # 是否可为空，默认为 True
    default_value: str = ""  # 默认值，默认为空字符串
    comment: str = ""  # 注释，默认为空字符串


# 数据库表定义类，继承自 Pydantic 的 BaseModel 类
class DbTable(BaseModel):
    table_name: str  # 表名，字符串类型
    comment: str  # 表的注释，字符串类型
    colunm: List[DbField]  # 列的列表，每个元素为 DbField 类的实例


# 聊天数据库轮次定义类，继承自 Pydantic 的 BaseModel 类
class ChatDbRounds(BaseModel):
    round: int  # 轮次，整数类型
    db_name: str  # 数据库名称，字符串类型
    round_name: str  # 轮次名称，字符串类型


# 图表列表定义类，继承自 Pydantic 的 BaseModel 类
class ChartList(BaseModel):
    round: int  # 轮次，整数类型
    db_name: str  # 数据库名称，字符串类型
    charts: List  # 图表列表，每个元素类型不限


# 图表详情定义类，继承自 Pydantic 的 BaseModel 类
class ChartDetail(BaseModel):
    chart_uid: str  # 图表唯一标识，字符串类型
    chart_type: str  # 图表类型，字符串类型
    chart_desc: str  # 图表描述，字符串类型
    chart_sql: str  # 图表 SQL 查询语句，字符串类型
    db_name: str  # 数据库名称，字符串类型
    chart_name: str  # 图表名称，字符串类型
    chart_value: Any  # 图表值，任意类型
    table_value: Any  # 表值，任意类型


# 聊天图表编辑上下文定义类，继承自 Pydantic 的 BaseModel 类
class ChatChartEditContext(BaseModel):
    conv_uid: str  # 对话唯一标识，字符串类型
    chart_title: str  # 图表标题，字符串类型
    db_name: str  # 数据库名称，字符串类型
    old_sql: str  # 原始 SQL 查询语句，字符串类型

    new_chart_type: str  # 新图表类型，字符串类型
    new_sql: str  # 新 SQL 查询语句，字符串类型
    new_comment: str  # 新注释，字符串类型
    gmt_create: int  # 创建时间戳，整数类型


# 聊天 SQL 编辑上下文定义类，继承自 Pydantic 的 BaseModel 类
class ChatSqlEditContext(BaseModel):
    conv_uid: str  # 对话唯一标识，字符串类型
    db_name: str  # 数据库名称，字符串类型
    conv_round: int  # 对话轮次，整数类型

    old_sql: str  # 原始 SQL 查询语句，字符串类型
    old_speak: str  # 原始对话内容，字符串类型
    gmt_create: int = 0  # 创建时间戳，默认为 0

    new_sql: str  # 新 SQL 查询语句，字符串类型
    new_speak: str = ""  # 新对话内容，默认为空字符串
```