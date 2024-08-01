# `.\DB-GPT-src\dbgpt\app\openapi\api_v1\editor\sql_editor.py`

```py
# 导入必要的模块和类型
from typing import Any, List, Optional

# 导入基础数据模型BaseModel
from dbgpt._private.pydantic import BaseModel

# 从特定模块导入ValueItem类
from dbgpt.app.scene.chat_dashboard.data_preparation.report_schma import ValueItem

# 定义数据节点的数据结构，继承自BaseModel
class DataNode(BaseModel):
    # 节点标题，必须是字符串类型
    title: str
    # 节点键，必须是字符串类型
    key: str

    # 节点类型，默认为空字符串
    type: str = ""
    # 默认值，可以是任意类型或者None
    default_value: Optional[Any] = None
    # 是否可为空，字符串类型，默认为"YES"
    can_null: str = "YES"
    # 注释信息，可选的字符串，可以为None
    comment: Optional[str] = None
    # 子节点列表，默认为空列表
    children: List = []


# 定义SQL运行数据的数据结构，继承自BaseModel
class SqlRunData(BaseModel):
    # SQL执行结果信息，必须是字符串类型
    result_info: str
    # SQL运行耗时，必须是整数类型
    run_cost: int
    # 列名列表，每个元素必须是字符串类型的列表
    colunms: List[str]
    # 值列表，可以包含任意类型的值的列表
    values: List


# 定义图表运行数据的数据结构，继承自BaseModel
class ChartRunData(BaseModel):
    # SQL运行数据，类型为SqlRunData
    sql_data: SqlRunData
    # 图表数值数据，类型为ValueItem的列表
    chart_values: List[ValueItem]
    # 图表类型，必须是字符串类型
    chart_type: str
```