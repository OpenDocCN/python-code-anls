# `.\DB-GPT-src\dbgpt\app\scene\chat_dashboard\data_preparation\report_schma.py`

```py
# 引入必要的模块和类型声明
from typing import Any, List, Optional

# 导入BaseModel类，用于定义数据模型
from dbgpt._private.pydantic import BaseModel

# 定义ValueItem类，继承自BaseModel，表示数据项
class ValueItem(BaseModel):
    # 数据项的名称，必须是字符串类型
    name: str
    # 数据项的类型，可选字段，默认为None
    type: str = None
    # 数据项的数值，必须是浮点数
    value: float

    # 将数据项转换为字典形式
    def dict(self, *args, **kwargs):
        return {"name": self.name, "type": self.type, "value": self.value}

# 定义ChartData类，继承自BaseModel，表示图表数据
class ChartData(BaseModel):
    # 图表的唯一标识符，必须是字符串类型
    chart_uid: str
    # 图表的名称，必须是字符串类型
    chart_name: str
    # 图表的类型，必须是字符串类型
    chart_type: str
    # 图表的描述，必须是字符串类型
    chart_desc: str
    # 图表对应的SQL查询语句，必须是字符串类型
    chart_sql: str
    # 图表的列名列表，每个元素必须是字符串类型
    column_name: List
    # 图表的数据项列表，每个元素必须是ValueItem类型
    values: List[ValueItem]
    # 图表的样式，类型为任意类型，默认为None
    style: Any = None

    # 将图表数据转换为字典形式
    def dict(self, *args, **kwargs):
        return {
            "chart_uid": self.chart_uid,
            "chart_name": self.chart_name,
            "chart_type": self.chart_type,
            "chart_desc": self.chart_desc,
            "chart_sql": self.chart_sql,
            "column_name": [str(item) for item in self.column_name],
            "values": [value.dict() for value in self.values],
            "style": self.style,
        }

# 定义ReportData类，继承自BaseModel，表示报告数据
class ReportData(BaseModel):
    # 报告的唯一标识符，必须是字符串类型
    conv_uid: str
    # 报告的模板名称，必须是字符串类型
    template_name: str
    # 报告模板的介绍，可选字段，默认为None
    template_introduce: Optional[str] = None
    # 报告中包含的图表数据列表，每个元素必须是ChartData类型
    charts: List[ChartData]

    # 准备报告数据并转换为字典形式
    def prepare_dict(self):
        return {
            "conv_uid": self.conv_uid,
            "template_name": self.template_name,
            "template_introduce": self.template_introduce,
            "charts": [chart.dict() for chart in self.charts],
        }
```