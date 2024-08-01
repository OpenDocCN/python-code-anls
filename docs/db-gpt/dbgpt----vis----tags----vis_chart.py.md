# `.\DB-GPT-src\dbgpt\vis\tags\vis_chart.py`

```py
"""Chart visualization protocol conversion class."""
# 导入所需模块
import json
from typing import Any, Dict, Optional

# 从base模块中导入Vis类
from ..base import Vis

# 定义函数，返回默认图表类型的提示信息
def default_chart_type_prompt() -> str:
    """Return prompt information for the default chart type.

    This function is moved from excel_analyze/chat.py,and used by subclass.

    Returns:
        str: prompt information for the default chart type.
    """
    # antv_charts定义了一组图表类型及其描述信息
    antv_charts = [
        {"response_line_chart": "used to display comparative trend analysis data"},
        {
            "response_pie_chart": "suitable for scenarios such as proportion and "
            "distribution statistics"
        },
        {
            "response_table": "suitable for display with many display columns or "
            "non-numeric columns"
        },
        {
            "response_scatter_plot": "Suitable for exploring relationships between "
            "variables, detecting outliers, etc."
        },
        {
            "response_bubble_chart": "Suitable for relationships between multiple "
            "variables, highlighting outliers or special situations, etc."
        },
        {
            "response_donut_chart": "Suitable for hierarchical structure representation"
            ", category proportion display and highlighting key categories, etc."
        },
        {
            "response_area_chart": "Suitable for visualization of time series data, "
            "comparison of multiple groups of data, analysis of data change trends, "
            "etc."
        },
        {
            "response_heatmap": "Suitable for visual analysis of time series data, "
            "large-scale data sets, distribution of classified data, etc."
        },
    ]
    # 将图表类型及其描述信息格式化成字符串并返回
    return "\n".join(
        f"{key}:{value}"
        for dict_item in antv_charts
        for key, value in dict_item.items()
    )

# 定义图表可视化类，继承自Vis类
class VisChart(Vis):
    """Chart visualization protocol conversion class."""

    # 渲染提示信息的方法
    def render_prompt(self) -> Optional[str]:
        """Return the prompt for the vis protocol."""
        return default_chart_type_prompt()

    # 生成图表所需参数的异步方法
    async def generate_param(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate the parameters required by the vis protocol."""
        # 获取参数中的图表类型和数据DataFrame
        chart = kwargs.get("chart", None)
        data_df = kwargs.get("data_df", None)

        # 如果未提供图表类型，则抛出数值错误异常
        if not chart:
            raise ValueError(
                f"Parameter information is missing and {self.vis_tag} protocol "
                "conversion cannot be performed."
            )

        # 从图表参数中获取SQL语句
        sql = chart.get("sql", None)
        param = {}

        # 如果未提供SQL语句或SQL长度为0，则返回空
        if not sql or len(sql) <= 0:
            return None

        # 设置参数字典的各项值
        param["sql"] = sql
        param["type"] = chart.get("display_type", "response_table")
        param["title"] = chart.get("title", "")
        param["describe"] = chart.get("thought", "")

        # 将数据DataFrame转换为JSON格式，并添加到参数字典中
        param["data"] = json.loads(
            data_df.to_json(orient="records", date_format="iso", date_unit="s")
        )
        return param

    @classmethod
    # 定义一个方法 vis_tag，返回可视化协议的标签名称
    def vis_tag(cls) -> str:
        """Return the tag name of the vis protocol."""
        # 直接返回字符串 "vis-chart"
        return "vis-chart"
```