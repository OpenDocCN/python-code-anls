# `.\DB-GPT-src\dbgpt\vis\tags\vis_dashboard.py`

```py
"""Protocol for the dashboard vis."""
# 导入必要的模块：json用于处理JSON数据，logging用于日志记录
import json
import logging
# 从typing模块导入必要的类型提示
from typing import Any, Dict, Optional

# 从父模块中导入Vis类
from ..base import Vis

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class VisDashboard(Vis):
    """Dashboard Vis Protocol."""

    async def generate_param(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate the parameters required by the vis protocol."""
        # 从kwargs中获取charts和title参数，如果不存在charts则抛出异常
        charts = kwargs.get("charts", None)
        title = kwargs.get("title", None)
        if not charts:
            # 如果charts为空，则抛出值错误异常，并指明协议无法转换的原因
            raise ValueError(
                f"Parameter information is missing and {self.vis_tag} protocol "
                "conversion cannot be performed."
            )

        # 初始化存储图表参数的列表
        chart_items = []
        # 遍历charts列表中的每一个图表
        for chart in charts:
            param = {}
            # 获取图表的SQL查询语句
            sql = chart.get("sql", "")
            param["sql"] = sql
            # 获取图表的展示类型，默认为响应表格
            param["type"] = chart.get("display_type", "response_table")
            # 获取图表的标题
            param["title"] = chart.get("title", "")
            # 获取图表的描述信息
            param["describe"] = chart.get("thought", "")
            try:
                # 尝试获取图表的数据帧（DataFrame）
                df = chart.get("data", None)
                err_msg = chart.get("err_msg", None)
                if df is None:
                    # 如果数据帧为空，则将错误信息存入参数中
                    param["err_msg"] = err_msg
                else:
                    # 如果数据帧不为空，则将数据转换为JSON格式并存入参数中
                    param["data"] = json.loads(
                        df.to_json(orient="records", date_format="iso", date_unit="s")
                    )
            except Exception as e:
                # 捕获任何异常，并记录异常日志
                logger.exception("dashboard chart build faild！")
                # 设置数据为空列表，错误信息为异常的字符串表示
                param["data"] = []
                param["err_msg"] = str(e)
            # 将当前图表参数添加到图表参数列表中
            chart_items.append(param)

        # 构建整体仪表盘参数字典
        dashboard_param = {
            "data": chart_items,
            "chart_count": len(chart_items),
            "title": title,
            "display_strategy": "default",
            "style": "default",
        }

        # 返回构建好的仪表盘参数字典
        return dashboard_param

    @classmethod
    def vis_tag(cls) -> str:
        """Return the tag name of the vis protocol."""
        # 返回当前协议的标签名字，固定为"vis-dashboard"
        return "vis-dashboard"
```