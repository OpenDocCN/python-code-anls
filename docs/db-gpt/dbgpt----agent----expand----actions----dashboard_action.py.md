# `.\DB-GPT-src\dbgpt\agent\expand\actions\dashboard_action.py`

```py
"""Dashboard Action Module."""

import json  # 导入json模块，用于处理JSON数据
import logging  # 导入logging模块，用于日志记录
from typing import List, Optional  # 导入类型提示模块，定义函数参数和返回类型

from dbgpt._private.pydantic import BaseModel, Field, model_to_dict  # 导入pydantic模块，用于数据验证和转换
from dbgpt.vis.tags.vis_dashboard import Vis, VisDashboard  # 导入可视化相关模块

from ...core.action.base import Action, ActionOutput  # 导入基础动作类和动作输出类
from ...resource.base import AgentResource, ResourceType  # 导入资源相关模块
from ...resource.database import DBResource  # 导入数据库资源模块

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器对象


class ChartItem(BaseModel):
    """Chart item model."""

    title: str = Field(
        ...,
        description="The title of the current analysis chart.",  # 定义当前分析图表的标题
    )
    display_type: str = Field(
        ...,
        description="The chart rendering method selected for SQL. If you don’t know "
        "what to output, just output 'response_table' uniformly.",  # 定义SQL的图表渲染方法
    )
    sql: str = Field(
        ..., description="Executable sql generated for the current target/problem"  # 定义生成的可执行SQL语句
    )
    thought: str = Field(
        ..., description="Summary of thoughts to the user"  # 用户的思考摘要
    )

    def to_dict(self):
        """Convert to dict."""
        return model_to_dict(self)  # 将对象转换为字典形式


class DashboardAction(Action[List[ChartItem]]):
    """Dashboard action class."""

    def __init__(self):
        """Create a dashboard action."""
        super().__init__()  # 调用父类的初始化方法
        self._render_protocol = VisDashboard()  # 创建一个可视化仪表盘对象

    @property
    def resource_need(self) -> Optional[ResourceType]:
        """Return the resource type needed for the action."""
        return ResourceType.DB  # 返回动作需要的资源类型为数据库类型

    @property
    def render_protocol(self) -> Optional[Vis]:
        """Return the render protocol."""
        return self._render_protocol  # 返回当前的渲染协议对象

    @property
    def out_model_type(self):
        """Return the output model type."""
        return List[ChartItem]  # 返回输出的数据模型类型为ChartItem列表类型

    async def run(
        self,
        ai_message: str,
        resource: Optional[AgentResource] = None,
        rely_action_out: Optional[ActionOutput] = None,
        need_vis_render: bool = True,
        **kwargs,
    ):
        """Run the dashboard action asynchronously."""
        # 执行仪表盘动作的异步方法
        pass  # 空操作，暂未实现具体逻辑
    ) -> ActionOutput:
        """定义函数的输入和输出类型，指定为ActionOutput类型。"""
        """Perform the action."""
        try:
            # 尝试将AI消息转换为ChartItem类型的列表
            input_param = self._input_convert(ai_message, List[ChartItem])
        except Exception as e:
            # 如果转换失败，记录异常并返回错误消息
            logger.exception(str(e))
            return ActionOutput(
                is_exe_success=False,
                content="The requested correctly structured answer could not be found.",
            )
        if not isinstance(input_param, list):
            # 如果输入参数不是列表，返回错误消息
            return ActionOutput(
                is_exe_success=False,
                content="The requested correctly structured answer could not be found.",
            )
        # 将input_param强制转换为ChartItem类型的列表
        chart_items: List[ChartItem] = input_param
        try:
            # 从资源中获取数据库资源列表
            db_resources: List[DBResource] = DBResource.from_resource(self.resource)
            if not db_resources:
                # 如果数据库资源列表为空，抛出数值错误异常
                raise ValueError("The database resource is not found！")

            # 选择第一个数据库资源
            db = db_resources[0]

            if not db:
                # 如果数据库资源为空，抛出数值错误异常
                raise ValueError("The database resource is not found！")

            # 初始化空的图表参数列表
            chart_params = []
            for chart_item in chart_items:
                chart_dict = {}
                try:
                    # 使用数据库对象db执行chart_item的SQL查询，并将结果转换为DataFrame
                    sql_df = await db.query_to_df(chart_item.sql)
                    # 将chart_item转换为字典
                    chart_dict = chart_item.to_dict()

                    # 将查询结果存入chart_dict的"data"字段
                    chart_dict["data"] = sql_df
                except Exception as e:
                    # 如果SQL执行失败，记录警告并将异常信息存入chart_dict的"err_msg"字段
                    logger.warning(f"Sql execute failed！{str(e)}")
                    chart_dict["err_msg"] = str(e)
                # 将chart_dict添加到chart_params列表中
                chart_params.append(chart_dict)
            if not self.render_protocol:
                # 如果渲染协议未初始化，抛出数值错误异常
                raise ValueError("The render protocol is not initialized!")
            # 使用render_protocol显示图表参数并获取视图
            view = await self.render_protocol.display(charts=chart_params)
            return ActionOutput(
                is_exe_success=True,
                content=json.dumps(
                    [chart_item.to_dict() for chart_item in chart_items]
                ),
                view=view,
            )
        except Exception as e:
            # 如果任何异常被抛出，记录异常信息并返回错误消息
            logger.exception("Dashboard generate Failed！")
            return ActionOutput(
                is_exe_success=False, content=f"Dashboard action run failed!{str(e)}"
            )
```