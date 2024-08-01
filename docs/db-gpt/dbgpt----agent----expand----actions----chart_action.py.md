# `.\DB-GPT-src\dbgpt\agent\expand\actions\chart_action.py`

```py
"""Chart Action for SQL execution and rendering."""

# 导入必要的库和模块
import json
import logging
from typing import List, Optional

# 导入基于 Pydantic 的数据模型相关模块和类
from dbgpt._private.pydantic import BaseModel, Field, model_to_json

# 导入可视化相关模块和类
from dbgpt.vis.tags.vis_chart import Vis, VisChart

# 导入基础操作类和输出类
from ...core.action.base import Action, ActionOutput

# 导入资源相关类和类型
from ...resource.base import AgentResource, ResourceType
from ...resource.database import DBResource

# 设置日志记录器
logger = logging.getLogger(__name__)


class SqlInput(BaseModel):
    """SQL input model."""
    
    # SQL 图表渲染方法的选择，如果不确定输出什么，请统一输出 'response_table'
    display_type: str = Field(
        ...,
        description="The chart rendering method selected for SQL. If you don’t know "
        "what to output, just output 'response_table' uniformly.",
    )
    
    # 为当前目标/问题生成的可执行 SQL
    sql: str = Field(
        ..., description="Executable sql generated for the current target/problem"
    )
    
    # 向用户概述的思考摘要
    thought: str = Field(..., description="Summary of thoughts to the user")


class ChartAction(Action[SqlInput]):
    """Chart action class."""

    def __init__(self):
        """Create a chart action."""
        super().__init__()
        # 初始化图表渲染协议
        self._render_protocol = VisChart()

    @property
    def resource_need(self) -> Optional[ResourceType]:
        """Return the resource type needed for the action."""
        return ResourceType.DB

    @property
    def render_protocol(self) -> Optional[Vis]:
        """Return the render protocol."""
        return self._render_protocol

    @property
    def out_model_type(self):
        """Return the output model type."""
        return SqlInput

    async def run(
        self,
        ai_message: str,
        resource: Optional[AgentResource] = None,
        rely_action_out: Optional[ActionOutput] = None,
        need_vis_render: bool = True,
        **kwargs,
    ):
        # 此处应包含该方法的具体实现，但在注释内不应省略任何代码
    ) -> ActionOutput:
        """定义一个方法，指定输入和输出类型"""
        try:
            # 尝试将 AI 消息转换为 SqlInput 对象
            param: SqlInput = self._input_convert(ai_message, SqlInput)
        except Exception as e:
            # 如果出现异常，记录异常信息并返回错误输出
            logger.exception(f"{str(e)}! \n {ai_message}")
            return ActionOutput(
                is_exe_success=False,
                content="The requested correctly structured answer could not be found.",
            )
        try:
            # 检查资源需求是否存在
            if not self.resource_need:
                raise ValueError("The resource type is not found！")

            # 检查渲染协议是否初始化
            if not self.render_protocol:
                raise ValueError("The rendering protocol is not initialized！")

            # 从资源中获取数据库资源列表
            db_resources: List[DBResource] = DBResource.from_resource(self.resource)
            if not db_resources:
                raise ValueError("The database resource is not found！")

            # 获取第一个数据库资源
            db = db_resources[0]
            # 使用数据库执行 SQL 查询，获取数据框
            data_df = await db.query_to_df(param.sql)
            # 使用渲染协议展示数据
            view = await self.render_protocol.display(
                chart=json.loads(model_to_json(param)), data_df=data_df
            )

            # 返回成功执行的输出
            return ActionOutput(
                is_exe_success=True,
                content=model_to_json(param),
                view=view,
                resource_type=self.resource_need.value,
                resource_value=db._db_name,
            )
        except Exception as e:
            # 如果出现异常，记录异常信息并返回错误输出
            logger.exception("Check your answers, the sql run failed！")
            return ActionOutput(
                is_exe_success=False,
                content=f"Check your answers, the sql run failed!Reason:{str(e)}",
            )
```