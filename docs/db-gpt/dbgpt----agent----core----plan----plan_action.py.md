# `.\DB-GPT-src\dbgpt\agent\core\plan\plan_action.py`

```py
"""Plan Action."""

# 导入所需的模块
import logging
from typing import List, Optional

from dbgpt._private.pydantic import BaseModel, Field
from dbgpt.vis.tags.vis_agent_plans import Vis, VisAgentPlans

from ...resource.base import AgentResource
from ..action.base import Action, ActionOutput
from ..agent import AgentContext
from ..memory.gpts.base import GptsPlan
from ..memory.gpts.gpts_memory import GptsPlansMemory
from ..schema import Status

# 设置日志记录器
logger = logging.getLogger(__name__)


class PlanInput(BaseModel):
    """Plan input model."""

    # 定义计划输入模型的字段
    serial_number: int = Field(
        0,
        description="Number of sub-tasks",
    )
    agent: str = Field(..., description="The agent name to complete current task")
    content: str = Field(
        ...,
        description="The task content of current step, make sure it can by executed by"
        " agent",
    )
    rely: str = Field(
        ...,
        description="The rely task number(serial_number), e.g. 1,2,3, empty if no rely",
    )


class PlanAction(Action[List[PlanInput]]):
    """Plan action class."""

    def __init__(self, **kwargs):
        """Create a plan action."""
        # 初始化计划动作类
        super().__init__()
        self._render_protocol = VisAgentPlans()

    @property
    def render_protocol(self) -> Optional[Vis]:
        """Return the render protocol."""
        # 返回渲染协议
        return self._render_protocol

    @property
    def out_model_type(self):
        """Output model type."""
        # 返回输出模型类型
        return List[PlanInput]

    async def run(
        self,
        ai_message: str,
        resource: Optional[AgentResource] = None,
        rely_action_out: Optional[ActionOutput] = None,
        need_vis_render: bool = True,
        **kwargs,
```