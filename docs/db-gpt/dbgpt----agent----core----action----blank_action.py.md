# `.\DB-GPT-src\dbgpt\agent\core\action\blank_action.py`

```py
"""Blank Action for the Agent."""

# 导入日志模块
import logging
# 导入类型提示模块中的 Optional 类型
from typing import Optional
# 导入 AgentResource 类型
from ...resource.base import AgentResource
# 导入 Action 和 ActionOutput 类
from .base import Action, ActionOutput

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class BlankAction(Action):
    """Blank action class."""

    def __init__(self):
        """Create a blank action."""
        super().__init__()

    @property
    def ai_out_schema(self) -> Optional[str]:
        """Return the AI output schema."""
        # 返回 AI 输出的模式，这里为可选的字符串类型
        return None

    async def run(
        self,
        ai_message: str,
        resource: Optional[AgentResource] = None,
        rely_action_out: Optional[ActionOutput] = None,
        need_vis_render: bool = True,
        **kwargs,
    ) -> ActionOutput:
        """Perform the action.

        Just return the AI message.
        """
        # 执行动作，返回 AI 消息作为内容和视图
        return ActionOutput(
            is_exe_success=True,
            content=ai_message,
            view=ai_message,
        )
```