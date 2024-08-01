# `.\DB-GPT-src\dbgpt\agent\expand\actions\tool_action.py`

```py
"""
Plugin Action Module.
"""

import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
from typing import Optional  # 导入类型提示模块

from dbgpt._private.pydantic import BaseModel, Field  # 导入 Pydantic 相关模块
from dbgpt.vis.tags.vis_plugin import Vis, VisPlugin  # 导入可视化插件相关模块

from ...core.action.base import Action, ActionOutput  # 导入基础动作相关模块
from ...core.schema import Status  # 导入状态相关模块
from ...resource.base import AgentResource, ResourceType  # 导入资源相关模块
from ...resource.tool.pack import ToolPack  # 导入工具包相关模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class ToolInput(BaseModel):
    """Plugin input model."""

    tool_name: str = Field(
        ...,
        description="The name of a tool that can be used to answer the current question"
        " or solve the current task.",
    )
    args: dict = Field(
        default={"arg name1": "", "arg name2": ""},
        description="The tool selected for the current target, the parameter "
        "information required for execution",
    )
    thought: str = Field(..., description="Summary of thoughts to the user")


class ToolAction(Action[ToolInput]):
    """Tool action class."""

    def __init__(self):
        """Create a plugin action."""
        super().__init__()
        self._render_protocol = VisPlugin()  # 创建一个可视化插件的实例

    @property
    def resource_need(self) -> Optional[ResourceType]:
        """Return the resource type needed for the action."""
        return ResourceType.Tool  # 返回所需的资源类型为工具类别

    @property
    def render_protocol(self) -> Optional[Vis]:
        """Return the render protocol."""
        return self._render_protocol  # 返回渲染协议对象

    @property
    def out_model_type(self):
        """Return the output model type."""
        return ToolInput  # 返回输出的模型类型为 ToolInput

    @property
    def ai_out_schema(self) -> Optional[str]:
        """Return the AI output schema."""
        out_put_schema = {
            "thought": "Summary of thoughts to the user",
            "tool_name": "The name of a tool that can be used to answer the current "
            "question or solve the current task.",
            "args": {
                "arg name1": "arg value1",
                "arg name2": "arg value2",
            },
        }

        return f"""Please response in the following json format:
        {json.dumps(out_put_schema, indent=2, ensure_ascii=False)}
        Make sure the response is correct json and can be parsed by Python json.loads.
        """

    async def run(
        self,
        ai_message: str,
        resource: Optional[AgentResource] = None,
        rely_action_out: Optional[ActionOutput] = None,
        need_vis_render: bool = True,
        **kwargs,
    ) -> ActionOutput:
        """Perform the plugin action.

        Args:
            ai_message (str): The AI message.
            resource (Optional[AgentResource], optional): The resource. Defaults to
                None.
            rely_action_out (Optional[ActionOutput], optional): The rely action output.
                Defaults to None.
            need_vis_render (bool, optional): Whether need visualization rendering.
                Defaults to True.
        """
        try:
            # 将 AI 消息转换为工具输入参数对象
            param: ToolInput = self._input_convert(ai_message, ToolInput)
        except Exception as e:
            # 如果转换失败，记录日志并返回执行结果对象
            logger.exception((str(e)))
            return ActionOutput(
                is_exe_success=False,
                content="The requested correctly structured answer could not be found.",
            )

        try:
            # 从资源中加载工具包
            tool_packs = ToolPack.from_resource(self.resource)
            if not tool_packs:
                # 如果未找到工具包，抛出值错误
                raise ValueError("The tool resource is not found！")
            # 获取第一个工具包
            tool_pack = tool_packs[0]
            response_success = True
            status = Status.RUNNING.value
            err_msg = None
            try:
                # 异步执行工具包中的工具
                tool_result = await tool_pack.async_execute(
                    resource_name=param.tool_name, **param.args
                )
                status = Status.COMPLETE.value
            except Exception as e:
                # 如果执行失败，记录日志，设置执行状态为失败
                response_success = False
                logger.exception(f"Tool [{param.tool_name}] execute failed!")
                status = Status.FAILED.value
                err_msg = f"Tool [{param.tool_name}] execute failed! {str(e)}"
                tool_result = err_msg

            # 构造插件参数字典
            plugin_param = {
                "name": param.tool_name,
                "args": param.args,
                "status": status,
                "logo": None,
                "result": str(tool_result),
                "err_msg": err_msg,
            }
            # 如果渲染协议未实现，抛出未实现错误
            if not self.render_protocol:
                raise NotImplementedError("The render_protocol should be implemented.")

            # 使用渲染协议显示视图
            view = await self.render_protocol.display(content=plugin_param)

            # 返回执行结果对象
            return ActionOutput(
                is_exe_success=response_success,
                content=str(tool_result),
                view=view,
                observations=str(tool_result),
            )
        except Exception as e:
            # 捕获所有异常，记录日志，并返回执行结果对象
            logger.exception("Tool Action Run Failed！")
            return ActionOutput(
                is_exe_success=False, content=f"Tool action run failed!{str(e)}"
            )
```