# `.\DB-GPT-src\dbgpt\agent\expand\actions\indicator_action.py`

```py
"""Indicator Action."""

import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
from typing import Optional  # 导入类型提示模块

from dbgpt._private.pydantic import BaseModel, Field  # 导入 Pydantic 的基础模型和字段
from dbgpt.vis.tags.vis_plugin import Vis, VisPlugin  # 导入可视化插件相关模块

from ...core.action.base import Action, ActionOutput  # 导入基础操作和操作输出相关模块
from ...core.schema import Status  # 导入状态相关模块
from ...resource.base import AgentResource, ResourceType  # 导入代理资源和资源类型相关模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class IndicatorInput(BaseModel):
    """Indicator input model."""
    
    # 指标输入模型，定义了指标操作的输入参数
    indicator_name: str = Field(
        ...,
        description="The name of a indicator.",
    )
    api: str = Field(
        ...,
        description="The api of a indicator.",
    )
    method: str = Field(
        ...,
        description="The api of a indicator request method.",
    )
    args: dict = Field(
        default={"arg name1": "", "arg name2": ""},
        description="The tool selected for the current target, the parameter "
        "information required for execution",
    )
    thought: str = Field(..., description="Summary of thoughts to the user")


class IndicatorAction(Action[IndicatorInput]):
    """Indicator action class."""

    # 指标操作类，继承自通用操作类 Action，处理指标相关的业务逻辑
    
    def __init__(self):
        """Create a indicator action."""
        super().__init__()
        self._render_protocol = VisPlugin()  # 初始化可视化插件对象

    @property
    def resource_need(self) -> Optional[ResourceType]:
        """Return the resource type needed for the action."""
        return ResourceType.Knowledge  # 返回操作需要的资源类型为 Knowledge（知识）

    @property
    def render_protocol(self) -> Optional[Vis]:
        """Return the render protocol."""
        return self._render_protocol  # 返回可视化渲染协议对象

    @property
    def out_model_type(self):
        """Return the output model type."""
        return IndicatorInput  # 返回操作的输出模型类型为 IndicatorInput

    @property
    def ai_out_schema(self) -> Optional[str]:
        """Return the AI output schema."""
        out_put_schema = {
            "indicator_name": "The name of a tool that can be used to answer the "
            "current question or solve the current task.",
            "api": "",
            "method": "",
            "args": {
                "arg name1": "Parameters in api definition",
                "arg name2": "Parameters in api definition",
            },
            "thought": "Summary of thoughts to the user",
        }
        
        # 返回 AI 输出的 JSON 模式说明，要求以 JSON 格式返回，可以被 Python 的 json.loads 解析
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
    ):
        # 运行方法，处理指标操作的具体逻辑
    ) -> ActionOutput:
        """定义函数的返回类型为 ActionOutput。"""
        import requests  # 导入 requests 库，用于发送 HTTP 请求
        from requests.exceptions import HTTPError  # 导入 HTTPError 异常类

        try:
            input_param = self._input_convert(ai_message, IndicatorInput)  # 调用 _input_convert 方法转换参数
        except Exception as e:
            logger.exception((str(e)))  # 记录异常日志
            return ActionOutput(
                is_exe_success=False,
                content="The requested correctly structured answer could not be found.",
            )  # 返回执行结果为失败的 ActionOutput 对象

        if isinstance(input_param, list):
            return ActionOutput(
                is_exe_success=False,
                content="The requested correctly structured answer could not be found.",
            )  # 如果输入参数是列表类型，返回执行结果为失败的 ActionOutput 对象

        param: IndicatorInput = input_param  # 将输入参数强制转换为 IndicatorInput 类型
        response_success = True  # 假设执行成功
        result: Optional[str] = None  # 初始化结果变量为 None

        try:
            status = Status.COMPLETE.value  # 设置状态为 COMPLETE
            err_msg = None  # 清空错误消息

            try:
                status = Status.RUNNING.value  # 更新状态为 RUNNING
                if param.method.lower() == "get":
                    response = requests.get(param.api, params=param.args)  # 发送 GET 请求
                elif param.method.lower() == "post":
                    response = requests.post(param.api, data=param.args)  # 发送 POST 请求
                else:
                    response = requests.request(
                        param.method.lower(), param.api, data=param.args
                    )  # 发送其他类型的请求

                response.raise_for_status()  # 检查请求是否成功，否则抛出 HTTPError 异常
                result = response.text  # 获取响应内容
            except HTTPError as http_err:
                response_success = False  # 请求失败
                print(f"HTTP error occurred: {http_err}")  # 打印 HTTP 错误信息
            except Exception as e:
                response_success = False  # 请求失败
                logger.exception(f"API [{param.indicator_name}] excute Failed!")  # 记录异常日志
                status = Status.FAILED.value  # 更新状态为 FAILED
                err_msg = f"API [{param.api}] request Failed!{str(e)}"  # 设置错误消息

            # 准备插件参数字典
            plugin_param = {
                "name": param.indicator_name,
                "args": param.args,
                "status": status,
                "logo": None,
                "result": result,
                "err_msg": err_msg,
            }

            if not self.render_protocol:
                raise NotImplementedError("The render_protocol should be implemented.")
            
            # 调用 render_protocol 的 display 方法显示插件参数
            view = await self.render_protocol.display(content=plugin_param)

            # 返回执行结果为成功的 ActionOutput 对象，包括执行状态和响应内容
            return ActionOutput(
                is_exe_success=response_success, content=result, view=view
            )
        except Exception as e:
            logger.exception("Indicator Action Run Failed！")  # 记录异常日志
            return ActionOutput(
                is_exe_success=False, content=f"Indicator action run failed!{str(e)}"
            )  # 返回执行结果为失败的 ActionOutput 对象
```