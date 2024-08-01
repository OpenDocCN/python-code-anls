# `.\DB-GPT-src\dbgpt\agent\expand\actions\code_action.py`

```py
"""Code Action Module."""

# 引入日志模块
import logging
# 引入类型提示相关模块
from typing import Optional, Union

# 引入工具函数和类
from dbgpt.util.code_utils import UNKNOWN, execute_code, extract_code, infer_lang
from dbgpt.util.utils import colored
from dbgpt.vis.tags.vis_code import Vis, VisCode

# 引入基础类和资源类
from ...core.action.base import Action, ActionOutput
from ...resource.base import AgentResource

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class CodeAction(Action[None]):
    """Code Action Module."""

    def __init__(self):
        """Create a code action."""
        # 调用父类的初始化方法
        super().__init__()
        # 设置渲染协议对象
        self._render_protocol = VisCode()
        # 初始化代码执行配置为空字典
        self._code_execution_config = {}

    @property
    def render_protocol(self) -> Optional[Vis]:
        """Return the render protocol."""
        # 返回渲染协议对象
        return self._render_protocol

    async def run(
        self,
        ai_message: str,
        resource: Optional[AgentResource] = None,
        rely_action_out: Optional[ActionOutput] = None,
        need_vis_render: bool = True,
        **kwargs,
    ) -> ActionOutput:
        """定义函数的返回类型为ActionOutput。"""
        """执行动作。"""
        try:
            # 提取AI消息中的代码块
            code_blocks = extract_code(ai_message)
            # 如果没有找到可执行的代码块
            if len(code_blocks) < 1:
                logger.info(
                    f"No executable code found in answer,{ai_message}",
                )
                return ActionOutput(
                    is_exe_success=False, content="No executable code found in answer."
                )
            # 如果找到多个代码块，并且第一个代码块的类型是未知的
            elif len(code_blocks) > 1 and code_blocks[0][0] == UNKNOWN:
                logger.info(
                    f"Missing available code block type, unable to execute code,"
                    f"{ai_message}",
                )
                return ActionOutput(
                    is_exe_success=False,
                    content="Missing available code block type, "
                    "unable to execute code.",
                )
            # 执行代码块
            exitcode, logs = self.execute_code_blocks(code_blocks)
            exit_success = exitcode == 0

            # 决定返回的内容是日志还是执行失败的信息
            content = (
                logs
                if exit_success
                else f"exitcode: {exitcode} (execution failed)\n {logs}"
            )

            # 构建参数字典
            param = {
                "exit_success": exit_success,
                "language": code_blocks[0][0],
                "code": code_blocks,
                "log": logs,
            }
            # 如果没有实现渲染协议，则抛出NotImplementedError异常
            if not self.render_protocol:
                raise NotImplementedError("The render_protocol should be implemented.")
            # 等待渲染协议显示内容
            view = await self.render_protocol.display(content=param)
            # 返回动作的输出对象
            return ActionOutput(
                is_exe_success=exit_success,
                content=content,
                view=view,
                thoughts=ai_message,
                observations=content,
            )
        # 捕获异常并记录日志
        except Exception as e:
            logger.exception("Code Action Run Failed！")
            # 返回包含异常信息的动作输出对象
            return ActionOutput(
                is_exe_success=False, content="Code execution exception，" + str(e)
            )
    # 执行给定的代码块并返回结果
    def execute_code_blocks(self, code_blocks):
        """Execute the code blocks and return the result."""
        # 初始化日志字符串
        logs_all = ""
        # 初始化退出码为-1
        exitcode = -1
        # 遍历所有代码块
        for i, code_block in enumerate(code_blocks):
            # 获取语言和代码内容
            lang, code = code_block
            # 如果语言未指定，则推断语言类型
            if not lang:
                lang = infer_lang(code)
            # 打印执行提示信息，包括代码块索引和推断的语言类型
            print(
                colored(
                    f"\n>>>>>>>> EXECUTING CODE BLOCK {i} "
                    f"(inferred language is {lang})...",
                    "red",
                ),
                flush=True,
            )
            # 根据语言类型执行相应的操作
            if lang in ["bash", "shell", "sh"]:
                # 执行 shell 或 bash 代码块
                exitcode, logs, image = execute_code(
                    code, lang=lang, **self._code_execution_config
                )
            elif lang in ["python", "Python"]:
                # 如果代码块以指定的文件名开头，则提取文件名
                if code.startswith("# filename: "):
                    filename = code[11 : code.find("\n")].strip()
                else:
                    filename = None
                # 执行 Python 代码块
                exitcode, logs, image = execute_code(
                    code,
                    lang="python",
                    filename=filename,
                    **self._code_execution_config,
                )
            else:
                # 如果语言不受支持，则返回错误消息
                exitcode, logs, image = (
                    1,
                    f"unknown language {lang}",
                    None,
                )
                # 可选：抛出未实现错误（暂时注释掉）
                # raise NotImplementedError
            # 如果返回的镜像不为空，更新使用 Docker 的配置
            if image is not None:
                self._code_execution_config["use_docker"] = image
            # 将当前代码块执行的日志追加到总日志中
            logs_all += "\n" + logs
            # 如果退出码不为零，直接返回退出码和累积的日志
            if exitcode != 0:
                return exitcode, logs_all
        # 所有代码块执行完成后，返回最终的退出码和所有日志
        return exitcode, logs_all

    @property
    def use_docker(self) -> Union[bool, str, None]:
        """Whether to use docker to execute the code.

        Bool value of whether to use docker to execute the code,
        or str value of the docker image name to use, or None when code execution is
        disabled.
        """
        # 返回当前代码执行配置中是否使用 Docker 的设置
        return (
            None
            if self._code_execution_config is False
            else self._code_execution_config.get("use_docker")
        )
```