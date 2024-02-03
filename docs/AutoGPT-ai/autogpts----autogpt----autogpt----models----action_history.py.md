# `.\AutoGPT\autogpts\autogpt\autogpt\models\action_history.py`

```py
# 导入必要的模块和类型提示
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field

# 导入自定义模块
from autogpt.processing.text import summarize_text
from autogpt.prompts.utils import format_numbered_list, indent

# 如果是类型检查阶段，导入额外的模块
if TYPE_CHECKING:
    from autogpt.config.config import Config
    from autogpt.core.resource.model_providers import ChatModelProvider

# 定义 Action 类，包含名称、参数和推理
class Action(BaseModel):
    name: str
    args: dict[str, Any]
    reasoning: str

    # 格式化调用信息
    def format_call(self) -> str:
        return (
            f"{self.name}"
            f"({', '.join([f'{a}={repr(v)}' for a, v in self.args.items()])})"
        )

# 定义 ActionSuccessResult 类，包含输出和状态
class ActionSuccessResult(BaseModel):
    outputs: Any
    status: Literal["success"] = "success"

    # 将结果转换为字符串形式
    def __str__(self) -> str:
        outputs = str(self.outputs).replace("```", r"\```")
        multiline = "\n" in outputs
        return f"```\n{self.outputs}\n```" if multiline else str(self.outputs)

# 定义 ErrorInfo 类，包含参数、消息、异常类型和表示
class ErrorInfo(BaseModel):
    args: tuple
    message: str
    exception_type: str
    repr: str

    # 从异常中创建 ErrorInfo 实例
    @staticmethod
    def from_exception(exception: Exception) -> ErrorInfo:
        return ErrorInfo(
            args=exception.args,
            message=getattr(exception, "message", exception.args[0]),
            exception_type=exception.__class__.__name__,
            repr=repr(exception),
        )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.repr

# 定义 ActionErrorResult 类，包含原因、错误信息和状态
class ActionErrorResult(BaseModel):
    reason: str
    error: Optional[ErrorInfo] = None
    status: Literal["error"] = "error"

    # 从异常中创建 ActionErrorResult 实例
    @staticmethod
    def from_exception(exception: Exception) -> ActionErrorResult:
        return ActionErrorResult(
            reason=getattr(exception, "message", exception.args[0]),
            error=ErrorInfo.from_exception(exception),
        )

    def __str__(self) -> str:
        return f"Action failed: '{self.reason}'"

# 定义 ActionInterruptedByHuman 类，暂时为空
class ActionInterruptedByHuman(BaseModel):
    # 定义一个名为 feedback 的字符串变量
    feedback: str
    # 定义一个名为 status 的字符串字面值类型变量，初始值为 "interrupted_by_human"
    status: Literal["interrupted_by_human"] = "interrupted_by_human"

    # 定义一个特殊方法 __str__，返回一个描述用户中断操作的字符串
    def __str__(self) -> str:
        return (
            # 返回包含用户反馈信息的字符串
            'The user interrupted the action with the following feedback: "%s"'
            % self.feedback
        )
# 定义一个枚举类型，表示操作的执行结果，可以是成功、错误或被人为中断
ActionResult = ActionSuccessResult | ActionErrorResult | ActionInterruptedByHuman

# 定义一个类 Episode，表示一个事件
class Episode(BaseModel):
    # 包含一个操作、一个结果和一个摘要
    action: Action
    result: ActionResult | None
    summary: str | None = None

    # 格式化输出事件的信息
    def format(self):
        step = f"Executed `{self.action.format_call()}`\n"
        step += f'- **Reasoning:** "{self.action.reasoning}"\n'
        step += (
            "- **Status:** "
            f"`{self.result.status if self.result else 'did_not_finish'}`\n"
        )
        if self.result:
            if self.result.status == "success":
                result = str(self.result)
                result = "\n" + indent(result) if "\n" in result else result
                step += f"- **Output:** {result}"
            elif self.result.status == "error":
                step += f"- **Reason:** {self.result.reason}\n"
                if self.result.error:
                    step += f"- **Error:** {self.result.error}\n"
            elif self.result.status == "interrupted_by_human":
                step += f"- **Feedback:** {self.result.feedback}\n"
        return step

    # 返回事件的字符串表示
    def __str__(self) -> str:
        executed_action = f"Executed `{self.action.format_call()}`"
        action_result = f": {self.result}" if self.result else "."
        return executed_action + action_result

# 定义一个类 EpisodicActionHistory，表示一个事件历史记录
class EpisodicActionHistory(BaseModel):
    """Utility container for an action history"""

    episodes: list[Episode] = Field(default_factory=list)
    cursor: int = 0
    _lock = asyncio.Lock()

    # 返回当前事件
    @property
    def current_episode(self) -> Episode | None:
        if self.cursor == len(self):
            return None
        return self[self.cursor]

    # 获取指定位置的事件
    def __getitem__(self, key: int) -> Episode:
        return self.episodes[key]

    # 迭代事件列表
    def __iter__(self) -> Iterator[Episode]:
        return iter(self.episodes)

    # 返回事件列表的长度
    def __len__(self) -> int:
        return len(self.episodes)

    # 判断事件列表是否为空
    def __bool__(self) -> bool:
        return len(self.episodes) > 0
    # 注册一个动作到当前周期中
    def register_action(self, action: Action) -> None:
        # 如果当前周期为空，则创建一个新的周期并添加动作
        if not self.current_episode:
            self.episodes.append(Episode(action=action, result=None))
            # 确保当前周期不为空
            assert self.current_episode
        # 如果当前周期已经有动作，则抛出数值错误
        elif self.current_episode.action:
            raise ValueError("Action for current cycle already set")

    # 注册一个结果到当前周期中
    def register_result(self, result: ActionResult) -> None:
        # 如果当前周期为空，则无法注册结果，抛出运行时错误
        if not self.current_episode:
            raise RuntimeError("Cannot register result for cycle without action")
        # 如果当前周期已经有结果，则抛出数值错误
        elif self.current_episode.result:
            raise ValueError("Result for current cycle already set")

        # 将结果注册到当前周期中，并更新游标位置
        self.current_episode.result = result
        self.cursor = len(self.episodes)

    # 回退历史记录到之前的状态
    def rewind(self, number_of_episodes: int = 0) -> None:
        """Resets the history to an earlier state.

        Params:
            number_of_cycles (int): The number of cycles to rewind. Default is 0.
                When set to 0, it will only reset the current cycle.
        """
        # 移除当前周期的部分记录
        if self.current_episode:
            if self.current_episode.action and not self.current_episode.result:
                self.episodes.pop(self.cursor)

        # 回退指定数量的周期
        if number_of_episodes > 0:
            self.episodes = self.episodes[:-number_of_episodes]
            self.cursor = len(self.episodes)

    # 处理压缩
    async def handle_compression(
        self, llm_provider: ChatModelProvider, app_config: Config
    ) -> None:
        """使用LLM压缩操作历史中的每个事件。

        该方法遍历操作历史中所有没有摘要的事件，并使用LLM为它们生成摘要。
        """
        compress_instruction = (
            "文本表示一个操作，执行该操作的原因，"
            "以及其结果。"
            "将执行的操作和其结果压缩成一行。"
            "保留操作收集的任何具体事实信息。"
        )
        async with self._lock:
            # 收集所有没有摘要的事件
            episodes_to_summarize = [ep for ep in self.episodes if ep.summary is None]

            # 并行化摘要调用
            summarize_coroutines = [
                summarize_text(
                    episode.format(),
                    instruction=compress_instruction,
                    llm_provider=llm_provider,
                    config=app_config,
                )
                for episode in episodes_to_summarize
            ]
            summaries = await asyncio.gather(*summarize_coroutines)

            # 将摘要分配给事件
            for episode, (summary, _) in zip(episodes_to_summarize, summaries):
                episode.summary = summary

    def fmt_list(self) -> str:
        return format_numbered_list(self.episodes)

    def fmt_paragraph(self) -> str:
        steps: list[str] = []

        for i, episode in enumerate(self.episodes, 1):
            step = f"### 步骤 {i}: {episode.format()}\n"

            steps.append(step)

        return "\n\n".join(steps)
```