# `.\AutoGPT\autogpts\forge\forge\actions\finish.py`

```py
# 导入 ForgeLogger 类
from sdk.forge_log import ForgeLogger

# 从 registry 模块中导入 action 装饰器
from .registry import action

# 创建名为 logger 的 ForgeLogger 实例
logger = ForgeLogger(__name__)

# 使用 action 装饰器注册一个名为 "finish" 的动作
@action(
    name="finish",
    description="Use this to shut down once you have accomplished all of your goals,"
    " or when there are insurmountable problems that make it impossible"
    " for you to finish your task.",
    parameters=[
        {
            "name": "reason",
            "description": "A summary to the user of how the goals were accomplished",
            "type": "string",
            "required": True,
        }
    ],
    output_type="None",
)
# 定义一个异步函数 finish，接受 agent、task_id 和 reason 三个参数，返回一个字符串
async def finish(
    agent,
    task_id: str,
    reason: str,
) -> str:
    """
    A function that takes in a string and exits the program

    Parameters:
        reason (str): A summary to the user of how the goals were accomplished.
    Returns:
        A result string from create chat completion. A list of suggestions to
            improve the code.
    """
    # 使用 logger 记录信息，标题为 "Shutting down..."
    logger.info(reason, extra={"title": "Shutting down...\n"})
    # 返回传入的 reason 参数
    return reason
```