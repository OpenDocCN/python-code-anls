# `.\AutoGPT\autogpts\autogpt\autogpt\commands\user_interaction.py`

```py
# 导入与用户交互相关的命令
from __future__ import annotations

# 导入Agent类
from autogpt.agents.agent import Agent
# 导入clean_input函数
from autogpt.app.utils import clean_input
# 导入command装饰器
from autogpt.command_decorator import command
# 导入JSONSchema类
from autogpt.core.utils.json_schema import JSONSchema

# 定义命令的类别和标题
COMMAND_CATEGORY = "user_interaction"
COMMAND_CATEGORY_TITLE = "User Interaction"

# 使用command装饰器定义ask_user命令
@command(
    "ask_user",
    (
        "If you need more details or information regarding the given goals,"
        " you can ask the user for input"
    ),
    {
        "question": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The question or prompt to the user",
            required=True,
        )
    },
    # 根据配置判断是否启用该命令
    enabled=lambda config: not config.noninteractive_mode,
)
# 异步函数，向用户提问并返回用户的回答
async def ask_user(question: str, agent: Agent) -> str:
    # 打印问题或提示给用户
    print(f"\nQ: {question}")
    # 清理用户输入并获取用户回答
    resp = await clean_input(agent.legacy_config, "A:")
    # 返回用户的回答
    return f"The user's answer: '{resp}'"
```