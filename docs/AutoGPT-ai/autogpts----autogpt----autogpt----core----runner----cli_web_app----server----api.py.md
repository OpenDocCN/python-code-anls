# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\cli_web_app\server\api.py`

```py
# 导入 logging 模块
import logging

# 从 agent_protocol 模块中导入 StepHandler 和 StepResult 类
from agent_protocol import StepHandler, StepResult

# 从 autogpt.agents 模块中导入 Agent 类
from autogpt.agents import Agent
# 从 autogpt.app.main 模块中导入 UserFeedback 类
from autogpt.app.main import UserFeedback
# 从 autogpt.commands 模块中导入 COMMAND_CATEGORIES 常量
from autogpt.commands import COMMAND_CATEGORIES
# 从 autogpt.config 模块中导入 AIProfile 和 ConfigBuilder 类
from autogpt.config import AIProfile, ConfigBuilder
# 从 autogpt.logs.helpers 模块中导入 user_friendly_output 函数
from autogpt.logs.helpers import user_friendly_output
# 从 autogpt.models.command_registry 模块中导入 CommandRegistry 类
from autogpt.models.command_registry import CommandRegistry
# 从 autogpt.prompts.prompt 模块中导入 DEFAULT_TRIGGERING_PROMPT 常量
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT

# 定义一个异步函数 task_handler，接受 task_input 参数，返回 StepHandler 对象
async def task_handler(task_input) -> StepHandler:
    # 从 task_input 中获取任务信息，如果不存在则为空字典
    task = task_input.__root__ if task_input else {}
    # 根据用户输入和是否为测试模式创建代理对象
    agent = bootstrap_agent(task.get("user_input"), False)

    # 初始化下一个命令的名称和参数
    next_command_name: str | None = None
    next_command_args: dict[str, str] | None = None

    # 定义一个异步函数 step_handler，接受 step_input 参数，返回 StepResult 对象
    async def step_handler(step_input) -> StepResult:
        # 从 step_input 中获取步骤信息，如果不存在则为空字典
        step = step_input.__root__ if step_input else {}

        nonlocal next_command_name, next_command_args

        # 调用 interaction_step 函数进行交互步骤处理
        result = await interaction_step(
            agent,
            step.get("user_input"),
            step.get("user_feedback"),
            next_command_name,
            next_command_args,
        )

        # 更新下一个命令的名称和参数
        next_command_name = result["next_step_command_name"] if result else None
        next_command_args = result["next_step_command_args"] if result else None

        # 如果结果为空，则返回最终结果
        if not result:
            return StepResult(output=None, is_last=True)
        return StepResult(output=result)

    return step_handler

# 定义一个异步函数 interaction_step，接受 agent、user_input、user_feedback、command_name 和 command_args 参数
async def interaction_step(
    agent: Agent,
    user_input,
    user_feedback: UserFeedback | None,
    command_name: str | None,
    command_args: dict[str, str] | None,
):
    """Run one step of the interaction loop."""
    # 如果用户反馈为退出，则返回
    if user_feedback == UserFeedback.EXIT:
        return
    # 如果用户反馈为文本，则设置命令名称为 "human_feedback"
    if user_feedback == UserFeedback.TEXT:
        command_name = "human_feedback"

    # 初始化结果变量
    result: str | None = None
    # 如果命令名称不为空
    if command_name is not None:
        # 执行指定命令，并获取结果
        result = agent.execute(command_name, command_args, user_input)
        # 如果结果为空
        if result is None:
            # 输出友好提示信息
            user_friendly_output(
                title="SYSTEM:", message="Unable to execute command", level=logging.WARN
            )
            # 返回
            return

    # 获取下一个建议动作的命令名称、参数和助手回复字典
    next_command_name, next_command_args, assistant_reply_dict = agent.propose_action()

    # 返回包含配置、AI配置、执行结果、助手回复字典、下一步命令名称和参数的字典
    return {
        "config": agent.config,
        "ai_profile": agent.ai_profile,
        "result": result,
        "assistant_reply_dict": assistant_reply_dict,
        "next_step_command_name": next_command_name,
        "next_step_command_args": next_command_args,
    }
# 根据任务和连续模式创建一个代理对象
def bootstrap_agent(task, continuous_mode) -> Agent:
    # 从环境中构建配置对象
    config = ConfigBuilder.build_config_from_env()
    # 设置日志级别为 DEBUG
    config.logging.level = logging.DEBUG
    # 设置是否在控制台输出日志
    config.logging.plain_console_output = True
    # 设置是否为连续模式
    config.continuous_mode = continuous_mode
    # 设置温度为0
    config.temperature = 0
    # 创建命令注册表
    command_registry = CommandRegistry.with_command_modules(COMMAND_CATEGORIES, config)
    # 设置内存后端为"no_memory"
    config.memory_backend = "no_memory"
    # 创建AI配置文件
    ai_profile = AIProfile(
        ai_name="AutoGPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task],
    )
    # 返回一个代理对象
    return Agent(
        command_registry=command_registry,
        ai_profile=ai_profile,
        config=config,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
    )
```