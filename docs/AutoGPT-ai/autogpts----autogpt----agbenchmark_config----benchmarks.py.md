# `.\AutoGPT\autogpts\autogpt\agbenchmark_config\benchmarks.py`

```py
# 导入必要的模块
import asyncio
import logging
import sys
from pathlib import Path

# 导入自定义模块
from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.app.main import _configure_openai_provider, run_interaction_loop
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIProfile, ConfigBuilder
from autogpt.logs.config import configure_logging
from autogpt.models.command_registry import CommandRegistry

# 设置日志目录为当前文件所在目录下的logs文件夹
LOG_DIR = Path(__file__).parent / "logs"

# 定义运行特定任务的函数
def run_specific_agent(task: str, continuous_mode: bool = False) -> None:
    # 初始化代理对象
    agent = bootstrap_agent(task, continuous_mode)
    # 运行交互循环
    asyncio.run(run_interaction_loop(agent))

# 初始化代理对象
def bootstrap_agent(task: str, continuous_mode: bool) -> Agent:
    # 从环境变量构建配置
    config = ConfigBuilder.build_config_from_env()
    # 设置日志级别为DEBUG
    config.logging.level = logging.DEBUG
    # 设置日志目录
    config.logging.log_dir = LOG_DIR
    # 设置在控制台输出纯文本日志
    config.logging.plain_console_output = True
    # 配置日志
    configure_logging(**config.logging.dict())

    # 设置连续模式和限制
    config.continuous_mode = continuous_mode
    config.continuous_limit = 20
    config.noninteractive_mode = True
    config.memory_backend = "no_memory"

    # 创建命令注册表
    command_registry = CommandRegistry.with_command_modules(COMMAND_CATEGORIES, config)

    # 创建AI配置文件
    ai_profile = AIProfile(
        ai_name="AutoGPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task],
    )

    # 复制代理默认设置的提示配置
    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    # 设置是否使用函数API
    agent_prompt_config.use_functions_api = config.openai_functions
    # 创建代理设置对象，包括名称、描述、AI配置、配置、提示配置和历史记录
    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        ai_profile=ai_profile,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            allow_fs_access=not config.restrict_to_workspace,
            use_functions_api=config.openai_functions,
            plugins=config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )

    # 创建代理对象，设置代理的设置、LLM提供者、命令注册表和旧配置
    agent = Agent(
        settings=agent_settings,
        llm_provider=_configure_openai_provider(config),
        command_registry=command_registry,
        legacy_config=config,
    )
    # 附加文件系统到代理，用于访问指定目录下的文件（这里是一个 HACK）
    agent.attach_fs(config.app_data_dir / "agents" / "AutoGPT-benchmark")  # HACK
    # 返回代理对象
    return agent
# 如果当前脚本被当作主程序执行
if __name__ == "__main__":
    # 检查命令行参数数量是否为2，如果不是则打印用法信息并退出程序
    if len(sys.argv) != 2:
        print("Usage: python script.py <task>")
        sys.exit(1)
    # 获取第二个命令行参数作为任务名称
    task = sys.argv[1]
    # 运行特定任务的代理程序，设置为连续模式
    run_specific_agent(task, continuous_mode=True)
```