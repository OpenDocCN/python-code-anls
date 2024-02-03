# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\cli_app\main.py`

```py
# 导入 click 库，用于处理命令行参数
import click

# 导入 AutoGPT 的相关模块
from autogpt.core.agent import AgentSettings, SimpleAgent
from autogpt.core.runner.client_lib.logging import (
    configure_root_logger,
    get_client_logger,
)
from autogpt.core.runner.client_lib.parser import (
    parse_ability_result,
    parse_agent_name_and_goals,
    parse_agent_plan,
    parse_next_ability,
)

# 异步函数，运行 AutoGPT CLI 客户端
async def run_auto_gpt(user_configuration: dict):
    """Run the AutoGPT CLI client."""

    # 配置根日志记录器
    configure_root_logger()

    # 获取客户端日志记录器
    client_logger = get_client_logger()
    client_logger.debug("Getting agent settings")

    # 获取代理工作空间
    agent_workspace = (
        user_configuration.get("workspace", {}).get("configuration", {}).get("root", "")
    )

    # 如果没有代理工作空间
    if not agent_workspace:  # We don't have an agent yet.
        #################
        # Bootstrapping #
        #################
        # 步骤 1. 将用户设置与默认系统设置合并
        agent_settings: AgentSettings = SimpleAgent.compile_settings(
            client_logger,
            user_configuration,
        )

        # 步骤 2. 为代理获取名称和目标
        # 首先需要弄清楚用户希望代理做什么
        # 通过询问用户一个提示来确定
        user_objective = click.prompt("What do you want AutoGPT to do?")
        # 请求语言模型确定适合代理的名称和目标
        name_and_goals = await SimpleAgent.determine_agent_name_and_goals(
            user_objective,
            agent_settings,
            client_logger,
        )
        print("\n" + parse_agent_name_and_goals(name_and_goals))
        # 最后，使用名称和目标更新代理设置
        agent_settings.update_agent_name_and_goals(name_and_goals)

        # 步骤 3. 配置代理
        agent_workspace = SimpleAgent.provision_agent(agent_settings, client_logger)
        client_logger.info("Agent is provisioned")

    # 启动代理交互循环
    # 从工作空间中加载简单代理
    agent = SimpleAgent.from_workspace(
        agent_workspace,
        client_logger,
    )
    # 记录代理已加载
    client_logger.info("Agent is loaded")

    # 构建初始计划并等待完成
    plan = await agent.build_initial_plan()
    # 解析代理计划并打印
    print(parse_agent_plan(plan))

    # 循环执行以下操作
    while True:
        # 确定当前任务和下一个能力
        current_task, next_ability = await agent.determine_next_ability(plan)
        # 解析并打印下一个能力
        print(parse_next_ability(current_task, next_ability))
        # 用户输入是否继续执行该能力
        user_input = click.prompt(
            "Should the agent proceed with this ability?",
            default="y",
        )
        # 执行下一个能力并获取结果
        ability_result = await agent.execute_next_ability(user_input)
        # 解析并打印能力执行结果
        print(parse_ability_result(ability_result))
```