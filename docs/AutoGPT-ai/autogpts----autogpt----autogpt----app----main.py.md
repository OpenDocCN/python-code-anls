# `.\AutoGPT\autogpts\autogpt\autogpt\app\main.py`

```py
"""
The application entry point. Can be invoked by a CLI or any other front end application.
"""

# 导入必要的库
import enum
import logging
import math
import os
import re
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Optional

# 导入 colorama 库中的 Fore 和 Style
from colorama import Fore, Style

# 导入 forge.sdk.db 中的 AgentDB
from forge.sdk.db import AgentDB

# 如果 TYPE_CHECKING 为真，则从 autogpt.agents.agent 中导入 Agent
if TYPE_CHECKING:
    from autogpt.agents.agent import Agent

# 从 autogpt.agent_factory.configurators 中导入 configure_agent_with_state 和 create_agent
from autogpt.agent_factory.configurators import configure_agent_with_state, create_agent

# 从 autogpt.agent_factory.profile_generator 中导入 generate_agent_profile_for_task
from autogpt.agent_factory.profile_generator import generate_agent_profile_for_task

# 从 autogpt.agent_manager 中导入 AgentManager
from autogpt.agent_manager import AgentManager

# 从 autogpt.agents 中导入 AgentThoughts, CommandArgs, CommandName
from autogpt.agents import AgentThoughts, CommandArgs, CommandName

# 从 autogpt.agents.utils.exceptions 中导入 AgentTerminated, InvalidAgentResponseError
from autogpt.agents.utils.exceptions import AgentTerminated, InvalidAgentResponseError

# 从 autogpt.config 中导入 AIDirectives, AIProfile, Config, ConfigBuilder, assert_config_has_openai_api_key
from autogpt.config import (
    AIDirectives,
    AIProfile,
    Config,
    ConfigBuilder,
    assert_config_has_openai_api_key,
)

# 从 autogpt.core.resource.model_providers.openai 中导入 OpenAIProvider
from autogpt.core.resource.model_providers.openai import OpenAIProvider

# 从 autogpt.core.runner.client_lib.utils 中导入 coroutine
from autogpt.core.runner.client_lib.utils import coroutine

# 从 autogpt.logs.config 中导入 configure_chat_plugins, configure_logging
from autogpt.logs.config import configure_chat_plugins, configure_logging

# 从 autogpt.logs.helpers 中导入 print_attribute, speak
from autogpt.logs.helpers import print_attribute, speak

# 从 autogpt.plugins 中导入 scan_plugins
from autogpt.plugins import scan_plugins

# 从 scripts.install_plugin_deps 中导入 install_plugin_dependencies
from scripts.install_plugin_deps import install_plugin_dependencies

# 从当前目录中的 configurator.py 中导入 apply_overrides_to_config
from .configurator import apply_overrides_to_config

# 从当前目录中的 setup.py 中导入 apply_overrides_to_ai_settings, interactively_revise_ai_settings
from .setup import apply_overrides_to_ai_settings, interactively_revise_ai_settings

# 从当前目录中的 spinner.py 中导入 Spinner
from .spinner import Spinner

# 从当前目录中的 utils.py 中导入 clean_input, get_legal_warning, markdown_to_ansi_style, print_git_branch_info, print_motd, print_python_version_info
from .utils import (
    clean_input,
    get_legal_warning,
    markdown_to_ansi_style,
    print_git_branch_info,
    print_motd,
    print_python_version_info,
)

# 定义异步函数 run_auto_gpt，接受多个参数
@coroutine
async def run_auto_gpt(
    continuous: bool = False,
    continuous_limit: Optional[int] = None,
    ai_settings: Optional[Path] = None,
    prompt_settings: Optional[Path] = None,
    skip_reprompt: bool = False,
    speak: bool = False,
    debug: bool = False,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    # 定义日志文件格式，默认为 None
    log_file_format: Optional[str] = None,
    # 是否仅使用 GPT-3 模型，默认为 False
    gpt3only: bool = False,
    # 是否仅使用 GPT-4 模型，默认为 False
    gpt4only: bool = False,
    # 浏览器名称，默认为 None
    browser_name: Optional[str] = None,
    # 是否允许下载，默认为 False
    allow_downloads: bool = False,
    # 是否跳过新闻，默认为 False
    skip_news: bool = False,
    # 工作空间目录，默认为 None
    workspace_directory: Optional[Path] = None,
    # 是否安装插件依赖，默认为 False
    install_plugin_deps: bool = False,
    # 覆盖 AI 名称，默认为 None
    override_ai_name: Optional[str] = None,
    # 覆盖 AI 角色，默认为 None
    override_ai_role: Optional[str] = None,
    # 资源列表，默认为 None
    resources: Optional[list[str]] = None,
    # 约束列表，默认为 None
    constraints: Optional[list[str]] = None,
    # 最佳实践列表，默认为 None
    best_practices: Optional[list[str]] = None,
    # 是否覆盖指令，默认为 False
    override_directives: bool = False,
# 从环境变量构建配置信息
config = ConfigBuilder.build_config_from_env()

# 确保配置信息中包含 OpenAI API 密钥
assert_config_has_openai_api_key(config)

# 应用对配置信息的覆盖
apply_overrides_to_config(
    config=config,
    continuous=continuous,
    continuous_limit=continuous_limit,
    ai_settings_file=ai_settings,
    prompt_settings_file=prompt_settings,
    skip_reprompt=skip_reprompt,
    speak=speak,
    debug=debug,
    log_level=log_level,
    log_format=log_format,
    log_file_format=log_file_format,
    gpt3only=gpt3only,
    gpt4only=gpt4only,
    browser_name=browser_name,
    allow_downloads=allow_downloads,
    skip_news=skip_news,
)

# 配置日志模块
configure_logging(
    **config.logging.dict(),
    tts_config=config.tts_config,
)

# 配置 OpenAI 提供者
llm_provider = _configure_openai_provider(config)

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 如果配置信息中包含连续模式，则输出法律警告信息
if config.continuous_mode:
    for line in get_legal_warning().split("\n"):
        logger.warning(
            extra={
                "title": "LEGAL:",
                "title_color": Fore.RED,
                "preserve_color": True,
            },
            msg=markdown_to_ansi_style(line),
        )
    # 如果不跳过新闻，则打印欢迎信息、Git 分支信息、Python 版本信息等
    if not config.skip_news:
        print_motd(config, logger)
        print_git_branch_info(logger)
        print_python_version_info(logger)
        print_attribute("Smart LLM", config.smart_llm)
        print_attribute("Fast LLM", config.fast_llm)
        print_attribute("Browser", config.selenium_web_browser)
        # 如果处于连续模式，则打印连续模式信息
        if config.continuous_mode:
            print_attribute("Continuous Mode", "ENABLED", title_color=Fore.YELLOW)
            # 如果有连续模式限制，则打印连续模式限制信息
            if continuous_limit:
                print_attribute("Continuous Limit", config.continuous_limit)
        # 如果 TTS 配置中的说话模式开启，则打印说话模式信息
        if config.tts_config.speak_mode:
            print_attribute("Speak Mode", "ENABLED")
        # 如果有 AI 设置，则打印使用 AI 设置文件信息
        if ai_settings:
            print_attribute("Using AI Settings File", ai_settings)
        # 如果有提示设置，则打印使用提示设置文件信息
        if prompt_settings:
            print_attribute("Using Prompt Settings File", prompt_settings)
        # 如果允许下载，则打印本地下载信息
        if config.allow_downloads:
            print_attribute("Native Downloading", "ENABLED")

    # 如果需要安装插件依赖，则安装插件依赖
    if install_plugin_deps:
        install_plugin_dependencies()

    # 扫描插件并配置聊天插件
    config.plugins = scan_plugins(config)
    configure_chat_plugins(config)

    # 让用户选择要运行的现有代理
    agent_manager = AgentManager(config.app_data_dir)
    existing_agents = agent_manager.list_agents()
    load_existing_agent = ""
    # 如果存在现有代理，则列出并让用户选择
    if existing_agents:
        print(
            "Existing agents\n---------------\n"
            + "\n".join(f"{i} - {id}" for i, id in enumerate(existing_agents, 1))
        )
        load_existing_agent = await clean_input(
            config,
            "Enter the number or name of the agent to run,"
            " or hit enter to create a new one:",
        )
        # 如果用户输入数字，则选择对应的现有代理
        if re.match(r"^\d+$", load_existing_agent):
            load_existing_agent = existing_agents[int(load_existing_agent) - 1]
        # 如果用户输入的代理不在现有代理列表中，则抛出异常
        elif load_existing_agent and load_existing_agent not in existing_agents:
            raise ValueError(f"Unknown agent '{load_existing_agent}'")

    # 加载现有代理或设置新代理状态
    agent = None
    # 初始化 agent_state 变量为 None
    agent_state = None

    ############################
    # Resume an Existing Agent #
    ############################
    # 如果需要加载现有的 agent
    if load_existing_agent:
        # 从 agent_manager 中检索 agent 的状态
        agent_state = agent_manager.retrieve_state(load_existing_agent)
        # 循环直到用户输入正确的回答
        while True:
            # 等待用户输入是否要继续
            answer = await clean_input(config, "Resume? [Y/n]")
            # 如果用户输入为 'y'，则继续
            if answer.lower() == "y":
                break
            # 如果用户输入为 'n'，则重置 agent_state 为 None 并退出循环
            elif answer.lower() == "n":
                agent_state = None
                break
            # 如果用户输入不是 'y' 或 'n'，则提示用户重新输入
            else:
                print("Please respond with 'y' or 'n'")

    # 如果存在 agent_state
    if agent_state:
        # 根据 agent_state 配置 agent
        agent = configure_agent_with_state(
            state=agent_state,
            app_config=config,
            llm_provider=llm_provider,
        )
        # 应用对 AI 设置的覆盖
        apply_overrides_to_ai_settings(
            ai_profile=agent.state.ai_profile,
            directives=agent.state.directives,
            override_name=override_ai_name,
            override_role=override_ai_role,
            resources=resources,
            constraints=constraints,
            best_practices=best_practices,
            replace_directives=override_directives,
        )

        # 如果有任何参数被指定，假设用户不想修改它们
        if not any(
            [
                override_ai_name,
                override_ai_role,
                resources,
                constraints,
                best_practices,
            ]
        ):
            # 交互式地修改 AI 设置
            ai_profile, ai_directives = await interactively_revise_ai_settings(
                ai_profile=agent.state.ai_profile,
                directives=agent.state.directives,
                app_config=config,
            )
        else:
            logger.info("AI config overrides specified through CLI; skipping revision")

    ######################
    # Set up a new Agent #
    ######################
    # 设置一个新的 agent

    #################
    # Run the Agent #
    #################
    # 运行 agent
    try:
        await run_interaction_loop(agent)
    # 捕获 AgentTerminated 异常
    except AgentTerminated:
        # 获取 agent 的 ID
        agent_id = agent.state.agent_id
        # 记录日志，提示正在保存 agent 的状态
        logger.info(f"Saving state of {agent_id}...")

        # 允许用户另存为其他 ID
        save_as_id = (
            # 等待用户输入，如果用户按下回车，则保存为当前 agent_id，否则保存为用户输入的 ID
            await clean_input(
                config,
                f"Press enter to save as '{agent_id}',"
                " or enter a different ID to save to:",
            )
            or agent_id
        )
        # 如果用户输入了新的 ID 并且不同于原来的 agent_id
        if save_as_id and save_as_id != agent_id:
            # 设置 agent 的新 ID 和新的 agent 目录
            agent.set_id(
                new_id=save_as_id,
                new_agent_dir=agent_manager.get_agent_dir(save_as_id),
            )
            # TODO: 如果用户想要克隆工作空间，则执行克隆操作
            # TODO: ... 或者允许多对一的 agent 和工作空间关系

        # 将 agent 的状态保存为 JSON 文件
        agent.state.save_to_json_file(agent.file_manager.state_file_path)
# 定义一个异步协程函数，用于运行自动 GPT 服务器
@coroutine
async def run_auto_gpt_server(
    prompt_settings: Optional[Path] = None,
    debug: bool = False,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file_format: Optional[str] = None,
    gpt3only: bool = False,
    gpt4only: bool = False,
    browser_name: Optional[str] = None,
    allow_downloads: bool = False,
    install_plugin_deps: bool = False,
):
    # 导入 AgentProtocolServer 类
    from .agent_protocol_server import AgentProtocolServer

    # 从环境变量构建配置对象
    config = ConfigBuilder.build_config_from_env()

    # 确保配置对象中包含 OpenAI API 密钥
    assert_config_has_openai_api_key(config)

    # 应用配置覆盖
    apply_overrides_to_config(
        config=config,
        prompt_settings_file=prompt_settings,
        debug=debug,
        log_level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        gpt3only=gpt3only,
        gpt4only=gpt4only,
        browser_name=browser_name,
        allow_downloads=allow_downloads,
    )

    # 设置日志模块
    configure_logging(
        **config.logging.dict(),
        tts_config=config.tts_config,
    )

    # 配置 OpenAIProvider 对象
    llm_provider = _configure_openai_provider(config)

    # 如果需要安装插件依赖，则安装
    if install_plugin_deps:
        install_plugin_dependencies()

    # 扫描插件
    config.plugins = scan_plugins(config)

    # 设置并启动服务器
    database = AgentDB(
        database_string=os.getenv("AP_SERVER_DB_URL", "sqlite:///data/ap_server.db"),
        debug_enabled=debug,
    )
    port: int = int(os.getenv("AP_SERVER_PORT", default=8000))
    server = AgentProtocolServer(
        app_config=config, database=database, llm_provider=llm_provider
    )
    await server.start(port=port)

    # 记录 OpenAI 会话总成本
    logging.getLogger().info(
        f"Total OpenAI session cost: "
        f"${round(sum(b.total_cost for b in server._task_budgets.values()), 2)}"
    )


# 创建配置好的 OpenAIProvider 对象
def _configure_openai_provider(config: Config) -> OpenAIProvider:
    """Create a configured OpenAIProvider object.

    Args:
        config: The program's configuration.
    # 返回一个配置好的 OpenAIProvider 对象
    """
    # 如果配置中没有设置 OpenAI 凭据，则抛出运行时错误
    if config.openai_credentials is None:
        raise RuntimeError("OpenAI key is not configured")

    # 复制默认的 OpenAIProvider 设置，并设置凭据为配置中的凭据
    openai_settings = OpenAIProvider.default_settings.copy(deep=True)
    openai_settings.credentials = config.openai_credentials
    # 返回一个配置好的 OpenAIProvider 对象，设置为配置中的凭据和日志记录器
    return OpenAIProvider(
        settings=openai_settings,
        logger=logging.getLogger("OpenAIProvider"),
    )
def _get_cycle_budget(continuous_mode: bool, continuous_limit: int) -> int | float:
    # 根据 continuous_mode 和 continuous_limit 配置，转换为 cycle_budget（不与用户交互时最大运行周期数）和 cycles_remaining（在与用户交互之前剩余的周期数）。
    if continuous_mode:
        # 如果处于连续模式，则 cycle_budget 为 continuous_limit，如果 continuous_limit 不存在则为无穷大
        cycle_budget = continuous_limit if continuous_limit else math.inf
    else:
        # 如果不处于连续模式，则 cycle_budget 为 1
        cycle_budget = 1

    return cycle_budget


class UserFeedback(str, enum.Enum):
    """用户反馈的枚举类型。"""

    AUTHORIZE = "GENERATE NEXT COMMAND JSON"
    EXIT = "EXIT"
    TEXT = "TEXT"


async def run_interaction_loop(
    agent: "Agent",
) -> None:
    """运行代理的主交互循环。

    Args:
        agent: 要运行交互循环的代理。

    Returns:
        None
    """
    # 这些包含应用程序配置和代理配置，因此在这里获取它们。
    legacy_config = agent.legacy_config
    ai_profile = agent.ai_profile
    logger = logging.getLogger(__name__)

    # 获取运行周期预算
    cycle_budget = cycles_remaining = _get_cycle_budget(
        legacy_config.continuous_mode, legacy_config.continuous_limit
    )
    # 创建一个 Spinner 对象，用于显示“Thinking...”，根据 legacy_config.logging.plain_console_output 决定是否显示简单输出
    spinner = Spinner(
        "Thinking...", plain_output=legacy_config.logging.plain_console_output
    )
    stop_reason = None
    # 定义一个处理优雅中断的函数，接收信号编号和帧对象作为参数，无返回值
    def graceful_agent_interrupt(signum: int, frame: Optional[FrameType]) -> None:
        # 使用 nonlocal 关键字声明在外部作用域中定义的变量
        nonlocal cycle_budget, cycles_remaining, spinner, stop_reason
        # 如果存在停止原因，则记录错误信息并退出程序
        if stop_reason:
            logger.error("Quitting immediately...")
            sys.exit()
        # 如果剩余循环次数为0或1，则记录警告信息，并设置停止原因为中断信号
        if cycles_remaining in [0, 1]:
            logger.warning("Interrupt signal received: shutting down gracefully.")
            logger.warning(
                "Press Ctrl+C again if you want to stop AutoGPT immediately."
            )
            stop_reason = AgentTerminated("Interrupt signal received")
        else:
            # 如果旋转器正在运行，则停止旋转器
            restart_spinner = spinner.running
            if spinner.running:
                spinner.stop()

            # 记录错误信息，停止连续命令执行，将剩余循环次数设置为1
            logger.error(
                "Interrupt signal received: stopping continuous command execution."
            )
            cycles_remaining = 1
            # 如果之前旋转器正在运行，则重新启动旋转器
            if restart_spinner:
                spinner.start()

    # 定义一个处理停止信号的函数，无参数，无返回值
    def handle_stop_signal() -> None:
        # 如果存在停止原因，则抛出停止原因异常
        if stop_reason:
            raise stop_reason

    # 设置一个中断信号处理函数为优雅中断函数
    signal.signal(signal.SIGINT, graceful_agent_interrupt)

    #########################
    # 应用程序主循环 #
    #########################

    # 记录代理连续失败的次数
    consecutive_failures = 0
# 更新用户信息，打印助手的想法和下一个要执行的命令给用户
def update_user(
    ai_profile: AIProfile,
    command_name: CommandName,
    command_args: CommandArgs,
    assistant_reply_dict: AgentThoughts,
    speak_mode: bool = False,
) -> None:
    """Prints the assistant's thoughts and the next command to the user.

    Args:
        config: The program's configuration.
        ai_profile: The AI's personality/profile
        command_name: The name of the command to execute.
        command_args: The arguments for the command.
        assistant_reply_dict: The assistant's reply.
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)

    # 打印助手的想法
    print_assistant_thoughts(
        ai_name=ai_profile.ai_name,
        assistant_reply_json_valid=assistant_reply_dict,
        speak_mode=speak_mode,
    )

    # 如果处于说话模式，说出要执行的命令
    if speak_mode:
        speak(f"I want to execute {command_name}")

    # 首先打印新行，以便用户在控制台中更好地区分各个部分
    print()
    # 记录命令和参数信息到日志
    logger.info(
        f"COMMAND = {Fore.CYAN}{remove_ansi_escape(command_name)}{Style.RESET_ALL}  "
        f"ARGUMENTS = {Fore.CYAN}{command_args}{Style.RESET_ALL}",
        extra={
            "title": "NEXT ACTION:",
            "title_color": Fore.CYAN,
            "preserve_color": True,
        },
    )


# 异步获取用户反馈
async def get_user_feedback(
    config: Config,
    ai_profile: AIProfile,
) -> tuple[UserFeedback, str, int | None]:
    """Gets the user's feedback on the assistant's reply.

    Args:
        config: The program's configuration.
        ai_profile: The AI's configuration.

    Returns:
        A tuple of the user's feedback, the user's input, and the number of
        cycles remaining if the user has initiated a continuous cycle.
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)

    # 获取用户授权以执行命令
    # 获取按键：提示用户按回车键继续或按ESC键退出
    # 输出提示信息，告知用户如何进行操作
    logger.info(
        f"Enter '{config.authorise_key}' to authorise command, "
        f"'{config.authorise_key} -N' to run N continuous commands, "
        f"'{config.exit_key}' to exit program, or enter feedback for "
        f"{ai_profile.ai_name}..."
    )

    # 初始化用户反馈、用户输入和新的循环次数
    user_feedback = None
    user_input = ""
    new_cycles_remaining = None

    # 循环直到用户提供有效反馈
    while user_feedback is None:
        # 从用户获取输入
        if config.chat_messages_enabled:
            console_input = await clean_input(config, "Waiting for your response...")
        else:
            console_input = await clean_input(
                config, Fore.MAGENTA + "Input:" + Style.RESET_ALL
            )

        # 解析用户输入
        if console_input.lower().strip() == config.authorise_key:
            user_feedback = UserFeedback.AUTHORIZE
        elif console_input.lower().strip() == "":
            logger.warning("Invalid input format.")
        elif console_input.lower().startswith(f"{config.authorise_key} -"):
            try:
                user_feedback = UserFeedback.AUTHORIZE
                new_cycles_remaining = abs(int(console_input.split(" ")[1]))
            except ValueError:
                logger.warning(
                    f"Invalid input format. "
                    f"Please enter '{config.authorise_key} -N'"
                    " where N is the number of continuous tasks."
                )
        elif console_input.lower() in [config.exit_key, "exit"]:
            user_feedback = UserFeedback.EXIT
        else:
            user_feedback = UserFeedback.TEXT
            user_input = console_input

    # 返回用户反馈、用户输入和新的循环次数
    return user_feedback, user_input, new_cycles_remaining
# 定义一个函数，用于打印助手的思考内容
def print_assistant_thoughts(
    ai_name: str,  # AI助手的名称
    assistant_reply_json_valid: dict,  # 包含有效回复的字典
    speak_mode: bool = False,  # 是否为说话模式，默认为False
) -> None:  # 函数返回空值

    # 获取日志记录器
    logger = logging.getLogger(__name__)

    # 初始化助手思考的各个部分
    assistant_thoughts_reasoning = None
    assistant_thoughts_plan = None
    assistant_thoughts_speak = None
    assistant_thoughts_criticism = None

    # 从回复中获取助手的思考内容
    assistant_thoughts = assistant_reply_json_valid.get("thoughts", {})
    assistant_thoughts_text = remove_ansi_escape(assistant_thoughts.get("text", ""))
    if assistant_thoughts:
        assistant_thoughts_reasoning = remove_ansi_escape(
            assistant_thoughts.get("reasoning", "")
        )
        assistant_thoughts_plan = remove_ansi_escape(assistant_thoughts.get("plan", ""))
        assistant_thoughts_criticism = remove_ansi_escape(
            assistant_thoughts.get("self_criticism", "")
        )
        assistant_thoughts_speak = remove_ansi_escape(
            assistant_thoughts.get("speak", "")
        )
    
    # 打印助手思考的文本内容
    print_attribute(
        f"{ai_name.upper()} THOUGHTS", assistant_thoughts_text, title_color=Fore.YELLOW
    )
    # 打印助手的推理过程
    print_attribute("REASONING", assistant_thoughts_reasoning, title_color=Fore.YELLOW)
    if assistant_thoughts_plan:
        print_attribute("PLAN", "", title_color=Fore.YELLOW)
        # 如果是列表，则将其连接成字符串
        if isinstance(assistant_thoughts_plan, list):
            assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
        elif isinstance(assistant_thoughts_plan, dict):
            assistant_thoughts_plan = str(assistant_thoughts_plan)

        # 使用换行符和破折号拆分输入字符串
        lines = assistant_thoughts_plan.split("\n")
        for line in lines:
            line = line.lstrip("- ")
            logger.info(line.strip(), extra={"title": "- ", "title_color": Fore.GREEN})
    # 打印助手的自我批评
    print_attribute(
        "CRITICISM", f"{assistant_thoughts_criticism}", title_color=Fore.YELLOW
    )

    # 说出助手的思考内容
    # 如果助手的思维有内容
    if assistant_thoughts_speak:
        # 如果处于说话模式
        if speak_mode:
            # 调用说话函数，输出助手的思维内容
            speak(assistant_thoughts_speak)
        else:
            # 打印助手的思维内容，标记为"SPEAK"，使用黄色标题颜色
            print_attribute("SPEAK", assistant_thoughts_speak, title_color=Fore.YELLOW)
# 定义一个函数，用于移除字符串中的 ANSI 转义字符
def remove_ansi_escape(s: str) -> str:
    # 使用 replace 方法将 ANSI 转义字符（\x1B）替换为空字符串，实现移除功能
    return s.replace("\x1B", "")
```