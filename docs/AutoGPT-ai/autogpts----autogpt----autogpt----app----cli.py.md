# `.\AutoGPT\autogpts\autogpt\autogpt\app\cli.py`

```py
# 主要脚本文件，用于 autogpt 包
from logging import _nameToLevel as logLevelMap
from pathlib import Path
from typing import Optional

import click

from autogpt.logs.config import LogFormatName

# 创建一个命令组，如果没有指定子命令，则默认调用 `run`
@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)

# 定义一个名为 `run` 的命令
@cli.command()
# 启用连续模式的选项
@click.option("-c", "--continuous", is_flag=True, help="Enable Continuous Mode")
# 定义连续模式运行次数的选项
@click.option(
    "-l",
    "--continuous-limit",
    type=int,
    help="Defines the number of times to run in continuous mode",
)
# 启用说话模式的选项
@click.option("--speak", is_flag=True, help="Enable Speak Mode")
# 启用仅使用 GPT3.5 模式的选项
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
# 启用仅使用 GPT4 模式的选项
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
# 指定使用的 web 浏览器的选项
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
# 允许 AutoGPT 原生下载文件的选项（潜在危险）
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows AutoGPT to download files natively.",
)
# 工作空间目录的选项（隐藏选项，用于集成测试）
@click.option(
    "--workspace-directory",
    "-w",
    type=click.Path(file_okay=False),
    hidden=True,
)
# 安装第三方插件的外部依赖的选项
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
# 是否跳过启动时的最新消息输出的选项
@click.option(
    "--skip-news",
    is_flag=True,
    help="Specifies whether to suppress the output of latest news on startup.",
)
# 是否跳过脚本开始时的重新提示消息的选项
@click.option(
    "--skip-reprompt",
    "-y",
    is_flag=True,
    help="Skips the re-prompting messages at the beginning of the script",
)
# AI 设置的选项，指定配置文件的路径
@click.option(
    "--ai-settings",
    "-C",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    # 帮助信息，指定要使用的 ai_settings.yaml 文件，相对于 AutoGPT 根目录
    # 同时会自动跳过重新提示
# 定义一个点击命令行选项，用于指定 AI 的名称覆盖
@click.option(
    "--ai-name",
    type=str,
    help="AI name override",
)

# 定义一个点击命令行选项，用于指定 AI 的角色覆盖
@click.option(
    "--ai-role",
    type=str,
    help="AI role override",
)

# 定义一个点击命令行选项，用于指定要使用的 prompt_settings.yaml 文件的路径
@click.option(
    "--prompt-settings",
    "-P",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Specifies which prompt_settings.yaml file to use.",
)

# 定义一个点击命令行选项，用于添加或覆盖要包含在提示中的 AI 约束
@click.option(
    "--constraint",
    type=str,
    multiple=True,
    help=(
        "Add or override AI constraints to include in the prompt;"
        " may be used multiple times to pass multiple constraints"
    ),
)

# 定义一个点击命令行选项，用于添加或覆盖要包含在提示中的 AI 资源
@click.option(
    "--resource",
    type=str,
    multiple=True,
    help=(
        "Add or override AI resources to include in the prompt;"
        " may be used multiple times to pass multiple resources"
    ),
)

# 定义一个点击命令行选项，用于添加或覆盖要包含在提示中的 AI 最佳实践
@click.option(
    "--best-practice",
    type=str,
    multiple=True,
    help=(
        "Add or override AI best practices to include in the prompt;"
        " may be used multiple times to pass multiple best practices"
    ),
)

# 定义一个点击命令行选项，如果指定，则 --constraint、--resource 和 --best-practice 将覆盖 AI 的指令而不是追加到它们
@click.option(
    "--override-directives",
    is_flag=True,
    help=(
        "If specified, --constraint, --resource and --best-practice will override"
        " the AI's directives instead of being appended to them"
    ),
)

# 定义一个点击命令行选项，用于指定是否启用调试模式
@click.option(
    "--debug", is_flag=True, help="Implies --log-level=DEBUG --log-format=debug"
)

# 定义一个点击命令行选项，用于指定日志级别
@click.option("--log-level", type=click.Choice([*logLevelMap.keys()]))

# 定义一个点击命令行选项，用于选择日志格式
@click.option(
    "--log-format",
    help=(
        "Choose a log format; defaults to 'simple'."
        " Also implies --log-file-format, unless it is specified explicitly."
        " Using the 'structured_google_cloud' format disables log file output."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)

# 定义一个点击命令行选项，用于覆盖用于日志文件输出的格式
@click.option(
    "--log-file-format",
    help=(
        "Override the format used for the log file output."
        " Defaults to the application's global --log-format."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)

# 定义一个函数，用于运行程序，接受一个布尔类型的参数 continuous
def run(
    continuous: bool,
    # 连续对话的限制，可选参数
    continuous_limit: Optional[int],
    # 是否朗读输出
    speak: bool,
    # 是否仅使用 GPT-3
    gpt3only: bool,
    # 是否仅使用 GPT-4
    gpt4only: bool,
    # 浏览器名称，可选参数
    browser_name: Optional[str],
    # 是否允许下载
    allow_downloads: bool,
    # 工作空间目录，可选参数
    workspace_directory: Optional[Path],
    # 是否安装插件依赖
    install_plugin_deps: bool,
    # 是否跳过新闻
    skip_news: bool,
    # 是否跳过重新提示
    skip_reprompt: bool,
    # AI 设置，可选参数
    ai_settings: Optional[Path],
    # AI 名称，可选参数
    ai_name: Optional[str],
    # AI 角色，可选参数
    ai_role: Optional[str],
    # 提示设置，可选参数
    prompt_settings: Optional[Path],
    # 资源，字符串元组
    resource: tuple[str],
    # 约束，字符串元组
    constraint: tuple[str],
    # 最佳实践，字符串元组
    best_practice: tuple[str],
    # 是否覆盖指令
    override_directives: bool,
    # 是否调试模式
    debug: bool,
    # 日志级别，可选参数
    log_level: Optional[str],
    # 日志格式，可选参数
    log_format: Optional[str],
    # 日志文件格式，可选参数
    log_file_format: Optional[str],
# 定义一个函数，设置并运行一个代理，基于用户指定的任务，或者恢复一个已存在的代理
def run_agent(
    continuous: bool,
    continuous_limit: int,
    ai_settings: dict,
    prompt_settings: str,
    skip_reprompt: bool,
    speak: bool,
    debug: bool,
    log_level: str,
    log_format: str,
    log_file_format: str,
    gpt3only: bool,
    gpt4only: bool,
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
    workspace_directory: str,
    install_plugin_deps: bool,
    override_ai_name: str,
    override_ai_role: str,
    resource: list,
    constraint: list,
    best_practice: list,
    override_directives: str,
) -> None:
    """
    Sets up and runs an agent, based on the task specified by the user, or resumes an
    existing agent.
    """
    # 将导入语句放在函数内部，以避免在启动 CLI 时导入所有内容
    from autogpt.app.main import run_auto_gpt

    # 运行 AutoGPT
    run_auto_gpt(
        continuous=continuous,
        continuous_limit=continuous_limit,
        ai_settings=ai_settings,
        prompt_settings=prompt_settings,
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
        workspace_directory=workspace_directory,
        install_plugin_deps=install_plugin_deps,
        override_ai_name=override_ai_name,
        override_ai_role=override_ai_role,
        resources=list(resource),
        constraints=list(constraint),
        best_practices=list(best_practice),
        override_directives=override_directives,
    )

# 定义一个 CLI 命令
@cli.command()
# 添加一个选项，指定 prompt_settings.yaml 文件的路径
@click.option(
    "--prompt-settings",
    "-P",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Specifies which prompt_settings.yaml file to use.",
)
# 添加一个选项，启用 GPT3.5 Only 模式
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
# 添加一个选项，启用 GPT4 Only 模式
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
# 添加一个选项，指定使用的浏览器
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
# 添加一个选项，允许 AutoGPT 原生下载文件
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows AutoGPT to download files natively.",
)
# 添加一个选项，安装第三方插件的外部依赖
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
# 添加一个选项，启用调试模式
@click.option(
    "--debug", is_flag=True, help="Implies --log-level=DEBUG --log-format=debug"
)
# 添加命令行选项，用于设置日志级别，选项值为预定义的日志级别
@click.option("--log-level", type=click.Choice([*logLevelMap.keys()]))
# 添加命令行选项，用于设置日志格式，可以选择不同的日志格式
@click.option(
    "--log-format",
    help=(
        "Choose a log format; defaults to 'simple'."
        " Also implies --log-file-format, unless it is specified explicitly."
        " Using the 'structured_google_cloud' format disables log file output."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
# 添加命令行选项，用于设置日志文件格式，可以覆盖全局的日志格式设置
@click.option(
    "--log-file-format",
    help=(
        "Override the format used for the log file output."
        " Defaults to the application's global --log-format."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
# 定义 serve 函数，启动符合 Agent Protocol 的 AutoGPT 服务器，为每个任务创建一个自定义代理
def serve(
    prompt_settings: Optional[Path],
    gpt3only: bool,
    gpt4only: bool,
    browser_name: Optional[str],
    allow_downloads: bool,
    install_plugin_deps: bool,
    debug: bool,
    log_level: Optional[str],
    log_format: Optional[str],
    log_file_format: Optional[str],
) -> None:
    """
    Starts an Agent Protocol compliant AutoGPT server, which creates a custom agent for
    every task.
    """
    # 将导入语句放在函数内部，以避免在启动 CLI 时导入所有内容
    from autogpt.app.main import run_auto_gpt_server

    # 运行 AutoGPT 服务器，传入相关参数
    run_auto_gpt_server(
        prompt_settings=prompt_settings,
        debug=debug,
        log_level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        gpt3only=gpt3only,
        gpt4only=gpt4only,
        browser_name=browser_name,
        allow_downloads=allow_downloads,
        install_plugin_deps=install_plugin_deps,
    )

# 如果当前脚本被直接执行，则调用 cli 函数
if __name__ == "__main__":
    cli()
```