# `.\AutoGPT\autogpts\autogpt\autogpt\commands\execute_code.py`

```py
# 导入所需的库
import logging
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.containers import Container as DockerContainer

# 导入自定义的类和异常
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import (
    CodeExecutionError,
    CommandExecutionError,
    InvalidArgumentError,
    OperationNotAllowedError,
)
from autogpt.command_decorator import command
from autogpt.config import Config
from autogpt.core.utils.json_schema import JSONSchema

# 导入本地的装饰器
from .decorators import sanitize_path_arg

# 定义命令类别和标题
COMMAND_CATEGORY = "execute_code"
COMMAND_CATEGORY_TITLE = "Execute Code"

# 设置日志记录器
logger = logging.getLogger(__name__)

# 定义控制白名单和黑名单的常量
ALLOWLIST_CONTROL = "allowlist"
DENYLIST_CONTROL = "denylist"

# 命令装饰器，用于执行给定的 Python 代码
@command(
    "execute_python_code",
    "Executes the given Python code inside a single-use Docker container"
    " with access to your workspace folder",
    {
        "code": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The Python code to run",
            required=True,
        ),
    },
)
def execute_python_code(code: str, agent: Agent) -> str:
    """
    Create and execute a Python file in a Docker container and return the STDOUT of the
    executed code.

    If the code generates any data that needs to be captured, use a print statement.

    Args:
        code (str): The Python code to run.
        agent (Agent): The Agent executing the command.

    Returns:
        str: The STDOUT captured from the code when it ran.
    """

    # 创建临时的 Python 文件，写入给定的代码
    tmp_code_file = NamedTemporaryFile(
        "w", dir=agent.workspace.root, suffix=".py", encoding="utf-8"
    )
    tmp_code_file.write(code)
    tmp_code_file.flush()

    try:
        # 调用执行 Python 文件的函数，并返回执行结果
        return execute_python_file(tmp_code_file.name, agent)  # type: ignore
    except Exception as e:
        # 捕获异常并抛出自定义的命令执行错误
        raise CommandExecutionError(*e.args)
    finally:
        # 关闭临时文件
        tmp_code_file.close()
# 命令装饰器，用于执行一个现有的 Python 文件，将其放在一个一次性的 Docker 容器中，并且可以访问您的工作区文件夹
@command(
    "execute_python_file",
    "Execute an existing Python file inside a single-use Docker container"
    " with access to your workspace folder",
    {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to execute",
            required=True,
        ),
        "args": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            description="The (command line) arguments to pass to the script",
            required=False,
            items=JSONSchema(type=JSONSchema.Type.STRING),
        ),
    },
)
# 对文件名参数进行路径清理
@sanitize_path_arg("filename")
def execute_python_file(
    filename: Path, agent: Agent, args: list[str] | str = []
) -> str:
    """执行一个 Python 文件在 Docker 容器中，并返回输出

    Args:
        filename (Path): 要执行的文件名
        args (list, optional): 用于运行 python 脚本的参数

    Returns:
        str: 文件的输出
    """
    # 记录日志，显示正在执行的 Python 文件和工作目录
    logger.info(
        f"Executing python file '{filename}' "
        f"in working directory '{agent.workspace.root}'"
    )

    # 如果参数是字符串，则将其转换为列表
    if isinstance(args, str):
        args = args.split()  # Convert space-separated string to a list

    # 如果文件名不以 .py 结尾，则抛出无效文件类型的错误
    if not str(filename).endswith(".py"):
        raise InvalidArgumentError("Invalid file type. Only .py files are allowed.")

    # 获取文件路径
    file_path = filename
    # 如果文件不存在，则抛出文件未找到的错误
    if not file_path.is_file():
        # 模拟命令行的响应，使其对 LLM 更直观理解
        raise FileNotFoundError(
            f"python: can't open file '{filename}': [Errno 2] No such file or directory"
        )
    # 检查是否在 Docker 容器中运行
    if we_are_running_in_a_docker_container():
        # 如果在 Docker 容器中运行，记录调试信息
        logger.debug(
            "AutoGPT is running in a Docker container; "
            f"executing {file_path} directly..."
        )
        # 在 Docker 容器中执行指定的 Python 脚本文件
        result = subprocess.run(
            ["python", "-B", str(file_path)] + args,
            capture_output=True,
            encoding="utf8",
            cwd=str(agent.workspace.root),
        )
        # 如果执行成功，返回标准输出
        if result.returncode == 0:
            return result.stdout
        # 如果执行失败，抛出代码执行错误
        else:
            raise CodeExecutionError(result.stderr)

    # 如果不在 Docker 容器中运行，记录调试信息
    logger.debug("AutoGPT is not running in a Docker container")
    # 捕获 Docker 异常并处理
    except DockerException as e:
        # 记录警告信息，提示安装 Docker
        logger.warning(
            "Could not run the script in a container. "
            "If you haven't already, please install Docker: "
            "https://docs.docker.com/get-docker/"
        )
        # 抛出命令执行错误，包含异常信息
        raise CommandExecutionError(f"Could not run the script in a container: {e}")
# 验证命令以确保其被允许
def validate_command(command: str, config: Config) -> bool:
    """Validate a command to ensure it is allowed

    Args:
        command (str): The command to validate
        config (Config): The config to use to validate the command

    Returns:
        bool: True if the command is allowed, False otherwise
    """
    # 如果命令为空，则返回 False
    if not command:
        return False

    # 获取命令的名称部分
    command_name = command.split()[0]

    # 根据配置中的控制方式，判断命令是否在允许列表中
    if config.shell_command_control == ALLOWLIST_CONTROL:
        return command_name in config.shell_allowlist
    else:
        return command_name not in config.shell_denylist


# 命令装饰器，用于执行 Shell 命令，仅支持非交互式命令
@command(
    "execute_shell",
    "Execute a Shell Command, non-interactive commands only",
    {
        "command_line": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The command line to execute",
            required=True,
        )
    },
    enabled=lambda config: config.execute_local_commands,
    disabled_reason="You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config file: .env - do not attempt to bypass the restriction.",
)
# 执行 Shell 命令并返回输出
def execute_shell(command_line: str, agent: Agent) -> str:
    """Execute a shell command and return the output

    Args:
        command_line (str): The command line to execute

    Returns:
        str: The output of the command
    """
    # 如果命令不被允许，则记录日志并抛出异常
    if not validate_command(command_line, agent.legacy_config):
        logger.info(f"Command '{command_line}' not allowed")
        raise OperationNotAllowedError("This shell command is not allowed.")

    # 获取当前工作目录
    current_dir = Path.cwd()
    # 如果当前目录不是代理的工作目录，则切换到工作目录
    if not current_dir.is_relative_to(agent.workspace.root):
        os.chdir(agent.workspace.root)

    # 记录执行命令的信息
    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    # 运行 Shell 命令并捕获输出
    result = subprocess.run(command_line, capture_output=True, shell=True)
    # 将 result.stdout 和 result.stderr 解码为字符串，并拼接成一个包含 STDOUT 和 STDERR 的输出字符串
    output = f"STDOUT:\n{result.stdout.decode()}\nSTDERR:\n{result.stderr.decode()}"

    # 切换回之前的工作目录
    os.chdir(current_dir)

    # 返回包含 STDOUT 和 STDERR 的输出字符串
    return output
# 定义一个名为 execute_shell_popen 的命令，用于执行非交互式的 Shell 命令
# 参数包括 command_line，表示要执行的命令行
# 如果要执行本地 Shell 命令，必须在配置中将 EXECUTE_LOCAL_COMMANDS 设置为 'True'，否则不允许执行
@command(
    "execute_shell_popen",
    "Execute a Shell Command, non-interactive commands only",
    {
        "command_line": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The command line to execute",
            required=True,
        )
    },
    lambda config: config.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config. Do not attempt to bypass the restriction.",
)
def execute_shell_popen(command_line: str, agent: Agent) -> str:
    """执行一个 Shell 命令，并使用 Popen 返回事件的英文描述和进程 ID

    Args:
        command_line (str): 要执行的命令行

    Returns:
        str: 进程启动的描述和其 ID
    """
    # 验证命令是否允许执行
    if not validate_command(command_line, agent.legacy_config):
        logger.info(f"Command '{command_line}' not allowed")
        raise OperationNotAllowedError("This shell command is not allowed.")

    # 获取当前工作目录
    current_dir = Path.cwd()
    # 如果当前目录不是相对于工作空间根目录的，则切换到工作空间根目录
    if not current_dir.is_relative_to(agent.workspace.root):
        os.chdir(agent.workspace.root)

    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    # 不显示输出
    do_not_show_output = subprocess.DEVNULL
    # 使用 Popen 执行命令
    process = subprocess.Popen(
        command_line, shell=True, stdout=do_not_show_output, stderr=do_not_show_output
    )

    # 切换回之前的工作目录
    os.chdir(current_dir)

    return f"Subprocess started with PID:'{str(process.pid)}'"


# 检查当前是否在 Docker 容器中运行
def we_are_running_in_a_docker_container() -> bool:
    """Check if we are running in a Docker container

    Returns:
        bool: True if we are running in a Docker container, False otherwise
    """
    return os.path.exists("/.dockerenv")
```