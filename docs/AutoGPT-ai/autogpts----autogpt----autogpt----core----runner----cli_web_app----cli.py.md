# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\cli_web_app\cli.py`

```py
# 导入必要的模块
import pathlib
import click
import yaml
from agent_protocol import Agent as AgentProtocol
from autogpt.core.runner.cli_web_app.server.api import task_handler
from autogpt.core.runner.client_lib.shared_click_commands import (
    DEFAULT_SETTINGS_FILE,
    make_settings,
)
from autogpt.core.runner.client_lib.utils import coroutine

# 创建命令组 autogpt
@click.group()
def autogpt():
    """Temporary command group for v2 commands."""
    pass

# 将 make_settings 命令添加到 autogpt 命令组中
autogpt.add_command(make_settings)

# 定义 server 命令，接收一个端口参数
@autogpt.command()
@click.option(
    "port",
    "--port",
    default=8080,
    help="The port of the webserver.",
    type=click.INT,
)
def server(port: int) -> None:
    """Run the AutoGPT runner httpserver."""
    # 输出提示信息
    click.echo("Running AutoGPT runner httpserver...")
    # 启动 AgentProtocol 处理任务，并监听指定端口
    AgentProtocol.handle_task(task_handler).start(port)

# 定义 client 命令，接收一个设置文件参数
@autogpt.command()
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
@coroutine
async def client(settings_file) -> None:
    """Run the AutoGPT runner client."""
    # 将设置文件路径转换为 pathlib.Path 对象
    settings_file = pathlib.Path(settings_file)
    # 初始化设置字典
    settings = {}
    # 如果设置文件存在，加载其中的设置内容
    if settings_file.exists():
        settings = yaml.safe_load(settings_file.read_text())

    settings
    # TODO: Call the API server with the settings and task,
    #   using the Python API client for agent protocol.

# 如果当前脚本被直接执行，则调用 autogpt 命令组
if __name__ == "__main__":
    autogpt()
```