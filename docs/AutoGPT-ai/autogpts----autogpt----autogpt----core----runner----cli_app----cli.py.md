# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\cli_app\cli.py`

```py
# 导入必要的模块
from pathlib import Path
import click
import yaml
from autogpt.core.runner.cli_app.main import run_auto_gpt
from autogpt.core.runner.client_lib.shared_click_commands import (
    DEFAULT_SETTINGS_FILE,
    make_settings,
)
from autogpt.core.runner.client_lib.utils import coroutine, handle_exceptions

# 创建命令组 autogpt
@click.group()
def autogpt():
    """Temporary command group for v2 commands."""
    pass

# 将 make_settings 命令添加到 autogpt 命令组中
autogpt.add_command(make_settings)

# 定义 run 命令，接受 settings_file 和 pdb 两个参数
@autogpt.command()
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
@click.option(
    "--pdb",
    is_flag=True,
    help="Drop into a debugger if an error is raised.",
)
# 将函数标记为协程
@coroutine
async def run(settings_file: str, pdb: bool) -> None:
    """Run the AutoGPT agent."""
    # 输出提示信息
    click.echo("Running AutoGPT agent...")
    # 将 settings_file 转换为 Path 对象
    settings_file: Path = Path(settings_file)
    # 初始化 settings 字典
    settings = {}
    # 如果 settings_file 存在，则加载其中的内容到 settings 字典中
    if settings_file.exists():
        settings = yaml.safe_load(settings_file.read_text())
    # 创建 main 函数，处理异常并可选择是否启用调试器
    main = handle_exceptions(run_auto_gpt, with_debugger=pdb)
    # 调用 main 函数并传入 settings 参数
    await main(settings)

# 如果当前脚本被直接执行，则执行 autogpt 命令组
if __name__ == "__main__":
    autogpt()
```