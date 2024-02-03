# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\client_lib\shared_click_commands.py`

```py
# 导入 pathlib 模块
import pathlib

# 导入 click 模块
import click

# 默认设置文件路径，使用 pathlib.Path 扩展用户目录
DEFAULT_SETTINGS_FILE = str(
    pathlib.Path("~/auto-gpt/default_agent_settings.yml").expanduser()
)

# 定义命令行函数
@click.command()
# 添加命令行选项 --settings-file，类型为 click.Path，默认为 DEFAULT_SETTINGS_FILE
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
# 定义 make_settings 函数，参数为 settings_file，返回 None
def make_settings(settings_file: str) -> None:
    # 导入 make_user_configuration 函数
    from autogpt.core.runner.client_lib.settings import make_user_configuration

    # 调用 make_user_configuration 函数，传入 settings_file 转换为 pathlib.Path 对象
    make_user_configuration(pathlib.Path(settings_file))
```