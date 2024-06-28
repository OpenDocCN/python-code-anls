# `.\commands\env.py`

```
# 导入所需的模块
import importlib.util
import os
import platform
from argparse import ArgumentParser

# 导入 Hugging Face Hub 库
import huggingface_hub

# 导入版本号
from .. import __version__ as version

# 导入一些实用函数
from ..utils import (
    is_accelerate_available,
    is_flax_available,
    is_safetensors_available,
    is_tf_available,
    is_torch_available,
)

# 导入基础命令类
from . import BaseTransformersCLICommand

# 定义一个工厂函数，用于创建环境命令对象
def info_command_factory(_):
    return EnvironmentCommand()

# 定义一个工厂函数，用于创建下载命令对象
def download_command_factory(args):
    return EnvironmentCommand(args.accelerate_config_file)

# 环境命令类，继承自基础 Transformers CLI 命令类
class EnvironmentCommand(BaseTransformersCLICommand):
    
    # 静态方法：注册子命令
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 添加一个名为 "env" 的子命令解析器
        download_parser = parser.add_parser("env")
        # 设置默认的命令函数为 info_command_factory
        download_parser.set_defaults(func=info_command_factory)
        # 添加参数：accelerate-config_file，用于指定加速配置文件的路径
        download_parser.add_argument(
            "--accelerate-config_file",
            default=None,
            help="The accelerate config file to use for the default values in the launching script.",
        )
        # 再次设置默认的命令函数为 download_command_factory
        download_parser.set_defaults(func=download_command_factory)
    
    # 初始化方法，接受加速配置文件作为参数
    def __init__(self, accelerate_config_file, *args) -> None:
        self._accelerate_config_file = accelerate_config_file

    # 静态方法：格式化字典为字符串，每个键值对前缀为 "-"，以换行连接
    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
```