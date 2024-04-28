# `.\transformers\commands\env.py`

```py
# 导入必要的模块
import importlib.util
import os
import platform
from argparse import ArgumentParser

# 导入 Hugging Face Hub 库
import huggingface_hub

# 导入当前包的版本信息
from .. import __version__ as version

# 导入检查加速库是否可用的函数
from ..utils import (
    is_accelerate_available,
    is_flax_available,
    is_safetensors_available,
    is_tf_available,
    is_torch_available,
)

# 导入基础命令类
from . import BaseTransformersCLICommand


# 创建环境信息命令的工厂函数
def info_command_factory(_):
    return EnvironmentCommand()


# 创建下载命令的工厂函数
def download_command_factory(args):
    return EnvironmentCommand(args.accelerate_config_file)


# 环境信息命令类
class EnvironmentCommand(BaseTransformersCLICommand):
    # 注册子命令
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 创建 'env' 子命令的解析器
        download_parser = parser.add_parser("env")
        # 设置默认函数为获取环境信息的工厂函数
        download_parser.set_defaults(func=info_command_factory)
        # 添加 '--accelerate-config_file' 参数，用于指定加速库的配置文件
        download_parser.add_argument(
            "--accelerate-config_file",
            default=None,
            help="The accelerate config file to use for the default values in the launching script.",
        )
        # 设置默认函数为下载命令的工厂函数
        download_parser.set_defaults(func=download_command_factory)

    # 初始化函数
    def __init__(self, accelerate_config_file, *args) -> None:
        # 存储加速库的配置文件路径
        self._accelerate_config_file = accelerate_config_file

    # 格式化字典为字符串
    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
```