# `.\commands\add_new_model.py`

```
# 导入必要的模块
import json
import os
import shutil
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

# 导入自定义的日志模块
from ..utils import logging
# 导入基础命令类
from . import BaseTransformersCLICommand

# 尝试导入cookiecutter模块，检查是否可用
try:
    from cookiecutter.main import cookiecutter
    _has_cookiecutter = True
except ImportError:
    _has_cookiecutter = False

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 工厂函数，创建添加新模型命令实例
def add_new_model_command_factory(args: Namespace):
    return AddNewModelCommand(args.testing, args.testing_file, path=args.path)

# 添加新模型命令类，继承自基础命令类
class AddNewModelCommand(BaseTransformersCLICommand):

    # 静态方法，用于注册子命令
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 添加"add-new-model"子命令及其参数
        add_new_model_parser = parser.add_parser("add-new-model")
        add_new_model_parser.add_argument("--testing", action="store_true", help="If in testing mode.")
        add_new_model_parser.add_argument("--testing_file", type=str, help="Configuration file on which to run.")
        add_new_model_parser.add_argument(
            "--path", type=str, help="Path to cookiecutter. Should only be used for testing purposes."
        )
        # 设置默认的命令处理函数为add_new_model_command_factory
        add_new_model_parser.set_defaults(func=add_new_model_command_factory)

    # 初始化方法，设置命令的属性
    def __init__(self, testing: bool, testing_file: str, path=None, *args):
        self._testing = testing
        self._testing_file = testing_file
        self._path = path
```