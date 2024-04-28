# `.\transformers\commands\add_new_model.py`

```
# 引入所需模块
import json  # 导入用于处理 JSON 数据的模块
import os  # 导入用于处理操作系统相关功能的模块
import shutil  # 导入用于执行文件和目录操作的模块
import warnings  # 导入用于处理警告的模块
from argparse import ArgumentParser, Namespace  # 从 argparse 模块中导入 ArgumentParser 和 Namespace 类
from pathlib import Path  # 导入处理文件路径的模块
from typing import List  # 导入 List 类型用于类型提示

# 从当前目录中的 utils 模块中导入 logging 函数
from ..utils import logging
# 从当前目录中的 __init__.py 模块中导入 BaseTransformersCLICommand 类
from . import BaseTransformersCLICommand

# 尝试导入 cookiecutter 模块
try:
    from cookiecutter.main import cookiecutter
    _has_cookiecutter = True  # 若导入成功，则标记为 True
except ImportError:
    _has_cookiecutter = False  # 若导入失败，则标记为 False

logger = logging.get_logger(__name__)  # 获取当前文件的日志记录器对象，以 __name__ 作为标识符

# 定义一个函数，用于创建 AddNewModelCommand 实例
def add_new_model_command_factory(args: Namespace):
    return AddNewModelCommand(args.testing, args.testing_file, path=args.path)

# 定义一个子命令类 AddNewModelCommand，继承自 BaseTransformersCLICommand
class AddNewModelCommand(BaseTransformersCLICommand):
    # 静态方法，用于注册子命令
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_new_model_parser = parser.add_parser("add-new-model")  # 添加一个子命令解析器
        # 添加命令行参数
        add_new_model_parser.add_argument("--testing", action="store_true", help="If in testing mode.")  # 测试模式标志
        add_new_model_parser.add_argument("--testing_file", type=str, help="Configuration file on which to run.")  # 测试文件路径
        add_new_model_parser.add_argument("--path", type=str, help="Path to cookiecutter. Should only be used for testing purposes.")  # cookiecutter 路径
        add_new_model_parser.set_defaults(func=add_new_model_command_factory)  # 设置默认函数为 add_new_model_command_factory

    # 初始化方法
    def __init__(self, testing: bool, testing_file: str, path=None, *args):
        self._testing = testing  # 是否处于测试模式的标志
        self._testing_file = testing_file  # 测试文件路径
        self._path = path  # cookiecutter 路径
```