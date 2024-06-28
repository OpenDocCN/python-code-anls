# `.\commands\transformers_cli.py`

```py
# 指定 Python 解释器的位置，并添加版权声明
#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入命令行参数解析工具
from argparse import ArgumentParser

# 导入各个命令模块
from .add_new_model import AddNewModelCommand
from .add_new_model_like import AddNewModelLikeCommand
from .convert import ConvertCommand
from .download import DownloadCommand
from .env import EnvironmentCommand
from .lfs import LfsCommands
from .pt_to_tf import PTtoTFCommand
from .run import RunCommand
from .serving import ServeCommand
from .user import UserCommands

# 定义主函数
def main():
    # 创建参数解析器对象，并设置程序的名称和用法说明
    parser = ArgumentParser("Transformers CLI tool", usage="transformers-cli <command> [<args>]")
    # 添加子命令的解析器
    commands_parser = parser.add_subparsers(help="transformers-cli command helpers")

    # 注册各个命令的子命令
    ConvertCommand.register_subcommand(commands_parser)
    DownloadCommand.register_subcommand(commands_parser)
    EnvironmentCommand.register_subcommand(commands_parser)
    RunCommand.register_subcommand(commands_parser)
    ServeCommand.register_subcommand(commands_parser)
    UserCommands.register_subcommand(commands_parser)
    AddNewModelCommand.register_subcommand(commands_parser)
    AddNewModelLikeCommand.register_subcommand(commands_parser)
    LfsCommands.register_subcommand(commands_parser)
    PTtoTFCommand.register_subcommand(commands_parser)

    # 检查是否有有效的命令函数被调用
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # 运行选定的命令函数，并获取其服务对象
    service = args.func(args)
    service.run()

# 如果该脚本被作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```