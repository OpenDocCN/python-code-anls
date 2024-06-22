# `.\transformers\commands\transformers_cli.py`

```py
#!/usr/bin/env python
# 指定脚本解释器为 Python

# 版权声明
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

from argparse import ArgumentParser
# 导入 ArgumentParser 类

from .add_new_model import AddNewModelCommand
# 从 add_new_model 模块导入 AddNewModelCommand 类
from .add_new_model_like import AddNewModelLikeCommand
# 从 add_new_model_like 模块导入 AddNewModelLikeCommand 类
from .convert import ConvertCommand
# 从 convert 模块导入 ConvertCommand 类
from .download import DownloadCommand
# 从 download 模块导入 DownloadCommand 类
from .env import EnvironmentCommand
# 从 env 模块导入 EnvironmentCommand 类
from .lfs import LfsCommands
# 从 lfs 模块导入 LfsCommands 类
from .pt_to_tf import PTtoTFCommand
# 从 pt_to_tf 模块导入 PTtoTFCommand 类
from .run import RunCommand
# 从 run 模块导入 RunCommand 类
from .serving import ServeCommand
# 从 serving 模块导入 ServeCommand 类
from .user import UserCommands
# 从 user 模块导入 UserCommands 类

def main():
    # 创建 ArgumentParser 对象，设置程序名称和用法说明
    parser = ArgumentParser("Transformers CLI tool", usage="transformers-cli <command> [<args>]")
    # 添加子命令解析器
    commands_parser = parser.add_subparsers(help="transformers-cli command helpers")

    # 注册命令
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

    # 开始执行
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # 运行命令
    service = args.func(args)
    service.run()

if __name__ == "__main__":
    main()
```