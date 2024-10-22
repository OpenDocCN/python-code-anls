# `.\diffusers\commands\diffusers_cli.py`

```py
# 指定解释器路径
#!/usr/bin/env python
# 版权信息，表明版权归 HuggingFace 团队所有
# 版权声明，受 Apache 2.0 许可协议约束
#
# 在使用此文件前必须遵循许可证的规定
# 许可证获取地址
#
# 软件在无任何保证的情况下按 "原样" 发行
# 参见许可证以了解权限和限制

# 导入命令行参数解析器
from argparse import ArgumentParser

# 导入环境命令模块
from .env import EnvironmentCommand
# 导入 FP16 Safetensors 命令模块
from .fp16_safetensors import FP16SafetensorsCommand

# 定义主函数
def main():
    # 创建命令行解析器，设置程序名称和使用说明
    parser = ArgumentParser("Diffusers CLI tool", usage="diffusers-cli <command> [<args>]")
    # 添加子命令解析器，帮助信息
    commands_parser = parser.add_subparsers(help="diffusers-cli command helpers")

    # 注册环境命令为子命令
    EnvironmentCommand.register_subcommand(commands_parser)
    # 注册 FP16 Safetensors 命令为子命令
    FP16SafetensorsCommand.register_subcommand(commands_parser)

    # 解析命令行参数
    args = parser.parse_args()

    # 检查是否有有效的命令函数
    if not hasattr(args, "func"):
        # 打印帮助信息并退出
        parser.print_help()
        exit(1)

    # 执行指定的命令函数
    service = args.func(args)
    # 运行服务
    service.run()

# 如果当前文件是主程序，则执行主函数
if __name__ == "__main__":
    main()
```