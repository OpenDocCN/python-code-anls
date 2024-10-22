# `.\diffusers\commands\env.py`

```py
# 版权声明，标明版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，软件按“原样”分发，
# 不附有任何形式的担保或条件，无论是明示或暗示的。
# 有关许可证的具体条款，请参阅许可证。
#
# 导入 platform 模块，用于获取系统平台信息
import platform
# 导入 subprocess 模块，用于创建子进程
import subprocess
# 从 argparse 模块导入 ArgumentParser 类，用于处理命令行参数
from argparse import ArgumentParser

# 导入 huggingface_hub 库，提供与 Hugging Face Hub 交互的功能
import huggingface_hub

# 从上层包中导入版本信息
from .. import __version__ as version
# 从 utils 模块中导入多个可用性检查函数
from ..utils import (
    is_accelerate_available,      # 检查 accelerate 库是否可用
    is_bitsandbytes_available,     # 检查 bitsandbytes 库是否可用
    is_flax_available,             # 检查 flax 库是否可用
    is_google_colab,              # 检查当前环境是否为 Google Colab
    is_peft_available,             # 检查 peft 库是否可用
    is_safetensors_available,      # 检查 safetensors 库是否可用
    is_torch_available,            # 检查 torch 库是否可用
    is_transformers_available,     # 检查 transformers 库是否可用
    is_xformers_available,         # 检查 xformers 库是否可用
)
# 从当前包中导入 BaseDiffusersCLICommand 基类
from . import BaseDiffusersCLICommand


# 定义一个工厂函数，返回 EnvironmentCommand 的实例
def info_command_factory(_):
    return EnvironmentCommand()


# 定义 EnvironmentCommand 类，继承自 BaseDiffusersCLICommand
class EnvironmentCommand(BaseDiffusersCLICommand):
    # 注册子命令的方法，接收 ArgumentParser 对象
    @staticmethod
    def register_subcommand(parser: ArgumentParser) -> None:
        # 在解析器中添加名为 "env" 的子命令
        download_parser = parser.add_parser("env")
        # 设置默认的处理函数为 info_command_factory
        download_parser.set_defaults(func=info_command_factory)

    # 格式化字典的方法，将字典转换为字符串
    @staticmethod
    def format_dict(d: dict) -> str:
        # 以特定格式将字典内容转换为字符串
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
```