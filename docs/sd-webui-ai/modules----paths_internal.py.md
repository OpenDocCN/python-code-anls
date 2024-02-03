# `stable-diffusion-webui\modules\paths_internal.py`

```py
# 定义程序使用的内部路径，可以在 launch.py 中安装依赖之前安全导入
import argparse  # 导入用于解析命令行参数的模块
import os  # 导入用于操作系统相关功能的模块
import sys  # 导入用于访问 Python 解释器的变量和函数的模块
import shlex  # 导入用于解析命令行字符串的模块

# 从环境变量中获取命令行参数，如果不存在则为空字符串
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
# 将命令行参数转换为列表并添加到 sys.argv 中
sys.argv += shlex.split(commandline_args)

# 获取当前工作目录
cwd = os.getcwd()
# 获取当前模块的路径
modules_path = os.path.dirname(os.path.realpath(__file__))
# 获取当前脚本的路径
script_path = os.path.dirname(modules_path)

# 定义配置文件路径
sd_configs_path = os.path.join(script_path, "configs")
# 定义默认配置文件路径
sd_default_config = os.path.join(sd_configs_path, "v1-inference.yaml")
# 定义模型文件路径
sd_model_file = os.path.join(script_path, 'model.ckpt')
# 定义默认模型文件路径
default_sd_model_file = sd_model_file

# 解析命令行参数中的 --data-dir 标志，用作其他参数默认值的基础
parser_pre = argparse.ArgumentParser(add_help=False)
parser_pre.add_argument("--data-dir", type=str, default=os.path.dirname(modules_path), help="base path where all user data is stored", )
cmd_opts_pre = parser_pre.parse_known_args()[0]

# 获取数据路径
data_path = cmd_opts_pre.data_dir

# 定义模型路径
models_path = os.path.join(data_path, "models")
# 定义扩展路径
extensions_dir = os.path.join(data_path, "extensions")
# 定义内置扩展路径
extensions_builtin_dir = os.path.join(script_path, "extensions-builtin")
# 定义配置状态路径
config_states_dir = os.path.join(script_path, "config_states")

# 定义 Roboto 字体文件路径
roboto_ttf_file = os.path.join(modules_path, 'Roboto-Regular.ttf')
```