# `.\pytorch\torch\distributed\_tensor\examples\comm_mode_features_example_argparser.py`

```py
# 导入 argparse 库，用于解析命令行参数
import argparse

# 创建 ArgumentParser 对象，并设置描述信息和格式化类
parser = argparse.ArgumentParser(
    description="comm_mode_feature examples",
    formatter_class=argparse.RawTextHelpFormatter,
)

# 设置命令行参数提示信息的字符串
example_prompt = (
    "choose one comm_mode_feature example from below:\n"
    "\t1. MLP_distributed_sharding_display\n"
    "\t2. MLPStacked_distributed_sharding_display\n"
    "\t3. MLP_module_tracing\n"
    "\t4. transformer_module_tracing\n"
    "e.g. you want to try the MLPModule sharding display example, please input 'MLP_distributed_sharding_display'\n"
)

# 添加一个命令行参数选项，指定其参数名 "-e" 或 "--example"，帮助信息为 example_prompt，且必须提供此参数
parser.add_argument("-e", "--example", help=example_prompt, required=True)

# 解析命令行参数，并获取 example 参数的值
example = parser.parse_args().example

# 定义一个函数 args()，返回命令行参数 example 的值
def args() -> str:
    return example
```