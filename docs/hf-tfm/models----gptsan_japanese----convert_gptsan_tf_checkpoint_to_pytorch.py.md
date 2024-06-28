# `.\models\gptsan_japanese\convert_gptsan_tf_checkpoint_to_pytorch.py`

```py
# 导入 argparse 模块，用于处理命令行参数解析
import argparse
# 导入 json 模块，用于读取和解析 JSON 格式的文件
import json
# 导入 os 模块，提供与操作系统交互的功能
import os
# 从 collections 模块中导入 OrderedDict 类，用于创建有序字典
from collections import OrderedDict

# 导入 numpy 库，一般用于科学计算，这里可能在后续的代码中使用
import numpy as np
# 导入 tensorflow 库，用于与 TensorFlow 模型相关的操作
import tensorflow as tf
# 导入 torch 库，用于与 PyTorch 模型相关的操作
import torch


# 定义函数 convert_tf_gptsan_to_pt，用于将 TensorFlow 模型转换为 PyTorch 模型
def convert_tf_gptsan_to_pt(args):
    # 构建参数文件的完整路径
    parameter_file = os.path.join(args.tf_model_dir, "parameters.json")
    # 读取并解析 JSON 格式的参数文件
    params = json.loads(open(parameter_file).read())
    # 如果参数文件为空，则抛出 ValueError 异常
    if not params:
        raise ValueError(
            f"It seems that the json file at {parameter_file} is empty. Make sure you have a correct json file."
        )
    # 如果输出路径不以 ".pt" 结尾，则自动添加 ".pt" 后缀
    if not args.output.endswith(".pt"):
        args.output = args.output + ".pt"
    # 创建一个空的有序字典 new_state
    new_state = OrderedDict()
    # 使用 PyTorch 的 torch.save 方法将 new_state 保存到指定的输出路径 args.output
    torch.save(new_state, args.output)


# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 创建 ArgumentParser 对象 parser，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description="model converter.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 添加命令行参数 --tf_model_dir，指定 TensorFlow 模型的路径，类型为字符串，必需参数
    parser.add_argument("--tf_model_dir", metavar="PATH", type=str, required=True, help="import model")
    # 添加命令行参数 --output，指定输出 PyTorch 模型的路径，类型为字符串，必需参数
    parser.add_argument("--output", metavar="PATH", type=str, required=True, help="output model")
    # 解析命令行参数，将结果存储在 args 对象中
    args = parser.parse_args()
    # 调用 convert_tf_gptsan_to_pt 函数，传入 args 对象，执行 TensorFlow 到 PyTorch 模型的转换
    convert_tf_gptsan_to_pt(args)
```