# `.\models\gptsan_japanese\convert_gptsan_tf_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明：2023 年由 HuggingFace Inc. 团队版权所有
#
# 根据 Apache 许可证 2.0 版本许可
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，不提供任何形式的保证或条件，无论是明示还是暗示。
# 请参阅许可证了解特定的语言权限和限制。
"""从原始仓库转换 GPTSANJapanese 检查点为 pytorch 模型"""

# 导入所需库和模块
import argparse  # 解析命令行参数的库
import json  # 处理 JSON 数据的库
import os  # 提供与操作系统交互的功能
from collections import OrderedDict  # 提供有序字典的库

import numpy as np  # 处理数组数据的库
import tensorflow as tf  # TensorFlow 深度学习框架
import torch  # PyTorch 深度学习框架


# 定义函数将 TensorFlow 模型转换为 PyTorch 模型
def convert_tf_gptsan_to_pt(args):
    # 拼接参数文件路径
    parameter_file = os.path.join(args.tf_model_dir, "parameters.json")
    # 加载参数文件
    params = json.loads(open(parameter_file).read())
    # 如果参数文件为空，则抛出 ValueError 异常
    if not params:
        raise ValueError(
            f"It seems that the json file at {parameter_file} is empty. Make sure you have a correct json file."
        )
    # 如果输出文件不以".pt"结尾，则添加".pt"后缀
    if not args.output.endswith(".pt"):
        args.output = args.output + ".pt"
    # 创建一个有序字典，用于存储转换后的模型参数
    new_state = OrderedDict()
    # 将转换后的模型参数保存为 PyTorch 模型文件
    torch.save(new_state, args.output)


# 如果当前脚本被直接执行，则运行以下代码
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="model converter.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 添加命令行参数：输入 TensorFlow 模型目录
    parser.add_argument("--tf_model_dir", metavar="PATH", type=str, required=True, help="import model")
    # 添加命令行参数：输出 PyTorch 模型文件路径
    parser.add_argument("--output", metavar="PATH", type=str, required=True, help="output model")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 TensorFlow 模型转换为 PyTorch 模型
    convert_tf_gptsan_to_pt(args)
```