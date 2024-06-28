# `.\models\byt5\convert_byt5_original_tf_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2018 The T5 authors and HuggingFace Inc. team.
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
"""Convert T5 checkpoint."""


import argparse  # 导入 argparse 模块，用于解析命令行参数

from transformers import T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5  # 导入转换所需的类和函数
from transformers.utils import logging  # 导入日志工具


logging.set_verbosity_info()  # 设置日志输出级别为 INFO


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = T5Config.from_json_file(config_file)  # 从配置文件加载 T5 模型配置
    print(f"Building PyTorch model from configuration: {config}")  # 打印正在根据配置构建 PyTorch 模型
    model = T5ForConditionalGeneration(config)  # 使用配置初始化 T5 条件生成模型

    # Load weights from tf checkpoint
    load_tf_weights_in_t5(model, config, tf_checkpoint_path)  # 从 TensorFlow checkpoint 中加载权重到 PyTorch 模型

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")  # 打印正在将 PyTorch 模型保存到指定路径
    model.save_pretrained(pytorch_dump_path)  # 保存预训练好的 PyTorch 模型到指定路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器

    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )  # 添加参数：TensorFlow checkpoint 的路径
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained T5 model. \nThis specifies the model architecture."
        ),
    )  # 添加参数：预训练 T5 模型对应的配置 JSON 文件路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )  # 添加参数：输出 PyTorch 模型的路径
    args = parser.parse_args()  # 解析命令行参数
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)  # 转换 TensorFlow checkpoint 到 PyTorch 模型
```