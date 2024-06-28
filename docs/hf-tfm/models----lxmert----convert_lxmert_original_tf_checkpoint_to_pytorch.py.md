# `.\models\lxmert\convert_lxmert_original_tf_checkpoint_to_pytorch.py`

```
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert LXMERT checkpoint."""


import argparse  # 导入 argparse 模块，用于处理命令行参数

import torch  # 导入 PyTorch 模块

from transformers import LxmertConfig, LxmertForPreTraining, load_tf_weights_in_lxmert  # 从 transformers 模块导入相关函数和类
from transformers.utils import logging  # 导入 logging 模块

logging.set_verbosity_info()  # 设置日志输出级别为 info


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = LxmertConfig.from_json_file(config_file)  # 从配置文件加载 LXMERT 模型配置
    print(f"Building PyTorch model from configuration: {config}")  # 打印正在根据配置构建 PyTorch 模型
    model = LxmertForPreTraining(config)  # 基于配置创建 LXMERT 的预训练模型实例

    # Load weights from tf checkpoint
    load_tf_weights_in_lxmert(model, config, tf_checkpoint_path)  # 加载 TensorFlow checkpoint 中的权重到 PyTorch 模型中

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")  # 打印正在保存 PyTorch 模型到指定路径
    torch.save(model.state_dict(), pytorch_dump_path)  # 将 PyTorch 模型的状态字典保存到指定路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器实例

    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )  # 添加 tf_checkpoint_path 参数，指定 TensorFlow checkpoint 的路径
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.",
    )  # 添加 config_file 参数，指定预训练模型对应的配置文件路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )  # 添加 pytorch_dump_path 参数，指定输出的 PyTorch 模型路径
    args = parser.parse_args()  # 解析命令行参数

    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)  # 调用函数将 TensorFlow checkpoint 转换为 PyTorch 模型
```