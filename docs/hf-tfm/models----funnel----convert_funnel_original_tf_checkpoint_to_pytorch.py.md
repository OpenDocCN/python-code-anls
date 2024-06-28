# `.\models\funnel\convert_funnel_original_tf_checkpoint_to_pytorch.py`

```
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert Funnel checkpoint."""


import argparse  # 导入解析命令行参数的模块

import torch  # 导入PyTorch模块

from transformers import FunnelBaseModel, FunnelConfig, FunnelModel, load_tf_weights_in_funnel  # 导入Transformers库中相关类和函数
from transformers.utils import logging  # 导入Transformers库中的日志模块


logging.set_verbosity_info()  # 设置日志输出级别为info


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, base_model):
    # Initialise PyTorch model
    config = FunnelConfig.from_json_file(config_file)  # 从配置文件加载Funnel模型的配置
    print(f"Building PyTorch model from configuration: {config}")  # 打印正在根据配置构建PyTorch模型的消息
    model = FunnelBaseModel(config) if base_model else FunnelModel(config)  # 根据base_model参数选择性地创建基础模型或完整模型

    # Load weights from tf checkpoint
    load_tf_weights_in_funnel(model, config, tf_checkpoint_path)  # 加载TensorFlow的checkpoint中的权重到PyTorch模型中

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")  # 打印正在保存PyTorch模型到指定路径的消息
    torch.save(model.state_dict(), pytorch_dump_path)  # 将PyTorch模型的状态字典保存到指定路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器

    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )  # 添加必需的命令行参数：TensorFlow checkpoint的路径
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.",
    )  # 添加必需的命令行参数：配置JSON文件的路径，指定预训练模型的架构
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )  # 添加必需的命令行参数：输出PyTorch模型的路径
    parser.add_argument(
        "--base_model", action="store_true", help="Whether you want just the base model (no decoder) or not."
    )  # 添加可选的命令行参数：是否只需要基础模型（没有解码器）

    args = parser.parse_args()  # 解析命令行参数
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.base_model
    )  # 调用转换函数，传入命令行参数
```