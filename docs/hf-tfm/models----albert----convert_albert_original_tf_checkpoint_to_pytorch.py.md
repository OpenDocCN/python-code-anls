# `.\models\albert\convert_albert_original_tf_checkpoint_to_pytorch.py`

```py
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
"""Convert ALBERT checkpoint."""

import argparse  # 导入解析命令行参数的模块

import torch  # 导入PyTorch库

from ...utils import logging  # 导入日志模块
from . import AlbertConfig, AlbertForPreTraining, load_tf_weights_in_albert  # 导入Albert相关模块

logging.set_verbosity_info()  # 设置日志输出级别为INFO


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, albert_config_file, pytorch_dump_path):
    # Initialise PyTorch model 初始化PyTorch模型
    config = AlbertConfig.from_json_file(albert_config_file)  # 从JSON配置文件加载Albert模型配置
    print(f"Building PyTorch model from configuration: {config}")  # 打印正在构建的PyTorch模型的配置信息
    model = AlbertForPreTraining(config)  # 根据配置创建Albert预训练模型对象

    # Load weights from tf checkpoint 从TensorFlow的checkpoint加载权重
    load_tf_weights_in_albert(model, config, tf_checkpoint_path)

    # Save pytorch-model 保存PyTorch模型
    print(f"Save PyTorch model to {pytorch_dump_path}")  # 打印正在保存的PyTorch模型的路径
    torch.save(model.state_dict(), pytorch_dump_path)  # 将模型的状态字典保存到指定路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # Required parameters 必需的命令行参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--albert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained ALBERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()  # 解析命令行参数
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.albert_config_file, args.pytorch_dump_path)  # 调用转换函数进行转换
```