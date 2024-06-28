# `.\models\canine\convert_canine_original_tf_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert CANINE checkpoint."""


import argparse  # 导入用于解析命令行参数的模块

from transformers import CanineConfig, CanineModel, CanineTokenizer, load_tf_weights_in_canine  # 导入转换和处理 CANINE 模型的相关模块
from transformers.utils import logging  # 导入日志记录模块


logging.set_verbosity_info()  # 设置日志输出级别为 info


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path):
    # 初始化 PyTorch 模型配置
    config = CanineConfig()
    # 基于配置初始化 CANINE 模型
    model = CanineModel(config)
    # 设置模型为评估模式（不进行训练）
    model.eval()

    print(f"Building PyTorch model from configuration: {config}")

    # 从 TensorFlow checkpoint 加载权重到 CANINE 模型
    load_tf_weights_in_canine(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型（包括权重和配置）
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # 保存 tokenizer 文件
    tokenizer = CanineTokenizer()
    print(f"Save tokenizer files to {pytorch_dump_path}")
    tokenizer.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必选参数
    parser.add_argument(
        "--tf_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint. Should end with model.ckpt",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to a folder where the PyTorch model will be placed.",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 转换 TensorFlow checkpoint 到 PyTorch 模型
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.pytorch_dump_path)
```