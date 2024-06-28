# `.\models\roformer\convert_roformer_original_tf_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Convert RoFormer checkpoint."""


import argparse  # 导入解析命令行参数的模块

import torch  # 导入 PyTorch 深度学习框架

from transformers import RoFormerConfig, RoFormerForMaskedLM, load_tf_weights_in_roformer  # 导入 RoFormer 模型相关的类和函数
from transformers.utils import logging  # 导入日志记录模块


logging.set_verbosity_info()  # 设置日志输出级别为 info


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # 初始化 PyTorch 模型
    config = RoFormerConfig.from_json_file(bert_config_file)
    print(f"Building PyTorch model from configuration: {config}")  # 打印配置信息
    model = RoFormerForMaskedLM(config)  # 基于配置构建 RoFormer 的 Masked LM 模型

    # 从 TensorFlow checkpoint 加载权重
    load_tf_weights_in_roformer(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器

    # 必需参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )

    args = parser.parse_args()  # 解析命令行参数
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)  # 执行转换操作
```