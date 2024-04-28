# `.\transformers\models\roformer\convert_roformer_original_tf_checkpoint_to_pytorch.py`

```
# 该代码实现了将 TensorFlow 预训练的 RoFormer 模型权重转换为 PyTorch 模型
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


import argparse # 导入命令行参数解析库

import torch # 导入 PyTorch

from transformers import RoFormerConfig, RoFormerForMaskedLM, load_tf_weights_in_roformer # 从 transformers 库导入相关类和函数
from transformers.utils import logging # 从 transformers 库导入日志记录工具


logging.set_verbosity_info() # 设置日志级别为信息级别


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # 初始化 PyTorch 模型
    config = RoFormerConfig.from_json_file(bert_config_file) # 从 JSON 文件中读取 RoFormer 模型配置
    print(f"Building PyTorch model from configuration: {config}") # 打印正在构建的模型配置
    model = RoFormerForMaskedLM(config) # 创建 RoFormerForMaskedLM 模型

    # 从 TensorFlow 检查点加载权重
    load_tf_weights_in_roformer(model, config, tf_checkpoint_path) # 将 TensorFlow 检查点中的权重加载到 PyTorch 模型中

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}") # 打印正在保存模型的路径
    torch.save(model.state_dict(), pytorch_dump_path, _use_new_zipfile_serialization=False) # 保存 PyTorch 模型权重


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # 创建命令行参数解析器
    # 必需参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path." # 指定 TensorFlow 检查点路径
    )
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ), # 指定 BERT 模型配置文件路径
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model." # 指定 PyTorch 模型保存路径
    )
    args = parser.parse_args() # 解析命令行参数
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path) # 调用转换函数
```