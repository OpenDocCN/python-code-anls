# `.\models\bert\convert_bert_original_tf_checkpoint_to_pytorch.py`

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
"""Convert BERT checkpoint."""


import argparse  # 导入用于处理命令行参数的模块

import torch  # 导入 PyTorch 库

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert  # 导入转换所需的类和函数
from transformers.utils import logging  # 导入日志记录工具


logging.set_verbosity_info()  # 设置日志记录的详细程度为 info


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # 初始化一个 PyTorch 模型
    config = BertConfig.from_json_file(bert_config_file)
    print(f"Building PyTorch model from configuration: {config}")  # 打印模型配置信息
    model = BertForPreTraining(config)  # 使用配置创建 BertForPreTraining 模型对象

    # 从 TensorFlow checkpoint 中加载权重
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")  # 打印保存路径信息
    torch.save(model.state_dict(), pytorch_dump_path)  # 将模型的状态字典保存到指定路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器对象
    # 必选参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )  # 添加 TensorFlow checkpoint 路径参数
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )  # 添加 BERT 配置文件路径参数
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )  # 添加输出 PyTorch 模型路径参数
    args = parser.parse_args()  # 解析命令行参数
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)  # 调用转换函数并传入参数
```