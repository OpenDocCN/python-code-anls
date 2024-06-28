# `.\models\rembert\convert_rembert_tf_checkpoint_to_pytorch.py`

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
"""Convert RemBERT checkpoint."""


import argparse  # 导入命令行参数解析模块

import torch  # 导入 PyTorch 深度学习库

from transformers import RemBertConfig, RemBertModel, load_tf_weights_in_rembert  # 导入转换所需的类和函数
from transformers.utils import logging  # 导入日志模块


logging.set_verbosity_info()  # 设置日志输出级别为信息


def convert_rembert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # 初始化 PyTorch 模型
    config = RemBertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))  # 打印模型配置信息
    model = RemBertModel(config)

    # 从 TensorFlow checkpoint 加载权重
    load_tf_weights_in_rembert(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器

    # 必需参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--rembert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained RemBERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()  # 解析命令行参数
    convert_rembert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.rembert_config_file, args.pytorch_dump_path)
```