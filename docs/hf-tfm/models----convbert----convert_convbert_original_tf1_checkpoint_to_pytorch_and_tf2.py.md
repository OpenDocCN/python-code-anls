# `.\models\convbert\convert_convbert_original_tf1_checkpoint_to_pytorch_and_tf2.py`

```py
# coding=utf-8
# 设置文件编码为 UTF-8

# Copyright 2020 The HuggingFace Inc. team.
# 版权声明：2020 年由 HuggingFace Inc. 团队拥有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可协议（"许可协议"）授权使用

# you may not use this file except in compliance with the License.
# 除非符合许可协议，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在以下链接处获取许可协议的副本
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则依据"现状"分发软件，无论是明示的还是暗示的任何类型的保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 详细信息请参阅许可协议，以了解权限的具体限制和责任

"""Convert ConvBERT checkpoint."""
# 脚本用于将 ConvBERT 模型的 TensorFlow 1.x 检查点转换为 PyTorch 模型

import argparse
# 导入 argparse 模块，用于处理命令行参数

from transformers import ConvBertConfig, ConvBertModel, TFConvBertModel, load_tf_weights_in_convbert
# 从 transformers 库中导入 ConvBertConfig, ConvBertModel, TFConvBertModel 和 load_tf_weights_in_convbert 函数

from transformers.utils import logging
# 从 transformers.utils 模块中导入 logging 模块

logging.set_verbosity_info()
# 设置日志记录器的详细程度为 info 级别

def convert_orig_tf1_checkpoint_to_pytorch(tf_checkpoint_path, convbert_config_file, pytorch_dump_path):
    # 定义函数，用于将原始 TensorFlow 1.x 检查点转换为 PyTorch 模型

    conf = ConvBertConfig.from_json_file(convbert_config_file)
    # 从指定的 JSON 文件加载 ConvBertConfig 配置

    model = ConvBertModel(conf)
    # 使用加载的配置创建 ConvBertModel 模型对象

    model = load_tf_weights_in_convbert(model, conf, tf_checkpoint_path)
    # 加载 TensorFlow 检查点中的权重到 PyTorch 模型中

    model.save_pretrained(pytorch_dump_path)
    # 将转换后的 PyTorch 模型保存到指定路径

    tf_model = TFConvBertModel.from_pretrained(pytorch_dump_path, from_pt=True)
    # 从保存的 PyTorch 模型中创建 TFConvBertModel 对象，标记为来自 PyTorch

    tf_model.save_pretrained(pytorch_dump_path)
    # 将转换后的 TFConvBertModel 模型保存到指定路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 创建命令行参数解析器对象

    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    # 添加必需的命令行参数 --tf_checkpoint_path，指定 TensorFlow 检查点的路径

    parser.add_argument(
        "--convbert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained ConvBERT model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加必需的命令行参数 --convbert_config_file，指定预训练 ConvBERT 模型的配置 JSON 文件路径

    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加必需的命令行参数 --pytorch_dump_path，指定输出 PyTorch 模型的路径

    args = parser.parse_args()
    # 解析命令行参数

    convert_orig_tf1_checkpoint_to_pytorch(args.tf_checkpoint_path, args.convbert_config_file, args.pytorch_dump_path)
    # 调用转换函数，执行 TensorFlow 1.x 到 PyTorch 模型的转换操作
```