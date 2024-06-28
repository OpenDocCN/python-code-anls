# `.\models\gpt2\convert_gpt2_original_tf_checkpoint_to_pytorch.py`

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
"""Convert OpenAI GPT checkpoint."""


import argparse  # 导入用于解析命令行参数的模块

import torch  # 导入 PyTorch 模块

from transformers import GPT2Config, GPT2Model, load_tf_weights_in_gpt2  # 导入 GPT2 相关类和函数
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging  # 导入配置文件和权重文件名常量，以及日志模块


logging.set_verbosity_info()  # 设置日志级别为 info


def convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path):
    # Construct model
    if gpt2_config_file == "":  # 如果没有提供配置文件路径，则使用默认配置创建 GPT2Config 对象
        config = GPT2Config()
    else:
        config = GPT2Config.from_json_file(gpt2_config_file)  # 从提供的 JSON 配置文件创建 GPT2Config 对象
    model = GPT2Model(config)  # 基于配置文件创建 GPT2Model 模型对象

    # Load weights from numpy
    load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path)  # 加载 TensorFlow 检查点中的权重到 PyTorch 模型中

    # Save pytorch-model
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME  # 构造 PyTorch 模型权重保存路径
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME  # 构造 PyTorch 模型配置保存路径
    print(f"Save PyTorch model to {pytorch_weights_dump_path}")  # 打印保存 PyTorch 模型权重的路径
    torch.save(model.state_dict(), pytorch_weights_dump_path)  # 将 PyTorch 模型的权重保存到指定路径
    print(f"Save configuration file to {pytorch_config_dump_path}")  # 打印保存配置文件的路径
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())  # 将模型配置对象转换为 JSON 字符串并写入文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器

    # Required parameters
    parser.add_argument(
        "--gpt2_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )  # 添加必需参数：TensorFlow 检查点路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )  # 添加必需参数：输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--gpt2_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained OpenAI model. \n"
            "This specifies the model architecture."
        ),
    )  # 添加可选参数：预训练 OpenAI 模型的配置 JSON 文件路径

    args = parser.parse_args()  # 解析命令行参数
    convert_gpt2_checkpoint_to_pytorch(args.gpt2_checkpoint_path, args.gpt2_config_file, args.pytorch_dump_folder_path)
```