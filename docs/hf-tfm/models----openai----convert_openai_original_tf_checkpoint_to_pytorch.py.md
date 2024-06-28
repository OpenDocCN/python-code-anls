# `.\models\openai\convert_openai_original_tf_checkpoint_to_pytorch.py`

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

import argparse  # 导入命令行参数解析模块

import torch  # 导入 PyTorch 库

from transformers import OpenAIGPTConfig, OpenAIGPTModel, load_tf_weights_in_openai_gpt  # 导入 transformers 库中的相关类和函数
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging  # 导入 transformers 库中的配置和日志模块


logging.set_verbosity_info()  # 设置日志级别为 INFO


def convert_openai_checkpoint_to_pytorch(openai_checkpoint_folder_path, openai_config_file, pytorch_dump_folder_path):
    # Construct model
    if openai_config_file == "":
        config = OpenAIGPTConfig()  # 如果没有指定配置文件，使用默认配置创建 OpenAIGPTConfig 对象
    else:
        config = OpenAIGPTConfig.from_json_file(openai_config_file)  # 否则，从指定的 JSON 文件中加载配置

    model = OpenAIGPTModel(config)  # 基于配置创建 OpenAIGPTModel 对象

    # Load weights from numpy
    load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path)  # 加载 TensorFlow checkpoint 中的权重到 PyTorch 模型中

    # Save pytorch-model
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME  # 设置 PyTorch 模型权重保存路径
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME  # 设置 PyTorch 模型配置保存路径
    print(f"Save PyTorch model to {pytorch_weights_dump_path}")  # 打印保存 PyTorch 模型权重的路径信息
    torch.save(model.state_dict(), pytorch_weights_dump_path)  # 保存 PyTorch 模型的权重
    print(f"Save configuration file to {pytorch_config_dump_path}")  # 打印保存配置文件的路径信息
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())  # 将模型配置以 JSON 格式写入配置文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # Required parameters
    parser.add_argument(
        "--openai_checkpoint_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path.",
    )  # 添加必需的参数：TensorFlow checkpoint 的路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model.",
    )  # 添加必需的参数：输出 PyTorch 模型的路径
    parser.add_argument(
        "--openai_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained OpenAI model. \n"
            "This specifies the model architecture."
        ),
    )  # 添加可选参数：预训练 OpenAI 模型的配置文件路径，指定模型架构
    args = parser.parse_args()  # 解析命令行参数
    convert_openai_checkpoint_to_pytorch(
        args.openai_checkpoint_folder_path, args.openai_config_file, args.pytorch_dump_folder_path
    )  # 调用函数，执行 OpenAI GPT 模型转换为 PyTorch 模型的操作
```