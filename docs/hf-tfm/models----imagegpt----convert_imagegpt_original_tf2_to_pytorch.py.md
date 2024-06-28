# `.\models\imagegpt\convert_imagegpt_original_tf2_to_pytorch.py`

```
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
"""Convert OpenAI Image GPT checkpoints."""


import argparse  # 导入 argparse 模块，用于解析命令行参数

import torch  # 导入 PyTorch 库

from transformers import ImageGPTConfig, ImageGPTForCausalLM, load_tf_weights_in_imagegpt
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging  # 导入相关类和函数


logging.set_verbosity_info()  # 设置日志输出级别为信息


def convert_imagegpt_checkpoint_to_pytorch(imagegpt_checkpoint_path, model_size, pytorch_dump_folder_path):
    # Construct configuration depending on size
    MODELS = {"small": (512, 8, 24), "medium": (1024, 8, 36), "large": (1536, 16, 48)}
    n_embd, n_head, n_layer = MODELS[model_size]  # 根据给定的模型大小设置模型超参数
    config = ImageGPTConfig(n_embd=n_embd, n_layer=n_layer, n_head=n_head)  # 构建 ImageGPT 的配置对象
    model = ImageGPTForCausalLM(config)  # 根据配置创建 ImageGPT 模型对象

    # Load weights from numpy
    load_tf_weights_in_imagegpt(model, config, imagegpt_checkpoint_path)  # 加载 TensorFlow 的权重到 PyTorch 模型中

    # Save pytorch-model
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME  # 设置 PyTorch 模型权重保存路径
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME  # 设置 PyTorch 模型配置保存路径
    print(f"Save PyTorch model to {pytorch_weights_dump_path}")  # 输出保存 PyTorch 模型权重的信息
    torch.save(model.state_dict(), pytorch_weights_dump_path)  # 保存 PyTorch 模型的权重
    print(f"Save configuration file to {pytorch_config_dump_path}")  # 输出保存配置文件的信息
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:  # 打开配置文件路径，准备写入配置
        f.write(config.to_json_string())  # 将配置对象转换为 JSON 字符串并写入文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # Required parameters
    parser.add_argument(
        "--imagegpt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path.",  # TensorFlow 检查点路径，必填参数
    )
    parser.add_argument(
        "--model_size",
        default=None,
        type=str,
        required=True,
        help="Size of the model (can be either 'small', 'medium' or 'large').",  # 模型大小，必填参数
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model.",  # 输出 PyTorch 模型路径，必填参数
    )
    args = parser.parse_args()  # 解析命令行参数
    convert_imagegpt_checkpoint_to_pytorch(
        args.imagegpt_checkpoint_path, args.model_size, args.pytorch_dump_folder_path
    )  # 调用函数将 TensorFlow 检查点转换为 PyTorch 模型并保存
```