# `.\models\t5\convert_t5_original_tf_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2018 The T5 authors and HuggingFace Inc. team.
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
"""Convert T5 checkpoint."""


import argparse  # 导入解析命令行参数的模块

from transformers import T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5  # 导入转换模型相关的类和函数
from transformers.utils import logging  # 导入日志记录工具


logging.set_verbosity_info()  # 设置日志记录级别为INFO


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    # 初始化一个PyTorch模型配置
    config = T5Config.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")  # 打印正在根据配置构建PyTorch模型的消息
    model = T5ForConditionalGeneration(config)  # 使用配置创建T5条件生成模型

    # 从TensorFlow的检查点文件中加载权重
    load_tf_weights_in_t5(model, config, tf_checkpoint_path)

    # 保存PyTorch模型
    print(f"Save PyTorch model to {pytorch_dump_path}")  # 打印保存PyTorch模型到指定路径的消息
    model.save_pretrained(pytorch_dump_path)  # 将模型保存为PyTorch可用的格式


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器

    # 必填参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True,
        help="Path to the TensorFlow checkpoint path."  # TensorFlow检查点文件的路径
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained T5 model. \n"
            "This specifies the model architecture."
        ),  # 预训练T5模型对应的配置JSON文件，指定了模型的架构
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True,
        help="Path to the output PyTorch model."  # 输出PyTorch模型的路径
    )
    args = parser.parse_args()  # 解析命令行参数
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)  # 执行转换操作
```