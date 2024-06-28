# `.\models\electra\convert_electra_original_tf_checkpoint_to_pytorch.py`

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
"""Convert ELECTRA checkpoint."""


import argparse  # 导入处理命令行参数的模块

import torch  # 导入PyTorch库

from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra  # 导入transformers相关模块
from transformers.utils import logging  # 导入logging模块


logging.set_verbosity_info()  # 设置日志级别为INFO


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator):
    # Initialise PyTorch model
    config = ElectraConfig.from_json_file(config_file)  # 从配置文件加载ElectraConfig对象
    print(f"Building PyTorch model from configuration: {config}")  # 打印配置信息

    if discriminator_or_generator == "discriminator":  # 判断是判别器还是生成器模型
        model = ElectraForPreTraining(config)  # 构建ElectraForPreTraining模型
    elif discriminator_or_generator == "generator":
        model = ElectraForMaskedLM(config)  # 构建ElectraForMaskedLM模型
    else:
        raise ValueError("The discriminator_or_generator argument should be either 'discriminator' or 'generator'")  # 参数错误时抛出异常

    # Load weights from tf checkpoint
    load_tf_weights_in_electra(
        model, config, tf_checkpoint_path, discriminator_or_generator=discriminator_or_generator
    )  # 加载TensorFlow的权重到PyTorch模型中

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")  # 打印保存路径信息
    torch.save(model.state_dict(), pytorch_dump_path)  # 保存PyTorch模型的状态字典到指定路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )  # 添加tf_checkpoint_path参数，指定TensorFlow检查点路径
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.",
    )  # 添加config_file参数，指定预训练模型的配置文件路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )  # 添加pytorch_dump_path参数，指定输出的PyTorch模型路径
    parser.add_argument(
        "--discriminator_or_generator",
        default=None,
        type=str,
        required=True,
        help=(
            "Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or "
            "'generator'."
        ),
    )  # 添加discriminator_or_generator参数，指定导出的是生成器还是判别器
    args = parser.parse_args()  # 解析命令行参数
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.discriminator_or_generator
    )  # 调用函数进行TensorFlow模型到PyTorch模型的转换
```