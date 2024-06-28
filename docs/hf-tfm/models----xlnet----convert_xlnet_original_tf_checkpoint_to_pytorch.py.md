# `.\models\xlnet\convert_xlnet_original_tf_checkpoint_to_pytorch.py`

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

import argparse  # 导入命令行参数解析库
import os  # 导入操作系统相关功能的库

import torch  # 导入PyTorch

from transformers import (  # 导入transformers库中的相关类和函数
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetForSequenceClassification,
    XLNetLMHeadModel,
    load_tf_weights_in_xlnet,
)
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging  # 从transformers.utils中导入配置名、权重名和日志功能


GLUE_TASKS_NUM_LABELS = {  # GLUE任务到标签数量的映射字典
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

logging.set_verbosity_info()  # 设置日志的详细程度为info


def convert_xlnet_checkpoint_to_pytorch(
    tf_checkpoint_path, bert_config_file, pytorch_dump_folder_path, finetuning_task=None
):
    # Initialise PyTorch model 初始化PyTorch模型
    config = XLNetConfig.from_json_file(bert_config_file)

    finetuning_task = finetuning_task.lower() if finetuning_task is not None else ""  # 将微调任务名称转换为小写（如果指定了的话）
    if finetuning_task in GLUE_TASKS_NUM_LABELS:  # 如果微调任务在GLUE任务标签数量字典中
        print(f"Building PyTorch XLNetForSequenceClassification model from configuration: {config}")
        config.finetuning_task = finetuning_task  # 设置配置对象的微调任务属性
        config.num_labels = GLUE_TASKS_NUM_LABELS[finetuning_task]  # 设置配置对象的标签数量属性
        model = XLNetForSequenceClassification(config)  # 创建XLNet序列分类模型
    elif "squad" in finetuning_task:  # 如果微调任务名称中包含"squad"
        config.finetuning_task = finetuning_task  # 设置配置对象的微调任务属性
        model = XLNetForQuestionAnswering(config)  # 创建XLNet问答模型
    else:
        model = XLNetLMHeadModel(config)  # 创建XLNet语言建模头模型

    # Load weights from tf checkpoint 从TensorFlow的checkpoint中加载权重
    load_tf_weights_in_xlnet(model, config, tf_checkpoint_path)

    # Save pytorch-model 保存PyTorch模型
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)  # 拼接PyTorch权重保存路径
    pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)  # 拼接PyTorch配置保存路径
    print(f"Save PyTorch model to {os.path.abspath(pytorch_weights_dump_path)}")  # 打印保存PyTorch权重模型的路径
    torch.save(model.state_dict(), pytorch_weights_dump_path)  # 保存PyTorch模型的权重
    print(f"Save configuration file to {os.path.abspath(pytorch_config_dump_path)}")  # 打印保存配置文件的路径
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())  # 将配置对象转换成JSON字符串并写入文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # Required parameters 必要的参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--xlnet_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained XLNet model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加命令行参数 --xlnet_config_file，用于指定预训练XLNet模型的配置文件路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to store the PyTorch model or dataset/vocab.",
    )
    # 添加命令行参数 --pytorch_dump_folder_path，指定存储PyTorch模型或数据集/词汇表的文件夹路径
    parser.add_argument(
        "--finetuning_task",
        default=None,
        type=str,
        help="Name of a task on which the XLNet TensorFlow model was fine-tuned",
    )
    # 添加命令行参数 --finetuning_task，用于指定XLNet TensorFlow模型进行微调的任务名称
    args = parser.parse_args()
    # 解析命令行参数，并将其存储在args变量中
    print(args)

    convert_xlnet_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.xlnet_config_file, args.pytorch_dump_folder_path, args.finetuning_task
    )
    # 调用函数convert_xlnet_checkpoint_to_pytorch，传递解析后的参数，执行XLNet模型的转换操作
```