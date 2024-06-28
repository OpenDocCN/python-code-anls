# `.\models\longformer\convert_longformer_original_pytorch_lightning_to_pytorch.py`

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
"""Convert RoBERTa checkpoint."""


import argparse  # 导入 argparse 模块，用于解析命令行参数

import pytorch_lightning as pl  # 导入 PyTorch Lightning 库
import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块

from transformers import LongformerForQuestionAnswering, LongformerModel  # 从 transformers 库导入 Longformer 模型和问答模型


class LightningModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model  # 初始化 Lightning 模型
        self.num_labels = 2  # 设置标签数量
        self.qa_outputs = nn.Linear(self.model.config.hidden_size, self.num_labels)  # 初始化问答输出层

    # implement only because lightning requires to do so
    def forward(self):
        pass


def convert_longformer_qa_checkpoint_to_pytorch(
    longformer_model: str, longformer_question_answering_ckpt_path: str, pytorch_dump_folder_path: str
):
    # load longformer model from model identifier
    longformer = LongformerModel.from_pretrained(longformer_model)  # 加载预训练的 Longformer 模型
    lightning_model = LightningModel(longformer)  # 初始化 Lightning 模型

    ckpt = torch.load(longformer_question_answering_ckpt_path, map_location=torch.device("cpu"))  # 加载 PyTorch Lightning 检查点
    lightning_model.load_state_dict(ckpt["state_dict"])  # 载入模型权重

    # init longformer question answering model
    longformer_for_qa = LongformerForQuestionAnswering.from_pretrained(longformer_model)  # 初始化 Longformer 问答模型

    # transfer weights
    longformer_for_qa.longformer.load_state_dict(lightning_model.model.state_dict())  # 转移 Longformer 权重
    longformer_for_qa.qa_outputs.load_state_dict(lightning_model.qa_outputs.state_dict())  # 转移问答输出层权重
    longformer_for_qa.eval()  # 设置为评估模式

    # save model
    longformer_for_qa.save_pretrained(pytorch_dump_folder_path)  # 保存 PyTorch 模型至指定路径

    print(f"Conversion successful. Model saved under {pytorch_dump_folder_path}")  # 打印成功转换信息


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    # Required parameters
    parser.add_argument(
        "--longformer_model",
        default=None,
        type=str,
        required=True,
        help="model identifier of longformer. Should be either `longformer-base-4096` or `longformer-large-4096`.",
    )  # 添加 Longformer 模型标识参数
    parser.add_argument(
        "--longformer_question_answering_ckpt_path",
        default=None,
        type=str,
        required=True,
        help="Path the official PyTorch Lightning Checkpoint.",
    )  # 添加 PyTorch Lightning 检查点路径参数
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )  # 添加 PyTorch 模型保存路径参数
    args = parser.parse_args()  # 解析命令行参数
    # 调用函数 convert_longformer_qa_checkpoint_to_pytorch 将给定的 Longformer QA 模型的检查点文件转换为 PyTorch 格式
    convert_longformer_qa_checkpoint_to_pytorch(
        args.longformer_model, args.longformer_question_answering_ckpt_path, args.pytorch_dump_folder_path
    )
```