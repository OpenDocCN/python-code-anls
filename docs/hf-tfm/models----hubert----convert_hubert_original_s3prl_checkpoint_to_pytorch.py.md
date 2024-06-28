# `.\models\hubert\convert_hubert_original_s3prl_checkpoint_to_pytorch.py`

```py
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
"""Convert Hubert checkpoint."""


import argparse  # 导入 argparse 模块，用于处理命令行参数

import torch  # 导入 PyTorch 库

from transformers import HubertConfig, HubertForSequenceClassification, Wav2Vec2FeatureExtractor, logging  # 导入 transformers 相关类和 logging 模块


logging.set_verbosity_info()  # 设置日志记录的详细程度为 info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

SUPPORTED_MODELS = ["UtteranceLevel"]  # 支持的模型列表，当前仅支持 "UtteranceLevel" 模型


@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    """
    Copy/paste/tweak model's weights to transformers design.
    将模型的权重复制/粘贴/调整到 transformers 设计中。
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # 加载 s3prl 检查点文件到内存中，使用 CPU
    if checkpoint["Config"]["downstream_expert"]["modelrc"]["select"] not in SUPPORTED_MODELS:
        raise NotImplementedError(f"The supported s3prl models are {SUPPORTED_MODELS}")  # 如果不支持当前模型，则抛出 NotImplementedError

    downstream_dict = checkpoint["Downstream"]  # 从检查点中获取下游模型的字典信息

    hf_congfig = HubertConfig.from_pretrained(config_path)  # 从预训练配置路径加载 Hubert 模型配置
    hf_model = HubertForSequenceClassification.from_pretrained(base_model_name, config=hf_congfig)  # 从预训练模型名称加载 Hubert 分类模型
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        base_model_name, return_attention_mask=True, do_normalize=False
    )  # 从预训练模型名称加载 Wav2Vec2 特征提取器，并设置返回注意力掩码和不进行标准化

    if hf_congfig.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint["Featurizer"]["weights"]  # 如果配置要求使用加权层求和，则加载权重到模型中的层权重属性

    hf_model.projector.weight.data = downstream_dict["projector.weight"]  # 加载下游模型投影层的权重
    hf_model.projector.bias.data = downstream_dict["projector.bias"]  # 加载下游模型投影层的偏置
    hf_model.classifier.weight.data = downstream_dict["model.post_net.linear.weight"]  # 加载下游模型分类器的权重
    hf_model.classifier.bias.data = downstream_dict["model.post_net.linear.bias"]  # 加载下游模型分类器的偏置

    hf_feature_extractor.save_pretrained(model_dump_path)  # 将特征提取器的配置保存到指定路径
    hf_model.save_pretrained(model_dump_path)  # 将 Hubert 分类模型保存到指定路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument(
        "--base_model_name", default=None, type=str, help="Name of the huggingface pretrained base model."
    )  # 添加命令行参数：huggingface 预训练基础模型的名称
    parser.add_argument("--config_path", default=None, type=str, help="Path to the huggingface classifier config.")  # 添加命令行参数：huggingface 分类器配置文件的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the s3prl checkpoint.")  # 添加命令行参数：s3prl 检查点文件的路径
    parser.add_argument("--model_dump_path", default=None, type=str, help="Path to the final converted model.")  # 添加命令行参数：转换后模型的保存路径
    args = parser.parse_args()  # 解析命令行参数
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)  # 调用函数进行 s3prl 检查点转换
```