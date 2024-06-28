# `.\models\roberta_prelayernorm\convert_roberta_prelayernorm_original_pytorch_checkpoint_to_pytorch.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert RoBERTa-PreLayerNorm checkpoint."""


import argparse  # 导入用于解析命令行参数的模块

import torch  # 导入PyTorch库
from huggingface_hub import hf_hub_download  # 从huggingface_hub模块中导入hf_hub_download函数

from transformers import AutoTokenizer, RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM  # 导入transformers库中的相关类和函数
from transformers.utils import logging  # 导入logging模块


logging.set_verbosity_info()  # 设置日志输出级别为info
logger = logging.get_logger(__name__)  # 获取当前模块的logger对象


def convert_roberta_prelayernorm_checkpoint_to_pytorch(checkpoint_repo: str, pytorch_dump_folder_path: str):
    """
    Copy/paste/tweak roberta_prelayernorm's weights to our BERT structure.
    将roberta_prelayernorm的权重复制/粘贴/调整到我们的BERT结构中。
    """
    # convert configuration
    config = RobertaPreLayerNormConfig.from_pretrained(
        checkpoint_repo, architectures=["RobertaPreLayerNormForMaskedLM"]
    )  # 从预训练模型路径加载配置信息，指定模型架构为RobertaPreLayerNormForMaskedLM

    # convert state_dict
    original_state_dict = torch.load(hf_hub_download(repo_id=checkpoint_repo, filename="pytorch_model.bin"))
    # 使用hf_hub_download函数下载指定checkpoint_repo和"pytorch_model.bin"的模型文件，并加载为原始state_dict
    state_dict = {}
    for tensor_key, tensor_value in original_state_dict.items():
        # The transformer implementation gives the model a unique name, rather than overwiriting 'roberta'
        # 转换器实现中给模型一个唯一的名称，而不是覆盖 'roberta'
        if tensor_key.startswith("roberta."):
            tensor_key = "roberta_prelayernorm." + tensor_key[len("roberta.") :]
            # 如果tensor_key以"roberta."开头，则替换为"roberta_prelayernorm."
        
        # The original implementation contains weights which are not used, remove them from the state_dict
        # 原始实现包含未使用的权重，从state_dict中移除它们
        if tensor_key.endswith(".self.LayerNorm.weight") or tensor_key.endswith(".self.LayerNorm.bias"):
            continue
            # 如果tensor_key以".self.LayerNorm.weight"或".self.LayerNorm.bias"结尾，则跳过不处理

        state_dict[tensor_key] = tensor_value  # 将处理后的tensor_key和对应的tensor_value加入state_dict

    model = RobertaPreLayerNormForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=None, config=config, state_dict=state_dict
    )
    # 使用from_pretrained方法根据配置和state_dict创建RobertaPreLayerNormForMaskedLM模型对象
    model.save_pretrained(pytorch_dump_folder_path)  # 将模型保存到指定的pytorch_dump_folder_path路径下

    # convert tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_repo)
    # 使用checkpoint_repo加载预训练的分词器模型
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    # 将分词器保存到指定的pytorch_dump_folder_path路径下


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    # Required parameters
    parser.add_argument(
        "--checkpoint-repo",
        default=None,
        type=str,
        required=True,
        help="Path the official PyTorch dump, e.g. 'andreasmadsen/efficient_mlm_m0.40'.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()  # 解析命令行参数
    convert_roberta_prelayernorm_checkpoint_to_pytorch(args.checkpoint_repo, args.pytorch_dump_folder_path)
    # 调用convert_roberta_prelayernorm_checkpoint_to_pytorch函数，进行模型转换和保存
```