# `.\models\bloom\convert_bloom_original_checkpoint_to_pytorch.py`

```py
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
"""Convert BigScience BLOOM checkpoint."""

import argparse   # 导入处理命令行参数的模块
import json   # 导入处理 JSON 格式的模块
import os   # 提供与操作系统相关的功能
import re   # 导入正则表达式模块

import torch   # 导入 PyTorch 库

from transformers import BloomConfig, BloomModel   # 导入 BLOOM 模型相关类
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME   # 导入文件操作相关函数
from transformers.utils import logging   # 导入日志记录工具

logging.set_verbosity_info()   # 设置日志记录级别为 INFO

WEIGHTS_TO_AVERAGE_ENDSWITH = [   # 指定需要平均的权重名称列表
    "word_embeddings_layernorm.weight",
    "word_embeddings_layernorm.bias",
    "input_layernorm.weight",
    "input_layernorm.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "self_attention.dense.bias",
    "mlp.dense_4h_to_h.bias",
    "ln_f.weight",
    "ln_f.bias",
]

WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN = [   # 指定包含行并行性的权重名称列表
    "mlp.dense_4h_to_h.weight",
    "self_attention.dense.weight",
]


def layer_name_mapping(key, file):
    """Convert Megatron-DeepSpeed TP/PP weights mapping in transformers PP only"""
    # 处理第一个和最后一个层的名称映射
    layer_rename_map = {
        "word_embeddings.weight": "word_embeddings.weight",
        "word_embeddings.norm.weight": "word_embeddings_layernorm.weight",
        "word_embeddings.norm.bias": "word_embeddings_layernorm.bias",
        "weight": "ln_f.weight",
        "bias": "ln_f.bias",
    }

    if key in layer_rename_map:
        return layer_rename_map[key]

    # 处理 Transformer 块的名称映射
    layer_number = int(re.match(r".*layer_(\d*).*", file)[1])
    layer_number -= 3
    return f"h.{layer_number}." + key


def get_dtype_size(dtype):
    """获取数据类型的字节大小"""
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def convert_bloom_checkpoint_to_pytorch(
    bloom_checkpoint_path, bloom_config_file, pytorch_dump_folder_path, shard_model, pretraining_tp
):
    """将 BLOOM 模型的检查点文件转换为 PyTorch 模型"""
    # 构建模型配置
    if bloom_config_file == "":
        config = BloomConfig()   # 如果未提供配置文件，则使用默认配置
    else:
        config = BloomConfig.from_json_file(bloom_config_file)   # 使用提供的 JSON 配置文件

if __name__ == "__main__":
    parser = argparse.ArgumentParser()   # 创建参数解析器对象
    # Required parameters
    parser.add_argument(
        "--bloom_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the Megatron-LM checkpoint path.",
    )
    # 添加命令行参数 --pytorch_dump_folder_path，指定输出的PyTorch模型路径，参数为必填项
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加命令行参数 --bloom_config_file，指定预训练模型对应的配置JSON文件路径，可选项
    parser.add_argument(
        "--bloom_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加命令行参数 --shard_model，设置是否对输出的模型进行分片处理，可选项
    parser.add_argument(
        "--shard_model",
        action="store_true",
        help="An optional setting to shard the output model \nThis enables sharding the converted checkpoint",
    )
    # 添加命令行参数 --pretraining_tp，指定在Megatron-LM中训练模型时使用的预训练TP等级，默认为4，可选项
    parser.add_argument(
        "--pretraining_tp",
        default=4,
        type=int,
        help="Pretraining TP rank that has been used when training the model in Megatron-LM \n",
    )
    # 解析命令行参数，将结果存储在args对象中
    args = parser.parse_args()
    # 调用函数convert_bloom_checkpoint_to_pytorch，将参数传递给函数进行模型转换操作
    convert_bloom_checkpoint_to_pytorch(
        args.bloom_checkpoint_path,
        args.bloom_config_file,
        args.pytorch_dump_folder_path,
        args.shard_model,
        args.pretraining_tp,
    )
```