# `.\models\pix2struct\convert_pix2struct_original_pytorch_to_hf.py`

```py
# coding=utf-8
# 设置文件编码为 UTF-8，确保可以正确处理中文和其他特殊字符

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 HuggingFace Inc. 团队所有，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License 2.0 版本进行许可，允许在符合条件的情况下使用本文件

# you may not use this file except in compliance with the License.
# 除非符合许可证的条件，否则不得使用本文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，无任何明示或暗示的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 详细了解许可证的具体条款和限制，请参阅许可证

import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统功能模块
import re  # 导入正则表达式模块

import torch  # 导入 PyTorch 模块
from flax.traverse_util import flatten_dict  # 从 Flax 模块中导入 flatten_dict 函数
from t5x import checkpoints  # 从 t5x 模块中导入 checkpoints 函数

from transformers import (  # 从 transformers 模块中导入以下类和函数
    AutoTokenizer,
    Pix2StructConfig,
    Pix2StructForConditionalGeneration,
    Pix2StructImageProcessor,
    Pix2StructProcessor,
    Pix2StructTextConfig,
    Pix2StructVisionConfig,
)


def get_flax_param(t5x_checkpoint_path):
    # 定义函数 get_flax_param，接收一个 t5x_checkpoint_path 参数
    flax_params = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    # 使用 t5x 模块的 load_t5x_checkpoint 函数加载 Flax 参数
    flax_params = flatten_dict(flax_params)
    # 调用 flatten_dict 函数，将 Flax 参数字典扁平化
    return flax_params


def rename_and_convert_flax_params(flax_dict):
    # 定义函数 rename_and_convert_flax_params，接收一个 flax_dict 参数
    converted_dict = {}  # 创建一个空字典 converted_dict

    # 定义两个转换映射字典，用于将 Flax 参数映射到特定的命名约定
    CONVERSION_MAPPING = {
        "token_embedder": "embeddings",
        "encoder_norm": "layernorm",
        "kernel": "weight",
        ".out": ".output",
        "scale": "weight",
        "embedders_0.pos_embedding": "row_embedder.weight",
        "embedders_1.pos_embedding": "column_embedder.weight",
    }

    DECODER_CONVERSION_MAPPING = {
        "query": "attention.query",
        "key": "attention.key",
        "value": "attention.value",
        "output.dense": "output",
        "encoder_decoder_attention.o": "encoder_decoder_attention.attention.o",
        "pre_self_attention_layer_norm": "self_attention.layer_norm",
        "pre_cross_attention_layer_norm": "encoder_decoder_attention.layer_norm",
        "mlp.": "mlp.DenseReluDense.",
        "pre_mlp_layer_norm": "mlp.layer_norm",
        "self_attention.o": "self_attention.attention.o",
        "decoder.embeddings.embedding": "decoder.embed_tokens.weight",
        "decoder.relpos_bias.rel_embedding": "decoder.layer.0.self_attention.attention.relative_attention_bias.weight",
        "decoder.decoder_norm.weight": "decoder.final_layer_norm.weight",
        "decoder.logits_dense.weight": "decoder.lm_head.weight",
    }
    # 遍历原始字典的键
    for key in flax_dict.keys():
        # 检查是否键名中包含 "target"
        if "target" in key:
            # 移除键名中的第一个前缀
            new_key = ".".join(key[1:])

            # 使用转换映射表重命名键名
            for old, new in CONVERSION_MAPPING.items():
                new_key = new_key.replace(old, new)

            # 如果新键名中包含 "decoder"
            if "decoder" in new_key:
                # 使用解码器转换映射表进一步处理键名
                for old, new in DECODER_CONVERSION_MAPPING.items():
                    new_key = new_key.replace(old, new)

            # 如果新键名中包含 "layers" 但不包含 "decoder"
            if "layers" in new_key and "decoder" not in new_key:
                # 使用正则表达式替换层号码格式
                new_key = re.sub(r"layers_(\d+)", r"layer.\1", new_key)
                # 替换 "encoder" 为 "encoder.encoder"
                new_key = new_key.replace("encoder", "encoder.encoder")

            # 如果新键名中包含 "layers" 并且包含 "decoder"
            elif "layers" in new_key and "decoder" in new_key:
                # 使用正则表达式替换层号码格式
                new_key = re.sub(r"layers_(\d+)", r"layer.\1", new_key)

            # 将处理过的键值对加入转换后的字典中
            converted_dict[new_key] = flax_dict[key]

    # 初始化一个空的转换后的 torch 字典
    converted_torch_dict = {}

    # 将转换后的字典转换成 torch 格式
    for key in converted_dict.keys():
        # 检查键名中不包含 "embed_tokens" 且不包含 "embedder"
        if ("embed_tokens" not in key) and ("embedder" not in key):
            # 将 numpy 数组转换为 torch 张量并转置
            converted_torch_dict[key] = torch.from_numpy(converted_dict[key].T)
        else:
            # 将 numpy 数组转换为 torch 张量
            converted_torch_dict[key] = torch.from_numpy(converted_dict[key])

    # 返回转换后的 torch 字典
    return converted_torch_dict
def convert_pix2struct_original_pytorch_checkpoint_to_hf(
    t5x_checkpoint_path, pytorch_dump_folder_path, use_large=False, is_vqa=False
):
    # 从T5x模型的检查点路径获取Flax参数
    flax_params = get_flax_param(t5x_checkpoint_path)

    # 根据是否使用大型模型选择编码器和解码器配置
    if not use_large:
        encoder_config = Pix2StructVisionConfig()
        decoder_config = Pix2StructTextConfig()
    else:
        encoder_config = Pix2StructVisionConfig(
            hidden_size=1536, d_ff=3968, num_attention_heads=24, num_hidden_layers=18
        )
        decoder_config = Pix2StructTextConfig(hidden_size=1536, d_ff=3968, num_heads=24, num_layers=18)

    # 创建Pix2Struct模型配置对象
    config = Pix2StructConfig(
        vision_config=encoder_config.to_dict(), text_config=decoder_config.to_dict(), is_vqa=is_vqa
    )

    # 根据配置创建Pix2StructForConditionalGeneration模型
    model = Pix2StructForConditionalGeneration(config)

    # 将Flax参数转换并加载到PyTorch模型中
    torch_params = rename_and_convert_flax_params(flax_params)
    model.load_state_dict(torch_params)

    # 加载预训练的分词器
    tok = AutoTokenizer.from_pretrained("ybelkada/test-pix2struct-tokenizer")

    # 创建Pix2StructImageProcessor实例
    image_processor = Pix2StructImageProcessor()

    # 创建Pix2StructProcessor处理器实例，使用给定的图像处理器和分词器
    processor = Pix2StructProcessor(image_processor=image_processor, tokenizer=tok)

    # 如果使用大型模型，则设置图像处理器的最大补丁数为4096
    if use_large:
        processor.image_processor.max_patches = 4096

    # 设置图像处理器的is_vqa属性为True
    processor.image_processor.is_vqa = True

    # 如果需要的话，创建输出目录
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    # 将模型保存为PyTorch模型
    model.save_pretrained(pytorch_dump_folder_path)

    # 保存处理器的预训练状态到指定目录
    processor.save_pretrained(pytorch_dump_folder_path)

    # 打印保存成功的消息
    print("Model saved in {}".format(pytorch_dump_folder_path))


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5x_checkpoint_path", default=None, type=str, help="Path to the original T5x checkpoint.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--use_large", action="store_true", help="Use large model.")
    parser.add_argument("--is_vqa", action="store_true", help="Use large model.")
    args = parser.parse_args()

    # 调用函数，将Pix2Struct的原始PyTorch检查点转换为Hugging Face模型
    convert_pix2struct_original_pytorch_checkpoint_to_hf(
        args.t5x_checkpoint_path, args.pytorch_dump_folder_path, args.use_large
    )
```