# `.\models\clvp\convert_clvp_to_hf.py`

```
# 设置编码方式为 UTF-8，确保脚本可以处理各种字符集
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

"""
CLVP权重转换脚本
"""

import argparse  # 导入处理命令行参数的模块
import os  # 导入操作系统功能的模块

import torch  # 导入PyTorch库
from huggingface_hub import hf_hub_download  # 从Hugging Face Hub下载模块

from transformers import ClvpConfig, ClvpModelForConditionalGeneration  # 导入CLVP模型相关组件


_MODELS = {
    "clvp": "https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/clvp2.pth",
    "decoder": "https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/autoregressive.pth",
}

dim = 1024  # 定义维度为1024的变量
sub_dim = dim // 16  # 计算子维度，为总维度除以16的结果

CLVP_ENCODERS_MAPPING = {
    "text_transformer.transformer.attn_layers": "text_encoder_model",
    "speech_transformer.transformer.attn_layers": "speech_encoder_model",
    "text_transformer.transformer.norm": "text_encoder_model.final_layer_norm",
    "speech_transformer.transformer.norm": "speech_encoder_model.final_layer_norm",
    "to_text_latent": "text_encoder_model.projection",
    "to_speech_latent": "speech_encoder_model.projection",
    "text_emb": "text_encoder_model.token_embedding",
    "speech_emb": "speech_encoder_model.token_embedding",
    "1.wrap.net.0": "mlp.fc1",
    "1.wrap.net.3": "mlp.fc2",
    "1.wrap": "self_attn",
    "to_out": "out_proj",
    "to_q": "q_proj",
    "to_k": "k_proj",
    "to_v": "v_proj",
    "temperature": "logit_scale",
}

CLVP_DECODER_MAPPING = {
    "conditioning_encoder.init": "conditioning_encoder.mel_conv",
    "conditioning_encoder.attn": "conditioning_encoder.mel_attn_blocks",
    "mel_attn_blocks": "group_norms",
    ".norm.weight": ".weight",
    ".norm.bias": ".bias",
    "text_embedding": "conditioning_encoder.text_token_embedding",
    "text_pos_embedding.emb": "conditioning_encoder.text_position_embedding",
    "final_norm": "speech_decoder_model.final_norm",
    "mel_head": "speech_decoder_model.lm_head",
    "gpt.ln_f": "speech_decoder_model.model.decoder.layer_norm",
    "mel_embedding": "speech_decoder_model.model.decoder.input_embeds_layer",
    "mel_pos_embedding.emb": "speech_decoder_model.model.decoder.position_embeds_layer",
    "gpt.h": "speech_decoder_model.model.decoder.layers",
    "ln_1": "input_layernorm",
    "ln_2": "post_attention_layernorm",
}


def update_index(present_index):
    # 如果给定索引为偶数，则返回其除以2的整数部分
    if present_index % 2 == 0:
        return int(present_index / 2)
    # 如果给定索引为奇数，则返回其减1后除以2的整数部分
    else:
        return int((present_index - 1) / 2)


def convert_encoder_weights(original_weights):
    converted_weights = {}
    # 对原始权重的键进行排序，以确保处理顺序一致性
    original_weights_keys = sorted(original_weights.keys())
    # 遍历排序后的原始权重键列表
    for original_key in original_weights_keys:
        # 初始化更新后的键为原始键
        updated_key = original_key
        
        # 替换特定模式的键名，根据条件替换为 "input_rmsnorm.weight" 或 "post_attention_rmsnorm.weight"
        if "0.0.g" in updated_key:
            # 提取特定位置的索引
            present_index = updated_key.split(".")[4]
            # 根据索引是否为偶数，决定替换为哪个新键名
            if int(present_index) % 2 == 0:
                updated_key = updated_key.replace("0.0.g", "input_rmsnorm.weight")
            else:
                updated_key = updated_key.replace("0.0.g", "post_attention_rmsnorm.weight")

        # 替换特定模式的键名，根据函数 update_index 处理索引更新
        if "transformer.attn_layers.layers" in updated_key:
            present_index = updated_key.split(".")[4]
            updated_index = update_index(int(present_index))
            updated_key = updated_key.replace(
                f"transformer.attn_layers.layers.{present_index}", f"transformer.attn_layers.layers.{updated_index}"
            )

        # 根据 CLVP_ENCODERS_MAPPING 字典替换键名中的特定字符串
        for k, v in CLVP_ENCODERS_MAPPING.items():
            if k in updated_key:
                updated_key = updated_key.replace(k, v)

        # 将更新后的键值对存入转换后的权重字典中，并从原始权重字典中移除原始键
        converted_weights[updated_key] = original_weights.pop(original_key)

    # 返回转换后的权重字典
    return converted_weights
# 定义一个函数，用于将原始权重转换为新的权重格式
def convert_decoder_weights(original_weights):
    # 创建一个空字典，用于存储转换后的权重
    converted_weights = {}
    # 获取原始权重字典的所有键，并按字母顺序排序
    original_weights_keys = sorted(original_weights.keys())
    # 返回转换后的权重字典
    return converted_weights


# 定义一个私有函数，用于从指定 URL 下载文件到指定路径
def _download(url: str, root: str):
    # 从 URL 提取仓库 ID 和文件名
    repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
    filename = f"{url.split('/')[-2]}/{url.split('/')[-1]}"
    # 调用函数从 Hugging Face Hub 下载文件到指定路径
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_filename=root,
        local_dir_use_symlinks=False,
    )


# 定义一个函数，用于转换 CLVP 模型的权重格式
def convert_clvp_weights(checkpoint_path, pytorch_dump_folder_path):
    # 创建一个空字典，用于存储转换后的检查点
    converted_checkpoint = {}

    # 遍历预定义的模型名称和其对应的下载 URL
    for each_model_name, each_model_url in _MODELS.items():
        # 构建每个模型文件的完整路径
        each_model_path = os.path.join(checkpoint_path, each_model_url.split("/")[-1])
        # 如果文件不存在，则下载该模型文件
        if not os.path.exists(each_model_path):
            print(f"\n{each_model_name} was not found! Downloading it to {each_model_path}")
            _download(url=each_model_url, root=each_model_path)

        # 根据模型名称选择加载对应的检查点文件
        if each_model_name == "clvp":
            clvp_checkpoint = torch.load(each_model_path, map_location="cpu")
        else:
            decoder_checkpoint = torch.load(each_model_path, map_location="cpu")

    # 将 CLVP 模型的编码器权重转换并更新到转换后的检查点中
    converted_checkpoint.update(**convert_encoder_weights(clvp_checkpoint))
    # 将解码器权重转换并更新到转换后的检查点中
    converted_checkpoint.update(**convert_decoder_weights(decoder_checkpoint))

    # 根据预训练配置创建 CLVP 模型配置对象
    config = ClvpConfig.from_pretrained("susnato/clvp_dev")
    # 根据配置对象创建条件生成用的 CLVP 模型
    model = ClvpModelForConditionalGeneration(config)

    # 加载转换后的检查点到模型中，严格模式
    model.load_state_dict(converted_checkpoint, strict=True)
    # 将模型保存到 PyTorch 转储文件夹路径中
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Model saved at {pytorch_dump_folder_path}!")


# 如果该脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数：检查点路径，指向已下载检查点的文件夹路径
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the folder of downloaded checkpoints. (Please enter full path)"
    )
    # 添加可选的参数：PyTorch 模型转储文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model. (Please enter full path)",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数，将 CLVP 模型的权重转换并保存为 PyTorch 模型
    convert_clvp_weights(args.checkpoint_path, args.pytorch_dump_folder_path)
```