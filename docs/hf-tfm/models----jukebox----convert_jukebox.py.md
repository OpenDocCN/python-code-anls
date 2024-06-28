# `.\models\jukebox\convert_jukebox.py`

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
"""Convert Jukebox checkpoints"""

import argparse  # 导入处理命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
from pathlib import Path  # 导入处理路径相关操作的模块

import requests  # 导入发送 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习框架

from transformers import JukeboxConfig, JukeboxModel  # 导入 Jukebox 模型相关类
from transformers.utils import logging  # 导入日志记录模块


logging.set_verbosity_info()  # 设置日志记录级别为信息
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


PREFIX = "https://openaipublic.azureedge.net/jukebox/models/"  # Jukebox 模型下载地址的前缀
MODEL_MAPPING = {
    "jukebox-1b-lyrics": [
        "5b/vqvae.pth.tar",
        "5b/prior_level_0.pth.tar",
        "5b/prior_level_1.pth.tar",
        "1b_lyrics/prior_level_2.pth.tar",
    ],
    "jukebox-5b-lyrics": [
        "5b/vqvae.pth.tar",
        "5b/prior_level_0.pth.tar",
        "5b/prior_level_1.pth.tar",
        "5b_lyrics/prior_level_2.pth.tar",
    ],
}


def replace_key(key):
    if key.endswith(".model.1.bias") and len(key.split(".")) > 10:
        key = key.replace(".model.1.bias", ".conv1d_1.bias")  # 替换模型参数键名中的 ".model.1.bias" 为 ".conv1d_1.bias"
    elif key.endswith(".model.1.weight") and len(key.split(".")) > 10:
        key = key.replace(".model.1.weight", ".conv1d_1.weight")  # 替换模型参数键名中的 ".model.1.weight" 为 ".conv1d_1.weight"
    elif key.endswith(".model.3.bias") and len(key.split(".")) > 10:
        key = key.replace(".model.3.bias", ".conv1d_2.bias")  # 替换模型参数键名中的 ".model.3.bias" 为 ".conv1d_2.bias"
    elif key.endswith(".model.3.weight") and len(key.split(".")) > 10:
        key = key.replace(".model.3.weight", ".conv1d_2.weight")  # 替换模型参数键名中的 ".model.3.weight" 为 ".conv1d_2.weight"

    if "conditioner_blocks.0." in key:
        key = key.replace("conditioner_blocks.0", "conditioner_blocks")  # 替换模型参数键名中的 "conditioner_blocks.0" 为 "conditioner_blocks"

    if "prime_prior" in key:
        key = key.replace("prime_prior", "encoder")  # 替换模型参数键名中的 "prime_prior" 为 "encoder"

    if ".emb." in key and "total" not in key and "absolute" not in key and "relative" not in key:
        key = key.replace(".emb.", ".")  # 替换模型参数键名中的 ".emb." 为 "."

    if key.endswith("k"):  # 如果键名以 "k" 结尾，替换为以 "codebook" 结尾
        return key.replace(".k", ".codebook")
    if "y_emb." in key:
        return key.replace("y_emb.", "metadata_embedding.")  # 替换模型参数键名中的 "y_emb." 为 "metadata_embedding."

    if "x_emb.emb." in key:
        key = key.replace("0.x_emb.emb", "embed_tokens")  # 替换模型参数键名中的 "0.x_emb.emb" 为 "embed_tokens"

    if "prime_state_ln" in key:
        return key.replace("prime_state_ln", "encoder.final_layer_norm")  # 替换模型参数键名中的 "prime_state_ln" 为 "encoder.final_layer_norm"
    if ".ln" in key:
        return key.replace(".ln", ".layer_norm")  # 替换模型参数键名中的 ".ln" 为 ".layer_norm"
    if "_ln" in key:
        return key.replace("_ln", "_layer_norm")  # 替换模型参数键名中的 "_ln" 为 "_layer_norm"

    if "prime_state_proj" in key:
        return key.replace("prime_state_proj", "encoder.proj_in")  # 替换模型参数键名中的 "prime_state_proj" 为 "encoder.proj_in"
    if "prime_x_out" in key:
        return key.replace("prime_x_out", "encoder.lm_head")  # 替换模型参数键名中的 "prime_x_out" 为 "encoder.lm_head"
    # 如果字符串 "prior.x_out" 在 key 中，将 "x_out" 替换为 "fc_proj_out" 并返回替换后的结果
    if "prior.x_out" in key:
        return key.replace("x_out", "fc_proj_out")
    # 如果字符串 "x_emb" 在 key 中，将 "x_emb" 替换为 "embed_tokens" 并返回替换后的结果
    if "x_emb" in key:
        return key.replace("x_emb", "embed_tokens")

    # 如果以上条件都不满足，则返回 key 本身
    return key
def fix_jukebox_keys(state_dict, model_state_dict, key_prefix, mapping):
    # 初始化一个空字典，用于存储修复后的模型权重
    new_dict = {}
    # 导入正则表达式模块用于匹配模型权重的键名
    import re

    # 正则表达式用于匹配编码器块的卷积层输入
    re_encoder_block_conv_in = re.compile(r"encoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).(bias|weight)")
    # 正则表达式用于匹配编码器块的ResNet结构
    re_encoder_block_resnet = re.compile(
        r"encoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).model.(\d*).model.(\d*).(bias|weight)"
    )
    # 正则表达式用于匹配编码器块的投影输出
    re_encoder_block_proj_out = re.compile(r"encoders.(\d*).level_blocks.(\d*).model.(\d*).(bias|weight)")

    # 正则表达式用于匹配解码器块的卷积层输出
    re_decoder_block_conv_out = re.compile(r"decoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).(bias|weight)")
    # 正则表达式用于匹配解码器块的ResNet结构
    re_decoder_block_resnet = re.compile(
        r"decoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).model.(\d*).model.(\d*).(bias|weight)"
    )
    # 正则表达式用于匹配解码器块的投影输入
    re_decoder_block_proj_in = re.compile(r"decoders.(\d*).level_blocks.(\d*).model.(\d*).(bias|weight)")

    # 正则表达式用于匹配先验条件块的卷积层输出
    re_prior_cond_conv_out = re.compile(r"conditioner_blocks.(\d*).cond.model.(\d*).(\d).(bias|weight)")
    # 正则表达式用于匹配先验条件块的ResNet结构
    re_prior_cond_resnet = re.compile(
        r"conditioner_blocks.(\d*).cond.model.(\d*).(\d).model.(\d*).model.(\d*).(bias|weight)"
    )
    # 正则表达式用于匹配先验条件块的投影输入
    re_prior_cond_proj_in = re.compile(r"conditioner_blocks.(\d*).cond.model.(\d*).(bias|weight)")

    # 返回初始化的空字典
    return new_dict


@torch.no_grad()
def convert_openai_checkpoint(model_name=None, pytorch_dump_folder_path=None):
    """
    Copy/paste/tweak model's weights to our Jukebox structure.
    """
    # 遍历模型映射中的每个文件路径
    for file in MODEL_MAPPING[model_name]:
        # 如果文件不存在于指定路径中，则从URL下载文件并保存
        if not os.path.isfile(f"{pytorch_dump_folder_path}/{file.split('/')[-1]}"):
            r = requests.get(f"{PREFIX}{file}", allow_redirects=True)
            os.makedirs(f"{pytorch_dump_folder_path}/", exist_ok=True)
            open(f"{pytorch_dump_folder_path}/{file.split('/')[-1]}", "wb").write(r.content)

    # 根据模型名称加载预训练配置和模型
    model_to_convert = MODEL_MAPPING[model_name.split("/")[-1]]
    config = JukeboxConfig.from_pretrained(model_name)
    model = JukeboxModel(config)

    # 初始化一个空列表用于存储模型的权重字典
    weight_dict = []
    # 初始化一个空字典用于存储映射关系
    mapping = {}

    # 遍历要转换的每个模型字典名称
    for i, dict_name in enumerate(model_to_convert):
        # 从PyTorch模型文件中加载旧的字典
        old_dic = torch.load(f"{pytorch_dump_folder_path}/{dict_name.split('/')[-1]}")["model"]

        # 初始化一个空字典用于存储新的修复后的字典
        new_dic = {}
        # 遍历旧字典的每个键
        for k in old_dic.keys():
            # 根据键名的后缀进行不同的处理
            if k.endswith(".b"):
                new_dic[k.replace("b", "bias")] = old_dic[k]
            elif k.endswith(".w"):
                new_dic[k.replace("w", "weight")] = old_dic[k]
            elif "level_2" not in dict_name and "cond.model." in k:
                new_dic[k.replace(".blocks.", ".model.")] = old_dic[k]
            else:
                new_dic[k] = old_dic[k]

        # 根据特定前缀修复Jukebox模型的键名
        key_prefix = "vqvae" if i == 0 else f"priors.{3 - i}"
        new_dic = fix_jukebox_keys(new_dic, model.state_dict(), key_prefix, mapping)
        # 将修复后的字典添加到权重列表中
        weight_dict.append(new_dic)

    # 从权重列表中取出VQ-VAE部分的状态字典并加载到模型中
    vqvae_state_dict = weight_dict.pop(0)
    model.vqvae.load_state_dict(vqvae_state_dict)
    # 遍历权重列表中的每个元素，将其加载到模型的先验概率分布部分
    for i in range(len(weight_dict)):
        model.priors[i].load_state_dict(weight_dict[2 - i])

    # 确保指定路径存在，用于保存转换后的PyTorch模型
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 将映射数据保存为 JSON 文件到指定路径
    with open(f"{pytorch_dump_folder_path}/mapping.json", "w") as txtfile:
        json.dump(mapping, txtfile)
    
    # 打印模型保存信息，包括模型名称和保存路径
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    
    # 使用 PyTorch 模型对象的方法保存模型权重到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    
    # 返回保存的权重字典
    return weight_dict
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则进入条件判断块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="jukebox-5b-lyrics",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    # 添加必选参数：模型名称，设置默认值为"jukebox-5b-lyrics"，类型为字符串，用于指定要转换的模型名称

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="jukebox-5b-lyrics-converted",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加必选参数：PyTorch 模型输出文件夹路径，设置默认值为"jukebox-5b-lyrics-converted"，类型为字符串，用于指定输出转换后的模型的存储路径

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 变量中

    convert_openai_checkpoint(args.model_name, args.pytorch_dump_folder_path)
    # 调用函数 convert_openai_checkpoint，传入模型名称和输出文件夹路径作为参数，执行模型转换操作
```