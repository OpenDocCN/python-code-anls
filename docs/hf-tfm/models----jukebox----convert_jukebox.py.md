# `.\models\jukebox\convert_jukebox.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""转换 Jukebox 检查点"""

# 导入所需的库
import argparse
import json
import os
from pathlib import Path

import requests
import torch

from transformers import JukeboxConfig, JukeboxModel
from transformers.utils import logging

# 设置日志级别为信息
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# Jukebox 模型下载链接前缀
PREFIX = "https://openaipublic.azureedge.net/jukebox/models/"
# Jukebox 模型映射
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

# 定义替换键的函数
def replace_key(key):
    if key.endswith(".model.1.bias") and len(key.split(".")) > 10:
        key = key.replace(".model.1.bias", ".conv1d_1.bias")
    elif key.endswith(".model.1.weight") and len(key.split(".")) > 10:
        key = key.replace(".model.1.weight", ".conv1d_1.weight")
    elif key.endswith(".model.3.bias") and len(key.split(".")) > 10:
        key = key.replace(".model.3.bias", ".conv1d_2.bias")
    elif key.endswith(".model.3.weight") and len(key.split(".")) > 10:
        key = key.replace(".model.3.weight", ".conv1d_2.weight")

    if "conditioner_blocks.0." in key:
        key = key.replace("conditioner_blocks.0", "conditioner_blocks")

    if "prime_prior" in key:
        key = key.replace("prime_prior", "encoder")

    if ".emb." in key and "total" not in key and "absolute" not in key and "relative" not in key:
        key = key.replace(".emb.", ".")

    if key.endswith("k"):  # 将 vqvae.X.k 替换为 vqvae.X.codebook
        return key.replace(".k", ".codebook")
    if "y_emb." in key:
        return key.replace("y_emb.", "metadata_embedding.")

    if "x_emb.emb." in key:
        key = key.replace("0.x_emb.emb", "embed_tokens")

    if "prime_state_ln" in key:
        return key.replace("prime_state_ln", "encoder.final_layer_norm")
    if ".ln" in key:
        return key.replace(".ln", ".layer_norm")
    if "_ln" in key:
        return key.replace("_ln", "_layer_norm")

    if "prime_state_proj" in key:
        return key.replace("prime_state_proj", "encoder.proj_in")
    if "prime_x_out" in key:
        return key.replace("prime_x_out", "encoder.lm_head")
    # 如果字符串中包含"prior.x_out"，则将"x_out"替换为"fc_proj_out"并返回
    if "prior.x_out" in key:
        return key.replace("x_out", "fc_proj_out")
    
    # 如果字符串中包含"x_emb"，则将"x_emb"替换为"embed_tokens"并返回
    if "x_emb" in key:
        return key.replace("x_emb", "embed_tokens")

    # 如果以上条件都不满足，则直接返回原始字符串
    return key
def fix_jukebox_keys(state_dict, model_state_dict, key_prefix, mapping):
    # 初始化一个空字典用于存储修正后的键值对
    new_dict = {}
    # 导入正则表达式模块
    import re

    # 定义正则表达式模式，用于匹配不同类型的键名
    re_encoder_block_conv_in = re.compile(r"encoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).(bias|weight)")
    re_encoder_block_resnet = re.compile(
        r"encoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).model.(\d*).model.(\d*).(bias|weight)"
    )
    re_encoder_block_proj_out = re.compile(r"encoders.(\d*).level_blocks.(\d*).model.(\d*).(bias|weight)")

    re_decoder_block_conv_out = re.compile(r"decoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).(bias|weight)")
    re_decoder_block_resnet = re.compile(
        r"decoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).model.(\d*).model.(\d*).(bias|weight)"
    )
    re_decoder_block_proj_in = re.compile(r"decoders.(\d*).level_blocks.(\d*).model.(\d*).(bias|weight)")

    re_prior_cond_conv_out = re.compile(r"conditioner_blocks.(\d*).cond.model.(\d*).(\d).(bias|weight)")
    re_prior_cond_resnet = re.compile(
        r"conditioner_blocks.(\d*).cond.model.(\d*).(\d).model.(\d*).model.(\d*).(bias|weight)"
    )
    re_prior_cond_proj_in = re.compile(r"conditioner_blocks.(\d*).cond.model.(\d*).(bias|weight)")

    # 返回修正后的字典
    return new_dict


@torch.no_grad()
def convert_openai_checkpoint(model_name=None, pytorch_dump_folder_path=None):
    """
    Copy/paste/tweak model's weights to our Jukebox structure.
    """
    # 遍历模型映射列表中的文件，下载并保存到指定路径
    for file in MODEL_MAPPING[model_name]:
        if not os.path.isfile(f"{pytorch_dump_folder_path}/{file.split('/')[-1]}"):
            r = requests.get(f"{PREFIX}{file}", allow_redirects=True)
            os.makedirs(f"{pytorch_dump_folder_path}/", exist_ok=True)
            open(f"{pytorch_dump_folder_path}/{file.split('/')[-1]}", "wb").write(r.content)

    # 获取要转换的模型名称
    model_to_convert = MODEL_MAPPING[model_name.split("/")[-1]]

    # 从预训练模型名称创建配置对象和模型对象
    config = JukeboxConfig.from_pretrained(model_name)
    model = JukeboxModel(config)

    # 初始化权重字典和映射字典
    weight_dict = []
    mapping = {}
    # 遍历要转换的模型列表
    for i, dict_name in enumerate(model_to_convert):
        # 加载旧模型的权重字典
        old_dic = torch.load(f"{pytorch_dump_folder_path}/{dict_name.split('/')[-1]}")["model"]

        # 初始化一个新的字典用于存储修正后的键值��
        new_dic = {}
        # 根据键名后缀进行修正
        for k in old_dic.keys():
            if k.endswith(".b"):
                new_dic[k.replace("b", "bias")] = old_dic[k]
            elif k.endswith(".w"):
                new_dic[k.replace("w", "weight")] = old_dic[k]
            elif "level_2" not in dict_name and "cond.model." in k:
                new_dic[k.replace(".blocks.", ".model.")] = old_dic[k]
            else:
                new_dic[k] = old_dic[k]

        # 根据前缀和映射修正新字典的键名
        key_prefix = "vqvae" if i == 0 else f"priors.{3 - i}"
        new_dic = fix_jukebox_keys(new_dic, model.state_dict(), key_prefix, mapping)
        weight_dict.append(new_dic)

    # 弹出 VQ-VAE 模型的权重字典并加载到模型中
    vqvae_state_dict = weight_dict.pop(0)
    model.vqvae.load_state_dict(vqvae_state_dict)
    # 加载其他模型的权重字典到对应的模型中
    for i in range(len(weight_dict)):
        model.priors[i].load_state_dict(weight_dict[2 - i])

    # 创建指定路径的文件夹
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打开一个文件，将字典 mapping 写入到文件中，以 JSON 格式保存
    with open(f"{pytorch_dump_folder_path}/mapping.json", "w") as txtfile:
        json.dump(mapping, txtfile)

    # 打印保存模型的信息，包括模型名称和保存路径
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 返回权重字典
    return weight_dict
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="jukebox-5b-lyrics",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="jukebox-5b-lyrics-converted",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将OpenAI的检查点转换为PyTorch模型
    convert_openai_checkpoint(args.model_name, args.pytorch_dump_folder_path)
```