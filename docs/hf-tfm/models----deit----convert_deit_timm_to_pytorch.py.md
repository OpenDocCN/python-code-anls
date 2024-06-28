# `.\models\deit\convert_deit_timm_to_pytorch.py`

```
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
"""Convert DeiT distilled checkpoints from the timm library."""


import argparse  # 导入解析命令行参数的模块
import json  # 导入处理 JSON 格式数据的模块
from pathlib import Path  # 导入处理路径操作的模块

import requests  # 导入处理 HTTP 请求的模块
import timm  # 导入处理图像模型的模块
import torch  # 导入 PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载模型的函数
from PIL import Image  # 导入处理图像的模块

from transformers import DeiTConfig, DeiTForImageClassificationWithTeacher, DeiTImageProcessor  # 导入 DeiT 相关模块
from transformers.utils import logging  # 导入日志记录工具


logging.set_verbosity_info()  # 设置日志记录级别为 info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []  # 初始化存储重命名键的列表
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        # 遍历编码器层，处理输出投影、两个前馈神经网络和两个层归一化层
        rename_keys.append((f"blocks.{i}.norm1.weight", f"deit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"deit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"deit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"deit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"deit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"deit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"deit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"deit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"deit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"deit.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    # 投影层和位置嵌入的重命名处理
    rename_keys.extend(
        [
            ("cls_token", "deit.embeddings.cls_token"),
            ("dist_token", "deit.embeddings.distillation_token"),
            ("patch_embed.proj.weight", "deit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "deit.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "deit.embeddings.position_embeddings"),
        ]
    )
    if base_model:
        # 如果存在基础模型，进行以下操作
        # 将原始键名重命名为新的键名，适用于layernorm和pooler的情况
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
                ("pre_logits.fc.weight", "pooler.dense.weight"),
                ("pre_logits.fc.bias", "pooler.dense.bias"),
            ]
        )

        # 如果仅有基础模型，移除所有以"deit"开头的键名中的"deit"
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("deit") else pair for pair in rename_keys]
    else:
        # 如果没有基础模型，进行以下操作
        # 将原始键名重命名为新的键名，适用于layernorm和分类头的情况
        rename_keys.extend(
            [
                ("norm.weight", "deit.layernorm.weight"),
                ("norm.bias", "deit.layernorm.bias"),
                ("head.weight", "cls_classifier.weight"),
                ("head.bias", "cls_classifier.bias"),
                ("head_dist.weight", "distillation_classifier.weight"),
                ("head_dist.bias", "distillation_classifier.bias"),
            ]
        )

    # 返回处理后的键名列表
    return rename_keys
# 将每个编码器层的矩阵分割为查询（query）、键（key）和值（value）
def read_in_q_k_v(state_dict, config, base_model=False):
    # 遍历每个编码器层
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "deit."
        
        # 从 state_dict 中弹出输入投影层权重和偏置的参数
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        
        # 将查询（query）、键（key）和值（value）的权重添加到 state_dict 中
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 将字典中的某个键（old）重命名为新键（new）
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 在一张可爱猫咪的图片上验证我们的结果
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 HTTP 请求打开并获取图片
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 在没有梯度的情况下执行函数
@torch.no_grad()
def convert_deit_checkpoint(deit_name, pytorch_dump_folder_path):
    """
    将模型权重从其它结构复制、粘贴并调整到我们的 DeiT 结构中。
    """
    
    # 定义默认的 DeiT 配置
    config = DeiTConfig()
    # 所有 DeiT 模型都有微调的头部
    base_model = False
    # 数据集（在 ImageNet 2012 上微调）、补丁大小和图像大小
    config.num_labels = 1000
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    
    # 加载 ImageNet 类别到标签映射
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    
    # 解析 DeiT 模型名称中的补丁大小和图像大小
    config.patch_size = int(deit_name[-6:-4])
    config.image_size = int(deit_name[-3:])
    
    # 根据模型名称设置架构大小
    if deit_name[9:].startswith("tiny"):
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 3
    elif deit_name[9:].startswith("small"):
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    # 如果模型名称从第9个字符开始以"base"开头，不执行任何操作
    if deit_name[9:].startswith("base"):
        pass
    # 如果模型名称从第4个字符开始以"large"开头，设置一些配置参数
    elif deit_name[4:].startswith("large"):
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16

    # 使用timm库创建指定预训练模型的实例并设为评估模式
    timm_model = timm.create_model(deit_name, pretrained=True)
    timm_model.eval()

    # 获取timm模型的state_dict，准备对其进行重命名和修改
    state_dict = timm_model.state_dict()
    # 创建需要重命名的键对列表
    rename_keys = create_rename_keys(config, base_model)
    # 遍历重命名键对列表，对state_dict中的键进行重命名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 根据config和base_model读入query、key、value
    read_in_q_k_v(state_dict, config, base_model)

    # 使用HuggingFace库创建DeiT图像分类模型实例并设为评估模式
    model = DeiTForImageClassificationWithTeacher(config).eval()
    # 加载预训练模型的state_dict到当前模型实例
    model.load_state_dict(state_dict)

    # 使用DeiTImageProcessor处理图像，以准备输入模型
    size = int(
        (256 / 224) * config.image_size
    )  # 维持与224像素图像相同的比例，参考https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/datasets.py#L103
    # 创建DeiTImageProcessor实例，设置大小和裁剪大小
    image_processor = DeiTImageProcessor(size=size, crop_size=config.image_size)
    # 使用DeiTImageProcessor处理图像，返回PyTorch张量
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # 获取像素值张量
    pixel_values = encoding["pixel_values"]
    # 将输入像素值传入模型，获取模型的输出
    outputs = model(pixel_values)

    # 使用timm模型对像素值进行推断，获取其logits
    timm_logits = timm_model(pixel_values)
    # 断言timm模型输出的形状与HuggingFace模型输出的logits形状相同
    assert timm_logits.shape == outputs.logits.shape
    # 断言timm模型输出的logits与HuggingFace模型输出的logits在指定容差范围内近似相等
    assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)

    # 创建保存PyTorch模型和图像处理器的文件夹路径
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印提示信息，保存当前模型到指定路径
    print(f"Saving model {deit_name} to {pytorch_dump_folder_path}")
    # 将当前模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印提示信息，保存图像处理器到指定路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本被直接执行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--deit_name",
        default="vit_deit_base_distilled_patch16_224",
        type=str,
        help="Name of the DeiT timm model you'd like to convert.",
    )
    # 添加一个必需的命令行参数 --deit_name，指定默认值和帮助信息

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加另一个命令行参数 --pytorch_dump_folder_path，指定默认值和帮助信息

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 args 对象中

    convert_deit_checkpoint(args.deit_name, args.pytorch_dump_folder_path)
    # 调用 convert_deit_checkpoint 函数，传入解析后的参数 args.deit_name 和 args.pytorch_dump_folder_path
```