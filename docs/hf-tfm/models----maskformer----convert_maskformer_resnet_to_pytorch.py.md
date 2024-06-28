# `.\models\maskformer\convert_maskformer_resnet_to_pytorch.py`

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
"""Convert MaskFormer checkpoints with ResNet backbone from the original repository. URL:
https://github.com/facebookresearch/MaskFormer"""


import argparse
import json
import pickle
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, ResNetConfig
from transformers.utils import logging


logging.set_verbosity_info()  # 设置日志输出级别为信息级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_maskformer_config(model_name: str):
    # 根据模型名称获取相应的 MaskFormer 配置
    if "resnet101c" in model_name:
        # TODO add support for ResNet-C backbone, which uses a "deeplab" stem
        raise NotImplementedError("To do")  # 抛出未实现的错误，提示需要添加对 ResNet-C 的支持
    elif "resnet101" in model_name:
        # 使用 Microsoft 的 ResNet-101 作为骨干网络配置
        backbone_config = ResNetConfig.from_pretrained(
            "microsoft/resnet-101", out_features=["stage1", "stage2", "stage3", "stage4"]
        )
    else:
        # 默认使用 Microsoft 的 ResNet-50 作为骨干网络配置
        backbone_config = ResNetConfig.from_pretrained(
            "microsoft/resnet-50", out_features=["stage1", "stage2", "stage3", "stage4"]
        )
    config = MaskFormerConfig(backbone_config=backbone_config)

    # 根据模型名称设置相应的标签数量和文件名
    repo_id = "huggingface/label-files"
    if "ade20k-full" in model_name:
        config.num_labels = 847
        filename = "maskformer-ade20k-full-id2label.json"
    elif "ade" in model_name:
        config.num_labels = 150
        filename = "ade20k-id2label.json"
    elif "coco-stuff" in model_name:
        config.num_labels = 171
        filename = "maskformer-coco-stuff-id2label.json"
    elif "coco" in model_name:
        # TODO
        config.num_labels = 133
        filename = "coco-panoptic-id2label.json"
    elif "cityscapes" in model_name:
        config.num_labels = 19
        filename = "cityscapes-id2label.json"
    elif "vistas" in model_name:
        config.num_labels = 65
        filename = "mapillary-vistas-id2label.json"

    # 从 HF Hub 下载指定的标签文件，并加载为 id 到 label 的映射字典
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}  # 将 id 转换为整数类型
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}  # 构建 label 到 id 的映射字典

    return config


def create_rename_keys(config):
    rename_keys = []
    # 添加重命名键，映射 backbone.stem.conv1.weight 到 model.pixel_level_module.encoder.embedder.embedder.convolution.weight
    rename_keys.append(("backbone.stem.conv1.weight", "model.pixel_level_module.encoder.embedder.embedder.convolution.weight"))
    # 添加新的键值对到 rename_keys 列表中，用于将模型中的特定参数路径重命名为新路径
    rename_keys.append(("backbone.stem.conv1.norm.weight", "model.pixel_level_module.encoder.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.stem.conv1.norm.bias", "model.pixel_level_module.encoder.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.stem.conv1.norm.running_mean", "model.pixel_level_module.encoder.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.stem.conv1.norm.running_var", "model.pixel_level_module.encoder.embedder.embedder.normalization.running_var"))

    # 在 fmt: on 之后的代码段，用于指示代码风格格式化工具保持打开状态

    # stages
    # FPN

    # 在 fmt: off 之后的代码段，用于指示代码风格格式化工具关闭格式化

    # 将 sem_seg_head.layer_4 的权重重命名为 model.pixel_level_module.decoder.fpn.stem.0 的权重
    rename_keys.append(("sem_seg_head.layer_4.weight", "model.pixel_level_module.decoder.fpn.stem.0.weight"))
    # 将 sem_seg_head.layer_4 的归一化权重重命名为 model.pixel_level_module.decoder.fpn.stem.1 的权重
    rename_keys.append(("sem_seg_head.layer_4.norm.weight", "model.pixel_level_module.decoder.fpn.stem.1.weight"))
    # 将 sem_seg_head.layer_4 的归一化偏置重命名为 model.pixel_level_module.decoder.fpn.stem.1 的偏置
    rename_keys.append(("sem_seg_head.layer_4.norm.bias", "model.pixel_level_module.decoder.fpn.stem.1.bias"))

    # 针对一系列逆序的源索引和目标索引，将 sem_seg_head.adapter_{source_index} 的权重和归一化参数
    # 重命名为 model.pixel_level_module.decoder.fpn.layers.{target_index} 下的对应投影层权重和偏置
    for source_index, target_index in zip(range(3, 0, -1), range(0, 3)):
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.0.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.bias"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.0.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.bias"))

    # 将 sem_seg_head.mask_features 的权重重命名为 model.pixel_level_module.decoder.mask_projection 的权重
    rename_keys.append(("sem_seg_head.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight"))
    # 将 sem_seg_head.mask_features 的偏置重命名为 model.pixel_level_module.decoder.mask_projection 的偏置
    rename_keys.append(("sem_seg_head.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias"))

    # 在 fmt: on 之后的代码段，用于指示代码风格格式化工具保持打开状态

    # Transformer decoder
    # fmt: off
    for idx in range(config.decoder_config.decoder_layers):
        # self-attention out projection
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.bias"))
        # cross-attention out projection
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.bias"))
        # MLP 1
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.weight", f"model.transformer_module.decoder.layers.{idx}.fc1.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.bias", f"model.transformer_module.decoder.layers.{idx}.fc1.bias"))
        # MLP 2
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.weight", f"model.transformer_module.decoder.layers.{idx}.fc2.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.bias", f"model.transformer_module.decoder.layers.{idx}.fc2.bias"))
        # layernorm 1 (self-attention layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.bias"))
        # layernorm 2 (cross-attention layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.bias"))
        # layernorm 3 (final layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.weight", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.bias", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.bias"))

    # Add renaming for the final layer norm weight
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.weight", "model.transformer_module.decoder.layernorm.weight"))
    # 将旧的键值对添加到重命名列表中，用新的键值对替换
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.bias", "model.transformer_module.decoder.layernorm.bias"))
    # fmt: on

    # 以下是将语句组织成一个块并且将其关闭

    # 网络，
# 从字典 dct 中弹出键 old 对应的值，并赋值给变量 val
def rename_key(dct, old, new):
    val = dct.pop(old)
    # 将键 new 添加到字典 dct，并将其值设为 val
    dct[new] = val


# 将每个编码器层的矩阵拆分为查询（queries）、键（keys）和值（values）
def read_in_decoder_q_k_v(state_dict, config):
    # fmt: off
    # 从配置中获取解码器隐藏层的大小
    hidden_size = config.decoder_config.hidden_size
    # 遍历所有解码器层
    for idx in range(config.decoder_config.decoder_layers):
        # 读取自注意力输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_bias")
        # 将查询（q_proj）、键（k_proj）和值（v_proj）添加到状态字典中
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
        
        # 读取交叉注意力输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_bias")
        # 将查询（q_proj）、键（k_proj）和值（v_proj）添加到状态字典中
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
    # fmt: on


# 我们将在一张可爱猫咪的图片上验证我们的结果
def prepare_img() -> torch.Tensor:
    # 定义一个 URL 变量，指向一个图像文件的地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 库发送 GET 请求，获取图像文件的内容流
    # 并使用 Image.open 方法打开流，返回一个图像对象
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回获取的图像对象
    return im
@torch.no_grad()
# 使用装饰器 torch.no_grad() 包装函数，确保在该函数内部的所有操作都不会进行梯度计算

def convert_maskformer_checkpoint(
    model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    Copy/paste/tweak model's weights to our MaskFormer structure.
    """
    # 根据模型名称获取对应的 MaskFormer 配置信息
    config = get_maskformer_config(model_name)

    # 从文件中加载原始的状态字典数据
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    state_dict = data["model"]

    # 根据预定义的映射关系重命名状态字典中的键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    
    # 读取 Decoder 部分的 q, k, v 参数信息并更新到状态字典中
    read_in_decoder_q_k_v(state_dict, config)

    # 将状态字典中的 numpy 数组转换为 torch 张量
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    # 加载 MaskFormer 模型，并设为评估模式
    model = MaskFormerForInstanceSegmentation(config)
    model.eval()

    # 加载状态字典到模型中
    model.load_state_dict(state_dict)

    # 验证模型预期输出
    image = prepare_img()
    if "vistas" in model_name:
        ignore_index = 65
    elif "cityscapes" in model_name:
        ignore_index = 65535
    else:
        ignore_index = 255
    reduce_labels = True if "ade" in model_name else False
    
    # 创建图像处理器实例，用于处理模型的输出
    image_processor = MaskFormerImageProcessor(ignore_index=ignore_index, reduce_labels=reduce_labels)

    # 准备输入数据
    inputs = image_processor(image, return_tensors="pt")

    # 调用模型进行推理
    outputs = model(**inputs)

    # 根据模型名称设置预期的 logits 值
    if model_name == "maskformer-resnet50-ade":
        expected_logits = torch.tensor(
            [[6.7710, -0.1452, -3.5687], [1.9165, -1.0010, -1.8614], [3.6209, -0.2950, -1.3813]]
        )
    elif model_name == "maskformer-resnet101-ade":
        expected_logits = torch.tensor(
            [[4.0381, -1.1483, -1.9688], [2.7083, -1.9147, -2.2555], [3.4367, -1.3711, -2.1609]]
        )
    elif model_name == "maskformer-resnet50-coco-stuff":
        expected_logits = torch.tensor(
            [[3.2309, -3.0481, -2.8695], [5.4986, -5.4242, -2.4211], [6.2100, -5.2279, -2.7786]]
        )
    elif model_name == "maskformer-resnet101-coco-stuff":
        expected_logits = torch.tensor(
            [[4.7188, -3.2585, -2.8857], [6.6871, -2.9181, -1.2487], [7.2449, -2.2764, -2.1874]]
        )
    elif model_name == "maskformer-resnet101-cityscapes":
        expected_logits = torch.tensor(
            [[-1.8861, -1.5465, 0.6749], [-2.3677, -1.6707, -0.0867], [-2.2314, -1.9530, -0.9132]]
        )
    elif model_name == "maskformer-resnet50-vistas":
        expected_logits = torch.tensor(
            [[-6.3917, -1.5216, -1.1392], [-5.5335, -4.5318, -1.8339], [-4.3576, -4.0301, 0.2162]]
        )
    elif model_name == "maskformer-resnet50-ade20k-full":
        expected_logits = torch.tensor(
            [[3.6146, -1.9367, -3.2534], [4.0099, 0.2027, -2.7576], [3.3913, -2.3644, -3.9519]]
        )
    elif model_name == "maskformer-resnet101-ade20k-full":
        expected_logits = torch.tensor(
            [[3.2211, -1.6550, -2.7605], [2.8559, -2.4512, -2.9574], [2.6331, -2.6775, -2.1844]]
        )
    # 断言：检查模型输出的前三个类别查询的对数概率是否与预期值在给定的误差范围内相等
    assert torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_logits, atol=1e-4)
    # 打印消息，表示检查通过
    print("Looks ok!")

    # 如果提供了 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        # 打印消息，指示正在保存模型和图像处理器到指定路径
        print(f"Saving model and image processor of {model_name} to {pytorch_dump_folder_path}")
        # 确保保存路径存在，如果不存在则创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将图像处理器保存到指定路径
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到模型中心（hub）
    if push_to_hub:
        # 打印消息，表示正在推送模型和图像处理器到中心（hub）
        print(f"Pushing model and image processor of {model_name} to the hub...")
        # 将模型推送到模型中心（hub）
        model.push_to_hub(f"facebook/{model_name}")
        # 将图像处理器推送到模型中心（hub）
        image_processor.push_to_hub(f"facebook/{model_name}")
if __name__ == "__main__":
    # 如果脚本作为主程序运行，则执行以下代码

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="maskformer-resnet50-ade",
        type=str,
        required=True,
        choices=[
            "maskformer-resnet50-ade",
            "maskformer-resnet101-ade",
            "maskformer-resnet50-coco-stuff",
            "maskformer-resnet101-coco-stuff",
            "maskformer-resnet101-cityscapes",
            "maskformer-resnet50-vistas",
            "maskformer-resnet50-ade20k-full",
            "maskformer-resnet101-ade20k-full",
        ],
        help=("Name of the MaskFormer model you'd like to convert",),
    )
    # 添加必需的参数：模型名称，指定默认值和可选的模型名称列表

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help=("Path to the original pickle file (.pkl) of the original checkpoint.",),
    )
    # 添加参数：原始检查点文件的路径，必须提供路径值

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加参数：输出 PyTorch 模型的目录路径，默认为 None

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加参数：是否将转换后的模型推送到 🤗 hub

    args = parser.parse_args()
    # 解析命令行参数并返回一个命名空间对象 args

    convert_maskformer_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
    # 调用函数 convert_maskformer_checkpoint，传递命令行参数中的模型名称、检查点路径、PyTorch 模型输出路径和推送标志作为参数
```