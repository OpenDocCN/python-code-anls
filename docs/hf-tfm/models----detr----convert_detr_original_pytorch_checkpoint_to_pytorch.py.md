# `.\models\detr\convert_detr_original_pytorch_checkpoint_to_pytorch.py`

```
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert DETR checkpoints with timm backbone."""


import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 数据的模块
from collections import OrderedDict  # 导入有序字典模块
from pathlib import Path  # 导入处理文件路径的模块

import requests  # 导入发送 HTTP 请求的模块
import torch  # 导入 PyTorch 模块
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载模型的方法
from PIL import Image  # 导入处理图像的模块

from transformers import DetrConfig, DetrForObjectDetection, DetrForSegmentation, DetrImageProcessor  # 导入 DETR 模型相关类
from transformers.utils import logging  # 导入日志记录模块


logging.set_verbosity_info()  # 设置日志输出为 info 级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# here we list all keys to be renamed (original name on the left, our name on the right)
rename_keys = []  # 创建空列表，用于存储需要重命名的键值对
for i in range(6):
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.weight", f"encoder.layers.{i}.self_attn.out_proj.weight")
    )  # 添加需要重命名的权重参数路径对，编码器层的注意力输出投影权重
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.bias", f"encoder.layers.{i}.self_attn.out_proj.bias")
    )  # 添加需要重命名的偏置参数路径对，编码器层的注意力输出投影偏置
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"encoder.layers.{i}.fc1.weight"))  # 编码器层的第一个全连接层权重
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"encoder.layers.{i}.fc1.bias"))  # 编码器层的第一个全连接层偏置
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"encoder.layers.{i}.fc2.weight"))  # 编码器层的第二个全连接层权重
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"encoder.layers.{i}.fc2.bias"))  # 编码器层的第二个全连接层偏置
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm1.weight", f"encoder.layers.{i}.self_attn_layer_norm.weight")
    )  # 编码器层的第一个 Layernorm 权重
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm1.bias", f"encoder.layers.{i}.self_attn_layer_norm.bias")
    )  # 编码器层的第一个 Layernorm 偏置
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm2.weight", f"encoder.layers.{i}.final_layer_norm.weight")
    )  # 编码器层的第二个 Layernorm 权重
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm2.bias", f"encoder.layers.{i}.final_layer_norm.bias")
    )  # 编码器层的第二个 Layernorm 偏置
    # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight")
    )  # 解码器层的注意力输出投影权重
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
    )  # 解码器层的注意力输出投影偏置
    # 将下列元组添加到 rename_keys 列表中，重命名 Transformer 模型中特定层的权重和偏置参数到 Decoder 模型对应层
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.multihead_attn.out_proj.weight",  # Transformer 解码器的多头注意力层输出投影权重
            f"decoder.layers.{i}.encoder_attn.out_proj.weight",  # 对应的解码器层的编码器注意力层输出投影权重
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.multihead_attn.out_proj.bias",  # Transformer 解码器的多头注意力层输出投影偏置
            f"decoder.layers.{i}.encoder_attn.out_proj.bias",  # 对应的解码器层的编码器注意力层输出投影偏置
        )
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.linear1.weight", f"decoder.layers.{i}.fc1.weight")  # Transformer 解码器的第一个线性层权重 -> 解码器层的第一个全连接层权重
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.linear1.bias", f"decoder.layers.{i}.fc1.bias")  # Transformer 解码器的第一个线性层偏置 -> 解码器层的第一个全连接层偏置
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.linear2.weight", f"decoder.layers.{i}.fc2.weight")  # Transformer 解码器的第二个线性层权重 -> 解码器层的第二个全连接层权重
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.linear2.bias", f"decoder.layers.{i}.fc2.bias")  # Transformer 解码器的第二个线性层偏置 -> 解码器层的第二个全连接层偏置
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight")  # Transformer 解码器的第一个归一化层权重 -> 解码器层的自注意力层归一化权重
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias")  # Transformer 解码器的第一个归一化层偏置 -> 解码器层的自注意力层归一化偏置
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")  # Transformer 解码器的第二个归一化层权重 -> 解码器层的编码器注意力层归一化权重
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")  # Transformer 解码器的第二个归一化层偏置 -> 解码器层的编码器注意力层归一化偏置
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight")  # Transformer 解码器的第三个归一化层权重 -> 解码器层的最终归一化层权重
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias")  # Transformer 解码器的第三个归一化层偏置 -> 解码器层的最终归一化层偏置
    )
# 将键名转换列表扩展到state_dict中，用于重命名模型参数
rename_keys.extend(
    [
        ("input_proj.weight", "input_projection.weight"),
        ("input_proj.bias", "input_projection.bias"),
        ("query_embed.weight", "query_position_embeddings.weight"),
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
        ("class_embed.weight", "class_labels_classifier.weight"),
        ("class_embed.bias", "class_labels_classifier.bias"),
        ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
        ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
        ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
        ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),
        ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),
        ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),
    ]
)

# 函数：根据给定的旧键名和新键名重命名state_dict中的键
def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val

# 函数：重命名state_dict中backbone的键名，将"backbone.0.body"替换为"backbone.conv_encoder.model"
def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# 函数：从state_dict中读取query、key、value的权重和偏置，并重新组织存储结构
def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "detr."

    # 首先处理transformer encoder部分
    for i in range(6):
        # 读取self attention的输入投影层权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        
        # 将权重和偏置按顺序分配到query、key、value的投影层权重和偏置中
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    # 处理transformer decoder部分，包含cross-attention，稍显复杂，需要在后续代码中继续处理
    for i in range(6):
        # 从状态字典中弹出自注意力机制的输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        
        # 将查询、键和值（按顺序）添加到状态字典中
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        
        # 从状态字典中弹出交叉注意力机制的输入投影层的权重和偏置
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        
        # 将查询、键和值（按顺序）添加到状态字典中，用于交叉注意力机制
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
# We will verify our results on an image of cute cats
def prepare_img():
    # 定义图像的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 库发送 GET 请求获取图像的原始数据流，并由 PIL 库打开为图像对象
    im = Image.open(requests.get(url, stream=True).raw)

    # 返回打开的图像对象
    return im


@torch.no_grad()
def convert_detr_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """

    # 加载默认的 DETR 配置
    config = DetrConfig()

    # 根据模型名称设置配置的背骨骨干和扩张属性
    if "resnet101" in model_name:
        config.backbone = "resnet101"
    if "dc5" in model_name:
        config.dilation = True

    # 检查模型是否为全景模型，根据需要设置标签数目
    is_panoptic = "panoptic" in model_name
    if is_panoptic:
        config.num_labels = 250
    else:
        config.num_labels = 91
        # 加载 COCO 数据集的标签文件，将其转换为 id 到标签名的映射字典
        repo_id = "huggingface/label-files"
        filename = "coco-detection-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # 根据格式加载相应的图像处理器
    format = "coco_panoptic" if is_panoptic else "coco_detection"
    image_processor = DetrImageProcessor(format=format)

    # 准备图像，调用 prepare_img 函数获取图像对象，并编码为 PyTorch 张量
    img = prepare_img()
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    # 记录信息：正在转换模型的名称
    logger.info(f"Converting model {model_name}...")

    # 从 Torch Hub 加载原始的 DETR 模型
    detr = torch.hub.load("facebookresearch/detr", model_name, pretrained=True).eval()
    state_dict = detr.state_dict()

    # 重命名模型权重中的键名
    for src, dest in rename_keys:
        if is_panoptic:
            src = "detr." + src
        rename_key(state_dict, src, dest)

    # 重命名骨干网络的键名
    state_dict = rename_backbone_keys(state_dict)

    # 处理查询、键和值矩阵的特殊处理
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)

    # 重要提示：需要为每个基础模型的键名添加前缀，因为头部模型使用不同的属性来表示它们
    prefix = "detr.model." if is_panoptic else "model."
    # 遍历状态字典的键的副本
    for key in state_dict.copy().keys():
        # 如果是全景视觉任务
        if is_panoptic:
            # 检查是否以"detr"开头且不以"class_labels_classifier"或"bbox_predictor"开头的键
            if (
                key.startswith("detr")
                and not key.startswith("class_labels_classifier")
                and not key.startswith("bbox_predictor")
            ):
                # 弹出该键对应的值，并将其以"detr.model" + key[4:]的格式重新命名并存入字典
                val = state_dict.pop(key)
                state_dict["detr.model" + key[4:]] = val
            # 如果键包含"class_labels_classifier"或"bbox_predictor"
            elif "class_labels_classifier" in key or "bbox_predictor" in key:
                # 弹出该键对应的值，并将其以"detr." + key的格式重新命名并存入字典
                val = state_dict.pop(key)
                state_dict["detr." + key] = val
            # 如果键以"bbox_attention"或"mask_head"开头，则跳过处理
            elif key.startswith("bbox_attention") or key.startswith("mask_head"):
                continue
            # 否则，弹出该键对应的值，并以prefix + key的格式重新命名并存入字典
            else:
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        # 如果不是全景视觉任务
        else:
            # 如果不以"class_labels_classifier"或"bbox_predictor"开头的键
            if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
                # 弹出该键对应的值，并以prefix + key的格式重新命名并存入字典
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
    
    # 最终，根据is_panoptic创建HuggingFace模型并加载状态字典
    model = DetrForSegmentation(config) if is_panoptic else DetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 验证转换的正确性
    original_outputs = detr(pixel_values)
    outputs = model(pixel_values)
    # 断言输出的logits和原始输出的"pred_logits"在指定的误差范围内相似
    assert torch.allclose(outputs.logits, original_outputs["pred_logits"], atol=1e-4)
    # 断言输出的预测框和原始输出的"pred_boxes"在指定的误差范围内相似
    assert torch.allclose(outputs.pred_boxes, original_outputs["pred_boxes"], atol=1e-4)
    # 如果是全景视觉任务，断言输出的预测掩码和原始输出的"pred_masks"在指定的误差范围内相似
    if is_panoptic:
        assert torch.allclose(outputs.pred_masks, original_outputs["pred_masks"], atol=1e-4)
    
    # 保存模型和图像处理器
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    # 确保路径存在，如果不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果脚本直接运行（而非被导入其他模块），则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加名为 "--model_name" 的命令行参数，用于指定要转换的 DETR 模型的名称，默认为 "detr_resnet50"
    parser.add_argument(
        "--model_name", default="detr_resnet50", type=str, help="Name of the DETR model you'd like to convert."
    )

    # 添加名为 "--pytorch_dump_folder_path" 的命令行参数，用于指定输出 PyTorch 模型的文件夹路径，默认为 None
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_detr_checkpoint，传入解析得到的模型名称和输出文件夹路径作为参数
    convert_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path)
```