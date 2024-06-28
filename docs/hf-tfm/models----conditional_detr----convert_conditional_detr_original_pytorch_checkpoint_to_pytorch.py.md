# `.\models\conditional_detr\convert_conditional_detr_original_pytorch_checkpoint_to_pytorch.py`

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
"""Convert Conditional DETR checkpoints."""


import argparse
import json
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrForObjectDetection,
    ConditionalDetrForSegmentation,
    ConditionalDetrImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()  # 设置日志输出级别为INFO
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

# here we list all keys to be renamed (original name on the left, our name on the right)
rename_keys = []
for i in range(6):
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.weight", f"encoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.bias", f"encoder.layers.{i}.self_attn.out_proj.bias")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"encoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"encoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"encoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"encoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm1.weight", f"encoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"encoder.layers.{i}.self_attn_layer_norm.bias"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"encoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"encoder.layers.{i}.final_layer_norm.bias"))
    # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
    )
    # 将下面的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.weight",
            f"decoder.layers.{i}.encoder_attn.out_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.bias",
            f"decoder.layers.{i}.encoder_attn.out_proj.bias",
        )
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"decoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"decoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"decoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"decoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))

    # 为条件式 DETR 的解码器中的自注意力和交叉注意力添加投影权重
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qcontent_proj.weight", f"decoder.layers.{i}.sa_qcontent_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kcontent_proj.weight", f"decoder.layers.{i}.sa_kcontent_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qpos_proj.weight", f"decoder.layers.{i}.sa_qpos_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kpos_proj.weight", f"decoder.layers.{i}.sa_kpos_proj.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_v_proj.weight", f"decoder.layers.{i}.sa_v_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qcontent_proj.weight", f"decoder.layers.{i}.ca_qcontent_proj.weight")
    )
    # 以下键注释被注释掉，因为它们在代码中已经被注释掉
    # rename_keys.append((f"transformer.decoder.layers.{i}.ca_qpos_proj.weight", f"decoder.layers.{i}.ca_qpos_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kcontent_proj.weight", f"decoder.layers.{i}.ca_kcontent_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kpos_proj.weight", f"decoder.layers.{i}.ca_kpos_proj.weight")
    )
    # 将以下键值对添加到 rename_keys 列表中，用于重命名模型参数的路径
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_v_proj.weight", f"decoder.layers.{i}.ca_v_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.weight", f"decoder.layers.{i}.ca_qpos_sine_proj.weight")
    )

    # 添加以下键值对到 rename_keys 列表，用于重命名模型参数的路径
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qcontent_proj.bias", f"decoder.layers.{i}.sa_qcontent_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kcontent_proj.bias", f"decoder.layers.{i}.sa_kcontent_proj.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_qpos_proj.bias", f"decoder.layers.{i}.sa_qpos_proj.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_kpos_proj.bias", f"decoder.layers.{i}.sa_kpos_proj.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_v_proj.bias", f"decoder.layers.{i}.sa_v_proj.bias"))

    # 添加以下键值对到 rename_keys 列表，用于重命名模型参数的路径
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qcontent_proj.bias", f"decoder.layers.{i}.ca_qcontent_proj.bias")
    )
    # 注释以下行代码被注释掉，不会添加到 rename_keys 列表中
    # rename_keys.append((f"transformer.decoder.layers.{i}.ca_qpos_proj.bias", f"decoder.layers.{i}.ca_qpos_proj.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kcontent_proj.bias", f"decoder.layers.{i}.ca_kcontent_proj.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_kpos_proj.bias", f"decoder.layers.{i}.ca_kpos_proj.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_v_proj.bias", f"decoder.layers.{i}.ca_v_proj.bias"))

    # 添加以下键值对到 rename_keys 列表，用于重命名模型参数的路径
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.bias", f"decoder.layers.{i}.ca_qpos_sine_proj.bias")
    )
# 定义需要重命名的键值对列表，用于转换模型参数命名空间
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
        ("transformer.decoder.ref_point_head.layers.0.weight", "decoder.ref_point_head.layers.0.weight"),
        ("transformer.decoder.ref_point_head.layers.0.bias", "decoder.ref_point_head.layers.0.bias"),
        ("transformer.decoder.ref_point_head.layers.1.weight", "decoder.ref_point_head.layers.1.weight"),
        ("transformer.decoder.ref_point_head.layers.1.bias", "decoder.ref_point_head.layers.1.bias"),
        ("transformer.decoder.query_scale.layers.0.weight", "decoder.query_scale.layers.0.weight"),
        ("transformer.decoder.query_scale.layers.0.bias", "decoder.query_scale.layers.0.bias"),
        ("transformer.decoder.query_scale.layers.1.weight", "decoder.query_scale.layers.1.weight"),
        ("transformer.decoder.query_scale.layers.1.bias", "decoder.query_scale.layers.1.bias"),
        ("transformer.decoder.layers.0.ca_qpos_proj.weight", "decoder.layers.0.ca_qpos_proj.weight"),
        ("transformer.decoder.layers.0.ca_qpos_proj.bias", "decoder.layers.0.ca_qpos_proj.bias"),
    ]
)


def rename_key(state_dict, old, new):
    # 从状态字典中弹出旧的键值对，然后插入新的键值对
    val = state_dict.pop(old)
    state_dict[new] = val


def rename_backbone_keys(state_dict):
    # 创建新的有序字典，用于存储重命名后的状态字典
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "backbone.0.body" in key:
            # 替换特定的键名以适应新的模型结构
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def read_in_q_k_v(state_dict, is_panoptic=False):
    # 根据条件是否为全景检测，选择不同的模型前缀
    prefix = ""
    if is_panoptic:
        prefix = "conditional_detr."

    # 第一步：Transformer 编码器
    # 循环处理6次，分别处理每个层的自注意力机制参数
    for i in range(6):
        # 从状态字典中弹出输入投影层权重和偏置（在PyTorch的MultiHeadAttention中，这是单个矩阵加偏置）
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        
        # 将权重切片并添加到状态字典中作为查询、键和值的投影权重
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
# 我们将在一张可爱猫咪的图片上验证我们的结果
def prepare_img():
    # 图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 HTTP 请求获取图片的原始数据流，并打开为图像对象
    im = Image.open(requests.get(url, stream=True).raw)

    # 返回处理后的图像对象
    return im


@torch.no_grad()
def convert_conditional_detr_checkpoint(model_name, pytorch_dump_folder_path):
    """
    将模型权重复制/粘贴/调整到我们的 CONDITIONAL_DETR 结构中。
    """

    # 加载默认配置
    config = ConditionalDetrConfig()
    # 设置骨干网络和膨胀属性
    if "resnet101" in model_name:
        config.backbone = "resnet101"
    if "dc5" in model_name:
        config.dilation = True
    is_panoptic = "panoptic" in model_name
    if is_panoptic:
        # 如果是全景分割模型，则设置类别数为 250
        config.num_labels = 250
    else:
        # 如果是检测模型，则设置类别数为 91
        config.num_labels = 91
        # 下载 COCO 检测任务标签映射文件并加载
        repo_id = "huggingface/label-files"
        filename = "coco-detection-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # 加载图像处理器
    format = "coco_panoptic" if is_panoptic else "coco_detection"
    image_processor = ConditionalDetrImageProcessor(format=format)

    # 准备图像
    img = prepare_img()
    # 使用图像处理器对图像进行编码，返回 PyTorch 张量
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    # 记录日志，显示正在转换的模型名称
    logger.info(f"Converting model {model_name}...")

    # 从 torch hub 加载原始模型
    conditional_detr = torch.hub.load("DeppMeng/ConditionalDETR", model_name, pretrained=True).eval()
    state_dict = conditional_detr.state_dict()
    # 重命名键名
    for src, dest in rename_keys:
        if is_panoptic:
            src = "conditional_detr." + src
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    # 针对查询、键和值矩阵进行特殊处理
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    # 重要提示：对于基础模型的每个键名，我们需要添加前缀，因为头模型使用不同的属性
    prefix = "conditional_detr.model." if is_panoptic else "model."
    # 遍历状态字典的复制版本中的所有键
    for key in state_dict.copy().keys():
        # 如果是全景视觉任务
        if is_panoptic:
            # 如果键以"conditional_detr"开头，并且不以"class_labels_classifier"或"bbox_predictor"开头
            if (
                key.startswith("conditional_detr")
                and not key.startswith("class_labels_classifier")
                and not key.startswith("bbox_predictor")
            ):
                # 弹出该键对应的值，并将其添加到新键中，加上前缀"conditional_detr.model"
                val = state_dict.pop(key)
                state_dict["conditional_detr.model" + key[4:]] = val
            # 如果键包含"class_labels_classifier"或"bbox_predictor"
            elif "class_labels_classifier" in key or "bbox_predictor" in key:
                # 弹出该键对应的值，并将其添加到新键中，加上前缀"conditional_detr."
                val = state_dict.pop(key)
                state_dict["conditional_detr." + key] = val
            # 如果键以"bbox_attention"或"mask_head"开头，则跳过此次循环
            elif key.startswith("bbox_attention") or key.startswith("mask_head"):
                continue
            # 否则，弹出该键对应的值，并将其添加到新键中，加上指定前缀
            else:
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        # 如果不是全景视觉任务
        else:
            # 如果键既不以"class_labels_classifier"开头也不以"bbox_predictor"开头
            if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
                # 弹出该键对应的值，并将其添加到新键中，加上指定前缀
                val = state_dict.pop(key)
                state_dict[prefix + key] = val

    # 最后，根据是否为全景视觉任务创建 HuggingFace 模型并加载状态字典
    model = ConditionalDetrForSegmentation(config) if is_panoptic else ConditionalDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 将模型推送到 Hub 上的指定仓库
    model.push_to_hub(repo_id=model_name, organization="DepuMeng", commit_message="Add model")
    
    # 验证转换的正确性
    original_outputs = conditional_detr(pixel_values)
    outputs = model(pixel_values)
    assert torch.allclose(outputs.logits, original_outputs["pred_logits"], atol=1e-4)
    assert torch.allclose(outputs.pred_boxes, original_outputs["pred_boxes"], atol=1e-4)
    # 如果是全景视觉任务，还需验证预测的掩膜
    if is_panoptic:
        assert torch.allclose(outputs.pred_masks, original_outputs["pred_masks"], atol=1e-4)
    
    # 保存模型和图像处理器
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本作为主程序运行（而非被导入到其他脚本中执行），则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 向参数解析器中添加一个参数，用于指定模型名称，默认为"conditional_detr_resnet50"
    parser.add_argument(
        "--model_name",
        default="conditional_detr_resnet50",
        type=str,
        help="Name of the CONDITIONAL_DETR model you'd like to convert.",
    )

    # 向参数解析器中添加一个参数，用于指定输出 PyTorch 模型的文件夹路径，默认为 None
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        help="Path to the folder to output PyTorch model."
    )

    # 解析命令行参数并将其存储在 args 对象中
    args = parser.parse_args()

    # 调用函数 convert_conditional_detr_checkpoint，将解析得到的模型名称和输出路径作为参数传递
    convert_conditional_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path)
```