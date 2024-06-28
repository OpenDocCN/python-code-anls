# `.\models\detr\convert_detr_to_pytorch.py`

```
# coding=utf-8
# 代码文件使用UTF-8编码

# Copyright 2023 The HuggingFace Inc. team.
# 版权声明：2023年由HuggingFace Inc.团队拥有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache许可证2.0版本授权使用本代码

# you may not use this file except in compliance with the License.
# 除非符合许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 除非适用法律要求或书面同意，否则按"原样"分发软件

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 无论明示或默示，不提供任何担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证以了解具体的权限和限制

"""Convert DETR checkpoints with native (Transformers) backbone."""
# 脚本用于将具有本地（Transformers）骨干结构的DETR检查点进行转换

import argparse
# 导入命令行参数解析库
import json
# 导入JSON处理库
from pathlib import Path
# 导入路径处理模块Path

import requests
# 导入请求库
import torch
# 导入PyTorch库
from huggingface_hub import hf_hub_download
# 从huggingface_hub库导入模型下载函数hf_hub_download
from PIL import Image
# 导入PIL图像处理库

from transformers import DetrConfig, DetrForObjectDetection, DetrForSegmentation, DetrImageProcessor, ResNetConfig
# 从transformers库中导入DETR模型相关组件和配置类
from transformers.utils import logging
# 从transformers的utils模块导入日志模块

logging.set_verbosity_info()
# 设置日志输出级别为info
logger = logging.get_logger(__name__)
# 获取logger对象


def get_detr_config(model_name):
    # 根据模型名称获取DETR配置

    # initialize config
    # 初始化配置对象
    if "resnet-50" in model_name:
        backbone_config = ResNetConfig.from_pretrained("microsoft/resnet-50")
    elif "resnet-101" in model_name:
        backbone_config = ResNetConfig.from_pretrained("microsoft/resnet-101")
    else:
        raise ValueError("Model name should include either resnet50 or resnet101")
    # 根据模型名称选择对应的ResNet配置，若模型名称不正确则抛出异常

    config = DetrConfig(use_timm_backbone=False, backbone_config=backbone_config)
    # 创建DETR配置对象，设置使用Timm背景为False，并传入选择的ResNet配置对象

    # set label attributes
    # 设置标签属性
    is_panoptic = "panoptic" in model_name
    # 检查模型名称中是否包含'panoptic'

    if is_panoptic:
        config.num_labels = 250
    else:
        config.num_labels = 91
        repo_id = "huggingface/label-files"
        filename = "coco-detection-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    # 根据模型类型设置标签数量和标签映射关系，如果是panoptic则设置标签数量为250，否则设置为91

    return config, is_panoptic
    # 返回配置对象和是否为panoptic类型的标志


def create_rename_keys(config):
    # 创建重命名键列表函数，用于将模型检查点中的键进行重命名

    # here we list all keys to be renamed (original name on the left, our name on the right)
    # 此处列出所有需要重命名的键（原始名称在左侧，我们的名称在右侧）
    rename_keys = []

    # stem
    # stem部分的重命名列表
    # fmt: off
    rename_keys.append(("backbone.0.body.conv1.weight", "backbone.conv_encoder.model.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.0.body.bn1.weight", "backbone.conv_encoder.model.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.0.body.bn1.bias", "backbone.conv_encoder.model.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.0.body.bn1.running_mean", "backbone.conv_encoder.model.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.0.body.bn1.running_var", "backbone.conv_encoder.model.embedder.embedder.normalization.running_var"))
    # stages
    # fmt: on
    # 格式控制注释开启，用于标记需要进行重命名的键和相应的新名称
    # 定义一个空列表，用于存储需要重命名的键值对
    rename_keys.extend(
        [
            # 将原始键名 "input_proj.weight" 重命名为 "input_projection.weight"
            ("input_proj.weight", "input_projection.weight"),
            # 将原始键名 "input_proj.bias" 重命名为 "input_projection.bias"
            ("input_proj.bias", "input_projection.bias"),
            # 将原始键名 "query_embed.weight" 重命名为 "query_position_embeddings.weight"
            ("query_embed.weight", "query_position_embeddings.weight"),
            # 将原始键名 "transformer.decoder.norm.weight" 重命名为 "decoder.layernorm.weight"
            ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
            # 将原始键名 "transformer.decoder.norm.bias" 重命名为 "decoder.layernorm.bias"
            ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
            # 将原始键名 "class_embed.weight" 重命名为 "class_labels_classifier.weight"
            ("class_embed.weight", "class_labels_classifier.weight"),
            # 将原始键名 "class_embed.bias" 重命名为 "class_labels_classifier.bias"
            ("class_embed.bias", "class_labels_classifier.bias"),
            # 将原始键名 "bbox_embed.layers.0.weight" 重命名为 "bbox_predictor.layers.0.weight"
            ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
            # 将原始键名 "bbox_embed.layers.0.bias" 重命名为 "bbox_predictor.layers.0.bias"
            ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
            # 将原始键名 "bbox_embed.layers.1.weight" 重命名为 "bbox_predictor.layers.1.weight"
            ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
            # 将原始键名 "bbox_embed.layers.1.bias" 重命名为 "bbox_predictor.layers.1.bias"
            ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),
            # 将原始键名 "bbox_embed.layers.2.weight" 重命名为 "bbox_predictor.layers.2.weight"
            ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),
            # 将原始键名 "bbox_embed.layers.2.bias" 重命名为 "bbox_predictor.layers.2.bias"
            ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),
        ]
    )

    # 返回存储重命名键值对的列表
    return rename_keys
# 重命名 state_dict 中的键：从 old 更改为 new，并返回 old 对应的值
def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val

# 从 state_dict 中读取查询、键和值的权重和偏置，并添加到 state_dict 中
def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "detr."

    # 遍历 transformer 编码器的6个层
    for i in range(6):
        # 读取输入投影层的权重和偏置（在 PyTorch 的 MultiHeadAttention 中，这是一个单独的矩阵加偏置）
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        
        # 将查询 q_proj 的权重和偏置添加到 state_dict 中
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        
        # 将键 k_proj 的权重和偏置添加到 state_dict 中
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        
        # 将值 v_proj 的权重和偏置添加到 state_dict 中
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    # 接下来处理 transformer 解码器（稍复杂，因为它还包括交叉注意力）
    # 循环遍历范围为0到5（共6次迭代）
    for i in range(6):
        # 读取self-attention层中输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        
        # 将权重的前256行作为query投影的权重
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        # 将偏置的前256个元素作为query投影的偏置
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        # 将权重的第256到511行作为key投影的权重
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        # 将偏置的第256到511个元素作为key投影的偏置
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        # 将权重的最后256行作为value投影的权重
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        # 将偏置的最后256个元素作为value投影的偏置
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        
        # 读取cross-attention层中输入投影层的权重和偏置
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        in_proj_bias_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias"
        )
        
        # 将权重的前256行作为cross-attention的query投影的权重
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        # 将偏置的前256个元素作为cross-attention的query投影的偏置
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        # 将权重的第256到511行作为cross-attention的key投影的权重
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        # 将偏置的第256到511个元素作为cross-attention的key投影的偏置
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        # 将权重的最后256行作为cross-attention的value投影的权重
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        # 将偏置的最后256个元素作为cross-attention的value投影的偏置
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
# 使用此函数准备一张可爱猫咪的图像，从指定 URL 下载图像并打开
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 requests 获取图像的原始流数据，并由 PIL 库打开为图像对象
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
# 将指定的 DETR 模型的检查点转换为我们的 DETR 结构
def convert_detr_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """

    # 加载默认配置
    config, is_panoptic = get_detr_config(model_name)

    # 从 torch hub 加载原始模型
    model_name_to_original_name = {
        "detr-resnet-50": "detr_resnet50",
        "detr-resnet-101": "detr_resnet101",
    }
    logger.info(f"Converting model {model_name}...")
    # 加载预训练模型并设置为评估模式
    detr = torch.hub.load("facebookresearch/detr", model_name_to_original_name[model_name], pretrained=True).eval()
    state_dict = detr.state_dict()

    # 重命名模型权重的键
    for src, dest in create_rename_keys(config):
        if is_panoptic:
            src = "detr." + src
        rename_key(state_dict, src, dest)

    # 处理查询、键和值矩阵需要特殊处理
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)

    # 对于基础模型键，需要在每个键前面添加特定前缀，因为头部模型使用不同的属性
    prefix = "detr.model." if is_panoptic else "model."
    for key in state_dict.copy().keys():
        if is_panoptic:
            # 对于全景分割模型，重新命名特定的键
            if (
                key.startswith("detr")
                and not key.startswith("class_labels_classifier")
                and not key.startswith("bbox_predictor")
            ):
                val = state_dict.pop(key)
                state_dict["detr.model" + key[4:]] = val
            elif "class_labels_classifier" in key or "bbox_predictor" in key:
                val = state_dict.pop(key)
                state_dict["detr." + key] = val
            elif key.startswith("bbox_attention") or key.startswith("mask_head"):
                continue
            else:
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        else:
            # 对于检测模型，重新命名特定的键
            if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
                val = state_dict.pop(key)
                state_dict[prefix + key] = val

    # 创建 HuggingFace 模型并加载状态字典
    model = DetrForSegmentation(config) if is_panoptic else DetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 验证转换结果在图像上的输出
    format = "coco_panoptic" if is_panoptic else "coco_detection"
    processor = DetrImageProcessor(format=format)

    encoding = processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    # 运行原始 DETR 模型和转换后的模型，并验证其输出的 logits 和预测框是否接近
    original_outputs = detr(pixel_values)
    outputs = model(pixel_values)

    assert torch.allclose(outputs.logits, original_outputs["pred_logits"], atol=1e-3)
    assert torch.allclose(outputs.pred_boxes, original_outputs["pred_boxes"], atol=1e-3)
    if is_panoptic:
        # 如果是全视角（panoptic）预测，则进行以下断言：
        # 检查模型输出的预测掩码是否与原始输出中的预测掩码在给定的绝对误差范围内相似
        assert torch.allclose(outputs.pred_masks, original_outputs["pred_masks"], atol=1e-4)
    # 打印提示信息，表示看起来一切正常
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        # 如果指定了 PyTorch 模型保存路径：
        # 记录日志，指示正在将模型和图像处理器保存到指定路径
        logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        # 确保保存路径存在，如果不存在则创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将图像处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # 如果需要将模型和图像处理器推送到 Hub 上：
        # 记录日志，指示正在上传 PyTorch 模型和图像处理器到 Hub
        logger.info("Uploading PyTorch model and image processor to the hub...")
        # 将模型推送到指定的 Hub 仓库（repository）
        model.push_to_hub(f"nielsr/{model_name}")
        # 将图像处理器推送到指定的 Hub 仓库（repository）
        processor.push_to_hub(f"nielsr/{model_name}")
# 如果当前脚本被直接运行而非被导入为模块，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数选项，用于指定要转换的DETR模型名称
    parser.add_argument(
        "--model_name",
        default="detr-resnet-50",
        type=str,
        choices=["detr-resnet-50", "detr-resnet-101"],
        help="Name of the DETR model you'd like to convert."
    )

    # 添加命令行参数选项，用于指定输出PyTorch模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        help="Path to the folder to output PyTorch model."
    )

    # 添加命令行参数选项，用于指定是否将模型推送到平台
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether to push the model to the hub or not."
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数convert_detr_checkpoint，传入解析后的命令行参数，执行模型转换操作
    convert_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```