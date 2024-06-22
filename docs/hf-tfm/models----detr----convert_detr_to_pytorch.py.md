# `.\models\detr\convert_detr_to_pytorch.py`

```py
# coding=utf-8
# 设置文件编码格式为utf-8

# 版权声明
# Copyright 2023 The HuggingFace Inc. team.
#
# 根据 Apache 许可证 2.0 版本获得许可；
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"实际情况"分发软件
# 没有任何形式的担保或条件，无论是明示的还是默示的
# 有关特定语言授权权限和限制的详细信息，请参阅许可证
"""Convert DETR checkpoints with native (Transformers) backbone."""


import argparse  # 导入参数解析模块
import json  # 导入json模块
from pathlib import Path # 从pathlib中导入Path类

import requests  # 导入requests模块
import torch  # 导入torch模块
from huggingface_hub import hf_hub_download  # 从huggingface_hub模块导入hf_hub_download函数
from PIL import Image  # 从PIL模块导入Image类

from transformers import DetrConfig, DetrForObjectDetection, DetrForSegmentation, DetrImageProcessor, ResNetConfig  # 从transformers模块导入DetrConfig、DetrForObjectDetection、DetrForSegmentation、DetrImageProcessor、ResNetConfig类
from transformers.utils import logging  # 从transformers.utils模块导入logging模块


logging.set_verbosity_info()  # 将日志级别设置为info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_detr_config(model_name):
    # 获取DETR模型的配置
    # 初始化配置
    if "resnet-50" in model_name:
        backbone_config = ResNetConfig.from_pretrained("microsoft/resnet-50")  # 从预训练的resnet-50模型获取配置
    elif "resnet-101" in model_name:
        backbone_config = ResNetConfig.from_pretrained("microsoft/resnet-101")  # 从预训练的resnet-101模型获取配置
    else:
        raise ValueError("Model name should include either resnet50 or resnet101")  # 抛出数值错误，模型名称应包括resnet50或resnet101

    config = DetrConfig(use_timm_backbone=False, backbone_config=backbone_config)  # 创建DETR配置

    # 设置标签属性
    is_panoptic = "panoptic" in model_name  # 如果模型名称中包含"panoptic"，则is_panoptic为True
    if is_panoptic:
        config.num_labels = 250  # 如果是全视角模型，则标签数为250
    else:
        config.num_labels = 91  # 否则，标签数为91
        repo_id = "huggingface/label-files"  # 存储库ID
        filename = "coco-detection-id2label.json"  # 文件名
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))  # 从hub下载标签文件并加载
        id2label = {int(k): v for k, v in id2label.items()}  # 将id2label的键转换为整数
        config.id2label = id2label  # 设置id2label属性
        config.label2id = {v: k for k, v in id2label.items()}  # 设置label2id属性

    return config, is_panoptic  # 返回配置和是否为全视角模型的标志


def create_rename_keys(config):
    # 创建并重命名键
    # 这里列出需要重命名的所有键（原始名称在左边，我们的名称在右边）
    rename_keys = []

    # stem
    # fmt: off
    rename_keys.append(("backbone.0.body.conv1.weight", "backbone.conv_encoder.model.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.0.body.bn1.weight", "backbone.conv_encoder.model.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.0.body.bn1.bias", "backbone.conv_encoder.model.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.0.body.bn1.running_mean", "backbone.conv_encoder.model.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.0.body.bn1.running_var", "backbone.conv_encoder.model.embedder.embedder.normalization.running_var"))
    # stages
    # fmt: on
    # 将指定的键名重命名为新的键名，并将其添加到列表 `rename_keys` 中
    rename_keys.extend(
        [
            # 重命名卷积投影层的权重和偏置
            ("input_proj.weight", "input_projection.weight"),
            ("input_proj.bias", "input_projection.bias"),
            # 重命名查询嵌入层的权重
            ("query_embed.weight", "query_position_embeddings.weight"),
            # 重命名解码器的层归一化层的权重和偏置
            ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
            ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
            # 重命名类别嵌入层的权重和偏置
            ("class_embed.weight", "class_labels_classifier.weight"),
            ("class_embed.bias", "class_labels_classifier.bias"),
            # 重命名边界框嵌入层的权重和偏置
            ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
            ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
            ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
            ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),
            ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),
            ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),
        ]
    )
    # 返回重命名后的键名列表
    return rename_keys
# 根据给定的旧键名和新键名重新命名状态字典中的键值对
def rename_key(state_dict, old, new):
    # 弹出旧键名对应的值
    val = state_dict.pop(old)
    # 将弹出的值以新键名加入状态字典中
    state_dict[new] = val

# 从状态字典中读取查询、键和数值，如果是全景，则使用特定前缀
def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "detr."

    # 首先处理：Transformer 编码器
    for i in range(6):
        # 读取输入投影层的权重和偏差（在 PyTorch 的 MultiHeadAttention 中，这是一个矩阵加上偏差）
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        # 接下来，按顺序添加查询、键和数值到状态字典中
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
    # 接下来处理：Transformer 解码器（稍微复杂一些，因为还包括交叉注意力）
    # 循环迭代范围为0到5，即6次，对每个self-attention层的参数进行处理
    for i in range(6):
        # 读取当前self-attention层输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # 将查询、键、值的投影权重（顺序为查询、键、值）加入状态字典
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        # 读取当前cross-attention层输入投影层的权重和偏置
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        # 将查询、键、值的投影权重（顺序为查询、键、值）加入状态字典
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
# 准备一张可爱猫咪的图片用于验证结果
def prepare_img():
    # 图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 URL 打开图片，获取 Image 对象
    im = Image.open(requests.get(url, stream=True).raw)

    return im


# 不计算梯度
@torch.no_grad()
def convert_detr_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    复制/粘贴/调整模型的权重到我们的 DETR 结构中。
    """

    # 加载默认配置
    config, is_panoptic = get_detr_config(model_name)

    # 从 torch hub 加载原始模型
    model_name_to_original_name = {
        "detr-resnet-50": "detr_resnet50",
        "detr-resnet-101": "detr_resnet101",
    }
    logger.info(f"Converting model {model_name}...")
    # 通过 torch hub 加载预训练模型
    detr = torch.hub.load("facebookresearch/detr", model_name_to_original_name[model_name], pretrained=True).eval()
    state_dict = detr.state_dict()
    # 重命名键
    for src, dest in create_rename_keys(config):
        if is_panoptic:
            src = "detr." + src
        rename_key(state_dict, src, dest)
    # 查询、键和值矩阵需要特殊处理
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    # 重要：对于基础模型的每个键，我们需要在其前面添加一个前缀，因为头模型使用不同的属性
    prefix = "detr.model." if is_panoptic else "model."
    for key in state_dict.copy().keys():
        if is_panoptic:
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
            if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
                val = state_dict.pop(key)
                state_dict[prefix + key] = val

    # 最后，创建 HuggingFace 模型并加载状态字典
    model = DetrForSegmentation(config) if is_panoptic else DetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 在图片上验证我们的转换结果
    format = "coco_panoptic" if is_panoptic else "coco_detection"
    processor = DetrImageProcessor(format=format)

    encoding = processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    original_outputs = detr(pixel_values)
    outputs = model(pixel_values)

    # 验证输出是否相似
    assert torch.allclose(outputs.logits, original_outputs["pred_logits"], atol=1e-3)
    assert torch.allclose(outputs.pred_boxes, original_outputs["pred_boxes"], atol=1e-3)
    # 如果是全景模式
    if is_panoptic:
        # 断言预测的掩模与原始输出的掩模是接近的
        assert torch.allclose(outputs.pred_masks, original_outputs["pred_masks"], atol=1e-4)
    # 打印"Looks ok!"，表示一切正常
    print("Looks ok!")

    # 如果存在 pytorch_dump_folder_path
    if pytorch_dump_folder_path is not None:
        # 保存模型和图像处理器
        logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        # 如果目录不存在则创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 保存预训练模型到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 保存图像处理器到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 上传模型和图像处理器到 Hub
        logger.info("Uploading PyTorch model and image processor to the hub...")
        # 推送模型到 Hub 中
        model.push_to_hub(f"nielsr/{model_name}")
        # 推送图像处理器到 Hub 中
        processor.push_to_hub(f"nielsr/{model_name}")
# 如果当前文件被运行，而不是被导入，则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加一个参数：模型名称
    parser.add_argument(
        "--model_name",
        default="detr-resnet-50",
        type=str,
        choices=["detr-resnet-50", "detr-resnet-101"],
        help="Name of the DETR model you'd like to convert.",
    )
    
    # 添加一个参数：输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    
    # 添加一个参数：是否将模型推送到 hub
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 convert_detr_checkpoint 函数，将模型转换为 PyTorch 格式
    convert_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```