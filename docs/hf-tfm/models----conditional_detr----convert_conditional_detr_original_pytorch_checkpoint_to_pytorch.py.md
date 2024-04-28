# `.\models\conditional_detr\convert_conditional_detr_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# 引入需要使用的模块和类
import argparse                          # 命令行参数解析模块
import json                             # 处理json数据的模块
from collections import OrderedDict   # 有序字典模块
from pathlib import Path               # 处理文件系统路径的模块

import requests                    # 发送HTTP请求的模块
import torch                       # PyTorch深度学习框架
from huggingface_hub import hf_hub_download   # 从Hugging Face Hub下载模型权重的方法
from PIL import Image                      # 图像处理模块

from transformers import (
    ConditionalDetrConfig,         # 条件解码器的配置类
    ConditionalDetrForObjectDetection,     # 用于目标检测的条件DET编码器-解码器模型类
    ConditionalDetrForSegmentation,      # 用于分割任务的条件DET编码器-解码器模型类
    ConditionalDetrImageProcessor,          # 图像预处理器类
)
from transformers.utils import logging    # 消息打印工具类


logging.set_verbosity_info()    # 设置日志打印级别为info
logger = logging.get_logger(__name__)   # 创建一个名为__name__的日志记录器

# 定义需要重命名的键值对列表（原名称->新名称）
rename_keys = []
for i in range(6):
    # 编码器层：输出投影、两个前馈神经网络和两个层归一化层
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
    # 解码器层：两次输出投影、两个前馈神经网络和三个层归一化层
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
    )
    # 将要重命名的键添加到列表中，这些键是由一个模型的层命名的，现在需要与另一个模型的对应层进行匹配
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.weight",  # 要重命名的键的原始名称
            f"decoder.layers.{i}.encoder_attn.out_proj.weight",  # 要重命名的键的新名称
        )
    )
    
    # 同上，这里是另一个键的重命名
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.bias",
            f"decoder.layers.{i}.encoder_attn.out_proj.bias",
        )
    )
    
    # 同上，这里是另一个键的重命名
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"decoder.layers.{i}.fc1.weight"))
    
    # 同上，这里是另一个键的重命名
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"decoder.layers.{i}.fc1.bias"))
    
    # 同上，这里是另一个键的重命名
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"decoder.layers.{i}.fc2.weight"))
    
    # 同上，这里是另一个键的重命名
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"decoder.layers.{i}.fc2.bias"))
    
    # 同上，这里是另一个键的重命名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight")
    )
    
    # 同上，这里是另一个键的重命名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias")
    )
    
    # 同上，这里是另一个键的重命名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")
    )
    
    # 同上，这里是另一个键的重命名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")
    )
    
    # 同上，这里是另一个键的重命名
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight"))
    
    # 同上，这里是另一个键的重命名
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))
    
    # 对于条件性DETR模型中解码器的自注意力/交叉注意力中的q, k, v投影，进行键的重命名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qcontent_proj.weight", f"decoder.layers.{i}.sa_qcontent_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kcontent_proj.weight", f"decoder.layers.{i}.sa_kcontent_proj.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_qpos_proj.weight", f"decoder.layers.{i}.sa_qpos_proj.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_kpos_proj.weight", f"decoder.layers.{i}.sa_kpos_proj.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_v_proj.weight", f"decoder.layers.{i}.sa_v_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qcontent_proj.weight", f"decoder.layers.{i}.ca_qcontent_proj.weight")
    )
    # 注意：下面一行代码被注释掉了，可能是因为条件性DETR模型中不需要这个键的重命名
    # rename_keys.append((f"transformer.decoder.layers.{i}.ca_qpos_proj.weight", f"decoder.layers.{i}.ca_qpos_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kcontent_proj.weight", f"decoder.layers.{i}.ca_kcontent_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kpos_proj.weight", f"decoder.layers.{i}.ca_kpos_proj.weight")
    )
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_v_proj.weight", f"decoder.layers.{i}.ca_v_proj.weight"))
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.weight", f"decoder.layers.{i}.ca_qpos_sine_proj.weight")
    )
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qcontent_proj.bias", f"decoder.layers.{i}.sa_qcontent_proj.bias")
    )
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kcontent_proj.bias", f"decoder.layers.{i}.sa_kcontent_proj.bias")
    )
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_qpos_proj.bias", f"decoder.layers.{i}.sa_qpos_proj.bias"))
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_kpos_proj.bias", f"decoder.layers.{i}.sa_kpos_proj.bias"))
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_v_proj.bias", f"decoder.layers.{i}.sa_v_proj.bias"))
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qcontent_proj.bias", f"decoder.layers.{i}.ca_qcontent_proj.bias")
    )
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    # rename_keys.append((f"transformer.decoder.layers.{i}.ca_qpos_proj.bias", f"decoder.layers.{i}.ca_qpos_proj.bias"))
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kcontent_proj.bias", f"decoder.layers.{i}.ca_kcontent_proj.bias")
    )
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_kpos_proj.bias", f"decoder.layers.{i}.ca_kpos_proj.bias"))
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_v_proj.bias", f"decoder.layers.{i}.ca_v_proj.bias"))
    
    # 将特定格式的键值对添加到 rename_keys 列表中，用于重命名模型参数的键名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.bias", f"decoder.layers.{i}.ca_qpos_sine_proj.bias")
    )
# 将指定的旧键名和新键名对加入到rename_keys列表中
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

# 定义一个函数rename_key用于重命名state_dict中的键值
def rename_key(state_dict, old, new):
    val = state_dict.pop(old)  # 删除旧键名对应的值
    state_dict[new] = val  # 将旧键名对应的值赋给新键名

# 重命名state_dict中的backbone键值
def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():  # 遍历state_dict的键值对
        if "backbone.0.body" in key:  # 如果键名中包含"backbone.0.body"
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model")  # 替换键名中的部分字符串
            new_state_dict[new_key] = value  # 将新的键值对加入new_state_dict中
        else:
            new_state_dict[key] = value  # 否则保持原样加入new_state_dict中
    return new_state_dict  # 返回重命名后的state_dict

# 定义一个函数read_in_q_k_v用于读取state_dict中的q_k_v
def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""  # 初始化前缀为空字符串
    if is_panoptic:  # 如果是全景条件DETR
        prefix = "conditional_detr."  # 设置前缀为"conditional_detr."
    # first: transformer encoder  # 注释：首先处理transformer编码器部分
    # 遍历6次，逐个处理编码器的自注意力层
    for i in range(6):
        # 从状态字典中弹出输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        # 接下来，按顺序将查询、键和数值添加到状态字典
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
# 导入必要的模块和函数
from PIL import Image
import requests
import torch
import json
import logging
from .config import ConditionalDetrConfig
from .image_processor import ConditionalDetrImageProcessor
from .utils import rename_key, rename_backbone_keys, read_in_q_k_v

# 预加载模型的权重
@torch.no_grad()
def convert_conditional_detr_checkpoint(model_name, pytorch_dump_folder_path):
    """
    复制/粘贴/调整模型的权重到我们的CONDITIONAL_DETR结构中。
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
        config.num_labels = 250
    else:
        config.num_labels = 91
        # 从Hugging Face远程仓库下载coco标签文件
        repo_id = "huggingface/label-files"
        filename = "coco-detection-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        # 设置ID到标签和标签到ID的映射关系
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # 加载图像处理器
    format = "coco_panoptic" if is_panoptic else "coco_detection"
    image_processor = ConditionalDetrImageProcessor(format=format)

    # 准备图像
    img = prepare_img()
    # 将图像编码成张量
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    # 输出当前正在转换的模型
    logger.info(f"Converting model {model_name}...")

    # 从torch hub加载原始模型
    conditional_detr = torch.hub.load("DeppMeng/ConditionalDETR", model_name, pretrained=True).eval()
    # 获取模型的状态字典
    state_dict = conditional_detr.state_dict()
    # 重命名模型权重的键
    for src, dest in rename_keys:
        if is_panoptic:
            src = "conditional_detr." + src
        rename_key(state_dict, src, dest)
    # 重命名骨干网络权重的键
    state_dict = rename_backbone_keys(state_dict)
    # 需要对查询、键和值矩阵进行特殊处理
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    # 重要提示：我们需要为基础模型的每个键添加一个前缀，因为头模块使用不同的属性来命名它们
    prefix = "conditional_detr.model." if is_panoptic else "model."
    # 遍历 state_dict 字典的 key（浅拷贝），即原始参数模型中的所有键
    for key in state_dict.copy().keys():
        # 如果是全景分割模型
        if is_panoptic:
            # 对于以 "conditional_detr" 开头但不是以 "class_labels_classifier" 和 "bbox_predictor" 开头的键
            if (
                key.startswith("conditional_detr")
                and not key.startswith("class_labels_classifier")
                and not key.startswith("bbox_predictor")
            ):
                # 弹出该键对应的值，并以新的键名存入 state_dict 中
                val = state_dict.pop(key)
                state_dict["conditional_detr.model" + key[4:]] = val
            # 对于包含 "class_labels_classifier" 和 "bbox_predictor" 的键
            elif "class_labels_classifier" in key or "bbox_predictor" in key:
                # 弹出该键对应的值，并以新的键名存入 state_dict 中
                val = state_dict.pop(key)
                state_dict["conditional_detr." + key] = val
            # 对于以 "bbox_attention" 或 "mask_head" 开头的键
            elif key.startswith("bbox_attention") or key.startswith("mask_head"):
                # 跳过继续下一次循环
                continue
            # 对于其它未提及的键
            else:
                # 弹出该键对应的值，并以 prefix + key 的新键名存入 state_dict 中
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        else:
            # 对于非全景模型的情况下，不以 "class_labels_classifier" 和 "bbox_predictor" 开头的键
            if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
                # 弹出该键对应的值，并以 prefix + key 的新键名存入 state_dict 中
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
    # 最终，根据 is_panoptic 创建 HuggingFace 模型并加载 state_dict
    model = ConditionalDetrForSegmentation(config) if is_panoptic else ConditionalDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    # 将模型推送到 Hub 中
    model.push_to_hub(repo_id=model_name, organization="DepuMeng", commit_message="Add model")
    # 验证转换的准确性
    original_outputs = conditional_detr(pixel_values)
    outputs = model(pixel_values)
    assert torch.allclose(outputs.logits, original_outputs["pred_logits"], atol=1e-4)
    assert torch.allclose(outputs.pred_boxes, original_outputs["pred_boxes"], atol=1e-4)
    # 如果是全景分割，要验证预测的掩模
    if is_panoptic:
        assert torch.allclose(outputs.pred_masks, original_outputs["pred_masks"], atol=1e-4)

    # 保存模型和图像处理器
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果当前模块被直接执行，则进行以下操作
if __name__ == "__main__":
    # 创建一个解析器对象
    parser = argparse.ArgumentParser()

    # 向解析器添加一个参数，表示模型名称，默认为"conditional_detr_resnet50"
    parser.add_argument(
        "--model_name",
        default="conditional_detr_resnet50",
        type=str,
        help="Name of the CONDITIONAL_DETR model you'd like to convert.",
    )

    # 向解析器添加一个参数，表示输出PyTorch模型的文件夹路径，默认为None
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )

    # 解析命令行参数，将其存储到args对象中
    args = parser.parse_args()

    # 调用函数convert_conditional_detr_checkpoint，传入模型名称和输出文件夹路径作为参数
    convert_conditional_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path)
```