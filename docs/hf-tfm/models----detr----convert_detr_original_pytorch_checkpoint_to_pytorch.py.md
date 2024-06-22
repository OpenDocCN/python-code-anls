# `.\models\detr\convert_detr_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可证信息
"""Convert DETR checkpoints with timm backbone."""

# 导入所需模块和库
import argparse  # 解析命令行参数
import json  # JSON 格式数据的处理
from collections import OrderedDict  # 有序字典的使用
from pathlib import Path  # 处理文件路径的模块

import requests  # 发送 HTTP 请求
import torch  # PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 从 Hugging Face 模型中心下载模型
from PIL import Image  # 图像处理库

# 导入 DETR 模型相关的类和函数
from transformers import DetrConfig, DetrForObjectDetection, DetrForSegmentation, DetrImageProcessor
# 导入日志记录相关的函数和类
from transformers.utils import logging


# 设置日志记录的详细程度为信息级别
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义需要重命名的键列表，包含原始名称和新名称的元组
rename_keys = []
# 循环遍历范围为 0 到 5
for i in range(6):
    # 编码器层：输出投影、2个前馈神经网络和2个层归一化
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
    # 解码器层：2次输出投影、2个前馈神经网络和3个层归一化
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
    )
    # 添加将transformer.decoder.layers中的权重名称映射到decoder.layers中的权重名称的元组到重命名键列表
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.multihead_attn.out_proj.weight",
            f"decoder.layers.{i}.encoder_attn.out_proj.weight",
        )
    )
    # 添加将transformer.decoder.layers中的偏置名称映射到decoder.layers中的偏置名称的元组到重命名键列表
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.multihead_attn.out_proj.bias",
            f"decoder.layers.{i}.encoder_attn.out_proj.bias",
        )
    )
    # 添加将transformer.decoder.layers中的weight名称映射到decoder.layers中的weight名称的元组到重命名键列表
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"decoder.layers.{i}.fc1.weight"))
    # 添加将transformer.decoder.layers中的偏置名称映射到decoder.layers中的偏置名称的元组到重命名键列表
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"decoder.layers.{i}.fc1.bias"))
    # 添加将transformer.decoder.layers中的weight名称映射到decoder.layers中的weight名称的元组到重命名键列表
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"decoder.layers.{i}.fc2.weight"))
    # 添加将transformer.decoder.layers中的偏置名称映射到decoder.layers中的偏置名称的元组到重命名键列表
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"decoder.layers.{i}.fc2.bias"))
    # 添加将transformer.decoder.layers中的norm1权重名称映射到decoder.layers中的self_attn_layer_norm权重名称的元组到重命名键列表
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight")
    )
    # 添加将transformer.decoder.layers中的norm1偏置名称映射到decoder.layers中的self_attn_layer_norm偏置名称的元组到重命名键列表
    rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias"))
    # 添加将transformer.decoder.layers中的norm2权重名称映射到decoder.layers中的encoder_attn_layer_norm权重名称的元组到重命名键列表
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")
    )
    # 添加将transformer.decoder.layers中的norm2偏置名称映射到decoder.layers中的encoder_attn_layer_norm偏置名称的元组到重命名键列表
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")
    )
    # 添加将transformer.decoder.layers中的norm3权重���称映射到decoder.layers中的final_layer_norm权重名称的元组到重命名键列表
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight"))
    # 添加将transformer.decoder.layers中的norm3偏置名称映射到decoder.layers中的final_layer_norm偏置名称的元组到重命名键列表
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))
# 将指定键名的字典项进行重命名，用新的键名替换旧的键名
rename_keys.extend(
    [
        # 重命名卷积投影、查询嵌入、解码器的层归一化、类别和边界框头部的键名
        ("input_proj.weight", "input_projection.weight"),  # 输入投影层权重
        ("input_proj.bias", "input_projection.bias"),  # 输入投影层偏置
        ("query_embed.weight", "query_position_embeddings.weight"),  # 查询嵌入权重
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),  # 解码器层归一化权重
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),  # 解码器层归一化偏置
        ("class_embed.weight", "class_labels_classifier.weight"),  # 类别嵌入权重
        ("class_embed.bias", "class_labels_classifier.bias"),  # 类别嵌入偏置
        ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),  # 边界框嵌入层0权重
        ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),  # 边界框嵌入层0偏置
        ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),  # 边界框嵌入层1权重
        ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),  # 边界框嵌入层1偏置
        ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),  # 边界框嵌入层2权重
        ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),  # 边界框嵌入层2偏置
    ]
)

# 根据给定的旧键名和新键名，重命名字典中的键
def rename_key(state_dict, old, new):
    # 弹出旧键名对应的值
    val = state_dict.pop(old)
    # 将值插入字典，以新键名作为键
    state_dict[new] = val

# 对模型的骨干网络键名进行重命名，将backbone.0.body替换为backbone.conv_encoder.model
def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict

# 读取查询、键和值的权重和偏置，并将其添加到状态字典中
def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "detr."

    # 首先：transformer编码器
    for i in range(6):
        # 读取输入投影层（在PyTorch的MultiHeadAttention中，这是一个单矩阵和偏置）的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        # 接下来，将查询、键和值（按顺序）添加到状态字典中
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]  # 查询投影权重
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]  # 查询投影偏置
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]  # 键投影权重
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]  # 键投影偏置
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]  # 值投影权重
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]  # 值投影偏置
    # 接下来：transformer解码器（稍微复杂一些，因为它还包括跨注意力）
```  
    # 循环6次，读取自注意力层输入投影层的权重和偏置
    for i in range(6):
        # 读取自注意力层输入投影层的权重
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        # 读取自注意力层输入投影层的偏置
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # 将查询、键和值（按顺序）添加到状态字典中
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        # 读取交叉注意力层输入投影层的权重和偏置
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        # 将查询、键和值（按顺序）的交叉注意力添加到状态字典中
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
# 准备图像，从指定 URL 获取图像
def prepare_img():
    # 定义图像的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 从 URL 获取图像的原始数据流，并使用 Image 对象打开
    im = Image.open(requests.get(url, stream=True).raw)

    # 返回图像对象
    return im

# 禁用梯度
@torch.no_grad()
# 转换预训练模型权重到我们的 DETR 结构
def convert_detr_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """

    # 加载默认配置
    config = DetrConfig()
    # 设置骨干网络和扩张系数属性
    if "resnet101" in model_name:
        config.backbone = "resnet101"
    if "dc5" in model_name:
        config.dilation = True
    is_panoptic = "panoptic" in model_name
    # 如果是全景分割
    if is_panoptic:
        config.num_labels = 250
    # 如果是检测任务
    else:
        config.num_labels = 91
        repo_id = "huggingface/label-files"
        filename = "coco-detection-id2label.json"
        # 从 Hugging Face Hub 下载 COCO 标签文件
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # 加载图像处理器
    format = "coco_panoptic" if is_panoptic else "coco_detection"
    image_processor = DetrImageProcessor(format=format)

    # 准备图像
    img = prepare_img()
    # 对图像进行编码处理，返回张量
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    # 打印日志信息
    logger.info(f"Converting model {model_name}...")

    # 从 torch hub 加载原始模型
    detr = torch.hub.load("facebookresearch/detr", model_name, pretrained=True).eval()
    state_dict = detr.state_dict()
    # 重命名键
    for src, dest in rename_keys:
        if is_panoptic:
            src = "detr." + src
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    # 查询、键和值矩阵需要特殊处理
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    # 重要：在每个基础模型键的前面加上前缀，因为头模型使用不同的属性
    prefix = "detr.model." if is_panoptic else "model."
    # 遍历 state_dict 的 key 的副本
    for key in state_dict.copy().keys():
        # 如果是全景图像检测
        if is_panoptic:
            # 如果 key 以 "detr" 开头，并且不以 "class_labels_classifier" 或 "bbox_predictor" 开头
            if (
                key.startswith("detr")
                and not key.startswith("class_labels_classifier")
                and not key.startswith("bbox_predictor")
            ):
                # 弹出并获取对应值，并在 state_dict 中以"detr.model" + key[4:]的新 key 存入对应值
                val = state_dict.pop(key)
                state_dict["detr.model" + key[4:]] = val
            # 如果 key 包含 "class_labels_classifier" 或 "bbox_predictor"
            elif "class_labels_classifier" in key or "bbox_predictor" in key:
                # 弹出并获取对应值，并在 state_dict 中以"detr." + key的新 key 存入对应值
                val = state_dict.pop(key)
                state_dict["detr." + key] = val
            # 如果 key 以 "bbox_attention" 或 "mask_head" 开头，则跳过
            elif key.startswith("bbox_attention") or key.startswith("mask_head"):
                continue
            # 否则
            else:
                # 弹出并获取对应值，并在 state_dict 中以 prefix + key 的新 key 存入对应值
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        # 如果不是全景图像检测
        else:
            # 如果 key 不以 "class_labels_classifier" 或 "bbox_predictor" 开头
            if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
                # 弹出并获取对应值，并在 state_dict 中以 prefix + key 的新 key 存入对应值
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
    # 创建 HuggingFace 模型并加载状态字典
    model = DetrForSegmentation(config) if is_panoptic else DetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    # 设为评估模式
    model.eval()
    # 验证转换结果
    original_outputs = detr(pixel_values)
    outputs = model(pixel_values)
    # 检查预测 logits 是否相近
    assert torch.allclose(outputs.logits, original_outputs["pred_logits"], atol=1e-4)
    # 检查预测边界框是否相近
    assert torch.allclose(outputs.pred_boxes, original_outputs["pred_boxes"], atol=1e-4)
    # 如果是全景图像检测，则检查预测掩模是否相近
    if is_panoptic:
        assert torch.allclose(outputs.pred_masks, original_outputs["pred_masks"], atol=1e-4)

    # 保存模型和图像处理器
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    # 创建文件夹（如果不存在）
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 保存图像处理器到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数：模型名称，默认值为"detr_resnet50"，类型为字符串，帮助信息为"Name of the DETR model you'd like to convert."
    parser.add_argument(
        "--model_name", default="detr_resnet50", type=str, help="Name of the DETR model you'd like to convert."
    )
    # 添加命令行参数：PyTorch模型输出路径，默认为None，类型为字符串，帮助信息为"Path to the folder to output PyTorch model."
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用convert_detr_checkpoint方法，传入模型名称和PyTorch模型输出路径作为参数
    convert_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path)
```