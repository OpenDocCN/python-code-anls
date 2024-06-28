# `.\models\table_transformer\convert_table_transformer_to_hf_no_timm.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert Table Transformer checkpoints with native (Transformers) backbone.

URL: https://github.com/microsoft/table-transformer
"""


import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import functional as F

from transformers import DetrImageProcessor, ResNetConfig, TableTransformerConfig, TableTransformerForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()  # 设置日志记录级别为信息级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def create_rename_keys(config):
    # here we list all keys to be renamed (original name on the left, our name on the right)
    rename_keys = []  # 创建一个空列表，用于存储需要重命名的键值对

    # stem
    # fmt: off
    rename_keys.append(("backbone.0.body.conv1.weight", "backbone.conv_encoder.model.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.0.body.bn1.weight", "backbone.conv_encoder.model.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.0.body.bn1.bias", "backbone.conv_encoder.model.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.0.body.bn1.running_mean", "backbone.conv_encoder.model.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.0.body.bn1.running_var", "backbone.conv_encoder.model.embedder.embedder.normalization.running_var"))
    # stages
    # fmt: on

    # convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
    # 将旧的模型参数名称与新的模型参数名称对应起来，用于重命名模型权重和偏置
    rename_keys.extend(
        [
            ("input_proj.weight", "input_projection.weight"),  # 重命名输入投影层的权重参数
            ("input_proj.bias", "input_projection.bias"),  # 重命名输入投影层的偏置参数
            ("query_embed.weight", "query_position_embeddings.weight"),  # 重命名查询位置嵌入的权重参数
            ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),  # 重命名解码器层归一化的权重参数
            ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),  # 重命名解码器层归一化的偏置参数
            ("class_embed.weight", "class_labels_classifier.weight"),  # 重命名类标签分类器的权重参数
            ("class_embed.bias", "class_labels_classifier.bias"),  # 重命名类标签分类器的偏置参数
            ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),  # 重命名边界框预测器第一层权重参数
            ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),  # 重命名边界框预测器第一层偏置参数
            ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),  # 重命名边界框预测器第二层权重参数
            ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),  # 重命名边界框预测器第二层偏置参数
            ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),  # 重命名边界框预测器第三层权重参数
            ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),  # 重命名边界框预测器第三层偏置参数
            ("transformer.encoder.norm.weight", "encoder.layernorm.weight"),  # 重命名编码器层归一化的权重参数
            ("transformer.encoder.norm.bias", "encoder.layernorm.bias"),  # 重命名编码器层归一化的偏置参数
        ]
    )
    
    # 返回重命名后的键列表
    return rename_keys
# 重命名状态字典中的键，将旧键（old）对应的值弹出，并将其存储在变量val中，然后将新键（new）和val的对应关系添加到状态字典中
def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val

# 从状态字典中读取查询（query）、键（keys）和值（values）的权重和偏置，并将它们重新组织存放到状态字典中
def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "detr."

    # 遍历六层Transformer编码器
    for i in range(6):
        # 读取输入投影层的权重和偏置（在PyTorch的MultiHeadAttention中，这是单个矩阵加偏置）
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")

        # 将查询投影的权重和偏置添加到状态字典中
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]

        # 将键投影的权重和偏置添加到状态字典中
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]

        # 将值投影的权重和偏置添加到状态字典中
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    # 接下来处理Transformer解码器（稍复杂，因为它还涉及跨注意力机制的处理）
    # 对于每个层次索引 i 在范围内从 0 到 5（共6个层次）
    for i in range(6):
        # 读取 self-attention 的输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        
        # 将查询、键和值（按顺序）添加到状态字典中的 self-attention 部分
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        
        # 读取 cross-attention 的输入投影层的权重和偏置
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        
        # 将查询、键和值（按顺序）添加到状态字典中的 cross-attention 部分
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
# 调整图像大小函数，根据指定的检查点 URL 判断目标大小，将图像调整为适当的尺寸
def resize(image, checkpoint_url):
    # 获取图像的宽度和高度
    width, height = image.size
    # 计算当前图像的最大尺寸
    current_max_size = max(width, height)
    # 根据检查点 URL 决定目标最大尺寸，检测模型使用 800，其他情况使用 1000
    target_max_size = 800 if "detection" in checkpoint_url else 1000
    # 计算调整比例
    scale = target_max_size / current_max_size
    # 调整图像大小
    resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

    return resized_image


# 标准化图像函数，使用 PyTorch 的转换工具进行图像标准化处理
def normalize(image):
    # 将图像转换为张量
    image = F.to_tensor(image)
    # 根据指定的均值和标准差对图像进行标准化
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image


# 转换表格 Transformer 模型的检查点函数，加载模型权重并进行转换操作
@torch.no_grad()
def convert_table_transformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """

    logger.info("Converting model...")

    # 创建 HuggingFace 模型并加载状态字典
    backbone_config = ResNetConfig.from_pretrained(
        "microsoft/resnet-18", out_features=["stage1", "stage2", "stage3", "stage4"]
    )

    # 使用给定配置创建 TableTransformerConfig 对象
    config = TableTransformerConfig(
        backbone_config=backbone_config,
        use_timm_backbone=False,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        ce_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.4,
        class_cost=1,
        bbox_cost=5,
        giou_cost=2,
    )

    # 加载原始状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    # 重命名键名
    for src, dest in create_rename_keys(config):
        rename_key(state_dict, src, dest)
    # 处理查询、键和值矩阵需要特殊处理
    read_in_q_k_v(state_dict)
    # 重要：对基础模型键名添加前缀，因为头部模型使用不同的属性
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val

    # 根据检查点 URL 设置不同的配置参数
    if "detection" in checkpoint_url:
        config.num_queries = 15
        config.num_labels = 2
        id2label = {0: "table", 1: "table rotated"}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        config.num_queries = 125
        config.num_labels = 6
        id2label = {
            0: "table",
            1: "table column",
            2: "table row",
            3: "table column header",
            4: "table projected row header",
            5: "table spanning cell",
        }
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # 创建图像处理器对象，指定输出格式为 coco_detection，最长边尺寸为 800
    image_processor = DetrImageProcessor(format="coco_detection", size={"longest_edge": 800})
    # 创建 TableTransformerForObjectDetection 模型对象
    model = TableTransformerForObjectDetection(config)
    # 加载转换后的状态字典
    model.load_state_dict(state_dict)
    # 设置模型为评估模式
    model.eval()

    # 验证转换结果
    filename = "example_pdf.png" if "detection" in checkpoint_url else "example_table.png"
    # 使用 hf_hub_download 函数下载指定 repository ID 的文件，并返回文件路径
    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename=filename)
    # 使用 PIL 库打开文件，并转换为 RGB 模式的图像对象
    image = Image.open(file_path).convert("RGB")
    # 调用 resize 函数对图像进行缩放并标准化像素值，然后添加一个维度以适应模型输入要求
    pixel_values = normalize(resize(image, checkpoint_url)).unsqueeze(0)

    # 将处理后的图像数据输入模型进行推断
    outputs = model(pixel_values)

    # 根据 checkpoint_url 是否包含 "detection" 字符串来设置预期的输出形状、logits 和 boxes
    if "detection" in checkpoint_url:
        expected_shape = (1, 15, 3)
        expected_logits = torch.tensor(
            [[-6.7897, -16.9985, 6.7937], [-8.0186, -22.2192, 6.9677], [-7.3117, -21.0708, 7.4055]]
        )
        expected_boxes = torch.tensor([[0.4867, 0.1767, 0.6732], [0.6718, 0.4479, 0.3830], [0.4716, 0.1760, 0.6364]])
    else:
        expected_shape = (1, 125, 7)
        expected_logits = torch.tensor(
            [[-18.1430, -8.3214, 4.8274], [-18.4685, -7.1361, -4.2667], [-26.3693, -9.3429, -4.9962]]
        )
        expected_boxes = torch.tensor([[0.4983, 0.5595, 0.9440], [0.4916, 0.6315, 0.5954], [0.6108, 0.8637, 0.1135]])

    # 使用断言检查模型输出的形状是否符合预期
    assert outputs.logits.shape == expected_shape
    # 使用断言检查模型输出的 logits 是否与预期的值在指定容差范围内相似
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4)
    # 使用断言检查模型输出的 pred_boxes 是否与预期的值在指定容差范围内相似
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-4)
    # 输出确认信息，表明检查通过
    print("Looks ok!")

    # 如果 pytorch_dump_folder_path 不为 None，则保存模型和图像处理器到指定路径
    if pytorch_dump_folder_path is not None:
        logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果 push_to_hub 为 True，则将模型推送到 HF hub
    if push_to_hub:
        logger.info("Pushing model to the hub...")
        # 根据 checkpoint_url 中是否包含 "detection" 字符串选择不同的模型名称
        model_name = (
            "microsoft/table-transformer-detection"
            if "detection" in checkpoint_url
            else "microsoft/table-transformer-structure-recognition"
        )
        # 调用模型对象的 push_to_hub 方法将模型推送到 HF hub
        model.push_to_hub(model_name, revision="no_timm")
        # 同样将图像处理器对象推送到 HF hub
        image_processor.push_to_hub(model_name, revision="no_timm")
# 如果当前脚本被直接执行（而不是被导入到其他模块），则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数：--checkpoint_url，用于指定模型检查点的下载链接，默认为公共模型的检查点链接
    parser.add_argument(
        "--checkpoint_url",
        default="https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth",
        type=str,
        choices=[
            "https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth",
            "https://pubtables1m.blob.core.windows.net/model/pubtables1m_structure_detr_r18.pth",
        ],
        help="URL of the Table Transformer checkpoint you'd like to convert."
    )

    # 添加命令行参数：--pytorch_dump_folder_path，用于指定输出 PyTorch 模型的文件夹路径，默认为 None
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the folder to output PyTorch model."
    )

    # 添加命令行参数：--push_to_hub，一个布尔标志，表示是否将转换后的模型推送到 🤗 hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数，并将它们存储在 args 对象中
    args = parser.parse_args()

    # 调用函数 convert_table_transformer_checkpoint，传入解析后的命令行参数
    convert_table_transformer_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
```