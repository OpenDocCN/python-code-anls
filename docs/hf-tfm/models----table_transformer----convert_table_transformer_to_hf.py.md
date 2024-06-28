# `.\models\table_transformer\convert_table_transformer_to_hf.py`

```py
# 设置文件编码为 UTF-8
# 版权声明：2022 年由 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本（“许可证”）进行许可；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 软件不附带任何明示或暗示的担保或条件。
# 有关具体语言的条款，请参阅许可证。
"""使用 timm-backbone 转换 Table Transformer 检查点。

URL: https://github.com/microsoft/table-transformer
"""


import argparse  # 导入命令行参数解析模块
from collections import OrderedDict  # 导入有序字典模块
from pathlib import Path  # 导入路径操作模块

import torch  # 导入 PyTorch 模块
from huggingface_hub import hf_hub_download  # 从 huggingface_hub 下载模块
from PIL import Image  # 导入 PIL 图像处理模块
from torchvision.transforms import functional as F  # 导入 torchvision 的变换功能

from transformers import DetrImageProcessor, TableTransformerConfig, TableTransformerForObjectDetection  # 导入 Transformers 模块
from transformers.utils import logging  # 导入日志记录模块


logging.set_verbosity_info()  # 设置日志记录级别为信息
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 这里列出所有要重命名的键（原始名称在左侧，我们的名称在右侧）
rename_keys = []
for i in range(6):
    # 编码器层：输出投影、两个前馈神经网络和两个层归一化
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
    # 解码器层：两次输出投影、两个前馈神经网络和三个层归一化
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
    )
    # 将以下两个键添加到重命名键列表中，用于对应变换后的模型参数命名
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.multihead_attn.out_proj.weight",
            f"decoder.layers.{i}.encoder_attn.out_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.multihead_attn.out_proj.bias",
            f"decoder.layers.{i}.encoder_attn.out_proj.bias",
        )
    )
    
    # 将以下两个键添加到重命名键列表中，用于对应变换后的模型参数命名
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"decoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"decoder.layers.{i}.fc1.bias"))
    
    # 将以下两个键添加到重命名键列表中，用于对应变换后的模型参数命名
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"decoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"decoder.layers.{i}.fc2.bias"))
    
    # 将以下两个键添加到重命名键列表中，用于对应变换后的模型参数命名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias"))
    
    # 将以下两个键添加到重命名键列表中，用于对应变换后的模型参数命名
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")
    )
    
    # 将以下两个键添加到重命名键列表中，用于对应变换后的模型参数命名
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))
# 扩展重命名键列表，用于转换模型参数命名
rename_keys.extend(
    [
        ("input_proj.weight", "input_projection.weight"),  # 将输入投影层权重重命名为input_projection.weight
        ("input_proj.bias", "input_projection.bias"),  # 将输入投影层偏置重命名为input_projection.bias
        ("query_embed.weight", "query_position_embeddings.weight"),  # 将查询嵌入权重重命名为query_position_embeddings.weight
        ("transformer.encoder.norm.weight", "encoder.layernorm.weight"),  # 将编码器层归一化层权重重命名为encoder.layernorm.weight
        ("transformer.encoder.norm.bias", "encoder.layernorm.bias"),  # 将编码器层归一化层偏置重命名为encoder.layernorm.bias
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),  # 将解码器层归一化层权重重命名为decoder.layernorm.weight
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),  # 将解码器层归一化层偏置重命名为decoder.layernorm.bias
        ("class_embed.weight", "class_labels_classifier.weight"),  # 将类别嵌入权重重命名为class_labels_classifier.weight
        ("class_embed.bias", "class_labels_classifier.bias"),  # 将类别嵌入偏置重命名为class_labels_classifier.bias
        ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),  # 将边界框嵌入第一层权重重命名为bbox_predictor.layers.0.weight
        ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),  # 将边界框嵌入第一层偏置重命名为bbox_predictor.layers.0.bias
        ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),  # 将边界框嵌入第二层权重重命名为bbox_predictor.layers.1.weight
        ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),  # 将边界框嵌入第二层偏置重命名为bbox_predictor.layers.1.bias
        ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),  # 将边界框嵌入第三层权重重命名为bbox_predictor.layers.2.weight
        ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),  # 将边界框嵌入第三层偏置重命名为bbox_predictor.layers.2.bias
    ]
)


def rename_key(state_dict, old, new):
    # 从状态字典中弹出旧键，并用新键重新添加值
    val = state_dict.pop(old)
    state_dict[new] = val


def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "backbone.0.body" in key:
            # 将backbone.0.body替换为backbone.conv_encoder.model作为新键
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def read_in_q_k_v(state_dict):
    prefix = ""

    # 第一部分：处理transformer编码器
    for i in range(6):
        # 读取编码器自注意力层中的输入投影层权重和偏置（在PyTorch的MultiHeadAttention中，这是一个矩阵加偏置）
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        # 将查询、键和值（按顺序）添加到状态字典中
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
    # 第二部分：处理transformer解码器（稍微复杂一些，因为它还包括交叉注意力）
    # 对每个层次的自注意力输入投影层的权重和偏置进行读取
    in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
    in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
    # 将权重切片并添加到状态字典中作为查询、键和值的投影
    state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
    state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
    state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
    state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
    state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
    state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
    # 读取每个层次的交叉注意力输入投影层的权重和偏置
    in_proj_weight_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight")
    in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
    # 将权重切片并添加到状态字典中作为交叉注意力的查询、键和值的投影
    state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
    state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
    state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
    state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
    state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
    state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
# 调整图像大小至指定的最大尺寸，保持宽高比不变
def resize(image, checkpoint_url):
    # 获取图像的宽度和高度
    width, height = image.size
    # 计算当前图像宽高中的最大值
    current_max_size = max(width, height)
    # 根据检查点 URL 判断目标最大尺寸
    target_max_size = 800 if "detection" in checkpoint_url else 1000
    # 计算缩放比例
    scale = target_max_size / current_max_size
    # 缩放图像，并返回缩放后的图像对象
    resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

    return resized_image


# 对图像进行标准化处理，转换为张量并进行归一化
def normalize(image):
    # 使用 TorchVision 将 PIL 图像转换为张量
    image = F.to_tensor(image)
    # 根据指定的均值和标准差进行图像归一化处理
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image


@torch.no_grad()
# 转换表格 Transformer 检查点
def convert_table_transformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """
    
    logger.info("Converting model...")

    # 从指定 URL 加载原始模型状态字典，使用 CPU 运行
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 重命名模型状态字典中的键名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 对骨干网络键名进行进一步重命名处理
    state_dict = rename_backbone_keys(state_dict)
    # 处理查询、键和值矩阵的特殊情况
    read_in_q_k_v(state_dict)
    # 需要在基础模型键名前添加前缀，因为头模型使用不同的属性
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val
    # 创建 HuggingFace 模型并加载状态字典
    config = TableTransformerConfig(
        backbone="resnet18",
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

    # 创建 DETR 图像处理器实例
    image_processor = DetrImageProcessor(
        format="coco_detection", max_size=800 if "detection" in checkpoint_url else 1000
    )
    # 创建表格 Transformer 目标检测模型实例并加载状态字典
    model = TableTransformerForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 验证转换后的模型
    filename = "example_pdf.png" if "detection" in checkpoint_url else "example_table.png"
    # 从 HuggingFace Hub 下载指定文件
    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename=filename)
    # 打开图像文件，并将其转换为RGB格式的图像对象
    image = Image.open(file_path).convert("RGB")
    # 调整图像大小并进行归一化处理，然后添加一个批次维度
    pixel_values = normalize(resize(image, checkpoint_url)).unsqueeze(0)

    # 使用模型进行推理，得到输出结果
    outputs = model(pixel_values)

    # 根据checkpoint_url判断模型预期输出的形状和内容
    if "detection" in checkpoint_url:
        # 如果是检测模型，预期输出的形状是(1, 15, 3)
        expected_shape = (1, 15, 3)
        # 预期的分类得分(logits)
        expected_logits = torch.tensor(
            [[-6.7897, -16.9985, 6.7937], [-8.0186, -22.2192, 6.9677], [-7.3117, -21.0708, 7.4055]]
        )
        # 预期的边界框
        expected_boxes = torch.tensor([[0.4867, 0.1767, 0.6732], [0.6718, 0.4479, 0.3830], [0.4716, 0.1760, 0.6364]])

    else:
        # 如果是结构识别模型，预期输出的形状是(1, 125, 7)
        expected_shape = (1, 125, 7)
        # 预期的分类得分(logits)
        expected_logits = torch.tensor(
            [[-18.1430, -8.3214, 4.8274], [-18.4685, -7.1361, -4.2667], [-26.3693, -9.3429, -4.9962]]
        )
        # 预期的边界框
        expected_boxes = torch.tensor([[0.4983, 0.5595, 0.9440], [0.4916, 0.6315, 0.5954], [0.6108, 0.8637, 0.1135]])

    # 断言检查模型输出的形状是否符合预期
    assert outputs.logits.shape == expected_shape
    # 断言检查模型输出的分类得分是否与预期一致（使用指定的容差）
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4)
    # 断言检查模型输出的边界框是否与预期一致（使用指定的容差）
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-4)
    # 输出提示信息，表明断言检查通过
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        # 如果提供了PyTorch模型保存路径，则保存模型和图像处理器
        logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        # 确保保存路径存在，若不存在则创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将图像处理器保存到指定路径
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # 如果需要将模型推送到Hub
        logger.info("Pushing model to the hub...")
        # 根据checkpoint_url选择对应的模型名称
        model_name = (
            "microsoft/table-transformer-detection"
            if "detection" in checkpoint_url
            else "microsoft/table-transformer-structure-recognition"
        )
        # 推送模型到Hub
        model.push_to_hub(model_name)
        # 推送图像处理器到Hub（与模型同名）
        image_processor.push_to_hub(model_name)
# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数，用于指定模型检查点的下载地址，默认为公共表格检测模型的地址
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
    
    # 添加命令行参数，用于指定输出 PyTorch 模型的文件夹路径，默认为 None
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the folder to output PyTorch model."
    )
    
    # 添加命令行参数，用于指定是否将转换后的模型推送到 🤗 hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub."
    )
    
    # 解析命令行参数，并将结果存储在 args 对象中
    args = parser.parse_args()
    
    # 调用函数 convert_table_transformer_checkpoint，传入命令行参数中指定的参数值
    convert_table_transformer_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
```