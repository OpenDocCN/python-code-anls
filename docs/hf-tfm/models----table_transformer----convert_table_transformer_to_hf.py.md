# `.\transformers\models\table_transformer\convert_table_transformer_to_hf.py`

```
# 导入 argparse 模块用于解析命令行参数
import argparse
# 从 collections 模块中导入 OrderedDict 用于创建有序字典
from collections import OrderedDict
# 从 pathlib 模块中导入 Path 用于处理文件路径
from pathlib import Path
# 导入 torch 模块
import torch
# 从 huggingface_hub 模块中导入 hf_hub_download 用于从 Hugging Face Hub 下载内容
from huggingface_hub import hf_hub_download
# 从 PIL 模块中导入 Image 用于图像处理
from PIL import Image
# 从 torchvision.transforms 模块中导入 functional 用于图像变换
from torchvision.transforms import functional as F
# 从 transformers 模块中导入 DetrImageProcessor, TableTransformerConfig, TableTransformerForObjectDetection
from transformers import DetrImageProcessor, TableTransformerConfig, TableTransformerForObjectDetection
# 从 transformers.utils 模块中导入 logging
from transformers.utils import logging

# 设置日志记录级别为信息
logging.set_verbosity_info()
# 获取 logger
logger = logging.get_logger(__name__)

# 列出需要重命名的所有键值对（原始名称在左边，我们的名称在右边）
rename_keys = []
# 遍历 0 到 5，共6次
for i in range(6):
    # 编码器层: 输出投影、2个前向神经网络和2个层归一化
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
    # 解码器层: 2次输出投影、2个前向神经网络和3个层归一化
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
    )
    # 将指定的键值对元组添加到列表中，用于重命名模型参数
    rename_keys.append(
        # 重命名 transformer.decoder.layers.{i}.multihead_attn.out_proj.weight 为 decoder.layers.{i}.encoder_attn.out_proj.weight
        (
            f"transformer.decoder.layers.{i}.multihead_attn.out_proj.weight",
            f"decoder.layers.{i}.encoder_attn.out_proj.weight",
        )
    )
    # 依次重命名其他模型参数的键值对，每次添加到列表中
    ...
# convolutional projection + query embeddings + layernorm of encoder + layernorm of decoder + class and bounding box heads
# 将模型参数的键名从原有的名称映射到新的名称
rename_keys.extend(
    [
        # 输入映射层的权重和偏移
        ("input_proj.weight", "input_projection.weight"),
        ("input_proj.bias", "input_projection.bias"),
        # 查询位置嵌入的权重
        ("query_embed.weight", "query_position_embeddings.weight"),
        # 编码器和解码器的LayerNorm层的权重和偏移
        ("transformer.encoder.norm.weight", "encoder.layernorm.weight"),
        ("transformer.encoder.norm.bias", "encoder.layernorm.bias"),
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
        # 类别标签分类器的权重和偏移
        ("class_embed.weight", "class_labels_classifier.weight"),
        ("class_embed.bias", "class_labels_classifier.bias"),
        # 边界框预测器各层的权重和偏移
        ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
        ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
        ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
        ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),
        ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),
        ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),
    ]
)


# 将模型参数中的键名更改为新的名称
def rename_key(state_dict, old, new):
    # 从原有的state_dict中取出旧的参数值
    val = state_dict.pop(old)
    # 将参数值更新到新的键名下
    state_dict[new] = val


# 将backbone层的键名更改为新的名称
def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # 如果键名包含"backbone.0.body"，则替换为"backbone.conv_encoder.model"
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model")
            new_state_dict[new_key] = value
        # 否则保留原有的键名
        else:
            new_state_dict[key] = value
    return new_state_dict


# 读取并重组transformer层的query、key和value参数
def read_in_q_k_v(state_dict):
    prefix = ""

    # 处理transformer编码器层
    for i in range(6):
        # 从state_dict中取出输入映射层的权重和偏移
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        # 将权重和偏移拆分为query、key和value的对应参数
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
    # 处理transformer解码器层
    # (解码器层相比编码器层更加复杂,因为它还包含了跨注意力机制)
    # 遍历6次，分别处理每一层的自注意力中的输入投影层的权重和偏置
        # 弹出并获取自注意力中输入投影层权重
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        # 弹出并获取自注意力中输入投影层偏置
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # 将查询、键和值分别添加到状态字典中（顺序为查询、键、值）
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        # 弹出并获取交叉注意力中的输入投影层权重
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        # 弹出并获取交叉注意力中的输入投影层偏置
        in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        # 将查询、键和值（顺序为查询、键、值）添加到状态字典中的交叉注意力中
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
# 调整图像大小的函数
def resize(image, checkpoint_url):
    # 获取原始图像的宽度和高度
    width, height = image.size
    # 获取当前最大尺寸
    current_max_size = max(width, height)
    # 根据检查点 URL 判断目标最大尺寸
    target_max_size = 800 if "detection" in checkpoint_url else 1000
    # 计算缩放比例
    scale = target_max_size / current_max_size
    # 调整图像大小
    resized_image = image.resize((int(round(scale * width)), int(round(scale * height)))
    # 返回调整后的图像
    return resized_image

# 标准化图像的函数
def normalize(image):
    # 将图像转换为张量
    image = F.to_tensor(image)
    # 对图像进行归一化处理
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 返回标准化后的图像
    return image

# 无需梯度下降的函数
@torch.no_grad()
def convert_table_transformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    """
    复制/粘贴/调整模型的权重以适应我们的DETR结构。
    """

    logger.info("转换模型...")

    # 加载原始状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 重命名键
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    # 查询、键和值矩阵需要特殊处理
    read_in_q_k_v(state_dict)
    # 重要: 我们需要对基础模型的每个键添加前缀，因为头模型使用不同的属性
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val
    # 创建HuggingFace模型并加载状态字典
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

    image_processor = DetrImageProcessor(
        format="coco_detection", max_size=800 if "detection" in checkpoint_url else 1000
    )
    model = TableTransformerForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 验证我们的转换
    filename = "example_pdf.png" if "detection" in checkpoint_url else "example_table.png"
    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename=filename)
    # 打开图像文件并转为RGB格式
    image = Image.open(file_path).convert("RGB")
    # 标准化并调整图像大小，然后增加一个维度
    pixel_values = normalize(resize(image, checkpoint_url)).unsqueeze(0)

    # 将像素值输入模型，获取输出
    outputs = model(pixel_values)

    # 如果checkpoint_url中包含"detection"字样
    if "detection" in checkpoint_url:
        # 设置期望的输出形状和对应的期望logits和boxes
        expected_shape = (1, 15, 3)
        expected_logits = torch.tensor(
            [[-6.7897, -16.9985, 6.7937], [-8.0186, -22.2192, 6.9677], [-7.3117, -21.0708, 7.4055]]
        )
        expected_boxes = torch.tensor([[0.4867, 0.1767, 0.6732], [0.6718, 0.4479, 0.3830], [0.4716, 0.1760, 0.6364]])

    # 如果checkpoint_url中不包含"detection"字样
    else:
        # 设置期望的输出形状和对应的期望logits和boxes
        expected_shape = (1, 125, 7)
        expected_logits = torch.tensor(
            [[-18.1430, -8.3214, 4.8274], [-18.4685, -7.1361, -4.2667], [-26.3693, -9.3429, -4.9962]]
        )
        expected_boxes = torch.tensor([[0.4983, 0.5595, 0.9440], [0.4916, 0.6315, 0.5954], [0.6108, 0.8637, 0.1135]])

    # 断言输出的logits形状和值符合期望
    assert outputs.logits.shape == expected_shape
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-4)
    print("Looks ok!")

    # 如果有pytorch_dump_folder_path
    if pytorch_dump_folder_path is not None:
        # 保存模型和图像处理器到指定路径
        logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果要推送到hub
    if push_to_hub:
        # 推送模型到HF hub
        logger.info("Pushing model to the hub...")
        # 根据检查点url中是否包含"detection"字样选择不同的模型名
        model_name = (
            "microsoft/table-transformer-detection"
            if "detection" in checkpoint_url
            else "microsoft/table-transformer-structure-recognition"
        )
        model.push_to_hub(model_name)
        image_processor.push_to_hub(model_name)
# 如果当前脚本被直接运行而不是被导入，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数，用于指定检查点的 URL，默认为公共地址之一
    parser.add_argument(
        "--checkpoint_url",
        default="https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth",
        type=str,
        choices=[
            "https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth",
            "https://pubtables1m.blob.core.windows.net/model/pubtables1m_structure_detr_r18.pth",
        ],
        help="URL of the Table Transformer checkpoint you'd like to convert.",
    )
    
    # 添加命令行参数，用于指定输出 PyTorch 模型的文件夹路径，默认为 None
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    
    # 添加命令行参数，用于指定是否将转换后的模型推送到 🤗 hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    
    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()
    
    # 调用函数 convert_table_transformer_checkpoint，传递命令行参数中指定的参数
    convert_table_transformer_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
```