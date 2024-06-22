# `.\transformers\models\table_transformer\convert_table_transformer_to_hf_no_timm.py`

```
# 设置脚本的编码格式为 UTF-8
# 版权声明及许可协议
# 在 Apache 许可协议下授权，允许在遵循许可协议的前提下使用本文件
# 获取许可协议的副本
# 如果未依据许可协议的规定使用本文件，可能会导致法律责任
"""将 Table Transformer 检查点转换为带有原生（Transformers）后端的格式。

URL: https://github.com/microsoft/table-transformer
"""


import argparse  # 导入解析命令行参数的模块
from pathlib import Path  # 导入处理文件路径的模块

import torch  # 导入 PyTorch 模块
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模块
from PIL import Image  # 导入处理图像的 PIL 模块
from torchvision.transforms import functional as F  # 导入 TorchVision 的图像变换模块

from transformers import DetrImageProcessor, ResNetConfig, TableTransformerConfig, TableTransformerForObjectDetection  # 导入 Transformers 模块
from transformers.utils import logging  # 导入 Transformers 的日志模块


logging.set_verbosity_info()  # 设置日志输出级别为信息
logger = logging.get_logger(__name__)  # 获取日志记录器


def create_rename_keys(config):
    # 在此列出所有需要重命名的键（原始键名在左侧，我们的键名在右侧）
    rename_keys = []

    # stem
    # fmt: off
    # 对应于 stem 部分
    rename_keys.append(("backbone.0.body.conv1.weight", "backbone.conv_encoder.model.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.0.body.bn1.weight", "backbone.conv_encoder.model.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.0.body.bn1.bias", "backbone.conv_encoder.model.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.0.body.bn1.running_mean", "backbone.conv_encoder.model.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.0.body.bn1.running_var", "backbone.conv_encoder.model.embedder.embedder.normalization.running_var"))
    # stages
    # fmt: on

    # 卷积投影 + 查询嵌入 + 解码器的层归一化 + 类别和边界框头部
    # 扩展重命名键列表，将对应关系以元组形式添加到列表中
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
            ("transformer.encoder.norm.weight", "encoder.layernorm.weight"),
            ("transformer.encoder.norm.bias", "encoder.layernorm.bias"),
        ]
    )

    # 返回包含重命名键的列表
    return rename_keys
# 重命名字典中的键
def rename_key(state_dict, old, new):
    # 弹出旧键对应的值
    val = state_dict.pop(old)
    # 添加新键和对应的值
    state_dict[new] = val

# 从状态字典中读取查询、键和值
def read_in_q_k_v(state_dict, is_panoptic=False):
    # 初始化前缀变量
    prefix = ""
    # 如果是全景视觉模式，则设置前缀为"detr."
    if is_panoptic:
        prefix = "detr."

    # 遍历处理transformer encoder的6层
    for i in range(6):
        # 从状态字典中弹出input projection层的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")

        # 将查询、键和值加入状态字典
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    # 处理transformer decoder，包含了交叉注意力机制等更复杂的部分
    for i in range(6):
        # 读取自注意力层的输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # 接下来，按顺序将查询、键和数值（顺序如上）添加到状态字典中
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        # 读取交叉注意力层的输入投影层的权重和偏置
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        # 接下来，按顺序将交叉注意力层的查询、键和数值添加到状态字典中
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
# 调整图像大小的函数，根据给定的检查点 URL 决定目标最大尺寸
def resize(image, checkpoint_url):
    # 获取图像的宽度和高度
    width, height = image.size
    # 计算当前图像的最大尺寸
    current_max_size = max(width, height)
    # 根据检查点 URL 决定目标最大尺寸
    target_max_size = 800 if "detection" in checkpoint_url else 1000
    # 计算缩放比例
    scale = target_max_size / current_max_size
    # 调整图像大小
    resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
    
    return resized_image


# 图像归一化函数，使用 PyTorch 提供的转换器将图像转换为张量并进行归一化
def normalize(image):
    # 将 PIL 图像转换为 PyTorch 张量
    image = F.to_tensor(image)
    # 使用给定的均值和标准差对图像进行归一化
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image


# 转换表格 Transformer 检查点的函数，将模型权重转换到我们的 DETR 结构中
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

    # 使用给定的参数创建 TableTransformerConfig 实例
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

    # 重命名键
    for src, dest in create_rename_keys(config):
        rename_key(state_dict, src, dest)
    # 查询、键和值矩阵需要特殊处理
    read_in_q_k_v(state_dict)
    # 重要：我们需要为基础模型的每个键添加前缀，因为头模型使用不同的属性
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val

    # 根据检查点 URL 决定配置的一些属性
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
    image_processor = DetrImageProcessor(format="coco_detection", size={"longest_edge": 800})
    # 创建 TableTransformerForObjectDetection 模型实例
    model = TableTransformerForObjectDetection(config)
    # 加载状态字典到模型
    model.load_state_dict(state_dict)
    # 设置模型为评估模式
    model.eval()

    # 验证我们的转换
    filename = "example_pdf.png" if "detection" in checkpoint_url else "example_table.png"
    # 从 HF hub 下载指定 repo_id 的文件，并返回文件路径
    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename=filename)
    # 打开并将图像转换为 RGB 模式
    image = Image.open(file_path).convert("RGB")
    # 标准化并调整图像大小，返回像素值并增加一个维度
    pixel_values = normalize(resize(image, checkpoint_url)).unsqueeze(0)

    # 使用模型处理像素值，产生输出结果
    outputs = model(pixel_values)

    # 如果是检测模型，则设置期望输出的形状、logits和边界框
    if "detection" in checkpoint_url:
        expected_shape = (1, 15, 3)
        expected_logits = torch.tensor(
            [[-6.7897, -16.9985, 6.7937], [-8.0186, -22.2192, 6.9677], [-7.3117, -21.0708, 7.4055]]
        )
        expected_boxes = torch.tensor([[0.4867, 0.1767, 0.6732], [0.6718, 0.4479, 0.3830], [0.4716, 0.1760, 0.6364]])

    # 如果不是检测模型，则设置另一种期望输出的形状、logits和边界框
    else:
        expected_shape = (1, 125, 7)
        expected_logits = torch.tensor(
            [[-18.1430, -8.3214, 4.8274], [-18.4685, -7.1361, -4.2667], [-26.3693, -9.3429, -4.9962]]
        )
        expected_boxes = torch.tensor([[0.4983, 0.5595, 0.9440], [0.4916, 0.6315, 0.5954], [0.6108, 0.8637, 0.1135]])

    # 断言输出结果的形状与期望形状相同
    assert outputs.logits.shape == expected_shape
    # 断言输出结果中的logits与期望的logits相似度在给定容差下一致
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4)
    # 断言输出结果中的边界框与期望的边界框相似度在给定容差下一致
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-4)
    # 打印 "Looks ok!"，表示输出结果符合预期
    print("Looks ok!")

    # 如果指定了 pytorch_dump_folder_path
    if pytorch_dump_folder_path is not None:
        # 保存模型和图像处理器到指定路径
        logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要上传到 HF hub
    if push_to_hub:
        # 记录上传模型到 HF hub 的信息
        logger.info("Pushing model to the hub...")
        # 根据模型类型选择模型名称
        model_name = (
            "microsoft/table-transformer-detection"
            if "detection" in checkpoint_url
            else "microsoft/table-transformer-structure-recognition"
        )
        # 将模型上传到 HF hub，并指定版本号
        model.push_to_hub(model_name, revision="no_timm")
        image_processor.push_to_hub(model_name, revision="no_timm")
# 如果是直接运行该脚本，则执行以下代码
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()

    # 添加参数--checkpoint_url，设置默认值并提供选择项和帮助信息
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
    # 添加参数--pytorch_dump_folder_path，设置默认值并提供帮助信息
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    # 添加参数--push_to_hub，设置为布尔类型，用于指定是否将转换后的模型推送到🤗hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数convert_table_transformer_checkpoint，传入解析后的参数
    convert_table_transformer_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
```