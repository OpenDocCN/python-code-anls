# `.\models\sam\convert_sam_to_hf.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可信息，指出代码归属和使用许可
# 根据 Apache License, Version 2.0 许可，除非符合许可要求，否则不得使用此文件
"""
从原始仓库中转换 SAM 模型的检查点。

URL: https://github.com/facebookresearch/segment-anything.

同时支持从 https://github.com/czg1225/SlimSAM/tree/master 转换 SlimSAM 检查点。
"""
import argparse  # 导入命令行参数解析模块
import re  # 导入正则表达式模块

import numpy as np  # 导入处理数组的库
import requests  # 导入处理 HTTP 请求的库
import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模型和数据
from PIL import Image  # 导入 Python Imaging Library 用于图像处理

from transformers import (  # 导入 Transformers 库的相关模块
    SamConfig,  # SAM 模型的配置类
    SamImageProcessor,  # 处理图像输入的 SAM 图像处理器
    SamModel,  # SAM 模型
    SamProcessor,  # SAM 数据处理器
    SamVisionConfig,  # SAM 视觉部分的配置类
)


def get_config(model_name):
    if "slimsam-50" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=384,  # 隐藏层大小
            mlp_dim=1536,  # MLP 层大小
            num_hidden_layers=12,  # 隐藏层层数
            num_attention_heads=12,  # 注意力头数
            global_attn_indexes=[2, 5, 8, 11],  # 全局注意力索引
        )
    elif "slimsam-77" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=168,
            mlp_dim=696,
            num_hidden_layers=12,
            num_attention_heads=12,
            global_attn_indexes=[2, 5, 8, 11],
        )
    elif "sam_vit_b" in model_name:
        vision_config = SamVisionConfig()  # 使用 SAM VIT_B 的默认配置
    elif "sam_vit_l" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        )
    elif "sam_vit_h" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        )

    config = SamConfig(
        vision_config=vision_config,  # 使用 SAM 的配置类来创建配置对象
    )

    return config


KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",  # 映射修改键值对
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
}
    # 定义一个字典，用于将旧模型的参数映射到新模型的对应参数上
    "point_embeddings": "point_embed",
    # 将旧模型中的 positional_encoding_gaussian_matrix 映射到新模型的 shared_embedding.positional_embedding
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    # 将旧模型中的 image_encoder 映射到新模型的 vision_encoder
    "image_encoder": "vision_encoder",
    # 将旧模型中的 neck.0 映射到新模型的 neck.conv1
    "neck.0": "neck.conv1",
    # 将旧模型中的 neck.1 映射到新模型的 neck.layer_norm1
    "neck.1": "neck.layer_norm1",
    # 将旧模型中的 neck.2 映射到新模型的 neck.conv2
    "neck.2": "neck.conv2",
    # 将旧模型中的 neck.3 映射到新模型的 neck.layer_norm2
    "neck.3": "neck.layer_norm2",
    # 将旧模型中的 patch_embed.proj 映射到新模型的 patch_embed.projection
    "patch_embed.proj": "patch_embed.projection",
    # 将旧模型中所有以 .norm 结尾的参数映射到新模型中以 .layer_norm 结尾的对应参数
    ".norm": ".layer_norm",
    # 将旧模型中的 blocks 映射到新模型的 layers
    "blocks": "layers",
}

# 替换模型状态字典中的键值，去除特定键"pixel_mean"和"pixel_std"
def replace_keys(state_dict):
    model_state_dict = {}
    state_dict.pop("pixel_mean", None)
    state_dict.pop("pixel_std", None)

    # 定义匹配模式，用于识别特定的键名格式
    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"

    # 遍历输入的状态字典中的每个键值对
    for key, value in state_dict.items():
        # 遍历预定义的键映射字典，替换键名中的特定字符串
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        # 如果键名符合output_hypernetworks_mlps_pattern模式，则进行进一步处理
        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            # 根据layer_nb的值替换特定的键名部分
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        # 将处理后的键值对存入模型状态字典中
        model_state_dict[key] = value

    # 将一个特定键的值复制到另一个键中
    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]

    # 返回替换键后的模型状态字典
    return model_state_dict


# 将SAM模型检查点转换为PyTorch格式，并在必要时进行处理和推理
def convert_sam_checkpoint(model_name, checkpoint_path, pytorch_dump_folder, push_to_hub):
    # 获取指定模型的配置信息
    config = get_config(model_name)

    # 加载检查点文件中的状态字典（在CPU上加载）
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # 使用替换键函数处理状态字典
    state_dict = replace_keys(state_dict)

    # 创建SAM图像处理器和SAM处理器对象
    image_processor = SamImageProcessor()
    processor = SamProcessor(image_processor=image_processor)
    # 使用SAM模型配置创建SAM模型对象，并设为评估模式
    hf_model = SamModel(config)
    hf_model.eval()

    # 根据CUDA是否可用，将SAM模型移动到适当的设备上
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model.load_state_dict(state_dict)
    hf_model = hf_model.to(device)

    # 从URL加载原始图像，并将其转换为RGB格式
    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    # 设置输入的点和标签
    input_points = [[[500, 375]]]
    input_labels = [[1]]

    # 使用SAM处理器处理图像并将其转换为PyTorch张量格式
    inputs = processor(images=np.array(raw_image), return_tensors="pt").to(device)

    # 在不计算梯度的情况下进行模型推理
    with torch.no_grad():
        output = hf_model(**inputs)
    # 提取IoU分数
    scores = output.iou_scores.squeeze()

    # 如果模型名称符合条件，则再次使用SAM处理器处理输入图像
    if model_name == "sam_vit_b_01ec64":
        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        # 在不计算梯度的情况下进行模型推理
        with torch.no_grad():
            output = hf_model(**inputs)
            # 提取IoU分数
            scores = output.iou_scores.squeeze()
    # 如果模型名称为 "sam_vit_h_4b8939"，执行以下操作
    elif model_name == "sam_vit_h_4b8939":
        # 使用 processor 处理原始图像数据，输入关键点和标签，返回 PyTorch 张量
        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        # 在无需梯度的上下文中，使用 hf_model 进行推理
        with torch.no_grad():
            output = hf_model(**inputs)
        # 提取输出的 IOU 得分并压缩为一维张量
        scores = output.iou_scores.squeeze()

        # 断言最后一个得分是否等于特定值
        assert scores[-1].item() == 0.9712603092193604

        # 定义输入框的坐标范围
        input_boxes = ((75, 275, 1725, 850),)

        # 使用 processor 处理原始图像数据，输入框作为输入，返回 PyTorch 张量
        inputs = processor(images=np.array(raw_image), input_boxes=input_boxes, return_tensors="pt").to(device)

        # 在无需梯度的上下文中，使用 hf_model 进行推理
        with torch.no_grad():
            output = hf_model(**inputs)
        # 提取输出的 IOU 得分并压缩为一维张量
        scores = output.iou_scores.squeeze()

        # 断言最后一个得分是否等于特定值
        assert scores[-1].item() == 0.8686015605926514

        # 测试包含两个关键点和一个图像的情况
        input_points = [[[400, 650], [800, 650]]]
        input_labels = [[1, 1]]

        # 使用 processor 处理原始图像数据，输入关键点和标签，返回 PyTorch 张量
        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        # 在无需梯度的上下文中，使用 hf_model 进行推理
        with torch.no_grad():
            output = hf_model(**inputs)
        # 提取输出的 IOU 得分并压缩为一维张量
        scores = output.iou_scores.squeeze()

        # 断言最后一个得分是否等于特定值
        assert scores[-1].item() == 0.9936047792434692

    # 如果 pytorch_dump_folder 不为 None，则保存 processor 和 hf_model 到指定文件夹
    if pytorch_dump_folder is not None:
        processor.save_pretrained(pytorch_dump_folder)
        hf_model.save_pretrained(pytorch_dump_folder)

    # 如果 push_to_hub 为 True，则根据模型名称推送到指定的 Hub 仓库
    if push_to_hub:
        # 如果模型名称中包含 "slimsam"，使用特定格式的 repo_id
        repo_id = f"nielsr/{model_name}" if "slimsam" in model_name else f"meta/{model_name}"
        # 将 processor 和 hf_model 推送到 Hub 仓库中
        processor.push_to_hub(repo_id)
        hf_model.push_to_hub(repo_id)
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 定义可选的模型名称列表
    choices = ["sam_vit_b_01ec64", "sam_vit_h_4b8939", "sam_vit_l_0b3195", "slimsam-50-uniform", "slimsam-77-uniform"]
    # 添加命令行参数：模型名称，包括默认值、可选值、类型和帮助信息
    parser.add_argument(
        "--model_name",
        default="sam_vit_h_4b8939",
        choices=choices,
        type=str,
        help="Name of the original model to convert",
    )
    # 添加命令行参数：检查点路径，包括类型和是否必需
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the original checkpoint",
    )
    # 添加命令行参数：PyTorch 模型输出路径，默认为 None，包括帮助信息
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数：推送到 Hub 的标志，是一个布尔值，包括帮助信息
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 如果模型名称中包含 'slimsam'
    if "slimsam" in args.model_name:
        # 检查点路径为命令行参数提供的检查点路径，如果未提供则抛出错误
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            raise ValueError("You need to provide a checkpoint path for SlimSAM models.")
    else:
        # 使用 Hugging Face Hub 下载指定模型名称的检查点文件路径
        checkpoint_path = hf_hub_download("ybelkada/segment-anything", f"checkpoints/{args.model_name}.pth")

    # 调用函数：转换 SAM 模型的检查点到 PyTorch 模型
    convert_sam_checkpoint(args.model_name, checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
```