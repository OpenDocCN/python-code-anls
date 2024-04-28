# `.\transformers\models\sam\convert_sam_original_to_hf_format.py`

```
# 指定编码格式为 UTF-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 软件按"原样"分发，不附带任何形式的担保或条件，
# 包括但不限于默示担保或适销性或特定用途的适用性
# 有关许可证的详细信息，请参见许可证
"""
从原始存储库转换 SAM 检查点。
"""
# 导入所需模块
import argparse  # 导入命令行解析模块
import re  # 导入正则表达式模块

import numpy as np  # 导入 NumPy 库
import requests  # 导入 requests 库
import torch  # 导入 PyTorch 库
from huggingface_hub import hf_hub_download  # 导入 Hugging Face Hub 模块
from PIL import Image  # 导入 Python Imaging Library（PIL）模块

# 从 transformers 模块导入所需内容
from transformers import (
    SamConfig,  # 导入 SAM 配置类
    SamImageProcessor,  # 导入 SAM 图像处理器类
    SamModel,  # 导入 SAM 模型类
    SamProcessor,  # 导入 SAM 处理器类
    SamVisionConfig,  # 导入 SAM 视觉配置类
)

# 定义一个字典，用于存储需要修改的键
KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",  # 修改键名映射关系
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",  # 修改键名映射关系
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",  # 修改键名映射关系
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",  # 修改键名映射关系
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",  # 修改键名映射关系
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",  # 修改键名映射关系
    "mask_downscaling.0": "mask_embed.conv1",  # 修改键名映射关系
    "mask_downscaling.1": "mask_embed.layer_norm1",  # 修改键名映射关系
    "mask_downscaling.3": "mask_embed.conv2",  # 修改键名映射关系
    "mask_downscaling.4": "mask_embed.layer_norm2",  # 修改键名映射关系
    "mask_downscaling.6": "mask_embed.conv3",  # 修改键名映射关系
    "point_embeddings": "point_embed",  # 修改键名映射关系
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",  # 修改键名映射关系
    "image_encoder": "vision_encoder",  # 修改键名映射关系
    "neck.0": "neck.conv1",  # 修改键名映射关系
    "neck.1": "neck.layer_norm1",  # 修改键名映射关系
    "neck.2": "neck.conv2",  # 修改键名映射关系
    "neck.3": "neck.layer_norm2",  # 修改键名映射关系
    "patch_embed.proj": "patch_embed.projection",  # 修改键名映射关系
    ".norm": ".layer_norm",  # 修改键名映射关系
    "blocks": "layers",  # 修改键名映射关系
}

# 定义函数用于替换字典键名
def replace_keys(state_dict):
    model_state_dict = {}  # 初始化模型状态字典为空字典
    state_dict.pop("pixel_mean", None)  # 移除键为"pixel_mean"的值
    state_dict.pop("pixel_std", None)  # 移除键为"pixel_std"的值

    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"  # 定义正则表达式模式

    # 遍历状态字典的键值对
    for key, value in state_dict.items():
        # 遍历需要修改的键值对
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            # 如果需要修改的键存在于当前键中
            if key_to_modify in key:
                # 替换键名
                key = key.replace(key_to_modify, new_key)

        # 匹配键是否符合指定模式
        if re.match(output_hypernetworks_mlps_pattern, key):
            # 提取层编号
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            # 根据层编号进行键名替换
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        # 将键值对加入模型状态字典
        model_state_dict[key] = value
    # 将模型状态字典中的位置编码信息从一个键复制到另一个键
    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]
    
    # 返回更新后的模型状态字典
    return model_state_dict
# 将 SAM 模型转换为检查点格式，以便在 Hugging Face Hub 上发布
def convert_sam_checkpoint(model_name, pytorch_dump_folder, push_to_hub, model_hub_id="ybelkada/segment-anything"):
    # 下载 Hugging Face Hub 上的模型检查点
    checkpoint_path = hf_hub_download(model_hub_id, f"checkpoints/{model_name}.pth")

    # 根据模型名选择不同的配置
    if "sam_vit_b" in model_name:
        config = SamConfig()
    elif "sam_vit_l" in model_name:
        # 针对较大的 SAM-ViT 模型创建视觉配置
        vision_config = SamVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        )

        config = SamConfig(
            vision_config=vision_config,
        )
    elif "sam_vit_h" in model_name:
        # 针对更大的 SAM-ViT 模型创建视觉配置
        vision_config = SamVisionConfig(
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        )

        config = SamConfig(
            vision_config=vision_config,
        )

    # 从检查点文件中加载状态字典
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # 替换状态字典中的键
    state_dict = replace_keys(state_dict)

    # 创建 SAM 图像处理器
    image_processor = SamImageProcessor()

    # 创建 SAM 处理器
    processor = SamProcessor(image_processor=image_processor)
    # 创建 SAM 模型
    hf_model = SamModel(config)

    # 加载模型权重到 GPU
    hf_model.load_state_dict(state_dict)
    hf_model = hf_model.to("cuda")

    # 加载示例图像
    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    # 设置示例输入点和标签
    input_points = [[[400, 650]]]
    input_labels = [[1]]

    # 处理示例输入并移到 GPU
    inputs = processor(images=np.array(raw_image), return_tensors="pt").to("cuda")

    # 使用模型进行推理
    with torch.no_grad():
        output = hf_model(**inputs)
    # 获取预测的 IOU 分数
    scores = output.iou_scores.squeeze()

    # 如果模型名是 "sam_vit_h_4b8939"，执行额外的测试
    if model_name == "sam_vit_h_4b8939":
        # 断言最后一个 IOU 分数符合预期值
        assert scores[-1].item() == 0.579890251159668

        # 使用输入点和标签进行额外测试
        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output.iou_scores.squeeze()

        # 断言最后一个 IOU 分数符合预期值
        assert scores[-1].item() == 0.9712603092193604

        # 使用输入框进行额外测试
        input_boxes = ((75, 275, 1725, 850),)

        inputs = processor(images=np.array(raw_image), input_boxes=input_boxes, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output.iou_scores.squeeze()

        # 断言最后一个 IOU 分数符合预期值
        assert scores[-1].item() == 0.8686015605926514

        # 使用 2 个点和 1 张图像进行额外测试
        input_points = [[[400, 650], [800, 650]]]
        input_labels = [[1, 1]]

        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output.iou_scores.squeeze()

        # 断言最后一个 IOU 分数符合预期值
        assert scores[-1].item() == 0.9936047792434692


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 提供模型选择的选项列表
    choices = ["sam_vit_b_01ec64", "sam_vit_h_4b8939", "sam_vit_l_0b3195"]
    # 添加解析器参数，用于指定要转换的模型名称，默认为 sam_vit_h_4b8939，可选值在 choices 中
    parser.add_argument(
        "--model_name",
        default="sam_vit_h_4b8939",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    # 添加解析器参数，用于指定输出 PyTorch 模型的文件夹路径，默认为 None
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加解析器参数，用于指定是否在转换后将模型和处理器推送到 hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )
    # 添加解析器参数，用于指定要转换的模型在 hub 上的 ID，默认为 "ybelkada/segment-anything"，可选值在 choices 中
    parser.add_argument(
        "--model_hub_id",
        default="ybelkada/segment-anything",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数将 SAM 检查点转换为 PyTorch 模型
    convert_sam_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.model_hub_id)
```