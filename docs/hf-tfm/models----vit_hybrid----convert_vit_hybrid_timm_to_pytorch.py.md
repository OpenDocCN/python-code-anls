# `.\transformers\models\vit_hybrid\convert_vit_hybrid_timm_to_pytorch.py`

```py
# 设置文件编码为 utf-8
# 版权声明
#
# 根据 Apache 许可证 2.0 版本进行许可
# 除非符合许可，否则不得使用此文件
# 您可以获取许可的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，软件按“原样”分发
# 没有任何明示或暗示的担保或条件。
# 有关特定语言支持的条件和限制，请参阅许可证
"""将 timm 库的 ViT 混合检查点转换为 Hugging Face 混合检查点"""


# 导入所需的库
import argparse
import json
from pathlib import Path

import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from transformers import (
    BitConfig,
    ViTHybridConfig,
    ViTHybridForImageClassification,
    ViTHybridImageProcessor,
    ViTHybridModel,
)
from transformers.image_utils import PILImageResampling
from transformers.utils import logging


# 设置日志级别为 info
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)


# 在这里列出所有需要重命名的键值对（原始名称在左侧，我们的名称在右侧）
def create_rename_keys(config, base_model=False):
    rename_keys = []

    # fmt: off
    # stem:
    rename_keys.append(("cls_token", "vit.embeddings.cls_token"))
    rename_keys.append(("pos_embed", "vit.embeddings.position_embeddings"))

    rename_keys.append(("patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"))

    # backbone
    rename_keys.append(("patch_embed.backbone.stem.conv.weight", "vit.embeddings.patch_embeddings.backbone.bit.embedder.convolution.weight"))
    rename_keys.append(("patch_embed.backbone.stem.norm.weight", "vit.embeddings.patch_embeddings.backbone.bit.embedder.norm.weight"))
    rename_keys.append(("patch_embed.backbone.stem.norm.bias", "vit.embeddings.patch_embeddings.backbone.bit.embedder.norm.bias"))
    # 遍历配置中背景骨干深度的索引范围
    for stage_idx in range(len(config.backbone_config.depths)):
        # 遍历当前深度下的每个层的索引范围
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # 将当前层的权重重命名为对应的 ViT 模型中的权重
            rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv1.weight", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv1.weight"))
            # 将当前层的第一个标准化层的权重重命名为对应的 ViT 模型中的权重
            rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm1.weight", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm1.weight"))
            # 将当前层的第一个标准化层的偏置重命名为对应的 ViT 模型中的偏置
            rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm1.bias", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm1.bias"))
            # 将当前层的第二个卷积层的权重重命名为对应的 ViT 模型中的权重
            rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv2.weight", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv2.weight"))
            # 将当前层的第二个标准化层的权重重命名为对应的 ViT 模型中的权重
            rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm2.weight", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm2.weight"))
            # 将当前层的第二个标准化层的偏置重命名为对应的 ViT 模型中的偏置
            rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm2.bias", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm2.bias"))
            # 将当前层的第三个卷积层的权重重命名为对应的 ViT 模型中的权重
            rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv3.weight", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv3.weight"))
            # 将当前层的第三个标准化层的权重重命名为对应的 ViT 模型中的权重
            rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm3.weight", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm3.weight"))
            # 将当前层的第三个标准化层的偏置重命名为对应的 ViT 模型中的偏置
            rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm3.bias", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm3.bias"))
    
        # 将当前深度下第一个块的下采样卷积层的权重重命名为对应的 ViT 模型中的权重
        rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.conv.weight", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.conv.weight"))
        # 将当前深度下第一个块的下采样标准化层的权重重命名为对应的 ViT 模型中的权重
        rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.norm.weight", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.norm.weight"))
        # 将当前深度下第一个块的下采样标准化层的偏置重命名为对应的 ViT 模型中的偏置
        rename_keys.append((f"patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.norm.bias", f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.norm.bias"))
    
    # transformer encoder
    # 循环遍历配置中的隐藏层数量次数，用于处理编码器层的映射关系
    for i in range(config.num_hidden_layers):
        # 为编码器层的输出投影、两个前馈神经网络和两个 layernorm 添加映射关系
        rename_keys.append((f"blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))
    
    # 如果是基础模型，则添加 layernorm 和 pooler 层的映射关系
    if base_model:
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
                ("pre_logits.fc.weight", "pooler.dense.weight"),
                ("pre_logits.fc.bias", "pooler.dense.bias"),
            ]
        )
        # 如果只是基础模型，则从所有以 "vit" 开头的键中删除 "vit" 前缀
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # 如果不是基础模型，则添加 layernorm 和分类头部的映射关系
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),
                ("norm.bias", "vit.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )
    
    # 返回键的映射关系列表
    return rename_keys
# 将每个编码器层的矩阵拆分为查询、键和值
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""  # 如果是基本模型，则前缀为空
        else:
            prefix = "vit."  # 否则，前缀为"vit."
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")  # 读取输入投影层的权重
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")  # 读取输入投影层的偏置
        # 将查询、键和值（按顺序）添加到状态字典
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 从状态字典中移除分类头
def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 重命名键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 准备图像
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 图像的URL
    im = Image.open(requests.get(url, stream=True).raw)  # 打开并返回图像对象
    return im


@torch.no_grad()
def convert_vit_checkpoint(vit_name, pytorch_dump_folder_path, push_to_hub=False):
    # 定义默认的 ViT 混合配置
    backbone_config = BitConfig(
        global_padding="same",
        layer_type="bottleneck",
        depths=(3, 4, 9),
        out_features=["stage3"],
        embedding_dynamic_padding=True,
    )
    config = ViTHybridConfig(backbone_config=backbone_config, image_size=384, num_labels=1000)  # 设置配置
    base_model = False  # 是否为基本模型

    # 从 timm 加载原始模型
    timm_model = timm.create_model(vit_name, pretrained=True)  # 创建预训练模型
    timm_model.eval()

    # 加载原始模型的状态字典，并移除/重命名一些键
    state_dict = timm_model.state_dict()
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)  # 读取查询、键和值
    repo_id = "huggingface/label-files"  # 仓库ID
    filename = "imagenet-1k-id2label.json"  # 文件名
    # 加载 JSON 文件中的 id 到 label 的映射关系
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将 id 转换为整数
    id2label = {int(k): v for k, v in id2label.items()}
    # 将 id2label 存入 config 对象中
    config.id2label = id2label
    # 创建 label 到 id 的映射关系
    config.label2id = {v: k for k, v in id2label.items()}

    # 加载 HuggingFace 模型
    if vit_name[-5:] == "in21k":
        # 如果模型名称的倒数第五个字符到结尾是"in21k"，则创建 ViTHybridModel 对象
        model = ViTHybridModel(config).eval()
    else:
        # 否则创建 ViTHybridForImageClassification 对象
        model = ViTHybridForImageClassification(config).eval()
    # 加载模型参数
    model.load_state_dict(state_dict)
    
    # 创建图像处理器
    transform = create_transform(**resolve_data_config({}, model=timm_model))
    timm_transforms = transform.transforms

    # 定义 Pillow 中的不同重采样方式的映射关系
    pillow_resamplings = {
        "bilinear": PILImageResampling.BILINEAR,
        "bicubic": PILImageResampling.BICUBIC,
        "nearest": PILImageResampling.NEAREST,
    }

    # 创建 ViTHybridImageProcessor 对象
    processor = ViTHybridImageProcessor(
        do_resize=True,
        size={"shortest_edge": timm_transforms[0].size},
        resample=pillow_resamplings[timm_transforms[0].interpolation.value],
        do_center_crop=True,
        crop_size={"height": timm_transforms[1].size[0], "width": timm_transforms[1].size[1]},
        do_normalize=True,
        image_mean=timm_transforms[-1].mean.tolist(),
        image_std=timm_transforms[-1].std.tolist(),
    )

    # 准备图片数据
    image = prepare_img()
    # 对图片进行转换，返回像素值张量
    timm_pixel_values = transform(image).unsqueeze(0)
    # 使用图像处理器处理图片，返回处理后的像素值张量
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 验证像素值
    assert torch.allclose(timm_pixel_values, pixel_values)

    # 验证模型输出
    with torch.no_grad():
        # 获取模型输出
        outputs = model(pixel_values)
        # 获取logits
        logits = outputs.logits

    # 打印预测的类别
    print("Predicted class:", logits.argmax(-1).item())
    # 如果有基础模型
    if base_model:
        # 获取预训练模型的池化输出
        timm_pooled_output = timm_model.forward_features(pixel_values)
        # 断言预训练模型的池化输出形状与模型的输出形状相等
        assert timm_pooled_output.shape == outputs.pooler_output.shape
        # 断言预训练模型的池化输出与模型的池化输出内容接近
        assert torch.allclose(timm_pooled_output, outputs.pooler_output, atol=1e-3)

    # 如果没有基础模型
    else:
        # 获取预训练模型的logits
        timm_logits = timm_model(pixel_values)
        # 断言预训练模型的logits形状与模型的logits形状相等
        assert timm_logits.shape == outputs.logits.shape
        # 断言预训练模型的logits与模型的logits内容接近
        assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)
    print("Looks ok!")

    # 如果有指定的 pytroch_dump_folder_path
    if pytorch_dump_folder_path is not None:
        # 创建路径
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 保存模型参数到指定路径
        print(f"Saving model {vit_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # 保存图像处理器参数到指定路径
        print(f"Saving processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果要推送到hub
    if push_to_hub:
        # 将模型和图像处理器推送到Hugging Face hub
        print(f"Pushing model and processor to the hub {vit_name}")
        model.push_to_hub(f"ybelkada/{vit_name}")
        processor.push_to_hub(f"ybelkada/{vit_name}")
# 主程序入口，判断是不是在主程序中执行
if __name__ == "__main__":
    # 创建一个参数解析对象
    parser = argparse.ArgumentParser()
    # 必选参数
    # 添加 "--vit_name" 参数，设置默认值，参数类型为字符串，提供帮助信息
    parser.add_argument(
        "--vit_name",
        default="vit_base_r50_s16_384",
        type=str,
        help="Name of the hybrid ViT timm model you'd like to convert.",
    )
    # 添加 "--pytorch_dump_folder_path" 参数，设置默认值为None，参数类型为字符串，提供帮助信息
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加 "--push_to_hub" 参数，设置为布尔类型，提供帮助信息
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether to upload the model to the HuggingFace hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将命令行参数传入函数中
    convert_vit_checkpoint(args.vit_name, args.pytorch_dump_folder_path, args.push_to_hub)
```