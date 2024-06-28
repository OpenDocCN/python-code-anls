# `.\models\vit_hybrid\convert_vit_hybrid_timm_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert ViT hybrid checkpoints from the timm library."""

import argparse  # 导入用于解析命令行参数的模块
import json  # 导入处理 JSON 格式数据的模块
from pathlib import Path  # 导入处理路径的模块

import requests  # 导入发送 HTTP 请求的模块
import timm  # 导入用于训练和评估神经网络模型的模块
import torch  # 导入 PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 导入与 Hugging Face Hub 集成的下载功能
from PIL import Image  # 导入处理图像的模块
from timm.data import resolve_data_config  # 导入用于配置数据加载的函数
from timm.data.transforms_factory import create_transform  # 导入创建数据转换的工厂函数

from transformers import (
    BitConfig,  # 导入 Bit 模型的配置类
    ViTHybridConfig,  # 导入 ViT Hybrid 模型的配置类
    ViTHybridForImageClassification,  # 导入用于图像分类的 ViT Hybrid 模型类
    ViTHybridImageProcessor,  # 导入用于处理图像的 ViT Hybrid 图像处理器类
    ViTHybridModel,  # 导入 ViT Hybrid 模型类
)
from transformers.image_utils import PILImageResampling  # 导入图像重采样功能
from transformers.utils import logging  # 导入日志记录工具

logging.set_verbosity_info()  # 设置日志记录的详细级别为信息
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []  # 初始化空列表用于存储重命名的键值对

    # fmt: off
    # stem:
    rename_keys.append(("cls_token", "vit.embeddings.cls_token"))  # 添加 cls_token 的重命名映射
    rename_keys.append(("pos_embed", "vit.embeddings.position_embeddings"))  # 添加 pos_embed 的重命名映射

    rename_keys.append(("patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"))  # 添加 patch_embed.proj.weight 的重命名映射
    rename_keys.append(("patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"))  # 添加 patch_embed.proj.bias 的重命名映射

    # backbone
    rename_keys.append(("patch_embed.backbone.stem.conv.weight", "vit.embeddings.patch_embeddings.backbone.bit.embedder.convolution.weight"))  # 添加 patch_embed.backbone.stem.conv.weight 的重命名映射
    rename_keys.append(("patch_embed.backbone.stem.norm.weight", "vit.embeddings.patch_embeddings.backbone.bit.embedder.norm.weight"))  # 添加 patch_embed.backbone.stem.norm.weight 的重命名映射
    rename_keys.append(("patch_embed.backbone.stem.norm.bias", "vit.embeddings.patch_embeddings.backbone.bit.embedder.norm.bias"))  # 添加 patch_embed.backbone.stem.norm.bias 的重命名映射
    # fmt: on
    # 遍历配置中每个阶段的深度
    for stage_idx in range(len(config.backbone_config.depths)):
        # 遍历当前阶段的每个层级
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # 添加重命名键值对，将原始模型中的权重和偏置名称映射到新的Transformer模型的对应位置
            rename_keys.append((
                f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv1.weight",
                f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv1.weight"))
            rename_keys.append((
                f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm1.weight",
                f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm1.weight"))
            rename_keys.append((
                f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm1.bias",
                f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm1.bias"))
            rename_keys.append((
                f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv2.weight",
                f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv2.weight"))
            rename_keys.append((
                f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm2.weight",
                f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm2.weight"))
            rename_keys.append((
                f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm2.bias",
                f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm2.bias"))
            rename_keys.append((
                f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv3.weight",
                f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv3.weight"))
            rename_keys.append((
                f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm3.weight",
                f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm3.weight"))
            rename_keys.append((
                f"patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm3.bias",
                f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm3.bias"))

        # 添加重命名键值对，将原始模型中的第一个块的下采样卷积和规范化层的名称映射到Transformer模型的对应位置
        rename_keys.append((
            f"patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.conv.weight",
            f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.conv.weight"))
        rename_keys.append((
            f"patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.norm.weight",
            f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.norm.weight"))
        rename_keys.append((
            f"patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.norm.bias",
            f"vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.norm.bias"))

    # transformer encoder
    for i in range(config.num_hidden_layers):
        # 遍历编码器层次：输出投影、2个前馈神经网络和2个层归一化模块
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

    if base_model:
        # 如果是基础模型，进行下面的重命名操作：层归一化 + 池化器
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
                ("pre_logits.fc.weight", "pooler.dense.weight"),
                ("pre_logits.fc.bias", "pooler.dense.bias"),
            ]
        )

        # 如果仅仅是基础模型，需要从所有以 "vit" 开头的键名中移除 "vit"
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # 如果不是基础模型，进行下面的重命名操作：层归一化 + 分类头
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),
                ("norm.bias", "vit.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )
    # 格式化结束
    return rename_keys
# 将每个编码器层的权重矩阵拆分为查询（query）、键（key）和值（value）
def read_in_q_k_v(state_dict, config, base_model=False):
    # 遍历每个编码器层
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        
        # 读取输入投影层（在timm中是一个单独的矩阵加偏置）的权重和偏置
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        
        # 将查询（query）、键（key）和值（value）依次添加到state_dict中
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


# 移除state_dict中的分类头部权重和偏置
def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 将字典中的旧键（old）替换为新键（new）
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 准备一个可爱猫咪的图片，用于验证结果
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_vit_checkpoint(vit_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    将模型权重复制/粘贴/调整为我们的ViT结构。
    """

    # 定义默认的ViT混合配置
    backbone_config = BitConfig(
        global_padding="same",
        layer_type="bottleneck",
        depths=(3, 4, 9),
        out_features=["stage3"],
        embedding_dynamic_padding=True,
    )
    config = ViTHybridConfig(backbone_config=backbone_config, image_size=384, num_labels=1000)
    base_model = False

    # 从timm中加载原始模型
    timm_model = timm.create_model(vit_name, pretrained=True)
    timm_model.eval()

    # 加载原始模型的state_dict，移除并重命名一些键
    state_dict = timm_model.state_dict()
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model)  # 此处缺少create_rename_keys函数的定义，但在原代码中未提及
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)

    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    # 使用 HuggingFace Hub 下载指定资源，并加载为 JSON 格式
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将 id2label 中的键转换为整数类型，保留其原始值作为对应的值
    id2label = {int(k): v for k, v in id2label.items()}
    # 将 id2label 设置为配置对象中的 id 到标签的映射
    config.id2label = id2label
    # 将 id2label 反转，创建标签到 id 的映射，并设置为配置对象中的 label2id
    config.label2id = {v: k for k, v in id2label.items()}

    # 加载 HuggingFace 模型
    if vit_name[-5:] == "in21k":
        # 如果 vit_name 的后缀是 "in21k"，则创建 ViTHybridModel 对象并设为评估模式
        model = ViTHybridModel(config).eval()
    else:
        # 否则创建 ViTHybridForImageClassification 对象并设为评估模式
        model = ViTHybridForImageClassification(config).eval()
    # 加载模型的状态字典
    model.load_state_dict(state_dict)

    # 创建图像处理器
    transform = create_transform(**resolve_data_config({}, model=timm_model))
    timm_transforms = transform.transforms

    # 定义 PIL 图像重采样方式的映射关系
    pillow_resamplings = {
        "bilinear": PILImageResampling.BILINEAR,
        "bicubic": PILImageResampling.BICUBIC,
        "nearest": PILImageResampling.NEAREST,
    }

    # 创建 ViTHybridImageProcessor 实例，配置各种图像处理选项
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

    # 准备图像数据
    image = prepare_img()
    # 对图像进行变换并扩展维度，以适应模型输入要求
    timm_pixel_values = transform(image).unsqueeze(0)
    # 使用图像处理器处理图像，并获取处理后的像素值
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 验证像素值是否一致
    assert torch.allclose(timm_pixel_values, pixel_values)

    # 使用无梯度计算上下文，对处理后的像素值进行模型推断
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    # 打印预测类别的 logit 最大值对应的预测类别
    print("Predicted class:", logits.argmax(-1).item())
    
    # 如果指定了 base_model，则使用 timm_model 进行特征前向传播并验证输出的形状和值是否一致
    if base_model:
        timm_pooled_output = timm_model.forward_features(pixel_values)
        assert timm_pooled_output.shape == outputs.pooler_output.shape
        assert torch.allclose(timm_pooled_output, outputs.pooler_output, atol=1e-3)
    else:
        # 否则直接使用 timm_model 进行推断并验证输出的形状和值是否一致
        timm_logits = timm_model(pixel_values)
        assert timm_logits.shape == outputs.logits.shape
        assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)
    # 打印验证通过信息
    print("Looks ok!")

    # 如果指定了 pytorch_dump_folder_path，则保存模型和处理器到指定路径
    if pytorch_dump_folder_path is not None:
        # 确保路径存在，如果不存在则创建
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印保存模型信息
        print(f"Saving model {vit_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印保存处理器信息
        print(f"Saving processor to {pytorch_dump_folder_path}")
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果指定了 push_to_hub，则推送模型和处理器到 HuggingFace Hub
    if push_to_hub:
        # 打印推送模型和处理器到 Hub 的信息
        print(f"Pushing model and processor to the hub {vit_name}")
        # 将模型推送到 Hub 上的指定路径
        model.push_to_hub(f"ybelkada/{vit_name}")
        # 将处理器推送到 Hub 上的指定路径
        processor.push_to_hub(f"ybelkada/{vit_name}")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必填参数
    parser.add_argument(
        "--vit_name",
        default="vit_base_r50_s16_384",
        type=str,
        help="Name of the hybrid ViT timm model you'd like to convert.",
    )
    # 添加参数 `vit_name`，默认为 "vit_base_r50_s16_384"，类型为字符串，用于指定要转换的混合 ViT 模型的名称

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加参数 `pytorch_dump_folder_path`，默认为 None，类型为字符串，用于指定输出的 PyTorch 模型保存路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether to upload the model to the HuggingFace hub."
    )
    # 添加参数 `push_to_hub`，如果提供该参数则将模型上传至 HuggingFace hub

    args = parser.parse_args()
    # 解析命令行参数并存储到 `args` 变量中

    convert_vit_checkpoint(args.vit_name, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用 `convert_vit_checkpoint` 函数，传入解析得到的参数 `vit_name`、`pytorch_dump_folder_path` 和 `push_to_hub`
```