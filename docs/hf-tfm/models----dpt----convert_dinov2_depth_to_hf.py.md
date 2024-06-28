# `.\models\dpt\convert_dinov2_depth_to_hf.py`

```py
# coding=utf-8
# 指定编码格式为 UTF-8

# Copyright 2023 The HuggingFace Inc. team.
# 版权声明，声明代码版权归 HuggingFace Inc. 团队所有。

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 进行许可

# you may not use this file except in compliance with the License.
# 除非遵守许可证规定，否则不得使用此文件。

# You may obtain a copy of the License at
# 可以从以下链接获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 在适用法律要求或书面同意的情况下，根据许可证分发的软件是基于"原样"分发的，没有任何明示或暗示的担保或条件。

# See the License for the specific language governing permissions and
# 请查阅许可证了解具体的权限和限制

# limitations under the License.
# 许可证下的限制。

"""Convert DINOv2 + DPT checkpoints from the original repository. URL:
https://github.com/facebookresearch/dinov2/tree/main"""
# 代码的简要描述和参考链接

import argparse
import itertools
import math
from pathlib import Path

import requests
import torch
from PIL import Image
from torchvision import transforms

from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging

# 设置日志输出级别为 info
logging.set_verbosity_info()

# 获取 logger 实例
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    if "small" in model_name:
        # 使用预训练的 Dinov2Config，选择特定的输出索引和参数设置
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-small", out_indices=[3, 6, 9, 12], apply_layernorm=False, reshape_hidden_states=False
        )
        # 设置 neck 层的隐藏层大小
        neck_hidden_sizes = [48, 96, 192, 384]
    elif "base" in model_name:
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-base", out_indices=[3, 6, 9, 12], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [96, 192, 384, 768]
    elif "large" in model_name:
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-large", out_indices=[5, 12, 18, 24], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [128, 256, 512, 1024]
    elif "giant" in model_name:
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-giant", out_indices=[10, 20, 30, 40], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [192, 384, 768, 1536]
    else:
        # 若未指定模型名称，抛出未实现的错误
        raise NotImplementedError("To do")

    # 创建 DPTConfig 实例
    config = DPTConfig(
        backbone_config=backbone_config,
        neck_hidden_sizes=neck_hidden_sizes,
        use_bias_in_fusion_residual=False,
        add_projection=True,
    )

    return config


# here we list all DPT keys to be renamed (original name on the left, our name on the right)
# 列出需要重命名的所有 DPT 键（左边为原始名称，右边为新名称）
def create_rename_keys_dpt(config):
    rename_keys = []

    # fmt: off
    # 格式化关闭，用于避免 IDE 格式化工具干扰代码的排版
    # activation postprocessing (projections, readout projections + resize blocks)
    for i in range(4):
        # 添加重命名键值对，将"decode_head.reassemble_blocks.projects.{i}.conv.weight"映射到"neck.reassemble_stage.layers.{i}.projection.weight"
        rename_keys.append((f"decode_head.reassemble_blocks.projects.{i}.conv.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        # 添加重命名键值对，将"decode_head.reassemble_blocks.projects.{i}.conv.bias"映射到"neck.reassemble_stage.layers.{i}.projection.bias"
        rename_keys.append((f"decode_head.reassemble_blocks.projects.{i}.conv.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        # 添加重命名键值对，将"decode_head.reassemble_blocks.readout_projects.{i}.0.weight"映射到"neck.reassemble_stage.readout_projects.{i}.0.weight"
        rename_keys.append((f"decode_head.reassemble_blocks.readout_projects.{i}.0.weight", f"neck.reassemble_stage.readout_projects.{i}.0.weight"))
        # 添加重命名键值对，将"decode_head.reassemble_blocks.readout_projects.{i}.0.bias"映射到"neck.reassemble_stage.readout_projects.{i}.0.bias"
        rename_keys.append((f"decode_head.reassemble_blocks.readout_projects.{i}.0.bias", f"neck.reassemble_stage.readout_projects.{i}.0.bias"))

        # 如果i不等于2，则添加重命名键值对，将"decode_head.reassemble_blocks.resize_layers.{i}.weight"映射到"neck.reassemble_stage.layers.{i}.resize.weight"
        if i != 2:
            rename_keys.append((f"decode_head.reassemble_blocks.resize_layers.{i}.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            # 添加重命名键值对，将"decode_head.reassemble_blocks.resize_layers.{i}.bias"映射到"neck.reassemble_stage.layers.{i}.resize.bias"
            rename_keys.append((f"decode_head.reassemble_blocks.resize_layers.{i}.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

    # fusion layers
    for i in range(4):
        # 添加重命名键值对，将"decode_head.fusion_blocks.{i}.project.conv.weight"映射到"neck.fusion_stage.layers.{i}.projection.weight"
        rename_keys.append((f"decode_head.fusion_blocks.{i}.project.conv.weight", f"neck.fusion_stage.layers.{i}.projection.weight"))
        # 添加重命名键值对，将"decode_head.fusion_blocks.{i}.project.conv.bias"映射到"neck.fusion_stage.layers.{i}.projection.bias"
        rename_keys.append((f"decode_head.fusion_blocks.{i}.project.conv.bias", f"neck.fusion_stage.layers.{i}.projection.bias"))
        
        # 如果i不等于0，则添加重命名键值对，将"decode_head.fusion_blocks.{i}.res_conv_unit1.conv1.conv.weight"映射到"neck.fusion_stage.layers.{i}.residual_layer1.convolution1.weight"
        if i != 0:
            rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit1.conv1.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer1.convolution1.weight"))
            # 添加重命名键值对，将"decode_head.fusion_blocks.{i}.res_conv_unit1.conv2.conv.weight"映射到"neck.fusion_stage.layers.{i}.residual_layer1.convolution2.weight"
            rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit1.conv2.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer1.convolution2.weight"))
        
        # 添加重命名键值对，将"decode_head.fusion_blocks.{i}.res_conv_unit2.conv1.conv.weight"映射到"neck.fusion_stage.layers.{i}.residual_layer2.convolution1.weight"
        rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit2.conv1.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer2.convolution1.weight"))
        # 添加重命名键值对，将"decode_head.fusion_blocks.{i}.res_conv_unit2.conv2.conv.weight"映射到"neck.fusion_stage.layers.{i}.residual_layer2.convolution2.weight"
        rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit2.conv2.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer2.convolution2.weight"))

    # neck convolutions
    for i in range(4):
        # 添加重命名键值对，将"decode_head.convs.{i}.conv.weight"映射到"neck.convs.{i}.weight"
        rename_keys.append((f"decode_head.convs.{i}.conv.weight", f"neck.convs.{i}.weight"))

    # head
    # 添加重命名键值对，将"decode_head.project.conv.weight"映射到"head.projection.weight"
    rename_keys.append(("decode_head.project.conv.weight", "head.projection.weight"))
    # 添加重命名键值对，将"decode_head.project.conv.bias"映射到"head.projection.bias"
    rename_keys.append(("decode_head.project.conv.bias", "head.projection.bias"))

    for i in range(0, 5, 2):
        # 添加重命名键值对，将"decode_head.conv_depth.head.{i}.weight"映射到"head.head.{i}.weight"
        rename_keys.append((f"decode_head.conv_depth.head.{i}.weight", f"head.head.{i}.weight"))
        # 添加重命名键值对，将"decode_head.conv_depth.head.{i}.bias"映射到"head.head.{i}.bias"
        rename_keys.append((f"decode_head.conv_depth.head.{i}.bias", f"head.head.{i}.bias"))

    # 返回所有重命名的键值对列表
    return rename_keys
# 定义函数：创建用于重命名骨干网络参数的键列表
def create_rename_keys_backbone(config):
    # 初始化一个空的重命名键列表
    rename_keys = []

    # fmt: off
    # 开始忽略格式化，便于在下面进行嵌套的键值对添加
    # patch embedding layer
    # 添加需要重命名的键值对：("原始名称", "我们的名称")
    rename_keys.append(("cls_token", "backbone.embeddings.cls_token"))
    rename_keys.append(("mask_token", "backbone.embeddings.mask_token"))
    rename_keys.append(("pos_embed", "backbone.embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))

    # Transfomer encoder
    # 对于每一个编码器层进行迭代
    for i in range(config.backbone_config.num_hidden_layers):
        # layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.norm2.bias"))

        # MLP
        # 根据配置选择使用不同的MLP结构
        if config.backbone_config.use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"backbone.encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"backbone.encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"backbone.encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"backbone.encoder.layer.{i}.mlp.w3.bias"))
        else:
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.mlp.fc2.bias"))

        # layerscale
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"backbone.encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"backbone.encoder.layer.{i}.layer_scale2.lambda1"))

        # attention projection layer
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))
    # fmt: on

    # 添加最后两个需要重命名的键值对
    rename_keys.append(("norm.weight", "backbone.layernorm.weight"))
    rename_keys.append(("norm.bias", "backbone.layernorm.bias"))

    # 返回最终的重命名键列表
    return rename_keys
    # 遍历指定范围内的隐藏层数量，按顺序处理每一层
    for i in range(config.backbone_config.num_hidden_layers):
        # 弹出当前层的注意力机制的查询、键、值的权重和偏置
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # 获取隐藏层的大小
        hidden_size = config.backbone_config.hidden_size
        
        # 将查询（query）、键（key）、值（value）依次添加到状态字典中
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]
# 从字典中删除旧键，将其对应的值存储在变量val中
def rename_key(dct, old, new):
    val = dct.pop(old)
    # 将旧键对应的值以新键的形式重新插入字典中
    dct[new] = val

# 下载一个可爱猫咪图片并返回其Image对象
def prepare_img():
    # 图片的URL地址
    url = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
    # 使用requests库获取图片的原始数据流，并由PIL库打开为Image对象
    im = Image.open(requests.get(url, stream=True).raw)
    return im

# 包含模型名称到其对应预训练权重文件URL的字典
name_to_url = {
    "dpt-dinov2-small-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_dpt_head.pth",
    "dpt-dinov2-small-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_kitti_dpt_head.pth",
    "dpt-dinov2-base-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_nyu_dpt_head.pth",
    "dpt-dinov2-base-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_kitti_dpt_head.pth",
    "dpt-dinov2-large-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_nyu_dpt_head.pth",
    "dpt-dinov2-large-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_kitti_dpt_head.pth",
    "dpt-dinov2-giant-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_nyu_dpt_head.pth",
    "dpt-dinov2-giant-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_dpt_head.pth",
}

# 获取图像的原始像素值
def get_original_pixel_values(image):
    # 定义一个用于图像预处理的类CenterPadding
    class CenterPadding(object):
        def __init__(self, multiple):
            super().__init__()
            self.multiple = multiple

        # 计算填充值以使图像大小成为multiple的整数倍
        def _get_pad(self, size):
            new_size = math.ceil(size / self.multiple) * self.multiple
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right

        # 对图像进行填充操作
        def __call__(self, img):
            pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in img.shape[-2:][::-1]))
            output = torch.nn.functional.pad(img, pads)
            return output

        # 返回类的描述字符串
        def __repr__(self):
            return self.__class__.__name__ + "()"

    # 定义图像转换的函数make_depth_transform
    def make_depth_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),  # 将图像转换为Tensor
                lambda x: 255.0 * x[:3],  # 丢弃alpha通道并将像素值缩放到0-255范围
                transforms.Normalize(
                    mean=(123.675, 116.28, 103.53),
                    std=(58.395, 57.12, 57.375),
                ),
                CenterPadding(multiple=14),  # 使用CenterPadding类进行图像填充
            ]
        )

    # 创建图像转换操作
    transform = make_depth_transform()
    # 对输入的图像应用转换操作，并在第0维度增加一个维度
    original_pixel_values = transform(image).unsqueeze(0)

    return original_pixel_values

# 用于无梯度计算的装饰器，用于转换DPT模型的检查点
@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # 根据模型名称获取检查点的URL
    checkpoint_url = name_to_url[model_name]
    # 根据模型名称获取DPT的配置信息
    config = get_dpt_config(model_name)

    # 打印检查点的URL地址
    print("URL:", checkpoint_url)
    # 从指定的 URL 加载预训练模型的状态字典，使用 CPU 运行
    dpt_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]

    # 根据配置文件创建重命名键列表
    rename_keys = create_rename_keys_dpt(config)
    # 遍历重命名键列表，将模型状态字典中的键进行重命名
    for src, dest in rename_keys:
        rename_key(dpt_state_dict, src, dest)

    # 根据模型名称加载原始的骨干网络状态字典
    if "small" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    elif "base" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    elif "large" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    elif "giant" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    else:
        raise NotImplementedError("To do")
    # 将模型设置为评估模式
    original_model.eval()
    # 获取原始骨干网络的状态字典
    backbone_state_dict = original_model.state_dict()

    # 根据配置文件创建重命名键列表
    rename_keys = create_rename_keys_backbone(config)
    # 遍历重命名键列表，将骨干网络状态字典中的键进行重命名
    for src, dest in rename_keys:
        rename_key(backbone_state_dict, src, dest)

    # 从骨干网络状态字典中读取 QKV 矩阵
    read_in_q_k_v(backbone_state_dict, config)

    # 复制骨干网络状态字典的条目，处理特定键的名称替换
    for key, val in backbone_state_dict.copy().items():
        val = backbone_state_dict.pop(key)
        if "w12" in key:
            key = key.replace("w12", "weights_in")
        if "w3" in key:
            key = key.replace("w3", "weights_out")
        backbone_state_dict[key] = val

    # 合并骨干网络状态字典和 DPT 模型状态字典
    state_dict = {**backbone_state_dict, **dpt_state_dict}

    # 加载 HuggingFace 模型
    model = DPTForDepthEstimation(config)
    # 加载模型的状态字典，并允许部分匹配
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    # 断言确保缺失的键符合预期
    assert missing_keys == [
        "neck.fusion_stage.layers.0.residual_layer1.convolution1.weight",
        "neck.fusion_stage.layers.0.residual_layer1.convolution2.weight",
    ]
    # 设置模型为评估模式
    model.eval()

    # 验证图像处理器配置
    processor = DPTImageProcessor(
        do_resize=False,
        do_rescale=False,
        do_pad=True,
        size_divisor=14,
        do_normalize=True,
        image_mean=(123.675, 116.28, 103.53),
        image_std=(58.395, 57.12, 57.375),
    )

    # 准备图像数据
    image = prepare_img()
    # 使用图像处理器处理图像并获取像素值张量
    pixel_values = processor(image, return_tensors="pt").pixel_values.float()
    # 获取原始图像的像素值
    original_pixel_values = get_original_pixel_values(image)

    # 断言确保处理后的像素值与原始像素值接近
    assert torch.allclose(pixel_values, original_pixel_values)

    # 验证模型的前向传播
    with torch.no_grad():
        outputs = model(pixel_values)

    # 获取预测的深度图
    predicted_depth = outputs.predicted_depth

    # 打印预测深度的形状信息和部分预测值
    print("Shape of predicted depth:", predicted_depth.shape)
    print("First values of predicted depth:", predicted_depth[0, :3, :3])

    # 断言确保 logits 的条件
    # 如果需要验证 logits，则执行以下操作
    if verify_logits:
        # 如果模型名称是 "dpt-dinov2-small-nyu"
        if model_name == "dpt-dinov2-small-nyu":
            # 设置预期的深度图形状
            expected_shape = torch.Size([1, 576, 736])
            # 设置预期的深度图片段数据
            expected_slice = torch.tensor(
                [[3.3576, 3.4741, 3.4345], [3.4324, 3.5012, 3.2775], [3.2560, 3.3563, 3.2354]]
            )

        # 断言预测的深度图形状是否符合预期
        assert predicted_depth.shape == torch.Size(expected_shape)
        # 断言预测的深度图片段是否与预期片段在指定的误差范围内一致
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-5)
        # 打印确认信息
        print("Looks ok!")

    # 如果指定了 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        # 创建目录（如果不存在）
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印保存模型和处理器的消息
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 打印推送模型和处理器到 Hub 的消息
        print("Pushing model and processor to hub...")
        # 将模型推送到 Hub
        model.push_to_hub(repo_id=f"facebook/{model_name}")
        # 将处理器推送到 Hub
        processor.push_to_hub(repo_id=f"facebook/{model_name}")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建命令行参数解析器对象

    # 必需的参数
    parser.add_argument(
        "--model_name",
        default="dpt-dinov2-small-nyu",
        type=str,
        choices=name_to_url.keys(),
        help="Name of the model you'd like to convert."
    )
    # 添加一个参数：模型名称，类型为字符串，默认为"dpt-dinov2-small-nyu"，可选值为name_to_url字典的键，用于指定要转换的模型名称

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory."
    )
    # 添加一个参数：PyTorch 模型输出目录的路径，类型为字符串，默认为None，用于指定输出的PyTorch模型存储目录的路径

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion."
    )
    # 添加一个参数：是否在转换后将模型推送到模型中心（hub），采用布尔值标志，默认为False，用于指定是否在转换后将模型推送到hub

    parser.add_argument(
        "--verify_logits",
        action="store_true",
        required=False,
        help="Path to the output PyTorch model directory."
    )
    # 添加一个参数：是否验证 logits，采用布尔值标志，默认为False，用于指定是否验证 logits，并指定验证结果的输出路径

    args = parser.parse_args()
    # 解析命令行参数并将其存储在args变量中

    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
    # 调用convert_dpt_checkpoint函数，传入解析后的参数args中的模型名称、PyTorch模型输出目录路径、推送到hub的标志、验证logits的标志作为参数
```