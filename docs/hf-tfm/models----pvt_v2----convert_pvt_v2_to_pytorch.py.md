# `.\models\pvt_v2\convert_pvt_v2_to_pytorch.py`

```
# coding=utf-8
# Copyright 2023 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# All rights reserved.
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
"""
Convert PvtV2 checkpoints from the original library.
"""

import argparse
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import PvtImageProcessor, PvtV2Config, PvtV2ForImageClassification
from transformers.utils import logging

# 设置日志输出级别为信息
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


# 定义需要重命名的键列表（左侧为原始名称，右侧为目标名称）
def create_rename_keys(config):
    rename_keys = []
    # 添加需要重命名的键
    rename_keys.extend(
        [
            ("head.weight", "classifier.weight"),
            ("head.bias", "classifier.bias"),
        ]
    )

    return rename_keys


# 将每个编码器层的权重矩阵拆分为查询（queries）、键（keys）和值（values）
def read_in_k_v(state_dict, config):
    # 遍历每个编码器块
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            # 读取键值（keys）和值（values）的权重和偏置
            kv_weight = state_dict.pop(f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.kv.weight")
            kv_bias = state_dict.pop(f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.kv.bias")
            # 将权重和偏置添加到状态字典中作为键和值的权重和偏置
            state_dict[f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.key.weight"] = kv_weight[
                : config.hidden_sizes[i], :
            ]
            state_dict[f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.key.bias"] = kv_bias[: config.hidden_sizes[i]]

            state_dict[f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.value.bias"] = kv_bias[
                config.hidden_sizes[i] :
            ]


# 重命名键名的辅助函数
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 在一张可爱猫咪的图片上验证结果
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 从 URL 获取图像并打开
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 使用 torch.no_grad 装饰器，确保在执行期间禁用梯度计算
@torch.no_grad()
# 根据给定的 PVT-V2 模型大小选择相应的配置路径
def convert_pvt_v2_checkpoint(pvt_v2_size, pvt_v2_checkpoint, pytorch_dump_folder_path, verify_imagenet_weights=False):
    """
    Copy/paste/tweak model's weights to our PVT structure.
    """

    # 定义默认的 PVT-V2 配置路径
    if pvt_v2_size == "b0":
        config_path = "OpenGVLab/pvt_v2_b0"
    elif pvt_v2_size == "b1":
        config_path = "OpenGVLab/pvt_v2_b1"
    elif pvt_v2_size == "b2":
        config_path = "OpenGVLab/pvt_v2_b2"
    elif pvt_v2_size == "b2-linear":
        config_path = "OpenGVLab/pvt_v2_b2_linear"
    elif pvt_v2_size == "b3":
        config_path = "OpenGVLab/pvt_v2_b3"
    elif pvt_v2_size == "b4":
        config_path = "OpenGVLab/pvt_v2_b4"
    elif pvt_v2_size == "b5":
        config_path = "OpenGVLab/pvt_v2_b5"
    else:
        # 如果给定的模型大小不在预定义的列表中，引发值错误异常
        raise ValueError(
            f"Available model sizes: 'b0', 'b1', 'b2', 'b2-linear', 'b3', 'b4', 'b5', but "
            f"'{pvt_v2_size}' was given"
        )
    
    # 使用预训练配置路径创建 PVT-V2 配置对象
    config = PvtV2Config.from_pretrained(config_path)
    
    # 从指定路径加载原始 PVT-V2 模型权重（通常从 https://github.com/whai362/PVT 下载）
    state_dict = torch.load(pvt_v2_checkpoint, map_location="cpu")

    # 创建重命名键列表以匹配当前模型结构
    rename_keys = create_rename_keys(config)
    
    # 根据重命名键重命名加载的状态字典中的键名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    
    # 将状态字典中的键值对读入 PVT-V2 配置对象
    read_in_k_v(state_dict, config)

    # 加载 HuggingFace 模型，用于图像分类
    model = PvtV2ForImageClassification(config).eval()
    model.load_state_dict(state_dict)
    
    # 创建与配置图像大小匹配的图像处理器对象
    image_processor = PvtImageProcessor(size=config.image_size)
    # 如果需要验证 ImageNet 权重
    if verify_imagenet_weights:
        # 打印信息，验证预训练 ImageNet 权重的转换
        print("Verifying conversion of pretrained ImageNet weights...")
        
        # 使用 PvtImageProcessor 准备图像并编码
        encoding = image_processor(images=prepare_img(), return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        
        # 使用模型推断图像
        outputs = model(pixel_values)
        logits = outputs.logits.detach().cpu()

        # 根据 PvtV2 模型大小选择预期的 logits 切片
        if pvt_v2_size == "b0":
            expected_slice_logits = torch.tensor([-1.1939, -1.4547, -0.1076])
        elif pvt_v2_size == "b1":
            expected_slice_logits = torch.tensor([-0.4716, -0.7335, -0.4600])
        elif pvt_v2_size == "b2":
            expected_slice_logits = torch.tensor([0.0795, -0.3170, 0.2247])
        elif pvt_v2_size == "b2-linear":
            expected_slice_logits = torch.tensor([0.0968, 0.3937, -0.4252])
        elif pvt_v2_size == "b3":
            expected_slice_logits = torch.tensor([-0.4595, -0.2870, 0.0940])
        elif pvt_v2_size == "b4":
            expected_slice_logits = torch.tensor([-0.1769, -0.1747, -0.0143])
        elif pvt_v2_size == "b5":
            expected_slice_logits = torch.tensor([-0.2943, -0.1008, 0.6812])
        else:
            # 如果提供的 PvtV2 模型大小无效，抛出 ValueError 异常
            raise ValueError(
                f"Available model sizes: 'b0', 'b1', 'b2', 'b2-linear', 'b3', 'b4', 'b5', but "
                f"'{pvt_v2_size}' was given"
            )

        # 断言实际的 logits 与预期的 logits 切片在指定的误差范围内相近，否则打印错误信息
        assert torch.allclose(
            logits[0, :3], expected_slice_logits, atol=1e-4
        ), "ImageNet weights not converted successfully."

        # 打印信息，ImageNet 权重验证成功
        print("ImageNet weights verified, conversion successful.")

    # 确保存储 PyTorch 模型的文件夹存在，如果不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印信息，保存模型的 pytorch_model.bin 文件到指定路径
    print(f"Saving model pytorch_model.bin to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印信息，保存图像处理器到指定路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象
    
    # 必选参数
    parser.add_argument(
        "--pvt_v2_size",
        default="b0",
        type=str,
        help="Size of the PVTv2 pretrained model you'd like to convert.",
    )
    # 指定 PVTv2 预训练模型的大小，作为字符串类型的参数，默认为 'b0'
    
    parser.add_argument(
        "--pvt_v2_checkpoint",
        default="pvt_v2_b0.pth",
        type=str,
        help="Checkpoint of the PVTv2 pretrained model you'd like to convert.",
    )
    # 指定 PVTv2 预训练模型的检查点文件路径，作为字符串类型的参数，默认为 'pvt_v2_b0.pth'
    
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 指定输出 PyTorch 模型的目录路径，作为字符串类型的参数，默认为 None
    
    parser.add_argument(
        "--verify-imagenet-weights",
        action="store_true",
        default=False,
        help="Verifies the correct conversion of author-published pretrained ImageNet weights.",
    )
    # 如果存在该选项，则设置为 True，用于验证作者发布的预训练 ImageNet 权重的正确转换
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数 convert_pvt_v2_checkpoint，传入解析后的参数
    convert_pvt_v2_checkpoint(
        pvt_v2_size=args.pvt_v2_size,
        pvt_v2_checkpoint=args.pvt_v2_checkpoint,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        verify_imagenet_weights=args.verify_imagenet_weights,
    )
```