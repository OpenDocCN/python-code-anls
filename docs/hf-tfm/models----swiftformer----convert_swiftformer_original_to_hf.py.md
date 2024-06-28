# `.\models\swiftformer\convert_swiftformer_original_to_hf.py`

```
# coding=utf-8
# 定义脚本的编码格式为 UTF-8

# Copyright 2023 The HuggingFace Inc. team.
# 版权声明，版权归 HuggingFace Inc. 团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 进行许可

# you may not use this file except in compliance with the License.
# 除非遵守许可证的规定，否则不得使用此文件。

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”提供的。

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有明示或暗示的任何保证或条件。

# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证，了解特定语言的权限和限制。

"""Convert SwiftFormer checkpoints from the original implementation."""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    SwiftFormerConfig,
    SwiftFormerForImageClassification,
    ViTImageProcessor,
)
from transformers.utils import logging

logging.set_verbosity_info()
# 设置日志记录的详细程度为信息级别

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

device = torch.device("cpu")
# 设置设备为 CPU

# We will verify our results on an image of cute cats
# 我们将在一张可爱猫咪的图片上验证我们的结果
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 定义图像的 URL
    im = Image.open(requests.get(url, stream=True).raw)
    # 从 URL 获取图像，并打开为 PIL 图像对象
    return im

def get_expected_output(swiftformer_name):
    # 根据 SwiftFormer 模型名称返回预期的输出
    if swiftformer_name == "swiftformer_xs":
        return torch.tensor([-2.1703e00, 2.1107e00, -2.0811e00, 8.8685e-01, 2.4360e-01])

    elif swiftformer_name == "swiftformer_s":
        return torch.tensor([3.9636e-01, 2.3478e-01, -1.6963e00, -1.7381e00, -8.6337e-01])

    elif swiftformer_name == "swiftformer_l1":
        return torch.tensor([-4.2768e-01, -4.7429e-01, -1.0897e00, -1.0248e00, 3.5523e-02])

    elif swiftformer_name == "swiftformer_l3":
        return torch.tensor([-2.5330e-01, 2.4211e-01, -6.0185e-01, -8.2789e-01, -6.0446e-02])

def rename_key(dct, old, new):
    # 将字典 dct 中的键 old 重命名为 new
    val = dct.pop(old)
    dct[new] = val

def create_rename_keys(state_dict):
    # 根据模型的 state_dict 创建重命名映射表
    rename_keys = []
    for k in state_dict.keys():
        k_new = k
        if ".pwconv" in k:
            k_new = k_new.replace(".pwconv", ".point_wise_conv")
        if ".dwconv" in k:
            k_new = k_new.replace(".dwconv", ".depth_wise_conv")
        if ".Proj." in k:
            k_new = k_new.replace(".Proj.", ".proj.")
        if "patch_embed" in k_new:
            k_new = k_new.replace("patch_embed", "swiftformer.patch_embed.patch_embedding")
        if "network" in k_new:
            ls = k_new.split(".")
            if ls[2].isdigit():
                k_new = "swiftformer.encoder.network." + ls[1] + ".blocks." + ls[2] + "." + ".".join(ls[3:])
            else:
                k_new = k_new.replace("network", "swiftformer.encoder.network")
        rename_keys.append((k, k_new))
    return rename_keys

@torch.no_grad()
# 使用 torch.no_grad() 修饰，表明下方的函数不需要进行梯度计算

def convert_swiftformer_checkpoint(swiftformer_name, pytorch_dump_folder_path, original_ckpt):
    """
    根据指定的 SwiftFormer 模型名称，转换原始的检查点文件到 PyTorch 格式。

    Args:
        swiftformer_name (str): SwiftFormer 模型名称
        pytorch_dump_folder_path (str): 转换后的 PyTorch 检查点保存路径
        original_ckpt (str): 原始的 SwiftFormer 检查点文件路径
    """
    Copy/paste/tweak model's weights to our SwiftFormer structure.
    """

    # 定义默认的 SwiftFormer 配置对象
    config = SwiftFormerConfig()

    # 设置模型的类别数为 1000
    config.num_labels = 1000
    # 定义 Hugging Face Hub 中的资源库 ID 和文件名
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    # 从 Hugging Face Hub 下载并加载类别映射文件，转换为整数映射到标签名的字典
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    # 根据 id2label 字典生成 label 到 id 的反向映射字典
    config.label2id = {v: k for k, v in id2label.items()}

    # 根据不同的 SwiftFormer 模型名配置模型的深度和嵌入维度
    if swiftformer_name == "swiftformer_xs":
        config.depths = [3, 3, 6, 4]
        config.embed_dims = [48, 56, 112, 220]

    elif swiftformer_name == "swiftformer_s":
        config.depths = [3, 3, 9, 6]
        config.embed_dims = [48, 64, 168, 224]

    elif swiftformer_name == "swiftformer_l1":
        config.depths = [4, 3, 10, 5]
        config.embed_dims = [48, 96, 192, 384]

    elif swiftformer_name == "swiftformer_l3":
        config.depths = [4, 4, 12, 6]
        config.embed_dims = [64, 128, 320, 512]

    # 如果提供了原始模型的检查点路径，则加载其状态字典并进行重命名处理
    if original_ckpt:
        if original_ckpt.startswith("https"):
            # 从 URL 加载模型状态字典
            checkpoint = torch.hub.load_state_dict_from_url(original_ckpt, map_location="cpu", check_hash=True)
        else:
            # 从本地文件加载模型状态字典
            checkpoint = torch.load(original_ckpt, map_location="cpu")
    state_dict = checkpoint

    # 根据预定义规则，创建重命名映射关系并对状态字典进行重命名
    rename_keys = create_rename_keys(state_dict)
    for rename_key_src, rename_key_dest in rename_keys:
        rename_key(state_dict, rename_key_src, rename_key_dest)

    # 加载 SwiftFormer 模型并载入处理后的状态字典
    hf_model = SwiftFormerForImageClassification(config).eval()
    hf_model.load_state_dict(state_dict)

    # 准备测试输入图像和预处理器
    image = prepare_img()
    processor = ViTImageProcessor.from_pretrained("preprocessor_config")
    inputs = processor(images=image, return_tensors="pt")

    # 获取预期输出结果，用于与 HuggingFace 模型输出进行比较
    timm_logits = get_expected_output(swiftformer_name)
    hf_logits = hf_model(inputs["pixel_values"]).logits

    # 断言检查 HuggingFace 模型输出的形状和预期的一致性
    assert hf_logits.shape == torch.Size([1, 1000])
    assert torch.allclose(hf_logits[0, 0:5], timm_logits, atol=1e-3)

    # 确保 PyTorch 导出文件夹存在，保存 SwiftFormer 模型
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {swiftformer_name} to {pytorch_dump_folder_path}")
    hf_model.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--swiftformer_name",
        default="swiftformer_xs",
        choices=["swiftformer_xs", "swiftformer_s", "swiftformer_l1", "swiftformer_l3"],
        type=str,
        help="Name of the SwiftFormer model you'd like to convert.",
    )
    # 添加一个必需的参数：SwiftFormer 模型的名称，可以选择默认为 "swiftformer_xs"
    # 允许的取值为预定义的几种模型名称
    # 参数类型为字符串，帮助信息描述了希望转换的 SwiftFormer 模型的名称

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="./converted_outputs/",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加一个参数：输出 PyTorch 模型的目录路径，默认为当前目录下的 "converted_outputs/"
    # 参数类型为字符串，描述了输出 PyTorch 模型的保存路径

    parser.add_argument("--original_ckpt", default=None, type=str, help="Path to the original model checkpoint.")
    # 添加一个参数：原始模型检查点的路径，默认为 None
    # 参数类型为字符串，描述了原始模型检查点文件的路径

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    convert_swiftformer_checkpoint(args.swiftformer_name, args.pytorch_dump_folder_path, args.original_ckpt)
    # 调用函数 convert_swiftformer_checkpoint，传递解析后的参数：
    #   - SwiftFormer 模型名称
    #   - 输出的 PyTorch 模型目录路径
    #   - 原始模型检查点路径
```