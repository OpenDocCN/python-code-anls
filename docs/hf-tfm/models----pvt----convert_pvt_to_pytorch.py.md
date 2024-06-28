# `.\models\pvt\convert_pvt_to_pytorch.py`

```py
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
"""Convert Pvt checkpoints from the original library."""

import argparse             # 导入解析命令行参数的模块
from pathlib import Path    # 导入处理路径的模块

import requests             # 导入处理HTTP请求的模块
import torch                # 导入PyTorch深度学习框架
from PIL import Image       # 导入Python Imaging Library (PIL) 图像处理库

from transformers import PvtConfig, PvtForImageClassification, PvtImageProcessor   # 导入转换模型用到的类
from transformers.utils import logging   # 导入日志记录工具

logging.set_verbosity_info()    # 设置日志记录的详细程度为信息级别
logger = logging.get_logger(__name__)   # 获取当前模块的日志记录器

# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []    # 初始化一个空列表用于存储重命名的键值对
    # Rename cls token
    rename_keys.extend(     # 扩展列表以添加元组的方式来指定需要重命名的键值对
        [
            ("cls_token", "pvt.encoder.patch_embeddings.3.cls_token"),
        ]
    )
    # Rename norm layer and classifier layer
    rename_keys.extend(     # 继续扩展列表以添加更多需要重命名的键值对
        [
            ("norm.weight", "pvt.encoder.layer_norm.weight"),
            ("norm.bias", "pvt.encoder.layer_norm.bias"),
            ("head.weight", "classifier.weight"),
            ("head.bias", "classifier.bias"),
        ]
    )

    return rename_keys    # 返回所有的重命名键值对列表

# we split up the matrix of each encoder layer into queries, keys and values
def read_in_k_v(state_dict, config):
    # for each of the encoder blocks:
    for i in range(config.num_encoder_blocks):   # 遍历编码器块的数量
        for j in range(config.depths[i]):       # 遍历每个编码器块中的层数
            # read in weights + bias of keys and values (which is a single matrix in the original implementation)
            kv_weight = state_dict.pop(f"pvt.encoder.block.{i}.{j}.attention.self.kv.weight")    # 弹出键值对中的权重
            kv_bias = state_dict.pop(f"pvt.encoder.block.{i}.{j}.attention.self.kv.bias")       # 弹出键值对中的偏置
            # next, add keys and values (in that order) to the state dict
            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.key.weight"] = kv_weight[: config.hidden_sizes[i], :]
            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.key.bias"] = kv_bias[: config.hidden_sizes[i]]   # 将键和偏置添加到状态字典中

            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.value.bias"] = kv_bias[config.hidden_sizes[i] :]   # 将值和偏置添加到状态字典中


def rename_key(dct, old, new):
    val = dct.pop(old)    # 弹出旧键对应的值
    dct[new] = val        # 添加新键并将值赋予该新键

# We will verify our results on an image of cute cats
def prepare_img():
    # 定义图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 库发送 GET 请求获取图片的二进制数据流，并通过 stream=True 确保以流式方式获取数据
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回打开的图片对象
    return im
@torch.no_grad()
def convert_pvt_checkpoint(pvt_size, pvt_checkpoint, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our PVT structure.
    """

    # 定义默认的 PVT 配置路径
    if pvt_size == "tiny":
        config_path = "Zetatech/pvt-tiny-224"
    elif pvt_size == "small":
        config_path = "Zetatech/pvt-small-224"
    elif pvt_size == "medium":
        config_path = "Zetatech/pvt-medium-224"
    elif pvt_size == "large":
        config_path = "Zetatech/pvt-large-224"
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but " f"'{pvt_size}' was given")

    # 使用指定的配置路径创建 PVTConfig 对象
    config = PvtConfig(name_or_path=config_path)
    
    # 从指定路径加载原始模型权重
    state_dict = torch.load(pvt_checkpoint, map_location="cpu")

    # 根据 PVT 配置创建重命名键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 根据 PVT 配置读取键值对
    read_in_k_v(state_dict, config)

    # 加载 HuggingFace 的 PVT 图像分类模型，并设置为评估模式
    model = PvtForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # 使用 PVTFeatureExtractor 准备图像，并检查输出
    image_processor = PvtImageProcessor(size=config.image_size)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)
    logits = outputs.logits.detach().cpu()

    # 根据 PVT 模型大小选择预期的输出片段 logits
    if pvt_size == "tiny":
        expected_slice_logits = torch.tensor([-1.4192, -1.9158, -0.9702])
    elif pvt_size == "small":
        expected_slice_logits = torch.tensor([0.4353, -0.1960, -0.2373])
    elif pvt_size == "medium":
        expected_slice_logits = torch.tensor([-0.2914, -0.2231, 0.0321])
    elif pvt_size == "large":
        expected_slice_logits = torch.tensor([0.3740, -0.7739, -0.4214])
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but " f"'{pvt_size}' was given")

    # 断言模型输出的前三个 logits 与预期的值非常接近
    assert torch.allclose(logits[0, :3], expected_slice_logits, atol=1e-4)

    # 创建输出路径文件夹（如果不存在）
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model pytorch_model.bin to {pytorch_dump_folder_path}")
    # 将模型保存为 PyTorch 预训练模型
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到 PyTorch 模型目录中
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必选参数
    parser.add_argument(
        "--pvt_size",
        default="tiny",
        type=str,
        help="Size of the PVT pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pvt_checkpoint",
        default="pvt_tiny.pth",
        type=str,
        help="Checkpoint of the PVT pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    # 调用函数以转换私有检查点文件格式到PyTorch格式
    convert_pvt_checkpoint(args.pvt_size, args.pvt_checkpoint, args.pytorch_dump_folder_path)
```