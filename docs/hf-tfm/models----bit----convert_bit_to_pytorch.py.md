# `.\models\bit\convert_bit_to_pytorch.py`

```
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
"""Convert BiT checkpoints from the timm library."""

import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 格式数据的模块
from pathlib import Path  # 导入处理文件路径的模块

import requests  # 导入处理 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载资源的函数
from PIL import Image  # 导入处理图像的模块
from timm import create_model  # 从 timm 库中导入创建模型的函数
from timm.data import resolve_data_config  # 从 timm 库中导入解析数据配置的函数
from timm.data.transforms_factory import create_transform  # 从 timm 库中导入创建数据转换的函数

from transformers import BitConfig, BitForImageClassification, BitImageProcessor  # 导入 BiT 模型相关类
from transformers.image_utils import PILImageResampling  # 导入图像处理相关函数
from transformers.utils import logging  # 导入日志记录相关函数


logging.set_verbosity_info()  # 设置日志记录的详细级别为信息
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_config(model_name):
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    conv_layer = "std_conv" if "bit" in model_name else False

    # note that when using BiT as backbone for ViT-hybrid checkpoints,
    # one needs to additionally set config.layer_type = "bottleneck", config.stem_type = "same",
    # config.conv_layer = "std_conv_same"
    config = BitConfig(
        conv_layer=conv_layer,
        num_labels=1000,
        id2label=id2label,
        label2id=label2id,
    )

    return config


def rename_key(name):
    if "stem.conv" in name:
        name = name.replace("stem.conv", "bit.embedder.convolution")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "head.fc" in name:
        name = name.replace("head.fc", "classifier.1")
    if name.startswith("norm"):
        name = "bit." + name
    if "bit" not in name and "classifier" not in name:
        name = "bit.encoder." + name

    return name


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_bit_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our BiT structure.
    """

    # define default BiT configuration
    config = get_config(model_name)  # 获取指定模型的配置信息

    # load original model from timm
    timm_model = create_model(model_name, pretrained=True)  # 从 timm 加载预训练模型
    timm_model.eval()  # 设置模型为评估模式，即不进行梯度计算和反向传播
    # 获取原始模型的状态字典
    state_dict = timm_model.state_dict()
    # 遍历状态字典的键（这里使用副本），更新键名并挤压张量（如果键名中包含 "head"）
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val.squeeze() if "head" in key else val

    # 加载 HuggingFace 模型
    model = BitForImageClassification(config)
    # 设置模型为评估模式
    model.eval()
    # 加载预训练好的模型参数
    model.load_state_dict(state_dict)

    # 创建图像处理器
    transform = create_transform(**resolve_data_config({}, model=timm_model))
    # 获取 timm_transforms 列表
    timm_transforms = transform.transforms

    # 定义 Pillow 图像重采样方法的映射关系
    pillow_resamplings = {
        "bilinear": PILImageResampling.BILINEAR,
        "bicubic": PILImageResampling.BICUBIC,
        "nearest": PILImageResampling.NEAREST,
    }

    # 创建 BitImageProcessor 实例
    processor = BitImageProcessor(
        do_resize=True,  # 是否执行调整大小操作
        size={"shortest_edge": timm_transforms[0].size},  # 调整大小后的最短边长度
        resample=pillow_resamplings[timm_transforms[0].interpolation.value],  # 图像重采样方法
        do_center_crop=True,  # 是否执行中心裁剪
        crop_size={"height": timm_transforms[1].size[0], "width": timm_transforms[1].size[1]},  # 裁剪尺寸
        do_normalize=True,  # 是否执行归一化
        image_mean=timm_transforms[-1].mean.tolist(),  # 图像均值
        image_std=timm_transforms[-1].std.tolist(),  # 图像标准差
    )

    # 准备图像
    image = prepare_img()
    # 对图像应用变换并扩展维度
    timm_pixel_values = transform(image).unsqueeze(0)
    # 使用图像处理器处理图像并获取像素值
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 验证像素值是否一致
    assert torch.allclose(timm_pixel_values, pixel_values)

    # 验证输出的 logits
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    # 打印前三个 logits 值
    print("Logits:", logits[0, :3])
    # 打印预测类别
    print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])
    # 使用 timm_model 计算 logits
    timm_logits = timm_model(pixel_values)
    # 断言 timm_logits 的形状与 outputs.logits 相同
    assert timm_logits.shape == outputs.logits.shape
    # 断言 timm_logits 与 outputs.logits 的值在容差范围内相等
    assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)
    # 打印确认信息
    print("Looks ok!")

    # 如果指定了 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        # 创建路径
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印保存信息
        print(f"Saving model {model_name} and processor to {pytorch_dump_folder_path}")
        # 保存模型和处理器
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果要推送到 Hub
    if push_to_hub:
        # 打印推送信息
        print(f"Pushing model {model_name} and processor to the hub")
        # 推送模型到 Hub
        model.push_to_hub(f"ybelkada/{model_name}")
        # 推送处理器到 Hub
        processor.push_to_hub(f"ybelkada/{model_name}")
if __name__ == "__main__":
    # 如果脚本直接运行而非被导入，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--model_name",
        default="resnetv2_50x1_bitm",
        type=str,
        help="Name of the BiT timm model you'd like to convert.",
    )
    # 添加模型名称参数，设置默认值为'resnetv2_50x1_bitm'，类型为字符串，用于指定要转换的 BiT timm 模型名称

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加参数，指定输出 PyTorch 模型的目录路径，类型为字符串，默认为None

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub.",
    )
    # 添加参数，用于指定是否将模型推送到 hub，采用布尔标志方式表示

    args = parser.parse_args()
    # 解析命令行参数，并将结果存储在 args 变量中

    convert_bit_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用函数 convert_bit_checkpoint，传递解析后的参数：模型名称、输出目录路径、是否推送到 hub
```