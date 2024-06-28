# `.\models\mobilevitv2\convert_mlcvnets_to_pytorch.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert MobileViTV2 checkpoints from the ml-cvnets library."""


import argparse
import collections
import json
from pathlib import Path

import requests
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    MobileViTImageProcessor,
    MobileViTV2Config,
    MobileViTV2ForImageClassification,
    MobileViTV2ForSemanticSegmentation,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_orig_config_file(orig_cfg_file):
    print("Loading config file...")

    def flatten_yaml_as_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.abc.MutableMapping):
                items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # 打开原始配置文件并将其展平为字典形式
    config = argparse.Namespace()
    with open(orig_cfg_file, "r") as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                setattr(config, k, v)
        except yaml.YAMLError as exc:
            # 如果加载配置文件出错，记录错误信息
            logger.error("Error while loading config file: {}. Error message: {}".format(orig_cfg_file, str(exc)))
    return config


def get_mobilevitv2_config(task_name, orig_cfg_file):
    # 创建一个 MobileViTV2Config 对象
    config = MobileViTV2Config()

    is_segmentation_model = False

    # 根据任务名设置不同的配置选项
    # imagenet1k 相关任务
    if task_name.startswith("imagenet1k_"):
        config.num_labels = 1000
        if int(task_name.strip().split("_")[-1]) == 384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = "imagenet-1k-id2label.json"
    # imagenet21k_to_1k 相关任务
    elif task_name.startswith("imagenet21k_to_1k_"):
        config.num_labels = 21000
        if int(task_name.strip().split("_")[-1]) == 384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = "imagenet-22k-id2label.json"
    # ade20k 相关分割任务
    elif task_name.startswith("ade20k_"):
        config.num_labels = 151
        config.image_size = 512
        filename = "ade20k-id2label.json"
        is_segmentation_model = True
    # 如果任务名称以 "voc_" 开头，则执行以下设置
    elif task_name.startswith("voc_"):
        # 设置配置文件的类别数量为 21
        config.num_labels = 21
        # 设置图像大小为 512x512
        config.image_size = 512
        # 指定文件名为 "pascal-voc-id2label.json"
        filename = "pascal-voc-id2label.json"
        # 标记这是一个分割模型
        is_segmentation_model = True

    # 加载原始配置文件
    orig_config = load_orig_config_file(orig_cfg_file)
    # 断言原始配置文件中的模型名称为 "mobilevit_v2"，否则抛出异常
    assert getattr(orig_config, "model.classification.name", -1) == "mobilevit_v2", "Invalid model"
    # 设置配置文件中的宽度乘数器为 mitv2 的宽度乘数器值，如果不存在则默认为 1.0
    config.width_multiplier = getattr(orig_config, "model.classification.mitv2.width_multiplier", 1.0)
    # 断言配置文件中的注意力归一化层为 "layer_norm_2d"，否则抛出异常
    assert (
        getattr(orig_config, "model.classification.mitv2.attn_norm_layer", -1) == "layer_norm_2d"
    ), "Norm layers other than layer_norm_2d is not supported"
    # 设置隐藏层激活函数为配置文件中的激活函数名称，如果不存在则默认为 "swish"
    config.hidden_act = getattr(orig_config, "model.classification.activation.name", "swish")
    # 设置图像大小为配置文件中采样器的裁剪宽度，但注释掉了，未生效
    # config.image_size == getattr(orig_config,  'sampler.bs.crop_size_width', 256)

    # 如果是分割模型，则进行以下设置
    if is_segmentation_model:
        # 设置输出步长为配置文件中分割模型的输出步长，默认为 16
        config.output_stride = getattr(orig_config, "model.segmentation.output_stride", 16)
        # 如果任务名称包含 "_deeplabv3"，则设置以下参数
        if "_deeplabv3" in task_name:
            # 设置 DeepLabv3 的空洞卷积率列表
            config.atrous_rates = getattr(orig_config, "model.segmentation.deeplabv3.aspp_rates", [12, 24, 36])
            # 设置 DeepLabv3 的 ASPP 输出通道数，默认为 512
            config.aspp_out_channels = getattr(orig_config, "model.segmentation.deeplabv3.aspp_out_channels", 512)
            # 设置 DeepLabv3 的 ASPP dropout 概率，默认为 0.1
            config.aspp_dropout_prob = getattr(orig_config, "model.segmentation.deeplabv3.aspp_dropout", 0.1)

    # 从 Hugging Face Hub 下载指定仓库中的文件，并加载为 JSON 格式
    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将加载的 id2label 映射转换为整数键值对
    id2label = {int(k): v for k, v in id2label.items()}
    # 设置配置文件中的 id 到 label 的映射
    config.id2label = id2label
    # 设置配置文件中的 label 到 id 的映射
    config.label2id = {v: k for k, v in id2label.items()}

    # 返回配置对象
    return config
# 重命名字典中的键值对，将键为 old 的项替换为 new，保留其对应的值
def rename_key(dct, old, new):
    val = dct.pop(old)  # 弹出字典 dct 中键为 old 的项，并将其值赋给 val
    dct[new] = val  # 将键 new 和对应的值 val 添加到字典 dct 中


def create_rename_keys(state_dict, base_model=False):
    if base_model:
        model_prefix = ""  # 如果 base_model 为 True，则模型前缀为空字符串
    else:
        model_prefix = "mobilevitv2."  # 否则，模型前缀为 "mobilevitv2."

    rename_keys = []  # 初始化空的重命名键列表
    return rename_keys  # 返回空的重命名键列表


def remove_unused_keys(state_dict):
    """remove unused keys (e.g.: seg_head.aux_head)"""
    keys_to_ignore = []  # 初始化空的忽略键列表
    for k in state_dict.keys():  # 遍历 state_dict 中的所有键
        if k.startswith("seg_head.aux_head."):  # 如果键 k 以 "seg_head.aux_head." 开头
            keys_to_ignore.append(k)  # 将该键 k 添加到忽略键列表 keys_to_ignore 中
    for k in keys_to_ignore:  # 遍历忽略键列表中的所有键
        state_dict.pop(k, None)  # 从 state_dict 中移除键 k 对应的项


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # url = "https://cdn.britannica.com/86/141086-050-9D7C75EE/Gulfstream-G450-business-jet-passengers.jpg"
    im = Image.open(requests.get(url, stream=True).raw)  # 使用给定的 URL 打开并加载图像
    return im  # 返回加载的图像对象


@torch.no_grad()
def convert_mobilevitv2_checkpoint(task_name, checkpoint_path, orig_config_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our MobileViTV2 structure.
    """
    config = get_mobilevitv2_config(task_name, orig_config_path)  # 获取移动视觉模型的配置信息

    # load original state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # 加载原始的模型检查点

    # load huggingface model
    if task_name.startswith("ade20k_") or task_name.startswith("voc_"):
        model = MobileViTV2ForSemanticSegmentation(config).eval()  # 创建用于语义分割任务的移动视觉模型
        base_model = False  # 设置基础模型标志为 False
    else:
        model = MobileViTV2ForImageClassification(config).eval()  # 创建用于图像分类任务的移动视觉模型
        base_model = False  # 设置基础模型标志为 False

    # remove and rename some keys of load the original model
    state_dict = checkpoint  # 将加载的检查点赋给 state_dict
    remove_unused_keys(state_dict)  # 移除 state_dict 中的一些未使用的键
    rename_keys = create_rename_keys(state_dict, base_model=base_model)  # 创建重命名键列表
    for rename_key_src, rename_key_dest in rename_keys:  # 遍历重命名键列表中的每一对键值对
        rename_key(state_dict, rename_key_src, rename_key_dest)  # 使用 rename_key 函数重命名 state_dict 中的键

    # load modified state_dict
    model.load_state_dict(state_dict)  # 加载修改后的状态字典到模型中

    # Check outputs on an image, prepared by MobileViTImageProcessor
    image_processor = MobileViTImageProcessor(crop_size=config.image_size, size=config.image_size + 32)  # 创建移动视觉图像处理器对象
    encoding = image_processor(images=prepare_img(), return_tensors="pt")  # 对准备好的图像进行编码处理
    outputs = model(**encoding)  # 使用模型进行推断

    # verify classification model
    if task_name.startswith("imagenet"):
        logits = outputs.logits  # 获取模型输出的逻辑回归值
        predicted_class_idx = logits.argmax(-1).item()  # 获取预测类别索引
        print("Predicted class:", model.config.id2label[predicted_class_idx])  # 打印预测类别标签
        if task_name.startswith("imagenet1k_256") and config.width_multiplier == 1.0:
            # expected_logits for base variant
            expected_logits = torch.tensor([-1.6336e00, -7.3204e-02, -5.1883e-01])
            assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)  # 断言预测的逻辑回归与预期值的接近度

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)  # 创建模型保存目录
    print(f"Saving model {task_name} to {pytorch_dump_folder_path}")  # 打印保存模型的信息
    model.save_pretrained(pytorch_dump_folder_path)  # 保存模型到指定路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")  # 打印保存图像处理器的信息
    # 将图像处理器的当前状态保存到指定的 PyTorch 模型导出文件夹路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 必需的参数
    parser.add_argument(
        "--task",
        default="imagenet1k_256",
        type=str,
        help=(
            "Name of the task for which the MobileViTV2 model you'd like to convert is trained on . "
            """
                Classification (ImageNet-1k)
                    - MobileViTV2 (256x256) : imagenet1k_256
                    - MobileViTV2 (Trained on 256x256 and Finetuned on 384x384) : imagenet1k_384
                    - MobileViTV2 (Trained on ImageNet-21k and Finetuned on ImageNet-1k 256x256) :
                      imagenet21k_to_1k_256
                    - MobileViTV2 (Trained on ImageNet-21k, Finetuned on ImageNet-1k 256x256, and Finetuned on
                      ImageNet-1k 384x384) : imagenet21k_to_1k_384
                Segmentation
                    - ADE20K Dataset : ade20k_deeplabv3
                    - Pascal VOC 2012 Dataset: voc_deeplabv3
            """
        ),
        choices=[
            "imagenet1k_256",
            "imagenet1k_384",
            "imagenet21k_to_1k_256",
            "imagenet21k_to_1k_384",
            "ade20k_deeplabv3",
            "voc_deeplabv3",
        ],
    )
    
    # 添加参数：原始检查点文件路径
    parser.add_argument(
        "--orig_checkpoint_path", required=True, type=str, help="Path to the original state dict (.pt file)."
    )
    
    # 添加参数：原始配置文件路径
    parser.add_argument("--orig_config_path", required=True, type=str, help="Path to the original config file.")
    
    # 添加参数：输出 PyTorch 模型目录路径
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数，将 MobileViTV2 模型检查点转换为 PyTorch 模型
    convert_mobilevitv2_checkpoint(
        args.task, args.orig_checkpoint_path, args.orig_config_path, args.pytorch_dump_folder_path
    )
```