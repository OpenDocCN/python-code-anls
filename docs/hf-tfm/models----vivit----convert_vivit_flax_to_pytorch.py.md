# `.\models\vivit\convert_vivit_flax_to_pytorch.py`

```
# coding=utf-8
# 声明编码格式为 UTF-8

# Copyright 2023 The HuggingFace Inc. team.
# 版权声明，版权归 The HuggingFace Inc. 团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 授权使用此代码

# you may not use this file except in compliance with the License.
# 除非遵循许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则按"原样"分发本软件，无论是明示的还是隐含的保证，包括但不限于，适销性、特定用途的适用性和非侵权性。

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证以获取权限和限制的详细信息

"""Convert Flax ViViT checkpoints from the original repository to PyTorch. URL:
https://github.com/google-research/scenic/tree/main/scenic/projects/vivit
"""
# 转换来自原始存储库的 Flax ViViT 检查点到 PyTorch 格式的工具
# 原始存储库 URL: https://github.com/google-research/scenic/tree/main/scenic/projects/vivit

import argparse
# 导入用于命令行解析的模块

import json
# 导入处理 JSON 的模块

import os.path
# 导入处理文件路径的模块

from collections import OrderedDict
# 导入有序字典的模块

import numpy as np
# 导入处理数值计算的模块

import requests
# 导入发送 HTTP 请求的模块

import torch
# 导入 PyTorch 深度学习框架

from flax.training.checkpoints import restore_checkpoint
# 从 Flax 框架中导入恢复检查点的功能

from huggingface_hub import hf_hub_download
# 从 Hugging Face Hub 导入下载函数

from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor
# 从 Transformers 库导入 ViViT 相关组件

from transformers.image_utils import PILImageResampling
# 从 Transformers 库导入图像处理的模块


def download_checkpoint(path):
    # 定义下载检查点文件的函数，参数为保存路径 `path`

    url = "https://storage.googleapis.com/scenic-bucket/vivit/kinetics_400/vivit_base_16x2_unfactorized/checkpoint"
    # 指定检查点文件的下载 URL

    with open(path, "wb") as f:
        # 以二进制写入模式打开文件 `path`

        with requests.get(url, stream=True) as req:
            # 发起带有流式传输的 GET 请求

            for chunk in req.iter_content(chunk_size=2048):
                # 遍历请求的数据块，每次处理大小为 2048 字节

                f.write(chunk)
                # 将数据块写入文件


def get_vivit_config() -> VivitConfig:
    # 定义获取 ViViT 配置的函数，返回类型为 VivitConfig

    config = VivitConfig()
    # 创建 ViViT 配置对象

    config.num_labels = 400
    # 设置标签数量为 400

    repo_id = "huggingface/label-files"
    # 定义标签文件的存储库 ID
    filename = "kinetics400-id2label.json"
    # 定义标签文件的名称

    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 从 Hugging Face Hub 下载标签文件，加载为字典格式
    id2label = {int(k): v for k, v in id2label.items()}
    # 将标签的键转换为整数类型
    config.id2label = id2label
    # 将 id2label 字典赋值给配置对象的 id2label 属性

    config.label2id = {v: k for k, v in id2label.items()}
    # 构建标签到 id 的反向映射字典

    return config
    # 返回配置对象


# We will verify our results on a video of eating spaghetti
# Frame indices used: [ 47, 51, 55, 59, 63, 67, 71, 75, 80, 84, 88, 92, 96, 100, 104, 108, 113, 117,
# 121, 125, 129, 133, 137, 141, 146, 150, 154, 158, 162, 166, 170, 174]
def prepare_video():
    # 定义准备视频数据的函数

    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti_32_frames.npy", repo_type="dataset"
    )
    # 从 Hugging Face Hub 下载具有吃意大利面视频帧的 NumPy 文件

    video = np.load(file)
    # 加载视频文件为 NumPy 数组

    return list(video)
    # 将视频数据转换为列表并返回


def transform_attention(current: np.ndarray):
    # 定义处理注意力张量的函数，参数为当前张量 `current`

    if np.ndim(current) == 2:
        # 如果张量是二维的

        return transform_attention_bias(current)
        # 调用处理注意力偏置的函数并返回结果

    elif np.ndim(current) == 3:
        # 如果张量是三维的

        return transform_attention_kernel(current)
        # 调用处理注意力核心的函数并返回结果

    else:
        # 如果张量维度不符合预期

        raise Exception(f"Invalid number of dimesions: {np.ndim(current)}")
        # 抛出异常，指示张量维度无效


def transform_attention_bias(current: np.ndarray):
    # 定义处理注意力偏置的函数，参数为当前偏置 `current`

    return current.flatten()
    # 将偏置展平并返回结果


def transform_attention_kernel(current: np.ndarray):
    # 定义处理注意力核心的函数，参数为当前核心 `current`

    return np.reshape(current, (current.shape[0], current.shape[1] * current.shape[2])).T
    # 调整核心张量的形状并返回转置结果


def transform_attention_output_weight(current: np.ndarray):
    # 定义处理注意力输出权重的函数，参数为当前权重 `current`
    # 将当前数组 `current` 重新整形为二维数组，行数为原数组行数乘以列数，列数不变
    return np.reshape(current, (current.shape[0] * current.shape[1], current.shape[2])).T
# 根据给定索引 i，从状态字典中获取 Transformer 模型的第 i 个编码器块的状态
def transform_state_encoder_block(state_dict, i):
    state = state_dict["optimizer"]["target"]["Transformer"][f"encoderblock_{i}"]

    # 构建当前编码器块在模型状态字典中的前缀
    prefix = f"encoder.layer.{i}."

    # 创建新的状态字典，将原始状态字典中的数据按指定格式进行转换和重组
    new_state = {
        prefix + "intermediate.dense.bias": state["MlpBlock_0"]["Dense_0"]["bias"],
        prefix + "intermediate.dense.weight": np.transpose(state["MlpBlock_0"]["Dense_0"]["kernel"]),
        prefix + "output.dense.bias": state["MlpBlock_0"]["Dense_1"]["bias"],
        prefix + "output.dense.weight": np.transpose(state["MlpBlock_0"]["Dense_1"]["kernel"]),
        prefix + "layernorm_before.bias": state["LayerNorm_0"]["bias"],
        prefix + "layernorm_before.weight": state["LayerNorm_0"]["scale"],
        prefix + "layernorm_after.bias": state["LayerNorm_1"]["bias"],
        prefix + "layernorm_after.weight": state["LayerNorm_1"]["scale"],
        prefix + "attention.attention.query.bias": transform_attention(
            state["MultiHeadDotProductAttention_0"]["query"]["bias"]
        ),
        prefix + "attention.attention.query.weight": transform_attention(
            state["MultiHeadDotProductAttention_0"]["query"]["kernel"]
        ),
        prefix + "attention.attention.key.bias": transform_attention(
            state["MultiHeadDotProductAttention_0"]["key"]["bias"]
        ),
        prefix + "attention.attention.key.weight": transform_attention(
            state["MultiHeadDotProductAttention_0"]["key"]["kernel"]
        ),
        prefix + "attention.attention.value.bias": transform_attention(
            state["MultiHeadDotProductAttention_0"]["value"]["bias"]
        ),
        prefix + "attention.attention.value.weight": transform_attention(
            state["MultiHeadDotProductAttention_0"]["value"]["kernel"]
        ),
        prefix + "attention.output.dense.bias": state["MultiHeadDotProductAttention_0"]["out"]["bias"],
        prefix + "attention.output.dense.weight": transform_attention_output_weight(
            state["MultiHeadDotProductAttention_0"]["out"]["kernel"]
        ),
    }

    return new_state


# 获取给定状态字典中的 Transformer 模型的编码器块总数
def get_n_layers(state_dict):
    # 使用列表推导计算包含字符串 "encoderblock_" 的键的数量
    return sum([1 if "encoderblock_" in k else 0 for k in state_dict["optimizer"]["target"]["Transformer"].keys()])


# 转换整个状态字典，根据需要添加分类头部分
def transform_state(state_dict, classification_head=False):
    # 获取 Transformer 模型中的编码器块总数
    transformer_layers = get_n_layers(state_dict)

    # 创建一个有序字典用于存储新的状态数据
    new_state = OrderedDict()

    # 转换编码器归一化层的偏置和权重
    new_state["layernorm.bias"] = state_dict["optimizer"]["target"]["Transformer"]["encoder_norm"]["bias"]
    new_state["layernorm.weight"] = state_dict["optimizer"]["target"]["Transformer"]["encoder_norm"]["scale"]

    # 转换嵌入层的投影权重和偏置
    new_state["embeddings.patch_embeddings.projection.weight"] = np.transpose(
        state_dict["optimizer"]["target"]["embedding"]["kernel"], (4, 3, 0, 1, 2)
    )
    new_state["embeddings.patch_embeddings.projection.bias"] = state_dict["optimizer"]["target"]["embedding"]["bias"]

    # 转换分类标记的嵌入向量
    new_state["embeddings.cls_token"] = state_dict["optimizer"]["target"]["cls"]

    # 返回转换后的新状态字典
    return new_state
    # 将输入状态字典中的位置嵌入张量更新到新状态字典中的指定键
    new_state["embeddings.position_embeddings"] = state_dict["optimizer"]["target"]["Transformer"]["posembed_input"][
        "pos_embedding"
    ]
    
    # 遍历每个Transformer层，更新新状态字典
    for i in range(transformer_layers):
        new_state.update(transform_state_encoder_block(state_dict, i))
    
    # 如果存在分类头部，调整新状态字典的键名并更新分类器权重和偏置
    if classification_head:
        # 更新新状态字典中的键名前缀为"vivit."
        new_state = {"vivit." + k: v for k, v in new_state.items()}
        # 转置并更新分类器权重
        new_state["classifier.weight"] = np.transpose(state_dict["optimizer"]["target"]["output_projection"]["kernel"])
        # 转置并更新分类器偏置
        new_state["classifier.bias"] = np.transpose(state_dict["optimizer"]["target"]["output_projection"]["bias"])
    
    # 将新状态字典中的值转换为PyTorch张量并返回
    return {k: torch.tensor(v) for k, v in new_state.items()}
# 检查图像处理器设置与原始实现是否一致
# 原始实现可以在此链接中找到：https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/data/video_tfrecord_dataset.py
# 数据集特定配置：
# https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/configs/kinetics400/vivit_base_k400.py
def get_processor() -> VivitImageProcessor:
    # 创建 VivitImageProcessor 实例
    extractor = VivitImageProcessor()

    # 断言确保是否执行了图像大小调整
    assert extractor.do_resize is True
    # 断言确保调整后的最短边为256像素
    assert extractor.size == {"shortest_edge": 256}
    # 断言确保是否执行了中心裁剪
    assert extractor.do_center_crop is True
    # 断言确保裁剪尺寸为224x224像素
    assert extractor.crop_size == {"width": 224, "height": 224}
    # 断言确保使用双线性重采样
    assert extractor.resample == PILImageResampling.BILINEAR

    # 在这里参考：https://github.com/deepmind/dmvr/blob/master/dmvr/modalities.py
    # 可以看到 add_image 函数中 normalization_mean 和 normalization_std 的默认值分别设为 0 和 1
    # 这意味着没有进行归一化操作（而 ViViT 在调用此函数时也没有覆盖这些值）
    assert extractor.do_normalize is False
    # 断言确保是否执行了重新缩放
    assert extractor.do_rescale is True
    # 断言确保重新缩放因子为 1/255
    assert extractor.rescale_factor == 1 / 255

    # 断言确保是否执行了零中心化
    assert extractor.do_zero_centering is True

    # 返回图像处理器实例
    return extractor


def convert(output_path: str):
    # Flax 模型的路径
    flax_model_path = "checkpoint"

    # 如果 Flax 模型路径不存在，则下载检查点
    if not os.path.exists(flax_model_path):
        download_checkpoint(flax_model_path)

    # 恢复检查点的状态字典
    state_dict = restore_checkpoint(flax_model_path, None)
    # 对状态字典进行转换，包括分类头部的变换
    new_state = transform_state(state_dict, classification_head=True)

    # 获取 ViViT 的配置
    config = get_vivit_config()

    # 断言确保图像大小为 224
    assert config.image_size == 224
    # 断言确保帧数为 32
    assert config.num_frames == 32

    # 创建 ViViT 的视频分类模型
    model = VivitForVideoClassification(config)
    # 加载模型的状态字典
    model.load_state_dict(new_state)
    # 设为评估模式
    model.eval()

    # 获取图像处理器实例
    extractor = get_processor()

    # 准备视频数据
    video = prepare_video()
    # 使用图像处理器处理视频数据，返回 PyTorch 张量
    inputs = extractor(video, return_tensors="pt")

    # 对模型进行推理
    outputs = model(**inputs)

    # 期望的输出形状
    expected_shape = torch.Size([1, 400])
    # 期望的输出切片
    expected_slice = torch.tensor([-1.0543, 2.0764, -0.2104, 0.4439, -0.9658])

    # 断言确保模型输出的 logits 的形状正确
    assert outputs.logits.shape == expected_shape
    # 断言确保前5个 logits 的值与期望值在指定的误差范围内
    assert torch.allclose(outputs.logits[0, :5], expected_slice, atol=1e-4), outputs.logits[0, :5]

    # 将模型保存为预训练模型
    model.save_pretrained(output_path)
    # 保存图像处理器的预训练状态
    extractor.save_pretrained(output_path)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_name", "-o", type=str, help="输出转换后的 HuggingFace 模型的路径")

    args = parser.parse_args()
    # 调用 convert 函数，传入输出路径参数
    convert(args.output_model_name)
```