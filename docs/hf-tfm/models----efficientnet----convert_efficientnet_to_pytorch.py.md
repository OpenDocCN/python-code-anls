# `.\models\efficientnet\convert_efficientnet_to_pytorch.py`

```py
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
"""Convert EfficientNet checkpoints from the original repository.

URL: https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/efficientnet.py"""

import argparse
import json
import os

import numpy as np
import PIL
import requests
import tensorflow.keras.applications.efficientnet as efficientnet
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.preprocessing import image

from transformers import (
    EfficientNetConfig,
    EfficientNetForImageClassification,
    EfficientNetImageProcessor,
)
from transformers.utils import logging

# 设置日志输出为 info 级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 EfficientNet 模型的类别映射，每个字符串键对应一个 EfficientNet 模型类
model_classes = {
    "b0": efficientnet.EfficientNetB0,
    "b1": efficientnet.EfficientNetB1,
    "b2": efficientnet.EfficientNetB2,
    "b3": efficientnet.EfficientNetB3,
    "b4": efficientnet.EfficientNetB4,
    "b5": efficientnet.EfficientNetB5,
    "b6": efficientnet.EfficientNetB6,
    "b7": efficientnet.EfficientNetB7,
}

# 定义每个 EfficientNet 模型的配置参数字典
CONFIG_MAP = {
    "b0": {
        "hidden_dim": 1280,
        "width_coef": 1.0,
        "depth_coef": 1.0,
        "image_size": 224,
        "dropout_rate": 0.2,
        "dw_padding": [],
    },
    "b1": {
        "hidden_dim": 1280,
        "width_coef": 1.0,
        "depth_coef": 1.1,
        "image_size": 240,
        "dropout_rate": 0.2,
        "dw_padding": [16],
    },
    "b2": {
        "hidden_dim": 1408,
        "width_coef": 1.1,
        "depth_coef": 1.2,
        "image_size": 260,
        "dropout_rate": 0.3,
        "dw_padding": [5, 8, 16],
    },
    "b3": {
        "hidden_dim": 1536,
        "width_coef": 1.2,
        "depth_coef": 1.4,
        "image_size": 300,
        "dropout_rate": 0.3,
        "dw_padding": [5, 18],
    },
    "b4": {
        "hidden_dim": 1792,
        "width_coef": 1.4,
        "depth_coef": 1.8,
        "image_size": 380,
        "dropout_rate": 0.4,
        "dw_padding": [6],
    },
    "b5": {
        "hidden_dim": 2048,
        "width_coef": 1.6,
        "depth_coef": 2.2,
        "image_size": 456,
        "dropout_rate": 0.4,
        "dw_padding": [13, 27],
    },
    "b6": {
        "hidden_dim": 2304,
        "width_coef": 1.8,
        "depth_coef": 2.6,
        "image_size": 528,
        "dropout_rate": 0.5,
        "dw_padding": [31],
    },
    "b7": {  # 定义一个名为 "b7" 的字典项
        "hidden_dim": 2560,  # 设置 "hidden_dim" 键的值为 2560，表示隐藏维度
        "width_coef": 2.0,   # 设置 "width_coef" 键的值为 2.0，表示宽度系数
        "depth_coef": 3.1,   # 设置 "depth_coef" 键的值为 3.1，表示深度系数
        "image_size": 600,   # 设置 "image_size" 键的值为 600，表示图像尺寸
        "dropout_rate": 0.5, # 设置 "dropout_rate" 键的值为 0.5，表示丢弃率
        "dw_padding": [18],  # 设置 "dw_padding" 键的值为 [18]，表示深度可分离卷积的填充
    },
# 结束上一个函数或代码块的定义，空行隔开，准备定义下一个函数
}


# 根据模型名称获取 EfficientNet 的配置信息
def get_efficientnet_config(model_name):
    # 创建一个 EfficientNetConfig 对象
    config = EfficientNetConfig()
    # 设置隐藏层维度
    config.hidden_dim = CONFIG_MAP[model_name]["hidden_dim"]
    # 设置宽度系数
    config.width_coefficient = CONFIG_MAP[model_name]["width_coef"]
    # 设置深度系数
    config.depth_coefficient = CONFIG_MAP[model_name]["depth_coef"]
    # 设置图像大小
    config.image_size = CONFIG_MAP[model_name]["image_size"]
    # 设置 dropout 率
    config.dropout_rate = CONFIG_MAP[model_name]["dropout_rate"]
    # 设置深度可分离卷积的填充方式
    config.depthwise_padding = CONFIG_MAP[model_name]["dw_padding"]

    # 下载并加载预训练模型对应的标签文件
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 转换标签的键值对，将键转为整数
    id2label = {int(k): v for k, v in id2label.items()}

    # 将转换后的标签映射设置到配置对象中
    config.id2label = id2label
    # 创建一个反向映射，从标签到 ID
    config.label2id = {v: k for k, v in id2label.items()}
    # 返回配置对象
    return config


# 准备一个包含可爱猫图像的函数用于验证结果
def prepare_img():
    # 图像 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过请求获取图像的原始字节流并打开为 Image 对象
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回图像对象
    return im


# 根据模型名称创建图像处理器对象
def convert_image_processor(model_name):
    # 获取图像处理器所需的图像大小
    size = CONFIG_MAP[model_name]["image_size"]
    # 创建 EfficientNetImageProcessor 对象，并设置参数
    preprocessor = EfficientNetImageProcessor(
        size={"height": size, "width": size},  # 设置图像高度和宽度
        image_mean=[0.485, 0.456, 0.406],  # 设置图像均值
        image_std=[0.47853944, 0.4732864, 0.47434163],  # 设置图像标准差
        do_center_crop=False,  # 是否进行中心裁剪
    )
    # 返回图像处理器对象
    return preprocessor


# 列出所有需要重命名的键值对（左侧为原始名称，右侧为新名称）
def rename_keys(original_param_names):
    # 从原始参数名称中提取出块的名称，并排序去重
    block_names = [v.split("_")[0].split("block")[1] for v in original_param_names if v.startswith("block")]
    block_names = sorted(set(block_names))
    num_blocks = len(block_names)
    # 创建块名称与数字索引的映射关系
    block_name_mapping = {b: str(i) for b, i in zip(block_names, range(num_blocks))}

    # 初始化重命名列表
    rename_keys = []
    # 添加需要重命名的键值对
    rename_keys.append(("stem_conv/kernel:0", "embeddings.convolution.weight"))
    rename_keys.append(("stem_bn/gamma:0", "embeddings.batchnorm.weight"))
    rename_keys.append(("stem_bn/beta:0", "embeddings.batchnorm.bias"))
    rename_keys.append(("stem_bn/moving_mean:0", "embeddings.batchnorm.running_mean"))
    rename_keys.append(("stem_bn/moving_variance:0", "embeddings.batchnorm.running_var"))
    # 遍历给定的块名称列表
    for b in block_names:
        # 获取当前块的映射索引
        hf_b = block_name_mapping[b]
        
        # 添加重命名键值对，映射卷积核权重的原始路径到目标路径
        rename_keys.append((f"block{b}_expand_conv/kernel:0", f"encoder.blocks.{hf_b}.expansion.expand_conv.weight"))
        # 添加重命名键值对，映射批归一化层的 gamma 参数路径
        rename_keys.append((f"block{b}_expand_bn/gamma:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.weight"))
        # 添加重命名键值对，映射批归一化层的 beta 参数路径
        rename_keys.append((f"block{b}_expand_bn/beta:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.bias"))
        # 添加重命名键值对，映射批归一化层的移动均值路径
        rename_keys.append((f"block{b}_expand_bn/moving_mean:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_mean"))
        # 添加重命名键值对，映射批归一化层的移动方差路径
        rename_keys.append((f"block{b}_expand_bn/moving_variance:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_var"))
        
        # 添加重命名键值对，映射深度可分离卷积层的深度卷积核权重路径
        rename_keys.append((f"block{b}_dwconv/depthwise_kernel:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_conv.weight"))
        # 添加重命名键值对，映射深度可分离卷积层的批归一化 gamma 参数路径
        rename_keys.append((f"block{b}_bn/gamma:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.weight"))
        # 添加重命名键值对，映射深度可分离卷积层的批归一化 beta 参数路径
        rename_keys.append((f"block{b}_bn/beta:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.bias"))
        # 添加重命名键值对，映射深度可分离卷积层的批归一化移动均值路径
        rename_keys.append((f"block{b}_bn/moving_mean:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_mean"))
        # 添加重命名键值对，映射深度可分离卷积层的批归一化移动方差路径
        rename_keys.append((f"block{b}_bn/moving_variance:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_var"))
        
        # 添加重命名键值对，映射 Squeeze-and-Excitation 网络中的压缩卷积核权重路径
        rename_keys.append((f"block{b}_se_reduce/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.weight"))
        # 添加重命名键值对，映射 Squeeze-and-Excitation 网络中的压缩偏置路径
        rename_keys.append((f"block{b}_se_reduce/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.bias"))
        # 添加重命名键值对，映射 Squeeze-and-Excitation 网络中的扩展卷积核权重路径
        rename_keys.append((f"block{b}_se_expand/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.weight"))
        # 添加重命名键值对，映射 Squeeze-and-Excitation 网络中的扩展偏置路径
        rename_keys.append((f"block{b}_se_expand/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.bias"))
        
        # 添加重命名键值对，映射最终投影卷积层的卷积核权重路径
        rename_keys.append((f"block{b}_project_conv/kernel:0", f"encoder.blocks.{hf_b}.projection.project_conv.weight"))
        # 添加重命名键值对，映射最终投影卷积层的批归一化 gamma 参数路径
        rename_keys.append((f"block{b}_project_bn/gamma:0", f"encoder.blocks.{hf_b}.projection.project_bn.weight"))
        # 添加重命名键值对，映射最终投影卷积层的批归一化 beta 参数路径
        rename_keys.append((f"block{b}_project_bn/beta:0", f"encoder.blocks.{hf_b}.projection.project_bn.bias"))
        # 添加重命名键值对，映射最终投影卷积层的批归一化移动均值路径
        rename_keys.append((f"block{b}_project_bn/moving_mean:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_mean"))
        # 添加重命名键值对，映射最终投影卷积层的批归一化移动方差路径
        rename_keys.append((f"block{b}_project_bn/moving_variance:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_var"))

    # 添加重命名键值对，映射顶部卷积层的卷积核权重路径
    rename_keys.append(("top_conv/kernel:0", "encoder.top_conv.weight"))
    # 添加重命名键值对，映射顶部卷积层的批归一化 gamma 参数路径
    rename_keys.append(("top_bn/gamma:0", "encoder.top_bn.weight"))
    # 添加重命名键值对，映射顶部卷积层的批归一化 beta 参数路径
    rename_keys.append(("top_bn/beta:0", "encoder.top_bn.bias"))
    # 添加重命名键值对，映射顶部卷积层的批归一化移动均值路径
    rename_keys.append(("top_bn/moving_mean:0", "encoder.top_bn.running_mean"))
    # 添加重命名键值对，映射顶部卷积层的批归一化移动方差路径
    rename_keys.append(("top_bn/moving_variance:0", "encoder.top_bn.running_var"))

    # 创建空字典，用于最终的键映射
    key_mapping = {}
    # 遍历重命名映射列表中的每个项
    for item in rename_keys:
        # 检查重命名映射的原始参数名是否在原始参数名列表中
        if item[0] in original_param_names:
            # 如果存在，将原始参数名映射到新的 efficientnet 模型中的对应位置
            key_mapping[item[0]] = "efficientnet." + item[1]

    # 将特定的预测层权重映射到分类器的权重和偏置项
    key_mapping["predictions/kernel:0"] = "classifier.weight"
    key_mapping["predictions/bias:0"] = "classifier.bias"

    # 返回最终的参数名映射字典
    return key_mapping
# 替换模型参数，将 TensorFlow 模型参数转换为 HuggingFace 模型参数
def replace_params(hf_params, tf_params, key_mapping):
    # 遍历 TensorFlow 模型参数字典
    for key, value in tf_params.items():
        # 如果参数名中包含 "normalization"，跳过当前循环
        if "normalization" in key:
            continue
        
        # 根据映射表获取对应的 HuggingFace 模型参数名
        hf_key = key_mapping[key]

        # 根据不同的参数类型进行转换和调整
        if "_conv" in key and "kernel" in key:
            # 对卷积核参数进行转置和维度置换，从 TensorFlow 到 PyTorch 格式
            new_hf_value = torch.from_numpy(value).permute(3, 2, 0, 1)
        elif "depthwise_kernel" in key:
            # 对深度可分离卷积核参数进行维度置换
            new_hf_value = torch.from_numpy(value).permute(2, 3, 0, 1)
        elif "kernel" in key:
            # 对一般卷积核参数进行转置
            new_hf_value = torch.from_numpy(np.transpose(value))
        else:
            # 直接转换为 PyTorch 张量
            new_hf_value = torch.from_numpy(value)

        # 使用新值替换 HuggingFace 模型的参数，并断言形状一致
        assert hf_params[hf_key].shape == new_hf_value.shape
        hf_params[hf_key].copy_(new_hf_value)


@torch.no_grad()
def convert_efficientnet_checkpoint(model_name, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our EfficientNet structure.
    """
    # 加载原始模型
    original_model = model_classes[model_name](
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    # 获取 TensorFlow 模型的可训练和不可训练参数
    tf_params = original_model.trainable_variables
    tf_non_train_params = original_model.non_trainable_variables
    tf_params = {param.name: param.numpy() for param in tf_params}
    for param in tf_non_train_params:
        tf_params[param.name] = param.numpy()
    tf_param_names = list(tf_params.keys())

    # 加载 HuggingFace 模型
    config = get_efficientnet_config(model_name)
    hf_model = EfficientNetForImageClassification(config).eval()
    hf_params = hf_model.state_dict()

    # 创建源到目标参数名的映射字典
    print("Converting parameters...")
    key_mapping = rename_keys(tf_param_names)
    
    # 调用替换参数函数，将 TensorFlow 参数转换为 HuggingFace 参数
    replace_params(hf_params, tf_params, key_mapping)

    # 初始化预处理器并对输入图像进行预处理
    preprocessor = convert_image_processor(model_name)
    inputs = preprocessor(images=prepare_img(), return_tensors="pt")

    # 在 HuggingFace 模型上进行推理
    hf_model.eval()
    with torch.no_grad():
        outputs = hf_model(**inputs)
    hf_logits = outputs.logits.detach().numpy()

    # 在原始模型上进行推理
    original_model.trainable = False
    image_size = CONFIG_MAP[model_name]["image_size"]
    img = prepare_img().resize((image_size, image_size), resample=PIL.Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    original_logits = original_model.predict(x)

    # 检查原始模型和 HuggingFace 模型输出是否匹配 -> 使用 np.allclose 函数
    assert np.allclose(original_logits, hf_logits, atol=1e-3), "The predicted logits are not the same."
    print("Model outputs match!")
    # 如果需要保存模型
    if save_model:
        # 创建用于保存模型的文件夹
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        # 将转换后的模型和图像处理器保存到指定路径
        hf_model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 打印消息，说明正在将模型推送到 Hub
        print(f"Pushing converted {model_name} to the hub...")
        # 修改模型名称为 efficientnet-<model_name>
        model_name = f"efficientnet-{model_name}"
        # 将预处理器推送到 Hub，使用修改后的模型名称
        preprocessor.push_to_hub(model_name)
        # 将模型推送到 Hub，使用修改后的模型名称
        hf_model.push_to_hub(model_name)
if __name__ == "__main__":
    # 如果这个模块是直接执行的主程序，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="b0",
        type=str,
        help="Version name of the EfficientNet model you want to convert, select from [b0, b1, b2, b3, b4, b5, b6, b7].",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="hf_model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )

    # 添加可选参数
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数，将命令行参数传递给函数
    convert_efficientnet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub)
```