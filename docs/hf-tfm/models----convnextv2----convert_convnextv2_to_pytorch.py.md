# `.\models\convnextv2\convert_convnextv2_to_pytorch.py`

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
"""Convert ConvNeXTV2 checkpoints from the original repository.

URL: https://github.com/facebookresearch/ConvNeXt"""

import argparse  # 导入用于处理命令行参数的模块
import json  # 导入用于处理JSON格式数据的模块
import os  # 导入用于操作系统相关功能的模块

import requests  # 导入用于发送HTTP请求的模块
import torch  # 导入PyTorch深度学习库
from huggingface_hub import hf_hub_download  # 导入Hugging Face Hub下载模块
from PIL import Image  # 导入Python Imaging Library，用于图像处理

from transformers import ConvNextImageProcessor, ConvNextV2Config, ConvNextV2ForImageClassification  # 导入转换器模块
from transformers.image_utils import PILImageResampling  # 导入图像处理工具模块
from transformers.utils import logging  # 导入日志记录工具模块


logging.set_verbosity_info()  # 设置日志记录详细程度为信息级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_convnextv2_config(checkpoint_url):
    config = ConvNextV2Config()  # 创建ConvNeXTV2配置对象

    # 根据checkpoint_url中的关键词设置深度和隐藏层大小
    if "atto" in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [40, 80, 160, 320]
    if "femto" in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [48, 96, 192, 384]
    if "pico" in checkpoint_url:
        depths = [2, 2, 6, 2]
        hidden_sizes = [64, 128, 256, 512]
    if "nano" in checkpoint_url:
        depths = [2, 2, 8, 2]
        hidden_sizes = [80, 160, 320, 640]
    if "tiny" in checkpoint_url:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if "base" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
    if "large" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
    if "huge" in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [352, 704, 1408, 2816]

    num_labels = 1000  # 设置分类标签数量为1000
    filename = "imagenet-1k-id2label.json"  # 设置包含标签映射的文件名
    expected_shape = (1, 1000)  # 设置预期输出形状

    repo_id = "huggingface/label-files"  # 设置Hugging Face Hub仓库ID
    config.num_labels = num_labels  # 设置配置对象中的分类标签数量
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))  # 下载并加载标签映射数据
    id2label = {int(k): v for k, v in id2label.items()}  # 转换标签映射数据格式为整数到标签的字典形式

    config.id2label = id2label  # 设置配置对象中的id到标签的映射
    config.label2id = {v: k for k, v in id2label.items()}  # 设置配置对象中的标签到id的映射
    config.hidden_sizes = hidden_sizes  # 设置配置对象中的隐藏层大小列表
    config.depths = depths  # 设置配置对象中的网络深度列表

    return config, expected_shape  # 返回配置对象和预期输出形状的元组


def rename_key(name):
    if "downsample_layers.0.0" in name:
        name = name.replace("downsample_layers.0.0", "embeddings.patch_embeddings")  # 将指定的键名替换为新的名称
    if "downsample_layers.0.1" in name:
        name = name.replace("downsample_layers.0.1", "embeddings.norm")  # 将指定的键名替换为新的名称（稍后重命名为layernorm）
    if "downsample_layers.1.0" in name:
        name = name.replace("downsample_layers.1.0", "stages.1.downsampling_layer.0")  # 将指定的键名替换为新的名称
    # 检查字符串变量 name 是否包含 "downsample_layers.1.1"，如果是则替换为 "stages.1.downsampling_layer.1"
    if "downsample_layers.1.1" in name:
        name = name.replace("downsample_layers.1.1", "stages.1.downsampling_layer.1")

    # 检查字符串变量 name 是否包含 "downsample_layers.2.0"，如果是则替换为 "stages.2.downsampling_layer.0"
    if "downsample_layers.2.0" in name:
        name = name.replace("downsample_layers.2.0", "stages.2.downsampling_layer.0")

    # 检查字符串变量 name 是否包含 "downsample_layers.2.1"，如果是则替换为 "stages.2.downsampling_layer.1"
    if "downsample_layers.2.1" in name:
        name = name.replace("downsample_layers.2.1", "stages.2.downsampling_layer.1")

    # 检查字符串变量 name 是否包含 "downsample_layers.3.0"，如果是则替换为 "stages.3.downsampling_layer.0"
    if "downsample_layers.3.0" in name:
        name = name.replace("downsample_layers.3.0", "stages.3.downsampling_layer.0")

    # 检查字符串变量 name 是否包含 "downsample_layers.3.1"，如果是则替换为 "stages.3.downsampling_layer.1"
    if "downsample_layers.3.1" in name:
        name = name.replace("downsample_layers.3.1", "stages.3.downsampling_layer.1")

    # 检查字符串变量 name 是否包含 "stages"，但不包含 "downsampling_layer"，则进行部分替换
    # 例如，"stages.0.0" 应替换为 "stages.0.layers.0"
    if "stages" in name and "downsampling_layer" not in name:
        name = name[: len("stages.0")] + ".layers" + name[len("stages.0") :]

    # 检查字符串变量 name 是否包含 "gamma"，如果是则替换为 "weight"
    if "gamma" in name:
        name = name.replace("gamma", "weight")

    # 检查字符串变量 name 是否包含 "beta"，如果是则替换为 "bias"
    if "beta" in name:
        name = name.replace("beta", "bias")

    # 检查字符串变量 name 是否包含 "stages"，如果是则替换为 "encoder.stages"
    if "stages" in name:
        name = name.replace("stages", "encoder.stages")

    # 检查字符串变量 name 是否包含 "norm"，如果是则替换为 "layernorm"
    if "norm" in name:
        name = name.replace("norm", "layernorm")

    # 检查字符串变量 name 是否包含 "head"，如果是则替换为 "classifier"
    if "head" in name:
        name = name.replace("head", "classifier")

    # 返回修改后的字符串变量 name
    return name
# 准备一个图像数据，用于模型验证
def prepare_img():
    # 图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 库获取图片数据流，并使用 PIL 库打开图像
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 根据 checkpoint_url 参数选择合适的预处理器对象
def convert_preprocessor(checkpoint_url):
    if "224" in checkpoint_url:
        # 如果 URL 中包含 "224"，则选择大小为 224，并计算裁剪比例
        size = 224
        crop_pct = 224 / 256
    elif "384" in checkpoint_url:
        # 如果 URL 中包含 "384"，则选择大小为 384，无需裁剪
        size = 384
        crop_pct = None
    else:
        # 否则，默认选择大小为 512，无需裁剪
        size = 512
        crop_pct = None

    # 返回一个 ConvNextImageProcessor 对象，其中包括所选参数和图像标准化的设置
    return ConvNextImageProcessor(
        size=size,
        crop_pct=crop_pct,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        resample=PILImageResampling.BICUBIC,
    )


# 使用 @torch.no_grad() 装饰器，确保转换过程中不进行梯度计算
@torch.no_grad()
# 根据给定的 checkpoint_url，将模型权重复制/调整到 ConvNeXTV2 结构中
def convert_convnextv2_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ConvNeXTV2 structure.
    """
    # 打印信息：从 checkpoint_url 下载原始模型
    print("Downloading original model from checkpoint...")
    # 根据 URL 获取 ConvNeXTV2 的配置和预期的形状
    config, expected_shape = get_convnextv2_config(checkpoint_url)
    # 从 URL 加载原始模型的状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]

    # 打印信息：转换模型参数
    print("Converting model parameters...")
    # 重命名状态字典的键
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # 给所有键添加前缀，除了以 "classifier" 开头的键
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith("classifier"):
            key = "convnextv2." + key
        state_dict[key] = val

    # 加载 HuggingFace 模型结构
    model = ConvNextV2ForImageClassification(config)
    # 加载转换后的状态字典到模型中
    model.load_state_dict(state_dict)
    # 将模型设置为评估模式
    model.eval()

    # 准备 ConvNextImageProcessor 对象，用于处理图像输入
    preprocessor = convert_preprocessor(checkpoint_url)
    # 准备图像数据，返回 PyTorch 张量
    inputs = preprocessor(images=prepare_img(), return_tensors="pt")
    # 执行模型推理，得到预测 logits
    logits = model(**inputs).logits

    # 检查 logits 是否符合预期，根据不同的 checkpoint_url 设置不同的预期 logits
    if checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.3930, 0.1747, -0.5246, 0.4177, 0.4295])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.1727, -0.5341, -0.7818, -0.4745, -0.6566])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.0333, 0.1563, -0.9137, 0.1054, 0.0381])
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.1744, -0.1555, -0.0713, 0.0950, -0.1431])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt":
        expected_logits = torch.tensor([0.9996, 0.1966, -0.4386, -0.3472, 0.6661])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.2553, -0.6708, -0.1359, 0.2518, -0.2488])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.0673, -0.5627, -0.3753, -0.2722, 0.0178])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt":
        expected_logits = torch.tensor([-0.6377, -0.7458, -0.2150, 0.1184, -0.0597])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt":
        expected_logits = torch.tensor([1.0799, 0.2322, -0.8860, 1.0219, 0.6231])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt":
        expected_logits = torch.tensor([0.3766, 0.4917, -1.1426, 0.9942, 0.6024])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt":
        expected_logits = torch.tensor([0.4220, -0.6919, -0.4317, -0.2881, -0.6609])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt":
        expected_logits = torch.tensor([0.1082, -0.8286, -0.5095, 0.4681, -0.8085])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt":
        expected_logits = torch.tensor([-0.2419, -0.6221, 0.2176, -0.0980, -0.7527])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt":
        expected_logits = torch.tensor([0.0391, -0.4371, 0.3786, 0.1251, -0.2784])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt":
        expected_logits = torch.tensor([-0.0504, 0.5636, -0.1729, -0.6507, -0.3949])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt":
        expected_logits = torch.tensor([0.3560, 0.9486, 0.3149, -0.2667, -0.5138])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_384_ema.pt":
        expected_logits = torch.tensor([-0.2469, -0.4550, -0.5853, -0.0810, 0.0309])
    # 如果 checkpoint_url 是特定的 URL，则设置预期的 logits 值为特定的张量
    elif checkpoint_url == "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt":
        expected_logits = torch.tensor([-0.3090, 0.0802, -0.0682, -0.1979, -0.2826])
    else:
        # 如果 URL 不匹配任何已知的模型文件，抛出 ValueError 异常
        raise ValueError(f"Unknown URL: {checkpoint_url}")

    # 使用 assert 语句检查 logits 的前五个元素是否与预期的 logits 非常接近
    assert torch.allclose(logits[0, :5], expected_logits, atol=1e-3)
    # 断言确保 logits 的形状与期望形状相匹配
    assert logits.shape == expected_shape
    # 打印信息表明模型输出与原始结果匹配
    print("Model outputs match the original results!")

    # 如果需要保存模型
    if save_model:
        # 打印信息表示正在保存模型到本地
        print("Saving model to local...")
        # 创建用于保存模型的文件夹（如果不存在）
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)

        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将预处理器保存到同一路径
        preprocessor.save_pretrained(pytorch_dump_folder_path)

    # 设置模型名称为 "convnextv2"
    model_name = "convnextv2"
    # 根据 checkpoint_url 的内容修改模型名称
    if "atto" in checkpoint_url:
        model_name += "-atto"
    if "femto" in checkpoint_url:
        model_name += "-femto"
    if "pico" in checkpoint_url:
        model_name += "-pico"
    # 以下是根据 checkpoint_url 中不同关键词修改模型名称的逻辑
    if "nano" in checkpoint_url:
        model_name += "-nano"
    elif "tiny" in checkpoint_url:
        model_name += "-tiny"
    elif "base" in checkpoint_url:
        model_name += "-base"
    elif "large" in checkpoint_url:
        model_name += "-large"
    elif "huge" in checkpoint_url:
        model_name += "-huge"
    # 进一步修改模型名称，考虑包含 "22k" 和 "1k" 的情况
    if "22k" in checkpoint_url and "1k" not in checkpoint_url:
        model_name += "-22k"
    elif "22k" in checkpoint_url and "1k" in checkpoint_url:
        model_name += "-22k-1k"
    elif "1k" in checkpoint_url:
        model_name += "-1k"
    # 最后根据 checkpoint_url 中包含的分辨率信息修改模型名称
    if "224" in checkpoint_url:
        model_name += "-224"
    elif "384" in checkpoint_url:
        model_name += "-384"
    elif "512" in checkpoint_url:
        model_name += "-512"

    # 如果需要将模型推送到 hub
    if push_to_hub:
        # 打印信息表示正在将模型推送到 hub
        print(f"Pushing {model_name} to the hub...")
        # 将模型推送到 hub 上
        model.push_to_hub(model_name)
        # 将预处理器也推送到 hub 上，使用相同的模型名称
        preprocessor.push_to_hub(model_name)
if __name__ == "__main__":
    # 如果这个脚本是作为主程序执行的话，则执行以下代码

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt",
        type=str,
        help="URL of the original ConvNeXTV2 checkpoint you'd like to convert.",
    )
    # 添加名为 `--checkpoint_url` 的参数，用于指定要转换的 ConvNeXTV2 原始检查点的 URL

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加名为 `--pytorch_dump_folder_path` 的参数，用于指定输出的 PyTorch 模型目录的路径，默认为 "model"

    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    # 添加名为 `--save_model` 的可选参数，如果指定则将模型保存到本地

    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")
    # 添加名为 `--push_to_hub` 的可选参数，如果指定则将模型和图像预处理器推送到 Hub

    args = parser.parse_args()
    # 解析命令行参数并返回一个 Namespace 对象

    convert_convnextv2_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub
    )
    # 调用函数 `convert_convnextv2_checkpoint`，传入命令行参数中解析的对应值作为参数
```