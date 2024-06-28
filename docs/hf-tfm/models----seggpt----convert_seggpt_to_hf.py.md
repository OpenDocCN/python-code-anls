# `.\models\seggpt\convert_seggpt_to_hf.py`

```py
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert SegGPT checkpoints from the original repository.

URL: https://github.com/baaivision/Painter/tree/main/SegGPT
"""


import argparse  # 导入命令行参数解析模块

import requests  # 导入处理 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习框架
from PIL import Image  # 导入处理图像的模块

from transformers import SegGptConfig, SegGptForImageSegmentation, SegGptImageProcessor  # 导入 SegGPT 相关模块
from transformers.utils import logging  # 导入日志模块

logging.set_verbosity_info()  # 设置日志的详细程度为 info 级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


# here we list all keys to be renamed (original name on the left, our name on the right)
# 定义一个函数，列出需要重命名的所有键值对（左边是原始名称，右边是我们使用的名称）
def create_rename_keys(config):
    rename_keys = []  # 初始化空的重命名键值对列表

    # fmt: off

    # rename embedding and its parameters
    # 重命名嵌入和其参数
    rename_keys.append(("patch_embed.proj.weight", "model.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "model.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("mask_token", "model.embeddings.mask_token"))
    rename_keys.append(("segment_token_x", "model.embeddings.segment_token_input"))
    rename_keys.append(("segment_token_y", "model.embeddings.segment_token_prompt"))
    rename_keys.append(("type_token_cls", "model.embeddings.type_token_semantic"))
    rename_keys.append(("type_token_ins", "model.embeddings.type_token_instance"))
    rename_keys.append(("pos_embed", "model.embeddings.position_embeddings"))

    # rename decoder and other
    # 重命名解码器和其他部分
    rename_keys.append(("norm.weight", "model.encoder.layernorm.weight"))
    rename_keys.append(("norm.bias", "model.encoder.layernorm.bias"))
    rename_keys.append(("decoder_embed.weight", "decoder.decoder_embed.weight"))
    rename_keys.append(("decoder_embed.bias", "decoder.decoder_embed.bias"))
    rename_keys.append(("decoder_pred.0.weight", "decoder.decoder_pred.conv.weight"))
    rename_keys.append(("decoder_pred.0.bias", "decoder.decoder_pred.conv.bias"))
    rename_keys.append(("decoder_pred.1.weight", "decoder.decoder_pred.layernorm.weight"))
    rename_keys.append(("decoder_pred.1.bias", "decoder.decoder_pred.layernorm.bias"))
    rename_keys.append(("decoder_pred.3.weight", "decoder.decoder_pred.head.weight"))
    rename_keys.append(("decoder_pred.3.bias", "decoder.decoder_pred.head.bias"))

    # rename blocks

    # fmt: on
    # 遍历从 0 到 config.num_hidden_layers-1 的范围，进行重命名键的添加
    for i in range(config.num_hidden_layers):
        # 添加注意力层的权重重命名键
        rename_keys.append((f"blocks.{i}.attn.qkv.weight", f"model.encoder.layers.{i}.attention.qkv.weight"))
        # 添加注意力层的偏置项重命名键
        rename_keys.append((f"blocks.{i}.attn.qkv.bias", f"model.encoder.layers.{i}.attention.qkv.bias"))
        # 添加注意力层投影层权重的重命名键
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"model.encoder.layers.{i}.attention.proj.weight"))
        # 添加注意力层投影层偏置项的重命名键
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"model.encoder.layers.{i}.attention.proj.bias"))
        # 添加注意力层相对位置编码（水平方向）的重命名键
        rename_keys.append((f"blocks.{i}.attn.rel_pos_h", f"model.encoder.layers.{i}.attention.rel_pos_h"))
        # 添加注意力层相对位置编码（垂直方向）的重命名键
        rename_keys.append((f"blocks.{i}.attn.rel_pos_w", f"model.encoder.layers.{i}.attention.rel_pos_w"))

        # 添加多层感知机（MLP）的第一个全连接层权重的重命名键
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"model.encoder.layers.{i}.mlp.lin1.weight"))
        # 添加多层感知机（MLP）的第一个全连接层偏置项的重命名键
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"model.encoder.layers.{i}.mlp.lin1.bias"))
        # 添加多层感知机（MLP）的第二个全连接层权重的重命名键
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"model.encoder.layers.{i}.mlp.lin2.weight"))
        # 添加多层感知机（MLP）的第二个全连接层偏置项的重命名键
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"model.encoder.layers.{i}.mlp.lin2.bias"))

        # 添加注意力层前层归一化权重的重命名键
        rename_keys.append((f"blocks.{i}.norm1.weight", f"model.encoder.layers.{i}.layernorm_before.weight"))
        # 添加注意力层前层归一化偏置项的重命名键
        rename_keys.append((f"blocks.{i}.norm1.bias", f"model.encoder.layers.{i}.layernorm_before.bias"))
        # 添加注意力层后层归一化权重的重命名键
        rename_keys.append((f"blocks.{i}.norm2.weight", f"model.encoder.layers.{i}.layernorm_after.weight"))
        # 添加注意力层后层归一化偏置项的重命名键
        rename_keys.append((f"blocks.{i}.norm2.bias", f"model.encoder.layers.{i}.layernorm_after.bias"))

    # 返回所有添加的重命名键列表
    return rename_keys
# 从字典中移除旧键，并将其对应的值存储在变量val中
def rename_key(dct, old, new):
    val = dct.pop(old)
    # 将旧键的值存储在新键下
    dct[new] = val


# 准备输入数据，包括图像和掩模
def prepare_input():
    # 定义输入图像的URL
    image_input_url = (
        "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_2.jpg"
    )
    # 定义提示图像的URL
    image_prompt_url = (
        "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1.jpg"
    )
    # 定义掩模图像的URL
    mask_prompt_url = (
        "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1_target.png"
    )

    # 使用requests库获取并打开输入图像、提示图像和掩模图像的二进制数据
    image_input = Image.open(requests.get(image_input_url, stream=True).raw)
    image_prompt = Image.open(requests.get(image_prompt_url, stream=True).raw)
    mask_prompt = Image.open(requests.get(mask_prompt_url, stream=True).raw)

    # 返回准备好的图像和掩模
    return image_input, image_prompt, mask_prompt


# 使用torch.no_grad()装饰器，以确保在推理时不会计算梯度
@torch.no_grad()
def convert_seggpt_checkpoint(args):
    # 从参数中获取模型名称、PyTorch模型保存路径、是否验证logits以及是否推送到Hub
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    verify_logits = args.verify_logits
    push_to_hub = args.push_to_hub

    # 定义SegGPT模型的配置，默认使用SegGptConfig()
    config = SegGptConfig()

    # 加载原始的检查点文件，从Hugging Face模型中心加载
    checkpoint_url = "https://huggingface.co/BAAI/SegGpt/blob/main/seggpt_vit_large.pth"
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    # 创建新的状态字典副本
    new_state_dict = original_state_dict.copy()

    # 调用create_rename_keys函数创建需要重命名的键列表
    rename_keys = create_rename_keys(config)

    # 遍历重命名键列表，将新旧键映射应用于new_state_dict
    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)

    # 实例化SegGptForImageSegmentation模型
    model = SegGptForImageSegmentation(config)
    model.eval()

    # 加载新的状态字典到模型中，strict=False表示允许缺失键和多余键
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    # 打印缺失的键和多余的键
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # 准备输入数据，获取输入图像、提示图像和掩模
    input_img, prompt_img, prompt_mask = prepare_input()

    # 实例化SegGptImageProcessor
    image_processor = SegGptImageProcessor()

    # 使用image_processor处理输入图像、提示图像和掩模，返回PyTorch张量
    inputs = image_processor(images=input_img, prompt_images=prompt_img, prompt_masks=prompt_mask, return_tensors="pt")

    # 预期的提示像素值张量，用于验证结果
    expected_prompt_pixel_values = torch.tensor(
        [
            [[-0.6965, -0.6965, -0.6965], [-0.6965, -0.6965, -0.6965], [-0.6965, -0.6965, -0.6965]],
            [[1.6583, 1.6583, 1.6583], [1.6583, 1.6583, 1.6583], [1.6583, 1.6583, 1.6583]],
            [[2.3088, 2.3088, 2.3088], [2.3088, 2.3088, 2.3088], [2.3088, 2.3088, 2.3088]],
        ]
    )

    # 预期的像素值张量，用于验证结果
    expected_pixel_values = torch.tensor(
        [
            [[1.6324, 1.6153, 1.5810], [1.6153, 1.5982, 1.5810], [1.5810, 1.5639, 1.5639]],
            [[1.2731, 1.2556, 1.2206], [1.2556, 1.2381, 1.2031], [1.2206, 1.2031, 1.1681]],
            [[1.6465, 1.6465, 1.6465], [1.6465, 1.6465, 1.6465], [1.6291, 1.6291, 1.6291]],
        ]
    )
    # 定义期望的像素值，这里使用 torch.tensor 创建张量
    expected_prompt_masks = torch.tensor(
        [
            [[-2.1179, -2.1179, -2.1179], [-2.1179, -2.1179, -2.1179], [-2.1179, -2.1179, -2.1179]],
            [[-2.0357, -2.0357, -2.0357], [-2.0357, -2.0357, -2.0357], [-2.0357, -2.0357, -2.0357]],
            [[-1.8044, -1.8044, -1.8044], [-1.8044, -1.8044, -1.8044], [-1.8044, -1.8044, -1.8044]],
        ]
    )

    # 检查模型输入的像素值是否与期望的像素值接近，设置容忍度为 1e-4
    assert torch.allclose(inputs.pixel_values[0, :, :3, :3], expected_pixel_values, atol=1e-4)
    # 检查模型输入的提示像素值是否与期望的像素值接近，设置容忍度为 1e-4
    assert torch.allclose(inputs.prompt_pixel_values[0, :, :3, :3], expected_prompt_values, atol=1e-4)
    # 检查模型输入的提示掩码是否与期望的掩码接近，设置容忍度为 1e-4
    assert torch.allclose(inputs.prompt_masks[0, :, :3, :3], expected_prompt_masks, atol=1e-4)

    # 设置随机种子为 2
    torch.manual_seed(2)
    # 使用模型处理给定的输入
    outputs = model(**inputs)
    # 打印模型输出
    print(outputs)

    # 如果需要验证 logits，检查模型输出的预测掩码是否与期望的输出接近，设置容忍度为 1e-4
    if verify_logits:
        expected_output = torch.tensor(
            [
                [[-2.1208, -2.1190, -2.1198], [-2.1237, -2.1228, -2.1227], [-2.1232, -2.1226, -2.1228]],
                [[-2.0405, -2.0396, -2.0403], [-2.0434, -2.0434, -2.0433], [-2.0428, -2.0432, -2.0434]],
                [[-1.8102, -1.8088, -1.8099], [-1.8131, -1.8126, -1.8129], [-1.8130, -1.8128, -1.8131]],
            ]
        )
        # 检查模型输出的预测掩码是否与期望的输出接近，设置容忍度为 1e-4
        assert torch.allclose(outputs.pred_masks[0, :, :3, :3], expected_output, atol=1e-4)
        # 打印验证通过信息
        print("Looks good!")
    else:
        # 如果不需要验证 logits，则打印转换完成信息
        print("Converted without verifying logits")

    # 如果指定了 PyTorch 导出文件夹路径
    if pytorch_dump_folder_path is not None:
        # 打印保存模型和处理器的信息
        print(f"Saving model and processor for {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将图像处理器保存到指定路径
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 打印推送模型和处理器到 Hub 的信息
        print(f"Pushing model and processor for {model_name} to hub")
        # 推送模型到 Hub
        model.push_to_hub(f"EduardoPacheco/{model_name}")
        # 推送图像处理器到 Hub
        image_processor.push_to_hub(f"EduardoPacheco/{model_name}")
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="seggpt-vit-large",
        type=str,
        choices=["seggpt-vit-large"],
        help="Name of the SegGpt model you'd like to convert.",
    )
    
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory."
    )
    
    # 添加可选的参数
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        help="Whether or not to verify the logits against the original implementation.",
    )
    
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数 convert_seggpt_checkpoint，传入解析后的参数对象 args
    convert_seggpt_checkpoint(args)
```