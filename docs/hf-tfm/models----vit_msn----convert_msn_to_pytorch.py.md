# `.\models\vit_msn\convert_msn_to_pytorch.py`

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
"""Convert ViT MSN checkpoints from the original repository: https://github.com/facebookresearch/msn"""

import argparse  # 导入解析命令行参数的库
import json  # 导入处理 JSON 格式数据的库

import requests  # 导入进行 HTTP 请求的库
import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载模型的功能
from PIL import Image  # 导入 Python Imaging Library，用于处理图像

from transformers import ViTImageProcessor, ViTMSNConfig, ViTMSNModel  # 导入用于处理 ViT 模型的相关类
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # 导入图像处理相关的常量


torch.set_grad_enabled(False)  # 禁用梯度计算

# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []  # 初始化空列表，用于存储重命名的键值对
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"module.blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"module.blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"module.blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append((f"module.blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"module.blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"module.blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("module.cls_token", "vit.embeddings.cls_token"),
            ("module.patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"),
            ("module.patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"),
            ("module.pos_embed", "vit.embeddings.position_embeddings"),
        ]
    )
    # 如果存在基础模型，则执行以下操作
    if base_model:
        # 将以下键值对添加到 rename_keys 列表中，用于重命名
        rename_keys.extend(
            [
                ("module.norm.weight", "layernorm.weight"),  # 将 "module.norm.weight" 重命名为 "layernorm.weight"
                ("module.norm.bias", "layernorm.bias"),      # 将 "module.norm.bias" 重命名为 "layernorm.bias"
            ]
        )

        # 如果只有基础模型，需要从所有以 "vit" 开头的键中删除 "vit"
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # 如果没有基础模型，则执行以下操作
        # 将以下键值对添加到 rename_keys 列表中，用于重命名
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),   # 将 "norm.weight" 重命名为 "vit.layernorm.weight"
                ("norm.bias", "vit.layernorm.bias"),       # 将 "norm.bias" 重命名为 "vit.layernorm.bias"
                ("head.weight", "classifier.weight"),      # 将 "head.weight" 重命名为 "classifier.weight"
                ("head.bias", "classifier.bias"),          # 将 "head.bias" 重命名为 "classifier.bias"
            ]
        )

    # 返回重命名后的键值对列表
    return rename_keys
# 将每个编码器层的权重矩阵分割为查询（query）、键（key）和值（value）
def read_in_q_k_v(state_dict, config, base_model=False):
    # 遍历每个编码器层
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        
        # 读取输入投影层的权重和偏置（在 timm 中，这是一个单独的矩阵加上偏置）
        in_proj_weight = state_dict.pop(f"module.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"module.blocks.{i}.attn.qkv.bias")
        
        # 将查询（query）、键（key）和值（value）依次添加到状态字典中
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 从状态字典中移除分类头部的权重和偏置
def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 从状态字典中移除投影头部的相关键
def remove_projection_head(state_dict):
    # 投影头部在自监督预训练中使用，但在下游任务中不需要
    ignore_keys = [
        "module.fc.fc1.weight",
        "module.fc.fc1.bias",
        "module.fc.bn1.weight",
        "module.fc.bn1.bias",
        "module.fc.bn1.running_mean",
        "module.fc.bn1.running_var",
        "module.fc.bn1.num_batches_tracked",
        "module.fc.fc2.weight",
        "module.fc.fc2.bias",
        "module.fc.bn2.weight",
        "module.fc.bn2.bias",
        "module.fc.bn2.running_mean",
        "module.fc.bn2.running_var",
        "module.fc.bn2.num_batches_tracked",
        "module.fc.fc3.weight",
        "module.fc.fc3.bias",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 将字典中的键从旧名称重命名为新名称
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 将 ViT-MSN 模型的检查点转换为 PyTorch 模型
def convert_vit_msn_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    config = ViTMSNConfig()
    config.num_labels = 1000

    repo_id = "datasets/huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    
    # 从 HF Hub 下载 imagenet-1k-id2label.json 文件，并加载为 id 到 label 的映射字典
    id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    # 根据 checkpoint_url 的内容设置不同的配置参数
    if "s16" in checkpoint_url:
        # 如果包含 "s16"，设置较小的隐藏层大小、中间层大小和注意力头数
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_attention_heads = 6
    elif "l16" in checkpoint_url:
        # 如果包含 "l16"，设置较大的隐藏层大小、中间层大小、层数、注意力头数和隐藏层的 dropout 概率
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.hidden_dropout_prob = 0.1
    elif "b4" in checkpoint_url:
        # 如果包含 "b4"，设置较小的图像块大小
        config.patch_size = 4
    elif "l7" in checkpoint_url:
        # 如果包含 "l7"，设置较大的图像块大小、较大的隐藏层大小、中间层大小、层数、注意力头数和隐藏层的 dropout 概率
        config.patch_size = 7
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.hidden_dropout_prob = 0.1

    # 使用配置参数初始化 ViTMSNModel 模型
    model = ViTMSNModel(config)

    # 从指定的 URL 加载预训练模型的状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["target_encoder"]

    # 创建图像处理器对象，设置图像大小为 config.image_size
    image_processor = ViTImageProcessor(size=config.image_size)

    # 移除模型状态字典中的投影头部分
    remove_projection_head(state_dict)
    
    # 根据配置创建新的键名映射列表
    rename_keys = create_rename_keys(config, base_model=True)

    # 对状态字典中的键名进行重命名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    
    # 读取状态字典中的查询、键、值信息，针对基础模型
    read_in_q_k_v(state_dict, config, base_model=True)

    # 加载模型的状态字典
    model.load_state_dict(state_dict)
    # 设置模型为评估模式
    model.eval()

    # 设置图像 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 使用 requests 库获取图像数据流，并用 PIL 库打开图像
    image = Image.open(requests.get(url, stream=True).raw)
    
    # 创建图像处理器对象，设置图像大小、图像均值和标准差
    image_processor = ViTImageProcessor(
        size=config.image_size, image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD
    )
    
    # 对输入图像进行处理，返回 PyTorch 张量
    inputs = image_processor(images=image, return_tensors="pt")

    # 执行前向传播
    torch.manual_seed(2)
    outputs = model(**inputs)
    # 获取最后一层隐藏状态
    last_hidden_state = outputs.last_hidden_state

    # 验证预测的对数值是否接近预期值
    if "s16" in checkpoint_url:
        expected_slice = torch.tensor([[-1.0915, -1.4876, -1.1809]])
    elif "b16" in checkpoint_url:
        expected_slice = torch.tensor([[14.2889, -18.9045, 11.7281]])
    elif "l16" in checkpoint_url:
        expected_slice = torch.tensor([[41.5028, -22.8681, 45.6475]])
    elif "b4" in checkpoint_url:
        expected_slice = torch.tensor([[-4.3868, 5.2932, -0.4137]])
    else:
        expected_slice = torch.tensor([[-0.1792, -0.6465, 2.4263]])

    # 使用 assert 验证张量的所有元素是否在指定的误差范围内接近预期的值
    assert torch.allclose(last_hidden_state[:, 0, :3], expected_slice, atol=1e-4)

    # 打印模型保存的路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 打印图像处理器保存的路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码
    parser = argparse.ArgumentParser()
    # 创建命令行参数解析器对象

    # 必选参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar",
        type=str,
        help="URL of the checkpoint you'd like to convert."
    )
    # 添加命令行参数，指定模型检查点的下载链接，默认为 Facebook 提供的一个预训练模型的链接

    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        help="Path to the output PyTorch model directory."
    )
    # 添加命令行参数，指定输出的 PyTorch 模型保存目录的路径，默认为 None，即没有指定路径

    # 解析命令行参数，并将其存储在 args 变量中
    args = parser.parse_args()

    # 调用 convert_vit_msn_checkpoint 函数，传入命令行参数中指定的模型下载链接和保存路径
    convert_vit_msn_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
```