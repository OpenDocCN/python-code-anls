# `.\models\vit\convert_dino_to_pytorch.py`

```
# coding=utf-8
# 定义编码格式为 UTF-8，确保脚本中可以使用中文注释和字符串

# 版权声明和许可证信息，指明代码的使用条款和条件
# Copyright 2021 The HuggingFace Inc. team.
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

"""Convert ViT checkpoints trained with the DINO method."""

# 导入必要的模块和库
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 格式数据
from pathlib import Path  # 用于处理文件路径的类

import requests  # 用于进行网络请求
import torch  # PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模型和文件
from PIL import Image  # Python Imaging Library，用于处理图像

# 导入需要转换的 ViT 相关类和函数
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor, ViTModel
from transformers.utils import logging  # 引入日志记录功能

# 设置日志输出等级为信息级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义函数：生成需要重命名的键值对列表
# 根据 ViT 模型配置，将原始的键名映射为新的键名
def create_rename_keys(config, base_model=False):
    rename_keys = []
    # 遍历 ViT 模型的所有隐藏层
    for i in range(config.num_hidden_layers):
        # 对每一层进行重命名映射
        rename_keys.append((f"blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))

    # 还需重命名投影层和位置嵌入
    rename_keys.extend(
        [
            ("cls_token", "vit.embeddings.cls_token"),
            ("patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "vit.embeddings.position_embeddings"),
        ]
    )
    # 如果存在基础模型（base_model为真），执行以下操作：
    if base_model:
        # 将以下键值对追加到rename_keys列表中，用于后续重命名
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),  # 将"norm.weight"重命名为"layernorm.weight"
                ("norm.bias", "layernorm.bias"),      # 将"norm.bias"重命名为"layernorm.bias"
            ]
        )

        # 对于以"vit"开头的所有键名，去除开头的"vit"（如果仅有基础模型时使用）
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # 如果不存在基础模型，执行以下操作：
        # 将以下键值对追加到rename_keys列表中，用于后续重命名
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),   # 将"norm.weight"重命名为"vit.layernorm.weight"
                ("norm.bias", "vit.layernorm.bias"),       # 将"norm.bias"重命名为"vit.layernorm.bias"
                ("head.weight", "classifier.weight"),      # 将"head.weight"重命名为"classifier.weight"
                ("head.bias", "classifier.bias"),          # 将"head.bias"重命名为"classifier.bias"
            ]
        )

    # 返回处理后的rename_keys列表，其中包含了根据条件不同而进行的键重命名操作
    return rename_keys
# 将每个编码器层的矩阵分割为查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config, base_model=False):
    # 遍历每一个编码器层
    for i in range(config.num_hidden_layers):
        # 如果是基础模型，则前缀为空字符串；否则前缀为"vit."
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        
        # 从状态字典中弹出输入投影层的权重和偏置（在timm中，这是一个单独的矩阵加上偏置）
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        
        # 将查询（query）、键（key）、和值（value）依次添加到状态字典中
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


# 从状态字典中移除分类头部（classification head）
def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 将字典中的旧键（old）重命名为新键（new）
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 在一张可爱猫咪的图片上准备我们的结果验证
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 使用torch.no_grad()装饰器，将函数设置为无需梯度的上下文
@torch.no_grad()
def convert_vit_checkpoint(model_name, pytorch_dump_folder_path, base_model=True):
    """
    将模型的权重复制/粘贴/调整到我们的ViT结构中。
    """

    # 定义默认的ViT配置
    config = ViTConfig()
    
    # 如果模型名称的最后一个字符是"8"，则设置patch_size为8
    if model_name[-1] == "8":
        config.patch_size = 8
    
    # 如果不是基础模型，则设置num_labels为1000，并加载对应的标签文件
    if not base_model:
        config.num_labels = 1000
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    
    # 如果模型名称在指定的列表中，则设置ViT的隐藏层大小、中间层大小等
    if model_name in ["dino_vits8", "dino_vits16"]:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    
    # 从torch hub加载原始模型
    original_model = torch.hub.load("facebookresearch/dino:main", model_name)
    # 将原始模型设置为评估模式
    original_model.eval()

    # 加载原始模型的状态字典，并移除/重命名一些键
    state_dict = original_model.state_dict()
    if base_model:
        # 如果指定了基础模型，移除分类头部分的参数
        remove_classification_head_(state_dict)
    
    # 根据配置文件创建需要重命名的键列表
    rename_keys = create_rename_keys(config, base_model=base_model)
    
    # 遍历重命名键列表，逐一重命名状态字典中的键
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    
    # 根据状态字典和配置信息读入查询、键、值的数据
    read_in_q_k_v(state_dict, config, base_model)

    # 加载 HuggingFace 模型
    if base_model:
        # 如果指定了基础模型，创建 ViT 模型对象（不添加池化层）并设置为评估模式
        model = ViTModel(config, add_pooling_layer=False).eval()
    else:
        # 否则创建用于图像分类的 ViT 模型对象并设置为评估模式
        model = ViTForImageClassification(config).eval()
    
    # 加载模型的状态字典
    model.load_state_dict(state_dict)

    # 使用 ViTImageProcessor 准备图像并编码
    image_processor = ViTImageProcessor()
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    
    # 将图像数据输入模型并获取输出
    outputs = model(pixel_values)

    if base_model:
        # 如果指定了基础模型，还需要使用原始模型对图像进行预测
        final_hidden_state_cls_token = original_model(pixel_values)
        # 断言原始模型的分类标记的最终隐藏状态与当前模型输出的第一个位置的隐藏状态在给定的误差范围内相等
        assert torch.allclose(final_hidden_state_cls_token, outputs.last_hidden_state[:, 0, :], atol=1e-1)
    else:
        # 否则，直接使用原始模型获取分类 logits
        logits = original_model(pixel_values)
        # 断言原始模型输出的 logits 形状与当前模型输出的 logits 形状相等
        assert logits.shape == outputs.logits.shape
        # 断言两个 logits 张量在给定的误差范围内相等
        assert torch.allclose(logits, outputs.logits, atol=1e-3)

    # 创建存储 PyTorch 模型的文件夹（如果不存在）
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的消息
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的消息
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必填参数
    parser.add_argument(
        "--model_name",
        default="dino_vitb16",
        type=str,
        help="Name of the model trained with DINO you'd like to convert.",
    )
    # 模型名称，指定使用的 DINO 训练的模型名称，默认为 dino_vitb16

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # PyTorch 模型输出目录的路径，用于存储转换后的模型

    parser.add_argument(
        "--base_model",
        action="store_true",
        help="Whether to only convert the base model (no projection head weights).",
    )
    # 是否仅转换基础模型（不包括投影头权重）

    parser.set_defaults(base_model=True)
    # 设置默认参数，base_model 默认为 True

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 变量中

    convert_vit_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.base_model)
    # 调用 convert_vit_checkpoint 函数，传递解析后的参数 model_name、pytorch_dump_folder_path 和 base_model
```