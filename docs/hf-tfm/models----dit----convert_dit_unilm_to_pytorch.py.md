# `.\models\dit\convert_dit_unilm_to_pytorch.py`

```py
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
"""Convert DiT checkpoints from the unilm repository."""

# 导入必要的库
import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 数据的模块
from pathlib import Path  # 导入处理文件路径的模块

import requests  # 导入处理 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载模型的功能
from PIL import Image  # 导入处理图像的 PIL 库

# 导入转换相关的类和函数
from transformers import BeitConfig, BeitForImageClassification, BeitForMaskedImageModeling, BeitImageProcessor
from transformers.image_utils import PILImageResampling  # 导入图像处理相关的功能
from transformers.utils import logging  # 导入日志记录功能

# 设置日志输出级别为信息级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# here we list all keys to be renamed (original name on the left, our name on the right)
# 定义函数用于创建需要重命名的键值对列表
def create_rename_keys(config, has_lm_head=False, is_semantic=False):
    # 如果是语义模型，则前缀为"backbone."，否则为空
    prefix = "backbone." if is_semantic else ""

    rename_keys = []
    # 遍历模型的隐藏层，生成需要重命名的键值对
    for i in range(config.num_hidden_layers):
        # 第一层归一化层的权重和偏置
        rename_keys.append((f"{prefix}blocks.{i}.norm1.weight", f"beit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm1.bias", f"beit.encoder.layer.{i}.layernorm_before.bias"))
        # 注意力机制输出的权重和偏置
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.weight", f"beit.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.bias", f"beit.encoder.layer.{i}.attention.output.dense.bias")
        )
        # 第二层归一化层的权重和偏置
        rename_keys.append((f"{prefix}blocks.{i}.norm2.weight", f"beit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm2.bias", f"beit.encoder.layer.{i}.layernorm_after.bias"))
        # 中间层的全连接层1的权重和偏置
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.weight", f"beit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.bias", f"beit.encoder.layer.{i}.intermediate.dense.bias"))
        # 中间层的全连接层2的权重和偏置
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.weight", f"beit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.bias", f"beit.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    # 扩展 rename_keys 列表，添加模型参数重命名对应关系
    rename_keys.extend(
        [
            (f"{prefix}cls_token", "beit.embeddings.cls_token"),  # 将 "{prefix}cls_token" 重命名为 "beit.embeddings.cls_token"
            (f"{prefix}patch_embed.proj.weight", "beit.embeddings.patch_embeddings.projection.weight"),  # 将 "{prefix}patch_embed.proj.weight" 重命名为 "beit.embeddings.patch_embeddings.projection.weight"
            (f"{prefix}patch_embed.proj.bias", "beit.embeddings.patch_embeddings.projection.bias"),  # 将 "{prefix}patch_embed.proj.bias" 重命名为 "beit.embeddings.patch_embeddings.projection.bias"
            (f"{prefix}pos_embed", "beit.embeddings.position_embeddings"),  # 将 "{prefix}pos_embed" 重命名为 "beit.embeddings.position_embeddings"
        ]
    )

    if has_lm_head:
        # 如果模型具有语言模型头部，则继续添加重命名对应关系
        rename_keys.extend(
            [
                ("mask_token", "beit.embeddings.mask_token"),  # 将 "mask_token" 重命名为 "beit.embeddings.mask_token"
                ("norm.weight", "layernorm.weight"),  # 将 "norm.weight" 重命名为 "layernorm.weight"
                ("norm.bias", "layernorm.bias"),  # 将 "norm.bias" 重命名为 "layernorm.bias"
            ]
        )
    else:
        # 如果模型没有语言模型头部，则添加分类头部的重命名对应关系
        rename_keys.extend(
            [
                ("fc_norm.weight", "beit.pooler.layernorm.weight"),  # 将 "fc_norm.weight" 重命名为 "beit.pooler.layernorm.weight"
                ("fc_norm.bias", "beit.pooler.layernorm.bias"),  # 将 "fc_norm.bias" 重命名为 "beit.pooler.layernorm.bias"
                ("head.weight", "classifier.weight"),  # 将 "head.weight" 重命名为 "classifier.weight"
                ("head.bias", "classifier.bias"),  # 将 "head.bias" 重命名为 "classifier.bias"
            ]
        )

    return rename_keys
# 将每个编码器层的权重矩阵分解为查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False):
    # 遍历编码器层的数量
    for i in range(config.num_hidden_layers):
        # 如果是语义模型，添加前缀 "backbone."
        prefix = "backbone." if is_semantic else ""

        # 获取查询、键和值的权重矩阵
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        # 获取查询的偏置
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        # 获取值的偏置
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")

        # 将查询权重放入新的键值对中
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        # 将查询偏置放入新的键值对中
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        # 将键的权重放入新的键值对中
        state_dict[f"beit.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        # 将值的权重放入新的键值对中
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        # 将值的偏置放入新的键值对中
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # 获取 gamma_1 和 gamma_2 的值
        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")

        # 将 gamma_1 重命名为 lambda_1，并放入新的键值对中
        state_dict[f"beit.encoder.layer.{i}.lambda_1"] = gamma_1
        # 将 gamma_2 重命名为 lambda_2，并放入新的键值对中
        state_dict[f"beit.encoder.layer.{i}.lambda_2"] = gamma_2


# 重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键的值
    val = dct.pop(old)
    # 将值用新键重新放入字典中
    dct[new] = val


# 准备图片数据，在线获取一张可爱猫咪的图片
def prepare_img():
    # 图片的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 URL 获取图片对象
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回获取的图片对象
    return im


# 使用无梯度计算的上下文环境，将某个检查点文件的权重转换到我们的 BEiT 结构中
@torch.no_grad()
def convert_dit_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our BEiT structure.
    """

    # 定义默认的 BEiT 配置
    # 根据检查点 URL 判断是否有语言模型头部
    has_lm_head = False if "rvlcdip" in checkpoint_url else True
    # 根据是否使用绝对位置嵌入和是否有语言模型头部来配置 BEiT
    config = BeitConfig(use_absolute_position_embeddings=True, use_mask_token=has_lm_head)

    # 根据检查点 URL 中是否包含 "large" 或 "dit-l" 来配置 BEiT 的架构大小
    if "large" in checkpoint_url or "dit-l" in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16

    # 如果检查点 URL 中包含 "rvlcdip"，配置 BEiT 的标签相关信息
    if "rvlcdip" in checkpoint_url:
        config.num_labels = 16
        # 设置用于加载 id2label 映射的存储库和文件名
        repo_id = "huggingface/label-files"
        filename = "rvlcdip-id2label.json"
        # 通过 Hugging Face Hub 下载并加载 id2label 映射
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # 加载原始模型的 state_dict，并移除和重命名一些键
    # 从指定的 URL 加载模型的状态字典，并选择在 CPU 上加载
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    # 根据配置创建重命名键列表，用于在加载状态字典时重命名模型参数
    rename_keys = create_rename_keys(config, has_lm_head=has_lm_head)
    for src, dest in rename_keys:
        # 重命名状态字典中的键名，根据预定义的映射关系进行修改
        rename_key(state_dict, src, dest)
    
    # 根据状态字典读取并初始化 Q、K、V（查询、键、值）的权重
    read_in_q_k_v(state_dict, config, has_lm_head=has_lm_head)

    # 根据是否有语言模型头部选择加载不同类型的 Beit 模型
    model = BeitForMaskedImageModeling(config) if has_lm_head else BeitForImageClassification(config)
    # 设置模型为评估模式
    model.eval()
    # 加载预训练模型的状态字典
    model.load_state_dict(state_dict)

    # 创建 Beit 图像处理器，用于预处理图像
    image_processor = BeitImageProcessor(
        size=config.image_size, resample=PILImageResampling.BILINEAR, do_center_crop=False
    )
    # 准备图像数据
    image = prepare_img()

    # 使用图像处理器对图像进行编码，并返回 PyTorch 张量表示
    encoding = image_processor(images=image, return_tensors="pt")
    # 提取像素值张量
    pixel_values = encoding["pixel_values"]

    # 使用加载的模型进行图像处理，获取输出结果
    outputs = model(pixel_values)
    # 提取模型的预测 logits（对数概率）
    logits = outputs.logits

    # 验证 logits 的形状是否符合预期
    expected_shape = [1, 16] if "rvlcdip" in checkpoint_url else [1, 196, 8192]
    assert logits.shape == torch.Size(expected_shape), "Shape of logits not as expected"

    # 确保保存模型的文件夹存在，如果不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印信息，指示正在保存模型到指定路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印信息，指示正在保存图像处理器到指定路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到模型 hub
    if push_to_hub:
        # 根据模型的具体配置选择模型名称
        if has_lm_head:
            model_name = "dit-base" if "base" in checkpoint_url else "dit-large"
        else:
            model_name = "dit-base-finetuned-rvlcdip" if "dit-b" in checkpoint_url else "dit-large-finetuned-rvlcdip"
        
        # 将图像处理器推送到指定的 hub 仓库
        image_processor.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add image processor",
            use_temp_dir=True,
        )
        # 将模型推送到指定的 hub 仓库
        model.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add model",
            use_temp_dir=True,
        )
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    # 添加名为--checkpoint_url的命令行参数，用于指定原始PyTorch检查点文件的URL

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    # 添加名为--pytorch_dump_folder_path的命令行参数，用于指定输出PyTorch模型的文件夹路径

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    # 添加名为--push_to_hub的命令行参数，如果设置则将其设为True

    args = parser.parse_args()
    # 解析命令行参数并将其存储在args变量中

    convert_dit_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用convert_dit_checkpoint函数，传递解析后的命令行参数作为参数
```