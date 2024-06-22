# `.\transformers\models\beit\convert_beit_unilm_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert BEiT checkpoints from the unilm repository."""


import argparse  # 导入用于解析命令行参数的模块
import json  # 导入用于处理 JSON 格式数据的模块
from pathlib import Path  # 导入用于处理文件路径的模块

import requests  # 导入用于发送 HTTP 请求的模块
import torch  # 导入 PyTorch 库
from datasets import load_dataset  # 导入用于加载数据集的函数
from huggingface_hub import hf_hub_download  # 导入用于从 Hugging Face Hub 下载模型的函数
from PIL import Image  # 导入 Python Imaging Library (PIL) 的 Image 模块

from transformers import (  # 导入转换器库的相关模块和类
    BeitConfig,  # 导入 BEiT 模型的配置类
    BeitForImageClassification,  # 导入用于图像分类的 BEiT 模型类
    BeitForMaskedImageModeling,  # 导入用于图像填充遮挡的 BEiT 模型类
    BeitForSemanticSegmentation,  # 导入用于语义分割的 BEiT 模型类
    BeitImageProcessor,  # 导入用于处理图像输入的 BEiT 图像处理器类
)
from transformers.image_utils import PILImageResampling  # 导入用于图像重采样的 PIL 图像重采样函数
from transformers.utils import logging  # 导入转换器库的日志模块


logging.set_verbosity_info()  # 设置日志级别为 info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


# here we list all keys to be renamed (original name on the left, our name on the right)
# 这里列出了所有需要重命名的键（原始名称在左侧，我们的名称在右侧）
def create_rename_keys(config, has_lm_head=False, is_semantic=False):
    # 如果是语义分割模型，则设置前缀为 "backbone."，否则为空字符串
    prefix = "backbone." if is_semantic else ""

    rename_keys = []  # 存储重命名键值对的列表
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        # 编码器层：输出投影、两个前馈神经网络和两个层归一化操作
        rename_keys.append((f"{prefix}blocks.{i}.norm1.weight", f"beit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm1.bias", f"beit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.weight", f"beit.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.bias", f"beit.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append((f"{prefix}blocks.{i}.norm2.weight", f"beit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm2.bias", f"beit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.weight", f"beit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.bias", f"beit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.weight", f"beit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.bias", f"beit.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    # 扩展重命名键列表，将原始键和新键对添加到列表中
    rename_keys.extend(
        [
            (f"{prefix}cls_token", "beit.embeddings.cls_token"),
            (f"{prefix}patch_embed.proj.weight", "beit.embeddings.patch_embeddings.projection.weight"),
            (f"{prefix}patch_embed.proj.bias", "beit.embeddings.patch_embeddings.projection.bias"),
        ]
    )

    if has_lm_head:
        # 如果模型有语言模型头部
        # 添加额外的重命名键，包括 mask token、共享的相对位置偏置和 layernorm
        rename_keys.extend(
            [
                ("mask_token", "beit.embeddings.mask_token"),
                (
                    "rel_pos_bias.relative_position_bias_table",
                    "beit.encoder.relative_position_bias.relative_position_bias_table",
                ),
                (
                    "rel_pos_bias.relative_position_index",
                    "beit.encoder.relative_position_bias.relative_position_index",
                ),
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )
    elif is_semantic:
        # 如果是语义分割模型
        # 添加额外的重命名键，包括解码头部和辅助头部的权重和偏置
        rename_keys.extend(
            [
                ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
                ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
                ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
                ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
            ]
        )
    else:
        # 如果不是语言模型头部也不是语义分割模型
        # 添加额外的重命名键，包括 layernorm 和分类头部的权重和偏置
        rename_keys.extend(
            [
                ("fc_norm.weight", "beit.pooler.layernorm.weight"),
                ("fc_norm.bias", "beit.pooler.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    # 返回所有重命名键的列表
    return rename_keys
# 将每个编码器层的矩阵拆分为查询、键和值
def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False):
    # 循环遍历编码器层数量
    for i in range(config.num_hidden_layers):
        # 如果是语义编码器，则添加前缀 "backbone."，否则为空字符串
        prefix = "backbone." if is_semantic else ""
        # 获取查询、键和值
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")

        # 更新状态字典中的键和值
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"beit.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # 更新状态字典中的 gamma_1 和 gamma_2
        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")
        state_dict[f"beit.encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"beit.encoder.layer.{i}.lambda_2"] = gamma_2

        # 如果没有语言模型头部，则处理相对位置偏置表和索引
        if not has_lm_head:
            # 每个编码器层都有自己的相对位置偏置表和索引
            table = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_bias_table")
            index = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_index")

            # 更新状态字典中的相对位置偏置表和索引
            state_dict[
                f"beit.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"
            ] = table
            state_dict[
                f"beit.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"
            ] = index


# 重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键，并将其对应的值赋给新键
    val = dct.pop(old)
    dct[new] = val


# 在一张可爱猫咪的图片上验证我们的结果
def prepare_img():
    # 图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 打开并返回图片对象
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 在没有梯度的情况下执行函数
@torch.no_grad()
def convert_beit_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our BEiT structure.
    """

    # 定义默认的 BEiT 配置
    config = BeitConfig()
    has_lm_head = False
    is_semantic = False
    repo_id = "huggingface/label-files"
    # 根据 URL 设置配置参数
    if checkpoint_url[-9:-4] == "pt22k":
        # 如果是掩码图像建模，则设置共享的相对位置偏置和使用掩码标记
        config.use_shared_relative_position_bias = True
        config.use_mask_token = True
        has_lm_head = True
    elif checkpoint_url[-9:-4] == "ft22k":
        # 如果模型是在ImageNet-22k上进行中间微调
        # 设定使用相对位置偏置
        config.use_relative_position_bias = True
        # 设置模型输出的类别数为21841
        config.num_labels = 21841
        # 定义ImageNet-22k类别到标签的映射文件名
        filename = "imagenet-22k-id2label.json"
        # 从Hub下载ImageNet-22k类别到标签的映射文件，并加载
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        # 将类别到标签的映射转换为整型
        id2label = {int(k): v for k, v in id2label.items()}
        # 删除不需要的类别，详情见https://github.com/google-research/big_transfer/issues/18
        del id2label[9205]
        del id2label[15027]
        # 设置模型的类别到标签映射和标签到类别映射
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    elif checkpoint_url[-8:-4] == "to1k":
        # 如果模型是在ImageNet-1k上进行微调
        # 设定使用相对位置偏置
        config.use_relative_position_bias = True
        # 设置模型输出的类别数为1000
        config.num_labels = 1000
        # 定义ImageNet-1k类别到标签的映射文件名
        filename = "imagenet-1k-id2label.json"
        # 从Hub下载ImageNet-1k类别到标签的映射文件，并加载
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        # 将类别到标签的映射转换为整型
        id2label = {int(k): v for k, v in id2label.items()}
        # 设置模型的类别到标签映射和标签到类别映射
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        # 根据URL设定图像尺寸
        if "384" in checkpoint_url:
            config.image_size = 384
        if "512" in checkpoint_url:
            config.image_size = 512
    elif "ade20k" in checkpoint_url:
        # 如果模型是在ADE20k上进行微调
        # 设定使用相对位置偏置
        config.use_relative_position_bias = True
        # 设置模型输出的类别数为150
        config.num_labels = 150
        # 定义ADE20k类别到标签的映射文件名
        filename = "ade20k-id2label.json"
        # 从Hub下载ADE20k类别到标签的映射文件，并加载
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        # 将类别到标签的映射转换为整型
        id2label = {int(k): v for k, v in id2label.items()}
        # 设置模型的类别到标签映射和标签到类别映射
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        # 设置图像尺寸为640
        config.image_size = 640
        # 设定为语义分割任务
        is_semantic = True
    else:
        # 抛出异常，不支持的模型
        raise ValueError("Checkpoint not supported, URL should either end with 'pt22k', 'ft22k', 'to1k' or 'ade20k'")

    # 设定模型的规模
    if "base" in checkpoint_url:
        # 如果URL中包含"base"，则模型规模为基础规模，不做修改
        pass
    elif "large" in checkpoint_url:
        # 如果URL中包含"large"，则设定模型的隐藏层大小、中间层大小、隐藏层数和注意力头数
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        # 如果是在ADE20k上微调，则设定特定的图像尺寸和输出层索引
        if "ade20k" in checkpoint_url:
            config.image_size = 640
            config.out_indices = [7, 11, 15, 23]
    else:
        # 抛出异常，URL中应包含"base"或"large"
        raise ValueError("Should either find 'base' or 'large' in checkpoint URL")

    # 加载原始模型的状态字典，并删除和重命名一些键
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)
    # 如果不是在ADE20k上微调，则只加载模型部分
    state_dict = state_dict["model"] if "ade20k" not in checkpoint_url else state_dict["state_dict"]

    # 创建重命名键列表
    rename_keys = create_rename_keys(config, has_lm_head=has_lm_head, is_semantic=is_semantic)
    # 遍历重命名键列表，执行重命名操作
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取Q、K、V矩阵
    read_in_q_k_v(state_dict, config, has_lm_head=has_lm_head, is_semantic=is_semantic)
    # 如果是语义分割任务
    if is_semantic:
        # 给解码器的键添加前缀
        for key, val in state_dict.copy().items():
            # 复制键值对并弹出原字典中的值
            val = state_dict.pop(key)
            # 如果键以"backbone.fpn"开头，则替换为"fpn"
            if key.startswith("backbone.fpn"):
                key = key.replace("backbone.fpn", "fpn")
            # 将处理后的键值对重新加入字典
            state_dict[key] = val

    # 加载 HuggingFace 模型
    if checkpoint_url[-9:-4] == "pt22k":
        # 根据不同的 checkpoint_url 类型选择对应的 BEiT 模型
        model = BeitForMaskedImageModeling(config)
    elif "ade20k" in checkpoint_url:
        model = BeitForSemanticSegmentation(config)
    else:
        model = BeitForImageClassification(config)
    # 将模型设置为评估模式
    model.eval()
    # 加载模型参数
    model.load_state_dict(state_dict)

    # 在图像上检查输出
    if is_semantic:
        # 对语义分割任务进行图像处理
        image_processor = BeitImageProcessor(size=config.image_size, do_center_crop=False)
        # 加载用于测试的数据集并打开第一张图像
        ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
        image = Image.open(ds[0]["file"])
    else:
        # 对图像分类任务进行图像处理
        image_processor = BeitImageProcessor(
            size=config.image_size, resample=PILImageResampling.BILINEAR, do_center_crop=False
        )
        # 准备图像
        image = prepare_img()

    # 对图像进行编码处理
    encoding = image_processor(images=image, return_tensors="pt")
    # 获取像素值
    pixel_values = encoding["pixel_values"]

    # 获取模型输出
    outputs = model(pixel_values)
    # 获取模型输出的 logits

    logits = outputs.logits

    # 验证 logits 的形状
    expected_shape = torch.Size([1, 1000])
    if checkpoint_url[:-4].endswith("beit_base_patch16_224_pt22k"):
        expected_shape = torch.Size([1, 196, 8192])
    elif checkpoint_url[:-4].endswith("beit_large_patch16_224_pt22k"):
        expected_shape = torch.Size([1, 196, 8192])
    elif checkpoint_url[:-4].endswith("beit_base_patch16_224_pt22k_ft22k"):
        expected_shape = torch.Size([1, 21841])
        expected_logits = torch.tensor([2.2288, 2.4671, 0.7395])
        expected_class_idx = 2397
    elif checkpoint_url[:-4].endswith("beit_large_patch16_224_pt22k_ft22k"):
        expected_shape = torch.Size([1, 21841])
        expected_logits = torch.tensor([1.6881, -0.2787, 0.5901])
        expected_class_idx = 2396
    elif checkpoint_url[:-4].endswith("beit_base_patch16_224_pt22k_ft1k"):
        expected_logits = torch.tensor([0.1241, 0.0798, -0.6569])
        expected_class_idx = 285
    elif checkpoint_url[:-4].endswith("beit_base_patch16_224_pt22k_ft22kto1k"):
        expected_logits = torch.tensor([-1.2385, -1.0987, -1.0108])
        expected_class_idx = 281
    elif checkpoint_url[:-4].endswith("beit_base_patch16_384_pt22k_ft22kto1k"):
        expected_logits = torch.tensor([-1.5303, -0.9484, -0.3147])
        expected_class_idx = 761
    elif checkpoint_url[:-4].endswith("beit_large_patch16_224_pt22k_ft1k"):
        expected_logits = torch.tensor([0.4610, -0.0928, 0.2086])
        expected_class_idx = 761
    elif checkpoint_url[:-4].endswith("beit_large_patch16_224_pt22k_ft22kto1k"):
        expected_logits = torch.tensor([-0.4804, 0.6257, -0.1837])
        expected_class_idx = 761
    # 如果模型的检查点 URL 以 "beit_large_patch16_384_pt22k_ft22kto1k" 结尾
    elif checkpoint_url[:-4].endswith("beit_large_patch16_384_pt22k_ft22kto1k"):
        # 设置预期的 logits
        expected_logits = torch.tensor([[-0.5122, 0.5117, -0.2113]])
        # 设置预期的类别索引
        expected_class_idx = 761
    # 如果模型的检查点 URL 以 "beit_large_patch16_512_pt22k_ft22kto1k" 结尾
    elif checkpoint_url[:-4].endswith("beit_large_patch16_512_pt22k_ft22kto1k"):
        # 设置预期的 logits
        expected_logits = torch.tensor([-0.3062, 0.7261, 0.4852])
        # 设置预期的类别索引
        expected_class_idx = 761
    # 如果模型的检查点 URL 以 "beit_base_patch16_640_pt22k_ft22ktoade20k" 结尾
    elif checkpoint_url[:-4].endswith("beit_base_patch16_640_pt22k_ft22ktoade20k"):
        # 设置预期的形状
        expected_shape = (1, 150, 160, 160)
        # 设置预期的 logits
        expected_logits = torch.tensor(
            [
                [[-4.9225, -2.3954, -3.0522], [-2.8822, -1.0046, -1.7561], [-2.9549, -1.3228, -2.1347]],
                [[-5.8168, -3.4129, -4.0778], [-3.8651, -2.2214, -3.0277], [-3.8356, -2.4643, -3.3535]],
                [[-0.0078, 3.9952, 4.0754], [2.9856, 4.6944, 5.0035], [3.2413, 4.7813, 4.9969]],
            ]
        )
    # 如果模型的检查点 URL 以 "beit_large_patch16_640_pt22k_ft22ktoade20k" 结尾
    elif checkpoint_url[:-4].endswith("beit_large_patch16_640_pt22k_ft22ktoade20k"):
        # 设置预期的形状
        expected_shape = (1, 150, 160, 160)
        # 设置预期的 logits
        expected_logits = torch.tensor(
            [
                [[-4.3305, -2.3049, -3.0161], [-2.9591, -1.5305, -2.2251], [-3.4198, -1.8004, -2.9062]],
                [[-5.8922, -3.7435, -4.3978], [-4.2063, -2.7872, -3.4755], [-4.2791, -3.1874, -4.1681]],
                [[0.9895, 4.3467, 4.7663], [4.2476, 5.6830, 6.1518], [4.5550, 6.2495, 6.5154]],
            ]
        )
    else:
        # 如果模型不受支持，则引发 ValueError
        raise ValueError("Can't verify logits as model is not supported")

    # 检查 logits 的形状是否符合预期
    if logits.shape != expected_shape:
        raise ValueError(f"Shape of logits not as expected. {logits.shape=}, {expected_shape=}")
    # 如果没有 lm_head
    if not has_lm_head:
        # 如果是语义模型
        if is_semantic:
            # 检查 logits 的前几个元素是否与预期的 logits 接近
            if not torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=1e-3):
                raise ValueError("First elements of logits not as expected")
        else:
            # 打印预测的类别索引
            print("Predicted class idx:", logits.argmax(-1).item())
            # 检查 logits 的前几个元素是否与预期的 logits 接近
            if not torch.allclose(logits[0, :3], expected_logits, atol=1e-3):
                raise ValueError("First elements of logits not as expected")
            # 检查预测的类别索引是否与预期的类别索引相同
            if logits.argmax(-1).item() != expected_class_idx:
                raise ValueError("Predicted class index not as expected")

    # 创建目录以保存 PyTorch 模型
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 保存图像处理器到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数：--checkpoint_url，用于指定原始 PyTorch 检查点文件的 URL
    parser.add_argument(
        "--checkpoint_url",
        default="https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    
    # 添加命令行参数：--pytorch_dump_folder_path，用于指定输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 convert_beit_checkpoint 函数，传入原始检查点文件的 URL 和输出模型文件夹路径
    convert_beit_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
```  
```