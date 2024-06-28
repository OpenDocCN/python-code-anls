# `.\models\upernet\convert_swin_upernet_to_pytorch.py`

```py
# coding=utf-8
# 设置脚本编码格式为UTF-8

# Copyright 2022 The HuggingFace Inc. team.
# 版权声明，版权归HuggingFace Inc.团队所有。

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache License 2.0许可证授权使用本代码

# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则依据“原样”分发此软件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证了解特定语言的权限和限制

"""Convert Swin Transformer + UperNet checkpoints from mmsegmentation.

从mmsegmentation转换Swin Transformer + UperNet检查点。

URL: https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin
"""

import argparse  # 导入命令行参数解析模块
import json  # 导入JSON操作模块

import requests  # 导入HTTP请求库
import torch  # 导入PyTorch深度学习框架
from huggingface_hub import hf_hub_download  # 从HuggingFace Hub下载模块导入函数
from PIL import Image  # 导入Python Imaging Library (PIL)中的Image模块

from transformers import SegformerImageProcessor, SwinConfig, UperNetConfig, UperNetForSemanticSegmentation  # 导入transformers库中的类和函数


def get_upernet_config(model_name):
    # 根据模型名称获取相应的UperNet配置

    auxiliary_in_channels = 384  # 设置辅助输入通道数
    window_size = 7  # 设置窗口大小初始值
    if "tiny" in model_name:
        embed_dim = 96  # 设置嵌入维度大小
        depths = (2, 2, 6, 2)  # 设置深度
        num_heads = (3, 6, 12, 24)  # 设置头数
    elif "small" in model_name:
        embed_dim = 96
        depths = (2, 2, 18, 2)
        num_heads = (3, 6, 12, 24)
    elif "base" in model_name:
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
        window_size = 12
        auxiliary_in_channels = 512
    elif "large" in model_name:
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)
        window_size = 12
        auxiliary_in_channels = 768

    # 设置标签信息
    num_labels = 150
    repo_id = "huggingface/label-files"
    filename = "ade20k-id2label.json"

    # 从HuggingFace Hub下载标签文件，并加载为JSON格式
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}  # 转换为整数键的字典
    label2id = {v: k for k, v in id2label.items()}  # 反转为值到整数键的字典

    # 创建Swin Transformer的配置
    backbone_config = SwinConfig(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )

    # 创建UperNet的配置
    config = UperNetConfig(
        backbone_config=backbone_config,
        auxiliary_in_channels=auxiliary_in_channels,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
# 列出需要重命名的所有键对（原始名称在左侧，我们的名称在右侧）
def create_rename_keys(config):
    rename_keys = []  # 初始化空的重命名键列表

    # fmt: off
    # stem
    # fmt: on

    # 添加需要重命名的键对到列表中
    rename_keys.append(("backbone.patch_embed.projection.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.projection.bias", "backbone.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.patch_embed.norm.weight", "backbone.embeddings.norm.weight"))
    # 将特定键值对添加到 rename_keys 列表中，用于后续的键名重命名
    rename_keys.append(("backbone.patch_embed.norm.bias", "backbone.embeddings.norm.bias"))

    # 遍历 backbone_config.depths 中的每个深度值
    for i in range(len(config.backbone_config.depths)):
        # 遍历每个深度下的层数量
        for j in range(config.backbone_config.depths[i]):
            # 将 backbone.stages.i.blocks.j.norm1.weight 的键重命名为 backbone.encoder.layers.i.blocks.j.layernorm_before.weight
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.weight"))
            # 将 backbone.stages.i.blocks.j.norm1.bias 的键重命名为 backbone.encoder.layers.i.blocks.j.layernorm_before.bias
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.bias"))
            # 将 backbone.stages.i.blocks.j.attn.w_msa.relative_position_bias_table 的键重命名为 backbone.encoder.layers.i.blocks.j.attention.self.relative_position_bias_table
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.relative_position_bias_table", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_bias_table"))
            # 将 backbone.stages.i.blocks.j.attn.w_msa.relative_position_index 的键重命名为 backbone.encoder.layers.i.blocks.j.attention.self.relative_position_index
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.relative_position_index", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_index"))
            # 将 backbone.stages.i.blocks.j.attn.w_msa.proj.weight 的键重命名为 backbone.encoder.layers.i.blocks.j.attention.output.dense.weight
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.proj.weight", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.weight"))
            # 将 backbone.stages.i.blocks.j.attn.w_msa.proj.bias 的键重命名为 backbone.encoder.layers.i.blocks.j.attention.output.dense.bias
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.proj.bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.bias"))
            # 将 backbone.stages.i.blocks.j.norm2.weight 的键重命名为 backbone.encoder.layers.i.blocks.j.layernorm_after.weight
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm2.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.weight"))
            # 将 backbone.stages.i.blocks.j.norm2.bias 的键重命名为 backbone.encoder.layers.i.blocks.j.layernorm_after.bias
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm2.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.bias"))
            # 将 backbone.stages.i.blocks.j.ffn.layers.0.0.weight 的键重命名为 backbone.encoder.layers.i.blocks.j.intermediate.dense.weight
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.0.0.weight", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.weight"))
            # 将 backbone.stages.i.blocks.j.ffn.layers.0.0.bias 的键重命名为 backbone.encoder.layers.i.blocks.j.intermediate.dense.bias
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.0.0.bias", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.bias"))
            # 将 backbone.stages.i.blocks.j.ffn.layers.1.weight 的键重命名为 backbone.encoder.layers.i.blocks.j.output.dense.weight
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.weight"))
            # 将 backbone.stages.i.blocks.j.ffn.layers.1.bias 的键重命名为 backbone.encoder.layers.i.blocks.j.output.dense.bias
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.bias"))

        # 如果 i 小于 3，则继续添加下述重命名键值对
        if i < 3:
            # 将 backbone.stages.i.downsample.reduction.weight 的键重命名为 backbone.encoder.layers.i.downsample.reduction.weight
            rename_keys.append((f"backbone.stages.{i}.downsample.reduction.weight", f"backbone.encoder.layers.{i}.downsample.reduction.weight"))
            # 将 backbone.stages.i.downsample.norm.weight 的键重命名为 backbone.encoder.layers.i.downsample.norm.weight
            rename_keys.append((f"backbone.stages.{i}.downsample.norm.weight", f"backbone.encoder.layers.{i}.downsample.norm.weight"))
            # 将 backbone.stages.i.downsample.norm.bias 的键重命名为 backbone.encoder.layers.i.downsample.norm.bias
            rename_keys.append((f"backbone.stages.{i}.downsample.norm.bias", f"backbone.encoder.layers.{i}.downsample.norm.bias"))
        
        # 将 backbone.norm{i}.weight 的键重命名为 backbone.hidden_states_norms.stage{i+1}.weight
        rename_keys.append((f"backbone.norm{i}.weight", f"backbone.hidden_states_norms.stage{i+1}.weight"))
        # 将 backbone.norm{i}.bias 的键重命名为 backbone.hidden_states_norms.stage{i+1}.bias
        rename_keys.append((f"backbone.norm{i}.bias", f"backbone.hidden_states_norms.stage{i+1}.bias"))

    # decode head
    rename_keys.extend(
        [
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )
    # fmt: on

    return rename_keys



    # 将以下四对键值对添加到 `rename_keys` 列表中
    rename_keys.extend(
        [
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )
    # 标记格式化的结束点，这里是 `fmt: on`

    # 返回已经更新的 `rename_keys` 列表
    return rename_keys
# 重命名字典中的键。
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将值与新键关联存入字典
    dct[new] = val


# 将每个编码器层的矩阵拆分为查询、键和值。
def read_in_q_k_v(state_dict, backbone_config):
    # 计算每个特征维度的大小
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    # 遍历不同深度和层级的编码器
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            # fmt: off
            # 读取输入投影层权重和偏置（在原始实现中，这是一个单独的矩阵加偏置）
            in_proj_weight = state_dict.pop(f"backbone.stages.{i}.blocks.{j}.attn.w_msa.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.stages.{i}.blocks.{j}.attn.w_msa.qkv.bias")
            # 按顺序将查询、键和值添加到状态字典中
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[
                dim : dim * 2
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim :, :
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]
            # fmt: on


# 修正通过unfold操作导致的张量重排顺序
def correct_unfold_reduction_order(x):
    # 获取输出通道数和输入通道数
    out_channel, in_channel = x.shape
    # 重塑张量形状以便重新排列
    x = x.reshape(out_channel, 4, in_channel // 4)
    x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
    return x


# 逆向修正unfold操作导致的张量重排顺序
def reverse_correct_unfold_reduction_order(x):
    # 获取输出通道数和输入通道数
    out_channel, in_channel = x.shape
    # 重塑张量形状以便逆向重排
    x = x.reshape(out_channel, in_channel // 4, 4)
    x = x[:, :, [0, 2, 1, 3]].transpose(1, 2).reshape(out_channel, in_channel)
    return x


# 修正标准化操作导致的张量重排顺序
def correct_unfold_norm_order(x):
    # 获取输入通道数
    in_channel = x.shape[0]
    # 重塑张量形状以便重新排列
    x = x.reshape(4, in_channel // 4)
    x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
    return x


# 逆向修正标准化操作导致的张量重排顺序
def reverse_correct_unfold_norm_order(x):
    # 获取输入通道数
    in_channel = x.shape[0]
    # 重塑张量形状以便逆向重排
    x = x.reshape(in_channel // 4, 4)
    x = x[:, [0, 2, 1, 3]].transpose(0, 1).reshape(in_channel)
    return x


# 在这个版本中，由于使用了nn.Unfold实现的新的下采样操作，出现了不兼容性。
# 问题已在以下链接中得到解决：https://github.com/open-mmlab/mmdetection/blob/31c84958f54287a8be2b99cbf87a6dcf12e57753/mmdet/models/utils/ckpt_convert.py#L96。
def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    pass
    # 定义模型名称到预训练模型权重 URL 的映射字典
    model_name_to_url = {
        "upernet-swin-tiny": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth",
        "upernet-swin-small": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth",
        "upernet-swin-base": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth",
        "upernet-swin-large": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k_20220318_091743-9ba68901.pth",
    }
    
    # 获取指定模型名称对应的预训练模型权重 URL
    checkpoint_url = model_name_to_url[model_name]
    
    # 使用 torch.hub 加载预训练模型的状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", file_name=model_name)["state_dict"]

    # 打印加载的每个参数名及其形状
    for name, param in state_dict.items():
        print(name, param.shape)

    # 根据模型名称获取对应的配置信息
    config = get_upernet_config(model_name)
    
    # 使用获取的配置信息创建 UperNetForSemanticSegmentation 模型
    model = UperNetForSemanticSegmentation(config)
    
    # 将模型设置为评估模式
    model.eval()

    # 替换状态字典中的键名中的 "bn" 为 "batch_norm"
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        state_dict[key] = val

    # 根据预定义的键名重命名状态字典中的键名
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    
    # 从配置中读取相关的 QKV（Query, Key, Value）信息到状态字典中
    read_in_q_k_v(state_dict, config.backbone_config)

    # 修正状态字典中 "downsample" 相关参数
    for key, value in state_dict.items():
        if "downsample" in key:
            if "reduction" in key:
                state_dict[key] = reverse_correct_unfold_reduction_order(value)
            if "norm" in key:
                state_dict[key] = reverse_correct_unfold_norm_order(value)

    # 加载修正后的状态字典到模型中
    model.load_state_dict(state_dict)

    # 在指定的图像 URL 上验证模型输出
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # 创建 SegformerImageProcessor 实例并处理图像获取像素值
    processor = SegformerImageProcessor()
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 禁用梯度计算环境下执行模型推理
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    # 打印 logits 的形状和其前3x3的值
    print(logits.shape)
    print("First values of logits:", logits[0, 0, :3, :3])
    
    # 如果模型名称为 "upernet-swin-tiny"，则进行断言验证
    if model_name == "upernet-swin-tiny":
        expected_slice = torch.tensor(
            [[-7.5958, -7.5958, -7.4302], [-7.5958, -7.5958, -7.4302], [-7.4797, -7.4797, -7.3068]]
        )
    elif model_name == "upernet-swin-small":
        # 如果模型名称是 "upernet-swin-small"
        expected_slice = torch.tensor(
            [[-7.1921, -7.1921, -6.9532], [-7.1921, -7.1921, -6.9532], [-7.0908, -7.0908, -6.8534]]
        )
    elif model_name == "upernet-swin-base":
        # 如果模型名称是 "upernet-swin-base"
        expected_slice = torch.tensor(
            [[-6.5851, -6.5851, -6.4330], [-6.5851, -6.5851, -6.4330], [-6.4763, -6.4763, -6.3254]]
        )
    elif model_name == "upernet-swin-large":
        # 如果模型名称是 "upernet-swin-large"
        expected_slice = torch.tensor(
            [[-7.5297, -7.5297, -7.3802], [-7.5297, -7.5297, -7.3802], [-7.4044, -7.4044, -7.2586]]
        )
    # 打印模型输出的前 3x3 的 logits
    print("Logits:", outputs.logits[0, 0, :3, :3])
    # 使用 torch.allclose 检查输出的 logits 是否与预期的片段相近
    assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4)
    # 打印确认信息
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        # 如果指定了 pytorch_dump_folder_path，则保存模型和处理器到指定路径
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # 如果需要推送到 Hub
        print(f"Pushing model and processor for {model_name} to hub")
        # 将模型推送到 Hub
        model.push_to_hub(f"openmmlab/{model_name}")
        # 将处理器推送到 Hub
        processor.push_to_hub(f"openmmlab/{model_name}")
if __name__ == "__main__":
    # 如果脚本直接运行而非被导入，则执行以下代码
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    # 必填参数
    parser.add_argument(
        "--model_name",
        default="upernet-swin-tiny",
        type=str,
        choices=[f"upernet-swin-{size}" for size in ["tiny", "small", "base", "large"]],
        help="Name of the Swin + UperNet model you'd like to convert.",
    )
    # 模型名称，可以选择的值为 upernet-swin-tiny、upernet-swin-small、upernet-swin-base、upernet-swin-large

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # PyTorch 模型输出目录的路径，可以是任意有效的字符串路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 是否将转换后的模型推送到 🤗 hub

    # 解析命令行参数并返回命名空间对象
    args = parser.parse_args()

    # 调用函数 convert_upernet_checkpoint，传入解析后的参数
    convert_upernet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```