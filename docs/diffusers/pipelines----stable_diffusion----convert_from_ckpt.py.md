# `.\diffusers\pipelines\stable_diffusion\convert_from_ckpt.py`

```py
# 指定文件编码为 UTF-8
# coding=utf-8
# 版权声明，表明文件版权归 HuggingFace Inc. 团队所有
# Copyright 2024 The HuggingFace Inc. team.
#
# 根据 Apache License 2.0 许可协议提供文件使用条款
# Licensed under the Apache License, Version 2.0 (the "License");
# 仅在遵循许可协议的情况下使用该文件
# you may not use this file except in compliance with the License.
# 可以通过以下链接获取许可协议的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律适用或书面协议规定，否则软件按 "现状" 基础分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解具体权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
# 稳定扩散检查点的转换脚本
"""Conversion script for the Stable Diffusion checkpoints."""

# 导入正则表达式模块
import re
# 从上下文管理器导入空上下文
from contextlib import nullcontext
# 导入字节流模块
from io import BytesIO
# 导入类型定义
from typing import Dict, Optional, Union

# 导入请求库
import requests
# 导入 PyTorch 库
import torch
# 导入 YAML 解析库
import yaml
# 从 transformers 库导入所需的类和函数
from transformers import (
    AutoFeatureExtractor,  # 自动特征提取器
    BertTokenizerFast,     # 快速的 BERT 分词器
    CLIPImageProcessor,    # CLIP 图像处理器
    CLIPTextConfig,       # CLIP 文本配置
    CLIPTextModel,        # CLIP 文本模型
    CLIPTextModelWithProjection,  # 带投影的 CLIP 文本模型
    CLIPTokenizer,        # CLIP 分词器
    CLIPVisionConfig,     # CLIP 视觉配置
    CLIPVisionModelWithProjection,  # 带投影的 CLIP 视觉模型
)

# 从本地模型导入所需的类
from ...models import (
    AutoencoderKL,         # 自动编码器
    ControlNetModel,      # 控制网络模型
    PriorTransformer,     # 先验变换模型
    UNet2DConditionModel,  # 2D 条件 U-Net 模型
)
# 从调度器导入所需的类
from ...schedulers import (
    DDIMScheduler,               # DDIM 调度器
    DDPMScheduler,               # DDPMScheduler
    DPMSolverMultistepScheduler, # DPM 多步求解调度器
    EulerAncestralDiscreteScheduler,  # 欧拉祖先离散调度器
    EulerDiscreteScheduler,      # 欧拉离散调度器
    HeunDiscreteScheduler,       # Heun 离散调度器
    LMSDiscreteScheduler,        # LMS 离散调度器
    PNDMScheduler,               # PNDM 调度器
    UnCLIPScheduler,             # UnCLIP 调度器
)
# 从工具模块导入功能
from ...utils import is_accelerate_available, logging
# 从潜在扩散管道导入所需的类
from ..latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel
# 从图像编码模块导入
from ..paint_by_example import PaintByExampleImageEncoder
# 从管道工具模块导入
from ..pipeline_utils import DiffusionPipeline
# 从安全检查模块导入
from .safety_checker import StableDiffusionSafetyChecker
# 从图像归一化模块导入
from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

# 检查加速库是否可用，如果可用则导入相关功能
if is_accelerate_available():
    # 从 accelerate 库导入初始化空权重的函数
    from accelerate import init_empty_weights
    # 从 accelerate.utils 导入设置模块张量到设备的函数
    from accelerate.utils import set_module_tensor_to_device

# 创建日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义函数以剃除路径中的段
def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    # 如果剃除段数为非负值
    if n_shave_prefix_segments >= 0:
        # 从路径中剃除前 n_shave_prefix_segments 段
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        # 从路径中剃除最后 n_shave_prefix_segments 段
        return ".".join(path.split(".")[:n_shave_prefix_segments])

# 定义函数以更新 ResNet 路径
def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    # 创建一个空的映射列表
    mapping = []
    # 遍历旧列表中的每个项目
        for old_item in old_list:
            # 将旧项目中的 "in_layers.0" 替换为 "norm1"
            new_item = old_item.replace("in_layers.0", "norm1")
            # 将旧项目中的 "in_layers.2" 替换为 "conv1"
            new_item = new_item.replace("in_layers.2", "conv1")
    
            # 将旧项目中的 "out_layers.0" 替换为 "norm2"
            new_item = new_item.replace("out_layers.0", "norm2")
            # 将旧项目中的 "out_layers.3" 替换为 "conv2"
            new_item = new_item.replace("out_layers.3", "conv2")
    
            # 将旧项目中的 "emb_layers.1" 替换为 "time_emb_proj"
            new_item = new_item.replace("emb_layers.1", "time_emb_proj")
            # 将旧项目中的 "skip_connection" 替换为 "conv_shortcut"
            new_item = new_item.replace("skip_connection", "conv_shortcut")
    
            # 对新项目进行修剪，去掉指定数量的前缀段
            new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)
    
            # 将旧项目和新项目的映射添加到列表中
            mapping.append({"old": old_item, "new": new_item})
    
        # 返回旧项目和新项目的映射列表
        return mapping
# 更新 VAE ResNet 中路径以符合新的命名规范（局部重命名）
def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    # 初始化一个映射列表，用于存储旧路径和新路径的对应关系
    mapping = []
    # 遍历旧路径列表中的每个项目
    for old_item in old_list:
        # 将当前旧路径赋值给新路径变量
        new_item = old_item

        # 将 'nin_shortcut' 替换为 'conv_shortcut'
        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        # 根据需要去除前缀段
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        # 将旧路径和新路径的映射添加到列表中
        mapping.append({"old": old_item, "new": new_item})

    # 返回旧路径和新路径的映射列表
    return mapping


# 更新注意力层中的路径以符合新的命名规范（局部重命名）
def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    # 初始化一个映射列表，用于存储旧路径和新路径的对应关系
    mapping = []
    # 遍历旧路径列表中的每个项目
    for old_item in old_list:
        # 将当前旧路径赋值给新路径变量
        new_item = old_item

        # 下面的代码行是注释掉的，用于替换 norm.weight 和 norm.bias 的命名
        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        # 下面的代码行是注释掉的，用于替换 proj_out.weight 和 proj_out.bias 的命名
        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        # 下面的代码行是注释掉的，根据需要去除前缀段
        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        # 将旧路径和新路径的映射添加到列表中
        mapping.append({"old": old_item, "new": new_item})

    # 返回旧路径和新路径的映射列表
    return mapping


# 更新 VAE 注意力层中的路径以符合新的命名规范（局部重命名）
def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    # 初始化一个映射列表，用于存储旧路径和新路径的对应关系
    mapping = []
    # 遍历旧路径列表中的每个项目
    for old_item in old_list:
        # 将当前旧路径赋值给新路径变量
        new_item = old_item

        # 将 'norm.weight' 替换为 'group_norm.weight'
        new_item = new_item.replace("norm.weight", "group_norm.weight")
        # 将 'norm.bias' 替换为 'group_norm.bias'
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        # 将 'q.weight' 替换为 'to_q.weight'
        new_item = new_item.replace("q.weight", "to_q.weight")
        # 将 'q.bias' 替换为 'to_q.bias'
        new_item = new_item.replace("q.bias", "to_q.bias")

        # 将 'k.weight' 替换为 'to_k.weight'
        new_item = new_item.replace("k.weight", "to_k.weight")
        # 将 'k.bias' 替换为 'to_k.bias'
        new_item = new_item.replace("k.bias", "to_k.bias")

        # 将 'v.weight' 替换为 'to_v.weight'
        new_item = new_item.replace("v.weight", "to_v.weight")
        # 将 'v.bias' 替换为 'to_v.bias'
        new_item = new_item.replace("v.bias", "to_v.bias")

        # 将 'proj_out.weight' 替换为 'to_out.0.weight'
        new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
        # 将 'proj_out.bias' 替换为 'to_out.0.bias'
        new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        # 根据需要去除前缀段
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        # 将旧路径和新路径的映射添加到列表中
        mapping.append({"old": old_item, "new": new_item})

    # 返回旧路径和新路径的映射列表
    return mapping


# 将转换后的权重分配给新的检查点，应用全局重命名
def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    # 确保路径是一个包含 'old' 和 'new' 键的字典列表
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # 将注意力层拆分为三个变量
    # 检查是否有需要拆分的注意力路径
    if attention_paths_to_split is not None:
        # 遍历需要拆分的注意力路径及其映射
        for path, path_map in attention_paths_to_split.items():
            # 获取旧检查点中对应路径的张量
            old_tensor = old_checkpoint[path]
            # 计算通道数，假设每个注意力有三个通道
            channels = old_tensor.shape[0] // 3

            # 根据张量维度确定目标形状
            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            # 计算头数
            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            # 重塑旧张量为（头数，通道数/头数，其他维度）
            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            # 将旧张量分割为查询、键和值
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            # 将查询、键、值重塑并存入检查点
            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    # 遍历所有路径
    for path in paths:
        # 获取新路径
        new_path = path["new"]

        # 如果新路径已在拆分路径中，跳过
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # 执行全局重命名
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        # 如果有额外的替换规则，应用它们
        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # 检查是否需要转换注意力权重
        is_attn_weight = "proj_attn.weight" in new_path or ("attentions" in new_path and "to_" in new_path)
        # 获取旧检查点中对应路径的形状
        shape = old_checkpoint[path["old"]].shape
        # 根据形状和类型存入新检查点
        if is_attn_weight and len(shape) == 3:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        elif is_attn_weight and len(shape) == 4:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]
# 将检查点中的注意力层转换为线性层
def conv_attn_to_linear(checkpoint):
    # 获取检查点字典中的所有键
    keys = list(checkpoint.keys())
    # 定义注意力层权重的关键字
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    # 遍历检查点的所有键
    for key in keys:
        # 如果键对应的权重在注意力层关键字中
        if ".".join(key.split(".")[-2:]) in attn_keys:
            # 如果权重的维度大于2，则进行切片操作
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        # 如果键中包含投影注意力权重
        elif "proj_attn.weight" in key:
            # 如果权重的维度大于2，则进行切片操作
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


# 创建适用于 Diffusers 的 UNet 配置
def create_unet_diffusers_config(original_config, image_size: int, controlnet=False):
    """
    根据 LDM 模型的配置创建 Diffusers 的配置。
    """
    # 如果使用 ControlNet，则获取相关参数
    if controlnet:
        unet_params = original_config["model"]["params"]["control_stage_config"]["params"]
    else:
        # 检查原始配置中是否包含 UNet 配置
        if (
            "unet_config" in original_config["model"]["params"]
            and original_config["model"]["params"]["unet_config"] is not None
        ):
            # 获取 UNet 参数
            unet_params = original_config["model"]["params"]["unet_config"]["params"]
        else:
            # 否则获取网络配置参数
            unet_params = original_config["model"]["params"]["network_config"]["params"]

    # 获取 VAE 的相关参数
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]

    # 计算每个块的输出通道数
    block_out_channels = [unet_params["model_channels"] * mult for mult in unet_params["channel_mult"]]

    # 初始化下采样块类型列表
    down_block_types = []
    # 初始化分辨率
    resolution = 1
    # 遍历每个块的输出通道
    for i in range(len(block_out_channels)):
        # 根据当前分辨率决定下采样块类型
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params["attention_resolutions"] else "DownBlock2D"
        down_block_types.append(block_type)
        # 更新分辨率（如果不是最后一个块）
        if i != len(block_out_channels) - 1:
            resolution *= 2

    # 初始化上采样块类型列表
    up_block_types = []
    # 遍历每个块的输出通道
    for i in range(len(block_out_channels)):
        # 根据当前分辨率决定上采样块类型
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params["attention_resolutions"] else "UpBlock2D"
        up_block_types.append(block_type)
        # 更新分辨率
        resolution //= 2

    # 如果定义了 transformer 深度
    if unet_params["transformer_depth"] is not None:
        # 判断 transformer 深度是整数还是列表
        transformer_layers_per_block = (
            unet_params["transformer_depth"]
            if isinstance(unet_params["transformer_depth"], int)
            else list(unet_params["transformer_depth"])
        )
    else:
        # 默认设置为1层
        transformer_layers_per_block = 1

    # 计算 VAE 的缩放因子
    vae_scale_factor = 2 ** (len(vae_params["ch_mult"]) - 1)

    # 获取头部维度（如果存在）
    head_dim = unet_params["num_heads"] if "num_heads" in unet_params else None
    # 确定是否使用线性投影
    use_linear_projection = (
        unet_params["use_linear_in_transformer"] if "use_linear_in_transformer" in unet_params else False
    )
    # 如果使用线性投影
    if use_linear_projection:
        # 针对稳定扩散的特定模型设置
        if head_dim is None:
            head_dim_mult = unet_params["model_channels"] // unet_params["num_head_channels"]
            head_dim = [head_dim_mult * c for c in list(unet_params["channel_mult"])]

    # 初始化额外的嵌入类型和维度
    class_embed_type = None
    addition_embed_type = None
    addition_time_embed_dim = None
    projection_class_embeddings_input_dim = None
    context_dim = None
    # 检查 unet_params 中的 context_dim 是否不为 None
    if unet_params["context_dim"] is not None:
        # 根据 context_dim 的类型设置其值
        context_dim = (
            # 如果是整数，则直接使用该值
            unet_params["context_dim"]
            if isinstance(unet_params["context_dim"], int)
            # 否则使用其第一个元素
            else unet_params["context_dim"][0]
        )

    # 检查 unet_params 中是否包含 num_classes
    if "num_classes" in unet_params:
        # 如果 num_classes 的值为 "sequential"
        if unet_params["num_classes"] == "sequential":
            # 如果 context_dim 在特定值中
            if context_dim in [2048, 1280]:
                # 设置附加嵌入类型为 "text_time"
                addition_embed_type = "text_time"
                # 设置附加时间嵌入维度为 256
                addition_time_embed_dim = 256
            else:
                # 否则设置类嵌入类型为 "projection"
                class_embed_type = "projection"
            # 确保 unet_params 中存在 "adm_in_channels"
            assert "adm_in_channels" in unet_params
            # 获取投影类嵌入的输入维度
            projection_class_embeddings_input_dim = unet_params["adm_in_channels"]

    # 构建配置字典
    config = {
        # 计算样本大小
        "sample_size": image_size // vae_scale_factor,
        # 获取输入通道数
        "in_channels": unet_params["in_channels"],
        # 将下采样块类型转换为元组
        "down_block_types": tuple(down_block_types),
        # 将块输出通道转换为元组
        "block_out_channels": tuple(block_out_channels),
        # 获取每个块的层数
        "layers_per_block": unet_params["num_res_blocks"],
        # 设置交叉注意力维度
        "cross_attention_dim": context_dim,
        # 设置注意力头的维度
        "attention_head_dim": head_dim,
        # 设置是否使用线性投影
        "use_linear_projection": use_linear_projection,
        # 设置类嵌入类型
        "class_embed_type": class_embed_type,
        # 设置附加嵌入类型
        "addition_embed_type": addition_embed_type,
        # 设置附加时间嵌入维度
        "addition_time_embed_dim": addition_time_embed_dim,
        # 设置投影类嵌入的输入维度
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        # 设置每个块的变换层数
        "transformer_layers_per_block": transformer_layers_per_block,
    }

    # 如果 unet_params 中包含 "disable_self_attentions"
    if "disable_self_attentions" in unet_params:
        # 设置仅使用交叉注意力
        config["only_cross_attention"] = unet_params["disable_self_attentions"]

    # 如果 unet_params 中包含 num_classes 并且是整数
    if "num_classes" in unet_params and isinstance(unet_params["num_classes"], int):
        # 设置类嵌入的数量
        config["num_class_embeds"] = unet_params["num_classes"]

    # 如果 controlnet 为 True
    if controlnet:
        # 设置条件通道数
        config["conditioning_channels"] = unet_params["hint_channels"]
    else:
        # 否则设置输出通道数
        config["out_channels"] = unet_params["out_channels"]
        # 设置上采样块类型
        config["up_block_types"] = tuple(up_block_types)

    # 返回配置字典
    return config
# 创建一个基于 LDM 模型配置的 diffusers 配置
def create_vae_diffusers_config(original_config, image_size: int):
    # 从原始配置中提取 VAE 参数
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    # 提取嵌入维度（未使用）
    _ = original_config["model"]["params"]["first_stage_config"]["params"]["embed_dim"]

    # 计算每个块的输出通道数量
    block_out_channels = [vae_params["ch"] * mult for mult in vae_params["ch_mult"]]
    # 创建每个下采样块的类型列表
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    # 创建每个上采样块的类型列表
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    # 创建配置字典，包含所需参数
    config = {
        "sample_size": image_size,  # 设置样本大小
        "in_channels": vae_params["in_channels"],  # 设置输入通道数
        "out_channels": vae_params["out_ch"],  # 设置输出通道数
        "down_block_types": tuple(down_block_types),  # 转换为元组并设置下采样块类型
        "up_block_types": tuple(up_block_types),  # 转换为元组并设置上采样块类型
        "block_out_channels": tuple(block_out_channels),  # 转换为元组并设置块输出通道
        "latent_channels": vae_params["z_channels"],  # 设置潜在通道数
        "layers_per_block": vae_params["num_res_blocks"],  # 设置每个块的层数
    }
    # 返回创建的配置
    return config


# 创建一个调度器配置基于原始配置
def create_diffusers_schedular(original_config):
    # 初始化 DDIMScheduler 对象，设置训练时间步和 beta 参数
    schedular = DDIMScheduler(
        num_train_timesteps=original_config["model"]["params"]["timesteps"],  # 设置训练时间步数
        beta_start=original_config["model"]["params"]["linear_start"],  # 设置 beta 开始值
        beta_end=original_config["model"]["params"]["linear_end"],  # 设置 beta 结束值
        beta_schedule="scaled_linear",  # 设置 beta 调度类型
    )
    # 返回创建的调度器
    return schedular


# 创建 LDM 的 BERT 配置
def create_ldm_bert_config(original_config):
    # 从原始配置中提取 BERT 参数
    bert_params = original_config["model"]["params"]["cond_stage_config"]["params"]
    # 创建 LDMBertConfig 对象，设置模型参数
    config = LDMBertConfig(
        d_model=bert_params.n_embed,  # 设置嵌入维度
        encoder_layers=bert_params.n_layer,  # 设置编码器层数
        encoder_ffn_dim=bert_params.n_embed * 4,  # 设置编码器前馈层维度
    )
    # 返回创建的配置
    return config


# 转换 LDM UNet 检查点
def convert_ldm_unet_checkpoint(
    checkpoint, config, path=None, extract_ema=False, controlnet=False, skip_extract_state_dict=False
):
    # 对给定的状态字典和配置进行转换，返回转换后的检查点
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # 如果跳过提取状态字典，直接使用检查点
    if skip_extract_state_dict:
        unet_state_dict = checkpoint  # 将 UNet 状态字典设为检查点
    else:
        # 提取 UNet 的状态字典
        unet_state_dict = {}  # 初始化一个空字典用于存储 UNet 的状态
        keys = list(checkpoint.keys())  # 获取检查点中所有键的列表

        if controlnet:
            unet_key = "control_model."  # 如果使用 controlnet，设置对应的键前缀
        else:
            unet_key = "model.diffusion_model."  # 否则设置为默认的模型前缀

        # 检查是否有超过 100 个参数以 `model_ema` 开头，如果是且需要提取 EMA
        if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
            logger.warning(f"Checkpoint {path} has both EMA and non-EMA weights.")  # 记录警告信息，表明检查点同时有 EMA 和非 EMA 权重
            logger.warning(
                "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
                " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
            )  # 提示用户如果想提取非 EMA 权重需要移除 `--extract_ema` 标志
            for key in keys:  # 遍历所有键
                if key.startswith("model.diffusion_model"):  # 检查键是否以指定前缀开头
                    flat_ema_key = "model_ema." + "".join(key.split(".")[1:])  # 创建相应的 EMA 键
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)  # 从检查点中移除 EMA 权重并存储在字典中
        else:
            # 如果有超过 100 个 `model_ema` 开头的参数但不提取 EMA
            if sum(k.startswith("model_ema") for k in keys) > 100:
                logger.warning(
                    "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                    " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
                )  # 提示用户如果想提取 EMA 权重需要添加 `--extract_ema` 标志

            for key in keys:  # 遍历所有键
                if key.startswith(unet_key):  # 检查键是否以 UNet 的前缀开头
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)  # 从检查点中移除对应的权重并存储在字典中

    new_checkpoint = {}  # 初始化一个新的检查点字典

    # 从 UNet 状态字典中提取时间嵌入的权重和偏置
    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]  # 提取时间嵌入第 1 层的权重
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]  # 提取时间嵌入第 1 层的偏置
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]  # 提取时间嵌入第 2 层的权重
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]  # 提取时间嵌入第 2 层的偏置

    if config["class_embed_type"] is None:  # 如果类嵌入类型为 None
        # 无需迁移的参数
        ...
    elif config["class_embed_type"] == "timestep" or config["class_embed_type"] == "projection":  # 如果类嵌入类型为 "timestep" 或 "projection"
        new_checkpoint["class_embedding.linear_1.weight"] = unet_state_dict["label_emb.0.0.weight"]  # 提取类嵌入第 1 层的权重
        new_checkpoint["class_embedding.linear_1.bias"] = unet_state_dict["label_emb.0.0.bias"]  # 提取类嵌入第 1 层的偏置
        new_checkpoint["class_embedding.linear_2.weight"] = unet_state_dict["label_emb.0.2.weight"]  # 提取类嵌入第 2 层的权重
        new_checkpoint["class_embedding.linear_2.bias"] = unet_state_dict["label_emb.0.2.bias"]  # 提取类嵌入第 2 层的偏置
    else:
        raise NotImplementedError(f"Not implemented `class_embed_type`: {config['class_embed_type']}")  # 抛出未实现错误，提示未知的类嵌入类型
    # 检查配置中的嵌入类型是否为文本时间
    if config["addition_embed_type"] == "text_time":
        # 将 UNet 状态字典中的权重赋值给新的检查点的线性层1的权重
        new_checkpoint["add_embedding.linear_1.weight"] = unet_state_dict["label_emb.0.0.weight"]
        # 将 UNet 状态字典中的偏置赋值给新的检查点的线性层1的偏置
        new_checkpoint["add_embedding.linear_1.bias"] = unet_state_dict["label_emb.0.0.bias"]
        # 将 UNet 状态字典中的权重赋值给新的检查点的线性层2的权重
        new_checkpoint["add_embedding.linear_2.weight"] = unet_state_dict["label_emb.0.2.weight"]
        # 将 UNet 状态字典中的偏置赋值给新的检查点的线性层2的偏置
        new_checkpoint["add_embedding.linear_2.bias"] = unet_state_dict["label_emb.0.2.bias"]

    # 相关于 StableDiffusionUpscalePipeline
    # 检查配置中是否存在 num_class_embeds 键
    if "num_class_embeds" in config:
        # 确保 num_class_embeds 不为空且 UNet 状态字典中存在 label_emb.weight
        if (config["num_class_embeds"] is not None) and ("label_emb.weight" in unet_state_dict):
            # 将 UNet 状态字典中的类嵌入权重赋值给新的检查点
            new_checkpoint["class_embedding.weight"] = unet_state_dict["label_emb.weight"]

    # 将 UNet 状态字典中的输入块权重赋值给新的检查点
    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    # 将 UNet 状态字典中的输入块偏置赋值给新的检查点
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    # 如果不使用 controlnet
    if not controlnet:
        # 将 UNet 状态字典中的输出块的权重赋值给新的检查点
        new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
        # 将 UNet 状态字典中的输出块的偏置赋值给新的检查点
        new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
        # 将 UNet 状态字典中的输出块的权重赋值给新的检查点
        new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
        # 将 UNet 状态字典中的输出块的偏置赋值给新的检查点
        new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # 检索输入块的键
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    # 创建一个字典，其中包含每个输入块的所有相关键
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # 检索中间块的键
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    # 创建一个字典，其中包含每个中间块的所有相关键
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # 检索输出块的键
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    # 创建一个字典，其中包含每个输出块的所有相关键
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }
    # 遍历输入块，从第一个输入块开始，直到指定数量的输入块
        for i in range(1, num_input_blocks):
            # 计算当前块所属的区块 ID
            block_id = (i - 1) // (config["layers_per_block"] + 1)
            # 计算当前层在块中的 ID
            layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)
    
            # 获取当前输入块中与 ResNet 相关的键，排除掉特定的操作键
            resnets = [
                key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
            ]
            # 获取当前输入块中与注意力相关的键
            attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]
    
            # 检查 UNet 状态字典中是否包含权重信息
            if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
                # 将权重从 UNet 状态字典移动到新的检查点字典中
                new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                    f"input_blocks.{i}.0.op.weight"
                )
                # 将偏置从 UNet 状态字典移动到新的检查点字典中
                new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                    f"input_blocks.{i}.0.op.bias"
                )
    
            # 更新 ResNet 路径
            paths = renew_resnet_paths(resnets)
            # 定义旧路径和新路径的映射关系
            meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
            # 将更新后的路径信息赋值到检查点中
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )
    
            # 如果当前块中存在注意力路径
            if len(attentions):
                # 更新注意力路径
                paths = renew_attention_paths(attentions)
    
                # 定义注意力的旧路径和新路径的映射关系
                meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
                # 将更新后的路径信息赋值到检查点中
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
    
        # 获取中间块的第一个 ResNet
        resnet_0 = middle_blocks[0]
        # 获取中间块的注意力部分
        attentions = middle_blocks[1]
        # 获取中间块的第二个 ResNet
        resnet_1 = middle_blocks[2]
    
        # 更新第一个 ResNet 的路径
        resnet_0_paths = renew_resnet_paths(resnet_0)
        # 将第一个 ResNet 的路径信息赋值到检查点中
        assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)
    
        # 更新第二个 ResNet 的路径
        resnet_1_paths = renew_resnet_paths(resnet_1)
        # 将第二个 ResNet 的路径信息赋值到检查点中
        assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)
    
        # 更新注意力路径
        attentions_paths = renew_attention_paths(attentions)
        # 定义注意力的旧路径和新路径的映射关系
        meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
        # 将更新后的路径信息赋值到检查点中
        assign_to_checkpoint(
            attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )
    # 遍历输出块的数量
    for i in range(num_output_blocks):
        # 计算当前块的 ID
        block_id = i // (config["layers_per_block"] + 1)
        # 计算当前块内层的 ID
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        # 对当前输出块的每一层进行修剪
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        # 初始化当前输出块的层列表
        output_block_list = {}

        # 遍历当前块的每一层
        for layer in output_block_layers:
            # 分离层的 ID 和名称
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            # 如果该层 ID 已存在，则添加层名称
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                # 否则新建该层 ID 的列表
                output_block_list[layer_id] = [layer_name]

        # 如果当前块的层数大于 1
        if len(output_block_list) > 1:
            # 获取当前块的残差网络路径
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            # 获取当前块的注意力路径
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            # 更新残差网络路径
            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            # 创建元路径字典以进行替换
            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            # 将路径和新检查点赋值
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            # 对输出块列表进行排序
            output_block_list = {k: sorted(v) for k, v in sorted(output_block_list.items())}
            # 检查是否存在特定的卷积层
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                # 获取卷积层的索引
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                # 将权重和偏差值赋值给新的检查点
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # 清除注意力层，因为它们已在上面处理
                if len(attentions) == 2:
                    attentions = []

            # 如果存在注意力层
            if len(attentions):
                # 更新注意力路径
                paths = renew_attention_paths(attentions)
                # 创建元路径字典以进行替换
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                # 将路径和新检查点赋值
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            # 更新残差网络路径，去除前缀
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            # 遍历每个路径
            for path in resnet_0_paths:
                # 生成旧路径和新路径
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                # 将新路径的值赋值给新的检查点
                new_checkpoint[new_path] = unet_state_dict[old_path]
    # 检查是否启用 ControlNet
    if controlnet:
        # 初始化原始索引，用于提取控制嵌入的权重和偏置

        orig_index = 0

        # 从 UNet 状态字典中弹出输入卷积层的权重，并存入新检查点
        new_checkpoint["controlnet_cond_embedding.conv_in.weight"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.weight"
        )
        # 从 UNet 状态字典中弹出输入卷积层的偏置，并存入新检查点
        new_checkpoint["controlnet_cond_embedding.conv_in.bias"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.bias"
        )

        # 更新原始索引以指向下一个层
        orig_index += 2

        # 初始化 Diffusers 索引，用于提取后续层的权重和偏置
        diffusers_index = 0

        # 循环提取 6 个控制嵌入块的权重和偏置
        while diffusers_index < 6:
            # 从 UNet 状态字典中弹出当前控制嵌入块的权重，并存入新检查点
            new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_index}.weight"] = unet_state_dict.pop(
                f"input_hint_block.{orig_index}.weight"
            )
            # 从 UNet 状态字典中弹出当前控制嵌入块的偏置，并存入新检查点
            new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_index}.bias"] = unet_state_dict.pop(
                f"input_hint_block.{orig_index}.bias"
            )
            # 更新 Diffusers 索引和原始索引以处理下一个块
            diffusers_index += 1
            orig_index += 2

        # 从 UNet 状态字典中弹出输出卷积层的权重，并存入新检查点
        new_checkpoint["controlnet_cond_embedding.conv_out.weight"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.weight"
        )
        # 从 UNet 状态字典中弹出输出卷积层的偏置，并存入新检查点
        new_checkpoint["controlnet_cond_embedding.conv_out.bias"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.bias"
        )

        # 提取下行块的权重和偏置
        for i in range(num_input_blocks):
            # 从 UNet 状态字典中弹出下行块的权重，并存入新检查点
            new_checkpoint[f"controlnet_down_blocks.{i}.weight"] = unet_state_dict.pop(f"zero_convs.{i}.0.weight")
            # 从 UNet 状态字典中弹出下行块的偏置，并存入新检查点
            new_checkpoint[f"controlnet_down_blocks.{i}.bias"] = unet_state_dict.pop(f"zero_convs.{i}.0.bias")

        # 提取中间块的权重和偏置
        new_checkpoint["controlnet_mid_block.weight"] = unet_state_dict.pop("middle_block_out.0.weight")
        new_checkpoint["controlnet_mid_block.bias"] = unet_state_dict.pop("middle_block_out.0.bias")

    # 返回新检查点
    return new_checkpoint
# 将 VAE 检查点转换为新的格式
def convert_ldm_vae_checkpoint(checkpoint, config):
    # 初始化 VAE 状态字典
    vae_state_dict = {}
    # 获取检查点的所有键
    keys = list(checkpoint.keys())
    # 检查是否有以 "first_stage_model." 开头的键
    vae_key = "first_stage_model." if any(k.startswith("first_stage_model.") for k in keys) else ""
    # 遍历所有键
    for key in keys:
        # 如果键以 vae_key 开头，则添加到 VAE 状态字典
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    # 初始化新的检查点字典
    new_checkpoint = {}

    # 从 VAE 状态字典中提取编码器的权重和偏置
    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    # 从 VAE 状态字典中提取解码器的权重和偏置
    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv.in.weight"]
    new_checkpoint["decoder.conv.in.bias"] = vae_state_dict["decoder.conv.in.bias"]
    new_checkpoint["decoder.conv.out.weight"] = vae_state_dict["decoder.conv.out.weight"]
    new_checkpoint["decoder.conv.out.bias"] = vae_state_dict["decoder.conv.out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm.out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm.out.bias"]

    # 提取量化层的权重和偏置
    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # 获取仅包含编码器下采样块的键
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # 获取仅包含解码器上采样块的键
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }
    # 遍历下采样块的数量
        for i in range(num_down_blocks):
            # 从当前下采样块中筛选出符合条件的残差网络层
            resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]
    
            # 检查 VAE 状态字典中是否存在对应的卷积权重
            if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
                # 从状态字典中移除卷积权重，并添加到新的检查点
                new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                    f"encoder.down.{i}.downsample.conv.weight"
                )
                # 从状态字典中移除卷积偏置，并添加到新的检查点
                new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                    f"encoder.down.{i}.downsample.conv.bias"
                )
    
            # 更新残差网络的路径
            paths = renew_vae_resnet_paths(resnets)
            # 定义旧路径和新路径的映射
            meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
            # 将路径分配到检查点中
            assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    
        # 从状态字典中筛选出中间残差网络的关键字
        mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
        # 设置中间残差块的数量
        num_mid_res_blocks = 2
        # 遍历中间残差块的数量
        for i in range(1, num_mid_res_blocks + 1):
            # 筛选当前中间块的残差网络
            resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]
    
            # 更新残差网络的路径
            paths = renew_vae_resnet_paths(resnets)
            # 定义旧路径和新路径的映射
            meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
            # 将路径分配到检查点中
            assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    
        # 从状态字典中筛选出中间注意力层的关键字
        mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
        # 更新注意力层的路径
        paths = renew_vae_attention_paths(mid_attentions)
        # 定义旧路径和新路径的映射
        meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
        # 将路径分配到检查点中
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
        # 将卷积注意力层转换为线性层
        conv_attn_to_linear(new_checkpoint)
    
        # 遍历上采样块的数量
        for i in range(num_up_blocks):
            # 计算当前上采样块的 ID
            block_id = num_up_blocks - 1 - i
            # 筛选当前上采样块中的残差网络层
            resnets = [
                key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
            ]
    
            # 检查 VAE 状态字典中是否存在对应的上采样卷积权重
            if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
                # 从状态字典中移除上采样卷积权重，并添加到新的检查点
                new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                    f"decoder.up.{block_id}.upsample.conv.weight"
                ]
                # 从状态字典中移除上采样卷积偏置，并添加到新的检查点
                new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                    f"decoder.up.{block_id}.upsample.conv.bias"
                ]
    
            # 更新残差网络的路径
            paths = renew_vae_resnet_paths(resnets)
            # 定义旧路径和新路径的映射
            meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
            # 将路径分配到检查点中
            assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    
        # 从状态字典中筛选出解码器中间块的关键字
        mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
        # 设置中间残差块的数量
        num_mid_res_blocks = 2
    # 遍历中间残差块的索引，从 1 到 num_mid_res_blocks（包含）
    for i in range(1, num_mid_res_blocks + 1):
        # 收集当前索引的中间残差网络中的所有相关键
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        # 更新 VAE 残差网络路径
        paths = renew_vae_resnet_paths(resnets)
        # 创建一个字典，记录旧路径和新路径的映射
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        # 将更新后的路径和映射信息分配到检查点
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    # 收集所有与中间注意力层相关的键
    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    # 更新 VAE 注意力层路径
    paths = renew_vae_attention_paths(mid_attentions)
    # 创建一个字典，记录旧注意力路径和新路径的映射
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    # 将更新后的路径和映射信息分配到检查点
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    # 将卷积注意力层转换为线性注意力层
    conv_attn_to_linear(new_checkpoint)
    # 返回更新后的检查点
    return new_checkpoint
# 定义函数，转换 LDM BERT 检查点到 Hugging Face 模型
def convert_ldm_bert_checkpoint(checkpoint, config):
    # 定义内部函数，复制注意力层的权重和偏置
    def _copy_attn_layer(hf_attn_layer, pt_attn_layer):
        # 复制查询权重
        hf_attn_layer.q_proj.weight.data = pt_attn_layer.to_q.weight
        # 复制键权重
        hf_attn_layer.k_proj.weight.data = pt_attn_layer.to_k.weight
        # 复制值权重
        hf_attn_layer.v_proj.weight.data = pt_attn_layer.to_v.weight

        # 复制输出层的权重
        hf_attn_layer.out_proj.weight = pt_attn_layer.to_out.weight
        # 复制输出层的偏置
        hf_attn_layer.out_proj.bias = pt_attn_layer.to_out.bias

    # 定义内部函数，复制线性层的权重和偏置
    def _copy_linear(hf_linear, pt_linear):
        # 复制线性层的权重
        hf_linear.weight = pt_linear.weight
        # 复制线性层的偏置
        hf_linear.bias = pt_linear.bias

    # 定义内部函数，复制整个层的参数
    def _copy_layer(hf_layer, pt_layer):
        # 复制层归一化
        _copy_linear(hf_layer.self_attn_layer_norm, pt_layer[0][0])
        _copy_linear(hf_layer.final_layer_norm, pt_layer[1][0])

        # 复制注意力层
        _copy_attn_layer(hf_layer.self_attn, pt_layer[0][1])

        # 复制 MLP
        pt_mlp = pt_layer[1][1]
        # 复制 MLP 的第一层
        _copy_linear(hf_layer.fc1, pt_mlp.net[0][0])
        # 复制 MLP 的第二层
        _copy_linear(hf_layer.fc2, pt_mlp.net[2])

    # 定义内部函数，复制多个层的参数
    def _copy_layers(hf_layers, pt_layers):
        # 遍历每一层
        for i, hf_layer in enumerate(hf_layers):
            # 跳过第一层（不复制）
            if i != 0:
                i += i
            # 获取对应的 PyTorch 层
            pt_layer = pt_layers[i : i + 2]
            # 复制当前层的参数
            _copy_layer(hf_layer, pt_layer)

    # 创建 LDM BERT 模型实例，并设置为评估模式
    hf_model = LDMBertModel(config).eval()

    # 复制嵌入层的权重
    hf_model.model.embed_tokens.weight = checkpoint.transformer.token_emb.weight
    # 复制位置嵌入层的权重
    hf_model.model.embed_positions.weight.data = checkpoint.transformer.pos_emb.emb.weight

    # 复制层归一化
    _copy_linear(hf_model.model.layer_norm, checkpoint.transformer.norm)

    # 复制隐藏层
    _copy_layers(hf_model.model.layers, checkpoint.transformer.attn_layers.layers)

    # 复制最终线性层
    _copy_linear(hf_model.to_logits, checkpoint.transformer.to_logits)

    # 返回转换后的 Hugging Face 模型
    return hf_model


# 定义函数，转换 LDM CLIP 检查点到 Hugging Face 模型
def convert_ldm_clip_checkpoint(checkpoint, local_files_only=False, text_encoder=None):
    # 如果没有提供文本编码器
    if text_encoder is None:
        # 定义默认配置名称
        config_name = "openai/clip-vit-large-patch14"
        try:
            # 从预训练模型中加载配置
            config = CLIPTextConfig.from_pretrained(config_name, local_files_only=local_files_only)
        # 捕获异常并提示用户
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: 'openai/clip-vit-large-patch14'."
            )

        # 根据是否可用选择上下文管理器
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        # 在上下文中初始化文本模型
        with ctx():
            text_model = CLIPTextModel(config)
    # 如果提供了文本编码器
    else:
        # 使用提供的文本编码器
        text_model = text_encoder

    # 获取检查点中的所有键
    keys = list(checkpoint.keys())

    # 创建一个空字典来存储文本模型的权重
    text_model_dict = {}

    # 定义需要移除的前缀
    remove_prefixes = ["cond_stage_model.transformer", "conditioner.embedders.0.transformer"]

    # 遍历所有键
    for key in keys:
        # 遍历每个需要移除的前缀
        for prefix in remove_prefixes:
            # 如果键以前缀开头
            if key.startswith(prefix):
                # 将去掉前缀后的键值对存入字典
                text_model_dict[key[len(prefix + ".") :]] = checkpoint[key]
    # 检查是否可以使用加速功能
        if is_accelerate_available():
            # 遍历文本模型字典中的参数名称及其对应的参数
            for param_name, param in text_model_dict.items():
                # 将参数的张量设置到文本模型的设备上，这里是 CPU
                set_module_tensor_to_device(text_model, param_name, "cpu", value=param)
        else:
            # 检查文本模型是否具有嵌入层及位置 ID
            if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
                # 从文本模型字典中移除位置 ID
                text_model_dict.pop("text_model.embeddings.position_ids", None)
    
            # 加载文本模型字典中的状态字典
            text_model.load_state_dict(text_model_dict)
    
        # 返回处理后的文本模型
        return text_model
# 创建文本编码转换列表，包含源名称和目标名称的元组
textenc_conversion_lst = [
    # 位置嵌入的源名称和目标名称
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
    # 令牌嵌入权重的源名称和目标名称
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    # 最终层归一化的权重源名称和目标名称
    ("ln_final.weight", "text_model.final_layer_norm.weight"),
    # 最终层归一化的偏置源名称和目标名称
    ("ln_final.bias", "text_model.final_layer_norm.bias"),
    # 文本投影的源名称和目标名称
    ("text_projection", "text_projection.weight"),
]
# 生成文本编码转换映射字典，键为源名称，值为目标名称
textenc_conversion_map = {x[0]: x[1] for x in textenc_conversion_lst}

# 创建文本编码转换列表，用于转换稳定扩散模型和 HF Diffusers
textenc_transformer_conversion_lst = [
    # (稳定扩散, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    # 层归一化1的名称映射
    ("ln_1", "layer_norm1"),
    # 层归一化2的名称映射
    ("ln_2", "layer_norm2"),
    # 全连接层的前向映射
    (".c_fc.", ".fc1."),
    # 全连接层的后向映射
    (".c_proj.", ".fc2."),
    # 注意力机制的名称映射
    (".attn", ".self_attn"),
    # 最终层归一化的名称映射
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    # 令牌嵌入权重的名称映射
    ("token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"),
    # 位置嵌入的名称映射
    ("positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"),
]
# 生成受保护的映射字典，使用正则表达式转义源名称
protected = {re.escape(x[0]): x[1] for x in textenc_transformer_conversion_lst}
# 创建正则表达式模式，匹配受保护的源名称
textenc_pattern = re.compile("|".join(protected.keys()))

# 定义函数，用于转换按示例绘制的检查点
def convert_paint_by_example_checkpoint(checkpoint, local_files_only=False):
    # 从预训练模型加载配置
    config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14", local_files_only=local_files_only)
    # 创建图像编码器模型
    model = PaintByExampleImageEncoder(config)

    # 获取检查点中的所有键
    keys = list(checkpoint.keys())

    # 初始化文本模型字典
    text_model_dict = {}

    # 遍历检查点键，提取符合条件的键值对
    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            text_model_dict[key[len("cond_stage_model.transformer.") :]] = checkpoint[key]

    # 加载 CLIP 视觉模型的状态字典
    model.model.load_state_dict(text_model_dict)

    # 加载映射器
    keys_mapper = {
        k[len("cond_stage_model.mapper.res") :]: v
        for k, v in checkpoint.items()
        if k.startswith("cond_stage_model.mapper")
    }

    # 定义映射规则
    MAPPING = {
        "attn.c_qkv": ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
        "attn.c_proj": ["attn1.to_out.0"],
        "ln_1": ["norm1"],
        "ln_2": ["norm3"],
        "mlp.c_fc": ["ff.net.0.proj"],
        "mlp.c_proj": ["ff.net.2"],
    }

    # 初始化映射权重字典
    mapped_weights = {}
    # 遍历映射键，进行权重映射
    for key, value in keys_mapper.items():
        # 获取前缀和后缀
        prefix = key[: len("blocks.i")]
        suffix = key.split(prefix)[-1].split(".")[-1]
        # 提取名称
        name = key.split(prefix)[-1].split(suffix)[0][1:-1]
        mapped_names = MAPPING[name]

        # 计算拆分数量
        num_splits = len(mapped_names)
        # 遍历映射名称并更新映射权重
        for i, mapped_name in enumerate(mapped_names):
            new_name = ".".join([prefix, mapped_name, suffix])
            shape = value.shape[0] // num_splits
            mapped_weights[new_name] = value[i * shape : (i + 1) * shape]

    # 加载映射器的状态字典
    model.mapper.load_state_dict(mapped_weights)

    # 加载最终层归一化的状态字典
    model.final_layer_norm.load_state_dict(
        {
            # 加载偏置
            "bias": checkpoint["cond_stage_model.final_ln.bias"],
            # 加载权重
            "weight": checkpoint["cond_stage_model.final_ln.weight"],
        }
    )

    # 加载最终投影
    # 加载模型的投影输出层的状态字典
        model.proj_out.load_state_dict(
            # 创建一个字典，包含偏置和权重参数
            {
                "bias": checkpoint["proj_out.bias"],
                "weight": checkpoint["proj_out.weight"],
            }
        )
    
        # 加载无条件向量
        # 将检查点中的可学习向量赋值给模型的无条件向量
        model.uncond_vector.data = torch.nn.Parameter(checkpoint["learnable_vector"])
        # 返回更新后的模型
        return model
# 定义一个函数，用于转换 OpenCLIP 的检查点
def convert_open_clip_checkpoint(
    # 检查点数据
    checkpoint,
    # 配置名称
    config_name,
    # 模型前缀，默认为 "cond_stage_model.model."
    prefix="cond_stage_model.model.",
    # 是否包含投影层，默认为 False
    has_projection=False,
    # 是否仅使用本地文件，默认为 False
    local_files_only=False,
    # 其他配置参数
    **config_kwargs,
):
    # 加载 CLIP 文本模型配置，可能抛出异常
    try:
        # 从预训练模型加载配置
        config = CLIPTextConfig.from_pretrained(config_name, **config_kwargs, local_files_only=local_files_only)
    # 捕获异常并抛出自定义错误信息
    except Exception:
        raise ValueError(
            # 指出需要本地保存的配置路径
            f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: '{config_name}'."
        )

    # 根据加速库的可用性选择上下文管理器
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    # 在选择的上下文中执行模型初始化
    with ctx():
        # 如果有投影层，则创建带投影的文本模型，否则创建普通文本模型
        text_model = CLIPTextModelWithProjection(config) if has_projection else CLIPTextModel(config)

    # 获取检查点中的所有键
    keys = list(checkpoint.keys())

    # 定义一个列表，用于存储需要忽略的键
    keys_to_ignore = []
    # 如果配置名称和隐藏层数符合条件，添加需要忽略的键
    if config_name == "stabilityai/stable-diffusion-2" and config.num_hidden_layers == 23:
        # 确保移除所有大于 22 的键
        keys_to_ignore += [k for k in keys if k.startswith("cond_stage_model.model.transformer.resblocks.23")]
        keys_to_ignore += ["cond_stage_model.model.text_projection"]

    # 初始化文本模型的字典
    text_model_dict = {}

    # 检查检查点中是否存在文本投影的键
    if prefix + "text_projection" in checkpoint:
        # 获取文本投影的维度
        d_model = int(checkpoint[prefix + "text_projection"].shape[0])
    else:
        # 默认维度为 1024
        d_model = 1024

    # 从文本模型中获取位置 IDs 并保存到字典
    text_model_dict["text_model.embeddings.position_ids"] = text_model.text_model.embeddings.get_buffer("position_ids")
    # 遍历所有关键字
    for key in keys:
        # 如果关键字在忽略列表中，则跳过当前循环
        if key in keys_to_ignore:
            continue
        # 检查去掉前缀后的关键字是否在文本编码转换映射中
        if key[len(prefix) :] in textenc_conversion_map:
            # 如果关键字以 "text_projection" 结尾
            if key.endswith("text_projection"):
                # 获取检查点中对应关键字的转置并保持为连续内存
                value = checkpoint[key].T.contiguous()
            else:
                # 否则直接获取检查点中对应关键字的值
                value = checkpoint[key]

            # 将转换后的关键字和对应的值添加到文本模型字典中
            text_model_dict[textenc_conversion_map[key[len(prefix) :]]] = value

        # 检查关键字是否以 "transformer." 为前缀
        if key.startswith(prefix + "transformer."):
            # 去掉前缀后的新关键字
            new_key = key[len(prefix + "transformer.") :]
            # 如果新关键字以 ".in_proj_weight" 结尾
            if new_key.endswith(".in_proj_weight"):
                # 去掉 ".in_proj_weight" 后缀
                new_key = new_key[: -len(".in_proj_weight")]
                # 使用正则表达式替换模式
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)
                # 将查询投影权重添加到文本模型字典中
                text_model_dict[new_key + ".q_proj.weight"] = checkpoint[key][:d_model, :]
                # 将键值投影权重添加到文本模型字典中
                text_model_dict[new_key + ".k_proj.weight"] = checkpoint[key][d_model : d_model * 2, :]
                # 将值投影权重添加到文本模型字典中
                text_model_dict[new_key + ".v_proj.weight"] = checkpoint[key][d_model * 2 :, :]
            # 如果新关键字以 ".in_proj_bias" 结尾
            elif new_key.endswith(".in_proj_bias"):
                # 去掉 ".in_proj_bias" 后缀
                new_key = new_key[: -len(".in_proj_bias")]
                # 使用正则表达式替换模式
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)
                # 将查询偏置添加到文本模型字典中
                text_model_dict[new_key + ".q_proj.bias"] = checkpoint[key][:d_model]
                # 将键值偏置添加到文本模型字典中
                text_model_dict[new_key + ".k_proj.bias"] = checkpoint[key][d_model : d_model * 2]
                # 将值偏置添加到文本模型字典中
                text_model_dict[new_key + ".v_proj.bias"] = checkpoint[key][d_model * 2 :]
            else:
                # 对新关键字应用正则表达式替换模式
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)

                # 将处理后的新关键字和对应值添加到文本模型字典中
                text_model_dict[new_key] = checkpoint[key]

    # 检查是否可用 accelerate 库
    if is_accelerate_available():
        # 遍历文本模型字典中的所有参数名及其参数
        for param_name, param in text_model_dict.items():
            # 将模型参数设置到设备（CPU）
            set_module_tensor_to_device(text_model, param_name, "cpu", value=param)
    else:
        # 如果文本模型没有嵌入或没有位置 ID 属性
        if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
            # 从文本模型字典中移除位置 ID
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        # 加载文本模型字典中的状态字典
        text_model.load_state_dict(text_model_dict)

    # 返回文本模型
    return text_model
# 定义函数，返回用于 img2img unclip 流水线的图像处理器和 clip 图像编码器
def stable_unclip_image_encoder(original_config, local_files_only=False):
    """
    返回 img2img unclip 流水线的图像处理器和 clip 图像编码器。

    我们目前知道有两种类型的稳定 unclip 模型，分别使用 clip 和 openclip 图像
    编码器。
    """

    # 获取嵌入器配置
    image_embedder_config = original_config["model"]["params"]["embedder_config"]

    # 提取目标嵌入器的类名
    sd_clip_image_embedder_class = image_embedder_config["target"]
    # 仅保留类名部分
    sd_clip_image_embedder_class = sd_clip_image_embedder_class.split(".")[-1]

    # 检查嵌入器类名是否为 ClipImageEmbedder
    if sd_clip_image_embedder_class == "ClipImageEmbedder":
        # 获取 CLIP 模型名称
        clip_model_name = image_embedder_config.params.model

        # 如果模型名称为 ViT-L/14，创建特征提取器和图像编码器
        if clip_model_name == "ViT-L/14":
            feature_extractor = CLIPImageProcessor()  # 初始化特征提取器
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14", local_files_only=local_files_only
            )  # 从预训练模型加载图像编码器
        else:
            # 如果模型名称未知，抛出未实现错误
            raise NotImplementedError(f"Unknown CLIP checkpoint name in stable diffusion checkpoint {clip_model_name}")

    # 检查嵌入器类名是否为 FrozenOpenCLIPImageEmbedder
    elif sd_clip_image_embedder_class == "FrozenOpenCLIPImageEmbedder":
        feature_extractor = CLIPImageProcessor()  # 初始化特征提取器
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", local_files_only=local_files_only
        )  # 从预训练模型加载图像编码器
    else:
        # 如果嵌入器类名未知，抛出未实现错误
        raise NotImplementedError(
            f"Unknown CLIP image embedder class in stable diffusion checkpoint {sd_clip_image_embedder_class}"
        )

    # 返回特征提取器和图像编码器
    return feature_extractor, image_encoder


# 定义函数，返回用于 img2img 和 txt2img unclip 流水线的噪声组件
def stable_unclip_image_noising_components(
    original_config, clip_stats_path: Optional[str] = None, device: Optional[str] = None
):
    """
    返回 img2img 和 txt2img unclip 流水线的噪声组件。

    将稳定性噪声增强器转换为
    1. 用于保存 CLIP 统计信息的 `StableUnCLIPImageNormalizer`
    2. 用于保存噪声调度的 `DDPMScheduler`

    如果噪声增强器配置指定了 CLIP 统计信息路径，则必须提供 `clip_stats_path`。
    """
    # 获取噪声增强器配置
    noise_aug_config = original_config["model"]["params"]["noise_aug_config"]
    # 提取噪声增强器的类名
    noise_aug_class = noise_aug_config["target"]
    # 仅保留类名部分
    noise_aug_class = noise_aug_class.split(".")[-1]
    # 检查是否使用 CLIP 噪声增强类
        if noise_aug_class == "CLIPEmbeddingNoiseAugmentation":
            # 获取噪声增强配置的参数
            noise_aug_config = noise_aug_config.params
            # 获取时间步长维度
            embedding_dim = noise_aug_config.timestep_dim
            # 获取最大噪声级别
            max_noise_level = noise_aug_config.noise_schedule_config.timesteps
            # 获取贝塔调度配置
            beta_schedule = noise_aug_config.noise_schedule_config.beta_schedule
    
            # 创建图像归一化器，基于嵌入维度
            image_normalizer = StableUnCLIPImageNormalizer(embedding_dim=embedding_dim)
            # 创建 DDPM 调度器，基于最大训练时间步长和贝塔调度
            image_noising_scheduler = DDPMScheduler(num_train_timesteps=max_noise_level, beta_schedule=beta_schedule)
    
            # 检查噪声增强配置中是否包含 clip_stats_path
            if "clip_stats_path" in noise_aug_config:
                # 如果 clip_stats_path 为空，则抛出错误
                if clip_stats_path is None:
                    raise ValueError("This stable unclip config requires a `clip_stats_path`")
    
                # 从给定路径加载 CLIP 均值和标准差，适应设备
                clip_mean, clip_std = torch.load(clip_stats_path, map_location=device)
                # 增加维度以适应后续操作
                clip_mean = clip_mean[None, :]
                clip_std = clip_std[None, :]
    
                # 创建字典以保存均值和标准差
                clip_stats_state_dict = {
                    "mean": clip_mean,
                    "std": clip_std,
                }
    
                # 加载 CLIP 统计信息到图像归一化器
                image_normalizer.load_state_dict(clip_stats_state_dict)
        else:
            # 如果噪声增强类未知，抛出未实现的错误
            raise NotImplementedError(f"Unknown noise augmentor class: {noise_aug_class}")
    
        # 返回图像归一化器和噪声调度器
        return image_normalizer, image_noising_scheduler
# 定义一个转换 ControlNet 检查点的函数
def convert_controlnet_checkpoint(
    # 检查点数据
    checkpoint,
    # 原始配置
    original_config,
    # 检查点路径
    checkpoint_path,
    # 图像大小
    image_size,
    # 是否上溯注意力
    upcast_attention,
    # 是否提取 EMA（指数移动平均）
    extract_ema,
    # 可选的线性投影参数
    use_linear_projection=None,
    # 可选的交叉注意力维度
    cross_attention_dim=None,
):
    # 创建 UNet Diffusers 配置，设置为 ControlNet
    ctrlnet_config = create_unet_diffusers_config(original_config, image_size=image_size, controlnet=True)
    # 设置上溯注意力配置
    ctrlnet_config["upcast_attention"] = upcast_attention

    # 移除样本大小配置
    ctrlnet_config.pop("sample_size")

    # 如果有线性投影参数，则加入配置
    if use_linear_projection is not None:
        ctrlnet_config["use_linear_projection"] = use_linear_projection

    # 如果有交叉注意力维度，则加入配置
    if cross_attention_dim is not None:
        ctrlnet_config["cross_attention_dim"] = cross_attention_dim

    # 根据是否可用加速功能，选择初始化上下文
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    # 在上下文中创建 ControlNet 模型
    with ctx():
        controlnet = ControlNetModel(**ctrlnet_config)

    # 检查点文件可能独立分发与模型组件
    if "time_embed.0.weight" in checkpoint:
        # 如果检查点包含特定权重，则跳过提取状态字典
        skip_extract_state_dict = True
    else:
        # 否则不跳过提取状态字典
        skip_extract_state_dict = False

    # 转换 LDM UNet 检查点
    converted_ctrl_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint,
        ctrlnet_config,
        path=checkpoint_path,
        extract_ema=extract_ema,
        controlnet=True,
        skip_extract_state_dict=skip_extract_state_dict,
    )

    # 如果可用加速功能，则将参数设置到 ControlNet 模型
    if is_accelerate_available():
        for param_name, param in converted_ctrl_checkpoint.items():
            set_module_tensor_to_device(controlnet, param_name, "cpu", value=param)
    else:
        # 否则直接加载状态字典
        controlnet.load_state_dict(converted_ctrl_checkpoint)

    # 返回构建好的 ControlNet 模型
    return controlnet


# 定义从原始 Stable Diffusion 检查点下载的函数
def download_from_original_stable_diffusion_ckpt(
    # 检查点路径或字典
    checkpoint_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    # 原始配置文件路径
    original_config_file: str = None,
    # 图像大小
    image_size: Optional[int] = None,
    # 预测类型
    prediction_type: str = None,
    # 模型类型
    model_type: str = None,
    # 是否提取 EMA
    extract_ema: bool = False,
    # 调度器类型
    scheduler_type: str = "pndm",
    # 输入通道数
    num_in_channels: Optional[int] = None,
    # 是否上溯注意力
    upcast_attention: Optional[bool] = None,
    # 设备类型
    device: str = None,
    # 是否从安全张量加载
    from_safetensors: bool = False,
    # 可选的稳定解码器路径
    stable_unclip: Optional[str] = None,
    # 可选的稳定解码器优先级路径
    stable_unclip_prior: Optional[str] = None,
    # 可选的剪辑统计路径
    clip_stats_path: Optional[str] = None,
    # 是否使用 ControlNet
    controlnet: Optional[bool] = None,
    # 是否使用适配器
    adapter: Optional[bool] = None,
    # 是否加载安全检查器
    load_safety_checker: bool = True,
    # 可选的安全检查器对象
    safety_checker: Optional[StableDiffusionSafetyChecker] = None,
    # 可选的特征提取器对象
    feature_extractor: Optional[AutoFeatureExtractor] = None,
    # 可选的管道类
    pipeline_class: DiffusionPipeline = None,
    # 是否只从本地文件加载
    local_files_only=False,
    # 可选的 VAE 路径
    vae_path=None,
    # 可选的 VAE 对象
    vae=None,
    # 可选的文本编码器对象
    text_encoder=None,
    # 可选的第二文本编码器对象
    text_encoder_2=None,
    # 可选的标记器对象
    tokenizer=None,
    # 可选的第二标记器对象
    tokenizer_2=None,
    # 可选的配置文件列表
    config_files=None,
) -> DiffusionPipeline:
    """
    从 CompVis 风格的 `.ckpt`/`.safetensors` 文件和（理想情况下）`.yaml` 配置文件加载 Stable Diffusion 管道对象。

    尽管许多参数可以自动推断，但其中一些依赖于脆弱的检查。
    # 声明全局变量 step，以便在多个函数中使用该变量，但会影响经过进一步微调的模型
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    # 建议在可能的情况下覆盖默认值和/或提供 original_config_file
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    """

    # 导入 pipelines 以避免使用 from_single_file 方法时的循环导入错误
    # 从 diffusers 库中导入所需的模型管道
    from diffusers import (
        # 导入 LDM 文本到图像管道
        LDMTextToImagePipeline,
        # 导入基于示例的绘画管道
        PaintByExamplePipeline,
        # 导入控制网络的稳定扩散管道
        StableDiffusionControlNetPipeline,
        # 导入稳定扩散的修复管道
        StableDiffusionInpaintPipeline,
        # 导入标准稳定扩散管道
        StableDiffusionPipeline,
        # 导入稳定扩散的超分辨率管道
        StableDiffusionUpscalePipeline,
        # 导入稳定扩散 XL 的控制网络修复管道
        StableDiffusionXLControlNetInpaintPipeline,
        # 导入稳定扩散 XL 的图像到图像管道
        StableDiffusionXLImg2ImgPipeline,
        # 导入稳定扩散 XL 的修复管道
        StableDiffusionXLInpaintPipeline,
        # 导入稳定扩散 XL 管道
        StableDiffusionXLPipeline,
        # 导入稳定 UnCLIP 的图像到图像管道
        StableUnCLIPImg2ImgPipeline,
        # 导入稳定 UnCLIP 管道
        StableUnCLIPPipeline,
    )

    # 如果预测类型是 "v-prediction"，则将其修改为 "v_prediction"
    if prediction_type == "v-prediction":
        prediction_type = "v_prediction"

    # 检查检查点路径或字典是否为字符串类型
    if isinstance(checkpoint_path_or_dict, str):
        # 如果使用安全张量加载
        if from_safetensors:
            # 从 safetensors 库导入安全加载函数
            from safetensors.torch import load_file as safe_load
            # 使用安全加载函数加载检查点到 CPU
            checkpoint = safe_load(checkpoint_path_or_dict, device="cpu")
        else:
            # 如果未指定设备，则根据可用性选择 CUDA 或 CPU
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # 加载检查点到指定设备
                checkpoint = torch.load(checkpoint_path_or_dict, map_location=device)
            else:
                # 加载检查点到指定设备
                checkpoint = torch.load(checkpoint_path_or_dict, map_location=device)
    # 如果检查点是字典类型
    elif isinstance(checkpoint_path_or_dict, dict):
        # 直接使用该字典作为检查点
        checkpoint = checkpoint_path_or_dict

    # 检查点中有时没有 global_step 项
    if "global_step" in checkpoint:
        # 从检查点中获取 global_step
        global_step = checkpoint["global_step"]
    else:
        # 记录调试信息：未找到 global_step 键
        logger.debug("global_step key not found in model")
        # 如果未找到，设置 global_step 为 None
        global_step = None

    # 注意：这个 while 循环不是很理想，但这个 controlnet 检查点有一个额外的 "state_dict" 键
    # https://huggingface.co/thibaud/controlnet-canny-sd21
    while "state_dict" in checkpoint:
        # 更新检查点为其 "state_dict" 内容
        checkpoint = checkpoint["state_dict"]
    # 检查原始配置文件是否为 None
    if original_config_file is None:
        # 定义 V2.1 模型的关键名称
        key_name_v2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        # 定义 SD XL 基础模型的关键名称
        key_name_sd_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
        # 定义 SD XL 精修模型的关键名称
        key_name_sd_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"
        # 判断是否是上采样管道
        is_upscale = pipeline_class == StableDiffusionUpscalePipeline

        # 初始化配置 URL 为 None
        config_url = None

        # model_type = "v1"
        # 检查 config_files 是否存在并包含 "v1"
        if config_files is not None and "v1" in config_files:
            # 设置原始配置文件为 v1 的配置文件
            original_config_file = config_files["v1"]
        else:
            # 设置配置 URL 为 v1 的 YAML 文件
            config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"

        # 检查检查点中是否包含 V2.1 的关键名称并且其形状最后一维为 1024
        if key_name_v2_1 in checkpoint and checkpoint[key_name_v2_1].shape[-1] == 1024:
            # model_type = "v2"
            # 检查 config_files 是否存在并包含 "v2"
            if config_files is not None and "v2" in config_files:
                # 设置原始配置文件为 v2 的配置文件
                original_config_file = config_files["v2"]
            else:
                # 设置配置 URL 为 v2 的 YAML 文件
                config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
            # 检查全局步骤是否为 110000
            if global_step == 110000:
                # v2.1 需要上采样注意力
                upcast_attention = True
        # 检查 SD XL 基础模型的关键名称是否在检查点中
        elif key_name_sd_xl_base in checkpoint:
            # 只有基础 XL 模型有两个文本嵌入器
            # 检查 config_files 是否存在并包含 "xl"
            if config_files is not None and "xl" in config_files:
                # 设置原始配置文件为 xl 的配置文件
                original_config_file = config_files["xl"]
            else:
                # 设置配置 URL 为 XL 基础模型的 YAML 文件
                config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
        # 检查 SD XL 精修模型的关键名称是否在检查点中
        elif key_name_sd_xl_refiner in checkpoint:
            # 只有精修 XL 模型有嵌入器和一个文本嵌入器
            # 检查 config_files 是否存在并包含 "xl_refiner"
            if config_files is not None and "xl_refiner" in config_files:
                # 设置原始配置文件为 xl_refiner 的配置文件
                original_config_file = config_files["xl_refiner"]
            else:
                # 设置配置 URL 为 XL 精修模型的 YAML 文件
                config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"

        # 如果是上采样，设置相应的配置 URL
        if is_upscale:
            config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml"

        # 如果配置 URL 不为 None
        if config_url is not None:
            # 将原始配置文件设置为从 URL 获取的内容
            original_config_file = BytesIO(requests.get(config_url).content)
        else:
            # 打开原始配置文件并读取内容
            with open(original_config_file, "r") as f:
                original_config_file = f.read()
    else:
        # 如果原始配置文件不为 None，直接打开并读取内容
        with open(original_config_file, "r") as f:
            original_config_file = f.read()

    # 使用 yaml 库安全加载原始配置文件
    original_config = yaml.safe_load(original_config_file)

    # 转换文本模型。
    if (
        model_type is None
        and "cond_stage_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["cond_stage_config"] is not None
    ):
        # 从原始配置中获取模型类型，并从字符串中提取最后一部分
        model_type = original_config["model"]["params"]["cond_stage_config"]["target"].split(".")[-1]
        # 记录调试信息，显示推断出的模型类型
        logger.debug(f"no `model_type` given, `model_type` inferred as: {model_type}")
    # 如果模型类型为 None 且网络配置不为空
    elif model_type is None and original_config["model"]["params"]["network_config"] is not None:
        # 检查上下文维度是否为 2048，并根据其值设置模型类型
        if original_config["model"]["params"]["network_config"]["params"]["context_dim"] == 2048:
            model_type = "SDXL"  # 设置模型类型为 SDXL
        else:
            model_type = "SDXL-Refiner"  # 设置模型类型为 SDXL-Refiner
        # 如果图像大小为 None，则默认设置为 1024
        if image_size is None:
            image_size = 1024

    # 如果管道类为 None
    if pipeline_class is None:
        # 检查当前模型类型，初始化默认管道
        if model_type not in ["SDXL", "SDXL-Refiner"]:
            # 根据控制网络的状态选择合适的管道类
            pipeline_class = StableDiffusionPipeline if not controlnet else StableDiffusionControlNetPipeline
        else:
            # 根据模型类型选择 SDXL 管道或 SDXL Img2Img 管道
            pipeline_class = StableDiffusionXLPipeline if model_type == "SDXL" else StableDiffusionXLImg2ImgPipeline

    # 如果输入通道数量为 None，且管道类在给定列表中
    if num_in_channels is None and pipeline_class in [
        StableDiffusionInpaintPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
    ]:
        num_in_channels = 9  # 设置输入通道数量为 9
    # 如果输入通道数量为 None 且管道类为超分辨率管道
    if num_in_channels is None and pipeline_class == StableDiffusionUpscalePipeline:
        num_in_channels = 7  # 设置输入通道数量为 7
    # 如果输入通道数量仍为 None
    elif num_in_channels is None:
        num_in_channels = 4  # 设置输入通道数量为 4

    # 如果原始配置中包含 "unet_config"
    if "unet_config" in original_config["model"]["params"]:
        # 设置 U-Net 配置中的输入通道数量
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels
    # 如果原始配置中包含 "network_config"
    elif "network_config" in original_config["model"]["params"]:
        # 设置网络配置中的输入通道数量
        original_config["model"]["params"]["network_config"]["params"]["in_channels"] = num_in_channels

    # 如果原始配置中包含 "parameterization" 且其值为 "v"
    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        # 如果预测类型为 None
        if prediction_type is None:
            # 记录提示信息，建议使用 "epsilon" 作为预测类型
            # 因为此处依赖于一个不稳定的全局步骤参数
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
        # 如果图像大小为 None
        if image_size is None:
            # 记录提示信息，建议设置图像大小为 512
            # 因为此处依赖于一个不稳定的全局步骤参数
            image_size = 512 if global_step == 875000 else 768
    else:
        # 如果预测类型为 None
        if prediction_type is None:
            prediction_type = "epsilon"  # 设置默认预测类型为 "epsilon"
        # 如果图像大小为 None
        if image_size is None:
            image_size = 512  # 设置默认图像大小为 512

    # 如果控制网络为 None 且原始配置中包含 "control_stage_config"
    if controlnet is None and "control_stage_config" in original_config["model"]["params"]:
        # 根据检查点路径或字典设置路径
        path = checkpoint_path_or_dict if isinstance(checkpoint_path_or_dict, str) else ""
        # 转换控制网络检查点
        controlnet = convert_controlnet_checkpoint(
            checkpoint, original_config, path, image_size, upcast_attention, extract_ema
        )

    # 如果原始配置中包含 "timesteps"
    if "timesteps" in original_config["model"]["params"]:
        # 从原始配置中获取训练时间步数
        num_train_timesteps = original_config["model"]["params"]["timesteps"]
    else:
        # 如果不是特定模型类型，则将训练时间步数设置为 1000
        num_train_timesteps = 1000

    # 检查模型类型是否为 SDXL 或 SDXL-Refiner
    if model_type in ["SDXL", "SDXL-Refiner"]:
        # 定义调度器的参数字典
        scheduler_dict = {
            "beta_schedule": "scaled_linear",  # 设置 beta 调度为 scaled_linear
            "beta_start": 0.00085,  # 设置 beta 的起始值
            "beta_end": 0.012,  # 设置 beta 的结束值
            "interpolation_type": "linear",  # 设置插值类型为线性
            "num_train_timesteps": num_train_timesteps,  # 使用之前定义的训练时间步数
            "prediction_type": "epsilon",  # 设置预测类型为 epsilon
            "sample_max_value": 1.0,  # 设置采样的最大值
            "set_alpha_to_one": False,  # 不将 alpha 设置为 1
            "skip_prk_steps": True,  # 启用跳过 PRK 步骤
            "steps_offset": 1,  # 设置步骤偏移量
            "timestep_spacing": "leading",  # 设置时间步间隔为 leading
        }
        # 从配置字典中创建调度器
        scheduler = EulerDiscreteScheduler.from_config(scheduler_dict)
        # 将调度器类型设置为 euler
        scheduler_type = "euler"
    else:
        # 如果不是上述模型类型，检查 original_config 中是否包含 linear_start
        if "linear_start" in original_config["model"]["params"]:
            # 如果存在，则从配置中获取 beta_start
            beta_start = original_config["model"]["params"]["linear_start"]
        else:
            # 否则设置 beta_start 为 0.02
            beta_start = 0.02

        # 检查 original_config 中是否包含 linear_end
        if "linear_end" in original_config["model"]["params"]:
            # 如果存在，则从配置中获取 beta_end
            beta_end = original_config["model"]["params"]["linear_end"]
        else:
            # 否则设置 beta_end 为 0.085
            beta_end = 0.085
        # 创建 DDIM 调度器，并传入相应参数
        scheduler = DDIMScheduler(
            beta_end=beta_end,  # 使用之前设置的 beta_end
            beta_schedule="scaled_linear",  # 设置 beta 调度为 scaled_linear
            beta_start=beta_start,  # 使用之前设置的 beta_start
            num_train_timesteps=num_train_timesteps,  # 使用训练时间步数
            steps_offset=1,  # 设置步骤偏移量
            clip_sample=False,  # 不剪裁样本
            set_alpha_to_one=False,  # 不将 alpha 设置为 1
            prediction_type=prediction_type,  # 使用预测类型
        )
    # 确保调度器与 DDIM 正常工作
    scheduler.register_to_config(clip_sample=False)

    # 根据调度器类型创建相应的调度器
    if scheduler_type == "pndm":
        # 从调度器的配置字典中创建新的 PNDM 调度器
        config = dict(scheduler.config)
        config["skip_prk_steps"] = True  # 启用跳过 PRK 步骤
        scheduler = PNDMScheduler.from_config(config)
    elif scheduler_type == "lms":
        # 从配置中创建 LMS 调度器
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "heun":
        # 从配置中创建 Heun 调度器
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler":
        # 从配置中创建 Euler 调度器
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler-ancestral":
        # 从配置中创建 Euler Ancestral 调度器
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "dpm":
        # 从配置中创建 DPM 调度器
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif scheduler_type == "ddim":
        # 如果是 DDIM 调度器，则直接使用现有调度器
        scheduler = scheduler
    else:
        # 如果调度器类型不存在，抛出错误
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    # 如果使用的是 StableDiffusionUpscalePipeline，获取图像大小
    if pipeline_class == StableDiffusionUpscalePipeline:
        # 从配置中获取 UNet 的图像大小
        image_size = original_config["model"]["params"]["unet_config"]["params"]["image_size"]

    # 转换 UNet2DConditionModel 模型
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    # 设置 upcast_attention 参数
    unet_config["upcast_attention"] = upcast_attention

    # 检查点路径或字典，如果是字符串，则使用该路径
    path = checkpoint_path_or_dict if isinstance(checkpoint_path_or_dict, str) else ""
    # 转换 LDM UNet 检查点
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint,  # 传入检查点
        unet_config,  # 传入 UNet 配置
        path=path,  # 传入路径
        extract_ema=extract_ema  # 是否提取 EMA
    )
    # 根据是否可用加速器，初始化上下文环境
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        # 使用上下文管理器创建 UNet2DConditionModel 实例
        with ctx():
            unet = UNet2DConditionModel(**unet_config)
    
        # 如果可用加速器
        if is_accelerate_available():
            # 检查模型类型是否为 SDXL 或 SDXL-Refiner
            if model_type not in ["SDXL", "SDXL-Refiner"]:  # SBM Delay this.
                # 遍历转换后的 UNet 检查点中的参数
                for param_name, param in converted_unet_checkpoint.items():
                    # 将模块参数设置到指定设备上
                    set_module_tensor_to_device(unet, param_name, "cpu", value=param)
        else:
            # 从检查点加载 UNet 状态字典
            unet.load_state_dict(converted_unet_checkpoint)
    
        # 转换 VAE 模型
        if vae_path is None and vae is None:
            # 创建 VAE 配置
            vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
            # 转换 LDM VAE 检查点
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)
    
            # 检查配置中是否存在 scale_factor 参数
            if (
                "model" in original_config
                and "params" in original_config["model"]
                and "scale_factor" in original_config["model"]["params"]
            ):
                # 获取 VAE 缩放因子
                vae_scaling_factor = original_config["model"]["params"]["scale_factor"]
            else:
                # 默认 SD 缩放因子
                vae_scaling_factor = 0.18215  # default SD scaling factor
    
            # 更新 VAE 配置中的缩放因子
            vae_config["scaling_factor"] = vae_scaling_factor
    
            # 根据加速器可用性初始化上下文
            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            # 使用上下文管理器创建 AutoencoderKL 实例
            with ctx():
                vae = AutoencoderKL(**vae_config)
    
            # 如果可用加速器
            if is_accelerate_available():
                # 遍历转换后的 VAE 检查点中的参数
                for param_name, param in converted_vae_checkpoint.items():
                    # 将模块参数设置到指定设备上
                    set_module_tensor_to_device(vae, param_name, "cpu", value=param)
            else:
                # 从检查点加载 VAE 状态字典
                vae.load_state_dict(converted_vae_checkpoint)
        # 如果 VAE 为 None，但 VAE 路径存在
        elif vae is None:
            # 从预训练模型加载 VAE
            vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=local_files_only)
    
        # 如果模型类型为 PaintByExample
        elif model_type == "PaintByExample":
            # 转换 PaintByExample 检查点
            vision_model = convert_paint_by_example_checkpoint(checkpoint)
            # 尝试加载 CLIPTokenizer
            try:
                tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14", local_files_only=local_files_only
                )
            except Exception:
                # 抛出错误提示本地文件必须保存
                raise ValueError(
                    f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'openai/clip-vit-large-patch14'."
                )
            # 尝试加载 AutoFeatureExtractor
            try:
                feature_extractor = AutoFeatureExtractor.from_pretrained(
                    "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only
                )
            except Exception:
                # 抛出错误提示本地文件必须保存
                raise ValueError(
                    f"With local_files_only set to {local_files_only}, you must first locally save the feature_extractor in the following path: 'CompVis/stable-diffusion-safety-checker'."
                )
            # 创建 PaintByExamplePipeline 实例
            pipe = PaintByExamplePipeline(
                vae=vae,
                image_encoder=vision_model,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=feature_extractor,
            )
    # 检查模型类型是否为 "FrozenCLIPEmbedder"
    elif model_type == "FrozenCLIPEmbedder":
        # 将 LDM CLIP 检查点转换为文本模型
        text_model = convert_ldm_clip_checkpoint(
            checkpoint, local_files_only=local_files_only, text_encoder=text_encoder
        )
        # 尝试加载 CLIP 分词器
        try:
            tokenizer = (
                # 从预训练模型中加载 CLIP 分词器，如果 tokenizer 为 None 则加载
                CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=local_files_only)
                if tokenizer is None
                else tokenizer
            )
        # 捕获加载分词器时的异常
        except Exception:
            raise ValueError(
                # 抛出错误，提示必须先本地保存分词器
                f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'openai/clip-vit-large-patch14'."
            )

        # 如果需要加载安全检查器
        if load_safety_checker:
            # 从预训练模型中加载稳定扩散安全检查器
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only
            )
            # 从预训练模型中加载特征提取器
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only
            )

        # 如果启用 ControlNet
        if controlnet:
            # 创建包含 ControlNet 的管道
            pipe = pipeline_class(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
        else:
            # 创建不包含 ControlNet 的管道
            pipe = pipeline_class(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
    # 处理其他模型类型
    else:
        # 创建 LDM BERT 配置
        text_config = create_ldm_bert_config(original_config)
        # 将 LDM BERT 检查点转换为文本模型
        text_model = convert_ldm_bert_checkpoint(checkpoint, text_config)
        # 从预训练模型中加载 BERT 分词器
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=local_files_only)
        # 创建 LDM 文本到图像的管道
        pipe = LDMTextToImagePipeline(vqvae=vae, bert=text_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

    # 返回创建的管道
    return pipe
# 下载并控制原始检查点，返回 DiffusionPipeline 对象
def download_controlnet_from_original_ckpt(
    # 检查点文件路径
    checkpoint_path: str,
    # 原始配置文件路径
    original_config_file: str,
    # 图像尺寸，默认512
    image_size: int = 512,
    # 是否提取 EMA 权重，默认 False
    extract_ema: bool = False,
    # 输入通道数，默认为 None
    num_in_channels: Optional[int] = None,
    # 是否上溯注意力，默认为 None
    upcast_attention: Optional[bool] = None,
    # 设备类型，默认为 None
    device: str = None,
    # 是否使用 safetensors 格式，默认为 False
    from_safetensors: bool = False,
    # 是否使用线性投影，默认为 None
    use_linear_projection: Optional[bool] = None,
    # 跨注意力维度，默认为 None
    cross_attention_dim: Optional[bool] = None,
) -> DiffusionPipeline:
    # 如果使用 safetensors 格式
    if from_safetensors:
        # 导入 safe_open 函数
        from safetensors import safe_open

        # 初始化检查点字典
        checkpoint = {}
        # 打开 safetensors 文件，使用 PyTorch 设备
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            # 遍历文件中的所有键
            for key in f.keys():
                # 获取张量并存储到检查点字典
                checkpoint[key] = f.get_tensor(key)
    else:
        # 如果设备未指定
        if device is None:
            # 根据 CUDA 可用性选择设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 从指定路径加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            # 使用指定设备加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=device)

    # 注：此 while 循环用于处理控制点检查点的 "state_dict" 键
    while "state_dict" in checkpoint:
        # 更新检查点为其状态字典
        checkpoint = checkpoint["state_dict"]

    # 打开原始配置文件进行读取
    with open(original_config_file, "r") as f:
        # 读取文件内容
        original_config_file = f.read()
    # 使用 YAML 加载原始配置
    original_config = yaml.safe_load(original_config_file)

    # 如果指定输入通道数
    if num_in_channels is not None:
        # 更新原始配置中的输入通道数
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels

    # 如果原始配置中不存在控制阶段配置
    if "control_stage_config" not in original_config["model"]["params"]:
        # 抛出值错误
        raise ValueError("`control_stage_config` not present in original config")

    # 转换控制点检查点为控制网络对象
    controlnet = convert_controlnet_checkpoint(
        checkpoint,
        original_config,
        checkpoint_path,
        image_size,
        upcast_attention,
        extract_ema,
        use_linear_projection=use_linear_projection,
        cross_attention_dim=cross_attention_dim,
    )

    # 返回转换后的控制网络对象
    return controlnet
```