# `.\diffusers\loaders\single_file_utils.py`

```py
# 设置文件编码为 UTF-8
# 版权信息，标明版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本授权该文件；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律另有规定或书面同意，软件在许可证下分发，
# 均按“原样”基础提供，不附带任何形式的保证或条件，
# 明示或暗示均不作任何承诺。
# 请参阅许可证以获取特定的语言管理权限和
# 限制条款。
"""用于 Stable Diffusion 检查点的转换脚本。"""

# 导入必要的模块
import os  # 用于操作系统功能的模块
import re  # 用于正则表达式操作的模块
from contextlib import nullcontext  # 提供上下文管理器功能
from io import BytesIO  # 用于处理字节流的模块
from urllib.parse import urlparse  # 用于解析 URL 的模块

import requests  # 用于发送 HTTP 请求的库
import torch  # PyTorch 深度学习库
import yaml  # 用于处理 YAML 文件的库

# 导入模型相关的工具
from ..models.modeling_utils import load_state_dict  # 加载模型状态字典的函数
from ..schedulers import (  # 导入不同的调度器类
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EDMDPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

# 导入实用工具
from ..utils import (  # 导入一些实用函数和常量
    SAFETENSORS_WEIGHTS_NAME,  # 安全张量权重名称
    WEIGHTS_NAME,  # 权重名称
    deprecate,  # 警告使用过时功能的函数
    is_accelerate_available,  # 检查 accelerate 模块是否可用的函数
    is_transformers_available,  # 检查 transformers 模块是否可用的函数
    logging,  # 日志记录功能
)
from ..utils.hub_utils import _get_model_file  # 获取模型文件的辅助函数

# 如果 transformers 可用，则导入相关类
if is_transformers_available():
    from transformers import AutoImageProcessor  # 自动图像处理器类

# 如果 accelerate 可用，则导入相关功能
if is_accelerate_available():
    from accelerate import init_empty_weights  # 初始化空权重的函数

    from ..models.modeling_utils import load_model_dict_into_meta  # 加载模型字典到元数据的函数

logger = logging.get_logger(__name__)  # 创建一个记录器实例，用于日志记录；禁用 pylint 的名称检查

# 定义一个字典，用于存储不同检查点关键名称的映射
CHECKPOINT_KEY_NAMES = {
    "v2": "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",  # v2 模型的权重名称
    "xl_base": "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias",  # xl_base 模型的偏置名称
    "xl_refiner": "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias",  # xl_refiner 模型的偏置名称
    "upscale": "model.diffusion_model.input_blocks.10.0.skip_connection.bias",  # upscale 模型的偏置名称
    "controlnet": "control_model.time_embed.0.weight",  # controlnet 模型的权重名称
    "playground-v2-5": "edm_mean",  # playground-v2-5 模型的平均值
    "inpainting": "model.diffusion_model.input_blocks.0.0.weight",  # inpainting 模型的权重名称
    "clip": "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",  # clip 模型的位置嵌入权重
    "clip_sdxl": "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",  # clip_sdxl 模型的位置嵌入权重
    "clip_sd3": "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight",  # clip_sd3 模型的位置嵌入权重
    "open_clip": "cond_stage_model.model.token_embedding.weight",  # open_clip 模型的嵌入权重
    "open_clip_sdxl": "conditioner.embedders.1.model.positional_embedding",  # open_clip_sdxl 模型的位置嵌入
    "open_clip_sdxl_refiner": "conditioner.embedders.0.model.text_projection",  # open_clip_sdxl_refiner 模型的文本投影
    "open_clip_sd3": "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight",  # open_clip_sd3 模型的位置嵌入权重
    "stable_cascade_stage_b": "down_blocks.1.0.channelwise.0.weight",  # stable_cascade_stage_b 模型的权重名称
    "stable_cascade_stage_c": "clip_txt_mapper.weight",  # stable_cascade_stage_c 模型的权重名称
}
    # 定义一个字典的键值对，键为模型名称，值为对应的模型参数路径
    "sd3": "model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.bias",
    # 定义另一个模型的参数路径，适用于 animatediff 模型
    "animatediff": "down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.pos_encoder.pe",
    # 定义 animatediff_v2 模型的参数路径
    "animatediff_v2": "mid_block.motion_modules.0.temporal_transformer.norm.bias",
    # 定义 animatediff_sdxl_beta 模型的参数路径
    "animatediff_sdxl_beta": "up_blocks.2.motion_modules.0.temporal_transformer.norm.weight",
    # 定义 animatediff_scribble 模型的参数路径
    "animatediff_scribble": "controlnet_cond_embedding.conv_in.weight",
    # 定义 animatediff_rgb 模型的参数路径
    "animatediff_rgb": "controlnet_cond_embedding.weight",
    # 定义一个列表，包含 flux 模型相关的参数路径
    "flux": [
        # flux 模型中 double_blocks 组件的参数路径
        "double_blocks.0.img_attn.norm.key_norm.scale",
        # flux 模型中另一个 double_blocks 组件的参数路径
        "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
    ],
# 定义默认的 Diffusers 管道路径，映射模型名称到其预训练模型的路径
DIFFUSERS_DEFAULT_PIPELINE_PATHS = {
    # xl_base 模型的预训练模型路径
    "xl_base": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0"},
    # xl_refiner 模型的预训练模型路径
    "xl_refiner": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-refiner-1.0"},
    # xl_inpaint 模型的预训练模型路径
    "xl_inpaint": {"pretrained_model_name_or_path": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"},
    # playground-v2-5 模型的预训练模型路径
    "playground-v2-5": {"pretrained_model_name_or_path": "playgroundai/playground-v2.5-1024px-aesthetic"},
    # upscale 模型的预训练模型路径
    "upscale": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-x4-upscaler"},
    # inpainting 模型的预训练模型路径
    "inpainting": {"pretrained_model_name_or_path": "Lykon/dreamshaper-8-inpainting"},
    # inpainting_v2 模型的预训练模型路径
    "inpainting_v2": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-inpainting"},
    # controlnet 模型的预训练模型路径
    "controlnet": {"pretrained_model_name_or_path": "lllyasviel/control_v11p_sd15_canny"},
    # v2 模型的预训练模型路径
    "v2": {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1"},
    # v1 模型的预训练模型路径
    "v1": {"pretrained_model_name_or_path": "Lykon/dreamshaper-8"},
    # stable_cascade_stage_b 模型的预训练模型路径及其子文件夹
    "stable_cascade_stage_b": {"pretrained_model_name_or_path": "stabilityai/stable-cascade", "subfolder": "decoder"},
    # stable_cascade_stage_b_lite 模型的预训练模型路径及其子文件夹
    "stable_cascade_stage_b_lite": {
        "pretrained_model_name_or_path": "stabilityai/stable-cascade",
        "subfolder": "decoder_lite",
    },
    # stable_cascade_stage_c 模型的预训练模型路径及其子文件夹
    "stable_cascade_stage_c": {
        "pretrained_model_name_or_path": "stabilityai/stable-cascade-prior",
        "subfolder": "prior",
    },
    # stable_cascade_stage_c_lite 模型的预训练模型路径及其子文件夹
    "stable_cascade_stage_c_lite": {
        "pretrained_model_name_or_path": "stabilityai/stable-cascade-prior",
        "subfolder": "prior_lite",
    },
    # sd3 模型的预训练模型路径
    "sd3": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
    },
    # animatediff_v1 模型的预训练模型路径
    "animatediff_v1": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-v1-5"},
    # animatediff_v2 模型的预训练模型路径
    "animatediff_v2": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-v1-5-2"},
    # animatediff_v3 模型的预训练模型路径
    "animatediff_v3": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-v1-5-3"},
    # animatediff_sdxl_beta 模型的预训练模型路径
    "animatediff_sdxl_beta": {"pretrained_model_name_or_path": "guoyww/animatediff-motion-adapter-sdxl-beta"},
    # animatediff_scribble 模型的预训练模型路径
    "animatediff_scribble": {"pretrained_model_name_or_path": "guoyww/animatediff-sparsectrl-scribble"},
    # animatediff_rgb 模型的预训练模型路径
    "animatediff_rgb": {"pretrained_model_name_or_path": "guoyww/animatediff-sparsectrl-rgb"},
    # flux-dev 模型的预训练模型路径
    "flux-dev": {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev"},
    # flux-schnell 模型的预训练模型路径
    "flux-schnell": {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-schnell"},
}

# 用于配置模型样本大小，当提供原始配置时
DIFFUSERS_TO_LDM_DEFAULT_IMAGE_SIZE_MAP = {
    # xl_base 模型的默认图像大小
    "xl_base": 1024,
    # xl_refiner 模型的默认图像大小
    "xl_refiner": 1024,
    # xl_inpaint 模型的默认图像大小
    "xl_inpaint": 1024,
    # playground-v2-5 模型的默认图像大小
    "playground-v2-5": 1024,
    # upscale 模型的默认图像大小
    "upscale": 512,
    # inpainting 模型的默认图像大小
    "inpainting": 512,
    # inpainting_v2 模型的默认图像大小
    "inpainting_v2": 512,
    # controlnet 模型的默认图像大小
    "controlnet": 512,
    # v2 模型的默认图像大小
    "v2": 768,
    # v1 模型的默认图像大小
    "v1": 512,
}

# 定义 Diffusers 到 LDM 的映射
DIFFUSERS_TO_LDM_MAPPING = {
    # 定义一个包含 UNet 模型参数的字典
    "unet": {
        # 定义 UNet 模型的层参数
        "layers": {
            # 将时间嵌入层的第一个线性层权重映射到新位置
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            # 将时间嵌入层的第一个线性层偏置映射到新位置
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            # 将时间嵌入层的第二个线性层权重映射到新位置
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            # 将时间嵌入层的第二个线性层偏置映射到新位置
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            # 将输入卷积层的权重映射到新位置
            "conv_in.weight": "input_blocks.0.0.weight",
            # 将输入卷积层的偏置映射到新位置
            "conv_in.bias": "input_blocks.0.0.bias",
            # 将输出归一化层的权重映射到新位置
            "conv_norm_out.weight": "out.0.weight",
            # 将输出归一化层的偏置映射到新位置
            "conv_norm_out.bias": "out.0.bias",
            # 将输出卷积层的权重映射到新位置
            "conv_out.weight": "out.2.weight",
            # 将输出卷积层的偏置映射到新位置
            "conv_out.bias": "out.2.bias",
        },
        # 定义分类嵌入层的参数
        "class_embed_type": {
            # 将分类嵌入层的第一个线性层权重映射到新位置
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            # 将分类嵌入层的第一个线性层偏置映射到新位置
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            # 将分类嵌入层的第二个线性层权重映射到新位置
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            # 将分类嵌入层的第二个线性层偏置映射到新位置
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        # 定义附加嵌入层的参数
        "addition_embed_type": {
            # 将附加嵌入层的第一个线性层权重映射到新位置
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            # 将附加嵌入层的第一个线性层偏置映射到新位置
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            # 将附加嵌入层的第二个线性层权重映射到新位置
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            # 将附加嵌入层的第二个线性层偏置映射到新位置
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    # 定义一个包含 ControlNet 模型参数的字典
    "controlnet": {
        # 定义 ControlNet 模型的层参数
        "layers": {
            # 将时间嵌入层的第一个线性层权重映射到新位置
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            # 将时间嵌入层的第一个线性层偏置映射到新位置
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            # 将时间嵌入层的第二个线性层权重映射到新位置
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            # 将时间嵌入层的第二个线性层偏置映射到新位置
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            # 将输入卷积层的权重映射到新位置
            "conv_in.weight": "input_blocks.0.0.weight",
            # 将输入卷积层的偏置映射到新位置
            "conv_in.bias": "input_blocks.0.0.bias",
            # 将 ControlNet 条件嵌入的输入卷积层权重映射到新位置
            "controlnet_cond_embedding.conv_in.weight": "input_hint_block.0.weight",
            # 将 ControlNet 条件嵌入的输入卷积层偏置映射到新位置
            "controlnet_cond_embedding.conv_in.bias": "input_hint_block.0.bias",
            # 将 ControlNet 条件嵌入的输出卷积层权重映射到新位置
            "controlnet_cond_embedding.conv_out.weight": "input_hint_block.14.weight",
            # 将 ControlNet 条件嵌入的输出卷积层偏置映射到新位置
            "controlnet_cond_embedding.conv_out.bias": "input_hint_block.14.bias",
        },
        # 定义分类嵌入层的参数
        "class_embed_type": {
            # 将分类嵌入层的第一个线性层权重映射到新位置
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            # 将分类嵌入层的第一个线性层偏置映射到新位置
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            # 将分类嵌入层的第二个线性层权重映射到新位置
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            # 将分类嵌入层的第二个线性层偏置映射到新位置
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        # 定义附加嵌入层的参数
        "addition_embed_type": {
            # 将附加嵌入层的第一个线性层权重映射到新位置
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            # 将附加嵌入层的第一个线性层偏置映射到新位置
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            # 将附加嵌入层的第二个线性层权重映射到新位置
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            # 将附加嵌入层的第二个线性层偏置映射到新位置
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    # 定义一个字典，包含 VAE 模型的参数映射
    "vae": {
        # 映射编码器输入卷积层的权重
        "encoder.conv_in.weight": "encoder.conv_in.weight",
        # 映射编码器输入卷积层的偏置
        "encoder.conv_in.bias": "encoder.conv_in.bias",
        # 映射编码器输出卷积层的权重
        "encoder.conv_out.weight": "encoder.conv_out.weight",
        # 映射编码器输出卷积层的偏置
        "encoder.conv_out.bias": "encoder.conv_out.bias",
        # 映射编码器归一化输出层的权重
        "encoder.conv_norm_out.weight": "encoder.norm_out.weight",
        # 映射编码器归一化输出层的偏置
        "encoder.conv_norm_out.bias": "encoder.norm_out.bias",
        # 映射解码器输入卷积层的权重
        "decoder.conv_in.weight": "decoder.conv_in.weight",
        # 映射解码器输入卷积层的偏置
        "decoder.conv_in.bias": "decoder.conv_in.bias",
        # 映射解码器输出卷积层的权重
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        # 映射解码器输出卷积层的偏置
        "decoder.conv_out.bias": "decoder.conv_out.bias",
        # 映射解码器归一化输出层的权重
        "decoder.conv_norm_out.weight": "decoder.norm_out.weight",
        # 映射解码器归一化输出层的偏置
        "decoder.conv_norm_out.bias": "decoder.norm_out.bias",
        # 映射量化卷积层的权重
        "quant_conv.weight": "quant_conv.weight",
        # 映射量化卷积层的偏置
        "quant_conv.bias": "quant_conv.bias",
        # 映射后量化卷积层的权重
        "post_quant_conv.weight": "post_quant_conv.weight",
        # 映射后量化卷积层的偏置
        "post_quant_conv.bias": "post_quant_conv.bias",
    },
    # 定义一个字典，包含 OpenCLIP 模型的参数映射
    "openclip": {
        # 定义嵌套字典，包含文本模型的层参数映射
        "layers": {
            # 映射文本模型的位置信息嵌入层权重
            "text_model.embeddings.position_embedding.weight": "positional_embedding",
            # 映射文本模型的标记嵌入层权重
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            # 映射文本模型最终归一化层的权重
            "text_model.final_layer_norm.weight": "ln_final.weight",
            # 映射文本模型最终归一化层的偏置
            "text_model.final_layer_norm.bias": "ln_final.bias",
            # 映射文本投影层的权重
            "text_projection.weight": "text_projection",
        },
        # 定义嵌套字典，包含 transformer 的参数映射
        "transformer": {
            # 映射文本模型编码器层的前缀
            "text_model.encoder.layers.": "resblocks.",
            # 映射 transformer 的第一层归一化
            "layer_norm1": "ln_1",
            # 映射 transformer 的第二层归一化
            "layer_norm2": "ln_2",
            # 映射全连接层的第一部分
            ".fc1.": ".c_fc.",
            # 映射全连接层的第二部分
            ".fc2.": ".c_proj.",
            # 映射 transformer 最终层归一化的前缀
            "transformer.text_model.final_layer_norm.": "ln_final.",
            # 映射 transformer 中的标记嵌入层权重
            "transformer.text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            # 映射 transformer 中的位置信息嵌入层权重
            "transformer.text_model.embeddings.position_embedding.weight": "positional_embedding",
        },
    },
# 定义一个列表，用于存储需要忽略的文本编码器键
SD_2_TEXT_ENCODER_KEYS_TO_IGNORE = [
    # 忽略的特定权重和偏置参数
    "cond_stage_model.model.transformer.resblocks.23.attn.in_proj_bias",
    "cond_stage_model.model.transformer.resblocks.23.attn.in_proj_weight",
    "cond_stage_model.model.transformer.resblocks.23.attn.out_proj.bias",
    "cond_stage_model.model.transformer.resblocks.23.attn.out_proj.weight",
    "cond_stage_model.model.transformer.resblocks.23.ln_1.bias",
    "cond_stage_model.model.transformer.resblocks.23.ln_1.weight",
    "cond_stage_model.model.transformer.resblocks.23.ln_2.bias",
    "cond_stage_model.model.transformer.resblocks.23.ln_2.weight",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_fc.bias",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_fc.weight",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_proj.bias",
    "cond_stage_model.model.transformer.resblocks.23.mlp.c_proj.weight",
    "cond_stage_model.model.text_projection",
]

# 定义调度器的默认配置，支持遗留的参数类型
SCHEDULER_DEFAULT_CONFIG = {
    # β调度类型
    "beta_schedule": "scaled_linear",
    # 调度的起始值
    "beta_start": 0.00085,
    # 调度的结束值
    "beta_end": 0.012,
    # 插值类型
    "interpolation_type": "linear",
    # 训练时间步数
    "num_train_timesteps": 1000,
    # 预测类型
    "prediction_type": "epsilon",
    # 采样最大值
    "sample_max_value": 1.0,
    # 是否将 alpha 设置为 1
    "set_alpha_to_one": False,
    # 是否跳过 PRK 步骤
    "skip_prk_steps": True,
    # 时间步偏移
    "steps_offset": 1,
    # 时间步间隔类型
    "timestep_spacing": "leading",
}

# 定义包含 VAE 相关键的列表
LDM_VAE_KEYS = ["first_stage_model.", "vae."]
# 定义 VAE 默认缩放因子
LDM_VAE_DEFAULT_SCALING_FACTOR = 0.18215
# 定义 Playground VAE 缩放因子
PLAYGROUND_VAE_SCALING_FACTOR = 0.5
# 定义 LDM UNet 的键
LDM_UNET_KEY = "model.diffusion_model."
# 定义 LDM ControlNet 的键
LDM_CONTROLNET_KEY = "control_model."
# 定义要去除的 CLIP 前缀列表
LDM_CLIP_PREFIX_TO_REMOVE = [
    "cond_stage_model.transformer.",
    "conditioner.embedders.0.transformer.",
]
# 定义 LDM Open CLIP 文本投影维度
LDM_OPEN_CLIP_TEXT_PROJECTION_DIM = 1024
# 定义遗留调度器的关键字参数列表
SCHEDULER_LEGACY_KWARGS = ["prediction_type", "scheduler_type"]

# 定义有效的 URL 前缀列表
VALID_URL_PREFIXES = ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]

# 定义自定义异常类，用于单文件组件错误
class SingleFileComponentError(Exception):
    # 初始化异常类
    def __init__(self, message=None):
        self.message = message
        # 调用父类构造函数
        super().__init__(self.message)

# 定义验证 URL 的函数
def is_valid_url(url):
    # 解析 URL
    result = urlparse(url)
    # 检查是否有有效的方案和网络地址
    if result.scheme and result.netloc:
        return True

    # 返回无效
    return False

# 定义提取模型 ID 和权重名称的私有函数
def _extract_repo_id_and_weights_name(pretrained_model_name_or_path):
    # 检查提供的路径是否是有效 URL
    if not is_valid_url(pretrained_model_name_or_path):
        raise ValueError("Invalid `pretrained_model_name_or_path` provided. Please set it to a valid URL.")

    # 定义匹配模式
    pattern = r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)"
    weights_name = None
    repo_id = (None,)
    # 遍历有效的 URL 前缀进行替换
    for prefix in VALID_URL_PREFIXES:
        pretrained_model_name_or_path = pretrained_model_name_or_path.replace(prefix, "")
    # 使用正则表达式匹配模式
    match = re.match(pattern, pretrained_model_name_or_path)
    # 如果没有匹配，记录警告并返回
    if not match:
        logger.warning("Unable to identify the repo_id and weights_name from the provided URL.")
        return repo_id, weights_name

    # 提取 repo_id 和 weights_name
    repo_id = f"{match.group(1)}/{match.group(2)}"
    weights_name = match.group(3)

    # 返回提取的结果
    return repo_id, weights_name
# 检查模型权重是否在缓存文件夹中
def _is_model_weights_in_cached_folder(cached_folder, name):
    # 拼接缓存文件夹路径和模型名称，形成预训练模型的路径
    pretrained_model_name_or_path = os.path.join(cached_folder, name)
    # 初始化权重存在标志为 False
    weights_exist = False

    # 遍历可能的权重文件名
    for weights_name in [WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME]:
        # 检查指定路径下是否存在权重文件
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            # 如果存在，设置权重存在标志为 True
            weights_exist = True

    # 返回权重是否存在的标志
    return weights_exist


# 检查传入的关键字参数是否包含遗留调度器的关键字
def _is_legacy_scheduler_kwargs(kwargs):
    # 检查关键字参数中是否有遗留调度器的关键字
    return any(k in SCHEDULER_LEGACY_KWARGS for k in kwargs.keys())


# 加载单个文件的检查点
def load_single_file_checkpoint(
    pretrained_model_link_or_path,
    force_download=False,
    proxies=None,
    token=None,
    cache_dir=None,
    local_files_only=None,
    revision=None,
):
    # 检查给定路径是否是一个文件
    if os.path.isfile(pretrained_model_link_or_path):
        # 如果是文件，保持路径不变
        pretrained_model_link_or_path = pretrained_model_link_or_path

    else:
        # 如果不是文件，提取仓库 ID 和权重名称
        repo_id, weights_name = _extract_repo_id_and_weights_name(pretrained_model_link_or_path)
        # 获取模型文件的路径
        pretrained_model_link_or_path = _get_model_file(
            repo_id,
            weights_name=weights_name,
            force_download=force_download,
            cache_dir=cache_dir,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )

    # 加载状态字典
    checkpoint = load_state_dict(pretrained_model_link_or_path)

    # 一些检查点的模型状态字典可能在 "state_dict" 键下
    while "state_dict" in checkpoint:
        # 取出状态字典
        checkpoint = checkpoint["state_dict"]

    # 返回最终的检查点
    return checkpoint


# 获取原始配置
def fetch_original_config(original_config_file, local_files_only=False):
    # 检查给定的配置文件是否是一个有效的文件
    if os.path.isfile(original_config_file):
        # 如果是文件，读取其内容
        with open(original_config_file, "r") as fp:
            original_config_file = fp.read()

    elif is_valid_url(original_config_file):
        # 如果是有效的 URL
        if local_files_only:
            # 如果设置为只允许本地文件，抛出错误
            raise ValueError(
                "`local_files_only` is set to True, but a URL was provided as `original_config_file`. "
                "Please provide a valid local file path."
            )

        # 下载 URL 的内容并封装为字节流
        original_config_file = BytesIO(requests.get(original_config_file).content)

    else:
        # 如果既不是文件也不是有效的 URL，抛出错误
        raise ValueError("Invalid `original_config_file` provided. Please set it to a valid file path or URL.")

    # 解析 YAML 格式的原始配置
    original_config = yaml.safe_load(original_config_file)

    # 返回解析后的配置
    return original_config


# 检查给定的检查点是否为 CLIP 模型
def is_clip_model(checkpoint):
    # 检查检查点中是否包含 CLIP 模型的键
    if CHECKPOINT_KEY_NAMES["clip"] in checkpoint:
        return True

    # 返回 False
    return False


# 检查给定的检查点是否为 CLIP SDXL 模型
def is_clip_sdxl_model(checkpoint):
    # 检查检查点中是否包含 CLIP SDXL 模型的键
    if CHECKPOINT_KEY_NAMES["clip_sdxl"] in checkpoint:
        return True

    # 返回 False
    return False


# 检查给定的检查点是否为 CLIP SD3 模型
def is_clip_sd3_model(checkpoint):
    # 检查检查点中是否包含 CLIP SD3 模型的键
    if CHECKPOINT_KEY_NAMES["clip_sd3"] in checkpoint:
        return True

    # 返回 False
    return False


# 检查给定的检查点是否为 Open CLIP 模型
def is_open_clip_model(checkpoint):
    # 检查检查点中是否包含 Open CLIP 模型的键
    if CHECKPOINT_KEY_NAMES["open_clip"] in checkpoint:
        return True

    # 返回 False
    return False


# 检查给定的检查点是否为 Open CLIP SDXL 模型
def is_open_clip_sdxl_model(checkpoint):
    # 检查检查点中是否包含 Open CLIP SDXL 模型的键
    if CHECKPOINT_KEY_NAMES["open_clip_sdxl"] in checkpoint:
        return True

    # 返回 False
    return False


# 检查给定的检查点是否为 Open CLIP SD3 模型
def is_open_clip_sd3_model(checkpoint):
    # 检查检查点中是否包含特定的键
        if CHECKPOINT_KEY_NAMES["open_clip_sd3"] in checkpoint:
            # 如果找到特定键，返回 True
            return True
    
        # 如果没有找到特定键，返回 False
        return False
# 检查给定的检查点是否包含 OpenCLIP SDXL Refiner 模型的关键字
def is_open_clip_sdxl_refiner_model(checkpoint):
    # 如果检查点中包含指定的关键字，则返回 True
    if CHECKPOINT_KEY_NAMES["open_clip_sdxl_refiner"] in checkpoint:
        return True

    # 否则返回 False
    return False


# 检查给定的类对象是否与单个文件中的 CLIP 模型匹配
def is_clip_model_in_single_file(class_obj, checkpoint):
    # 检查检查点是否包含任何 CLIP 模型的关键字
    is_clip_in_checkpoint = any(
        [
            is_clip_model(checkpoint),  # 检查是否是 CLIP 模型
            is_clip_sd3_model(checkpoint),  # 检查是否是 SD3 模型
            is_open_clip_model(checkpoint),  # 检查是否是 OpenCLIP 模型
            is_open_clip_sdxl_model(checkpoint),  # 检查是否是 OpenCLIP SDXL 模型
            is_open_clip_sdxl_refiner_model(checkpoint),  # 检查是否是 OpenCLIP SDXL Refiner 模型
            is_open_clip_sd3_model(checkpoint),  # 检查是否是 OpenCLIP SD3 模型
        ]
    )
    # 如果类对象名称是 CLIPTextModel 或 CLIPTextModelWithProjection，并且检查点中存在 CLIP 模型
    if (
        class_obj.__name__ == "CLIPTextModel" or class_obj.__name__ == "CLIPTextModelWithProjection"
    ) and is_clip_in_checkpoint:
        return True  # 返回 True，表示匹配

    # 否则返回 False
    return False


# 推断给定检查点的 Diffusers 模型类型
def infer_diffusers_model_type(checkpoint):
    # 检查点中是否包含“inpainting”关键字，并且其形状的第二维为 9
    if (
        CHECKPOINT_KEY_NAMES["inpainting"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["inpainting"]].shape[1] == 9
    ):
        # 检查点中是否包含“v2”关键字，并且其形状的最后一维为 1024
        if CHECKPOINT_KEY_NAMES["v2"] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES["v2"]].shape[-1] == 1024:
            model_type = "inpainting_v2"  # 设置模型类型为 inpainting_v2
        else:
            model_type = "inpainting"  # 设置模型类型为 inpainting

    # 检查点中是否仅包含“v2”关键字，并且其形状的最后一维为 1024
    elif CHECKPOINT_KEY_NAMES["v2"] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES["v2"]].shape[-1] == 1024:
        model_type = "v2"  # 设置模型类型为 v2

    # 检查点中是否包含“playground-v2-5”关键字
    elif CHECKPOINT_KEY_NAMES["playground-v2-5"] in checkpoint:
        model_type = "playground-v2-5"  # 设置模型类型为 playground-v2-5

    # 检查点中是否包含“xl_base”关键字
    elif CHECKPOINT_KEY_NAMES["xl_base"] in checkpoint:
        model_type = "xl_base"  # 设置模型类型为 xl_base

    # 检查点中是否包含“xl_refiner”关键字
    elif CHECKPOINT_KEY_NAMES["xl_refiner"] in checkpoint:
        model_type = "xl_refiner"  # 设置模型类型为 xl_refiner

    # 检查点中是否包含“upscale”关键字
    elif CHECKPOINT_KEY_NAMES["upscale"] in checkpoint:
        model_type = "upscale"  # 设置模型类型为 upscale

    # 检查点中是否包含“controlnet”关键字
    elif CHECKPOINT_KEY_NAMES["controlnet"] in checkpoint:
        model_type = "controlnet"  # 设置模型类型为 controlnet

    # 检查点中是否包含“stable_cascade_stage_c”关键字，且其形状的第一维为 1536
    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"]].shape[0] == 1536
    ):
        model_type = "stable_cascade_stage_c_lite"  # 设置模型类型为 stable_cascade_stage_c_lite

    # 检查点中是否包含“stable_cascade_stage_c”关键字，且其形状的第一维为 2048
    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"]].shape[0] == 2048
    ):
        model_type = "stable_cascade_stage_c"  # 设置模型类型为 stable_cascade_stage_c

    # 检查点中是否包含“stable_cascade_stage_b”关键字，且其形状的最后一维为 576
    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"]].shape[-1] == 576
    ):
        model_type = "stable_cascade_stage_b_lite"  # 设置模型类型为 stable_cascade_stage_b_lite

    # 检查点中是否包含“stable_cascade_stage_b”关键字，且其形状的最后一维为 640
    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"] in checkpoint
        and checkpoint[CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"]].shape[-1] == 640
    ):
        model_type = "stable_cascade_stage_b"  # 设置模型类型为 stable_cascade_stage_b

    # 检查点中是否包含“sd3”关键字
    elif CHECKPOINT_KEY_NAMES["sd3"] in checkpoint:
        model_type = "sd3"  # 设置模型类型为 sd3
    # 检查 checkpoint 中是否包含 "animatediff" 的键
    elif CHECKPOINT_KEY_NAMES["animatediff"] in checkpoint:
        # 检查 checkpoint 中是否包含 "animatediff_scribble" 的键
        if CHECKPOINT_KEY_NAMES["animatediff_scribble"] in checkpoint:
            # 设置模型类型为 "animatediff_scribble"
            model_type = "animatediff_scribble"

        # 检查 checkpoint 中是否包含 "animatediff_rgb" 的键
        elif CHECKPOINT_KEY_NAMES["animatediff_rgb"] in checkpoint:
            # 设置模型类型为 "animatediff_rgb"
            model_type = "animatediff_rgb"

        # 检查 checkpoint 中是否包含 "animatediff_v2" 的键
        elif CHECKPOINT_KEY_NAMES["animatediff_v2"] in checkpoint:
            # 设置模型类型为 "animatediff_v2"
            model_type = "animatediff_v2"

        # 检查 checkpoint 中 "animatediff_sdxl_beta" 的形状最后一维是否为 320
        elif checkpoint[CHECKPOINT_KEY_NAMES["animatediff_sdxl_beta"]].shape[-1] == 320:
            # 设置模型类型为 "animatediff_sdxl_beta"
            model_type = "animatediff_sdxl_beta"

        # 检查 checkpoint 中 "animatediff" 的形状第二维是否为 24
        elif checkpoint[CHECKPOINT_KEY_NAMES["animatediff"]].shape[1] == 24:
            # 设置模型类型为 "animatediff_v1"
            model_type = "animatediff_v1"

        # 以上条件都不满足时
        else:
            # 设置模型类型为 "animatediff_v3"
            model_type = "animatediff_v3"

    # 检查 checkpoint 中是否包含 "flux" 相关的任意键
    elif any(key in checkpoint for key in CHECKPOINT_KEY_NAMES["flux"]):
        # 检查 checkpoint 中是否包含特定的权重偏置键
        if any(
            g in checkpoint for g in ["guidance_in.in_layer.bias", "model.diffusion_model.guidance_in.in_layer.bias"]
        ):
            # 设置模型类型为 "flux-dev"
            model_type = "flux-dev"
        # 如果不包含特定的权重偏置键
        else:
            # 设置模型类型为 "flux-schnell"
            model_type = "flux-schnell"
    # 以上条件都不满足时
    else:
        # 设置模型类型为 "v1"
        model_type = "v1"

    # 返回确定的模型类型
    return model_type
# 根据检查点获取 diffuser 配置
def fetch_diffusers_config(checkpoint):
    # 推断模型类型
    model_type = infer_diffusers_model_type(checkpoint)
    # 从默认路径获取模型路径
    model_path = DIFFUSERS_DEFAULT_PIPELINE_PATHS[model_type]

    # 返回模型路径
    return model_path


# 设置图像大小，如果未提供，则基于检查点推断
def set_image_size(checkpoint, image_size=None):
    # 如果提供了图像大小，直接返回
    if image_size:
        return image_size

    # 推断模型类型
    model_type = infer_diffusers_model_type(checkpoint)
    # 根据模型类型获取默认图像大小
    image_size = DIFFUSERS_TO_LDM_DEFAULT_IMAGE_SIZE_MAP[model_type]

    # 返回图像大小
    return image_size


# 从检查点转换卷积注意力为线性形式
def conv_attn_to_linear(checkpoint):
    # 获取检查点中的所有键
    keys = list(checkpoint.keys())
    # 定义需要转换的注意力权重键
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    # 遍历每个键
    for key in keys:
        # 如果键属于注意力权重
        if ".".join(key.split(".")[-2:]) in attn_keys:
            # 如果权重维度大于2，则只保留第一维
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        # 如果键是投影注意力权重
        elif "proj_attn.weight" in key:
            # 如果权重维度大于2，则只保留第一维
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


# 从 LDM 创建 UNet diffuser 配置
def create_unet_diffusers_config_from_ldm(
    original_config, checkpoint, image_size=None, upcast_attention=None, num_in_channels=None
):
    """
    基于 LDM 模型配置创建 diffuser 配置。
    """
    # 如果提供了图像大小，记录弃用信息
    if image_size is not None:
        deprecation_message = (
            "Configuring UNet2DConditionModel with the `image_size` argument to `from_single_file`"
            "is deprecated and will be ignored in future versions."
        )
        # 调用弃用警告函数
        deprecate("image_size", "1.0.0", deprecation_message)

    # 设置图像大小
    image_size = set_image_size(checkpoint, image_size=image_size)

    # 获取 UNet 参数，如果存在 unet_config
    if (
        "unet_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["unet_config"] is not None
    ):
        unet_params = original_config["model"]["params"]["unet_config"]["params"]
    else:
        # 否则从网络配置中获取
        unet_params = original_config["model"]["params"]["network_config"]["params"]

    # 如果提供了输入通道数，记录弃用信息
    if num_in_channels is not None:
        deprecation_message = (
            "Configuring UNet2DConditionModel with the `num_in_channels` argument to `from_single_file`"
            "is deprecated and will be ignored in future versions."
        )
        # 调用弃用警告函数
        deprecate("image_size", "1.0.0", deprecation_message)
        # 设置输入通道数
        in_channels = num_in_channels
    else:
        # 否则从 UNet 参数中获取输入通道数
        in_channels = unet_params["in_channels"]

    # 获取 VAE 参数
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    # 计算每个块的输出通道数
    block_out_channels = [unet_params["model_channels"] * mult for mult in unet_params["channel_mult"]]

    # 定义向下块的类型
    down_block_types = []
    resolution = 1
    # 遍历每个输出通道块
    for i in range(len(block_out_channels)):
        # 根据分辨率选择块类型
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params["attention_resolutions"] else "DownBlock2D"
        down_block_types.append(block_type)
        # 更新分辨率
        if i != len(block_out_channels) - 1:
            resolution *= 2

    # 定义向上块的类型
    up_block_types = []
    # 遍历输出通道的数量
    for i in range(len(block_out_channels)):
        # 根据分辨率判断块类型，选择跨注意力块或上采样块
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params["attention_resolutions"] else "UpBlock2D"
        # 将块类型添加到上采样块列表中
        up_block_types.append(block_type)
        # 更新分辨率，进行下一个块的处理
        resolution //= 2

    # 检查是否设置了变换器的深度
    if unet_params["transformer_depth"] is not None:
        # 获取每个块的变换器层数，支持整型或列表
        transformer_layers_per_block = (
            unet_params["transformer_depth"]
            if isinstance(unet_params["transformer_depth"], int)
            else list(unet_params["transformer_depth"])
        )
    else:
        # 如果没有设置，默认每块只有一层
        transformer_layers_per_block = 1

    # 计算 VAE 的缩放因子
    vae_scale_factor = 2 ** (len(vae_params["ch_mult"]) - 1)

    # 获取头部维度，如果存在的话
    head_dim = unet_params["num_heads"] if "num_heads" in unet_params else None
    # 检查是否使用线性投影
    use_linear_projection = (
        unet_params["use_linear_in_transformer"] if "use_linear_in_transformer" in unet_params else False
    )
    # 如果使用线性投影
    if use_linear_projection:
        # 针对稳定扩散 2 的特定配置
        if head_dim is None:
            # 计算头部维度的乘数
            head_dim_mult = unet_params["model_channels"] // unet_params["num_head_channels"]
            # 根据通道乘数生成头部维度列表
            head_dim = [head_dim_mult * c for c in list(unet_params["channel_mult"])]

    # 初始化额外的嵌入类型和维度
    class_embed_type = None
    addition_embed_type = None
    addition_time_embed_dim = None
    projection_class_embeddings_input_dim = None
    context_dim = None

    # 如果上下文维度存在
    if unet_params["context_dim"] is not None:
        # 获取上下文维度，支持整型或列表
        context_dim = (
            unet_params["context_dim"]
            if isinstance(unet_params["context_dim"], int)
            else unet_params["context_dim"][0]
        )

    # 检查类别数量设置
    if "num_classes" in unet_params:
        # 如果类别为顺序
        if unet_params["num_classes"] == "sequential":
            # 根据上下文维度决定额外嵌入类型
            if context_dim in [2048, 1280]:
                # 针对 SDXL 的配置
                addition_embed_type = "text_time"
                addition_time_embed_dim = 256
            else:
                # 其他情况下使用投影嵌入
                class_embed_type = "projection"
            # 确保包含 ADM 输入通道
            assert "adm_in_channels" in unet_params
            # 获取投影嵌入的输入维度
            projection_class_embeddings_input_dim = unet_params["adm_in_channels"]

    # 配置字典，包含各种模型参数
    config = {
        # 计算样本大小
        "sample_size": image_size // vae_scale_factor,
        # 输入通道数
        "in_channels": in_channels,
        # 各个下采样块类型
        "down_block_types": down_block_types,
        # 输出通道数量
        "block_out_channels": block_out_channels,
        # 每个块的层数
        "layers_per_block": unet_params["num_res_blocks"],
        # 上下文维度
        "cross_attention_dim": context_dim,
        # 注意力头的维度
        "attention_head_dim": head_dim,
        # 是否使用线性投影
        "use_linear_projection": use_linear_projection,
        # 类别嵌入类型
        "class_embed_type": class_embed_type,
        # 额外嵌入类型
        "addition_embed_type": addition_embed_type,
        # 额外时间嵌入维度
        "addition_time_embed_dim": addition_time_embed_dim,
        # 投影类别嵌入的输入维度
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        # 每个块的变换器层数
        "transformer_layers_per_block": transformer_layers_per_block,
    }
    # 检查是否提供了 upcast_attention 参数
    if upcast_attention is not None:
        # 构造弃用提示信息，告知用户该参数在未来版本中将被忽略
        deprecation_message = (
            "Configuring UNet2DConditionModel with the `upcast_attention` argument to `from_single_file`"
            "is deprecated and will be ignored in future versions."
        )
        # 调用 deprecate 函数，记录该参数的弃用信息
        deprecate("image_size", "1.0.0", deprecation_message)
        # 将 upcast_attention 参数存储到 config 字典中
        config["upcast_attention"] = upcast_attention

    # 检查 unet_params 中是否包含 disable_self_attentions 键
    if "disable_self_attentions" in unet_params:
        # 如果存在，设置 config 字典中的 only_cross_attention 为对应值
        config["only_cross_attention"] = unet_params["disable_self_attentions"]

    # 检查 unet_params 中是否包含 num_classes 键且其值为整数
    if "num_classes" in unet_params and isinstance(unet_params["num_classes"], int):
        # 将 num_classes 的值存储到 config 字典中的 num_class_embeds 键
        config["num_class_embeds"] = unet_params["num_classes"]

    # 将 unet_params 中的 out_channels 值存储到 config 字典中
    config["out_channels"] = unet_params["out_channels"]
    # 将 up_block_types 存储到 config 字典中
    config["up_block_types"] = up_block_types

    # 返回配置字典
    return config
# 从 LDM 模型的配置创建 ControlNet 的 Diffusers 配置
def create_controlnet_diffusers_config_from_ldm(original_config, checkpoint, image_size=None, **kwargs):
    # 检查 image_size 参数是否提供
    if image_size is not None:
        # 创建弃用提示信息
        deprecation_message = (
            "Configuring ControlNetModel with the `image_size` argument"
            "is deprecated and will be ignored in future versions."
        )
        # 调用 deprecate 函数记录弃用信息
        deprecate("image_size", "1.0.0", deprecation_message)

    # 设置 image_size，使用检查点中的值
    image_size = set_image_size(checkpoint, image_size=image_size)

    # 从原始配置中提取 UNet 参数
    unet_params = original_config["model"]["params"]["control_stage_config"]["params"]
    # 创建 Diffusers 的 UNet 配置
    diffusers_unet_config = create_unet_diffusers_config_from_ldm(original_config, image_size=image_size)

    # 构建 ControlNet 配置字典
    controlnet_config = {
        "conditioning_channels": unet_params["hint_channels"],
        "in_channels": diffusers_unet_config["in_channels"],
        "down_block_types": diffusers_unet_config["down_block_types"],
        "block_out_channels": diffusers_unet_config["block_out_channels"],
        "layers_per_block": diffusers_unet_config["layers_per_block"],
        "cross_attention_dim": diffusers_unet_config["cross_attention_dim"],
        "attention_head_dim": diffusers_unet_config["attention_head_dim"],
        "use_linear_projection": diffusers_unet_config["use_linear_projection"],
        "class_embed_type": diffusers_unet_config["class_embed_type"],
        "addition_embed_type": diffusers_unet_config["addition_embed_type"],
        "addition_time_embed_dim": diffusers_unet_config["addition_time_embed_dim"],
        "projection_class_embeddings_input_dim": diffusers_unet_config["projection_class_embeddings_input_dim"],
        "transformer_layers_per_block": diffusers_unet_config["transformer_layers_per_block"],
    }

    # 返回构建好的 ControlNet 配置
    return controlnet_config


# 从 LDM 模型的配置创建 VAE 的 Diffusers 配置
def create_vae_diffusers_config_from_ldm(original_config, checkpoint, image_size=None, scaling_factor=None):
    """
    根据 LDM 模型的配置创建 Diffusers 配置。
    """
    # 检查 image_size 参数是否提供
    if image_size is not None:
        # 创建弃用提示信息
        deprecation_message = (
            "Configuring AutoencoderKL with the `image_size` argument"
            "is deprecated and will be ignored in future versions."
        )
        # 调用 deprecate 函数记录弃用信息
        deprecate("image_size", "1.0.0", deprecation_message)

    # 设置 image_size，使用检查点中的值
    image_size = set_image_size(checkpoint, image_size=image_size)

    # 检查检查点中是否包含 edm_mean 和 edm_std
    if "edm_mean" in checkpoint and "edm_std" in checkpoint:
        # 提取潜变量的均值
        latents_mean = checkpoint["edm_mean"]
        # 提取潜变量的标准差
        latents_std = checkpoint["edm_std"]
    else:
        # 如果未提供，设置为 None
        latents_mean = None
        latents_std = None

    # 从原始配置中提取 VAE 参数
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    # 根据条件设置缩放因子
    if (scaling_factor is None) and (latents_mean is not None) and (latents_std is not None):
        scaling_factor = PLAYGROUND_VAE_SCALING_FACTOR

    elif (scaling_factor is None) and ("scale_factor" in original_config["model"]["params"]):
        scaling_factor = original_config["model"]["params"]["scale_factor"]

    elif scaling_factor is None:
        scaling_factor = LDM_VAE_DEFAULT_SCALING_FACTOR
    # 计算每个块的输出通道数，乘以相应的倍数
        block_out_channels = [vae_params["ch"] * mult for mult in vae_params["ch_mult"]]
        # 生成与输出通道数相同数量的下采样块类型
        down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
        # 生成与输出通道数相同数量的上采样块类型
        up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)
    
        # 创建配置字典，存储各类参数
        config = {
            # 图像样本大小
            "sample_size": image_size,
            # 输入通道数
            "in_channels": vae_params["in_channels"],
            # 输出通道数
            "out_channels": vae_params["out_ch"],
            # 下采样块类型列表
            "down_block_types": down_block_types,
            # 上采样块类型列表
            "up_block_types": up_block_types,
            # 块的输出通道数
            "block_out_channels": block_out_channels,
            # 潜在通道数
            "latent_channels": vae_params["z_channels"],
            # 每个块的层数
            "layers_per_block": vae_params["num_res_blocks"],
            # 缩放因子
            "scaling_factor": scaling_factor,
        }
        # 如果潜在均值和标准差不为 None，更新配置字典
        if latents_mean is not None and latents_std is not None:
            config.update({"latents_mean": latents_mean, "latents_std": latents_std})
    
        # 返回配置字典
        return config
# 更新 UNet 中 ResNet 结构的 LDM 到 Diffusers 格式
def update_unet_resnet_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping=None):
    # 遍历所有 LDM 键
    for ldm_key in ldm_keys:
        # 根据规则替换 LDM 键为对应的 Diffusers 键
        diffusers_key = (
            ldm_key.replace("in_layers.0", "norm1")
            .replace("in_layers.2", "conv1")
            .replace("out_layers.0", "norm2")
            .replace("out_layers.3", "conv2")
            .replace("emb_layers.1", "time_emb_proj")
            .replace("skip_connection", "conv_shortcut")
        )
        # 如果有映射，则替换旧键为新键
        if mapping:
            diffusers_key = diffusers_key.replace(mapping["old"], mapping["new"])
        # 从 checkpoint 获取 LDM 键的数据并存入新的 checkpoint
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)


# 更新 UNet 中注意力结构的 LDM 到 Diffusers 格式
def update_unet_attention_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping):
    # 遍历所有 LDM 键
    for ldm_key in ldm_keys:
        # 根据映射替换 LDM 键为对应的 Diffusers 键
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"])
        # 从 checkpoint 获取 LDM 键的数据并存入新的 checkpoint
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)


# 更新 VAE 中 ResNet 结构的 LDM 到 Diffusers 格式
def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    # 遍历所有 LDM 键
    for ldm_key in keys:
        # 根据映射替换 LDM 键，并替换特定的 nin_shortcut
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"]).replace("nin_shortcut", "conv_shortcut")
        # 从 checkpoint 获取 LDM 键的数据并存入新的 checkpoint
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)


# 更新 VAE 中注意力结构的 LDM 到 Diffusers 格式
def update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    # 遍历所有 LDM 键
    for ldm_key in keys:
        # 根据映射替换 LDM 键为对应的 Diffusers 键，并进行多个字段的替换
        diffusers_key = (
            ldm_key.replace(mapping["old"], mapping["new"])
            .replace("norm.weight", "group_norm.weight")
            .replace("norm.bias", "group_norm.bias")
            .replace("q.weight", "to_q.weight")
            .replace("q.bias", "to_q.bias")
            .replace("k.weight", "to_k.weight")
            .replace("k.bias", "to_k.bias")
            .replace("v.weight", "to_v.weight")
            .replace("v.bias", "to_v.bias")
            .replace("proj_out.weight", "to_out.0.weight")
            .replace("proj_out.bias", "to_out.0.bias")
        )
        # 从 checkpoint 获取 LDM 键的数据并存入新的 checkpoint
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)

        # proj_attn.weight 需要从一维卷积转换为线性
        shape = new_checkpoint[diffusers_key].shape

        # 如果形状为三维，截取第一个维度
        if len(shape) == 3:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0]
        # 如果形状为四维，截取前两个维度
        elif len(shape) == 4:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0, 0]


# 转换稳定级联 UNet 单文件到 Diffusers 格式
def convert_stable_cascade_unet_single_file_to_diffusers(checkpoint, **kwargs):
    # 检查是否包含特定的权重
    is_stage_c = "clip_txt_mapper.weight" in checkpoint
    # 检查是否处于阶段 C
    if is_stage_c:
        # 初始化一个空字典，用于存储状态
        state_dict = {}
        # 遍历检查点中的所有键
        for key in checkpoint.keys():
            # 如果键以 "in_proj_weight" 结尾
            if key.endswith("in_proj_weight"):
                # 将权重分块为三个部分
                weights = checkpoint[key].chunk(3, 0)
                # 替换键名并保存对应的权重到字典
                state_dict[key.replace("attn.in_proj_weight", "to_q.weight")] = weights[0]
                state_dict[key.replace("attn.in_proj_weight", "to_k.weight")] = weights[1]
                state_dict[key.replace("attn.in_proj_weight", "to_v.weight")] = weights[2]
            # 如果键以 "in_proj_bias" 结尾
            elif key.endswith("in_proj_bias"):
                # 将偏置分块为三个部分
                weights = checkpoint[key].chunk(3, 0)
                # 替换键名并保存对应的偏置到字典
                state_dict[key.replace("attn.in_proj_bias", "to_q.bias")] = weights[0]
                state_dict[key.replace("attn.in_proj_bias", "to_k.bias")] = weights[1]
                state_dict[key.replace("attn.in_proj_bias", "to_v.bias")] = weights[2]
            # 如果键以 "out_proj.weight" 结尾
            elif key.endswith("out_proj.weight"):
                # 获取权重
                weights = checkpoint[key]
                # 替换键名并保存对应的权重到字典
                state_dict[key.replace("attn.out_proj.weight", "to_out.0.weight")] = weights
            # 如果键以 "out_proj.bias" 结尾
            elif key.endswith("out_proj.bias"):
                # 获取偏置
                weights = checkpoint[key]
                # 替换键名并保存对应的偏置到字典
                state_dict[key.replace("attn.out_proj.bias", "to_out.0.bias")] = weights
            # 对于其它情况，直接保存键值对
            else:
                state_dict[key] = checkpoint[key]
    # 如果不在阶段 C
    else:
        # 初始化一个空字典，用于存储状态
        state_dict = {}
        # 遍历检查点中的所有键
        for key in checkpoint.keys():
            # 如果键以 "in_proj_weight" 结尾
            if key.endswith("in_proj_weight"):
                # 将权重分块为三个部分
                weights = checkpoint[key].chunk(3, 0)
                # 替换键名并保存对应的权重到字典
                state_dict[key.replace("attn.in_proj_weight", "to_q.weight")] = weights[0]
                state_dict[key.replace("attn.in_proj_weight", "to_k.weight")] = weights[1]
                state_dict[key.replace("attn.in_proj_weight", "to_v.weight")] = weights[2]
            # 如果键以 "in_proj_bias" 结尾
            elif key.endswith("in_proj_bias"):
                # 将偏置分块为三个部分
                weights = checkpoint[key].chunk(3, 0)
                # 替换键名并保存对应的偏置到字典
                state_dict[key.replace("attn.in_proj_bias", "to_q.bias")] = weights[0]
                state_dict[key.replace("attn.in_proj_bias", "to_k.bias")] = weights[1]
                state_dict[key.replace("attn.in_proj_bias", "to_v.bias")] = weights[2]
            # 如果键以 "out_proj.weight" 结尾
            elif key.endswith("out_proj.weight"):
                # 获取权重
                weights = checkpoint[key]
                # 替换键名并保存对应的权重到字典
                state_dict[key.replace("attn.out_proj.weight", "to_out.0.weight")] = weights
            # 如果键以 "out_proj.bias" 结尾
            elif key.endswith("out_proj.bias"):
                # 获取偏置
                weights = checkpoint[key]
                # 替换键名并保存对应的偏置到字典
                state_dict[key.replace("attn.out_proj.bias", "to_out.0.bias")] = weights
            # 如果键以 "clip_mapper.weight" 结尾
            elif key.endswith("clip_mapper.weight"):
                # 获取权重
                weights = checkpoint[key]
                # 替换键名并保存对应的权重到字典
                state_dict[key.replace("clip_mapper.weight", "clip_txt_pooled_mapper.weight")] = weights
            # 如果键以 "clip_mapper.bias" 结尾
            elif key.endswith("clip_mapper.bias"):
                # 获取偏置
                weights = checkpoint[key]
                # 替换键名并保存对应的偏置到字典
                state_dict[key.replace("clip_mapper.bias", "clip_txt_pooled_mapper.bias")] = weights
            # 对于其它情况，直接保存键值对
            else:
                state_dict[key] = checkpoint[key]

    # 返回构建好的状态字典
    return state_dict
# 转换 LDM UNet 检查点，接受检查点和配置，并返回转换后的检查点
def convert_ldm_unet_checkpoint(checkpoint, config, extract_ema=False, **kwargs):
    """
    接受状态字典和配置，并返回转换后的检查点。
    """
    # 创建一个字典用于提取 UNet 的状态字典
    unet_state_dict = {}
    # 获取检查点的所有键
    keys = list(checkpoint.keys())
    # 定义 UNet 的关键字
    unet_key = LDM_UNET_KEY

    # 检查有多少参数以 `model_ema` 开头，以确定是否为 EMA 检查点
    if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
        # 记录警告：检查点包含 EMA 和非 EMA 权重
        logger.warning("Checkpoint has both EMA and non-EMA weights.")
        # 记录警告：仅提取 EMA 权重
        logger.warning(
            "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
            " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
        )
        # 遍历所有键
        for key in keys:
            # 如果键以 "model.diffusion_model" 开头
            if key.startswith("model.diffusion_model"):
                # 替换键前缀并获取对应的 EMA 权重
                flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.get(flat_ema_key)
    else:
        # 如果存在 EMA 权重，但不提取 EMA
        if sum(k.startswith("model_ema") for k in keys) > 100:
            # 记录警告：仅提取非 EMA 权重
            logger.warning(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )
        # 遍历所有键
        for key in keys:
            # 如果键以 UNet 的关键字开头
            if key.startswith(unet_key):
                # 将对应的权重添加到 UNet 状态字典中
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.get(key)

    # 创建一个新的检查点字典
    new_checkpoint = {}
    # 获取 UNet 图层的映射键
    ldm_unet_keys = DIFFUSERS_TO_LDM_MAPPING["unet"]["layers"]
    # 遍历 Diffusers 和 LDM 键的映射
    for diffusers_key, ldm_key in ldm_unet_keys.items():
        # 如果 LDM 键不在 UNet 状态字典中，则跳过
        if ldm_key not in unet_state_dict:
            continue
        # 将 UNet 状态字典中的权重添加到新的检查点
        new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]

    # 如果配置中存在 class_embed_type 且其值为 "timestep" 或 "projection"
    if ("class_embed_type" in config) and (config["class_embed_type"] in ["timestep", "projection"]):
        # 获取对应的 class_embed 键映射
        class_embed_keys = DIFFUSERS_TO_LDM_MAPPING["unet"]["class_embed_type"]
        # 遍历并添加 class_embed 的权重到新的检查点
        for diffusers_key, ldm_key in class_embed_keys.items():
            new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]

    # 如果配置中存在 addition_embed_type 且其值为 "text_time"
    if ("addition_embed_type" in config) and (config["addition_embed_type"] == "text_time"):
        # 获取对应的 addition_embed 键映射
        addition_embed_keys = DIFFUSERS_TO_LDM_MAPPING["unet"]["addition_embed_type"]
        # 遍历并添加 addition_embed 的权重到新的检查点
        for diffusers_key, ldm_key in addition_embed_keys.items():
            new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]

    # 与 StableDiffusionUpscalePipeline 相关
    if "num_class_embeds" in config:
        # 检查 num_class_embeds 是否不为空且 UNet 状态字典中存在 label_emb.weight
        if (config["num_class_embeds"] is not None) and ("label_emb.weight" in unet_state_dict):
            # 将 label_emb.weight 添加到新的检查点中
            new_checkpoint["class_embedding.weight"] = unet_state_dict["label_emb.weight"]

    # 仅获取输入块的键
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    # 创建一个字典，存储每个输入块的相关键
    input_blocks = {
        # 遍历每个输入块的层ID，生成对应的键列表
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # 获取中间块的数量
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    # 创建一个字典，存储每个中间块的相关键
    middle_blocks = {
        # 遍历每个中间块的层ID，生成对应的键列表
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # 获取输出块的数量
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    # 创建一个字典，存储每个输出块的相关键
    output_blocks = {
        # 遍历每个输出块的层ID，生成对应的键列表
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    # 处理输入块
    for i in range(1, num_input_blocks):
        # 计算当前块的ID
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        # 计算当前层在块内的ID
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        # 找到当前输入块的残差连接
        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        # 更新 UNet 残差连接到新的检查点
        update_unet_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            unet_state_dict,
            {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"},
        )

        # 如果当前输入块的权重存在，则将其更新到新的检查点
        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.get(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.get(
                f"input_blocks.{i}.0.op.bias"
            )

        # 找到当前输入块的注意力连接
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]
        # 如果注意力连接存在，更新它到新的检查点
        if attentions:
            update_unet_attention_ldm_to_diffusers(
                attentions,
                new_checkpoint,
                unet_state_dict,
                {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"},
            )

    # 处理中间块
    for key in middle_blocks.keys():
        # 计算对应的 diffusers 键
        diffusers_key = max(key - 1, 0)
        # 如果是偶数层，更新残差连接
        if key % 2 == 0:
            update_unet_resnet_ldm_to_diffusers(
                middle_blocks[key],
                new_checkpoint,
                unet_state_dict,
                mapping={"old": f"middle_block.{key}", "new": f"mid_block.resnets.{diffusers_key}"},
            )
        # 如果是奇数层，更新注意力连接
        else:
            update_unet_attention_ldm_to_diffusers(
                middle_blocks[key],
                new_checkpoint,
                unet_state_dict,
                mapping={"old": f"middle_block.{key}", "new": f"mid_block.attentions.{diffusers_key}"},
            )

    # 处理上升块
    # 遍历输出块的数量
    for i in range(num_output_blocks):
        # 计算当前块的 ID
        block_id = i // (config["layers_per_block"] + 1)
        # 计算当前层在块中的 ID
        layer_in_block_id = i % (config["layers_per_block"] + 1)

        # 筛选当前输出块中的 ResNet 相关键，排除特定的操作键
        resnets = [
            key for key in output_blocks[i] if f"output_blocks.{i}.0" in key and f"output_blocks.{i}.0.op" not in key
        ]
        # 更新 U-Net 中 ResNet 的状态字典，映射旧的键到新的键
        update_unet_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            unet_state_dict,
            {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"},
        )

        # 筛选当前输出块中的注意力相关键，排除特定的卷积键
        attentions = [
            key for key in output_blocks[i] if f"output_blocks.{i}.1" in key and f"output_blocks.{i}.1.conv" not in key
        ]
        # 如果找到注意力键，则更新状态字典
        if attentions:
            update_unet_attention_ldm_to_diffusers(
                attentions,
                new_checkpoint,
                unet_state_dict,
                {"old": f"output_blocks.{i}.1", "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}"},
            )

        # 如果在状态字典中找到当前卷积层的权重，则更新新的检查点字典
        if f"output_blocks.{i}.1.conv.weight" in unet_state_dict:
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                f"output_blocks.{i}.1.conv.weight"
            ]
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                f"output_blocks.{i}.1.conv.bias"
            ]
        # 如果在状态字典中找到下一个卷积层的权重，则更新新的检查点字典
        if f"output_blocks.{i}.2.conv.weight" in unet_state_dict:
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                f"output_blocks.{i}.2.conv.weight"
            ]
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                f"output_blocks.{i}.2.conv.bias"
            ]

    # 返回更新后的检查点字典
    return new_checkpoint
# 定义一个函数，用于转换 ControlNet 的检查点文件
def convert_controlnet_checkpoint(
    checkpoint,  # 输入的检查点数据
    config,  # 配置参数
    **kwargs,  # 额外的关键字参数
):
    # 检查点中如果包含时间嵌入权重，则将其直接赋值
    if "time_embed.0.weight" in checkpoint:
        controlnet_state_dict = checkpoint
    # 否则，初始化空的状态字典
    else:
        controlnet_state_dict = {}
        keys = list(checkpoint.keys())  # 获取检查点的所有键
        controlnet_key = LDM_CONTROLNET_KEY  # 定义 ControlNet 的关键字
        # 遍历检查点中的所有键
        for key in keys:
            # 如果键以 ControlNet 的关键字开头，则提取相关数据
            if key.startswith(controlnet_key):
                controlnet_state_dict[key.replace(controlnet_key, "")] = checkpoint.get(key)

    new_checkpoint = {}  # 初始化新的检查点字典
    ldm_controlnet_keys = DIFFUSERS_TO_LDM_MAPPING["controlnet"]["layers"]  # 获取 LDM 和 Diffusers 的映射
    # 将 Diffusers 的键映射到新的检查点
    for diffusers_key, ldm_key in ldm_controlnet_keys.items():
        if ldm_key not in controlnet_state_dict:  # 如果 LDM 键不存在，则跳过
            continue
        new_checkpoint[diffusers_key] = controlnet_state_dict[ldm_key]  # 添加映射数据到新检查点

    # 仅检索输入块的键
    num_input_blocks = len(
        {".".join(layer.split(".")[:2]) for layer in controlnet_state_dict if "input_blocks" in layer}
    )  # 计算输入块的数量
    # 创建输入块字典
    input_blocks = {
        layer_id: [key for key in controlnet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # 处理下块
    for i in range(1, num_input_blocks):  # 从第一个输入块开始处理
        block_id = (i - 1) // (config["layers_per_block"] + 1)  # 计算当前块的ID
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)  # 计算当前层在块中的ID

        # 获取当前块中的所有 ResNet
        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        # 更新 UNet 中 ResNet 的映射
        update_unet_resnet_ldm_to_diffusers(
            resnets,  # ResNet 列表
            new_checkpoint,  # 新检查点
            controlnet_state_dict,  # 原状态字典
            {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"},
        )

        # 如果有权重数据，则映射到新的检查点
        if f"input_blocks.{i}.0.op.weight" in controlnet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = controlnet_state_dict.get(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = controlnet_state_dict.get(
                f"input_blocks.{i}.0.op.bias"
            )

        # 获取当前块中的所有注意力层
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]
        if attentions:  # 如果存在注意力层，则进行更新
            update_unet_attention_ldm_to_diffusers(
                attentions,
                new_checkpoint,
                controlnet_state_dict,
                {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"},
            )

    # 处理 ControlNet 的下块
    for i in range(num_input_blocks):
        # 将零卷积的权重映射到新的检查点
        new_checkpoint[f"controlnet_down_blocks.{i}.weight"] = controlnet_state_dict.get(f"zero_convs.{i}.0.weight")
        new_checkpoint[f"controlnet_down_blocks.{i}.bias"] = controlnet_state_dict.get(f"zero_convs.{i}.0.bias")
    # 仅检索中间块的键
    num_middle_blocks = len(
        # 从 controlnet_state_dict 中提取包含 'middle_block' 的层的前两个部分，去重后形成集合
        {".".join(layer.split(".")[:2]) for layer in controlnet_state_dict if "middle_block" in layer}
    )
    # 为每个中间块的层 ID 创建一个字典，映射到对应的控制网络状态字典中的键列表
    middle_blocks = {
        layer_id: [key for key in controlnet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }
    
    # 遍历中间块的键
    for key in middle_blocks.keys():
        # 获取前一个块的索引，确保不为负数
        diffusers_key = max(key - 1, 0)
        # 如果键是偶数，调用更新函数处理 ResNet
        if key % 2 == 0:
            update_unet_resnet_ldm_to_diffusers(
                middle_blocks[key],
                new_checkpoint,
                controlnet_state_dict,
                # 映射旧的和新的层名称
                mapping={"old": f"middle_block.{key}", "new": f"mid_block.resnets.{diffusers_key}"},
            )
        # 如果键是奇数，调用更新函数处理 Attention
        else:
            update_unet_attention_ldm_to_diffusers(
                middle_blocks[key],
                new_checkpoint,
                controlnet_state_dict,
                # 映射旧的和新的层名称
                mapping={"old": f"middle_block.{key}", "new": f"mid_block.attentions.{diffusers_key}"},
            )
    
    # 处理中间块的输出权重和偏差
    new_checkpoint["controlnet_mid_block.weight"] = controlnet_state_dict.get("middle_block_out.0.weight")
    new_checkpoint["controlnet_mid_block.bias"] = controlnet_state_dict.get("middle_block_out.0.bias")
    
    # 控制网络条件嵌入块
    cond_embedding_blocks = {
        # 提取包含 'input_hint_block' 的层的前两部分，去重后形成集合，排除特定的键
        ".".join(layer.split(".")[:2])
        for layer in controlnet_state_dict
        if "input_hint_block" in layer and ("input_hint_block.0" not in layer) and ("input_hint_block.14" not in layer)
    }
    # 计算条件嵌入块的数量
    num_cond_embedding_blocks = len(cond_embedding_blocks)
    
    # 遍历条件嵌入块索引
    for idx in range(1, num_cond_embedding_blocks + 1):
        diffusers_idx = idx - 1  # 转换为 Diffusers 索引
        cond_block_id = 2 * idx  # 计算条件块的 ID
    
        # 从控制网络状态字典获取对应权重并添加到新检查点
        new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_idx}.weight"] = controlnet_state_dict.get(
            f"input_hint_block.{cond_block_id}.weight"
        )
        # 从控制网络状态字典获取对应偏差并添加到新检查点
        new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_idx}.bias"] = controlnet_state_dict.get(
            f"input_hint_block.{cond_block_id}.bias"
        )
    
    # 返回更新后的检查点
    return new_checkpoint
# 将 LDM VAE 检查点转换为适用于 Diffusers 的格式
def convert_ldm_vae_checkpoint(checkpoint, config):
    # 提取 VAE 的状态字典
    vae_state_dict = {}
    # 获取检查点中所有键的列表
    keys = list(checkpoint.keys())
    vae_key = ""
    # 找到 LDM VAE 关键字
    for ldm_vae_key in LDM_VAE_KEYS:
        # 检查是否有键以当前 LDM VAE 关键字开头
        if any(k.startswith(ldm_vae_key) for k in keys):
            vae_key = ldm_vae_key

    # 从检查点中提取与 VAE 相关的键
    for key in keys:
        # 如果键以 VAE 关键字开头，则替换关键字并存储在状态字典中
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}
    # 获取 VAE 对应的 Diffusers 键映射
    vae_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING["vae"]
    # 构建新的检查点字典
    for diffusers_key, ldm_key in vae_diffusers_ldm_map.items():
        # 如果状态字典中没有对应的 LDM 键，则跳过
        if ldm_key not in vae_state_dict:
            continue
        # 将 LDM 状态字典中的值映射到新的检查点中
        new_checkpoint[diffusers_key] = vae_state_dict[ldm_key]

    # 仅获取编码器下块的键
    num_down_blocks = len(config["down_block_types"])
    # 构建每个下块的键映射
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # 遍历下块进行处理
    for i in range(num_down_blocks):
        # 获取当前下块的所有 ResNet 键
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]
        # 更新 VAE ResNet 的键映射到 Diffusers
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"},
        )
        # 如果存在下采样权重，则添加到新的检查点中
        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.bias"
            )

    # 获取中间 ResNet 的键
    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    # 遍历中间 ResNet 进行处理
    for i in range(1, num_mid_res_blocks + 1):
        # 获取当前中间块的所有 ResNet 键
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]
        # 更新中间 ResNet 的键映射到 Diffusers
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    # 获取中间注意力的键
    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    # 更新中间注意力的键映射到 Diffusers
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )

    # 仅获取解码器上块的键
    num_up_blocks = len(config["up_block_types"])
    # 构建每个上块的键映射
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }
    # 遍历向上块的数量
    for i in range(num_up_blocks):
        # 计算当前块的 ID，从最后一个块开始往前
        block_id = num_up_blocks - 1 - i
        # 收集当前块的所有 ResNet 相关的键，排除上采样的键
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]
        # 更新 VAE 的 ResNet 模块到新的 Diffusers 格式
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            # 映射旧键到新键
            mapping={"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"},
        )
        # 检查 VAE 状态字典中是否包含当前块的上采样卷积权重
        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            # 将上采样卷积权重更新到新检查点中
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            # 将上采样卷积偏置更新到新检查点中
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

    # 收集中间块的所有 ResNet 相关的键
    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    # 定义中间 ResNet 块的数量
    num_mid_res_blocks = 2
    # 遍历中间块的数量
    for i in range(1, num_mid_res_blocks + 1):
        # 收集当前中间块的所有 ResNet 相关的键
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]
        # 更新 VAE 的中间块 ResNet 模块到新的 Diffusers 格式
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            # 映射旧键到新键
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    # 收集中间块的所有注意力相关的键
    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    # 更新 VAE 的中间注意力模块到新的 Diffusers 格式
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )
    # 将卷积注意力转换为线性注意力
    conv_attn_to_linear(new_checkpoint)

    # 返回更新后的新检查点
    return new_checkpoint
# 将 LDM-CLIP 检查点转换为文本模型的字典
def convert_ldm_clip_checkpoint(checkpoint, remove_prefix=None):
    # 获取检查点的所有键，转换为列表
    keys = list(checkpoint.keys())
    # 初始化一个空字典，用于存储文本模型的键值对
    text_model_dict = {}

    # 创建一个空列表用于存储要移除的前缀
    remove_prefixes = []
    # 将预定义的前缀添加到列表中
    remove_prefixes.extend(LDM_CLIP_PREFIX_TO_REMOVE)
    # 如果提供了移除前缀，则添加到列表
    if remove_prefix:
        remove_prefixes.append(remove_prefix)

    # 遍历所有键
    for key in keys:
        # 对每个前缀进行检查
        for prefix in remove_prefixes:
            # 如果键以当前前缀开头
            if key.startswith(prefix):
                # 替换前缀，得到新的键
                diffusers_key = key.replace(prefix, "")
                # 将原始检查点中的值赋给新的键
                text_model_dict[diffusers_key] = checkpoint.get(key)

    # 返回文本模型字典
    return text_model_dict


# 将 Open-CLIP 检查点转换为文本模型的字典
def convert_open_clip_checkpoint(
    text_model,
    checkpoint,
    prefix="cond_stage_model.model.",
):
    # 初始化一个空字典，用于存储文本模型的键值对
    text_model_dict = {}
    # 构造文本投影的键
    text_proj_key = prefix + "text_projection"

    # 如果文本投影键在检查点中
    if text_proj_key in checkpoint:
        # 获取文本投影的维度
        text_proj_dim = int(checkpoint[text_proj_key].shape[0])
    # 如果文本模型配置中有投影维度属性
    elif hasattr(text_model.config, "projection_dim"):
        # 获取该投影维度
        text_proj_dim = text_model.config.projection_dim
    # 否则使用默认的投影维度
    else:
        text_proj_dim = LDM_OPEN_CLIP_TEXT_PROJECTION_DIM

    # 获取检查点的所有键，转换为列表
    keys = list(checkpoint.keys())
    # 获取要忽略的键列表
    keys_to_ignore = SD_2_TEXT_ENCODER_KEYS_TO_IGNORE

    # 获取 Open-CLIP 到 LDM 的映射
    openclip_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING["openclip"]["layers"]
    # 遍历映射中的每个键
    for diffusers_key, ldm_key in openclip_diffusers_ldm_map.items():
        # 将 LDM 键添加前缀
        ldm_key = prefix + ldm_key
        # 如果 LDM 键不在检查点中，则跳过
        if ldm_key not in checkpoint:
            continue
        # 如果 LDM 键在要忽略的键列表中，则跳过
        if ldm_key in keys_to_ignore:
            continue
        # 如果 LDM 键以文本投影结尾
        if ldm_key.endswith("text_projection"):
            # 转置并存储值到文本模型字典
            text_model_dict[diffusers_key] = checkpoint[ldm_key].T.contiguous()
        else:
            # 否则直接存储值到文本模型字典
            text_model_dict[diffusers_key] = checkpoint[ldm_key]
    # 遍历给定的键列表
        for key in keys:
            # 如果当前键在忽略的键列表中，则跳过
            if key in keys_to_ignore:
                continue
    
            # 如果当前键不以指定前缀开头，则跳过
            if not key.startswith(prefix + "transformer."):
                continue
    
            # 移除前缀，得到变换器的键
            diffusers_key = key.replace(prefix + "transformer.", "")
            # 获取变换器到 LDM 映射的字典
            transformer_diffusers_to_ldm_map = DIFFUSERS_TO_LDM_MAPPING["openclip"]["transformer"]
            # 遍历新旧键的映射
            for new_key, old_key in transformer_diffusers_to_ldm_map.items():
                # 替换旧键为新键，并移除特定后缀
                diffusers_key = (
                    diffusers_key.replace(old_key, new_key).replace(".in_proj_weight", "").replace(".in_proj_bias", "")
                )
    
            # 如果当前键以 ".in_proj_weight" 结尾
            if key.endswith(".in_proj_weight"):
                # 从检查点获取权重值
                weight_value = checkpoint.get(key)
    
                # 将权重值的子集赋值给查询投影权重
                text_model_dict[diffusers_key + ".q_proj.weight"] = weight_value[:text_proj_dim, :].clone().detach()
                # 将权重值的子集赋值给键投影权重
                text_model_dict[diffusers_key + ".k_proj.weight"] = (
                    weight_value[text_proj_dim : text_proj_dim * 2, :].clone().detach()
                )
                # 将权重值的子集赋值给值投影权重
                text_model_dict[diffusers_key + ".v_proj.weight"] = weight_value[text_proj_dim * 2 :, :].clone().detach()
    
            # 如果当前键以 ".in_proj_bias" 结尾
            elif key.endswith(".in_proj_bias"):
                # 从检查点获取偏置值
                weight_value = checkpoint.get(key)
                # 将偏置值的子集赋值给查询投影偏置
                text_model_dict[diffusers_key + ".q_proj.bias"] = weight_value[:text_proj_dim].clone().detach()
                # 将偏置值的子集赋值给键投影偏置
                text_model_dict[diffusers_key + ".k_proj.bias"] = (
                    weight_value[text_proj_dim : text_proj_dim * 2].clone().detach()
                )
                # 将偏置值的子集赋值给值投影偏置
                text_model_dict[diffusers_key + ".v_proj.bias"] = weight_value[text_proj_dim * 2 :].clone().detach()
            # 如果当前键既不是权重也不是偏置，则直接获取该键的值
            else:
                text_model_dict[diffusers_key] = checkpoint.get(key)
    
        # 返回最终的文本模型字典
        return text_model_dict
# 创建一个从 LDM 生成 Diffusers CLIP 模型的函数
def create_diffusers_clip_model_from_ldm(
    cls,  # 模型类
    checkpoint,  # 训练检查点
    subfolder="",  # 子文件夹名称，默认为空
    config=None,  # 配置参数，默认为 None
    torch_dtype=None,  # PyTorch 数据类型，默认为 None
    local_files_only=None,  # 是否仅使用本地文件，默认为 None
    is_legacy_loading=False,  # 是否为旧版加载，默认为 False
):
    # 如果提供了配置，则将其封装为字典
    if config:
        config = {"pretrained_model_name_or_path": config}
    # 如果未提供配置，则从检查点中获取配置
    else:
        config = fetch_diffusers_config(checkpoint)

    # 向后兼容处理
    # 旧版的 `from_single_file` 期望 CLIP 配置放在原始 transformers 模型库的缓存目录中
    # 而不是放在 Diffusers 模型的子文件夹中
    if is_legacy_loading:
        # 发出警告，提示用户进行兼容性更新
        logger.warning(
            (
                "Detected legacy CLIP loading behavior. Please run `from_single_file` with `local_files_only=False once to update "
                "the local cache directory with the necessary CLIP model config files. "
                "Attempting to load CLIP model from legacy cache directory."
            )
        )

        # 如果检查点是 CLIP 模型或 CLIP SDXL 模型
        if is_clip_model(checkpoint) or is_clip_sdxl_model(checkpoint):
            clip_config = "openai/clip-vit-large-patch14"  # 设置 CLIP 配置为 OpenAI 的模型
            config["pretrained_model_name_or_path"] = clip_config  # 更新配置
            subfolder = ""  # 子文件夹设为空

        # 如果检查点是 OpenCLIP 模型
        elif is_open_clip_model(checkpoint):
            clip_config = "stabilityai/stable-diffusion-2"  # 设置 CLIP 配置为 StabilityAI 的模型
            config["pretrained_model_name_or_path"] = clip_config  # 更新配置
            subfolder = "text_encoder"  # 子文件夹设为 text_encoder

        # 如果不满足以上条件，使用默认 CLIP 配置
        else:
            clip_config = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"  # 设置为默认的 CLIP 配置
            config["pretrained_model_name_or_path"] = clip_config  # 更新配置
            subfolder = ""  # 子文件夹设为空

    # 从预训练配置加载模型配置
    model_config = cls.config_class.from_pretrained(**config, subfolder=subfolder, local_files_only=local_files_only)
    # 根据是否可用的加速库选择上下文
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    # 使用上下文初始化模型
    with ctx():
        model = cls(model_config)  # 实例化模型

    # 获取位置嵌入的维度
    position_embedding_dim = model.text_model.embeddings.position_embedding.weight.shape[-1]

    # 如果检查点是 CLIP 模型
    if is_clip_model(checkpoint):
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint)  # 转换检查点格式

    # 如果检查点是 CLIP SDXL 模型并且形状与位置嵌入维度匹配
    elif (
        is_clip_sdxl_model(checkpoint)
        and checkpoint[CHECKPOINT_KEY_NAMES["clip_sdxl"]].shape[-1] == position_embedding_dim
    ):
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint)  # 转换检查点格式

    # 如果检查点是 CLIP SD3 模型并且形状与位置嵌入维度匹配
    elif (
        is_clip_sd3_model(checkpoint)
        and checkpoint[CHECKPOINT_KEY_NAMES["clip_sd3"]].shape[-1] == position_embedding_dim
    ):
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint, "text_encoders.clip_l.transformer.")  # 转换检查点格式
        diffusers_format_checkpoint["text_projection.weight"] = torch.eye(position_embedding_dim)  # 设置权重为单位矩阵

    # 如果检查点是 OpenCLIP 模型
    elif is_open_clip_model(checkpoint):
        prefix = "cond_stage_model.model."  # 设置前缀
        diffusers_format_checkpoint = convert_open_clip_checkpoint(model, checkpoint, prefix=prefix)  # 转换检查点格式

    # 如果检查点是 OpenCLIP SDXL 模型并且形状与位置嵌入维度匹配
    elif (
        is_open_clip_sdxl_model(checkpoint)
        and checkpoint[CHECKPOINT_KEY_NAMES["open_clip_sdxl"]].shape[-1] == position_embedding_dim
    ):
    # 检查条件，开始处理不同类型的检查点
    ):
        # 设置前缀，用于转换模型检查点
        prefix = "conditioner.embedders.1.model."
        # 将检查点转换为 diffusers 格式，使用指定前缀
        diffusers_format_checkpoint = convert_open_clip_checkpoint(model, checkpoint, prefix=prefix)

    # 检查是否为 SDXL 精炼器模型
    elif is_open_clip_sdxl_refiner_model(checkpoint):
        # 设置前缀，用于转换模型检查点
        prefix = "conditioner.embedders.0.model."
        # 将检查点转换为 diffusers 格式，使用指定前缀
        diffusers_format_checkpoint = convert_open_clip_checkpoint(model, checkpoint, prefix=prefix)

    # 检查是否为 SD3 模型，并验证位置嵌入维度是否匹配
    elif (
        is_open_clip_sd3_model(checkpoint)
        and checkpoint[CHECKPOINT_KEY_NAMES["open_clip_sd3"]].shape[-1] == position_embedding_dim
    ):
        # 将检查点转换为 LDM 格式，指定文本编码器的前缀
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint, "text_encoders.clip_g.transformer.")

    # 如果以上条件都不满足，抛出异常
    else:
        raise ValueError("The provided checkpoint does not seem to contain a valid CLIP model.")

    # 检查是否可以使用加速功能
    if is_accelerate_available():
        # 加载模型字典并处理意外的键
        unexpected_keys = load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)
    else:
        # 加载模型状态字典，允许不严格匹配
        _, unexpected_keys = model.load_state_dict(diffusers_format_checkpoint, strict=False)

    # 如果模型有忽略的意外键，进行过滤
    if model._keys_to_ignore_on_load_unexpected is not None:
        for pat in model._keys_to_ignore_on_load_unexpected:
            # 过滤掉与忽略模式匹配的意外键
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    # 如果存在意外的键，记录警告
    if len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
        )

    # 如果指定了数据类型，转换模型
    if torch_dtype is not None:
        model.to(torch_dtype)

    # 设置模型为评估模式
    model.eval()

    # 返回初始化后的模型
    return model
# 定义一个私有方法来加载调度器
def _legacy_load_scheduler(
    cls,
    checkpoint,  # 传入检查点数据
    component_name,  # 组件名称
    original_config=None,  # 原始配置，可选
    **kwargs,  # 其他关键字参数
):
    # 从关键字参数获取调度器类型，默认值为 None
    scheduler_type = kwargs.get("scheduler_type", None)
    # 从关键字参数获取预测类型，默认值为 None
    prediction_type = kwargs.get("prediction_type", None)

    # 如果调度器类型不为 None，发出弃用警告
    if scheduler_type is not None:
        deprecation_message = (
            "Please pass an instance of a Scheduler object directly to the `scheduler` argument in `from_single_file`\n\n"
            "Example:\n\n"
            "from diffusers import StableDiffusionPipeline, DDIMScheduler\n\n"
            "scheduler = DDIMScheduler()\n"
            "pipe = StableDiffusionPipeline.from_single_file(<checkpoint path>, scheduler=scheduler)\n"
        )
        # 调用弃用函数，记录 scheduler_type 的弃用信息
        deprecate("scheduler_type", "1.0.0", deprecation_message)

    # 如果预测类型不为 None，发出弃用警告
    if prediction_type is not None:
        deprecation_message = (
            "Please configure an instance of a Scheduler with the appropriate `prediction_type` and "
            "pass the object directly to the `scheduler` argument in `from_single_file`.\n\n"
            "Example:\n\n"
            "from diffusers import StableDiffusionPipeline, DDIMScheduler\n\n"
            'scheduler = DDIMScheduler(prediction_type="v_prediction")\n'
            "pipe = StableDiffusionPipeline.from_single_file(<checkpoint path>, scheduler=scheduler)\n"
        )
        # 调用弃用函数，记录 prediction_type 的弃用信息
        deprecate("prediction_type", "1.0.0", deprecation_message)

    # 初始化调度器配置为默认配置
    scheduler_config = SCHEDULER_DEFAULT_CONFIG
    # 推断模型类型
    model_type = infer_diffusers_model_type(checkpoint=checkpoint)

    # 获取全局步数，如果不存在则为 None
    global_step = checkpoint["global_step"] if "global_step" in checkpoint else None

    # 如果原始配置存在，获取训练时间步数，否则使用默认值
    if original_config:
        num_train_timesteps = getattr(original_config["model"]["params"], "timesteps", 1000)
    else:
        num_train_timesteps = 1000

    # 将训练时间步数存入调度器配置
    scheduler_config["num_train_timesteps"] = num_train_timesteps

    # 如果模型类型是 v2
    if model_type == "v2":
        if prediction_type is None:
            # 对于稳定扩散 2 基础版本，建议传递 `prediction_type=="epsilon"`，因为这里依赖于脆弱的全局步数参数
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"

    else:
        # 如果模型类型不是 v2，设置预测类型为 epsilon 或现有值
        prediction_type = prediction_type or "epsilon"

    # 将预测类型存入调度器配置
    scheduler_config["prediction_type"] = prediction_type

    # 根据模型类型设置调度器类型和相关参数
    if model_type in ["xl_base", "xl_refiner"]:
        scheduler_type = "euler"
    elif model_type == "playground":
        scheduler_type = "edm_dpm_solver_multistep"
    else:
        # 如果原始配置存在，获取 beta_start 和 beta_end 值
        if original_config:
            beta_start = original_config["model"]["params"].get("linear_start")
            beta_end = original_config["model"]["params"].get("linear_end")

        else:
            # 否则使用默认的 beta_start 和 beta_end 值
            beta_start = 0.02
            beta_end = 0.085

        # 将 beta 参数和其他调度器配置存入调度器配置
        scheduler_config["beta_start"] = beta_start
        scheduler_config["beta_end"] = beta_end
        scheduler_config["beta_schedule"] = "scaled_linear"
        scheduler_config["clip_sample"] = False
        scheduler_config["set_alpha_to_one"] = False
    # 处理特殊情况，StableDiffusionUpscale 管道有两个调度器
    if component_name == "low_res_scheduler":
        # 从配置中创建并返回一个调度器实例
        return cls.from_config(
            {
                # 设置 Beta 结束值
                "beta_end": 0.02,
                # 设置 Beta 调度类型
                "beta_schedule": "scaled_linear",
                # 设置 Beta 起始值
                "beta_start": 0.0001,
                # 是否剪辑样本
                "clip_sample": True,
                # 训练时间步数
                "num_train_timesteps": 1000,
                # 预测类型
                "prediction_type": "epsilon",
                # 训练的 Beta 值
                "trained_betas": None,
                # 方差类型
                "variance_type": "fixed_small",
            }
        )

    # 如果调度器类型为空
    if scheduler_type is None:
        # 从给定的调度器配置中创建调度器
        return cls.from_config(scheduler_config)

    # 如果调度器类型为 "pndm"
    elif scheduler_type == "pndm":
        # 设置跳过 PRK 步骤为真
        scheduler_config["skip_prk_steps"] = True
        # 从配置中创建 PNDM 调度器
        scheduler = PNDMScheduler.from_config(scheduler_config)

    # 如果调度器类型为 "lms"
    elif scheduler_type == "lms":
        # 从配置中创建 LMS 离散调度器
        scheduler = LMSDiscreteScheduler.from_config(scheduler_config)

    # 如果调度器类型为 "heun"
    elif scheduler_type == "heun":
        # 从配置中创建 Heun 离散调度器
        scheduler = HeunDiscreteScheduler.from_config(scheduler_config)

    # 如果调度器类型为 "euler"
    elif scheduler_type == "euler":
        # 从配置中创建 Euler 离散调度器
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)

    # 如果调度器类型为 "euler-ancestral"
    elif scheduler_type == "euler-ancestral":
        # 从配置中创建 Euler 祖先离散调度器
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)

    # 如果调度器类型为 "dpm"
    elif scheduler_type == "dpm":
        # 从配置中创建 DPM 多步调度器
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

    # 如果调度器类型为 "ddim"
    elif scheduler_type == "ddim":
        # 从配置中创建 DDIM 调度器
        scheduler = DDIMScheduler.from_config(scheduler_config)

    # 如果调度器类型为 "edm_dpm_solver_multistep"
    elif scheduler_type == "edm_dpm_solver_multistep":
        # 定义调度器配置字典
        scheduler_config = {
            # 算法类型
            "algorithm_type": "dpmsolver++",
            # 动态阈值比例
            "dynamic_thresholding_ratio": 0.995,
            # 最终是否使用欧拉法
            "euler_at_final": False,
            # 最终 sigma 类型
            "final_sigmas_type": "zero",
            # 较低阶最终设置
            "lower_order_final": True,
            # 训练时间步数
            "num_train_timesteps": 1000,
            # 预测类型
            "prediction_type": "epsilon",
            # rho 值
            "rho": 7.0,
            # 样本最大值
            "sample_max_value": 1.0,
            # 数据 sigma
            "sigma_data": 0.5,
            # 最大 sigma
            "sigma_max": 80.0,
            # 最小 sigma
            "sigma_min": 0.002,
            # 求解器阶数
            "solver_order": 2,
            # 求解器类型
            "solver_type": "midpoint",
            # 是否使用阈值处理
            "thresholding": False,
        }
        # 从配置中创建 EDM DPM 多步调度器
        scheduler = EDMDPMSolverMultistepScheduler(**scheduler_config)

    # 如果调度器类型不匹配，抛出异常
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    # 返回创建的调度器实例
    return scheduler
# 定义一个类方法，用于加载旧版 CLIP 分词器
def _legacy_load_clip_tokenizer(cls, checkpoint, config=None, local_files_only=False):
    # 如果提供了 config，则将其包装为包含模型路径的字典
    if config:
        config = {"pretrained_model_name_or_path": config}
    # 如果未提供 config，则从检查点获取配置信息
    else:
        config = fetch_diffusers_config(checkpoint)

    # 检查点是否为 CLIP 模型或 CLIP SDXL 模型
    if is_clip_model(checkpoint) or is_clip_sdxl_model(checkpoint):
        # 设置使用的 CLIP 配置
        clip_config = "openai/clip-vit-large-patch14"
        # 将配置中的模型路径设置为 CLIP 配置
        config["pretrained_model_name_or_path"] = clip_config
        # 设置子文件夹为空
        subfolder = ""

    # 检查点是否为 Open CLIP 模型
    elif is_open_clip_model(checkpoint):
        # 设置使用的 Open CLIP 配置
        clip_config = "stabilityai/stable-diffusion-2"
        # 将配置中的模型路径设置为 Open CLIP 配置
        config["pretrained_model_name_or_path"] = clip_config
        # 设置子文件夹为 tokenizer
        subfolder = "tokenizer"

    # 如果不是以上模型，则使用默认的 CLIP 配置
    else:
        clip_config = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        # 将配置中的模型路径设置为默认 CLIP 配置
        config["pretrained_model_name_or_path"] = clip_config
        # 设置子文件夹为空
        subfolder = ""

    # 从预训练模型加载分词器，传入配置和子文件夹
    tokenizer = cls.from_pretrained(**config, subfolder=subfolder, local_files_only=local_files_only)

    # 返回加载的分词器
    return tokenizer


# 定义一个加载安全检查器的函数
def _legacy_load_safety_checker(local_files_only, torch_dtype):
    # 使用过时的 `load_safety_checker` 参数支持加载安全检查器组件

    # 从指定路径导入稳定扩散安全检查器
    from ..pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

    # 从预训练模型加载特征提取器
    feature_extractor = AutoImageProcessor.from_pretrained(
        "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only, torch_dtype=torch_dtype
    )
    # 从预训练模型加载安全检查器
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only, torch_dtype=torch_dtype
    )

    # 返回包含安全检查器和特征提取器的字典
    return {"safety_checker": safety_checker, "feature_extractor": feature_extractor}


# 在 SD3 的原始实现中，AdaLayerNormContinuous 将线性投影输出分为 shift 和 scale；
# 而在 diffusers 中，顺序为 scale 和 shift。这里交换线性投影的权重，以便能使用 diffusers 的实现
def swap_scale_shift(weight, dim):
    # 将权重在指定维度分成两部分：shift 和 scale
    shift, scale = weight.chunk(2, dim=0)
    # 重新组合权重，将 scale 放在前面，shift 放在后面
    new_weight = torch.cat([scale, shift], dim=0)
    # 返回新的权重
    return new_weight


# 定义一个将 SD3 转换为 diffusers 的检查点转换函数
def convert_sd3_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    # 初始化一个空的字典用于保存转换后的状态字典
    converted_state_dict = {}
    # 获取检查点中的所有键并转换为列表
    keys = list(checkpoint.keys())
    # 遍历所有键
    for k in keys:
        # 如果键包含特定字符串，则将其替换
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    # 获取 joint_blocks 的层数，并计算出总层数
    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "joint_blocks" in k))[-1] + 1  # noqa: C401
    # 设置 caption projection 的维度
    caption_projection_dim = 1536

    # 处理位置和补丁嵌入
    converted_state_dict["pos_embed.pos_embed"] = checkpoint.pop("pos_embed")
    converted_state_dict["pos_embed.proj.weight"] = checkpoint.pop("x_embedder.proj.weight")
    converted_state_dict["pos_embed.proj.bias"] = checkpoint.pop("x_embedder.proj.bias")

    # 处理时间步嵌入
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = checkpoint.pop(
        "t_embedder.mlp.0.weight"
    )
    # 从检查点中弹出时间文本嵌入器的线性层1的偏置，并赋值给转换后的状态字典
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = checkpoint.pop("t_embedder.mlp.0.bias")
    # 从检查点中弹出时间文本嵌入器的线性层2的权重，并赋值给转换后的状态字典
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = checkpoint.pop(
        "t_embedder.mlp.2.weight"
    )
    # 从检查点中弹出时间文本嵌入器的线性层2的偏置，并赋值给转换后的状态字典
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = checkpoint.pop("t_embedder.mlp.2.bias")
    
    # 从检查点中弹出上下文嵌入器的权重，并赋值给转换后的状态字典
    converted_state_dict["context_embedder.weight"] = checkpoint.pop("context_embedder.weight")
    # 从检查点中弹出上下文嵌入器的偏置，并赋值给转换后的状态字典
    converted_state_dict["context_embedder.bias"] = checkpoint.pop("context_embedder.bias")
    
    # 从检查点中弹出时间文本嵌入的线性层1的权重，并赋值给转换后的状态字典
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = checkpoint.pop("y_embedder.mlp.0.weight")
    # 从检查点中弹出时间文本嵌入的线性层1的偏置，并赋值给转换后的状态字典
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = checkpoint.pop("y_embedder.mlp.0.bias")
    # 从检查点中弹出时间文本嵌入的线性层2的权重，并赋值给转换后的状态字典
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = checkpoint.pop("y_embedder.mlp.2.weight")
    # 从检查点中弹出时间文本嵌入的线性层2的偏置，并赋值给转换后的状态字典
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = checkpoint.pop("y_embedder.mlp.2.bias")
    
    # 从检查点中弹出最终层的线性权重，并赋值给转换后的状态字典
    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    # 从检查点中弹出最终层的偏置，并赋值给转换后的状态字典
    converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")
    # 从检查点中弹出最终层的自适应层归一化调制的权重，进行维度调整后赋值给转换后的状态字典
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.weight"), dim=caption_projection_dim
    )
    # 从检查点中弹出最终层的自适应层归一化调制的偏置，进行维度调整后赋值给转换后的状态字典
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.bias"), dim=caption_projection_dim
    )
    
    # 返回转换后的状态字典
    return converted_state_dict
# 检查给定的检查点是否包含特定的 T5 模型权重
def is_t5_in_single_file(checkpoint):
    # 如果检查点中包含 T5 权重，则返回 True
    if "text_encoders.t5xxl.transformer.shared.weight" in checkpoint:
        return True

    # 否则返回 False
    return False


# 将 SD3 格式的 T5 检查点转换为 Diffusers 格式
def convert_sd3_t5_checkpoint_to_diffusers(checkpoint):
    # 获取检查点中的所有键
    keys = list(checkpoint.keys())
    # 初始化空的字典以存储转换后的模型权重
    text_model_dict = {}

    # 定义需要移除的前缀
    remove_prefixes = ["text_encoders.t5xxl.transformer."]

    # 遍历每个键
    for key in keys:
        # 对每个前缀进行检查
        for prefix in remove_prefixes:
            # 如果键以前缀开头
            if key.startswith(prefix):
                # 替换前缀并获取新的键名
                diffusers_key = key.replace(prefix, "")
                # 将原键对应的值存入新字典中
                text_model_dict[diffusers_key] = checkpoint.get(key)

    # 返回转换后的模型权重字典
    return text_model_dict


# 从检查点创建 Diffusers 格式的 T5 模型
def create_diffusers_t5_model_from_checkpoint(
    cls,
    checkpoint,
    subfolder="",
    config=None,
    torch_dtype=None,
    local_files_only=None,
):
    # 如果提供了配置，则使用它
    if config:
        config = {"pretrained_model_name_or_path": config}
    else:
        # 否则从检查点中获取配置
        config = fetch_diffusers_config(checkpoint)

    # 从配置中加载模型配置
    model_config = cls.config_class.from_pretrained(**config, subfolder=subfolder, local_files_only=local_files_only)
    # 根据是否可用的加速初始化上下文
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    # 使用上下文创建模型
    with ctx():
        model = cls(model_config)

    # 将检查点转换为 Diffusers 格式
    diffusers_format_checkpoint = convert_sd3_t5_checkpoint_to_diffusers(checkpoint)

    # 如果加速可用，加载模型权重
    if is_accelerate_available():
        unexpected_keys = load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)
        # 检查是否有需要忽略的意外键
        if model._keys_to_ignore_on_load_unexpected is not None:
            for pat in model._keys_to_ignore_on_load_unexpected:
                # 过滤意外键
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # 如果存在意外键，发出警告
        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
            )

    else:
        # 否则直接加载权重
        model.load_state_dict(diffusers_format_checkpoint)

    # 检查是否需要保持 FP32 模块
    use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (torch_dtype == torch.float16)
    if use_keep_in_fp32_modules:
        # 获取需要保持为 FP32 的模块
        keep_in_fp32_modules = model._keep_in_fp32_modules
    else:
        keep_in_fp32_modules = []

    # 如果存在需要保持为 FP32 的模块
    if keep_in_fp32_modules is not None:
        # 遍历模型的每个参数
        for name, param in model.named_parameters():
            # 如果参数名中包含需要保持为 FP32 的模块
            if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules):
                # 将参数数据转换为 FP32（只在局部作用域有效）
                param.data = param.data.to(torch.float32)

    # 返回最终模型
    return model


# 将 Animatediff 格式的检查点转换为 Diffusers 格式
def convert_animatediff_checkpoint_to_diffusers(checkpoint, **kwargs):
    # 初始化空字典以存储转换后的状态字典
    converted_state_dict = {}
    # 遍历检查点字典中的每个键值对
        for k, v in checkpoint.items():
            # 如果键中包含 "pos_encoder"，则跳过此项
            if "pos_encoder" in k:
                continue
    
            else:
                # 替换键名中的特定子字符串，并将值赋给新字典
                converted_state_dict[
                    k.replace(".norms.0", ".norm1")  # 替换 ".norms.0" 为 ".norm1"
                    .replace(".norms.1", ".norm2")  # 替换 ".norms.1" 为 ".norm2"
                    .replace(".ff_norm", ".norm3")  # 替换 ".ff_norm" 为 ".norm3"
                    .replace(".attention_blocks.0", ".attn1")  # 替换 ".attention_blocks.0" 为 ".attn1"
                    .replace(".attention_blocks.1", ".attn2")  # 替换 ".attention_blocks.1" 为 ".attn2"
                    .replace(".temporal_transformer", "")  # 移除 ".temporal_transformer"
                ] = v  # 将原值 v 赋给新的键
    
        # 返回转换后的状态字典
        return converted_state_dict
# 将给定的检查点转换为 Diffusers 格式的模型
def convert_flux_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    # 初始化一个空字典，用于存储转换后的状态字典
    converted_state_dict = {}
    # 获取检查点中所有键的列表
    keys = list(checkpoint.keys())
    # 遍历每个键
    for k in keys:
        # 如果键包含 "model.diffusion_model."，则替换该部分并更新检查点
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    # 计算双层块的数量
    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "double_blocks." in k))[-1] + 1  # noqa: C401
    # 计算单层块的数量
    num_single_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "single_blocks." in k))[-1] + 1  # noqa: C401
    # 设置 MLP 比率
    mlp_ratio = 4.0
    # 设置内部维度
    inner_dim = 3072

    # 定义一个函数，用于交换线性投影的权重顺序
    def swap_scale_shift(weight):
        # 将权重拆分为 shift 和 scale
        shift, scale = weight.chunk(2, dim=0)
        # 重新连接为新的权重
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    ## 将时间嵌入的线性层权重从检查点中提取并赋值
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = checkpoint.pop(
        "time_in.in_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = checkpoint.pop("time_in.in_layer.bias")
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = checkpoint.pop(
        "time_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = checkpoint.pop("time_in.out_layer.bias")

    ## 将文本嵌入的线性层权重从检查点中提取并赋值
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = checkpoint.pop("vector_in.in_layer.weight")
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = checkpoint.pop("vector_in.in_layer.bias")
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = checkpoint.pop(
        "vector_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = checkpoint.pop("vector_in.out_layer.bias")

    # 检查是否有引导信息
    has_guidance = any("guidance" in k for k in checkpoint)
    # 如果存在引导信息，从检查点中提取并赋值
    if has_guidance:
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = checkpoint.pop(
            "guidance_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = checkpoint.pop(
            "guidance_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = checkpoint.pop(
            "guidance_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = checkpoint.pop(
            "guidance_in.out_layer.bias"
        )

    # 提取上下文嵌入的权重和偏置
    converted_state_dict["context_embedder.weight"] = checkpoint.pop("txt_in.weight")
    converted_state_dict["context_embedder.bias"] = checkpoint.pop("txt_in.bias")
    # x_embedder
    # 从检查点中弹出图像输入的权重，赋值给转换后的状态字典的 x_embedder 权重
    converted_state_dict["x_embedder.weight"] = checkpoint.pop("img_in.weight")
    # 从检查点中弹出图像输入的偏置，赋值给转换后的状态字典的 x_embedder 偏置
    converted_state_dict["x_embedder.bias"] = checkpoint.pop("img_in.bias")

    # double transformer blocks
    # single transfomer blocks
    # 遍历单个变换器层的数量
    for i in range(num_single_layers):
        # 生成当前单个变换器块的前缀
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear  <- single_blocks.0.modulation.lin
        # 从检查点中弹出当前层的线性权重，赋值给转换后的状态字典的归一化层权重
        converted_state_dict[f"{block_prefix}norm.linear.weight"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.weight"
        )
        # 从检查点中弹出当前层的线性偏置，赋值给转换后的状态字典的归一化层偏置
        converted_state_dict[f"{block_prefix}norm.linear.bias"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.bias"
        )
        # Q, K, V, mlp
        # 计算 MLP 的隐藏维度
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        # 定义分割大小，用于分割线性权重
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        # 从检查点中弹出当前层的线性权重，并按分割大小分割成 Q, K, V 和 MLP
        q, k, v, mlp = torch.split(checkpoint.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
        # 从检查点中弹出当前层的线性偏置，并按分割大小分割成 Q, K, V 和 MLP 偏置
        q_bias, k_bias, v_bias, mlp_bias = torch.split(
            checkpoint.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
        )
        # 将 Q 的权重和偏置添加到转换后的状态字典
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
        # 将 K 的权重和偏置添加到转换后的状态字典
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
        # 将 V 的权重和偏置添加到转换后的状态字典
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
        # 将 MLP 的权重和偏置添加到转换后的状态字典
        converted_state_dict[f"{block_prefix}proj_mlp.weight"] = torch.cat([mlp])
        converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
        # qk norm
        # 从检查点中弹出当前层的 Q 归一化权重，赋值给转换后的状态字典
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        # 从检查点中弹出当前层的 K 归一化权重，赋值给转换后的状态字典
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )
        # output projections.
        # 从检查点中弹出当前层的输出线性权重，赋值给转换后的状态字典
        converted_state_dict[f"{block_prefix}proj_out.weight"] = checkpoint.pop(f"single_blocks.{i}.linear2.weight")
        # 从检查点中弹出当前层的输出线性偏置，赋值给转换后的状态字典
        converted_state_dict[f"{block_prefix}proj_out.bias"] = checkpoint.pop(f"single_blocks.{i}.linear2.bias")

    # 从检查点中弹出最终层的线性权重，赋值给转换后的状态字典
    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    # 从检查点中弹出最终层的线性偏置，赋值给转换后的状态字典
    converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")
    # 从检查点中弹出最终层的归一化调制权重，并进行换位操作
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.weight")
    )
    # 从检查点中弹出最终层的归一化调制偏置，并进行换位操作
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.bias")
    )

    # 返回转换后的状态字典
    return converted_state_dict
```