# `.\models\deformable_detr\modeling_deformable_detr.py`

```py
# coding=utf-8
# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Deformable DETR model."""

import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# Importing various utilities and dependencies from transformers and related libraries
from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_timm_available,
    is_torch_cuda_available,
    is_vision_available,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_deformable_detr import DeformableDetrConfig

# Get logger instance for logging messages
logger = logging.get_logger(__name__)

# Initialize MultiScaleDeformableAttention to None initially
MultiScaleDeformableAttention = None

# Function to load CUDA kernels required for MultiScaleDeformableAttention
def load_cuda_kernels():
    from torch.utils.cpp_extension import load

    global MultiScaleDeformableAttention

    # Define the path to the CUDA and CPU source files
    root = Path(__file__).resolve().parent.parent.parent / "kernels" / "deformable_detr"
    src_files = [
        root / filename
        for filename in [
            "vision.cpp",
            os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
            os.path.join("cuda", "ms_deform_attn_cuda.cu"),
        ]
    ]

    # Load the CUDA kernels using torch's cpp_extension.load()
    MultiScaleDeformableAttention = load(
        "MultiScaleDeformableAttention",
        src_files,
        with_cuda=True,
        extra_include_paths=[str(root)],
        extra_cflags=["-DWITH_CUDA=1"],
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    )

# Check if vision utilities are available and import center_to_corners_format
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

# Check if accelerate library is available and import necessary components
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

# Define the Function class for MultiScaleDeformableAttentionFunction
class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    # 定义静态方法 `forward`，用于执行前向传播操作
    def forward(
        context,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        # 将 im2col_step 参数保存到上下文对象 context 中
        context.im2col_step = im2col_step
        # 调用 MultiScaleDeformableAttention 类的静态方法 ms_deform_attn_forward 进行多尺度可变形注意力的前向传播
        output = MultiScaleDeformableAttention.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            context.im2col_step,
        )
        # 将前向传播中需要保存的张量保存到 context 的备忘录中
        context.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        # 返回前向传播的输出结果
        return output

    # 定义静态方法 `backward`，用于执行反向传播操作
    @staticmethod
    @once_differentiable
    def backward(context, grad_output):
        # 从 context 的备忘录中获取前向传播时保存的张量
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = context.saved_tensors
        # 调用 MultiScaleDeformableAttention 类的静态方法 ms_deform_attn_backward 进行多尺度可变形注意力的反向传播
        # 返回各个梯度值：grad_value 为输入值的梯度，grad_sampling_loc 为采样位置的梯度，grad_attn_weight 为注意力权重的梯度
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttention.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            context.im2col_step,
        )

        # 返回梯度值，其中输入值的梯度 grad_value 需要返回，其他梯度项为 None
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
# 如果 scipy 可用，导入线性求解模块
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 如果 timm 可用，导入模型创建函数
if is_timm_available():
    from timm import create_model

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 文档用的配置名称
_CONFIG_FOR_DOC = "DeformableDetrConfig"

# 文档用的检查点名称
_CHECKPOINT_FOR_DOC = "sensetime/deformable-detr"

# Deformable DETR 预训练模型存档列表
DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sensetime/deformable-detr",
    # 查看所有 Deformable DETR 模型的列表链接
    # https://huggingface.co/models?filter=deformable-detr
]

@dataclass
class DeformableDetrDecoderOutput(ModelOutput):
    """
    DeformableDetrDecoder 的输出的基类。这个类向 BaseModelOutputWithCrossAttentions 添加了两个属性：
    - 一个堆叠的中间解码器隐藏状态张量（即每个解码器层的输出）
    - 一个堆叠的中间参考点张量

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            堆叠的中间隐藏状态（解码器每层的输出）。
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            堆叠的中间参考点（解码器每层的参考点）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 元组（一个用于嵌入输出 + 一个用于每层输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            模型每层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 元组（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力权重经过注意力 softmax 后的结果，在自注意力头中用于计算加权平均。
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 元组（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            解码器交叉注意力层的注意力权重，在注意力 softmax 后用于计算加权平均。
    """
    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    # 定义一个变量 intermediate_reference_points，类型为 torch.FloatTensor，初始值为 None
    intermediate_reference_points: torch.FloatTensor = None
    
    # 定义一个变量 hidden_states，类型为 Optional[Tuple[torch.FloatTensor]]，初始值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义一个变量 attentions，类型为 Optional[Tuple[torch.FloatTensor]]，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义一个变量 cross_attentions，类型为 Optional[Tuple[torch.FloatTensor]]，初始值为 None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 使用 `dataclass` 装饰器定义了一个数据类 `DeformableDetrModelOutput`，表示Deformable DETR模型的输出。
class DeformableDetrModelOutput(ModelOutput):
    """
    Base class for outputs of the Deformable DETR encoder-decoder model.
    """

    # 下面是该类的属性定义，每个属性都是一个 `torch.FloatTensor` 类型的张量，用于存储不同的模型输出。
    init_reference_points: torch.FloatTensor = None  # 初始参考点
    last_hidden_state: torch.FloatTensor = None  # 最后隐藏状态
    intermediate_hidden_states: torch.FloatTensor = None  # 中间隐藏状态
    intermediate_reference_points: torch.FloatTensor = None  # 中间参考点
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 解码器隐藏状态
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 解码器注意力
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 交叉注意力
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # 编码器最后隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 编码器隐藏状态
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 编码器注意力
    enc_outputs_class: Optional[torch.FloatTensor] = None  # 输出类别
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None  # 输出坐标对数

# 使用 `dataclass` 装饰器定义了另一个数据类 `DeformableDetrObjectDetectionOutput`，表示Deformable DETR模型的目标检测输出。
class DeformableDetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`DeformableDetrForObjectDetection`].
    """

    # 下面是该类的属性定义，每个属性都是与模型输出相关的数据。
    loss: Optional[torch.FloatTensor] = None  # 损失值
    loss_dict: Optional[Dict] = None  # 损失字典
    logits: torch.FloatTensor = None  # 对数
    pred_boxes: torch.FloatTensor = None  # 预测框
    auxiliary_outputs: Optional[List[Dict]] = None  # 辅助输出
    init_reference_points: Optional[torch.FloatTensor] = None  # 初始参考点
    last_hidden_state: Optional[torch.FloatTensor] = None  # 最后隐藏状态
    intermediate_hidden_states: Optional[torch.FloatTensor] = None  # 中间隐藏状态
    intermediate_reference_points: Optional[torch.FloatTensor] = None  # 中间参考点
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 解码器隐藏状态
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 解码器注意力
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 交叉注意力
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # 编码器最后隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 编码器隐藏状态
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 编码器注意力
    enc_outputs_class: Optional = None  # 输出类别
    enc_outputs_coord_logits: Optional = None  # 输出坐标对数

# 定义了一个函数 `_get_clones`，用于克隆指定模块多次。
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# 定义了一个函数 `inverse_sigmoid`，计算输入张量的逆sigmoid值。
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)  # 将输入张量限制在 [0, 1] 范围内
    x1 = x.clamp(min=eps)  # 将输入张量在较小值处截断为 `eps`
    x2 = (1 - x).clamp(min=eps)  # 将 (1 - x) 在较小值处截断为 `eps`
    return torch.log(x1 / x2)  # 返回计算后的逆sigmoid值

# 定义了一个类 `DeformableDetrFrozenBatchNorm2d`，继承自 `nn.Module`，用于冻结统计数据和仿射参数的批标准化。
class DeformableDetrFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    
    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    # 类初始化函数，接受一个参数 `n`。
    def __init__(self, n):
        super().__init__()
        # 注册 `weight`、`bias`、`running_mean`、`running_var` 四个缓冲区。
        self.register_buffer("weight", torch.ones(n))  # 权重初始化为全1
        self.register_buffer("bias", torch.zeros(n))  # 偏置初始化为全0
        self.register_buffer("running_mean", torch.zeros(n))  # 运行时均值初始化为全0
        self.register_buffer("running_var", torch.ones(n))  # 运行时方差初始化为全1
    # 从状态字典中加载模型参数，并根据给定的前缀处理键名，处理缺失和意外的键
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 构建用于追踪批次数的键名
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # 如果追踪批次数的键名存在于状态字典中，则删除该键
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类的_load_from_state_dict方法，加载模型参数
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # 定义前向传播方法
    def forward(self, x):
        # 将权重张量重塑为指定形状，以便适合前向传播的需求
        weight = self.weight.reshape(1, -1, 1, 1)
        # 将偏置张量重塑为指定形状，以便适合前向传播的需求
        bias = self.bias.reshape(1, -1, 1, 1)
        # 将运行时方差张量重塑为指定形状，以便适合前向传播的需求
        running_var = self.running_var.reshape(1, -1, 1, 1)
        # 将运行时均值张量重塑为指定形状，以便适合前向传播的需求
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        # 定义一个极小值常量 epsilon，用于数值稳定性
        epsilon = 1e-5
        # 计算缩放系数，用于规范化输入数据
        scale = weight * (running_var + epsilon).rsqrt()
        # 计算偏置项，用于将规范化后的数据重新调整到正确的范围
        bias = bias - running_mean * scale
        # 返回经过规范化和调整后的数据
        return x * scale + bias
# 从 transformers.models.detr.modeling_detr.replace_batch_norm 复制，并将 Detr 替换为 DeformableDetr
def replace_batch_norm(model):
    """
    递归地将所有 `torch.nn.BatchNorm2d` 替换为 `DeformableDetrFrozenBatchNorm2d`。

    Args:
        model (torch.nn.Module):
            输入的模型
    """
    # 遍历模型的每个子模块
    for name, module in model.named_children():
        # 如果当前模块是 nn.BatchNorm2d 类型
        if isinstance(module, nn.BatchNorm2d):
            # 创建一个新的 DeformableDetrFrozenBatchNorm2d 模块，与原始模块的特征数相同
            new_module = DeformableDetrFrozenBatchNorm2d(module.num_features)

            # 如果原始模块的权重不在 torch.device("meta") 上
            if not module.weight.device == torch.device("meta"):
                # 复制原始模块的权重、偏置、运行时均值和方差到新模块
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            # 将模型中原始的 BatchNorm2d 模块替换为新创建的 DeformableDetrFrozenBatchNorm2d 模块
            model._modules[name] = new_module

        # 如果当前模块还有子模块，则递归调用替换函数
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class DeformableDetrConvEncoder(nn.Module):
    """
    使用 AutoBackbone API 或 timm 库之一的卷积主干网络。

    所有 nn.BatchNorm2d 层都被上面定义的 DeformableDetrFrozenBatchNorm2d 替换。

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # 根据配置选择使用 timm 库的 backbone 还是自定义的加载
        if config.use_timm_backbone:
            # 确保需要的后端库已导入
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            # 创建 timm 库中指定的 backbone 模型
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(2, 3, 4) if config.num_feature_levels > 1 else (4,),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            # 自定义加载 backbone 模型
            backbone = load_backbone(config)

        # 使用 torch.no_grad() 替换所有的 BatchNorm 层为冻结的 BatchNorm 层
        with torch.no_grad():
            replace_batch_norm(backbone)

        # 将处理后的 backbone 设置为模型的一部分
        self.model = backbone
        # 获取中间层的通道数信息
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        # 根据 backbone 的类型和配置冻结特定的参数
        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # 将像素值通过模型传递，以获取特征图列表
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps
        
        out = []
        for feature_map in features:
            # 将像素掩码下采样至与对应特征图相同的形状
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        # 返回特征图和相应的掩码组成的列表
        return out
# 从 transformers.models.detr.modeling_detr.DetrConvModel 复制并修改为 DeformableDetrConvModel
class DeformableDetrConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder  # 初始化卷积编码器
        self.position_embedding = position_embedding  # 初始化位置编码器

    def forward(self, pixel_values, pixel_mask):
        # 通过骨干网络（backbone）传递像素值和像素掩码，获取 (特征图, 像素掩码) 元组列表
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # 执行位置编码
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class DeformableDetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim  # 设置嵌入维度
        self.temperature = temperature  # 温度参数
        self.normalize = normalize  # 是否进行归一化
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 缩放参数，默认为2π

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")  # 如果未提供像素掩码，抛出异常
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)  # 在第一维度上累积求和，得到y方向的位置编码
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)  # 在第二维度上累积求和，得到x方向的位置编码
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale  # 对y方向位置编码进行归一化和缩放
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale  # 对x方向位置编码进行归一化和缩放

        dim_t = torch.arange(self.embedding_dim, dtype=torch.int64, device=pixel_values.device).float()  # 创建维度参数
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)  # 计算温度调整

        pos_x = x_embed[:, :, :, None] / dim_t  # 计算x方向的位置编码
        pos_y = y_embed[:, :, :, None] / dim_t  # 计算y方向的位置编码
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 对x方向位置编码应用正弦余弦变换
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 对y方向位置编码应用正弦余弦变换
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 组合并转置位置编码，以适应模型输入要求
        return pos


# 从 transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding 复制的类，已用于 DeformableDetr
class DeformableDetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    # 初始化函数，定义了类的初始化方法，设置了嵌入维度为256
    def __init__(self, embedding_dim=256):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个行索引的嵌入层，50表示索引的范围，embedding_dim表示每个嵌入的维度
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        # 创建一个列索引的嵌入层，参数与行索引的嵌入层类似
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    # 前向传播方法，接收像素值和像素掩码作为输入
    def forward(self, pixel_values, pixel_mask=None):
        # 获取像素值的高度和宽度
        height, width = pixel_values.shape[-2:]
        # 生成宽度的索引张量，设备与像素值张量相同
        width_values = torch.arange(width, device=pixel_values.device)
        # 生成高度的索引张量，设备与像素值张量相同
        height_values = torch.arange(height, device=pixel_values.device)
        # 通过列嵌入层获取每个宽度索引的嵌入表示
        x_emb = self.column_embeddings(width_values)
        # 通过行嵌入层获取每个高度索引的嵌入表示
        y_emb = self.row_embeddings(height_values)
        # 拼接 x_emb 和 y_emb 成为位置嵌入张量，最后一个维度为2*embedding_dim
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # 将 pos 张量的维度重新排列为 (2*embedding_dim, height, width)
        pos = pos.permute(2, 0, 1)
        # 在第一维度上添加一个维度，成为 (1, 2*embedding_dim, height, width)
        pos = pos.unsqueeze(0)
        # 将 pos 张量沿着第一维度复制 pixel_values.shape[0] 次，形成 (batch_size, 2*embedding_dim, height, width)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # 返回位置编码张量 pos
        return pos
# 从 transformers.models.detr.modeling_detr.build_position_encoding 复制并修改为 DeformableDetr 的位置编码构建函数
def build_position_encoding(config):
    # 计算位置编码的步数，使用模型维度的一半
    n_steps = config.d_model // 2
    # 根据配置选择位置编码类型为 "sine"
    if config.position_embedding_type == "sine":
        # 使用 DeformableDetrSinePositionEmbedding 类创建正弦位置编码对象，进行正则化
        position_embedding = DeformableDetrSinePositionEmbedding(n_steps, normalize=True)
    # 根据配置选择位置编码类型为 "learned"
    elif config.position_embedding_type == "learned":
        # 使用 DeformableDetrLearnedPositionEmbedding 类创建学习位置编码对象
        position_embedding = DeformableDetrLearnedPositionEmbedding(n_steps)
    else:
        # 若配置中的位置编码类型不支持，则抛出异常
        raise ValueError(f"Not supported {config.position_embedding_type}")

    # 返回创建的位置编码对象
    return position_embedding


def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    # 获取 value 张量的维度信息
    batch_size, _, num_heads, hidden_dim = value.shape
    # 获取 sampling_locations 张量的维度信息
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # 根据 value 的空间形状将 value 分割为列表
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    # 计算采样网格位置
    sampling_grids = 2 * sampling_locations - 1
    # 初始化采样值列表
    sampling_value_list = []
    # 遍历 value 的空间形状和对应的 value 列表
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # 将 value_list[level_id] 展平并转置，以便进行 grid_sample 操作
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # 获取当前级别的采样网格
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # 使用 bilinear 插值方式进行 grid_sample，得到采样值
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        # 将采样值添加到列表中
        sampling_value_list.append(sampling_value_l_)
    # 调整注意力权重的形状，以便与采样值列表进行乘积操作
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    # 计算最终输出，将采样值与注意力权重相乘并求和，然后调整形状
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    # 调整输出的维度顺序并保持连续性
    return output.transpose(1, 2).contiguous()


class DeformableDetrMultiscaleDeformableAttention(nn.Module):
    """
    Deformable DETR 中提出的多尺度可变形注意力模块。
    """
    # 初始化函数，接受配置对象、注意力头数目和采样点数目作为参数
    def __init__(self, config: DeformableDetrConfig, num_heads: int, n_points: int):
        # 调用父类的初始化方法
        super().__init__()

        # 检查是否已经加载了多尺度可变形注意力的内核
        kernel_loaded = MultiScaleDeformableAttention is not None
        # 如果CUDA可用并且已安装Ninja并且内核未加载，则尝试加载CUDA内核
        if is_torch_cuda_available() and is_ninja_available() and not kernel_loaded:
            try:
                load_cuda_kernels()
            except Exception as e:
                # 记录警告信息，指出无法加载多尺度可变形注意力的自定义内核
                logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")

        # 检查配置中的d_model是否能被num_heads整除，否则抛出数值错误
        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )

        # 计算每个注意力头的维度
        dim_per_head = config.d_model // num_heads
        # 检查dim_per_head是否是2的幂
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            # 发出警告，建议将embed_dim设置为2的幂，这在CUDA实现中更有效
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        # 初始化im2col步长为64
        self.im2col_step = 64

        # 设置对象的属性值
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        # 初始化采样偏移量的线性层，输出维度为num_heads * n_levels * n_points * 2
        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        # 初始化注意力权重的线性层，输出维度为num_heads * n_levels * n_points
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        # 初始化值投影的线性层，输入和输出维度都是config.d_model
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        # 初始化输出投影的线性层，输入和输出维度都是config.d_model
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        # 设置是否禁用自定义内核的标志
        self.disable_custom_kernels = config.disable_custom_kernels

        # 调用内部方法_reset_parameters，用于初始化参数
        self._reset_parameters()

    # 内部方法，用于初始化模型的参数
    def _reset_parameters(self):
        # 将采样偏移量的权重初始化为常数0.0
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        # 获取默认数据类型
        default_dtype = torch.get_default_dtype()
        # 创建一组角度thetas，用于初始化采样偏移量
        thetas = torch.arange(self.n_heads, dtype=torch.int64).to(default_dtype) * (2.0 * math.pi / self.n_heads)
        # 初始化网格grid_init，形状为(n_heads, n_levels, n_points, 2)，用于采样偏移量
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        # 根据采样点的索引调整grid_init的值
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # 使用torch.no_grad()上下文管理器，设置采样偏移量的偏置值为grid_init
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # 将注意力权重的权重和偏置初始化为常数0.0
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        # 使用xavier_uniform方法初始化值投影和输出投影的权重，并将偏置初始化为常数0.0
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)
    # 如果位置嵌入不为 None，则将其加到输入张量上，实现位置编码的加法操作
    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    # Transformer 模型的前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选参数，默认为 None
        encoder_hidden_states=None,  # 编码器的隐藏状态，用于注意力机制中的键值对
        encoder_attention_mask=None,  # 编码器的注意力遮罩，用于注意力机制中的键值对
        position_embeddings: Optional[torch.Tensor] = None,  # 位置嵌入，可选参数，默认为 None
        reference_points=None,  # 参考点，用于空间注意力机制
        spatial_shapes=None,  # 空间形状，用于空间注意力机制
        level_start_index=None,  # 层级开始索引，用于分层注意力机制
        output_attentions: bool = False,  # 是否输出注意力权重，默认为 False
class DeformableDetrMultiheadAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the Deformable DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 设置注意力机制的输入/输出维度
        self.num_heads = num_heads  # 设置注意力头的数量
        self.dropout = dropout  # 设置dropout比率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 初始化投影矩阵 k
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 初始化投影矩阵 v
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 初始化投影矩阵 q
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 初始化输出投影矩阵

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # 重塑输入张量的形状，以便多头注意力机制处理

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings
        # 将位置嵌入添加到输入张量中，如果位置嵌入不为 None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,



class DeformableDetrEncoderLayer(nn.Module):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 设置编码器层的输入/输出维度
        self.self_attn = DeformableDetrMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )  # 初始化多尺度可变形注意力机制
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # Layer normalization层
        self.dropout = config.dropout  # dropout比率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout比率
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终的Layer normalization层

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    # 定义一个函数，用于处理多尺度变形注意力模块的输入数据
    def forward(
        hidden_states: `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            """
            Args:
                hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                    输入的张量数据，代表层的输入。
                attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                    注意力掩码，用于指示哪些元素需要被忽略。
                position_embeddings (`torch.FloatTensor`, *optional*):
                    位置嵌入，将被加到 `hidden_states` 上。
                reference_points (`torch.FloatTensor`, *optional*):
                    参考点。
                spatial_shapes (`torch.LongTensor`, *optional*):
                    主干特征图的空间形状。
                level_start_index (`torch.LongTensor`, *optional*):
                    级别起始索引。
                output_attentions (`bool`, *optional*):
                    是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions`。
            """
            # 将原始输入保存为残差连接的基础
            residual = hidden_states
    
            # 在多尺度特征图上应用多尺度变形注意力模块
            hidden_states, attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )
    
            # 应用 dropout 层，用于防止过拟合
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            
            # 将残差连接和处理后的数据相加
            hidden_states = residual + hidden_states
            
            # 应用自注意力层的 layer normalization
            hidden_states = self.self_attn_layer_norm(hidden_states)
    
            # 将处理后的数据保存为新的残差连接基础
            residual = hidden_states
    
            # 应用激活函数和全连接层 fc1
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            
            # 再次应用 dropout 层，用于进一步防止过拟合
            hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    
            # 应用最后的全连接层 fc2
            hidden_states = self.fc2(hidden_states)
            
            # 一最后一次应用 dropout 层
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    
            # 将残差连接和处理后的数据相加
            hidden_states = residual + hidden_states
            
            # 应用最后的 layer normalization
            hidden_states = self.final_layer_norm(hidden_states)
    
            # 在训练模式下，检查处理后的数据是否包含无穷大或 NaN
            if self.training:
                if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                    # 将数据的范围限制在一个较小的值域内，防止数值溢出
                    clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                    hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    
            # 将最终处理后的结果打包成一个 tuple 输出
            outputs = (hidden_states,)
    
            # 如果需要输出注意力权重，则将它们添加到输出中
            if output_attentions:
                outputs += (attn_weights,)
    
            return outputs
# 定义 DeformableDetrDecoderLayer 类，继承自 nn.Module
class DeformableDetrDecoderLayer(nn.Module):
    # 初始化方法，接受一个 DeformableDetrConfig 类型的 config 参数
    def __init__(self, config: DeformableDetrConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 embed_dim 属性为 config.d_model，即模型的维度
        self.embed_dim = config.d_model

        # self-attention
        # 初始化 self.self_attn 属性为 DeformableDetrMultiheadAttention 对象
        self.self_attn = DeformableDetrMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 设置 dropout 属性为 config.dropout
        self.dropout = config.dropout
        # 设置 activation_fn 属性为 ACT2FN[config.activation_function]，激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置 activation_dropout 属性为 config.activation_dropout
        self.activation_dropout = config.activation_dropout

        # 初始化 self.self_attn_layer_norm 属性为 nn.LayerNorm 对象，对 self-attention 结果进行归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # cross-attention
        # 初始化 self.encoder_attn 属性为 DeformableDetrMultiscaleDeformableAttention 对象
        self.encoder_attn = DeformableDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        # 初始化 self.encoder_attn_layer_norm 属性为 nn.LayerNorm 对象，对 cross-attention 结果进行归一化
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # feedforward neural networks
        # 初始化 self.fc1 属性为 nn.Linear 对象，进行线性变换
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 初始化 self.fc2 属性为 nn.Linear 对象，进行线性变换
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 初始化 self.final_layer_norm 属性为 nn.LayerNorm 对象，对最终输出进行归一化

    # 前向传播方法，接受多个输入参数并返回结果
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # 进行自注意力计算，通过 self.self_attn 属性
        # hidden_states 是输入的张量，根据给定的参数进行计算
        self_attn_output = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        # 对自注意力输出结果进行 dropout
        hidden_states = F.dropout(self_attn_output, p=self.dropout, training=self.training)
        # 将激活函数应用于输出
        hidden_states = self.activation_fn(hidden_states)
        # 对输出再次进行 dropout
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 对结果进行 LayerNorm 归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 进行跨注意力计算，通过 self.encoder_attn 属性
        # hidden_states 是上一步的输出，根据给定的参数进行计算
        encoder_attn_output = self.encoder_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        # 对跨注意力输出结果进行 dropout
        hidden_states = F.dropout(encoder_attn_output, p=self.dropout, training=self.training)
        # 对结果进行 LayerNorm 归一化
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # 使用第一个全连接层进行前馈神经网络计算
        hidden_states = self.fc1(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 使用第二个全连接层进行前馈神经网络计算
        hidden_states = self.fc2(hidden_states)
        # 对结果进行 LayerNorm 归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回处理后的结果
        return hidden_states
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        std = self.config.init_std  # 获取初始化标准差

        # 如果模块是 DeformableDetrLearnedPositionEmbedding 类型
        if isinstance(module, DeformableDetrLearnedPositionEmbedding):
            # 对行和列嵌入的权重进行均匀分布初始化
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        
        # 如果模块是 DeformableDetrMultiscaleDeformableAttention 类型
        elif isinstance(module, DeformableDetrMultiscaleDeformableAttention):
            # 调用模块的参数重置方法
            module._reset_parameters()
        
        # 如果模块是 nn.Linear, nn.Conv2d, nn.BatchNorm2d 中的一种
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 使用正态分布初始化权重，均值为0，标准差为设定的std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，则将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        
        # 如果模块是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为设定的std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果设置了padding_idx，将其对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        # 如果模块具有 "reference_points" 属性且不是两阶段配置
        if hasattr(module, "reference_points") and not self.config.two_stage:
            # 使用Xavier均匀分布初始化 reference_points 的权重
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            # 将 reference_points 的偏置项初始化为0
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        
        # 如果模块具有 "level_embed" 属性
        if hasattr(module, "level_embed"):
            # 使用正态分布初始化 level_embed 的权重
            nn.init.normal_(module.level_embed)
DEFORMABLE_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DeformableDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DEFORMABLE_DETR_INPUTS_DOCSTRING = r"""
    Inputs:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are not masked,
            - 0 for tokens that are masked.
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selecting a position_id equal to :obj:`padding_idx` will result in padding token. Position embeddings are
            not used by default in Deformable DETR. Therefore, this argument can be safely ignored.
        bbox (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, 4)`, optional):
            The normalized coordinates of the bounding boxes for the input queries.
            Coordinates are normalized in the format `(y_min, x_min, y_max, x_max)` and their values are in the
            interval `[0, 1]`.
        query_embed (:obj:`torch.FloatTensor` of shape :obj:`(num_queries, embed_dim)`, optional):
            The learnable embedding of each query token in the object queries. It is a learnable parameter initialized
            randomly if not provided.
        relation_embed (:obj:`torch.FloatTensor` of shape :obj:`(num_object_queries, num_object_queries, embed_dim)`, optional):
            The learnable embedding of each pair of object queries in the object queries.
            It is a learnable parameter initialized randomly if not provided.
        masks (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, num_object_queries)`, optional):
            The relation mask used to calculate the attention between object queries.
        return_dict (:obj:`bool`, optional, defaults to :obj:`True`):
            Whether or not to return a :obj:`Dict` with the output of the model. If set to :obj:`False`, returns a
            :obj:`Tuple` with the sequence of token logits and the attention.
            Returns:
                If :obj:`return_dict` is :obj:`True`, a :obj:`Dict` with the model's outputs will be returned that
                include the logits and hidden states.

    Returns:
        :obj:`Dict[str, torch.FloatTensor]`: Dictionary of outputs containing:
            - **logits** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, config.num_classes)`):
              Classification logits (scores) for each query.
            - **hidden_states** (:obj:`List[torch.FloatTensor]` of length :obj:`config.num_hidden_layers`):
              Hidden states for each layer in the model. Each hidden state is a :obj:`torch.FloatTensor` of shape
              :obj:`(batch_size, sequence_length, hidden_size)`.
"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下会忽略填充部分。

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DeformableDetrImageProcessor.__call__`]
            for details.
            # 可以使用 `AutoImageProcessor` 获取像素值。详见 [`DeformableDetrImageProcessor.__call__`]。

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            # 遮罩，用于在填充像素值上避免执行注意力操作。遮罩的值在 `[0, 1]` 之间：

            - 1 表示真实像素（即**未遮罩**），
            - 0 表示填充像素（即**已遮罩**）。

            [What are attention masks?](../glossary#attention-mask)
            # 注意力遮罩是什么？

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            # 默认不使用。可以用来遮罩对象查询。

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            # 元组包含 (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            # `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，可选部分是编码器最后一层的隐藏状态。
            # 在解码器的交叉注意力中使用。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选的，可以直接传递图像的平坦表示，而不是传递后骨干网络和投影层的输出特征图。

        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            # 可选的，可以直接传递嵌入表示来初始化查询，而不是用零张量初始化。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 获取更多细节。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states` 获取更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~file_utils.ModelOutput`] 而不是普通元组。
"""
class DeformableDetrEncoder(DeformableDetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`DeformableDetrEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: DeformableDetrConfig
    """

    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.gradient_checkpointing = False

        # 设置 dropout 概率
        self.dropout = config.dropout
        # 创建多个 DeformableDetrEncoderLayer 层，并放入 ModuleList 中
        self.layers = nn.ModuleList([DeformableDetrEncoderLayer(config) for _ in range(config.encoder_layers)])

        # 初始化权重并进行最终处理
        self.post_init()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        # 遍历每个特征图的空间形状
        for level, (height, width) in enumerate(spatial_shapes):
            # 创建网格矩阵，作为参考点的初始值
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
                indexing="ij",
            )
            # 对参考点进行调整，考虑有效比例因子和特征图的高度和宽度
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # 将参考点列表堆叠起来，形成最终的参考点张量
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    """
    Some tweaks for Deformable DETR:

    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.

    Args:
        config: DeformableDetrConfig
    """

    # 初始化函数，根据给定的配置参数初始化 Deformable DETR 模型
    def __init__(self, config: DeformableDetrConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设定模型中使用的 dropout 概率
        self.dropout = config.dropout
        # 创建多个 DeformableDetrDecoderLayer 层组成的列表
        self.layers = nn.ModuleList([DeformableDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 是否使用渐变检查点，默认为 False
        self.gradient_checkpointing = False

        # hack 实现，用于迭代边界框细化和两阶段 Deformable DETR
        self.bbox_embed = None  # 边界框嵌入，目前未指定具体的实现
        self.class_embed = None  # 类别嵌入，目前未指定具体的实现

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接收多个输入和参数，执行模型的前向计算过程
    def forward(
        self,
        inputs_embeds=None,  # 输入的嵌入表示，通常是编码器的输出
        encoder_hidden_states=None,  # 编码器的隐藏状态
        encoder_attention_mask=None,  # 编码器的注意力掩码
        position_embeddings=None,  # 位置嵌入，用于处理空间信息的嵌入向量
        reference_points=None,  # 参考点，用于变形注意力机制
        spatial_shapes=None,  # 空间形状，用于处理不同层次的空间信息
        level_start_index=None,  # 层级开始索引，用于多层级处理
        valid_ratios=None,  # 有效比率，用于多尺度处理
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出隐藏状态
        return_dict=None,  # 是否返回字典形式的输出
"""
The bare Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
hidden-states without any specific head on top.
"""
# 使用装饰器将类的文档字符串与已有的文档字符串合并
@add_start_docstrings(
    """
    The bare Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    DEFORMABLE_DETR_START_DOCSTRING,
)
# 定义 DeformableDetrModel 类，继承自 DeformableDetrPreTrainedModel 类
class DeformableDetrModel(DeformableDetrPreTrainedModel):
    # 构造函数，接收一个 DeformableDetrConfig 类型的 config 参数
    def __init__(self, config: DeformableDetrConfig):
        # 调用父类的构造函数
        super().__init__(config)

        # 创建 backbone + positional encoding
        # 使用 DeformableDetrConvEncoder 创建 backbone
        backbone = DeformableDetrConvEncoder(config)
        # 构建位置编码
        position_embeddings = build_position_encoding(config)
        # 将 backbone 和位置编码传递给 DeformableDetrConvModel，并赋值给 self.backbone
        self.backbone = DeformableDetrConvModel(backbone, position_embeddings)

        # 创建输入投影层
        if config.num_feature_levels > 1:
            # 获取 backbone 的中间通道大小列表
            num_backbone_outs = len(backbone.intermediate_channel_sizes)
            input_proj_list = []
            # 根据中间通道大小列表创建输入投影层列表
            for _ in range(num_backbone_outs):
                in_channels = backbone.intermediate_channel_sizes[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
            # 如果配置中的特征级别数大于 backbone 输出的特征级别数，则继续添加投影层
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
                in_channels = config.d_model
            # 将输入投影层列表转换为 ModuleList，并赋值给 self.input_proj
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # 如果只有一个特征级别，创建单个输入投影层并赋值给 self.input_proj
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                ]
            )

        # 如果不是两阶段模型，创建查询位置编码层
        if not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)

        # 创建 DeformableDetrEncoder 和 DeformableDetrDecoder 实例，并赋值给 self.encoder 和 self.decoder
        self.encoder = DeformableDetrEncoder(config)
        self.decoder = DeformableDetrDecoder(config)

        # 创建级别嵌入参数，并赋值给 self.level_embed
        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        # 如果是两阶段模型，创建额外的层和正则化
        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)
            self.pos_trans = nn.Linear(config.d_model * 2, config.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(config.d_model * 2)
        else:
            # 如果不是两阶段模型，创建参考点层
            self.reference_points = nn.Linear(config.d_model, 2)

        # 执行初始化后的操作
        self.post_init()

    # 返回 encoder 对象
    def get_encoder(self):
        return self.encoder

    # 返回 decoder 对象
    def get_decoder(self):
        return self.decoder

    # 冻结 backbone 的参数
    def freeze_backbone(self):
        # 遍历 backbone 的模型参数，并设置为不可训练
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)
    def unfreeze_backbone(self):
        # 解冻模型的骨干网络（backbone）中的所有参数，使其可以进行梯度计算
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""
        
        # 获取掩码（mask）的高度和宽度
        _, height, width = mask.shape
        # 计算每个特征图在高度和宽度上的有效比例
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_height = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        # 将高度和宽度的有效比例组合成一个张量
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals):
        """Get the position embedding of the proposals."""
        
        # 获取位置嵌入（position embedding）的维度
        num_pos_feats = self.config.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        # 生成维度张量，用于计算位置嵌入
        dim_t = torch.arange(num_pos_feats, dtype=torch.int64, device=proposals.device).float()
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        
        # 对提议框进行 sigmoid 转换，并乘以比例尺度
        proposals = proposals.sigmoid() * scale
        
        # 计算位置嵌入，将结果展开为(batch_size, num_queries, 512)的形式
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos
    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (Tensor[num_feature_levels, 2]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]  # 获取批量大小
        proposals = []  # 初始化建议列表
        _cur = 0  # 当前处理的位置索引初始化为0
        for level, (height, width) in enumerate(spatial_shapes):  # 遍历空间形状列表
            mask_flatten_ = padding_mask[:, _cur : (_cur + height * width)].view(batch_size, height, width, 1)  # 根据当前级别的高度和宽度计算扁平化的掩码
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)  # 计算有效的高度
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)  # 计算有效的宽度

            grid_y, grid_x = meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )  # 创建网格坐标

            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # 合并网格坐标

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)  # 计算比例
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale  # 根据比例调整网格
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0**level)  # 计算宽度和高度
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)  # 构建建议
            proposals.append(proposal)  # 将建议添加到列表中
            _cur += height * width  # 更新当前位置索引

        output_proposals = torch.cat(proposals, 1)  # 合并所有建议
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)  # 确定有效的建议
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # 对建议进行逆sigmoid转换
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))  # 将填充位置置为无穷大
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))  # 将无效的建议位置置为无穷大

        # 每个像素分配为一个对象查询
        object_query = enc_output  # 使用编码输出作为对象查询
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))  # 将填充位置置为0
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))  # 将无效的建议位置置为0
        object_query = self.enc_output_norm(self.enc_output(object_query))  # 对对象查询进行归一化处理
        return object_query, output_proposals  # 返回对象查询和输出建议
    # 给模型的前向传播方法添加文档字符串，文档字符串的内容来源于 DEFORMABLE_DETR_INPUTS_DOCSTRING
    @add_start_docstrings_to_model_forward(DEFORMABLE_DETR_INPUTS_DOCSTRING)
    # 替换前向传播方法的返回文档字符串，指定输出类型为 DeformableDetrModelOutput，配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=DeformableDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
top, for tasks such as COCO detection.
"""
# 导入开始文档字符串装饰器和相关的模块文档字符串
@add_start_docstrings(
    """
    Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    DEFORMABLE_DETR_START_DOCSTRING,
)
# 继承自预训练的 Deformable DETR 模型
class DeformableDetrForObjectDetection(DeformableDetrPreTrainedModel):
    # 当使用克隆时，所有大于 0 的层都将被克隆，但层 0 是必需的
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*", r"class_embed\.[1-9]\d*"]
    # 不能在元设备上初始化模型，因为某些权重在初始化过程中会被修改
    _no_split_modules = None

    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)

        # Deformable DETR encoder-decoder 模型
        self.model = DeformableDetrModel(config)

        # 放置在顶部的检测头
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        # 设置先验概率和偏置值
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # 如果是两阶段模型，最后的 class_embed 和 bbox_embed 用于区域提议生成
        num_pred = (config.decoder_layers + 1) if config.two_stage else config.decoder_layers
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # 对迭代式边界框细化的 hack 实现
            self.model.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None
        if config.two_stage:
            # 对两阶段模型的 hack 实现
            self.model.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 https://github.com/facebookresearch/detr/blob/master/models/detr.py 中获取的未使用的 torch.jit 注解
    @torch.jit.unused
    # 设置辅助损失函数，接受分类输出和坐标输出作为参数
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是为了使 torchscript 能够正常工作的一种解决方法，因为 torchscript
        # 不支持包含非同质值的字典，例如既有张量又有列表的字典。
        # 返回一个列表，其中每个元素是一个字典，包含"logits"和"pred_boxes"两个键，分别对应 outputs_class 和 outputs_coord 的每个元素（除最后一个）。
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # 将模型前向方法（forward）添加文档字符串
    @add_start_docstrings_to_model_forward(DEFORMABLE_DETR_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串为 DeformableDetrObjectDetectionOutput 类型，使用 _CONFIG_FOR_DOC 作为配置类
    @replace_return_docstrings(output_type=DeformableDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# Copied from transformers.models.detr.modeling_detr.dice_loss
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    # 对模型输出进行 sigmoid 激活，使其在 (0, 1) 范围内
    inputs = inputs.sigmoid()
    # 将输入扁平化，以便计算损失
    inputs = inputs.flatten(1)
    # 计算 DICE 损失的分子部分
    numerator = 2 * (inputs * targets).sum(1)
    # 计算 DICE 损失的分母部分
    denominator = inputs.sum(-1) + targets.sum(-1)
    # 计算最终的 DICE 损失
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 对所有样本的损失求和并取平均
    return loss.sum() / num_boxes


# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    # 对模型输出进行 sigmoid 激活，将其转换为概率值
    prob = inputs.sigmoid()
    # 使用二元交叉熵损失计算损失，reduction="none"表示不进行求和
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 计算 modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # 计算最终的 focal loss
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        # 计算 alpha 加权
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 对所有样本的损失求和并取平均
    return loss.mean(1).sum() / num_boxes


class DeformableDetrLoss(nn.Module):
    """
    This class computes the losses for `DeformableDetrForObjectDetection`. The process happens in two steps: 1) we
    compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of
    matched ground-truth / prediction (supervise class and box).

    Args:
        matcher (`DeformableDetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """
    def __init__(self, matcher, num_classes, focal_alpha, losses):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses



        # 初始化函数，设置模型的匹配器、类别数、focal loss 的 alpha 参数和损失函数
        super().__init__()
        # 保存参数到对象实例中
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses



    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        # 检查输出中是否存在 "logits" 键
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        # 获取模型输出中的 logits
        source_logits = outputs["logits"]

        # 获取源索引的排列顺序
        idx = self._get_source_permutation_idx(indices)
        # 从目标中提取类别标签
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        # 创建一个填充了默认类别值的张量
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        # 创建一个 one-hot 编码的类别张量
        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        # 在目标类别张量上进行 scatter 操作，填充 one-hot 编码
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # 去除多余的最后一个类别维度
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        # 计算分类交叉熵损失
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        # 返回损失字典
        losses = {"loss_ce": loss_ce}

        return losses



    @torch.no_grad()
    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_cardinality
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        # 获取模型输出中的 logits
        logits = outputs["logits"]
        # 确定设备类型
        device = logits.device
        # 计算目标长度的张量
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算预测的非空盒子数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 计算基于 L1 损失的基数错误
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        # 返回基数错误的损失字典
        losses = {"cardinality_error": card_err}
        return losses



    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_boxes



    # 从 transformers.models.detr.modeling_detr.DetrLoss.loss_boxes 复制过来
    # 定义计算边界框损失的方法，包括 L1 回归损失和 GIoU 损失
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # 检查输出中是否包含预测的边界框
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        # 根据 indices 获取源排列的索引
        idx = self._get_source_permutation_idx(indices)
        # 获取预测的边界框和目标边界框，并按照 indices 给定的顺序连接起来
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算 L1 损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        # 将 L1 损失求和并归一化
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算 GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        # 将 GIoU 损失求和并归一化
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    # 从 DETR 模型中复制的方法，用于获取源排列的索引
    def _get_source_permutation_idx(self, indices):
        # 根据 indices 重新排列预测结果
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # 从 DETR 模型中复制的方法，用于获取目标排列的索引
    def _get_target_permutation_idx(self, indices):
        # 根据 indices 重新排列目标标签
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    # 根据给定的损失类型选择相应的损失计算方法，并调用
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)
    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        # Filter out auxiliary outputs from the main outputs dictionary
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs" and k != "enc_outputs"}

        # Retrieve the indices that match the outputs with the corresponding targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the total number of target boxes for normalization
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1

        # Adjust num_boxes and world_size if using the `accelerate` library
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes

        # Normalize num_boxes and clamp the result to ensure it's at least 1
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all requested losses and store them in the losses dictionary
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # If there are auxiliary outputs, compute losses for each and append to the losses dictionary
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # If there are encoder outputs, compute losses specific to these outputs and add to the losses dictionary
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])  # Zero out class labels
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # Return the computed losses dictionary
        return losses
# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class DeformableDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 创建一个由多个线性层组成的神经网络，用于预测边界框的中心坐标、高度和宽度
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # 前向传播函数，通过多个线性层进行特征提取和预测
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        # 引入后端依赖的函数库
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # 如果所有的成本都为零，则抛出异常
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # 将分类 logits 展平并应用 sigmoid 函数，得到概率 [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # 将预测框坐标展平 [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])  # 将所有目标的类别标签拼接起来
        target_bbox = torch.cat([v["boxes"] for v in targets])  # 将所有目标的框坐标拼接起来

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())  # 计算分类损失中的负类损失项
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())  # 计算分类损失中的正类损失项
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]  # 根据目标类别计算分类损失

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)  # 计算框之间的 L1 损失

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))  # 计算框之间的 GIoU 损失

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost  # 组合成最终的损失矩阵
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()  # 将损失矩阵调整形状并转移到 CPU 上处理

        sizes = [len(v["boxes"]) for v in targets]  # 获取每个目标的框数量
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]  # 使用匈牙利算法计算最佳匹配索引

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  # 将匹配索引转换为张量并返回
# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # 如果输入张量是浮点型，则保护免受乘法溢出风险，通过升级到相应更高的类型进行处理
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，这些边界框由它们的 (x1, y1, x2, y2) 坐标指定。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            需要计算面积的边界框。它们应以 (x1, y1, x2, y2) 格式提供，其中 `0 <= x1 < x2` 和 `0 <= y1 < y2`。

    Returns:
        `torch.FloatTensor`: 包含每个边界框面积的张量。
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    来自 https://giou.stanford.edu/ 的广义 IoU 计算方法。边界框应处于 [x0, y0, x1, y1] (角点) 格式。

    Returns:
        `torch.FloatTensor`: 一个 [N, M] 的成对矩阵，其中 N = len(boxes1)，M = len(boxes2)
    """
    # 退化的边界框会产生无穷大 / NaN 的结果，因此进行早期检查
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 必须以 [x0, y0, x1, y1] (角点) 格式提供，但给定的是 {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 必须以 [x0, y0, x1, y1] (角点) 格式提供，但给定的是 {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# Copied from transformers.models.detr.modeling_detr._max_by_axis
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes
# 定义了一个 NestedTensor 类，用于处理包含张量和可选遮罩的嵌套张量对象
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors  # 初始化对象时传入的张量列表
        self.mask = mask  # 初始化对象时传入的遮罩张量（可选）

    # 将嵌套张量对象转移到指定的设备上
    def to(self, device):
        cast_tensor = self.tensors.to(device)  # 将张量列表转移到指定设备上
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)  # 如果存在遮罩张量，将其也转移到指定设备上
        else:
            cast_mask = None  # 如果没有遮罩张量，则设置为 None
        return NestedTensor(cast_tensor, cast_mask)  # 返回转移后的嵌套张量对象

    # 返回嵌套张量对象的原始张量和遮罩张量（如果存在）
    def decompose(self):
        return self.tensors, self.mask

    # 返回嵌套张量对象的字符串表示，即其张量列表的字符串表示
    def __repr__(self):
        return str(self.tensors)


# 从给定的张量列表创建嵌套张量对象
# 函数来自于 transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:  # 检查张量列表中的第一个张量是否为三维张量
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])  # 获取张量列表中张量的最大尺寸
        batch_shape = [len(tensor_list)] + max_size  # 计算批次的形状
        batch_size, num_channels, height, width = batch_shape  # 解构批次形状
        dtype = tensor_list[0].dtype  # 获取张量的数据类型
        device = tensor_list[0].device  # 获取张量的设备
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)  # 创建全零张量作为批次张量
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)  # 创建全一的遮罩张量
        # 将每个张量复制到批次张量中，并生成相应的遮罩
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False  # 根据张量的实际尺寸设置遮罩
    else:
        raise ValueError("Only 3-dimensional tensors are supported")  # 抛出错误，只支持三维张量
    return NestedTensor(tensor, mask)  # 返回创建的嵌套张量对象
```