# `.\models\deformable_detr\modeling_deformable_detr.py`

```
# 设定文件编码为utf-8
# 版权声明，保留所有权利
#
# 根据Apache许可证2.0许可使用此文件
# 除依照许可证允许的使用外，您不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按“原样”分发，
# 没有任何明示或暗示的担保或条件
# 查看特定语言的许可证以获取权限和限制

""" PyTorch Deformable DETR model."""

# 引入必要的包和模块
import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

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
from ...utils import is_ninja_available, logging
from ..auto import AutoBackbone
from .configuration_deformable_detr import DeformableDetrConfig
from .load_custom import load_cuda_kernels


logger = logging.get_logger(__name__)

# 如果torch cuda可用，并且ninja可用，则加载自定义CUDA核心
if is_torch_cuda_available() and is_ninja_available():
    logger.info("Loading custom CUDA kernels...")
    try:
        MultiScaleDeformableAttention = load_cuda_kernels()
    except Exception as e:
        logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")
        MultiScaleDeformableAttention = None
else:
    MultiScaleDeformableAttention = None

# 如果视觉包可用，则导入相关的模块
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


# 定义多尺度可变形注意力函数
class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    # 定义前向传播
    def forward(
        context,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        # 设置上下文对象的 im2col_step 属性为给定的 im2col_step
        context.im2col_step = im2col_step
        # 调用 MultiScaleDeformableAttention 类的 ms_deform_attn_forward 方法，进行多尺度可变形注意力前向传播
        output = MultiScaleDeformableAttention.ms_deform_attn_forward(
            value,  # 输入值
            value_spatial_shapes,  # 输入值的空间形状
            value_level_start_index,  # 输入值级别的起始索引
            sampling_locations,  # 采样位置
            attention_weights,  # 注意力权重
            context.im2col_step,  # im2col 步骤数
        )
        # 将当前计算所需的值保存在上下文中，以备后向传播使用
        context.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        # 返回前向传播的输出
        return output

    @staticmethod
    @once_differentiable
    def backward(context, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = context.saved_tensors
        # 调用 MultiScaleDeformableAttention 类的 ms_deform_attn_backward 方法，进行多尺度可变形注意力的反向传播
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttention.ms_deform_attn_backward(
            value,  # 输入值
            value_spatial_shapes,  # 输入值的空间形状
            value_level_start_index,  # 输入值级别的起始索引
            sampling_locations,  # 采样位置
            attention_weights,  # 注意力权重
            grad_output,  # 梯度输出
            context.im2col_step,  # im2col 步骤数
        )

        # 返回输入值、采样位置梯度、注意力权重梯度以及其它梯度（空），用于后续计算
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
# 如果 scipy 可用，导入线性求解模块
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 如果 timm 可用，导入模型创建函数
if is_timm_available():
    from timm import create_model

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 配置文档字符串
_CONFIG_FOR_DOC = "DeformableDetrConfig"
# 检查点文档字符串
_CHECKPOINT_FOR_DOC = "sensetime/deformable-detr"

# Deformable DETR 预训练模型存档列表
DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sensetime/deformable-detr",
    # 查看所有 Deformable DETR 模型: https://huggingface.co/models?filter=deformable-detr
]

# 定义 DeformableDetrDecoderOutput 类
@dataclass
class DeformableDetrDecoderOutput(ModelOutput):
    """
    对 DeformableDetrDecoder 的输出进行描述
    添加了两个属性到 BaseModelOutputWithCrossAttentions，即:
    - 一个堆叠的中间解码器隐藏状态张量（即每个解码器层的输出）
    - 一个堆叠的中间参考点张量

    参数:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            堆叠的中间隐藏状态（解码器每一层的输出）
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            堆叠的中间参考点（解码器每一层的参考点）
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型隐藏状态（每一层的输出 + 初始嵌入输出）的元组
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            自注意力头中注意力权重的元组，用于计算加权平均值
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            解码器交叉注意力层中注意力权重的元组，用于计算加权平均值
    """
    
    # 最后一个隐藏状态张量
    last_hidden_state: torch.FloatTensor = None
    # 中间隐藏状态张量
    intermediate_hidden_states: torch.FloatTensor = None
    # 创建一个类型为 torch.FloatTensor 的变量 intermediate_reference_points，初始值为 None
    intermediate_reference_points: torch.FloatTensor = None
    # 创建一个类型为 Optional[Tuple[torch.FloatTensor]] 的变量 hidden_states，初始值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 创建一个类型为 Optional[Tuple[torch.FloatTensor]] 的变量 attentions，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 创建一个类型为 Optional[Tuple[torch.FloatTensor]] 的变量 cross_attentions，初始值为 None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储 Deformable DETR 模型的输出
@dataclass
class DeformableDetrModelOutput(ModelOutput):
    """
    Base class for outputs of the Deformable DETR encoder-decoder model.

    """

    # 初始参考点张量，默认为 None
    init_reference_points: torch.FloatTensor = None
    # 最后的隐藏状态张量，默认为 None
    last_hidden_state: torch.FloatTensor = None
    # 中间隐藏状态张量，默认为 None
    intermediate_hidden_states: torch.FloatTensor = None
    # 中间参考点张量，默认为 None
    intermediate_reference_points: torch.FloatTensor = None
    # 解码器的隐藏状态张量的可选元组，默认为 None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力张量的可选元组，默认为 None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力张量的可选元组，默认为 None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的最后隐藏状态张量的可选元组，默认为 None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态张量的可选元组，默认为 None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力张量的可选元组，默认为 None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器输出的类别张量的可选项，默认为 None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    # 编码器输出的坐标 logits 张量的可选项，默认为 None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None


# 定义一个数据类，用于存储 Deformable DETR 对象检测模型的输出
@dataclass
class DeformableDetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`DeformableDetrForObjectDetection`].

    """

    # 损失张量的可选项，默认为 None
    loss: Optional[torch.FloatTensor] = None
    # 损失字典的可选项，默认为 None
    loss_dict: Optional[Dict] = None
    # logits 张量，默认为 None
    logits: torch.FloatTensor = None
    # 预测框张量，默认为 None
    pred_boxes: torch.FloatTensor = None
    # 辅助输出的可选列表，默认为 None
    auxiliary_outputs: Optional[List[Dict]] = None
    # 初始参考点张量的可选项，默认为 None
    init_reference_points: Optional[torch.FloatTensor] = None
    # 最后的隐藏状态张量的可选项，默认为 None
    last_hidden_state: Optional[torch.FloatTensor] = None
    # 中间隐藏状态张量的可选项，默认为 None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    # 中间参考点张量的可选项，默认为 None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    # 解码器的隐藏状态张量的可选元组，默认为 None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力张量的可选元组，默认为 None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力张量的可选元组，默认为 None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的最后隐藏状态张量的可选项，默认为 None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态张量的可选元组，默认为 None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力张量的可选元组，默认为 None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器输出的类别张量的可选项，默认为 None
    enc_outputs_class: Optional = None
    # 编码器输出的坐标 logits 张量的可选项，默认为 None
    enc_outputs_coord_logits: Optional = None


# 定义一个函数，用于克隆给定模块 N 次
def _get_clones(module, N):
    # 返回一个包含 N 个 module 的 ModuleList
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# 定义一个函数，用于计算逆 Sigmoid 函数的值
def inverse_sigmoid(x, eps=1e-5):
    # 限制 x 的取值范围在 [0, 1]，避免出现无效值
    x = x.clamp(min=0, max=1)
    # 将 x 限制在 (eps, 1) 之间，避免分母为 0
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    # 计算逆 Sigmoid 函数值
    return torch.log(x1 / x2)


# 从 `transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d` 复制过来的类，将 Detr 改为 DeformableDetr
class DeformableDetrFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    # 初始化函数
    def __init__(self, n):
        super().__init__()
        # 注册固定的权重张量，初始化为全 1
        self.register_buffer("weight", torch.ones(n))
        # 注册固定的偏置张量，初始化为全 0
        self.register_buffer("bias", torch.zeros(n))
        # 注册固定的均值张量，初始化为全 0
        self.register_buffer("running_mean", torch.zeros(n))
        # 注册固定的方差张量，初始化为全 1
        self.register_buffer("running_var", torch.ones(n))
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 构建 num_batches_tracked 在 state_dict 中的 key
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # 如果 num_batches_tracked 在 state_dict 中，则删除它
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类的 _load_from_state_dict 方法，加载参数
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # 将权重重塑为适合张量操作的形状
        weight = self.weight.reshape(1, -1, 1, 1)
        # 将偏置重塑为适合张量操作的形状
        bias = self.bias.reshape(1, -1, 1, 1)
        # 将运行时方差重塑为适合张量操作的形状
        running_var = self.running_var.reshape(1, -1, 1, 1)
        # 将运行时均值重塑为适合张量操作的形状
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        # 设置 epsilon 值
        epsilon = 1e-5
        # 计算缩放值
        scale = weight * (running_var + epsilon).rsqrt()
        # 计算偏移值
        bias = bias - running_mean * scale
        # 返回经批归一化后的张量
        return x * scale + bias
# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->DeformableDetr
# 从transformers.models.detr.modeling_detr.replace_batch_norm复制并将Detr->DeformableDetr
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `DeformableDetrFrozenBatchNorm2d`.
    递归替换所有`torch.nn.BatchNorm2d`层为`DeformableDetrFrozenBatchNorm2d`。

    Args:
        model (torch.nn.Module):
            input model
            输入模型
    """
    # 遍历模型的每个子模块
    for name, module in model.named_children():
        # 如果当前子模块是`nn.BatchNorm2d`类型
        if isinstance(module, nn.BatchNorm2d):
            # 创建一个`DeformableDetrFrozenBatchNorm2d`
            new_module = DeformableDetrFrozenBatchNorm2d(module.num_features)

            # 如果`module`的权重不在设备"meta"上，则复制权重、偏置、running mean和running variance的数据
            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            # 用新创建的模块替换原有的模块
            model._modules[name] = new_module

        # 如果当前子模块的子模块数大于0，则继续递归地替换其子模块
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class DeformableDetrConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.
    卷积骨架，使用AutoBackbone API或来自timm库的一个。

    nn.BatchNorm2d layers are replaced by DeformableDetrFrozenBatchNorm2d as defined above.
    通过上面定义的方法，nn.BatchNorm2d层被DeformableDetrFrozenBatchNorm2d替换。

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # 如果使用timm骨干网络
        if config.use_timm_backbone:
            # 判断是否安装了timm
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            # 创建timm骨干网络
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(2, 3, 4) if config.num_feature_levels > 1 else (4,),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            # 使用AutoBackbone API创建骨干网络
            backbone = AutoBackbone.from_config(config.backbone_config)

        # 使用DeformableDetrFrozenBatchNorm2d替换批归一化层
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        # 如果使用timm骨干网络，获取中间层通道数信息
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)

    # Copied from transformers.models.detr.modeling_detr.DetrConvEncoder.forward with Detr->DeformableDetr
    # 定义一个函数，输入为像素数值和像素掩码
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # 如果配置使用了 timm 的 backbone，将像素数值传递给模型得到特征图列表，否则将像素数值传递给模型得到 feature_maps
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        out = []
        # 遍历特征图列表
        for feature_map in features:
            # 将像素掩码的大小下采样至对应特征图的大小
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            # 将特征图和下采样后的像素掩码组成元组，添加到列表中
            out.append((feature_map, mask))
        # 返回特征图和相应像素掩码组成的列表
        return out
# 从transformers.models.detr.modeling_detr.DetrConvModel复制并将Detr更改为DeformableDetr
class DeformableDetrConvModel(nn.Module):
    """
    这个模块为卷积编码器的所有中间特征图添加2D位置嵌入。
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):
        # 通过骨干网络将pixel_values和pixel_mask发送，以获得 (feature_map, pixel_mask) 元组的列表
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # 位置编码
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class DeformableDetrSinePositionEmbedding(nn.Module):
    """
    这是一个更标准的位置嵌入版本，非常类似于Attention is all you need论文中使用的版本，可以推广到工作在图像上。
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("如果传递了scale，则normalize应该为True")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("未提供像素蒙版")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# 从transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding复制
class DeformableDetrLearnedPositionEmbedding(nn.Module):
    """
    这个模块学习固定最大尺寸的位置嵌入。
    """
    # 初始化函数，设置嵌入维度为256
    def __init__(self, embedding_dim=256):
        # 调用父类的初始化函数
        super().__init__()
        # 创建行坐标的嵌入层，共有50个坐标，每个坐标的嵌入维度为embedding_dim
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        # 创建列坐标的嵌入层，共有50个坐标，每个坐标的嵌入维度为embedding_dim
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    # 正向传播函数
    def forward(self, pixel_values, pixel_mask=None):
        # 获取像素值的高度和宽度
        height, width = pixel_values.shape[-2:]
        # 创建宽度值张量，值从0到width-1
        width_values = torch.arange(width, device=pixel_values.device)
        # 创建高度值张量，值从0到height-1
        height_values = torch.arange(height, device=pixel_values.device)
        # 对列坐标进行嵌入得到x_emb
        x_emb = self.column_embeddings(width_values)
        # 对行坐标进行嵌入得到y_emb
        y_emb = self.row_embeddings(height_values)
        # 沿着最后一个维度进行拼接，得到位置张量pos
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # 调换pos张量的维度顺序
        pos = pos.permute(2, 0, 1)
        # 在第0维添加一个维度
        pos = pos.unsqueeze(0)
        # 将pos张量在第0维上进行复制，复制pixel_values.shape[0]次
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # 返回位置张量
        return pos
# 从transformers.models.detr.modeling_detr.build_position_encoding复制并修改为DeformableDetr
def build_position_encoding(config):
    # 计算位置编码矩阵的步数
    n_steps = config.d_model // 2
    # 根据配置选择使用正弦函数的位置编码或者学习的位置编码
    if config.position_embedding_type == "sine":
        # 创建基于正弦函数的位置编码对象
        position_embedding = DeformableDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        # 创建基于学习的位置编码对象
        position_embedding = DeformableDetrLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    # 获取批量大小、注意力头数、隐藏维度
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # 将输入的value按空间形状分割成列表
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    # 生成采样网格
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # 将value列表中的每个元素展平并重新排列维度
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # 将采样网格转置并展平
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # 使用双线性插值的方式，根据采样网格在value上获取采样值
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # 调整注意力权重的维度顺序并reshape
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    # 计算输出值
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    # 调整输出的维度顺序并进行连续化
    return output.transpose(1, 2).contiguous()


class DeformableDetrMultiscaleDeformableAttention(nn.Module):
    """
    Deformable DETR中提出的多尺度可变注意力。
    """
    # 初始化函数，接受配置文件、注意力头数和采样点数作为参数
    def __init__(self, config: DeformableDetrConfig, num_heads: int, n_points: int):
        # 调用父类的初始化函数
        super().__init__()
        # 如果嵌入维度不能被注意力头数整除，抛出数值错误
        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        # 计算每个注意力头的维度
        dim_per_head = config.d_model // num_heads
        # 检查 dim_per_head 是否是2的幂
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            # 如果不是2的幂，发出警告
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        # 设置 im2col 步长为 64
        self.im2col_step = 64

        # 保存配置中的模型维度、特征层级数、注意力头数和采样点数
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        # 创建线性层，用于学习采样偏移
        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        # 创建线性层，用于学习注意力权重
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        # 创建线性层，用于值的投影
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        # 创建线性层，用于输出的投影
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        # 设置是否禁用自定义内核的标志
        self.disable_custom_kernels = config.disable_custom_kernels

        # 重置参数
        self._reset_parameters()

    # 重置参数的函数
    def _reset_parameters(self):
        # 初始化采样偏移的权重为常数 0
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        
        # 生成一些角度，并初始化采样偏移的偏置
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # 初始化注意力权重的权重和偏置为常数 0
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        
        # 用均匀分布初始化值的投影权重
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        
        # 用均匀分布初始化输出的投影权重
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    # 将位置编码加到张量上的函数
    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        # 如果没有位置编码，返回原始张量，否则返回原始张量加上位置编码
        return tensor if position_embeddings is None else tensor + position_embeddings
    # 前向传播函数，用于模型的前向计算
    def forward(
        # 隐藏状态，输入张量
        hidden_states: torch.Tensor,
        # 注意力遮罩，可选输入张量，默认为None
        attention_mask: Optional[torch.Tensor] = None,
        # 编码器隐藏状态，默认为None
        encoder_hidden_states=None,
        # 编码器注意力遮罩，默认为None
        encoder_attention_mask=None,
        # 位置嵌入，可选输入张量，默认为None
        position_embeddings: Optional[torch.Tensor] = None,
        # 参考点，默认为None
        reference_points=None,
        # 空间形状，默认为None
        spatial_shapes=None,
        # 层级起始索引，默认为None
        level_start_index=None,
        # 输出注意力值，布尔类型，默认为False
        output_attentions: bool = False,
class DeformableDetrMultiheadAttention(nn.Module):
    """
    多头注意力机制，基于《Attention Is All You Need》论文。

    在这里，我们将位置嵌入添加到查询和键中（如《Deformable DETR》论文中所解释）。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        # 初始化注意力机制的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        # 检查 embed_dim 是否可以整除 num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        # 缩放因子
        self.scaling = self.head_dim**-0.5

        # 用于投影 keys、values 和 queries 的线性层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 输出层的线性层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        # 将张量重塑为适合多头注意力的形状
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        # 如果有位置嵌入，将其添加到张量中
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
class DeformableDetrEncoderLayer(nn.Module):
    """
    Deformable DETR 编码器层。
    """

    def __init__(self, config: DeformableDetrConfig):
        super().__init__()
        # 初始化编码器层的参数
        self.embed_dim = config.d_model
        # 自注意力层
        self.self_attn = DeformableDetrMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        # 自注意力层的 LayerNormalization
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        # 激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        # 前馈全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 前馈全连接层后的 LayerNormalization
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                输入到该层的数据。
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                注意力遮罩。
            position_embeddings (`torch.FloatTensor`, *optional*):
                位置嵌入，待添加到 `hidden_states` 上。
            reference_points (`torch.FloatTensor`, *optional*):
                参考点。
            spatial_shapes (`torch.LongTensor`, *optional*):
                主干特征图的空间形状。
            level_start_index (`torch.LongTensor`, *optional*):
                级别起始索引。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请查看返回的张量中的 `attentions`。
        """
        residual = hidden_states

        # 在多尺度特征图上应用多尺度变形注意模块。
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

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
class DeformableDetrDecoderLayer(nn.Module):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = DeformableDetrMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # cross-attention
        self.encoder_attn = DeformableDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

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
)
# Copied from transformers.models.detr.modeling_detr.DetrClassificationHead
class DeformableDetrClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class DeformableDetrPreTrainedModel(PreTrainedModel):
    config_class = DeformableDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DeformableDetrConvEncoder", r"DeformableDetrEncoderLayer", r"DeformableDetrDecoderLayer"]
    # 初始化权重函数，接受一个模块作为参数
    def _init_weights(self, module):
        # 从配置中获取初始化标准差
        std = self.config.init_std

        # 如果模块是 DeformableDetrLearnedPositionEmbedding 类型
        if isinstance(module, DeformableDetrLearnedPositionEmbedding):
            # 对行嵌入权重进行均匀分布初始化
            nn.init.uniform_(module.row_embeddings.weight)
            # 对列嵌入权重进行均匀分布初始化
            nn.init.uniform_(module.column_embeddings.weight)
        # 如果模块是 DeformableDetrMultiscaleDeformableAttention 类型
        elif isinstance(module, DeformableDetrMultiscaleDeformableAttention):
            # 重置多尺度可变形注意力模块的参数
            module._reset_parameters()
        # 如果模块是 nn.Linear、nn.Conv2d、nn.BatchNorm2d 中的一种
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 使用正态分布初始化权重，均值为 0，标准差为预设的标准差
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果模块存在偏置，则将偏置数据初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0，标准差为预设的标准差
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果模块设置了填充索引，则将对应索引的权重初始化为 0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块具有 "reference_points" 属性且不是两阶段模型
        if hasattr(module, "reference_points") and not self.config.two_stage:
            # 使用 xavier 均匀分布初始化参考点权重，增益为 1.0
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            # 将参考点的偏置初始化为 0
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        # 如果模块具有 "level_embed" 属性
        if hasattr(module, "level_embed"):
            # 使用正态分布初始化级别嵌入
            nn.init.normal_(module.level_embed)
# 定义 DEFORMABLE_DETR_START_DOCSTRING 字符串，用于说明模型继承关系和参数配置
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
# 定义 DEFORMABLE_DETR_INPUTS_DOCSTRING 字符串，用于说明模型的输入
DEFORMABLE_DETR_INPUTS_DOCSTRING = r"""
``` 
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DeformableDetrImageProcessor.__call__`]
            for details.
        # 像素值。默认情况下将忽略填充。
        # 如果提供了填充，将被忽略。
        # 像素值可以使用[`AutoImageProcessor`]获取。详见[`DeformableDetrImageProcessor.__call__`]。
        
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        # 避免对填充像素值进行注意力操作的掩蔽。
        # 掩蔽值取值范围为`[0, 1]`：

        # - 1 表示像素为真（即**未掩蔽**），
        # - 0 表示像素为填充（即**掩蔽**）。

        # [什么是注意力掩码？](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        # 默认情况下不使用。可以用于掩蔽对象查询。

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        # 元组由 (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`) 构成
        # `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*optional*) 是编码器最后一层输出的隐藏状态序列。
        # 用于解码器的交叉注意力中。
        
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        # 可选项，可以直接传入图像的扁平化表示，而不是传递扁平化特征图（骨干网络 + 投影层的输出）。
        
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        # 可选项，可以选择通过直接传入嵌入表示来初始化查询，而不是使用零张量进行初始化。
        
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        # 是否返回所有注意力层的注意力张量。在返回的张量中查看详细信息，请参见`attentions`。
        
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # 是否返回所有层的隐藏状态。在返回的张量中查看详细信息，请参见`hidden_states`。

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        # 是否返回[`~file_utils.ModelOutput`]，而不是一个普通的元组。
# 定义 DeformableDetrEncoder 类，继承自 DeformableDetrPreTrainedModel
# DeformableDetrEncoder 是一个 Transformer 编码器，由 config.encoder_layers 个可变注意力层组成。每一层都是 DeformableDetrEncoderLayer
# 该编码器通过多个可变注意力层更新了多尺度特征图
class DeformableDetrEncoder(DeformableDetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`DeformableDetrEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: DeformableDetrConfig
    """
    # 初始化方法
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)  # 调用父类的初始化方法

        self.dropout = config.dropout  # 从配置中获取 dropout 参数
        self.layers = nn.ModuleList([DeformableDetrEncoderLayer(config) for _ in range(config.encoder_layers)])  # 创建一个由 DeformableDetrEncoderLayer 构成的列表，总共有 config.encoder_layers 个层

        # Initialize weights and apply final processing
        self.post_init()  # 对权重进行初始化，并进行最终处理

    # 静态方法，用于获取每个特征图的参考点，用于解码器中
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
        reference_points_list = []  # 创建一个列表，用于存储每个特征图的参考点
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)  # 计算参考点的 y 坐标
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)  # 计算参考点的 x 坐标
            ref = torch.stack((ref_x, ref_y), -1)  # 将 x 和 y 坐标合并成一个张量
            reference_points_list.append(ref)  # 将参考点添加到列表中
        reference_points = torch.cat(reference_points_list, 1)  # 将每个特征图的参考点链接成一个张量
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # 为参考点应用有效比例
        return reference_points  # 返回参考点

    # 前向传播方法
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
        # ...
    # 添加了position_embeddings, reference_points, spatial_shapes和valid_ratios到前向传递
    # 还返回所有解码层的中间输出和参考点的堆栈
    
    # 参数:
    #     config: DeformableDetrConfig
    # """
    def __init__(self, config: DeformableDetrConfig):
        # 调用父类构造函数，传入config参数
        super().__init__(config)
    
        # 设置dropout参数
        self.dropout = config.dropout
        # 创建解码层，并存储到layers列表中
        self.layers = nn.ModuleList([DeformableDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 设置梯度检查点为False
        self.gradient_checkpointing = False
    
        # 用于迭代边界框细化和两阶段Deformable DETR的hack实现
        self.bbox_embed = None
        self.class_embed = None
    
        # 初始化权重并应用最终处理
        self.post_init()
    
    # 前向传递函数
    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
# 使用自定义的文档字符串装饰器，添加Deformable DETR模型的说明文档
@add_start_docstrings(
    """
    The bare Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    DEFORMABLE_DETR_START_DOCSTRING,
)
# DeformableDetrModel类，继承自DeformableDetrPreTrainedModel类
class DeformableDetrModel(DeformableDetrPreTrainedModel):
    # 初始化方法
    def __init__(self, config: DeformableDetrConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建骨干网络 + 位置编码
        backbone = DeformableDetrConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = DeformableDetrConvModel(backbone, position_embeddings)

        # 创建输入投影层
        if config.num_feature_levels > 1:
            num_backbone_outs = len(backbone.intermediate_channel_sizes)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.intermediate_channel_sizes[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
                in_channels = config.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                ]
            )

        if not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)

        self.encoder = DeformableDetrEncoder(config)
        self.decoder = DeformableDetrDecoder(config)

        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)
            self.pos_trans = nn.Linear(config.d_model * 2, config.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(config.d_model * 2)
        else:
            self.reference_points = nn.Linear(config.d_model, 2)

        # 调用后续初始化方法
        self.post_init()

    # 获取编码器模块
    def get_encoder(self):
        return self.encoder

    # 获取解码器模块
    def get_decoder(self):
        return self.decoder

    # 冻结骨干网络
    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)
    # 解冻骨干网络的参数，使其可训练
    def unfreeze_backbone(self):
        # 遍历骨干网络的参数，并将参数设置为可训练
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    # 获取所有特征图的有效比例
    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        # 获取掩码的维度信息
        _, height, width = mask.shape
        # 计算每行的有效像素数量
        valid_height = torch.sum(mask[:, :, 0], 1)
        # 计算每列的有效像素数量
        valid_width = torch.sum(mask[:, 0, :], 1)
        # 计算高度方向的有效比例
        valid_ratio_heigth = valid_height.float() / height
        # 计算宽度方向的有效比例
        valid_ratio_width = valid_width.float() / width
        # 组合高度和宽度的有效比例成为一个张量
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    # 获取候选框的位置嵌入
    def get_proposal_pos_embed(self, proposals):
        """Get the position embedding of the proposals."""

        # 定义位置嵌入特征的维度
        num_pos_feats = self.config.d_model // 2
        # 定义温度参数
        temperature = 10000
        # 定义缩放系数
        scale = 2 * math.pi

        # 生成位置嵌入所需的维度张量
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # 对候选框进行 sigmoid 操作，得到 0 到 scale 的值
        proposals = proposals.sigmoid() * scale
        # 计算位置嵌入
        pos = proposals[:, :, :, None] / dim_t
        # 对位置嵌入进行处理，将 sin 和 cos 值分别堆叠，并展平成一维
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
        batch_size = enc_output.shape[0]  # 获取批次大小
        proposals = []  # 初始化存储提议的列表
        _cur = 0  # 初始化当前索引为0
        for level, (height, width) in enumerate(spatial_shapes):  # 遍历空间形状
            mask_flatten_ = padding_mask[:, _cur : (_cur + height * width)].view(batch_size, height, width, 1)  # 获取当前级别的填充掩码
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)  # 计算有效高度
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)  # 计算有效宽度

            grid_y, grid_x = meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )  # 生成网格坐标
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # 将网格坐标拼接

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)  # 计算缩放比例
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale  # 缩放网格坐标
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0**level)  # 计算宽度和高度
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)  # 拼接提议坐标和宽高信息
            proposals.append(proposal)  # 将提议添加到列表中
            _cur += height * width  # 更新当前索引
        output_proposals = torch.cat(proposals, 1)  # 拼接所有提议
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)  # 确保提议在有效范围内
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # 对提议进行反向 sigmoid 转换
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))  # 将填充部分设置为正无穷
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))  # 将无效的提议设置为正无穷

        # assign each pixel as an object query
        object_query = enc_output  # 将编码输出作为对象查询
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))  # 将填充部分设置为0
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))  # 将无效的提议部分设置为0
        object_query = self.enc_output_norm(self.enc_output(object_query))  # 对对象查询进行归一化
        return object_query, output_proposals  # 返回对象查询和提议
    @add_start_docstrings_to_model_forward(DEFORMABLE_DETR_INPUTS_DOCSTRING)
    # 将给定的文档字符串添加到模型的前向方法中
    @replace_return_docstrings(output_type=DeformableDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    # 替换返回文档字符串，指定输出类型和配置类
    
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
# 添加文档字符串，描述 Deformable DETR 模型的作用和功能
@add_start_docstrings(
    """
    Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    DEFORMABLE_DETR_START_DOCSTRING,
)
# DeformableDetrForObjectDetection 类继承自 DeformableDetrPreTrainedModel
class DeformableDetrForObjectDetection(DeformableDetrPreTrainedModel):
    # 当使用克隆层时，所有的层 > 0 都将是克隆层，但层 0 是必需的
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*", r"class_embed\.[1-9]\d*"]
    # 由于一些权重在初始化过程中被修改，所以不能在元设备上初始化模型
    _no_split_modules = None

    # 初始化方法，接收一个 DeformableDetrConfig 类型的参数
    def __init__(self, config: DeformableDetrConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 Deformable DETR 模型
        self.model = DeformableDetrModel(config)

        # 在顶部添加检测头
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # 设置偏置项为指定值
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value
        # 初始化权重为常数
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # 如果是两阶段模型，最后一个 class_embed 和 bbox_embed 用于区域提议生成
        num_pred = (config.decoder_layers + 1) if config.two_stage else config.decoder_layers
        if config.with_box_refine:
            # 根据预测的次数克隆 class_embed 和 bbox_embed
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # 用于迭代边界框细化的实现
            self.model.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None
        if config.two_stage:
            # 用于两阶段的实现
            self.model.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 https://github.com/facebookresearch/detr/blob/master/models/detr.py 中借用的未使用的方法
    @torch.jit.unused
    # 设置辅助损失函数，接受分类输出和坐标输出作为参数
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是一个解决方案，使得 torchscript 能够正常工作，因为 torchscript 不支持具有非同构值的字典，
        # 比如一个同时具有张量和列表的字典。
        # 创建一个列表推导式，将分类输出和坐标输出组合成字典，以使得 torchscript 可以正常工作
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # 将文档字符串添加到模型前向传播函数，用于描述函数的输入参数和输出结果
    # 通过该装饰器添加输入参数的文档字符串
    # 替换输出结果的文档字符串为 DeformableDetrObjectDetectionOutput 类型，并指定配置类为 _CONFIG_FOR_DOC
    @add_start_docstrings_to_model_forward(DEFORMABLE_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DeformableDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    # 声明模型的前向传播函数
    def forward(
        self,
        pixel_values: torch.FloatTensor,  # 输入像素值的张量
        pixel_mask: Optional[torch.LongTensor] = None,  # 可选的像素掩码张量，默认为 None
        decoder_attention_mask: Optional[torch.FloatTensor] = None,  # 可选的解码器注意力掩码张量，默认为 None
        encoder_outputs: Optional[torch.FloatTensor] = None,  # 可选的编码器输出张量，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的输入嵌入张量，默认为 None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的解码器输入嵌入张量，默认为 None
        labels: Optional[List[dict]] = None,  # 可选的标签列表，默认为 None
        output_attentions: Optional[bool] = None,  # 可选的输出注意力张量，默认为 None
        output_hidden_states: Optional[bool] = None,  # 可选的输出隐藏状态张量，默认为 None
        return_dict: Optional[bool] = None,  # 可选的返回字典，默认为 None
# 从transformers.models.detr.modeling_detr.dice_loss中复制过来的函数，用于计算DICE损失，类似于面具的广义IOU
def dice_loss(inputs, targets, num_boxes):
    """
    计算DICE损失，类似于面具的广义IOU

    Args:
        inputs: 任意形状的浮点张量，每个示例的预测值
        targets: 与输入相同形状的浮点张量，存储每个元素的二分类标签（0表示负类，1表示正类）
    """
    # 对输入进行sigmoid激活
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    # 计算分子
    numerator = 2 * (inputs * targets).sum(1)
    # 计算分母
    denominator = inputs.sum(-1) + targets.sum(-1)
    # 计算损失
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


# 从transformers.models.detr.modeling_detr.sigmoid_focal_loss中复制过来的函数，用于计算sigmoid focal损失
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    用于密集检测中RetinaNet的损失函数：https://arxiv.org/abs/1708.02002.

    Args:
        inputs: 任意形状的torch.FloatTensor张量，每个示例的预测值
        targets: 与输入相同形状的torch.FloatTensor张量，存储每个元素的二分类标签（0表示负类，1表示正类）
        alpha: （可选）范围在(0,1)中用于平衡正类与负类的加权系数
        gamma: （可选）默认为2，调整因子(1 - p_t)的指数，平衡简单示例与困难例

    Returns:
        Loss张量
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 添加调整因子
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class DeformableDetrLoss(nn.Module):
    """
    该类计算DeformableDetrForObjectDetection的损失。过程分两步进行：1) 我们计算目标框和模型输出之间的匈牙利分配
    2) 我们监督每对匹配的目标地面实况/预测（监督类别和框）

    Args:
        matcher (`DeformableDetrHungarianMatcher`):
            能够计算目标和提议之间匹配的模块
        num_classes (`int`):
            对象类别的数量，不包括特殊的无对象类别
        focal_alpha (`float`):
            Focal损失的Alpha参数
        losses (`List[str]`):
            要应用的所有损失的列表。参见`get_loss`以获取所有可用损失的列表
    """
    # 初始化函数，接受匹配器、类别数量、焦距系数和损失函数作为参数
    def __init__(self, matcher, num_classes, focal_alpha, losses):
        super().__init__()
        # 设置匹配器
        self.matcher = matcher
        # 设置类别数量
        self.num_classes = num_classes
        # 设置焦距系数
        self.focal_alpha = focal_alpha
        # 设置损失函数
        self.losses = losses

    # 分类损失函数，二元焦距损失，目标字典必须包含键为"class_labels"的张量，维度为[nb_target_boxes]
    def loss_labels(self, outputs, targets, indices, num_boxes):
        # 如果输出中没有"logits"，抛出关键错误
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        # 获取源 logits
        source_logits = outputs["logits"]

        # 获取源置换索引
        idx = self._get_source_permutation_idx(indices)
        # 从目标中合并目标类别
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        # 创建全为类别数量的目标类别张量
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        # 创建独热编码的目标类别张量
        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # 去掉最后一个类别
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        # 计算交叉熵损失
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    # 无梯度计算函数，用于计算绝对卡迪尔数误差
    @torch.no_grad()
    # 从 transformers.models.detr.modeling_detr.DetrLoss.loss_cardinality 复制而来
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        # 获取 logits
        logits = outputs["logits"]
        # 获取设备信息
        device = logits.device
        # 获取目标长度
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算非空盒子的预测数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 计算绝对卡迪尔数误差
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    # 从 transformers.models.detr.modeling_detr.DetrLoss.loss_boxes 中复制的
    # 定义函数 loss_boxes，用于计算与边界框相关的损失，包括 L1 回归损失和 GIoU 损失。
    # Targets 字典必须包含键 "boxes"，其中包含维度为 [nb_target_boxes, 4] 的张量。
    # 目标框的格式应为 (center_x, center_y, w, h)，按图像大小进行标准化。
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        # 如果输出中没有预测框，则引发 KeyError
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        # 获取源排列索引
        idx = self._get_source_permutation_idx(indices)
        # 从目标中获取源框和目标框
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算边界框损失，使用 L1 损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        # 将边界框损失添加到损失字典中
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算 GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        # 将 GIoU 损失添加到损失字典中
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        # 返回损失字典
        return losses

    # 从 transformers.models.detr.modeling_detr.DetrLoss._get_source_permutation_idx 复制而来
    def _get_source_permutation_idx(self, indices):
        # 根据索引对预测进行排列
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # 从 transformers.models.detr.modeling_detr.DetrLoss._get_target_permutation_idx 复制而来
    def _get_target_permutation_idx(self, indices):
        # 根据索引对目标进行排列
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    # 获取损失的函数
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        # 定义损失映射
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        # 如果损失不在损失映射中，则引发 ValueError
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        # 返回相应损失的计算结果
        return loss_map[loss](outputs, targets, indices, num_boxes)
    # 定义一个名为forward的方法，用于执行损失计算
    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional`):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional`):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        # 创建一个不包含"auxiliary_outputs"和"enc_outputs"的新字典
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs" and k != "enc_outputs"}

        # 使用matcher函数获取最后一层输出和targets之间的匹配
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点上目标框的平均数量，用于归一化
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # 分布式训练功能暂时注释掉，分布式训练将会被添加
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 如果有辅助损失，需要使用每个中间层的输出重复上面的计算过程
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 如果有enc_outputs，需要使用其执行类似的损失计算
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
# 定义一个名为 DeformableDetrMLPPredictionHead 的类，继承自 nn.Module
class DeformableDetrMLPPredictionHead(nn.Module):
    """
    非常简单的多层感知器（MLP，也称为 FFN），用于预测边界框相对于图像的归一化中心坐标、高度和宽度。

    从https://github.com/facebookresearch/detr/blob/master/models/detr.py复制而来
    """

    # 初始化方法，接收输入维度、隐藏层维度、输出维度和层数
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        # 调用父类的初始化方法
        super().__init__()
        # 记录层数
        self.num_layers = num_layers
        # 定义隐藏层维度列表
        h = [hidden_dim] * (num_layers - 1)
        # 创建多个线性层，并放入 nn.ModuleList 中
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    # 前向传播方法，接收输入张量 x
    def forward(self, x):
        # 遍历每一层的线性层
        for i, layer in enumerate(self.layers):
            # 对输入张量进行线性变换和激活函数处理
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # 返回处理后的张量
        return x


# 定义一个名为 DeformableDetrHungarianMatcher 的类，继承自 nn.Module
class DeformableDetrHungarianMatcher(nn.Module):
    """
    该类计算网络的预测和目标之间的匹配。

    由于效率原因，目标不包括无对象。因此，一般来说，预测数量会多于目标数量。
    在这种情况下，我们对最佳预测进行一对一匹配，而其他预测无法匹配（因此被视为非对象）。

    Args:
        class_cost:
            匹配成本中分类错误的相对权重。
        bbox_cost:
            匹配成本中边界框坐标的 L1 误差的相对权重。
        giou_cost:
            匹配成本中边界框的 giou 损失的相对权重。
    """

    # 初始化方法，接收分类成本、边界框成本、giou成本的相对权重
    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        # 调用父类的初始化方法
        super().__init__()
        # 判断是否需要后端支持 scipy
        requires_backends(self, ["scipy"])

        # 记录分类成本、边界框成本、giou成本的相对权重
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # 如果所有成本都为0，则抛出异常
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    # 设置为 torch 无梯度操作
    @torch.no_grad()
    # 定义一个函数，接收两个参数并返回一个列表
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                包含以下参数的字典:
                * "logits": 维度为[batch_size, num_queries, num_classes]的张量，包含分类logits
                * "pred_boxes": 维度为[batch_size, num_queries, 4]的张量，包含预测的框坐标。
            targets (`List[dict]`):
                一个目标列表（len(targets) = batch_size），每个目标是一个包含以下内容的字典:
                * "class_labels": 维度为[num_target_boxes]（num_target_boxes是目标中真实物体的数量）的张量，包含类别标签
                * "boxes": 维度为[num_target_boxes, 4]的张量，包含目标框坐标。

        Returns:
            `List[Tuple]`: 一个大小为`batch_size`的列表，包含(index_i, index_j)元组，其中:
            - index_i是选择的预测的索引（按顺序）
            - index_j是对应选择的目标的索引（按顺序）
            对于每个batch元素，它保持: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # 将输出logits扁平化以计算成批的成本矩阵
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 同样连接目标标签和框
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算分类成本
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # 计算框之间的L1成本
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # 计算框之间的giou成本
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # 最终成本矩阵
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# Copied from transformers.models.detr.modeling_detr._upcast
# 将输入的张量t上溢转换成更高类型的张量
def _upcast(t: Tensor) -> Tensor:
    # 如果t是浮点型
    if t.is_floating_point():
        # 如果t的数据类型是32位或64位浮点型，返回t
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        # 如果t的数据类型是32位或64位整型，返回t
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
# 计算一组边界框的面积，这些边界框由其（x1，y1，x2，y2）坐标指定
def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    # 将输入的边界框转换为_upcast函数返回的类型
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
# 计算两组边界框的IoU（交并比）
def box_iou(boxes1, boxes2):
    # 计算第一组边界框的面积
    area1 = box_area(boxes1)
    # 计算第二组边界框的面积
    area2 = box_area(boxes2)

    # 计算两组边界框的左上角和右下角的交集
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # 计算交集的宽度和高度
    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    # 计算交集的面积
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    # 计算并集的面积
    union = area1[:, None] + area2 - inter

    # 计算IoU
    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # 检查边界框是否退化为点或线，这将导致inf / nan结果
    # 检查第一组边界框是否满足格式要求
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    # 检查第二组边界框是否满足格式要求
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    # 计算box_iou函数的返回值
    iou, union = box_iou(boxes1, boxes2)

    # 计算两组边界框的左上角和右下角的交集
    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 计算交集的宽度和高度
    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    # 返回generalized IoU
    return iou - (area - union) / area


# Copied from transformers.models.detr.modeling_detr._max_by_axis
# 返回列表中每个子列表对应位置元素的最大值
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    # 将列表第一个子列表作为初始的最大值
    maxes = the_list[0]
    # 遍历列表中的每一个子列表
    for sublist in the_list[1:]:
        # 遍历子列表中的元素
        for index, item in enumerate(sublist):
            # 更新最大值
            maxes[index] = max(maxes[index], item)
    return maxes
# 定义嵌套张量类
class NestedTensor(object):
    # 初始化函数，接收张量列表和可选的遮罩
    def __init__(self, tensors, mask: Optional[Tensor]):
        # 将张量和遮罩赋给对象的属性
        self.tensors = tensors
        self.mask = mask

    # 将嵌套张量移动到指定的设备
    def to(self, device):
        # 将张量移动到指定设备
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            # 如果存在遮罩，则也将遮罩移动到指定设备
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    # 解构嵌套张量，返回张量列表和遮罩
    def decompose(self):
        return self.tensors, self.mask

    # 重写打印函数，输出张量的字符串表示
    def __repr__(self):
        return str(self.tensors)


# 从张量列表创建嵌套张量的函数，复制自transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # 检查第一个张量的维度是否为3
    if tensor_list[0].ndim == 3:
        # 计算最大尺寸
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # 计算批次形状
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # 创建全0张量，指定数据类型和设备
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 创建全1遮罩张量，指定数据类型和设备
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # 将张量复制到全0张量的对应位置
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            # 更新遮罩
            m[: img.shape[1], : img.shape[2]] = False
    else:
        # 抛出异常，只支持3维张量
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)
```