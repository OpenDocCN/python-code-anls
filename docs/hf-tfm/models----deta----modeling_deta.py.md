# `.\models\deta\modeling_deta.py`

```
# 指定编码方式为 utf-8

# 版权声明
# SenseTime 和 The HuggingFace Inc. 团队版权所有。
# 根据 Apache 许可证 2.0 版（"许可证"）许可；
# 除非符合许可证的条款，否则您不能使用此文件。
# 您可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本。
#
# 除非适用法律要求或书面同意，否则依据许可证分发的软件都是基于"AS IS"基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证，了解特定语言下的权限和限制。
""" PyTorch DETA model."""

# 导入模块和库
import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# 导入 HuggingFace 库中的各种实用工具
from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_vision_available,
    replace_return_docstrings,
)
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_torchvision_available, logging, requires_backends

# 判断是否使用 accelerate 库，如果可用，则导入相关内容
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

# 判断是否使用 vision 库，如果可用，则导入相关内容
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

# 判断是否使用 torchvision 库，如果可用，则导入相关内容
if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms

# 判断是否使用 scipy 库，如果可用，则导入相关内容
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DetaConfig"
_CHECKPOINT_FOR_DOC = "jozhang97/deta-swin-large-o365"

# DETA 预训练模型列表
DETA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "jozhang97/deta-swin-large-o365",
    # 查看所有 DETA 模型可在 https://huggingface.co/models?filter=deta
]

@dataclass
# 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrDecoderOutput 复制并将 DeformableDetr 更名为 Deta
class DetaDecoderOutput(ModelOutput):
    """
    Base class for outputs of the DetaDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None  # 初始化变量last_hidden_state为None
    intermediate_hidden_states: torch.FloatTensor = None  # 初始化变量intermediate_hidden_states为None
    intermediate_reference_points: torch.FloatTensor = None  # 初始化变量intermediate_reference_points为None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 初始化变量hidden_states为None，可选的元组类型，元素为torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 初始化变量attentions为None，可选的元组类型，元素为torch.FloatTensor
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 初始化变量cross_attentions为None，可选的元组类型，元素为torch.FloatTensor
from dataclasses import dataclass
# 导入 dataclass 模块

class DetaModelOutput(ModelOutput):
    """
    Base class for outputs of the Deformable DETR encoder-decoder model.
    """
    # 定义 DetaModelOutput 类，用于 Deformable DETR 编码器-解码器模型的输出

    init_reference_points: torch.FloatTensor = None
    # 初始参考点张量
    last_hidden_state: torch.FloatTensor = None
    # 最后隐藏状态张量
    intermediate_hidden_states: torch.FloatTensor = None
    # 中间隐藏状态张量
    intermediate_reference_points: torch.FloatTensor = None
    # 中间参考点张量
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器隐藏状态
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器最后隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器隐藏状态
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力
    enc_outputs_class: Optional[torch.FloatTensor] = None
    # 输出类别
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None
    # 输出坐标对数
    output_proposals: Optional[torch.FloatTensor] = None
    # 输出提议框

class DetaObjectDetectionOutput(ModelOutput):
    """
    Output type of [`DetaForObjectDetection`].
    """
    # DetaObjectDetectionOutput 类的输出类型

    loss: Optional[torch.FloatTensor] = None
    # 损失
    loss_dict: Optional[Dict] = None
    # 损失字典
    logits: torch.FloatTensor = None
    # 逻辑张量
    pred_boxes: torch.FloatTensor = None
    # 预测框
    auxiliary_outputs: Optional[List[Dict]] = None
    # 辅助输出
    init_reference_points: Optional[torch.FloatTensor] = None
    # 初始参考点张量
    last_hidden_state: Optional[torch.FloatTensor] = None
    # 最后隐藏状态张量
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    # 中间隐藏状态张量
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    # 中间参考点张量
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器隐藏状态
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器最后隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器隐藏状态
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力
    enc_outputs_class: Optional = None
    # 输出类别
    enc_outputs_coord_logits: Optional = None
    # 输出坐标对数
    output_proposals: Optional[torch.FloatTensor] = None
    # 输出提议框

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
# 克隆函数，用于克隆神经网络模块

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    # 将输入张量限制在 [0, 1] 之间
    x1 = x.clamp(min=eps)
    # 将输入张量限制在 [eps, 1] 之间
    x2 = (1 - x).clamp(min=eps)
    # 将 (1-x) 张量限制在 [eps, 1] 之间
    return torch.log(x1 / x2)
    # 返回对数减法

class DetaFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 构造 num_batches_tracked 的键名
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # 如果 num_batches_tracked 存在于状态字典中，则删除它
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类的_load_from_state_dict方法，加载模型参数
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # 将权重、偏置、方差和均值进行维度调整以便之后的计算
        # 使其更易于使用
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        # 返回经过缩放和偏移的输入 x
        return x * scale + bias
# 从transformers.models.detr.modeling_detr.replace_batch_norm复制并替换为Detr->Deta
def replace_batch_norm(model):
    r"""
    递归替换所有`torch.nn.BatchNorm2d`为`DetaFrozenBatchNorm2d`。

    Args:
        model (torch.nn.Module):
            输入模型
    """
    for name, module in model.named_children():
        # 如果是`nn.BatchNorm2d`类型的模块
        if isinstance(module, nn.BatchNorm2d):
            # 创建一个新的`DetaFrozenBatchNorm2d`模块，并复制参数
            new_module = DetaFrozenBatchNorm2d(module.num_features)

            # 如果模块的权重不在"meta"设备上
            if not module.weight.device == torch.device("meta"):
                # 复制权重和偏置
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            # 用新模块替换原模块
            model._modules[name] = new_module

        # 如果模块有子模块，继续替换
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class DetaBackboneWithPositionalEncodings(nn.Module):
    """
    具有位置嵌入的主干模型。

    `nn.BatchNorm2d`层将被上面定义的`DetaFrozenBatchNorm2d`替换。
    """

    def __init__(self, config):
        super().__init__()

        # 从配置文件中创建主干模型
        backbone = AutoBackbone.from_config(config.backbone_config)
        # 用`replace_batch_norm`替换所有的`nn.BatchNorm2d`
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels

        # TODO 修复这部分
        # 如果主干模型类型是"resnet"
        if config.backbone_config.model_type == "resnet":
            # 对于不在"stages.1"、"stages.2"和"stages.3"中的参数，设置requires_grad为False
            for name, parameter in self.model.named_parameters():
                if "stages.1" not in name and "stages.2" not in name and "stages.3" not in name:
                    parameter.requires_grad_(False)

        # 创建位置嵌入
        self.position_embedding = build_position_encoding(config)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        """
        如果`config.num_feature_levels > 1`，则输出ResNet中后续阶段C_3到C_5的特征映射，否则输出C_5的特征映射。
        """
        # 首先，将像素值通过主干模型得到特征映射列表
        features = self.model(pixel_values).feature_maps

        # 然后，创建位置嵌入
        out = []
        pos = []
        for feature_map in features:
            # 将pixel_mask下采样以匹配相应feature_map的形状
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            position_embeddings = self.position_embedding(feature_map, mask).to(feature_map.dtype)
            out.append((feature_map, mask))
            pos.append(position_embeddings)

        return out, pos


# 从transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrSinePositionEmbedding复制并替换为DeformableDetr->Deta
class DetaSinePositionEmbedding(nn.Module):
    """
    这是位置嵌入的更标准版本，非常类似于Attention is all you
    # 需要纸张，泛化到适用于图像。
    """
    
    # 初始化函数，设置嵌入维度、温度、是否归一化以及比例
    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度
        self.embedding_dim = embedding_dim
        # 设置温度
        self.temperature = temperature
        # 设置是否归一化
        self.normalize = normalize
        # 如果指定了比例但未指定是否归一化，则抛出数值错误
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 如果未指定比例，则默认采用2π
        if scale is None:
            scale = 2 * math.pi
        # 设置比例
        self.scale = scale

    # 前向传播函数
    def forward(self, pixel_values, pixel_mask):
        # 如果未提供像素掩码，则抛出数值错误
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        # 计算像素的y方向嵌入
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        # 计算像素的x方向嵌入
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        # 如果需要归一化
        if self.normalize:
            # 微小值
            eps = 1e-6
            # 归一化y方向嵌入
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            # 归一化x方向嵌入
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        # 创建维度张量
        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        # 计算x方向位置编码
        pos_x = x_embed[:, :, :, None] / dim_t
        # 计算y方向位置编码
        pos_y = y_embed[:, :, :, None] / dim_t
        # 计算x方向位置编码的sin和cos并拼接
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 计算y方向位置编码的sin和cos并拼接
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 拼接x和y方向位置编码并转置
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # 返回位置编码
        return pos
# 从transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding复制过来的代码
class DetaLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)  # 创建一个大小为50 * embedding_dim的行位置embedding
        self.column_embeddings = nn.Embedding(50, embedding_dim)  # 创建一个大小为50 * embedding_dim的列位置embedding

    def forward(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)  # 创建一个宽度值的Tensor
        height_values = torch.arange(height, device=pixel_values.device)  # 创建一个高度值的Tensor
        x_emb = self.column_embeddings(width_values)  # 获取列位置的embedding
        y_emb = self.row_embeddings(height_values)  # 获取行位置的embedding
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)  # 拼接列和行位置embedding
        pos = pos.permute(2, 0, 1)  # 调整位置
        pos = pos.unsqueeze(0)  # 在0维度增加一个维度
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)  # 在指定维度上复制元素
        return pos  # 返回位置信息


# 从transformers.models.detr.modeling_detr.build_position_encoding复制过来的代码，将Detr替换为Deta
def build_position_encoding(config):
    n_steps = config.d_model // 2  # 计算步长
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = DetaSinePositionEmbedding(n_steps, normalize=True)  # 使用DetaSinePositionEmbedding创建位置编码
    elif config.position_embedding_type == "learned":
        position_embedding = DetaLearnedPositionEmbedding(n_steps)  # 使用DetaLearnedPositionEmbedding创建位置编码
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")  # 抛出数值错误异常

    return position_embedding  # 返回位置编码模块


# 从transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention复制过来的代码
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape  # 获取value的形状信息
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape  # 获取sampling_locations的形状信息
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)  # 将value拆分成多个Tensor
    sampling_grids = 2 * sampling_locations - 1  # 对采样位置进行变换
    sampling_value_list = []  # 初始化采样值列表
    # 对于每个level，获取其id和对应的空间形状(height, width)，并使用enumerate函数遍历
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # 将value_list[level_id]展平成(batch_size, height*width, num_heads, hidden_dim)的形状，然后进行维度转换和形状调整
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # 将sampling_grids在指定维度上的数据进行转置和展平
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # 使用grid_sample函数对value_l_进行采样，得到sampling_value_l_
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        # 将sampling_value_l_添加到sampling_value_list中
        sampling_value_list.append(sampling_value_l_)
    # 将attention_weights在指定维度上进行转置和形状调整
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    # 将stack后的sampling_value_list在指定维度上压平，再和attention_weights相乘求和，最后调整形状
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    # 对output进行维度转换和连续内存重新布局
    return output.transpose(1, 2).contiguous()
class DetaMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        # 初始化方法，设置模型参数
        super().__init__()
        # 判断 embed_dim 是否能整除 num_heads，若不能则抛出错误
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        # 计算每个头的维度
        dim_per_head = embed_dim // num_heads
        # 判断 dim_per_head 是否是 2 的幂，若不是则发出警告
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DetaMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        # 设置 im2col_step 为 64
        self.im2col_step = 64

        # 设置模型参数
        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        # 创建偏移量线性层
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        # 创建 attention 权重线性层
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        # 创建 value 投影线性层
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # 创建输出投影线性层
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # 初始化权重
        self._reset_parameters()

    def _reset_parameters(self):
        # 设置偏移量权重为常数
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        # 初始化角度值
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # 创建初始网格
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        # 调整初始网格
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # 设置偏移量偏置
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # 初始化注意力权重的权重和偏置
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        # 初始化 value 投影的权重和偏置
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        # 初始化输出投影的权重和偏置
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        # 如果位置嵌入不为空，则添加到张量中
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
        # 检查函数结束的括号
        ):
        # 在将隐藏状态投影到查询和键之前，向隐藏状态添加位置嵌入
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # 获取隐藏状态的批大小、查询数量和维度
        batch_size, num_queries, _ = hidden_states.shape
        # 获取编码器隐藏状态的批大小、序列长度和维度
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        # 如果空间形状乘积之和不等于序列长度，则引发 ValueError
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        # 使用值投影函数对编码器隐藏状态进行投影
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # 反转注意力掩码
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        # 获取隐藏状态的采样偏移
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        # 获取注意力权重
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # 如果参考点的最后一个维度为2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        # 如果参考点的最后一个维度为4
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        # 如果不符合前述条件，则引发 ValueError
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
        # 使用多尺度可变形注意力函数处理输出
        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        # 对输出进行投影
        output = self.output_proj(output)

        # 返回输出和注意力权重
        return output, attention_weights
# 从transformers.models.deformable_detr.modeling_deformable_detr中复制DetaMultiheadAttention类，将DeformableDetr->Deta，Deformable DETR->DETA
class DetaMultiheadAttention(nn.Module):
    """
    从'Attention Is All You Need'论文中的多头注意力机制中复制过来。

    在这里，我们根据Deformable DETR论文的说明，将位置嵌入添加到查询和键中。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 头的数量
        self.dropout = dropout  # 丢弃概率
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放系数

        # 线性变换得到查询/键/值/输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 将张量的形状调整为(batch_size, seq_len, num_heads, head_dim)，并对维度进行转置
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 将位置嵌入加到张量中
    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        position_embeddings: Optional[torch.Tensor] = None,  # 位置嵌入
        output_attentions: bool = False,  # 是否输出注意力
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                输入层的输入。
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                注意力掩码。
            position_embeddings (`torch.FloatTensor`, *optional*):
                位置嵌入，将要添加到 `hidden_states` 中。
            reference_points (`torch.FloatTensor`, *optional*):
                参考点。
            spatial_shapes (`torch.LongTensor`, *optional*):
                主干特征图的空间形状。
            level_start_index (`torch.LongTensor`, *optional*):
                级别起始索引。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量下的 `attentions`。
        """
        residual = hidden_states

        # 在多尺度特征图上应用多尺度可变形注意力模块。
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
class DetaDecoderLayer(nn.Module):
    def __init__(self, config: DetaConfig):
        super().__init__()
        self.embed_dim = config.d_model
        
        # self-attention
        # 创建一个self-attention层，传入参数包括embed_dim、num_heads和dropout
        self.self_attn = DetaMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        
        # 对self-attention的输出进行Layer Normalization
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # cross-attention
        # 创建一个cross-attention层，传入参数包括embed_dim、num_heads、n_levels和n_points
        self.encoder_attn = DetaMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            n_levels=config.num_feature_levels,
            n_points=config.decoder_n_points,
        )
        # 对cross-attention的输出进行Layer Normalization
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # feedforward neural networks
        # 创建两个全连接层，分别为fc1和fc2
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 对最终输出进行Layer Normalization
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



# Copied from transformers.models.detr.modeling_detr.DetrClassificationHead
class DetaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        # 创建一个全连接层，使用input_dim和inner_dim的线性变换
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 创建一个全连接层，使用inner_dim和num_classes的线性变换
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        # 对输入的hidden_states进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 使��全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对 hidden_states 进行 tanh 激活函数
        hidden_states = torch.tanh(hidden_states)
        # 对 hidden_states 进行 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 使用全连接层进行线性变换
        hidden_states = self.out_proj(hidden_states)
        return hidden_states



# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrPreTrainedModel with DeformableDetrConvEncoder->DetaBackboneWithPositionalEncodings,DeformableDetr->Deta
class DetaPreTrainedModel(PreTrainedModel):
    config_class = DetaConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DetaBackboneWithPositionalEncodings", r"DetaEncoderLayer", r"DetaDecoderLayer"]


注释：
    # 初始化权重参数
    def _init_weights(self, module):
        std = self.config.init_std

        # 如果 module 是 DetaLearnedPositionEmbedding 类型
        if isinstance(module, DetaLearnedPositionEmbedding):
            # 初始化位置编码的行和列的权重参数
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        # 如果 module 是 DetaMultiscaleDeformableAttention 类型
        elif isinstance(module, DetaMultiscaleDeformableAttention):
            # 重置参数
            module._reset_parameters()
        # 如果 module 是 nn.Linear, nn.Conv2d, nn.BatchNorm2d 类型
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 使用正态分布初始化权重参数
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重参数
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 具有 "reference_points" 属性并且不是 two_stage 模式
        if hasattr(module, "reference_points") and not self.config.two_stage:
            # 使用 xavier_uniform 初始化 reference_points 的权重参数和常数项
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        # 如果 module 具有 "level_embed" 属性
        if hasattr(module, "level_embed"):
            # 使用正态分布初始化 level_embed 的参数
            nn.init.normal_(module.level_embed)
# 定义文档字符串，描述了该模型的继承关系和PyTorch Module的用法
DETA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DetaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义输入文档字符串，描述模型的输入参数
DETA_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下将忽略填充。
            # 可以使用[`AutoImageProcessor`]的方法获取像素值。有关详细信息，请参见[`AutoImageProcessor.__call__`]。

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            # 用于避免在填充像素值上执行注意力的掩码。
            # 掩码值在`[0, 1]`范围内：
            # - 1 表示真实像素（即**未掩码**），
            # - 0 表示填充像素（即**已掩码**）。
            # [什么是注意力掩码？](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            # 默认情况下不使用。可用于遮蔽对象查询。
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            # 元组包含(`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            # `last_hidden_state`的形状为`(batch_size, sequence_length, hidden_size)`，*optional*）是编码器最后一层的输出的隐藏状态序列。在解码器的交叉注意力中使用。
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选择地，可以直接传递图像的平坦表示，而不是传递平坦的特征图（骨干网输出+投影层的输出）。
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            # 可选择地，可以选择直接传递嵌入表示，而不是使用零张量初始化查询。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关详细信息，请参见返回的张量下的`attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关详细信息，请参见返回的张量下的`hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回[`~file_utils.ModelOutput`]而不是普通的元组。
# 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrEncoder 复制出 DetaEncoder 类，修改 DeformableDetr 为 Deta
class DetaEncoder(DetaPreTrainedModel):
    """
    由 config.encoder_layers 个可变形注意力层组成的 Transformer 编码器。
    编码器通过多个可变形注意力层更新多尺度特征图。

    参数：
        config: DetaConfig
    """

    def __init__(self, config: DetaConfig):
        super().__init__(config)

        # 随机失活概率
        self.dropout = config.dropout
        # 通过多个 DetaEncoderLayer 实例组成的列表
        self.layers = nn.ModuleList([DetaEncoderLayer(config) for _ in range(config.encoder_layers)])

        # 初始化权重并应用最终处理
        self.post_init()

    @staticmethod
    # 获取每个特征图的参考点。在解码器中使用。
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        获取每个特征图的参考点。在解码器中使用。

        参数：
            spatial_shapes（`torch.LongTensor`，形状为 `(num_feature_levels, 2)`）：
                每个特征图的空间形状。
            valid_ratios（`torch.FloatTensor`，形状为 `(batch_size, num_feature_levels, 2)`）：
                每个特征图的有效比率。
            device（`torch.device`）：
                在其上创建张量的设备。
        返回：
            `torch.FloatTensor`，形状为 `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # TODO: 这里的 valid_ratios 可能是多余的。请查看 https://github.com/fundamentalvision/Deformable-DETR/issues/36
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
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
# 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrDecoder 复制出 DetaDecoder 类，修改 DeformableDetr 为 Deta，Deformable DETR 修改为 DETA
class DetaDecoder(DetaPreTrainedModel):
    """
    由 *config.decoder_layers* 层组成的 Transformer 解码器。每一层都是 [`DetaDecoderLayer`]。
    """
    # 这部分代码是关于解码器的说明，描述了解码器通过多个自注意力和交叉注意力层更新查询嵌入的过程。

    # 对于Deformable DETR的一些调整：

    # - 在前向传递中添加了`position_embeddings`、`reference_points`、`spatial_shapes`和`valid_ratios`。
    # - 它还返回了所有解码层的中间输出和参考点的堆栈。

    # 参数：
    #     config: DetaConfig
    """
    
    # 初始化解码器对象
    def __init__(self, config: DetaConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置解码器的丢弃率
        self.dropout = config.dropout
        # 创建解码器层的模块列表，根据config的decoder_layers创建多个DetaDecoderLayer对象
        self.layers = nn.ModuleList([DetaDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 设置梯度检查点标志为False
        self.gradient_checkpointing = False

        # 用于迭代边界框细化和两阶段Deformable DETR的hack实现
        self.bbox_embed = None
        self.class_embed = None

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
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
# 定义 DetaModel 类，包含一个骨干和编码器-解码器 Transformer，输出原始隐藏状态而不带特定头部
@add_start_docstrings(
    """
    The bare DETA Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
    any specific head on top.
    """,
    DETA_START_DOCSTRING,
)
class DetaModel(DetaPreTrainedModel):
    # 初始化方法，接受一个 DetaConfig 对象作为参数
    def __init__(self, config: DetaConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中包含两阶段，则需要引入 torchvisions
        if config.two_stage:
            requires_backends(self, ["torchvision"])

        # 创建带有位置编码的骨干
        self.backbone = DetaBackboneWithPositionalEncodings(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        # 创建输入投影层
        if config.num_feature_levels > 1:
            num_backbone_outs = len(intermediate_channel_sizes)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = intermediate_channel_sizes[_]
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
                        nn.Conv2d(intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                ]
            )

        # 如果不是两阶段模型，则创建查询位置嵌入
        if not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)

        # 创建编码器和解码器
        self.encoder = DetaEncoder(config)
        self.decoder = DetaDecoder(config)

        # 创建级别嵌入
        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        # 如果是两阶段模型，则创建额外的层和归一化层
        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)
            self.pos_trans = nn.Linear(config.d_model * 2, config.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(config.d_model * 2)
            self.pix_trans = nn.Linear(config.d_model, config.d_model)
            self.pix_trans_norm = nn.LayerNorm(config.d_model)
        else:
            self.reference_points = nn.Linear(config.d_model, 2)

        # 设置一些额外的配置参数
        self.assign_first_stage = config.assign_first_stage
        self.two_stage_num_proposals = config.two_stage_num_proposals

        # 调用初始化后处理方法
        self.post_init()
    # 从DeformableDetrModel类中获取编码器对象
    def get_encoder(self):
        return self.encoder

    # 从DeformableDetrModel类中获取解码器对象
    def get_decoder(self):
        return self.decoder

    # 冻结骨干网络的参数，使其不可训练
    def freeze_backbone(self):
        for name, param in self.backbone.model.named_parameters():
            param.requires_grad_(False)

    # 解冻骨干网络的参数，使其可训练
    def unfreeze_backbone(self):
        for name, param in self.backbone.model.named_parameters():
            param.requires_grad_(True)

    # 计算所有特征图的有效比例
    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    # 获取提议的位置嵌入
    def get_proposal_pos_embed(self, proposals):
        """Get the position embedding of the proposals."""

        num_pos_feats = self.config.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # batch_size, num_queries, 4
        proposals = proposals.sigmoid() * scale
        # batch_size, num_queries, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # batch_size, num_queries, 4, 64, 2 -> batch_size, num_queries, 512
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    # DeformableDetrModel类的前向传播函数
    @add_start_docstrings_to_model_forward(DETA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetaModelOutput, config_class=_CONFIG_FOR_DOC)
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
# 添加模型文档字符串，描述该模型是一个包含骨干和编码器-解码器Transformer的DETA模型，用于目标检测任务，如COCO检测
@add_start_docstrings(
    """
    DETA Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
    such as COCO detection.
    """,
    DETA_START_DOCSTRING,
)
# 定义一个继承自DetaPreTrainedModel的DETA目标检测模型类
class DetaForObjectDetection(DetaPreTrainedModel):
    # 当使用克隆时，所有层 > 0 将被克隆，但层 0 *是* 必需的
    _tied_weights_keys = [r"bbox_embed\.\d+"]
    # 无法在元设备上初始化模型，因为在初始化过程中会修改一些权重
    _no_split_modules = None

    # 从transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrForObjectDetection.__init__中复制而来，将DeformableDetr->Deta
    def __init__(self, config: DetaConfig):
        super().__init__(config)

        # 创建DETA模型，包括编码器-解码器模型
        self.model = DetaModel(config)

        # 在顶部添加检测头
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = DetaMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # 如果是两阶段，最后一个class_embed和bbox_embed用于区域提议生成
        num_pred = (config.decoder_layers + 1) if config.two_stage else config.decoder_layers
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # 用于迭代边界框细化的hack实现
            self.model.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None
        if config.two_stage:
            # 用于两阶段的hack实现
            self.model.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # 初始化权重并应用最终处理
        self.post_init()

    @torch.jit.unused
    # 设置辅助损失函数，接收分类和坐标输出
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是一个解决方案，使 torchscript 能够正常工作，因为 torchscript
        # 不支持具有非同质值的字典，比如一个既有张量又有列表的字典。
        # 通过 outputs_class 和 outputs_coord 的转置，创建包含 logits 和 pred_boxes 的字典列表
        aux_loss = [
            {"logits": logits, "pred_boxes": pred_boxes}
            for logits, pred_boxes in zip(outputs_class.transpose(0, 1)[:-1], outputs_coord.transpose(0, 1)[:-1])
        ]
        # 返回辅助损失列表
        return aux_loss

    # 前向传播函数，接收多个输入参数
    @add_start_docstrings_to_model_forward(DETA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetaObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
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
# 从transformers.models.detr.modeling_detr.dice_loss中复制的函数，计算DICE损失，类似于掩模的广义IOU
def dice_loss(inputs, targets, num_boxes):
    """
    计算DICE损失，类似于掩模的广义IOU

    Args:
        inputs: 任意形状的浮点张量。
                每个示例的预测。
        targets: 与inputs相同形状的浮点张量。存储每个元素的二进制分类标签（0表示负类，1表示正类）。
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


# 从transformers.models.detr.modeling_detr.sigmoid_focal_loss中复制的函数，用于RetinaNet中的密集检测
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    在RetinaNet中用于密集检测的损失函数：https://arxiv.org/abs/1708.02002。

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            每个示例的预测。
        targets (`torch.FloatTensor`与`inputs`相同形状):
            存储每个`inputs`元素的二进制分类标签（0表示负类，1表示正类）。
        alpha (`float`, *optional*, 默认为`0.25`):
            用于平衡正负样本的可选加权因子范围（0,1）。
        gamma (`int`, *optional*, 默认为`2`):
            调节因子（1 - p_t）的指数，用于平衡易于难以的示例。

    Returns:
        损失张量
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 添加调节因子
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class DetaLoss(nn.Module):
    """
    此类计算`DetaForObjectDetection`的损失。过程分为两步：1）计算目标框和模型输出之间的匈牙利分配 2）监督每对匹配的地面真实框/预测（监督类别和框）。

    Args:
        matcher (`DetaHungarianMatcher`):
            能够计算目标和提议之间匹配的模块。
        num_classes (`int`):
            对象类别的数量，不包括特殊的无对象类别。
        focal_alpha (`float`):
            Focal损失中的Alpha参数。
        losses (`List[str]`):
            要应用的所有损失的列表。查看`get_loss`以获取所有可用损失的列表。
    """
    # 初始化函数，设置模型参数和标志位
    def __init__(
        self,
        matcher,
        num_classes,
        focal_alpha,
        losses,
        num_queries,
        assign_first_stage=False,
        assign_second_stage=False,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型参数
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses
        self.assign_first_stage = assign_first_stage
        self.assign_second_stage = assign_second_stage

        # 如果需要为第一阶段分配任务，则创建第一阶段分配器
        if self.assign_first_stage:
            self.stg1_assigner = DetaStage1Assigner()
        # 如果需要为第二阶段分配任务，则创建第二阶段分配器
        if self.assign_second_stage:
            self.stg2_assigner = DetaStage2Assigner(num_queries)

    # 定义分类损失函数，计算分类损失
    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_labels
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        # 检查输出中是否包含logits
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        # 获取源排列的索引
        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    # 无梯度计算
    @torch.no_grad()
    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_cardinality
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        # 获取模型输出中的logits
        logits = outputs["logits"]
        device = logits.device
        # 计算目标长度，即每个目标中类别标签的数量
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算预测中非"no-object"（最后一个类别）的数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 计算绝对误差
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_boxes
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # 检查是否在输出中找到了预测框
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        # 获取源排列索引
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算边界框的L1回归损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算GIoU损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss._get_source_permutation_idx
    def _get_source_permutation_idx(self, indices):
        # 重新排列预测，按照给定的索引
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss._get_target_permutation_idx
    def _get_target_permutation_idx(self, indices):
        # 重新排列目标，按照给定的索引
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.get_loss
    # 定义一个字典，将损失函数名映射到对应的损失函数方法
    loss_map = {
        "labels": self.loss_labels,
        "cardinality": self.loss_cardinality,
        "boxes": self.loss_boxes,
    }
    # 如果输入的损失函数名不在映射字典中，则抛出数值错误异常
    if loss not in loss_map:
        raise ValueError(f"Loss {loss} not supported")
    # 根据输入的损失函数名调用对应的损失函数方法，并传入相应的参数
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
        # 从输出中排除辅助输出和编码器输出，创建新的输出字典
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in ("auxiliary_outputs", "enc_outputs")}

        # 根据是否启用第二阶段分配器，获取输出和目标之间的匹配
        if self.assign_second_stage:
            indices = self.stg2_assigner(outputs_without_aux, targets)
        else:
            indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点上目标框的平均数量，用于归一化
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # 检查是否已初始化分布式状态
        world_size = 1
        if PartialState._shared_state != {}:
            num_boxes = reduce(num_boxes)
            world_size = PartialState().num_processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 对于辅助损失，重复上述过程以处理每个中间层的输出
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                if not self.assign_second_stage:
                    indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 如果存在编码器输出，处理编码器输出
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
            if self.assign_first_stage:
                indices = self.stg1_assigner(enc_outputs, bin_targets)
            else:
                indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
# 从transformers.models.detr.modeling_detr.DetrMLPPredictionHead复制而来的类
class DetaMLPPredictionHead(nn.Module):
    """
    非常简单的多层感知器（MLP，也称为FFN），用于预测边界框相对于图像的归一化中心坐标、高度和宽度。

    从https://github.com/facebookresearch/detr/blob/master/models/detr.py复制而来

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 创建多层感知器的各层线性变换
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 对输入数据进行多层感知器的前向传播
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 从transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrHungarianMatcher复制而来，将DeformableDetr->Deta
class DetaHungarianMatcher(nn.Module):
    """
    此类计算网络的目标和预测之间的分配。

    由于效率原因，目标不包括no_object。因此，一般情况下，预测比目标多。在这种情况下，我们对最佳预测进行1对1匹配，而其他预测则未匹配（因此被视为非对象）。

    Args:
        class_cost:
            匹配成本中分类错误的相对权重。
        bbox_cost:
            匹配成本中边界框坐标的L1误差的相对权重。
        giou_cost:
            匹配成本中边界框giou损失的相对权重。
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("Matcher的所有成本不能为0")

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
        # 获取输出的形状信息
        batch_size, num_queries = outputs["logits"].shape[:2]

        # 将输出概率展平以计算成本矩阵
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 合并目标标签和框
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算分类成本
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # 计算框之间的 L1 成本
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # 计算框之间的 giou 成本
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # 最终成本矩阵
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # 计算目标框的大小和索引
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# 从 transformers.models.detr.modeling_detr._upcast 复制代码
def _upcast(t: Tensor) -> Tensor:
    # 通过升级到相应更高类型来防止乘法时的数值溢出
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# 从 transformers.models.detr.modeling_detr.box_area 复制代码
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，这些边界框由其 (x1, y1, x2, y2) 坐标指定。

    参数:
        boxes (`torch.FloatTensor`，形状为 `(number_of_boxes, 4)`):
            将计算面积的边界框。它们应该以 (x1, y1, x2, y2) 格式给出，其中 `0 <= x1 < x2` 且 `0 <= y1 < y2`。

    返回:
        `torch.FloatTensor`: 包含每个边界框面积的张量。
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# 从 transformers.models.detr.modeling_detr.box_iou 复制代码
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


# 从 transformers.models.detr.modeling_detr.generalized_box_iou 复制代码
def generalized_box_iou(boxes1, boxes2):
    """
    来自 https://giou.stanford.edu/ 的广义 IoU。这些框应该以 [x0, y0, x1, y1]（角）格式给出。

    返回:
        `torch.FloatTensor`: 一个 [N, M] 的成对矩阵，其中 N = len(boxes1)，M = len(boxes2)
    """
    # 异常框将导致 inf / nan 结果
    # 因此进行早期检查
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 必须以 [x0, y0, x1, y1]（角）格式给出，但得到 {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 必须以 [x0, y0, x1, y1]（角）格式给出，但得到 {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# 来自 https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/layers/wrappers.py#L100
def nonzero_tuple(x):
    """
    'as_tuple=True' 版本的 torch.nonzero，以支持 torchscript。因为
    https://github.com/pytorch/pytorch/issues/38718
    """
    # 检查当前是否处于 Torch 脚本模式
    if torch.jit.is_scripting():
        # 如果是脚本模式，并且张量 x 的维度为 0（标量），则将其增加一个维度并返回其非零索引的元组
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        # 如果是脚本模式，但张量 x 的维度不为 0，则返回张量 x 的非零元素索引的元组
        return x.nonzero().unbind(1)
    else:
        # 如果不是脚本模式，则返回张量 x 的非零元素索引的元组，以元组形式返回
        return x.nonzero(as_tuple=True)
# 导入 List 类型
from typing import List

# 创建 DetaMatcher 类，用于将每个预测的元素（如盒子）分配给一个地面实况元素。每个预测元素将具有零个或一个匹配；每个地面实况元素可能与零个或多个预测元素匹配。
class DetaMatcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth element. Each predicted element will
    have exactly zero or one matches; each ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes how well each (ground-truth,
    prediction)-pair match each other. For example, if the elements are boxes, this matrix may contain box
    intersection-over-union overlap values.

    The matcher returns (a) a vector of length N containing the index of the ground-truth element m in [0, M) that
    matches to prediction n in [0, N). (b) a vector of length N containing the labels for each prediction.
    """

    # 初始化 DetaMatcher 类，接受阈值（thresholds）、标签（labels）和是否允许低质量匹配（allow_low_quality_matches）作为参数
    def __init__(self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool = False):
        """
        Args:
            thresholds (`list[float]`):
                A list of thresholds used to stratify predictions into levels.
            labels (`list[int`):
                A list of values to label predictions belonging at each level. A label can be one of {-1, 0, 1}
                signifying {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (`bool`, *optional*, defaults to `False`):
                If `True`, produce additional matches for predictions with maximum match quality lower than
                high_threshold. See `set_low_quality_matches_` for more details.

            For example,
                thresholds = [0.3, 0.5] labels = [0, -1, 1] All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training. All predictions with 0.3 <= iou < 0.5 will
                be marked with -1 and thus will be ignored. All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        # 复制阈值列表并在列表开头和结尾添加负无穷和正无穷
        thresholds = thresholds[:]
        if thresholds[0] < 0:
            raise ValueError("Thresholds should be positive")
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        # 检查阈值是否按升序排列
        if not all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])):
            raise ValueError("Thresholds should be sorted.")
        # 检查标签是否为-1、0或1
        if not all(l in [-1, 0, 1] for l in labels):
            raise ValueError("All labels should be either -1, 0 or 1")
        # 检查标签数量是否等于阈值数量减1
        if len(labels) != len(thresholds) - 1:
            raise ValueError("Number of labels should be equal to number of thresholds - 1")
        # 初始化阈值、标签和是否允许低质量匹配的属性
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches
    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted elements. All elements must be >= 0
                (due to the us of `torch.nonzero` for selecting indices in `set_low_quality_matches_`).
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2  # 检查输入张量的维度是否为2
        if match_quality_matrix.numel() == 0:  # 检查输入张量是否为空
            default_matches = match_quality_matrix.new_full((match_quality_matrix.size(1),), 0, dtype=torch.int64)  # 创建一个全为0的默认匹配张量
            default_match_labels = match_quality_matrix.new_full(  # 创建一个全为self.labels[0]的默认匹配标签张量
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels  # 返回默认匹配张量和默认匹配标签张量
        assert torch.all(match_quality_matrix >= 0)  # 检查输入张量所有元素是否都大于等于0
        matched_vals, matches = match_quality_matrix.max(dim=0)  # 在每列上找到最大值和对应的索引，返回最大值和索引张量
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)  # 创建一个全为1的匹配标签张量
        for l, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):  # 遍历self.labels和self.thresholds
            low_high = (matched_vals >= low) & (matched_vals < high)  # 创建一个布尔张量，表示matched_vals是否在low和high之间
            match_labels[low_high] = l  # 根据条件将对应位置的match_labels设置为l
        if self.allow_low_quality_matches:  # 如果允许低质量匹配
            self.set_low_quality_matches_(match_labels, match_quality_matrix)  # 调用set_low_quality_matches_函数
        return matches, match_labels  # 返回匹配张量和匹配标签张量
    # 为预测值中仅具有低质量匹配的情况生成额外的匹配。
    # 具体来说，对于每个地面真实框 G，找到与其具有最大重叠的预测框集合（包括平局）；
    # 对于该集合中的每个预测框，如果它尚未匹配，则将其与地面真实框 G 进行匹配。
    # 该函数实现了 :paper:`Faster R-CNN` 中第 3.1.2 节中的 RPN 分配情况 (i)。

    # 对于每个地面真实框，找到与其具有最高质量的预测框
    highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
    # 找到可用的最高质量匹配，即使它是低质量的，包括平局。
    # 注意，由于使用了 `torch.nonzero`，匹配质量必须是正的。
    _, pred_inds_with_highest_quality = nonzero_tuple(match_quality_matrix == highest_quality_foreach_gt[:, None])
    # 如果一个锚框只因为与 gt_A 的低质量匹配而被标记为正，但它与 gt_B 的重叠更大，它的匹配索引仍将是 gt_B。
    # 这遵循 Detectron 中的实现，并且发现没有显著影响。
    match_labels[pred_inds_with_highest_quality] = 1
# 从 https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/modeling/sampling.py#L9 导入 torch.Tensor 类型的 labels、int 类型的 num_samples、float 类型的 positive_fraction 以及 int 类型的 bg_label
def subsample_labels(labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int):
    """
    从包含正负样本混合的标签 labels 中随机选取 `num_samples` 个样本（或更少，如果找不到足够的样本）。
    它将尽可能返回尽可能多的正样本，但不超过 `positive_fraction * num_samples`，然后尝试用负样本填充剩余的位置。

    参数:
        labels (Tensor): (N, ) 包含以下值的标签向量:
            * -1: 忽略
            * bg_label: 背景（"negative"）类
            * 其他: 一个或多个前景（"positive"）类
        num_samples (int): 要返回值大于等于 0 的标签总数。
            未被抽取的值将用 -1（忽略）填充。
        positive_fraction (float): 采样的正值为 `min(num_positives, int(positive_fraction * num_samples))`。
            采样的负值为 `min(num_negatives, num_samples - num_positives_sampled)`。
            换句话说，如果正样本不够，采样将用负样本填充。如果负样本也不够，则尽可能多地采样元素。
        bg_label (int): 背景（"negative"）类的标签索引。

    返回:
        pos_idx, neg_idx (Tensor):
            1D 索引向量。长度为 `num_samples` 或更少。
    """
    找到所有非 -1 且非 bg_label 的正样本索引
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    找到所有等于 bg_label 的负样本索引
    negative = nonzero_tuple(labels == bg_label)[0]

    计算需要保留的正样本数
    num_pos = int(num_samples * positive_fraction)
    # 防止正例样本不足
    num_pos = min(positive.numel(), num_pos)
    计算需要保留的负样本数
    num_neg = num_samples - num_pos
    # 防止负例样本不足
    num_neg = min(negative.numel(), num_neg)

    # 随机选择正例和负例样本
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    选择的正例索引
    pos_idx = positive[perm1]
    选择的负例索引
    neg_idx = negative[perm2]
    返回正例索引和负例索引
    return pos_idx, neg_idx


def sample_topk_per_gt(pr_inds, gt_inds, iou, k):
    如果 gt_inds 的长度为 0，则返回 pr_inds 和 gt_inds
    if len(gt_inds) == 0:
        return pr_inds, gt_inds
    # 找到每个 gt 的前 k 个匹配项
    查询 gt_inds 中唯一值，返回唯一值和计数
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    通过 iou[gt_inds2] 找到最高的 k 个分数和对应的 pr 索引
    scores, pr_inds2 = iou[gt_inds2].topk(k, dim=1)
    将 gt_inds2 扩展为与 pr_inds2 相同的维度
    gt_inds2 = gt_inds2[:, None].repeat(1, k)

    # 过滤到与 gt 数量相同的匹配
    展平 pr_inds2，保留每个 gt 对应数量的 pr 索引
    pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2)])
    展平 gt_inds2，保留每个 gt 对应数量的 gt 索引
    gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])
    返回 pr_inds3 和 gt_inds3
    return pr_inds3, gt_inds3
class DetaStage2Assigner(nn.Module):
    # 初始化函数，初始化正样本比例、背景标签、每张图片的候选框数和匹配器
    def __init__(self, num_queries, max_k=4):
        super().__init__()
        self.positive_fraction = 0.25  # 正样本比例
        self.bg_label = 400  # 背景标签，用于过滤
        self.batch_size_per_image = num_queries  # 每张图片的候选框数
        self.proposal_matcher = DetaMatcher(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True)  # 匹配器
        self.k = max_k  # 最大K值

    # 根据匹配结果和真实类别对候选框进行采样，并设置它们的分类标签
    def _sample_proposals(self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor):
        """
        根据N个候选框和M个真实框的匹配关系，对候选框进行采样，并设置它们的分类标签。

        参数:
            matched_idxs (Tensor): 长度为N的向量，每个元素是每个候选框对应的最佳匹配的真实框索引，取值范围[0, M)。
            matched_labels (Tensor): 长度为N的向量，每个元素是匹配器的标签（cfg.MODEL.ROI_HEADS.IOU_LABELS）。
            gt_classes (Tensor): 长度为M的向量，每个元素是真实框的类别。

        返回:
            Tensor: 采样后的候选框的索引向量，取值范围[0, N)。
            Tensor: 与采样后的候选框对应的分类标签向量，每个样本的类别为[0, num_classes)或背景类别（num_classes）。
        """
        has_gt = gt_classes.numel() > 0
        # 获取每个候选框对应的真实框的类别
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # 将未匹配的候选框（匹配器输出标签为0）标记为背景（标签=num_classes）
            gt_classes[matched_labels == 0] = self.bg_label
            # 标记忽略的候选框（标签为-1）
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.bg_label

        # 对正负样本进行采样
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.bg_label
        )

        # 将采样得到的正负样本合并成最终的采样结果
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]
    # 前向传播函数，计算输出和目标的匹配信息
    def forward(self, outputs, targets, return_cost_matrix=False):
        # 获取批量大小
        bs = len(targets)
        # 初始化索引列表和IoU列表
        indices = []
        ious = []
        # 遍历批量中的每个样本
        for b in range(bs):
            # 计算输出和目标框之间的IoU
            iou, _ = box_iou(
                center_to_corners_format(targets[b]["boxes"]),
                center_to_corners_format(outputs["init_reference"][b].detach()),
            )
            # 利用提议匹配器获取匹配索引和标签
            matched_idxs, matched_labels = self.proposal_matcher(
                iou
            )  # proposal_id -> highest_iou_gt_id, proposal_id -> [1 if iou > 0.6, 0 ow]
            # 采样提议
            (
                sampled_idxs,
                sampled_gt_classes,
            ) = self._sample_proposals(  # list of sampled proposal_ids, sampled_id -> [0, num_classes)+[bg_label]
                matched_idxs, matched_labels, targets[b]["class_labels"]
            )
            # 获取正样本提议和对应的目标索引
            pos_pr_inds = sampled_idxs[sampled_gt_classes != self.bg_label]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            # 后处理提议索引
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            # 将结果存入索引列表和IoU列表
            indices.append((pos_pr_inds, pos_gt_inds))
            ious.append(iou)
        # 如果需要返回成本矩阵，则返回索引和IoU列表
        if return_cost_matrix:
            return indices, ious
        # 否则只返回索引列表
        return indices

    # 后处理提议索引的函数
    def postprocess_indices(self, pr_inds, gt_inds, iou):
        # 返回按每个目标进行采样的top-k提议索引
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)
# 自定义的 DetaStage1Assigner 类，继承自 nn.Module
class DetaStage1Assigner(nn.Module):
    def __init__(self, t_low=0.3, t_high=0.7, max_k=4):
        super().__init__()
        self.positive_fraction = 0.5
        self.batch_size_per_image = 256
        self.k = max_k
        self.t_low = t_low
        self.t_high = t_high
        # 创建 DetaMatcher 对象用于匹配锚框和真实框
        self.anchor_matcher = DetaMatcher(
            thresholds=[t_low, t_high], labels=[0, -1, 1], allow_low_quality_matches=True
        )

    # 根据 label 进行正负样本的子抽样，用 -1 标记未抽样到的样本
    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite the label vector to the ignore value
        (-1) for all elements that are not included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        # 随机抽样正负样本的索引
        pos_idx, neg_idx = subsample_labels(label, self.batch_size_per_image, self.positive_fraction, 0)
        # 填充为 ignore 标签 (-1)，然后设置正负样本标签
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    # 前向传播函数
    def forward(self, outputs, targets):
        bs = len(targets)
        indices = []
        for b in range(bs):
            anchors = outputs["anchors"][b]
            if len(targets[b]["boxes"]) == 0:
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=anchors.device),
                        torch.tensor([], dtype=torch.long, device=anchors.device),
                    )
                )
                continue
            # 计算真实框与锚框的 IoU，进行匹配
            iou, _ = box_iou(
                center_to_corners_format(targets[b]["boxes"]),
                center_to_corners_format(anchors),
            )
            matched_idxs, matched_labels = self.anchor_matcher(
                iou
            )  # proposal_id -> highest_iou_gt_id, proposal_id -> [1 if iou > 0.7, 0 if iou < 0.3, -1 ow]
            matched_labels = self._subsample_labels(matched_labels)

            all_pr_inds = torch.arange(len(anchors), device=matched_labels.device)
            pos_pr_inds = all_pr_inds[matched_labels == 1]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            pos_pr_inds, pos_gt_inds = pos_pr_inds.to(anchors.device), pos_gt_inds.to(anchors.device)
            indices.append((pos_pr_inds, pos_gt_inds))
        return indices

    # 后处理索引，对正例进行筛选
    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)
```