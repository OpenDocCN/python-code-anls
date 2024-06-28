# `.\models\deta\modeling_deta.py`

```
# 设置编码格式为 UTF-8
# 版权声明，指明 SenseTime 和 The HuggingFace Inc. 团队的所有权，保留所有权利
#
# 根据 Apache 许可证 2.0 版本使用本文件，除非符合许可证中的条款，否则不得使用本文件
# 您可以在以下网址获取许可证的副本:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 软件没有任何形式的担保或条件，明示或暗示
# 有关特定语言的权限，请参阅许可证

""" PyTorch DETA model. """

# 导入所需的库和模块
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

# 导入自定义模块和函数
from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_torch_cuda_available,
    is_vision_available,
    replace_return_docstrings,
)
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 多尺度可变形注意力模块的全局变量
MultiScaleDeformableAttention = None


# 从 models.deformable_detr.load_cuda_kernels 复制过来的函数
def load_cuda_kernels():
    # 导入 torch.utils.cpp_extension 中的 load 函数
    from torch.utils.cpp_extension import load

    global MultiScaleDeformableAttention

    # 获取 CUDA 内核源文件的路径
    root = Path(__file__).resolve().parent.parent.parent / "kernels" / "deta"
    src_files = [
        root / filename
        for filename in [
            "vision.cpp",
            os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
            os.path.join("cuda", "ms_deform_attn_cuda.cu"),
        ]
    ]

    # 加载 CUDA 内核
    load(
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


# 从 transformers.models.deformable_detr.modeling_deformable_detr.MultiScaleDeformableAttentionFunction 复制过来的类
class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    def forward(
        context,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
        # forward 方法的静态方法，执行前向传播
    ):
        # 设置上下文对象的 im2col_step 属性为传入的 im2col_step 值
        context.im2col_step = im2col_step
        # 调用 MultiScaleDeformableAttention 类的静态方法 ms_deform_attn_forward 进行前向传播
        output = MultiScaleDeformableAttention.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            context.im2col_step,  # 使用上下文中的 im2col_step 参数
        )
        # 将需要反向传播的张量保存到上下文中
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
        # 调用 MultiScaleDeformableAttention 类的静态方法 ms_deform_attn_backward 进行反向传播计算梯度
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttention.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            context.im2col_step,  # 使用上下文中的 im2col_step 参数
        )

        # 返回梯度，其中第二和第三个返回值为 None
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
# 如果加速库可用，则导入 PartialState 和 reduce 函数
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

# 如果视觉库可用，则导入 center_to_corners_format 函数
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

# 如果 TorchVision 库可用，则导入 batched_nms 函数
if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms

# 如果 SciPy 库可用，则导入 linear_sum_assignment 函数
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置字符串，指定为 "DetaConfig"
_CONFIG_FOR_DOC = "DetaConfig"

# 用于文档的检查点字符串，指定为 "jozhang97/deta-swin-large-o365"
_CHECKPOINT_FOR_DOC = "jozhang97/deta-swin-large-o365"

# DETA 预训练模型存档列表，列出了一个预训练模型
DETA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "jozhang97/deta-swin-large-o365",
    # 查看所有 DETA 模型的列表：https://huggingface.co/models?filter=deta
]

@dataclass
# 从 DeformableDetrDecoderOutput 复制而来，用于 DETA 模型的解码器输出
# 继承自 ModelOutput 类，添加了两个属性：
# - 中间解码器隐藏状态的堆叠张量（即每个解码器层的输出）
# - 中间参考点的堆叠张量
class DetaDecoderOutput(ModelOutput):
    """
    DetaDecoder 的输出基类。这个类在 BaseModelOutputWithCrossAttentions 基础上增加了两个属性：
    - 中间解码器隐藏状态的堆叠张量（即每个解码器层的输出）
    - 中间参考点的堆叠张量。
    """
    # 定义函数参数和它们的类型注解，用于描述模型输出的不同隐藏状态和注意力权重
    
    last_hidden_state: torch.FloatTensor = None
    # 最后一个编码器层的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
    
    intermediate_hidden_states: torch.FloatTensor = None
    # 解码器各层的中间隐藏状态堆叠，形状为(batch_size, config.decoder_layers, num_queries, hidden_size)
    
    intermediate_reference_points: torch.FloatTensor = None
    # 解码器各层的中间参考点（每层解码器的参考点），形状为(batch_size, config.decoder_layers, sequence_length, hidden_size)
    
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 模型隐藏状态的元组，包括初始嵌入层输出和每个层的输出，形状为(batch_size, sequence_length, hidden_size)
    
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重的元组，每个元素对应每个解码器层的注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)。在自注意力头中用于计算加权平均值。
    
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的交叉注意力层的注意力权重的元组，每个元素对应每个解码器层的交叉注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)。在交叉注意力头中用于计算加权平均值。
# 用于存储Deformable DETR编码器-解码器模型输出的基类。
@dataclass
class DetaModelOutput(ModelOutput):
    """
    Deformable DETR编码器-解码器模型输出的基类。

    """

    # 初始化参考点张量
    init_reference_points: torch.FloatTensor = None
    # 最后一个隐藏状态张量
    last_hidden_state: torch.FloatTensor = None
    # 中间隐藏状态张量
    intermediate_hidden_states: torch.FloatTensor = None
    # 中间参考点张量
    intermediate_reference_points: torch.FloatTensor = None
    # 解码器的隐藏状态（可选的元组列表）
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力权重（可选的元组列表）
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力权重（可选的元组列表）
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后一个隐藏状态张量（可选）
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态（可选的元组列表）
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力权重（可选的元组列表）
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器输出类别（可选的张量）
    enc_outputs_class: Optional[torch.FloatTensor] = None
    # 编码器输出坐标逻辑（可选的张量）
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None
    # 输出提议（可选的张量）
    output_proposals: Optional[torch.FloatTensor] = None


# 用于存储DetaForObjectDetection模型输出类型的基类。
@dataclass
class DetaObjectDetectionOutput(ModelOutput):
    """
    DetaForObjectDetection模型的输出类型。

    """

    # 损失（可选的浮点张量）
    loss: Optional[torch.FloatTensor] = None
    # 损失字典（可选的字典）
    loss_dict: Optional[Dict] = None
    # logits张量
    logits: torch.FloatTensor = None
    # 预测框张量
    pred_boxes: torch.FloatTensor = None
    # 辅助输出（可选的字典列表）
    auxiliary_outputs: Optional[List[Dict]] = None
    # 初始化参考点张量（可选的浮点张量）
    init_reference_points: Optional[torch.FloatTensor] = None
    # 最后一个隐藏状态张量（可选的浮点张量）
    last_hidden_state: Optional[torch.FloatTensor] = None
    # 中间隐藏状态张量（可选的浮点张量）
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    # 中间参考点张量（可选的浮点张量）
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    # 解码器的隐藏状态（可选的元组列表）
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力权重（可选的元组列表）
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力权重（可选的元组列表）
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后一个隐藏状态张量（可选的浮点张量）
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态（可选的元组列表）
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力权重（可选的元组列表）
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器输出类别（可选）
    enc_outputs_class: Optional = None
    # 编码器输出坐标逻辑（可选）
    enc_outputs_coord_logits: Optional = None
    # 输出提议（可选的浮点张量）
    output_proposals: Optional[torch.FloatTensor] = None


# 创建一个复制指定模块的列表函数。
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# 计算逆sigmoid函数。
def inverse_sigmoid(x, eps=1e-5):
    # 将x限制在0到1之间
    x = x.clamp(min=0, max=1)
    # 对x应用逆sigmoid变换并添加小的eps以避免数值稳定性问题
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# DetaFrozenBatchNorm2d类，从DetrFrozenBatchNorm2d类复制并修改一些部分。
class DetaFrozenBatchNorm2d(nn.Module):
    """
    批量归一化层，其中批次统计信息和仿射参数被固定。

    从torchvision.misc.ops中复制粘贴，添加eps以在没有此项的情况下保证任何模型（而不仅仅是torchvision.models.resnet[18,34,50,101]）不产生nan。
    """

    def __init__(self, n):
        super().__init__()
        # 注册权重张量，并初始化为全1
        self.register_buffer("weight", torch.ones(n))
        # 注册偏置张量，并初始化为全0
        self.register_buffer("bias", torch.zeros(n))
        # 注册运行时均值张量，并初始化为全0
        self.register_buffer("running_mean", torch.zeros(n))
        # 注册运行时方差张量，并初始化为全1
        self.register_buffer("running_var", torch.ones(n))
    # 从模型状态字典中加载权重和偏差，忽略 num_batches_tracked 键
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 构建 num_batches_tracked 的键名
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # 如果 num_batches_tracked 存在于状态字典中，则删除它
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类的加载状态字典方法，传递所有参数
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # 前向传播函数
    def forward(self, x):
        # 将权重重塑为 (1, -1, 1, 1) 的形状，以便与输入张量相乘
        weight = self.weight.reshape(1, -1, 1, 1)
        # 将偏置重塑为 (1, -1, 1, 1) 的形状，以便与输入张量相加
        bias = self.bias.reshape(1, -1, 1, 1)
        # 将运行时方差重塑为 (1, -1, 1, 1) 的形状
        running_var = self.running_var.reshape(1, -1, 1, 1)
        # 将运行时均值重塑为 (1, -1, 1, 1) 的形状
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        # 定义一个极小值 epsilon，用于数值稳定性
        epsilon = 1e-5
        # 计算缩放因子 scale，用于标准化输入数据
        scale = weight * (running_var + epsilon).rsqrt()
        # 调整偏置，使其适应标准化后的输入数据
        bias = bias - running_mean * scale
        # 返回经过标准化和偏置处理的输入数据
        return x * scale + bias
# 从 transformers.models.detr.modeling_detr.replace_batch_norm 复制，并将 Detr->Deta
def replace_batch_norm(model):
    r"""
    递归替换所有的 `torch.nn.BatchNorm2d` 层为 `DetaFrozenBatchNorm2d`。

    Args:
        model (torch.nn.Module):
            输入的模型
    """
    for name, module in model.named_children():
        # 如果当前模块是 nn.BatchNorm2d 类型
        if isinstance(module, nn.BatchNorm2d):
            # 创建一个新的 DetaFrozenBatchNorm2d 实例，参数为原始模块的特征数量
            new_module = DetaFrozenBatchNorm2d(module.num_features)

            # 如果原始模块的权重不在 "meta" 设备上
            if not module.weight.device == torch.device("meta"):
                # 将新模块的权重、偏置、运行均值和方差复制为原始模块的相应值
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            # 将模型中的原始模块替换为新的 DetaFrozenBatchNorm2d 实例
            model._modules[name] = new_module

        # 如果当前模块有子模块，则递归调用 replace_batch_norm 函数
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class DetaBackboneWithPositionalEncodings(nn.Module):
    """
    带有位置编码的主干模型。

    nn.BatchNorm2d 层被上述定义的 DetaFrozenBatchNorm2d 替换。
    """

    def __init__(self, config):
        super().__init__()

        # 加载指定配置的主干模型
        backbone = load_backbone(config)
        
        # 使用 torch.no_grad() 包装，递归替换主干模型中的 BatchNorm 层为 DetaFrozenBatchNorm2d
        with torch.no_grad():
            replace_batch_norm(backbone)
        
        # 将替换后的主干模型设置为当前对象的模型属性
        self.model = backbone
        
        # 获取主干模型中的通道尺寸信息
        self.intermediate_channel_sizes = self.model.channels

        # TODO 修复这个部分
        # 如果主干模型的类型是 "resnet"
        if config.backbone_config.model_type == "resnet":
            # 遍历主干模型的所有参数
            for name, parameter in self.model.named_parameters():
                # 如果参数名中不包含 "stages.1"、"stages.2" 或 "stages.3"
                if "stages.1" not in name and "stages.2" not in name and "stages.3" not in name:
                    # 将参数的 requires_grad 属性设为 False，即冻结参数
                    parameter.requires_grad_(False)

        # 构建位置编码器
        self.position_embedding = build_position_encoding(config)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        """
        如果 `config.num_feature_levels > 1`，则输出 ResNet 中 C_3 到 C_5 的后续阶段的特征图，否则输出 C_5 的特征图。
        """
        # 首先，通过主干模型传递像素值以获取特征图列表
        features = self.model(pixel_values).feature_maps

        # 接下来，创建位置编码
        out = []
        pos = []
        for feature_map in features:
            # 将像素掩码下采样以匹配相应特征图的形状
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            # 使用位置编码器对特征图和掩码生成位置编码
            position_embeddings = self.position_embedding(feature_map, mask).to(feature_map.dtype)
            out.append((feature_map, mask))
            pos.append(position_embeddings)

        return out, pos


# 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrSinePositionEmbedding 复制，并将 DeformableDetr->Deta
class DetaSinePositionEmbedding(nn.Module):
    """
    这是一种更标准的位置编码版本，与 Attention is all you
    """
    需要纸张，通用于处理图像。

    初始化函数，设置模型参数并进行必要的检查。
    """
    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        # 设置嵌入维度
        self.embedding_dim = embedding_dim
        # 温度参数，影响位置编码的范围
        self.temperature = temperature
        # 是否进行归一化处理
        self.normalize = normalize
        # 如果传入了scale但未开启normalize，则引发错误
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 如果未传入scale，则默认设置为2π
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        # 如果未提供像素掩码，则引发错误
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        # 在y方向上对掩码进行累积求和，作为位置编码的一部分
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        # 在x方向上对掩码进行累积求和，作为位置编码的一部分
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        # 如果设置了归一化，则对位置编码进行归一化处理
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成维度向量，用于计算位置编码
        dim_t = torch.arange(self.embedding_dim, dtype=torch.int64, device=pixel_values.device).float()
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        # 计算x和y方向上的位置编码
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 对位置编码进行正弦和余弦变换，然后展平处理
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 拼接x和y方向的位置编码，并将通道维度放到正确的位置
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # 返回位置编码结果
        return pos
# Copied from transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding
class DetaLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        # 创建一个嵌入层用于行位置编码，嵌入维度为 embedding_dim，总共有 50 个位置
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        # 创建一个嵌入层用于列位置编码，嵌入维度为 embedding_dim，总共有 50 个位置
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    def forward(self, pixel_values, pixel_mask=None):
        # 获取输入像素值的高度和宽度
        height, width = pixel_values.shape[-2:]
        # 创建一个张量，包含从 0 到 width-1 的整数，设备类型与 pixel_values 相同
        width_values = torch.arange(width, device=pixel_values.device)
        # 创建一个张量，包含从 0 到 height-1 的整数，设备类型与 pixel_values 相同
        height_values = torch.arange(height, device=pixel_values.device)
        # 获取列位置的嵌入向量
        x_emb = self.column_embeddings(width_values)
        # 获取行位置的嵌入向量
        y_emb = self.row_embeddings(height_values)
        # 拼接列和行位置的嵌入向量，形成位置编码矩阵
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # 将位置编码矩阵进行维度置换，变为 (embedding_dim, height, width) 的形式
        pos = pos.permute(2, 0, 1)
        # 在最前面添加一个维度，变为 (1, embedding_dim, height, width) 的形式
        pos = pos.unsqueeze(0)
        # 将位置编码矩阵扩展为与输入像素值相同的张量形状
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # 返回位置编码张量
        return pos


# Copied from transformers.models.detr.modeling_detr.build_position_encoding with Detr->Deta
def build_position_encoding(config):
    # 计算位置编码的步数，为模型维度除以 2
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # 若使用正弦位置编码类型，则创建 DetaSinePositionEmbedding 对象
        # 此处暂未暴露其他参数的更好方法
        position_embedding = DetaSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        # 若使用学习得到的位置编码类型，则创建 DetaLearnedPositionEmbedding 对象
        position_embedding = DetaLearnedPositionEmbedding(n_steps)
    else:
        # 抛出异常，指出不支持的位置编码类型
        raise ValueError(f"Not supported {config.position_embedding_type}")

    # 返回创建的位置编码对象
    return position_embedding


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    # 获取输入 value 张量的维度信息
    batch_size, _, num_heads, hidden_dim = value.shape
    # 获取 sampling_locations 张量的维度信息
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # 将 value 张量按照空间形状进行分割，并存储到 value_list 中
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    # 计算采样网格，将采样位置放大为 2 倍并减去 1
    sampling_grids = 2 * sampling_locations - 1
    # 初始化采样值列表为空列表
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # 遍历每个级别的空间形状，level_id 是级别索引，(height, width) 是高度和宽度元组
        # 扁平化 value_list[level_id]，将其转置，然后重塑为指定形状
        # 得到形状为 batch_size*num_heads, hidden_dim, height, width 的 value_l_
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        
        # 提取 sampling_grids 中特定 level_id 的数据，进行转置和扁平化操作
        # 得到形状为 batch_size*num_heads, num_queries, num_points, 2 的 sampling_grid_l_
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        
        # 使用双线性插值的方式，根据 sampling_grid_l_ 对 value_l_ 进行采样
        # 得到形状为 batch_size*num_heads, hidden_dim, num_queries, num_points 的 sampling_value_l_
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        
        # 将当前级别的采样值列表添加到 sampling_value_list 中
        sampling_value_list.append(sampling_value_l_)
    
    # 调整注意力权重的形状，转置以匹配后续计算的需求
    # 最终形状为 batch_size, num_heads, num_queries, num_levels * num_points
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    
    # 计算最终的输出，对采样值列表进行堆叠和扁平化操作，并乘以注意力权重，最后重新调整形状
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    
    # 返回输出，并重新调整其形状以匹配预期的输出格式
    return output.transpose(1, 2).contiguous()
# 从transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention复制而来，修改为DetaMultiscaleDeformableAttention
class DetaMultiscaleDeformableAttention(nn.Module):
    """
    在Deformable DETR中提出的多尺度可变形注意力机制。
    """

    def __init__(self, config: DetaConfig, num_heads: int, n_points: int):
        super().__init__()

        # 检查是否加载了CUDA内核函数并且Ninja库可用，如果没有加载则尝试加载
        kernel_loaded = MultiScaleDeformableAttention is not None
        if is_torch_cuda_available() and is_ninja_available() and not kernel_loaded:
            try:
                load_cuda_kernels()
            except Exception as e:
                logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")

        # 检查d_model是否能被num_heads整除，如果不能则抛出错误
        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        
        # 计算每个注意力头的维度
        dim_per_head = config.d_model // num_heads
        # 检查dim_per_head是否是2的幂
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DetaMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        # 初始化im2col步长为64
        self.im2col_step = 64

        # 保存配置参数
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        # 初始化偏移量线性层、注意力权重线性层、值投影线性层和输出投影线性层
        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        # 根据配置参数决定是否禁用自定义内核函数
        self.disable_custom_kernels = config.disable_custom_kernels

        # 重置模型参数
        self._reset_parameters()
    # 重置模型参数的方法
    def _reset_parameters(self):
        # 初始化采样偏移权重为常数 0.0
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        
        # 获取默认的数据类型
        default_dtype = torch.get_default_dtype()
        
        # 创建角度列表 thetas，作为每个注意力头的角度偏移量
        thetas = torch.arange(self.n_heads, dtype=torch.int64).to(default_dtype) * (2.0 * math.pi / self.n_heads)
        
        # 初始化网格 grid_init，其中包含每个头的位置编码
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        
        # 标准化网格使其范围在 [-1, 1] 内，并重复以匹配所有级别和点的数量
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        
        # 根据点的索引调整网格初始化值
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        
        # 使用 torch.no_grad() 上下文管理器，将初始化的网格作为偏移量的偏置参数
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # 初始化注意力权重为常数 0.0
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        
        # 使用 xavier_uniform 方法初始化值投影权重和偏置
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)
# 从transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiheadAttention复制代码，将DeformableDetr->Deta,Deformable DETR->DETA
class DetaMultiheadAttention(nn.Module):
    """
    'Attention Is All You Need'论文中的多头注意力机制。

    这里，我们根据Deformable DETR论文的说明，为查询和键添加位置嵌入。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除 (得到 `embed_dim`: {self.embed_dim} 和 `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        # 线性层，用于投影键（key）、值（value）和查询（query）
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        # 重新形状张量，以便进行多头注意力计算
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        # 如果提供了位置嵌入，则将其添加到张量中
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 前向传播函数，计算多头注意力机制



class DetaEncoderLayer(nn.Module):
    def __init__(self, config: DetaConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = DetaMultiscaleDeformableAttention(
            config,
            num_heads=config.encoder_attention_heads,
            n_points=config.encoder_n_points,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
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
    ):
        # 前向传播函数，包括多尺度可变形注意力和前馈神经网络层的计算
    # 定义一个方法，用于处理多尺度变形注意力模块的计算
    def forward(
        self,
        hidden_states,  # 输入到层的隐藏状态张量，形状为(batch_size, sequence_length, hidden_size)
        attention_mask,  # 注意力掩码张量，形状为(batch_size, sequence_length)
        position_embeddings=None,  # 位置嵌入张量，可选参数，将添加到hidden_states中
        reference_points=None,  # 参考点张量，可选参数
        spatial_shapes=None,  # 主干特征图的空间形状张量，可选参数
        level_start_index=None,  # 级别开始索引张量，可选参数
        output_attentions=False,  # 是否返回所有注意力层的注意力张量的标志，详见返回的张量中的'attentions'
    ):
        residual = hidden_states  # 保存初始输入的残差连接
    
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
    
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 使用dropout进行正则化
        hidden_states = residual + hidden_states  # 添加残差连接
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 对结果进行层归一化处理
    
        residual = hidden_states  # 保存第一次处理后的残差连接
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 使用激活函数处理全连接层fc1的结果
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 使用dropout进行正则化
    
        hidden_states = self.fc2(hidden_states)  # 进行第二个全连接层fc2的线性变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 使用dropout进行正则化
    
        hidden_states = residual + hidden_states  # 添加第二次残差连接
        hidden_states = self.final_layer_norm(hidden_states)  # 对结果进行最终的层归一化处理
    
        if self.training:  # 如果处于训练模式
            # 检查是否存在无穷大或NaN值，如果有，则进行数值截断
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    
        outputs = (hidden_states,)  # 输出结果为隐藏状态张量的元组
    
        if output_attentions:  # 如果需要返回注意力张量
            outputs += (attn_weights,)  # 将注意力张量添加到输出元组中
    
        return outputs  # 返回最终输出
class DetaDecoderLayer(nn.Module):
    def __init__(self, config: DetaConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = DetaMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # Layer normalization for self-attention output
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # cross-attention
        self.encoder_attn = DetaMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        # Layer normalization for cross-attention output
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # Layer normalization for final output after feedforward networks
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
    ):
        # Forward pass through the decoder layer
        # Self-attention mechanism
        self_attn_output = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=None,  # Optional masking
            output_attentions=output_attentions,
        )
        self_attn_output = self.dropout(self_attn_output[0])  # Apply dropout
        hidden_states = hidden_states + self_attn_output  # Residual connection
        hidden_states = self.self_attn_layer_norm(hidden_states)  # Layer normalization

        # Cross-attention mechanism
        encoder_attn_output = self.encoder_attn(
            hidden_states,
            encoder_hidden_states,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            attn_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        encoder_attn_output = self.dropout(encoder_attn_output[0])  # Apply dropout
        hidden_states = hidden_states + encoder_attn_output  # Residual connection
        hidden_states = self.encoder_attn_layer_norm(hidden_states)  # Layer normalization

        # Feedforward neural network
        intermediate_output = self.activation_fn(self.fc1(hidden_states))  # First linear layer + activation
        intermediate_output = self.dropout(intermediate_output)  # Apply dropout
        ffn_output = self.fc2(intermediate_output)  # Second linear layer
        ffn_output = self.dropout(ffn_output)  # Apply dropout
        hidden_states = hidden_states + ffn_output  # Residual connection
        hidden_states = self.final_layer_norm(hidden_states)  # Layer normalization

        return hidden_states



# Copied from transformers.models.detr.modeling_detr.DetrClassificationHead
class DetaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        # Fully connected layer for dimension reduction
        self.dense = nn.Linear(input_dim, inner_dim)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=pooler_dropout)
        # Output projection layer for classification
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)  # Apply dropout
        hidden_states = self.dense(hidden_states)  # Linear transformation
        hidden_states = torch.tanh(hidden_states)  # Apply activation function (tanh)
        hidden_states = self.dropout(hidden_states)  # Apply dropout
        hidden_states = self.out_proj(hidden_states)  # Final linear transformation for classification
        return hidden_states


class DetaPreTrainedModel(PreTrainedModel):
    config_class = DetaConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DetaBackboneWithPositionalEncodings", r"DetaEncoderLayer", r"DetaDecoderLayer"]
    supports_gradient_checkpointing = True
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        std = self.config.init_std  # 获取初始化标准差参数

        if isinstance(module, DetaLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)  # 均匀分布初始化行嵌入权重
            nn.init.uniform_(module.column_embeddings.weight)  # 均匀分布初始化列嵌入权重
        elif isinstance(module, DetaMultiscaleDeformableAttention):
            module._reset_parameters()  # 重置多尺度可变形注意力模块的参数
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 对于线性层、二维卷积层、批归一化层，使用正态分布初始化权重，偏置初始化为零
            # 与 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对于嵌入层，使用正态分布初始化权重，如果定义了填充索引，则对应权重置零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if hasattr(module, "reference_points") and not self.config.two_stage:
            # 如果模块具有 reference_points 属性且非两阶段配置，则使用 Xavier 均匀分布初始化权重，偏置初始化为零
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            # 如果模块具有 level_embed 属性，则使用正态分布初始化权重
            nn.init.normal_(module.level_embed)
# DETA_START_DOCSTRING 是一个原始文档字符串，描述了这个模型的继承和一般行为。
# 它继承自 PreTrainedModel 类，可以查阅该超类的文档以了解其实现的通用方法，
# 如下载或保存模型、调整输入嵌入大小、修剪头部等。

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

# DETA_INPUTS_DOCSTRING 是一个空的文档字符串，将用于描述输入参数的信息。
DETA_INPUTS_DOCSTRING = r"""
    # 定义函数参数，接受一个四维的浮点型张量作为输入，表示像素值，维度为(batch_size, num_channels, height, width)
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.
            
            像素数值。默认情况下将忽略填充部分的像素值。

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`AutoImageProcessor.__call__`] for details.
            
            可以使用 [`AutoImageProcessor`] 获得像素值。详细信息请参见 [`AutoImageProcessor.__call__`]。

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
            
            遮罩，用于避免在填充像素值上执行注意力操作。遮罩的取值范围为 `[0, 1]`：

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
            
            - 值为 1 表示真实像素（即**未遮罩**），
            - 值为 0 表示填充像素（即**已遮罩**）。

            [What are attention masks?](../glossary#attention-mask)
            
            [注意力遮罩是什么？](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
            
            默认情况下不使用。可以用于屏蔽对象查询。

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            
            元组包含 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)

            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            
            `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选*，是编码器最后一层的输出隐藏状态序列。用于解码器的交叉注意力。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
            
            可选项，可以直接传递一个展平的图像表示，而不是传递扁平化的特征图（骨干网络输出 + 投影层的输出）。

        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
            
            可选项，可以直接传递嵌入表示，而不是用零张量初始化查询。

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
            
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
            
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            
            是否返回 [`~file_utils.ModelOutput`] 而不是普通元组。
"""


class DetaEncoder(DetaPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`DetaEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: DetaConfig
    """

    def __init__(self, config: DetaConfig):
        super().__init__(config)

        self.dropout = config.dropout
        # 创建多个 DetaEncoderLayer 组成的层列表
        self.layers = nn.ModuleList([DetaEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
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
            # 创建网格矩阵，生成参考点
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # 重新形状和缩放参考点，考虑有效比率
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # 拼接所有参考点并应用有效比率
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
    # 将 `position_embeddings`, `reference_points`, `spatial_shapes` 和 `valid_ratios` 添加到前向传播中。
    # 同时返回所有解码层的中间输出和参考点的堆栈。

    Args:
        config: DetaConfig
    """

    def __init__(self, config: DetaConfig):
        super().__init__(config)

        # 初始化配置中的参数
        self.dropout = config.dropout
        # 创建指定数量的解码层，并存储在模块列表中
        self.layers = nn.ModuleList([DetaDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        # 用于迭代边界框细化和两阶段可变形DETR的特殊实现
        self.bbox_embed = None
        self.class_embed = None

        # 初始化权重并应用最终处理
        self.post_init()

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
"""
The bare DETA Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
any specific head on top.
"""
# 继承自预训练模型基类 DetaPreTrainedModel 的 DetaModel 类
@add_start_docstrings(
    """
    The bare DETA Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
    any specific head on top.
    """,
    DETA_START_DOCSTRING,
)
class DetaModel(DetaPreTrainedModel):
    # 初始化函数，接收一个 DetaConfig 对象作为配置参数
    def __init__(self, config: DetaConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置指定了 two_stage 为 True，则要求导入 torch 和 torchvision 库
        if config.two_stage:
            requires_backends(self, ["torchvision"])

        # 创建带有位置编码的背景骨干网络
        self.backbone = DetaBackboneWithPositionalEncodings(config)
        # 获取背景骨干网络中间层的通道大小信息
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        # 创建输入投影层
        if config.num_feature_levels > 1:
            num_backbone_outs = len(intermediate_channel_sizes)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = intermediate_channel_sizes[_]
                # 对每个输入层进行卷积和分组归一化
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                # 对于额外的输入层，使用卷积核大小为 3，步幅为 2，填充为 1 的卷积层和分组归一化
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
                in_channels = config.d_model
            # 将输入投影层作为模块列表保存
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # 如果只有一个特征级别，则直接创建一个输入投影层
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                ]
            )

        # 如果不是两阶段模型，则创建查询位置嵌入层
        if not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)

        # 创建编码器和解码器
        self.encoder = DetaEncoder(config)
        self.decoder = DetaDecoder(config)

        # 创建特征级别嵌入参数
        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        # 如果是两阶段模型，则创建额外的输出层和归一化层
        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)
            self.pos_trans = nn.Linear(config.d_model * 2, config.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(config.d_model * 2)
            self.pix_trans = nn.Linear(config.d_model, config.d_model)
            self.pix_trans_norm = nn.LayerNorm(config.d_model)
        else:
            # 否则创建参考点线性层
            self.reference_points = nn.Linear(config.d_model, 2)

        # 设置两阶段模型相关的配置参数
        self.assign_first_stage = config.assign_first_stage
        self.two_stage_num_proposals = config.two_stage_num_proposals

        # 执行初始化后的其他操作
        self.post_init()
    # 从DeformableDetrModel类中复制的方法，返回编码器(encoder)模型
    def get_encoder(self):
        return self.encoder

    # 从DeformableDetrModel类中复制的方法，返回解码器(decoder)模型
    def get_decoder(self):
        return self.decoder

    # 冻结骨干(backbone)模型的参数，使其不需要梯度更新
    def freeze_backbone(self):
        for name, param in self.backbone.model.named_parameters():
            param.requires_grad_(False)

    # 解冻骨干(backbone)模型的参数，使其需要梯度更新
    def unfreeze_backbone(self):
        for name, param in self.backbone.model.named_parameters():
            param.requires_grad_(True)

    # 从DeformableDetrModel类中复制的方法，计算特征图的有效比例
    def get_valid_ratio(self, mask, dtype=torch.float32):
        """获取所有特征图的有效比例。"""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_height = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

    # 从DeformableDetrModel类中复制的方法，获取提议(proposals)的位置嵌入
    def get_proposal_pos_embed(self, proposals):
        """获取提议的位置嵌入。"""

        num_pos_feats = self.config.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.int64, device=proposals.device).float()
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # batch_size, num_queries, 4
        proposals = proposals.sigmoid() * scale
        # batch_size, num_queries, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # batch_size, num_queries, 4, 64, 2 -> batch_size, num_queries, 512
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    # 将模型的输入和输出文档字符串添加到forward方法
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
@add_start_docstrings(
    """
    DETA Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
    such as COCO detection.
    """,
    DETA_START_DOCSTRING,
)
class DetaForObjectDetection(DetaPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = [r"bbox_embed\.\d+"]
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrForObjectDetection.__init__ with DeformableDetr->Deta
    def __init__(self, config: DetaConfig):
        super().__init__(config)

        # Deformable DETR encoder-decoder model
        self.model = DetaModel(config)

        # Detection heads on top
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = DetaMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        # Initialize bias for classification head to adjust for prior probability
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value

        # Initialize weights for bounding box regression head
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # Adjust bias for bounding box regression to focus on object presence
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)

        # Configure model components based on whether two-stage training is enabled
        num_pred = (config.decoder_layers + 1) if config.two_stage else config.decoder_layers
        if config.with_box_refine:
            # Clone heads for each decoder layer in two-stage training
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # Connect bounding box embed to decoder for iterative refinement
            self.model.decoder.bbox_embed = self.bbox_embed
        else:
            # Clone heads for each decoder layer in single-stage training
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None
        
        if config.two_stage:
            # Connect class embed to decoder for two-stage training
            self.model.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # Initialize weights and perform final processing
        self.post_init()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 为了使 torchscript 正常运行，因为 torchscript 不支持值非同质的字典，
        # 比如一个字典包含张量和列表。
        # 创建辅助损失列表，每个元素是一个字典包含 "logits" 和 "pred_boxes"
        aux_loss = [
            {"logits": logits, "pred_boxes": pred_boxes}
            for logits, pred_boxes in zip(outputs_class.transpose(0, 1)[:-1], outputs_coord.transpose(0, 1)[:-1])
        ]
        return aux_loss

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
        num_boxes: Number of boxes (or examples) to compute the average loss over.
    """
    # Apply sigmoid activation to convert inputs to probabilities
    inputs = inputs.sigmoid()
    # Flatten the inputs to shape (batch_size, -1)
    inputs = inputs.flatten(1)
    # Compute the numerator for DICE coefficient
    numerator = 2 * (inputs * targets).sum(1)
    # Compute the denominator for DICE coefficient
    denominator = inputs.sum(-1) + targets.sum(-1)
    # Compute the DICE loss as 1 - DICE coefficient
    loss = 1 - (numerator + 1) / (denominator + 1)
    # Compute the average loss over all examples (boxes)
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
        num_boxes: Number of boxes (or examples) to compute the average loss over.
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    # Apply sigmoid activation to convert inputs to probabilities
    prob = inputs.sigmoid()
    # Compute binary cross entropy loss
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # Compute the modulating factor (1 - p_t)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # Compute the focal loss as modified by the modulating factor
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Apply optional alpha balancing factor if alpha is non-negative
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Compute the average loss over all examples (boxes)
    return loss.mean(1).sum() / num_boxes


class DetaLoss(nn.Module):
    """
    This class computes the losses for `DetaForObjectDetection`. The process happens in two steps: 1) we compute
    hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of matched
    ground-truth / prediction (supervised class and box).

    Args:
        matcher (`DetaHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """
    pass
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
        super().__init__()
        self.matcher = matcher  # 初始化匹配器对象
        self.num_classes = num_classes  # 设置类别数
        self.focal_alpha = focal_alpha  # 设置焦点损失的 alpha 参数
        self.losses = losses  # 损失函数对象
        self.assign_first_stage = assign_first_stage  # 是否在第一阶段分配
        self.assign_second_stage = assign_second_stage  # 是否在第二阶段分配

        if self.assign_first_stage:
            self.stg1_assigner = DetaStage1Assigner()  # 如果在第一阶段分配，创建第一阶段分配器对象
        if self.assign_second_stage:
            self.stg2_assigner = DetaStage2Assigner(num_queries)  # 如果在第二阶段分配，创建第二阶段分配器对象

    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_labels
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]  # 获取输出中的逻辑回归 logits

        idx = self._get_source_permutation_idx(indices)  # 根据索引获取源排列索引
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])  # 从目标中获取类别标签
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )  # 创建与 logits 相同大小的目标类别张量
        target_classes[idx] = target_classes_o  # 将目标类别分配给对应的索引位置

        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )  # 创建一个独热编码的目标类别张量，多出的一维用于处理背景类别
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)  # 使用目标类别填充独热编码张量

        target_classes_onehot = target_classes_onehot[:, :, :-1]  # 移除多余的背景类别维度
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]  # 计算二元焦点损失
        )
        losses = {"loss_ce": loss_ce}  # 损失函数为交叉熵损失

        return losses

    @torch.no_grad()
    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_cardinality
    # 计算基数误差，即预测的非空框数量的绝对误差
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        logits = outputs["logits"]  # 获取输出中的逻辑张量
        device = logits.device  # 获取逻辑张量所在的设备
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 统计预测中不是“无物体”类别（最后一个类别）的数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 使用 L1 损失计算基数误差
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}  # 将基数误差保存在损失字典中
        return losses

    # 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_boxes 复制而来
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)  # 获取源排列的索引
        source_boxes = outputs["pred_boxes"][idx]  # 获取预测框的源框
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # 使用 L1 损失计算框的损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes  # 将边界框损失保存在损失字典中

        # 计算广义 IoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes  # 将 GIoU 损失保存在损失字典中
        return losses

    # 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss._get_source_permutation_idx 复制而来
    def _get_source_permutation_idx(self, indices):
        # 根据 indices 对预测进行排列
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss._get_target_permutation_idx 复制而来
    def _get_target_permutation_idx(self, indices):
        # 根据 indices 对目标进行排列
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    # 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.get_loss 复制而来
    # 定义一个方法，用于根据指定的损失类型计算损失值
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        # 定义损失类型和对应的损失函数的映射关系
        loss_map = {
            "labels": self.loss_labels,        # 损失类型为标签时使用 self.loss_labels 方法计算损失
            "cardinality": self.loss_cardinality,  # 损失类型为基数时使用 self.loss_cardinality 方法计算损失
            "boxes": self.loss_boxes,          # 损失类型为框坐标时使用 self.loss_boxes 方法计算损失
        }
        # 如果指定的损失类型不在预定义的映射关系中，则抛出数值错误
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        # 返回根据损失类型映射得到的损失值
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
        # Filter out auxiliary and encoder outputs from the main outputs dictionary
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in ("auxiliary_outputs", "enc_outputs")}

        # Determine which function to use for matching outputs to targets based on `assign_second_stage` flag
        if self.assign_second_stage:
            indices = self.stg2_assigner(outputs_without_aux, targets)
        else:
            indices = self.matcher(outputs_without_aux, targets)

        # Compute the total number of target boxes across all samples for normalization
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Check and adjust `num_boxes` based on distributed training setup
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute losses for all specified loss functions
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Handle auxiliary losses if present in outputs
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                if not self.assign_second_stage:
                    indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Handle encoder outputs if present in outputs
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

        # Return computed losses dictionary
        return losses
# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class DetaMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # 定义隐藏层的维度列表，从输入维度到输出维度，构建多层线性变换
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # 逐层进行前向传播，使用ReLU作为激活函数，最后一层不使用激活函数
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrHungarianMatcher with DeformableDetr->Deta
class DetaHungarianMatcher(nn.Module):
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
        # 检查是否安装了SciPy库，这是后端计算所需
        requires_backends(self, ["scipy"])

        # 设置分类错误、边界框坐标L1误差和边界框giou损失的权重
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # 如果三种损失都为0，则抛出值错误异常
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
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # 将分类 logits 展平并应用 sigmoid 函数，得到概率值 [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # 将预测的框坐标展平 [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])  # 连接所有目标的类标签 [sum(num_target_boxes)]
        target_bbox = torch.cat([v["boxes"] for v in targets])  # 连接所有目标的框坐标 [sum(num_target_boxes), 4]

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())  # 计算负分类损失
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())  # 计算正分类损失
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]  # 根据目标类标签选择对应的分类损失

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)  # 计算框坐标之间的 L1 损失

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))  # 计算框之间的 giou 损失

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost  # 组合成最终的损失矩阵
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()  # 调整形状并转移到 CPU

        sizes = [len(v["boxes"]) for v in targets]  # 获取每个目标的框数量
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]  # 使用线性求和分配算法求解最优匹配

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    """
    Protects from numerical overflows in multiplications by upcasting to the equivalent higher type.
    
    Args:
        t (`torch.Tensor`): The input tensor to be upcasted.
    
    Returns:
        `torch.Tensor`: The upcasted tensor.
    """
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
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
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    """
    Computes the IoU (Intersection over Union) between two sets of bounding boxes.

    Args:
        boxes1 (`torch.Tensor`): Bounding boxes in format (x1, y1, x2, y2).
        boxes2 (`torch.Tensor`): Bounding boxes in format (x1, y1, x2, y2).

    Returns:
        `torch.Tensor`: A tensor containing IoU values for each pair of boxes.
    """
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
    Computes the Generalized IoU (GIoU) between two sets of bounding boxes.

    Args:
        boxes1 (`torch.Tensor`): Bounding boxes in format (x1, y1, x2, y2).
        boxes2 (`torch.Tensor`): Bounding boxes in format (x1, y1, x2, y2).

    Returns:
        `torch.Tensor`: A [N, M] pairwise matrix containing GIoU values.
    """
    # Check for degenerate boxes to prevent NaN results
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# from https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/layers/wrappers.py#L100
def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript. because of
    https://github.com/pytorch/pytorch/issues/38718
    """
    # 如果当前处于 Torch 脚本模式
    if torch.jit.is_scripting():
        # 检查张量 x 的维度是否为 0
        if x.dim() == 0:
            # 如果是，将其扩展为一维张量，然后获取非零元素的索引
            return x.unsqueeze(0).nonzero().unbind(1)
        # 如果 x 的维度不为 0，直接获取非零元素的索引
        return x.nonzero().unbind(1)
    # 如果不处于 Torch 脚本模式
    else:
        # 直接返回张量 x 的非零元素的索引，返回结果为元组形式
        return x.nonzero(as_tuple=True)
# from https://github.com/facebookresearch/detectron2/blob/9921a2caa585d4fa66c4b534b6fab6e74d89b582/detectron2/modeling/matcher.py#L9
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
        # 将阈值列表复制到新变量 thresholds 中
        thresholds = thresholds[:]
        # 如果第一个阈值小于 0，则抛出 ValueError 异常
        if thresholds[0] < 0:
            raise ValueError("Thresholds should be positive")
        # 在 thresholds 的开头和结尾分别插入负无穷和正无穷
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        # 检查 thresholds 列表是否按照升序排列
        if not all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])):
            raise ValueError("Thresholds should be sorted.")
        # 检查 labels 列表中的所有元素是否都属于 {-1, 0, 1} 这三个值
        if not all(l in [-1, 0, 1] for l in labels):
            raise ValueError("All labels should be either -1, 0 or 1")
        # 检查 labels 列表的长度是否与 thresholds 列表长度减 1 相等
        if len(labels) != len(thresholds) - 1:
            raise ValueError("Number of labels should be equal to number of thresholds - 1")
        # 初始化对象的属性
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches
    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted elements. All elements must be >= 0
                (due to the use of `torch.nonzero` for selecting indices in `set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2  # 确保输入的 match_quality_matrix 是二维张量

        if match_quality_matrix.numel() == 0:
            # 创建一个全零的张量作为默认匹配结果
            default_matches = match_quality_matrix.new_full((match_quality_matrix.size(1),), 0, dtype=torch.int64)
            # 当没有 ground-truth 盒子存在时，我们定义 IOU = 0，因此将标签设置为 self.labels[0]，
            # 通常默认为背景类别 0；也可以选择忽略，设置 labels=[-1,0,-1,1] 并设置适当的阈值
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)  # 确保所有元素都大于等于 0

        # 对每个预测元素，选择与其IOU最大的 ground-truth 元素作为匹配
        matched_vals, matches = match_quality_matrix.max(dim=0)

        # 创建一个全一的标签张量，初始化为真正例
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        # 根据阈值和标签设置规则，将匹配标签调整为正确的预测标签
        for l, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        # 如果允许低质量匹配，调用函数设置低质量匹配的标签
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels
        # 对于每一个ground-truth (gt)，找到与其具有最高质量匹配的预测
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # 找到可用的最高质量匹配，即使质量较低，包括平局情况。
        # 注意，由于使用了 `torch.nonzero`，匹配质量必须是正数。
        _, pred_inds_with_highest_quality = nonzero_tuple(match_quality_matrix == highest_quality_foreach_gt[:, None])
        # 如果一个anchor仅因与gt_A的低质量匹配而被标记为正样本，
        # 但它与gt_B有更大的重叠，其匹配索引仍将是gt_B。
        # 这遵循Detectron中的实现，并且已经证明没有显著影响。
        match_labels[pred_inds_with_highest_quality] = 1
# 从torch中导入张量类型
import torch

# 定义函数subsmaple_labels，用于对标签进行子采样
def subsample_labels(labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int):
    """
    Return `num_samples` (or fewer, if not enough found) random samples from `labels` which is a mixture of positives &
    negatives. It will try to return as many positives as possible without exceeding `positive_fraction * num_samples`,
    and then try to fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number of negatives sampled is
            `min(num_negatives, num_samples - num_positives_sampled)`. In order words, if there are not enough
            positives, the sample is filled with negatives. If there are also not enough negatives, then as many
            elements are sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    # 找出正样本的索引，即标签值不为-1且不为背景标签的位置
    positive = torch.nonzero((labels != -1) & (labels != bg_label)).squeeze(1)
    # 找出负样本的索引，即标签为背景标签的位置
    negative = torch.nonzero(labels == bg_label).squeeze(1)

    # 计算需要采样的正样本数量，并保护不超过正样本总数
    num_pos = int(num_samples * positive_fraction)
    num_pos = min(positive.numel(), num_pos)

    # 计算需要采样的负样本数量，并保护不超过负样本总数
    num_neg = num_samples - num_pos
    num_neg = min(negative.numel(), num_neg)

    # 随机选择正样本和负样本的索引
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    # 根据随机选择的索引获取正样本和负样本的最终索引
    pos_idx = positive[perm1]
    neg_idx = negative[perm2]

    return pos_idx, neg_idx


# 定义函数sample_topk_per_gt，用于对每个真实框(gt)进行top-k匹配采样
def sample_topk_per_gt(pr_inds, gt_inds, iou, k):
    if len(gt_inds) == 0:
        return pr_inds, gt_inds

    # 找出每个真实框(gt)的top-k匹配
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    scores, pr_inds2 = iou[gt_inds2].topk(k, dim=1)
    gt_inds2 = gt_inds2[:, None].repeat(1, k)

    # 根据每个真实框的匹配数量过滤top-k匹配结果
    pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2)])
    gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])

    return pr_inds3, gt_inds3
    # 定义一个名为 DetaStage2Assigner 的神经网络模块，用于第二阶段的分配任务
    class DetaStage2Assigner(nn.Module):
        def __init__(self, num_queries, max_k=4):
            super().__init__()
            # 设置正样本比例为 0.25
            self.positive_fraction = 0.25
            # 设置背景标签为 400，大于91用于稍后过滤
            self.bg_label = 400  
            # 每张图像的每个批次的大小为 num_queries
            self.batch_size_per_image = num_queries
            # 创建一个 DetaMatcher 对象，用于提议与真实数据匹配
            self.proposal_matcher = DetaMatcher(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True)
            # 最大 K 值设定为 max_k
            self.k = max_k

        def _sample_proposals(self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor):
            """
            根据 N 个提议与 M 个真实数据的匹配情况，采样提议并设置它们的分类标签。

            Args:
                matched_idxs (Tensor): 长度为 N 的向量，每个元素是每个提议最佳匹配的真实数据索引，取值范围为 [0, M)。
                matched_labels (Tensor): 长度为 N 的向量，每个元素是提议的分类标签（来自 cfg.MODEL.ROI_HEADS.IOU_LABELS）。
                gt_classes (Tensor): 长度为 M 的向量，每个元素是真实数据的类别。

            Returns:
                Tensor: 采样提议的索引向量，每个元素在 [0, N) 范围内。
                Tensor: 与采样提议对应的分类标签向量，每个元素与采样提议的索引向量一一对应。每个样本被标记为一个类别在 [0, num_classes) 或背景 (num_classes)。
            """
            # 判断是否存在真实数据
            has_gt = gt_classes.numel() > 0
            # 如果存在真实数据
            if has_gt:
                # 根据匹配结果为每个提议获取相应的真实类别
                gt_classes = gt_classes[matched_idxs]
                # 将未匹配的提议（matcher 标记为 0 的标签）标记为背景 (label=num_classes)
                gt_classes[matched_labels == 0] = self.bg_label
                # 将忽略的提议（标签为 -1）标记为 -1
                gt_classes[matched_labels == -1] = -1
            else:
                # 如果不存在真实数据，则将所有提议标记为背景
                gt_classes = torch.zeros_like(matched_idxs) + self.bg_label

            # 从 gt_classes 中采样前景和背景索引
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                gt_classes, self.batch_size_per_image, self.positive_fraction, self.bg_label
            )

            # 将前景和背景索引连接起来作为最终采样的索引
            sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
            # 返回最终的采样索引和对应的分类标签
            return sampled_idxs, gt_classes[sampled_idxs]
    # 定义前向传播方法，计算目标检测的损失
    def forward(self, outputs, targets, return_cost_matrix=False):
        # COCO 数据集的类别编号范围为 1 到 90，模型设置 num_classes=91 并应用 sigmoid 函数。

        # 获取批量大小
        bs = len(targets)
        # 初始化空列表，用于存储匹配的索引和 IoU 值
        indices = []
        ious = []
        # 遍历每个批次中的目标
        for b in range(bs):
            # 计算预测框和目标框之间的 IoU，并转换为中心点到角点的格式
            iou, _ = box_iou(
                center_to_corners_format(targets[b]["boxes"]),
                center_to_corners_format(outputs["init_reference"][b].detach()),
            )
            # 使用 IoU 值进行匹配，得到匹配的索引和标签
            matched_idxs, matched_labels = self.proposal_matcher(
                iou
            )  # proposal_id -> highest_iou_gt_id, proposal_id -> [1 if iou > 0.6, 0 ow]
            # 根据匹配结果，对提议框进行采样，得到采样后的索引和对应的目标类别
            (
                sampled_idxs,
                sampled_gt_classes,
            ) = self._sample_proposals(  # list of sampled proposal_ids, sampled_id -> [0, num_classes)+[bg_label]
                matched_idxs, matched_labels, targets[b]["class_labels"]
            )
            # 筛选出正样本提议框的索引和对应的正样本目标框的索引
            pos_pr_inds = sampled_idxs[sampled_gt_classes != self.bg_label]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            # 后处理正样本索引，可能包括降采样等操作
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            # 将正样本索引和 IoU 存入列表
            indices.append((pos_pr_inds, pos_gt_inds))
            ious.append(iou)
        # 如果需要返回损失矩阵，则返回索引和 IoU 值
        if return_cost_matrix:
            return indices, ious
        # 否则仅返回索引
        return indices

    # 后处理索引方法，对给定的提议框索引、目标框索引和 IoU 进行处理
    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)
# 从 https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/modeling/proposal_generator/rpn.py#L181 修改而来

class DetaStage1Assigner(nn.Module):
    def __init__(self, t_low=0.3, t_high=0.7, max_k=4):
        super().__init__()
        # 初始化正样本的比例和每张图像的每批样本数
        self.positive_fraction = 0.5
        self.batch_size_per_image = 256
        # 设置最大匹配数和IoU阈值的下限和上限
        self.k = max_k
        self.t_low = t_low
        self.t_high = t_high
        # 创建锚框匹配器对象
        self.anchor_matcher = DetaMatcher(
            thresholds=[t_low, t_high], labels=[0, -1, 1], allow_low_quality_matches=True
        )

    def _subsample_labels(self, label):
        """
        随机抽样一部分正负样本，并将标签向量中未包含在抽样中的元素设置为忽略标签(-1)。

        Args:
            labels (Tensor): 包含标签值-1, 0, 1的向量。将在原地被修改并返回。
        """
        # 对正负样本进行抽样
        pos_idx, neg_idx = subsample_labels(label, self.batch_size_per_image, self.positive_fraction, 0)
        # 将标签向量填充为忽略标签(-1)，然后设置正负样本的标签值
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def forward(self, outputs, targets):
        bs = len(targets)
        indices = []
        for b in range(bs):
            # 获取当前图像的锚框和目标框
            anchors = outputs["anchors"][b]
            if len(targets[b]["boxes"]) == 0:
                # 如果当前图像没有目标框，则返回空张量对
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=anchors.device),
                        torch.tensor([], dtype=torch.long, device=anchors.device),
                    )
                )
                continue
            # 计算锚框与目标框之间的IoU
            iou, _ = box_iou(
                center_to_corners_format(targets[b]["boxes"]),
                center_to_corners_format(anchors),
            )
            # 使用锚框匹配器确定锚框与目标框的匹配情况
            matched_idxs, matched_labels = self.anchor_matcher(
                iou
            )  # proposal_id -> highest_iou_gt_id, proposal_id -> [1 if iou > 0.7, 0 if iou < 0.3, -1 ow]
            # 对匹配后的标签进行正负样本抽样
            matched_labels = self._subsample_labels(matched_labels)

            # 获取所有正样本的索引
            all_pr_inds = torch.arange(len(anchors), device=matched_labels.device)
            # 根据正样本的标签获取正样本对应的锚框和目标框的索引
            pos_pr_inds = all_pr_inds[matched_labels == 1]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            # 后处理正样本的索引
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            pos_pr_inds, pos_gt_inds = pos_pr_inds.to(anchors.device), pos_gt_inds.to(anchors.device)
            indices.append((pos_pr_inds, pos_gt_inds))
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        # 对每个目标框保留前k个置信度最高的锚框
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)
```