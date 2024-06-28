# `.\models\table_transformer\modeling_table_transformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可信息，指明版权归属及许可协议
# 除非符合许可协议，否则禁止使用此文件
# 可在以下链接获取 Apache License 2.0 的详细信息：http://www.apache.org/licenses/LICENSE-2.0
#
# 根据适用法律或书面协议约定，本软件是基于“按原样”提供的，没有任何明示或暗示的保证或条件
# 请查看许可协议，了解详细信息

""" PyTorch Table Transformer model."""

# 导入必要的库
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch  # 导入 PyTorch 库
from torch import Tensor, nn  # 导入张量和神经网络模块

# 导入各种辅助函数和模型组件
from ...activations import ACT2FN  # 激活函数映射
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask  # 准备注意力掩码的实用函数
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithCrossAttentions,
    Seq2SeqModelOutput,
)  # 模型输出相关类
from ...modeling_utils import PreTrainedModel  # 预训练模型基类
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_accelerate_available,
    is_scipy_available,
    is_timm_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)  # 各种实用函数和工具

from ...utils.backbone_utils import load_backbone  # 加载骨干网络工具函数
from .configuration_table_transformer import TableTransformerConfig  # 导入 Table Transformer 的配置类


# 如果 SciPy 可用，则导入线性求和分配函数
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 如果 timm 可用，则导入创建模型函数
if is_timm_available():
    from timm import create_model

# 如果 vision 可用，则导入图像转换函数
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

# 如果 accelerate 可用，则导入部分状态和数据减少函数
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 文档中引用的配置信息
_CONFIG_FOR_DOC = "TableTransformerConfig"
# 文档中引用的检查点信息
_CHECKPOINT_FOR_DOC = "microsoft/table-transformer-detection"

# 预训练模型的存档列表
TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/table-transformer-detection",
    # 查看所有 Table Transformer 模型：https://huggingface.co/models?filter=table-transformer
]

# 数据类，用于封装 Table Transformer 解码器的输出
@dataclass
class TableTransformerDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    TABLE_TRANSFORMER 解码器输出的基类。该类在 BaseModelOutputWithCrossAttentions 基础上添加了一个属性，
    即中间解码器激活的堆栈，即每个解码器层的输出，每个输出通过 layernorm 处理。
    在使用辅助解码损失训练模型时特别有用。
    # 定义函数参数及其类型注释，说明函数接受的输入和返回值的数据类型和形状

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态的序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含模型每一层的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`。当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含注意力权重张量，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。这些是经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            元组，包含解码器交叉注意力层的注意力权重张量，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。这些是经过注意力 softmax 后的注意力权重，用于计算交叉注意力头中的加权平均值。
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            中间的解码器激活状态，即每个解码器层的输出，每个都经过 layernorm 处理。
    """

    # intermediate_hidden_states 可选的张量类型，表示中间解码器的隐藏状态，默认为 None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
# 定义一个数据类 TableTransformerModelOutput，继承自 Seq2SeqModelOutput，用于 TABLE_TRANSFORMER 模型的输出结果
@dataclass
class TableTransformerModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the TABLE_TRANSFORMER encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.
    """

    # 可选的中间隐藏状态堆栈，即每个解码器层的输出，经过 layernorm 处理
    intermediate_hidden_states: Optional[torch.FloatTensor] = None


# 定义一个数据类 TableTransformerObjectDetectionOutput，继承自 ModelOutput，用于 TABLE_TRANSFORMER 目标检测模型的输出结果
@dataclass
class TableTransformerObjectDetectionOutput(ModelOutput):
    """
    Output type of [`TableTransformerForObjectDetection`].
    """

    # 可选的损失值张量
    loss: Optional[torch.FloatTensor] = None
    # 可选的损失字典
    loss_dict: Optional[Dict] = None
    # logits，即模型的原始输出
    logits: torch.FloatTensor = None
    # 预测框的张量
    pred_boxes: torch.FloatTensor = None
    # 可选的辅助输出列表，每个元素是一个字典
    auxiliary_outputs: Optional[List[Dict]] = None
    # 最后的隐藏状态张量
    last_hidden_state: Optional[torch.FloatTensor] = None
    # 解码器的隐藏状态元组
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力张量元组
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力张量元组
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后的隐藏状态张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态张量元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力张量元组
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个类 TableTransformerFrozenBatchNorm2d，继承自 nn.Module，用于冻结统计和仿射参数的 BatchNorm2d
class TableTransformerFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        # 注册权重张量，初始化为全 1
        self.register_buffer("weight", torch.ones(n))
        # 注册偏置张量，初始化为全 0
        self.register_buffer("bias", torch.zeros(n))
        # 注册运行时均值张量，初始化为全 0
        self.register_buffer("running_mean", torch.zeros(n))
        # 注册运行时方差张量，初始化为全 1
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 移除状态字典中的 num_batches_tracked 键，防止加载时出错
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类方法加载状态字典
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
    # 定义一个前向传播函数，接受输入张量 x
    def forward(self, x):
        # 将权重张量重塑为形状为 (1, C, 1, 1)，其中 C 是通道数
        weight = self.weight.reshape(1, -1, 1, 1)
        # 将偏置张量重塑为形状为 (1, C, 1, 1)，其中 C 是通道数
        bias = self.bias.reshape(1, -1, 1, 1)
        # 将运行时方差张量重塑为形状为 (1, C, 1, 1)，其中 C 是通道数
        running_var = self.running_var.reshape(1, -1, 1, 1)
        # 将运行时均值张量重塑为形状为 (1, C, 1, 1)，其中 C 是通道数
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        # 定义一个极小值常量 epsilon
        epsilon = 1e-5
        # 计算缩放因子，即 weight 乘以 (running_var + epsilon) 的倒数平方根
        scale = weight * (running_var + epsilon).rsqrt()
        # 根据运行时均值和缩放因子调整偏置
        bias = bias - running_mean * scale
        # 返回经过缩放和偏置调整后的输入张量 x
        return x * scale + bias
# 从 transformers.models.detr.modeling_detr.replace_batch_norm 复制而来，将所有的 torch.nn.BatchNorm2d 替换为 TableTransformerFrozenBatchNorm2d
def replace_batch_norm(model):
    """
    递归地将所有的 `torch.nn.BatchNorm2d` 替换为 `TableTransformerFrozenBatchNorm2d`。

    Args:
        model (torch.nn.Module):
            输入的模型
    """
    # 遍历模型的所有子模块
    for name, module in model.named_children():
        # 如果当前模块是 nn.BatchNorm2d 类型
        if isinstance(module, nn.BatchNorm2d):
            # 创建一个新的 TableTransformerFrozenBatchNorm2d 模块
            new_module = TableTransformerFrozenBatchNorm2d(module.num_features)

            # 如果原始的 BatchNorm 模块的设备不是 "meta"
            if not module.weight.device == torch.device("meta"):
                # 将权重、偏置、运行均值和方差复制到新模块中
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            # 用新模块替换原始模块
            model._modules[name] = new_module

        # 如果当前模块还有子模块，则递归调用 replace_batch_norm
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


# 从 transformers.models.detr.modeling_detr.DetrConvEncoder 复制而来，将 Detr 替换为 TableTransformer
class TableTransformerConvEncoder(nn.Module):
    """
    卷积骨干网络，使用 AutoBackbone API 或 timm 库中的一个模型。

    nn.BatchNorm2d 层被上面定义的 TableTransformerFrozenBatchNorm2d 替换。
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # 根据配置选择使用 timm 的 backbone 还是加载自定义的 backbone
        if config.use_timm_backbone:
            # 如果使用 timm backbone，则确保 timm 被加载
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            # 创建 timm 模型，仅提取特征
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            # 否则加载自定义的 backbone
            backbone = load_backbone(config)

        # 用 frozen batch norm 替换 batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        # 获取骨干网络输出通道数信息
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        # 根据 backbone 模型类型设置是否冻结某些参数
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
        # 将像素值通过模型传递，获取特征图列表
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        # 初始化输出列表
        out = []
        # 遍历特征图列表
        for feature_map in features:
            # 将像素掩码下采样至与对应特征图相同的形状
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            # 将特征图及其对应的掩码添加到输出列表
            out.append((feature_map, mask))
        # 返回输出列表
        return out
# 从transformers.models.detr.modeling_detr.DetrConvModel复制到TableTransformerConvModel，将Detr替换为TableTransformer
class TableTransformerConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder  # 初始化卷积编码器
        self.position_embedding = position_embedding  # 初始化位置嵌入模块

    def forward(self, pixel_values, pixel_mask):
        # 通过骨干网络处理像素值和像素掩码，得到（特征图，像素掩码）元组的列表
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # 执行位置编码
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


# 从transformers.models.detr.modeling_detr.DetrSinePositionEmbedding复制到TableTransformerSinePositionEmbedding，将Detr替换为TableTransformer
class TableTransformerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.temperature = temperature  # 温度参数
        self.normalize = normalize  # 是否进行归一化
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 缩放参数

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")  # 如果未提供像素掩码则抛出错误
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)  # 在y方向上累积和
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)  # 在x方向上累积和
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale  # 归一化y方向上的累积和
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale  # 归一化x方向上的累积和

        dim_t = torch.arange(self.embedding_dim, dtype=torch.int64, device=pixel_values.device).float()  # 创建维度张量
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)  # 计算温度向量

        pos_x = x_embed[:, :, :, None] / dim_t  # 计算x方向位置编码
        pos_y = y_embed[:, :, :, None] / dim_t  # 计算y方向位置编码
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 奇偶数位进行sin和cos变换，然后展开
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 奇偶数位进行sin和cos变换，然后展开
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 连接并转置位置编码

        return pos


# 从transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding复制到TableTransformerLearnedPositionEmbedding，将Detr替换为TableTransformer
class TableTransformerLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    # 初始化函数，设置嵌入维度并调用父类的初始化方法
    def __init__(self, embedding_dim=256):
        super().__init__()
        # 创建一个包含50个元素的行嵌入对象，每个元素的维度为embedding_dim
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        # 创建一个包含50个元素的列嵌入对象，每个元素的维度为embedding_dim
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    # 前向传播函数，接收像素值和可选的像素掩码作为输入
    def forward(self, pixel_values, pixel_mask=None):
        # 获取像素值的高度和宽度
        height, width = pixel_values.shape[-2:]
        # 在设备上生成从0到width-1的张量，用于列嵌入查询
        width_values = torch.arange(width, device=pixel_values.device)
        # 在设备上生成从0到height-1的张量，用于行嵌入查询
        height_values = torch.arange(height, device=pixel_values.device)
        # 查询列嵌入，x_emb的形状为[width, embedding_dim]
        x_emb = self.column_embeddings(width_values)
        # 查询行嵌入，y_emb的形状为[height, embedding_dim]
        y_emb = self.row_embeddings(height_values)
        # 创建位置编码张量pos，将列嵌入和行嵌入连接起来，形状为[height, width, 2*embedding_dim]
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # 将pos张量的维度重新排列为[2*embedding_dim, height, width]
        pos = pos.permute(2, 0, 1)
        # 在第0维度上添加一个维度，形状变为[1, 2*embedding_dim, height, width]
        pos = pos.unsqueeze(0)
        # 将pos张量在第0维度上复制pixel_values.shape[0]次，形状变为[pixel_values.shape[0], 2*embedding_dim, height, width]
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # 返回位置编码张量pos作为模型的输出
        return pos
# 从transformers.models.detr.modeling_detr.build_position_encoding复制而来，将Detr->TableTransformer
def build_position_encoding(config):
    # 根据配置计算位置编码的步数
    n_steps = config.d_model // 2
    # 如果位置嵌入类型是"sine"
    if config.position_embedding_type == "sine":
        # TODO 找到更好的方式暴露其他参数
        # 使用正弦位置嵌入初始化位置嵌入对象
        position_embedding = TableTransformerSinePositionEmbedding(n_steps, normalize=True)
    # 如果位置嵌入类型是"learned"
    elif config.position_embedding_type == "learned":
        # 使用学习得到的位置嵌入初始化位置嵌入对象
        position_embedding = TableTransformerLearnedPositionEmbedding(n_steps)
    else:
        # 抛出数值错误，指明不支持的位置嵌入类型
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


# 从transformers.models.detr.modeling_detr.DetrAttention复制而来，将DETR->TABLE_TRANSFORMER,Detr->TableTransformer
class TableTransformerAttention(nn.Module):
    """
    'Attention Is All You Need' 论文中的多头注意力机制。

    在这里，我们为查询和键添加位置嵌入（如TABLE_TRANSFORMER论文中所解释的）。
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
        # 如果embed_dim不能被num_heads整除，抛出数值错误
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: "
                f"{num_heads})."
            )
        # 缩放因子为头维度的倒数平方根
        self.scaling = self.head_dim**-0.5

        # 线性映射函数，用于变换查询、键和值
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 将张量重塑为适合多头注意力机制的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 在给定位置嵌入的情况下，添加位置嵌入到张量中
    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)

        # 如果有未预期的参数，则抛出数值错误
        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        # 如果同时指定了position_embeddings和object_queries，则抛出数值错误
        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        # 如果使用了position_embeddings，则发出警告，建议使用object_queries
        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        # 如果object_queries为None，则直接返回张量；否则返回张量加上object_queries
        return tensor if object_queries is None else tensor + object_queries
    # 定义一个前向传播方法，用于模型推断或训练过程中的向前计算
    def forward(
        self,
        # 隐藏状态作为输入，通常是模型中的中间表示
        hidden_states: torch.Tensor,
        # 注意力掩码，用于指定哪些位置的输入需要注意力处理
        attention_mask: Optional[torch.Tensor] = None,
        # 目标查询，用于指定模型关注的特定目标或查询信息
        object_queries: Optional[torch.Tensor] = None,
        # 键值状态，用于注意力机制中的键值对应
        key_value_states: Optional[torch.Tensor] = None,
        # 空间位置嵌入，可能是与空间相关的位置信息的嵌入表示
        spatial_position_embeddings: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: bool = False,
        # 其他可选的关键字参数，传递给函数的额外参数
        **kwargs,
class TableTransformerEncoderLayer(nn.Module):
    # 使用 TableTransformerConfig 配置初始化编码器层
    # 从 transformers.models.detr.modeling_detr.DetrEncoderLayer.__init__ 复制而来，将 Detr 替换为 TableTransformer
    def __init__(self, config: TableTransformerConfig):
        super().__init__()
        # 设定嵌入维度为模型配置中的 d_model
        self.embed_dim = config.d_model
        # 初始化自注意力层，使用 TableTransformerAttention 类
        self.self_attn = TableTransformerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 初始化自注意力层后的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设定 dropout 率
        self.dropout = config.dropout
        # 激活函数根据配置选择对应的函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 激活函数的 dropout 率
        self.activation_dropout = config.activation_dropout
        # 第一个全连接层，将嵌入维度映射到编码器中的前馈网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 第二个全连接层，将前馈网络维度映射回嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，接受隐藏状态、注意力掩码以及可能的对象查询作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
                输入到层的隐藏状态张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
                注意力掩码张量，大小为 `(batch, 1, target_len, source_len)`，其中填充元素由非常大的负值表示。
            object_queries (`torch.FloatTensor`, *optional*): object queries, to be added to hidden_states.
                目标查询张量，可选项，将添加到隐藏状态中。
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
                是否返回所有注意力层的注意力张量。详细信息请参阅返回的张量中的 `attentions` 部分。

        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 从 transformers.models.detr.modeling_detr.DetrDecoderLayer.__init__ 复制而来，用于定义 TableTransformerDecoderLayer 类
class TableTransformerDecoderLayer(nn.Module):
    # 构造函数，初始化 TableTransformerDecoderLayer 类
    def __init__(self, config: TableTransformerConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 自注意力机制，用于学习输入序列内部的关系
        self.self_attn = TableTransformerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 层归一化，用于标准化自注意力输出
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 编码器-解码器注意力机制，用于学习输入序列与编码器隐藏状态之间的关系
        self.encoder_attn = TableTransformerAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 前馈神经网络的第一层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 前馈神经网络的第二层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        
        # 最终的层归一化，用于标准化前馈神经网络的输出
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，定义了数据从输入到输出的流动过程
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # 略
        pass

# 从 transformers.models.detr.modeling_detr.DetrClassificationHead 复制而来，用于定义 TableTransformerClassificationHead 类
class TableTransformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # 构造函数，初始化 TableTransformerClassificationHead 类
    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        # 密集连接层，将输入维度映射到内部维度
        self.dense = nn.Linear(input_dim, inner_dim)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 输出投影层，将内部维度映射到类别数量
        self.out_proj = nn.Linear(inner_dim, num_classes)

    # 前向传播函数，定义了数据从输入到输出的流动过程
    def forward(self, hidden_states: torch.Tensor):
        # Dropout 操作，随机置零部分输入数据，防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 密集连接层，将输入数据映射到内部维度
        hidden_states = self.dense(hidden_states)
        # Tanh 激活函数，增加非线性特性
        hidden_states = torch.tanh(hidden_states)
        # Dropout 操作，再次防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 输出投影层，将映射后的数据映射到类别数量
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# TableTransformerPreTrainedModel 类，用于预训练模型的基类
class TableTransformerPreTrainedModel(PreTrainedModel):
    # 指定配置类
    config_class = TableTransformerConfig
    # 基础模型前缀
    base_model_prefix = "model"
    # 主输入名称，通常是像素值
    main_input_name = "pixel_values"
    # 不拆分的模块列表，用于标记不应拆分的模块名称
    _no_split_modules = [
        r"TableTransformerConvEncoder",
        r"TableTransformerEncoderLayer",
        r"TableTransformerDecoderLayer",
    ]
    # 初始化模型的权重，使用给定的标准差
    def _init_weights(self, module):
        std = self.config.init_std  # 从配置中获取初始化的标准差

        # 如果模块是TableTransformerLearnedPositionEmbedding类型，对其行和列的嵌入进行均匀分布初始化
        if isinstance(module, TableTransformerLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)  # 均匀分布初始化行嵌入权重
            nn.init.uniform_(module.column_embeddings.weight)  # 均匀分布初始化列嵌入权重

        # 如果模块是线性层、二维卷积层或批归一化层，使用正态分布初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 与TensorFlow版本稍有不同，PyTorch使用正态分布而不是截断正态分布进行初始化
            # 参考：https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)  # 使用正态分布初始化权重
            if module.bias is not None:
                module.bias.data.zero_()  # 如果有偏置项，初始化为零向量

        # 如果模块是嵌入层，使用正态分布初始化权重
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)  # 使用正态分布初始化权重
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 如果有填充索引，将填充索引位置的权重初始化为零向量
# 表格转换器的输入文档字符串，用于说明模型的输入格式和参数
TABLE_TRANSFORMER_INPUTS_DOCSTRING = r"""
    # Args: 函数的参数说明开始
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        # 像素值。默认情况下将忽略填充部分。
        # 使用 `DetrImageProcessor` 可以获取像素值。详见 [`DetrImageProcessor.__call__`]。
        
    pixel_mask (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional*):
        # 用于避免在填充像素值上执行注意力的掩码。
        # 掩码值在 `[0, 1]` 之间：
        # - 1 表示真实像素（即**未遮蔽**），
        # - 0 表示填充像素（即**已遮蔽**）。
        # [What are attention masks?](../glossary#attention-mask)
        
    decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
        # 默认情况下不使用。可用于屏蔽对象查询。
        
    encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
        # 元组包含 (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
        # `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*optional*）是编码器最后一层的隐藏状态的序列。在解码器的交叉注意力中使用。
        
    inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        # 可选，而不是传递扁平化特征图（骨干网络输出 + 投影层的输出），可以直接传递图像的扁平化表示。
        
    decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
        # 可选，而不是使用零张量初始化查询，可以直接传递嵌入表示。
        
    output_attentions (`bool`, *optional*):
        # 是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 以获取更多细节。
        
    output_hidden_states (`bool`, *optional*):
        # 是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states` 以获取更多细节。
        
    return_dict (`bool`, *optional*):
        # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
@add_start_docstrings(
    """
    The bare Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    TABLE_TRANSFORMER_START_DOCSTRING,
)
"""
# 从TableTransformerPreTrainedModel类继承，并重写__init__方法，初始化TableTransformerModel对象
class TableTransformerModel(TableTransformerPreTrainedModel):
    # 从transformers.models.detr.modeling_detr.DetrModel.__init__复制而来，将Detr替换为TableTransformer
    def __init__(self, config: TableTransformerConfig):
        # 调用父类的构造方法初始化模型
        super().__init__(config)

        # 创建骨干网络(backbone)和位置编码
        backbone = TableTransformerConvEncoder(config)
        object_queries = build_position_encoding(config)
        self.backbone = TableTransformerConvModel(backbone, object_queries)

        # 创建投影层，将骨干网络的输出进行卷积投影到config.d_model维度
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)

        # 创建查询位置嵌入层，根据config.num_queries和config.d_model创建一个嵌入层
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        # 创建编码器和解码器
        self.encoder = TableTransformerEncoder(config)
        self.decoder = TableTransformerDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 冻结骨干网络的参数，使其在反向传播中不更新
    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    # 解除骨干网络参数的冻结，使其在反向传播中更新
    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    # 重写forward方法，定义模型的前向传播逻辑，接收多个输入参数
    @add_start_docstrings_to_model_forward(TABLE_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TableTransformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Table Transformer Model（由骨干网络和编码器-解码器Transformer组成），顶部带有用于例如COCO检测的对象检测头部。
        """
        # 实现前向传播逻辑，具体实现由具体的Table Transformer模型定义
        pass
    def __init__(self, config: TableTransformerConfig):
        # 调用父类的构造函数，初始化对象
        super().__init__(config)

        # 创建表格转换器模型
        self.model = TableTransformerModel(config)

        # 创建对象检测头部的类别分类器
        self.class_labels_classifier = nn.Linear(
            config.d_model, config.num_labels + 1
        )  # 添加一个类别用于表示“没有对象”

        # 创建表格转换器的边界框预测器
        self.bbox_predictor = TableTransformerMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        # 初始化权重并进行最终处理
        self.post_init()

    @torch.jit.unused
    # 从 transformers.models.detr.modeling_detr.DetrForObjectDetection._set_aux_loss 复制而来
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是为了使 torchscript 正常工作的一种解决方法，因为 torchscript 不支持具有非同构值的字典，
        # 例如同时包含张量和列表的字典。
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(TABLE_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TableTransformerObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[Dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# Copied from transformers.models.detr.modeling_detr.dice_loss
# 计算 DICE 损失，类似于面向掩码的广义 IOU

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
    # 对预测进行 sigmoid 激活，确保在 (0, 1) 范围内
    inputs = inputs.sigmoid()
    # 将输入展平为二维（batch_size, -1）
    inputs = inputs.flatten(1)
    # 计算 DICE 损失的分子部分
    numerator = 2 * (inputs * targets).sum(1)
    # 计算 DICE 损失的分母部分
    denominator = inputs.sum(-1) + targets.sum(-1)
    # 计算最终的 DICE 损失
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 返回平均损失除以框的数量
    return loss.sum() / num_boxes


# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
# 计算用于密集检测的 sigmoid focal loss

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
    # 对输入进行 sigmoid 激活
    prob = inputs.sigmoid()
    # 计算二元交叉熵损失
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 计算 p_t，用于调节损失
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # 计算最终损失，加入调节因子 gamma
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 如果 alpha 大于等于 0，则应用平衡因子 alpha_t
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 返回每个样本的平均损失除以框的数量
    return loss.mean(1).sum() / num_boxes


# Copied from transformers.models.detr.modeling_detr.DetrLoss with Detr->TableTransformer,detr->table_transformer
# 计算用于 TableTransformerForObjectDetection/TableTransformerForSegmentation 的损失

class TableTransformerLoss(nn.Module):
    """
    This class computes the losses for TableTransformerForObjectDetection/TableTransformerForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in table_transformer.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    """
    # 此类计算 TableTransformerForObjectDetection/TableTransformerForSegmentation 的损失。过程分为两步：1）
    # 计算真实框与模型输出之间的匈牙利分配 2）监督每对匹配的真实框/预测（监督类别和框）

    def __init__(self):
        super().__init__()

    def forward(self):
        # 此处应该实现具体的损失计算逻辑，但代码未完整给出
        pass
    """
    Module for computing losses in table transformer model.

    This module defines methods for calculating classification and cardinality errors for
    object detection tasks using a transformer-based approach.

    Args:
        matcher (`TableTransformerHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__()
        self.matcher = matcher  # Initialize the matcher module for matching targets and proposals
        self.num_classes = num_classes  # Set the number of object categories
        self.eos_coef = eos_coef  # Set the coefficient for the no-object category weight
        self.losses = losses  # Store the list of losses to be used
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)  # Register the empty weight tensor

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]  # Extract logits from model outputs

        idx = self._get_source_permutation_idx(indices)  # Get indices for permutation
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])  # Gather target class labels
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o  # Assign target class labels according to permutation indices

        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}  # Compute and store classification loss

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]  # Extract logits from model outputs
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)  # Calculate target lengths
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())  # Compute cardinality error
        losses = {"cardinality_error": card_err}  # Store cardinality error

        return losses
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # 如果输出中没有预测的边界框，则抛出关键错误信息
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        
        # 根据索引获取源排列的索引
        idx = self._get_source_permutation_idx(indices)
        
        # 从输出中获取预测的边界框并按照索引排列
        source_boxes = outputs["pred_boxes"][idx]
        
        # 从目标中获取所有目标边界框并连接成一个张量
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算 L1 回归损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        # 将 L1 损失求和并除以边界框的数量，作为边界框损失
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算 GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        # 将 GIoU 损失求和并除以边界框的数量，作为 GIoU 损失
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        # 如果输出中没有预测的掩码，则抛出关键错误信息
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        # 获取源排列索引和目标排列索引
        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        
        # 从输出中获取预测的掩码并按照源排列索引排序
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        
        # 从目标中获取所有目标掩码并转换为嵌套张量
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # 将预测掩码插值到目标尺寸
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        
        losses = {
            # 计算 sigmoid focal loss 作为掩码损失
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            # 计算 dice loss 作为掩码损失
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # 返回批次索引和源索引
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx
    def _get_target_permutation_idx(self, indices):
        # 根据给定的索引重排目标数据的批次索引和目标索引
        # 生成与每个目标对应的批次索引
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        # 生成所有目标的合并索引
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        # 定义损失函数字典
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        # 如果指定的损失不在损失函数字典中，则引发错误
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        # 调用相应的损失函数并返回计算结果
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
        # 过滤掉辅助输出以减少处理量
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # 利用匹配器函数找到输出与目标之间的匹配关系
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点上目标框的平均数，用于归一化
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        # 如果加速库可用，执行以下操作
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                # 通过降维处理目标框数量
                num_boxes = reduce(num_boxes)
                # 获取当前处理器数量
                world_size = PartialState().num_processes
        # 将目标框数量归一化，并确保不低于1
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 如果存在辅助损失，则对每个中间层的输出进行相同的损失计算
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # 中间层的掩码损失计算成本过高，忽略掉
                        continue
                    # 生成特定中间层的损失字典，并加入到总损失字典中
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
# 从 transformers.models.detr.modeling_detr.DetrMLPPredictionHead 复制而来，仅修改了 Detr -> TableTransformer，定义了 TableTransformerMLPPredictionHead 类
class TableTransformerMLPPredictionHead(nn.Module):
    """
    简单的多层感知机（MLP，也称为 FFN），用于预测相对于图像的标准化中心坐标、高度和宽度的类。

    从 https://github.com/facebookresearch/table_transformer/blob/master/models/table_transformer.py 复制而来
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 创建多层线性层组成的列表，用于构建 MLP 网络
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # 遍历 MLP 网络的每一层并应用激活函数 ReLU（除了最后一层）
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 从 transformers.models.detr.modeling_detr.DetrHungarianMatcher 复制而来，仅修改了 Detr -> TableTransformer
class TableTransformerHungarianMatcher(nn.Module):
    """
    这个类计算网络目标和预测之间的分配。

    由于效率原因，目标不包括 no_object。因此，通常预测比目标多。在这种情况下，我们对最佳预测进行一对一匹配，而其他预测则未匹配（因此视为非对象）。

    Args:
        class_cost:
            在匹配成本中分类错误的相对权重。
        bbox_cost:
            在匹配成本中边界框坐标的 L1 误差的相对权重。
        giou_cost:
            在匹配成本中边界框的 giou 损失的相对权重。
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        # 确保需要的后端库已经安装
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            # 如果匹配器的所有成本都为 0，则引发 ValueError
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
        # 将 logits 展平并进行 softmax，用于计算分类损失
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        
        # 将预测的边界框坐标展平，用于计算边界框回归损失
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # 拼接所有目标的类别标签，用于计算分类损失
        target_ids = torch.cat([v["class_labels"] for v in targets])
        
        # 拼接所有目标的边界框坐标，用于计算边界框回归损失
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # 计算分类损失，使用 1 - proba[target class] 的近似值
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        # 计算边界框之间的 L1 损失
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        # 计算边界框之间的 giou 损失
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        # 组合所有损失成最终的损失矩阵
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # Perform linear sum assignment for each batch element
        # 为每个批次元素执行线性求和分配
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        
        # Return indices as tensors
        # 返回索引作为张量
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # 如果输入张量是浮点类型，则根据需要将其转换为等效更高精度的类型，以防止数值溢出
    if t.is_floating_point():
        # 如果张量类型为 float32 或 float64，则直接返回，否则转换为 float32
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        # 如果张量类型为 int32 或 int64，则直接返回，否则转换为 int32
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，这些边界框由它们的 (x1, y1, x2, y2) 坐标指定。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            要计算面积的边界框。它们应该以 (x1, y1, x2, y2) 格式给出，要求 `0 <= x1 < x2` 和 `0 <= y1 < y2`。

    Returns:
        `torch.FloatTensor`: 包含每个边界框面积的张量。
    """
    # 将输入边界框张量转换为高精度类型，以防止数值溢出
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
    从 https://giou.stanford.edu/ 中获取的广义 IoU。边界框应处于 [x0, y0, x1, y1]（角点）格式。

    Returns:
        `torch.FloatTensor`: 一个 [N, M] 的成对矩阵，其中 N = len(boxes1)，M = len(boxes2)
    """
    # 如果边界框退化将导致无穷大/无穷小结果，则进行早期检查
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 必须在 [x0, y0, x1, y1]（角点）格式内，但得到了 {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 必须在 [x0, y0, x1, y1]（角点）格式内，但得到了 {boxes2}")
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


# Copied from transformers.models.detr.modeling_detr.NestedTensor
# 定义一个名为 NestedTensor 的类，表示嵌套张量对象
class NestedTensor(object):
    # 初始化方法，接受张量列表和一个可选的遮罩张量作为参数
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors  # 初始化对象的张量属性
        self.mask = mask  # 初始化对象的遮罩属性

    # 将嵌套张量对象转移到指定设备的方法
    def to(self, device):
        cast_tensor = self.tensors.to(device)  # 转换张量到指定设备
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)  # 若存在遮罩张量，则也转移到指定设备
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)  # 返回转移后的嵌套张量对象

    # 将嵌套张量对象解构成张量和遮罩张量的方法
    def decompose(self):
        return self.tensors, self.mask  # 返回嵌套张量对象内部的张量和遮罩张量

    # 返回嵌套张量对象的字符串表示形式
    def __repr__(self):
        return str(self.tensors)


# 从张量列表创建嵌套张量对象的方法，来自于 transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:  # 检查列表中第一个张量的维度是否为3
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])  # 计算张量列表中各张量在各维度上的最大尺寸
        batch_shape = [len(tensor_list)] + max_size  # 计算批量张量的形状
        batch_size, num_channels, height, width = batch_shape  # 解包批量张量的形状
        dtype = tensor_list[0].dtype  # 获取列表中第一个张量的数据类型
        device = tensor_list[0].device  # 获取列表中第一个张量所在的设备
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)  # 创建全零张量作为批量张量
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)  # 创建全一遮罩张量
        # 将每个张量复制到批量张量中，并更新遮罩张量
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")  # 抛出错误，仅支持处理三维张量
    return NestedTensor(tensor, mask)  # 返回创建的嵌套张量对象
```