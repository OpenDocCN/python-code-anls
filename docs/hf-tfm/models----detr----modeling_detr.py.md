# `.\models\detr\modeling_detr.py`

```py
# 设置文件编码格式为utf-8
# 版权声明
# 根据Apache许可版本2.0（“许可证”）授权
# 您只能在遵守许可证的情况下使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得以任何方式分发软件
# 根据许可证的规定，软件以"现状"（AS IS）分发，不附带任何保证或条件，无论是明示的还是隐含的
# 请查看特定语言的许可证，了解权限和限制
# PyTorch DETR model 模块

# 导入依赖包
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_timm_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
# 导入模型配置文件
from .configuration_detr import DetrConfig


if is_scipy_available():
    # 导入线性求解库
    from scipy.optimize import linear_sum_assignment

if is_timm_available():
    # 从timm库中导入模型创建函数
    from timm import create_model

if is_vision_available():
    # 从transformers库中导入图像转换函数
    from transformers.image_transforms import center_to_corners_format

# 获取日志记录
logger = logging.get_logger(__name__)

# 模型文档用到的配置
_CONFIG_FOR_DOC = "DetrConfig"
_CHECKPOINT_FOR_DOC = "facebook/detr-resnet-50"

# 预训练模型列表
DETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/detr-resnet-50",
    # 查看所有DETR模型的详细信息 https://huggingface.co/models?filter=detr
]

# DETR解码器输出的基类
@dataclass
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    DETR解码器输出的基类。该类在BaseModelOutputWithCrossAttentions基础上添加了一个属性，
    即可选的中间解码器激活堆栈，即每个解码器层的输出，每个输出都经过了一个layernorm。
    在使用辅助解码损失训练模型时非常有用。
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    # Optional[torch.FloatTensor] type parameter for intermediate decoder activations
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
# 定义包含 DETR 编码-解码模型输出的基类，继承自 Seq2SeqModelOutput
@dataclass
class DetrModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the DETR encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    """
    intermediate_hidden_states: Optional[torch.FloatTensor] = None


# 定义包含 DETR 对象检测输出的模型输出类，继承自 ModelOutput
@dataclass
class DetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`DetrForObjectDetection`].

    """
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义包含 DETR 分割输出的模型输出类，继承自 ModelOutput
@dataclass
class DetrSegmentationOutput(ModelOutput):
    """
    Output type of [`DetrForSegmentation`].

    """
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义冻结的 BatchNorm2d 类，继承自 nn.Module
class DetrFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    # 检查是否有 num_batches_tracked_key，如果有则删除
    num_batches_tracked_key = prefix + "num_batches_tracked"
    if num_batches_tracked_key in state_dict:
        del state_dict[num_batches_tracked_key]

    # 调用父类的_load_from_state_dict方法，加载模型参数
    super()._load_from_state_dict(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    )

def forward(self, x):
    # 将weight、bias、running_var、running_mean重塑为合适的维度
    # 以提高用户友好性
    weight = self.weight.reshape(1, -1, 1, 1)
    bias = self.bias.reshape(1, -1, 1, 1)
    running_var = self.running_var.reshape(1, -1, 1, 1)
    running_mean = self.running_mean.reshape(1, -1, 1, 1)
    epsilon = 1e-5
    # 计算缩放系数scale
    scale = weight * (running_var + epsilon).rsqrt()
    # 根据缩放系数和偏置调整输入数据x
    bias = bias - running_mean * scale
    return x * scale + bias
# 递归替换所有的`torch.nn.BatchNorm2d`层为`DetrFrozenBatchNorm2d`
def replace_batch_norm(model):
    # 遍历模型的每一个子模块
    for name, module in model.named_children():
        # 如果该子模块是`nn.BatchNorm2d`类型
        if isinstance(module, nn.BatchNorm2d):
            # 创建一个`DetrFrozenBatchNorm2d`对象用于替代已有的`nn.BatchNorm2d`层
            new_module = DetrFrozenBatchNorm2d(module.num_features)
            
            # 如果`nn.BatchNorm2d`层不在设备"meta"上，则将其权重、偏置、running mean和running variance数据复制到`DetrFrozenBatchNorm2d`层中
            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            # 将新的`DetrFrozenBatchNorm2d`层替代掉原有的`nn.BatchNorm2d`层
            model._modules[name] = new_module

        # 如果该子模块还有子模块，递归调用`replace_batch_norm`函数
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class DetrConvEncoder(nn.Module):
    """
    使用AutoBackbone API或者timm库创建卷积主干网络。

    让`nn.BatchNorm2d`层被上面定义的`DetrFrozenBatchNorm2d`层替代。
    """

    def __init__(self, config):
        super().__init__()

        # 保存配置信息
        self.config = config

        # 如果使用timm主干网络
        if config.use_timm_backbone:
            # 需要timm库的支持
            requires_backends(self, ["timm"])

            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16

            # 使用timm库创建主干网络
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            # 使用AutoBackbone从配置中创建主干网络
            backbone = AutoBackbone.from_config(config.backbone_config)

        # 用`DetrFrozenBatchNorm2d`替换BatchNorm2d层
        with torch.no_grad():
            replace_batch_norm(backbone)

        # 保存主干网络
        self.model = backbone

        # 获取中间层的通道大小
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        # 获取主干网络类型
        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type

        # 如果主干网络采用的是resnet
        if "resnet" in backbone_model_type:
            # 设置名字为`layer2`,`layer3`和`layer4`的参数不需要梯度更新
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)
    # 前向传播函数，接收像素数值和像素掩码作为输入
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # 如果配置中使用了timm的骨干网络，则将像素数值通过模型传递以获取特征图列表
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        out = []
        # 遍历特征图列表
        for feature_map in features:
            # 将像素掩码下采样以匹配相应特征图的形状
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            # 将特征图和掩码作为元组添加到输出列表中
            out.append((feature_map, mask))
        return out
# 定义一个名为DetrConvModel的类，继承自nn.Module
class DetrConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    # 初始化方法，接收conv_encoder和position_embedding两个参数
    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        # 将传入的conv_encoder和position_embedding分别赋值给当前对象的属性
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    # 前向传播方法，接收pixel_values和pixel_mask两个参数
    def forward(self, pixel_values, pixel_mask):
        # 通过backbone将pixel_values和pixel_mask传递，获取(feature_map, pixel_mask)元组列表
        out = self.conv_encoder(pixel_values, pixel_mask)
        # 初始化一个空列表用于存放位置信息
        pos = []
        # 遍历输出的(feature_map, pixel_mask)元组列表
        for feature_map, mask in out:
            # 对feature_map和mask进行位置编码，将结果转换成feature_map的数据类型后加入pos列表中
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        # 返回out和pos
        return out, pos


# 定义一个名为DetrSinePositionEmbedding的类，继承自nn.Module
class DetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    # 初始化方法，接收embedding_dim、temperature、normalize和scale四个参数
    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        # 将传入的参数赋值给当前对象的属性
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # 前向传播方法，接收pixel_values和pixel_mask两个参数
    def forward(self, pixel_values, pixel_mask):
        # 如果未提供像素掩码，则抛出数值错误
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        # 对像素掩码进行累积和计算得到y方向和x方向的位置编码
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        # 如果标准化为真，则将位置编码除以尺寸并乘以比例
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        # 创建一个tensor用于位置编码
        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        # 计算x和y方向上的位置偏移
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 使用sin和cos函数进行位置编码
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 拼接x和y方向的位置编码，并进行维度变换
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # 返回位置编码
        return pos


# 定义一个名为DetrLearnedPositionEmbedding的类，继承自nn.Module
class DetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    # 初始化方法，接收embedding_dim作为参数
    def __init__(self, embedding_dim=256):
        super().__init__()
        # 使用Embedding层创建行和列的位置编码
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)
    # 定义前向传播函数，计算位置嵌入
    def forward(self, pixel_values, pixel_mask=None):
        # 获取图像的高度和宽度
        height, width = pixel_values.shape[-2:]
        # 生成宽度范围的序列张量
        width_values = torch.arange(width, device=pixel_values.device)
        # 生成高度范围的序列张量
        height_values = torch.arange(height, device=pixel_values.device)
        # 使用宽度值得到列嵌入
        x_emb = self.column_embeddings(width_values)
        # 使用高度值得到行嵌入
        y_emb = self.row_embeddings(height_values)
        # 组合行和列嵌入到一个位置张量中
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # 调整位置张量的维度顺序
        pos = pos.permute(2, 0, 1)
        # 增加一个新的批次维度
        pos = pos.unsqueeze(0)
        # 复制位置张量以匹配输入的批次大小
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # 返回最终的位置嵌入张量
        return pos
def build_position_encoding(config):
    # 根据配置获取模型维度的一半，用于计算步数
    n_steps = config.d_model // 2
    # 根据位置编码类型选择不同的位置编码模块
    if config.position_embedding_type == "sine":
        # TODO: 找到更好的暴露其他参数的方式
        position_embedding = DetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = DetrLearnedPositionEmbedding(n_steps)
    else:
        # 抛出异常，指定的位置编码类型不支持
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


class DetrAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
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
        if self.head_dim * num_heads != self.embed_dim:
            # 抛出异常，确保 embed_dim 必须可以被 num_heads 整除
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        # 缩放系数
        self.scaling = self.head_dim**-0.5

        # 初始化线性投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        # 重塑张量形状以方便处理
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        # 获取位置编码张量
        position_embeddings = kwargs.pop("position_embeddings", None)

        if kwargs:
            # 如果存在其他未知参数，则抛出异常
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            # 如果同时指定了位置编码和对象查询，则抛出异常
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if position_embeddings is not None:
            # 提示位置编码参数即将被移除，请使用对象查询
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        return tensor if object_queries is None else tensor + object_queries

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        spatial_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
class DetrEncoderLayer(nn.Module):
    # 初始化函数，用于创建一个新的DetrEncoderLayer对象
    def __init__(self, config: DetrConfig):
        # 调用父类的初始化函数，确保正确设置继承自父类的属性和方法
        super().__init__()
        # 设定嵌入维度，使用配置中的模型维度作为嵌入维度
        self.embed_dim = config.d_model
        # 创建自注意力机制（self-attention）层，使用DetrAttention类
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,  # 嵌入维度
            num_heads=config.encoder_attention_heads,  # 注意力头数，来自配置
            dropout=config.attention_dropout,  # 注意力层的dropout率，来自配置
        )
        # 创建自注意力机制层后的LayerNorm层，用于归一化输入
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设定dropout率，用于隐藏层之间的丢弃
        self.dropout = config.dropout
        # 设定激活函数，从配置中获取对应的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设定激活函数的dropout率，来自配置
        self.activation_dropout = config.activation_dropout
        # 创建第一个全连接层，连接输入和FFN（Feed Forward Network）维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 创建第二个全连接层，连接FFN和输入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 创建最终的LayerNorm层，用于归一化输出
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    # 前向传播函数，用于执行模型的前向传播过程
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: torch.Tensor,  # 注意力掩码张量
        object_queries: torch.Tensor = None,  # 目标查询张量，默认为None
        output_attentions: bool = False,  # 是否输出注意力张量，默认为False
        **kwargs,  # 其他关键字参数
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            输入到该层的隐藏状态，形状为`(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            注意力掩码的大小为`(batch, 1, target_len, source_len)`，其中填充元素由非常大的负值表示。
            object_queries (`torch.FloatTensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
                对象查询（也称为内容嵌入），要添加到隐藏状态中。
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
                是否返回所有注意力层的注意力张量。具体细节见返回的张量中的`attentions`。

        """
        # 从关键字参数中弹出"position_embeddings"，如果没有则为None
        position_embeddings = kwargs.pop("position_embeddings", None)

        # 如果关键字参数非空，则引发值错误
        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        # 如果位置嵌入和对象查询都不为None，则引发值错误
        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        # 如果位置嵌入不为None，则警告该参数已被弃用，并将对象查询设置为位置嵌入
        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        # 保留隐藏状态为残差连接的输入
        residual = hidden_states
        # 通过self_attn方法得到新的隐藏状态和注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        # 对隐藏状态应用dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接操作
        hidden_states = residual + hidden_states
        # 对隐藏状态应用self-attention层的layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保留隐藏状态为残差连接的输入
        residual = hidden_states
        # 对隐藏状态应用激活函数
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对隐藏状态应用激活函数的dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        # 对隐藏状态应用全连接层1
        hidden_states = self.fc2(hidden_states)
        # 对隐藏状态应用全连接层的dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 残差连接操作
        hidden_states = residual + hidden_states
        # 对隐藏状态应用最终层的layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果处于训练模式且隐藏状态包含无穷大或NaN的值，则进行值的限制
        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 返回输出结果
        outputs = (hidden_states,)

        # 如果需要输出attention权重，则将其加入到输出结果中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义一个类用于实现Detr模型的解码器层
class DetrDecoderLayer(nn.Module):
    # 初始化函数，接受DetrConfig作为参数
    def __init__(self, config: DetrConfig):
        super().__init__()
        # 设置嵌入维度
        self.embed_dim = config.d_model

        # 创建自注意力层
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 创建自注意力层的LayerNorm层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 创建编码器注意力层
        self.encoder_attn = DetrAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 创建全连接层1
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        
        # 创建全连接层2
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        
        # 创建最终的LayerNorm层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        
    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
        
# 定义一个类用于实现Detr模型的分类头
class DetrClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # 初始化函数，接受输入维度、内部维度、类别数量和pooler_dropout参数
    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        # 创建全连接层
        self.dense = nn.Linear(input_dim, inner_dim)
        
        # 创建Dropout层
        self.dropout = nn.Dropout(p=pooler_dropout)
        
        # 创建输出投影层
        self.out_proj = nn.Linear(inner_dim, num_classes)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# 定义一个类用于实现Detr预训练模型
class DetrPreTrainedModel(PreTrainedModel):
    config_class = DetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DetrConvEncoder", r"DetrEncoderLayer", r"DetrDecoderLayer"]
    # 初始化模型参数的函数，根据模块类型进行不同的初始化操作
    def _init_weights(self, module):
        # 获取配置中的标准差和Xavier初始化的标准差
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        # 如果是 DetrMHAttentionMap 模块
        if isinstance(module, DetrMHAttentionMap):
            # 初始化 k_linear 和 q_linear 模块的偏置为零
            nn.init.zeros_(module.k_linear.bias)
            nn.init.zeros_(module.q_linear.bias)
            # 使用 Xavier 均匀分布初始化 k_linear 和 q_linear 模块的权重
            nn.init.xavier_uniform_(module.k_linear.weight, gain=xavier_std)
            nn.init.xavier_uniform_(module.q_linear.weight, gain=xavier_std)
        # 如果是 DetrLearnedPositionEmbedding 模块
        elif isinstance(module, DetrLearnedPositionEmbedding):
            # 使用均匀分布初始化 row_embeddings 和 column_embeddings 模块的权重
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        # 如果是 nn.Linear、nn.Conv2d 或 nn.BatchNorm2d 模块
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 使用正态分布初始化权重，均值为0，标准差为std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 nn.Embedding 模块
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果设置了padding_idx，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# 定义一个用于 DETR 模型文档字符串的常量
DETR_START_DOCSTRING = r"""
    这个模型继承自 [`PreTrainedModel`]。检查超类的文档以了解该库为所有模型实现的通用方法
    （例如下载或保存、调整输入嵌入、修剪头部等）。

    这个模型也是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。
    可以像常规 PyTorch 模块一样使用它，并参考 PyTorch 文档了解所有与一般用法和行为相关的事项。

    参数:
        config ([`DetrConfig`]):
            模型配置类，包含模型的所有参数。用配置文件初始化不会加载与模型关联的权重，只会加载配置。
            请查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

# 定义一个用于 DETR 模型输入文档字符串的常量
DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下将忽略填充。
            # 可以使用 [`AutoImageProcessor`] 获取像素值。有关详细信息，请参阅 [`DetrImageProcessor.__call__`]。

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            # 遮罩，用于避免在填充像素值上执行注意力操作。遮罩值在 `[0, 1]` 之间：

            # - 1 表示真实像素（即**未遮罩**），
            # - 0 表示填充像素（即**已遮罩**）。

            # [什么是注意力遮罩？](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            # 默认情况下不使用。可用于遮罩对象查询。
        
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            # 元组包含 (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            # `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*optional*) 是编码器最后一层的隐藏状态序列。用于解码器的交叉注意力。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选地，可以选择直接传递图像的扁平化表示，而不是传递经过骨干网络和投影层的扁平化特征图输出。

        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            # 可选地，可以选择直接传递嵌入表示，而不是使用零张量初始化查询。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量下的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量下的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 定义 DETR 模型的编码器部分，包含多个自注意力层
class DetrEncoder(DetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`DetrEncoderLayer`].

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for DETR:

    - object_queries are added to the forward pass.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 创建多个编码器层
        self.layers = nn.ModuleList([DetrEncoderLayer(config) for _ in range(config.encoder_layers)])

        # 在原始的 DETR 中，编码器末尾没有使用 layernorm，因为 "normalize_before" 默认设置为 False

        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        object_queries=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
# 定义 DETR 模型的解码器部分，包含多个层
class DetrDecoder(DetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        # 创建多个解码器层
        self.layers = nn.ModuleList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 在 DETR 中，解码器在最后一个解码器层输出后使用 layernorm
        self.layernorm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        object_queries=None,
        query_position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
# 定义 DETR 模型，输出原始隐藏状态而不带特定头部
@add_start_docstrings(
    """
    The bare DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
    any specific head on top.
    """,
    DETR_START_DOCSTRING,
)
class DetrModel(DetrPreTrainedModel):
    # 初始化函数，接受一个DetrConfig对象作为参数
    def __init__(self, config: DetrConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建backbone和位置编码
        backbone = DetrConvEncoder(config)
        object_queries = build_position_encoding(config)
        self.backbone = DetrConvModel(backbone, object_queries)

        # 创建投影层
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)

        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        self.encoder = DetrEncoder(config)
        self.decoder = DetrDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 冻结backbone的参数
    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    # 解冻backbone的参数
    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrModelOutput, config_class=_CONFIG_FOR_DOC)
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
# 定义一个 DETR 模型，包括一个骨干网络和编码器-解码器 Transformer，顶部有目标检测头部，用于诸如 COCO 检测等任务
class DetrForObjectDetection(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 DETR 编码器-解码器模型
        self.model = DetrModel(config)

        # 目标检测头部
        self.class_labels_classifier = nn.Linear(
            config.d_model, config.num_labels + 1
        )  # 我们为“无对象”类添加一个类别
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 https://github.com/facebookresearch/detr/blob/master/models/detr.py 中获取
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是一个解决方案，使 torchscript 满意，因为 torchscript 不支持具有非同质值的字典，例如一个同时具有张量和列表的字典。
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # 前向传播函数
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
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



# 定义一个 DETR 模型，包括一个骨干网络和编码器-解码器 Transformer，顶部有一个分割头部，用于诸如 COCO 全景等任务
class DetrForSegmentation(DetrPreTrainedModel):
    # 初始化函数，接受一个DetrConfig对象作为参数
    def __init__(self, config: DetrConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建目标检测模型
        self.detr = DetrForObjectDetection(config)

        # 创建分割头部
        hidden_size, number_of_heads = config.d_model, config.encoder_attention_heads
        intermediate_channel_sizes = self.detr.model.backbone.conv_encoder.intermediate_channel_sizes

        self.mask_head = DetrMaskHeadSmallConv(
            hidden_size + number_of_heads, intermediate_channel_sizes[::-1][-3:], hidden_size
        )

        self.bbox_attention = DetrMHAttentionMap(
            hidden_size, hidden_size, number_of_heads, dropout=0.0, std=config.init_xavier_std
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrSegmentationOutput, config_class=_CONFIG_FOR_DOC)
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
# 定义一个函数，用于将张量在指定维度上进行扩展
def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

# 从 https://github.com/facebookresearch/detr/blob/master/models/segmentation.py 中引用的类
class DetrMaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm. Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        # 检查隐藏层大小和注意力头的数量是否能被8整除，因为 GroupNorm 的组数设置为8
        if dim % 8 != 0:
            raise ValueError(
                "The hidden_size + number of attention heads must be divisible by 8 as the number of groups in"
                " GroupNorm is set to 8"
            )

        # 定义中间层的维度
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        # 定义卷积层和 GroupNorm 层
        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, inter_dims[1]), inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(min(8, inter_dims[2]), inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(min(8, inter_dims[3]), inter_dims[3])
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        # 定义适配器层
        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        # 初始化所有卷积层的权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        # 将x（投影后的特征图，形状为(batch_size, d_model, heigth/32, width/32)）与bbox_mask（注意力图，形状为(batch_size, n_queries, n_heads, height/32, width/32)）拼接在一起
        # 将投影后的特征图扩展到与注意力图的头数相匹配
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        # 经过第一个全连接层
        x = self.lay1(x)
        # 经过第一个 GroupNorm 层
        x = self.gn1(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)
        
        # 经过第二个全连接层
        x = self.lay2(x)
        # 经过第二个 GroupNorm 层
        x = self.gn2(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)

        # 获取当前的 FPN（Feature Pyramid Network）并通过第一个适配器
        cur_fpn = self.adapter1(fpns[0])
        # 如果当前 FPN 的大小与 x 的大小不匹配，则进行扩展
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 将当前 FPN 与通过插值调整大小后的 x 相加
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 经过第三个全连接层
        x = self.lay3(x)
        # 经过第三个 GroupNorm 层
        x = self.gn3(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)

        # 获取当前的 FPN 并通过第二个适配器
        cur_fpn = self.adapter2(fpns[1])
        # 如果当前 FPN 的大小与 x 的大小不匹配，则进行扩展
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 将当前 FPN 与通过插值调整大小后的 x 相加
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 经过第四个全连接层
        x = self.lay4(x)
        # 经过第四个 GroupNorm 层
        x = self.gn4(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)

        # 获取当前的 FPN 并通过第三个适配器
        cur_fpn = self.adapter3(fpns[2])
        # 如果当前 FPN 的大小与 x 的大小不匹配，则进行扩展
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 将当前 FPN 与通过插值调整大小后的 x 相加
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 经过第五个全连接层
        x = self.lay5(x)
        # 经过第五个 GroupNorm 层
        x = self.gn5(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)

        # 经过输出层
        x = self.out_lay(x)
        return x
class DetrMHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None):
        # 初始化函数，定义了注意力模块的结构和参数
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 定义线性变换层，用于将输入进行线性变换
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        # 计算归一化因子
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        # 前向传播函数，计算注意力权重
        q = self.q_linear(q)
        k = nn.functional.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        queries_per_head = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        keys_per_head = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), torch.finfo(weights.dtype).min)
        weights = nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


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
    # 计算 DICE 损失函数
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


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
    # 计算输入的概率值
    prob = inputs.sigmoid()
    # 使用二元交叉熵损失函数计算损失，不进行汇总
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 添加调制因子
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # 计算最终损失，考虑调制因子和 gamma 参数
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 如果 alpha 大于等于 0
    if alpha >= 0:
        # 计算 alpha_t
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        # 更新损失值
        loss = alpha_t * loss

    # 返回损失的均值并求和，再除以盒子的数量
    return loss.mean(1).sum() / num_boxes
# 从 https://github.com/facebookresearch/detr/blob/master/models/detr.py 中提取的代码
class DetrLoss(nn.Module):
    """
    这个类计算 DetrForObjectDetection/DetrForSegmentation 的损失。 过程分为两步：1) 计算真实框和模型输出之间的匈牙利分配 2) 监督每对匹配的真实框/预测（监督类别和框）。

    关于 `num_classes` 参数的说明（从原始仓库 detr.py 中复制）："损失函数的 `num_classes` 参数的命名有点误导。它实际上对应于 `max_obj_id` + 1，其中 `max_obj_id` 是数据集中类别的最大 id。例如，COCO 的 `max_obj_id` 为 90，所以我们传入 `num_classes` 为 91。另一个例子，对于一个只有一个 id 为 1 的类别的数据集，应该传入 `num_classes` 为 2（`max_obj_id` + 1）。有关此内容的更多详细信息，请查看以下讨论 https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"

    Args:
        matcher (`DetrHungarianMatcher`):
            能够计算目标和提议之间匹配的模块。
        num_classes (`int`):
            对象类别的数量，不包括特殊的无对象类别。
        eos_coef (`float`):
            应用于无对象类别的相对分类权重。
        losses (`List[str]`):
            要应用的所有损失的列表。查看 `get_loss` 获取所有可用损失的列表。
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # 原始实现中的 logging 参数已被移除
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        分类损失（NLL）目标字典必须包含键为 "class_labels" 的张量，维度为 [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("在输出中未找到 logits")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        # 获取模型输出中的logits
        logits = outputs["logits"]
        # 获取logits所在设备
        device = logits.device
        # 计算目标长度，即每个目标中类别标签的数量
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算预测中非"no-object"（最后一个类别）的数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 计算基于L1损失的cardinality error
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        # 构建损失字典
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # 检查是否在模型输出中存在预测框
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        # 获取源索引的排列顺序
        idx = self._get_source_permutation_idx(indices)
        # 获取预测框和目标框
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算L1回归损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        # 计算并添加边界框损失
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算并添加GIoU损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        计算与掩模相关的损失：焦点损失和Dice损失。

        目标字典必须包含键“masks”，其中包含维度为[nb_target_boxes, h, w]的张量。
        """
        if "pred_masks" not in outputs:
            raise KeyError("outputs中未找到预测掩模")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO 使用valid来掩盖由于填充而导致的无效区域
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # 将预测上采样到目标大小
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # 根据索引重新排列预测
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # 根据索引重新排列目标
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"损失 {loss} 不支持")
        return loss_map[loss](outputs, targets, indices, num_boxes)
    # 对模型输出和目标进行计算损失

    # 针对模型输出进行处理，去掉辅助输出
    outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

    # 使用匹配器查找最后一层输出和目标之间的匹配
    indices = self.matcher(outputs_without_aux, targets)

    # 计算所有节点上的平均目标框数，用于规范化
    num_boxes = sum(len(t["class_labels"]) for t in targets)
    # 将目标框数转换为张量，并移动到与输出设备相同的设备上
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    
    # （Niels）注释掉下面的函数，分布式训练将会添加
    # if is_dist_avail_and_initialized():
    #     torch.distributed.all_reduce(num_boxes)
    # （Niels）在原始实现中，num_boxes会被 get_world_size() 效果除
    num_boxes = torch.clamp(num_boxes, min=1).item()

    # 计算所有请求的损失
    losses = {}
    for loss in self.losses:
        losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

    # 如果存在辅助损失，我们会针对每个中间层的输出重复该过程
    if "auxiliary_outputs" in outputs:
        for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
            # 使用匹配器对辅助输出和目标进行匹配
            indices = self.matcher(auxiliary_outputs, targets)
            for loss in self.losses:
                if loss == "masks":
                    # 中间层的掩模损失计算开销太大，我们忽略它们
                    continue
                # 获取损失值
                l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                # 给损失值的键加上编号，以区分不同的中间层
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

    # 返回损失值
    return losses
# 从 https://github.com/facebookresearch/detr/blob/master/models/detr.py 获取的代码
# 定义一个多层感知机（MLP，也称为 FFN），用于预测相对于图像的标准化中心坐标、高度和宽度的简单网络结构。

class DetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # 定义隐藏层维度列表
        h = [hidden_dim] * (num_layers - 1)
        # 初始化多个线性层组成的列表
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    # 前向传播函数
    def forward(self, x):
        # 循环遍历所有层，对输入数据进行前向传播
        for i, layer in enumerate(self.layers):
            # 如果不是最后一层，使用ReLU激活函数
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # 返回网络输出
        return x


# 从 https://github.com/facebookresearch/detr/blob/master/models/matcher.py 获取的代码
# 这个类计算网络的目标和预测之间的匹配。

class DetrHungarianMatcher(nn.Module):
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
        # 检查所需的后端是否已安装
        requires_backends(self, ["scipy"])

        # 设置分类错误在匹配成本中的相对权重
        self.class_cost = class_cost
        # 设置边界框坐标的 L1 错误在匹配成本中的相对权重
        self.bbox_cost = bbox_cost
        # 设置边界框的 giou 损失在匹配成本中的相对权重
        self.giou_cost = giou_cost
        # 如果所有匹配器的成本都为0，则引发值错误
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    # 定义 forward 函数，用于执行模型前向传播计算
    # 参数说明:
    # - outputs (dict): 包含以下信息的字典:
    #   * "logits": 维度为 [batch_size, num_queries, num_classes] 的张量，包含分类 logits
    #   * "pred_boxes": 维度为 [batch_size, num_queries, 4] 的张量，包含预测框坐标
    # - targets (List[dict]): 每个目标标注为一个字典的列表 (len(targets) = batch_size)，每个字典包含:
    #   * "class_labels": 维度为 [num_target_boxes] 的张量 (num_target_boxes 为目标中真实对象的数量)，包含类别标签
    #   * "boxes": 维度为 [num_target_boxes, 4] 的张量，包含目标框坐标

    def forward(self, outputs, targets):
        # 获取 batch_size 和 num_queries
        batch_size, num_queries = outputs["logits"].shape[:2]

        # 将 logits 和 pred_boxes 展平，以便在批处理中计算成本矩阵
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 连接目标标签和框
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算分类成本。与损失不同，我们不使用 NLL，而是用 1 - proba[target class] 近似。
        # 这里的 1 是一个常量，不影响匹配，可以省略。
        class_cost = -out_prob[:, target_ids]

        # 计算框之间的 L1 成本
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # 计算框之间的 giou 成本
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # 最终成本矩阵
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # 获取每个目标的大小
        sizes = [len(v["boxes"]) for v in targets]
        
        # 使用 linear_sum_assignment 求解成本矩阵中每个分块的最佳匹配
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        # 返回最终匹配结果以列表形式
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# 下方的边界框实用程序取自 https://github.com/facebookresearch/detr/blob/master/util/box_ops.py

# 定义一个函数用于将张量提升为更高的类型，防止在乘法中出现数值溢出
def _upcast(t: Tensor) -> Tensor:
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

# 计算一组边界框的面积，这些边界框由它们的（x1，y1，x2，y2）坐标指定
def box_area(boxes: Tensor) -> Tensor:
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# 修改自 torchvision，还需要返回并集
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

# 计算广义 IoU 参考 https://giou.stanford.edu/，边界框应为 [x0, y0, x1, y1]（角点）格式
def generalized_box_iou(boxes1, boxes2):
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

# 取自 https://github.com/facebookresearch/detr/blob/master/util/misc.py#L306
# 按轴查找列表最大值
def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

# NestedTensor 类的定义
class NestedTensor(object):
    # 初始化函数，接收张量列表和可选的掩码张量作为参数
    def __init__(self, tensors, mask: Optional[Tensor]):
        # 将传入的张量列表赋值给对象的 tensors 属性
        self.tensors = tensors
        # 将传入的掩码张量赋值给对象的 mask 属性
        self.mask = mask
    
    # 将 NestedTensor 对象的张量和掩码张量移动到指定的设备上
    def to(self, device):
        # 将张量列表移动到指定的设备上，并赋值给 cast_tensor 变量
        cast_tensor = self.tensors.to(device)
        # 获取对象的掩码张量
        mask = self.mask
        # 如果掩码张量不为 None
        if mask is not None:
            # 将掩码张量移动到指定的设备上，并赋值给 cast_mask 变量
            cast_mask = mask.to(device)
        else:
            # 如果掩码张量为 None，则将 cast_mask 设为 None
            cast_mask = None
        # 返回一个新的 NestedTensor 对象，其中的张量和掩码张量已经移动到指定的设备上
        return NestedTensor(cast_tensor, cast_mask)
    
    # 将 NestedTensor 对象分解为张量列表和掩码张量
    def decompose(self):
        # 返回对象的张量列表和掩码张量
        return self.tensors, self.mask
    
    # 重写对象的字符串表示形式，返回张量列表的字符串表示形式
    def __repr__(self):
        return str(self.tensors)
# 从给定的张量列表中创建嵌套张量
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # 检查张量列表第一个张量的维度是否为3
    if tensor_list[0].ndim == 3:
        # 获取张量列表中每个张量的最大尺寸
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # 创建批处理形状
        batch_shape = [len(tensor_list)] + max_size
        # 获取批处理的大小、通道数、高度和宽度
        batch_size, num_channels, height, width = batch_shape
        # 获取张量列表中第一个张量的数据类型和设备
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # 创建零张量，形状为批处理形状
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 创建全为True的蒙版张量，形状为(batch_size, height, width)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # 遍历张量列表，将每个张量的数据拷贝到对应的零张量中，并更新蒙版张量
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        # 如果张量列表中的张量维度不为3，则抛出异常
        raise ValueError("Only 3-dimensional tensors are supported")
    # 返回嵌套张量，包括数据张量和蒙版张量
    return NestedTensor(tensor, mask)
```