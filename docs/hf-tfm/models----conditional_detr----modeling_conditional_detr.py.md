# `.\models\conditional_detr\modeling_conditional_detr.py`

```
# 设置文件编码为 utf-8
# 代码版权信息
#
# 使用 Apache License, Version 2.0 许可证
# 你不得使用本文件，除非符合许可证的规定
# 可在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或以书面形式同意，否则根据许可证分发的软件将按“原样”基础分发，
# 没有明示或暗示的任何担保或条件。请查看许可证以获取特定语言规定的权限和
# 限制
""" PyTorch Conditional DETR model."""

# 导入所需的库和模块
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
from ..auto import AutoBackbone
from .configuration_conditional_detr import ConditionalDetrConfig

# 检查是否可用的模块
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

if is_timm_available():
    from timm import create_model

if is_vision_available():
    from ...image_transforms import center_to_corners_format

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点
_CONFIG_FOR_DOC = "ConditionalDetrConfig"
_CHECKPOINT_FOR_DOC = "microsoft/conditional-detr-resnet-50"

# 预训练模型的列表
CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/conditional-detr-resnet-50",
    # 查看所有 Conditional DETR 模型 https://huggingface.co/models?filter=conditional_detr
]


@dataclass
class ConditionalDetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Conditional DETR 解码器输出的基类。该类向 BaseModelOutputWithCrossAttentions 添加了一个属性，
    即可选的中间解码器激活的堆栈，即每个解码器层的输出，每个输出都经过了一个 layernorm。
    当使用辅助解码损失训练模型时，这很有用。
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
    
    # 定义一个可选的中间隐藏状态，默认为None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    # 定义一个可选的参考点元组，默认为None
    reference_points: Optional[Tuple[torch.FloatTensor]] = None
# 定义了一个数据类ConditionalDetrModelOutput，继承自Seq2SeqModelOutput类
@dataclass
class ConditionalDetrModelOutput(Seq2SeqModelOutput):
    # ConditionalDetrModelOutput类的基类为Seq2SeqModelOutput类，此处对其属性进行了解释
    """
    Base class for outputs of the Conditional DETR encoder-decoder model. This class adds one attribute to
    Seq2SeqModelOutput, namely an optional stack of intermediate decoder activations, i.e. the output of each decoder
    layer, each of them gone through a layernorm. This is useful when training the model with auxiliary decoding
    losses.

    """
    # 可选属性，保存了中间层解码激活状态的堆栈，每个解码层的输出都经过了一个layernorm。在使用辅助解码损失训练模型时很有用
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    # 可选属性，保存了参考点的元组
    reference_points: Optional[Tuple[torch.FloatTensor]] = None

# 定义了一个数据类ConditionalDetrObjectDetectionOutput，继承自ModelOutput类
@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrObjectDetectionOutput with Detr->ConditionalDetr
class ConditionalDetrObjectDetectionOutput(ModelOutput):
    # ConditionalDetrObjectDetectionOutput类的基类为ModelOutput类，此处对其属性进行了解释
    """
    Output type of [`ConditionalDetrForObjectDetection`].

    """
    # 可选属性，保存了损失的浮点tensor
    loss: Optional[torch.FloatTensor] = None
    # 可选属性，保存了损失字典
    loss_dict: Optional[Dict] = None
    # tensor属性，保存了logits
    logits: torch.FloatTensor = None
    # tensor属性，保存了预测框
    pred_boxes: torch.FloatTensor = None
    # 可选属性，保存了辅助输出列表
    auxiliary_outputs: Optional[List[Dict]] = None
    # 可选属性，保存了最后的隐藏状态
    last_hidden_state: Optional[torch.FloatTensor] = None
    # 可选属性，保存了解码器隐藏状态的元组
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选属性，保存了解码器注意力的元组
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选属性，保存了交叉注意力的元组
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选属性，保存了编码器最后隐藏状态
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 可选属性，保存了编码器隐藏状态的元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选属性，保存了编码器注意力的元组
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

# 定义了一个数据类ConditionalDetrSegmentationOutput，继承自ModelOutput类
@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrSegmentationOutput with Detr->ConditionalDetr
class ConditionalDetrSegmentationOutput(ModelOutput):
    # ConditionalDetrSegmentationOutput类的基类为ModelOutput类，此处对其属性进行了解释
    """
    Output type of [`ConditionalDetrForSegmentation`].

    """
    # 可选属性，保存了损失的浮点tensor
    loss: Optional[torch.FloatTensor] = None
    # 可选属性，保存了损失字典
    loss_dict: Optional[Dict] = None
    # tensor属性，保存了logits
    logits: torch.FloatTensor = None
    # tensor属性，保存了预测框
    pred_boxes: torch.FloatTensor = None
    # tensor属性，保存了预测掩码
    pred_masks: torch.FloatTensor = None
    # 可选属性，保存了辅助输出列表
    auxiliary_outputs: Optional[List[Dict]] = None
    # 可选属性，保存了最后的隐藏状态
    last_hidden_state: Optional[torch.FloatTensor] = None
    # 可选属性，保存了解码器隐藏状态的元组
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选属性，保存了解码器注意力的元组
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选属性，保存了交叉注意力的元组
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选属性，保存了编码器最后隐藏状态
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 可选属性，保存了编码器隐藏状态的元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选属性，保存了编码器注意力的元组
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

# 定义了一个nn.Module类 ConditionalDetrFrozenBatchNorm2d
# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->ConditionalDetr
class ConditionalDetrFrozenBatchNorm2d(nn.Module):
    # ConditionalDetrFrozenBatchNorm2d类继承自nn.Module类，此处对其作用进行了解释
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """
    # 初始化方法，接受参数 n
    def __init__(self, n):
        # 调用父类的初始化方法
        super().__init__()
        # 注册权重参数并初始化为全1
        self.register_buffer("weight", torch.ones(n))
        # 注册偏置参数并初始化为全0
        self.register_buffer("bias", torch.zeros(n))
        # 注册用于记录均值的缓冲区，并初始化为全0
        self.register_buffer("running_mean", torch.zeros(n))
        # 注册用于记录方差的缓冲区，并初始化为全1
        self.register_buffer("running_var", torch.ones(n))

    # 从状态字典中加载模型参数的方法
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 构造用于记录迭代次数的参数名
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # 如果状态字典中存在该参数名，就从状态字典中删除该参数
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类的从状态字典加载方法
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # 前向传播方法
    def forward(self, x):
        # 将权重参数重塑为 (1, n, 1, 1) 的形状
        weight = self.weight.reshape(1, -1, 1, 1)
        # 将偏置参数重塑为 (1, n, 1, 1) 的形状
        bias = self.bias.reshape(1, -1, 1, 1)
        # 将运行方差参数重塑为 (1, n, 1, 1) 的形状
        running_var = self.running_var.reshape(1, -1, 1, 1)
        # 将运行均值参数重塑为 (1, n, 1, 1) 的形状
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        # 设置一个很小的数 epsilon
        epsilon = 1e-5
        # 计算缩放因子 scale，用于归一化数据
        scale = weight * (running_var + epsilon).rsqrt()
        # 计算偏置参数，用于归一化数据
        bias = bias - running_mean * scale
        # 返回归一化处理后的数据
        return x * scale + bias
# 从transformers.models.detr.modeling_detr.replace_batch_norm复制并将Detr->ConditionalDetr
def replace_batch_norm(model):
    """
    递归替换所有的`torch.nn.BatchNorm2d`为`ConditionalDetrFrozenBatchNorm2d`。

    Args:
        model (torch.nn.Module):
            输入模型
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = ConditionalDetrFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


# 从transformers.models.detr.modeling_detr.DetrConvEncoder复制
class ConditionalDetrConvEncoder(nn.Module):
    """
    使用AutoBackbone API或来自timm库中的一个卷积主干。

    将nn.BatchNorm2d图层替换为上面定义的DetrFrozenBatchNorm2d。

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            # 使用timm库中的创建模型函数，创建相应卷积主干
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            # 使用AutoBackbone根据配置创建相应卷积主干
            backbone = AutoBackbone.from_config(config.backbone_config)

        # 用冻结的batch norm替换batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        # 如果使用timm主干，则使用其特征信息通道数，否则使用其通道数
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        # 如果使用resnet主干，则冻结特定的参数
        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)
    # 前向传播函数，接收像素数值和像素遮罩作为输入
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # 通过模型处理像素数值，得到特征图列表
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        # 初始化输出列表
        out = []
        # 遍历特征图列表
        for feature_map in features:
            # 将像素遮罩下采样至与对应特征图形状相匹配
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            # 将特征图和对应的遮罩添加到输出列表中
            out.append((feature_map, mask))
        # 返回输出列表
        return out
# 从transformers.models.detr.modeling_detr.DetrConvModel中复制代码，并将Detr改为ConditionalDetr
class ConditionalDetrConvModel(nn.Module):
    """
    这个模块在所有卷积编码器的中间特征图上添加了2D位置嵌入。
    """

    def __init__(self, conv_encoder, position_embedding):
        # 调用父类的构造函数
        super().__init__()
        # 保存卷积编码器和位置嵌入
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):
        # 将pixel_values和pixel_mask通过骨干网络获得(feature_map, pixel_mask)元组的列表
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # 添加位置编码
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class ConditionalDetrSinePositionEmbedding(nn.Module):
    """
    这是一个更标准的版本的位置嵌入，非常类似于Attention is all you need论文中使用的版本，适用于图像。
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        # 调用父类的构造函数
        super().__init__()
        # 保存嵌入维度、温度、是否归一化、以及比例
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        # 如果没有提供pixel_mask，则抛出 ValueError
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        # 沿着行和列对pixel_mask进行累加，并将数据类型设为torch.float32
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        # 如果需要归一化，则将位置嵌入除以最后一个像素值加上一个很小的值，并乘以比例
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        # 在[0, embedding_dim)范围内创建一个张量
        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        # 计算位置编码
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # 返回位置编码
        return pos


# 从transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding中复制代码，并将Detr改为ConditionalDetr
class ConditionalDetrLearnedPositionEmbedding(nn.Module):
    """
    这个模块学习一个到固定大小的位置嵌入。
    """
    # 定义初始化方法，参数为嵌入维度
    def __init__(self, embedding_dim=256):
        # 调用父类的初始化方法
        super().__init__()
        # 创建行嵌入对象，通过将50个整数映射为嵌入特征向量，维度为embedding_dim
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        # 创建列嵌入对象，通过将50个整数映射为嵌入特征向量，维度为embedding_dim
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    # 定义前向传播方法，接收像素值和像素掩码参数
    def forward(self, pixel_values, pixel_mask=None):
        # 获取像素值的高度和宽度
        height, width = pixel_values.shape[-2:]
        # 在设备上创建一个从0到宽度-1的张量
        width_values = torch.arange(width, device=pixel_values.device)
        # 在设备上创建一个从0到高度-1的张量
        height_values = torch.arange(height, device=pixel_values.device)
        # 对列嵌入进行索引，获取对应的嵌入特征向量
        x_emb = self.column_embeddings(width_values)
        # 对行嵌入进行索引，获取对应的嵌入特征向量
        y_emb = self.row_embeddings(height_values)
        # 按照指定维度连接列嵌入和行嵌入的张量，并形成二维张量
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # 对pos的维度进行转置，将第0维和第2维交换位置
        pos = pos.permute(2, 0, 1)
        # 对pos的维度进行扩展，增加一维，形成三维张量
        pos = pos.unsqueeze(0)
        # 对pos的维度进行复制，将第0维复制为与pixel_values相同的大小，形成四维张量
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # 返回pos
        return pos
# 从transformers.models.detr.modeling_detr.build_position_encoding复制的代码，将Detr更改为ConditionalDetr
def build_position_encoding(config):
    # 将d_model除以2作为步长
    n_steps = config.d_model // 2
    # 如果位置嵌入类型为"sine"
    if config.position_embedding_type == "sine":
        # TODO 找到一种更好的暴露其他参数的方法
        position_embedding = ConditionalDetrSinePositionEmbedding(n_steps, normalize=True)
    # 如果位置嵌入类型为"learned"
    elif config.position_embedding_type == "learned":
        position_embedding = ConditionalDetrLearnedPositionEmbedding(n_steps)
    else:
        # 抛出异常，位置嵌入类型不支持
        raise ValueError(f"Not supported {config.position_embedding_type}")
    # 返回位置嵌入
    return position_embedding


# 用于生成二维坐标的正弦位置嵌入的函数
def gen_sine_position_embeddings(pos_tensor, d_model):
    scale = 2 * math.pi
    dim = d_model // 2
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


# 计算逆sigmoid函数
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# 从transformers.models.detr.modeling_detr.DetrAttention复制的代码
class DetrAttention(nn.Module):
    """
    'Attention Is All You Need' 纸中的多头注意力。

    在这里，我们将位置嵌入添加到查询和键中（如DETR论文中所解释的）。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        # 嵌入维度
        self.embed_dim = embed_dim
        # 头的数量
        self.num_heads = num_heads
        # 丢弃率
        self.dropout = dropout
        # 头维度
        self.head_dim = embed_dim // num_heads
        # 如果头维度乘以头数不等于嵌入维度，则引发错误
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        # 缩放
        self.scaling = self.head_dim**-0.5

        # 线性变换，用于生成查询、键、值和输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # 定义具有位置嵌入的操作，接受一个张量、对象查询和其他参数
    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        # 从参数中弹出 position_embeddings，并赋给变量 position_embeddings
        position_embeddings = kwargs.pop("position_embeddings", None)
        
        # 如果 kwargs 不为空，则抛出 ValueError 异常
        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")
        
        # 如果同时指定了 position_embeddings 和 object_queries，则抛出 ValueError 异常
        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )
        
        # 如果 position_embeddings 不为空，则将 position_embeddings 弃用的警告信息打印出来，将 position_embeddings 赋给 object_queries
        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings
        
        # 如果 object_queries 为空，则返回原始张量；否则返回原始张量加上 object_queries
        return tensor if object_queries is None else tensor + object_queries
    
    # 前向方法，接受隐藏状态、注意力蒙版、对象查询、键值状态、空间位置嵌入、是否输出注意力权重等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        spatial_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
# 定义一个条件性 DETR 编码器层
class ConditionalDetrEncoderLayer(nn.Module):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__()
        # 设定嵌入维度
        self.embed_dim = config.d_model
        # 创建自注意力模块
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 创建自注意力层规范化模块
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置丢弃率
        self.dropout = config.dropout
        # 激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 激活层丢弃率
        self.activation_dropout = config.activation_dropout
        # 全连接层1
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 全连接层2
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 最终层规范化模块
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    # 定义一个前向传播函数，用于处理输入数据
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量，形状为(batch, seq_len, embed_dim)
        attention_mask: torch.Tensor,  # 注意力掩码张量，形状为(batch, 1, target_len, source_len)，用极大负值表示填充元素
        object_queries: torch.Tensor = None,  # 目标查询（也称为内容嵌入），要添加到隐藏状态中的张量，默认为None
        output_attentions: bool = False,  # 是否返回所有注意力层的注意力张量，默认为False
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`torch.FloatTensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 从关键字参数中弹出"position_embeddings"，如果存在的话
        position_embeddings = kwargs.pop("position_embeddings", None)

        # 如果还有其他未预期的参数存在，则抛出值错误
        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        # 如果同时指定了position_embeddings和object_queries，则抛出值错误
        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        # 如果position_embeddings不为None，则警告已废弃，并设置object_queries为position_embeddings
        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        # 保存输入的隐藏状态作为残差连接的一部分
        residual = hidden_states
        # 使用自注意力机制计算新的隐藏状态以及可能的注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        # 对新的隐藏状态进行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接的隐藏状态和新的隐藏状态相加
        hidden_states = residual + hidden_states
        # 对相加后的隐藏状态进行层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存加了自注意力机制的隐藏状态作为残差连接的一部分
        residual = hidden_states
        # 使用激活函数对隐藏状态进行变换
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对变换后的隐藏状态进行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        # 对新的隐藏状态进行线性变换
        hidden_states = self.fc2(hidden_states)
        # 对线性变换后的隐藏状态进行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 将残差连接的隐藏状态和新的隐藏状态相加
        hidden_states = residual + hidden_states
        # 对相加后的隐藏状态进行层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果处于训练模式，则进行特定的处理以防止梯度爆炸或梯度消失
        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 定义输出为最终的隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出
        return outputs
# 定义一个条件化的 DETR 解码器层
class ConditionalDetrDecoderLayer(nn.Module):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model

        d_model = config.d_model
        # 定义解码器 Self-Attention 的投影
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)

        self.self_attn = ConditionalDetrAttention(
            embed_dim=self.embed_dim,
            out_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 定义解码器 Cross-Attention 的投影
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        self.encoder_attn = ConditionalDetrAttention(
            self.embed_dim * 2, self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.nhead = config.decoder_attention_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        query_sine_embed: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        is_first: Optional[bool] = False,
        **kwargs,
    # 从 transformers.models.detr.modeling_detr.DetrClassificationHead 复制，将 Detr 替换为 ConditionalDetr
class ConditionalDetrClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
    # 定义一个向前传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states: torch.Tensor):
        # 对隐藏状态进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行双曲正切激活函数的处理
        hidden_states = torch.tanh(hidden_states)
        # 对处理后的隐藏状态再次进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 通过输出投影层进行最终的输出
        hidden_states = self.out_proj(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.detr.modeling_detr.DetrMLPPredictionHead复制到MLP
class MLP(nn.Module):
    """
    非常简单的多层感知器（MLP，也称为FFN），用于预测边界框相对于图像的归一化中心坐标、高度和宽度。
    
    从https://github.com/facebookresearch/detr/blob/master/models/detr.py复制
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 创建多层神经网络
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 从transformers.models.detr.modeling_detr.DetrPreTrainedModel复制到ConditionalDetrPreTrainedModel
class ConditionalDetrPreTrainedModel(PreTrainedModel):
    config_class = ConditionalDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"ConditionalDetrConvEncoder", r"ConditionalDetrEncoderLayer", r"ConditionalDetrDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        if isinstance(module, ConditionalDetrMHAttentionMap):
            nn.init.zeros_(module.k_linear.bias)
            nn.init.zeros_(module.q_linear.bias)
            nn.init.xavier_uniform_(module.k_linear.weight, gain=xavier_std)
            nn.init.xavier_uniform_(module.q_linear.weight, gain=xavier_std)
        elif isinstance(module, ConditionalDetrLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 与TF版本稍有不同，TF版本使用截断正态分布初始化
            # 参见 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# r"""
#     This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
#     library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
#     etc.)
# 
#     This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
#     Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
# """
    # 该部分文本给出了函数的参数说明
    # config: 一个ConditionalDetrConfig类的实例，包含模型的所有参数。使用配置文件初始化不会加载与模型相关联的权重，只会加载配置。可以使用PreTrainedModel.from_pretrained方法加载模型权重。
"""

CONDITIONAL_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下会忽略填充。

            可以使用 [`AutoImageProcessor`] 获取像素值。有关详细信息，请参阅 [`ConditionalDetrImageProcessor.__call__`]。

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            避免在填充像素值上执行注意力的遮罩。遮罩值在 `[0, 1]` 范围内：

            - 对于实际像素（即**未屏蔽**）为 1，
            - 对于填充像素（即**已屏蔽**）为 0。

            [什么是注意力遮罩?](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            默认情况下不使用。可用于屏蔽对象查询。
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            元组包含 (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*optional*) 是编码器最后一层的输出的隐藏状态序列。用于解码器的交叉注意力。
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            可选地，可以直接传递图像的平坦表示，而不是传递平坦的特征图（backbone + 投影层的输出）。
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            可选地，可以选择直接传递嵌入的表示，而不是使用零张量初始化查询。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的`attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的`hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""


# 从 transformers.models.detr.modeling_detr.DetrEncoder 复制到 ConditionalDetrEncoder with Detr->ConditionalDetr,DETR->ConditionalDETR
class ConditionalDetrEncoder(ConditionalDetrPreTrainedModel):
    """
    Transformer 编码器，包含 *config.encoder_layers* 自注意层。每层都是一个 [`ConditionalDetrEncoderLayer`]。

    编码器通过多个自注意层更新平坦特征图。

    适用于 ConditionalDETR 的小调整：

    - 在前向传递中添加了对象查询。
        Args:
            config: ConditionalDetrConfig
        """

        def __init__(self, config: ConditionalDetrConfig):
            # 调用父类构造函数初始化
            super().__init__(config)

            # 从config中获取dropout值和encoder_layerdrop值
            self.dropout = config.dropout
            self.layerdrop = config.encoder_layerdrop

            # 使用列表推导式创建包含多个ConditionalDetrEncoderLayer对象的ModuleList
            self.layers = nn.ModuleList([ConditionalDetrEncoderLayer(config) for _ in range(config.encoder_layers)])

            # 在原始ConditionalDETR中，encoder结尾没有使用layernorm，因为默认的"normalize_before"为False

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
class ConditionalDetrDecoder(ConditionalDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`ConditionalDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for Conditional DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: ConditionalDetrConfig
    """

    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        # Create a list of decoder layers
        self.layers = nn.ModuleList([ConditionalDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # in Conditional DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = nn.LayerNorm(config.d_model)
        # Save the d_model value from config for later use
        d_model = config.d_model
        self.gradient_checkpointing = False

        # Create FFN (Feed-Forward Network) modules
        # query_scale is the FFN applied on f to generate transformation T
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        # Set the ca_qpos_proj of all layers except the last layer to None
        for layer_id in range(config.decoder_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None

        # Initialize weights and apply final processing
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


@add_start_docstrings(
    """
    The bare Conditional DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    CONDITIONAL_DETR_START_DOCSTRING,
)
class ConditionalDetrModel(ConditionalDetrPreTrainedModel):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        # Create a ConditionalDetrConvEncoder instance
        backbone = ConditionalDetrConvEncoder(config)
        object_queries = build_position_encoding(config)
        # Create a ConditionalDetrConvModel instance, passing in the created backbone and object_queries
        self.backbone = ConditionalDetrConvModel(backbone, object_queries)

        # Create projection layer
        # Create a 1x1 convolutional layer to project the output of the backbone to the d_model size
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)

        # Create an embedding layer for query position embeddings
        # The embedding size is d_model and the number of queries is config.num_queries
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        # Create encoder and decoder
        self.encoder = ConditionalDetrEncoder(config)
        self.decoder = ConditionalDetrDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder


注释：
    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 冻结骨干网络的参数，使其不可训练
    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    # 解冻骨干网络的参数，使其可训练
    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    # 模型的前向传播方法，接受多种输入参数
    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ConditionalDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加起始文档字符串，描述 CONDITIONAL_DETR 模型的结构和用途，包括了一个骨干网络和编码器-解码器 Transformer，
# 并在顶部添加了用于目标检测的对象检测头
@add_start_docstrings(
    """
    CONDITIONAL_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    CONDITIONAL_DETR_START_DOCSTRING,
)
# 定义 ConditionalDetrForObjectDetection 类，继承自 ConditionalDetrPreTrainedModel 类
class ConditionalDetrForObjectDetection(ConditionalDetrPreTrainedModel):
    # 初始化方法，接受一个 ConditionalDetrConfig 类型的参数
    def __init__(self, config: ConditionalDetrConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # CONDITIONAL DETR encoder-decoder 模型
        # 创建一个 ConditionalDetrModel 对象，传入配置信息
        self.model = ConditionalDetrModel(config)

        # 对象检测头
        # 分类器线性层，输入维度为 config.d_model，输出维度为 config.num_labels
        # 这里为了兼容 "无对象" 类别，输出类别数需加一
        self.class_labels_classifier = nn.Linear(
            config.d_model, config.num_labels
        )  # We add one for the "no object" class
        # 边界框预测器，使用 ConditionalDetrMLPPredictionHead 类创建对象
        # 输入维度为 config.d_model，隐藏层维度为 config.d_model，输出维度为 4（边界框坐标），隐藏层数为 3
        self.bbox_predictor = ConditionalDetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 https://github.com/Atten4Vis/conditionalDETR/blob/master/models/conditional_detr.py 获取的函数
    # 用于设置辅助损失，在 torchscript 中需要将输出转换成列表格式以适应不同类型值的字典
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是一个解决方案，使 torchscript 能够正常运行，因为 torchscript 不支持具有非同类值的字典，
        # 例如一个同时包含张量和列表的字典。
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # 添加起始文档字符串到模型前向方法，描述输入和输出，以及配置信息
    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    # 替换返回值文档字符串，指定输出类型为 ConditionalDetrObjectDetectionOutput，配置信息为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=ConditionalDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法，接受多个输入参数并返回模型输出
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化函数，接受一个 ConditionalDetrConfig 类型的参数，并调用父类的初始化函数
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)

        # 创建条件化的目标检测模型对象
        self.conditional_detr = ConditionalDetrForObjectDetection(config)

        # 创建分割头部
        # 获取隐藏层大小和注意力头数
        hidden_size, number_of_heads = config.d_model, config.encoder_attention_heads
        # 获取中间通道的大小信息
        intermediate_channel_sizes = self.conditional_detr.model.backbone.conv_encoder.intermediate_channel_sizes

        # 创建 ConditionalDetrMaskHeadSmallConv 对象用于分割
        self.mask_head = ConditionalDetrMaskHeadSmallConv(
            hidden_size + number_of_heads, intermediate_channel_sizes[::-1][-3:], hidden_size
        )

        # 创建边框的注意力图
        self.bbox_attention = ConditionalDetrMHAttentionMap(
            hidden_size, hidden_size, number_of_heads, dropout=0.0, std=config.init_xavier_std
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受一系列输入参数，返回条件化 Detr 分割输出
    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ConditionalDetrSegmentationOutput, config_class=_CONFIG_FOR_DOC)
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
        # ...
    ):
# 将输入的张量扩展为指定长度
def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

# 从transformers.models.detr.modeling_detr.DetrMaskHeadSmallConv中复制到ConditionalDetr中的代码
class ConditionalDetrMaskHeadSmallConv(nn.Module):
    """
    使用分组归一化的简单卷积头，使用FPN方法进行上采样
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        if dim % 8 != 0:
            raise ValueError(
                "The hidden_size + number of attention heads must be divisible by 8 as the number of groups in"
                " GroupNorm is set to 8"
            )

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        # 设定卷积层和分组归一化层
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

        # 适配FPN维度
        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        # 初始化所有卷积层的权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        # 将投影特征图x（形状为（batch_size，d_model，heigth/32，width/32））与bbox_mask（注意力图，形状为（batch_size，n_queries，n_heads，height/32，width/32））拼接在一起
        # 我们扩展投影特征图以匹配头的数量
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        
        # 通过layer1处理x
        x = self.lay1(x)
        # 通过gn1处理x
        x = self.gn1(x)
        # 通过激活函数ReLU处理x
        x = nn.functional.relu(x)
        
        # 通过layer2处理x
        x = self.lay2(x)
        # 通过gn2处理x
        x = self.gn2(x)
        # 通过激活函数ReLU处理x
        x = nn.functional.relu(x)
        
        # 使用adapter1对fpns列表中的第一个张量进行处理
        cur_fpn = self.adapter1(fpns[0])
        # 如果当前张量的大小与x的大小不一致，进行扩展
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 根据cur_fpn和x的尺寸进行插值操作后相加
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 通过layer3处理x
        x = self.lay3(x)
        # 通过gn3处理x
        x = self.gn3(x)
        # 通过激活函数ReLU处理x
        x = nn.functional.relu(x)
        
        # 使用adapter2对fpns列表中的第二个张量进行处理
        cur_fpn = self.adapter2(fpns[1])
        # 如果当前张量的大小与x的大小不一致，进行扩展
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 根据cur_fpn和x的尺寸进行插值操作后相加
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 通过layer4处理x
        x = self.lay4(x)
        # 通过gn4处理x
        x = self.gn4(x)
        # 通过激活函数ReLU处理x
        x = nn.functional.relu(x)
        
        # 使用adapter3对fpns列表中的第三个张量进行处理
        cur_fpn = self.adapter3(fpns[2])
        # 如果当前张量的大小与x的大小不一致，进行扩展
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 根据cur_fpn和x的尺寸进行插值操作后相加
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 通过layer5处理x
        x = self.lay5(x)
        # 通过gn5处理x
        x = self.gn5(x)
        # 通过激活函数ReLU处理x
        x = nn.functional.relu(x)
        
        # 通过输出层处理x
        x = self.out_lay(x)
        # 返回处理后的x
        return x
# 从transformers.models.detr.modeling_detr.DetrMHAttentionMap中复制的代码，并将名称Detr->ConditionalDetr
class ConditionalDetrMHAttentionMap(nn.Module):
    """这是一个2D注意力模块，仅返回注意力softmax（不需要乘以值）"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None):
        super().__init__()
        # 定义属性
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 定义query向量的线性变换
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        # 定义key向量的线性变换
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        # 定义标准化系数
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        # 对query向量进行线性变换
        q = self.q_linear(q)
        # 对key向量进行线性变换并做二维卷积
        k = nn.functional.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        # 将query分割为每个attention head所包含的向量
        queries_per_head = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        # 将key分割为每个attention head所包含的向量
        keys_per_head = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        # 计算注意力权重
        weights = torch.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

        # 如果存在遮罩，则将权重值为True的位置替换为最小值
        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), torch.finfo(weights.dtype).min)
        # 计算softmax并加上dropout
        weights = nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


# 从transformers.models.detr.modeling_detr中复制的dice_loss函数
def dice_loss(inputs, targets, num_boxes):
    """
    计算DICE损失，类似于mask的广义IOU
    Args:
        inputs: 任意形状的浮点张量。每个示例的预测。
        targets: 与输入具有相同形状的浮点张量。存储每个输入元素的二进制分类标签（0为负类，1为正类）。
    """
    # 对输入进行sigmoid激活函数
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


# 从transformers.models.detr.modeling_detr中复制的sigmoid_focal_loss函数
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    在RetinaNet中用于密集检测的损失函数
    Args:
        inputs: 任意形状的浮点张量。每个示例的预测值。
        targets: 与输入具有相同形状的浮点张量。存储每个输入元素的二进制分类标签（0为负类，1为正类）。
    """
    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            输入数据，包含每个样本的预测值。
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            包含与 `inputs` 相同形状的张量，存储每个元素的二元分类标签（0 表示负类，1 表示正类）。
        alpha (`float`, *optional*, defaults to `0.25`):
            权重因子，范围在 (0,1) 之间，用于平衡正负样本。
        gamma (`int`, *optional*, defaults to `2`):
            调节因子的指数（1 - p_t），用于平衡简单与困难样本。

    Returns:
        Loss tensor
    """
    # 计算概率值，通过 sigmoid 函数将输入数据转换为概率值
    prob = inputs.sigmoid()
    # 计算交叉熵损失，不进行减少操作
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 添加调节因子
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # 计算损失值，采用交叉熵损失乘以调节因子的指数
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 如果 alpha 大于等于 0，则根据 alpha 对损失进行加权
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 返回损失的均值，并进行元素求和，再除以盒子数量
    return loss.mean(1).sum() / num_boxes
## 根据给定匹配的索引，从outputs中获取source的索引
idx = self._get_source_permutation_idx(indices)
## 从targets中获取真实的类别label
target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
## 创建一个全是num_classes的tensor用于表示target中的label，device和source_logits相同
target_classes = torch.full(
    source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
)
target_classes[idx] = target_classes_o
## 创建一个source_logits shape的全0的tensor，并根据target_classes的值标记为1
target_classes_onehot = torch.zeros(
    [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
    dtype=source_logits.dtype,
    layout=source_logits.layout,
    device=source_logits.device,
)
target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
## 去掉全1的最后一列，只留下前面num_classes列
target_classes_onehot = target_classes_onehot[:, :, :-1]
## 根据source_logits和target_classes_onehot计算sigmoid focal loss的损失
## 乘以source_logits的第二维度，目的是将loss按照输入张量的第二个维度大小进行缩放
loss_ce = (
    sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
    * source_logits.shape[1]
)
## 将损失添加到字典中
losses = {"loss_ce": loss_ce}
    # 计算基数误差，即预测的非空框数量的绝对误差。
    # 这实际上不是一个损失，仅用于记录目的。它不传播梯度。
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        # 获取输出中的逻辑值
        logits = outputs["logits"]
        # 获取设备信息
        device = logits.device
        # 获取目标长度，即每个目标的类标签数量
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算预测的非空框数量，即预测的类别不是“no-object”的数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 计算基数误差，使用 L1 损失函数
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        # 构建损失字典
        losses = {"cardinality_error": card_err}
        # 返回损失字典
        return losses

    # 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_boxes 复制过来的
    # 计算与边界框相关的损失，包括 L1 回归损失和 GIoU 损失。
    # 目标字典必须包含键“boxes”，其中包含维度为[nb_target_boxes，4]的张量。目标框的格式应为（center_x，center_y，w，h），并已归一化为图像大小。
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        # 如果输出中没有预测框，则引发 KeyError
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        # 获取源排列的索引
        idx = self._get_source_permutation_idx(indices)
        # 获取输出中的预测框
        source_boxes = outputs["pred_boxes"][idx]
        # 获取目标框，将所有目标框拼接成一个张量
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算边界框的 L1 损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        # 构建损失字典
        losses = {}
        # 将 L1 损失汇总，并除以框的数量
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算 GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        # 将 GIoU 损失汇总，并除以框的数量
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        # 返回损失字典
        return losses

    # 从 transformers.models.detr.modeling_detr.DetrLoss.loss_masks 复制过来的
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        计算与masks相关的损失：focal loss和dice loss。

        目标字典必须包含键“masks”，其中包含维度为[nb_target_boxes, h, w]的张量。
        """
        if "pred_masks" not in outputs:
            raise KeyError("outputs中找不到预测的masks")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO 使用valid来屏蔽由于填充而导致的无效区域的损失
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # 上采样预测到目标大小
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

    # 从transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss._get_source_permutation_idx复制而来
    def _get_source_permutation_idx(self, indices):
        # 根据索引重排预测
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # 从transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss._get_target_permutation_idx复制而来
    def _get_target_permutation_idx(self, indices):
        # 根据索引重排目标
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    # 从transformers.models.detr.modeling_detr.DetrLoss.get_loss复制而来
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"不支持损失{loss}")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    # 从transformers.models.detr.modeling_detr.DetrLoss.forward复制而来
    # 定义一个用于计算损失的方法，接收模型输出和目标作为参数
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
        # 去除辅助输出，创建 outputs_without_aux 字典
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # 获取最后一层输出和目标之间的匹配
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点上目标框的平均数量，用于归一化
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 如果存在辅助输出，针对每个中间层的输出重复上面的过程
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # 中间层的掩码损失计算成本较高，被忽略
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 返回损失字典
        return losses
# 从transformers.models.detr.modeling_detr.DetrMLPPredictionHead复制代码，并将Detr->ConditionalDetr
class ConditionalDetrMLPPredictionHead(nn.Module):
    """
    非常简单的多层感知器（MLP，也称为FFN），用于预测边界框相对于图像的规范化中心坐标、高度和宽度。

    从https://github.com/facebookresearch/detr/blob/master/models/detr.py复制而来
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 从transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrHungarianMatcher复制代码，并将DeformableDetr->ConditionalDetr
class ConditionalDetrHungarianMatcher(nn.Module):
    """
    该类计算网络目标和预测之间的分配。

    出于效率原因，目标不包括无对象。由于这个原因，一般来说，预测比目标更多。在这种情况下，我们对最佳预测进行一对一匹配，而其他预测未匹配（因此被视为非对象）。

    参数：
        class_cost：
            匹配成本中分类错误的相对权重。
        bbox_cost：
            匹配成本中边界框坐标的L1误差的相对权重。
        giou_cost：
            匹配成本中边界框的giou损失的相对权重。
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("所有匹配器成本不能为0")

    @torch.no_grad()
    # 前向传播函数
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
                      ground-truth
                     objects in the target) containing the class labels
                    * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.
    
            Returns:
                `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
                For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            """
            # 获取 batch_size 和 num_queries
            batch_size, num_queries = outputs["logits"].shape[:2]
    
            # 将预测结果展平，用于计算成本矩阵
            out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
    
            # 将目标标签和框进行拼接
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
    
            # 根据成本矩阵进行任务分配
            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# 从 transformers.models.detr.modeling_detr._upcast 复制而来
def _upcast(t: Tensor) -> Tensor:
    # 避免在乘法中出现数值溢出，将数据类型提升到相应的更高类型
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# 从 transformers.models.detr.modeling_detr.box_area 复制而来
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，这些边界框由其 (x1, y1, x2, y2) 坐标指定。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            要计算面积的边界框。它们应该采用 (x1, y1, x2, y2) 格式，其中 `0 <= x1 < x2` 和 `0 <= y1 < y2`。

    Returns:
        `torch.FloatTensor`: 包含每个边界框的面积的张量。
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# 从 transformers.models.detr.modeling_detr.box_iou 复制而来
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


# 从 transformers.models.detr.modeling_detr.generalized_box_iou 复制而来
def generalized_box_iou(boxes1, boxes2):
    """
    从 https://giou.stanford.edu/ 中获得的广义 IoU。这些边界框应该采用 [x0, y0, x1, y1]（角）格式。

    Returns:
        `torch.FloatTensor`: 一个 [N, M] 的成对矩阵，其中 N = len(boxes1) 而 M = len(boxes2)
    """
    # 如果边界框退化，会导致 inf / nan 结果，所以进行早期检查
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 必须采用 [x0, y0, x1, y1]（角）格式，但是得到了 {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 必须采用 [x0, y0, x1, y1]（角）格式，但是得到了 {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# 从 transformers.models.detr.modeling_detr._max_by_axis 复制而来
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# 从 transformers.models.detr.modeling_detr.NestedTensor
# 定义嵌套张量类
class NestedTensor(object):
    # 初始化方法，接受张量和可选的遮罩作为参数
    def __init__(self, tensors, mask: Optional[Tensor]):
        # 保存传入的张量和遮罩
        self.tensors = tensors
        self.mask = mask

    # 转换方法，将内部张量和遮罩移到指定设备上
    def to(self, device):
        # 将内部张量转移到指定设备上
        cast_tensor = self.tensors.to(device)
        # 如果存在遮罩，则也将遮罩转移到指定设备上，否则设置为None
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        # 返回转移后的嵌套张量
        return NestedTensor(cast_tensor, cast_mask)

    # 分解方法，返回内部张量和遮罩
    def decompose(self):
        return self.tensors, self.mask

    # 重写打印方法，返回内部张量的字符串表示
    def __repr__(self):
        return str(self.tensors)


# 从张量列表创建嵌套张量，来自transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # 如果张量列表中的第一个张量是3维的
    if tensor_list[0].ndim == 3:
        # 计算张量列表中张量的最大尺寸
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # 计算批次形状
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        # 获取张量的数据类型和设备
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # 创建全0张量和全1遮罩
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # 遍历张量列表，将张量拷贝到全0张量上并更新遮罩
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        # 如果张量不是3维的，抛出数值异常
        raise ValueError("Only 3-dimensional tensors are supported")
    # 返回创建的嵌套张量
    return NestedTensor(tensor, mask)
```