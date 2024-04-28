# `.\transformers\models\table_transformer\modeling_table_transformer.py`

```
# 设置文件编码为utf-8
# 版权声明，版权归 Microsoft Research 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证，版本 2.0 进行许可
# 除非在协议下明确要求或书面同意，否则您不得使用此文件
# 您可以从以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按照“原样”提供软件
# 没有任何种类的明示或默示担保或条件
# 请参阅协议，了解权限控制和限制
# PyTorch Table Transformer 模型

# 导入必要的模块
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

# 导入自定义的激活函数映射
from ...activations import ACT2FN
# 导入处理注意力掩码的工具函数
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
# 导入模型输出的基类
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
# 导入模型相关的工具函数和类
from ...modeling_utils import PreTrainedModel
# 导入辅助函数
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
# 自动加载不同模块
from ..auto import AutoBackbone
# 导入配置类
from .configuration_table_transformer import TableTransformerConfig

# 检测是否安装了 scipy 库，如果有，则导入
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 检测是否安装了 timm 库，如果有，则导入
if is_timm_available():
    from timm import create_model

# 检测是否安装了视觉库，如果有，则导入相关函数
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 模型文档使用的配置信息
_CONFIG_FOR_DOC = "TableTransformerConfig"
# 模型文档使用的 checkpoint 信息
_CHECKPOINT_FOR_DOC = "microsoft/table-transformer-detection"

# 预训练模型的列表
TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/table-transformer-detection",
    # 查看所有 Table Transformer 模型： https://huggingface.co/models?filter=table-transformer
]

# 定义数据类
@dataclass
# 基于 BaseModelOutputWithCrossAttentions 的 TABLE_TRANSFORMER 解码器的输出
class TableTransformerDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    TABLE_TRANSFORMER 解码器输出的基类。此类在 BaseModelOutputWithCrossAttentions 的基础上增加了一个属性，
    用于存储中间解���器激活层的堆栈，即每个解码器层的输出，每个输出都经过了一个层归一化。在使用辅助解码损失训练模型时很有用。
    # 定义函数参数，表示模型最后一层的隐藏状态
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        # 可选参数，表示模型隐藏状态，在配置中设置`output_hidden_states=True`或者`config.output_hidden_states=True`时返回
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        # 可选参数，表示注意力权重，在配置中设置`output_attentions=True`时返回
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        # 可选参数，表示跨层注意力权重，在配置中设置`output_attentions=True`和`config.add_cross_attention=True`时返回
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        # 可选参数，表示中间隐藏状态，在配置中设置`config.auxiliary_loss=True`时返回
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    # 表示中间隐藏状态，默认为None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
# 使用装饰器 @dataclass，定义了一个名为 TableTransformerModelOutput 的数据类，继承自 Seq2SeqModelOutput
class TableTransformerModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the TABLE_TRANSFORMER encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    """
    # 可选的中间隐藏状态，类型为 torch.FloatTensor，初始值为 None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None


# 使用装饰器 @dataclass，定义了一个名为 TableTransformerObjectDetectionOutput 的数据类，继承自 ModelOutput
class TableTransformerObjectDetectionOutput(ModelOutput):
    """
    Output type of [`TableTransformerForObjectDetection`].

    """
    # 可选的损失值，类型为 torch.FloatTensor，初始值为 None
    loss: Optional[torch.FloatTensor] = None
    # 可选的损失值字典，类型为 Dict，初始值为 None
    loss_dict: Optional[Dict] = None
    # 预测 logits，类型为 torch.FloatTensor，初始值为 None
    logits: torch.FloatTensor = None
    # 预测框位置，类型为 torch.FloatTensor，初始值为 None
    pred_boxes: torch.FloatTensor = None
    # 可选的辅助输出，类型为 List[Dict]，初始值为 None
    auxiliary_outputs: Optional[List[Dict]] = None
    # 最后的隐藏状态，类型为 torch.FloatTensor，初始值为 None
    last_hidden_state: Optional[torch.FloatTensor] = None
    # 解码器的隐藏状态，类型为 Tuple[torch.FloatTensor]，初始值为 None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力结果，类型为 Tuple[torch.FloatTensor]，初始值为 None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器-解码器的注意力结果，类型为 Tuple[torch.FloatTensor]，初始值为 None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的最终隐藏状态，类型为 torch.FloatTensor，初始值为 None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态，类型为 Tuple[torch.FloatTensor]，初始值为 None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力结果，类型为 Tuple[torch.FloatTensor]，初始值为 None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个名为 TableTransformerFrozenBatchNorm2d 的 nn.Module 类
class TableTransformerFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """
    # 初始化方法
    def __init__(self, n):
        # 父类初始化
        super().__init__()
        # 注册缓冲区，用于固定批处理统计和仿射参数
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 加载模型状态时的处理方法
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 删除状态字典中的 num_batches_tracked_key
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        # 调用父类的加载模型状态的方法
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
    def forward(self, x):
        # 将权重重塑为1行，-1列，1深度，1宽度
        # 以使其更加用户友好
        weight = self.weight.reshape(1, -1, 1, 1)
        # 将偏置重塑为1行，-1列，1深度，1宽度
        bias = self.bias.reshape(1, -1, 1, 1)
        # 将运行方差重塑为1行，-1列，1深度，1宽度
        running_var = self.running_var.reshape(1, -1, 1, 1)
        # 将运行均值重塑为1行，-1列，1深度，1宽度
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        # 设置一个很小的值 epsilon
        epsilon = 1e-5
        # 计算 scale，即权重乘以（运行方差加上 epsilon 后）的平方根的倒数
        scale = weight * (running_var + epsilon).rsqrt()
        # 计算偏置，即偏置减去运行均值乘以 scale
        bias = bias - running_mean * scale
        # 返回 x 经过 scale 和偏置处理后的结果
        return x * scale + bias
# 从transformers.models.detr.modeling_detr.replace_batch_norm复制代码，并将Detr改为TableTransformer
# 递归地将所有的torch.nn.BatchNorm2d替换为TableTransformerFrozenBatchNorm2d
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `TableTransformerFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        # 判断module是否为nn.BatchNorm2d的实例
        if isinstance(module, nn.BatchNorm2d):
            # 创建一个新的TableTransformerFrozenBatchNorm2d实例
            new_module = TableTransformerFrozenBatchNorm2d(module.num_features)

            # 如果module的weight不在"meta"设备上，则将new_module的参数拷贝给new_module
            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            # 将new_module替换原来的module
            model._modules[name] = new_module

        # 如果module有子模块，则继续递归替换
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


# 从transformers.models.detr.modeling_detr.DetrConvEncoder复制代码，并将Detr改为TableTransformer
class TableTransformerConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by TableTransformerFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.use_timm_backbone:
            # 如果配置中使用timm backbone，则添加timm依赖
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            # 根据配置创建timm backbone模型
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            # 如果配置中使用AutoBackbone，则根据配置创建AutoBackbone模型
            backbone = AutoBackbone.from_config(config.backbone_config)

        # 使用TableTransformerFrozenBatchNorm2d替换batch norm层
        with torch.no_grad():
            replace_batch_norm(backbone)

        self.model = backbone
        # 获取backbone模型的通道数信息，并赋值给self.intermediate_channel_sizes
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if "resnet" in backbone_model_type:
            # 设置backbone模型部分参数的梯度为不可训练的
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)
    # 定义一个名为 forward 的方法，接受两个参数 pixel_values 和 pixel_mask，都是 torch.Tensor 类型
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # 如果配置中使用了 timm backbone，则将 pixel_values 通过模型传递，得到特征图列表；否则直接通过模型得到特征图列表
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps
        
        # 创建一个空列表 out 用于存储处理后的特征图和对应的 mask
        out = []
        # 遍历特征图列表
        for feature_map in features:
            # 将 pixel_mask 下采样（插值）到与对应特征图相同的形状
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            # 将处理后的特征图和对应的 mask 添加到 out 列表
            out.append((feature_map, mask))
        # 返回处理后的特征图及对应的 mask 列表
        return out
# 从transformers.models.detr.modeling_detr.DetrConvModel复制而来，将Detr->TableTransformer
class TableTransformerConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):
        # 通过骨干网络将像素值和像素掩码传递，得到（特征图，像素掩码）元组列表
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # 位置编码
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


# 从transformers.models.detr.modeling_detr.DetrSinePositionEmbedding复制而来，将Detr->TableTransformer
class TableTransformerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# 从transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding复制而来，将Detr->TableTransformer
class TableTransformerLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, embedding_dim=256):
        # 子类初始化函数，继承父类初始化方法
        super().__init__()
        # 创建一个大小为(50, embedding_dim)的Embedding层，用于存储行序号的嵌入向量
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        # 创建一个大小为(50, embedding_dim)的Embedding层，用于存储列序号的嵌入向量

    def forward(self, pixel_values, pixel_mask=None):
        # 定义前向传播函数
        height, width = pixel_values.shape[-2:]
        # 获取像素值矩阵的高和宽
        width_values = torch.arange(width, device=pixel_values.device)
        # 创建一个大小为width的一维张量，设备为pixel_values的设备
        height_values = torch.arange(height, device=pixel_values.device)
        # 创建一个大小为height的一维张量，设备为pixel_values的设备
        x_emb = self.column_embeddings(width_values)
        # 获取列序号的嵌入向量
        y_emb = self.row_embeddings(height_values)
        # 获取行序号的嵌入向量
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # 在最后维度上将行和列的嵌入向量进行拼接
        pos = pos.permute( 2, 0, 1)
        # 调换pos的维度顺序
        pos = pos.unsqueeze(0)
        # 在第0维度上增加一个维度
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # 在第0维度上重复360次
        return pos
# 从transformers.models.detr.modeling_detr.build_position_encoding中复制代码，并将Detr->TableTransformer
def build_position_encoding(config):
    # 根据配置计算位置编码的步数
    n_steps = config.d_model // 2
    # 如果位置嵌入类型为"sine"
    if config.position_embedding_type == "sine":
        # TODO 找到一个更好的方法来暴露其他参数
        # 创建TableTransformerSinePositionEmbedding对象，传入步数和normalize参数
        position_embedding = TableTransformerSinePositionEmbedding(n_steps, normalize=True)
    # 如果位置嵌入类型为"learned"
    elif config.position_embedding_type == "learned":
        # 创建TableTransformerLearnedPositionEmbedding对象，传入步数
        position_embedding = TableTransformerLearnedPositionEmbedding(n_steps)
    else:
        # 抛出数值错误，表示不支持的位置嵌入类型
        raise ValueError(f"Not supported {config.position_embedding_type}")

    # 返回位置编码对象
    return position_embedding

# 从transformers.models.detr.modeling_detr.DetrAttention中复制代码，并将DETR->TABLE_TRANSFORMER,Detr->TableTransformer
class TableTransformerAttention(nn.Module):
    """
    从'Attention Is All You Need'论文中的多头注意力机制。
    
    在这里，我们将位置嵌入添加到查询和键中（如TABLE_TRANSFORMER论文中所述）。
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
            # 如果embed_dim不能被num_heads整除，抛出数值错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        # 初始化线性映射矩阵
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 定义一个方法用于改变张量的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 用于添加位置嵌入到张量中，object_queries参数可选
    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)

        if kwargs:
            # 如果有未知的参数，则抛出数值错误
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            # 不能同时指定position_embeddings和object_queries，抛出数值错误
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if position_embeddings is not None:
            # 弃用position_embeddings属性，并发出警告
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        # 如果object_queries为None，则返回原张量；否则返回原张量加上object_queries
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



# 预测或运行模型时的前向传播函数
def forward(
    self,
    # 输入的隐藏状态张量
    hidden_states: torch.Tensor,
    # 可选的注意力掩码张量
    attention_mask: Optional[torch.Tensor] = None,
    # 可选的物体查询张量
    object_queries: Optional[torch.Tensor] = None,
    # 可选的键值状态张量
    key_value_states: Optional[torch.Tensor] = None,
    # 可选的空间位置嵌入张量
    spatial_position_embeddings: Optional[torch.Tensor] = None,
    # 是否输出注意力得分张量
    output_attentions: bool = False,
    # 其他参数
    **kwargs,
class TableTransformerEncoderLayer(nn.Module):
    # 从transformers.models.detr.modeling_detr.DetrEncoderLayer.__init__复制而来，将Detr替换为TableTransformer
    def __init__(self, config: TableTransformerConfig):
        super().__init__()
        self.embed_dim = config.d_model
        # 初始化self-attention层
        self.self_attn = TableTransformerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 对self-attention输出进行LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        # 激活函数使用配置中指定的函数
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        # 第一个全连接层，将输入维度转换为config中指定的encoder_ffn_dim维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 第二个全连接层，将维度转换回self.embed_dim
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 对最终输出进行LayerNorm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数
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
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`torch.FloatTensor`, *optional*): object queries, to be added to hidden_states.
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参见`返回的张量`下的`attentions`。
        """
        # 保存残差连接
        residual = hidden_states
        # 对输入进行 Layer Norm 处理
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 使用自注意力机制
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        # 使用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 经过残差连接
        hidden_states = residual + hidden_states

        # 保存残差连接
        residual = hidden_states
        # 再次进行 Layer Norm 处理
        hidden_states = self.final_layer_norm(hidden_states)

        # 使用激活函数
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 再次使用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        # 使用全连接层
        hidden_states = self.fc2(hidden_states)
        # 使用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 经过残差连接
        hidden_states = residual + hidden_states

        # 在训练模式下，处理隐藏状态中的无穷大和 NaN 值
        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 准备输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义表格转换器解码器层
class TableTransformerDecoderLayer(nn.Module):
    # 从transformers.models.detr.modeling_detr.DetrDecoderLayer.__init__中复制过来，将Detr->TableTransformer
    def __init__(self, config: TableTransformerConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 创建self-attention模块
        self.self_attn = TableTransformerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 创建Layer normalization模块
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 创建encoder-attention模块
        self.encoder_attn = TableTransformerAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 创建两个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
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
    )

# 从transformers.models.detr.modeling_detr.DetrClassificationHead中复制过来，将Detr->TableTransformer
# 类别头用于句子级别的分类任务
class TableTransformerClassificationHead(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        # 创建线性层
        self.dense = nn.Linear(input_dim, inner_dim)
        # 创建dropout层
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 创建输出线性层
        self.out_proj = nn.Linear(inner_dim, num_classes)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        # 使用双曲正切函数作为激活函数
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class TableTransformerPreTrainedModel(PreTrainedModel):
    config_class = TableTransformerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [
        r"TableTransformerConvEncoder",
        r"TableTransformerEncoderLayer",
        r"TableTransformerDecoderLayer",
    ]
    # 初始化模型参数权重
    def _init_weights(self, module):
        # 从配置中获取初始化的标准差
        std = self.config.init_std

        # 如果模块是 TableTransformerLearnedPositionEmbedding 类型
        if isinstance(module, TableTransformerLearnedPositionEmbedding):
            # 对行和列的嵌入权重进行均匀分布初始化
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        
        # 如果模块是 nn.Linear, nn.Conv2d, nn.BatchNorm2d 中的任意一种
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 使用正态分布随机初始化权重，标准差为设定值
            # 与 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，将偏置项初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        
        # 如果模块是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布随机初始化权重，标准差为设定值
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# 表格转换器类的文档字符串起始部分，提供了关于该模型的继承、用法和参数信息
TABLE_TRANSFORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TableTransformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 表格转换器类的输入文档字符串部分，暂未提供具体信息
TABLE_TRANSFORMER_INPUTS_DOCSTRING = r"""
    # 定义函数的参数列表及其类型
    Args:
        # 图像的像素数值，形状为(batch_size, num_channels, height, width)
        # 默认情况下会忽略填充值
        # 可以使用DetrImageProcessor获取像素值。详情请参考DetrImageProcessor.__call__
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        
        # 用于避免在填充像素值上执行注意力操作的掩码
        # 掩码值选在[0, 1]范围内：
        #   - 1表示真实像素（即未被掩盖）
        #   - 0表示填充像素（即被掩盖）
        #   什么是注意力掩码?请查看glossary#attention-mask
        pixel_mask (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional*):
        
        # 用于掩盖对象查询的掩码，默认情况下不使用
        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
        
        # 编码器输出，为一个元组，包括(last_hidden_state, hidden_states, attentions)，默认情况下可选
        # last_hidden_state的形状为(batch_size, sequence_length, hidden_size)，是编码器最后一层的隐藏状态序列，用于解码器的交叉注意力
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
        
        # 输入嵌入，形状为(batch_size, sequence_length, hidden_size)，默认情况下可选
        # 可选择直接传递图像的平展特征图（backbone + projection层的输出），而不是传递它
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        
        # 解码器输入嵌入，形状为(batch_size, num_queries, hidden_size)，默认情况下可选
        # 可选择直接传递一个嵌入表示来初始化查询，而不是使用零张量
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
        
        # 是否返回所有注意层的注意张量，默认情况下可选
        # 更多详细信息，请查看返回的张量下面的注意
        output_attentions (`bool`, *optional*):
        
        # 是否返回所有层的隐藏状态，默认情况下可选
        # 更多详细信息，请查看返回的张量下面的隐藏状态
        output_hidden_states (`bool`, *optional*):
        
        # 是否返回ModelOutput而不是纯元组，默认情况下可选
        return_dict (`bool`, *optional*):
"""
# Table Transformer 编码器，由多个自注意力层组成，每一层都是 TableTransformerEncoderLayer
class TableTransformerEncoder(TableTransformerPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TableTransformerEncoderLayer`]。

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for Table Transformer:

    - object_queries are added to the forward pass.

    Args:
        config: TableTransformerConfig
    """

    def __init__(self, config: TableTransformerConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 初始化 Encoder 层的列表
        self.layers = nn.ModuleList([TableTransformerEncoderLayer(config) for _ in range(config.encoder_layers)])

        self.layernorm = nn.LayerNorm(config.d_model)

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
# 从 transformers.models.detr.modeling_detr 中拷贝代码，将 DETR->TABLE_TRANSFORMER，Detr->TableTransformer
class TableTransformerDecoder(TableTransformerPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TableTransformerDecoderLayer`]。

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for TABLE_TRANSFORMER:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: TableTransformerConfig
    """

    def __init__(self, config: TableTransformerConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        # 初始化 Decoder 层的列表
        self.layers = nn.ModuleList([TableTransformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 在 TABLE_TRANSFORMER 中，decoder 在最后一个 decoder 层输出后使用 layernorm
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
@add_start_docstrings(
    """
    The bare Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    TABLE_TRANSFORMER_START_DOCSTRING,
)
class TableTransformerModel(TableTransformerPreTrainedModel):
    # 从transformers.models.detr.modeling_detr.DetrModel.__init__中复制代码，用TableTransformer替代Detr
    def __init__(self, config: TableTransformerConfig):
        # 调用父类的构造函数
        super().__init__(config)

        # 创建backbone + 位置编码
        backbone = TableTransformerConvEncoder(config)
        object_queries = build_position_encoding(config)
        self.backbone = TableTransformerConvModel(backbone, object_queries)

        # 创建投影层
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)

        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        self.encoder = TableTransformerEncoder(config)
        self.decoder = TableTransformerDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 冻结backbone的权重
    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    # 解冻backbone的权重
    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    # 前向传播函数
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
@add_start_docstrings(
    """
    Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    TABLE_TRANSFORMER_START_DOCSTRING,
)
class TableTransformerForObjectDetection(TableTransformerPreTrainedModel):
    # 从transformers.models.detr.modeling_detr.DetrForObjectDetection.__init__中复制代码，用TableTransformer替代Detr
    # 初始化函数，接受一个TableTransformerConfig对象作为参数
    def __init__(self, config: TableTransformerConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 初始化TableTransformerModel对象作为属性
        self.model = TableTransformerModel(config)

        # 初始化对象检测头部的分类器线性层，输出维度为config.num_labels + 1，用于添加"无对象"类别
        self.class_labels_classifier = nn.Linear(
            config.d_model, config.num_labels + 1
        )
        
        # 初始化对象检测头部的边界框预测器，输入维度为config.d_model，隐藏层维度为config.d_model，输出维度为4，层数为3
        self.bbox_predictor = TableTransformerMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        # 调用post_init函数，用于初始化权重和进行最终处理
        self.post_init()

    @torch.jit.unused
    # 从transformers.models.detr.modeling_detr.DetrForObjectDetection._set_aux_loss复制得到的函数
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是为了让torchscript能够正常工作的一种解决方法，因为torchscript不支持具有非同质值的字典，比如一个同时包含Tensor和列表的字典
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # 调用add_start_docstrings_to_model_forward与replace_return_docstrings装饰器，对forward函数进行修饰
    @add_start_docstrings_to_model_forward(TABLE_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TableTransformerObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，接受多个输入参数
    def forward(
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
# 从transformers.models.detr.modeling_detr.dice_loss复制而来
def dice_loss(inputs, targets, num_boxes):
    """
    计算DICE损失，类似于面向掩模的广义IOU

    Args:
        inputs: 任意形状的浮点张量。
                每个示例的预测结果。
        targets: 与inputs相同形状的浮点张量。
                 存储每个元素的二元分类标签 (0表示负类，1表示正类)。
        num_boxes: 输入的框数量
    """
    # 将输入经过sigmoid函数激活
    inputs = inputs.sigmoid()
    # 将inputs展平为二维张量
    inputs = inputs.flatten(1)
    # 计算DICE系数的分子
    numerator = 2 * (inputs * targets).sum(1)
    # 计算DICE系数的分母
    denominator = inputs.sum(-1) + targets.sum(-1)
    # 计算DICE损失
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 返回DICE损失的均值
    return loss.sum() / num_boxes


# 从transformers.models.detr.modeling_detr.sigmoid_focal_loss复制而来
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    RetinaNet中使用的损失函数：https://arxiv.org/abs/1708.02002。

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape): 每个示例的预测结果。
        targets (`torch.FloatTensor`与`inputs`形状相同): 存储每个输入元素的二元分类标签 (0表示负类，1表示正类)。
        alpha (`float`, *optional*, 默认为 `0.25`): 用于平衡正类和负类的可选权重因子。
        gamma (`int`, *optional*, 默认为 `2`): 调节因子 (1 - p_t) 的指数，用于平衡简单和困难示例。

    Returns:
        损失张量
    """
    # 对输入应用sigmoid激活函数
    prob = inputs.sigmoid()
    # 计算交叉熵损失
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 添加调节因子
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# 从transformers.models.detr.modeling_detr.DetrLoss复制而来，Detr->TableTransformer,detr->table_transformer
class TableTransformerLoss(nn.Module):
    """
    该类计算TableTransformerForObjectDetection/TableTransformerForSegmentation的损失。过程分为两步：1)
    我们计算ground truth框与模型输出之间的匈牙利分配 2) 监督每对匹配的ground-truth/prediction (监督类别和框)。

    关于`num_classes`参数的注释（从table_transformer.py中复制）："损失函数中`num_classes`参数的命名有些误导性。
    它实际上对应于`max_obj_id` + 1，其中`max_obj_id`是数据集中类别的最大ID。例如，COCO数据集的`max_obj_id`为90，
    因此我们将`num_classes`传递给
    """
    # 定义构造函数
    def __init__(self, num_classes, matcher, bbox_criterion, segmentation_criterion):
        # 调用父类的构造函数
        super().__init__()
        # 初始化类别数量
        self.num_classes = num_classes
        # 初始化匹配器
        self.matcher = matcher
        # 初始化bbox损失
        self.bbox_criterion = bbox_criterion
        # 初始化分割损失
        self.segmentation_criterion = segmentation_criterion

    # 定义前向传播函数
    def forward(self, outputs, targets):
        # 进行匹配
        indices = self.matcher(outputs, targets)
        # 计算bbox损失
        loss_bbox = self.bbox_criterion(outputs['pred_boxes'][indices], targets['boxes'][indices])
        # 计算分割损失
        loss_seg = self.segmentation_criterion(outputs['pred_masks'], targets['masks'])
        # 返回总损失
        return loss_bbox + loss_seg
    """
    初始化函数，初始化匹配器、类别数量、相对分类权重和损失函数列表
    Args:
        matcher (`TableTransformerHungarianMatcher`):
            能够计算目标和提议之间匹配的模块
        num_classes (`int`):
            目标分类的数量，不包括特殊的非目标类别
        eos_coef (`float`):
            相对分类权重，应用于非目标类别
        losses (`List[str]`):
            要应用的所有损失的列表。有关所有可用损失的列表，请参阅 `get_loss`
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

    # 从原始实现中删除了日志参数
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        分类损失（NLL），目标字典必须包含键“class_labels”，其中包含维数为[nb_target_boxes]的张量
        """
        if "logits" not in outputs:
            raise KeyError("在输出中未找到logits")
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
        计算基数错误，即预测的非空框数量的绝对误差。
        
        这实际上不是一个损失，仅用于记录目的。不会传播梯度。
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算不是“非目标”（即最后一个类别）的预测数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses
    # 计算与边界框相关的损失，包括 L1 回归损失和 GIoU 损失
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # 如果输出中找不到预测的边界框，则引发 KeyError
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        # 获取源排列索引
        idx = self._get_source_permutation_idx(indices)
        # 获取预测的边界框和目标边界框
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算 L1 损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        # 损失字典，包含边界框损失
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算 GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    # 计算与掩码相关的损失，包括焦点损失和 Dice 损失
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        # 如果输出中找不到预测的掩码，则引发 KeyError
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        # 获取源和目标排列索引
        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # 将预测的掩码插值到目标大小
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        # 损失字典，包含掩码损失和 Dice 损失
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    # 获取源排列索引
    def _get_source_permutation_idx(self, indices):
        # 根据索引重新排列预测
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx
    def _get_target_permutation_idx(self, indices):
        # 根据提供的索引重新排列目标数据
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        # 定义损失函数与对应的处理函数的映射关系
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        # 调用对应损失函数的处理函数
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
        # 从输出中排除辅助输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # 获取最后一层输出与目标之间的匹配关系
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点上目标框的平均数，用于归一化
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): 注释掉下面的函数调用，将在添加分布式训练时添加
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) 在原始实现中，num_boxes 会除以 get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 对于辅助损失，重复上述过程，针对每个中间层的输出
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # 中间层的 masks 损失计算成本过高，忽略它们
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
# 从transformers.models.detr.modeling_detr.DetrMLPPredictionHead复制而来，将Detr->TableTransformer，detr->table_transformer
class TableTransformerMLPPredictionHead(nn.Module):
    """
    非常简单的多层感知器（MLP，也称为FFN），用于预测边界框相对于图像的归一化中心坐标、高度和宽度。

    从https://github.com/facebookresearch/table_transformer/blob/master/models/table_transformer.py复制而来

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 创建包含多个线性层的模块列表，用于多层感知器的构建
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 对于前num_layers-1层，使用ReLU作为激活函数，最后一层不使用激活函数
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 从transformers.models.detr.modeling_detr.DetrHungarianMatcher复制而来，将Detr->TableTransformer
class TableTransformerHungarianMatcher(nn.Module):
    """
    该类计算网络的预测与目标之间的分配。

    由于效率原因，目标不包括no_object。因此，一般情况下，预测数量比目标多。
    在这种情况下，我们对最佳预测进行一对一匹配，而其他预测则未匹配（因此被视为非对象）。

    Args:
        class_cost:
            匹配成本中分类错误的相对权重。
        bbox_cost:
            匹配成本中边界框坐标的L1误差的相对权重。
        giou_cost:
            匹配成本中边界框的giou损失的相对权重。
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        # 检查是否导入了scipy
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # 如果Matcher的所有成本都为0，则引发值错误
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
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # 使用 TensorFlow 的 flatten 方法将 logits tensor 展平，以便在批处理中计算成本矩阵
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # 使用 TensorFlow 的 flatten 方法将 pred_boxes tensor 展平
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # 连接目标标签和框
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        # 计算框之间的 L1 成本
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        # 计算框之间的 giou 成本
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        # 最终成本矩阵
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        # 将成本矩阵调整成 batch_size, num_queries, -1 的形状，并移到 CPU 上
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # 分别取出每个目标的大小和成本矩阵，利用 linear_sum_assignment 方法计算索引
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        # 返回索引的列表，每个索引采用 torch.as_tensor 转换成 tensor，并指定 dtype 为 torch.int64
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# Defines a function to upcast a tensor to protect from numerical overflows
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Defines a function to compute the area of a set of bounding boxes
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
    # Upcast the input boxes to protect from numerical overflows
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Defines a function to compute the IoU (Intersection over Union) of two sets of bounding boxes
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


# Defines a function to compute the generalized IoU of two sets of bounding boxes
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # Perform early checks for degenerate boxes to avoid inf / nan results
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


# Defines a function to find the maximum value across each axis of a list of lists
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# Copies the code from transformers.models.detr.modeling_detr.NestedTensor
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        # 初始化函数，接受一个张量列表和一个可选的掩码张量作为参数
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # 将张量列表和掩码张量移动到指定的设备
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        # 返回张量列表和掩码张量
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# 从transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list复制而来
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        # 获取张量列表中张量的最大尺寸作为批量尺寸，并创建对应形状的零张量和掩码张量
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # 将原始张量复制到零张量上，并更新掩码张量
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        # 如果张量维度不为3，抛出异常
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)
```