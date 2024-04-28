# `.\transformers\models\siglip\modeling_siglip.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件，包括但不限于特定用途的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
""" PyTorch Siglip 模型。"""

# 导入所需的库
import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out

# 导入自定义的模块和函数
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "google/siglip-base-patch16-224"

# 预训练模型的存档列表
SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/siglip-base-patch16-224",
    # 查看所有 SigLIP 模型 https://huggingface.co/models?filter=siglip
]

# 截断正态分布函数
def _trunc_normal_(tensor, mean, std, a, b):
    # 从 PyTorch 官方 master 分支复制粘贴，直到它在几个官方版本中发布 - RW
    # 方法基于 https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # 计算标准正态分布的累积分布函数
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # 通过使用截断均匀分布生成值，然后使用正态分布的逆 CDF。
    # 获取上下界的累积分布函数值
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # 使用 [l, u] 范围内的值均匀填充张量，然后转换为 [2l-1, 2u-1] 范围
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # 使用正态分布的逆 CDF 转换为截断标准正态分布
    tensor.erfinv_()

    # 转换为正确的均值和标准差
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # 确保在正确范围内
    tensor.clamp_(min=a, max=b)
    # 定义函数参数，tensor为torch.Tensor类型，mean默认值为0.0，std默认值为1.0，a默认值为-2.0，b默认值为2.0
# 定义一个函数，用截断正态分布填充输入的张量
def _trunc_normal_(tensor, mean, std, a, b) -> torch.Tensor:
    """
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution N(mean, std^2) with values outside [a, b] redrawn until they are within
    the bounds. The method used for generating the random values works
    best when a <= mean <= b.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    # 使用 torch.no_grad() 上下文管理器，确保不会进行梯度计算
    with torch.no_grad():
        # 从标准正态分布 N(0, 1.0) 中采样，然后将结果缩放和平移以匹配给定的均值和标准差
        _trunc_normal_(tensor, 0, 1.0, a, b)
        # 将张量乘以标准差并加上均值
        tensor.mul_(std).add_(mean)


# 定义一个函数，用于初始化张量的方差缩放
def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    # 计算张量的输入和输出通道数
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    # 根据模式选择分母
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    # 计算方差
    variance = scale / denom

    if distribution == "truncated_normal":
        # 使用截断正态分布填充张量
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        # 使用正态分布填充张量
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        # 使用均匀分布填充张量
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


# 定义一个函数，使用 LeCun 初始化方法填充张量
def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


# 定义一个函数，使用默认的 Flax 嵌入初始化方法填充张量
def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


# 定义一个数据类，用于表示 Siglip 视觉模型的输出
@dataclass
# 从 transformers.models.clip.modeling_clip.CLIPVisionModelOutput 复制并将 CLIP 改为 Siglip
class SiglipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    """
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义变量image_embeds，类型为torch.FloatTensor，可选参数，默认为None
    image_embeds: Optional[torch.FloatTensor] = None
    # 定义变量last_hidden_state，类型为torch.FloatTensor，必需参数
    last_hidden_state: torch.FloatTensor = None
    # 定义变量hidden_states，类型为tuple(torch.FloatTensor)，可选参数，默认为None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义变量attentions，类型为tuple(torch.FloatTensor)，可选参数，默认为None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器定义 SiglipTextModelOutput 类，该类继承自 ModelOutput
# 用于表示文本模型的输出，并包含最后隐藏状态的池化结果
class SiglipTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.
    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """
    # 定义类的属性，包括 text_embeds、last_hidden_state、hidden_states、attentions
    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# 使用 dataclass 装饰器定义 SiglipOutput 类
class SiglipOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`SiglipTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`SiglipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`SiglipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`SiglipVisionModel`].
    """

    # 初始化属性
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    # 将属性值转换为元组
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果属性值不在 ["text_model_output", "vision_model_output"] 中，则直接返回该属性值
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 定义名为SiglipVisionEmbeddings的类，继承自nn.Module
class SiglipVisionEmbeddings(nn.Module):
    # 初始化方法，接收一个config参数，类型为SiglipVisionConfig
    def __init__(self, config: SiglipVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将config赋值给self.config
        self.config = config
        # 将config.hidden_size赋值给self.embed_dim
        self.embed_dim = config.hidden_size
        # 将config.image_size赋值给self.image_size
        self.image_size = config.image_size
        # 将config.patch_size赋值给self.patch_size
        self.patch_size = config.patch_size

        # 创建一个二维卷积层，设置输入通道数为config.num_channels，输出通道数为self.embed_dim，卷积核大小为self.patch_size，步长为self.patch_size，填充方式为"valid"
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        # 计算图像分成的块数，并赋值给self.num_patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 将self.num_patches赋值给self.num_positions
        self.num_positions = self.num_patches
        # 创建一个位置嵌入层，设置输入大小为self.num_positions，输出大小为self.embed_dim
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 将torch.arange(self.num_positions).expand((1, -1))的结果作为缓冲区的数据，并设置为非持久
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # 前向传播方法，接收一个名为pixel_values的torch.FloatTensor参数，返回一个torch.Tensor
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 对pixel_values进行patch嵌入，得到patch_embeds，形状为[*, width, grid, grid]
        patch_embeds = self.patch_embedding(pixel_values)
        # 将patch_embeds按照第2个维度展平，并转置第1和第2个维度，得到embeddings
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        # 将embeddings加上位置嵌入的结果，得到最终的嵌入结果
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # 返回嵌入结果
        return embeddings


# 从transformers.models.clip.modeling_clip.CLIPTextEmbeddings中复制而来，修改CLIP为Siglip
# 定义名为SiglipTextEmbeddings的类，继承自nn.Module
class SiglipTextEmbeddings(nn.Module):
    # 初始化方法，接收一个config参数，类型为SiglipTextConfig
    def __init__(self, config: SiglipTextConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 从config中获取hidden_size，并赋值给embed_dim
        embed_dim = config.hidden_size

        # 创建一个词嵌入层，设置词表大小为config.vocab_size，嵌入维度为embed_dim
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 创建一个位置嵌入层，设置输入大小为config.max_position_embeddings，输出大小为embed_dim
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 将torch.arange(config.max_position_embeddings).expand((1, -1))的结果作为缓冲区的数据，并设置为非持久
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # 前向传播方法，接收三个可选的参数input_ids、position_ids、inputs_embeds，返回一个torch.Tensor
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 如果input_ids不为空，则获取其最后一个维度的大小作为seq_length
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        # 如果position_ids为空，则将self.position_ids的前seq_length列赋值给position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果inputs_embeds为空，则使用token_embedding对input_ids进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 获取位置嵌入结果
        position_embeddings = self.position_embedding(position_ids)
        # 将词嵌入和位置嵌入结果相加，得到最终的嵌入结果
        embeddings = inputs_embeds + position_embeddings

        # 返回嵌入结果
        return embeddings


# 定义名为SiglipAttention的类，继承自nn.Module
class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 从transformers.models.clip.modeling_clip.CLIPAttention.__init__中复制而来
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置对象到实例属性中
        self.config = config
        # 设置嵌入维度等于配置中的隐藏层大小
        self.embed_dim = config.hidden_size
        # 设置注意力头的数量等于配置中的注意力头数量
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 如果嵌入维度不能被注意力头数量整除，抛出数值错误异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scale = self.head_dim**-0.5
        # 设置注意力层的丢弃率
        self.dropout = config.attention_dropout

        # 初始化线性层，用于将输入向量投影到不同空间
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 前向传播函数，用于计算自注意力
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩张量，默认为空
        output_attentions: Optional[bool] = False,  # 是否输出注意力分数，默认为False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """定义函数签名和文档字符串，指明函数参数和返回类型"""

        # 获取输入张量的形状信息
        batch_size, q_len, _ = hidden_states.size()

        # 通过线性变换将隐藏状态投影到查询空间
        query_states = self.q_proj(hidden_states)
        # 通过线性变换将隐藏状态投影到键空间
        key_states = self.k_proj(hidden_states)
        # 通过线性变换将隐藏状态投影到值空间
        value_states = self.v_proj(hidden_states)

        # 重新组织张量形状以便进行多头自注意力计算
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 获取键的序列长度
        k_v_seq_len = key_states.shape[-2]
        # 计算注意力权重
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # 检查注意力权重的形状是否符合预期
        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 如果存在注意力掩码，则将其应用于注意力权重
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # 将注意力权重转换为浮点数类型，并应用 softmax 归一化
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 使用 dropout 进行注意力权重的随机失活
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # 根据注意力权重计算注意力输出
        attn_output = torch.matmul(attn_weights, value_states)

        # 检查注意力输出的形状是否符合预期
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 重新组织注意力输出的形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        # 通过线性变换将注意力输出投影到最终输出空间
        attn_output = self.out_proj(attn_output)

        # 返回注意力输出和注意力权重
        return attn_output, attn_weights
```  
# 从transformers.models.clip.modeling_clip.CLIPMLP复制过来，将CLIP更改为Siglip
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # 使用配置中的隐藏层激活函数
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 第一个全连接层
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 第二个全连接层

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 输入经过第一个全连接层
        hidden_states = self.activation_fn(hidden_states)  # 激活函数
        hidden_states = self.fc2(hidden_states)  # 经过第二个全连接层
        return hidden_states


# 从transformers.models.clip.modeling_clip.CLIPEncoderLayer复制过来，将CLIP更改为Siglip
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)  # 自注意力机制
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第一个层归一化
        self.mlp = SiglipMLP(config)  # 多层感知机
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第二个层归一化

    # 忽略复制
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                输入到该层的张量，形状为`(batch, seq_len, embed_dim)`。
            attention_mask (`torch.FloatTensor`):
                形状为`(batch, 1, q_len, k_v_seq_len)`的注意力掩码，其中填充元素由非常大的负值指示。
            output_attentions (`bool`, *可选*, 默认为`False`):
                是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量下的`attentions`。
        """
        residual = hidden_states  # 残差连接

        hidden_states = self.layer_norm1(hidden_states)  # 第一个层归一化
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )  # 自注意力机制
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 残差连接
        hidden_states = self.layer_norm2(hidden_states)  # 第二个层归一化
        hidden_states = self.mlp(hidden_states)  # 多层感知机
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)  # 输出

        if output_attentions:  # 如果需要输出注意力张量
            outputs += (attn_weights,)  # 添加注意力张量到输出元组

        return outputs  # 返回输出元组


class SiglipPreTrainedModel(PreTrainedModel):
    """
    一个用于处理权重初始化和下载/加载预训练模型的简单接口的抽象类。
    """

    config_class = SiglipConfig  # 配置类为SiglipConfig
    base_model_prefix = "siglip"  # 基本模型前缀为"siglip"
    supports_gradient_checkpointing = True  # 支持梯度检查点
    # 初始化模型参数的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是 SiglipVisionEmbeddings 模块
        if isinstance(module, SiglipVisionEmbeddings):
            # 根据配置文件确定隐藏层大小
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, SiglipConfig)
                else self.config.hidden_size
            )
            # 对位置嵌入权重进行标准差为 1/sqrt(width) 的正态分布初始化
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        # 如果是 nn.Embedding 模块
        elif isinstance(module, nn.Embedding):
            # 使用默认的 Flax 嵌入初始化函数
            default_flax_embed_init(module.weight)
        # 如果是 SiglipAttention 模块
        elif isinstance(module, SiglipAttention):
            # 使用 Xavier 均匀分布初始化查询、键、值和输出投影权重
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            # 将查询、键、值投影的偏置初始化为零
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        # 如果是 SiglipMLP 模块
        elif isinstance(module, SiglipMLP):
            # 使用 Xavier 均匀分布初始化 MLP 的两个全连接层权重
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            # 使用标准差为 1e-6 的正态分布初始化 MLP 的两个全连接层偏置
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        # 如果是 SiglipMultiheadAttentionPoolingHead 模块
        elif isinstance(module, SiglipMultiheadAttentionPoolingHead):
            # 使用 Xavier 均匀分布初始化 Probe 和注意力模块的输入投影权重
            nn.init.xavier_uniform_(module.probe.data)
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            # 将注意力模块的输入投影偏置初始化为零
            nn.init.zeros_(module.attention.in_proj_bias.data)
        # 如果是 SiglipModel 模块
        elif isinstance(module, SiglipModel):
            # 使用对数函数初始化模型的 logit_scale 参数为 1.0，logit_bias 参数为零
            logit_scale_init = torch.log(torch.tensor(1.0))
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()
        # 如果是 nn.Linear 或 nn.Conv2d 模块
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用 LeCun 正态分布初始化权重，并将偏置初始化为零
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # 如果是 nn.LayerNorm 模块
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 模块的偏置初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
```  
# 定义模型的起始文档字符串，包含继承关系说明、参数说明等
SIGLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SiglipConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义文本输入的文档字符串，包含参数说明、数据类型等
SIGLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义视觉输入的文档字符串
SIGLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下，将忽略填充。您可以使用 [`AutoImageProcessor`] 获取像素值。有关详细信息，请参见 [`CLIPImageProcessor.__call__`]。
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# SIGLIP 输入的文档字符串
SIGLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下会忽略填充。

            可以使用 [`AutoTokenizer`] 来获取索引。详细信息请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            遮蔽掩码，避免对填充的标记进行注意力计算。掩码值选在 `[0, 1]` 范围内:

            - 1 表示**未被遮蔽**的标记，
            - 0 表示**被遮蔽**的标记。

            [什么是注意力掩码?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            输入序列各标记在位置嵌入中的位置索引。取值范围为 `[0, config.max_position_embeddings - 1]`。

            [什么是位置 ID?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下会忽略填充。可以使用 [`AutoImageProcessor`] 来获取像素值。详细信息请参阅 [`CLIPImageProcessor.__call__`]。
        return_loss (`bool`, *optional*):
            是否返回对比损失。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""


# 从transformers.models.clip.modeling_clip.CLIPEncoder中复制，将CLIP->Siglip
class SiglipEncoder(nn.Module):
    """
    由`config.num_hidden_layers`个自注意力层组成的Transformer编码器。每一层都是[`SiglipEncoderLayer`]。

    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        # 创建模块列表，包含`config.num_hidden_layers`个`SiglipEncoderLayer`
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # 忽略复制
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class SiglipTextTransformer(nn.Module):
    # 初始化函数，接受一个 SiglipTextConfig 类型的参数
    def __init__(self, config: SiglipTextConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置类属性 config 为输入的 config 参数
        self.config = config
        # 从 config 中获取隐藏层的维度，作为嵌入层的维度
        embed_dim = config.hidden_size
        # 创建 SiglipTextEmbeddings 实例，用于文本嵌入
        self.embeddings = SiglipTextEmbeddings(config)
        # 创建 SiglipEncoder 实例，用于编码器
        self.encoder = SiglipEncoder(config)
        # 创建 LayerNorm 层，用于对最终输出进行归一化
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 创建线性层，用于生成最终输出的线性变换

        self.head = nn.Linear(embed_dim, embed_dim)

    # 前向传播函数，接受多个输入参数，并返回模型预测结果
    @add_start_docstrings_to_model_forward(SIGLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=SiglipTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # 获取输出注意力的选项
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 获取输出隐藏状态的选项
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 获取是否返回字典形式结果的选项
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果输入的 input_ids 为 None，则抛出数值错误
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # 获取输入数据的形状，并将 input_ids 转换成二维张量
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # 对输入的 input_ids 进行嵌入操作，得到隐藏状态
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # 注意：SigLIP 的文本模型不使用因果掩码，不同于原始的CLIP模型。
        # 扩展注意力掩码
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # 使用编码器对隐藏状态进行编码，得到编码器的输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后一个隐藏状态，并对其进行 LayerNorm 归一化
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # 假设"sticky"的EOS标记化，最后一个标记始终是EOS。对最后一个隐藏状态进行汇总
        pooled_output = last_hidden_state[:, -1, :]
        # 对汇总输出进行线性变换
        pooled_output = self.head(pooled_output)

        # 如果不以字典形式返回结果
        if not return_dict:
            # 返回最后一个隐藏状态、汇总输出和编码器输出的其他隐藏状态和注意力权重
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 以字典形式返回结果
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 将文本模型从 SigLIP 中提取出来，没有额外的头或顶部投影
# SigLIP 模型的起始文档字符串
class SiglipTextModel(SiglipPreTrainedModel):
    # 配置类设置为 SiglipTextConfig
    config_class = SiglipTextConfig

    # 不需要分割的模块列表
    _no_split_modules = ["SiglipTextEmbeddings", "SiglipEncoderLayer"]

    def __init__(self, config: SiglipTextConfig):
        super().__init__(config)
        # 创建 SiglipTextTransformer 模型
        self.text_model = SiglipTextTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    # 正向传播方法
    @add_start_docstrings_to_model_forward(SIGLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=SiglipTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, SiglipTextModel

        >>> model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 text_model 的 forward 方法
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 定义 SiglipVisionTransformer 类
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        # 保存配置
        self.config = config
        # 嵌入维度等于隐藏大小
        embed_dim = config.hidden_size

        # 创建视觉嵌入
        self.embeddings = SiglipVisionEmbeddings(config)
        # 创建编码器
        self.encoder = SiglipEncoder(config)
        # 创建后层规范化层
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 创建头部
        self.head = SiglipMultiheadAttentionPoolingHead(config)

    # 正向传播方法
    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
```  
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=SiglipVisionConfig)
    # 装饰器，用于替换返回文档字符串，指定输出类型和配置类

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 前向传播函数，接受像素值和一些可选参数，返回类型为元组或BaseModelOutputWithPooling

        r"""
        Returns:
        # 返回空字符串，用于指示该函数是一个返回函数

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_attentions参数不为None，则使用它，否则使用配置中的output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果output_hidden_states参数不为None，则使用它，否则使用配置中的output_hidden_states

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果return_dict参数不为None，则使用它，否则使用配置中的use_return_dict

        hidden_states = self.embeddings(pixel_values)
        # 使用像素值进行嵌入，得到隐藏状态

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 将隐藏状态作为输入嵌入到编码器中，得到编码器的输出

        last_hidden_state = encoder_outputs[0]
        # 取编码器输出的第一个值作为最后的隐藏状态

        last_hidden_state = self.post_layernorm(last_hidden_state)
        # 对最后的隐藏状态进行层归一化处理

        pooled_output = self.head(last_hidden_state)
        # 使用最后的隐藏状态作为输入，得到池化输出

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
            # 如果不需要返回字典，则返回元组形式的结果

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        # 返回BaseModelOutputWithPooling对象，包含最后的隐藏状态、池化输出、隐藏状态和注意力权重
class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        # 创建一个可学习的参数probe，形状为(1, 1, config.hidden_size)
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # 创建一个多头注意力机制对象
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        # 创建 LayerNormalization 对象
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 SiglipMLP 对象
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        # 获取批量大小
        batch_size = hidden_state.shape[0]
        # 复制 probe 参数以匹配批量大小
        probe = self.probe.repeat(batch_size, 1, 1)

        # 使用注意力机制计算隐藏状态
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        # 保存残差连接
        residual = hidden_state
        # 应用 LayerNormalization
        hidden_state = self.layernorm(hidden_state)
        # 使用 MLP 处理隐藏状态
        hidden_state = residual + self.mlp(hidden_state)

        # 返回第一个位置的隐藏状态
        return hidden_state[:, 0]

@add_start_docstrings(
    """The vision model from SigLIP without any head or projection on top.""",
    SIGLIP_START_DOCSTRING,
)
class SiglipVisionModel(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        # 创建 SiglipVisionTransformer 模型
        self.vision_model = SiglipVisionTransformer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回输入嵌入
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=SiglipVisionConfig)
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SiglipVisionModel

        >>> model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        # 如果 return_dict 为 None，则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 vision_model 的 forward 方法
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
# 应用指定的文档字符串到模型类上
@add_start_docstrings(SIGLIP_START_DOCSTRING)
class SiglipModel(SiglipPreTrainedModel):
    # 指定配置类
    config_class = SiglipConfig

    def __init__(self, config: SiglipConfig):
        super().__init__(config)

        # 检查文本配置和视觉配置是否符合预期类型
        if not isinstance(config.text_config, SiglipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type SiglipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type SiglipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 提取文本配置和视觉配置
        text_config = config.text_config
        vision_config = config.vision_config

        # 初始化文本模型和视觉模型
        self.text_model = SiglipTextTransformer(text_config)
        self.vision_model = SiglipVisionTransformer(vision_config)

        # 初始化logit_scale和logit_bias
        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        # 初始化权重并应用最终处理
        self.post_init()

    # 应用文档字符串到模型前向方法上
    @add_start_docstrings_to_model_forward(SIGLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        返回：
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): 通过将投影层应用于[`SiglipTextModel`]的池化输出获得的文本嵌入。

        示例：

        ```python
        >>> from transformers import AutoTokenizer, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        >>> # 重要提示：确保将 padding="max_length"，因为模型是这样训练的
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        # 使用 SigLIP 模型的配置来替换一些字段（如果有指定的话）而不是视觉和文本组件的字段。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = text_outputs[1]

        return pooled_output

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     image_features = model.get_image_features(**inputs)
        ```"""
        # Use SiglipModel's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 vision_model 对象处理像素值，根据参数获取返回值
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 vision_outputs 中获取池化后的输出
        pooled_output = vision_outputs[1]

        # 返回池化后的输出
        return pooled_output

    # 将输入文档字符串添加到模型前面
    @add_start_docstrings_to_model_forward(SIGLIP_INPUTS_DOCSTRING)
    # 替换返回文档字符串
    @replace_return_docstrings(output_type=SiglipOutput, config_class=SiglipConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```