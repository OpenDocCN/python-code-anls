# `.\models\siglip\modeling_siglip.py`

```
# 导入 math 库，用于数学运算
import math
# 导入 warnings 库，用于警告处理
import warnings
# 导入 dataclass 模块中的 dataclass 装饰器，用于创建数据类
from dataclasses import dataclass
# 导入 typing 库，用于类型提示
from typing import Any, Optional, Tuple, Union

# 导入 numpy 库，通常用于科学计算
import numpy as np
# 导入 torch 库，主要深度学习框架
import torch
# 导入 torch.utils.checkpoint 模块，用于模型的检查点
import torch.utils.checkpoint
# 导入 torch.nn 模块，用于神经网络相关操作
from torch import nn
# 导入 torch.nn 中的损失函数，如 BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# 导入 torch.nn.init 中的 _calculate_fan_in_and_fan_out 函数，用于计算初始化时的 fan_in 和 fan_out
from torch.nn.init import _calculate_fan_in_and_fan_out

# 导入 ACT2FN，用于激活函数
from ...activations import ACT2FN
# 导入 _prepare_4d_attention_mask 函数，用于准备四维注意力掩码
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
# 导入 BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput 等模型输出相关类
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
# 导入 PreTrainedModel 类，作为所有预训练模型的基类
from ...modeling_utils import PreTrainedModel
# 导入各种辅助函数和工具函数，如日志记录、代码示例文档字符串添加等
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# General docstring
# 针对文档的配置信息
_CONFIG_FOR_DOC = "SiglipConfig"
# 针对文档的检查点信息
_CHECKPOINT_FOR_DOC = "google/siglip-base-patch16-224"

# Image classification docstring
# 图像分类的检查点信息
_IMAGE_CLASS_CHECKPOINT = "google/siglip-base-patch16-224"
# 图像分类的预期输出信息
_IMAGE_CLASS_EXPECTED_OUTPUT = "LABEL_1"

# SigLIP 预训练模型存档列表
SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/siglip-base-patch16-224",
    # 查看所有 SigLIP 模型，请访问 https://huggingface.co/models?filter=siglip
]

def _trunc_normal_(tensor, mean, std, a, b):
    # 从 PyTorch 官方代码库复制的截断正态分布初始化方法，直到它包含在几个官方发布版本中 - RW
    # 基于 https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf 的方法

    def norm_cdf(x):
        # 计算标准正态分布的累积分布函数
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        # 如果均值 mean 超出了 [a, b] 区间的两个标准差之外，发出警告
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # 通过截断的均匀分布生成值，然后使用正态分布的逆累积分布函数进行转换
    # 获取上下限的累积分布函数值
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # 在 [l, u] 区间均匀填充张量的值，然后转换到 [2l-1, 2u-1] 区间
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    # 使用逆CDF变换将张量转换为截断标准正态分布
    tensor.erfinv_()
    
    # 将张量缩放到正确的均值和标准差
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    
    # 使用 clamp 方法确保张量值在指定范围内
    tensor.clamp_(min=a, max=b)
# 使用截断正态分布填充给定的张量。值从正态分布中抽取，但超出[a, b]范围的值将重新抽取，直到在范围内。
def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        # 调用内部函数 _trunc_normal_，从标准正态分布中抽取值并进行截断处理
        _trunc_normal_(tensor, 0, 1.0, a, b)
        # 对张量进行缩放（乘以std）和平移（加上mean）
        tensor.mul_(std).add_(mean)


# 根据张量的形状计算“fan_in”和“fan_out”，并根据给定的比例因子和分布类型初始化张量
def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # 设置截断正态分布的标准差常量，该值是标准正态分布截断到(-2, 2)区间的标准差
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            # 从正态分布中抽取值并填充张量
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        # 计算均匀分布的上下界
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            # 从均匀分布中抽取值并填充张量
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


# 使用“fan_in”模式和截断正态分布初始化张量
def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


# 使用“fan_in”模式和正态分布初始化张量
def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


@dataclass
# 从transformers.models.clip.modeling_clip.CLIPVisionModelOutput类复制而来，仅修改为使用Siglip
class SiglipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    # 定义函数的参数列表，用于描述函数接受的输入参数以及它们的数据类型和形状
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            图像嵌入，通过将投影层应用于池化输出得到。是一个可选参数，当模型使用 `with_projection=True` 初始化时返回。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的输出隐藏状态序列。是一个形状为 `(batch_size, sequence_length, hidden_size)` 的张量。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态元组。如果模型具有嵌入层，则包括嵌入输出，形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重元组，用于计算自注意力头中的加权平均值。形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    """
    
    # 定义四个参数，分别对应图像嵌入、最后隐藏状态、隐藏状态元组和注意力权重元组，都有各自的数据类型和可选性
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 @dataclass 装饰器声明一个数据类，用于表示 SiglipTextModelOutput 类
@dataclass
# 从 transformers.models.clip.modeling_clip.CLIPTextModelOutput 复制过来，并将 CLIP 替换为 Siglip
class SiglipTextModelOutput(ModelOutput):
    """
    文本模型输出的基类，同时包含最后隐藏状态的汇聚。

    Args:
        text_embeds (`torch.FloatTensor`，形状为 `(batch_size, output_dim)`，可选项，在初始化模型时设置 `with_projection=True` 时返回):
            通过将投影层应用于池化输出获得的文本嵌入。
        last_hidden_state (`torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态的序列。
        hidden_states (`tuple(torch.FloatTensor)`，可选项，在传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            元组 `torch.FloatTensor`（如果模型具有嵌入层，则为嵌入层的输出，以及每一层的输出），
            形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每层输出的隐藏状态以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`，可选项，在传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            元组 `torch.FloatTensor`（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 @dataclass 装饰器声明一个数据类，用于表示 SiglipOutput 类
@dataclass
# 从 transformers.models.clip.modeling_clip.CLIPOutput 复制过来，并将 CLIP 替换为 Siglip
class SiglipOutput(ModelOutput):
    """
    Siglip 输出的基类。
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image: (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text: (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            The text embeddings obtained by applying the projection layer to the pooled output of `SiglipTextModel`.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            The image embeddings obtained by applying the projection layer to the pooled output of `SiglipVisionModel`.
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the `SiglipTextModel`.
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the `SiglipVisionModel`.
    """
    
    # 定义一个类，用于封装对比损失和模型输出
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None
    
    def to_tuple(self) -> Tuple[Any]:
        # 返回包含类属性的元组，但是对于"text_model_output"和"vision_model_output"属性，返回其转换为元组后的值
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 从配置中获取隐藏大小作为嵌入维度
        self.image_size = config.image_size  # 从配置中获取图像大小
        self.patch_size = config.patch_size  # 从配置中获取补丁大小

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,  # 输入通道数
            out_channels=self.embed_dim,      # 输出通道数（嵌入维度）
            kernel_size=self.patch_size,      # 卷积核大小（补丁大小）
            stride=self.patch_size,           # 卷积步长（补丁大小）
            padding="valid",                  # 卷积填充方式为有效填充
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 计算图像中的补丁数量
        self.num_positions = self.num_patches  # 位置嵌入的位置数量等于补丁数量
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # 创建位置嵌入层
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)
        # 注册位置 ID 缓冲区，用于存储位置索引的张量

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # 使用卷积层对像素值进行补丁嵌入
        embeddings = patch_embeds.flatten(2).transpose(1, 2)  # 将补丁嵌入展平并进行维度转置

        embeddings = embeddings + self.position_embedding(self.position_ids)
        # 加上位置嵌入，以增强补丁嵌入的语义表示
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->Siglip
class SiglipTextEmbeddings(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size  # 从配置中获取隐藏大小作为嵌入维度

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)  # 创建标记嵌入层
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)  # 创建位置嵌入层

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册位置 ID 缓冲区，用于存储位置索引的张量，支持序列化时的导出

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        # 计算输入序列的长度

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            # 如果未提供位置 ID，则使用预注册的位置 ID，并根据序列长度截取

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
            # 如果未提供嵌入张量，则使用输入标记 ID 进行嵌入

        position_embeddings = self.position_embedding(position_ids)
        # 获取位置嵌入张量

        embeddings = inputs_embeds + position_embeddings
        # 将标记嵌入和位置嵌入相加，生成最终的嵌入表示

        return embeddings


class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    # 初始化函数，用于初始化一个注意力机制模型对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置对象保存在实例变量中
        self.config = config
        # 设置嵌入维度为配置对象中的隐藏大小
        self.embed_dim = config.hidden_size
        # 设置注意力头的数量为配置对象中的注意力头数量
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查嵌入维度是否可以整除注意力头数量，否则抛出数值错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 设置缩放因子为头维度的负半数
        self.scale = self.head_dim**-0.5
        # 设置注意力机制中的丢弃率为配置对象中的注意力丢弃率
        self.dropout = config.attention_dropout

        # 初始化线性层，用于键、值、查询、输出的投影
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 前向传播函数，执行输入张量的注意力计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # 获取隐藏状态张量的维度信息
        batch_size, q_len, _ = hidden_states.size()

        # 将隐藏状态张量投影到查询向量空间
        query_states = self.q_proj(hidden_states)
        # 将隐藏状态张量投影到键向量空间
        key_states = self.k_proj(hidden_states)
        # 将隐藏状态张量投影到值向量空间
        value_states = self.v_proj(hidden_states)

        # 将投影后的张量重新形状为 (batch_size, q_len, num_heads, head_dim)，并交换维度
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 获取键-值对应的序列长度
        k_v_seq_len = key_states.shape[-2]
        # 计算注意力权重，使用 query 和 key 的点积，并乘以缩放因子
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # 检查注意力权重的维度是否符合预期
        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 如果有注意力掩码，则将其加到注意力权重上
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # 将注意力权重转换为 float32 类型，并进行 dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # 计算加权后的值向量
        attn_output = torch.matmul(attn_weights, value_states)

        # 检查输出的注意力张量的维度是否符合预期
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 重新调整注意力输出的维度顺序，并保证连续的内存布局
        attn_output = attn_output.transpose(1, 2).contiguous()
        # 将注意力输出重新形状为 (batch_size, q_len, embed_dim)
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        # 对输出应用最终的投影层变换
        attn_output = self.out_proj(attn_output)

        # 返回注意力输出以及注意力权重
        return attn_output, attn_weights
# 从 transformers.models.clip.modeling_clip.CLIPMLP 复制而来，将 CLIP 替换为 Siglip
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # 获取激活函数
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 第一个全连接层
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 第二个全连接层

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 第一个全连接层的前向传播
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states = self.fc2(hidden_states)  # 第二个全连接层的前向传播
        return hidden_states


# 从 transformers.models.clip.modeling_clip.CLIPEncoderLayer 复制而来，将 CLIP 替换为 Siglip
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.self_attn = SiglipAttention(config)  # 自注意力机制
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第一个层归一化
        self.mlp = SiglipMLP(config)  # 多层感知机
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第二个层归一化

    # 忽略复制部分
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                输入的张量形状为 `(batch, seq_len, embed_dim)`。
            attention_mask (`torch.FloatTensor`):
                注意力掩码形状为 `(batch, 1, q_len, k_v_seq_len)`，其中填充元素由非常大的负值表示。
            output_attentions (`bool`, *optional*, defaults to `False`):
                是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 获取更多细节。
        """
        residual = hidden_states  # 保留残差连接

        hidden_states = self.layer_norm1(hidden_states)  # 第一个层归一化
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )  # 自注意力层的前向传播
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 更新残差连接
        hidden_states = self.layer_norm2(hidden_states)  # 第二个层归一化
        hidden_states = self.mlp(hidden_states)  # 多层感知机的前向传播
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)  # 输出结果

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则加入到输出中

        return outputs


class SiglipPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化和下载预训练模型的抽象类。
    """

    config_class = SiglipConfig  # 配置类
    base_model_prefix = "siglip"  # 基础模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是 SiglipVisionEmbeddings 类型
        if isinstance(module, SiglipVisionEmbeddings):
            # 根据配置选择隐藏大小，初始化位置嵌入权重
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, SiglipConfig)
                else self.config.hidden_size
            )
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        
        # 如果模块是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 调用默认的 Flax 嵌入初始化方法
            default_flax_embed_init(module.weight)
        
        # 如果模块是 SiglipAttention 类型
        elif isinstance(module, SiglipAttention):
            # 使用 Xavier 均匀分布初始化权重
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            # 初始化偏置为零
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        
        # 如果模块是 SiglipMLP 类型
        elif isinstance(module, SiglipMLP):
            # 使用 Xavier 均匀分布初始化全连接层权重
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            # 使用小的正态分布初始化偏置
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        
        # 如果模块是 SiglipMultiheadAttentionPoolingHead 类型
        elif isinstance(module, SiglipMultiheadAttentionPoolingHead):
            # 使用 Xavier 均匀分布初始化 probe 数据
            nn.init.xavier_uniform_(module.probe.data)
            # 使用 Xavier 均匀分布初始化注意力层的权重
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            # 初始化注意力层的偏置为零
            nn.init.zeros_(module.attention.in_proj_bias.data)
        
        # 如果模块是 SiglipModel 类型
        elif isinstance(module, SiglipModel):
            # 初始化 logit_scale 数据为 log(1.0)
            logit_scale_init = torch.log(torch.tensor(1.0))
            module.logit_scale.data.fill_(logit_scale_init)
            # 初始化 logit_bias 数据为零
            module.logit_bias.data.zero_()
        
        # 如果模块是 nn.Linear 或 nn.Conv2d 类型
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用 LeCun 正态分布初始化权重
            lecun_normal_(module.weight)
            # 如果有偏置，初始化偏置为零
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # 如果模块是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零
            module.bias.data.zero_()
            # 初始化权重为 1.0
            module.weight.data.fill_(1.0)
# SIGLIP_START_DOCSTRING 是一个包含模型介绍信息的原始字符串，用于说明该模型继承自 PreTrainedModel 类，
# 可以查看超类文档以了解通用方法（如下载或保存模型、调整输入嵌入大小、修剪头等）。
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

# SIGLIP_TEXT_INPUTS_DOCSTRING 是一个包含文本输入信息的原始字符串，用于说明模型输入的参数和类型。
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

# SIGLIP_VISION_INPUTS_DOCSTRING 是一个空字符串，暂未填充任何文档内容。
SIGLIP_VISION_INPUTS_DOCSTRING = r"""
    
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 输入的像素数值张量，形状为(batch_size, num_channels, height, width)，包含图像的像素值。
            # 默认情况下会忽略填充部分。可以使用`AutoImageProcessor`获取像素值。详见`CLIPImageProcessor.__call__`。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。返回的张量中`attentions`字段会提供更详细的信息。
            # 可选参数，默认为False。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回的张量中`hidden_states`字段会提供更详细的信息。
            # 可选参数，默认为False。
        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]格式的结果，而不是普通的元组。
            # 可选参数，默认为False。
"""
SIGLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下将忽略填充。
            
            可以使用 [`AutoTokenizer`] 获得这些索引。详情请参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            遮罩，避免在填充的标记索引上执行注意力计算。遮罩值在 `[0, 1]` 之间：

            - 1 表示 **未被遮罩** 的标记，
            - 0 表示 **被遮罩** 的标记。

            [什么是注意力遮罩？](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。

            [什么是位置 ID？](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下将忽略填充。可以使用 [`AutoImageProcessor`] 获取像素值。详情请参见 [`CLIPImageProcessor.__call__`]。

        return_loss (`bool`, *optional*):
            是否返回对比损失。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。返回的张量中有关 `attentions` 的更多细节。

        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。返回的张量中有关 `hidden_states` 的更多细节。

        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而非普通元组。
"""


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->Siglip
class SiglipEncoder(nn.Module):
    """
    由 `config.num_hidden_layers` 个自注意力层组成的 Transformer 编码器。每一层都是一个 [`SiglipEncoderLayer`]。

    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        # 创建包含多个 `SiglipEncoderLayer` 的模块列表
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

    # 忽略复制
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(config)  # 初始化文本嵌入层对象
        self.encoder = SiglipEncoder(config)  # 初始化编码器对象
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 初始化最终层规范化对象

        self.head = nn.Linear(embed_dim, embed_dim)  # 创建线性层，用于处理池化输出

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")  # 如果没有提供输入ID，则抛出数值错误

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])  # 将输入ID调整为二维形状

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)  # 调用文本嵌入层进行输入嵌入

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)  # 准备四维注意力掩码

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 调用编码器进行前向传播

        last_hidden_state = encoder_outputs[0]  # 取编码器输出的最后隐藏状态
        last_hidden_state = self.final_layer_norm(last_hidden_state)  # 对最后隐藏状态进行规范化

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = last_hidden_state[:, -1, :]  # 汇集最终的输出，假设“sticky” EOS 标记化

        pooled_output = self.head(pooled_output)  # 通过线性层处理池化输出

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]  # 如果不返回字典形式，则返回元组

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )  # 返回字典形式的输出结果，包括最后隐藏状态、池化输出、隐藏状态和注意力权重
# 从 SigLIP 模型派生的文本模型，没有额外的头部或顶层投影
@add_start_docstrings(
    """The text model from SigLIP without any head or projection on top.""",
    SIGLIP_START_DOCSTRING,  # 添加了 SigLIP 的起始文档字符串
)
class SiglipTextModel(SiglipPreTrainedModel):
    config_class = SiglipTextConfig  # 设置配置类为 SiglipTextConfig

    _no_split_modules = ["SiglipTextEmbeddings", "SiglipEncoderLayer"]  # 不可分割的模块列表

    def __init__(self, config: SiglipTextConfig):
        super().__init__(config)
        self.text_model = SiglipTextTransformer(config)  # 初始化 SiglipTextTransformer 模型
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding  # 获取输入嵌入层

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value  # 设置输入嵌入层

    @add_start_docstrings_to_model_forward(SIGLIP_TEXT_INPUTS_DOCSTRING)  # 添加前向传播方法的文档字符串
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=SiglipTextConfig)  # 替换返回值文档字符串
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)  # 初始化视觉嵌入层
        self.encoder = SiglipEncoder(config)  # 初始化编码器
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 初始化后层归一化
        self.head = SiglipMultiheadAttentionPoolingHead(config)  # 初始化多头注意力池化头部

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)  # 添加前向传播方法的文档字符串
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=SiglipVisionConfig)
    # 使用装饰器 @replace_return_docstrings 替换返回值的文档字符串，指定输出类型为 BaseModelOutputWithPooling，配置类为 SiglipVisionConfig
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        该方法不返回具体文本，但应根据输出类型 BaseModelOutputWithPooling 进行说明。

        """
        # 如果 output_attentions 不为 None，则使用该值；否则使用 self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 不为 None，则使用该值；否则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 不为 None，则使用该值；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将像素值传递给嵌入层，得到隐藏状态
        hidden_states = self.embeddings(pixel_values)

        # 调用编码器进行前向传播，传递隐藏状态和其他配置参数
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态并通过后层标准化层处理
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 将处理后的最后隐藏状态传递给头部层，得到池化输出
        pooled_output = self.head(last_hidden_state)

        # 如果 return_dict 为 False，则返回元组形式的结果
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回 BaseModelOutputWithPooling 类型的对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        # 使用注意力机制处理隐藏状态，probe作为query和key，hidden_state作为value
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        # 使用 LayerNorm 进行归一化处理
        hidden_state = self.layernorm(hidden_state)
        # 使用 MLP 进行多层感知机处理，然后加上残差连接
        hidden_state = residual + self.mlp(hidden_state)

        # 返回处理后的隐藏状态的第一个维度（通常是batch维度）的第一个元素
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

        self.vision_model = SiglipVisionTransformer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回视觉模型的嵌入层
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
        返回：

        示例：

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
        >>> pooled_output = outputs.pooler_output  # 汇聚的特征
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
# 使用指定的文档字符串模板为类添加文档字符串
@add_start_docstrings(SIGLIP_START_DOCSTRING)
class SiglipModel(SiglipPreTrainedModel):
    # 设置配置类为 SiglipConfig
    config_class = SiglipConfig

    def __init__(self, config: SiglipConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 检查配置文件中的文本配置是否为 SiglipTextConfig 类型
        if not isinstance(config.text_config, SiglipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type SiglipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查配置文件中的视觉配置是否为 SiglipVisionConfig 类型
        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type SiglipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 获取文本和视觉配置
        text_config = config.text_config
        vision_config = config.vision_config

        # 初始化文本模型和视觉模型
        self.text_model = SiglipTextTransformer(text_config)
        self.vision_model = SiglipVisionTransformer(vision_config)

        # 初始化用于缩放和偏置的参数
        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        # 执行额外的初始化步骤和最终处理
        self.post_init()

    # 使用指定的文档字符串模板为方法添加文档字符串
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
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        # 根据参数设置或默认配置，确定是否返回注意力权重、隐藏状态及字典形式的返回
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 SigLIP 文本模型处理输入，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从文本输出中获取池化的输出作为文本特征表示
        pooled_output = text_outputs[1]

        # 返回文本特征表示
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
        # 使用 self.config 中的 output_attentions 字段，如果未指定则使用 vision_model 的默认值
        # 使用 self.config 中的 output_hidden_states 字段，如果未指定则使用 vision_model 的默认值
        # 使用 self.config 中的 use_return_dict 字段，如果未指定则使用 vision_model 的默认值

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用 vision_model 进行前向传播，传入像素值、注意力输出、隐藏状态输出和返回字典标志位

        pooled_output = vision_outputs[1]
        # 从 vision_model 的输出中获取汇聚的特征向量作为 pooled_output

        return pooled_output
# 声明一个用于 SigLIP 图像分类的编码器模型，其顶部有一个图像分类头部（线性层，位于补丁标记的最终隐藏状态之上），例如用于 ImageNet。
@add_start_docstrings(
    """
    SigLIP vision encoder with an image classification head on top (a linear layer on top of the pooled final hidden states of
    the patch tokens) e.g. for ImageNet.
    """,
    SIGLIP_START_DOCSTRING,
)
class SiglipForImageClassification(SiglipPreTrainedModel):
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化函数，接受一个 SiglipConfig 类型的配置对象
    def __init__(self, config: SiglipConfig) -> None:
        # 调用父类的初始化函数
        super().__init__(config)

        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 创建 SiglipVisionTransformer 类型的视觉模型
        self.vision_model = SiglipVisionTransformer(config.vision_config)

        # 分类器头部
        self.classifier = (
            nn.Linear(config.vision_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 声明 forward 方法，用于模型的前向传播
    @add_start_docstrings_to_model_forward(SIGLIP_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 output_attentions 参数为 None，则使用 self.config.output_attentions；否则使用传入的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数为 None，则使用 self.config.output_hidden_states；否则使用传入的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 参数为 None，则使用 self.config.use_return_dict；否则使用传入的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 vision_model 对象进行前向传播，传入 pixel_values 和各参数
        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 对 patch tokens 进行平均池化
        sequence_output = torch.mean(sequence_output[:, 1:, :], dim=1)
        # 应用分类器对序列输出进行分类预测
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 将 labels 移动到与 logits 相同的设备，以支持模型并行计算
            labels = labels.to(logits.device)
            # 确定问题类型（回归、单标签分类或多标签分类）
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不要求返回字典形式的结果，则返回元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典形式的结果，则返回 ImageClassifierOutput 对象
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```