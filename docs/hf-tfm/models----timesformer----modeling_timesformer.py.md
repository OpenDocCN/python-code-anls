# `.\transformers\models\timesformer\modeling_timesformer.py`

```
# 设置文件编码为 utf-8
# 版权声明和许可证信息
# 2022 Meta 和 The HuggingFace Inc. 团队保留所有权利。
# 根据 Apache 许可证 2.0 版本（"许可证"）授权使用本文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按"原样"分发，
# 不附带任何明示或暗示的担保或条件
# 查看许可证以获取特定语言的特定权限和限制
""" PyTorch TimeSformer 模型。"""

import collections
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数映射表
from ...activations import ACT2FN
# 导入模型输出相关类
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
# 导入预训练模型基类
from ...modeling_utils import PreTrainedModel
# 导入工具类函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 导入 Timesformer 模型配置
from .configuration_timesformer import TimesformerConfig

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# Timesformer 相关的配置和检查点信息
_CONFIG_FOR_DOC = "TimesformerConfig"
_CHECKPOINT_FOR_DOC = "facebook/timesformer"

# Timesformer 预训练模型列表
TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/timesformer-base-finetuned-k400",
    # 查看所有 TimeSformer 模型：https://huggingface.co/models?filter=timesformer
]

# 从 https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L155 改编而来
class TimesformerPatchEmbeddings(nn.Module):
    """图像到 Patch 嵌入"""

    def __init__(self, config):
        super().__init__()

        image_size = config.image_size
        patch_size = config.patch_size

        # 将图像大小和 Patch 大小转为迭代对象
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        # 计算 Patch 的数目
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 使用 Conv2d 进行投影
        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # 调整输入形状
        pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)

        # 投影输入像素值
        embeddings = self.projection(pixel_values)
        patch_width = embeddings.size(-1)
        # 展平和转置嵌入
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings, num_frames, patch_width

class TimesformerEmbeddings(nn.Module):
    """
    构建 Patch 和位置嵌入。
    """
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 从配置参数中获取隐藏层大小、帧数、隐藏层丢弃率和注意力类型
        embed_dim = config.hidden_size
        num_frames = config.num_frames
        drop_rate = config.hidden_dropout_prob
        attention_type = config.attention_type

        # 将注意力类型保存到实例变量中
        self.attention_type = attention_type
        # 创建 TimeSformerPatchEmbeddings 对象，并保存到实例变量中
        self.patch_embeddings = TimesformerPatchEmbeddings(config)
        # 获取图像分块的数量
        self.num_patches = self.patch_embeddings.num_patches

        # 位置编码部分
        # 创建一个可学习的参数用于表示类别标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 创建一个可学习的参数用于表示位置编码
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        # 定义一个丢弃层，用于位置编码的丢弃
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 如果注意力类型不是 "space_only"，则需要创建时间编码部分
        if attention_type != "space_only":
            # 创建一个可学习的参数用于表示时间编码
            self.time_embeddings = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            # 定义一个丢弃层，用于时间编码的丢弃
            self.time_drop = nn.Dropout(p=drop_rate)
    def forward(self, pixel_values):
        # 获取输入张量的批大小
        batch_size = pixel_values.shape[0]

        # 创建patch embeddings
        embeddings, num_frames, patch_width = self.patch_embeddings(pixel_values)

        # 扩展cls token以匹配embeddings大小，然后连接到embeddings中
        cls_tokens = self.cls_token.expand(embeddings.size(0), -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 如果位置嵌入的维度与输入不匹配，则调整位置嵌入的大小
        if embeddings.size(1) != self.position_embeddings.size(1):
            position_embeddings = self.position_embeddings
            cls_pos_embed = position_embeddings[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = position_embeddings[0, 1:, :].unsqueeze(0).transpose(1, 2)
            patch_num = int(other_pos_embed.size(2) ** 0.5)
            patch_height = embeddings.size(1) // patch_width
            other_pos_embed = other_pos_embed.reshape(1, embeddings.size(2), patch_num, patch_num)
            new_pos_embed = nn.functional.interpolate(
                other_pos_embed, size=(patch_height, patch_width), mode="nearest"
            )
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            embeddings = embeddings + new_pos_embed
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.pos_drop(embeddings)

        # 时间嵌入
        if self.attention_type != "space_only":
            # 获取cls token，并排除它，重新调整embeddings的形状
            cls_tokens = embeddings[:batch_size, 0, :].unsqueeze(1)
            embeddings = embeddings[:, 1:]
            _, patch_height, patch_width = embeddings.shape
            embeddings = (
                embeddings.reshape(batch_size, num_frames, patch_height, patch_width)
                .permute(0, 2, 1, 3)
                .reshape(batch_size * patch_height, num_frames, patch_width)
            )
            # 如果帧数与时间嵌入的维度不匹配，则调整时间嵌入的大小
            if num_frames != self.time_embeddings.size(1):
                time_embeddings = self.time_embeddings.transpose(1, 2)
                new_time_embeddings = nn.functional.interpolate(time_embeddings, size=(num_frames), mode="nearest")
                new_time_embeddings = new_time_embeddings.transpose(1, 2)
                embeddings = embeddings + new_time_embeddings
            else:
                embeddings = embeddings + self.time_embeddings
            embeddings = self.time_drop(embeddings)
            embeddings = embeddings.view(batch_size, patch_height, num_frames, patch_width).reshape(
                batch_size, patch_height * num_frames, patch_width
            )
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings
# 从transformers.models.beit.modeling_beit.drop_path中复制过来的函数drop_path，用于实现Stochastic Depth(随机深度)。
# 该函数在残差块的主路径中应用，根据概率丢弃路径
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    在每个样本中丢弃路径（当应用在残差块的主路径中时）。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.beit.modeling_beit.BeitDropPath中复制过来的类TimeSformerDropPath
class TimeSformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# 从https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L57中改编的类TimesformerSelfAttention
class TimesformerSelfAttention(nn.Module):
    def __init__(self, config: TimesformerConfig):
        super().__init__()

        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attention_dropout_prob = config.attention_probs_dropout_prob

        self.num_heads = num_heads
        head_dim = config.hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_prob)
    # 定义一个方法用于执行前向传播，接收隐藏状态和是否输出注意力权重的标志
    def forward(self, hidden_states, output_attentions: bool = False):
        # 获取隐藏状态张量的维度信息
        batch_size, hidden_size, num_channels = hidden_states.shape
        # 将隐藏状态通过 qkv 网络得到的 query、key、value 分别进行变换和重排列
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, hidden_size, 3, self.num_heads, num_channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        # 计算注意力得分，使用 query 与 key 的点乘，然后通过 scale 进行缩放
        attention_probs = (query @ key.transpose(-2, -1)) * self.scale
        # 对注意力得分进行 softmax 操作
        attention_probs = attention_probs.softmax(dim=-1)
        # 对注意力得分应用 dropout 操作
        attention_probs = self.attn_drop(attention_probs)

        # 计算上下文向量，将注意力得分与 value 相乘并进行重排列
        context_layer = (attention_probs @ value).transpose(1, 2).reshape(batch_size, hidden_size, num_channels)

        # 如果需要输出注意力权重，则将注意力权重加入到输出结果中
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
class TimesformerSelfOutput(nn.Module):
    """
    The residual connection is defined in TimesformerLayer instead of here (as is the case with other models), due to
    the layernorm applied before each block.
    """
    # TimesformerSelfOutput 类定义，用于处理 Timesformer 模型自输出部分的计算

    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        # 初始化线性层，将输入特征转换为隐藏状态维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化丢弃层，用于随机丢弃隐藏状态中的部分数据
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理隐藏状态，改变隐藏状态的维度
        hidden_states = self.dense(hidden_states)
        # 对处理后的隐藏状态进行丢弃操作
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TimeSformerAttention(nn.Module):
    # TimeSformerAttention 类用于处理 TimeSformer 模型的注意力机制

    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        # 初始化自注意力机制，用于计算注意力权重
        self.attention = TimesformerSelfAttention(config)
        # 初始化自输出层，处理注意力输出的结果
        self.output = TimesformerSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 通过自注意力机制处理输入的隐藏状态
        self_outputs = self.attention(hidden_states, output_attentions)

        # 将自注意力输出的隐藏状态传递给自输出层进行处理
        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L39
class TimesformerIntermediate(nn.Module):
    # TimesformerIntermediate 类用于处理 TimeSformer 模型的中间层计算

    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        # 初始化线性层，将隐藏状态变换为中间层的维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 初始化丢弃层，用于随机丢弃中间层中的部分数据
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 判断隐藏激活函数是否为字符串类型，选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理隐藏状态，将隐藏状态变换为中间层的维度
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理中间层隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对中间层隐藏状态进行丢弃操作
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TimesformerOutput(nn.Module):
    # TimesformerOutput 类用于处理 TimeSformer 模型的输出层计算

    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        # 初始化线性层，将中间层的隐藏状态转换为输出层的维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化丢弃层，用于随机丢弃输出层中的部分数据
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理中间层隐藏状态，将其转换为输出层的隐藏状态维度
        hidden_states = self.dense(hidden_states)
        # 对输出层隐藏状态进行丢弃操作
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L89
class TimesformerLayer(nn.Module):
    # TimesformerLayer 类用于定义 TimeSformer 模型的一个层次
    # 初始化函数，接受配置和层索引作为参数
    def __init__(self, config: TimesformerConfig, layer_index: int) -> None:
        # 调用父类的初始化函数
        super().__init__()

        # 获取配置中的注意力类型
        attention_type = config.attention_type

        # 根据随机深度衰减规则生成丢弃路径率列表
        drop_path_rates = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]  # stochastic depth decay rule
        # 获取当前层的丢弃路径率
        drop_path_rate = drop_path_rates[layer_index]

        # 如果丢弃路径率大于0，创建丢弃路径对象，否则创建单位矩阵
        self.drop_path = TimeSformerDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 创建时间变换器的注意力模块
        self.attention = TimeSformerAttention(config)
        # 创建时间变换器的中间层模块
        self.intermediate = TimesformerIntermediate(config)
        # 创建时间变换器的输出层模块
        self.output = TimesformerOutput(config)
        # 创建层之前的 LayerNorm
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建层之后的 LayerNorm
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 保存配置和注意力类型
        self.config = config
        self.attention_type = attention_type
        # 如果注意力类型不在指定范围内，则抛出值错误
        if attention_type not in ["divided_space_time", "space_only", "joint_space_time"]:
            raise ValueError("Unknown attention type: {}".format(attention_type))

        # 如果注意力类型为 "divided_space_time"，则创建时间注意力层和临时密集层
        if self.attention_type == "divided_space_time":
            # 创建临时层的 LayerNorm
            self.temporal_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            # 创建临时时间注意力模块
            self.temporal_attention = TimeSformerAttention(config)
            # 创建临时密集层
            self.temporal_dense = nn.Linear(config.hidden_size, config.hidden_size)
# 定义一个 TimesformerEncoder 类，继承自 nn.Module 类
class TimesformerEncoder(nn.Module):
    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        # 将传入的配置保存到 config 属性中
        self.config = config
        # 初始化一个 nn.ModuleList 对象，包含 config.num_hidden_layers 个 TimesformerLayer 实例
        self.layer = nn.ModuleList([TimesformerLayer(config, ind) for ind in range(config.num_hidden_layers)])
        # 设置 gradient_checkpointing 属性为 False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 初始化 all_hidden_states 变量为空元组或 None
        all_hidden_states = () if output_hidden_states else None
        # 初始化 all_self_attentions 变量为空元组或 None
        all_self_attentions = () if output_attentions else None

        # 遍历 TimesformerLayer 实例
        for i, layer_module in enumerate(self.layer):
            # 如果 output_hidden_states 为 True，将当前 hidden_states 加入 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 判断梯度检查点开关，如果开启且处于训练模式，则使用梯度检查点来计算输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    output_attentions,
                )
            else:
                # 否则直接调用 layer_module 进行前向传播
                layer_outputs = layer_module(hidden_states, output_attentions)

            # 获取当前层的输出结果
            hidden_states = layer_outputs[0]

            # 如果 output_attentions 为 True，将当前层的注意力加入 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果 output_hidden_states 为 True，在循环结束后将最后的 hidden_states 加入 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 判断 return_dict，如果为 False，返回非 None 变量构成的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回一个 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TimesformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # 定义 config_class、base_model_prefix、main_input_name 和 supports_gradient_checkpointing 属性
    
    config_class = TimesformerConfig
    base_model_prefix = "timesformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    # 初始化权重函数
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对线性层和卷积层的权重进行截断正态分布初始化
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # 如果是 LayerNorm 层，将其偏置项初始化为 0，权重初始化为 1
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        # 如果是 TimesformerEmbeddings 层，对 cls_token、position_embeddings 和 patch_embeddings 进行初始化
        elif isinstance(module, TimesformerEmbeddings):
            nn.init.trunc_normal_(module.cls_token, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.position_embeddings, std=self.config.initializer_range)
            module.patch_embeddings.apply(self._init_weights)


# 定义 TIMESFORMER_START_DOCSTRING 变量
TIMESFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    # 将该类视为常规的 PyTorch 模块，并参考 PyTorch 文档了解与一般使用和行为相关的所有事项。

    # 参数:
    #     config ([`TimesformerConfig`]): 包含模型所有参数的模型配置类。
    #         使用配置文件初始化不会加载与模型关联的权重，只会加载配置信息。
    #         查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

TIMESFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`VideoMAEImageProcessor.preprocess`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare TimeSformer Model transformer outputting raw hidden-states without any specific head on top.",
    TIMESFORMER_START_DOCSTRING,
)
class TimesformerModel(TimesformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TimesformerEmbeddings(config)
        self.encoder = TimesformerEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TIMESFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        The forward pass for the TimeSformer model.

        Args:
            pixel_values (torch.FloatTensor): Input pixel values
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers
            output_hidden_states (bool, optional): Whether or not to return the hidden states of all layers
            return_dict (bool, optional): Whether or not to return a ModelOutput instead of a plain tuple

        Returns:
            BaseModelOutput: Output including last_hidden_state, pooler_output, hidden_states, attentions
        """
        
@add_start_docstrings(
    """TimeSformer Model transformer with a video classification head on top (a linear layer on top of the final hidden state
of the [CLS] token) e.g. for ImageNet.""",
    TIMESFORMER_START_DOCSTRING,
)
class TimesformerForVideoClassification(TimesformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.timesformer = TimesformerModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TIMESFORMER_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        The forward pass for the TimeSformer model for video classification.

        Args:
            pixel_values (torch.FloatTensor): Input pixel values
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers
            output_hidden_states (bool, optional): Whether or not to return the hidden states of all layers
            return_dict (bool, optional): Whether or not to return a ModelOutput instead of a plain tuple
        """ 
    # 使用装饰器替换返回文档字符串，指定输出类型为ImageClassifierOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法，接受输入的像素值、标签、是否输出注意力、是否输出隐藏状态、是否返回字典等参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 像素值，默认为None
        labels: Optional[torch.Tensor] = None,         # 标签，默认为None
        output_attentions: Optional[bool] = None,      # 是否输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,   # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,            # 是否返回字典，默认为None
```