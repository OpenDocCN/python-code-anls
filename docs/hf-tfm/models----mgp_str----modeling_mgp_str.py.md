# `.\models\mgp_str\modeling_mgp_str.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及保留所有权利给 Alibaba Research 和 HuggingFace Inc. 团队
#
# 根据 Apache 许可证 2.0 版本授权使用本文件
# 除非符合许可证规定，否则不得使用本文件
# 您可以从以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于"原样"提供的，不提供任何形式的担保或条件，
# 包括但不限于，适销性、特定用途适用性和非侵权性担保。
# 有关详细信息，请参阅许可证。

""" PyTorch MGP-STR model."""

import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mgp_str import MgpstrConfig

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# 用于文档的通用说明
_CONFIG_FOR_DOC = "MgpstrConfig"
_TOKENIZER_FOR_DOC = "MgpstrTokenizer"

# 模型检查点的基本说明
_CHECKPOINT_FOR_DOC = "alibaba-damo/mgp-str-base"

# 预训练模型存档列表
MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "alibaba-damo/mgp-str-base",
    # 查看所有 MGP-STR 模型的列表：https://huggingface.co/models?filter=mgp-str
]

# 以下是函数定义和类定义，用于模型中的路径丢弃功能
# 从 transformers.models.beit.modeling_beit.drop_path 中复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# 从 transformers.models.beit.modeling_beit.BeitDropPath 中复制的类，将 Beit 改为 Mgpstr
class MgpstrDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 设置对象的属性，用于指定 dropout 的概率
        self.drop_prob = drop_prob

    # 前向传播函数，处理输入的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数，对隐藏状态进行随机的 dropout 操作
        return drop_path(hidden_states, self.drop_prob, self.training)

    # 提供额外的表示信息，用于描述当前对象的状态
    def extra_repr(self) -> str:
        # 返回一个字符串，表示对象的 dropout 概率
        return "p={}".format(self.drop_prob)
# 定义了一个数据类 `MgpstrModelOutput`，继承自 `ModelOutput` 类，用于表示模型输出结果
@dataclass
class MgpstrModelOutput(ModelOutput):
    
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    Args:
        logits (`tuple(torch.FloatTensor)` of shape `(batch_size, config.num_character_labels)`):
            Tuple of `torch.FloatTensor` containing classification scores (before SoftMax) for characters, bpe, and wordpiece.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` containing hidden states of the model at each layer and optional initial embeddings.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` containing attention weights for each layer after softmax computation.
        a3_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_a3_attentions=True` is passed or when `config.output_a3_attentions=True`):
            Tuple of `torch.FloatTensor` containing attention weights for character, bpe, and wordpiece after softmax computation.
    """

    # logits 包含分类分数，形状为 (batch_size, config.num_character_labels)
    logits: Tuple[torch.FloatTensor] = None
    # hidden_states 包含模型每层的隐藏状态，形状为 (batch_size, sequence_length, hidden_size)，可选
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions 包含每层注意力权重，形状为 (batch_size, config.max_token_length, sequence_length, sequence_length)，可选
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # a3_attentions 包含字符、bpe、wordpiece 的注意力权重，形状为 (batch_size, config.max_token_length, sequence_length)，可选
    a3_attentions: Optional[Tuple[torch.FloatTensor]] = None


class MgpstrEmbeddings(nn.Module):
    """2D Image to Patch Embedding"""
    # 初始化函数，接受一个MgpstrConfig类型的配置对象作为参数
    def __init__(self, config: MgpstrConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置对象中的image_size属性确定图像大小，若为可迭代对象则直接使用，否则将其转换为元组
        image_size = (
            config.image_size
            if isinstance(config.image_size, collections.abc.Iterable)
            else (config.image_size, config.image_size)
        )
        # 根据配置对象中的patch_size属性确定patch大小，若为可迭代对象则直接使用，否则将其转换为元组
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        # 设置对象的image_size属性为确定后的图像大小
        self.image_size = image_size
        # 设置对象的patch_size属性为确定后的patch大小
        self.patch_size = patch_size
        # 根据图像大小和patch大小计算出网格大小，以元组形式保存
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        # 计算总的patch数目
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 如果配置对象中指定为精炼模式，则token数目为2，否则为1
        self.num_tokens = 2 if config.distilled else 1

        # 使用nn.Conv2d定义一个投影层，将输入通道数转换为隐藏大小，卷积核大小为patch_size，步长也为patch_size
        self.proj = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

        # 定义一个可学习的分类token，维度为1x1x隐藏大小，作为分类信息的表示
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # 定义一个可学习的位置嵌入矩阵，维度为1x(num_patches + num_tokens)x隐藏大小，表示每个patch和token的位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, config.hidden_size))
        # 使用Dropout进行位置嵌入的随机失活，概率为配置对象中指定的drop_rate
        self.pos_drop = nn.Dropout(p=config.drop_rate)

    # 前向传播函数，接受输入的像素值张量，返回嵌入向量张量
    def forward(self, pixel_values):
        # 获取输入像素值张量的形状信息
        batch_size, channel, height, width = pixel_values.shape
        # 检查输入图像的高度和宽度是否与预期的image_size匹配，若不匹配则抛出数值错误异常
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        # 将输入的像素值张量通过投影层转换为patch嵌入张量，同时将其展平并转置以适应后续操作
        patch_embeddings = self.proj(pixel_values)
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)  # BCHW -> BNC

        # 使用分类token扩展为batch_size份，形状为(batch_size, 1, 隐藏大小)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 将分类token与patch嵌入拼接在一起，形状为(batch_size, num_patches + num_tokens, 隐藏大小)
        embedding_output = torch.cat((cls_tokens, patch_embeddings), dim=1)
        # 加上位置嵌入信息，形状为(1, num_patches + num_tokens, 隐藏大小)，与embedding_output形状相加
        embedding_output = embedding_output + self.pos_embed
        # 对加和后的embedding_output进行位置嵌入的随机失活
        embedding_output = self.pos_drop(embedding_output)

        # 返回最终的嵌入向量张量
        return embedding_output
class MgpstrMlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, config: MgpstrConfig, hidden_features):
        super().__init__()
        hidden_features = hidden_features or config.hidden_size  # 如果未提供 hidden_features，则使用 config 中的 hidden_size
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)  # 第一个全连接层，输入维度为 config.hidden_size，输出维度为 hidden_features
        self.act = nn.GELU()  # GELU 激活函数
        self.fc2 = nn.Linear(hidden_features, config.hidden_size)  # 第二个全连接层，输入维度为 hidden_features，输出维度为 config.hidden_size
        self.drop = nn.Dropout(config.drop_rate)  # Dropout 操作，丢弃率为 config.drop_rate

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)  # 第一个全连接层的前向传播
        hidden_states = self.act(hidden_states)  # 应用 GELU 激活函数
        hidden_states = self.drop(hidden_states)  # Dropout 操作
        hidden_states = self.fc2(hidden_states)  # 第二个全连接层的前向传播
        hidden_states = self.drop(hidden_states)  # 再次应用 Dropout
        return hidden_states  # 返回处理后的隐藏状态


class MgpstrAttention(nn.Module):
    def __init__(self, config: MgpstrConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads  # 注意力头的数量
        head_dim = config.hidden_size // config.num_attention_heads  # 每个注意力头的维度
        self.scale = head_dim ** -0.5  # 缩放因子

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)  # QKV 线性变换
        self.attn_drop = nn.Dropout(config.attn_drop_rate)  # Attention Dropout 操作
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)  # 投影层线性变换
        self.proj_drop = nn.Dropout(config.drop_rate)  # Dropout 操作

    def forward(self, hidden_states):
        batch_size, num, channel = hidden_states.shape  # 获取输入张量的形状信息
        qkv = (
            self.qkv(hidden_states)  # 执行 QKV 线性变换
            .reshape(batch_size, num, 3, self.num_heads, channel // self.num_heads)  # 重塑张量形状以便后续处理
            .permute(2, 0, 3, 1, 4)  # 调整维度顺序
        )
        query, key, value = qkv[0], qkv[1], qkv[2]  # 分割 QKV 信息以便后续处理（为了兼容 TorchScript）

        attention_probs = (query @ key.transpose(-2, -1)) * self.scale  # 计算注意力分数
        attention_probs = attention_probs.softmax(dim=-1)  # 对注意力分数进行 softmax 操作
        attention_probs = self.attn_drop(attention_probs)  # 应用 Attention Dropout

        context_layer = (attention_probs @ value).transpose(1, 2).reshape(batch_size, num, channel)  # 计算上下文向量
        context_layer = self.proj(context_layer)  # 应用投影层线性变换
        context_layer = self.proj_drop(context_layer)  # 应用 Dropout 操作
        return (context_layer, attention_probs)  # 返回上下文层及注意力分数


class MgpstrLayer(nn.Module):
    def __init__(self, config: MgpstrConfig, drop_path=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # Layer Normalization 操作
        self.attn = MgpstrAttention(config)  # 注意力机制
        self.drop_path = MgpstrDropPath(drop_path) if drop_path is not None else nn.Identity()  # 随机深度路径（用于随机深度）
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # Layer Normalization 操作
        mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)  # MLP 隐藏层维度
        self.mlp = MgpstrMlp(config, mlp_hidden_dim)  # 多层感知机模块
    # 定义模型的前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用 self.attn 对隐藏状态进行自注意力机制计算，经过 self.norm1 归一化处理
        self_attention_outputs = self.attn(self.norm1(hidden_states))
        # 获取自注意力机制的输出结果和中间层输出
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1]

        # 第一个残差连接：将经过注意力计算后的输出加上原始输入隐藏状态，并施加随机丢弃(drop path)
        hidden_states = self.drop_path(attention_output) + hidden_states

        # 第二个残差连接：将经过 self.norm2 归一化后的隐藏状态经过 MLP 处理，再次施加随机丢弃(drop path)，然后加上之前的 hidden_states
        layer_output = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))

        # 将最终的层输出和注意力输出组成元组作为最终的输出
        outputs = (layer_output, outputs)
        return outputs
class MgpstrEncoder(nn.Module):
    def __init__(self, config: MgpstrConfig):
        super().__init__()
        # stochastic depth decay rule
        # 根据配置中的drop_path_rate生成随机深度的衰减规则列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]

        # 使用MgpstrLayer创建神经网络模型的多层堆叠
        self.blocks = nn.Sequential(
            *[MgpstrLayer(config=config, drop_path=dpr[i]) for i in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历并执行神经网络模型的每个块（layer）
        for _, blk in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 执行当前块（layer）的前向传播，更新隐藏状态
            layer_outputs = blk(hidden_states)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态加入到所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不要求返回字典，则根据需要返回不同的元组或对象
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MgpstrA3Module(nn.Module):
    def __init__(self, config: MgpstrConfig):
        super().__init__()
        # 初始化层归一化层，用于标准化token的向量表示
        self.token_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 通过卷积操作生成token学习器，用于生成token和注意力权重
        self.tokenLearner = nn.Sequential(
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=(1, 1), stride=1, groups=8, bias=False),
            nn.Conv2d(config.hidden_size, config.max_token_length, kernel_size=(1, 1), stride=1, bias=False),
        )
        # 初始化特征提取器的卷积层，用于生成特征表示
        self.feat = nn.Conv2d(
            config.hidden_size, config.hidden_size, kernel_size=(1, 1), stride=1, groups=8, bias=False
        )
        # 初始化层归一化层，用于标准化特征向量的表示
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 标准化token的向量表示
        hidden_states = self.token_norm(hidden_states)
        # 调整张量维度以便进入token学习器
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
        # 使用token学习器生成token及其注意力权重
        selected = self.tokenLearner(hidden_states)
        selected = selected.flatten(2)
        attentions = F.softmax(selected, dim=-1)

        # 使用特征提取器生成特征表示
        feat = self.feat(hidden_states)
        feat = feat.flatten(2).transpose(1, 2)
        # 使用注意力权重和特征表示计算A3模块的输出
        feat = torch.einsum("...si,...id->...sd", attentions, feat)
        a3_out = self.norm(feat)

        return (a3_out, attentions)


class MgpstrPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类和基础模型前缀
    config_class = MgpstrConfig
    base_model_prefix = "mgp_str"
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果 module 是 MgpstrEmbeddings 类型的实例
        if isinstance(module, MgpstrEmbeddings):
            # 对 module 的位置嵌入和类别标记进行截断正态分布初始化
            nn.init.trunc_normal_(module.pos_embed, mean=0.0, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.cls_token, mean=0.0, std=self.config.initializer_range)
        # 如果 module 是 nn.Linear 或者 nn.Conv2d 类型的实例
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            # 初始化 module 的权重数据为截断正态分布
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            # 如果 module 有偏置项，则将其数据置零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.LayerNorm 类型的实例
        elif isinstance(module, nn.LayerNorm):
            # 将 module 的偏置项数据置零
            module.bias.data.zero_()
            # 将 module 的权重数据填充为 1.0
            module.weight.data.fill_(1.0)
# 定义多行字符串，用于描述 MGP-STR 模型的基本信息和使用说明
MGP_STR_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MgpstrConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义多行字符串，用于描述 MGP-STR 模型前向传播方法的输入参数说明
MGP_STR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 使用装饰器 `add_start_docstrings`，为 MgpstrModel 类添加类级别的文档字符串，描述该类为裸模型变换器，输出未经特定头部处理的原始隐藏状态
@add_start_docstrings(
    "The bare MGP-STR Model transformer outputting raw hidden-states without any specific head on top.",
    MGP_STR_START_DOCSTRING,
)
# 定义 MgpstrModel 类，继承自 MgpstrPreTrainedModel 类
class MgpstrModel(MgpstrPreTrainedModel):
    def __init__(self, config: MgpstrConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 将配置信息存储在实例变量中
        self.config = config
        # 创建并初始化嵌入层对象
        self.embeddings = MgpstrEmbeddings(config)
        # 创建并初始化编码器对象
        self.encoder = MgpstrEncoder(config)

    # 定义方法用于获取输入嵌入层对象
    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.proj

    # 使用装饰器 `add_start_docstrings_to_model_forward`，为 forward 方法添加文档字符串，描述其输入参数的详细用法
    @add_start_docstrings_to_model_forward(MGP_STR_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 注意：这里的方法还未完整定义，继续定义在后续的代码中
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        # 如果 output_attentions 参数为 None，则使用模型配置中的设定
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数为 None，则使用模型配置中的设定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 参数为 None，则使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为空，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值 pixel_values 输入到 embeddings 层，得到嵌入输出 embedding_output
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出 embedding_output 输入到编码器 encoder 中进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果 return_dict 为 False，则直接返回编码器的输出 encoder_outputs
        if not return_dict:
            return encoder_outputs
        
        # 如果 return_dict 为 True，则封装编码器的输出为 BaseModelOutput 对象并返回
        return BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用装饰器为类添加文档字符串，描述了该类是一个 MGP-STR 模型转换器，具有三个分类头部，用于场景文本识别 (STR)。
# 该模型在变换编码器输出的基础上添加了三个 A^3 模块和三个线性层。
class MgpstrForSceneTextRecognition(MgpstrPreTrainedModel):
    # 指定配置类为 MgpstrConfig
    config_class = MgpstrConfig
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    def __init__(self, config: MgpstrConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化时从配置中获取标签数目
        self.num_labels = config.num_labels
        # 创建 MGP-STR 模型
        self.mgp_str = MgpstrModel(config)

        # 创建三个不同的 A^3 模块，分别用于字符级别、BPE（Byte Pair Encoding）级别和词片段级别的处理
        self.char_a3_module = MgpstrA3Module(config)
        self.bpe_a3_module = MgpstrA3Module(config)
        self.wp_a3_module = MgpstrA3Module(config)

        # 创建三个线性头部，分别用于字符级别、BPE 级别和词片段级别的分类
        self.char_head = nn.Linear(config.hidden_size, config.num_character_labels)
        self.bpe_head = nn.Linear(config.hidden_size, config.num_bpe_labels)
        self.wp_head = nn.Linear(config.hidden_size, config.num_wordpiece_labels)

    # 使用装饰器为 forward 方法添加输入文档字符串，描述输入参数的含义
    # 并替换返回值的文档字符串为 MgpstrModelOutput 类型和 MgpstrConfig 配置类的描述
    @add_start_docstrings_to_model_forward(MGP_STR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MgpstrModelOutput, config_class=MgpstrConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_a3_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], MgpstrModelOutput]:
        r"""
        output_a3_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of a3 modules. See `a3_attentions` under returned tensors
            for more detail.

        Returns:
            This function returns either a tuple of torch.FloatTensor or an instance of MgpstrModelOutput.

        Example:

        ```
        >>> from transformers import (
        ...     MgpstrProcessor,
        ...     MgpstrForSceneTextRecognition,
        ... )
        >>> import requests
        >>> from PIL import Image

        >>> # load image from the IIIT-5k dataset
        >>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> processor = MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base")
        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

        >>> model = MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base")

        >>> # inference
        >>> outputs = model(pixel_values)
        >>> out_strs = processor.batch_decode(outputs.logits)
        >>> out_strs["generated_text"]
        '["ticket"]'
        ```

        Initialize variables to default values if not provided by the caller.
        `output_attentions`, `output_hidden_states`, and `return_dict` are set based on the model configuration.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mgp_outputs = self.mgp_str(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = mgp_outputs[0]

        # Apply attention modules to sequence_output
        char_a3_out, char_attention = self.char_a3_module(sequence_output)
        bpe_a3_out, bpe_attention = self.bpe_a3_module(sequence_output)
        wp_a3_out, wp_attention = self.wp_a3_module(sequence_output)

        # Compute logits using corresponding head modules
        char_logits = self.char_head(char_a3_out)
        bpe_logits = self.bpe_head(bpe_a3_out)
        wp_logits = self.wp_head(wp_a3_out)

        # Aggregate all attention tensors if output_a3_attentions is True
        all_a3_attentions = (char_attention, bpe_attention, wp_attention) if output_a3_attentions else None
        all_logits = (char_logits, bpe_logits, wp_logits)

        # Return either a tuple of outputs or MgpstrModelOutput based on return_dict
        if not return_dict:
            outputs = (all_logits, all_a3_attentions) + mgp_outputs[1:]
            return tuple(output for output in outputs if output is not None)
        return MgpstrModelOutput(
            logits=all_logits,
            hidden_states=mgp_outputs.hidden_states,
            attentions=mgp_outputs.attentions,
            a3_attentions=all_a3_attentions,
        )
```