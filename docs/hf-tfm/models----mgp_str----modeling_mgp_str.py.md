# `.\transformers\models\mgp_str\modeling_mgp_str.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 版权所有 2023 年阿里巴巴研究和 HuggingFace 公司团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，不附带任何明示或暗示的担保。
# 请查看许可证以获取特定语言的权限和
# 许可证下的限制。
""" PyTorch MGP-STR 模型。"""

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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的通用字符串
_CONFIG_FOR_DOC = "MgpstrConfig"
_TOKENIZER_FOR_DOC = "MgpstrTokenizer"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "alibaba-damo/mgp-str-base"

# MGP-STR 预训练模型存档列表
MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "alibaba-damo/mgp-str-base",
    # 查看所有 MGP-STR 模型 https://huggingface.co/models?filter=mgp-str
]

# 从 transformers.models.beit.modeling_beit.drop_path 复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    每个样本丢弃路径（随机深度）（应用于残差块的主路径时）。

    Ross Wightman 的评论：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    但原始名称具有误导性，因为“Drop Connect”是另一篇论文中的不同形式的 dropout...
    请参阅讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    我选择将层和参数名称更改为“drop path”，而不是将 DropConnect 作为层名称混合使用，并使用“生存率”作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output

# 从 transformers.models.beit.modeling_beit.BeitDropPath 复制的类，将 Beit->Mgpstr
class MgpstrDropPath(nn.Module):
    """每个样本丢弃路径（随机深度）（应用于残差块的主路径时）。"""
    # 初始化函数，初始化DropPath对象，可选参数为丢弃概率
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 设置丢弃概率
        self.drop_prob = drop_prob
    
    # 前向传播函数，对隐藏状态进行丢弃路径操作
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)
    
    # 返回额外表示信息的字符串，格式为p=丢弃概率
    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)
# 使用 dataclass 装饰器创建 MgpstrModelOutput 类，继承自 ModelOutput 类
class MgpstrModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        logits (`tuple(torch.FloatTensor)` of shape `(batch_size, config.num_character_labels)`):
            Tuple of `torch.FloatTensor` (one for the output of character of shape `(batch_size,
            config.max_token_length, config.num_character_labels)`, + one for the output of bpe of shape `(batch_size,
            config.max_token_length, config.num_bpe_labels)`, + one for the output of wordpiece of shape `(batch_size,
            config.max_token_length, config.num_wordpiece_labels)`) .

            Classification scores (before SoftMax) of character, bpe and wordpiece.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, config.max_token_length,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        a3_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_a3_attentions=True` is passed or when `config.output_a3_attentions=True`):
            Tuple of `torch.FloatTensor` (one for the attention of character, + one for the attention of bpe`, + one
            for the attention of wordpiece) of shape `(batch_size, config.max_token_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 分类得分的元组，包含字符、bpe 和 wordpiece 的输出
    logits: Tuple[torch.FloatTensor] = None
    # 隐藏状态的元组，包含每个层的输出和可选的初始嵌入输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重的元组，用于计算自注意力头中的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # a3 注意力权重的元组，用于计算自注意力头中的加权平均值
    a3_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 创建 MgpstrEmbeddings 类，用于 2D 图像到补丁嵌入的转换
class MgpstrEmbeddings(nn.Module):
    """2D Image to Patch Embedding"""
    # 定义一个名为__init__的初始化方法，接受一个MgpstrConfig类型的config参数
    def __init__(self, config: MgpstrConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 获取图像大小，如果config.image_size是可迭代对象，则直接使用；否则将其作为元组的第一个元素，并复制给image_size
        image_size = (
            config.image_size
            if isinstance(config.image_size, collections.abc.Iterable)
            else (config.image_size, config.image_size)
        )
        # 获取图像分块大小，如果config.patch_size是可迭代对象，则直接使用；否则将其作为元组的第一个元素，并复制给patch_size
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        # 将计算得到的image_size赋值给self.image_size
        self.image_size = image_size
        # 将计算得到的patch_size赋值给self.patch_size
        self.patch_size = patch_size
        # 计算图像的网格大小，通过整除计算得到，赋值给self.grid_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        # 计算总的分块数量，赋值给self.num_patches
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 如果config.distilled为True，则self.num_tokens为2，否则为1
        self.num_tokens = 2 if config.distilled else 1
    
        # 使用nn.Conv2d创建一个卷积层，参数包括输入通道数、隐藏层大小、卷积核大小、步幅为分块大小
        self.proj = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
    
        # 创建一个可学习的参数，用于表示类别的嵌入向量，大小为1*1*config.hidden_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
    
        # 创建一个可学习的位置嵌入向量，大小为1*(num_patches + num_tokens)*config.hidden_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, config.hidden_size))
        # 创建一个dropout层，参数为dropout概率
        self.pos_drop = nn.Dropout(p=config.drop_rate)
    
    # 定义前向传播方法forward，接受输入的像素值张量pixel_values
    def forward(self, pixel_values):
        # 获取输入像素值张量的维度信息
        batch_size, channel, height, width = pixel_values.shape
        # 检查输入图像大小是否与模型定义的图像大小一致，如果不一致则抛出ValueError异常
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
    
        # 将输入像素值张量通过卷积层self.proj，得到分块的嵌入向量patch_embeddings
        patch_embeddings = self.proj(pixel_values)
        # 将分块的嵌入向量patch_embeddings展开，并交换维度，得到BCHW -> BNC的张量
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)  # BCHW -> BNC
    
        # 创建一个类别嵌入向量，通过复制cls_token得到大小为batch_size*1*config.hidden_size的张量
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 将类别嵌入向量和分块嵌入向量拼接在一起，沿着第1维度拼接，得到embedding_output
        embedding_output = torch.cat((cls_tokens, patch_embeddings), dim=1)
        # 将位置嵌入向量和embedding_output相加，得到新的embedding_output
        embedding_output = embedding_output + self.pos_embed
        # 对embedding_output进行dropout操作
        embedding_output = self.pos_drop(embedding_output)
    
        # 返回处理后的embedding_output张量
        return embedding_output
class MgpstrMlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, config: MgpstrConfig, hidden_features):
        super().__init__()
        # 如果未指定隐藏层特征数，则使用配置中的隐藏层大小
        hidden_features = hidden_features or config.hidden_size
        # 第一个全连接层，将隐藏状态的维度映射到隐藏特征维度
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
        # 激活函数 GELU
        self.act = nn.GELU()
        # 第二个全连接层，将隐藏特征维度映射回隐藏层大小
        self.fc2 = nn.Linear(hidden_features, config.hidden_size)
        # Dropout 层，用于随机失活以防止过拟合
        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, hidden_states):
        # 第一个全连接层的前向传播
        hidden_states = self.fc1(hidden_states)
        # GELU 激活函数
        hidden_states = self.act(hidden_states)
        # Dropout 操作
        hidden_states = self.drop(hidden_states)
        # 第二个全连接层的前向传播
        hidden_states = self.fc2(hidden_states)
        # 再次进行 Dropout 操作
        hidden_states = self.drop(hidden_states)
        return hidden_states


class MgpstrAttention(nn.Module):
    def __init__(self, config: MgpstrConfig):
        super().__init__()
        # 注意力头的数量
        self.num_heads = config.num_attention_heads
        # 注意力头的维度
        head_dim = config.hidden_size // config.num_attention_heads
        # 缩放因子，用于缩放点积注意力
        self.scale = head_dim**-0.5

        # 查询-键-值(QKV) 线性映射层
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        # Dropout 层，用于 QKV 的注意力权重
        self.attn_drop = nn.Dropout(config.attn_drop_rate)
        # 投影层，将注意力输出映射回隐藏层大小
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        # Dropout 层，用于投影层的输出
        self.proj_drop = nn.Dropout(config.drop_rate)

    def forward(self, hidden_states):
        batch_size, num, channel = hidden_states.shape
        # QKV 映射及维度重塑
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, num, 3, self.num_heads, channel // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]  # 使 torchscript 符合要求（不能将张量用作元组）

        # 计算注意力权重
        attention_probs = (query @ key.transpose(-2, -1)) * self.scale
        attention_probs = attention_probs.softmax(dim=-1)
        # 对注意力权重应用 Dropout
        attention_probs = self.attn_drop(attention_probs)

        # 计算上下文向量
        context_layer = (attention_probs @ value).transpose(1, 2).reshape(batch_size, num, channel)
        # 通过投影层映射上下文向量
        context_layer = self.proj(context_layer)
        # 对投影输出应用 Dropout
        context_layer = self.proj_drop(context_layer)
        return (context_layer, attention_probs)


class MgpstrLayer(nn.Module):
    def __init__(self, config: MgpstrConfig, drop_path=None):
        super().__init__()
        # LayerNorm 层，用于归一化输入
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 自注意力层
        self.attn = MgpstrAttention(config)
        # 注意力模块的 DropPath，用于随机深度丢弃
        # 注意：这里是为了与 dropout 进行比较，看是否更适合
        self.drop_path = MgpstrDropPath(drop_path) if drop_path is not None else nn.Identity()
        # LayerNorm 层，用于归一化自注意力输出
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # MLP 层，用于特征提取和转换
        mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        self.mlp = MgpstrMlp(config, mlp_hidden_dim)
    # 定义模型的前向传播过程
    def forward(self, hidden_states):
        # 计算自注意力输出和相关输出
        self_attention_outputs = self.attn(self.norm1(hidden_states))
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1]
    
        # 进行第一个残差连接
        hidden_states = self.drop_path(attention_output) + hidden_states
    
        # 进行第二个残差连接
        layer_output = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
    
        # 返回层输出和其他输出
        outputs = (layer_output, outputs)
        return outputs
# 定义 MgpstrEncoder 类，继承 nn.Module
class MgpstrEncoder(nn.Module):
    # 初始化函数
    def __init__(self, config: MgpstrConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 根据配置信息计算丢弃率decay，生成一个与隐藏层数量等长的列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # 用 MgpstrLayer 构建隐藏层列表
        self.blocks = nn.Sequential(
            *[MgpstrLayer(config=config, drop_path=dpr[i]) for i in range(config.num_hidden_layers)]
        )

    # 前向传播函数
    def forward(self, hidden_states, output_attentions=False, output_hidden_states=False, return_dict=True):
        # 如果需要输出注意力权重或中间隐藏状态，初始化相应的元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历隐藏层列表
        for _, blk in enumerate(self.blocks):
            # 如果需要输出中间隐藏状态，将当前隐藏状态加入到元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 将隐藏状态输入到当前隐藏层
            layer_outputs = blk(hidden_states)
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，将当前注意力权重加入到元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出中间隐藏状态，将最终隐藏状态加入到元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据返回字典的要求返回最终结果
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# 定义 MgpstrA3Module 类，继承 nn.Module
class MgpstrA3Module(nn.Module):
    # 初始化函数
    def __init__(self, config: MgpstrConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 创建令牌归一化层
        self.token_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建令牌学习器层
        self.tokenLearner = nn.Sequential(
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=(1, 1), stride=1, groups=8, bias=False),
            nn.Conv2d(config.hidden_size, config.max_token_length, kernel_size=(1, 1), stride=1, bias=False),
        )
        # 创建特征提取层
        self.feat = nn.Conv2d(
            config.hidden_size, config.hidden_size, kernel_size=(1, 1), stride=1, groups=8, bias=False
        )
        # 创建特征归一化层
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数
    def forward(self, hidden_states):
        # 对输入的隐藏状态进行归一化
        hidden_states = self.token_norm(hidden_states)
        # 对归一化后的隐藏状态进行维度变换
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
        # 通过令牌学习器层提取注意力权重
        selected = self.tokenLearner(hidden_states)
        selected = selected.flatten(2)
        attentions = F.softmax(selected, dim=-1)

        # 通过特征提取层提取特征
        feat = self.feat(hidden_states)
        feat = feat.flatten(2).transpose(1, 2)
        # 使用注意力权重加权平均特征
        feat = torch.einsum("...si,...id->...sd", attentions, feat)
        # 对加权平均后的特征进行归一化
        a3_out = self.norm(feat)

        # 返回加权平均特征和注意力权重
        return (a3_out, attentions)


# 定义 MgpstrPreTrainedModel 类，继承 PreTrainedModel
class MgpstrPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 MgpstrConfig
    config_class = MgpstrConfig
    # 设置基础模型前缀为 "mgp_str"
    base_model_prefix = "mgp_str"
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果是 MgpstrEmbeddings 类型的模块
        if isinstance(module, MgpstrEmbeddings):
            # 对位置嵌入和类别标记进行截断正态分布初始化
            nn.init.trunc_normal_(module.pos_embed, mean=0.0, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.cls_token, mean=0.0, std=self.config.initializer_range)
        # 如果是 nn.Linear 或 nn.Conv2d 类型的模块
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对权重进行截断正态分布初始化
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 nn.LayerNorm 类型的模块
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
# 定义了 MGP-STR 模型的文档字符串的起始部分，包含模型的介绍和参数说明
MGP_STR_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MgpstrConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义了 MGP-STR 模型的输入文档字符串，描述了输入参数的含义
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

# 使用装饰器添加模型类的文档字符串，包含模型的介绍和参数说明
@add_start_docstrings(
    "The bare MGP-STR Model transformer outputting raw hidden-states without any specific head on top.",
    MGP_STR_START_DOCSTRING,
)
# 定义 MGP-STR 模型类
class MgpstrModel(MgpstrPreTrainedModel):
    # 初始化方法，接受模型配置参数并初始化模型
    def __init__(self, config: MgpstrConfig):
        super().__init__(config)
        self.config = config
        # 初始化模型的嵌入层
        self.embeddings = MgpstrEmbeddings(config)
        # 初始化模型的编码器
        self.encoder = MgpstrEncoder(config)

    # 获取输入嵌入层的方法
    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.proj

    # 重写模型的前向传播方法，并添加文档字符串说明输入参数的含义
    @add_start_docstrings_to_model_forward(MGP_STR_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 确定是否需要输出注意力权重信息，若未指定，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否需要输出隐藏层状态信息，若未指定，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否需要返回字典格式结果，若未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查是否提供了像素值，若未提供，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值输入embedding层，获取嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传入编码器，获取编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 若不需要返回字典格式结果，则直接返回编码器输出
        if not return_dict:
            return encoder_outputs
        # 否则使用BaseModelOutput将编码器输出组装成字典格式结果并返回
        return BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
```  
# 使用add_start_docstrings装饰器添加模型文档字符串，描述了MGP-STR模型的特点和用途
# 继承自MgpstrPreTrainedModel的MgpstrForSceneTextRecognition类，用于场景文本识别，包含三个分类头部（三个A^3模块和三个线性层）
class MgpstrForSceneTextRecognition(MgpstrPreTrainedModel):
    # 指定配置类
    config_class = MgpstrConfig
    # 主要输入的特征名为"pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法
    def __init__(self, config: MgpstrConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存标签数量
        self.num_labels = config.num_labels
        # 创建MgpstrModel实例
        self.mgp_str = MgpstrModel(config)
        # 创建字符A^3模块实例
        self.char_a3_module = MgpstrA3Module(config)
        # 创建BPE A^3模块实例
        self.bpe_a3_module = MgpstrA3Module(config)
        # 创建词片段A^3模块实例
        self.wp_a3_module = MgpstrA3Module(config)
        # 创建字符头部线性层
        self.char_head = nn.Linear(config.hidden_size, config.num_character_labels)
        # 创建BPE头部线性层
        self.bpe_head = nn.Linear(config.hidden_size, config.num_bpe_labels)
        # 创建词片段头部线性层
        self.wp_head = nn.Linear(config.hidden_size, config.num_wordpiece_labels)

    # 使用add_start_docstrings_to_model_forward装饰器添加模型前向传播方法的文档字符串，描述输入参数说明和返回结果类型
    # 替换返回值文档字符串的类型并指定配置类
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_a3_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    ) -> Union[Tuple[torch.FloatTensor], MgpstrModelOutput]:
        r"""
        output_a3_attentions (`bool`, *optional*):
            是否返回a3模块的注意力张量。有关更多详细信息，请参见返回张量中的`a3_attentions`。

        Returns:
            返回值：Union[Tuple[torch.FloatTensor], MgpstrModelOutput]

        Example:

        ```py
        >>> from transformers import (
        ...     MgpstrProcessor,
        ...     MgpstrForSceneTextRecognition,
        ... )
        >>> import requests
        >>> from PIL import Image

        >>> # 从IIIT-5k数据集加载图像
        >>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> processor = MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base")
        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

        >>> model = MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base")

        >>> # 推理
        >>> outputs = model(pixel_values)
        >>> out_strs = processor.batch_decode(outputs.logits)
        >>> out_strs["generated_text"]
        '["ticket"]'
        ```"""
        # 如果指定了output_attention，则使用指定值，否则使用模型配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果指定了output_hidden_states，则使用指定值，否则使用模型配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果指定了return_dict，则使用指定值，否则使用模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用Mgpstr模型对输入进行处理
        mgp_outputs = self.mgp_str(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = mgp_outputs[0]

        # 使用字符a3模块处理序列输出
        char_a3_out, char_attention = self.char_a3_module(sequence_output)
        # 使用bpe a3模块处理序列输出
        bpe_a3_out, bpe_attention = self.bpe_a3_module(sequence_output)
        # 使用wp a3模块处理序列输出
        wp_a3_out, wp_attention = self.wp_a3_module(sequence_output)

        # 计算字符a3模块的logits
        char_logits = self.char_head(char_a3_out)
        # 计算bpe a3模块的logits
        bpe_logits = self.bpe_head(bpe_a3_out)
        # 计算wp a3模块的logits
        wp_logits = self.wp_head(wp_a3_out)

        # 如果需要返回a3模块的所有注意力，则组合成元组，否则为None
        all_a3_attentions = (char_attention, bpe_attention, wp_attention) if output_a3_attentions else None
        all_logits = (char_logits, bpe_logits, wp_logits)

        # 如果不需要返回字典，则将结果组成元组返回
        if not return_dict:
            outputs = (all_logits, all_a3_attentions) + mgp_outputs[1:]
            return tuple(output for output in outputs if output is not None)
        # 返回MgpstrModelOutput对象
        return MgpstrModelOutput(
            logits=all_logits,
            hidden_states=mgp_outputs.hidden_states,
            attentions=mgp_outputs.attentions,
            a3_attentions=all_a3_attentions,
        )
```