# `.\models\focalnet\modeling_focalnet.py`

```py
# 设置文件编码为 utf-8
# 版权声明，版权所有 2023 年微软研究和 HuggingFace 公司团队保留所有权利
# 根据 Apache 许可证 2.0 版本进行许可
# 只有在遵守许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得使用该软件
# 根据许可证分发的软件是基于"原样"分发的，没有任何担保或条件，无论是明示的还是暗示的
# 有关详细语言，请查看许可证，以获取特定语言的权限和限制
""" PyTorch FocalNet 模型。"""


# 导入必要的库和模块
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_focalnet import FocalNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的常量和注释
# FocalNetConfig 用于文档的配置
_CONFIG_FOR_DOC = "FocalNetConfig"
# 模型检查点
_CHECKPOINT_FOR_DOC = "microsoft/focalnet-tiny"
# 预期输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]

# 图像分类文档
# 图像分类检查点
_IMAGE_CLASS_CHECKPOINT = "microsoft/focalnet-tiny"
# 预期输出
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/focalnet-tiny",
    # 查看所有 FocalNet 模型 https://huggingface.co/models?filter=focalnet
]


@dataclass
class FocalNetEncoderOutput(ModelOutput):
    """
    FocalNet 编码器的输出，可能包含隐藏状态。
    # 定义函数参数和返回值的类型注解，说明下列变量的数据类型
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            表示模型最后一层的输出的隐藏状态的序列，形状为(batch_size, sequence_length, hidden_size)。
    
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含了模型每一层的输出的隐藏状态的序列，形状为(batch_size, sequence_length, hidden_size)。
            当`output_hidden_states=True`或者`config.output_hidden_states=True`时，返回该元组。
        
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含了模型每一层的输出的隐藏状态的序列，形状为(batch_size, hidden_size, height, width)。
            当`output_hidden_states=True`或者`config.output_hidden_states=True`时，返回该元组。
    """
    
    # 初始化变量，用于存储模型的隐藏状态数据
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器定义 FocalNetModelOutput 类
@dataclass
class FocalNetModelOutput(ModelOutput):
    """
    FocalNet model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    # 定义类属性
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器定义 FocalNetMaskedImageModelingOutput 类
@dataclass
class FocalNetMaskedImageModelingOutput(ModelOutput):
    """
    FocalNet masked image model outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss. # 定义代表 masked image modeling (MLM) 损失的 torch.FloatTensor 类型的变量 loss，当提供 bool_masked_pos 时返回
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values. # 重建的像素值，类型为 torch.FloatTensor，形状为 (batch_size, num_channels, height, width)
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs. # 模型在每一层输出的隐藏状态，包括初始嵌入输出，类型为元组(tuple) torch.FloatTensor
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions. # 重塑以包含空间维度的模型在每一层输出的隐藏状态，包括初始嵌入输出，类型为元组(tuple) torch.FloatTensor
    """

    # 定义变量
    loss: Optional[torch.FloatTensor] = None
    reconstruction: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器定义 FocalNetImageClassifierOutput 类
@dataclass
class FocalNetImageClassifierOutput(ModelOutput):
    """
    FocalNet outputs for image classification.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    # 定义 loss 属性，类型为 Optional[torch.FloatTensor]
    loss: Optional[torch.FloatTensor] = None
    # 定义 logits 属性，类型为 torch.FloatTensor
    logits: torch.FloatTensor = None
    # 定义 hidden_states 属性，类型为 Optional[Tuple[torch.FloatTensor]]
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义 reshaped_hidden_states 属性，类型为 Optional[Tuple[torch.FloatTensor]]
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 定义 FocalNetEmbeddings 类
class FocalNetEmbeddings(nn.Module):
    """
    Construct the patch embeddings and layernorm. Optionally, also the mask token.
    """

    # 初始化方法
    def __init__(self, config, use_mask_token=False):
        super().__init__()

        # 创建 patch_embeddings 属性，类型为 FocalNetPatchEmbeddings
        self.patch_embeddings = FocalNetPatchEmbeddings(
            config=config,
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.embed_dim,
            use_conv_embed=config.use_conv_embed,
            is_stem=True,
        )
        # 获取 patch_grid 属性
        self.patch_grid = self.patch_embeddings.grid_size
        # 创建 mask_token 属性，类型为 nn.Parameter，根据 use_mask_token 的值决定是否创建
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        # 创建 norm 属性，类型为 nn.LayerNorm
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        # 创建 dropout 属性，类型为 nn.Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法
    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
        ) -> Tuple[torch.Tensor]:
        # 使用像素数值获取patch的嵌入和输出维度
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        # 对嵌入进行规范化
        embeddings = self.norm(embeddings)
        # 获取批处理大小、序列长度和嵌入的维度
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            # 创建与嵌入形状相同的mask_tokens
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 将被遮蔽的视觉标记替换为mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 对嵌入进行dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入和输出维度
        return embeddings, output_dimensions
class FocalNetPatchEmbeddings(nn.Module):
    def __init__(
        self,
        config,
        image_size,
        patch_size,
        num_channels,
        embed_dim,
        add_norm=False,
        use_conv_embed=False,
        is_stem=False,
    ):
        super().__init__()
        # 将图像大小转换为可迭代对象，如果已经是可迭代对象则保持不变
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        # 将patch大小转换为可迭代对象，如果已经是可迭代对象则保持不变
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算patch的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        # 计算网格大小
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        if use_conv_embed:
            # 如果选择使用卷积嵌入，则根据是否是干细胞进行不同的处理
            if is_stem:
                kernel_size = 7
                padding = 2
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2
            # 创建卷积层
            self.projection = nn.Conv2d(
                num_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
            )
        else:
            # 创建卷积层
            self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        if add_norm:
            # 添加 LayerNorm
            self.norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.norm = None

    def maybe_pad(self, pixel_values, height, width):
        # 如果宽度不是patch大小的整数倍，则进行填充
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        # 如果高度不是patch大小的整数倍，则进行填充
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 获取输入张量的形状
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            # 如果通道数不匹配配置文件中设置的通道数，则引发错误
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 对输入进行填充，使其能够被self.patch_size整除
        pixel_values = self.maybe_pad(pixel_values, height, width)
        # 将输入通过投影层进行嵌入
        embeddings = self.projection(pixel_values)
        # 获取嵌入后的形状
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        # 将嵌入展平并转置
        embeddings = embeddings.flatten(2).transpose(1, 2)

        if self.norm is not None:
            # 如果有LayerNorm，则对嵌入进行标准化处理
            embeddings = self.norm(embeddings)

        return embeddings, output_dimensions


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果dropout概率为0或者不处于训练状态，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留的概率
    keep_prob = 1 - drop_prob
    # 根据输入的维度创建与之对应的形状
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 在给定形状下生成符合均匀分布的随机数张量
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 对随机张量进行取整，使其成为二值化的张量
    random_tensor.floor_()  # binarize
    # 计算输出，使用随机张量二值化后作为选择保留的路径
    output = input.div(keep_prob) * random_tensor
    # 返回处理后的输出
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->FocalNet
class FocalNetDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用drop_path函数对隐藏状态进行处理
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回额外的表示信息
        return "p={}".format(self.drop_prob)


class FocalNetModulation(nn.Module):
    def __init__(self, config, index, dim, focal_factor=2, bias=True, projection_dropout=0.0):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化变量
        self.dim = dim
        self.focal_window = config.focal_windows[index]
        self.focal_level = config.focal_levels[index]
        self.focal_factor = focal_factor
        self.use_post_layernorm_in_modulation = config.use_post_layernorm_in_modulation
        self.normalize_modulator = config.normalize_modulator

        # 初始化线性投影层
        self.projection_in = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.projection_context = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        # 激活函数
        self.activation = nn.GELU()
        self.projection_out = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)
        self.focal_layers = nn.ModuleList()

        # 初始化焦点层的卷积核大小列表
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            # 对焦点层进行初始化
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2, bias=False
                    ),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_post_layernorm_in_modulation:
            # 使用后层归一化
            self.layernorm = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state:
                Input features with shape of (batch_size, height, width, num_channels)
        """
        num_channels = hidden_state.shape[-1]

        # pre linear projection
        x = self.projection_in(hidden_state).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (num_channels, num_channels, self.focal_level + 1), 1)

        # context aggreation
        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, level : level + 1]
        ctx_global = self.activation(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level :]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.projection_context(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_post_layernorm_in_modulation:
            x_out = self.layernorm(x_out)

        # post linear porjection
        x_out = self.projection_out(x_out)
        x_out = self.projection_dropout(x_out)
        return x_out
class FocalNetMlp(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        # 如果未提供输出特征数，则将其设为输入特征数
        out_features = out_features or in_features
        # 如果未提供隐藏层特征数，则将其设为输入特征数
        hidden_features = hidden_features or in_features
        # 创建第一个全连接层，输入特征数为 in_features，输出特征数为 hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 选择激活函数，根据配置文件中的隐藏层激活函数名称从预定义的字典中获取相应的激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 创建第二个全连接层，输入特征数为 hidden_features，输出特征数为 out_features
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 创建一个 dropout 层，用于进行随机失活
        self.drop = nn.Dropout(drop)

    def forward(self, hidden_state):
        # 前向传播过程，首先通过第一个全连接层
        hidden_state = self.fc1(hidden_state)
        # 使用激活函数对结果进行激活
        hidden_state = self.activation(hidden_state)
        # 对结果进行 dropout 处理
        hidden_state = self.drop(hidden_state)
        # 通过第二个全连接层
        hidden_state = self.fc2(hidden_state)
        # 再次进行 dropout 处理
        hidden_state = self.drop(hidden_state)
        return hidden_state


class FocalNetLayer(nn.Module):
    r"""Focal Modulation Network layer (block).

    Args:
        config (`FocalNetConfig`):
            Model config.
        index (`int`):
            Layer index.
        dim (`int`):
            Number of input channels.
        input_resolution (`Tuple[int]`):
            Input resulotion.
        drop_path (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.
    """

    def __init__(self, config, index, dim, input_resolution, drop_path=0.0):
        super().__init__()

        self.config = config

        # layer-specific attributes
        self.dim = dim
        self.input_resolution = input_resolution

        # general attributes
        self.drop = config.hidden_dropout_prob
        self.use_post_layernorm = config.use_post_layernorm

        # 创建 LayerNorm 层，用于归一化输入特征
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建 FocalNetModulation 实例，用于注意力调制
        self.modulation = FocalNetModulation(
            config=config,
            index=index,
            dim=dim,
            projection_dropout=self.drop,
        )

        # 根据给定的 drop_path 参数创建 FocalNetDropPath 层，用于随机深度跳连
        self.drop_path = FocalNetDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 创建第二个 LayerNorm 层，用于归一化特征
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 计算 MLP 隐藏层特征数
        mlp_hidden_dim = int(dim * config.mlp_ratio)
        # 创建 FocalNetMlp 实例，用于多层感知机操作
        self.mlp = FocalNetMlp(config=config, in_features=dim, hidden_features=mlp_hidden_dim, drop=self.drop)

        # 初始化 layerscale 参数，默认为 1.0
        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        # 如果配置中指定使用 layerscale，则创建相应的可学习参数
        if config.use_layerscale:
            self.gamma_1 = nn.Parameter(config.layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(config.layerscale_value * torch.ones((dim)), requires_grad=True)
    # 定义网络前向传播过程的函数
    def forward(self, hidden_state, input_dimensions):
        # 解包输入尺寸参数为高度和宽度
        height, width = input_dimensions
        # 从隐藏状态中提取批次大小、_（忽略）、和通道数
        batch_size, _, num_channels = hidden_state.shape
        # 定义快捷连接，指向隐藏状态的引用
        shortcut = hidden_state

        # Focal Modulation处理部分
        # 判断是否使用层后归一化，不使用则直接传递hidden_state，使用则先进行归一化处理
        hidden_state = hidden_state if self.use_post_layernorm else self.norm1(hidden_state)
        # 重构hidden_state的形状为(batch_size, height, width, num_channels)
        hidden_state = hidden_state.view(batch_size, height, width, num_channels)
        # 对hidden_state应用调制操作并重新形状为(batch_size, height * width, num_channels)
        hidden_state = self.modulation(hidden_state).view(batch_size, height * width, num_channels)
        # 判断是否使用层后归一化，使用则对hidden_state再次归一化
        hidden_state = hidden_state if not self.use_post_layernorm else self.norm1(hidden_state)

        # FFN (Feed-Forward Network) 部分
        # 将hidden_state与shortcut相加，再通过drop_path进行路径选择并乘以调节因子gamma_1
        hidden_state = shortcut + self.drop_path(self.gamma_1 * hidden_state)
        # 对hidden_state加上调制的结果，结果通过一个多层感知机处理，并乘以调节因子gamma_2
        # 根据是否使用层后归一化选择不同的处理顺序
        hidden_state = hidden_state + self.drop_path(
            self.gamma_2
            * (self.norm2(self.mlp(hidden_state)) if self.use_post_layernorm else self.mlp(self.norm2(hidden_state)))
        )

        # 返回最终的隐藏状态
        return hidden_state
# 定义 FocalNetStage 类，继承自 nn.Module
class FocalNetStage(nn.Module):
    # 初始化函数
    def __init__(self, config, index, input_resolution):
        super().__init__()

        # 保存配置、阶段数量、嵌入维度等参数
        self.config = config
        self.num_stages = len(config.depths)

        embed_dim = [config.embed_dim * (2**i) for i in range(self.num_stages)]
        dim = embed_dim[index]
        out_dim = embed_dim[index + 1] if (index < self.num_stages - 1) else None
        downsample = FocalNetPatchEmbeddings if (index < self.num_stages - 1) else None

        # 设置随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        drop_path = dpr[sum(config.depths[:index]) : sum(config.depths[: index + 1])]

        # 创建包含多个 FocalNetLayer 实例的列表
        self.layers = nn.ModuleList(
            [
                FocalNetLayer(
                    config=config,
                    index=index,
                    dim=dim,
                    input_resolution=input_resolution,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(config.depths[index])
            ]
        )

        # 如果有下采样，创建下采样实例，否则设置为 None
        if downsample is not None:
            self.downsample = downsample(
                config=config,
                image_size=input_resolution,
                patch_size=2,
                num_channels=dim,
                embed_dim=out_dim,
                add_norm=True,
                use_conv_embed=config.use_conv_embed,
                is_stem=False,
            )
        else:
            self.downsample = None

        # 初始状态下 pointing 设置为 False
        self.pointing = False

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int]) -> Tuple[torch.Tensor]:
        height, width = input_dimensions
        # 对每个 FocalNetLayer 实例进行前向传播
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, input_dimensions)

        # 记录下采样操作前的隐藏状态
        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height, width = input_dimensions
            # 转置和重塑输入，然后进行下采样
            hidden_states = hidden_states.transpose(1, 2).reshape(
                hidden_states_before_downsampling.shape[0], -1, height, width
            )
            hidden_states, output_dimensions = self.downsample(hidden_states)

        else:
            # 如果没有下采样，输出尺寸为输入尺寸的四元组
            output_dimensions = (height, width, height, width)

        # 返回阶段的输出
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        return stage_outputs


class FocalNetEncoder(nn.Module):
    # 初始化函数，传入配置和网格大小作为参数
    def __init__(self, config, grid_size):
        # 调用父类的初始化函数
        super().__init__()
        # 根据配置的深度确定阶段的数量
        self.num_stages = len(config.depths)
        # 保存配置参数
        self.config = config

        # 创建阶段列表，每个阶段都是一个 FocalNetStage 对象
        self.stages = nn.ModuleList(
            [
                FocalNetStage(
                    config=config,
                    index=i_layer,
                    # 计算每个阶段的输入分辨率
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                )
                for i_layer in range(self.num_stages)
            ]
        )

        # 初始化梯度检查点标志
        self.gradient_checkpointing = False

    # 前向传播函数，接受隐藏状态、输入维度作为参数，并返回隐藏状态或特定输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    # 定义函数的输入和输出类型为元组或 FocalNetEncoderOutput
    ) -> Union[Tuple, FocalNetEncoderOutput]:
        # 根据输出配置确定是否收集所有隐藏层状态信息
        all_hidden_states = () if output_hidden_states else None
        # 根据输出配置确定是否收集所有调整过形状的隐藏层状态信息
        all_reshaped_hidden_states = () if output_hidden_states else None

        # 如果需要输出所有隐藏层状态信息
        if output_hidden_states:
            # 获取隐藏状态的形状信息
            batch_size, _, hidden_size = hidden_states.shape
            # 重新排列隐藏状态的维度 b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            # 收集隐藏层状态信息和调整形状后的隐藏层状态信息
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        # 遍历所有阶段的模块
        for i, stage_module in enumerate(self.stages):
            # 如果启用梯度检查点并且处于训练阶段
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数计算阶段的输出
                stage_outputs = self._gradient_checkpointing_func(
                    stage_module.__call__,
                    hidden_states,
                    input_dimensions,
                )
            else:
                # 计算阶段的输出
                stage_outputs = stage_module(hidden_states, input_dimensions)

            # 更新隐藏状态、下采样前的隐藏状态以及输出维度
            hidden_states = stage_outputs[0]
            hidden_states_before_downsampling = stage_outputs[1]
            output_dimensions = stage_outputs[2]

            # 更新输入维度
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            # 如果需要输出所有隐藏层状态信息和下采样前的隐藏状态信息
            if output_hidden_states and output_hidden_states_before_downsampling:
                # 获取下采样前的隐藏状态的形状信息
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # 重新排列下采样前隐藏状态的维度 b (h w) c -> b c h w
                # 在这里使用原始（未下采样）的高度和宽度
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                # 收集下采样前的隐藏状态信息和调整形状后的隐藏状态信息
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            # 如果需要输出所有隐藏层状态信息但不需要输出下采样前的隐藏状态信息
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                # 获取隐藏状态的形状信息
                batch_size, _, hidden_size = hidden_states.shape
                # 重新排列隐藏状态的维度 b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                # 收集隐藏层状态信息和调整形状后的隐藏层状态信息
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

        # 如果不需要返回字典形式的结果
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回 FocalNetEncoderOutput 格式的结果
        return FocalNetEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )
# 从transformers.models.swin.modeling_swin.SwinPreTrainedModel 复制代码并修改类名、前缀和输入名称
class FocalNetPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化和一个下载和加载预训练模型的简单接口。
    """

    # 指定使用的配置类
    config_class = FocalNetConfig
    # 模型前缀
    base_model_prefix = "focalnet"
    # 主要输入名称
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重函数
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与TF版本略有不同，TF版本使用截断正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# FocalNet的起始文档字符串
FOCALNET_START_DOCSTRING = r"""
    这个模型是一个PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)子类。将其作为普通的PyTorch Module使用，
    并查阅PyTorch文档以获取与一般用法和行为有关的所有内容。

    参数：
        config ([`FocalNetConfig`]): 包含模型所有参数的模型配置类。
            使用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
"""

# FocalNet的输入文档字符串
FOCALNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。可以使用[`AutoImageProcessor`]获得像素值。查看[`AutoImageProcessor.__call__`]以了解更多细节。

        output_hidden_states (`bool`, *可选*):
            是否返回所有层的隐藏状态。有关更多细节，请查看返回张量中的`hidden_states`。
        return_dict (`bool`, *可选*):
            是否返回[`~utils.ModelOutput`]而不是简单的元组。
"""


@add_start_docstrings(
    "The bare FocalNet Model outputting raw hidden-states without any specific head on top.", 
    FOCALNET_START_DOCSTRING,
)
class FocalNetModel(FocalNetPreTrainedModel):
    # 初始化函数，接受配置参数和两个额外参数，初始化神经网络模型
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        # 调用父类的初始化函数
        super().__init__(config)
        # 将传入的配置参数存储在实例中
        self.config = config
        # 计算深度信息和特征数量
        self.num_stages = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_stages - 1))
    
        # 创建嵌入层对象和编码器对象
        self.embeddings = FocalNetEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = FocalNetEncoder(config, self.embeddings.patch_grid)
    
        # 创建 LayerNorm 层，用于标准化
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        # 添加池化层，在需要时进行适应性平均池化
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None
    
        # 初始化权重并应用最终处理过程
        self.post_init()
    
    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings
    
    # 定义前向传播函数，接受不同类型的输入参数，返回输出
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FocalNetModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 设置隐藏状态输出参数和返回字典参数为配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 检查像素值是否为空，如果是则抛出错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
    
        # 获取嵌入输出和输入维度信息
        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
    
        # 对编码器进行处理
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取序列输出，并进行 LayerNorm 处理
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
    
        pooled_output = None
        # 如果存在池化层，则进行池化操作
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)
    
        # 如果不需返回字典，则返回序列输出和池化输出以及其他编码器输出
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output
    
        # 如果需要返回字典，则返回经过整理的输出
        return FocalNetModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
# 使用add_start_docstrings装饰器为FocalNetForMaskedImageModeling类添加文档字符串，指定了模型的描述和示例链接
# FOCALNET_START_DOCSTRING是定义好的引用文档字符串的标识符
class FocalNetForMaskedImageModeling(FocalNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 根据配置参数初始化FocalNetModel模型，并关闭池化层，使用mask token
        self.focalnet = FocalNetModel(config, add_pooling_layer=False, use_mask_token=True)

        # 计算特征数量
        self.num_stages = len(config.depths)
        num_features = int(config.embed_dim * 2 ** (self.num_stages - 1))
        # 创建解码器，使用卷积层和像素转换层
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=config.encoder_stride**2 * config.num_channels, kernel_size=1
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用add_start_docstrings_to_model_forward和replace_return_docstrings装饰器为forward方法添加文档字符串，指定了输入和输出的描述
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    FocalNet Model with an image classification head on top (a linear layer on top of the pooled output) e.g. for
    ImageNet.
    """,
    FOCALNET_START_DOCSTRING,
)
class FocalNetForImageClassification(FocalNetPreTrainedModel):
    # 从transformers.models.swin.modeling_swin.SwinForImageClassification.__init__拷贝代码，将Swin替换为FocalNet
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        # 根据配置参数初始化FocalNetModel模型
        self.focalnet = FocalNetModel(config)

        # 分类器头部，根据标签数量决定是线性层还是恒等映射
        self.classifier = (
            nn.Linear(self.focalnet.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用add_start_docstrings_to_model_forward和add_code_sample_docstrings装饰器为forward方法添加文档字符串，指定了输入、输出和代码示例
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FocalNetImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保返回字典已定义，如果未定义，则使用模型配置里的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过FocalNet模型计算输出
        outputs = self.focalnet(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出结果中获取池化后的特征
        pooled_output = outputs[1]

        # 使用分类器处理池化输出，得到分类结果
        logits = self.classifier(pooled_output)

        # 初始化损失值为None
        loss = None
        # 如果有标签，计算损失
        if labels is not None:
            # 确保问题类型被正确定义
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果只有一个标签，使用均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 否则使用均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 单标签分类使用交叉熵损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 多标签分类使用二元交叉熵损失
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典，则返回分类结果和特征输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则返回详细输出
        return FocalNetImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
@add_start_docstrings(
    """
    FocalNet backbone, to be used with frameworks like X-Decoder.
    """,
    FOCALNET_START_DOCSTRING,
)
class FocalNetBackbone(FocalNetPreTrainedModel, BackboneMixin):
    # 初始化函数，接收配置参数
    def __init__(self, config: FocalNetConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化骨干网
        super()._init_backbone(config)

        # 设置特征数列表
        self.num_features = [config.embed_dim] + config.hidden_sizes
        # 创建 FocalNet 模型
        self.focalnet = FocalNetModel(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，接收输入像素值
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/focalnet-tiny-lrf")
        >>> model = AutoBackbone.from_pretrained("microsoft/focalnet-tiny-lrf")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```py"""
        # 根据参数决定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据参数决定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 使用 FocalNet 模型进行前向传播
        outputs = self.focalnet(pixel_values, output_hidden_states=True, return_dict=True)

        # 获取隐藏状态
        hidden_states = outputs.reshaped_hidden_states

        # 构建特征图
        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        # 如果不返回字典，构建输出
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        # 返回 BackboneOutput 对象
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
```