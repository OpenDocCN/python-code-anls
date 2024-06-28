# `.\models\swiftformer\modeling_swiftformer.py`

```py
# 设置编码格式为 UTF-8

# 版权声明和许可协议信息
# Copyright 2023 MBZUAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch SwiftFormer model.
"""

import collections.abc
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义模块和类
from ...activations import ACT2CLS
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_swiftformer import SwiftFormerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置参数说明
_CONFIG_FOR_DOC = "SwiftFormerConfig"

# 用于文档的检查点说明
_CHECKPOINT_FOR_DOC = "MBZUAI/swiftformer-xs"

# 预期输出形状的说明
_EXPECTED_OUTPUT_SHAPE = [1, 220, 7, 7]

# 图像分类任务的检查点说明
_IMAGE_CLASS_CHECKPOINT = "MBZUAI/swiftformer-xs"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型的存档列表
SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MBZUAI/swiftformer-xs",
    # 查看所有 SwiftFormer 模型，请访问 https://huggingface.co/models?filter=swiftformer
]

class SwiftFormerPatchEmbedding(nn.Module):
    """
    Patch Embedding Layer constructed of two 2D convolutional layers.

    输入: 形状为 `[batch_size, in_channels, height, width]` 的张量

    输出: 形状为 `[batch_size, out_channels, height/4, width/4]` 的张量
    """

    def __init__(self, config: SwiftFormerConfig):
        super().__init__()

        in_chs = config.num_channels
        out_chs = config.embed_dims[0]
        
        # 定义补丁嵌入层，包括两个二维卷积层
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs // 2, eps=config.batch_norm_eps),
            nn.ReLU(),
            nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs, eps=config.batch_norm_eps),
            nn.ReLU(),
        )

    def forward(self, x):
        # 执行补丁嵌入层的前向传播
        return self.patch_embedding(x)

# 从 transformers.models.beit.modeling_beit.drop_path 复制过来的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    按样本（在残差块的主路径中应用）随机删除路径（随机深度）。

    Args:
        input (torch.Tensor): 输入张量
        drop_prob (float, optional): 删除概率，默认为 0.0
        training (bool, optional): 是否在训练模式中，默认为 False

    Returns:
        torch.Tensor: 处理后的张量
    """
    # 如果 drop_prob 为 0.0 或者不处于训练模式，则直接返回输入 input，不进行 Dropout
    if drop_prob == 0.0 or not training:
        return input
    
    # 计算保留节点的概率
    keep_prob = 1 - drop_prob
    
    # 根据输入张量的形状，创建一个随机张量，用于决定每个节点是否保留
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度的张量，而不仅仅是二维卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化（取整）
    
    # 将输入张量除以 keep_prob，然后乘以随机张量，以实现 Dropout 操作
    output = input.div(keep_prob) * random_tensor
    
    # 返回经过 Dropout 后的输出张量
    return output
# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Swiftformer
class SwiftFormerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数，用于在训练时按照给定的概率丢弃部分隐藏状态
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回描述当前实例的字符串，包括 drop_prob 的概率值
        return "p={}".format(self.drop_prob)


class SwiftFormerEmbeddings(nn.Module):
    """
    Embeddings layer consisting of a single 2D convolutional and batch normalization layer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height/stride, width/stride]`
    """

    def __init__(self, config: SwiftFormerConfig, index: int):
        super().__init__()

        # 从配置中获取所需的参数
        patch_size = config.down_patch_size
        stride = config.down_stride
        padding = config.down_pad
        embed_dims = config.embed_dims

        # 获取输入和输出通道数
        in_chans = embed_dims[index]
        embed_dim = embed_dims[index + 1]

        # 确保 patch_size、stride 和 padding 是可迭代对象
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)

        # 定义卷积和批量归一化层
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim, eps=config.batch_norm_eps)

    def forward(self, x):
        # 前向传播过程，依次通过卷积和批量归一化层处理输入张量 x
        x = self.proj(x)
        x = self.norm(x)
        return x


class SwiftFormerConvEncoder(nn.Module):
    """
    `SwiftFormerConvEncoder` with 3*3 and 1*1 convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * dim)

        # 定义深度可分离卷积、批量归一化、点卷积层和激活函数
        self.depth_wise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=config.batch_norm_eps)
        self.point_wise_conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.point_wise_conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = nn.Identity()  # 默认情况下不应用 drop path
        self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        input = x
        # 执行深度可分离卷积、批量归一化、点卷积和激活函数
        x = self.depth_wise_conv(x)
        x = self.norm(x)
        x = self.point_wise_conv1(x)
        x = self.act(x)
        x = self.point_wise_conv2(x)
        # 应用 drop path 和缩放因子到输入
        x = input + self.drop_path(self.layer_scale * x)
        return x


class SwiftFormerMlp(nn.Module):
    """
    """
    MLP layer with 1*1 convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    # 初始化函数，接受配置对象和输入特征数
    def __init__(self, config: SwiftFormerConfig, in_features: int):
        super().__init__()  # 调用父类的构造函数
        hidden_features = int(in_features * config.mlp_ratio)  # 计算隐藏层特征数
        self.norm1 = nn.BatchNorm2d(in_features, eps=config.batch_norm_eps)  # 批量归一化层
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 第一个卷积层，1x1卷积
        act_layer = ACT2CLS[config.hidden_act]  # 获取激活函数类
        self.act = act_layer()  # 实例化激活函数对象
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)  # 第二个卷积层，1x1卷积
        self.drop = nn.Dropout(p=0.0)  # Dropout层，概率为0.0，即不进行dropout操作

    # 前向传播函数，接受输入张量x
    def forward(self, x):
        x = self.norm1(x)  # 应用批量归一化
        x = self.fc1(x)  # 第一个卷积层的计算
        x = self.act(x)  # 应用激活函数
        x = self.drop(x)  # 应用dropout
        x = self.fc2(x)  # 第二个卷积层的计算
        x = self.drop(x)  # 再次应用dropout
        return x  # 返回计算结果张量
class SwiftFormerEncoderBlock(nn.Module):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2)
    SwiftFormerEfficientAdditiveAttention, and (3) MLP block.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels,height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int):
        super().__init__()

        # 定义局部表示模块，使用3*3深度卷积和点卷积
        self.local_representations = SwiftFormerLocalRepresentation(config, dim)
        
        # 定义注意力模块，使用高效加性注意力
        self.attention = SwiftFormerEfficientAdditiveAttention(config, dim)
        
        # 定义MLP块
        self.mlp_block = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        # 应用局部表示模块
        x = self.local_representations(x)
        
        # 应用注意力模块
        x = self.attention(x)
        
        # 应用MLP块
        x = self.mlp_block(x)
        
        return x
    def __init__(self, config: SwiftFormerConfig, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()

        # 从配置对象中获取层缩放初始化值和是否使用层缩放的标志
        layer_scale_init_value = config.layer_scale_init_value
        use_layer_scale = config.use_layer_scale

        # 创建本地表示层、注意力层、线性层和DropPath层
        self.local_representation = SwiftFormerLocalRepresentation(config, dim=dim)
        self.attn = SwiftFormerEfficientAdditiveAttention(config, dim=dim)
        self.linear = SwiftFormerMlp(config, in_features=dim)
        self.drop_path = SwiftFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale

        # 如果使用层缩放，则创建两个层缩放参数
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True
            )

    def forward(self, x):
        # 将输入x传递给本地表示层处理
        x = self.local_representation(x)
        batch_size, channels, height, width = x.shape
        
        # 如果使用层缩放，则将层缩放因子应用于注意力层和线性层的输出
        if self.use_layer_scale:
            # 计算注意力层的输出，并应用层缩放因子和DropPath层
            x = x + self.drop_path(
                self.layer_scale_1
                * self.attn(x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels))
                .reshape(batch_size, height, width, channels)
                .permute(0, 3, 1, 2)
            )
            # 计算线性层的输出，并应用层缩放因子和DropPath层
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            # 如果不使用层缩放，则直接应用注意力层和线性层的输出与DropPath层
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels))
                .reshape(batch_size, height, width, channels)
                .permute(0, 3, 1, 2)
            )
            x = x + self.drop_path(self.linear(x))
        
        # 返回处理后的输出张量x
        return x
class SwiftFormerStage(nn.Module):
    """
    A Swiftformer stage consisting of a series of `SwiftFormerConvEncoder` blocks and a final
    `SwiftFormerEncoderBlock`.

    Input: tensor in shape `[batch_size, channels, height, width]`

    Output: tensor in shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, index: int) -> None:
        super().__init__()

        layer_depths = config.depths
        dim = config.embed_dims[index]
        depth = layer_depths[index]

        blocks = []
        for block_idx in range(depth):
            # 计算当前 block 的 drop path rate
            block_dpr = config.drop_path_rate * (block_idx + sum(layer_depths[:index])) / (sum(layer_depths) - 1)

            if depth - block_idx <= 1:
                # 如果是最后一个 block，则添加 SwiftFormerEncoderBlock
                blocks.append(SwiftFormerEncoderBlock(config, dim=dim, drop_path=block_dpr))
            else:
                # 否则添加 SwiftFormerConvEncoder
                blocks.append(SwiftFormerConvEncoder(config, dim=dim))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        # 依次通过所有的 block 进行前向传播
        for block in self.blocks:
            input = block(input)
        return input


class SwiftFormerEncoder(nn.Module):
    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__()
        self.config = config

        embed_dims = config.embed_dims
        downsamples = config.downsamples
        layer_depths = config.depths

        # Transformer model
        network = []
        for i in range(len(layer_depths)):
            # 创建 SwiftFormerStage，并将其添加到网络中
            stage = SwiftFormerStage(config=config, index=i)
            network.append(stage)
            if i >= len(layer_depths) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # 如果需要下采样或者维度变化，则添加 SwiftFormerEmbeddings
                network.append(SwiftFormerEmbeddings(config, index=i))
        self.network = nn.ModuleList(network)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for block in self.network:
            # 依次通过所有的 block 进行前向传播
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回 BaseModelOutputWithNoAttention 对象
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class SwiftFormerPreTrainedModel(PreTrainedModel):
    """
    This class is not completed in the provided snippet.
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 SwiftFormerConfig 类作为配置类
    config_class = SwiftFormerConfig
    # 模型的基础名称前缀为 "swiftformer"
    base_model_prefix = "swiftformer"
    # 主输入的名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果模块是线性层或者二维卷积层，使用截断正态分布初始化权重
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            # 如果存在偏置项，则初始化为常数 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # 如果模块是层归一化层，则初始化偏置项为常数 0，权重为常数 1.0
        elif isinstance(module, (nn.LayerNorm)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
# SWIFTFORMER_START_DOCSTRING 变量，包含了 SwiftFormerModel 类的文档字符串，描述了这个模型是一个 PyTorch 的 nn.Module 子类，
# 可以像普通的 PyTorch 模块一样使用，详细的使用和行为相关信息可以查阅 PyTorch 文档。

# SWIFTFORMER_INPUTS_DOCSTRING 变量，包含了 SwiftFormerModel 类的输入文档字符串，描述了模型的输入参数和返回值。
# pixel_values 参数是一个 torch.FloatTensor，表示像素值，形状为 (batch_size, num_channels, height, width)。
# output_hidden_states 参数是一个可选的布尔值，指定是否返回所有层的隐藏状态。
# return_dict 参数是一个可选的布尔值，指定是否返回一个 ModelOutput 对象而不是简单的元组。

# 使用 add_start_docstrings 装饰器为 SwiftFormerModel 类添加了一个开头的文档字符串，描述了它是一个输出原始隐藏状态的
# SwiftFormer 模型变压器，没有特定的顶部头。

class SwiftFormerModel(SwiftFormerPreTrainedModel):
    def __init__(self, config: SwiftFormerConfig):
        # 调用 SwiftFormerPreTrainedModel 的初始化方法，并传入配置对象 config
        super().__init__(config)
        # 将传入的配置对象保存到 self.config 中
        self.config = config

        # 创建 SwiftFormerPatchEmbedding 对象并保存到 self.patch_embed
        self.patch_embed = SwiftFormerPatchEmbedding(config)
        
        # 创建 SwiftFormerEncoder 对象并保存到 self.encoder
        self.encoder = SwiftFormerEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 装饰器为 forward 方法添加了开头的文档字符串，
    # 描述了方法的输入参数和输出预期，模型用于 vision 模态。

    # forward 方法，接收 pixel_values, output_hidden_states 和 return_dict 作为输入参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        r""" """
        # 设置函数签名，指定返回类型为元组或BaseModelOutputWithNoAttention类的实例

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果output_hidden_states不为None，则使用其自身值；否则使用self.config.output_hidden_states的值

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果return_dict不为None，则使用其自身值；否则使用self.config.use_return_dict的值

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 如果pixel_values为None，则抛出值错误异常，要求指定pixel_values

        embedding_output = self.patch_embed(pixel_values)
        # 使用self.patch_embed方法对pixel_values进行嵌入编码

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用self.encoder方法对嵌入输出进行编码，传入参数为embedding_output、output_hidden_states和return_dict

        if not return_dict:
            return tuple(v for v in encoder_outputs if v is not None)
        # 如果return_dict为False，则返回encoder_outputs中所有非None值的元组

        return BaseModelOutputWithNoAttention(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )
        # 使用BaseModelOutputWithNoAttention类创建一个实例，传入encoder_outputs的最后隐藏状态和隐藏状态列表作为参数
# 使用自定义的文档字符串装饰器为类添加描述信息，指定其是基于SwiftFormer模型的图像分类器
@add_start_docstrings(
    """
    SwiftFormer Model transformer with an image classification head on top (e.g. for ImageNet).
    """,
    SWIFTFORMER_START_DOCSTRING,
)
# 声明SwiftFormerForImageClassification类，继承自SwiftFormerPreTrainedModel
class SwiftFormerForImageClassification(SwiftFormerPreTrainedModel):
    
    # 初始化方法，接受一个SwiftFormerConfig类型的参数config，并调用其父类的初始化方法
    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__(config)

        # 从config中获取嵌入维度
        embed_dims = config.embed_dims

        # 设置类别数量为config中指定的类别数
        self.num_labels = config.num_labels
        # 创建SwiftFormerModel模型实例，并赋值给self.swiftformer
        self.swiftformer = SwiftFormerModel(config)

        # 分类器头部
        # 根据最后一个嵌入维度设置批量归一化层
        self.norm = nn.BatchNorm2d(embed_dims[-1], eps=config.batch_norm_eps)
        # 如果有类别数量大于0，则创建线性层作为分类头部，否则创建一个恒等映射（nn.Identity()）
        self.head = nn.Linear(embed_dims[-1], self.num_labels) if self.num_labels > 0 else nn.Identity()
        # 同上，创建一个用于距离度量的线性层或者恒等映射
        self.dist_head = nn.Linear(embed_dims[-1], self.num_labels) if self.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用自定义的文档字符串装饰器为forward方法添加描述信息，指定其输入和输出类型
    @add_start_docstrings_to_model_forward(SWIFTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 定义forward方法，接受像素值pixel_values、标签labels以及其他可选参数，并返回一个字典或者张量，具体依赖于return_dict参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数声明未完成，余下的代码在下一个注释中
        ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用传入的 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 运行基础模型，将输入的像素值传递给 Swiftformer 模型
        outputs = self.swiftformer(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果 return_dict 为 False，则从 outputs 中获取最后一个隐藏状态；否则从 outputs 的第一个元素获取序列输出
        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]

        # 将序列输出应用归一化操作
        sequence_output = self.norm(sequence_output)

        # 将归一化后的序列输出展平，然后在第二个维度上取平均值
        sequence_output = sequence_output.flatten(2).mean(-1)

        # 将平均值后的序列输出传递给分类头部模型和蒸馏头部模型
        cls_out = self.head(sequence_output)
        distillation_out = self.dist_head(sequence_output)

        # 计算 logits，即分类头部模型输出和蒸馏头部模型输出的平均值
        logits = (cls_out + distillation_out) / 2

        # 计算损失值
        loss = None
        if labels is not None:
            # 根据配置文件中的问题类型确定损失函数的类型
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

        # 如果 return_dict 为 False，则返回 logits 和其他输出；否则返回 ImageClassifierOutputWithNoAttention 对象
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
```