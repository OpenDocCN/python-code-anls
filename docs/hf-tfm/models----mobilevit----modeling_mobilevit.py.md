# `.\transformers\models\mobilevit\modeling_mobilevit.py`

```py
# 设置文件编码格式为 utf-8
# 版权声明，版权归 Apple Inc. 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用该文件，未经许可不得使用该文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件以"原样"的基础分发，不提供任何担保或条件，无论明示或默示
# 请参阅许可证以获取有关语言授权权限和限制的详细信息
# 原始许可证链接：https://github.com/apple/ml-cvnets/blob/main/LICENSE
""" PyTorch MobileViT 模型."""


import math
from typing import Dict, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mobilevit import MobileViTConfig

# 获取 logger 实例
logger = logging.get_logger(__name__)


# 用于文档的配置信息
_CONFIG_FOR_DOC = "MobileViTConfig"

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "apple/mobilevit-small"
_EXPECTED_OUTPUT_SHAPE = [1, 640, 8, 8]

# 图像分类的检查点
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevit-small"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 可用的 MobileViT 预训练模型列表
MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "apple/mobilevit-small",
    "apple/mobilevit-x-small",
    "apple/mobilevit-xx-small",
    "apple/deeplabv3-mobilevit-small",
    "apple/deeplabv3-mobilevit-x-small",
    "apple/deeplabv3-mobilevit-xx-small",
    # 查看所有 MobileViT 模型请访问：https://huggingface.co/models?filter=mobilevit
]


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    确保所有层的通道数能够被 `divisor` 整除。该函数来自于原始的 TensorFlow 仓库，
    可以在以下链接找到具体实现：
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # 确保向下取整不会下降超过原值的10%
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

# 定义 MobileViTConvLayer 类
class MobileViTConvLayer(nn.Module):
    # 初始化函数，用于初始化 MobileViTBlock 类的实例
    def __init__(
        self,
        config: MobileViTConfig,  # MobileViTConfig 类型的配置参数
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        kernel_size: int,  # 卷积核大小
        stride: int = 1,  # 步长，默认为1
        groups: int = 1,  # 分组卷积的组数，默认为1
        bias: bool = False,  # 是否使用偏置，默认为 False
        dilation: int = 1,  # 空洞卷积的膨胀率，默认为1
        use_normalization: bool = True,  # 是否使用归一化，默认为 True
        use_activation: Union[bool, str] = True,  # 是否使用激活函数，默认为 True
    ) -> None:  # 返回类型为 None
        # 调用父类的初始化函数
        super().__init__()
        # 计算填充大小
        padding = int((kernel_size - 1) / 2) * dilation

        # 检查输入通道数是否能被分组数整除
        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        # 检查输出通道数是否能被分组数整除
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        # 创建卷积层对象
        self.convolution = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=out_channels,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            stride=stride,  # 步长
            padding=padding,  # 填充大小
            dilation=dilation,  # 空洞卷积膨胀率
            groups=groups,  # 分组卷积的组数
            bias=bias,  # 是否使用偏置
            padding_mode="zeros",  # 填充模式
        )

        # 如果使用归一化
        if use_normalization:
            # 创建 BatchNorm2d 归一化层对象
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,  # 归一化的特征通道数
                eps=1e-5,  # 数值稳定性参数 epsilon
                momentum=0.1,  # 动量
                affine=True,  # 是否进行仿射变换
                track_running_stats=True,  # 是否跟踪统计信息
            )
        else:
            # 如果不使用归一化，归一化层设为 None
            self.normalization = None

        # 如果使用激活函数
        if use_activation:
            # 如果激活函数是字符串类型
            if isinstance(use_activation, str):
                # 使用预定义的激活函数字典中的对应激活函数
                self.activation = ACT2FN[use_activation]
            # 如果隐藏层激活函数是字符串类型
            elif isinstance(config.hidden_act, str):
                # 使用预定义的激活函数字典中的对应激活函数
                self.activation = ACT2FN[config.hidden_act]
            else:
                # 否则使用配置中的隐藏层激活函数
                self.activation = config.hidden_act
        else:
            # 如果不使用激活函数，激活函数设为 None
            self.activation = None

    # 前向传播函数
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 使用卷积层进行前向传播
        features = self.convolution(features)
        # 如果使用归一化，进行归一化处理
        if self.normalization is not None:
            features = self.normalization(features)
        # 如果使用激活函数，进行激活处理
        if self.activation is not None:
            features = self.activation(features)
        # 返回处理后的特征
        return features
class MobileViTInvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        # 初始化函数，定义了 MobileViTInvertedResidual 类
        super().__init__()
        # 根据输入通道数和配置参数计算扩展后的通道数
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        # 如果步长不是1或2，抛出异常
        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        # 是否使用残差连接（stride为1且输入输出通道数相同）
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        # 1x1 卷积扩展通道
        self.expand_1x1 = MobileViTConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        # 3x3 深度可分离卷积
        self.conv_3x3 = MobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

        # 1x1 卷积减少通道
        self.reduce_1x1 = MobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 保存输入特征
        residual = features

        # 依次经过各层
        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        # 如果使用残差连接，则加上输入特征
        return residual + features if self.use_residual else features


class MobileViTMobileNetLayer(nn.Module):
    # MobileNet 层，包含多个 InvertedResidual 模块
    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int = 1, num_stages: int = 1
    ) -> None:
        super().__init__()

        # 初始化模块列表
        self.layer = nn.ModuleList()
        # 创建指定数量的 InvertedResidual 模块
        for i in range(num_stages):
            layer = MobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
            )
            self.layer.append(layer)
            in_channels = out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 依次经过每个 InvertedResidual 模块
        for layer_module in self.layer:
            features = layer_module(features)
        return features


class MobileViTSelfAttention(nn.Module):
    # 初始化函数，接受配置和隐藏层大小参数
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        # 调用父类初始化函数
        super().__init__()

        # 检查隐藏层大小是否是注意力头的整数倍
        if hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义用于查询的线性层
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        # 定义用于键的线性层
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        # 定义用于值的线性层
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入 tensor 调整为注意力分数矩阵的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 创建新的 tensor 形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重新调整 tensor 形状
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 计算混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 计算键层
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 计算值层
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 计算查询层
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始注意力分数，通过"查询"和"键"的点积得到
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数标准化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 Dropout 层
        attention_probs = self.dropout(attention_probs)

        # 计算上下文层，通过注意力概率加权值层
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
# 定义 MobileViTSelfOutput 类，继承自 nn.Module
class MobileViTSelfOutput(nn.Module):
    # 初始化方法
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，用于处理隐藏状态的维度
        self.dense = nn.Linear(hidden_size, hidden_size)
        # 创建一个 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 通过 Dropout 层处理隐藏状态
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states

# 定义 MobileViTAttention 类，继承自 nn.Module
class MobileViTAttention(nn.Module):
    # 初始化方法
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个 MobileViTSelfAttention 对象
        self.attention = MobileViTSelfAttention(config, hidden_size)
        # 创建一个 MobileViTSelfOutput 对象
        self.output = MobileViTSelfOutput(config, hidden_size)
        # 初始化要剪枝的头部集合
        self.pruned_heads = set()

    # 剪枝操作
    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到可剪枝头部的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对隐藏状态进行自注意力计算
        self_outputs = self.attention(hidden_states)
        # 对自注意力输出进行处理
        attention_output = self.output(self_outputs)
        # 返回处理后的注意力输出
        return attention_output

# 定义 MobileViTIntermediate 类，继承自 nn.Module
class MobileViTIntermediate(nn.Module):
    # 初始化方法
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，用于将隐藏状态转换为中间状态
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # 判断隐藏激活函数是字符串还是函数，并赋值给 intermediate_act_fn
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 通过中间激活函数处理中间状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的中间状态
        return hidden_states

# 定义 MobileViTOutput 类，继承自 nn.Module
class MobileViTOutput(nn.Module):
    # 初始化方法
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，用于将中间状态转换为隐藏状态
        self.dense = nn.Linear(intermediate_size, hidden_size)
        # 创建一个 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 前向传播函数，用于处理模型中的一层
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层，进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行 Dropout 操作，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 将线性变换后的隐藏状态与输入张量相加，实现残差连接
        hidden_states = hidden_states + input_tensor
        # 返回处理后的隐藏状态张量
        return hidden_states
# 定义一个 MobileViTTransformerLayer 类，继承自 PyTorch 的 nn.Module，用于构建 MobileViT 模型中的一个 Transformer 层
class MobileViTTransformerLayer(nn.Module):
    # 初始化 MobileViTTransformerLayer 类
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建 MobileViTAttention 对象，用于注意力机制
        self.attention = MobileViTAttention(config, hidden_size)
        # 创建 MobileViTIntermediate 对象，用于中间层
        self.intermediate = MobileViTIntermediate(config, hidden_size, intermediate_size)
        # 创建 MobileViTOutput 对象，用于输出层
        self.output = MobileViTOutput(config, hidden_size, intermediate_size)
        # 定义层归一化层，用于正则化和稳定训练
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    # 定义前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入进行归一化，然后通过注意力机制
        attention_output = self.attention(self.layernorm_before(hidden_states))
        # 将注意力机制的输出与原输入相加
        hidden_states = attention_output + hidden_states

        # 再次归一化，并通过中间层
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        # 通过输出层，返回最终输出
        layer_output = self.output(layer_output, hidden_states)
        return layer_output


# 定义一个 MobileViTTransformer 类，继承自 nn.Module，用于构建 MobileViT 模型中的 Transformer 结构
class MobileViTTransformer(nn.Module):
    # 初始化 MobileViTTransformer 类
    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int) -> None:
        # 调用父类的初始化方法
        super().__init__()

        # 创建一个 ModuleList，用于存储多个 Transformer 层
        self.layer = nn.ModuleList()
        # 循环创建指定数量的 Transformer 层，并添加到 ModuleList
        for _ in range(num_stages):
            transformer_layer = MobileViTTransformerLayer(
                config,
                hidden_size=hidden_size,
                intermediate_size=int(hidden_size * config.mlp_ratio),
            )
            self.layer.append(transformer_layer)

    # 定义前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 遍历所有 Transformer 层，并依次应用到输入的隐含状态上
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        # 返回最后的隐含状态
        return hidden_states


# 定义一个 MobileViTLayer 类，继承自 nn.Module，描述 MobileViT 模型的一个块
class MobileViTLayer(nn.Module):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    # 初始化 MobileViTLayer 类
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = 1,
    # 初始化函数，设置默认参数
    def __init__(self, config: Config, in_channels: int, out_channels: int, stride: int, num_stages: int, hidden_size: int, dilation: int = 1) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 设置补丁的宽度和高度
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size

        # 如果步幅为2，则创建下采样层，并更新输入通道数
        if stride == 2:
            self.downsampling_layer = MobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
            )
            in_channels = out_channels
        else:
            # 如果步幅不为2，则不创建下采样层
            self.downsampling_layer = None

        # 创建尺寸为kxk的MobileViT卷积层
        self.conv_kxk = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
        )

        # 创建尺寸为1x1的MobileViT卷积层，用于变换输入特征映射到隐藏大小的维度
        self.conv_1x1 = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        # 创建MobileViT的Transformer层
        self.transformer = MobileViTTransformer(
            config,
            hidden_size=hidden_size,
            num_stages=num_stages,
        )

        # 创建Layer Norm层，用于在每个样本上将特征标准化
        self.layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        # 创建尺寸为1x1的MobileViT卷积层，用于将隐藏大小的维度变换为输入通道数的维度
        self.conv_projection = MobileViTConvLayer(
            config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1
        )

        # 创建融合层，将输入通道数扩展为两倍，并输出与输入通道数相同的维度
        self.fusion = MobileViTConvLayer(
            config, in_channels=2 * in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size
        )
    def unfolding(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # 获取patch的宽度和高度
        patch_width, patch_height = self.patch_width, self.patch_height
        # 计算patch的面积
        patch_area = int(patch_width * patch_height)

        # 获取features的维度信息
        batch_size, channels, orig_height, orig_width = features.shape

        # 计算未折叠时的新高度和宽度
        new_height = int(math.ceil(orig_height / patch_height) * patch_height)
        new_width = int(math.ceil(orig_width / patch_width) * patch_width)

        interpolate = False
        # 如果新宽度或者新高度不等于原始宽度或高度，则进行插值
        if new_width != orig_width or new_height != orig_height:
            # 注意: 可以进行填充，但需要在注意力函数中处理
            features = nn.functional.interpolate(
                features, size=(new_height, new_width), mode="bilinear", align_corners=False
            )
            interpolate = True

        # 计算宽度和高度方向的patch数量
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # 将features的形状从(batch_size, channels, orig_height, orig_width)变为(batch_size * patch_area, num_patches, channels)
        patches = features.reshape(
            batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width
        )
        patches = patches.transpose(1, 2)
        patches = patches.reshape(batch_size, channels, num_patches, patch_area)
        patches = patches.transpose(1, 3)
        patches = patches.reshape(batch_size * patch_area, num_patches, -1)

        # 创建信息字典
        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        return patches, info_dict
    # 对输入的图像特征进行折叠操作，将图像分成小块并进行处理
    def folding(self, patches: torch.Tensor, info_dict: Dict) -> torch.Tensor:
        # 获取单个块的宽度和高度
        patch_width, patch_height = self.patch_width, self.patch_height
        # 计算块的面积
        patch_area = int(patch_width * patch_height)

        # 从 info_dict 中提取批处理大小、通道数、块的数量、高度和宽度
        batch_size = info_dict["batch_size"]
        channels = info_dict["channels"]
        num_patches = info_dict["num_patches"]
        num_patch_height = info_dict["num_patches_height"]
        num_patch_width = info_dict["num_patches_width"]

        # 重新排列张量的形状，将它们转换为合适的形状
        features = patches.contiguous().view(batch_size, patch_area, num_patches, -1)
        features = features.transpose(1, 3) # 转置张量
        features = features.reshape(
            batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width
        )
        features = features.transpose(1, 2)
        features = features.reshape(
            batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width
        )

        # 如果需要插值，则使用双线性插值将特征图调整到原始尺寸
        if info_dict["interpolate"]:
            features = nn.functional.interpolate(
                features, size=info_dict["orig_size"], mode="bilinear", align_corners=False
            )

        return features

    # 向前传播操作，对输入特征进行处理后返回
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 如果存在下采样层，对特征进行空间维度的降维操作
        if self.downsampling_layer:
            features = self.downsampling_layer(features)

        # 备份输入特征
        residual = features

        # 对输入特征执行局部表示操作
        features = self.conv_kxk(features) # 卷积操作
        features = self.conv_1x1(features) # 1x1 卷积操作

        # 将特征图转换为块
        patches, info_dict = self.unfolding(features)

        # 学习全局表示
        patches = self.transformer(patches) # 使用变换器处理块
        patches = self.layernorm(patches) # 对块进行层归一化操作

        # 将块转换回特征图
        features = self.folding(patches, info_dict)

        # 对特征图进行投影操作
        features = self.conv_projection(features) # 卷积投影
        features = self.fusion(torch.cat((residual, features), dim=1)) # 将输入特征和处理后的特征进行融合操作
        return features
class MobileViTEncoder(nn.Module):
    # MobileViTEncoder 类的构造函数
    def __init__(self, config: MobileViTConfig) -> None:
        # 调用父类构造函数
        super().__init__()
        # 保存 MobileViT 模型配置
        self.config = config

        # 初始化一个空的模块列表
        self.layer = nn.ModuleList()
        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

        # 根据输出步幅配置来设置是否需要对特定层进行膨胀卷积
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True

        # 初始化膨胀系数
        dilation = 1

        # 添加 MobileViTMobileNetLayer 层
        layer_1 = MobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[0],
            out_channels=config.neck_hidden_sizes[1],
            stride=1,
            num_stages=1,
        )
        self.layer.append(layer_1)

        # 添加 MobileViTMobileNetLayer 层
        layer_2 = MobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[1],
            out_channels=config.neck_hidden_sizes[2],
            stride=2,
            num_stages=3,
        )
        self.layer.append(layer_2)

        # 添加 MobileViTLayer 层
        layer_3 = MobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[2],
            out_channels=config.neck_hidden_sizes[3],
            stride=2,
            hidden_size=config.hidden_sizes[0],
            num_stages=2,
        )
        self.layer.append(layer_3)

        # 如果需要对第四层进行膨胀卷积，则更新膨胀系数
        if dilate_layer_4:
            dilation *= 2

        # 添加 MobileViTLayer 层
        layer_4 = MobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[3],
            out_channels=config.neck_hidden_sizes[4],
            stride=2,
            hidden_size=config.hidden_sizes[1],
            num_stages=4,
            dilation=dilation,
        )
        self.layer.append(layer_4)

        # 如果需要对第五层进行膨胀卷积，则更新膨胀系数
        if dilate_layer_5:
            dilation *= 2

        # 添加 MobileViTLayer 层
        layer_5 = MobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[4],
            out_channels=config.neck_hidden_sizes[5],
            stride=2,
            hidden_size=config.hidden_sizes[2],
            num_stages=3,
            dilation=dilation,
        )
        self.layer.append(layer_5)

    # MobileViTEncoder 类的前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 定义一个函数，接受一个可能为 tuple 或 BaseModelOutputWithNoAttention 类型的返回值
        ) -> Union[tuple, BaseModelOutputWithNoAttention]:
            # 如果 output_hidden_states 为 False，则初始化一个空的 all_hidden_states 元组
            all_hidden_states = () if output_hidden_states else None
    
            # 遍历 self.layer 中的每个层模块
            for i, layer_module in enumerate(self.layer):
                # 如果启用了梯度检查点功能并且处于训练模式
                if self.gradient_checkpointing and self.training:
                    # 使用 _gradient_checkpointing_func 函数计算当前层的隐藏状态
                    hidden_states = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                    )
                # 否则正常计算当前层的隐藏状态
                else:
                    hidden_states = layer_module(hidden_states)
    
                # 如果需要输出所有隐藏状态
                if output_hidden_states:
                    # 将当前层的隐藏状态添加到 all_hidden_states 元组中
                    all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 如果不需要返回字典类型
            if not return_dict:
                # 返回一个元组，包含隐藏状态和所有隐藏状态（如果有的话）
                return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
    
            # 否则返回 BaseModelOutputWithNoAttention 类型的对象，包含最终隐藏状态和所有隐藏状态
            return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)
# 定义了一个 MobileViTPreTrainedModel 类，用于处理权重初始化、下载和加载预训练模型
class MobileViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 MobileViTConfig
    config_class = MobileViTConfig
    # 设置基础模型前缀为 "mobilevit"
    base_model_prefix = "mobilevit"
    # 设置主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 设置支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重的方法
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 初始化线性层或卷积层的权重，使用正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 层，将偏置初始化为零，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 添加关于 MobileViTModel 的文档字符串
MOBILEVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 添加有关 MobileViTModel 输入的文档字符串
MOBILEVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileViTImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MobileViT model outputting raw hidden-states without any specific head on top.",
    MOBILEVIT_START_DOCSTRING,
)
# 定义 MobileViTModel 类，继承自 MobileViTPreTrainedModel 类
class MobileViTModel(MobileViTPreTrainedModel):
    # 初始化 MobileViTModel 类
    def __init__(self, config: MobileViTConfig, expand_output: bool = True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置信息
        self.config = config
        # 是否扩展输出
        self.expand_output = expand_output
    
        # 创建一个卷积层作为输入层
        self.conv_stem = MobileViTConvLayer(
            config,
            in_channels=config.num_channels, # 输入通道数
            out_channels=config.neck_hidden_sizes[0], # 输出通道数
            kernel_size=3, # 卷积核大小
            stride=2, # 步长
        )
    
        # 创建 MobileViTEncoder 对象
        self.encoder = MobileViTEncoder(config)
    
        # 如果需要扩展输出
        if self.expand_output:
            # 创建一个 1x1 卷积层
            self.conv_1x1_exp = MobileViTConvLayer(
                config,
                in_channels=config.neck_hidden_sizes[5], # 输入通道数
                out_channels=config.neck_hidden_sizes[6], # 输出通道数
                kernel_size=1, # 卷积核大小
            )
    
        # 初始化权重并应用最终处理
        self.post_init()
    
    # 裁剪模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        # 遍历需要裁剪的头
        for layer_index, heads in heads_to_prune.items():
            # 获取对应的 MobileViTLayer 层
            mobilevit_layer = self.encoder.layer[layer_index]
            # 如果是 MobileViTLayer 层
            if isinstance(mobilevit_layer, MobileViTLayer):
                # 遍历层内部的变换层
                for transformer_layer in mobilevit_layer.transformer.layer:
                    # 裁剪注意力头
                    transformer_layer.attention.prune_heads(heads)
    
    # 前向传播
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播逻辑
        pass
    # 函数返回类型注解，可以返回元组或者带有池化和无注意力的基础模型输出
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 如果没有传入输出隐藏状态参数，则使用配置文件中的输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有传入返回字典参数，则使用配置文件中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未传入像素值，则引发未指定像素值的错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用卷积干扰转换像素值
        embedding_output = self.conv_stem(pixel_values)

        # 使用编码器处理转换后的嵌入输出
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果需要扩展输出
        if self.expand_output:
            # 使用1x1卷积扩展最后一个隐藏状态
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])

            # 全局平均池化：(batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = torch.mean(last_hidden_state, dim=[-2, -1], keepdim=False)
        else:
            last_hidden_state = encoder_outputs[0]
            pooled_output = None

        # 如果不需要返回字典
        if not return_dict:
            # 如果有池化输出，则组成元组，否则只返回最后一个隐藏状态
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
            # 返回隐藏状态和其余编码器输出
            return output + encoder_outputs[1:]

        # 返回带有池化和无注意力的基础模型输出
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 为图像分类任务设计的 MobileViT 模型，采用图像分类头部（在池化特征之上的线性层），例如用于 ImageNet 数据集
@add_start_docstrings(
    """
    MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILEVIT_START_DOCSTRING,
)
class MobileViTForImageClassification(MobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__(config)

        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 MobileViT 模型
        self.mobilevit = MobileViTModel(config)

        # 分类器头部
        # 设置丢弃率
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)
        # 如果标签数量大于 0，则创建线性分类器；否则创建一个恒等映射
        self.classifier = (
            nn.Linear(config.neck_hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 检查是否需要返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 运行 MobileViT，获取模型输出
        outputs = self.mobilevit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 根据需要将输出的 pooler 输出或索引 1 的输出作为 pooled_output
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将 pooled_output 输入分类器，并获取 logits
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        # 如果存在 labels
        if labels is not None:
            # 如果问题类型未定义
            if self.config.problem_type is None:
                # 根据 num_labels 的值确定问题类型
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失函数
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

        # 如果不需要返回字典
        if not return_dict:
            # 组装输出元组
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 ImageClassifierOutputWithNoAttention 对象
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
```  
class MobileViTASPPPooling(nn.Module):
    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # 创建一个自适应平均池化层，输出大小为1
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 创建一个1x1卷积层
        self.conv_1x1 = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 获取输入特征的空间维度
        spatial_size = features.shape[-2:]
        # 对输入特征进行全局平均池化
        features = self.global_pool(features)
        # 通过1x1卷积层处理特征
        features = self.conv_1x1(features)
        # 使用双线性插值对特征进行上采样，恢复空间尺寸
        features = nn.functional.interpolate(features, size=spatial_size, mode="bilinear", align_corners=False)
        return features


class MobileViTASPP(nn.Module):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()

        # 获取输入通道数和ASPP的输出通道数
        in_channels = config.neck_hidden_sizes[-2]
        out_channels = config.aspp_out_channels

        # 如果空洞卷积率不为3，则抛出数值错误
        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        # 创建一系列卷积层
        self.convs = nn.ModuleList()

        # 创建一个输入投影卷积层，输出通道数和ASPP输出通道数相等
        in_projection = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
        )
        self.convs.append(in_projection)

        # 创建多个卷积层，每个卷积层的空洞率从空洞卷积率列表中获取
        self.convs.extend(
            [
                MobileViTConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=rate,
                    use_activation="relu",
                )
                for rate in config.atrous_rates
            ]
        )

        # 创建ASPP池化层
        pool_layer = MobileViTASPPPooling(config, in_channels, out_channels)
        self.convs.append(pool_layer)

        # 创建一个投影卷积层，输入通道数为5倍的ASPP输出通道数，输出通道数与ASPP输出通道数相等
        self.project = MobileViTConvLayer(
            config, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, use_activation="relu"
        )

        # 创建一个Dropout层，按指定概率随机置零输入张量的元素
        self.dropout = nn.Dropout(p=config.aspp_dropout_prob)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 创建空列表存储金字塔特征
        pyramid = []
        # 遍历所有卷积层，将处理后的特征存储到金字塔列表中
        for conv in self.convs:
            pyramid.append(conv(features))
        # 将金字塔列表中的特征在通道维度上拼接
        pyramid = torch.cat(pyramid, dim=1)

        # 通过投影卷积层处理金字塔特征
        pooled_features = self.project(pyramid)
        # 对处理后的特征进行Dropout
        pooled_features = self.dropout(pooled_features)
        return pooled_features


class MobileViTDeepLabV3(nn.Module):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """
    # 初始化方法，接收 MobileViTConfig 类型的参数
    def __init__(self, config: MobileViTConfig) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建 MobileViTASPP 对象，并将其赋值给实例属性 aspp
        self.aspp = MobileViTASPP(config)

        # 创建一个二维的 Dropout 层，丢弃概率为 config.classifier_dropout_prob
        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)

        # 创建一个 MobileViTConvLayer，用于分类，设定其输入和输出通道数、卷积核大小等参数
        self.classifier = MobileViTConvLayer(
            config,
            in_channels=config.aspp_out_channels,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
        )

    # 前向传播方法，接收 torch.Tensor 类型的 hidden_states 参数，并返回 torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用 aspp 对隐藏状态的最后一个时间步的特征进行处理
        features = self.aspp(hidden_states[-1])
        # 对特征进行 dropout 处理
        features = self.dropout(features)
        # 使用 classifier 对特征进行分类
        features = self.classifier(features)
        # 返回分类后的特征
        return features
# 添加模型文档字符串，说明该模型是在 MobileViT 模型基础上增加了语义分割头部的模型，例如用于 Pascal VOC 数据集
@add_start_docstrings(
    """
    MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILEVIT_START_DOCSTRING,
)
# 定义一个 MobileViT 用于语义分割的模型类，继承自 MobileViTPreTrainedModel 类
class MobileViTForSemanticSegmentation(MobileViTPreTrainedModel):
    # 初始化函数
    def __init__(self, config: MobileViTConfig) -> None:
        # 调用父类的初始化函数
        super().__init__(config)

        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 创建一个 MobileViT 模型，参数 expand_output 设置为 False
        self.mobilevit = MobileViTModel(config, expand_output=False)
        # 创建一个 MobileViTDeepLabV3 的语义分割头部
        self.segmentation_head = MobileViTDeepLabV3(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播函数
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional`):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:
        
        Examples:

        ```py
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
        >>> model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否需要输出隐藏状态，如果为None，则使用配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 检查是否需要返回字典，如果为None，则使用配置中的设定

        outputs = self.mobilevit(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        # 在 MobileViT 模型上执行前向传播操作，返回中间表示的隐藏状态

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # 根据返回字典标志来选择处理隐藏状态的方式

        logits = self.segmentation_head(encoder_hidden_states)
        # 使用编码器隐藏状态进行语义分割操作，生成分类结果

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
                # 如果存在标签，使用交叉熵损失计算损失值

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            # 如果不需要返回字典格式的结果，则根据需求选择返回隐藏状态或输出结果

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
        # 返回语义分割器的输出结果，包括损失值、分类结果、隐藏状态及注意力结果
```