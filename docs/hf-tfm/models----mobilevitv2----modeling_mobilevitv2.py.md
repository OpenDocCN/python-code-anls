# `.\transformers\models\mobilevitv2\modeling_mobilevitv2.py`

```py
# coding=utf-8  # 指定编码格式为utf-8

# 此文件是 MobileViTV2 模型的实现，版权归 Apple Inc. 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache License, Version 2.0 (the "License") 授权进行使用
# 可从下面链接获得许可证：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非依法需要或者与相关法规共同约定，本软件根据本 "AS IS" 条款发布，不提供任何担保或条件
# 参阅许可证以获得更多信息

# 其他头文件和函数的导入语句

# 日志模块的导入
import logging

# PyTorch 模块的导入
import torch
from torch import nn

# transformers 模块的导入
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mobilevitv2 import MobileViTV2Config

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 下面是一些全局变量的文档字符串
_CONFIG_FOR_DOC = "MobileViTV2Config"
_CHECKPOINT_FOR_DOC = "apple/mobilevitv2-1.0-imagenet1k-256"
_EXPECTED_OUTPUT_SHAPE = [1, 512, 8, 8]
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevitv2-1.0-imagenet1k-256"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# MobileViTV2 模型的预训练模型列表
MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "apple/mobilevitv2-1.0-imagenet1k-256"
]

# 从 "transformers.models.mobilevit.modeling_mobilevit.make_divisible" 复制过来的函数
# 用于将输入值变成能被另一个指定数整除的最接近的数
# 其中的参数为 value：输入值；divisor：指定的数；min_value：最小值
def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    确保所有的层的通道数都是可以被 `divisor` 整除的。此函数是源于原始 TensorFlow 代码库。
    此函数的原代码可见于以下链接：
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # 确保圆整到指定的数后的结果不比原来的数小超过 10%
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

# 将数值限制在指定的范围内，以防止超出范围
def clip(value: float, min_val: float = float("-inf"), max_val: float = float("inf")) -> float:
    return max(min_val, min(max_val, value))

# 从 "transformers.models.mobilevit.modeling_mobilevit.MobileViTConvLayer" 复制过来的类
# 修改类名 from MobileViT->MobileViTV2
class MobileViTV2ConvLayer(nn.Module):


此代码块是 MobileViTV2 模型的实现，其中包含了依赖库导入、全局变量的定义，以及两个辅助函数的实现。
    # 定义一个类，用于创建MobileViTV2Config模型的卷积层
    def __init__(
        self,
        config: MobileViTV2Config,  # MobileViTV2Config类型的配置参数
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        kernel_size: int,  # 卷积核大小
        stride: int = 1,  # 步长，默认为1
        groups: int = 1,  # 分组卷积的组数，默认为1
        bias: bool = False,  # 是否使用偏置，默认为False
        dilation: int = 1,  # 膨胀率，默认为1
        use_normalization: bool = True,  # 是否使用归一化，默认为True
        use_activation: Union[bool, str] = True,  # 是否使用激活函数，默认为True，或者指定激活函数的名称
    ) -> None:  # 返回值为空
        # 调用父类的构造函数
        super().__init__()
        # 根据卷积核大小、膨胀率计算填充大小
        padding = int((kernel_size - 1) / 2) * dilation
    
        # 如果输入通道数不能被分组数整除，抛出数值错误异常
        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        # 如果输出通道数不能被分组数整除，抛出数值错误异常
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")
    
        # 创建卷积层对象
        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )
    
        # 如果使用归一化
        if use_normalization:
            # 创建批归一化层对象
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        else:
            # 否则，归一化层为空
            self.normalization = None
    
        # 如果使用激活函数
        if use_activation:
            # 如果指定了激活函数名称
            if isinstance(use_activation, str):
                # 根据激活函数名称从预定义字典中获取对应的激活函数对象
                self.activation = ACT2FN[use_activation]
            # 否则，如果配置参数中指定了隐藏层激活函数名称
            elif isinstance(config.hidden_act, str):
                # 根据配置参数中的隐藏层激活函数名称从预定义字典中获取对应的激活函数对象
                self.activation = ACT2FN[config.hidden_act]
            # 否则，使用配置参数中指定的隐藏层激活函数对象
            else:
                self.activation = config.hidden_act
        else:
            # 否则，激活函数为空
            self.activation = None
    
    # 定义前向传播方法
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 对输入特征进行卷积操作
        features = self.convolution(features)
        # 如果使用归一化层
        if self.normalization is not None:
            # 对卷积后的特征进行归一化
            features = self.normalization(features)
        # 如果使用激活函数
        if self.activation is not None:
            # 对归一化后的特征进行激活函数处理
            features = self.activation(features)
        # 返回处理后的特征
        return features
# 使用 MobileViT->MobileViTV2 的模型参数构建 MobileViTV2InvertedResidual 类
class MobileViTV2InvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self, config: MobileViTV2Config, in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        super().__init__()
        # 根据配置计算扩展后的通道数
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        # 检查是否合法的步幅值
        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        # 确定是否使用残差连接
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        # 1x1 卷积层，扩展通道数
        self.expand_1x1 = MobileViTV2ConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        # 3x3 卷积层
        self.conv_3x3 = MobileViTV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

        # 1x1 卷积层，减少通道数
        self.reduce_1x1 = MobileViTV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features

        # 依次进行卷积操作
        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        # 如果需要残差连接则进行残差相加操作，否则直接返回特征值
        return residual + features if self.use_residual else features


# 使用 MobileViT->MobileViTV2 的模型参数构建 MobileViTV2MobileNetLayer 类
class MobileViTV2MobileNetLayer(nn.Module):
    def __init__(
        self, config: MobileViTV2Config, in_channels: int, out_channels: int, stride: int = 1, num_stages: int = 1
    ) -> None:
        super().__init__()

        # 保存 MobileViTV2InvertedResidual 实例的 module 列表
        self.layer = nn.ModuleList()
        for i in range(num_stages):
            # 创建 MobileViTV2InvertedResidual 实例并添加到列表中
            layer = MobileViTV2InvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
            )
            self.layer.append(layer)
            in_channels = out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 依次通过 MobileViTV2InvertedResidual 实例进行前向传播
        for layer_module in self.layer:
            features = layer_module(features)
        return features


# MobileViTV2LinearSelfAttention 类用于应用具有线性复杂度的自注意力
class MobileViTV2LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in MobileViTV2 paper:
    https://arxiv.org/abs/2206.02680

    Args:
        config (`MobileVitv2Config`):
             Model configuration object
        embed_dim (`int`):
            `input_channels` from an expected input of size :math:`(batch_size, input_channels, height, width)`
    """
``` 
    # 初始化函数，接受 MobileViTV2Config 实例和维度参数 embed_dim，无返回值
    def __init__(self, config: MobileViTV2Config, embed_dim: int) -> None:
        # 调用父类的初始化函数
        super().__init__()

        # 创建 MobileViTV2ConvLayer 实例用于从输入向量投影为 query、key 和 value 的向量
        self.qkv_proj = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=True,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        # 定义一个 Dropout 层，概率为 config 中的 attn_dropout 参数
        self.attn_dropout = nn.Dropout(p=config.attn_dropout)

        # 创建 MobileViTV2ConvLayer 实例用于处理输出向量
        self.out_proj = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=True,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )
        
        # 存储输入的维度参数为实例的 embed_dim 属性
        self.embed_dim = embed_dim

    # 前向传播函数，接受输入张量 hidden_states，返回输出张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用 qkv_proj 投影输入张量得到 qkv
        qkv = self.qkv_proj(hidden_states)

        # 在第二个维度上拆分 qkv，得到 query、key 和 value
        query, key, value = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)

        # 对 query 进行 softmax 操作，压缩 num_patches 维度
        context_scores = torch.nn.functional.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # 计算上下文向量
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # 将上下文向量与 value 结合
        out = torch.nn.functional.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out
class MobileViTV2FFN(nn.Module):
    def __init__(
        self,
        config: MobileViTV2Config,  # 初始化函数，接受 MobileViTV2Config 类型的参数 config
        embed_dim: int,  # 接受整数类型的参数 embed_dim
        ffn_latent_dim: int,  # 接受整数类型的参数 ffn_latent_dim
        ffn_dropout: float = 0.0,  # 设置默认值为 0.0 的浮点类型参数 ffn_dropout
    ) -> None:  # 指定返回类型为 None
        super().__init__()  # 调用父类的初始化函数
        self.conv1 = MobileViTV2ConvLayer(  # 初始化类变量 self.conv1，调用 MobileViTV2ConvLayer 类
            config=config,  # 传入 config 参数
            in_channels=embed_dim,  # 传入 embed_dim 参数作为 in_channels
            out_channels=ffn_latent_dim,  # 传入 ffn_latent_dim 参数作为 out_channels
            kernel_size=1,  # 设置 kernel_size 为 1
            stride=1,  # 设置 stride 为 1
            bias=True,  # 启用偏置
            use_normalization=False,  # 禁用归一化
            use_activation=True,  # 使用激活函数
        )
        self.dropout1 = nn.Dropout(ffn_dropout)  # 初始化类变量 self.dropout1，使用 nn.Dropout 类创建对象，传入 ffn_dropout 参数

        self.conv2 = MobileViTV2ConvLayer(  # 初始化类变量 self.conv2，调用 MobileViTV2ConvLayer 类
            config=config,  # 传入 config 参数
            in_channels=ffn_latent_dim,  # 传入 ffn_latent_dim 参数作为 in_channels
            out_channels=embed_dim,  # 传入 embed_dim 参数作为 out_channels
            kernel_size=1,  # 设置 kernel_size 为 1
            stride=1,  # 设置 stride 为 1
            bias=True,  # 启用偏置
            use_normalization=False,  # 禁用归一化
            use_activation=False,  # 禁用激活函数
        )
        self.dropout2 = nn.Dropout(ffn_dropout)  # 初始化类变量 self.dropout2，使用 nn.Dropout 类创建对象，传入 ffn_dropout 参数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播函数，接受 torch.Tensor 类型的参数 hidden_states，返回 torch.Tensor 类型
        hidden_states = self.conv1(hidden_states)  # 使用 self.conv1 执行前向传播操作
        hidden_states = self.dropout1(hidden_states)  # 使用 self.dropout1 执行前向传播操作
        hidden_states = self.conv2(hidden_states)  # 使用 self.conv2 执行前向传播操作
        hidden_states = self.dropout2(hidden_states)  # 使用 self.dropout2 执行前向传播操作
        return hidden_states  # 返回处理后的 hidden_states


class MobileViTV2TransformerLayer(nn.Module):
    def __init__(
        self,
        config: MobileViTV2Config,  # 初始化函数，接受 MobileViTV2Config 类型的参数 config
        embed_dim: int,  # 接受整数类型的参数 embed_dim
        ffn_latent_dim: int,  # 接受整数类型的参数 ffn_latent_dim
        dropout: float = 0.0,  # 设置默认值为 0.0 的浮点类型参数 dropout
    ) -> None:  # 指定返回类型为 None
        super().__init__()  # 调用父类的初始化函数
        self.layernorm_before = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=config.layer_norm_eps)  # 初始化类变量 self.layernorm_before，使用 nn.GroupNorm 类创建对象
        self.attention = MobileViTV2LinearSelfAttention(config, embed_dim)  # 初始化类变量 self.attention，调用 MobileViTV2LinearSelfAttention 类
        self.dropout1 = nn.Dropout(p=dropout)  # 初始化类变量 self.dropout1，使用 nn.Dropout 类创建对象

        self.layernorm_after = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=config.layer_norm_eps)  # 初始化类变量 self.layernorm_after，使用 nn.GroupNorm 类创建对象
        self.ffn = MobileViTV2FFN(config, embed_dim, ffn_latent_dim, config.ffn_dropout)  # 初始化类变量 self.ffn，调用 MobileViTV2FFN 类

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播函数，接受 torch.Tensor 类型的参数 hidden_states，返回 torch.Tensor 类型
        layernorm_1_out = self.layernorm_before(hidden_states)  # 使用 self.layernorm_before 执行前向传播操作
        attention_output = self.attention(layernorm_1_out)  # 使用 self.attention 执行前向传播操作
        hidden_states = attention_output + hidden_states  # 与原 hidden_states 相加

        layer_output = self.layernorm_after(hidden_states)  # 使用 self.layernorm_after 执行前向传播操作
        layer_output = self.ffn(layer_output)  # 使用 self.ffn 执行前向传播操作

        layer_output = layer_output + hidden_states  # 与原 hidden_states 相加
        return layer_output  # 返回处理后的 layer_output


class MobileViTV2Transformer(nn.Module):
    def __init__(self, config: MobileViTV2Config, n_layers: int, d_model: int) -> None:  # 初始化函数，接受 MobileViTV2Config 类型的参数 config，整型参数 n_layers 和 d_model
        super().__init__()  # 调用父类的初始化函数

        ffn_multiplier = config.ffn_multiplier  # 获取 config 对象的 ffn_multiplier 属性值

        ffn_dims = [ffn_multiplier * d_model] * n_layers  # 构建具有 n_layers 个元素的列表，每个元素为 ffn_multiplier 乘以 d_model

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]  # 将列表中每个元素取整以确保是 16 的倍数

        self.layer = nn.ModuleList()  # 初始化类变量 self.layer 为 nn.ModuleList 类对象
        for block_idx in range(n_layers):  # 遍历 n_layers 次
            transformer_layer = MobileViTV2TransformerLayer(  # 初始化 transformer_layer，调用 MobileViTV2TransformerLayer 类
                config, embed_dim=d_model, ffn_latent_dim=ffn_dims[block_idx]  # 传入 config，d_model 和 ffn_dims[block_idx] 作为参数
            )
            self.layer.append(transformer_layer)  # 将 transformer_layer 添加到 self.layer 列表中
    # 定义一个向前传播函数，接收隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 遍历神经网络的每一层模块
        for layer_module in self.layer:
            # 将隐藏状态传递给当前层模块进行处理，更新隐藏状态
            hidden_states = layer_module(hidden_states)
        # 返回最终处理后的隐藏状态作为输出
        return hidden_states
# MobileViTV2 layer 类
class MobileViTV2Layer(nn.Module):
    """
    # MobileViTV2 layer 的实现: https://arxiv.org/abs/2206.02680
    """

    # 初始化方法
    def __init__(
        self,
        config: MobileViTV2Config, # 配置对象
        in_channels: int, # 输入通道数
        out_channels: int, # 输出通道数
        attn_unit_dim: int, # 注意力单元维度
        n_attn_blocks: int = 2, # 注意力块数量
        dilation: int = 1, # 膨胀因子
        stride: int = 2, # 步长
    ) -> None:
        super().__init__()
        self.patch_width = config.patch_size # patch 宽度
        self.patch_height = config.patch_size # patch 高度

        cnn_out_dim = attn_unit_dim # CNN 输出维度

        # 如果步长为 2，执行下采样
        if stride == 2:
            self.downsampling_layer = MobileViTV2InvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
            )
            in_channels = out_channels
        else:
            self.downsampling_layer = None

        # 局部表示
        self.conv_kxk = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            groups=in_channels,
        )
        self.conv_1x1 = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        # 全局表示
        self.transformer = MobileViTV2Transformer(config, d_model=attn_unit_dim, n_layers=n_attn_blocks)

        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=attn_unit_dim, eps=config.layer_norm_eps)

        # 融合
        self.conv_projection = MobileViTV2ConvLayer(
            config,
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            use_normalization=True,
            use_activation=False,
        )

    # 将特征图划分为 patches
    def unfolding(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_height, img_width = feature_map.shape
        patches = nn.functional.unfold(
            feature_map,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        patches = patches.reshape(batch_size, in_channels, self.patch_height * self.patch_width, -1)

        return patches, (img_height, img_width)
    def folding(self, patches: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        # 获取输入张量的维度信息
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # 重塑张量的形状，使之合适于折叠操作
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        # 使用折叠函数将patches转换为特征图
        feature_map = nn.functional.fold(
            patches,
            output_size=output_size,  # 输出特征图的大小
            kernel_size=(self.patch_height, self.patch_width),  # 卷积核大小
            stride=(self.patch_height, self.patch_width),  # 步长
        )

        return feature_map

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 如果需要降低空间维度
        if self.downsampling_layer:
            # 使用下采样层降低空间维度
            features = self.downsampling_layer(features)

        # 将特征图转换为局部表示
        features = self.conv_kxk(features)
        features = self.conv_1x1(features)

        # 将特征图转换为patches
        patches, output_size = self.unfolding(features)

        # 学习全局表示
        patches = self.transformer(patches)
        patches = self.layernorm(patches)

        # 将patches转换回特征图
        # [batch_size, patch_height, patch_width, input_dim] --> [batch_size, input_dim, patch_height, patch_width]
        features = self.folding(patches, output_size)

        # 使用卷积投影操作处理特征图
        features = self.conv_projection(features)
        return features
class MobileViTV2Encoder(nn.Module):
    # MobileViTV2 编码器类，继承自 nn.Module
    def __init__(self, config: MobileViTV2Config) -> None:
        # 初始化函数，接受 MobileViTV2Config 类型的配置参数，并且没有返回值
        super().__init__()
        # 调用父类的初始化函数
        self.config = config
        # 将传入的配置参数保存为对象的属性

        self.layer = nn.ModuleList()
        # 初始化一个空的 nn.ModuleList 对象，用于存储网络层
        self.gradient_checkpointing = False
        # 梯度检查点标志，暂时设为 False

        # segmentation architectures like DeepLab and PSPNet modify the strides
        # of the classification backbones
        # 如果输出步长为 8，则需要对层 4 和层 5 进行扩张操作
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        # 如果输出步长为 16，则需要对层 5 进行扩张操作
        elif config.output_stride == 16:
            dilate_layer_5 = True

        dilation = 1
        # 初始化扩张率为 1

        layer_0_dim = make_divisible(
            clip(value=32 * config.width_multiplier, min_val=16, max_val=64), divisor=8, min_value=16
        )
        # 计算并得到输入层的维度，确保是 8 的倍数，并且最小值为 16

        layer_1_dim = make_divisible(64 * config.width_multiplier, divisor=16)
        # 计算并得到层 1 的维度，确保是 16 的倍数

        layer_2_dim = make_divisible(128 * config.width_multiplier, divisor=8)
        # 计算并得到层 2 的维度，确保是 8 的倍数

        layer_3_dim = make_divisible(256 * config.width_multiplier, divisor=8)
        # 计算并得到层 3 的维度，确保是 8 的倍数

        layer_4_dim = make_divisible(384 * config.width_multiplier, divisor=8)
        # 计算并得到层 4 的维度，确保是 8 的倍数

        layer_5_dim = make_divisible(512 * config.width_multiplier, divisor=8)
        # 计算并得到层 5 的维度，确保是 8 的倍数

        # 创建并添加层 1
        layer_1 = MobileViTV2MobileNetLayer(
            config,
            in_channels=layer_0_dim,
            out_channels=layer_1_dim,
            stride=1,
            num_stages=1,
        )
        self.layer.append(layer_1)

        # 创建并添加层 2
        layer_2 = MobileViTV2MobileNetLayer(
            config,
            in_channels=layer_1_dim,
            out_channels=layer_2_dim,
            stride=2,
            num_stages=2,
        )
        self.layer.append(layer_2)

        # 创建并添加层 3
        layer_3 = MobileViTV2Layer(
            config,
            in_channels=layer_2_dim,
            out_channels=layer_3_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[0] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[0],
        )
        self.layer.append(layer_3)

        # 如果需要扩张层 4，则更新扩张率
        if dilate_layer_4:
            dilation *= 2

        # 创建并添加层 4
        layer_4 = MobileViTV2Layer(
            config,
            in_channels=layer_3_dim,
            out_channels=layer_4_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[1] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[1],
            dilation=dilation,
        )
        self.layer.append(layer_4)

        # 如果需要扩张层 5，则再次更新扩张率
        if dilate_layer_5:
            dilation *= 2

        # 创建并添加层 5
        layer_5 = MobileViTV2Layer(
            config,
            in_channels=layer_4_dim,
            out_channels=layer_5_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[2] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[2],
            dilation=dilation,
        )
        self.layer.append(layer_5)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 定义函数，用于获取模型的隐藏状态。返回值可以是元组或带有无注意力的基本模型输出
    def forward(
        self, 
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为空
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，默认为空
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，默认为空
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力掩码，默认为空
        past_key_value: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对，默认为空
        use_cache: Optional[bool] = None,  # 是否使用缓存，默认为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为空
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为空
    ) -> Union[tuple, BaseModelOutputWithNoAttention]:  # 返回值类型可以是元组或带有无注意力的基本模型输出
        # 如果不输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个层
        for i, layer_module in enumerate(self.layer):
            # 如果启用了梯度检查点且在训练模式下
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数，对当前层进行计算
                hidden_states = self._gradient_checkpointing_func(
                    layer_module.__call__,  # 调用当前层的call方法
                    hidden_states,  # 输入的隐藏状态
                )
            else:
                # 否则，正常调用当前层进行计算
                hidden_states = layer_module(hidden_states)

            # 如果需要输出所有隐藏状态
            if output_hidden_states:
                # 将当前层的隐藏状态添加到所有隐藏状态元组中
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典
        if not return_dict:
            # 返回不为空的元组（隐藏状态和所有隐藏状态）
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 否则，返回带有最终隐藏状态和所有隐藏状态的基本模型输出
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)
# 从 transformers.models.mobilevit.modeling_mobilevit.MobileViTPreTrainedModel 复制而来，将MobileViT->MobileViTV2,mobilevit->mobilevitv2
class MobileViTV2PreTrainedModel(PreTrainedModel):
    """
    用于处理权重初始化和下载/加载预训练模型的抽象类。
    """

    config_class = MobileViTV2Config
    base_model_prefix = "mobilevitv2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与TF版本略有不同，TF版本使用截断正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


MOBILEVITV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileViTV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOBILEVITV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileViTImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MobileViTV2 model outputting raw hidden-states without any specific head on top.",
    MOBILEVITV2_START_DOCSTRING,
)
class MobileViTV2Model(MobileViTV2PreTrainedModel):
    # 构造函数，初始化 MobileViTV2 模型的实例
    def __init__(self, config: MobileViTV2Config, expand_output: bool = True):
        # 调用父类的构造函数，传递 config 参数
        super().__init__(config)
        # 将配置对象保存到实例属性中
        self.config = config
        # 保存是否展开输出的标志到实例属性中
        self.expand_output = expand_output

        # 计算第0层的输出通道数，使其符合可被8整除的要求，最小值为16
        layer_0_dim = make_divisible(
            clip(value=32 * config.width_multiplier, min_val=16, max_val=64), divisor=8, min_value=16
        )

        # 创建卷积干层（stem）层，指定输入和输出通道、卷积核大小、步长等参数
        self.conv_stem = MobileViTV2ConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=layer_0_dim,
            kernel_size=3,
            stride=2,
            use_normalization=True,
            use_activation=True,
        )
        # 创建编码器层，传递配置对象作为参数
        self.encoder = MobileViTV2Encoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 方法，用于修剪模型的头部
    def _prune_heads(self, heads_to_prune):
        """修剪模型的头部。
        heads_to_prune: 字典，{layer_num: list of heads to prune in this layer} 参见基类 PreTrainedModel
        """
        # 遍历需要修剪的头部
        for layer_index, heads in heads_to_prune.items():
            # 获取指定层的 MobileViTV2 层
            mobilevitv2_layer = self.encoder.layer[layer_index]
            # 如果是 MobileViTV2Layer 类型
            if isinstance(mobilevitv2_layer, MobileViTV2Layer):
                # 遍历 transformer 层中的层
                for transformer_layer in mobilevitv2_layer.transformer.layer:
                    # 修剪注意力机制中的头部
                    transformer_layer.attention.prune_heads(heads)

    # 使用装饰器为模型的前向方法添加文档和示例
    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 前向方法，处理输入并返回输出
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 函数定义，接收参数并返回联合类型为元组或 BaseModelOutputWithPoolingAndNoAttention 对象
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        
        # 如果 output_hidden_states 为 None，则使用配置文件中的 output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        # 如果 return_dict 为 None，则使用配置文件中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为 None，则抛出值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用卷积层处理输入像素值生成嵌入向量
        embedding_output = self.conv_stem(pixel_values)

        # 使用编码器处理嵌入向量，获取编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果需要扩展输出
        if self.expand_output:
            last_hidden_state = encoder_outputs[0]

            # 全局平均池化
            # (batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = torch.mean(last_hidden_state, dim=[-2, -1], keepdim=False)
        else:
            last_hidden_state = encoder_outputs[0]
            pooled_output = None

        # 如果不需要返回字典
        if not return_dict:
            # 输出为元组，包括最后的隐藏状态和池化输出（如果存在）
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
            return output + encoder_outputs[1:]

        # 返回 BaseModelOutputWithPoolingAndNoAttention 对象
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 添加对MobileViTV2模型的图像分类头部的文档字符串
# 这里的注释是对模型类的介绍

@add_start_docstrings(
    """
    MobileViTV2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILEVITV2_START_DOCSTRING,
)

# 创建MobileViTV2ForImageClassification类，继承自MobileViTV2PreTrainedModel类
class MobileViTV2ForImageClassification(MobileViTV2PreTrainedModel):
    
    # 初始化方法，设置模型的参数
    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilevitv2 = MobileViTV2Model(config)

        # 计算输出通道数量
        out_channels = make_divisible(512 * config.width_multiplier, divisor=8)  # layer 5 output dimension
        
        # 分类器头部，根据配置项设置线性层或者使用恒等映射
        self.classifier = (
            nn.Linear(in_features=out_channels, out_features=config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接收输入像素值、是否输出隐藏状态、标签和返回字典的可选参数
    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
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
        # 将返回字典设置为传入参数的值，如果没有传入参数，则使用默认的配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 mobilevitv2 模型进行预测，得到输出结果
        outputs = self.mobilevitv2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果需要返回数据字典，则从输出结果中获取池化的输出；否则从输出结果的第二个元素中获取
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器对池化输出进行分类，得到 logits
        logits = self.classifier(pooled_output)

        loss = None
        # 如果传入了 labels
        if labels is not None:
            # 如果问题类型为空
            if self.config.problem_type is None:
                # 如果标签数量为 1，则问题类型为回归
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                # 如果标签数量大于 1 且标签的 dtype 是 torch.long 或 torch.int，则问题类型为单标签分类
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                # 否则问题类型为多标签分类
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失函数并计算损失
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

        # 如果不需要返回数据字典，则返回 logits 和输出结果的后续元素
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回数据字典，则返回 ImageClassifierOutputWithNoAttention 对象
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
# 从transformers.models.mobilevit.modeling_mobilevit中复制MobileViTASPPPooling类，将MobileViT->MobileViTV2
class MobileViTV2ASPPPooling(nn.Module):
    def __init__(self, config: MobileViTV2Config, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # 创建全局平均池化层，输出尺寸为1
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 创建1x1卷积层对象，使用MobileViTV2ConvLayer
        self.conv_1x1 = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_size = features.shape[-2:]  # 获取输入特征的空间维度
        features = self.global_pool(features)  # 对输入特征进行全局平均池化
        features = self.conv_1x1(features)  # 对池化后的特征进行1x1卷积
        features = nn.functional.interpolate(features, size=spatial_size, mode="bilinear", align_corners=False)  # 对特征进行插值恢复原空间尺寸
        return features  # 返回特征


class MobileViTV2ASPP(nn.Module):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__()

        encoder_out_channels = make_divisible(512 * config.width_multiplier, divisor=8)  # 获取编码器输出通道数（layer 5输出维度），做可分离卷积
        in_channels = encoder_out_channels  # 输入通道数
        out_channels = config.aspp_out_channels  # 输出通道数

        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        self.convs = nn.ModuleList()  # 初始化卷积层列表

        # 创建1x1卷积层对象，使用MobileViTV2ConvLayer
        in_projection = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
        )
        self.convs.append(in_projection)  # 将in_projection添加到模型的卷积层列表中

        # 对每个atrous_rate创建卷积层对象，使用MobileViTV2ConvLayer，并将其添加到模型的卷积层列表中
        self.convs.extend(
            [
                MobileViTV2ConvLayer(
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

        # 创建ASPP池化层对象，使用MobileViTV2ASPPPooling
        pool_layer = MobileViTV2ASPPPooling(config, in_channels, out_channels)
        self.convs.append(pool_layer)  # 将pool_layer添加到模型的卷积层列表中

        # 创建1x1卷积层对象，使用MobileViTV2ConvLayer，将所有卷积层的输出拼接到一起进行降维处理
        self.project = MobileViTV2ConvLayer(
            config, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, use_activation="relu"
        )

        # 创建Dropout层
        self.dropout = nn.Dropout(p=config.aspp_dropout_prob)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pyramid = []  # 初始化金字塔列表
        for conv in self.convs:  # 遍历所有卷积层
            pyramid.append(conv(features))  # 将每个卷积层的输出添加到金字塔列表中
        pyramid = torch.cat(pyramid, dim=1)  # 在通道维度上对金字塔中的特征进行拼接

        pooled_features = self.project(pyramid)  # 对拼接后的特征进行降维处理
        pooled_features = self.dropout(pooled_features)  # 对降维后的特征进行Dropout处理
        return pooled_features  # 返回处理后的特征
# 根据 MobileViTDeepLabV3 更改了类名，现在是 MobileViTV2DeepLabV3
class MobileViTV2DeepLabV3(nn.Module):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTV2Config) -> None:
        # 调用父类的构造函数
        super().__init__()
        # 创建 MobileViTV2ASPP 对象实例，并赋给 self.aspp 属性
        self.aspp = MobileViTV2ASPP(config)

        # 创建 Dropout2d 对象实例，并赋给 self.dropout 属性
        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)

        # 创建 MobileViTV2ConvLayer 对象实例，并赋给 self.classifier 属性
        self.classifier = MobileViTV2ConvLayer(
            config,
            in_channels=config.aspp_out_channels,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 获取最后一个隐藏状态，将其传递给 self.aspp 方法得到特征
        features = self.aspp(hidden_states[-1])
        # 对特征进行 dropout 操作
        features = self.dropout(features)
        # 使用 self.classifier 方法将特征映射到输出类别空间
        features = self.classifier(features)
        # 返回映射后的特征
        return features


@add_start_docstrings(
    """
    MobileViTV2 model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILEVITV2_START_DOCSTRING,
)
class MobileViTV2ForSemanticSegmentation(MobileViTV2PreTrainedModel):
    def __init__(self, config: MobileViTV2Config) -> None:
        # 调用父类的构造函数
        super().__init__(config)

        # 将输出的类别数保存到 self.num_labels 属性
        self.num_labels = config.num_labels
        # 创建 MobileViTV2Model 对象实例，并赋给 self.mobilevitv2 属性
        self.mobilevitv2 = MobileViTV2Model(config, expand_output=False)
        # 创建 MobileViTV2DeepLabV3 对象实例，并赋给 self.segmentation_head 属性
        self.segmentation_head = MobileViTV2DeepLabV3(config)

        # 初始化权重并应用最终的处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=SemanticSegmenterOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
    # 语义分割任务的前向传播
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        # 根据输入参数决定是否输出隐藏状态和是否使用返回字典格式
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 通过 MobileViT-V2 主干网络得到编码器的中间隐藏状态
        outputs = self.mobilevitv2(
            pixel_values,
            output_hidden_states=True, # 需要中间隐藏状态
            return_dict=return_dict,
        )
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
    
        # 使用语义分割头部将编码器特征映射到语义分割logits
        logits = self.segmentation_head(encoder_hidden_states)
    
        # 如果有标签，则计算语义分割损失
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # 将logits上采样到原图大小
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
    
        # 根据输出格式返回结果
        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
```