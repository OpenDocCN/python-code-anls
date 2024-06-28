# `.\models\levit\modeling_levit.py`

```
# coding=utf-8
# 声明文件编码为 UTF-8

# 版权声明及许可证信息

# 导入所需的库和模块
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# 导入 PyTorch 相关模块
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入模型输出类
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    ModelOutput,
)

# 导入预训练模型基类
from ...modeling_utils import PreTrainedModel

# 导入工具函数和日志记录
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 导入 LevitConfig 配置类
from .configuration_levit import LevitConfig

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 模型配置文档字符串
_CONFIG_FOR_DOC = "LevitConfig"

# 模型检查点文档字符串
_CHECKPOINT_FOR_DOC = "facebook/levit-128S"

# 预期输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 16, 384]

# 图像分类模型检查点
_IMAGE_CLASS_CHECKPOINT = "facebook/levit-128S"

# 图像分类预期输出示例
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型存档列表
LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/levit-128S",
    # 更多 Levit 模型可在 https://huggingface.co/models?filter=levit 查看
]

# LevitForImageClassificationWithTeacherOutput 类，继承自 ModelOutput 类
@dataclass
class LevitForImageClassificationWithTeacherOutput(ModelOutput):
    """
    [`LevitForImageClassificationWithTeacher`] 的输出类型。
    """

# 此处为代码块结束
    # logits参数是一个形状为(batch_size, config.num_labels)的张量，包含了预测分数，
    # 这些分数是cls_logits和distillation_logits的平均值。
    # cls_logits是分类头部的预测分数，即在类标记的最终隐藏状态之上的线性层。
    # distillation_logits是蒸馏头部的预测分数，即在蒸馏标记的最终隐藏状态之上的线性层。
    # hidden_states参数是一个可选的元组，包含了模型每一层的隐藏状态张量，
    # 形状为(batch_size, sequence_length, hidden_size)，包括初始嵌入输出。
    
    logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    distillation_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
class LevitConvEmbeddings(nn.Module):
    """
    LeViT Conv Embeddings with Batch Norm, used in the initial patch embedding layer.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bn_weight_init=1
    ):
        super().__init__()
        # 定义卷积层，用于将输入的图像数据转换成特征图
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False
        )
        # 定义批归一化层，用于规范化卷积输出的特征图
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, embeddings):
        # 执行卷积操作，将输入的嵌入数据转换成特征图
        embeddings = self.convolution(embeddings)
        # 执行批归一化操作，规范化卷积输出的特征图
        embeddings = self.batch_norm(embeddings)
        return embeddings


class LevitPatchEmbeddings(nn.Module):
    """
    LeViT patch embeddings, for final embeddings to be passed to transformer blocks. It consists of multiple
    `LevitConvEmbeddings`.
    """

    def __init__(self, config):
        super().__init__()
        # 第一个卷积嵌入层及其激活函数
        self.embedding_layer_1 = LevitConvEmbeddings(
            config.num_channels, config.hidden_sizes[0] // 8, config.kernel_size, config.stride, config.padding
        )
        self.activation_layer_1 = nn.Hardswish()

        # 第二个卷积嵌入层及其激活函数
        self.embedding_layer_2 = LevitConvEmbeddings(
            config.hidden_sizes[0] // 8, config.hidden_sizes[0] // 4, config.kernel_size, config.stride, config.padding
        )
        self.activation_layer_2 = nn.Hardswish()

        # 第三个卷积嵌入层及其激活函数
        self.embedding_layer_3 = LevitConvEmbeddings(
            config.hidden_sizes[0] // 4, config.hidden_sizes[0] // 2, config.kernel_size, config.stride, config.padding
        )
        self.activation_layer_3 = nn.Hardswish()

        # 第四个卷积嵌入层，不带激活函数
        self.embedding_layer_4 = LevitConvEmbeddings(
            config.hidden_sizes[0] // 2, config.hidden_sizes[0], config.kernel_size, config.stride, config.padding
        )
        self.num_channels = config.num_channels

    def forward(self, pixel_values):
        # 检查输入的像素值张量是否与配置中设置的通道数匹配
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 依次执行四个卷积嵌入层及其后的激活函数
        embeddings = self.embedding_layer_1(pixel_values)
        embeddings = self.activation_layer_1(embeddings)
        embeddings = self.embedding_layer_2(embeddings)
        embeddings = self.activation_layer_2(embeddings)
        embeddings = self.embedding_layer_3(embeddings)
        embeddings = self.activation_layer_3(embeddings)
        embeddings = self.embedding_layer_4(embeddings)
        # 将结果展平并转置，以便传递给变压器块
        return embeddings.flatten(2).transpose(1, 2)


class MLPLayerWithBN(nn.Module):
    """
    MLP layer with Batch Norm, used in the transformer blocks.
    """

    def __init__(self, input_dim, output_dim, bn_weight_init=1):
        super().__init__()
        # 定义线性层，用于进行全连接操作
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)
        # 定义批归一化层，用于规范化全连接层的输出
        self.batch_norm = nn.BatchNorm1d(output_dim)
    # 定义前向传播方法，接收隐藏状态作为输入参数
    def forward(self, hidden_state):
        # 将隐藏状态通过线性层进行变换
        hidden_state = self.linear(hidden_state)
        # 将变换后的隐藏状态展平并应用批归一化处理
        hidden_state = self.batch_norm(hidden_state.flatten(0, 1)).reshape_as(hidden_state)
        # 返回处理后的隐藏状态
        return hidden_state
# 定义一个名为 LevitSubsample 的自定义神经网络模块，继承自 nn.Module
class LevitSubsample(nn.Module):
    # 初始化函数，接受步长（stride）和分辨率（resolution）两个参数
    def __init__(self, stride, resolution):
        super().__init__()
        # 设置对象属性 stride 和 resolution
        self.stride = stride
        self.resolution = resolution

    # 前向传播函数，接受隐藏状态（hidden_state）作为输入
    def forward(self, hidden_state):
        # 获取输入张量的批量大小（batch_size）、通道数（channels）
        batch_size, _, channels = hidden_state.shape
        # 将隐藏状态重新视图化为指定分辨率的形状，并进行下采样
        hidden_state = hidden_state.view(batch_size, self.resolution, self.resolution, channels)[
            :, :: self.stride, :: self.stride
        ].reshape(batch_size, -1, channels)
        # 返回下采样后的隐藏状态张量
        return hidden_state


# 定义一个名为 LevitAttention 的自定义神经网络模块，继承自 nn.Module
class LevitAttention(nn.Module):
    # 初始化函数，接受隐藏层大小（hidden_sizes）、键维度（key_dim）、注意力头数（num_attention_heads）、
    # 注意力比率（attention_ratio）、分辨率（resolution）五个参数
    def __init__(self, hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution):
        super().__init__()
        # 设置对象属性
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        # 计算键-值对和投影输出的维度
        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads * 2
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads

        # 创建查询、键和值的 MLP 层并进行批归一化
        self.queries_keys_values = MLPLayerWithBN(hidden_sizes, self.out_dim_keys_values)
        # 激活函数采用 Hardswish
        self.activation = nn.Hardswish()
        # 创建投影层的 MLP 层并进行批归一化
        self.projection = MLPLayerWithBN(self.out_dim_projection, hidden_sizes, bn_weight_init=0)

        # 生成所有可能点的笛卡尔积
        points = list(itertools.product(range(resolution), range(resolution)))
        len_points = len(points)
        attention_offsets, indices = {}, []

        # 计算所有点对之间的偏移量及其对应的索引
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])

        # 初始化注意力偏置的缓存和参数
        self.attention_bias_cache = {}
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_attention_heads, len(attention_offsets)))
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(indices).view(len_points, len_points), persistent=False
        )

    # 用于训练时无梯度更新的装饰器函数
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        # 如果是训练模式且存在注意力偏置缓存，则清空缓存
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # 清空注意力偏置缓存

    # 获取注意力偏置的函数，根据设备类型缓存不同的注意力偏置
    def get_attention_biases(self, device):
        if self.training:
            # 如果是训练模式，则直接返回计算得到的注意力偏置
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            # 如果是推断模式，则根据设备类型缓存注意力偏置
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]
    # 定义一个前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_state):
        # 获取输入隐藏状态的批大小、序列长度和特征维度
        batch_size, seq_length, _ = hidden_state.shape
        # 使用self.queries_keys_values方法计算查询、键和值
        queries_keys_values = self.queries_keys_values(hidden_state)
        # 将查询、键、值重新组织成指定形状，以便进行多头注意力计算
        query, key, value = queries_keys_values.view(batch_size, seq_length, self.num_attention_heads, -1).split(
            [self.key_dim, self.key_dim, self.attention_ratio * self.key_dim], dim=3
        )
        # 将查询张量转置，以适应多头注意力计算的形状要求
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # 计算注意力分数，包括缩放和注意力偏置
        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(hidden_state.device)
        # 对注意力分数进行 softmax 归一化
        attention = attention.softmax(dim=-1)
        # 计算加权后的值向量，然后重新排列以恢复原始形状
        hidden_state = (attention @ value).transpose(1, 2).reshape(batch_size, seq_length, self.out_dim_projection)
        # 应用激活函数、投影层和最终投影，得到最终的隐藏状态表示
        hidden_state = self.projection(self.activation(hidden_state))
        # 返回处理后的隐藏状态
        return hidden_state
class LevitAttentionSubsample(nn.Module):
    # LevitAttentionSubsample 类，继承自 nn.Module
    def __init__(
        self,
        input_dim,
        output_dim,
        key_dim,
        num_attention_heads,
        attention_ratio,
        stride,
        resolution_in,
        resolution_out,
    ):
        # 初始化函数，设置模块参数
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5  # 缩放因子，用于缩放注意力机制中的键值
        self.key_dim = key_dim  # 注意力键的维度
        self.attention_ratio = attention_ratio  # 注意力比率
        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads
        self.resolution_out = resolution_out
        # resolution_in 是初始分辨率，resolution_out 是下采样后的最终分辨率

        # 初始化模块：MLPLayerWithBN 是带批量归一化的 MLP 层
        self.keys_values = MLPLayerWithBN(input_dim, self.out_dim_keys_values)
        self.queries_subsample = LevitSubsample(stride, resolution_in)  # 对查询进行下采样
        self.queries = MLPLayerWithBN(input_dim, key_dim * num_attention_heads)  # 查询的 MLP 层
        self.activation = nn.Hardswish()  # 激活函数使用 Hardswish
        self.projection = MLPLayerWithBN(self.out_dim_projection, output_dim)  # 投影到最终输出维度

        self.attention_bias_cache = {}  # 初始化注意力偏置缓存

        # 计算注意力偏置的索引
        points = list(itertools.product(range(resolution_in), range(resolution_in)))
        points_ = list(itertools.product(range(resolution_out), range(resolution_out)))
        len_points, len_points_ = len(points), len(points_)
        attention_offsets, indices = {}, []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2), abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])

        # 初始化注意力偏置参数和索引
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_attention_heads, len(attention_offsets)))
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(indices).view(len_points_, len_points), persistent=False
        )

    @torch.no_grad()
    def train(self, mode=True):
        # 重写父类的 train 方法，并设置为不需要梯度
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # 如果是训练模式且注意力偏置缓存不为空，则清空缓存

    def get_attention_biases(self, device):
        # 获取注意力偏置方法
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]  # 如果是训练模式，直接返回注意力偏置
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
                # 如果设备键不在缓存中，则将注意力偏置缓存到设备键
            return self.attention_bias_cache[device_key]  # 返回设备键对应的注意力偏置
    # 定义前向传播方法，接收隐藏状态作为输入
    def forward(self, hidden_state):
        # 获取输入张量的批量大小、序列长度和特征维度
        batch_size, seq_length, _ = hidden_state.shape
        
        # 使用 self.keys_values 方法生成键和值，然后重新组织张量形状
        key, value = (
            self.keys_values(hidden_state)
            .view(batch_size, seq_length, self.num_attention_heads, -1)
            .split([self.key_dim, self.attention_ratio * self.key_dim], dim=3)
        )
        
        # 对键张量进行维度重排，以便后续计算注意力
        key = key.permute(0, 2, 1, 3)
        
        # 对值张量进行维度重排，以便后续计算注意力
        value = value.permute(0, 2, 1, 3)

        # 使用 self.queries_subsample 方法对隐藏状态进行查询抽样
        query = self.queries(self.queries_subsample(hidden_state))
        
        # 重新组织查询张量的形状，以便后续计算注意力
        query = query.view(batch_size, self.resolution_out**2, self.num_attention_heads, self.key_dim).permute(
            0, 2, 1, 3
        )

        # 计算注意力分数，包括缩放、添加注意力偏置
        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(hidden_state.device)
        
        # 对注意力分数进行 softmax 归一化
        attention = attention.softmax(dim=-1)
        
        # 计算加权后的值张量，然后进行维度重排和形状调整
        hidden_state = (attention @ value).transpose(1, 2).reshape(batch_size, -1, self.out_dim_projection)
        
        # 对加权后的值张量应用激活函数和投影层
        hidden_state = self.projection(self.activation(hidden_state))
        
        # 返回处理后的隐藏状态张量
        return hidden_state
# 定义一个 LevitMLPLayer 类，继承自 nn.Module，用于实现 MLP 层，相比 ViT 只扩展 2 倍。
class LevitMLPLayer(nn.Module):
    """
    MLP Layer with `2X` expansion in contrast to ViT with `4X`.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 使用带有批归一化的 MLPLayerWithBN 类来定义线性变换（升维）
        self.linear_up = MLPLayerWithBN(input_dim, hidden_dim)
        # 激活函数使用 Hardswish
        self.activation = nn.Hardswish()
        # 使用带有批归一化的 MLPLayerWithBN 类定义线性变换（降维）
        self.linear_down = MLPLayerWithBN(hidden_dim, input_dim)

    def forward(self, hidden_state):
        # 执行升维操作
        hidden_state = self.linear_up(hidden_state)
        # 应用激活函数 Hardswish
        hidden_state = self.activation(hidden_state)
        # 执行降维操作
        hidden_state = self.linear_down(hidden_state)
        return hidden_state


# 定义一个 LevitResidualLayer 类，继承自 nn.Module，用于实现 LeViT 的残差块。
class LevitResidualLayer(nn.Module):
    """
    Residual Block for LeViT
    """

    def __init__(self, module, drop_rate):
        super().__init__()
        # 保存作为残差的模块
        self.module = module
        # 设定丢弃率（dropout rate）
        self.drop_rate = drop_rate

    def forward(self, hidden_state):
        # 如果处于训练模式并且设置了丢弃率
        if self.training and self.drop_rate > 0:
            # 随机生成与隐藏状态维度相同的随机数张量，用于丢弃
            rnd = torch.rand(hidden_state.size(0), 1, 1, device=hidden_state.device)
            # 将随机数张量转换为掩码，根据丢弃率进行归一化
            rnd = rnd.ge_(self.drop_rate).div(1 - self.drop_rate).detach()
            # 计算残差块的输出，同时应用丢弃掩码
            hidden_state = hidden_state + self.module(hidden_state) * rnd
            return hidden_state
        else:
            # 计算残差块的输出
            hidden_state = hidden_state + self.module(hidden_state)
            return hidden_state


# 定义一个 LevitStage 类，继承自 nn.Module，表示 LeViT 模型中的一个阶段，包括 LevitMLPLayer 和 LevitAttention 层。
class LevitStage(nn.Module):
    """
    LeViT Stage consisting of `LevitMLPLayer` and `LevitAttention` layers.
    """

    def __init__(
        self,
        config,
        idx,
        hidden_sizes,
        key_dim,
        depths,
        num_attention_heads,
        attention_ratio,
        mlp_ratio,
        down_ops,
        resolution_in,
        ):
        super().__init__()
        # 初始化 LeViT 阶段的参数和配置
        self.config = config
        self.idx = idx
        self.hidden_sizes = hidden_sizes
        self.key_dim = key_dim
        self.depths = depths
        self.num_attention_heads = num_attention_heads
        self.attention_ratio = attention_ratio
        self.mlp_ratio = mlp_ratio
        self.down_ops = down_ops
        self.resolution_in = resolution_in
    ):
        # 调用父类的构造函数初始化对象
        super().__init__()
        # 初始化图层列表
        self.layers = []
        # 设置配置参数
        self.config = config
        # 设置初始分辨率和最终分辨率
        self.resolution_in = resolution_in
        # resolution_in 是初始分辨率，resolution_out 是经过降采样后的最终分辨率

        # 根据深度循环构建层对象
        for _ in range(depths):
            # 添加注意力机制层到层列表
            self.layers.append(
                LevitResidualLayer(
                    LevitAttention(hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution_in),
                    self.config.drop_path_rate,
                )
            )
            # 如果 mlp_ratio 大于 0，则构建 MLP 层并添加到层列表
            if mlp_ratio > 0:
                hidden_dim = hidden_sizes * mlp_ratio
                self.layers.append(
                    LevitResidualLayer(LevitMLPLayer(hidden_sizes, hidden_dim), self.config.drop_path_rate)
                )

        # 如果第一个 down_ops 是 "Subsample"，则执行下采样操作
        if down_ops[0] == "Subsample":
            # 计算经过下采样后的分辨率 resolution_out
            self.resolution_out = (self.resolution_in - 1) // down_ops[5] + 1
            # 添加下采样注意力机制层到层列表
            self.layers.append(
                LevitAttentionSubsample(
                    *self.config.hidden_sizes[idx : idx + 2],
                    key_dim=down_ops[1],
                    num_attention_heads=down_ops[2],
                    attention_ratio=down_ops[3],
                    stride=down_ops[5],
                    resolution_in=resolution_in,
                    resolution_out=self.resolution_out,
                )
            )
            # 更新当前分辨率为下采样后的分辨率
            self.resolution_in = self.resolution_out
            # 如果 down_ops[4] 大于 0，则构建 MLP 层并添加到层列表
            if down_ops[4] > 0:
                hidden_dim = self.config.hidden_sizes[idx + 1] * down_ops[4]
                self.layers.append(
                    LevitResidualLayer(
                        LevitMLPLayer(self.config.hidden_sizes[idx + 1], hidden_dim), self.config.drop_path_rate
                    )
                )

        # 将层列表转换为 nn.ModuleList 对象
        self.layers = nn.ModuleList(self.layers)

    # 获取当前模型的分辨率
    def get_resolution(self):
        return self.resolution_in

    # 前向传播函数
    def forward(self, hidden_state):
        # 对每一层进行前向传播计算
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        # 返回最终的隐藏状态
        return hidden_state
class LevitEncoder(nn.Module):
    """
    LeViT Encoder consisting of multiple `LevitStage` stages.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config  # 初始化模型配置参数
        resolution = self.config.image_size // self.config.patch_size  # 计算分辨率
        self.stages = []  # 初始化阶段列表
        self.config.down_ops.append([""])  # 将空字符串追加到下采样操作列表中（可能是个bug）

        for stage_idx in range(len(config.depths)):  # 遍历每个阶段的深度
            stage = LevitStage(  # 创建LevitStage阶段实例
                config,
                stage_idx,
                config.hidden_sizes[stage_idx],
                config.key_dim[stage_idx],
                config.depths[stage_idx],
                config.num_attention_heads[stage_idx],
                config.attention_ratio[stage_idx],
                config.mlp_ratio[stage_idx],
                config.down_ops[stage_idx],
                resolution,
            )
            resolution = stage.get_resolution()  # 获取当前阶段的分辨率
            self.stages.append(stage)  # 将当前阶段添加到阶段列表中

        self.stages = nn.ModuleList(self.stages)  # 转换阶段列表为PyTorch的模块列表

    def forward(self, hidden_state, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None  # 初始化所有隐藏状态的元组或空值

        for stage in self.stages:  # 遍历所有阶段
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)  # 将当前隐藏状态添加到所有隐藏状态元组中
            hidden_state = stage(hidden_state)  # 将隐藏状态传递给当前阶段进行前向计算

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)  # 将最终隐藏状态添加到所有隐藏状态元组中
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states] if v is not None)  # 如果不返回字典，则返回所有非空的值元组

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=all_hidden_states)  # 返回基本模型输出对象


class LevitClassificationLayer(nn.Module):
    """
    LeViT Classification Layer
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)  # 初始化批标准化层
        self.linear = nn.Linear(input_dim, output_dim)  # 初始化线性层

    def forward(self, hidden_state):
        hidden_state = self.batch_norm(hidden_state)  # 批标准化操作
        logits = self.linear(hidden_state)  # 计算输出logits
        return logits  # 返回logits


class LevitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LevitConfig  # 设置配置类为LevitConfig
    base_model_prefix = "levit"  # 基础模型前缀名
    main_input_name = "pixel_values"  # 主输入名称

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):  # 如果是线性层或卷积层
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # 使用正态分布初始化权重
            if module.bias is not None:
                module.bias.data.zero_()  # 如果存在偏置，则初始化为零
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):  # 如果是批标准化层
            module.bias.data.zero_()  # 初始化偏置为零
            module.weight.data.fill_(1.0)  # 初始化权重为1.0
# 定义 LevitModel 类，继承自 LevitPreTrainedModel，用于构建 Levit 模型
@add_start_docstrings(
    "The bare Levit model outputting raw features without any specific head on top.",  # 添加关于 Levit 模型的文档说明
    LEVIT_START_DOCSTRING,  # 添加 Levit 模型的配置参数说明和初始化信息
)
class LevitModel(LevitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)  # 调用父类 LevitPreTrainedModel 的初始化方法
        self.config = config  # 将传入的配置参数 config 存储为实例变量
        self.patch_embeddings = LevitPatchEmbeddings(config)  # 初始化图像的 patch embeddings
        self.encoder = LevitEncoder(config)  # 初始化 Levit 编码器
        # Initialize weights and apply final processing
        self.post_init()  # 调用自定义的 post_init 方法，用于初始化权重和应用最终处理

    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)  # 添加前向传播函数的文档说明，包括输入参数
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 添加代码示例的文档说明，显示如何使用模型
        output_type=BaseModelOutputWithPoolingAndNoAttention,  # 指定输出类型的文档说明
        config_class=_CONFIG_FOR_DOC,  # 指定模型配置类的文档说明
        modality="vision",  # 指明模型适用的领域为视觉
        expected_output=_EXPECTED_OUTPUT_SHAPE,  # 添加预期输出形状的文档说明
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,  # 输入参数 pixel_values，代表像素值的浮点张量
        output_hidden_states: Optional[bool] = None,  # 是否返回所有层的隐藏状态的布尔值参数
        return_dict: Optional[bool] = None,  # 是否返回 ModelOutput 对象而不是普通元组的布尔值参数
        ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 定义函数签名，指定返回类型为元组或BaseModelOutputWithPoolingAndNoAttention类型

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果output_hidden_states不为None，则使用其值；否则使用self.config.output_hidden_states的值

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果return_dict不为None，则使用其值；否则使用self.config.use_return_dict的值

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 如果pixel_values为None，则抛出数值错误异常，要求指定pixel_values

        embeddings = self.patch_embeddings(pixel_values)
        # 将像素值转换为嵌入向量

        encoder_outputs = self.encoder(
            embeddings,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 使用编码器对嵌入向量进行编码，返回编码器的输出

        last_hidden_state = encoder_outputs[0]
        # 取编码器输出的第一个元素作为最终的隐藏状态表示

        # global average pooling, (batch_size, seq_length, hidden_sizes) -> (batch_size, hidden_sizes)
        pooled_output = last_hidden_state.mean(dim=1)
        # 对最终隐藏状态进行全局平均池化，将每个序列的隐藏状态平均到一个向量中

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        # 如果return_dict为False，则返回元组形式的输出：最终隐藏状态、池化输出以及其余的编码器输出

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
        # 如果return_dict为True，则使用BaseModelOutputWithPoolingAndNoAttention类封装最终隐藏状态、池化输出和所有隐藏状态的列表，并返回该对象
# 定义一个 Levit 图像分类模型，基于 Levit 模型并添加一个分类器头部
@add_start_docstrings(
    """
    Levit Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    LEVIT_START_DOCSTRING,
)
class LevitForImageClassification(LevitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config  # 保存配置信息
        self.num_labels = config.num_labels  # 获取标签数量
        self.levit = LevitModel(config)  # 初始化基础的 Levit 模型

        # 分类器头部
        self.classifier = (
            # 如果标签数量大于 0，则创建 Levit 分类层；否则创建一个恒等映射
            LevitClassificationLayer(config.hidden_sizes[-1], config.num_labels)
            if config.num_labels > 0
            else torch.nn.Identity()
        )

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.levit 方法，传入像素值 pixel_values 和其他参数，获取模型输出
        outputs = self.levit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 从模型输出中获取序列输出（通常是最后一层隐藏状态的输出），并计算其在第1维度上的平均值
        sequence_output = outputs[0]
        sequence_output = sequence_output.mean(1)

        # 将平均后的序列输出输入分类器，得到 logits（未经 softmax 处理的分类器输出）
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None

        # 如果提供了 labels，则计算损失
        if labels is not None:
            # 如果未指定问题类型，则根据情况自动判断问题类型（回归、单标签分类、多标签分类）
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失
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

        # 如果 return_dict 为 False，则返回一个元组，包含 logits 和可能的其他模型输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则构建一个 ImageClassifierOutputWithNoAttention 对象，并返回
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
@add_start_docstrings(
    """
    LeViT Model transformer with image classification heads on top (a linear layer on top of the final hidden state and
    a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet. .. warning::
           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.
    """,
    LEVIT_START_DOCSTRING,
)
class LevitForImageClassificationWithTeacher(LevitPreTrainedModel):
    """
    构建一个基于 LeViT 模型的图像分类器，带有两个分类头部（一个用于最终隐藏状态的线性层，另一个用于蒸馏令牌最终隐藏状态的线性层），适用于 ImageNet 等数据集。
    注意：该模型仅支持推断，暂不支持使用蒸馏（即与教师模型进行微调）。

    Attributes:
        config (LevitConfig): 模型的配置对象，包含模型的各种参数设定。
        num_labels (int): 分类任务中的标签数量。
        levit (LevitModel): 底层的 LeViT 模型实例。

    """
    def __init__(self, config):
        """
        初始化方法，用于创建一个新的 LevitForImageClassificationWithTeacher 实例。

        Args:
            config (LevitConfig): 模型的配置对象，包含模型的各种参数设定。
        """
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.levit = LevitModel(config)

        # Classifier head
        self.classifier = (
            LevitClassificationLayer(config.hidden_sizes[-1], config.num_labels)
            if config.num_labels > 0
            else torch.nn.Identity()
        )
        self.classifier_distill = (
            LevitClassificationLayer(config.hidden_sizes[-1], config.num_labels)
            if config.num_labels > 0
            else torch.nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=LevitForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LevitForImageClassificationWithTeacherOutput]:
        """
        前向传播方法，执行模型的推断过程。

        Args:
            pixel_values (torch.FloatTensor, optional): 输入的像素值张量。默认为 None。
            output_hidden_states (bool, optional): 是否返回隐藏状态。默认为 None。
            return_dict (bool, optional): 是否以字典形式返回输出。默认为 None。

        Returns:
            Union[Tuple, LevitForImageClassificationWithTeacherOutput]: 根据 return_dict 的设置，返回不同的输出形式。
                如果 return_dict 为 False，则返回一个元组。
                如果 return_dict 为 True，则返回一个 LevitForImageClassificationWithTeacherOutput 对象。

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 LeViT 模型进行前向传播
        outputs = self.levit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 获取序列输出并对其进行平均池化
        sequence_output = outputs[0]
        sequence_output = sequence_output.mean(1)

        # 分别通过分类头部和蒸馏头部计算 logits
        cls_logits, distill_logits = self.classifier(sequence_output), self.classifier_distill(sequence_output)
        logits = (cls_logits + distill_logits) / 2

        if not return_dict:
            # 如果 return_dict 为 False，则返回一个元组形式的输出
            output = (logits, cls_logits, distill_logits) + outputs[2:]
            return output

        # 如果 return_dict 为 True，则返回一个 LevitForImageClassificationWithTeacherOutput 对象
        return LevitForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distill_logits,
            hidden_states=outputs.hidden_states,
        )
```