# `.\transformers\models\levit\modeling_levit.py`

```py
# 设置编码为 UTF-8

# 版权声明及许可证信息
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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

""" PyTorch LeViT model."""

# 导入所需模块
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入模型输出相关的类
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    ModelOutput,
)
# 导入预训练模型的类和函数
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入配置文件类
from .configuration_levit import LevitConfig

# 设置日志记录器
logger = logging.get_logger(__name__)

# 用于文档的通用字符串
_CONFIG_FOR_DOC = "LevitConfig"

# 用于文档的基本字符串
_CHECKPOINT_FOR_DOC = "facebook/levit-128S"
_EXPECTED_OUTPUT_SHAPE = [1, 16, 384]

# 图像分类的文档字符串
_IMAGE_CLASS_CHECKPOINT = "facebook/levit-128S"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型的存档列表
LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/levit-128S",
    # 查看所有 LeViT 模型 https://huggingface.co/models?filter=levit
]


@dataclass
class LevitForImageClassificationWithTeacherOutput(ModelOutput):
    """
    [`LevitForImageClassificationWithTeacher`] 的输出类型。
    """
    # 定义函数的参数，包括：
    # logits: 预测分数，形状为(batch_size, config.num_labels)，是分类预测和蒸馏预测的平均值
    # cls_logits: 分类头部的预测分数，即最终隐藏状态的线性层的预测分数
    # distillation_logits: 蒸馏头部的预测分数，即最终隐藏状态的线性层的预测分数
    # hidden_states: 可选参数，当传递output_hidden_states=True时返回，或者当config.output_hidden_states=True时返回，
    #                是一个元组的torch.FloatTensor，形状为(batch_size, sequence_length, hidden_size)，
    #                包含模型在每个层的隐藏状态以及初始嵌入输出
    """

    # 初始化 logits 参数，默认值为 None
    logits: torch.FloatTensor = None
    # 初始化 cls_logits 参数，默认值为 None
    cls_logits: torch.FloatTensor = None
    # 初始化 distillation_logits 参数，默认值为 None
    distillation_logits: torch.FloatTensor = None
    # 初始化 hidden_states 参数，默认值为 None，可选参数
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个继承自 PyTorch nn.Module 的 LevitConvEmbeddings 类
class LevitConvEmbeddings(nn.Module):
    """
    LeViT 卷积嵌入层，包括批量归一化，用于初始的 patch 嵌入层。
    """

    # 类的初始化方法，接收多个参数
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bn_weight_init=1
    ):
        # 初始化父类
        super().__init__()
        # 创建二维卷积层，没有偏置参数
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False
        )
        # 创建二维批量归一化层，指定输出通道
        self.batch_norm = nn.BatchNorm2d(out_channels)

    # 定义前向传播函数
    def forward(self, embeddings):
        # 使用卷积层对输入进行卷积操作
        embeddings = self.convolution(embeddings)
        # 使用批量归一化层对卷积结果进行归一化
        embeddings = self.batch_norm(embeddings)
        # 返回归一化后的嵌入
        return embeddings


# 定义 LevitPatchEmbeddings 类，继承自 nn.Module
class LevitPatchEmbeddings(nn.Module):
    """
    LeViT 的 patch 嵌入层，输出最终嵌入数据，用于传递到 transformer 模块中。
    该类由多个 LevitConvEmbeddings 组成。
    """

    # 初始化方法，接收配置对象作为参数
    def __init__(self, config):
        # 初始化父类
        super().__init__()
        # 初始化第一个 LevitConvEmbeddings 层和激活函数
        self.embedding_layer_1 = LevitConvEmbeddings(
            config.num_channels, config.hidden_sizes[0] // 8, config.kernel_size, config.stride, config.padding
        )
        self.activation_layer_1 = nn.Hardswish()

        # 初始化第二个 LevitConvEmbeddings 层和激活函数
        self.embedding_layer_2 = LevitConvEmbeddings(
            config.hidden_sizes[0] // 8, config.hidden_sizes[0] // 4, config.kernel_size, config.stride, config.padding
        )
        self.activation_layer_2 = nn.Hardswish()

        # 初始化第三个 LevitConvEmbeddings 层和激活函数
        self.embedding_layer_3 = LevitConvEmbeddings(
            config.hidden_sizes[0] // 4, config.hidden_sizes[0] // 2, config.kernel_size, config.stride, config.padding
        )
        self.activation_layer_3 = nn.Hardswish()

        # 初始化第四个 LevitConvEmbeddings 层
        self.embedding_layer_4 = LevitConvEmbeddings(
            config.hidden_sizes[0] // 2, config.hidden_sizes[0], config.kernel_size, config.stride, config.padding
        )
        # 保存配置的通道数
        self.num_channels = config.num_channels

    # 定义前向传播函数
    def forward(self, pixel_values):
        # 获取输入数据的通道数
        num_channels = pixel_values.shape[1]
        # 确保输入通道数与配置中的通道数一致
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 执行第一层嵌入操作
        embeddings = self.embedding_layer_1(pixel_values)
        # 执行第一个激活函数
        embeddings = self.activation_layer_1(embeddings)
        # 执行第二层嵌入操作
        embeddings = self.embedding_layer_2(embeddings)
        # 执行第二个激活函数
        embeddings = self.activation_layer_2(embeddings)
        # 执行第三层嵌入操作
        embeddings = self.embedding_layer_3(embeddings)
        # 执行第三个激活函数
        embeddings = self.activation_layer_3(embeddings)
        # 执行第四层嵌入操作
        embeddings = self.embedding_layer_4(embeddings)
        # 对嵌入结果进行扁平化并转置
        return embeddings.flatten(2).transpose(1, 2)


# 定义 MLPLayerWithBN 类，继承自 nn.Module
class MLPLayerWithBN(nn.Module):
    # 初始化方法，接收输入和输出维度作为参数
    def __init__(self, input_dim, output_dim, bn_weight_init=1):
        # 初始化父类
        super().__init__()
        # 创建线性层，没有偏置参数
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)
        # 创建一维批量归一化层
        self.batch_norm = nn.BatchNorm1d(output_dim)
    # 前向传播方法，接收隐藏状态作为输入
    def forward(self, hidden_state):
        # 将隐藏状态通过线性层处理
        hidden_state = self.linear(hidden_state)
        # 将处理后的隐藏状态展平，并进行批量归一化，然后恢复原始形状
        hidden_state = self.batch_norm(hidden_state.flatten(0, 1)).reshape_as(hidden_state)
        # 返回处理后的隐藏状态
        return hidden_state
class LevitSubsample(nn.Module):
    # 初始化 LevitSubsample 类
    def __init__(self, stride, resolution):
        super().__init__()
        # 设置步幅和分辨率
        self.stride = stride
        self.resolution = resolution

    # 前向传播函数
    def forward(self, hidden_state):
        # 获取批大小、通道数
        batch_size, _, channels = hidden_state.shape
        # 重塑隐藏状态，并进行下采样
        hidden_state = hidden_state.view(batch_size, self.resolution, self.resolution, channels)[
            :, :: self.stride, :: self.stride
        ].reshape(batch_size, -1, channels)
        return hidden_state

class LevitAttention(nn.Module):
    # 初始化 LevitAttention 类
    def __init__(self, hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution):
        super().__init__()
        # 设置注意力头数量、缩放因子、关键维度、注意力比例和输出维度
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads * 2
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads

        # 定义查询、关键、数值的全连接层和激活函数
        self.queries_keys_values = MLPLayerWithBN(hidden_sizes, self.out_dim_keys_values)
        self.activation = nn.Hardswish()
        self.projection = MLPLayerWithBN(self.out_dim_projection, hidden_sizes, bn_weight_init=0)

        # 生成所有点的组合和注意力偏移量
        points = list(itertools.product(range(resolution), range(resolution)))
        len_points = len(points)
        attention_offsets, indices = {}, []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])

        # 初始化注意力偏置参数
        self.attention_bias_cache = {}
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_attention_heads, len(attention_offsets)))
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(indices).view(len_points, len_points), persistent=False
        )

    # 非训练时调用，用于清空注意力偏置缓存
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    # 获取注意力偏置
    def get_attention_biases(self, device):
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]
    def forward(self, hidden_state):
        # 获取输入的张量hidden_state的维度信息
        batch_size, seq_length, _ = hidden_state.shape
        # 使用queries_keys_values方法处理hidden_state张量，将结果保存为变量queries_keys_values
        queries_keys_values = self.queries_keys_values(hidden_state)
        # 将queries_keys_values按照维度进行切片，得到query、key、value三个子张量
        query, key, value = queries_keys_values.view(batch_size, seq_length, self.num_attention_heads, -1).split(
            [self.key_dim, self.key_dim, self.attention_ratio * self.key_dim], dim=3
        )
        # 将query张量的维度重新排列为(batch_size, num_attention_heads, seq_length, key_dim)
        query = query.permute(0, 2, 1, 3)
        # 将key张量的维度重新排列为(batch_size, num_attention_heads, seq_length, key_dim)
        key = key.permute(0, 2, 1, 3)
        # 将value张量的维度重新排列为(batch_size, num_attention_heads, seq_length, attention_ratio * key_dim)
        value = value.permute(0, 2, 1, 3)

        # 计算注意力分数，将query和key张量进行矩阵乘法，并按照字符维度进行缩放
        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(hidden_state.device)
        # 将注意力分数通过softmax函数进行归一化
        attention = attention.softmax(dim=-1)
        # 使用注意力分数对value张量进行加权求和，并进行维度的转置和重塑
        hidden_state = (attention @ value).transpose(1, 2).reshape(batch_size, seq_length, self.out_dim_projection)
        # 将hidden_state张量通过激活函数activation进行激活，并通过投影层projection进行线性变换
        hidden_state = self.projection(self.activation(hidden_state))
        # 返回处理后的hidden_state张量
        return hidden_state
# 定义 LevitAttentionSubsample 类，继承自 nn.Module
class LevitAttentionSubsample(nn.Module):
    # 初始化函数，接受输入维度、输出维度、键维度、注意力头数、注意力比例、步幅、初始分辨率和最终分辨率等参数
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
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads
        self.resolution_out = resolution_out
        # 以下是用于创建各种 MLP 层
        self.keys_values = MLPLayerWithBN(input_dim, self.out_dim_keys_values)
        self.queries_subsample = LevitSubsample(stride, resolution_in)
        self.queries = MLPLayerWithBN(input_dim, key_dim * num_attention_heads)
        self.activation = nn.Hardswish()
        self.projection = MLPLayerWithBN(self.out_dim_projection, output_dim)

        self.attention_bias_cache = {}  # 初始化注意力偏置缓存为空字典

        points = list(itertools.product(range(resolution_in), range(resolution_in)))  # 生成 (resolution_in * resolution_in) 个坐标点
        points_ = list(itertools.product(range(resolution_out), range(resolution_out)))  # 生成 (resolution_out * resolution_out) 个坐标点
        len_points, len_points_ = len(points), len(points_)  # 获取坐标点列表的长度
        attention_offsets, indices = {}, []  # 初始化偏移量字典和索引列表
        # 计算注意力偏移量和索引
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2), abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])

        # 创建注意力偏置参数
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_attention_heads, len(attention_offsets)))
        # 注册注意力偏置索引为 buffer
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(indices).view(len_points_, len_points), persistent=False
        )

    # 在无梯度下的训练模式下执行的函数
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # 清空注意力偏置缓存

    # 获取注意力偏置的函数
    def get_attention_biases(self, device):
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]
    # 前向传播方法，用于处理输入的隐藏状态
    def forward(self, hidden_state):
        # 获取隐藏状态的维度信息：批量大小、序列长度和特征维度
        batch_size, seq_length, _ = hidden_state.shape
        
        # 使用 self.keys_values 方法处理隐藏状态，得到 key 和 value
        # 调整维度以适应多头注意力的计算，并拆分得到 key 和 value
        key, value = (
            self.keys_values(hidden_state)
            .view(batch_size, seq_length, self.num_attention_heads, -1)
            .split([self.key_dim, self.attention_ratio * self.key_dim], dim=3)
        )
        
        # 将 key 的维度重新排列，以适应注意力计算所需的形状
        key = key.permute(0, 2, 1, 3)
        # 将 value 的维度重新排列，以适应注意力计算所需的形状
        value = value.permute(0, 2, 1, 3)

        # 使用 self.queries_subsample 方法对隐藏状态进行子采样，然后通过 self.queries 处理得到 query
        # 调整维度以适应多头注意力的计算，并将其维度重新排列
        query = self.queries(self.queries_subsample(hidden_state))
        query = query.view(batch_size, self.resolution_out**2, self.num_attention_heads, self.key_dim).permute(
            0, 2, 1, 3
        )

        # 计算注意力分数，包括 query 和 key 的乘积，加上注意力偏置和缩放因子
        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(hidden_state.device)
        # 对注意力分数进行 softmax 归一化
        attention = attention.softmax(dim=-1)
        
        # 根据注意力分数和 value 计算加权和，然后重新排列维度以恢复原始形状
        hidden_state = (attention @ value).transpose(1, 2).reshape(batch_size, -1, self.out_dim_projection)
        
        # 应用激活函数和投影层到新的隐藏状态，然后返回结果
        hidden_state = self.projection(self.activation(hidden_state))
        return hidden_state
# 定义一个名为 LevitMLPLayer 的类，用于实现 LeViT 模型中的 MLP 层
class LevitMLPLayer(nn.Module):
    """
    MLP Layer with `2X` expansion in contrast to ViT with `4X`.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 创建一个包含 BN 的 MLP 层，输入维度为 input_dim，隐藏层维度为 hidden_dim
        self.linear_up = MLPLayerWithBN(input_dim, hidden_dim)
        # 激活函数使用 Hardswish
        self.activation = nn.Hardswish()
        # 创建另一个包含 BN 的 MLP 层，输入维度为 hidden_dim，隐藏层维度为 input_dim
        self.linear_down = MLPLayerWithBN(hidden_dim, input_dim)

    def forward(self, hidden_state):
        # 前向传播函数
        # 上层 MLP 层的前向传播
        hidden_state = self.linear_up(hidden_state)
        # 使用 Hardswish 激活函数
        hidden_state = self.activation(hidden_state)
        # 下层 MLP 层的前向传播
        hidden_state = self.linear_down(hidden_state)
        return hidden_state


# 定义一个名为 LevitResidualLayer 的类，用于实现 LeViT 模型中的残差块
class LevitResidualLayer(nn.Module):
    """
    Residual Block for LeViT
    """

    def __init__(self, module, drop_rate):
        super().__init__()
        # module 是要添加残差的模块
        self.module = module
        # drop_rate 是指定的丢弃率
        self.drop_rate = drop_rate

    def forward(self, hidden_state):
        # 前向传播函数
        # 如果处于训练模式并且 drop_rate 大于 0
        if self.training and self.drop_rate > 0:
            # 生成一个与 hidden_state 相同大小的随机张量，用于控制丢弃
            rnd = torch.rand(hidden_state.size(0), 1, 1, device=hidden_state.device)
            # 按照概率 drop_rate 进行丢弃
            rnd = rnd.ge_(self.drop_rate).div(1 - self.drop_rate).detach()
            # 计算残差并加上随机丢弃的模块输出
            hidden_state = hidden_state + self.module(hidden_state) * rnd
            return hidden_state
        else:
            # 计算残差并加上模块输出
            hidden_state = hidden_state + self.module(hidden_state)
            return hidden_state


# 定义一个名为 LevitStage 的类，用于实现 LeViT 模型中的阶段
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
    # 定义Levit网络的类，继承自nn.Module
    class Levit(nn.Module):
        # 初始化函数，配置网络的参数并构建网络层
        def __init__(
            self,
            depths,
            hidden_sizes,
            key_dim,
            num_attention_heads,
            attention_ratio,
            mlp_ratio,
            config,
            resolution_in,
            down_ops,
            idx,
        ):
            # 调用父类的初始化函数
            super().__init__()
            # 初始化网络层列表
            self.layers = []
            # 保存配置参数
            self.config = config
            self.resolution_in = resolution_in
            # 根据传入的深度循环构建LevitResidualLayer，并将每一层添加到网络层列表中
            for _ in range(depths):
                # 创建LevitAttention层并添加到网络层列表中，并传入相应参数
                self.layers.append(
                    LevitResidualLayer(
                        LevitAttention(hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution_in),
                        self.config.drop_path_rate,
                    )
                )
                # 如果mlp_ratio大于0
                if mlp_ratio > 0:
                    # 计算隐藏维度
                    hidden_dim = hidden_sizes * mlp_ratio
                    # 创建LevitMLPLayer层并添加到网络层列表中，并传入相应参数
                    self.layers.append(
                        LevitResidualLayer(LevitMLPLayer(hidden_sizes, hidden_dim), self.config.drop_path_rate)
                    )
    
            # 如果down_ops列表的第一个元素是"Subsample"
            if down_ops[0] == "Subsample":
                # 计算resolution_out的值
                self.resolution_out = (self.resolution_in - 1) // down_ops[5] + 1
                # 创建LevitAttentionSubsample层并添加到网络层列表中，并传入相应参数
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
                # 更新resolution_in的值为resolution_out
                self.resolution_in = self.resolution_out
                # 如果down_ops列表的第四个元素大于0
                if down_ops[4] > 0:
                    # 计算隐藏维度
                    hidden_dim = self.config.hidden_sizes[idx + 1] * down_ops[4]
                    # 创建LevitMLPLayer层并添加到网络层列表中，并传入相应参数
                    self.layers.append(
                        LevitResidualLayer(
                            LevitMLPLayer(self.config.hidden_sizes[idx + 1], hidden_dim), self.config.drop_path_rate
                        )
                    )
    
            # 将网络层列表转换为nn.ModuleList
            self.layers = nn.ModuleList(self.layers)
    
        # 获取当前网络的分辨率
        def get_resolution(self):
            return self.resolution_in
    
        # 前向传播函数，对输入的hidden_state进行操作，并返回最终结果
        def forward(self, hidden_state):
            # 遍历网络层列表，依次对hidden_state进行操作
            for layer in self.layers:
                hidden_state = layer(hidden_state)
            # 返回最终结果
            return hidden_state
class LevitEncoder(nn.Module):
    """
    LeViT Encoder consisting of multiple `LevitStage` stages.
    """

    def __init__(self, config):
        # 初始化LevitEncoder类
        super().__init__()
        self.config = config
        resolution = self.config.image_size // self.config.patch_size
        self.stages = []
        self.config.down_ops.append([""])

        for stage_idx in range(len(config.depths)):
            # 遍历LevitStage的层数
            stage = LevitStage(
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
            resolution = stage.get_resolution()
            self.stages.append(stage)

        # 将stages转换为ModuleList
        self.stages = nn.ModuleList(self.stages)

    def forward(self, hidden_state, output_hidden_states=False, return_dict=True):
        # 初始化隐藏状态列表
        all_hidden_states = () if output_hidden_states else None

        for stage in self.stages:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            hidden_state = stage(hidden_state)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            # 如果不返回字典，则返回特定的元组
            return tuple(v for v in [hidden_state, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=all_hidden_states)


class LevitClassificationLayer(nn.Module):
    """
    LeViT Classification Layer
    """

    def __init__(self, input_dim, output_dim):
        # 初始化LevitClassificationLayer类
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_state):
        # 前向传播函数
        hidden_state = self.batch_norm(hidden_state)
        logits = self.linear(hidden_state)
        return logits


class LevitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LevitConfig
    base_model_prefix = "levit"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 如果是线性层或卷积层
            # 与TF版本稍有不同，使用正态分布初始化权重
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # 如果是BatchNorm层
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
# Levit 模型是一个 PyTorch 的 torch.nn.Module 子类，可以像普通的 PyTorch Module 一样使用，详细的使用方法和行为可以参考 PyTorch 文档。
# 参数：
#     config ([`LevitConfig`]): 包含模型所有参数的模型配置类。用配置文件初始化模型并不会加载模型的权重，只会加载配置信息。加载模型权重可以使用 [`~PreTrainedModel.from_pretrained`] 方法。

LEVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LevitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Levit 模型的输入文档字符串
LEVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。可以使用 [`AutoImageProcessor`] 来获取像素值。具体细节参见 [`LevitImageProcessor.__call__`]。

        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。更多细节请参考返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
"""

# 为 Levit 模型添加文档字符串
@add_start_docstrings(
    "The bare Levit model outputting raw features without any specific head on top.",
    LEVIT_START_DOCSTRING,
)
class LevitModel(LevitPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        super().__init__(config)
        # 存储配置信息
        self.config = config
        # 像素块嵌入层
        self.patch_embeddings = LevitPatchEmbeddings(config)
        # 编码器
        self.encoder = LevitEncoder(config)
        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 如果未提供output_hidden_states参数，则使用配置中的output_hidden_states值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供return_dict参数，则使用配置中的use_return_dict值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供pixel_values参数，则引发数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值传递给patch_embeddings方法，得到嵌入向量
        embeddings = self.patch_embeddings(pixel_values)
        # 将嵌入向量传递给编码器进行编码，得到编码器的输出
        encoder_outputs = self.encoder(
            embeddings,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出中的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 对最后一个隐藏状态进行全局平均池化，将(batch_size, seq_length, hidden_sizes) -> (batch_size, hidden_sizes)
        pooled_output = last_hidden_state.mean(dim=1)

        # 如果不需要返回字典，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果需要返回字典，则返回BaseModelOutputWithPoolingAndNoAttention对象
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
    """
    对Levit模型进行图像分类头部的添加（在特征池化后加上一个线性层），例如用于ImageNet
    """
    # 定义 LevitForImageClassification 类，继承自 LevitPreTrainedModel
    class LevitForImageClassification(LevitPreTrainedModel):
        # 初始化函数
        def __init__(self, config):
            # 调用父类的初始化函数
            super().__init__(config)
            # 将传入的配置参数保存到类属性中
            self.config = config
            # 将配置中的标签数量保存到 num_labels 类属性中
            self.num_labels = config.num_labels
            # 创建 LevitModel 类的实例，并保存到 levit 类属性中
            self.levit = LevitModel(config)

            # 分类器头
            # 如果配置参数中标签数量大于0，使用 LevitClassificationLayer 类创建一个分类器头部，并保存到 classifier 类属性中，否则使用 torch.nn.Identity() 创建一个空的头部
            self.classifier = (
                LevitClassificationLayer(config.hidden_sizes[-1], config.num_labels)
                if config.num_labels > 0
                else torch.nn.Identity()
            )

            # 初始化权重并应用最终处理
            self.post_init()

        # 前向传播函数
        def forward(
            self,
            pixel_values: torch.FloatTensor = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果没有指定 return_dict，则根据配置决定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取分类器的输出
        outputs = self.levit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 获取序列输出并计算平均值
        sequence_output = outputs[0]
        sequence_output = sequence_output.mean(1)
        # 通过分类器获取 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为None
        loss = None
        # 如果有标签
        if labels is not None:
            # 如果问题类型未指定，则根据标签类型自动确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
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
        # 如果不返回字典，则将输出打包成元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回图像分类器输出
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
# 导入必要的库和模块
@add_start_docstrings(
    """
    LeViT 模型的图像分类头部（在最终隐藏状态的顶部有一个线性层和在蒸馏令牌的最终隐藏状态的顶部有一个线性层），
    例如用于 ImageNet。.. warning::
           此模型仅支持推理。尚不支持使用蒸馏进行微调（即使用教师）。
    """,
    LEVIT_START_DOCSTRING,
)
# 定义一个带有教师的图像分类器模型
class LevitForImageClassificationWithTeacher(LevitPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 设置标签数目
        self.num_labels = config.num_labels
        # 初始化 LeViT 模型
        self.levit = LevitModel(config)

        # 分类器头部
        self.classifier = (
            LevitClassificationLayer(config.hidden_sizes[-1], config.num_labels)
            if config.num_labels > 0
            else torch.nn.Identity()
        )
        # 教师分类器头部
        self.classifier_distill = (
            LevitClassificationLayer(config.hidden_sizes[-1], config.num_labels)
            if config.num_labels > 0
            else torch.nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=LevitForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 前向传播方法
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LevitForImageClassificationWithTeacherOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取 LeViT 模型输出
        outputs = self.levit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        sequence_output = outputs[0]
        # 对序列输出进行平均池化
        sequence_output = sequence_output.mean(1)
        # 分类器和教师分类器的 Logits
        cls_logits, distill_logits = self.classifier(sequence_output), self.classifier_distill(sequence_output)
        # 将分类器和教师分类器的 Logits 平均
        logits = (cls_logits + distill_logits) / 2

        # 如果不返回字典
        if not return_dict:
            output = (logits, cls_logits, distill_logits) + outputs[2:]
            return output

        # 返回结果字典
        return LevitForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distill_logits,
            hidden_states=outputs.hidden_states,
        )
```