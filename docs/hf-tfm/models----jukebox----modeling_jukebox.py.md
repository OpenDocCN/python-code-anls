# `.\models\jukebox\modeling_jukebox.py`

```
# coding=utf-8
# Copyright 2022 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""PyTorch Jukebox model."""

import math  # 导入数学库
import os  # 导入操作系统相关功能
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch中的函数模块
from torch import nn  # 导入PyTorch的神经网络模块
from torch.nn import LayerNorm as FusedLayerNorm  # 导入PyTorch的归一化层模块

from ...activations import ACT2FN  # 导入自定义的激活函数
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import add_start_docstrings, logging  # 导入工具函数和日志模块
from ...utils.logging import tqdm  # 导入进度条显示模块
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig  # 导入配置文件

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/jukebox-1b-lyrics",
    "openai/jukebox-5b-lyrics",
    # See all Jukebox models at https://huggingface.co/models?filter=jukebox
]


def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits (`torch.Tensor`):
            logits distribution shape (vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            When `top_k >0` keep only top key tokens with highest probability (top-k filtering).
        top_p (`int`, *optional*, defaults to 0):
            When `top_p>0.0` keep the top tokens with cumulative probability >= `top_p` (nucleus filtering).
    """
    logits = logits.clone()  # 复制logits张量，确保不改变原始数据
    top_k = min(top_k, logits.size(-1))  # 安全检查，确保top_k不超过logits的最后一个维度大小

    if top_k > 0:
        # 移除概率小于top-k中的最后一个概率的所有标记
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]
        logits[indices_to_remove] = filter_value  # 将这些标记的概率值设置为filter_value
    # 如果给定的 top_p 阈值大于 0，则执行以下操作
    if top_p > 0.0:
        # 对 logits 进行降序排序，并返回排序后的 logits 和对应的索引
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # 计算排序后的 logits 的累积 softmax 概率
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 根据累积概率超过阈值的情况，标记需要移除的索引
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 将超过阈值的索引右移一位，以保留第一个超过阈值的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 根据排序后的索引，创建一个布尔张量表示需要移除的位置
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        # 将需要移除的位置对应的 logits 设置为指定的 filter_value
        logits[indices_to_remove] = filter_value
    
    # 返回经过处理后的 logits
    return logits
# 根据给定的参数，从完整的歌词标记中提取相关的标记。
# 返回的标记数为 `max_n_lyric_tokens`。如果提供的标记序列较小，将进行填充；否则，只返回从中点向左右偏移 `max_n_lyric_tokens//2` 的标记。
# 这个过程专注于时间上最相关的标记。

def get_relevant_lyric_tokens(full_tokens, max_n_lyric_tokens, total_length, offset, duration):
    """
    Extract only the relevant tokens based on the character position. A total of `max_n_lyric_tokens` tokens will be
    returned. If the provided token sequence is smaller, it will be padded, otherwise, only characters ranging from the
    midpoint - `max_n_lyric_tokens//2` to the midpoint + `max_n_lyric_tokens//2` will be returned. This *focuses* on
    the most relevant tokens (in time) for the sequence.

    Args:
        full_tokens (`List[int]`):
            List containing the token ids of the entire lyrics.
        max_n_lyric_tokens (`int`):
            Maximum number of lyric tokens to return.
        total_length (`int`):
            Total expected length of the music (not all of it is generated, see duration), in samples.
        offset (`int`):
            Starting sample in the music. If the offset is greater than 0, the lyrics will be shifted take that into
            account
        duration (`int`):
            Expected duration of the generated music, in samples. The duration has to be smaller than the total length,
            which represent the overall length of the signal,
    """
    full_tokens = full_tokens[0]  # 取出列表中的第一个元素（预计是整数列表）
    if len(full_tokens) < max_n_lyric_tokens:
        # 如果标记序列长度小于 `max_n_lyric_tokens`，进行填充
        tokens = torch.cat(
            [torch.zeros(max_n_lyric_tokens - len(full_tokens), dtype=torch.long).to(full_tokens.device), full_tokens]
        )
        indices = [-1] * (max_n_lyric_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
    else:
        # 计算中点位置
        midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
        # 限制中点位置在有效范围内
        midpoint = min(max(midpoint, max_n_lyric_tokens // 2), len(full_tokens) - max_n_lyric_tokens // 2)
        # 提取中心周围的标记
        tokens = full_tokens[midpoint - max_n_lyric_tokens // 2 : midpoint + max_n_lyric_tokens // 2]
        indices = list(range(midpoint - max_n_lyric_tokens // 2, midpoint + max_n_lyric_tokens // 2))
    return tokens.unsqueeze(dim=0), indices


# 将总长度 `total_length` 拆分为大小为 `n_ctx` 的窗口，每隔 `hop_length` 个样本分隔开
def get_starts(total_length, n_ctx, hop_length):
    starts = []
    for start in range(0, total_length - n_ctx + hop_length, hop_length):
        if start + n_ctx >= total_length:
            # 最后一个窗口可能会较小，我们设定为 `n_ctx` 以最大化上下文
            start = total_length - n_ctx
        starts.append(start)
    return starts


# 获取音乐标记、标签、先验值和配置信息，返回对齐信息
def get_alignment(music_tokens, labels, prior, config):
    level = prior.levels - 1  # 使用的顶层
    n_ctx = prior.n_ctx
    tokens = music_tokens[level]
    batch_size, total_length = tokens.shape[0], tokens.shape[1]
    if total_length < n_ctx:
        # 如果总长度小于 `n_ctx`，进行填充
        padding_length = n_ctx - total_length
        tokens = torch.cat(
            [tokens, torch.zeros(batch_size, n_ctx - total_length, dtype=tokens.dtype, device=tokens.device)], dim=1
        )
        total_length = tokens.shape[1]
    else:
        padding_length = 0

    # 计算 `hop_length`，这是根据配置的音频片段长度的分数决定的
    hop_length = int(config.hop_fraction[-level - 1] * prior.n_ctx)
    # 从配置中获取对齐头部和对齐层信息，并选择第一个
    alignment_head, alignment_layer = config.prior_alignment_head[0], config.prior_alignment_layer[0]
    # 创建包含alignment_layer的集合
    attn_layers = {alignment_layer}
    # 创建空的对齐跳数字典
    alignment_hops = {}
    # 创建空的索引跳数字典
    indices_hops = {}
    # 对于每个从get_starts生成的起始位置进行迭代，显示"Computing lyric to music alignment"进度条
    for start in tqdm(get_starts(total_length, n_ctx, hop_length), desc="Computing lyric to music alignment "):
        end = start + n_ctx
        # 获取metadata和indices_hop，从prior获取标签，开始，采样长度，并获取indices
        metadata, indices_hop = prior.get_metadata(labels, start, config.sample_length, get_indices=True, offset=0)
        # 将tokens分块为batch_size大小的张量块
        tokens_bs = torch.chunk(tokens, batch_size, dim=0)
        # 将metadata分块为batch_size大小的张量块
        metadata_bs = torch.chunk(metadata, batch_size, dim=0)
        # 创建空列表w_hops
        w_hops = []
        # 对于tokens_bs和metadata_bs中的每一对，执行以下操作
        for tokens_i, metadata_i in zip(tokens_bs, metadata_bs):
            # 调用prior的forward_tokens函数，传递tokens_i[:, start:end]，空列表，metadata_i参数，获取attn_layers的注意力权重
            w_hop = prior.forward_tokens(tokens_i[:, start:end], [], metadata_i, get_attn_weights=attn_layers)
            # 将第一个返回的注意力权重的alignment_head列添加到w_hops中
            w_hops.append(w_hop[0][:, alignment_head])
            # 删除w_hop变量以释放内存
            del w_hop
        # 将w_hops中的张量连接成一个张量weights
        weights = torch.cat(w_hops, dim=0)
        # 删除w_hops以释放内存
        del w_hops
        # 将weights转换为float类型，移动到CPU上，并转换为numpy数组，存储在alignment_hop中
        alignment_hop = weights.float().cpu().numpy()
        # 删除weights以释放内存
        del weights

        # alignment_hop的形状为(bs, n_ctx, nb_relevant_lyric_tokens)
        # indices_hop是长度为bs的列表，每个条目长度为hps.nb_relevant_lyric_tokens
        # 将indices_hop和alignment_hop存储在对应的start位置
        indices_hops[start] = indices_hop
        alignment_hops[start] = alignment_hop

    # 将每个跳的attn组合成全范围的attn
    # 使用indices将它们放置到相应源tokens的正确位置
    alignments = []
    for item in range(batch_size):
        # 注意每个item具有不同长度的歌词
        full_tokens = labels[0, 3:]
        # 创建全零数组alignment，形状为(total_length, len(full_tokens) + 1)
        alignment = np.zeros((total_length, len(full_tokens) + 1))
        # 对于反向遍历的每个start，执行以下操作
        for start in reversed(get_starts(total_length, n_ctx, hop_length)):
            end = start + n_ctx
            # 获取alignment_hops中的alignment_hop[item]和indices_hops中的indices[item]
            alignment_hop = alignment_hops[start][item]
            indices = indices_hops[start][item]
            # 将alignment_hop放置到对应的indices位置
            alignment[start:end, indices] = alignment_hop
        # 去除token填充和最后一个歌词索引，截取alignment数组
        alignment = alignment[: total_length - padding_length, :-1]
        # 将alignment添加到alignments列表中
        alignments.append(alignment)
    # 返回alignments列表作为函数结果
    return alignments
# 定义一个函数，用于保存临时音频数据
def save_temp_audio(fname, lvl, metas, aud):
    # 将音频数据限制在[-1, 1]范围内，并转换为numpy数组
    aud = torch.clamp(aud, -1, 1).cpu().numpy()
    # 遍历音频数据的每一个片段
    for i in list(range(aud.shape[0])):
        # 如果提供了元数据信息
        if metas is not None:
            # 获取当前片段的艺术家、流派和歌词信息
            artists, genres, lyrics = list(metas)[i].values()
            # 构建保存路径，包含文件夹名、级别、艺术家、流派、歌词前5个字符和索引信息
            path = f"{fname}/lvl_{lvl}-{artists}-{genres}-{lyrics[:5]}-{i}"
            # 保存numpy数组为.npy文件
            np.save(path, aud[i])
        else:
            # 如果未提供元数据信息，直接保存为.npy文件，文件名包含级别和索引信息
            np.save(f"{fname}/lvl_{lvl}-sample-{i}", aud[i])


# 定义一个函数，根据不同的掩码类型生成掩码张量
def get_mask(mask, query_length, key_value_length, blocks, spread, device, sample, sample_t):
    # 如果掩码为None或者查询长度为1，则返回None，表示无需掩码
    if mask is None or query_length == 1:
        return None
    # 计算偏移量，用于掩码生成的起始位置
    offset = sample_t - query_length if sample else max(key_value_length - query_length, 0)
    # 根据不同的掩码类型生成相应的掩码张量
    if mask == "autoregressive":
        # 自回归掩码：下三角形式的矩阵，掩盖查询和键值之间的依赖关系
        mask = torch.ones(query_length, key_value_length, device=device).tril(offset)
    elif mask == "summary":
        # 摘要掩码：用于对输入进行汇总处理时使用的掩码
        mask = torch.ones(query_length, query_length, device=device).tril()
        mask = mask.view(query_length, blocks, query_length // blocks)[:, :-1, -key_value_length // blocks :]
        mask = (
            torch.nn.functional.pad(
                mask,
                (0, 0, 1, 0),
                value=1,
            )
            .contiguous()
            .view(query_length, key_value_length)
        )
    elif mask == "prime":
        # 主掩码：一种特定形式的下三角掩码
        mask = torch.ones(query_length, key_value_length, device=device).tril(offset)
    return mask.view(1, 1, query_length, key_value_length)


# 定义一个神经网络模型类，实现基于卷积的Jukebox模型
class JukeboxConv1D(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        # 初始化权重和偏置参数
        weight = torch.empty(input_width, output_width)
        bias = torch.zeros(output_width)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, hidden_states):
        # 计算输出大小
        size_out = (*hidden_states.size()[:-1], self.output_width)
        # 执行卷积操作，并加上偏置项
        hidden_states = torch.addmm(
            self.bias.type_as(hidden_states),
            hidden_states.view(-1, hidden_states.size(-1)),
            self.weight.type_as(hidden_states),
        )
        # 重新调整输出形状并返回结果
        hidden_states = hidden_states.view(*size_out)
        return hidden_states


# 定义一个神经网络模型类，实现基于残差卷积的Jukebox模型块
class JukeboxResConv1DBlock(nn.Module):
    def __init__(self, config, conv_width, depth=1, res_scale=1.0):
        super().__init__()
        # 根据配置参数计算隐藏层维度、膨胀率和填充大小
        hidden_dim = config.res_convolution_multiplier * conv_width
        dilation = config.res_dilation_growth_rate**depth
        padding = dilation

        self.res_scale = res_scale
        self.activation = nn.ReLU()
        # 定义第一个卷积层和第二个卷积层
        self.conv1d_1 = nn.Conv1d(conv_width, hidden_dim, 3, 1, padding, dilation)
        self.conv1d_2 = nn.Conv1d(hidden_dim, conv_width, 1, 1, 0)
    # 定义一个前向传播方法，用于神经网络模型中
    def forward(self, hidden_states):
        # 将输入的隐藏状态保存为残差项
        residuals = hidden_states
        # 对隐藏状态应用激活函数
        hidden_states = self.activation(hidden_states)
        # 应用第一个一维卷积层
        hidden_states = self.conv1d_1(hidden_states)
        # 再次应用激活函数
        hidden_states = self.activation(hidden_states)
        # 应用第二个一维卷积层
        hidden_states = self.conv1d_2(hidden_states)
        # 返回残差项与带有残差缩放系数的隐藏状态的和
        return residuals + self.res_scale * hidden_states
# 定义 JukeboxResnet1D 类，继承自 nn.Module 类，实现了一维卷积神经网络的残差结构
class JukeboxResnet1D(nn.Module):
    # 初始化函数，接受配置 config、卷积宽度 conv_width、深度 n_depth、是否反向扩张 reverse_dilation 参数
    def __init__(self, config, conv_width, n_depth, reverse_dilation=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置设置残差扩张周期
        self.dilation_cycle = config.res_dilation_cycle
        # 如果配置了卷积残差缩放，则根据深度设置缩放系数
        res_scale = 1.0 if not config.conv_res_scale else 1.0 / math.sqrt(n_depth)

        # 创建空的块列表
        blocks = []
        # 根据深度循环创建残差卷积块
        for depth in range(n_depth):
            # 如果设置了扩张周期，则取当前深度对扩张周期取模得到块深度
            block_depth = depth if self.dilation_cycle is None else depth % self.dilation_cycle
            # 创建并添加 JukeboxResConv1DBlock 到块列表中
            blocks.append(JukeboxResConv1DBlock(config, conv_width, block_depth, res_scale))

        # 如果设置了反向扩张，则对块列表进行反向排序
        if reverse_dilation:
            blocks = blocks[::-1]
        # 将块列表转换为 nn.ModuleList 类型的模块列表并赋值给实例变量
        self.resnet_block = nn.ModuleList(blocks)

    # 前向传播函数，接受输入 hidden_states
    def forward(self, hidden_states):
        # 遍历每个残差卷积块，依次对输入进行处理
        for block in self.resnet_block:
            hidden_states = block(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义 JukeboxEncoderConvBlock 类，继承自 nn.Module 类，实现了编码器的卷积块结构
class JukeboxEncoderConvBlock(nn.Module):
    # 初始化函数，接受配置 config、嵌入维度 embed_dim、隐藏维度 hidden_dim、深度 depth、down_t、stride_t 参数
    def __init__(self, config, embed_dim, hidden_dim, depth, down_t, stride_t):
        # 调用父类的初始化方法
        super().__init__()
        # 创建空的块列表
        blocks = []
        # 计算滤波器大小 filter_t 和填充大小 pad_t
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        # 如果 down_t 大于 0，则循环添加卷积层和残差卷积块到块列表中
        if down_t > 0:
            for i in range(down_t):
                # 添加 1 维卷积层到块列表中
                blocks.append(nn.Conv1d(embed_dim if i == 0 else hidden_dim, hidden_dim, filter_t, stride_t, pad_t))
                # 添加 JukeboxResnet1D 模块到块列表中
                blocks.append(JukeboxResnet1D(config, hidden_dim, depth))

        # 创建输出投影层
        self.proj_out = nn.Conv1d(hidden_dim, config.embed_dim, 3, 1, 1)
        # 将块列表转换为 nn.ModuleList 类型的模块列表并赋值给实例变量
        self.downsample_block = nn.ModuleList(blocks)

    # 前向传播函数，接受输入 hidden_states
    def forward(self, hidden_states):
        # 遍历每个块，依次对输入进行处理
        for block in self.downsample_block:
            hidden_states = block(hidden_states)
        # 将处理后的 hidden_states 经过投影层处理后返回
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


# 定义 JukeboxEncoder 类，继承自 nn.Module 类，实现了 Jukebox 编码器结构
class JukeboxEncoder(nn.Module):
    # 初始化函数，接受配置 config、宽度 width、深度 depth、层级 levels、downs_t、strides_t 参数
    def __init__(self, config, width, depth, levels, downs_t, strides_t):
        # 调用父类的初始化方法
        super().__init__()
        # 设置层级数
        self.levels = levels
        # 创建模块列表 level_blocks
        self.level_blocks = nn.ModuleList()

        # 使用 zip 函数迭代 levels、downs_t 和 strides_t，并根据迭代结果生成 JukeboxEncoderConvBlock 模块并添加到 level_blocks 中
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for i, down_t, stride_t in iterator:
            self.level_blocks.append(
                JukeboxEncoderConvBlock(
                    config, config.conv_input_shape if i == 0 else config.embed_dim, width, depth, down_t, stride_t
                )
            )

    # 前向传播函数，接受输入 hidden_states
    def forward(self, hidden_states):
        # 创建空列表 all_hidden_states 用于存储所有层级的隐藏状态
        all_hidden_states = []

        # 遍历每个层级
        for level in range(self.levels):
            # 获取当前层级的 JukeboxEncoderConvBlock 模块
            level_block = self.level_blocks[level]
            # 对输入 hidden_states 应用当前层级的模块处理
            hidden_states = level_block(hidden_states)
            # 将处理后的隐藏状态添加到 all_hidden_states 列表中
            all_hidden_states.append(hidden_states)

        # 返回所有层级的隐藏状态列表
        return all_hidden_states


# 定义 JukeboxDecoderConvBock 类，继承自 nn.Module 类，未完成的类定义
class JukeboxDecoderConvBock(nn.Module):
    # 初始化函数，用于初始化类实例
    def __init__(self, config, embed_dim, hidden_dim, depth, down_t, stride_t, reverse_dilation=True):
        # 设置类的属性 embed_dim 和 hidden_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # 调用父类的初始化方法
        super().__init__()
        # 初始化空列表用于存储模块
        blocks = []
        # 如果 down_t 大于 0，执行以下操作
        if down_t > 0:
            # 计算滤波器长度和填充长度
            filter_t = stride_t * 2
            pad_t = stride_t // 2
            # 创建输入投影层，将 embed_dim 维度的输入转换为 hidden_dim 维度
            self.proj_in = nn.Conv1d(embed_dim, hidden_dim, 3, 1, 1)
            # 循环 down_t 次，添加 JukeboxResnet1D 模块和反卷积层到 blocks 列表中
            for i in range(down_t):
                blocks.append(JukeboxResnet1D(config, hidden_dim, depth, reverse_dilation))
                blocks.append(
                    nn.ConvTranspose1d(
                        hidden_dim, hidden_dim if i < down_t - 1 else embed_dim, filter_t, stride_t, pad_t
                    )
                )
        # 将 blocks 列表作为 ModuleList 赋给实例的 upsample_block 属性
        self.upsample_block = nn.ModuleList(blocks)

    # 前向传播函数，处理输入的隐藏状态
    def forward(self, hidden_states):
        # 将输入的隐藏状态通过投影层 proj_in 进行转换
        hidden_states = self.proj_in(hidden_states)
        # 对 upsample_block 中的每个模块进行前向传播
        for block in self.upsample_block:
            hidden_states = block(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
class JukeboxDecoder(nn.Module):
    # 定义JukeboxDecoder类，继承自nn.Module
    def __init__(self, config, hidden_dim, depth, levels, downs_t, strides_t):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList()
        for level, down_t, stride_t in zip(list(range(self.levels)), downs_t, strides_t):
            self.level_blocks.append(
                JukeboxDecoderConvBock(config, config.embed_dim, hidden_dim, depth, down_t, stride_t)
            )

        self.out = nn.Conv1d(config.embed_dim, config.conv_input_shape, 3, 1, 1)
        # 初始化各个网络层

    def forward(self, hidden_states, all_levels=True):
        hidden_state = hidden_states[-1]

        # 32, 64 ...
        for level in reversed(range(self.levels)):
            level_block = self.level_blocks[level]
            hidden_state = level_block(hidden_state)

            if level != 0 and all_levels:
                hidden_state = hidden_state + hidden_states[level - 1]
        # 在不同的层级进行前向传播，并根据需要进行级联

        hidden_state = self.out(hidden_state)
        return hidden_state
        # 返回隐藏状态


class JukeboxBottleneckBlock(nn.Module):
    # 定义JukeboxBottleneckBlock类，继承自nn.Module
    def __init__(self, config: JukeboxVQVAEConfig):
        super().__init__()
        self.nb_discrete_codes = config.nb_discrete_codes
        self.codebook_width = config.embed_dim
        self.mu = config.lmu
        self.threshold = 1.0
        self.init = False
        self.codebook_sum = None
        self.codebook_elem = None
        self.register_buffer("codebook", torch.zeros(self.nb_discrete_codes, self.codebook_width))
        # 初始化相关变量，并注册缓冲区

    def _tile(self, hidden_states):
        dim, embed_width = hidden_states.shape
        if dim < self.nb_discrete_codes:
            n_repeats = (self.nb_discrete_codes + dim - 1) // dim
            std = 0.01 / np.sqrt(embed_width)
            hidden_states = hidden_states.repeat(n_repeats, 1)
            hidden_states = hidden_states + torch.randn_like(hidden_states) * std
        return hidden_states
        # 定义辅助函数_tile，用于重复和扩展隐藏状态

    def init_codebook(self, hidden_states):
        nb_discrete_codes = self.nb_discrete_codes
        self.init = True
        codes = self._tile(hidden_states)
        self.codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]
        self.codebook_sum = self.codebook
        self.codebook_elem = torch.ones(nb_discrete_codes, device=self.codebook.device)
        # 初始化码书信息
    # 更新代码本函数，更新代码簿中的中心点
    def update_codebook(self, hidden_states, latent_states):
        # 从对象属性中获取参数
        mu, codebook_width, nb_discrete_codes = self.mu, self.codebook_width, self.nb_discrete_codes
        # 禁止梯度计算
        with torch.no_grad():
            # 计算新的中心点
            # 将离散状态转换为独热编码
            latent_states_onehot = torch.zeros(nb_discrete_codes, hidden_states.shape[0], device=hidden_states.device)
            latent_states_onehot.scatter_(0, latent_states.view(1, hidden_states.shape[0]), 1)

            # 计算每个簇的加权和
            _codebook_sum = torch.matmul(latent_states_onehot, hidden_states)
            # 计算每个簇的元素数量
            _codebook_elem = latent_states_onehot.sum(dim=-1)  # nb_discrete_codes
            # 复制隐藏状态以扩展簇的数量
            codes = self._tile(hidden_states)
            # 随机选取一些代码本的样本
            _random_codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]

            # 更新中心点
            old_codebook = self.codebook
            # 更新加权和
            self.codebook_sum = mu * self.codebook_sum + (1.0 - mu) * _codebook_sum
            # 更新簇的元素数量
            self.codebook_elem = mu * self.codebook_elem + (1.0 - mu) * _codebook_elem  # nb_discrete_codes
            # 计算每个簇的使用情况
            usage = (self.codebook_elem.view(nb_discrete_codes, 1) >= self.threshold).float()

            # 归一化簇的中心点
            norm_code = self.codebook_sum.view(nb_discrete_codes, codebook_width) / self.codebook_elem.view(
                nb_discrete_codes, 1
            )
            # 更新代码本
            self.codebook = usage * (norm_code) + (1 - usage) * _random_codebook
            # 计算每个簇的概率
            _codebook_prob = _codebook_elem / torch.sum(_codebook_elem)  # prob of each bin
            # 计算熵，用于衡量多样性
            entropy = -torch.sum(_codebook_prob * torch.log(_codebook_prob + 1e-8))  # entropy ie how diverse
            # 计算当前使用的簇的数量
            used_curr = (_codebook_elem >= self.threshold).sum()
            # 计算簇的使用情况
            usage = torch.sum(usage)
            # 计算 K-L 散度
            dk = torch.norm(self.codebook - old_codebook) / np.sqrt(np.prod(old_codebook.shape))
        # 返回更新结果
        return {"entropy": entropy, "used_curr": used_curr, "usage": usage, "dk": dk}

    # 预处理函数，用于规范化隐藏状态
    def preprocess(self, hidden_states):
        # 调整张量形状以便后续处理
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # 如果隐藏状态的维度等于代码本的宽度
        if hidden_states.shape[-1] == self.codebook_width:
            # 计算预规范化值
            prenorm = torch.norm(hidden_states - torch.mean(hidden_states)) / np.sqrt(np.prod(hidden_states.shape))
        # 如果隐藏状态的维度是代码本宽度的两倍
        elif hidden_states.shape[-1] == 2 * self.codebook_width:
            # 分离隐藏状态的两部分
            x1, x2 = hidden_states[..., : self.codebook_width], hidden_states[..., self.codebook_width :]
            # 分别计算两部分的预规范化值，并相加
            prenorm = (torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (
                torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape))
            )
            # 合并隐藏状态的两部分
            hidden_states = x1 + x2

        # 返回预处理后的隐藏状态及其规范化值
        return hidden_states, prenorm
    def postprocess(self, latent_states, dequantised_states, x_shape):
        # 获取输入数据的批次大小和时间步数
        batch_size, time = x_shape
        # 重新组织 dequantised_states 的形状，使其变为 (batch_size, -1, time)
        dequantised_states = dequantised_states.view(batch_size, time, -1).permute(0, 2, 1).contiguous()
        # 重新组织 latent_states 的形状，使其变为 (batch_size, time)
        latent_states = latent_states.view(batch_size, time)
        return latent_states, dequantised_states

    def quantise(self, latent_states):
        # 计算 latent_states 与 codebook 中的距离
        codebook_weights = self.codebook.t()
        distance = (
            torch.sum(latent_states**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(latent_states, codebook_weights)
            + torch.sum(codebook_weights**2, dim=0, keepdim=True)
        )  # 形状为 (batch_size * latent_states , codebook_weights)
        # 找到每个 latent_state 最接近的 codebook 中的索引 music_tokens
        min_distance, music_tokens = torch.min(distance, dim=-1)
        # 计算平均最小距离
        fit = torch.mean(min_distance)
        return music_tokens, fit

    def dequantise(self, music_tokens):
        # 使用 music_tokens 从 codebook 中获取对应的 dequantised_states
        dequantised_states = F.embedding(music_tokens, self.codebook)
        return dequantised_states

    def encode(self, latent_states):
        samples, _, seq_len = latent_states.shape

        # 数据预处理
        latent_states, _ = self.preprocess(latent_states)

        # 量化过程
        music_tokens, _ = self.quantise(latent_states)

        # 后处理
        music_tokens = music_tokens.view(samples, seq_len)
        return music_tokens

    def decode(self, music_tokens):
        samples, seq_len = music_tokens.shape

        # 反量化过程
        dequantised_states = self.dequantise(music_tokens)

        # 后处理
        dequantised_states = (
            dequantised_states.view(samples, seq_len, self.codebook_width).permute(0, 2, 1).contiguous()
        )
        return dequantised_states

    def forward(self, hidden_states, update_codebook=True):
        samples, _, seq_len = hidden_states.shape

        # 数据预处理
        hidden_states, prenorm = self.preprocess(hidden_states)

        # 如果需要更新 codebook 并且未初始化，则进行初始化
        if update_codebook and not self.init:
            self.init_codebook(hidden_states)

        # 通过编码和解码过程量化和反量化
        music_tokens, fit = self.quantise(hidden_states)
        dequantised_states = self.dequantise(music_tokens)

        # 如果需要更新 codebook，则更新相关指标
        if update_codebook:
            update_metrics = self.update_codebook(hidden_states, music_tokens)
        else:
            update_metrics = {}

        # 计算损失
        commit_loss = torch.norm(dequantised_states.detach() - hidden_states) ** 2 / np.prod(hidden_states.shape)

        # 通过传递增强数据流
        dequantised_states = hidden_states + (dequantised_states - hidden_states).detach()

        # 后处理
        music_tokens, dequantised_states = self.postprocess(music_tokens, dequantised_states, (samples, seq_len))
        return music_tokens, dequantised_states, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)
# 导入 PyTorch 的 nn 模块
import torch.nn as nn

# 定义一个名为 JukeboxBottleneck 的类，继承自 nn.Module
class JukeboxBottleneck(nn.Module):
    
    # 初始化方法，接受 config 和 levels 参数
    def __init__(self, config, levels):
        super().__init__()
        self.levels = levels  # 设置 levels 属性
        self.level_blocks = nn.ModuleList()  # 初始化一个 nn.ModuleList 用于存储每个 level 的 block
        
        # 遍历 levels 创建 JukeboxBottleneckBlock，并添加到 level_blocks 中
        for level in range(self.levels):
            self.level_blocks.append(JukeboxBottleneckBlock(config))

    # 编码方法，接受 raw_audio 参数
    def encode(self, raw_audio):
        # 使用列表推导式对每个 level_block 和对应的 hidden_states 进行编码
        music_tokens = [
            level_block.encode(hidden_states) for (level_block, hidden_states) in zip(self.level_blocks, raw_audio)
        ]
        return music_tokens  # 返回编码后的音乐 tokens

    # 解码方法，接受 music_tokens、start_level 和 end_level 参数
    def decode(self, music_tokens, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels  # 如果未指定 end_level，默认为 levels
        
        # 使用列表推导式对每个 level_block 和对应的 music_tokens 进行解码
        quantised_audio = [
            level_block.decode(z) for (level_block, z) in zip(self.level_blocks[start_level:end_level], music_tokens)
        ]
        return quantised_audio  # 返回量化后的音频数据

    # 前向传播方法，接受 input_audio 参数
    def forward(self, input_audio):
        music_tokens, quantised_states, commit_losses, metrics = [], [], [], []
        
        # 遍历每个 level
        for level in range(self.levels):
            level_block = self.level_blocks[-level - 1]  # 获取当前 level 的 block
            hidden_states = input_audio[level]  # 获取对应的输入音频的隐藏状态
            
            # 调用 level_block 进行处理，获取返回值
            sampled_tokens, quantised_state, commit_loss, metric = level_block(
                hidden_states, update_codebook=self.training
            )
            
            music_tokens.append(sampled_tokens)  # 将 sampled_tokens 添加到 music_tokens 列表中
            
            if not self.training:
                # 在非训练模式下，确保编码器权重不会从直通估计中更改
                quantised_state = quantised_state.detach()
            
            quantised_states.append(quantised_state)  # 将 quantised_state 添加到 quantised_states 列表中
            commit_losses.append(commit_loss)  # 将 commit_loss 添加到 commit_losses 列表中
            
            if self.training:
                metrics.append(metric)  # 在训练模式下，将 metric 添加到 metrics 列表中
        
        # 返回 music_tokens、quantised_states、commit_losses 和 metrics
        return music_tokens, quantised_states, commit_losses, metrics

# 设置 JUKEBOX_START_DOCSTRING 变量，包含模型的一些基本文档信息
JUKEBOX_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config (`JukeboxConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 使用 @add_start_docstrings 装饰器添加额外的文档信息到 JukeboxVQVAE 类
@add_start_docstrings(
    """The Hierarchical VQ-VAE model used in Jukebox. This model follows the Hierarchical VQVAE paper from [Will Williams, Sam
Ringer, Tom Ash, John Hughes, David MacLeod, Jamie Dougherty](https://arxiv.org/abs/2002.08111).

    """,
    JUKEBOX_START_DOCSTRING,
)
# 定义 JukeboxVQVAE 类，继承自 PreTrainedModel
class JukeboxVQVAE(PreTrainedModel):
    config_class = JukeboxVQVAEConfig  # 设置 config_class 属性
    # 设置基础模型前缀为 "vqvae"
    base_model_prefix = "vqvae"

    # 初始化权重的函数，用于初始化模块的权重
    def _init_weights(self, module):
        # 如果模块是 nn.Embedding 类型，例如 embed_tokens
        if isinstance(module, nn.Embedding):
            # 初始化权重为正态分布，均值为 0，标准差为 0.02 乘以配置参数中的初始化比例
            module.weight.data.normal_(mean=0.0, std=0.02 * self.config.init_scale)
        # 如果模块是 JukeboxConv1D 类型
        elif isinstance(module, JukeboxConv1D):
            # 根据配置参数决定是否将权重初始化为零，否则初始化为正态分布
            if self.config.zero_out:
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=0.02 * self.config.init_scale)
        # 如果模块是 JukeboxResConv1DBlock 类型，并且配置参数中指定了 zero_out 为 True
        elif isinstance(module, JukeboxResConv1DBlock) and self.config.zero_out:
            # 将第二个卷积层的权重和偏置初始化为零
            module.conv1d_2.weight.data.zero_()
            module.conv1d_2.bias.data.zero_()
        # 如果模块是 nn.LayerNorm 类型
        if isinstance(module, nn.LayerNorm):
            # 将偏置初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全 1
            module.weight.data.fill_(1.0)
        # 如果模块是 nn.Linear 类型，并且有偏置项
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置项初始化为零
            module.bias.data.zero_()

    # 初始化函数，接受一个 JukeboxVQVAEConfig 类型的配置参数
    def __init__(self, config: JukeboxVQVAEConfig):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 获取配置参数中的 res_downs_t 和 res_strides_t
        downs_t = config.res_downs_t
        strides_t = config.res_strides_t
        # 如果配置参数中没有指定 sample_length
        if not config.sample_length:
            # 计算 downsamples 数组，每个元素为 stride**down 的结果
            downsamples = [stride**down for stride, down in zip(strides_t, downs_t)]
            # 计算 top_raw_to_tokens，即 downsamples 的乘积
            top_raw_to_tokens = np.prod(downsamples)
            # 根据采样率和 top_raw_to_tokens 计算 sample_length
            config.sample_length = (
                config.sample_length_in_seconds * config.sampling_rate // top_raw_to_tokens
            ) * top_raw_to_tokens
            # 将 sample_length 转换为整数类型
            config.sample_length = config.sample_length.astype(int)

        # 设置一些模型参数，从配置中获取
        self.nb_discrete_codes = config.nb_discrete_codes
        self.commit = config.commit
        self.sample_length = config.sample_length

        # 计算 downsamples 数组和 hop_lengths 数组
        self.downsamples = [stride**down for stride, down in zip(strides_t, downs_t)]
        self.hop_lengths = np.cumprod(self.downsamples)
        self.levels = levels = config.levels
        # 计算 music_tokens_shapes 数组
        self.music_tokens_shapes = [
            (int(self.sample_length // self.hop_lengths[-level - 1])) for level in range(levels)
        ]

        # 设置 multipliers 数组，如果配置中没有指定，则全部设置为 1
        self.multipliers = config.multipliers if config.multipliers is not None else [1] * levels

        # 初始化 encoders 和 decoders，都是 nn.ModuleList 类型
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            # 计算当前层的宽度和深度
            width = config.res_conv_width * self.multipliers[level]
            depth = config.res_conv_depth * self.multipliers[level]
            # 分别创建 JukeboxEncoder 和 JukeboxDecoder 并加入到 encoders 和 decoders 中
            self.encoders.append(
                JukeboxEncoder(config, width, depth, level + 1, downs_t[: level + 1], strides_t[: level + 1])
            )
            self.decoders.append(
                JukeboxDecoder(config, width, depth, level + 1, downs_t[: level + 1], strides_t[: level + 1])
            )

        # 初始化 bottleneck 层
        self.bottleneck = JukeboxBottleneck(config, levels)
    # 解码函数，将音乐编码 `music_tokens` 解码为原始音频表示
    def _decode(self, music_tokens, start_level=0, end_level=None):
        # 如果未指定结束级别，则使用最大级别
        if end_level is None:
            end_level = self.levels
        # 使用瓶颈网络进行解码，获取潜在状态
        latent_states = self.bottleneck.decode(music_tokens, start_level=start_level, end_level=end_level)
        # 只使用最低级别的解码器
        decoder, dequantised_state = self.decoders[start_level], latent_states[0:1]
        # 使用解码器对去量化状态进行解码
        dequantised_state = decoder(dequantised_state, all_levels=False)
        # 调整维度顺序，将时间轴移至第二个维度
        dequantised_state = dequantised_state.permute(0, 2, 1)
        return dequantised_state

    # 解码函数，将音乐编码 `music_tokens` 解码为原始音频表示，支持批处理
    def decode(self, music_tokens, start_level=0, end_level=None, bs_chunks=1) -> torch.Tensor:
        """
        将输入的 `music_tokens` 解码为它们的 `raw_audio` 表示。

        Args:
            music_tokens (`torch.LongTensor`):
                音乐编码的张量，通过使用码本将其解码为原始音频。每个音乐编码应该是码本中相应 `code` 向量的索引。
            start_level (`int`, *optional*):
                解码过程开始的级别。默认为 0。
            end_level (`int`, *optional*):
                解码过程结束的级别。默认为 None。
            bs_chunks (int, *optional*):
                同时处理的块数。

        Returns:
            `torch.Tensor`: 解码后的原始音频张量。
        """
        # 将音乐编码分块，以便并行处理
        token_chunks = [torch.chunk(token, bs_chunks, dim=0) for token in music_tokens]
        dequantised_states = []
        for i in range(bs_chunks):
            music_tokens_i = [chunks[i] for chunks in token_chunks]
            # 调用 `_decode` 函数进行解码
            dequantised_state = self._decode(music_tokens_i, start_level=start_level, end_level=end_level)
            dequantised_states.append(dequantised_state)
        # 拼接所有解码后的状态张量
        return torch.cat(dequantised_states, dim=0)

    # 编码函数，将原始音频 `raw_audio` 编码为音乐编码 `music_tokens`
    def _encode(self, raw_audio, start_level=0, end_level=None):
        # 编码
        if end_level is None:
            end_level = self.levels
        # 调整输入音频的维度顺序，确保正确的输入格式
        input_audio = raw_audio.permute(0, 2, 1).float()
        latent_states = []
        # 遍历所有级别的编码器，获取潜在状态
        for level in range(self.levels):
            encoder = self.encoders[level]
            latent_state = encoder(input_audio)
            latent_states.append(latent_state[-1])  # 仅保留每级别最后一个潜在状态
        # 使用瓶颈网络对潜在状态进行编码，得到音乐编码 `music_tokens`
        music_tokens = self.bottleneck.encode(latent_states)
        return music_tokens[start_level:end_level]
    # 将输入音频分割成若干块，每块作为一个处理单元
    audio_chunks = torch.chunk(input_audio, bs_chunks, dim=0)
    # 初始化一个空列表，用于存储每个音频块的离散表示
    music_tokens_list = []
    # 遍历每个音频块
    for chunk_i in audio_chunks:
        # 调用内部方法 `_encode` 对当前音频块进行编码，生成其离散表示
        music_tokens_i = self._encode(chunk_i, start_level=start_level, end_level=end_level)
        # 将编码后的离散表示添加到列表中
        music_tokens_list.append(music_tokens_i)
    # 将每个音频块的离散表示进行合并，按照维度0连接在一起，形成最终的音乐表示
    music_tokens = [torch.cat(music_tokens_level, dim=0) for music_tokens_level in zip(*music_tokens_list)]
    # 返回整个音乐表示
    return music_tokens

# 生成指定数量的音乐样本的离散表示
def sample(self, n_samples):
    # 为每个离散表示形状生成随机整数，表示从0到nb_discrete_codes之间的离散码
    music_tokens = [
        torch.randint(0, self.nb_discrete_codes, size=(n_samples, *music_tokens_shape), device="cpu")
        for music_tokens_shape in self.music_tokens_shapes
    ]
    # 调用解码方法，将生成的离散表示解码为音频样本
    return self.decode(music_tokens)
    # Encode/Decode
    input_audio = raw_audio.permute(0, 2, 1).float()
    # 将输入音频数据重新排列维度，使其符合模型要求的格式，并转换为浮点数类型

    latent_states = []
    # 创建空列表，用于存储每个级别的潜在状态

    for level in range(self.levels):
        # 遍历所有编码器级别
        encoder = self.encoders[level]
        # 获取当前级别的编码器
        latent_state = encoder(input_audio)
        # 对输入音频进行编码，得到潜在状态
        latent_states.append(latent_state[-1])
        # 将编码后的潜在状态加入列表中，取最后一个状态

    _, music_tokens, commit_losses, _ = self.bottleneck(latent_states)
    # 使用瓶颈模型处理潜在状态，得到音乐编码、commit loss 等结果

    dequantised_states = []
    # 创建空列表，用于存储每个级别的反量化状态

    for level in range(self.levels):
        # 遍历所有解码器级别
        decoder = self.decoders[level]
        # 获取当前级别的解码器
        dequantised_state = decoder(music_tokens[level : level + 1], all_levels=False)
        # 使用解码器解码音乐编码得到反量化状态
        dequantised_states.append(dequantised_state.permute(0, 2, 1))
        # 将反量化状态重新排列维度并加入列表中

    commit_loss = sum(commit_losses)
    # 计算总的 commit loss
    loss = self.commit * commit_loss
    # 根据 commit 系数计算最终损失值

    return dequantised_states, loss
    # 返回解码后的状态列表及计算得到的损失值
class JukeboxMLP(nn.Module):
    def __init__(self, config):
        # 初始化函数，定义一个多层感知机（MLP）模型
        super().__init__()
        # 从配置中获取隐藏层大小作为嵌入维度
        embed_dim = config.hidden_size
        # 计算隐藏层大小的倍数作为MLP的隐藏层大小
        hidden_dim = int(config.mlp_multiplier * embed_dim)

        # 创建第一个卷积层，输入维度为embed_dim，输出维度为hidden_dim
        self.c_fc = JukeboxConv1D(embed_dim, hidden_dim)
        # 创建第二个卷积层，输入维度为hidden_dim，输出维度为embed_dim
        self.c_proj = JukeboxConv1D(hidden_dim, embed_dim)
        # 选择激活函数，从预定义的函数字典ACT2FN中获取对应配置的激活函数
        self.act = ACT2FN[config.act_fn]
        # 定义Dropout层，使用配置中的残差丢弃率
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, hidden_states):
        # MLP模型的前向传播函数
        # 应用第一个卷积层
        hidden_states = self.c_fc(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 应用第二个卷积层
        hidden_states = self.c_proj(hidden_states)
        # 应用Dropout层
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class JukeboxLayerNorm(FusedLayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        # 初始化函数，定义Jukebox模型的LayerNorm层
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        # 计算输入张量的总元素个数（维度的乘积）
        self.width = np.prod(normalized_shape)
        # 计算能够处理的最大元素个数，限制为65535 * self.width
        self.max_numel = 65535 * self.width

    def forward(self, input):
        # Jukebox模型LayerNorm层的前向传播函数
        if input.numel() > self.max_numel:
            # 如果输入张量的元素个数超过self.max_numel，使用PyTorch的layer_norm函数处理
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps).type_as(input)
        else:
            # 否则调用父类FusedLayerNorm的forward方法处理
            return super().forward(input).type_as(input)


class JukeboxAttention(nn.Module):
    # Jukebox模型的Attention模块定义，未完整展示，不做进一步注释
    # 初始化函数，用于初始化一个模型对象
    def __init__(self, config, n_ctx, attn_func="dense_attn"):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度为配置文件中的隐藏大小
        self.embed_dim = config.hidden_size
        # 设置注意力头数为配置文件中的头数
        self.n_heads = config.n_heads
        # 设置注意力的dropout概率为配置文件中的注意力dropout
        self.dropout = config.attn_dropout
        # 计算隐藏层维度，根据注意力乘子乘以嵌入维度
        hidden_dim = int(config.attention_multiplier * self.embed_dim)

        # 设置每个头的维度
        self.head_dim = hidden_dim // config.n_heads
        # 设置上下文长度
        self.n_ctx = n_ctx
        # 设置隐藏层维度
        self.hidden_dim = hidden_dim
        # 设置缩放因子，用于注意力机制中的缩放
        self.scale = self.head_dim**-0.25
        # 设置是否使用掩码
        self.mask = config.mask

        # 根据注意力函数类型选择不同的处理方式
        if attn_func == "cross_attention":
            # 如果是交叉注意力，设置交叉注意力部分的卷积模块
            self.c_attn = JukeboxConv1D(self.embed_dim, hidden_dim)
            # 设置交叉注意力中的编码键值的卷积模块
            self.c_enc_kv = JukeboxConv1D(self.embed_dim, hidden_dim * 2)
        else:
            # 对于其他类型的注意力，设置通用的卷积模块
            self.c_attn = JukeboxConv1D(self.embed_dim, hidden_dim * 3)

        # 设置投影层的卷积模块，用于最终的投影
        self.c_proj = JukeboxConv1D(hidden_dim, self.embed_dim)
        # 设置注意力的dropout层
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        # 设置残差连接的dropout层
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        # 根据序列长度seq_len将其分解为[块数, seq_len // 块数]的形式
        self.attn_func = attn_func
        # 根据注意力函数类型选择对应的QKV处理函数
        if attn_func == "cross_attention":
            self.qkv = self.decode_qkv
        elif attn_func == "prime_attn":
            self.qkv = self.prime_qkv
        else:
            self.qkv = self.factored_qkv

        # 定义不同注意力类型的映射关系
        ATTENTION_MAP = {
            "dense_attn": (self.dense_attn, "autoregressive"),
            "block_attn": (self.block_attn, "autoregressive"),
            "transpose_block_attn": (self.transpose_block_attn, "autoregressive"),
            "prev_block_attn": (self.prev_block_attn, None),
            "summary_attn": (self.summary_attn, "summary"),
            "summary_spread_attn": (self.summary_spread_attn, "summary"),
            "cross_attention": (self.dense_attn, None),
            "prime_attn": (self.prime_attn, "prime"),
        }
        # 根据传入的注意力函数名称选择对应的注意力函数及其掩码
        self.attn, self.attn_mask = ATTENTION_MAP[attn_func]

        # 设置块数和扩展数
        self.blocks = config.blocks
        self.spread = config.spread
        # 如果定义了块数，则设置块上下文长度
        if self.blocks is not None:
            self.block_ctx = self.n_ctx // self.blocks

        # 设置采样时间为0
        self.sample_t = 0
        # 初始化缓存字典
        self.cache = {}
        # 设置编码器长度，即编码器输入标识符的长度
        self.encoder_len = config.nb_relevant_lyric_tokens  # length of the encoder input ids
        # 记录是否记录注意力权重
        self.record_attn = False
    # 定义注意力机制函数，接受查询、键和值状态以及采样参数
    def _attn(self, query_states, key_states, value_states, sample):
        scale = self.scale
        # 如果处于训练阶段，应用缩放因子后计算注意力权重
        if self.training:
            attention_weight = torch.matmul(query_states * scale, key_states * scale)
        else:
            # 否则直接计算注意力权重，并乘以缩放因子的平方
            attention_weight = torch.matmul(query_states, key_states)
            attention_weight.mul_(scale * scale)
        attn_weight_type = attention_weight.dtype
        # 将注意力权重转换为 float 类型
        attention_weight = attention_weight.float()
        
        # 如果有掩码需求
        if self.mask:
            # 生成适当的掩码以遮蔽当前位置之前的所有位置
            # 对于稠密运算可能占用大量内存，因此可以缓存
            mask = get_mask(
                self.attn_mask,
                query_states.size(-2),
                key_states.size(-1),
                self.blocks,
                self.spread,
                attention_weight.device,
                sample,
                self.sample_t,
            )
            # 如果掩码存在，则应用掩码；否则令未被掩码的位置的注意力权重为一个极小的值
            if mask is not None:
                attention_weight = attention_weight * mask + -1e9 * (1 - mask)
        
        # 对注意力权重进行 softmax 归一化，并根据原始类型重新转换
        attention_prob = F.softmax(attention_weight, dim=-1).type(attn_weight_type)
        
        # 如果记录注意力权重
        if self.record_attn:
            self.attention_prob = attention_prob
            # 如果使用的是特定的注意力函数，只保留音乐查询和歌词键/值对应的注意力权重
            if self.attn_func == "prime_attn":
                self.attention_prob = self.attention_prob[:, :, self.encoder_len :, : self.encoder_len]
        
        # 对注意力权重应用 dropout
        attention_prob = self.attn_dropout(attention_prob)
        
        # 计算上下文状态，通过注意力权重加权求和值状态
        context_states = torch.matmul(attention_prob, value_states)
        return context_states

    # 合并多头注意力机制的结果
    def merge_heads(self, hidden_states):
        # 对隐藏状态进行维度置换，以便后续合并多头注意力的结果
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = (*hidden_states.size()[:-2], hidden_states.size(-2) * hidden_states.size(-1))
        # 将维度变换后的隐藏状态返回，与 TensorFlow 实现中的 merge_states 函数相对应
        return hidden_states.view(*new_hidden_states_shape)

    # 将隐藏状态拆分为多头注意力机制所需的形状
    def split_heads(self, hidden_states, is_key=False):
        # 计算新的隐藏状态形状，以便进行多头注意力机制的拆分
        new_hidden_states_shape = (
            *hidden_states.size()[:-1],
            self.n_heads,
            hidden_states.size(-1) // self.n_heads,
        )
        # 根据新形状对隐藏状态进行视图变换，与 TensorFlow 实现中的 split_states 函数对应
        hidden_states = hidden_states.view(*new_hidden_states_shape)
        
        # 如果是键，进一步置换维度以满足多头注意力机制的要求
        if is_key:
            return hidden_states.permute(0, 2, 3, 1)
        else:
            return hidden_states.permute(0, 2, 1, 3)

    # 密集注意力机制的实现，接受查询、键、值和采样参数
    def dense_attn(self, query, key, value, sample):
        # 对查询、键和值分别进行多头拆分
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        # 应用注意力机制计算上下文状态
        context_states = self._attn(query, key, value, sample)
        # 合并多头注意力机制的结果
        context_states = self.merge_heads(context_states)
        return context_states
    # 定义一个方法用于处理分块注意力机制，接受查询(query)、键(key)、值(value)和一个是否抽样的标志(sample)
    def block_attn(self, query, key, value, sample):
        # 将当前对象的块上下文(block_ctx)存储到局部变量block_ctx中
        block_ctx = self.block_ctx
        # 获取值(value)的形状，其中包括批量大小(batch_size)、序列长度(seq_len)和嵌入维度(embed_dim)
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        
        # 如果抽样标志为True，调用dense_attn方法处理注意力计算，并将结果调整为(batch_size, 1, embed_dim)的形状
        if sample:
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            # 否则，根据查询(query)的长度重新组织查询(query)张量
            query_length = query.shape[1]
            query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)
            
            # 如果查询长度小于序列长度(seq_len)，更新序列长度为查询长度，同时截取键(key)和值(value)的最后一部分
            if query_length < seq_len:
                seq_len = query_length
                key = key[:, -seq_len:].contiguous()
                value = value[:, -seq_len:].contiguous()
            
            # 将键(key)和值(value)重新组织为适合分块上下文的形状
            key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            
            # 调用dense_attn方法计算分块注意力，并将结果调整为(batch_size, seq_len, embed_dim)的形状
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    # 定义一个方法用于处理转置的分块注意力机制，接受查询(query)、键(key)、值(value)和一个是否抽样的标志(sample)
    def transpose_block_attn(self, query, key, value, sample):
        # 将当前对象的块上下文(block_ctx)存储到局部变量block_ctx中
        block_ctx = self.block_ctx
        # 获取值(value)的形状，其中包括批量大小(batch_size)、序列长度(seq_len)和嵌入维度(embed_dim)
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        
        # 如果抽样标志为True，计算最后一个分块长度，截取键(key)和值(value)的特定分块，并调用dense_attn方法计算注意力
        if sample:
            block_len = (seq_len - 1) % block_ctx
            key = key[:, block_len::block_ctx, :]
            value = value[:, block_len::block_ctx, :]
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            # 否则，重新组织查询(query)、键(key)和值(value)，以便进行分块转置操作
            query_length = query.shape[1]
            query = query.view(batch_size, query_length // block_ctx, block_ctx, embed_dim)
            query = query.transpose(1, 2).contiguous()
            query = query.view(batch_size * block_ctx, query_length // block_ctx, embed_dim)

            key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
            key = key.transpose(1, 2).contiguous()
            key = key.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)

            value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
            value = value.transpose(1, 2).contiguous()
            value = value.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)

            # 调用dense_attn方法计算分块注意力，并进行转置以匹配原始序列的形状
            block_attn = self.dense_attn(query, key, value, sample)
            block_attn = block_attn.view(batch_size, block_ctx, query_length // block_ctx, embed_dim)
            block_attn = block_attn.transpose(1, 2).contiguous()
            block_attn = block_attn.view(batch_size, query_length, embed_dim)

            return block_attn
    # 定义一个方法，用于处理前一个块的注意力计算
    def prev_block_attn(self, query, key, value, sample):
        # 获取块的上下文大小
        block_ctx = self.block_ctx
        # 获取 value 的形状信息：batch_size（批大小）、seq_len（序列长度）、embed_dim（嵌入维度）
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        
        # 如果需要采样（sample=True），则处理前一个块的注意力
        if sample:
            # 计算当前块的数量
            block = (seq_len - 1) // block_ctx
            # 计算前一个块的长度
            prev_l = (block - 1) * block_ctx
            
            # 如果存在前一个块
            if block > 0:
                # 截取前一个块的 key 和 value
                key = key[:, prev_l : prev_l + block_ctx, :]
                value = value[:, prev_l : prev_l + block_ctx, :]
            else:
                # 如果不存在前一个块，则创建零张量填充
                key = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
                value = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
            
            # 调用 self.dense_attn 方法进行注意力计算，并将结果 reshape 成 (batch_size, 1, embed_dim) 的形式
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        
        # 如果不需要采样
        else:
            # 获取 query 的长度
            query_length = query.shape[1]
            # 将 query reshape 成适合块大小的形状
            query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)

            # 将 key 和 value 根据块大小进行 reshape
            key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0))
            key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)

            value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0))
            value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)

            # 如果 query 的长度小于 seq_len，则对 key 和 value 进行进一步处理以匹配 query 的长度
            if query_length < seq_len:
                nb_query_blocks = query_length // block_ctx
                nb_key_blocks = seq_len // block_ctx
                seq_len = query_length
                key = key.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
                key = key.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)

                value = value.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
                value = value.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)

            # 调用 self.dense_attn 方法进行注意力计算，并将结果 reshape 成 (batch_size, seq_len, embed_dim) 的形式
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)
    # 计算自注意力摘要
    def summary_attn(self, query, key, value, sample):
        # 获取模型的块数和块上下文大小
        blocks = self.blocks
        block_ctx = self.block_ctx
        batch_size, seq_len, embed_dim = value.shape  # 获取值的形状，其中值的形状为(batch_size, seq_len, embed_dim)，用于sample情况下，query_len= 1, key_len = value_len = sample_t
        if sample:
            # 对样本进行处理，目前未实现该分支的处理方式
            raise NotImplementedError
        else:
            # 对非样本进行处理
            # 调整key的形状以匹配块结构，并进行零填充以适应模型要求
            key = key.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -1, :]
            key = torch.nn.functional.pad(key, (0, 0, 1, 0))  # 在最后一维上进行零填充，确保形状为(batch_size, blocks, embed_dim)

            # 调整value的形状以匹配块结构，并进行零填充以适应模型要求
            value = value.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -1, :]
            value = torch.nn.functional.pad(value, (0, 0, 1, 0))  # 在最后一维上进行零填充，确保形状为(batch_size, blocks, embed_dim)

            # 使用自定义的注意力函数dense_attn进行注意力计算，并重新调整输出的形状以匹配输入value的形状
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    # 计算分散注意力摘要
    def summary_spread_attn(self, query, key, value, sample):
        # 获取模型的块数和分散度大小
        blocks = self.blocks
        spread = self.spread

        batch_size, seq_len, embed_dim = value.shape  # 获取值的形状，其中值的形状为(batch_size, seq_len, embed_dim)，用于sample情况下，query_len= 1, key_len = value_len = sample_t
        if sample:
            # 对样本进行处理，目前未实现该分支的处理方式
            raise NotImplementedError
        else:
            # 对非样本进行处理
            # 调整key的形状以匹配块结构并减少尾部的spread，然后进行零填充和连续化处理以适应模型要求
            key = key.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0)).contiguous()  # 在维度1和2上进行零填充，确保形状为(batch_size, blocks * spread, embed_dim)

            # 调整value的形状以匹配块结构并减少尾部的spread，然后进行零填充和连续化处理以适应模型要求
            value = value.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0)).contiguous()  # 在维度1和2上进行零填充，确保形状为(batch_size, blocks * spread, embed_dim)

            # 使用自定义的注意力函数dense_attn进行注意力计算，并重新调整输出的形状以匹配输入value的形状
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    # 计算主要注意力摘要
    def prime_attn(self, query, key, value, sample):
        # 获取编码器长度
        encoder_len = self._encoder_len

        # 调整key和value的形状以匹配编码器长度，并返回dense_attn函数计算的结果
        key = key[:, :encoder_len]
        value = value[:, :encoder_len]
        return self.dense_attn(query, key, value, sample)
    # 根据给定的隐藏状态张量计算查询、键、值
    def factored_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 获取当前上下文大小
        curr_ctx = hidden_states.shape[1]
        # 如果存在上一个编码器的隐藏状态，则抛出类型错误
        if last_encoder_hidden_states is not None:
            raise TypeError("last_encoder_hidden_states should be None")

        # 将隐藏状态张量按照最后一个维度分成查询、键、值三部分
        query, key, value = hidden_states.chunk(3, dim=2)
        
        # 如果需要进行采样
        if sample:
            # 增加采样计数器
            self.sample_t += curr_ctx
            # 将键和值追加到缓存中
            key, value = self._append_cache(key, value)
            # 计算当前缓存长度
            l_cache = self._suff_cache_len()
            # 如果整体缓存长度超过阈值，进行缓存切片
            if self._cache_len() > l_cache:
                self._slice_cache(-l_cache)
            # 如果当前上下文大于1
            if curr_ctx > 1:
                # 如果注意力函数不是 "dense_attn"，对查询、键、值进行块填充
                if self.attn_func != "dense_attn":
                    query = self._pad_to_block_ctx(query, query=True)
                    key = self._pad_to_block_ctx(key)
                    value = self._pad_to_block_ctx(value)
                # 禁用采样标志
                sample = False
            else:
                # 如果当前上下文为1，则从缓存中获取键和值
                key = self.cache["key"]
                value = self.cache["value"]

        # 返回查询、键、值以及采样标志
        return query, key, value, sample

    # 根据给定的隐藏状态张量计算查询、键、值
    def prime_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 获取当前上下文大小
        curr_ctx = hidden_states.shape[1]
        # 如果存在上一个编码器的隐藏状态，则抛出类型错误
        if last_encoder_hidden_states is not None:
            raise TypeError("last_encoder_hidden_states should be None")
        
        # 将隐藏状态张量按照最后一个维度分成查询、键、值三部分
        query, key, value = hidden_states.chunk(3, dim=2)
        
        # 如果需要进行采样
        if sample:
            # 如果缓存长度小于编码器长度，则将键和值追加到缓存中
            if self._cache_len() < self._encoder_len:
                self._append_cache(key, value)
            # 如果缓存长度大于编码器长度，则对缓存进行切片操作
            if self._cache_len() > self._encoder_len:
                self._slice_cache(0, self._encoder_len)
            # 从缓存中获取键和值
            key, value = self.cache["key"], self.cache["value"]
            # 增加采样计数器
            self.sample_t += curr_ctx
        
        # 返回查询、键、值以及采样标志
        return query, key, value, sample

    # 根据给定的隐藏状态张量计算查询、键、值
    def decode_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 获取当前上下文大小
        curr_ctx = hidden_states.shape[1]
        # 将隐藏状态作为查询
        query = hidden_states
        
        # 如果需要进行采样
        if sample:
            # 如果采样计数器为0，则从编码器的隐藏状态生成键和值，并存入缓存
            if self.sample_t == 0:
                self.cache["key"], self.cache["value"] = self.c_enc_kv(
                    last_encoder_hidden_states.type_as(hidden_states)
                ).chunk(2, dim=2)
            # 从缓存中获取键和值
            key, value = self.cache["key"], self.cache["value"]
            # 增加采样计数器
            self.sample_t += curr_ctx
        else:
            # 否则，根据给定的隐藏状态生成键和值
            key, value = self.c_enc_kv(last_encoder_hidden_states.type_as(hidden_states)).chunk(2, dim=2)
        
        # 返回查询、键、值以及采样标志
        return query, key, value, sample
    # 定义一个方法，用于进行前向传播计算
    def forward(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 获取当前上下文的长度
        curr_ctx = hidden_states.shape[1]
        # 对输入的隐藏状态应用注意力机制
        hidden_states = self.c_attn(hidden_states)
        # 使用查询、键、值进行注意力机制的计算
        query, key, value, sample = self.qkv(
            hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=sample
        )
        # 计算注意力分数
        attention_scores = self.attn(query, key, value, sample)
        # 如果注意力分数的长度与当前上下文长度不一致，则进行偏移操作
        if attention_scores.shape[1] != curr_ctx:
            offset = self._offset(curr_ctx)
            attention_scores = attention_scores[:, offset : offset + curr_ctx, :].contiguous()
        # 应用变换投影到输出空间
        attention_scores = self.c_proj(attention_scores)
        # 应用残差连接的dropout操作并返回结果
        return self.resid_dropout(attention_scores)

    # 定义一个属性方法，用于获取编码器的长度
    @property
    def _encoder_len(self):
        # 获取编码器长度属性
        encoder_len = self.encoder_len
        # 计算编码器块的数量
        encoder_blocks = (encoder_len // self.blocks) + 1
        # 返回调整后的编码器长度
        return encoder_blocks * self.blocks

    # 定义一个方法，用于计算偏移量
    def _offset(self, curr_ctx):
        # 如果使用密集注意力机制，则返回0
        if self.attn_func == "dense_attn":
            return 0
        # 否则，计算偏移量并返回
        return (self.sample_t - curr_ctx) % self.block_ctx

    # 定义一个方法，用于将隐藏状态填充到块上下文的长度
    def _pad_to_block_ctx(self, hidden_states, query=False):
        # 获取序列长度
        seq_len = hidden_states.shape[1]
        # 如果是查询，则计算偏移量
        offset = self._offset(seq_len) if query else 0
        # 计算块的数量
        n_blocks = (seq_len + offset + self.block_ctx - 1) // self.block_ctx
        # 计算填充的长度
        pad = n_blocks * self.block_ctx - seq_len - offset
        # 如果无需填充，则直接返回隐藏状态
        if pad == 0 and offset == 0:
            return hidden_states
        else:
            # 否则，对隐藏状态进行填充并返回
            return F.pad(hidden_states, (0, 0, offset, pad))

    # 定义一个方法，用于获取缓存的长度
    def _cache_len(self):
        # 如果缓存中没有键值对，则返回0；否则返回键的长度
        return 0 if "key" not in self.cache else self.cache["key"].shape[1]

    # 定义一个方法，用于获取必要的缓存长度
    def _suff_cache_len(self):
        """
        前提条件:
            键和值已经附加了当前上下文，并且self.sample_t反映了上下文中的1索引样本位置。
        """
        # 计算前一个块的长度
        previous_block_length = (self.sample_t - 1) % self.block_ctx + 1 + self.block_ctx
        # 定义必要的缓存长度字典
        REQUIRED_CACHE_LEN = {
            "dense_attn": self.sample_t,
            "block_attn": (self.sample_t - 1) % self.block_ctx + 1,
            "transpose_block_attn": self.sample_t,
            "prev_block_attn": self.sample_t if self.sample_t <= self.block_ctx else previous_block_length,
            "cross_attn": self.encoder_len,
            "prime_attn": min(self.sample_t, self._encoder_len),
        }
        # 返回根据注意力机制类型选择的必要缓存长度
        return REQUIRED_CACHE_LEN[self.attn_func]

    # 定义一个方法，用于对缓存进行切片
    def _slice_cache(self, start, end=None):
        # 对键和值缓存进行切片操作
        self.cache["key"] = self.cache["key"][:, start:end]
        self.cache["value"] = self.cache["value"][:, start:end]
    # 将键值对添加到缓存中，如果键不存在则创建新的缓存项，否则更新现有缓存项
    def _append_cache(self, key, value):
        # 检查缓存中是否已存在键
        if "key" not in self.cache:
            # 如果不存在，则将提供的键和值存入缓存
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            # 如果存在，则合并现有键值和新的键值对，并更新缓存
            old_key, old_value = key, value
            key = torch.cat([self.cache["key"], old_key], dim=1)
            value = torch.cat([self.cache["value"], old_value], dim=1)
            # 删除旧的键和值以释放内存
            del self.cache["key"]
            del self.cache["value"]
            del old_key
            del old_value
            # 更新缓存的键和值
            self.cache["key"] = key
            self.cache["value"] = value
        # 返回更新后的键和值
        return self.cache["key"], self.cache["value"]
    
    # 清空缓存中的所有项，并重置样本计数器
    def del_cache(self):
        self.sample_t = 0  # 重置样本计数器为0
        if "key" in self.cache:
            del self.cache["key"]  # 删除缓存中的键
        if "value" in self.cache:
            del self.cache["value"]  # 删除缓存中的值
        self.cache = {}  # 清空整个缓存字典
class JukeboxBlock(nn.Module):
    # JukeboxBlock 类，用于实现一个模块
    def __init__(self, config, n_ctx, attn_func="dense_attn"):
        super().__init__()
        # 设置模块的宽度为隐藏层大小
        self.width = config.hidden_size
        # 创建 JukeboxAttention 对象，并存储在 self.attn 中
        self.attn = JukeboxAttention(config, n_ctx, attn_func=attn_func)

        # 创建第一个 Layer Normalization 层，并存储在 self.layer_norm_0 中
        self.layer_norm_0 = JukeboxLayerNorm(config.hidden_size)
        # 创建 JukeboxMLP 对象，并存储在 self.mlp 中
        self.mlp = JukeboxMLP(config)
        # 创建第二个 Layer Normalization 层，并存储在 self.layer_norm_1 中
        self.layer_norm_1 = JukeboxLayerNorm(config.hidden_size)
        # 设置残差比例，如果启用注意力残差缩放，为 1/层数，否则为 1.0
        self.res_scale = 1.0 / config.num_layers if config.attn_res_scale else 1.0
        # 存储注意力函数名称
        self.attn_func = attn_func

    def forward(self, hidden_states, last_encoder_hidden_states, sample=False):
        # 复制输入的隐藏状态作为残差
        residuals = hidden_states
        # 应用第一个 Layer Normalization 层
        hidden_states = self.layer_norm_0(hidden_states)
        # 应用注意力机制，并更新隐藏状态
        hidden_states = self.attn(hidden_states, last_encoder_hidden_states, sample)

        # 计算输出状态，结合残差和更新后的隐藏状态
        output_states = self.layer_norm_1(residuals + hidden_states)
        # 应用 MLP 层
        output_states = self.mlp(output_states)
        # 计算最终输出，结合残差、更新后的隐藏状态和 MLP 输出
        if self.res_scale == 1.0:
            output = residuals + hidden_states + output_states
        else:
            output = residuals + self.res_scale * (hidden_states + output_states)
        return output


class JukeboxLayerStack(nn.Module):
    # JukeboxLayerStack 类，用于堆叠多个 JukeboxBlock 模块
    def __init__(self, config, n_ctx):
        super().__init__()
        # 初始化上下文长度和宽度为隐藏层大小
        self.n_ctx = n_ctx
        self.width = config.hidden_size
        # 设置层数和块数
        self.num_layers = config.num_layers
        self.blocks = config.blocks
        # 设置注意力模式
        self.attention_pattern = config.attention_pattern
        # 如果定义了块数，则计算每个块的上下文长度
        if self.blocks is not None:
            self.block_ctx = n_ctx // self.blocks
        # 设置编码器长度
        self.encoder_len = config.nb_relevant_lyric_tokens
        # 设置头数
        self.n_heads = config.n_heads

        # 根据注意力模式创建注意力模块列表
        attention_pattern = ATTENTION_PATTERNS[self.attention_pattern]
        self._attn_mods = nn.ModuleList()
        for depth in range(self.num_layers):
            # 向 _attn_mods 列表添加 JukeboxBlock 模块
            self._attn_mods.append(JukeboxBlock(config, n_ctx, attn_func=attention_pattern(depth)))

        # 用于存储注意力权重
        self.saved_attn_weights = []

    def set_record_attn(self, record_attn):
        """
        设置是否记录注意力 softmax 到 self.saved_attn_weights 中。

        Args:
            record_attn (`Union[bool,set]`):
                若为 set 类型，表示要记录哪些层的注意力 softmax；若为 bool 类型，表示是否全部记录。
        """
        # 判断是否记录每一层的注意力 softmax
        def _should_record_attn(layer_idx):
            if isinstance(record_attn, bool):
                return record_attn
            return layer_idx in record_attn

        # 设置每个层的注意力记录属性
        for i, layer in enumerate(self._attn_mods):
            layer.attn.record_attn = _should_record_attn(i)

        # 若不记录任何注意力 softmax，则清空 self.saved_attn_weights
        if not record_attn:
            self.saved_attn_weights = []
    # 前向传播函数，用于处理隐藏状态和可能的编码器最后隐藏状态，支持采样
    def forward(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 遍历注意力层模块列表
        for i, attn_layer in enumerate(self._attn_mods):
            # 如果当前注意力层为跨注意力机制，即跨编码器-解码器注意力
            if attn_layer.attn_func == "cross_attention":  # attend to the lyrics
                # 执行跨注意力机制，将当前隐藏状态和最后编码器隐藏状态作为参数传入
                hidden_states = attn_layer(
                    hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=sample
                )
            else:
                # 否则，执行普通的注意力机制，不使用编码器的隐藏状态
                hidden_states = attn_layer(hidden_states, last_encoder_hidden_states=None, sample=sample)
            # 如果当前注意力层记录了注意力权重
            if attn_layer.attn.record_attn:
                # 将当前注意力层的注意力权重保存到列表中
                self.saved_attn_weights.append(attn_layer.attn.c_attn.weight)
        # 返回处理后的隐藏状态
        return hidden_states

    # 删除缓存函数，用于清空所有注意力层的缓存
    def del_cache(self):
        # 遍历所有注意力层模块
        for attn_layer in self._attn_mods:
            # 调用注意力层对象的删除缓存方法
            attn_layer.attn.del_cache()
class JukeboxPositionalEmbedding(nn.Module):
    # JukeboxPositionalEmbedding 类定义，继承自 nn.Module
    def __init__(self, embed_dim, width):
        # 初始化方法
        super().__init__()
        # 创建一个可学习的参数 pos_emb，其形状为 (embed_dim, width)
        self.pos_emb = nn.Parameter(torch.empty((embed_dim, width)))

    def forward(self):
        # 前向传播方法
        pos_emb = self.pos_emb
        # 返回位置嵌入参数 pos_emb
        return pos_emb


class JukeboxConditionalAutoregressive(nn.Module):
    # JukeboxConditionalAutoregressive 类定义，继承自 nn.Module
    def __init__(
        self,
        config,
        n_ctx=None,
        embed_dim=None,
        audio_conditioning=False,
        metadata_conditioning=False,
        is_encoder=False,
    ):
        # 初始化方法，接受多个参数，包括模型配置、上下文长度、嵌入维度等
        super().__init__()
        # 此处缺少进一步的代码，可能涉及模型的具体定义和初始化，需查看完整代码以添加详细注释
        """
        Autoregressive model on either lyric tokens or music tokens, or both. The attention pattern should be properly
        set fro each configuration.

        Args:
            config (`JukeboxPriorConfig`):
                Model configuration class with all the parameters of the model. Initializing with a config file does
                not load the weights associated with the model, only the configuration. Check out the
                [`~PreTrainedModel.from_pretrained`] method to load the model weights.
            n_ctx (`int`, *optional*):
                Number of tokens or lyrics tokens provided in a single pass.
            embed_dim (`int`, *optional*):
                Either equals to the dimension of the codebook, or the sum of n_vocab (lyrics) and codeboook dimension,
                if the model combines lyrics and music tokens, or simply n_vocab if the model is a seperate encoder
            audio_conditioning (`bool`, *optional`, defaults to `False`):
                Whether or not the prior supports conditioning on audio.
            metadata_conditioning (`bool`, *optional`, defaults to `False`):
                Whether or not the prior supports conditioning on artist, genres, lyrics, and timing.
            is_encoder (`bool`, *optional`, defaults to `False`):
                Whether the model is an encoder only model.
        """

        # Initialize the class inheriting from nn.Module
        super().__init__()

        # Set the width of the model from the configuration
        self.width = config.hidden_size
        # Set the number of layers from the configuration
        self.num_layers = config.num_layers
        # Set the context length from the provided argument or the configuration
        self.n_ctx = n_ctx if n_ctx is not None else config.n_ctx
        # Set the embedding dimension based on the argument or configuration's music vocabulary size
        self.embed_dim = embed_dim if embed_dim is not None else config.music_vocab_size

        # Initialize embedding tokens using nn.Embedding with embed_dim and hidden_size from configuration
        self.embed_tokens = nn.Embedding(self.embed_dim, config.hidden_size)
        # Apply dropout to embed_tokens based on config's embedding dropout rate
        self.embed_tokens_dropout = nn.Dropout(config.emb_dropout)

        # Set metadata and audio conditioning flags
        self.metadata_conditioning = metadata_conditioning
        self.audio_conditioning = audio_conditioning

        # If metadata_conditioning is False, initialize start_token as a learnable parameter
        if not metadata_conditioning:
            self.start_token = nn.Parameter(torch.empty((1, config.hidden_size)))

        # Initialize positional embedding using JukeboxPositionalEmbedding with n_ctx and hidden_size
        self.pos_emb = JukeboxPositionalEmbedding(self.n_ctx, config.hidden_size)
        # Apply dropout to positional embedding based on config's embedding dropout rate
        self.pos_emb_dropout = nn.Dropout(config.emb_dropout)

        # Initialize transformer layer stack using JukeboxLayerStack with config and n_ctx
        self.transformer = JukeboxLayerStack(config, n_ctx=self.n_ctx)
        # Set whether the model is an encoder based on is_encoder flag
        self.is_encoder = is_encoder
        # Set encoder length from configuration's relevant lyric tokens count
        self.encoder_len = config.nb_relevant_lyric_tokens

        # Conditional setups based on config's merged_decoder flag
        if config.merged_decoder:
            self.add_cond_after_transformer = False
            self.share_embed_tokens_fc_proj_out = False
        else:
            self.add_cond_after_transformer = True
            self.share_embed_tokens_fc_proj_out = True

        # If not an encoder, initialize output projection layer and loss function
        if not is_encoder:
            # Linear projection layer from hidden_size to embed_dim
            self.fc_proj_out = nn.Linear(config.hidden_size, self.embed_dim, bias=False)
            # If sharing embed tokens and fc_proj_out weights, synchronize them
            if self.share_embed_tokens_fc_proj_out:
                self.fc_proj_out.weight = self.embed_tokens.weight
            # Cross-entropy loss function initialization
            self.loss = torch.nn.CrossEntropyLoss()
    def forward(
        self,
        tokens,
        audio_conditioning=None,
        metadata_conditioning=None,
        last_encoder_hidden_states=None,
        get_preds=False,
        get_acts=False,
        get_sep_loss=False,
    ):
        """
        Args:
            tokens (`torch.tensor`):
                Can represent music tokens, lyrics tokens or both, depending on the configuration.
        """
        # Preprocess.
        batch_size = tokens.shape[0]  # 获取批处理大小
        with torch.no_grad():
            tokens = tokens.view(batch_size, -1).long()  # 转换 tokens 的形状

        if not self.audio_conditioning:
            # 如果没有音频条件，则创建全零的音频条件张量
            audio_conditioning = torch.zeros(
                (batch_size, 1, self.width),
                device=tokens.device,
                dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype,
            )

        target = tokens  # 目标 tokens
        hidden_states = self.embed_tokens(tokens)  # 嵌入 tokens
        # Shift by 1, and fill in start token
        hidden_states = torch.cat((hidden_states[:, -1:], hidden_states[:, :-1]), dim=1)  # 将 tokens 向右移动一个位置，并填充起始 token
        if self.metadata_conditioning:
            hidden_states[:, 0] = metadata_conditioning.view(batch_size, self.width)  # 如果有元数据条件，则使用元数据条件
        else:
            hidden_states[:, 0] = self.start_token  # 否则使用预定义的起始 token

        hidden_states = (
            self.embed_tokens_dropout(hidden_states) + self.pos_emb_dropout(self.pos_emb()) + audio_conditioning
        )  # 添加嵌入 tokens 的 dropout、位置编码的 dropout 和音频条件

        hidden_states = self.transformer(
            hidden_states, last_encoder_hidden_states=last_encoder_hidden_states
        )  # 应用 transformer 模型进行编码

        if self.add_cond_after_transformer:  # 如果在 transformer 后添加条件
            hidden_states = hidden_states + audio_conditioning  # 添加音频条件

        activations = hidden_states  # 激活值等于隐藏状态
        if self.is_encoder:
            return hidden_states  # 如果是编码器，直接返回隐藏状态

        hidden_states = self.fc_proj_out(hidden_states)  # 使用全连接层进行预测
        loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

        if get_sep_loss:
            # 如果需要单独计算损失
            lyric_hidden_states = hidden_states[:, : self.encoder_len].reshape(-1, self.embed_dim)
            token_hidden_states = hidden_states[:, self.encoder_len :].reshape(-1, self.embed_dim)

            lyric_loss = loss_fn(lyric_hidden_states, target[:, : self.encoder_len].reshape(-1)) / np.log(2.0)  # 计算歌词部分的损失
            music_token_loss = loss_fn(token_hidden_states, target[:, self.encoder_len :].reshape(-1)) / np.log(2.0)  # 计算音乐 token 部分的损失

            loss = (lyric_loss, music_token_loss)  # 返回歌词损失和音乐 token 损失
        else:
            loss = loss_fn(hidden_states.view(-1, self.embed_dim), target.view(-1)) / np.log(2.0)  # 计算整体损失

        if get_preds:
            return loss, hidden_states  # 如果需要预测，返回损失和隐藏状态
        elif get_acts:
            return loss, activations  # 如果需要激活值，返回损失和激活值
        else:
            return loss, None  # 否则只返回损失
    # 定义一个方法，用于获取嵌入表示
    def get_emb(self, sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning):
        # 如果是第一个样本
        if sample_t == 0:
            # 创建一个空的张量用于存储隐藏状态，形状为 (n_samples, 1, self.width)，数据类型与权重张量相同，并移到相同的设备上
            hidden_states = torch.empty(n_samples, 1, self.width, dtype=self.embed_tokens.weight.dtype).to(
                self.embed_tokens.weight.device
            )
            # 如果有元数据条件
            if self.metadata_conditioning:
                # 将元数据条件视图重塑为 (n_samples, self.width)，并赋值给隐藏状态的第一个位置
                hidden_states[:, 0] = metadata_conditioning.view(n_samples, self.width)
            else:
                # 否则将起始标记赋值给隐藏状态的第一个位置
                hidden_states[:, 0] = self.start_token
        else:
            # 对于非第一个样本，使用嵌入的 token 表示 tokens
            hidden_states = self.embed_tokens(tokens)
        
        # 如果音频条件的形状与期望的形状相同
        if audio_conditioning.shape == (n_samples, self.n_ctx, self.width):
            # 则将对应的音频条件切片赋给 cond，形状为 (n_samples, 1, self.width)
            cond = audio_conditioning[:, sample_t : sample_t + 1, :]
        else:
            # 否则直接使用原始的音频条件
            cond = audio_conditioning
        
        # 添加位置嵌入和音频条件到隐藏状态中，位置嵌入在评估时的 dropout 是恒等映射
        hidden_states = hidden_states + self.pos_emb()[sample_t : sample_t + 1] + cond
        
        # 返回更新后的隐藏状态和条件
        return hidden_states, cond
        ):
        # 如果未指定采样的 tokens 数量，则使用默认值 self.n_ctx
        if sample_tokens is None:
            sample_tokens = self.n_ctx

        # 如果不需要音频调节，则创建一个全零张量作为音频调节
        if not self.audio_conditioning:
            audio_conditioning = torch.zeros(
                (n_samples, 1, self.width), dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype
            ).to(self.fc_proj_out.device)

        # 禁止梯度更新
        with torch.no_grad():
            sampled_tokens = []
            tokens = None
            if get_preds:
                preds = []

            # 使用 tqdm 创建进度条迭代器
            iter = tqdm(range(0, sample_tokens), leave=False)
            for sample_t in iter:
                iter.set_description(f"Ancestral sampling {sample_tokens} music tokens", refresh=True)
                # 获取嵌入向量和条件
                hidden_states, cond = self.get_emb(
                    sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning
                )

                # 使用 transformer 进行前向传播
                hidden_states = self.transformer(
                    hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=True
                )
                # 如果设置了在 transformer 后添加条件
                if self.add_cond_after_transformer:
                    hidden_states = hidden_states + cond
                # 使用全连接层进行预测
                hidden_states = self.fc_proj_out(hidden_states)  # Predictions
                # 如果需要获取预测值，则保存预测结果
                if get_preds:
                    preds.append(hidden_states.clone())
                # 调整 logits 的值
                hidden_states = hidden_states / temp
                hidden_states = filter_logits(hidden_states, top_k=top_k, top_p=top_p)
                # 从 logits 中采样生成 tokens
                tokens = torch.distributions.Categorical(logits=hidden_states).sample()
                sampled_tokens.append(tokens.clone())

            del tokens
            # 清除 transformer 的缓存
            self.transformer.del_cache()

            # 拼接所有采样的 tokens
            tokens = torch.cat(sampled_tokens, dim=1)
            if get_preds:
                preds = torch.cat(preds, dim=1)
        # 如果需要获取预测值，则返回 tokens 和 preds
        if get_preds:
            return tokens, preds
        # 否则，只返回 tokens
        else:
            return tokens

    def split_chunks(self, length, chunk_size):
        # 计算分块的数量
        n_passes = (length + chunk_size - 1) // chunk_size
        # 计算每个分块的大小列表
        chunk_sizes = [*[chunk_size] * (n_passes - 1), (length - 1) % chunk_size + 1]
        return chunk_sizes

    def primed_sample(
        self,
        n_samples,
        lyric_and_music_tokens,
        audio_conditioning=None,
        metadata_conditioning=None,
        last_encoder_hidden_states=None,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        get_preds=False,
        chunk_size=None,
        sample_tokens=None,
class JukeboxMusicTokenConditioner(nn.Module):
    """
    The `JukeboxMusicTokenConditioner` takes music tokens as an input (coresponding to the codes of the VQVAE's
    codebook) and upsamples it using a single layer of decoder convolution block (the same is used in the VQVAE).
    """

    def __init__(self, config, level):
        super().__init__()
        # Initialize an embedding layer for music tokens based on vocabulary size and hidden size
        self.embed_tokens = nn.Embedding(config.music_vocab_size, config.hidden_size)
        # Set the embed_dim attribute in config to music_vocab_size for compatibility with JukeboxDecoder
        config.embed_dim = config.music_vocab_size  # setting correct argument for the `JukeboxDecoder`

        # Initialize the upsampler using a custom convolutional block
        self.upsampler = JukeboxDecoderConvBock(
            config,
            config.hidden_size,
            config.res_conv_width,
            config.res_conv_depth,
            config.res_downs_t[level],
            config.res_strides_t[level],
            reverse_dilation=False,
        )
        # Initialize layer normalization for the hidden states
        self.layer_norm = JukeboxLayerNorm(config.hidden_size)

    def forward(self, music_tokens, raw_audio_conditionning=None):
        """
        Args:
            music_tokens (`torch.LongTensor`):
                Music tokens form the uper level in range(nb_discrete_codes)
            raw_audio_conditionning (`torch.LongTensor`, *optional*):
                Audio used when primed sampling, raw audio information that conditions the generation
        """
        # Set default value for raw_audio_conditioning if not provided
        if raw_audio_conditionning is None:
            raw_audio_conditionning = 0.0
        # Convert music_tokens to long type
        music_tokens = music_tokens.long()
        # Embed music_tokens using the previously initialized embedding layer
        hidden_states = self.embed_tokens(music_tokens)
        # Add raw_audio_conditioning to the embedded music tokens
        hidden_states = hidden_states + raw_audio_conditionning

        # Permute dimensions for upsampling
        hidden_states = hidden_states.permute(0, 2, 1)
        # Apply the upsampler to the permuted hidden states
        hidden_states = self.upsampler(hidden_states)
        # Permute dimensions back to original shape
        hidden_states = hidden_states.permute(0, 2, 1)
        # Apply layer normalization to the processed hidden states
        hidden_states = self.layer_norm(hidden_states)
        # Return the normalized hidden states
        return hidden_states


class JukeboxRangeEmbedding(nn.Module):
    """
    The `JukeboxRangeEmbedding` interpolate the given [pos_start, pos_end] to obtain an equivalent of time positional
    embedding of length `n_ctx`.

    Binning process : For each pos in position tensor, find its bin [start,end) mapped to [0,1,...,bins-1] [start,end)
    -> [0,1) -> [0, bins) -> floor -> [0,...,bins-1] NOTE: Open ended interval on right, so start <= pos < end, not <=
    end
    """

    def __init__(self, n_time, embed_dim, range, out_width, clamp=False):
        super().__init__()
        # Initialize an embedding layer with size embed_dim and output width out_width
        self.emb = nn.Embedding(embed_dim, out_width)
        self.n_time = n_time
        self.embed_dim = embed_dim
        # Define positional range [pos_min, pos_max]
        self.pos_min, self.pos_max = range
        self.clamp = clamp
    # 定义一个方法用于将位置起始点和结束点进行前向传播
    def forward(self, pos_start, pos_end=None):
        # 检查 pos_start 的形状是否为二维
        if not len(pos_start.shape) == 2:
            raise TypeError(f"Expected shape with 2 dims, got {pos_start.shape}")
        # 检查 pos_start 是否在指定范围 [pos_min, pos_max) 内
        if not (self.pos_min <= pos_start).all() and (pos_start < self.pos_max).all():
            raise TypeError(f"Range is [{self.pos_min},{self.pos_max}), got {pos_start}")

        # 将 pos_start 转换为 float 类型
        pos_start = pos_start.float()
        # 如果 pos_end 不为 None
        if pos_end is not None:
            # 如果设置了 clamp 标志，将 pos_end 限制在 pos_min 和 pos_max 范围内
            if self.clamp:
                pos_end = pos_end.clamp(self.pos_min, self.pos_max)

            # 将 pos_end 转换为 float 类型
            pos_end = pos_end.float()

        # 计算插值以使得 [pos_start, ..., pos_end] <-> 长度为 n_ctx 的位置张量
        n_time = self.n_time
        if n_time != 1:
            # 生成插值张量，用于在 pos_start 到 pos_end 之间进行线性插值
            interpolation = (
                torch.arange(0, n_time, dtype=torch.float, device=pos_start.device).view(1, n_time) / n_time
            )
            position = pos_start + (pos_end - pos_start) * interpolation
        else:
            position = pos_start

        # 将位置归一化到 [0, 1] 范围内
        normalised_position = (position - self.pos_min) / (self.pos_max - self.pos_min)
        # 将归一化后的位置映射到 bins_，用于离散化表示
        bins_ = (self.embed_dim * normalised_position).floor().long().detach()
        # 返回根据 bins_ 索引得到的嵌入向量
        return self.emb(bins_)
class JukeboxLabelConditioner(nn.Module):
    def __init__(self, config, include_time_signal):
        super().__init__()

        embed_dim = config.hidden_size  # 从配置中获取隐藏单元的维度
        timing_dims = config.timing_dims  # 从配置中获取时间维度
        sampling_rate = config.sampling_rate  # 从配置中获取采样率
        nb_genres, nb_artists = config.metadata_dims  # 从配置中获取流派和艺术家的维度
        music_tokens_shape = config.n_ctx  # 从配置中获取音乐令牌的形状

        self.max_nb_genres = config.max_nb_genres  # 设置最大流派数量
        self.bow_genre_emb = nn.Embedding(nb_genres, embed_dim)  # 创建流派嵌入层
        self.artist_emb = nn.Embedding(nb_artists, embed_dim)  # 创建艺术家嵌入层
        self.include_time_signal = include_time_signal  # 设置是否包含时间信号的标志
        if self.include_time_signal:
            # 如果包含时间信号，设置总长度范围、绝对位置范围和相对位置范围
            total_length_range = (config.min_duration * sampling_rate, config.max_duration * sampling_rate)
            absolute_pos_range = (0.0, config.max_duration * sampling_rate)
            relative_pos_range = (0.0, 1.0)
            # 创建总长度、绝对位置和相对位置的嵌入层
            self.total_length_emb = JukeboxRangeEmbedding(1, timing_dims, total_length_range, embed_dim)
            self.absolute_pos_emb = JukeboxRangeEmbedding(
                music_tokens_shape, timing_dims, absolute_pos_range, embed_dim
            )
            self.relative_pos_emb = JukeboxRangeEmbedding(
                music_tokens_shape, timing_dims, relative_pos_range, embed_dim, clamp=True
            )

    def forward(self, metadata):
        total_length = metadata[:, 0:1]  # 提取元数据中的总长度
        offset = metadata[:, 1:2]  # 提取元数据中的偏移量
        length = metadata[:, 2:3]  # 提取元数据中的长度
        artist = metadata[:, 3:4]  # 提取元数据中的艺术家
        genre = metadata[:, 4:]  # 提取元数据中的流派

        # 起始嵌入，长度为1
        artist_emb = self.artist_emb(artist)  # 计算艺术家的嵌入表示
        # 空的流派插槽用-1表示，对其进行屏蔽处理
        mask = (genre >= 0).float().unsqueeze(2)  # 创建流派屏蔽掩码
        genre_emb = (self.bow_genre_emb(genre.clamp(0)) * mask).sum(dim=1, keepdim=True)  # 计算流派的嵌入表示
        start_emb = genre_emb + artist_emb  # 合并艺术家和流派的嵌入表示作为起始嵌入

        # 位置嵌入，长度为n_ctx
        if self.include_time_signal:
            start, end = offset, offset + length  # 计算起始和结束位置
            total_length = total_length.float()  # 将总长度转换为浮点数
            start = start.float()  # 将起始位置转换为浮点数
            end = end.float()  # 将结束位置转换为浮点数
            # 计算总长度、绝对位置和相对位置的嵌入表示
            pos_emb = (
                self.total_length_emb(total_length)
                + self.absolute_pos_emb(start, end)
                + self.relative_pos_emb(start / total_length, end / total_length)
            )
        else:
            pos_emb = None  # 如果不包含时间信号，则位置嵌入为None
        return start_emb, pos_emb  # 返回起始嵌入和位置嵌入
    # 定义一个类变量，指定配置类为 JukeboxPriorConfig
    config_class = JukeboxPriorConfig

    # 初始化模型权重的方法，接受一个模块作为参数
    def _init_weights(self, module):
        # 从配置中获取初始化比例
        init_scale = self.config.init_scale

        # 如果模块是 nn.Embedding 类型
        if isinstance(module, nn.Embedding):
            # 对权重数据进行正态分布初始化，均值为 0，标准差为 0.02 * init_scale
            module.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        
        # 如果模块是 JukeboxConv1D 类型
        elif isinstance(module, JukeboxConv1D):
            # 如果配置中指定需要将权重置零
            if self.config.zero_out:
                # 将权重数据置零
                module.weight.data.zero_()
            else:
                # 否则对权重数据进行正态分布初始化，均值为 0，标准差为 0.02 * init_scale
                module.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        
        # 如果模块是 JukeboxPositionalEmbedding 类型
        elif isinstance(module, JukeboxPositionalEmbedding):
            # 对位置嵌入数据进行正态分布初始化，均值为 0，标准差为 0.01 * init_scale
            module.pos_emb.data.normal_(mean=0.0, std=0.01 * init_scale)
        
        # 如果模块是 JukeboxRangeEmbedding 类型
        elif isinstance(module, JukeboxRangeEmbedding):
            # 对范围嵌入的权重数据进行正态分布初始化，均值为 0，标准差为 0.01 * init_scale
            module.emb.weight.data.normal_(mean=0.0, std=0.01 * init_scale)
        
        # 如果模块是 JukeboxConditionalAutoregressive 类型，并且具有 lm_head 属性
        elif isinstance(module, JukeboxConditionalAutoregressive) and hasattr(module, "lm_head"):
            # 对 lm_head 的权重数据进行正态分布初始化，均值为 0，标准差为 0.02 * init_scale
            module.lm_head.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        
        # 如果模块是 JukeboxConditionalAutoregressive 类型，并且具有 start_token 属性
        elif isinstance(module, JukeboxConditionalAutoregressive) and hasattr(module, "start_token"):
            # 对 start_token 的数据进行正态分布初始化，均值为 0，标准差为 0.01 * init_scale
            module.start_token.data.normal_(mean=0.0, std=0.01 * init_scale)
        
        # 如果模块是 JukeboxResConv1DBlock 类型，并且配置中指定需要置零
        elif isinstance(module, JukeboxResConv1DBlock) and self.config.zero_out:
            # 将 conv1d_2 的权重和偏置数据置零
            module.conv1d_2.weigth.data.zero_()
            module.conv1d_2.bias.data.zero_()
        
        # 如果模块是 nn.LayerNorm 类型
        if isinstance(module, nn.LayerNorm):
            # 将偏置数据置零
            module.bias.data.zero_()
            # 将权重数据填充为 1.0
            module.weight.data.fill_(1.0)
        
        # 如果模块是 nn.Linear 类型，并且具有偏置
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置数据置零
            module.bias.data.zero_()
    ```python`
        def get_metadata(self, labels, start, total_length, offset, get_indices=False):
            # 克隆 labels 张量以避免修改原始数据，创建 metadata 张量
            metadata = labels.clone()
            # 设置 metadata 的第一列为总长度
            metadata[:, 0] = total_length
            # 设置 sample_length 列以匹配当前层级的样本长度
            metadata[:, 2] = int(self.sample_length)
    
            # 设置偏移量，计算偏移量在 token 的索引
            metadata[:, 1:2] = int(offset * self.raw_to_tokens) + int(start * self.raw_to_tokens)
            # 由于 metadata 包含完整的 token_list，只需选择相关的部分
    
            # 设置歌词 token，调用 set_metadata_lyric_tokens 方法
            metadata, indices = self.set_metadata_lyric_tokens(metadata)
            # 根据 get_indices 参数返回 metadata 和 indices
            if get_indices:
                return metadata, indices
            else:
                return metadata
    
        def set_metadata_lyric_tokens(self, labels):
            """
            处理完整的标签，只提取相关的歌词 token，并保持元数据的条件 token。
            """
            # 如果有相关的歌词 token
            if self.nb_relevant_lyric_tokens > 0:
                # 初始化 tokens_list 张量，尺寸为 (labels 行数, nb_relevant_lyric_tokens)，数据类型为 long，设备为 labels 的设备
                tokens_list = torch.zeros(
                    (labels.shape[0], self.nb_relevant_lyric_tokens), dtype=torch.long, device=labels.device
                )
                indices_list = []  # 存储原始数组中每个字符的索引
                # 遍历每一行标签数据
                for idx in range(labels.shape[0]):
                    # 克隆 labels 的所有行，但不包括前四列和元数据嵌入的最大生成数量
                    full_tokens = labels.clone()[:, 4 + self.metadata_embedding.max_nb_genres :]
                    total_length, offset, duration = labels[idx, 0], labels[idx, 1], labels[idx, 2]
                    # 获取相关的歌词 token 和其索引
                    tokens, indices = get_relevant_lyric_tokens(
                        full_tokens, self.nb_relevant_lyric_tokens, total_length, offset, duration
                    )
                    # 将获取的 tokens 存入 tokens_list
                    tokens_list[idx, :] = tokens
                    indices_list.append(indices)
    
                # 返回更新后的 labels 和索引列表，合并原标签的前几列与 tokens_list
                return (
                    torch.cat((labels[:, : 4 + self.metadata_embedding.max_nb_genres], tokens_list), dim=-1),
                    indices_list,
                )
            else:
                # 如果没有相关的歌词 token，直接返回原 labels 和 None
                return labels, None
    
        def get_music_tokens_conds(self, music_tokens, start, end):
            """
            提取当前层级的条件音乐 token。
            """
            # 如果不是第一层级
            if self.level != 0:
                # 获取上一层级的音乐 token 条件
                music_tokens_cond = music_tokens[self.level - 1]
                # 根据 start 和 end 索引提取音乐 token
                music_tokens = music_tokens_cond[:, start // self.cond_downsample : end // self.cond_downsample]
                # 计算缺失的条件长度
                missing_cond_len = self.n_ctx // self.cond_downsample - music_tokens_cond[-1].shape[-1]
                # 如果有缺失的条件长度，填充零
                if missing_cond_len > 0:
                    init_cond = torch.zeros(1, missing_cond_len).to(music_tokens_cond.device)
                    music_tokens_cond = torch.cat((music_tokens_cond, init_cond), dim=-1).long()
                # 返回处理后的音乐 token 条件列表
                music_tokens_conds = [music_tokens_cond]
            else:
                music_tokens_conds = None
            return music_tokens_conds
    def prior_preprocess(self, tokens, conds):
        """
        Shifts the input tokens to account for the dictionary merge. The embed_dim_shift give by how much the music
        tokens should be shifted by. It is equal to `lyric_vocab_size`.
        """
        # 获取批次大小
        batch_size = tokens[0].shape[0]
        # 对每个输入的 token 进行偏移处理
        for i in range(len(tokens)):
            tokens[i] = (tokens[i] + int(self.embed_dim_shift[i])).view(batch_size, -1)

        # 对每个条件进行处理，如果条件为 None，则用零填充
        for i in range(len(conds)):
            if conds[i] is None:
                conds[i] = torch.zeros(
                    (batch_size, self.input_shapes[i], self.width), dtype=tokens[0].dtype, device=tokens[0].device
                )

        # 将处理后的 tokens 和 conds 拼接起来返回
        return torch.cat(tokens, dim=1), torch.cat(conds, dim=1)

    def prior_postprocess(self, tokens):
        """
        Shifts back the input tokens if the model uses an encoder decoder architecture. As the embedding layer is
        shared, `prior_embed_dim_shift` shifts the music token ids by `lyric_vocab_size`. Only returns the music
        tokens.
        """
        # 获取批次大小
        batch_size = tokens.shape[0]
        # 划分 tokens 为列表，按照指定维度进行切分
        dims = (self.input_shapes[0], tokens.shape[1] - self.input_shapes[0])
        tokens = list(torch.split(tokens, dims, dim=1))

        # 对每个切分后的 token 进行逆向处理，将其偏移值减去
        for i in range(len(tokens)):
            bins_shift = int(self.embed_dim_shift[i])
            tokens[i] = (tokens[i] - bins_shift).view(batch_size, -1)
            tokens[i] = torch.clamp(tokens[i], min=0)
            # 如果不屏蔽损失，模型可能生成的歌词/音符 token 可能会被 bin_shift 偏移小于0
        # 返回处理后的最后一个 token
        return tokens[-1]

    def embed_tokens(self, music_tokens_conds):
        """
        Embeds the upper level music tokens and upsamples them to provide as audio conditioning.
        """
        # 仅处理 music_tokens_conds 中指定条件级别以下的内容
        music_tokens_conds = music_tokens_conds[: self.cond_level + 1]
        audio_conditioning = None
        # 对 music_tokens_conds 和条件块进行逆向处理
        for music_tokens_cond, conditioner_block in reversed(list(zip(music_tokens_conds, [self.conditioner_blocks]))):
            audio_conditioning = conditioner_block(music_tokens_cond, audio_conditioning)
        # 返回音频条件化结果
        return audio_conditioning

    def encode(self, hidden_states, start_level=None, end_level=None, bs_chunks=1):
        """
        Encodes the hidden states (raw audio) using the VQVAE's encoder. Returns latent_states.
        """
        # 如果未指定起始级别，则使用默认级别
        if start_level is None:
            start_level = self.level
        # 如果未指定结束级别，则使用默认级别
        if end_level is None:
            end_level = self.levels
        # 使用 VQVAE 编码器获取潜在状态
        with torch.no_grad():
            latent_states = self.vqvae_encoder(
                hidden_states, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks
            )
        # 返回潜在状态
        return latent_states
    # 使用给定的音乐令牌解码成原始音频序列
    def decode(self, music_tokens, start_level=None, end_level=None, bs_chunks=1):
        """
        Usamples the sequence of codebook vectors to a raw audio.
        """
        # 如果未指定起始级别，默认使用对象的级别
        if start_level is None:
            start_level = self.level
        # 如果未指定结束级别，默认使用对象的级别数
        if end_level is None:
            end_level = self.levels
        # 使用禁用梯度环境运行以下代码
        with torch.no_grad():
            # 调用 VQ-VAE 解码器来生成输出
            output = self.vqvae_decoder(
                music_tokens, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks
            )
        return output

    # 获取条件信息，将音乐令牌转换为输入嵌入。将歌词与其余元数据分开，歌词令牌可以为 None。
    def get_cond(self, music_tokens_conds, metadata):
        """
        Converts the input tokens to input_embeddings. Splits the lyrics form the rest of the metadata. Lyric tokens
        can be None.
        """
        # 如果存在元数据，则从中提取标签和歌词令牌
        if metadata is not None:
            n_labels = metadata.shape[1] - self.nb_relevant_lyric_tokens
            metadata, lyric_tokens = metadata[:, :n_labels], metadata[:, n_labels:]
        else:
            # 否则设置元数据和歌词令牌为 None
            metadata, lyric_tokens = None, None
        # 根据是否有元数据条件，生成相应的元数据嵌入和位置编码
        metadata_conditioning, metadata_pos = (
            self.metadata_embedding(metadata) if self.metadata_conditioning else (None, None)
        )
        # 根据音频条件设置，生成音频条件输入嵌入或者使用元数据位置编码
        audio_conditioning = self.embed_tokens(music_tokens_conds) if self.audio_conditioning else metadata_pos
        return audio_conditioning, metadata_conditioning, lyric_tokens

    # 对模型进行采样生成
    def sample(
        self,
        n_samples,
        music_tokens=None,
        music_tokens_conds=None,
        metadata=None,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        chunk_size=None,
        sample_tokens=None,
    ):
        # 该函数在这里不完整，需要在此处添加代码以完成功能

    # 获取编码器状态，提取将由解码器关注的歌词编码器的最后隐藏状态。通过歌词编码器向前传播。
    def get_encoder_states(self, lyric_tokens, sample=False):
        """
        Retreive the last hidden_states of the lyric encoder that will be attended to by the decoder. Forwards through
        the lyric encoder.
        """
        # 如果存在相关歌词令牌且歌词条件为真
        if self.nb_relevant_lyric_tokens != 0 and self.lyric_conditioning:
            # 如果需要进行采样，则将编码器转移到设备上
            if sample:
                self.encoder = self.encoder.to(lyric_tokens.device)
            # 通过编码器生成歌词编码活动
            lyric_acts = self.encoder(lyric_tokens, None, None, None)
            # 将歌词编码活动投影到输入空间
            lyric_acts = self.encoder.proj_in(lyric_acts)
            # 对编码后的结果进行最终的层归一化处理
            last_encoder_hidden_states = self.encoder.final_layer_norm(lyric_acts)
        else:
            # 否则将最终隐藏状态设置为 None
            last_encoder_hidden_states = None
        return last_encoder_hidden_states

    # 获取编码器损失，计算歌词编码器的损失：下一个歌词令牌的预测。
    def get_encoder_loss(self, last_encoder_hidden_states, target_lyrics):
        """
        Computes the loss for the lyric encoder: next lyric token prediction.
        """
        # 如果启用了歌词条件
        if self.lyric_conditioning:
            # 对最终隐藏状态进行语言模型头部处理
            last_encoder_hidden_states = self.encoder.lm_head(last_encoder_hidden_states)
            # 计算交叉熵损失
            encoder_loss = nn.functional.cross_entropy(
                last_encoder_hidden_states.view(-1, self.encoder_dim), target_lyrics.view(-1)
            ) / np.log(2.0)
        else:
            # 否则将编码器损失设置为 0
            encoder_loss = torch.tensor(0.0, device=last_encoder_hidden_states.device)
        return encoder_loss
    def forward_tokens(
        self, music_tokens, music_tokens_conds=[], metadata=None, get_preds=False, get_attn_weights=False
    ):
        """
        Applies a forward pass using the conditioning tokens. Different from the classic forward as it does not use the
        vqvae's encoding layers.
        """
        # 如果需要记录注意力权重，则设置记录
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        
        # 获取音频、元数据条件和歌词 token
        audio_conditioning, metadata_conditioning, lyric_tokens = self.get_cond(music_tokens_conds, metadata)

        # 如果模型是编码-解码器结构
        if self.is_encoder_decoder:
            # 预处理歌词和音乐 token，返回的 tokens 包含歌词和音乐 token，audio_conditioning 也被修改
            tokens, audio_conditioning = self.prior_preprocess(
                [lyric_tokens, music_tokens], [None, audio_conditioning]
            )
            # 使用 prior 模型进行前向传播，包括获取预测值和分离损失
            (encoder_loss, next_token_prediction_loss), preds = self.prior(
                tokens, audio_conditioning, metadata_conditioning, get_sep_loss=True, get_preds=get_preds
            )
        else:
            # 获取最后一个编码器隐藏状态
            last_encoder_hidden_states = self.get_encoder_states(lyric_tokens)
            # 计算编码器损失
            encoder_loss = self.get_encoder_loss(last_encoder_hidden_states, lyric_tokens)
            # 使用 prior 模型进行前向传播，获取下一个 token 预测损失和预测值
            next_token_prediction_loss, preds = self.prior(
                music_tokens,
                audio_conditioning,
                metadata_conditioning,
                last_encoder_hidden_states,
                get_preds=get_preds,
            )
        
        # 计算总损失，包括编码器损失和下一个 token 预测损失
        loss = self.encoder_loss_fraction * encoder_loss * self.nb_relevant_lyric_tokens / self.total_loss_dims
        loss += next_token_prediction_loss * self.next_token_prediction_loss_dims / self.total_loss_dims

        # 定义需要返回的指标
        metrics = {
            "bpd": next_token_prediction_loss.clone().detach(),
            "encoder_loss": encoder_loss.clone().detach(),
            "next_token_prediction_loss": next_token_prediction_loss.clone().detach(),
        }
        
        # 如果需要返回预测值，则加入指标中
        if get_preds:
            metrics["preds"] = preds.clone().detach()
        
        # 如果需要记录注意力权重，将保存的注意力权重返回并关闭记录
        if get_attn_weights:
            saved_attn_weights = self.prior.transformer.saved_attn_weights
            self.prior.transformer.set_record_attn(False)
            return saved_attn_weights
        else:
            # 否则返回计算得到的损失和指标
            return loss, metrics
    ) -> List[torch.Tensor]:
        """
        Encode the hidden states using the `vqvae` encoder, and then predicts the next token in the `forward_tokens`
        function. The loss is the sum of the `encoder` loss and the `decoder` loss.

        Args:
            hidden_states (`torch.Tensor`):
                Hidden states which should be raw audio
            metadata (`List[torch.LongTensor]`, *optional*):
                List containing the metadata conditioning tensor with the lyric and the metadata tokens.
            decode (`bool`, *optional*, defaults to `False`):
                Whether or not to decode the encoded to tokens.
            get_preds (`bool`, *optional*, defaults to `False`):
                Whether or not to return the actual predictions of the model.
        """
        # 获取批处理的大小
        batch_size = hidden_states.shape[0]
        # 使用 `vqvae` 编码器对隐藏状态（原始音频）进行编码，得到音乐 tokens 和可能的条件
        music_tokens, *music_tokens_conds = self.encode(hidden_states, bs_chunks=batch_size)
        # 调用 forward_tokens 函数计算损失和指标
        loss, metrics = self.forward_tokens(
            music_tokens=music_tokens,
            music_tokens_conds=music_tokens_conds,
            metadata=metadata,
            get_preds=get_preds,
        )
        # 如果需要解码，则使用 `vqvae` 解码器进行解码
        if decode:
            dequantised_states = self.decode([music_tokens, *music_tokens_conds])
        else:
            dequantised_states = None
        # 返回解码后的状态、损失和指标
        return dequantised_states, loss, metrics
class JukeboxPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 JukeboxConfig 作为配置类
    config_class = JukeboxConfig
    # 基础模型前缀为 "jukebox"
    base_model_prefix = "jukebox"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        # 如果模块是 JukeboxPrior 或者 JukeboxVQVAE 类型，调用其 _init_weights 方法进行初始化
        if isinstance(module, JukeboxPrior) or isinstance(module, JukeboxVQVAE):
            module.apply(module._init_weights)

    def __init__(self, *inputs, **kwargs):
        # 调用父类的构造函数
        super().__init__(*inputs, **kwargs)


# JUKEBOX_SAMPLING_INPUT_DOCSTRING 是用于描述采样输入的文档字符串常量
JUKEBOX_SAMPLING_INPUT_DOCSTRING = r"""
            labels (`List[torch.LongTensor]` of length `n_sample`, and shape `(self.levels, self.config.max_nb_genre + lyric_sequence_length)` :
                List of metadata such as `artist_id`, `genre_id` and the full list of lyric tokens which are used to
                condition the generation.
            sampling_kwargs (`Dict[Any]`):
                Various additional sampling arguments that are used by the `_sample` function. A detail list of the
                arguments can bee seen in the [`_sample`] function documentation.
"""


@add_start_docstrings(
    """The bare JUKEBOX Model used for music generation. 4 sampling techniques are supported : `primed_sample`, `upsample`,
    `continue_sample` and `ancestral_sample`. It does not have a `forward` method as the training is not end to end. If
    you want to fine-tune the model, it is recommended to use the `JukeboxPrior` class and train each prior
    individually.
    """,
    JUKEBOX_START_DOCSTRING,
)
class JukeboxModel(JukeboxPreTrainedModel):
    _no_split_modules = ["JukeboxBlock"]

    def __init__(self, config):
        # 调用父类构造函数，并传入配置对象
        super().__init__(config)
        # 使用给定的 vqvae_config 初始化 JukeboxVQVAE 对象
        vqvae_config = config.vqvae_config
        self.vqvae = JukeboxVQVAE(vqvae_config)
        # 设置共享参数
        self.set_shared_params(config)
        # 初始化 priors 列表，每个元素为 JukeboxPrior 类的实例
        self.priors = nn.ModuleList(
            [JukeboxPrior(config.prior_configs[level], level) for level in range(config.nb_priors)]
        )

    def set_shared_params(self, model_config):
        """
        Initialises the parameters that are shared. This has to be done here because the list of `JukeboxPriorConfig`
        is nest, and is thus unreachable in the `from_dict` function
        """
        # 遍历 model_config.prior_configs 列表，并为每个配置对象设置共享参数
        for config in model_config.prior_configs:
            config.sampling_rate = model_config.sampling_rate
            config.timing_dims = model_config.timing_dims
            config.min_duration = model_config.min_duration
            config.max_duration = model_config.max_duration
            config.max_nb_genres = model_config.max_nb_genres
            config.metadata_conditioning = model_config.metadata_conditioning

    def decode(self, music_tokens, start_level=0, end_level=None, bs_chunks=1):
        # 调用 vqvae 对象的 decode 方法进行音乐解码
        return self.vqvae.decode(music_tokens, start_level, end_level, bs_chunks)
    # 调用 VQ-VAE 模型的 encode 方法对输入音频进行编码
    def encode(self, input_audio, start_level=0, end_level=None, bs_chunks=1):
        return self.vqvae.encode(input_audio, start_level, end_level, bs_chunks)

    # 将对象 obj 拆分成大小为 split_size 的 batch，总共 n_samples 个样本
    def split_batch(self, obj, n_samples, split_size):
        # 计算需要多少个 passes 才能处理完所有样本
        n_passes = (n_samples + split_size - 1) // split_size
        if isinstance(obj, torch.Tensor):  # 如果 obj 是 torch.Tensor 类型
            return torch.split(obj, split_size, dim=0)  # 在 dim=0 上拆分 Tensor
        elif isinstance(obj, list):  # 如果 obj 是 list 类型
            # 对 list 中的每个元素分别在 dim=0 上进行拆分，并将结果打包成 list 返回
            return list(zip(*[torch.split(item, split_size, dim=0) for item in obj]))
        elif obj is None:  # 如果 obj 为 None
            # 返回包含 n_passes 个 None 的列表
            return [None] * n_passes
        else:
            # 抛出类型错误异常
            raise TypeError("Unknown input type")

    # 在 level 层级上，从 music_tokens 中采样一个长度小于 n_ctx 的部分窗口，新增 tokens_to_sample 个新标记
    def sample_partial_window(
        self, music_tokens, labels, offset, sampling_kwargs, level, tokens_to_sample, max_batch_size
    ):
        prior = self.priors[level]  # 获取指定层级的 prior 模型
        sampled_tokens = music_tokens[level]  # 获取在指定层级的音乐 tokens
        n_ctx = prior.n_ctx  # 获取 prior 模型的上下文长度
        nb_sampled_tokens = sampled_tokens.shape[1]  # 获取已采样 tokens 的数量

        if nb_sampled_tokens < n_ctx - tokens_to_sample:
            # 如果已采样 tokens 的数量小于 n_ctx - tokens_to_sample
            sampling_kwargs["sample_tokens"] = nb_sampled_tokens + tokens_to_sample
            start = 0
        else:
            # 否则设置采样的 tokens 数量为 n_ctx
            sampling_kwargs["sample_tokens"] = n_ctx
            start = nb_sampled_tokens - n_ctx + tokens_to_sample

        # 调用 sample_single_window 方法进行单个窗口的采样
        return self.sample_single_window(music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size)

    # 在 level 层级上，从 start 位置开始采样一个长度为 n_ctx 的单个窗口
    # 从先验分布中采样一个单窗口的音乐序列片段
    def sample_single_window(self, music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size):
        # 获取当前层级的先验分布
        prior = self.priors[level]
        # 获取音乐片段的总数
        n_samples = music_tokens[0].shape[0]
        # 获取先验分布中的上下文长度
        n_ctx = prior.n_ctx
        # 计算当前片段的结束位置
        end = start + n_ctx
        # 获取已经在当前层级采样的音乐片段
        previous_sampled_tokens = music_tokens[level][:, start:end]

        # 从采样参数中获取要采样的令牌数
        sample_tokens = sampling_kwargs.get("sample_tokens", None)
        if "sample_tokens" in sampling_kwargs:
            sample_tokens = end - start

        # 计算当前条件下的令牌数量
        conditioning_tokens = previous_sampled_tokens.shape[1]
        # 计算新采样的令牌数量
        new_tokens = sample_tokens - previous_sampled_tokens.shape[1]

        # 记录采样信息日志
        logger.info(
            f"Sampling {sample_tokens} tokens for [{start},{start+sample_tokens}]. Conditioning on"
            f" {conditioning_tokens} tokens"
        )

        # 如果没有新的令牌需要采样，则直接返回原始音乐令牌
        if new_tokens <= 0:
            return music_tokens

        # 获取上一层级的音乐令牌条件
        music_tokens_conds = prior.get_music_tokens_conds(music_tokens, start, end)
        # 如果没有上一层级，应该返回None！

        # 设置元数据的偏移量、采样长度和歌词令牌
        metadata = prior.get_metadata(labels, start, self.total_length, offset)

        # 将音乐令牌、音乐令牌条件和元数据拆分成批次
        music_tokens_list = self.split_batch(previous_sampled_tokens, n_samples, max_batch_size)
        music_tokens_conds_list = self.split_batch(music_tokens_conds, n_samples, max_batch_size)
        metadata_list = self.split_batch(metadata, n_samples, max_batch_size)
        tokens = []
        # 迭代处理每个批次的音乐令牌和条件
        iterator = tqdm(zip(music_tokens_list, music_tokens_conds_list, metadata_list), leave=False)
        for music_tokens_i, music_tokens_conds_i, metadata_i in iterator:
            # 确定当前使用的名称（"祖先"或"主导"），基于是否有音乐令牌条件
            name = ["Ancestral", "Primed"][music_tokens_i.shape[1] == 0]
            iterator.set_description(
                f"[prior level {level}] {name} Sampling {sample_tokens} tokens out of"
                f" {self.total_length//prior.raw_to_tokens}",
                refresh=True,
            )
            # 从先验分布中采样音乐令牌
            tokens_i = prior.sample(
                n_samples=music_tokens_i.shape[0],
                music_tokens=music_tokens_i,
                music_tokens_conds=music_tokens_conds_i,
                metadata=metadata_i,
                **sampling_kwargs,
            )
            tokens.append(tokens_i)
        # 将所有采样的音乐令牌连接起来
        sampled_tokens = torch.cat(tokens, dim=0)

        # 更新音乐令牌序列，加入新的采样片段
        music_tokens_new = sampled_tokens[:, -new_tokens:]
        music_tokens[level] = torch.cat([music_tokens[level], music_tokens_new], dim=1)
        return music_tokens

    # 以指定级别、总长度和跳跃长度进行采样
    def sample_level(
        self, music_tokens, labels, offset, sampling_kwargs, level, total_length, hop_length, max_batch_size
    ):
        # 如果总长度超过当前先验模型的上下文长度
        if total_length >= self.priors[level].n_ctx:
            # 获取起始位置迭代器，根据指定的步长和先验模型的上下文长度
            iterator = get_starts(total_length, self.priors[level].n_ctx, hop_length)
            # 对于迭代器中的每个起始位置
            for start in iterator:
                # 对单个窗口进行采样
                music_tokens = self.sample_single_window(
                    music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size
                )

        else:
            # 对部分窗口进行采样，因为总长度小于当前先验模型的上下文长度
            music_tokens = self.sample_partial_window(
                music_tokens, labels, offset, sampling_kwargs, level, total_length, max_batch_size
            )
        # 返回采样后的音乐 tokens
        return music_tokens

    @torch.no_grad()
    def _sample(
        self,
        music_tokens,
        labels,
        sample_levels,
        metas=None,
        chunk_size=32,
        sampling_temperature=0.98,
        lower_batch_size=16,
        max_batch_size=16,
        sample_length_in_seconds=24,
        compute_alignments=False,
        sample_tokens=None,
        offset=0,
        save_results=True,
        sample_length=None,
    ):
        # 添加文档字符串作为函数注释，描述生成音乐 tokens 的过程
        @add_start_docstrings(
            """
            Generates music tokens based on the provided `labels. Will start at the desired prior level and automatically
            upsample the sequence. If you want to create the audio, you should call `model.decode(tokens)`, which will use
            the VQ-VAE decoder to convert the music tokens to raw audio.

            Args:
                labels (`List[torch.LongTensor]`) :
                    List of length `n_sample`, and shape `(self.levels, 4 + self.config.max_nb_genre +
                    lyric_sequence_length)` metadata such as `artist_id`, `genre_id` and the full list of lyric tokens
                    which are used to condition the generation.
                n_samples (`int`, *optional*, default to 1) :
                    Number of samples to be generated in parallel.
            """,
        )
    def ancestral_sample(self, labels, n_samples=1, **sampling_kwargs) -> List[torch.LongTensor]:
        """
        Example:

        ```python
        >>> from transformers import AutoTokenizer, JukeboxModel, set_seed

        >>> model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/jukebox-1b-lyrics")

        >>> lyrics = "Hey, are you awake? Can you talk to me?"
        >>> artist = "Zac Brown Band"
        >>> genre = "Country"
        >>> metas = tokenizer(artist=artist, genres=genre, lyrics=lyrics)
        >>> set_seed(0)
        >>> music_tokens = model.ancestral_sample(metas.input_ids, sample_length=400)

        >>> with torch.no_grad():
        ...     model.decode(music_tokens)[:, :10].squeeze(-1)
        tensor([[-0.0219, -0.0679, -0.1050, -0.1203, -0.1271, -0.0936, -0.0396, -0.0405,
            -0.0818, -0.0697]])
        ```
        """

        # 从参数中获取采样层级列表，如果没有则使用默认值（self.priors 的长度）
        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors))))
        # 初始化一个空的音乐 tokens 列表，用于存储采样后的结果
        music_tokens = [
            torch.zeros(n_samples, 0, dtype=torch.long, device=labels[0].device) for _ in range(len(self.priors))
        ]
        # 使用 _sample 方法进行采样生成音乐 tokens
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        # 返回生成的音乐 tokens 列表
        return music_tokens

    @add_start_docstrings(
        """Generates a continuation of the previously generated tokens.

        Args:
            music_tokens (`List[torch.LongTensor]` of length `self.levels` ) :
                A sequence of music tokens which will be used as context to continue the sampling process. Should have
                `self.levels` tensors, each corresponding to the generation at a certain level.
        """,
        JUKEBOX_SAMPLING_INPUT_DOCSTRING,
    )
    def continue_sample(self, music_tokens, labels, **sampling_kwargs) -> List[torch.LongTensor]:
        # 从参数中获取采样层级列表，如果没有则使用默认值（self.priors 的长度）
        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors))))
        # 使用 _sample 方法继续生成音乐 tokens
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        # 返回生成的音乐 tokens 列表
        return music_tokens

    @add_start_docstrings(
        """Upsamples a sequence of music tokens using the prior at level `level`.

        Args:
            music_tokens (`List[torch.LongTensor]` of length `self.levels` ) :
                A sequence of music tokens which will be used as context to continue the sampling process. Should have
                `self.levels` tensors, each corresponding to the generation at a certain level.
        """,
        JUKEBOX_SAMPLING_INPUT_DOCSTRING,
    )
    def upsample(self, music_tokens, labels, **sampling_kwargs) -> List[torch.LongTensor]:
        # 从参数中获取采样层级列表，如果没有则使用默认值（self.priors 的长度减一）
        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors) - 1)))
        # 使用 _sample 方法上采样生成音乐 tokens
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        # 返回生成的音乐 tokens 列表
        return music_tokens
    @add_start_docstrings(
        """Generate a raw audio conditioned on the provided `raw_audio` which is used as conditioning at each of the
        generation levels. The audio is encoded to music tokens using the 3 levels of the VQ-VAE. These tokens are
        used: as conditioning for each level, which means that no ancestral sampling is required.

        Args:
            raw_audio (`List[torch.Tensor]` of length `n_samples` ) :
                A list of raw audio that will be used as conditioning information for each samples that will be
                generated.
        """,
        JUKEBOX_SAMPLING_INPUT_DOCSTRING,
    )

这是一个装饰器函数，用于给 `primed_sample` 方法添加文档字符串。文档字符串描述了函数的作用、参数和返回值。


    def primed_sample(self, raw_audio, labels, **sampling_kwargs) -> List[torch.LongTensor]:

定义了一个名为 `primed_sample` 的方法，用于生成基于提供的 `raw_audio` 条件的原始音频。返回一个列表，其中每个元素是包含音乐 token 的 torch LongTensor。


        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors))))

从 `sampling_kwargs` 中获取 `sample_levels` 参数，如果不存在则默认为 `self.priors` 的长度范围内的列表。


        self.vqvae.to(raw_audio.device).float()

将 `self.vqvae` 移动到 `raw_audio` 的设备上，并将其数据类型转换为 float。


        with torch.no_grad():

进入一个禁用梯度跟踪的上下文管理器，以确保在该部分代码中不会进行梯度计算。


            music_tokens = self.vqvae.encode(
                raw_audio, start_level=0, end_level=len(self.priors), bs_chunks=raw_audio.shape[0]
            )

使用 `self.vqvae` 对 `raw_audio` 进行编码，生成音乐 token。使用从 0 到 `len(self.priors)` 的级别作为起始和结束级别，并根据 `raw_audio` 的形状分块处理。


        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)

调用 `_sample` 方法，使用 `music_tokens`、`labels` 和 `sample_levels` 进行采样，传递额外的 `sampling_kwargs`。


        return music_tokens

返回生成的音乐 token 列表。
```