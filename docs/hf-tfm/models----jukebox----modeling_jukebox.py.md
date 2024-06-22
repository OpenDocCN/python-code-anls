# `.\models\jukebox\modeling_jukebox.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The OpenAI Team Authors 和 HuggingFace Inc. team 所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""PyTorch Jukebox model."""

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# Jukebox 预训练模型存档列表
JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/jukebox-1b-lyrics",
    "openai/jukebox-5b-lyrics",
    # 查看所有 Jukebox 模型：https://huggingface.co/models?filter=jukebox
]

# 过滤 logits 分布，使用 top-k 和/或 nucleus (top-p) 过滤
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
    # 克隆 logits 张量
    logits = logits.clone()
    # 安全检查，确保 top_k 不超过 logits 的最后一个维度大小
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        # 移除所有概率小于 top-k 中最后一个标记的标记
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]
        logits[indices_to_remove] = filter_value
    # 如果 top_p 大于 0.0，则执行以下操作
    if top_p > 0.0:
        # 对 logits 进行降序排序，并返回排序后的值和索引
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # 计算累积概率
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的标记
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引向右移动一位，以保留超过阈值的第一个标记
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 根据排序后的索引和移除标记，生成需要移除的索引
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        # 将需要移除的标记值替换为过滤值
        logits[indices_to_remove] = filter_value
    # 返回处理后的 logits
    return logits
# 从完整的歌词标记中提取相关的标记，返回最多 `max_n_lyric_tokens` 个标记。如果提供的标记序列较小，将进行填充；否则，只返回从中点 - `max_n_lyric_tokens//2` 到中点 + `max_n_lyric_tokens//2` 的字符。这样*聚焦*于序列中最相关的标记（按时间）。

def get_relevant_lyric_tokens(full_tokens, max_n_lyric_tokens, total_length, offset, duration):
    """
    从完整的歌词标记中提取相关的标记，返回最多 `max_n_lyric_tokens` 个标记。如果提供的标记序列较小，将进行填充；否则，只返回从中点 - `max_n_lyric_tokens//2` 到中点 + `max_n_lyric_tokens//2` 的字符。这样*聚焦*于序列中最相关的标记（按时间）。

    Args:
        full_tokens (`List[int]`):
            包含整个歌词标记的列表。
        total_length (`int`):
            音乐的总预期长度（并非所有都生成，参见 duration），以样本为单位。
        offset (`int`):
            音乐中的起始样本。如果偏移大于 0，歌词将被移动以考虑这一点。
        duration (`int`):
            生成音乐的预期持续时间，以样本为单位。持续时间必须小于总长度，表示信号的整体长度。
    """
    full_tokens = full_tokens[0]
    如果完整标记长度小于 max_n_lyric_tokens，则进行填充
    if len(full_tokens) < max_n_lyric_tokens:
        tokens = torch.cat(
            [torch.zeros(max_n_lyric_tokens - len(full_tokens), dtype=torch.long).to(full_tokens.device), full_tokens]
        )
        indices = [-1] * (max_n_lyric_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
    否则，只返回最相关的标记
    else:
        midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
        midpoint = min(max(midpoint, max_n_lyric_tokens // 2), len(full_tokens) - max_n_lyric_tokens // 2)
        tokens = full_tokens[midpoint - max_n_lyric_tokens // 2 : midpoint + max_n_lyric_tokens // 2]
        indices = list(range(midpoint - max_n_lyric_tokens // 2, midpoint + max_n_lyric_tokens // 2)
    返回标记的张量和索引

    return tokens.unsqueeze(dim=0), indices


# 将 total_length 分解为大小为 n_ctx 的窗口，以 hop_length 分隔
def get_starts(total_length, n_ctx, hop_length):
    starts = []
    遍历每个窗口的起始位置
    for start in range(0, total_length - n_ctx + hop_length, hop_length):
        如果起始位置加上 n_ctx 大于等于总长度，则将最后一个窗口的大小设为 n_ctx 以最大化上下文
        if start + n_ctx >= total_length:
            start = total_length - n_ctx
        添加起始位置到列表中
        starts.append(start)
    返回所有窗口的起始位置列表

    return starts


# 获取对齐
def get_alignment(music_tokens, labels, prior, config):
    level = prior.levels - 1  # 使用的顶层
    n_ctx = prior.n_ctx
    tokens = music_tokens[level]
    batch_size, total_length = tokens.shape[0], tokens.shape[1]
    如果总长度小于 n_ctx，则进行填充
    if total_length < n_ctx:
        padding_length = n_ctx - total_length
        tokens = torch.cat(
            [tokens, torch.zeros(batch_size, n_ctx - total_length, dtype=tokens.dtype, device=tokens.device)], dim=1
        )
        total_length = tokens.shape[1]
    否则，填充长度为 0
    else:
        padding_length = 0

    计算跳跃长度
    hop_length = int(config.hop_fraction[-level - 1] * prior.n_ctx)
    # 从配置中获取对齐头和对齐层
    alignment_head, alignment_layer = config.prior_alignment_head[0], config.prior_alignment_layer[0]
    # 创建包含对齐层的集合
    attn_layers = {alignment_layer}
    # 初始化对齐跳数和索引跳数字典
    alignment_hops = {}
    indices_hops = {}
    # 遍历每个起始位置，计算歌词到音乐的对齐
    for start in tqdm(get_starts(total_length, n_ctx, hop_length), desc="Computing lyric to music alignment "):
        end = start + n_ctx
        # 设置元数据偏移、采样长度和歌词标记
        metadata, indices_hop = prior.get_metadata(labels, start, config.sample_length, get_indices=True, offset=0)
        # 将 tokens 和 metadata 分块
        tokens_bs = torch.chunk(tokens, batch_size, dim=0)
        metadata_bs = torch.chunk(metadata, batch_size, dim=0)
        w_hops = []
        # 遍历分块的 tokens 和 metadata，计算注意力权重
        for tokens_i, metadata_i in zip(tokens_bs, metadata_bs):
            w_hop = prior.forward_tokens(tokens_i[:, start:end], [], metadata_i, get_attn_weights=attn_layers)
            w_hops.append(w_hop[0][:, alignment_head])
            del w_hop
        # 拼接权重并转换为 numpy 数组
        weights = torch.cat(w_hops, dim=0)
        del w_hops
        alignment_hop = weights.float().cpu().numpy()
        del weights

        # alignment_hop 的形状为 (bs, n_ctx, nb_relevant_lyric_tokens)
        # indices_hop 是长度为 bs 的列表，每个条目长度为 hps.nb_relevant_lyric_tokens
        indices_hops[start] = indices_hop
        alignment_hops[start] = alignment_hop

    # 将每个跳的注意力合并为整个范围的注意力
    # 使用索引将它们放入相应源 tokens 的正确位置
    alignments = []
    for item in range(batch_size):
        # 注意每个项目具有不同长度的歌词
        full_tokens = labels[0, 3:]
        alignment = np.zeros((total_length, len(full_tokens) + 1))
        for start in reversed(get_starts(total_length, n_ctx, hop_length)):
            end = start + n_ctx
            alignment_hop = alignment_hops[start][item]
            indices = indices_hops[start][item]
            alignment[start:end, indices] = alignment_hop
        alignment = alignment[: total_length - padding_length, :-1]  # 移除标记填充和最后一个歌词索引
        alignments.append(alignment)
    return alignments
def save_temp_audio(fname, lvl, metas, aud):
    # 将音频数据限制在 -1 到 1 之间，并转换为 numpy 数组
    aud = torch.clamp(aud, -1, 1).cpu().numpy()
    # 遍历音频数据的第一维度
    for i in list(range(aud.shape[0])):
        # 如果存在元数据
        if metas is not None:
            # 获取艺术家、流派、歌词信息
            artists, genres, lyrics = list(metas)[i].values()
            # 构建保存路径
            path = f"{fname}/lvl_{lvl}-{artists}-{genres}-{lyrics[:5]}-{i}"
            # 保存音频数据为 numpy 文件
            np.save(path, aud[i])
        else:
            # 如果不存在元数据，直接保存音频数据为 numpy 文件
            np.save(f"{fname}/lvl_{lvl}-sample-{i}", aud[i])


def get_mask(mask, query_length, key_value_length, blocks, spread, device, sample, sample_t):
    # 返回形状为 1 x 1 x query_length x key_value_length 的掩码，如果不需要掩码则返回 None
    if mask is None or query_length == 1:
        return None
    offset = sample_t - query_length if sample else max(key_value_length - query_length, 0)
    if mask == "autoregressive":
        # 自回归掩码
        mask = torch.ones(query_length, key_value_length, device=device).tril(offset)
    elif mask == "summary":
        # 摘要掩码
        mask = torch.ones(query_length, query_length, device=device).tril()
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
        mask = torch.ones(query_length, key_value_length, device=device).tril(offset)
    return mask.view(1, 1, query_length, key_value_length)


class JukeboxConv1D(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        weight = torch.empty(input_width, output_width)
        bias = torch.zeros(output_width)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, hidden_states):
        size_out = (*hidden_states.size()[:-1], self.output_width)
        # 线性变换
        hidden_states = torch.addmm(
            self.bias.type_as(hidden_states),
            hidden_states.view(-1, hidden_states.size(-1)),
            self.weight.type_as(hidden_states),
        )
        hidden_states = hidden_states.view(*size_out)
        return hidden_states


class JukeboxResConv1DBlock(nn.Module):
    def __init__(self, config, conv_width, depth=1, res_scale=1.0):
        super().__init__()
        hidden_dim = config.res_convolution_multiplier * conv_width
        dilation = config.res_dilation_growth_rate**depth
        padding = dilation

        self.res_scale = res_scale
        self.activation = nn.ReLU()
        # 第一个卷积层
        self.conv1d_1 = nn.Conv1d(conv_width, hidden_dim, 3, 1, padding, dilation)
        # 第二个卷积层
        self.conv1d_2 = nn.Conv1d(hidden_dim, conv_width, 1, 1, 0)
    # 前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 保存输入的残差
        residuals = hidden_states
        # 使用激活函数处理隐藏状态
        hidden_states = self.activation(hidden_states)
        # 使用第一个一维卷积层处理隐藏状态
        hidden_states = self.conv1d_1(hidden_states)
        # 再次使用激活函数处理隐藏状态
        hidden_states = self.activation(hidden_states)
        # 使用第二个一维卷积层处理隐藏状态
        hidden_states = self.conv1d_2(hidden_states)
        # 返回残差加上残差比例乘以处理后的隐藏状态
        return residuals + self.res_scale * hidden_states
# 定义一个基于 ResNet 结构的 1D 卷积神经网络模型
class JukeboxResnet1D(nn.Module):
    def __init__(self, config, conv_width, n_depth, reverse_dilation=False):
        super().__init__()
        self.dilation_cycle = config.res_dilation_cycle
        res_scale = 1.0 if not config.conv_res_scale else 1.0 / math.sqrt(n_depth)

        blocks = []
        # 循环创建 ResNet 卷积块
        for depth in range(n_depth):
            block_depth = depth if self.dilation_cycle is None else depth % self.dilation_cycle
            blocks.append(JukeboxResConv1DBlock(config, conv_width, block_depth, res_scale))

        # 如果需要反向循环扩张，则反转卷积块列表
        if reverse_dilation:
            blocks = blocks[::-1]
        self.resnet_block = nn.ModuleList(blocks)

    def forward(self, hidden_states):
        # 对每个 ResNet 卷积块进行前向传播
        for block in self.resnet_block:
            hidden_states = block(hidden_states)
        return hidden_states


# 定义一个编码器卷积块
class JukeboxEncoderConvBlock(nn.Module):
    def __init__(self, config, embed_dim, hidden_dim, depth, down_t, stride_t):
        super().__init__()
        blocks = []
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        if down_t > 0:
            # 创建下采样卷积块
            for i in range(down_t):
                blocks.append(nn.Conv1d(embed_dim if i == 0 else hidden_dim, hidden_dim, filter_t, stride_t, pad_t))
                blocks.append(JukeboxResnet1D(config, hidden_dim, depth))
        self.proj_out = nn.Conv1d(hidden_dim, config.embed_dim, 3, 1, 1)
        self.downsample_block = nn.ModuleList(blocks)

    def forward(self, hidden_states):
        # 对每个下采样卷积块进行前向传播
        for block in self.downsample_block:
            hidden_states = block(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


# 定义一个编码器模型
class JukeboxEncoder(nn.Module):
    def __init__(self, config, width, depth, levels, downs_t, strides_t):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList()

        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        # 创建多层编码器卷积块
        for i, down_t, stride_t in iterator:
            self.level_blocks.append(
                JukeboxEncoderConvBlock(
                    config, config.conv_input_shape if i == 0 else config.embed_dim, width, depth, down_t, stride_t
                )
            )

    def forward(self, hidden_states):
        all_hidden_states = []

        # 对每个编码器卷积块进行前向���播
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            hidden_states = level_block(hidden_states)
            all_hidden_states.append(hidden_states)

        return all_hidden_states


# 定义一个解码器卷积块
class JukeboxDecoderConvBock(nn.Module):
    # 初始化函数，接受配置、嵌入维度、隐藏层维度、深度、下采样次数、下采样步幅、是否反向膨胀为参数
    def __init__(self, config, embed_dim, hidden_dim, depth, down_t, stride_t, reverse_dilation=True):
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置隐藏层维度
        self.hidden_dim = hidden_dim
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个空列表用于存储模块
        blocks = []
        # 如果下采样次数大于 0
        if down_t > 0:
            # 计算滤波器大小
            filter_t = stride_t * 2
            # 计算填充大小
            pad_t = stride_t // 2
            # 创建输入投影层
            self.proj_in = nn.Conv1d(embed_dim, hidden_dim, 3, 1, 1)
            # 循环创建下采样块
            for i in range(down_t):
                # 添加 JukeboxResnet1D 模块到列表
                blocks.append(JukeboxResnet1D(config, hidden_dim, depth, reverse_dilation))
                # 添加反卷积层到列表
                blocks.append(
                    nn.ConvTranspose1d(
                        hidden_dim, hidden_dim if i < down_t - 1 else embed_dim, filter_t, stride_t, pad_t
                    )
                )
        # 将模块列表转换为模块容器
        self.upsample_block = nn.ModuleList(blocks)

    # 前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 将隐藏状态投影到隐藏层维度
        hidden_states = self.proj_in(hidden_states)
        # 遍历上采样模块列表
        for block in self.upsample_block:
            # 将隐藏状态输入到当前模块中，更新隐藏状态
            hidden_states = block(hidden_states)
        # 返回最终的隐藏状态
        return hidden_states
# 定义一个 JukeboxDecoder 类，继承自 nn.Module 类
class JukeboxDecoder(nn.Module):
    # 初始化函数，接收配置、隐藏维度、深度、层级数、下采样、步长参数
    def __init__(self, config, hidden_dim, depth, levels, downs_t, strides_t):
        # 调用父类的初始化函数
        super().__init__()
        # 设置对象的层级数属性
        self.levels = levels
        # 创建一个空的模块列表
        self.level_blocks = nn.ModuleList()
        # 遍历层级数、下采样、步长参数，并添加 JukeboxDecoderConvBock 到模块列表中
        for level, down_t, stride_t in zip(list(range(self.levels)), downs_t, strides_t):
            self.level_blocks.append(
                JukeboxDecoderConvBock(config, config.embed_dim, hidden_dim, depth, down_t, stride_t)
            )

        # 初始化输出层
        self.out = nn.Conv1d(config.embed_dim, config.conv_input_shape, 3, 1, 1)

    # 前向传播函数，接收隐藏状态和是否使用所有层级作为输入
    def forward(self, hidden_states, all_levels=True):
        # 设置隐藏状态为输入隐藏状态的最后一个
        hidden_state = hidden_states[-1]

        # 逆向遍历层级数
        for level in reversed(range(self.levels)):
            # 获取当前层级的模块
            level_block = self.level_blocks[level]
            # 使用当前层级模块处理隐藏状态
            hidden_state = level_block(hidden_state)

            # 如果不是第一层且需要使用所有层级，则将隐藏状态与前一层的隐藏状态相加
            if level != 0 and all_levels:
                hidden_state = hidden_state + hidden_states[level - 1]

        # 使用输出层处理隐藏状态并返回结果
        hidden_state = self.out(hidden_state)
        return hidden_state


# 定义一个 JukeboxBottleneckBlock 类，继承自 nn.Module 类
class JukeboxBottleneckBlock(nn.Module):
    # 初始化函数，接收 JukeboxVQVAEConfig 类型的配置
    def __init__(self, config: JukeboxVQVAEConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置离散编码数、码书宽度、mu、阈值、初始化标志、码书总和、码书元素为对应属性
        self.nb_discrete_codes = config.nb_discrete_codes
        self.codebook_width = config.embed_dim
        self.mu = config.lmu
        self.threshold = 1.0
        self.init = False
        self.codebook_sum = None
        self.codebook_elem = None
        # 注册码书缓冲区
        self.register_buffer("codebook", torch.zeros(self.nb_discrete_codes, self.codebook_width))

    # 隐藏状态复制函数，接收隐藏状态并进行复制处理
    def _tile(self, hidden_states):
        # 获取维度和嵌入宽度
        dim, embed_width = hidden_states.shape
        # 如果维度小于离散编码数，则进行复制处理
        if dim < self.nb_discrete_codes:
            n_repeats = (self.nb_discrete_codes + dim - 1) // dim
            std = 0.01 / np.sqrt(embed_width)
            hidden_states = hidden_states.repeat(n_repeats, 1)
            hidden_states = hidden_states + torch.randn_like(hidden_states) * std
        return hidden_states

    # 初始化码书函数，接收隐藏状态并初始化码书
    def init_codebook(self, hidden_states):
        # 获取离散编码数
        nb_discrete_codes = self.nb_discrete_codes
        # 设置初始化标志为真
        self.init = True
        # 对隐藏状态进行复制处理
        codes = self._tile(hidden_states)
        # 从复制后的隐藏状态中获取编码并初始化码书
        self.codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]
        self.codebook_sum = self.codebook
        self.codebook_elem = torch.ones(nb_discrete_codes, device=self.codebook.device)
    def update_codebook(self, hidden_states, latent_states):
        # 获取模型参数
        mu, codebook_width, nb_discrete_codes = self.mu, self.codebook_width, self.nb_discrete_codes
        # 不进行梯度更新
        with torch.no_grad():
            # 计算新的聚类中心
            # nb_discrete_codes, batch_size * seq_length
            # 将离散状态转换为 one-hot 编码
            latent_states_onehot = torch.zeros(nb_discrete_codes, hidden_states.shape[0], device=hidden_states.device)
            latent_states_onehot.scatter_(0, latent_states.view(1, hidden_states.shape[0]), 1)

            # 计算每个聚类中心的加权和
            _codebook_sum = torch.matmul(latent_states_onehot, hidden_states)
            # 统计每个聚类中心的样本数量
            _codebook_elem = latent_states_onehot.sum(dim=-1)  # nb_discrete_codes
            # 对隐藏状态进行复制和打乱顺序得到一个随机聚类中心
            codes = self._tile(hidden_states)
            _random_codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]

            # 更新聚类中心
            old_codebook = self.codebook
            self.codebook_sum = mu * self.codebook_sum + (1.0 - mu) * _codebook_sum
            self.codebook_elem = mu * self.codebook_elem + (1.0 - mu) * _codebook_elem  # nb_discrete_codes
            usage = (self.codebook_elem.view(nb_discrete_codes, 1) >= self.threshold).float()

            # 对聚类中心进行归一化
            norm_code = self.codebook_sum.view(nb_discrete_codes, codebook_width) / self.codebook_elem.view(
                nb_discrete_codes, 1
            )
            # 根据阈值来更新聚类中心
            self.codebook = usage * (norm_code) + (1 - usage) * _random_codebook
            # 计算聚类中心的分布熵
            _codebook_prob = _codebook_elem / torch.sum(_codebook_elem)  # prob of each bin
            entropy = -torch.sum(_codebook_prob * torch.log(_codebook_prob + 1e-8))  # entropy ie how diverse
            # 统计当前使用的聚类中心数量
            used_curr = (_codebook_elem >= self.threshold).sum()
            # 统计使用的聚类中心的数量
            usage = torch.sum(usage)
            # 计算聚类中心更新的幅度
            dk = torch.norm(self.codebook - old_codebook) / np.sqrt(np.prod(old_codebook.shape))
        # 返回更新结果
        return {"entropy": entropy, "used_curr": used_curr, "usage": usage, "dk": dk}

    def preprocess(self, hidden_states):
        # 调整 hidden_states 的维度
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if hidden_states.shape[-1] == self.codebook_width:
            # 计算 hidden_states 的预归一化范数
            prenorm = torch.norm(hidden_states - torch.mean(hidden_states)) / np.sqrt(np.prod(hidden_states.shape))
        elif hidden_states.shape[-1] == 2 * self.codebook_width:
            # 如果 hidden_states 的维度是两倍的聚类中心宽度
            x1, x2 = hidden_states[..., : self.codebook_width], hidden_states[..., self.codebook_width :]
            # 分别计算两部分 hidden_states 的预归一化范数再相加
            prenorm = (torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (
                torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape))
            )

            # 对 hidden_states 进行归一化
            hidden_states = x1 + x2

        # 返回预处理后的 hidden_states 和预归一化范数
        return hidden_states, prenorm
    # 对latent_states和dequantised_states进行后处理，并改变形状
    def postprocess(self, latent_states, dequantised_states, x_shape):
        batch_size, time = x_shape
        dequantised_states = dequantised_states.view(batch_size, time, -1).permute(0, 2, 1).contiguous()
        latent_states = latent_states.view(batch_size, time)
        return latent_states, dequantised_states

    # 对latent_states进行量化
    def quantise(self, latent_states):
        # 计算latent_states和codebook的距离
        codebook_weights = self.codebook.t()
        distance = (
            torch.sum(latent_states**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(latent_states, codebook_weights)
            + torch.sum(codebook_weights**2, dim=0, keepdim=True)
        )  # (batch_size * latent_states , codebook_weights)
        min_distance, music_tokens = torch.min(distance, dim=-1)
        fit = torch.mean(min_distance)
        return music_tokens, fit

    # 对music_tokens进行反量化
    def dequantise(self, music_tokens):
        dequantised_states = F.embedding(music_tokens, self.codebook)
        return dequantised_states

    # 对latent_states进行编码
    def encode(self, latent_states):
        samples, _, seq_len = latent_states.shape

        # 预处理
        latent_states, _ = self.preprocess(latent_states)

        # 量化
        music_tokens, _ = self.quantise(latent_states)

        # 后处理
        music_tokens = music_tokens.view(samples, seq_len)
        return music_tokens

    # 对music_tokens进行解码
    def decode(self, music_tokens):
        samples, seq_len = music_tokens.shape

        # 反量化
        dequantised_states = self.dequantise(music_tokens)

        # 后处理
        dequantised_states = (
            dequantised_states.view(samples, seq_len, self.codebook_width).permute(0, 2, 1).contiguous()
        )
        return dequantised_states

    # 前向传播过程
    def forward(self, hidden_states, update_codebook=True):
        samples, _, seq_len = hidden_states.shape

        # 预处理
        hidden_states, prenorm = self.preprocess(hidden_states)

        # 如果未初始化codebook，则进行初始化
        if update_codebook and not self.init:
            self.init_codebook(hidden_states)

        # 通过瓶颈进行量化和反量化
        music_tokens, fit = self.quantise(hidden_states)
        dequantised_states = self.dequantise(music_tokens)

        # 更新嵌入
        if update_codebook:
            update_metrics = self.update_codebook(hidden_states, music_tokens)
        else:
            update_metrics = {}

        # 计算损失
        commit_loss = torch.norm(dequantised_states.detach() - hidden_states) ** 2 / np.prod(hidden_states.shape)

        # 通过
        dequantised_states = hidden_states + (dequantised_states - hidden_states).detach()

        # 后处理
        music_tokens, dequantised_states = self.postprocess(music_tokens, dequantised_states, (samples, seq_len))
        return music_tokens, dequantised_states, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)
class JukeboxBottleneck(nn.Module):
    def __init__(self, config, levels):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(JukeboxBottleneckBlock(config))

    def encode(self, raw_audio):
        # 对原始音频进行编码
        music_tokens = [
            level_block.encode(hidden_states) for (level_block, hidden_states) in zip(self.level_blocks, raw_audio)
        ]
        return music_tokens

    def decode(self, music_tokens, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        # 对音乐编码进行解码
        quantised_audio = [
            level_block.decode(z) for (level_block, z) in zip(self.level_blocks[start_level:end_level], music_tokens)
        ]
        return quantised_audio

    def forward(self, input_audio):
        # 初始化音乐编码、量化状态、损失和指标
        music_tokens, quantised_states, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            # 获取当前级别的模块
            level_block = self.level_blocks[-level - 1]
            # 获取输入音频的隐藏状态
            hidden_states = input_audio[level]
            # 对隐藏状态进行采样，并得到量化状态、损失和指标
            sampled_tokens, quantised_state, commit_loss, metric = level_block(
                hidden_states, update_codebook=self.training
            )
            music_tokens.append(sampled_tokens)
            if not self.training:
                # 如果不是训练状态，确保量化状态不可变
                quantised_state = quantised_state.detach()
            quantised_states.append(quantised_state)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return music_tokens, quantised_states, commit_losses, metrics


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

@add_start_docstrings(
    """The Hierarchical VQ-VAE model used in Jukebox. This model follows the Hierarchical VQVAE paper from [Will Williams, Sam
Ringer, Tom Ash, John Hughes, David MacLeod, Jamie Dougherty](https://arxiv.org/abs/2002.08111).

    """,
    JUKEBOX_START_DOCSTRING,
)
class JukeboxVQVAE(PreTrainedModel):
    config_class = JukeboxVQVAEConfig
    # 定义基础模型前缀
    base_model_prefix = "vqvae"

    # 初始化模型权重
    def _init_weights(self, module):
        # 如果是嵌入层模块
        if isinstance(module, nn.Embedding):  # embed_tokens
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=0.02 * self.config.init_scale)
        # 如果是 JukeboxConv1D 类型的模块
        elif isinstance(module, JukeboxConv1D):
            # 如果配置了零初始化
            if self.config.zero_out:
                # 将权重置零
                module.weight.data.zero_()
            else:
                # 否则使用正态分布初始化权重
                module.weight.data.normal_(mean=0.0, std=0.02 * self.config.init_scale)
        # 如果是 JukeboxResConv1DBlock 类型的模块，且配置了零初始化
        elif isinstance(module, JukeboxResConv1DBlock) and self.config.zero_out:
            # 将第二个卷积层的权重和偏置置零
            module.conv1d_2.weight.data.zero_()
            module.conv1d_2.bias.data.zero_()
        # 如果是 nn.LayerNorm 类型的模块
        if isinstance(module, nn.LayerNorm):
            # 将偏置置零
            module.bias.data.zero_()
            # 将权重设为全1
            module.weight.data.fill_(1.0)
        # 如果是 nn.Linear 类型的模块且存在偏置
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置置零
            module.bias.data.zero_()

    # 初始化函数
    def __init__(self, config: JukeboxVQVAEConfig):
        # 调用父类初始化函数
        super().__init__(config)
        # 获取配置参数
        downs_t = config.res_downs_t
        strides_t = config.res_strides_t
        # 如果未指定采样长度
        if not config.sample_length:
            # 计算下采样倍数
            downsamples = [stride**down for stride, down in zip(strides_t, downs_t)]
            top_raw_to_tokens = np.prod(downsamples)
            # 根据采样率和采样长度计算采样长度
            config.sample_length = (
                config.sample_length_in_seconds * config.sampling_rate // top_raw_to_tokens
            ) * top_raw_to_tokens
            # 转换为整数类型
            config.sample_length = config.sample_length.astype(int)

        # 获取配置参数
        self.nb_discrete_codes = config.nb_discrete_codes
        self.commit = config.commit
        self.sample_length = config.sample_length

        # 计算下采样和跳数
        self.downsamples = [stride**down for stride, down in zip(strides_t, downs_t)]
        self.hop_lengths = np.cumprod(self.downsamples)
        self.levels = levels = config.levels
        # 计算音乐 tokens 的形状
        self.music_tokens_shapes = [
            (int(self.sample_length // self.hop_lengths[-level - 1])) for level in range(levels)
        ]

        self.multipliers = config.multipliers if config.multipliers is not None else [1] * levels

        # 创建编码器和解码器列表
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # 逐级创建编码器和解码器
        for level in range(levels):
            width = config.res_conv_width * self.multipliers[level]
            depth = config.res_conv_depth * self.multipliers[level]
            self.encoders.append(
                JukeboxEncoder(config, width, depth, level + 1, downs_t[: level + 1], strides_t[: level + 1])
            )
            self.decoders.append(
                JukeboxDecoder(config, width, depth, level + 1, downs_t[: level + 1], strides_t[: level + 1])
            )

        # 创建瓶颈模块
        self.bottleneck = JukeboxBottleneck(config, levels)
    def _decode(self, music_tokens, start_level=0, end_level=None):
        # 解码音乐 tokens
        # 如果未指定结束级别，则默认使用 self.levels
        if end_level is None:
            end_level = self.levels
        # 使用瓶颈层解码音乐 tokens，获取潜在状态
        latent_states = self.bottleneck.decode(music_tokens, start_level=start_level, end_level=end_level)
        # 仅使用最低级别的解码器
        decoder, dequantised_state = self.decoders[start_level], latent_states[0:1]
        # 使用解码器对去量化态进行解码
        dequantised_state = decoder(dequantised_state, all_levels=False)
        # 调整去量化态的维度顺序
        dequantised_state = dequantised_state.permute(0, 2, 1)
        return dequantised_state

    def decode(self, music_tokens, start_level=0, end_level=None, bs_chunks=1) -> torch.Tensor:
        """
        将输入的 `music_tokens` 转换为它们的 `raw_audio` 表示形式。

        Args:
            music_tokens (`torch.LongTensor`):
                将通过使用码书将其解码为原始音频的音乐 tokens 的张量。每个音乐 token 应该是码书中对应 `code` 向量的索引。
            start_level (`int`, *optional*):
                解码过程开始的级别。默认为 0。
            end_level (`int`, *optional*):
                解码过程开始的级别。默认为 None。
            bs_chunks (int, *optional*):
                同时处理的块数。
        """
        # 将音乐 tokens 划分成多个块，每个块包含 bs_chunks 个 token
        token_chunks = [torch.chunk(token, bs_chunks, dim=0) for token in music_tokens]
        dequantised_states = []
        for i in range(bs_chunks):
            # 对每个块的音乐 tokens 进行解码
            music_tokens_i = [chunks[i] for chunks in token_chunks]
            dequantised_state = self._decode(music_tokens_i, start_level=start_level, end_level=end_level)
            dequantised_states.append(dequantised_state)
        return torch.cat(dequantised_states, dim=0)

    def _encode(self, raw_audio, start_level=0, end_level=None):
        # 编码原始音频
        # 如果未指定结束级别，则默认使用 self.levels
        if end_level is None:
            end_level = self.levels
        # 调整输入音频的维度顺序，并转换为浮点数型
        input_audio = raw_audio.permute(0, 2, 1).float()
        latent_states = []
        for level in range(self.levels):
            # 对每个级别使用编码器进行编码，并获取潜在状态的最后一个值
            encoder = self.encoders[level]
            latent_state = encoder(input_audio)
            latent_states.append(latent_state[-1])
        # 使用瓶颈层对潜在状态进行编码，获取音乐 tokens
        music_tokens = self.bottleneck.encode(latent_states)
        return music_tokens[start_level:end_level]
    # 将输入音频转换为由“音乐标记”构成的离散表示
    def encode(self, input_audio, start_level=0, end_level=None, bs_chunks=1):
        """
        Transforms the `input_audio` to a discrete representation made out of `music_tokens`.

        Args:
            input_audio (`torch.Tensor`):
                Raw audio which will be encoded to its discrete representation using the codebook. The closest `code`
                form the codebook will be computed for each sequence of samples.
            start_level (`int`, *optional*, defaults to 0):
                Level at which the encoding process will start. Default to 0.
            end_level (`int`, *optional*):
                Level at which the encoding process will start. Default to None.
            bs_chunks (int, *optional*, defaults to 1):
                Number of chunks of raw audio to process at the same time.
        """
        # 将输入音频按照指定的块大小进行分块
        audio_chunks = torch.chunk(input_audio, bs_chunks, dim=0)
        music_tokens_list = []
        # 对每个音频块进行编码
        for chunk_i in audio_chunks:
            music_tokens_i = self._encode(chunk_i, start_level=start_level, end_level=end_level)
            music_tokens_list.append(music_tokens_i)
        # 将编码后的结果合并为一个列表
        music_tokens = [torch.cat(music_tokens_level, dim=0) for music_tokens_level in zip(*music_tokens_list)]
        return music_tokens

    # 随机生成给定数量的音乐标记序列
    def sample(self, n_samples):
        # 生成n_samples个音乐标记列表，每个列表的形状由self.music_tokens_shapes定义
        music_tokens = [
            torch.randint(0, self.nb_discrete_codes, size=(n_samples, *music_tokens_shape), device="cpu")
            for music_tokens_shape in self.music_tokens_shapes
        ]
        # 解码音乐标记列表，返回音频数据
        return self.decode(music_tokens)
    def forward(self, raw_audio: torch.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VQ-VAE, encodes the `raw_audio` to latent states, which are then decoded for each level.
        The commit loss, which ensure that the encoder's computed embeddings are close to the codebook vectors, is
        computed.

        Args:
            raw_audio (`torch.FloatTensor`):
                Audio input which will be encoded and decoded.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`


        Example:
        ```python
        >>> from transformers import JukeboxVQVAE, set_seed
        >>> import torch

        >>> model = JukeboxVQVAE.from_pretrained("openai/jukebox-1b-lyrics").eval()
        >>> set_seed(0)
        >>> zs = [torch.randint(100, (4, 1))]
        >>> model.decode(zs).shape
        torch.Size([4, 8, 1])
        ```py
        """

        # 开始编码/解码

        # 将输入音频进行轴置换，并转换为浮点型
        input_audio = raw_audio.permute(0, 2, 1).float()

        # 存储编码后的潜在状态
        latent_states = []

        # 遍历每个级别的编码器
        for level in range(self.levels):
            # 获取当前级别的编码器
            encoder = self.encoders[level]

            # 对输入音频进行编码
            latent_state = encoder(input_audio)

            # 将编码后的潜在状态存储起来
            latent_states.append(latent_state[-1])

        # 获取音乐 tokens、commit losses 等信息
        _, music_tokens, commit_losses, _ = self.bottleneck(latent_states)

        # 存储解量化后的状态
        dequantised_states = []

        # 遍历每个级别的解码器
        for level in range(self.levels):
            # 获取当前级别的解码器
            decoder = self.decoders[level]

            # 对音乐 tokens 进行解码，获取解量化后的状态
            dequantised_state = decoder(music_tokens[level : level + 1], all_levels=False)

            # 存储解码后的状态
            dequantised_states.append(dequantised_state.permute(0, 2, 1))

        # 计算 commit loss
        commit_loss = sum(commit_losses)

        # 计算损失
        loss = self.commit * commit_loss

        # 返回解码后的状态以及损失
        return dequantised_states, loss
# 定义一个多层感知机（MLP）模型类，用于音乐生成
class JukeboxMLP(nn.Module):
    def __init__(self, config):
        # 原始代码中始终使用单个通道
        super().__init__()
        # 设置嵌入维度
        embed_dim = config.hidden_size
        # 计算隐藏层维度，根据配置中的 MLP 倍增系数
        hidden_dim = int(config.mlp_multiplier * embed_dim)

        # 定义输入到隐藏层的卷积层
        self.c_fc = JukeboxConv1D(embed_dim, hidden_dim)
        # 定义隐藏层到输出的卷积层
        self.c_proj = JukeboxConv1D(hidden_dim, embed_dim)
        # 设置激活函数
        self.act = ACT2FN[config.act_fn]
        # 设置 dropout 层
        self.dropout = nn.Dropout(config.resid_dropout)

    # 前向传播函数
    def forward(self, hidden_states):
        # 输入到隐藏层的卷积操作
        hidden_states = self.c_fc(hidden_states)
        # 激活函数操作
        hidden_states = self.act(hidden_states)
        # 隐藏层到输出的卷积操作
        hidden_states = self.c_proj(hidden_states)
        # dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 返回结果
        return hidden_states


# 定义 Jukebox 的层归一化类，继承自融合层归一化类
class JukeboxLayerNorm(FusedLayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        # 初始化函数
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        # 计算归一化维度的总数
        self.width = np.prod(normalized_shape)
        # 设置最大元素数量
        self.max_numel = 65535 * self.width

    # 前向传播函数
    def forward(self, input):
        # 如果输入元素数量超过最大数量，则使用标准层归一化
        if input.numel() > self.max_numel:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps).type_as(input)
        else:
            # 否则使用融合层归一化
            return super().forward(input).type_as(input)


# 定义 Jukebox 的注意力机制类
class JukeboxAttention(nn.Module):
    # 初始化函数，传入配置、上下文长度和注意力机制函数名称
    def __init__(self, config, n_ctx, attn_func="dense_attn"):
        super().__init__()
        # 隐藏层的维度
        self.embed_dim = config.hidden_size
        # 注意力头的数量
        self.n_heads = config.n_heads
        # 注意力机制的dropout率
        self.dropout = config.attn_dropout
        # 隐藏层的维度乘以注意力扩大因子
        hidden_dim = int(config.attention_multiplier * self.embed_dim)
    
        # 头的维度
        self.head_dim = hidden_dim // config.n_heads
        # 上下文长度
        self.n_ctx = n_ctx
        # 隐藏层的维度
        self.hidden_dim = hidden_dim
        # 缩放因子
        self.scale = self.head_dim**-0.25
        # 是否使用mask
        self.mask = config.mask
    
        # 根据不同的注意力机制函数选择相应的子模块
        if attn_func == "cross_attention":
            self.c_attn = JukeboxConv1D(self.embed_dim, hidden_dim)
            self.c_enc_kv = JukeboxConv1D(self.embed_dim, hidden_dim * 2)
        else:
            self.c_attn = JukeboxConv1D(self.embed_dim, hidden_dim * 3)
    
        # 投影层
        self.c_proj = JukeboxConv1D(hidden_dim, self.embed_dim)
        # 注意力机制的dropout
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        # 输出的残差连接的dropout
        self.resid_dropout = nn.Dropout(config.resid_dropout)
    
        # 根据不同的注意力机制函数选择对应的qkv函数
        self.attn_func = attn_func
        if attn_func == "cross_attention":
            self.qkv = self.decode_qkv
        elif attn_func == "prime_attn":
            self.qkv = self.prime_qkv
        else:
            self.qkv = self.factored_qkv
        
        # 不同的注意力机制函数对应的具体实现
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
        self.attn, self.attn_mask = ATTENTION_MAP[attn_func]
    
        # 所分成的块数和扩散因子
        self.blocks = config.blocks
        self.spread = config.spread
        if self.blocks is not None:
            # 每个块的上下文长度
            self.block_ctx = self.n_ctx // self.blocks
    
        # 采样时间和缓存
        self.sample_t = 0
        self.cache = {}
        # 编码器输入id的长度
        self.encoder_len = config.nb_relevant_lyric_tokens
        # 是否记录注意力分布
        self.record_attn = False
    # 定义注意力函数，根据查询、键、值和样本生成上下文状态
    def _attn(self, query_states, key_states, value_states, sample):
        # 缩放参数
        scale = self.scale
        # 如果处于训练状态，使用缩放后的查询和键进行点积操作
        if self.training:
            attention_weight = torch.matmul(query_states * scale, key_states * scale)
        else:
            # 否则，直接进行点积操作并乘以缩放的平方
            attention_weight = torch.matmul(query_states, key_states)
            attention_weight.mul_(scale * scale)
        attn_weight_type = attention_weight.dtype
        attention_weight = attention_weight.float()
        # 如果需要进行掩码操作
        if self.mask:
            # 生成适当的掩码，用以在当前位置之前掩盖所有位置
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
            # 如果存在掩码
            if mask is not None:
                # 对注意力权重应用掩码，同时对未被掩码的位置加上一个很大的负数
                attention_weight = attention_weight * mask + -1e9 * (1 - mask)
        # 对注意力权重进行 softmax 操作
        attention_prob = F.softmax(attention_weight, dim=-1).type(attn_weight_type)
        # 如果记录注意力权重
        if self.record_attn:
            self.attention_prob = attention_prob
            # 如果使用的是 "prime_attn" 注意力函数
            if self.attn_func == "prime_attn":
                # 仅保留音乐查询和歌词键/值
                self.attention_prob = self.attention_prob[:, :, self.encoder_len :, : self.encoder_len]
        # 对注意力权重应用 dropout
        attention_prob = self.attn_dropout(attention_prob)
        # 计算上下文状态
        context_states = torch.matmul(attention_prob, value_states)
        return context_states

    # 合并注意力头部
    def merge_heads(self, hidden_states):
        # 对隐藏状态进行维度置换和重排，将注意力头部合并到一个维度
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = (*hidden_states.size()[:-2], hidden_states.size(-2) * hidden_states.size(-1))
        return hidden_states.view(*new_hidden_states_shape)

    # 拆分注意力头部
    def split_heads(self, hidden_states, is_key=False):
        new_hidden_states_shape = (
            *hidden_states.size()[:-1],
            self.n_heads,
            hidden_states.size(-1) // self.n_heads,
        )
        hidden_states = hidden_states.view(*new_hidden_states_shape)
        # 如果是键，进行维度置换
        if is_key:
            return hidden_states.permute(0, 2, 3, 1)
        else:
            return hidden_states.permute(0, 2, 1, 3)

    # 密集注意力计算
    def dense_attn(self, query, key, value, sample):
        # 分割查询、键、值的注意力头部
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        # 使用注意力函数计算上下文状态
        context_states = self._attn(query, key, value, sample)
        # 合并注意力头部
        context_states = self.merge_heads(context_states)
        return context_states
    # 定义一个函数，用于实现块注意力机制
    def block_attn(self, query, key, value, sample):
        # 获取块大小
        block_ctx = self.block_ctx
        # 获取批次大小、序列长度、嵌入维度
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        # 如果是样本
        if sample:
            # 调用稠密注意力函数，返回的结果维度变为(batch_size, 1, embed_dim)
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            # 获取查询向量的长度
            query_length = query.shape[1]
            # 重塑查询向量的形状为(batch_size * query_length // block_ctx, block_ctx, embed_dim)
            query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)
            # 如果查询向量的长度小于序列长度
            if query_length < seq_len:
                seq_len = query_length
                # 截取键和值的后部分，使其长度与查询向量一致
                key = key[:, -seq_len:].contiguous()
                value = value[:, -seq_len:].contiguous()
            # 重塑键和值的形状为(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            # 调用稠密注意力函数，返回的结果维度变为(batch_size, seq_len, embed_dim)
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    # 定义一个函数，用于实现转置块注意力机制
    def transpose_block_attn(self, query, key, value, sample):
        # 获取块大小
        block_ctx = self.block_ctx
        # 获取批次大小、序列长度、嵌入维度
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        # 如果是样本
        if sample:
            # 计算块长度
            block_len = (seq_len - 1) % block_ctx
            # 根据块长度截取键和值的数据，以块大小为步长
            key = key[:, block_len::block_ctx, :]
            value = value[:, block_len::block_ctx, :]
            # 调用稠密注意力函数，返回的结果维度变为(batch_size, 1, embed_dim)
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            # 获取查询向量的长度
            query_length = query.shape[1]
            # 重塑查询向量的形状为(batch_size, query_length // block_ctx, block_ctx, embed_dim)
            query = query.view(batch_size, query_length // block_ctx, block_ctx, embed_dim)
            # 对查询向量进行转置操作
            query = query.transpose(1, 2).contiguous()
            # 重塑查询向量的形状为(batch_size * block_ctx, query_length // block_ctx, embed_dim)
            query = query.view(batch_size * block_ctx, query_length // block_ctx, embed_dim)

            # 重塑键和值的形状
            key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
            key = key.transpose(1, 2).contiguous()
            key = key.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)

            value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
            value = value.transpose(1, 2).contiguous()
            value = value.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)

            # 调用稠密注意力函数，返回的结果维度变为(batch_size, block_ctx, query_length // block_ctx, embed_dim)
            block_attn = self.dense_attn(query, key, value, sample)
            # 对块注意力进行转置操作
            block_attn = block_attn.view(batch_size, block_ctx, query_length // block_ctx, embed_dim)
            block_attn = block_attn.transpose(1, 2).contiguous()
            # 重塑形状为(batch_size, query_length, embed_dim)
            block_attn = block_attn.view(batch_size, query_length, embed_dim)

            return block_attn
    def prev_block_attn(self, query, key, value, sample):
        # 获取当前模型的块上下文大小
        block_ctx = self.block_ctx
        # 获取数值的维度信息
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        if sample:
            # 根据sample标志选择不同的处理方式
            block = (seq_len - 1) // block_ctx
            prev_l = (block - 1) * block_ctx
            # 对key和value进行截断操作
            if block > 0:
                key = key[:, prev_l : prev_l + block_ctx, :]
                value = value[:, prev_l : prev_l + block_ctx, :]
            else:
                key = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
                value = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
            # 调用dense_attn函数进行查询操作，返回结果并reshape为(batch_size, 1, embed_dim)
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            # 对query进行reshape操作
            query_length = query.shape[1]
            query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)

            # 对key和value进行reshape操作
            key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0))
            key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)

            value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0))
            value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)

            # 如果查询长度小于序列长度，则截取对应的块进行处理
            if query_length < seq_len:
                nb_query_blocks = query_length // block_ctx
                nb_key_blocks = seq_len // block_ctx
                seq_len = query_length
                key = key.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
                key = key.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)

                value = value.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
                value = value.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)

            # 调用dense_attn函数进行查询操作，返回结果并reshape为(batch_size, seq_len, embed_dim)
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)
    # 计算注意力摘要，根据给定的查询、键和数值，以及是否是样本
    def summary_attn(self, query, key, value, sample):
        # 获取块的数量和块的上下文大小
        blocks = self.blocks
        block_ctx = self.block_ctx
        # 获取数值张量的批量大小、序列长度和嵌入维度
        batch_size, seq_len, embed_dim = value.shape

        # 如果是样本
        if sample:
            # 从键中选择块上下文的内容
            key = key[:, block_ctx - 1 : blocks * block_ctx - 1 : block_ctx, :]
            # 使用零填充对键进行填充
            key = torch.nn.functional.pad(key, (0, 0, 1, 0))

            # 从数值中选择块上下文的内容
            value = value[:, block_ctx - 1 : blocks * block_ctx - 1 : block_ctx, :]
            # 使用零填充对数值进行填充
            value = torch.nn.functional.pad(value, (0, 0, 1, 0))

            # 使用密集注意力计算结果并调整形状
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        # 如果不是样本
        else:
            # 重新组织键的形状，并对其进行填充
            key = key.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -1, :]
            key = torch.nn.functional.pad(key, (0, 0, 1, 0))

            # 重新组织数值的形状，并对其进行填充
            value = value.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -1, :]
            value = torch.nn.functional.pad(value, (0, 0, 1, 0))

            # 使用密集注意力计算结果并调整形状
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    # 计算扩展注意力摘要，根据给定的查询、键和数值，以及是否是样本
    def summary_spread_attn(self, query, key, value, sample):
        # 获取块的数量和扩展数量
        blocks = self.blocks
        spread = self.spread

        # 获取数值张量的批量大小、序列长度和嵌入维度
        batch_size, seq_len, embed_dim = value.shape

        # 如果是样本，抛出未实现的错误
        if sample:
            raise NotImplementedError
        # 如果不是样本
        else:
            # 重新组织键的形状，并对其进行填充和连续化
            key = key.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0)).contiguous()
            key = key.view(batch_size, blocks * spread, embed_dim)

            # 重新组织数值的形状，并对其进行填充和连续化
            value = value.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0)).contiguous()
            value = value.view(batch_size, blocks * spread, embed_dim)

            # 使用密集注意力计算结果并调整形状
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    # 计算主要注意力，根据给定的查询、键和数值，以及是否是样本
    def prime_attn(self, query, key, value, sample):
        # 获取编码器长度，截取键和数值
        encoder_len = self._encoder_len
        key = key[:, :encoder_len]
        value = value[:, :encoder_len]
        return self.dense_attn(query, key, value, sample)
    # 计算查询、键和值，并返回结果
    def factored_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 获取隐藏状态的当前上下文长度
        curr_ctx = hidden_states.shape[1]
        # 如果最后一个编码器隐藏状态不是None，则引发TypeError异常
        if last_encoder_hidden_states is not None:
            raise TypeError("last_encoder_hidden_states should be None")
        # 将隐藏状态分割为查询、键和值
        query, key, value = hidden_states.chunk(3, dim=2)
        # 如果需要采样
        if sample:
            # 增加采样时间步
            self.sample_t += curr_ctx
            # 向缓存中添加键和值
            key, value = self._append_cache(key, value)
            # 获取缓存的长度
            l_cache = self._suff_cache_len()
            # 如果总缓存长度超过了指定的长度
            if self._cache_len() > l_cache:
                # 则删除超出部分
                self._slice_cache(-l_cache)
            # 如果当前上下文长度大于1
            if curr_ctx > 1:
                # 如果注意力函数不是"dense_attn"
                if self.attn_func != "dense_attn":
                    # 将查询、键和值进行填充到块上下文
                    query = self._pad_to_block_ctx(query, query=True)
                    key = self._pad_to_block_ctx(key)
                    value = self._pad_to_block_ctx(value)
                # 取消采样标志
                sample = False
            else:
                # 获取缓存中的键和值
                key = self.cache["key"]
                value = self.cache["value"]
        # 返回查询、键、值和采样标志的结果
        return query, key, value, sample

    # 计算查询、键和值，并返回结果
    def prime_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 获取隐藏状态的当前上下文长度
        curr_ctx = hidden_states.shape[1]
        # 如果最后一个编码器隐藏状态不是None，则引发TypeError异常
        if last_encoder_hidden_states is not None:
            raise TypeError("last_encoder_hidden_states should be None")
        # 将隐藏状态分割为查询、键和值
        query, key, value = hidden_states.chunk(3, dim=2)
        # 如果需要采样
        if sample:
            # 如果缓存长度小于编码器长度
            if self._cache_len() < self._encoder_len:
                # 向缓存中添加键和值
                self._append_cache(key, value)
            # 如果缓存长度大于编码器长度
            if self._cache_len() > self._encoder_len:
                # 则删除超出部分
                self._slice_cache(0, self._encoder_len)
            # 获取缓存中的键和值
            key, value = self.cache["key"], self.cache["value"]
            # 增加采样时间步
            self.sample_t += curr_ctx
        # 返回查询、键、值和采样标志的结果
        return query, key, value, sample

    # 计算查询、键和值，并返回结果
    def decode_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 获取隐藏状态的当前上下文长度
        curr_ctx = hidden_states.shape[1]
        # 查询即为隐藏状态
        query = hidden_states
        # 如果需要采样
        if sample:
            # 如果采样时间为0
            if self.sample_t == 0:
                # 从编码器的键和值中取出缓存的键和值
                self.cache["key"], self.cache["value"] = self.c_enc_kv(
                    last_encoder_hidden_states.type_as(hidden_states)
                ).chunk(2, dim=2)
            # 获取缓存中的键和值
            key, value = self.cache["key"], self.cache["value"]
            # 增加采样时间步
            self.sample_t += curr_ctx
        else:
            # 获取编码器的键和值
            key, value = self.c_enc_kv(last_encoder_hidden_states.type_as(hidden_states)).chunk(2, dim=2)
        # 返回查询、键、值和采样标志的结果
        return query, key, value, sample
    # 前向传播函数，接受隐藏状态数据，计算注意力，返回结果
    def forward(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 获取当前上下文长度
        curr_ctx = hidden_states.shape[1]
        # 使用自注意力层进行处理隐藏状态数据
        hidden_states = self.c_attn(hidden_states)
        # 调用查询、键、值的函数，获取注意力得分
        query, key, value, sample = self.qkv(
            hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=sample
        )
        attention_scores = self.attn(query, key, value, sample)
        # 如果注意力得分的上下文长度不一致，则调整注意力得分的上下文长度
        if attention_scores.shape[1] != curr_ctx:
            offset = self._offset(curr_ctx)
            attention_scores = attention_scores[:, offset : offset + curr_ctx, :].contiguous()
        # 将注意力得分经过线性变换
        attention_scores = self.c_proj(attention_scores)
        # 返回具有残差的注意力得分
        return self.resid_dropout(attention_scores)

    @property
    # 获取编码器长度的私有函数
    def _encoder_len(self):
        encoder_len = self.encoder_len
        encoder_blocks = (encoder_len // self.blocks) + 1
        return encoder_blocks * self.blocks

    # 计算补偿偏移量的私有函数
    def _offset(self, curr_ctx):
        if self.attn_func == "dense_attn":
            return 0
        return (self.sample_t - curr_ctx) % self.block_ctx

    # 将隐藏状态数据填充到块上下文长度的私有函数
    def _pad_to_block_ctx(self, hidden_states, query=False):
        seq_len = hidden_states.shape[1]
        offset = self._offset(seq_len) if query else 0
        n_blocks = (seq_len + offset + self.block_ctx - 1) // self.block_ctx
        pad = n_blocks * self.block_ctx - seq_len - offset
        if pad == 0 and offset == 0:
            return hidden_states
        else:
            return F.pad(hidden_states, (0, 0, offset, pad))

    # 获取缓存长度的私有函数
    def _cache_len(self):
        return 0 if "key" not in self.cache else self.cache["key"].shape[1]

    # 获取足够的缓存长度的私有函数
    def _suff_cache_len(self):
        """
        Precondition:
            key and value are appended with the current context and self.sample_t reflects the 1-indexed sample
            location in the context.
        """
        previous_block_length = (self.sample_t - 1) % self.block_ctx + 1 + self.block_ctx
        REQUIRED_CACHE_LEN = {
            "dense_attn": self.sample_t,
            "block_attn": (self.sample_t - 1) % self.block_ctx + 1,
            "transpose_block_attn": self.sample_t,
            "prev_block_attn": self.sample_t if self.sample_t <= self.block_ctx else previous_block_length,
            "cross_attn": self.encoder_len,
            "prime_attn": min(self.sample_t, self._encoder_len),
        }

        return REQUIRED_CACHE_LEN[self.attn_func]

    # 切片缓存数据的私有函数
    def _slice_cache(self, start, end=None):
        self.cache["key"] = self.cache["key"][:, start:end]
        self.cache["value"] = self.cache["value"][:, start:end]
    # 向缓存中添加键值对
    def _append_cache(self, key, value):
        # 如果缓存中不存在当前键
        if "key" not in self.cache:
            # 将当前键值对添加到缓存中
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            # 如果缓存中已经存在键值对，则进行合并操作
            old_key, old_value = key, value
            key = torch.cat([self.cache["key"], old_key], dim=1)
            value = torch.cat([self.cache["value"], old_value], dim=1)
            # 删除旧的键值对
            del self.cache["key"]
            del self.cache["value"]
            del old_key
            del old_value
            # 更新缓存中的键值对
            self.cache["key"] = key
            self.cache["value"] = value
        # 返回缓存中的键值对
        return self.cache["key"], self.cache["value"]

    # 清空缓存
    def del_cache(self):
        # 重置采样时间
        self.sample_t = 0
        # 如果缓存中存在键，则删除该键
        if "key" in self.cache:
            del self.cache["key"]
        # 如果缓存中存在值，则删除该值
        if "value" in self.cache:
            del self.cache["value"]
        # 清空整个缓存字典
        self.cache = {}
class JukeboxBlock(nn.Module):
    # JukeboxBlock 类的初始化函数
    def __init__(self, config, n_ctx, attn_func="dense_attn"):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()
        # 设置隐藏层的宽度
        self.width = config.hidden_size
        # 创建注意力模块
        self.attn = JukeboxAttention(config, n_ctx, attn_func=attn_func)

        # 创建 LayerNorm 层
        self.layer_norm_0 = JukeboxLayerNorm(config.hidden_size)
        # 创建 MLP（多层感知机）模块
        self.mlp = JukeboxMLP(config)
        # 创建 LayerNorm 层
        self.layer_norm_1 = JukeboxLayerNorm(config.hidden_size)
        # 计算注意力残差的比例
        self.res_scale = 1.0 / config.num_layers if config.attn_res_scale else 1.0
        # 记录使用的注意力函数名称
        self.attn_func = attn_func

    # 前向传播函数
    def forward(self, hidden_states, last_encoder_hidden_states, sample=False):
        # 保存残差连接
        residuals = hidden_states
        # LayerNorm 层处理隐藏状态
        hidden_states = self.layer_norm_0(hidden_states)
        # 进行注意力计算
        hidden_states = self.attn(hidden_states, last_encoder_hidden_states, sample)

        # 计算输出状态
        output_states = self.layer_norm_1(residuals + hidden_states)
        output_states = self.mlp(output_states)
        # 计算最终输出，根据配置选择是否使用残差连接
        if self.res_scale == 1.0:
            output = residuals + hidden_states + output_states
        else:
            output = residuals + self.res_scale * (hidden_states + output_states)
        # 返回输出
        return output


class JukeboxLayerStack(nn.Module):
    # JukeboxLayerStack 类的初始化函数
    def __init__(self, config, n_ctx):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()
        # 设置上下文长度
        self.n_ctx = n_ctx
        # 设置隐藏层宽度
        self.width = config.hidden_size
        # 设置层数
        self.num_layers = config.num_layers
        # 设置块数
        self.blocks = config.blocks
        # 设置注意力模式
        self.attention_pattern = config.attention_pattern
        # 计算块上下文长度
        if self.blocks is not None:
            self.block_ctx = n_ctx // self.blocks
        # 设置编码器长度
        self.encoder_len = config.nb_relevant_lyric_tokens
        # 设置头数
        self.n_heads = config.n_heads

        # Orders of attn_func
        # 根据注意力模式创建注意力模块
        attention_pattern = ATTENTION_PATTERNS[self.attention_pattern]
        self._attn_mods = nn.ModuleList()
        for depth in range(self.num_layers):
            self._attn_mods.append(JukeboxBlock(config, n_ctx, attn_func=attention_pattern(depth)))

        # 保存自注意力权重
        self.saved_attn_weights = []

    # 设置是否记录注意力权重
    def set_record_attn(self, record_attn):
        """
        Makes forward prop dump self-attention softmaxes to self.saved_attn_weights.

        Args:
            record_attn (`Union[bool,set]`):
                Either a set of layer indices indicating which layers to store, or a boolean value indicating Whether
                to dump all.
        """

        # 判断是否应该记录注意力权重
        def _should_record_attn(layer_idx):
            if isinstance(record_attn, bool):
                return record_attn
            return layer_idx in record_attn

        # 设置每一层是否记录注意力权重
        for i, layer in enumerate(self._attn_mods):
            layer.attn.record_attn = _should_record_attn(i)

        # 如果不记录注意力权重，则清空保存的自注意力权重
        if not record_attn:
            self.saved_attn_weights = []
    # 定义一个前向传播函数，用于处理隐藏状态，可以选择是否使用上一个编码器的隐藏状态
    def forward(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # 遍历注意力层模块
        for i, attn_layer in enumerate(self._attn_mods):
            # 如果是交叉注意力层，则执行跨编码器关注操作
            if attn_layer.attn_func == "cross_attention":  # attend to the lyrics
                hidden_states = attn_layer(
                    hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=sample
                )
            else:
                # 否则执行普通的注意力操作
                hidden_states = attn_layer(hidden_states, last_encoder_hidden_states=None, sample=sample)
            # 如果注意力层记录注意力权重，则将权重保存在 saved_attn_weights 列表中
            if attn_layer.attn.record_attn:
                self.saved_attn_weights.append(attn_layer.attn.c_attn.weight)
        # 返回处理后的隐藏状态
        return hidden_states

    # 删除缓存的函数
    def del_cache(self):
        # 遍历所有注意力层模块，调用删除缓存的方法
        for attn_layer in self._attn_mods:
            attn_layer.attn.del_cache()
# 定义一个名为 JukeboxPositionalEmbedding 的类，继承自 nn.Module
class JukeboxPositionalEmbedding(nn.Module):
    # 初始化方法，接受两个参数，embed_dim 表示嵌入的维度，width 表示嵌入的宽度
    def __init__(self, embed_dim, width):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个可学习的参数，表示位置嵌入，形状为 (embed_dim, width)
        self.pos_emb = nn.Parameter(torch.empty((embed_dim, width)))

    # 前向传播方法
    def forward(self):
        # 将位置嵌入赋值给 pos_emb
        pos_emb = self.pos_emb
        # 返回位置嵌入
        return pos_emb


# 定义一个名为 JukeboxConditionalAutoregressive 的类，继承自 nn.Module
class JukeboxConditionalAutoregressive(nn.Module):
    # 初始化方法，接受多个参数，config 表示配置，n_ctx 表示上下文的大小，embed_dim 表示嵌入的维度
    # audio_conditioning 表示是否进行音频条件，metadata_conditioning 表示是否进行元数据条件，is_encoder 表示是否是编码器
    def __init__(
        self,
        config,
        n_ctx=None,
        embed_dim=None,
        audio_conditioning=False,
        metadata_conditioning=False,
        is_encoder=False,
    ):
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
                if the model combines lyrics and music tokens, or simply n_vocab if the model is a separate encoder
            audio_conditioning (`bool`, *optional`, defaults to `False`):
                Whether or not the prior supports conditioning on audio.
            metadata_conditioning (`bool`, *optional`, defaults to `False`):
                Whether or not the prior supports conditioning on artist, genres, lyrics and timing.
            is_encoder (`bool`, *optional`, defaults to `False`):
                Whether the model is an encoder only model.
        """

        # Call the superclass constructor
        super().__init__()
        
        # Initialize instance variables based on constructor arguments and configuration
        self.width = config.hidden_size
        self.num_layers = config.num_layers
        self.n_ctx = n_ctx if n_ctx is not None else config.n_ctx
        self.embed_dim = embed_dim if embed_dim is not None else config.music_vocab_size
        self.embed_tokens = nn.Embedding(self.embed_dim, config.hidden_size)
        self.embed_tokens_dropout = nn.Dropout(config.emb_dropout)
        self.metadata_conditioning = metadata_conditioning
        self.audio_conditioning = audio_conditioning
        
        # Set start token parameter if metadata conditioning is false
        if not metadata_conditioning:
            self.start_token = nn.Parameter(torch.empty((1, config.hidden_size)))
        self.pos_emb = JukeboxPositionalEmbedding(self.n_ctx, config.hidden_size)
        self.pos_emb_dropout = nn.Dropout(config.emb_dropout)

        # Create transformer layer stack
        self.transformer = JukeboxLayerStack(config, n_ctx=self.n_ctx)
        self.is_encoder = is_encoder
        self.encoder_len = config.nb_relevant_lyric_tokens

        if config.merged_decoder:
            # Merged piped model setup
            self.add_cond_after_transformer = False
            self.share_embed_tokens_fc_proj_out = False
        else:
            self.add_cond_after_transformer = True
            self.share_embed_tokens_fc_proj_out = True

        if not is_encoder:
            # Initialize output projection layer and loss function for decoder
            self.fc_proj_out = nn.Linear(config.hidden_size, self.embed_dim, bias=False)
            if self.share_embed_tokens_fc_proj_out:
                self.fc_proj_out.weight = self.embed_tokens.weight
            self.loss = torch.nn.CrossEntropyLoss()
``` 
    # 前向传播函数，用于生成模型的输出
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
        # 预处理。
        batch_size = tokens.shape[0]  # 获取批次大小
        with torch.no_grad():
            tokens = tokens.view(batch_size, -1).long()  # 将 tokens 重新形状为二维张量并转换为长整型

        if not self.audio_conditioning:
            # 如果未提供音频条件，则创建全零张量作为音频条件
            audio_conditioning = torch.zeros(
                (batch_size, 1, self.width),
                device=tokens.device,
                dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype,
            )

        target = tokens  # 目标（预测的输出）
        hidden_states = self.embed_tokens(tokens)  # 嵌入 tokens
        # 将 hidden_states 向右移动一位，并在开头填充起始令牌
        hidden_states = torch.cat((hidden_states[:, -1:], hidden_states[:, :-1]), dim=1)
        if self.metadata_conditioning:
            hidden_states[:, 0] = metadata_conditioning.view(batch_size, self.width)
        else:
            hidden_states[:, 0] = self.start_token  # 如果没有元数据条件，则使用模型的起始令牌

        hidden_states = (
            self.embed_tokens_dropout(hidden_states) + self.pos_emb_dropout(self.pos_emb()) + audio_conditioning
        )  # 添加位置嵌入和丢弃嵌入的结果

        hidden_states = self.transformer(
            hidden_states, last_encoder_hidden_states=last_encoder_hidden_states
        )  # Transformer 模型的前向传播

        if self.add_cond_after_transformer:  # 如果在 Transformer 之后添加条件
            hidden_states = hidden_states + audio_conditioning

        activations = hidden_states  # 激活值等于隐藏状态
        if self.is_encoder:  # 如果是编码器模型，则直接返回隐藏状态
            return hidden_states

        hidden_states = self.fc_proj_out(hidden_states)  # 将隐藏状态投影到输出空间，进行预测
        loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失函数进行损失计算
        if get_sep_loss:  # 如果需要分离的损失
            lyric_hidden_states = hidden_states[:, : self.encoder_len].reshape(-1, self.embed_dim)
            token_hidden_states = hidden_states[:, self.encoder_len :].reshape(-1, self.embed_dim)

            # 分别计算歌词部分和音乐 token 部分的损失
            lyric_loss = loss_fn(lyric_hidden_states, target[:, : self.encoder_len].reshape(-1)) / np.log(2.0)
            music_token_loss = loss_fn(token_hidden_states, target[:, self.encoder_len :].reshape(-1)) / np.log(2.0)

            loss = (lyric_loss, music_token_loss)  # 损失按顺序组成元组，歌词部分在前
        else:
            # 计算整体损失
            loss = loss_fn(hidden_states.view(-1, self.embed_dim), target.view(-1)) / np.log(2.0)

        if get_preds:  # 如果需要返回预测值
            return loss, hidden_states
        elif get_acts:  # 如果需要返回激活值
            return loss, activations
        else:
            return loss, None  # 如果不需要返回额外信息，则只返回损失
```py  
    # 定义一个获取词嵌入的函数
    def get_emb(self, sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning):
        # 如果是第一个样本
        if sample_t == 0:
            # 创建一个具有指定尺寸和数据类型的张量，用于存储隐藏状态
            hidden_states = torch.empty(n_samples, 1, self.width, dtype=self.embed_tokens.weight.dtype).to(
                self.embed_tokens.weight.device
            )
            # 如果有元数据条件
            if self.metadata_conditioning:
                # 将元数据转换成指定形状的张量作为隐藏状态的初始值
                hidden_states[:, 0] = metadata_conditioning.view(n_samples, self.width)
            else:
                # 使用预定义开始标记作为隐藏状态的初始值
                hidden_states[:, 0] = self.start_token
        else:
            # 获取嵌入的 token
            hidden_states = self.embed_tokens(tokens)
        # 如果音频条件的形状符合要求
        if audio_conditioning.shape == (n_samples, self.n_ctx, self.width):
            # 从音频条件中提取特定时间步的条件信息
            cond = audio_conditioning[:, sample_t : sample_t + 1, :]
        else:
            # 否则，直接使用完整的音频条件
            cond = audio_conditioning
        # 添加位置嵌入，评估时的 dropout 是恒等变换
        hidden_states = hidden_states + self.pos_emb()[sample_t : sample_t + 1] + cond
        # 返回隐藏状态和条件信息
        return hidden_states, cond

    # 定义一个采样函数
    def sample(
        self,
        n_samples,
        audio_conditioning=None,
        metadata_conditioning=None,
        last_encoder_hidden_states=None,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        get_preds=False,
        sample_tokens=None,
        ):
        # 检查是否指定了样本标记，如果未指定则使用默认值self.n_ctx
        if sample_tokens is None:
            sample_tokens = self.n_ctx

        # 如果没有音频条件，则初始化一个形状为(n_samples, 1, self.width)的torch张量，并转移到特定设备上
        if not self.audio_conditioning:
            audio_conditioning = torch.zeros(
                (n_samples, 1, self.width), dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype
            ).to(self.fc_proj_out.device)

        # 禁用梯度追踪
        with torch.no_grad():
            sampled_tokens = []
            tokens = None
            if get_preds:
                preds = []

            # 创建进度条迭代器
            iter = tqdm(range(0, sample_tokens), leave=False)
            for sample_t in iter:
                iter.set_description(f"Ancestral sampling {sample_tokens} music tokens", refresh=True)
                # 获取嵌入向量和条件信息
                hidden_states, cond = self.get_emb(
                    sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning
                )

                # 输入隐层状态到transformer，进行预测
                hidden_states = self.transformer(
                    hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=True
                )
                # 如果add_cond_after_transformer为True，添加条件信息到隐层状态
                if self.add_cond_after_transformer:
                    hidden_states = hidden_states + cond
                # 使用输出层进行预测
                hidden_states = self.fc_proj_out(hidden_states)  # Predictions
                # 如果需要保存预测结果，则将预测结果添加到preds列表中
                if get_preds:
                    preds.append(hidden_states.clone())
                # 调整logits，并根据top_k和top_p筛选logits
                hidden_states = hidden_states / temp
                hidden_states = filter_logits(hidden_states, top_k=top_k, top_p=top_p)
                # 从logits中随机采样得到token
                tokens = torch.distributions.Categorical(logits=hidden_states).sample()
                sampled_tokens.append(tokens.clone())

            del tokens
            self.transformer.del_cache()

            # 合并所有采样的tokens
            tokens = torch.cat(sampled_tokens, dim=1)
            if get_preds:
                preds = torch.cat(preds, dim=1)
        # 如果需要保存预测结果，则返回tokens和preds，否则返回tokens
        if get_preds:
            return tokens, preds
        else:
            return tokens

    # 将长度分割成指定大小的段落
    def split_chunks(self, length, chunk_size):
        n_passes = (length + chunk_size - 1) // chunk_size
        chunk_sizes = [*[chunk_size] * (n_passes - 1), (length - 1) % chunk_size + 1]
        return chunk_sizes

    # 使用预设的token进行采样
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
        # Embedding layer for music tokens
        self.embed_tokens = nn.Embedding(config.music_vocab_size, config.hidden_size)
        # Setting the embedding dimension to music vocabulary size for `JukeboxDecoder`
        config.embed_dim = config.music_vocab_size  

        # Initializing upsampler using `JukeboxDecoderConvBock`
        self.upsampler = JukeboxDecoderConvBock(
            config,
            config.hidden_size,
            config.res_conv_width,
            config.res_conv_depth,
            config.res_downs_t[level],
            config.res_strides_t[level],
            reverse_dilation=False,
        )
        # Layer normalization for hidden states
        self.layer_norm = JukeboxLayerNorm(config.hidden_size)

    def forward(self, music_tokens, raw_audio_conditionning=None):
        """
        Args:
            music_tokens (`torch.LongTensor`):
                Music tokens form the upper level in range(nb_discrete_codes)
            raw_audio_conditionning (`torch.LongTensor`, *optional*):
                Audio used when primed sampling, raw audio information that conditions the generation
        """
        # Handling missing raw_audio_conditionning
        if raw_audio_conditionning is None:
            raw_audio_conditionning = 0.0
        # Embedding music tokens
        music_tokens = music_tokens.long()
        hidden_states = self.embed_tokens(music_tokens)
        # Adding raw_audio_conditionning to hidden states
        hidden_states = hidden_states + raw_audio_conditionning

        # Permute dimensions for upsampling
        hidden_states = hidden_states.permute(0, 2, 1)
        # Upsample using upsampler
        hidden_states = self.upsampler(hidden_states)
        # Restore original dimensions after upsampling
        hidden_states = hidden_states.permute(0, 2, 1)
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
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
        # Parameters for range embedding
        self.n_time = n_time
        self.embed_dim = embed_dim
        # Embedding layer
        self.emb = nn.Embedding(embed_dim, out_width)
        # Range for interpolation
        self.pos_min, self.pos_max = range
        # Clamp option
        self.clamp = clamp
    # 定义一个方法，用于将位置范围[pos_start, pos_end]映射到对应的嵌入向量
    def forward(self, pos_start, pos_end=None):
        # 检查pos_start的形状是否为二维，如果不是则抛出类型错误异常
        if not len(pos_start.shape) == 2:
            raise TypeError(f"Expected shape with 2 dims, got {pos_start.shape}")
        # 检查[pos_start,pos_end]范围是否在[pos_min, pos_max)内，如果不是则抛出类型错误异常
        if not (self.pos_min <= pos_start).all() and (pos_start < self.pos_max).all():
            raise TypeError(f"Range is [{self.pos_min},{self.pos_max}), got {pos_start}")

        # 将pos_start转换为浮点型
        pos_start = pos_start.float()
        # 如果pos_end不为None，则进行一些处理
        if pos_end is not None:
            if self.clamp:
                # 如果self.clamp为真，则将pos_end限制在[pos_min, pos_max)范围内
                pos_end = pos_end.clamp(self.pos_min, self.pos_max)

            # 将pos_end转换为浮点型
            pos_end = pos_end.float()
        # 对位置数据进行插值，使得[pos_start, ..., pos_end]与长度为n_ctx的位置张量相对应
        n_time = self.n_time
        if n_time != 1:
            # 计算插值的系数
            interpolation = (
                torch.arange(0, n_time, dtype=torch.float, device=pos_start.device).view(1, n_time) / n_time
            )
            # 计算位置张量
            position = pos_start + (pos_end - pos_start) * interpolation
        else:
            position = pos_start

        # 将位置数据正规化到[0,1]范围内
        normalised_position = (position - self.pos_min) / (self.pos_max - self.pos_min)
        # 将正规化的位置数据映射到对应的嵌入向量的索引
        bins_ = (self.embed_dim * normalised_position).floor().long().detach()
        # 返回嵌入向量
        return self.emb(bins_)
class JukeboxLabelConditioner(nn.Module):
    # 初始化标签调节器类，接受config和include_time_signal参数
    def __init__(self, config, include_time_signal):
        # 调用父类的初始化方法
        super().__init__()

        # 从config中获取隐藏层的维度
        embed_dim = config.hidden_size
        # 从config中获取时间维度
        timing_dims = config.timing_dims
        # 从config中获取采样率
        sampling_rate = config.sampling_rate
        # 从config中获取音乐流派和艺术家的维度
        nb_genres, nb_artists = config.metadata_dims
        # 从config中获取音乐token的形状
        music_tokens_shape = config.n_ctx

        # 从config中获取最大音乐流派数
        self.max_nb_genres = config.max_nb_genres
        # 创建音乐流派的嵌入层
        self.bow_genre_emb = nn.Embedding(nb_genres, embed_dim)
        # 创建艺术家的嵌入层
        self.artist_emb = nn.Embedding(nb_artists, embed_dim)
        # 设置是否包含时间信号的标志位
        self.include_time_signal = include_time_signal
        # 如果包含时间信号
        if self.include_time_signal:
            # 根据config设置长度范围
            total_length_range = (config.min_duration * sampling_rate, config.max_duration * sampling_rate)
            # 设置绝对位置范围
            absolute_pos_range = (0.0, config.max_duration * sampling_rate)
            # 设置相对位置范围
            relative_pos_range = (0.0, 1.0)
            # 创建总长度的嵌入层
            self.total_length_emb = JukeboxRangeEmbedding(1, timing_dims, total_length_range, embed_dim)
            # 创建绝对位置的嵌入层
            self.absolute_pos_emb = JukeboxRangeEmbedding(
                music_tokens_shape, timing_dims, absolute_pos_range, embed_dim
            )
            # 创建相对位置的嵌入层
            self.relative_pos_emb = JukeboxRangeEmbedding(
                music_tokens_shape, timing_dims, relative_pos_range, embed_dim, clamp=True
            )

    # 前向传播函数
    def forward(self, metadata):
        # 提取metadata中的总长度、偏移、长度、艺术家和流派等信息
        total_length = metadata[:, 0:1]
        offset = metadata[:, 1:2]
        length = metadata[:, 2:3]
        artist = metadata[:, 3:4]
        genre = metadata[:, 4:]

        # 艺术家的嵌入计算
        artist_emb = self.artist_emb(artist)
        # 空的流派用-1表示，对其进行掩码
        mask = (genre >= 0).float().unsqueeze(2)
        genre_emb = (self.bow_genre_emb(genre.clamp(0)) * mask).sum(dim=1, keepdim=True)
        start_emb = genre_emb + artist_emb

        # 计算位置嵌入
        if self.include_time_signal:
            start, end = offset, offset + length
            total_length = total_length.float()
            start = start.float()
            end = end.float()
            pos_emb = (
                self.total_length_emb(total_length)
                + self.absolute_pos_emb(start, end)
                + self.relative_pos_emb(start / total_length, end / total_length)
            )
        else:
            pos_emb = None
        return start_emb, pos_emb


class JukeboxPrior(PreTrainedModel):
    """
    The JukeboxPrior class, which is a wrapper around the various conditioning and the transformer. JukeboxPrior can be
    seen as language models trained on music. They model the next `music token` prediction task. If a (lyric) `encoderù
    is defined, it also models the `next character` prediction on the lyrics. Can be conditionned on timing, artist,
    genre, lyrics and codes from lower-levels Priors.
    Args:
        config (`JukeboxPriorConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        level (`int`, *optional*):
            Current level of the Prior. Should be in range `[0,nb_priors]`.
        nb_priors (`int`, *optional*, defaults to 3):
            Total number of priors.
        vqvae_encoder (`Callable`, *optional*):
            Encoding method of the VQVAE encoder used in the forward pass of the model. Passing functions instead of
            the vqvae module to avoid getting the parameters.
        vqvae_decoder (`Callable`, *optional*):
            Decoding method of the VQVAE decoder used in the forward pass of the model. Passing functions instead of
            the vqvae module to avoid getting the parameters.
    """
    
    # 定义配置类为JukeboxPriorConfig
    config_class = JukeboxPriorConfig

    # 初始化模型参数的函数
    def _init_weights(self, module):
        # 获取初始化尺度
        init_scale = self.config.init_scale

        # 若module类型为nn.Embedding
        if isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        # 若module类型为JukeboxConv1D
        elif isinstance(module, JukeboxConv1D):
            # 如果配置为zero_out，则将权重设为0，否则使用正态分布初始化权重
            if self.config.zero_out:
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        # 若module类型为JukeboxPositionalEmbedding
        elif isinstance(module, JukeboxPositionalEmbedding):
            # 使用正态分布初始化位置嵌入参数
            module.pos_emb.data.normal_(mean=0.0, std=0.01 * init_scale)
        # 若module类型为JukeboxRangeEmbedding
        elif isinstance(module, JukeboxRangeEmbedding):
            # 使用正态分布初始化范围嵌入权重
            module.emb.weight.data.normal_(mean=0.0, std=0.01 * init_scale)
        # 若module类型为JukeboxConditionalAutoregressive且含有lm_head属性
        elif isinstance(module, JukeboxConditionalAutoregressive) and hasattr(module, "lm_head"):
            # 使用正态分布初始化lm_head权重
            module.lm_head.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        # 若module类型为JukeboxConditionalAutoregressive且含有start_token属性
        elif isinstance(module, JukeboxConditionalAutoregressive) and hasattr(module, "start_token"):
            # 使用正态分布初始化start_token参数
            module.start_token.data.normal_(mean=0.0, std=0.01 * init_scale)
        # 若module类型为JukeboxResConv1DBlock且配置为zero_out
        elif isinstance(module, JukeboxResConv1DBlock) and self.config.zero_out:
            # 将conv1d_2层的权重和偏置都设为0
            module.conv1d_2.weigth.data.zero_()
            module.conv1d_2.bias.data.zero_()
        
        # 若module类型为nn.LayerNorm
        if isinstance(module, nn.LayerNorm):
            # 将偏置置为0，将权重置为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 若module类型为nn.Linear且含有bias
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置置为0
            module.bias.data.zero_()
````
    def get_metadata(self, labels, start, total_length, offset, get_indices=False):
        # 复制标签信息，创建一个元数据的副本
        metadata = labels.clone()
        # 将元数据中的总长度字段设置为给定的 total_length 
        metadata[:, 0] = total_length
        # 将样本长度设置为和该级别相匹配的长度
        metadata[:, 2] = int(self.sample_length)

        # 设置偏移量
        metadata[:, 1:2] = int(offset * self.raw_to_tokens) + int(start * self.raw_to_tokens)
        # 在此处，由于元数据具有完整的令牌列表，我们只需选择相关的部分

        # 设置歌词令牌
        metadata, indices = self.set_metadata_lyric_tokens(metadata)
        if get_indices:
            return metadata, indices
        else:
            return metadata

    def set_metadata_lyric_tokens(self, labels):
        """
        处理完整的标签，只检索相关的歌词令牌并保留元数据调节令牌。
        """
        if self.nb_relevant_lyric_tokens > 0:
            tokens_list = torch.zeros(
                (labels.shape[0], self.nb_relevant_lyric_tokens), dtype=torch.long, device=labels.device
            )
            indices_list = []  # 当前字符在原始数组中的索引
            for idx in range(labels.shape[0]):
                full_tokens = labels.clone()[:, 4 + self.metadata_embedding.max_nb_genres :]
                total_length, offset, duration = labels[idx, 0], labels[idx, 1], labels[idx, 2]
                tokens, indices = get_relevant_lyric_tokens(
                    full_tokens, self.nb_relevant_lyric_tokens, total_length, offset, duration
                )
                tokens_list[idx, :] = tokens
                indices_list.append(indices)

            return (
                torch.cat((labels[:, : 4 + self.metadata_embedding.max_nb_genres], tokens_list), dim=-1),
                indices_list,
            )
        else:
            return labels, None

    def get_music_tokens_conds(self, music_tokens, start, end):
        """
        提取当前级别的条件音乐令牌。
        """
        if self.level != 0:
            music_tokens_cond = music_tokens[self.level - 1]
            music_tokens = music_tokens_cond[:, start // self.cond_downsample : end // self.cond_downsample]
            missing_cond_len = self.n_ctx // self.cond_downsample - music_tokens_cond[-1].shape[-1]
            if missing_cond_len > 0:
                init_cond = torch.zeros(1, missing_cond_len).to(music_tokens_cond.device)
                music_tokens_cond = torch.cat((music_tokens_cond, init_cond), dim=-1).long()
            music_tokens_conds = [music_tokens_cond]
        else:
            music_tokens_conds = None
        return music_tokens_conds
    # 对输入 tokens 进行预处理，以考虑词典合并。embed_dim_shift 表示音乐 tokens 应该偏移的数量。它等于 `lyric_vocab_size`。
    def prior_preprocess(self, tokens, conds):
        # 获取批量大小
        batch_size = tokens[0].shape[0]
        # 对每个 tokens 进行偏移操作
        for i in range(len(tokens)):
            tokens[i] = (tokens[i] + int(self.embed_dim_shift[i])).view(batch_size, -1)

        # 对每个条件进行处理，如果条件为空，则填充为0
        for i in range(len(conds)):
            if conds[i] is None:
                conds[i] = torch.zeros(
                    (batch_size, self.input_shapes[i], self.width), dtype=tokens[0].dtype, device=tokens[0].device
                )

        # 拼接处理后的 tokens 和条件，并返回
        return torch.cat(tokens, dim=1), torch.cat(conds, dim=1)

    # 对输入 tokens 进行后处理，如果模型使用编码器-解码器架构，则会将音乐 tokens 恢复原状。由于嵌入层是共享的，`prior_embed_dim_shift` 将音乐 token id 偏移 `lyric_vocab_size`。
    def prior_postprocess(self, tokens):
        # 获取批量大小
        batch_size = tokens.shape[0]
        # 分割 tokens
        dims = (self.input_shapes[0], tokens.shape[1] - self.input_shapes[0])
        tokens = list(torch.split(tokens, dims, dim=1))

        # 一些输入 tokens 可能已经经过偏移以适应词汇的合并
        for i in range(len(tokens)):
            bins_shift = int(self.embed_dim_shift[i])
            # 将 tokens 进行反向偏移，并做限制，保证不小于0
            tokens[i] = (tokens[i] - bins_shift).view(batch_size, -1)
            tokens[i] = torch.clamp(tokens[i], min=0)
            # 如果不屏蔽损失，模型可能会生成歌词/音符 tokens，这些 tokens 现在被 bin_shift 偏移小于0
        # 仅返回音乐 tokens
        return tokens[-1]

    # 嵌入上层音乐 tokens 并将其上采样以作为音频条件
    def embed_tokens(self, music_tokens_conds):
        music_tokens_conds = music_tokens_conds[: self.cond_level + 1]
        audio_conditioning = None
        for music_tokens_cond, conditioner_block in reversed(list(zip(music_tokens_conds, [self.conditioner_blocks]))):
            audio_conditioning = conditioner_block(music_tokens_cond, audio_conditioning)
        return audio_conditioning

    # 使用 VQVAE 的编码器对隐藏状态（原始音频）进行编码。返回潜在状态。
    def encode(self, hidden_states, start_level=None, end_level=None, bs_chunks=1):
        # 如果未指定起始级别，则默认使用 self.level
        if start_level is None:
            start_level = self.level
        # 如果未指定终止级别，则默认使用 self.levels
        if end_level is None:
            end_level = self.levels
        # 获取潜在状态
        with torch.no_grad():
            latent_states = self.vqvae_encoder(
                hidden_states, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks
            )
        # 返回潜在状态
        return latent_states
    def decode(self, music_tokens, start_level=None, end_level=None, bs_chunks=1):
        """
        将音乐编码的序列解码为原始音频。
        """
        # 如果未指定起始级别，则使用默认级别
        if start_level is None:
            start_level = self.level
        # 如果未指定结束级别，则使用默认级别
        if end_level is None:
            end_level = self.levels
        # 使用 torch.no_grad() 上下文管理器，确保不进行梯度计算
        with torch.no_grad():
            # 调用 VQ-VAE 解码器进行解码
            output = self.vqvae_decoder(
                music_tokens, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks
            )
        return output

    def get_cond(self, music_tokens_conds, metadata):
        """
        将输入的 tokens 转换为输入的嵌入。将歌词从元数据中分离出来。歌词 tokens 可以为 None。
        """
        # 如果有元数据
        if metadata is not None:
            # 计算非歌词部分的标签数量
            n_labels = metadata.shape[1] - self.nb_relevant_lyric_tokens
            # 将元数据分为非歌词部分和歌词部分
            metadata, lyric_tokens = metadata[:, :n_labels], metadata[:, n_labels:]
        else:
            metadata, lyric_tokens = None, None
        # 如果模型使用元数据作为条件
        if self.metadata_conditioning:
            # 将元数据转换为嵌入表示
            metadata_conditioning, metadata_pos = self.metadata_embedding(metadata)
        else:
            metadata_conditioning, metadata_pos = None, None
        # 如果模型使用音频 tokens 作为条件
        if self.audio_conditioning:
            # 将音频 tokens 转换为嵌入表示
            audio_conditioning = self.embed_tokens(music_tokens_conds)
        else:
            audio_conditioning = metadata_pos
        return audio_conditioning, metadata_conditioning, lyric_tokens

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
        # 用于生成样本，暂时未实现
        pass

    def get_encoder_states(self, lyric_tokens, sample=False):
        """
        获取将由解码器注意到的歌词编码器的最后隐藏状态。通过歌词编码器前向传播。
        """
        # 如果模型使用歌词作为条件
        if self.nb_relevant_lyric_tokens != 0 and self.lyric_conditioning:
            # 如果需要对输入 tokens 进行采样
            if sample:
                # 将编码器移至相同设备上的 lyric_tokens
                self.encoder = self.encoder.to(lyric_tokens.device)
            # 将歌词 tokens 输入到编码器中
            lyric_acts = self.encoder(lyric_tokens, None, None, None)
            # 将编码器输出投影到输入维度
            lyric_acts = self.encoder.proj_in(lyric_acts)
            # 对编码器输出进行最后一层归一化
            last_encoder_hidden_states = self.encoder.final_layer_norm(lyric_acts)
        else:
            last_encoder_hidden_states = None
        return last_encoder_hidden_states

    def get_encoder_loss(self, last_encoder_hidden_states, target_lyrics):
        """
        计算歌词编码器的损失：下一个歌词 token 的预测。
        """
        # 如果模型使用歌词作为条件
        if self.lyric_conditioning:
            # 使用语言模型头对编码器的最后隐藏状态进行映射
            last_encoder_hidden_states = self.encoder.lm_head(last_encoder_hidden_states)
            # 计算交叉熵损失
            encoder_loss = nn.functional.cross_entropy(
                last_encoder_hidden_states.view(-1, self.encoder_dim), target_lyrics.view(-1)
            ) / np.log(2.0)
        else:
            encoder_loss = torch.tensor(0.0, device=last_encoder_hidden_states.device)
        return encoder_loss
    def forward_tokens(
        self, music_tokens, music_tokens_conds=[], metadata=None, get_preds=False, get_attn_weights=False
    ):
        """
        Applies a forward pass using the conditioning tokens. Different from the classic forward as it does not use the
        vqvae's encoding layers.
        """
        # 如果需要获取注意力权重，则设置transformer记录注意力权重
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        # 获取音频条件、元数据条件和歌词 tokens
        audio_conditioning, metadata_conditioning, lyric_tokens = self.get_cond(music_tokens_conds, metadata)

        # 如果是编码器-解码器结构
        if self.is_encoder_decoder:  
            # 前处理返回完整的 tokens（歌词和音乐 tokens），并进行了偏移
            tokens, audio_conditioning = self.prior_preprocess(
                [lyric_tokens, music_tokens], [None, audio_conditioning]
            )
            # 使用 prior 模型进行编码器-解码器操作，获取损失和预测结果
            (encoder_loss, next_token_prediction_loss), preds = self.prior(
                tokens, audio_conditioning, metadata_conditioning, get_sep_loss=True, get_preds=get_preds
            )
        else:
            # 获取最后一个编码器隐藏状态
            last_encoder_hidden_states = self.get_encoder_states(lyric_tokens)
            # 获取编码器损失
            encoder_loss = self.get_encoder_loss(last_encoder_hidden_states, lyric_tokens)
            # 使用 prior 模型进行前向操作，获取下一个 token 的预测损失和预测结果
            next_token_prediction_loss, preds = self.prior(
                music_tokens,
                audio_conditioning,
                metadata_conditioning,
                last_encoder_hidden_states,
                get_preds=get_preds,
            )
        # 计算总损失
        loss = self.encoder_loss_fraction * encoder_loss * self.nb_relevant_lyric_tokens / self.total_loss_dims
        loss += next_token_prediction_loss * self.next_token_prediction_loss_dims / self.total_loss_dims

        # 设置指标
        metrics = {
            "bpd": next_token_prediction_loss.clone().detach(),
            "encoder_loss": encoder_loss.clone().detach(),
            "next_token_prediction_loss": next_token_prediction_loss.clone().detach(),
        }
        # 如果需要获取预测结果，则设置预测结果指标
        if get_preds:
            metrics["preds"] = preds.clone().detach()
        # 如果需要获取注意力权重，则返回保存的注意力权重并关闭transformer记录注意力权重
        if get_attn_weights:
            saved_attn_weights = self.prior.transformer.saved_attn_weights
            self.prior.transformer.set_record_attn(False)
            return saved_attn_weights
        else:
            # 否则返回总损失和指标
            return loss, metrics

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: Optional[List[torch.LongTensor]],
        decode: Optional[bool] = False,
        get_preds: Optional[bool] = False,
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
        # 获取批处理大小
        batch_size = hidden_states.shape[0]
        # 使用`vqvae`编码器对隐藏状态进行编码
        music_tokens, *music_tokens_conds = self.encode(hidden_states, bs_chunks=batch_size)
        # 计算损失和指标
        loss, metrics = self.forward_tokens(
            music_tokens=music_tokens,
            music_tokens_conds=music_tokens_conds,
            metadata=metadata,
            get_preds=get_preds,
        )
        # 如果需要解码，则将编码的数据解码为tokens
        if decode:
            dequantised_states = self.decode([music_tokens, *music_tokens_conds])
        # 否则返回None
        else:
            dequantised_states = None
        # 返回解码后的数据、损失和指标
        return dequantised_states, loss, metrics
# JukeboxPreTrainedModel 类用于处理权重初始化，并提供一个简单的接口用于下载和加载预训练模型
class JukeboxPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # JukeboxPreTrainedModel 类对应的配置类为 JukeboxConfig
    config_class = JukeboxConfig
    # JukeboxPreTrainedModel 类的基础模型前缀为 "jukebox"
    base_model_prefix = "jukebox"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False

    # 对模块进行初始化权重操作
    def _init_weights(self, module):
        if isinstance(module, JukeboxPrior) or isinstance(module, JukeboxVQVAE):
            # 应用模块自己的初始化权重函数
            module.apply(module._init_weights)

    # JukeboxPreTrainedModel 类的构造函数
    def __init__(self, *inputs, **kwargs):
        # 调用父类 PreTrainedModel 的构造函数
        super().__init__(*inputs, **kwargs)

# JUKEBOX_SAMPLING_INPUT_DOCSTRING 是用于描述输入格式的文档字符串
JUKEBOX_SAMPLING_INPUT_DOCSTRING = r"""
            labels (`List[torch.LongTensor]` of length `n_sample`, and shape `(self.levels, self.config.max_nb_genre + lyric_sequence_length)` :
                List of metadata such as `artist_id`, `genre_id` and the full list of lyric tokens which are used to
                condition the generation.
            sampling_kwargs (`Dict[Any]`):
                Various additional sampling arguments that are used by the `_sample` function. A detail list of the
                arguments can bee seen in the [`_sample`] function documentation.
"""

# JukeboxModel 是用于音乐生成的基本 JUKEBOX 模型，支持 4 种采样技术：`primed_sample`, `upsample`, `continue_sample` 和 `ancestral_sample`
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

    # JukeboxModel 类的构造函数
    def __init__(self, config):
        # 调用父类 JukeboxPreTrainedModel 的构造函数
        super().__init__(config)
        vqvae_config = config.vqvae_config
        # 创建 JukeboxVQVAE 对象
        self.vqvae = JukeboxVQVAE(vqvae_config)
        # 设置共享参数
        self.set_shared_params(config)
        # 创建一个包含多个 JukeboxPrior 对象的模块列表
        self.priors = nn.ModuleList(
            [JukeboxPrior(config.prior_configs[level], level) for level in range(config.nb_priors)]
        )

    # 设置共享参数的方法
    def set_shared_params(self, model_config):
        """
        Initialises the parameters that are shared. This has to be done here because the list of `JukeboxPriorConfig`
        is nest, and is thus unreachable in the `from_dict` function
        """
        # 遍历优先配置列表，设置相关参数
        for config in model_config.prior_configs:
            config.sampling_rate = model_config.sampling_rate
            config.timing_dims = model_config.timing_dims
            config.min_duration = model_config.min_duration
            config.max_duration = model_config.max_duration
            config.max_nb_genres = model_config.max_nb_genres
            config.metadata_conditioning = model_config.metadata_conditioning

    # 解码方法，用于解码音乐令牌
    def decode(self, music_tokens, start_level=0, end_level=None, bs_chunks=1):
        return self.vqvae.decode(music_tokens, start_level, end_level, bs_chunks)
    # 调用 VQ-VAE 模型的 encode 方法来对音频进行编码
    def encode(self, input_audio, start_level=0, end_level=None, bs_chunks=1):
        return self.vqvae.encode(input_audio, start_level, end_level, bs_chunks)

    # 将输入对象 obj 分割成若干个大小为 split_size 的部分
    def split_batch(self, obj, n_samples, split_size):
        # 根据总样本数和分割大小计算需要分割的次数
        n_passes = (n_samples + split_size - 1) // split_size
        if isinstance(obj, torch.Tensor):  # 如果输入对象是 PyTorch 的 Tensor 类型
            return torch.split(obj, split_size, dim=0)  # 利用 PyTorch 的 split 方法进行分割
        elif isinstance(obj, list):  # 如果输入对象是列表
            # 对列表中的每个元素进行相同的分割操作，然后将结果合并成一个列表
            return list(zip(*[torch.split(item, split_size, dim=0) for item in obj]))
        elif obj is None:  # 如果输入对象是 None
            return [None] * n_passes  # 返回包含 n_passes 个 None 的列表
        else:
            raise TypeError("Unknown input type")  # 抛出类型错误异常

    # 从 music_tokens 中的 level 级别开始对 tokens_to_sample 个新令牌进行抽样
    def sample_partial_window(
        self, music_tokens, labels, offset, sampling_kwargs, level, tokens_to_sample, max_batch_size
    ):
        prior = self.priors[level]  # 根据 level 获取先验对象
        sampled_tokens = music_tokens[level]  # 获取 music_tokens 中 level 级别的令牌
        n_ctx = prior.n_ctx  # 获取先验对象的上下文大小
        nb_sampled_tokens = sampled_tokens.shape[1]  # 获取已抽样令牌的数量
        if nb_sampled_tokens < n_ctx - tokens_to_sample:  # 判断已抽样令牌数量是否小于 n_ctx - tokens_to_sample
            sampling_kwargs["sample_tokens"] = nb_sampled_tokens + tokens_to_sample
            start = 0
        else:
            sampling_kwargs["sample_tokens"] = n_ctx
            start = nb_sampled_tokens - n_ctx + tokens_to_sample

        return self.sample_single_window(music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size)

    # 从 music_tokens 中的 level 级别开始在位置 start 处抽样一个长度为 n_ctx 的窗口
    # 从音乐令牌中对单个窗口进行采样
    def sample_single_window(self, music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size):
        # 获取先验分布
        prior = self.priors[level]
        # 音乐令牌的数量
        n_samples = music_tokens[0].shape[0]
        # 先验分布的上下文长度
        n_ctx = prior.n_ctx
        # 计算窗口的结束位置
        end = start + n_ctx
        # 获取当前级别已经采样的音乐令牌
        previous_sampled_tokens = music_tokens[level][:, start:end]

        # 检查是否有指定的样本令牌数
        sample_tokens = sampling_kwargs.get("sample_tokens", None)
        if "sample_tokens" in sampling_kwargs:
            sample_tokens = end - start

        # 条件音乐令牌的数量
        conditioning_tokens = previous_sampled_tokens.shape[1]
        # 新的音乐令牌数量
        new_tokens = sample_tokens - previous_sampled_tokens.shape[1]

        # 打印日志，记录采样信息
        logger.info(
            f"Sampling {sample_tokens} tokens for [{start},{start+sample_tokens}]. Conditioning on"
            f" {conditioning_tokens} tokens"
        )

        if new_tokens <= 0:
            # 没有新的令牌需要采样
            return music_tokens

        # 获取上一级的音乐令牌条件
        music_tokens_conds = prior.get_music_tokens_conds(music_tokens, start, end)
        # 如果没有上一级，应该返回None！

        # 设置元数据的偏移、采样长度和歌词令牌
        metadata = prior.get_metadata(labels, start, self.total_length, offset)

        # 将音乐令牌、条件音乐令牌和元数据拆分成批次
        music_tokens_list = self.split_batch(previous_sampled_tokens, n_samples, max_batch_size)
        music_tokens_conds_list = self.split_batch(music_tokens_conds, n_samples, max_batch_size)
        metadata_list = self.split_batch(metadata, n_samples, max_batch_size)
        tokens = []
        iterator = tqdm(zip(music_tokens_list, music_tokens_conds_list, metadata_list), leave=False)
        for music_tokens_i, music_tokens_conds_i, metadata_i in iterator:
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
        sampled_tokens = torch.cat(tokens, dim=0)

        # 更新音乐令牌，添加新的采样
        music_tokens_new = sampled_tokens[:, -new_tokens:]
        music_tokens[level] = torch.cat([music_tokens[level], music_tokens_new], dim=1)
        return music_tokens

    # 在level级别上以hop_length为步长采样total_length个令牌
    def sample_level(
        self, music_tokens, labels, offset, sampling_kwargs, level, total_length, hop_length, max_batch_size
        ):
        # 如果总长度大于当前先验模型的上下文长度
        if total_length >= self.priors[level].n_ctx:
            # 获取开始位置的迭代器
            iterator = get_starts(total_length, self.priors[level].n_ctx, hop_length)
            # 遍历开始位置并对单个窗口进行采样
            for start in iterator:
                music_tokens = self.sample_single_window(
                    music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size
                )

        else:
            # 对部分窗口进行采样
            music_tokens = self.sample_partial_window(
                music_tokens, labels, offset, sampling_kwargs, level, total_length, max_batch_size
            )
        # 返回采样结果
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
    # 生成祖先样本，返回一个包含生成音乐 tokens 的列表
    def ancestral_sample(self, labels, n_samples=1, **sampling_kwargs) -> List[torch.LongTensor]:
        """
        Example:

        ```py
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

        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors))))
        # 初始化一个包含音乐 tokens 的列表
        music_tokens = [
            torch.zeros(n_samples, 0, dtype=torch.long, device=labels[0].device) for _ in range(len(self.priors))
        ]
        # 生成祖先样本
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
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
    # 生成先前生成 tokens 的延续
    def continue_sample(self, music_tokens, labels, **sampling_kwargs) -> List[torch.LongTensor]:
        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors)))
        # 生成延续音乐 tokens
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
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
    # 使用给定水平上的先验对音乐 tokens 进行上采样
    def upsample(self, music_tokens, labels, **sampling_kwargs) -> List[torch.LongTensor]:
        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors) - 1)))
        # 对音乐 tokens 进行上采样
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
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
    # 定义 primed_sample 方法，接受原始音频和标签作为输入参数，并返回音乐 token 列表
    def primed_sample(self, raw_audio, labels, **sampling_kwargs) -> List[torch.LongTensor]:
        # 获取采样级别，默认为所有级别
        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors))))
        # 将 VQ-VAE 模型转移到原始音频所在设备，并转换为 float 类型
        self.vqvae.to(raw_audio.device).float()
        # 关闭梯度计算上下文管理器，避免在推理阶段进行梯度计算
        with torch.no_grad():
            # 使用 VQ-VAE 模型编码原始音频为音乐 token
            music_tokens = self.vqvae.encode(
                raw_audio, start_level=0, end_level=len(self.priors), bs_chunks=raw_audio.shape[0]
            )
        # 对音乐 token 进行采样
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        # 返回采样后的音乐 token 列表
        return music_tokens
```