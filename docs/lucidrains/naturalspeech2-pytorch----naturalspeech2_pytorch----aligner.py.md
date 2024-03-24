# `.\lucidrains\naturalspeech2-pytorch\naturalspeech2_pytorch\aligner.py`

```py
from typing import Tuple
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat

from beartype import beartype
from beartype.typing import Optional

# 检查变量是否存在
def exists(val):
    return val is not None

# 定义对齐模型类
class AlignerNet(Module):
    """alignment model https://arxiv.org/pdf/2108.10447.pdf """
    def __init__(
        self,
        dim_in=80,
        dim_hidden=512,
        attn_channels=80,
        temperature=0.0005,
    ):
        super().__init__()
        self.temperature = temperature

        # 定义关键字层
        self.key_layers = nn.ModuleList([
            nn.Conv1d(
                dim_hidden,
                dim_hidden * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_hidden * 2, attn_channels, kernel_size=1, padding=0, bias=True)
        ])

        # 定义查询层
        self.query_layers = nn.ModuleList([
            nn.Conv1d(
                dim_in,
                dim_in * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_in * 2, dim_in, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_in, attn_channels, kernel_size=1, padding=0, bias=True)
        ])

    # 前向传播函数
    @beartype
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        mask: Optional[Tensor] = None
    ):
        key_out = keys
        for layer in self.key_layers:
            key_out = layer(key_out)

        query_out = queries
        for layer in self.query_layers:
            query_out = layer(query_out)

        key_out = rearrange(key_out, 'b c t -> b t c')
        query_out = rearrange(query_out, 'b c t -> b t c')

        attn_logp = torch.cdist(query_out, key_out)
        attn_logp = rearrange(attn_logp, 'b ... -> b 1 ...')

        if exists(mask):
            mask = rearrange(mask.bool(), '... c -> ... 1 c')
            attn_logp.data.masked_fill_(~mask, -torch.finfo(attn_logp.dtype).max)

        attn = attn_logp.softmax(dim = -1)
        return attn, attn_logp

# 填充张量函数
def pad_tensor(input, pad, value=0):
    pad = [item for sublist in reversed(pad) for item in sublist]  # Flatten the tuple
    assert len(pad) // 2 == len(input.shape), 'Padding dimensions do not match input dimensions'
    return F.pad(input, pad, mode='constant', value=value)

# 最大路径函数
def maximum_path(value, mask, const=None):
    device = value.device
    dtype = value.dtype
    if not exists(const):
        const = torch.tensor(float('-inf')).to(device)  # Patch for Sphinx complaint
    value = value * mask

    b, t_x, t_y = value.shape
    direction = torch.zeros(value.shape, dtype=torch.int64, device=device)
    v = torch.zeros((b, t_x), dtype=torch.float32, device=device)
    x_range = torch.arange(t_x, dtype=torch.float32, device=device).view(1, -1)

    for j in range(t_y):
        v0 = pad_tensor(v, ((0, 0), (1, 0)), value = const)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = torch.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = torch.where(index_mask.view(1,-1), v_max + value[:, :, j], const)

    direction = torch.where(mask.bool(), direction, 1)

    path = torch.zeros(value.shape, dtype=torch.float32, device=device)
    index = mask[:, :, 0].sum(1).long() - 1
    index_range = torch.arange(b, device=device)

    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1

    path = path * mask.float()
    path = path.to(dtype=dtype)
    return path

# 前向求和损失类
class ForwardSumLoss(Module):
    def __init__(
        self,
        blank_logprob = -1
    # 初始化类，继承父类的属性和方法
    ):
        super().__init__()
        # 设置空白标签的对数概率
        self.blank_logprob = blank_logprob

        # 创建 CTC 损失函数对象
        self.ctc_loss = torch.nn.CTCLoss(
            blank = 0,  # 设置空白标签的值为0
            zero_infinity = True  # 设置是否将无穷大值转换为零
        )

    # 前向传播函数
    def forward(self, attn_logprob, key_lens, query_lens):
        # 获取设备信息和空白标签对数概率
        device, blank_logprob  = attn_logprob.device, self.blank_logprob
        # 获取输入的最大键长度
        max_key_len = attn_logprob.size(-1)

        # 重新排列输入数据的维度为[query_len, batch_size, key_len]
        attn_logprob = rearrange(attn_logprob, 'b 1 c t -> c b t')

        # 添加空白标签
        attn_logprob = F.pad(attn_logprob, (1, 0, 0, 0, 0, 0), value = blank_logprob)

        # 转换为对数概率
        # 注意：屏蔽超出键长度的概率
        mask_value = -torch.finfo(attn_logprob.dtype).max
        attn_logprob.masked_fill_(torch.arange(max_key_len + 1, device=device, dtype=torch.long).view(1, 1, -1) > key_lens.view(1, -1, 1), mask_value)

        attn_logprob = attn_logprob.log_softmax(dim = -1)

        # 目标序列
        target_seqs = torch.arange(1, max_key_len + 1, device=device, dtype=torch.long)
        target_seqs = repeat(target_seqs, 'n -> b n', b = key_lens.numel())

        # 计算 CTC 损失
        cost = self.ctc_loss(attn_logprob, target_seqs, query_lens, key_lens)

        return cost
class BinLoss(Module):
    # 定义一个继承自 Module 的 BinLoss 类
    def forward(self, attn_hard, attn_logprob, key_lens):
        # 前向传播函数，接受注意力机制的硬分配、对数概率和键长度作为输入
        batch, device = attn_logprob.shape[0], attn_logprob.device
        # 获取 batch 大小和设备信息
        max_key_len = attn_logprob.size(-1)
        # 获取键的最大长度

        # 重新排列输入为 [query_len, batch_size, key_len]
        attn_logprob = rearrange(attn_logprob, 'b 1 c t -> c b t')
        attn_hard = rearrange(attn_hard, 'b t c -> c b t')
        # 重新排列注意力机制的输入形状

        mask_value = -torch.finfo(attn_logprob.dtype).max
        # 创建一个用于掩码的值

        attn_logprob.masked_fill_(torch.arange(max_key_len, device=device, dtype=torch.long).view(1, 1, -1) > key_lens.view(1, -1, 1), mask_value)
        # 使用掩码值对注意力对数概率进行填充
        attn_logprob = attn_logprob.log_softmax(dim = -1)
        # 对注意力对数概率进行 log_softmax 操作

        return (attn_hard * attn_logprob).sum() / batch
        # 返回加权后的结果除以 batch 大小

class Aligner(Module):
    # 定义一个继承自 Module 的 Aligner 类
    def __init__(
        self,
        dim_in,
        dim_hidden,
        attn_channels=80,
        temperature=0.0005
    ):
        # 初始化函数，接受输入维度、隐藏维度、注意力通道数和温度参数
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.attn_channels = attn_channels
        self.temperature = temperature
        # 设置类的属性

        self.aligner = AlignerNet(
            dim_in = self.dim_in, 
            dim_hidden = self.dim_hidden,
            attn_channels = self.attn_channels,
            temperature = self.temperature
        )
        # 初始化 AlignerNet 模型

    def forward(
        self,
        x,
        x_mask,
        y,
        y_mask
    ):
        # 前向传播函数，接受输入 x、x_mask、y、y_mask
        alignment_soft, alignment_logprob = self.aligner(y, rearrange(x, 'b d t -> b t d'), x_mask)
        # 使用 AlignerNet 模型计算软对齐和对数概率

        x_mask = rearrange(x_mask, '... i -> ... i 1')
        y_mask = rearrange(y_mask, '... j -> ... 1 j')
        attn_mask = x_mask * y_mask
        attn_mask = rearrange(attn_mask, 'b 1 i j -> b i j')
        # 生成注意力掩码

        alignment_soft = rearrange(alignment_soft, 'b 1 c t -> b t c')
        alignment_mask = maximum_path(alignment_soft, attn_mask)
        # 重新排列软对齐结果并计算最大路径

        alignment_hard = torch.sum(alignment_mask, -1).int()
        # 计算硬对齐结果
        return alignment_hard, alignment_soft, alignment_logprob, alignment_mask
        # 返回硬对齐结果、软对齐结果、对数概率和对齐掩码

if __name__ == '__main__':
    # 如果作为脚本运行
    batch_size = 10
    seq_len_y = 200   # 序列 y 的长度
    seq_len_x = 35
    feature_dim = 80  # 特征维度

    x = torch.randn(batch_size, 512, seq_len_x)
    y = torch.randn(batch_size, seq_len_y, feature_dim)
    y = y.transpose(1,2) #dim-1 is the channels for conv
    # 生成输入 x 和 y，并对 y 进行转置

    # 创建掩码
    x_mask = torch.ones(batch_size, 1, seq_len_x)
    y_mask = torch.ones(batch_size, 1, seq_len_y)

    align = Aligner(dim_in = 80, dim_hidden=512, attn_channels=80)
    # 初始化 Aligner 模型
    alignment_hard, alignment_soft, alignment_logprob, alignment_mas = align(x, x_mask, y, y_mask)
    # 进行对齐操作
```