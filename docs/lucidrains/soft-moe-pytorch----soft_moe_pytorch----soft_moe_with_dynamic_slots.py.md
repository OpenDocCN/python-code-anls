# `.\lucidrains\soft-moe-pytorch\soft_moe_pytorch\soft_moe_with_dynamic_slots.py`

```py
# 导入数学库
import math

# 导入 PyTorch 库
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, einsum, Tensor

# 导入 einops 库中的函数
from einops import rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 对输入张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 将张量填充到指定的倍数
def pad_to_multiple(
    tensor,
    multiple,
    dim = -1,
    value = 0
):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple

    if m.is_integer():
        return False, tensor

    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# 归一化模块

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma

# 专家模块

# 创建前馈神经网络
def FeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

# GEGLU 激活函数
class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

# 创建 GLU 前馈神经网络
def GLUFeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

# 主类

class DynamicSlotsSoftMoE(Module):
    def __init__(
        self,
        dim,
        *,
        num_experts = 4,
        expert_mult = 4,
        dropout = 0.,
        geglu = False
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        self.num_experts = num_experts

        # 将输入映射到槽位嵌入
        self.to_slot_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_experts, bias = False),
            Rearrange('b n (e d) -> b e n d', e = num_experts),
            RMSNorm(dim)
        )

        # 根据是否使用 GEGLU 创建专家模块
        expert_klass = GLUFeedForward if geglu else FeedForward

        # 创建多个专家模块
        self.experts = nn.ModuleList([
            expert_klass(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)
        ])
    # 定义前向传播函数，接受输入 x 和 mask（可选）
    def forward(self, x, mask = None):
        """
        einstein notation
        b - batch
        n - sequence length
        e - number of experts
        s - number of slots per expert
        d - feature dimension
        """

        # 获取输入 x 的序列长度、是否为图像、专家数量等信息
        seq_len, is_image, num_experts = x.shape[-2], x.ndim == 4, self.num_experts

        # 如果输入为图像，则重新排列维度
        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')

        # 对输入进行归一化处理
        x = self.norm(x)

        # 动态槽嵌入
        # 首先对连续的令牌进行平均，然后将每个位置投影到相应数量的专家槽令牌
        # 槽的数量应该约等于序列长度，就像通常的具有 1 个专家的 MoE 一样

        # 检查是否需要填充，对输入进行填充
        is_padded, x = pad_to_multiple(x, num_experts, dim = -2)

        # 如果需要填充，且没有提供 mask，则创建一个全为 True 的 mask
        if is_padded:
            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool)

            _, mask = pad_to_multiple(mask, num_experts, dim = -1, value = False)

        # 对输入进行分段处理
        x_segmented = rearrange(x, 'b (n e) d -> b n e d', e = num_experts)

        # 如果存在 mask，则根据 mask 进行填充
        if exists(mask):
            segmented_mask = rearrange(mask, 'b (n e) -> b n e', e = num_experts)
            x_segmented = x_segmented.masked_fill(~rearrange(segmented_mask, '... -> ... 1'), 0.)

        # 执行带有 mask 的均值计算
        if exists(mask):
            num = reduce(x_segmented, 'b n e d -> b n d', 'sum')
            den = reduce(segmented_mask.float(), 'b n e -> b n 1', 'sum').clamp(min = 1e-5)
            x_consecutive_mean = num / den
            slots_mask = segmented_mask.any(dim = -1)
        else:
            x_consecutive_mean = reduce(x_segmented, 'b n e d -> b n d', 'mean')

        # 投影以获取动态槽嵌入
        slot_embeds = self.to_slot_embeds(x_consecutive_mean)

        logits = einsum('b n d, b e s d -> b n e s', x, slot_embeds)

        # 考虑键填充 mask

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            slots_mask = rearrange(slots_mask, 'b s -> b 1 1 s')

            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
            logits = logits.masked_fill(~slots_mask, -torch.finfo(logits.dtype).max)

        # 获取分发权重和组合权重（在正确的维度上进行 softmax）

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # 通过使用上述分发权重对输入令牌进行加权平均，得到槽
        slots = einsum('b n d, b n e s -> e b s d', x, dispatch_weights)

        # 将每个专家的槽路由到每个专家

        out = []
        for slots_per_expert, expert in zip(slots, self.experts):
            out.append(expert(slots_per_expert))

        out = torch.stack(out)

        # 合并输出

        out = rearrange(out, 'e b s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        # 如果输入为图像，则恢复原始维度
        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')

        return out[:, :seq_len]
```