# `.\lucidrains\soft-moe-pytorch\soft_moe_pytorch\soft_moe.py`

```py
# 导入 torch 库
import torch
# 从 torch.nn 中导入 Module 类
from torch.nn import Module
# 从 torch.nn.functional 中导入 F
import torch.nn.functional as F
# 从 torch.distributed 中导入 dist
import torch.distributed as dist
# 从 torch 中导入 nn, einsum, Tensor
from torch import nn, einsum, Tensor

# 从 einops 中导入 rearrange, pack, unpack
from einops import rearrange, pack, unpack

# 从 soft_moe_pytorch.distributed 中导入 AllGather, split_by_rank, gather_sizes, has_only_one_value
from soft_moe_pytorch.distributed import (
    AllGather,
    split_by_rank,
    gather_sizes,
    has_only_one_value
)

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# 将一个数均匀分成若干份
def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out

# 将一个张量按照指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将一个打包后的张量按照指定模式解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = - 1)

# 计算张量的累积和（exclusive）
def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

# 计算张量的对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 归一化

# LayerNorm 类
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# RMSNorm 类
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma

# expert

# 创建 FeedForward 网络
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

# GEGLU 类
class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

# 创建 GLUFeedForward 网络
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

# experts

# 专家类
class Experts(nn.Module):
    def __init__(
        self,
        experts,
        is_distributed = None,
        offload_unused_experts_to_cpu = True
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        # 是否将未使用的专家转移到 CPU，需要优化器处理梯度转换到正确设备
        self.offload_unused_experts_to_cpu = offload_unused_experts_to_cpu

        self.all_gather = AllGather()
        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    # 将所有专家转移到 CPU，除了指定的专家
    def all_experts_to_cpu_besides(self, selection):
        if not self.offload_unused_experts_to_cpu:
            return

        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = None
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        # 检查是否为分布式环境，默认为 self.is_distributed
        is_distributed = default(is_distributed, self.is_distributed)
        # 获取输入张量 x 的形状和专家数量
        shape, num_experts = x.shape, self.num_experts

        # 如果是分布式环境，则在批次维度上进行全局收集，暂时简单处理，后续优化
        if is_distributed:
            # 收集每个专家的序列大小
            seq_sizes = gather_sizes(x, dim=-2)
            assert has_only_one_value(seq_sizes), 'number of tokens per expert must be the same'

            # 在批次维度上进行全局收集
            x, batch_sizes = self.all_gather(x)
            total_batch_size = x.shape[0]

            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # 在当前 rank 上使用的专家

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim=-1)

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # 目前，剩余的机器不处理任何内容

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)
        else:
            num_experts_per_rank = num_experts
            expert_slice = slice(0, num_experts)

        # 如果是分布式的，每台机器只处理专家和批次的子集

        # 重新排列输入张量 x 的维度
        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            # 打包 x，获取打包后的形状
            x, expert_batch_packed_shape = pack_one(x, '* n d')
            x = x.split(num_experts_batches_across_ranks, dim=0)
            x = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, '(e b) n d -> e b n d', e=num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # 获取正在使用的专家

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # 将标记路由到适当的专家

        outs = []
        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x).requires_grad_()

        # 在合并的专家批次维度上进行全局收集，然后将批次维度拆分回来

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs = split_by_rank(outs)

        assert outs.shape == shape
        return outs
# 主类 SoftMoE
class SoftMoE(Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        *,
        seq_len = None,
        num_experts = 4,
        num_slots = None,
        expert_mult = 4,
        dropout = 0.,
        geglu = False,
        is_distributed = None,
        offload_unused_experts_to_cpu = True,
        use_layernorm = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 断言语句，确保 seq_len 或 num_slots 必须传入 SoftMoE
        assert exists(seq_len) ^ exists(num_slots), 'either seq_len, or num_slots must be passed into SoftMoE'

        # 如果 num_slots 为 None，则计算默认值
        num_slots = default(num_slots, seq_len // num_experts)

        # 根据 use_layernorm 的值选择不同的归一化类
        norm_klass = LayerNorm if use_layernorm else RMSNorm
        # 初始化 norm 层
        self.norm = norm_klass(dim)

        # 初始化 slot_norm 层
        self.slot_norm = norm_klass(dim)
        # 初始化 slot_embeds 参数
        self.slot_embeds = nn.Parameter(torch.randn(num_experts, num_slots, dim))

        # 根据 geglu 的值选择不同的 FeedForward 类
        expert_klass = GLUFeedForward if geglu else FeedForward

        # 初始化 experts 层
        self.experts = Experts(
            experts = [expert_klass(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)],
            is_distributed = is_distributed,
            offload_unused_experts_to_cpu = offload_unused_experts_to_cpu
        )

    # 前向传播函数
    def forward(self, x, mask = None, add_noise = False, noise_mult = 1.):
        """
        einstein notation
        b - batch
        n - sequence length
        e - number of experts
        s - number of slots per expert
        d - feature dimension
        """

        # 判断输入是否为单个 token
        is_single_token = x.ndim == 2
        # 判断输入是否为图像
        is_image = x.ndim == 4

        # 如果输入为图像，则重新排列维度
        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')
        # 如果输入为单个 token，则重新排列维度
        elif is_single_token:
            x = rearrange(x, 'b d -> b 1 d')

        # 对输入进行归一化
        x = self.norm(x)
        # 对 slot_embeds 进行归一化
        slot_embeds = self.slot_norm(self.slot_embeds)

        # 计算 logits
        logits = einsum('b n d, e s d -> b n e s', x, slot_embeds)

        # 添加噪音到 dispatch 和 combine gate logits，如果需要则进行退火
        if add_noise:
            noise = gumbel_noise(logits) * noise_mult
            logits = logits + noise

        # 处理 key padding mask
        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

        # 获取 dispatch 和 combine 权重（在正确的维度上进行 softmax）
        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # 通过使用上面的 dispatch 权重对输入 token 进行加权平均，得到 slots
        slots = einsum('b n d, b n e s -> b e s d', x, dispatch_weights)

        # 将每个专家的 slots 路由到每个专家
        out = self.experts(slots)

        # 合并输出
        out = rearrange(out, ' b e s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        # 如果输入为图像，则恢复原始维度
        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')
        # 如果输入为单个 token，则恢复原始维度
        elif is_single_token:
            out = rearrange(out, 'b 1 d -> b d')

        return out
```