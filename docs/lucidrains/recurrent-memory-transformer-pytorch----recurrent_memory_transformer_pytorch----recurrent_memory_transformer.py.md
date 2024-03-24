# `.\lucidrains\recurrent-memory-transformer-pytorch\recurrent_memory_transformer_pytorch\recurrent_memory_transformer.py`

```
# 导入数学库
import math
# 导入partial函数
from functools import partial
# 导入zip_longest函数
from itertools import zip_longest
# 导入nullcontext函数
from contextlib import nullcontext

# 导入类型提示相关库
from typing import Optional, List, Tuple

# 导入torch库
import torch
# 导入torch.nn.functional库
import torch.nn.functional as F
# 导入torch.nn、einsum、Tensor
from torch import nn, einsum, Tensor

# 导入rearrange、repeat、pack、unpack函数
from einops import rearrange, repeat, pack, unpack

# 导入Attend类
from recurrent_memory_transformer_pytorch.attend import Attend

# 定义常量Linear为nn.Linear函数的偏函数，不包含偏置
Linear = partial(nn.Linear, bias = False)

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 返回输入的第一个参数
def identity(t, *args, **kwargs):
    return t

# 返回输入参数中第一个不为None的值
def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

# 评估装饰器，用于在评估模式下运行函数
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# 判断一个数是否能被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 采样辅助函数

# 计算输入张量的对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 生成Gumbel噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 生成Gumbel采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# 生成top-k采样
def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 令牌移位函数
def token_shift_fn(t, ps):
    read_mem, t, write_mem = unpack(t, ps, 'b * d')
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value = 0.)
    t = torch.cat((t, t_shift), dim = -1)
    return torch.cat((read_mem, t, write_mem), dim = -2)

# 分数梯度函数
def frac_gradient(t, frac = 1.):
    if frac == 1.:
        return t

    return t * frac + t.detach() * (1. - frac)

# 旋转嵌入

# 旋转嵌入类
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 32768):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        freqs = torch.einsum('i , j -> i j', positions, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

# 旋转半个周期
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# 规范化

# 均方根规范化类
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# 前馈网络

# GEGLU激活函数
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

# 前馈网络函数
def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        Linear(dim, dim_inner * 2, bias = False),
        GEGLU(),
        RMSNorm(dim_inner),
        nn.Dropout(dropout),
        Linear(dim_inner, dim, bias = False)
    )

# 注意力机制

# 注意力类
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_flash_attn = False,
        use_custom_causal_attn_mask = False
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads

        self.attend = Attend(
            causal = causal and not use_custom_causal_attn_mask,
            dropout = dropout,
            use_flash = use_flash_attn
        )

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_q = Linear(dim, dim_inner)
        self.to_kv = Linear(dim, dim_inner * 2)
        self.to_out = Linear(dim_inner, dim)
    # 定义一个前向传播函数，接受输入 x，旋转嵌入 rotary_emb（可选），掩码 mask，XL 内存 xl_memories（可选）
    def forward(
        self,
        x,
        rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
        mask = None,
        xl_memories = None
    ):
        # 获取头数 h
        h = self.heads

        # 将输入 x 转换为查询 q
        q = self.to_q(x)
        # 将输入 x 转换为键 k 和值 v
        k, v = self.to_kv(x).chunk(2, dim = -1)

        # 对查询 q、键 k、值 v 进行重排列，以适应多头注意力的计算
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 添加一个空键/值对，以防止整个序列被完全掩码，同时使注意力能够关注空值
        nk, nv = map(lambda t: repeat(t, 'h d -> b h 1 d', b = x.shape[0]), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # 如果存在掩码，则在掩码前面填充一个位置
        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)

        # 管理记忆
        next_xl_memories = torch.stack((k, v))

        # 如果存在 XL 记忆，则将 XL 记忆与当前键 k 和值 v 连接起来
        if exists(xl_memories):
            kx, vx = xl_memories
            k = torch.cat((kx, k), dim = -2)
            v = torch.cat((vx, v), dim = -2)

            # 如果存在掩码，则在掩码前面填充 XL 记忆的长度个位置
            if exists(mask):
                mask = F.pad(mask, (xl_memories.shape[-2], 0), value = True)

        # 如果存在旋转嵌入，则将查询 q 和键 k 应用旋转位置嵌入
        if exists(rotary_emb):
            q_rotary_emb, k_rotary_emb = rotary_emb

            q = apply_rotary_pos_emb(q_rotary_emb, q)
            k = apply_rotary_pos_emb(k_rotary_emb, k)

        # 使用注意力机制计算输出
        out = self.attend(q, k, v, mask = mask)

        # 将输出重排列为原始形状
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 将输出传递给输出层，并返回下一个 XL 记忆
        return self.to_out(out), next_xl_memories
# 定义一个名为 RecurrentMemoryTransformer 的类，继承自 nn.Module
class RecurrentMemoryTransformer(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        num_memory_tokens,
        seq_len,
        causal = True,        
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_flash_attn = False,
        ignore_index = -1,
        abs_pos_emb = True,
        rotary_pos_emb = False,
        token_shift = True,
        use_xl_memories = True,
        xl_mem_len = None,
        enhanced_xl_recurrence = False,      # 是否使用增强的 XL 记忆方法，来自 ernie-doc 论文
        emb_gradient_frac = 0.1,             # 来自 cogview 论文的技巧，导致更稳定一些
        memory_not_causal = True,            # 如果闪光注意力在没有显式传递因果掩码的情况下表现更佳，那么有必要将其打开
        add_write_to_next_write_mem = False, # 将上一步的写记忆添加到下一步的写步骤中 - 感谢 @IcarusWizard 指出这个不一致之处
        next_write_mem_stop_grad = True,     # 是否停止前一个读记忆的梯度 -> 下一个写记忆
        always_have_read_memories = True,    # 是否始终具有读记忆，即使在第一步也是如此，以使模型能够导出为 ONNX
        resi_dual_scale = 1.,                # 在 prenorm 分支中发生 fp16 溢出的情况下，将其设置为小于 1 的值
        ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化模型参数
        self.causal = causal
        self.seq_len = seq_len

        self.emb_gradient_frac = emb_gradient_frac

        # 断言保证 resi_dual_scale 在 0 和 1 之间
        assert 0 < resi_dual_scale <= 1., 'resiDual scale must be between 0 and 1'
        self.resi_dual_scale = resi_dual_scale

        assert num_memory_tokens > 0

        # 初始化 token embedding 层
        self.token_emb = nn.Embedding(num_tokens, dim)

        # 初始化位置编码
        assert any([abs_pos_emb, rotary_pos_emb, token_shift])

        self.pos_emb = nn.Embedding(seq_len, dim) if abs_pos_emb else None

        self.rotary_pos_emb = RotaryEmbedding(dim_head) if rotary_pos_emb else None

        self.maybe_token_shift = token_shift_fn if token_shift else identity

        # 初始化与记忆相关的参数
        self.num_memory_tokens = num_memory_tokens

        self.read_memory_emb = nn.Parameter(torch.zeros(num_memory_tokens, dim))
        nn.init.normal_(self.read_memory_emb, std = 0.02)

        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std = 0.02)

        # 初始化 xl memories
        xl_mem_len = default(xl_mem_len, seq_len)
        assert xl_mem_len <= seq_len
        self.xl_mem_len = xl_mem_len

        self.use_xl_memories = use_xl_memories
        self.enhanced_xl_recurrence = enhanced_xl_recurrence

        # 初始化层
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim = dim,
                    dim_head = dim_head,
                    causal = causal,
                    heads = heads,
                    use_flash_attn = use_flash_attn,
                    use_custom_causal_attn_mask = memory_not_causal,
                    dropout = attn_dropout
                ),
                RMSNorm(dim),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
                RMSNorm(dim)
            ]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

        self.ignore_index = ignore_index

        # 是否使用自定义注意力掩码，如果是因果性的且记忆不应该是因果性的
        self.use_custom_causal_attn_mask = causal and memory_not_causal

        # 在论文中，他们实际上还使用前一个写入记忆来生成下一个写入记忆
        self.add_write_to_next_write_mem = add_write_to_next_write_mem
        self.next_write_mem_stop_grad = next_write_mem_stop_grad

        # 允许在第一步时关注原始读取记忆的位置编码
        # 为了使其能够在 onnx 中运行，并且不会有影响
        self.always_have_read_memories = always_have_read_memories

    # 初始化记忆
    def init_memory(self, batch):
        return repeat(self.memory_tokens, 'm d -> b m d', b = batch)

    # 前向传播函数
    def forward(
        self,
        x,
        read_memories = None,
        *,
        mask = None,
        labels = None,
        xl_memories: Optional[List[Tensor]] = None,
        mask_out_read_memories = False   # 在传入读取记忆为 0 时，用于 onnx 模型
# 管理多个段的包装器

class RecurrentMemoryTransformerWrapper(nn.Module):
    def __init__(
        self,
        transformer: RecurrentMemoryTransformer,
        truncate_at_step = None  # 在分离记忆之前截断步骤的数量（截断 bptt）。通过记忆重播检查点，不应该有记忆问题，但如果出现不稳定性，如初始论文中报告的那样
    ):
        super().__init__()
        self.transformer = transformer
        self.seq_len = transformer.seq_len
        self.truncate_at_step = truncate_at_step

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prime,
        *,
        length,
        memories = None,
        xl_memories: Optional[List[Tensor]] = None,
        temperature = 1.,
        filter_thres = 0.9
    ):
        assert self.transformer.causal, 'only autoregressive transformers can generate'

        start_len, seq_len = prime.shape[-1], self.seq_len

        assert length >= start_len

        *past_segments, curr_segment = prime.split(seq_len, dim = -1)

        # catch memories up to the current segment

        for past_segment in past_segments:
            _, memories, xl_memories = self.transformer(past_segment, memories, xl_memories = xl_memories)

        # sample for the remaining length

        for ind in range(length - start_len):
            logits, next_memories, next_xl_memories = self.transformer(curr_segment, memories, xl_memories = xl_memories)

            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            curr_segment = torch.cat((curr_segment, sampled), dim = -1)

            if divisible_by(curr_segment.shape[-1] - 1, seq_len):
                memories = next_memories
                xl_memories = next_xl_memories

                past_segment, curr_segment = curr_segment[..., :seq_len], curr_segment[..., -1:]
                past_segments.append(past_segment)

        # add current segment to all segments

        past_segments.append(curr_segment)

        # reconcat all segments

        output = torch.cat(past_segments, dim = -1)

        output = output[:, start_len:]
        return output

    def forward(
        self,
        x,
        memories = None,
        *,
        mask = None,
        xl_memories: Optional[List[Tensor]] = None,
        return_loss = False,
        labels = None,
        truncate_at_step = None,         # 如果设置，这将覆盖初始化时的 truncate_at_step
        memory_replay_backprop = False,  # 是否让类进行内存高效的反向传播
        mrbp_loss_weight = 1.            # 如果使用内存重播反向传播与梯度累积，通过此因子缩放损失，例如（1. / <num grad accum steps>）
        ):
            # 设置序列长度和截断步数
            seq_len, truncate_at_step = self.seq_len, default(truncate_at_step, self.truncate_at_step)

            labels = None
            # 如果需要返回损失或进行记忆重播反向传播，并且标签不存在，则从输入中获取标签
            if (return_loss or memory_replay_backprop) and not exists(labels):
                x, labels = x[:, :-1], x[:, 1:]

            # 分割输入
            segments = x.split(seq_len, dim = -1)
            total_length = x.shape[-1]
            num_segments = len(segments)
            segment_length_frac = tuple(map(lambda t: t.shape[-1] / total_length, segments))

            # 默认值
            label_segments = mask_segments = (None,)

            # 处理标签
            if exists(labels):
                label_segments = labels.split(seq_len, dim = -1)

            # 处理掩码
            if exists(mask):
                mask_segments = mask.split(seq_len, dim = -1)

            # 保留重播缓冲区
            replay_buffer = [memories]

            # 用于xl记忆的重播缓冲区
            xl_segments = [xl_memories]

            # 根据是否进行记忆重播反向传播决定前向上下文
            forward_context = nullcontext if not memory_replay_backprop else torch.no_grad

            # 前向传播并获取所有输出（可以是损失或逻辑值）
            logits = []
            losses = []

            for step, (segment, mask_segment, label_segment, loss_weight) in enumerate(zip_longest(segments, mask_segments, label_segments, segment_length_frac):

                with forward_context():
                    output, memories, xl_memories = self.transformer(segment, memories, mask = mask_segment, labels = label_segment)

                if exists(truncate_at_step) and divisible_by(step + 1, truncate_at_step):
                    memories = memories.detach()

                replay_buffer.append(memories)

                xl_segments.append(xl_memories)

                if return_loss:
                    losses.append(output * loss_weight)
                else:
                    logits.append(output)

            # 是否进行记忆重播反向传播
            # https://arxiv.org/abs/2010.06891
            # 算法1
            if memory_replay_backprop:
                memories_grad = torch.zeros_like(replay_buffer[-1])

                reversed_inputs = zip_longest(*map(reversed, [
                    range(num_segments),
                    segments,
                    replay_buffer[:-1],
                    xl_segments[:-1],
                    mask_segments,
                    label_segments,
                    segment_length_frac,
                ]))

                total_loss = 0.

                for step, segment, segment_memories, segment_xl_memories, mask_segment, label_segment, loss_weight in reversed_inputs:
                    is_first = step == 0

                    if exists(segment_memories):
                        segment_memories.requires_grad_()

                    loss, next_segment_memories, _ = self.transformer(segment, segment_memories, mask = mask_segment, xl_memories = segment_xl_memories, labels = label_segment)

                    weighted_loss = loss * loss_weight * mrbp_loss_weight

                    weighted_loss.backward(retain_graph = True)

                    next_segment_memories.backward(memories_grad)

                    total_loss += weighted_loss

                    if is_first:
                        continue

                    if exists(truncate_at_step) and divisible_by(step, truncate_at_step):
                        memories_grad.zero_()
                    else:
                        memories_grad.copy_(segment_memories.grad.data)

                return total_loss

            # 如果不需要返回损失，则返回逻辑值
            if not return_loss:
                logits = torch.cat(logits, dim = -2)
                return logits, memories

            # 否则返回损失
            return sum(losses), memories
```