# `.\lucidrains\simple-hierarchical-transformer\simple_hierarchical_transformer\simple_hierarchical_transformer.py`

```py
# 从 math 模块中导入 log2 和 ceil 函数
# 从 functools 模块中导入 partial 函数
# 从 itertools 模块中导入 zip_longest 函数
# 导入 torch 库
import torch
# 从 torch.nn.functional 模块中导入 F
import torch.nn.functional as F
# 从 torch.cuda.amp 模块中导入 autocast 函数
from torch.cuda.amp import autocast
# 从 torch 模块中导入 nn, einsum, Tensor
from torch import nn, einsum, Tensor
# 从 torch.nn 模块中导入 Module, ModuleList
from torch.nn import Module, ModuleList
# 从 einops 模块中导入 rearrange, repeat
from einops import rearrange, repeat
# 从 einops.layers.torch 模块中导入 Rearrange
from einops.layers.torch import Rearrange
# 从 simple_hierarchical_transformer.attention 模块中导入 Attend
from simple_hierarchical_transformer.attention import Attend
# 从 typing 模块中导入 Tuple
from typing import Tuple
# 从 local_attention 模块中导入 LocalMHA

# 定义常量 Linear，使用 nn.Linear 函数，设置 bias 参数为 False
Linear = partial(nn.Linear, bias = False)
# 定义 LocalMHA，使用 partial 函数，设置 LocalMHA 函数的 causal 和 prenorm 参数为 True

# 定义辅助函数 exists，判断值是否存在
def exists(val):
    return val is not None

# 定义辅助函数 is_power_of_two，判断一个数是否为2的幂
def is_power_of_two(n):
    return log2(n).is_integer()

# 定义辅助函数 all_unique，判断列表中的元素是否唯一
def all_unique(arr):
    return len(set(arr)) == len(arr

# 定义辅助函数 apply_fns，对输入的函数列表和张量列表进行函数应用
def apply_fns(fns, tensors):
    return [fn(tensor) for fn, tensor in zip(fns, tensors)]

# 定义辅助函数 cast_tuple，将输入转换为元组
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 定义辅助函数 default，返回第一个非空值
def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

# 定义 eval_decorator 装饰器函数，用于在模型评估时切换为 eval 模式
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 定义张量辅助函数 l2norm，对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 定义余弦相似度损失函数 cosine_sim_loss，计算余弦相似度损失
def cosine_sim_loss(x, y):
    x, y = map(l2norm, (x, y))
    return 1. - einsum('b n d, b n d -> b n', x, y).mean()

# 定义采样辅助函数 log，对张量进行对数运算
def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

# 定义采样辅助函数 gumbel_noise，生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 定义采样辅助函数 gumbel_sample，使用 Gumbel 噪声进行采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# 定义采样辅助函数 top_k，对 logits 进行 top-k 采样
def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    probs.scatter_(1, ind, val)
    return probs

# 旋转位置嵌入类 RotaryEmbedding
class RotaryEmbedding(Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        scale_base = 512,
        use_xpos = True
    ):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    # 获取设备信息
    @property
    def device(self):
        return next(self.buffers()).device

    # 前向传播函数
    @autocast(enabled = False)
    def forward(self, seq_len):
        device = self.device
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

# 旋转半部分函数 rotate_half
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入函数 apply_rotary_pos_emb
def apply_rotary_pos_emb(pos, t, scale = 1.):
    seq_len = t.shape[-2]

    pos = pos[..., -seq_len:, :]
    if not isinstance(scale, (int, float)):
        scale = scale[..., -seq_len:, :]

    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)

# 应用旋转位置嵌入到查询和键函数 apply_rotary_pos_emb_qk
@autocast(enabled = False)
def apply_rotary_pos_emb_qk(rotary_emb, q, k):
    freqs, scale = rotary_emb
    q = apply_rotary_pos_emb(freqs, q, scale)
    k = apply_rotary_pos_emb(freqs, k, scale ** -1)
    return q, k

# 令牌移位函数 token_shift
def token_shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim = -1)
# hierarchy related classes

# 将序列填充到指定倍数
def pad_seq_to_multiple(t, mult):
    # 获取序列长度
    seq_len = t.shape[-2]
    # 计算下一个序列长度的倍数
    next_seq_len_mult = ceil(seq_len / mult) * mult
    # 计算需要填充的长度
    remainder = next_seq_len_mult - seq_len

    # 如果不需要填充，则直接返回原序列和序列长度
    if remainder == 0:
        return t, seq_len

    # 对序列进行填充
    t = F.pad(t, (0, 0, 0, remainder), value = 0.)
    return t, seq_len

# 将序列截断到指定倍数
def curtail_seq_to_multiple(t, mult):
    # 获取序列长度
    seq_len = t.shape[-2]
    # 计算前一个序列长度的倍数
    prev_seq_len_mult = (seq_len // mult) * mult
    # 计算需要截断的长度
    remainder = seq_len - prev_seq_len_mult

    # 如果不需要截断，则直接返回原序列
    if remainder == 0:
        return t

    # 对序列进行截断
    t = t[..., :prev_seq_len_mult, :]
    return t

# 将多个序列按照指定步长合并
def hierarchical_cat(tokens, strides: Tuple[int, ...]):
    # 断言tokens和strides的长度相等
    assert len(tokens) == len(strides)

    # 如果所有步长都为1，则直接拼接所有序列
    if all([s == 1 for s in strides]):
        return torch.cat(tokens, dim = -1)

    # 对每个序列进行重复以匹配步长
    tokens = [repeat(t, 'b n d -> b (n s) d', s = s) for t, s in zip(tokens, strides)]
    # 获取最小序列长度
    min_seq_len = min([t.shape[-2] for t in tokens])
    # 截取所有序列到最小序列长度
    tokens = [t[..., :min_seq_len, :] for t in tokens]
    return torch.cat(tokens, dim = -1)

# 定义CausalConv类
class CausalConv(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride = 1
    ):
        super().__init__()
        # 设置causal_padding为kernel_size - 1
        self.causal_padding = kernel_size - 1
        # 创建Conv1d层
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, stride = stride)

    def forward(self, x):
        # 对输入进行padding
        x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)

# 定义Compress类
class Compress(Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        num_tokens = None,
        stride = 1,
        compress_factor = 1,
        expansion_factor = 4,
        dim_head = 64,
        heads = 8,
        ignore_index = 0,
        should_recon = False
    ):
        super().__init__()
        # 断��compress_factor大于0且为2的幂
        assert compress_factor > 0 and is_power_of_two(compress_factor)

        self.stride = stride
        self.no_compress = compress_factor == 1
        self.compress_factor = compress_factor

        self.should_recon = should_recon

        # 如果不压缩，则使用Linear层或者Identity层
        if self.no_compress:
            self.compress_fn = Linear(dim, dim_out) if dim != dim_out else nn.Identity()
            return

        dim_inner = int(dim * expansion_factor)

        # 使用Sequential定义压缩函数
        self.compress_fn = nn.Sequential(
            Rearrange('b n d -> b d n'),
            CausalConv(dim, dim_inner, compress_factor, stride = stride),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim_out, 1),
            Rearrange('b d n -> b n d')
        )

        # 如果需要重构，则定义Linear层
        if should_recon:
            assert exists(num_tokens)
            self.to_recon = Linear(dim_out, compress_factor * num_tokens)

        self.ignore_index = ignore_index

    # 重构函数
    def recon(self, h, ids):
        assert self.should_recon

        if self.no_compress:
            return torch.zeros((), device = h.device).requires_grad_()

        c = self.compress_factor
        seq_len = ids.shape[-1]

        recon_logits = self.to_recon(h)
        recon_logits = rearrange(recon_logits, 'b n (c d) -> (b c) d n', c = c)

        recon_ids = F.pad(ids, (c - 1, 0), value = self.ignore_index)
        recon_ids = tuple(recon_ids[:, i:(seq_len + i)] for i in range(c))
        recon_ids = torch.stack(recon_ids, dim = 1)
        recon_ids = rearrange(recon_ids, 'b c n -> (b c) n')

        if self.stride > 1:
            recon_ids = recon_ids[..., ::self.stride]

        recon_loss = F.cross_entropy(recon_logits, recon_ids, ignore_index = self.ignore_index)
        return recon_loss

    def forward(self, x):
        return self.compress_fn(x)

# 定义HierarchicalMerge类
class HierarchicalMerge(Module):
    def __init__(
        self,
        dims: Tuple[int, ...],
        dim_out,
        h_strides = 1
    ):
        super().__init__()
        dim = sum(dims)

        strides = cast_tuple(h_strides, len(dims))
        assert len(strides) == len(dims)

        self.strides = strides

        # 使用Sequential定义网络结构
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_out * 2),
            nn.SiLU(),
            nn.Linear(dim_out * 2, dim_out)
        )
    # 定义一个前向传播函数，接收 tokens 作为输入
    def forward(self, tokens):
        # 调用 hierarchical_cat 函数对 tokens 进行处理，得到 x
        x = hierarchical_cat(tokens, self.strides)
        # 将处理后的 x 传入神经网络中进行前向传播，返回结果
        return self.net(x)
# 定义 RMSNorm 类，继承自 Module 类
class RMSNorm(Module):
    # 初始化方法，接受维度参数 dim
    def __init__(self, dim):
        # 调用父类的初始化方法
        super().__init__()
        # 计算缩放因子
        self.scale = dim ** 0.5
        # 初始化可学习参数 gamma
        self.gamma = nn.Parameter(torch.ones(dim))

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 对输入 x 进行归一化处理，乘以缩放因子和 gamma
        return F.normalize(x, dim=-1) * self.scale * self.gamma

# 定义 FeedForward 类，继承自 Module 类
class FeedForward(Module):
    # 初始化方法，接受维度参数 dim 和倍数参数 mult，默认为 4
    def __init__(self, dim, mult=4):
        # 调用父类的初始化方法
        super().__init__()
        # 计算内部维度
        dim_inner = int(dim * mult)

        # 定义神经网络结构
        self.net = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, dim_inner),
            nn.GELU(),
            Linear(dim_inner, dim)
        )

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 将输入 x 传入神经网络
        return self.net(x)

# 定义 Attention 类，继承自 Module 类
class Attention(Module):
    # 初始化方法，接受维度参数 dim，头部维度参数 dim_head，默认为 64，头部数量参数 heads，默认为 8，是否使用 Flash Attention 参数 use_flash_attn，默认为 False
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        use_flash_attn=False
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 计算缩放因子
        self.scale = dim_head ** -0.5
        # 头部数量
        self.heads = heads
        # 内部维度
        dim_inner = dim_head * heads

        # 初始化 RMSNorm 和 RotaryEmbedding
        self.norm = RMSNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

        # 初始化 Attend 层
        self.attend = Attend(causal=True, use_flash_attn=use_flash_attn)

        # 初始化线性层，用于计算 Q、K、V
        self.to_qkv = Linear(dim, dim_inner * 3)
        # 初始化线性层，用于输出
        self.to_out = Linear(dim_inner, dim)

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 获取输入 x 的倒数第二维度大小
        n = x.shape[-2]
        # 对输入 x 进行归一化处理
        x = self.norm(x)

        # 将输入 x 经过线性层得到 Q、K、V，并按头部维度拆分
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # 获取旋转位置编码
        rotary_emb = self.rotary_emb(n)
        # 对 Q、K 应用旋转位置编码
        q, k = apply_rotary_pos_emb_qk(rotary_emb, q, k)

        # 进行注意力计算
        out = self.attend(q, k, v)

        # 重排输出维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 经过输出线性层
        return self.to_out(out)

# 定义 HierarchicalBlock 类，继承自 Module 类
class HierarchicalBlock(Module):
    # 初始化方法，接受维度参数 dim，头部维度参数 dim_head，默认为 64，头部数量参数 heads，默认为 8，窗口大小参数 window_size，默认为 None，压缩因子参数 compress_factor，默认为 1，步长参数 stride，默认为 1，FeedForward 倍数参数 ff_mult，默认为 4
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        window_size=None,
        compress_factor=1,
        stride=1,
        ff_mult=4
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 步长
        self.stride = stride

        # 断言压缩因子为 2 的幂
        assert is_power_of_two(compress_factor)
        self.compress_factor = compress_factor
        self.no_compress = compress_factor == 1

        # 断言窗口大小为非负数
        assert not exists(window_size) or window_size >= 0
        self.has_attn = window_size != 0

        # 初始化注意力层
        self.attn = None

        if self.has_attn:
            attn_klass = Attention
            if exists(window_size):
                attn_klass = partial(LocalMHA, window_size=window_size)

            self.attn = attn_klass(dim=dim, dim_head=dim_head, heads=heads)

        # 初始化 FeedForward 层
        self.ff = FeedForward(dim=dim, mult=ff_mult)

    # 前向传播方法，接受输入 x
    def forward(self, x):
        c = self.compress_factor
        axial_dim = c // self.stride

        # 将输入 x 进行填充，使其长度为压缩因子的整数倍
        x, orig_seq_len = pad_seq_to_multiple(x, axial_dim)

        # 如果不需要压缩，则直接返回
        if not self.no_compress:
            x = rearrange(x, 'b (n c) d -> (b c) n d', c=axial_dim)

        # 如果存在注意力层，则进行注意力计算
        if exists(self.attn):
            x = self.attn(token_shift(x)) + x

        # 经过 FeedForward 层
        x = self.ff(token_shift(x)) + x

        # 如果不需要压缩，则重排维度
        if not self.no_compress:
            x = rearrange(x, '(b c) n d -> b (n c) d', c=axial_dim)

        # 返回结果，截取原始序列长度
        return x[:, :orig_seq_len]

# 定义 HierarchicalTransformer 类
class HierarchicalTransformer(Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        num_tokens,  # 标记数量
        dim,  # 向量维度
        depth,  # 深度
        seq_len = 2048,  # 序列长度，默认为2048
        dim_head = 64,  # 头部维度
        heads = 8,  # 头部数量
        ff_mult = 4,  # FeedForward 层的倍数
        hierarchies = 1,  # 分层数量
        window_sizes = None,  # 窗口大小
        hierarchical_stride = 1,  # 分层步长
        hierarchy_merge_all = False,  # 是否将汇总的分层信息传递回所有分层或只传递给一个进行预测
        predict_hierarchy = None,  # 预测分层
        predict_use_all_hierarchy = False,  # 是否使用所有分层进行预测
        recon_loss_weight = 0.1,  # 重构损失权重
        hierarchical_ar_loss_weight = 0.25,  # 分层自回归损失权重
        ignore_index = 0,  # 忽略的索引
        use_flash_attn = False,  # 是否使用 Flash Attention
    @torch.no_grad()  # 禁用梯度计算
    @eval_decorator  # 评估装饰器
    def generate(
        self,
        prompt,  # 提示
        seq_len,  # 序列长度
        temperature = 1.0,  # 温度
        filter_thres = 0.9,  # 过滤阈值
        **kwargs  # 其他参数
    ):
        b, t, device = *prompt.shape, prompt.device

        out = prompt

        # 生成序列
        for _ in range(seq_len):
            logits = self.forward(out[:, -self.seq_len:], **kwargs)[:, -1]
            filtered_logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature)
            sample = rearrange(sample, 'b -> b 1')
            out = torch.cat((out, sample), dim = -1)

        return out[:, t:]  # 返回生成的序列

    @property
    def device(self):
        return next(self.parameters()).device  # 返回模型参数的设备

    # 前向传播函数
    def forward(
        self,
        ids,  # 标识符
        return_loss = False,  # 是否返回损失
        return_hierarchical_token_embeds = False,  # 是否返回分层标记嵌入
        return_hierarchical_embeds = False,  # 是否返回分层嵌入
        ablate_hierarchical_merge = False  # 是否消融分层合并
        ):
        """
        einops notation:

        b - batch
        n - sequence length
        c - compression factor
        d - dimension
        """

        # 如果是训练阶段，预测序列中的下一个标记

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        # 断言序列长度

        assert ids.shape[-1] <= self.seq_len

        # 获取标记嵌入，并填充到压缩因子的倍数

        x = self.token_emb(ids)

        # 对于每个层次结构，适当地压缩标记嵌入到层次嵌入中

        tokens = []

        for compress in self.compressors:
            tokens.append(compress(x))

        # 后嵌入规范化

        tokens = apply_fns(self.post_token_emb_norms, tokens)

        # 如果想要所有压缩后的标记嵌入
        # 仅用于研究空间

        if return_hierarchical_token_embeds:
            return tokens

        # 层次结构

        for layer, merge in zip_longest(self.layers, self.hierarchical_merges):

            tokens = apply_fns(layer, tokens)

            # 汇总所有层次的信息
            # 然后更新将用于进行最终自回归预测的标记

            if not self.need_hierarchical_merge or ablate_hierarchical_merge:
                continue

            pooled = merge(tokens)

            if self.hierarchy_merge_all:
                tokens = [(t + p[..., ::s, :]) for t, p, s in zip(tokens, pooled.split(self.dims, dim = -1), self.h_strides)]
            else:
                predict_tokens = tokens[self.predict_hierarchy_index]
                predict_tokens = predict_tokens + pooled
                tokens[self.predict_hierarchy_index] = predict_tokens

        # 最终规范化嵌入

        embeds = apply_fns(self.norms, tokens)

        # 如果想要所有规范化的层次嵌入

        if return_hierarchical_embeds:
            return embeds

        # 选择将进行预测的层次嵌入

        if self.predict_use_all_hierarchy:
            predict_embed = hierarchical_cat(embeds, self.h_strides)
        else:
            predict_embed = embeds[self.predict_hierarchy_index]

        # 用于预测下一个标记的对数

        logits = self.to_logits(predict_embed)

        if not return_loss:
            return logits

        # 自回归损失（预测编码）

        logits = rearrange(logits, 'b n c -> b c n')
        ce_loss = F.cross_entropy(logits, labels, ignore_index = self.ignore_index)

        # 层次标记的重建损失

        recon_losses = self.zeros.requires_grad_()

        if self.should_recon:
            for compress, t in zip(self.compressors, embeds):
                recon_loss = compress.recon(t, ids)
                recon_losses = recon_losses + recon_loss

        # 层次自回归损失

        hierarchical_ar_losses = self.zeros.requires_grad_()

        for h_embed, maybe_h_pred_linear in zip(embeds, self.to_hierarchical_preds):
            if not exists(maybe_h_pred_linear):
                continue

            h_pred = maybe_h_pred_linear(h_embed)
            h_ar_loss = cosine_sim_loss(h_pred[:, :-1], h_embed[:, 1:])

            hierarchical_ar_losses = hierarchical_ar_losses + h_ar_loss

        # 总损失

        total_loss = ce_loss + \
                     recon_losses * self.recon_loss_weight + \
                     hierarchical_ar_losses * self.hierarchical_ar_loss_weight

        return total_loss, (ce_loss, recon_losses, hierarchical_ar_losses)
```