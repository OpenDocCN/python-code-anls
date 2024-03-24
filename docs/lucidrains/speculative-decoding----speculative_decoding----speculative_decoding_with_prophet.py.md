# `.\lucidrains\speculative-decoding\speculative_decoding\speculative_decoding_with_prophet.py`

```
import math
import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from beartype import beartype
from collections import namedtuple
from einops import rearrange

# 定义一个命名元组Cache，包含cached_kvs和embeds两个字段
Cache = namedtuple('Cache', ['cached_kvs', 'embeds'])

# 定义一些辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果val存在则返回val，否则返回默认值d
def default(val, d):
    return val if exists(val) else d

# 采样辅助函数

# 计算输入张量的对数，避免出现负无穷
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 生成Gumbel噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 从Gumbel分布中采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# 保留top-k的概率值，其余设置为负无穷
def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

# 旋转嵌入

# 定义旋转嵌入类
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.inv_freq.device).type_as(self.inv_freq)
        freqs = einsum('i, j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

# 将输入张量的一半旋转
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    seq_len = t.shape[-2]
    pos = pos[-seq_len:, :]
    return t * pos.cos() + rotate_half(t) * pos.sin()

# 不同的解码策略

# 基础解码函数，用于生成序列
@torch.no_grad()
def base_decoding(
    net: Module,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None

    for _ in range(sample_num_times):
        logits, cache = net(out, cache = cache, return_cache = True)
        logits = logits[:, -1]

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample[..., None]), dim = -1)

    return out[..., prompt_seq_len:]

# 归一化

# 均方根归一化类
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# 注意力和前馈

# 因果注意力类
class CausalAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        cache = None,
        context_mask = None,
        rotary_emb = None
        ):
        # 获取头数和输入张量的设备信息
        h, device = self.heads, x.device

        # 对输入张量进行归一化处理
        x = self.norm(x)

        # 将输入张量转换为查询、键、值，并重新排列维度
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)

        # 如果存在缓存，则将缓存的键值与当前计算的键值拼接
        if exists(cache):
            ck, cv = cache.unbind(dim = 1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # 将键值对堆叠在一起
        cached_kv = torch.stack((k, v), dim = 1)

        # 如果存在旋转位置编码，则应用旋转位置编码到查询和键
        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        # 计算注意力矩阵
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        i, j = sim.shape[-2:]
        # 创建因果掩码
        causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

        # 使用因果掩码填充注意力矩阵
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 如果存在上下文掩码，则使用上下文掩码填充注意力矩阵
        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, -torch.finfo(sim.dtype).max)

        # 计算注意力权重
        attn = sim.softmax(dim = -1)

        # 计算输出张量
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 重新排列输出张量的维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出张量转换为输出
        out = self.to_out(out)

        # 返回输出张量和缓存的键值对
        return out, cached_kv
# 定义一个前馈神经网络模块，包含 RMSNorm 层、线性层、GELU 激活函数和另一个线性层
def FeedForward(dim, mult = 4):
    # 计算内部维度
    dim_inner = dim * mult
    return nn.Sequential(
        RMSNorm(dim),  # 使用 RMSNorm 对输入进行归一化
        nn.Linear(dim, dim_inner),  # 线性变换，将输入维度转换为内部维度
        nn.GELU(),  # GELU 激活函数
        nn.Linear(dim_inner, dim)  # 线性变换，将内部维度转换为输出维度
    )

# 主要类

class Decoder(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        ignore_index = -1
    ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)  # 创建一个嵌入层，将标记映射到指定维度的向量

        self.layers = ModuleList([])  # 创建一个空的模块列表

        self.rotary_emb = RotaryEmbedding(dim = dim_head)  # 创建一个旋转嵌入层，用于相对位置编码

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),  # 创建一个因果注意力层
                FeedForward(dim = dim, mult = ff_mult)  # 创建一个前馈神经网络模块
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),  # 使用 RMSNorm 对输入进行归一化
            nn.Linear(dim, num_tokens, bias = False)  # 线性变换，将维度转换为标记数量，不使用偏置
        )

        self.ignore_index = ignore_index  # 设置忽略的索引值

    def forward(
        self,
        x,
        start_tokens = None,
        return_loss = False,
        return_cache = False,
        seq_start_pos = None,
        cache = None
    ):
        has_start_tokens = exists(start_tokens)  # 检查是否存在起始标记

        start_token_len = 0
        if exists(start_tokens):
            if start_tokens.ndim == 2:
                start_tokens = rearrange(start_tokens, 'b d -> b 1 d')  # 重新排列起始标记的维度

            start_token_len = start_tokens.shape[-2]  # 获取起始标记的长度

        if return_loss:
            x, labels = x[:, start_token_len:-1], x[:, 1:]  # 如果需要返回损失，则截取输入和标签序列

        x = self.token_emb(x)  # 将输入序列映射为嵌入向量

        if exists(start_tokens):
            x = torch.cat((start_tokens, x), dim = 1)  # 如果存在起始标记，则将其与输入序列连接起来

        # 处理序列起始位置偏移

        self_attn_kv_mask = None  # 初始化自注意力键值掩码为 None
        if exists(seq_start_pos):
            batch, seq_len = x.shape[:2]
            seq_range = torch.arange(seq_len, device = x.device, dtype = torch.long)
            self_attn_kv_mask = seq_range >= seq_start_pos[..., None]  # 生成自注意力键值掩码

        # 相对位置编码

        rotary_emb = self.rotary_emb(x.shape[-2])  # 获取相对位置编码

        # 设置缓存

        new_cached_kvs = []  # 创建一个新的缓存键值对列表

        cache_kvs = cache_embeds = None  # 初始化缓存键值对和嵌入向量为 None

        if exists(cache):
            cache_kvs, cache_embeds = cache  # 如果存在缓存，则获取缓存键值对和嵌入向量

        if exists(cache_kvs):
            iter_cache_kvs = iter(cache_kvs.unbind(dim = 1))  # 迭代缓存键值对
        else:
            iter_cache_kvs = iter([])  # 否则创建一个空迭代器

        # 如果传入了缓存，则只使用最后一个标记

        if exists(cache):
            num_tokens_keep = x.shape[-2] - cache_kvs.shape[-2]  # 计算保留的标记数量
            x = x[:, -num_tokens_keep:]  # 截取保留的标记

        # 主要的变换器体

        for ind, (attn, ff) in enumerate(self.layers):
            layer = ind + 1  # 获取当前层索引

            residual = x  # 保存残差连接
            attn_out, cached_kv = attn(x, rotary_emb = rotary_emb, cache = next(iter_cache_kvs, None))  # 执行注意力计算
            x = residual + attn_out  # 添加残差连接

            new_cached_kvs.append(cached_kv)  # 将缓存键值对添加到列表中

        new_cached_kvs = torch.stack(new_cached_kvs, dim = 1)  # 将新的缓存键值对堆叠在一起

        logits = self.to_logits(x)  # 获取输出 logits

        if not return_loss:
            if not return_cache:
                return logits  # 如果不需要返回损失和缓存，则直接返回 logits

            return logits, Cache(new_cached_kvs, x)  # 否则返回 logits 和缓存

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),  # 重新排列 logits 的维度
            labels,  # 标签
            ignore_index = self.ignore_index  # 忽略的索引值
        )

        return loss, Cache(new_cached_kvs, x)  # 返回损失和缓存

class ModelWithProphetWrapper(Module):
    def __init__(
        self,
        model: Decoder,
        prophet: Decoder,
        prophet_train_length = 8,  # 先知训练长度，应大于主模型解码伽马，因为主模型缓存嵌入是滞后一步的
        detach_model_embed_for_prophet = False,
        num_leading_start_tokens = 1
    # 初始化函数，继承父类的初始化方法
    def __init__(
        super().__init__()
        # 初始化模型和prophet
        self.model = model
        self.prophet = prophet

        # 判断模型和prophet的维度是否相同
        model_prophet_same_dim = model.dim == prophet.dim
        # 如果维度相同，则使用nn.Identity()，否则使用nn.Linear()进行维度转换
        self.to_prophet_start_token = nn.Identity() if model_prophet_same_dim else nn.Linear(model.dim, prophet.dim, bias = False)

        # 确保num_leading_start_tokens大于等于1
        assert num_leading_start_tokens >= 1
        self.num_leading_start_tokens = num_leading_start_tokens

        # 设置prophet的训练长度和是否在模型嵌入中分离prophet
        self.prophet_train_length = prophet_train_length
        self.detach_model_embed_for_prophet = detach_model_embed_for_prophet

    # 前向传播函数
    def forward(self, x):
        # 获取num_start_tokens、batch、seq_len、device
        num_start_tokens = self.num_leading_start_tokens
        batch, seq_len, device = *x.shape, x.device
        prophet_seq_len = self.prophet_train_length
        # 确保序列长度大于等于prophet训练长度
        assert seq_len >= prophet_seq_len

        total_loss = 0.

        # 调用模型的前向传播函数，返回主要损失和缓存的键值对以及嵌入
        main_loss, (cached_kvs, embeds) = self.model(x, return_loss = True)

        # 累加主要损失
        total_loss = total_loss + main_loss

        # 如果需要分离模型嵌入用于prophet
        if self.detach_model_embed_for_prophet:
            embeds = embeds.detach()

        # 将嵌入转换为prophet的起始标记
        prophet_start_tokens = self.to_prophet_start_token(embeds)

        # 创建batch索引和prophet序列长度索引
        batch_arange = torch.arange(batch, device = device, dtype = torch.long)
        prophet_seq_arange = torch.arange(prophet_seq_len, device = device, dtype = torch.long)

        # 计算用于prophet训练的序列数量
        num_seq_train_prophet = seq_len - prophet_seq_len - (num_start_tokens - 1)

        # 创建偏移量
        offsets = torch.arange(num_seq_train_prophet, device = device, dtype = torch.long)

        # 获取prophet的输入序列
        prophet_input = x[
            batch_arange[:, None, None],
            offsets[..., None] + prophet_seq_arange
        ]

        # 重新排列prophet的输入序列
        prophet_input = rearrange(prophet_input, '... n -> (...) n')

        # 创建起始标记索引
        start_tokens_arange = torch.arange(num_start_tokens, device = device, dtype = torch.long)

        # 获取prophet的起始标记
        prophet_start_tokens = prophet_start_tokens[
            batch_arange[:, None, None],
            offsets[..., None] + start_tokens_arange
        ]

        # 重新排列prophet的起始标记
        prophet_start_tokens = rearrange(prophet_start_tokens[:, :num_seq_train_prophet], 'b n l d -> (b n) l d')

        # 调用prophet的前向传播函数，返回prophet损失
        prophet_loss, _ = self.prophet(prophet_input, start_tokens = prophet_start_tokens, return_loss = True)

        # 累加prophet损失
        total_loss = total_loss + prophet_loss

        # 返回总损失和主要损失、prophet损失
        return total_loss, (main_loss, prophet_loss)
# 安全除法函数，避免分母为零的情况
def safe_div(num, den, eps = 1e-10):
    return num / max(den, eps)

# 在布尔张量中查找第一个为True的索引
def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

# 使用Prophet模型进行推测解码
@torch.no_grad()
def speculative_decoding_with_prophet_model(
    net: ModelWithProphetWrapper,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature = 1.,
    filter_thres = 0.9,
    lenience = 1.,
    pad_id = 0
):
    """
    eq. algorithm 1 in paper https://arxiv.org/abs/2211.17192
    """

    # 提取模型、Prophet模型和模型到Prophet模型的转换（如果它们的模型维度不同）

    model = net.model
    to_prophet_start_token = net.to_prophet_start_token
    prophet = net.prophet
    num_start_tokens = net.num_leading_start_tokens

    batch, prompt_seq_len, out, device = *prompt.shape, prompt.clone(), prompt.device

    if (seq_len - prompt_seq_len) <= 0:
        return prompt, None

    cache = None
    small_cache = None

    num_steps = 0
    total_accepted = 0

    batch_range = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    seq_lens = torch.full((batch,), prompt_seq_len, device = device, dtype = torch.long)

    # 从主模型中随机抽样第一个标记

    for _ in range(max(1, num_start_tokens - prompt_seq_len)):
        logits, cache = model(out, cache = cache, return_cache = True)
        logits = logits[:, -1:]
        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)
        out = torch.cat((out, sample), dim = -1)
        seq_lens += 1

    # 现在我们有第一个缓存的嵌入，用作推测抽样的Prophet网络的起始标记

    _, embeds = cache
    next_prophet_start_tokens = to_prophet_start_token(embeds[:, -num_start_tokens:])
    # 当序列长度小于给定的序列长度时，执行循环
    while (seq_lens < seq_len).any():

        # 使用较小的网络进行预测

        # 存储所有较小网络的logits和采样输出
        all_small_logits = []
        q_sampled_out = []

        small_cache = None
        num_tokens = 2  # 主模型的嵌入比主序列滞后1步

        # 运行gamma次循环
        for _ in range(gamma):
            # 使用prophet函数进行预测
            small_logits, small_cache = prophet(
                out[..., -num_tokens:],
                start_tokens = next_prophet_start_tokens,
                cache = small_cache,
                return_cache = True
            )

            small_logits = small_logits[:, -1:]

            # 对logits进行top-k筛选
            small_logits = top_k(small_logits, thres = filter_thres)
            all_small_logits.append(small_logits)

            # 使用gumbel采样得到样本
            sample = gumbel_sample(small_logits, temperature = temperature, dim = -1)
            out = torch.cat((out, sample), dim = -1)

            seq_lens += 1
            num_tokens += 1

            q_sampled_out.append(rearrange(sample, '... -> ... 1'))

        q_sampled_out = torch.cat(q_sampled_out, dim = -2)
        small_logits = torch.cat(all_small_logits, dim = -2)

        # 使用较大的网络进行验证

        logits, cache = model(
            out,
            cache = cache,
            return_cache = True,
            seq_start_pos = out.shape[-1] - seq_lens
        )

        logits = logits[..., -(gamma + 1):, :]
        logits = top_k(logits, thres = filter_thres)

        # 计算较大网络和较小网络的概率（算法1中的p(x)和q(x)）

        prob = safe_div(logits, temperature).softmax(dim = -1)
        small_prob = safe_div(small_logits, temperature).softmax(dim = -1)

        p, prob_next = prob[:, :-1], prob[:, -1]

        p = p.gather(-1, q_sampled_out)
        q = small_prob.gather(-1, q_sampled_out) * lenience

        p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]

        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)

        accepted = find_first_true_index(r > (p / q))

        total_accepted += accepted.float().mean()
        num_steps += 1

        num_rejected = gamma - accepted
        has_rejected = num_rejected > 0

        accepted = rearrange(accepted, 'b -> b 1')
        accepted.clamp_(max = gamma - 1)

        adjusted_prob = F.relu(prob[batch_range, accepted] - small_prob[batch_range, accepted])
        adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
        adjusted_prob = rearrange(adjusted_prob, 'b 1 d -> b d')

        prob_next = torch.where(
            rearrange(has_rejected, '... -> ... 1'),
            adjusted_prob,
            prob_next
        )

        # 进行一系列切片操作，将所有内容对齐到右侧，包括kv缓存

        max_num_rejected = num_rejected.amax()
        seq_arange = torch.arange(out.shape[-1], device = device, dtype = torch.long)
        seq_offset_indices = seq_arange + (max_num_rejected - num_rejected)[..., None]

        seq_lens -= num_rejected
        max_seq_len = seq_lens.amax()

        if batch > 1:
            out = F.pad(out, (0, max_num_rejected), value = pad_id)
            out = out[batch_range, seq_offset_indices]

            cache = tuple(F.pad(t, (0, 0, 0, max_num_rejected), value = pad_id) for t in cache)
            cache = tuple(rearrange(t, 'b ... n d -> b n ... d') for t in cache)
            cache = tuple(t[batch_range, seq_offset_indices] for t in cache)
            cache = tuple(rearrange(t, 'b n ... d -> b ... n d') for t in cache)

            if out.shape[-1] > max_seq_len:
                left_index = out.shape[-1] - max_seq_len
                out = out[:, left_index:]
                cache = tuple(t[..., left_index:, :] for t in cache)

        # 采样额外的token，这是论文中的一个技巧，用于更好地限制最坏情况

        next_token = torch.multinomial(prob_next, 1)

        out = torch.cat((out, next_token), dim = -1)
        seq_lens += 1

        _, embeds = cache
        next_prophet_start_tokens = to_prophet_start_token(embeds[:, -num_start_tokens:])
    # 将输出向左对齐

    # 计算需要左侧填充的数量
    num_pad_left = out.shape[-1] - seq_lens
    # 计算最大的左侧填充数量
    max_pad_left = num_pad_left.amax()
    # 在输出张量的最后一个维度上进行填充，左侧填充0，右侧填充最大填充数量，填充值为pad_id
    out = F.pad(out, (0, max_pad_left), value=pad_id)

    # 创建一个序列长度范围的张量
    seq_len_range = torch.arange(seq_len, device=device, dtype=torch.long)
    # 从out张量中选择出需要的部分，根据batch_range和seq_len_range进行索引
    out = out[batch_range, seq_len_range + num_pad_left[..., None]]

    # 返回去除prompt_seq_len长度后的out张量和total_accepted除以num_steps的结果
    return out[..., prompt_seq_len:], total_accepted / num_steps
```