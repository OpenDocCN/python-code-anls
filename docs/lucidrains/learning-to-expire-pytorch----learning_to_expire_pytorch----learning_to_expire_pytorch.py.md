# `.\lucidrains\learning-to-expire-pytorch\learning_to_expire_pytorch\learning_to_expire_pytorch.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch.nn.functional 中导入 F 模块
import torch.nn.functional as F
# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 collections 模块中导入 namedtuple 类
from collections import namedtuple

# 定义一个命名元组 Memory，包含 mems 和 elapsed_times 两个字段
Memory = namedtuple('Memory', ['mems', 'elapsed_times'])

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 安全地拼接张量
def safe_cat(tensors, dim = -1):
    tensors = list(filter(exists, tensors))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim = dim)

# 安全地对张量进行加法操作
def safe_add(tensor, n):
    if not exists(tensor):
        return None
    return tensor + n

# 位置嵌入

# 相对位移函数
def rel_shift(t):
    b, h, i, j, device, dtype = *t.shape, t.device, t.dtype
    zero_pad = torch.zeros((b, h, i, 1), device = device, dtype = dtype)
    concatted = torch.cat([zero_pad, t], dim = -1)
    shifted = concatted.view(b, h, j + 1, i)[:, :, 1:]
    return shifted.view_as(t)

# 正弦嵌入类
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n - 1, -1, -1, device = device).type_as(self.inv_freq)
        sinusoid_inp = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim = -1)
        return emb

# 过期时间跨度逻辑

# 过期时间跨度类
class ExpireSpan(nn.Module):
    def __init__(self, dim, max_mem_len, ramp_length):
        super().__init__()
        self.max_mem_len = max_mem_len
        self.ramp_length = ramp_length
        self.to_expiration = nn.Linear(dim, 1)
        nn.init.constant_(self.to_expiration.bias.data, val = -self.max_mem_len)

    def forward(self, mem, time, seq_len):
        exps = self.to_expiration(mem).squeeze(-1).sigmoid() * self.max_mem_len
        exps = rearrange(exps, 'b j -> b () () j')
        t = rearrange(time, 'b j -> b () () j')
        r = F.pad(exps - t, (0, seq_len), value = 1.)
        mask = torch.clamp((r / self.ramp_length) + 1, min = 0., max = 1.)
        return exps, mask

# 类

# 预层归一化类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# 因果注意力类
class CausalAttention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_pos = nn.Linear(dim, dim_head)
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)
    # 定义一个前向传播函数，接受输入 x，位置编码 pos_emb，记忆 mem，默认为 None，过期掩码 expire_mask，默认为 None
    def forward(self, x, pos_emb, mem = None, expire_mask = None):
        # 获取输入 x 的维度信息：n 为序列长度，h 为头数，scale 为缩放因子，device 为设备信息
        n, h, scale, device = x.shape[1], self.heads, self.scale, x.device

        # 将输入 x 转换为查询向量 q
        q = self.to_q(x)

        # 如果存在记忆 mem，则获取其长度，否则记忆长度为 0
        mem_len = mem.shape[1] if exists(mem) else 0
        # 将记忆 mem 和输入 x 拼接在一起，形成上下文 context
        context = safe_cat((mem, x), dim = 1)

        # 将上下文 context 转换为键值对 kv，并按键值对拆分为 k 和 v
        kv = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))

        # 计算点积注意力得分 dots
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        # 计算相对位置贡献
        pos = self.to_pos(pos_emb)
        pos_dots = einsum('b h i d, j d -> b h i j', q, pos) * scale
        pos_dots = rel_shift(pos_dots)
        pos_dots = F.pad(pos_dots, (mem_len, 0), value = 0)
        dots += pos_dots

        # 生成因果掩码
        mask = torch.ones(dots.shape[-2:], device = device).triu_(mem_len + 1).bool()
        mask = rearrange(mask, 'i j -> () () i j')
        dots.masked_fill_(mask, float('-inf'))
        del mask

        # 计算注意力权重
        attn = dots.softmax(dim = -1)

        # 如果存在过期掩码，则将注意力权重乘以过期掩码
        if exists(expire_mask):
            attn  = attn * expire_mask

        # 计算输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# 定义一个名为 ExpireSpanTransformerXL 的类，继承自 nn.Module
class ExpireSpanTransformerXL(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        num_tokens,  # 标记的数量
        dim,  # 向量的维度
        depth,  # 模型的深度
        seq_len,  # 序列的长度
        heads = 8,  # 多头注意力机制的头数，默认为 8
        num_memory_blocks = 10,  # 记忆块的数量，默认为 10
        expire_loss_coef = 1e-6,  # 过期损失系数，默认为 1e-6
        ramp_length = 128):  # 渐变长度，默认为 128
        super().__init__()  # 调用父类的初始化函数
        # 创建一个标记嵌入层，将标记映射到指定维度的向量
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建一个正弦嵌入层，用于添加正弦位置编码

        self.sinusoidal_emb = SinusoidalEmbedding(dim)

        self.dim = dim  # 将维度赋值给类属性
        self.depth = depth  # 将深度赋值给类属性
        self.seq_len = seq_len  # 将序列长度赋值给类属性
        self.max_mem_len = num_memory_blocks * seq_len  # 计算最大记忆长度

        self.expire_loss_coef = expire_loss_coef  # 将过期损失系数赋值给类属性

        self.layers = nn.ModuleList([])  # 创建一个空的模块列表
        # 循环创建深度次数的层，并添加到模块列表中
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ExpireSpan(dim, self.max_mem_len, ramp_length),  # 添加过期跨度模块
                PreNorm(dim, CausalAttention(dim, heads = heads)),  # 添加预归一化的因果注意力模块
                PreNorm(dim, FeedForward(dim)),  # 添加预归一化的前馈神经网络模块
            ]))

        self.to_logits = nn.Linear(dim, num_tokens)  # 创建一个线性层，将输出维度映射到标记数量
    # 定义前向传播函数，接受输入 x 和记忆 memory，默认为 None
    def forward(self, x, memory = None):
        # 获取输入 x 的形状信息，包括 batch 大小 b，序列长度 n，维度 d，设备信息 device
        b, n, d, device = *x.shape, self.dim, x.device
        # 对输入 x 进行 token embedding
        x = self.token_emb(x)
        # 生成位置编码
        pos_emb = self.sinusoidal_emb(x)

        hidden_states = []
        expire_masks_layers = []
        # 如果存在记忆，则获取记忆中的 mems 和 elapsed_times，否则初始化为 None
        mems_layers = memory.mems if exists(memory) else ((None,) * self.depth)
        times_layers = memory.elapsed_times if exists(memory) else ((None,) * self.depth)
        # 初始化辅助损失为 0
        aux_loss = torch.tensor(0., requires_grad = True)

        # 遍历每个层的记忆和时间信息，以及每个层的注意力和前馈网络
        for (mem, time, (expire_span, attn, ff)) in zip(mems_layers, times_layers, self.layers):
            hidden_states.append(x)

            # 计算过期时间和过期掩码
            exps, expire_mask = expire_span(mem, time, seq_len = n) if exists(mem) else (None, None)
            expire_masks_layers.append(expire_mask)

            # 训练模式下，根据时间信息生成遗忘掩码
            if self.training and exists(time):
                forget_time_thres = torch.randint(0, self.max_mem_len, (b, 1), device = device)
                forget_dropout_mask = (time < forget_time_thres).float()
                forget_dropout_mask = rearrange(forget_dropout_mask, 'b n -> b () () n')
                forget_dropout_mask = F.pad(forget_dropout_mask, (0, n), value = 1.)
                expire_mask *= forget_dropout_mask

            # 执行注意力和前馈网络操作
            x = attn(x, pos_emb = pos_emb, mem = mem, expire_mask = expire_mask) + x
            x = ff(x) + x

            if exists(exps):
                # 计算辅助损失，仅对产生软掩码值的过期进行 L1 辅助损失
                expiring_exps_mask = (expire_mask > 0) & (expire_mask < 1.)
                expiring_exps = exps.masked_select(expiring_exps_mask[..., :-n])
                aux_loss = aux_loss + (expiring_exps / self.seq_len).sum() * self.expire_loss_coef

        # 生成最终的 logits
        logits = self.to_logits(x)

        # 如果序列长度等于 n
        if self.seq_len == n:
            if exists(expire_mask):
                mems_layers_new = []
                times_layers_new = []

                # 遍���每个层的记忆、时间和过期掩码信息
                for mems, times, expire_mask in zip(mems_layers, times_layers, expire_masks_layers):
                    expire_mask = rearrange(expire_mask, 'b () () i -> b i')
                    # 丢弃已过期的记忆
                    expired_exps_mask = (expire_mask <= 0)[..., :-n]
                    num_to_expire = min(expired_exps_mask.sum(dim = -1)
                    _, indices = expired_exps_mask.float().topk(k = num_to_expire, dim = -1)
                    even_expired_exps_mask = torch.zeros_like(expired_exps_mask, device = device).scatter(-1, indices, 1.).bool()

                    mems = mems.masked_select(~even_expired_exps_mask.unsqueeze(-1))
                    mems = mems.reshape(b, -1, d)
                    mems_layers_new.append(mems)

                    times = times.masked_select(~even_expired_exps_mask)
                    times = times.reshape(b, -1)
                    times_layers_new.append(times)

                mems_layers = mems_layers_new
                times_layers = times_layers_new

            # 更新记忆和时间信息
            new_memories = map(lambda t: safe_cat(t, dim = 1), list(zip(mems_layers, hidden_states)))
            new_memories = map(lambda t: t[:, -self.max_mem_len:].detach(), new_memories)

            new_times = torch.arange(n - 1, -1, -1, device = device)
            new_times = repeat(new_times, 'n -> b n', b = b)
            new_elapsed_times = map(lambda t: safe_cat((safe_add(t, n), new_times), dim = 1), times_layers)
            new_elapsed_times = map(lambda t: t[-self.max_mem_len:], new_elapsed_times)

            memory = Memory(list(new_memories), list(new_elapsed_times))

        # 返回 logits、memory 和辅助损失
        return logits, memory, aux_loss
```