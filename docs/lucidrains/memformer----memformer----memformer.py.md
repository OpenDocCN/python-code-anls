# `.\lucidrains\memformer\memformer\memformer.py`

```
# 导入数学库和 PyTorch 库
import math
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 functools 库中导入 partial 函数
from functools import partial
# 从 torch.nn.functional 库中导入 F
import torch.nn.functional as F
# 从 inspect 库中导入 isfunction 函数
from inspect import isfunction
# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 collections 库中导入 namedtuple 类
from collections import namedtuple
# 从 memformer.autoregressive_wrapper 模块中导入 AutoregressiveWrapper 类

# 常量

# 创建一个名为 Results 的命名元组，包含 enc_out、mem 和 dec_out 三个字段
Results = namedtuple('Results', ['enc_out', 'mem', 'dec_out'])
# 创建一个名为 EncOnlyResults 的命名元组，包含 enc_out 和 mem 两个字段
EncOnlyResults = namedtuple('EncOnlyResults', ['enc_out', 'mem'])

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回值或默认值
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 返回张量的最大负值
def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# 关键字参数辅助函数

# 从字典中选择指定键的值并弹出这些键
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key, None), keys))
    return dict(zip(keys, values))

# 根据条件将字典分组
def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

# 判断字符串是否以指定前缀开头
def string_begins_with(prefix, str):
    return str.startswith(prefix)

# 根据前缀将字典分组
def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

# 根据前缀将字典分组并去除前缀
def group_by_key_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# 辅助类

# 带残差连接的模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 带预层归一化的模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 位置嵌入

# 相对位置偏置模块
class RelativePositionBias(nn.Module):
    def __init__(self, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qlen, klen):
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(qlen, dtype = torch.long, device = device)
        k_pos = torch.arange(klen, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> () h i j')

# 主要类

# 前馈神经网络模块
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

# 注意力模块
class Attention(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, dim, heads = 8, causal = False, rel_pos_emb = False):
        # 调用父类的初始化函数
        super().__init__()
        # 确保维度可以被头数整除
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        # 计算每个头的维度
        dim_head = dim // heads
        # 缩放因子
        self.scale = dim_head ** -0.5
        # 头数
        self.heads = heads
        # 是否使用自回归
        self.causal = causal

        # 线性变换，将输入转换为查询向量
        self.to_q = nn.Linear(dim, dim)
        # 线性变换，将输入转换为键值对
        self.to_kv = nn.Linear(dim, dim * 2)
        # 线性变换，将输出转换为最终结果
        self.to_out = nn.Linear(dim, dim)

    # 前向传播函数
    def forward(self, x, context = None, pos_emb = None, mask = None, query_mask = None, kv_mask = None, attend_self = False):
        # 获取输入张量的形状和设备信息
        b, n, _, h, scale, device = *x.shape, self.heads, self.scale, x.device

        # 如果需要自注意力机制
        if attend_self:
            # 将输入和上下文拼接在一起
            kv_input = torch.cat((x, context), dim = 1)
        else:
            # 否则使用默认的上下文
            kv_input = default(context, x)

        # 计算查询向量
        q = self.to_q(x)
        # 计算键值对
        kv = self.to_kv(kv_input).chunk(2, dim = -1)

        # 重排查询、键、值张量的形状
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        # 计算点积注意力
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        # 如果存在位置编码
        if exists(pos_emb):
            # 添加位置编码偏置
            pos_emb_bias = pos_emb(*dots.shape[-2:])
            dots += pos_emb_bias

        # 设置掩码值为最大负值
        mask_value = max_neg_value(dots)

        # 如果是自回归模型
        if self.causal:
            # 创建自回归掩码
            causal_mask = torch.ones((n, n), device = device).triu_(1).bool()
            dots.masked_fill_(causal_mask, mask_value)
            del causal_mask

        # 如果存在查询掩码或键值掩码
        if any(map(exists, (query_mask, kv_mask))):
            # 默认查询掩码为全 1
            query_mask = default(query_mask, lambda: torch.ones((b, n), device = device).bool())

            # 如果存在上下文
            if exists(context):
                # 默认键值掩码为全 1
                kv_mask = default(kv_mask, lambda: torch.ones((b, context.shape[1]), device = device).bool())
            else:
                kv_mask = default(kv_mask, query_mask)

            # 重排查询掩码和键值掩码的形状
            query_mask = rearrange(query_mask, 'b i -> b () i ()')
            kv_mask = rearrange(kv_mask, 'b j -> b () () j')
            seq_mask = query_mask * kv_mask
            dots.masked_fill_(~seq_mask, mask_value)
            del seq_mask

        # 如果存在额外掩码
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b () i j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        # 计算注意力权重
        attn = dots.softmax(dim = -1)
        # 计算输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # 重排输出形状
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class Encoder(nn.Module):
    # 编码器类，包含初始化函数
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        # 初始化相对位置偏置
        self.rel_pos_emb = RelativePositionBias(heads = heads)
        # 初始化层列表
        self.layers = nn.ModuleList([])
        # 循环创建指定数量的层
        for _ in range(depth):
            # 向层列表中添加编码器层
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, rel_pos_emb = True))),
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim)))
            ]))
    # 前向传播函数
    def forward(self, x, context = None, src_mask = None):
        # 遍历编码器层
        for (self_attn, cross_attn, ff) in self.layers:
            # 自注意力机制
            x = self_attn(x, pos_emb = self.rel_pos_emb, query_mask = src_mask)
            # 交叉注意力机制
            x = cross_attn(x, context = context)
            # 前馈神经网络
            x = ff(x)
        return x

class Decoder(nn.Module):
    # 解码器类，包含初始化函数
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        # 初始化相对位置偏置
        self.rel_pos_emb = RelativePositionBias(heads = heads, causal = True)
        # 初始化层列表
        self.layers = nn.ModuleList([])
        # 循环创建指定数量的层
        for _ in range(depth):
            # 向层列表中添加解码器层
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, causal = True, rel_pos_emb = True))),
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim))),
            ]))
    # 前向传播函数
    def forward(self, x, context = None, src_mask = None, tgt_mask = None):
        # 遍历解码器层
        for (self_attn, cross_attn, ff) in self.layers:
            # 自注意力机制
            x = self_attn(x, pos_emb = self.rel_pos_emb, query_mask = src_mask)
            # 交叉注意力机制
            x = cross_attn(x, context = context, query_mask = src_mask, kv_mask = tgt_mask)
            # 前馈神经网络
            x = ff(x)
        return x

class TransformerWrapper(nn.Module):
    # 转换器包装器类，包含初始化函数
    def __init__(self, *, num_tokens, max_seq_len, dim, layer_blocks, heads = 8, return_logits = True):
        super().__init__()
        # 初始化标记嵌入
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.max_seq_len = max_seq_len
        self.layer_blocks = layer_blocks
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens) if return_logits else nn.Identity()

    # 前向传播函数
    def forward(self, x, **kwargs):
        _, n, device = *x.shape, x.device
        # 标记嵌入
        x = self.token_emb(x)
        # 层块
        x = self.layer_blocks(x, **kwargs)
        x = self.norm(x)
        return self.to_logits(x)

class Memformer(nn.Module):
    # 记忆形式类，包含初始化函数
    def __init__(
        self,
        *,
        dim,
        num_memory_slots,
        num_mem_updates = 1,
        encoder_only = False,
        mem_update_attn_heads = 8,
        **kwargs):
        super().__init__()
        # 分组关键字参数
        enc_kwargs, kwargs = group_by_key_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = group_by_key_prefix_and_trim('dec_', kwargs)
        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)

        # 初始化编码器
        self.encoder = TransformerWrapper(
            dim = dim,
            layer_blocks = Encoder(dim = dim, **enc_kwargs),
            return_logits = False,
            **enc_transformer_kwargs
        )

        # 初始化解码器
        self.decoder = TransformerWrapper(
            dim = dim,
            layer_blocks = Decoder(dim = dim, **dec_kwargs),
            return_logits = True,
            **dec_transformer_kwargs
        ) if not encoder_only else None

        if exists(self.decoder):
            self.decoder = AutoregressiveWrapper(self.decoder)

        self.num_mem = num_memory_slots
        self.memory_slots = nn.Parameter(torch.randn(num_memory_slots, dim))

        self.num_mem_updates = num_mem_updates
        self.mem_updater = Attention(dim, heads = mem_update_attn_heads)
        self.gru = nn.GRUCell(dim, dim)
        self.mem_ff = Residual(PreNorm(dim, FeedForward(dim)))
    # 获取初始记忆，将记忆槽复制多份以适应批处理大小
    def get_initial_mem(self, batch_size):
        return repeat(self.memory_slots, 'n d -> b n d', b = batch_size)

    # 前向传播函数，接收源数据、目标数据、记忆、源数据掩码、目标数据掩码等参数
    def forward(self, src, tgt = None, mems = None, src_mask = None, tgt_mask = None):
        # 获取源数据的形状信息
        b, n, num_mem, device = *src.shape, self.num_mem, src.device
        # 如果没有传入记忆，则使用默认的初始记忆
        mems = default(mems, lambda: self.get_initial_mem(b))

        # 编码器处理源数据和记忆，生成编码结果
        enc = self.encoder(src, context = mems, src_mask = src_mask)

        # 如果存在解码器和目标数据，则进行解码操作
        if exists(self.decoder) and exists(tgt):
            dec_out = self.decoder(tgt, context = enc, src_mask = tgt_mask, tgt_mask = src_mask, return_loss = True)
        else:
            # 否则创建一个梯度可求的张量作为占位符
            dec_out = torch.tensor(0., requires_grad = True, device = device)

        # 更新记忆，使用注意力机制
        mem_mask = torch.eye(num_mem, num_mem, device = device).bool()
        mem_mask = repeat(mem_mask, 'i j -> b i j', b = b)
        mem_mask = F.pad(mem_mask, (0, n), value = True)

        # 如果存在源数据掩码，则将其与记忆掩码相结合
        if exists(src_mask):
            src_mask = rearrange(src_mask, 'b j -> b () j')
            mem_enc_mask = F.pad(src_mask, (num_mem, 0), value = True)
            mem_mask &= mem_enc_mask

        # 多次更新记忆
        for _ in range(self.num_mem_updates):
            prev_mems = mems
            updated_mems = self.mem_updater(mems, enc, mask = mem_mask, attend_self = True)

            next_mems = self.gru(
                rearrange(updated_mems, 'b n d -> (b n) d'),
                rearrange(prev_mems, 'b n d -> (b n) d')
            )

            mems = rearrange(next_mems, '(b n) d -> b n d', b = b)
            mems = self.mem_ff(mems)

        # 如果没有解码器，则返回编码结果和记忆
        if not exists(self.decoder):
            return EncOnlyResults(enc, mems)

        # 否则返回编码结果、记忆和解码结果
        return Results(enc, mems, dec_out)
```