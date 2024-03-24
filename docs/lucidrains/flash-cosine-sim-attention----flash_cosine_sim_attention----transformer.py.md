# `.\lucidrains\flash-cosine-sim-attention\flash_cosine_sim_attention\transformer.py`

```
import torch
from functools import partial
from torch import nn, einsum
import torch.nn.functional as F

try:
    from einops import rearrange
except ImportError:
    print('pip install einops to use transformer')

from flash_cosine_sim_attention.flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

# helper function

# 检查变量是否存在的辅助函数
def exists(val):
    return val is not None

# 使用 Xavier 初始化权重的函数
def init_weight_xavier_normal_(module, beta):
    nn.init.xavier_normal_(module.weight.data, gain = beta)

# 评估装饰器函数，用于在模型评估时切换模型状态
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 非余弦相似度注意力函数
def non_cosine_sim_attn_fn(q, k, v, **kwargs):
    q = q * (q.shape[-1] ** -0.5)
    sim = einsum('b h i d, b h j d -> b h i j', q, k)
    i, j = sim.shape[-2:]
    causal_mask = torch.ones((i, j), dtype = torch.bool, device = q.device).triu(j - i + 1)
    sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
    attn = sim.softmax(dim = -1)
    return einsum('b h i j, b h j d -> b h i d', attn, v)

# top k 过滤函数
def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 注意力和前馈网络

# 前馈网络函数
def FeedForward(dim, mult = 4, pre_norm = False):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim) if pre_norm else nn.Identity(),
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# 注意力模块
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 8,
        l2norm_groups = 1,
        pre_norm = False,
        use_cuda_kernel = False,
        non_cosine_sim_attn = False,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()

        self.scale = scale
        self.heads = heads

        self.l2norm_groups = l2norm_groups

        if non_cosine_sim_attn:
            self.attn_fn = non_cosine_sim_attn_fn
        elif use_cuda_kernel:
            self.attn_fn = partial(flash_cosine_sim_attention, **kwargs)
        else:
            self.attn_fn = plain_cosine_sim_attention

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        h, scale, l2norm_groups = self.heads, self.scale, self.l2norm_groups

        x = self.norm(x)

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        o = self.attn_fn(q, k, v, causal = True, scale = scale, groups = l2norm_groups)

        o = rearrange(o, 'b h n d -> b n (h d)')
        return self.to_out(o)

# 用于测试的变换器模型
class CosineSimCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        depth,
        attn_scale = 8,
        attn_l2norm_groups = 1,
        heads = 8,
        dim_head = 64,
        use_cuda_kernel = False,
        pre_norm = False,
        non_cosine_sim_attn = False,
        **kwargs
    # 初始化模型参数
    def __init__(
        self,
        max_seq_len,
        num_tokens,
        dim,
        depth,
        dim_head,
        heads,
        use_cuda_kernel,
        attn_scale,
        attn_l2norm_groups,
        pre_norm,
        non_cosine_sim_attn,
        **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置最大序列长度
        self.max_seq_len = max_seq_len

        # 创建 token embedding 层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建位置 embedding 层
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 计算残差连接的缩放因子
        self.residual_scale = 1 if pre_norm else ((2 * depth) ** 0.25)

        # 创建多层 Transformer 模型
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 添加注意力机制层
                Attention(dim, dim_head=dim_head, heads=heads, use_cuda_kernel=use_cuda_kernel, scale=attn_scale, groups=attn_l2norm_groups, pre_norm=pre_norm, non_cosine_sim_attn=non_cosine_sim_attn, **kwargs),
                # 添加 LayerNorm 层或者恒等映射层
                nn.LayerNorm(dim) if not pre_norm else nn.Identity(),
                # 添加前馈神经网络层
                FeedForward(dim, pre_norm=pre_norm),
                # 添加 LayerNorm 层或者恒等映射层
                nn.LayerNorm(dim) if not pre_norm else nn.Identity(),
            ]))

        # 创建输出层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim) if pre_norm else nn.Identity(),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # 如果不使用预层归一化，则初始化模型参数
        if not pre_norm:
            self.init_(depth)

    # 初始化模型参数
    def init_(self, depth):
        # 初始化 token embedding 层和位置 embedding 层的权重
        nn.init.normal_(self.token_emb.weight, std=1e-5)
        nn.init.normal_(self.pos_emb.weight, std=1e-5)

        # 计算初始化权重的增益
        init_gain = (8 * depth) ** -0.25

        # 初始化每一层的权重
        for attn, _, ff, _ in self.layers:
            init_weight_xavier_normal_(attn.to_q, 1.)
            init_weight_xavier_normal_(attn.to_k, 1.)
            init_weight_xavier_normal_(attn.to_v, init_gain)
            init_weight_xavier_normal_(attn.to_out, init_gain)
            init_weight_xavier_normal_(ff[1], init_gain)
            init_weight_xavier_normal_(ff[3], init_gain)

        init_weight_xavier_normal_(self.to_logits[-1], 1)

    # 生成序列
    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, temperature=1., filter_thres=0.9, **kwargs):
        # 获取输入序列的形状和设备信息
        b, n, device = *start_tokens.shape, start_tokens.device

        # 初始化输出序列
        out = start_tokens

        # 生成序列
        for _ in range(seq_len):
            logits = self.forward(out[:, -self.max_seq_len:], **kwargs)[:, -1, :]
            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)

        return out[:, n:]

    # 前向传播
    def forward(self, x, return_loss=False):
        # 如果需要计算损失，则获取输入序列和标签序列
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # 对输入序列进行 token embedding 和位置 embedding
        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(x.shape[1], device=x.device))

        # 多层 Transformer 模型的前向传播
        for attn, attn_norm, ff, ff_norm in self.layers:
            x = attn(x) + x * self.residual_scale
            x = attn_norm(x)
            x = ff(x) + x * self.residual_scale
            x = ff_norm(x)

        # 输出层得到 logits
        logits = self.to_logits(x)

        # 如果不需要计算损失，则返回 logits
        if not return_loss:
            return logits

        # 计算交叉熵损失
        loss = F.cross_entropy(rearrange(logits, 'b c n -> b n c'), labels)
        return loss
```