# `.\lucidrains\PaLM-rlhf-pytorch\palm_rlhf_pytorch\palm.py`

```py
# 导入数学库
import math
# 导入拷贝库
import copy
# 导入路径库
from pathlib import Path
# 导入命名元组库
from collections import namedtuple
# 导入装饰器库
from functools import wraps
# 导入zip_longest函数
from itertools import zip_longest

# 导入进度条库
from tqdm import tqdm
# 导入beartype库
from beartype import beartype
# 导入beartype中的Tuple和Optional
from beartype.typing import Tuple, Optional

# 导入torch库
import torch
# 从torch中导入einsum和nn
from torch import einsum, nn
# 从torch.nn中导入functional模块
import torch.nn.functional as F

# 导入einops库
from einops import rearrange, repeat, reduce, pack, unpack
# 从einops.layers.torch中导入Rearrange和Reduce
from einops.layers.torch import Rearrange, Reduce

# 从palm_rlhf_pytorch.attention中导入Attention
from palm_rlhf_pytorch.attention import Attention
# 从palm_rlhf_pytorch.utils中导入top_p, top_k, masked_mean, gumbel_sample, eval_decorator
from palm_rlhf_pytorch.utils import top_p, top_k, masked_mean, gumbel_sample, eval_decorator
# 从palm_rlhf_pytorch.lora中导入LoRA

# 函数和装饰器

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回输入值
def identity(t, *args, **kwargs):
    return t

# 对输入张量进行L2范数归一化
def l2norm(t):
    return F.normalize(t, dim=-1)

# 标准化
# 他们使用没有偏置的layernorm，这是PyTorch不提供的功能

# 标准化层
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 残差连接

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        y = self.fn(x, **kwargs)

        if not any([t.requires_grad for t in (x, y)]):
            return x.add_(y)

        return y + x

# 旋转位置嵌入带xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale_base=512, use_xpos=True):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale

# 旋转半个张量
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t, scale=1.):
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)

# 经典的Noam Shazeer论文，但这里他们使用SwiGLU而不是更流行的GEGLU来门控前馈
# https://arxiv.org/abs/2002.05202

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# 并行注意力和前馈与残差
# 王等人和GPT-J的EleutherAI发现

class ParallelTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        causal=True,
        heads=8,
        qk_rmsnorm=False,
        qk_scale=8,
        ff_mult=4,
        attn_dropout=0.,
        ff_dropout=0.,
        use_xpos=True,
        xpos_scale_base=512,
        flash_attn=False,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化 LayerNorm 层
        self.norm = LayerNorm(dim)

        # 计算注意力内部维度
        attn_inner_dim = dim_head * heads
        # 计算前馈内部维度
        ff_inner_dim = dim * ff_mult
        # 定义融合维度
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        # 设置是否进行 qk rmsnorm
        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            # 初始化 q 的缩放参数
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            # 初始化 k 的缩放参数
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 初始化注意力模块
        self.attend = Attention(
            causal = causal,
            dropout = attn_dropout,
            use_flash_attn = flash_attn
        )

        # 设置头数
        self.heads = heads
        # 设置缩放因子
        self.scale = (dim_head ** -0.5) if not qk_rmsnorm else qk_scale
        # 设置是否是因果关系
        self.causal = causal

        # 初始化旋转嵌入
        self.rotary_emb = RotaryEmbedding(dim_head, scale_base = xpos_scale_base, use_xpos = use_xpos and causal)

        # 初始化融合的注意力和前馈投影
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)

        # 设置是否使用 Flash Attention
        self.flash_attn = flash_attn
        # 初始化注意力输出层
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        # 初始化注意力的 Dropout 层
        self.attn_dropout = nn.Dropout(attn_dropout)
        # 设置 Flash Attention 的 Dropout
        self.flash_attn_dropout = attn_dropout

        # 并行前馈尾部

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # 用于缓存因果掩码和旋转嵌入

        self.register_buffer("pos_emb", None, persistent=False)
        self.register_buffer("pos_emb_scale", None, persistent=False)

    def get_rotary_embedding(self, n, device):
        if exists(self.pos_emb) and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n], self.pos_emb_scale[:n]

        pos_emb, scale = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        self.register_buffer("pos_emb_scale", scale, persistent=False)
        return pos_emb, scale

    def forward(
        self,
        x,
        mask = None,
        finetune_modules = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # 预 Layernorm

        x = self.norm(x)

        # 注意力查询、键、值和前馈内部

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # 调整 LORAS

        lora_q = lora_k = lora_v = lora_o = None

        if exists(finetune_modules):
            lora_q, lora_k, lora_v, lora_o = finetune_modules
            q = q + lora_q(x)
            k = k + lora_k(x)
            v = v + lora_v(x)

        # 分割头部
        # 他们使用多查询单键值注意力，另一篇 Noam Shazeer 的论文
        # 他们发现在一定规模之后没有性能损失，并且解码更有效
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # qk rmsnorm

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        # 使用 xpos 衰减的旋转嵌入以获得更好的长度外推

        positions, scale = self.get_rotary_embedding(n, device)

        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale ** -1)

        # 注意力函数，常规或 Flash

        out = self.attend(q, k, v, mask = mask)

        # 合并头部

        out = rearrange(out, "b h n d -> b n (h d)")

        attn_out = self.attn_out(out)

        ff_out = self.ff_out(ff)

        if exists(lora_o):
            attn_out = attn_out + lora_o(out)

        return attn_out + ff_out
# 定义一个名为 PaLM 的类，继承自 nn.Module 类，用于实现一个基于 Transformer 的模型
@beartype
class PaLM(nn.Module):
    # 初始化函数，接收多个参数用于配置模型的各种属性
    def __init__(
        self,
        *,
        dim,  # 模型的维度
        num_tokens,  # token 的数量
        depth,  # Transformer 的深度
        causal = True,  # 是否使用 causal attention
        dim_head = 64,  # 每个头的维度
        heads = 8,  # 头的数量
        ff_mult = 4,  # FeedForward 层的倍数
        attn_dropout = 0.,  # 注意力层的 dropout 概率
        ff_dropout = 0.,  # FeedForward 层的 dropout 概率
        qk_rmsnorm = False,  # 是否对 QK 矩阵进行 RMS 归一化
        lora_r = 8,  # LoRA 模块的参数 r
        rotary_xpos_scale_base = 512,  # 旋转位置编码的基数
        flash_attn = False,  # 是否使用 Flash Attention
        finetune_scopes = tuple(),  # 微调的范围
        cross_entropy_ignore_index = 0  # 交叉熵损失的忽略索引
    ):
        super().__init__()
        # 初始化模型的各种属性
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.causal = causal
        self.num_tokens = num_tokens

        # 创建 token 的嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])

        # 根据深度循环创建多个 Transformer Block
        for _ in range(depth):
            block = Residual(ParallelTransformerBlock(
                dim = dim,
                causal = causal,
                dim_head = dim_head,
                heads = heads,
                qk_rmsnorm = qk_rmsnorm,
                ff_mult = ff_mult,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                xpos_scale_base = rotary_xpos_scale_base,
                flash_attn = flash_attn
            ))

            self.layers.append(block)

        # 创建 LayerNorm 层
        self.norm = LayerNorm(dim)
        # 创建输出层，用于将模型输出转换为 token 的概率分布
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        # 将输出层的权重与 token 嵌入层的权重共享
        self.to_logits.weight = self.token_emb.weight

        # 对 token 嵌入层的权重进行正态分布初始化
        nn.init.normal_(self.token_emb.weight, std=0.02)

        # 微调相关

        self.lora_r = lora_r
        self.finetune_modules = nn.ModuleDict({})

        # 根据微调范围添加微调参数
        for scope in finetune_scopes:
            self.add_finetune_params(scope)

        # 损失相关

        self.cross_entropy_ignore_index = cross_entropy_ignore_index

    # 定义 device 属性，用于获取模型参数所在的设备
    @property
    def device(self):
        return next(self.parameters()).device

    # 加载模型参数
    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    # 设置模型中的 Dropout 层的概率
    def set_dropout(self, dropout):
        for module in self.layers.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        return self

    # 添加微调参数
    def add_finetune_params(self, scope, lora_r = None):
        assert scope not in self.finetune_modules, f'finetune scope {scope} already found'
        dim, dim_head, heads, r, device = self.dim, self.dim_head, self.heads, default(lora_r, self.lora_r), self.device

        q_inner_dim = heads * dim_head
        kv_inner_dim = dim_head

        lora_modules = nn.ModuleList([])

        for _ in range(len(self.layers)):
            lora_modules.append(nn.ModuleList([
                LoRA(dim, q_inner_dim, r = r),   # queries
                LoRA(dim, kv_inner_dim, r = r),  # keys
                LoRA(dim, kv_inner_dim, r = r),  # values
                LoRA(q_inner_dim, dim, r = r)    # wo
            ]))

        self.finetune_modules[scope] = lora_modules.to(device)

    # 移除微调参数
    def remove_finetune_params(self, scope):
        assert scope in self.finetune_modules, f'finetune scope {scope} not found'
        return self.finetune_modules.pop(scope)

    # 禁用梯度计算
    @torch.no_grad()
    # 合并微调的 actor LORA 参数，用于多轮不同奖励模型的微调
    def merge_finetune_params(self, scope):
        """ in the case one wants to merge the fine-tuned actor LORA parameters and do multiple rounds of fine tuning off different reward models """

        # 确保指定的微调范围存在
        assert scope in self.finetune_modules, f'finetune scope {scope} not found'

        # 弹出指定范围的 LORA 模块
        lora_modules = self.finetune_modules.pop(scope)

        # 遍历每个层和对应的 LORA 模块
        for layer, (lora_q, lora_k, lora_v, lora_o) in zip(self.layers, lora_modules):
            block = layer.fn

            # 获取融合的注意力和前馈权重
            fused_attn_ff_weight = block.fused_attn_ff_proj.weight
            attn_out_weight = block.attn_out.weight

            # 获取融合后的投影输出维度
            fused_proj_out_dim = fused_attn_ff_weight.shape[0]

            # 打包 Q、K、V 权重
            lora_qkv_weight, _ = pack([lora_q.weight, lora_k.weight, lora_v.weight], 'i *')
            lora_qkv_weight = F.pad(lora_qkv_weight, (0, fused_proj_out_dim - lora_qkv_weight.shape[1]))

            # 重排 QKV 权重
            lora_qkv_weight = rearrange(lora_qkv_weight, 'i o -> o i')
            lora_o_weight = rearrange(lora_o.weight, 'i o -> o i')

            # 更新融合的注意力和前馈权重
            fused_attn_ff_weight.add_(lora_qkv_weight)
            attn_out_weight.add_(lora_o_weight)

    # 研究员首先训练 PALM 参数，然后进行微调

    # 获取 PALM 参数
    def palm_parameters(self):
        return set(self.parameters()) - set(self.finetune_modules.parameters())

    # 获取微调参数
    def finetune_parameters(self, scope = 'default'):
        assert scope in self.finetune_modules, f'finetune parameters of scope {scope} not found'
        return self.finetune_modules[scope].parameters()

    # 生成函数

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        seq_len,
        prompt = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        pad_value = 0.,
        eos_token = None,
        return_seq_without_prompt = True,
        use_tqdm = False,
        **kwargs
    ):
        # 如果没有指定提示，则随机生成一个
        if not exists(prompt):
            prompt = torch.randint(0, self.num_tokens, (1, 1))
            prompt = prompt.to(self.device)
            return_seq_without_prompt = False

        prompt, leading_dims = pack([prompt], '* n')

        n, out = prompt.shape[-1], prompt.clone()

        wrapper_fn = identity if not use_tqdm else tqdm
        sample_num_times = max(1, seq_len - prompt.shape[-1])

        for _ in wrapper_fn(range(sample_num_times)):
            logits, embeds = self.forward(out, return_logits_with_embedding = True, **kwargs)
            logits, embeds = logits[:, -1], embeds[:, -1]

            if exists(filter_logits_fn):
                logits = filter_logits_fn(logits, thres = filter_thres)

            sample = gumbel_sample(logits, temperature = temperature, dim = -1)
            out, _ = pack([out, sample], 'b *')

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # 掩盖掉 EOS 标记后的所有内容
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, pad_value)
                    break

        out, = unpack(out, leading_dims, '* n')

        if not return_seq_without_prompt:
            return out

        return out[..., n:]

    # 前向传播函数
    def forward(
        self,
        x,
        return_loss = False,
        disable_lora = False,
        finetune_scope = None,
        extra_embed = None,
        return_only_embedding = False,
        return_logits_with_embedding = False
        ):
        # 如果需要返回损失，则将输入数据 x 切片，分别作为输入和标签
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # 如果不是自回归模型，对编码器进行掩码处理
        # 将任何负数的标记视为需要屏蔽的标记 - 仅在非自回归情况下需要
        if not self.causal:
            mask = x >= 0
            x = x.masked_fill(~mask, 0)
        else:
            mask = None

        # 获取标记嵌入
        x = self.token_emb(x)

        # 如果存在额外的嵌入，则将其加到标记嵌入中
        if exists(extra_embed):
            x = x + extra_embed

        # 微调模块
        finetune_modules = tuple()
        if exists(finetune_scope) and not disable_lora:
            assert finetune_scope in self.finetune_modules
            finetune_modules = self.finetune_modules[finetune_scope]

        # 并行注意力 / 前馈块，传入微调 lora
        for layer, finetune_modules in zip_longest(self.layers, finetune_modules):
            x = layer(x, mask = mask, finetune_modules = finetune_modules)

        # 最终规范化
        embeds = self.norm(x)

        # 如果只需要返回嵌入，则直接返回嵌入
        if return_only_embedding:
            return embeds

        # 转换为逻辑值
        logits = self.to_logits(embeds)

        # 返回结果，根据需要返回逻辑值和嵌入或仅逻辑值
        ret = (logits, embeds) if return_logits_with_embedding else logits

        # 如果不需要返回损失，则直接返回结果
        if not return_loss:
            return ret

        # 重新排列逻辑值的维度，以便计算交叉熵损失
        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels, ignore_index = self.cross_entropy_ignore_index)
```