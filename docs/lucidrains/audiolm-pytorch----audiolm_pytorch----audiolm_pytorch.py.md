# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\audiolm_pytorch.py`

```py
# 导入数学库
import math
# 导入 functools 模块中的 partial 和 wraps 函数
from functools import partial, wraps

# 导入 beartype 库中的 Optional, Union, List 类型
from beartype.typing import Optional, Union, List
# 导入 beartype 库中的 beartype 装饰器
from beartype import beartype

# 导入 torch 库
import torch
# 导入 torch 库中的 nn, einsum, Tensor 模块
from torch import nn, einsum, Tensor
# 导入 torch 库中的 grad 函数，并重命名为 torch_grad
from torch.autograd import grad as torch_grad
# 导入 torch.nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 导入 torch.nn.utils.rnn 中的 pad_sequence 函数
from torch.nn.utils.rnn import pad_sequence

# 导入 torchaudio 库
import torchaudio

# 导入 einops 库中的 rearrange, repeat, reduce 函数
from einops import rearrange, repeat, reduce
# 导入 einops.layers.torch 中的 Rearrange 类
from einops.layers.torch import Rearrange

# 导入 audiolm_pytorch 库中的 FairseqVQWav2Vec 类
from audiolm_pytorch.vq_wav2vec import FairseqVQWav2Vec
# 导入 audiolm_pytorch 库中的 HubertWithKmeans 类
from audiolm_pytorch.hubert_kmeans import HubertWithKmeans

# 导入 audiolm_pytorch 库中的 t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME 函数
from audiolm_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# 导入 torchaudio.functional 中的 resample 函数
from torchaudio.functional import resample

# 导入 audiolm_pytorch 库中的 SoundStream 类
from audiolm_pytorch.soundstream import SoundStream
# 导入 audiolm_pytorch 库中的 EncodecWrapper 类
from audiolm_pytorch.encodec import EncodecWrapper
# 导入 audiolm_pytorch 库中的 AudioConditionerBase 类
from audiolm_pytorch.utils import AudioConditionerBase
# 导入 audiolm_pytorch 库中的 Attend 类
from audiolm_pytorch.attend import Attend

# 导入 tqdm 库中的 tqdm 函数
from tqdm import tqdm
# 导入 pathlib 库中的 Path 类
from pathlib import Path
# 导入 audiolm_pytorch.version 中的 __version__ 变量
from audiolm_pytorch.version import __version__
# 导入 packaging 库中的 version 模块
from packaging import version

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在，则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回一个始终返回指定值的函数
def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

# 如果函数存在，则返回该函数，否则返回一个始终返回 None 的函数
def maybe(fn):
    if not exists(fn):
        return always(None)

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

# 对两个数进行向上取整除法运算
def ceil_div(numer, denom):
    return (numer + denom - 1) // denom

# 计算使得 n 成为 mult 的倍数所需的余数
def remainder_needed_until_multiple(n, mult):
    return (ceil_div(n, mult) * mult) - n

# 将值向下舍入到最接近的倍数
def round_down_nearest_multiple(val, mult):
    return (val // mult) * mult

# 评估装饰器
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 张量辅助函数

# 生成一个与给定形状相同的掩码张量，其中一定比例的值被置为 0
def generate_mask_with_prob(shape, mask_prob, device):
    seq = shape[-1]
    rand = torch.randn(shape, device = device)
    rand[:, 0] = -torch.finfo(rand.dtype).max
    num_mask = min(int(seq * mask_prob), seq - 1)
    indices = rand.topk(num_mask, dim = -1).indices
    mask = ~torch.zeros(shape, device = device).scatter(1, indices, 1.).bool()
    return mask

# 注意力相关工具函数

# 缩小梯度的函数
def grad_shrink(t, alpha = 0.1):
    return t * alpha + t.detach() * (1 - alpha)

# 采样辅助函数

# 计算张量的自然对数
def log(t, eps = 1e-20):
    return torch.log(t + eps)

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 从 Gumbel 分布中采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# 保留前 k 个最大值，其余值置为负无穷
def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 在遇到特定值后的位置进行掩码
def mask_out_after_eos_id(t, eos_id, mask_value = -1, keep_eos = True):
    eos_mask = (t == eos_id).float()

    if keep_eos:
        eos_mask = F.pad(eos_mask, (1, -1))

    after_eos_mask = eos_mask.cumsum(dim = -1) > 0
    return t.masked_fill(after_eos_mask, mask_value)

# 检查所有行是否都包含特定值
def all_rows_have_eos_id(t, eos_id):
    eos_mask = (t == eos_id)
    return torch.any(eos_mask, dim = -1).all()

# 安全地拼接张量
def safe_cat(*tensors, dim = -2):
    args = [*filter(exists, tensors)]

    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        return torch.cat(args, dim = dim)

# 无监督分类器指导函数

# 生成与给定形状相同的概率掩码张量
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# 移除语义标记 id 中的唯一连续值
# 定义一个函数，用于在输入的 ids 张量末尾添加一个特定的 eos_id
def append_eos_id(ids, eos_id):
    # 获取 ids 张量的形状和设备信息
    b, device = ids.shape[0], ids.device
    # 创建一个只包含 eos_id 的张量，形状为 (1, )，设备与 ids 相同
    eos_ids = torch.ones(1, device=device).long() * eos_id
    # 将 eos_ids 重复 b 次，形状变为 (b, 1)
    eos_ids = repeat(eos_ids, '1 -> b 1', b=b)
    # 在 ids 张量的末尾拼接 eos_ids，dim=-1 表示在最后一个维度上拼接
    ids = torch.cat((ids, eos_ids), dim=-1)
    return ids

# 批量处理输入张量 t 中每个元素，使每个元素的值连续且唯一，用 pad_value 进行填充
def batch_unique_consecutive(t, pad_value=0.):
    # 对 t 沿着第 0 维度进行拆分，并对每个元素进行唯一连续化处理
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim=0)]
    # 对处理后的结果进行填充，batch_first=True 表示第一个维度为 batch 维度
    return pad_sequence(unique_arr, batch_first=True, padding_value=pad_value)

# 从 nn.Embedding 中获取嵌入向量，对于超出嵌入表范围的填充值使用 pad_id
def get_embeds(
    embeddings: nn.Embedding,
    codes: torch.Tensor,
    pad_id=-1,
    return_mask=False,
    mask_pad_pos_to=0
):
    # 创建一个与 codes 相同形状的布尔掩码，用于标记 pad_id 的位置
    pad_mask = codes == pad_id
    # 将 codes 中的 pad_id 替换为 0，作为嵌入表的索引
    codes_without_pad = codes.masked_fill(pad_mask, 0)
    # 从嵌入表中获取嵌入向量
    embeds = embeddings(codes_without_pad)

    # 如果指定了 mask_pad_pos_to，则将 pad_id 的位置替换为指定值
    if exists(mask_pad_pos_to):
        embeds = embeds.masked_fill(rearrange(pad_mask, '... -> ... 1'), mask_pad_pos_to)

    # 如果需要返回掩码，则返回嵌入向量和掩码的逻辑非
    if return_mask:
        return embeds, ~pad_mask

    return embeds

# 无偏置的 Layernorm，用于提高稳定性
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 相对位置偏置
class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        layers=3
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert j >= i
        device = self.device

        i_pos = torch.arange(i, device=device) + (j - i)
        j_pos = torch.arange(j, device=device)

        rel_pos = (rearrange(i_pos, 'i -> i 1') - rearrange(j_pos, 'j -> 1 j'))
        rel_pos += (j - 1)

        x = torch.arange(-j + 1, j, device=device).float()
        x = rearrange(x, '... -> ... 1')

        for layer in self.net:
            x = layer(x)

        x = x[rel_pos]
        return rearrange(x, 'i j h -> h i j')

# GEGLU 激活函数
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x

# FeedForward 层
def FeedForward(dim, mult=4, dropout=0.1):
    inner_dim = int(dim * 2 * mult / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )

# 注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        dim_head=64,
        dim_context=None,
        heads=8,
        norm_context=False,
        num_null_kv=0,
        dropout=0.1,
        scale=8,
        flash=False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化头数和是否使用因果关系
        self.heads = heads
        self.causal = causal
        # 计算内部维度
        inner_dim = dim_head * heads

        # 设置上下文维度，默认为输入维度
        dim_context = default(dim_context, dim)

        # 初始化 LayerNorm 层
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        # 初始化 Dropout 层
        self.attn_dropout = nn.Dropout(dropout)

        # 初始化空键值对数量和空键值对参数
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head)) if num_null_kv > 0 else None

        # 初始化线性变换层
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)

        # 初始化 Attend 层
        self.attend = Attend(
            flash = flash,
            dropout = dropout,
            causal = causal
        )

        # 初始化输出层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        prefix_context = None,
        prefix_context_mask = None,
        return_kv_cache = False,
        kv_cache = None
    ):
        # 获取输入张量的形状和设备信息
        b, n, _, device = *x.shape, x.device

        # 如果存在上下文信息，则进行归一化处理
        if exists(context):
            context = self.context_norm(context)

        # 如果存在前缀上下文信息，则进行处理
        kv_input = default(context, x)

        # 处理基于前缀的自注意力条件
        if exists(prefix_context):
            kv_input = torch.cat((prefix_context, kv_input), dim = -2)
            prefix_seq_len = prefix_context.shape[-2]

            if not exists(mask):
                mask = torch.ones((b, n), device = device, dtype = torch.bool)

            if exists(prefix_context_mask):
                mask = torch.cat((prefix_context_mask, mask), dim = -1)
            else:
                mask = F.pad(mask, (prefix_seq_len, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (prefix_seq_len, 0), value = 0.)

        # 预处理
        x = self.norm(x)

        # 为查询、键、值进行投影
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        # 处理键值缓存
        if exists(kv_cache):
            ck, cv = kv_cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # 存储键值缓存
        if return_kv_cache:
            kv_cache = torch.stack((k, v))

        # 处理空键/值对
        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b = b).unbind(dim = 0)
            k = torch.cat((null_k, k), dim = -2)
            v = torch.cat((null_v, v), dim = -2)

        # 分割为多头注意力
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # 处理掩码和空键/值对
        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)

        # 注意力计算
        out = self.attend(q, k, v, attn_bias = attn_bias, mask = mask)

        # 合并多头
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # 如果不需要返回键值缓存，则直接返回输出
        if not return_kv_cache:
            return out

        # 返回输出和键值缓存
        return out, kv_cache
# 定义 Transformer 类，用于实现 Transformer 模型
class Transformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 输入维度
        depth,  # Transformer 层数
        heads,  # 多头注意力头数
        dim_context = None,  # 上下文维度，默认为 None
        cross_attend = False,  # 是否进行跨注意力
        attn_dropout = 0.,  # 注意力层的 dropout 概率
        ff_dropout = 0.,  # FeedForward 层的 dropout 概率
        grad_shrink_alpha = 0.1,  # 梯度缩放参数
        cond_as_self_attn_prefix = False,  # 是否将条件作为自注意力前缀
        rel_pos_bias = True,  # 是否使用相对位置偏置
        flash_attn = False,  # 是否使用 Flash Attention
        **kwargs  # 其他参数
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        assert not (cross_attend and cond_as_self_attn_prefix)

        self.dim_context = default(dim_context, dim)

        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix

        self.grad_shrink = partial(grad_shrink, alpha = grad_shrink_alpha)

        self.layers = nn.ModuleList([])

        self.rel_pos_bias = RelativePositionBias(dim = dim // 2, heads = heads) if rel_pos_bias else None

        # 构建 Transformer 层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dropout = attn_dropout, flash = flash_attn, causal = True, **kwargs),
                Attention(dim = dim, heads = heads, dropout = attn_dropout, dim_context = dim_context, flash = flash_attn, num_null_kv = 1, norm_context = True, **kwargs) if cross_attend else None,
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

        self.norm = LayerNorm(dim)

    # 前向传播函数
    def forward(
        self,
        x,  # 输入张量
        self_attn_mask = None,  # 自注意力掩码
        context = None,  # 上下文张量
        context_mask = None,  # 上下文掩码
        attn_bias = None,  # 注意力偏置
        return_kv_cache = False,  # 是否返回键值缓存
        kv_cache = None  # 键值缓存
    ):
        assert not (self.cond_as_self_attn_prefix and not exists(context))
        assert not (exists(context) and context.shape[-1] != self.dim_context), f'you had specified a conditioning dimension of {self.dim_context}, yet what was received by the transformer has dimension of {context.shape[-1]}'

        n, device = x.shape[1], x.device

        # 从 cogview 论文中采用，GLM 130B LLM 采用，减少注意力网络不稳定性的可能性

        x = self.grad_shrink(x)

        # ���果使用条件作为自注意力前缀，则关闭键值缓存
        if self.cond_as_self_attn_prefix:
            kv_cache = None

        # 处理键值缓存
        new_kv_cache = []

        if exists(kv_cache):
            cache_len = kv_cache.shape[-2]
            kv_cache = iter(kv_cache)
        else:
            cache_len = 0
            kv_cache = iter([])

        x = x[:, cache_len:]

        # 相对位置偏置
        if exists(attn_bias):
            rel_pos_bias = attn_bias
        else:
            rel_pos_bias = maybe(self.rel_pos_bias)(n, n)

        if exists(rel_pos_bias):
            rel_pos_bias = rel_pos_bias[..., cache_len:, :]

        # 自注意力关键字参数
        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(
                prefix_context = context,
                prefix_context_mask = context_mask
            )

        # Transformer 层
        for attn, cross_attn, ff in self.layers:

            residual = x

            x, layer_kv_cache = attn(x, attn_bias = rel_pos_bias, mask = self_attn_mask, kv_cache = next(kv_cache, None), return_kv_cache = True, **self_attn_kwargs)
            new_kv_cache.append(layer_kv_cache)

            x = x + residual

            if exists(cross_attn):
                assert exists(context)

                x = cross_attn(x, context = context, mask = context_mask) + x

            x = ff(x) + x

        x = self.norm(x)

        if not return_kv_cache:
            return x

        return x, torch.stack(new_kv_cache)

# 定义 SemanticTransformer 类，用于实现语义 Transformer
class SemanticTransformer(nn.Module):
    @beartype
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        dim,  # 维度
        depth,  # 深度
        num_semantic_tokens,  # 语义标记数量
        heads = 8,  # 头数
        attn_dropout = 0.,  # 注意力丢弃率
        ff_dropout = 0.,  # 前馈网络丢弃率
        t5_name = DEFAULT_T5_NAME,  # T5模型名称
        cond_dim = None,  # 条件维度
        has_condition = False,  # 是否有条件
        audio_text_condition = False,  # 音频文本条件
        cond_as_self_attn_prefix = False,  # 条件作为自注意力前缀
        cond_drop_prob = 0.5,  # 条件丢弃概率
        grad_shrink_alpha = 0.1,  # 梯度缩减系数
        rel_pos_bias = True,  # 相对位置偏置
        flash_attn = False,  # 闪电注意力
        **kwargs  # 其他参数
    ):
        super().__init__()
        # 根据条件设置相对位置偏置
        rel_pos_bias = rel_pos_bias and not flash_attn

        self.num_semantic_tokens = num_semantic_tokens

        if audio_text_condition:
            has_condition = True
            cond_dim = default(cond_dim, dim)

        self.has_condition = has_condition
        # 文本嵌入函数
        self.embed_text = partial(t5_encode_text, name = t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.start_token = nn.Parameter(torch.randn(dim))

        # 语义嵌入
        self.semantic_embedding = nn.Embedding(num_semantic_tokens + 1, dim)
        self.eos_id = num_semantic_tokens

        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        # 文本嵌入投影
        self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()

        # Transformer模型
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            cross_attend = has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix = cond_as_self_attn_prefix,
            grad_shrink_alpha = grad_shrink_alpha,
            rel_pos_bias = rel_pos_bias,
            flash_attn = flash_attn,
            **kwargs
        )

        # 输出层
        self.to_logits = nn.Linear(dim, num_semantic_tokens + 1)

    # 设备属性
    @property
    def device(self):
        return next(self.parameters()).device

    # 加载模型
    def load(self, path):
        # 返回 pkg，以便如果此函数从 Trainer 函数调用中调用，则 Trainer 也可以访问从检查点加载的包
        device = self.device
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = device)
        # 检查版本
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')
        self.load_state_dict(pkg['model'])
        return pkg

    # 带条件缩放的前向传播
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,  # 条件缩放
        kv_cache = None,
        return_kv_cache = False,
        **kwargs
    ):
        kv_cache = iter(default(kv_cache, []))
        new_kv_caches = []

        logits, new_kv_cache = self.forward(*args, cond_drop_prob = 0., kv_cache = next(kv_cache, None), return_kv_cache = True, **kwargs)
        new_kv_caches.append(new_kv_cache)

        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return logits

            return logits, torch.stack(new_kv_caches)

        null_logits, null_new_kv_cache = self.forward(*args, cond_drop_prob = 1., kv_cache = next(kv_cache, None), return_kv_cache = True, **kwargs)
        new_kv_caches.append(null_new_kv_cache)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if not return_kv_cache:
            return scaled_logits

        return scaled_logits, torch.stack(new_kv_caches)

    # 前向传播
    @beartype
    def forward(
        self,
        *,
        ids = None,
        return_loss = False,
        text: Optional[List[str]] = None,
        text_embeds = None,
        self_attn_mask = None,
        cond_drop_prob = None,
        unique_consecutive = None,
        kv_cache = None,
        return_kv_cache = False
        ):
            # 获取当前设备
            device = self.device

            # 获取输入张量的批量大小
            b = ids.shape[0]

            # 检查是否存在文本或文本嵌入
            has_text = exists(text) or exists(text_embeds)
            # 断言条件：self.has_condition 与 has_text 不应该同时为真
            assert not (self.has_condition ^ has_text)

            # 初始化文本掩码为 None
            text_mask = None
            # 如果不存在文本嵌入且存在文本
            if not exists(text_embeds) and exists(text):
                # 在推理模式下
                with torch.inference_mode():
                    # 通过调用 self.embed_text 方法获取文本嵌入，输出设备为 device
                    text_embeds = self.embed_text(text, output_device = device)
                    # 生成文本掩码，标记非零元素
                    text_mask = torch.any(text_embeds != 0, dim = -1)

            # 如果存在文本嵌入
            if exists(text_embeds):
                # 通过 self.proj_text_embed 方法处理文本嵌入
                text_embeds = self.proj_text_embed(text_embeds)

            # 获取条件丢弃概率，默认为 self.cond_drop_prob
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

            # 如果存在文本掩码且条件丢弃概率大于 0
            if exists(text_mask) and cond_drop_prob > 0:
                # 生成保留掩码，概率为 1 - cond_drop_prob，设备为 device
                keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
                # 更新文本掩码，保留掩码与文本掩码按位与
                text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

            # 如果需要返回损失
            if return_loss:
                # 复制 ids 到 labels，并截取最后一个元素
                labels, ids = ids.clone(), ids[:, :-1]

            # 获取 tokens，通过 self.semantic_embedding 获取嵌入
            tokens = get_embeds(self.semantic_embedding, ids)

            # 生成起始 tokens，重复 self.start_token，维度变换为 'd -> b 1 d'，批量大小为 ids.shape[0]
            start_tokens = repeat(self.start_token, 'd -> b 1 d', b = ids.shape[0])

            # 拼接起始 tokens 和 tokens，沿着第二维度拼接
            tokens = torch.cat((start_tokens, tokens), dim = 1)

            # 如果存在 self_attn_mask
            if exists(self_attn_mask):
                # 在第二维度前面填充一个元素，值为 True
                self_attn_mask = F.pad(self_attn_mask, (1, 0), value = True)

            # 使用 transformer 处理 tokens，传入文本嵌入、自注意力掩码、文本掩码、kv_cache，并返回 kv_cache
            tokens, kv_cache = self.transformer(tokens, context = text_embeds, self_attn_mask = self_attn_mask, context_mask = text_mask, kv_cache = kv_cache, return_kv_cache = True)
            # 将 tokens 转换为 logits
            logits = self.to_logits(tokens)

            # 如果不需要返回 kv_cache，则返回 logits
            if not return_kv_cache:
                return logits

            # 返回 logits 和 kv_cache
            return logits, kv_cache
class CoarseTransformer(nn.Module):
    # 定义一个名为CoarseTransformer的类，继承自nn.Module

    @beartype
    def __init__(
        self,
        *,
        codebook_size,
        num_coarse_quantizers,
        dim,
        depth,
        num_semantic_tokens,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        t5_name = DEFAULT_T5_NAME,
        has_condition = False,
        cond_dim = None,
        audio_text_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        project_semantic_logits = True,
        rel_pos_bias = True,
        flash_attn = False,
        **kwargs
    ):
        # 初始化函数，接受一系列参数

        super().__init__()
        # 调用父类的初始化函数

        rel_pos_bias = rel_pos_bias and not flash_attn
        # 更新rel_pos_bias的值

        self.num_semantic_tokens = num_semantic_tokens
        # 设置类属性num_semantic_tokens为传入的num_semantic_tokens的值

        if audio_text_condition:
            # 如果audio_text_condition为True
            has_condition = True
            # 将has_condition设置为True
            cond_dim = default(cond_dim, dim)
            # 如果cond_dim为None，则将其设置为dim

        self.has_condition = has_condition
        # 设置类属性has_condition为传入的has_condition的值
        self.embed_text = partial(t5_encode_text, name = t5_name)
        # 设置类属性embed_text为t5_encode_text函数的partial函数，name参数为t5_name
        self.cond_drop_prob = cond_drop_prob
        # 设置类属性cond_drop_prob为传入的cond_drop_prob的值

        self.semantic_start_token = nn.Parameter(torch.randn(dim))
        # 设置类属性semantic_start_token为一个dim维的随机张量
        self.coarse_start_token = nn.Parameter(torch.randn(dim))
        # 设置类属性coarse_start_token为一个dim维的随机张量

        self.semantic_eos_id = num_semantic_tokens
        # 设置类属性semantic_eos_id为num_semantic_tokens
        self.semantic_embedding = nn.Embedding(num_semantic_tokens + 1, dim)
        # 设置类属性semantic_embedding为一个Embedding层，词汇表大小为num_semantic_tokens + 1，embedding维度为dim

        self.coarse_eos_id = codebook_size
        # 设置类属性coarse_eos_id为codebook_size
        codebook_size_with_eos = codebook_size + 1
        # 计算codebook_size_with_eos为codebook_size + 1

        self.coarse_embedding = nn.Embedding(num_coarse_quantizers * codebook_size_with_eos, dim)
        # 设置类属性coarse_embedding为一个Embedding层，词汇表大小为num_coarse_quantizers * codebook_size_with_eos，embedding维度为dim
        self.coarse_quantize_embedding = nn.Embedding(num_coarse_quantizers, dim)
        # 设置类属性coarse_quantize_embedding为一个Embedding层，词汇表大小为num_coarse_quantizers，embedding维度为dim

        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        # 计算text_dim为cond_dim或者get_encoded_dim(t5_name)的值
        self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()
        # 设置类属性proj_text_embed为一个线性层，输入维度为text_dim，输出维度为dim，不使用偏置项

        self.cross_attn_bias = nn.Parameter(torch.zeros(heads, 1, 1)) if rel_pos_bias else None
        # 设置类属性cross_attn_bias为一个形状为(heads, 1, 1)的参数张量，如果rel_pos_bias为True，否则为None

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            cross_attend = has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix = cond_as_self_attn_prefix,
            grad_shrink_alpha = grad_shrink_alpha,
            rel_pos_bias = rel_pos_bias,
            flash_attn = flash_attn,
            **kwargs
        )
        # 设置类属性transformer为一个Transformer模型，传入各种参数

        self.codebook_size = codebook_size
        # 设置类属性codebook_size为传入的codebook_size的值
        self.num_coarse_quantizers = num_coarse_quantizers
        # 设置类属性num_coarse_quantizers为传入的num_coarse_quantizers的值

        self.to_semantic_logits = nn.Linear(dim, num_semantic_tokens + 1) if project_semantic_logits else None
        # 设置类属性to_semantic_logits为一个线性层，输入维度为dim，输出维度为num_semantic_tokens + 1，如果project_semantic_logits为True，否则为None
        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size_with_eos, dim))
        # 设置类属性coarse_logit_weights为一个形状为(num_coarse_quantizers, codebook_size_with_eos, dim)的参数张量

    @property
    def device(self):
        # 定义一个device属性，返回第一个参数的设备
        return next(self.parameters()).device

    def load(self, path):
        # 定义一个load方法，加载模型参数

        device = self.device
        # 获取设备信息
        path = Path(path)
        # 将path转换为Path对象
        assert path.exists()
        # 断言path存在
        pkg = torch.load(str(path), map_location = device)
        # 加载模型参数
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            # 如果版本信息在pkg中且小于当前版本
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')
            # 打印模型训练的旧版本信息
        self.load_state_dict(pkg['model'])
        # 加载模型参数
        return pkg
        # 返回加载的模型参数

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        return_kv_cache = False,
        kv_cache = None,
        embed_cache = None,
        **kwargs
        # 定义一个前向传播方法，接受一系列参数
        ):
        # 从缓存中获取键值对缓存的迭代器
        iter_kv_cache = iter(default(kv_cache, []))
        # 从缓存中获取嵌入缓存的迭代器
        iter_embed_cache = iter(default(embed_cache, []))
        # 创建新的键值对缓存列表
        new_kv_caches = []
        # 创建新的嵌入缓存列表
        new_embed_caches = []

        # 调用 forward 方法进行前向传播，获取语义和粗糙logits，以及新的键值对缓存和嵌入缓存
        (semantic_logits, coarse_logits), (new_kv_cache, new_embed_cache) = self.forward(*args, cond_drop_prob = 0., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        # 将新的键值对缓存添加到列表中
        new_kv_caches.append(new_kv_cache)
        # 将新的嵌入缓存添加到列表中
        new_embed_caches.append(new_embed_cache)

        # 如果条件缩放为1或者没有条件
        if cond_scale == 1 or not self.has_condition:
            # 如果不需要返回键值对缓存，则返回语义logits和粗糙logits
            if not return_kv_cache:
                return semantic_logits, coarse_logits

            # 否则返回语义logits、粗糙logits以及新的键值对缓存和嵌入缓存
            return (semantic_logits, coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

        # 调用 forward 方法进行前向传播，获取空的语义和粗糙logits，以及新的空的键值对缓存和嵌入缓存
        (null_semantic_logits, null_coarse_logits), (null_new_kv_cache, null_new_embed_cache) = self.forward(*args, cond_drop_prob = 1., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        # 将新的空的键值对缓存添加到列表中
        new_kv_caches.append(null_new_kv_cache)
        # 将新的空的嵌入缓存添加到列表中
        new_embed_caches.append(null_new_embed_cache)

        # 初始化缩放后的语义logits为None
        scaled_semantic_logits = None
        # 如果空的语义logits存在
        if exists(null_semantic_logits):
            # 计算缩放后的语义logits
            scaled_semantic_logits = null_semantic_logits + (semantic_logits - null_semantic_logits) * cond_scale

        # 计算缩放后的粗糙logits
        scaled_coarse_logits = null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale

        # 如果不需要返回键值对缓存，则返回缩放后的语义logits和粗糙logits
        if not return_kv_cache:
            return scaled_semantic_logits, scaled_coarse_logits

        # 否则返回缩放后的语义logits、粗糙logits以及新的键值对缓存和嵌入缓存
        return (scaled_semantic_logits, scaled_coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

    @beartype
    def forward(
        self,
        *,
        semantic_token_ids,
        coarse_token_ids,
        self_attn_mask = None,
        text: Optional[List[str]] = None,
        text_embeds = None,
        cond_drop_prob = None,
        return_only_coarse_logits = False,
        return_cache = False,
        kv_cache = None,
        embed_cache = None
class FineTransformer(nn.Module):
    # 定义 FineTransformer 类，继承自 nn.Module
    def __init__(
        self,
        *,
        num_coarse_quantizers,
        num_fine_quantizers,
        codebook_size,
        dim,
        depth,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        t5_name = DEFAULT_T5_NAME,
        has_condition = False,
        cond_dim = None,
        audio_text_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        project_coarse_logits = True,
        pad_id = -1,
        rel_pos_bias = True,
        flash_attn = False,
        **kwargs
    ):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数

        rel_pos_bias = rel_pos_bias and not flash_attn
        # 更新 rel_pos_bias 变量的值

        if audio_text_condition:
            # 如果 audio_text_condition 为真
            has_condition = True
            # 将 has_condition 设置为 True
            cond_dim = default(cond_dim, dim)
            # 如果 cond_dim 为 None，则将其设置为 dim

        self.has_condition = has_condition
        # 设置类属性 has_condition
        self.embed_text = partial(t5_encode_text, name = t5_name)
        # 设置类属性 embed_text，使用 t5_encode_text 函数的部分应用
        self.cond_drop_prob = cond_drop_prob
        # 设置类属性 cond_drop_prob

        self.num_coarse_quantizers = num_coarse_quantizers
        # 设置类属性 num_coarse_quantizers

        self.coarse_start_token = nn.Parameter(torch.randn(dim))
        self.fine_start_token = nn.Parameter(torch.randn(dim))
        # 创建 nn.Parameter 类型的 coarse_start_token 和 fine_start_token

        self.coarse_embedding = nn.Embedding(num_coarse_quantizers * codebook_size, dim)
        self.fine_embedding = nn.Embedding(num_fine_quantizers * codebook_size, dim)
        # 创建 nn.Embedding 类型的 coarse_embedding 和 fine_embedding

        self.coarse_quantize_embedding = nn.Embedding(num_coarse_quantizers, dim)
        self.fine_quantize_embedding = nn.Embedding(num_fine_quantizers, dim)
        # 创建 nn.Embedding 类型的 coarse_quantize_embedding 和 fine_quantize_embedding

        self.pad_id = pad_id
        self.eos_id = codebook_size
        # 设置类属性 pad_id 和 eos_id

        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()
        # 根据条件设置类属性 proj_text_embed

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            cross_attend = has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix = cond_as_self_attn_prefix,
            rel_pos_bias = False,
            grad_shrink_alpha = grad_shrink_alpha,
            flash_attn = flash_attn,
            **kwargs
        )
        # 创建 Transformer 类型的 transformer

        self.null_pos_bias = nn.Parameter(torch.randn(heads, 1, 1)) if rel_pos_bias else None
        # 创建 nn.Parameter 类型的 null_pos_bias

        pos_bias_mlp_dim = dim // 2
        self.pos_bias_mlp = nn.Sequential(
            nn.Linear(2, pos_bias_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_bias_mlp_dim, pos_bias_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_bias_mlp_dim, heads)
        ) if rel_pos_bias else None
        # 创建 nn.Sequential 类型的 pos_bias_mlp

        self.codebook_size = codebook_size
        self.num_coarse_quantizers = num_coarse_quantizers
        self.num_fine_quantizers = num_fine_quantizers
        # 设置类属性 codebook_size, num_coarse_quantizers, num_fine_quantizers

        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size, dim)) if project_coarse_logits else None
        self.fine_logit_weights = nn.Parameter(torch.randn(num_fine_quantizers, codebook_size, dim))
        # 创建 nn.Parameter 类型的 coarse_logit_weights 和 fine_logit_weights

    @property
    def device(self):
        return next(self.parameters()).device
    # 定义 device 属性，返回第一个参数的设备信息

    def load(self, path):
        # 加载模型参数
        device = self.device
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = device)
        # 加载模型参数
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')
        self.load_state_dict(pkg['model'])
        # 加载模型参数
        return pkg
        # 返回加载的模型参数
    # 定义一个带有条件缩放的前向传播函数
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,  # 设置默认的条件缩放比例为3
        return_kv_cache = False,  # 设置默认不返回kv缓存
        kv_cache = None,  # 初始化kv缓存为None
        embed_cache = None,  # 初始化嵌入缓存为None
        **kwargs
    ):
        # 生成kv缓存的迭代器
        iter_kv_cache = iter(default(kv_cache, []))
        # 生成嵌入缓存的迭代器
        iter_embed_cache = iter(default(embed_cache, []))
        # 初始化新的kv缓存列表
        new_kv_caches = []
        # 初始化新的嵌入缓存列表
        new_embed_caches = []

        # 调用self.forward函数进行前向传播，并返回新的kv缓存和嵌入缓存
        (semantic_logits, coarse_logits), (new_kv_cache, new_embed_cache) = self.forward(*args, cond_drop_prob = 0., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        # 将新的kv缓存添加到列表中
        new_kv_caches.append(new_kv_cache)
        # 将新的嵌入缓存添加到列表中
        new_embed_caches.append(new_embed_cache)

        # 如果条件缩放为1或者没有条件，则直接返回结果
        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return semantic_logits, coarse_logits

            return (semantic_logits, coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

        # 调用self.forward函数进行前向传播，条件概率为1，返回新的kv缓存和嵌入缓存
        (null_semantic_logits, null_coarse_logits), (null_new_kv_cache, null_new_embed_cache) = self.forward(*args, cond_drop_prob = 1., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        # 将新的kv缓存添加到列表中
        new_kv_caches.append(null_new_kv_cache)
        # 将新的嵌入缓存添加到列表中
        new_embed_caches.append(null_new_embed_cache)

        # 计算缩放后的语义logits
        scaled_semantic_logits = None
        if exists(null_semantic_logits):
            scaled_semantic_logits = null_semantic_logits + (semantic_logits - null_semantic_logits) * cond_scale

        # 计算缩放后的粗糙logits
        scaled_coarse_logits = null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale

        # 如果不返回kv缓存，则直接返回缩放后的结果
        if not return_kv_cache:
            return scaled_semantic_logits, scaled_coarse_logits

        return (scaled_semantic_logits, scaled_coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

    # 定义一个前向传播函数
    def forward(
        self,
        coarse_token_ids,
        fine_token_ids,
        text: Optional[List[str]] = None,
        text_embeds = None,
        cond_drop_prob = None,
        self_attn_mask = None,
        kv_cache = None,
        embed_cache = None,
        return_cache = False,
        return_only_fine_logits = False
# 定义一个语义转换器包装类
class SemanticTransformerWrapper(nn.Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        *,
        transformer: SemanticTransformer,  # 语义转换器对象
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,  # 可选的音频编码器对象
        audio_conditioner: Optional[AudioConditionerBase] = None,  # 可选的音频调节器对象
        pad_id = -1,  # 填充标识符，默认为-1
        unique_consecutive = True,  # 是否唯一连续，默认为True
        mask_prob = 0.15  # 掩码概率，默认为0.15
    ):
        super().__init__()  # 调用父类的初始化函数
        self.wav2vec = wav2vec  # 设置音频编码器对象
        self.transformer = transformer  # 设置语义转换器对象
        self.to(transformer.device)  # 将模型移动到语义转换器所在的设备
        self.audio_conditioner = audio_conditioner  # 设置音频调节器对象

        # 断言条件，如果音频调节器存在且语义转换器没有条件，则抛出异常
        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

        # 断言条件，如果音频编码器存在且音频编码器的码书大小与语义转换器的语义标记数相同，则通过，否则抛出异常
        assert not exists(self.wav2vec) or self.wav2vec.codebook_size == transformer.num_semantic_tokens, f'num_semantic_tokens on SemanticTransformer must be set to {self.wav2vec.codebook_size}'

        self.unique_consecutive = unique_consecutive  # 设置是否唯一连续
        self.pad_id = pad_id  # 设置填充标识符
        self.eos_id = transformer.eos_id  # 设置结束标识符
        self.mask_prob = mask_prob  # 设置掩码概率

    # 返回模型所在设备
    @property
    def device(self):
        return next(self.parameters()).device

    # 嵌入文本
    def embed_text(self, text):
        return self.transformer.embed_text(text, output_device = self.device)

    # 生成函数
    @eval_decorator
    @torch.inference_mode()
    @beartype
    def generate(
        self,
        *,
        max_length,  # 最大长度
        text: Optional[List[str]] = None,  # 文本列表
        text_embeds = None,  # 文本嵌入
        prime_wave = None,  # 主要波形
        prime_wave_input_sample_hz = None,  # 主要波形输入采样频率
        prime_ids = None,  # 主要标识符
        batch_size = 1,  # 批大小
        cond_scale = 3,  # 条件规模
        filter_thres = 0.9,  # 过滤阈值
        temperature = 1.,  # 温度
        use_kv_cache = True,  # 是否使用键值缓存
        include_eos_in_output = True,  # 输出中是否包含结束标识符，如果进行分层采样，必须保留结束标识符以便操作
        **kwargs  # 其他参数
    ):
        # 获取当前对象的设备
        device = self.device

        # 从输入波形派生 wav2vec ids

        # 如果存在 prime_wave
        if exists(prime_wave):
            # 确保 prime_ids 不存在
            assert not exists(prime_ids)
            # 确保 self.wav2vec 存在
            assert exists(self.wav2vec)
            # 使用 self.wav2vec 从 prime_wave 中获取 ids
            ids = self.wav2vec(
                prime_wave,
                flatten = False,
                input_sample_hz = prime_wave_input_sample_hz
            )
        # 如果存在 prime_ids
        elif exists(prime_ids):
            ids = prime_ids
        else:
            # 创建一个空的张量作为 ids
            ids = torch.empty((batch_size, 0), dtype = torch.long, device = device)

        # 如果需要唯一连续的 ids
        if self.unique_consecutive:
            # 对 ids 进行唯一连续处理
            ids = batch_unique_consecutive(ids, pad_value = self.pad_id)

        # 如果需要派生联合音频文本嵌入
        if exists(self.audio_conditioner) and exists(prime_wave):
            # 确保 text 和 text_embeds 不存在
            assert not exists(text) and not exists(text_embeds)
            # 使用 self.audio_conditioner 从 prime_wave 中获取文本嵌入
            text_embeds = self.audio_conditioner(wavs = prime_wave, namespace = 'semantic')

        # 如果需要派生文本嵌入
        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            # 使用 transformer.embed_text 从文本中获取文本嵌入
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        # 初始化变量
        batch = ids.shape[0]
        start_length = ids.shape[-1]
        sample_semantic_ids = ids.clone()
        last_logit_indices = (ids != self.pad_id).sum(dim = -1).long()
        kv_cache = None
        logits = None

        # 从 transformer 中采样
        for ind in tqdm(range(start_length, max_length), desc = 'generating semantic'):

            new_logits, new_kv_cache = self.transformer.forward_with_cond_scale(
                ids = sample_semantic_ids,
                text_embeds = text_embeds,
                cond_scale = cond_scale,
                kv_cache = kv_cache,
                return_kv_cache = True,
                **kwargs
            )

            if use_kv_cache:
                kv_cache = new_kv_cache
                logits = safe_cat(logits, new_logits, dim = -2)
            else:
                logits = new_logits

            last_logit_indices_expanded = repeat(last_logit_indices, 'b -> b 1 c', b = batch, c = logits.shape[-1])
            last_logits = logits.gather(1, last_logit_indices_expanded)
            last_logits = rearrange(last_logits, 'b 1 c -> b c')

            filtered_logits = top_k(last_logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            sampled = rearrange(sampled, 'b -> b 1')
            sample_semantic_ids = torch.cat((sample_semantic_ids, sampled), dim = -1)

            if all_rows_have_eos_id(sample_semantic_ids, self.eos_id):
                break

            last_logit_indices += 1

        sample_semantic_ids = mask_out_after_eos_id(sample_semantic_ids, self.eos_id, keep_eos = False)

        return sample_semantic_ids

    # 前向传播函数
    def forward(
        self,
        *,
        semantic_token_ids = None,
        raw_wave = None,
        text = None,
        text_embeds = None,
        return_loss = False,
        **kwargs
        ):
            # 断言要么给定原始波形（raw_wave），要么给定语义标记（semantic_token_ids）
            assert exists(raw_wave) or exists(semantic_token_ids), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

            if exists(self.audio_conditioner):
                # 断言存在原始波形
                assert exists(raw_wave)
                # 断言不存在文本和文本嵌入
                assert not exists(text) and not exists(text_embeds)
                # 使用音频调节器处理原始波形，生成语义嵌入
                text_embeds = self.audio_conditioner(wavs = raw_wave, namespace = 'semantic')

            if not exists(semantic_token_ids):
                # 断言存在 VQWav2Vec 模型
                assert exists(self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
                # 使用 VQWav2Vec 模型处理原始波形，生成语义标记
                semantic_token_ids = self.wav2vec(raw_wave, flatten = False)

            # 重新排列语义标记的维度
            semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')

            if self.training:
                # 如果是训练模式，为语义标记添加结束标记
                semantic_token_ids = append_eos_id(semantic_token_ids, self.transformer.eos_id)

            if self.unique_consecutive:
                # 如果需要唯一连续的语义标记，进行处理
                semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value = self.pad_id)

            # 输入标记为语义标记
            input_ids = semantic_token_ids
            if return_loss:
                # 如果需要返回损失，将输入标记截断最后一个标记
                input_ids = semantic_token_ids[:, :-1]

            self_attn_mask = None
            if self.mask_prob > 0. and self.training:
                # 如果需要进行掩码处理，生成掩码
                self_attn_mask = generate_mask_with_prob(input_ids.shape, self.mask_prob, input_ids.device)

            # 使用 Transformer 模型进行前向传播
            logits = self.transformer(
                ids = input_ids,
                text = text,
                text_embeds = text_embeds,
                self_attn_mask = self_attn_mask,
                **kwargs
            )

            if not return_loss:
                # 如果不需要返回损失，直接返回预测结果
                return logits

            # 计算交叉熵损失
            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                semantic_token_ids,
                ignore_index = self.pad_id
            )

            return loss
class CoarseTransformerWrapper(nn.Module):
    # 定义一个名为CoarseTransformerWrapper的类，继承自nn.Module
    @beartype
    def __init__(
        self,
        *,
        transformer: CoarseTransformer,
        codec: Optional[Union[SoundStream, EncodecWrapper]]  = None,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        pad_id = -1,
        unique_consecutive = True,
        semantic_cross_entropy_loss_weight = 1.,
        mask_prob = 0.15
    ):
        # 初始化函数，接受一系列参数
        super().__init__()
        # 调用父类的初始化函数
        self.codec = codec
        # 将参数codec赋值给实例变量self.codec
        self.wav2vec = wav2vec
        # 将参数wav2vec赋值给实例变量self.wav2vec

        self.transformer = transformer
        # 将参数transformer赋值给实例变量self.transformer
        self.to(transformer.device)
        # 将transformer的设备信息赋值给当前实例
        self.audio_conditioner = audio_conditioner
        # 将参数audio_conditioner赋值给实例变量self.audio_conditioner

        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'
        # 断言条件，如果条件不成立则抛出异常

        self.unique_consecutive = unique_consecutive
        # 将参数unique_consecutive赋值给实例变量self.unique_consecutive
        self.pad_id = pad_id
        # 将参数pad_id赋值给实例变量self.pad_id

        self.semantic_cross_entropy_loss_weight = semantic_cross_entropy_loss_weight
        # 将参数semantic_cross_entropy_loss_weight赋值给实例变量self.semantic_cross_entropy_loss_weight

        self.num_coarse_quantizers = transformer.num_coarse_quantizers * codec.rq_groups
        # 计算粗粒度量化器的数量
        self.semantic_eos_id = transformer.semantic_eos_id
        # 将transformer的语义结束符ID赋值给实例变量self.semantic_eos_id
        self.coarse_eos_id = transformer.coarse_eos_id
        # 将transformer的粗粒度结束符ID赋值给实例变量self.coarse_eos_id

        self.mask_prob = mask_prob
        # 将参数mask_prob赋值给实例变量self.mask_prob

    @property
    def device(self):
        # 定义一个device属性，返回参数的设备信息
        return next(self.parameters()).device

    @eval_decorator
    @torch.inference_mode()
    @beartype
    def generate(
        self,
        *,
        semantic_token_ids,
        prime_wave: Optional[Tensor] = None,
        prime_wave_input_sample_hz = None,
        prime_coarse_token_ids: Optional[Tensor] = None,
        text: Optional[List[str]] = None,
        text_embeds = None,
        max_time_steps = 512,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        reconstruct_wave = False,
        use_kv_cache = True,
        **kwargs
    ):
        # 定义一个生成函数，接受一系列参数
        pass
        # 占位符，暂时不做任何操作

    def forward(
        self,
        *,
        semantic_token_ids = None,
        raw_wave = None,
        raw_wave_for_codec = None,
        text = None,
        text_embeds = None,
        coarse_token_ids = None,
        return_loss = False,
        **kwargs
    ):
        # 定义一个前向传播函数，接受一系列参数
        pass
        # 占位符，暂时不做任何操作

class FineTransformerWrapper(nn.Module):
    # 定义一个名为FineTransformerWrapper的类，继承自nn.Module
    @beartype
    def __init__(
        self,
        *,
        transformer: FineTransformer,
        codec: Optional[Union[SoundStream, EncodecWrapper]] = None,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        coarse_cross_entropy_loss_weight = 1.,
        pad_id = -1,
        mask_prob = 0.15
    ):
        # 初始化函数，接受一系列参数
        super().__init__()
        # 调用父类的初始化函数
        self.codec = codec
        # 将参数codec赋值给实例变量self.codec

        self.transformer = transformer
        # 将参数transformer赋值给实例变量self.transformer
        self.to(transformer.device)
        # 将transformer的设备信息赋值给当前实例
        self.audio_conditioner = audio_conditioner
        # 将参数audio_conditioner赋值给实例变量self.audio_conditioner

        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'
        # 断言条件，如果条件不成立则抛出异常

        self.num_fine_quantizers = transformer.num_fine_quantizers * codec.rq_groups
        # 计算细粒度量化器的数量
        self.num_coarse_quantizers = transformer.num_coarse_quantizers * codec.rq_groups
        # 计算粗粒度量化器的数量

        if exists(codec):
            assert (self.num_fine_quantizers + self.num_coarse_quantizers) == (codec.num_quantizers * codec.rq_groups), 'number of fine and coarse quantizers on fine transformer must add up to total number of quantizers on codec'
        # 断言条件，如果条件不成立则抛出异常

        self.eos_id = transformer.eos_id
        # 将transformer的结束符ID赋值给实例变量self.eos_id

        assert self.num_coarse_quantizers > 0
        # 断言条件，如果条件不成立则抛出异常

        self.pad_id = pad_id
        # 将参数pad_id赋值给实例变量self.pad_id
        self.coarse_cross_entropy_loss_weight = coarse_cross_entropy_loss_weight
        # 将参数coarse_cross_entropy_loss_weight赋值给实例变量self.coarse_cross_entropy_loss_weight

        self.mask_prob = mask_prob
        # 将参数mask_prob赋值给实例变量self.mask_prob

    @property
    def device(self):
        # 定义一个device属性，返回参数的设备信息
        return next(self.parameters()).device

    @eval_decorator
    @torch.inference_mode()
    @beartype
    # 装饰器，用于评估和推断模式
    # 定义一个生成函数，用于生成音频波形
    def generate(
        self,
        *,
        coarse_token_ids,  # 粗粒度音频标记的张量
        prime_wave: Optional[Tensor] = None,  # 初始波形张量，默认为None
        prime_wave_input_sample_hz = None,  # 初始波形输入采样率，默认为None
        prime_fine_token_ids: Optional[Tensor] = None,  # 初始细粒度音频标记的张量，默认为None
        text: Optional[List[str]] = None,  # 文本列表，默认为None
        text_embeds = None,  # 文本嵌入，默认为None
        cond_scale = 3.,  # 条件缩放，默认为3.0
        filter_thres = 0.9,  # 过滤阈值，默认为0.9
        temperature = 1.,  # 温度，默认为1.0
        reconstruct_wave = False,  # 是否重建波形，默认为False
        use_kv_cache = True,  # 是否使用键值缓存，默认为True
        mask_out_generated_fine_tokens = False,  # 是否屏蔽生成的细粒度标记，默认为False
        **kwargs  # 其他关键字参数
    # 定义一个前向传播函数，用于模型的前向传播计算
    def forward(
        self,
        *,
        raw_wave = None,  # 原始波形，默认为None
        text = None,  # 文本，默认为None
        text_embeds = None,  # 文本嵌入，默认为None
        token_ids = None,  # 标记ID，默认为None
        coarse_token_ids = None,  # 粗粒度音频标记的张量，默认为None
        fine_token_ids = None,  # 细粒度音频标记的张量，默认为None
        return_loss = False,  # 是否返回损失，默认为False
        **kwargs  # 其他关键字参数
        ):
            # 断言条件：要么存在原始波形数据，要么存在粗糙和细粒度的令牌ID，但不能同时存在
            assert exists(raw_wave) ^ (exists(token_ids) ^ (exists(coarse_token_ids) and exists(fine_token_ids))), 'either raw waveform (raw_wav) is given, or coarse and fine token ids (coarse_token_ids, fine_token_ids)'

            if exists(self.audio_conditioner):
                # 断言条件：存在原始波形数据
                assert exists(raw_wave)
                # 断言条件：不存在文本和文本嵌入
                assert not exists(text) and not exists(text_embeds)
                # 使用音频调节器处理原始波形数据，生成细粒度的文本嵌入
                text_embeds = self.audio_conditioner(wavs = raw_wave, namespace = 'fine') # technically audio embeds, but shared text-audio joint embedding space for mulan

            if exists(raw_wave):
                # 断言条件：存在编解码器
                assert exists(self.codec), 'Codec must be provided if given raw wave for training'

                with torch.inference_mode():
                    # 设置编解码器为评估模式
                    self.codec.eval()
                    # 使用编解码器处理原始波形数据，返回编码后的令牌ID
                    _, token_ids, _ = self.codec(raw_wave, return_encoded = True)

                    batch, num_timesteps = raw_wave.shape
                    num_frames = int(num_timesteps / self.codec.seq_len_multiple_of)

                    # 断言条件：令牌ID的形状应为(batch, num_frames, num_coarse_quantizers + num_fine_quantizers)
                    assert token_ids.shape == torch.Size((batch, num_frames, self.num_coarse_quantizers + self.num_fine_quantizers)), \
                        f'Expected token ids to have shape (batch, num_frames, num_coarse_quantizers + num_fine_quantizers), but got {token_ids.shape}'

            if exists(token_ids):
                # 将令牌ID分为粗糙和细粒度的令牌ID
                coarse_token_ids, fine_token_ids = token_ids[..., :self.num_coarse_quantizers], token_ids[..., self.num_coarse_quantizers:]

            # 重新排列粗糙和细粒度的令牌ID
            coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')
            fine_token_ids = rearrange(fine_token_ids, 'b ... -> b (...)')

            # 如果是训练阶段，确定标签，应从细粒度的令牌ID中删除一个
            if return_loss:
                coarse_labels = coarse_token_ids
                fine_labels = fine_token_ids
                fine_token_ids = fine_token_ids[:, :-1]

            # 忘记性因果掩码 - 结构化丢失
            self_attn_mask = None

            if self.mask_prob > 0 and self.training:
                mask_shape = (
                    coarse_token_ids.shape[0],
                    coarse_token_ids.shape[-1] + fine_token_ids.shape[-1] + 2
                )

                # 生成具有概率的掩码
                self_attn_mask = generate_mask_with_prob(mask_shape, self.mask_prob, device = self.device)

            # 获取粗糙和细粒度的逻辑值
            coarse_logits, fine_logits = self.transformer(
                coarse_token_ids = coarse_token_ids,
                fine_token_ids = fine_token_ids,
                self_attn_mask = self_attn_mask,
                text = text,
                text_embeds = text_embeds,
                **kwargs
            )

            # 提前返回逻辑值
            if not return_loss:
                return coarse_logits, fine_logits

            # 重新排列逻辑值的维度
            coarse_logits, fine_logits = map(lambda t: maybe(rearrange)(t, 'b n c -> b c n'), (coarse_logits, fine_logits))

            num_fine_logits = fine_logits.shape[-1]

            num_coarse_logits = 0
            coarse_loss = 0.

            if self.coarse_cross_entropy_loss_weight > 0 and exists(coarse_logits):
                num_coarse_logits = coarse_logits.shape[-1]

                # 计算粗糙损失
                coarse_loss = F.cross_entropy(
                    coarse_logits,
                    coarse_labels,
                    ignore_index = self.pad_id
                )

            # 计算细粒度损失
            fine_loss = F.cross_entropy(
                fine_logits,
                fine_labels,
                ignore_index = self.pad_id
            )

            # 返回损失值
            return (
                coarse_loss * num_coarse_logits * self.coarse_cross_entropy_loss_weight +
                fine_loss * num_fine_logits
            ) / (num_coarse_logits + num_fine_logits)
# 定义一个名为 AudioLM 的类，用于处理音频语言模型相关任务
class AudioLM(nn.Module):
    # 初始化函数，接受多个参数
    @beartype
    def __init__(
        self,
        *,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]], 
        codec: Union[SoundStream, EncodecWrapper],
        semantic_transformer: SemanticTransformer,
        coarse_transformer: CoarseTransformer,
        fine_transformer: FineTransformer,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        unique_consecutive = True
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 将传入的音频条件器参数赋值给对象属性
        self.audio_conditioner = audio_conditioner

        # 断言语义变换器的语义标记数与粗糙变换器的语义标记数相等
        assert semantic_transformer.num_semantic_tokens == coarse_transformer.num_semantic_tokens
        # 断言粗糙变换器的码书大小与细化变换器的码书大小相等
        assert coarse_transformer.codebook_size == fine_transformer.codebook_size
        # 断言粗糙变换器的粗糙量化器数量与细化变换器的粗糙量化器数量相等
        assert coarse_transformer.num_coarse_quantizers == fine_transformer.num_coarse_quantizers
        # 断言细化变换器的粗糙量化器数量与细化量化器数量之和等于编解码器的量化器数量
        assert (fine_transformer.num_coarse_quantizers + fine_transformer.num_fine_quantizers) == codec.num_quantizers

        # 检查是否需要文本输入
        self.semantic_has_condition = semantic_transformer.has_condition
        self.coarse_has_condition = coarse_transformer.has_condition
        self.fine_has_condition = fine_transformer.has_condition
        self.needs_text = any([self.semantic_has_condition, self.coarse_has_condition, self.fine_has_condition])

        # 创建语义变换器包装器对象
        self.semantic = SemanticTransformerWrapper(
            wav2vec = wav2vec,
            transformer = semantic_transformer,
            audio_conditioner = audio_conditioner,
            unique_consecutive = unique_consecutive
        )

        # 创建粗糙变换器包装器对象
        self.coarse = CoarseTransformerWrapper(
            wav2vec = wav2vec,
            codec = codec,
            transformer = coarse_transformer,
            audio_conditioner = audio_conditioner,
            unique_consecutive = unique_consecutive
        )

        # 创建细化变换器包装器对象
        self.fine = FineTransformerWrapper(
            codec= codec,
            transformer = fine_transformer,
            audio_conditioner = audio_conditioner
        )

    # 定义 device 属性，返回模型参数所在的设备
    @property
    def device(self):
        return next(self.parameters()).device

    # 定义前向传播函数，接受多个参数
    @eval_decorator
    @torch.inference_mode()
    def forward(
        self,
        *,
        batch_size = 1,
        text: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        prime_wave = None,
        prime_wave_input_sample_hz = None,
        prime_wave_path = None,
        max_length = 2048,
        return_coarse_generated_wave = False,
        mask_out_generated_fine_tokens = False
    ):
        # 断言条件：如果需要文本信息，但文本信息和文本嵌入都不存在，则抛出异常
        assert not (self.needs_text and (not exists(text) and not exists(text_embeds))), 'text needs to be passed in if one of the transformer requires conditioning'

        # 如果需要文本信息
        if self.needs_text:
            # 如果文本信息存在，则使用语义模型将文本嵌入
            if exists(text):
                text_embeds = self.semantic.embed_text(text)

        # 断言条件：如果提示音频既存在`prime_wave`又存在`prime_wave_path`，则抛出异常
        assert not (exists(prime_wave) and exists(prime_wave_path)), 'prompt audio must be given as either `prime_wave: Tensor` or `prime_wave_path: str`'

        # 如果`prime_wave`存在
        if exists(prime_wave):
            # 断言条件：必须提供提示音频的输入采样频率`prime_wave_input_sample_hz`
            assert exists(prime_wave_input_sample_hz), 'the input sample frequency for the prompt audio must be given as `prime_wave_input_sample_hz: int`'
            # 将`prime_wave`转移到指定设备
            prime_wave = prime_wave.to(self.device)
        # 如果`prime_wave_path`存在
        elif exists(prime_wave_path):
            # 将`prime_wave_path`转换为路径对象
            prime_wave_path = Path(prime_wave_path)
            # 断言条件：确保文件存在于指定路径
            assert exists(prime_wave_path), f'file does not exist at {str(prime_wave_path)}'

            # 加载提示音频和其输入采样频率
            prime_wave, prime_wave_input_sample_hz = torchaudio.load(str(prime_wave_path))
            prime_wave = prime_wave.to(self.device)

        # 使用语义模型生成语义标记
        semantic_token_ids = self.semantic.generate(
            text_embeds = text_embeds if self.semantic_has_condition else None,
            batch_size = batch_size,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            max_length = max_length
        )

        # 使用粗糙模型生成粗糙标记或重构音频波形
        coarse_token_ids_or_recon_wave = self.coarse.generate(
            text_embeds = text_embeds if self.coarse_has_condition else None,
            semantic_token_ids = semantic_token_ids,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            reconstruct_wave = return_coarse_generated_wave
        )

        # 如果需要返回生成的粗糙音频波形
        if return_coarse_generated_wave:
            return coarse_token_ids_or_recon_wave

        # 使用精细模型生成细化标记或重构音频波形
        generated_wave = self.fine.generate(
            text_embeds = text_embeds if self.fine_has_condition else None,
            coarse_token_ids = coarse_token_ids_or_recon_wave,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            reconstruct_wave = True,
            mask_out_generated_fine_tokens = mask_out_generated_fine_tokens
        )

        # 返回生成的音频波形
        return generated_wave
```