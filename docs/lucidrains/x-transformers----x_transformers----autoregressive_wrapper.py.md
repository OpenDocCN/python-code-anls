# `.\lucidrains\x-transformers\x_transformers\autoregressive_wrapper.py`

```py
# 从 math 模块中导入 ceil 和 log 函数
# 从 typing 模块中导入 Optional, Union, Tuple, Callable 类型
# 导入 torch 模块及其子模块
# 导入 nn, Tensor, Module 类
# 导入 torch.nn.functional 模块
# 导入 einops 模块中的 rearrange, pack, unpack 函数
from math import ceil, log
from typing import Optional, Union, Tuple, Callable

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, pack, unpack

# 检查变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回输入的函数
def identity(t, *args, **kwargs):
    return t

# 将输入转换为元组的函数
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else (t,) * length

# 评估装饰器函数
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# 对变长前缀进行右对齐的函数
def align_right(t, lens, pad_id = 0):
    batch, seq_len, device, dtype = *t.shape, t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device = device, dtype = torch.long)

    t = F.pad(t, (max_pad_len, 0), value = 0)
    offset = max_pad_len - pad_lens

    aligned = t[batch_arange, prompt_len_arange + offset[..., None]]
    return aligned

# nucleus 函数
def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk 函数
def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# top_a 函数
def top_a(logits, min_p_pow = 2.0, min_p_ratio = 0.02):
    probs = F.softmax(logits, dim = -1)
    max_probs = torch.amax(probs, dim = -1, keepdim = True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)

# 对比解码函数
def contrastive_decode_fn(
    expert_logits,
    amateur_logits,
    alpha = 0.1,
    beta = 0.5
):
    """
    Appendix A Algorithm 2
    https://arxiv.org/abs/2309.09117
    """

    cutoff = log(alpha) + expert_logits.amax(dim = -1, keepdim = True)
    diffs = (1 + beta) * expert_logits - beta * amateur_logits
    contrastive_decode_logits = diffs.masked_fill(expert_logits < cutoff, -torch.finfo(expert_logits.dtype).max)
    return contrastive_decode_logits

# 自回归包装器类
class AutoregressiveWrapper(Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        mask_prob = 0.,
        add_attn_z_loss = False
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # 论文表明在自回归解码器训练中与掩码（MLM）结合使用会带来很大的改进 https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

        # 是否添加路由器 z-loss
        self.add_attn_z_loss = add_attn_z_loss

    @torch.no_grad()
    @eval_decorator
    # 生成文本序列
    def generate(
        self,
        prompts,  # 输入的提示文本
        seq_len,  # 生成的序列长度
        eos_token = None,  # 结束标记
        temperature = 1.,  # 温度参数
        prompt_lens: Optional[Tensor] = None,  # 提示文本长度
        filter_logits_fn: Callable = top_k,  # 过滤 logits 的函数
        restrict_to_max_seq_len = True,  # 是否限制最大序列长度
        amateur_model: Optional[Union[Module, Tuple[Module]]] = None,  # 业余模型
        filter_kwargs: dict = dict(),  # 过滤参数
        contrastive_decode_kwargs: Union[dict, Tuple[dict]] = dict(  # 对比解码参数
            beta = 0.5,
            alpha = 0.1
        ),
        cache_kv = True,  # 是否缓存键值对
        **kwargs  # 其他参数
    def forward(self, x, return_outputs = False, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss

        # 输入和目标序列
        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        # 如果存在 mask_prob，则进行 mask 处理
        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max  # 第一个 token 不应被 mask 掉
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim = -1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask = mask)

        # 获取 logits 和缓存
        logits, cache = self.net(
            inp,
            return_intermediates = True,
            return_attn_z_loss = add_attn_z_loss,
            **kwargs
        )

        # 计算交叉熵损失
        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        # 如果存在注意力 z 损失，则加上
        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss

        # 如果不需要返回输出，则返回损失
        if not return_outputs:
            return loss

        # 否则返回损失和输出
        return loss, (logits, cache)
```