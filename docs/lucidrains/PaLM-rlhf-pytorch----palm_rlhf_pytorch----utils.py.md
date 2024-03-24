# `.\lucidrains\PaLM-rlhf-pytorch\palm_rlhf_pytorch\utils.py`

```
# 导入 math、torch 模块，以及从 torch 模块中导入 einsum、nn 和 nn.functional 模块
import math
import torch
from torch import einsum, nn
import torch.nn.functional as F

# 从 einops 模块中导入 rearrange 函数
from einops import rearrange

# 检查变量是否存在的函数
def exists(val):
    return val is not None

# 装饰器函数

# 评估装饰器函数，用于在执行函数时将模型设置为评估模式
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# 张量辅助函数

# 对张量取对数，避免取对数时出现负无穷
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 计算带掩码的平均值，如果没有掩码则直接计算平均值
def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if not exists(mask):
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

# 采样辅助函数

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 使用 Gumbel 噪声对张量进行采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# Top-p 采样方法
def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# Top-k 采样方法
def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs
```