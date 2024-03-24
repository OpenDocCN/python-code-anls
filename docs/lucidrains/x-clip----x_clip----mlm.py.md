# `.\lucidrains\x-clip\x_clip\mlm.py`

```
import math
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F

# 定义一些辅助函数

# 根据概率生成掩码
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# 根据特定的 token_ids 生成掩码
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

# 根据概率和掩码生成子集掩码
def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

# 主要类

class MLM(nn.Module):
    def __init__(
        self,
        transformer,
        *,
        dim,
        num_tokens,
        mask_prob = 0.15,
        replace_prob = 0.9,
        random_token_prob = 0.,
        mask_token_id = 2,
        pad_token_id = 0,
        mask_ignore_token_ids = []):
        super().__init__()

        self.transformer = transformer

        # MLM 相关概率
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

        # 转换为文本 logits
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, seq, **kwargs):
        # 不要掩码 [pad] tokens，或者任何在被排除的 tokens 中的 tokens ([cls], [sep])
        # 也不要在随机选择的 tokens 中包含这些特殊 tokens
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # 将原本不需要掩码的 tokens 掩码为 padding tokens
        labels = seq.masked_fill(~mask, self.pad_token_id)

        # 使用 mask tokens 掩码 seq，掩码的概率为 `replace_prob`（以概率 1 - replace_prob 保持 tokens 不变）
        masked_seq = seq.clone().detach()

        # 如果随机 token 概率 > 0 用于 MLM
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = prob_mask_like(seq, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, seq.shape, device = seq.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

            # 从掩码中减去随机 token 概率掩码
            mask = mask & ~random_token_prob

        # [mask] seq
        replace_prob = prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id)

        # 获取生成器输出并计算 MLM 损失
        embedding = self.transformer(masked_seq, **kwargs)

        # 投影到 logits 并移除 CLS
        logits = self.to_logits(embedding)
        logits = logits[:, 1:]

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index = self.pad_token_id
        )

        return mlm_loss
```