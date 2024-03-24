# `.\lucidrains\mlm-pytorch\mlm_pytorch\mlm_pytorch.py`

```
# 导入数学库
import math
# 从 functools 库中导入 reduce 函数
from functools import reduce

# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch 库中导入 functional 模块
import torch.nn.functional as F

# 辅助函数

# 根据概率生成掩码
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# 使用特定的标记生成掩码
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

# 根据概率获取掩码的子集
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

# 主类

class MLM(nn.Module):
    def __init__(
        self,
        transformer,
        mask_prob = 0.15,
        replace_prob = 0.9,
        num_tokens = None,
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

        # 标记 ID

        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, seq, **kwargs):

        # 不要对 [pad] 标记或任何在被排除的标记中的标记进行掩码，也不要在随机选择的标记中包含这些特殊标记

        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # 使用概率 `replace_prob` 对输入进行 [mask] 处理（以概率 1 - replace_prob 保持标记不变）

        masked_seq = seq.clone().detach()

        # 推导出要预测的标签

        labels = seq.masked_fill(~mask, self.pad_token_id)

        # 如果 MLM 中随机标记概率 > 0

        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = prob_mask_like(seq, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, seq.shape, device=seq.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

            # 从后续的 [mask] 中移除被随机替换的标记
            mask = mask & ~random_token_prob

        # [mask] 输入

        replace_prob = prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id)

        # 获取生成器输出并计算 MLM 损失

        logits = self.transformer(masked_seq, **kwargs)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index = self.pad_token_id
        )

        return mlm_loss
```