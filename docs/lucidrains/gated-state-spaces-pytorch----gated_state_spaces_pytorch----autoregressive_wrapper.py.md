# `.\lucidrains\gated-state-spaces-pytorch\gated_state_spaces_pytorch\autoregressive_wrapper.py`

```
# 导入 torch 库
import torch
# 导入 torch.nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 einops 库中导入 rearrange 函数
from einops import rearrange
# 从 torch 库中导入 nn 模块
from torch import nn

# 定义一个辅助函数，用于检查值是否存在
def exists(val):
    return val is not None

# 定义一个装饰器函数，用于在模型评估时切换模型状态
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 定义一个函数，用于对 logits 进行 top k 过滤
def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 定义一个自回归封装器类
class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, pad_value=0, max_seq_len=4096):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.net = net

    # 生成函数，用于生成序列
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        temperature=1.0,
        filter_thres=0.9,
        **kwargs
    ):
        b, n, device = *start_tokens.shape, start_tokens.device

        out = start_tokens

        for _ in range(seq_len):
            logits = self.net(
                out[:, -self.max_seq_len:],
                **kwargs
            )[:, -1]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_token = out == eos_token

                if is_eos_token.any(dim=-1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        return out[:, n:]

    # 前向传播函数，用于模型训练
    def forward(self, x, **kwargs):
        inp, labels = x[:, :-1], x[:, 1:]
        return self.net(inp, labels=labels, **kwargs)
```