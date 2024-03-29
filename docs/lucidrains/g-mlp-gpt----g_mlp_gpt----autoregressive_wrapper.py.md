# `.\lucidrains\g-mlp-gpt\g_mlp_gpt\autoregressive_wrapper.py`

```py
import torch
from torch import nn
import torch.nn.functional as F

# 定义一个装饰器函数，用于在模型评估时切换为eval模式
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 定义一个函数用于对logits进行top k过滤
def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 定义一个包装类，用于自回归模型
class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = -100, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.seq_len

    # 生成函数，用于生成序列
    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        device = start_tokens.device
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]

            logits = self.net(x, **kwargs)[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        return out

    # 前向传播函数，用于计算损失
    def forward(self, x, **kwargs):
        xi, xo = x[:, :-1], x[:, 1:]
        out = self.net(xi, **kwargs)
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index)
        return loss
```