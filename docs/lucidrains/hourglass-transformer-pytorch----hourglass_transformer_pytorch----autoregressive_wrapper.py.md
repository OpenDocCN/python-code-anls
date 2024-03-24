# `.\lucidrains\hourglass-transformer-pytorch\hourglass_transformer_pytorch\autoregressive_wrapper.py`

```py
import torch
from torch import nn
import torch.nn.functional as F

# helper function

# 检查值是否存在
def exists(val):
    return val is not None

# 装饰器函数，用于在模型评估时切换模型状态
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# top k filtering

# 根据阈值过滤 logits，保留前 k 个值
def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 自回归包装器类
class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, pad_value = 0):
        super().__init__()
        assert hasattr(net, 'max_seq_len'), 'your transformer class must have max_seq_len set to the maximum sequence length'

        self.pad_value = pad_value
        self.net = net
        self.max_seq_len = net.max_seq_len

    # 生成序列的方法
    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_thres = 0.9, **kwargs):
        b, t, device = *start_tokens.shape, start_tokens.device

        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]

            logits = self.net(x, **kwargs)[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_token = (out == eos_token)

                if is_eos_token.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]
        return out

    # 前向传播方法
    def forward(self, x, **kwargs):
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        logits = self.net(x_inp, **kwargs)
        return F.cross_entropy(logits.transpose(1, 2), x_labels, ignore_index = self.pad_value)
```