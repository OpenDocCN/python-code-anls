# `.\lucidrains\Mega-pytorch\mega_pytorch\autoregressive_wrapper.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 定义一个辅助函数 exists，用于检查值是否存在
def exists(val):
    return val is not None

# 定义一个装饰器 eval_decorator，用于在模型评估时切换模型状态
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 定义一个函数 top_k，用于对 logits 进行 top-k 过滤
def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 定义一个类 AutoregressiveWrapper，用于包装模型
class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.net = net

    # 生成函数，用于生成序列
    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, temperature = 1., filter_thres = 0.9, **kwargs):
        b, t, device = *start_tokens.shape, start_tokens.device

        out = start_tokens

        for _ in range(seq_len):
            logits = self.net(out, **kwargs)[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

        out = out[:, t:]
        return out

    # 前向传播函数，用于计算损失
    def forward(self, x, **kwargs):
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        logits = self.net(x_inp, **kwargs)
        return F.cross_entropy(rearrange(logits, 'b c n -> b n c'), x_labels)
```