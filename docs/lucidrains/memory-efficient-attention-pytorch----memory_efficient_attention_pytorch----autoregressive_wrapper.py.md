# `.\lucidrains\memory-efficient-attention-pytorch\memory_efficient_attention_pytorch\autoregressive_wrapper.py`

```
import torch
from torch import nn
import torch.nn.functional as F

# helper function

# 检查值是否存在的辅助函数
def exists(val):
    return val is not None

# 评估装饰器函数
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        # 保存模型原始训练状态
        was_training = model.training
        # 将模型设置为评估模式
        model.eval()
        # 调用传入的函数
        out = fn(model, *args, **kwargs)
        # 恢复模型原始训练状态
        model.train(was_training)
        return out
    return inner

# top k filtering

# 根据阈值过滤 logits 中的 top k 值
def top_k(logits, thres = 0.9):
    # 计算 top k 的数量
    k = int((1 - thres) * logits.shape[-1])
    # 获取 top k 的值和索引
    val, ind = torch.topk(logits, k)
    # 创建与 logits 相同形状的全为负无穷的张量
    probs = torch.full_like(logits, float('-inf'))
    # 根据索引将 top k 的值填充到 probs 中
    probs.scatter_(1, ind, val)
    return probs

# 自回归包装器类
class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.net = net
        self.max_seq_len = net.max_seq_len

    # 生成序列的方法
    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_thres = 0.9, **kwargs):
        # 获取起始 tokens 的形状和设备信息
        b, t, device = *start_tokens.shape, start_tokens.device

        out = start_tokens

        for _ in range(seq_len):
            # 获取最后 self.max_seq_len 个 token
            x = out[:, -self.max_seq_len:]

            # 获取模型预测的 logits
            logits = self.net(x, **kwargs)[:, -1, :]

            # 过滤 logits 中的 top k 值
            filtered_logits = top_k(logits, thres = filter_thres)
            # 计算 softmax 温度调节后的概率
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            # 从概率分布中采样一个 token
            sample = torch.multinomial(probs, 1)

            # 将采样的 token 添加到输出序列中
            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                # 检查是否存在 eos_token
                is_eos_token = (out == eos_token)

                if is_eos_token.any(dim = -1).all():
                    # 如果所有序列中都存在 eos_token，则停止生成
                    # 创建一个向右移动一位�� eos_token mask
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    # 创建一个 mask，标记 eos_token 后的所有位置
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    # 将 mask 标记的位置填充为 pad_value
                    out = out.masked_fill(mask, self.pad_value)
                    break

        # 去除起始 tokens，返回生成的序列
        out = out[:, t:]
        return out

    # 前向传播方法
    def forward(self, x, **kwargs):
        # 将输入拆分为输入和标签
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        return self.net(x_inp, labels = x_labels, **kwargs)
```