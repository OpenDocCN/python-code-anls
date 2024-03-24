# `.\lucidrains\FLASH-pytorch\flash_pytorch\autoregressive_wrapper.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 辅助函数

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 评估装饰器函数
def eval_decorator(fn):
    # 内部函数
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

# top k 过滤

# 根据阈值过滤 logits 中的 top k 值
def top_k(logits, thres = 0.9):
    # 计算 top k 的数量
    k = int((1 - thres) * logits.shape[-1])
    # 获取 top k 的值和索引
    val, ind = torch.topk(logits, k)
    # 创建与 logits 相同形状的张量，填充为负无穷
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

    # 无需梯度的装饰器
    @torch.no_grad()
    @eval_decorator
    # 生成序列的方法
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_thres = 0.9, **kwargs):
        # 获取起始 tokens 的形状和设备信息
        b, t, device = *start_tokens.shape, start_tokens.device

        # 初始化输出为起始 tokens
        out = start_tokens

        # 循环生成序列
        for _ in range(seq_len):
            # 获取模型输出 logits
            logits = self.net(out, **kwargs)[:, -1, :]

            # 过滤 logits 中的 top k 值
            filtered_logits = top_k(logits, thres = filter_thres)
            # 计算概率分布
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            # 从概率分布中采样一个值作为下一个 token
            sample = torch.multinomial(probs, 1)

            # 将采样值拼接到输出序列中
            out = torch.cat((out, sample), dim=-1)

            # 如果存在 eos_token
            if exists(eos_token):
                # 判断是否出现 eos_token
                is_eos_token = (out == eos_token)

                if is_eos_token.any(dim = -1).all():
                    # 创建一个 mask，用于标记 eos_token 后的内容
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    # 将 mask 标记的内容替换为 pad_value
                    out = out.masked_fill(mask, self.pad_value)
                    break

        # 去除起始 tokens，返回生成的序列
        out = out[:, t:]
        return out

    # 前向传播方法
    def forward(self, x, **kwargs):
        # 分离输入和标签
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        # 获取模型输出 logits
        logits = self.net(x_inp, **kwargs)
        # 计算交叉熵损失
        return F.cross_entropy(rearrange(logits, 'b c n -> b n c'), x_labels)
```