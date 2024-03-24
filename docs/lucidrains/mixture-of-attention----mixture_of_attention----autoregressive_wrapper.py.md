# `.\lucidrains\mixture-of-attention\mixture_of_attention\autoregressive_wrapper.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 评估装饰器函数
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        # 保存模型当前是否为训练状态
        was_training = model.training
        # 将模型设置为评估状态
        model.eval()
        # 调用传入的函数，并传入模型、参数和关键字参数
        out = fn(model, *args, **kwargs)
        # 恢复模型之前的训练状态
        model.train(was_training)
        return out
    return inner

# top k 过滤

# 根据阈值过滤 logits 中的 top k 值
def top_k(logits, thres = 0.9):
    # 计算需要保留的 top k 值的数量
    k = int((1 - thres) * logits.shape[-1])
    # 获取 top k 值和对应的索引
    val, ind = torch.topk(logits, k)
    # 创建一个与 logits 相同形状的张量，填充为负的最大值
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    # 根据索引将 top k 值填充到 probs 中
    probs.scatter_(1, ind, val)
    return probs

# 自回归包装器类
class AutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,        
        pad_value = 0
    ):
        super().__init__()
        # 初始化属性
        self.seq_len = net.seq_len
        self.pad_value = pad_value
        self.net = net

    # 生成函数装饰器，用于生成序列
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompt,
        seq_len,
        temperature=1.0,
        filter_thres=0.9,
        **kwargs
    ):
        # 获取 prompt 的形状和设备信息
        b, t, device = *prompt.shape, prompt.device

        out = prompt

        # 生成序列
        for _ in range(seq_len):
            # 获取最后 self.seq_len 长度的序列，并传入网络获取 logits
            logits = self.net(out[:, -self.seq_len:], **kwargs)[:, -1]

            # 对 logits 进行 top k 过滤
            filtered_logits = top_k(logits, thres = filter_thres)
            # 计算概率分布
            probs = F.softmax(filtered_logits / temperature, dim = -1)

            # 从概率分布中采样一个值
            sample = torch.multinomial(probs, 1)
            # 将采样值拼接到输出序列中
            out = torch.cat((out, sample), dim = -1)

        # 去除前面的 prompt 部分，返回生成的序列
        out = out[:, t:]
        return out

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 获取输入 x 和标签 labels
        x, labels = x[:, :-1], x[:, 1:]
        # 将输入传入网络获取 logits
        logits = self.net(x, **kwargs)
        # 重新排列 logits 的维度
        logits = rearrange(logits, "b c n -> b n c")
        # 计算交叉熵损失
        return F.cross_entropy(logits, labels)
```