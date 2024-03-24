# `.\lucidrains\complex-valued-transformer\complex_valued_transformer\autoregressive_wrapper.py`

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

# 返回输入的函数
def identity(t):
    return t

# 评估装饰器函数
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        # 保存模型当前是否为训练状态
        was_training = model.training
        # 将模型设置为评估状态
        model.eval()
        # 调用传入的函数
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
    # 获取 top k 值及其索引
    val, ind = torch.topk(logits, k)
    # 创建与 logits 相同形状的张量，填充为负的最大值
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    # 根据索引将 top k 值填充到 probs 中
    probs.scatter_(1, ind, val)
    return probs

# 自回归包装器类
class AutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,        
        seq_len,
        pad_value = 0,
        logits_fn = identity
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.net = net
        self.logits_fn = logits_fn

    # 生成函数，用于生成序列
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompt,
        seq_len,
        temperature = 1.0,
        filter_thres = 0.9,
        **kwargs
    ):
        # 获取 prompt 的形状、设备信息
        b, t, device = *prompt.shape, prompt.device

        out = prompt

        # 生成序列
        for _ in range(seq_len):
            # 获取最后 seq_len 长度的输出
            logits = self.net(out[:, -self.seq_len:], **kwargs)[:, -1]
            logits = self.logits_fn(logits)

            # 过滤 logits 中的 top k 值
            filtered_logits = top_k(logits, thres = filter_thres)
            # 计算 softmax 温度调节后的概率
            probs = F.softmax(filtered_logits / temperature, dim = -1)

            # 从概率分布中采样一个值
            sample = torch.multinomial(probs, 1)
            # 将采样值拼接到输出序列中
            out = torch.cat((out, sample), dim = -1)

        return out[:, t:]

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 获取输入 x 的特征和标签
        x, labels = x[:, :-1], x[:, 1:]
        # 获取模型输出的 logits
        logits = self.net(x, **kwargs)
        # 重排 logits 的维度
        logits = rearrange(self.logits_fn(logits), "b c n -> b n c")
        # 计算交叉熵损失
        return F.cross_entropy(logits, labels)
```