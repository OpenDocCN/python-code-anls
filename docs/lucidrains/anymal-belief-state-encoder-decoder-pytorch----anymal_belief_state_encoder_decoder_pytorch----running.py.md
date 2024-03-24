# `.\lucidrains\anymal-belief-state-encoder-decoder-pytorch\anymal_belief_state_encoder_decoder_pytorch\running.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 定义 RunningStats 类，继承自 nn.Module 类
class RunningStats(nn.Module):
    # 初始化方法，接受 shape 和 eps 两个参数
    def __init__(self, shape, eps = 1e-5):
        super().__init__()
        # 如果 shape 不是元组，则转换为元组
        shape = shape if isinstance(shape, tuple) else (shape,)

        # 初始化对象的 shape、eps 和 n 属性
        self.shape = shape
        self.eps = eps
        self.n = 0

        # 注册缓冲区 old_mean、new_mean、old_std、new_std，并设置为非持久化
        self.register_buffer('old_mean', torch.zeros(shape), persistent = False)
        self.register_buffer('new_mean', torch.zeros(shape), persistent = False)
        self.register_buffer('old_std', torch.zeros(shape), persistent = False)
        self.register_buffer('new_std', torch.zeros(shape), persistent = False)

    # 清空方法，将 n 属性重置为 0
    def clear(self):
        self.n = 0

    # 推送方法，接受输入 x，并更新均值和标准差
    def push(self, x):
        self.n += 1

        # 如果 n 为 1，则将 old_mean 和 new_mean 设置为 x 的数据，old_std 和 new_std 设置为 0
        if self.n == 1:
            self.old_mean.copy_(x.data)
            self.new_mean.copy_(x.data)
            self.old_std.zero_()
            self.new_std.zero_()
            return

        # 更新均值和标准差
        self.new_mean.copy_(self.old_mean + (x - self.old_mean) / self.n)
        self.new_std.copy_(self.old_std + (x - self.old_mean) * (x - self.new_mean))

        self.old_mean.copy_(self.new_mean)
        self.old_std.copy_(self.new_std)

    # 返回均值的方法
    def mean(self):
        return self.new_mean if self.n else torch.zeros_like(self.new_mean)

    # 返回方差的方法
    def variance(self):
        return (self.new_std / (self.n - 1)) if self.n > 1 else torch.zeros_like(self.new_std)

    # 返回标准差的倒数的方法
    def rstd(self):
        return torch.rsqrt(self.variance() + self.eps)

    # 归一化方法，接受输入 x，返回归一化后的结果
    def norm(self, x):
        return (x - self.mean()) * self.rstd()
```