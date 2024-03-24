# `.\lucidrains\hamburger-pytorch\hamburger_pytorch\hamburger_pytorch.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum 模块
from torch import nn, einsum
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 contextlib 模块中导入 contextmanager 上下文管理器
from contextlib import contextmanager
# 从 einops 模块中导入 repeat, rearrange 函数
from einops import repeat, rearrange

# 辅助函数

# 定义一个空上下文管理器
@contextmanager
def null_context():
    yield

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 类

# 定义 NMF 类，继承自 nn.Module
class NMF(nn.Module):
    def __init__(
        self,
        dim,
        n,
        ratio = 8,
        K = 6,
        eps = 2e-8
    ):
        super().__init__()
        r = dim // ratio

        # 初始化 D 和 C 为随机数
        D = torch.zeros(dim, r).uniform_(0, 1)
        C = torch.zeros(r, n).uniform_(0, 1)

        self.K = K
        self.D = nn.Parameter(D)
        self.C = nn.Parameter(C)

        self.eps = eps

    def forward(self, x):
        b, D, C, eps = x.shape[0], self.D, self.C, self.eps

        # 将输入 x 转为非负数
        x = F.relu(x)

        # 将 D 和 C 扩展为与输入 x 相同的 batch 维度
        D = repeat(D, 'd r -> b d r', b = b)
        C = repeat(C, 'r n -> b r n', b = b)

        # 转置函数
        t = lambda tensor: rearrange(tensor, 'b i j -> b j i')

        for k in reversed(range(self.K)):
            # 只在最后一步计算梯度，根据 'One-step Gradient' 提议
            context = null_context if k == 0 else torch.no_grad
            with context():
                C_new = C * ((t(D) @ x) / ((t(D) @ D @ C) + eps))
                D_new = D * ((x @ t(C)) / ((D @ C @ t(C)) + eps))
                C, D = C_new, D_new

        return D @ C

# 定义 Hamburger 类，继承自 nn.Module
class Hamburger(nn.Module):
    def __init__(
        self,
        *,
        dim,
        n,
        inner_dim = None,
        ratio = 8,
        K = 6
    ):
        super().__init__()
        inner_dim = default(inner_dim, dim)

        # 定义 lower_bread 为一维卷积层
        self.lower_bread = nn.Conv1d(dim, inner_dim, 1, bias = False)
        # 定义 ham 为 NMF 类的实例
        self.ham = NMF(inner_dim, n, ratio = ratio, K = K)
        # 定义 upper_bread 为一维卷积层
        self.upper_bread = nn.Conv1d(inner_dim, dim, 1, bias = False)

    def forward(self, x):
        shape = x.shape
        # 将输入 x 展平为二维
        x = x.flatten(2)

        x = self.lower_bread(x)
        x = self.ham(x)
        x = self.upper_bread(x)
        # 将 x 重新 reshape 成原始形状
        return x.reshape(shape)
```