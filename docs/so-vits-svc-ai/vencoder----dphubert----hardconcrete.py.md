# `so-vits-svc\vencoder\dphubert\hardconcrete.py`

```
# 导入必要的库
import math
import torch
import torch.nn as nn

# 定义 HardConcrete 类，用于创建大小为 N 的掩码，以便执行 L0 正则化
class HardConcrete(nn.Module):
    """A HarcConcrete module.
    Use this module to create a mask of size N, which you can
    then use to perform L0 regularization.

    To obtain a mask, simply run a forward pass through the module
    with no input data. The mask is sampled in training mode, and
    fixed during evaluation mode, e.g.:

    >>> module = HardConcrete(n_in=100)
    >>> mask = module()
    >>> norm = module.l0_norm()
    """

    # 初始化方法
    def __init__(
        self,
        n_in: int,
        init_mean: float = 0.5,
        init_std: float = 0.01,
        temperature: float = 2/3,     # from CoFi
        stretch: float = 0.1,
        eps: float = 1e-6
    # 初始化 HardConcrete 模块
    def __init__(self, n_in: int, init_mean: float = 0.5, init_std: float = 0.01, temperature: float = 1.0, stretch: float = 0.1) -> None:
        # 调用父类的初始化方法
        super().__init__()

        # 设置属性值
        self.n_in = n_in
        self.limit_l = -stretch
        self.limit_r = 1.0 + stretch
        self.log_alpha = nn.Parameter(torch.zeros(n_in))
        self.beta = temperature
        self.init_mean = init_mean
        self.init_std = init_std
        self.bias = -self.beta * math.log(-self.limit_l / self.limit_r)

        # 设置默认值
        self.eps = eps
        self.compiled_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        """重置模块的参数"""
        self.compiled_mask = None
        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        self.log_alpha.data.normal_(mean, self.init_std)

    def l0_norm(self) -> torch.Tensor:
        """计算该掩码的期望 L0 范数
        返回
        -------
        torch.Tensor
            期望的 L0 范数
        """
        return (self.log_alpha + self.bias).sigmoid().sum()
    def forward(self) -> torch.Tensor:
        """Sample a hard concrete mask.
        Returns
        -------
        torch.Tensor
            The sampled binary mask
        """
        if self.training:
            # 如果处于训练状态，重置编译的掩码
            self.compiled_mask = None
            # 动态采样掩码
            u = self.log_alpha.new(self.n_in).uniform_(self.eps, 1 - self.eps)
            s = torch.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            mask = s.clamp(min=0., max=1.)

        else:
            # 如果不处于训练状态，如果未缓存编译的掩码，则编译新的掩码
            if self.compiled_mask is None:
                # 获取期望的稀疏度
                expected_num_zeros = self.n_in - self.l0_norm().item()
                num_zeros = round(expected_num_zeros)
                # 近似每个掩码变量 z 的期望值；
                # 使用经验验证的魔术数字 0.8
                soft_mask = torch.sigmoid(self.log_alpha / self.beta * 0.8)
                # 剪枝小值设为 0
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
                self.compiled_mask = soft_mask
            mask = self.compiled_mask

        return mask

    def extra_repr(self) -> str:
        return str(self.n_in)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())
```