# `.\pytorch\torch\distributions\dirichlet.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的 Function 类
from torch.autograd import Function
# 导入 PyTorch 中的自动微分相关功能
from torch.autograd.function import once_differentiable
# 导入 PyTorch 中的分布约束
from torch.distributions import constraints
# 导入 PyTorch 中指数族分布的基类
from torch.distributions.exp_family import ExponentialFamily

# 定义公开的辅助函数，用于测试
__all__ = ["Dirichlet"]


# 这个函数为测试目的而暴露出来
def _Dirichlet_backward(x, concentration, grad_output):
    # 计算浓度的总和
    total = concentration.sum(-1, True).expand_as(concentration)
    # 计算 Dirichlet 分布的梯度
    grad = torch._dirichlet_grad(x, concentration, total)
    return grad * (grad_output - (x * grad_output).sum(-1, True))


class _Dirichlet(Function):
    @staticmethod
    def forward(ctx, concentration):
        # 从浓度参数中采样得到 Dirichlet 分布的样本
        x = torch._sample_dirichlet(concentration)
        # 保存变量以便在反向传播时使用
        ctx.save_for_backward(x, concentration)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # 恢复保存的张量
        x, concentration = ctx.saved_tensors
        # 调用内部函数计算梯度
        return _Dirichlet_backward(x, concentration, grad_output)


class Dirichlet(ExponentialFamily):
    r"""
    创建由浓度参数 :attr:`concentration` 参数化的 Dirichlet 分布。

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentration [0.5, 0.5]
        tensor([ 0.1046,  0.8954])

    Args:
        concentration (Tensor): 分布的浓度参数
            （通常称为 alpha）
    """
    # 约束参数的定义
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1)
    }
    # 支持的约束：单位单纯形
    support = constraints.simplex
    # 具有 rsample 方法
    has_rsample = True

    def __init__(self, concentration, validate_args=None):
        # 确保浓度参数至少是一维的
        if concentration.dim() < 1:
            raise ValueError(
                "`concentration` 参数必须至少是一维的。"
            )
        self.concentration = concentration
        # 获取批次形状和事件形状
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # 创建新的 Dirichlet 分布实例
        new = self._get_checked_instance(Dirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        # 扩展浓度参数
        new.concentration = self.concentration.expand(batch_shape + self.event_shape)
        super(Dirichlet, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=()):
        # 计算扩展形状
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        # 调用自定义的 _Dirichlet 类来进行采样
        return _Dirichlet.apply(concentration)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 计算对数概率密度函数值
        return (
            torch.xlogy(self.concentration - 1.0, value).sum(-1)
            + torch.lgamma(self.concentration.sum(-1))
            - torch.lgamma(self.concentration).sum(-1)
        )

    @property
    # 计算均值，返回浓度除以浓度总和
    def mean(self):
        return self.concentration / self.concentration.sum(-1, True)

    # 计算众数，对浓度进行调整并计算相对频率
    @property
    def mode(self):
        # 将浓度减1并截断小于0的部分，得到调整后的浓度
        concentrationm1 = (self.concentration - 1).clamp(min=0.0)
        # 计算调整后的浓度相对频率
        mode = concentrationm1 / concentrationm1.sum(-1, True)
        # 创建一个布尔掩码，标识所有浓度小于1的情况
        mask = (self.concentration < 1).all(axis=-1)
        # 对于浓度小于1的情况，用最大的相对频率值填充
        mode[mask] = torch.nn.functional.one_hot(
            mode[mask].argmax(axis=-1), concentrationm1.shape[-1]
        ).to(mode)
        return mode

    # 计算方差
    @property
    def variance(self):
        # 计算浓度总和
        con0 = self.concentration.sum(-1, True)
        # 返回方差的计算结果
        return (
            self.concentration
            * (con0 - self.concentration)
            / (con0.pow(2) * (con0 + 1))
        )

    # 计算熵
    def entropy(self):
        # 获取浓度张量的最后一个维度的大小
        k = self.concentration.size(-1)
        # 计算浓度总和
        a0 = self.concentration.sum(-1)
        # 返回熵的计算结果
        return (
            torch.lgamma(self.concentration).sum(-1)
            - torch.lgamma(a0)
            - (k - a0) * torch.digamma(a0)
            - ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1)
        )

    # 返回自然参数，即浓度本身
    @property
    def _natural_params(self):
        return (self.concentration,)

    # 计算对数归一化常数
    def _log_normalizer(self, x):
        # 返回对数伽玛函数的和减去对数伽玛函数
        return x.lgamma().sum(-1) - torch.lgamma(x.sum(-1))
```