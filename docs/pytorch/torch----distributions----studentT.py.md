# `.\pytorch\torch\distributions\studentT.py`

```py
# mypy: allow-untyped-defs
# 导入数学库
import math

# 导入 PyTorch 库
import torch
# 从 torch 模块中导入 inf 和 nan 常量
from torch import inf, nan
# 从 torch.distributions 模块中导入 Chi2 和 constraints
from torch.distributions import Chi2, constraints
# 从 torch.distributions.distribution 模块中导入 Distribution 类
from torch.distributions.distribution import Distribution
# 从 torch.distributions.utils 模块中导入 _standard_normal 和 broadcast_all 函数
from torch.distributions.utils import _standard_normal, broadcast_all

# 定义模块中公开的类名列表
__all__ = ["StudentT"]


# 定义 StudentT 类，继承自 Distribution 类
class StudentT(Distribution):
    r"""
    创建一个由自由度 :attr:`df`、均值 :attr:`loc` 和标度 :attr:`scale` 参数化的学生 t 分布。

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = StudentT(torch.tensor([2.0]))
        >>> m.sample()  # 自由度为2的学生 t 分布
        tensor([ 0.1046])

    Args:
        df (float or Tensor): 自由度
        loc (float or Tensor): 分布的均值
        scale (float or Tensor): 分布的标度
    """
    
    # 定义参数约束字典
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    
    # 定义分布的支持范围
    support = constraints.real
    
    # 声明具有 rsample 方法
    has_rsample = True

    # 定义均值属性
    @property
    def mean(self):
        # 复制均值张量，使用连续内存格式
        m = self.loc.clone(memory_format=torch.contiguous_format)
        # 根据自由度调整均值张量的部分值
        m[self.df <= 1] = nan
        return m

    # 定义众数属性
    @property
    def mode(self):
        return self.loc

    # 定义方差属性
    @property
    def variance(self):
        # 复制自由度张量，使用连续内存格式
        m = self.df.clone(memory_format=torch.contiguous_format)
        # 根据自由度调整方差张量的部分值
        m[self.df > 2] = (
            self.scale[self.df > 2].pow(2)
            * self.df[self.df > 2]
            / (self.df[self.df > 2] - 2)
        )
        m[(self.df <= 2) & (self.df > 1)] = inf
        m[self.df <= 1] = nan
        return m

    # 定义初始化方法
    def __init__(self, df, loc=0.0, scale=1.0, validate_args=None):
        # 广播输入参数，以匹配最大维度
        self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        # 创建自由度为 df 的 Chi2 分布对象
        self._chi2 = Chi2(self.df)
        # 获取批处理形状
        batch_shape = self.df.size()
        # 调用父类的初始化方法
        super().__init__(batch_shape, validate_args=validate_args)

    # 定义扩展方法
    def expand(self, batch_shape, _instance=None):
        # 获取检查后的实例
        new = self._get_checked_instance(StudentT, _instance)
        # 扩展自由度张量
        new.df = self.df.expand(batch_shape)
        # 扩展均值张量
        new.loc = self.loc.expand(batch_shape)
        # 扩展标度张量
        new.scale = self.scale.expand(batch_shape)
        # 扩展 Chi2 分布对象
        new._chi2 = self._chi2.expand(batch_shape)
        # 调用父类的初始化方法
        super(StudentT, new).__init__(batch_shape, validate_args=False)
        # 继承验证参数
        new._validate_args = self._validate_args
        return new
    # 从学生 t 分布中采样，返回指定形状的样本
    def rsample(self, sample_shape=torch.Size()):
        # 注意：这个实现与 scipy 的实现并不完全一致，详情见链接：
        # https://github.com/fritzo/notebooks/blob/master/debug-student-t.ipynb。使用 DoubleTensor 参数似乎有帮助。

        # X ~ Normal(0, 1)
        # Z ~ Chi2(df)
        # Y = X / sqrt(Z / df) ~ StudentT(df)

        # 根据指定的形状扩展采样形状
        shape = self._extended_shape(sample_shape)
        # 从标准正态分布 N(0, 1) 中采样
        X = _standard_normal(shape, dtype=self.df.dtype, device=self.df.device)
        # 从卡方分布 Chi2(df) 中采样
        Z = self._chi2.rsample(sample_shape)
        # 计算 Student t 分布的采样值
        Y = X * torch.rsqrt(Z / self.df)
        return self.loc + self.scale * Y

    # 计算给定值的对数概率密度函数值
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 标准化值
        y = (value - self.loc) / self.scale
        # 计算常数项 Z
        Z = (
            self.scale.log()
            + 0.5 * self.df.log()
            + 0.5 * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + 1.0))
        )
        # 计算对数概率密度函数值
        return -0.5 * (self.df + 1.0) * torch.log1p(y**2.0 / self.df) - Z

    # 计算分布的熵
    def entropy(self):
        # 计算 lbeta 函数值
        lbeta = (
            torch.lgamma(0.5 * self.df)
            + math.lgamma(0.5)
            - torch.lgamma(0.5 * (self.df + 1))
        )
        # 计算熵值
        return (
            self.scale.log()
            + 0.5
            * (self.df + 1)
            * (torch.digamma(0.5 * (self.df + 1)) - torch.digamma(0.5 * self.df))
            + 0.5 * self.df.log()
            + lbeta
        )
```