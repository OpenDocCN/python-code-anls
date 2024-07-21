# `.\pytorch\torch\distributions\lkj_cholesky.py`

```py
"""
This closely follows the implementation in NumPyro (https://github.com/pyro-ppl/numpyro).

Original copyright notice:

# Copyright: Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
"""

import math  # 导入math库，用于数学运算

import torch  # 导入PyTorch库
from torch.distributions import Beta, constraints  # 从torch.distributions模块导入Beta分布和约束
from torch.distributions.distribution import Distribution  # 导入分布基类Distribution
from torch.distributions.utils import broadcast_all  # 导入broadcast_all函数

__all__ = ["LKJCholesky"]  # 定义公开的模块成员列表，只包含LKJCholesky类

class LKJCholesky(Distribution):
    r"""
    LKJ distribution for lower Cholesky factor of correlation matrices.
    The distribution is controlled by ``concentration`` parameter :math:`\eta`
    to make the probability of the correlation matrix :math:`M` generated from
    a Cholesky factor proportional to :math:`\det(M)^{\eta - 1}`. Because of that,
    when ``concentration == 1``, we have a uniform distribution over Cholesky
    factors of correlation matrices::

        L ~ LKJCholesky(dim, concentration)
        X = L @ L' ~ LKJCorr(dim, concentration)

    Note that this distribution samples the
    Cholesky factor of correlation matrices and not the correlation matrices
    themselves and thereby differs slightly from the derivations in [1] for
    the `LKJCorr` distribution. For sampling, this uses the Onion method from
    [1] Section 3.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> l = LKJCholesky(3, 0.5)
        >>> l.sample()  # l @ l.T is a sample of a correlation 3x3 matrix
        tensor([[ 1.0000,  0.0000,  0.0000],
                [ 0.3516,  0.9361,  0.0000],
                [-0.1899,  0.4748,  0.8593]])

    Args:
        dimension (dim): dimension of the matrices
        concentration (float or Tensor): concentration/shape parameter of the
            distribution (often referred to as eta)

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method` (2009),
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    Journal of Multivariate Analysis. 100. 10.1016/j.jmva.2009.04.008
    """
    arg_constraints = {"concentration": constraints.positive}  # 参数约束，确保concentration为正数
    support = constraints.corr_cholesky  # 支持的值域，要求是相关性矩阵的Cholesky因子
    def __init__(self, dim, concentration=1.0, validate_args=None):
        # 如果维度小于2，抛出数值错误异常，要求维度必须大于等于2
        if dim < 2:
            raise ValueError(
                f"Expected dim to be an integer greater than or equal to 2. Found dim={dim}."
            )
        # 设置对象的维度属性
        self.dim = dim
        # 广播操作，确保浓度参数是可以广播的
        (self.concentration,) = broadcast_all(concentration)
        # 计算批次形状和事件形状
        batch_shape = self.concentration.size()
        event_shape = torch.Size((dim, dim))
        # 在[1]的第3.2节中用于从beta分布中绘制向量化样本
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        # 创建偏移量张量
        offset = torch.arange(
            self.dim - 1,
            dtype=self.concentration.dtype,
            device=self.concentration.device,
        )
        offset = torch.cat([offset.new_zeros((1,)), offset])
        # 计算beta分布的参数
        beta_conc1 = offset + 0.5
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
        # 初始化Beta分布对象
        self._beta = Beta(beta_conc1, beta_conc0)
        # 调用父类初始化方法
        super().__init__(batch_shape, event_shape, validate_args)

    def expand(self, batch_shape, _instance=None):
        # 获取已检查的新实例
        new = self._get_checked_instance(LKJCholesky, _instance)
        # 转换为torch.Size形式的批次形状
        batch_shape = torch.Size(batch_shape)
        # 设置新实例的维度属性
        new.dim = self.dim
        # 扩展浓度参数
        new.concentration = self.concentration.expand(batch_shape)
        # 扩展_beta对象
        new._beta = self._beta.expand(batch_shape + (self.dim,))
        # 调用父类初始化方法
        super(LKJCholesky, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        # 继承验证参数设置
        new._validate_args = self._validate_args
        # 返回新实例
        return new

    def sample(self, sample_shape=torch.Size()):
        # 使用Onion方法进行采样，与[1]第3.2节有些不同：
        # - 这里向量化了for循环，并且适用于异构eta。
        # - 相同的算法推广到n=1。
        # - 由于我们只需生成相关矩阵的cholesky因子，所以该过程被简化了。
        #   因此，我们只需生成`w`。
        # 从Beta分布中采样y，并扩展维度以便进行后续计算
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        # 从标准正态分布中采样u_normal，并生成下三角矩阵
        u_normal = torch.randn(
            self._extended_shape(sample_shape), dtype=y.dtype, device=y.device
        ).tril(-1)
        # 将u_normal转换为单位超球面上的向量
        u_hypersphere = u_normal / u_normal.norm(dim=-1, keepdim=True)
        # 替换第一行中的NaN值
        u_hypersphere[..., 0, :].fill_(0.0)
        # 计算w矩阵
        w = torch.sqrt(y) * u_hypersphere
        # 填充对角元素，为数值稳定性进行截断
        eps = torch.finfo(w.dtype).tiny
        diag_elems = torch.clamp(1 - torch.sum(w**2, dim=-1), min=eps).sqrt()
        w += torch.diag_embed(diag_elems)
        # 返回采样结果
        return w
    def log_prob(self, value):
        # 如果设置了验证参数，则验证输入的样本值
        if self._validate_args:
            self._validate_sample(value)
        
        # 提取对角元素，这些元素构成了Cholesky因子的对角线元素（不包括主对角线）
        diag_elems = value.diagonal(dim1=-1, dim2=-2)[..., 1:]
        
        # 计算每个对角元素的指数（order_i）
        order = torch.arange(2, self.dim + 1, device=self.concentration.device)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order
        
        # 计算未归一化的对数概率密度
        unnormalized_log_pdf = torch.sum(order * diag_elems.log(), dim=-1)
        
        # 计算归一化常数（见参考文献[1]第1999页）
        dm1 = self.dim - 1
        alpha = self.concentration + 0.5 * dm1
        denominator = torch.lgamma(alpha) * dm1
        numerator = torch.mvlgamma(alpha - 0.5, dm1)
        
        # pi常数在[1]中是 D * (D - 1) / 4 * log(pi)
        # 在multigammaln中，pi常数是 (D - 1) * (D - 2) / 4 * log(pi)
        # 因此，我们需要添加 pi常数 = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * dm1 * math.log(math.pi)
        normalize_term = pi_constant + numerator - denominator
        
        # 返回未归一化的对数概率密度减去归一化常数
        return unnormalized_log_pdf - normalize_term
```