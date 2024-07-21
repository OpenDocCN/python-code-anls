# `.\pytorch\torch\distributions\kl.py`

```
# 设置 mypy 参数允许未声明的函数
mypy: allow-untyped-defs
# 导入数学模块
import math
# 导入警告模块
import warnings
# 导入 functools 模块的 total_ordering 装饰器
from functools import total_ordering
# 导入类型提示模块
from typing import Callable, Dict, Tuple, Type

# 导入 PyTorch 库
import torch
# 导入 torch 的 inf（无穷大）常量
from torch import inf

# 导入各种分布类
from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .gamma import Gamma
from .geometric import Geometric
from .gumbel import Gumbel
from .half_normal import HalfNormal
from .independent import Independent
from .laplace import Laplace
from .lowrank_multivariate_normal import (
    _batch_lowrank_logdet,
    _batch_lowrank_mahalanobis,
    LowRankMultivariateNormal,
)
from .multivariate_normal import _batch_mahalanobis, MultivariateNormal
from .normal import Normal
from .one_hot_categorical import OneHotCategorical
from .pareto import Pareto
from .poisson import Poisson
from .transformed_distribution import TransformedDistribution
from .uniform import Uniform
from .utils import _sum_rightmost, euler_constant as _euler_gamma

# _KL_REGISTRY: 用于存储 (type_p, type_q) 到函数的映射，用作 kl_divergence 的来源
_KL_REGISTRY: Dict[
    Tuple[Type, Type], Callable
] = {}
# _KL_MEMOIZE: 用于存储经过缓存的 (type_p, type_q) 到函数的映射，优化多个具体 (type, type) 对应的查找
_KL_MEMOIZE: Dict[
    Tuple[Type, Type], Callable
] = {}

# __all__: 模块公开的接口列表，包括 "register_kl" 和 "kl_divergence"
__all__ = ["register_kl", "kl_divergence"]

def register_kl(type_p, type_q):
    """
    装饰器，用于向 kl_divergence 注册一个成对函数。

    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    查找返回最具体的（type,type）匹配，按子类排序。如果匹配不明确，会引发 `RuntimeWarning`。
    例如，解决不明确的情况::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    应该注册第三个最具体的实现，例如::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # 打破决策

    Args:
        type_p (type): 一个 :class:`~torch.distributions.Distribution` 的子类。
        type_q (type): 一个 :class:`~torch.distributions.Distribution` 的子类。
    """
    # 确保 type_p 是 Distribution 的子类
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError(
            f"Expected type_p to be a Distribution subclass but got {type_p}"
        )
    # 确保 type_q 是 Distribution 的子类
    if not isinstance(type_q, type) and issubclass(type_q, Distribution):
        raise TypeError(
            f"Expected type_q to be a Distribution subclass but got {type_q}"
        )

    def decorator(fun):
        # 将函数注册到 _KL_REGISTRY 中
        _KL_REGISTRY[type_p, type_q] = fun
        # 清空 _KL_MEMOIZE 缓存，因为查找顺序可能已更改
        _KL_MEMOIZE.clear()
        return fun

    return decorator
@total_ordering
class _Match:
    __slots__ = ["types"]

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True

# 根据单继承假设，查找最具体的近似匹配
def _dispatch_kl(type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [
        (super_p, super_q)
        for super_p, super_q in _KL_REGISTRY
        if issubclass(type_p, super_p) and issubclass(type_q, super_q)
    ]
    if not matches:
        return NotImplemented
    
    # 检查左右字典序是否一致
    # mypy 无法识别 _Match 实现了 __lt__
    # 参考：https://github.com/python/typing/issues/760#issuecomment-710670503
    left_p, left_q = min(_Match(*m) for m in matches).types  # type: ignore[type-var]
    right_q, right_p = min(_Match(*reversed(m)) for m in matches).types  # type: ignore[type-var]
    
    left_fun = _KL_REGISTRY[left_p, left_q]
    right_fun = _KL_REGISTRY[right_p, right_q]
    
    # 如果左右函数不一致，发出警告
    if left_fun is not right_fun:
        warnings.warn(
            f"Ambiguous kl_divergence({type_p.__name__}, {type_q.__name__}). "
            f"Please register_kl({left_p.__name__}, {right_q.__name__})",
            RuntimeWarning,
        )
    
    return left_fun


# 辅助函数，用于返回与输入张量同形状的无穷 KL 散度
def _infinite_like(tensor):
    """
    Helper function for obtaining infinite KL Divergence throughout
    """
    return torch.full_like(tensor, inf)


# 辅助函数，用于计算 x log x
def _x_log_x(tensor):
    """
    Utility function for calculating x log x
    """
    return tensor * tensor.log()


# 辅助函数，用于计算带有任意尾部批次维度的 XX^{T} 的迹
def _batch_trace_XXT(bmat):
    """
    Utility function for calculating the trace of XX^{T} with X having arbitrary trailing batch dimensions
    """
    n = bmat.size(-1)
    m = bmat.size(-2)
    flat_trace = bmat.reshape(-1, m * n).pow(2).sum(-1)
    return flat_trace.reshape(bmat.shape[:-2])


# 计算两个分布之间的 Kullback-Leibler 散度 KL(p \| q)
def kl_divergence(p: Distribution, q: Distribution) -> torch.Tensor:
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    try:
        fun = _KL_MEMOIZE[type(p), type(q)]
    except KeyError:
        fun = _dispatch_kl(type(p), type(q))
        _KL_MEMOIZE[type(p), type(q)] = fun
    # 如果fun是NotImplemented，则抛出NotImplementedError异常
    if fun is NotImplemented:
        raise NotImplementedError(
            f"No KL(p || q) is implemented for p type {p.__class__.__name__} and q type {q.__class__.__name__}"
        )
    # 返回fun函数应用于p和q的结果
    return fun(p, q)
# 注册 KL 散度函数用于 Bernoulli 分布与 Bernoulli 分布之间的计算
@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(p, q):
    # 计算 KL 散度的第一部分
    t1 = p.probs * (
        torch.nn.functional.softplus(-q.logits)
        - torch.nn.functional.softplus(-p.logits)
    )
    # 处理 q.probs 为 0 的情况，设为无穷大，确保 KL 散度的计算正确性
    t1[q.probs == 0] = inf
    # 处理 p.probs 为 0 的情况，对应 KL 散度中的特殊情形
    t1[p.probs == 0] = 0
    # 计算 KL 散度的第二部分
    t2 = (1 - p.probs) * (
        torch.nn.functional.softplus(q.logits) - torch.nn.functional.softplus(p.logits)
    )
    # 处理 q.probs 为 1 的情况，设为无穷大，确保 KL 散度的计算正确性
    t2[q.probs == 1] = inf
    # 处理 p.probs 为 1 的情况，对应 KL 散度中的特殊情形
    t2[p.probs == 1] = 0
    # 返回计算得到的 KL 散度
    return t1 + t2


# 注册 KL 散度函数用于 Beta 分布与 Beta 分布之间的计算
@register_kl(Beta, Beta)
def _kl_beta_beta(p, q):
    # 计算 KL 散度的第一部分
    sum_params_p = p.concentration1 + p.concentration0
    sum_params_q = q.concentration1 + q.concentration0
    t1 = q.concentration1.lgamma() + q.concentration0.lgamma() + (sum_params_p).lgamma()
    # 计算 KL 散度的第二部分
    t2 = p.concentration1.lgamma() + p.concentration0.lgamma() + (sum_params_q).lgamma()
    # 计算 KL 散度的第三部分
    t3 = (p.concentration1 - q.concentration1) * torch.digamma(p.concentration1)
    # 计算 KL 散度的第四部分
    t4 = (p.concentration0 - q.concentration0) * torch.digamma(p.concentration0)
    # 计算 KL 散度的第五部分
    t5 = (sum_params_q - sum_params_p) * torch.digamma(sum_params_p)
    # 返回计算得到的 KL 散度
    return t1 - t2 + t3 + t4 + t5


# 注册 KL 散度函数用于 Binomial 分布与 Binomial 分布之间的计算
@register_kl(Binomial, Binomial)
def _kl_binomial_binomial(p, q):
    # 根据数学推导实现的 Binomial 分布之间的 KL 散度计算
    if (p.total_count < q.total_count).any():
        # 如果 q.total_count > p.total_count，抛出未实现的异常
        raise NotImplementedError(
            "KL between Binomials where q.total_count > p.total_count is not implemented"
        )
    # 计算 KL 散度
    kl = p.total_count * (
        p.probs * (p.logits - q.logits) + (-p.probs).log1p() - (-q.probs).log1p()
    )
    # 处理 p.total_count > q.total_count 的情况，将对应位置的 KL 散度设为无穷大
    inf_idxs = p.total_count > q.total_count
    kl[inf_idxs] = _infinite_like(kl[inf_idxs])
    # 返回计算得到的 KL 散度
    return kl


# 注册 KL 散度函数用于 Categorical 分布与 Categorical 分布之间的计算
@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    # 计算 KL 散度
    t = p.probs * (p.logits - q.logits)
    # 处理 q.probs == 0 的情况，设为无穷大，确保 KL 散度的计算正确性
    t[(q.probs == 0).expand_as(t)] = inf
    # 处理 p.probs == 0 的情况，对应 KL 散度中的特殊情形
    t[(p.probs == 0).expand_as(t)] = 0
    # 求和得到最终的 KL 散度值
    return t.sum(-1)


# 注册 KL 散度函数用于 ContinuousBernoulli 分布与 ContinuousBernoulli 分布之间的计算
@register_kl(ContinuousBernoulli, ContinuousBernoulli)
def _kl_continuous_bernoulli_continuous_bernoulli(p, q):
    # 计算 KL 散度的第一部分
    t1 = p.mean * (p.logits - q.logits)
    # 计算 KL 散度的第二部分
    t2 = p._cont_bern_log_norm() + torch.log1p(-p.probs)
    # 计算 KL 散度的第三部分
    t3 = -q._cont_bern_log_norm() - torch.log1p(-q.probs)
    # 返回计算得到的 KL 散度
    return t1 + t2 + t3


# 注册 KL 散度函数用于 Dirichlet 分布与 Dirichlet 分布之间的计算
@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p, q):
    # 根据指定的公式实现的 Dirichlet 分布之间的 KL 散度计算
    sum_p_concentration = p.concentration.sum(-1)
    sum_q_concentration = q.concentration.sum(-1)
    # 计算 KL 散度的第一部分
    t1 = sum_p_concentration.lgamma() - sum_q_concentration.lgamma()
    # 计算 KL 散度的第二部分
    t2 = (p.concentration.lgamma() - q.concentration.lgamma()).sum(-1)
    # 计算 KL 散度的第三部分
    t3 = p.concentration - q.concentration
    # 计算 KL 散度的第四部分
    t4 = p.concentration.digamma() - sum_p_concentration.digamma().unsqueeze(-1)
    # 返回计算得到的 KL 散度
    return t1 - t2 + (t3 * t4).sum(-1)


# 注册 KL 散度函数用于 Exponential 分布与 Exponential 分布之间的计算
@register_kl(Exponential, Exponential)
@register_kl(ExponentialFamily, ExponentialFamily)
def _kl_expfamily_expfamily(p, q):
    # 检查输入的概率分布类型是否相同，如果不同则抛出错误
    if not type(p) == type(q):
        raise NotImplementedError(
            "The cross KL-divergence between different exponential families cannot \
                            be computed using Bregman divergences"
        )
    
    # 分别获取自然参数列表并创建可计算梯度的新参数列表
    p_nparams = [np.detach().requires_grad_() for np in p._natural_params]
    q_nparams = q._natural_params
    
    # 计算 p 分布的对数归一化常数及其对自然参数的梯度
    lg_normal = p._log_normalizer(*p_nparams)
    gradients = torch.autograd.grad(lg_normal.sum(), p_nparams, create_graph=True)
    
    # 计算 KL 散度的结果，使用 q 分布的对数归一化常数减去 p 分布的对数归一化常数及其梯度乘积
    result = q._log_normalizer(*q_nparams) - lg_normal
    
    # 根据每个自然参数的差异及其梯度调整结果
    for pnp, qnp, g in zip(p_nparams, q_nparams, gradients):
        term = (qnp - pnp) * g
        result -= _sum_rightmost(term, len(q.event_shape))
    
    # 返回计算得到的 KL 散度结果
    return result


@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p, q):
    # 计算 Gamma 分布之间的 KL 散度
    t1 = q.concentration * (p.rate / q.rate).log()
    t2 = torch.lgamma(q.concentration) - torch.lgamma(p.concentration)
    t3 = (p.concentration - q.concentration) * torch.digamma(p.concentration)
    t4 = (q.rate - p.rate) * (p.concentration / p.rate)
    return t1 + t2 + t3 + t4


@register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(p, q):
    # 计算 Gumbel 分布之间的 KL 散度
    ct1 = p.scale / q.scale
    ct2 = q.loc / q.scale
    ct3 = p.loc / q.scale
    t1 = -ct1.log() - ct2 + ct3
    t2 = ct1 * _euler_gamma
    t3 = torch.exp(ct2 + (1 + ct1).lgamma() - ct3)
    return t1 + t2 + t3 - (1 + _euler_gamma)


@register_kl(Geometric, Geometric)
def _kl_geometric_geometric(p, q):
    # 计算 Geometric 分布之间的 KL 散度
    return -p.entropy() - torch.log1p(-q.probs) / p.probs - q.logits


@register_kl(HalfNormal, HalfNormal)
def _kl_halfnormal_halfnormal(p, q):
    # 计算 HalfNormal 分布之间的 KL 散度，其实现与 Normal 分布相同
    return _kl_normal_normal(p.base_dist, q.base_dist)


@register_kl(Laplace, Laplace)
def _kl_laplace_laplace(p, q):
    # 计算 Laplace 分布之间的 KL 散度
    scale_ratio = p.scale / q.scale
    loc_abs_diff = (p.loc - q.loc).abs()
    t1 = -scale_ratio.log()
    t2 = loc_abs_diff / q.scale
    t3 = scale_ratio * torch.exp(-loc_abs_diff / p.scale)
    return t1 + t2 + t3 - 1


@register_kl(LowRankMultivariateNormal, LowRankMultivariateNormal)
def _kl_lowrankmultivariatenormal_lowrankmultivariatenormal(p, q):
    # 检查输入的 Low Rank Multivariate Normal 分布的事件形状是否相同，如果不同则抛出错误
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two Low Rank Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    # 计算 KL 散度的各项
    term1 = _batch_lowrank_logdet(
        q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q._capacitance_tril
    ) - _batch_lowrank_logdet(
        p._unbroadcasted_cov_factor, p._unbroadcasted_cov_diag, p._capacitance_tril
    )
    term3 = _batch_lowrank_mahalanobis(
        q._unbroadcasted_cov_factor,
        q._unbroadcasted_cov_diag,
        q.loc - p.loc,
        q._capacitance_tril,
    )
    # 返回计算得到的 KL 散度结果
    return term1 + term3
    # 计算变量 qWt_qDinv，表示 q._unbroadcasted_cov_factor.mT 除以 q._unbroadcasted_cov_diag.unsqueeze(-2)
    qWt_qDinv = q._unbroadcasted_cov_factor.mT / q._unbroadcasted_cov_diag.unsqueeze(-2)
    
    # 使用 torch.linalg.solve_triangular 求解线性三角系统，解出 A，其中 q._capacitance_tril 是系数矩阵，qWt_qDinv 是右侧向量
    A = torch.linalg.solve_triangular(q._capacitance_tril, qWt_qDinv, upper=False)
    
    # 计算 term21，是 p._unbroadcasted_cov_diag 除以 q._unbroadcasted_cov_diag 后按最后一个维度求和
    term21 = (p._unbroadcasted_cov_diag / q._unbroadcasted_cov_diag).sum(-1)
    
    # 计算 term22，使用 _batch_trace_XXT 函数计算 p._unbroadcasted_cov_factor 与 q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1) 的批量迹
    term22 = _batch_trace_XXT(
        p._unbroadcasted_cov_factor * q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1)
    )
    
    # 计算 term23，使用 _batch_trace_XXT 函数计算 A 与 p._unbroadcasted_cov_diag.sqrt().unsqueeze(-2) 的批量迹
    term23 = _batch_trace_XXT(A * p._unbroadcasted_cov_diag.sqrt().unsqueeze(-2))
    
    # 计算 term24，使用 _batch_trace_XXT 函数计算 A 与 p._unbroadcasted_cov_factor 的批量迹
    term24 = _batch_trace_XXT(A.matmul(p._unbroadcasted_cov_factor))
    
    # 计算 term2，是 term21、term22、term23 和 term24 的线性组合
    term2 = term21 + term22 - term23 - term24
    
    # 返回结果，这里是损失函数的计算，0.5 乘以括号内的表达式
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])
# 注册 KL 散度计算函数，适用于 MultivariateNormal 和 LowRankMultivariateNormal 类型的分布
@register_kl(MultivariateNormal, LowRankMultivariateNormal)
def _kl_multivariatenormal_lowrankmultivariatenormal(p, q):
    # 如果两个分布的事件形状不同，则抛出数值错误
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two (Low Rank) Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    # 计算 KL 散度的第一项
    term1 = _batch_lowrank_logdet(
        q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q._capacitance_tril
    ) - 2 * p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    
    # 计算 KL 散度的第三项
    term3 = _batch_lowrank_mahalanobis(
        q._unbroadcasted_cov_factor,
        q._unbroadcasted_cov_diag,
        q.loc - p.loc,
        q._capacitance_tril,
    )
    
    # 计算 KL 散度的第二项，根据给定的公式展开
    qWt_qDinv = q._unbroadcasted_cov_factor.mT / q._unbroadcasted_cov_diag.unsqueeze(-2)
    A = torch.linalg.solve_triangular(q._capacitance_tril, qWt_qDinv, upper=False)
    term21 = _batch_trace_XXT(
        p._unbroadcasted_scale_tril * q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1)
    )
    term22 = _batch_trace_XXT(A.matmul(p._unbroadcasted_scale_tril))
    term2 = term21 - term22
    
    # 返回计算得到的 KL 散度的值
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


# 注册 KL 散度计算函数，适用于 LowRankMultivariateNormal 和 MultivariateNormal 类型的分布
@register_kl(LowRankMultivariateNormal, MultivariateNormal)
def _kl_lowrankmultivariatenormal_multivariatenormal(p, q):
    # 如果两个分布的事件形状不同，则抛出数值错误
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two (Low Rank) Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    # 计算 KL 散度的第一项
    term1 = 2 * q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) \
            - _batch_lowrank_logdet(
                p._unbroadcasted_cov_factor, p._unbroadcasted_cov_diag, p._capacitance_tril
            )
    
    # 计算 KL 散度的第三项
    term3 = _batch_mahalanobis(q._unbroadcasted_scale_tril, (q.loc - p.loc))
    
    # 计算 KL 散度的第二项，根据给定的公式展开
    combined_batch_shape = torch._C._infer_size(
        q._unbroadcasted_scale_tril.shape[:-2], p._unbroadcasted_cov_factor.shape[:-2]
    )
    n = p.event_shape[0]
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_cov_factor = p._unbroadcasted_cov_factor.expand(
        combined_batch_shape + (n, p.cov_factor.size(-1))
    )
    p_cov_diag = torch.diag_embed(p._unbroadcasted_cov_diag.sqrt()).expand(
        combined_batch_shape + (n, n)
    )
    term21 = _batch_trace_XXT(
        torch.linalg.solve_triangular(q_scale_tril, p_cov_factor, upper=False)
    )
    term22 = _batch_trace_XXT(
        torch.linalg.solve_triangular(q_scale_tril, p_cov_diag, upper=False)
    )
    term2 = term21 + term22
    
    # 返回计算得到的 KL 散度的值
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


# 注册 KL 散度计算函数，适用于 MultivariateNormal 和 MultivariateNormal 类型的分布
# 定义计算两个多变量正态分布之间 KL 散度的函数
def _kl_multivariatenormal_multivariatenormal(p, q):
    # 引用自维基百科，说明这段代码基于多变量正态分布的 KL 散度计算方法
    if p.event_shape != q.event_shape:
        # 如果两个分布的事件形状不同，抛出数值错误
        raise ValueError(
            "KL-divergence between two Multivariate Normals with different event shapes cannot be computed"
        )

    # 计算 KL 散度的第一项的一半
    half_term1 = q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) - p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    
    # 推断组合的批次形状
    combined_batch_shape = torch._C._infer_size(
        q._unbroadcasted_scale_tril.shape[:-2], p._unbroadcasted_scale_tril.shape[:-2]
    )
    
    # 获取事件形状的维度
    n = p.event_shape[0]
    
    # 扩展 q 和 p 的 scale_tril 到相同的批次形状
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_scale_tril = p._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    
    # 计算 KL 散度的第二项（_batch_trace_XXT 是计算矩阵的批量迹）
    term2 = _batch_trace_XXT(
        torch.linalg.solve_triangular(q_scale_tril, p_scale_tril, upper=False)
    )
    
    # 计算 KL 散度的第三项（_batch_mahalanobis 是计算马氏距离的批量）
    term3 = _batch_mahalanobis(q._unbroadcasted_scale_tril, (q.loc - p.loc))
    
    # 返回 KL 散度的结果
    return half_term1 + 0.5 * (term2 + term3 - n)


# 注册 Normal 分布与 Normal 分布之间的 KL 散度计算函数
@register_kl(Normal, Normal)
def _kl_normal_normal(p, q):
    # 计算方差比率
    var_ratio = (p.scale / q.scale).pow(2)
    
    # 计算 t1
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    
    # 返回 KL 散度的结果
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


# 注册 OneHotCategorical 分布与 OneHotCategorical 分布之间的 KL 散度计算函数
@register_kl(OneHotCategorical, OneHotCategorical)
def _kl_onehotcategorical_onehotcategorical(p, q):
    # 调用通用的分类分布 KL 散度计算函数
    return _kl_categorical_categorical(p._categorical, q._categorical)


# 注册 Pareto 分布与 Pareto 分布之间的 KL 散度计算函数
@register_kl(Pareto, Pareto)
def _kl_pareto_pareto(p, q):
    # 参考自文献 http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf
    scale_ratio = p.scale / q.scale
    alpha_ratio = q.alpha / p.alpha
    t1 = q.alpha * scale_ratio.log()
    t2 = -alpha_ratio.log()
    
    # 计算结果，若 p 支持的下界小于 q 支持的下界，则置为无穷大
    result = t1 + t2 + alpha_ratio - 1
    result[p.support.lower_bound < q.support.lower_bound] = inf
    return result


# 注册 Poisson 分布与 Poisson 分布之间的 KL 散度计算函数
@register_kl(Poisson, Poisson)
def _kl_poisson_poisson(p, q):
    # 计算 Poisson 分布之间的 KL 散度
    return p.rate * (p.rate.log() - q.rate.log()) - (p.rate - q.rate)


# 注册 TransformedDistribution 分布与 TransformedDistribution 分布之间的 KL 散度计算函数
@register_kl(TransformedDistribution, TransformedDistribution)
def _kl_transformed_transformed(p, q):
    # 如果两个分布的变换不同，则抛出未实现错误
    if p.transforms != q.transforms:
        raise NotImplementedError
    # 如果两个分布的事件形状不同，则抛出未实现错误
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    
    # 调用 KL 散度函数计算基础分布之间的 KL 散度
    return kl_divergence(p.base_dist, q.base_dist)


# 注册 Uniform 分布与 Uniform 分布之间的 KL 散度计算函数
@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p, q):
    # 计算 Uniform 分布之间的 KL 散度
    result = ((q.high - q.low) / (p.high - p.low)).log()
    result[(q.low > p.low) | (q.high < p.high)] = inf
    return result


# 注册 Bernoulli 分布与 Poisson 分布之间的 KL 散度计算函数
@register_kl(Bernoulli, Poisson)
def _kl_bernoulli_poisson(p, q):
    # 计算 Bernoulli 分布与 Poisson 分布之间的 KL 散度
    return -p.entropy() - (p.probs * q.rate.log() - q.rate)


# 注册 Beta 分布与 ContinuousBernoulli 分布之间的 KL 散度计算函数
@register_kl(Beta, ContinuousBernoulli)
def _kl_beta_continuous_bernoulli(p, q):
    # 计算 Beta 分布与 ContinuousBernoulli 分布之间的 KL 散度
    return (
        -p.entropy()
        - p.mean * q.logits
        - torch.log1p(-q.probs)
        - q._cont_bern_log_norm()
    )


# 注册 Beta 分布与 Pareto 分布之间的 KL 散度计算函数
@register_kl(Beta, Pareto)
def _kl_beta_infinity(p, q):
    # 返回类似于无穷大的值
    return _infinite_like(p.concentration1)
# 注册 Beta 和 Exponential 分布之间的 KL 散度计算函数
@register_kl(Beta, Exponential)
def _kl_beta_exponential(p, q):
    # 返回 Beta 分布和 Exponential 分布之间的 KL 散度
    return (
        -p.entropy()  # Beta 分布的熵的负值
        - q.rate.log()  # Exponential 分布的速率参数的自然对数
        + q.rate * (p.concentration1 / (p.concentration1 + p.concentration0))  # KL 散度的主要计算部分
    )


# 注册 Beta 和 Gamma 分布之间的 KL 散度计算函数
@register_kl(Beta, Gamma)
def _kl_beta_gamma(p, q):
    t1 = -p.entropy()  # Beta 分布的熵的负值
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()  # Gamma 分布的对数 Gamma 函数项
    t3 = (q.concentration - 1) * (
        p.concentration1.digamma() - (p.concentration1 + p.concentration0).digamma()
    )  # Gamma 分布和 Beta 分布之间的特定项计算
    t4 = q.rate * p.concentration1 / (p.concentration1 + p.concentration0)  # KL 散度的主要计算部分
    return t1 + t2 - t3 + t4  # 返回 Beta 和 Gamma 分布之间的 KL 散度


# TODO: 添加 Beta-Laplace 分布之间的 KL 散度计算函数


# 注册 Beta 和 Normal 分布之间的 KL 散度计算函数
@register_kl(Beta, Normal)
def _kl_beta_normal(p, q):
    E_beta = p.concentration1 / (p.concentration1 + p.concentration0)  # Beta 分布的期望
    var_normal = q.scale.pow(2)  # Normal 分布的方差
    t1 = -p.entropy()  # Beta 分布的熵的负值
    t2 = 0.5 * (var_normal * 2 * math.pi).log()  # Normal 分布的标准化系数
    t3 = (
        E_beta * (1 - E_beta) / (p.concentration1 + p.concentration0 + 1)
        + E_beta.pow(2)
    ) * 0.5  # KL 散度的特定项计算
    t4 = q.loc * E_beta  # KL 散度的主要计算部分
    t5 = q.loc.pow(2) * 0.5  # KL 散度的特定项计算
    return t1 + t2 + (t3 - t4 + t5) / var_normal  # 返回 Beta 和 Normal 分布之间的 KL 散度


# 注册 Beta 和 Uniform 分布之间的 KL 散度计算函数
@register_kl(Beta, Uniform)
def _kl_beta_uniform(p, q):
    result = -p.entropy() + (q.high - q.low).log()  # Beta 分布和 Uniform 分布之间的 KL 散度的主要计算部分
    result[(q.low > p.support.lower_bound) | (q.high < p.support.upper_bound)] = inf  # 处理无效区域的情况
    return result  # 返回 Beta 和 Uniform 分布之间的 KL 散度


# 注册 ContinuousBernoulli 和 Pareto 分布之间的 KL 散度计算函数
@register_kl(ContinuousBernoulli, Pareto)
def _kl_continuous_bernoulli_infinity(p, q):
    return _infinite_like(p.probs)  # 返回无限大的值，因为 ContinuousBernoulli 和 Pareto 分布之间没有闭式的 KL 散度计算方法


# 注册 ContinuousBernoulli 和 Exponential 分布之间的 KL 散度计算函数
@register_kl(ContinuousBernoulli, Exponential)
def _kl_continuous_bernoulli_exponential(p, q):
    return (
        -p.entropy()  # ContinuousBernoulli 分布的熵的负值
        - torch.log(q.rate)  # Exponential 分布的速率参数的自然对数
        + q.rate * p.mean  # KL 散度的主要计算部分
    )


# 注册 ContinuousBernoulli 和 Gamma 分布之间的 KL 散度计算函数
# 注意：ContinuousBernoulli 和 Gamma 分布之间没有闭式的 KL 散度计算方法
# TODO: 添加 ContinuousBernoulli-Laplace 分布之间的 KL 散度计算函数


# 注册 ContinuousBernoulli 和 Normal 分布之间的 KL 散度计算函数
@register_kl(ContinuousBernoulli, Normal)
def _kl_continuous_bernoulli_normal(p, q):
    t1 = -p.entropy()  # ContinuousBernoulli 分布的熵的负值
    t2 = 0.5 * (math.log(2.0 * math.pi) + torch.square(q.loc / q.scale)) + torch.log(
        q.scale
    )  # Normal 分布的标准化系数
    t3 = (p.variance + torch.square(p.mean) - 2.0 * q.loc * p.mean) / (
        2.0 * torch.square(q.scale)
    )  # KL 散度的特定项计算
    return t1 + t2 + t3  # 返回 ContinuousBernoulli 和 Normal 分布之间的 KL 散度


# 注册 ContinuousBernoulli 和 Uniform 分布之间的 KL 散度计算函数
@register_kl(ContinuousBernoulli, Uniform)
def _kl_continuous_bernoulli_uniform(p, q):
    result = -p.entropy() + (q.high - q.low).log()  # ContinuousBernoulli 分布和 Uniform 分布之间的 KL 散度的主要计算部分
    return torch.where(
        torch.max(
            torch.ge(q.low, p.support.lower_bound),
            torch.le(q.high, p.support.upper_bound),
        ),
        torch.ones_like(result) * inf,  # 处理无效区域的情况
        result,
    )


# 注册 Exponential 和 Beta 分布之间的 KL 散度计算函数
# 注册 Exponential 和 ContinuousBernoulli 分布之间的 KL 散度计算函数
# 注册 Exponential 和 Pareto 分布之间的 KL 散度计算函数
# 注册 Exponential 和 Uniform 分布之间的 KL 散度计算函数
def _kl_exponential_infinity(p, q):
    return _infinite_like(p.rate)  # 返回无限大的值，因为 Exponential 和其他分布之间没有闭式的 KL 散度计算方法


# 注册 Exponential 和 Gamma 分布之间的 KL 散度计算函数
@register_kl(Exponential, Gamma)
def _kl_exponential_gamma(p, q):
    ratio = q.rate / p.rate  # 比率参数
    t1 = -q.concentration * torch.log(ratio)  # KL 散度的主要计算部分
    return (
        t1
        + ratio
        + q.concentration.lgamma()
        + q.concentration * _euler_gamma
        - (1 + _euler_gamma)
    )  # 返回 Exponential 和 Gamma 分布之间的 KL 散度
    # 在控制台打印输出字符串 "Hello World!"
    print("Hello World!")
# 注册 KL 散度计算函数，计算 Exponential 分布和 Gumbel 分布之间的 KL 散度
@register_kl(Exponential, Gumbel)
def _kl_exponential_gumbel(p, q):
    # 计算参数的乘积：Exponential 分布的 rate 参数乘以 Gumbel 分布的 scale 参数
    scale_rate_prod = p.rate * q.scale
    # 计算 loc 和 scale 的比率：Gumbel 分布的 loc 参数除以 scale 参数
    loc_scale_ratio = q.loc / q.scale
    # 第一个项 t1：使用乘积的自然对数减去 1
    t1 = scale_rate_prod.log() - 1
    # 第二个项 t2：计算指数函数，乘以乘积除以（乘积加一）
    t2 = torch.exp(loc_scale_ratio) * scale_rate_prod / (scale_rate_prod + 1)
    # 第三个项 t3：乘以乘积的倒数
    t3 = scale_rate_prod.reciprocal()
    # 返回 KL 散度的计算结果
    return t1 - loc_scale_ratio + t2 + t3


# TODO: Add Exponential-Laplace KL Divergence


# 注册 KL 散度计算函数，计算 Exponential 分布和 Normal 分布之间的 KL 散度
@register_kl(Exponential, Normal)
def _kl_exponential_normal(p, q):
    # 计算 Normal 分布的方差
    var_normal = q.scale.pow(2)
    # 计算 Exponential 分布的 rate 参数的平方
    rate_sqr = p.rate.pow(2)
    # 第一个项 t1：0.5 乘以 rate 参数的自然对数和 Normal 分布方差的自然对数，再加上常数项
    t1 = 0.5 * torch.log(rate_sqr * var_normal * 2 * math.pi)
    # 第二个项 t2：rate 参数的倒数
    t2 = rate_sqr.reciprocal()
    # 第三个项 t3：Normal 分布的 loc 参数除以 Exponential 分布的 rate 参数
    t3 = q.loc / p.rate
    # 第四个项 t4：Normal 分布的 loc 参数的平方乘以 0.5
    t4 = q.loc.pow(2) * 0.5
    # 返回 KL 散度的计算结果
    return t1 - 1 + (t2 - t3 + t4) / var_normal


# 注册 KL 散度计算函数，计算 Gamma 分布和 Beta 分布之间的 KL 散度
@register_kl(Gamma, Beta)
@register_kl(Gamma, ContinuousBernoulli)
@register_kl(Gamma, Pareto)
@register_kl(Gamma, Uniform)
def _kl_gamma_infinity(p, q):
    # 返回 Gamma 分布对无穷的 KL 散度，调用 _infinite_like 函数计算
    return _infinite_like(p.concentration)


# 注册 KL 散度计算函数，计算 Gamma 分布和 Exponential 分布之间的 KL 散度
@register_kl(Gamma, Exponential)
def _kl_gamma_exponential(p, q):
    # 返回 Gamma 分布和 Exponential 分布之间的 KL 散度
    return -p.entropy() - q.rate.log() + q.rate * p.concentration / p.rate


# 注册 KL 散度计算函数，计算 Gamma 分布和 Gumbel 分布之间的 KL 散度
@register_kl(Gamma, Gumbel)
def _kl_gamma_gumbel(p, q):
    # 计算参数的乘积：Gamma 分布的 rate 参数乘以 Gumbel 分布的 scale 参数
    beta_scale_prod = p.rate * q.scale
    # 计算 loc 和 scale 的比率：Gumbel 分布的 loc 参数除以 scale 参数
    loc_scale_ratio = q.loc / q.scale
    # 第一个项 t1：Gamma 分布 concentration 参数减一乘以 concentration 参数的 digamma 函数值，
    # 减去 concentration 参数的对数，再减去 concentration 参数本身
    t1 = (
        (p.concentration - 1) * p.concentration.digamma()
        - p.concentration.lgamma()
        - p.concentration
    )
    # 第二个项 t2：乘积的自然对数，加上 concentration 参数除以乘积
    t2 = beta_scale_prod.log() + p.concentration / beta_scale_prod
    # 第三个项 t3：指数函数，乘以一加上乘积的倒数，再乘以负的 concentration 参数，
    # 减去 loc 和 scale 的比率
    t3 = (
        torch.exp(loc_scale_ratio)
        * (1 + beta_scale_prod.reciprocal()).pow(-p.concentration)
        - loc_scale_ratio
    )
    # 返回 KL 散度的计算结果
    return t1 + t2 + t3


# TODO: Add Gamma-Laplace KL Divergence


# 注册 KL 散度计算函数，计算 Gamma 分布和 Normal 分布之间的 KL 散度
@register_kl(Gamma, Normal)
def _kl_gamma_normal(p, q):
    # 计算 Normal 分布的方差
    var_normal = q.scale.pow(2)
    # 计算 Gamma 分布的 rate 参数的平方
    beta_sqr = p.rate.pow(2)
    # 第一个项 t1：0.5 乘以 rate 参数的自然对数和 Normal 分布方差的自然对数，再减去 concentration 参数的对数
    t1 = (
        0.5 * torch.log(beta_sqr * var_normal * 2 * math.pi)
        - p.concentration
        - p.concentration.lgamma()
    )
    # 第二个项 t2：0.5 乘以 concentration 参数的平方加上 concentration 参数，再除以 rate 参数的平方
    t2 = 0.5 * (p.concentration.pow(2) + p.concentration) / beta_sqr
    # 第三个项 t3：Normal 分布的 loc 参数乘以 concentration 参数除以 rate 参数
    t3 = q.loc * p.concentration / p.rate
    # 第四个项 t4：Normal 分布的 loc 参数的平方乘以 0.5
    t4 = 0.5 * q.loc.pow(2)
    # 返回 KL 散度的计算结果
    return (
        t1
        + (p.concentration - 1) * p.concentration.digamma()
        + (t2 - t3 + t4) / var_normal
    )


# 注册 KL 散度计算函数，计算 Gumbel 分布和 Beta 分布之间的 KL 散度
@register_kl(Gumbel, Beta)
@register_kl(Gumbel, ContinuousBernoulli)
@register_kl(Gumbel, Exponential)
@register_kl(Gumbel, Gamma)
@register_kl(Gumbel, Pareto)
@register_kl(Gumbel, Uniform)
def _kl_gumbel_infinity(p, q):
    # 返回 Gumbel 分布对无穷的 KL 散度，调用 _infinite_like 函数计算
    return _infinite_like(p.loc)


# TODO: Add Gumbel-Laplace KL Divergence


# 注册 KL 散度计算函数，计算 Gumbel 分布和 Normal 分布之间的 KL 散度
@register_kl(Gumbel, Normal)
def _kl_gumbel_normal(p, q):
    # 计算参数的比率：Gumbel 分布的 scale 参数除以 Normal 分布的 scale 参数
    param_ratio = p.scale / q.scale
    # 第一个项 t1：比率的自然对数除以根号 2π
    t1 = (param_ratio / math.sqrt(2 * math.pi)).log()
    # 第二个项 t2：π 乘以比率的平方乘以 0.5，再除以 3
    t2 = (math.pi * param_ratio * 0.5).pow(2) / 3
    # 第三个项 t3：loc 和 scale 的比率的平方乘以 0.5
    t3 = ((p.loc + p.scale * _euler_gamma - q.loc) / q.scale).pow(2) * 0.5
    # 返回 KL 散度的计算结果
    return -t1 + t2 + t3 - (_euler_gamma + 1)


# 注册 KL 散度计算函数，计算 Laplace 分布和 Beta 分布之间的 KL 散度
@register_kl(Laplace, Beta)
@register_kl(Laplace, ContinuousBernoulli)
    # 返回一个与输入参数 p.loc 类似的无限对象
    return _infinite_like(p.loc)
# 注册函数 _kl_laplace_normal 用于计算 Laplace 分布和 Normal 分布之间的 KL 散度
@register_kl(Laplace, Normal)
def _kl_laplace_normal(p, q):
    # 计算 Normal 分布方差的平方
    var_normal = q.scale.pow(2)
    # 计算比例 p.scale^2 / var_normal
    scale_sqr_var_ratio = p.scale.pow(2) / var_normal
    # 计算 KL 散度中的第一项
    t1 = 0.5 * torch.log(2 * scale_sqr_var_ratio / math.pi)
    # 计算 KL 散度中的第二项
    t2 = 0.5 * p.loc.pow(2)
    # 计算 KL 散度中的第三项
    t3 = p.loc * q.loc
    # 计算 KL 散度中的第四项
    t4 = 0.5 * q.loc.pow(2)
    # 返回计算得到的 KL 散度值
    return -t1 + scale_sqr_var_ratio + (t2 - t3 + t4) / var_normal - 1


# 注册函数 _kl_normal_infinity 处理 Normal 分布与无限分布（如 Beta、ContinuousBernoulli 等）之间的 KL 散度
@register_kl(Normal, Beta)
@register_kl(Normal, ContinuousBernoulli)
@register_kl(Normal, Exponential)
@register_kl(Normal, Gamma)
@register_kl(Normal, Pareto)
@register_kl(Normal, Uniform)
def _kl_normal_infinity(p, q):
    # 返回无限分布的 KL 散度，通过调用 _infinite_like 函数处理
    return _infinite_like(p.loc)


# 注册函数 _kl_normal_gumbel 用于计算 Normal 分布和 Gumbel 分布之间的 KL 散度
@register_kl(Normal, Gumbel)
def _kl_normal_gumbel(p, q):
    # 计算均值比例 p.loc / q.scale
    mean_scale_ratio = p.loc / q.scale
    # 计算方差比例 (p.scale / q.scale)^2
    var_scale_sqr_ratio = (p.scale / q.scale).pow(2)
    # 计算均值和 loc/scale 之间的差异
    loc_scale_ratio = q.loc / q.scale
    # 计算 KL 散度中的第一项
    t1 = var_scale_sqr_ratio.log() * 0.5
    # 计算 KL 散度中的第二项
    t2 = mean_scale_ratio - loc_scale_ratio
    # 计算 KL 散度中的第三项
    t3 = torch.exp(-mean_scale_ratio + 0.5 * var_scale_sqr_ratio + loc_scale_ratio)
    # 返回计算得到的 KL 散度值
    return -t1 + t2 + t3 - (0.5 * (1 + math.log(2 * math.pi)))


# 注册函数 _kl_normal_laplace 用于计算 Normal 分布和 Laplace 分布之间的 KL 散度
@register_kl(Normal, Laplace)
def _kl_normal_laplace(p, q):
    # 计算均值之差 p.loc - q.loc
    loc_diff = p.loc - q.loc
    # 计算比例 p.scale / q.scale
    scale_ratio = p.scale / q.scale
    # 计算 (p.loc - q.loc) / p.scale
    loc_diff_scale_ratio = loc_diff / p.scale
    # 计算 KL 散度中的第一项
    t1 = torch.log(scale_ratio)
    # 计算 KL 散度中的第二项
    t2 = (
        math.sqrt(2 / math.pi) * p.scale * torch.exp(-0.5 * loc_diff_scale_ratio.pow(2))
    )
    # 计算 KL 散度中的第三项
    t3 = loc_diff * torch.erf(math.sqrt(0.5) * loc_diff_scale_ratio)
    # 返回计算得到的 KL 散度值
    return -t1 + (t2 + t3) / q.scale - (0.5 * (1 + math.log(0.5 * math.pi)))


# 注册函数 _kl_pareto_infinity 处理 Pareto 分布与无限分布（如 Beta、ContinuousBernoulli、Uniform）之间的 KL 散度
@register_kl(Pareto, Beta)
@register_kl(Pareto, ContinuousBernoulli)
@register_kl(Pareto, Uniform)
def _kl_pareto_infinity(p, q):
    # 返回无限分布的 KL 散度，通过调用 _infinite_like 函数处理
    return _infinite_like(p.scale)


# 注册函数 _kl_pareto_exponential 用于计算 Pareto 分布和 Exponential 分布之间的 KL 散度
@register_kl(Pareto, Exponential)
def _kl_pareto_exponential(p, q):
    # 计算尺度和率的乘积 p.scale * q.rate
    scale_rate_prod = p.scale * q.rate
    # 计算 KL 散度中的第一项
    t1 = (p.alpha / scale_rate_prod).log()
    # 计算 KL 散度中的第二项
    t2 = p.alpha.reciprocal()
    # 计算 KL 散度中的第三项
    t3 = p.alpha * scale_rate_prod / (p.alpha - 1)
    # 将 alpha <= 1 的结果设置为无穷大
    result = t1 - t2 + t3 - 1
    result[p.alpha <= 1] = inf
    # 返回计算得到的 KL 散度值
    return result


# 注册函数 _kl_pareto_gamma 用于计算 Pareto 分布和 Gamma 分布之间的 KL 散度
@register_kl(Pareto, Gamma)
def _kl_pareto_gamma(p, q):
    # 计算常用项 p.scale.log() + p.alpha.reciprocal()
    common_term = p.scale.log() + p.alpha.reciprocal()
    # 计算 KL 散度中的第一项
    t1 = p.alpha.log() - common_term
    # 计算 KL 散度中的第二项
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    # 计算 KL 散度中的第三项
    t3 = (1 - q.concentration) * common_term
    # 计算 KL 散度中的第四项
    t4 = q.rate * p.alpha * p.scale / (p.alpha - 1)
    # 将 alpha <= 1 的结果设置为无穷大
    result = t1 + t2 + t3 + t4 - 1
    result[p.alpha <= 1] = inf
    # 返回计算得到的 KL 散度值
    return result


# 注册函数 _kl_pareto_normal 用于计算 Pareto 分布和 Normal 分布之间的 KL 散度
@register_kl(Pareto, Normal)
def _kl_pareto_normal(p, q):
    # 计算 Normal 分布方差的两倍
    var_normal = 2 * q.scale.pow(2)
    # 计算常用项 p.scale / (p.alpha - 1)
    common_term = p.scale / (p.alpha - 1)
    # 计算 KL 散度中的第一项
    t1 = (math.sqrt(2 * math.pi) * q.scale * p.alpha / p.scale).log()
    # 计算 KL 散度中的第二项
    t2 = p.alpha.reciprocal()
    # 计算 KL 散度中的第三项
    t3 = p.alpha * common_term.pow(2) / (p.alpha - 2)
    # 计算 KL 散度中的第四项
    t4 = (p.alpha * common_term - q.loc).pow(2)
    # 将 alpha <= 2 的结果设置为无穷大
    result = t1 - t2 + (t3 + t4) / var_normal - 1
    result[p.alpha <= 2] = inf
    # 返回计算得到的 KL 散度值
    return result


# 注册函数 _kl_poisson_infinity 处理 Poisson 分布与无限分布（如 Bernoulli、Binomial）之间的 KL 散度
@register_kl(Poisson, Bernoulli)
@register_kl(Poisson, Binomial)
def _kl_poisson_infinity(p, q):
    # 返回无限分布的 KL 散度，通过调用 _infinite_like 函数处理
    return _infinite_like(p.rate)


这些注释详
# 注册 KL 散度计算函数，Uniform 和 Beta 分布之间的 KL 散度
@register_kl(Uniform, Beta)
def _kl_uniform_beta(p, q):
    # 计算公共项
    common_term = p.high - p.low
    # 计算 t1
    t1 = torch.log(common_term)
    # 计算 t2
    t2 = (
        (q.concentration1 - 1)
        * (_x_log_x(p.high) - _x_log_x(p.low) - common_term)
        / common_term
    )
    # 计算 t3
    t3 = (
        (q.concentration0 - 1)
        * (_x_log_x(1 - p.high) - _x_log_x(1 - p.low) + common_term)
        / common_term
    )
    # 计算 t4
    t4 = (
        q.concentration1.lgamma()
        + q.concentration0.lgamma()
        - (q.concentration1 + q.concentration0).lgamma()
    )
    # 计算最终结果
    result = t3 + t4 - t1 - t2
    # 将不合理区域设为无穷大
    result[(p.high > q.support.upper_bound) | (p.low < q.support.lower_bound)] = inf
    return result


# 注册 KL 散度计算函数，Uniform 和 ContinuousBernoulli 分布之间的 KL 散度
@register_kl(Uniform, ContinuousBernoulli)
def _kl_uniform_continuous_bernoulli(p, q):
    # 计算结果
    result = (
        -p.entropy()
        - p.mean * q.logits
        - torch.log1p(-q.probs)
        - q._cont_bern_log_norm()
    )
    # 使用条件判断处理不合理区域
    return torch.where(
        torch.max(
            torch.ge(p.high, q.support.upper_bound),
            torch.le(p.low, q.support.lower_bound),
        ),
        torch.ones_like(result) * inf,
        result,
    )


# 注册 KL 散度计算函数，Uniform 和 Exponential 分布之间的 KL 散度
@register_kl(Uniform, Exponential)
def _kl_uniform_exponetial(p, q):
    # 计算结果
    result = q.rate * (p.high + p.low) / 2 - ((p.high - p.low) * q.rate).log()
    # 将不合理区域设为无穷大
    result[p.low < q.support.lower_bound] = inf
    return result


# 注册 KL 散度计算函数，Uniform 和 Gamma 分布之间的 KL 散度
@register_kl(Uniform, Gamma)
def _kl_uniform_gamma(p, q):
    # 计算公共项
    common_term = p.high - p.low
    # 计算 t1
    t1 = common_term.log()
    # 计算 t2
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    # 计算 t3
    t3 = (
        (1 - q.concentration)
        * (_x_log_x(p.high) - _x_log_x(p.low) - common_term)
        / common_term
    )
    # 计算 t4
    t4 = q.rate * (p.high + p.low) / 2
    # 计算最终结果
    result = -t1 + t2 + t3 + t4
    # 将不合理区域设为无穷大
    result[p.low < q.support.lower_bound] = inf
    return result


# 注册 KL 散度计算函数，Uniform 和 Gumbel 分布之间的 KL 散度
@register_kl(Uniform, Gumbel)
def _kl_uniform_gumbel(p, q):
    # 计算公共项
    common_term = q.scale / (p.high - p.low)
    # 计算高位与位置参数的差异
    high_loc_diff = (p.high - q.loc) / q.scale
    low_loc_diff = (p.low - q.loc) / q.scale
    # 计算 t1
    t1 = common_term.log() + 0.5 * (high_loc_diff + low_loc_diff)
    # 计算 t2
    t2 = common_term * (torch.exp(-high_loc_diff) - torch.exp(-low_loc_diff))
    # 返回结果
    return t1 - t2


# TODO: Uniform-Laplace KL Divergence


# 注册 KL 散度计算函数，Uniform 和 Normal 分布之间的 KL 散度
@register_kl(Uniform, Normal)
def _kl_uniform_normal(p, q):
    # 计算公共项
    common_term = p.high - p.low
    # 计算 t1
    t1 = (math.sqrt(math.pi * 2) * q.scale / common_term).log()
    # 计算 t2
    t2 = (common_term).pow(2) / 12
    # 计算 t3
    t3 = ((p.high + p.low - 2 * q.loc) / 2).pow(2)
    # 返回结果
    return t1 + 0.5 * (t2 + t3) / q.scale.pow(2)


# 注册 KL 散度计算函数，Uniform 和 Pareto 分布之间的 KL 散度
@register_kl(Uniform, Pareto)
def _kl_uniform_pareto(p, q):
    # 计算 Uniform 分布的支持范围
    support_uniform = p.high - p.low
    # 计算 t1
    t1 = (q.alpha * q.scale.pow(q.alpha) * (support_uniform)).log()
    # 计算 t2
    t2 = (_x_log_x(p.high) - _x_log_x(p.low) - support_uniform) / support_uniform
    # 计算结果
    result = t2 * (q.alpha + 1) - t1
    # 将不合理区域设为无穷大
    result[p.low < q.support.lower_bound] = inf
    return result


# 注册 KL 散度计算函数，Independent 分布间的 KL 散度（未完成）
@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    # 检查两个概率分布对象的重新解释的批次维度是否相等，如果不相等则引发未实现的错误
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    
    # 计算两个概率分布的 KL 散度，并将结果保存在变量 result 中
    result = kl_divergence(p.base_dist, q.base_dist)
    
    # 调用函数 _sum_rightmost，对结果 result 的右侧若干维度进行求和，具体维度由 p.reinterpreted_batch_ndims 决定
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)
@register_kl(Cauchy, Cauchy)
def _kl_cauchy_cauchy(p, q):
    # 注册一个 KL 散度函数，计算两个 Cauchy 分布之间的 KL 散度
    # 参考文献：https://arxiv.org/abs/1905.10965

    # 计算 KL 散度的第一项：log((scale_p + scale_q)^2 + (loc_p - loc_q)^2)
    t1 = ((p.scale + q.scale).pow(2) + (p.loc - q.loc).pow(2)).log()

    # 计算 KL 散度的第二项：log(4 * scale_p * scale_q)
    t2 = (4 * p.scale * q.scale).log()

    # 返回两项之差作为最终的 KL 散度值
    return t1 - t2


def _add_kl_info():
    """Appends a list of implemented KL functions to the doc for kl_divergence."""
    # 创建一个列表，列出已实现 KL 散度的分布对
    rows = [
        "KL divergence is currently implemented for the following distribution pairs:"
    ]
    
    # 遍历已注册的 KL 散度函数列表，并按分布名称排序
    for p, q in sorted(
        _KL_REGISTRY, key=lambda p_q: (p_q[0].__name__, p_q[1].__name__)
    ):
        # 将每个分布对的信息添加到列表中，格式为 "* :class:`~torch.distributions.{p}` and :class:`~torch.distributions.{q}`"
        rows.append(
            f"* :class:`~torch.distributions.{p.__name__}` and :class:`~torch.distributions.{q.__name__}`"
        )
    
    # 将列表转换为字符串，每个条目缩进，并添加到 kl_divergence 函数的文档字符串末尾
    kl_info = "\n\t".join(rows)
    if kl_divergence.__doc__:
        kl_divergence.__doc__ += kl_info  # type: ignore[operator]
```