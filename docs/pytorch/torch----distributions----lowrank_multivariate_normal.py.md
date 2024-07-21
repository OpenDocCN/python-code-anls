# `.\pytorch\torch\distributions\lowrank_multivariate_normal.py`

```
# mypy: allow-untyped-defs  # 允许不对定义进行类型注解（针对mypy类型检查工具的设置）
import math  # 导入数学库

import torch  # 导入PyTorch库
from torch.distributions import constraints  # 导入约束模块
from torch.distributions.distribution import Distribution  # 导入分布基类
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_mv  # 导入多元正态分布相关函数
from torch.distributions.utils import _standard_normal, lazy_property  # 导入工具函数

__all__ = ["LowRankMultivariateNormal"]  # 指定模块导出的公共接口

def _batch_capacitance_tril(W, D):
    r"""
    Computes Cholesky of :math:`I + W.T @ inv(D) @ W` for a batch of matrices :math:`W`
    and a batch of vectors :math:`D`.
    """
    m = W.size(-1)  # 获取最后一个维度大小
    Wt_Dinv = W.mT / D.unsqueeze(-2)  # 计算 W.T @ inv(D)
    K = torch.matmul(Wt_Dinv, W).contiguous()  # 计算 Wt_Dinv @ W，并确保张量连续性
    K.view(-1, m * m)[:, :: m + 1] += 1  # 将 identity 矩阵加到 K 上
    return torch.linalg.cholesky(K)  # 返回 K 的 Cholesky 分解结果

def _batch_lowrank_logdet(W, D, capacitance_tril):
    r"""
    Uses "matrix determinant lemma"::
        log|W @ W.T + D| = log|C| + log|D|,
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute
    the log determinant.
    """
    return 2 * capacitance_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) + D.log().sum(
        -1
    )  # 使用矩阵行列式引理计算对数行列式

def _batch_lowrank_mahalanobis(W, D, x, capacitance_tril):
    r"""
    Uses "Woodbury matrix identity"::
        inv(W @ W.T + D) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D),
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute the squared
    Mahalanobis distance :math:`x.T @ inv(W @ W.T + D) @ x`.
    """
    Wt_Dinv = W.mT / D.unsqueeze(-2)  # 计算 W.T @ inv(D)
    Wt_Dinv_x = _batch_mv(Wt_Dinv, x)  # 计算 Wt_Dinv @ x
    mahalanobis_term1 = (x.pow(2) / D).sum(-1)  # 计算 Mahalanobis 距离的第一项
    mahalanobis_term2 = _batch_mahalanobis(capacitance_tril, Wt_Dinv_x)  # 计算 Mahalanobis 距离的第二项
    return mahalanobis_term1 - mahalanobis_term2  # 返回 Mahalanobis 距离的结果

class LowRankMultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal distribution with covariance matrix having a low-rank form
    parameterized by :attr:`cov_factor` and :attr:`cov_diag`::

        covariance_matrix = cov_factor @ cov_factor.T + cov_diag

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LowRankMultivariateNormal(torch.zeros(2), torch.tensor([[1.], [0.]]), torch.ones(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]`, cov_factor=`[[1],[0]]`, cov_diag=`[1,1]`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution with shape `batch_shape + event_shape`
        cov_factor (Tensor): factor part of low-rank form of covariance matrix with shape
            `batch_shape + event_shape + (rank,)`
        cov_diag (Tensor): diagonal part of low-rank form of covariance matrix with shape
            `batch_shape + event_shape`
    """
    The computation for determinant and inverse of covariance matrix is avoided when
    `cov_factor.shape[1] << cov_factor.shape[0]` thanks to `Woodbury matrix identity
    <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_ and
    `matrix determinant lemma <https://en.wikipedia.org/wiki/Matrix_determinant_lemma>`_.
    Thanks to these formulas, we just need to compute the determinant and inverse of
    the small size "capacitance" matrix::

        capacitance = I + cov_factor.T @ inv(cov_diag) @ cov_factor
    """

    # 参数约束定义，loc必须是实数向量，cov_factor是独立的实数约束，至少是二维的，cov_diag是独立的正数约束，至少是一维的
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.independent(constraints.real, 2),
        "cov_diag": constraints.independent(constraints.positive, 1),
    }

    # 支持的输出是实数向量
    support = constraints.real_vector

    # 具有重参数化样本
    has_rsample = True

    def __init__(self, loc, cov_factor, cov_diag, validate_args=None):
        # 如果loc的维度小于1，抛出值错误
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        # 事件形状是loc的最后一个维度的形状
        event_shape = loc.shape[-1:]
        # 如果cov_factor的维度小于2，抛出值错误，要求至少是二维的，可以有额外的批次维度
        if cov_factor.dim() < 2:
            raise ValueError(
                "cov_factor must be at least two-dimensional, "
                "with optional leading batch dimensions"
            )
        # 如果cov_factor的倒数第二个维度不等于事件形状，抛出值错误，要求形状是事件形状[0] x m的批量矩阵
        if cov_factor.shape[-2:-1] != event_shape:
            raise ValueError(
                f"cov_factor must be a batch of matrices with shape {event_shape[0]} x m"
            )
        # 如果cov_diag的最后一个维度不等于事件形状，抛出值错误，要求形状是事件形状的向量批量
        if cov_diag.shape[-1:] != event_shape:
            raise ValueError(
                f"cov_diag must be a batch of vectors with shape {event_shape}"
            )

        # 在loc的最后一个维度上增加一个维度，确保广播匹配
        loc_ = loc.unsqueeze(-1)
        # 在cov_diag的最后一个维度上增加一个维度，确保广播匹配
        cov_diag_ = cov_diag.unsqueeze(-1)
        try:
            # 使用torch广播张量来确保loc，cov_factor和cov_diag具有相同的形状
            loc_, self.cov_factor, cov_diag_ = torch.broadcast_tensors(
                loc_, cov_factor, cov_diag_
            )
        except RuntimeError as e:
            # 如果广播失败，抛出值错误，指示不兼容的批量形状
            raise ValueError(
                f"Incompatible batch shapes: loc {loc.shape}, cov_factor {cov_factor.shape}, cov_diag {cov_diag.shape}"
            ) from e
        # 只保留loc_的第一个维度，以移除广播后的维度
        self.loc = loc_[..., 0]
        # 只保留cov_diag_的第一个维度，以移除广播后的维度
        self.cov_diag = cov_diag_[..., 0]
        # 批量形状是loc的所有维度，去掉最后一个维度
        batch_shape = self.loc.shape[:-1]

        # 未广播的cov_factor和cov_diag用于后续计算
        self._unbroadcasted_cov_factor = cov_factor
        self._unbroadcasted_cov_diag = cov_diag
        # 计算_capacitance_tril，这是一个特定计算的函数，用于生成"capacitance"三角矩阵
        self._capacitance_tril = _batch_capacitance_tril(cov_factor, cov_diag)
        # 调用父类的初始化方法，传入批量形状和事件形状，进行参数验证
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
    def expand(self, batch_shape, _instance=None):
        # 创建一个新的 LowRankMultivariateNormal 对象，确保类型兼容性
        new = self._get_checked_instance(LowRankMultivariateNormal, _instance)
        # 将输入的 batch_shape 转换为 torch.Size 对象
        batch_shape = torch.Size(batch_shape)
        # 计算新的 loc 的形状，扩展为 batch_shape + self.event_shape
        loc_shape = batch_shape + self.event_shape
        # 使用 expand 方法扩展 loc 到新的形状
        new.loc = self.loc.expand(loc_shape)
        # 使用 expand 方法扩展 cov_diag 到新的形状
        new.cov_diag = self.cov_diag.expand(loc_shape)
        # 使用 expand 方法扩展 cov_factor 到新的形状，包括最后一维的形状
        new.cov_factor = self.cov_factor.expand(loc_shape + self.cov_factor.shape[-1:])
        # 复制 _unbroadcasted_cov_factor 和 _unbroadcasted_cov_diag
        new._unbroadcasted_cov_factor = self._unbroadcasted_cov_factor
        new._unbroadcasted_cov_diag = self._unbroadcasted_cov_diag
        # 复制 _capacitance_tril
        new._capacitance_tril = self._capacitance_tril
        # 调用父类的构造函数初始化新对象的 batch_shape 和 event_shape
        super(LowRankMultivariateNormal, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        # 继承原始对象的 validate_args 属性
        new._validate_args = self._validate_args
        # 返回新创建的对象
        return new

    @property
    def mean(self):
        # 返回当前对象的 loc 作为均值
        return self.loc

    @property
    def mode(self):
        # 返回当前对象的 loc 作为众数
        return self.loc

    @lazy_property
    def variance(self):
        # 计算方差，使用 lazy evaluation，保证性能
        return (
            self._unbroadcasted_cov_factor.pow(2).sum(-1) + self._unbroadcasted_cov_diag
        ).expand(self._batch_shape + self._event_shape)

    @lazy_property
    def scale_tril(self):
        # 计算 scale_tril 矩阵，增强 Cholesky 分解的数值稳定性
        n = self._event_shape[0]
        cov_diag_sqrt_unsqueeze = self._unbroadcasted_cov_diag.sqrt().unsqueeze(-1)
        Dinvsqrt_W = self._unbroadcasted_cov_factor / cov_diag_sqrt_unsqueeze
        K = torch.matmul(Dinvsqrt_W, Dinvsqrt_W.transpose(-1, -2)).contiguous()
        K.view(-1, n * n)[:, :: n + 1] += 1  # 添加单位矩阵到 K
        scale_tril = cov_diag_sqrt_unsqueeze * torch.linalg.cholesky(K)
        # 使用 expand 方法扩展 scale_tril 到新的形状
        return scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        # 计算协方差矩阵，使用 lazy evaluation，保证性能
        covariance_matrix = torch.matmul(
            self._unbroadcasted_cov_factor, self._unbroadcasted_cov_factor.transpose(-1, -2)
        ) + torch.diag_embed(self._unbroadcasted_cov_diag)
        # 使用 expand 方法扩展 covariance_matrix 到新的形状
        return covariance_matrix.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )
    def precision_matrix(self):
        # 使用"Woodbury矩阵恒等式"来利用低秩形式：
        #     inv(W @ W.T + D) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D)
        # 其中 :math:`C` 是电容矩阵。
        Wt_Dinv = (
            self._unbroadcasted_cov_factor.mT
            / self._unbroadcasted_cov_diag.unsqueeze(-2)
        )
        # 解三角矩阵方程 self._capacitance_tril @ A = Wt_Dinv，求解得到 A
        A = torch.linalg.solve_triangular(self._capacitance_tril, Wt_Dinv, upper=False)
        # 计算精度矩阵 precision_matrix = diag(inv(D)) - A.T @ A
        precision_matrix = (
            torch.diag_embed(self._unbroadcasted_cov_diag.reciprocal()) - A.mT @ A
        )
        # 将精度矩阵扩展到指定形状
        return precision_matrix.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    def rsample(self, sample_shape=torch.Size()):
        # 扩展样本形状
        shape = self._extended_shape(sample_shape)
        W_shape = shape[:-1] + self.cov_factor.shape[-1:]
        # 生成服从标准正态分布的随机数 eps_W 和 eps_D
        eps_W = _standard_normal(W_shape, dtype=self.loc.dtype, device=self.loc.device)
        eps_D = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        # 返回随机样本
        return (
            self.loc
            + _batch_mv(self._unbroadcasted_cov_factor, eps_W)
            + self._unbroadcasted_cov_diag.sqrt() * eps_D
        )

    def log_prob(self, value):
        # 如果启用参数验证，验证样本值
        if self._validate_args:
            self._validate_sample(value)
        # 计算差值 diff = value - self.loc
        diff = value - self.loc
        # 计算马氏距离的负半对数概率密度函数值
        M = _batch_lowrank_mahalanobis(
            self._unbroadcasted_cov_factor,
            self._unbroadcasted_cov_diag,
            diff,
            self._capacitance_tril,
        )
        # 计算对数行列式的负半概率密度函数值
        log_det = _batch_lowrank_logdet(
            self._unbroadcasted_cov_factor,
            self._unbroadcasted_cov_diag,
            self._capacitance_tril,
        )
        # 返回对数概率密度函数值
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + log_det + M)

    def entropy(self):
        # 计算对数行列式的负半熵值
        log_det = _batch_lowrank_logdet(
            self._unbroadcasted_cov_factor,
            self._unbroadcasted_cov_diag,
            self._capacitance_tril,
        )
        # 计算熵值
        H = 0.5 * (self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + log_det)
        # 如果批次形状长度为零，则直接返回熵值，否则扩展熵值到指定批次形状
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
```