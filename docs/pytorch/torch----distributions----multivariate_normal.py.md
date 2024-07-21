# `.\pytorch\torch\distributions\multivariate_normal.py`

```py
# mypy: allow-untyped-defs
import math  # 导入数学模块

import torch  # 导入PyTorch库
from torch.distributions import constraints  # 导入约束模块
from torch.distributions.distribution import Distribution  # 导入分布基类
from torch.distributions.utils import _standard_normal, lazy_property  # 导入工具函数和延迟属性装饰器

__all__ = ["MultivariateNormal"]  # 模块公开的接口列表


def _batch_mv(bmat, bvec):
    r"""
    执行批量矩阵-向量乘积，具有兼容但不同的批量形状。

    该函数接受输入 `bmat`，包含 :math:`n \times n` 矩阵，
    和 `bvec`，包含长度为 :math:`n` 的向量。

    `bmat` 和 `bvec` 可以有任意数量的前导维度，这些维度对应批量形状。
    它们不一定假定具有相同的批量形状，只要可以广播即可。
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def _batch_mahalanobis(bL, bx):
    r"""
    计算基于因子分解 :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top` 的马氏距离的平方 :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`。

    接受 `bL` 和 `bx` 的批量输入。它们不一定假定具有相同的批量形状，但 `bL` 应该能够广播到 `bx` 的形状。
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # 假设 bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # 我们将使 bx 具有形状 (..., 1, j,  i, 1, n) 以应用批量三角求解
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # 用形状 (..., 1, i, j, 1, n) 重新整形 bx
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # 排列 bx 使其具有形状 (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # 形状为 b x n x n 的扁平化 L
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # 形状为 c x b x n 的扁平化 x
    flat_x_swap = flat_x.permute(1, 2, 0)  # 形状为 b x n x c 的交换 x
    M_swap = (
        torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)
    )  # 形状为 b x c 的交换 M
    M = M_swap.t()  # 形状为 c x b 的 M

    # 现在我们恢复上述 reshape 和 permute 操作的逆操作。
    permuted_M = M.reshape(bx.shape[:-1])  # 形状为 (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # 形状为 (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)
# 定义一个函数 `_precision_to_scale_tril`，用于将精度矩阵转换为下三角形矩阵的函数
def _precision_to_scale_tril(P):
    # 使用 Cholesky 分解计算矩阵 P 的逆的下三角形矩阵 Lf
    Lf = torch.linalg.cholesky(torch.flip(P, (-2, -1)))
    # 转置并翻转 Lf 得到 L_inv
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    # 创建 P.shape[-1] 维度的单位矩阵 Id，数据类型和设备与 P 一致
    Id = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    # 解三角形方程 L_inv @ L = Id，得到下三角形矩阵 L
    L = torch.linalg.solve_triangular(L_inv, Id, upper=False)
    # 返回下三角形矩阵 L
    return L


# 定义一个类 MultivariateNormal，表示多元正态分布
class MultivariateNormal(Distribution):
    r"""
    创建一个多元正态（高斯）分布，由均值向量和协方差矩阵参数化。

    多元正态分布可以通过正定协方差矩阵 :math:`\mathbf{\Sigma}`
    或者正定精度矩阵 :math:`\mathbf{\Sigma}^{-1}`
    或者下三角形矩阵 :math:`\mathbf{L}` 来参数化，其中对角线元素为正值，
    满足 :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`。
    可以通过例如协方差的 Cholesky 分解获得这个三角形矩阵。

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): 分布的均值
        covariance_matrix (Tensor): 正定的协方差矩阵
        precision_matrix (Tensor): 正定的精度矩阵
        scale_tril (Tensor): 协方差的下三角形矩阵，对角线元素为正值

    Note:
        只能指定其中之一: :attr:`covariance_matrix` 或 :attr:`precision_matrix` 或
        :attr:`scale_tril`。

        使用 :attr:`scale_tril` 将更高效: 所有的计算内部都基于 :attr:`scale_tril`。
        如果传入 :attr:`covariance_matrix` 或 :attr:`precision_matrix`，则仅用于计算
        相应的下三角形矩阵，使用 Cholesky 分解。

    """
    
    # 参数约束
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    
    # 支持
    support = constraints.real_vector
    
    # 具有 rsample 方法
    has_rsample = True

    def __init__(
        self,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
        ):
            # 检查 loc 张量是否至少是一维的，否则抛出数值错误
            if loc.dim() < 1:
                raise ValueError("loc must be at least one-dimensional.")
            # 检查 covariance_matrix、scale_tril、precision_matrix 中只能有一个被指定
            if (covariance_matrix is not None) + (scale_tril is not None) + (
                precision_matrix is not None
            ) != 1:
                raise ValueError(
                    "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
                )

            # 如果 scale_tril 被指定
            if scale_tril is not None:
                # 检查 scale_tril 矩阵是否至少是二维的，可以包含批量维度
                if scale_tril.dim() < 2:
                    raise ValueError(
                        "scale_tril matrix must be at least two-dimensional, "
                        "with optional leading batch dimensions"
                    )
                # 根据 loc 和 scale_tril 形状广播并扩展 scale_tril
                batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], loc.shape[:-1])
                self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
            # 如果 covariance_matrix 被指定
            elif covariance_matrix is not None:
                # 检查 covariance_matrix 是否至少是二维的，可以包含批量维度
                if covariance_matrix.dim() < 2:
                    raise ValueError(
                        "covariance_matrix must be at least two-dimensional, "
                        "with optional leading batch dimensions"
                    )
                # 根据 loc 和 covariance_matrix 形状广播并扩展 covariance_matrix
                batch_shape = torch.broadcast_shapes(
                    covariance_matrix.shape[:-2], loc.shape[:-1]
                )
                self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
            else:
                # precision_matrix 被指定
                # 检查 precision_matrix 是否至少是二维的，可以包含批量维度
                if precision_matrix.dim() < 2:
                    raise ValueError(
                        "precision_matrix must be at least two-dimensional, "
                        "with optional leading batch dimensions"
                    )
                # 根据 loc 和 precision_matrix 形状广播并扩展 precision_matrix
                batch_shape = torch.broadcast_shapes(
                    precision_matrix.shape[:-2], loc.shape[:-1]
                )
                self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))
            # 扩展 loc 到匹配的 batch_shape
            self.loc = loc.expand(batch_shape + (-1,))

            # 设置事件形状为 loc 的最后一个维度的形状
            event_shape = self.loc.shape[-1:]
            # 调用父类初始化函数，传递批量形状、事件形状以及验证参数的设置
            super().__init__(batch_shape, event_shape, validate_args=validate_args)

            # 根据不同的情况设置 _unbroadcasted_scale_tril 属性
            if scale_tril is not None:
                self._unbroadcasted_scale_tril = scale_tril
            elif covariance_matrix is not None:
                # 如果 covariance_matrix 被指定，则计算其 Cholesky 分解作为 _unbroadcasted_scale_tril
                self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
            else:  # precision_matrix is not None
                # 否则，将 precision_matrix 转换为 scale_tril 并赋给 _unbroadcasted_scale_tril
                self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)
    # 扩展当前对象为具有新的批处理形状的新对象，并返回该对象
    def expand(self, batch_shape, _instance=None):
        # 获取一个验证后的 MultivariateNormal 实例
        new = self._get_checked_instance(MultivariateNormal, _instance)
        # 将 batch_shape 转换为 torch.Size 对象
        batch_shape = torch.Size(batch_shape)
        # 计算新均值的形状，包括批处理形状和事件形状
        loc_shape = batch_shape + self.event_shape
        # 计算新协方差的形状，包括批处理形状、事件形状和事件形状
        cov_shape = batch_shape + self.event_shape + self.event_shape
        # 使用原始均值扩展到新的 loc_shape
        new.loc = self.loc.expand(loc_shape)
        # 复制未广播的 scale_tril 到新对象
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        # 如果当前对象有 covariance_matrix 属性，则扩展到新的 cov_shape
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        # 如果当前对象有 scale_tril 属性，则扩展到新的 cov_shape
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        # 如果当前对象有 precision_matrix 属性，则扩展到新的 cov_shape
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        # 调用父类的构造函数初始化新对象的批处理形状和事件形状，关闭参数验证
        super(MultivariateNormal, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        # 将参数验证的设置应用到新对象
        new._validate_args = self._validate_args
        # 返回新对象
        return new

    @lazy_property
    def scale_tril(self):
        # 将未广播的 scale_tril 扩展到当前对象的批处理形状、事件形状和事件形状
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        # 计算未广播的 scale_tril 的乘积，并扩展到当前对象的批处理形状、事件形状和事件形状
        return torch.matmul(
            self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.t()
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        # 计算未广播的 scale_tril 的逆矩阵，并扩展到当前对象的批处理形状、事件形状和事件形状
        return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        # 返回当前对象的均值
        return self.loc

    @property
    def mode(self):
        # 返回当前对象的众数（即均值）
        return self.loc

    @property
    def variance(self):
        # 返回当前对象的方差
        return (
            self._unbroadcasted_scale_tril.pow(2)
            .sum(-1)
            .expand(self._batch_shape + self._event_shape)
        )

    def rsample(self, sample_shape=torch.Size()):
        # 生成随机样本，形状由 sample_shape 指定
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

    def log_prob(self, value):
        # 计算给定值的对数概率密度函数
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        # 计算当前分布的熵
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
```