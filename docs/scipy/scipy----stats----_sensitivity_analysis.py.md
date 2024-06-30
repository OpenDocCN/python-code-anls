# `D:\src\scipysrc\scipy\scipy\stats\_sensitivity_analysis.py`

```
from __future__ import annotations
# 引入未来支持的注解功能，使得可以在类型提示中使用类型本身作为返回类型

import inspect
# 导入inspect模块，用于获取对象信息

from dataclasses import dataclass
# 导入dataclass装饰器，用于声明数据类

from typing import (
    Callable, Literal, Protocol, TYPE_CHECKING
)
# 导入类型提示相关的模块和类，包括Callable、Literal和Protocol

import numpy as np
# 导入NumPy库，用于科学计算

from scipy.stats._common import ConfidenceInterval
# 导入置信区间相关的模块

from scipy.stats._qmc import check_random_state
# 导入检查随机状态的模块

from scipy.stats._resampling import BootstrapResult
# 导入BootstrapResult类，用于统计重采样

from scipy.stats import qmc, bootstrap
# 导入qmc和bootstrap模块，用于蒙特卡洛和Bootstrap方法的统计分析

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import DecimalNumber, IntNumber, SeedType
# 如果在类型检查模式下，导入特定的类型

__all__ = [
    'sobol_indices'
]
# 模块公开的接口，限制为'sobol_indices'

def f_ishigami(x: npt.ArrayLike) -> np.ndarray:
    r"""Ishigami function.

    .. math::

        Y(\mathbf{x}) = \sin x_1 + 7 \sin^2 x_2 + 0.1 x_3^4 \sin x_1

    with :math:`\mathbf{x} \in [-\pi, \pi]^3`.

    Parameters
    ----------
    x : array_like ([x1, x2, x3], n)

    Returns
    -------
    f : array_like (n,)
        Function evaluation.

    References
    ----------
    .. [1] Ishigami, T. and T. Homma. "An importance quantification technique
       in uncertainty analysis for computer models." IEEE,
       :doi:`10.1109/ISUMA.1990.151285`, 1990.
    """
    x = np.atleast_2d(x)
    # 确保x至少为二维数组
    f_eval = (
        np.sin(x[0])
        + 7 * np.sin(x[1])**2
        + 0.1 * (x[2]**4) * np.sin(x[0])
    )
    # 计算Ishigami函数的值
    return f_eval
    # 返回函数值


def sample_A_B(
    n: IntNumber,
    dists: list[PPFDist],
    random_state: SeedType = None
) -> np.ndarray:
    """Sample two matrices A and B.

    Uses a Sobol' sequence with 2`d` columns to have 2 uncorrelated matrices.
    This is more efficient than using 2 random draw of Sobol'.
    See sec. 5 from [1]_.

    Output shape is (d, n).

    References
    ----------
    .. [1] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
       S. Tarantola. "Variance based sensitivity analysis of model
       output. Design and estimator for the total sensitivity index."
       Computer Physics Communications, 181(2):259-270,
       :doi:`10.1016/j.cpc.2009.09.018`, 2010.
    """
    d = len(dists)
    # 确定分布列表的长度
    A_B = qmc.Sobol(d=2*d, seed=random_state, bits=64).random(n).T
    # 使用Sobol'序列生成两个矩阵A和B
    A_B = A_B.reshape(2, d, -1)
    # 调整形状为(2, d, n)，其中n为样本数
    try:
        for d_, dist in enumerate(dists):
            A_B[:, d_] = dist.ppf(A_B[:, d_])
    except AttributeError as exc:
        message = "Each distribution in `dists` must have method `ppf`."
        raise ValueError(message) from exc
    # 尝试将Sobol'序列转换为指定分布的样本，处理属性错误异常
    return A_B
    # 返回A_B矩阵


def sample_AB(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """AB matrix.

    AB: rows of B into A. Shape (d, d, n).
    - Copy A into d "pages"
    - In the first page, replace 1st rows of A with 1st row of B.
    ...
    - In the dth page, replace dth row of A with dth row of B.
    - return the stack of pages
    """
    d, n = A.shape
    # 获取矩阵A的形状信息
    AB = np.tile(A, (d, 1, 1))
    # 使用np.tile复制A为(d, 1, 1)形状的多页矩阵AB
    i = np.arange(d)
    # 创建长度为d的索引数组i
    AB[i, i] = B[i]
    # 替换AB中每一页的对应行
    return AB
    # 返回AB矩阵


def saltelli_2010(
    f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Saltelli2010 formulation.
    # 计算基于 A 和 B 输出的经验方差，因为它们是独立的。AB 的输出不独立，不能用于计算。
    var = np.var([f_A, f_B], axis=(0, -1))
    
    # 将方差用作比率，得到方差的比率，对应公式 2
    s = np.mean(f_B * (f_AB - f_A), axis=-1) / var  # 表 2 (b)
    
    # 总敏感度指数的一半，对应公式 2 (f)
    st = 0.5 * np.mean((f_A - f_AB) ** 2, axis=-1) / var  # 表 2 (f)
    
    # 返回 s 和 st 的转置，即第一阶和总阶 Sobol' 指数
    return s.T, st.T
@dataclass
class BootstrapSobolResult:
    first_order: BootstrapResult
    total_order: BootstrapResult

@dataclass
class SobolResult:
    first_order: np.ndarray
    total_order: np.ndarray
    _indices_method: Callable
    _f_A: np.ndarray
    _f_B: np.ndarray
    _f_AB: np.ndarray
    _A: np.ndarray | None = None
    _B: np.ndarray | None = None
    _AB: np.ndarray | None = None
    _bootstrap_result: BootstrapResult | None = None

    def bootstrap(
        self,
        confidence_level: DecimalNumber = 0.95,
        n_resamples: IntNumber = 999
    ) -> BootstrapSobolResult:
        """Bootstrap Sobol' indices to provide confidence intervals.

        Parameters
        ----------
        confidence_level : float, default: ``0.95``
            The confidence level of the confidence intervals.
        n_resamples : int, default: ``999``
            The number of resamples performed to form the bootstrap
            distribution of the indices.

        Returns
        -------
        res : BootstrapSobolResult
            Bootstrap result containing the confidence intervals and the
            bootstrap distribution of the indices.

            An object with attributes:

            first_order : BootstrapResult
                Bootstrap result of the first order indices.
            total_order : BootstrapResult
                Bootstrap result of the total order indices.
            See `BootstrapResult` for more details.

        """
        # Define a statistic function for bootstrap resampling
        def statistic(idx):
            # Extract relevant data for the statistic calculation
            f_A_ = self._f_A[:, idx]
            f_B_ = self._f_B[:, idx]
            f_AB_ = self._f_AB[..., idx]
            # Compute indices using the specified method
            return self._indices_method(f_A_, f_B_, f_AB_)

        # Number of samples
        n = self._f_A.shape[1]

        # Perform bootstrap resampling
        res = bootstrap(
            [np.arange(n)], statistic=statistic, method="BCa",
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            bootstrap_result=self._bootstrap_result
        )
        # Store the bootstrap result for potential future use
        self._bootstrap_result = res

        # Extract confidence intervals and distributions for first and total order indices
        first_order = BootstrapResult(
            confidence_interval=ConfidenceInterval(
                res.confidence_interval.low[0], res.confidence_interval.high[0]
            ),
            bootstrap_distribution=res.bootstrap_distribution[0],
            standard_error=res.standard_error[0],
        )
        total_order = BootstrapResult(
            confidence_interval=ConfidenceInterval(
                res.confidence_interval.low[1], res.confidence_interval.high[1]
            ),
            bootstrap_distribution=res.bootstrap_distribution[1],
            standard_error=res.standard_error[1],
        )

        # Return BootstrapSobolResult with the computed results
        return BootstrapSobolResult(
            first_order=first_order, total_order=total_order
        )


class PPFDist(Protocol):
    @property
    def ppf(self) -> Callable[..., float]:
        ...


def sobol_indices(
    *,
    func: Callable[[np.ndarray], npt.ArrayLike] |
          dict[Literal['f_A', 'f_B', 'f_AB'], np.ndarray],
    n: IntNumber,
    dists: list[PPFDist] | None = None,
):
    # Function to compute Sobol' indices given a function or dictionary of arrays
    pass
    method: Callable | Literal['saltelli_2010'] = 'saltelli_2010',
    random_state: SeedType = None


    method: Callable | Literal['saltelli_2010'] = 'saltelli_2010',
    # 参数method的类型注释，指定为Callable或Literal类型，限定取值为'saltelli_2010'字符串，默认为'saltelli_2010'
    random_state: SeedType = None
    # 参数random_state的类型注释，指定为SeedType类型，默认为None
# 定义函数签名，函数用于计算 Sobol' 全局敏感性指数
def sensitivity_indices(func: Callable or dict(str, array_like),
                        n: int,
                        dists: Optional[list(distributions)] = None,
                        method: Callable or str = 'saltelli_2010',
                        random_state: Optional[Union[None, int, np.random.Generator]] = None) -> SobolResult:
    r"""Global sensitivity indices of Sobol'.

    Parameters
    ----------
    func : callable or dict(str, array_like)
        如果 `func` 是一个可调用对象，用于计算 Sobol' 指数。
        其签名必须为::

            func(x: ArrayLike) -> ArrayLike

        其中 `x` 的形状为 ``(d, n)``，输出的形状为 ``(s, n)``，其中：

        - ``d`` 是 `func` 的输入维度（输入变量的数量），
        - ``s`` 是 `func` 的输出维度（输出变量的数量），
        - ``n`` 是样本数（见下面的 `n` 参数）。

        函数计算结果必须是有限的。

        如果 `func` 是一个字典，则包含来自三个不同数组的函数评估值。键必须为：``f_A``、``f_B`` 和 ``f_AB``。
        ``f_A`` 和 ``f_B`` 应具有形状 ``(s, n)``，``f_AB`` 应具有形状 ``(d, s, n)``。
        这是一个高级特性，误用可能导致分析错误。
    n : int
        用于生成矩阵 ``A`` 和 ``B`` 的样本数。必须是2的幂次方。
        `func` 评估的总点数将为 ``n*(d+2)``。
    dists : list(distributions), optional
        每个参数的分布列表。参数的分布取决于应用程序，应谨慎选择。
        假设参数是独立分布的，即它们的值之间没有约束或关系。

        分布必须是具有 ``ppf`` 方法的类的实例。

        如果 `func` 是可调用对象，则必须指定；否则将被忽略。
    method : Callable or str, default: 'saltelli_2010'
        用于计算第一和总 Sobol' 指数的方法。

        如果是可调用对象，其签名必须为::

            func(f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray)
            -> Tuple[np.ndarray, np.ndarray]

        其中 ``f_A, f_B`` 的形状为 ``(s, n)``，``f_AB`` 的形状为 ``(d, s, n)``。
        这些数组包含来自三个不同样本集的函数评估。输出是形状为 ``(s, d)`` 的第一和总指数的元组。
        这是一个高级特性，误用可能导致分析错误。
    random_state : {None, int, `numpy.random.Generator`}, optional
        如果 `random_state` 是整数或 None，则使用 ``np.random.default_rng(random_state)`` 创建一个新的 `numpy.random.Generator`。
        如果 `random_state` 已经是 ``Generator`` 实例，则使用提供的实例。

    Returns
    -------
    res : SobolResult
        # 定义变量 res，类型为 SobolResult，包含以下属性和方法：

        first_order : ndarray of shape (s, d)
            # 二维数组，存储一阶 Sobol' 指数，形状为 (s, d)，其中 s 是样本数量，d 是参数数量。

        total_order : ndarray of shape (s, d)
            # 二维数组，存储总阶 Sobol' 指数，形状为 (s, d)，其中 s 是样本数量，d 是参数数量。

        And method:

        bootstrap(confidence_level: float, n_resamples: int)
        -> BootstrapSobolResult
            # 方法名 bootstrap，接受置信水平 confidence_level 和重采样次数 n_resamples 作为参数，返回 BootstrapSobolResult 对象。
            # 该方法用于计算指数的置信区间。详细信息请参阅 `scipy.stats.bootstrap`。

            A method providing confidence intervals on the indices.
            See `scipy.stats.bootstrap` for more details.
            # 提供指数的置信区间计算方法。详细信息请参阅 `scipy.stats.bootstrap`。

            The bootstrapping is done on both first and total order indices,
            and they are available in `BootstrapSobolResult` as attributes
            ``first_order`` and ``total_order``.
            # 对一阶和总阶指数进行自助法重采样，结果存储在 `BootstrapSobolResult` 对象中，可以通过 ``first_order`` 和 ``total_order`` 属性访问。

    Notes
    -----
    # 注意事项部分开始

    The Sobol' method [1]_, [2]_ is a variance-based Sensitivity Analysis which
    obtains the contribution of each parameter to the variance of the
    quantities of interest (QoIs; i.e., the outputs of `func`).
    # Sobol' 方法是基于方差的敏感性分析方法，用于获取每个参数对感兴趣量（QoIs，即 `func` 输出）方差的贡献。

    Respective contributions can be used to rank the parameters and
    also gauge the complexity of the model by computing the
    model's effective (or mean) dimension.
    # 可以使用各自的贡献来对参数进行排序，并通过计算模型的有效维度（或平均维度）来衡量模型的复杂性。

    .. note::

        Parameters are assumed to be independently distributed. Each
        parameter can still follow any distribution. In fact, the distribution
        is very important and should match the real distribution of the
        parameters.
        # 注意：假设参数是独立分布的。每个参数仍然可以遵循任何分布。实际上，分布非常重要，应该与参数的真实分布匹配。

    It uses a functional decomposition of the variance of the function to
    explore

    .. math::

        \mathbb{V}(Y) = \sum_{i}^{d} \mathbb{V}_i (Y) + \sum_{i<j}^{d}
        \mathbb{V}_{ij}(Y) + ... + \mathbb{V}_{1,2,...,d}(Y),

    introducing conditional variances:

    .. math::

        \mathbb{V}_i(Y) = \mathbb{\mathbb{V}}[\mathbb{E}(Y|x_i)]
        \qquad
        \mathbb{V}_{ij}(Y) = \mathbb{\mathbb{V}}[\mathbb{E}(Y|x_i x_j)]
        - \mathbb{V}_i(Y) - \mathbb{V}_j(Y),
        # 引入条件方差，通过函数方差的功能分解来探索条件方差。

    Sobol' indices are expressed as

    .. math::

        S_i = \frac{\mathbb{V}_i(Y)}{\mathbb{V}[Y]}
        \qquad
        S_{ij} =\frac{\mathbb{V}_{ij}(Y)}{\mathbb{V}[Y]}.
        # Sobol' 指数定义为函数方差分解的结果。

    :math:`S_{i}` corresponds to the first-order term which apprises the
    contribution of the i-th parameter, while :math:`S_{ij}` corresponds to the
    second-order term which informs about the contribution of interactions
    between the i-th and the j-th parameters. These equations can be
    generalized to compute higher order terms; however, they are expensive to
    compute and their interpretation is complex.
    # :math:`S_{i}` 对应一阶项，反映第 i 个参数的贡献，而 :math:`S_{ij}` 对应二阶项，反映第 i 和 j 个参数之间交互的贡献。

    Total order indices represent the global contribution of the parameters
    to the variance of the QoI and are defined as:

    .. math::

        S_{T_i} = S_i + \sum_j S_{ij} + \sum_{j,k} S_{ijk} + ...
        = 1 - \frac{\mathbb{V}[\mathbb{E}(Y|x_{\sim i})]}{\mathbb{V}[Y]}.
        # 总阶指数代表参数对 QoI 方差的整体贡献，定义如上。

    First order indices sum to at most 1, while total order indices sum to at
    least 1. If there are no interactions, then first and total order indices
    are equal, and both first and total order indices sum to 1.
    # 一阶指数之和最多为 1，总阶指数之和至少为 1。如果没有交互作用，则一阶和总阶指数相等，且它们之和为 1。
    # 引入必要的库和模块
    import numpy as np
    from scipy.stats import sobol_indices, uniform
    
    # 创建一个随机数生成器对象
    rng = np.random.default_rng()
    
    # 定义 Ishigami 函数，接受一个三维向量 x 作为输入
    def f_ishigami(x):
        # 计算 Ishigami 函数的值
        f_eval = (
            np.sin(x[0])
            + 7 * np.sin(x[1])**2
            + 0.1 * (x[2]**4) * np.sin(x[0])
        )
        return f_eval
    
    # 计算 Sobol' 灵敏度指数
    indices = sobol_indices(
        func=f_ishigami,  # 待分析的函数
        n=1024,            # 样本点数目
        dists=[            # 每个维度上的分布
            uniform(loc=-np.pi, scale=2*np.pi),
            uniform(loc=-np.pi, scale=2*np.pi),
            uniform(loc=-np.pi, scale=2*np.pi)
        ],
        random_state=rng    # 随机数生成器的状态
    )
    
    # 打印一阶 Sobol' 指数，表示每个输入变量对输出的单独贡献
    indices.first_order
    >>> indices.total_order
    array([0.56122127, 0.44287857, 0.24229595])
    
    
    # 获取 indices 对象中的 total_order 属性，它是一个包含三个浮点数的数组
    >>> boot = indices.bootstrap()
    
    
    # 调用 indices 对象的 bootstrap() 方法，返回一个 boot 对象，用于后续的统计分析
    >>> import matplotlib.pyplot as plt
    >>> fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    
    
    # 导入 matplotlib 库，并创建一个包含两个子图的图表对象 fig，以及这两个子图的 Axes 对象数组 axs
    >>> _ = axs[0].errorbar(
    ...     [1, 2, 3], indices.first_order, fmt='o',
    ...     yerr=[
    ...         indices.first_order - boot.first_order.confidence_interval.low,
    ...         boot.first_order.confidence_interval.high - indices.first_order
    ...     ],
    ... )
    
    
    # 在 axs[0] 子图上绘制误差条形图，显示 indices.first_order 的值，以及其置信区间的上下界
    >>> axs[0].set_ylabel("First order Sobol' indices")
    >>> axs[0].set_xlabel('Input parameters')
    >>> axs[0].set_xticks([1, 2, 3])
    
    
    # 设置 axs[0] 子图的纵坐标标签、横坐标标签和刻度
    >>> _ = axs[1].errorbar(
    ...     [1, 2, 3], indices.total_order, fmt='o',
    ...     yerr=[
    ...         indices.total_order - boot.total_order.confidence_interval.low,
    ...         boot.total_order.confidence_interval.high - indices.total_order
    ...     ],
    ... )
    
    
    # 在 axs[1] 子图上绘制误差条形图，显示 indices.total_order 的值，以及其置信区间的上下界
    >>> axs[1].set_ylabel("Total order Sobol' indices")
    >>> axs[1].set_xlabel('Input parameters')
    >>> axs[1].set_xticks([1, 2, 3])
    
    
    # 设置 axs[1] 子图的纵坐标标签、横坐标标签和刻度
    >>> plt.tight_layout()
    >>> plt.show()
    
    
    # 调整图表的布局使其紧凑，并展示图表
    
    
    .. note::
    
        By default, `scipy.stats.uniform` has support ``[0, 1]``.
        Using the parameters ``loc`` and ``scale``, one obtains the uniform
        distribution on ``[loc, loc + scale]``.
    
    
    # 提示信息：scipy.stats.uniform 默认支持区间 [0, 1]，通过参数 loc 和 scale 可以获得区间 [loc, loc + scale] 上的均匀分布
    # 这段文本解释了 scipy 中 uniform 分布的默认行为和如何自定义分布的范围
    
    
    This result is particularly interesting because the first order index
    :math:`S_{x_3} = 0` whereas its total order is :math:`S_{T_{x_3}} = 0.244`.
    This means that higher order interactions with :math:`x_3` are responsible
    for the difference. Almost 25% of the observed variance
    on the QoI is due to the correlations between :math:`x_3` and :math:`x_1`,
    although :math:`x_3` by itself has no impact on the QoI.
    
    
    # 对 Sobol 指数的计算结果进行解释：虽然 :math:`x_3` 本身对输出质量指标 (QoI) 没有影响，但其与 :math:`x_1` 的相关性导致了观察方差的近 25%
    
    
    The following gives a visual explanation of Sobol' indices on this
    function. Let's generate 1024 samples in :math:`[-\pi, \pi]^3` and
    calculate the value of the output.
    
    
    # 提示继续讨论 Sobol' 指数在函数中的视觉解释：生成 1024 个样本，范围为 :math:`[-\pi, \pi]^3`，计算输出值
    >>> from scipy.stats import qmc
    >>> n_dim = 3
    >>> p_labels = ['$x_1$', '$x_2$', '$x_3$']
    >>> sample = qmc.Sobol(d=n_dim, seed=rng).random(1024)
    
    
    # 导入 QMC (Quasi-Monte Carlo) 相关库，生成 3 维 Sobol 序列的 1024 个随机样本
    >>> sample = qmc.scale(
    ...     sample=sample,
    ...     l_bounds=[-np.pi, -np.pi, -np.pi],
    ...     u_bounds=[np.pi, np.pi, np.pi]
    ... )
    
    
    # 将生成的样本进行缩放，将其范围调整为 :math:`[-\pi, \pi]^3`
    >>> output = f_ishigami(sample.T)
    
    
    # 将样本转置后作为 f_ishigami 函数的输入，计算输出值
    >>> fig, ax = plt.subplots(1, n_dim, figsize=(12, 4))
    >>> for i in range(n_dim):
    ...     xi = sample[:, i]
    ...     ax[i].scatter(xi, output, marker='+')
    ...     ax[i].set_xlabel(p_labels[i])
    
    
    # 创建一个包含 n_dim 个子图的图表对象 fig，每个子图分别表示一个参数的散点图
    # 遍历每个维度，绘制参数 xi 和输出 output 的散点图，并设置子图的横坐标标签为对应的参数名称
    >>> ax[0].set_ylabel('Y')
    >>> plt.tight_layout()
    >>> plt.show()
    
    
    # 设置第一个子图的纵坐标标签为 'Y'，调整图表布局使其紧凑，并展示图表
    
    
    Now Sobol' goes a step further:
    by conditioning the output value by given values of the parameter
    
    
    # 这段文字介绍 Sobol' 指数进一步的分析步骤，通过对参数值的给定进行输出值的条件分析
    """
    Calculate Sobol' indices for a given set of samples and outputs.

    Parameters:
    - n: int
        Number of samples.
    - method: str or callable
        Method used to calculate Sobol' indices.

    Raises:
    - ValueError: If 'n' is not a power of 2.
    - ValueError: If 'method' is not a valid string or callable.

    Returns:
    - Sobol' indices for the given samples and outputs.
    """

    # Ensure reproducibility of results by setting the random state
    random_state = check_random_state(random_state)

    # Check if 'n' is a power of 2
    n_ = int(n)
    if not (n_ & (n_ - 1) == 0) or n != n_:
        raise ValueError(
            "The balance properties of Sobol' points require 'n' "
            "to be a power of 2."
        )
    n = n_

    # Validate and assign the method for calculating Sobol' indices
    if not callable(method):
        indices_methods: dict[str, Callable] = {
            "saltelli_2010": saltelli_2010,
        }
        try:
            method = method.lower()  # type: ignore[assignment]
            indices_method_ = indices_methods[method]
        except KeyError as exc:
            message = (
                f"{method!r} is not a valid 'method'. It must be one of"
                f" {set(indices_methods)!r} or a callable."
            )
            raise ValueError(message) from exc
    else:
        indices_method_ = method
        sig = inspect.signature(indices_method_)

        # Check if the callable method has required signature
        if set(sig.parameters) != {'f_A', 'f_B', 'f_AB'}:
            message = (
                "If 'method' is a callable, it must have the following"
                f" signature: {inspect.signature(saltelli_2010)}"
            )
            raise ValueError(message)
    def indices_method(f_A, f_B, f_AB):
        """Wrap indices method to ensure proper output dimension.

        1D when single output, 2D otherwise.
        """
        # 调用 indices_method_ 函数并确保输出的维度正确
        return np.squeeze(indices_method_(f_A=f_A, f_B=f_B, f_AB=f_AB))

    if callable(func):
        if dists is None:
            # 如果 func 是可调用的且 dists 未定义，则引发 ValueError 异常
            raise ValueError(
                "'dists' must be defined when 'func' is a callable."
            )

        def wrapped_func(x):
            # 将 func 函数的输出至少转换为二维数组
            return np.atleast_2d(func(x))

        # 从分布中采样数据 A 和 B
        A, B = sample_A_B(n=n, dists=dists, random_state=random_state)
        # 从 A 和 B 中采样得到 AB
        AB = sample_AB(A=A, B=B)

        # 对 A 应用 wrapped_func 函数
        f_A = wrapped_func(A)

        # 检查 func 输出的形状是否为 (s, -1)，其中 s 是输出的数量
        if f_A.shape[1] != n:
            # 如果不是，引发 ValueError 异常
            raise ValueError(
                "'func' output should have a shape ``(s, -1)`` with ``s`` "
                "the number of output."
            )

        def funcAB(AB):
            # 将 AB 的轴移动以适应 wrapped_func 函数的要求，并返回重新组织后的结果
            d, d, n = AB.shape
            AB = np.moveaxis(AB, 0, -1).reshape(d, n*d)
            f_AB = wrapped_func(AB)
            return np.moveaxis(f_AB.reshape((-1, n, d)), -1, 0)

        # 对 B 应用 wrapped_func 函数
        f_B = wrapped_func(B)
        # 对 AB 应用 funcAB 函数
        f_AB = funcAB(AB)
    else:
        message = (
            "When 'func' is a dictionary, it must contain the following "
            "keys: 'f_A', 'f_B' and 'f_AB'."
            "'f_A' and 'f_B' should have a shape ``(s, n)`` and 'f_AB' "
            "should have a shape ``(d, s, n)``."
        )
        try:
            # 尝试从 func 字典中获取 'f_A'、'f_B' 和 'f_AB' 的值，并至少将它们转换为二维数组
            f_A, f_B, f_AB = np.atleast_2d(
                func['f_A'], func['f_B'], func['f_AB']
            )
        except KeyError as exc:
            # 如果缺少键，则引发 ValueError 异常
            raise ValueError(message) from exc

        # 检查 'f_A' 的列数是否为 n，且 'f_A' 和 'f_B' 的形状是否相同，以及 'f_AB' 的最后一个维度是否为 n 的倍数
        if f_A.shape[1] != n or f_A.shape != f_B.shape or \
                f_AB.shape == f_A.shape or f_AB.shape[-1] % n != 0:
            # 如果不符合要求，则引发 ValueError 异常
            raise ValueError(message)

    # 根据均值对 f_A、f_B 和 f_AB 进行归一化处理
    mean = np.mean([f_A, f_B], axis=(0, -1)).reshape(-1, 1)
    f_A -= mean
    f_B -= mean
    f_AB -= mean

    # 计算敏感度指数
    # 忽略常量输出的警告，因为方差为零
    with np.errstate(divide='ignore', invalid='ignore'):
        first_order, total_order = indices_method(f_A=f_A, f_B=f_B, f_AB=f_AB)

    # 如果方差为零，则将一阶和总体敏感度指数置为零
    first_order[~np.isfinite(first_order)] = 0
    total_order[~np.isfinite(total_order)] = 0

    # 组织结果字典
    res = dict(
        first_order=first_order,
        total_order=total_order,
        _indices_method=indices_method,
        _f_A=f_A,
        _f_B=f_B,
        _f_AB=f_AB
    )

    if callable(func):
        # 如果 func 是可调用的，则更新结果字典，添加 A、B 和 AB 的信息
        res.update(
            dict(
                _A=A,
                _B=B,
                _AB=AB,
            )
        )

    # 返回 SobolResult 类的实例，包含计算的结果
    return SobolResult(**res)
```