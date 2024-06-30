# `D:\src\scipysrc\scipy\scipy\stats\_stats_mstats_common.py`

```
import warnings
import numpy as np
from . import distributions  # 导入当前包中的distributions模块
from .._lib._bunch import _make_tuple_bunch  # 导入上级包中的_lib._bunch模块中的_make_tuple_bunch函数
from ._stats_pythran import siegelslopes as siegelslopes_pythran  # 导入当前包中的_stats_pythran模块中的siegelslopes，并重命名为siegelslopes_pythran

__all__ = ['_find_repeats', 'theilslopes', 'siegelslopes']

# 这不是一个 namedtuple，是为了向后兼容性而设。参见PR #12983
TheilslopesResult = _make_tuple_bunch('TheilslopesResult',  # 定义名为TheilslopesResult的命名元组
                                      ['slope', 'intercept',
                                       'low_slope', 'high_slope'])
SiegelslopesResult = _make_tuple_bunch('SiegelslopesResult',  # 定义名为SiegelslopesResult的命名元组
                                       ['slope', 'intercept'])


def theilslopes(y, x=None, alpha=0.95, method='separate'):
    r"""
    计算一组点（x, y）的Theil-Sen估计器。

    `theilslopes` 实现了一种鲁棒的线性回归方法。它将斜率计算为所有配对值之间斜率的中位数。

    Parameters
    ----------
    y : array_like
        因变量。
    x : array_like 或 None，可选
        自变量。如果为 None，则使用 ``arange(len(y))`` 代替。
    alpha : float，可选
        置信度，介于0和1之间。默认为95%置信度。
        注意 `alpha` 在0.5周围对称，即0.1和0.9都被解释为“找到90%置信区间”。
    method : {'joint', 'separate'}，可选
        用于计算截距估计的方法。
        支持以下方法：

            * 'joint': 使用 np.median(y - slope * x) 作为截距。
            * 'separate': 使用 np.median(y) - slope * np.median(x)
                          作为截距。

        默认为 'separate'。

        .. versionadded:: 1.8.0

    Returns
    -------
    result : ``TheilslopesResult`` 实例
        返回值是一个对象，具有以下属性：

        slope : float
            Theil 斜率。
        intercept : float
            Theil 线的截距。
        low_slope : float
            斜率的置信区间下界。
        high_slope : float
            斜率的置信区间上界。

    See Also
    --------
    siegelslopes : 使用重复中位数的类似技术

    Notes
    -----
    `theilslopes` 的实现遵循 [1]_。截距在 [1]_ 中没有定义，在这里定义为 ``median(y) - slope*median(x)``，
    这在 [3]_ 中有说明。文献中还存在其他截距的定义，例如在 [4]_ 中为 ``median(y - slope*x)``。
    可以通过参数 ``method`` 确定计算截距的方法。截距的置信区间未给出，因为 [1]_ 中未讨论这个问题。

    为了与旧版 SciPy 兼容，返回值的行为类似长度为4的命名元组，具有字段 ``slope``、``intercept``、
    """
    if method not in ['joint', 'separate']:
        raise ValueError("method must be either 'joint' or 'separate'."
                         f"'{method}' is invalid.")
    # 如果指定的方法不是 'joint' 或者 'separate'，则抛出值错误异常
    # 校验方法参数是否有效，确保只能是 'joint' 或者 'separate'
    
    # We copy both x and y so we can use _find_repeats.
    y = np.array(y, dtype=float, copy=True).ravel()
    # 将 y 转换为浮点数数组，并展平，确保数据格式正确
    
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.array(x, dtype=float, copy=True).ravel()
        if len(x) != len(y):
            raise ValueError(f"Incompatible lengths ! ({len(y)}<>{len(x)})")
    # 如果 x 为 None，则创建一个浮点数数组作为 x，长度与 y 相同
    # 否则，将 x 转换为浮点数数组，并展平，同时检查 x 和 y 的长度是否一致
    
    # Compute sorted slopes only when deltax > 0
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    # 仅在 deltax > 0 时计算排序后的斜率
    # 计算 x 之间的差值 deltax 和 y 之间的差值 deltay
    # 计算斜率 slopes，避免除以零
    
    if not slopes.size:
        msg = "All `x` coordinates are identical."
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    # 如果 slopes 数组为空，则发出运行时警告，表明所有的 x 坐标都是相同的
    
    slopes.sort()
    # 对斜率数组进行排序
    medslope = np.median(slopes)
    # 计算斜率的中位数 medslope
    
    if method == 'joint':
        medinter = np.median(y - medslope * x)
    else:
        medinter = np.median(y) - medslope * np.median(x)
    # 根据选择的方法计算截距 medinter
    # 如果方法为 'joint'，则计算 y - medslope * x 的中位数
    # 否则，计算 y 的中位数减去 medslope 乘以 x 的中位数
    
    # Now compute confidence intervals
    # 现在计算置信区间
    ```
    # 如果 alpha 大于 0.5，则取 1 减去 alpha 的值
    if alpha > 0.5:
        alpha = 1. - alpha

    # 计算正态分布的 alpha/2 分位点
    z = distributions.norm.ppf(alpha / 2.)
    
    # 使用 _find_repeats 函数找到 x 和 y 中的重复项，并分别赋值给 nxreps 和 nyreps
    _, nxreps = _find_repeats(x)
    _, nyreps = _find_repeats(y)
    
    # slopes 的长度作为 nt，表示样本数目，对应 Sen (1968) 中的 N
    nt = len(slopes)       # N in Sen (1968)
    # y 的长度作为 ny，表示观测数目，对应 Sen (1968) 中的 n
    ny = len(y)            # n in Sen (1968)
    
    # 根据 Sen (1968) 中的公式 2.6 计算 sigsq
    sigsq = 1/18. * (ny * (ny-1) * (2*ny+5) -
                     sum(k * (k-1) * (2*k + 5) for k in nxreps) -
                     sum(k * (k-1) * (2*k + 5) for k in nyreps))
    
    # 尝试计算标准差 sigma
    try:
        sigma = np.sqrt(sigsq)
        # 计算上下界索引 Ru 和 Rl，用于构建斜率 slopes 的置信区间
        Ru = min(int(np.round((nt - z*sigma)/2.)), len(slopes)-1)
        Rl = max(int(np.round((nt + z*sigma)/2.)) - 1, 0)
        # 取出斜率 slopes 的上下界 delta
        delta = slopes[[Rl, Ru]]
    except (ValueError, IndexError):
        # 处理异常情况，如果计算失败则置 delta 为 NaN
        delta = (np.nan, np.nan)

    # 返回 Theil-Sen 斜率估计的结果对象 TheilslopesResult
    return TheilslopesResult(slope=medslope, intercept=medinter,
                             low_slope=delta[0], high_slope=delta[1])
# 寻找数组中的重复值并返回它们的频率
def _find_repeats(arr):
    # 这个函数假设可能会修改其输入。
    if len(arr) == 0:
        # 如果数组为空，返回一个浮点数和整数类型的零数组
        return np.array(0, np.float64), np.array(0, np.intp)

    # XXX 这种转换以前是 Fortran 实现所需的，我们应该放弃它吗？
    # 将输入数组转换为 numpy 的 float64 类型，并将其展平后排序
    arr = np.asarray(arr, np.float64).ravel()
    arr.sort()

    # 从 NumPy 1.9 版本的 np.unique 函数获取
    # 创建一个布尔数组，标记数组中的唯一值位置
    change = np.concatenate(([True], arr[1:] != arr[:-1]))
    # 获取唯一值数组
    unique = arr[change]
    # 计算每个唯一值的出现次数
    change_idx = np.concatenate(np.nonzero(change) + ([arr.size],))
    freq = np.diff(change_idx)
    atleast2 = freq > 1
    # 返回出现频率大于1的唯一值数组及其频率数组
    return unique[atleast2], freq[atleast2]


# 计算 Siegel 估计器以拟合点集 (x, y) 的斜率和截距
def siegelslopes(y, x=None, method="hierarchical"):
    r"""
    计算 Siegel 估计器以拟合点集 (x, y) 的斜率和截距。

    `siegelslopes` 实现了一种使用重复中位数（参见 [1]_）进行鲁棒线性回归的方法，
    用于拟合点集 (x, y)。该方法对异常值具有鲁棒性，渐近破坏点为 50%。

    Parameters
    ----------
    y : array_like
        因变量。
    x : array_like 或 None, optional
        自变量。如果为 None，则使用 ``arange(len(y))``。
    method : {'hierarchical', 'separate'}
        如果为 'hierarchical'，使用估计的斜率 ``slope`` 估计截距（默认选项）。
        如果为 'separate'，独立估计截距。详见 Notes。

    Returns
    -------
    result : ``SiegelslopesResult`` 实例
        返回值是一个对象，具有以下属性：

        slope : float
            回归线斜率的估计值。
        intercept : float
            回归线截距的估计值。

    See Also
    --------
    theilslopes : 不使用重复中位数的类似技术。

    Notes
    -----
    对于长度为 `n` 的数组 `y`，计算 `m_j`，其为从点 ``(x[j], y[j])`` 到所有其他 `n-1` 个点的斜率中位数。
    ``slope`` 是所有斜率 ``m_j`` 的中位数。
    [1]_ 中提供了两种方法来估计截距，可以通过参数 ``method`` 选择。
    hierarchical 方法使用估计的斜率 ``slope``，计算 ``intercept`` 为 ``y - slope*x`` 的中位数。
    另一种方法独立估计截距：对于每个点 ``(x[j], y[j])``，计算通过剩余点的所有 `n-1` 条线的截距，并取中位数 ``i_j``。
    ``intercept`` 是所有 ``i_j`` 的中位数。

    此实现计算大小为 `n` 的向量的中位数 `n` 次，对于大向量可能效率较低。有更高效的算法（见 [2]_），这里未实现。

    为了与旧版本的 SciPy 兼容，返回值表现得像长度为 2 的命名元组，具有字段 ``slope`` 和 ``intercept``，因此可以继续写成：

        slope, intercept = siegelslopes(y, x)

    References
    ----------
    ```
    """
    如果指定的方法不是 'hierarchical' 或 'separate'，则抛出数值错误异常。
    """
    if method not in ['hierarchical', 'separate']:
        raise ValueError("method can only be 'hierarchical' or 'separate'")
    
    """
    将 y 转换为一维数组。
    """
    y = np.asarray(y).ravel()
    
    """
    如果 x 为 None，则创建一个浮点型的数组作为 x，长度与 y 相同。
    否则，将 x 转换为一维浮点型数组，并检查其长度是否与 y 相同，不同则抛出数值错误异常。
    """
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != len(y):
            raise ValueError(f"Incompatible lengths ! ({len(y)}<>{len(x)})")
    
    """
    确定 x、y、np.float32 中的最终数据类型。
    """
    dtype = np.result_type(x, y, np.float32)  # use at least float32
    
    """
    将 y 和 x 转换为指定的数据类型。
    """
    y, x = y.astype(dtype), x.astype(dtype)
    
    """
    使用 siegelslopes_pythran 函数计算 Siegel 斜率和截距。
    """
    medslope, medinter = siegelslopes_pythran(y, x, method)
    
    """
    返回 SiegelslopesResult 对象，包含计算得到的斜率和截距。
    """
    return SiegelslopesResult(slope=medslope, intercept=medinter)
```