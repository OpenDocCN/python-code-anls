# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\common.py`

```
from itertools import groupby  # 导入 itertools 模块中的 groupby 函数，用于按条件分组
from warnings import warn  # 导入 warnings 模块中的 warn 函数，用于发出警告
import numpy as np  # 导入 NumPy 库，用于数值计算
from scipy.sparse import find, coo_matrix  # 从 SciPy 库中导入 find 和 coo_matrix 函数

EPS = np.finfo(float).eps  # 计算机中浮点数的最小增量，作为误差限

def validate_first_step(first_step, t0, t_bound):
    """Assert that first_step is valid and return it."""
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")  # 如果 first_step 小于等于 0，则抛出值错误
    if first_step > np.abs(t_bound - t0):
        raise ValueError("`first_step` exceeds bounds.")  # 如果 first_step 超出了界限范围，则抛出值错误
    return first_step  # 返回经验证后的 first_step

def validate_max_step(max_step):
    """Assert that max_Step is valid and return it."""
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")  # 如果 max_step 小于等于 0，则抛出值错误
    return max_step  # 返回经验证后的 max_step

def warn_extraneous(extraneous):
    """Display a warning for extraneous keyword arguments.

    The initializer of each solver class is expected to collect keyword
    arguments that it doesn't understand and warn about them. This function
    prints a warning for each key in the supplied dictionary.

    Parameters
    ----------
    extraneous : dict
        Extraneous keyword arguments
    """
    if extraneous:
        warn("The following arguments have no effect for a chosen solver: {}."
             .format(", ".join(f"`{x}`" for x in extraneous)),
             stacklevel=3)  # 如果存在无效关键字参数，则发出警告并指出其无效性

def validate_tol(rtol, atol, n):
    """Validate tolerance values."""

    if np.any(rtol < 100 * EPS):
        warn("At least one element of `rtol` is too small. "
             f"Setting `rtol = np.maximum(rtol, {100 * EPS})`.",
             stacklevel=3)  # 如果 rtol 中至少有一个元素太小，则发出警告并调整至最小误差限的倍数

        rtol = np.maximum(rtol, 100 * EPS)  # 将 rtol 调整为最小误差限的倍数

    atol = np.asarray(atol)
    if atol.ndim > 0 and atol.shape != (n,):
        raise ValueError("`atol` has wrong shape.")  # 如果 atol 的形状不正确，则抛出值错误

    if np.any(atol < 0):
        raise ValueError("`atol` must be positive.")  # 如果 atol 中有负数，则抛出值错误

    return rtol, atol  # 返回经验证后的 rtol 和 atol

def norm(x):
    """Compute RMS norm."""
    return np.linalg.norm(x) / x.size ** 0.5  # 计算向量 x 的均方根范数

def select_initial_step(fun, t0, y0, t_bound,
                        max_step, f0, direction, order, rtol, atol):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray - shape (n,)
        Initial value of the dependent variable.
    t_bound : float
        End-point of integration interval; used to ensure that t0+step<=tbound
        and that fun is only evaluated in the interval [t0,tbound]
    max_step : float
        Maximum allowable step size.
    f0 : ndarray - shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float or ndarray - shape (n,)
        Desired absolute tolerance.

    Returns
    -------
    # 定义变量 h_abs，表示建议的初始步长的绝对值
    h_abs : float
        Absolute value of the suggested initial step.

    # 参考文献
    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    # 如果初始值 y0 的大小为 0，则返回正无穷
    if y0.size == 0:
        return np.inf

    # 计算时间间隔的长度
    interval_length = abs(t_bound - t0)
    # 如果时间间隔长度为 0，则返回 0.0
    if interval_length == 0.0:
        return 0.0
    
    # 计算缩放因子
    scale = atol + np.abs(y0) * rtol
    # 计算初始值 y0 的范数
    d0 = norm(y0 / scale)
    # 计算函数值 f0 的范数
    d1 = norm(f0 / scale)
    # 如果 d0 或 d1 小于 1e-5，则将 h0 设为 1e-6，否则计算 h0
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    # 检查 t0+h0*direction 是否超出 t_bound
    h0 = min(h0, interval_length)
    # 计算下一个时间步的值 y1
    y1 = y0 + h0 * direction * f0
    # 计算下一个时间步的函数值 f1
    f1 = fun(t0 + h0 * direction, y1)
    # 计算斜率的变化率
    d2 = norm((f1 - f0) / scale) / h0

    # 如果 d1 和 d2 都小于等于 1e-15，则将 h1 设为 max(1e-6, h0 * 1e-3)，否则计算 h1
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    # 返回 h0、h1、时间间隔长度和最大步长的最小值
    return min(100 * h0, h1, interval_length, max_step)
    """Continuous ODE solution.

    It is organized as a collection of `DenseOutput` objects which represent
    local interpolants. It provides an algorithm to select a right interpolant
    for each given point.

    The interpolants cover the range between `t_min` and `t_max` (see
    Attributes below). Evaluation outside this interval is not forbidden, but
    the accuracy is not guaranteed.

    When evaluating at a breakpoint (one of the values in `ts`) a segment with
    the lower index is selected.

    Parameters
    ----------
    ts : array_like, shape (n_segments + 1,)
        Time instants between which local interpolants are defined. Must
        be strictly increasing or decreasing (zero segment with two points is
        also allowed).
    interpolants : list of DenseOutput with n_segments elements
        Local interpolants. An i-th interpolant is assumed to be defined
        between ``ts[i]`` and ``ts[i + 1]``.
    alt_segment : boolean
        Requests the alternative interpolant segment selection scheme. At each
        solver integration point, two interpolant segments are available. The
        default (False) and alternative (True) behaviours select the segment
        for which the requested time corresponded to ``t`` and ``t_old``,
        respectively. This functionality is only relevant for testing the
        interpolants' accuracy: different integrators use different
        construction strategies.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """
    def __init__(self, ts, interpolants, alt_segment=False):
        # Convert `ts` into a numpy array
        ts = np.asarray(ts)
        # Compute differences between consecutive elements in `ts`
        d = np.diff(ts)
        # Check if `ts` is strictly increasing or decreasing or a single segment with two points
        if not ((ts.size == 2 and ts[0] == ts[-1])
                or np.all(d > 0) or np.all(d < 0)):
            raise ValueError("`ts` must be strictly increasing or decreasing.")

        # Number of segments is determined by the number of interpolants
        self.n_segments = len(interpolants)
        # Validate that the number of timestamps matches the number of interpolants
        if ts.shape != (self.n_segments + 1,):
            raise ValueError("Numbers of time stamps and interpolants "
                             "don't match.")

        # Store the original `ts` and the list of interpolants
        self.ts = ts
        self.interpolants = interpolants

        # Determine `t_min` and `t_max` based on the direction of `ts`
        if ts[-1] >= ts[0]:
            self.t_min = ts[0]
            self.t_max = ts[-1]
            self.ascending = True
            # Set the side of interpolation selection based on `alt_segment`
            self.side = "right" if alt_segment else "left"
            # Store `ts` sorted in ascending order
            self.ts_sorted = ts
        else:
            self.t_min = ts[-1]
            self.t_max = ts[0]
            self.ascending = False
            # Set the side of interpolation selection based on `alt_segment`
            self.side = "left" if alt_segment else "right"
            # Store `ts` sorted in descending order
            self.ts_sorted = ts[::-1]
    def _call_single(self, t):
        # 在这里保持一定的对称性，当 t 在 self.ts 中时，
        # 如果 alt_segment=False，则优先选择索引较低的段。
        ind = np.searchsorted(self.ts_sorted, t, side=self.side)

        # 确定 t 所在的段索引，确保不超出段的范围
        segment = min(max(ind - 1, 0), self.n_segments - 1)
        # 如果不是升序排列，则调整段索引
        if not self.ascending:
            segment = self.n_segments - 1 - segment

        # 调用对应段的插值函数来计算 t 的值
        return self.interpolants[segment](t)

    def __call__(self, t):
        """Evaluate the solution.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate at.

        Returns
        -------
        y : ndarray, shape (n_states,) or (n_states, n_points)
            Computed values. Shape depends on whether `t` is a scalar or a
            1-D array.
        """
        # 将 t 转换为 numpy 数组，确保能够处理标量和一维数组
        t = np.asarray(t)

        if t.ndim == 0:
            # 如果 t 是标量，直接调用 _call_single 方法计算结果
            return self._call_single(t)

        # 对 t 进行排序并记录排序后的逆序索引
        order = np.argsort(t)
        reverse = np.empty_like(order)
        reverse[order] = np.arange(order.shape[0])
        t_sorted = t[order]

        # 查找每个 t_sorted 在 self.ts_sorted 中的位置
        segments = np.searchsorted(self.ts_sorted, t_sorted, side=self.side)
        segments -= 1
        segments[segments < 0] = 0
        segments[segments > self.n_segments - 1] = self.n_segments - 1
        # 如果不是升序排列，则调整段索引
        if not self.ascending:
            segments = self.n_segments - 1 - segments

        ys = []
        group_start = 0
        # 根据 segments 对 t_sorted 进行分组，对每个分组调用对应的插值函数
        for segment, group in groupby(segments):
            group_end = group_start + len(list(group))
            y = self.interpolants[segment](t_sorted[group_start:group_end])
            ys.append(y)
            group_start = group_end

        # 合并结果，并根据 reverse 将结果恢复到原始顺序
        ys = np.hstack(ys)
        ys = ys[:, reverse]

        return ys
# 计算用于拒绝数值雅可比矩阵近似的阈值
NUM_JAC_DIFF_REJECT = EPS ** 0.875
# 计算较小的数值雅可比矩阵近似的阈值
NUM_JAC_DIFF_SMALL = EPS ** 0.75
# 计算较大的数值雅可比矩阵近似的阈值
NUM_JAC_DIFF_BIG = EPS ** 0.25
# 最小因子用于计算数值雅可比矩阵近似
NUM_JAC_MIN_FACTOR = 1e3 * EPS
# 增加因子的倍数
NUM_JAC_FACTOR_INCREASE = 10
# 减少因子的倍数
NUM_JAC_FACTOR_DECREASE = 0.1

# 数值雅可比矩阵近似函数，专为求解ODE的数值求导器设计

def num_jac(fun, t, y, f, threshold, factor, sparsity=None):
    """Finite differences Jacobian approximation tailored for ODE solvers.

    This function computes finite difference approximation to the Jacobian
    matrix of `fun` with respect to `y` using forward differences.
    The Jacobian matrix has shape (n, n) and its element (i, j) is equal to
    ``d f_i / d y_j``.

    A special feature of this function is the ability to correct the step
    size from iteration to iteration. The main idea is to keep the finite
    difference significantly separated from its round-off error which
    approximately equals ``EPS * np.abs(f)``. It reduces a possibility of a
    huge error and assures that the estimated derivative are reasonably close
    to the true values (i.e., the finite difference approximation is at least
    qualitatively reflects the structure of the true Jacobian).

    Parameters
    ----------
    fun : callable
        Right-hand side of the system implemented in a vectorized fashion.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Value of the right hand side at (t, y).
    threshold : float
        Threshold for `y` value used for computing the step size as
        ``factor * np.maximum(np.abs(y), threshold)``. Typically, the value of
        absolute tolerance (atol) for a solver should be passed as `threshold`.
    factor : ndarray with shape (n,) or None
        Factor to use for computing the step size. Pass None for the very
        evaluation, then use the value returned from this function.
    sparsity : tuple (structure, groups) or None
        Sparsity structure of the Jacobian, `structure` must be csc_matrix.

    Returns
    -------
    J : ndarray or csc_matrix, shape (n, n)
        Jacobian matrix.
    factor : ndarray, shape (n,)
        Suggested `factor` for the next evaluation.
    """
    y = np.asarray(y)  # 将输入状态y转换为NumPy数组
    n = y.shape[0]  # 确定状态y的长度n
    if n == 0:
        return np.empty((0, 0)), factor  # 如果n为0，返回空矩阵和factor

    if factor is None:
        factor = np.full(n, EPS ** 0.5)  # 如果factor为None，则初始化为默认值
    else:
        factor = factor.copy()  # 复制传入的factor，以防止在函数内部修改原始值

    # 根据ODE的指导方向，使用f的实部作为步长方向的指示
    f_sign = 2 * (np.real(f) >= 0).astype(float) - 1
    # 计算步长的估计值，确保它远离舍入误差，大致等于EPS * np.abs(f)
    y_scale = f_sign * np.maximum(threshold, np.abs(y))
    h = (y + factor * y_scale) - y

    # 确保步长不为0
    for i in np.nonzero(h == 0)[0]:
        while h[i] == 0:
            factor[i] *= 10
            h[i] = (y[i] + factor[i] * y_scale[i]) - y[i]
    # 如果稀疏性参数 sparsity 为 None，则调用 _dense_num_jac 函数，并返回其结果
    if sparsity is None:
        return _dense_num_jac(fun, t, y, f, h, factor, y_scale)
    else:
        # 否则，解构 sparsity 参数为 structure 和 groups
        structure, groups = sparsity
        # 调用 _sparse_num_jac 函数，并返回其结果，传递了额外的稀疏性结构参数和分组参数
        return _sparse_num_jac(fun, t, y, f, h, factor, y_scale,
                               structure, groups)
def _dense_num_jac(fun, t, y, f, h, factor, y_scale):
    # 获取状态向量维度
    n = y.shape[0]
    # 创建对角矩阵，每个对角元素为步长向量 h
    h_vecs = np.diag(h)
    # 计算使用扰动后的函数值 f_new
    f_new = fun(t, y[:, None] + h_vecs)
    # 计算函数值的差分 diff
    diff = f_new - f[:, None]
    # 找出每列中绝对值最大的索引
    max_ind = np.argmax(np.abs(diff), axis=0)
    # 创建一个范围数组
    r = np.arange(n)
    # 计算每列的最大差分值
    max_diff = np.abs(diff[max_ind, r])
    # 计算比例因子 scale
    scale = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))

    # 判断是否有差分值过小的情况
    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if np.any(diff_too_small):
        # 找出过小的索引
        ind, = np.nonzero(diff_too_small)
        # 计算新的扩大因子
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        # 计算新的步长 h_new
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        # 更新对角矩阵中的对应列
        h_vecs[ind, ind] = h_new
        # 重新计算 f_new
        f_new = fun(t, y[:, None] + h_vecs[:, ind])
        # 计算更新后的函数值差分
        diff_new = f_new - f[:, None]
        # 找出更新后每列中绝对值最大的索引
        max_ind = np.argmax(np.abs(diff_new), axis=0)
        # 更新范围数组
        r = np.arange(ind.shape[0])
        # 计算更新后的最大差分值
        max_diff_new = np.abs(diff_new[max_ind, r])
        # 计算更新后的比例因子
        scale_new = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))

        # 判断是否需要更新
        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if np.any(update):
            # 找出需要更新的索引
            update, = np.nonzero(update)
            # 更新对应索引处的因子和步长
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    # 将差分除以步长得到数值雅可比矩阵 diff
    diff /= h

    # 根据最大差分值的大小调整因子 factor
    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    # 将因子限制在最小因子 NUM_JAC_MIN_FACTOR 以上
    factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)

    # 返回数值雅可比矩阵 diff 和更新后的因子 factor
    return diff, factor
    # 如果任何一个元素在 diff_too_small 中为真，则进入条件语句
    if np.any(diff_too_small):
        # 找到 diff_too_small 中为真的元素的索引
        ind, = np.nonzero(diff_too_small)
        # 根据元素的索引计算新的因子值
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        # 计算新的步长 h_new
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        # 初始化一个全零数组 h_new_all，长度为 n
        h_new_all = np.zeros(n)
        # 将计算得到的 h_new 值放入 h_new_all 中对应的位置
        h_new_all[ind] = h_new

        # 找到 groups[ind] 中唯一的分组值
        groups_unique = np.unique(groups[ind])
        # 创建一个空的映射数组 groups_map，长度为 n_groups
        groups_map = np.empty(n_groups, dtype=int)
        # 创建一个空的二维数组 h_vecs，形状为 (groups_unique.shape[0], n)
        h_vecs = np.empty((groups_unique.shape[0], n))
        # 遍历 groups_unique 中的每个分组
        for k, group in enumerate(groups_unique):
            # 创建一个布尔数组 e，标记出当前分组 group 在 groups 中的位置
            e = np.equal(group, groups)
            # 将 h_new_all 乘以 e 后的结果存入 h_vecs 的第 k 行
            h_vecs[k] = h_new_all * e
            # 将当前分组 group 映射到索引 k
            groups_map[group] = k
        # 将 h_vecs 进行转置，使其形状变为 (n, groups_unique.shape[0])
        h_vecs = h_vecs.T

        # 计算新的函数值 f_new，传入参数 t 和 y 加上 h_vecs
        f_new = fun(t, y[:, None] + h_vecs)
        # 计算函数值的差 df
        df = f_new - f[:, None]
        # 找到结构矩阵 structure[:, ind] 中非零元素的行索引 i、列索引 j 和值 _
        i, j, _ = find(structure[:, ind])
        # 根据 df 和 groups_map[groups[ind]] 构建一个稀疏矩阵 diff_new
        diff_new = coo_matrix((df[i, groups_map[groups[ind[j]]]],
                               (i, j)), shape=(n, ind.shape[0])).tocsc()

        # 找到每列最大值的索引 max_ind_new
        max_ind_new = np.array(abs(diff_new).argmax(axis=0)).ravel()
        # 创建一个序列 r，长度为 ind.shape[0]
        r = np.arange(ind.shape[0])
        # 找到最大差值 max_diff_new，并转换为数组
        max_diff_new = np.asarray(np.abs(diff_new[max_ind_new, r])).ravel()
        # 计算新的比例尺度 scale_new
        scale_new = np.maximum(
            np.abs(f[max_ind_new]),
            np.abs(f_new[max_ind_new, groups_map[groups[ind]]]))

        # 检查是否需要更新 factor、h、diff、scale 和 max_diff
        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if np.any(update):
            # 找到需要更新的索引
            update, = np.nonzero(update)
            update_ind = ind[update]
            # 更新 factor、h、diff、scale 和 max_diff 的值
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    # 对 diff.data 进行归一化处理，除以每行对应的步长 h
    diff.data /= np.repeat(h, np.diff(diff.indptr))

    # 根据条件更新 factor 的值
    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    # 将 factor 的值与 NUM_JAC_MIN_FACTOR 取最大值
    factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)

    # 返回计算得到的 diff 和 factor
    return diff, factor
```