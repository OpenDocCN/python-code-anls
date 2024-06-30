# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_onenormest.py`

```
# 稀疏块1-范数估计器。
"""

import numpy as np  # 导入 NumPy 库，用于数值计算
from scipy.sparse.linalg import aslinearoperator  # 导入 aslinearoperator 函数，用于将对象转换为线性操作符


__all__ = ['onenormest']  # 定义模块的公开接口，只包含 onenormest 函数


def onenormest(A, t=2, itmax=5, compute_v=False, compute_w=False):
    """
    计算稀疏矩阵1-范数的下界估计值。

    Parameters
    ----------
    A : ndarray 或其他线性操作符
        可以进行转置和矩阵乘积的线性操作符。
    t : int, optional
        控制精度与时间及内存使用之间的权衡参数。
        较大的值会花费更多时间和内存，但会提供更精确的输出。
    itmax : int, optional
        最多使用这么多次迭代。
    compute_v : bool, optional
        如果为 True，则请求一个用于范数最大化的线性操作符输入向量。
    compute_w : bool, optional
        如果为 True，则请求一个具有相对较大1-范数的线性操作符输出向量。

    Returns
    -------
    est : float
        稀疏矩阵1-范数的一个低估计值。
    v : ndarray, optional
        向量 v 满足 ||Av||_1 == est*||v||_1。
        可以将其视为线性操作符的输入，产生具有特别大范数的输出。
    w : ndarray, optional
        向量 Av 具有相对较大的1-范数。
        可以将其视为线性操作符的输出，与输入相比具有较大的范数。

    Notes
    -----
    这是参考文献 [1] 中的算法 2.4。

    在 [2] 中描述如下。
    "这个算法通常需要评估大约 4t 次矩阵-向量乘积，几乎总是产生一个（实际上是范数的下界）估计值，精确到一个因子 3。"

    .. versionadded:: 0.13.0

    References
    ----------
    .. [1] Nicholas J. Higham and Francoise Tisseur (2000),
           "A Block Algorithm for Matrix 1-Norm Estimation,
           with an Application to 1-Norm Pseudospectra."
           SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.

    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2009),
           "A new scaling and squaring algorithm for the matrix exponential."
           SIAM J. Matrix Anal. Appl. Vol. 31, No. 3, pp. 970-989.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import onenormest
    >>> A = csc_matrix([[1., 0., 0.], [5., 8., 2.], [0., -1., 0.]], dtype=float)
    >>> A.toarray()
    array([[ 1.,  0.,  0.],
           [ 5.,  8.,  2.],
           [ 0., -1.,  0.]])
    >>> onenormest(A)
    9.0
    >>> np.linalg.norm(A.toarray(), ord=1)
    9.0
    """

    # 检查输入。
    A = aslinearoperator(A)  # 将输入转换为线性操作符
    if A.shape[0] != A.shape[1]:  # 如果操作符不像一个方阵那样作用
        raise ValueError('expected the operator to act like a square matrix')

    # 如果操作符的大小相对于 t 来说很小，
    # 那么计算精确范数会更容易。
    # 否则，估计范数。

    # 获取矩阵 A 的列数
    n = A.shape[1]

    # 如果 t 大于等于 n
    if t >= n:
        # 将 A 转换为线性操作符，然后计算其乘以单位矩阵后的显式表示
        A_explicit = np.asarray(aslinearoperator(A).matmat(np.identity(n)))

        # 检查 A_explicit 的形状是否为 (n, n)，如果不是则抛出异常
        if A_explicit.shape != (n, n):
            raise Exception('internal error: ',
                    'unexpected shape ' + str(A_explicit.shape))

        # 计算 A_explicit 的每列绝对值之和
        col_abs_sums = abs(A_explicit).sum(axis=0)

        # 检查 col_abs_sums 的形状是否为 (n, )，如果不是则抛出异常
        if col_abs_sums.shape != (n, ):
            raise Exception('internal error: ',
                    'unexpected shape ' + str(col_abs_sums.shape))

        # 找到 col_abs_sums 中绝对值之和最大的列的索引
        argmax_j = np.argmax(col_abs_sums)

        # 构造单位向量 v，使得 v[argmax_j] = 1
        v = elementary_vector(n, argmax_j)

        # 取出 A_explicit 的第 argmax_j 列作为向量 w
        w = A_explicit[:, argmax_j]

        # 估计的范数为 col_abs_sums[argmax_j]
        est = col_abs_sums[argmax_j]

    else:
        # 使用 _onenormest_core 函数估计 A 和其共轭转置 A.H 的范数
        est, v, w, nmults, nresamples = _onenormest_core(A, A.H, t, itmax)

    # 返回范数的估计值以及相关的证明
    # 如果需要计算 v 或 w，则返回结果中包括 v 或 w
    if compute_v or compute_w:
        result = (est,)
        if compute_v:
            result += (v,)
        if compute_w:
            result += (w,)
        return result
    else:
        # 否则只返回范数的估计值
        return est
# 创建一个装饰器，用于对元素级函数进行装饰，使其沿着第一个维度以块的方式应用，以避免临时内存使用过多。
def _blocked_elementwise(func):
    # 定义块大小为 2 的 20 次方
    block_size = 2**20

    # 定义包装函数 wrapper，接收输入参数 x
    def wrapper(x):
        # 如果输入 x 的第一个维度小于块大小，则直接应用函数 func
        if x.shape[0] < block_size:
            return func(x)
        else:
            # 否则，按块大小分块处理
            # 计算 func 在前一个块上的结果 y0
            y0 = func(x[:block_size])
            # 创建一个与输入 x 形状相同的全零数组 y，类型与 y0 相同
            y = np.zeros((x.shape[0],) + y0.shape[1:], dtype=y0.dtype)
            # 将 y0 的结果放入 y 的相应位置
            y[:block_size] = y0
            # 释放 y0 的内存
            del y0
            # 对剩余的数据块依次计算 func 并放入 y 中
            for j in range(block_size, x.shape[0], block_size):
                y[j:j+block_size] = func(x[j:j+block_size])
            # 返回最终结果 y
            return y
    # 返回包装函数 wrapper
    return wrapper


@_blocked_elementwise
# 对矩阵 X 进行符号函数和舍入操作，适用于实数和复数矩阵
def sign_round_up(X):
    """
    This should do the right thing for both real and complex matrices.

    From Higham and Tisseur:
    "Everything in this section remains valid for complex matrices
    provided that sign(A) is redefined as the matrix (aij / |aij|)
    (and sign(0) = 1) transposes are replaced by conjugate transposes."
    """
    # 复制输入矩阵 X 到 Y
    Y = X.copy()
    # 将 Y 中的零元素替换为 1，避免除零错误，并将 Y 归一化
    Y[Y == 0] = 1
    Y /= np.abs(Y)
    # 返回处理后的矩阵 Y
    return Y


@_blocked_elementwise
# 计算矩阵 X 每行绝对值的最大值
def _max_abs_axis1(X):
    return np.max(np.abs(X), axis=1)


# 对矩阵 X 按第 0 轴进行绝对值求和，使用块大小为 2 的 20 次方
def _sum_abs_axis0(X):
    block_size = 2**20
    r = None
    for j in range(0, X.shape[0], block_size):
        # 对 X 中每个块的绝对值进行求和，沿第 0 轴
        y = np.sum(np.abs(X[j:j+block_size]), axis=0)
        if r is None:
            r = y
        else:
            r += y
    return r


# 创建一个 n 维零向量，其中第 i 个元素为 1
def elementary_vector(n, i):
    v = np.zeros(n, dtype=float)
    v[i] = 1
    return v


# 判断两个向量 v 和 w 是否平行，平行条件是它们相等或者有一个是另一个的相反数
def vectors_are_parallel(v, w):
    # 如果 v 或 w 不是 1 维向量，或者它们的形状不匹配，则引发 ValueError
    if v.ndim != 1 or v.shape != w.shape:
        raise ValueError('expected conformant vectors with entries in {-1,1}')
    n = v.shape[0]
    # 返回向量 v 和 w 的点积是否等于向量长度 n
    return np.dot(v, w) == n


# 判断矩阵 X 的每一列是否与矩阵 Y 的某一列平行
def every_col_of_X_is_parallel_to_a_col_of_Y(X, Y):
    # 遍历 X 的每一列 v
    for v in X.T:
        # 如果存在某一列 w 在 Y 中与 v 平行，则返回 True
        if not any(vectors_are_parallel(v, w) for w in Y.T):
            return False
    # 如果所有列都找到了匹配，则返回 True
    return True


# 判断矩阵 X 的第 i 列是否需要重新采样，需要重新采样的条件是它与前面的列或者 Y 中的某一列平行
def column_needs_resampling(i, X, Y=None):
    n, t = X.shape
    v = X[:, i]
    # 如果存在前面的列与 v 平行，则需要重新采样
    if any(vectors_are_parallel(v, X[:, j]) for j in range(i)):
        return True
    # 如果 Y 存在且存在某一列与 v 平行，则需要重新采样
    if Y is not None:
        if any(vectors_are_parallel(v, w) for w in Y.T):
            return True
    # 否则不需要重新采样
    return False


# 对矩阵 X 的第 i 列进行重新采样，使用随机整数填充元素值
def resample_column(i, X):
    X[:, i] = np.random.randint(0, 2, size=X.shape[0])*2 - 1


# 判断 a 是否小于或接近于 b，使用 np.allclose 判断是否接近
def less_than_or_close(a, b):
    return np.allclose(a, b) or (a < b)


def _algorithm_2_2(A, AT, t):
    """
    This is Algorithm 2.2.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can produce matrix products.
    AT : ndarray or other linear operator
        The transpose of A.
    t : int, optional
        控制精度与时间、内存使用之间的权衡的正整数参数。

    Returns
    -------
    g : sequence
        一个非负递减向量，其中 g[j] 是矩阵 A 的第 j 大的 1-范数的下界。
        因此，这个向量的第一个条目是线性操作 A 的 1-范数的下界。
        这个序列的长度为 t。
    ind : sequence
        ind 的第 i 个条目是矩阵 A 中 1-范数由 g[i] 给出的列的索引。
        这个索引序列的长度为 t，其条目从范围(n)中选择，可能会有重复，
        其中 n 是操作 A 的阶数。

    Notes
    -----
    这个算法主要用于测试。
    它使用 'ind' 数组的方式类似于算法 2.4 中的使用。这个算法 2.2 可能更容易测试，
    因此它有机会发现与索引相关的错误，这些错误可能以不太明显的方式传播到算法 2.4。

    """
    A_linear_operator = aslinearoperator(A)
    AT_linear_operator = aslinearoperator(AT)
    n = A_linear_operator.shape[0]

    # Initialize the X block with columns of unit 1-norm.
    # 使用单位 1-范数的列初始化 X 块。
    X = np.ones((n, t))
    if t > 1:
        X[:, 1:] = np.random.randint(0, 2, size=(n, t-1))*2 - 1
    X /= float(n)

    # Iteratively improve the lower bounds.
    # 迭代改进下界。
    # 跟踪额外的内容，用于调试时的不变性断言。
    g_prev = None
    h_prev = None
    k = 1
    ind = range(t)
    # 进入主循环，直到满足退出条件
    while True:
        # 计算 Y = A_linear_operator 对 X 进行矩阵乘法后的结果，并转换为 NumPy 数组
        Y = np.asarray(A_linear_operator.matmat(X))
        # 计算 g = Y 按轴 0 求绝对值后的和
        g = _sum_abs_axis0(Y)
        # 找到 g 中最大值的索引
        best_j = np.argmax(g)
        # 对 g 进行排序
        g.sort()
        # 对 g 进行逆序排列
        g = g[::-1]
        # 对 Y 进行符号舍入处理，得到 S
        S = sign_round_up(Y)
        # 计算 Z = AT_linear_operator 对 S 进行矩阵乘法后的结果，并转换为 NumPy 数组
        Z = np.asarray(AT_linear_operator.matmat(S))
        # 计算 h = Z 按轴 1 求绝对值后的最大值
        h = _max_abs_axis1(Z)

        # 如果迭代次数 k 大于等于 2，则检查退出条件
        if k >= 2:
            # 检查是否满足退出条件
            if less_than_or_close(max(h), np.dot(Z[:, best_j], X[:, best_j])):
                break
        
        # 根据 h 的排序结果，选取前 t 个索引
        ind = np.argsort(h)[::-1][:t]
        h = h[ind]
        # 更新 X 的前 t 列为单位向量
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])

        # 检查不变量 (2.2)
        if k >= 2:
            if not less_than_or_close(g_prev[0], h_prev[0]):
                raise Exception('invariant (2.2) is violated')
            if not less_than_or_close(h_prev[0], g[0]):
                raise Exception('invariant (2.2) is violated')

        # 检查不变量 (2.3)
        if k >= 3:
            for j in range(t):
                if not less_than_or_close(g[j], g_prev[j]):
                    raise Exception('invariant (2.3) is violated')

        # 更新 g_prev 和 h_prev 为当前 g 和 h 的值
        g_prev = g
        h_prev = h
        # 增加迭代次数 k
        k += 1

    # 返回 g 和 ind，即下界和对应的列索引
    return g, ind
# 定义一个函数来计算稀疏矩阵的1-范数的下界估计。

def _onenormest_core(A, AT, t, itmax):
    """
    Compute a lower bound of the 1-norm of a sparse matrix.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can produce matrix products.
    AT : ndarray or other linear operator
        The transpose of A.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
    itmax : int, optional
        Use at most this many iterations.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.
    nmults : int, optional
        The number of matrix products that were computed.
    nresamples : int, optional
        The number of times a parallel column was observed,
        necessitating a re-randomization of the column.

    Notes
    -----
    This is algorithm 2.4.

    """

    # 这个函数基本上是从Higham和Tisseur（2000）的论文中的算法2.4直接翻译而来。

    # 将输入的A转换为线性操作对象
    A_linear_operator = aslinearoperator(A)
    # 将输入的AT（A的转置）转换为线性操作对象
    AT_linear_operator = aslinearoperator(AT)

    # 检查迭代次数itmax是否小于2，如果是则抛出异常
    if itmax < 2:
        raise ValueError('at least two iterations are required')
    
    # 检查参数t是否小于1，如果是则抛出异常
    if t < 1:
        raise ValueError('at least one column is required')
    
    # 获取矩阵A的行数
    n = A.shape[0]
    
    # 检查参数t是否大于等于矩阵A的阶数n，如果是则抛出异常
    if t >= n:
        raise ValueError('t should be smaller than the order of A')

    # 跟踪大*小矩阵乘积的数量以及重新采样的次数
    nmults = 0  # 记录矩阵乘积的计数
    nresamples = 0  # 记录重新采样的计数

    # "我们现在解释我们选择起始矩阵的原因。我们取X的第一列为全1向量。这样做的好处是对于具有非负元素的矩阵，算法在第二次迭代中收敛，并且这样的矩阵在应用中出现。"
    X = np.ones((n, t), dtype=float)  # 创建一个n行t列的全1矩阵作为起始矩阵X

    # "其余的列被选择为rand{-1,1}，并检查并纠正平行列，就像算法主体中的S一样。"
    if t > 1:
        for i in range(1, t):
            # 这些技术上是初始样本，而不是重新采样，所以重新采样计数不增加。
            resample_column(i, X)  # 对第i列进行重新采样
        for i in range(t):
            while column_needs_resampling(i, X):
                resample_column(i, X)  # 如果第i列需要重新采样，则重新采样
                nresamples += 1  # 增加重新采样计数

    # "选择起始矩阵X，其列具有单位1-范数。"
    X /= float(n)  # 将矩阵X的所有列单位化为1-范数

    # "已使用的单位向量e_j的索引"
    ind_hist = np.zeros(0, dtype=np.intp)  # 创建一个空数组用于存储使用过的单位向量的索引
    est_old = 0  # 初始化旧的1-范数估计值为0
    # 初始化大小为 (n, t) 的全零矩阵 S，数据类型为浮点数
    S = np.zeros((n, t), dtype=float)
    # 设置迭代计数器 k 初始值为 1
    k = 1
    # 初始化指示器变量 ind 为 None
    ind = None
    
    # 进入主循环，直到满足退出条件才结束
    while True:
        # 将线性操作 A_linear_operator 对 X 执行矩阵乘法，并转换为 NumPy 数组 Y
        Y = np.asarray(A_linear_operator.matmat(X))
        # 更新乘法操作计数器
        nmults += 1
        # 计算 Y 沿第一轴绝对值之和，得到向量 mags
        mags = _sum_abs_axis0(Y)
        # 计算 mags 中的最大值作为估计值 est
        est = np.max(mags)
        # 找到 mags 中最大值对应的索引 best_j
        best_j = np.argmax(mags)
        
        # 如果当前估计值大于上一次的估计值或者 k 等于 2，则执行以下操作
        if est > est_old or k == 2:
            if k >= 2:
                # 如果 k 大于等于 2，则记录当前 ind 的 best_j 索引
                ind_best = ind[best_j]
            # 取出 Y 的第 best_j 列作为向量 w
            w = Y[:, best_j]
        
        # (1) 如果 k 大于等于 2 并且当前估计值 est 小于等于上一次的估计值 est_old，则更新 est 为 est_old，并跳出循环
        if k >= 2 and est <= est_old:
            est = est_old
            break
        
        # 更新上一次的估计值为当前 est
        est_old = est
        # 将上一次的 S 赋值给 S_old
        S_old = S
        
        # 如果 k 超过了最大迭代次数 itmax，则跳出循环
        if k > itmax:
            break
        
        # 对 Y 执行符号取整操作，并将结果赋值给 S
        S = sign_round_up(Y)
        # 删除 Y 变量释放内存
        del Y
        
        # (2) 如果每列 X 都与 S 或者 S_old 中的某列平行，则跳出循环
        if every_col_of_X_is_parallel_to_a_col_of_Y(S, S_old):
            break
        
        # 如果 t 大于 1，则对每列进行检查和重采样操作
        if t > 1:
            for i in range(t):
                # 当列 i 需要重新采样时，调用 resample_column 函数
                while column_needs_resampling(i, S, S_old):
                    resample_column(i, S)
                    # 增加重采样计数器
                    nresamples += 1
        # 删除 S_old 变量释放内存
        del S_old
        
        # (3) 将 S 传递给 AT_linear_operator 执行转置线性操作，并转换为 NumPy 数组 Z
        Z = np.asarray(AT_linear_operator.matmat(S))
        # 更新乘法操作计数器
        nmults += 1
        # 计算 Z 沿第二轴绝对值的最大值，得到向量 h
        h = _max_abs_axis1(Z)
        # 删除 Z 变量释放内存
        del Z
        
        # (4) 如果 k 大于等于 2 并且 h 中的最大值等于 h[ind_best]，则跳出循环
        if k >= 2 and max(h) == h[ind_best]:
            break
        
        # 对 h 进行排序，使得 h_first >= ... >= h_last，并相应地重新排序 ind
        ind = np.argsort(h)[::-1][:t+len(ind_hist)].copy()
        # 删除 h 变量释放内存
        del h
        
        # 如果 t 大于 1，则执行以下操作
        if t > 1:
            # (5) 如果已经访问过最有前景的 t 个向量，则跳出循环
            if np.isin(ind[:t], ind_hist).all():
                break
            # 将未访问过的最有前景的向量放在列表的前面，已访问过的放在后面，并保持与 h 排序相关的索引顺序
            seen = np.isin(ind, ind_hist)
            ind = np.concatenate((ind[~seen], ind[seen]))
        
        # 对于每个 j，更新 X 的第 j 列为 n 维单位向量的 ind[j] 列
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])
        
        # 更新 ind_hist 记录，将新的 ind[:t] 中未在 ind_hist 中出现过的索引添加到 ind_hist 中
        new_ind = ind[:t][~np.isin(ind[:t], ind_hist)]
        ind_hist = np.concatenate((ind_hist, new_ind))
        
        # 增加迭代计数器 k
        k += 1
    
    # 将最佳向量 v 设置为 n 维单位向量的 ind_best 列
    v = elementary_vector(n, ind_best)
    # 返回结果 est、最佳向量 v、向量 w、乘法操作计数器 nmults、重采样计数器 nresamples
    return est, v, w, nmults, nresamples
```