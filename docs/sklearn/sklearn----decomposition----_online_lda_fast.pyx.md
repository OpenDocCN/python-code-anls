# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_online_lda_fast.pyx`

```
# 导入 NumPy 库，用于处理数组和数学运算
import numpy as np

# 从 Cython 库导入特定的浮点类型
from cython cimport floating

# 从 C 标准库中导入数学函数，包括 exp（指数函数）、fabs（绝对值函数）、log（自然对数函数）
from libc.math cimport exp, fabs, log

# 从 utils 模块的 _typedefs 中导入 float64_t 和 intp_t 类型定义
from ..utils._typedefs cimport float64_t, intp_t

# 计算两个数组的平均变化量
def mean_change(const floating[:] arr_1, const floating[:] arr_2):
    """Calculate the mean difference between two arrays.

    Equivalent to np.abs(arr_1 - arr2).mean().
    """

    # 声明变量
    cdef float64_t total, diff
    cdef intp_t i, size

    # 获取数组的长度
    size = arr_1.shape[0]

    # 初始化总和为 0
    total = 0.0

    # 遍历数组，计算每个元素的绝对差值并累加到总和中
    for i in range(size):
        diff = fabs(arr_1[i] - arr_2[i])
        total += diff

    # 返回平均变化量
    return total / size


# 计算单个样本的 Dirichlet 期望
def _dirichlet_expectation_1d(
    floating[:] doc_topic,
    floating doc_topic_prior,
    floating[:] out
):
    """Dirichlet expectation for a single sample:
        exp(E[log(theta)]) for theta ~ Dir(doc_topic)
    after adding doc_topic_prior to doc_topic, in-place.

    Equivalent to
        doc_topic += doc_topic_prior
        out[:] = np.exp(psi(doc_topic) - psi(np.sum(doc_topic)))
    """

    # 声明变量
    cdef floating dt, psi_total, total
    cdef intp_t i, size

    # 获取数组的长度
    size = doc_topic.shape[0]

    # 初始化总和为 0
    total = 0.0

    # 将 doc_topic_prior 添加到 doc_topic 中每个元素上，并计算总和
    for i in range(size):
        dt = doc_topic[i] + doc_topic_prior
        doc_topic[i] = dt
        total += dt

    # 计算总和的 Digamma 函数值
    psi_total = psi(total)

    # 计算每个元素的 Dirichlet 期望，并存储到 out 数组中
    for i in range(size):
        out[i] = exp(psi(doc_topic[i]) - psi_total)


# 计算多个样本的 Dirichlet 期望
def _dirichlet_expectation_2d(const floating[:, :] arr):
    """Dirichlet expectation for multiple samples:
    E[log(theta)] for theta ~ Dir(arr).

    Equivalent to psi(arr) - psi(np.sum(arr, axis=1))[:, np.newaxis].

    Note that unlike _dirichlet_expectation_1d, this function doesn't compute
    the exp and doesn't add in the prior.
    """

    # 声明变量
    cdef floating row_total, psi_row_total
    cdef floating[:, :] d_exp
    cdef intp_t i, j, n_rows, n_cols

    # 获取数组的行数和列数
    n_rows = arr.shape[0]
    n_cols = arr.shape[1]

    # 创建一个和 arr 形状相同的空数组，用于存储 Dirichlet 期望
    d_exp = np.empty_like(arr)

    # 计算每行的总和和对应的 Digamma 函数值
    for i in range(n_rows):
        row_total = 0

        # 计算当前行的总和
        for j in range(n_cols):
            row_total += arr[i, j]

        # 计算当前行总和的 Digamma 函数值
        psi_row_total = psi(row_total)

        # 计算当前行每个元素的 Dirichlet 期望，并存储到 d_exp 数组中
        for j in range(n_cols):
            d_exp[i, j] = psi(arr[i, j]) - psi_row_total

    # 返回存储 Dirichlet 期望的数组的基础对象（base）
    return d_exp.base


# Digamma 函数的实现，针对正参数进行优化以提高速度，不保证精度
#
# 参考文献：J. Bernardo (1976). Algorithm AS 103: Psi (Digamma) Function.
# https://www.uv.es/~bernardo/1976AppStatist.pdf
cdef floating psi(floating x) noexcept nogil:
    cdef double EULER = 0.577215664901532860606512090082402431

    # 当 x 很小时，使用近似公式计算 Digamma 函数的值
    if x <= 1e-6:
        # psi(x) = -EULER - 1/x + O(x)
        return -EULER - 1. / x

    # 声明变量
    cdef floating r, result = 0

    # 使用近似公式计算 Digamma 函数的值
    while x < 6:
        result -= 1. / x
        x += 1

    # 使用级数展开计算 Digamma 函数的值
    r = 1. / x
    result += log(x) - .5 * r
    r = r * r
    result -= r * ((1./12.) - r * ((1./120.) - r * (1./252.)))

    # 返回 Digamma 函数的值
    return result
```