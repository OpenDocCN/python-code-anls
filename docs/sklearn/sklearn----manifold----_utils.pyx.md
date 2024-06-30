# `D:\src\scipysrc\scikit-learn\sklearn\manifold\_utils.pyx`

```
# 导入NumPy库，用于数组操作
import numpy as np

# 从libc中导入math模块和math模块中的INFINITY常量
from libc cimport math
from libc.math cimport INFINITY

# 从当前包的utils._typedefs模块中导入float32_t和float64_t类型
from ..utils._typedefs cimport float32_t, float64_t

# 定义双精度浮点型的常量EPSILON_DBL和PERPLEXITY_TOLERANCE
cdef float EPSILON_DBL = 1e-8
cdef float PERPLEXITY_TOLERANCE = 1e-5


# TODO: have this function support float32 and float64 and preserve inputs' dtypes.
# 二分搜索函数，用于计算条件高斯分布的sigma
def _binary_search_perplexity(
        const float32_t[:, :] sqdistances,
        float desired_perplexity,
        int verbose):
    """Binary search for sigmas of conditional Gaussians.

    This approximation reduces the computational complexity from O(N^2) to
    O(uN).

    Parameters
    ----------
    sqdistances : ndarray of shape (n_samples, n_neighbors), dtype=np.float32
        Distances between training samples and their k nearest neighbors.
        When using the exact method, this is a square (n_samples, n_samples)
        distance matrix. The TSNE default metric is "euclidean" which is
        interpreted as squared euclidean distance.

    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : ndarray of shape (n_samples, n_samples), dtype=np.float64
        Probabilities of conditional Gaussian distributions p_i|j.
    """
    # 最大二分搜索步数
    cdef long n_steps = 100

    # 样本数和最近邻数
    cdef long n_samples = sqdistances.shape[0]
    cdef long n_neighbors = sqdistances.shape[1]
    # 是否使用了最近邻信息
    cdef int using_neighbors = n_neighbors < n_samples

    # 条件高斯分布的精度参数
    cdef double beta
    cdef double beta_min
    cdef double beta_max
    cdef double beta_sum = 0.0

    # 使用对数尺度
    cdef double desired_entropy = math.log(desired_perplexity)
    cdef double entropy_diff

    # 计算中使用的变量
    cdef double entropy
    cdef double sum_Pi
    cdef double sum_disti_Pi
    cdef long i, j, l

    # 该数组后续作为32位数组使用，其中有多个中间浮点数加法，受益于额外的精度
    cdef float64_t[:, :] P = np.zeros(
        (n_samples, n_neighbors), dtype=np.float64)
    for i in range(n_samples):
        beta_min = -INFINITY
        beta_max = INFINITY
        beta = 1.0

        # 为第 i 个条件分布进行二分查找精度
        for l in range(n_steps):
            # 计算当前的熵和对应的概率
            # 只计算最近邻居或全部数据的情况下
            # 如果使用邻居的话
            sum_Pi = 0.0
            for j in range(n_neighbors):
                if j != i or using_neighbors:
                    P[i, j] = math.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            sum_disti_Pi = 0.0

            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]

            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            # 如果熵的变化小于等于指定的困惑度容差，结束循环
            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            # 根据熵的变化调整 beta 值的范围
            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == INFINITY:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -INFINITY:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        beta_sum += beta

        # 如果 verbose 为真且每 1000 次循环输出一次进度信息或者循环结束时输出信息
        if verbose and ((i + 1) % 1000 == 0 or i + 1 == n_samples):
            print("[t-SNE] Computed conditional probabilities for sample "
                  "%d / %d" % (i + 1, n_samples))

    # 如果 verbose 为真，输出平均 sigma 值的信息
    if verbose:
        print("[t-SNE] Mean sigma: %f"
              % np.mean(math.sqrt(n_samples / beta_sum)))
    # 返回 P 的 NumPy 数组表示
    return np.asarray(P)
```