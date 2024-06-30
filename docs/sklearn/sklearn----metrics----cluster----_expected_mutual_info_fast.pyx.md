# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\_expected_mutual_info_fast.pyx`

```
# 从libc.math导入exp和lgamma函数
from libc.math cimport exp, lgamma

# 从...utils._typedefs导入float64_t和int64_t
from ...utils._typedefs cimport float64_t, int64_t

# 导入numpy库并简写为np
import numpy as np
# 从scipy.special模块中导入gammaln函数
from scipy.special import gammaln

# 定义一个函数，计算两种标签分配的期望互信息
def expected_mutual_information(contingency, int64_t n_samples):
    """Calculate the expected mutual information for two labelings."""
    # 声明Cython变量和类型
    cdef:
        float64_t emi = 0  # 初始化期望互信息为0
        int64_t n_rows, n_cols  # 声明行数和列数变量
        float64_t term2, term3, gln  # 声明浮点数变量
        int64_t[::1] a_view, b_view  # 声明一维整型数组变量
        float64_t[::1] term1  # 声明一维浮点数数组变量
        float64_t[::1] gln_a, gln_b, gln_Na, gln_Nb, gln_Nnij, log_Nnij  # 声明一维浮点数数组变量
        float64_t[::1] log_a, log_b  # 声明一维浮点数数组变量
        Py_ssize_t i, j, nij  # 声明Python对象的大小类型变量
        int64_t start, end  # 声明整型变量

    # 获取列联表的行数和列数
    n_rows, n_cols = contingency.shape
    # 计算行求和和列求和，并展平为一维数组
    a = np.ravel(contingency.sum(axis=1).astype(np.int64, copy=False))
    b = np.ravel(contingency.sum(axis=0).astype(np.int64, copy=False))
    a_view = a  # 将a数组赋给Cython的数组视图变量a_view
    b_view = b  # 将b数组赋给Cython的数组视图变量b_view

    # 如果任何一个标签分配的熵为零，则期望互信息为0
    if a.size == 1 or b.size == 1:
        return 0.0

    # 期望互信息方程有三个主要项，这些项在不同的nij值上相乘并求和。
    # nijs[0]永远不会被使用，但是将其设置为1可以简化索引。
    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype='float')
    nijs[0] = 1  # 防止除零警告，由于它不会被使用，没有问题。
    
    # term1为nij / N
    term1 = nijs / n_samples
    
    # term2是log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
    log_a = np.log(a)
    log_b = np.log(b)
    
    # term2使用log(N * nij) = log(N) + log(nij)
    log_Nnij = np.log(n_samples) + np.log(nijs)
    
    # term3非常大，并涉及许多阶乘。在对数空间中计算这些以防止溢出。
    gln_a = gammaln(a + 1)
    gln_b = gammaln(b + 1)
    gln_Na = gammaln(n_samples - a + 1)
    gln_Nb = gammaln(n_samples - b + 1)
    gln_Nnij = gammaln(nijs + 1) + gammaln(n_samples + 1)
    
    # emi本身是对各个值进行求和。
    for i in range(n_rows):
        for j in range(n_cols):
            start = max(1, a_view[i] - n_samples + b_view[j])
            end = min(a_view[i], b_view[j]) + 1
            for nij in range(start, end):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                # 分子为正，分母为负。
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                       - gln_Nnij[nij] - lgamma(a_view[i] - nij + 1)
                       - lgamma(b_view[j] - nij + 1)
                       - lgamma(n_samples - a_view[i] - b_view[j] + nij + 1))
                term3 = exp(gln)
                emi += (term1[nij] * term2 * term3)
    
    # 返回计算出的期望互信息
    return emi
```