# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_cdnmf_fast.pyx`

```
# 从Cython库中导入特定的类型和函数
from cython cimport floating
from libc.math cimport fabs

# 定义一个名为 _update_cdnmf_fast 的函数，实现快速更新CDNMF算法的一部分
def _update_cdnmf_fast(floating[:, ::1] W, floating[:, :] HHt,
                       floating[:, :] XHt, Py_ssize_t[::1] permutation):
    # 定义本地变量
    cdef:
        floating violation = 0  # 违规度量，初始化为0
        Py_ssize_t n_components = W.shape[1]  # 成分数目，即W的列数
        Py_ssize_t n_samples = W.shape[0]  # 样本数目，即W的行数，也是H更新的特征数目
        floating grad, pg, hess  # 梯度、投影梯度、Hessian矩阵的元素
        Py_ssize_t i, r, s, t  # 循环变量

    # 使用 nogil 关键字进入无全局解锁区域，允许并行执行以下代码
    with nogil:
        # 对每个成分s进行循环
        for s in range(n_components):
            t = permutation[s]  # 根据排列顺序选择目标成分t

            # 对每个样本i进行循环，计算梯度
            for i in range(n_samples):
                # 计算梯度 = GW[t, i]，其中 GW = np.dot(W, HHt) - XHt
                grad = -XHt[i, t]

                for r in range(n_components):
                    grad += HHt[t, r] * W[i, r]

                # 计算投影梯度
                pg = min(0., grad) if W[i, t] == 0 else grad
                violation += fabs(pg)  # 累积违规度量

                # 计算Hessian矩阵元素
                hess = HHt[t, t]

                # 如果Hessian矩阵元素非零，更新W矩阵元素
                if hess != 0:
                    W[i, t] = max(W[i, t] - grad / hess, 0.)

    # 返回违规度量
    return violation
```