# `D:\src\scipysrc\scipy\scipy\linalg\_matfuncs_sqrtm_triu.pyx`

```
# cython: boundscheck=False, wraparound=False, cdivision=True
# 导入自定义异常类SqrtmError，用于抛出特定错误信息
from ._matfuncs_sqrtm import SqrtmError

# 导入必要的Cython类型
from numpy cimport complex128_t, float64_t, intp_t

# 定义浮点数类型的别名
cdef fused floating:
    float64_t
    complex128_t

# 定义函数within_block_loop，接收特定参数和变量进行块内循环计算
def within_block_loop(floating[:,::1] R, const floating[:,::1] T, start_stop_pairs, intp_t nblocks):
    # 声明循环中使用的整数变量和浮点数变量
    cdef intp_t start, stop, i, j, k
    cdef floating s, denom, num

    # 外层循环遍历start_stop_pairs中的每一对(start, stop)
    for start, stop in start_stop_pairs:
        # 第二层循环遍历从start到stop之间的每一个列索引j
        for j in range(start, stop):
            # 第三层循环遍历从j-1到start-1之间的每一个行索引i，反向遍历
            for i in range(j-1, start-1, -1):
                s = 0
                # 如果j - i > 1，计算R[i,i+1:j] @ R[i+1:j,j]的内积s
                if j - i > 1:
                    for k in range(i + 1, j):
                        s += R[i,k] * R[k,j]

                # 计算denom和num的值
                denom = R[i, i] + R[j, j]
                num = T[i, j] - s

                # 根据denom和num的值更新R[i, j]的值
                if denom != 0:
                    R[i, j] = (T[i, j] - s) / denom
                elif denom == 0 and num == 0:
                    R[i, j] = 0
                else:
                    # 如果denom为0且num不为0，抛出SqrtmError异常
                    raise SqrtmError('failed to find the matrix square root')
```