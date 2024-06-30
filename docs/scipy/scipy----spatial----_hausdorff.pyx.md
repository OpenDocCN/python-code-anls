# `D:\src\scipysrc\scipy\scipy\spatial\_hausdorff.pyx`

```
# cython: cpow=True
# 指定Cython编译器选项，启用对复数幂函数的支持

"""
Directed Hausdorff Code

.. versionadded:: 0.19.0

"""
# 版本信息和版权声明
# 版权所有 Tyler Reddy、Richard Gowers 和 Max Linke，2016年
# 使用与Scipy相同的BSD许可证进行分发

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

np.import_array()

__all__ = ['directed_hausdorff']

@cython.boundscheck(False)
# 使用Cython装饰器禁用数组边界检查，以提高性能
def directed_hausdorff(const double[:,::1] ar1, const double[:,::1] ar2, seed=0):
    # 函数定义，接受两个二维双精度浮点型数组和一个可选的种子值作为参数

    cdef double cmax, cmin, d = 0
    # 声明C语言级别的double类型变量cmax、cmin和d，初始化为0
    cdef Py_ssize_t N1 = ar1.shape[0]
    # 声明N1为ar1数组的行数
    cdef Py_ssize_t N2 = ar2.shape[0]
    # 声明N2为ar2数组的行数
    cdef int data_dims = ar1.shape[1]
    # 声明data_dims为ar1数组的列数
    cdef Py_ssize_t i, j, k
    # 声明循环变量i、j、k为Py_ssize_t类型
    cdef Py_ssize_t j_store = 0, i_ret = 0, j_ret = 0
    # 声明j_store、i_ret、j_ret为Py_ssize_t类型，初始化为0
    cdef np.ndarray[np.int64_t, ndim=1, mode='c'] resort1, resort2
    # 声明resort1和resort2为一维、C连续布局的64位整数NumPy数组

    # shuffling the points in each array generally increases the likelihood of
    # an advantageous break in the inner search loop and never decreases the
    # performance of the algorithm
    # 在每个数组中随机重排点，通常增加内部搜索循环中有利的中断可能性，且不会降低算法性能
    rng = np.random.RandomState(seed)
    # 创建一个指定种子的随机数生成器对象rng
    resort1 = np.arange(N1, dtype=np.int64)
    # 创建一个包含N1个整数的一维数组resort1，并初始化为[0, 1, ..., N1-1]
    resort2 = np.arange(N2, dtype=np.int64)
    # 创建一个包含N2个整数的一维数组resort2，并初始化为[0, 1, ..., N2-1]
    rng.shuffle(resort1)
    # 使用rng对象随机打乱resort1数组的顺序
    rng.shuffle(resort2)
    # 使用rng对象随机打乱resort2数组的顺序
    ar1 = np.asarray(ar1)[resort1]
    # 将ar1数组按照resort1的顺序重新排序
    ar2 = np.asarray(ar2)[resort2]
    # 将ar2数组按照resort2的顺序重新排序

    cmax = 0
    # 初始化cmax为0
    for i in range(N1):
        # 外部循环遍历ar1数组的每一行
        cmin = np.inf
        # 初始化cmin为无穷大
        for j in range(N2):
            # 内部循环遍历ar2数组的每一行
            d = 0
            # 初始化d为0
            # faster performance with square of distance
            # avoid sqrt until very end
            # 使用距离的平方可以提高性能，在最后避免使用sqrt函数
            for k in range(data_dims):
                # 最内层循环遍历数据维度
                d += (ar1[i, k] - ar2[j, k])**2
                # 计算欧氏距离的平方并累加到d中
            if d < cmax: # break out of `for j` loop
                # 如果d小于cmax，则跳出内部的for j循环
                break

            if d < cmin: # always true on first iteration of for-j loop
                # 如果d小于cmin，则总是成立（在for-j循环的第一次迭代中）
                cmin = d
                # 更新cmin为d
                j_store = j
                # 更新j_store为当前的j值

        # Note: The reference paper by A. A. Taha and A. Hanbury has this line
        # (Algorithm 2, line 16) as:
        #
        # if cmin > cmax:
        #
        # That logic is incorrect, as cmin could still be np.inf if breaking early.
        # The logic here accounts for that case.
        # 注释：A. A. Taha 和 A. Hanbury 的参考文献中的这行
        # （算法2，第16行）为：
        #
        # if cmin > cmax:
        #
        # 这个逻辑是错误的，因为如果提前中断，cmin仍可能是np.inf。
        # 这里的逻辑考虑了这种情况。
        if cmin >= cmax and d >= cmax:
            # 如果cmin大于等于cmax且d大于等于cmax
            cmax = cmin
            # 更新cmax为cmin
            i_ret = i
            # 更新i_ret为当前的i值
            j_ret = j_store
            # 更新j_ret为j_store的值

    return (sqrt(cmax), resort1[i_ret], resort2[j_ret])
    # 返回结果元组，包含sqrt(cmax)、resort1中的i_ret索引和resort2中的j_ret索引
```