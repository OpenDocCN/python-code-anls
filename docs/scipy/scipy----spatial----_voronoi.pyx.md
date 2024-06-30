# `D:\src\scipysrc\scipy\scipy\spatial\_voronoi.pyx`

```
"""
Spherical Voronoi Cython Code

.. versionadded:: 0.19.0

"""
#
# Copyright (C)  Tyler Reddy, 2016
#
# Distributed under the same BSD license as Scipy.
#

import numpy as np                     # 导入 NumPy 库
cimport numpy as np                    # 使用 NumPy 的 C 扩展功能
cimport cython                         # 导入 Cython 声明

np.import_array()                      # 导入 NumPy 的 C API

__all__ = ['sort_vertices_of_regions']  # 模块的公开接口，包括函数 `sort_vertices_of_regions`


@cython.boundscheck(False)
def sort_vertices_of_regions(const int[:,::1] simplices, list regions):
    # 定义 Cython 函数，用于对 Voronoi 区域的顶点排序
    cdef np.npy_intp n, k, s, i, max_len  # 定义 Cython 的整数类型变量
    cdef np.npy_intp num_regions = len(regions)  # 获取 Voronoi 区域的数量
    cdef np.npy_intp current_simplex, current_vertex  # 当前单纯形和当前顶点
    cdef np.npy_intp remaining_size  # 剩余顶点数
    cdef np.npy_intp[:] remaining    # 剩余顶点数组
    cdef np.ndarray[np.intp_t, ndim=1] sorted_vertices  # 排序后的顶点数组

    max_len = 0  # 初始化最大长度为 0
    for region in regions:
        max_len = max(max_len, len(region))  # 计算最大的区域长度

    sorted_vertices = np.empty(max_len, dtype=np.intp)  # 创建空的排序后顶点数组

    for n in range(num_regions):
        remaining = np.asarray(regions[n][:])  # 获取当前区域的剩余顶点数组
        remaining_size = remaining.shape[0]    # 获取剩余顶点数组的大小
        current_simplex = remaining[0]         # 当前单纯形为剩余顶点数组的第一个元素

        # 找到当前单纯形的相邻顶点作为起始顶点
        for i in range(3):
            k = simplices[current_simplex, i]
            if k != n:
                current_vertex = k
                break

        # 对剩余顶点数组进行排序
        for k in range(remaining_size):
            sorted_vertices[k] = current_simplex  # 将当前单纯形加入排序后的顶点数组
            for i in range(remaining_size):
                if remaining[i] == sorted_vertices[k]:
                    continue
                s = remaining[i]
                # 找到与当前顶点相连的单纯形
                for j in range(3):
                    if current_vertex == simplices[s, j]:
                        current_simplex = s
            # 更新当前顶点为与当前单纯形不同且不是当前顶点的顶点
            for i in range(3):
                s = simplices[current_simplex, i]
                if s != n and s != current_vertex:
                    current_vertex = s
                    break

        regions[n] = list(sorted_vertices[:remaining_size])  # 更新区域中的顶点顺序
```