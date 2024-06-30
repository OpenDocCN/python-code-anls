# `D:\src\scipysrc\scipy\scipy\spatial\_qhull.pxd`

```
# -*-cython-*-
"""
Qhull shared definitions, for use by other Cython modules

"""
#
# Copyright (C)  Pauli Virtanen, 2010.
#
# Distributed under the same BSD license as Scipy.
#

# 导入 NumPy 的 ndarrayobject.h 文件中定义的内容
cdef extern from "numpy/ndarrayobject.h":
    # 定义 NumPy 的枚举常量 NPY_MAXDIMS
    cdef enum:
        NPY_MAXDIMS

# 定义 Cython 结构体 DelaunayInfo_t，用于存储 Delaunay 三角化的相关信息
ctypedef struct DelaunayInfo_t:
    int ndim  # 维度数
    int npoints  # 点的数量
    int nsimplex  # 单形数量
    double *points  # 点的数组指针
    int *simplices  # 单形数组指针
    int *neighbors  # 邻居数组指针
    double *equations  # 方程组数组指针
    double *transform  # 变换数组指针
    int *vertex_to_simplex  # 顶点到单形的映射数组指针
    double paraboloid_scale  # 抛物面缩放比例
    double paraboloid_shift  # 抛物面偏移量
    double *max_bound  # 最大边界数组指针
    double *min_bound  # 最小边界数组指针
    int *vertex_neighbors_indices  # 顶点邻居索引数组指针
    int *vertex_neighbors_indptr  # 顶点邻居指针数组指针

# 定义获取 Delaunay 信息的函数，返回整数，可能抛出异常（-1 表示异常）
cdef int _get_delaunay_info(DelaunayInfo_t *, obj,
                            int compute_transform,
                            int compute_vertex_to_simplex,
                            int compute_vertex_neighbors) except -1

#
# N-D geometry
#

# 判断点是否在 N 维空间中的单形内部，返回整数
cdef int _barycentric_inside(int ndim, double *transform,
                             const double *x, double *c, double eps) noexcept nogil

# 计算单个点在 N 维空间中的重心坐标，无异常抛出，无 GIL
cdef void _barycentric_coordinate_single(int ndim, double *transform,
                                         const double *x, double *c, int i) noexcept nogil

# 计算点在 N 维空间中的重心坐标，无异常抛出，无 GIL
cdef void _barycentric_coordinates(int ndim, double *transform,
                                   const double *x, double *c) noexcept nogil

#
# N+1-D geometry
#

# 将点从 N 维空间提升到 N+1 维空间，无异常抛出，无 GIL
cdef void _lift_point(DelaunayInfo_t *d, const double *x, double *z) noexcept nogil

# 计算点到平面的距离，返回双精度浮点数，无异常抛出，无 GIL
cdef double _distplane(DelaunayInfo_t *d, int isimplex, double *point) noexcept nogil

#
# Finding simplices
#

# 判断点是否完全在单形外部，返回整数，无异常抛出，无 GIL
cdef int _is_point_fully_outside(DelaunayInfo_t *d, const double *x, double eps) noexcept nogil

# 在 DelaunayInfo_t 对象中使用蛮力方法查找单形，返回整数，无异常抛出，无 GIL
cdef int _find_simplex_bruteforce(DelaunayInfo_t *d, double *c, const double *x,
                                  double eps, double eps_broad) noexcept nogil

# 在 DelaunayInfo_t 对象中使用定向方法查找单形，返回整数，无异常抛出，无 GIL
cdef int _find_simplex_directed(DelaunayInfo_t *d, double *c, const double *x,
                                int *start, double eps, double eps_broad) noexcept nogil

# 在 DelaunayInfo_t 对象中查找单形，返回整数，无异常抛出，无 GIL
cdef int _find_simplex(DelaunayInfo_t *d, double *c, const double *x, int *start,
                       double eps, double eps_broad) noexcept nogil
```