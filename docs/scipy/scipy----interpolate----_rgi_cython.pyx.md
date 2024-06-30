# `D:\src\scipysrc\scipy\scipy\interpolate\_rgi_cython.pyx`

```
# cython: language_level=3
"""
RegularGridInterpolator的Cython化例程。
"""

# 导入必要的C库函数
from libc.math cimport NAN
# 导入NumPy和Cython专用的NumPy
import numpy as np
cimport numpy as np
# 导入Cython库
cimport cython

# 导入_poly_common.pxi中定义的内容
include "_poly_common.pxi"

# 导入NumPy的C函数库
np.import_array()

# 使用Cython装饰器设置函数的边界检查和数组访问检查为关闭状态
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
# 定义evaluate_linear_2d函数，用于二维线性插值
def evaluate_linear_2d(const double_or_complex[:, :] values,  # 二维数组，存储值（实数或复数）
                       const np.intp_t[:, :] indices,         # 二维整数数组，存储索引
                       const double[:, :] norm_distances,     # 二维双精度浮点数数组，存储规范化距离
                       tuple grid not None,                   # 元组grid，不能为空
                       double_or_complex[:] out):             # 一维数组，用于存储输出结果
    cdef:
        long num_points = indices.shape[1]      # 获取点的数量，indices的列数
        long i0, i1, point                      # 声明整型变量
        double_or_complex y0, y1, result        # 声明实数或复数变量

    assert out.shape[0] == num_points           # 断言输出数组的长度等于点的数量

    if grid[1].shape[0] == 1:
        # 如果grid的第二维长度为1，则沿axis=0进行线性插值
        for point in range(num_points):
            i0 = indices[0, point]
            if i0 >= 0:
                y0 = norm_distances[0, point]
                result = values[i0, 0]*(1 - y0) + values[i0+1, 0]*y0
                out[point] = result
            else:
                # 如果xi是nan，则find_interval返回-1
                out[point] = NAN
    elif grid[0].shape[0] == 1:
        # 如果grid的第一维长度为1，则沿axis=1进行线性插值
        for point in range(num_points):
            i1 = indices[1, point]
            if i1 >= 0:
                y1 = norm_distances[1, point]
                result = values[0, i1]*(1 - y1) + values[0, i1+1]*y1
                out[point] = result
            else:
                # 如果xi是nan，则find_interval返回-1
                out[point] = NAN
    else:
        # 否则进行双线性插值
        for point in range(num_points):
            i0, i1 = indices[0, point], indices[1, point]
            if i0 >=0 and i1 >=0:
                y0, y1 = norm_distances[0, point], norm_distances[1, point]

                result = 0.0
                result = result + values[i0, i1] * (1 - y0) * (1 - y1)
                result = result + values[i0, i1+1] * (1 - y0) * y1
                result = result + values[i0+1, i1] * y0 * (1 - y1)
                result = result + values[i0+1, i1+1] * y0 * y1
                out[point] = result
            else:
                # 如果xi是nan，则设置输出为NAN
                out[point] = NAN

    return np.asarray(out)


# 使用Cython装饰器设置函数的边界检查和数组访问检查为关闭状态，并启用C除法
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
# 定义find_indices函数，用于查找索引
def find_indices(tuple grid not None, const double[:, :] xi):
    # xi参数被声明为const，表示它是只读的
    cdef:
        # 声明整型变量
        long i, j, grid_i_size
        # 声明双精度浮点数变量
        double denom, value
        # 声明常量数组，grid_i 是只读的
        const double[::1] grid_i

        # 定义需要迭代的轴的大小
        long I = xi.shape[0]
        long J = xi.shape[1]

        # 初始化索引变量
        int index = 0

        # 用于存储 xi 所在的相关边界的索引数组
        np.intp_t[:,::1] indices = np.empty_like(xi, dtype=np.intp)

        # 初始化距离下边界的单位距离数组
        double[:,::1] norm_distances = np.zeros_like(xi, dtype=float)

    # 迭代各维度
    for i in range(I):
        # 获取当前维度上的网格数据
        grid_i = grid[i]
        # 获取当前网格数据的大小
        grid_i_size = grid_i.shape[0]

        # 处理长度为1的特殊情况
        if grid_i_size == 1:
            # 对于长度为1的维度，特殊处理
            for j in range(J):
                # 设置索引为 -1，这是一个临时解决方案，后续需要通过重构 evaluate_linear 来处理
                indices[i, j] = -1
                # norm_distances[i, j] 已经是零，无需额外操作
        else:
            # 处理一般情况
            for j in range(J):
                # 获取当前位置的 xi 值
                value = xi[i, j]
                # 查找 value 在 grid_i 中的区间，并返回索引
                index = find_interval_ascending(&grid_i[0],
                                                grid_i_size,
                                                value,
                                                prev_interval=index,
                                                extrapolate=1)
                # 将找到的索引赋值给 indices 数组
                indices[i, j] = index

                # 如果 value 是一个有效数值
                if value == value:
                    # 计算归一化距离
                    denom = grid_i[index + 1] - grid_i[index]
                    norm_distances[i, j] = (value - grid_i[index]) / denom
                else:
                    # 如果 xi[i, j] 是 NaN，将 norm_distances 设置为 NaN
                    norm_distances[i, j] = NAN

    # 返回处理后的数组
    return np.asarray(indices), np.asarray(norm_distances)
```