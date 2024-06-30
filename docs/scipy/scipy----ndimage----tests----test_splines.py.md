# `D:\src\scipysrc\scipy\scipy\ndimage\tests\test_splines.py`

```
"""Tests for spline filtering."""  # 导入 spline 滤波的测试模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库

from numpy.testing import assert_almost_equal  # 导入 NumPy 测试工具中的近似相等断言
from scipy import ndimage  # 导入 SciPy 中的图像处理模块

def get_spline_knot_values(order):
    """获取 B 样条中心右侧的节点值。"""
    knot_values = {0: [1],  # 0阶 B 样条的节点值
                   1: [1],  # 1阶 B 样条的节点值
                   2: [6, 1],  # 2阶 B 样条的节点值
                   3: [4, 1],  # 3阶 B 样条的节点值
                   4: [230, 76, 1],  # 4阶 B 样条的节点值
                   5: [66, 26, 1]}  # 5阶 B 样条的节点值

    return knot_values[order]  # 返回指定阶数的节点值列表


def make_spline_knot_matrix(n, order, mode='mirror'):
    """生成用于求解 B 样条系数的矩阵。"""
    knot_values = get_spline_knot_values(order)  # 获取指定阶数的节点值

    matrix = np.zeros((n, n))  # 创建一个 n x n 的零矩阵
    for diag, knot_value in enumerate(knot_values):
        indices = np.arange(diag, n)  # 生成从 diag 到 n-1 的索引数组
        if diag == 0:
            matrix[indices, indices] = knot_value  # 主对角线赋值为节点值
        else:
            matrix[indices, indices - diag] = knot_value  # 主对角线下方的对角线赋值为节点值
            matrix[indices - diag, indices] = knot_value  # 主对角线上方的对角线赋值为节点值

    knot_values_sum = knot_values[0] + 2 * sum(knot_values[1:])  # 计算节点值之和

    if mode == 'mirror':
        start, step = 1, 1  # 镜像模式下的起始值和步长
    elif mode == 'reflect':
        start, step = 0, 1  # 反射模式下的起始值和步长
    elif mode == 'grid-wrap':
        start, step = -1, -1  # 网格包裹模式下的起始值和步长
    else:
        raise ValueError(f'unsupported mode {mode}')  # 抛出不支持的模式异常

    for row in range(len(knot_values) - 1):
        for idx, knot_value in enumerate(knot_values[row + 1:]):
            matrix[row, start + step*idx] += knot_value  # 更新矩阵的指定位置的值
            matrix[-row - 1, -start - 1 - step*idx] += knot_value  # 更新矩阵的指定位置的值

    return matrix / knot_values_sum  # 返回归一化后的矩阵


@pytest.mark.parametrize('order', [0, 1, 2, 3, 4, 5])  # 参数化测试：阶数
@pytest.mark.parametrize('mode', ['mirror', 'grid-wrap', 'reflect'])  # 参数化测试：模式
def test_spline_filter_vs_matrix_solution(order, mode):
    """测试 spline 滤波与矩阵解法的结果是否一致。"""
    n = 100  # 矩阵大小
    eye = np.eye(n, dtype=float)  # 创建 n x n 的单位矩阵
    spline_filter_axis_0 = ndimage.spline_filter1d(eye, axis=0, order=order,
                                                   mode=mode)  # 对单位矩阵进行沿 axis=0 方向的 spline 滤波
    spline_filter_axis_1 = ndimage.spline_filter1d(eye, axis=1, order=order,
                                                   mode=mode)  # 对单位矩阵进行沿 axis=1 方向的 spline 滤波
    matrix = make_spline_knot_matrix(n, order, mode=mode)  # 生成指定模式下的 B 样条系数矩阵
    assert_almost_equal(eye, np.dot(spline_filter_axis_0, matrix))  # 断言两个矩阵近似相等
    assert_almost_equal(eye, np.dot(spline_filter_axis_1, matrix.T))  # 断言两个矩阵近似相等（转置后）
```