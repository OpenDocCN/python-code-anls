# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_simplification.py`

```
# 导入必要的模块和库
import base64  # 导入 base64 编码相关功能
import io  # 导入用于处理 IO 的模块
import platform  # 导入用于获取平台信息的模块

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import assert_array_almost_equal, assert_array_equal  # 导入用于测试的 NumPy 函数

import pytest  # 导入 pytest 测试框架

from matplotlib.testing.decorators import (
    check_figures_equal, image_comparison, remove_ticks_and_titles)  # 导入 Matplotlib 测试相关装饰器
import matplotlib.pyplot as plt  # 导入 Matplotlib 绘图功能

from matplotlib import patches, transforms  # 导入 Matplotlib 的图形相关模块
from matplotlib.path import Path  # 导入 Matplotlib 中的路径类


# NOTE: All of these tests assume that path.simplify is set to True
# (the default)

# 使用 image_comparison 装饰器测试图像是否一致
@image_comparison(['clipping'], remove_text=True)
def test_clipping():
    # 创建时间数组 t 和对应的正弦值数组 s
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2*np.pi*t)

    # 创建图形和坐标系
    fig, ax = plt.subplots()
    # 绘制正弦曲线
    ax.plot(t, s, linewidth=1.0)
    # 设置 y 轴范围
    ax.set_ylim((-0.20, -0.28))


# 使用 image_comparison 装饰器测试图像是否一致，并设置容差
@image_comparison(['overflow'], remove_text=True,
                  tol=0.007 if platform.machine() == 'arm64' else 0)
def test_overflow():
    # 创建包含不同大小数值的数组 x 和对应的索引数组 y
    x = np.array([1.0, 2.0, 3.0, 2.0e5])
    y = np.arange(len(x))

    # 创建图形和坐标系
    fig, ax = plt.subplots()
    # 绘制 x 和 y 的关系图
    ax.plot(x, y)
    # 设置 x 轴范围
    ax.set_xlim(2, 6)


# 使用 image_comparison 装饰器测试图像是否一致
@image_comparison(['clipping_diamond'], remove_text=True)
def test_diamond():
    # 创建表示菱形顶点的数组 x 和 y
    x = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    y = np.array([1.0, 0.0, -1.0, 0.0, 1.0])

    # 创建图形和坐标系
    fig, ax = plt.subplots()
    # 绘制菱形
    ax.plot(x, y)
    # 设置 x 和 y 轴范围
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)


def test_clipping_out_of_bounds():
    # 对没有代码的路径进行剪切
    path = Path([(0, 0), (1, 2), (2, 1)])
    simplified = path.cleaned(clip=(10, 10, 20, 20))
    assert_array_equal(simplified.vertices, [(0, 0)])
    assert simplified.codes == [Path.STOP]

    # 对包含代码但没有曲线的路径进行剪切
    path = Path([(0, 0), (1, 2), (2, 1)],
                [Path.MOVETO, Path.LINETO, Path.LINETO])
    simplified = path.cleaned(clip=(10, 10, 20, 20))
    assert_array_equal(simplified.vertices, [(0, 0)])
    assert simplified.codes == [Path.STOP]

    # 对包含曲线的路径暂不进行剪切
    path = Path([(0, 0), (1, 2), (2, 3)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    simplified = path.cleaned()
    simplified_clipped = path.cleaned(clip=(10, 10, 20, 20))
    assert_array_equal(simplified.vertices, simplified_clipped.vertices)
    assert_array_equal(simplified.codes, simplified_clipped.codes)


def test_noise():
    # 设置随机种子
    np.random.seed(0)
    # 创建随机数数组 x
    x = np.random.uniform(size=50000) * 50

    # 创建图形和坐标系
    fig, ax = plt.subplots()
    # 绘制 x 的折线图，设置线段连接方式和线宽
    p1 = ax.plot(x, solid_joinstyle='round', linewidth=2.0)

    # 确保路径的变换考虑新的坐标轴限制
    fig.canvas.draw()
    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = path.cleaned(simplify=True)

    # 断言简化后的路径顶点数目
    assert simplified.vertices.size == 25512


def test_antiparallel_simplification():
    # 还未实现反向简化路径的测试
    # 定义一个函数用于简化给定的曲线数据，并返回简化后的路径对象

    def _get_simplified(x, y):
        # 创建一个新的图形和轴对象
        fig, ax = plt.subplots()
        # 绘制曲线并获取绘制对象
        p1 = ax.plot(x, y)

        # 获取绘制对象的路径对象
        path = p1[0].get_path()
        # 获取绘制对象的变换信息
        transform = p1[0].get_transform()
        # 根据绘制对象的变换信息对路径对象进行变换
        path = transform.transform_path(path)
        # 对路径对象进行简化处理
        simplified = path.cleaned(simplify=True)
        # 根据逆变换信息对简化后的路径对象进行变换
        simplified = transform.inverted().transform_path(simplified)

        # 返回简化后的路径对象
        return simplified

    # test ending on a maximum
    # 测试以最大值结束的情况
    x = [0, 0, 0, 0, 0, 1]
    y = [.5, 1, -1, 1, 2, .5]

    # 调用 _get_simplified 函数进行简化处理
    simplified = _get_simplified(x, y)

    # 断言简化后的路径顶点与预期结果的几乎相等
    assert_array_almost_equal([[0., 0.5],
                               [0., -1.],
                               [0., 2.],
                               [1., 0.5]],
                              simplified.vertices[:-2, :])

    # test ending on a minimum
    # 测试以最小值结束的情况
    x = [0, 0,  0, 0, 0, 1]
    y = [.5, 1, -1, 1, -2, .5]

    # 调用 _get_simplified 函数进行简化处理
    simplified = _get_simplified(x, y)

    # 断言简化后的路径顶点与预期结果的几乎相等
    assert_array_almost_equal([[0., 0.5],
                               [0., 1.],
                               [0., -2.],
                               [1., 0.5]],
                              simplified.vertices[:-2, :])

    # test ending in between
    # 测试在中间结束的情况
    x = [0, 0, 0, 0, 0, 1]
    y = [.5, 1, -1, 1, 0, .5]

    # 调用 _get_simplified 函数进行简化处理
    simplified = _get_simplified(x, y)

    # 断言简化后的路径顶点与预期结果的几乎相等
    assert_array_almost_equal([[0., 0.5],
                               [0., 1.],
                               [0., -1.],
                               [0., 0.],
                               [1., 0.5]],
                              simplified.vertices[:-2, :])

    # test no anti-parallel ending at max
    # 测试不以反平行结束的情况（以最大值结束）
    x = [0, 0, 0, 0, 0, 1]
    y = [.5, 1, 2, 1, 3, .5]

    # 调用 _get_simplified 函数进行简化处理
    simplified = _get_simplified(x, y)

    # 断言简化后的路径顶点与预期结果的几乎相等
    assert_array_almost_equal([[0., 0.5],
                               [0., 3.],
                               [1., 0.5]],
                              simplified.vertices[:-2, :])

    # test no anti-parallel ending in middle
    # 测试不以反平行结束的情况（在中间结束）
    x = [0, 0, 0, 0, 0, 1]
    y = [.5, 1, 2, 1, 1, .5]

    # 调用 _get_simplified 函数进行简化处理
    simplified = _get_simplified(x, y)

    # 断言简化后的路径顶点与预期结果的几乎相等
    assert_array_almost_equal([[0., 0.5],
                               [0., 2.],
                               [0., 1.],
                               [1., 0.5]],
                              simplified.vertices[:-2, :])
# Only consider angles in 0 <= angle <= pi/2, otherwise
# using min/max will get the expected results out of order:
# min/max for simplification code depends on original vector,
# and if angle is outside above range then simplification
# min/max will be opposite from actual min/max.
@pytest.mark.parametrize('angle', [0, np.pi/4, np.pi/3, np.pi/2])
@pytest.mark.parametrize('offset', [0, .5])
def test_angled_antiparallel(angle, offset):
    scale = 5
    np.random.seed(19680801)
    # get 15 random offsets
    # TODO: guarantee offset > 0 results in some offsets < 0
    # 生成15个随机偏移量
    vert_offsets = (np.random.rand(15) - offset) * scale
    # always start at 0 so rotation makes sense
    # 始终从0开始，确保旋转有意义
    vert_offsets[0] = 0
    # always take the first step the same direction
    # 始终朝同一方向迈出第一步
    vert_offsets[1] = 1
    # compute points along a diagonal line
    # 计算沿对角线的点
    x = np.sin(angle) * vert_offsets
    y = np.cos(angle) * vert_offsets

    # will check these later
    # 之后将检查这些值
    x_max = x[1:].max()
    x_min = x[1:].min()

    y_max = y[1:].max()
    y_min = y[1:].min()

    if offset > 0:
        # If offset > 0, create expected path with four points
        # 如果offset > 0，创建包含四个点的期望路径
        p_expected = Path([[0, 0],
                           [x_max, y_max],
                           [x_min, y_min],
                           [x[-1], y[-1]],
                           [0, 0]],
                          codes=[1, 2, 2, 2, 0])

    else:
        # If offset <= 0, create expected path with three points
        # 如果offset <= 0，创建包含三个点的期望路径
        p_expected = Path([[0, 0],
                           [x_max, y_max],
                           [x[-1], y[-1]],
                           [0, 0]],
                          codes=[1, 2, 2, 0])

    p = Path(np.vstack([x, y]).T)
    p2 = p.cleaned(simplify=True)

    assert_array_almost_equal(p_expected.vertices,
                              p2.vertices)
    assert_array_equal(p_expected.codes, p2.codes)


def test_sine_plus_noise():
    np.random.seed(0)
    x = (np.sin(np.linspace(0, np.pi * 2.0, 50000)) +
         np.random.uniform(size=50000) * 0.01)

    fig, ax = plt.subplots()
    p1 = ax.plot(x, solid_joinstyle='round', linewidth=2.0)

    # Ensure that the path's transform takes the new axes limits into account.
    # 确保路径的变换考虑了新的坐标轴限制
    fig.canvas.draw()
    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = path.cleaned(simplify=True)

    assert simplified.vertices.size == 25240


@image_comparison(['simplify_curve'], remove_text=True, tol=0.017)
def test_simplify_curve():
    pp1 = patches.PathPatch(
        Path([(0, 0), (1, 0), (1, 1), (np.nan, 1), (0, 0), (2, 0), (2, 2),
              (0, 0)],
             [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3,
              Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
        fc="none")

    fig, ax = plt.subplots()
    ax.add_patch(pp1)
    ax.set_xlim((0, 2))
    ax.set_ylim((0, 2))


@check_figures_equal()
def test_closed_path_nan_removal(fig_test, fig_ref):
    ax_test = fig_test.subplots(2, 2).flatten()
    ax_ref = fig_ref.subplots(2, 2).flatten()

    # NaN on the first point also removes the last point, because it's closed.
    # 第一个点是NaN时，也会移除最后一个点，因为它是闭合的。
    # 创建一个包含五个点的路径对象，描述一个多边形，最后一个点使用 Path.CLOSEPOLY 关闭路径
    path = Path(
        [[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, -3]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    # 将路径对象添加到第一个子图中作为一个不填充的路径补丁
    ax_test[0].add_patch(patches.PathPatch(path, facecolor='none'))

    # 创建一个包含五个点的路径对象，描述一个多边形，所有点使用 Path.LINETO 连接
    path = Path(
        [[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, np.nan]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    # 将路径对象添加到第一个子图中作为一个不填充的路径补丁
    ax_ref[0].add_patch(patches.PathPatch(path, facecolor='none'))

    # 创建一个路径对象，包含五个点，其中第四个点的 Y 值为 NaN，使用 Path.CLOSEPOLY 关闭路径
    path = Path(
        [[-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    # 将路径对象添加到第一个子图中作为一个不填充的路径补丁
    ax_test[0].add_patch(patches.PathPatch(path, facecolor='none'))

    # 创建一个路径对象，包含五个点，其中第四个点的 Y 值为 NaN，所有点使用 Path.LINETO 连接
    path = Path(
        [[-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    # 将路径对象添加到第一个子图中作为一个不填充的路径补丁
    ax_ref[0].add_patch(patches.PathPatch(path, facecolor='none'))

    # 创建一个路径对象，描述包含两个独立循环的多边形，每个循环中的点使用不同的连接方式
    path = Path(
        [[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, -3],
         [-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,
         Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    # 将路径对象添加到第二个子图中作为一个不填充的路径补丁
    ax_test[1].add_patch(patches.PathPatch(path, facecolor='none'))

    # 创建一个路径对象，描述包含两个独立循环的多边形，每个循环中的点使用相同的连接方式
    path = Path(
        [[-3, np.nan], [3, -3], [3, 3], [-3, 3], [-3, np.nan],
         [-2, -2], [2, -2], [2, 2], [-2, np.nan], [-2, -2]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
         Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    # 将路径对象添加到第二个子图中作为一个不填充的路径补丁
    ax_ref[1].add_patch(patches.PathPatch(path, facecolor='none'))

    # 创建一个路径对象，描述包含一条 CURVE3 和一条 LINETO 的路径，CURVE3 中第一个点的 Y 值为 NaN
    path = Path(
        [[-1, -1], [1, -1], [1, np.nan], [0, 1], [-1, 1], [-1, -1]],
        [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
         Path.CLOSEPOLY])
    # 将路径对象添加到第三个子图中作为一个不填充的路径补丁
    ax_test[2].add_patch(patches.PathPatch(path, facecolor='none'))

    # 创建一个路径对象，描述包含一条 CURVE3 和一条 LINETO 的路径，CURVE3 中第一个点的 Y 值为 NaN
    path = Path(
        [[-1, -1], [1, -1], [1, np.nan], [0, 1], [-1, 1], [-1, -1]],
        [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
         Path.CLOSEPOLY])
    # 将路径对象添加到第三个子图中作为一个不填充的路径补丁
    ax_ref[2].add_patch(patches.PathPatch(path, facecolor='none'))

    # 创建一个路径对象，描述包含一条 CURVE3 和一条 LINETO 的路径，CURVE3 中第二个点的 X 值为 NaN
    path = Path(
        [[-3, -3], [3, -3], [3, 0], [0, np.nan], [-3, 3], [-3, -3]],
        [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
         Path.LINETO])
    # 将路径对象添加到第三个子图中作为一个不填充的路径补丁
    ax_test[2].add_patch(patches.PathPatch(path, facecolor='none'))

    # 创建一个路径对象，描述包含一条 CURVE3 和一条 LINETO 的路径，CURVE3 中第二个点的 X 值为 NaN
    path = Path(
        [[-3, -3], [3, -3], [3, 0], [0, np.nan], [-3, 3], [-3, -3]],
        [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
         Path.LINETO])
    # 将路径对象添加到第三个子图中作为一个不填充的路径补丁
    ax_ref[2].add_patch(patches.PathPatch(path, facecolor='none'))
    # NaN in first point of CURVE4 should not re-close, and hide entire curve.
    path = Path(
        [[-1, -1], [1, -1], [1, np.nan], [0, 0], [0, 1], [-1, 1], [-1, -1]],
        [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
         Path.LINETO, Path.CLOSEPOLY])
    # 将定义的路径添加到测试图表的第四个子图上，面颜色设置为无色
    ax_test[3].add_patch(patches.PathPatch(path, facecolor='none'))
    
    # 创建一个与上述相同的路径，并将其添加到参考图表的第四个子图上，面颜色设置为无色
    path = Path(
        [[-1, -1], [1, -1], [1, np.nan], [0, 0], [0, 1], [-1, 1], [-1, -1]],
        [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
         Path.LINETO, Path.CLOSEPOLY])
    ax_ref[3].add_patch(patches.PathPatch(path, facecolor='none'))

    # NaN in second point of CURVE4 should not re-close, and hide entire curve.
    path = Path(
        [[-2, -2], [2, -2], [2, 0], [0, np.nan], [0, 2], [-2, 2], [-2, -2]],
        [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
         Path.LINETO, Path.LINETO])
    # 将定义的路径添加到测试图表的第四个子图上，面颜色设置为无色
    ax_test[3].add_patch(patches.PathPatch(path, facecolor='none'))
    
    # 创建一个与上述相同的路径，并将其添加到参考图表的第四个子图上，面颜色设置为无色
    path = Path(
        [[-2, -2], [2, -2], [2, 0], [0, np.nan], [0, 2], [-2, 2], [-2, -2]],
        [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
         Path.LINETO, Path.LINETO])
    ax_ref[3].add_patch(patches.PathPatch(path, facecolor='none'))

    # NaN in third point of CURVE4 should not re-close, and hide entire curve
    # plus next line segment.
    path = Path(
        [[-3, -3], [3, -3], [3, 0], [0, 0], [0, np.nan], [-3, 3], [-3, -3]],
        [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
         Path.LINETO, Path.LINETO])
    # 将定义的路径添加到测试图表的第四个子图上，面颜色设置为无色
    ax_test[3].add_patch(patches.PathPatch(path, facecolor='none'))
    
    # 创建一个与上述相同的路径，并将其添加到参考图表的第四个子图上，面颜色设置为无色
    path = Path(
        [[-3, -3], [3, -3], [3, 0], [0, 0], [0, np.nan], [-3, 3], [-3, -3]],
        [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
         Path.LINETO, Path.LINETO])
    ax_ref[3].add_patch(patches.PathPatch(path, facecolor='none'))

    # Keep everything clean.
    # 设置所有测试图表和参考图表的子图的 X 轴和 Y 轴范围为 (-3.5, 3.5)，以保持清晰
    for ax in [*ax_test.flat, *ax_ref.flat]:
        ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
    # 移除测试图表的刻度和标题
    remove_ticks_and_titles(fig_test)
    # 移除参考图表的刻度和标题
    remove_ticks_and_titles(fig_ref)
# 使用装饰器 @check_figures_equal() 标记的测试函数，用于检查两个图形对象是否相等
@check_figures_equal()
def test_closed_path_clipping(fig_test, fig_ref):
    # 初始化空列表，用于存储多个顶点数组
    vertices = []
    # 循环8次，生成不同偏移的 U 形模式的顶点数组
    for roll in range(8):
        offset = 0.1 * roll + 0.1

        # 定义 U 形模式的顶点数组
        pattern = [
            [-0.5, 1.5], [-0.5, -0.5], [1.5, -0.5], [1.5, 1.5],  # 外部正方形
            # 顶部有一个凹口
            [1 - offset / 2, 1.5], [1 - offset / 2, offset],
            [offset / 2, offset], [offset / 2, 1.5],
        ]

        # 根据 roll 值将模式数组进行旋转
        pattern = np.roll(pattern, roll, axis=0)
        # 将数组的第一行复制到最后一行，闭合路径
        pattern = np.concatenate((pattern, pattern[:1, :]))

        # 将生成的顶点数组添加到列表中
        vertices.append(pattern)

    # 用于路径代码的数组，初始化为全部是 LINETO 类型
    codes = np.full(len(vertices[0]), Path.LINETO)
    # 将第一个点设为 MOVETO 类型
    codes[0] = Path.MOVETO
    # 将最后一个点设为 CLOSEPOLY 类型，闭合多边形
    codes[-1] = Path.CLOSEPOLY
    # 复制 codes 数组以适应多个子路径
    codes = np.tile(codes, len(vertices))
    # 将所有子路径的顶点数组连接起来
    vertices = np.concatenate(vertices)

    # 设置测试图形的尺寸为 (5, 5) 英寸
    fig_test.set_size_inches((5, 5))
    # 创建 Path 对象，使用顶点数组和代码数组
    path = Path(vertices, codes)
    # 在测试图形上添加 PathPatch 对象
    fig_test.add_artist(patches.PathPatch(path, facecolor='none'))

    # 在参考图形上绘制相同的图形，但是未闭合，使用线段到最后一个点
    fig_ref.set_size_inches((5, 5))
    # 复制 codes 数组，将 CLOSEPOLY 类型改为 LINETO 类型
    codes = codes.copy()
    codes[codes == Path.CLOSEPOLY] = Path.LINETO
    # 创建 Path 对象，使用顶点数组和修改后的代码数组
    path = Path(vertices, codes)
    # 在参考图形上添加 PathPatch 对象
    fig_ref.add_artist(patches.PathPatch(path, facecolor='none'))


# 使用装饰器 @image_comparison(['hatch_simplify'], remove_text=True) 标记的测试函数，用于比较图像是否一致
@image_comparison(['hatch_simplify'], remove_text=True)
def test_hatch():
    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上添加具有斜线填充的矩形对象
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, hatch="/"))
    # 设置坐标轴的 x 范围
    ax.set_xlim((0.45, 0.55))
    # 设置坐标轴的 y 范围
    ax.set_ylim((0.45, 0.55))


# 使用装饰器 @image_comparison(['fft_peaks'], remove_text=True) 标记的测试函数，用于比较图像是否一致
@image_comparison(['fft_peaks'], remove_text=True)
def test_fft_peaks():
    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 生成长度为 65536 的正弦波信号，并对其进行傅里叶变换，绘制其幅度谱
    t = np.arange(65536)
    p1 = ax.plot(abs(np.fft.fft(np.sin(2*np.pi*.01*t)*np.blackman(len(t)))))

    # 确保路径的变换考虑到新的坐标轴限制
    fig.canvas.draw()
    # 获取路径对象和其变换
    path = p1[0].get_path()
    transform = p1[0].get_transform()
    # 对路径进行坐标变换
    path = transform.transform_path(path)
    # 简化路径
    simplified = path.cleaned(simplify=True)

    # 断言简化后的顶点数是否为 36
    assert simplified.vertices.size == 36


# 未使用装饰器的普通测试函数，没有图像比较
def test_start_with_moveto():
    # 数据为二进制内容，表示一个路径
    data = b"""
    ZwAAAAku+v9UAQAA+Tj6/z8CAADpQ/r/KAMAANlO+v8QBAAAyVn6//UEAAC6ZPr/2gUAAKpv+v+8
    BgAAm3r6/50HAACLhfr/ewgAAHyQ+v9ZCQAAbZv6/zQKAABepvr/DgsAAE+x+v/lCwAAQLz6/7wM
    AAAxx/r/kA0AACPS+v9jDgAAFN36/zQPAAAF6Pr/AxAAAPfy+v/QEAAA6f36/5wRAADbCPv/ZhIA
    AMwT+/8uEwAAvh77//UTAACwKfv/uRQAAKM0+/98FQAAlT/7/z0WAACHSvv//RYAAHlV+/+7FwAA
    bGD7/3cYAABea/v/MRkAAFF2+//pGQAARIH7/6AaAAA3jPv/VRsAACmX+/8JHAAAHKL7/7ocAAAP
    rfv/ah0AAAO4+/8YHgAA9sL7/8QeAADpzfv/bx8AANzY+/8YIAAA0OP7/78gAADD7vv/ZCEAALf5
    +/8IIgAAqwT8/6kiAACeD/z/SiMAAJIa/P/oIwAAhiX8/4QkAAB6MPz/HyUAAG47/P+4JQAAYkb8
    /1AmAABWUfz/5SYAAEpc/P95JwAAPmf8/wsoAAAzcvz/nCgAACd9/P8qKQAAHIj8/7cpAAAQk/z/
    QyoAAAWe/P/MKgAA+aj8/1QrAADus/z/2isAAOO+/P9eLAAA2Mn8/+AsAADM1Pz/YS0AAMHf/P/g
    """

    # 此处代码被剪切到单个 MOVETO
# 定义一个 Base64 编码的字符串，包含了一组二进制数据
data = """
LQAAtur8/10uAACr9fz/2C4AAKEA/f9SLwAAlgv9/8ovAACLFv3/QDAAAIAh/f+1MAAAdSz9/ycx
AABrN/3/mDEAAGBC/f8IMgAAVk39/3UyAABLWP3/4TIAAEFj/f9LMwAANm79/7MzAAAsef3/GjQA
ACKE/f9+NAAAF4/9/+E0AAANmv3/QzUAAAOl/f+iNQAA+a/9/wA2AADvuv3/XDYAAOXF/f+2NgAA
29D9/w83AADR2/3/ZjcAAMfm/f+7NwAAvfH9/w44AACz/P3/XzgAAKkH/v+vOAAAnxL+//04AACW
Hf7/SjkAAIwo/v+UOQAAgjP+/905AAB5Pv7/JDoAAG9J/v9pOgAAZVT+/606AABcX/7/7zoAAFJq
/v8vOwAASXX+/207AAA/gP7/qjsAADaL/v/lOwAALZb+/x48AAAjof7/VTwAABqs/v+LPAAAELf+
/788AAAHwv7/8TwAAP7M/v8hPQAA9df+/1A9AADr4v7/fT0AAOLt/v+oPQAA2fj+/9E9AADQA///
+T0AAMYO//8fPgAAvRn//0M+AAC0JP//ZT4AAKsv//+GPgAAojr//6U+AACZRf//wj4AAJBQ///d
PgAAh1v///c+AAB+Zv//Dz8AAHRx//8lPwAAa3z//zk/AABih///TD8AAFmS//9dPwAAUJ3//2w/
AABHqP//ej8AAD6z//+FPwAANb7//48/AAAsyf//lz8AACPU//+ePwAAGt///6M/AAAR6v//pj8A
AAj1//+nPwAA/////w==
"""

# 使用 base64 解码将二进制数据转换为 numpy 的 32 位整数数组
verts = np.frombuffer(base64.decodebytes(data), dtype='<i4')
# 将一维数组重新形状为二维数组，每行包含两个整数
verts = verts.reshape((len(verts) // 2, 2))
# 创建 Path 对象，使用 verts 数组作为路径数据
path = Path(verts)
# 获取路径的迭代器，每次返回路径的一个线段
segs = path.iter_segments(transforms.IdentityTransform(),
                          clip=(0.0, 0.0, 100.0, 100.0))
# 将迭代器转换为列表，以便进行后续的断言和验证
segs = list(segs)
# 确保只有一个线段被返回
assert len(segs) == 1
# 确保返回的线段是一个 MOVETO 类型的路径指令
assert segs[0][1] == Path.MOVETO
    # 断言语句，用于检查表达式是否为真，若不为真则抛出 AssertionError 异常
    assert ([(list(x), y) for x, y in simplified] ==
            [([50, 40], 1)])
    # 上述断言检查两个列表推导式的结果是否相等，即检查 simplified 中的每对元素是否与指定的列表 `[([50, 40], 1)]` 相匹配
```