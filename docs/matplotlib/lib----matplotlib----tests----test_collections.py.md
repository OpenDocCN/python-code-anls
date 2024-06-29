# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_collections.py`

```py
# 导入所需模块和库
from datetime import datetime               # 导入 datetime 模块中的 datetime 类
import io                                   # 导入 io 模块，用于处理文件流
import itertools                            # 导入 itertools 模块，用于高效循环和迭代操作
import platform                             # 导入 platform 模块，用于获取平台信息
import re                                   # 导入 re 模块，用于正则表达式操作
from types import SimpleNamespace           # 导入 SimpleNamespace 类

import numpy as np                          # 导入 NumPy 库并使用 np 别名
from numpy.testing import assert_array_equal, assert_array_almost_equal  # 从 numpy.testing 模块导入数组比较函数
import pytest                                # 导入 pytest 测试框架

import matplotlib as mpl                    # 导入 Matplotlib 主模块并使用 mpl 别名
import matplotlib.pyplot as plt             # 导入 Matplotlib 的 pyplot 模块并使用 plt 别名
import matplotlib.collections as mcollections  # 导入 Matplotlib 的集合模块并使用 mcollections 别名
import matplotlib.colors as mcolors         # 导入 Matplotlib 的颜色模块并使用 mcolors 别名
import matplotlib.path as mpath             # 导入 Matplotlib 的路径模块并使用 mpath 别名
import matplotlib.transforms as mtransforms  # 导入 Matplotlib 的变换模块并使用 mtransforms 别名
from matplotlib.collections import (Collection, LineCollection,  # 从 Matplotlib 的集合模块导入特定的类
                                    EventCollection, PolyCollection)
from matplotlib.testing.decorators import check_figures_equal, image_comparison  # 导入 Matplotlib 的测试装饰器函数


@pytest.fixture(params=["pcolormesh", "pcolor"])
def pcfunc(request):
    return request.param
    # 定义 pytest 的参数化 fixture，用于测试不同的 pcolormesh 和 pcolor 参数


def generate_EventCollection_plot():
    """生成初始集合并绘制它。"""
    positions = np.array([0., 1., 2., 3., 5., 8., 13., 21.])  # 创建位置数组
    extra_positions = np.array([34., 55., 89.])               # 创建额外位置数组
    orientation = 'horizontal'                                # 设置方向为水平
    lineoffset = 1                                             # 设置线偏移量
    linelength = .5                                            # 设置线长度
    linewidth = 2                                              # 设置线宽度
    color = [1, 0, 0, 1]                                       # 设置颜色为红色
    linestyle = 'solid'                                        # 设置线型为实线
    antialiased = True                                         # 设置抗锯齿为开启

    coll = EventCollection(positions,                          # 创建事件集合对象
                           orientation=orientation,            # 设置方向属性
                           lineoffset=lineoffset,              # 设置线偏移属性
                           linelength=linelength,              # 设置线长度属性
                           linewidth=linewidth,                # 设置线宽属性
                           color=color,                        # 设置颜色属性
                           linestyle=linestyle,                # 设置线型属性
                           antialiased=antialiased             # 设置抗锯齿属性
                           )

    fig, ax = plt.subplots()                                   # 创建图形和轴对象
    ax.add_collection(coll)                                    # 将集合对象添加到轴上
    ax.set_title('EventCollection: default')                   # 设置图表标题
    props = {'positions': positions,                           # 定义属性字典
             'extra_positions': extra_positions,               # 包含位置和额外位置
             'orientation': orientation,                       # 方向
             'lineoffset': lineoffset,                         # 线偏移
             'linelength': linelength,                         # 线长度
             'linewidth': linewidth,                           # 线宽度
             'color': color,                                   # 颜色
             'linestyle': linestyle,                           # 线型
             'antialiased': antialiased                        # 抗锯齿
             }
    ax.set_xlim(-1, 22)                                        # 设置 X 轴范围
    ax.set_ylim(0, 2)                                          # 设置 Y 轴范围
    return ax, coll, props                                     # 返回轴对象、集合对象和属性字典


@image_comparison(['EventCollection_plot__default'])
def test__EventCollection__get_props():
    _, coll, props = generate_EventCollection_plot()           # 调用生成集合和绘制的函数
    # 检查默认段是否具有正确的坐标
    check_segments(coll,                                      # 调用检查段函数
                   props['positions'],                         # 使用属性字典中的位置
                   props['linelength'],                        # 使用属性字典中的线长度
                   props['lineoffset'],                        # 使用属性字典中的线偏移
                   props['orientation'])                       # 使用属性字典中的方向
    # 检查默认位置是否与输入位置匹配
    np.testing.assert_array_equal(props['positions'], coll.get_positions())
    # 检查默认方向是否与输入方向匹配
    assert props['orientation'] == coll.get_orientation()
    # 检查是否为水平方向
    assert coll.is_horizontal()
    # 检查默认线长度是否与输入线长度匹配
    assert props['linelength'] == coll.get_linelength()
    # 检查默认的行偏移是否与输入的行偏移匹配
    assert props['lineoffset'] == coll.get_lineoffset()
    
    # 检查默认的线条样式是否与输入的线条样式匹配
    assert coll.get_linestyle() == [(0, None)]
    
    # 检查默认的颜色是否与输入的颜色匹配
    # 对于每个颜色，包括单个颜色和多个颜色，使用 NumPy 测试确保其数组相等于 props 中的颜色数组
    for color in [coll.get_color(), *coll.get_colors()]:
        np.testing.assert_array_equal(color, props['color'])
# 使用装饰器来比较生成的图像与预期图像是否相同，测试事件集合的设置位置功能
@image_comparison(['EventCollection_plot__set_positions'])
def test__EventCollection__set_positions():
    # 生成事件集合的绘图对象、事件集合对象及其属性
    splt, coll, props = generate_EventCollection_plot()
    # 创建新的位置数组，包括主要位置和额外位置
    new_positions = np.hstack([props['positions'], props['extra_positions']])
    # 设置事件集合的位置
    coll.set_positions(new_positions)
    # 断言新位置数组与事件集合的当前位置相等
    np.testing.assert_array_equal(new_positions, coll.get_positions())
    # 检查线段的绘制是否符合预期
    check_segments(coll, new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    # 设置子图的标题为 'EventCollection: set_positions'
    splt.set_title('EventCollection: set_positions')
    # 设置子图的 x 轴范围
    splt.set_xlim(-1, 90)


# 使用装饰器来比较生成的图像与预期图像是否相同，测试事件集合的添加位置功能
@image_comparison(['EventCollection_plot__add_positions'])
def test__EventCollection__add_positions():
    # 生成事件集合的绘图对象、事件集合对象及其属性
    splt, coll, props = generate_EventCollection_plot()
    # 创建新的位置数组，包括主要位置和第一个额外位置
    new_positions = np.hstack([props['positions'],
                               props['extra_positions'][0]])
    # 切换事件集合的方向（测试在垂直方向添加位置）
    coll.switch_orientation()
    # 添加第一个额外位置到事件集合
    coll.add_positions(props['extra_positions'][0])
    # 切换事件集合的方向回水平方向
    coll.switch_orientation()
    # 断言新位置数组与事件集合的当前位置相等
    np.testing.assert_array_equal(new_positions, coll.get_positions())
    # 检查线段的绘制是否符合预期
    check_segments(coll,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    # 设置子图的标题为 'EventCollection: add_positions'
    splt.set_title('EventCollection: add_positions')
    # 设置子图的 x 轴范围
    splt.set_xlim(-1, 35)


# 使用装饰器来比较生成的图像与预期图像是否相同，测试事件集合的追加位置功能
@image_comparison(['EventCollection_plot__append_positions'])
def test__EventCollection__append_positions():
    # 生成事件集合的绘图对象、事件集合对象及其属性
    splt, coll, props = generate_EventCollection_plot()
    # 创建新的位置数组，包括主要位置和第三个额外位置
    new_positions = np.hstack([props['positions'],
                               props['extra_positions'][2]])
    # 追加第三个额外位置到事件集合
    coll.append_positions(props['extra_positions'][2])
    # 断言新位置数组与事件集合的当前位置相等
    np.testing.assert_array_equal(new_positions, coll.get_positions())
    # 检查线段的绘制是否符合预期
    check_segments(coll,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    # 设置子图的标题为 'EventCollection: append_positions'
    splt.set_title('EventCollection: append_positions')
    # 设置子图的 x 轴范围
    splt.set_xlim(-1, 90)


# 使用装饰器来比较生成的图像与预期图像是否相同，测试事件集合的扩展位置功能
@image_comparison(['EventCollection_plot__extend_positions'])
def test__EventCollection__extend_positions():
    # 生成事件集合的绘图对象、事件集合对象及其属性
    splt, coll, props = generate_EventCollection_plot()
    # 创建新的位置数组，包括主要位置和第二个到最后的额外位置
    new_positions = np.hstack([props['positions'],
                               props['extra_positions'][1:]])
    # 扩展事件集合的位置数组，从第二个额外位置开始
    coll.extend_positions(props['extra_positions'][1:])
    # 断言新位置数组与事件集合的当前位置相等
    np.testing.assert_array_equal(new_positions, coll.get_positions())
    # 检查线段的绘制是否符合预期
    check_segments(coll,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    # 设置子图的标题为 'EventCollection: extend_positions'
    splt.set_title('EventCollection: extend_positions')
    # 设置子图的 x 轴范围
    splt.set_xlim(-1, 90)


# 使用装饰器来比较生成的图像与预期图像是否相同，测试事件集合的切换方向功能
@image_comparison(['EventCollection_plot__switch_orientation'])
def test__EventCollection__switch_orientation():
    # 生成事件集合的绘图对象、事件集合对象及其属性
    splt, coll, props = generate_EventCollection_plot()
    # 新的方向为垂直方向
    new_orientation = 'vertical'
    # 切换事件集合的方向
    coll.switch_orientation()
    # 断言当前事件集合的方向与新方向相等
    assert new_orientation == coll.get_orientation()
    # 断言集合不是水平的（即不是横向的）
    assert not coll.is_horizontal()
    # 获取集合中对象的位置信息
    new_positions = coll.get_positions()
    # 检查集合的线段，使用新的位置信息、线段长度、线段偏移和新的方向
    check_segments(coll,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'], new_orientation)
    # 设置子图标题为 'EventCollection: switch_orientation'
    splt.set_title('EventCollection: switch_orientation')
    # 设置子图的纵轴范围为 -1 到 22
    splt.set_ylim(-1, 22)
    # 设置子图的横轴范围为 0 到 2
    splt.set_xlim(0, 2)
@image_comparison(['EventCollection_plot__switch_orientation__2x'])
def test__EventCollection__switch_orientation_2x():
    """
    Check that calling switch_orientation twice sets the orientation back to
    the default.
    """
    # 生成 EventCollection 绘图所需的数据：子图对象 splt，事件集合对象 coll，属性字典 props
    splt, coll, props = generate_EventCollection_plot()
    # 调用事件集合对象的 switch_orientation 方法，切换方向
    coll.switch_orientation()
    # 再次调用 switch_orientation 方法，将方向设置回默认值
    coll.switch_orientation()
    # 获取更新后的位置数据
    new_positions = coll.get_positions()
    # 断言：属性字典中的方向与事件集合对象的当前方向相同
    assert props['orientation'] == coll.get_orientation()
    # 断言：事件集合对象当前为水平方向
    assert coll.is_horizontal()
    # 使用 np.testing.assert_array_equal 验证属性字典中的位置与新位置数据相等
    np.testing.assert_array_equal(props['positions'], new_positions)
    # 检查线段，验证位置、线段长度、线段偏移、方向与属性字典中的一致性
    check_segments(coll,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    # 设置子图标题为 'EventCollection: switch_orientation 2x'
    splt.set_title('EventCollection: switch_orientation 2x')


@image_comparison(['EventCollection_plot__set_orientation'])
def test__EventCollection__set_orientation():
    # 生成 EventCollection 绘图所需的数据：子图对象 splt，事件集合对象 coll，属性字典 props
    splt, coll, props = generate_EventCollection_plot()
    # 新方向设为 'vertical'
    new_orientation = 'vertical'
    # 调用事件集合对象的 set_orientation 方法，设置新方向
    coll.set_orientation(new_orientation)
    # 断言：事件集合对象当前方向与设定的新方向相同
    assert new_orientation == coll.get_orientation()
    # 断言：事件集合对象当前不为水平方向
    assert not coll.is_horizontal()
    # 检查线段，验证位置、线段长度、线段偏移、方向与新方向一致性
    check_segments(coll,
                   props['positions'],
                   props['linelength'],
                   props['lineoffset'],
                   new_orientation)
    # 设置子图标题为 'EventCollection: set_orientation'
    splt.set_title('EventCollection: set_orientation')
    # 设置子图的 y 轴范围为 -1 到 22
    splt.set_ylim(-1, 22)
    # 设置子图的 x 轴范围为 0 到 2
    splt.set_xlim(0, 2)


@image_comparison(['EventCollection_plot__set_linelength'])
def test__EventCollection__set_linelength():
    # 生成 EventCollection 绘图所需的数据：子图对象 splt，事件集合对象 coll，属性字典 props
    splt, coll, props = generate_EventCollection_plot()
    # 新线段长度设为 15
    new_linelength = 15
    # 调用事件集合对象的 set_linelength 方法，设置新线段长度
    coll.set_linelength(new_linelength)
    # 断言：事件集合对象当前线段长度与设定的新线段长度相同
    assert new_linelength == coll.get_linelength()
    # 检查线段，验证位置、新线段长度、线段偏移、方向与属性字典中的一致性
    check_segments(coll,
                   props['positions'],
                   new_linelength,
                   props['lineoffset'],
                   props['orientation'])
    # 设置子图标题为 'EventCollection: set_linelength'
    splt.set_title('EventCollection: set_linelength')
    # 设置子图的 y 轴范围为 -20 到 20
    splt.set_ylim(-20, 20)


@image_comparison(['EventCollection_plot__set_lineoffset'])
def test__EventCollection__set_lineoffset():
    # 生成 EventCollection 绘图所需的数据：子图对象 splt，事件集合对象 coll，属性字典 props
    splt, coll, props = generate_EventCollection_plot()
    # 新线段偏移设为 -5.0
    new_lineoffset = -5.
    # 调用事件集合对象的 set_lineoffset 方法，设置新线段偏移
    coll.set_lineoffset(new_lineoffset)
    # 断言：事件集合对象当前线段偏移与设定的新线段偏移相同
    assert new_lineoffset == coll.get_lineoffset()
    # 检查线段，验证位置、线段长度、新线段偏移、方向与属性字典中的一致性
    check_segments(coll,
                   props['positions'],
                   props['linelength'],
                   new_lineoffset,
                   props['orientation'])
    # 设置子图标题为 'EventCollection: set_lineoffset'
    splt.set_title('EventCollection: set_lineoffset')
    # 设置子图的 y 轴范围为 -6 到 -4
    splt.set_ylim(-6, -4)


@image_comparison([
    'EventCollection_plot__set_linestyle',
    'EventCollection_plot__set_linestyle',
    'EventCollection_plot__set_linewidth',
])
def test__EventCollection__set_prop():
    # 针对多个属性进行测试，每个属性的预期值列在 expected 中
    for prop, value, expected in [
            ('linestyle', 'dashed', [(0, (6.0, 6.0))]),
            ('linestyle', (0, (6., 6.)), [(0, (6.0, 6.0))]),
            ('linewidth', 5, 5),
            # ...
    ]:
        # 调用 generate_EventCollection_plot() 函数生成 subplot、collection 对象以及额外数据
        splt, coll, _ = generate_EventCollection_plot()
        # 设置 collection 对象的属性为指定的 {prop: value} 键值对
        coll.set(**{prop: value})
        # 使用断言检查设置后的属性值是否符合预期
        assert plt.getp(coll, prop) == expected
        # 设置 subplot 的标题为特定格式的字符串，反映了所设置属性的操作
        splt.set_title(f'EventCollection: set_{prop}')
@image_comparison(['EventCollection_plot__set_color'])
# 使用装饰器进行图像比较，比较当前函数生成的图像与指定图像是否相同
def test__EventCollection__set_color():
    # 生成 EventCollection 的图表，并获取其子图、集合和空变量
    splt, coll, _ = generate_EventCollection_plot()
    # 创建新的颜色数组，用于设置集合的颜色
    new_color = np.array([0, 1, 1, 1])
    # 设置集合的颜色为新颜色
    coll.set_color(new_color)
    # 断言集合和其颜色数组中的每个颜色是否与新颜色相等
    for color in [coll.get_color(), *coll.get_colors()]:
        np.testing.assert_array_equal(color, new_color)
    # 设置子图的标题为 'EventCollection: set_color'
    splt.set_title('EventCollection: set_color')


def check_segments(coll, positions, linelength, lineoffset, orientation):
    """
    Test helper checking that all values in the segment are correct, given a
    particular set of inputs.
    """
    # 获取集合的线段
    segments = coll.get_segments()
    if (orientation.lower() == 'horizontal'
            or orientation.lower() == 'none' or orientation is None):
        # 如果方向为水平或无方向或为空，则位置在 y 轴
        pos1 = 1
        pos2 = 0
    elif orientation.lower() == 'vertical':
        # 如果方向为垂直，则位置在 x 轴
        pos1 = 0
        pos2 = 1
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    
    # 测试确保每个线段的值都正确
    for i, segment in enumerate(segments):
        assert segment[0, pos1] == lineoffset + linelength / 2
        assert segment[1, pos1] == lineoffset - linelength / 2
        assert segment[0, pos2] == positions[i]
        assert segment[1, pos2] == positions[i]


def test_collection_norm_autoscale():
    # 当数组设置时，norm 应自动缩放，而不是延迟到绘制时
    lines = np.arange(24).reshape((4, 3, 2))
    coll = mcollections.LineCollection(lines, array=np.arange(4))
    # 断言 coll.norm(2) 的值为 2 / 3
    assert coll.norm(2) == 2 / 3
    # 设置新的数组不应更新已缩放的限制
    coll.set_array(np.arange(4) + 5)
    # 断言 coll.norm(2) 的值为 2 / 3
    assert coll.norm(2) == 2 / 3


def test_null_collection_datalim():
    # 空集合应返回空数据限制框
    col = mcollections.PathCollection([])
    col_data_lim = col.get_datalim(mtransforms.IdentityTransform())
    # 断言数组是否相等
    assert_array_equal(col_data_lim.get_points(),
                       mtransforms.Bbox.null().get_points())


def test_no_offsets_datalim():
    # 无偏移且非 transData 变换的集合应返回空边界框
    ax = plt.axes()
    coll = mcollections.PathCollection([mpath.Path([(0, 0), (1, 0)])])
    ax.add_collection(coll)
    coll_data_lim = coll.get_datalim(mtransforms.IdentityTransform())
    # 断言数组是否相等
    assert_array_equal(coll_data_lim.get_points(),
                       mtransforms.Bbox.null().get_points())


def test_add_collection():
    # 测试通过添加空集合来检查数据限制是否保持不变
    # GitHub 问题 #1490，拉取请求 #1497
    plt.figure()
    ax = plt.axes()
    ax.scatter([0, 1], [0, 1])
    bounds = ax.dataLim.bounds
    ax.scatter([], [])
    # 断言数据限制的边界是否与原来相同
    assert ax.dataLim.bounds == bounds


@mpl.style.context('mpl20')
@check_figures_equal(extensions=['png'])
def test_collection_log_datalim(fig_test, fig_ref):
    # 使用对数尺度时，数据限制应尊重最小的 x/y 值
    # 定义 X 轴的数值列表，对数刻度下的坐标点
    x_vals = [4.38462e-6, 5.54929e-6, 7.02332e-6, 8.88889e-6, 1.12500e-5,
              1.42383e-5, 1.80203e-5, 2.28070e-5, 2.88651e-5, 3.65324e-5,
              4.62363e-5, 5.85178e-5, 7.40616e-5, 9.37342e-5, 1.18632e-4]
    # 定义 Y 轴的数值列表，对应每个 X 轴点的数据
    y_vals = [0.0, 0.1, 0.182, 0.332, 0.604, 1.1, 2.0, 3.64, 6.64, 12.1, 22.0,
              39.6, 71.3]
    
    # 根据给定的 X 和 Y 值，创建二维网格
    x, y = np.meshgrid(x_vals, y_vals)
    # 将二维网格展平，得到一维的 X 和 Y 数组
    x = x.flatten()
    y = y.flatten()
    
    # 在新的测试图上创建子图 ax_test
    ax_test = fig_test.subplots()
    # 设置子图 ax_test 的 X 轴为对数刻度
    ax_test.set_xscale('log')
    # 设置子图 ax_test 的 Y 轴为对数刻度
    ax_test.set_yscale('log')
    # 设置子图 ax_test 的边缘为零
    ax_test.margins = 0
    # 在 ax_test 上绘制散点图，使用展平后的 X 和 Y 数组
    ax_test.scatter(x, y)
    
    # 在新的参考图上创建子图 ax_ref
    ax_ref = fig_ref.subplots()
    # 设置子图 ax_ref 的 X 轴为对数刻度
    ax_ref.set_xscale('log')
    # 设置子图 ax_ref 的 Y 轴为对数刻度
    ax_ref.set_yscale('log')
    # 在 ax_ref 上绘制散点图，使用展平后的 X 和 Y 数组，以圆圈标记，无连线
    ax_ref.plot(x, y, marker="o", ls="")
def test_quiver_limits():
    # 创建一个新的图形并获取当前轴对象
    ax = plt.axes()
    # 创建一维数组 x 和 y
    x, y = np.arange(8), np.arange(10)
    # 创建二维数组 u 和 v，用于表示箭头的方向
    u = v = np.linspace(0, 10, 80).reshape(10, 8)
    # 在当前轴上绘制箭头图，并返回箭头对象
    q = plt.quiver(x, y, u, v)
    # 断言箭头对象的数据范围与指定的边界相匹配
    assert q.get_datalim(ax.transData).bounds == (0., 0., 7., 9.)

    # 创建一个新的图形
    plt.figure()
    # 获取当前轴对象
    ax = plt.axes()
    # 创建一维数组 x 和 y
    x = np.linspace(-5, 10, 20)
    y = np.linspace(-2, 4, 10)
    # 使用 meshgrid 函数创建 x 和 y 的网格
    y, x = np.meshgrid(y, x)
    # 创建一个仿射变换对象，并将其与当前轴的数据变换对象相结合
    trans = mtransforms.Affine2D().translate(25, 32) + ax.transData
    # 在当前轴上绘制箭头图，使用指定的仿射变换
    plt.quiver(x, y, np.sin(x), np.cos(y), transform=trans)
    # 断言当前轴的数据范围与指定的边界相匹配
    assert ax.dataLim.bounds == (20.0, 30.0, 15.0, 6.0)


def test_barb_limits():
    # 获取当前轴对象
    ax = plt.axes()
    # 创建一维数组 x 和 y
    x = np.linspace(-5, 10, 20)
    y = np.linspace(-2, 4, 10)
    # 使用 meshgrid 函数创建 x 和 y 的网格
    y, x = np.meshgrid(y, x)
    # 创建一个仿射变换对象，并将其与当前轴的数据变换对象相结合
    trans = mtransforms.Affine2D().translate(25, 32) + ax.transData
    # 在当前轴上绘制 barb 图，使用指定的仿射变换
    plt.barbs(x, y, np.sin(x), np.cos(y), transform=trans)
    # 断言当前轴的数据范围与指定的边界相匹配
    # 计算出的边界大致等于原始数据的边界，因为更新数据范围时考虑了整个路径。
    assert_array_almost_equal(ax.dataLim.bounds, (20, 30, 15, 6),
                              decimal=1)


@image_comparison(['EllipseCollection_test_image.png'], remove_text=True,
                  tol=0.021 if platform.machine() == 'arm64' else 0)
def test_EllipseCollection():
    # 测试基本功能
    fig, ax = plt.subplots()
    # 创建一维数组 x 和 y
    x = np.arange(4)
    y = np.arange(3)
    # 使用 meshgrid 函数创建 X 和 Y 的网格
    X, Y = np.meshgrid(x, y)
    # 将 X 和 Y 合并为一个 (N, 2) 的数组
    XY = np.vstack((X.ravel(), Y.ravel())).T

    # 计算椭圆的宽度、高度和角度
    ww = X / x[-1]
    hh = Y / y[-1]
    aa = np.ones_like(ww) * 20  # 第一个轴与 x 轴逆时针偏转 20 度

    # 创建椭圆集合对象
    ec = mcollections.EllipseCollection(
        ww, hh, aa, units='x', offsets=XY, offset_transform=ax.transData,
        facecolors='none')
    # 将椭圆集合对象添加到轴上
    ax.add_collection(ec)
    # 自动调整视图
    ax.autoscale_view()


def test_EllipseCollection_setter_getter():
    # 测试宽度、高度和角度的设置和获取函数
    rng = np.random.default_rng(0)

    # 设置宽度、高度和角度
    widths = (2, )
    heights = (3, )
    angles = (45, )
    offsets = rng.random((10, 2)) * 10

    fig, ax = plt.subplots()

    # 创建椭圆集合对象
    ec = mcollections.EllipseCollection(
        widths=widths,
        heights=heights,
        angles=angles,
        offsets=offsets,
        units='x',
        offset_transform=ax.transData,
        )

    # 断言椭圆集合对象的内部宽度、高度和角度与预期值几乎相等
    assert_array_almost_equal(ec._widths, np.array(widths).ravel() * 0.5)
    assert_array_almost_equal(ec._heights, np.array(heights).ravel() * 0.5)
    assert_array_almost_equal(ec._angles, np.deg2rad(angles).ravel())

    # 断言通过获取函数获得的宽度、高度和角度与预期值几乎相等
    assert_array_almost_equal(ec.get_widths(), widths)
    assert_array_almost_equal(ec.get_heights(), heights)
    assert_array_almost_equal(ec.get_angles(), angles)

    # 将椭圆集合对象添加到轴上
    ax.add_collection(ec)
    # 设置轴的 x 和 y 范围
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)

    # 随机生成新的宽度、高度和角度
    new_widths = rng.random((10, 2)) * 2
    new_heights = rng.random((10, 2)) * 3
    new_angles = rng.random((10, 2)) * 180

    # 使用设置函数设置新的宽度、高度和角度
    ec.set(widths=new_widths, heights=new_heights, angles=new_angles)

    # 断言通过获取函数获得的宽度与新的宽度几乎相等
    assert_array_almost_equal(ec.get_widths(), new_widths.ravel())
    # 断言：验证两个数组几乎相等，用于检查 ec 对象的高度和角度是否与 new_heights 和 new_angles 的展平版本几乎相等。
    assert_array_almost_equal(ec.get_heights(), new_heights.ravel())
    # 断言：验证两个数组几乎相等，用于检查 ec 对象的角度是否与 new_angles 的展平版本几乎相等。
    assert_array_almost_equal(ec.get_angles(), new_angles.ravel())
@image_comparison(['polycollection_close.png'], remove_text=True, style='mpl20')
# 定义一个测试函数，用于比较生成的图像是否与预期的图像相匹配
def test_polycollection_close():
    # 导入需要的模块和函数，这里导入了Axes3D
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore
    # 设置全局参数，使3D图的轴自动调整边距
    plt.rcParams['axes3d.automargin'] = True

    # 定义一个包含多个四边形顶点的列表
    vertsQuad = [
        [[0., 0.], [0., 1.], [1., 1.], [1., 0.]],
        [[0., 1.], [2., 3.], [2., 2.], [1., 1.]],
        [[2., 2.], [2., 3.], [4., 1.], [3., 1.]],
        [[3., 0.], [3., 1.], [4., 1.], [4., 0.]]]

    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个3D坐标轴到图形上
    ax = fig.add_axes(Axes3D(fig))

    # 定义一组颜色
    colors = ['r', 'g', 'b', 'y', 'k']
    # 定义每个多边形的z位置
    zpos = list(range(5))

    # 创建一个PolyCollection对象，用于存储多个多边形
    poly = mcollections.PolyCollection(
        vertsQuad * len(zpos), linewidth=0.25)
    # 设置多边形的透明度
    poly.set_alpha(0.7)

    # 为每个多边形设置颜色和z位置
    zs = []
    cs = []
    for z, c in zip(zpos, colors):
        zs.extend([z] * len(vertsQuad))
        cs.extend([c] * len(vertsQuad))

    poly.set_color(cs)

    # 将PolyCollection对象添加到3D坐标轴上，并指定z轴的位置
    ax.add_collection3d(poly, zs=zs, zdir='y')

    # 设置3D坐标轴的限制
    ax.set_xlim3d(0, 4)
    ax.set_zlim3d(0, 3)
    ax.set_ylim3d(0, 4)


@image_comparison(['regularpolycollection_rotate.png'], remove_text=True)
# 定义另一个测试函数，用于比较生成的图像是否与预期的图像相匹配
def test_regularpolycollection_rotate():
    # 创建一个网格
    xx, yy = np.mgrid[:10, :10]
    # 将网格坐标转置并展平，得到一组点坐标
    xy_points = np.transpose([xx.flatten(), yy.flatten()])
    # 创建一组旋转角度
    rotations = np.linspace(0, 2*np.pi, len(xy_points))

    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 对于每个点和对应的旋转角度，创建一个RegularPolyCollection对象，并将其添加到坐标轴上
    for xy, alpha in zip(xy_points, rotations):
        col = mcollections.RegularPolyCollection(
            4, sizes=(100,), rotation=alpha,
            offsets=[xy], offset_transform=ax.transData)
        ax.add_collection(col, autolim=True)
    # 自动调整坐标轴视图
    ax.autoscale_view()


@image_comparison(['regularpolycollection_scale.png'], remove_text=True)
# 定义另一个测试函数，用于比较生成的图像是否与预期的图像相匹配
def test_regularpolycollection_scale():
    # 创建一个自定义的RegularPolyCollection子类
    class SquareCollection(mcollections.RegularPolyCollection):
        def __init__(self, **kwargs):
            super().__init__(4, rotation=np.pi/4., **kwargs)

        def get_transform(self):
            """Return transform scaling circle areas to data space."""
            # 获取当前坐标轴对象
            ax = self.axes

            # 计算像素和数据空间之间的比例
            pts2pixels = 72.0 / ax.figure.dpi

            scale_x = pts2pixels * ax.bbox.width / ax.viewLim.width
            scale_y = pts2pixels * ax.bbox.height / ax.viewLim.height
            # 返回一个仿射变换对象，用于将圆的面积缩放到数据空间
            return mtransforms.Affine2D().scale(scale_x, scale_y)

    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()

    xy = [(0, 0)]
    # 定义圆的面积
    circle_areas = [np.pi / 2]
    # 创建SquareCollection对象，并将其添加到坐标轴上
    squares = SquareCollection(
        sizes=circle_areas, offsets=xy, offset_transform=ax.transData)
    ax.add_collection(squares, autolim=True)
    # 设置坐标轴的范围
    ax.axis([-1, 1, -1, 1])


# 定义一个测试函数，用于测试拾取功能
def test_picking():
    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 创建一个包含一个点的散点图，并启用拾取功能
    col = ax.scatter([0], [0], [1000], picker=True)
    # 将图形保存为字节流
    fig.savefig(io.BytesIO(), dpi=fig.dpi)
    # 创建一个模拟的鼠标事件
    mouse_event = SimpleNamespace(x=325, y=240)
    # 检查鼠标事件是否在散点图上，并返回是否找到和索引
    found, indices = col.contains(mouse_event)
    # 断言是否找到指定的点
    assert found
    assert_array_equal(indices['ind'], [0])


# 定义一个测试函数，用于测试QuadMesh对象的contains方法
def test_quadmesh_contains():
    # 创建一个一维数组
    x = np.arange(4)
    # 创建二维数组 X，每个元素为输入向量 x 的乘积
    X = x[:, None] * x[None, :]

    # 创建一个新的图形窗口和坐标轴
    fig, ax = plt.subplots()

    # 在坐标轴上绘制一个基于 X 的伪彩色网格
    mesh = ax.pcolormesh(X)

    # 绘制图形但不进行渲染，用于预览
    fig.draw_without_rendering()

    # 设置待查询的鼠标事件的坐标
    xdata, ydata = 0.5, 0.5

    # 将鼠标事件的逻辑坐标转换为图形的坐标系
    x, y = mesh.get_transform().transform((xdata, ydata))

    # 创建一个命名空间来存储鼠标事件的信息
    mouse_event = SimpleNamespace(xdata=xdata, ydata=ydata, x=x, y=y)

    # 检查鼠标事件是否在网格内，返回布尔值和包含索引的字典
    found, indices = mesh.contains(mouse_event)

    # 断言鼠标事件确实在网格内，并且索引为 [0]
    assert found
    assert_array_equal(indices['ind'], [0])

    # 设置另一个待查询的鼠标事件的坐标
    xdata, ydata = 1.5, 1.5

    # 将第二个鼠标事件的逻辑坐标转换为图形的坐标系
    x, y = mesh.get_transform().transform((xdata, ydata))

    # 更新命名空间中的鼠标事件信息
    mouse_event = SimpleNamespace(xdata=xdata, ydata=ydata, x=x, y=y)

    # 再次检查鼠标事件是否在网格内，返回布尔值和包含索引的字典
    found, indices = mesh.contains(mouse_event)

    # 断言第二个鼠标事件确实在网格内，并且索引为 [5]
    assert found
    assert_array_equal(indices['ind'], [5])
def test_quadmesh_contains_concave():
    # Test a concave polygon, V-like shape
    # 定义一个包含凹多边形（V形状）的测试
    x = [[0, -1], [1, 0]]  # 定义 x 坐标
    y = [[0, 1], [1, -1]]  # 定义 y 坐标
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    mesh = ax.pcolormesh(x, y, [[0]])  # 绘制一个彩色网格，使用指定的 x, y 和颜色数据
    fig.draw_without_rendering()  # 在不渲染的情况下绘制图形
    # xdata, ydata, expected
    points = [(-0.5, 0.25, True),  # 左翼
              (0, 0.25, False),   # 在两翼之间
              (0.5, 0.25, True),  # 右翼
              (0, -0.25, True),  # 主体部分
              ]
    for point in points:
        xdata, ydata, expected = point
        x, y = mesh.get_transform().transform((xdata, ydata))  # 转换坐标点到数据坐标系
        mouse_event = SimpleNamespace(xdata=xdata, ydata=ydata, x=x, y=y)  # 创建一个命名空间对象表示鼠标事件
        found, indices = mesh.contains(mouse_event)  # 判断鼠标事件是否在网格内
        assert found is expected  # 断言判断结果是否符合预期


def test_quadmesh_cursor_data():
    x = np.arange(4)  # 创建一个包含 0 到 3 的数组
    X = x[:, None] * x[None, :]  # 创建一个 4x4 的二维数组

    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    mesh = ax.pcolormesh(X)  # 绘制一个基于 X 数据的彩色网格
    # Empty array data
    mesh._A = None  # 设置网格的数据为空数组
    fig.draw_without_rendering()  # 在不渲染的情况下绘制图形
    xdata, ydata = 0.5, 0.5  # 定义鼠标事件的 x 和 y 坐标
    x, y = mesh.get_transform().transform((xdata, ydata))  # 转换坐标点到数据坐标系
    mouse_event = SimpleNamespace(xdata=xdata, ydata=ydata, x=x, y=y)  # 创建一个命名空间对象表示鼠标事件
    # Empty collection should return None
    assert mesh.get_cursor_data(mouse_event) is None  # 断言空集合应该返回 None

    # Now test adding the array data, to make sure we do get a value
    mesh.set_array(np.ones(X.shape))  # 设置网格的数据为全为 1 的数组
    assert_array_equal(mesh.get_cursor_data(mouse_event), [1])  # 断言获取到的鼠标数据是否为 [1]


def test_quadmesh_cursor_data_multiple_points():
    x = [1, 2, 1, 2]  # 定义 x 坐标数组
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    mesh = ax.pcolormesh(x, x, np.ones((3, 3)))  # 绘制一个基于 x, y 和数据的彩色网格
    fig.draw_without_rendering()  # 在不渲染的情况下绘制图形
    xdata, ydata = 1.5, 1.5  # 定义鼠标事件的 x 和 y 坐标
    x, y = mesh.get_transform().transform((xdata, ydata))  # 转换坐标点到数据坐标系
    mouse_event = SimpleNamespace(xdata=xdata, ydata=ydata, x=x, y=y)  # 创建一个命名空间对象表示鼠标事件
    # All quads are covering the same square
    assert_array_equal(mesh.get_cursor_data(mouse_event), np.ones(9))  # 断言获取到的鼠标数据是否全为 1


def test_linestyle_single_dashes():
    plt.scatter([0, 1, 2], [0, 1, 2], linestyle=(0., [2., 2.]))  # 绘制带有单虚线样式的散点图
    plt.draw()  # 绘制图形


@image_comparison(['size_in_xy.png'], remove_text=True)
def test_size_in_xy():
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴

    widths, heights, angles = (10, 10), 10, 0  # 定义椭圆的宽度、高度和角度
    widths = 10, 10  # 更新宽度值
    coords = [(10, 10), (15, 15)]  # 定义椭圆的偏移坐标
    e = mcollections.EllipseCollection(
        widths, heights, angles, units='xy',
        offsets=coords, offset_transform=ax.transData)  # 创建一个基于坐标轴数据变换的椭圆集合

    ax.add_collection(e)  # 将椭圆集合添加到坐标轴

    ax.set_xlim(0, 30)  # 设置 x 轴的显示范围
    ax.set_ylim(0, 30)  # 设置 y 轴的显示范围


def test_pandas_indexing(pd):
    # Should not fail break when faced with a
    # non-zero indexed series
    index = [11, 12, 13]  # 定义索引数组
    ec = fc = pd.Series(['red', 'blue', 'green'], index=index)  # 创建包含颜色数据的 Series 对象
    lw = pd.Series([1, 2, 3], index=index)  # 创建包含线宽数据的 Series 对象
    ls = pd.Series(['solid', 'dashed', 'dashdot'], index=index)  # 创建包含线样式数据的 Series 对象
    aa = pd.Series([True, False, True], index=index)  # 创建包含抗锯齿属性数据的 Series 对象

    Collection(edgecolors=ec)  # 使用边界颜色创建集合
    Collection(facecolors=fc)  # 使用填充颜色创建集合
    Collection(linewidths=lw)  # 使用线宽创建集合
    Collection(linestyles=ls)  # 使用线样式创建集合
    Collection(antialiaseds=aa)  # 使用抗锯齿属性创建集合


@mpl.style.context('default')
def test_lslw_bcast():
    # Test the broadcast of linestyle and linewidth
    # across collections
    # 测试在集合之间广播线样式和线宽
    pass  # 仅占位，没有实际代码
    # 创建一个空的路径集合对象
    col = mcollections.PathCollection([])
    
    # 设置路径集合对象的线型样式为['-', '-']
    col.set_linestyles(['-', '-'])
    
    # 设置路径集合对象的线宽为[1, 2, 3]
    col.set_linewidths([1, 2, 3])
    
    # 断言检查路径集合对象当前的线型样式是否为[(0, None)] * 6，即每个元素为元组 (0, None)，共6个元素
    assert col.get_linestyles() == [(0, None)] * 6
    
    # 断言检查路径集合对象当前的线宽是否为[1, 2, 3] * 2，即[1, 2, 3, 1, 2, 3]
    assert col.get_linewidths() == [1, 2, 3] * 2
    
    # 设置路径集合对象的线型样式为['-', '-', '-']
    col.set_linestyles(['-', '-', '-'])
    
    # 断言检查路径集合对象当前的线型样式是否为[(0, None)] * 3，即每个元素为元组 (0, None)，共3个元素
    assert col.get_linestyles() == [(0, None)] * 3
    
    # 断言检查路径集合对象当前的线宽是否为[1, 2, 3] 的所有元素都是 True
    assert (col.get_linewidths() == [1, 2, 3]).all()
# 定义一个测试函数，用于测试设置错误的线条样式是否会引发 ValueError 异常，且异常信息需包含字符串 "Do not know how to convert 'fuzzy'"
def test_set_wrong_linestyle():
    # 创建一个空的 Collection 对象
    c = Collection()
    # 使用 pytest 的上下文管理器检查是否会引发 ValueError 异常，并验证异常信息是否包含 "Do not know how to convert 'fuzzy'"
    with pytest.raises(ValueError, match="Do not know how to convert 'fuzzy'"):
        # 调用被测试的方法，尝试设置线条样式为 'fuzzy'
        c.set_linestyle('fuzzy')


# 使用默认风格上下文，定义测试线帽样式的函数
@mpl.style.context('default')
def test_capstyle():
    # 创建一个空的 PathCollection 对象
    col = mcollections.PathCollection([])
    # 断言当前线帽样式为 None
    assert col.get_capstyle() is None
    # 创建带有指定线帽样式 'round' 的 PathCollection 对象
    col = mcollections.PathCollection([], capstyle='round')
    # 断言线帽样式已设置为 'round'
    assert col.get_capstyle() == 'round'
    # 设置线帽样式为 'butt'
    col.set_capstyle('butt')
    # 断言线帽样式已设置为 'butt'
    assert col.get_capstyle() == 'butt'


# 使用默认风格上下文，定义测试连接样式的函数
@mpl.style.context('default')
def test_joinstyle():
    # 创建一个空的 PathCollection 对象
    col = mcollections.PathCollection([])
    # 断言当前连接样式为 None
    assert col.get_joinstyle() is None
    # 创建带有指定连接样式 'round' 的 PathCollection 对象
    col = mcollections.PathCollection([], joinstyle='round')
    # 断言连接样式已设置为 'round'
    assert col.get_joinstyle() == 'round'
    # 设置连接样式为 'miter'
    col.set_joinstyle('miter')
    # 断言连接样式已设置为 'miter'
    assert col.get_joinstyle() == 'miter'


# 图像比较测试，对线帽样式和连接样式进行设置，并生成图片比较结果
@image_comparison(['cap_and_joinstyle.png'])
def test_cap_and_joinstyle_image():
    # 创建一个图形和轴对象
    fig, ax = plt.subplots()
    # 设置坐标轴范围
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 2.5])

    # 定义线段的 x 和 y 数据
    x = np.array([0.0, 1.0, 0.5])
    ys = np.array([[0.0], [0.5], [1.0]]) + np.array([[0.0, 0.0, 1.0]])

    # 初始化线段数据
    segs = np.zeros((3, 3, 2))
    segs[:, :, 0] = x
    segs[:, :, 1] = ys

    # 创建 LineCollection 对象，设置线宽和线帽样式、连接样式
    line_segments = LineCollection(segs, linewidth=[10, 15, 20])
    line_segments.set_capstyle("round")
    line_segments.set_joinstyle("miter")

    # 将 LineCollection 对象添加到轴上
    ax.add_collection(line_segments)
    # 设置图表标题
    ax.set_title('Line collection with customized caps and joinstyle')


# 图像比较测试，测试散点图设置透明度后的效果，生成图片比较结果
@image_comparison(['scatter_post_alpha.png'],
                  remove_text=True, style='default')
def test_scatter_post_alpha():
    # 创建一个图形和轴对象
    fig, ax = plt.subplots()
    # 创建散点图，设置颜色和透明度
    sc = ax.scatter(range(5), range(5), c=range(5))
    sc.set_alpha(.1)


# 测试散点图使用数组设置透明度的效果
def test_scatter_alpha_array():
    # 创建数据数组
    x = np.arange(5)
    alpha = x / 5

    # 第一个子图：使用颜色映射
    fig, (ax0, ax1) = plt.subplots(2)
    sc0 = ax0.scatter(x, x, c=x, alpha=alpha)
    sc1 = ax1.scatter(x, x, c=x)
    # 设置散点图透明度
    sc1.set_alpha(alpha)
    plt.draw()

    # 断言透明度设置正确
    assert_array_equal(sc0.get_facecolors()[:, -1], alpha)
    assert_array_equal(sc1.get_facecolors()[:, -1], alpha)

    # 第二个子图：不使用颜色映射
    fig, (ax0, ax1) = plt.subplots(2)
    sc0 = ax0.scatter(x, x, color=['r', 'g', 'b', 'c', 'm'], alpha=alpha)
    sc1 = ax1.scatter(x, x, color='r', alpha=alpha)
    plt.draw()

    # 断言透明度设置正确
    assert_array_equal(sc0.get_facecolors()[:, -1], alpha)
    assert_array_equal(sc1.get_facecolors()[:, -1], alpha)

    # 第三个子图：不使用颜色映射，并在设置透明度后
    fig, (ax0, ax1) = plt.subplots(2)
    sc0 = ax0.scatter(x, x, color=['r', 'g', 'b', 'c', 'm'])
    sc0.set_alpha(alpha)
    sc1 = ax1.scatter(x, x, color='r')
    sc1.set_alpha(alpha)
    plt.draw()

    # 断言透明度设置正确
    assert_array_equal(sc0.get_facecolors()[:, -1], alpha)
    assert_array_equal(sc1.get_facecolors()[:, -1], alpha)


# 测试 PathCollection 类的图例元素生成
def test_pathcollection_legend_elements():
    # 设置随机种子，保证结果可重复
    np.random.seed(19680801)
    # 生成随机数据
    x, y = np.random.rand(2, 10)
    y = np.random.rand(10)
    c = np.random.randint(0, 5, size=10)
    s = np.random.randint(10, 300, size=10)

    # 创建一个图形和轴对象
    fig, ax = plt.subplots()
    # 创建散点图，并返回散点对象
    sc = ax.scatter(x, y, c=c, s=s, cmap="jet", marker="o", linewidths=0)

    # 从散点对象中获取用于图例的元素
    h, l = sc.legend_elements(fmt="{x:g}")
    # 确保图例元素的数量为5
    assert len(h) == 5
    # 确保图例的标签列表正确
    assert l == ["0", "1", "2", "3", "4"]
    # 获取散点的颜色列表
    colors = np.array([line.get_color() for line in h])
    # 生成预期的颜色列表
    colors2 = sc.cmap(np.arange(5)/4)
    # 确保两个颜色列表相等
    assert_array_equal(colors, colors2)
    # 在图上添加第一个图例
    l1 = ax.legend(h, l, loc=1)

    # 从散点对象中获取更多的图例元素，数量为9
    h2, lab2 = sc.legend_elements(num=9)
    # 确保图例元素的数量为9
    assert len(h2) == 9
    # 在图上添加第二个图例
    l2 = ax.legend(h2, lab2, loc=2)

    # 从散点对象中获取基于尺寸的图例元素，设置透明度和颜色
    h, l = sc.legend_elements(prop="sizes", alpha=0.5, color="red")
    # 确保所有图例元素的透明度为0.5
    assert all(line.get_alpha() == 0.5 for line in h)
    # 确保所有图例元素的标记颜色为红色
    assert all(line.get_markerfacecolor() == "red" for line in h)
    # 在图上添加第三个图例
    l3 = ax.legend(h, l, loc=4)

    # 从散点对象中获取基于尺寸的图例元素，数量为4，并应用自定义格式和函数
    h, l = sc.legend_elements(prop="sizes", num=4, fmt="{x:.2f}",
                              func=lambda x: 2*x)
    # 获取实际的标记大小列表
    actsizes = [line.get_markersize() for line in h]
    # 计算应显示的标签大小列表
    labeledsizes = np.sqrt(np.array(l, float) / 2)
    # 确保实际标记大小与预期标签大小接近
    assert_array_almost_equal(actsizes, labeledsizes)
    # 在图上添加第四个图例
    l4 = ax.legend(h, l, loc=3)

    # 根据最大定位器创建自定义刻度位置
    loc = mpl.ticker.MaxNLocator(nbins=9, min_n_ticks=9-1,
                                 steps=[1, 2, 2.5, 3, 5, 6, 8, 10])
    # 从散点对象中获取图例元素，数量由定位器定义
    h5, lab5 = sc.legend_elements(num=loc)
    # 确保第二个和第五个图例元素的数量相等
    assert len(h2) == len(h5)

    # 指定特定的级别创建图例元素，基于尺寸，并使用自定义格式
    levels = [-1, 0, 55.4, 260]
    h6, lab6 = sc.legend_elements(num=levels, prop="sizes", fmt="{x:g}")
    # 确保图例标签的浮点数值与预期级别匹配
    assert [float(l) for l in lab6] == levels[2:]

    # 将所有图例添加到图形上
    for l in [l1, l2, l3, l4]:
        ax.add_artist(l)

    # 刷新画布，以显示更新后的图例
    fig.canvas.draw()
def test_EventCollection_nosort():
    # 检查 EventCollection 不会直接修改输入数组
    arr = np.array([3, 2, 1, 10])
    coll = EventCollection(arr)
    np.testing.assert_array_equal(arr, np.array([3, 2, 1, 10]))


def test_collection_set_verts_array():
    # 创建包含顶点数组的 PolyCollection 对象
    verts = np.arange(80, dtype=np.double).reshape(10, 4, 2)
    col_arr = PolyCollection(verts)
    col_list = PolyCollection(list(verts))
    # 断言两个 PolyCollection 对象的路径数相等
    assert len(col_arr._paths) == len(col_list._paths)
    # 比较两个 PolyCollection 对象中每个路径的顶点和代码数组是否相等
    for ap, lp in zip(col_arr._paths, col_list._paths):
        assert np.array_equal(ap._vertices, lp._vertices)
        assert np.array_equal(ap._codes, lp._codes)

    # 创建包含元组顶点数组的 PolyCollection 对象
    verts_tuple = np.empty(10, dtype=object)
    verts_tuple[:] = [tuple(tuple(y) for y in x) for x in verts]
    col_arr_tuple = PolyCollection(verts_tuple)
    # 断言原始 verts 和 verts_tuple 创建的 PolyCollection 对象路径数相等
    assert len(col_arr._paths) == len(col_arr_tuple._paths)
    # 比较两个 PolyCollection 对象中每个路径的顶点和代码数组是否相等
    for ap, atp in zip(col_arr._paths, col_arr_tuple._paths):
        assert np.array_equal(ap._vertices, atp._vertices)
        assert np.array_equal(ap._codes, atp._codes)


def test_collection_set_array():
    vals = [*range(10)]

    # 使用列表测试 set_array 方法
    c = Collection()
    c.set_array(vals)

    # 使用错误的数据类型测试 set_array 方法
    with pytest.raises(TypeError, match="^Image data of dtype"):
        c.set_array("wrong_input")

    # 检查数组参数是否被复制
    vals[5] = 45
    assert np.not_equal(vals, c.get_array()).any()


def test_blended_collection_autolim():
    a = [1, 2, 4]
    height = .2

    # 创建线段的坐标对数组
    xy_pairs = np.column_stack([np.repeat(a, 2), np.tile([0, height], len(a))])
    line_segs = xy_pairs.reshape([len(a), 2, 2])

    f, ax = plt.subplots()
    # 创建混合变换对象
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.add_collection(LineCollection(line_segs, transform=trans))
    ax.autoscale_view(scalex=True, scaley=False)
    np.testing.assert_allclose(ax.get_xlim(), [1., 4.])


def test_singleton_autolim():
    fig, ax = plt.subplots()
    # 绘制散点图在 (0, 0) 处
    ax.scatter(0, 0)
    np.testing.assert_allclose(ax.get_ylim(), [-0.06, 0.06])
    np.testing.assert_allclose(ax.get_xlim(), [-0.06, 0.06])


@pytest.mark.parametrize("transform, expected", [
    ("transData", (-0.5, 3.5)),
    ("transAxes", (2.8, 3.2)),
])
def test_autolim_with_zeros(transform, expected):
    # 1) 测试在数据坐标 (0, 0) 处绘制散点图，验证是否自动缩放
    # 2) 测试指定 transAxes 变换是否不会影响自动缩放
    fig, ax = plt.subplots()
    ax.scatter(0, 0, transform=getattr(ax, transform))
    ax.scatter(3, 3)
    np.testing.assert_allclose(ax.get_ylim(), expected)
    np.testing.assert_allclose(ax.get_xlim(), expected)


def test_quadmesh_set_array_validation(pcfunc):
    x = np.arange(11)
    y = np.arange(8)
    z = np.random.random((7, 10))
    fig, ax = plt.subplots()
    # 根据 pcfunc 参数创建 QuadMesh 对象并添加到坐标系中
    coll = getattr(ax, pcfunc)(x, y, z)
    # 使用 pytest 检查是否引发值错误，并验证错误消息是否匹配预期格式
    with pytest.raises(ValueError, match=re.escape(
            "For X (11) and Y (8) with flat shading, A should have shape "
            "(7, 10, 3) or (7, 10, 4) or (7, 10) or (70,), not (10, 7)")):
        # 调用 coll 对象的 set_array 方法，设置数组 z 的形状为 (10, 7)
        coll.set_array(z.reshape(10, 7))

    # 创建一个 6x9 的数组 z，并验证是否引发预期的值错误异常
    z = np.arange(54).reshape((6, 9))
    with pytest.raises(ValueError, match=re.escape(
            "For X (11) and Y (8) with flat shading, A should have shape "
            "(7, 10, 3) or (7, 10, 4) or (7, 10) or (70,), not (6, 9)")):
        # 调用 coll 对象的 set_array 方法，设置数组 z
        coll.set_array(z)
    with pytest.raises(ValueError, match=re.escape(
            "For X (11) and Y (8) with flat shading, A should have shape "
            "(7, 10, 3) or (7, 10, 4) or (7, 10) or (70,), not (54,)")):
        # 调用 coll 对象的 set_array 方法，设置数组 z 的展平版本
        coll.set_array(z.ravel())

    # 创建一个 9x6x3 的数组 z，并验证是否引发预期的值错误异常
    # RGB(A) 测试
    z = np.ones((9, 6, 3))  # RGB with wrong X/Y dims
    with pytest.raises(ValueError, match=re.escape(
            "For X (11) and Y (8) with flat shading, A should have shape "
            "(7, 10, 3) or (7, 10, 4) or (7, 10) or (70,), not (9, 6, 3)")):
        # 调用 coll 对象的 set_array 方法，设置数组 z
        coll.set_array(z)

    # 创建一个 9x6x4 的数组 z，并验证是否引发预期的值错误异常
    z = np.ones((9, 6, 4))  # RGBA with wrong X/Y dims
    with pytest.raises(ValueError, match=re.escape(
            "For X (11) and Y (8) with flat shading, A should have shape "
            "(7, 10, 3) or (7, 10, 4) or (7, 10) or (70,), not (9, 6, 4)")):
        # 调用 coll 对象的 set_array 方法，设置数组 z
        coll.set_array(z)

    # 创建一个 7x10x2 的数组 z，并验证是否引发预期的值错误异常
    z = np.ones((7, 10, 2))  # Right X/Y dims, bad 3rd dim
    with pytest.raises(ValueError, match=re.escape(
            "For X (11) and Y (8) with flat shading, A should have shape "
            "(7, 10, 3) or (7, 10, 4) or (7, 10) or (70,), not (7, 10, 2)")):
        # 调用 coll 对象的 set_array 方法，设置数组 z
        coll.set_array(z)

    # 创建 x、y 和 z，并绘制基于 x、y、z 的伪彩色图
    x = np.arange(10)
    y = np.arange(7)
    z = np.random.random((7, 10))
    fig, ax = plt.subplots()
    coll = ax.pcolormesh(x, y, z, shading='gouraud')
def test_polyquadmesh_masked_vertices_array():
    xx, yy = np.meshgrid([0, 1, 2], [0, 1, 2, 3])
    # 创建一个二维网格，xx和yy表示X轴和Y轴上的坐标点
    zz = (xx*yy)[:-1, :-1]
    # 计算网格点的数值，生成一个与xx和yy大小相同的数组zz
    quadmesh = plt.pcolormesh(xx, yy, zz)
    # 使用pcolormesh创建一个四边形网格对象
    quadmesh.update_scalarmappable()
    # 更新网格对象的标量映射属性
    quadmesh_fc = quadmesh.get_facecolor()[1:, :]
    # 获取四边形网格的面颜色，切片操作取第一行之后的所有行

    # 在X轴上屏蔽原点顶点
    xx = np.ma.masked_where((xx == 0) & (yy == 0), xx)
    # 使用np.ma.masked_where()函数，根据条件屏蔽数组中的元素
    polymesh = plt.pcolor(xx, yy, zz)
    # 使用pcolor创建一个伪彩色图对象
    polymesh.update_scalarmappable()
    # 更新伪彩色图对象的标量映射属性
    assert len(polymesh.get_paths()) == 5
    # 断言确保伪彩色图对象的路径数为5，验证屏蔽效果

    # Poly版本的面颜色应与quadmesh的末尾相同
    assert_array_equal(quadmesh_fc, polymesh.get_facecolor())

    # 在Y轴上屏蔽原点顶点
    yy = np.ma.masked_where((xx == 0) & (yy == 0), yy)
    # 使用np.ma.masked_where()函数，根据条件屏蔽数组中的元素
    polymesh = plt.pcolor(xx, yy, zz)
    # 使用pcolor创建一个伪彩色图对象
    polymesh.update_scalarmappable()
    # 更新伪彩色图对象的标量映射属性
    assert len(polymesh.get_paths()) == 5
    # 断言确保伪彩色图对象的路径数为5，验证屏蔽效果

    # 屏蔽原始网格数据中的原始单元格
    zz = np.ma.masked_where((xx[:-1, :-1] == 0) & (yy[:-1, :-1] == 0), zz)
    # 使用np.ma.masked_where()函数，根据条件屏蔽数组中的元素
    polymesh = plt.pcolor(zz)
    # 使用pcolor创建一个伪彩色图对象
    polymesh.update_scalarmappable()
    # 更新伪彩色图对象的标量映射属性
    assert len(polymesh.get_paths()) == 5
    # 断言确保伪彩色图对象的路径数为5，验证屏蔽效果

    # 使用压缩的1D值设置数组已经过时
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match="Setting a PolyQuadMesh"):
        polymesh.set_array(np.ones(5))
    # 断言确保触发MatplotlibDeprecationWarning警告，说明设置PolyQuadMesh已被弃用

    # 我们也应该能够使用新的掩码调用set_array并获取更新的多边形
    # 移除掩码，应该将所有多边形添加回去
    zz = np.arange(6).reshape((3, 2))
    polymesh.set_array(zz)
    # 使用新的数组设置伪彩色图对象的数组
    polymesh.update_scalarmappable()
    # 更新伪彩色图对象的标量映射属性
    assert len(polymesh.get_paths()) == 6
    # 断言确保伪彩色图对象的路径数为6，验证添加效果

    # 添加掩码应该会移除多边形
    zz = np.ma.masked_less(zz, 2)
    # 使用np.ma.masked_less()函数，根据条件屏蔽数组中小于2的元素
    polymesh.set_array(zz)
    # 使用新的数组设置伪彩色图对象的数组
    polymesh.update_scalarmappable()
    # 更新伪彩色图对象的标量映射属性
    assert len(polymesh.get_paths()) == 4
    # 断言确保伪彩色图对象的路径数为4，验证移除效果
    # 创建一个颜色网格对象，使用 x, y 坐标和 z 的形状来填充，采用 gouraud 渲染方式
    coll = ax.pcolormesh(x, y, np.ones(z.shape), shading='gouraud')
    # 测试收集器是否能够通过一个二维数组进行更新

    # 设置颜色网格对象的数据为 z
    coll.set_array(z)
    # 更新图形画布
    fig.canvas.draw()
    # 断言：验证颜色网格对象的数据是否与 z 相等
    assert np.array_equal(coll.get_array(), z)

    # 检查预先展开的数组是否也能正常工作

    # 设置颜色网格对象的数据为一个全为 1 的 1 维数组（长度为 16）
    coll.set_array(np.ones(16))
    # 再次更新图形画布
    fig.canvas.draw()
    # 断言：验证颜色网格对象的数据是否与全为 1 的数组相等
    assert np.array_equal(coll.get_array(), np.ones(16))
def test_quadmesh_vmin_vmax(pcfunc):
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 选择色彩映射
    cmap = mpl.colormaps['plasma']
    # 创建归一化对象，并设置其范围为0到1
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # 使用给定的pcfunc绘制一个quadmesh对象，并应用颜色映射和归一化
    coll = getattr(ax, pcfunc)([[1]], cmap=cmap, norm=norm)
    # 绘制图形
    fig.canvas.draw()
    # 断言quadmesh的颜色与归一化后的值相等
    assert np.array_equal(coll.get_facecolors()[0, :], cmap(norm(1)))

    # 修改归一化的vmin和vmax，使得颜色从色彩映射的底部开始
    norm.vmin, norm.vmax = 1, 2
    # 再次绘制图形
    fig.canvas.draw()
    # 断言quadmesh的颜色与新的归一化值相等
    assert np.array_equal(coll.get_facecolors()[0, :], cmap(norm(1)))


def test_quadmesh_alpha_array(pcfunc):
    # 创建一些数据
    x = np.arange(4)
    y = np.arange(4)
    z = np.arange(9).reshape((3, 3))
    alpha = z / z.max()  # 根据数据z计算alpha值
    alpha_flat = alpha.ravel()  # 将alpha展平为一维数组
    # 在两个子图中创建quadmesh对象，一个使用2D alpha值，另一个不使用
    fig, (ax0, ax1) = plt.subplots(2)
    coll1 = getattr(ax0, pcfunc)(x, y, z, alpha=alpha)
    coll2 = getattr(ax0, pcfunc)(x, y, z)
    coll2.set_alpha(alpha)  # 设置第二个quadmesh的alpha值
    plt.draw()
    # 断言quadmesh的颜色数组的alpha通道与预期的alpha_flat数组相等
    assert_array_equal(coll1.get_facecolors()[:, -1], alpha_flat)
    assert_array_equal(coll2.get_facecolors()[:, -1], alpha_flat)
    # 再次创建两个子图中的quadmesh对象，一个使用1D alpha值，另一个不使用
    fig, (ax0, ax1) = plt.subplots(2)
    coll1 = getattr(ax0, pcfunc)(x, y, z, alpha=alpha)
    coll2 = getattr(ax1, pcfunc)(x, y, z)
    coll2.set_alpha(alpha)  # 设置第二个quadmesh的alpha值
    plt.draw()
    # 断言quadmesh的颜色数组的alpha通道与预期的alpha_flat数组相等
    assert_array_equal(coll1.get_facecolors()[:, -1], alpha_flat)
    assert_array_equal(coll2.get_facecolors()[:, -1], alpha_flat)


def test_alpha_validation(pcfunc):
    # 大部分相关的测试在test_artist和test_colors中完成。
    fig, ax = plt.subplots()
    pc = getattr(ax, pcfunc)(np.arange(12).reshape((3, 4)))
    # 使用pytest断言抛出值错误，并匹配特定的错误信息
    with pytest.raises(ValueError, match="^Data array shape"):
        pc.set_alpha([0.5, 0.6])
        pc.update_scalarmappable()


def test_legend_inverse_size_label_relationship():
    """
    确保图例标记在标签和大小成反比时能够适当缩放。
    这里标签 = 5 / 大小
    """
    np.random.seed(19680801)
    X = np.random.random(50)
    Y = np.random.random(50)
    C = 1 - np.random.random(50)
    S = 5 / C  # 计算大小与标签的反比关系

    legend_sizes = [0.2, 0.4, 0.6, 0.8]
    fig, ax = plt.subplots()
    sc = ax.scatter(X, Y, s=S)  # 创建散点图，并使用计算的大小S
    handles, labels = sc.legend_elements(
      prop='sizes', num=legend_sizes, func=lambda s: 5 / s
    )

    # 将标记大小比例转换为 's' 比例
    handle_sizes = [x.get_markersize() for x in handles]
    handle_sizes = [5 / x**2 for x in handle_sizes]

    # 断言标记大小与预期的legend_sizes数组几乎相等
    assert_array_almost_equal(handle_sizes, legend_sizes, decimal=1)


@mpl.style.context('default')
def test_color_logic(pcfunc):
    pcfunc = getattr(plt, pcfunc)
    z = np.arange(12).reshape(3, 4)
    # 显式设置边缘颜色为红色。
    pc = pcfunc(z, edgecolors='red', facecolors='none')
    pc.update_scalarmappable()  # 在绘制中调用这个方法。
    # 这里定义两个参考的 "颜色" 用于多次使用。
    face_default = mcolors.to_rgba_array(pc._get_default_facecolor())
    mapped = pc.get_cmap()(pc.norm(z.ravel()))
    # GitHub issue #1302:
    # 断言检查边缘颜色是否与 'red' 相同
    assert mcolors.same_color(pc.get_edgecolor(), 'red')

    # Check setting attributes after initialization:
    # 创建对象 pcfunc，并设置参数 z
    pc = pcfunc(z)
    # 设置面颜色为 'none'
    pc.set_facecolor('none')
    # 设置边缘颜色为 'red'
    pc.set_edgecolor('red')
    # 更新标量映射
    pc.update_scalarmappable()
    # 断言检查面颜色是否与 'none' 相同
    assert mcolors.same_color(pc.get_facecolor(), 'none')
    # 断言检查边缘颜色是否为 [[1, 0, 0, 1]]
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])

    # 设置 alpha 值为 0.5
    pc.set_alpha(0.5)
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查边缘颜色是否为 [[1, 0, 0, 0.5]]
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 0.5]])

    # 恢复默认 alpha 值
    pc.set_alpha(None)
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查边缘颜色是否为 [[1, 0, 0, 1]]
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])

    # 将边缘颜色重置为默认值
    pc.set_edgecolor(None)
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查边缘颜色是否与 mapped 相等
    assert np.array_equal(pc.get_edgecolor(), mapped)

    # 恢复默认面颜色
    pc.set_facecolor(None)
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查面颜色是否与 mapped 相等
    assert np.array_equal(pc.get_facecolor(), mapped)
    # 断言检查边缘颜色是否为 'none'
    assert mcolors.same_color(pc.get_edgecolor(), 'none')

    # 关闭颜色映射功能
    pc.set_array(None)
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查边缘颜色是否为 'none'
    assert mcolors.same_color(pc.get_edgecolor(), 'none')
    # 断言检查面颜色是否为 face_default（没有映射）
    assert mcolors.same_color(pc.get_facecolor(), face_default)

    # 恢复颜色映射功能，使用 z 数组作为数据源（必须是一维数组）
    pc.set_array(z)
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查面颜色是否与 mapped 相等
    assert np.array_equal(pc.get_facecolor(), mapped)
    # 断言检查边缘颜色是否为 'none'
    assert mcolors.same_color(pc.get_edgecolor(), 'none')

    # 使用元组而不是字符串来设置颜色
    pc = pcfunc(z, edgecolors=(1, 0, 0), facecolors=(0, 1, 0))
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查面颜色是否与 mapped 相等
    assert np.array_equal(pc.get_facecolor(), mapped)
    # 断言检查边缘颜色是否为 [[1, 0, 0, 1]]
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])

    # 使用 RGB 数组来设置面颜色，但颜色映射会覆盖它
    pc = pcfunc(z, edgecolors=(1, 0, 0), facecolors=np.ones((12, 3)))
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查面颜色是否与 mapped 相等
    assert np.array_equal(pc.get_facecolor(), mapped)
    # 断言检查边缘颜色是否为 [[1, 0, 0, 1]]
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])

    # 关闭颜色映射功能
    pc.set_array(None)
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查面颜色是否为 np.ones((12, 3))
    assert mcolors.same_color(pc.get_facecolor(), np.ones((12, 3)))
    # 断言检查边缘颜色是否为 [[1, 0, 0, 1]]
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])

    # 使用 RGBA 数组来设置面颜色，但颜色映射会覆盖它
    pc = pcfunc(z, edgecolors=(1, 0, 0), facecolors=np.ones((12, 4)))
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查面颜色是否与 mapped 相等
    assert np.array_equal(pc.get_facecolor(), mapped)
    # 断言检查边缘颜色是否为 [[1, 0, 0, 1]]
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])

    # 关闭颜色映射功能
    pc.set_array(None)
    # 再次更新标量映射
    pc.update_scalarmappable()
    # 断言检查面颜色是否为 np.ones((12, 4))
    assert mcolors.same_color(pc.get_facecolor(), np.ones((12, 4)))
    # 断言检查边缘颜色是否为 [[1, 0, 0, 1]]
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])
# 测试 LineCollection 的参数设置
def test_LineCollection_args():
    # 创建一个 LineCollection 对象，无线段数据，设定 linewidth=2.2, edgecolor='r',
    # zorder=3, facecolors=[0, 1, 0, 1]
    lc = LineCollection(None, linewidth=2.2, edgecolor='r',
                        zorder=3, facecolors=[0, 1, 0, 1])
    # 断言获取的 linewidth 是否为 2.2
    assert lc.get_linewidth()[0] == 2.2
    # 断言 edgecolor 是否与 'r' 相同
    assert mcolors.same_color(lc.get_edgecolor(), 'r')
    # 断言获取的 zorder 是否为 3
    assert lc.get_zorder() == 3
    # 断言获取的 facecolor 是否与 [[0, 1, 0, 1]] 相同
    assert mcolors.same_color(lc.get_facecolor(), [[0, 1, 0, 1]])
    
    # 为了避免破坏 mplot3d，LineCollection 内部如果未指定 facecolor，则会设置它。
    # 因此我们需要下面的测试来检验 LineCollection._set_default()。
    lc = LineCollection(None, facecolor=None)
    assert mcolors.same_color(lc.get_facecolor(), 'none')


# 测试数组的维度设置
def test_array_dimensions(pcfunc):
    # 确保可以设置1D、2D和3D数组形状
    z = np.arange(12).reshape(3, 4)
    pc = getattr(plt, pcfunc)(z)
    # 设置为1D数组
    pc.set_array(z.ravel())
    pc.update_scalarmappable()
    # 设置为2D数组
    pc.set_array(z)
    pc.update_scalarmappable()
    # 也可以设置为3D RGB数组
    z = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    pc.set_array(z)
    pc.update_scalarmappable()


# 测试获取 LineCollection 的线段数据
def test_get_segments():
    # 创建一个包含重复线段数据的 LineCollection 对象
    segments = np.tile(np.linspace(0, 1, 256), (2, 1)).T
    lc = LineCollection([segments])

    # 获取 LineCollection 的线段数据
    readback, = lc.get_segments()
    # 断言这些数据是否没有改变
    assert np.all(segments == readback)


# 测试在初始化后设置偏移量
def test_set_offsets_late():
    # 创建一个 IdentityTransform 对象和 CircleCollection 对象
    identity = mtransforms.IdentityTransform()
    sizes = [2]

    # 创建不同设置方式的 CircleCollection 对象
    null = mcollections.CircleCollection(sizes=sizes)
    init = mcollections.CircleCollection(sizes=sizes, offsets=(10, 10))
    late = mcollections.CircleCollection(sizes=sizes)
    late.set_offsets((10, 10))

    # 获取不同方式下的数据边界
    null_bounds = null.get_datalim(identity).bounds
    init_bounds = init.get_datalim(identity).bounds
    late_bounds = late.get_datalim(identity).bounds

    # 断言初始化后设置偏移量和变换是否应用
    assert null_bounds != init_bounds
    assert init_bounds == late_bounds


# 测试设置偏移变换
def test_set_offset_transform():
    # 创建一个仿射变换对象
    skew = mtransforms.Affine2D().skew(2, 2)

    # 创建两个 Collection 对象，分别在初始化和后设置偏移变换
    init = mcollections.Collection(offset_transform=skew)
    late = mcollections.Collection()
    late.set_offset_transform(skew)

    # 断言初始化和后设置的偏移变换是否相同
    assert skew == init.get_offset_transform() == late.get_offset_transform()


# 测试设置偏移单位
def test_set_offset_units():
    # 创建 x 和 y 数据
    x = np.linspace(0, 10, 5)
    y = np.sin(x)
    # 创建日期时间数据
    d = x * np.timedelta64(24, 'h') + np.datetime64('2021-11-29')

    # 创建散点图对象
    sc = plt.scatter(d, y)
    off0 = sc.get_offsets()
    # 使用列表形式设置偏移量，并断言设置前后的偏移量是否相同
    sc.set_offsets(list(zip(d, y)))
    np.testing.assert_allclose(off0, sc.get_offsets())

    # 创建子图对象和散点图对象
    fig, ax = plt.subplots()
    sc = ax.scatter(y, d)
    off0 = sc.get_offsets()
    # 使用列表形式设置偏移量，并断言设置前后的偏移量是否相同
    sc.set_offsets(list(zip(y, d)))
    np.testing.assert_allclose(off0, sc.get_offsets())
def test_check_masked_offsets():
    # 检查 scatter 方法是否正确处理掩码数据
    # 参考: Issue #24545

    # 创建未掩码的 x 数据列表
    unmasked_x = [
        datetime(2022, 12, 15, 4, 49, 52),
        datetime(2022, 12, 15, 4, 49, 53),
        datetime(2022, 12, 15, 4, 49, 54),
        datetime(2022, 12, 15, 4, 49, 55),
        datetime(2022, 12, 15, 4, 49, 56),
    ]

    # 创建掩码的 y 数据数组
    masked_y = np.ma.array([1, 2, 3, 4, 5], mask=[0, 1, 1, 0, 0])

    # 创建图形和轴对象
    fig, ax = plt.subplots()

    # 使用 scatter 方法绘制散点图，传入未掩码的 x 数据和掩码的 y 数据
    ax.scatter(unmasked_x, masked_y)


@check_figures_equal(extensions=["png"])
def test_masked_set_offsets(fig_ref, fig_test):
    # 创建掩码的 x 和未掩码的 y 数据数组
    x = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 1, 0])
    y = np.arange(1, 6)

    # 添加子图到测试和参考图形对象中
    ax_test = fig_test.add_subplot()
    ax_ref = fig_ref.add_subplot()

    # 在测试图中使用 scatter 绘制散点图，并设置偏移量为掩码后的 x 和 y 数据
    scat = ax_test.scatter(x, y)
    scat.set_offsets(np.ma.column_stack([x, y]))

    # 设置测试图的 x 和 y 轴刻度为空
    ax_test.set_xticks([])
    ax_test.set_yticks([])

    # 在参考图中使用 scatter 绘制散点图，并设置偏移量为未掩码的 x 和 y 数据
    ax_ref.scatter([1, 2, 5], [1, 2, 5])
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])


def test_check_offsets_dtype():
    # 检查设置偏移量是否不改变数据类型
    x = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 1, 0])
    y = np.arange(1, 6)

    # 创建图形和轴对象
    fig, ax = plt.subplots()

    # 使用 scatter 绘制散点图，并设置偏移量为掩码后的 x 和 y 数据
    scat = ax.scatter(x, y)
    masked_offsets = np.ma.column_stack([x, y])
    scat.set_offsets(masked_offsets)

    # 断言散点图的偏移量数据类型与掩码后的数据类型一致
    assert isinstance(scat.get_offsets(), type(masked_offsets))

    # 设置偏移量为未掩码的 x 和 y 数据
    unmasked_offsets = np.column_stack([x, y])
    scat.set_offsets(unmasked_offsets)

    # 断言散点图的偏移量数据类型与未掩码的数据类型一致
    assert isinstance(scat.get_offsets(), type(unmasked_offsets))


@pytest.mark.parametrize('gapcolor', ['orange', ['r', 'k']])
@check_figures_equal(extensions=['png'])
@mpl.rc_context({'lines.linewidth': 20})
def test_striped_lines(fig_test, fig_ref, gapcolor):
    # 添加子图到测试和参考图形对象中
    ax_test = fig_test.add_subplot(111)
    ax_ref = fig_ref.add_subplot(111)

    # 设置子图的 x 和 y 轴限制
    for ax in [ax_test, ax_ref]:
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)

    # 创建 x 数据范围
    x = range(1, 6)
    linestyles = [':', '-', '--']

    # 在测试图中使用 vlines 绘制垂直线，设置线型、间隔颜色和透明度
    ax_test.vlines(x, 0, 1, linestyle=linestyles, gapcolor=gapcolor, alpha=0.5)

    # 如果间隔颜色是字符串，则转换为列表
    if isinstance(gapcolor, str):
        gapcolor = [gapcolor]

    # 在参考图中使用 axvline 绘制垂直线，设置线型、间隔颜色和透明度
    for x, gcol, ls in zip(x, itertools.cycle(gapcolor),
                           itertools.cycle(linestyles)):
        ax_ref.axvline(x, 0, 1, linestyle=ls, gapcolor=gcol, alpha=0.5)
```