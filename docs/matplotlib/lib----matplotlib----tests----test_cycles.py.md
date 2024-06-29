# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_cycles.py`

```py
# 导入 contextlib 模块，用于处理上下文管理
# 导入 StringIO 类从 io 模块中，用于创建内存中的文本 I/O 流
import contextlib
from io import StringIO

# 导入 matplotlib 库并使用 mpl 别名，导入其中的 pyplot 模块并使用 plt 别名
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# 导入 pytest 模块，用于编写和运行测试
import pytest

# 从 cycler 模块中导入 cycler 类，用于自定义循环属性
from cycler import cycler


# 定义一个测试函数 test_colorcycle_basic
def test_colorcycle_basic():
    # 创建一个图形和一个坐标系对象
    fig, ax = plt.subplots()
    # 设置坐标系对象 ax 的属性循环，设置颜色为 ['r', 'g', 'y']
    ax.set_prop_cycle(cycler('color', ['r', 'g', 'y']))
    # 绘制十次线条，每次都是相同的数据
    for _ in range(4):
        ax.plot(range(10), range(10))
    # 断言所有线条的颜色与预期的颜色列表相同
    assert [l.get_color() for l in ax.lines] == ['r', 'g', 'y', 'r']


# 定义一个测试函数 test_marker_cycle
def test_marker_cycle():
    # 创建一个图形和一个坐标系对象
    fig, ax = plt.subplots()
    # 设置坐标系对象 ax 的属性循环，设置颜色为 ['r', 'g', 'y'] 和标记为 ['.', '*', 'x']
    ax.set_prop_cycle(cycler('c', ['r', 'g', 'y']) +
                      cycler('marker', ['.', '*', 'x']))
    # 绘制十次线条，每次都是相同的数据
    for _ in range(4):
        ax.plot(range(10), range(10))
    # 断言所有线条的颜色与预期的颜色列表相同
    assert [l.get_color() for l in ax.lines] == ['r', 'g', 'y', 'r']
    # 断言所有线条的标记与预期的标记列表相同
    assert [l.get_marker() for l in ax.lines] == ['.', '*', 'x', '.']


# 定义一个测试函数 test_valid_marker_cycles
def test_valid_marker_cycles():
    # 创建一个图形和一个坐标系对象
    fig, ax = plt.subplots()
    # 设置坐标系对象 ax 的属性循环，使用 marker 参数，但是没有颜色参数
    ax.set_prop_cycle(cycler(marker=[1, "+", ".", 4]))


# 定义一个测试函数 test_marker_cycle_kwargs_arrays_iterators
def test_marker_cycle_kwargs_arrays_iterators():
    # 创建一个图形和一个坐标系对象
    fig, ax = plt.subplots()
    # 设置坐标系对象 ax 的属性循环，颜色使用 numpy 数组 ['r', 'g', 'y']，标记使用迭代器 ['.', '*', 'x']
    ax.set_prop_cycle(c=np.array(['r', 'g', 'y']),
                      marker=iter(['.', '*', 'x']))
    # 绘制十次线条，每次都是相同的数据
    for _ in range(4):
        ax.plot(range(10), range(10))
    # 断言所有线条的颜色与预期的颜色列表相同
    assert [l.get_color() for l in ax.lines] == ['r', 'g', 'y', 'r']
    # 断言所有线条的标记与预期的标记列表相同
    assert [l.get_marker() for l in ax.lines] == ['.', '*', 'x', '.']


# 定义一个测试函数 test_linestylecycle_basic
def test_linestylecycle_basic():
    # 创建一个图形和一个坐标系对象
    fig, ax = plt.subplots()
    # 设置坐标系对象 ax 的属性循环，设置线条风格为 ['-', '--', ':']
    ax.set_prop_cycle(cycler('ls', ['-', '--', ':']))
    # 绘制十次线条，每次都是相同的数据
    for _ in range(4):
        ax.plot(range(10), range(10))
    # 断言所有线条的线条风格与预期的风格列表相同
    assert [l.get_linestyle() for l in ax.lines] == ['-', '--', ':', '-']


# 定义一个测试函数 test_fillcycle_basic
def test_fillcycle_basic():
    # 创建一个图形和一个坐标系对象
    fig, ax = plt.subplots()
    # 设置坐标系对象 ax 的属性循环，设置颜色为 ['r', 'g', 'y']，填充图案为 ['xx', 'O', '|-']，线条风格为 ['-', '--', ':']
    ax.set_prop_cycle(cycler('c',  ['r', 'g', 'y']) +
                      cycler('hatch', ['xx', 'O', '|-']) +
                      cycler('linestyle', ['-', '--', ':']))
    # 绘制十次填充，每次都是相同的数据
    for _ in range(4):
        ax.fill(range(10), range(10))
    # 断言所有填充图形的面颜色与预期的颜色列表相同
    assert ([p.get_facecolor() for p in ax.patches]
            == [mpl.colors.to_rgba(c) for c in ['r', 'g', 'y', 'r']])
    # 断言所有填充图形的填充图案与预期的图案列表相同
    assert [p.get_hatch() for p in ax.patches] == ['xx', 'O', '|-', 'xx']
    # 断言所有填充图形的线条风格与预期的风格列表相同
    assert [p.get_linestyle() for p in ax.patches] == ['-', '--', ':', '-']


# 定义一个测试函数 test_fillcycle_ignore
def test_fillcycle_ignore():
    # 创建一个图形和一个坐标系对象
    fig, ax = plt.subplots()
    # 设置坐标系对象 ax 的属性循环，设置颜色为 ['r', 'g', 'y']，填充图案为 ['xx', 'O', '|-']，标记为 ['.', '*', 'D']
    ax.set_prop_cycle(cycler('color',  ['r', 'g', 'y']) +
                      cycler('hatch', ['xx', 'O', '|-']) +
                      cycler('marker', ['.', '*', 'D']))
    t = range(10)
    # 第一个填充，指定了颜色和填充图案，但是标记属性不是 Polygon 的属性，会被忽略
    ax.fill(t, t, 'r', hatch='xx')
    # 第二个填充，允许循环器继续，但是指定了填充图案
    ax.fill(t, t, hatch='O')
    # 后两个填充，使用默认颜色和填充图案
    ax.fill(t, t)
    ax.fill(t, t)
    # 断言所有填充图形的面颜色与预期的颜色列表相同
    assert ([p.get_facecolor() for p in ax.patches]
            == [mpl.colors.to_rgba(c) for c in ['r', 'r', 'g', 'y']])
    # 断言所有填充图形的填充图案与预期的图案列表相同
    assert [p.get_hatch() for p in ax.patches] == ['xx', 'O', 'O', '|-']


# 定义一个测试函数 test_property_collision_plot，此处代码有错误，无法正常运行
# 因为 set_prop_cycle 期望接收一个 cycler 对象，但是给定了一个字符串 'linewidth' 和一个列表 [2, 4]
# 此处代码只作为示例，展示了一个无效的用法，不符合预期的 cycler 参数格式
def test_property_collision_plot():
    fig, ax = plt.subplots()
    ax.set_prop_cycle('linewidth', [2, 4])
    # 创建一个包含 0 到 9 的整数范围的对象 t
    t = range(10)
    
    # 使用循环变量 c 遍历从 1 到 3 的整数（即 1, 2, 3）
    for c in range(1, 4):
        # 在当前图形对象 ax 上绘制以 t 为横纵坐标的线条，线宽为 0.1
        ax.plot(t, t, lw=0.1)
    
    # 在当前图形对象 ax 上绘制以 t 为横纵坐标的线条，默认线宽
    ax.plot(t, t)
    
    # 在当前图形对象 ax 上绘制以 t 为横纵坐标的线条，默认线宽
    ax.plot(t, t)
    
    # 断言当前图形对象 ax 的所有线条的线宽是否与给定列表匹配
    assert [l.get_linewidth() for l in ax.lines] == [0.1, 0.1, 0.1, 2, 4]
def test_property_collision_fill():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 设置属性循环，指定线宽和颜色序列
    ax.set_prop_cycle(linewidth=[2, 3, 4, 5, 6], facecolor='bgcmy')
    # 创建一个长度为10的迭代对象
    t = range(10)
    # 遍历1到3的整数序列
    for c in range(1, 4):
        # 在坐标轴上填充数据
        ax.fill(t, t, lw=0.1)
    # 再次在坐标轴上填充数据
    ax.fill(t, t)
    # 又一次在坐标轴上填充数据
    ax.fill(t, t)
    # 断言检查填充对象的面颜色是否与指定颜色序列匹配
    assert ([p.get_facecolor() for p in ax.patches]
            == [mpl.colors.to_rgba(c) for c in 'bgcmy'])
    # 断言检查填充对象的线宽是否与指定线宽序列匹配
    assert [p.get_linewidth() for p in ax.patches] == [0.1, 0.1, 0.1, 5, 6]


def test_valid_input_forms():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 下面的设置不应引发错误
    ax.set_prop_cycle(None)
    ax.set_prop_cycle(cycler('linewidth', [1, 2]))
    ax.set_prop_cycle('color', 'rgywkbcm')
    ax.set_prop_cycle('lw', (1, 2))
    ax.set_prop_cycle('linewidth', [1, 2])
    ax.set_prop_cycle('linewidth', iter([1, 2]))
    ax.set_prop_cycle('linewidth', np.array([1, 2]))
    ax.set_prop_cycle('color', np.array([[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]]))
    ax.set_prop_cycle('dashes', [[], [13, 2], [8, 3, 1, 3]])
    ax.set_prop_cycle(lw=[1, 2], color=['k', 'w'], ls=['-', '--'])
    ax.set_prop_cycle(lw=np.array([1, 2]),
                      color=np.array(['k', 'w']),
                      ls=np.array(['-', '--']))


def test_cycle_reset():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建三个用于捕获输出的字符串IO对象
    prop0 = StringIO()
    prop1 = StringIO()
    prop2 = StringIO()

    # 用于捕获标准输出到prop0
    with contextlib.redirect_stdout(prop0):
        plt.getp(ax.plot([1, 2], label="label")[0])

    # 设置新的属性循环，指定线宽序列
    ax.set_prop_cycle(linewidth=[10, 9, 4])
    # 用于捕获标准输出到prop1
    with contextlib.redirect_stdout(prop1):
        plt.getp(ax.plot([1, 2], label="label")[0])
    # 断言：prop1的值不应该与prop0相同，即属性循环的设置已生效
    assert prop1.getvalue() != prop0.getvalue()

    # 重置属性循环为默认值
    ax.set_prop_cycle(None)
    # 用于捕获标准输出到prop2
    with contextlib.redirect_stdout(prop2):
        plt.getp(ax.plot([1, 2], label="label")[0])
    # 断言：prop2的值应该与prop0相同，即属性循环已成功重置
    assert prop2.getvalue() == prop0.getvalue()


def test_invalid_input_forms():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()

    # 使用pytest断言：设置属性循环时应该引发TypeError或ValueError异常
    with pytest.raises((TypeError, ValueError)):
        ax.set_prop_cycle(1)
    with pytest.raises((TypeError, ValueError)):
        ax.set_prop_cycle([1, 2])

    with pytest.raises((TypeError, ValueError)):
        ax.set_prop_cycle('color', 'fish')

    with pytest.raises((TypeError, ValueError)):
        ax.set_prop_cycle('linewidth', 1)
    with pytest.raises((TypeError, ValueError)):
        ax.set_prop_cycle('linewidth', {1, 2})
    with pytest.raises((TypeError, ValueError)):
        ax.set_prop_cycle(linewidth=1, color='r')

    with pytest.raises((TypeError, ValueError)):
        ax.set_prop_cycle('foobar', [1, 2])
    with pytest.raises((TypeError, ValueError)):
        ax.set_prop_cycle(foobar=[1, 2])

    with pytest.raises((TypeError, ValueError)):
        ax.set_prop_cycle(cycler(foobar=[1, 2]))
    with pytest.raises(ValueError):
        ax.set_prop_cycle(cycler(color='rgb', c='cmy'))
```