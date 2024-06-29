# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_spines.py`

```py
import numpy as np   # 导入 NumPy 库，用于数值计算
import pytest       # 导入 Pytest 库，用于单元测试

import matplotlib.pyplot as plt         # 导入 Matplotlib 中的 pyplot 模块，用于绘图
from matplotlib.spines import Spines    # 导入 Matplotlib 中的 Spines 类
from matplotlib.testing.decorators import check_figures_equal, image_comparison   # 导入 Matplotlib 测试装饰器

def test_spine_class():
    """Test Spines and SpinesProxy in isolation."""
    class SpineMock:
        def __init__(self):
            self.val = None   # 初始化一个属性 val

        def set(self, **kwargs):
            vars(self).update(kwargs)   # 更新对象的属性

        def set_val(self, val):
            self.val = val    # 设置对象的 val 属性为给定值

    spines_dict = {   # 创建一个包含 SpineMock 对象的字典
        'left': SpineMock(),
        'right': SpineMock(),
        'top': SpineMock(),
        'bottom': SpineMock(),
    }
    spines = Spines(**spines_dict)   # 创建 Spines 对象，传入字典作为参数

    assert spines['left'] is spines_dict['left']   # 断言：访问 spines 左边的 SpineMock 对象与原始字典中的相同
    assert spines.left is spines_dict['left']      # 断言：访问 spines 的 left 属性与原始字典中的相同

    spines[['left', 'right']].set_val('x')   # 设置左右边的 SpineMock 对象的 val 属性为 'x'
    assert spines.left.val == 'x'            # 断言：左边的 SpineMock 对象的 val 属性为 'x'
    assert spines.right.val == 'x'           # 断言：右边的 SpineMock 对象的 val 属性为 'x'
    assert spines.top.val is None            # 断言：顶部的 SpineMock 对象的 val 属性为 None
    assert spines.bottom.val is None         # 断言：底部的 SpineMock 对象的 val 属性为 None

    spines[:].set_val('y')    # 设置所有 SpineMock 对象的 val 属性为 'y'
    assert all(spine.val == 'y' for spine in spines.values())   # 断言：所有 SpineMock 对象的 val 属性都为 'y'

    spines[:].set(foo='bar')   # 设置所有 SpineMock 对象的 foo 属性为 'bar'
    assert all(spine.foo == 'bar' for spine in spines.values())   # 断言：所有 SpineMock 对象的 foo 属性都为 'bar'

    with pytest.raises(AttributeError, match='foo'):   # 断言：访问不存在的属性 foo 会引发 AttributeError 异常
        spines.foo
    with pytest.raises(KeyError, match='foo'):        # 断言：访问不存在的键 'foo' 会引发 KeyError 异常
        spines['foo']
    with pytest.raises(KeyError, match='foo, bar'):   # 断言：同时访问不存在的键 'foo' 和 'bar' 会引发 KeyError 异常
        spines[['left', 'foo', 'right', 'bar']]
    with pytest.raises(ValueError, match='single list'):   # 断言：传递不支持的切片索引会引发 ValueError 异常
        spines['left', 'right']
    with pytest.raises(ValueError, match='Spines does not support slicing'):   # 断言：尝试使用切片索引会引发 ValueError 异常
        spines['left':'right']
    with pytest.raises(ValueError, match='Spines does not support slicing'):   # 断言：尝试使用切片索引会引发 ValueError 异常
        spines['top':]

@image_comparison(['spines_axes_positions'])
def test_spines_axes_positions():
    # SF bug 2852168
    fig = plt.figure()   # 创建一个新的 Matplotlib 图形对象
    x = np.linspace(0, 2*np.pi, 100)   # 生成一个包含 100 个元素的等间距数组
    y = 2*np.sin(x)   # 计算正弦函数的值
    ax = fig.add_subplot(1, 1, 1)   # 在图形中添加一个子图
    ax.set_title('centered spines')   # 设置子图的标题
    ax.plot(x, y)   # 绘制正弦函数曲线
    ax.spines.right.set_position(('axes', 0.1))   # 设置右边脊柱的位置
    ax.yaxis.set_ticks_position('right')   # 设置 y 轴的刻度位置为右侧
    ax.spines.top.set_position(('axes', 0.25))   # 设置顶部脊柱的位置
    ax.xaxis.set_ticks_position('top')   # 设置 x 轴的刻度位置为顶部
    ax.spines.left.set_color('none')   # 隐藏左边脊柱
    ax.spines.bottom.set_color('none')   # 隐藏底部脊柱

@image_comparison(['spines_data_positions'])
def test_spines_data_positions():
    fig, ax = plt.subplots()   # 创建一个包含单个子图的图形对象
    ax.spines.left.set_position(('data', -1.5))   # 设置左边脊柱的位置
    ax.spines.top.set_position(('data', 0.5))     # 设置顶部脊柱的位置
    ax.spines.right.set_position(('data', -0.5))  # 设置右边脊柱的位置
    ax.spines.bottom.set_position('zero')         # 设置底部脊柱的位置为零点
    ax.set_xlim([-2, 2])   # 设置 x 轴的显示范围
    ax.set_ylim([-2, 2])   # 设置 y 轴的显示范围

@check_figures_equal(extensions=["png"])
def test_spine_nonlinear_data_positions(fig_test, fig_ref):
    plt.style.use("default")   # 使用默认的 Matplotlib 样式

    ax = fig_test.add_subplot()   # 在测试图形对象中添加一个子图
    ax.set(xscale="log", xlim=(.1, 1))   # 设置 x 轴的缩放类型为对数轴，显示范围为 0.1 到 1
    # Use position="data" to visually swap the left and right spines, using
    # linewidth to distinguish them.  The calls to tick_params removes labels
    # (for image comparison purposes) and harmonizes tick positions with the
    # reference).
    # 设置左边轴脊柱位置为数据点 1
    ax.spines.left.set_position(("data", 1))
    # 设置左边轴脊柱线宽为 2
    ax.spines.left.set_linewidth(2)
    # 设置右边轴脊柱位置为数据点 0.1
    ax.spines.right.set_position(("data", .1))
    # 设置 y 轴的刻度参数：不显示左边标签，刻度线朝内方向

    # 在参考图上添加一个子图 ax
    ax = fig_ref.add_subplot()
    # 设置 x 轴为对数坐标轴，限定 x 轴范围为 (0.1, 1)
    ax.set(xscale="log", xlim=(.1, 1))
    # 设置右边轴脊柱线宽为 2
    ax.spines.right.set_linewidth(2)
    # 设置 y 轴的刻度参数：不显示左边标签，不显示左边刻度线，显示右边刻度线
@image_comparison(['spines_capstyle'])
def test_spines_capstyle():
    # 使用装饰器创建图像比较测试函数，比较 spines_capstyle 图像
    # 解决问题编号 2542
    plt.rc('axes', linewidth=20)  # 设置所有轴的线宽为 20
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    ax.set_xticks([])  # 设置 X 轴刻度为空列表，即无刻度显示
    ax.set_yticks([])  # 设置 Y 轴刻度为空列表，即无刻度显示


def test_label_without_ticks():
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.3)  # 调整子图布局，左边和底部各留出 30% 的空白
    ax.plot(np.arange(10))  # 绘制折线图，X 轴为 np.arange(10)
    ax.yaxis.set_ticks_position('left')  # 设置 Y 轴刻度位置在左侧
    ax.spines.left.set_position(('outward', 30))  # 设置左边框线向外移动 30 个单位
    ax.spines.right.set_visible(False)  # 隐藏右边框线
    ax.set_ylabel('y label')  # 设置 Y 轴标签为 'y label'
    ax.xaxis.set_ticks_position('bottom')  # 设置 X 轴刻度位置在底部
    ax.spines.bottom.set_position(('outward', 30))  # 设置底部框线向外移动 30 个单位
    ax.spines.top.set_visible(False)  # 隐藏顶部框线
    ax.set_xlabel('x label')  # 设置 X 轴标签为 'x label'
    ax.xaxis.set_ticks([])  # 设置 X 轴刻度为空列表，即无刻度显示
    ax.yaxis.set_ticks([])  # 设置 Y 轴刻度为空列表，即无刻度显示
    plt.draw()  # 绘制图形

    # 获取左边框线对象并计算其边界框
    spine = ax.spines.left
    spinebbox = spine.get_transform().transform_path(
        spine.get_path()).get_extents()
    assert ax.yaxis.label.get_position()[0] < spinebbox.xmin, \
        "Y-Axis label not left of the spine"  # 断言：Y 轴标签应位于左边框线左侧

    # 获取底部框线对象并计算其边界框
    spine = ax.spines.bottom
    spinebbox = spine.get_transform().transform_path(
        spine.get_path()).get_extents()
    assert ax.xaxis.label.get_position()[1] < spinebbox.ymin, \
        "X-Axis label not below the spine"  # 断言：X 轴标签应位于底部框线下方


@image_comparison(['black_axes'])
def test_spines_black_axes():
    # 使用装饰器创建图像比较测试函数，比较 black_axes 图像
    # GitHub 问题编号 #18804
    plt.rcParams["savefig.pad_inches"] = 0  # 设置保存图像时的边距为 0
    plt.rcParams["savefig.bbox"] = 'tight'  # 设置保存图像时的边界框为紧凑模式
    fig = plt.figure(0, figsize=(4, 4))  # 创建大小为 4x4 的图形对象
    ax = fig.add_axes((0, 0, 1, 1))  # 在图形上添加大小为整个图形的坐标轴
    ax.set_xticklabels([])  # 设置 X 轴刻度标签为空列表，即无刻度标签显示
    ax.set_yticklabels([])  # 设置 Y 轴刻度标签为空列表，即无刻度标签显示
    ax.set_xticks([])  # 设置 X 轴刻度为空列表，即无刻度显示
    ax.set_yticks([])  # 设置 Y 轴刻度为空列表，即无刻度显示
    ax.set_facecolor((0, 0, 0))  # 设置坐标轴背景颜色为黑色
```