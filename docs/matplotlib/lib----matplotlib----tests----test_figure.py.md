# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_figure.py`

```
# 导入必要的模块和库
import copy  # 导入深拷贝功能
from datetime import datetime  # 导入日期时间处理功能
import io  # 导入输入输出流功能
import pickle  # 导入对象序列化和反序列化功能
import platform  # 导入获取平台信息功能
from threading import Timer  # 导入定时器功能
from types import SimpleNamespace  # 导入简单命名空间类型
import warnings  # 导入警告处理功能

import numpy as np  # 导入数值计算库 NumPy
import pytest  # 导入测试框架 Pytest
from PIL import Image  # 导入图像处理库 PIL

import matplotlib as mpl  # 导入绘图库 Matplotlib
from matplotlib import gridspec  # 导入网格布局
from matplotlib.testing.decorators import image_comparison, check_figures_equal  # 导入图像对比和图像相等性检查装饰器
from matplotlib.axes import Axes  # 导入图轴功能
from matplotlib.backend_bases import KeyEvent, MouseEvent  # 导入键盘事件和鼠标事件
from matplotlib.figure import Figure, FigureBase  # 导入图形和图形基类
from matplotlib.layout_engine import (  # 导入布局引擎类
    ConstrainedLayoutEngine,
    TightLayoutEngine,
    PlaceHolderLayoutEngine,
)
from matplotlib.ticker import (  # 导入刻度线格式化类
    AutoMinorLocator,
    FixedFormatter,
    ScalarFormatter,
)
import matplotlib.pyplot as plt  # 导入绘图模块
import matplotlib.dates as mdates  # 导入日期格式化功能

# 定义测试函数，对比图像是否对齐标签
@image_comparison(['figure_align_labels'], extensions=['png', 'svg'],
                  tol=0 if platform.machine() == 'x86_64' else 0.01)
def test_align_labels():
    # 创建一个紧凑布局的图形对象
    fig = plt.figure(layout='tight')
    # 创建一个3x3的网格布局
    gs = gridspec.GridSpec(3, 3)

    # 在第一行的前两列添加子图
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(np.arange(0, 1e6, 1000))  # 绘制数据
    ax.set_ylabel('Ylabel0 0')  # 设置y轴标签
    ax = fig.add_subplot(gs[0, -1])
    ax.plot(np.arange(0, 1e4, 100))  # 绘制数据

    # 遍历第二行的三列
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.set_ylabel('YLabel1 %d' % i)  # 设置y轴标签
        ax.set_xlabel('XLabel1 %d' % i)  # 设置x轴标签
        if i in [0, 2]:  # 如果是第一列或第三列
            ax.xaxis.set_label_position("top")  # 将x轴标签位置设置在顶部
            ax.xaxis.tick_top()  # 设置x轴刻度在顶部显示
        if i == 0:  # 如果是第一列
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)  # 将x轴刻度标签旋转90度
        if i == 2:  # 如果是第三列
            ax.yaxis.set_label_position("right")  # 将y轴标签位置设置在右侧
            ax.yaxis.tick_right()  # 设置y轴刻度在右侧显示

    # 遍历第三行的三列
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.set_xlabel(f'XLabel2 {i}')  # 设置x轴标签
        ax.set_ylabel(f'YLabel2 {i}')  # 设置y轴标签

        if i == 2:  # 如果是第三列
            ax.plot(np.arange(0, 1e4, 10))  # 绘制数据
            ax.yaxis.set_label_position("right")  # 将y轴标签位置设置在右侧
            ax.yaxis.tick_right()  # 设置y轴刻度在右侧显示
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)  # 将x轴刻度标签旋转90度

    fig.align_labels()  # 对齐所有子图的标签


# 定义测试函数，对比图像是否对齐标题
@image_comparison(['figure_align_titles_tight.png',
                   'figure_align_titles_constrained.png'],
                  tol=0 if platform.machine() == 'x86_64' else 0.022,
                  style='mpl20')
def test_align_titles():
    # 对于紧凑布局和约束布局分别进行测试
    for layout in ['tight', 'constrained']:
        fig, axs = plt.subplots(1, 2, layout=layout, width_ratios=[2, 1])

        ax = axs[0]
        ax.plot(np.arange(0, 1e6, 1000))  # 绘制数据
        ax.set_title('Title0 left', loc='left')  # 设置标题在左侧
        ax.set_title('Title0 center', loc='center')  # 设置标题在中间
        ax.set_title('Title0 right', loc='right')  # 设置标题在右侧

        ax = axs[1]
        ax.plot(np.arange(0, 1e4, 100))  # 绘制数据
        ax.set_title('Title1')  # 设置标题
        ax.set_xlabel('Xlabel0')  # 设置x轴标签
        ax.xaxis.set_label_position("top")  # 将x轴标签位置设置在顶部
        ax.xaxis.tick_top()  # 设置x轴刻度在顶部显示
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)  # 将x轴刻度标签旋转90度

        fig.align_titles()  # 对齐所有子图的标题


# 定义测试函数，检查当存在多余子图时对齐标签功能是否正常
def test_align_labels_stray_axes():
    # 创建一个包含2行2列子图的图像对象
    fig, axs = plt.subplots(2, 2)
    
    # 遍历所有子图，并设置它们的 x 轴标签为 'Boo'，注意应为 set_xlabel
    for nn, ax in enumerate(axs.flat):
        ax.set_xlabel('Boo')
        # 设置每个子图的 y 轴标签为 'Who'，应为 set_ylabel
        ax.set_xlabel('Who')
        # 绘制每个子图中的数据，x 轴和 y 轴的数据为 nn 次方
        ax.plot(np.arange(4)**nn, np.arange(4)**nn)
    
    # 对整个图像对象进行纵向对齐标签
    fig.align_ylabels()
    # 对整个图像对象进行横向对齐标签
    fig.align_xlabels()
    # 在绘图时不进行渲染的情况下绘制图像，应为 fig.canvas.draw_idle()
    fig.draw_without_rendering()
    
    # 创建一个长度为 4 的全零数组 xn 和 yn
    xn = np.zeros(4)
    yn = np.zeros(4)
    
    # 再次遍历所有子图，并获取它们 x 轴和 y 轴标签的位置信息，更新 xn 和 yn 数组
    for nn, ax in enumerate(axs.flat):
        # 获取当前子图 x 轴标签的纵向位置
        yn[nn] = ax.xaxis.label.get_position()[1]
        # 获取当前子图 y 轴标签的横向位置
        xn[nn] = ax.yaxis.label.get_position()[0]
    
    # 使用 np.testing.assert_allclose 检查 xn 数组的前两个元素与后两个元素是否近似相等
    np.testing.assert_allclose(xn[:2], xn[2:])
    # 使用 np.testing.assert_allclose 检查 yn 数组的奇数索引与偶数索引元素是否近似相等
    np.testing.assert_allclose(yn[::2], yn[1::2])
    
    # 创建一个包含2行2列子图的图像对象，并启用约束布局
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    
    # 遍历所有子图，并设置它们的 x 轴标签为 'Boo'，注意应为 set_xlabel
    for nn, ax in enumerate(axs.flat):
        ax.set_xlabel('Boo')
        # 设置每个子图的 y 轴标签为 'Who'，应为 set_ylabel
        ax.set_xlabel('Who')
        # 在每个子图上绘制随机颜色的网格图
        pc = ax.pcolormesh(np.random.randn(10, 10))
    
    # 在整个图像对象上创建一个颜色条，并将其添加到最后一个子图上，应为 fig.colorbar(pc, ax=ax)
    fig.colorbar(pc, ax=ax)
    # 对整个图像对象进行纵向对齐标签
    fig.align_ylabels()
    # 对整个图像对象进行横向对齐标签
    fig.align_xlabels()
    # 在绘图时不进行渲染的情况下绘制图像，应为 fig.canvas.draw_idle()
    fig.draw_without_rendering()
    
    # 创建一个长度为 4 的全零数组 xn 和 yn
    xn = np.zeros(4)
    yn = np.zeros(4)
    
    # 再次遍历所有子图，并获取它们 x 轴和 y 轴标签的位置信息，更新 xn 和 yn 数组
    for nn, ax in enumerate(axs.flat):
        # 获取当前子图 x 轴标签的纵向位置
        yn[nn] = ax.xaxis.label.get_position()[1]
        # 获取当前子图 y 轴标签的横向位置
        xn[nn] = ax.yaxis.label.get_position()[0]
    
    # 使用 np.testing.assert_allclose 检查 xn 数组的前两个元素与后两个元素是否近似相等
    np.testing.assert_allclose(xn[:2], xn[2:])
    # 使用 np.testing.assert_allclose 检查 yn 数组的奇数索引与偶数索引元素是否近似相等
    np.testing.assert_allclose(yn[::2], yn[1::2])
def test_figure_label():
    # pyplot figure creation, selection, and closing with label/number/instance

    # 关闭所有已存在的图形窗口
    plt.close('all')

    # 创建一个名为 'today' 的新图形并返回其引用
    fig_today = plt.figure('today')

    # 创建编号为 3 的新图形
    plt.figure(3)

    # 创建一个名为 'tomorrow' 的新图形
    plt.figure('tomorrow')

    # 创建一个无标题的新图形
    plt.figure()

    # 创建编号为 0 的新图形
    plt.figure(0)

    # 创建编号为 1 的新图形
    plt.figure(1)

    # 选择已存在的编号为 3 的图形
    plt.figure(3)

    # 断言当前所有图形的编号
    assert plt.get_fignums() == [0, 1, 3, 4, 5]

    # 断言当前所有图形的标签
    assert plt.get_figlabels() == ['', 'today', '', 'tomorrow', '']

    # 关闭编号为 10 的图形
    plt.close(10)

    # 关闭当前活动图形
    plt.close()

    # 关闭编号为 5 的图形
    plt.close(5)

    # 关闭标签为 'tomorrow' 的图形
    plt.close('tomorrow')

    # 断言当前所有图形的编号
    assert plt.get_fignums() == [0, 1]

    # 断言当前所有图形的标签
    assert plt.get_figlabels() == ['', 'today']

    # 选择之前定义的 fig_today 图形
    plt.figure(fig_today)

    # 断言当前活动图形是否为 fig_today
    assert plt.gcf() == fig_today

    # 使用 pytest 断言捕获 ValueError 异常，验证 Figure() 作为参数时的行为
    with pytest.raises(ValueError):
        plt.figure(Figure())


def test_fignum_exists():
    # pyplot figure creation, selection and closing with fignum_exists

    # 创建一个标签为 'one' 的新图形
    plt.figure('one')

    # 创建一个编号为 2 的新图形
    plt.figure(2)

    # 创建一个标签为 'three' 的新图形
    plt.figure('three')

    # 创建一个无标题的新图形
    plt.figure()

    # 断言标签为 'one' 的图形是否存在
    assert plt.fignum_exists('one')

    # 断言编号为 2 的图形是否存在
    assert plt.fignum_exists(2)

    # 断言标签为 'three' 的图形是否存在
    assert plt.fignum_exists('three')

    # 断言编号为 4 的图形是否存在（实际不存在）
    assert plt.fignum_exists(4)

    # 关闭标签为 'one' 的图形
    plt.close('one')

    # 关闭编号为 4 的图形
    plt.close(4)

    # 断言标签为 'one' 的图形是否不存在
    assert not plt.fignum_exists('one')

    # 断言编号为 4 的图形是否不存在
    assert not plt.fignum_exists(4)


def test_clf_keyword():
    # test if existing figure is cleared with figure() and subplots()

    text1 = 'A fancy plot'
    text2 = 'Really fancy!'

    # 创建一个编号为 1 的新图形，并设置标题为 text1
    fig0 = plt.figure(num=1)
    fig0.suptitle(text1)

    # 断言图形中所有文本对象的内容是否为 text1
    assert [t.get_text() for t in fig0.texts] == [text1]

    # 创建一个编号为 1 的新图形，不清空已存在的内容
    fig1 = plt.figure(num=1, clear=False)

    # 在图形中添加文本内容 text2
    fig1.text(0.5, 0.5, text2)

    # 断言 fig0 和 fig1 引用相同的图形对象
    assert fig0 is fig1

    # 断言图形中所有文本对象的内容为 text1 和 text2
    assert [t.get_text() for t in fig1.texts] == [text1, text2]

    # 创建一个包含两个子图的编号为 1 的新图形，并清空已存在的内容
    fig2, ax2 = plt.subplots(2, 1, num=1, clear=True)

    # 断言 fig0 和 fig2 引用相同的图形对象
    assert fig0 is fig2

    # 断言图形中所有文本对象的内容为空列表
    assert [t.get_text() for t in fig2.texts] == []


@image_comparison(['figure_today'],
                  tol=0.015 if platform.machine() == 'arm64' else 0)
def test_figure():
    # named figure support

    # 创建一个标签为 'today' 的新图形并返回其引用
    fig = plt.figure('today')

    # 在图形中添加一个子图
    ax = fig.add_subplot()

    # 设置子图的标题为图形的标签
    ax.set_title(fig.get_label())

    # 在另一个标签为 'tomorrow' 的新图形中绘制一条红色线
    plt.figure('tomorrow')
    plt.plot([0, 1], [1, 0], 'r')

    # 返回到之前的 'today' 图形，并确保红色线不存在
    plt.figure('today')
    plt.close('tomorrow')


@image_comparison(['figure_legend'])
def test_figure_legend():
    # 创建包含两个子图的新图形
    fig, axs = plt.subplots(2)

    # 在第一个子图中绘制两条线并添加图例
    axs[0].plot([0, 1], [1, 0], label='x', color='g')
    axs[0].plot([0, 1], [0, 1], label='y', color='r')
    axs[0].plot([0, 1], [0.5, 0.5], label='y', color='k')

    # 在第二个子图中绘制两条线
    axs[1].plot([0, 1], [1, 0], label='_y', color='r')
    axs[1].plot([0, 1], [0, 1], label='z', color='b')

    # 在图形中添加图例
    fig.legend()


def test_gca():
    # 创建一个新图形
    fig = plt.figure()

    # 使用 add_axes() 方法创建一个新的 Axes 对象，并将其设置为当前 Axes
    ax0 = fig.add_axes([0, 0, 1, 1])

    # 断言当前的 gca() 返回创建的 Axes 对象 ax0
    assert fig.gca() is ax0

    # 使用 add_subplot() 方法创建一个新的 Axes 对象，并将其设置为当前 Axes
    ax1 = fig.add_subplot(111)

    # 断言当前的 gca() 返回创建的 Axes 对象 ax1
    assert fig.gca() is ax1

    # 将之前创建的 ax0 添加到现有的 Axes 中，不改变存储顺序，但会使其成为当前 Axes
    fig.add_axes(ax0)

    # 断言当前所有的 Axes 对象列表
    assert fig.axes == [ax0, ax1]
    # 断言当前图形 fig 的当前轴（gca）是 ax0
    assert fig.gca() is ax0
    
    # sca() 方法不应改变轴的存储顺序，即按照添加的顺序排列。
    fig.sca(ax0)
    # 断言图形 fig 的轴列表应为 [ax0, ax1]
    assert fig.axes == [ax0, ax1]
    
    # 在现有的轴 ax1 上调用 add_subplot 不应改变轴的存储顺序，但会将其设置为当前轴。
    fig.add_subplot(ax1)
    # 断言图形 fig 的轴列表仍应为 [ax0, ax1]
    assert fig.axes == [ax0, ax1]
    # 断言当前图形 fig 的当前轴（gca）现在是 ax1
    assert fig.gca() is ax1
# 定义一个测试函数，用于验证向图形对象添加子图的各种情况是否正确处理
def test_add_subplot_subclass():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 向图形对象添加一个子图，使用默认的坐标轴类型 Axes
    fig.add_subplot(axes_class=Axes)
    # 预期抛出 ValueError 异常，因为在指定了 Axes 类型的情况下不支持 3D 投影
    with pytest.raises(ValueError):
        fig.add_subplot(axes_class=Axes, projection="3d")
    # 预期抛出 ValueError 异常，因为在指定了 Axes 类型的情况下不支持极坐标
    with pytest.raises(ValueError):
        fig.add_subplot(axes_class=Axes, polar=True)
    # 预期抛出 ValueError 异常，因为在指定了 3D 投影的情况下不支持极坐标
    with pytest.raises(ValueError):
        fig.add_subplot(projection="3d", polar=True)
    # 预期抛出 TypeError 异常，因为projection参数需要一个字符串，而不是整数
    with pytest.raises(TypeError):
        fig.add_subplot(projection=42)


# 定义一个测试函数，用于验证向图形对象添加子图时的无效参数处理情况
def test_add_subplot_invalid():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 预期抛出 ValueError 异常，因为列数必须是正整数
    with pytest.raises(ValueError, match='Number of columns must be a positive integer'):
        fig.add_subplot(2, 0, 1)
    # 预期抛出 ValueError 异常，因为行数必须是正整数
    with pytest.raises(ValueError, match='Number of rows must be a positive integer'):
        fig.add_subplot(0, 2, 1)
    # 预期抛出 ValueError 异常，因为子图编号必须是1到4之间的整数
    with pytest.raises(ValueError, match='num must be an integer with 1 <= num <= 4'):
        fig.add_subplot(2, 2, 0)
    # 预期抛出 ValueError 异常，因为子图编号必须是1到4之间的整数
    with pytest.raises(ValueError, match='num must be an integer with 1 <= num <= 4'):
        fig.add_subplot(2, 2, 5)
    # 预期抛出 ValueError 异常，因为子图编号必须是1到4之间的整数
    with pytest.raises(ValueError, match='num must be an integer with 1 <= num <= 4'):
        fig.add_subplot(2, 2, 0.5)

    # 预期抛出 ValueError 异常，因为子图编号必须是三位整数
    with pytest.raises(ValueError, match='must be a three-digit integer'):
        fig.add_subplot(42)
    # 预期抛出 ValueError 异常，因为子图编号必须是三位整数
    with pytest.raises(ValueError, match='must be a three-digit integer'):
        fig.add_subplot(1000)

    # 预期抛出 TypeError 异常，因为给定的参数个数与预期不符
    with pytest.raises(TypeError, match='takes 1 or 3 positional arguments but 2 were given'):
        fig.add_subplot(2, 2)
    # 预期抛出 TypeError 异常，因为给定的参数个数与预期不符
    with pytest.raises(TypeError, match='takes 1 or 3 positional arguments but 4 were given'):
        fig.add_subplot(1, 2, 3, 4)
    # 预期抛出 ValueError 异常，因为行数必须是正整数，而不是字符串'2'
    with pytest.raises(ValueError, match="Number of rows must be a positive integer, not '2'"):
        fig.add_subplot('2', 2, 1)
    # 预期抛出 ValueError 异常，因为列数必须是正整数，而不是浮点数2.0
    with pytest.raises(ValueError, match='Number of columns must be a positive integer, not 2.0'):
        fig.add_subplot(2, 2.0, 1)
    # 获取图形对象的坐标轴，然后预期抛出 ValueError 异常，因为尝试在不同的图形对象中添加坐标轴
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match='The Axes must have been created in the present figure'):
        fig.add_subplot(ax)


# 使用图像对比测试装饰器来定义一个测试函数，用于验证图形对象的超标题设置情况
@image_comparison(['figure_suptitle'])
def test_suptitle():
    # 创建一个新的图形对象和一个轴对象
    fig, _ = plt.subplots()
    # 设置图形对象的超标题文本和颜色
    fig.suptitle('hello', color='r')
    # 设置图形对象的超标题文本、颜色和旋转角度
    fig.suptitle('title', color='g', rotation=30)


# 定义一个测试函数，用于验证图形对象的超标题设置字体属性情况
def test_suptitle_fontproperties():
    # 创建一个新的图形对象和一个轴对象
    fig, ax = plt.subplots()
    # 创建一个自定义的字体属性对象
    fps = mpl.font_manager.FontProperties(size='large', weight='bold')
    # 设置图形对象的超标题文本和自定义的字体属性
    txt = fig.suptitle('fontprops title', fontproperties=fps)
    # 验证超标题的字体大小是否与自定义字体属性中的大小一致
    assert txt.get_fontsize() == fps.get_size_in_points()
    # 验证超标题的字体粗细是否与自定义字体属性中的粗细一致
    assert txt.get_weight() == fps.get_weight()


# 定义一个测试函数，用于验证在不同尺寸下创建图形对象的情况
def test_suptitle_subfigures():
    # 创建一个尺寸为4x3的新图形对象
    fig = plt.figure(figsize=(4, 3))
    # 将图形分割为 1 行 2 列，并分别赋值给 sf1 和 sf2
    sf1, sf2 = fig.subfigures(1, 2)
    # 设置第二个子图的背景色为白色
    sf2.set_facecolor('white')
    # 在第一个子图上创建子图
    sf1.subplots()
    # 在第二个子图上创建子图
    sf2.subplots()
    # 设置整个图形的总标题
    fig.suptitle("This is a visible suptitle.")

    # 验证第一个子图的背景色是否为默认的透明色
    assert sf1.get_facecolor() == (0.0, 0.0, 0.0, 0.0)
    # 验证第二个子图的背景色是否为白色
    assert sf2.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
def test_get_suptitle_supxlabel_supylabel():
    # 创建一个包含一个图形和轴对象的图形
    fig, ax = plt.subplots()
    # 断言图形的总标题为空字符串
    assert fig.get_suptitle() == ""
    # 断言图形的上方 X 轴标签为空字符串
    assert fig.get_supxlabel() == ""
    # 断言图形的右侧 Y 轴标签为空字符串
    assert fig.get_supylabel() == ""
    # 设置图形的总标题为 'suptitle'
    fig.suptitle('suptitle')
    # 断言图形的总标题为 'suptitle'
    assert fig.get_suptitle() == 'suptitle'
    # 设置图形的上方 X 轴标签为 'supxlabel'
    fig.supxlabel('supxlabel')
    # 断言图形的上方 X 轴标签为 'supxlabel'
    assert fig.get_supxlabel() == 'supxlabel'
    # 设置图形的右侧 Y 轴标签为 'supylabel'
    fig.supylabel('supylabel')
    # 断言图形的右侧 Y 轴标签为 'supylabel'
    assert fig.get_supylabel() == 'supylabel'


@image_comparison(['alpha_background'],
                  # 只测试 png 和 svg 格式。PDF 输出看起来是正确的，
                  # 但 Ghostscript 无法保留背景颜色。
                  extensions=['png', 'svg'],
                  savefig_kwarg={'facecolor': (0, 1, 0.4),
                                 'edgecolor': 'none'})
def test_alpha():
    # 创建一个尺寸为 [2, 1] 的图形
    fig = plt.figure(figsize=[2, 1])
    # 设置图形的背景颜色为 (0, 1, 0.4)
    fig.set_facecolor((0, 1, 0.4))
    # 设置图形的背景透明度为 0.4
    fig.patch.set_alpha(0.4)
    # 向图形添加一个半径为 15 的红色圆形，透明度为 0.6
    fig.patches.append(mpl.patches.CirclePolygon(
        [20, 20], radius=15, alpha=0.6, facecolor='red'))


def test_too_many_figures():
    # 通过 pytest.warns 捕获 RuntimeWarning
    with pytest.warns(RuntimeWarning):
        # 创建超过设定最大数量的图形，触发警告
        for i in range(mpl.rcParams['figure.max_open_warning'] + 1):
            plt.figure()


def test_iterability_axes_argument():

    # 这是对 matplotlib/matplotlib#3196 的回归测试。如果 _as_mpl_axes 返回的参数之一定义了 __getitem__，
    # 但不可迭代，这会引发异常。这是因为我们检查参数是否可迭代，如果是，我们尝试将其转换为元组。
    # 现在在 try...except 中进行元组转换以防失败。

    class MyAxes(Axes):
        def __init__(self, *args, myclass=None, **kwargs):
            Axes.__init__(self, *args, **kwargs)

    class MyClass:

        def __getitem__(self, item):
            if item != 'a':
                raise ValueError("item should be a")

        def _as_mpl_axes(self):
            return MyAxes, {'myclass': self}

    # 创建一个图形对象
    fig = plt.figure()
    # 向图形添加一个子图，使用 MyClass 的实例作为投影
    fig.add_subplot(1, 1, 1, projection=MyClass())
    # 关闭图形
    plt.close(fig)


def test_set_fig_size():
    # 创建一个图形对象
    fig = plt.figure()

    # 检查设置图形宽度
    fig.set_figwidth(5)
    assert fig.get_figwidth() == 5

    # 检查设置图形高度
    fig.set_figheight(1)
    assert fig.get_figheight() == 1

    # 检查使用 set_size_inches 方法设置图形尺寸
    fig.set_size_inches(2, 4)
    assert fig.get_figwidth() == 2
    assert fig.get_figheight() == 4

    # 检查使用元组作为第一个参数设置图形尺寸
    fig.set_size_inches((1, 3))
    assert fig.get_figwidth() == 1
    assert fig.get_figheight() == 3


def test_axes_remove():
    # 创建一个包含 2x2 子图的图形对象
    fig, axs = plt.subplots(2, 2)
    # 移除最后一个子图
    axs[-1, -1].remove()
    # 断言所有剩余的子图仍在图形的 axes 列表中
    for ax in axs.ravel()[:-1]:
        assert ax in fig.axes
    # 断言被移除的子图不在图形的 axes 列表中
    assert axs[-1, -1] not in fig.axes
    # 断言图形的 axes 列表长度为 3
    assert len(fig.axes) == 3


def test_figaspect():
    # 根据指定的宽高比例创建图形尺寸
    w, h = plt.figaspect(np.float64(2) / np.float64(1))
    # 断言确保高度与宽度比例为2:1
    assert h / w == 2
    
    # 根据指定的宽高比例创建图形尺寸
    w, h = plt.figaspect(2)
    # 断言确保高度与宽度比例为2:1
    assert h / w == 2
    
    # 根据指定的数组形状创建图形尺寸
    w, h = plt.figaspect(np.zeros((1, 2)))
    # 断言确保高度与宽度比例为0.5:1
    assert h / w == 0.5
    
    # 根据指定的数组形状创建图形尺寸
    w, h = plt.figaspect(np.zeros((2, 2)))
    # 断言确保高度与宽度比例为1:1
    assert h / w == 1
# 使用 pytest.mark.parametrize 装饰器标记 test_autofmt_xdate 函数，参数 which 可以取 'both'、'major' 或 'minor'
@pytest.mark.parametrize('which', ['both', 'major', 'minor'])
def test_autofmt_xdate(which):
    # 创建日期和时间的列表
    date = ['3 Jan 2013', '4 Jan 2013', '5 Jan 2013', '6 Jan 2013',
            '7 Jan 2013', '8 Jan 2013', '9 Jan 2013', '10 Jan 2013',
            '11 Jan 2013', '12 Jan 2013', '13 Jan 2013', '14 Jan 2013']

    time = ['16:44:00', '16:45:00', '16:46:00', '16:47:00', '16:48:00',
            '16:49:00', '16:51:00', '16:52:00', '16:53:00', '16:55:00',
            '16:56:00', '16:57:00']

    angle = 60
    minors = [1, 2, 3, 4, 5, 6, 7]

    # 将日期和时间转换为数值
    x = mdates.datestr2num(date)
    y = mdates.datestr2num(time)

    # 创建图形和轴对象
    fig, ax = plt.subplots()

    # 在轴上绘制图形
    ax.plot(x, y)
    # 设置 y 轴为日期格式
    ax.yaxis_date()
    # 设置 x 轴为日期格式
    ax.xaxis_date()

    # 设置 x 轴次要刻度的定位器为自动次要定位器
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    # 忽略警告："FixedFormatter 应仅与 FixedLocator 一起使用"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'FixedFormatter should only be used together with FixedLocator')
        # 设置 x 轴次要刻度的格式化器为固定格式化器
        ax.xaxis.set_minor_formatter(FixedFormatter(minors))

    # 自动格式化 x 轴日期标签
    fig.autofmt_xdate(0.2, angle, 'right', which)

    # 如果 which 是 'both' 或 'major'，验证主要刻度的标签旋转角度是否为 angle
    if which in ('both', 'major'):
        for label in fig.axes[0].get_xticklabels(False, 'major'):
            assert int(label.get_rotation()) == angle

    # 如果 which 是 'both' 或 'minor'，验证次要刻度的标签旋转角度是否为 angle
    if which in ('both', 'minor'):
        for label in fig.axes[0].get_xticklabels(True, 'minor'):
            assert int(label.get_rotation()) == angle


# 使用 mpl.style.context 装饰器标记 test_change_dpi 函数，将当前风格设置为 'default'
@mpl.style.context('default')
def test_change_dpi():
    # 创建图形，设置大小为 (4, 4)
    fig = plt.figure(figsize=(4, 4))
    # 绘制图形但不渲染
    fig.draw_without_rendering()
    # 断言图形 canvas 渲染器的高度和宽度为 400
    assert fig.canvas.renderer.height == 400
    assert fig.canvas.renderer.width == 400
    # 修改图形的 DPI 为 50
    fig.dpi = 50
    # 再次绘制图形但不渲染
    fig.draw_without_rendering()
    # 断言图形 canvas 渲染器的高度和宽度为 200
    assert fig.canvas.renderer.height == 200
    assert fig.canvas.renderer.width == 200


# 使用 pytest.mark.parametrize 装饰器标记 test_invalid_figure_size 函数，参数 width 和 height 可以取不同的值
@pytest.mark.parametrize('width, height', [
    (1, np.nan),   # 宽度为 1，高度为 NaN，预期引发 ValueError 异常
    (-1, 1),       # 宽度为 -1，高度为 1，预期引发 ValueError 异常
    (np.inf, 1)    # 宽度为正无穷大，高度为 1，预期引发 ValueError 异常
])
def test_invalid_figure_size(width, height):
    # 使用 pyplot 创建大小为 (width, height) 的图形，预期引发 ValueError 异常
    with pytest.raises(ValueError):
        plt.figure(figsize=(width, height))

    # 创建一个图形对象
    fig = plt.figure()
    # 使用 set_size_inches 方法设置图形的尺寸为 (width, height)，预期引发 ValueError 异常
    with pytest.raises(ValueError):
        fig.set_size_inches(width, height)


# 测试无效的图形添加轴操作
def test_invalid_figure_add_axes():
    # 创建一个图形对象
    fig = plt.figure()
    # 使用 add_axes 方法不传递任何参数，预期引发 TypeError 异常，错误信息包含缺少 'rect' 位置参数
    with pytest.raises(TypeError,
                       match="missing 1 required positional argument: 'rect'"):
        fig.add_axes()

    # 使用 add_axes 方法传递一个包含 NaN 的矩形参数，预期引发 ValueError 异常
    with pytest.raises(ValueError):
        fig.add_axes((.1, .1, .5, np.nan))

    # 使用 add_axes 方法同时传递位置参数和关键字参数 rect，预期引发 TypeError 异常，错误信息包含多次指定 'rect' 参数
    with pytest.raises(TypeError, match="multiple values for argument 'rect'"):
        fig.add_axes([0, 0, 1, 1], rect=[0, 0, 1, 1])

    # 创建另一个图形对象和轴对象
    fig2, ax = plt.subplots()
    # 使用 add_axes 方法尝试向另一个图形对象中添加现有轴对象，预期引发 ValueError 异常，错误信息指示轴必须在当前图形中创建
    with pytest.raises(ValueError,
                       match="The Axes must have been created in the present "
                             "figure"):
        fig.add_axes(ax)

    # 删除 fig2 中的轴对象 ax
    fig2.delaxes(ax)
    # 使用 add_axes 方法向 fig2 添加轴对象 ax，并传递额外的位置参数，预期发出 MatplotlibDeprecationWarning 警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match="Passing more than one positional argument"):
        fig2.add_axes(ax, "extra positional argument")
    # 使用 pytest 库中的 warns 上下文管理器，捕获 matplotlib 中的 MatplotlibDeprecationWarning 警告，
    # 并匹配警告消息中包含 "Passing more than one positional argument" 的警告信息
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match="Passing more than one positional argument"):
        # 向图形对象 fig 添加一个坐标轴，使用坐标轴参数 [0, 0, 1, 1]，
        # 同时传递了额外的位置参数 "extra positional argument"，此处为触发警告的操作
        fig.add_axes([0, 0, 1, 1], "extra positional argument")
def test_subplots_shareax_loglabels():
    # 创建一个包含4个子图的图形对象，子图共享 x 和 y 轴，不压缩子图布局
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False)
    # 在每个子图中绘制一条对数坐标轴的直线图
    for ax in axs.flat:
        ax.plot([10, 20, 30], [10, 20, 30])

    # 设置所有子图的 y 轴为对数坐标轴
    ax.set_yscale("log")
    # 设置所有子图的 x 轴为对数坐标轴
    ax.set_xscale("log")

    # 检查第一行子图的 x 轴刻度标签数是否为0
    for ax in axs[0, :]:
        assert 0 == len(ax.xaxis.get_ticklabels(which='both'))

    # 检查第二行子图的 x 轴刻度标签数是否大于0
    for ax in axs[1, :]:
        assert 0 < len(ax.xaxis.get_ticklabels(which='both'))

    # 检查第二列子图的 y 轴刻度标签数是否为0
    for ax in axs[:, 1]:
        assert 0 == len(ax.yaxis.get_ticklabels(which='both'))

    # 检查第一列子图的 y 轴刻度标签数是否大于0
    for ax in axs[:, 0]:
        assert 0 < len(ax.yaxis.get_ticklabels(which='both'))


def test_savefig():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 定义异常消息
    msg = r"savefig\(\) takes 2 positional arguments but 3 were given"
    # 检查是否引发了预期的 TypeError 异常，并且异常消息匹配
    with pytest.raises(TypeError, match=msg):
        fig.savefig("fname1.png", "fname2.png")


def test_savefig_warns():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 针对多种文件格式进行测试
    for format in ['png', 'pdf', 'svg', 'tif', 'jpg']:
        # 检查是否引发了预期的 TypeError 异常
        with pytest.raises(TypeError):
            fig.savefig(io.BytesIO(), format=format, non_existent_kwarg=True)


def test_savefig_backend():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 故意使用一个不存在的模块名来测试异常处理
    with pytest.raises(ModuleNotFoundError, match="No module named '@absent'"):
        fig.savefig("test", backend="module://@absent")
    # 检查是否引发了预期的 ValueError 异常，并且异常消息匹配
    with pytest.raises(ValueError,
                       match="The 'pdf' backend does not support png output"):
        fig.savefig("test.png", backend="pdf")


@pytest.mark.parametrize('backend', [
    pytest.param('Agg', marks=[pytest.mark.backend('Agg')]),
    pytest.param('Cairo', marks=[pytest.mark.backend('Cairo')]),
])
def test_savefig_pixel_ratio(backend):
    # 创建一个包含一个子图的图形对象
    fig, ax = plt.subplots()
    # 在字节流中保存图像，并加载图像数据以获取像素比例
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        ratio1 = Image.open(buf)
        ratio1.load()

    # 创建一个包含一个子图的新图形对象
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    # 设置设备像素比例为2
    fig.canvas._set_device_pixel_ratio(2)
    # 在字节流中保存图像，并加载图像数据以获取像素比例
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        ratio2 = Image.open(buf)
        ratio2.load()

    # 检查两个图像的像素比例是否相等
    assert ratio1 == ratio2


def test_savefig_preserve_layout_engine():
    # 创建一个使用压缩布局的新图形对象
    fig = plt.figure(layout='compressed')
    # 在字节流中保存图像，并调整边界框以适应子图
    fig.savefig(io.BytesIO(), bbox_inches='tight')

    # 检查图形对象的布局引擎是否为压缩布局
    assert fig.get_layout_engine()._compress


def test_savefig_locate_colorbar():
    # 创建一个包含一个子图的图形对象
    fig, ax = plt.subplots()
    # 在子图中绘制一个伪彩色网格
    pc = ax.pcolormesh(np.random.randn(2, 2))
    # 在图形对象中添加一个颜色条，并保存图像以适应给定的边界框
    cbar = fig.colorbar(pc, aspect=40)
    fig.savefig(io.BytesIO(), bbox_inches=mpl.transforms.Bbox([[0, 0], [4, 4]]))

    # 检查是否已应用指定的长宽比
    assert (cbar.ax.get_position(original=True).bounds !=
            cbar.ax.get_position(original=False).bounds)


@mpl.rc_context({"savefig.transparent": True})
@check_figures_equal(extensions=["png"])
def test_savefig_transparent(fig_test, fig_ref):
    # 创建两个包含透明插图的透明子图
    # 整个图像的背景应该是透明的
    gs1 = fig_test.add_gridspec(3, 3, left=0.05, wspace=0.05)
    # 添加一个子图到 fig_test 图形对象中，占据整个网格
    f1 = fig_test.add_subfigure(gs1[:, :])
    
    # 在 f1 子图中添加一个子图 f2，占据 gs1 网格的第一行第一列位置
    f2 = f1.add_subfigure(gs1[0, 0])
    
    # 在 f2 子图中添加一个子图 ax12，占据 gs1 网格的全部区域
    ax12 = f2.add_subplot(gs1[:, :])
    
    # 在 f1 子图中添加一个子图 ax1，占据 gs1 网格除最后一行以外的所有行
    ax1 = f1.add_subplot(gs1[:-1, :])
    
    # 在 ax1 子图中插入一个轴 iax1，位置在相对于 ax1 左上角的偏移（0.1, 0.2），大小为 ax1 的 30% x 40%
    iax1 = ax1.inset_axes([.1, .2, .3, .4])
    
    # 在 iax1 子图中插入一个轴 iax2，位置在相对于 iax1 左上角的偏移（0.1, 0.2），大小为 iax1 的 30% x 40%
    iax2 = iax1.inset_axes([.1, .2, .3, .4])
    
    # 添加一个子图 ax2 到 fig_test 图形对象中，占据 gs1 网格的倒数第一行除最后一列以外的所有列
    ax2 = fig_test.add_subplot(gs1[-1, :-1])
    
    # 添加一个子图 ax3 到 fig_test 图形对象中，占据 gs1 网格的倒数第一行最后一列位置
    ax3 = fig_test.add_subplot(gs1[-1, -1])
    
    # 遍历以下子图列表，并设置它们的 x 轴和 y 轴刻度为空，隐藏所有边框
    for ax in [ax12, ax1, iax1, iax2, ax2, ax3]:
        ax.set(xticks=[], yticks=[])
        ax.spines[:].set_visible(False)
# 测试函数，验证图表对象的字符串表示形式是否符合预期
def test_figure_repr():
    # 创建一个大小为 100x200 像素、分辨率为 10 的图表对象
    fig = plt.figure(figsize=(10, 20), dpi=10)
    # 断言图表对象的字符串表示是否为 "<Figure size 100x200 with 0 Axes>"
    assert repr(fig) == "<Figure size 100x200 with 0 Axes>"


# 测试不同布局参数下图表对象的布局属性
def test_valid_layouts():
    # 创建一个没有布局的图表对象
    fig = Figure(layout=None)
    # 断言图表对象的紧凑布局不可用
    assert not fig.get_tight_layout()
    # 断言图表对象的约束布局不可用
    assert not fig.get_constrained_layout()

    # 创建一个紧凑布局的图表对象
    fig = Figure(layout='tight')
    # 断言图表对象的紧凑布局可用
    assert fig.get_tight_layout()
    # 断言图表对象的约束布局不可用
    assert not fig.get_constrained_layout()

    # 创建一个约束布局的图表对象
    fig = Figure(layout='constrained')
    # 断言图表对象的紧凑布局不可用
    assert not fig.get_tight_layout()
    # 断言图表对象的约束布局可用
    assert fig.get_constrained_layout()


# 测试图表对象的无效布局参数处理
def test_invalid_layouts():
    # 创建一个约束布局的图表对象和对应的轴对象
    fig, ax = plt.subplots(layout="constrained")
    # 使用 pytest 捕获 UserWarning
    with pytest.warns(UserWarning):
        # 调整图表的子图以接近顶部的距离
        fig.subplots_adjust(top=0.8)
    # 断言图表对象的布局引擎为 ConstrainedLayoutEngine 类型
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)

    # 测试同时使用 layout 和 (tight|constrained)_layout 参数时的警告
    wst = "The Figure parameters 'layout' and 'tight_layout'"
    with pytest.warns(UserWarning, match=wst):
        fig = Figure(layout='tight', tight_layout=False)
    # 断言图表对象的布局引擎为 TightLayoutEngine 类型
    assert isinstance(fig.get_layout_engine(), TightLayoutEngine)
    wst = "The Figure parameters 'layout' and 'constrained_layout'"
    with pytest.warns(UserWarning, match=wst):
        fig = Figure(layout='constrained', constrained_layout=False)
    # 断言图表对象的布局引擎不为 TightLayoutEngine 类型，而是 ConstrainedLayoutEngine 类型
    assert not isinstance(fig.get_layout_engine(), TightLayoutEngine)
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)

    # 使用无效的布局参数 'foobar' 创建图表对象，测试是否会抛出 ValueError 异常
    with pytest.raises(ValueError,
                       match="Invalid value for 'layout'"):
        Figure(layout='foobar')

    # 测试在没有颜色条的情况下，是否可以交换布局类型
    fig, ax = plt.subplots(layout="constrained")
    fig.set_layout_engine("tight")
    # 断言图表对象的布局引擎为 TightLayoutEngine 类型
    assert isinstance(fig.get_layout_engine(), TightLayoutEngine)
    fig.set_layout_engine("constrained")
    # 断言图表对象的布局引擎为 ConstrainedLayoutEngine 类型
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)

    # 测试如果图表中存在颜色条，则不可交换布局类型
    fig, ax = plt.subplots(layout="constrained")
    pc = ax.pcolormesh(np.random.randn(2, 2))
    fig.colorbar(pc)
    with pytest.raises(RuntimeError, match='Colorbar layout of new layout'):
        fig.set_layout_engine("tight")
    fig.set_layout_engine("none")
    with pytest.raises(RuntimeError, match='Colorbar layout of new layout'):
        fig.set_layout_engine("tight")

    fig, ax = plt.subplots(layout="tight")
    pc = ax.pcolormesh(np.random.randn(2, 2))
    fig.colorbar(pc)
    with pytest.raises(RuntimeError, match='Colorbar layout of new layout'):
        fig.set_layout_engine("constrained")
    fig.set_layout_engine("none")
    # 断言图表对象的布局引擎为 PlaceHolderLayoutEngine 类型
    assert isinstance(fig.get_layout_engine(), PlaceHolderLayoutEngine)

    with pytest.raises(RuntimeError, match='Colorbar layout of new layout'):
        fig.set_layout_engine("constrained")
    # 使用 zip 函数同时迭代 fig_ref 和 fig_test 列表，autolayout 分别为 False 和 True
    for fig, autolayout in zip([fig_ref, fig_test], [False, True]):
        # 创建 matplotlib 的上下文管理器，设置 figure.autolayout 的值为 autolayout
        with mpl.rc_context({'figure.autolayout': autolayout}):
            # 创建包含两个子图的 Figure 对象，并返回子图的 axes 数组
            axes = fig.subplots(ncols=2)
            # 调整子图的布局，设置水平间距为 10
            fig.tight_layout(w_pad=10)
        # 断言当前 Figure 对象使用的布局引擎为 PlaceHolderLayoutEngine 的实例
        assert isinstance(fig.get_layout_engine(), PlaceHolderLayoutEngine)
@pytest.mark.parametrize('layout', ['constrained', 'compressed'])
# 使用pytest的参数化装饰器，指定layout参数为'constrained'和'compressed'两种取值
def test_layout_change_warning(layout):
    """
    Raise a warning when a previously assigned layout changes to tight using
    plt.tight_layout().
    """
    # 创建一个新的图形和轴，使用指定的layout参数
    fig, ax = plt.subplots(layout=layout)
    # 使用pytest的warns上下文管理器来检查是否引发了UserWarning，且匹配特定字符串
    with pytest.warns(UserWarning, match='The figure layout has changed to'):
        # 调用plt.tight_layout()方法，期望引发警告
        plt.tight_layout()


def test_repeated_tightlayout():
    # 创建一个Figure对象
    fig = Figure()
    # 第一次调用tight_layout方法
    fig.tight_layout()
    # 连续的调用不应该引发警告
    fig.tight_layout()
    fig.tight_layout()


@check_figures_equal(extensions=["png", "pdf"])
# 使用自定义的装饰器check_figures_equal，指定支持的文件扩展名为["png", "pdf"]
def test_add_artist(fig_test, fig_ref):
    fig_test.dpi = 100
    fig_ref.dpi = 100

    # 在测试图形上创建子图
    fig_test.subplots()
    # 创建多个Line2D和Circle对象，并添加到测试图形中
    l1 = plt.Line2D([.2, .7], [.7, .7], gid='l1')
    l2 = plt.Line2D([.2, .7], [.8, .8], gid='l2')
    r1 = plt.Circle((20, 20), 100, transform=None, gid='C1')
    r2 = plt.Circle((.7, .5), .05, gid='C2')
    r3 = plt.Circle((4.5, .8), .55, transform=fig_test.dpi_scale_trans,
                    facecolor='crimson', gid='C3')
    for a in [l1, l2, r1, r2, r3]:
        # 将每个艺术家对象添加到测试图形中
        fig_test.add_artist(a)
    # 从图形中移除l2对象
    l2.remove()

    # 在参考图形上创建子图
    ax2 = fig_ref.subplots()
    # 创建多个Line2D和Circle对象，并添加到参考图形的特定子图中
    l1 = plt.Line2D([.2, .7], [.7, .7], transform=fig_ref.transFigure,
                    gid='l1', zorder=21)
    r1 = plt.Circle((20, 20), 100, transform=None, clip_on=False, zorder=20,
                    gid='C1')
    r2 = plt.Circle((.7, .5), .05, transform=fig_ref.transFigure, gid='C2',
                    zorder=20)
    r3 = plt.Circle((4.5, .8), .55, transform=fig_ref.dpi_scale_trans,
                    facecolor='crimson', clip_on=False, zorder=20, gid='C3')
    for a in [l1, r1, r2, r3]:
        # 将每个艺术家对象添加到参考图形的特定子图中
        ax2.add_artist(a)


@pytest.mark.parametrize("fmt", ["png", "pdf", "ps", "eps", "svg"])
# 使用pytest的参数化装饰器，指定fmt参数为["png", "pdf", "ps", "eps", "svg"]中的一种格式
def test_fspath(fmt, tmp_path):
    # 设置输出路径为临时目录下的特定文件名
    out = tmp_path / f"test.{fmt}"
    # 保存图形为指定格式的文件
    plt.savefig(out)
    with out.open("rb") as file:
        # 检查文件的前100字节中是否包含指定格式的名称（不区分大小写）
        assert fmt.encode("ascii") in file.read(100).lower()


def test_tightbbox():
    # 创建一个新的图形和轴
    fig, ax = plt.subplots()
    # 设置轴的x轴限制
    ax.set_xlim(0, 1)
    # 在轴上添加文本对象
    t = ax.text(1., 0.5, 'This dangles over end')
    # 获取图形的渲染器对象
    renderer = fig.canvas.get_renderer()
    x1Nom0 = 9.035  # inches
    # 检查文本对象的紧凑边界框的右侧位置
    assert abs(t.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    # 检查轴的紧凑边界框的右侧位置
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    # 检查图形的紧凑边界框的右侧位置
    assert abs(fig.get_tightbbox(renderer).x1 - x1Nom0) < 0.05
    # 暂时排除文本对象t，重新计算紧凑边界框，检查右侧位置
    t.set_in_layout(False)
    x1Nom = 7.333
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom * fig.dpi) < 2
    assert abs(fig.get_tightbbox(renderer).x1 - x1Nom) < 0.05

    # 恢复文本对象t到布局中，重新计算紧凑边界框，检查右侧位置
    t.set_in_layout(True)
    x1Nom = 7.333
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    # 测试bbox_extra_artists方法...
    # 使用assert语句验证以下条件：
    # 确保通过ax对象获取的紧凑边界框的右侧边界x1与期望值x1Nom * fig.dpi之间的差异绝对值小于2
    assert abs(ax.get_tightbbox(renderer, bbox_extra_artists=[]).x1
               - x1Nom * fig.dpi) < 2
def test_axes_removal():
    # 检查在移除 Axes 后是否能设置格式化程序
    # 创建一个包含两个子图的图形对象
    fig, axs = plt.subplots(1, 2, sharex=True)
    # 移除第二个子图
    axs[1].remove()
    # 在第一个子图上绘制日期时间序列
    axs[0].plot([datetime(2000, 1, 1), datetime(2000, 2, 1)], [0, 1])
    # 断言第一个子图的 x 轴主要格式化程序是否为自动日期格式化程序
    assert isinstance(axs[0].xaxis.get_major_formatter(),
                      mdates.AutoDateFormatter)

    # 检查手动设置格式化程序后移除 Axes 是否保留了设置的格式化程序
    fig, axs = plt.subplots(1, 2, sharex=True)
    # 在第二个子图的 x 轴上设置主要格式化程序为标量格式化程序
    axs[1].xaxis.set_major_formatter(ScalarFormatter())
    # 移除第二个子图
    axs[1].remove()
    # 在第一个子图上绘制日期时间序列
    axs[0].plot([datetime(2000, 1, 1), datetime(2000, 2, 1)], [0, 1])
    # 断言第一个子图的 x 轴主要格式化程序是否为标量格式化程序
    assert isinstance(axs[0].xaxis.get_major_formatter(),
                      ScalarFormatter)


def test_removed_axis():
    # 简单的功能测试，确保移除共享轴能正常工作
    # 创建一个包含两个子图的图形对象，并共享 x 轴
    fig, axs = plt.subplots(2, sharex=True)
    # 移除第一个子图
    axs[0].remove()
    # 绘制图形
    fig.canvas.draw()


@pytest.mark.parametrize('clear_meth', ['clear', 'clf'])
def test_figure_clear(clear_meth):
    # 测试图形清除的多种场景：
    fig = plt.figure()

    # a) 空图形
    fig.clear()
    assert fig.axes == []

    # b) 包含单个非嵌套子图的图形
    ax = fig.add_subplot(111)
    getattr(fig, clear_meth)()
    assert fig.axes == []

    # c) 包含多个非嵌套子图的图形
    axes = [fig.add_subplot(2, 1, i+1) for i in range(2)]
    getattr(fig, clear_meth)()
    assert fig.axes == []

    # d) 包含子图的图形
    gs = fig.add_gridspec(ncols=2, nrows=1)
    subfig = fig.add_subfigure(gs[0])
    subaxes = subfig.add_subplot(111)
    getattr(fig, clear_meth)()
    assert subfig not in fig.subfigs
    assert fig.axes == []

    # e) 包含子图和子图的图形
    subfig = fig.add_subfigure(gs[0])
    subaxes = subfig.add_subplot(111)
    mainaxes = fig.add_subplot(gs[1])

    # e.1) 仅移除子图保留子图
    mainaxes.remove()
    assert fig.axes == [subaxes]

    # e.2) 仅移除子图的子图保留子图和子图
    mainaxes = fig.add_subplot(gs[1])
    subaxes.remove()
    assert fig.axes == [mainaxes]
    assert subfig in fig.subfigs

    # e.3) 清除子图保留子图
    subaxes = subfig.add_subplot(111)
    assert mainaxes in fig.axes
    assert subaxes in fig.axes
    getattr(subfig, clear_meth)()
    assert subfig in fig.subfigs
    assert subaxes not in subfig.axes
    assert subaxes not in fig.axes
    assert mainaxes in fig.axes

    # e.4) 清除整个图形
    subaxes = subfig.add_subplot(111)
    getattr(fig, clear_meth)()
    assert fig.axes == []
    assert fig.subfigs == []

    # f) 多个子图
    subfigs = [fig.add_subfigure(gs[i]) for i in [0, 1]]
    subaxes = [sfig.add_subplot(111) for sfig in subfigs]
    assert all(ax in fig.axes for ax in subaxes)
    assert all(sfig in fig.subfigs for sfig in subfigs)

    # f.1) 仅清除一个子图
    getattr(subfigs[0], clear_meth)()
    # 检查 subaxes[0] 是否不在 fig.axes 中
    assert subaxes[0] not in fig.axes
    # 检查 subaxes[1] 是否在 fig.axes 中
    assert subaxes[1] in fig.axes
    # 检查 subfigs[1] 是否在 fig.subfigs 中
    assert subfigs[1] in fig.subfigs

    # f.2) 清空整个图形
    # 获取 subfigs[1] 的清空方法并执行
    getattr(subfigs[1], clear_meth)()
    # 使用 subfigs[1] 的格子布局创建两个子图
    subfigs = [fig.add_subfigure(gs[i]) for i in [0, 1]]
    # 分别为两个子图创建坐标轴
    subaxes = [sfig.add_subplot(111) for sfig in subfigs]
    # 检查所有 subaxes 中的轴是否都在 fig.axes 中
    assert all(ax in fig.axes for ax in subaxes)
    # 检查所有 subfigs 中的子图是否都在 fig.subfigs 中
    assert all(sfig in fig.subfigs for sfig in subfigs)
    # 获取 fig 的清空方法并执行
    getattr(fig, clear_meth)()
    # 检查 fig.subfigs 是否为空列表
    assert fig.subfigs == []
    # 检查 fig.axes 是否为空列表
    assert fig.axes == []
def test_clf_not_redefined():
    # 遍历所有继承自FigureBase的子类
    for klass in FigureBase.__subclasses__():
        # 断言在每个子类的__dict__中不存在名为'clf'的属性或方法
        assert 'clf' not in klass.__dict__


@mpl.style.context('mpl20')
def test_picking_does_not_stale():
    # 使用mpl20风格的上下文环境进行测试
    fig, ax = plt.subplots()
    # 在图形上创建一个散点图，并启用选取功能
    ax.scatter([0], [0], [1000], picker=True)
    # 绘制图形的画布
    fig.canvas.draw()
    # 断言图形不处于陈旧状态
    assert not fig.stale

    # 创建一个模拟的鼠标事件
    mouse_event = SimpleNamespace(x=ax.bbox.x0 + ax.bbox.width / 2,
                                  y=ax.bbox.y0 + ax.bbox.height / 2,
                                  inaxes=ax, guiEvent=None)
    # 模拟鼠标点击事件
    fig.pick(mouse_event)
    # 断言图形不处于陈旧状态
    assert not fig.stale


def test_add_subplot_twotuple():
    fig = plt.figure()
    # 添加一个跨度为3行2列的子图，位置为(3, 5)
    ax1 = fig.add_subplot(3, 2, (3, 5))
    # 断言第一个子图的行跨度为1到2行
    assert ax1.get_subplotspec().rowspan == range(1, 3)
    # 断言第一个子图的列跨度为0到1列
    assert ax1.get_subplotspec().colspan == range(0, 1)
    
    # 添加一个跨度为3行2列的子图，位置为(4, 6)
    ax2 = fig.add_subplot(3, 2, (4, 6))
    # 断言第二个子图的行跨度为1到2行
    assert ax2.get_subplotspec().rowspan == range(1, 3)
    # 断言第二个子图的列跨度为1到2列
    assert ax2.get_subplotspec().colspan == range(1, 2)
    
    # 添加一个跨度为3行2列的子图，位置为(3, 6)
    ax3 = fig.add_subplot(3, 2, (3, 6))
    # 断言第三个子图的行跨度为1到2行
    assert ax3.get_subplotspec().rowspan == range(1, 3)
    # 断言第三个子图的列跨度为0到2列
    assert ax3.get_subplotspec().colspan == range(0, 2)
    
    # 添加一个跨度为3行2列的子图，位置为(4, 5)
    ax4 = fig.add_subplot(3, 2, (4, 5))
    # 断言第四个子图的行跨度为1到2行
    assert ax4.get_subplotspec().rowspan == range(1, 3)
    # 断言第四个子图的列跨度为0到2列
    assert ax4.get_subplotspec().colspan == range(0, 2)
    
    # 使用pytest断言预期引发IndexError异常
    with pytest.raises(IndexError):
        fig.add_subplot(3, 2, (6, 3))


@image_comparison(['tightbbox_box_aspect.svg'], style='mpl20',
                  savefig_kwarg={'bbox_inches': 'tight',
                                 'facecolor': 'teal'},
                  remove_text=True)
def test_tightbbox_box_aspect():
    fig = plt.figure()
    # 添加一个包含2列的网格布局
    gs = fig.add_gridspec(1, 2)
    # 在网格的第一格添加子图
    ax1 = fig.add_subplot(gs[0, 0])
    # 在网格的第二格添加子图，并指定为3D投影
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    # 设置第一个子图的盒子比例为0.5
    ax1.set_box_aspect(.5)
    # 设置第二个子图的盒子比例为(2, 1, 1)
    ax2.set_box_aspect((2, 1, 1))


@check_figures_equal(extensions=["svg", "pdf", "eps", "png"])
def test_animated_with_canvas_change(fig_test, fig_ref):
    # 创建参考图形的子图
    ax_ref = fig_ref.subplots()
    ax_ref.plot(range(5))

    # 创建测试图形的子图，并指定为动画模式
    ax_test = fig_test.subplots()
    ax_test.plot(range(5), animated=True)


class TestSubplotMosaic:
    @check_figures_equal(extensions=["png"])
    @pytest.mark.parametrize(
        "x", [
            [["A", "A", "B"], ["C", "D", "B"]],
            [[1, 1, 2], [3, 4, 2]],
            (("A", "A", "B"), ("C", "D", "B")),
            ((1, 1, 2), (3, 4, 2))
        ]
    )
    def test_basic(self, fig_test, fig_ref, x):
        # 使用给定的布局x创建子图网格
        grid_axes = fig_test.subplot_mosaic(x)

        # 为每个子图设置标题
        for k, ax in grid_axes.items():
            ax.set_title(k)

        # 获取x中唯一的标签，并断言其数量与子图网格的数量相等
        labels = sorted(np.unique(x))
        assert len(labels) == len(grid_axes)

        # 在参考图形中添加一个2x3的网格布局
        gs = fig_ref.add_gridspec(2, 3)
        axA = fig_ref.add_subplot(gs[:1, :2])
        axA.set_title(labels[0])

        axB = fig_ref.add_subplot(gs[:, 2])
        axB.set_title(labels[1])

        axC = fig_ref.add_subplot(gs[1, 0])
        axC.set_title(labels[2])

        axD = fig_ref.add_subplot(gs[1, 1])
        axD.set_title(labels[3])
    @check_figures_equal(extensions=["png"])
    # 使用装饰器检查两个图形是否相等，比较的文件扩展名为 PNG
    def test_all_nested(self, fig_test, fig_ref):
        x = [["A", "B"], ["C", "D"]]
        y = [["E", "F"], ["G", "H"]]

        # 设置参考图形和测试图形的布局引擎为 constrained
        fig_ref.set_layout_engine("constrained")
        fig_test.set_layout_engine("constrained")

        # 在测试图形上创建子图网格，并返回包含所有子图的字典 grid_axes
        grid_axes = fig_test.subplot_mosaic([[x, y]])
        for ax in grid_axes.values():
            # 设置每个子图的标题为其标签
            ax.set_title(ax.get_label())

        # 在参考图形上添加一个 1x2 的子图网格对象 gs
        gs = fig_ref.add_gridspec(1, 2)

        # 在 gs 的左侧子图网格对象 gs_left 中添加子图
        gs_left = gs[0, 0].subgridspec(2, 2)
        for j, r in enumerate(x):
            for k, label in enumerate(r):
                # 在 gs_left 的每个位置添加子图，并设置标题为标签
                fig_ref.add_subplot(gs_left[j, k]).set_title(label)

        # 在 gs 的右侧子图网格对象 gs_right 中添加子图
        gs_right = gs[0, 1].subgridspec(2, 2)
        for j, r in enumerate(y):
            for k, label in enumerate(r):
                # 在 gs_right 的每个位置添加子图，并设置标题为标签
                fig_ref.add_subplot(gs_right[j, k]).set_title(label)

    @check_figures_equal(extensions=["png"])
    # 使用装饰器检查两个图形是否相等，比较的文件扩展名为 PNG
    def test_nested(self, fig_test, fig_ref):

        # 设置参考图形和测试图形的布局引擎为 constrained
        fig_ref.set_layout_engine("constrained")
        fig_test.set_layout_engine("constrained")

        x = [["A", "B"], ["C", "D"]]

        # 定义一个嵌套列表 y，包含一个单独的 "F" 和之前定义的 x
        y = [["F"], [x]]

        # 在测试图形上创建子图网格，并返回包含所有子图的字典 grid_axes
        grid_axes = fig_test.subplot_mosaic(y)

        # 设置每个子图的标题为其键值
        for k, ax in grid_axes.items():
            ax.set_title(k)

        # 在参考图形上添加一个 2x1 的子图网格对象 gs
        gs = fig_ref.add_gridspec(2, 1)

        # 在 gs 的右下角 (1, 0) 处添加一个子图网格对象 gs_n
        gs_n = gs[1, 0].subgridspec(2, 2)

        # 添加子图 axA 到 gs_n 的左上角 (0, 0) 处，并设置标题为 "A"
        axA = fig_ref.add_subplot(gs_n[0, 0])
        axA.set_title("A")

        # 添加子图 axB 到 gs_n 的右上角 (0, 1) 处，并设置标题为 "B"
        axB = fig_ref.add_subplot(gs_n[0, 1])
        axB.set_title("B")

        # 添加子图 axC 到 gs_n 的左下角 (1, 0) 处，并设置标题为 "C"
        axC = fig_ref.add_subplot(gs_n[1, 0])
        axC.set_title("C")

        # 添加子图 axD 到 gs_n 的右下角 (1, 1) 处，并设置标题为 "D"
        axD = fig_ref.add_subplot(gs_n[1, 1])
        axD.set_title("D")

        # 在 gs 的左上角 (0, 0) 处添加一个子图 axF，并设置标题为 "F"
        axF = fig_ref.add_subplot(gs[0, 0])
        axF.set_title("F")

    @check_figures_equal(extensions=["png"])
    # 使用装饰器检查两个图形是否相等，比较的文件扩展名为 PNG
    def test_nested_tuple(self, fig_test, fig_ref):
        x = [["A", "B", "B"], ["C", "C", "D"]]
        xt = (("A", "B", "B"), ("C", "C", "D"))

        # 在参考图形和测试图形上创建子图网格，分别使用 x 和 xt
        fig_ref.subplot_mosaic([["F"], [x]])
        fig_test.subplot_mosaic([["F"], [xt]])

    def test_nested_width_ratios(self):
        x = [["A", [["B"],
                    ["C"]]]]
        width_ratios = [2, 1]

        # 使用 plt.subplot_mosaic 创建图形 fig 和包含所有子图的字典 axd
        fig, axd = plt.subplot_mosaic(x, width_ratios=width_ratios)

        # 断言子图 "A" 的宽度比例是否与给定的 width_ratios 相同
        assert axd["A"].get_gridspec().get_width_ratios() == width_ratios
        # 断言子图 "B" 的宽度比例是否与给定的 width_ratios 不同
        assert axd["B"].get_gridspec().get_width_ratios() != width_ratios

    def test_nested_height_ratios(self):
        x = [["A", [["B"],
                    ["C"]]], ["D", "D"]]
        height_ratios = [1, 2]

        # 使用 plt.subplot_mosaic 创建图形 fig 和包含所有子图的字典 axd
        fig, axd = plt.subplot_mosaic(x, height_ratios=height_ratios)

        # 断言子图 "D" 的高度比例是否与给定的 height_ratios 相同
        assert axd["D"].get_gridspec().get_height_ratios() == height_ratios
        # 断言子图 "B" 的高度比例是否与给定的 height_ratios 不同
        assert axd["B"].get_gridspec().get_height_ratios() != height_ratios

    @check_figures_equal(extensions=["png"])
    # 使用装饰器检查两个图形是否相等，比较的文件扩展名为 PNG
    @pytest.mark.parametrize(
        "x, empty_sentinel",
        [
            ([["A", None], [None, "B"]], None),  # 参数化测试，定义了多组测试数据
            ([["A", "."], [".", "B"]], "SKIP"),
            ([["A", 0], [0, "B"]], 0),
            ([[1, None], [None, 2]], None),
            ([[1, "."], [".", 2]], "SKIP"),
            ([[1, 0], [0, 2]], 0),
        ],
    )
    def test_empty(self, fig_test, fig_ref, x, empty_sentinel):
        if empty_sentinel != "SKIP":
            kwargs = {"empty_sentinel": empty_sentinel}  # 根据 empty_sentinel 设置参数
        else:
            kwargs = {}
        grid_axes = fig_test.subplot_mosaic(x, **kwargs)  # 调用被测试的 subplot_mosaic 方法，生成网格图

        for k, ax in grid_axes.items():
            ax.set_title(k)  # 设置每个子图的标题

        labels = sorted(
            {name for row in x for name in row} - {empty_sentinel, "."}  # 从 x 中提取标签，排除 empty_sentinel 和 '.'
        )

        assert len(labels) == len(grid_axes)  # 断言标签数与生成的子图数相同

        gs = fig_ref.add_gridspec(2, 2)  # 在参考图中添加 2x2 的网格布局
        axA = fig_ref.add_subplot(gs[0, 0])  # 在网格布局的特定位置添加子图 axA
        axA.set_title(labels[0])  # 设置 axA 的标题为第一个标签

        axB = fig_ref.add_subplot(gs[1, 1])  # 在网格布局的特定位置添加子图 axB
        axB.set_title(labels[1])  # 设置 axB 的标题为第二个标签

    def test_fail_list_of_str(self):
        with pytest.raises(ValueError, match='must be 2D'):  # 检查是否会抛出 ValueError 异常
            plt.subplot_mosaic(['foo', 'bar'])  # 调用 subplot_mosaic 方法传入错误的参数
        with pytest.raises(ValueError, match='must be 2D'):
            plt.subplot_mosaic(['foo'])  # 再次调用 subplot_mosaic 方法传入错误的参数
        with pytest.raises(ValueError, match='must be 2D'):
            plt.subplot_mosaic([['foo', ('bar',)]])  # 再次调用 subplot_mosaic 方法传入错误的参数
        with pytest.raises(ValueError, match='must be 2D'):
            plt.subplot_mosaic([['a', 'b'], [('a', 'b'), 'c']])  # 再次调用 subplot_mosaic 方法传入错误的参数

    @check_figures_equal(extensions=["png"])  # 使用装饰器 check_figures_equal，比较生成的图像是否相等
    @pytest.mark.parametrize("subplot_kw", [{}, {"projection": "polar"}, None])
    def test_subplot_kw(self, fig_test, fig_ref, subplot_kw):
        x = [[1, 2]]  # 定义测试数据 x
        grid_axes = fig_test.subplot_mosaic(x, subplot_kw=subplot_kw)  # 调用 subplot_mosaic 方法生成网格图
        subplot_kw = subplot_kw or {}  # 如果 subplot_kw 为 None，则设为空字典

        gs = fig_ref.add_gridspec(1, 2)  # 在参考图中添加 1x2 的网格布局
        axA = fig_ref.add_subplot(gs[0, 0], **subplot_kw)  # 在网格布局的特定位置添加子图 axA，使用 subplot_kw
        axB = fig_ref.add_subplot(gs[0, 1], **subplot_kw)  # 在网格布局的特定位置添加子图 axB，使用 subplot_kw

    @check_figures_equal(extensions=["png"])  # 使用装饰器 check_figures_equal，比较生成的图像是否相等
    @pytest.mark.parametrize("multi_value", ['BC', tuple('BC')])
    def test_per_subplot_kw(self, fig_test, fig_ref, multi_value):
        x = 'AB;CD'  # 定义测试数据 x
        grid_axes = fig_test.subplot_mosaic(
            x,
            subplot_kw={'facecolor': 'red'},  # 设置子图全局属性
            per_subplot_kw={
                'D': {'facecolor': 'blue'},  # 设置特定子图属性
                multi_value: {'facecolor': 'green'},  # 根据多值设置特定子图属性
            }
        )

        gs = fig_ref.add_gridspec(2, 2)  # 在参考图中添加 2x2 的网格布局
        for color, spec in zip(['red', 'green', 'green', 'blue'], gs):
            fig_ref.add_subplot(spec, facecolor=color)  # 在特定位置添加子图，并设置背景色
    # 定义测试字符串解析功能的方法
    def test_string_parser(self):
        # 引用 Figure 类的 _normalize_grid_string 方法
        normalize = Figure._normalize_grid_string

        # 断言正常化 'ABC' 字符串后的结果为 [['A', 'B', 'C']]
        assert normalize('ABC') == [['A', 'B', 'C']]
        # 断言正常化 'AB;CC' 字符串后的结果为 [['A', 'B'], ['C', 'C']]
        assert normalize('AB;CC') == [['A', 'B'], ['C', 'C']]
        # 断言正常化 'AB;CC;DE' 字符串后的结果为 [['A', 'B'], ['C', 'C'], ['D', 'E']]
        assert normalize('AB;CC;DE') == [['A', 'B'], ['C', 'C'], ['D', 'E']]
        # 断言正常化多行字符串 'ABC' 后的结果为 [['A', 'B', 'C']]
        assert normalize("""
                         ABC
                         """) == [['A', 'B', 'C']]
        # 断言正常化多行字符串 'AB\nCC' 后的结果为 [['A', 'B'], ['C', 'C']]
        assert normalize("""
                         AB
                         CC
                         """) == [['A', 'B'], ['C', 'C']]
        # 断言正常化多行字符串 'AB\nCC\nDE' 后的结果为 [['A', 'B'], ['C', 'C'], ['D', 'E']]
        assert normalize("""
                         AB
                         CC
                         DE
                         """) == [['A', 'B'], ['C', 'C'], ['D', 'E']]

    # 定义测试每个子图参数扩展功能的方法
    def test_per_subplot_kw_expander(self):
        # 引用 Figure 类的 _norm_per_subplot_kw 方法
        normalize = Figure._norm_per_subplot_kw
        # 断言对 {"A": {}, "B": {}} 进行扩展后结果不变
        assert normalize({"A": {}, "B": {}}) == {"A": {}, "B": {}}
        # 断言对 {("A", "B"): {}} 进行扩展后结果为 {"A": {}, "B": {}}
        assert normalize({("A", "B"): {}}) == {"A": {}, "B": {}}
        # 断言对包含重复键的字典抛出 ValueError 异常
        with pytest.raises(
                ValueError, match=f'The key {"B"!r} appears multiple times'
        ):
            normalize({("A", "B"): {}, "B": {}})
        # 断言对包含重复键的字典抛出 ValueError 异常
        with pytest.raises(
                ValueError, match=f'The key {"B"!r} appears multiple times'
        ):
            normalize({"B": {}, ("A", "B"): {}})

    # 定义测试额外子图参数的方法
    def test_extra_per_subplot_kw(self):
        # 断言调用 Figure 类的 subplot_mosaic 方法时，包含不可哈希键 'B' 会抛出 ValueError 异常
        with pytest.raises(
                ValueError, match=f'The keys {set("B")!r} are in'
        ):
            Figure().subplot_mosaic("A", per_subplot_kw={"B": {}})

    # 使用装饰器设置图片比较扩展为 'png'，参数化测试单个字符串输入的方法
    @check_figures_equal(extensions=["png"])
    @pytest.mark.parametrize("str_pattern",
                             ["AAA\nBBB", "\nAAA\nBBB\n", "ABC\nDEF"]
                             )
    def test_single_str_input(self, fig_test, fig_ref, str_pattern):
        # 在 fig_test 上调用 subplot_mosaic 方法，并将结果存储在 grid_axes 中
        grid_axes = fig_test.subplot_mosaic(str_pattern)

        # 在 fig_ref 上调用 subplot_mosaic 方法，传入解析后的字符串模式的列表，并将结果存储在 grid_axes 中
        grid_axes = fig_ref.subplot_mosaic(
            [list(ln) for ln in str_pattern.strip().split("\n")]
        )

    # 参数化测试失败情况的方法
    @pytest.mark.parametrize(
        "x,match",
        [
            (
                [["A", "."], [".", "A"]],
                (
                    "(?m)we found that the label .A. specifies a "
                    + "non-rectangular or non-contiguous area."
                ),
            ),
            (
                [["A", "B"], [None, [["A", "B"], ["C", "D"]]]],
                "There are duplicate keys .* between the outer layout",
            ),
            ("AAA\nc\nBBB", "All of the rows must be the same length"),
            (
                [["A", [["B", "C"], ["D"]]], ["E", "E"]],
                "All of the rows must be the same length",
            ),
        ],
    )
    def test_fail(self, x, match):
        # 创建新的图形对象 fig
        fig = plt.figure()
        # 断言在调用 fig 的 subplot_mosaic 方法时，会抛出 ValueError 异常并匹配指定的错误信息
        with pytest.raises(ValueError, match=match):
            fig.subplot_mosaic(x)

    # 使用装饰器设置图片比较扩展为 'png'，测试可哈希键的情况
    @check_figures_equal(extensions=["png"])
    def test_hashable_keys(self, fig_test, fig_ref):
        # 在 fig_test 上调用 subplot_mosaic 方法，并传入包含对象的列表
        fig_test.subplot_mosaic([[object(), object()]])
        # 在 fig_ref 上调用 subplot_mosaic 方法，并传入包含字符串 'A' 和 'B' 的列表
        fig_ref.subplot_mosaic([["A", "B"]])
    # 使用 pytest 的参数化装饰器，定义多个测试用例，每个测试用例使用不同的字符串模式
    @pytest.mark.parametrize('str_pattern',
                             ['abc', 'cab', 'bca', 'cba', 'acb', 'bac'])
    def test_user_order(self, str_pattern):
        # 创建一个新的图形对象
        fig = plt.figure()
        # 根据给定的字符串模式布局图中的子图，返回一个子图字典
        ax_dict = fig.subplot_mosaic(str_pattern)
        # 断言：字符串模式应与子图字典的键列表相同
        assert list(str_pattern) == list(ax_dict)
        # 断言：图形对象的所有轴应与子图字典的值列表相同
        assert list(fig.axes) == list(ax_dict.values())

    # 定义测试嵌套布局的函数
    def test_nested_user_order(self):
        # 定义复杂的嵌套布局列表
        layout = [
            ["A", [["B", "C"],
                   ["D", "E"]]],
            ["F", "G"],
            [".", [["H", [["I"],
                          ["."]]]]]
        ]

        # 创建一个新的图形对象
        fig = plt.figure()
        # 根据复杂布局布置子图，并返回子图字典
        ax_dict = fig.subplot_mosaic(layout)
        # 断言：子图字典的键列表应与指定的字符串 "ABCDEFGHI" 相同
        assert list(ax_dict) == list("ABCDEFGHI")
        # 断言：图形对象的所有轴应与子图字典的值列表相同
        assert list(fig.axes) == list(ax_dict.values())

    # 定义测试共享所有轴的函数
    def test_share_all(self):
        # 定义复杂的嵌套布局列表
        layout = [
            ["A", [["B", "C"],
                   ["D", "E"]]],
            ["F", "G"],
            [".", [["H", [["I"],
                          ["."]]]]]
        ]
        # 创建一个新的图形对象
        fig = plt.figure()
        # 根据布局布置子图，并设置共享 x 和 y 轴
        ax_dict = fig.subplot_mosaic(layout, sharex=True, sharey=True)
        # 设置子图字典中键为 "A" 的子图的 x 轴为对数尺度，y 轴为 logit 尺度
        ax_dict["A"].set(xscale="log", yscale="logit")
        # 断言：所有子图的 x 轴尺度为对数尺度，y 轴尺度为 logit 尺度
        assert all(ax.get_xscale() == "log" and ax.get_yscale() == "logit"
                   for ax in ax_dict.values())
def test_reused_gridspec():
    """Test that these all use the same gridspec"""
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加第一个子图，使用3行2列的布局，占据第3和第5个位置
    ax1 = fig.add_subplot(3, 2, (3, 5))
    # 在图形上添加第二个子图，使用3行2列的布局，占据第4个位置
    ax2 = fig.add_subplot(3, 2, 4)
    # 使用subplot2grid方式添加第三个子图，位置在(2,1)，跨越两列
    ax3 = plt.subplot2grid((3, 2), (2, 1), colspan=2, fig=fig)

    # 获取每个子图的SubplotSpec对象，再从中获取其所属的GridSpec对象
    gs1 = ax1.get_subplotspec().get_gridspec()
    gs2 = ax2.get_subplotspec().get_gridspec()
    gs3 = ax3.get_subplotspec().get_gridspec()

    # 断言这三个子图使用的是同一个GridSpec对象
    assert gs1 == gs2
    assert gs1 == gs3


@image_comparison(['test_subfigure.png'], style='mpl20',
                  savefig_kwarg={'facecolor': 'teal'})
def test_subfigure():
    # 设置随机种子确保可重复性
    np.random.seed(19680801)
    # 创建一个约束布局的新图形对象
    fig = plt.figure(layout='constrained')
    # 添加一个包含2个子图的SubFigure对象
    sub = fig.subfigures(1, 2)

    # 在第一个子图中创建2x2的子图数组
    axs = sub[0].subplots(2, 2)
    # 对每个子图应用伪彩图
    for ax in axs.flat:
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2, vmax=2)
    # 在第一个子图上添加颜色条
    sub[0].colorbar(pc, ax=axs)
    # 设置第一个子图的背景色为白色
    sub[0].suptitle('Left Side')
    sub[0].set_facecolor('white')

    # 在第二个子图中创建1x3的子图数组
    axs = sub[1].subplots(1, 3)
    # 对每个子图应用伪彩图
    for ax in axs.flat:
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2, vmax=2)
    # 在第二个子图上添加底部位置的颜色条
    sub[1].colorbar(pc, ax=axs, location='bottom')
    # 设置第二个子图的标题
    sub[1].suptitle('Right Side')
    # 设置第二个子图的背景色为白色
    sub[1].set_facecolor('white')

    # 设置整个图形的总标题
    fig.suptitle('Figure suptitle', fontsize='xx-large')

    # 测试子图的绘制顺序
    leg = fig.legend(handles=[plt.Line2D([0], [0], label='Line{}'.format(i))
                     for i in range(5)], loc='center')
    # 设置第一个子图的绘制顺序在图例之前
    sub[0].set_zorder(leg.get_zorder() - 1)
    # 设置第二个子图的绘制顺序在图例之后
    sub[1].set_zorder(leg.get_zorder() + 1)


def test_subfigure_tightbbox():
    """Test that we can get the tightbbox with a subfigure..."""
    # 创建一个约束布局的新图形对象
    fig = plt.figure(layout='constrained')
    # 添加一个包含2个子图的SubFigure对象
    sub = fig.subfigures(1, 2)

    # 断言整个图形的紧凑边界框的宽度为8.0
    np.testing.assert_allclose(
            fig.get_tightbbox(fig.canvas.get_renderer()).width,
            8.0)


def test_subfigure_dpi():
    # 创建一个DPI为100的新图形对象
    fig = plt.figure(dpi=100)
    # 创建一个包含子图的SubFigure对象
    sub_fig = fig.subfigures()
    # 断言子图对象的DPI与图形对象的DPI相同
    assert sub_fig.get_dpi() == fig.get_dpi()

    # 设置子图对象的DPI为200
    sub_fig.set_dpi(200)
    # 再次断言子图对象的DPI为200
    assert sub_fig.get_dpi() == 200
    # 断言整个图形对象的DPI也为200
    assert fig.get_dpi() == 200


@image_comparison(['test_subfigure_ss.png'], style='mpl20',
                  savefig_kwarg={'facecolor': 'teal'}, tol=0.02)
def test_subfigure_ss():
    # 使用子图规范添加子图测试
    np.random.seed(19680801)
    # 创建一个约束布局的新图形对象
    fig = plt.figure(layout='constrained')
    # 添加一个1行2列的GridSpec对象
    gs = fig.add_gridspec(1, 2)

    # 在第一个子图规范位置添加一个SubFigure对象，背景色为粉色
    sub = fig.add_subfigure(gs[0], facecolor='pink')

    # 在第一个子图中创建2x2的子图数组
    axs = sub.subplots(2, 2)
    # 对每个子图应用伪彩图
    for ax in axs.flat:
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2, vmax=2)
    # 在第一个子图上添加颜色条
    sub.colorbar(pc, ax=axs)
    # 设置第一个子图的总标题
    sub.suptitle('Left Side')

    # 在第二个子图规范位置添加一个普通子图
    ax = fig.add_subplot(gs[1])
    ax.plot(np.arange(20))
    # 设置第二个子图的标题
    ax.set_title('Axes')

    # 设置整个图形的总标题
    fig.suptitle('Figure suptitle', fontsize='xx-large')


@image_comparison(['test_subfigure_double.png'], style='mpl20',
                  savefig_kwarg={'facecolor': 'teal'})
def test_subfigure_double():
    # 创建一个约束布局大小为10x8的新图形对象
    fig = plt.figure(layout='constrained', figsize=(10, 8))

    # 设置整个图形的总标题
    fig.suptitle('fig')
    # 创建一个包含两个子图的子图对象，水平间距为0.07
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    # 设置第一个子图的背景颜色为珊瑚色，并设置标题
    subfigs[0].set_facecolor('coral')
    subfigs[0].suptitle('subfigs[0]')

    # 设置第二个子图的背景颜色为珊瑚色，并设置标题
    subfigs[1].set_facecolor('coral')
    subfigs[1].suptitle('subfigs[1]')

    # 在第一个子图中创建一个包含两行一列的子图对象，高度比例分别为1和1.4
    subfigsnest = subfigs[0].subfigures(2, 1, height_ratios=[1, 1.4])
    # 设置第一个子图的第一个嵌套子图的标题并设置背景颜色为红色
    subfigsnest[0].suptitle('subfigsnest[0]')
    subfigsnest[0].set_facecolor('r')
    # 在第一个嵌套子图中创建一个包含一行两列的子图对象，共享y轴
    axsnest0 = subfigsnest[0].subplots(1, 2, sharey=True)
    # 对每个子图进行循环处理
    for ax in axsnest0:
        fontsize = 12
        # 在子图上绘制一个基于随机数据的伪彩色图
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2.5, vmax=2.5)
        ax.set_xlabel('x-label', fontsize=fontsize)  # 设置x轴标签
        ax.set_ylabel('y-label', fontsize=fontsize)  # 设置y轴标签
        ax.set_title('Title', fontsize=fontsize)  # 设置标题
    # 在第一个嵌套子图的第一个子图上添加颜色条
    subfigsnest[0].colorbar(pc, ax=axsnest0)

    # 设置第一个嵌套子图的第二个子图的标题并设置背景颜色为绿色
    subfigsnest[1].suptitle('subfigsnest[1]')
    subfigsnest[1].set_facecolor('g')
    # 在第二个嵌套子图中创建一个包含三行一列的子图对象，共享x轴
    axsnest1 = subfigsnest[1].subplots(3, 1, sharex=True)
    # 对每个子图进行循环处理
    for nn, ax in enumerate(axsnest1):
        ax.set_ylabel(f'ylabel{nn}')  # 设置y轴标签
    # 设置第二个嵌套子图的全局x轴标签
    subfigsnest[1].supxlabel('supxlabel')
    # 设置第二个嵌套子图的全局y轴标签
    subfigsnest[1].supylabel('supylabel')

    # 在第二个子图中创建一个包含两行两列的子图对象
    axsRight = subfigs[1].subplots(2, 2)
def test_subfigure_spanning():
    # 测试子图的布局是否正确安排...

    # 创建一个带有约束布局的图形对象
    fig = plt.figure(constrained_layout=True)
    
    # 创建一个3x3的网格规格对象
    gs = fig.add_gridspec(3, 3)
    
    # 向图形中添加子图，并将其存储在列表中
    sub_figs = [
        fig.add_subfigure(gs[0, 0]),      # 添加到第一行第一列
        fig.add_subfigure(gs[0:2, 1]),    # 添加到第一列第二列
        fig.add_subfigure(gs[2, 1:3]),    # 添加到第三行第二列和第三列
        fig.add_subfigure(gs[0:, 1:])     # 添加到所有行的第二列和第三列
    ]

    # 设置图形的宽度和高度
    w = 640
    h = 480
    
    # 使用数值近似断言确保子图的边界框最小值和最大值符合预期
    np.testing.assert_allclose(sub_figs[0].bbox.min, [0., h * 2/3])
    np.testing.assert_allclose(sub_figs[0].bbox.max, [w / 3, h])

    np.testing.assert_allclose(sub_figs[1].bbox.min, [w / 3, h / 3])
    np.testing.assert_allclose(sub_figs[1].bbox.max, [w * 2/3, h])

    np.testing.assert_allclose(sub_figs[2].bbox.min, [w / 3, 0])
    np.testing.assert_allclose(sub_figs[2].bbox.max, [w, h / 3])

    # 在此检查切片是否有效。最后一个子图
    # 使用for循环添加子图
    for i in range(4):
        sub_figs[i].add_subplot()
    
    # 绘制图形但不进行渲染
    fig.draw_without_rendering()


@mpl.style.context('mpl20')
def test_subfigure_ticks():
    # 这个测试检查一个在保存到文件时才显现出来的刻度间隔错误。很难复制

    # 创建一个带有约束布局和特定尺寸的图形对象
    fig = plt.figure(constrained_layout=True, figsize=(10, 3))
    
    # 在底部子图中创建左右子图并指定宽度比例
    (subfig_bl, subfig_br) = fig.subfigures(1, 2, wspace=0.01,
                                            width_ratios=[7, 2])

    # 在底部左侧子图的网格规格中放置ax1-ax3
    gs = subfig_bl.add_gridspec(nrows=1, ncols=14)

    ax1 = subfig_bl.add_subplot(gs[0, :1])            # 添加到第一列
    ax1.scatter(x=[-56.46881504821776, 24.179891162109396], y=[1500, 3600])

    ax2 = subfig_bl.add_subplot(gs[0, 1:3], sharey=ax1)   # 添加到第二列到第三列，与ax1共享y轴
    ax2.scatter(x=[-126.5357270050049, 94.68456736755368], y=[1500, 3600])
    
    ax3 = subfig_bl.add_subplot(gs[0, 3:14], sharey=ax1)   # 添加到第四列到第十四列，与ax1共享y轴

    # 设置图形的dpi为120并绘制图形但不进行渲染
    fig.dpi = 120
    fig.draw_without_rendering()
    
    # 获取ax2的x刻度并存储在ticks120中
    ticks120 = ax2.get_xticks()
    
    # 将图形的dpi设置为300并绘制图形但不进行渲染
    fig.dpi = 300
    fig.draw_without_rendering()
    
    # 获取ax2的x刻度并存储在ticks300中
    ticks300 = ax2.get_xticks()
    
    # 使用数值近似断言确保两次获取的刻度值接近
    np.testing.assert_allclose(ticks120, ticks300)


@image_comparison(['test_subfigure_scatter_size.png'], style='mpl20',
                   remove_text=True)
def test_subfigure_scatter_size():
    # 左侧和右侧子图中的标记应该相同

    # 创建一个图形对象
    fig = plt.figure()
    
    # 添加一个1x2的网格规格
    gs = fig.add_gridspec(1, 2)
    
    # 添加第一个子图到第二个网格规格
    ax0 = fig.add_subplot(gs[1])
    ax0.scatter([1, 2, 3], [1, 2, 3], s=30, marker='s')
    ax0.scatter([3, 4, 5], [1, 2, 3], s=[20, 30, 40], marker='s')

    # 添加一个子图到第一个网格规格
    sfig = fig.add_subfigure(gs[0])
    
    # 在子图中添加1x2的子图布局
    axs = sfig.subplots(1, 2)
    
    # 遍历并在ax0和axs[0]上分别绘制散点图
    for ax in [ax0, axs[0]]:
        ax.scatter([1, 2, 3], [1, 2, 3], s=30, marker='s', color='r')
        ax.scatter([3, 4, 5], [1, 2, 3], s=[20, 30, 40], marker='s', color='g')


def test_subfigure_pdf():
    # 创建一个带有约束布局的图形对象
    fig = plt.figure(layout='constrained')
    
    # 添加一个子图到图形中
    sub_fig = fig.subfigures()
    
    # 添加一个子图布局的坐标轴到子图中
    ax = sub_fig.add_subplot(111)
    
    # 在坐标轴上添加一个柱状图，并添加柱状图标签
    b = ax.bar(1, 1)
    ax.bar_label(b)
    
    # 创建一个字节流缓冲区，并将图形保存为PDF格式
    buffer = io.BytesIO()
    fig.savefig(buffer, format='pdf')


def test_subfigures_wspace_hspace():
    # 这个函数尚未实现，只是一个函数定义
    # 创建一个包含多个子图的画布对象，2行3列的布局，水平间隔为图像宽度的1/6，垂直间隔为0.5
    sub_figs = plt.figure().subfigures(2, 3, hspace=0.5, wspace=1/6.)
    
    # 设置画布的宽度和高度
    w = 640
    h = 480
    
    # 对第一个子图的边界框进行测试，验证其最小边界和最大边界是否符合预期
    np.testing.assert_allclose(sub_figs[0, 0].bbox.min, [0., h * 0.6])
    np.testing.assert_allclose(sub_figs[0, 0].bbox.max, [w * 0.3, h])
    
    # 对第二个子图的边界框进行测试，验证其最小边界和最大边界是否符合预期
    np.testing.assert_allclose(sub_figs[0, 1].bbox.min, [w * 0.35, h * 0.6])
    np.testing.assert_allclose(sub_figs[0, 1].bbox.max, [w * 0.65, h])
    
    # 对第三个子图的边界框进行测试，验证其最小边界和最大边界是否符合预期
    np.testing.assert_allclose(sub_figs[0, 2].bbox.min, [w * 0.7, h * 0.6])
    np.testing.assert_allclose(sub_figs[0, 2].bbox.max, [w, h])
    
    # 对第四个子图的边界框进行测试，验证其最小边界和最大边界是否符合预期
    np.testing.assert_allclose(sub_figs[1, 0].bbox.min, [0, 0])
    np.testing.assert_allclose(sub_figs[1, 0].bbox.max, [w * 0.3, h * 0.4])
    
    # 对第五个子图的边界框进行测试，验证其最小边界和最大边界是否符合预期
    np.testing.assert_allclose(sub_figs[1, 1].bbox.min, [w * 0.35, 0])
    np.testing.assert_allclose(sub_figs[1, 1].bbox.max, [w * 0.65, h * 0.4])
    
    # 对第六个子图的边界框进行测试，验证其最小边界和最大边界是否符合预期
    np.testing.assert_allclose(sub_figs[1, 2].bbox.min, [w * 0.7, 0])
    np.testing.assert_allclose(sub_figs[1, 2].bbox.max, [w, h * 0.4])
def test_subfigure_remove():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 创建一个包含 2x2 子图的对象数组
    sfs = fig.subfigures(2, 2)
    # 移除数组中索引为 (1, 1) 的子图
    sfs[1, 1].remove()
    # 断言当前图形对象的子图数量为 3
    assert len(fig.subfigs) == 3


def test_add_subplot_kwargs():
    # fig.add_subplot() 总是创建新的坐标轴对象，即使参数不同。
    fig = plt.figure()
    # 添加一个 1x1 的子图
    ax = fig.add_subplot(1, 1, 1)
    # 再次添加一个 1x1 的子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 断言两个子图对象不是同一个对象
    assert ax is not ax1
    plt.close()

    fig = plt.figure()
    # 添加一个 1x1 的极坐标子图
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    # 再次添加一个 1x1 的极坐标子图
    ax1 = fig.add_subplot(1, 1, 1, projection='polar')
    # 断言两个子图对象不是同一个对象
    assert ax is not ax1
    plt.close()

    fig = plt.figure()
    # 添加一个 1x1 的极坐标子图
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    # 再次添加一个 1x1 的矩形坐标子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 断言 ax1 是矩形坐标系
    assert ax1.name == 'rectilinear'
    # 断言两个子图对象不是同一个对象
    assert ax is not ax1
    plt.close()


def test_add_axes_kwargs():
    # fig.add_axes() 总是创建新的坐标轴对象，即使参数不同。
    fig = plt.figure()
    # 添加一个具有指定位置和大小的坐标轴
    ax = fig.add_axes([0, 0, 1, 1])
    # 再次添加一个具有相同位置和大小的坐标轴
    ax1 = fig.add_axes([0, 0, 1, 1])
    # 断言两个坐标轴对象不是同一个对象
    assert ax is not ax1
    plt.close()

    fig = plt.figure()
    # 添加一个具有指定位置和大小的极坐标系坐标轴
    ax = fig.add_axes([0, 0, 1, 1], projection='polar')
    # 再次添加一个具有相同位置和大小的极坐标系坐标轴
    ax1 = fig.add_axes([0, 0, 1, 1], projection='polar')
    # 断言两个坐标轴对象不是同一个对象
    assert ax is not ax1
    plt.close()

    fig = plt.figure()
    # 添加一个具有指定位置和大小的极坐标系坐标轴
    ax = fig.add_axes([0, 0, 1, 1], projection='polar')
    # 再次添加一个具有相同位置和大小的矩形坐标系坐标轴
    ax1 = fig.add_axes([0, 0, 1, 1])
    # 断言 ax1 是矩形坐标系
    assert ax1.name == 'rectilinear'
    # 断言两个坐标轴对象不是同一个对象
    assert ax is not ax1
    plt.close()


def test_ginput(recwarn):  # recwarn 用于在退出时撤销警告过滤器。
    warnings.filterwarnings("ignore", "cannot show the figure")
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    trans = ax.transData.transform

    def single_press():
        # 模拟单次鼠标按下事件
        MouseEvent("button_press_event", fig.canvas, *trans((.1, .2)), 1)._process()

    # 设置一个定时器，在0.1秒后执行单次鼠标按下事件模拟
    Timer(.1, single_press).start()
    # 断言图形的 ginput 方法返回的坐标与预期相符
    assert fig.ginput() == [(.1, .2)]

    def multi_presses():
        # 模拟多次鼠标按下和一个键盘按下事件
        MouseEvent("button_press_event", fig.canvas, *trans((.1, .2)), 1)._process()
        KeyEvent("key_press_event", fig.canvas, "backspace")._process()
        MouseEvent("button_press_event", fig.canvas, *trans((.3, .4)), 1)._process()
        MouseEvent("button_press_event", fig.canvas, *trans((.5, .6)), 1)._process()
        MouseEvent("button_press_event", fig.canvas, *trans((0, 0)), 2)._process()

    # 设置一个定时器，在0.1秒后执行多次按下事件模拟
    Timer(.1, multi_presses).start()
    # 断言图形的 ginput 方法返回的坐标与预期相符
    np.testing.assert_allclose(fig.ginput(3), [(.3, .4), (.5, .6)])


def test_waitforbuttonpress(recwarn):  # recwarn 用于在退出时撤销警告过滤器。
    warnings.filterwarnings("ignore", "cannot show the figure")
    # 创建一个新的图形对象
    fig = plt.figure()
    # 断言在超时时间为0.1秒内未检测到按钮按下事件
    assert fig.waitforbuttonpress(timeout=.1) is None
    # 设置一个定时器，在0.1秒后发送一个键盘按下事件
    Timer(.1, KeyEvent("key_press_event", fig.canvas, "z")._process).start()
    # 断言在等待中检测到按钮按下事件
    assert fig.waitforbuttonpress() is True
    # 设置一个定时器，在0.1秒后发送一个鼠标按下事件
    Timer(.1, MouseEvent("button_press_event", fig.canvas, 0, 0, 1)._process).start()
    # 断言在等待中未检测到按钮按下事件
    assert fig.waitforbuttonpress() is False


def test_kwargs_pass():
    # 创建一个新的图形对象，传递一个不支持的参数
    fig = Figure(label='whole Figure')
    # 创建一个包含一个子图的子图对象，标签为 'sub figure'
    sub_fig = fig.subfigures(1, 1, label='sub figure')
    
    # 断言主图对象的标签为 'whole Figure'
    assert fig.get_label() == 'whole Figure'
    # 断言子图对象的标签为 'sub figure'
    assert sub_fig.get_label() == 'sub figure'
@check_figures_equal(extensions=["png"])
def test_rcparams(fig_test, fig_ref):
    # 设置参考图表的标签和标题样式
    fig_ref.supxlabel("xlabel", weight='bold', size=15)
    fig_ref.supylabel("ylabel", weight='bold', size=15)
    fig_ref.suptitle("Title", weight='light', size=20)
    
    # 在上下文中设置测试图表的全局样式
    with mpl.rc_context({'figure.labelweight': 'bold',
                         'figure.labelsize': 15,
                         'figure.titleweight': 'light',
                         'figure.titlesize': 20}):
        # 应用样式到测试图表的标签和标题
        fig_test.supxlabel("xlabel")
        fig_test.supylabel("ylabel")
        fig_test.suptitle("Title")


def test_deepcopy():
    # 创建一个图表和轴
    fig1, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])
    ax.set_yscale('log')

    # 深拷贝图表对象
    fig2 = copy.deepcopy(fig1)

    # 确保深拷贝生成了新对象
    assert fig2.axes[0] is not ax
    # 确保轴的比例尺被正确复制
    assert fig2.axes[0].get_yscale() == 'log'
    
    # 更新深拷贝后的轴，并检查原始轴未被修改
    fig2.axes[0].set_yscale('linear')
    assert ax.get_yscale() == 'log'

    # 测试轴的限制未被复制
    ax.set_xlim(1e-1, 1e2)
    # 绘制以确保限制被更新
    fig1.draw_without_rendering()
    fig2.draw_without_rendering()

    assert ax.get_xlim() == (1e-1, 1e2)
    assert fig2.axes[0].get_xlim() == (0, 1)


def test_unpickle_with_device_pixel_ratio():
    # 创建一个 DPI 设置为 42 的图表
    fig = Figure(dpi=42)
    fig.canvas._set_device_pixel_ratio(7)
    assert fig.dpi == 42*7
    
    # 使用 pickle 实现深拷贝和反序列化
    fig2 = pickle.loads(pickle.dumps(fig))
    assert fig2.dpi == 42


def test_gridspec_no_mutate_input():
    # 定义一个网格规格字典
    gs = {'left': .1}
    gs_orig = dict(gs)
    
    # 创建一个带有自定义网格规格的子图
    plt.subplots(1, 2, width_ratios=[1, 2], gridspec_kw=gs)
    assert gs == gs_orig
    
    # 使用 subplot_mosaic 创建带有相同网格规格的子图
    plt.subplot_mosaic('AB', width_ratios=[1, 2], gridspec_kw=gs)


@pytest.mark.parametrize('fmt', ['eps', 'pdf', 'png', 'ps', 'svg', 'svgz'])
def test_savefig_metadata(fmt):
    # 测试保存图表时不添加元数据
    Figure().savefig(io.BytesIO(), format=fmt, metadata={})


@pytest.mark.parametrize('fmt', ['jpeg', 'jpg', 'tif', 'tiff', 'webp', "raw", "rgba"])
def test_savefig_metadata_error(fmt):
    # 测试保存图表时添加不支持的元数据格式
    with pytest.raises(ValueError, match="metadata not supported"):
        Figure().savefig(io.BytesIO(), format=fmt, metadata={})


def test_get_constrained_layout_pads():
    # 测试获取受限布局的填充参数
    params = {'w_pad': 0.01, 'h_pad': 0.02, 'wspace': 0.03, 'hspace': 0.04}
    expected = tuple([*params.values()])
    
    # 创建一个使用受限布局引擎的图表
    fig = plt.figure(layout=mpl.layout_engine.ConstrainedLayoutEngine(**params))
    
    # 发出即将弃用警告，确保填充参数被正确获取
    with pytest.warns(PendingDeprecationWarning, match="will be deprecated"):
        assert fig.get_constrained_layout_pads() == expected


def test_not_visible_figure():
    # 测试不可见图表的保存行为
    fig = Figure()

    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    buf.seek(0)
    assert '<g ' in buf.read()

    # 设置图表不可见并再次保存
    fig.set_visible(False)
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    buf.seek(0)
    assert '<g ' not in buf.read()


def test_warn_colorbar_mismatch():
    # 测试警告色条不匹配的情况
    fig1, ax1 = plt.subplots()
    fig2, (ax2_1, ax2_2) = plt.subplots(2)
    im = ax1.imshow([[1, 2], [3, 4]])
    fig1.colorbar(im)  # 给 fig1 添加 colorbar，不应产生警告

    with pytest.warns(UserWarning, match="different Figure"):
        fig2.colorbar(im)
    # 使用 pytest 检查，期望捕获 UserWarning 并匹配 "different Figure"，给 fig2 添加 colorbar

    # 在没有推断宿主图的情况下警告不匹配
    with pytest.warns(UserWarning, match="different Figure"):
        fig2.colorbar(im, ax=ax1)
    # 使用 pytest 检查，期望捕获 UserWarning 并匹配 "different Figure"，给 fig2 在 ax1 轴上添加 colorbar

    with pytest.warns(UserWarning, match="different Figure"):
        fig2.colorbar(im, ax=ax2_1)
    # 使用 pytest 检查，期望捕获 UserWarning 并匹配 "different Figure"，给 fig2 在 ax2_1 轴上添加 colorbar

    with pytest.warns(UserWarning, match="different Figure"):
        fig2.colorbar(im, cax=ax2_2)
    # 使用 pytest 检查，期望捕获 UserWarning 并匹配 "different Figure"，给 fig2 在 cax=ax2_2 上添加 colorbar

    # 边缘情况：在子图的情况下仅比较顶层艺术家
    fig3 = plt.figure()
    fig4 = plt.figure()
    subfig3_1 = fig3.subfigures()
    subfig3_2 = fig3.subfigures()
    subfig4_1 = fig4.subfigures()
    ax3_1 = subfig3_1.subplots()
    ax3_2 = subfig3_1.subplots()
    ax4_1 = subfig4_1.subplots()
    im3_1 = ax3_1.imshow([[1, 2], [3, 4]])
    im3_2 = ax3_2.imshow([[1, 2], [3, 4]])
    im4_1 = ax4_1.imshow([[1, 2], [3, 4]])

    fig3.colorbar(im3_1)   # 给 fig3 添加 colorbar，不应产生警告
    subfig3_1.colorbar(im3_1)   # 给 fig3 的子图 subfig3_1 添加 colorbar，不应产生警告
    subfig3_1.colorbar(im3_2)   # 给 fig3 的子图 subfig3_1 添加 colorbar，不应产生警告

    with pytest.warns(UserWarning, match="different Figure"):
        subfig3_1.colorbar(im4_1)
    # 使用 pytest 检查，期望捕获 UserWarning 并匹配 "different Figure"，给 fig3 的子图 subfig3_1 添加 fig4 的 im4_1 的 colorbar
# 测试`
# 测试子图以行主顺序绘制的功能
def test_subfigure_row_order():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 创建一个4行3列的子图数组
    sf_arr = fig.subfigures(4, 3)
    # 遍历并比较平铺后的子图数组与原始子图数组的每个元素
    for a, b in zip(sf_arr.ravel(), fig.subfigs):
        # 断言每个平铺后的子图对象与原始子图对象相同
        assert a is b


# 测试子图的过期属性传播
def test_subfigure_stale_propagation():
    # 创建一个新的图形对象
    fig = plt.figure()

    # 在不渲染的情况下绘制图形
    fig.draw_without_rendering()
    # 断言图形对象不过期
    assert not fig.stale

    # 创建子图对象 sfig1
    sfig1 = fig.subfigures()
    # 断言图形对象过期
    assert fig.stale

    # 再次在不渲染的情况下绘制图形
    fig.draw_without_rendering()
    # 断言图形对象不过期
    assert not fig.stale
    # 断言 sfig1 对象不过期
    assert not sfig1.stale

    # 创建 sfig1 的子图对象 sfig2
    sfig2 = sfig1.subfigures()
    # 断言图形对象过期
    assert fig.stale

    # 再次在不渲染的情况下绘制图形
    fig.draw_without_rendering()
    # 断言图形对象不过期
    assert not fig.stale
    # 断言 sfig2 对象不过期
    assert not sfig2.stale

    # 将 sfig2 对象标记为过期
    sfig2.stale = True
    # 断言图形对象过期
    assert fig.stale
```