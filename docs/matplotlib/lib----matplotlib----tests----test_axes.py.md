# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_axes.py`

```
# 导入 contextlib 模块，用于创建上下文管理器和处理上下文相关的功能
import contextlib
# 从 collections 模块导入 namedtuple，用于创建命名元组
from collections import namedtuple
# 导入 datetime 模块，用于处理日期和时间
import datetime
# 从 decimal 模块导入 Decimal，用于高精度的十进制浮点运算
from decimal import Decimal
# 从 functools 模块导入 partial，用于部分函数应用
from functools import partial
# 导入 inspect 模块，用于解析源代码和检查对象
import inspect
# 导入 io 模块，用于处理流（stream）相关的操作
import io
# 从 itertools 模块导入 product，用于生成迭代器的笛卡尔积
from itertools import product
# 导入 platform 模块，用于访问平台相关的系统信息
import platform
# 从 types 模块导入 SimpleNamespace，用于创建简单命名空间对象
from types import SimpleNamespace

# 导入 dateutil.tz 模块，处理时区相关的日期和时间
import dateutil.tz

# 导入 numpy 库，并将其别名为 np
import numpy as np
# 从 numpy 中导入 ma 模块，用于处理带掩码数组（masked arrays）
from numpy import ma
# 导入 cycler 模块，用于生成用于循环的样式列表
from cycler import cycler
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 matplotlib 库，并将其别名为 mpl
import matplotlib
# 再次导入 matplotlib 并将其别名为 mpl，这次导入的是 mpl
import matplotlib as mpl
# 从 matplotlib 中导入 rc_context 和 patheffects，分别用于上下文管理和路径效果
from matplotlib import rc_context, patheffects
# 导入 matplotlib.colors 模块，处理颜色相关的功能
import matplotlib.colors as mcolors
# 导入 matplotlib.dates 模块，处理日期相关的功能
import matplotlib.dates as mdates
# 从 matplotlib.figure 模块导入 Figure 类，表示图形对象
from matplotlib.figure import Figure
# 从 matplotlib.axes 模块导入 Axes 类，表示坐标轴对象
from matplotlib.axes import Axes
# 导入 matplotlib.font_manager 模块，用于管理字体
import matplotlib.font_manager as mfont_manager
# 导入 matplotlib.markers 模块，处理标记相关的功能
import matplotlib.markers as mmarkers
# 导入 matplotlib.patches 模块，处理图形块相关的功能
import matplotlib.patches as mpatches
# 导入 matplotlib.path 模块，处理路径相关的功能
import matplotlib.path as mpath
# 从 matplotlib.projections.geo 模块导入 HammerAxes 类，表示锤形投影的坐标轴
from matplotlib.projections.geo import HammerAxes
# 从 matplotlib.projections.polar 模块导入 PolarAxes 类，表示极坐标投影的坐标轴
from matplotlib.projections.polar import PolarAxes
# 导入 matplotlib.pyplot 模块，并将其别名为 plt
import matplotlib.pyplot as plt
# 从 matplotlib.text 模块导入 mtext，处理文本相关的功能
import matplotlib.text as mtext
# 从 matplotlib.ticker 模块导入 mticker，处理刻度相关的功能
import matplotlib.ticker as mticker
# 从 matplotlib.transforms 模块导入 mtransforms，处理坐标变换相关的功能
import matplotlib.transforms as mtransforms
# 导入 mpl_toolkits.axisartist 模块，并将其别名为 AA，用于处理坐标轴艺术家
import mpl_toolkits.axisartist as AA  # type: ignore
# 从 numpy.testing 模块导入几个函数，用于进行数组测试和比较
from numpy.testing import (
    assert_allclose, assert_array_equal, assert_array_almost_equal)
# 从 matplotlib.testing.decorators 模块导入几个装饰器，用于测试图片和移除标签和标题
from matplotlib.testing.decorators import (
    image_comparison, check_figures_equal, remove_ticks_and_titles)
# 从 matplotlib.testing._markers 模块导入 needs_usetex 标记，用于需要使用 TeX 的测试

# 注释说明：
# 下面的测试函数使用了不同的装饰器来测试 Matplotlib 的各种功能，包括图形的可见性、标签的获取和设置、对象的字符串表示、标签的位置设置等。
    # 设置图形的 x 轴标签为 'X Label'，位置位于左边
    ax.set_xlabel('X Label', loc='left')
    
    # 在图形 fig_test 上创建水平方向的颜色条，用于散点图 sc
    cbar = fig_test.colorbar(sc, orientation='horizontal')
    
    # 设置颜色条的标签为 "Z Label"，位置位于左边
    cbar.set_label("Z Label", loc='left')
    
    # 在图形 fig_ref 上创建子图 ax
    ax = fig_ref.subplots()
    
    # 绘制散点图，散点坐标为 (1, 1) 和 (2, 2)，颜色值为 [1, 2]，设置图例为 'scatter'
    sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
    
    # 添加图例到子图 ax
    ax.legend()
    
    # 设置子图 ax 的 y 轴标签为 'Y Label'，y 轴标签位置为 0，水平对齐方式为左对齐
    ax.set_ylabel('Y Label', y=0, ha='left')
    
    # 设置子图 ax 的 x 轴标签为 'X Label'，x 轴标签位置为 0，水平对齐方式为左对齐
    ax.set_xlabel('X Label', x=0, ha='left')
    
    # 在图形 fig_ref 上创建水平方向的颜色条，用于散点图 sc
    cbar = fig_ref.colorbar(sc, orientation='horizontal')
    
    # 设置颜色条的标签为 "Z Label"，x 轴标签位置为 0，水平对齐方式为左对齐
    cbar.set_label("Z Label", x=0, ha='left')
# 使用装饰器 @check_figures_equal() 来测试两个图形对象是否相等
@check_figures_equal()
def test_label_loc_rc(fig_test, fig_ref):
    # 设置 matplotlib 的上下文环境，更改坐标轴标签的位置为右上角
    with matplotlib.rc_context({"xaxis.labellocation": "right",
                                "yaxis.labellocation": "top"}):
        # 在 fig_test 上创建子图 ax
        ax = fig_test.subplots()
        # 绘制散点图并设置图例
        sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
        ax.legend()
        # 设置 y 轴和 x 轴的标签
        ax.set_ylabel('Y Label')
        ax.set_xlabel('X Label')
        # 添加水平方向的色标
        cbar = fig_test.colorbar(sc, orientation='horizontal')
        cbar.set_label("Z Label")

    # 在 fig_ref 上创建子图 ax
    ax = fig_ref.subplots()
    # 绘制散点图并设置图例
    sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
    ax.legend()
    # 设置右对齐的 y 轴标签和 x 轴标签
    ax.set_ylabel('Y Label', y=1, ha='right')
    ax.set_xlabel('X Label', x=1, ha='right')
    # 添加水平方向的色标，并右对齐标签
    cbar = fig_ref.colorbar(sc, orientation='horizontal')
    cbar.set_label("Z Label", x=1, ha='right')


# 定义测试函数 test_label_shift
def test_label_shift():
    # 创建新的图形和轴对象
    fig, ax = plt.subplots()

    # 测试在 x 轴上标签的重新居中
    ax.set_xlabel("Test label", loc="left")  # 将 x 轴标签设置在左侧
    ax.set_xlabel("Test label", loc="center")  # 将 x 轴标签居中
    assert ax.xaxis.get_label().get_horizontalalignment() == "center"  # 断言 x 轴标签的水平对齐方式为居中
    ax.set_xlabel("Test label", loc="right")  # 将 x 轴标签设置在右侧
    assert ax.xaxis.get_label().get_horizontalalignment() == "right"  # 断言 x 轴标签的水平对齐方式为右对齐
    ax.set_xlabel("Test label", loc="center")  # 再次将 x 轴标签居中
    assert ax.xaxis.get_label().get_horizontalalignment() == "center"  # 断言 x 轴标签的水平对齐方式为居中

    # 测试在 y 轴上标签的重新居中
    ax.set_ylabel("Test label", loc="top")  # 将 y 轴标签设置在顶部
    ax.set_ylabel("Test label", loc="center")  # 将 y 轴标签居中
    assert ax.yaxis.get_label().get_horizontalalignment() == "center"  # 断言 y 轴标签的水平对齐方式为居中
    ax.set_ylabel("Test label", loc="bottom")  # 将 y 轴标签设置在底部
    assert ax.yaxis.get_label().get_horizontalalignment() == "left"  # 断言 y 轴标签的水平对齐方式为左对齐
    ax.set_ylabel("Test label", loc="center")  # 再次将 y 轴标签居中
    assert ax.yaxis.get_label().get_horizontalalignment() == "center"  # 断言 y 轴标签的水平对齐方式为居中


# 使用装饰器 @check_figures_equal(extensions=["png"]) 来测试两个图形对象是否相等，生成 PNG 扩展名的文件
@check_figures_equal(extensions=["png"])
def test_acorr(fig_test, fig_ref):
    np.random.seed(19680801)
    Nx = 512
    x = np.random.normal(0, 1, Nx).cumsum()
    maxlags = Nx-1

    # 在 fig_test 上创建子图 ax_test，绘制 x 的自相关图
    ax_test = fig_test.subplots()
    ax_test.acorr(x, maxlags=maxlags)

    # 在 fig_ref 上创建子图 ax_ref，绘制 x 的自相关图（标准化）
    ax_ref = fig_ref.subplots()
    # 标准化自相关
    norm_auto_corr = np.correlate(x, x, mode="full")/np.dot(x, x)
    lags = np.arange(-maxlags, maxlags+1)
    norm_auto_corr = norm_auto_corr[Nx-1-maxlags:Nx+maxlags]
    # 在 ax_ref 上绘制垂直线和水平线
    ax_ref.vlines(lags, [0], norm_auto_corr)
    ax_ref.axhline(y=0, xmin=0, xmax=1)


# 使用装饰器 @check_figures_equal(extensions=["png"]) 来测试两个图形对象是否相等，生成 PNG 扩展名的文件
@check_figures_equal(extensions=["png"])
def test_acorr_integers(fig_test, fig_ref):
    np.random.seed(19680801)
    Nx = 51
    x = (np.random.rand(Nx) * 10).cumsum()
    x = (np.ceil(x)).astype(np.int64)
    maxlags = Nx-1

    # 在 fig_test 上创建子图 ax_test，绘制 x 的自相关图
    ax_test = fig_test.subplots()
    ax_test.acorr(x, maxlags=maxlags)

    # 在 fig_ref 上创建子图 ax_ref，绘制 x 的自相关图（标准化）
    ax_ref = fig_ref.subplots()
    # 标准化自相关
    norm_auto_corr = np.correlate(x, x, mode="full")/np.dot(x, x)
    lags = np.arange(-maxlags, maxlags+1)
    norm_auto_corr = norm_auto_corr[Nx-1-maxlags:Nx+maxlags]
    # 在 ax_ref 上绘制垂直线和水平线
    ax_ref.vlines(lags, [0], norm_auto_corr)
    ax_ref.axhline(y=0, xmin=0, xmax=1)


# 使用装饰器 @check_figures_equal(extensions=["png"]) 来测试两个图形对象是否相等，生成 PNG 扩展名的文件
@check_figures_equal(extensions=["png"])
def test_spy(fig_test, fig_ref):
    np.random.seed(19680801)
    # 创建一个包含 32*32 个元素的数组，所有元素初始化为1
    a = np.ones(32 * 32)
    # 将数组的前 16*32 个元素置为0，其余保持为1
    a[:16 * 32] = 0
    # 随机打乱数组中的元素顺序
    np.random.shuffle(a)
    # 将数组重新调整形状为 32x32 的二维数组
    a = a.reshape((32, 32))
    
    # 在测试图表上创建包含两个子图的轴对象
    axs_test = fig_test.subplots(2)
    # 在第一个子图上绘制稀疏图像，显示数组a中的非零元素位置
    axs_test[0].spy(a)
    # 在第二个子图上绘制稀疏图像，使用"."作为标记，并以底部为原点
    axs_test[1].spy(a, marker=".", origin="lower")
    
    # 在参考图表上创建包含两个子图的轴对象
    axs_ref = fig_ref.subplots(2)
    # 在第一个子图上显示灰度图像，使用灰度反转的颜色映射，最近邻插值
    axs_ref[0].imshow(a, cmap="gray_r", interpolation="nearest")
    # 设置第一个子图的x轴刻度在顶部显示
    axs_ref[0].xaxis.tick_top()
    # 在第二个子图上绘制非零元素的散点图，反转y轴和x轴的非零元素坐标
    axs_ref[1].plot(*np.nonzero(a)[::-1], ".", markersize=10)
    # 设置第二个子图的比例为1:1，x轴和y轴的范围与第一个子图相同
    axs_ref[1].set(
        aspect=1, xlim=axs_ref[0].get_xlim(), ylim=axs_ref[0].get_ylim()[::-1])
    # 设置两个子图的x轴刻度位置为"both"，即显示在上方和下方
    for ax in axs_ref:
        ax.xaxis.set_ticks_position("both")
def test_spy_invalid_kwargs():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 遍历每个不支持的关键字参数字典
    for unsupported_kw in [{'interpolation': 'nearest'},
                           {'marker': 'o', 'linestyle': 'solid'}]:
        # 断言使用不支持的关键字参数会引发 TypeError 异常
        with pytest.raises(TypeError):
            ax.spy(np.eye(3, 3), **unsupported_kw)


@check_figures_equal(extensions=["png"])
def test_matshow(fig_test, fig_ref):
    # 使用 mpl20 样式
    mpl.style.use("mpl20")
    # 创建一个随机数组
    a = np.random.rand(32, 32)
    # 在测试图形中添加子图并显示矩阵
    fig_test.add_subplot().matshow(a)
    # 在参考图形中添加子图
    ax_ref = fig_ref.add_subplot()
    # 显示矩阵在参考图形中
    ax_ref.imshow(a)
    # 设置参考图形的 x 轴刻度显示在顶部
    ax_ref.xaxis.tick_top()
    # 设置参考图形的 x 轴刻度位置为两边都显示


@image_comparison(['formatter_ticker_001',
                   'formatter_ticker_002',
                   'formatter_ticker_003',
                   'formatter_ticker_004',
                   'formatter_ticker_005',
                   ],
                  tol=0.031 if platform.machine() == 'arm64' else 0)
def test_formatter_ticker():
    # 导入 JPL 单位模块并注册
    import matplotlib.testing.jpl_units as units
    units.register()

    # 这会影响刻度线的大小（测试问题＃543）
    matplotlib.rcParams['lines.markeredgewidth'] = 30

    # 这个测试基本上是为了看看用户指定的标签是否会被轴的自动标签功能覆盖
    xdata = [x*units.sec for x in range(10)]
    ydata1 = [(1.5*y - 0.5)*units.km for y in range(10)]
    ydata2 = [(1.75*y - 1.0)*units.km for y in range(10)]

    # 创建一个图形对象的轴
    ax = plt.figure().subplots()
    ax.set_xlabel("x-label 001")

    ax = plt.figure().subplots()
    ax.set_xlabel("x-label 001")
    # 绘制蓝色线条，使用秒作为 x 单位
    ax.plot(xdata, ydata1, color='blue', xunits="sec")

    ax = plt.figure().subplots()
    ax.set_xlabel("x-label 001")
    ax.plot(xdata, ydata1, color='blue', xunits="sec")
    ax.set_xlabel("x-label 003")

    ax = plt.figure().subplots()
    ax.plot(xdata, ydata1, color='blue', xunits="sec")
    ax.plot(xdata, ydata2, color='green', xunits="hour")
    ax.set_xlabel("x-label 004")

    # 查看 SF bug 2846058
    # https://sourceforge.net/tracker/?func=detail&aid=2846058&group_id=80706&atid=560720
    ax = plt.figure().subplots()
    ax.plot(xdata, ydata1, color='blue', xunits="sec")
    ax.plot(xdata, ydata2, color='green', xunits="hour")
    ax.set_xlabel("x-label 005")
    # 自动调整视图


def test_funcformatter_auto_formatter():
    def _formfunc(x, pos):
        return ''

    # 创建一个图形对象的轴
    ax = plt.figure().subplots()

    # 断言 x 轴和 y 轴的主要和次要格式化器都是默认的
    assert ax.xaxis.isDefault_majfmt
    assert ax.xaxis.isDefault_minfmt
    assert ax.yaxis.isDefault_majfmt
    assert ax.yaxis.isDefault_minfmt

    # 设置 x 轴的主要格式化器为自定义函数 _formfunc
    ax.xaxis.set_major_formatter(_formfunc)

    # 断言 x 轴的主要格式化器不再是默认的，而其他轴的格式化器仍为默认
    assert not ax.xaxis.isDefault_majfmt
    assert ax.xaxis.isDefault_minfmt
    assert ax.yaxis.isDefault_majfmt
    assert ax.yaxis.isDefault_minfmt

    # 断言 x 轴的主要格式化器确实是 FuncFormatter 类型的，且函数与目标函数格式器一致
    targ_funcformatter = mticker.FuncFormatter(_formfunc)
    assert isinstance(ax.xaxis.get_major_formatter(),
                      mticker.FuncFormatter)
    assert ax.xaxis.get_major_formatter().func == targ_funcformatter.func


def test_strmethodformatter_auto_formatter():
    # 留空，待继续完善
    pass
    # 创建一个格式化字符串，用于在图表中显示坐标轴次要刻度的标签
    formstr = '{x}_{pos}'

    # 创建一个包含一个子图的图表对象
    ax = plt.figure().subplots()

    # 断言 x 轴的主要和次要刻度格式化器是默认的
    assert ax.xaxis.isDefault_majfmt
    assert ax.xaxis.isDefault_minfmt
    assert ax.yaxis.isDefault_majfmt
    assert ax.yaxis.isDefault_minfmt

    # 设置 y 轴的次要刻度格式化器为指定的格式化字符串
    ax.yaxis.set_minor_formatter(formstr)

    # 断言再次确认 x 轴的主要和次要刻度格式化器仍然是默认的
    assert ax.xaxis.isDefault_majfmt
    assert ax.xaxis.isDefault_minfmt
    assert ax.yaxis.isDefault_majfmt
    # 断言确认 y 轴的次要刻度格式化器已经不再是默认的
    assert not ax.yaxis.isDefault_minfmt

    # 创建一个新的格式化器对象，用于设置 y 轴的次要刻度格式化器
    targ_strformatter = mticker.StrMethodFormatter(formstr)

    # 断言确认 y 轴的次要刻度格式化器确实是 StrMethodFormatter 类的实例
    assert isinstance(ax.yaxis.get_minor_formatter(),
                      mticker.StrMethodFormatter)

    # 断言确认 y 轴的次要刻度格式化器的格式与目标格式化器的格式相同
    assert ax.yaxis.get_minor_formatter().fmt == targ_strformatter.fmt
@image_comparison(["twin_axis_locators_formatters"])
def test_twin_axis_locators_formatters():
    vals = np.linspace(0, 1, num=5, endpoint=True)  # 生成一个包含5个元素的数组，范围从0到1
    locs = np.sin(np.pi * vals / 2.0)  # 计算vals数组中每个元素的正弦值

    majl = plt.FixedLocator(locs)  # 创建一个固定位置的主刻度定位器，使用locs作为刻度位置
    minl = plt.FixedLocator([0.1, 0.2, 0.3])  # 创建一个固定位置的次刻度定位器，指定了刻度位置列表

    fig = plt.figure()  # 创建一个新的图形对象
    ax1 = fig.add_subplot(1, 1, 1)  # 在图形中添加一个子图，1行1列的第1个子图
    ax1.plot([0.1, 100], [0, 1])  # 在ax1子图上绘制一条线段
    ax1.yaxis.set_major_locator(majl)  # 设置ax1子图的y轴主刻度定位器
    ax1.yaxis.set_minor_locator(minl)  # 设置ax1子图的y轴次刻度定位器
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%08.2lf'))  # 设置ax1子图的y轴主刻度格式化器
    ax1.yaxis.set_minor_formatter(plt.FixedFormatter(['tricks', 'mind', 'jedi']))  # 设置ax1子图的y轴次刻度格式化器，使用指定标签

    ax1.xaxis.set_major_locator(plt.LinearLocator())  # 设置ax1子图的x轴主刻度定位器为线性定位器
    ax1.xaxis.set_minor_locator(plt.FixedLocator([15, 35, 55, 75]))  # 设置ax1子图的x轴次刻度定位器，指定刻度位置列表
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%05.2lf'))  # 设置ax1子图的x轴主刻度格式化器
    ax1.xaxis.set_minor_formatter(plt.FixedFormatter(['c', '3', 'p', 'o']))  # 设置ax1子图的x轴次刻度格式化器，使用指定标签
    ax1.twiny()  # 创建ax1子图的双x轴
    ax1.twinx()  # 创建ax1子图的双y轴


def test_twinx_cla():
    fig, ax = plt.subplots()  # 创建一个包含主轴的新图形和轴对象
    ax2 = ax.twinx()  # 在ax上创建一个双y轴的新轴对象
    ax3 = ax2.twiny()  # 在ax2上创建一个双x轴的新轴对象
    plt.draw()  # 绘制图形
    assert not ax2.xaxis.get_visible()  # 检查ax2的x轴不可见
    assert not ax2.patch.get_visible()  # 检查ax2的背景不可见
    ax2.cla()  # 清除ax2的轴内容
    ax3.cla()  # 清除ax3的轴内容

    assert not ax2.xaxis.get_visible()  # 再次检查ax2的x轴不可见
    assert not ax2.patch.get_visible()  # 再次检查ax2的背景不可见
    assert ax2.yaxis.get_visible()  # 检查ax2的y轴可见

    assert ax3.xaxis.get_visible()  # 检查ax3的x轴可见
    assert not ax3.patch.get_visible()  # 检查ax3的背景不可见
    assert not ax3.yaxis.get_visible()  # 检查ax3的y轴不可见

    assert ax.xaxis.get_visible()  # 检查ax的x轴可见
    assert ax.patch.get_visible()  # 检查ax的背景可见
    assert ax.yaxis.get_visible()  # 检查ax的y轴可见


@pytest.mark.parametrize('twin', ('x', 'y'))
def test_twin_units(twin):
    axis_name = f'{twin}axis'  # 设置轴名称字符串变量
    twin_func = f'twin{twin}'  # 设置双轴函数字符串变量

    a = ['0', '1']  # 创建字符串列表a
    b = ['a', 'b']  # 创建字符串列表b

    fig = Figure()  # 创建一个新的图形对象
    ax1 = fig.subplots()  # 在图形中创建一个子图
    ax1.plot(a, b)  # 在ax1子图上绘制a和b的图形
    assert getattr(ax1, axis_name).units is not None  # 断言ax1的指定轴单位不为None
    ax2 = getattr(ax1, twin_func)()  # 调用ax1的双轴函数，返回新的双轴对象
    assert getattr(ax2, axis_name).units is not None  # 断言ax2的指定轴单位不为None
    assert getattr(ax2, axis_name).units is getattr(ax1, axis_name).units  # 断言ax2和ax1的指定轴单位相同


@pytest.mark.parametrize('twin', ('x', 'y'))
@check_figures_equal(extensions=['png'], tol=0.19)
def test_twin_logscale(fig_test, fig_ref, twin):
    twin_func = f'twin{twin}'  # 设置双轴函数字符串变量，用于测试twinx或twiny
    set_scale = f'set_{twin}scale'  # 设置轴比例尺函数字符串变量
    x = np.arange(1, 100)  # 创建一个包含1到99的数组

    # 在双轴之后更改比例尺
    ax_test = fig_test.add_subplot(2, 1, 1)  # 在测试图中添加一个子图
    ax_twin = getattr(ax_test, twin_func)()  # 在ax_test上创建一个双轴对象
    getattr(ax_test, set_scale)('log')  # 设置ax_test的指定轴为对数比例尺
    ax_twin.plot(x, x)  # 在ax_twin上绘制x和x

    # 在更改比例尺之后创建双轴
    ax_test = fig_test.add_subplot(2, 1, 2)  # 在测试图中添加另一个子图
    getattr(ax_test, set_scale)('log')  # 设置ax_test的指定轴为对数比例尺
    ax_twin = getattr(ax_test, twin_func)()  # 在ax_test上创建一个双轴对象
    ax_twin.plot(x, x)  # 在ax_twin上绘制x和x
    # 对于列表 [1, 2] 中的每个元素 i，创建一个子图并添加到 fig_ref 中
    for i in [1, 2]:
        # 在 fig_ref 中添加一个子图，布局为 2 行 1 列，当前子图编号为 i
        ax_ref = fig_ref.add_subplot(2, 1, i)
        # 调用 ax_ref 对象的方法，根据 set_scale 指定的比例设置对数坐标轴
        getattr(ax_ref, set_scale)('log')
        # 在 ax_ref 子图上绘制简单的直线图，x 是横坐标和纵坐标的数据
        ax_ref.plot(x, x)

        # 这是一种修复方法，因为双重绘制的坐标轴会导致框架重复绘制。
        # 当问题修复后，可以删除这部分代码。
        Path = matplotlib.path.Path
        # 向 fig_ref 中添加一个路径补丁对象，用于绘制自定义的图形路径
        fig_ref.add_artist(
            matplotlib.patches.PathPatch(
                # 定义一个简单的闭合路径，描述一个矩形
                Path([[0, 0], [0, 1],
                      [0, 1], [1, 1],
                      [1, 1], [1, 0],
                      [1, 0], [0, 0]],
                     [Path.MOVETO, Path.LINETO] * 4),
                # 指定路径补丁的坐标系转换为当前子图的坐标系
                transform=ax_ref.transAxes,
                facecolor='none',
                # 设置路径补丁的边缘颜色为当前图形属性中定义的坐标轴边缘颜色
                edgecolor=mpl.rcParams['axes.edgecolor'],
                # 设置路径补丁的线宽为当前图形属性中定义的坐标轴线宽
                linewidth=mpl.rcParams['axes.linewidth'],
                # 设置路径补丁的线帽样式为 projecting
                capstyle='projecting'))

    # 清除 fig_test 图形对象中的刻度和标题
    remove_ticks_and_titles(fig_test)
    # 清除 fig_ref 图形对象中的刻度和标题
    remove_ticks_and_titles(fig_ref)
@image_comparison(['twin_autoscale.png'],
                  tol=0.009 if platform.machine() == 'arm64' else 0)
def test_twinx_axis_scales():
    # 创建测试函数，比较生成的图像与预期图像是否匹配
    x = np.array([0, 0.5, 1])
    y = 0.5 * x
    x2 = np.array([0, 1, 2])
    y2 = 2 * x2

    # 创建新的图形对象
    fig = plt.figure()
    # 添加坐标轴对象，手动关闭自动缩放
    ax = fig.add_axes((0, 0, 1, 1), autoscalex_on=False, autoscaley_on=False)
    # 在第一个坐标轴上绘制蓝色线条
    ax.plot(x, y, color='blue', lw=10)

    # 创建与第一个坐标轴共享 x 轴的第二个坐标轴
    ax2 = plt.twinx(ax)
    # 在第二个坐标轴上绘制红色虚线
    ax2.plot(x2, y2, 'r--', lw=5)

    # 设置第一个坐标轴和第二个坐标轴的边距为零
    ax.margins(0, 0)
    ax2.margins(0, 0)


def test_twin_inherit_autoscale_setting():
    # 测试共享 x 轴和 y 轴的坐标轴对象自动缩放设置的继承
    fig, ax = plt.subplots()
    # 创建与主坐标轴共享 x 轴的新坐标轴对象
    ax_x_on = ax.twinx()
    # 关闭主坐标轴的自动缩放
    ax.set_autoscalex_on(False)
    # 创建与主坐标轴共享 x 轴的新坐标轴对象
    ax_x_off = ax.twinx()

    # 断言新坐标轴对象是否继承了主坐标轴的自动缩放设置
    assert ax_x_on.get_autoscalex_on()
    assert not ax_x_off.get_autoscalex_on()

    # 创建与主坐标轴共享 y 轴的新坐标轴对象
    ax_y_on = ax.twiny()
    # 关闭主坐标轴的自动缩放
    ax.set_autoscaley_on(False)
    # 创建与主坐标轴共享 y 轴的新坐标轴对象
    ax_y_off = ax.twiny()

    # 断言新坐标轴对象是否继承了主坐标轴的自动缩放设置
    assert ax_y_on.get_autoscaley_on()
    assert not ax_y_off.get_autoscaley_on()


def test_inverted_cla():
    # 测试在清除图形和绘制不同类型图形时坐标轴的反转状态
    # GitHub PR #5450. 设置自动缩放应重置坐标轴为非反转状态。
    fig = plt.figure(0)
    ax = fig.gca()
    # 1. 测试新坐标轴默认不反转
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    img = np.random.random((100, 100))
    ax.imshow(img)
    # 2. 测试绘制图像后 y 轴反转
    assert not ax.xaxis_inverted()
    assert ax.yaxis_inverted()
    # 3. 清除后绘制线条，坐标轴不反转
    ax.cla()
    x = np.linspace(0, 2*np.pi, 100)
    ax.plot(x, np.cos(x))
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()

    # 4. 自动缩放不应恢复坐标轴为正常状态
    ax.cla()
    ax.imshow(img)
    plt.autoscale()
    assert not ax.xaxis_inverted()
    assert ax.yaxis_inverted()

    for ax in fig.axes:
        ax.remove()
    # 5. 两个共享坐标轴。反转主坐标轴应该反转共享坐标轴；清除主坐标轴应该将共享坐标轴恢复到正常状态。
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212, sharey=ax0)
    ax0.yaxis.set_inverted(True)
    assert ax1.yaxis_inverted()
    ax1.plot(x, np.cos(x))
    ax0.cla()
    assert not ax1.yaxis_inverted()
    ax1.cla()
    # 6. 清除从属坐标轴不应触及限制
    ax0.imshow(img)
    ax1.plot(x, np.cos(x))
    ax1.cla()
    assert ax.yaxis_inverted()

    # 清理
    plt.close(fig)


def test_subclass_clear_cla():
    # 确保 Axes 的子类正确调用 cla/clear 方法。
    # 注意，这里不能使用模拟，因为我们要确保超类的回退不会递归。

    with pytest.warns(PendingDeprecationWarning,
                      match='Overriding `Axes.cla`'):
        class ClaAxes(Axes):
            def cla(self):
                nonlocal called
                called = True
    # 使用 pytest 的 warns 方法捕获 PendingDeprecationWarning 警告，并匹配字符串 'Overriding `Axes.cla`'
    with pytest.warns(PendingDeprecationWarning,
                      match='Overriding `Axes.cla`'):
        # 定义一个继承自 Axes 的新类 ClaSuperAxes
        class ClaSuperAxes(Axes):
            # 重写 cla 方法
            def cla(self):
                # 使用 nonlocal 关键字声明 called 变量为非局部变量
                nonlocal called
                # 将 called 设为 True
                called = True
                # 调用父类 Axes 的 cla 方法
                super().cla()

    # 定义一个继承自 ClaAxes 的新类 SubClaAxes
    class SubClaAxes(ClaAxes):
        pass

    # 定义一个继承自 Axes 的新类 ClearAxes
    class ClearAxes(Axes):
        # 定义一个新的 clear 方法
        def clear(self):
            # 使用 nonlocal 关键字声明 called 变量为非局部变量
            nonlocal called
            # 将 called 设为 True
            called = True

    # 定义一个继承自 Axes 的新类 ClearSuperAxes
    class ClearSuperAxes(Axes):
        # 重写 clear 方法
        def clear(self):
            # 使用 nonlocal 关键字声明 called 变量为非局部变量
            nonlocal called
            # 将 called 设为 True
            called = True
            # 调用父类 Axes 的 clear 方法
            super().clear()

    # 定义一个继承自 ClearAxes 的新类 SubClearAxes
    class SubClearAxes(ClearAxes):
        pass

    # 创建一个新的 Figure 对象
    fig = Figure()
    # 遍历列表中的每一个 axes_class 类型
    for axes_class in [ClaAxes, ClaSuperAxes, SubClaAxes,
                       ClearAxes, ClearSuperAxes, SubClearAxes]:
        # 初始化 called 变量为 False
        called = False
        # 使用 axes_class 类型和 fig 对象创建一个 axes 对象 ax
        ax = axes_class(fig, [0, 0, 1, 1])
        # 断言 called 已经被设为 True
        # 在 Axes.__init__ 中已经调用了 clear 方法（它是 cla 的别名或子类中的方法）
        assert called

        # 重新将 called 设为 False
        called = False
        # 调用 ax 对象的 cla 方法
        ax.cla()
        # 再次断言 called 已经被设为 True
        assert called
def test_cla_not_redefined_internally():
    # Iterate through all subclasses of Axes
    for klass in Axes.__subclasses__():
        # Check that the attribute 'cla' is not defined in subclasses,
        # except in the test function 'test_subclass_clear_cla'
        if 'test_subclass_clear_cla' not in klass.__qualname__:
            # Assert that 'cla' is not present in the class dictionary
            assert 'cla' not in klass.__dict__


@check_figures_equal(extensions=["png"])
def test_minorticks_on_rcParams_both(fig_test, fig_ref):
    # Set the context for rcParams to enable minor ticks visibility
    with matplotlib.rc_context({"xtick.minor.visible": True,
                                "ytick.minor.visible": True}):
        # Create subplot for testing figure
        ax_test = fig_test.subplots()
        ax_test.plot([0, 1], [0, 1])
    # Create subplot for reference figure
    ax_ref = fig_ref.subplots()
    ax_ref.plot([0, 1], [0, 1])
    # Enable minor ticks on the reference subplot
    ax_ref.minorticks_on()


@image_comparison(["autoscale_tiny_range"], remove_text=True)
def test_autoscale_tiny_range():
    # GitHub pull request #904
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2)
    for i, ax in enumerate(axs.flat):
        # Generate y-values that decrease exponentially
        y1 = 10**(-11 - i)
        # Plot a line on each subplot
        ax.plot([0, 1], [1, 1 + y1])


@mpl.style.context('default')
def test_autoscale_tight():
    # Test autoscale behavior with tight limits
    fig, ax = plt.subplots(1, 1)
    ax.plot([1, 2, 3, 4])
    # Enable autoscale with tight limits for x and y axes
    ax.autoscale(enable=True, axis='x', tight=False)
    ax.autoscale(enable=True, axis='y', tight=True)
    # Assert that the x-axis and y-axis limits are as expected
    assert_allclose(ax.get_xlim(), (-0.15, 3.15))
    assert_allclose(ax.get_ylim(), (1.0, 4.0))

    # Check that autoscale is on for both axes
    assert ax.get_autoscalex_on()
    assert ax.get_autoscaley_on()
    assert ax.get_autoscale_on()
    # Set enable to None and ensure limits remain unchanged
    ax.autoscale(enable=None)
    assert_allclose(ax.get_xlim(), (-0.15, 3.15))
    assert_allclose(ax.get_ylim(), (1.0, 4.0))
    # Autoscale should still be on after setting enable to None
    assert ax.get_autoscalex_on()
    assert ax.get_autoscaley_on()
    assert ax.get_autoscale_on()


@mpl.style.context('default')
def test_autoscale_log_shared():
    # Related to GitHub issue #7587
    # Create an array starting from zero to trigger _minpos handling
    x = np.arange(100, dtype=float)
    # Create a subplot grid with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # Plot on log-log and semilog-x scales
    ax1.loglog(x, x)
    ax2.semilogx(x, x)
    # Enable autoscale with tight limits for both subplots
    ax1.autoscale(tight=True)
    ax2.autoscale(tight=True)
    plt.draw()
    # Assert that the limits of both axes are as expected
    lims = (x[1], x[-1])
    assert_allclose(ax1.get_xlim(), lims)
    assert_allclose(ax1.get_ylim(), lims)
    assert_allclose(ax2.get_xlim(), lims)
    assert_allclose(ax2.get_ylim(), (x[0], x[-1]))


@mpl.style.context('default')
def test_use_sticky_edges():
    # Test sticky edges behavior with imshow plot
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]], origin='lower')
    # Assert initial xlim and ylim
    assert_allclose(ax.get_xlim(), (-0.5, 1.5))
    assert_allclose(ax.get_ylim(), (-0.5, 1.5))
    # Disable use of sticky edges and perform autoscale
    ax.use_sticky_edges = False
    ax.autoscale()
    # Assert xlim and ylim with expanded margins
    xlim = (-0.5 - 2 * ax._xmargin, 1.5 + 2 * ax._xmargin)
    ylim = (-0.5 - 2 * ax._ymargin, 1.5 + 2 * ax._ymargin)
    assert_allclose(ax.get_xlim(), xlim)
    assert_allclose(ax.get_ylim(), ylim)
    # Re-enable use of sticky edges and perform autoscale
    ax.use_sticky_edges = True
    ax.autoscale()
    # Assert xlim and ylim return to initial values
    assert_allclose(ax.get_xlim(), (-0.5, 1.5))
    assert_allclose(ax.get_ylim(), (-0.5, 1.5))


@check_figures_equal(extensions=["png"])
# 测试共享轴的粘性边缘效果，确保无论设置在主轴还是从轴上，都能正常工作。
def test_sticky_shared_axes(fig_test, fig_ref):
    # 创建一个 3x5 的数组 Z
    Z = np.arange(15).reshape(3, 5)

    # 在测试图表中添加子图1，ax0为主轴
    ax0 = fig_test.add_subplot(211)
    # 在测试图表中添加子图2，ax1为从轴，与ax0共享x轴
    ax1 = fig_test.add_subplot(212, sharex=ax0)
    # 在从轴上绘制色块图
    ax1.pcolormesh(Z)

    # 在参考图表中添加子图2，ax0为从轴
    ax0 = fig_ref.add_subplot(212)
    # 在参考图表中添加子图1，ax1为主轴，与ax0共享x轴
    ax1 = fig_ref.add_subplot(211, sharex=ax0)
    # 在主轴上绘制色块图
    ax0.pcolormesh(Z)


# 测试粘性容差的函数
@image_comparison(['sticky_tolerance.png'], remove_text=True, style="mpl20")
def test_sticky_tolerance():
    # 创建一个2x2的图表
    fig, axs = plt.subplots(2, 2)

    # 设置柱状图的宽度
    width = .1

    # 绘制第一个子图中的柱状图
    axs.flat[0].bar(x=0, height=width, bottom=20000.6)
    axs.flat[0].bar(x=1, height=width, bottom=20000.1)

    # 绘制第二个子图中的柱状图
    axs.flat[1].bar(x=0, height=-width, bottom=20000.6)
    axs.flat[1].bar(x=1, height=-width, bottom=20000.1)

    # 绘制第三个子图中的水平柱状图
    axs.flat[2].barh(y=0, width=-width, left=-20000.6)
    axs.flat[2].barh(y=1, width=-width, left=-20000.1)

    # 绘制第四个子图中的水平柱状图
    axs.flat[3].barh(y=0, width=width, left=-20000.6)
    axs.flat[3].barh(y=1, width=width, left=-20000.1)


# 测试stem()函数的参数个数错误
def test_nargs_stem():
    with pytest.raises(TypeError, match='0 were given'):
        # 调用stem()函数，期望引发TypeError异常，提示未给定参数
        plt.stem()


# 测试legend()函数的参数个数错误
def test_nargs_legend():
    with pytest.raises(TypeError, match='3 were given'):
        ax = plt.subplot()
        # 调用legend()函数，期望引发TypeError异常，提示给定了3个参数，但只允许0-2个参数
        ax.legend(['First'], ['Second'], 3)


# 测试pcolorfast()函数的参数个数错误
def test_nargs_pcolorfast():
    with pytest.raises(TypeError, match='2 were given'):
        ax = plt.subplot()
        # 调用pcolorfast()函数，期望引发TypeError异常，提示给定了2个参数，但只允许1个或3个参数
        ax.pcolorfast([(0, 1), (0, 2)], [[1, 2, 3], [1, 2, 3]])


# 测试基本的annotate()函数
@image_comparison(['offset_points'], remove_text=True)
def test_basic_annotate():
    # 设置一些数据
    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2.0*np.pi * t)

    # 创建图表和子图
    fig = plt.figure()
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 5), ylim=(-3, 5))
    line, = ax.plot(t, s, lw=3, color='purple')

    # 在图上添加注释，显示“local max”文本，指向(3, 1)坐标，偏移(3, 3)像素
    ax.annotate('local max', xy=(3, 1), xycoords='data',
                xytext=(3, 3), textcoords='offset points')


# 测试ax.arrow的简单图像
@image_comparison(['arrow_simple.png'], remove_text=True)
def test_arrow_simple():
    # 创建一组离散值的参数组合
    length_includes_head = (True, False)
    shape = ('full', 'left', 'right')
    head_starts_at_zero = (True, False)
    # 创建参数的笛卡尔积
    kwargs = product(length_includes_head, shape, head_starts_at_zero)

    # 创建3x4的图表
    fig, axs = plt.subplots(3, 4)
    # 遍历 axs.flat 中的每个子图和 kwargs 中的每个元组
    for i, (ax, kwarg) in enumerate(zip(axs.flat, kwargs)):
        # 设置当前子图的 X 轴和 Y 轴范围为 -2 到 2
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        # 解包 kwarg 元组，获取箭头绘制所需的参数
        (length_includes_head, shape, head_starts_at_zero) = kwarg
        
        # 计算当前箭头的角度 theta，每个箭头间隔为 2π/12
        theta = 2 * np.pi * i / 12
        
        # 在当前子图上绘制箭头
        ax.arrow(0, 0, np.sin(theta), np.cos(theta),
                 width=theta/100,
                 length_includes_head=length_includes_head,
                 shape=shape,
                 head_starts_at_zero=head_starts_at_zero,
                 head_width=theta / 10,
                 head_length=theta / 10)
def test_arrow_empty():
    _, ax = plt.subplots()
    # 创建一个空的FancyArrow对象
    ax.arrow(0, 0, 0, 0, head_length=0)


def test_arrow_in_view():
    _, ax = plt.subplots()
    # 在图中绘制一个箭头，起点(1, 1)，终点(2, 2)
    ax.arrow(1, 1, 1, 1)
    # 断言X轴的范围是否为(0.8, 2.2)
    assert ax.get_xlim() == (0.8, 2.2)
    # 断言Y轴的范围是否为(0.8, 2.2)
    assert ax.get_ylim() == (0.8, 2.2)


def test_annotate_default_arrow():
    # 检查是否可以使用默认属性创建注释箭头
    fig, ax = plt.subplots()
    # 创建一个带箭头的注释"foo"，箭头使用默认属性
    ann = ax.annotate("foo", (0, 1), xytext=(2, 3))
    # 断言注释对象的箭头是否为None
    assert ann.arrow_patch is None
    # 创建一个带箭头的注释"foo"，箭头属性为空字典
    ann = ax.annotate("foo", (0, 1), xytext=(2, 3), arrowprops={})
    # 断言注释对象的箭头是否不为None
    assert ann.arrow_patch is not None


def test_annotate_signature():
    """检查Axes.annotate()方法的签名是否与Annotation类的签名匹配。"""
    fig, ax = plt.subplots()
    # 获取Axes.annotate()方法的参数签名
    annotate_params = inspect.signature(ax.annotate).parameters
    # 获取Annotation类的参数签名
    annotation_params = inspect.signature(mtext.Annotation).parameters
    # 断言两者的参数名列表是否一致
    assert list(annotate_params.keys()) == list(annotation_params.keys())
    # 逐一断言Axes.annotate()方法和Annotation类的每个参数是否一致
    for p1, p2 in zip(annotate_params.values(), annotation_params.values()):
        assert p1 == p2


@image_comparison(['fill_units.png'], savefig_kwarg={'dpi': 60})
def test_fill_units():
    import matplotlib.testing.jpl_units as units
    units.register()

    # 生成一些数据
    t = units.Epoch("ET", dt=datetime.datetime(2009, 4, 27))
    value = 10.0 * units.deg
    day = units.Duration("ET", 24.0 * 60.0 * 60.0)
    dt = np.arange('2009-04-27', '2009-04-29', dtype='datetime64[D]')
    dtn = mdates.date2num(dt)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # 在第一个子图中绘制数据
    ax1.plot([t], [value], yunits='deg', color='red')
    ind = [0, 0, 1, 1]
    # 在第一个子图中填充区域，X轴使用日期，Y轴使用角度
    ax1.fill(dtn[ind], [0.0, 0.0, 90.0, 0.0], 'b')

    # 在第二个子图中绘制数据
    ax2.plot([t], [value], yunits='deg', color='red')
    # 在第二个子图中填充区域，使用时间间隔作为X轴单位，角度作为Y轴单位
    ax2.fill([t, t, t + day, t + day],
             [0.0, 0.0, 90.0, 0.0], 'b')

    # 在第三个子图中绘制数据
    ax3.plot([t], [value], yunits='deg', color='red')
    # 在第三个子图中填充区域，X轴使用日期，Y轴使用角度，使用角度单位
    ax3.fill(dtn[ind],
             [0 * units.deg, 0 * units.deg, 90 * units.deg, 0 * units.deg],
             'b')

    # 在第四个子图中绘制数据
    ax4.plot([t], [value], yunits='deg', color='red')
    # 在第四个子图中填充区域，使用时间间隔作为X轴单位，角度作为Y轴单位，指定填充颜色为蓝色
    ax4.fill([t, t, t + day, t + day],
             [0 * units.deg, 0 * units.deg, 90 * units.deg, 0 * units.deg],
             facecolor="blue")
    # 自动调整日期格式
    fig.autofmt_xdate()


def test_plot_format_kwarg_redundant():
    # 测试当格式化参数冗余定义时是否会发出警告
    with pytest.warns(UserWarning, match="marker .* redundantly defined"):
        plt.plot([0], [0], 'o', marker='x')
    with pytest.warns(UserWarning, match="linestyle .* redundantly defined"):
        plt.plot([0], [0], '-', linestyle='--')
    with pytest.warns(UserWarning, match="color .* redundantly defined"):
        plt.plot([0], [0], 'r', color='blue')
    # 烟雾测试：不应该发出警告
    plt.errorbar([0], [0], fmt='none', color='blue')


@check_figures_equal(extensions=["png"])
def test_errorbar_dashes(fig_test, fig_ref):
    x = [1, 2, 3, 4]
    y = np.sin(x)

    ax_ref = fig_ref.gca()
    ax_test = fig_test.gca()

    # 在参考图中绘制误差条，并设置虚线样式
    line, *_ = ax_ref.errorbar(x, y, xerr=np.abs(y), yerr=np.abs(y))
    line.set_dashes([2, 2])
    # 在 ax_test 上绘制带有误差条的误差线图
    ax_test.errorbar(x, y, xerr=np.abs(y), yerr=np.abs(y), dashes=[2, 2])
# 定义一个测试函数，用于测试 errorbar 函数在地图视图关键字参数下的行为
def test_errorbar_mapview_kwarg():
    # 创建一个包含10个键值对的字典 D，键和值相同
    D = {ii: ii for ii in range(10)}
    # 创建一个包含单个子图的图形对象和轴对象
    fig, ax = plt.subplots()
    # 使用 errorbar 函数在轴上绘制误差条，x 值为字典 D 的键，y 值为字典 D 的值，xerr 值为字典 D 的值

@image_comparison(['single_point', 'single_point'])
def test_single_point():
    # 问题 #1796：确保 lines.marker 不影响网格
    # 设置全局参数，设置线的标记为 'o'，开启坐标轴网格
    matplotlib.rcParams['lines.marker'] = 'o'
    matplotlib.rcParams['axes.grid'] = True

    # 创建包含两个子图的图形对象和轴对象
    fig, (ax1, ax2) = plt.subplots(2)
    # 在第一个子图上绘制一个单点，坐标为 (0, 0)
    ax1.plot([0], [0], 'o')
    # 在第二个子图上绘制一个单点，坐标为 (1, 1)
    ax2.plot([1], [1], 'o')

    # 使用上面的测试用例再次测试带标签数据的绘制
    # 创建包含两个子图的图形对象和轴对象
    fig, (ax1, ax2) = plt.subplots(2)
    # 使用数据字典绘制点，数据包含键 'a' 和 'b'，分别对应值 [0] 和 [1]
    ax1.plot('a', 'a', 'o', data=data)
    ax2.plot('b', 'b', 'o', data=data)


@image_comparison(['single_date.png'], style='mpl20')
def test_single_date():

    # 使用先前的默认设置以匹配现有的基准图像
    # 设置全局参数，设置坐标轴格式限制
    plt.rcParams['axes.formatter.limits'] = -7, 7
    # 将 datetime 对象转换为 matplotlib 支持的时间格式
    dt = mdates.date2num(np.datetime64('0000-12-31'))

    time1 = [721964.0]
    data1 = [-65.54]

    # 创建包含两个子图的图形对象和轴对象
    fig, ax = plt.subplots(2, 1)
    # 使用 plot_date 函数在第一个子图上绘制日期-数据图，数据点为 ('time1' + 'dt', 'data1')，标记为 'o'，颜色为红色
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        ax[0].plot_date(time1 + dt, data1, 'o', color='r')
    # 在第二个子图上绘制普通的日期-数据图，数据点为 ('time1', 'data1')，标记为 'o'，颜色为红色
    ax[1].plot(time1, data1, 'o', color='r')


@check_figures_equal(extensions=["png"])
def test_shaped_data(fig_test, fig_ref):
    # 创建一个包含10个元素的一维数组，并将其重新形状为包含单行的二维数组
    row = np.arange(10).reshape((1, -1))
    # 创建一个包含10个元素的一维数组，并将其重新形状为包含单列的二维数组
    col = np.arange(0, 100, 10).reshape((-1, 1))

    # 创建包含两个子图的图形对象和轴对象
    axs = fig_test.subplots(2)
    # 在第一个子图上绘制数据，但实际上不会绘制任何内容（列是单个点）
    axs[0].plot(row)
    # 在第二个子图上绘制数据，效果与绘制一维数据相同
    axs[1].plot(col)

    # 创建包含两个子图的图形对象和轴对象
    axs = fig_ref.subplots(2)
    # 设置第一个子图的 x 轴限制和 y 轴限制
    axs[0].set(xlim=(-.06, .06), ylim=(0, 9))
    # 在第二个子图上绘制扁平化后的列数据
    axs[1].plot(col.ravel())


def test_structured_data():
    # 支持结构化数据
    # 创建一个包含两个元素的结构化数组，字段为 'ones' 和 'twos'，类型为 float
    pts = np.array([(1, 1), (2, 2)], dtype=[("ones", float), ("twos", float)])

    # 创建一个包含两个子图的图形对象和轴对象
    axs = plt.figure().subplots(2)
    # 使用结构化数组中的字段名绘制点，字段 'ones' 对应 x 坐标，'twos' 对应 y 坐标
    axs[0].plot("ones", "twos", data=pts)
    axs[1].plot("ones", "twos", "r", data=pts)


@image_comparison(['aitoff_proj'], extensions=["png"],
                  remove_text=True, style='mpl20')
def test_aitoff_proj():
    """
    测试 aitoff 投影
    参考链接: https://github.com/matplotlib/matplotlib/pull/14451
    """
    # 创建从 -π 到 π 等间距的20个点的数组
    x = np.linspace(-np.pi, np.pi, 20)
    # 创建从 -π/2 到 π/2 等间距的20个点的数组
    y = np.linspace(-np.pi / 2, np.pi / 2, 20)
    # 创建网格 X 和 Y
    X, Y = np.meshgrid(x, y)

    # 创建具有 aitoff 投影的图形对象和轴对象，设置图形尺寸
    fig, ax = plt.subplots(figsize=(8, 4.2),
                           subplot_kw=dict(projection="aitoff"))
    # 绘制散点图，使用 'o' 标记，大小为4
    ax.grid()
    ax.plot(X.flat, Y.flat, 'o', markersize=4)


@image_comparison(['axvspan_epoch'])
def test_axvspan_epoch():
    # 导入 JPL 单位测试模块
    import matplotlib.testing.jpl_units as units
    # 注册单位
    units.register()

    # 生成一些数据
    # 创建 Epoch 对象 t0 和 tf，分别表示 2009年1月21日 和 2009年1月22日
    t0 = units.Epoch("ET", dt=datetime.datetime(2009, 1, 21))
    tf = units.Epoch("ET", dt=datetime.datetime(2009, 1, 22))
    # 创建 Duration 对象 dt，表示一天的时间间隔
    dt = units.Duration("ET", units.day.convert("sec"))

    # 获取当前图形的当前轴对象
    ax = plt.gca()
    # 在轴上绘制垂直区间，从 t0 到 tf，面颜色为蓝色，透明度为0.25
    ax.axvspan(t0, tf, facecolor="blue", alpha=0.25)
    # 设置 x 轴限制，从 t0 减去 5.0*dt，到 tf 加上 5.0*dt
    ax.set_xlim(t0 - 5.0*dt, tf + 5.0*dt)
@image_comparison(['axhspan_epoch'], tol=0.02)
# 定义一个测试函数，比较生成的图像与参考图的差异
def test_axhspan_epoch():
    import matplotlib.testing.jpl_units as units  # 导入JPL单位测试模块
    units.register()  # 注册单位

    # 生成一些数据
    t0 = units.Epoch("ET", dt=datetime.datetime(2009, 1, 21))  # 定义起始时间
    tf = units.Epoch("ET", dt=datetime.datetime(2009, 1, 22))  # 定义结束时间
    dt = units.Duration("ET", units.day.convert("sec"))  # 定义时间间隔

    ax = plt.gca()  # 获取当前图形的坐标轴
    ax.axhspan(t0, tf, facecolor="blue", alpha=0.25)  # 在y轴上创建水平带
    ax.set_ylim(t0 - 5.0*dt, tf + 5.0*dt)  # 设置y轴的上下限


@image_comparison(['hexbin_extent.png', 'hexbin_extent.png'], remove_text=True)
# 定义一个测试函数，比较生成的图像与参考图的差异
def test_hexbin_extent():
    # 此测试暴露了sf bug 2856228

    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    data = (np.arange(2000) / 2000).reshape((2, 1000))  # 生成数据
    x, y = data  # 将数据拆分为x和y坐标

    ax.hexbin(x, y, extent=[.1, .3, .6, .7])  # 绘制六边形bin图，指定数据范围

    # 从上面重用测试用例进行标记数据测试
    data = {"x": x, "y": y}  # 创建带有标签数据的字典

    fig, ax = plt.subplots()  # 创建另一个新的图形和坐标轴
    ax.hexbin("x", "y", extent=[.1, .3, .6, .7], data=data)  # 使用标签数据绘制六边形bin图


def test_hexbin_bad_extents():
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    data = (np.arange(20) / 20).reshape((2, 10))  # 生成数据
    x, y = data  # 将数据拆分为x和y坐标

    with pytest.raises(ValueError, match="In extent, xmax must be greater than xmin"):
        ax.hexbin(x, y, extent=(1, 0, 0, 1))  # 使用错误的范围参数绘制六边形bin图

    with pytest.raises(ValueError, match="In extent, ymax must be greater than ymin"):
        ax.hexbin(x, y, extent=(0, 1, 1, 0))  # 使用错误的范围参数绘制六边形bin图


def test_hexbin_string_norm():
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    hex = ax.hexbin(np.random.rand(10), np.random.rand(10), norm="log", vmin=2, vmax=5)  # 绘制六边形bin图，使用对数归一化

    assert isinstance(hex, matplotlib.collections.PolyCollection)  # 断言返回的对象是多边形集合
    assert isinstance(hex.norm, matplotlib.colors.LogNorm)  # 断言使用的归一化是对数归一化
    assert hex.norm.vmin == 2  # 断言归一化的最小值为2
    assert hex.norm.vmax == 5  # 断言归一化的最大值为5


@image_comparison(['hexbin_empty.png'], remove_text=True)
# 定义一个测试函数，比较生成的图像与参考图的差异
def test_hexbin_empty():
    # 来自#3886：从空数据集创建六边形bin图会引发ValueError异常
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    ax.hexbin([], [])  # 从空数据集创建六边形bin图
    # 来自#23922：从空数据集创建具有对数尺度的六边形bin图会引发ValueError异常
    ax.hexbin([], [], bins='log')  # 从空数据集创建具有对数尺度的六边形bin图
    # 来自#27103：当传递空数据时，np.max函数会引发错误
    ax.hexbin([], [], C=[], reduce_C_function=np.max)  # 使用自定义的reduce_C_function参数创建六边形bin图
    # NumPy不会警告字符串比较
    ax.hexbin([], [], bins=np.arange(10))  # 使用NumPy数组作为bins参数创建六边形bin图


def test_hexbin_pickable():
    # 来自#1973：测试能否正确选取六边形bin图集合
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    data = (np.arange(200) / 200).reshape((2, 100))  # 生成数据
    x, y = data  # 将数据拆分为x和y坐标
    hb = ax.hexbin(x, y, extent=[.1, .3, .6, .7], picker=-1)  # 绘制六边形bin图，设置picker属性为-1
    mouse_event = SimpleNamespace(x=400, y=300)  # 创建模拟鼠标事件
    assert hb.contains(mouse_event)[0]  # 断言六边形bin图能够包含鼠标事件的位置


@image_comparison(['hexbin_log.png'], style='mpl20')
# 定义一个测试函数，比较生成的图像与参考图的差异，使用mpl20风格
def test_hexbin_log():
    # 问题#1636：测试对数尺度的六边形bin图和对数尺度的颜色条

    # 当重新生成此测试图像时，请删除此行。
    plt.rcParams['pcolormesh.snap'] = False  # 设置pcolormesh.snap参数为False，防止图像快照

    np.random.seed(19680801)  # 设定随机数种子
    n = 100000  # 数据点数量
    x = np.random.standard_normal(n)  # 生成标准正态分布的随机数
    y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)  # 生成y数据
    y = np.power(2, y * 0.5)  # 对y数据应用对数变换

    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    # 使用 hexbin 函数创建二维直方图，并设置 y 轴为对数尺度，使用对数划分的直方图箱
    # marginals=True 表示显示边缘直方图
    # reduce_C_function=np.sum 表示在每个箱中用 np.sum 函数来汇总值
    h = ax.hexbin(x, y, yscale='log', bins='log',
                  marginals=True, reduce_C_function=np.sum)
    
    # 在图上添加颜色条
    plt.colorbar(h)
    
    # 确保直方图的偏移量设置正确
    assert h.get_offsets().shape == (11558, 2)
def test_hexbin_log_offsets():
    # 创建一个等比数列，范围从1到100，共500个点
    x = np.geomspace(1, 100, 500)

    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 在坐标轴上绘制 hexbin 图，使用对数坐标轴，网格大小为2
    h = ax.hexbin(x, x, xscale='log', yscale='log', gridsize=2)
    # 断言 hexbin 的偏移量与预期的二维数组近似相等
    np.testing.assert_almost_equal(
        h.get_offsets(),
        np.array(
            [[0, 0],
             [0, 2],
             [1, 0],
             [1, 2],
             [2, 0],
             [2, 2],
             [0.5, 1],
             [1.5, 1]]))


@image_comparison(["hexbin_linear.png"], style="mpl20", remove_text=True)
def test_hexbin_linear():
    # 设置随机数种子
    np.random.seed(19680801)
    n = 100000
    # 生成标准正态分布的随机数
    x = np.random.standard_normal(n)
    # 生成 y 数据，线性相关于 x，并加上额外的随机噪声
    y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)

    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 在坐标轴上绘制 hexbin 图，使用指定网格大小，计算边缘分布
    ax.hexbin(x, y, gridsize=(10, 5), marginals=True,
              reduce_C_function=np.sum)


def test_hexbin_log_clim():
    # 创建 x, y 数组，分别为 0 到 199，reshape 成两行一百列的数组
    x, y = np.arange(200).reshape((2, 100))
    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 在坐标轴上绘制 hexbin 图，使用对数尺度，设置颜色限制范围
    h = ax.hexbin(x, y, bins='log', vmin=2, vmax=100)
    # 断言 hexbin 的颜色限制与预期值相等
    assert h.get_clim() == (2, 100)


@check_figures_equal(extensions=['png'])
def test_hexbin_mincnt_behavior_upon_C_parameter(fig_test, fig_ref):
    # 见：gh:12926
    # 数据点列表，包含 (x, y) 坐标
    datapoints = [
        (0, 0),
        (0, 0),
        (6, 0),
        (0, 6),
    ]
    # 解压 datapoints 到 X, Y 数组
    X, Y = zip(*datapoints)
    # 创建 C 数组，每个点的值均为 1
    C = [1] * len(X)
    # 定义坐标轴的范围
    extent = [-10., 10, -10., 10]
    # 定义网格大小
    gridsize = (7, 7)

    # 测试图形的坐标轴
    ax_test = fig_test.subplots()
    # 参考图形的坐标轴
    ax_ref = fig_ref.subplots()

    # 在参考图形上绘制 hexbin 图，没有使用 C 参数
    ax_ref.hexbin(
        X, Y,
        extent=extent,
        gridsize=gridsize,
        mincnt=1,
    )
    # 设置参考图形的背景颜色为绿色，以增强对比度

    # 在测试图形上绘制 hexbin 图，使用了 C 参数和自定义的 reduce_C_function
    ax_test.hexbin(
        X, Y,
        C=[1] * len(X),
        reduce_C_function=lambda v: sum(v),
        mincnt=1,
        extent=extent,
        gridsize=gridsize,
    )
    # 设置测试图形的背景颜色为绿色


def test_inverted_limits():
    # 测试 gh:1553
    # 在绘图前调用 invert_xaxis 不应禁用自动缩放，但保持反向方向
    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 反转 x 轴
    ax.invert_xaxis()
    # 绘制折线图
    ax.plot([-5, -3, 2, 4], [1, 2, -3, 5])

    # 断言当前坐标轴的 x 范围与预期相等（反转后）
    assert ax.get_xlim() == (4, -5)
    # 断言当前坐标轴的 y 范围与预期相等
    assert ax.get_ylim() == (-3, 5)
    # 关闭图形

    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 反转 y 轴
    ax.invert_yaxis()
    # 绘制折线图
    ax.plot([-5, -3, 2, 4], [1, 2, -3, 5])

    # 断言当前坐标轴的 x 范围与预期相等
    assert ax.get_xlim() == (-5, 4)
    # 断言当前坐标轴的 y 范围与预期相等（反转后）
    assert ax.get_ylim() == (5, -3)

    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 设置 y 轴为对数尺度，并设置范围
    ax.set_yscale("log")
    ax.set_ylim(10, 1)
    # 断言当前坐标轴的 y 范围与预期相等
    assert ax.get_ylim() == (10, 1)


@image_comparison(['nonfinite_limits'])
def test_nonfinite_limits():
    # 创建 x 数组，从 0 到 e，步长为 0.01
    x = np.arange(0., np.e, 0.01)
    # 禁止 log(0) 的除零警告
    with np.errstate(divide='ignore'):
        # 计算 x 的自然对数
        y = np.log(x)
    # 将 x 的中间一个值设为 NaN
    x[len(x)//2] = np.nan
    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 绘制 x, y 折线图
    ax.plot(x, y)
# 测试空数据情况下绘图函数的行为，确保日期的自动缩放不受影响
def test_limits_empty_data(plot_fun, fig_test, fig_ref):
    # 生成一个包含日期的数组，从"2010-01-01"到"2011-01-01"
    x = np.arange("2010-01-01", "2011-01-01", dtype="datetime64[D]")

    # 创建测试图和参考图的坐标轴
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()

    # 在测试图上调用指定的绘图函数，传入空列表作为数据
    getattr(ax_test, plot_fun)([], [])

    # 在测试图和参考图上，分别使用日期和其索引绘制折线图
    for ax in [ax_test, ax_ref]:
        getattr(ax, plot_fun)(x, range(len(x)), color='C0')


@image_comparison(['imshow', 'imshow'], remove_text=True, style='mpl20')
def test_imshow():
    # 使用旧的默认设置以匹配现有的基准图像
    matplotlib.rcParams['image.interpolation'] = 'nearest'

    # 创建一个 NxN 的图像
    N = 100
    (x, y) = np.indices((N, N))
    x -= N//2
    y -= N//2
    r = np.sqrt(x**2+y**2-x*y)

    # 在新的图中创建一个 imshow 显示图像
    fig, ax = plt.subplots()
    ax.imshow(r)

    # 使用上面的测试用例进行标记数据测试
    data = {"r": r}
    fig, ax = plt.subplots()
    ax.imshow("r", data=data)


@image_comparison(
    ['imshow_clip'], style='mpl20',
    tol=1.24 if platform.machine() in ('aarch64', 'ppc64le', 's390x') else 0)
def test_imshow_clip():
    # 如 Gellule Xg <gellule.xg@free.fr> 报告的原始问题
    # 使用旧的默认设置以匹配现有的基准图像
    matplotlib.rcParams['image.interpolation'] = 'nearest'

    # 创建一个 NxN 的图像
    N = 100
    (x, y) = np.indices((N, N))
    x -= N//2
    y -= N//2
    r = np.sqrt(x**2+y**2-x*y)

    # 在新的图中创建一个 imshow 显示图像
    fig, ax = plt.subplots()

    # 创建一个轮廓图，在 N/4 处提取裁剪路径和变换
    c = ax.contour(r, [N/4])
    clip_path = mtransforms.TransformedPath(c.get_paths()[0], c.get_transform())

    # 使用轮廓图裁剪后的图像绘制 imshow 图像
    ax.imshow(r, clip_path=clip_path)


def test_imshow_norm_vminvmax():
    """当使用 norm 参数时，vmin 和 vmax 应当报错。"""
    a = [[1, 2], [3, 4]]
    ax = plt.axes()
    with pytest.raises(ValueError,
                       match="Passing a Normalize instance simultaneously "
                             "with vmin/vmax is not supported."):
        ax.imshow(a, norm=mcolors.Normalize(-10, 10), vmin=0, vmax=5)


@image_comparison(['polycollection_joinstyle'], remove_text=True)
def test_polycollection_joinstyle():
    # 由 Matthew West 报告的 Bug #2890979
    fig, ax = plt.subplots()
    verts = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
    c = mpl.collections.PolyCollection([verts], linewidths=40)
    ax.add_collection(c)
    ax.set_xbound(0, 3)
    ax.set_ybound(0, 3)


@pytest.mark.parametrize(
    'x, y1, y2', [
        (np.zeros((2, 2)), 3, 3),
        (np.arange(0.0, 2, 0.02), np.zeros((2, 2)), 3),
        (np.arange(0.0, 2, 0.02), 3, np.zeros((2, 2)))
    ], ids=[
        '2d_x_input',
        '2d_y1_input',
        '2d_y2_input'
    ]
)
def test_fill_between_input(x, y1, y2):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.fill_between(x, y1, y2)


@pytest.mark.parametrize(
    # 定义一个元组列表，每个元组包含三个元素：第一个元素是 numpy 的数组，其余两个是数字或者其他数组
    'y, x1, x2', [
        # 第一个元组：包含一个2x2的全零数组，以及两个整数值 3
        (np.zeros((2, 2)), 3, 3),
        # 第二个元组：包含一个从0.0到2.0（不包括2.0）的浮点数数组，一个2x2的全零数组，以及一个整数值 3
        (np.arange(0.0, 2, 0.02), np.zeros((2, 2)), 3),
        # 第三个元组：包含一个从0.0到2.0（不包括2.0）的浮点数数组，一个整数值 3，以及一个2x2的全零数组
        (np.arange(0.0, 2, 0.02), 3, np.zeros((2, 2)))
    ], ids=[
        # 每个输入元组对应的标识符字符串
        '2d_y_input',
        '2d_x1_input',
        '2d_x2_input'
    ]
# 定义一个测试函数，用于测试在给定条件下抛出 ValueError 异常
def test_fill_betweenx_input(y, x1, x2):
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 使用 pytest 的上下文管理器检查是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 在坐标轴上使用 fill_betweenx 函数进行填充区域操作
        ax.fill_betweenx(y, x1, x2)


# 创建一个图像对比测试函数，比较带有插值的 fill_between 的输出图像
@image_comparison(['fill_between_interpolate'], remove_text=True,
                  tol=0.012 if platform.machine() == 'arm64' else 0)
def test_fill_between_interpolate():
    # 创建一组均匀分布的 x 值
    x = np.arange(0.0, 2, 0.02)
    # 计算两个不同的正弦波
    y1 = np.sin(2*np.pi*x)
    y2 = 1.2*np.sin(4*np.pi*x)

    # 创建包含两个子图的图形对象
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # 在第一个子图上绘制两条正弦波曲线
    ax1.plot(x, y1, x, y2, color='black')
    # 使用 fill_between 函数填充 y2 >= y1 条件下的区域
    ax1.fill_between(x, y1, y2, where=y2 >= y1, facecolor='white', hatch='/',
                     interpolate=True)
    # 使用 fill_between 函数填充 y2 <= y1 条件下的区域
    ax1.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red',
                     interpolate=True)

    # 测试对掩码数组的支持
    y2 = np.ma.masked_greater(y2, 1.0)
    # 将第一个元素掩码化，测试掩码数组的绘图效果
    y2[0] = np.ma.masked
    # 在第二个子图上绘制两条曲线，其中一个使用掩码数组
    ax2.plot(x, y1, x, y2, color='black')
    # 使用 fill_between 函数填充 y2 >= y1 条件下的区域，使用绿色填充
    ax2.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green',
                     interpolate=True)
    # 使用 fill_between 函数填充 y2 <= y1 条件下的区域，使用红色填充
    ax2.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red',
                     interpolate=True)


# 创建一个图像对比测试函数，比较在递减情况下使用插值的 fill_betweenx 的输出图像
@image_comparison(['fill_between_interpolate_decreasing'],
                  style='mpl20', remove_text=True)
def test_fill_between_interpolate_decreasing():
    # 创建三个数值数组
    p = np.array([724.3, 700, 655])
    t = np.array([9.4, 7, 2.2])
    prof = np.array([7.9, 6.6, 3.8])

    # 创建一个大尺寸的图形对象和坐标轴对象
    fig, ax = plt.subplots(figsize=(9, 9))

    # 在坐标轴上绘制 t 和 p 数组的关系，使用红色
    ax.plot(t, p, 'tab:red')
    # 在坐标轴上绘制 prof 和 p 数组的关系，使用黑色
    ax.plot(prof, p, 'k')

    # 使用 fill_betweenx 函数填充 prof < t 条件下的区域，使用蓝色填充
    ax.fill_betweenx(p, t, prof, where=prof < t,
                     facecolor='blue', interpolate=True, alpha=0.4)
    # 使用 fill_betweenx 函数填充 prof > t 条件下的区域，使用红色填充
    ax.fill_betweenx(p, t, prof, where=prof > t,
                     facecolor='red', interpolate=True, alpha=0.4)

    # 设置坐标轴的 x 轴和 y 轴的范围
    ax.set_xlim(0, 30)
    ax.set_ylim(800, 600)


# 创建一个图像对比测试函数，比较在包含 NaN 值时使用插值的 fill_between 的输出图像
@image_comparison(['fill_between_interpolate_nan'], remove_text=True)
def test_fill_between_interpolate_nan():
    # 测试问题 #18986 的修复
    x = np.arange(10)
    y1 = np.asarray([8, 18, np.nan, 18, 8, 18, 24, 18, 8, 18])
    y2 = np.asarray([18, 11, 8, 11, 18, 26, 32, 30, np.nan, np.nan])

    # 创建一个图形对象和坐标轴对象
    fig, ax = plt.subplots()

    # 在坐标轴上绘制两条曲线，一条使用黑色，一条使用蓝色
    ax.plot(x, y1, c='k')
    ax.plot(x, y2, c='b')
    # 使用 fill_between 函数填充 y2 >= y1 条件下的区域，使用绿色填充
    ax.fill_between(x, y1, y2, where=y2 >= y1, facecolor="green",
                    interpolate=True, alpha=0.5)
    # 使用 fill_between 函数填充 y1 >= y2 条件下的区域，使用红色填充
    ax.fill_between(x, y1, y2, where=y1 >= y2, facecolor="red",
                    interpolate=True, alpha=0.5)


# 创建一个图像对比测试函数，测试对对数对称刻度的支持
@image_comparison(['symlog.pdf'])
def test_symlog():
    # 创建两个数组，分别代表 x 和 y 轴的数据点
    x = np.array([0, 1, 2, 4, 6, 9, 12, 24])
    y = np.array([1000000, 500000, 100000, 100, 5, 0, 0, 0])

    # 创建一个图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制 x 和 y 数组的关系
    ax.plot(x, y)
    # 设置 y 轴的对称对数刻度
    ax.set_yscale('symlog')
    # 设置 x 轴的线性刻度
    ax.set_xscale('linear')
    # 设置 y 轴的范围
    ax.set_ylim(-1, 10000000)


# 创建一个图像对比测试函数，测试对对数对称刻度的支持（不显示文本）
@image_comparison(['symlog2.pdf'], remove_text=True)
def test_symlog2():
    # 省略内容，因为函数尚未实现内容，仅占位而已
    pass
    # 从-50到50生成一个包含浮点数的数组，步长为0.001
    x = np.arange(-50, 50, 0.001)

    # 创建一个包含5个子图的图形对象和对应的轴对象数组
    fig, axs = plt.subplots(5, 1)

    # 对每个子图进行循环处理，设置对应的对数坐标轴和对应的linthresh参数
    for ax, linthresh in zip(axs, [20., 2., 1., 0.1, 0.01]):
        # 在当前轴对象上绘制 x vs x 的图像
        ax.plot(x, x)
        # 设置x轴为对数坐标轴，并指定linthresh参数
        ax.set_xscale('symlog', linthresh=linthresh)
        # 启用网格线
        ax.grid(True)

    # 设置最后一个子图的y轴范围为-0.1到0.1
    axs[-1].set_ylim(-0.1, 0.1)
def test_pcolorargs_5205():
    # 检查在 gh:5205 中发现的问题的简单测试
    x = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    y = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0,
         0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    X, Y = np.meshgrid(x, y)
    Z = np.hypot(X, Y)

    plt.pcolor(Z)  # 绘制基于 Z 的伪彩图
    plt.pcolor(list(Z))  # 绘制基于 Z 列表的伪彩图
    plt.pcolor(x, y, Z[:-1, :-1])  # 绘制基于部分 Z 数据的伪彩图
    plt.pcolor(X, Y, list(Z[:-1, :-1]))  # 绘制基于部分 Z 列表数据的伪彩图


@image_comparison(['pcolormesh'], remove_text=True)
def test_pcolormesh():
    # 当测试图像重新生成时，请移除此行。
    plt.rcParams['pcolormesh.snap'] = False

    n = 12
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n*2)
    X, Y = np.meshgrid(x, y)
    Qx = np.cos(Y) - np.cos(X)
    Qz = np.sin(Y) + np.sin(X)
    Qx = (Qx + 1.1)
    Z = np.hypot(X, Y) / 5
    Z = (Z - Z.min()) / np.ptp(Z)

    # 色彩数组可以包含掩码值：
    Zm = ma.masked_where(np.abs(Qz) < 0.5 * np.max(Qz), Z)

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.pcolormesh(Qx, Qz, Zm[:-1, :-1], lw=0.5, edgecolors='k')  # 绘制网格状伪彩图
    ax2.pcolormesh(Qx, Qz, Zm[:-1, :-1], lw=2, edgecolors=['b', 'w'])  # 绘制带有指定边缘颜色的伪彩图
    ax3.pcolormesh(Qx, Qz, Zm, shading="gouraud")  # 使用 gouraud 着色绘制伪彩图


@image_comparison(['pcolormesh_small'], extensions=["eps"])
def test_pcolormesh_small():
    n = 3
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n*2)
    X, Y = np.meshgrid(x, y)
    Qx = np.cos(Y) - np.cos(X)
    Qz = np.sin(Y) + np.sin(X)
    Qx = (Qx + 1.1)
    Z = np.hypot(X, Y) / 5
    Z = (Z - Z.min()) / np.ptp(Z)
    Zm = ma.masked_where(np.abs(Qz) < 0.5 * np.max(Qz), Z)
    Zm2 = ma.masked_where(Qz < -0.5 * np.max(Qz), Z)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.pcolormesh(Qx, Qz, Zm[:-1, :-1], lw=0.5, edgecolors='k')  # 绘制网格状伪彩图
    ax2.pcolormesh(Qx, Qz, Zm[:-1, :-1], lw=2, edgecolors=['b', 'w'])  # 绘制带有指定边缘颜色的伪彩图
    ax3.pcolormesh(Qx, Qz, Zm, shading="gouraud")  # 使用 gouraud 着色绘制伪彩图
    ax4.pcolormesh(Qx, Qz, Zm2, shading="gouraud")  # 使用 gouraud 着色绘制伪彩图，减少掩码以获得绘图


@image_comparison(['pcolormesh_alpha'], extensions=["png", "pdf"],
                  remove_text=True)
def test_pcolormesh_alpha():
    # 当测试图像重新生成时，请移除此行。
    plt.rcParams['pcolormesh.snap'] = False

    n = 12
    X, Y = np.meshgrid(
        np.linspace(-1.5, 1.5, n),
        np.linspace(-1.5, 1.5, n*2)
    )
    Qx = X
    Qy = Y + np.sin(X)
    Z = np.hypot(X, Y) / 5
    Z = (Z - Z.min()) / np.ptp(Z)
    vir = mpl.colormaps["viridis"].resampled(16)
    # 使用不同 alpha 的颜色映射创建另一个颜色映射
    colors = vir(np.arange(16))
    colors[:, 3] = 0.5 + 0.5*np.sin(np.arange(16))
    cmap = mcolors.ListedColormap(colors)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    for ax in ax1, ax2, ax3, ax4:
        ax.add_patch(mpatches.Rectangle(
            (0, -1.5), 1.5, 3, facecolor=[.7, .1, .1, .5], zorder=0
        ))
    # ax1, ax2: 常量 alpha
    # 在 ax1 中绘制伪彩色图像，使用 vir 颜色映射，alpha 设置为 0.4，使用 flat 渲染方式，层级为 1
    ax1.pcolormesh(Qx, Qy, Z[:-1, :-1], cmap=vir, alpha=0.4, shading='flat', zorder=1)
    
    # 在 ax2 中绘制伪彩色图像，使用 vir 颜色映射，alpha 设置为 0.4，使用 gouraud 渲染方式，层级为 1
    ax2.pcolormesh(Qx, Qy, Z, cmap=vir, alpha=0.4, shading='gouraud', zorder=1)
    
    # 在 ax3 中绘制伪彩色图像，使用指定的 cmap 颜色映射，使用 flat 渲染方式，层级为 1
    ax3.pcolormesh(Qx, Qy, Z[:-1, :-1], cmap=cmap, shading='flat', zorder=1)
    
    # 在 ax4 中绘制伪彩色图像，使用指定的 cmap 颜色映射，使用 gouraud 渲染方式，层级为 1
    ax4.pcolormesh(Qx, Qy, Z, cmap=cmap, shading='gouraud', zorder=1)
@pytest.mark.parametrize("dims,alpha", [(3, 1), (4, 0.5)])
@check_figures_equal(extensions=["png"])
def test_pcolormesh_rgba(fig_test, fig_ref, dims, alpha):
    # 在测试图形中创建子图并绘制彩色网格，使用给定的维度和透明度参数
    ax = fig_test.subplots()
    c = np.ones((5, 6, dims), dtype=float) / 2
    ax.pcolormesh(c)

    # 在参考图形中创建子图并绘制灰度彩色网格，设置颜色映射为灰度，同时设定透明度和颜色范围
    ax = fig_ref.subplots()
    ax.pcolormesh(c[..., 0], cmap="gray", vmin=0, vmax=1, alpha=alpha)


@image_comparison(['pcolormesh_datetime_axis.png'], style='mpl20')
def test_pcolormesh_datetime_axis():
    # 在生成测试图像时，设置pcolormesh的snap属性为False
    plt.rcParams['pcolormesh.snap'] = False

    # 创建一个图形对象并调整其子图的间距和位置
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.98, bottom=.15)
    base = datetime.datetime(2013, 1, 1)
    x = np.array([base + datetime.timedelta(days=d) for d in range(21)])
    y = np.arange(21)
    z1, z2 = np.meshgrid(np.arange(20), np.arange(20))
    z = z1 * z2
    # 绘制四个子图中的第一个，使用pcolormesh函数显示日期时间轴的数据
    plt.subplot(221)
    plt.pcolormesh(x[:-1], y[:-1], z[:-1, :-1])
    # 绘制四个子图中的第二个，使用pcolormesh函数显示日期时间轴的数据
    plt.subplot(222)
    plt.pcolormesh(x, y, z)
    x = np.repeat(x[np.newaxis], 21, axis=0)
    y = np.repeat(y[:, np.newaxis], 21, axis=1)
    # 绘制四个子图中的第三个，使用pcolormesh函数显示日期时间轴的数据
    plt.subplot(223)
    plt.pcolormesh(x[:-1, :-1], y[:-1, :-1], z[:-1, :-1])
    # 绘制四个子图中的第四个，使用pcolormesh函数显示日期时间轴的数据
    plt.subplot(224)
    plt.pcolormesh(x, y, z)
    # 对每个子图的X轴刻度标签进行调整，使其右对齐并以30度角度旋转
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)


@image_comparison(['pcolor_datetime_axis.png'], style='mpl20')
def test_pcolor_datetime_axis():
    # 创建一个图形对象并调整其子图的间距和位置
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.98, bottom=.15)
    base = datetime.datetime(2013, 1, 1)
    x = np.array([base + datetime.timedelta(days=d) for d in range(21)])
    y = np.arange(21)
    z1, z2 = np.meshgrid(np.arange(20), np.arange(20))
    z = z1 * z2
    # 绘制四个子图中的第一个，使用pcolor函数显示日期时间轴的数据
    plt.subplot(221)
    plt.pcolor(x[:-1], y[:-1], z[:-1, :-1])
    # 绘制四个子图中的第二个，使用pcolor函数显示日期时间轴的数据
    plt.subplot(222)
    plt.pcolor(x, y, z)
    x = np.repeat(x[np.newaxis], 21, axis=0)
    y = np.repeat(y[:, np.newaxis], 21, axis=1)
    # 绘制四个子图中的第三个，使用pcolor函数显示日期时间轴的数据
    plt.subplot(223)
    plt.pcolor(x[:-1, :-1], y[:-1, :-1], z[:-1, :-1])
    # 绘制四个子图中的第四个，使用pcolor函数显示日期时间轴的数据
    plt.subplot(224)
    plt.pcolor(x, y, z)
    # 对每个子图的X轴刻度标签进行调整，使其右对齐并以30度角度旋转
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)


def test_pcolorargs():
    n = 12
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n*2)
    X, Y = np.meshgrid(x, y)
    Z = np.hypot(X, Y) / 5

    _, ax = plt.subplots()
    # 测试pcolormesh函数的参数异常情况：传递y作为X轴数据引发TypeError异常
    with pytest.raises(TypeError):
        ax.pcolormesh(y, x, Z)
    # 测试pcolormesh函数的参数异常情况：传递X的转置作为Y轴数据引发TypeError异常
    with pytest.raises(TypeError):
        ax.pcolormesh(X, Y, Z.T)
    # 测试pcolormesh函数的参数异常情况：使用gouraud着色模式引发TypeError异常
    with pytest.raises(TypeError):
        ax.pcolormesh(x, y, Z[:-1, :-1], shading="gouraud")
    # 测试pcolormesh函数的参数异常情况：使用gouraud着色模式引发TypeError异常
    with pytest.raises(TypeError):
        ax.pcolormesh(X, Y, Z[:-1, :-1], shading="gouraud")
    x[0] = np.nan
    # 测试pcolormesh函数的参数异常情况：传递包含NaN值的数组引发ValueError异常
    with pytest.raises(ValueError):
        ax.pcolormesh(x, y, Z[:-1, :-1])
    # 忽略无效值的错误状态，并测试pcolormesh函数的参数异常情况：传递包含无效值的数组引发ValueError异常
    with np.errstate(invalid='ignore'):
        x = np.ma.array(x, mask=(x < 0))
    with pytest.raises(ValueError):
        ax.pcolormesh(x, y, Z[:-1, :-1])
    # 定义 x 和 y 数组，分别表示 x 和 y 坐标轴上的值
    x = [359, 0, 1]
    y = [-10, 10]
    
    # 使用 NumPy 的 meshgrid 函数，生成 x 和 y 坐标的网格矩阵
    X, Y = np.meshgrid(x, y)
    
    # 创建一个与 X 相同形状的全零数组，用作 pcolormesh 函数的 Z 参数
    Z = np.zeros(X.shape)
    
    # 使用 pytest 的 warns 上下文管理器，捕获预期的 UserWarning
    # 匹配字符串 'are not monotonically increasing or decreasing' 的警告信息
    with pytest.warns(UserWarning,
                      match='are not monotonically increasing or decreasing'):
        # 调用 ax 对象的 pcolormesh 方法，绘制网格数据 X, Y，使用 Z 进行填充，自动选择渲染方式
        ax.pcolormesh(X, Y, Z, shading='auto')
def test_pcolormesh_underflow_error():
    """
    Test that underflow errors don't crop up in pcolormesh.  Probably
    a numpy bug (https://github.com/numpy/numpy/issues/25810).
    """
    # 设置 numpy 的错误状态，当浮点数下溢时抛出异常
    with np.errstate(under="raise"):
        # 创建一维数组 x 包含 0 到 2.9 的数据点
        x = np.arange(0, 3, 0.1)
        # 创建一维数组 y 包含 0 到 5.9 的数据点
        y = np.arange(0, 6, 0.1)
        # 生成一个随机数据矩阵，大小为 len(y) x len(x)
        z = np.random.randn(len(y), len(x))
        # 创建一个新的图形和一个子图
        fig, ax = plt.subplots()
        # 绘制一个伪彩色网格，使用 x, y, z 数组
        ax.pcolormesh(x, y, z)


def test_pcolorargs_with_read_only():
    # 创建一个二维数组 x 包含 0 到 5 的数据点
    x = np.arange(6).reshape(2, 3)
    # 创建一个只读的布尔掩码数组 xmask
    xmask = np.broadcast_to([False, True, False], x.shape)  # read-only array
    # 断言 xmask 是只读的
    assert xmask.flags.writeable is False
    # 创建一个带有掩码的 NumPy MaskedArray masked_x
    masked_x = np.ma.array(x, mask=xmask)
    # 绘制带有掩码的伪彩色网格
    plt.pcolormesh(masked_x)

    # 创建一个从 0 到 1 的 10 个数据点的数组 x
    x = np.linspace(0, 1, 10)
    # 创建一个从 0 到 1 的 10 个数据点的数组 y
    y = np.linspace(0, 1, 10)
    # 创建一个网格 X, Y，表示在 x 和 y 上的网格点坐标
    X, Y = np.meshgrid(x, y)
    # 创建一个正弦余弦函数的矩阵 Z
    Z = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    # 创建一个布尔掩码数组 mask，最后一个元素为 True
    mask = np.zeros(10, dtype=bool)
    mask[-1] = True
    # 使用掩码 mask 广播创建掩码数组 mask
    mask = np.broadcast_to(mask, Z.shape)
    # 断言 mask 是只读的
    assert mask.flags.writeable is False
    # 创建一个带有掩码的 NumPy MaskedArray masked_Z
    masked_Z = np.ma.array(Z, mask=mask)
    # 绘制带有掩码的伪彩色网格
    plt.pcolormesh(X, Y, masked_Z)

    # 创建带有掩码的 NumPy MaskedArray masked_X 和 masked_Y
    masked_X = np.ma.array(X, mask=mask)
    masked_Y = np.ma.array(Y, mask=mask)
    # 绘制带有掩码的伪彩色图
    plt.pcolor(masked_X, masked_Y, masked_Z)


@check_figures_equal(extensions=["png"])
def test_pcolornearest(fig_test, fig_ref):
    # 创建一个子图 ax
    ax = fig_test.subplots()
    # 创建一个一维数组 x 包含 0 到 9 的数据点
    x = np.arange(0, 10)
    # 创建一个一维数组 y 包含 0 到 2 的数据点
    y = np.arange(0, 3)
    # 设置随机种子
    np.random.seed(19680801)
    # 生成一个随机数据矩阵 Z，大小为 2 x 9
    Z = np.random.randn(2, 9)
    # 绘制一个使用平坦着色的伪彩色网格
    ax.pcolormesh(x, y, Z, shading='flat')

    # 创建一个子图 ax
    ax = fig_ref.subplots()
    # 创建一个一维数组 x2，包含 x 的中心点
    x2 = x[:-1] + np.diff(x) / 2
    # 创建一个一维数组 y2，包含 y 的中心点
    y2 = y[:-1] + np.diff(y) / 2
    # 绘制一个使用最近着色的伪彩色网格
    ax.pcolormesh(x2, y2, Z, shading='nearest')


@check_figures_equal(extensions=["png"])
def test_pcolornearestunits(fig_test, fig_ref):
    # 创建一个子图 ax
    ax = fig_test.subplots()
    # 创建一个包含日期时间的数组 x
    x = [datetime.datetime.fromtimestamp(x * 3600) for x in range(10)]
    # 创建一个一维数组 y 包含 0 到 2 的数据点
    y = np.arange(0, 3)
    # 设置随机种子
    np.random.seed(19680801)
    # 生成一个随机数据矩阵 Z，大小为 2 x 9
    Z = np.random.randn(2, 9)
    # 绘制一个使用平坦着色的伪彩色网格
    ax.pcolormesh(x, y, Z, shading='flat')

    # 创建一个子图 ax
    ax = fig_ref.subplots()
    # 创建一个包含日期时间的数组 x2，包含 x 的中心点
    x2 = [datetime.datetime.fromtimestamp((x + 0.5) * 3600) for x in range(9)]
    # 创建一个一维数组 y2，包含 y 的中心点
    y2 = y[:-1] + np.diff(y) / 2
    # 绘制一个使用最近着色的伪彩色网格
    ax.pcolormesh(x2, y2, Z, shading='nearest')


def test_pcolorflaterror():
    # 创建一个新的图形和一个子图
    fig, ax = plt.subplots()
    # 创建一个一维数组 x 包含 0 到 8 的数据点
    x = np.arange(0, 9)
    # 创建一个一维数组 y 包含 0 到 2 的数据点
    y = np.arange(0, 3)
    # 设置随机种子
    np.random.seed(19680801)
    # 生成一个随机数据矩阵 Z，大小为 3 x 9
    Z = np.random.randn(3, 9)
    # 在绘制伪彩色网格时，使用 flat 着色，并期望 TypeError 异常
    with pytest.raises(TypeError, match='Dimensions of C'):
        ax.pcolormesh(x, y, Z, shading='flat')


def test_samesizepcolorflaterror():
    # 创建一个新的图形和一个子图
    fig, ax = plt.subplots()
    # 创建一个网格 x, y，包含从 0 到 4 的 5 个数据点和从 0 到 2 的 3 个数据点
    x, y = np.meshgrid(np.arange(5), np.arange(3))
    # 创建一个矩阵 Z，包含 x 和 y 元素的和
    Z = x + y
    # 在绘制伪彩色网格时，使用 flat 着色，并期望 TypeError 异常，错误消息包含 "one smaller than X"
    with pytest.raises(TypeError, match=r".*one smaller than X"):
        ax.pcolormesh(x, y, Z, shading='flat')


@pytest.mark.parametrize('snap', [False, True])
@check_figures_equal(extensions=["png"])
def test_pcolorauto(fig_test, fig_ref, snap):
    # 创建一个子图 ax
    ax = fig_test.subplots()
    # 创建一个一维数组 x 包含 0 到 9 的数据点
    x = np.arange(0, 10)
    # 创建一个一维数组 y 包含 0 到 3 的数据点
    y = np.arange(0, 4)
    # 设置随机种子
    np.random.seed(19680801)
    # 生成一个随机数据矩阵 Z，大小为 3 x 9
    Z = np.random.randn(3, 9)
    # 绘制一个使用默认自动着色的伪彩
    ax.pcolormesh(x, y, Z, snap=snap)



# 在当前坐标系(ax)上绘制一个伪彩色网格，使用给定的x、y坐标和Z数据，snap参数控制对齐方式
ax.pcolormesh(x, y, Z, snap=snap)



    ax = fig_ref.subplots()



# 在fig_ref指定的Figure对象上创建一个新的子图(ax)，并将其赋值给ax变量
ax = fig_ref.subplots()



    x2 = x[:-1] + np.diff(x) / 2
    y2 = y[:-1] + np.diff(y) / 2



# 计算x和y的中心点坐标，将其保存在x2和y2中
x2 = x[:-1] + np.diff(x) / 2
y2 = y[:-1] + np.diff(y) / 2



    ax.pcolormesh(x2, y2, Z, snap=snap)



# 在当前坐标系(ax)上绘制一个伪彩色网格，使用计算得到的中心点坐标x2、y2和Z数据，snap参数控制对齐方式
ax.pcolormesh(x2, y2, Z, snap=snap)
# 比较图像是否匹配，根据平台决定容忍度
@image_comparison(['canonical'], tol=0.02 if platform.machine() == 'arm64' else 0)
def test_canonical():
    # 创建图像和轴对象
    fig, ax = plt.subplots()
    # 绘制简单的线图
    ax.plot([1, 2, 3])


# 比较椭圆弧角度图像，设置参数和样式
@image_comparison(['arc_angles.png'], remove_text=True, style='default')
def test_arc_angles():
    # 椭圆参数
    w = 2
    h = 1
    centre = (0.2, 0.5)
    scale = 2

    # 创建3x3子图
    fig, axs = plt.subplots(3, 3)
    for i, ax in enumerate(axs.flat):
        theta2 = i * 360 / 9
        theta1 = theta2 - 45

        # 添加椭圆和弧形到图中
        ax.add_patch(mpatches.Ellipse(centre, w, h, alpha=0.3))
        ax.add_patch(mpatches.Arc(centre, w, h, theta1=theta1, theta2=theta2))
        
        # 绘制直线，连接弧形的起始和结束点
        ax.plot([scale * np.cos(np.deg2rad(theta1)) + centre[0],
                 centre[0],
                 scale * np.cos(np.deg2rad(theta2)) + centre[0]],
                [scale * np.sin(np.deg2rad(theta1)) + centre[1],
                 centre[1],
                 scale * np.sin(np.deg2rad(theta2)) + centre[1]])

        # 设置坐标轴范围
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)

        # 改变参数以展示不同的图像特性
        w *= 10
        h *= 10
        centre = (centre[0] * 10, centre[1] * 10)
        scale *= 10


# 比较椭圆和弧形的图像
@image_comparison(['arc_ellipse'], remove_text=True)
def test_arc_ellipse():
    xcenter, ycenter = 0.38, 0.52
    width, height = 1e-1, 3e-1
    angle = -30

    # 计算椭圆形状
    theta = np.deg2rad(np.arange(360))
    x = width / 2. * np.cos(theta)
    y = height / 2. * np.sin(theta)

    # 旋转椭圆
    rtheta = np.deg2rad(angle)
    R = np.array([
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta), np.cos(rtheta)]])

    x, y = np.dot(R, [x, y])
    x += xcenter
    y += ycenter

    # 创建两个子图，分别展示不同样式的图像
    fig = plt.figure()
    ax = fig.add_subplot(211, aspect='auto')
    ax.fill(x, y, alpha=0.2, facecolor='yellow', edgecolor='yellow',
            linewidth=1, zorder=1)

    e1 = mpatches.Arc((xcenter, ycenter), width, height,
                      angle=angle, linewidth=2, fill=False, zorder=2)

    ax.add_patch(e1)

    ax = fig.add_subplot(212, aspect='equal')
    ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
    e2 = mpatches.Arc((xcenter, ycenter), width, height,
                      angle=angle, linewidth=2, fill=False, zorder=2)

    ax.add_patch(e2)


# 测试使用MarkerStyle作为标记器
def test_marker_as_markerstyle():
    # 创建图像和轴对象
    fix, ax = plt.subplots()
    # 使用MarkerStyle 'o'来绘制线图、散点图和误差线图
    m = mmarkers.MarkerStyle('o')
    ax.plot([1, 2, 3], [3, 2, 1], marker=m)
    ax.scatter([1, 2, 3], [4, 3, 2], marker=m)
    ax.errorbar([1, 2, 3], [5, 4, 3], marker=m)


# 比较带有不同标记点的图像
@image_comparison(['markevery'], remove_text=True)
def test_markevery():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.sqrt(x/10 + 0.5)

    # 检查仅有标记点的绘图
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', label='default')
    ax.plot(x, y, 'd', markevery=None, label='mark all')
    ax.plot(x, y, 's', markevery=10, label='mark every 10')
    # 在坐标轴 ax 上绘制散点图，使用加号 '+' 表示散点，设置每隔一定间隔标记一个点
    # markevery=(5, 20) 表示从第10个点开始，每隔5个点标记一个，因此标记的点序列为 10, 15, 20, ...
    ax.plot(x, y, '+', markevery=(5, 20), label='mark every 5 starting at 10')
    
    # 在图形中添加图例，显示标签为 'mark every 5 starting at 10' 的说明
    ax.legend()
@image_comparison(['markevery_line'], remove_text=True, tol=0.005)
# 定义一个测试函数，用于比较图像差异，其中包括了多个测试案例。
def test_markevery_line():
    # 对于SVG测试，由于Inkscape版本之间渲染略有差异，引入了一个小的非零容差值可能有助于通过测试。
    # 当Travis的Inkscape版本足够现代时，可以尝试移除此调整。在我的电脑上使用0.92.3版本时没有失败（#11358）。
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.sqrt(x/10 + 0.5)

    # 检查不同的线条/标记组合
    fig, ax = plt.subplots()
    ax.plot(x, y, '-o', label='default')
    ax.plot(x, y, '-d', markevery=None, label='mark all')
    ax.plot(x, y, '-s', markevery=10, label='mark every 10')
    ax.plot(x, y, '-+', markevery=(5, 20), label='mark every 5 starting at 10')
    ax.legend()


@image_comparison(['markevery_linear_scales'], remove_text=True, tol=0.001)
# 定义一个测试函数，比较在线性比例尺上的markevery效果。
def test_markevery_linear_scales():
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30], [0, -1],
             slice(100, 200, 3),
             0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    cols = 3
    # 创建一个网格布局，用于显示测试案例的子图
    gs = matplotlib.gridspec.GridSpec(len(cases) // cols + 1, cols)

    delta = 0.11
    x = np.linspace(0, 10 - 2 * delta, 200) + delta
    y = np.sin(x) + 1.0 + delta

    for i, case in enumerate(cases):
        row = (i // cols)
        col = i % cols
        # 在网格布局的每个位置创建子图
        plt.subplot(gs[row, col])
        plt.title('markevery=%s' % str(case))
        plt.plot(x, y, 'o', ls='-', ms=4, markevery=case)


@image_comparison(['markevery_linear_scales_zoomed'], remove_text=True)
# 定义一个测试函数，比较在线性比例尺上的markevery效果，并对部分图像进行缩放。
def test_markevery_linear_scales_zoomed():
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30], [0, -1],
             slice(100, 200, 3),
             0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    cols = 3
    gs = matplotlib.gridspec.GridSpec(len(cases) // cols + 1, cols)

    delta = 0.11
    x = np.linspace(0, 10 - 2 * delta, 200) + delta
    y = np.sin(x) + 1.0 + delta

    for i, case in enumerate(cases):
        row = (i // cols)
        col = i % cols
        plt.subplot(gs[row, col])
        plt.title('markevery=%s' % str(case))
        plt.plot(x, y, 'o', ls='-', ms=4, markevery=case)
        plt.xlim((6, 6.7))  # 设置x轴范围
        plt.ylim((1.1, 1.7))  # 设置y轴范围


@image_comparison(['markevery_log_scales'], remove_text=True)
# 定义一个测试函数，比较在对数比例尺上的markevery效果。
def test_markevery_log_scales():
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30], [0, -1],
             slice(100, 200, 3),
             0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    cols = 3
    gs = matplotlib.gridspec.GridSpec(len(cases) // cols + 1, cols)

    delta = 0.11
    x = np.linspace(0, 10 - 2 * delta, 200) + delta
    y = np.sin(x) + 1.0 + delta
    # 使用 enumerate 函数遍历 cases 列表，并同时获取索引 i 和元素 case
    for i, case in enumerate(cases):
        # 计算当前元素在二维网格中的行索引
        row = (i // cols)
        # 计算当前元素在二维网格中的列索引
        col = i % cols
        # 在指定的网格位置创建子图，并设置为当前活动子图
        plt.subplot(gs[row, col])
        # 设置子图标题，标题内容为当前 case 的字符串表示
        plt.title('markevery=%s' % str(case))
        # 设置 x 轴为对数坐标
        plt.xscale('log')
        # 设置 y 轴为对数坐标
        plt.yscale('log')
        # 在当前子图中绘制散点图，每个点的样式为圆形 'o'，线型为实线 '-'，点的大小为 4
        # 设置 markevery 参数为当前 case，用于控制显示哪些点
        plt.plot(x, y, 'o', ls='-', ms=4, markevery=case)
# 使用 image_comparison 装饰器比较测试结果，图像名称为 ['markevery_polar']，样式为 'default'，移除图中文本
@image_comparison(['markevery_polar'], style='default', remove_text=True)
def test_markevery_polar():
    # 定义测试用例列表 cases，包含多种 markevery 参数设置
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30], [0, -1],
             slice(100, 200, 3),
             0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    # 设置列数为 3
    cols = 3
    # 创建 GridSpec 对象 gs，用于排列子图
    gs = matplotlib.gridspec.GridSpec(len(cases) // cols + 1, cols)

    # 创建极坐标下的数据
    r = np.linspace(0, 3.0, 200)
    theta = 2 * np.pi * r

    # 遍历 cases 列表
    for i, case in enumerate(cases):
        # 计算当前子图的行和列
        row = (i // cols)
        col = i % cols
        # 创建极坐标子图
        plt.subplot(gs[row, col], polar=True)
        # 设置子图标题为当前 case 的字符串表示
        plt.title('markevery=%s' % str(case))
        # 在极坐标子图中绘制数据点，设置 markevery 参数
        plt.plot(theta, r, 'o', ls='-', ms=4, markevery=case)


# 使用 image_comparison 装饰器比较测试结果，图像名称为 ['markevery_linear_scales_nans']，移除图中文本
@image_comparison(['markevery_linear_scales_nans'], remove_text=True)
def test_markevery_linear_scales_nans():
    # 定义测试用例列表 cases，包含多种 markevery 参数设置
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30], [0, -1],
             slice(100, 200, 3),
             0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    # 设置列数为 3
    cols = 3
    # 创建 GridSpec 对象 gs，用于排列子图
    gs = matplotlib.gridspec.GridSpec(len(cases) // cols + 1, cols)

    # 创建带有 NaN 值的线性数据
    delta = 0.11
    x = np.linspace(0, 10 - 2 * delta, 200) + delta
    y = np.sin(x) + 1.0 + delta
    y[:10] = y[-20:] = y[50:70] = np.nan

    # 遍历 cases 列表
    for i, case in enumerate(cases):
        # 计算当前子图的行和列
        row = (i // cols)
        col = i % cols
        # 创建普通子图
        plt.subplot(gs[row, col])
        # 设置子图标题为当前 case 的字符串表示
        plt.title('markevery=%s' % str(case))
        # 在普通子图中绘制数据点，设置 markevery 参数
        plt.plot(x, y, 'o', ls='-', ms=4, markevery=case)


# 使用 image_comparison 装饰器比较测试结果，图像名称为 ['marker_edges']，移除图中文本
@image_comparison(['marker_edges'], remove_text=True)
def test_marker_edges():
    # 创建线性数据 x
    x = np.linspace(0, 1, 10)
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制散点图，设置标记大小、边缘宽度和边缘颜色
    ax.plot(x, np.sin(x), 'y.', ms=30.0, mew=0, mec='r')
    ax.plot(x+0.1, np.sin(x), 'y.', ms=30.0, mew=1, mec='r')
    ax.plot(x+0.2, np.sin(x), 'y.', ms=30.0, mew=2, mec='b')


# 使用 image_comparison 装饰器比较测试结果，图像名称为 ['bar_tick_label_single.png', 'bar_tick_label_single.png']
@image_comparison(['bar_tick_label_single.png', 'bar_tick_label_single.png'])
def test_bar_tick_label_single():
    # 从 2516 号问题中复用的测试用例：使用字符串标签的 x 轴绘制柱状图
    ax = plt.gca()
    ax.bar(0, 1, align='edge', tick_label='0')

    # 再次使用上述测试用例，进行带有标签数据的测试
    data = {"a": 0, "b": 1}
    fig, ax = plt.subplots()
    ax = plt.gca()
    ax.bar("a", "b", align='edge', tick_label='0', data=data)


# 测试函数：测试包含 NaN 值的柱状图数据
def test_nan_bar_values():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制包含 NaN 值的柱状图
    ax.bar([0, 1], [np.nan, 4])


# 测试函数：测试空列表作为参数时的柱状图绘制
def test_bar_ticklabel_fail():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制空列表的柱状图
    ax.bar([], [])


# 使用 image_comparison 装饰器比较测试结果，图像名称为 ['bar_tick_label_multiple.png']
@image_comparison(['bar_tick_label_multiple.png'])
def test_bar_tick_label_multiple():
    # 从 2516 号问题中复用的测试用例：使用字符串标签的 x 轴绘制多柱状图
    ax = plt.gca()
    ax.bar([1, 2.5], [1, 2], width=[0.2, 0.5], tick_label=['a', 'b'],
           align='center')


# 使用 image_comparison 装饰器比较测试结果，图像名称为 ['bar_tick_label_multiple_old_label_alignment.png']
@image_comparison(['bar_tick_label_multiple_old_label_alignment.png'])
def test_bar_tick_label_multiple_old_alignment():
    # 测试：检查 'ytick.alignment' 参数是否向后兼容
    matplotlib.rcParams["ytick.alignment"] = "center"
    ax = plt.gca()
    ax.bar([1, 2.5], [1, 2], width=[0.2, 0.5], tick_label=['a', 'b'],
           align='center')
@check_figures_equal(extensions=["png"])
def test_bar_decimal_center(fig_test, fig_ref):
    # 创建一个包含子图的测试图形对象
    ax = fig_test.subplots()
    # 输入数据转换为 Decimal 类型，用于垂直居中对齐的条形图测试
    x0 = [1.5, 8.4, 5.3, 4.2]
    y0 = [1.1, 2.2, 3.3, 4.4]
    x = [Decimal(x) for x in x0]
    y = [Decimal(y) for y in y0]
    ax.bar(x, y, align='center')  # 测试图像 - 垂直条形图
    # 参考图像
    ax = fig_ref.subplots()
    ax.bar(x0, y0, align='center')


@check_figures_equal(extensions=["png"])
def test_barh_decimal_center(fig_test, fig_ref):
    # 创建一个包含子图的测试图形对象
    ax = fig_test.subplots()
    # 输入数据转换为 Decimal 类型，用于水平居中对齐的条形图测试
    x0 = [1.5, 8.4, 5.3, 4.2]
    y0 = [1.1, 2.2, 3.3, 4.4]
    x = [Decimal(x) for x in x0]
    y = [Decimal(y) for y in y0]
    ax.barh(x, y, height=[0.5, 0.5, 1, 1], align='center')  # 测试图像 - 水平条形图
    # 参考图像
    ax = fig_ref.subplots()
    ax.barh(x0, y0, height=[0.5, 0.5, 1, 1], align='center')


@check_figures_equal(extensions=["png"])
def test_bar_decimal_width(fig_test, fig_ref):
    # 输入数据
    x = [1.5, 8.4, 5.3, 4.2]
    y = [1.1, 2.2, 3.3, 4.4]
    w0 = [0.7, 1.45, 1, 2]
    w = [Decimal(i) for i in w0]
    ax = fig_test.subplots()
    ax.bar(x, y, width=w, align='center')  # 测试图像 - 使用 Decimal 宽度的垂直条形图
    # 参考图像
    ax = fig_ref.subplots()
    ax.bar(x, y, width=w0, align='center')


@check_figures_equal(extensions=["png"])
def test_barh_decimal_height(fig_test, fig_ref):
    # 输入数据
    x = [1.5, 8.4, 5.3, 4.2]
    y = [1.1, 2.2, 3.3, 4.4]
    h0 = [0.7, 1.45, 1, 2]
    h = [Decimal(i) for i in h0]
    ax = fig_test.subplots()
    ax.barh(x, y, height=h, align='center')  # 测试图像 - 使用 Decimal 高度的水平条形图
    # 参考图像
    ax = fig_ref.subplots()
    ax.barh(x, y, height=h0, align='center')


def test_bar_color_none_alpha():
    ax = plt.gca()
    rects = ax.bar([1, 2], [2, 4], alpha=0.3, color='none', edgecolor='r')
    for rect in rects:
        assert rect.get_facecolor() == (0, 0, 0, 0)
        assert rect.get_edgecolor() == (1, 0, 0, 0.3)


def test_bar_edgecolor_none_alpha():
    ax = plt.gca()
    rects = ax.bar([1, 2], [2, 4], alpha=0.3, color='r', edgecolor='none')
    for rect in rects:
        assert rect.get_facecolor() == (1, 0, 0, 0.3)
        assert rect.get_edgecolor() == (0, 0, 0, 0)


@image_comparison(['barh_tick_label.png'])
def test_barh_tick_label():
    # 从2516号问题：绘制带有字符串标签数组的水平条形图
    ax = plt.gca()
    ax.barh([1, 2.5], [1, 2], height=[0.2, 0.5], tick_label=['a', 'b'],
            align='center')


def test_bar_timedelta():
    """Smoketest that bar can handle width and height in delta units."""
    fig, ax = plt.subplots()
    ax.bar(datetime.datetime(2018, 1, 1), 1.,
           width=datetime.timedelta(hours=3))
    ax.bar(datetime.datetime(2018, 1, 1), 1.,
           xerr=datetime.timedelta(hours=2),
           width=datetime.timedelta(hours=3))
    fig, ax = plt.subplots()
    # 在坐标轴 ax 上绘制一个水平条形图，位置在 datetime.datetime(2018, 1, 1)，宽度为 1，高度为 3 小时
    ax.barh(datetime.datetime(2018, 1, 1), 1,
            height=datetime.timedelta(hours=3))
    
    # 在坐标轴 ax 上绘制一个水平条形图，位置在 datetime.datetime(2018, 1, 1)，宽度为 1，高度为 3 小时，同时指定误差为 2 小时
    ax.barh(datetime.datetime(2018, 1, 1), 1,
            height=datetime.timedelta(hours=3),
            yerr=datetime.timedelta(hours=2))
    
    # 创建一个新的图形 fig 和坐标轴 ax
    fig, ax = plt.subplots()
    
    # 在坐标轴 ax 上绘制两个水平条形图，位置分别为 datetime.datetime(2018, 1, 1) 和 datetime.datetime(2018, 1, 1)，宽度为 1 和 1.5，高度为 3 小时
    ax.barh([datetime.datetime(2018, 1, 1), datetime.datetime(2018, 1, 1)],
            np.array([1, 1.5]),
            height=datetime.timedelta(hours=3))
    
    # 在坐标轴 ax 上绘制两个水平条形图，位置分别为 datetime.datetime(2018, 1, 1) 和 datetime.datetime(2018, 1, 1)，宽度为 1 和 1.5，高度分别为 1 小时和 2 小时
    ax.barh([datetime.datetime(2018, 1, 1), datetime.datetime(2018, 1, 1)],
            np.array([1, 1.5]),
            height=[datetime.timedelta(hours=t) for t in [1, 2]])
    
    # 在坐标轴 ax 上绘制一组断裂的水平条形图，起始时间为 datetime.datetime(2018, 1, 1)，持续时间为 1 小时，位于竖直方向上的位置从 10 开始，长度为 20
    ax.broken_barh([(datetime.datetime(2018, 1, 1),
                     datetime.timedelta(hours=1))],
                   (10, 20))
def test_bar_datetime_start():
    """测试日期时间对应的柱状图"""
    # 创建起始日期时间数组
    start = np.array([np.datetime64('2012-01-01'), np.datetime64('2012-02-01'),
                      np.datetime64('2012-01-15')])
    # 创建结束日期时间数组
    stop = np.array([np.datetime64('2012-02-07'), np.datetime64('2012-02-13'),
                     np.datetime64('2012-02-12')])

    # 创建图表和子图
    fig, ax = plt.subplots()
    # 绘制柱状图，高度为时间差，底部为起始日期
    ax.bar([0, 1, 3], height=stop-start, bottom=start)
    # 断言 y 轴主要格式化器的类型为自动日期格式化器
    assert isinstance(ax.yaxis.get_major_formatter(), mdates.AutoDateFormatter)

    # 创建新的图表和子图
    fig, ax = plt.subplots()
    # 绘制水平柱状图，宽度为时间差，左边界为起始日期
    ax.barh([0, 1, 3], width=stop-start, left=start)
    # 断言 x 轴主要格式化器的类型为自动日期格式化器
    assert isinstance(ax.xaxis.get_major_formatter(), mdates.AutoDateFormatter)


def test_boxplot_dates_pandas(pd):
    # pandas 中箱线图和日期的烟雾测试
    data = np.random.rand(5, 2)
    # 创建日期范围
    years = pd.date_range('1/1/2000',
                          periods=2, freq=pd.DateOffset(years=1)).year
    plt.figure()
    # 绘制箱线图，指定位置为年份
    plt.boxplot(data, positions=years)


def test_boxplot_capwidths():
    data = np.random.rand(5, 3)
    fig, axs = plt.subplots(9)

    # 绘制多个子图中的箱线图，设置上下边界宽度和箱体宽度
    axs[0].boxplot(data, capwidths=[0.3, 0.2, 0.1], widths=[0.1, 0.2, 0.3])
    axs[1].boxplot(data, capwidths=[0.3, 0.2, 0.1], widths=0.2)
    axs[2].boxplot(data, capwidths=[0.3, 0.2, 0.1])

    axs[3].boxplot(data, capwidths=0.5, widths=[0.1, 0.2, 0.3])
    axs[4].boxplot(data, capwidths=0.5, widths=0.2)
    axs[5].boxplot(data, capwidths=0.5)

    axs[6].boxplot(data, widths=[0.1, 0.2, 0.3])
    axs[7].boxplot(data, widths=0.2)
    axs[8].boxplot(data)


def test_pcolor_regression(pd):
    # 测试 pcolormesh 在 pandas 中的回归
    from pandas.plotting import (
        register_matplotlib_converters,
        deregister_matplotlib_converters,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 创建时间序列
    times = [datetime.datetime(2021, 1, 1)]
    while len(times) < 7:
        times.append(times[-1] + datetime.timedelta(seconds=120))

    y_vals = np.arange(5)

    time_axis, y_axis = np.meshgrid(times, y_vals)
    shape = (len(y_vals) - 1, len(times) - 1)
    z_data = np.arange(shape[0] * shape[1])

    z_data.shape = shape
    try:
        register_matplotlib_converters()

        # 绘制伪彩色图
        im = ax.pcolormesh(time_axis, y_axis, z_data)
        # 确保此操作不会引发异常！
        fig.canvas.draw()
    finally:
        deregister_matplotlib_converters()


def test_bar_pandas(pd):
    # pandas 的烟雾测试
    df = pd.DataFrame(
        {'year': [2018, 2018, 2018],
         'month': [1, 1, 1],
         'day': [1, 2, 3],
         'value': [1, 2, 3]})
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # 按日期分组并求和
    monthly = df[['date', 'value']].groupby(['date']).sum()
    dates = monthly.index
    forecast = monthly['value']
    baseline = monthly['value']

    # 创建图表和子图
    fig, ax = plt.subplots()
    # 绘制柱状图，宽度为10，对齐方式为居中
    ax.bar(dates, forecast, width=10, align='center')
    # 绘制基准线，颜色为橙色，线宽为4
    ax.plot(dates, baseline, color='orange', lw=4)


def test_bar_pandas_indexed(pd):
    # pandas 索引测试
    pass
    # 使用 Pandas 创建一个包含两列数据的 DataFrame，其中 "x" 列包含浮点数 [1., 2., 3.]，"width" 列包含浮点数 [.2, .4, .6]，并指定索引为 [1, 2, 3]
    df = pd.DataFrame({"x": [1., 2., 3.], "width": [.2, .4, .6]},
                      index=[1, 2, 3])
    
    # 使用 Matplotlib 创建一个新的图形（Figure）和一个包含单个子图（Axes）的坐标系（Axes）对象
    fig, ax = plt.subplots()
    
    # 在创建的坐标系对象 ax 上绘制一个柱状图，柱的 x 坐标为 df.x，高度为 1.0，柱的宽度为 df.width 对应的值
    ax.bar(df.x, 1., width=df.width)
@mpl.style.context('default')
@check_figures_equal()
def test_bar_hatches(fig_test, fig_ref):
    # 使用默认风格上下文，确保测试和参考图的图形一致性
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()

    x = [1, 2]
    y = [2, 3]
    hatches = ['x', 'o']
    for i in range(2):
        # 在参考图中绘制有阴影的条形图，使用指定的填充图案
        ax_ref.bar(x[i], y[i], color='C0', hatch=hatches[i])

    # 在测试图中绘制条形图，使用相同的填充图案
    ax_test.bar(x, y, hatch=hatches)


@pytest.mark.parametrize(
    ("x", "width", "label", "expected_labels", "container_label"),
    [
        ("x", 1, "x", ["_nolegend_"], "x"),
        (["a", "b", "c"], [10, 20, 15], ["A", "B", "C"],
         ["A", "B", "C"], "_nolegend_"),
        (["a", "b", "c"], [10, 20, 15], ["R", "Y", "_nolegend_"],
         ["R", "Y", "_nolegend_"], "_nolegend_"),
        (["a", "b", "c"], [10, 20, 15], "bars",
         ["_nolegend_", "_nolegend_", "_nolegend_"], "bars"),
    ]
)
def test_bar_labels(x, width, label, expected_labels, container_label):
    _, ax = plt.subplots()
    # 创建条形图容器，包括指定的标签
    bar_container = ax.bar(x, width, label=label)
    # 获取每个条形的标签
    bar_labels = [bar.get_label() for bar in bar_container]
    # 断言预期的标签与实际的条形图标签一致
    assert expected_labels == bar_labels
    # 断言条形图容器的整体标签与预期一致
    assert bar_container.get_label() == container_label


def test_bar_labels_length():
    _, ax = plt.subplots()
    # 使用 pytest 检查条形图标签的长度是否符合预期，应引发 ValueError
    with pytest.raises(ValueError):
        ax.bar(["x", "y"], [1, 2], label=["X", "Y", "Z"])
    _, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.bar(["x", "y"], [1, 2], label=["X"])


def test_pandas_minimal_plot(pd):
    # 简单测试确保 Series 和 Index 对象不会产生警告
    for x in [pd.Series([1, 2], dtype="float64"),
              pd.Series([1, 2], dtype="Float64")]:
        plt.plot(x, x)
        plt.plot(x.index, x)
        plt.plot(x)
        plt.plot(x.index)
    # 创建 DataFrame 并绘制其内容
    df = pd.DataFrame({'col': [1, 2, 3]})
    plt.plot(df)
    plt.plot(df, df)


@image_comparison(['hist_log'], remove_text=True)
def test_hist_log():
    # 测试对数坐标轴下的直方图绘制
    data0 = np.linspace(0, 1, 200)**3
    data = np.concatenate([1 - data0, 1 + data0])
    fig, ax = plt.subplots()
    # 绘制对数坐标轴下的直方图
    ax.hist(data, fill=False, log=True)


@check_figures_equal(extensions=["png"])
def test_hist_log_2(fig_test, fig_ref):
    # 测试多个直方图类型在不同设置下的图形一致性
    axs_test = fig_test.subplots(2, 3)
    axs_ref = fig_ref.subplots(2, 3)
    for i, histtype in enumerate(["bar", "step", "stepfilled"]):
        # 设置对数坐标轴，然后绘制直方图
        axs_test[0, i].set_yscale("log")
        axs_test[0, i].hist(1, 1, histtype=histtype)
        # 先绘制直方图，然后设置对数坐标轴
        axs_test[1, i].hist(1, 1, histtype=histtype)
        axs_test[1, i].set_yscale("log")
        # 使用 log=True 参数绘制直方图
        for ax in axs_ref[:, i]:
            ax.hist(1, 1, log=True, histtype=histtype)


def test_hist_log_barstacked():
    # 测试堆叠条形图在对数坐标轴下的绘制
    fig, axs = plt.subplots(2)
    axs[0].hist([[0], [0, 1]], 2, histtype="barstacked")
    axs[0].set_yscale("log")
    axs[1].hist([0, 0, 1], 2, histtype="barstacked")
    axs[1].set_yscale("log")
    fig.canvas.draw()
    # 断言两个子图的 y 轴限制相同
    assert axs[0].get_ylim() == axs[1].get_ylim()


@image_comparison(['hist_bar_empty.png'], remove_text=True)
# 定义测试函数，测试在空数据集上创建直方图是否会引发 ValueError
def test_hist_bar_empty():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 创建一个直方图，数据为空，类型为柱状图
    ax.hist([], histtype='bar')


# 定义测试函数，测试在 float16 类型数据上创建直方图
def test_hist_float16():
    # 设定随机数种子
    np.random.seed(19680801)
    # 生成均值为 0.5，标准差为 0.3 的正态分布随机数，并将其范围限制在 [0, 1]，转换为 float16 类型
    values = np.clip(np.random.normal(0.5, 0.3, size=1000), 0, 1).astype(np.float16)
    # 创建直方图，显示数据分布，分为 3 个区间，透明度为 0.5
    h = plt.hist(values, bins=3, alpha=0.5)
    # 获取直方图中的条形容器
    bc = h[2]
    # 检查是否没有重叠的矩形条
    for r in range(1, len(bc)):
        rleft = bc[r-1].get_corners()
        rright = bc[r].get_corners()
        # 断言：左侧矩形的右边界 <= 右侧矩形的左边界
        assert rleft[1][0] <= rright[0][0]


# 使用图像比较函数装饰的测试函数，测试在空数据集上创建 step 类型直方图是否会引发 ValueError
@image_comparison(['hist_step_empty.png'], remove_text=True)
def test_hist_step_empty():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 创建一个直方图，数据为空，类型为 step
    ax.hist([], histtype='step')


# 使用图像比较函数装饰的测试函数，测试在填充数据集上创建 stepfilled 类型直方图的正确性
@image_comparison(['hist_step_filled.png'], remove_text=True)
def test_hist_step_filled():
    # 设定随机数种子
    np.random.seed(0)
    # 生成服从标准正态分布的 1000x3 维随机数组
    x = np.random.randn(1000, 3)
    n_bins = 10

    # 定义不同参数组合和类型的直方图绘制
    kwargs = [{'fill': True}, {'fill': False}, {'fill': None}, {}]*2
    types = ['step']*4+['stepfilled']*4
    fig, axs = plt.subplots(nrows=2, ncols=4)

    # 遍历不同参数组合和类型，绘制直方图并设置标题和坐标轴限制
    for kg, _type, ax in zip(kwargs, types, axs.flat):
        ax.hist(x, n_bins, histtype=_type, stacked=True, **kg)
        ax.set_title(f'{kg}/{_type}')
        ax.set_ylim(bottom=-50)

    # 获取图中的所有矩形条，并断言它们的填充颜色和边界颜色相同
    patches = axs[0, 0].patches
    assert all(p.get_facecolor() == p.get_edgecolor() for p in patches)


# 使用图像比较函数装饰的测试函数，测试创建密度直方图的正确性
@image_comparison(['hist_density.png'])
def test_hist_density():
    # 设定随机数种子
    np.random.seed(19680801)
    # 生成 2000 个标准正态分布的随机数
    data = np.random.standard_normal(2000)
    # 创建密度归一化的直方图
    fig, ax = plt.subplots()
    ax.hist(data, density=True)


# 测试函数，测试在不等宽的时间数据集上创建直方图的正确性
def test_hist_unequal_bins_density():
    # 测试 GitHub 问题链接：https://github.com/matplotlib/matplotlib/issues/9557
    rng = np.random.RandomState(57483)
    t = rng.randn(100)
    bins = [-3, -1, -0.5, 0, 1, 5]
    # 使用 Matplotlib 创建直方图并进行密度归一化
    mpl_heights, _, _ = plt.hist(t, bins=bins, density=True)
    # 使用 NumPy 创建直方图并进行密度归一化
    np_heights, _ = np.histogram(t, bins=bins, density=True)
    # 断言：Matplotlib 创建的直方图与 NumPy 创建的直方图在归一化高度上近似相等
    assert_allclose(mpl_heights, np_heights)


# 测试函数，测试在日期时间数据集上创建直方图的正确性
def test_hist_datetime_datasets():
    # 创建包含日期时间数据的二维列表
    data = [[datetime.datetime(2017, 1, 1), datetime.datetime(2017, 1, 1)],
            [datetime.datetime(2017, 1, 1), datetime.datetime(2017, 1, 2)]]
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建堆叠直方图
    ax.hist(data, stacked=True)
    # 创建非堆叠直方图
    ax.hist(data, stacked=False)


# 使用 pytest 装饰器参数化的测试函数，测试在日期时间数据集上创建直方图并处理不同类型的日期时间数据
@pytest.mark.parametrize("bins_preprocess",
                         [mpl.dates.date2num,
                          lambda bins: bins,
                          lambda bins: np.asarray(bins, 'datetime64')],
                         ids=['date2num', 'datetime.datetime',
                              'np.datetime64'])
def test_hist_datetime_datasets_bins(bins_preprocess):
    # 定义一个包含两个子列表的二维列表，每个子列表包含四个 datetime.datetime 对象
    data = [
        [datetime.datetime(2019, 1, 5), datetime.datetime(2019, 1, 11),
         datetime.datetime(2019, 2, 1), datetime.datetime(2019, 3, 1)],
        [datetime.datetime(2019, 1, 11), datetime.datetime(2019, 2, 5),
         datetime.datetime(2019, 2, 18), datetime.datetime(2019, 3, 1)]
    ]
    
    # 定义一个包含三个 datetime.datetime 对象的列表，代表日期边界
    date_edges = [
        datetime.datetime(2019, 1, 1),
        datetime.datetime(2019, 2, 1),
        datetime.datetime(2019, 3, 1)
    ]
    
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    
    # 绘制堆叠的直方图，使用预处理后的日期边界作为分箱依据，并获取返回的直方图信息
    _, bins, _ = ax.hist(data, bins=bins_preprocess(date_edges), stacked=True)
    
    # 使用 NumPy 测试断言确保直方图的分箱边界与日期边界的数值表示相近
    np.testing.assert_allclose(bins, mpl.dates.date2num(date_edges))
    
    # 绘制非堆叠的直方图，使用相同的日期边界，并获取返回的直方图信息
    _, bins, _ = ax.hist(data, bins=bins_preprocess(date_edges), stacked=False)
    
    # 再次使用 NumPy 测试断言确保直方图的分箱边界与日期边界的数值表示相近
    np.testing.assert_allclose(bins, mpl.dates.date2num(date_edges))
# 使用 pytest 模块的 parametrize 装饰器，为 test_hist_with_empty_input 函数提供多组测试参数
@pytest.mark.parametrize('data, expected_number_of_hists',
                         [([], 1),
                          ([[]], 1),
                          ([[], []], 2)])
def test_hist_with_empty_input(data, expected_number_of_hists):
    # 调用 matplotlib 的 hist 函数生成直方图，并获取返回值中的 hists（直方图数据）、_、_
    hists, _, _ = plt.hist(data)
    # 将 hists 转换为 numpy 数组
    hists = np.asarray(hists)

    # 根据 hists 的维度判断，如果维度为 1，则断言直方图的数量应为 expected_number_of_hists
    if hists.ndim == 1:
        assert 1 == expected_number_of_hists
    else:
        # 否则，断言 hists 的行数应等于 expected_number_of_hists
        assert hists.shape[0] == expected_number_of_hists


# 使用 pytest 模块的 parametrize 装饰器，为 test_hist_zorder 函数提供多组测试参数
@pytest.mark.parametrize("histtype, zorder",
                         [("bar", mpl.patches.Patch.zorder),
                          ("step", mpl.lines.Line2D.zorder),
                          ("stepfilled", mpl.patches.Patch.zorder)])
def test_hist_zorder(histtype, zorder):
    # 创建一个包含单个子图的新图形，并获取子图对象
    ax = plt.figure().add_subplot()
    # 调用子图对象的 hist 方法生成直方图，指定类型为 histtype
    ax.hist([1, 2], histtype=histtype)
    # 断言子图对象包含 patches 对象（可能是图中的柱状图或填充图的表示）
    assert ax.patches
    # 遍历子图对象的 patches，断言每个 patch 的 zorder 属性等于预期值 zorder
    for patch in ax.patches:
        assert patch.get_zorder() == zorder


# 测试在 baseline=None 且 fill=True 时是否会发出 UserWarning 警告
def test_stairs_no_baseline_fill_warns():
    # 创建一个新的图形和子图对象
    fig, ax = plt.subplots()
    # 使用 pytest 的 warns 上下文管理器，捕获 UserWarning 类型的警告，并匹配警告信息中包含 "baseline=None and fill=True" 的警告
    with pytest.warns(UserWarning, match="baseline=None and fill=True"):
        # 在子图对象上调用 stairs 方法生成阶梯图，指定 facecolor="blue"、baseline=None、fill=True
        ax.stairs(
            [4, 5, 1, 0, 2],
            [1, 2, 3, 4, 5, 6],
            facecolor="blue",
            baseline=None,
            fill=True
        )


# 使用 check_figures_equal 装饰器，为 test_stairs 函数指定比较扩展为 'png'
@check_figures_equal(extensions=['png'])
def test_stairs(fig_test, fig_ref):
    # 导入 matplotlib 的 Line2D 模块
    import matplotlib.lines as mlines
    # 定义数据数组 y 和 x 作为测试数据
    y = np.array([6, 14, 32, 37, 48, 32, 21,  4])  # hist
    x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])  # bins

    # 创建测试图形 fig_test，并获取包含六个子图的数组 test_axes
    test_axes = fig_test.subplots(3, 2).flatten()
    # 在第一个测试子图上调用 stairs 方法生成阶梯图，指定 baseline=None
    test_axes[0].stairs(y, x, baseline=None)
    # 在第二个测试子图上调用 stairs 方法生成水平方向的阶梯图，指定 baseline=None 和 orientation='horizontal'
    test_axes[1].stairs(y, x, baseline=None, orientation='horizontal')
    # 在第三个测试子图上调用 stairs 方法生成阶梯图
    test_axes[2].stairs(y, x)
    # 在第四个测试子图上调用 stairs 方法生成水平方向的阶梯图，指定 orientation='horizontal'
    test_axes[3].stairs(y, x, orientation='horizontal')
    # 在第五个测试子图上调用 stairs 方法生成阶梯图
    test_axes[4].stairs(y, x)
    # 在第五个测试子图上调用 semilogy 方法，将 y 轴设置为对数刻度
    test_axes[4].semilogy()
    # 在第六个测试子图上调用 stairs 方法生成水平方向的阶梯图，同时调用 semilogx 方法，将 x 轴设置为对数刻度
    test_axes[5].stairs(y, x, orientation='horizontal')
    test_axes[5].semilogx()

    # 定义字典 style，包含 PathPatch 的默认参数，用于所有后续 Line2D 的绘制
    style = {'solid_joinstyle': 'miter', 'solid_capstyle': 'butt'}

    # 创建参考图形 fig_ref，并获取包含六个子图的数组 ref_axes
    ref_axes = fig_ref.subplots(3, 2).flatten()
    # 在第一个参考子图上绘制阶梯图，使用 steps-post 绘制风格，应用 style 字典中的样式参数
    ref_axes[0].plot(x, np.append(y, y[-1]), drawstyle='steps-post', **style)
    # 在第二个参考子图上绘制水平方向的阶梯图，使用 steps-post 绘制风格，应用 style 字典中的样式参数
    ref_axes[1].plot(np.append(y[0], y), x, drawstyle='steps-post', **style)

    # 在第三个参考子图上绘制阶梯图，使用 steps-post 绘制风格，应用 style 字典中的样式参数
    ref_axes[2].plot(x, np.append(y, y[-1]), drawstyle='steps-post', **style)
    # 在第三个参考子图上添加两条 Line2D 对象，表示阶梯图的起始和结束线段，应用 style 字典中的样式参数
    ref_axes[2].add_line(mlines.Line2D([x[0], x[0]], [0, y[0]], **style))
    ref_axes[2].add_line(mlines.Line2D([x[-1], x[-1]], [0, y[-1]], **style))
    # 设置第三个参考子图的 y 轴范围为 0 到 None（自动确定上限）
    ref_axes[2].set_ylim(0, None)

    # 在第四个参考子图上绘制水平方向的阶梯图，使用 steps-post 绘制风格，应用 style 字典中的样式参数
    ref_axes[3].plot(np.append(y[0], y), x, drawstyle='steps-post', **style)
    # 在第四个参考子图上添加两条 Line2D 对象，表示阶梯图的起始和结束线段，应用 style 字典中的样式参数
    ref_axes[3].add_line(mlines.Line2D([0, y[0]], [x[0], x[0]], **style))
    ref_axes[3].add_line(mlines.Line2D([0, y[-1]], [x[-1], x[-1]], **style))
    # 设置第四个参考子图的 x 轴范围为 0 到 None（自动确定上限）
    ref_axes[3].set_xlim(0, None)

    # 在第五个参考子图上绘制阶梯图，使用 steps-post 绘制风格，应用 style 字典中的样式参数
    ref_axes[4].plot(x, np.append(y, y[-1]), drawstyle='steps-post', **style)
    # 在第五个参考子图上添加两条 Line2D 对象，表示阶梯图的起始和结束线段，应用 style 字典中的样式参数
    ref_axes[4].add_line(mlines.Line2D([x[0], x[0]], [0, y
    # 在第6个参考轴上绘制步进后的 y 和 x 数据，使用指定的样式参数
    ref_axes[5].plot(np.append(y[0], y), x, drawstyle='steps-post', **style)
    # 在第6个参考轴上添加一条水平线段，起点到终点的 x 坐标与样式参数相关联
    ref_axes[5].add_line(mlines.Line2D([0, y[0]], [x[0], x[0]], **style))
    # 在第6个参考轴上添加另一条水平线段，起点到终点的 x 坐标与样式参数相关联
    ref_axes[5].add_line(mlines.Line2D([0, y[-1]], [x[-1], x[-1]], **style))
    # 在第6个参考轴上绘制半对数 x 轴（以10为底），无参数传入，使用默认设置
    ref_axes[5].semilogx()
@check_figures_equal(extensions=['png'])
# 使用装饰器检查两个图形是否相等，生成PNG格式的扩展文件
def test_stairs_fill(fig_test, fig_ref):
    h, bins = [1, 2, 3, 4, 2], [0, 1, 2, 3, 4, 5]
    bs = -2
    # 测试部分
    # 在测试图中创建2x2的子图，并展开为一维数组
    test_axes = fig_test.subplots(2, 2).flatten()
    # 在第一个子图中使用stairs方法绘制阶梯图，填充区域为True
    test_axes[0].stairs(h, bins, fill=True)
    # 在第二个子图中使用stairs方法绘制水平阶梯图，填充区域为True
    test_axes[1].stairs(h, bins, orientation='horizontal', fill=True)
    # 在第三个子图中使用stairs方法绘制阶梯图，设置基线为bs，填充区域为True
    test_axes[2].stairs(h, bins, baseline=bs, fill=True)
    # 在第四个子图中使用stairs方法绘制水平阶梯图，设置基线为bs，填充区域为True
    test_axes[3].stairs(h, bins, baseline=bs, orientation='horizontal',
                        fill=True)

    # 参考部分
    # 在参考图中创建2x2的子图，并展开为一维数组
    ref_axes = fig_ref.subplots(2, 2).flatten()
    # 在第一个子图中使用fill_between方法绘制填充区域，设置y轴范围为0到None
    ref_axes[0].fill_between(bins, np.append(h, h[-1]), step='post', lw=0)
    ref_axes[0].set_ylim(0, None)
    # 在第二个子图中使用fill_betweenx方法绘制填充区域，设置x轴范围为0到None
    ref_axes[1].fill_betweenx(bins, np.append(h, h[-1]), step='post', lw=0)
    ref_axes[1].set_xlim(0, None)
    # 在第三个子图中使用fill_between方法绘制填充区域，设置y轴范围为bs到None
    ref_axes[2].fill_between(bins, np.append(h, h[-1]),
                             np.ones(len(h)+1)*bs, step='post', lw=0)
    ref_axes[2].set_ylim(bs, None)
    # 在第四个子图中使用fill_betweenx方法绘制填充区域，设置x轴范围为bs到None
    ref_axes[3].fill_betweenx(bins, np.append(h, h[-1]),
                              np.ones(len(h)+1)*bs, step='post', lw=0)
    ref_axes[3].set_xlim(bs, None)


@check_figures_equal(extensions=['png'])
# 使用装饰器检查两个图形是否相等，生成PNG格式的扩展文件
def test_stairs_update(fig_test, fig_ref):
    # 固定y轴限制，因为stairs()自动缩放，但更新数据时不会
    ylim = -3, 4
    # 测试部分
    test_ax = fig_test.add_subplot()
    # 使用stairs方法绘制阶梯图，初始化数据为[1, 2, 3]
    h = test_ax.stairs([1, 2, 3])
    test_ax.set_ylim(ylim)
    # 更新数据
    h.set_data([3, 2, 1])
    h.set_data(edges=np.arange(4)+2)
    h.set_data([1, 2, 1], np.arange(4)/2)
    h.set_data([1, 2, 3])
    h.set_data(None, np.arange(4))
    # 断言检查更新后的数据是否符合预期
    assert np.allclose(h.get_data()[0], np.arange(1, 4))
    assert np.allclose(h.get_data()[1], np.arange(4))
    h.set_data(baseline=-2)
    assert h.get_data().baseline == -2

    # 参考部分
    ref_ax = fig_ref.add_subplot()
    # 使用stairs方法绘制阶梯图，设置基线为-2
    h = ref_ax.stairs([1, 2, 3], baseline=-2)
    ref_ax.set_ylim(ylim)


@check_figures_equal(extensions=['png'])
# 使用装饰器检查两个图形是否相等，生成PNG格式的扩展文件
def test_stairs_baseline_None(fig_test, fig_ref):
    x = np.array([0, 2, 3, 5, 10])
    y = np.array([1.148, 1.231, 1.248, 1.25])

    test_axes = fig_test.add_subplot()
    # 使用stairs方法绘制阶梯图，基线为None
    test_axes.stairs(y, x, baseline=None)

    style = {'solid_joinstyle': 'miter', 'solid_capstyle': 'butt'}

    ref_axes = fig_ref.add_subplot()
    # 使用plot方法绘制步阶曲线图，设置样式参数
    ref_axes.plot(x, np.append(y, y[-1]), drawstyle='steps-post', **style)


def test_stairs_empty():
    ax = plt.figure().add_subplot()
    # 使用stairs方法绘制空数据的阶梯图
    ax.stairs([], [42])
    # 断言检查x轴和y轴的限制是否符合预期
    assert ax.get_xlim() == (39, 45)
    assert ax.get_ylim() == (-0.06, 0.06)


def test_stairs_invalid_nan():
    with pytest.raises(ValueError, match='Nan values in "edges"'):
        # 使用stairs方法绘制包含NaN值的阶梯图，预期引发值错误异常
        plt.stairs([1, 2], [0, np.nan, 1])


def test_stairs_invalid_mismatch():
    with pytest.raises(ValueError, match='Size mismatch'):
        # 使用stairs方法绘制数据大小不匹配的阶梯图，预期引发大小不匹配异常
        plt.stairs([1, 2], [0, 1])


def test_stairs_invalid_update():
    h = plt.stairs([1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match='Nan values in "edges"'):
        # 使用set_data方法更新阶梯图数据，包含NaN值，预期引发值错误异常
        h.set_data(edges=[1, np.nan, 2])


def test_stairs_invalid_update2():
    h = plt.stairs([1, 2], [0, 1, 2])
    # 使用 pytest 的上下文管理器，测试是否会引发 ValueError 异常，并且匹配异常消息为 'Size mismatch'
    with pytest.raises(ValueError, match='Size mismatch'):
        # 调用 h 对象的 set_data 方法，传入 edges 参数为 np.arange(5)
        h.set_data(edges=np.arange(5))
@image_comparison(['test_stairs_options.png'], remove_text=True)
def test_stairs_options():
    # 定义数据点
    x, y = np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4]).astype(float)
    # 复制数组y并将索引为1的元素设为NaN
    yn = y.copy()
    yn[1] = np.nan

    # 创建图形和子图
    fig, ax = plt.subplots()
    # 绘制stairs图形A，填充颜色为绿色，标签为"A"
    ax.stairs(y*3, x, color='green', fill=True, label="A")
    # 绘制stairs图形B，填充颜色为红色，横向绘制，标签为"B"
    ax.stairs(y, x*3-3, color='red', fill=True,
              orientation='horizontal', label="B")
    # 绘制stairs图形C，填充颜色为橙色，线条样式为虚线，线宽为2，标签为"C"
    ax.stairs(yn, x, color='orange', ls='--', lw=2, label="C")
    # 绘制stairs图形D，横向绘制，填充颜色为无，线条样式为虚线，线宽为2，基线位置为0.5，标签为"D"
    ax.stairs(yn/3, x*3-2, ls='--', lw=2, baseline=0.5,
              orientation='horizontal', label="D")
    # 绘制stairs图形E，填充颜色为红色，线条样式为虚线，线宽为2，基线位置为None，标签为"E"
    ax.stairs(y[::-1]*3+13, x-1, color='red', ls='--', lw=2, baseline=None,
              label="E")
    # 绘制stairs图形F，填充颜色为紫色，线条样式为虚线，线宽为2，基线位置为26，标签为"F"
    ax.stairs(y[::-1]*3+14, x, baseline=26,
              color='purple', ls='--', lw=2, label="F")
    # 绘制stairs图形G，填充颜色为蓝色，线条样式为虚线，标签为"G"，填充数据
    ax.stairs(yn[::-1]*3+15, x+1, baseline=np.linspace(27, 25, len(y)),
              color='blue', ls='--', label="G", fill=True)
    # 绘制stairs图形H，填充颜色为黑色，线条样式为虚线，线宽为2，基线位置为12，标签为"H"，使用hatch填充
    ax.stairs(y[:-1][::-1]*2+11, x[:-1]+0.5, color='black', ls='--', lw=2,
              baseline=12, hatch='//', label="H")
    # 添加图例
    ax.legend(loc=0)


@image_comparison(['test_stairs_datetime.png'])
def test_stairs_datetime():
    # 创建图形和子图，保证布局约束为True
    f, ax = plt.subplots(constrained_layout=True)
    # 绘制stairs图形，使用日期时间作为横轴
    ax.stairs(np.arange(36),
              np.arange(np.datetime64('2001-12-27'),
                        np.datetime64('2002-02-02')))
    # 设置横轴刻度标签旋转30度
    plt.xticks(rotation=30)


@check_figures_equal(extensions=['png'])
def test_stairs_edge_handling(fig_test, fig_ref):
    # 测试图像
    test_ax = fig_test.add_subplot()
    # 在测试图像上绘制stairs图形，填充颜色为红色
    test_ax.stairs([1, 2, 3], color='red', fill=True)

    # 参考图像
    ref_ax = fig_ref.add_subplot()
    # 在参考图像上绘制stairs图形，填充颜色为红色
    st = ref_ax.stairs([1, 2, 3], fill=True)
    # 设置绘图对象的颜色为红色


def contour_dat():
    # 生成x和y的均匀分布
    x = np.linspace(-3, 5, 150)
    y = np.linspace(-3, 5, 120)
    # 生成z，使用x和y的余弦加正弦
    z = np.cos(x) + np.sin(y[:, np.newaxis])
    return x, y, z


@image_comparison(['contour_hatching'], remove_text=True, style='mpl20')
def test_contour_hatching():
    # 获取contour_dat函数生成的数据
    x, y, z = contour_dat()
    # 创建图形和子图
    fig, ax = plt.subplots()
    # 绘制等高线图，使用指定的hatches填充样式
    ax.contourf(x, y, z, 7, hatches=['/', '\\', '//', '-'],
                cmap=mpl.colormaps['gray'],
                extend='both', alpha=0.5)


@image_comparison(
    ['contour_colorbar'], style='mpl20',
    tol=0.54 if platform.machine() in ('aarch64', 'ppc64le', 's390x') else 0)
def test_contour_colorbar():
    # 获取contour_dat函数生成的数据
    x, y, z = contour_dat()

    # 创建图形和子图
    fig, ax = plt.subplots()
    # 绘制填充的等高线图，使用指定的levels和colormap，设置颜色条范围和扩展方式
    cs = ax.contourf(x, y, z, levels=np.arange(-1.8, 1.801, 0.2),
                     cmap=mpl.colormaps['RdBu'],
                     vmin=-0.6,
                     vmax=0.6,
                     extend='both')
    # 添加等高线，设置指定颜色和线条样式
    cs1 = ax.contour(x, y, z, levels=np.arange(-2.2, -0.599, 0.2),
                     colors=['y'],
                     linestyles='solid',
                     linewidths=2)
    cs2 = ax.contour(x, y, z, levels=np.arange(0.6, 2.2, 0.2),
                     colors=['c'],
                     linewidths=2)
    # 添加颜色条
    cbar = fig.colorbar(cs, ax=ax)
    # 将等高线添加到颜色条中
    cbar.add_lines(cs1)
    cbar.add_lines(cs2, erase=False)
@image_comparison(['hist2d', 'hist2d'], remove_text=True, style='mpl20')
def test_hist2d():
    # 当这个测试图像重新生成时，请移除此行。
    # 设置 matplotlib 参数，使 pcolormesh.snap 参数为 False
    plt.rcParams['pcolormesh.snap'] = False

    # 设置随机数种子为 0
    np.random.seed(0)
    # 生成一个非对称的数据集，以确保在切换 x 和 y 轴时不对称
    x = np.random.randn(100) * 2 + 5
    y = np.random.randn(100) - 2

    # 创建一个新的图和轴对象
    fig, ax = plt.subplots()
    # 绘制二维直方图
    ax.hist2d(x, y, bins=10, rasterized=True)

    # 使用上面生成的数据字典进行标记数据的测试
    data = {"x": x, "y": y}
    fig, ax = plt.subplots()
    # 从数据字典中绘制二维直方图
    ax.hist2d("x", "y", bins=10, data=data, rasterized=True)


@image_comparison(['hist2d_transpose'], remove_text=True, style='mpl20')
def test_hist2d_transpose():
    # 当这个测试图像重新生成时，请移除此行。
    # 设置 matplotlib 参数，使 pcolormesh.snap 参数为 False
    plt.rcParams['pcolormesh.snap'] = False

    # 设置随机数种子为 0
    np.random.seed(0)
    # 确保从 np.histogram 输出的数据在传递给 pcolorfast 之前进行了转置
    x = np.array([5] * 100)
    y = np.random.randn(100) - 2

    # 创建一个新的图和轴对象
    fig, ax = plt.subplots()
    # 绘制二维直方图
    ax.hist2d(x, y, bins=10, rasterized=True)


def test_hist2d_density():
    # 生成随机数据
    x, y = np.random.random((2, 100))
    # 创建一个图和轴对象
    ax = plt.figure().subplots()
    # 对图和 plt 对象分别绘制密度为 True 的二维直方图
    for obj in [ax, plt]:
        obj.hist2d(x, y, density=True)


class TestScatter:
    @image_comparison(['scatter'], style='mpl20', remove_text=True)
    def test_scatter_plot(self):
        # 设置测试数据
        data = {"x": np.array([3, 4, 2, 6]), "y": np.array([2, 5, 2, 3]),
                "c": ['r', 'y', 'b', 'lime'], "s": [24, 15, 19, 29],
                "c2": ['0.5', '0.6', '0.7', '0.8']}

        # 创建一个新的图和轴对象
        fig, ax = plt.subplots()
        # 绘制散点图，使用 data 中的数据
        ax.scatter(data["x"] - 1., data["y"] - 1., c=data["c"], s=data["s"])
        ax.scatter(data["x"] + 1., data["y"] + 1., c=data["c2"], s=data["s"])
        # 从 data 中绘制散点图，使用字符串标识的数据列名
        ax.scatter("x", "y", c="c", s="s", data=data)

    @image_comparison(['scatter_marker.png'], remove_text=True)
    def test_scatter_marker(self):
        # 创建一个新的图和多个轴对象
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
        
        # 在三个子图中分别绘制带有不同标记和样式的散点图
        ax0.scatter([3, 4, 2, 6], [2, 5, 2, 3],
                    c=[(1, 0, 0), 'y', 'b', 'lime'],
                    s=[60, 50, 40, 30],
                    edgecolors=['k', 'r', 'g', 'b'],
                    marker='s')
        ax1.scatter([3, 4, 2, 6], [2, 5, 2, 3],
                    c=[(1, 0, 0), 'y', 'b', 'lime'],
                    s=[60, 50, 40, 30],
                    edgecolors=['k', 'r', 'g', 'b'],
                    marker=mmarkers.MarkerStyle('o', fillstyle='top'))
        
        # 创建一个单位面积的椭圆
        rx, ry = 3, 1
        area = rx * ry * np.pi
        theta = np.linspace(0, 2 * np.pi, 21)
        verts = np.column_stack([np.cos(theta) * rx / area,
                                 np.sin(theta) * ry / area])
        ax2.scatter([3, 4, 2, 6], [2, 5, 2, 3],
                    c=[(1, 0, 0), 'y', 'b', 'lime'],
                    s=[60, 50, 40, 30],
                    edgecolors=['k', 'r', 'g', 'b'],
                    marker=verts)

    @image_comparison(['scatter_2D'], remove_text=True, extensions=['png'])
    # 定义一个测试方法，用于绘制2D散点图
    def test_scatter_2D(self):
        # 创建一个包含3个元素的一维数组
        x = np.arange(3)
        # 创建一个包含2个元素的一维数组
        y = np.arange(2)
        # 使用x和y创建网格
        x, y = np.meshgrid(x, y)
        # 计算z值，即x和y的元素级加法结果
        z = x + y
        # 创建图形和轴对象
        fig, ax = plt.subplots()
        # 绘制散点图，颜色使用z值，大小为200，边缘颜色与填充颜色相同
        ax.scatter(x, y, c=z, s=200, edgecolors='face')

    # 通过装饰器标记为使用check_figures_equal方法进行测试，并生成扩展名为"png"的图像
    def test_scatter_decimal(self, fig_test, fig_ref):
        # 创建包含四个浮点数的numpy数组
        x0 = np.array([1.5, 8.4, 5.3, 4.2])
        y0 = np.array([1.1, 2.2, 3.3, 4.4])
        # 将x0和y0中的每个元素转换为Decimal对象，并创建包含Decimal对象的数组
        x = np.array([Decimal(i) for i in x0])
        y = np.array([Decimal(i) for i in y0])
        # 指定每个点的颜色和大小
        c = ['r', 'y', 'b', 'lime']
        s = [24, 15, 19, 29]
        # 在测试图像上绘制散点图，使用Decimal类型的输入
        ax = fig_test.subplots()
        ax.scatter(x, y, c=c, s=s)
        # 在参考图像上绘制散点图，使用原始浮点数输入
        ax = fig_ref.subplots()
        ax.scatter(x0, y0, c=c, s=s)

    # 定义一个测试方法，用于检测scatter方法在颜色参数使用错误时是否会引发异常
    def test_scatter_color(self):
        # 检测当颜色参数'c'使用列表形式时，是否会引发值错误异常
        with pytest.raises(ValueError):
            plt.scatter([1, 2], [1, 2], color=[0.1, 0.2])
        with pytest.raises(ValueError):
            plt.scatter([1, 2, 3], [1, 2, 3], color=[1, 2, 3])

    # 使用pytest参数化装饰器，测试scatter方法在不同kwargs情况下是否会发出警告
    @pytest.mark.parametrize('kwargs',
                                [
                                    {'cmap': 'gray'},
                                    {'norm': mcolors.Normalize()},
                                    {'vmin': 0},
                                    {'vmax': 0}
                                ])
    def test_scatter_color_warning(self, kwargs):
        # 警告匹配文本，用于捕获可能覆盖'cmap'，'norm'，'vmin'，'vmax'参数的情况
        warn_match = "No data for colormapping provided "
        # 测试是否会发出警告，如果覆盖了'cmap'，'norm'，'vmin'，'vmax'参数
        with pytest.warns(Warning, match=warn_match):
            plt.scatter([], [], **kwargs)
        with pytest.warns(Warning, match=warn_match):
            plt.scatter([1, 2], [3, 4], c=[], **kwargs)
        # 对于'c'与'x'和'y'匹配的情况，不应该发出警告
        plt.scatter([], [], c=[], **kwargs)
        plt.scatter([1, 2], [3, 4], c=[4, 5], **kwargs)

    # 定义一个测试方法，用于测试scatter方法中未填充的散点图情况
    def test_scatter_unfilled(self):
        # 创建未填充的散点图，指定颜色、标记和线宽度
        coll = plt.scatter([0, 1, 2], [1, 3, 2], c=['0.1', '0.3', '0.5'],
                           marker=mmarkers.MarkerStyle('o', fillstyle='none'),
                           linewidths=[1.1, 1.2, 1.3])
        # 断言未填充散点的面颜色应该是空数组
        assert coll.get_facecolors().shape == (0, 4)  # no facecolors
        # 断言边框颜色应该与指定的颜色数组一致
        assert_array_equal(coll.get_edgecolors(), [[0.1, 0.1, 0.1, 1],
                                                   [0.3, 0.3, 0.3, 1],
                                                   [0.5, 0.5, 0.5, 1]])
        # 断言线宽应该与指定的数组一致
        assert_array_equal(coll.get_linewidths(), [1.1, 1.2, 1.3])

    # 使用mpl.style.context('default')上下文管理器进行默认风格的测试
    @check_figures_equal(extensions=["png"])
    # 使用装饰器检查两个图形是否相等，输出为 PNG 格式的图像
    def test_scatter_invalid_color(self, fig_test, fig_ref):
        # 在 fig_test 中创建子图 ax
        ax = fig_test.subplots()
        # 从 viridis 颜色映射中获取 16 个颜色样本，并设置无效值为黑色，透明度为 1
        cmap = mpl.colormaps["viridis"].resampled(16)
        cmap.set_bad("k", 1)
        # 在测试图中绘制散点图，包括无效颜色值，设置点的大小和颜色映射，允许绘制无效点
        ax.scatter(range(4), range(4),
                   c=[1, np.nan, 2, np.nan], s=[1, 2, 3, 4],
                   cmap=cmap, plotnonfinite=True)
        # 在 fig_ref 中创建子图 ax
        ax = fig_ref.subplots()
        # 从 viridis 颜色映射中获取 16 个颜色样本
        cmap = mpl.colormaps["viridis"].resampled(16)
        # 在参考图中绘制散点图，设置点的位置和大小，颜色映射为 cmap
        ax.scatter([0, 2], [0, 2], c=[1, 2], s=[1, 3], cmap=cmap)
        # 在参考图中绘制散点图，设置点的位置和大小，颜色为黑色
        ax.scatter([1, 3], [1, 3], s=[2, 4], color="k")

    @check_figures_equal(extensions=["png"])
    # 使用装饰器检查两个图形是否相等，输出为 PNG 格式的图像
    def test_scatter_no_invalid_color(self, fig_test, fig_ref):
        # 在 fig_test 中创建子图 ax
        ax = fig_test.subplots()
        # 从 viridis 颜色映射中获取 16 个颜色样本，并设置无效值为黑色，透明度为 1
        cmap = mpl.colormaps["viridis"].resampled(16)
        cmap.set_bad("k", 1)
        # 在测试图中绘制散点图，只包括有效颜色值，设置点的大小和颜色映射，不绘制无效点
        ax.scatter(range(4), range(4),
                   c=[1, np.nan, 2, np.nan], s=[1, 2, 3, 4],
                   cmap=cmap, plotnonfinite=False)
        # 在 fig_ref 中创建子图 ax
        ax = fig_ref.subplots()
        # 在参考图中绘制散点图，设置点的位置和大小，颜色映射为 cmap
        ax.scatter([0, 2], [0, 2], c=[1, 2], s=[1, 3], cmap=cmap)
    def test_scatter_norm_vminvmax(self):
        """测试 scatter 函数中的 vmin 和 vmax 参数在同时指定 norm 参数时应该报错。"""
        x = [1, 2, 3]  # 创建一个包含几个数值的列表 x
        ax = plt.axes()  # 获取当前图形的坐标轴对象
        with pytest.raises(ValueError,
                           match="Passing a Normalize instance simultaneously "
                                 "with vmin/vmax is not supported."):
            # 使用 pytest 的异常断言检查 ValueError 是否被正确抛出，
            # 并且错误信息匹配指定的正则表达式
            ax.scatter(x, x, c=x, norm=mcolors.Normalize(-10, 10),
                       vmin=0, vmax=5)

    @check_figures_equal(extensions=["png"])
    def test_scatter_single_point(self, fig_test, fig_ref):
        """测试 scatter 函数绘制单个点的情况。"""
        ax = fig_test.subplots()  # 在测试图中创建子图对象
        ax.scatter(1, 1, c=1)  # 绘制一个单个点，颜色为整数 1
        ax = fig_ref.subplots()  # 在参考图中创建子图对象
        ax.scatter([1], [1], c=[1])  # 绘制一个单个点，颜色为列表 [1]

    @check_figures_equal(extensions=["png"])
    def test_scatter_different_shapes(self, fig_test, fig_ref):
        """测试 scatter 函数对不同形状输入的处理。"""
        x = np.arange(10)  # 创建一个包含 0 到 9 的数组 x
        ax = fig_test.subplots()  # 在测试图中创建子图对象
        ax.scatter(x, x.reshape(2, 5), c=x.reshape(5, 2))  # 绘制散点图，传入不同形状的数据
        ax = fig_ref.subplots()  # 在参考图中创建子图对象
        ax.scatter(x.reshape(5, 2), x, c=x.reshape(2, 5))  # 绘制散点图，传入不同形状的数据

    # Parameters for *test_scatter_c*. NB: assuming that the
    # scatter plot will have 4 elements. The tuple scheme is:
    # (*c* parameter case, exception regexp key or None if no exception)
    params_test_scatter_c = [
        # single string:
        ('0.5', None),
        # Single letter-sequences
        (["rgby"], "conversion"),
        # Special cases
        ("red", None),
        ("none", None),
        (None, None),
        (["r", "g", "b", "none"], None),
        # Non-valid color spec (FWIW, 'jaune' means yellow in French)
        ("jaune", "conversion"),
        (["jaune"], "conversion"),  # wrong type before wrong size
        (["jaune"]*4, "conversion"),
        # Value-mapping like
        ([0.5]*3, None),  # should emit a warning for user's eyes though
        ([0.5]*4, None),  # NB: no warning as matching size allows mapping
        ([0.5]*5, "shape"),
        # list of strings:
        (['0.5', '0.4', '0.6', '0.7'], None),
        (['0.5', 'red', '0.6', 'C5'], None),
        (['0.5', 0.5, '0.6', 'C5'], "conversion"),
        # RGB values
        ([[1, 0, 0]], None),
        ([[1, 0, 0]]*3, "shape"),
        ([[1, 0, 0]]*4, None),
        ([[1, 0, 0]]*5, "shape"),
        # RGBA values
        ([[1, 0, 0, 0.5]], None),
        ([[1, 0, 0, 0.5]]*3, "shape"),
        ([[1, 0, 0, 0.5]]*4, None),
        ([[1, 0, 0, 0.5]]*5, "shape"),
        # Mix of valid color specs
        ([[1, 0, 0, 0.5]]*3 + [[1, 0, 0]], None),
        ([[1, 0, 0, 0.5], "red", "0.0"], "shape"),
        ([[1, 0, 0, 0.5], "red", "0.0", "C5"], None),
        ([[1, 0, 0, 0.5], "red", "0.0", "C5", [0, 1, 0]], "shape"),
        # Mix of valid and non valid color specs
        ([[1, 0, 0, 0.5], "red", "jaune"], "conversion"),
        ([[1, 0, 0, 0.5], "red", "0.0", "jaune"], "conversion"),
        ([[1, 0, 0, 0.5], "red", "0.0", "C5", "jaune"], "conversion"),
    ]

    @pytest.mark.parametrize('c_case, re_key', params_test_scatter_c)
    def test_scatter_c(self, c_case, re_key):
        # 定义获取下一个颜色的函数，当前未使用
        def get_next_color():
            return 'blue'  # currently unused

        xsize = 4
        # *c* 参数的额外检查（自 #11383 引入）。
        REGEXP = {
            "shape": "^'c' argument has [0-9]+ elements",  # 形状不匹配
            "conversion": "^'c' argument must be a color",  # 值不正确
            }

        # 根据 re_key 的值选择不同的断言上下文
        assert_context = (
            pytest.raises(ValueError, match=REGEXP[re_key])
            if re_key is not None
            else pytest.warns(match="argument looks like a single numeric RGB")
            if isinstance(c_case, list) and len(c_case) == 3
            else contextlib.nullcontext()
        )
        # 运行带有断言上下文的代码块
        with assert_context:
            mpl.axes.Axes._parse_scatter_color_args(
                c=c_case, edgecolors="black", kwargs={}, xsize=xsize,
                get_next_color_func=get_next_color)

    @mpl.style.context('default')
    @check_figures_equal(extensions=["png"])
    def test_scatter_single_color_c(self, fig_test, fig_ref):
        rgb = [[1, 0.5, 0.05]]
        rgba = [[1, 0.5, 0.05, .5]]

        # 通过 color 关键字参数设置颜色
        ax_ref = fig_ref.subplots()
        ax_ref.scatter(np.ones(3), range(3), color=rgb)
        ax_ref.scatter(np.ones(4)*2, range(4), color=rgba)

        # 通过 c 参数广播设置颜色
        ax_test = fig_test.subplots()
        ax_test.scatter(np.ones(3), range(3), c=rgb)
        ax_test.scatter(np.ones(4)*2, range(4), c=rgba)

    def test_scatter_linewidths(self):
        x = np.arange(5)

        fig, ax = plt.subplots()
        for i in range(3):
            pc = ax.scatter(x, np.full(5, i), c=f'C{i}', marker='x', s=100,
                            linewidths=i + 1)
            assert pc.get_linewidths() == i + 1

        pc = ax.scatter(x, np.full(5, 3), c='C3', marker='x', s=100,
                        linewidths=[*range(1, 5), None])
        assert_array_equal(pc.get_linewidths(),
                           [*range(1, 5), mpl.rcParams['lines.linewidth']])

    def test_scatter_singular_plural_arguments(self):

        # 抛出 TypeError，指出 'linewidth' 和 'linewidths' 是彼此的别名
        with pytest.raises(TypeError,
                           match="Got both 'linewidth' and 'linewidths', which are aliases of one another"):
            plt.scatter([1, 2, 3], [1, 2, 3], linewidths=[0.5, 0.4, 0.3], linewidth=0.2)

        # 抛出 TypeError，指出 'edgecolor' 和 'edgecolors' 是彼此的别名
        with pytest.raises(TypeError,
                           match="Got both 'edgecolor' and 'edgecolors', which are aliases of one another"):
            plt.scatter([1, 2, 3], [1, 2, 3],
                        edgecolors=["#ffffff", "#000000", "#f0f0f0"],
                        edgecolor="#ffffff")

        # 抛出 TypeError，指出 'facecolors' 和 'facecolor' 是彼此的别名
        with pytest.raises(TypeError,
                           match="Got both 'facecolors' and 'facecolor', which are aliases of one another"):
            plt.scatter([1, 2, 3], [1, 2, 3],
                        facecolors=["#ffffff", "#000000", "#f0f0f0"],
                        facecolor="#ffffff")
def _params(c=None, xsize=2, *, edgecolors=None, **kwargs):
    # 返回一个包含参数 c, edgecolors, kwargs 和 xsize 的元组
    return (c, edgecolors, kwargs if kwargs is not None else {}, xsize)

# 定义一个命名元组 _result，包含字段 'c' 和 'colors'
_result = namedtuple('_result', 'c, colors')

# 使用 pytest 的 parametrize 装饰器，对 test_parse_scatter_color_args 进行参数化测试
@pytest.mark.parametrize(
    'params, expected_result',
    [(_params(),
      _result(c='b', colors=np.array([[0, 0, 1, 1]]))),
     (_params(c='r'),
      _result(c='r', colors=np.array([[1, 0, 0, 1]]))),
     (_params(c='r', colors='b'),
      _result(c='r', colors=np.array([[1, 0, 0, 1]]))),
     # color
     (_params(color='b'),
      _result(c='b', colors=np.array([[0, 0, 1, 1]]))),
     (_params(color=['b', 'g']),
      _result(c=['b', 'g'], colors=np.array([[0, 0, 1, 1], [0, .5, 0, 1]]))),
     ])
def test_parse_scatter_color_args(params, expected_result):
    # 定义一个内部函数，用于返回下一个颜色，但当前未使用
    def get_next_color():
        return 'blue'  # currently unused
    
    # 调用 mpl.axes.Axes._parse_scatter_color_args 方法，解析参数，获取返回值
    c, colors, _edgecolors = mpl.axes.Axes._parse_scatter_color_args(
        *params, get_next_color_func=get_next_color)
    # 断言返回的 c 和 colors 符合预期结果
    assert c == expected_result.c
    assert_allclose(colors, expected_result.colors)

# 删除定义的函数和命名元组，清理环境
del _params
del _result

# 对 test_parse_scatter_color_args_edgecolors 进行参数化测试
@pytest.mark.parametrize(
    'kwargs, expected_edgecolors',
    [(dict(), None),
     (dict(c='b'), None),
     (dict(edgecolors='r'), 'r'),
     (dict(edgecolors=['r', 'g']), ['r', 'g']),
     (dict(edgecolor='r'), 'r'),
     (dict(edgecolors='face'), 'face'),
     (dict(edgecolors='none'), 'none'),
     (dict(edgecolor='r', edgecolors='g'), 'r'),
     (dict(c='b', edgecolor='r', edgecolors='g'), 'r'),
     (dict(color='r'), 'r'),
     (dict(color='r', edgecolor='g'), 'g'),
     ])
def test_parse_scatter_color_args_edgecolors(kwargs, expected_edgecolors):
    # 定义一个内部函数，用于返回下一个颜色，但当前未使用
    def get_next_color():
        return 'blue'  # currently unused

    # 从 kwargs 中弹出 'c' 和 'edgecolors' 参数，并调用 mpl.axes.Axes._parse_scatter_color_args 方法
    # 解析参数，获取返回值中的 edgecolors
    c = kwargs.pop('c', None)
    edgecolors = kwargs.pop('edgecolors', None)
    _, _, result_edgecolors = \
        mpl.axes.Axes._parse_scatter_color_args(
            c, edgecolors, kwargs, xsize=2, get_next_color_func=get_next_color)
    # 断言返回的 result_edgecolors 符合预期结果
    assert result_edgecolors == expected_edgecolors

# 对于 test_parse_scatter_color_args_error 进行异常情况测试
def test_parse_scatter_color_args_error():
    # 定义一个内部函数，用于返回下一个颜色，但当前未使用
    def get_next_color():
        return 'blue'  # currently unused

    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配错误信息
    with pytest.raises(ValueError,
                       match="RGBA values should be within 0-1 range"):
        # 定义一个超出范围的 RGBA 数组，调用 mpl.axes.Axes._parse_scatter_color_args 方法
        # 期望抛出异常
        c = np.array([[0.1, 0.2, 0.7], [0.2, 0.4, 1.4]])  # value > 1
        mpl.axes.Axes._parse_scatter_color_args(
            c, None, kwargs={}, xsize=2, get_next_color_func=get_next_color)

# 对 test_as_mpl_axes_api 进行测试
def test_as_mpl_axes_api():
    # 定义一个 Polar 类，实现 _as_mpl_axes 接口方法
    class Polar:
        def __init__(self):
            self.theta_offset = 0

        def _as_mpl_axes(self):
            # 实现 matplotlib axes 接口，返回 PolarAxes 类型和 theta_offset 属性
            return PolarAxes, {'theta_offset': self.theta_offset}

    # 创建两个 Polar 实例
    prj = Polar()
    prj2 = Polar()
    prj2.theta_offset = np.pi

    # 使用 plt.axes 方法测试创建轴，期望返回 PolarAxes 类型
    ax = plt.axes((0, 0, 1, 1), projection=prj)
    assert type(ax) is PolarAxes
    plt.close()

    # 使用 plt.subplot 方法测试创建子图，期望返回 PolarAxes 类型
    ax = plt.subplot(121, projection=prj)
    # 断言 ax 的类型是 PolarAxes 类型，用于确保变量 ax 是预期的极坐标轴对象
    assert type(ax) is PolarAxes
    # 关闭当前的图形窗口，清空图形状态
    plt.close()
def test_pyplot_axes():
    # 创建两个子图，返回两个对象：fig1为Figure对象，ax1为Axes对象
    fig1, ax1 = plt.subplots()
    # 创建两个子图，返回两个对象：fig2为Figure对象，ax2为Axes对象
    fig2, ax2 = plt.subplots()
    # 设置当前活动的Axes对象为ax1
    plt.sca(ax1)
    # 断言当前活动的Axes对象是ax1
    assert ax1 is plt.gca()
    # 断言当前活动的Figure对象是fig1
    assert fig1 is plt.gcf()
    # 关闭Figure对象fig1
    plt.close(fig1)
    # 关闭Figure对象fig2
    plt.close(fig2)


def test_log_scales():
    # 创建一个新的Figure对象和一个新的Axes对象
    fig, ax = plt.subplots()
    # 绘制以自然对数为底的线性空间，然后设置y轴为对数尺度，底数为5.5
    ax.plot(np.log(np.linspace(0.1, 100)))
    ax.set_yscale('log', base=5.5)
    # 反转y轴
    ax.invert_yaxis()
    # 设置x轴为对数尺度，底数为9.0，并返回更新后的刻度位置和标签
    xticks, yticks = [
        [(t.get_loc(), t.label1.get_text()) for t in axis._update_ticks()]
        for axis in [ax.xaxis, ax.yaxis]
    ]
    # 断言x轴刻度的位置和标签与预期相符
    assert xticks == [
        (1.0, '$\\mathdefault{9^{0}}$'),
        (9.0, '$\\mathdefault{9^{1}}$'),
        (81.0, '$\\mathdefault{9^{2}}$'),
        (2.0, ''),
        (3.0, ''),
        (4.0, ''),
        (5.0, ''),
        (6.0, ''),
        (7.0, ''),
        (8.0, ''),
        (18.0, ''),
        (27.0, ''),
        (36.0, ''),
        (45.0, ''),
        (54.0, ''),
        (63.0, ''),
        (72.0, ''),
    ]
    # 断言y轴刻度的位置和标签与预期相符
    assert yticks == [
        (0.18181818181818182, '$\\mathdefault{5.5^{-1}}$'),
        (1.0, '$\\mathdefault{5.5^{0}}$'),
        (5.5, '$\\mathdefault{5.5^{1}}$'),
        (0.36363636363636365, ''),
        (0.5454545454545454, ''),
        (0.7272727272727273, ''),
        (0.9090909090909092, ''),
        (2.0, ''),
        (3.0, ''),
        (4.0, ''),
        (5.0, ''),
    ]


def test_log_scales_no_data():
    # 创建一个新的Figure对象和一个新的Axes对象，设置x轴和y轴都为对数尺度
    _, ax = plt.subplots()
    ax.set(xscale="log", yscale="log")
    # 设置x轴主要刻度的定位器为MultipleLocator，步长为1
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    # 断言x轴和y轴的限制范围为(1, 10)
    assert ax.get_xlim() == ax.get_ylim() == (1, 10)


def test_log_scales_invalid():
    # 创建一个新的Figure对象和一个新的Axes对象，设置x轴为对数尺度
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    # 使用pytest断言捕获UserWarning，匹配包含'Attempt to set non-positive'的警告信息
    with pytest.warns(UserWarning, match='Attempt to set non-positive'):
        # 设置x轴的限制范围为(-1, 10)
        ax.set_xlim(-1, 10)
    ax.set_yscale('log')
    # 使用pytest断言捕获UserWarning，匹配包含'Attempt to set non-positive'的警告信息
    with pytest.warns(UserWarning, match='Attempt to set non-positive'):
        # 设置y轴的限制范围为(-1, 10)
        ax.set_ylim(-1, 10)


@image_comparison(['stackplot_test_image', 'stackplot_test_image'],
                  tol=0.031 if platform.machine() == 'arm64' else 0)
def test_stackplot():
    # 创建一个新的Figure对象
    fig = plt.figure()
    # 创建数据
    x = np.linspace(0, 10, 10)
    y1 = 1.0 * x
    y2 = 2.0 * x + 1
    y3 = 3.0 * x + 2
    # 向Figure对象添加一个子图
    ax = fig.add_subplot(1, 1, 1)
    # 绘制堆叠面积图
    ax.stackplot(x, y1, y2, y3)
    # 设置x轴和y轴的限制范围
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 70))

    # 使用以上测试用例进行测试，包含带标签数据和Axes属性循环中的颜色
    data = {"x": x, "y1": y1, "y2": y2, "y3": y3}
    fig, ax = plt.subplots()
    # 绘制堆叠面积图，使用data中的标签和指定颜色
    ax.stackplot("x", "y1", "y2", "y3", data=data, colors=["C0", "C1", "C2"])
    # 设置x轴和y轴的限制范围
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 70))


@image_comparison(['stackplot_test_baseline'], remove_text=True)
def test_stackplot_baseline():
    # 设置随机数种子为0
    np.random.seed(0)
    # 定义一个函数 layers，生成一个大小为 (m, n) 的零矩阵 a
    def layers(n, m):
        a = np.zeros((m, n))
        # 循环 n 次，对每列进行操作
        for i in range(n):
            # 对每列循环 5 次
            for j in range(5):
                # 生成随机数 x，y，z，并根据它们对列向量 a[:, i] 进行加权
                x = 1 / (.1 + np.random.random())
                y = 2 * np.random.random() - .5
                z = 10 / (.1 + np.random.random())
                a[:, i] += x * np.exp(-((np.arange(m) / m - y) * z) ** 2)
        # 返回生成的矩阵 a
        return a

    # 调用 layers 函数生成一个大小为 (3, 100) 的矩阵 d
    d = layers(3, 100)
    
    # 将 d 矩阵第 50 行所有列的值设为 0，用于测试固定加权波动（issue #6313）
    d[50, :] = 0  

    # 创建一个包含四个子图的图形 fig 和 axs
    fig, axs = plt.subplots(2, 2)

    # 在第一个子图 axs[0, 0] 中绘制堆叠区域图，基线为 zero
    axs[0, 0].stackplot(range(100), d.T, baseline='zero')
    
    # 在第二个子图 axs[0, 1] 中绘制堆叠区域图，基线为 sym
    axs[0, 1].stackplot(range(100), d.T, baseline='sym')
    
    # 在第三个子图 axs[1, 0] 中绘制堆叠区域图，基线为 wiggle
    axs[1, 0].stackplot(range(100), d.T, baseline='wiggle')
    
    # 在第四个子图 axs[1, 1] 中绘制堆叠区域图，基线为 weighted_wiggle
    axs[1, 1].stackplot(range(100), d.T, baseline='weighted_wiggle')
# 使用装饰器执行图像比较测试，并确保堆栈图中的填充图案正确 (issue #27146)
@check_figures_equal()
def test_stackplot_hatching(fig_ref, fig_test):
    # 生成一组线性间隔的数据点
    x = np.linspace(0, 10, 10)
    y1 = 1.0 * x
    y2 = 2.0 * x + 1
    y3 = 3.0 * x + 2
    
    # 在测试图中创建子图并绘制堆叠图，每层使用不同的填充图案和白色颜色
    ax_test = fig_test.subplots()
    ax_test.stackplot(x, y1, y2, y3, hatch=["x", "//", "\\\\"], colors=["white"])
    ax_test.set_xlim((0, 10))
    ax_test.set_ylim((0, 70))
    
    # 在参考图中创建子图并逐层填充，每层使用不同的填充图案和白色颜色，用于比较
    stack_baseline = np.zeros(len(x))
    ax_ref = fig_ref.subplots()
    ax_ref.fill_between(x, stack_baseline, y1, hatch="x", facecolor="white")
    ax_ref.fill_between(x, y1, y1+y2, hatch="//", facecolor="white")
    ax_ref.fill_between(x, y1+y2, y1+y2+y3, hatch="\\\\", facecolor="white")
    ax_ref.set_xlim((0, 10))
    ax_ref.set_ylim((0, 70))


# 辅助函数，用于执行箱线图的测试，生成随机数据并创建相应的箱线图
def _bxp_test_helper(
        stats_kwargs={}, transform_stats=lambda s: s, bxp_kwargs={}):
    np.random.seed(937)
    # 生成对数正态分布的随机数据并计算箱线图统计信息
    logstats = mpl.cbook.boxplot_stats(
        np.random.lognormal(mean=1.25, sigma=1., size=(37, 4)), **stats_kwargs)
    # 创建新的图形和轴对象
    fig, ax = plt.subplots()
    # 根据箱线图参数设置轴的比例尺
    if bxp_kwargs.get('orientation', 'vertical') == 'vertical':
        ax.set_yscale('log')
    else:
        ax.set_xscale('log')
    # 当 patch_artist 为 False 时，修复旧版本中的问题，确保箱线图的线宽度参数生效
    if not bxp_kwargs.get('patch_artist', False):
        mpl.rcParams['boxplot.boxprops.linewidth'] = \
            mpl.rcParams['lines.linewidth']
    # 绘制箱线图，使用转换后的统计信息
    ax.bxp(transform_stats(logstats), **bxp_kwargs)


# 使用装饰器进行图像比较测试，测试默认箱线图生成的基线图
@image_comparison(['bxp_baseline.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_baseline():
    _bxp_test_helper()


# 使用装饰器进行图像比较测试，测试自定义 whisker 范围的箱线图生成
@image_comparison(['bxp_rangewhis.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_rangewhis():
    _bxp_test_helper(stats_kwargs=dict(whis=[0, 100]))


# 使用装饰器进行图像比较测试，测试百分位数 whisker 范围的箱线图生成
@image_comparison(['bxp_percentilewhis.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_percentilewhis():
    _bxp_test_helper(stats_kwargs=dict(whis=[5, 95]))


# 使用装饰器进行图像比较测试，测试带有自定义标签的箱线图生成
@image_comparison(['bxp_with_xlabels.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_with_xlabels():
    # 定义转换函数，将箱线图统计信息与自定义标签关联
    def transform(stats):
        for s, label in zip(stats, list('ABCD')):
            s['label'] = label
        return stats
    _bxp_test_helper(transform_stats=transform)


# 使用装饰器进行图像比较测试，测试水平方向的箱线图生成
@image_comparison(['bxp_horizontal.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default',
                  tol=0.1)
def test_bxp_horizontal():
    _bxp_test_helper(bxp_kwargs=dict(orientation='horizontal'))


# 使用装饰器进行图像比较测试，测试带有自定义 y 轴标签的箱线图生成
@image_comparison(['bxp_with_ylabels.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default',
                  tol=0.1)
def test_bxp_with_ylabels():
    _bxp_test_helper()
    # 定义一个函数 `transform`，接收一个名为 `stats` 的参数
    def transform(stats):
        # 使用 zip 函数将 stats 和列表 'ABCD' 中的元素一一对应，并遍历它们
        for s, label in zip(stats, list('ABCD')):
            # 为每个字典 s 添加一个键为 'label' 的新键值对，值为当前循环的 label 字符
            s['label'] = label
        # 返回经过标记处理后的 stats 列表
        return stats
    
    # 调用名为 `_bxp_test_helper` 的函数，并传入以下两个参数:
    # - `transform_stats` 参数，使用之前定义的 `transform` 函数
    # - `bxp_kwargs` 参数，包含一个键为 'orientation'，值为字符串 'horizontal' 的字典
    _bxp_test_helper(transform_stats=transform,
                     bxp_kwargs=dict(orientation='horizontal'))
# 使用 image_comparison 装饰器，比较测试结果图像是否匹配，并保存结果图像为 'bxp_patchartist.png'，同时移除图像中的文本
@image_comparison(['bxp_patchartist.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_patchartist():
    # 调用 _bxp_test_helper 函数，传递 patch_artist=True 的参数给 bxp_kwargs
    _bxp_test_helper(bxp_kwargs=dict(patch_artist=True))


# 使用 image_comparison 装饰器，比较测试结果图像是否匹配，并保存结果图像为 'bxp_custompatchartist.png'，同时移除图像中的文本
@image_comparison(['bxp_custompatchartist.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 100},
                  style='default')
def test_bxp_custompatchartist():
    # 调用 _bxp_test_helper 函数，传递 patch_artist=True 和自定义 boxprops 参数给 bxp_kwargs
    _bxp_test_helper(bxp_kwargs=dict(
        patch_artist=True,
        boxprops=dict(facecolor='yellow', edgecolor='green', ls=':')))


# 使用 image_comparison 装饰器，比较测试结果图像是否匹配，并保存结果图像为 'bxp_customoutlier.png'，同时移除图像中的文本
@image_comparison(['bxp_customoutlier.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_customoutlier():
    # 调用 _bxp_test_helper 函数，传递自定义 flierprops 参数给 bxp_kwargs
    _bxp_test_helper(bxp_kwargs=dict(
        flierprops=dict(linestyle='none', marker='d', mfc='g')))


# 使用 image_comparison 装饰器，比较测试结果图像是否匹配，并保存结果图像为 'bxp_withmean_custompoint.png'，同时移除图像中的文本
@image_comparison(['bxp_withmean_custompoint.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_showcustommean():
    # 调用 _bxp_test_helper 函数，传递 showmeans=True 和自定义 meanprops 参数给 bxp_kwargs
    _bxp_test_helper(bxp_kwargs=dict(
        showmeans=True,
        meanprops=dict(linestyle='none', marker='d', mfc='green'),
    ))


# 使用 image_comparison 装饰器，比较测试结果图像是否匹配，并保存结果图像为 'bxp_custombox.png'，同时移除图像中的文本
@image_comparison(['bxp_custombox.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_custombox():
    # 调用 _bxp_test_helper 函数，传递自定义 boxprops 参数给 bxp_kwargs
    _bxp_test_helper(bxp_kwargs=dict(
        boxprops=dict(linestyle='--', color='b', lw=3)))


# 使用 image_comparison 装饰器，比较测试结果图像是否匹配，并保存结果图像为 'bxp_custommedian.png'，同时移除图像中的文本
@image_comparison(['bxp_custommedian.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_custommedian():
    # 调用 _bxp_test_helper 函数，传递自定义 medianprops 参数给 bxp_kwargs
    _bxp_test_helper(bxp_kwargs=dict(
        medianprops=dict(linestyle='--', color='b', lw=3)))


# 使用 image_comparison 装饰器，比较测试结果图像是否匹配，并保存结果图像为 'bxp_customcap.png'，同时移除图像中的文本
@image_comparison(['bxp_customcap.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_customcap():
    # 调用 _bxp_test_helper 函数，传递自定义 capprops 参数给 bxp_kwargs
    _bxp_test_helper(bxp_kwargs=dict(
        capprops=dict(linestyle='--', color='g', lw=3)))


# 使用 image_comparison 装饰器，比较测试结果图像是否匹配，并保存结果图像为 'bxp_customwhisker.png'，同时移除图像中的文本
@image_comparison(['bxp_customwhisker.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_customwhisker():
    # 调用 _bxp_test_helper 函数，传递自定义 whiskerprops 参数给 bxp_kwargs
    _bxp_test_helper(bxp_kwargs=dict(
        whiskerprops=dict(linestyle='-', color='m', lw=3)))


# 使用 check_figures_equal 装饰器，检查两个图形是否相等
@check_figures_equal()
def test_boxplot_median_bound_by_box(fig_test, fig_ref):
    # 创建数据数组
    data = np.arange(3)
    # 定义 medianprops_test 和 medianprops_ref 参数
    medianprops_test = {"linewidth": 12}
    medianprops_ref = {**medianprops_test, "solid_capstyle": "butt"}
    # 在 fig_test 上创建子图并绘制箱线图，传递 medianprops_test 参数给 medianprops
    fig_test.subplots().boxplot(data,  medianprops=medianprops_test)
    # 在 fig_ref 上创建子图并绘制箱线图，传递 medianprops_ref 参数给 medianprops
    fig_ref.subplots().boxplot(data, medianprops=medianprops_ref)


# 使用 image_comparison 装饰器，比较测试结果图像是否匹配，并保存结果图像为 'bxp_withnotch.png'，同时移除图像中的文本
@image_comparison(['bxp_withnotch.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_shownotches():
    # 调用 _bxp_test_helper 函数，传递 shownotches=True 的参数给 bxp_kwargs
    _bxp_test_helper(bxp_kwargs=dict(shownotches=True))
@image_comparison(['bxp_nocaps.png'],  # 用于比较图像，检查没有箱线图顶部和底部的线条
                  remove_text=True,   # 移除图像中的文本信息
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_nocaps():
    _bxp_test_helper(bxp_kwargs=dict(showcaps=False))  # 调用辅助函数，设置箱线图参数不显示顶部和底部的线条


@image_comparison(['bxp_nobox.png'],  # 用于比较图像，检查没有箱线图的箱体
                  remove_text=True,   # 移除图像中的文本信息
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_nobox():
    _bxp_test_helper(bxp_kwargs=dict(showbox=False))  # 调用辅助函数，设置箱线图参数不显示箱体


@image_comparison(['bxp_no_flier_stats.png'],  # 用于比较图像，检查没有异常值的箱线图统计数据
                  remove_text=True,   # 移除图像中的文本信息
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_no_flier_stats():
    def transform(stats):
        for s in stats:
            s.pop('fliers', None)   # 从统计数据中移除异常值（fliers）
        return stats

    _bxp_test_helper(transform_stats=transform,  # 调用辅助函数，并传入自定义的统计数据转换函数
                     bxp_kwargs=dict(showfliers=False))  # 设置箱线图参数不显示异常值


@image_comparison(['bxp_withmean_point.png'],  # 用于比较图像，显示均值点的箱线图
                  remove_text=True,   # 移除图像中的文本信息
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_showmean():
    _bxp_test_helper(bxp_kwargs=dict(showmeans=True, meanline=False))  # 调用辅助函数，设置显示均值点，但不显示均值线


@image_comparison(['bxp_withmean_line.png'],  # 用于比较图像，显示均值线的箱线图
                  remove_text=True,   # 移除图像中的文本信息
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_showmeanasline():
    _bxp_test_helper(bxp_kwargs=dict(showmeans=True, meanline=True))  # 调用辅助函数，设置显示均值点和均值线


@image_comparison(['bxp_scalarwidth.png'],  # 用于比较图像，显示标量宽度的箱线图
                  remove_text=True,   # 移除图像中的文本信息
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_scalarwidth():
    _bxp_test_helper(bxp_kwargs=dict(widths=.25))  # 调用辅助函数，设置箱线图的箱体宽度为0.25


@image_comparison(['bxp_customwidths.png'],  # 用于比较图像，显示自定义宽度的箱线图
                  remove_text=True,   # 移除图像中的文本信息
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_customwidths():
    _bxp_test_helper(bxp_kwargs=dict(widths=[0.10, 0.25, 0.65, 0.85]))  # 调用辅助函数，设置自定义的箱体宽度列表


@image_comparison(['bxp_custompositions.png'],  # 用于比较图像，显示自定义位置的箱线图
                  remove_text=True,   # 移除图像中的文本信息
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_custompositions():
    _bxp_test_helper(bxp_kwargs=dict(positions=[1, 5, 6, 7]))  # 调用辅助函数，设置自定义的箱体位置列表


def test_bxp_bad_widths():
    with pytest.raises(ValueError):  # 断言捕获异常，确保以下代码块抛出 ValueError 异常
        _bxp_test_helper(bxp_kwargs=dict(widths=[1]))  # 调用辅助函数，设置错误的箱体宽度参数


def test_bxp_bad_positions():
    with pytest.raises(ValueError):  # 断言捕获异常，确保以下代码块抛出 ValueError 异常
        _bxp_test_helper(bxp_kwargs=dict(positions=[2, 3]))  # 调用辅助函数，设置错误的箱体位置参数


@image_comparison(['bxp_custom_capwidths.png'],  # 用于比较图像，显示自定义顶部线宽度的箱线图
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_custom_capwidths():
    _bxp_test_helper(bxp_kwargs=dict(capwidths=[0.0, 0.1, 0.5, 1.0]))  # 调用辅助函数，设置自定义的顶部线宽度列表


@image_comparison(['bxp_custom_capwidth.png'],  # 用于比较图像，显示单一顶部线宽度的箱线图
                  savefig_kwarg={'dpi': 40},  # 保存图像时的分辨率设置为40 DPI
                  style='default')    # 使用默认的风格样式进行绘图
def test_bxp_custom_capwidth():
    _bxp_test_helper(bxp_kwargs=dict(capwidths=0.6))  # 调用辅助函数，设置单一的顶部线宽度


def test_bxp_bad_capwidths():
    with pytest.raises(ValueError):  # 断言捕获异常，确保以下代码块抛出 ValueError 异常
        _bxp_test_helper(bxp_kwargs=dict(capwidths=[1]))  # 调用辅助函数，设置错误的顶部线宽度参数
```python`
@image_comparison(['boxplot', 'boxplot'], tol=1.28, style='default')
# 定义一个测试函数，比较两个图形，使用指定的容差和默认样式
def test_boxplot():
    # 设置随机种子，用于重现结果
    np.random.seed(937)

    # 生成一维数组 x，范围从 -7 到 7，共 140 个点
    x = np.linspace(-7, 7, 140)
    # 在 x 数组两端添加额外的值 -25 和 25
    x = np.hstack([-25, x, 25])
    # 创建一个图形对象和一个坐标轴对象
    fig, ax = plt.subplots()

    # 绘制箱线图，数据源为两个相同的 x 数组，使用 10000 次自助重抽样，带有缺口
    ax.boxplot([x, x], bootstrap=10000, notch=1)
    # 设置 y 轴的显示范围
    ax.set_ylim((-30, 30))

    # 为标记数据的测试，创建另一个图形对象和坐标轴对象
    data = {"x": [x, x]}
    fig, ax = plt.subplots()
    # 绘制箱线图，数据源为标记数据 "x"，使用 10000 次自助重抽样，带有缺口
    ax.boxplot("x", bootstrap=10000, notch=1, data=data)
    # 设置 y 轴的显示范围
    ax.set_ylim((-30, 30))


@check_figures_equal(extensions=["png"])
# 定义一个测试函数，检查两个图形是否相等，输出为 PNG 格式
def test_boxplot_masked(fig_test, fig_ref):
    # 检查箱线图绘制时是否忽略掩码值
    x_orig = np.linspace(-1, 1, 200)

    # 在测试图形上创建坐标轴对象
    ax = fig_test.subplots()
    # 从 x_orig 中选取大于等于 0 的部分，并绘制箱线图
    x = x_orig[x_orig >= 0]
    ax.boxplot(x)

    # 创建参考图形上的坐标轴对象
    ax = fig_ref.subplots()
    # 使用掩码将 x_orig 中小于 0 的部分屏蔽，并绘制箱线图
    ax.boxplot(np.ma.masked_less(x_orig, 0))


@image_comparison(['boxplot_custom_capwidths.png'],
                  savefig_kwarg={'dpi': 40}, style='default')
# 定义一个测试函数，比较箱线图与自定义的顶端线宽度的效果
def test_boxplot_custom_capwidths():

    # 生成一维数组 x，范围从 -7 到 7，共 140 个点
    x = np.linspace(-7, 7, 140)
    # 在 x 数组两端添加额外的值 -25 和 25
    x = np.hstack([-25, x, 25])
    # 创建一个图形对象和一个坐标轴对象
    fig, ax = plt.subplots()

    # 绘制箱线图，数据源为两个相同的 x 数组，带有缺口，自定义顶端线宽度
    ax.boxplot([x, x], notch=1, capwidths=[0.01, 0.2])


@image_comparison(['boxplot_sym2.png'], remove_text=True, style='default')
# 定义一个测试函数，比较不同符号形状的箱线图效果
def test_boxplot_sym2():
    # 设置随机种子，用于重现结果
    np.random.seed(937)

    # 生成一维数组 x，范围从 -7 到 7，共 140 个点
    x = np.linspace(-7, 7, 140)
    # 在 x 数组两端添加额外的值 -25 和 25
    x = np.hstack([-25, x, 25])
    # 创建一个包含两个坐标轴对象的图形对象
    fig, [ax1, ax2] = plt.subplots(1, 2)

    # 在第一个坐标轴上绘制箱线图，数据源为两个相同的 x 数组，使用 10000 次自助重抽样，符号为 '^'
    ax1.boxplot([x, x], bootstrap=10000, sym='^')
    # 设置 y 轴的显示范围
    ax1.set_ylim((-30, 30))

    # 在第二个坐标轴上绘制箱线图，数据源为两个相同的 x 数组，使用 10000 次自助重抽样，符号为 'g'
    ax2.boxplot([x, x], bootstrap=10000, sym='g')
    # 设置 y 轴的显示范围
    ax2.set_ylim((-30, 30))


@image_comparison(['boxplot_sym.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
# 定义一个测试函数，比较带有自定义符号的箱线图效果
def test_boxplot_sym():
    # 生成一维数组 x，范围从 -7 到 7，共 140 个点
    x = np.linspace(-7, 7, 140)
    # 在 x 数组两端添加额外的值 -25 和 25
    x = np.hstack([-25, x, 25])
    # 创建一个图形对象和一个坐标轴对象
    fig, ax = plt.subplots()

    # 绘制箱线图，数据源为两个相同的 x 数组，使用自定义符号 'gs'
    ax.boxplot([x, x], sym='gs')
    # 设置 y 轴的显示范围
    ax.set_ylim((-30, 30))


@image_comparison(['boxplot_autorange_false_whiskers.png',
                   'boxplot_autorange_true_whiskers.png'],
                  style='default')
# 定义一个测试函数，比较是否自动调整箱线图的数据范围对于绘制须的影响
def test_boxplot_autorange_whiskers():
    # 设置随机种子，用于重现结果
    np.random.seed(937)

    # 生成一维数组 x，包含 140 个值，大部分为 1，头尾各有一个 0 和 2
    x = np.ones(140)
    x = np.hstack([0, x, 2])

    # 创建第一个图形对象和坐标轴对象
    fig1, ax1 = plt.subplots()
    # 绘制箱线图，数据源为两个相同的 x 数组，使用 10000 次自助重抽样，带有缺口
    ax1.boxplot([x, x], bootstrap=10000, notch=1)
    # 设置 y 轴的显示范围
    ax1.set_ylim((-5, 5))

    # 创建第二个图形对象和坐标轴对象
    fig2, ax2 = plt.subplots()
    # 绘制箱线图，数据源为两个相同的 x 数组，使用 10000 次自助重抽样，带有缺口，自动调整数据范围
    ax2.boxplot([x, x], bootstrap=10000, notch=1, autorange=True)
    # 设置 y 轴的显示范围
    ax2.set_ylim((-5, 5))


def _rc_test_bxp_helper(ax, rc_dict):
    # 生成一维数组 x，范围从 -7 到 7，共 140 个点
    x = np.linspace(-7, 7, 140)
    # 在 x 数组两端添加额外的值 -25 和 25
    x = np.hstack([-25, x, 25])
    # 使用给定的 rc 参数字典上下文，绘制箱线图，并返回修改后的坐标轴对象
    with matplotlib.rc_context(rc_dict):
        ax.boxplot([x, x])
    return ax


@image_comparison(['boxplot_rc_parameters'],
                  savefig_kwarg={'dpi': 100}, remove_text=True,
                  tol=1, style='default')
# 定义一个测试函数，检查在不同 rc 参数下绘制的箱线图效果
def test_boxplot_rc_parameters():
    # 设置```python
@image_comparison(['boxplot', 'boxplot'], tol=1.28, style='default')
def test_boxplot():
    # 设置随机种子以保证结果可复现
    np.random.seed(937)

    # 创建一个从-7到7的等差数列，共140个点
    x = np.linspace(-7, 7, 140)
    # 在数组前后各加入-25和25两个点
    x = np.hstack([-25, x, 25])
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 绘制箱线图，包括两个箱线图，使用10000次bootstrap，显示缺口
    ax.boxplot([x, x], bootstrap=10000, notch=1)
    # 设置y轴的显示范围
    ax.set_ylim((-30, 30))

    # 重复上面的测试用例，用于带标签数据的测试
    data = {"x": [x, x]}
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制箱线图，从data字典中获取"x"键对应的数据，使用10000次bootstrap，显示缺口
    ax.boxplot("x", bootstrap=10000, notch=1, data=data)
    # 设置y轴的显示范围
    ax.set_ylim((-30, 30))


@check_figures_equal(extensions=["png"])
def test_boxplot_masked(fig_test, fig_ref):
    # 检查在绘制箱线图时屏蔽值被忽略的情况
    x_orig = np.linspace(-1, 1, 200)

    # 从测试图形获取子图对象
    ax = fig_test.subplots()
    # 创建一个x数组，只包括大于等于0的值
    x = x_orig[x_orig >= 0]
    # 绘制箱线图
    ax.boxplot(x)

    # 创建一个屏蔽小于0值的x数组
    x = np.ma.masked_less(x_orig, 0)
    # 从参考图形获取子图对象
    ax = fig_ref.subplots()
    # 绘制箱线图
    ax.boxplot(x)


@image_comparison(['boxplot_custom_capwidths.png'],
                  savefig_kwarg={'dpi': 40}, style='default')
def test_boxplot_custom_capwidths():

    # 创建一个从-7到7的等差数列，共140个点
    x = np.linspace(-7, 7, 140)
    # 在数组前后各加入-25和25两个点
    x = np.hstack([-25, x, 25])
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 绘制箱线图，包括两个箱线图，显示缺口，并指定自定义的capwidths参数
    ax.boxplot([x, x], notch=1, capwidths=[0.01, 0.2])


@image_comparison(['boxplot_sym2.png'], remove_text=True, style='default')
def test_boxplot_sym2():
    # 设置随机种子以保证结果可复现
    np.random.seed(937)

    # 创建一个从-7到7的等差数列，共140个点
    x = np.linspace(-7, 7, 140)
    # 在数组前后各加入-25和25两个点
    x = np.hstack([-25, x, 25])
    # 创建包含1行2列的子图对象
    fig, [ax1, ax2] = plt.subplots(1, 2)

    # 在第一个子图上绘制两个箱线图，包括10000次bootstrap，显示帽子为'^'
    ax1.boxplot([x, x], bootstrap=10000, sym='^')
    # 设置y轴的显示范围
    ax1.set_ylim((-30, 30))

    # 在第二个子图上绘制两个箱线图，包括10000次bootstrap，显示帽子为'g'
    ax2.boxplot([x, x], bootstrap=10000, sym='g')
    # 设置y轴的显示范围
    ax2.set_ylim((-30, 30))


@image_comparison(['boxplot_sym.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_boxplot_sym():
    # 创建一个从-7到7的等差数列，共140个点
    x = np.linspace(-7, 7, 140)
    # 在数组前```python
@image_comparison(['boxplot', 'boxplot'], tol=1.28, style='default')
def test_boxplot():
    # 设置随机种子以保证结果可复现
    np.random.seed(937)

    # 创建一个从-7到7的等差数列，共140个点
    x = np.linspace(-7, 7, 140)
    # 在数组前后各加入-25和25两个点
    x = np.hstack([-25, x, 25])
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 绘制箱线图，包括两个箱线图，使用10000次bootstrap，显示缺口
    ax.boxplot([x, x], bootstrap=10000, notch=1)
    # 设置y轴的显示范围
    ax.set_ylim((-30, 30))

    # 重复上面的测试用例，用于带标签数据的测试
    data = {"x": [x, x]}
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制箱线图，从data字典中获取"x"键对应的数据，使用10000次bootstrap，显示缺口
    ax.boxplot("x", bootstrap=10000, notch=1, data=data)
    # 设置y轴的显示范围
    ax.set_ylim((-30, 30))


@check_figures_equal(extensions=["png"])
def test_boxplot_masked(fig_test, fig_ref):
    # 检查在绘制箱线图时屏蔽值被忽略的情况
    x_orig = np.linspace(-1, 1, 200)

    # 从测试图形获取子图对象
    ax = fig_test.subplots()
    # 创建一个x数组，只包括大于等于0的值
    x = x_orig[x_orig >= 0]
    # 绘制箱线图
    ax.boxplot(x)

    # 创建一个屏蔽小于0值的x数组
    x = np.ma.masked_less(x_orig, 0)
    # 从参考图形获取子图对象
    ax = fig_ref.subplots()
    # 绘制箱线图
    ax.boxplot(x)


@image_comparison(['boxplot_custom_capwidths.png'],
                  savefig_kwarg={'dpi': 40}, style='default')
def test_boxplot_custom_capwidths():

    # 创建一个从-7到7的等差数列，共140个点
    x = np.linspace(-7, 7, 140)
    # 在数组前后各加入-25和25两个点
    x = np.hstack([-25, x, 25])
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 绘制箱线图，包括两个箱线图，显示缺口，并指定自定义的capwidths参数
    ax.boxplot([x, x], notch=1, capwidths=[0.01, 0.2])


@image_comparison(['boxplot_sym2.png'], remove_text=True, style='default')
def test_boxplot_sym2():
    # 设置随机种子以保证结果可复现
    np.random.seed(937)

    # 创建一个从-7到7的等差数列，共140个点
    x = np.linspace(-7, 7, 140)
    # 在数组前后各加入-25和25两个点
    x = np.hstack([-25, x, 25])
    # 创建包含1行2列的子图对象
    fig, [ax1, ax2] = plt.subplots(1, 2)

    # 在第一个子图上绘制两个箱线图，包括10000次bootstrap，显示帽子为'^'
    ax1.boxplot([x, x], bootstrap=10000, sym='^')
    # 设置y轴的显示范围
    ax1.set_ylim((-30, 30))

    # 在第二个子图上绘制两个箱线图，包括10000次bootstrap，显示帽子为'g'
    ax2.boxplot([x, x], bootstrap=10000, sym='g')
    # 设置y轴的显示范围
    ax2.set_ylim((-30, 30))


@image_comparison(['boxplot_sym.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_boxplot_sym():
    # 创建一个从-7到7的等差数列，共140个点
    x = np.linspace(-7, 7, 140)
    # 在数组前后各加入-25和25两个点
    x = np.hstack([-25, x, 25])
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 绘制两个箱线图，设置帽子为绿色方块
    ax.boxplot([x, x], sym='gs')
    # 设置y轴的显示范围
    ax.set_ylim((-30, 30))


@image_comparison(['boxplot_autorange_false_whiskers.png',
                   'boxplot_autorange_true_whiskers.png'],
                  style='default')
def test_boxplot_autorange_whiskers():
    # 设置随机种子以保证结果可复现
    np.random.seed(937)

    # 创建一个含有140个1和1个0、2的数组
    x = np.ones(140)
    x = np.hstack([0, x, 2])

    # 创建第一个图形和轴对象
    fig1, ax1 = plt.subplots()
    # 绘制两个箱线图，包括10000次bootstrap，显示缺口
    ax1.boxplot([x, x], bootstrap=10000, notch=1)
    # 设置y轴的显示范围
    ax1.set_ylim((-5, 5))

    # 创建第二个图形和轴对象
    fig2, ax2 = plt.subplots()
    # 绘制两个箱线图，包括10000次bootstrap，显示缺口，自动调整whiskers
    ax2.boxplot([x, x], bootstrap=10000, notch=1, autorange=True)
    # 设置y轴的显示范围
    ax2.set_ylim((-5, 5))


def _rc_test_bxp_helper(ax, rc_dict):
    # 创建一个从-7到7的等差数列，共140个点
    x = np.linspace(-7, 7, 140)
    # 在数组前后各加入-25和25两个点
    x = np.hstack([-25, x, 25])
    # 使用指定的rc参数绘制箱线图
    with matplotlib.rc_context(rc_dict):
        ax.boxplot([x, x])
    return ax


@image_comparison(['boxplot_rc_parameters'],
                  savefig_kwarg={'dpi': 100}, remove_text=True,
                  tol=1, style='default')
def test_boxplot_rc_parameters():
    # 设置随机种子以保证结果可复现
    np.random.seed(937)

    # 创建包含3行的图形和轴对象
    fig, ax = plt.subplots(3)
    rc_axis0 = {
        'boxplot.notch': True,                   # 设置箱线图是否显示缺口
        'boxplot.whiskers': [5, 95],             # 设置箱线图须的百分位数范围
        'boxplot.bootstrap': 10000,              # 设置箱线图计算置信区间时的bootstrap重抽样次数

        'boxplot.flierprops.color': 'b',         # 设置离群值标记的颜色为蓝色
        'boxplot.flierprops.marker': 'o',        # 设置离群值标记的形状为圆圈
        'boxplot.flierprops.markerfacecolor': 'g',# 设置离群值标记的填充颜色为绿色
        'boxplot.flierprops.markeredgecolor': 'b',# 设置离群值标记的边缘颜色为蓝色
        'boxplot.flierprops.markersize': 5,       # 设置离群值标记的大小为5
        'boxplot.flierprops.linestyle': '--',    # 设置离群值标记的线条风格为虚线
        'boxplot.flierprops.linewidth': 2.0,     # 设置离群值标记的线条宽度为2.0

        'boxplot.boxprops.color': 'r',           # 设置箱体边框的颜色为红色
        'boxplot.boxprops.linewidth': 2.0,       # 设置箱体边框的线条宽度为2.0
        'boxplot.boxprops.linestyle': '--',      # 设置箱体边框的线条风格为虚线

        'boxplot.capprops.color': 'c',           # 设置箱线图顶端和底端边界线的颜色为青色
        'boxplot.capprops.linewidth': 2.0,       # 设置箱线图顶端和底端边界线的线条宽度为2.0
        'boxplot.capprops.linestyle': '--',      # 设置箱线图顶端和底端边界线的线条风格为虚线

        'boxplot.medianprops.color': 'k',        # 设置中位数标记线的颜色为黑色
        'boxplot.medianprops.linewidth': 2.0,    # 设置中位数标记线的线条宽度为2.0
        'boxplot.medianprops.linestyle': '--',   # 设置中位数标记线的线条风格为虚线
    }

    rc_axis1 = {
        'boxplot.whiskers': [0, 100],            # 设置箱线图须的百分位数范围为[0, 100]
        'boxplot.patchartist': True,             # 设置箱线图是否使用PatchArtist绘制
    }

    rc_axis2 = {
        'boxplot.whiskers': 2.0,                 # 设置箱线图须的长度倍数为2.0
        'boxplot.showcaps': False,               # 不显示箱线图的顶端和底端边界线
        'boxplot.showbox': False,                # 不显示箱线图的箱体
        'boxplot.showfliers': False,             # 不显示箱线图的离群值
        'boxplot.showmeans': True,               # 显示箱线图的均值
        'boxplot.meanline': True,                # 在箱线图中显示均值线

        'boxplot.meanprops.color': 'c',          # 设置均值线的颜色为青色
        'boxplot.meanprops.linewidth': 2.0,      # 设置均值线的线条宽度为2.0
        'boxplot.meanprops.linestyle': '--',     # 设置均值线的线条风格为虚线

        'boxplot.whiskerprops.color': 'r',       # 设置须线的颜色为红色
        'boxplot.whiskerprops.linewidth': 2.0,   # 设置须线的线条宽度为2.0
        'boxplot.whiskerprops.linestyle': '-.',  # 设置须线的线条风格为点划线
    }
    dict_list = [rc_axis0, rc_axis1, rc_axis2]    # 创建包含三个配置字典的列表
    for axis, rc_axis in zip(ax, dict_list):      # 遍历每个轴和其对应的配置字典
        _rc_test_bxp_helper(axis, rc_axis)        # 调用辅助函数处理每个轴的配置

    assert (matplotlib.patches.PathPatch in      # 断言箱线图的图形对象中包含PathPatch类型
            [type(t) for t in ax[1].get_children()])
@image_comparison(['boxplot_with_CIarray.png'],
                  remove_text=True, savefig_kwarg={'dpi': 40}, style='default')
# 定义一个测试函数，用于生成带有置信区间数组的箱线图
def test_boxplot_with_CIarray():
    # 设置随机种子，用于bootstrap过程中的随机性
    np.random.seed(937)

    # 创建一个从-7到7的等差数列，共140个点，再加上额外的两个点-25和25
    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 定义置信区间的数组
    CIs = np.array([[-1.5, 3.], [-1., 3.5]])

    # 绘制一个带有Matplotlib默认中位数和置信区间的箱线图，另一个使用手动指定的值
    ax.boxplot([x, x], bootstrap=10000, usermedians=[None, 1.0],
               conf_intervals=CIs, notch=1)
    # 设置y轴的显示范围
    ax.set_ylim((-30, 30))


@image_comparison(['boxplot_no_inverted_whisker.png'],
                  remove_text=True, savefig_kwarg={'dpi': 40}, style='default')
# 定义一个测试函数，生成不奇怪的箱线图（不会出现反向的须）
def test_boxplot_no_weird_whisker():
    # 创建一个包含不同数值的numpy数组x，数据类型为float64
    x = np.array([3, 9000, 150, 88, 350, 200000, 1400, 960],
                 dtype=np.float64)
    # 创建一个图形轴对象
    ax1 = plt.axes()
    # 绘制箱线图
    ax1.boxplot(x)
    # 设置y轴的比例为对数尺度
    ax1.set_yscale('log')
    # 关闭y轴上的次要网格线
    ax1.yaxis.grid(False, which='minor')
    # 关闭x轴上的网格线


# 定义一个测试函数，测试箱线图中不良的中位数设置
def test_boxplot_bad_medians():
    # 创建一个从-7到7的等差数列，共140个点，再加上额外的两个点-25和25
    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 使用pytest断言检查是否引发了值错误异常
    with pytest.raises(ValueError):
        ax.boxplot(x, usermedians=[1, 2])
    with pytest.raises(ValueError):
        ax.boxplot([x, x], usermedians=[[1, 2], [1, 2]])


# 定义一个测试函数，测试箱线图中不良的置信区间设置
def test_boxplot_bad_ci():
    # 创建一个从-7到7的等差数列，共140个点，再加上额外的两个点-25和25
    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 使用pytest断言检查是否引发了值错误异常
    with pytest.raises(ValueError):
        ax.boxplot([x, x], conf_intervals=[[1, 2]])
    with pytest.raises(ValueError):
        ax.boxplot([x, x], conf_intervals=[[1, 2], [1]])


# 定义一个测试函数，测试箱线图中Z轴顺序设置
def test_boxplot_zorder():
    # 创建一个从0到9的整数数组
    x = np.arange(10)
    # 创建图形和轴对象
    fix, ax = plt.subplots()
    # 使用断言确保第一个箱线图对象的Z轴顺序为2
    assert ax.boxplot(x)['boxes'][0].get_zorder() == 2
    assert ax.boxplot(x, zorder=10)['boxes'][0].get_zorder() == 10


# 定义一个测试函数，测试箱线图中标记行为设置
def test_boxplot_marker_behavior():
    # 设置默认的线条标记样式为's'
    plt.rcParams['lines.marker'] = 's'
    # 设置箱线图中异常值标记的样式为'o'
    plt.rcParams['boxplot.flierprops.marker'] = 'o'
    # 设置箱线图中均值标记的样式为'^'
    plt.rcParams['boxplot.meanprops.marker'] = '^'
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 创建一个包含100个元素的整数数组作为测试数据
    test_data = np.arange(100)
    # 将最后一个数据点改为150，成为一个异常值
    test_data[-1] = 150  # a flier point
    # 绘制带有均值标记的箱线图
    bxp_handle = ax.boxplot(test_data, showmeans=True)
    # 遍历箱线图中的每一条线，确保rcParams['lines.marker']被空字符串覆盖
    for bxp_lines in ['whiskers', 'caps', 'boxes', 'medians']:
        for each_line in bxp_handle[bxp_lines]:
            assert each_line.get_marker() == ''

    # 确保异常值标记和均值标记不受空字符串的影响
    assert bxp_handle['fliers'][0].get_marker() == 'o'
    assert bxp_handle['means'][0].get_marker() == '^'


@image_comparison(['boxplot_mod_artists_after_plotting.png'],
                  remove_text=True, savefig_kwarg={'dpi': 40}, style='default')
# 定义一个测试函数，在绘图后修改艺术对象的箱线图
def test_boxplot_mod_artist_after_plotting():
    # 创建一个包含浮点数的列表x
    x = [0.15, 0.11, 0.06, 0.06, 0.12, 0.56, -0.56]
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 绘制箱线图，并指定异常值的标记为'o'
    bp = ax.boxplot(x, sym="o")
    # 遍历箱线图的每个元素，将它们的颜色设置为绿色
    for key in bp:
        for obj in bp[key]:
            obj.set_color('green')
@image_comparison(['violinplot_vert_baseline.png',
                   'violinplot_vert_baseline.png'])
# 定义一个测试函数，用于比较垂直小提琴图的基线图像
def test_vert_violinplot_baseline():
    # 设置随机种子为 frac(sqrt(2)) 的前9位数字
    np.random.seed(414213562)
    # 生成四组包含各自100个随机数的正态分布数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 创建一个坐标轴对象
    ax = plt.axes()
    # 绘制垂直小提琴图，不显示均值、极值和中位数
    ax.violinplot(data, positions=range(4), showmeans=False, showextrema=False,
                  showmedians=False)

    # 使用上面生成的数据字典进行标记数据的测试
    data = {"d": data}
    # 创建新的图像和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制垂直小提琴图，不显示均值、极值和中位数，使用数据字典进行标记
    ax.violinplot("d", positions=range(4), showmeans=False, showextrema=False,
                  showmedians=False, data=data)


@image_comparison(['violinplot_vert_showmeans.png'])
# 定义一个测试函数，用于比较垂直小提琴图显示均值的图像
def test_vert_violinplot_showmeans():
    # 创建一个坐标轴对象
    ax = plt.axes()
    # 设置随机种子为 frac(sqrt(3)) 的前9位数字
    np.random.seed(732050807)
    # 生成四组包含各自100个随机数的正态分布数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制垂直小提琴图，显示均值，不显示极值和中位数
    ax.violinplot(data, positions=range(4), showmeans=True, showextrema=False,
                  showmedians=False)


@image_comparison(['violinplot_vert_showextrema.png'])
# 定义一个测试函数，用于比较垂直小提琴图显示极值的图像
def test_vert_violinplot_showextrema():
    # 创建一个坐标轴对象
    ax = plt.axes()
    # 设置随机种子为 frac(sqrt(5)) 的前9位数字
    np.random.seed(236067977)
    # 生成四组包含各自100个随机数的正态分布数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制垂直小提琴图，不显示均值和中位数，显示极值
    ax.violinplot(data, positions=range(4), showmeans=False, showextrema=True,
                  showmedians=False)


@image_comparison(['violinplot_vert_showmedians.png'])
# 定义一个测试函数，用于比较垂直小提琴图显示中位数的图像
def test_vert_violinplot_showmedians():
    # 创建一个坐标轴对象
    ax = plt.axes()
    # 设置随机种子为 frac(sqrt(7)) 的前9位数字
    np.random.seed(645751311)
    # 生成四组包含各自100个随机数的正态分布数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制垂直小提琴图，不显示均值和极值，显示中位数
    ax.violinplot(data, positions=range(4), showmeans=False, showextrema=False,
                  showmedians=True)


@image_comparison(['violinplot_vert_showall.png'])
# 定义一个测试函数，用于比较垂直小提琴图显示所有元素的图像
def test_vert_violinplot_showall():
    # 创建一个坐标轴对象
    ax = plt.axes()
    # 设置随机种子为 frac(sqrt(11)) 的前9位数字
    np.random.seed(316624790)
    # 生成四组包含各自100个随机数的正态分布数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制垂直小提琴图，显示均值、极值和中位数，设置分位数范围
    ax.violinplot(data, positions=range(4), showmeans=True, showextrema=True,
                  showmedians=True,
                  quantiles=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])


@image_comparison(['violinplot_vert_custompoints_10.png'])
# 定义一个测试函数，用于比较垂直小提琴图自定义点数（10个点）的图像
def test_vert_violinplot_custompoints_10():
    # 创建一个坐标轴对象
    ax = plt.axes()
    # 设置随机种子为 frac(sqrt(13)) 的前9位数字
    np.random.seed(605551275)
    # 生成四组包含各自100个随机数的正态分布数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制垂直小提琴图，不显示均值、极值和中位数，自定义绘制点数为10个
    ax.violinplot(data, positions=range(4), showmeans=False, showextrema=False,
                  showmedians=False, points=10)


@image_comparison(['violinplot_vert_custompoints_200.png'])
# 定义一个测试函数，用于比较垂直小提琴图自定义点数（200个点）的图像
def test_vert_violinplot_custompoints_200():
    # 创建一个坐标轴对象
    ax = plt.axes()
    # 设置随机种子为 frac(sqrt(17)) 的前9位数字
    np.random.seed(123105625)
    # 生成四组包含各自100个随机数的正态分布数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制垂直小提琴图，不显示均值、极值和中位数，自定义绘制点数为200个
    ax.violinplot(data, positions=range(4), showmeans=False, showextrema=False,
                  showmedians=False, points=200)


@image_comparison(['violinplot_horiz_baseline.png'])
# 定义一个测试函数，用于比较水平小提琴图的基线图像
def test_horiz_violinplot_baseline():
    # 由于这里未完整提供代码，无法添加进一步的注释
    # 创建一个新的绘图轴对象
    ax = plt.axes()
    # 设定随机数种子，确保生成的随机数是确定的
    # 这里使用的种子是 frac(sqrt(19)) 的前 9 位
    np.random.seed(358898943)
    # 生成四组包含各自 100 个随机正态分布数值的数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 使用小提琴图（violin plot）绘制数据
    # 将四组数据绘制在水平方向上，每组数据的位置为 0, 1, 2, 3
    # 不显示均值点、极值和中位数线
    ax.violinplot(data, positions=range(4), orientation='horizontal', showmeans=False,
                  showextrema=False, showmedians=False)
@image_comparison(['violinplot_horiz_showmedians.png'])
def test_horiz_violinplot_showmedians():
    # 创建一个图形坐标轴对象
    ax = plt.axes()
    # 设定随机数生成种子，确保结果可重复
    np.random.seed(795831523)
    # 生成包含100个正态分布随机数的列表，共4个列表
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制水平小提琴图，显示中位数，不显示均值和极值
    ax.violinplot(data, positions=range(4), orientation='horizontal', showmeans=False,
                  showextrema=False, showmedians=True)


@image_comparison(['violinplot_horiz_showmeans.png'])
def test_horiz_violinplot_showmeans():
    # 创建一个图形坐标轴对象
    ax = plt.axes()
    # 设定随机数生成种子，确保结果可重复
    np.random.seed(385164807)
    # 生成包含100个正态分布随机数的列表，共4个列表
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制水平小提琴图，显示均值，不显示极值和中位数
    ax.violinplot(data, positions=range(4), orientation='horizontal', showmeans=True,
                  showextrema=False, showmedians=False)


@image_comparison(['violinplot_horiz_showextrema.png'])
def test_horiz_violinplot_showextrema():
    # 创建一个图形坐标轴对象
    ax = plt.axes()
    # 设定随机数生成种子，确保结果可重复
    np.random.seed(567764362)
    # 生成包含100个正态分布随机数的列表，共4个列表
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制水平小提琴图，显示极值，不显示均值和中位数
    ax.violinplot(data, positions=range(4), orientation='horizontal', showmeans=False,
                  showextrema=True, showmedians=False)


@image_comparison(['violinplot_horiz_showall.png'])
def test_horiz_violinplot_showall():
    # 创建一个图形坐标轴对象
    ax = plt.axes()
    # 设定随机数生成种子，确保结果可重复
    np.random.seed(82762530)
    # 生成包含100个正态分布随机数的列表，共4个列表
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制水平小提琴图，显示均值、极值和中位数，并指定四分位数范围
    ax.violinplot(data, positions=range(4), orientation='horizontal', showmeans=True,
                  showextrema=True, showmedians=True,
                  quantiles=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])


@image_comparison(['violinplot_horiz_custompoints_10.png'])
def test_horiz_violinplot_custompoints_10():
    # 创建一个图形坐标轴对象
    ax = plt.axes()
    # 设定随机数生成种子，确保结果可重复
    np.random.seed(403124237)
    # 生成包含100个正态分布随机数的列表，共4个列表
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制水平小提琴图，不显示均值、极值和中位数，指定定制点数为10
    ax.violinplot(data, positions=range(4), orientation='horizontal', showmeans=False,
                  showextrema=False, showmedians=False, points=10)


@image_comparison(['violinplot_horiz_custompoints_200.png'])
def test_horiz_violinplot_custompoints_200():
    # 创建一个图形坐标轴对象
    ax = plt.axes()
    # 设定随机数生成种子，确保结果可重复
    np.random.seed(557438524)
    # 生成包含100个正态分布随机数的列表，共4个列表
    data = [np.random.normal(size=100) for _ in range(4)]
    # 绘制水平小提琴图，不显示均值、极值和中位数，指定定制点数为200
    ax.violinplot(data, positions=range(4), orientation='horizontal', showmeans=False,
                  showextrema=False, showmedians=False, points=200)


@image_comparison(['violinplot_sides.png'], remove_text=True, style='mpl20')
def test_violinplot_sides():
    # 创建一个图形坐标轴对象
    ax = plt.axes()
    # 设定随机数生成种子，确保结果可重复
    np.random.seed(19680801)
    # 生成包含100个正态分布随机数的列表
    data = [np.random.normal(size=100)]
    # 检查水平小提琴图
    for pos, side in zip([0, -0.5, 0.5], ['both', 'low', 'high']):
        # 绘制水平小提琴图，显示极值和中位数，根据指定的侧面参数
        ax.violinplot(data, positions=[pos], orientation='horizontal', showmeans=False,
                      showextrema=True, showmedians=True, side=side)
    # 检查垂直小提琴图
    # 使用 zip 函数将两个列表进行并行迭代，生成位置和侧面类型的元组
    for pos, side in zip([4, 3.5, 4.5], ['both', 'low', 'high']):
        # 在图形 ax 上绘制小提琴图，数据为 data
        ax.violinplot(data,
                      # 设置小提琴图的位置
                      positions=[pos],
                      # 设置小提琴图的方向为垂直
                      orientation='vertical',
                      # 不显示均值点
                      showmeans=False,
                      # 显示极值
                      showextrema=True,
                      # 显示中位数线
                      showmedians=True,
                      # 设置小提琴图的位置（左侧、右侧或两侧）
                      side=side)
def test_violinplot_bad_positions():
    ax = plt.axes()
    # 设置随机种子确保可复现性，使用特定的种子值
    np.random.seed(855654600)
    # 生成四组包含各自100个正态分布随机数的数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 使用 pytest 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 violinplot 方法，指定数据和位置参数为一个范围对象
        ax.violinplot(data, positions=range(5))


def test_violinplot_bad_widths():
    ax = plt.axes()
    # 设置随机种子确保可复现性，使用特定的种子值
    np.random.seed(280109889)
    # 生成四组包含各自100个正态分布随机数的数据
    data = [np.random.normal(size=100) for _ in range(4)]
    # 使用 pytest 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 violinplot 方法，指定数据和宽度参数为一个列表
        ax.violinplot(data, positions=range(4), widths=[1, 2, 3])


def test_violinplot_bad_quantiles():
    ax = plt.axes()
    # 设置随机种子确保可复现性，使用特定的种子值
    np.random.seed(544003745)
    # 生成一个包含100个正态分布随机数的数据列表
    data = [np.random.normal(size=100)]
    # 使用 pytest 检查是否会抛出 ValueError 异常
    # 由于指定了不同长度的分位数列表，预期会抛出异常
    with pytest.raises(ValueError):
        # 调用 violinplot 方法，指定数据和分位数参数为一个嵌套列表
        ax.violinplot(data, quantiles=[[0.1, 0.2], [0.5, 0.7]])


def test_violinplot_outofrange_quantiles():
    ax = plt.axes()
    # 设置随机种子确保可复现性，使用特定的种子值
    np.random.seed(888194417)
    # 生成一个包含100个正态分布随机数的数据列表
    data = [np.random.normal(size=100)]
    # 使用 pytest 检查是否会抛出 ValueError 异常
    # 由于分位数超出了允许范围，预期会抛出异常
    with pytest.raises(ValueError):
        # 调用 violinplot 方法，指定数据和分位数参数为一个嵌套列表
        ax.violinplot(data, quantiles=[[0.1, 0.2, 0.3, 1.05]])

    # 同样的测试，但这次分位数低于允许范围
    with pytest.raises(ValueError):
        # 调用 violinplot 方法，指定数据和分位数参数为一个嵌套列表
        ax.violinplot(data, quantiles=[[-0.05, 0.2, 0.3, 0.75]])


@check_figures_equal(extensions=["png"])
def test_violinplot_single_list_quantiles(fig_test, fig_ref):
    # 确保可以将1D情况下的分位数列表作为单一列表传入
    # 设置随机种子确保可复现性，使用特定的种子值
    np.random.seed(110433579)
    # 生成一个包含100个正态分布随机数的数据列表
    data = [np.random.normal(size=100)]

    # 测试图像
    ax = fig_test.subplots()
    # 调用 violinplot 方法，指定数据和分位数参数为一个列表
    ax.violinplot(data, quantiles=[0.1, 0.3, 0.9])

    # 参考图像
    ax = fig_ref.subplots()
    # 调用 violinplot 方法，指定数据和分位数参数为一个嵌套列表
    ax.violinplot(data, quantiles=[[0.1, 0.3, 0.9]])


@check_figures_equal(extensions=["png"])
def test_violinplot_pandas_series(fig_test, fig_ref, pd):
    # 设置随机种子确保可复现性，使用特定的种子值
    np.random.seed(110433579)
    # 创建三个不同的 Pandas Series 对象，分别包含不同长度的随机数
    s1 = pd.Series(np.random.normal(size=7), index=[9, 8, 7, 6, 5, 4, 3])
    s2 = pd.Series(np.random.normal(size=9), index=list('ABCDEFGHI'))
    s3 = pd.Series(np.random.normal(size=11))
    # 测试图像
    fig_test.subplots().violinplot([s1, s2, s3])
    # 参考图像
    fig_ref.subplots().violinplot([s1.values, s2.values, s3.values])


def test_manage_xticks():
    # 创建图像和轴对象
    _, ax = plt.subplots()
    # 设置 X 轴的范围
    ax.set_xlim(0, 4)
    # 获取设置前的 X 轴范围
    old_xlim = ax.get_xlim()
    # 设置随机种子确保可复现性，使用特定的种子值
    np.random.seed(0)
    # 生成两组包含各自随机数的数据
    y1 = np.random.normal(10, 3, 20)
    y2 = np.random.normal(3, 1, 20)
    # 调用 boxplot 方法，指定数据和位置参数，同时禁止管理刻度
    ax.boxplot([y1, y2], positions=[1, 2], manage_ticks=False)
    # 获取设置后的 X 轴范围
    new_xlim = ax.get_xlim()
    # 断言设置前后 X 轴范围一致
    assert_array_equal(old_xlim, new_xlim)


def test_boxplot_not_single():
    # 创建图像和轴对象
    fig, ax = plt.subplots()
    # 绘制第一个箱线图，指定数据和位置参数
    ax.boxplot(np.random.rand(100), positions=[3])
    # 绘制第二个箱线图，指定数据和位置参数
    ax.boxplot(np.random.rand(100), positions=[5])
    # 绘图更新画布
    fig.canvas.draw()
    # 断言 X 轴范围为预期值
    assert ax.get_xlim() == (2.5, 5.5)
    # 断言 X 轴刻度值为预期列表
    assert list(ax.get_xticks()) == [3, 5]
    # 断言 X 轴刻度标签文本为预期列表
    assert [t.get_text() for t in ax.get_xticklabels()] == ["3", "5"]


def test_tick_space_size_0():
    # 允许字体大小为零时的刻度设置，当存在时
    # （这个函数尚未完整）
    # 导入 matplotlib.pyplot 库并绘制一条从点 (0,0) 到点 (1,1) 的直线
    plt.plot([0, 1], [0, 1])
    # 更新 matplotlib 的全局参数，将字体大小设置为 0，可能意味着不显示文本
    matplotlib.rcParams.update({'font.size': 0})
    # 创建一个 BytesIO 对象 b，用于保存图像数据
    b = io.BytesIO()
    # 将当前的图形保存到 BytesIO 对象 b 中，设置分辨率为 80 dpi，保存格式为原始格式（raw）
    plt.savefig(b, dpi=80, format='raw')
# 使用 @image_comparison 装饰器，比较生成的图像是否与给定的图像文件匹配
@image_comparison(['errorbar_basic', 'errorbar_mixed', 'errorbar_basic'])
# 定义一个名为 test_errorbar 的测试函数
def test_errorbar():
    # 使用 np.longdouble 类型创建一个包含特定数值范围的数组 x，用于避免特定计算芯片的浮点舍入问题
    x = np.arange(0.1, 4, 0.5, dtype=np.longdouble)
    # 计算数组 y，表示指数衰减函数在给定 x 值处的结果
    y = np.exp(-x)

    # 计算误差条的 y 方向误差，包含基础误差和与 x 相关的额外误差
    yerr = 0.1 + 0.2*np.sqrt(x)
    # 计算误差条的 x 方向误差，基于 y 方向误差计算得出
    xerr = 0.1 + yerr

    # 创建一个新的图形对象
    fig = plt.figure()
    # 获取当前图形的当前坐标轴（axes）
    ax = fig.gca()
    # 在坐标轴上绘制误差条，指定 x 方向和 y 方向的误差大小
    ax.errorbar(x, y, xerr=0.2, yerr=0.4)
    # 设置坐标轴的标题
    ax.set_title("Simplest errorbars, 0.2 in x, 0.4 in y")

    # 创建一个新的图形对象和一组子图，并共享 x 轴
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    # 获取第一个子图的坐标轴
    ax = axs[0, 0]
    # 在子图上绘制误差条，指定 y 方向的对称误差和数据点的格式
    ax.errorbar(x, y, yerr=yerr, fmt='o')
    # 设置子图的标题
    ax.set_title('Vert. symmetric')

    # 对于包含 4 个子图的情况，减少轴上刻度的数量，以避免拥挤
    ax.locator_params(nbins=4)

    # 获取第二个子图的坐标轴
    ax = axs[0, 1]
    # 在子图上绘制误差条，指定 x 方向的对称误差、数据点的格式和透明度
    ax.errorbar(x, y, xerr=xerr, fmt='o', alpha=0.4)
    # 设置子图的标题
    ax.set_title('Hor. symmetric w/ alpha')

    # 获取第三个子图的坐标轴
    ax = axs[1, 0]
    # 在子图上绘制误差条，指定 x 和 y 方向的不对称误差、数据点的格式
    ax.errorbar(x, y, yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
    # 设置子图的标题
    ax.set_title('H, V asymmetric')

    # 获取第四个子图的坐标轴
    ax = axs[1, 1]
    # 将 y 轴的比例设置为对数尺度
    ax.set_yscale('log')
    # 确保所有 y 值为正值：
    ylower = np.maximum(1e-2, y - yerr)
    yerr_lower = y - ylower

    # 在子图上绘制误差条，指定 x 方向的对称误差、y 方向的下限误差、数据点的格式和误差线的颜色
    ax.errorbar(x, y, yerr=[yerr_lower, 2*yerr], xerr=xerr,
                fmt='o', ecolor='g', capthick=2)
    # 设置子图的标题
    ax.set_title('Mixed sym., log y')
    # 由于浮点误差可能扩展范围，强制设置 y 轴的限制
    ax.set_ylim(1e-2, 1e1)

    # 设置整个图形的标题
    fig.suptitle('Variable errorbars')

    # 使用先前定义的数据字典创建一个新的图形对象
    data = {"x": x, "y": y}
    fig = plt.figure()
    # 获取当前图形的当前坐标轴（axes）
    ax = fig.gca()
    # 在坐标轴上绘制误差条，使用数据字典中的键作为 x 和 y 数据，指定 x 和 y 方向的误差大小
    ax.errorbar("x", "y", xerr=0.2, yerr=0.4, data=data)
    # 设置坐标轴的标题
    ax.set_title("Simplest errorbars, 0.2 in x, 0.4 in y")


# 使用 @image_comparison 装饰器，比较生成的图像是否与给定的图像文件匹配，并且移除图像中的文本
@image_comparison(['mixed_errorbar_polar_caps'], extensions=['png'],
                  remove_text=True)
# 定义一个名为 test_mixed_errorbar_polar_caps 的测试函数
def test_mixed_errorbar_polar_caps():
    """
    在单个测试图中混合几种极坐标误差条使用情况。

    建议将单个点定位在网格外。如果这个测试的可重复性存在问题，考虑移除网格。
    """
    # 创建一个新的图形对象
    fig = plt.figure()
    # 获取极坐标投影的子图坐标轴
    ax = plt.subplot(111, projection='polar')

    # 绘制对称误差条
    th_sym = [1, 2, 3]
    r_sym = [0.9]*3
    ax.errorbar(th_sym, r_sym, xerr=0.35, yerr=0.2, fmt="o")

    # 绘制长误差条
    th_long = [np.pi/2 + .1, np.pi + .1]
    r_long = [1.8, 2.2]
    ax.errorbar(th_long, r_long, xerr=0.8 * np.pi, yerr=0.15, fmt="o")

    # 绘制不对称误差条
    th_asym = [4*np.pi/3 + .1, 5*np.pi/3 + .1, 2*np.pi-0.1]
    r_asym = [1.1]*3
    xerr = [[.3, .3, .2], [.2, .3, .3]]
    yerr = [[.35, .5, .5], [.5, .35, .5]]
    ax.errorbar(th_asym, r_asym, xerr=xerr, yerr=yerr, fmt="o")

    # 绘制重叠误差条
    th_over = [2.1]
    r_over = [3.1]
    ax.errorbar(th_over, r_over, xerr=10, yerr=.2, fmt="o")
    # 创建一个新的图形窗口和一个坐标轴
    f, ax = plt.subplots()
    # 创建一个包含10个元素的数组
    x = np.arange(10)
    # 创建一个数组，每个元素是x数组对应元素的两倍
    y = 2*x
    
    # 在坐标轴上绘制第一个误差条，颜色参数为None，使其使用默认颜色
    e1, _, _ = ax.errorbar(x, y, c=None)
    # 在坐标轴上绘制第二个误差条，颜色参数为None，使其使用默认颜色
    e2, _, _ = ax.errorbar(x, 2*y, c=None)
    # 在坐标轴上绘制一条曲线，用x作为横坐标，4*y作为纵坐标
    ln1, = ax.plot(x, 4*y)
    
    # 断言第一个误差条的颜色与预设颜色'C0'的RGBA值相同
    assert mcolors.to_rgba(e1.get_color()) == mcolors.to_rgba('C0')
    # 断言第二个误差条的颜色与预设颜色'C1'的RGBA值相同
    assert mcolors.to_rgba(e2.get_color()) == mcolors.to_rgba('C1')
    # 断言曲线的颜色与预设颜色'C2'的RGBA值相同
    assert mcolors.to_rgba(ln1.get_color()) == mcolors.to_rgba('C2')
@check_figures_equal()
# 装饰器，用于测试函数，确保两个图形对象相等
def test_errorbar_cycle_ecolor(fig_test, fig_ref):
    # 创建数据
    x = np.arange(0.1, 4, 0.5)
    y = [np.exp(-x+n) for n in range(4)]

    # 创建测试图和参考图的子图对象
    axt = fig_test.subplots()
    axr = fig_ref.subplots()

    # 循环遍历y和颜色列表，为每个子图对象绘制errorbar图形
    for yi, color in zip(y, ['C0', 'C1', 'C2', 'C3']):
        axt.errorbar(x, yi, yerr=(yi * 0.25), linestyle='-',
                     marker='o', ecolor='black')
        axr.errorbar(x, yi, yerr=(yi * 0.25), linestyle='-',
                     marker='o', color=color, ecolor='black')


def test_errorbar_shape():
    # 创建图形和坐标轴对象
    fig = plt.figure()
    ax = fig.gca()

    # 创建数据
    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)
    yerr1 = 0.1 + 0.2*np.sqrt(x)
    yerr = np.vstack((yerr1, 2*yerr1)).T
    xerr = 0.1 + yerr

    # 使用pytest的断言检查错误是否被正确引发
    with pytest.raises(ValueError):
        ax.errorbar(x, y, yerr=yerr, fmt='o')
    with pytest.raises(ValueError):
        ax.errorbar(x, y, xerr=xerr, fmt='o')
    with pytest.raises(ValueError):
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o')


@image_comparison(['errorbar_limits'])
# 装饰器，用于比较生成的图像与参考图像，确认它们相同
def test_errorbar_limits():
    # 创建数据
    x = np.arange(0.5, 5.5, 0.5)
    y = np.exp(-x)
    xerr = 0.1
    yerr = 0.2
    ls = 'dotted'

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    # 绘制标准误差条
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, ls=ls, color='blue')

    # 包含上限的误差条
    uplims = np.zeros_like(x)
    uplims[[1, 5, 9]] = True
    ax.errorbar(x, y+0.5, xerr=xerr, yerr=yerr, uplims=uplims, ls=ls,
                color='green')

    # 包含下限的误差条
    lolims = np.zeros_like(x)
    lolims[[2, 4, 8]] = True
    ax.errorbar(x, y+1.0, xerr=xerr, yerr=yerr, lolims=lolims, ls=ls,
                color='red')

    # 包含上限和下限的误差条
    ax.errorbar(x, y+1.5, marker='o', ms=8, xerr=xerr, yerr=yerr,
                lolims=lolims, uplims=uplims, ls=ls, color='magenta')

    # 包含x轴上限和下限的误差条
    xerr = 0.2
    yerr = np.full_like(x, 0.2)
    yerr[[3, 6]] = 0.3
    xlolims = lolims
    xuplims = uplims
    lolims = np.zeros_like(x)
    uplims = np.zeros_like(x)
    lolims[[6]] = True
    uplims[[3]] = True
    ax.errorbar(x, y+2.1, marker='o', ms=8, xerr=xerr, yerr=yerr,
                xlolims=xlolims, xuplims=xuplims, uplims=uplims,
                lolims=lolims, ls='none', mec='blue', capsize=0,
                color='cyan')
    ax.set_xlim((0, 5.5))
    ax.set_title('Errorbar upper and lower limits')


def test_errorbar_nonefmt():
    # 检查使用'none'作为格式时仍然绘制误差条的情况
    x = np.arange(5)
    y = np.arange(5)

    # 调用errorbar函数，并断言返回的plotline为None，barlines的颜色为C0
    plotline, _, barlines = plt.errorbar(x, y, xerr=1, yerr=1, fmt='none')
    assert plotline is None
    for errbar in barlines:
        assert np.all(errbar.get_color() == mcolors.to_rgba('C0'))


def test_errorbar_line_specific_kwargs():
    # 检查传递特定于线条的关键字参数不会引发错误
    x = np.arange(5)
    y = np.arange(5)
    # 调用 plt.errorbar() 方法创建误差条形图，并获取返回的 plotline 对象以及两个未使用的占位符
    plotline, _, _ = plt.errorbar(x, y, xerr=1, yerr=1, ls='None',
                                  marker='s', fillstyle='full',
                                  drawstyle='steps-mid',
                                  dash_capstyle='round',
                                  dash_joinstyle='miter',
                                  solid_capstyle='butt',
                                  solid_joinstyle='bevel')
    # 使用断言确保误差条形图的填充样式为 'full'
    assert plotline.get_fillstyle() == 'full'
    # 使用断言确保误差条形图的绘制样式为 'steps-mid'
    assert plotline.get_drawstyle() == 'steps-mid'
# 在函数上应用装饰器，检查测试中两个图形是否相等，使用扩展名为 'png'
@check_figures_equal(extensions=['png'])
# 定义测试函数，测试 errorbar 方法使用属性循环的情况
def test_errorbar_with_prop_cycle(fig_test, fig_ref):
    # 在参考图中创建子图对象 ax
    ax = fig_ref.subplots()
    # 绘制误差条图，指定 x 和 y 值，误差为 0.5，线型为虚线，标记为正方形，填充颜色为黑色
    ax.errorbar(x=[2, 4, 10], y=[0, 1, 2], yerr=0.5,
                ls='--', marker='s', mfc='k')
    # 绘制第二个误差条图，指定 x 和 y 值，误差为 0.5，线型为点线，标记为正方形，填充颜色为黄色
    ax.errorbar(x=[2, 4, 10], y=[2, 3, 4], yerr=0.5, color='tab:green',
                ls=':', marker='s', mfc='y')
    # 绘制第三个误差条图，指定 x 和 y 值，误差为 0.5，格式为蓝色，线型为点划线，标记为圆形，填充颜色为青色
    ax.errorbar(x=[2, 4, 10], y=[4, 5, 6], yerr=0.5, fmt='tab:blue',
                ls='-.', marker='o', mfc='c')
    # 设置 x 轴的显示范围为 1 到 11
    ax.set_xlim(1, 11)

    # 创建属性循环对象 _cycle，指定线型、标记、填充颜色、颜色的循环方式
    _cycle = cycler(ls=['--', ':', '-.'], marker=['s', 's', 'o'],
                    mfc=['k', 'y', 'c'], color=['b', 'g', 'r'])
    # 将属性循环对象 _cycle 应用于当前图形的坐标轴属性
    plt.rc("axes", prop_cycle=_cycle)
    # 在测试图中创建子图对象 ax
    ax = fig_test.subplots()
    # 绘制误差条图，指定 x 和 y 值，误差为 0.5，根据属性循环自动选择线型、标记和填充颜色
    ax.errorbar(x=[2, 4, 10], y=[0, 1, 2], yerr=0.5)
    # 绘制第二个误差条图，指定 x 和 y 值，误差为 0.5，颜色为绿色，根据属性循环自动选择线型和标记
    ax.errorbar(x=[2, 4, 10], y=[2, 3, 4], yerr=0.5, color='tab:green')
    # 绘制第三个误差条图，指定 x 和 y 值，误差为 0.5，格式为蓝色，根据属性循环自动选择线型和标记
    ax.errorbar(x=[2, 4, 10], y=[4, 5, 6], yerr=0.5, fmt='tab:blue')
    # 设置 x 轴的显示范围为 1 到 11
    ax.set_xlim(1, 11)


# 定义测试函数，测试 errorbar 方法在错误情况下的处理
def test_errorbar_every_invalid():
    # 生成一个包含 15 个元素的等间距数组 x
    x = np.linspace(0, 1, 15)
    # 生成与 x 对应的 y 值，代表某种函数关系
    y = x * (1-x)
    # 根据 y 计算 yerr，代表误差
    yerr = y / 6

    # 在一个新的图形中创建子图对象 ax
    ax = plt.figure().subplots()

    # 使用 pytest 的断言检查错误的情况
    with pytest.raises(ValueError, match='not a tuple of two integers'):
        # 测试错误情况：errorevery 参数不是两个整数的元组
        ax.errorbar(x, y, yerr, errorevery=(1, 2, 3))
    with pytest.raises(ValueError, match='not a tuple of two integers'):
        # 测试错误情况：errorevery 参数不是两个整数的元组
        ax.errorbar(x, y, yerr, errorevery=(1.3, 3))
    with pytest.raises(ValueError, match='not a valid NumPy fancy index'):
        # 测试错误情况：errorevery 参数不是有效的 NumPy 索引
        ax.errorbar(x, y, yerr, errorevery=[False, True])
    with pytest.raises(ValueError, match='not a recognized value'):
        # 测试错误情况：errorevery 参数值无法识别
        ax.errorbar(x, y, yerr, errorevery='foobar')


# 定义测试函数，测试 errorbar 方法中 xerr 和 yerr 参数不包含负值的情况
def test_xerr_yerr_not_negative():
    # 在一个新的图形中创建子图对象 ax
    ax = plt.figure().subplots()

    # 使用 pytest 的断言检查错误的情况
    with pytest.raises(ValueError,
                       match="'xerr' must not contain negative values"):
        # 测试错误情况：xerr 参数包含负值
        ax.errorbar(x=[0], y=[0], xerr=[[-0.5], [1]], yerr=[[-0.5], [1]])
    with pytest.raises(ValueError,
                       match="'xerr' must not contain negative values"):
        # 测试错误情况：xerr 参数包含负值
        ax.errorbar(x=[0], y=[0], xerr=[[-0.5], [1]])
    with pytest.raises(ValueError,
                       match="'yerr' must not contain negative values"):
        # 测试错误情况：yerr 参数包含负值
        ax.errorbar(x=[0], y=[0], yerr=[[-0.5], [1]])
    with pytest.raises(ValueError,
                       match="'yerr' must not contain negative values"):
        # 测试错误情况：yerr 参数包含负值
        x = np.arange(5)
        y = [datetime.datetime(2021, 9, i * 2 + 1) for i in x]
        ax.errorbar(x=x,
                    y=y,
                    yerr=datetime.timedelta(days=-10))


# 定义测试函数，测试 errorbar 方法中 xerr 和 yerr 参数不包含 None 的情况
def test_xerr_yerr_not_none():
    # 在一个新的图形中创建子图对象 ax
    ax = plt.figure().subplots()

    # 使用 pytest 的断言检查错误的情况
    with pytest.raises(ValueError,
                       match="'xerr' must not contain None"):
        # 测试错误情况：xerr 参数包含 None
        ax.errorbar(x=[0], y=[0], xerr=[[None], [1]], yerr=[[None], [1]])
    with pytest.raises(ValueError,
                       match="'xerr' must not contain None"):
        # 测试错误情况：xerr 参数包含 None
        ax.errorbar(x=[0], y=[0], xerr=[[None], [1]])
    with pytest.raises(ValueError,
                       match="'yerr' must not contain None"):
        # 测试错误情况：yerr 参数包含 None
        ax.errorbar(x=[0], y=[0], yerr=[[None], [1]])
def test_errorbar_every(fig_test, fig_ref):
    # 生成一个等间距的数组作为 x 值
    x = np.linspace(0, 1, 15)
    # 根据 x 计算 y 值，这里使用了一个简单的函数关系
    y = x * (1-x)
    # 计算误差条的值
    yerr = y / 6

    # 在 fig_ref 图形上创建子图 ax_ref
    ax_ref = fig_ref.subplots()
    # 在 fig_test 图形上创建子图 ax_test
    ax_test = fig_test.subplots()

    # 使用 zip 函数迭代颜色和偏移值
    for color, shift in zip('rgbk', [0, 0, 2, 7]):
        # 调整 y 值以便可视化
        y += .02

        # 使用指定的偏移和步长检查 errorevery 参数
        ax_test.errorbar(x, y, yerr, errorevery=(shift, 4),
                         capsize=4, c=color)

        # 手动绘制误差条
        # 注意：errorbar 默认在 z=2.1 位置绘制主要图形
        ax_ref.plot(x, y, c=color, zorder=2.1)
        ax_ref.errorbar(x[shift::4], y[shift::4], yerr[shift::4],
                        capsize=4, c=color, fmt='none')

    # 检查 markevery 参数是否对线条生效，而不影响误差条
    ax_test.errorbar(x, y + 0.1, yerr, markevery=(1, 4), capsize=4, fmt='o')
    ax_ref.plot(x[1::4], y[1::4] + 0.1, 'o', zorder=2.1)
    ax_ref.errorbar(x, y + 0.1, yerr, capsize=4, fmt='none')

    # 检查将切片传递给 markevery/errorevery 是否起作用
    ax_test.errorbar(x, y + 0.2, yerr, errorevery=slice(2, None, 3),
                     markevery=slice(2, None, 3),
                     capsize=4, c='C0', fmt='o')
    ax_ref.plot(x[2::3], y[2::3] + 0.2, 'o', c='C0', zorder=2.1)
    ax_ref.errorbar(x[2::3], y[2::3] + 0.2, yerr[2::3],
                    capsize=4, c='C0', fmt='none')

    # 检查将可迭代对象传递给 markevery/errorevery 是否起作用
    ax_test.errorbar(x, y + 0.2, yerr, errorevery=[False, True, False] * 5,
                     markevery=[False, True, False] * 5,
                     capsize=4, c='C1', fmt='o')
    ax_ref.plot(x[1::3], y[1::3] + 0.2, 'o', c='C1', zorder=2.1)
    ax_ref.errorbar(x[1::3], y[1::3] + 0.2, yerr[1::3],
                    capsize=4, c='C1', fmt='none')


@pytest.mark.parametrize('elinewidth', [[1, 2, 3],
                                        np.array([1, 2, 3]),
                                        1])
def test_errorbar_linewidth_type(elinewidth):
    # 测试不同类型的 elinewidth 参数
    plt.errorbar([1, 2, 3], [1, 2, 3], yerr=[1, 2, 3], elinewidth=elinewidth)


@check_figures_equal(extensions=["png"])
def test_errorbar_nan(fig_test, fig_ref):
    # 测试处理 NaN 值的 errorbar 行为
    ax = fig_test.add_subplot()
    xs = range(5)
    ys = np.array([1, 2, np.nan, np.nan, 3])
    es = np.array([4, 5, np.nan, np.nan, 6])
    ax.errorbar(xs, ys, es)
    ax = fig_ref.add_subplot()
    ax.errorbar([0, 1], [1, 2], [4, 5])
    ax.errorbar([4], [3], [6], fmt="C0")


@image_comparison(['hist_stacked_stepfilled', 'hist_stacked_stepfilled'])
def test_hist_stacked_stepfilled():
    # 创建一些数据
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    fig, ax = plt.subplots()
    ax.hist((d1, d2), histtype="stepfilled", stacked=True)

    # 重用上述用于带标签数据的测试用例
    data = {"x": (d1, d2)}
    fig, ax = plt.subplots()
    ax.hist("x", histtype="stepfilled", stacked=True, data=data)


@image_comparison(['hist_offset'])
def test_hist_offset():
    # 创建一些数据
    # 使用 NumPy 生成一个包含从 0 到 10 的均匀间隔的 50 个数的数组
    d1 = np.linspace(0, 10, 50)
    # 使用 NumPy 生成一个包含从 1 到 3 的均匀间隔的 20 个数的数组
    d2 = np.linspace(1, 3, 20)
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在当前的坐标轴上绘制直方图，数据来自于数组 d1，并将直方图底部设为 5
    ax.hist(d1, bottom=5)
    # 在当前的坐标轴上绘制直方图，数据来自于数组 d2，并将直方图底部设为 15
    ax.hist(d2, bottom=15)
@image_comparison(['hist_step.png'], remove_text=True)
# 定义一个测试函数，用于生成包含直方图的图像，并进行图像比较
def test_hist_step():
    # 生成一些数据
    d1 = np.linspace(1, 3, 20)
    # 创建一个图形和一个坐标轴
    fig, ax = plt.subplots()
    # 绘制直方图，指定直方图类型为“step”
    ax.hist(d1, histtype="step")
    # 设置 y 轴的范围
    ax.set_ylim(0, 10)
    # 设置 x 轴的范围
    ax.set_xlim(-1, 5)


@image_comparison(['hist_step_horiz.png'])
# 定义一个测试函数，用于生成包含水平方向直方图的图像，并进行图像比较
def test_hist_step_horiz():
    # 生成一些数据
    d1 = np.linspace(0, 10, 50)
    d2 = np.linspace(1, 3, 20)
    # 创建一个图形和一个坐标轴
    fig, ax = plt.subplots()
    # 绘制直方图，指定直方图类型为“step”，方向为水平
    ax.hist((d1, d2), histtype="step", orientation="horizontal")


@image_comparison(['hist_stacked_weights'])
# 定义一个测试函数，用于生成包含加权堆叠直方图的图像，并进行图像比较
def test_hist_stacked_weighted():
    # 生成一些数据
    d1 = np.linspace(0, 10, 50)
    d2 = np.linspace(1, 3, 20)
    w1 = np.linspace(0.01, 3.5, 50)
    w2 = np.linspace(0.05, 2., 20)
    # 创建一个图形和一个坐标轴
    fig, ax = plt.subplots()
    # 绘制直方图，指定直方图类型为“stepfilled”，堆叠方式为 True，同时使用权重
    ax.hist((d1, d2), weights=(w1, w2), histtype="stepfilled", stacked=True)


@image_comparison(['stem.png'], style='mpl20', remove_text=True)
# 定义一个测试函数，用于生成包含诸如茎叶图的图像，并进行图像比较
def test_stem():
    x = np.linspace(0.1, 2 * np.pi, 100)

    fig, ax = plt.subplots()
    # Label is a single space to force a legend to be drawn, but to avoid any
    # text being drawn
    # 绘制茎叶图，指定线条格式、标记格式和基线格式，同时添加一个空的标签以绘制图例
    ax.stem(x, np.cos(x),
            linefmt='C2-.', markerfmt='k+', basefmt='C1-.', label=' ')
    # 添加图例
    ax.legend()


# 定义一个函数，用于测试茎叶图函数的参数
def test_stem_args():
    """Test that stem() correctly identifies x and y values."""
    def _assert_equal(stem_container, expected):
        # 提取茎叶容器中的 x 和 y 值，并进行断言验证
        x, y = map(list, stem_container.markerline.get_data())
        assert x == expected[0]
        assert y == expected[1]

    fig, ax = plt.subplots()

    x = [1, 3, 5]
    y = [9, 8, 7]

    # 测试不同的调用签名
    _assert_equal(ax.stem(y), expected=([0, 1, 2], y))
    _assert_equal(ax.stem(x, y), expected=(x, y))
    _assert_equal(ax.stem(x, y, linefmt='r--'), expected=(x, y))
    _assert_equal(ax.stem(x, y, 'r--'), expected=(x, y))
    _assert_equal(ax.stem(x, y, linefmt='r--', basefmt='b--'), expected=(x, y))
    _assert_equal(ax.stem(y, linefmt='r--'), expected=([0, 1, 2], y))
    _assert_equal(ax.stem(y, 'r--'), expected=([0, 1, 2], y))


# 定义一个函数，用于测试茎叶图函数的 markerfmt 参数
def test_stem_markerfmt():
    """Test that stem(..., markerfmt=...) produces the intended markers."""
    def _assert_equal(stem_container, linecolor=None, markercolor=None,
                      marker=None):
        """
        Check that the given StemContainer has the properties listed as
        keyword-arguments.
        """
        if linecolor is not None:
            # 验证线条颜色是否符合预期
            assert mcolors.same_color(
                stem_container.stemlines.get_color(),
                linecolor)
        if markercolor is not None:
            # 验证标记颜色是否符合预期
            assert mcolors.same_color(
                stem_container.markerline.get_color(),
                markercolor)
        if marker is not None:
            # 验证标记类型是否符合预期
            assert stem_container.markerline.get_marker() == marker
        # 验证标记线条样式是否为无线条
        assert stem_container.markerline.get_linestyle() == 'None'

    fig, ax = plt.subplots()

    x = [1, 3, 5]
    y = [9, 8, 7]

    # 测试不同的 markerfmt 参数设置
    # 无线条设置，标记颜色为 'C0'，标记类型为 'o'
    _assert_equal(ax.stem(x, y), markercolor='C0', marker='o')
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 markerfmt='x' 参数
    _assert_equal(ax.stem(x, y, markerfmt='x'), markercolor='C0', marker='x')
    
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 markerfmt='rx' 参数
    _assert_equal(ax.stem(x, y, markerfmt='rx'), markercolor='r', marker='x')

    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='r' 参数，
    # marker color 遵循 linefmt 如果未指定
    _assert_equal(
        ax.stem(x, y, 'r'),
        linecolor='r', markercolor='r', marker='o')
    
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='rx' 参数，
    # marker 目前不受 linefmt 影响
    _assert_equal(
        ax.stem(x, y, 'rx'),
        linecolor='r', markercolor='r', marker='o')
    
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='r' 和 markerfmt='x' 参数，
    # 只指定了 marker 类型
    _assert_equal(
        ax.stem(x, y, 'r', markerfmt='x'),
        linecolor='r', markercolor='r', marker='x')
    
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='r' 和 markerfmt='g' 参数，
    # 只指定了 marker 颜色
    _assert_equal(
        ax.stem(x, y, 'r', markerfmt='g'),
        linecolor='r', markercolor='g', marker='o')
    
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='r' 和 markerfmt='gx' 参数，
    # 同时指定了 marker 类型和颜色
    _assert_equal(
        ax.stem(x, y, 'r', markerfmt='gx'),
        linecolor='r', markercolor='g', marker='x')
    
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='r' 和 markerfmt=' ' 参数，
    # markerfmt=' ' 表示无 marker
    _assert_equal(
        ax.stem(x, y, 'r', markerfmt=' '),
        linecolor='r', markercolor='r', marker='None')
    
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='r' 和 markerfmt='' 参数，
    # markerfmt='' 表示无 marker
    _assert_equal(
        ax.stem(x, y, 'r', markerfmt=''),
        linecolor='r', markercolor='r', marker='None')

    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='r' 参数
    _assert_equal(
        ax.stem(x, y, linefmt='r'),
        linecolor='r', markercolor='r', marker='o')
    
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='r' 和 markerfmt='x' 参数
    _assert_equal(
        ax.stem(x, y, linefmt='r', markerfmt='x'),
        linecolor='r', markercolor='r', marker='x')
    
    # 使用 _assert_equal 函数验证 ax.stem 的返回值与预期结果是否相等，带有 linefmt='r' 和 markerfmt='gx' 参数
    _assert_equal(
        ax.stem(x, y, linefmt='r', markerfmt='gx'),
        linecolor='r', markercolor='g', marker='x')
def test_stem_dates():
    # 创建一个包含一个子图的图形对象和坐标轴对象
    fig, ax = plt.subplots(1, 1)
    # 解析日期字符串并生成日期对象作为 x 轴数据
    xs = [dateutil.parser.parse("2013-9-28 11:00:00"),
          dateutil.parser.parse("2013-9-28 12:00:00")]
    # 设置 y 轴数据
    ys = [100, 200]
    # 在指定的坐标轴上绘制某些点的垂直线条图
    ax.stem(xs, ys)


@image_comparison(['stem_orientation.png'], style='mpl20', remove_text=True)
def test_stem_orientation():
    # 生成一组均匀分布的 x 轴数据
    x = np.linspace(0.1, 2*np.pi, 50)

    # 创建一个图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 在指定的坐标轴上绘制余弦函数的水平线条图，设定线条、标记和基线的格式
    ax.stem(x, np.cos(x),
            linefmt='C2-.', markerfmt='kx', basefmt='C1-.',
            orientation='horizontal')


@image_comparison(['hist_stacked_stepfilled_alpha'])
def test_hist_stacked_stepfilled_alpha():
    # 创建一些数据
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    # 创建一个图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制堆叠的步骤填充直方图，设置透明度为 0.5
    ax.hist((d1, d2), histtype="stepfilled", stacked=True, alpha=0.5)


@image_comparison(['hist_stacked_step'])
def test_hist_stacked_step():
    # 创建一些数据
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    # 创建一个图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制堆叠的步骤直方图
    ax.hist((d1, d2), histtype="step", stacked=True)


@image_comparison(['hist_stacked_normed'])
def test_hist_stacked_density():
    # 创建一些数据
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    # 创建一个图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制堆叠的归一化直方图（密度图）
    ax.hist((d1, d2), stacked=True, density=True)


@image_comparison(['hist_step_bottom.png'], remove_text=True)
def test_hist_step_bottom():
    # 创建一些数据
    d1 = np.linspace(1, 3, 20)
    # 创建一个图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制带有底部偏移量的步骤填充直方图
    ax.hist(d1, bottom=np.arange(10), histtype="stepfilled")


def test_hist_step_geometry():
    bins = [0, 1, 2, 3]
    data = [0, 0, 1, 1, 1, 2]
    top = [[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 1], [3, 1], [3, 0]]
    bottom = [[2, 0], [2, 0], [1, 0], [1, 0], [0, 0]]

    for histtype, xy in [('step', top), ('stepfilled', top + bottom)]:
        # 绘制步骤或步骤填充直方图，并获取返回的多边形对象
        _, _, (polygon, ) = plt.hist(data, bins=bins, histtype=histtype)
        # 断言多边形对象的顶点坐标与预期的坐标相等
        assert_array_equal(polygon.get_xy(), xy)


def test_hist_step_bottom_geometry():
    bins = [0, 1, 2, 3]
    data = [0, 0, 1, 1, 1, 2]
    top = [[0, 1], [0, 3], [1, 3], [1, 5], [2, 5], [2, 2.5], [3, 2.5], [3, 1.5]]
    bottom = [[2, 1.5], [2, 2], [1, 2], [1, 1], [0, 1]]

    for histtype, xy in [('step', top), ('stepfilled', top + bottom)]:
        # 绘制带有底部偏移量的步骤或步骤填充直方图，并获取返回的多边形对象
        _, _, (polygon, ) = plt.hist(data, bins=bins, bottom=[1, 2, 1.5],
                                     histtype=histtype)
        # 断言多边形对象的顶点坐标与预期的坐标相等
        assert_array_equal(polygon.get_xy(), xy)


def test_hist_stacked_step_geometry():
    bins = [0, 1, 2, 3]
    data_1 = [0, 0, 1, 1, 1, 2]
    data_2 = [0, 1, 2]
    tops = [
        [[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 1], [3, 1], [3, 0]],
        [[0, 2], [0, 3], [1, 3], [1, 4], [2, 4], [2, 2], [3, 2], [3, 1]],
    ]
    bottoms = [
        [[2, 0], [2, 0], [1, 0], [1, 0], [0, 0]],
        [[2, 1], [2, 3], [1, 3], [1, 2], [0, 2]],
    ]
    combined = [t + b for t, b in zip(tops, bottoms)]
    # 对于每个直方图类型和对应的数据点集合，分别绘制直方图并返回绘制的图形对象列表
    for histtype, xy in [('step', tops), ('stepfilled', combined)]:
        # 绘制堆叠直方图，获取每个直方图的图形对象列表
        _, _, patches = plt.hist([data_1, data_2], bins=bins, stacked=True,
                                 histtype=histtype)
        # 断言每个直方图的图形对象数量为2，用于验证绘制的正确性
        assert len(patches) == 2
        # 获取第一个直方图的多边形对象，并断言其顶点坐标与预期相等
        polygon, = patches[0]
        assert_array_equal(polygon.get_xy(), xy[0])
        # 获取第二个直方图的多边形对象，并断言其顶点坐标与预期相等
        polygon, = patches[1]
        assert_array_equal(polygon.get_xy(), xy[1])
def test_hist_stacked_step_bottom_geometry():
    bins = [0, 1, 2, 3]  # 定义直方图的分 bin 区间
    data_1 = [0, 0, 1, 1, 1, 2]  # 第一组数据
    data_2 = [0, 1, 2]  # 第二组数据
    tops = [  # 顶部数据的坐标列表
        [[0, 1], [0, 3], [1, 3], [1, 5], [2, 5], [2, 2.5], [3, 2.5], [3, 1.5]],
        [[0, 3], [0, 4], [1, 4], [1, 6], [2, 6], [2, 3.5], [3, 3.5], [3, 2.5]],
    ]
    bottoms = [  # 底部数据的坐标列表
        [[2, 1.5], [2, 2], [1, 2], [1, 1], [0, 1]],
        [[2, 2.5], [2, 5], [1, 5], [1, 3], [0, 3]],
    ]
    combined = [t + b for t, b in zip(tops, bottoms)]  # 将顶部和底部数据结合起来

    for histtype, xy in [('step', tops), ('stepfilled', combined)]:  # 针对每种类型进行测试
        _, _, patches = plt.hist([data_1, data_2], bins=bins, stacked=True,
                                 bottom=[1, 2, 1.5], histtype=histtype)  # 创建直方图并获取 patches
        assert len(patches) == 2  # 断言确保生成了两个 patches
        polygon, = patches[0]  # 获取第一个 patch
        assert_array_equal(polygon.get_xy(), xy[0])  # 断言第一个 patch 的坐标与预期的顶部或结合数据一致
        polygon, = patches[1]  # 获取第二个 patch
        assert_array_equal(polygon.get_xy(), xy[1])  # 断言第二个 patch 的坐标与预期的顶部或结合数据一致


@image_comparison(['hist_stacked_bar'])  # 生成图片对比测试
def test_hist_stacked_bar():
    # 生成一些数据
    d = [[100, 100, 100, 100, 200, 320, 450, 80, 20, 600, 310, 800],
         [20, 23, 50, 11, 100, 420], [120, 120, 120, 140, 140, 150, 180],
         [60, 60, 60, 60, 300, 300, 5, 5, 5, 5, 10, 300],
         [555, 555, 555, 30, 30, 30, 30, 30, 100, 100, 100, 100, 30, 30],
         [30, 30, 30, 30, 400, 400, 400, 400, 400, 400, 400, 400]]
    colors = [(0.5759849696758961, 1.0, 0.0), (0.0, 1.0, 0.350624650815206),
              (0.0, 1.0, 0.6549834156005998), (0.0, 0.6569064625276622, 1.0),
              (0.28302699607823545, 0.0, 1.0), (0.6849123462299822, 0.0, 1.0)]
    labels = ['green', 'orange', ' yellow', 'magenta', 'black']
    fig, ax = plt.subplots()  # 创建图和坐标系
    ax.hist(d, bins=10, histtype='barstacked', align='mid', color=colors,
            label=labels)  # 创建堆叠条形直方图
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncols=1)  # 添加图例


def test_hist_barstacked_bottom_unchanged():
    b = np.array([10, 20])  # 定义底部的起始位置
    plt.hist([[0, 1], [0, 1]], 2, histtype="barstacked", bottom=b)  # 创建堆叠条形直方图
    assert b.tolist() == [10, 20]  # 断言底部位置未改变


def test_hist_emptydata():
    fig, ax = plt.subplots()  # 创建图和坐标系
    ax.hist([[], range(10), range(10)], histtype="step")  # 创建步进式直方图


def test_hist_labels():
    # 测试单个标签是否正常
    fig, ax = plt.subplots()  # 创建图和坐标系
    _, _, bars = ax.hist([0, 1], label=0)  # 创建直方图并获取 bars
    assert bars[0].get_label() == '0'  # 断言第一个 bar 的标签为 '0'
    _, _, bars = ax.hist([0, 1], label=[0])  # 创建直方图并获取 bars
    assert bars[0].get_label() == '0'  # 断言第一个 bar 的标签为 '0'
    _, _, bars = ax.hist([0, 1], label=None)  # 创建直方图并获取 bars
    assert bars[0].get_label() == '_nolegend_'  # 断言第一个 bar 的标签为 '_nolegend_'
    _, _, bars = ax.hist([0, 1], label='0')  # 创建直方图并获取 bars
    assert bars[0].get_label() == '0'  # 断言第一个 bar 的标签为 '0'
    _, _, bars = ax.hist([0, 1], label='00')  # 创建直方图并获取 bars
    assert bars[0].get_label() == '00'  # 断言第一个 bar 的标签为 '00'


@image_comparison(['transparent_markers'], remove_text=True)
def test_transparent_markers():
    np.random.seed(0)  # 设置随机种子
    data = np.random.random(50)  # 生成随机数据

    fig, ax = plt.subplots()  # 创建图和坐标系
    ax.plot(data, 'D', mfc='none', markersize=100)  # 绘制散点图，设置透明标记


@image_comparison(['rgba_markers'], remove_text=True)
def test_rgba_markers():
    fig, axs = plt.subplots(ncols=2)  # 创建包含两个坐标系的图
    # 定义红色标记和蓝色标记的颜色和透明度的组合列表
    rcolors = [(1, 0, 0, 1), (1, 0, 0, 0.5)]
    bcolors = [(0, 0, 1, 1), (0, 0, 1, 0.5)]
    # 定义不同图形的透明度设置，第一个图形默认为 None，第二个图形透明度为 0.2
    alphas = [None, 0.2]
    # 定义绘图参数字典，包括标记大小和边界宽度
    kw = dict(ms=100, mew=20)
    
    # 遍历透明度列表和对应索引
    for i, alpha in enumerate(alphas):
        # 遍历红色标记列表和对应索引
        for j, rcolor in enumerate(rcolors):
            # 遍历蓝色标记列表和对应索引
            for k, bcolor in enumerate(bcolors):
                # 在每个轴对象 axs[i] 上绘制圆形标记 'o'，填充色为 bcolor，边界色为 rcolor，
                # 设置透明度为 alpha，应用绘图参数 kw
                axs[i].plot(j+1, k+1, 'o', mfc=bcolor, mec=rcolor,
                            alpha=alpha, **kw)
                # 在每个轴对象 axs[i] 上绘制叉形标记 'x'，边界色为 rcolor，
                # 设置透明度为 alpha，应用绘图参数 kw
                axs[i].plot(j+1, k+3, 'x', mec=rcolor, alpha=alpha, **kw)
    
    # 设置每个轴对象的坐标范围为 x 轴 -1 到 4，y 轴 0 到 5
    for ax in axs:
        ax.axis([-1, 4, 0, 5])
@image_comparison(['mollweide_grid'], remove_text=True)
def test_mollweide_grid():
    # 定义一个测试函数，用于验证 Mollweide 投影中水平和垂直网格线的显示
    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection='mollweide')  # 在图形上添加一个 Mollweide 投影的子图
    ax.grid()  # 在子图上显示网格


def test_mollweide_forward_inverse_closure():
    # 验证 Mollweide 投影的前向->反向变换的闭合性，即近似恒等变换
    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection='mollweide')  # 在图形上添加一个 Mollweide 投影的子图

    # 设置经度、纬度上的1度网格
    lon = np.linspace(-np.pi, np.pi, 360)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, 180)[1:-1]
    lon, lat = np.meshgrid(lon, lat)
    ll = np.vstack((lon.flatten(), lat.flatten())).T

    # 执行前向变换
    xy = ax.transProjection.transform(ll)

    # 执行反向变换
    ll2 = ax.transProjection.inverted().transform(xy)

    # 比较结果
    np.testing.assert_array_almost_equal(ll, ll2, 3)


def test_mollweide_inverse_forward_closure():
    # 验证 Mollweide 投影的反向->前向变换的闭合性，即近似恒等变换
    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection='mollweide')  # 在图形上添加一个 Mollweide 投影的子图

    # 设置x、y上的网格
    x = np.linspace(0, 1, 500)
    x, y = np.meshgrid(x, x)
    xy = np.vstack((x.flatten(), y.flatten())).T

    # 执行反向变换
    ll = ax.transProjection.inverted().transform(xy)

    # 执行前向变换
    xy2 = ax.transProjection.transform(ll)

    # 比较结果
    np.testing.assert_array_almost_equal(xy, xy2, 3)


@image_comparison(['test_alpha'], remove_text=True)
def test_alpha():
    np.random.seed(0)
    data = np.random.random(50)

    fig, ax = plt.subplots()  # 创建一个新的图形对象和一个轴对象

    # 设置 alpha=.5 的标记，实线
    ax.plot(data, '-D', color=[1, 0, 0], mfc=[1, 0, 0, .5],
            markersize=20, lw=10)

    # 所有内容通过关键字参数设置为实线
    ax.plot(data + 2, '-D', color=[1, 0, 0, .5], mfc=[1, 0, 0, .5],
            markersize=20, lw=10,
            alpha=1)

    # 所有内容通过关键字参数设置为 alpha=.5
    ax.plot(data + 4, '-D', color=[1, 0, 0], mfc=[1, 0, 0],
            markersize=20, lw=10,
            alpha=.5)

    # 所有内容通过颜色设置为 alpha=.5
    ax.plot(data + 6, '-D', color=[1, 0, 0, .5], mfc=[1, 0, 0, .5],
            markersize=20, lw=10)

    # 设置 alpha=.5 的线条，实线标记
    ax.plot(data + 8, '-D', color=[1, 0, 0, .5], mfc=[1, 0, 0],
            markersize=20, lw=10)


@image_comparison(['eventplot', 'eventplot'], remove_text=True)
def test_eventplot():
    np.random.seed(0)

    data1 = np.random.random([32, 20]).tolist()
    data2 = np.random.random([6, 20]).tolist()
    data = data1 + data2
    num_datasets = len(data)

    colors1 = [[0, 1, .7]] * len(data1)
    colors2 = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [1, .75, 0],
               [1, 0, 1],
               [0, 1, 1]]
    colors = colors1 + colors2
    # 计算第一个数据集的线偏移量，起始于12，每个数据点增加0.33
    lineoffsets1 = 12 + np.arange(0, len(data1)) * .33
    
    # 手动指定第二个数据集的线偏移量
    lineoffsets2 = [-15, -3, 1, 1.5, 6, 10]
    
    # 合并两个数据集的线偏移量
    lineoffsets = lineoffsets1.tolist() + lineoffsets2

    # 创建第一个数据集的线长度，每个数据点的长度均为0.33
    linelengths1 = [.33] * len(data1)
    
    # 手动指定第二个数据集的线长度
    linelengths2 = [5, 2, 1, 1, 3, 1.5]
    
    # 合并两个数据集的线长度
    linelengths = linelengths1 + linelengths2

    # 创建一个新的图形对象
    fig = plt.figure()
    
    # 在图形上添加一个子图
    axobj = fig.add_subplot()
    
    # 使用eventplot方法绘制事件的集合，使用给定的数据
    colls = axobj.eventplot(data, colors=colors, lineoffsets=lineoffsets,
                            linelengths=linelengths)

    # 检查生成的集合数量是否等于数据集的数量
    num_collections = len(colls)
    assert num_collections == num_datasets

    # 使用上面的测试用例，为标记数据进行测试
    # 将数据包装成字典，以便传递给eventplot方法
    data = {"pos": data, "c": colors, "lo": lineoffsets, "ll": linelengths}
    
    # 创建另一个新的图形对象
    fig = plt.figure()
    
    # 在图形上添加一个子图
    axobj = fig.add_subplot()
    
    # 使用eventplot方法绘制事件的集合，使用包含标记数据的字典
    colls = axobj.eventplot("pos", colors="c", lineoffsets="lo",
                            linelengths="ll", data=data)
    
    # 再次检查生成的集合数量是否等于数据集的数量
    num_collections = len(colls)
    assert num_collections == num_datasets
@image_comparison(['test_eventplot_defaults.png'], remove_text=True)
def test_eventplot_defaults():
    """
    test that eventplot produces the correct output given the default params
    (see bug #3728)
    """
    np.random.seed(0)  # 设置随机种子为0，以确保可重复性

    data1 = np.random.random([32, 20]).tolist()  # 创建一个32x20的随机数列表并转换为Python列表
    data2 = np.random.random([6, 20]).tolist()  # 创建一个6x20的随机数列表并转换为Python列表
    data = data1 + data2  # 将两个数据列表合并成一个列表

    fig = plt.figure()  # 创建一个新的图形对象
    axobj = fig.add_subplot()  # 在图形对象上添加一个子图对象
    axobj.eventplot(data)  # 在子图对象上绘制事件图


@pytest.mark.parametrize(('colors'), [
    ('0.5',),  # 使用字符串颜色表示，长度超过1字符，测试先前的bug #8193修复前不应该成功
    ('tab:orange', 'tab:pink', 'tab:cyan', 'bLacK'),  # 大小写不敏感的颜色参数列表
    ('red', (0, 1, 0), None, (1, 0, 1, 0.5)),  # 包含多种类型的颜色参数列表
])
def test_eventplot_colors(colors):
    """Test the *colors* parameter of eventplot. Inspired by issue #8193."""
    data = [[0], [1], [2], [3]]  # 包含4个不同事件的数据列表

    # 构建预期颜色列表
    expected = [c if c is not None else 'C0' for c in colors]
    # 将颜色列表转换为RGBA颜色数组
    # 注意：['rgbk']不是to_rgba_array的有效参数，而'rgbk'是有效参数
    if len(expected) == 1:
        expected = expected[0]
    expected = np.broadcast_to(mcolors.to_rgba_array(expected), (len(data), 4))

    fig, ax = plt.subplots()  # 创建一个新的图形对象和子图对象
    if len(colors) == 1:  # 如果颜色参数只有一个元素（如'0.5'或'rgbk'）
        colors = colors[0]
    collections = ax.eventplot(data, colors=colors)  # 在子图对象上绘制事件图，使用给定颜色参数

    for coll, color in zip(collections, expected):
        assert_allclose(coll.get_color(), color)  # 检查每个事件图集合的颜色是否与预期一致


def test_eventplot_alpha():
    fig, ax = plt.subplots()  # 创建一个新的图形对象和子图对象

    # 统一设置所有事件图集合的透明度
    collections = ax.eventplot([[0, 2, 4], [1, 3, 5, 7]], alpha=0.7)
    assert collections[0].get_alpha() == 0.7  # 检查第一个事件图集合的透明度设置是否正确
    assert collections[1].get_alpha() == 0.7  # 检查第二个事件图集合的透明度设置是否正确

    # 分别设置每个事件图集合的透明度
    collections = ax.eventplot([[0, 2, 4], [1, 3, 5, 7]], alpha=[0.5, 0.7])
    assert collections[0].get_alpha() == 0.5  # 检查第一个事件图集合的透明度设置是否正确
    assert collections[1].get_alpha() == 0.7  # 检查第二个事件图集合的透明度设置是否正确

    with pytest.raises(ValueError, match="alpha and positions are unequal"):
        ax.eventplot([[0, 2, 4], [1, 3, 5, 7]], alpha=[0.5, 0.7, 0.9])  # 引发异常：透明度参数与位置列表长度不一致

    with pytest.raises(ValueError, match="alpha and positions are unequal"):
        ax.eventplot([0, 2, 4], alpha=[0.5, 0.7])  # 引发异常：透明度参数与位置列表长度不一致


@image_comparison(['test_eventplot_problem_kwargs.png'], remove_text=True)
def test_eventplot_problem_kwargs(recwarn):
    """
    test that 'singular' versions of LineCollection props raise an
    MatplotlibDeprecationWarning rather than overriding the 'plural' versions
    (e.g., to prevent 'color' from overriding 'colors', see issue #4297)
    """
    np.random.seed(0)

    data1 = np.random.random([20]).tolist()  # 创建一个包含20个随机数的列表并转换为Python列表
    data2 = np.random.random([10]).tolist()  # 创建一个包含10个随机数的列表并转换为Python列表
    data = [data1, data2]  # 将两个数据列表合并成一个列表

    fig = plt.figure()  # 创建一个新的图形对象
    axobj = fig.add_subplot()  # 在图形对象上添加一个子图对象
    # 使用 eventplot 函数绘制事件图，显示给定的数据
    axobj.eventplot(data,
                    # 指定事件的颜色，第一个数据集用红色和蓝色表示
                    colors=['r', 'b'],
                    # DEPRECATED: 请使用 `color` 参数代替 `colors` 参数
                    color=['c', 'm'],
                    # 指定线条的宽度，第一个数据集的线宽为2，第二个为1
                    linewidths=[2, 1],
                    # DEPRECATED: 请使用 `linewidth` 参数代替 `linewidths` 参数
                    linewidth=[1, 2],
                    # 指定线条的样式，第一个数据集用实线，第二个用虚线
                    linestyles=['solid', 'dashed'],
                    # DEPRECATED: 请使用 `linestyle` 参数代替 `linestyles` 参数
                    linestyle=['dashdot', 'dotted'])
    
    # 断言语句，验证警告记录数是否为3，如果不是会引发 AssertionError
    assert len(recwarn) == 3
    # 断言语句，验证所有的警告是否都是 MatplotlibDeprecationWarning 的子类
    assert all(issubclass(wi.category, mpl.MatplotlibDeprecationWarning)
               for wi in recwarn)
# 定义一个测试函数，用于测试空的 eventplot
def test_empty_eventplot():
    # 创建一个包含单个子图的 Figure 对象和对应的 Axes 对象
    fig, ax = plt.subplots(1, 1)
    # 在 Axes 对象上绘制一个空的 eventplot，颜色设置为透明
    ax.eventplot([[]], colors=[(0.0, 0.0, 0.0, 0.0)])
    # 绘制图形
    plt.draw()


# 使用参数化测试装饰器进行多组测试数据和参数的测试
@pytest.mark.parametrize('data', [[[]], [[], [0, 1]], [[0, 1], []]])
@pytest.mark.parametrize('orientation', [None, 'vertical', 'horizontal'])
def test_eventplot_orientation(data, orientation):
    """修复问题 #6412 时引入的测试函数。"""
    # 根据给定的方向参数，设置绘图选项
    opts = {} if orientation is None else {'orientation': orientation}
    # 创建一个包含单个子图的 Figure 对象和对应的 Axes 对象
    fig, ax = plt.subplots(1, 1)
    # 在 Axes 对象上绘制 eventplot，使用给定的数据和选项
    ax.eventplot(data, **opts)
    # 绘制图形
    plt.draw()


# 使用自定义装饰器 check_figures_equal 进行比较两个图形对象的测试
@check_figures_equal(extensions=['png'])
def test_eventplot_units_list(fig_test, fig_ref):
    # 测试列表转换是否正确：
    ts_1 = [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 2),
            datetime.datetime(2021, 1, 3)]
    ts_2 = [datetime.datetime(2021, 1, 15), datetime.datetime(2021, 1, 16)]

    # 在参考图形上创建一个子图对象
    ax = fig_ref.subplots()
    # 在子图上绘制 eventplot，设置线偏移量为0
    ax.eventplot(ts_1, lineoffsets=0)
    # 在子图上绘制 eventplot，设置线偏移量为1
    ax.eventplot(ts_2, lineoffsets=1)

    # 在测试图形上创建一个子图对象
    ax = fig_test.subplots()
    # 在子图上绘制 eventplot，传入两组时间序列数据
    ax.eventplot([ts_1, ts_2])


# 使用图像比较装饰器 image_comparison 进行测试，比较 marker_styles.png 图像
@image_comparison(['marker_styles.png'], remove_text=True)
def test_marker_styles():
    fig, ax = plt.subplots()
    # 生成测试图像时，去除了 'None'，添加了 'none'。通过将 'none' 移动到最前面，可以避免重新生成测试图像。
    markers = sorted(matplotlib.markers.MarkerStyle.markers,
                     key=lambda x: str(type(x))+str(x))
    markers.remove('none')
    markers = ['none', *markers]
    # 遍历不同的 marker 类型，绘制带有不同样式的数据点
    for y, marker in enumerate(markers):
        ax.plot((y % 2)*5 + np.arange(10)*10, np.ones(10)*10*y, linestyle='',
                marker=marker, markersize=10+y/5, label=marker)


# 使用图像比较装饰器 image_comparison 进行测试，比较 rc_markerfill.png 图像
@image_comparison(['rc_markerfill.png'],
                  tol=0.037 if platform.machine() == 'arm64' else 0)
def test_markers_fillstyle_rcparams():
    fig, ax = plt.subplots()
    x = np.arange(7)
    # 遍历不同的 markers.fillstyle 配置，绘制带有不同填充风格的数据点
    for idx, (style, marker) in enumerate(
            [('top', 's'), ('bottom', 'o'), ('none', '^')]):
        matplotlib.rcParams['markers.fillstyle'] = style
        ax.plot(x+idx, marker=marker)


# 使用图像比较装饰器 image_comparison 进行测试，比较 vertex_markers.png 图像
@image_comparison(['vertex_markers.png'], remove_text=True)
def test_vertex_markers():
    data = list(range(10))
    marker_as_tuple = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    marker_as_list = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    fig, ax = plt.subplots()
    # 在 Axes 对象上绘制使用元组和列表作为 marker 的数据点
    ax.plot(data, linestyle='', marker=marker_as_tuple, mfc='k')
    ax.plot(data[::-1], linestyle='', marker=marker_as_list, mfc='b')
    # 设置 X 和 Y 轴的显示范围
    ax.set_xlim([-1, 10])
    ax.set_ylim([-1, 10])


# 使用图像比较装饰器 image_comparison 进行测试，比较 vline_hline_zorder 和 errorbar_zorder 图像
@image_comparison(['vline_hline_zorder', 'errorbar_zorder'],
                  tol=0 if platform.machine() == 'x86_64' else 0.026)
def test_eb_line_zorder():
    x = list(range(10))

    # 创建一个 Figure 对象
    fig = plt.figure()
    # 获取当前 Axes 对象
    ax = fig.gca()
    # 在 Axes 对象上绘制数据线条，设置线宽和 Z 轴顺序
    ax.plot(x, lw=10, zorder=5)
    # 在图形上添加水平线，位置在 y=1，颜色为红色，线宽为10，层级为1
    ax.axhline(1, color='red', lw=10, zorder=1)
    # 在图形上添加水平线，位置在 y=5，颜色为绿色，线宽为10，层级为10
    ax.axhline(5, color='green', lw=10, zorder=10)
    # 在图形上添加垂直线，位置在 x=7，颜色为品红色，线宽为10，层级为7
    ax.axvline(7, color='m', lw=10, zorder=7)
    # 在图形上添加垂直线，位置在 x=2，颜色为黑色，线宽为10，层级为3
    ax.axvline(2, color='k', lw=10, zorder=3)

    # 设置图形的标题为 "axvline and axhline zorder test"

    # 现在切换到更多面向对象的接口，以使用更多功能。
    fig = plt.figure()
    # 获取当前图形的坐标轴
    ax = fig.gca()
    # 创建一个包含从0到9的列表
    x = list(range(10))
    # 创建一个全零数组，长度为10
    y = np.zeros(10)
    # 创建一个包含从0到9的列表
    yerr = list(range(10))
    # 在坐标轴上绘制误差条，x坐标为x，y坐标为y，y误差为yerr，层级为5，线宽为5，颜色为红色
    ax.errorbar(x, y, yerr=yerr, zorder=5, lw=5, color='r')
    # 对于0到9的每个数值j，绘制水平线，位置在y=j，线宽为5，颜色为黑色，层级为j
    for j in range(10):
        ax.axhline(j, lw=5, color='k', zorder=j)
        # 绘制水平线，位置在y=-j，线宽为5，颜色为黑色，层级为j
        ax.axhline(-j, lw=5, color='k', zorder=j)

    # 设置图形的标题为 "errorbar zorder test"
@check_figures_equal()
# 定义一个装饰器，用于检查两个图形是否相等
def test_axline_loglog(fig_test, fig_ref):
    # 在测试图形上创建子图
    ax = fig_test.subplots()
    # 设置子图的 x 和 y 轴的范围
    ax.set(xlim=(0.1, 10), ylim=(1e-3, 1))
    # 在对数-对数坐标轴上绘制一条线段
    ax.loglog([.3, .6], [.3, .6], ".-")
    # 在对数-对数坐标轴上绘制一条通过两点的直线
    ax.axline((1, 1e-3), (10, 1e-2), c="k")

    # 在参考图形上创建子图
    ax = fig_ref.subplots()
    # 设置子图的 x 和 y 轴的范围
    ax.set(xlim=(0.1, 10), ylim=(1e-3, 1))
    # 在对数-对数坐标轴上绘制一条线段
    ax.loglog([.3, .6], [.3, .6], ".-")
    # 在对数-对数坐标轴上绘制一条直线
    ax.loglog([1, 10], [1e-3, 1e-2], c="k")


@check_figures_equal()
# 定义一个装饰器，用于检查两个图形是否相等
def test_axline(fig_test, fig_ref):
    # 在测试图形上创建子图
    ax = fig_test.subplots()
    # 设置子图的 x 和 y 轴的范围
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    # 在子图上绘制一条通过两点的直线
    ax.axline((0, 0), (1, 1))
    # 在子图上绘制一条通过两点的直线，指定颜色
    ax.axline((0, 0), (1, 0), color='C1')
    # 在子图上绘制一条通过两点的直线，指定颜色
    ax.axline((0, 0.5), (1, 0.5), color='C2')
    # 在子图上绘制一条斜率为 0 的直线，指定颜色
    ax.axline((-0.7, -0.5), slope=0, color='C3')
    # 在子图上绘制一条斜率为 -0.5 的直线，指定颜色
    ax.axline((1, -0.5), slope=-0.5, color='C4')
    # 在子图上绘制一条斜率为正无穷的直线，指定颜色
    ax.axline((-0.5, 1), slope=float('inf'), color='C5')

    # 在参考图形上创建子图
    ax = fig_ref.subplots()
    # 设置子图的 x 和 y 轴的范围
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    # 在子图上绘制一条对角线
    ax.plot([-1, 1], [-1, 1])
    # 在子图上绘制一条水平线，指定颜色
    ax.axhline(0, color='C1')
    # 在子图上绘制一条水平线，指定颜色
    ax.axhline(0.5, color='C2')
    # 在子图上绘制一条斜率为 -0.5 的直线
    ax.axhline(-0.5, color='C3')
    # 在子图上绘制一条斜率为 -0.5 的线段，指定颜色
    ax.plot([-1, 1], [0.5, -0.5], color='C4')
    # 在子图上绘制一条垂直线，指定颜色
    ax.axvline(-0.5, color='C5')


@check_figures_equal()
# 定义一个装饰器，用于检查两个图形是否相等
def test_axline_transaxes(fig_test, fig_ref):
    # 在测试图形上创建子图
    ax = fig_test.subplots()
    # 设置子图的 x 和 y 轴的范围
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    # 在子图上绘制一条通过两点的直线，使用轴变换
    ax.axline((0, 0), slope=1, transform=ax.transAxes)
    # 在子图上绘制一条斜率为 1 的直线，指定颜色，使用轴变换
    ax.axline((1, 0.5), slope=1, color='C1', transform=ax.transAxes)
    # 在子图上绘制一条斜率为 0 的直线，指定颜色，使用轴变换
    ax.axline((0.5, 0.5), slope=0, color='C2', transform=ax.transAxes)
    # 在子图上绘制一条垂直线，指定颜色，使用轴变换
    ax.axline((0.5, 0), (0.5, 1), color='C3', transform=ax.transAxes)

    # 在参考图形上创建子图
    ax = fig_ref.subplots()
    # 设置子图的 x 和 y 轴的范围
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    # 在子图上绘制一条对角线
    ax.plot([-1, 1], [-1, 1])
    # 在子图上绘制一条线段，指定颜色
    ax.plot([0, 1], [-1, 0], color='C1')
    # 在子图上绘制一条水平线，指定颜色
    ax.plot([-1, 1], [0, 0], color='C2')
    # 在子图上绘制一条垂直线，指定颜色
    ax.plot([0, 0], [-1, 1], color='C3')


@check_figures_equal()
# 定义一个装饰器，用于检查两个图形是否相等
def test_axline_transaxes_panzoom(fig_test, fig_ref):
    # 测试绘图在进行平移/缩放以及绘图后的图形调整时的稳健性
    # 在测试图形上创建子图
    ax = fig_test.subplots()
    # 设置子图的 x 和 y 轴的范围
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    # 在子图上绘制一条通过两点的直线，使用轴变换
    ax.axline((0, 0), slope=1, transform=ax.transAxes)
    # 在子图上绘制一条斜率为 2 的直线，指定颜色，使用轴变换
    ax.axline((0.5, 0.5), slope=2, color='C1', transform=ax.transAxes)
    # 在子图上绘制一条斜率为 0 的直线，指定颜色，使用轴变换
    ax.axline((0.5, 0.5), slope=0, color='C2', transform=ax.transAxes)
    # 设置子图的 x 和 y 轴的范围
    ax.set(xlim=(0, 5), ylim=(0, 10))
    # 设置图形的尺寸
    fig_test.set_size_inches(3, 3)

    # 在参考图形上创建子图
    ax = fig_ref.subplots()
    # 设置子图的 x 和 y 轴的范围
    ax.set(xlim=(0, 5), ylim=(0, 10))
    # 设置图形的尺寸
    fig_ref.set_size_inches(3, 3)
    # 在子图上绘制一条线段
    ax.plot([0, 5], [0, 5])
    # 在子图上绘制一条线段，指定颜色
    ax.plot([0, 5], [0, 10], color='C1')
    # 在子图上绘制一条线段
    # 使用 pytest 的上下文管理器检查是否引发 ValueError 异常
    with pytest.raises(ValueError):
        # 在图形对象 ax 上调用 axline 方法，画一条直线从 (0, 0) 到 (0, 0)
        # 这里注释指出，这种情况下传入两个相同的点是不允许的，会引发 ValueError 异常
        ax.axline((0, 0), (0, 0))  
        
        # 绘制图形，更新显示
        plt.draw()
# 使用图像比较功能，测试垂直线绘制的不同情况
@image_comparison(['vlines_basic', 'vlines_with_nan', 'vlines_masked'],
                  extensions=['png'])
def test_vlines():
    # normal case
    # 定义普通情况下的 x 和 y 值
    x1 = [2, 3, 4, 5, 7]
    y1 = [2, -6, 3, 8, 2]
    # 创建一个新的图表和轴对象
    fig1, ax1 = plt.subplots()
    # 绘制垂直线，颜色为绿色，线宽为5
    ax1.vlines(x1, 0, y1, colors='g', linewidth=5)

    # GH issue #7406 case
    # 处理 GH issue #7406 的情况
    x2 = [2, 3, 4, 5, 6, 7]
    y2 = [2, -6, 3, 8, np.nan, 2]
    # 创建一个包含三个轴对象的图表
    fig2, (ax2, ax3, ax4) = plt.subplots(nrows=3, figsize=(4, 8))
    # 在第一个轴上绘制垂直线，处理 NaN 值，颜色为绿色，线宽为5
    ax2.vlines(x2, 0, y2, colors='g', linewidth=5)

    x3 = [2, 3, 4, 5, 6, 7]
    y3 = [np.nan, 2, -6, 3, 8, 2]
    # 在第二个轴上绘制垂直线，处理 NaN 值，颜色为红色，线宽为3，线型为虚线
    ax3.vlines(x3, 0, y3, colors='r', linewidth=3, linestyle='--')

    x4 = [2, 3, 4, 5, 6, 7]
    y4 = [np.nan, 2, -6, 3, 8, np.nan]
    # 在第三个轴上绘制垂直线，处理 NaN 值，颜色为黑色，线宽为2
    ax4.vlines(x4, 0, y4, colors='k', linewidth=2)

    # 调整 x 轴的范围以便更好地观察线条
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, 10)

    # 检查所有轴的 y 轴限制是否自动设置为相同的值
    assert ax1.get_ylim() == ax2.get_ylim()
    assert ax1.get_ylim() == ax3.get_ylim()
    assert ax1.get_ylim() == ax4.get_ylim()

    fig3, ax5 = plt.subplots()
    # 创建带有屏蔽值的掩码数组
    x5 = np.ma.masked_equal([2, 4, 6, 8, 10, 12], 8)
    ymin5 = np.ma.masked_equal([0, 1, -1, 0, 2, 1], 2)
    ymax5 = np.ma.masked_equal([13, 14, 15, 16, 17, 18], 18)
    # 在单个轴上绘制带有屏蔽值的垂直线，颜色为黑色，线宽为2
    ax5.vlines(x5, ymin5, ymax5, colors='k', linewidth=2)
    ax5.set_xlim(0, 15)


def test_vlines_default():
    # 测试默认情况下的垂直线绘制
    fig, ax = plt.subplots()
    with mpl.rc_context({'lines.color': 'red'}):
        # 绘制一个垂直线，颜色为红色
        lines = ax.vlines(0.5, 0, 1)
        # 断言线条颜色与设定的红色相同
        assert mpl.colors.same_color(lines.get_color(), 'red')


@image_comparison(['hlines_basic', 'hlines_with_nan', 'hlines_masked'],
                  extensions=['png'])
def test_hlines():
    # normal case
    # 定义普通情况下的 y 和 x 值
    y1 = [2, 3, 4, 5, 7]
    x1 = [2, -6, 3, 8, 2]
    # 创建一个新的图表和轴对象
    fig1, ax1 = plt.subplots()
    # 绘制水平线，颜色为绿色，线宽为5
    ax1.hlines(y1, 0, x1, colors='g', linewidth=5)

    # GH issue #7406 case
    # 处理 GH issue #7406 的情况
    y2 = [2, 3, 4, 5, 6, 7]
    x2 = [2, -6, 3, 8, np.nan, 2]
    # 创建一个包含三个轴对象的图表
    fig2, (ax2, ax3, ax4) = plt.subplots(nrows=3, figsize=(4, 8))
    # 在第一个轴上绘制水平线，处理 NaN 值，颜色为绿色，线宽为5
    ax2.hlines(y2, 0, x2, colors='g', linewidth=5)

    y3 = [2, 3, 4, 5, 6, 7]
    x3 = [np.nan, 2, -6, 3, 8, 2]
    # 在第二个轴上绘制水平线，处理 NaN 值，颜色为红色，线宽为3，线型为虚线
    ax3.hlines(y3, 0, x3, colors='r', linewidth=3, linestyle='--')

    y4 = [2, 3, 4, 5, 6, 7]
    x4 = [np.nan, 2, -6, 3, 8, np.nan]
    # 在第三个轴上绘制水平线，处理 NaN 值，颜色为黑色，线宽为2
    ax4.hlines(y4, 0, x4, colors='k', linewidth=2)

    # 调整 y 轴的范围以便更好地观察线条
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim(0, 10)

    # 检查所有轴的 x 轴限制是否自动设置为相同的值
    assert ax1.get_xlim() == ax2.get_xlim()
    assert ax1.get_xlim() == ax3.get_xlim()
    assert ax1.get_xlim() == ax4.get_xlim()

    fig3, ax5 = plt.subplots()
    # 创建带有屏蔽值的掩码数组
    y5 = np.ma.masked_equal([2, 4, 6, 8, 10, 12], 8)
    xmin5 = np.ma.masked_equal([0, 1, -1, 0, 2, 1], 2)
    xmax5 = np.ma.masked_equal([13, 14, 15, 16, 17, 18], 18)
    # 在单个轴上绘制带有屏蔽值的水平线，颜色为黑色，线宽为2
    ax5.hlines(y5, xmin5, xmax5, colors='k', linewidth=2)
    ax5.set_ylim(0, 15)


def test_hlines_default():
    # 测试默认情况下的水平线绘制
    fig, ax = plt.subplots()
    # 使用 matplotlib 的上下文管理器修改绘图配置，设置水平线的颜色为红色
    with mpl.rc_context({'lines.color': 'red'}):
        # 在坐标轴 ax 上绘制一条 y=0.5 的水平线，范围从 x=0 到 x=1
        lines = ax.hlines(0.5, 0, 1)
        # 断言绘制的水平线的颜色是否与预期的红色相同
        assert mpl.colors.same_color(lines.get_color(), 'red')
@pytest.mark.parametrize('data', [[1, 2, 3, np.nan, 5],
                                  np.ma.masked_equal([1, 2, 3, 4, 5], 4)])
@check_figures_equal(extensions=["png"])
def test_lines_with_colors(fig_test, fig_ref, data):
    # 定义测试中使用的颜色列表
    test_colors = ['red', 'green', 'blue', 'purple', 'orange']
    
    # 在测试图中添加子图1，绘制垂直线图
    fig_test.add_subplot(2, 1, 1).vlines(data, 0, 1,
                                         colors=test_colors, linewidth=5)
    
    # 在测试图中添加子图2，绘制水平线图
    fig_test.add_subplot(2, 1, 2).hlines(data, 0, 1,
                                         colors=test_colors, linewidth=5)

    # 期望的 x 坐标和对应的颜色
    expect_xy = [1, 2, 3, 5]
    expect_color = ['red', 'green', 'blue', 'orange']
    
    # 在参考图中添加子图1，绘制垂直线图
    fig_ref.add_subplot(2, 1, 1).vlines(expect_xy, 0, 1,
                                        colors=expect_color, linewidth=5)
    
    # 在参考图中添加子图2，绘制水平线图
    fig_ref.add_subplot(2, 1, 2).hlines(expect_xy, 0, 1,
                                        colors=expect_color, linewidth=5)


@image_comparison(['vlines_hlines_blended_transform'],
                  extensions=['png'], style='mpl20')
def test_vlines_hlines_blended_transform():
    # 创建时间序列 t 和对应的函数值 s
    t = np.arange(5.0, 10.0, 0.1)
    s = np.exp(-t) + np.sin(2 * np.pi * t) + 10
    
    # 创建包含两个子图的图形对象
    fig, (hax, vax) = plt.subplots(2, 1, figsize=(6, 6))
    
    # 在水平子图上绘制函数图像
    hax.plot(t, s, '^')
    # 在水平子图上绘制水平线
    hax.hlines([10, 9], xmin=0, xmax=0.5,
               transform=hax.get_yaxis_transform(), colors='r')
    
    # 在垂直子图上绘制函数图像
    vax.plot(t, s, '^')
    # 在垂直子图上绘制垂直线
    vax.vlines([6, 7], ymin=0, ymax=0.15, transform=vax.get_xaxis_transform(),
               colors='r')


@image_comparison(['step_linestyle', 'step_linestyle'], remove_text=True,
                  tol=0.2)
def test_step_linestyle():
    # 定义 x 和 y 数组
    x = y = np.arange(10)

    # 创建包含四个子图的图形对象
    fig, ax_lst = plt.subplots(2, 2)
    ax_lst = ax_lst.flatten()

    # 不同线条样式列表
    ln_styles = ['-', '--', '-.', ':']

    # 在每个子图上绘制步阶图
    for ax, ls in zip(ax_lst, ln_styles):
        ax.step(x, y, lw=5, linestyle=ls, where='pre')
        ax.step(x, y + 1, lw=5, linestyle=ls, where='mid')
        ax.step(x, y + 2, lw=5, linestyle=ls, where='post')
        ax.set_xlim([-1, 5])
        ax.set_ylim([-1, 7])

    # 使用标签数据进行测试，重复上述步骤
    data = {"X": x, "Y0": y, "Y1": y+1, "Y2": y+2}
    fig, ax_lst = plt.subplots(2, 2)
    ax_lst = ax_lst.flatten()
    for ax, ls in zip(ax_lst, ln_styles):
        ax.step("X", "Y0", lw=5, linestyle=ls, where='pre', data=data)
        ax.step("X", "Y1", lw=5, linestyle=ls, where='mid', data=data)
        ax.step("X", "Y2", lw=5, linestyle=ls, where='post', data=data)
        ax.set_xlim([-1, 5])
        ax.set_ylim([-1, 7])


@image_comparison(['mixed_collection'], remove_text=True)
def test_mixed_collection():
    # 创建包含一个子图的图形对象
    fig, ax = plt.subplots()

    # 创建一个圆形对象
    c = mpatches.Circle((8, 8), radius=4, facecolor='none', edgecolor='green')
    # 创建一个 PatchCollection 对象 p1，包含一个图形 c，且要求与原始匹配
    p1 = mpl.collections.PatchCollection([c], match_original=True)
    # 设置 p1 的偏移量
    p1.set_offsets([[0, 0], [24, 24]])
    # 设置 p1 中图形的线宽
    p1.set_linewidths([1, 5])

    # 创建另一个 PatchCollection 对象 p2，包含一个图形 c，但不能与原始匹配，因为边缘的 alpha 值有变化
    p2 = mpl.collections.PatchCollection([c], match_original=True)
    # 设置 p2 的偏移量
    p2.set_offsets([[48, 0], [-32, -16]])
    # 设置 p2 中图形的线宽
    p2.set_linewidths([1, 5])
    # 设置 p2 中图形的边缘颜色，包括 alpha 值
    p2.set_edgecolors([[0, 0, 0.1, 1.0], [0, 0, 0.1, 0.5]])

    # 设置坐标轴的背景颜色为灰色
    ax.patch.set_color('0.5')
    # 将 p1 添加到坐标轴
    ax.add_collection(p1)
    # 将 p2 添加到坐标轴
    ax.add_collection(p2)

    # 设置坐标轴的 x 范围
    ax.set_xlim(0, 16)
    # 设置坐标轴的 y 范围
    ax.set_ylim(0, 16)
def test_subplot_key_hash():
    # 创建一个包含5行1列子图的绘图区域，并选择第一个子图
    ax = plt.subplot(np.int32(5), np.int64(1), 1)
    # 添加一个与当前轴共享x轴的新y轴
    ax.twinx()
    # 断言当前子图的几何配置是否为(5, 1, 0, 0)
    assert ax.get_subplotspec().get_geometry() == (5, 1, 0, 0)


@image_comparison(
    ["specgram_freqs.png", "specgram_freqs_linear.png",
     "specgram_noise.png", "specgram_noise_linear.png"],
    remove_text=True, tol=0.07, style="default")
def test_specgram():
    """Test axes.specgram in default (psd) mode."""

    # 使用旧的默认设置以匹配现有基线图像
    matplotlib.rcParams['image.interpolation'] = 'nearest'

    # 定义参数
    n = 1000  # 信号长度
    Fs = 10.  # 采样频率

    # 频率刺激
    fstims = [[Fs/4, Fs/5, Fs/11], [Fs/4.7, Fs/5.6, Fs/11.9]]
    NFFT_freqs = int(10 * Fs / np.min(fstims))  # 频率刺激的NFFT值
    x = np.arange(0, n, 1/Fs)  # 时间轴

    # 频率刺激信号
    y_freqs = np.concatenate(
        np.sin(2 * np.pi * np.multiply.outer(fstims, x)).sum(axis=1))

    # 噪声信号
    NFFT_noise = int(10 * Fs / 11)
    np.random.seed(0)
    y_noise = np.concatenate([np.random.standard_normal(n), np.random.rand(n)])

    all_sides = ["default", "onesided", "twosided"]

    # 对每一种信号类型（频率刺激和噪声）进行循环处理
    for y, NFFT in [(y_freqs, NFFT_freqs), (y_noise, NFFT_noise)]:
        noverlap = NFFT // 2
        pad_to = int(2 ** np.ceil(np.log2(NFFT)))

        # 对每一种图形边界类型进行循环处理
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            # 在当前子图上绘制谱图（默认psd模式）
            ax.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap,
                        pad_to=pad_to, sides=sides)

        # 对每一种图形边界类型进行循环处理
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            # 在当前子图上绘制谱图（线性模式），使用对数归一化
            ax.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap,
                        pad_to=pad_to, sides=sides,
                        scale="linear", norm=matplotlib.colors.LogNorm())


@image_comparison(
    ["specgram_magnitude_freqs.png", "specgram_magnitude_freqs_linear.png",
     "specgram_magnitude_noise.png", "specgram_magnitude_noise_linear.png"],
    remove_text=True, tol=0.07, style="default")
def test_specgram_magnitude():
    """Test axes.specgram in magnitude mode."""

    # 使用旧的默认设置以匹配现有基线图像
    matplotlib.rcParams['image.interpolation'] = 'nearest'

    # 定义参数
    n = 1000  # 信号长度
    Fs = 10.  # 采样频率

    # 频率刺激
    fstims = [[Fs/4, Fs/5, Fs/11], [Fs/4.7, Fs/5.6, Fs/11.9]]
    NFFT_freqs = int(100 * Fs / np.min(fstims))  # 频率刺激的NFFT值
    x = np.arange(0, n, 1/Fs)  # 时间轴

    # 频率刺激信号
    y = np.sin(2 * np.pi * np.multiply.outer(fstims, x)).sum(axis=1)
    y[:, -1] = 1  # 设置最后一个元素为1
    y_freqs = np.hstack(y)

    # 噪声信号
    NFFT_noise = int(10 * Fs / 11)
    np.random.seed(0)
    y_noise = np.concatenate([np.random.standard_normal(n), np.random.rand(n)])

    all_sides = ["default", "onesided", "twosided"]
    # 对于每一对频率数据和对应的NFFT，分别执行以下操作
    for y, NFFT in [(y_freqs, NFFT_freqs), (y_noise, NFFT_noise)]:
        # 计算重叠窗口的长度
        noverlap = NFFT // 2
        # 将NFFT填充到最接近的2的幂次方的整数
        pad_to = int(2 ** np.ceil(np.log2(NFFT)))
        # 对每个子图进行循环，创建3个子图对象
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            # 绘制谱图，使用频率和参数进行设置
            ax.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap,
                        pad_to=pad_to, sides=sides, mode="magnitude")
        # 再次对每个子图进行循环，创建3个子图对象
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            # 绘制谱图，使用额外的线性和对数参数进行设置
            ax.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap,
                        pad_to=pad_to, sides=sides, mode="magnitude",
                        scale="linear", norm=matplotlib.colors.LogNorm())
@image_comparison(
    ["specgram_angle_freqs.png", "specgram_phase_freqs.png",
     "specgram_angle_noise.png", "specgram_phase_noise.png"],
    remove_text=True, tol=0.07, style="default")
def test_specgram_angle():
    """Test axes.specgram in angle and phase modes."""

    # 使用前面的默认设置以匹配现有基准图像
    matplotlib.rcParams['image.interpolation'] = 'nearest'

    # 定义信号长度和采样频率
    n = 1000
    Fs = 10.

    # 定义频率刺激
    fstims = [[Fs/4, Fs/5, Fs/11], [Fs/4.7, Fs/5.6, Fs/11.9]]
    # 计算频率刺激对应的NFFT
    NFFT_freqs = int(10 * Fs / np.min(fstims))
    # 生成时间序列
    x = np.arange(0, n, 1/Fs)
    # 生成正弦信号，并求和以产生两个模式的信号
    y = np.sin(2 * np.pi * np.multiply.outer(fstims, x)).sum(axis=1)
    # 将每个信号的最后一个值设置为1
    y[:, -1] = 1
    # 将所有信号合并成一个数组
    y_freqs = np.hstack(y)

    # 定义噪声对应的NFFT
    NFFT_noise = int(10 * Fs / 11)
    # 设置随机数种子
    np.random.seed(0)
    # 生成噪声信号
    y_noise = np.concatenate([np.random.standard_normal(n), np.random.rand(n)])

    # 定义所有可能的边界情况
    all_sides = ["default", "onesided", "twosided"]
    # 遍历信号和对应的NFFT
    for y, NFFT in [(y_freqs, NFFT_freqs), (y_noise, NFFT_noise)]:
        # 设置重叠部分
        noverlap = NFFT // 2
        # 将pad_to设置为最接近NFFT的2的幂
        pad_to = int(2 ** np.ceil(np.log2(NFFT)))
        # 遍历模式：角度和相位
        for mode in ["angle", "phase"]:
            # 创建三个子图，并在每个上绘制specgram
            for ax, sides in zip(plt.figure().subplots(3), all_sides):
                ax.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap,
                            pad_to=pad_to, sides=sides, mode=mode)
                # 断言设置scale="dB"时抛出 ValueError 异常
                with pytest.raises(ValueError):
                    ax.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap,
                                pad_to=pad_to, sides=sides, mode=mode,
                                scale="dB")


def test_specgram_fs_none():
    """Test axes.specgram when Fs is None, should not throw error."""
    # 调用plt.specgram测试当Fs为None时不应抛出错误
    spec, freqs, t, im = plt.specgram(np.ones(300), Fs=None, scale='linear')
    # 检查图像的边界值
    xmin, xmax, freq0, freq1 = im.get_extent()
    assert xmin == 32 and xmax == 96


@check_figures_equal(extensions=["png"])
def test_specgram_origin_rcparam(fig_test, fig_ref):
    """Test specgram ignores image.origin rcParam and uses origin 'upper'."""
    # 创建500个点的时间序列
    t = np.arange(500)
    # 生成正弦信号
    signal = np.sin(t)

    # 设置全局参数，图像的origin属性为'upper'
    plt.rcParams["image.origin"] = 'upper'

    # 参考图像：使用默认的origin属性'upper'绘制第一个图
    fig_ref.subplots().specgram(signal)

    # 尝试覆盖origin设置为'lower'
    plt.rcParams["image.origin"] = 'lower'

    # 测试：origin='lower'应该被忽略
    fig_test.subplots().specgram(signal)


def test_specgram_origin_kwarg():
    """Ensure passing origin as a kwarg raises a TypeError."""
    # 创建500个点的时间序列
    t = np.arange(500)
    # 生成正弦信号
    signal = np.sin(t)

    # 断言传递origin作为关键字参数会引发TypeError异常
    with pytest.raises(TypeError):
        plt.specgram(signal, origin='lower')


@image_comparison(
    ["psd_freqs.png", "csd_freqs.png", "psd_noise.png", "csd_noise.png"],
    remove_text=True, tol=0.002)
def test_psd_csd():
    n = 10000
    Fs = 100.

    fstims = [[Fs/4, Fs/5, Fs/11], [Fs/4.7, Fs/5.6, Fs/11.9]]
    NFFT_freqs = int(1000 * Fs / np.min(fstims))
    x = np.arange(0, n, 1/Fs)
    ys_freqs = np.sin(2 * np.pi * np.multiply.outer(fstims, x)).sum(axis=1)

    NFFT_noise = int(1000 * Fs / 11)
    np.random.seed(0)
    # 生成长度为 n 的标准正态分布随机数列表和长度为 n 的 [0, 1) 均匀分布随机数列表，存放在 ys_noise 中
    ys_noise = [np.random.standard_normal(n), np.random.rand(n)]
    
    # 定义包含三个字典的列表 all_kwargs，每个字典包含不同的参数设置
    all_kwargs = [{"sides": "default"},
                  {"sides": "onesided", "return_line": False},
                  {"sides": "twosided", "return_line": True}]
    
    # 遍历 ys_freqs 和 ys_noise，分别执行以下循环
    for ys, NFFT in [(ys_freqs, NFFT_freqs), (ys_noise, NFFT_noise)]:
        # 计算重叠区域长度
        noverlap = NFFT // 2
        # 计算应该填充的数据点数，即大于等于 NFFT 的最小 2 的整数次幂
        pad_to = int(2 ** np.ceil(np.log2(NFFT)))
        
        # 使用 plt.figure().subplots(3) 创建包含 3 个子图的图像，并依次对每个子图进行以下操作
        for ax, kwargs in zip(plt.figure().subplots(3), all_kwargs):
            # 绘制功率谱密度图，将 ys 列表中的所有数据连接起来作为输入
            ret = ax.psd(np.concatenate(ys), NFFT=NFFT, Fs=Fs,
                         noverlap=noverlap, pad_to=pad_to, **kwargs)
            # 断言返回值的长度为 2 或者 3（根据 return_line 参数的有无）
            assert len(ret) == 2 + kwargs.get("return_line", False)
            # 设置 x 轴和 y 轴的标签为空
            ax.set(xlabel="", ylabel="")
        
        # 使用 plt.figure().subplots(3) 再次创建包含 3 个子图的图像，并依次对每个子图进行以下操作
        for ax, kwargs in zip(plt.figure().subplots(3), all_kwargs):
            # 绘制交叉功率谱密度图，将 ys 列表中的所有数据作为输入
            ret = ax.csd(*ys, NFFT=NFFT, Fs=Fs,
                         noverlap=noverlap, pad_to=pad_to, **kwargs)
            # 断言返回值的长度为 2 或者 3（根据 return_line 参数的有无）
            assert len(ret) == 2 + kwargs.get("return_line", False)
            # 设置 x 轴的标签为空，y 轴的标签为 ""
            ax.set(xlabel="", ylabel="")
@image_comparison(
    ["magnitude_spectrum_freqs_linear.png",
     "magnitude_spectrum_freqs_dB.png",
     "angle_spectrum_freqs.png",
     "phase_spectrum_freqs.png",
     "magnitude_spectrum_noise_linear.png",
     "magnitude_spectrum_noise_dB.png",
     "angle_spectrum_noise.png",
     "phase_spectrum_noise.png"],
    remove_text=True)
# 定义用于测试频谱绘图的函数装饰器，比较生成的图像与参考图像

def test_spectrum():
    # 设置数据长度和采样频率
    n = 10000
    Fs = 100.

    # 定义频率刺激
    fstims1 = [Fs/4, Fs/5, Fs/11]
    # 计算 FFT 的长度
    NFFT = int(1000 * Fs / min(fstims1))
    pad_to = int(2 ** np.ceil(np.log2(NFFT)))

    # 生成时间序列
    x = np.arange(0, n, 1/Fs)
    # 生成频率刺激信号
    y_freqs = ((np.sin(2 * np.pi * np.outer(x, fstims1)) * 10**np.arange(3))
               .sum(axis=1))
    
    # 生成随机噪声信号
    np.random.seed(0)
    y_noise = np.hstack([np.random.standard_normal(n), np.random.rand(n)]) - .5

    # 设置频谱绘图的选项
    all_sides = ["default", "onesided", "twosided"]
    kwargs = {"Fs": Fs, "pad_to": pad_to}

    # 对频率信号和噪声分别绘制不同类型的频谱图
    for y in [y_freqs, y_noise]:
        # 绘制幅度谱
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            spec, freqs, line = ax.magnitude_spectrum(y, sides=sides, **kwargs)
            ax.set(xlabel="", ylabel="")
        
        # 绘制幅度谱（单位 dB）
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            spec, freqs, line = ax.magnitude_spectrum(y, sides=sides, **kwargs,
                                                      scale="dB")
            ax.set(xlabel="", ylabel="")
        
        # 绘制相位谱
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            spec, freqs, line = ax.angle_spectrum(y, sides=sides, **kwargs)
            ax.set(xlabel="", ylabel="")
        
        # 绘制相位角谱
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            spec, freqs, line = ax.phase_spectrum(y, sides=sides, **kwargs)
            ax.set(xlabel="", ylabel="")


def test_psd_csd_edge_cases():
    # 处理特殊情况，如输入数据全为零或逆序
    axs = plt.figure().subplots(2)
    for ax in axs:
        ax.yaxis.set(inverted=True)
    with np.errstate(divide="ignore"):
        axs[0].psd(np.zeros(5))  # 绘制功率谱密度图
        axs[1].csd(np.zeros(5), np.zeros(5))  # 绘制交叉功率谱密度图


@check_figures_equal(extensions=['png'])
# 装饰器，检查两个图形对象是否相等，输出为 PNG 格式

def test_twin_remove(fig_test, fig_ref):
    # 创建测试图形和参考图形的子图对象
    ax_test = fig_test.add_subplot()
    ax_twinx = ax_test.twinx()
    ax_twiny = ax_test.twiny()
    
    # 移除双轴对象
    ax_twinx.remove()
    ax_twiny.remove()

    ax_ref = fig_ref.add_subplot()
    # 手动设置参考图像的刻度以匹配测试图像，理想情况下，还应恢复调用“remove()”时进行的刻度更改
    ax_ref.xaxis.tick_bottom()
    ax_ref.yaxis.tick_left()


@image_comparison(['twin_spines.png'], remove_text=True,
                  tol=0.022 if platform.machine() == 'arm64' else 0)
# 比较生成的图像与参考图像，输出为 PNG 格式，可选容差

def test_twin_spines():
    # 定义函数，使子图的脊柱透明
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        ax.spines[:].set_visible(False)

    # 创建主图及其两个双轴对象
    fig = plt.figure(figsize=(4, 3))
    fig.subplots_adjust(right=0.75)

    host = fig.add_subplot()
    par1 = host.twinx()
    par2 = host.twinx()
    # 将 par2 的右边框位移调整。ticks 和 label 已经通过上面的 twinx 放置在右边。
    par2.spines.right.set_position(("axes", 1.2))
    # 由于被 twinx 创建，par2 的框架已关闭，因此其分离的边框线是不可见的。
    # 首先激活框架，但是使补丁和边框不可见。
    make_patch_spines_invisible(par2)
    # 其次，显示右边框线。
    par2.spines.right.set_visible(True)

    # 使用 host 创建一条蓝色线 p1
    p1, = host.plot([0, 1, 2], [0, 1, 2], "b-")
    # 使用 par1 创建一条红色线 p2
    p2, = par1.plot([0, 1, 2], [0, 3, 2], "r-")
    # 使用 par2 创建一条绿色线 p3
    p3, = par2.plot([0, 1, 2], [50, 30, 15], "g-")

    # 设置 host 的 x 轴范围为 0 到 2
    host.set_xlim(0, 2)
    # 设置 host 的 y 轴范围为 0 到 2
    host.set_ylim(0, 2)
    # 设置 par1 的 y 轴范围为 0 到 4
    par1.set_ylim(0, 4)
    # 设置 par2 的 y 轴范围为 1 到 65
    par2.set_ylim(1, 65)

    # 将 host 的 y 轴标签颜色设为 p1 的颜色
    host.yaxis.label.set_color(p1.get_color())
    # 将 par1 的 y 轴标签颜色设为 p2 的颜色
    par1.yaxis.label.set_color(p2.get_color())
    # 将 par2 的 y 轴标签颜色设为 p3 的颜色
    par2.yaxis.label.set_color(p3.get_color())

    # 设置 tick 参数，包括大小为 4，宽度为 1.5
    tkw = dict(size=4, width=1.5)
    # 设置 host 的 y 轴刻度颜色为 p1 的颜色，并应用 tkw 的其他参数
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    # 设置 par1 的 y 轴刻度颜色为 p2 的颜色，并应用 tkw 的其他参数
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    # 设置 par2 的 y 轴刻度颜色为 p3 的颜色，并应用 tkw 的其他参数
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    # 设置 host 的 x 轴刻度，应用 tkw 的参数
    host.tick_params(axis='x', **tkw)
# 使用 image_comparison 装饰器比较两个图像是否相等
@image_comparison(['twin_spines_on_top.png', 'twin_spines_on_top.png'],
                  remove_text=True)
def test_twin_spines_on_top():
    # 设置 matplotlib 的参数，设置轴线宽度和线条宽度
    matplotlib.rcParams['axes.linewidth'] = 48.0
    matplotlib.rcParams['lines.linewidth'] = 48.0

    # 创建一个图形对象
    fig = plt.figure()
    # 在图形对象中添加一个子图
    ax1 = fig.add_subplot(1, 1, 1)

    # 创建一个包含数据的数组
    data = np.array([[1000, 1100, 1200, 1250],
                     [310, 301, 360, 400]])

    # 在同一个图上创建第二个 y 轴
    ax2 = ax1.twinx()

    # 绘制第一个数据集的线条和填充区域
    ax1.plot(data[0], data[1]/1E3, color='#BEAED4')
    ax1.fill_between(data[0], data[1]/1E3, color='#BEAED4', alpha=.8)

    # 绘制第二个数据集的线条和填充区域
    ax2.plot(data[0], data[1]/1E3, color='#7FC97F')
    ax2.fill_between(data[0], data[1]/1E3, color='#7FC97F', alpha=.5)

    # 重用上面的测试用例进行带标签数据的测试
    data = {"i": data[0], "j": data[1]/1E3}
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax1.plot("i", "j", color='#BEAED4', data=data)
    ax1.fill_between("i", "j", color='#BEAED4', alpha=.8, data=data)
    ax2.plot("i", "j", color='#7FC97F', data=data)
    ax2.fill_between("i", "j", color='#7FC97F', alpha=.5, data=data)


# 使用 pytest 的参数化装饰器定义多个测试参数
@pytest.mark.parametrize("grid_which, major_visible, minor_visible", [
    ("both", True, True),
    ("major", True, False),
    ("minor", False, True),
])
def test_rcparam_grid_minor(grid_which, major_visible, minor_visible):
    # 更新 matplotlib 的参数，设置网格可见性和类型
    mpl.rcParams.update({"axes.grid": True, "axes.grid.which": grid_which})
    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()
    fig.canvas.draw()
    # 断言主要刻度线和次要刻度线的网格可见性
    assert all(tick.gridline.get_visible() == major_visible
               for tick in ax.xaxis.majorTicks)
    assert all(tick.gridline.get_visible() == minor_visible
               for tick in ax.xaxis.minorTicks)


def test_grid():
    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()
    # 启用网格线
    ax.grid()
    fig.canvas.draw()
    # 断言主要刻度线的网格可见性
    assert ax.xaxis.majorTicks[0].gridline.get_visible()
    # 禁用网格线
    ax.grid(visible=False)
    fig.canvas.draw()
    # 断言主要刻度线的网格不可见
    assert not ax.xaxis.majorTicks[0].gridline.get_visible()
    # 再次启用网格线
    ax.grid(visible=True)
    fig.canvas.draw()
    # 断言主要刻度线的网格可见
    assert ax.xaxis.majorTicks[0].gridline.get_visible()
    # 再次启用网格线
    ax.grid()
    fig.canvas.draw()
    # 断言主要刻度线的网格不可见


def test_reset_grid():
    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()
    # 重置刻度参数，设置主要刻度线的标签大小
    ax.tick_params(reset=True, which='major', labelsize=10)
    # 断言主要刻度线的网格不可见
    assert not ax.xaxis.majorTicks[0].gridline.get_visible()
    # 启用网格线
    ax.grid(color='red')
    # 断言主要刻度线的网格可见
    assert ax.xaxis.majorTicks[0].gridline.get_visible()

    # 在上下文中设置参数，启用网格线
    with plt.rc_context({'axes.grid': True}):
        ax.clear()
        # 重置刻度参数，设置主要刻度线的标签大小
        ax.tick_params(reset=True, which='major', labelsize=10)
        # 断言主要刻度线的网格可见


# 使用 check_figures_equal 装饰器比较两个图形是否相等
@check_figures_equal(extensions=['png'])
def test_reset_ticks(fig_test, fig_ref):
    # 遍历包含 fig_ref 和 fig_test 的列表
    for fig in [fig_ref, fig_test]:
        # 为图形添加子图
        ax = fig.add_subplot()
        # 打开子图的网格线
        ax.grid(True)
        # 设置刻度参数
        ax.tick_params(
            direction='in', length=10, width=5, color='C0', pad=12,
            labelsize=14, labelcolor='C1', labelrotation=45,
            grid_color='C2', grid_alpha=0.8, grid_linewidth=3,
            grid_linestyle='--')
        # 在不渲染的情况下绘制图形
        fig.draw_without_rendering()

    # 在更改任何刻度设置后，reset_ticks 将重新创建刻度，应该与不重置刻度看起来相同
    for ax in fig_test.axes:
        # 重置 x 轴刻度
        ax.xaxis.reset_ticks()
        # 重置 y 轴刻度
        ax.yaxis.reset_ticks()
# 定义一个测试函数，用于验证垂直线的限制
def test_vline_limit():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 获取当前轴对象
    ax = fig.gca()
    # 在轴上绘制一条垂直线，位置在 x=0.5 处
    ax.axvline(0.5)
    # 绘制一条折线图，包含点 (-0.1, -0.1), (0, 0), (0.2, 0.2), (0.1, 0.1)
    ax.plot([-0.1, 0, 0.2, 0.1])
    # 断言当前轴的 y 轴限制在 (-0.1, 0.2) 之间
    assert_allclose(ax.get_ylim(), (-.1, .2))


# 使用参数化测试装饰器进行多组测试参数化
@pytest.mark.parametrize('fv, fh, args', [[plt.axvline, plt.axhline, (1,)],
                                          [plt.axvspan, plt.axhspan, (1, 1)]])
def test_axline_minmax(fv, fh, args):
    # 创建一个无效的限制值 bad_lim
    bad_lim = matplotlib.dates.num2date(1)
    # 检查垂直函数
    with pytest.raises(ValueError, match='ymin must be a single scalar value'):
        # 断言调用 fv 函数时，传入 ymin 参数为 bad_lim，ymax 参数为 1 会触发 ValueError 异常
        fv(*args, ymin=bad_lim, ymax=1)
    with pytest.raises(ValueError, match='ymax must be a single scalar value'):
        # 断言调用 fv 函数时，传入 ymin 参数为 1，ymax 参数为 bad_lim 会触发 ValueError 异常
        fv(*args, ymin=1, ymax=bad_lim)
    # 检查水平函数
    with pytest.raises(ValueError, match='xmin must be a single scalar value'):
        # 断言调用 fh 函数时，传入 xmin 参数为 bad_lim，xmax 参数为 1 会触发 ValueError 异常
        fh(*args, xmin=bad_lim, xmax=1)
    with pytest.raises(ValueError, match='xmax must be a single scalar value'):
        # 断言调用 fh 函数时，传入 xmin 参数为 1，xmax 参数为 bad_lim 会触发 ValueError 异常
        fh(*args, xmin=1, xmax=bad_lim)


def test_empty_shared_subplots():
    # 创建一个包含两个子图的图形对象，这两个子图共享相同的 x 和 y 轴
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # 在第一个子图上绘制一条线，数据点为 (1, 2), (2, 4), (3, 6)
    axs[0].plot([1, 2, 3], [2, 4, 6])
    # 获取第二个子图的 x 轴和 y 轴的限制
    x0, x1 = axs[1].get_xlim()
    y0, y1 = axs[1].get_ylim()
    # 断言第二个子图的 x 轴限制在 [1, 3] 之间
    assert x0 <= 1
    assert x1 >= 3
    # 断言第二个子图的 y 轴限制在 [2, 6] 之间
    assert y0 <= 2
    assert y1 >= 6


def test_shared_with_aspect_1():
    # 允许共享一个轴
    for adjustable in ['box', 'datalim']:
        # 创建两行子图，共享 x 轴
        fig, axs = plt.subplots(nrows=2, sharex=True)
        # 设置第一个子图的纵横比为 2，调整方式为 adjustable，共享 y 轴
        axs[0].set_aspect(2, adjustable=adjustable, share=True)
        # 断言第二个子图的纵横比为 2
        assert axs[1].get_aspect() == 2
        # 断言第二个子图的可调整属性为 adjustable
        assert axs[1].get_adjustable() == adjustable

        # 创建两行子图，共享 x 轴
        fig, axs = plt.subplots(nrows=2, sharex=True)
        # 设置第一个子图的纵横比为 2，调整方式为 adjustable
        axs[0].set_aspect(2, adjustable=adjustable)
        # 断言第二个子图的纵横比为 'auto'
        assert axs[1].get_aspect() == 'auto'


def test_shared_with_aspect_2():
    # 仅用 'box' 方式共享两个轴
    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)
    # 设置第一个子图的纵横比为 2，共享两个轴
    axs[0].set_aspect(2, share=True)
    # 在第一个子图上绘制数据点 (1, 3), (2, 4)
    axs[0].plot([1, 2], [3, 4])
    # 在第二个子图上绘制数据点 (3, 1), (4, 2)
    axs[1].plot([3, 4], [1, 2])
    # 绘制图形，触发 apply_aspect() 函数
    plt.draw()
    # 断言两个子图的 x 轴限制相同
    assert axs[0].get_xlim() == axs[1].get_xlim()
    # 断言两个子图的 y 轴限制相同
    assert axs[0].get_ylim() == axs[1].get_ylim()


def test_shared_with_aspect_3():
    # 不同的纵横比:
    # 针对两种可调整的参数（'box' 和 'datalim'），分别创建图形和轴子图
    for adjustable in ['box', 'datalim']:
        # 创建一个包含两个子图的图形对象
        fig, axs = plt.subplots(nrows=2, sharey=True)
        
        # 设置第一个子图的纵横比，并指定可调整的参数
        axs[0].set_aspect(2, adjustable=adjustable)
        
        # 设置第二个子图的纵横比，并指定可调整的参数
        axs[1].set_aspect(0.5, adjustable=adjustable)
        
        # 在第一个子图上绘制数据
        axs[0].plot([1, 2], [3, 4])
        
        # 在第二个子图上绘制数据
        axs[1].plot([3, 4], [1, 2])
        
        # 执行绘图的绘制操作，触发 apply_aspect() 方法
        plt.draw()  # Trigger apply_aspect().
        
        # 断言两个子图的 x 轴的范围不相等
        assert axs[0].get_xlim() != axs[1].get_xlim()
        
        # 断言两个子图的 y 轴的范围相等
        assert axs[0].get_ylim() == axs[1].get_ylim()
        
        # 计算整个图形的纵横比
        fig_aspect = fig.bbox_inches.height / fig.bbox_inches.width
        
        # 遍历每个子图的轴对象
        for ax in axs:
            # 获取当前轴在图形中的位置和大小
            p = ax.get_position()
            
            # 计算轴子图的盒子纵横比
            box_aspect = p.height / p.width
            
            # 计算轴子图的视图限制的纵横比
            lim_aspect = ax.viewLim.height / ax.viewLim.width
            
            # 计算预期的轴子图纵横比
            expected = fig_aspect * box_aspect / lim_aspect
            
            # 断言当前轴子图的纵横比与预期值四舍五入后相等
            assert round(expected, 4) == round(ax.get_aspect(), 4)
def test_shared_aspect_error():
    # 创建一个包含两个子图的图形对象，并共享相同的坐标轴
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    # 设置第一个子图的坐标轴比例为相等
    axes[0].axis("equal")
    # 测试是否会抛出 RuntimeError 异常，并检查异常消息中是否包含指定的字符串
    with pytest.raises(RuntimeError, match=r"set_aspect\(..., adjustable="):
        fig.draw_without_rendering()


@pytest.mark.parametrize('err, args, kwargs, match',
                         # 使用参数化测试，分别测试不同的异常情况
                         ((TypeError, (1, 2), {},
                           r"axis\(\) takes from 0 to 1 positional arguments "
                           "but 2 were given"),
                          (ValueError, ('foo', ), {},
                           "Unrecognized string 'foo' to axis; try 'on' or "
                           "'off'"),
                          (TypeError, ([1, 2], ), {},
                           "The first argument to axis*"),
                          (TypeError, tuple(), {'foo': None},
                           r"axis\(\) got an unexpected keyword argument "
                           "'foo'"),
                          ))
def test_axis_errors(err, args, kwargs, match):
    # 每次测试中，使用 pytest.raises 检查是否抛出指定类型的异常，并验证异常消息
    with pytest.raises(err, match=match):
        plt.axis(*args, **kwargs)


def test_axis_method_errors():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 测试各种可能的错误情况，并验证异常消息
    with pytest.raises(ValueError, match="unknown value for which: 'foo'"):
        ax.get_xaxis_transform('foo')
    with pytest.raises(ValueError, match="unknown value for which: 'foo'"):
        ax.get_yaxis_transform('foo')
    with pytest.raises(TypeError, match="Cannot supply both positional and"):
        ax.set_prop_cycle('foo', label='bar')
    with pytest.raises(ValueError, match="argument must be among"):
        ax.set_anchor('foo')
    with pytest.raises(ValueError, match="scilimits must be a sequence"):
        ax.ticklabel_format(scilimits=1)
    with pytest.raises(TypeError, match="Specifying 'loc' is disallowed"):
        ax.set_xlabel('foo', loc='left', x=1)
    with pytest.raises(TypeError, match="Specifying 'loc' is disallowed"):
        ax.set_ylabel('foo', loc='top', y=1)
    with pytest.raises(TypeError, match="Cannot pass both 'left'"):
        ax.set_xlim(left=0, xmin=1)
    with pytest.raises(TypeError, match="Cannot pass both 'right'"):
        ax.set_xlim(right=0, xmax=1)
    with pytest.raises(TypeError, match="Cannot pass both 'bottom'"):
        ax.set_ylim(bottom=0, ymin=1)
    with pytest.raises(TypeError, match="Cannot pass both 'top'"):
        ax.set_ylim(top=0, ymax=1)


@pytest.mark.parametrize('twin', ('x', 'y'))
def test_twin_with_aspect(twin):
    # 创建一个包含单个子图的图形对象
    fig, ax = plt.subplots()
    # 测试在指定方向上创建镜像坐标轴
    ax_twin = getattr(ax, f'twin{twin}')()
    # 设置主坐标轴的纵横比为5
    ax.set_aspect(5)
    # 设置镜像坐标轴的纵横比为2
    ax_twin.set_aspect(2)
    # 断言主坐标轴和镜像坐标轴的包围框范围数组是否相等
    assert_array_equal(ax.bbox.extents,
                       ax_twin.bbox.extents)


def test_relim_visible_only():
    # 定义两组数据点
    x1 = (0., 10.)
    y1 = (0., 10.)
    x2 = (-10., 20.)
    y2 = (-10., 30.)

    # 创建一个新的图形对象
    fig = matplotlib.figure.Figure()
    # 添加一个子图到图形对象中
    ax = fig.add_subplot()
    # 绘制第一组数据点的曲线，并验证坐标轴的范围
    ax.plot(x1, y1)
    assert ax.get_xlim() == x1
    assert ax.get_ylim() == y1
    # 绘制第二组数据点的曲线，并验证坐标轴的范围
    line, = ax.plot(x2, y2)
    assert ax.get_xlim() == x2
    # 断言当前绘图的 y 轴限制是否与 y2 相等
    assert ax.get_ylim() == y2
    
    # 将线条设置为不可见状态
    line.set_visible(False)
    
    # 再次断言当前绘图的 x 轴限制是否与 x2 相等
    assert ax.get_xlim() == x2
    
    # 再次断言当前绘图的 y 轴限制是否与 y2 相等
    assert ax.get_ylim() == y2
    
    # 重新计算轴的数据限制，只考虑可见部分
    ax.relim(visible_only=True)
    
    # 自动调整视图以适应数据范围
    ax.autoscale_view()
    
    # 最后断言当前绘图的 x 轴限制是否与 x1 相等
    assert ax.get_xlim() == x1
    
    # 最后断言当前绘图的 y 轴限制是否与 y1 相等
    assert ax.get_ylim() == y1
def test_text_labelsize():
    """
    tests for issue #1172
    """
    # 创建一个新的图形对象
    fig = plt.figure()
    # 获取当前图形的坐标轴
    ax = fig.gca()
    # 设置坐标轴刻度标签的大小为'large'
    ax.tick_params(labelsize='large')
    # 设置坐标轴刻度的方向为'out'
    ax.tick_params(direction='out')


# 注意：`pie` 图像测试受到 Numpy 2.0 更改提升（NEP 50）的影响。
# 虽然更改只是轻微的，但引入了容差。
# 当 Numpy 2.0 是最低支持的 numpy 版本并且重新生成图像时，这些容差可能会消失。

@image_comparison(['pie_default.png'], tol=0.01)
def test_pie_default():
    # 饼图的扇形将按逆时针顺序排序和绘制
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0, 0.1, 0, 0)  # 仅“爆炸”第二个扇形（即'Hogs'）
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    # 绘制饼图
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)


@image_comparison(['pie_linewidth_0', 'pie_linewidth_0', 'pie_linewidth_0'],
                  extensions=['png'], style='mpl20', tol=0.01)
def test_pie_linewidth_0():
    # 饼图的扇形将按逆时针顺序排序和绘制
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0, 0.1, 0, 0)  # 仅“爆炸”第二个扇形（即'Hogs'）

    # 绘制饼图，设置扇形边框宽度为0
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            wedgeprops={'linewidth': 0})
    # 设置纵横比相等，以便绘制的饼图为圆形
    plt.axis('equal')

    # 重用上面的测试用例进行带标签数据的测试
    data = {"l": labels, "s": sizes, "c": colors, "ex": explode}
    fig = plt.figure()
    ax = fig.gca()
    ax.pie("s", explode="ex", labels="l", colors="c",
           autopct='%1.1f%%', shadow=True, startangle=90,
           wedgeprops={'linewidth': 0}, data=data)
    ax.axis('equal')

    # 再次测试 pyplot 函数，这些函数也应该能够使用数据关键字参数调用
    plt.figure()
    plt.pie("s", explode="ex", labels="l", colors="c",
            autopct='%1.1f%%', shadow=True, startangle=90,
            wedgeprops={'linewidth': 0}, data=data)
    plt.axis('equal')


@image_comparison(['pie_center_radius.png'], style='mpl20', tol=0.01)
def test_pie_center_radius():
    # 饼图的扇形将按逆时针顺序排序和绘制
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0, 0.1, 0, 0)  # 仅“爆炸”第二个扇形（即'Hogs'）

    # 绘制饼图，设置中心和半径
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            wedgeprops={'linewidth': 0}, center=(1, 2), radius=1.5)
    # 在图表中添加注释文本，标注为“Center point”，箭头指向目标坐标 (1, 2)，注释文本位置在 (1, 1.3)
    plt.annotate("Center point", xy=(1, 2), xytext=(1, 1.3),
                 # 定义箭头样式为箭头形状，连接样式为弧形
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3"),
                 # 定义注释框的样式为方形，背景颜色为浅灰色
                 bbox=dict(boxstyle="square", facecolor="lightgrey"))
    
    # 设置坐标轴的纵横比为相等，以便绘制的饼图形状为圆形
    plt.axis('equal')
@image_comparison(['pie_linewidth_2.png'], style='mpl20', tol=0.01)
def test_pie_linewidth_2():
    # The slices will be ordered and plotted counter-clockwise.
    # 定义饼图每个部分的标签
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    # 定义每个部分的大小（百分比）
    sizes = [15, 30, 45, 10]
    # 定义每个部分的颜色
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    # 定义哪些部分需要爆炸效果（这里只有第二个部分 'Hogs'）
    explode = (0, 0.1, 0, 0)

    # 绘制饼图
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            wedgeprops={'linewidth': 2})
    # 设置坐标轴的比例为相等，以便绘制的饼图是一个圆形
    plt.axis('equal')


@image_comparison(['pie_ccw_true.png'], style='mpl20', tol=0.01)
def test_pie_ccw_true():
    # The slices will be ordered and plotted counter-clockwise.
    # 定义饼图每个部分的标签
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    # 定义每个部分的大小（百分比）
    sizes = [15, 30, 45, 10]
    # 定义每个部分的颜色
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    # 定义哪些部分需要爆炸效果（这里只有第二个部分 'Hogs'）
    explode = (0, 0.1, 0, 0)

    # 绘制饼图，设置顺时针方向为True
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            counterclock=True)
    # 设置坐标轴的比例为相等，以便绘制的饼图是一个圆形
    plt.axis('equal')


@image_comparison(['pie_frame_grid.png'], style='mpl20', tol=0.002)
def test_pie_frame_grid():
    # The slices will be ordered and plotted counter-clockwise.
    # 定义饼图每个部分的标签
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    # 定义每个部分的大小（百分比）
    sizes = [15, 30, 45, 10]
    # 定义每个部分的颜色
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    # 定义哪些部分需要爆炸效果（这里只有第二个部分 'Hogs'）
    explode = (0, 0.1, 0, 0)

    # 绘制饼图，同时设置多个子图，每个子图有不同的参数
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            wedgeprops={'linewidth': 0},  # 饼图边界线宽度设为0
            frame=True, center=(2, 2))  # 显示框架并设定中心点位置

    plt.pie(sizes[::-1], explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            wedgeprops={'linewidth': 0},  # 饼图边界线宽度设为0
            frame=True, center=(5, 2))  # 显示框架并设定中心点位置

    plt.pie(sizes, explode=explode[::-1], labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            wedgeprops={'linewidth': 0},  # 饼图边界线宽度设为0
            frame=True, center=(3, 5))  # 显示框架并设定中心点位置
    # 设置坐标轴的比例为相等，以便绘制的饼图是一个圆形
    plt.axis('equal')


@image_comparison(['pie_rotatelabels_true.png'], style='mpl20', tol=0.009)
def test_pie_rotatelabels_true():
    # The slices will be ordered and plotted counter-clockwise.
    # 定义饼图每个部分的标签
    labels = 'Hogwarts', 'Frogs', 'Dogs', 'Logs'
    # 定义每个部分的大小（百分比）
    sizes = [15, 30, 45, 10]
    # 定义每个部分的颜色
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    # 定义哪些部分需要爆炸效果（这里只有第二个部分 'Hogs'）
    explode = (0, 0.1, 0, 0)

    # 绘制饼图，设置标签文本沿半径方向旋转
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            rotatelabels=True)
    # 设置坐标轴的比例为相等，以便绘制的饼图是一个圆形
    plt.axis('equal')
@image_comparison(['pie_no_label.png'], tol=0.01)
def test_pie_nolabel_but_legend():
    # 设置饼图的标签和对应的数据
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0, 0.1, 0, 0)  # 仅爆炸第二个扇形（即 'Hogs'）
    # 绘制饼图，并设置各种属性如百分比格式、阴影、起始角度等
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90, labeldistance=None,
            rotatelabels=True)
    # 设置坐标轴相等
    plt.axis('equal')
    # 设置纵坐标轴范围
    plt.ylim(-1.2, 1.2)
    # 添加图例
    plt.legend()


@image_comparison(['pie_shadow.png'], style='mpl20', tol=0.002)
def test_pie_shadow():
    # 测试阴影功能，也用于测试阴影参数的使用
    sizes = [15, 30, 45, 10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0, 0.1, 0, 0)  # 仅爆炸第二个扇形

    # 创建2x2的子图，并在每个子图上绘制饼图，设置不同的阴影属性
    _, axes = plt.subplots(2, 2)
    axes[0][0].pie(sizes, explode=explode, colors=colors,
                   shadow=True, startangle=90,
                   wedgeprops={'linewidth': 0})

    axes[0][1].pie(sizes, explode=explode, colors=colors,
                   shadow=False, startangle=90,
                   wedgeprops={'linewidth': 0})

    axes[1][0].pie(sizes, explode=explode, colors=colors,
                   shadow={'ox': -0.05, 'oy': -0.05, 'shade': 0.9, 'edgecolor': 'none'},
                   startangle=90, wedgeprops={'linewidth': 0})

    axes[1][1].pie(sizes, explode=explode, colors=colors,
                   shadow={'ox': 0.05, 'linewidth': 2, 'shade': 0.2},
                   startangle=90, wedgeprops={'linewidth': 0})


def test_pie_textprops():
    # 测试文本属性的设置
    data = [23, 34, 45]
    labels = ["Long name 1", "Long name 2", "Long name 3"]

    textprops = dict(horizontalalignment="center",
                     verticalalignment="top",
                     rotation=90,
                     rotation_mode="anchor",
                     size=12, color="red")

    # 绘制饼图，并设置文本属性
    _, texts, autopct = plt.gca().pie(data, labels=labels, autopct='%.2f',
                                      textprops=textprops)
    # 验证文本属性是否正确应用到绘制的文本上
    for labels in [texts, autopct]:
        for tx in labels:
            assert tx.get_ha() == textprops["horizontalalignment"]
            assert tx.get_va() == textprops["verticalalignment"]
            assert tx.get_rotation() == textprops["rotation"]
            assert tx.get_rotation_mode() == textprops["rotation_mode"]
            assert tx.get_size() == textprops["size"]
            assert tx.get_color() == textprops["color"]


def test_pie_get_negative_values():
    # 测试当传入负值时是否会引发 ValueError
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.pie([5, 5, -3], explode=[0, .1, .2])


def test_normalize_kwarg_pie():
    # 测试饼图的 normalize 参数
    fig, ax = plt.subplots()
    x = [0.3, 0.3, 0.1]
    t1 = ax.pie(x=x, normalize=True)
    assert abs(t1[0][-1].theta2 - 360.) < 1e-3
    t2 = ax.pie(x=x, normalize=False)
    assert abs(t2[0][-1].theta2 - 360.) > 1e-3
@check_figures_equal()
def test_pie_hatch_single(fig_test, fig_ref):
    # 定义数据和阴影样式
    x = [0.3, 0.3, 0.1]
    hatch = '+'
    # 在测试图中创建饼图，应用阴影样式
    fig_test.subplots().pie(x, hatch=hatch)
    # 在参考图中创建饼图，无阴影样式
    wedges, _ = fig_ref.subplots().pie(x)
    # 设置测试图中饼图的阴影样式与参考图相同
    [w.set_hatch(hatch) for w in wedges]


@check_figures_equal()
def test_pie_hatch_multi(fig_test, fig_ref):
    # 定义数据和多个阴影样式
    x = [0.3, 0.3, 0.1]
    hatch = ['/', '+', '.']
    # 在测试图中创建饼图，应用多个阴影样式
    fig_test.subplots().pie(x, hatch=hatch)
    # 在参考图中创建饼图，无阴影样式
    wedges, _ = fig_ref.subplots().pie(x)
    # 设置测试图中饼图的每个部分的阴影样式与参考图相同
    [w.set_hatch(hp) for w, hp in zip(wedges, hatch)]


@image_comparison(['set_get_ticklabels.png'],
                  tol=0.025 if platform.machine() == 'arm64' else 0)
def test_set_get_ticklabels():
    # 测试问题编号 2246
    fig, ax = plt.subplots(2)
    ha = ['normal', 'set_x/yticklabels']

    ax[0].plot(np.arange(10))
    ax[0].set_title(ha[0])

    ax[1].plot(np.arange(10))
    ax[1].set_title(ha[1])

    # 在第一个图中设置刻度标签
    ax[0].set_xticks(range(10))
    ax[0].set_yticks(range(10))
    ax[0].set_xticklabels(['a', 'b', 'c', 'd'] + 6 * [''])
    ax[0].set_yticklabels(['11', '12', '13', '14'] + 6 * [''])

    # 在第二个图中设置刻度标签，保证两个图的标签相同
    # 通过 get_ticklabels 返回值设置为 ticklabels 参数
    ax[1].set_xticks(ax[0].get_xticks())
    ax[1].set_yticks(ax[0].get_yticks())
    ax[1].set_xticklabels(ax[0].get_xticklabels())
    ax[1].set_yticklabels(ax[0].get_yticklabels())


def test_set_ticks_kwargs_raise_error_without_labels():
    """
    当 labels=None 且传入任何关键字参数时，axis.set_ticks() 会引发 ValueError。
    """
    fig, ax = plt.subplots()
    ticks = [1, 2, 3]
    with pytest.raises(ValueError, match="Incorrect use of keyword argument 'alpha'"):
        ax.xaxis.set_ticks(ticks, alpha=0.5)


@check_figures_equal(extensions=["png"])
def test_set_ticks_with_labels(fig_test, fig_ref):
    """
    测试这两个方法是否相同：

        set_xticks(ticks); set_xticklabels(labels, **kwargs)
        set_xticks(ticks, labels, **kwargs)

    """
    ax = fig_ref.subplots()
    ax.set_xticks([1, 2, 4, 6])
    ax.set_xticklabels(['a', 'b', 'c', 'd'], fontweight='bold')
    ax.set_yticks([1, 3, 5])
    ax.set_yticks([2, 4], minor=True)
    ax.set_yticklabels(['A', 'B'], minor=True)

    ax = fig_test.subplots()
    ax.set_xticks([1, 2, 4, 6], ['a', 'b', 'c', 'd'], fontweight='bold')
    ax.set_yticks([1, 3, 5])
    ax.set_yticks([2, 4], ['A', 'B'], minor=True)


def test_xticks_bad_args():
    ax = plt.figure().add_subplot()
    with pytest.raises(TypeError, match='must be a sequence'):
        ax.set_xticks([2, 9], 3.1)
    with pytest.raises(ValueError, match='must be 1D'):
        plt.xticks(np.arange(4).reshape((-1, 1)))
    with pytest.raises(ValueError, match='must be 1D'):
        plt.xticks(np.arange(4).reshape((1, -1)))
    with pytest.raises(ValueError, match='must be 1D'):
        plt.xticks(np.arange(4).reshape((-1, 1)), labels=range(4))
    # 使用 pytest 的断言语法，期望抛出 ValueError 异常并且异常信息匹配 'must be 1D'
    with pytest.raises(ValueError, match='must be 1D'):
        # 设置 x 轴刻度的位置为一个二维数组，此处会触发异常
        plt.xticks(np.arange(4).reshape((1, -1)), labels=range(4))
# 测试函数，用于验证在子采样时的刻度标签处理
def test_subsampled_ticklabels():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制一条包含10个点的折线图
    ax.plot(np.arange(10))
    # 设置 x 轴刻度点，使其稍微偏移
    ax.xaxis.set_ticks(np.arange(10) + 0.1)
    # 设置定位器参数，将 x 轴刻度数量设置为5
    ax.locator_params(nbins=5)
    # 设置 x 轴刻度标签为字符 "bcdefghijk" 中的字符
    ax.xaxis.set_ticklabels([c for c in "bcdefghijk"])
    # 绘制图形
    plt.draw()
    # 获取当前 x 轴刻度标签的文本内容
    labels = [t.get_text() for t in ax.xaxis.get_ticklabels()]
    # 断言获取到的刻度标签内容与预期相符
    assert labels == ['b', 'd', 'f', 'h', 'j']


# 测试函数，用于验证不匹配的刻度标签处理
def test_mismatched_ticklabels():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制一条包含10个点的折线图
    ax.plot(np.arange(10))
    # 设置 x 轴特定的刻度点，不包含整数值
    ax.xaxis.set_ticks([1.5, 2.5])
    # 使用 pytest 检测是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        # 设置 x 轴的刻度标签为 ['a', 'b', 'c']
        ax.xaxis.set_ticklabels(['a', 'b', 'c'])


# 测试函数，用于验证空刻度标签的固定位置处理
def test_empty_ticks_fixed_loc():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制一个柱状图，包含两个柱子
    ax.bar([1, 2], [1, 2])
    # 设置 x 轴的刻度点为 [1, 2]
    ax.set_xticks([1, 2])
    # 设置 x 轴的刻度标签为空列表，清除所有刻度标签
    ax.set_xticklabels([])


# 使用图像比较装饰器，验证保持刻度可见性的处理
@image_comparison(['retain_tick_visibility.png'])
def test_retain_tick_visibility():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制一条线性图
    plt.plot([0, 1, 2], [0, -1, 4])
    # 设置 y 轴所有刻度标签不可见
    plt.setp(ax.get_yticklabels(), visible=False)
    # 设置 y 轴的刻度参数，长度为0
    ax.tick_params(axis="y", which="both", length=0)


# 测试函数，用于验证警告不足标签数量时的处理
def test_warn_too_few_labels():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 使用 pytest 检测是否会发出 UserWarning 警告
    with pytest.warns(
           UserWarning,
           match=r'set_ticklabels\(\) should only be used with a fixed number'):
        # 设置 x 轴的刻度标签为 ['0', '0.1']
        ax.set_xticklabels(['0', '0.1'])
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 设置 x 轴的刻度点为 [0, 0.5, 1]
    ax.set_xticks([0, 0.5, 1])
    # 使用 pytest 检测是否会引发 ValueError 异常
    with pytest.raises(ValueError,
                       match='The number of FixedLocator locations'):
        # 设置 x 轴的刻度标签为 ['0', '0.1']
        ax.set_xticklabels(['0', '0.1'])


# 测试函数，用于验证刻度标签更新的处理
def test_tick_label_update():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 定义一个虚拟的格式化函数
    def formatter_func(x, pos):
        return "unit value" if x == 1 else ""
    # 设置 x 轴的主要刻度标签格式化函数
    ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter_func))
    # 强制一些 x 轴刻度点超出绘制范围
    ax.set_xticks([-1, 0, 1, 2, 3])
    # 设置 x 轴的绘制范围为 (-0.5, 2.5)
    ax.set_xlim(-0.5, 2.5)
    # 绘制图形
    ax.figure.canvas.draw()
    # 获取当前 x 轴刻度标签的文本内容
    tick_texts = [tick.get_text() for tick in ax.xaxis.get_ticklabels()]
    # 断言获取到的刻度标签内容与预期相符
    assert tick_texts == ["", "", "unit value", "", ""]


# 使用图像比较装饰器，验证 'o' 标记路径的处理
@image_comparison(['o_marker_path_snap.png'], savefig_kwarg={'dpi': 72})
def test_o_marker_path_snap():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 设置轴的边距为 0.1
    ax.margins(.1)
    # 在图上绘制多个 'o' 标记，不同大小
    for ms in range(1, 15):
        ax.plot([1, 2, ], np.ones(2) + ms, 'o', ms=ms)

    # 在图上绘制多个 'o' 标记，大小在范围内线性分布
    for ms in np.linspace(1, 10, 25):
        ax.plot([3, 4, ], np.ones(2) + ms, 'o', ms=ms)


# 测试函数，用于验证所有调用边距的方式
def test_margins():
    # 定义一组数据点
    data = [1, 10]
    xmin = 0.0
    xmax = len(data) - 1.0
    ymin = min(data)
    ymax = max(data)

    # 创建一个新的图形和轴对象
    fig1, ax1 = plt.subplots(1, 1)
    # 绘制数据点的折线图
    ax1.plot(data)
    # 设置轴的边距为 1
    ax1.margins(1)
    # 断言获取到的轴边距与预期相符
    assert ax1.margins() == (1, 1)
    # 断言获取到的 x 轴绘制范围与预期相符
    assert ax1.get_xlim() == (xmin - (xmax - xmin) * 1,
                              xmax + (xmax - xmin) * 1)
    # 断言获取到的 y 轴绘制范围与预期相符
    assert ax1.get_ylim() == (ymin - (ymax - ymin) * 1,
                              ymax + (ymax - ymin) * 1)
    # 创建一个包含一个子图的新图形对象和一个坐标轴对象
    fig2, ax2 = plt.subplots(1, 1)
    # 在坐标轴上绘制数据图形
    ax2.plot(data)
    # 设置坐标轴的边距为指定的数值，单位是数据单位的比例
    ax2.margins(0.5, 2)
    # 断言检查设置的坐标轴边距是否符合预期值
    assert ax2.margins() == (0.5, 2)
    # 断言检查获取的 X 轴的数据范围是否符合预期
    assert ax2.get_xlim() == (xmin - (xmax - xmin) * 0.5,
                              xmax + (xmax - xmin) * 0.5)
    # 断言检查获取的 Y 轴的数据范围是否符合预期
    assert ax2.get_ylim() == (ymin - (ymax - ymin) * 2,
                              ymax + (ymax - ymin) * 2)
    
    # 创建另一个包含一个子图的新图形对象和一个坐标轴对象
    fig3, ax3 = plt.subplots(1, 1)
    # 在坐标轴上绘制数据图形
    ax3.plot(data)
    # 设置坐标轴的边距为指定的数值，分别对 X 轴和 Y 轴进行设置
    ax3.margins(x=-0.2, y=0.5)
    # 断言检查设置的坐标轴边距是否符合预期值
    assert ax3.margins() == (-0.2, 0.5)
    # 断言检查获取的 X 轴的数据范围是否符合预期
    assert ax3.get_xlim() == (xmin - (xmax - xmin) * -0.2,
                              xmax + (xmax - xmin) * -0.2)
    # 断言检查获取的 Y 轴的数据范围是否符合预期
    assert ax3.get_ylim() == (ymin - (ymax - ymin) * 0.5,
                              ymax + (ymax - ymin) * 0.5)
def test_margin_getters():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 向图形中添加一个子图
    ax = fig.add_subplot()
    # 设置子图的边距为 (0.2, 0.3)
    ax.margins(0.2, 0.3)
    # 断言获取子图的 x 轴边距为 0.2
    assert ax.get_xmargin() == 0.2
    # 断言获取子图的 y 轴边距为 0.3
    assert ax.get_ymargin() == 0.3


def test_set_margin_updates_limits():
    # 使用默认样式
    mpl.style.use("default")
    # 创建一个新的图形对象和一个子图
    fig, ax = plt.subplots()
    # 绘制一条线段
    ax.plot([1, 2], [1, 2])
    # 设置 x 轴为对数尺度，并设置 x 轴边距为 0
    ax.set(xscale="log", xmargin=0)
    # 断言获取子图的 x 轴限制为 (1, 2)
    assert ax.get_xlim() == (1, 2)


@pytest.mark.parametrize('err, args, kwargs, match', (
        # 参数化测试：测试各种边距设置错误的情况
        (ValueError, (-1,), {}, r'margin must be greater than -0\.5'),
        (ValueError, (1, -1), {}, r'margin must be greater than -0\.5'),
        (ValueError, tuple(), {'x': -1}, r'margin must be greater than -0\.5'),
        (ValueError, tuple(), {'y': -1}, r'margin must be greater than -0\.5'),
        (TypeError, (1, ), {'x': 1, 'y': 1},
         'Cannot pass both positional and keyword arguments for x and/or y'),
        (TypeError, (1, ), {'x': 1},
         'Cannot pass both positional and keyword arguments for x and/or y'),
        (TypeError, (1, 1, 1), {}, 'Must pass a single positional argument'),
))
def test_margins_errors(err, args, kwargs, match):
    # 使用 pytest 来断言特定错误被触发
    with pytest.raises(err, match=match):
        # 创建一个新的图形对象和一个子图
        fig = plt.figure()
        ax = fig.add_subplot()
        # 调用 ax.margins() 方法，传入参数 args 和 kwargs
        ax.margins(*args, **kwargs)


def test_length_one_hist():
    # 创建一个新的图形对象和一个子图
    fig, ax = plt.subplots()
    # 绘制直方图，数据为单个数值 1
    ax.hist(1)
    # 绘制直方图，数据为列表 [1]
    ax.hist([1])


def test_set_xy_bound():
    # 创建一个新的图形对象和一个子图
    fig = plt.figure()
    ax = fig.add_subplot()
    # 设置 x 轴限制为 (2.0, 3.0)
    ax.set_xbound(2.0, 3.0)
    # 断言获取子图的 x 轴限制为 (2.0, 3.0)
    assert ax.get_xbound() == (2.0, 3.0)
    # 断言获取子图的 x 轴显示限制为 (2.0, 3.0)
    assert ax.get_xlim() == (2.0, 3.0)
    # 设置 x 轴上限为 4.0
    ax.set_xbound(upper=4.0)
    # 断言获取子图的 x 轴限制为 (2.0, 4.0)
    assert ax.get_xbound() == (2.0, 4.0)
    # 断言获取子图的 x 轴显示限制为 (2.0, 4.0)
    assert ax.get_xlim() == (2.0, 4.0)
    # 设置 x 轴下限为 3.0
    ax.set_xbound(lower=3.0)
    # 断言获取子图的 x 轴限制为 (3.0, 4.0)
    assert ax.get_xbound() == (3.0, 4.0)
    # 断言获取子图的 x 轴显示限制为 (3.0, 4.0)

    # 设置 y 轴限制为 (2.0, 3.0)
    ax.set_ybound(2.0, 3.0)
    # 断言获取子图的 y 轴限制为 (2.0, 3.0)
    assert ax.get_ybound() == (2.0, 3.0)
    # 断言获取子图的 y 轴显示限制为 (2.0, 3.0)
    assert ax.get_ylim() == (2.0, 3.0)
    # 设置 y 轴上限为 4.0
    ax.set_ybound(upper=4.0)
    # 断言获取子图的 y 轴限制为 (2.0, 4.0)
    assert ax.get_ybound() == (2.0, 4.0)
    # 断言获取子图的 y 轴显示限制为 (2.0, 4.0)
    assert ax.get_ylim() == (2.0, 4.0)
    # 设置 y 轴下限为 3.0
    ax.set_ybound(lower=3.0)
    # 断言获取子图的 y 轴限制为 (3.0, 4.0)
    assert ax.get_ybound() == (3.0, 4.0)
    # 断言获取子图的 y 轴显示限制为 (3.0, 4.0)


def test_pathological_hexbin():
    # 问题 #2863
    mylist = [10] * 100
    # 创建一个新的图形对象和一个子图
    fig, ax = plt.subplots(1, 1)
    # 绘制二维直方图
    ax.hexbin(mylist, mylist)
    # 将图形保存到一个字节流中，检查是否发出警告
    fig.savefig(io.BytesIO())


def test_color_None():
    # 问题 #3855
    # 创建一个新的图形对象和一个子图
    fig, ax = plt.subplots()
    # 绘制线段，颜色为 None
    ax.plot([1, 2], [1, 2], color=None)


def test_color_alias():
    # 问题 #4157 和 #4162
    # 创建一个新的图形对象和一个子图
    fig, ax = plt.subplots()
    # 绘制线段，颜色为 'lime'
    line = ax.plot([0, 1], c='lime')[0]
    # 断言线段的颜色为 'lime'
    assert 'lime' == line.get_color()


def test_numerical_hist_label():
    # 创建一个新的图形对象和一个子图
    fig, ax = plt.subplots()
    # 绘制直方图，数据为多个长度为 15 的列表，标签为 range(5)
    ax.hist([range(15)] * 5, label=range(5))
    # 添加图例
    ax.legend()


def test_unicode_hist_label():
    # 创建一个新的图形对象和一个子图
    fig, ax = plt.subplots()
    # 创建字节字符串 a 和 b
    a = (b'\xe5\xbe\x88\xe6\xbc\x82\xe4\xba\xae, ' +
         b'r\xc3\xb6m\xc3\xa4n ch\xc3\xa4r\xc3\xa0ct\xc3\xa8rs')
    b = b'\xd7\xa9\xd7\x9c\xd7\x95\xd7\x9d'
    # 创建包含 Unicode 标签的列表
    labels = [a.decode('utf-8'),
              'hi aardvark',
              b.decode('utf-8'),
              ]
    # 在图形对象 ax 上绘制三个高度为 15 的柱状图，每个柱状图的标签由参数 labels 指定
    ax.hist([range(15)] * 3, label=labels)
    # 在图形对象 ax 上添加图例，图例内容为之前绘制的柱状图的标签
    ax.legend()
# 定义一个测试函数，用于测试移动偏移文本标签的功能
def test_move_offsetlabel():
    # 创建一个包含随机数据的 NumPy 数组
    data = np.random.random(10) * 1e-22

    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制数据图
    ax.plot(data)
    # 绘图后更新画布
    fig.canvas.draw()
    # 获取偏移文本的初始位置
    before = ax.yaxis.offsetText.get_position()
    # 断言偏移文本的水平对齐方式为左对齐
    assert ax.yaxis.offsetText.get_horizontalalignment() == 'left'
    # 将 y 轴刻度移到右侧
    ax.yaxis.tick_right()
    # 再次更新画布
    fig.canvas.draw()
    # 获取偏移文本的移动后位置
    after = ax.yaxis.offsetText.get_position()
    # 断言偏移文本的水平位置向右移动，垂直位置不变
    assert after[0] > before[0] and after[1] == before[1]
    # 断言偏移文本的水平对齐方式变为右对齐
    assert ax.yaxis.offsetText.get_horizontalalignment() == 'right'

    # 创建另一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制数据图
    ax.plot(data)
    # 绘图后更新画布
    fig.canvas.draw()
    # 获取偏移文本的初始位置
    before = ax.xaxis.offsetText.get_position()
    # 断言偏移文本的垂直对齐方式为顶部对齐
    assert ax.xaxis.offsetText.get_verticalalignment() == 'top'
    # 将 x 轴刻度移到顶部
    ax.xaxis.tick_top()
    # 再次更新画布
    fig.canvas.draw()
    # 获取偏移文本的移动后位置
    after = ax.xaxis.offsetText.get_position()
    # 断言偏移文本的水平位置不变，垂直位置向下移动
    assert after[0] == before[0] and after[1] > before[1]
    # 断言偏移文本的垂直对齐方式变为底部对齐
    assert ax.xaxis.offsetText.get_verticalalignment() == 'bottom'


# 使用图像对比功能，测试 matplotlib 的图形外观设定：隐藏所有脊柱
@image_comparison(['rc_spines.png'], savefig_kwarg={'dpi': 40})
def test_rc_spines():
    # 设定 matplotlib 的参数字典，隐藏图形的所有脊柱
    rc_dict = {
        'axes.spines.left': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.bottom': False}
    # 在设定的参数上下文中创建一个图形和轴对象，其脊柱属性将被隐藏


# 使用图像对比功能，测试 matplotlib 的网格线设定
@image_comparison(['rc_grid.png'], savefig_kwarg={'dpi': 40})
def test_rc_grid():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 定义不同的参数字典，分别设定图形的网格显示方式
    rc_dict0 = {
        'axes.grid': True,
        'axes.grid.axis': 'both'
    }
    rc_dict1 = {
        'axes.grid': True,
        'axes.grid.axis': 'x'
    }
    rc_dict2 = {
        'axes.grid': True,
        'axes.grid.axis': 'y'
    }
    # 将参数字典存入列表中
    dict_list = [rc_dict0, rc_dict1, rc_dict2]

    # 使用 for 循环依次在不同的参数上下文中创建子图
    for i, rc_dict in enumerate(dict_list, 1):
        with matplotlib.rc_context(rc_dict):
            fig.add_subplot(3, 1, i)


# 测试 matplotlib 的刻度设定
def test_rc_tick():
    # 定义刻度参数字典
    d = {'xtick.bottom': False, 'xtick.top': True,
         'ytick.left': True, 'ytick.right': False}
    # 在定义的参数上下文中创建一个新的图形对象
    with plt.rc_context(rc=d):
        fig = plt.figure()
        # 添加一个子图到图形中
        ax1 = fig.add_subplot(1, 1, 1)
        # 获取 x 轴和 y 轴对象
        xax = ax1.xaxis
        yax = ax1.yaxis
        # 断言主刻度和次刻度的显示设置
        assert not xax._major_tick_kw['tick1On']  # tick1On 为 False，即不显示底部的主刻度
        assert xax._major_tick_kw['tick2On']  # tick2On 为 True，即显示顶部的主刻度
        assert not xax._minor_tick_kw['tick1On']  # tick1On 为 False，即不显示底部的次刻度
        assert xax._minor_tick_kw['tick2On']  # tick2On 为 True，即显示顶部的次刻度

        assert yax._major_tick_kw['tick1On']  # tick1On 为 True，即显示左侧的主刻度
        assert not yax._major_tick_kw['tick2On']  # tick2On 为 False，即不显示右侧的主刻度
        assert yax._minor_tick_kw['tick1On']  # tick1On 为 True，即显示左侧的次刻度
        assert not yax._minor_tick_kw['tick2On']  # tick2On 为 False，即不显示右侧的次刻度


# 测试 matplotlib 的主次刻度设定
def test_rc_major_minor_tick():
    # 定义刻度参数字典，启用所有的主刻度和次刻度，选择性地禁用部分刻度
    d = {'xtick.top': True, 'ytick.right': True,
         'xtick.bottom': True, 'ytick.left': True,
         'xtick.minor.bottom': False, 'xtick.major.bottom': False,
         'ytick.major.left': False, 'ytick.minor.left': False}
    # 使用指定的绘图参数上下文环境进行绘图设置
    with plt.rc_context(rc=d):
        # 创建一个新的图形对象
        fig = plt.figure()
        # 添加一个子图到当前图形，1行1列的第1个位置
        ax1 = fig.add_subplot(1, 1, 1)
        # 获取子图的X轴和Y轴对象
        xax = ax1.xaxis
        yax = ax1.yaxis
        
        # 断言验证主要X轴刻度的属性
        assert not xax._major_tick_kw['tick1On']  # 主要刻度线1在底部/左侧关闭
        assert xax._major_tick_kw['tick2On']      # 主要刻度线2在底部/左侧开启
        assert not xax._minor_tick_kw['tick1On']  # 次要刻度线1在底部/左侧关闭
        assert xax._minor_tick_kw['tick2On']      # 次要刻度线2在底部/左侧开启
        
        # 断言验证主要Y轴刻度的属性
        assert not yax._major_tick_kw['tick1On']  # 主要刻度线1在底部/左侧关闭
        assert yax._major_tick_kw['tick2On']      # 主要刻度线2在底部/左侧开启
        assert not yax._minor_tick_kw['tick1On']  # 次要刻度线1在底部/左侧关闭
        assert yax._minor_tick_kw['tick2On']      # 次要刻度线2在底部/左侧开启
# 定义一个测试函数，用于绘制正方形的图形
def test_square_plot():
    # 创建一个包含四个元素的 NumPy 数组，表示 x 轴的数据点
    x = np.arange(4)
    # 创建一个包含四个浮点数的 NumPy 数组，表示 y 轴的数据点
    y = np.array([1., 3., 5., 7.])
    # 创建一个图形对象和一个坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制 x 和 y 的散点图，颜色为品红色
    ax.plot(x, y, 'mo')
    # 设置坐标轴的纵横比为1:1，使图形呈现为正方形
    ax.axis('square')
    # 获取当前坐标轴的 x 和 y 轴的限制范围
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # 断言 x 轴和 y 轴的长度相等
    assert np.diff(xlim) == np.diff(ylim)
    # 断言坐标轴的纵横比为1
    assert ax.get_aspect() == 1
    # 断言当前坐标轴的位置和尺寸与给定的原始值匹配
    assert_array_almost_equal(
        ax.get_position(original=True).extents, (0.125, 0.1, 0.9, 0.9))
    # 断言当前坐标轴的位置和尺寸与给定的非原始值匹配
    assert_array_almost_equal(
        ax.get_position(original=False).extents, (0.2125, 0.1, 0.8125, 0.9))


# 定义一个测试函数，用于测试绘图函数对于错误参数的处理
def test_bad_plot_args():
    # 断言调用 plt.plot(None) 会引发 ValueError 异常
    with pytest.raises(ValueError):
        plt.plot(None)
    # 断言调用 plt.plot(None, None) 会引发 ValueError 异常
    with pytest.raises(ValueError):
        plt.plot(None, None)
    # 断言调用 plt.plot(np.zeros((2, 2)), np.zeros((2, 3))) 会引发 ValueError 异常
    with pytest.raises(ValueError):
        plt.plot(np.zeros((2, 2)), np.zeros((2, 3)))
    # 断言调用 plt.plot((np.arange(5).reshape((1, -1)), np.arange(5).reshape(-1, 1))) 会引发 ValueError 异常
    with pytest.raises(ValueError):
        plt.plot((np.arange(5).reshape((1, -1)), np.arange(5).reshape(-1, 1)))


# 使用参数化测试装饰器，定义测试函数，用于测试 pcolorfast 函数的功能
@pytest.mark.parametrize(
    "xy, cls", [
        ((), mpl.image.AxesImage),  # (0, N)
        (((3, 7), (2, 6)), mpl.image.AxesImage),  # (xmin, xmax)
        ((range(5), range(4)), mpl.image.AxesImage),  # regular grid
        (([1, 2, 4, 8, 16], [0, 1, 2, 3]),  # irregular grid
         mpl.image.PcolorImage),
        ((np.random.random((4, 5)), np.random.random((4, 5))),  # 2D coords
         mpl.collections.QuadMesh),
    ]
)
@pytest.mark.parametrize(
    "data", [np.arange(12).reshape((3, 4)), np.random.rand(3, 4, 3)]
)
def test_pcolorfast(xy, data, cls):
    # 创建一个图形对象和一个坐标轴对象
    fig, ax = plt.subplots()
    # 断言调用 ax.pcolorfast(*xy, data) 返回的对象类型与预期的 cls 类型匹配
    assert type(ax.pcolorfast(*xy, data)) == cls


# 定义一个测试函数，用于测试 pcolorfast 函数在维度错误情况下的处理
def test_pcolorfast_bad_dims():
    # 创建一个图形对象和一个坐标轴对象
    fig, ax = plt.subplots()
    # 断言调用 ax.pcolorfast(np.empty(6), np.empty((4, 7)), np.empty((8, 8))) 会引发 TypeError 异常，并且匹配指定的错误消息
    with pytest.raises(
            TypeError, match=("the given X was 1D and the given Y was 2D")):
        ax.pcolorfast(np.empty(6), np.empty((4, 7)), np.empty((8, 8)))


# 定义一个测试函数，用于测试共享坐标轴比例的设置
def test_shared_scale():
    # 创建一个包含 2x2 子图的图形对象和子图数组
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    # 在左上角的子图上设置 x 和 y 轴的缩放为对数尺度
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_yscale("log")

    # 断言所有子图的 y 轴缩放均为对数尺度
    for ax in axs.flat:
        assert ax.get_yscale() == 'log'
        assert ax.get_xscale() == 'log'

    # 在右下角的子图上设置 x 和 y 轴的缩放为线性尺度
    axs[1, 1].set_xscale("linear")
    axs[1, 1].set_yscale("linear")

    # 再次断言所有子图的 y 轴缩放均为线性尺度
    for ax in axs.flat:
        assert ax.get_yscale() == 'linear'
        assert ax.get_xscale() == 'linear'


# 定义一个测试函数，用于测试共享坐标轴时的参数错误处理
def test_shared_bool():
    # 断言调用 plt.subplot(sharex=True) 会引发 TypeError 异常
    with pytest.raises(TypeError):
        plt.subplot(sharex=True)
    # 断言调用 plt.subplot(sharey=True) 会引发 TypeError 异常
    with pytest.raises(TypeError):
        plt.subplot(sharey=True)


# 定义一个测试函数，用于测试小提琴图处理点质量概率密度函数的情况
def test_violin_point_mass():
    # 绘制一个小提琴图，显示一个点质量概率密度函数
    plt.violinplot(np.array([0, 0]))


# 定义一个生成错误条输入的辅助函数
def generate_errorbar_inputs():
    # 创建一个基础的 cycler 对象，用于循环 x 和 y 轴的数据
    base_xy = cycler('x', [np.arange(5)]) + cycler('y', [np.ones(5)])
    # 创建一个错误条的 cycler 对象，包含不同的错误值组合
    err_cycler = cycler('err', [1,
                                [1, 1, 1, 1, 1],
                                [[1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1]],
                                np.ones(5),
                                np.ones((2, 5)),
                                None
                                ])
    # 创建 xerr 的 cycler 对象，包含不同的 x 轴错误值组合
    xerr_cy = cycler('xerr', err_cycler)
    # 创建 yerr 的 cycler 对象，包含不同的 y 轴错误值组合
    yerr_cy = cycler('yerr', err_cycler)
    # 创建一个空的循环迭代器，包含两个空列表作为 'x' 和 'y' 的初始值
    empty = ((cycler('x', [[]]) + cycler('y', [[]])) *
             cycler('xerr', [[], None]) * cycler('yerr', [[], None]))
    
    # 基于 'base_xy' 和 'xerr_cy' 创建新的循环迭代器，表示仅包含 'xerr' 的情况
    xerr_only = base_xy * xerr_cy
    
    # 基于 'base_xy' 和 'yerr_cy' 创建新的循环迭代器，表示仅包含 'yerr' 的情况
    yerr_only = base_xy * yerr_cy
    
    # 基于 'base_xy'、'yerr_cy' 和 'xerr_cy' 创建新的循环迭代器，表示同时包含 'xerr' 和 'yerr' 的情况
    both_err = base_xy * yerr_cy * xerr_cy

    # 返回一个包含所有上述循环迭代器的展开列表
    return [*xerr_only, *yerr_only, *both_err, *empty]
# 使用 pytest.mark.parametrize 装饰器来参数化测试，生成多个测试用例，每个用例使用 generate_errorbar_inputs() 生成的参数
@pytest.mark.parametrize('kwargs', generate_errorbar_inputs())
def test_errorbar_inputs_shotgun(kwargs):
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 调用 errorbar 方法绘制误差条图，使用参数 kwargs
    eb = ax.errorbar(**kwargs)
    # 移除误差条图形对象
    eb.remove()


# 使用 image_comparison 装饰器进行测试，比较生成的图像与预期的 'dash_offset' 图像
@image_comparison(["dash_offset"], remove_text=True)
def test_dash_offset():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 生成一系列数据点
    x = np.linspace(0, 10)
    y = np.ones_like(x)
    # 循环绘制多条具有不同线型和宽度的曲线
    for j in range(0, 100, 2):
        ax.plot(x, j*y, ls=(j, (10, 10)), lw=5, color='k')


# 测试标题的内边距设置效果
def test_title_pad():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 设置标题为 'aardvark'，并指定内边距为 30
    ax.set_title('aardvark', pad=30.)
    # 获取标题的偏移矩阵
    m = ax.titleOffsetTrans.get_matrix()
    # 断言标题偏移的像素值符合预期
    assert m[1, -1] == (30. / 72. * fig.dpi)
    # 设置标题的内边距为 0
    ax.set_title('aardvark', pad=0.)
    # 再次获取标题的偏移矩阵
    m = ax.titleOffsetTrans.get_matrix()
    # 断言标题偏移的像素值为 0
    assert m[1, -1] == 0.
    # 恢复为默认的标题内边距设置
    ax.set_title('aardvark', pad=None)
    # 获取标题的偏移矩阵
    m = ax.titleOffsetTrans.get_matrix()
    # 断言标题偏移的像素值符合默认配置
    assert m[1, -1] == (matplotlib.rcParams['axes.titlepad'] / 72. * fig.dpi)


# 测试标题位置设置往返
def test_title_location_roundtrip():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 设置默认的标题位置为 'center'
    plt.rcParams['axes.titlelocation'] = 'center'
    # 设置标题为 'aardvark'
    ax.set_title('aardvark')
    # 分别设置标题位置为 'left' 和 'right'
    ax.set_title('left', loc='left')
    ax.set_title('right', loc='right')

    # 断言获取的 'left' 位置的标题符合预期
    assert 'left' == ax.get_title(loc='left')
    # 断言获取的 'right' 位置的标题符合预期
    assert 'right' == ax.get_title(loc='right')
    # 断言获取的 'center' 位置的标题符合预期
    assert 'aardvark' == ax.get_title(loc='center')

    # 使用 pytest.raises 检查设置不支持的位置时是否引发 ValueError 异常
    with pytest.raises(ValueError):
        ax.get_title(loc='foo')
    with pytest.raises(ValueError):
        ax.set_title('fail', loc='foo')


# 测试共享坐标轴的标题位置设置
@pytest.mark.parametrize('sharex', [True, False])
def test_title_location_shared(sharex):
    # 创建包含两个子图的图形对象，共享 x 轴设置由参数 sharex 决定
    fig, axs = plt.subplots(2, 1, sharex=sharex)
    # 设置第一个子图的标题为 'A'，并指定负的内边距
    axs[0].set_title('A', pad=-40)
    # 设置第二个子图的标题为 'B'，并指定负的内边距
    axs[1].set_title('B', pad=-40)
    # 在不渲染的情况下绘制图形
    fig.draw_without_rendering()
    # 获取两个子图标题的位置
    x, y1 = axs[0].title.get_position()
    x, y2 = axs[1].title.get_position()
    # 断言两个子图标题的垂直位置相同
    assert y1 == y2 == 1.0


# 使用 image_comparison 装饰器进行测试，比较生成的图像与预期的 'loglog.png' 图像
@image_comparison(["loglog.png"], remove_text=True, tol=0.02)
def test_loglog():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 生成数据点
    x = np.arange(1, 11)
    # 对数对数图
    ax.loglog(x, x**3, lw=5)
    # 设置主刻度的长度和宽度
    ax.tick_params(length=25, width=2)
    # 设置次刻度的长度、宽度和显示方式
    ax.tick_params(length=15, width=2, which='minor')


# 使用 image_comparison 装饰器进行测试，比较生成的图像与预期的 'test_loglog_nonpos.png' 图像
@image_comparison(["test_loglog_nonpos.png"], remove_text=True, style='mpl20',
                  tol=0.029 if platform.machine() == 'arm64' else 0)
def test_loglog_nonpos():
    # 创建包含 3x3 子图的图形对象
    fig, axs = plt.subplots(3, 3)
    # 生成数据点
    x = np.arange(1, 11)
    y = x**3
    y[7] = -3.
    x[4] = -10
    # 循环绘制每个子图中的曲线，并设置非正数值的处理方式
    for (mcy, mcx), ax in zip(product(['mask', 'clip', ''], repeat=2),
                              axs.flat):
        if mcx == mcy:
            if mcx:
                ax.loglog(x, y**3, lw=2, nonpositive=mcx)
            else:
                ax.loglog(x, y**3, lw=2)
        else:
            ax.loglog(x, y**3, lw=2)
            if mcx:
                ax.set_xscale("log", nonpositive=mcx)
            if mcy:
                ax.set_yscale("log", nonpositive=mcy)


# 使用 mpl.style.context('default') 上下文管理器来测试默认风格下的坐标轴边距
def test_axes_margins():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制简单的折线图
    ax.plot([0, 1, 2, 3])
    # 断言确保图形的 y 轴下限不为零
    assert ax.get_ybound()[0] != 0
    
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制柱状图
    ax.bar([0, 1, 2, 3], [1, 1, 1, 1])
    # 断言确保图形的 y 轴下限为零
    assert ax.get_ybound()[0] == 0
    
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制水平柱状图
    ax.barh([0, 1, 2, 3], [1, 1, 1, 1])
    # 断言确保图形的 x 轴下限为零
    assert ax.get_xbound()[0] == 0
    
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制颜色格子图
    ax.pcolor(np.zeros((10, 10)))
    # 断言确保图形的 x 和 y 轴的边界为 (0, 10)
    assert ax.get_xbound() == (0, 10)
    assert ax.get_ybound() == (0, 10)
    
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 快速绘制颜色格子图
    ax.pcolorfast(np.zeros((10, 10)))
    # 断言确保图形的 x 和 y 轴的边界为 (0, 10)
    assert ax.get_xbound() == (0, 10)
    assert ax.get_ybound() == (0, 10)
    
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制直方图
    ax.hist(np.arange(10))
    # 断言确保图形的 y 轴下限为零
    assert ax.get_ybound()[0] == 0
    
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 显示图像
    ax.imshow(np.zeros((10, 10)))
    # 断言确保图形的 x 和 y 轴的边界为 (-0.5, 9.5)
    assert ax.get_xbound() == (-0.5, 9.5)
    assert ax.get_ybound() == (-0.5, 9.5)
@pytest.fixture(params=['x', 'y'])
def shared_axis_remover(request):
    def _helper_x(ax):
        # 创建一个新的y轴，并将其移除，以确保当前图形不再共享y轴
        ax2 = ax.twinx()
        ax2.remove()
        # 设置x轴的限制为0到15
        ax.set_xlim(0, 15)
        # 获取x轴主定位器，并调用以获取主刻度位置
        r = ax.xaxis.get_major_locator()()
        # 断言最后一个主刻度位置大于14
        assert r[-1] > 14

    def _helper_y(ax):
        # 创建一个新的x轴，并将其移除，以确保当前图形不再共享x轴
        ax2 = ax.twiny()
        ax2.remove()
        # 设置y轴的限制为0到15
        ax.set_ylim(0, 15)
        # 获取y轴主定位器，并调用以获取主刻度位置
        r = ax.yaxis.get_major_locator()()
        # 断言最后一个主刻度位置大于14
        assert r[-1] > 14

    return {"x": _helper_x, "y": _helper_y}[request.param]


@pytest.fixture(params=['gca', 'subplots', 'subplots_shared', 'add_axes'])
def shared_axes_generator(request):
    # 测试获取图和轴的所有方式
    if request.param == 'gca':
        # 创建一个新的图形，并获取当前轴
        fig = plt.figure()
        ax = fig.gca()
    elif request.param == 'subplots':
        # 创建一个包含单个轴的新图形
        fig, ax = plt.subplots()
    elif request.param == 'subplots_shared':
        # 创建一个包含2x2子图的新图形，并共享所有x轴和y轴
        fig, ax_lst = plt.subplots(2, 2, sharex='all', sharey='all')
        ax = ax_lst[0][0]
    elif request.param == 'add_axes':
        # 创建一个新的图形，并添加一个轴，位置为(0.1, 0.1)，大小为(0.8, 0.8)
        fig = plt.figure()
        ax = fig.add_axes([.1, .1, .8, .8])
    return fig, ax


def test_remove_shared_axes(shared_axes_generator, shared_axis_remover):
    # 测试获取图和轴的所有方式
    fig, ax = shared_axes_generator
    # 使用共享轴移除函数
    shared_axis_remover(ax)


def test_remove_shared_axes_relim():
    # 创建一个包含2x2子图的新图形，并共享所有x轴和y轴
    fig, ax_lst = plt.subplots(2, 2, sharex='all', sharey='all')
    ax = ax_lst[0][0]
    # 获取第一行第二列轴的原始x轴限制
    orig_xlim = ax_lst[0][1].get_xlim()
    # 移除当前轴
    ax.remove()
    # 设置当前轴的x轴限制为0到5
    ax.set_xlim(0, 5)
    # 断言第一行第二列轴的x轴限制与原始限制相等
    assert_array_equal(ax_lst[0][1].get_xlim(), orig_xlim)


def test_shared_axes_autoscale():
    # 创建一个包含2个子图的新图形，并共享x轴和y轴
    l = np.arange(-80, 90, 40)
    t = np.random.random_sample((l.size, l.size))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    # 设置第一个子图的x轴和y轴限制
    ax1.set_xlim(-1000, 1000)
    ax1.set_ylim(-1000, 1000)
    # 在第一个子图上绘制等高线
    ax1.contour(l, l, t)

    # 在第二个子图上绘制等高线
    ax2.contour(l, l, t)
    # 断言第一个子图和第二个子图的自动缩放状态为关闭
    assert not ax1.get_autoscalex_on() and not ax2.get_autoscalex_on()
    assert not ax1.get_autoscaley_on() and not ax2.get_autoscaley_on()
    # 断言第一个子图和第二个子图的x轴和y轴限制相等
    assert ax1.get_xlim() == ax2.get_xlim() == (-1000, 1000)
    assert ax1.get_ylim() == ax2.get_ylim() == (-1000, 1000)


def test_adjust_numtick_aspect():
    # 创建一个新图形和轴
    fig, ax = plt.subplots()
    # 设置y轴主定位器的参数为'auto'
    ax.yaxis.get_major_locator().set_params(nbins='auto')
    # 设置x轴限制为0到1000
    ax.set_xlim(0, 1000)
    # 设置图形的纵横比为相等
    ax.set_aspect('equal')
    fig.canvas.draw()
    # 断言y轴主定位器的刻度数量为2
    assert len(ax.yaxis.get_major_locator()()) == 2
    # 设置y轴限制为0到1000
    ax.set_ylim(0, 1000)
    fig.canvas.draw()
    # 断言y轴主定位器的刻度数量大于2
    assert len(ax.yaxis.get_major_locator()()) > 2


@mpl.style.context("default")
def test_auto_numticks():
    # 创建一个包含16个子图的新图形
    axs = plt.figure().subplots(4, 4)
    for ax in axs.flat:  # 小型、空白的子图只有3个刻度
        assert [*ax.get_xticks()] == [*ax.get_yticks()] == [0, 0.5, 1]


@mpl.style.context("default")
def test_auto_numticks_log():
    # 验证具有大的对数范围时没有太多的刻度
    fig, ax = plt.subplots()
    # 设置axes.autolimit_mode参数为'round_numbers'
    mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
    # 在对数-对数坐标系上绘制图形
    ax.loglog([1e-20, 1e5], [1e-16, 10])
    # 断言x轴刻度的对数值与指定范围内的对数值相等
    assert (np.log10(ax.get_xticks()) == np.arange(-26, 18, 4)).all()
    # 使用断言来验证条件，确保所有 y 轴刻度的对数值在指定范围内连续分布
    assert (np.log10(ax.get_yticks()) == np.arange(-20, 10, 3)).all()
def test_broken_barh_empty():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制一个空的 broken_barh，没有任何矩形条
    ax.broken_barh([], (.1, .5))


def test_broken_barh_timedelta():
    """Check that timedelta works as x, dx pair for this method."""
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 设置起始日期时间
    d0 = datetime.datetime(2018, 11, 9, 0, 0, 0)
    # 在坐标轴上绘制一个 broken_barh，使用 timedelta 作为 x, dx 对
    pp = ax.broken_barh([(d0, datetime.timedelta(hours=1))], [1, 2])
    # 断言检查第一个路径的顶点是否正确设置为起始日期的数值化形式
    assert pp.get_paths()[0].vertices[0, 0] == mdates.date2num(d0)
    # 断言检查第一个路径的第三个顶点是否正确设置为起始日期加上一小时的数值化形式
    assert pp.get_paths()[0].vertices[2, 0] == mdates.date2num(d0) + 1 / 24


def test_pandas_pcolormesh(pd):
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制 pcolormesh 图像，使用给定的时间、深度和数据
    ax.pcolormesh(time, depth, data)


def test_pandas_indexing_dates(pd):
    # 创建日期范围
    dates = np.arange('2005-02', '2005-03', dtype='datetime64[D]')
    # 创建一个正弦值数据的 DataFrame
    values = np.sin(range(len(dates)))
    df = pd.DataFrame({'dates': dates, 'values': values})

    # 获取当前的坐标轴对象
    ax = plt.gca()

    # 选择 DataFrame 中索引为奇数的行，并创建一个副本
    without_zero_index = df[np.array(df.index) % 2 == 1].copy()
    # 在坐标轴上绘制 dates 列和 values 列的折线图
    ax.plot('dates', 'values', data=without_zero_index)


def test_pandas_errorbar_indexing(pd):
    # 创建一个具有随机均匀分布数据的 DataFrame
    df = pd.DataFrame(np.random.uniform(size=(5, 4)),
                      columns=['x', 'y', 'xe', 'ye'],
                      index=[1, 2, 3, 4, 5])
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制带有误差棒的数据点图，使用 DataFrame 中的列作为数据源
    ax.errorbar('x', 'y', xerr='xe', yerr='ye', data=df)


def test_pandas_index_shape(pd):
    # 创建一个具有两列的 DataFrame
    df = pd.DataFrame({"XX": [4, 5, 6], "YY": [7, 1, 2]})
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制 df 的索引与 YY 列的关系图
    ax.plot(df.index, df['YY'])


def test_pandas_indexing_hist(pd):
    # 创建一个包含重复数据的 Series
    ser_1 = pd.Series(data=[1, 2, 2, 3, 3, 4, 4, 4, 4, 5])
    # 创建一个切片，去除第一个元素
    ser_2 = ser_1.iloc[1:]
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制 ser_2 的直方图
    ax.hist(ser_2)


def test_pandas_bar_align_center(pd):
    # 测试解决问题 8767 的修复
    # 创建一个具有两列的 DataFrame
    df = pd.DataFrame({'a': range(2), 'b': range(2)})

    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots(1)

    # 在坐标轴上绘制柱状图，通过 align='center' 修复问题
    ax.bar(df.loc[df['a'] == 1, 'b'],
           df.loc[df['a'] == 1, 'b'],
           align='center')

    # 绘制完成后刷新图形显示
    fig.canvas.draw()


def test_axis_get_tick_params():
    # 获取当前 subplot 的 y 轴对象
    axis = plt.subplot().yaxis
    # 获取初始的主要刻度参数，并进行反向转换
    initial_major_style_translated = {**axis.get_tick_params(which='major')}
    initial_minor_style_translated = {**axis.get_tick_params(which='minor')}

    # 对主要和次要刻度参数进行转换
    translated_major_kw = axis._translate_tick_params(
        axis._major_tick_kw, reverse=True
    )
    translated_minor_kw = axis._translate_tick_params(
        axis._minor_tick_kw, reverse=True
    )

    # 断言初始和转换后的主要、次要刻度参数相等
    assert translated_major_kw == initial_major_style_translated
    assert translated_minor_kw == initial_minor_style_translated

    # 设置刻度参数的标签大小、颜色和方向，对主要和次要刻度都应用
    axis.set_tick_params(labelsize=30, labelcolor='red',
                         direction='out', which='both')

    # 获取更新后的主要和次要刻度参数，并进行转换
    new_major_style_translated = {**axis.get_tick_params(which='major')}
    new_minor_style_translated = {**axis.get_tick_params(which='minor')}
    new_major_style = axis._translate_tick_params(new_major_style_translated)
    new_minor_style = axis._translate_tick_params(new_minor_style_translated)

    # 断言更新后的主要刻度参数与初始不相等
    assert initial_major_style_translated != new_major_style_translated
    # 断言：检查轴的主要刻度线样式是否与新的主要样式相等
    assert axis._major_tick_kw == new_major_style
    
    # 断言：检查初始的次要刻度线样式转换后是否不等于新的次要样式转换后的结果
    assert initial_minor_style_translated != new_minor_style_translated
    
    # 断言：检查轴的次要刻度线样式是否与新的次要样式相等
    assert axis._minor_tick_kw == new_minor_style
def test_axis_set_tick_params_labelsize_labelcolor():
    # Tests fix for issue 4346
    # 创建一个子图对象
    axis_1 = plt.subplot()
    # 设置y轴刻度的参数：标签大小为30，标签颜色为红色，刻度方向为外部
    axis_1.yaxis.set_tick_params(labelsize=30, labelcolor='red',
                                 direction='out')

    # 预期设置刻度后的数值
    assert axis_1.yaxis.majorTicks[0]._size == 4.0
    assert axis_1.yaxis.majorTicks[0].tick1line.get_color() == 'k'
    assert axis_1.yaxis.majorTicks[0].label1.get_size() == 30.0
    assert axis_1.yaxis.majorTicks[0].label1.get_color() == 'red'


def test_axes_tick_params_gridlines():
    # Now treating grid params like other Tick params
    # 创建一个子图对象
    ax = plt.subplot()
    # 设置坐标轴刻度的参数：网格线颜色为蓝色，线宽为5，透明度为0.5，线型为点划线
    ax.tick_params(grid_color='b', grid_linewidth=5, grid_alpha=0.5,
                   grid_linestyle='dashdot')
    for axis in ax.xaxis, ax.yaxis:
        # 检查主刻度的网格线属性
        assert axis.majorTicks[0].gridline.get_color() == 'b'
        assert axis.majorTicks[0].gridline.get_linewidth() == 5
        assert axis.majorTicks[0].gridline.get_alpha() == 0.5
        assert axis.majorTicks[0].gridline.get_linestyle() == '-.'


def test_axes_tick_params_ylabelside():
    # Tests fix for issue 10267
    # 创建一个子图对象
    ax = plt.subplot()
    # 设置y轴刻度的参数：左侧标签不可见，右侧标签可见，作用于主刻度和次刻度
    ax.tick_params(labelleft=False, labelright=True,
                   which='major')
    ax.tick_params(labelleft=False, labelright=True,
                   which='minor')
    # 预期左侧标签不可见，右侧标签可见
    assert ax.yaxis.majorTicks[0].label1.get_visible() is False
    assert ax.yaxis.majorTicks[0].label2.get_visible() is True
    assert ax.yaxis.minorTicks[0].label1.get_visible() is False
    assert ax.yaxis.minorTicks[0].label2.get_visible() is True


def test_axes_tick_params_xlabelside():
    # Tests fix for issue 10267
    # 创建一个子图对象
    ax = plt.subplot()
    # 设置x轴刻度的参数：顶部标签可见，底部标签不可见，作用于主刻度和次刻度
    ax.tick_params(labeltop=True, labelbottom=False,
                   which='major')
    ax.tick_params(labeltop=True, labelbottom=False,
                   which='minor')
    # 预期顶部标签可见，底部标签不可见
    assert ax.xaxis.majorTicks[0].label1.get_visible() is False
    assert ax.xaxis.majorTicks[0].label2.get_visible() is True
    assert ax.xaxis.minorTicks[0].label1.get_visible() is False
    assert ax.xaxis.minorTicks[0].label2.get_visible() is True


def test_none_kwargs():
    # 创建一个图形对象，并添加子图
    ax = plt.figure().subplots()
    # 绘制折线图，设置线型为空（实际上默认为实线）
    ln, = ax.plot(range(32), linestyle=None)
    # 断言线型是否为默认的实线
    assert ln.get_linestyle() == '-'


def test_bar_uint8():
    xs = [0, 1, 2, 3]
    # 绘制柱状图，将x数据类型设置为无符号8位整数
    b = plt.bar(np.array(xs, dtype=np.uint8), [2, 3, 4, 5], align="edge")
    for (patch, x) in zip(b.patches, xs):
        # 检查每个柱的起始x坐标是否与数据一致
        assert patch.xy[0] == x


@image_comparison(['date_timezone_x.png'], tol=1.0)
def test_date_timezone_x():
    # Tests issue 5575
    # 创建一个日期-时区相关的时间索引
    time_index = [datetime.datetime(2016, 2, 22, hour=x,
                                    tzinfo=dateutil.tz.gettz('Canada/Eastern'))
                  for x in range(3)]

    # 创建一个大图，设置子图布局为2行1列中的第1个
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    # 在使用 pytest 运行测试时，捕获并忽略 Matplotlib 的过时警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        # 使用 plot_date 绘制时间序列图，时间索引为 time_index，值为 [3, 3, 3]，时区设为 'Canada/Eastern'
        plt.plot_date(time_index, [3] * 3, tz='Canada/Eastern')
    
    # 在子图中绘制另一个时间序列图
    plt.subplot(2, 1, 2)
    # 再次捕获并忽略 Matplotlib 的过时警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        # 使用 plot_date 绘制时间序列图，时间索引为 time_index，值为 [3, 3, 3]，时区设为 'UTC'
        plt.plot_date(time_index, [3] * 3, tz='UTC')
@image_comparison(['date_timezone_y.png'])
# 定义一个装饰器，用于测试函数，并生成名为'date_timezone_y.png'的图像比较结果
def test_date_timezone_y():
    # 测试问题 5575
    # 创建一个包含三个日期时间对象的列表，每个对象在不同的小时数下，使用加拿大东部时区
    time_index = [datetime.datetime(2016, 2, 22, hour=x,
                                    tzinfo=dateutil.tz.gettz('Canada/Eastern'))
                  for x in range(3)]

    # 绘制图像，设置图像大小为20x12
    plt.figure(figsize=(20, 12))
    # 在第一个子图中绘制日期-时间数据，使用Canada/Eastern时区，x轴不使用日期格式，y轴使用日期格式
    plt.subplot(2, 1, 1)
    # 捕获MatplotlibDeprecationWarning警告，以确保测试的稳定性
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        plt.plot_date([3] * 3, time_index, tz='Canada/Eastern', xdate=False, ydate=True)

    # 在第二个子图中绘制日期-时间数据，使用UTC时区，x轴不使用日期格式，y轴使用日期格式
    plt.subplot(2, 1, 2)
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        plt.plot_date([3] * 3, time_index, tz='UTC', xdate=False, ydate=True)


@image_comparison(['date_timezone_x_and_y.png'], tol=1.0)
# 定义一个装饰器，用于测试函数，并生成名为'date_timezone_x_and_y.png'的图像比较结果，容差为1.0
def test_date_timezone_x_and_y():
    # 测试问题 5575
    UTC = datetime.timezone.utc
    # 创建一个包含三个日期时间对象的列表，每个对象在不同的小时数下，使用UTC时区
    time_index = [datetime.datetime(2016, 2, 22, hour=x, tzinfo=UTC)
                  for x in range(3)]

    # 绘制图像，设置图像大小为20x12
    plt.figure(figsize=(20, 12))
    # 在第一个子图中绘制日期-时间数据，x轴和y轴均使用UTC时区
    plt.subplot(2, 1, 1)
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        plt.plot_date(time_index, time_index, tz='UTC', ydate=True)

    # 在第二个子图中绘制日期-时间数据，x轴和y轴均使用US/Eastern时区
    plt.subplot(2, 1, 2)
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        plt.plot_date(time_index, time_index, tz='US/Eastern', ydate=True)


@image_comparison(['axisbelow.png'], remove_text=True)
# 定义一个装饰器，用于测试函数，并生成名为'axisbelow.png'的图像比较结果，移除文本
def test_axisbelow():
    # 测试在6287号问题中添加的'line'设置
    # 创建包含3个子图的图像，并共享相同的x轴和y轴
    axs = plt.figure().subplots(ncols=3, sharex=True, sharey=True)
    settings = (False, 'line', True)

    for ax, setting in zip(axs, settings):
        # 在每个子图中绘制一条线，设置线宽为10，颜色为magenta
        ax.plot((0, 10), (0, 10), lw=10, color='m')
        # 在每个子图中添加一个红色圆形
        circ = mpatches.Circle((3, 3), color='r')
        ax.add_patch(circ)
        # 在每个子图中绘制网格线，颜色为青色，线型为实线，线宽为3
        ax.grid(color='c', linestyle='-', linewidth=3)
        # 设置每个子图的刻度参数，不显示顶部、底部、左侧和右侧的刻度线
        ax.tick_params(top=False, bottom=False,
                       left=False, right=False)
        # 隐藏每个子图的轴线
        ax.spines[:].set_visible(False)
        # 设置轴线在网格线之下的绘制顺序
        ax.set_axisbelow(setting)
        # 断言当前轴线的绘制顺序与设置的值相符
        assert ax.get_axisbelow() == setting


def test_titletwiny():
    # 测试标题是否在Xlabel2的上方保持不变
    plt.style.use('mpl20')
    # 创建分辨率为72的图像和坐标系
    fig, ax = plt.subplots(dpi=72)
    # 在同一个坐标系上创建第二个x轴
    ax2 = ax.twiny()
    # 设置第二个x轴的标签为'Xlabel2'
    xlabel2 = ax2.set_xlabel('Xlabel2')
    # 设置坐标系的标题为'Title'
    title = ax.set_title('Title')
    # 绘制图像的渲染器
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # ------- 测试标题是否位于Xlabel2的上方 ----------
    # 获取标题的窗口范围，并计算其底部坐标
    bbox_y0_title = title.get_window_extent(renderer).y0  # 标题底部
    # 获取Xlabel2的窗口范围，并计算其顶部坐标
    bbox_y1_xlabel2 = xlabel2.get_window_extent(renderer).y1  # Xlabel2顶部
    # 计算标题底部与Xlabel2顶部之间的距离差异
    y_diff = bbox_y0_title - bbox_y1_xlabel2
    # 断言标题底部与Xlabel2顶部之间的距离是否接近3个单位
    assert np.isclose(y_diff, 3)


def test_titlesetpos():
    # 测试当手动设置时，标题是否保持不变
    # 创建图像和坐标系
    fig, ax = plt.subplots()
    # 调整图像的子图间距，使得顶部间距为0.8
    fig.subplots_adjust(top=0.8)
    # 在同一个坐标系上创建第二个x轴
    ax2 = ax.twiny()
    # 设置坐标系的x轴标签为'Xlabel'
    ax.set_xlabel('Xlabel')
    # 设置第二个x轴的标签为'Xlabel2'
    ax2.set_xlabel('Xlabel2')
    # 设置坐标系的标题为'Title'
    ax.set_title('Title')
    # 设置标题的位置为(0.5, 1.11)
    pos = (0.5, 1.11)
    ax.title.set_position(pos)
    # 获取图表的渲染器对象，用于后续操作
    renderer = fig.canvas.get_renderer()
    # 更新轴标题的位置信息，使用给定的渲染器对象
    ax._update_title_position(renderer)
    # 断言轴的标题位置与预期位置相同
    assert ax.title.get_position() == pos
def test_title_xticks_top():
    # 测试当xticks位于坐标轴顶部时，标题是否移动。
    mpl.rcParams['axes.titley'] = None  # 设置标题的y轴位置为None
    fig, ax = plt.subplots()  # 创建一个新的图和坐标轴
    ax.xaxis.set_ticks_position('top')  # 设置x轴ticks在坐标轴顶部显示
    ax.set_title('xlabel top')  # 设置坐标轴标题为'xlabel top'
    fig.canvas.draw()  # 绘制图形
    assert ax.title.get_position()[1] > 1.04  # 断言标题的y轴位置是否大于1.04


def test_title_xticks_top_both():
    # 测试当xticks同时位于坐标轴顶部和底部时，标题是否移动。
    mpl.rcParams['axes.titley'] = None  # 设置标题的y轴位置为None
    fig, ax = plt.subplots()  # 创建一个新的图和坐标轴
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)  # 设置x轴的ticks参数
    ax.set_title('xlabel top')  # 设置坐标轴标题为'xlabel top'
    fig.canvas.draw()  # 绘制图形
    assert ax.title.get_position()[1] > 1.04  # 断言标题的y轴位置是否大于1.04


@pytest.mark.parametrize(
    'left, center', [
        ('left', ''),
        ('', 'center'),
        ('left', 'center')
    ], ids=[
        'left title moved',
        'center title kept',
        'both titles aligned'
    ]
)
def test_title_above_offset(left, center):
    # 测试当标题与y轴偏移文本重叠时，标题是否移动。
    mpl.rcParams['axes.titley'] = None  # 设置标题的y轴位置为None
    fig, ax = plt.subplots()  # 创建一个新的图和坐标轴
    ax.set_ylim(1e11)  # 设置y轴的数据限制
    ax.set_title(left, loc='left')  # 设置左侧或中心的坐标轴标题
    ax.set_title(center)  # 设置中心的坐标轴标题
    fig.draw_without_rendering()  # 绘制图形但不进行渲染
    if left and not center:
        assert ax._left_title.get_position()[1] > 1.0  # 如果左侧标题存在且中心标题不存在，断言左侧标题的y轴位置大于1.0
    elif not left and center:
        assert ax.title.get_position()[1] == 1.0  # 如果左侧标题不存在且中心标题存在，断言标题的y轴位置为1.0
    else:
        yleft = ax._left_title.get_position()[1]  # 获取左侧标题的y轴位置
        ycenter = ax.title.get_position()[1]  # 获取中心标题的y轴位置
        assert yleft > 1.0  # 断言左侧标题的y轴位置大于1.0
        assert ycenter == yleft  # 断言中心标题的y轴位置与左侧标题相同


def test_title_no_move_off_page():
    # 如果坐标轴位于图形之外（例如在保存时被裁剪），确保不自动重新调整标题位置。
    mpl.rcParams['axes.titley'] = None  # 设置标题的y轴位置为None
    fig = plt.figure()  # 创建一个新的图形
    ax = fig.add_axes([0.1, -0.5, 0.8, 0.2])  # 在图形上添加一个坐标轴，位置为[0.1, -0.5, 0.8, 0.2]
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)  # 设置x轴的ticks参数
    tt = ax.set_title('Boo')  # 设置坐标轴标题为'Boo'
    fig.canvas.draw()  # 绘制图形
    assert tt.get_position()[1] == 1.0  # 断言标题的y轴位置为1.0


def test_title_inset_ax():
    # 标题应位于任何子坐标轴之上。
    mpl.rcParams['axes.titley'] = None  # 设置标题的y轴位置为None
    fig, ax = plt.subplots()  # 创建一个新的图和坐标轴
    ax.set_title('Title')  # 设置坐标轴标题为'Title'
    fig.draw_without_rendering()  # 绘制图形但不进行渲染
    assert ax.title.get_position()[1] == 1  # 断言标题的y轴位置为1
    ax.inset_axes([0, 1, 1, 0.1])  # 在坐标轴上插入一个子坐标轴
    fig.draw_without_rendering()  # 绘制图形但不进行渲染
    assert ax.title.get_position()[1] == 1.1  # 断言标题的y轴位置为1.1


def test_offset_label_color():
    # 测试问题6440
    fig, ax = plt.subplots()  # 创建一个新的图和坐标轴
    ax.plot([1.01e9, 1.02e9, 1.03e9])  # 在坐标轴上绘制数据
    ax.yaxis.set_tick_params(labelcolor='red')  # 设置y轴刻度标签的颜色为红色
    assert ax.yaxis.get_offset_text().get_color() == 'red'  # 断言y轴偏移文本的颜色为红色


def test_offset_text_visible():
    fig, ax = plt.subplots()  # 创建一个新的图和坐标轴
    ax.plot([1.01e9, 1.02e9, 1.03e9])  # 在坐标轴上绘制数据
    ax.yaxis.set_tick_params(label1On=False, label2On=True)  # 设置y轴刻度参数
    assert ax.yaxis.get_offset_text().get_visible()  # 断言y轴偏移文本可见
    ax.yaxis.set_tick_params(label2On=False)  # 设置y轴第二个刻度不可见
    assert not ax.yaxis.get_offset_text().get_visible()  # 断言y轴偏移文本不可见


def test_large_offset():
    fig, ax = plt.subplots()  # 创建一个新的图和坐标轴
    ax.plot((1 + np.array([0, 1.e-12])) * 1.e27)  # 在坐标轴上绘制数据
    # 绘制图形并刷新画布
    fig.canvas.draw()
def test_barb_units():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 创建日期时间列表，每10分钟一个时间点
    dates = [datetime.datetime(2017, 7, 15, 18, i) for i in range(0, 60, 10)]
    # 在y轴上创建线性空间
    y = np.linspace(0, 5, len(dates))
    # 创建风羽图的u和v分量
    u = v = np.linspace(0, 50, len(dates))
    # 绘制风羽图
    ax.barbs(dates, y, u, v)


def test_quiver_units():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 创建日期时间列表，每10分钟一个时间点
    dates = [datetime.datetime(2017, 7, 15, 18, i) for i in range(0, 60, 10)]
    # 在y轴上创建线性空间
    y = np.linspace(0, 5, len(dates))
    # 创建矢量场图的u和v分量
    u = v = np.linspace(0, 50, len(dates))
    # 绘制矢量场图
    ax.quiver(dates, y, u, v)


def test_bar_color_cycle():
    # 将颜色转换函数to_rgb赋值给变量to_rgb
    to_rgb = mcolors.to_rgb
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 循环绘制五条线
    for j in range(5):
        ln, = ax.plot(range(3))
        # 在同一个轴上创建并绘制三个条形图
        brs = ax.bar(range(3), range(3))
        # 遍历每个条形对象，验证其颜色与线条对象的颜色相同
        for br in brs:
            assert to_rgb(ln.get_color()) == to_rgb(br.get_facecolor())


def test_tick_param_label_rotation():
    # 创建一个包含两个子图的新图形对象
    fix, (ax, ax2) = plt.subplots(1, 2)
    # 在第一个子图上绘制一条线
    ax.plot([0, 1], [0, 1])
    # 在第二个子图上绘制一条线
    ax2.plot([0, 1], [0, 1])
    # 设定第一个子图x轴和y轴刻度标签的旋转角度为75和90度
    ax.xaxis.set_tick_params(which='both', rotation=75)
    ax.yaxis.set_tick_params(which='both', rotation=90)
    # 验证第一个子图所有刻度标签的旋转角度为75度
    for text in ax.get_xticklabels(which='both'):
        assert text.get_rotation() == 75
    # 验证第一个子图所有刻度标签的旋转角度为90度
    for text in ax.get_yticklabels(which='both'):
        assert text.get_rotation() == 90

    # 设定第二个子图x轴和y轴刻度标签的旋转角度为53和35度
    ax2.tick_params(axis='x', labelrotation=53)
    ax2.tick_params(axis='y', rotation=35)
    # 验证第二个子图主要刻度标签的旋转角度为53和35度
    for text in ax2.get_xticklabels(which='major'):
        assert text.get_rotation() == 53
    for text in ax2.get_yticklabels(which='major'):
        assert text.get_rotation() == 35


@mpl.style.context('default')
def test_fillbetween_cycle():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 循环绘制三个区域填充图
    for j in range(3):
        cc = ax.fill_between(range(3), range(3))
        # 验证区域填充图的面颜色与预期颜色匹配
        target = mcolors.to_rgba(f'C{j}')
        assert tuple(cc.get_facecolors().squeeze()) == tuple(target)

    # 循环绘制三个垂直方向的区域填充图
    for j in range(3, 6):
        cc = ax.fill_betweenx(range(3), range(3))
        # 验证垂直方向区域填充图的面颜色与预期颜色匹配
        target = mcolors.to_rgba(f'C{j}')
        assert tuple(cc.get_facecolors().squeeze()) == tuple(target)

    # 验证填充颜色为黑色的面颜色与预期颜色匹配
    target = mcolors.to_rgba('k')
    for al in ['facecolor', 'facecolors', 'color']:
        cc = ax.fill_between(range(3), range(3), **{al: 'k'})
        assert tuple(cc.get_facecolors().squeeze()) == tuple(target)

    # 验证填充颜色为黑色的边框颜色与预期颜色匹配
    edge_target = mcolors.to_rgba('k')
    for j, el in enumerate(['edgecolor', 'edgecolors'], start=6):
        cc = ax.fill_between(range(3), range(3), **{el: 'k'})
        face_target = mcolors.to_rgba(f'C{j}')
        assert tuple(cc.get_facecolors().squeeze()) == tuple(face_target)
        assert tuple(cc.get_edgecolors().squeeze()) == tuple(edge_target)


def test_log_margins():
    # 设定图形参数，使得轴限制模式为数据模式
    plt.rcParams['axes.autolimit_mode'] = 'data'
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 设定x轴边缘百分比为0.05
    margin = 0.05
    ax.set_xmargin(margin)
    # 在x轴上绘制半对数图形
    ax.semilogx([10, 100], [10, 100])
    # 获取x轴当前的数据限制
    xlim0, xlim1 = ax.get_xlim()
    # 获取x轴的坐标变换对象
    transform = ax.xaxis.get_transform()
    # 将数据限制转换为坐标系中的值
    xlim0t, xlim1t = transform.transform([xlim0, xlim1])
    x0t, x1t = transform.transform([10, 100])
    # 计算数据限制的偏移量
    delta = (x1t - x0t) * margin
    # 验证新的数据限制与预期的转换值匹配
    assert_allclose([xlim0t + delta, xlim1t - delta], [x0t, x1t])
def test_color_length_mismatch():
    N = 5
    x, y = np.arange(N), np.arange(N)
    colors = np.arange(N+1)  # 创建一个长度为 N+1 的数组作为颜色数据
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴对象
    with pytest.raises(ValueError):  # 检查是否会引发 ValueError 异常
        ax.scatter(x, y, c=colors)  # 绘制散点图，使用颜色数据
    with pytest.warns(match="argument looks like a single numeric RGB"):  # 检查是否会产生警告信息
        ax.scatter(x, y, c=(0.5, 0.5, 0.5))  # 绘制散点图，使用单个 RGB 颜色
    ax.scatter(x, y, c=[(0.5, 0.5, 0.5)] * N)  # 绘制散点图，使用相同的 RGB 颜色数据


def test_eventplot_legend():
    plt.eventplot([1.0], label='Label')  # 绘制事件图
    plt.legend()  # 显示图例


@pytest.mark.parametrize('err, args, kwargs, match', (
        (ValueError, [[1]], {'lineoffsets': []}, 'lineoffsets cannot be empty'),  # 检查 lineoffsets 参数为空时是否会引发 ValueError 异常
        (ValueError, [[1]], {'linelengths': []}, 'linelengths cannot be empty'),  # 检查 linelengths 参数为空时是否会引发 ValueError 异常
        (ValueError, [[1]], {'linewidths': []}, 'linewidths cannot be empty'),  # 检查 linewidths 参数为空时是否会引发 ValueError 异常
        (ValueError, [[1]], {'linestyles': []}, 'linestyles cannot be empty'),  # 检查 linestyles 参数为空时是否会引发 ValueError 异常
        (ValueError, [[1]], {'alpha': []}, 'alpha cannot be empty'),  # 检查 alpha 参数为空时是否会引发 ValueError 异常
        (ValueError, [1], {}, 'positions must be one-dimensional'),  # 检查 positions 参数不是一维数组时是否会引发 ValueError 异常
        (ValueError, [[1]], {'lineoffsets': [1, 2]}, 'lineoffsets and positions are unequal sized sequences'),  # 检查 lineoffsets 和 positions 数组大小不一致时是否会引发 ValueError 异常
        (ValueError, [[1]], {'linelengths': [1, 2]}, 'linelengths and positions are unequal sized sequences'),  # 检查 linelengths 和 positions 数组大小不一致时是否会引发 ValueError 异常
        (ValueError, [[1]], {'linewidths': [1, 2]}, 'linewidths and positions are unequal sized sequences'),  # 检查 linewidths 和 positions 数组大小不一致时是否会引发 ValueError 异常
        (ValueError, [[1]], {'linestyles': [1, 2]}, 'linestyles and positions are unequal sized sequences'),  # 检查 linestyles 和 positions 数组大小不一致时是否会引发 ValueError 异常
        (ValueError, [[1]], {'alpha': [1, 2]}, 'alpha and positions are unequal sized sequences'),  # 检查 alpha 和 positions 数组大小不一致时是否会引发 ValueError 异常
        (ValueError, [[1]], {'colors': [1, 2]}, 'colors and positions are unequal sized sequences'),  # 检查 colors 和 positions 数组大小不一致时是否会引发 ValueError 异常
))
def test_eventplot_errors(err, args, kwargs, match):
    with pytest.raises(err, match=match):  # 检查是否会引发特定类型的异常，并且异常信息匹配指定的字符串
        plt.eventplot(*args, **kwargs)  # 绘制事件图


def test_bar_broadcast_args():
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴对象
    # 检查所有柱的高度都相同时的柱状图绘制
    ax.bar(range(4), 1)  # 绘制柱状图，所有柱的高度为 1
    # 检查所有柱的宽度和左侧位置都相同时的水平柱状图绘制
    ax.barh(0, 1, left=range(4), height=1)  # 绘制水平柱状图，所有柱的宽度为 1，左侧位置依次增加
    # 检查边框颜色是否可以广播
    rect1, rect2 = ax.bar([0, 1], [0, 1], edgecolor=(.1, .2, .3, .4))  # 绘制带有指定边框颜色的柱状图，并获取每个柱的对象
    assert rect1.get_edgecolor() == rect2.get_edgecolor() == (.1, .2, .3, .4)  # 断言两个柱的边框颜色相同


def test_invalid_axis_limits():
    plt.plot([0, 1], [0, 1])  # 绘制简单的线图
    with pytest.raises(ValueError):  # 检查是否会引发 ValueError 异常
        plt.xlim(np.nan)  # 尝试设置不合法的 x 轴限制
    with pytest.raises(ValueError):  # 检查是否会引发 ValueError 异常
        plt.xlim(np.inf)  # 尝试设置不合法的 x 轴限制
    with pytest.raises(ValueError):  # 检查是否会引发 ValueError 异常
        plt.ylim(np.nan)  # 尝试设置不合法的 y 轴限制
    with pytest.raises(ValueError):  # 检查是否会引发 ValueError 异常
        plt.ylim(np.inf)  # 尝试设置不合法的 y 轴限制


# Test all 4 combinations of logs/symlogs for minorticks_on()
@pytest.mark.parametrize('xscale', ['symlog', 'log'])
@pytest.mark.parametrize('yscale', ['symlog', 'log'])
def test_minorticks_on(xscale, yscale):
    ax = plt.subplot()  # 创建一个新的子图对象
    ax.plot([1, 2, 3, 4])  # 绘制简单的线图
    ax.set_xscale(xscale)  # 设置 x 轴的刻度类型
    ax.set_yscale(yscale)  # 设置 y 轴的刻度类型
    ax.minorticks_on()  # 打开次要刻度显示


def test_twinx_knows_limits():
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴对象

    ax.axvspan(1, 2)  # 在 x 轴上绘制一个垂直带区域
    # 在同一个图上创建一个新的共享 X 轴的轴对象
    xtwin = ax.twinx()
    # 在 xtwin 轴上绘制一条线段，起点 (0, 1)，终点 (0.5, 2)
    xtwin.plot([0, 0.5], [1, 2])

    # 创建一个新的图形 fig2 和它的轴对象 ax2
    fig2, ax2 = plt.subplots()

    # 在 ax2 轴上绘制一个垂直于 x 轴的色块，横跨 x 轴上从 1 到 2 的区间
    ax2.axvspan(1, 2)
    # 在 ax2 轴上绘制一条线段，起点 (0, 1)，终点 (0.5, 2)
    ax2.plot([0, 0.5], [1, 2])

    # 断言 xtwin 轴和 ax2 轴的视图限制（view limit）的 x 轴间隔是相等的
    assert_array_equal(xtwin.viewLim.intervalx, ax2.viewLim.intervalx)
def test_zero_linewidth():
    # 检查设置零线宽是否会引发错误
    plt.plot([0, 1], [0, 1], ls='--', lw=0)


def test_empty_errorbar_legend():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制一个误差条，指定x误差为空，添加标签'empty y'
    ax.errorbar([], [], xerr=[], label='empty y')
    # 在轴上绘制一个误差条，指定y误差为空，添加标签'empty x'
    ax.errorbar([], [], yerr=[], label='empty x')
    # 添加图例到轴
    ax.legend()


@check_figures_equal(extensions=["png"])
def test_plot_decimal(fig_test, fig_ref):
    # 创建一个包含Decimal输入的x坐标数组
    x0 = np.arange(-10, 10, 0.3)
    # 计算对应的y坐标数组，使用Decimal对象
    y0 = [5.2 * x ** 3 - 2.1 * x ** 2 + 7.34 * x + 4.5 for x in x0]
    # 将x0转换为Decimal对象的数组
    x = [Decimal(i) for i in x0]
    # 将y0转换为Decimal对象的数组
    y = [Decimal(i) for i in y0]
    # 在测试图上创建子图，并绘制线图，使用Decimal对象作为输入
    fig_test.subplots().plot(x, y)
    # 在参考图上创建子图，并绘制线图，使用浮点数作为输入
    fig_ref.subplots().plot(x0, y0)


# pdf和svg测试在Travis的旧版本gs和inkscape中失败。
@check_figures_equal(extensions=["png"])
def test_markerfacecolor_none_alpha(fig_test, fig_ref):
    # 在测试图上创建子图，并绘制带有透明度的未填充的标记
    fig_test.subplots().plot(0, "o", mfc="none", alpha=.5)
    # 在参考图上创建子图，并绘制带有白色填充和透明度的标记
    fig_ref.subplots().plot(0, "o", mfc="w", alpha=.5)


def test_tick_padding_tightbbox():
    """测试当轴关闭时，刻度间距是否被关闭"""
    # 设置xtick和ytick的方向为'out'
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 获取紧凑包围框，使用绘图区域的渲染器
    bb = ax.get_tightbbox(fig.canvas.get_renderer())
    # 关闭轴
    ax.axis('off')
    # 再次获取紧凑包围框，使用绘图区域的渲染器
    bb2 = ax.get_tightbbox(fig.canvas.get_renderer())
    # 断言第二个紧凑包围框的左下角比第一个的左下角更靠右和更高
    assert bb.x0 < bb2.x0
    assert bb.y0 < bb2.y0


def test_inset():
    """
    确保inset_ax参数确实是可选的
    """
    dx, dy = 0.05, 0.05
    # 生成x和y边界的2D网格
    y, x = np.mgrid[slice(1, 5 + dy, dy),
                    slice(1, 5 + dx, dx)]
    # 计算z值
    z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制伪彩色图
    ax.pcolormesh(x, y, z[:-1, :-1])
    # 设置轴的长宽比为1
    ax.set_aspect(1.)
    # 应用轴的长宽比设置
    ax.apply_aspect()
    # 为了使下面的绘图正常工作，我们需要应用轴的长宽比设置。

    # 定义x和y的限制范围
    xlim = [1.5, 2.15]
    ylim = [2, 2.5]

    # 定义矩形的位置和大小
    rect = [xlim[0], ylim[0], xlim[1] - xlim[0], ylim[1] - ylim[0]]

    # 指示轴的内嵌，指定边界矩形
    rec, connectors = ax.indicate_inset(bounds=rect)
    # 断言连接器为None
    assert connectors is None
    # 更新图形的绘图区域
    fig.canvas.draw()
    # 定义期望的矩形顶点数组
    xx = np.array([[1.5, 2.],
                   [2.15, 2.5]])
    # 断言实际矩形的顶点与期望的顶点数组相等
    assert np.all(rec.get_bbox().get_points() == xx)


def test_zoom_inset():
    dx, dy = 0.05, 0.05
    # 生成x和y边界的2D网格
    y, x = np.mgrid[slice(1, 5 + dy, dy),
                    slice(1, 5 + dx, dx)]
    # 计算z值
    z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)

    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制伪彩色图
    ax.pcolormesh(x, y, z[:-1, :-1])
    # 设置轴的长宽比为1
    ax.set_aspect(1.)
    # 应用轴的长宽比设置
    ax.apply_aspect()
    # 为了使下面的绘图正常工作，我们需要应用轴的长宽比设置。

    # 创建一个内嵌轴，指定位置和大小
    axin1 = ax.inset_axes([0.7, 0.7, 0.35, 0.35])
    # 在内嵌轴上重新绘制数据
    axin1.pcolormesh(x, y, z[:-1, :-1])
    # 设置内嵌轴的x轴限制
    axin1.set_xlim([1.5, 2.15])
    # 设置内嵌轴的y轴限制
    axin1.set_ylim([2, 2.5])
    # 应用主轴的长宽比到内嵌轴
    axin1.set_aspect(ax.get_aspect())

    # 指示缩放内嵌的轴
    rec, connectors = ax.indicate_inset_zoom(axin1)
    # 断言连接器的数量为4
    assert len(connectors) == 4
    # 更新图形的绘图区域
    fig.canvas.draw()
    # 创建一个 NumPy 数组 xx，包含两个二维数组作为其元素
    xx = np.array([[1.5,  2.],
                   [2.15, 2.5]])
    
    # 使用断言确保 rec 对象的边界框的顶点坐标等于数组 xx 中的值
    assert np.all(rec.get_bbox().get_points() == xx)
    
    # 重新赋值给 xx 一个新的 NumPy 数组，包含两个二维数组作为其元素
    xx = np.array([[0.6325, 0.692308],
                   [0.8425, 0.907692]])
    
    # 使用 np.testing.assert_allclose 函数检查 axin1 对象的位置的顶点坐标近似等于数组 xx 中的值，
    # 设置相对误差容差 rtol=1e-4
    np.testing.assert_allclose(axin1.get_position().get_points(), xx, rtol=1e-4)
@image_comparison(['inset_polar.png'], remove_text=True, style='mpl20')
def test_inset_polar():
    # 创建一个包含主图和极坐标子图的图形和坐标轴对象
    _, ax = plt.subplots()
    # 在主图中创建一个极坐标子图，并返回子图对象
    axins = ax.inset_axes([0.5, 0.1, 0.45, 0.45], polar=True)
    # 断言子图对象是极坐标子图类型
    assert isinstance(axins, PolarAxes)

    # 在主图中绘制极坐标图
    r = np.arange(0, 2, 0.01)
    theta = 2 * np.pi * r
    ax.plot(theta, r)
    # 在极坐标子图中绘制极坐标图
    axins.plot(theta, r)


def test_inset_projection():
    # 创建一个包含主图的图形和坐标轴对象
    _, ax = plt.subplots()
    # 在主图中创建一个投影为'hammer'的插入坐标轴，并返回插入坐标轴对象
    axins = ax.inset_axes([0.2, 0.2, 0.3, 0.3], projection="hammer")
    # 断言插入坐标轴对象是'hammer'类型的投影
    assert isinstance(axins, HammerAxes)


def test_inset_subclass():
    # 创建一个包含主图的图形和坐标轴对象
    _, ax = plt.subplots()
    # 在主图中创建一个指定为AA.Axes类的插入坐标轴，并返回插入坐标轴对象
    axins = ax.inset_axes([0.2, 0.2, 0.3, 0.3], axes_class=AA.Axes)
    # 断言插入坐标轴对象是AA.Axes类的实例
    assert isinstance(axins, AA.Axes)


@pytest.mark.parametrize('x_inverted', [False, True])
@pytest.mark.parametrize('y_inverted', [False, True])
def test_indicate_inset_inverted(x_inverted, y_inverted):
    """
    Test that the inset lines are correctly located with inverted data axes.
    """
    # 创建一个包含两个子图的图形和坐标轴对象
    fig, (ax1, ax2) = plt.subplots(1, 2)

    x = np.arange(10)
    ax1.plot(x, x, 'o')
    # 如果x轴被反转，则反转x轴
    if x_inverted:
        ax1.invert_xaxis()
    # 如果y轴被反转，则反转y轴
    if y_inverted:
        ax1.invert_yaxis()

    # 在ax1中指示一个矩形区域，并返回矩形对象和边界对象
    rect, bounds = ax1.indicate_inset([2, 2, 5, 4], ax2)
    lower_left, upper_left, lower_right, upper_right = bounds

    # 根据坐标轴反转状态计算符号
    sign_x = -1 if x_inverted else 1
    sign_y = -1 if y_inverted else 1
    # 断言插入区域的边界坐标符合预期
    assert sign_x * (lower_right.xy2[0] - lower_left.xy2[0]) > 0
    assert sign_x * (upper_right.xy2[0] - upper_left.xy2[0]) > 0
    assert sign_y * (upper_left.xy2[1] - lower_left.xy2[1]) > 0
    assert sign_y * (upper_right.xy2[1] - lower_right.xy2[1]) > 0


def test_set_position():
    # 创建一个包含主图的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 设置坐标轴的纵横比为3
    ax.set_aspect(3.)
    # 设置坐标轴的位置和大小，对所有的坐标轴生效
    ax.set_position([0.1, 0.1, 0.4, 0.4], which='both')
    # 断言坐标轴的宽度符合预期
    assert np.allclose(ax.get_position().width, 0.1)
    # 修改纵横比为2
    ax.set_aspect(2.)
    # 仅对原始坐标轴设置位置和大小
    ax.set_position([0.1, 0.1, 0.4, 0.4], which='original')
    # 断言坐标轴的宽度符合预期
    assert np.allclose(ax.get_position().width, 0.15)
    # 激活当前坐标轴，并设置位置和大小
    ax.set_aspect(3.)
    ax.set_position([0.1, 0.1, 0.4, 0.4], which='active')
    # 断言坐标轴的宽度符合预期
    assert np.allclose(ax.get_position().width, 0.1)


def test_spines_properbbox_after_zoom():
    # 创建一个包含主图的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 获取底部轴脊柱的边界框
    bb = ax.spines.bottom.get_window_extent(fig.canvas.get_renderer())
    # 执行缩放操作后获取底部轴脊柱的新边界框
    ax._set_view_from_bbox((320, 320, 500, 500), 'in',
                           None, False, False)
    bb2 = ax.spines.bottom.get_window_extent(fig.canvas.get_renderer())
    # 断言两个边界框的点坐标相似度符合预期
    np.testing.assert_allclose(bb.get_points(), bb2.get_points(), rtol=1e-6)


def test_limits_after_scroll_zoom():
    # 创建一个包含主图的图形和坐标轴对象
    fig, ax = plt.subplots()
    #
    xlim = (-0.5, 0.5)
    ylim = (-1, 2)
    # 设置x轴和y轴的限制
    ax.set_xlim(xlim)
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    # 模拟滚动缩放操作
    # 缩放因子为1，数值微小变化
    ax._set_view_from_bbox((200, 200, 1.))
    # 断言x轴和y轴的限制符合预期
    np.testing.assert_allclose(xlim, ax.get_xlim(), atol=1e-16)
    np.testing.assert_allclose(ylim, ax.get_ylim(), atol=1e-16)

    # 进行放大操作
    ax._set_view_from_bbox((200, 200, 2.))
    # Hard-coded values
    # 设置新的 x 轴和 y 轴的范围
    new_xlim = (-0.3790322580645161, 0.12096774193548387)
    new_ylim = (-0.40625, 1.09375)

    # 获取当前轴对象的 x 轴和 y 轴的当前范围
    res_xlim = ax.get_xlim()
    res_ylim = ax.get_ylim()

    # 断言当前 x 轴和 y 轴的范围差值接近于指定值
    np.testing.assert_allclose(res_xlim[1] - res_xlim[0], 0.5)
    np.testing.assert_allclose(res_ylim[1] - res_ylim[0], 1.5)

    # 断言新设置的 x 轴和 y 轴的范围与当前轴对象的范围接近，允许误差为 1e-16
    np.testing.assert_allclose(new_xlim, res_xlim, atol=1e-16)
    np.testing.assert_allclose(new_ylim, res_ylim)

    # 缩小视图范围，期望结果与之前相同，但可能存在数值问题
    ax._set_view_from_bbox((200, 200, 0.5))
    res_xlim = ax.get_xlim()
    res_ylim = ax.get_ylim()

    # 断言当前 x 轴和 y 轴的范围差值接近于指定值
    np.testing.assert_allclose(res_xlim[1] - res_xlim[0], 1)
    np.testing.assert_allclose(res_ylim[1] - res_ylim[0], 3)

    # 断言之前设置的 x 轴和 y 轴的范围与当前轴对象的范围接近，允许误差为 1e-16
    np.testing.assert_allclose(xlim, res_xlim, atol=1e-16)
    np.testing.assert_allclose(ylim, res_ylim, atol=1e-16)
def test_gettightbbox_ignore_nan():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 移除图形中的刻度和标题
    remove_ticks_and_titles(fig)
    # 在轴上添加文本，其中包含 NaN 值
    ax.text(np.nan, 1, 'Boo')
    # 获取图形的渲染器对象
    renderer = fig.canvas.get_renderer()
    # 使用渲染器计算轴的紧凑边界框，并断言其宽度接近496
    np.testing.assert_allclose(ax.get_tightbbox(renderer).width, 496)


def test_scatter_series_non_zero_index(pd):
    # 创建一个具有非零索引的 Series
    ids = range(10, 18)
    x = pd.Series(np.random.uniform(size=8), index=ids)
    y = pd.Series(np.random.uniform(size=8), index=ids)
    c = pd.Series([1, 1, 1, 1, 1, 0, 0, 0], index=ids)
    # 绘制散点图
    plt.scatter(x, y, c)


def test_scatter_empty_data():
    # 确保下面的操作不会引发异常
    # 绘制空的散点图
    plt.scatter([], [])
    # 绘制空的散点图，指定大小和颜色参数为空列表
    plt.scatter([], [], s=[], c=[])


@image_comparison(['annotate_across_transforms.png'], style='mpl20', remove_text=True,
                  tol=0.025 if platform.machine() == 'arm64' else 0)
def test_annotate_across_transforms():
    # 创建一个新的图形和轴对象，绘制指定的函数图像
    x = np.linspace(0, 10, 200)
    y = np.exp(-x) * np.sin(x)

    fig, ax = plt.subplots(figsize=(3.39, 3))
    ax.plot(x, y)
    # 在轴的内嵌坐标系中添加标注，包含箭头注释
    axins = ax.inset_axes([0.4, 0.5, 0.3, 0.3])
    axins.set_aspect(0.2)
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    ax.annotate("", xy=(x[150], y[150]), xycoords=ax.transData,
                xytext=(1, 0), textcoords=axins.transAxes,
                arrowprops=dict(arrowstyle="->"))


class _Translation(mtransforms.Transform):
    input_dims = 1
    output_dims = 1

    def __init__(self, dx):
        self.dx = dx

    def transform(self, values):
        # 对输入值进行平移变换
        return values + self.dx

    def inverted(self):
        # 返回反变换对象
        return _Translation(-self.dx)


@image_comparison(['secondary_xy.png'], style='mpl20',
                  tol=0.027 if platform.machine() == 'arm64' else 0)
def test_secondary_xy():
    # 创建包含两个子图的图形对象
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    def invert(x):
        # 定义一个反转函数，处理除零错误
        with np.errstate(divide='ignore'):
            return 1 / x

    for nn, ax in enumerate(axs):
        ax.plot(np.arange(2, 11), np.arange(2, 11))
        # 根据子图索引选择主轴的辅助轴
        if nn == 0:
            secax = ax.secondary_xaxis
        else:
            secax = ax.secondary_yaxis

        # 添加多个辅助轴，每个轴使用不同的函数变换
        secax(0.2, functions=(invert, invert))
        secax(0.4, functions=(lambda x: 2 * x, lambda x: x / 2))
        secax(0.6, functions=(lambda x: x**2, lambda x: x**(1/2)))
        secax(0.8)
        secax("top" if nn == 0 else "right", functions=_Translation(2))
        secax(6.25, transform=ax.transData)


def test_secondary_fail():
    # 创建一个包含单个子图的图形对象，并绘制函数图像
    fig, ax = plt.subplots()
    ax.plot(np.arange(2, 11), np.arange(2, 11))
    
    # 使用 pytest 断言检查在不同情况下辅助轴的预期错误
    with pytest.raises(ValueError):
        ax.secondary_xaxis(0.2, functions=(lambda x: 1 / x))
    with pytest.raises(ValueError):
        ax.secondary_xaxis('right')
    with pytest.raises(ValueError):
        ax.secondary_yaxis('bottom')
    with pytest.raises(TypeError):
        ax.secondary_xaxis(0.2, transform='error')


def test_secondary_resize():
    # 创建一个包含单个子图的大尺寸图形对象，并绘制函数图像
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(2, 11), np.arange(2, 11))
    # 定义一个函数 invert，用于计算输入参数的倒数，忽略除以零的错误
    def invert(x):
        # 设置 numpy 的错误状态，忽略除以零的错误
        with np.errstate(divide='ignore'):
            # 返回输入参数 x 的倒数
            return 1 / x

    # 在图形 ax 上添加一个副次 X 轴，位置在顶部，并使用 invert 函数作为转换函数
    ax.secondary_xaxis('top', functions=(invert, invert))
    
    # 绘制图形对象 fig 的画布
    fig.canvas.draw()
    
    # 设置图形对象 fig 的尺寸为 7x4 英寸
    fig.set_size_inches((7, 4))
    
    # 断言当前轴 ax 的位置范围是否与指定的数值列表相匹配
    assert_allclose(ax.get_position().extents, [0.125, 0.1, 0.9, 0.9])
# 定义一个测试函数，用于测试次要的次要 x 轴设置
def test_secondary_minorloc():
    # 创建一个新的图形和坐标系，设置图形大小为 10x5
    fig, ax = plt.subplots(figsize=(10, 5))
    # 在坐标系上绘制一条线，x 轴从 2 到 11，y 轴从 2 到 11
    ax.plot(np.arange(2, 11), np.arange(2, 11))

    # 定义一个内部函数 invert，用于计算输入的倒数
    def invert(x):
        # 忽略除以零的错误，返回倒数结果
        with np.errstate(divide='ignore'):
            return 1 / x

    # 创建一个次要 x 轴，位置在顶部，使用 invert 函数来转换坐标
    secax = ax.secondary_xaxis('top', functions=(invert, invert))
    # 断言次要 x 轴的次要定位器是 NullLocator 类型
    assert isinstance(secax._axis.get_minor_locator(),
                      mticker.NullLocator)
    # 打开次要刻度
    secax.minorticks_on()
    # 断言次要 x 轴的次要定位器是 AutoMinorLocator 类型
    assert isinstance(secax._axis.get_minor_locator(),
                      mticker.AutoMinorLocator)
    # 设置主 x 轴为对数刻度，并重新绘制图形
    ax.set_xscale('log')
    plt.draw()
    # 断言次要 x 轴的次要定位器是 LogLocator 类型
    assert isinstance(secax._axis.get_minor_locator(),
                      mticker.LogLocator)
    # 设置主 x 轴为线性刻度，并重新绘制图形
    ax.set_xscale('linear')
    plt.draw()
    # 断言次要 x 轴的次要定位器是 NullLocator 类型
    assert isinstance(secax._axis.get_minor_locator(),
                      mticker.NullLocator)


# 定义一个测试函数，用于测试次要 x 轴的格式化器
def test_secondary_formatter():
    # 创建一个新的图形和坐标系
    fig, ax = plt.subplots()
    # 设置主 x 轴为对数刻度
    ax.set_xscale("log")
    # 创建一个次要 x 轴，位置在顶部
    secax = ax.secondary_xaxis("top")
    # 设置次要 x 轴的主刻度格式化器为 ScalarFormatter
    secax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # 绘制图形
    fig.canvas.draw()
    # 断言次要 x 轴的主刻度格式化器是 ScalarFormatter 类型
    assert isinstance(
        secax.xaxis.get_major_formatter(), mticker.ScalarFormatter)


# 定义一个测试函数，用于测试次要 x 轴的字符串表示形式
def test_secondary_repr():
    # 创建一个新的图形和坐标系
    fig, ax = plt.subplots()
    # 创建一个次要 x 轴，位置在顶部
    secax = ax.secondary_xaxis("top")
    # 断言次要 x 轴的字符串表示形式符合预期
    assert repr(secax) == '<SecondaryAxis: >'


# 使用图像比较功能比较生成的图像是否符合预期
@image_comparison(['axis_options.png'], remove_text=True, style='mpl20')
def test_axis_options():
    # 创建一个 2x3 的子图数组
    fig, axes = plt.subplots(2, 3)
    for i, option in enumerate(('scaled', 'tight', 'image')):
        # 在每个子图中绘制线段和圆圈，并根据选项调整坐标轴的显示
        # 对于 'scaled' 和 'image' 选项，圆圈应该看起来像一个圆
        # 高/窄纵横比
        axes[0, i].plot((1, 2), (1, 3.2))
        axes[0, i].axis(option)
        axes[0, i].add_artist(mpatches.Circle((1.5, 1.5), radius=0.5,
                                              facecolor='none', edgecolor='k'))
        # 低/宽纵横比
        axes[1, i].plot((1, 2.25), (1, 1.75))
        axes[1, i].axis(option)
        axes[1, i].add_artist(mpatches.Circle((1.5, 1.25), radius=0.25,
                                              facecolor='none', edgecolor='k'))


def color_boxes(fig, ax):
    """
    用于测试下面测试的辅助函数，测试各种 Axes 元素的边界范围
    """
    # 绘制图形
    fig.canvas.draw()

    # 获取图形渲染器
    renderer = fig.canvas.get_renderer()
    bbaxis = []
    # 对每个轴（x 轴和 y 轴）获取紧凑边界框并绘制矩形
    for nn, axx in enumerate([ax.xaxis, ax.yaxis]):
        bb = axx.get_tightbbox(renderer)
        if bb:
            axisr = mpatches.Rectangle(
                (bb.x0, bb.y0), width=bb.width, height=bb.height,
                linewidth=0.7, edgecolor='y', facecolor="none", transform=None,
                zorder=3)
            fig.add_artist(axisr)
        bbaxis += [bb]

    bbspines = []
    # 遍历包含四个元素的列表，每个元素表示一个轴的边界名称：bottom, top, left, right
    for nn, a in enumerate(['bottom', 'top', 'left', 'right']):
        # 获取当前轴指定边界的窗口范围，并使用指定的渲染器进行计算
        bb = ax.spines[a].get_window_extent(renderer)
        # 创建一个矩形对象，表示当前边界的区域
        spiner = mpatches.Rectangle(
            (bb.x0, bb.y0), width=bb.width, height=bb.height,
            linewidth=0.7, edgecolor="green", facecolor="none", transform=None,
            zorder=3)
        # 将矩形对象添加到图形中
        fig.add_artist(spiner)
        # 将当前边界的窗口范围对象添加到列表中
        bbspines += [bb]

    # 获取整个轴的窗口范围
    bb = ax.get_window_extent()
    # 创建一个矩形对象，表示整个轴的区域
    rect2 = mpatches.Rectangle(
        (bb.x0, bb.y0), width=bb.width, height=bb.height,
        linewidth=1.5, edgecolor="magenta", facecolor="none", transform=None,
        zorder=2)
    # 将矩形对象添加到图形中
    fig.add_artist(rect2)
    # 将整个轴的窗口范围对象赋值给变量
    bbax = bb

    # 获取轴的紧凑边界框
    bb2 = ax.get_tightbbox(renderer)
    # 创建一个矩形对象，表示轴的紧凑边界框的区域
    rect2 = mpatches.Rectangle(
        (bb2.x0, bb2.y0), width=bb2.width, height=bb2.height,
        linewidth=3, edgecolor="red", facecolor="none", transform=None,
        zorder=1)
    # 将矩形对象添加到图形中
    fig.add_artist(rect2)
    # 将轴的紧凑边界框对象赋值给变量
    bbtb = bb2

    # 返回四个边界框的对象：轴边界列表、整个轴的边界、轴的紧凑边界框
    return bbspines, bbax, bbtb
def test_normal_axes():
    # 使用上下文管理器设置绘图参数，确保经典模式被禁用
    with rc_context({'_internal.classic_mode': False}):
        # 创建一个图形和轴对象，设置 DPI 为 200，尺寸为 6x6 英寸
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        # 绘制图形的画布
        fig.canvas.draw()
        # 关闭图形对象
        plt.close(fig)
        # 调用 color_boxes 函数获取边界框信息
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)

    # 测试轴边界框
    target = [
        [123.375, 75.88888888888886, 983.25, 33.0],
        [85.51388888888889, 99.99999999999997, 53.375, 993.0]
    ]
    # 遍历并比较边界框数组中的每个边界框对象
    for nn, b in enumerate(bbaxis):
        # 创建目标边界框对象
        targetbb = mtransforms.Bbox.from_bounds(*target[nn])
        # 断言当前边界框对象的边界与目标边界框对象的边界近似相等
        assert_array_almost_equal(b.bounds, targetbb.bounds, decimal=2)

    target = [
        [150.0, 119.999, 930.0, 11.111],
        [150.0, 1080.0, 930.0, 0.0],
        [150.0, 119.9999, 11.111, 960.0],
        [1068.8888, 119.9999, 11.111, 960.0]
    ]
    # 遍历并比较边界框数组中的每个边界框对象
    for nn, b in enumerate(bbspines):
        # 创建目标边界框对象
        targetbb = mtransforms.Bbox.from_bounds(*target[nn])
        # 断言当前边界框对象的边界与目标边界框对象的边界近似相等，允许小数点后两位的误差
        assert_array_almost_equal(b.bounds, targetbb.bounds, decimal=2)

    # 创建目标边界框对象
    target = [150.0, 119.99999999999997, 930.0, 960.0]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    # 断言当前边界框对象的边界与目标边界框对象的边界近似相等，允许小数点后两位的误差
    assert_array_almost_equal(bbax.bounds, targetbb.bounds, decimal=2)

    # 创建目标边界框对象
    target = [85.5138, 75.88888, 1021.11, 1017.11]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    # 断言当前边界框对象的边界与目标边界框对象的边界近似相等，允许小数点后两位的误差
    assert_array_almost_equal(bbtb.bounds, targetbb.bounds, decimal=2)

    # 测试 get_position 方法与 get_window_extent 方法的回转一致性
    axbb = ax.get_position().transformed(fig.transFigure).bounds
    # 断言轴对象的位置边界与其窗口范围的边界近似相等，允许小数点后两位的误差
    assert_array_almost_equal(axbb, ax.get_window_extent().bounds, decimal=2)


def test_nodecorator():
    # 使用上下文管理器设置绘图参数，确保经典模式被禁用
    with rc_context({'_internal.classic_mode': False}):
        # 创建一个图形和轴对象，设置 DPI 为 200，尺寸为 6x6 英寸
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        # 绘制图形的画布
        fig.canvas.draw()
        # 设置轴对象的刻度标签为空列表
        ax.set(xticklabels=[], yticklabels=[])
        # 调用 color_boxes 函数获取边界框信息
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)

    # 测试轴边界框
    for nn, b in enumerate(bbaxis):
        # 断言边界框对象为空
        assert b is None

    target = [
        [150.0, 119.999, 930.0, 11.111],
        [150.0, 1080.0, 930.0, 0.0],
        [150.0, 119.9999, 11.111, 960.0],
        [1068.8888, 119.9999, 11.111, 960.0]
    ]
    # 遍历并比较边界框数组中的每个边界框对象
    for nn, b in enumerate(bbspines):
        # 创建目标边界框对象
        targetbb = mtransforms.Bbox.from_bounds(*target[nn])
        # 断言当前边界框对象的边界与目标边界框对象的边界近似相等，允许绝对误差在 1e-2 范围内
        assert_allclose(b.bounds, targetbb.bounds, atol=1e-2)

    # 创建目标边界框对象
    target = [150.0, 119.99999999999997, 930.0, 960.0]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    # 断言当前边界框对象的边界与目标边界框对象的边界近似相等，允许绝对误差在 1e-2 范围内
    assert_allclose(bbax.bounds, targetbb.bounds, atol=1e-2)

    # 创建目标边界框对象
    target = [150., 120., 930., 960.]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    # 断言当前边界框对象的边界与目标边界框对象的边界近似相等，允许绝对误差在 1e-2 范围内
    assert_allclose(bbtb.bounds, targetbb.bounds, atol=1e-2)


def test_displaced_spine():
    # 使用上下文管理器设置绘图参数，确保经典模式被禁用
    with rc_context({'_internal.classic_mode': False}):
        # 创建一个图形和轴对象，设置 DPI 为 200，尺寸为 6x6 英寸
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        # 设置轴对象的刻度标签为空列表
        ax.set(xticklabels=[], yticklabels=[])
        # 设置轴底部脊柱的位置偏移
        ax.spines.bottom.set_position(('axes', -0.1))
        # 绘制图形的画布
        fig.canvas.draw()
        # 调用 color_boxes 函数获取边界框信息
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
    # 定义一个包含多个目标边界框坐标的列表
    targets = [
        [150., 24., 930., 11.111111],  # 第一个目标边界框的坐标
        [150.0, 1080.0, 930.0, 0.0],   # 第二个目标边界框的坐标
        [150.0, 119.9999, 11.111, 960.0],  # 第三个目标边界框的坐标
        [1068.8888, 119.9999, 11.111, 960.0]  # 第四个目标边界框的坐标
    ]
    # 遍历每个目标边界框坐标和对应的 bbspines 对象
    for target, bbspine in zip(targets, bbspines):
        # 根据给定的坐标值创建一个 Bbox 对象
        targetbb = mtransforms.Bbox.from_bounds(*target)
        # 断言 bbspine 对象的边界框与目标边界框的边界在给定的容差范围内相等
        assert_allclose(bbspine.bounds, targetbb.bounds, atol=1e-2)

    # 定义一个单独的目标边界框坐标
    target = [150.0, 119.99999999999997, 930.0, 960.0]
    # 根据给定的坐标值创建一个 Bbox 对象
    targetbb = mtransforms.Bbox.from_bounds(*target)
    # 断言 bbax 对象的边界框与目标边界框的边界在给定的容差范围内相等
    assert_allclose(bbax.bounds, targetbb.bounds, atol=1e-2)

    # 定义另一个单独的目标边界框坐标
    target = [150., 24., 930., 1056.]
    # 根据给定的坐标值创建一个 Bbox 对象
    targetbb = mtransforms.Bbox.from_bounds(*target)
    # 断言 bbtb 对象的边界框与目标边界框的边界在给定的容差范围内相等
    assert_allclose(bbtb.bounds, targetbb.bounds, atol=1e-2)
def test_tickdirs():
    """
    Switch the tickdirs and make sure the bboxes switch with them
    """
    # 预设的目标边界框列表
    targets = [[[150.0, 120.0, 930.0, 11.1111],
                [150.0, 120.0, 11.111, 960.0]],
               [[150.0, 108.8889, 930.0, 11.111111111111114],
                [138.889, 120, 11.111, 960.0]],
               [[150.0, 114.44444444444441, 930.0, 11.111111111111114],
                [144.44444444444446, 119.999, 11.111, 960.0]]]
    
    # 枚举处理目标列表中的每个元素及其索引
    for dnum, dirs in enumerate(['in', 'out', 'inout']):
        # 设置绘图配置上下文，禁用经典模式
        with rc_context({'_internal.classic_mode': False}):
            # 创建具有指定 DPI 和尺寸的图形和轴
            fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
            # 设置刻度参数的方向
            ax.tick_params(direction=dirs)
            # 绘制图形画布
            fig.canvas.draw()
            # 调用函数获取颜色框的边界
            bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
            # 验证颜色框的边界是否与预期目标边界框接近
            for nn, num in enumerate([0, 2]):
                targetbb = mtransforms.Bbox.from_bounds(*targets[dnum][nn])
                assert_allclose(
                    bbspines[num].bounds, targetbb.bounds, atol=1e-2)


def test_minor_accountedfor():
    # 设置绘图配置上下文，禁用经典模式
    with rc_context({'_internal.classic_mode': False}):
        # 创建具有指定 DPI 和尺寸的图形和轴
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        # 绘制图形画布
        fig.canvas.draw()
        # 设置刻度参数，包括方向为外部
        ax.tick_params(which='both', direction='out')

        # 调用函数获取颜色框的边界
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
        # 预设的目标边界框列表
        targets = [[150.0, 108.88888888888886, 930.0, 11.111111111111114],
                   [138.8889, 119.9999, 11.1111, 960.0]]
        # 遍历预设目标边界框列表
        for n in range(2):
            targetbb = mtransforms.Bbox.from_bounds(*targets[n])
            # 验证颜色框的边界是否与预期目标边界框接近
            assert_allclose(
                bbspines[n * 2].bounds, targetbb.bounds, atol=1e-2)

        # 创建具有指定 DPI 和尺寸的图形和轴
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        # 绘制图形画布
        fig.canvas.draw()
        # 设置刻度参数，包括方向为外部
        ax.tick_params(which='both', direction='out')
        # 启用次要刻度线
        ax.minorticks_on()
        # 设置轴的次要刻度线长度
        ax.tick_params(axis='both', which='minor', length=30)
        # 绘制图形画布
        fig.canvas.draw()
        # 调用函数获取颜色框的边界
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
        # 预设的目标边界框列表
        targets = [[150.0, 36.66666666666663, 930.0, 83.33333333333334],
                   [66.6667, 120.0, 83.3333, 960.0]]

        # 遍历预设目标边界框列表
        for n in range(2):
            targetbb = mtransforms.Bbox.from_bounds(*targets[n])
            # 验证颜色框的边界是否与预期目标边界框接近
            assert_allclose(
                bbspines[n * 2].bounds, targetbb.bounds, atol=1e-2)


@check_figures_equal(extensions=["png"])
def test_axis_bool_arguments(fig_test, fig_ref):
    # 测试 False 和 "off" 是否相同
    fig_test.add_subplot(211).axis(False)
    fig_ref.add_subplot(211).axis("off")
    # 测试在 False 后设置 True 是否与 "on" 相同
    ax = fig_test.add_subplot(212)
    ax.axis(False)
    ax.axis(True)
    fig_ref.add_subplot(212).axis("on")


def test_axis_extent_arg():
    # 创建图形和轴
    fig, ax = plt.subplots()
    xmin = 5
    xmax = 10
    ymin = 15
    ymax = 20
    # 设置轴的限制范围，并返回当前的轴限制
    extent = ax.axis([xmin, xmax, ymin, ymax])

    # 验证文档字符串的描述是否正确
    assert tuple(extent) == (xmin, xmax, ymin, ymax)

    # 测试是否根据文档字符串设置了限制
    # 断言：验证当前图表对象的 X 轴限制范围是否与预期的 xmin 和 xmax 相同
    assert (xmin, xmax) == ax.get_xlim()
    
    # 断言：验证当前图表对象的 Y 轴限制范围是否与预期的 ymin 和 ymax 相同
    assert (ymin, ymax) == ax.get_ylim()
def test_axis_extent_arg2():
    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 定义坐标轴的范围
    xmin = 5
    xmax = 10
    ymin = 15
    ymax = 20
    # 设置坐标轴的范围并返回实际范围（extent）
    extent = ax.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    # 验证文档字符串的正确性
    assert tuple(extent) == (xmin, xmax, ymin, ymax)

    # 验证限制是否按照文档字符串设置
    assert (xmin, xmax) == ax.get_xlim()
    assert (ymin, ymax) == ax.get_ylim()


def test_hist_auto_bins():
    # 绘制直方图并获取自动计算的箱子边界
    _, bins, _ = plt.hist([[1, 2, 3], [3, 4, 5, 6]], bins='auto')
    assert bins[0] <= 1
    assert bins[-1] >= 6


def test_hist_nan_data():
    # 创建包含两个子图的图形对象
    fig, (ax1, ax2) = plt.subplots(2)

    data = [1, 2, 3]
    nan_data = data + [np.nan]

    # 在第一个子图中绘制普通数据的直方图
    bins, edges, _ = ax1.hist(data)
    # 在第二个子图中绘制包含 NaN 数据的直方图，并忽略无效值的错误
    with np.errstate(invalid='ignore'):
        nanbins, nanedges, _ = ax2.hist(nan_data)

    # 断言两个直方图的结果在数值上是相近的
    np.testing.assert_allclose(bins, nanbins)
    np.testing.assert_allclose(edges, nanedges)


def test_hist_range_and_density():
    # 绘制指定范围和密度的随机数据直方图
    _, bins, _ = plt.hist(np.random.rand(10), "auto",
                          range=(0, 1), density=True)
    assert bins[0] == 0
    assert bins[-1] == 1


def test_bar_errbar_zorder():
    # 检查误差线的 zorder 是否始终大于柱状图的 zorder
    fig, ax = plt.subplots()
    x = [1, 2, 3]
    # 创建包含误差条的柱状图，并指定 zorder 为 3
    barcont = ax.bar(x=x, height=x, yerr=x, capsize=5, zorder=3)

    # 获取误差线的相关对象
    data_line, caplines, barlinecols = barcont.errorbar.lines
    # 遍历每个柱状图对象
    for bar in barcont.patches:
        # 验证每个误差线的 zorder 是否大于柱状图的 zorder
        for capline in caplines:
            assert capline.zorder > bar.zorder
        for barlinecol in barlinecols:
            assert barlinecol.zorder > bar.zorder


def test_set_ticks_inverted():
    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 反转 x 轴
    ax.invert_xaxis()
    # 设置特定的 x 轴刻度
    ax.set_xticks([.3, .7])
    assert ax.get_xlim() == (1, 0)
    # 再次设置不同的 x 轴刻度
    ax.set_xticks([-1])
    assert ax.get_xlim() == (1, -1)


def test_aspect_nonlinear_adjustable_box():
    # 创建一个大小为 10x10 的正方形图形对象
    fig = plt.figure(figsize=(10, 10))

    # 在图形对象上添加一个子图
    ax = fig.add_subplot()
    # 绘制一条线，并设置特定的比例尺和数据范围
    ax.plot([.4, .6], [.4, .6])  # Set minpos to keep logit happy.
    ax.set(xscale="log", xlim=(1, 10),
           yscale="logit", ylim=(1/11, 1/1001),
           aspect=1, adjustable="box")
    ax.margins(0)
    # 计算子图的位置和大小，并验证其高宽比是否接近于 2
    pos = fig.transFigure.transform_bbox(ax.get_position())
    assert pos.height / pos.width == pytest.approx(2)


def test_aspect_nonlinear_adjustable_datalim():
    # 创建一个大小为 10x10 的正方形图形对象
    fig = plt.figure(figsize=(10, 10))

    # 在图形对象上添加一个具有特定位置和大小的坐标轴
    ax = fig.add_axes([.1, .1, .8, .8])  # Square.
    # 绘制一条线，并设置特定的比例尺和数据范围
    ax.plot([.4, .6], [.4, .6])  # Set minpos to keep logit happy.
    ax.set(xscale="log", xlim=(1, 100),
           yscale="logit", ylim=(1 / 101, 1 / 11),
           aspect=1, adjustable="datalim")
    ax.margins(0)
    ax.apply_aspect()

    # 验证坐标轴的 x 和 y 范围是否在数值上接近预期值
    assert ax.get_xlim() == pytest.approx([1*10**(1/2), 100/10**(1/2)])
    assert ax.get_ylim() == (1 / 101, 1 / 11)


def test_box_aspect():
    # 测试具有 box_aspect=1 的坐标轴是否具有相同的尺寸
    # 作为 aspect=1 和 adjustable="box" 的坐标轴

    fig1, ax1 = plt.subplots()
    # 在 ax1 图表上创建一个双 Y 轴的副图表 axtwin
    axtwin = ax1.twinx()
    
    # 在 axtwin 副图表上绘制一条直线，数据点为 [12, 344]
    axtwin.plot([12, 344])
    
    # 设置 ax1 图表的盒子纵横比为 1
    ax1.set_box_aspect(1)
    
    # 断言 ax1 图表的盒子纵横比是否为 1.0
    assert ax1.get_box_aspect() == 1.0
    
    # 创建一个新的图表 fig2 和其相应的轴 ax2
    fig2, ax2 = plt.subplots()
    
    # 设置 ax2 图表的边距为 0
    ax2.margins(0)
    
    # 在 ax2 图表上绘制一条线段，连接坐标点 [0, 6] 和 [2, 8]
    ax2.plot([0, 2], [6, 8])
    
    # 设置 ax2 图表的纵横比为 "equal"，并且调整为可调整的盒子
    ax2.set_aspect("equal", adjustable="box")
    
    # 在 fig1 图表上重新绘制所有的画布内容
    fig1.canvas.draw()
    
    # 在 fig2 图表上重新绘制所有的画布内容
    fig2.canvas.draw()
    
    # 获取 ax1 图表的位置信息，并赋值给 bb1 变量
    bb1 = ax1.get_position()
    
    # 获取 axtwin 副图表的位置信息，并赋值给 bbt 变量
    bbt = axtwin.get_position()
    
    # 获取 ax2 图表的位置信息，并赋值给 bb2 变量
    bb2 = ax2.get_position()
    
    # 断言 bb1 和 bb2 的范围是否完全相等
    assert_array_equal(bb1.extents, bb2.extents)
    
    # 断言 bbt 和 bb2 的范围是否完全相等
    assert_array_equal(bbt.extents, bb2.extents)
def test_box_aspect_custom_position():
    # 测试自定义位置和盒子纵横比的轴行为，无论设置顺序如何，表现应一致。

    # 创建第一个图和轴
    fig1, ax1 = plt.subplots()
    # 设置轴的位置
    ax1.set_position([0.1, 0.1, 0.9, 0.2])
    # 绘制第一个图的画布
    fig1.canvas.draw()
    # 设置轴的盒子纵横比
    ax1.set_box_aspect(1.)

    # 创建第二个图和轴
    fig2, ax2 = plt.subplots()
    # 设置轴的盒子纵横比
    ax2.set_box_aspect(1.)
    # 绘制第二个图的画布
    fig2.canvas.draw()
    # 设置轴的位置
    ax2.set_position([0.1, 0.1, 0.9, 0.2])

    # 再次绘制两个图的画布
    fig1.canvas.draw()
    fig2.canvas.draw()

    # 获取轴的位置矩形框
    bb1 = ax1.get_position()
    bb2 = ax2.get_position()

    # 断言两个轴的位置矩形框的四个坐标值应相等
    assert_array_equal(bb1.extents, bb2.extents)


def test_bbox_aspect_axes_init():
    # 测试可以在轴初始化时设置盒子纵横比，生成所有相等的正方形轴。

    # 创建图和多个子图轴，设置子图轴的盒子纵横比为1，并启用约束布局
    fig, axs = plt.subplots(2, 3, subplot_kw=dict(box_aspect=1),
                            constrained_layout=True)
    # 绘制图的画布
    fig.canvas.draw()
    # 获取渲染器
    renderer = fig.canvas.get_renderer()
    sizes = []
    # 遍历所有子图轴
    for ax in axs.flat:
        # 获取子图轴的窗口范围
        bb = ax.get_window_extent(renderer)
        # 扩展大小列表以包含宽度和高度
        sizes.extend([bb.width, bb.height])

    # 断言所有子图轴的宽度和高度应相等
    assert_allclose(sizes, sizes[0])


def test_set_aspect_negative():
    # 测试设置负值盒子纵横比应引发 ValueError 异常，错误消息为"must be finite and positive"。

    # 创建图和轴
    fig, ax = plt.subplots()
    # 使用 pytest 检查设置不同的非法盒子纵横比是否引发预期的异常
    with pytest.raises(ValueError, match="must be finite and positive"):
        ax.set_aspect(-1)
    with pytest.raises(ValueError, match="must be finite and positive"):
        ax.set_aspect(0)
    with pytest.raises(ValueError, match="must be finite and positive"):
        ax.set_aspect(np.inf)
    with pytest.raises(ValueError, match="must be finite and positive"):
        ax.set_aspect(-np.inf)


def test_redraw_in_frame():
    # 测试在框架中重新绘制轴。

    # 创建图和轴
    fig, ax = plt.subplots(1, 1)
    # 在轴上绘制线图
    ax.plot([1, 2, 3])
    # 绘制图的画布
    fig.canvas.draw()
    # 在当前帧中重新绘制轴
    ax.redraw_in_frame()


def test_invisible_axes_events():
    # 测试不可见的轴不响应事件...

    # 创建图和轴
    fig, ax = plt.subplots()
    # 断言事件在图的画布中的坐标(200, 200)不在任何轴内
    assert fig.canvas.inaxes((200, 200)) is not None
    # 设置轴不可见
    ax.set_visible(False)
    # 断言事件在图的画布中的坐标(200, 200)不在任何轴内
    assert fig.canvas.inaxes((200, 200)) is None


def test_xtickcolor_is_not_markercolor():
    # 测试 x 轴刻度线颜色不等于标记边缘颜色...

    # 设置全局参数，线的标记边缘颜色为白色
    plt.rcParams['lines.markeredgecolor'] = 'white'
    # 创建轴
    ax = plt.axes()
    # 获取 x 轴主刻度
    ticks = ax.xaxis.get_major_ticks()
    # 遍历每个刻度
    for tick in ticks:
        # 断言每个刻度的刻度线标记边缘颜色不是白色
        assert tick.tick1line.get_markeredgecolor() != 'white'


def test_ytickcolor_is_not_markercolor():
    # 测试 y 轴刻度线颜色不等于标记边缘颜色...

    # 设置全局参数，线的标记边缘颜色为白色
    plt.rcParams['lines.markeredgecolor'] = 'white'
    # 创建轴
    ax = plt.axes()
    # 获取 y 轴主刻度
    ticks = ax.yaxis.get_major_ticks()
    # 遍历每个刻度
    for tick in ticks:
        # 断言每个刻度的刻度线标记边缘颜色不是白色
        assert tick.tick1line.get_markeredgecolor() != 'white'


@pytest.mark.parametrize('axis', ('x', 'y'))
@pytest.mark.parametrize('auto', (True, False, None))
def test_unautoscale(axis, auto):
    # 测试取消自动缩放的功能

    # 创建图和轴
    fig, ax = plt.subplots()
    # 创建数据
    x = np.arange(100)
    y = np.linspace(-.1, .1, 100)
    # 在轴上绘制散点图
    ax.scatter(y, x)

    # 获取轴的自动缩放状态函数和设置函数，根据轴选择 x 或 y
    get_autoscale_on = getattr(ax, f'get_autoscale{axis}_on')
    set_lim = getattr(ax, f'set_{axis}lim')
    get_lim = getattr(ax, f'get_{axis}lim')

    # 记录自动缩放状态
    post_auto = get_autoscale_on() if auto is None else auto

    # 设置轴的限制和自动缩放状态
    set_lim((-0.5, 0.5), auto=auto)
    # 断言设置后的自动缩放状态与预期一致
    assert post_auto == get_autoscale_on()
    # 绘制图的画布
    fig.canvas.draw()
    # 断言轴的限制被正确设置
    assert_array_equal(get_lim(), (-0.5, 0.5))
# 使用装饰器检查两个图是否相等，扩展名为 "png"
@check_figures_equal(extensions=["png"])
def test_polar_interpolation_steps_variable_r(fig_test, fig_ref):
    # 向 fig_test 添加极坐标子图，并绘制直线 l，其中角度范围为 [0, np.pi/2]，半径范围为 [1, 2]
    l, = fig_test.add_subplot(projection="polar").plot([0, np.pi/2], [1, 2])
    # 设置直线 l 的插值步数为 100
    l.get_path()._interpolation_steps = 100
    # 在 fig_ref 中添加极坐标子图，并绘制曲线，角度从 0 到 np.pi/2，半径从 1 到 2，共 101 个点
    fig_ref.add_subplot(projection="polar").plot(
        np.linspace(0, np.pi/2, 101), np.linspace(1, 2, 101))


# 使用默认风格的上下文管理器来测试自动缩放的小数粘性
@mpl.style.context('default')
def test_autoscale_tiny_sticky():
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制一个高度为 1e-9 的条形
    ax.bar(0, 1e-9)
    # 绘制完成后刷新画布
    fig.canvas.draw()
    # 断言轴的 y 轴限制为 (0, 1.05e-9)
    assert ax.get_ylim() == (0, 1.05e-9)


# 测试 x 轴刻度线颜色不等于 x 轴刻度标签颜色
def test_xtickcolor_is_not_xticklabelcolor():
    # 设置 x 轴刻度线颜色为 'yellow'
    plt.rcParams['xtick.color'] = 'yellow'
    # 设置 x 轴刻度标签颜色为 'blue'
    plt.rcParams['xtick.labelcolor'] = 'blue'
    # 创建一个轴对象
    ax = plt.axes()
    # 获取主刻度
    ticks = ax.xaxis.get_major_ticks()
    # 遍历所有刻度
    for tick in ticks:
        # 断言刻度线的颜色为 'yellow'
        assert tick.tick1line.get_color() == 'yellow'
        # 断言刻度标签的颜色为 'blue'
        assert tick.label1.get_color() == 'blue'


# 测试 y 轴刻度线颜色不等于 y 轴刻度标签颜色
def test_ytickcolor_is_not_yticklabelcolor():
    # 设置 y 轴刻度线颜色为 'yellow'
    plt.rcParams['ytick.color'] = 'yellow'
    # 设置 y 轴刻度标签颜色为 'blue'
    plt.rcParams['ytick.labelcolor'] = 'blue'
    # 创建一个轴对象
    ax = plt.axes()
    # 获取主刻度
    ticks = ax.yaxis.get_major_ticks()
    # 遍历所有刻度
    for tick in ticks:
        # 断言刻度线的颜色为 'yellow'
        assert tick.tick1line.get_color() == 'yellow'
        # 断言刻度标签的颜色为 'blue'
        assert tick.label1.get_color() == 'blue'


# 测试 x 轴偏移文本颜色
def test_xaxis_offsetText_color():
    # 设置 x 轴刻度标签颜色为 'blue'
    plt.rcParams['xtick.labelcolor'] = 'blue'
    # 创建一个轴对象
    ax = plt.axes()
    # 断言 x 轴偏移文本的颜色为 'blue'
    assert ax.xaxis.offsetText.get_color() == 'blue'

    # 设置 x 轴刻度线颜色为 'yellow'，并将 x 轴刻度标签颜色设置为 'inherit'
    plt.rcParams['xtick.color'] = 'yellow'
    plt.rcParams['xtick.labelcolor'] = 'inherit'
    # 重新创建一个轴对象
    ax = plt.axes()
    # 断言 x 轴偏移文本的颜色为 'yellow'
    assert ax.xaxis.offsetText.get_color() == 'yellow'


# 测试 y 轴偏移文本颜色
def test_yaxis_offsetText_color():
    # 设置 y 轴刻度标签颜色为 'green'
    plt.rcParams['ytick.labelcolor'] = 'green'
    # 创建一个轴对象
    ax = plt.axes()
    # 断言 y 轴偏移文本的颜色为 'green'
    assert ax.yaxis.offsetText.get_color() == 'green'

    # 设置 y 轴刻度线颜色为 'red'，并将 y 轴刻度标签颜色设置为 'inherit'
    plt.rcParams['ytick.color'] = 'red'
    plt.rcParams['ytick.labelcolor'] = 'inherit'
    # 重新创建一个轴对象
    ax = plt.axes()
    # 断言 y 轴偏移文本的颜色为 'red'
    assert ax.yaxis.offsetText.get_color() == 'red'


# 使用装饰器参数化测试不同的标签大小
@pytest.mark.parametrize('size', [size for size in mfont_manager.font_scalings
                                  if size is not None] + [8, 10, 12])
@mpl.style.context('default')
def test_relative_ticklabel_sizes(size):
    # 设置 x 和 y 轴刻度标签大小为参数化的大小
    mpl.rcParams['xtick.labelsize'] = size
    mpl.rcParams['ytick.labelsize'] = size
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 绘制完成后刷新画布
    fig.canvas.draw()

    # 遍历 x 和 y 轴的主刻度
    for name, axis in zip(['x', 'y'], [ax.xaxis, ax.yaxis]):
        for tick in axis.get_major_ticks():
            # 断言刻度标签的大小与轴对象的刻度标签大小相等
            assert tick.label1.get_size() == axis._get_tick_label_size(name)


# 测试多子图自动缩放
def test_multiplot_autoscale():
    # 创建一个图形对象
    fig = plt.figure()
    # 创建两个子图，共享 x 轴
    ax1, ax2 = fig.subplots(2, 1, sharex='all')
    # 在第一个子图上绘制散点图
    ax1.scatter([1, 2, 3, 4], [2, 3, 2, 3])
    # 在第二个子图上绘制水平带
    ax2.axhspan(-5, 5)
    # 获取第一个子图的 x 轴限制
    xlim = ax1.get_xlim()
    # 断言 x 轴限制的范围与预期接近
    assert np.allclose(xlim, [0.5, 4.5])


# 测试共享不会链接位置
def test_sharing_does_not_link_positions():
    # 创建一个图形对象
    fig = plt.figure()
    # 添加一个 2x2 网格的子图，位置为 (2, 2, 1)
    ax0 = fig.add_subplot(221)
    # 添加一个位置在 (0.6, 0.6)，大小为 (0.3, 0.3) 的子图，并共享 x 轴
    ax1 = fig.add_axes([.6, .6, .3, .3], sharex=ax0)
    # 获取初始位置
    init_pos = ax1.get_position()
    # 调整子图之间的空白
    fig.subplots_adjust(left=0)
    # 断言子图的位置点与初始位置点相等
    assert (ax1.get_position().get_points() == init_pos.get_points()).all()


# 使用装饰器检查两个图是否相等，扩展名为 "pdf"
@check_figures_equal(extensions=["pdf"])
def test_2dcolor_plot(fig_test, fig_ref):
    # 创建一个包含三个元素的一维数组作为颜色值
    color = np.array([0.1, 0.2, 0.3])
    
    # 使用 `fig_test` 创建包含5个子图的图表对象
    axs = fig_test.subplots(5)
    
    # 在第一个子图上绘制一条线，使用颜色 `color`，颜色是一维数组展开后的结果
    axs[0].plot([1, 2], [1, 2], c=color.reshape(-1))
    
    # 在第二个子图上绘制散点图，使用颜色 `color`，颜色是一维数组展开后的结果
    with pytest.warns(match="argument looks like a single numeric RGB"):
        axs[1].scatter([1, 2], [1, 2], c=color.reshape(-1))
    
    # 在第三个子图上绘制阶梯图，使用颜色 `color`，颜色是一维数组展开后的结果
    axs[2].step([1, 2], [1, 2], c=color.reshape(-1))
    
    # 在第四个子图上绘制直方图，使用颜色 `color`，颜色是一维数组展开后的结果
    axs[3].hist(np.arange(10), color=color.reshape(-1))
    
    # 在第五个子图上绘制条形图，使用颜色 `color`，颜色是一维数组展开后的结果
    axs[4].bar(np.arange(10), np.arange(10), color=color.reshape(-1))
    
    # 使用 `fig_ref` 创建包含5个子图的图表对象
    axs = fig_ref.subplots(5)
    
    # 在第一个子图上绘制一条线，使用颜色 `color`，颜色是二维数组展开后的结果
    axs[0].plot([1, 2], [1, 2], c=color.reshape((1, -1)))
    
    # 在第二个子图上绘制散点图，使用颜色 `color`，颜色是二维数组展开后的结果
    axs[1].scatter([1, 2], [1, 2], c=color.reshape((1, -1)))
    
    # 在第三个子图上绘制阶梯图，使用颜色 `color`，颜色是二维数组展开后的结果
    axs[2].step([1, 2], [1, 2], c=color.reshape((1, -1)))
    
    # 在第四个子图上绘制直方图，使用颜色 `color`，颜色是二维数组展开后的结果
    axs[3].hist(np.arange(10), color=color.reshape((1, -1)))
    
    # 在第五个子图上绘制条形图，使用颜色 `color`，颜色是二维数组展开后的结果
    axs[4].bar(np.arange(10), np.arange(10), color=color.reshape((1, -1)))
# 使用装饰器 @check_figures_equal，验证两个图形对象是否相等，扩展名为 'png'
@check_figures_equal(extensions=['png'])
def test_shared_axes_clear(fig_test, fig_ref):
    # 创建数据 x 和 y，x 从 0 到 2π，步长为 0.01，y 是 x 的正弦值
    x = np.arange(0.0, 2*np.pi, 0.01)
    y = np.sin(x)

    # 在参考图形 fig_ref 上创建 2x2 的子图，共享 x 轴和 y 轴
    axs = fig_ref.subplots(2, 2, sharex=True, sharey=True)
    # 在每个子图上绘制正弦曲线
    for ax in axs.flat:
        ax.plot(x, y)

    # 在测试图形 fig_test 上创建 2x2 的子图，共享 x 轴和 y 轴
    axs = fig_test.subplots(2, 2, sharex=True, sharey=True)
    # 在每个子图上清空绘图并重新绘制正弦曲线
    for ax in axs.flat:
        ax.clear()
        ax.plot(x, y)


# 测试共享轴的重新设置
def test_shared_axes_retick():
    # 创建包含 2x2 子图的图形对象 fig，并共享所有子图的 x 轴和 y 轴
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')

    # 在每个子图上绘制线性图 [0, 2]
    for ax in axs.flat:
        ax.plot([0, 2], 'o-')

    # 设置第一个子图 axs[0, 0] 的 x 轴刻度，预期会影响所有子图的 x 轴范围
    axs[0, 0].set_xticks([-0.5, 0, 1, 1.5])
    # 验证所有子图的 x 轴范围是否与 axs[0, 0] 的 x 轴范围相同
    for ax in axs.flat:
        assert ax.get_xlim() == axs[0, 0].get_xlim()

    # 设置第一个子图 axs[0, 0] 的 y 轴刻度，预期会影响所有子图的 y 轴范围
    axs[0, 0].set_yticks([-0.5, 0, 2, 2.5])
    # 验证所有子图的 y 轴范围是否与 axs[0, 0] 的 y 轴范围相同
    for ax in axs.flat:
        assert ax.get_ylim() == axs[0, 0].get_ylim()


# 使用参数化测试，测试 ylabel 的水平对齐方式 ha
@pytest.mark.parametrize('ha', ['left', 'center', 'right'])
def test_ylabel_ha_with_position(ha):
    # 创建一个空的 Figure 对象 fig
    fig = Figure()
    # 在 fig 上创建一个子图 ax
    ax = fig.subplots()
    # 设置 y 轴标签为 "test"，y 轴位置为 1，水平对齐方式为参数 ha 指定的值
    ax.set_ylabel("test", y=1, ha=ha)
    # 设置 y 轴标签位置为右侧
    ax.yaxis.set_label_position("right")
    # 验证 y 轴标签的水平对齐方式是否为 ha
    assert ax.yaxis.get_label().get_ha() == ha


# 测试竖直方向的条形图标签位置
def test_bar_label_location_vertical():
    # 获取当前图形的当前轴对象 ax
    ax = plt.gca()
    # 定义条形图的 x 坐标和高度
    xs, heights = [1, 2], [3, -4]
    # 在 ax 上绘制竖直方向的条形图，返回矩形对象 rects
    rects = ax.bar(xs, heights)
    # 在条形图上添加标签，并返回标签对象 labels
    labels = ax.bar_label(rects)
    # 验证第一个标签的位置是否在 (xs[0], heights[0])
    assert labels[0].xy == (xs[0], heights[0])
    # 验证第一个标签的水平对齐方式是否为 'center'
    assert labels[0].get_horizontalalignment() == 'center'
    # 验证第一个标签的垂直对齐方式是否为 'bottom'
    assert labels[0].get_verticalalignment() == 'bottom'
    # 验证第二个标签的位置是否在 (xs[1], heights[1])
    assert labels[1].xy == (xs[1], heights[1])
    # 验证第二个标签的水平对齐方式是否为 'center'
    assert labels[1].get_horizontalalignment() == 'center'
    # 验证第二个标签的垂直对齐方式是否为 'top'
    assert labels[1].get_verticalalignment() == 'top'


# 测试竖直方向的条形图标签位置，同时反转 y 轴
def test_bar_label_location_vertical_yinverted():
    # 获取当前图形的当前轴对象 ax
    ax = plt.gca()
    # 反转当前轴的 y 轴方向
    ax.invert_yaxis()
    # 定义条形图的 x 坐标和高度
    xs, heights = [1, 2], [3, -4]
    # 在 ax 上绘制竖直方向的条形图，返回矩形对象 rects
    rects = ax.bar(xs, heights)
    # 在条形图上添加标签，并返回标签对象 labels
    labels = ax.bar_label(rects)
    # 验证第一个标签的位置是否在 (xs[0], heights[0])
    assert labels[0].xy == (xs[0], heights[0])
    # 验证第一个标签的水平对齐方式是否为 'center'
    assert labels[0].get_horizontalalignment() == 'center'
    # 验证第一个标签的垂直对齐方式是否为 'top'
    assert labels[0].get_verticalalignment() == 'top'
    # 验证第二个标签的位置是否在 (xs[1], heights[1])
    assert labels[1].xy == (xs[1], heights[1])
    # 验证第二个标签的水平对齐方式是否为 'center'
    assert labels[1].get_horizontalalignment() == 'center'
    # 验证第二个标签的垂直对齐方式是否为 'bottom'
    assert labels[1].get_verticalalignment() == 'bottom'


# 测试水平方向的条形图标签位置
def test_bar_label_location_horizontal():
    # 获取当前图形的当前轴对象 ax
    ax = plt.gca()
    # 定义条形图的 y 坐标和宽度
    ys, widths = [1, 2], [3, -4]
    # 在 ax 上绘制水平方向的条形图，返回矩形对象 rects
    rects = ax.barh(ys, widths)
    # 在条形图上添加标签，并返回标签对象 labels
    labels = ax.bar_label(rects)
    # 验证第一个标签的位置是否在 (widths[0], ys[0])
    assert labels[0].xy == (widths[0], ys[0])
    # 验证第一个标签的水平对齐方式是否为 'left'
    assert labels[0].get_horizontalalignment() == 'left'
    # 验证第一个标签的垂直对齐方式是否为 'center'
    assert labels[0].get_verticalalignment() == 'center'
    # 验证第二个标签的位置是否在 (widths[1], ys[1])
    assert labels[1].xy == (widths[1], ys[1])
    # 验证第二个标签的水平对齐方式是否为 'right'
    assert labels[1].get_horizontalalignment() == 'right'
    # 验证第二个标签的垂直对齐方式是否为 'center'


# 测试水平方向的条形图标签位置，同时反转 y 轴
def test_bar_label_location_horizontal_yinverted():
    # 获取当前图形的当前轴对象 ax
    ax = plt.gca()
    # 反转
    # 断言：验证第二个标签的水平对齐方式是否为 'right'
    assert labels[1].get_horizontalalignment() == 'right'
    
    # 断言：验证第二个标签的垂直对齐方式是否为 'center'
    assert labels[1].get_verticalalignment() == 'center'
def test_bar_label_location_horizontal_xinverted():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 反转 x 轴
    ax.invert_xaxis()
    # 定义条形图的纵坐标 ys 和宽度 widths
    ys, widths = [1, 2], [3, -4]
    # 在坐标轴上创建水平条形图，并返回条形对象的集合 rects
    rects = ax.barh(ys, widths)
    # 在条形图上添加标签，并返回标签对象的集合 labels
    labels = ax.bar_label(rects)
    # 断言第一个标签的位置坐标为 (widths[0], ys[0])
    assert labels[0].xy == (widths[0], ys[0])
    # 断言第一个标签的水平对齐方式为 'right'
    assert labels[0].get_horizontalalignment() == 'right'
    # 断言第一个标签的垂直对齐方式为 'center'
    assert labels[0].get_verticalalignment() == 'center'
    # 断言第二个标签的位置坐标为 (widths[1], ys[1])
    assert labels[1].xy == (widths[1], ys[1])
    # 断言第二个标签的水平对齐方式为 'left'
    assert labels[1].get_horizontalalignment() == 'left'
    # 断言第二个标签的垂直对齐方式为 'center'
    assert labels[1].get_verticalalignment() == 'center'


def test_bar_label_location_horizontal_xyinverted():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 反转 x 轴
    ax.invert_xaxis()
    # 反转 y 轴
    ax.invert_yaxis()
    # 定义条形图的纵坐标 ys 和宽度 widths
    ys, widths = [1, 2], [3, -4]
    # 在坐标轴上创建水平条形图，并返回条形对象的集合 rects
    rects = ax.barh(ys, widths)
    # 在条形图上添加标签，并返回标签对象的集合 labels
    labels = ax.bar_label(rects)
    # 断言第一个标签的位置坐标为 (widths[0], ys[0])
    assert labels[0].xy == (widths[0], ys[0])
    # 断言第一个标签的水平对齐方式为 'right'
    assert labels[0].get_horizontalalignment() == 'right'
    # 断言第一个标签的垂直对齐方式为 'center'
    assert labels[0].get_verticalalignment() == 'center'
    # 断言第二个标签的位置坐标为 (widths[1], ys[1])
    assert labels[1].xy == (widths[1], ys[1])
    # 断言第二个标签的水平对齐方式为 'left'
    assert labels[1].get_horizontalalignment() == 'left'
    # 断言第二个标签的垂直对齐方式为 'center'
    assert labels[1].get_verticalalignment() == 'center'


def test_bar_label_location_center():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 定义条形图的纵坐标 ys 和宽度 widths
    ys, widths = [1, 2], [3, -4]
    # 在坐标轴上创建水平条形图，并返回条形对象的集合 rects
    rects = ax.barh(ys, widths)
    # 在条形图上添加中心位置的标签，并返回标签对象的集合 labels
    labels = ax.bar_label(rects, label_type='center')
    # 断言第一个标签的位置坐标为 (0.5, 0.5)
    assert labels[0].xy == (0.5, 0.5)
    # 断言第一个标签的水平对齐方式为 'center'
    assert labels[0].get_horizontalalignment() == 'center'
    # 断言第一个标签的垂直对齐方式为 'center'
    assert labels[0].get_verticalalignment() == 'center'
    # 断言第二个标签的位置坐标为 (0.5, 0.5)
    assert labels[1].xy == (0.5, 0.5)
    # 断言第二个标签的水平对齐方式为 'center'
    assert labels[1].get_horizontalalignment() == 'center'
    # 断言第二个标签的垂直对齐方式为 'center'
    assert labels[1].get_verticalalignment() == 'center'


@image_comparison(['test_centered_bar_label_nonlinear.svg'])
def test_centered_bar_label_nonlinear():
    # 创建图形和坐标轴对象
    _, ax = plt.subplots()
    # 创建水平条形图，设置条形的高度和标签
    bar_container = ax.barh(['c', 'b', 'a'], [1_000, 5_000, 7_000])
    # 设置 x 轴为对数坐标轴，限制 x 轴的范围从 1 开始
    ax.set_xscale('log')
    ax.set_xlim(1, None)
    # 在条形图上添加中心位置的标签
    ax.bar_label(bar_container, label_type='center')
    # 关闭坐标轴的显示
    ax.set_axis_off()


def test_centered_bar_label_label_beyond_limits():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    last = 0
    # 遍历标签和值的组合
    for label, value in zip(['a', 'b', 'c'], [10, 20, 50]):
        # 创建水平条形图，设置条形的高度、标签和位置
        bar_container = ax.barh('col', value, label=label, left=last)
        # 在条形图上添加中心位置的标签
        ax.bar_label(bar_container, label_type='center')
        last += value
    # 限制 x 轴的范围到 20
    ax.set_xlim(None, 20)

    # 绘制图形但不渲染
    fig.draw_without_rendering()


def test_bar_label_location_errorbars():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 定义条形图的横坐标 xs 和高度 heights
    xs, heights = [1, 2], [3, -4]
    # 在坐标轴上创建垂直条形图，并返回条形对象的集合 rects
    rects = ax.bar(xs, heights, yerr=1)
    # 在条形图上添加标签，并返回标签对象的集合 labels
    labels = ax.bar_label(rects)
    # 断言第一个标签的位置坐标为 (xs[0], heights[0] + 1)
    assert labels[0].xy == (xs[0], heights[0] + 1)
    # 断言第一个标签的水平对齐方式为 'center'
    assert labels[0].get_horizontalalignment() == 'center'
    # 断言第一个标签的垂直对齐方式为 'bottom'
    assert labels[0].get_verticalalignment() == 'bottom'
    # 断言第二个标签的位置坐标为 (xs[1], heights[1] - 1)
    assert labels[1].xy == (xs[1], heights[1] - 1)
    # 断言第二个标签的水平对齐方式为 'center'
    assert labels[1].get_horizontalalignment() == 'center'
    # 断言第二个标签的垂直对齐方式为 'top'


@pytest.mark.parametrize('fmt', [
    '%.2f', '{:.2f}', '{:.2f}'.format
])
def test_bar_label_fmt(fmt):
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 创建垂直条形图，设置条形的高度和标签
    rects = ax.bar([1, 2], [3, -4])
    # 在条形图上添加格式化的标签，并返回标签对象的集合 labels
    labels = ax.bar_label(rects, fmt=fmt)
    # 断言第一个标签的文
    # 断言确保页面中第二个标签的文本内容为 '-4.00'
    assert labels[1].get_text() == '-4.00'
# 测试函数：检查当格式错误时是否引发异常
def test_bar_label_fmt_error():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 创建柱状图，并返回矩形对象列表
    rects = ax.bar([1, 2], [3, -4])
    # 使用 pytest 检查是否引发 TypeError 异常，异常消息应匹配 'str or callable'
    with pytest.raises(TypeError, match='str or callable'):
        # 尝试在柱状图上标注文本，使用不支持的格式（整数 10）
        _ = ax.bar_label(rects, fmt=10)


# 测试函数：检查在指定标签情况下是否正确标注柱状图
def test_bar_label_labels():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 创建柱状图，并返回矩形对象列表
    rects = ax.bar([1, 2], [3, -4])
    # 在柱状图上标注指定的标签
    labels = ax.bar_label(rects, labels=['A', 'B'])
    # 检查第一个标签是否为 'A'
    assert labels[0].get_text() == 'A'
    # 检查第二个标签是否为 'B'
    assert labels[1].get_text() == 'B'


# 测试函数：检查当数据包含 NaN 时是否正确标注柱状图
def test_bar_label_nan_ydata():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 创建包含 NaN 值的柱状图，并返回矩形对象列表
    bars = ax.bar([2, 3], [np.nan, 1])
    # 在柱状图上标注标签
    labels = ax.bar_label(bars)
    # 检查标签是否正确显示空字符串和 '1'
    assert [l.get_text() for l in labels] == ['', '1']
    # 检查第一个标签的位置是否正确
    assert labels[0].xy == (2, 0)
    # 检查第一个标签的垂直对齐方式是否为 'bottom'
    assert labels[0].get_verticalalignment() == 'bottom'


# 测试函数：检查当数据包含 NaN 且坐标轴反转时是否正确标注柱状图
def test_bar_label_nan_ydata_inverted():
    # 获取当前图形的坐标轴对象
    ax = plt.gca()
    # 反转坐标轴
    ax.yaxis_inverted()
    # 创建包含 NaN 值的柱状图，并返回矩形对象列表
    bars = ax.bar([2, 3], [np.nan, 1])
    # 在柱状图上标注标签
    labels = ax.bar_label(bars)
    # 检查标签是否正确显示空字符串和 '1'
    assert [l.get_text() for l in labels] == ['', '1']
    # 检查第一个标签的位置是否正确
    assert labels[0].xy == (2, 0)
    # 检查第一个标签的垂直对齐方式是否为 'bottom'
    assert labels[0].get_verticalalignment() == 'bottom'


# 测试函数：检查当数据包含 NaN 时是否正确标注柱状图，并处理特殊情况
def test_nan_barlabels():
    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建包含 NaN 值的柱状图，并返回矩形对象列表
    bars = ax.bar([1, 2, 3], [np.nan, 1, 2], yerr=[0.2, 0.4, 0.6])
    # 在柱状图上标注标签
    labels = ax.bar_label(bars)
    # 检查标签是否正确显示空字符串、'1' 和 '2'
    assert [l.get_text() for l in labels] == ['', '1', '2']
    # 检查坐标轴的上下限是否正确设置
    assert np.allclose(ax.get_ylim(), (0.0, 3.0))

    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建包含 NaN 值的柱状图，并返回矩形对象列表
    bars = ax.bar([1, 2, 3], [0, 1, 2], yerr=[0.2, np.nan, 0.6])
    # 在柱状图上标注标签
    labels = ax.bar_label(bars)
    # 检查标签是否正确显示 '0'、'1' 和 '2'
    assert [l.get_text() for l in labels] == ['0', '1', '2']
    # 检查坐标轴的上下限是否正确设置
    assert np.allclose(ax.get_ylim(), (-0.5, 3.0))

    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建包含 NaN 值的柱状图，并返回矩形对象列表
    bars = ax.bar([1, 2, 3], [np.nan, 1, 2], yerr=[np.nan, np.nan, 0.6])
    # 在柱状图上标注标签
    labels = ax.bar_label(bars)
    # 检查标签是否正确显示空字符串、'1' 和 '2'
    assert [l.get_text() for l in labels] == ['', '1', '2']
    # 检查坐标轴的上下限是否正确设置
    assert np.allclose(ax.get_ylim(), (0.0, 3.0))


# 测试函数：检查绘制的补丁的边界是否正确
def test_patch_bounds():  # PR 19078
    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 添加一个楔形补丁到坐标轴上
    ax.add_patch(mpatches.Wedge((0, -1), 1.05, 60, 120, width=0.1))
    # 计算底部的值
    bot = 1.9*np.sin(15*np.pi/180)**2
    # 检查补丁对象的边界是否与预期的几乎相等
    np.testing.assert_array_almost_equal_nulp(
        np.array((-0.525, -(bot+0.05), 1.05, bot+0.1)), ax.dataLim.bounds, 16)


# 测试函数：检查当传递被忽略的散点图参数时是否发出警告
@mpl.style.context('default')
def test_warn_ignored_scatter_kwargs():
    # 使用 pytest 检查是否发出 UserWarning 警告，警告消息应匹配特定模式
    with pytest.warns(UserWarning,
                      match=r"You passed a edgecolor/edgecolors"):
        # 绘制散点图，并传递一个被忽略的参数 edgecolor/edgecolors
        plt.scatter([0], [0], marker="+", s=500, facecolor="r", edgecolor="b")


# 测试函数：检查在坐标轴上的各种艺术家子列表是否正确
def test_artist_sublists():
    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建多条线并获取其列表
    lines = [ax.plot(np.arange(i, i + 5))[0] for i in range(6)]
    # 创建散点图并获取其对象
    col = ax.scatter(np.arange(5), np.arange(5))
    # 创建图像显示并获取其对象
    im = ax.imshow(np.zeros((5, 5)))
    # 添加矩形补丁并获取其对象
    patch = ax.add_patch(mpatches.Rectangle((0, 0), 5, 5))
    # 添加文本并获取其对象
    text = ax.text(0, 0, 'foo')

    # 检查坐标轴上的集合是否与预期的列表相同
    assert list(ax.collections) == [col]
    assert list(ax.images) == [im]
    assert list(ax.lines) == lines
    assert list(ax.patches) == [patch]
    assert not ax.tables
    assert list(ax.texts
    # 使用 pytest 来测试是否会引发 IndexError 异常，并检查异常消息是否包含 'out of range'
    with pytest.raises(IndexError, match='out of range'):
        # 访问超出范围的 ax.lines 元素，预期引发 IndexError
        ax.lines[len(lines) + 1]

    # 将其他列表添加到 ax.lines 应该生成一个普通的列表。
    assert ax.lines + [1, 2, 3] == [*lines, 1, 2, 3]
    assert [1, 2, 3] + ax.lines == [1, 2, 3, *lines]

    # 将其他元组添加到 ax.lines 应该生成一个普通的元组。
    assert ax.lines + (1, 2, 3) == (*lines, 1, 2, 3)
    assert (1, 2, 3) + ax.lines == (1, 2, 3, *lines)

    # 在移除项目后，列表应该为空。
    col.remove()
    assert not ax.collections  # 检查 ax.collections 是否为空
    im.remove()
    assert not ax.images  # 检查 ax.images 是否为空
    patch.remove()
    assert not ax.patches  # 检查 ax.patches 是否为空
    assert not ax.tables  # 检查 ax.tables 是否为空
    text.remove()
    assert not ax.texts  # 检查 ax.texts 是否为空

    # 移除所有 ax.lines 元素
    for ln in ax.lines:
        ln.remove()
    assert len(ax.lines) == 0  # 检查 ax.lines 是否为空列表
# 定义一个测试函数，用于测试空行图的情况
def test_empty_line_plots():
    # 创建一个长度为10的全1数组作为 x 坐标
    x = np.ones(10)
    # 创建一个形状为(10, 0)的全1数组作为 y 坐标，形成空的 y 数据
    y = np.ones((10, 0))
    # 创建一个新的图形对象和坐标轴对象
    _, ax = plt.subplots()
    # 绘制 x 和 y 的线条，返回线条对象
    line = ax.plot(x, y)
    # 断言线条数量为0，即确保没有绘制出任何线条
    assert len(line) == 0

    # 确保使用 plot([],[]) 语法能够创建一条线
    _, ax = plt.subplots()
    line = ax.plot([], [])
    assert len(line) == 1


# 使用参数化测试来测试不同的绘图格式错误情况
@pytest.mark.parametrize('fmt, match', (
    ("f", r"'f' is not a valid format string \(unrecognized character 'f'\)"),
    ("o+", r"'o\+' is not a valid format string \(two marker symbols\)"),
    (":-", r"':-' is not a valid format string \(two linestyle symbols\)"),
    ("rk", r"'rk' is not a valid format string \(two color symbols\)"),
    (":o-r", r"':o-r' is not a valid format string \(two linestyle symbols\)"),
    ("C", r"'C' is not a valid format string \('C' must be followed by a number\)"),
    (".C", r"'.C' is not a valid format string \('C' must be followed by a number\)"),
))
# 对不同的数据情况进行参数化测试
@pytest.mark.parametrize("data", [None, {"string": range(3)}])
def test_plot_format_errors(fmt, match, data):
    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 如果有提供数据，则修改匹配字符串，指示数据键或格式均不符合预期
    if data is not None:
        match = match.replace("not", "neither a data key nor")
    # 使用 pytest 的异常断言来确保对于给定的 fmt 格式，会触发 ValueError 异常，并且匹配预期的异常信息
    with pytest.raises(ValueError, match=r"\A" + match + r"\Z"):
        ax.plot("string", fmt, data=data)


# 测试不同的绘图格式情况
def test_plot_format():
    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一条带有特定格式的线，检查其颜色是否为白色
    line = ax.plot([1, 2, 3], '1.0')
    assert line[0].get_color() == (1.0, 1.0, 1.0, 1.0)
    # 检查线条的标记是否为 'None'
    assert line[0].get_marker() == 'None'

    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一条带有特定格式的线，检查其标记是否为 '1'
    line = ax.plot([1, 2, 3], '1')
    assert line[0].get_marker() == '1'

    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一条带有特定格式的线，检查其颜色是否为白色，以及 y 轴标签的文本内容是否为 '1'
    line = ax.plot([1, 2], [1, 2], '1.0', "1")
    fig.canvas.draw()
    assert line[0].get_color() == (1.0, 1.0, 1.0, 1.0)
    assert ax.get_yticklabels()[0].get_text() == '1'

    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一条带有特定格式的线，检查其标记是否为 '1'，以及 y 轴标签的文本内容是否为 '1.0'
    line = ax.plot([1, 2], [1, 2], '1', "1.0")
    fig.canvas.draw()
    assert line[0].get_marker() == '1'
    assert ax.get_yticklabels()[0].get_text() == '1.0'

    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一条带有特定格式的线，检查其标记是否为 '3'，颜色是否为黑色
    line = ax.plot([1, 2, 3], 'k3')
    assert line[0].get_marker() == '3'
    assert line[0].get_color() == 'k'

    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一条带有特定格式的线，检查其标记是否为 '.'，颜色是否为 'C12' 对应的 RGBA 值，以及线型是否为 ':'
    line = ax.plot([1, 2, 3], '.C12:')
    assert line[0].get_marker() == '.'
    assert line[0].get_color() == mcolors.to_rgba('C12')
    assert line[0].get_linestyle() == ':'


# 测试自动创建图例的功能
def test_automatic_legend():
    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一个使用数据键 'a' 和 'b' 的线条
    ax.plot("a", "b", data={"d": 2})
    # 获取图例对象
    leg = ax.legend()
    fig.canvas.draw()
    # 检查图例中第一项的文本内容是否为 'a'
    assert leg.get_texts()[0].get_text() == 'a'
    # 检查 y 轴的第一项刻度标签文本内容是否为 'a'

    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一个使用数据键 'a', 'b', 'c' 的线条
    ax.plot("a", "b", "c", data={"d": 2})
    # 获取图例对象
    leg = ax.legend()
    fig.canvas.draw()
    # 检查图例中第一项的文本内容是否为 'b'
    assert leg.get_texts()[0].get_text() == 'b'
    # 检查 x 轴的第一项刻度标签文本内容是否为 'a'
    assert ax.get_xticklabels()[0].get_text() == 'a'
    # 检查 y 轴的第一项刻度标签文本内容是否为 'b'


# 测试绘图函数的错误情况
def test_plot_errors():
    # 使用 pytest 的异常断言来确保在使用不支持的关键字参数时会触发 TypeError 异常
    with pytest.raises(TypeError, match=r"plot\(\) got an unexpected keyword"):
        plt.plot([1, 2, 3], x=1)
    # 使用 pytest 的断言检查是否会引发 ValueError 异常，并匹配特定的错误信息 "plot() with multiple groups"
    with pytest.raises(ValueError, match=r"plot\(\) with multiple groups"):
        # 调用 plt.plot() 尝试绘制包含多个数据组的图形，但是传入的参数不符合要求
        plt.plot([1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4], label=['1', '2'])
    
    # 使用 pytest 的断言检查是否会引发 ValueError 异常，并匹配特定的错误信息 "x and y must have same first"
    with pytest.raises(ValueError, match="x and y must have same first"):
        # 调用 plt.plot() 时，传入的 x 和 y 数据数组长度不匹配
        plt.plot([1, 2, 3], [1])
    
    # 使用 pytest 的断言检查是否会引发 ValueError 异常，并匹配特定的错误信息 "x and y can be no greater than"
    with pytest.raises(ValueError, match="x and y can be no greater than"):
        # 调用 plt.plot() 时，传入的参数超出了预期的长度范围
        plt.plot(np.ones((2, 2, 2)))
    
    # 使用 pytest 的断言检查是否会引发 ValueError 异常，并匹配特定的错误信息 "Using arbitrary long args with"
    with pytest.raises(ValueError, match="Using arbitrary long args with"):
        # 调用 plt.plot() 时，传入了过多的参数，不符合函数的预期用法
        plt.plot("a", "b", "c", "d", data={"a": 2})
# 定义一个测试函数，用于验证绘图对象的色标（color limits）设置是否正确
def test_clim():
    # 创建一个包含一个子图的新图形对象
    ax = plt.figure().add_subplot()
    # 遍历不同的绘图方法，并执行相应的部分绘图操作
    for plot_method in [
            partial(ax.scatter, range(3), range(3), c=range(3)),
            partial(ax.imshow, [[0, 1], [2, 3]]),
            partial(ax.pcolor,  [[0, 1], [2, 3]]),
            partial(ax.pcolormesh, [[0, 1], [2, 3]]),
            partial(ax.pcolorfast, [[0, 1], [2, 3]]),
    ]:
        # 设置色标的范围为 (7, 8)
        clim = (7, 8)
        # 执行当前绘图方法，并获取其色彩映射的标准化对象
        norm = plot_method(clim=clim).norm
        # 断言色标的最小值和最大值与预期的范围 clim 相同
        assert (norm.vmin, norm.vmax) == clim


def test_bezier_autoscale():
    # 验证贝塞尔曲线在自动缩放时会考虑曲线而非控制点
    # 定义贝塞尔曲线的顶点坐标
    verts = [[-1, 0],
             [0, -1],
             [1, 0],
             [1, 0]]
    # 定义贝塞尔曲线的绘制代码
    codes = [mpath.Path.MOVETO,
             mpath.Path.CURVE3,
             mpath.Path.CURVE3,
             mpath.Path.CLOSEPOLY]
    # 创建路径对象表示贝塞尔曲线
    p = mpath.Path(verts, codes)

    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 向坐标轴添加贝塞尔曲线的路径补丁
    ax.add_patch(mpatches.PathPatch(p))
    # 自动调整坐标轴范围以适应所有元素
    ax.autoscale()
    # 断言坐标轴的底部限制应该是曲线边缘的值 (-0.5)，而不包括控制点的值 (-1)
    assert ax.get_ylim()[0] == -0.5


def test_small_autoscale():
    # 验证包含小值的路径是否能正确自动缩放，解决问题 #24097.
    # 定义一个包含顶点坐标的 NumPy 数组，每个顶点都有 x 和 y 坐标
    verts = np.array([
        [-5.45, 0.00], [-5.45, 0.00], [-5.29, 0.00], [-5.29, 0.00],
        [-5.13, 0.00], [-5.13, 0.00], [-4.97, 0.00], [-4.97, 0.00],
        [-4.81, 0.00], [-4.81, 0.00], [-4.65, 0.00], [-4.65, 0.00],
        [-4.49, 0.00], [-4.49, 0.00], [-4.33, 0.00], [-4.33, 0.00],
        [-4.17, 0.00], [-4.17, 0.00], [-4.01, 0.00], [-4.01, 0.00],
        [-3.85, 0.00], [-3.85, 0.00], [-3.69, 0.00], [-3.69, 0.00],
        [-3.53, 0.00], [-3.53, 0.00], [-3.37, 0.00], [-3.37, 0.00],
        [-3.21, 0.00], [-3.21, 0.01], [-3.05, 0.01], [-3.05, 0.01],
        [-2.89, 0.01], [-2.89, 0.01], [-2.73, 0.01], [-2.73, 0.02],
        [-2.57, 0.02], [-2.57, 0.04], [-2.41, 0.04], [-2.41, 0.04],
        [-2.25, 0.04], [-2.25, 0.06], [-2.09, 0.06], [-2.09, 0.08],
        [-1.93, 0.08], [-1.93, 0.10], [-1.77, 0.10], [-1.77, 0.12],
        [-1.61, 0.12], [-1.61, 0.14], [-1.45, 0.14], [-1.45, 0.17],
        [-1.30, 0.17], [-1.30, 0.19], [-1.14, 0.19], [-1.14, 0.22],
        [-0.98, 0.22], [-0.98, 0.25], [-0.82, 0.25], [-0.82, 0.27],
        [-0.66, 0.27], [-0.66, 0.29], [-0.50, 0.29], [-0.50, 0.30],
        [-0.34, 0.30], [-0.34, 0.32], [-0.18, 0.32], [-0.18, 0.33],
        [-0.02, 0.33], [-0.02, 0.32], [0.13, 0.32], [0.13, 0.33], [0.29, 0.33],
        [0.29, 0.31], [0.45, 0.31], [0.45, 0.30], [0.61, 0.30], [0.61, 0.28],
        [0.77, 0.28], [0.77, 0.25], [0.93, 0.25], [0.93, 0.22], [1.09, 0.22],
        [1.09, 0.19], [1.25, 0.19], [1.25, 0.17], [1.41, 0.17], [1.41, 0.15],
        [1.57, 0.15], [1.57, 0.12], [1.73, 0.12], [1.73, 0.10], [1.89, 0.10],
        [1.89, 0.08], [2.05, 0.08], [2.05, 0.07], [2.21, 0.07], [2.21, 0.05],
        [2.37, 0.05], [2.37, 0.04], [2.53, 0.04], [2.53, 0.02], [2.69, 0.02],
        [2.69, 0.02], [2.85, 0.02], [2.85, 0.01], [3.01, 0.01], [3.01, 0.01],
        [3.17, 0.01], [3.17, 0.00], [3.33, 0.00], [3.33, 0.00], [3.49, 0.00],
        [3.49, 0.00], [3.65, 0.00], [3.65, 0.00], [3.81, 0.00], [3.81, 0.00],
        [3.97, 0.00], [3.97, 0.00], [4.13, 0.00], [4.13, 0.00], [4.29, 0.00],
        [4.29, 0.00], [4.45, 0.00], [4.45, 0.00], [4.61, 0.00], [4.61, 0.00],
        [4.77, 0.00], [4.77, 0.00], [4.93, 0.00], [4.93, 0.00],
    ])
    
    # 计算顶点数组中的最小和最大 x、y 坐标
    minx = np.min(verts[:, 0])
    miny = np.min(verts[:, 1])
    maxx = np.max(verts[:, 0])
    maxy = np.max(verts[:, 1])
    
    # 使用 verts 数组创建一个路径对象
    p = mpath.Path(verts)
    
    # 创建一个图形和轴对象
    fig, ax = plt.subplots()
    
    # 向轴对象中添加路径对象的路径补丁
    ax.add_patch(mpatches.PathPatch(p))
    
    # 自动调整轴对象的显示范围
    ax.autoscale()
    
    # 断言，确保轴对象的 x 和 y 范围包含计算出的最小和最大值
    assert ax.get_xlim()[0] <= minx
    assert ax.get_xlim()[1] >= maxx
    assert ax.get_ylim()[0] <= miny
    assert ax.get_ylim()[1] >= maxy
def test_get_xticklabel():
    # 创建一个包含单个子图的图形对象和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制一条简单的折线图，包含 0 到 9 的整数
    ax.plot(np.arange(10))
    # 遍历轴对象的 x 轴刻度标签
    for ind in range(10):
        # 断言每个 x 轴刻度标签的文本内容应该是对应的整数字符串
        assert ax.get_xticklabels()[ind].get_text() == f'{ind}'
        # 断言每个 y 轴刻度标签的文本内容应该是对应的整数字符串
        assert ax.get_yticklabels()[ind].get_text() == f'{ind}'


def test_bar_leading_nan():
    # 创建一个包含单个子图的图形对象和轴对象
    barx = np.arange(3, dtype=float)
    barheights = np.array([0.5, 1.5, 2.0])
    barstarts = np.array([0.77]*3)

    # 将第一个条形图的 x 坐标设置为 NaN
    barx[0] = np.nan

    fig, ax = plt.subplots()

    # 在轴上绘制垂直条形图，使用给定的高度和底部位置
    bars = ax.bar(barx, barheights, bottom=barstarts)

    # 在轴上绘制水平条形图，使用给定的高度和左侧位置
    hbars = ax.barh(barx, barheights, left=barstarts)

    # 遍历两种条形图集合
    for bar_set in (bars, hbars):
        # 断言第一个条形图集合中，至少有一个条形图的位置包含 NaN 值
        nanful, *rest = bar_set
        assert (~np.isfinite(nanful.xy)).any()
        # 断言除了第一个条形图外，其余所有条形图的位置都是有限数值
        assert np.isfinite(nanful.get_width())
        for b in rest:
            assert np.isfinite(b.xy).all()
            assert np.isfinite(b.get_width())


@check_figures_equal(extensions=["png"])
def test_bar_all_nan(fig_test, fig_ref):
    mpl.style.use("mpl20")
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()

    # 在测试图中绘制具有 NaN 的条形图
    ax_test.bar([np.nan], [np.nan])
    ax_test.bar([1], [1])

    # 在参考图中绘制具有 1 的条形图，并移除之
    ax_ref.bar([1], [1]).remove()
    ax_ref.bar([1], [1])


@image_comparison(["extent_units.png"], style="mpl20")
def test_extent_units():
    _, axs = plt.subplots(2, 2)
    date_first = np.datetime64('2020-01-01', 'D')
    date_last = np.datetime64('2020-01-11', 'D')
    arr = [[i+j for i in range(10)] for j in range(10)]

    # 在第一个子图上绘制热图，使用日期范围作为扩展
    axs[0, 0].set_title('Date extents on y axis')
    im = axs[0, 0].imshow(arr, origin='lower',
                          extent=[1, 11, date_first, date_last],
                          cmap=mpl.colormaps["plasma"])

    # 在第二个子图上绘制热图，使用日期范围作为扩展，并指定 X 轴的日期格式化
    axs[0, 1].set_title('Date extents on x axis (Day of Jan 2020)')
    im = axs[0, 1].imshow(arr, origin='lower',
                          extent=[date_first, date_last, 1, 11],
                          cmap=mpl.colormaps["plasma"])
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%d'))

    # 在第三个子图上绘制热图，使用相同的日期范围扩展，并指定 X 轴的日期格式化
    im = axs[1, 0].imshow(arr, origin='lower',
                          extent=[date_first, date_last,
                                  date_first, date_last],
                          cmap=mpl.colormaps["plasma"])
    axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    axs[1, 0].set(xlabel='Day of Jan 2020')

    # 在第四个子图上绘制热图，使用彩色映射并设置范围
    im = axs[1, 1].imshow(arr, origin='lower',
                          cmap=mpl.colormaps["plasma"])
    im.set_extent([date_last, date_first, date_last, date_first])
    axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    axs[1, 1].set(xlabel='Day of Jan 2020')

    # 断言设置异常以确保引发 TypeError 异常
    with pytest.raises(TypeError, match=r"set_extent\(\) got an unexpected"):
        im.set_extent([2, 12, date_first, date_last], clip=False)


def test_cla_clears_children_axes_and_fig():
    # 创建一个包含单个子图的图形对象和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制空数据的线条
    lines = ax.plot([], [], [], [])
    # 在轴上绘制单个像素的图像
    img = ax.imshow([[1]])
    # 遍历所有艺术对象，确保它们的父轴和父图都正确设置
    for art in lines + [img]:
        assert art.axes is ax
        assert art.figure is fig
    # 清除轴的内容
    ax.clear()
    # 对于列表 `lines` 中的每个元素 `art`，以及列表 `[img]` 中的唯一元素 `img`，依次执行以下断言：
    
    # 断言：验证 art 对象的 axes 属性为空
    assert art.axes is None
    
    # 断言：验证 art 对象的 figure 属性为空
    assert art.figure is None
# 定义测试函数，用于测试删除子轴功能
def test_child_axes_removal():
    # 创建图形对象和主轴对象
    fig, ax = plt.subplots()
    # 在主轴上插入次轴，位置在主轴右侧，与主轴共享y轴
    marginal = ax.inset_axes([1, 0, .1, 1], sharey=ax)
    # 在次轴上创建y轴双生轴
    marginal_twin = marginal.twinx()
    # 删除次轴
    marginal.remove()
    # 设置主轴的x轴和y轴范围
    ax.set(xlim=(-1, 1), ylim=(10, 20))


# 定义测试函数，测试散点图颜色参数解析错误
def test_scatter_color_repr_error():
    
    # 内部函数，返回下一个颜色值
    def get_next_color():
        return 'blue'  # pragma: no cover
    
    # 预期的错误消息
    msg = (
            r"'c' argument must be a color, a sequence of colors"
            r", or a sequence of numbers, not 'red\\n'"
        )
    
    # 使用pytest断言检查是否引发了预期的值错误，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        c = 'red\n'
        mpl.axes.Axes._parse_scatter_color_args(
            c, None, kwargs={}, xsize=2, get_next_color_func=get_next_color)


# 定义测试函数，测试zorder和显式光栅化
def test_zorder_and_explicit_rasterization():
    # 创建图形对象和主轴对象
    fig, ax = plt.subplots()
    # 设置光栅化的zorder
    ax.set_rasterization_zorder(5)
    # 绘制线条，并设置为光栅化和指定的zorder
    ln, = ax.plot(range(5), rasterized=True, zorder=1)
    # 使用字节流保存图形为PDF格式
    with io.BytesIO() as b:
        fig.savefig(b, format='pdf')


# 使用图像比较，测试预设的剪切路径
@image_comparison(["preset_clip_paths.png"], remove_text=True, style="mpl20",
                  tol=0.027 if platform.machine() == "arm64" else 0)
def test_preset_clip_paths():
    # 创建图形对象和主轴对象
    fig, ax = plt.subplots()

    # 创建多边形对象，并添加到主轴
    poly = mpl.patches.Polygon(
        [[1, 0], [0, 1], [-1, 0], [0, -1]], facecolor="#ddffdd",
        edgecolor="#00ff00", linewidth=2, alpha=0.5)
    ax.add_patch(poly)

    # 创建线条对象，并添加到主轴，设置剪切路径为前面创建的多边形
    line = mpl.lines.Line2D((-1, 1), (0.5, 0.5), clip_on=True, clip_path=poly)
    line.set_path_effects([patheffects.withTickedStroke()])
    ax.add_artist(line)

    # 创建另一个线条对象，并添加到主轴，设置剪切路径为前面创建的多边形
    line = mpl.lines.Line2D((-1, 1), (-0.5, -0.5), color='r', clip_on=True,
                            clip_path=poly)
    ax.add_artist(line)

    # 创建第二个多边形对象，并添加到主轴，设置剪切路径为前面创建的多边形
    poly2 = mpl.patches.Polygon(
        [[-1, 1], [0, 1], [0, -0.25]], facecolor="#beefc0", alpha=0.3,
        edgecolor="#faded0", linewidth=2, clip_on=True, clip_path=poly)
    ax.add_artist(poly2)

    # 当文本剪切生效时，"Annotation"文本应该被剪切
    ax.annotate('Annotation', (-0.75, -0.75), xytext=(0.1, 0.75),
                arrowprops={'color': 'k'}, clip_on=True, clip_path=poly)

    # 创建第三个多边形对象，并添加到图形对象，设置剪切属性为True
    poly3 = mpl.patches.Polygon(
        [[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], facecolor="g", edgecolor="y",
        linewidth=2, alpha=0.3, clip_on=True, clip_path=poly)
    fig.add_artist(poly3, clip=True)

    # 设置主轴的x轴和y轴范围
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)


# 使用mpl样式上下文，测试RC参数控制轴标签格式化
@mpl.style.context('default')
def test_rc_axes_label_formatting():
    # 设置RC参数控制轴标签的颜色、大小和粗细
    mpl.rcParams['axes.labelcolor'] = 'red'
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.labelweight'] = 'bold'

    # 创建轴对象
    ax = plt.axes()
    # 使用断言检查轴标签的颜色、大小和粗细是否符合预期
    assert ax.xaxis.label.get_color() == 'red'
    assert ax.xaxis.label.get_fontsize() == 20
    assert ax.xaxis.label.get_fontweight() == 'bold'


# 使用图形比较，测试ECDF函数
@check_figures_equal(extensions=["png"])
def test_ecdf(fig_test, fig_ref):
    # 创建包含特定值的数据数组和权重
    data = np.array([0, -np.inf, -np.inf, np.inf, 1, 1, 2])
    weights = range(len(data))
    # 创建子图对象，包含2个子图
    axs_test = fig_test.subplots(1, 2)
    # 对 axs_test 列表中的每个图形 ax 和对应的方向 orientation 进行迭代处理
    for ax, orientation in zip(axs_test, ["vertical", "horizontal"]):
        # 使用 ax 对象的 ecdf 方法生成 l0，方向为 orientation
        l0 = ax.ecdf(data, orientation=orientation)
        # 使用 ax 对象的 ecdf 方法生成 l1，传入 "d" 和 "w" 作为数据，包括 weights，方向为 orientation，补充为 True，压缩为 True，线条风格为 ":"
        l1 = ax.ecdf("d", "w", data={"d": np.ma.array(data), "w": weights},
                     orientation=orientation,
                     complementary=True, compress=True, ls=":")
        # 断言 l0 的 x 数据长度应等于 data 中非 NaN 值的数量加上 1
        assert len(l0.get_xdata()) == (~np.isnan(data)).sum() + 1
        # 断言 l1 的 x 数据长度应等于 data 中去重后的非 NaN 值的数量加上 1
        assert len(l1.get_xdata()) == len({*data[~np.isnan(data)]}) + 1
    
    # 创建一个包含两个子图的 axs_ref 数组
    axs_ref = fig_ref.subplots(1, 2)
    # 在 axs_ref[0] 子图上绘制步进后的步数图，x 轴数据为无穷小到 np.inf，y 轴数据为 0 到 1
    axs_ref[0].plot([-np.inf, -np.inf, -np.inf, 0, 1, 1, 2, np.inf],
                    np.arange(8) / 7, ds="steps-post")
    # 在 axs_ref[0] 子图上绘制步进前的步数图，x 轴数据为无穷小到 np.inf，y 轴数据为 21 到 0
    axs_ref[0].plot([-np.inf, 0, 1, 2, np.inf, np.inf],
                    np.array([21, 20, 18, 14, 3, 0]) / 21,
                    ds="steps-pre", ls=":")
    # 在 axs_ref[1] 子图上绘制步进前的步数图，y 轴数据为无穷小到 np.inf，x 轴数据为 0 到 1
    axs_ref[1].plot(np.arange(8) / 7,
                    [-np.inf, -np.inf, -np.inf, 0, 1, 1, 2, np.inf],
                    ds="steps-pre")
    # 在 axs_ref[1] 子图上绘制步进后的步数图，y 轴数据为无穷小到 np.inf，x 轴数据为 21 到 0，线条风格为 ":"
    axs_ref[1].plot(np.array([21, 20, 18, 14, 3, 0]) / 21,
                    [-np.inf, 0, 1, 2, np.inf, np.inf],
                    ds="steps-post", ls=":")
# 测试无效的 ECDF（累积分布函数）
def test_ecdf_invalid():
    # 断言：绘制 ECDF 时对包含 NaN 的数组会引发 ValueError 异常
    with pytest.raises(ValueError):
        plt.ecdf([1, np.nan])
    # 断言：绘制 ECDF 时对包含掩码的数组会引发 ValueError 异常
    with pytest.raises(ValueError):
        plt.ecdf(np.ma.array([1, 2], mask=[True, False]))


# 测试填充在坐标轴限制之间的区域
def test_fill_between_axes_limits():
    # 创建图像和轴对象
    fig, ax = plt.subplots()
    # 生成数据
    x = np.arange(0, 4 * np.pi, 0.01)
    y = 0.1*np.sin(x)
    threshold = 0.075
    # 在轴上绘制曲线
    ax.plot(x, y, color='black')

    # 记录原始的坐标轴限制
    original_lims = (ax.get_xlim(), ax.get_ylim())

    # 在指定阈值处绘制水平线
    ax.axhline(threshold, color='green', lw=2, alpha=0.7)
    # 填充 y > threshold 的区域，使用坐标轴转换
    ax.fill_between(x, 0, 1, where=y > threshold,
                    color='green', alpha=0.5, transform=ax.get_xaxis_transform())

    # 断言：填充操作后，坐标轴限制未改变
    assert (ax.get_xlim(), ax.get_ylim()) == original_lims


# 测试设置刻度参数及标签字体
def test_tick_param_labelfont():
    # 创建图像和轴对象
    fig, ax = plt.subplots()
    # 绘制简单的线图
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    # 设置 X 轴标签和字体
    ax.set_xlabel('X label in Impact font', fontname='Impact')
    # 设置 Y 轴标签和字体
    ax.set_ylabel('Y label in xkcd script', fontname='xkcd script')
    # 设置刻度的颜色和标签字体族
    ax.tick_params(color='r', labelfontfamily='monospace')
    # 设置图表标题字体
    plt.title('Title in sans-serif')
    # 断言：所有 X 轴刻度标签的字体族都是 monospace
    for text in ax.get_xticklabels():
        assert text.get_fontfamily()[0] == 'monospace'


# 测试设置次要轴的颜色
def test_set_secondary_axis_color():
    # 创建图像和主要轴对象
    fig, ax = plt.subplots()
    # 创建次要 X 轴，并设置其颜色为红色
    sax = ax.secondary_xaxis("top", color="red")
    # 断言：次要 X 轴的底部和顶部脊柱颜色与设置的颜色相同
    assert mcolors.same_color(sax.spines["bottom"].get_edgecolor(), "red")
    assert mcolors.same_color(sax.spines["top"].get_edgecolor(), "red")
    # 断言：次要 X 轴的刻度线颜色与设置的颜色相同
    assert mcolors.same_color(sax.xaxis.get_tick_params()["color"], "red")
    # 断言：次要 X 轴的刻度标签颜色与设置的颜色相同
    assert mcolors.same_color(sax.xaxis.get_tick_params()["labelcolor"], "red")
    # 断言：次要 X 轴的标签文字颜色与设置的颜色相同
    assert mcolors.same_color(sax.xaxis.label.get_color(), "red")


# 测试共享坐标轴时的坐标轴限制更改事件
def test_xylim_changed_shared():
    # 创建包含两个共享坐标轴的图像
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    events = []
    # 连接次要坐标轴的限制更改事件
    axs[1].callbacks.connect("xlim_changed", events.append)
    axs[1].callbacks.connect("ylim_changed", events.append)
    # 设置主要坐标轴的限制
    axs[0].set(xlim=[1, 3], ylim=[2, 4])
    # 断言：次要坐标轴的限制更改事件被触发了两次
    assert events == [axs[1], axs[1]]


# 测试绘制水平和垂直线/区域插值
@image_comparison(["axhvlinespan_interpolation.png"], style="default")
def test_axhvlinespan_interpolation():
    # 创建极坐标图的子图
    ax = plt.figure().add_subplot(projection="polar")
    # 关闭坐标轴显示
    ax.set_axis_off()
    # 绘制垂直线
    ax.axvline(.1, c="C0")
    # 绘制垂直区域
    ax.axvspan(.2, .3, fc="C1")
    # 绘制垂直区域（带插值）
    ax.axvspan(.4, .5, .1, .2, fc="C2")
    # 绘制水平线
    ax.axhline(1, c="C0", alpha=.5)
    # 绘制水平区域
    ax.axhspan(.8, .9, fc="C1", alpha=.5)
    # 绘制水平区域（带插值）
    ax.axhspan(.6, .7, .8, .9, fc="C2", alpha=.5)


# 测试坐标轴清空行为
@check_figures_equal(extensions=["png"])
@pytest.mark.parametrize("which", ("x", "y"))
def test_axes_clear_behavior(fig_ref, fig_test, which):
    """测试 ax.clear() 不会重置给定的刻度参数。"""
    # 获取测试图像的轴对象
    ax_test = fig_test.subplots()
    # 获取参考图像的轴对象
    ax_ref = fig_ref.subplots()
    # 选择这些刻度参数值以及它们的默认值，以确保它们各自产生视觉上的差异
    # 设置图表刻度的参数，应用于指定的坐标轴
    target = {
        "direction": "in",                  # 刻度的方向，向内
        "length": 10,                       # 刻度的长度
        "width": 10,                        # 刻度的宽度
        "color": "xkcd:wine red",           # 刻度线的颜色
        "pad": 0,                           # 刻度与标签的间距
        "labelfontfamily": "serif",         # 标签字体的家族
        "zorder": 7,                        # 刻度的绘制顺序
        "labelrotation": 45,                # 标签的旋转角度
        "labelcolor": "xkcd:shocking pink", # 标签的颜色
        # 以下项将覆盖 color 和 labelcolor 的设置，因此被忽略
        # colors: ,
        "grid_color": "xkcd:fluorescent green",   # 网格线的颜色
        "grid_alpha": 0.5,                  # 网格线的透明度
        "grid_linewidth": 3,                # 网格线的宽度
        "grid_linestyle": ":",              # 网格线的线型
        "bottom": False,                    # 底部刻度是否可见
        "top": True,                        # 顶部刻度是否可见
        "left": False,                      # 左侧刻度是否可见
        "right": True,                      # 右侧刻度是否可见
        "labelbottom": True,                # 底部标签是否可见
        "labeltop": True,                   # 顶部标签是否可见
        "labelleft": True,                  # 左侧标签是否可见
        "labelright": True,                 # 右侧标签是否可见
    }

    # 设置参考图表的刻度参数
    ax_ref.tick_params(axis=which, **target)

    # 设置测试图表的刻度参数，并清除所有内容
    ax_test.tick_params(axis=which, **target)
    ax_test.clear()

    # 在参考图表和测试图表上启用网格线
    ax_ref.grid(True)
    ax_test.grid(True)
def test_boxplot_tick_labels():
    # Test the renamed `tick_labels` parameter.
    # Test for deprecation of old name `labels`.
    np.random.seed(19680801)
    data = np.random.random((10, 3))

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
    # Should get deprecation warning for `labels`
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match='has been renamed \'tick_labels\''):
        axs[0].boxplot(data, labels=['A', 'B', 'C'])
    assert [l.get_text() for l in axs[0].get_xticklabels()] == ['A', 'B', 'C']

    # Test the new tick_labels parameter
    axs[1].boxplot(data, tick_labels=['A', 'B', 'C'])
    assert [l.get_text() for l in axs[1].get_xticklabels()] == ['A', 'B', 'C']


@needs_usetex
@check_figures_equal()
def test_latex_pie_percent(fig_test, fig_ref):
    # Test the percentage formatting in LaTeX for pie chart.
    data = [20, 10, 70]

    ax = fig_test.subplots()
    ax.pie(data, autopct="%1.0f%%", textprops={'usetex': True})

    ax1 = fig_ref.subplots()
    ax1.pie(data, autopct=r"%1.0f\%%", textprops={'usetex': True})


@check_figures_equal(extensions=['png'])
def test_violinplot_orientation(fig_test, fig_ref):
    # Test the `orientation : {'vertical', 'horizontal'}`
    # parameter and deprecation of `vert: bool`.
    fig, axs = plt.subplots(nrows=1, ncols=3)
    np.random.seed(19680801)
    all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

    axs[0].violinplot(all_data)  # Default vertical plot.
    # xticks and yticks should be at their default position.
    assert all(axs[0].get_xticks() == np.array(
        [0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]))
    assert all(axs[0].get_yticks() == np.array(
        [-30., -20., -10., 0., 10., 20., 30.]))

    # Horizontal plot using new `orientation` keyword.
    axs[1].violinplot(all_data, orientation='horizontal')
    # xticks and yticks should be swapped.
    assert all(axs[1].get_xticks() == np.array(
        [-30., -20., -10., 0., 10., 20., 30.]))
    assert all(axs[1].get_yticks() == np.array(
        [0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]))

    plt.close()

    # Deprecation of `vert: bool` keyword
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match='vert: bool was deprecated in Matplotlib 3.10'):
        # Compare images between a figure that
        # uses vert and one that uses orientation.
        ax_ref = fig_ref.subplots()
        ax_ref.violinplot(all_data, vert=False)

        ax_test = fig_test.subplots()
        ax_test.violinplot(all_data, orientation='horizontal')


@check_figures_equal(extensions=['png'])
def test_boxplot_orientation(fig_test, fig_ref):
    # Test the `orientation : {'vertical', 'horizontal'}`
    # parameter and deprecation of `vert: bool`.
    fig, axs = plt.subplots(nrows=1, ncols=2)
    np.random.seed(19680801)
    all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

    axs[0].boxplot(all_data)  # Default vertical plot.
    # xticks and yticks should be at their default position.
    ```
    # 确保第一个子图的 x 轴刻度与给定的数组相同
    assert all(axs[0].get_xticks() == np.array(
        [1, 2, 3, 4]))
    # 确保第一个子图的 y 轴刻度与给定的数组相同
    assert all(axs[0].get_yticks() == np.array(
        [-30., -20., -10., 0., 10., 20., 30.]))

    # 使用新的 `orientation` 关键字绘制水平箱线图
    axs[1].boxplot(all_data, orientation='horizontal')
    # 确保第二个子图的 x 轴刻度与给定的数组相同
    assert all(axs[1].get_xticks() == np.array(
        [-30., -20., -10., 0., 10., 20., 30.]))
    # 确保第二个子图的 y 轴刻度与给定的数组相同
    assert all(axs[1].get_yticks() == np.array(
        [1, 2, 3, 4]))

    # 关闭当前绘图窗口
    plt.close()

    # 警告：`vert: bool` 关键字已被弃用，并且 'boxplot.vertical' rcparam 也被弃用
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match='was deprecated in Matplotlib 3.10'):
        # 比较两个图之间的图像差异，一个使用 `vert`，一个使用 `orientation`
        with mpl.rc_context({'boxplot.vertical': False}):
            # 创建参考图中的坐标轴并绘制箱线图
            ax_ref = fig_ref.subplots()
            ax_ref.boxplot(all_data)

        # 创建测试图中的坐标轴并绘制水平箱线图
        ax_test = fig_test.subplots()
        ax_test.boxplot(all_data, orientation='horizontal')
```