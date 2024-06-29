# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_colorbar.py`

```py
# 导入platform模块，用于获取系统平台信息
import platform

# 导入numpy库，并用np作为别名
import numpy as np

# 导入pytest库，用于单元测试
import pytest

# 导入matplotlib的cm模块，用于颜色映射
from matplotlib import cm

# 导入matplotlib.colors模块，用于处理颜色
import matplotlib.colors as mcolors

# 导入matplotlib库，并用mpl作为别名
import matplotlib as mpl

# 导入rc_context函数，用于临时修改rc参数
from matplotlib import rc_context

# 导入image_comparison装饰器，用于图像对比测试
from matplotlib.testing.decorators import image_comparison

# 导入matplotlib.pyplot库，并用plt作为别名
import matplotlib.pyplot as plt

# 导入颜色标准化类，包括BoundaryNorm, LogNorm, PowerNorm, Normalize, NoNorm
from matplotlib.colors import (
    BoundaryNorm, LogNorm, PowerNorm, Normalize, NoNorm
)

# 导入Colorbar类，用于创建颜色条
from matplotlib.colorbar import Colorbar

# 导入FixedLocator, LogFormatter, StrMethodFormatter类，用于设置坐标轴标签
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter

# 导入check_figures_equal装饰器，用于检查两个图形是否相等
from matplotlib.testing.decorators import check_figures_equal


def _get_cmap_norms():
    """
    Define a colormap and appropriate norms for each of the four
    possible settings of the extend keyword.

    Helper function for _colorbar_extension_shape and
    colorbar_extension_length.
    """
    # 从mpl.colormaps中获取RdBu颜色映射并进行重采样，生成一个包含5个颜色的颜色映射对象
    cmap = mpl.colormaps["RdBu"].resampled(5)

    # 定义颜色映射的级别
    clevs = [-5., -2.5, -.5, .5, 1.5, 3.5]

    # 为不同的extend设置定义颜色标准化对象
    norms = dict()
    norms['neither'] = BoundaryNorm(clevs, len(clevs) - 1)
    norms['min'] = BoundaryNorm([-10] + clevs[1:], len(clevs) - 1)
    norms['max'] = BoundaryNorm(clevs[:-1] + [10], len(clevs) - 1)
    norms['both'] = BoundaryNorm([-10] + clevs[1:-1] + [10], len(clevs) - 1)
    return cmap, norms


def _colorbar_extension_shape(spacing):
    """
    Produce 4 colorbars with rectangular extensions for either uniform
    or proportional spacing.

    Helper function for test_colorbar_extension_shape.
    """
    # 获取颜色映射和相应的标准化对象
    cmap, norms = _get_cmap_norms()

    # 创建一个新的图形对象，并调整子图之间的垂直间距为4
    fig = plt.figure()
    fig.subplots_adjust(hspace=4)

    # 遍历四种extend类型并创建相应的颜色条
    for i, extension_type in enumerate(('neither', 'min', 'max', 'both')):
        # 获取对应的标准化对象
        norm = norms[extension_type]
        boundaries = values = norm.boundaries
        values = values[:-1]  # 注意在3.3版本之前最后一个值会被静默删除

        # 添加一个新的子图
        cax = fig.add_subplot(4, 1, i + 1)

        # 生成颜色条
        Colorbar(cax, cmap=cmap, norm=norm,
                 boundaries=boundaries, values=values,
                 extend=extension_type, extendrect=True,
                 orientation='horizontal', spacing=spacing)

        # 关闭子图的文本和刻度
        cax.tick_params(left=False, labelleft=False,
                        bottom=False, labelbottom=False)

    # 将生成的图形对象返回给调用者
    return fig


def _colorbar_extension_length(spacing):
    """
    Produce 12 colorbars with variable length extensions for either
    uniform or proportional spacing.

    Helper function for test_colorbar_extension_length.
    """
    # 获取颜色映射和相应的标准化对象
    cmap, norms = _get_cmap_norms()

    # 创建一个新的图形对象，并调整子图之间的垂直间距为0.6
    fig = plt.figure()
    fig.subplots_adjust(hspace=.6)
    for i, extension_type in enumerate(('neither', 'min', 'max', 'both')):
        # 使用 enumerate() 函数遍历 ('neither', 'min', 'max', 'both') 元组，并获取索引 i 和扩展类型 extension_type
        norm = norms[extension_type]
        # 从 norms 字典中获取与当前扩展类型对应的归一化对象 norm
        boundaries = values = norm.boundaries
        # 从 norm 对象中获取边界值 boundaries，并将其赋给 values
        values = values[:-1]
        # 将 values 列表切片去掉最后一个元素，赋值给 values
        for j, extendfrac in enumerate((None, 'auto', 0.1)):
            # 使用 enumerate() 函数遍历 (None, 'auto', 0.1) 元组，并获取索引 j 和扩展因子 extendfrac
            # 创建一个子图。
            cax = fig.add_subplot(12, 1, i*3 + j + 1)
            # 在图形 fig 上添加一个子图 cax
            # 生成颜色条。
            Colorbar(cax, cmap=cmap, norm=norm,
                     boundaries=boundaries, values=values,
                     extend=extension_type, extendfrac=extendfrac,
                     orientation='horizontal', spacing=spacing)
            # 使用 Colorbar 类在子图 cax 上生成颜色条，指定颜色映射 cmap、归一化对象 norm、边界 boundaries、值 values、扩展类型 extension_type、扩展因子 extendfrac，水平方向布局，间距 spacing
            # 关闭文本和刻度。
            cax.tick_params(left=False, labelleft=False,
                              bottom=False, labelbottom=False)
            # 设置子图 cax 的刻度参数，左侧刻度、底部刻度及其标签均关闭
    # 将图形 fig 返回给调用者。
    return fig
# 比较两个图像，确保测试结果正确性
@image_comparison(['colorbar_extensions_shape_uniform.png',
                   'colorbar_extensions_shape_proportional.png'])
def test_colorbar_extension_shape():
    """测试矩形色条的扩展功能。"""

    # 在重新生成此测试图像时，删除此行。
    plt.rcParams['pcolormesh.snap'] = False

    # 创建均匀和比例间距色条的图像。
    _colorbar_extension_shape('uniform')
    _colorbar_extension_shape('proportional')


# 比较两个图像，设置允许的像素差异范围为1.0
@image_comparison(['colorbar_extensions_uniform.png',
                   'colorbar_extensions_proportional.png'],
                  tol=1.0)
def test_colorbar_extension_length():
    """测试可变长度色条的扩展功能。"""

    # 在重新生成此测试图像时，删除此行。
    plt.rcParams['pcolormesh.snap'] = False

    # 创建均匀和比例间距色条的图像。
    _colorbar_extension_length('uniform')
    _colorbar_extension_length('proportional')


# 参数化测试函数，测试在反转轴上使用扩展颜色功能
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
@pytest.mark.parametrize("extend,expected", [("min", (0, 0, 0, 1)),
                                             ("max", (1, 1, 1, 1)),
                                             ("both", (1, 1, 1, 1))])
def test_colorbar_extension_inverted_axis(orientation, extend, expected):
    """测试在反转轴上使用扩展颜色功能。"""
    
    data = np.arange(12).reshape(3, 4)
    fig, ax = plt.subplots()
    
    # 使用 viridis 颜色映射，指定下限和上限颜色
    cmap = mpl.colormaps["viridis"].with_extremes(under=(0, 0, 0, 1),
                                                  over=(1, 1, 1, 1))
    im = ax.imshow(data, cmap=cmap)
    
    # 创建颜色条
    cbar = fig.colorbar(im, orientation=orientation, extend=extend)
    
    # 如果是水平方向，反转 x 轴
    if orientation == "horizontal":
        cbar.ax.invert_xaxis()
    else:
        cbar.ax.invert_yaxis()
    
    # 断言扩展面片的颜色是否符合预期
    assert cbar._extend_patches[0].get_facecolor() == expected
    
    # 如果扩展类型为 'both'，还要检查第二个扩展面片的颜色
    if extend == "both":
        assert len(cbar._extend_patches) == 2
        assert cbar._extend_patches[1].get_facecolor() == (0, 0, 0, 1)
    else:
        assert len(cbar._extend_patches) == 1


# 图像比较测试函数，测试颜色条的位置
@pytest.mark.parametrize('use_gridspec', [True, False])
@image_comparison(['cbar_with_orientation',
                   'cbar_locationing',
                   'double_cbar',
                   'cbar_sharing',
                   ],
                  extensions=['png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
def test_colorbar_positioning(use_gridspec):
    """测试颜色条的位置设置。"""

    # 在重新生成此测试图像时，删除此行。
    plt.rcParams['pcolormesh.snap'] = False

    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]

    # -------------------
    # 创建等高线图
    plt.figure()
    plt.contourf(data, levels=levels)
    
    # 添加水平方向的颜色条，使用 gridspec 或不使用
    plt.colorbar(orientation='horizontal', use_gridspec=use_gridspec)

    locations = ['left', 'right', 'top', 'bottom']
    # 创建新图
    plt.figure()
    # 对于给定的位置列表，使用enumerate函数生成索引和位置信息
    for i, location in enumerate(locations):
        # 在2x2的子图中的第(i+1)个位置绘制等高线图，数据为data，使用预设的levels
        plt.subplot(2, 2, i + 1)
        plt.contourf(data, levels=levels)
        # 添加颜色条，位置由location指定，使用gridspec来确定位置

    # -------------------
    # 创建新的图形窗口
    plt.figure()
    # 创建另外的数据（随机整数）
    data_2nd = np.array([[2, 3, 2, 3], [1.5, 2, 2, 3], [2, 3, 3, 4]])
    # 将随机数据扩展到与主数据相同的形状
    data_2nd = np.repeat(np.repeat(data_2nd, 10, axis=1), 10, axis=0)

    # 绘制等高线填充图，数据为data，使用预设的levels，并指定颜色映射对象
    color_mappable = plt.contourf(data, levels=levels, extend='both')
    # 在第二组数据上绘制带有指定hatches的等高线填充图，用于表示不同的区域
    hatch_mappable = plt.contourf(data_2nd, levels=[1, 2, 3], colors='none',
                                  hatches=['/', 'o', '+'], extend='max')
    # 绘制黑色的等高线图
    plt.contour(hatch_mappable, colors='black')

    # 添加颜色条，位置在左侧，标签为'variable 1'，使用gridspec来确定位置
    plt.colorbar(color_mappable, location='left', label='variable 1',
                 use_gridspec=use_gridspec)
    # 添加颜色条，位置在右侧，标签为'variable 2'，使用gridspec来确定位置
    plt.colorbar(hatch_mappable, location='right', label='variable 2',
                 use_gridspec=use_gridspec)

    # -------------------
    # 创建新的图形窗口
    plt.figure()
    # 在211子图中绘制等高线填充图，数据为data，使用预设的levels
    ax1 = plt.subplot(211, anchor='NE', aspect='equal')
    plt.contourf(data, levels=levels)
    # 在223子图中绘制等高线填充图，数据为data，使用预设的levels
    ax2 = plt.subplot(223)
    plt.contourf(data, levels=levels)
    # 在224子图中绘制等高线填充图，数据为data，使用预设的levels
    ax3 = plt.subplot(224)
    plt.contourf(data, levels=levels)

    # 添加颜色条，位置在右侧，使用ax1, ax2, ax3三个子图作为参考，指定位置、尺寸等参数
    plt.colorbar(ax=[ax2, ax3, ax1], location='right', pad=0.0, shrink=0.5,
                 panchor=False, use_gridspec=use_gridspec)
    # 添加颜色条，位置在左侧，使用ax1, ax2, ax3三个子图作为参考，指定位置、尺寸等参数
    plt.colorbar(ax=[ax2, ax3, ax1], location='left', shrink=0.5,
                 panchor=False, use_gridspec=use_gridspec)
    # 添加颜色条，位置在底部，使用ax1作为参考，指定位置、尺寸等参数
    plt.colorbar(ax=[ax1], location='bottom', panchor=False,
                 anchor=(0.8, 0.5), shrink=0.6, use_gridspec=use_gridspec)
def test_colorbar_single_ax_panchor_false():
    # 创建一个包含单个子图的图像，锚定点为 'N'
    ax = plt.subplot(111, anchor='N')
    # 在子图中显示一个简单的图像，二维数组
    plt.imshow([[0, 1]])
    # 添加一个颜色条，锚定点设置为 False
    plt.colorbar(panchor=False)
    # 断言子图的锚定点是否为 'N'
    assert ax.get_anchor() == 'N'


@pytest.mark.parametrize('constrained', [False, True],
                         ids=['standard', 'constrained'])
def test_colorbar_single_ax_panchor_east(constrained):
    # 创建一个图像对象，如果 constrained=True 则使用约束布局
    fig = plt.figure(constrained_layout=constrained)
    # 向图像中添加一个包含单个子图的坐标系，锚定点为 'N'
    ax = fig.add_subplot(111, anchor='N')
    # 在子图中显示一个简单的图像，二维数组
    plt.imshow([[0, 1]])
    # 添加一个颜色条，锚定点设置为 'E'
    plt.colorbar(panchor='E')
    # 断言子图的锚定点是否为 'E'
    assert ax.get_anchor() == 'E'


@image_comparison(['contour_colorbar.png'], remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.054)
def test_contour_colorbar():
    # 创建一个包含一个子图的图像对象，设置尺寸为 (4, 2)
    fig, ax = plt.subplots(figsize=(4, 2))
    # 创建一个 30x40 的数组数据
    data = np.arange(1200).reshape(30, 40) - 500
    # 定义等高线的级别
    levels = np.array([0, 200, 400, 600, 800, 1000, 1200]) - 500

    # 在子图中绘制等高线图，使用指定的级别
    CS = ax.contour(data, levels=levels, extend='both')
    # 在图像对象中添加一个水平方向的颜色条，使用与等高线相同的级别
    fig.colorbar(CS, orientation='horizontal', extend='both')
    # 在图像对象中添加一个垂直方向的颜色条，使用与等高线相同的级别
    fig.colorbar(CS, orientation='vertical')


@image_comparison(['cbar_with_subplots_adjust.png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
def test_gridspec_make_colorbar():
    # 创建一个新的图像对象
    plt.figure()
    # 创建一个 30x40 的数组数据
    data = np.arange(1200).reshape(30, 40)
    # 定义等高线的级别
    levels = [0, 200, 400, 600, 800, 1000, 1200]

    # 在第一个子图中绘制填充的等高线图，使用指定的级别
    plt.subplot(121)
    plt.contourf(data, levels=levels)
    # 在第一个子图中添加一个垂直方向的颜色条，使用网格规范
    plt.colorbar(use_gridspec=True, orientation='vertical')

    # 在第二个子图中绘制填充的等高线图，使用指定的级别
    plt.subplot(122)
    plt.contourf(data, levels=levels)
    # 在第二个子图中添加一个水平方向的颜色条，使用网格规范
    plt.colorbar(use_gridspec=True, orientation='horizontal')

    # 调整子图之间的间距和整体布局
    plt.subplots_adjust(top=0.95, right=0.95, bottom=0.2, hspace=0.25)


@image_comparison(['colorbar_single_scatter.png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
def test_colorbar_single_scatter():
    # 创建一个新的图像对象
    plt.figure()
    # 定义单个点的 x 和 y 坐标
    x = y = [0]
    # 定义单个点的 z 值
    z = [50]
    # 使用 'jet' 颜色映射创建一个颜色条，包含 16 个颜色
    cmap = mpl.colormaps['jet'].resampled(16)
    # 绘制散点图，使用指定的颜色映射和颜色数据
    cs = plt.scatter(x, y, z, c=z, cmap=cmap)
    # 添加一个颜色条
    plt.colorbar(cs)


@pytest.mark.parametrize('use_gridspec', [True, False])
@pytest.mark.parametrize('nested_gridspecs', [True, False])
def test_remove_from_figure(nested_gridspecs, use_gridspec):
    """Test `remove` with the specified ``use_gridspec`` setting."""
    # 创建一个新的图像对象
    fig = plt.figure()
    if nested_gridspecs:
        # 如果 nested_gridspecs=True，则使用嵌套的网格规范
        gs = fig.add_gridspec(2, 2)[1, 1].subgridspec(2, 2)
        # 向图像中添加一个子图，位置位于 gs 的第 [1, 1] 个位置
        ax = fig.add_subplot(gs[1, 1])
    else:
        # 否则，添加一个简单的子图
        ax = fig.add_subplot()
    # 在子图中绘制散点图
    sc = ax.scatter([1, 2], [3, 4])
    # 设置散点图的数据数组
    sc.set_array(np.array([5, 6]))
    # 获取添加颜色条时子图的初始位置
    pre_position = ax.get_position()
    # 添加一个颜色条，并根据 use_gridspec 的设置使用网格规范
    cb = fig.colorbar(sc, use_gridspec=use_gridspec)
    # 调整图像的布局
    fig.subplots_adjust()
    # 移除颜色条
    cb.remove()
    # 再次调整图像的布局
    fig.subplots_adjust()
    # 获取移除颜色条后子图的位置
    post_position = ax.get_position()
    # 使用断言检查两个对象的点集是否完全相同
    assert (pre_position.get_points() == post_position.get_points()).all()
# 定义函数，用于测试在 constrained_layout 下的 colorbar 的移除操作
def test_remove_from_figure_cl():
    """Test `remove` with constrained_layout."""
    # 创建一个带有 constrained_layout 的图形和坐标轴
    fig, ax = plt.subplots(constrained_layout=True)
    # 在坐标轴上绘制散点图
    sc = ax.scatter([1, 2], [3, 4])
    # 设置散点的数组属性
    sc.set_array(np.array([5, 6]))
    # 绘制图形但不进行渲染
    fig.draw_without_rendering()
    # 获取坐标轴的初始位置
    pre_position = ax.get_position()
    # 添加 colorbar 到图形上
    cb = fig.colorbar(sc)
    # 移除 colorbar
    cb.remove()
    # 再次绘制图形但不进行渲染
    fig.draw_without_rendering()
    # 获取坐标轴的最终位置
    post_position = ax.get_position()
    # 使用 np.testing 断言初始位置与最终位置相近
    np.testing.assert_allclose(pre_position.get_points(),
                               post_position.get_points())


# 定义函数，测试 Colorbar 类的基本功能
def test_colorbarbase():
    # 获取当前坐标轴
    ax = plt.gca()
    # 使用默认色图创建 Colorbar
    Colorbar(ax, cmap=plt.cm.bone)


# 定义函数，测试没有父对象的可映射对象的情况
def test_parentless_mappable():
    # 创建一个空的 PatchCollection，使用 viridis 色图和空数组
    pc = mpl.collections.PatchCollection([], cmap=plt.get_cmap('viridis'), array=[])
    # 使用 pytest 断言捕获异常，并匹配特定消息
    with pytest.raises(ValueError, match='Unable to determine Axes to steal'):
        # 尝试为 PatchCollection 创建 colorbar
        plt.colorbar(pc)


# 使用图像对比测试装饰器，测试 colorbar 在闭合路径下的行为
@image_comparison(['colorbar_closed_patch.png'], remove_text=True)
def test_colorbar_closed_patch():
    # 当测试图像重新生成时，删除此行。
    # 设置 rcParams，关闭 pcolormesh 的快照功能
    plt.rcParams['pcolormesh.snap'] = False

    # 创建一个大小为 (8, 6) 的图形
    fig = plt.figure(figsize=(8, 6))
    # 添加五个具有不同位置的坐标轴
    ax1 = fig.add_axes([0.05, 0.85, 0.9, 0.1])
    ax2 = fig.add_axes([0.1, 0.65, 0.75, 0.1])
    ax3 = fig.add_axes([0.05, 0.45, 0.9, 0.1])
    ax4 = fig.add_axes([0.05, 0.25, 0.9, 0.1])
    ax5 = fig.add_axes([0.05, 0.05, 0.9, 0.1])

    # 重新采样 RdBu 色图为 5 个颜色
    cmap = mpl.colormaps["RdBu"].resampled(5)

    # 在 ax1 上绘制 pcolormesh 图像，使用 cmap
    im = ax1.pcolormesh(np.linspace(0, 10, 16).reshape((4, 4)), cmap=cmap)

    # 定义 values 参数，与图像数据范围和 LUT 中的颜色数量匹配
    values = np.linspace(0, 10, 5)
    cbar_kw = dict(orientation='horizontal', values=values, ticks=[])

    # 使用特定的 rc_context 设置，为四个不同的 ax 添加 colorbar
    with rc_context({'axes.linewidth': 16}):
        plt.colorbar(im, cax=ax2, extend='both', extendfrac=0.5, **cbar_kw)
        plt.colorbar(im, cax=ax3, extend='both', **cbar_kw)
        plt.colorbar(im, cax=ax4, extend='both', extendrect=True, **cbar_kw)
        plt.colorbar(im, cax=ax5, extend='neither', **cbar_kw)


# 定义函数，测试 colorbar 的 ticks 行为
def test_colorbar_ticks():
    # 创建一个包含等高线的图形和坐标轴
    fig, ax = plt.subplots()
    # 创建 X, Y 网格和对应的数据 Z
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    # 定义等高线的 levels 和 colors
    clevs = np.array([-12, -5, 0, 5, 12], dtype=float)
    colors = ['r', 'g', 'b', 'c']
    # 在坐标轴上绘制等高线填充图
    cs = ax.contourf(X, Y, Z, clevs, colors=colors, extend='neither')
    # 为等高线图添加 colorbar，水平方向，使用指定的 ticks
    cbar = fig.colorbar(cs, ax=ax, orientation='horizontal', ticks=clevs)
    # 使用 assert 断言 colorbar 的 x 轴刻度数量与 levels 数量相同
    assert len(cbar.ax.xaxis.get_ticklocs()) == len(clevs)


# 定义函数，测试 colorbar 的 minorticks 开关功能
def test_colorbar_minorticks_on_off():
    # 测试 GitHub 问题 #11510 和 PR #11584
    np.random.seed(seed=12345)
    data = np.random.randn(20, 20)
    with rc_context({'_internal.classic_mode': False}):
        # 在非经典模式下创建图形和轴对象
        fig, ax = plt.subplots()
        # 故意设置 vmin 和 vmax 为奇怪的分数，以便检查次要刻度的正确位置
        im = ax.pcolormesh(data, vmin=-2.3, vmax=3.3)

        # 添加颜色条，并在其上启用次要刻度
        cbar = fig.colorbar(im, extend='both')
        cbar.minorticks_on()
        
        # 断言颜色条的次要刻度位置是否准确
        np.testing.assert_almost_equal(
            cbar.ax.yaxis.get_minorticklocs(),
            [-2.2, -1.8, -1.6, -1.4, -1.2, -0.8, -0.6, -0.4, -0.2,
             0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2.2, 2.4, 2.6, 2.8, 3.2])
        
        # 在关闭次要刻度后进行测试
        cbar.minorticks_off()
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_minorticklocs(), [])

        # 修改色块的颜色限制，并再次启用次要刻度
        im.set_clim(vmin=-1.2, vmax=1.2)
        cbar.minorticks_on()
        np.testing.assert_almost_equal(
            cbar.ax.yaxis.get_minorticklocs(),
            [-1.2, -1.1, -0.9, -0.8, -0.7, -0.6, -0.4, -0.3, -0.2, -0.1,
             0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2])

    # 测试 GitHub 问题 #13257 和 PR #13265
    # 创建一个随机数据数组
    data = np.random.uniform(low=1, high=10, size=(20, 20))

    # 创建新的图形和轴对象
    fig, ax = plt.subplots()
    # 使用对数标准化创建颜色块
    im = ax.pcolormesh(data, norm=LogNorm())
    # 添加颜色条
    cbar = fig.colorbar(im)
    fig.canvas.draw()

    # 获取默认的次要刻度位置
    default_minorticklocs = cbar.ax.yaxis.get_minorticklocs()

    # 测试对数标准化情况下是否成功关闭了次要刻度
    cbar.minorticks_off()
    np.testing.assert_equal(cbar.ax.yaxis.get_minorticklocs(), [])

    # 测试对数标准化情况下是否成功重新启用了次要刻度
    cbar.minorticks_on()
    np.testing.assert_equal(cbar.ax.yaxis.get_minorticklocs(),
                            default_minorticklocs)

    # 测试问题 #13339: 对数标准化情况下次要刻度应该保持关闭状态
    cbar.minorticks_off()
    # 设置新的主要刻度，并断言次要刻度是否保持关闭状态
    cbar.set_ticks([3, 5, 7, 9])
    np.testing.assert_equal(cbar.ax.yaxis.get_minorticklocs(), [])
def test_cbar_minorticks_for_rc_xyminortickvisible():
    """
    issue gh-16468.

    Making sure that minor ticks on the colorbar are turned on
    (internally) using the cbar.minorticks_on() method when
    rcParams['xtick.minor.visible'] = True (for horizontal cbar)
    rcParams['ytick.minor.visible'] = True (for vertical cbar).
    Using cbar.minorticks_on() ensures that the minor ticks
    don't overflow into the extend regions of the colorbar.
    """

    # 设置全局参数使得垂直和水平色条的次要刻度可见
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.minor.visible'] = True

    # 设置色条的颜色映射范围
    vmin, vmax = 0.4, 2.6
    # 创建一个包含色块的图像
    fig, ax = plt.subplots()
    im = ax.pcolormesh([[1, 2]], vmin=vmin, vmax=vmax)

    # 在图像上创建颜色条，垂直方向，两侧均展示
    cbar = fig.colorbar(im, extend='both', orientation='vertical')
    # 断言颜色条的 y 轴次要刻度在指定范围内
    assert cbar.ax.yaxis.get_minorticklocs()[0] >= vmin
    assert cbar.ax.yaxis.get_minorticklocs()[-1] <= vmax

    # 在图像上创建颜色条，水平方向，两侧均展示
    cbar = fig.colorbar(im, extend='both', orientation='horizontal')
    # 断言颜色条的 x 轴次要刻度在指定范围内
    assert cbar.ax.xaxis.get_minorticklocs()[0] >= vmin
    assert cbar.ax.xaxis.get_minorticklocs()[-1] <= vmax


def test_colorbar_autoticks():
    # 测试新的自动刻度模式。需要经典模式因为非经典模式不走这个路线。
    with rc_context({'_internal.classic_mode': False}):
        # 创建包含两个子图的图像
        fig, ax = plt.subplots(2, 1)
        # 创建 X 和 Y 轴的数据
        x = np.arange(-3.0, 4.001)
        y = np.arange(-4.0, 3.001)
        X, Y = np.meshgrid(x, y)
        Z = X * Y
        Z = Z[:-1, :-1]
        # 在第一个子图上创建颜色块
        pcm = ax[0].pcolormesh(X, Y, Z)
        # 在第一个子图上创建颜色条，垂直方向，两侧均展示
        cbar = fig.colorbar(pcm, ax=ax[0], extend='both',
                            orientation='vertical')

        # 在第二个子图上创建颜色块
        pcm = ax[1].pcolormesh(X, Y, Z)
        # 在第二个子图上创建颜色条，垂直方向，两侧均展示，并缩小尺寸
        cbar2 = fig.colorbar(pcm, ax=ax[1], extend='both',
                             orientation='vertical', shrink=0.4)
        # 断言第一个颜色条的 y 轴主刻度在指定范围内
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_ticklocs(),
                                       np.arange(-15, 16, 5))
        # 断言第二个颜色条的 y 轴主刻度在指定范围内
        np.testing.assert_almost_equal(cbar2.ax.yaxis.get_ticklocs(),
                                       np.arange(-20, 21, 10))


def test_colorbar_autotickslog():
    # 测试新的对数自动刻度模式...
    # 使用上下文管理器设置 matplotlib 的参数，确保不使用经典模式
    with rc_context({'_internal.classic_mode': False}):
        # 创建一个包含两个子图的图形对象和对应的轴对象
        fig, ax = plt.subplots(2, 1)
        
        # 生成一维数组 x 包含值从 -3 到 4 的浮点数
        x = np.arange(-3.0, 4.001)
        
        # 生成一维数组 y 包含值从 -4 到 3 的浮点数
        y = np.arange(-4.0, 3.001)
        
        # 使用 x 和 y 生成网格矩阵 X 和 Y
        X, Y = np.meshgrid(x, y)
        
        # 计算 Z 矩阵，其元素值为 X 和 Y 对应位置元素的乘积
        Z = X * Y
        
        # 去除 Z 矩阵最后一行和最后一列，调整其维度
        Z = Z[:-1, :-1]
        
        # 在第一个子图 ax[0] 上绘制彩色网格，颜色值为 10 的 Z 次方，使用对数标准化
        pcm = ax[0].pcolormesh(X, Y, 10**Z, norm=LogNorm())
        
        # 在第一个子图 ax[0] 上添加颜色条，参数包括方向为垂直（vertical）
        cbar = fig.colorbar(pcm, ax=ax[0], extend='both', orientation='vertical')

        # 在第二个子图 ax[1] 上绘制彩色网格，颜色值为 10 的 Z 次方，使用对数标准化
        pcm = ax[1].pcolormesh(X, Y, 10**Z, norm=LogNorm())
        
        # 在第二个子图 ax[1] 上添加颜色条，参数包括方向为垂直（vertical）和收缩比例为 0.4
        cbar2 = fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical', shrink=0.4)
        
        # 断言第一个颜色条的刻度位置，确保其刻度值接近 10 的从 -16 到 16.2 步长为 4 的幂次方
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_ticklocs(),
                                       10**np.arange(-16., 16.2, 4.))
        
        # 断言第二个颜色条的刻度位置，确保其刻度值接近 10 的从 -24 到 25 步长为 12 的幂次方
        np.testing.assert_almost_equal(cbar2.ax.yaxis.get_ticklocs(),
                                       10**np.arange(-24., 25., 12.))
# 测试用例：测试 colorbar 获取 ticks 的功能
def test_colorbar_get_ticks():
    # 创建一个新的图形对象
    plt.figure()
    # 创建一个 30x40 的数据数组
    data = np.arange(1200).reshape(30, 40)
    # 指定 contourf 的 levels 参数
    levels = [0, 200, 400, 600, 800, 1000, 1200]

    # 绘制等高线图，并填充颜色
    plt.contourf(data, levels=levels)

    # 测试设置 ticks 的 getter 方法
    userTicks = plt.colorbar(ticks=[0, 600, 1200])
    assert userTicks.get_ticks().tolist() == [0, 600, 1200]

    # 测试调用 set_ticks 后的 getter 方法
    userTicks.set_ticks([600, 700, 800])
    assert userTicks.get_ticks().tolist() == [600, 700, 800]

    # 测试调用 set_ticks 时，一些 ticks 超出界限的情况
    # 移除 #20054: 其他轴不修剪固定列表，所以 colorbar 也不应该：
    # userTicks.set_ticks([600, 1300, 1400, 1500])
    # assert userTicks.get_ticks().tolist() == [600]

    # 测试未指定 ticks 时的 getter 方法
    defTicks = plt.colorbar(orientation='horizontal')
    np.testing.assert_allclose(defTicks.get_ticks().tolist(), levels)

    # 测试普通 ticks 和次要 ticks
    fig, ax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    Z = Z[:-1, :-1]
    pcm = ax.pcolormesh(X, Y, Z)
    # 创建一个包含 colorbar 的图，并指定一些参数
    cbar = fig.colorbar(pcm, ax=ax, extend='both',
                        orientation='vertical')
    # 获取 ticks
    ticks = cbar.get_ticks()
    np.testing.assert_allclose(ticks, np.arange(-15, 16, 5))
    # 检查是否没有设置次要 ticks
    assert len(cbar.get_ticks(minor=True)) == 0


# 使用参数化测试扩展功能测试
@pytest.mark.parametrize("extend", ['both', 'min', 'max'])
def test_colorbar_lognorm_extension(extend):
    # 测试带有对数规范的 colorbar 是否正确扩展
    f, ax = plt.subplots()
    cb = Colorbar(ax, norm=LogNorm(vmin=0.1, vmax=1000.0),
                  orientation='vertical', extend=extend)
    assert cb._values[0] >= 0.0


# 测试带有幂规范的 colorbar 是否正确扩展
def test_colorbar_powernorm_extension():
    # 这只是一个烟雾测试，检查添加 colorbar 是否会引发错误或警告
    fig, ax = plt.subplots()
    Colorbar(ax, norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0),
             orientation='vertical', extend='both')


# 测试 colorbar 的 axes 相关关键字
def test_colorbar_axes_kw():
    # 测试修复 #8493：验证与 axes 相关的关键字是否传递正确，并且不会引发异常。
    plt.figure()
    plt.imshow([[1, 2], [3, 4]])
    plt.colorbar(orientation='horizontal', fraction=0.2, pad=0.2, shrink=0.5,
                 aspect=10, anchor=(0., 0.), panchor=(0., 1.))


# 测试带有对数规范的 colorbar 是否显示次要刻度标签
def test_colorbar_log_minortick_labels():
    # 设置运行上下文，禁用 matplotlib 的经典模式
    with rc_context({'_internal.classic_mode': False}):
        # 创建一个新的图形和一个轴对象
        fig, ax = plt.subplots()
        # 在轴上绘制图像，使用对数标准化
        pcm = ax.imshow([[10000, 50000]], norm=LogNorm())
        # 添加颜色条到图形
        cb = fig.colorbar(pcm)
        # 绘制图形的画布
        fig.canvas.draw()
        # 获取颜色条上所有刻度标签的文本内容
        lb = [l.get_text() for l in cb.ax.yaxis.get_ticklabels(which='both')]
        # 预期的刻度标签文本内容列表
        expected = [r'$\mathdefault{10^{4}}$',
                    r'$\mathdefault{2\times10^{4}}$',
                    r'$\mathdefault{3\times10^{4}}$',
                    r'$\mathdefault{4\times10^{4}}$']
        # 断言预期的文本内容是否存在于获取到的标签文本中
        for exp in expected:
            assert exp in lb
def test_colorbar_renorm():
    # 创建二维网格，x和y范围从-4到4，间距为31个点，用于生成z值
    x, y = np.ogrid[-4:4:31j, -4:4:31j]
    # 根据x和y计算z值，采用指数衰减
    z = 120000*np.exp(-x**2 - y**2)

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制z的颜色图像
    im = ax.imshow(z)
    # 在图形上创建颜色条
    cbar = fig.colorbar(im)
    # 断言颜色条的主要刻度位置是否与指定范围和间隔一致
    np.testing.assert_allclose(cbar.ax.yaxis.get_majorticklocs(),
                               np.arange(0, 120000.1, 20000))

    # 设置颜色条的刻度位置
    cbar.set_ticks([1, 2, 3])
    # 断言颜色条的定位器是否为FixedLocator类型
    assert isinstance(cbar.locator, FixedLocator)

    # 创建对数标准化对象，并将其应用于图像对象
    norm = LogNorm(z.min(), z.max())
    im.set_norm(norm)
    # 断言颜色条的主要刻度位置是否与对数间隔一致
    np.testing.assert_allclose(cbar.ax.yaxis.get_majorticklocs(),
                               np.logspace(-10, 7, 18))
    # 注意set_norm方法会移除FixedLocator...

    # 断言颜色条的最小值是否接近于z的最小值
    assert np.isclose(cbar.vmin, z.min())
    # 再次设置颜色条的刻度位置
    cbar.set_ticks([1, 2, 3])
    # 再次断言颜色条的定位器是否为FixedLocator类型
    assert isinstance(cbar.locator, FixedLocator)
    # 断言颜色条的主要刻度位置是否与指定刻度一致
    np.testing.assert_allclose(cbar.ax.yaxis.get_majorticklocs(),
                               [1.0, 2.0, 3.0])

    # 创建新的对数标准化对象，并将其应用于图像对象
    norm = LogNorm(z.min() * 1000, z.max() * 1000)
    im.set_norm(norm)
    # 断言颜色条的最小值是否接近于z的最小值乘以1000
    assert np.isclose(cbar.vmin, z.min() * 1000)
    # 断言颜色条的最大值是否接近于z的最大值乘以1000


@pytest.mark.parametrize('fmt', ['%4.2e', '{x:.2e}'])
def test_colorbar_format(fmt):
    # 确保格式正确传递
    x, y = np.ogrid[-4:4:31j, -4:4:31j]
    z = 120000*np.exp(-x**2 - y**2)

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制z的颜色图像
    im = ax.imshow(z)
    # 在图形上创建颜色条，并设置格式
    cbar = fig.colorbar(im, format=fmt)
    fig.canvas.draw()
    # 断言颜色条的第五个刻度标签文本是否为指定值
    assert cbar.ax.yaxis.get_ticklabels()[4].get_text() == '8.00e+04'

    # 修改映射的颜色极值范围，确保格式不会丢失
    im.set_clim([4, 200])
    fig.canvas.draw()
    # 再次断言颜色条的第五个刻度标签文本是否为指定值
    assert cbar.ax.yaxis.get_ticklabels()[4].get_text() == '2.00e+02'

    # 但如果修改了归一化方式：
    im.set_norm(LogNorm(vmin=0.1, vmax=10))
    fig.canvas.draw()
    # 断言颜色条的第一个刻度标签文本是否为指定的数学格式
    assert (cbar.ax.yaxis.get_ticklabels()[0].get_text() ==
            '$\\mathdefault{10^{-2}}$')


def test_colorbar_scale_reset():
    x, y = np.ogrid[-4:4:31j, -4:4:31j]
    z = 120000*np.exp(-x**2 - y**2)

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制z的伪彩色网格图
    pcm = ax.pcolormesh(z, cmap='RdBu_r', rasterized=True)
    # 在图形上创建颜色条，绑定到指定的坐标轴
    cbar = fig.colorbar(pcm, ax=ax)
    # 设置颜色条的外边框颜色
    cbar.outline.set_edgecolor('red')
    # 断言颜色条的Y轴刻度的标度是否为线性

    assert cbar.ax.yaxis.get_scale() == 'linear'

    # 修改归一化方式为对数尺度
    pcm.set_norm(LogNorm(vmin=1, vmax=100))
    # 断言颜色条的Y轴刻度的标度是否为对数
    assert cbar.ax.yaxis.get_scale() == 'log'

    # 再次修改归一化方式为线性
    pcm.set_norm(Normalize(vmin=-20, vmax=20))
    # 断言颜色条的Y轴刻度的标度是否为线性
    assert cbar.ax.yaxis.get_scale() == 'linear'

    # 断言颜色条的外边框颜色是否与指定颜色一致
    assert cbar.outline.get_edgecolor() == mcolors.to_rgba('red')

    # 如果没有设置vmin/vmax，对数尺度将根据数据来调整，而不是默认范围(0, 1)
    pcm.norm = LogNorm()
    # 断言归一化对象的最小值是否接近于z的最小值
    assert pcm.norm.vmin == z.min()
    # 断言归一化对象的最大值是否接近于z的最大值


def test_colorbar_get_ticks_2():
    # 设置matplotlib配置项以禁用内部经典模式
    plt.rcParams['_internal.classic_mode'] = False
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制颜色网格图
    pc = ax.pcolormesh([[.05, .95]])
    # 在图形上创建颜色条
    cb = fig.colorbar(pc)
    # 断言颜色条的刻度位置是否与指定的值接近
    np.testing.assert_allclose(cb.get_ticks(), [0., 0.2, 0.4, 0.6, 0.8, 1.0])


def test_colorbar_inverted_ticks():
    # 创建包含两个子图的图形对象
    fig, axs = plt.subplots(2)
    # 获取第一个子图对象
    ax = axs[0]
    # 在子图上创建彩色网格图，使用对数标准化
    pc = ax.pcolormesh(10**np.arange(1, 5).reshape(2, 2), norm=LogNorm())
    # 添加颜色条到当前子图，使用 'both' 方式扩展
    cbar = fig.colorbar(pc, ax=ax, extend='both')
    # 获取颜色条上的刻度值
    ticks = cbar.get_ticks()
    # 颜色条反转y轴方向
    cbar.ax.invert_yaxis()
    # 断言所有主刻度值与颜色条上获取的刻度值相等
    np.testing.assert_allclose(ticks, cbar.get_ticks())

    # 获取第二个子图对象
    ax = axs[1]
    # 在子图上创建彩色网格图，使用默认颜色映射
    pc = ax.pcolormesh(np.arange(1, 5).reshape(2, 2))
    # 添加颜色条到当前子图，使用 'both' 方式扩展
    cbar = fig.colorbar(pc, ax=ax, extend='both')
    # 开启颜色条的次刻度
    cbar.minorticks_on()
    # 获取颜色条上的主刻度值
    ticks = cbar.get_ticks()
    # 获取颜色条上的次刻度值，断言其为 NumPy 数组
    minorticks = cbar.get_ticks(minor=True)
    assert isinstance(minorticks, np.ndarray)
    # 颜色条反转y轴方向
    cbar.ax.invert_yaxis()
    # 断言所有主刻度值与颜色条上获取的刻度值相等
    np.testing.assert_allclose(ticks, cbar.get_ticks())
    # 断言所有次刻度值与颜色条上获取的次刻度值相等
    np.testing.assert_allclose(minorticks, cbar.get_ticks(minor=True))
def test_mappable_no_alpha():
    # 创建一个包含单个子图和轴对象的图形
    fig, ax = plt.subplots()
    # 创建一个标量映射对象，用于颜色映射
    sm = cm.ScalarMappable(norm=mcolors.Normalize(), cmap='viridis')
    # 在图形中添加颜色条，使用当前轴对象
    fig.colorbar(sm, ax=ax)
    # 设置标量映射对象的颜色映射为 'plasma'
    sm.set_cmap('plasma')
    # 绘制图形
    plt.draw()


def test_mappable_2d_alpha():
    # 创建一个包含单个子图和轴对象的图形
    fig, ax = plt.subplots()
    # 创建一个 2x2 的二维数组
    x = np.arange(1, 5).reshape(2, 2)/4
    # 在轴上绘制二维图，并设置透明度为 x 中的值
    pc = ax.pcolormesh(x, alpha=x)
    # 在图形中添加颜色条，使用当前轴对象
    cb = fig.colorbar(pc, ax=ax)
    # 断言：颜色条的透明度应为 None，并且映射对象应保留原始的 alpha 数组
    assert cb.alpha is None
    assert pc.get_alpha() is x
    # 绘制图形（但不执行渲染）
    fig.draw_without_rendering()


def test_colorbar_label():
    """
    Test the label parameter. It should just be mapped to the xlabel/ylabel of
    the axes, depending on the orientation.
    """
    # 创建一个包含单个子图和轴对象的图形
    fig, ax = plt.subplots()
    # 在轴上显示一个简单的图像
    im = ax.imshow([[1, 2], [3, 4]])
    # 添加带标签的颜色条，标签为 'cbar'
    cbar = fig.colorbar(im, label='cbar')
    # 断言：颜色条的 y 轴标签应为 'cbar'
    assert cbar.ax.get_ylabel() == 'cbar'
    # 移除颜色条的标签
    cbar.set_label(None)
    # 断言：颜色条的 y 轴标签应为空字符串
    assert cbar.ax.get_ylabel() == ''
    # 重新设置颜色条的标签为 'cbar 2'
    cbar.set_label('cbar 2')
    # 断言：颜色条的 y 轴标签应为 'cbar 2'
    assert cbar.ax.get_ylabel() == 'cbar 2'

    # 创建一个不带标签的颜色条
    cbar2 = fig.colorbar(im, label=None)
    # 断言：颜色条的 y 轴标签应为空字符串
    assert cbar2.ax.get_ylabel() == ''

    # 创建一个水平方向的带标签的颜色条
    cbar3 = fig.colorbar(im, orientation='horizontal', label='horizontal cbar')
    # 断言：颜色条的 x 轴标签应为 'horizontal cbar'
    assert cbar3.ax.get_xlabel() == 'horizontal cbar'


@image_comparison(['colorbar_keeping_xlabel.png'], style='mpl20')
def test_keeping_xlabel():
    # github issue #23398 - xlabels being ignored in colorbar axis
    # 创建一个包含单个子图和轴对象的图形
    arr = np.arange(25).reshape((5, 5))
    fig, ax = plt.subplots()
    # 在轴上显示一个数组的图像
    im = ax.imshow(arr)
    # 添加带标签的颜色条，并设置标签为 'YLabel'
    cbar = plt.colorbar(im)
    cbar.ax.set_xlabel('Visible Xlabel')
    # 设置颜色条的标签为 'YLabel'
    cbar.set_label('YLabel')


@pytest.mark.parametrize("clim", [(-20000, 20000), (-32768, 0)])
def test_colorbar_int(clim):
    # Check that we cast to float early enough to not
    # overflow ``int16(20000) - int16(-20000)`` or
    # run into ``abs(int16(-32768)) == -32768``.
    # 创建一个包含单个子图和轴对象的图形
    fig, ax = plt.subplots()
    # 在轴上显示一个包含整数值的图像
    im = ax.imshow([[*map(np.int16, clim)]])
    # 添加颜色条到图形中
    fig.colorbar(im)
    # 断言：图像的规范化最小值和最大值应为 clim 中指定的值
    assert (im.norm.vmin, im.norm.vmax) == clim


def test_anchored_cbar_position_using_specgrid():
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]
    shrink = 0.5
    anchor_y = 0.3
    # right
    # 创建一个包含单个子图和轴对象的图形
    fig, ax = plt.subplots()
    # 在轴上显示一个等高线填充图
    cs = ax.contourf(data, levels=levels)
    # 添加使用网格规范的颜色条到图形中
    cbar = plt.colorbar(
            cs, ax=ax, use_gridspec=True,
            location='right', anchor=(1, anchor_y), shrink=shrink)

    # 计算轴和颜色条的位置信息
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (y1 - y0) * anchor_y + y0

    # 断言：颜色条的位置应符合预期
    np.testing.assert_allclose(
            [cy1, cy0],
            [y1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + y0 * shrink])

    # left
    # 创建一个包含单个子图和轴对象的图形
    fig, ax = plt.subplots()
    # 在轴上显示一个等高线填充图
    cs = ax.contourf(data, levels=levels)
    # 创建颜色条（colorbar），将其放置在图形中的指定位置
    cbar = plt.colorbar(
            cs, ax=ax, use_gridspec=True,
            location='left', anchor=(1, anchor_y), shrink=shrink)

    # 获取当前轴（ax）的位置信息：左下角为 (x0, y0)，右上角为 (x1, y1)
    # 计算锚点（anchor）的垂直位置（p0）
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (y1 - y0) * anchor_y + y0

    # 使用 np.testing.assert_allclose 检查计算出的颜色条位置与预期位置的接近程度
    np.testing.assert_allclose(
            [cy1, cy0],
            [y1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + y0 * shrink])

    # 创建新的图形（fig）和轴（ax），用于绘制图形
    # 绘制轮廓填充图（contourf）并添加颜色条
    shrink = 0.5
    anchor_x = 0.3
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(
            cs, ax=ax, use_gridspec=True,
            location='top', anchor=(anchor_x, 1), shrink=shrink)

    # 获取当前轴（ax）的位置信息：左下角为 (x0, y0)，右上角为 (x1, y1)
    # 计算锚点（anchor）的水平位置（p0）
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (x1 - x0) * anchor_x + x0

    # 使用 np.testing.assert_allclose 检查计算出的颜色条位置与预期位置的接近程度
    np.testing.assert_allclose(
            [cx1, cx0],
            [x1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + x0 * shrink])

    # 创建新的图形（fig）和轴（ax），用于绘制图形
    # 绘制轮廓填充图（contourf）并添加颜色条
    shrink = 0.5
    anchor_x = 0.3
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(
            cs, ax=ax, use_gridspec=True,
            location='bottom', anchor=(anchor_x, 1), shrink=shrink)

    # 获取当前轴（ax）的位置信息：左下角为 (x0, y0)，右上角为 (x1, y1)
    # 计算锚点（anchor）的水平位置（p0）
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (x1 - x0) * anchor_x + x0

    # 使用 np.testing.assert_allclose 检查计算出的颜色条位置与预期位置的接近程度
    np.testing.assert_allclose(
            [cx1, cx0],
            [x1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + x0 * shrink])
@image_comparison(['colorbar_change_lim_scale.png'], remove_text=True,
                  style='mpl20')
# 定义测试函数，比较两个图像是否相同，移除图像文本，使用样式 'mpl20'
def test_colorbar_change_lim_scale():
    # 创建包含两个子图的图像和轴对象
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    # 在第一个子图上创建伪彩色图，并生成颜色条
    pc = ax[0].pcolormesh(np.arange(100).reshape(10, 10)+1)
    cb = fig.colorbar(pc, ax=ax[0], extend='both')
    # 设置颜色条的 Y 轴刻度为对数尺度
    cb.ax.set_yscale('log')

    # 在第二个子图上创建伪彩色图，并生成颜色条
    pc = ax[1].pcolormesh(np.arange(100).reshape(10, 10)+1)
    cb = fig.colorbar(pc, ax=ax[1], extend='both')
    # 设置颜色条的 Y 轴限制在 [20, 90] 之间
    cb.ax.set_ylim([20, 90])


@check_figures_equal(extensions=["png"])
# 定义检查两个图像是否相同的函数，输出为 PNG 格式
def test_axes_handles_same_functions(fig_ref, fig_test):
    # 验证 cax 和 cb.ax 是否在功能上相同
    for nn, fig in enumerate([fig_ref, fig_test]):
        # 添加子图到图像
        ax = fig.add_subplot()
        # 在子图上创建伪彩色图，并在指定位置添加颜色条
        pc = ax.pcolormesh(np.ones(300).reshape(10, 30))
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cb = fig.colorbar(pc, cax=cax)
        # 根据索引号选择 cax 或 cb.ax，并设置其 Y 轴刻度和尺度
        if nn == 0:
            caxx = cax
        else:
            caxx = cb.ax
        caxx.set_yticks(np.arange(0, 20))
        caxx.set_yscale('log')
        caxx.set_position([0.92, 0.1, 0.02, 0.7])


def test_inset_colorbar_layout():
    # 创建包含嵌入颜色条的图像和轴对象
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3, 6))
    # 在轴对象上创建图像，并在指定位置添加嵌入式颜色条
    pc = ax.imshow(np.arange(100).reshape(10, 10))
    cax = ax.inset_axes([1.02, 0.1, 0.03, 0.8])
    cb = fig.colorbar(pc, cax=cax)

    fig.draw_without_rendering()
    # 验证颜色条的位置是否满足特定的边界要求
    np.testing.assert_allclose(cb.ax.get_position().bounds,
                               [0.87, 0.342, 0.0237, 0.315], atol=0.01)
    # 断言颜色条的轴对象是否是主轴对象的子对象
    assert cb.ax in ax.child_axes


@image_comparison(['colorbar_twoslope.png'], remove_text=True,
                  style='mpl20')
# 定义测试函数，比较两个图像是否相同，移除图像文本，使用样式 'mpl20'
def test_twoslope_colorbar():
    # 注意第二个刻度 = 20，在颜色条中间（白色）
    # 底部和顶部不应该有刻度
    fig, ax = plt.subplots()
    # 创建自定义的双斜坡归一化对象
    norm = mcolors.TwoSlopeNorm(20, 5, 95)
    # 在轴对象上创建伪彩色图，并生成颜色条
    pc = ax.pcolormesh(np.arange(1, 11), np.arange(1, 11),
                       np.arange(100).reshape(10, 10),
                       norm=norm, cmap='RdBu_r')
    fig.colorbar(pc)


@check_figures_equal(extensions=["png"])
# 定义检查两个图像是否相同的函数，输出为 PNG 格式
def test_remove_cb_whose_mappable_has_no_figure(fig_ref, fig_test):
    # 添加子图到图像
    ax = fig_test.add_subplot()
    # 在子图上创建颜色映射对象，并添加颜色条
    cb = fig_test.colorbar(cm.ScalarMappable(), cax=ax)
    # 移除颜色条
    cb.remove()


def test_aspects():
    # 创建包含多个子图的图像和轴对象
    fig, ax = plt.subplots(3, 2, figsize=(8, 8))
    aspects = [20, 20, 10]
    extends = ['neither', 'both', 'both']
    cb = [[None, None, None], [None, None, None]]
    # 遍历方向和宽高比的组合
    for nn, orient in enumerate(['vertical', 'horizontal']):
        for mm, (aspect, extend) in enumerate(zip(aspects, extends)):
            # 在轴对象上创建伪彩色图，并生成颜色条
            pc = ax[mm, nn].pcolormesh(np.arange(100).reshape(10, 10))
            cb[nn][mm] = fig.colorbar(pc, ax=ax[mm, nn], orientation=orient,
                                      aspect=aspect, extend=extend)
    fig.draw_without_rendering()
    # 检查比例是否正确：
    np.testing.assert_almost_equal(cb[0][1].ax.get_position().height,
                                   cb[0][0].ax.get_position().height * 0.9,
                                   decimal=2)
    # 水平方向
    np.testing.assert_almost_equal(cb[1][1].ax.get_position().width,
                                   cb[1][0].ax.get_position().width * 0.9,
                                   decimal=2)
    # 检查正确的长宽比：
    # 获取第一个颜色条的位置信息
    pos = cb[0][0].ax.get_position(original=False)
    np.testing.assert_almost_equal(pos.height, pos.width * 20, decimal=2)
    # 获取第二个颜色条的位置信息
    pos = cb[1][0].ax.get_position(original=False)
    np.testing.assert_almost_equal(pos.height * 20, pos.width, decimal=2)
    # 当长宽比为10而不是20时，检查宽度是否是高度的两倍
    np.testing.assert_almost_equal(
        cb[0][0].ax.get_position(original=False).width * 2,
        cb[0][2].ax.get_position(original=False).width, decimal=2)
    np.testing.assert_almost_equal(
        cb[1][0].ax.get_position(original=False).height * 2,
        cb[1][2].ax.get_position(original=False).height, decimal=2)
@image_comparison(['proportional_colorbars.png'], remove_text=True,
                  style='mpl20')
def test_proportional_colorbars():

    # 创建 X 和 Y 的等距数组，用于生成网格
    x = y = np.arange(-3.0, 3.01, 0.025)
    X, Y = np.meshgrid(x, y)
    
    # 创建两个高斯分布的 Z 值
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    
    # 计算 Z 值的差，并放大两倍
    Z = (Z1 - Z2) * 2

    # 设置等高线的级别
    levels = [-1.25, -0.5, -0.125, 0.125, 0.5, 1.25]
    
    # 创建自定义的颜色映射
    cmap = mcolors.ListedColormap(
        ['0.3', '0.5', 'white', 'lightblue', 'steelblue'])
    
    # 设置颜色映射的下限和上限颜色
    cmap.set_under('darkred')
    cmap.set_over('crimson')
    
    # 创建边界规范化对象
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # 设置颜色条的扩展方式
    extends = ['neither', 'both']
    spacings = ['uniform', 'proportional']
    
    # 创建子图和轴对象
    fig, axs = plt.subplots(2, 2)
    
    # 在每个子图中绘制等高线填充图和相应的颜色条
    for i in range(2):
        for j in range(2):
            CS3 = axs[i, j].contourf(X, Y, Z, levels, cmap=cmap, norm=norm,
                                     extend=extends[i])
            fig.colorbar(CS3, spacing=spacings[j], ax=axs[i, j])


@image_comparison(['extend_drawedges.png'], remove_text=True, style='mpl20')
def test_colorbar_extend_drawedges():
    # 定义不同的颜色条设置参数
    params = [
        ('both', 1, [[[1.1, 0], [1.1, 1]],
                     [[2, 0], [2, 1]],
                     [[2.9, 0], [2.9, 1]]]),
        ('min', 0, [[[1.1, 0], [1.1, 1]],
                    [[2, 0], [2, 1]]]),
        ('max', 0, [[[2, 0], [2, 1]],
                    [[2.9, 0], [2.9, 1]]]),
        ('neither', -1, [[[2, 0], [2, 1]]]),
    ]

    # 设置全局参数，调整轴线宽度
    plt.rcParams['axes.linewidth'] = 2

    # 创建包含两个子图的图形对象
    fig = plt.figure(figsize=(10, 4))
    subfigs = fig.subfigures(1, 2)

    # 遍历水平和垂直方向的子图
    for orientation, subfig in zip(['horizontal', 'vertical'], subfigs):
        # 根据方向设置子图布局
        if orientation == 'horizontal':
            axs = subfig.subplots(4, 1)
        else:
            axs = subfig.subplots(1, 4)
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

        # 在每个子图中设置颜色条和边界的验证
        for ax, (extend, coloroffset, res) in zip(axs, params):
            # 使用 viridis 颜色映射创建颜色条和规范化对象
            cmap = mpl.colormaps["viridis"]
            bounds = np.arange(5)
            nb_colors = len(bounds) + coloroffset
            colors = cmap(np.linspace(100, 255, nb_colors).astype(int))
            cmap, norm = mcolors.from_levels_and_colors(bounds, colors,
                                                        extend=extend)

            # 创建颜色条对象
            cbar = Colorbar(ax, cmap=cmap, norm=norm, orientation=orientation,
                            drawedges=True)
            
            # 根据方向设置轴限制
            if orientation == 'horizontal':
                ax.set_xlim(1.1, 2.9)
            else:
                ax.set_ylim(1.1, 2.9)
                res = np.array(res)[:, :, [1, 0]]
            
            # 验证颜色条分隔线的段落
            np.testing.assert_array_equal(cbar.dividers.get_segments(), res)


@image_comparison(['contourf_extend_patches.png'], remove_text=True,
                  style='mpl20')
def test_colorbar_contourf_extend_patches():
    # 定义参数列表，每个元素包含渐变方向、级别数和填充图案列表
    params = [
        ('both', 5, ['\\', '//']),
        ('min', 7, ['+']),
        ('max', 2, ['|', '-', '/', '\\', '//']),
        ('neither', 10, ['//', '\\', '||']),
    ]

    # 设置全局参数，设置坐标轴线宽度为2
    plt.rcParams['axes.linewidth'] = 2

    # 创建一个大小为10x4英寸的图形对象
    fig = plt.figure(figsize=(10, 4))
    # 创建包含2个子图的图形对象
    subfigs = fig.subfigures(1, 2)
    # 调整子图的位置，使其左边界、底边界、右边界和上边界分别为0.05和0.95
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

    # 在指定范围内生成均匀间隔的数据点
    x = np.linspace(-2, 3, 50)
    y = np.linspace(-2, 3, 30)
    # 生成二维数组的数据，其中z的值为x和y的三角函数值之和
    z = np.cos(x[np.newaxis, :]) + np.sin(y[:, np.newaxis])

    # 获取名为"viridis"的色彩映射对象
    cmap = mpl.colormaps["viridis"]
    # 对于水平和垂直两种方向，分别处理每个子图
    for orientation, subfig in zip(['horizontal', 'vertical'], subfigs):
        # 在每个子图上创建2x2的轴对象，并扁平化为一维数组
        axs = subfig.subplots(2, 2).ravel()
        for ax, (extend, levels, hatches) in zip(axs, params):
            # 在轴对象上绘制二维等高线填充图，并设置级别数、填充图案和扩展方式
            cs = ax.contourf(x, y, z, levels, hatches=hatches,
                             cmap=cmap, extend=extend)
            # 在子图上创建颜色条，方向与轴对象的方向一致
            subfig.colorbar(cs, ax=ax, orientation=orientation, fraction=0.4,
                            extendfrac=0.2, aspect=5)


def test_negative_boundarynorm():
    # 创建大小为1x3英寸的图形对象和相应的轴对象
    fig, ax = plt.subplots(figsize=(1, 3))
    # 获取名为"viridis"的色彩映射对象
    cmap = mpl.colormaps["viridis"]

    # 创建负边界标准化对象，并使用给定的边界数和色彩映射中的颜色数
    clevs = np.arange(-94, -85)
    norm = BoundaryNorm(clevs, cmap.N)
    # 在图形对象上创建颜色条，并指定颜色条的轴对象
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    # 断言颜色条的y轴限制与给定的边界数组的首尾元素相等
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    # 断言颜色条的y轴刻度与给定的边界数组相等
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)

    # 依次处理其他的边界数组和负边界标准化对象，重复上述过程
    clevs = np.arange(85, 94)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)

    clevs = np.arange(-3, 3)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)

    clevs = np.arange(-8, 1)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)


def test_centerednorm():
    # 测试默认的中心化标准化对象，用于数据都相等的情况下自动扩展非奇异限制
    fig, ax = plt.subplots(figsize=(1, 3))

    # 创建中心化标准化对象
    norm = mcolors.CenteredNorm()
    # 在轴对象上绘制颜色网格，并使用中心化标准化对象
    mappable = ax.pcolormesh(np.zeros((3, 3)), norm=norm)
    # 在图形对象上创建颜色条
    fig.colorbar(mappable)
    # 断言中心化标准化对象的限制范围为[-0.1, 0.1]
    assert (norm.vmin, norm.vmax) == (-0.1, 0.1)


@image_comparison(['nonorm_colorbars.svg'], style='mpl20')
def test_nonorm():
    # 设置SVG输出中的字体类型
    plt.rcParams['svg.fonttype'] = 'none'
    data = [1, 2, 3, 4, 5]

    # 创建大小为6x1英寸的图形对象和相应的轴对象
    fig, ax = plt.subplots(figsize=(6, 1))
    # 调整子图的底边界
    fig.subplots_adjust(bottom=0.5)

    # 创建非标准化对象，指定数据的最小值和最大值
    norm = NoNorm(vmin=min(data), vmax=max(data))
    # 对色彩映射进行重新采样，使其具有与数据长度相等的颜色数
    cmap = mpl.colormaps["viridis"].resampled(len(data))
    # 创建标量映射对象，使用非标准化对象和重新采样后的色彩映射
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    # 创建一个颜色条（colorbar），将其添加到图形（fig）上，并指定它的轴（ax）为ax，方向为水平方向
    cbar = fig.colorbar(mappable, cax=ax, orientation="horizontal")
@image_comparison(['test_boundaries.png'], remove_text=True,
                  style='mpl20')
# 定义名为 test_boundaries 的测试函数，用于比较生成的图像是否与参考图像一致
def test_boundaries():
    # 设定随机数种子，保证随机数可复现性
    np.random.seed(seed=19680808)
    # 创建一个大小为 2x2 的子图
    fig, ax = plt.subplots(figsize=(2, 2))
    # 使用随机数生成网格图，并将其赋值给变量 pc
    pc = ax.pcolormesh(np.random.randn(10, 10), cmap='RdBu_r')
    # 在图上添加颜色条，并设定颜色条的边界为 -3 到 3，共 7 个间隔
    cb = fig.colorbar(pc, ax=ax, boundaries=np.linspace(-3, 3, 7))


def test_colorbar_no_warning_rcparams_grid_true():
    # 解决 GitHub 问题 #21723：当 mpl 样式设置 'axes.grid' = True 时，
    # fig.colorbar 会因为 pcolor() 和 pcolormesh() 的自动删除网格而发出警告。
    # 这个问题已通过 PR #22216 修复。
    plt.rcParams['axes.grid'] = True
    # 创建一个图和相应的轴对象
    fig, ax = plt.subplots()
    # 在轴上关闭网格线
    ax.grid(False)
    # 创建一个简单的网格图并将其赋值给变量 im
    im = ax.pcolormesh([0, 1], [0, 1], [[1]])
    # 确保 fig.colorbar 不会触发警告
    fig.colorbar(im)


def test_colorbar_set_formatter_locator():
    # 检查定位器属性是否与轴上的一致
    fig, ax = plt.subplots()
    # 创建一个随机数生成的网格图并将其赋值给变量 pc
    pc = ax.pcolormesh(np.random.randn(10, 10))
    # 添加颜色条并将其赋值给变量 cb
    cb = fig.colorbar(pc)
    # 设置主要定位器和次要定位器
    cb.ax.yaxis.set_major_locator(FixedLocator(np.arange(10)))
    cb.ax.yaxis.set_minor_locator(FixedLocator(np.arange(0, 10, 0.2)))
    # 断言颜色条的定位器与轴上的主要定位器和次要定位器一致
    assert cb.locator is cb.ax.yaxis.get_major_locator()
    assert cb.minorlocator is cb.ax.yaxis.get_minor_locator()
    # 设置主要格式化器和次要格式化器
    cb.ax.yaxis.set_major_formatter(LogFormatter())
    cb.ax.yaxis.set_minor_formatter(LogFormatter())
    # 断言颜色条的格式化器与轴上的主要格式化器和次要格式化器一致
    assert cb.formatter is cb.ax.yaxis.get_major_formatter()
    assert cb.minorformatter is cb.ax.yaxis.get_minor_formatter()

    # 检查设置器的工作是否符合预期
    loc = FixedLocator(np.arange(7))
    cb.locator = loc
    assert cb.ax.yaxis.get_major_locator() is loc
    loc = FixedLocator(np.arange(0, 7, 0.1))
    cb.minorlocator = loc
    assert cb.ax.yaxis.get_minor_locator() is loc
    fmt = LogFormatter()
    cb.formatter = fmt
    assert cb.ax.yaxis.get_major_formatter() is fmt
    fmt = LogFormatter()
    cb.minorformatter = fmt
    assert cb.ax.yaxis.get_minor_formatter() is fmt


@image_comparison(['colorbar_extend_alpha.png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
# 定义名为 test_colorbar_extend_alpha 的测试函数，用于比较生成的图像是否与参考图像一致
def test_colorbar_extend_alpha():
    # 创建一个图和相应的轴对象
    fig, ax = plt.subplots()
    # 创建一个简单的图像，并将其赋值给变量 im
    im = ax.imshow([[0, 1], [2, 3]], alpha=0.3, interpolation="none")
    # 在图上添加颜色条，并设置其延伸方式为 'both'，边界为 [0.5, 1.5, 2.5]
    fig.colorbar(im, extend='both', boundaries=[0.5, 1.5, 2.5])


def test_offset_text_loc():
    # 使用 'mpl20' 样式
    plt.style.use('mpl20')
    # 创建一个图和相应的轴对象
    fig, ax = plt.subplots()
    # 设定随机数种子，保证随机数可复现性
    np.random.seed(seed=19680808)
    # 创建一个大数值的网格图并将其赋值给变量 pc
    pc = ax.pcolormesh(np.random.randn(10, 10)*1e6)
    # 在右侧添加一个颜色条，并设定延伸方式为 'max'
    cb = fig.colorbar(pc, location='right', extend='max')
    # 绘制图像但不渲染
    fig.draw_without_rendering()
    # 断言偏移文本位于正确的位置，即位于颜色条轴的上方
    assert cb.ax.yaxis.offsetText.get_position()[1] > ax.bbox.y1


def test_title_text_loc():
    # 使用 'mpl20' 样式
    plt.style.use('mpl20')
    # 创建一个图和相应的轴对象
    fig, ax = plt.subplots()
    # 设定随机数种子，保证随机数可复现性
    np.random.seed(seed=19680808)
    # 创建一个随机数生成的网格图并将其赋值给变量 pc
    pc = ax.pcolormesh(np.random.randn(10, 10))
    # 创建一个颜色条(colorbar)，关联到图中的pc对象，位置在图的右侧，颜色条的长度与图的数据范围相匹配
    cb = fig.colorbar(pc, location='right', extend='max')
    
    # 设置颜色条上方的标题为'Aardvark'
    cb.ax.set_title('Aardvark')
    
    # 在不重新渲染整个图的情况下绘制图形
    fig.draw_without_rendering()
    
    # 检查标题是否正确地位于颜色条轴的上方，包括其延伸三角形...
    assert (cb.ax.title.get_window_extent(fig.canvas.get_renderer()).ymax >
            cb.ax.spines['outline'].get_window_extent().ymax)
# 装饰器，用于检查测试函数中颜色条参数的正确性，指定了扩展名为 "png"
@check_figures_equal(extensions=["png"])
# 测试通过位置参数fig_ref和fig_test传递的图表对象是否相等
def test_passing_location(fig_ref, fig_test):
    # 在参考图表(fig_ref)上添加一个子图
    ax_ref = fig_ref.add_subplot()
    # 在子图上展示一个2x2的图像，数据为[[0, 1], [2, 3]]
    im = ax_ref.imshow([[0, 1], [2, 3]])
    # 在参考图表上添加一个水平方向的颜色条，位置在图表顶部
    ax_ref.figure.colorbar(im, cax=ax_ref.inset_axes([0, 1.05, 1, 0.05]),
                           orientation="horizontal", ticklocation="top")

    # 在测试图表(fig_test)上添加一个子图
    ax_test = fig_test.add_subplot()
    # 在子图上展示一个2x2的图像，数据为[[0, 1], [2, 3]]
    im = ax_test.imshow([[0, 1], [2, 3]])
    # 在测试图表上添加一个水平方向的颜色条，位置在图表顶部
    ax_test.figure.colorbar(im, cax=ax_test.inset_axes([0, 1.05, 1, 0.05]),
                            location="top")


# 参数化测试函数，验证颜色条设置中可能出现的错误情况
@pytest.mark.parametrize("kwargs,error,message", [
    ({'location': 'top', 'orientation': 'vertical'}, TypeError,
     "location and orientation are mutually exclusive"),
    ({'location': 'top', 'orientation': 'vertical', 'cax': True}, TypeError,
     "location and orientation are mutually exclusive"),  # 与上面的情况不同
    ({'ticklocation': 'top', 'orientation': 'vertical', 'cax': True},
     ValueError, "'top' is not a valid value for position"),
    ({'location': 'top', 'extendfrac': (0, None)}, ValueError,
     "invalid value for extendfrac"),
    ])
# 测试颜色条设置中可能引发的错误
def test_colorbar_errors(kwargs, error, message):
    # 创建一个图表和一个子图
    fig, ax = plt.subplots()
    # 在子图上展示一个2x2的图像，数据为[[0, 1], [2, 3]]
    im = ax.imshow([[0, 1], [2, 3]])
    # 如果kwargs中有'cax'参数且为True，则使用子图的内嵌轴作为颜色条的位置
    if kwargs.get('cax', None) is True:
        kwargs['cax'] = ax.inset_axes([0, 1.05, 1, 0.05])
    # 断言颜色条设置过程中是否会抛出指定类型的错误，并匹配指定的错误消息
    with pytest.raises(error, match=message):
        fig.colorbar(im, **kwargs)


# 测试颜色条设置中对轴参数的各种输入情况
def test_colorbar_axes_parmeters():
    # 创建包含两个子图的图表
    fig, ax = plt.subplots(2)
    # 在第一个子图上展示一个1x2的图像，数据为[[0, 1], [2, 3]]
    im = ax[0].imshow([[0, 1], [2, 3]])
    # 验证颜色条是否接受任何形式的轴序列作为参数
    fig.colorbar(im, ax=ax)
    fig.colorbar(im, ax=ax[0])
    fig.colorbar(im, ax=[_ax for _ax in ax])
    fig.colorbar(im, ax=(ax[0], ax[1]))
    fig.colorbar(im, ax={i: _ax for i, _ax in enumerate(ax)}.values())
    # 在不进行渲染的情况下绘制图表
    fig.draw_without_rendering()


# 测试在不正确的图表上调用颜色条设置时的情况
def test_colorbar_wrong_figure():
    # 创建一个布局为"tight"的图表
    fig_tl = plt.figure(layout="tight")
    # 创建一个布局为"constrained"的图表
    fig_cl = plt.figure(layout="constrained")
    # 在"constrained"图表上添加一个子图，并展示一个1x1的图像，数据为[[0, 1]]
    im = fig_cl.add_subplot().imshow([[0, 1]])
    # 确保在"constrained"图表上不会设置网格控制的颜色条，否则会崩溃
    with pytest.warns(UserWarning, match="different Figure"):
        fig_tl.colorbar(im)
    # 在不进行渲染的情况下绘制两个图表
    fig_tl.draw_without_rendering()
    fig_cl.draw_without_rendering()


# 测试颜色条的格式化字符串和旧版本兼容性
def test_colorbar_format_string_and_old():
    # 在当前图表上展示一个1x1的图像，数据为[[0, 1]]
    plt.imshow([[0, 1]])
    # 添加一个颜色条，并指定格式化字符串为"{x}%"
    cb = plt.colorbar(format="{x}%")
    # 断言颜色条的格式化对象类型为StrMethodFormatter
    assert isinstance(cb._formatter, StrMethodFormatter)
```