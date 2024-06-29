# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_contour.py`

```py
# 导入必要的模块和库
import datetime  # 导入处理日期和时间的模块
import platform  # 导入获取平台信息的模块
import re  # 导入支持正则表达式操作的模块
from unittest import mock  # 导入用于模拟测试的模块

import contourpy  # 导入ContourPy库
import numpy as np  # 导入NumPy库，用于数值计算
from numpy.testing import (  # 导入NumPy测试模块中的几个断言函数
    assert_array_almost_equal, assert_array_almost_equal_nulp, assert_array_equal)
import matplotlib as mpl  # 导入Matplotlib库的核心模块
from matplotlib import pyplot as plt, rc_context, ticker  # 导入Matplotlib的绘图接口、设置和刻度
from matplotlib.colors import LogNorm, same_color  # 导入Matplotlib的颜色映射和颜色函数
import matplotlib.patches as mpatches  # 导入Matplotlib的图形修饰模块
from matplotlib.testing.decorators import check_figures_equal, image_comparison  # 导入Matplotlib的测试装饰器
import pytest  # 导入用于Python测试的Pytest库


# 测试辅助函数，用于检查从多个集合到单个集合的过渡，一旦废弃的旧布局过期，将删除
def _maybe_split_collections(do_split):
    if not do_split:
        return
    for fig in map(plt.figure, plt.get_fignums()):
        for ax in fig.axes:
            for coll in ax.collections:
                if isinstance(coll, mpl.contour.ContourSet):
                    with pytest.warns(mpl._api.MatplotlibDeprecationWarning):
                        coll.collections


# 测试1D情况下的等高线形状是否有效
def test_contour_shape_1d_valid():

    x = np.arange(10)  # 创建一个包含10个元素的数组
    y = np.arange(9)  # 创建一个包含9个元素的数组
    z = np.random.random((9, 10))  # 创建一个9x10的随机数数组

    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    ax.contour(x, y, z)  # 绘制二维等高线


# 测试2D情况下的等高线形状是否有效
def test_contour_shape_2d_valid():

    x = np.arange(10)  # 创建一个包含10个元素的数组
    y = np.arange(9)  # 创建一个包含9个元素的数组
    xg, yg = np.meshgrid(x, y)  # 创建x和y的网格
    z = np.random.random((9, 10))  # 创建一个9x10的随机数数组

    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    ax.contour(xg, yg, z)  # 绘制二维等高线


# 参数化测试，测试不同输入情况下的等高线形状错误
@pytest.mark.parametrize("args, message", [
    ((np.arange(9), np.arange(9), np.empty((9, 10))),
     'Length of x (9) must match number of columns in z (10)'),
    ((np.arange(10), np.arange(10), np.empty((9, 10))),
     'Length of y (10) must match number of rows in z (9)'),
    ((np.empty((10, 10)), np.arange(10), np.empty((9, 10))),
     'Number of dimensions of x (2) and y (1) do not match'),
    ((np.arange(10), np.empty((10, 10)), np.empty((9, 10))),
     'Number of dimensions of x (1) and y (2) do not match'),
    ((np.empty((9, 9)), np.empty((9, 10)), np.empty((9, 10))),
     'Shapes of x (9, 9) and z (9, 10) do not match'),
    ((np.empty((9, 10)), np.empty((9, 9)), np.empty((9, 10))),
     'Shapes of y (9, 9) and z (9, 10) do not match'),
    ((np.empty((3, 3, 3)), np.empty((3, 3, 3)), np.empty((9, 10))),
     'Inputs x and y must be 1D or 2D, not 3D'),
    ((np.empty((3, 3, 3)), np.empty((3, 3, 3)), np.empty((3, 3, 3))),
     'Input z must be 2D, not 3D'),
    (([[0]],),  # github issue 8197
     'Input z must be at least a (2, 2) shaped array, but has shape (1, 1)'),
    (([0], [0], [[0]]),
     'Input z must be at least a (2, 2) shaped array, but has shape (1, 1)'),
])
def test_contour_shape_error(args, message):
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    with pytest.raises(TypeError, match=re.escape(message)):
        ax.contour(*args)  # 传递参数并绘制等高线


# 测试当levels为空时是否没有警告
def test_contour_no_valid_levels():
    fig, ax = plt.subplots()  # 创建一个新的图形和坐标轴
    ax.contour(np.random.rand(9, 9), levels=[])  # 绘制二维随机数组的等高线，levels为空列表
    # 测试当levels被给定但不在z的范围内时是否没有警告
    ax.contour(np.random.rand(9, 9), levels=[0.1, 0.5, 0.9])
    # 使用 np.arange(81) 创建一个 9x9 的数组，然后通过 ax.contour 绘制等高线图，指定绘制等级为 100
    cs = ax.contour(np.arange(81).reshape((9, 9)), levels=[100])
    
    # 如果 fmt 参数被指定，则在等高线 cs 上添加标签，使用格式化字符串 {100: '%1.2f'}
    ax.clabel(cs, fmt={100: '%1.2f'})
    
    # 在坐标轴 ax 上绘制一个由全部元素为 1 的 9x9 数组的等高线，表示 z 是均匀的情况
    ax.contour(np.ones((9, 9)))
def test_contour_Nlevels():
    # A scalar levels arg or kwarg should trigger auto level generation.
    # https://github.com/matplotlib/matplotlib/issues/11913
    # 创建一个 3x4 的数组 z
    z = np.arange(12).reshape((3, 4))
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制 z 的等高线图，自动生成 5 个等级
    cs1 = ax.contour(z, 5)
    # 断言生成的等级数量大于 1
    assert len(cs1.levels) > 1
    # 绘制 z 的等高线图，指定 5 个等级
    cs2 = ax.contour(z, levels=5)
    # 断言两种方式生成的等级数组相同
    assert (cs1.levels == cs2.levels).all()


@check_figures_equal(extensions=['png'])
def test_contour_set_paths(fig_test, fig_ref):
    # 在测试和参考图形上创建等高线对象 cs_test 和 cs_ref
    cs_test = fig_test.subplots().contour([[0, 1], [1, 2]])
    cs_ref = fig_ref.subplots().contour([[1, 0], [2, 1]])

    # 设置测试图形上等高线对象的路径为参考图形上等高线对象的路径
    cs_test.set_paths(cs_ref.get_paths())


@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(['contour_manual_labels'], remove_text=True, style='mpl20', tol=0.26)
def test_contour_manual_labels(split_collections):
    # 创建网格 x, y 和相应的 z 数组
    x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 10))
    z = np.max(np.dstack([abs(x), abs(y)]), 2)

    # 创建图形，设置图形大小和 dpi
    plt.figure(figsize=(6, 2), dpi=200)
    # 绘制 x, y, z 的等高线图
    cs = plt.contour(x, y, z)

    # 可能分割集合，根据参数决定是否分割
    _maybe_split_collections(split_collections)

    # 手动添加标签到等高线
    pts = np.array([(1.0, 3.0), (1.0, 4.4), (1.0, 6.0)])
    plt.clabel(cs, manual=pts)
    pts = np.array([(2.0, 3.0), (2.0, 4.4), (2.0, 6.0)])
    plt.clabel(cs, manual=pts, fontsize='small', colors=('r', 'g'))


def test_contour_manual_moveto():
    # 创建线性空间数组 x 和 y
    x = np.linspace(-10, 10)
    y = np.linspace(-10, 10)

    # 创建网格 X, Y 和相应的 Z 数组
    X, Y = np.meshgrid(x, y)

    # 计算 Z 值
    Z = X**2 * 1 / Y**2 - 1

    # 绘制 Z 的等高线图，指定特定的等级
    contours = plt.contour(X, Y, Z, levels=[0, 100])

    # 指定一个点，用于手动添加标签
    point = (1.3, 1)
    clabels = plt.clabel(contours, manual=[point])

    # 断言所选择的等高线为 0 而不是 100
    assert clabels[0].get_text() == "0"


@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(['contour_disconnected_segments'],
                  remove_text=True, style='mpl20', extensions=['png'])
def test_contour_label_with_disconnected_segments(split_collections):
    # 创建网格 x, y 和相应的 z 数组
    x, y = np.mgrid[-1:1:21j, -1:1:21j]
    z = 1 / np.sqrt(0.01 + (x + 0.3) ** 2 + y ** 2)
    z += 1 / np.sqrt(0.01 + (x - 0.3) ** 2 + y ** 2)

    # 创建一个新的图形
    plt.figure()
    # 绘制 x, y, z 的等高线图，指定一个特定的等级
    cs = plt.contour(x, y, z, levels=[7])

    # 添加标签以使旧样式失效
    _maybe_split_collections(split_collections)

    # 手动添加标签到等高线
    cs.clabel(manual=[(0.2, 0.1)])

    # 再次根据参数决定是否分割集合
    _maybe_split_collections(split_collections)


@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(['contour_manual_colors_and_levels.png'], remove_text=True,
                  tol=0.018 if platform.machine() == 'arm64' else 0)
def test_given_colors_levels_and_extends(split_collections):
    # 当这个测试图像重新生成时，请移除此行。
    plt.rcParams['pcolormesh.snap'] = False

    # 创建 2x4 的子图数组 axs
    _, axs = plt.subplots(2, 4)

    # 创建数据数组
    data = np.arange(12).reshape(3, 4)

    # 指定颜色和等级
    colors = ['red', 'yellow', 'pink', 'blue', 'black']
    levels = [2, 4, 8, 10]
    # 使用 enumerate 函数遍历 axs.flat 中的轴对象及其索引
    for i, ax in enumerate(axs.flat):
        # 判断当前轴是否填充颜色，通过 i 的奇偶性来确定
        filled = i % 2 == 0.
        # 根据 i 的值计算 extend 的取值，确定颜色映射的扩展方式
        extend = ['neither', 'min', 'max', 'both'][i // 2]

        if filled:
            # 如果轴填充颜色，根据 extend 的值选择相应的颜色映射范围
            # 'neither' 或 'max' 时取第一个颜色，'min' 或 'neither' 时取最后一个颜色
            first_color = 1 if extend in ['max', 'neither'] else None
            last_color = -1 if extend in ['min', 'neither'] else None
            # 在轴上创建填充颜色的等高线图，使用指定的颜色和范围
            c = ax.contourf(data, colors=colors[first_color:last_color],
                            levels=levels, extend=extend)
        else:
            # 如果轴不填充颜色，使用指定的颜色创建等高线图
            c = ax.contour(data, colors=colors[:-1],
                           levels=levels, extend=extend)

        # 在轴上添加颜色条，用于显示颜色映射
        plt.colorbar(c, ax=ax)

    # 调用 _maybe_split_collections 函数，处理可能存在的图形集合分离操作
    _maybe_split_collections(split_collections)
@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(['contour_log_locator.svg'], style='mpl20', remove_text=False)
def test_log_locator_levels(split_collections):
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 创建网格点
    N = 100
    x = np.linspace(-3.0, 3.0, N)
    y = np.linspace(-2.0, 2.0, N)
    X, Y = np.meshgrid(x, y)

    # 创建数据
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
    data = Z1 + 50 * Z2

    # 绘制等高线填充图，并使用对数定位器
    c = ax.contourf(data, locator=ticker.LogLocator())
    assert_array_almost_equal(c.levels, np.power(10.0, np.arange(-6, 3)))

    # 添加颜色条
    cb = fig.colorbar(c, ax=ax)
    assert_array_almost_equal(cb.ax.get_yticks(), c.levels)

    # 可能拆分集合对象
    _maybe_split_collections(split_collections)


@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(['contour_datetime_axis.png'], style='mpl20')
def test_contour_datetime_axis(split_collections):
    # 创建一个新的图形对象
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.98, bottom=.15)

    # 创建日期时间基准和数据
    base = datetime.datetime(2013, 1, 1)
    x = np.array([base + datetime.timedelta(days=d) for d in range(20)])
    y = np.arange(20)
    z1, z2 = np.meshgrid(np.arange(20), np.arange(20))
    z = z1 * z2

    # 绘制轮廓线图和填充图
    plt.subplot(221)
    plt.contour(x, y, z)
    plt.subplot(222)
    plt.contourf(x, y, z)

    # 扩展坐标轴数据
    x = np.repeat(x[np.newaxis], 20, axis=0)
    y = np.repeat(y[:, np.newaxis], 20, axis=1)
    plt.subplot(223)
    plt.contour(x, y, z)
    plt.subplot(224)
    plt.contourf(x, y, z)

    # 调整 x 轴刻度标签的方向和旋转
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)

    # 可能拆分集合对象
    _maybe_split_collections(split_collections)


@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(['contour_test_label_transforms.png'],
                  remove_text=True, style='mpl20', tol=1.1)
def test_labels(split_collections):
    # 从 pylab_examples 示例代码修改而来：contour_demo.py
    # 参见问题 #2475, #2843, 和 #2818 的解释
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))

    # 高斯差分
    Z = 10.0 * (Z2 - Z1)

    # 创建图形和轴对象
    fig, ax = plt.subplots(1, 1)
    CS = ax.contour(X, Y, Z)

    # 显示单位和数据单位
    disp_units = [(216, 177), (359, 290), (521, 406)]
    data_units = [(-2, .5), (0, -1.5), (2.8, 1)]

    # 可能拆分集合对象
    _maybe_split_collections(split_collections)

    # 添加轮廓线标签
    CS.clabel()

    # 在数据单位和显示单位位置添加标签
    for x, y in data_units:
        CS.add_label_near(x, y, inline=True, transform=None)

    for x, y in disp_units:
        CS.add_label_near(x, y, inline=True, transform=False)

    # 可能拆分集合对象
    _maybe_split_collections(split_collections)


def test_label_contour_start():
    # 设置数据和图形/轴，使得自动标记在轮廓线的起始位置添加标签
    pass
    # 创建一个带有单个子图的绘图对象，并设置 DPI 为 100
    _, ax = plt.subplots(dpi=100)
    # 在纬度和经度范围内生成均匀分布的网格点，用于绘制地图数据
    lats = lons = np.linspace(-np.pi / 2, np.pi / 2, 50)
    # 生成经度和纬度的网格
    lons, lats = np.meshgrid(lons, lats)
    # 生成波浪和均值数据
    wave = 0.75 * (np.sin(2 * lats) ** 8) * np.cos(4 * lons)
    mean = 0.5 * np.cos(2 * lats) * ((np.sin(2 * lats)) ** 2 + 2)
    data = wave + mean

    # 绘制等高线并返回等高线对象
    cs = ax.contour(lons, lats, data)

    # 使用 mock 模块中的 patch.object 方法，替换等高线对象的方法 _split_path_and_get_label_rotation
    # wraps 参数确保我们可以后续访问原始方法
    with mock.patch.object(
            cs, '_split_path_and_get_label_rotation',
            wraps=cs._split_path_and_get_label_rotation) as mocked_splitter:
        # 检验是否能够成功添加标签
        cs.clabel(fontsize=9)

    # 验证至少调用了一次 _split_path_and_get_label_rotation 方法，且参数 idx=0 至少被调用过一次
    idxs = [cargs[0][1] for cargs in mocked_splitter.call_args_list]
    assert 0 in idxs
@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(['contour_corner_mask_False.png', 'contour_corner_mask_True.png'],
                  remove_text=True, tol=1.88)
def test_corner_mask(split_collections):
    # 设置参数
    n = 60  # 网格点数
    mask_level = 0.95  # 掩模级别
    noise_amp = 1.0  # 噪声振幅
    np.random.seed([1])  # 设置随机种子

    # 创建二维坐标网格
    x, y = np.meshgrid(np.linspace(0, 2.0, n), np.linspace(0, 2.0, n))
    # 生成数据
    z = np.cos(7*x)*np.sin(8*y) + noise_amp*np.random.rand(n, n)
    # 创建随机掩模
    mask = np.random.rand(n, n) >= mask_level
    # 应用掩模到数据上
    z = np.ma.array(z, mask=mask)

    # 针对每种角落掩模情况创建图像
    for corner_mask in [False, True]:
        plt.figure()
        plt.contourf(z, corner_mask=corner_mask)

    # 可能拆分集合操作
    _maybe_split_collections(split_collections)


def test_contourf_decreasing_levels():
    # github问题编号5477。
    z = [[0.1, 0.3], [0.5, 0.7]]
    plt.figure()
    # 检查是否引发值错误异常
    with pytest.raises(ValueError):
        plt.contourf(z, [1.0, 0.0])


def test_contourf_symmetric_locator():
    # github问题编号7271
    z = np.arange(12).reshape((3, 4))
    locator = plt.MaxNLocator(nbins=4, symmetric=True)
    # 创建等高线填充图，并使用对称定位器
    cs = plt.contourf(z, locator=locator)
    # 断言等高线级别与预期的线性空间一致
    assert_array_almost_equal(cs.levels, np.linspace(-12, 12, 5))


def test_circular_contour_warning():
    # 检查几乎圆形的等高线不会引发警告
    x, y = np.meshgrid(np.linspace(-2, 2, 4), np.linspace(-2, 2, 4))
    r = np.hypot(x, y)
    plt.figure()
    cs = plt.contour(x, y, r)
    plt.clabel(cs)


@pytest.mark.parametrize("use_clabeltext, contour_zorder, clabel_zorder",
                         [(True, 123, 1234), (False, 123, 1234),
                          (True, 123, None), (False, 123, None)])
def test_clabel_zorder(use_clabeltext, contour_zorder, clabel_zorder):
    x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 10))
    z = np.max(np.dstack([abs(x), abs(y)]), 2)

    # 创建包含两个子图的图形对象
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    # 绘制轮廓线
    cs = ax1.contour(x, y, z, zorder=contour_zorder)
    # 绘制轮廓填充图
    cs_filled = ax2.contourf(x, y, z, zorder=contour_zorder)
    # 标记轮廓线上的标签
    clabels1 = cs.clabel(zorder=clabel_zorder, use_clabeltext=use_clabeltext)
    # 标记轮廓填充图上的标签
    clabels2 = cs_filled.clabel(zorder=clabel_zorder,
                                use_clabeltext=use_clabeltext)

    # 如果标签顺序为None，预期标签顺序等于轮廓线顺序加2
    if clabel_zorder is None:
        expected_clabel_zorder = 2 + contour_zorder
    else:
        expected_clabel_zorder = clabel_zorder

    # 断言标签的绘制顺序
    for clabel in clabels1:
        assert clabel.get_zorder() == expected_clabel_zorder
    for clabel in clabels2:
        assert clabel.get_zorder() == expected_clabel_zorder


def test_clabel_with_large_spacing():
    # 当内联间距相对于等高线较大时，可能导致整个等高线被删除。
    # 在当前实现中，保留识别点之间的一条线段。
    # 这种行为值得重新考虑，但确保我们不产生无效路径，这会导致在调用clabel时出错。
    # 参见gh-27045获取更多信息
    x = y = np.arange(-3.0, 3.01, 0.05)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2 - Y**2)
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在指定的 X, Y 网格上绘制等高线，使用给定的 Z 值和指定的高度级别
    contourset = ax.contour(X, Y, Z, levels=[0.01, 0.2, .5, .8])
    # 在等高线图上添加标签，设定标签与等高线的间距为 100
    ax.clabel(contourset, inline_spacing=100)
# tol因为刻度恰好落在像素边界，所以小浮点数的变化会导致刻度位置翻转，决定了哪个像素获取刻度。
# floating point changes in tick location flip which pixel gets
# the tick.
@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(['contour_log_extension.png'],
                  remove_text=True, style='mpl20',
                  tol=1.444)
def test_contourf_log_extension(split_collections):
    # 当重新生成此测试图像时，请移除此行。
    plt.rcParams['pcolormesh.snap'] = False

    # 测试lognorm下contourf的扩展行为是否正确
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=0.95)

    # 创建数据集，包含大范围的数据，例如在1e-8到1e10之间
    data_exp = np.linspace(-7.5, 9.5, 1200)
    data = np.power(10, data_exp).reshape(30, 40)
    # 创建手动设置的levels，例如在1e-4到1e-6之间
    levels_exp = np.arange(-4., 7.)
    levels = np.power(10., levels_exp)

    # 原始数据
    c1 = ax1.contourf(data,
                      norm=LogNorm(vmin=data.min(), vmax=data.max()))
    # 仅显示指定levels的数据
    c2 = ax2.contourf(data, levels=levels,
                      norm=LogNorm(vmin=levels.min(), vmax=levels.max()),
                      extend='neither')
    # 扩展超出levels的数据
    c3 = ax3.contourf(data, levels=levels,
                      norm=LogNorm(vmin=levels.min(), vmax=levels.max()),
                      extend='both')
    cb = plt.colorbar(c1, ax=ax1)
    assert cb.ax.get_ylim() == (1e-8, 1e10)
    cb = plt.colorbar(c2, ax=ax2)
    assert_array_almost_equal_nulp(cb.ax.get_ylim(), np.array((1e-4, 1e6)))
    cb = plt.colorbar(c3, ax=ax3)

    _maybe_split_collections(split_collections)


@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(
    ['contour_addlines.png'], remove_text=True, style='mpl20',
    tol=0.15 if platform.machine() in ('aarch64', 'ppc64le', 's390x')
        else 0.03)
# 容忍度因为当清理颜色条上的刻度查找时，图像细微变化...
def test_contour_addlines(split_collections):
    # 当重新生成此测试图像时，请移除此行。
    plt.rcParams['pcolormesh.snap'] = False

    fig, ax = plt.subplots()
    np.random.seed(19680812)
    X = np.random.rand(10, 10)*10000
    pcm = ax.pcolormesh(X)
    # 添加1000以使颜色可见...
    cont = ax.contour(X+1000)
    cb = fig.colorbar(pcm)
    cb.add_lines(cont)
    assert_array_almost_equal(cb.ax.get_ylim(), [114.3091, 9972.30735], 3)

    _maybe_split_collections(split_collections)


@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(baseline_images=['contour_uneven'],
                  extensions=['png'], remove_text=True, style='mpl20')
def test_contour_uneven(split_collections):
    # 当重新生成此测试图像时，请移除此行。
    plt.rcParams['pcolormesh.snap'] = False

    z = np.arange(24).reshape(4, 6)
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    # 在第一个子图(axs[0])上创建填充等高线图，并设定填充区域的级别
    cs = ax.contourf(z, levels=[2, 4, 6, 10, 20])
    
    # 在第一个子图(axs[0])上创建颜色条，使其与等高线填充区域比例匹配
    fig.colorbar(cs, ax=ax, spacing='proportional')
    
    # 将当前操作的子图切换到第二个子图(axs[1])
    ax = axs[1]
    
    # 在第二个子图(axs[1])上创建填充等高线图，并设定填充区域的级别
    cs = ax.contourf(z, levels=[2, 4, 6, 10, 20])
    
    # 在第二个子图(axs[1])上创建颜色条，使其在等高线填充区域上均匀分布
    fig.colorbar(cs, ax=ax, spacing='uniform')
    
    # 根据具体情况分割集合对象
    _maybe_split_collections(split_collections)
# 使用 pytest.mark.parametrize 装饰器，为 test_contour_linewidth 函数定义多组参数进行参数化测试
@pytest.mark.parametrize(
    "rc_lines_linewidth, rc_contour_linewidth, call_linewidths, expected", [
        (1.23, None, None, 1.23),  # 第一组参数：设置 rc_lines_linewidth 为 1.23，其他参数为 None，期望结果为 1.23
        (1.23, 4.24, None, 4.24),  # 第二组参数：设置 rc_lines_linewidth 为 1.23，rc_contour_linewidth 为 4.24，期望结果为 4.24
        (1.23, 4.24, 5.02, 5.02)   # 第三组参数：设置 rc_lines_linewidth 为 1.23，rc_contour_linewidth 为 4.24，call_linewidths 为 5.02，期望结果为 5.02
        ])
# 定义测试函数 test_contour_linewidth，接收参数 rc_lines_linewidth, rc_contour_linewidth, call_linewidths, expected
def test_contour_linewidth(
        rc_lines_linewidth, rc_contour_linewidth, call_linewidths, expected):

    # 使用 rc_context 上下文管理器设置 matplotlib 的参数，其中包括 lines.linewidth 和 contour.linewidth
    with rc_context(rc={"lines.linewidth": rc_lines_linewidth,
                        "contour.linewidth": rc_contour_linewidth}):
        # 创建图形和坐标轴
        fig, ax = plt.subplots()
        # 创建一个数组 X 用于轮廓线绘制
        X = np.arange(4*3).reshape(4, 3)
        # 绘制轮廓线，设置线宽为 call_linewidths
        cs = ax.contour(X, linewidths=call_linewidths)
        # 断言获取第一条轮廓线的线宽是否符合期望值
        assert cs.get_linewidths()[0] == expected
        # 使用 pytest.warns 断言捕获 MatplotlibDeprecationWarning 警告，并匹配 "tlinewidths"
        with pytest.warns(mpl.MatplotlibDeprecationWarning, match="tlinewidths"):
            assert cs.tlinewidths[0][0] == expected


# 使用 pytest.mark.backend 装饰器，设置测试函数 test_label_nonagg 的后端为 "pdf"
@pytest.mark.backend("pdf")
# 定义测试函数 test_label_nonagg，测试 plt.clabel 在没有 get_renderer() 的情况下不会崩溃
def test_label_nonagg():
    # 绘制等高线标签，输入参数为简单的二维数组
    plt.clabel(plt.contour([[1, 2], [3, 4]]))


# 使用 pytest.mark.parametrize 装饰器，为 test_contour_closed_line_loop 函数定义 split_collections 参数的两种取值进行参数化测试
@pytest.mark.parametrize("split_collections", [False, True])
# 使用 image_comparison 装饰器，设置基准图像为 'contour_closed_line_loop.png'，文件扩展名为 'png'，移除图中的文本信息
@image_comparison(baseline_images=['contour_closed_line_loop'],
                  extensions=['png'], remove_text=True)
# 定义测试函数 test_contour_closed_line_loop，测试绘制封闭轮廓线的情况
def test_contour_closed_line_loop(split_collections):
    # 创建一个二维数组作为绘制的数据源
    z = [[0, 0, 0], [0, 2, 0], [0, 0, 0], [2, 1, 2]]

    # 创建图形和坐标轴，并设置图形大小
    fig, ax = plt.subplots(figsize=(2, 2))
    # 绘制轮廓线，指定绘制等值线为 0.5，线宽为 [20]，透明度为 0.7
    ax.contour(z, [0.5], linewidths=[20], alpha=0.7)
    # 设置坐标轴的显示范围
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 3.1)

    # 调用 _maybe_split_collections 函数，根据 split_collections 参数决定是否进行图形分割处理
    _maybe_split_collections(split_collections)


# 定义测试函数 test_quadcontourset_reuse，测试 QuadContourSet 对象在复用 C++ 轮廓生成器的情况
def test_quadcontourset_reuse():
    # 创建一个简单的网格数据
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    z = x + y
    # 创建图形和坐标轴
    fig, ax = plt.subplots()
    # 使用 contourf 绘制填充等值线，并返回 QuadContourSet 对象 qcs1
    qcs1 = ax.contourf(x, y, z)
    # 使用 contour 绘制轮廓线，并返回 QuadContourSet 对象 qcs2
    qcs2 = ax.contour(x, y, z)
    # 断言 qcs2 使用的轮廓生成器不等于 qcs1 使用的轮廓生成器
    assert qcs2._contour_generator != qcs1._contour_generator
    # 将 qcs1 作为参数再次传入 contour 函数，返回 QuadContourSet 对象 qcs3
    qcs3 = ax.contour(qcs1, z)
    # 断言 qcs3 使用的轮廓生成器等于 qcs1 使用的轮廓生成器
    assert qcs3._contour_generator == qcs1._contour_generator


# 使用 pytest.mark.parametrize 装饰器，为 test_contour_manual 函数定义 split_collections 参数的两种取值进行参数化测试
@pytest.mark.parametrize("split_collections", [False, True])
# 使用 image_comparison 装饰器，设置基准图像为 'contour_manual.png'，文件扩展名为 'png'，移除图中的文本信息，设置允许的像素容差为 0.89
@image_comparison(baseline_images=['contour_manual'],
                  extensions=['png'], remove_text=True, tol=0.89)
# 定义测试函数 test_contour_manual，测试手动指定轮廓线/多边形绘制情况
def test_contour_manual(split_collections):
    # 导入 ContourSet 类
    from matplotlib.contour import ContourSet

    # 创建图形和坐标轴，并设置图形大小
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = 'viridis'

    # 定义只包含线段的轮廓数据，以及填充多边形的轮廓数据
    lines0 = [[[2, 0], [1, 2], [1, 3]]]  # 单条线段
    lines1 = [[[3, 0], [3, 2]], [[3, 3], [3, 4]]]  # 两条线段
    filled01 = [[[0, 0], [0, 4], [1, 3], [1, 2], [2, 0]]]
    filled12 = [[[2, 0], [3, 0], [3, 2], [1, 3], [1, 2]],  # 两个多边形
                [[1, 4], [3, 4], [3, 3]]]
    # 创建 ContourSet 对象，并进行轮廓线/多边形的绘制
    ContourSet(ax, [0, 1, 2], [filled01, filled12], filled=True, cmap=cmap)
    ContourSet(ax, [1, 2], [lines0, lines1], linewidths=3, colors=['r', 'k'])

    # Segments and kind codes (1 = MOVETO, 2 = LINETO, 79 = CLOSEPOLY).
    # 定义一个包含多个多边形轮廓的列表，每个轮廓由一组顶点坐标构成
    segs = [[[4, 0], [7, 0], [7, 3], [4, 3], [4, 0],
             [5, 1], [5, 2], [6, 2], [6, 1], [5, 1]]]
    
    # 定义一个多边形轮廓的种类列表，每个元素表示对应轮廓的种类
    kinds = [[1, 2, 2, 2, 79, 1, 2, 2, 2, 79]]  # 包含内孔的多边形
    
    # 创建 ContourSet 对象，并在绘图区域 ax 上绘制填充的等高线图
    ContourSet(ax, [2, 3], [segs], [kinds], filled=True, cmap=cmap)
    
    # 创建 ContourSet 对象，并在绘图区域 ax 上绘制线条的等高线图，线条颜色为黑色，线宽为3
    ContourSet(ax, [2], [segs], [kinds], colors='k', linewidths=3)
    
    # 调用函数 _maybe_split_collections，可能会对绘图中的集合进行拆分操作
    _maybe_split_collections(split_collections)
@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(baseline_images=['contour_line_start_on_corner_edge'],
                  extensions=['png'], remove_text=True)
# 定义测试函数，用于测试在角落边缘开始的等高线绘制情况
def test_contour_line_start_on_corner_edge(split_collections):
    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots(figsize=(6, 5))

    # 创建网格数据
    x, y = np.meshgrid([0, 1, 2, 3, 4], [0, 1, 2])
    # 定义一个二维高度数据
    z = 1.2 - (x - 2)**2 + (y - 1)**2
    # 创建一个与 z 大小相同的布尔类型的遮罩数组
    mask = np.zeros_like(z, dtype=bool)
    # 将遮罩数组的特定位置设为 True
    mask[1, 1] = mask[1, 3] = True
    # 创建一个带有遮罩的 numpy.ma 数组
    z = np.ma.array(z, mask=mask)

    # 绘制填充的等高线图
    filled = ax.contourf(x, y, z, corner_mask=True)
    # 添加颜色条
    cbar = fig.colorbar(filled)
    # 绘制等高线
    lines = ax.contour(x, y, z, corner_mask=True, colors='k')
    # 将等高线添加到颜色条上
    cbar.add_lines(lines)

    # 可能会拆分集合对象
    _maybe_split_collections(split_collections)


# 测试找到最近等高线的功能
def test_find_nearest_contour():
    # 创建一个坐标索引数组
    xy = np.indices((15, 15))
    # 创建一个指数下降的二维数组
    img = np.exp(-np.pi * (np.sum((xy - 5)**2, 0)/5.**2))
    # 绘制等高线
    cs = plt.contour(img, 10)

    # 寻找最近的等高线，并断言其结果与预期值几乎相等
    nearest_contour = cs.find_nearest_contour(1, 1, pixel=False)
    expected_nearest = (1, 0, 33, 1.965966, 1.965966, 1.866183)
    assert_array_almost_equal(nearest_contour, expected_nearest)

    # 重复上述过程，用不同的输入参数
    nearest_contour = cs.find_nearest_contour(8, 1, pixel=False)
    expected_nearest = (1, 0, 5, 7.550173, 1.587542, 0.547550)
    assert_array_almost_equal(nearest_contour, expected_nearest)

    nearest_contour = cs.find_nearest_contour(2, 5, pixel=False)
    expected_nearest = (3, 0, 21, 1.884384, 5.023335, 0.013911)
    assert_array_almost_equal(nearest_contour, expected_nearest)

    nearest_contour = cs.find_nearest_contour(2, 5, indices=(5, 7), pixel=False)
    expected_nearest = (5, 0, 16, 2.628202, 5.0, 0.394638)
    assert_array_almost_equal(nearest_contour, expected_nearest)


# 测试在没有填充情况下找到最近等高线的功能
def test_find_nearest_contour_no_filled():
    # 创建一个坐标索引数组
    xy = np.indices((15, 15))
    # 创建一个指数下降的二维数组，并填充等高线
    img = np.exp(-np.pi * (np.sum((xy - 5)**2, 0)/5.**2))
    cs = plt.contourf(img, 10)

    # 使用 pytest 断言检查期望的错误消息
    with pytest.raises(ValueError, match="Method does not support filled contours"):
        cs.find_nearest_contour(1, 1, pixel=False)

    with pytest.raises(ValueError, match="Method does not support filled contours"):
        cs.find_nearest_contour(1, 10, indices=(5, 7), pixel=False)

    with pytest.raises(ValueError, match="Method does not support filled contours"):
        cs.find_nearest_contour(2, 5, indices=(2, 7), pixel=True)


# 测试在默认样式上的等高线自动标记功能
@mpl.style.context("default")
def test_contour_autolabel_beyond_powerlimits():
    # 创建一个包含单个子图的图形对象
    ax = plt.figure().add_subplot()
    # 绘制等高线，并指定特定的等级
    cs = plt.contour(np.geomspace(1e-6, 1e-4, 100).reshape(10, 10),
                     levels=[.25e-5, 1e-5, 4e-5])
    # 添加等高线标签
    ax.clabel(cs)
    # 断言文本标签的内容符合预期
    assert {text.get_text() for text in ax.texts} == {"0.25", "1.00", "4.00"}


# 测试带有图例元素的填充等高线图
def test_contourf_legend_elements():
    # 导入所需的图形补丁模块
    from matplotlib.patches import Rectangle
    # 创建一维数组和二维数组
    x = np.arange(1, 10)
    y = x.reshape(-1, 1)
    h = x * y

    # 绘制填充等高线图，并指定参数
    cs = plt.contourf(h, levels=[10, 30, 50],
                      colors=['#FFFF00', '#FF00FF', '#00FFFF'],
                      extend='both')
    # 设置颜色映射的上限为红色
    cs.cmap.set_over('red')
    # 设置颜色映射的下限为蓝色
    cs.cmap.set_under('blue')
    # 标记颜色映射已修改
    cs.changed()
    # 获取图例元素和标签
    artists, labels = cs.legend_elements()
    # 断言标签内容是否如预期
    assert labels == ['$x \\leq -1e+250s$',
                      '$10.0 < x \\leq 30.0$',
                      '$30.0 < x \\leq 50.0$',
                      '$x > 1e+250s$']
    # 预期的颜色顺序
    expected_colors = ('blue', '#FFFF00', '#FF00FF', 'red')
    # 断言所有图例元素是矩形对象
    assert all(isinstance(a, Rectangle) for a in artists)
    # 断言每个图例元素的填充颜色是否与预期一致
    assert all(same_color(a.get_facecolor(), c)
               for a, c in zip(artists, expected_colors))
def test_contour_legend_elements():
    # 创建一个从 1 到 9 的一维数组
    x = np.arange(1, 10)
    # 将 x 数组转换为二维数组，形状为 9x1
    y = x.reshape(-1, 1)
    # 计算外积得到一个 9x9 的二维数组
    h = x * y

    # 定义颜色列表
    colors = ['blue', '#00FF00', 'red']
    # 绘制轮廓图，并指定轮廓线的颜色和级别
    cs = plt.contour(h, levels=[10, 30, 50],
                     colors=colors,
                     extend='both')
    # 获取轮廓图中的图例元素
    artists, labels = cs.legend_elements()
    # 断言图例标签的内容
    assert labels == ['$x = 10.0$', '$x = 30.0$', '$x = 50.0$']
    # 断言图例元素的类型为 Line2D
    assert all(isinstance(a, mpl.lines.Line2D) for a in artists)
    # 断言图例元素的颜色与预期颜色一致
    assert all(same_color(a.get_color(), c)
               for a, c in zip(artists, colors))


@pytest.mark.parametrize(
    "algorithm, klass",
    [('mpl2005', contourpy.Mpl2005ContourGenerator),
     ('mpl2014', contourpy.Mpl2014ContourGenerator),
     ('serial', contourpy.SerialContourGenerator),
     ('threaded', contourpy.ThreadedContourGenerator),
     ('invalid', None)])
def test_algorithm_name(algorithm, klass):
    # 创建一个2x2的数组
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    # 根据参数选择算法，并断言所使用的轮廓生成器的类别
    if klass is not None:
        cs = plt.contourf(z, algorithm=algorithm)
        assert isinstance(cs._contour_generator, klass)
    else:
        # 对于无效算法参数，断言会触发 ValueError 异常
        with pytest.raises(ValueError):
            plt.contourf(z, algorithm=algorithm)


@pytest.mark.parametrize(
    "algorithm", ['mpl2005', 'mpl2014', 'serial', 'threaded'])
def test_algorithm_supports_corner_mask(algorithm):
    # 创建一个2x2的数组
    z = np.array([[1.0, 2.0], [3.0, 4.0]])

    # 所有算法都支持 corner_mask=False 的情况
    plt.contourf(z, algorithm=algorithm, corner_mask=False)

    # 只有部分算法支持 corner_mask=True 的情况
    if algorithm != 'mpl2005':
        plt.contourf(z, algorithm=algorithm, corner_mask=True)
    else:
        # 对于不支持的算法，断言会触发 ValueError 异常
        with pytest.raises(ValueError):
            plt.contourf(z, algorithm=algorithm, corner_mask=True)


@pytest.mark.parametrize("split_collections", [False, True])
@image_comparison(baseline_images=['contour_all_algorithms'],
                  extensions=['png'], remove_text=True, tol=0.06)
def test_all_algorithms(split_collections):
    # 定义算法列表
    algorithms = ['mpl2005', 'mpl2014', 'serial', 'threaded']

    # 创建一个随机数生成器
    rng = np.random.default_rng(2981)
    # 创建网格
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 6))
    # 计算 z 值
    z = np.sin(15*x)*np.cos(10*y) + rng.normal(scale=0.5, size=(6, 10))
    # 创建一个布尔掩码数组
    mask = np.zeros_like(z, dtype=bool)
    mask[3, 7] = True
    z = np.ma.array(z, mask=mask)

    # 创建包含四个子图的图形
    _, axs = plt.subplots(2, 2)
    # 对每个子图应用不同的算法绘制轮廓填充图和轮廓线图
    for ax, algorithm in zip(axs.ravel(), algorithms):
        ax.contourf(x, y, z, algorithm=algorithm)
        ax.contour(x, y, z, algorithm=algorithm, colors='k')
        ax.set_title(algorithm)

    # 根据参数决定是否拆分集合
    _maybe_split_collections(split_collections)


def test_subfigure_clabel():
    # 对 gh#23173 进行烟雾测试
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X**2) - Y**2)
    Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2

    # 创建包含两个子图的图形对象
    fig = plt.figure()
    figs = fig.subfigures(nrows=1, ncols=2)
    # 对于列表 figs 中的每个图形对象 f，依次进行以下操作：
    for f in figs:
        # 在图形对象 f 上创建子图 ax
        ax = f.subplots()
        # 使用 contour 方法在子图 ax 上绘制等高线图，使用 X, Y, Z 数据
        CS = ax.contour(X, Y, Z)
        # 在等高线图上添加标签，inline=True 表示标签嵌入等高线内部，fontsize=10 设置标签字体大小
        ax.clabel(CS, inline=True, fontsize=10)
        # 设置子图 ax 的标题为 "Simplest default with labels"
        ax.set_title("Simplest default with labels")
@pytest.mark.parametrize(
    "style", ['solid', 'dashed', 'dashdot', 'dotted'])
def test_linestyles(style):
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)  # 创建包含 x 值的数组
    y = np.arange(-2.0, 2.0, delta)  # 创建包含 y 值的数组
    X, Y = np.meshgrid(x, y)  # 生成网格数据 X 和 Y
    Z1 = np.exp(-X**2 - Y**2)  # 计算高斯分布的第一个函数
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)  # 计算高斯分布的第二个函数
    Z = (Z1 - Z2) * 2  # 计算函数之间的差异，乘以 2

    # Positive contour defaults to solid
    fig1, ax1 = plt.subplots()  # 创建图形和坐标轴
    CS1 = ax1.contour(X, Y, Z, 6, colors='k')  # 绘制等高线，默认实线
    ax1.clabel(CS1, fontsize=9, inline=True)  # 添加等高线标签
    ax1.set_title('Single color - positive contours solid (default)')  # 设置标题
    assert CS1.linestyles is None  # 断言默认情况下等高线样式为 None

    # Change linestyles using linestyles kwarg
    fig2, ax2 = plt.subplots()  # 创建图形和坐标轴
    CS2 = ax2.contour(X, Y, Z, 6, colors='k', linestyles=style)  # 绘制等高线，指定样式
    ax2.clabel(CS2, fontsize=9, inline=True)  # 添加等高线标签
    ax2.set_title(f'Single color - positive contours {style}')  # 设置标题，包含样式信息
    assert CS2.linestyles == style  # 断言等高线样式是否为指定的样式

    # Ensure linestyles do not change when negative_linestyles is defined
    fig3, ax3 = plt.subplots()  # 创建图形和坐标轴
    CS3 = ax3.contour(X, Y, Z, 6, colors='k', linestyles=style,
                      negative_linestyles='dashdot')  # 绘制等高线，指定正负样式
    ax3.clabel(CS3, fontsize=9, inline=True)  # 添加等高线标签
    ax3.set_title(f'Single color - positive contours {style}')  # 设置标题
    assert CS3.linestyles == style  # 断言等高线样式是否为指定的样式


@pytest.mark.parametrize(
    "style", ['solid', 'dashed', 'dashdot', 'dotted'])
def test_negative_linestyles(style):
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)  # 创建包含 x 值的数组
    y = np.arange(-2.0, 2.0, delta)  # 创建包含 y 值的数组
    X, Y = np.meshgrid(x, y)  # 生成网格数据 X 和 Y
    Z1 = np.exp(-X**2 - Y**2)  # 计算高斯分布的第一个函数
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)  # 计算高斯分布的第二个函数
    Z = (Z1 - Z2) * 2  # 计算函数之间的差异，乘以 2

    # Negative contour defaults to dashed
    fig1, ax1 = plt.subplots()  # 创建图形和坐标轴
    CS1 = ax1.contour(X, Y, Z, 6, colors='k')  # 绘制等高线，默认虚线
    ax1.clabel(CS1, fontsize=9, inline=True)  # 添加等高线标签
    ax1.set_title('Single color - negative contours dashed (default)')  # 设置标题
    assert CS1.negative_linestyles == 'dashed'  # 断言默认情况下负等高线样式为虚线

    # Change negative_linestyles using rcParams
    plt.rcParams['contour.negative_linestyle'] = style  # 设置全局负等高线样式
    fig2, ax2 = plt.subplots()  # 创建图形和坐标轴
    CS2 = ax2.contour(X, Y, Z, 6, colors='k')  # 绘制等高线
    ax2.clabel(CS2, fontsize=9, inline=True)  # 添加等高线标签
    ax2.set_title(f'Single color - negative contours {style}'
                   '(using rcParams)')  # 设置标题，包含样式信息
    assert CS2.negative_linestyles == style  # 断言负等高线样式是否为指定的样式

    # Change negative_linestyles using negative_linestyles kwarg
    fig3, ax3 = plt.subplots()  # 创建图形和坐标轴
    CS3 = ax3.contour(X, Y, Z, 6, colors='k', negative_linestyles=style)  # 绘制等高线，指定负样式
    ax3.clabel(CS3, fontsize=9, inline=True)  # 添加等高线标签
    ax3.set_title(f'Single color - negative contours {style}')  # 设置标题
    assert CS3.negative_linestyles == style  # 断言负等高线样式是否为指定的样式

    # Ensure negative_linestyles do not change when linestyles is defined
    fig4, ax4 = plt.subplots()  # 创建图形和坐标轴
    CS4 = ax4.contour(X, Y, Z, 6, colors='k', linestyles='dashdot',
                      negative_linestyles=style)  # 绘制等高线，同时指定正负样式
    ax4.clabel(CS4, fontsize=9, inline=True)  # 添加等高线标签
    ax4.set_title(f'Single color - negative contours {style}')  # 设置标题
    assert CS4.negative_linestyles == style  # 断言负等高线样式是否为指定的样式
    # 创建一个新的图形并添加一个子图到该图形，返回子图对象 ax
    ax = plt.figure().add_subplot()
    
    # 获取子图 ax 的所有子元素（例如轮廓线、标签等）
    orig_children = ax.get_children()
    
    # 在子图 ax 上绘制由 np.arange(16).reshape((4, 4)) 构成的等高线
    cs = ax.contour(np.arange(16).reshape((4, 4)))
    
    # 在等高线上添加标签
    cs.clabel()
    
    # 断言：检查子图 ax 的子元素是否已经发生了变化
    assert ax.get_children() != orig_children
    
    # 从子图 ax 上移除等高线
    cs.remove()
    
    # 断言：检查子图 ax 的子元素是否恢复到最初的状态
    assert ax.get_children() == orig_children
def test_contour_no_args():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 定义一个二维数据数组
    data = [[0, 1], [1, 0]]
    # 使用 pytest 来检测是否会引发 TypeError 异常，匹配给定的错误消息
    with pytest.raises(TypeError, match=r"contour\(\) takes from 1 to 4"):
        # 调用坐标轴对象的 contour 方法，并传入参数 Z=data
        ax.contour(Z=data)


def test_contour_clip_path():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 定义一个二维数据数组
    data = [[0, 1], [1, 0]]
    # 创建一个圆形路径对象，将其应用在坐标轴的坐标变换上
    circle = mpatches.Circle([0.5, 0.5], 0.5, transform=ax.transAxes)
    # 调用坐标轴对象的 contour 方法，并传入数据和 clip_path 参数
    cs = ax.contour(data, clip_path=circle)
    # 断言 contour 对象的剪裁路径不为空
    assert cs.get_clip_path() is not None


def test_bool_autolevel():
    # 生成随机的 x 和 y 数据
    x, y = np.random.rand(2, 9)
    # 创建一个布尔类型的二维数组 z
    z = (np.arange(9) % 2).reshape((3, 3)).astype(bool)
    # 测试 plt.contour 函数对 z 列表化数据的等高线水平线是否为 [0.5]
    assert plt.contour(z.tolist()).levels.tolist() == [.5]
    # 测试 plt.contour 函数对 z 的等高线水平线是否为 [0.5]
    assert plt.contour(z).levels.tolist() == [.5]
    # 测试 plt.contour 函数对带有屏蔽区域 m 的 z 数据的等高线水平线是否为 [0.5]
    assert plt.contour(np.ma.array(z, mask=m)).levels.tolist() == [.5]
    # 测试 plt.contourf 函数对 z 列表化数据的等高线水平线是否为 [0, 0.5, 1]
    assert plt.contourf(z.tolist()).levels.tolist() == [0, .5, 1]
    # 测试 plt.contourf 函数对 z 的等高线水平线是否为 [0, 0.5, 1]
    assert plt.contourf(z).levels.tolist() == [0, .5, 1]
    # 测试 plt.contourf 函数对带有屏蔽区域 m 的 z 数据的等高线水平线是否为 [0, 0.5, 1]
    assert plt.contourf(np.ma.array(z, mask=m)).levels.tolist() == [0, .5, 1]
    # 将 z 摊平，测试 plt.tricontour 函数对其列化数据的等高线水平线是否为 [0.5]
    z = z.ravel()
    assert plt.tricontour(x, y, z.tolist()).levels.tolist() == [.5]
    # 测试 plt.tricontour 函数对 z 的等高线水平线是否为 [0.5]
    assert plt.tricontour(x, y, z).levels.tolist() == [.5]
    # 测试 plt.tricontourf 函数对其列化数据的等高线水平线是否为 [0, 0.5, 1]
    assert plt.tricontourf(x, y, z.tolist()).levels.tolist() == [0, .5, 1]
    # 测试 plt.tricontourf 函数对 z 的等高线水平线是否为 [0, 0.5, 1]
    assert plt.tricontourf(x, y, z).levels.tolist() == [0, .5, 1]


def test_all_nan():
    # 创建一个包含 NaN 的二维数组 x
    x = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    # 使用 plt.contour 函数绘制 x 的等高线，并断言其水平线
    assert_array_almost_equal(plt.contour(x).levels,
                              [-1e-13, -7.5e-14, -5e-14, -2.4e-14, 0.0,
                                2.4e-14, 5e-14, 7.5e-14, 1e-13])


def test_allsegs_allkinds():
    # 创建网格数据 x, y 和相应的 z 值
    x, y = np.meshgrid(np.arange(0, 10, 2), np.arange(0, 10, 2))
    z = np.sin(x) * np.cos(y)
    # 使用 plt.contour 函数绘制 x, y, z 的等高线，并获取返回的 contourset 对象
    cs = plt.contour(x, y, z, levels=[0, 0.5])

    # 断言 contourset 对象的所有线段和类型列表的长度符合预期
    for result in [cs.allsegs, cs.allkinds]:
        assert len(result) == 2
        assert len(result[0]) == 5
        assert len(result[1]) == 4


def test_deprecated_apis():
    # 使用 plt.contour 函数绘制一个简单的等高线图并获取 contourset 对象
    cs = plt.contour(np.arange(16).reshape((4, 4)))
    # 使用 pytest 检测特定的警告消息类型，匹配 "collections"
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match="collections"):
        colls = cs.collections
    # 使用 pytest 检测特定的警告消息类型，匹配 "tcolors"
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match="tcolors"):
        assert_array_equal(cs.tcolors, [c.get_edgecolor() for c in colls])
    # 使用 pytest 检测特定的警告消息类型，匹配 "tlinewidths"
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match="tlinewidths"):
        assert cs.tlinewidths == [c.get_linewidth() for c in colls]
    # 使用 pytest 检测特定的警告消息类型，匹配 "antialiased"
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match="antialiased"):
        assert cs.antialiased
    # 使用 pytest 检测特定的警告消息类型，匹配 "antialiased"
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match="antialiased"):
        cs.antialiased = False
    # 使用 pytest 检测特定的警告消息类型，匹配 "antialiased"
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match="antialiased"):
        assert not cs.antialiased
```