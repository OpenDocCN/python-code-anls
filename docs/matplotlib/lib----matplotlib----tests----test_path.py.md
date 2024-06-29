# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_path.py`

```
import platform
import re

import numpy as np  # 导入 NumPy 库
from numpy.testing import assert_array_equal  # 导入 NumPy 测试模块中的数组相等断言
import pytest  # 导入 pytest 测试框架

from matplotlib import patches  # 导入 matplotlib 库中的 patches 模块
from matplotlib.path import Path  # 导入 matplotlib 库中的路径 Path 类
from matplotlib.patches import Polygon  # 导入 matplotlib 库中的多边形图形模块
from matplotlib.testing.decorators import image_comparison  # 导入 matplotlib 测试模块中的图像比较装饰器
import matplotlib.pyplot as plt  # 导入 matplotlib 绘图库中的 pyplot 模块
from matplotlib import transforms  # 导入 matplotlib 库中的 transforms 模块
from matplotlib.backend_bases import MouseEvent  # 导入 matplotlib 库中的鼠标事件模块


def test_empty_closed_path():
    path = Path(np.zeros((0, 2)), closed=True)  # 创建一个空的闭合路径对象
    assert path.vertices.shape == (0, 2)  # 断言路径顶点数组的形状为 (0, 2)
    assert path.codes is None  # 断言路径的代码为 None
    assert_array_equal(path.get_extents().extents,
                       transforms.Bbox.null().extents)  # 断言路径的边界框和空边界框的边界一致


def test_readonly_path():
    path = Path.unit_circle()  # 创建一个单位圆路径对象

    def modify_vertices():
        path.vertices = path.vertices * 2.0  # 尝试修改路径对象的顶点

    with pytest.raises(AttributeError):  # 使用 pytest 检查是否引发 AttributeError 异常
        modify_vertices()


def test_path_exceptions():
    bad_verts1 = np.arange(12).reshape(4, 3)  # 创建一个不合法的顶点数组
    with pytest.raises(ValueError,
                       match=re.escape(f'has shape {bad_verts1.shape}')):  # 使用 pytest 检查是否引发 ValueError 异常，并匹配异常信息
        Path(bad_verts1)

    bad_verts2 = np.arange(12).reshape(2, 3, 2)  # 创建另一个不合法的顶点数组
    with pytest.raises(ValueError,
                       match=re.escape(f'has shape {bad_verts2.shape}')):  # 使用 pytest 检查是否引发 ValueError 异常，并匹配异常信息
        Path(bad_verts2)

    good_verts = np.arange(12).reshape(6, 2)  # 创建一个合法的顶点数组
    bad_codes = np.arange(2)  # 创建一个不合法的代码数组
    msg = re.escape(f"Your vertices have shape {good_verts.shape} "
                    f"but your codes have shape {bad_codes.shape}")
    with pytest.raises(ValueError, match=msg):  # 使用 pytest 检查是否引发 ValueError 异常，并匹配异常信息
        Path(good_verts, bad_codes)


def test_point_in_path():
    # 测试 #1787
    path = Path._create_closed([(0, 0), (0, 1), (1, 1), (1, 0)])  # 创建一个闭合路径对象
    points = [(0.5, 0.5), (1.5, 0.5)]  # 创建点集
    ret = path.contains_points(points)  # 检查点集中的点是否在路径内
    assert ret.dtype == 'bool'  # 断言返回的结果类型为布尔型
    np.testing.assert_equal(ret, [True, False])  # 使用 NumPy 测试模块断言结果相等


@pytest.mark.parametrize(
    "other_path, inside, inverted_inside",
    [(Path([(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25), (0.25, 0.25)],
          closed=True), True, False),  # 参数化测试：检查路径是否包含另一条路径
     (Path([(-0.25, -0.25), (-0.25, 1.75), (1.75, 1.75), (1.75, -0.25), (-0.25, -0.25)],
          closed=True), False, True),  # 参数化测试：检查路径是否包含另一条路径（反向）
     (Path([(-0.25, -0.25), (-0.25, 1.75), (0.5, 0.5),
           (1.75, 1.75), (1.75, -0.25), (-0.25, -0.25)],
          closed=True), False, False),  # 参数化测试：检查路径是否包含另一条路径（无交集）
     (Path([(0.25, 0.25), (0.25, 1.25), (1.25, 1.25), (1.25, 0.25), (0.25, 0.25)],
          closed=True), False, False),  # 参数化测试：检查路径是否包含另一条路径（重叠边界）
     (Path([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)], closed=True), False, False),  # 参数化测试：检查路径是否包含另一条路径（相等路径）
     (Path([(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)], closed=True), False, False)])  # 参数化测试：检查路径是否包含另一条路径（完全不相交）
def test_contains_path(other_path, inside, inverted_inside):
    path = Path([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)], closed=True)  # 创建一个闭合路径对象
    assert path.contains_path(other_path) is inside  # 断言路径是否包含另一条路径
    assert other_path.contains_path(path) is inverted_inside  # 断言另一条路径是否包含路径


def test_contains_points_negative_radius():
    path = Path.unit_circle()  # 创建一个单位圆路径对象

    points = [(0.0, 0.0), (1.25, 0.0), (0.9, 0.9)]  # 创建点集
    result = path.contains_points(points, radius=-0.5)  # 检查点集中的点是否在路径内（使用负半径）
    # 使用 NumPy 的测试工具 np.testing.assert_equal 进行断言，比较 result 是否等于 [True, False, False]
    np.testing.assert_equal(result, [True, False, False])
_test_paths = [
    # 定义路径对象，包含四个点，其中第一个点为起始点
    Path([[0, 0], [1, 0], [1, 1], [0, 1]],
           [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
    # 定义二次曲线路径对象，包含三个点，第一个点为起始点
    Path([[0, 0], [0, 1], [1, 0]], [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
    # 定义线性路径对象，包含两个点，第一个点为起始点
    Path([[0, 1], [1, 1]], [Path.MOVETO, Path.LINETO]),
    # 定义单点路径对象，包含一个点，为起始点
    Path([[1, 2]], [Path.MOVETO]),
]

_test_path_extents = [(0., 0., 0.75, 1.), (0., 0., 1., 0.5), (0., 1., 1., 1.),
                      (1., 2., 1., 2.)]

@pytest.mark.parametrize('path, extents', zip(_test_paths, _test_path_extents))
def test_exact_extents(path, extents):
    # 检查路径对象的边界框是否与预期的边界框 extents 相同
    # 注意，path.get_extents() 返回一个 Bbox 对象，我们需要使用 `.extents` 获取其范围
    assert np.all(path.get_extents().extents == extents)


@pytest.mark.parametrize('ignored_code', [Path.CLOSEPOLY, Path.STOP])
def test_extents_with_ignored_codes(ignored_code):
    # 检查在只有直线的路径中，忽略 STOP 和 CLOSEPOLY 点时计算的边界框是否正确
    path = Path([[0, 0],
                 [1, 1],
                 [2, 2]], [Path.MOVETO, Path.MOVETO, ignored_code])
    assert np.all(path.get_extents().extents == (0., 0., 1., 1.))


def test_point_in_path_nan():
    # 创建一个包含 NaN 的测试点，检查其是否在指定的路径中
    box = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    p = Path(box)
    test = np.array([[np.nan, 0.5]])
    contains = p.contains_points(test)
    assert len(contains) == 1
    assert not contains[0]


def test_nonlinear_containment():
    # 创建一个包含非线性变换的图形对象，并检查点是否包含在路径中
    fig, ax = plt.subplots()
    ax.set(xscale="log", ylim=(0, 1))
    polygon = ax.axvspan(1, 10)
    assert polygon.get_path().contains_point(
        ax.transData.transform((5, .5)), polygon.get_transform())
    assert not polygon.get_path().contains_point(
        ax.transData.transform((.5, .5)), polygon.get_transform())
    assert not polygon.get_path().contains_point(
        ax.transData.transform((50, .5)), polygon.get_transform())


@image_comparison(['arrow_contains_point.png'], remove_text=True, style='mpl20',
                  tol=0.027 if platform.machine() == 'arm64' else 0)
def test_arrow_contains_point():
    # 修复 bug (#8384)
    fig, ax = plt.subplots()
    ax.set_xlim((0, 2))
    ax.set_ylim((0, 2))

    # 创建一个 Curve 样式的箭头对象
    arrow = patches.FancyArrowPatch((0.5, 0.25), (1.5, 0.75),
                                    arrowstyle='->',
                                    mutation_scale=40)
    # 在图形 ax 上添加箭头 arrow
    ax.add_patch(arrow)

    # 创建一个带有 Bracket 风格的箭头 arrow1
    arrow1 = patches.FancyArrowPatch((0.5, 1), (1.5, 1.25),
                                     arrowstyle=']-[',
                                     mutation_scale=40)
    ax.add_patch(arrow1)

    # 创建一个带有 Fancy 风格的箭头 arrow2
    arrow2 = patches.FancyArrowPatch((0.5, 1.5), (1.5, 1.75),
                                     arrowstyle='fancy',
                                     fill=False,
                                     mutation_scale=40)
    ax.add_patch(arrow2)

    # 将所有箭头对象存入列表 patches_list
    patches_list = [arrow, arrow1, arrow2]

    # 生成一些坐标点
    X, Y = np.meshgrid(np.arange(0, 2, 0.1),
                       np.arange(0, 2, 0.1))
    for k, (x, y) in enumerate(zip(X.ravel(), Y.ravel())):
        # 将数据坐标转换为画布坐标
        xdisp, ydisp = ax.transData.transform([x, y])
        # 创建一个鼠标事件对象
        event = MouseEvent('button_press_event', fig.canvas, xdisp, ydisp)
        for m, patch in enumerate(patches_list):
            # 检查当前坐标点是否在箭头 patch 内部
            inside, res = patch.contains(event)
            if inside:
                # 如果点在箭头内部，将该点标记为红色
                ax.scatter(x, y, s=5, c="r")
@image_comparison(['path_clipping.svg'], remove_text=True)
def test_path_clipping():
    # 创建一个尺寸为 6.0x6.2 的新图形对象
    fig = plt.figure(figsize=(6.0, 6.2))

    for i, xy in enumerate([
            [(200, 200), (200, 350), (400, 350), (400, 200)],
            [(200, 200), (200, 350), (400, 350), (400, 100)],
            [(200, 100), (200, 350), (400, 350), (400, 100)],
            [(200, 100), (200, 415), (400, 350), (400, 100)],
            [(200, 100), (200, 415), (400, 415), (400, 100)],
            [(200, 415), (400, 415), (400, 100), (200, 100)],
            [(400, 415), (400, 100), (200, 100), (200, 415)]]):
        # 在图形对象中添加一个子图，4 行 2 列，当前为第 i+1 个子图
        ax = fig.add_subplot(4, 2, i+1)
        # 定义裁剪框的边界框，左下角坐标 (0, 140)，宽度 640，高度 260
        bbox = [0, 140, 640, 260]
        # 设置当前子图的 x 轴范围
        ax.set_xlim(bbox[0], bbox[0] + bbox[2])
        # 设置当前子图的 y 轴范围
        ax.set_ylim(bbox[1], bbox[1] + bbox[3])
        # 在当前子图中添加一个多边形补丁，指定顶点坐标、面颜色为空、边框颜色为红色，并闭合
        ax.add_patch(Polygon(
            xy, facecolor='none', edgecolor='red', closed=True))


@image_comparison(['semi_log_with_zero.png'], style='mpl20')
def test_log_transform_with_zero():
    # 生成一个从 -10 到 9 的数组作为 x 值
    x = np.arange(-10, 10)
    # 根据 x 值计算 y 值，进行对数变换
    y = (1.0 - 1.0/(x**2+1))**20

    # 创建一个包含图形和轴的新图形对象
    fig, ax = plt.subplots()
    # 在对数坐标轴上绘制 x 和 y 数据，线型为 "-o"，线宽 15，标记边缘颜色为黑色
    ax.semilogy(x, y, "-o", lw=15, markeredgecolor='k')
    # 设置 y 轴的限制范围
    ax.set_ylim(1e-7, 1)
    # 显示网格
    ax.grid(True)


def test_make_compound_path_empty():
    # 我们应该能够创建一个没有参数的复合路径。
    # 这使得编写基于路径的通用代码变得更加容易。
    # 创建一个空的复合路径对象
    empty = Path.make_compound_path()
    # 断言空路径对象的顶点形状为 (0, 2)
    assert empty.vertices.shape == (0, 2)
    # 创建另一个复合路径对象 r2，由两个空路径对象 empty 组成
    r2 = Path.make_compound_path(empty, empty)
    # 断言 r2 的顶点形状为 (0, 2)
    assert r2.vertices.shape == (0, 2)
    # 断言 r2 的代码形状为 (0,)
    assert r2.codes.shape == (0,)
    # 创建另一个复合路径对象 r3，包含一个非空路径对象和一个空路径对象 empty
    r3 = Path.make_compound_path(Path([(0, 0)]), empty)
    # 断言 r3 的顶点形状为 (1, 2)
    assert r3.vertices.shape == (1, 2)
    # 断言 r3 的代码形状为 (1,)
    assert r3.codes.shape == (1,)


def test_make_compound_path_stops():
    # 创建一个简单路径对象的起始点坐标
    zero = [0, 0]
    # 创建包含三个简单路径对象的列表，每个对象都有一个起始点和一个结束点
    paths = 3*[Path([zero, zero], [Path.MOVETO, Path.STOP])]
    # 创建一个复合路径对象 compound_path，包含上述三个简单路径对象
    compound_path = Path.make_compound_path(*paths)
    # 断言复合路径对象中的 STOP 代码的数量为 0
    assert np.sum(compound_path.codes == Path.STOP) == 0


@image_comparison(['xkcd.png'], remove_text=True)
def test_xkcd():
    # 设置随机数生成的种子，以确保可重复生成相同的随机序列
    np.random.seed(0)

    # 生成一个包含 100 个点的 x 坐标数组，范围从 0 到 2π
    x = np.linspace(0, 2 * np.pi, 100)
    # 计算相应的 y 值，为 x 的正弦值
    y = np.sin(x)

    # 使用 xkcd 风格绘制图形
    with plt.xkcd():
        # 创建包含图形和轴的新图形对象
        fig, ax = plt.subplots()
        # 绘制 x 和 y 数据的线图
        ax.plot(x, y)


@image_comparison(['xkcd_marker.png'], remove_text=True)
def test_xkcd_marker():
    # 设置随机数生成的种子，以确保可重复生成相同的随机序列
    np.random.seed(0)

    # 生成一个包含 8 个点的 x 坐标数组，范围从 0 到 5
    x = np.linspace(0, 5, 8)
    # 定义三条线的 y 值
    y1 = x
    y2 = 5 - x
    y3 = 2.5 * np.ones(8)

    # 使用 xkcd 风格绘制图形
    with plt.xkcd():
        # 创建包含图形和轴的新图形对象
        fig, ax = plt.subplots()
        # 在图形中绘制三条带有不同标记的线
        ax.plot(x, y1, '+', ms=10)
        ax.plot(x, y2, 'o', ms=10)
        ax.plot(x, y3, '^', ms=10)


@image_comparison(['marker_paths.pdf'], remove_text=True)
def test_marker_paths_pdf():
    N = 7

    # 绘制一个带有误差棒的简单折线图
    plt.errorbar(np.arange(N),
                 np.ones(N) + 4,
                 np.ones(N))
    # 设置 x 轴和 y 轴的限制范围
    plt.xlim(-1, N)
    plt.ylim(-1, 7)
# 使用 image_comparison 装饰器比较图像结果，参数包括要比较的图像名称列表 'nan_path'，
# 图形样式 'default'，移除文本信息，文件扩展名为 'pdf', 'svg', 'eps', 'png'，
# 公差参数根据平台架构选择不同值（如果是 'arm64' 架构则为 0.009，否则为 0）
@image_comparison(['nan_path'], style='default', remove_text=True,
                  extensions=['pdf', 'svg', 'eps', 'png'],
                  tol=0.009 if platform.machine() == 'arm64' else 0)
def test_nan_isolated_points():
    # 定义包含 NaN 值的数据列表
    y0 = [0, np.nan, 2, np.nan, 4, 5, 6]
    y1 = [np.nan, 7, np.nan, 9, 10, np.nan, 12]

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    # 在坐标轴上绘制包含 NaN 值的数据线图
    ax.plot(y0, '-o')
    ax.plot(y1, '-o')


def test_path_no_doubled_point_in_to_polygon():
    # 定义手绘多边形的顶点坐标数组
    hand = np.array(
        [[1.64516129, 1.16145833],
         [1.64516129, 1.59375],
         [1.35080645, 1.921875],
         [1.375, 2.18229167],
         [1.68548387, 1.9375],
         [1.60887097, 2.55208333],
         [1.68548387, 2.69791667],
         [1.76209677, 2.56770833],
         [1.83064516, 1.97395833],
         [1.89516129, 2.75],
         [1.9516129, 2.84895833],
         [2.01209677, 2.76041667],
         [1.99193548, 1.99479167],
         [2.11290323, 2.63020833],
         [2.2016129, 2.734375],
         [2.25403226, 2.60416667],
         [2.14919355, 1.953125],
         [2.30645161, 2.36979167],
         [2.39112903, 2.36979167],
         [2.41532258, 2.1875],
         [2.1733871, 1.703125],
         [2.07782258, 1.16666667]])

    # 定义剪切矩形的坐标范围
    (r0, c0, r1, c1) = (1.0, 1.5, 2.1, 2.5)

    # 根据手绘多边形顶点创建 Path 对象
    poly = Path(np.vstack((hand[:, 1], hand[:, 0])).T, closed=True)
    # 创建剪切矩形的边界框对象
    clip_rect = transforms.Bbox([[r0, c0], [r1, c1]])
    # 对多边形进行剪切，得到剪切后的多边形顶点数组
    poly_clipped = poly.clip_to_bbox(clip_rect).to_polygons()[0]

    # 断言最后两个顶点不相等
    assert np.all(poly_clipped[-2] != poly_clipped[-1])
    # 断言第一个和最后一个顶点相等（多边形封闭）
    assert np.all(poly_clipped[-1] == poly_clipped[0])


def test_path_to_polygons():
    # 定义简单的路径数据
    data = [[10, 10], [20, 20]]
    p = Path(data)

    # 测试路径转换为多边形时的各种情况
    assert_array_equal(p.to_polygons(width=40, height=40), [])
    assert_array_equal(p.to_polygons(width=40, height=40, closed_only=False),
                       [data])
    assert_array_equal(p.to_polygons(), [])
    assert_array_equal(p.to_polygons(closed_only=False), [data])

    # 添加第三个点后的测试数据
    data = [[10, 10], [20, 20], [30, 30]]
    closed_data = [[10, 10], [20, 20], [30, 30], [10, 10]]
    p = Path(data)

    # 测试路径转换为多边形时的各种情况
    assert_array_equal(p.to_polygons(width=40, height=40), [closed_data])
    assert_array_equal(p.to_polygons(width=40, height=40, closed_only=False),
                       [data])
    assert_array_equal(p.to_polygons(), [closed_data])
    assert_array_equal(p.to_polygons(closed_only=False), [data])


def test_path_deepcopy():
    # 测试路径对象的深拷贝操作
    verts = [[0, 0], [1, 1]]
    codes = [Path.MOVETO, Path.LINETO]
    path1 = Path(verts)
    path2 = Path(verts, codes)
    path1_copy = path1.deepcopy()
    path2_copy = path2.deepcopy()
    assert path1 is not path1_copy
    assert path1.vertices is not path1_copy.vertices
    assert path2 is not path2_copy
    assert path2.vertices is not path2_copy.vertices
    assert path2.codes is not path2_copy.codes


def test_path_shallowcopy():
    # 测试路径对象的浅拷贝操作
    verts = [[0, 0], [1, 1]]
    codes = [Path.MOVETO, Path.LINETO]
    path1 = Path(verts)
    path2 = Path(verts, codes)
    # 检查是否可以创建浅拷贝而不引发错误
    path1_copy = path1.shallow_copy()
    path2_copy = path2.shallow_copy()
    # 复制 path1 对象，生成一个新的对象 path1_copy
    path1_copy = path1.copy()
    # 复制 path2 对象，生成一个新的对象 path2_copy
    path2_copy = path2.copy()
    # 断言确保 path1 与 path1_copy 不是同一个对象
    assert path1 is not path1_copy
    # 断言确保 path1 和 path1_copy 的 vertices 属性指向同一对象
    assert path1.vertices is path1_copy.vertices
    # 断言确保 path2 与 path2_copy 不是同一个对象
    assert path2 is not path2_copy
    # 断言确保 path2 和 path2_copy 的 vertices 属性指向同一对象
    assert path2.vertices is path2_copy.vertices
    # 断言确保 path2 和 path2_copy 的 codes 属性指向同一对象
    assert path2.codes is path2_copy.codes
@pytest.mark.parametrize('phi', np.concatenate([
    np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135]) + delta
    for delta in [-1, 0, 1]]))
def test_path_intersect_path(phi):
    # 测试不同交点角度的范围

    eps_array = [1e-5, 1e-8, 1e-10, 1e-12]

    transform = transforms.Affine2D().rotate(np.deg2rad(phi))

    # a and b intersect at angle phi
    # 创建路径 a 和 b，使其在角度 phi 处相交
    a = Path([(-2, 0), (2, 0)])
    b = transform.transform_path(a)
    assert a.intersects_path(b) and b.intersects_path(a)

    # a and b touch at angle phi at (0, 0)
    # 创建路径 a 和 b，使其在角度 phi 处于 (0, 0) 处接触
    a = Path([(0, 0), (2, 0)])
    b = transform.transform_path(a)
    assert a.intersects_path(b) and b.intersects_path(a)

    # a and b are orthogonal and intersect at (0, 3)
    # 创建路径 a 和 b，使其正交，并在 (0, 3) 处相交
    a = transform.transform_path(Path([(0, 1), (0, 3)]))
    b = transform.transform_path(Path([(1, 3), (0, 3)]))
    assert a.intersects_path(b) and b.intersects_path(a)

    # a and b are collinear and intersect at (0, 3)
    # 创建路径 a 和 b，使其共线，并在 (0, 3) 处相交
    a = transform.transform_path(Path([(0, 1), (0, 3)]))
    b = transform.transform_path(Path([(0, 5), (0, 3)]))
    assert a.intersects_path(b) and b.intersects_path(a)

    # self-intersect
    # 自交路径 a
    assert a.intersects_path(a)

    # a contains b
    # 创建路径 a 和 b，使其在 a 包含 b
    a = transform.transform_path(Path([(0, 0), (5, 5)]))
    b = transform.transform_path(Path([(1, 1), (3, 3)]))
    assert a.intersects_path(b) and b.intersects_path(a)

    # a and b are collinear but do not intersect
    # 创建路径 a 和 b，使其共线但不相交
    a = transform.transform_path(Path([(0, 1), (0, 5)]))
    b = transform.transform_path(Path([(3, 0), (3, 3)]))
    assert not a.intersects_path(b) and not b.intersects_path(a)

    # a and b are on the same line but do not intersect
    # 创建路径 a 和 b，使其在同一直线上但不相交
    a = transform.transform_path(Path([(0, 1), (0, 5)]))
    b = transform.transform_path(Path([(0, 6), (0, 7)]))
    assert not a.intersects_path(b) and not b.intersects_path(a)

    # Note: 1e-13 is the absolute tolerance error used for
    # `isclose` function from src/_path.h
    # 注意：1e-13 是用于 `isclose` 函数的绝对容差误差

    # a and b are parallel but do not touch
    # 创建路径 a 和 b，使其平行但不接触
    for eps in eps_array:
        a = transform.transform_path(Path([(0, 1), (0, 5)]))
        b = transform.transform_path(Path([(0 + eps, 1), (0 + eps, 5)]))
        assert not a.intersects_path(b) and not b.intersects_path(a)

    # a and b are on the same line but do not intersect (really close)
    # 创建路径 a 和 b，使其在同一直线上但非常接近而不相交
    for eps in eps_array:
        a = transform.transform_path(Path([(0, 1), (0, 5)]))
        b = transform.transform_path(Path([(0, 5 + eps), (0, 7)]))
        assert not a.intersects_path(b) and not b.intersects_path(a)

    # a and b are on the same line and intersect (really close)
    # 创建路径 a 和 b，使其在同一直线上非常接近且相交
    for eps in eps_array:
        a = transform.transform_path(Path([(0, 1), (0, 5)]))
        b = transform.transform_path(Path([(0, 5 - eps), (0, 7)]))
        assert a.intersects_path(b) and b.intersects_path(a)

    # b is the same as a but with an extra point
    # 创建路径 a 和 b，使其在 b 中多一个点
    a = transform.transform_path(Path([(0, 1), (0, 5)]))
    b = transform.transform_path(Path([(0, 1), (0, 2), (0, 5)]))
    # 确保路径a和路径b相交，并且路径b和路径a也相交
    assert a.intersects_path(b) and b.intersects_path(a)
    
    # 当路径a和路径b共线但不相交时的情况
    a = transform.transform_path(Path([(1, -1), (0, -1)]))
    b = transform.transform_path(Path([(0, 1), (0.9, 1)]))
    # 断言路径a和路径b不相交，并且路径b和路径a也不相交
    assert not a.intersects_path(b) and not b.intersects_path(a)
    
    # 当路径a和路径b共线但不相交时的情况
    a = transform.transform_path(Path([(0., -5.), (1., -5.)]))
    b = transform.transform_path(Path([(1., 5.), (0., 5.)]))
    # 断言路径a和路径b不相交，并且路径b和路径a也不相交
    assert not a.intersects_path(b) and not b.intersects_path(a)
# 使用 pytest 的 parametrize 装饰器为 test_full_arc 函数创建多个参数化测试用例，offset 的取值范围为 -720 到 360（含边界），步长为 45
@pytest.mark.parametrize('offset', range(-720, 361, 45))
def test_full_arc(offset):
    # 设定低点和高点的值，以便创建路径对象
    low = offset
    high = 360 + offset

    # 调用 Path 类的 arc 方法创建路径对象 path
    path = Path.arc(low, high)
    
    # 计算路径对象 path 的顶点中的最小值和最大值
    mins = np.min(path.vertices, axis=0)
    maxs = np.max(path.vertices, axis=0)
    
    # 使用 NumPy 的 assert_allclose 方法断言 mins 中所有值都接近 -1
    np.testing.assert_allclose(mins, -1)
    
    # 使用 NumPy 的 assert_allclose 方法断言 maxs 中所有值都接近 1
    np.testing.assert_allclose(maxs, 1)


# 定义测试函数 test_disjoint_zero_length_segment
def test_disjoint_zero_length_segment():
    # 创建 this_path 路径对象，包含5个顶点和相应的代码类型
    this_path = Path(
        np.array([
            [824.85064295, 2056.26489203],
            [861.69033931, 2041.00539016],
            [868.57864109, 2057.63522175],
            [831.73894473, 2072.89472361],
            [824.85064295, 2056.26489203]]),
        np.array([1, 2, 2, 2, 79], dtype=Path.code_type))
    
    # 创建 outline_path 路径对象，包含5个顶点和相应的代码类型
    outline_path = Path(
        np.array([
            [859.91051028, 2165.38461538],
            [859.06772495, 2149.30331334],
            [859.06772495, 2181.46591743],
            [859.91051028, 2165.38461538],
            [859.91051028, 2165.38461538]]),
        np.array([1, 2, 2, 2, 2],
                 dtype=Path.code_type))
    
    # 使用 assert 语句检查 outline_path 是否与 this_path 不相交
    assert not outline_path.intersects_path(this_path)
    
    # 使用 assert 语句检查 this_path 是否与 outline_path 不相交
    assert not this_path.intersects_path(outline_path)


# 定义测试函数 test_intersect_zero_length_segment
def test_intersect_zero_length_segment():
    # 创建 this_path 路径对象，包含2个顶点，默认使用 MOVETO 和 LINETO 代码类型
    this_path = Path(
        np.array([
            [0, 0],
            [1, 1],
        ]))
    
    # 创建 outline_path 路径对象，包含4个顶点，默认使用 MOVETO 和 LINETO 代码类型
    outline_path = Path(
        np.array([
            [1, 0],
            [.5, .5],
            [.5, .5],
            [0, 1],
        ]))
    
    # 使用 assert 语句检查 outline_path 是否与 this_path 相交
    assert outline_path.intersects_path(this_path)
    
    # 使用 assert 语句检查 this_path 是否与 outline_path 相交
    assert this_path.intersects_path(outline_path)


# 定义测试函数 test_cleanup_closepoly
def test_cleanup_closepoly():
    # 创建包含不同路径的列表 paths
    paths = [
        # 第一个路径对象，包含2个顶点和一个 CLOSEPOLY 代码类型
        Path([[np.nan, np.nan], [np.nan, np.nan]],
             [Path.MOVETO, Path.CLOSEPOLY]),
        
        # 第二个路径对象，包含2个顶点，默认使用 MOVETO 和 LINETO 代码类型
        Path([[np.nan, np.nan], [np.nan, np.nan]]),
        
        # 第三个路径对象，包含4个顶点和 MOVETO、CURVE3、CLOSEPOLY 代码类型
        Path([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
              [np.nan, np.nan]],
             [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY])
    ]
    
    # 遍历 paths 列表中的每个路径对象 p
    for p in paths:
        # 调用路径对象 p 的 cleaned 方法，移除 NaN 并返回清理后的路径对象 cleaned
        cleaned = p.cleaned(remove_nans=True)
        
        # 使用 assert 语句检查 cleaned 中路径数量是否为1
        assert len(cleaned) == 1
        
        # 使用 assert 语句检查 cleaned 中第一个路径的代码是否为 STOP
        assert cleaned.codes[0] == Path.STOP
```