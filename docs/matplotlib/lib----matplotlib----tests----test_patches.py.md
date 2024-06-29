# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_patches.py`

```py
"""
Tests specific to the patches module.
"""
# 导入必要的模块和库
import platform

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest

import matplotlib as mpl
# 导入绘图相关的模块和类
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
                                FancyArrowPatch, FancyArrow, BoxStyle, Arc)
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
    collections as mcollections, colors as mcolors, patches as mpatches,
    path as mpath, transforms as mtransforms, rcParams)


def test_Polygon_close():
    #: GitHub issue #1018 identified a bug in the Polygon handling
    #: of the closed attribute; the path was not getting closed
    #: when set_xy was used to set the vertices.

    # open set of vertices:
    # 定义一个开放的顶点集合
    xy = [[0, 0], [0, 1], [1, 1]]
    # closed set:
    # 定义一个封闭的顶点集合
    xyclosed = xy + [[0, 0]]

    # start with open path and close it:
    # 使用开放路径开始并关闭它
    p = Polygon(xy, closed=True)
    assert p.get_closed()
    assert_array_equal(p.get_xy(), xyclosed)
    p.set_xy(xy)
    assert_array_equal(p.get_xy(), xyclosed)

    # start with closed path and open it:
    # 使用封闭路径开始并打开它
    p = Polygon(xyclosed, closed=False)
    assert_array_equal(p.get_xy(), xy)
    p.set_xy(xyclosed)
    assert_array_equal(p.get_xy(), xy)

    # start with open path and leave it open:
    # 使用开放路径开始并保持开放状态
    p = Polygon(xy, closed=False)
    assert not p.get_closed()
    assert_array_equal(p.get_xy(), xy)
    p.set_xy(xy)
    assert_array_equal(p.get_xy(), xy)

    # start with closed path and leave it closed:
    # 使用封闭路径开始并保持封闭状态
    p = Polygon(xyclosed, closed=True)
    assert_array_equal(p.get_xy(), xyclosed)
    p.set_xy(xyclosed)
    assert_array_equal(p.get_xy(), xyclosed)


def test_corner_center():
    loc = [10, 20]
    width = 1
    height = 2

    # Rectangle
    # No rotation
    # 矩形，无旋转
    corners = ((10, 20), (11, 20), (11, 22), (10, 22))
    rect = Rectangle(loc, width, height)
    assert_array_equal(rect.get_corners(), corners)
    assert_array_equal(rect.get_center(), (10.5, 21))

    # 90 deg rotation
    # 90度旋转
    corners_rot = ((10, 20), (10, 21), (8, 21), (8, 20))
    rect.set_angle(90)
    assert_array_equal(rect.get_corners(), corners_rot)
    assert_array_equal(rect.get_center(), (9, 20.5))

    # Rotation not a multiple of 90 deg
    # 旋转角度非90度的倍数
    theta = 33
    t = mtransforms.Affine2D().rotate_around(*loc, np.deg2rad(theta))
    corners_rot = t.transform(corners)
    rect.set_angle(theta)
    assert_almost_equal(rect.get_corners(), corners_rot)

    # Ellipse
    # 椭圆
    loc = [loc[0] + width / 2,
           loc[1] + height / 2]
    ellipse = Ellipse(loc, width, height)

    # No rotation
    # 无旋转
    assert_array_equal(ellipse.get_corners(), corners)

    # 90 deg rotation
    # 90度旋转
    corners_rot = ((11.5, 20.5), (11.5, 21.5), (9.5, 21.5), (9.5, 20.5))
    ellipse.set_angle(90)
    assert_array_equal(ellipse.get_corners(), corners_rot)
    # Rotation shouldn't change ellipse center
    # 旋转不应改变椭圆的中心
    # 使用断言检查椭圆对象的中心是否等于预期位置 loc
    assert_array_equal(ellipse.get_center(), loc)
    
    # 设置旋转角度 theta 为 33 度
    theta = 33
    
    # 创建 Affine2D 变换对象 t，通过 rotate_around 方法以 loc 为中心旋转 theta 角度
    t = mtransforms.Affine2D().rotate_around(*loc, np.deg2rad(theta))
    
    # 将 corners 列表中的点根据 t 变换对象进行坐标变换，得到旋转后的 corners_rot
    corners_rot = t.transform(corners)
    
    # 设置椭圆对象的角度为 theta
    ellipse.set_angle(theta)
    
    # 使用断言检查椭圆对象的角落点是否与旋转后的 corners_rot 几乎相等
    assert_almost_equal(ellipse.get_corners(), corners_rot)
def test_ellipse_vertices():
    # 定义一个椭圆对象，中心在 (0, 0)，宽度和高度都为 0，角度为 0
    ellipse = Ellipse(xy=(0, 0), width=0, height=0, angle=0)
    # 断言椭圆顶点的坐标，期望为 [(0.0, 0.0), (0.0, 0.0)]
    assert_almost_equal(
        ellipse.get_vertices(),
        [(0.0, 0.0), (0.0, 0.0)],
    )
    # 断言椭圆边界框的顶点坐标，期望为 [(0.0, 0.0), (0.0, 0.0)]
    assert_almost_equal(
        ellipse.get_co_vertices(),
        [(0.0, 0.0), (0.0, 0.0)],
    )

    # 定义一个椭圆对象，中心在 (0, 0)，宽度为 2，高度为 1，角度为 30
    ellipse = Ellipse(xy=(0, 0), width=2, height=1, angle=30)
    # 断言椭圆顶点的坐标，根据椭圆参数计算
    assert_almost_equal(
        ellipse.get_vertices(),
        [
            (
                ellipse.center[0] + ellipse.width / 4 * np.sqrt(3),
                ellipse.center[1] + ellipse.width / 4,
            ),
            (
                ellipse.center[0] - ellipse.width / 4 * np.sqrt(3),
                ellipse.center[1] - ellipse.width / 4,
            ),
        ],
    )
    # 断言椭圆边界框的顶点坐标，根据椭圆参数计算
    assert_almost_equal(
        ellipse.get_co_vertices(),
        [
            (
                ellipse.center[0] - ellipse.height / 4,
                ellipse.center[1] + ellipse.height / 4 * np.sqrt(3),
            ),
            (
                ellipse.center[0] + ellipse.height / 4,
                ellipse.center[1] - ellipse.height / 4 * np.sqrt(3),
            ),
        ],
    )
    # 取得椭圆顶点的坐标并计算其平均值，与椭圆中心坐标进行近似断言
    v1, v2 = np.array(ellipse.get_vertices())
    np.testing.assert_almost_equal((v1 + v2) / 2, ellipse.center)
    # 取得椭圆边界框顶点的坐标并计算其平均值，与椭圆中心坐标进行近似断言
    v1, v2 = np.array(ellipse.get_co_vertices())
    np.testing.assert_almost_equal((v1 + v2) / 2, ellipse.center)

    # 定义一个椭圆对象，中心在 (2.252, -10.859)，宽度为 2.265，高度为 1.98，角度为 68.78
    ellipse = Ellipse(xy=(2.252, -10.859), width=2.265, height=1.98, angle=68.78)
    # 取得椭圆顶点的坐标并计算其平均值，与椭圆中心坐标进行近似断言
    v1, v2 = np.array(ellipse.get_vertices())
    np.testing.assert_almost_equal((v1 + v2) / 2, ellipse.center)
    # 取得椭圆边界框顶点的坐标并计算其平均值，与椭圆中心坐标进行近似断言
    v1, v2 = np.array(ellipse.get_co_vertices())
    np.testing.assert_almost_equal((v1 + v2) / 2, ellipse.center)


def test_rotate_rect():
    loc = np.asarray([1.0, 2.0])
    width = 2
    height = 3
    angle = 30.0

    # 创建一个旋转的矩形对象
    rect1 = Rectangle(loc, width, height, angle=angle)

    # 创建一个未旋转的矩形对象
    rect2 = Rectangle(loc, width, height)

    # 设置一个显式的旋转矩阵（以弧度表示）
    angle_rad = np.pi * angle / 180.0
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad),  np.cos(angle_rad)]])

    # 将矩形顶点平移到原点，应用旋转矩阵，然后再平移回原位置
    new_verts = np.inner(rotation_matrix, rect2.get_verts() - loc).T + loc

    # 断言旋转后的矩形顶点坐标与手动旋转后的顶点坐标相近
    assert_almost_equal(rect1.get_verts(), new_verts)


@check_figures_equal(extensions=['png'])
def test_rotate_rect_draw(fig_test, fig_ref):
    ax_test = fig_test.add_subplot()
    ax_ref = fig_ref.add_subplot()

    loc = (0, 0)
    width, height = (1, 1)
    angle = 30
    # 创建一个旋转角度为 30 的矩形对象
    rect_ref = Rectangle(loc, width, height, angle=angle)
    ax_ref.add_patch(rect_ref)
    # 断言矩形对象的角度属性与设置的角度相等
    assert rect_ref.get_angle() == angle

    # 检查当将矩形对象添加到坐标轴后更新角度属性，确保图形被标记为过时并在正确位置重绘
    rect_test = Rectangle(loc, width, height)
    # 断言检查矩形对象的角度是否为0
    assert rect_test.get_angle() == 0
    # 将矩形对象添加到图形坐标轴中
    ax_test.add_patch(rect_test)
    # 设置矩形对象的角度为给定的角度值
    rect_test.set_angle(angle)
    # 断言检查矩形对象的角度是否已经设置为给定的角度值
    assert rect_test.get_angle() == angle
@check_figures_equal(extensions=['png'])
def test_dash_offset_patch_draw(fig_test, fig_ref):
    # 在测试和参考图表中各添加一个子图
    ax_test = fig_test.add_subplot()
    ax_ref = fig_ref.add_subplot()

    # 设置矩形的位置、宽度和高度，指定线宽、边框颜色和虚线样式
    loc = (0.1, 0.1)
    width, height = (0.8, 0.8)
    rect_ref = Rectangle(loc, width, height, linewidth=3, edgecolor='b',
                                                linestyle=(0, [6, 6]))
    # 使用线型 (0, [0, 6, 6, 0]) 填充线条间隙，等同于 (6, [6, 6]) 但没有虚线偏移
    rect_ref2 = Rectangle(loc, width, height, linewidth=3, edgecolor='r',
                                            linestyle=(0, [0, 6, 6, 0]))
    # 断言确认矩形的线型设置正确
    assert rect_ref.get_linestyle() == (0, [6, 6])
    assert rect_ref2.get_linestyle() == (0, [0, 6, 6, 0])

    # 将矩形添加到参考图表的子图中
    ax_ref.add_patch(rect_ref)
    ax_ref.add_patch(rect_ref2)

    # 检查矩形的虚线偏移，一种是在初始化方法中传递，另一种是创建带有正确 onoff 序列的两个矩形
    rect_test = Rectangle(loc, width, height, linewidth=3, edgecolor='b',
                                                    linestyle=(0, [6, 6]))
    rect_test2 = Rectangle(loc, width, height, linewidth=3, edgecolor='r',
                                                    linestyle=(6, [6, 6]))
    assert rect_test.get_linestyle() == (0, [6, 6])
    assert rect_test2.get_linestyle() == (6, [6, 6])

    # 将矩形添加到测试图表的子图中
    ax_test.add_patch(rect_test)
    ax_test.add_patch(rect_test2)


def test_negative_rect():
    # 这两个矩形具有相同的顶点，但从不同的起点开始。（同时删除最后一个重复的顶点。）
    pos_vertices = Rectangle((-3, -2), 3, 2).get_verts()[:-1]
    neg_vertices = Rectangle((0, 0), -3, -2).get_verts()[:-1]
    # 断言确认数组是否相等，通过滚动实现顶点位置的比较
    assert_array_equal(np.roll(neg_vertices, 2, 0), pos_vertices)


@image_comparison(['clip_to_bbox'])
def test_clip_to_bbox():
    # 创建一个图表和一个坐标轴
    fig, ax = plt.subplots()
    # 设置坐标轴的 X 和 Y 范围
    ax.set_xlim([-18, 20])
    ax.set_ylim([-150, 100])

    # 创建一个正则星形路径并做适当的变换和偏移
    path = mpath.Path.unit_regular_star(8).deepcopy()
    path.vertices *= [10, 100]
    path.vertices -= [5, 25]

    # 创建一个单位圆形路径并做适当的变换和偏移
    path2 = mpath.Path.unit_circle().deepcopy()
    path2.vertices *= [10, 100]
    path2.vertices += [10, -25]

    # 将两个路径合并成一个复合路径
    combined = mpath.Path.make_compound_path(path, path2)

    # 创建一个路径补丁，并设置其透明度、填充颜色和边框颜色
    patch = mpatches.PathPatch(
        combined, alpha=0.5, facecolor='coral', edgecolor='none')
    # 将路径补丁添加到坐标轴中
    ax.add_patch(patch)

    # 创建一个边界框，并将复合路径裁剪到该边界框内
    bbox = mtransforms.Bbox([[-12, -77.5], [50, -110]])
    result_path = combined.clip_to_bbox(bbox)
    # 创建一个新的路径补丁，设置其透明度、填充颜色、线宽和边框颜色
    result_patch = mpatches.PathPatch(
        result_path, alpha=0.5, facecolor='green', lw=4, edgecolor='black')

    # 将裁剪后的路径补丁添加到坐标轴中
    ax.add_patch(result_patch)


@image_comparison(['patch_alpha_coloring'], remove_text=True)
def test_patch_alpha_coloring():
    """
    测试检查补丁和集合是否以指定的透明度渲染其填充颜色和边框颜色。
    """
    # 创建一个正则星形路径
    star = mpath.Path.unit_regular_star(6)
    # 创建一个单位圆形路径
    circle = mpath.Path.unit_circle()
    # 将星形和圆形的顶点连接起来，创建一个新的路径
    verts = np.concatenate([circle.vertices, star.vertices[::-1]])
    # 将星形和圆形的路径代码连接起来，以定义新路径的形状
    codes = np.concatenate([circle.codes, star.codes])
    # 创建一个新的路径对象，表示带有内部剪切的星形加圆形的复合图形
    cut_star1 = mpath.Path(verts, codes)
    # 创建另一个带有稍微移动顶点的路径对象，用于显示效果对比
    cut_star2 = mpath.Path(verts + 1, codes)

    # 创建一个新的绘图轴对象
    ax = plt.axes()
    # 创建一个路径集合对象，用于绘制带有内部剪切的星形加圆形的复合图形
    col = mcollections.PathCollection([cut_star2],
                                      linewidth=5, linestyles='dashdot',
                                      facecolor=(1, 0, 0, 0.5),
                                      edgecolor=(0, 0, 1, 0.75))
    # 将路径集合对象添加到绘图轴中
    ax.add_collection(col)

    # 创建一个路径补丁对象，用于绘制带有内部剪切的星形加圆形的复合图形
    patch = mpatches.PathPatch(cut_star1,
                               linewidth=5, linestyle='dashdot',
                               facecolor=(1, 0, 0, 0.5),
                               edgecolor=(0, 0, 1, 0.75))
    # 将路径补丁对象添加到绘图轴中
    ax.add_patch(patch)

    # 设置绘图轴的 X 轴范围
    ax.set_xlim(-1, 2)
    # 设置绘图轴的 Y 轴范围
    ax.set_ylim(-1, 2)
@image_comparison(['patch_alpha_override'], remove_text=True)
def test_patch_alpha_override():
    #: Test checks that specifying an alpha attribute for a patch or
    #: collection will override any alpha component of the facecolor
    #: or edgecolor.
    # 创建一个六角星形路径对象
    star = mpath.Path.unit_regular_star(6)
    # 创建一个单位圆路径对象
    circle = mpath.Path.unit_circle()
    # 将圆的顶点与星形的顶点（反向）拼接起来
    verts = np.concatenate([circle.vertices, star.vertices[::-1]])
    # 将圆的路径代码与星形的路径代码拼接起来
    codes = np.concatenate([circle.codes, star.codes])
    # 创建一个带有内部剪裁的星形路径对象
    cut_star1 = mpath.Path(verts, codes)
    cut_star2 = mpath.Path(verts + 1, codes)

    # 在当前图形对象上创建一个轴对象
    ax = plt.axes()
    # 创建一个路径集合对象，包含cut_star2路径
    col = mcollections.PathCollection([cut_star2],
                                      linewidth=5, linestyles='dashdot',
                                      alpha=0.25,
                                      facecolor=(1, 0, 0, 0.5),
                                      edgecolor=(0, 0, 1, 0.75))
    # 将路径集合对象添加到轴对象中
    ax.add_collection(col)

    # 创建一个路径补丁对象，使用cut_star1路径
    patch = mpatches.PathPatch(cut_star1,
                               linewidth=5, linestyle='dashdot',
                               alpha=0.25,
                               facecolor=(1, 0, 0, 0.5),
                               edgecolor=(0, 0, 1, 0.75))
    # 将路径补丁对象添加到轴对象中
    ax.add_patch(patch)

    # 设置轴的X轴范围
    ax.set_xlim(-1, 2)
    # 设置轴的Y轴范围
    ax.set_ylim(-1, 2)


@mpl.style.context('default')
def test_patch_color_none():
    # Make sure the alpha kwarg does not override 'none' facecolor.
    # Addresses issue #7478.
    # 创建一个半径为1，面颜色为'none'的圆对象
    c = plt.Circle((0, 0), 1, facecolor='none', alpha=1)
    # 断言圆的面颜色的第一个分量是否为0
    assert c.get_facecolor()[0] == 0


@image_comparison(['patch_custom_linestyle'], remove_text=True)
def test_patch_custom_linestyle():
    #: A test to check that patches and collections accept custom dash
    #: patterns as linestyle and that they display correctly.
    # 创建一个六角星形路径对象
    star = mpath.Path.unit_regular_star(6)
    # 创建一个单位圆路径对象
    circle = mpath.Path.unit_circle()
    # 将圆的顶点与星形的顶点（反向）拼接起来
    verts = np.concatenate([circle.vertices, star.vertices[::-1]])
    # 将圆的路径代码与星形的路径代码拼接起来
    codes = np.concatenate([circle.codes, star.codes])
    # 创建一个带有内部剪裁的星形路径对象
    cut_star1 = mpath.Path(verts, codes)
    cut_star2 = mpath.Path(verts + 1, codes)

    # 在当前图形对象上创建一个轴对象
    ax = plt.axes()
    # 创建一个路径集合对象，包含cut_star2路径
    col = mcollections.PathCollection(
        [cut_star2],
        linewidth=5, linestyles=[(0, (5, 7, 10, 7))],
        facecolor=(1, 0, 0), edgecolor=(0, 0, 1))
    # 将路径集合对象添加到轴对象中
    ax.add_collection(col)

    # 创建一个路径补丁对象，使用cut_star1路径
    patch = mpatches.PathPatch(
        cut_star1,
        linewidth=5, linestyle=(0, (5, 7, 10, 7)),
        facecolor=(1, 0, 0), edgecolor=(0, 0, 1))
    # 将路径补丁对象添加到轴对象中
    ax.add_patch(patch)

    # 设置轴的X轴范围
    ax.set_xlim(-1, 2)
    # 设置轴的Y轴范围
    ax.set_ylim(-1, 2)


def test_patch_linestyle_accents():
    #: Test if linestyle can also be specified with short mnemonics like "--"
    #: c.f. GitHub issue #2136
    # 创建一个六角星形路径对象
    star = mpath.Path.unit_regular_star(6)
    # 创建一个单位圆路径对象
    circle = mpath.Path.unit_circle()
    # 将圆的顶点与星形的顶点（反向）拼接起来
    verts = np.concatenate([circle.vertices, star.vertices[::-1]])
    # 将圆的路径代码与星形的路径代码拼接起来
    codes = np.concatenate([circle.codes, star.codes])
    # 定义不同线型样式的列表
    linestyles = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
    
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    
    # 遍历线型列表，使用索引 i 和线型 ls
    for i, ls in enumerate(linestyles):
        # 创建一个路径对象，使用预定义的顶点和代码
        star = mpath.Path(verts + i, codes)
        
        # 创建一个路径补丁对象，将路径对象转换为图形对象
        patch = mpatches.PathPatch(star,
                                   linewidth=3, linestyle=ls,
                                   facecolor=(1, 0, 0),  # 设置填充颜色为红色
                                   edgecolor=(0, 0, 1))  # 设置边缘颜色为蓝色
        
        # 将路径补丁对象添加到坐标轴对象上
        ax.add_patch(patch)
    
    # 设置坐标轴的 x 和 y 范围
    ax.set_xlim([-1, i + 1])
    ax.set_ylim([-1, i + 1])
    
    # 在绘制完成后更新图形对象的画布
    fig.canvas.draw()
# 使用装饰器 `@check_figures_equal`，用于测试图片的特定属性是否相等，扩展名为 `png`
@check_figures_equal(extensions=['png'])
# 定义测试函数 `test_patch_linestyle_none`，测试不同线型的图形绘制
def test_patch_linestyle_none(fig_test, fig_ref):
    # 创建一个单位圆形路径对象
    circle = mpath.Path.unit_circle()

    # 在测试图中添加子图 `ax_test`
    ax_test = fig_test.add_subplot()
    # 在参考图中添加子图 `ax_ref`
    ax_ref = fig_ref.add_subplot()

    # 遍历不同线型的列表
    for i, ls in enumerate(['none', 'None', ' ', '']):
        # 创建路径对象 `path`，基于圆形顶点和代码
        path = mpath.Path(circle.vertices + i, circle.codes)
        # 创建路径补丁对象 `patch`，设置线宽为 3，线型为当前循环变量 `ls`，填充颜色为红色，边缘颜色为蓝色
        patch = mpatches.PathPatch(path,
                                   linewidth=3, linestyle=ls,
                                   facecolor=(1, 0, 0),
                                   edgecolor=(0, 0, 1))
        # 将补丁对象添加到 `ax_test` 子图中
        ax_test.add_patch(patch)

        # 创建另一个路径补丁对象 `patch`，设置线宽为 3，线型为实线 `-`，填充颜色为红色，无边缘颜色
        patch = mpatches.PathPatch(path,
                                   linewidth=3, linestyle='-',
                                   facecolor=(1, 0, 0),
                                   edgecolor='none')
        # 将补丁对象添加到 `ax_ref` 子图中
        ax_ref.add_patch(patch)

    # 设置测试图 `ax_test` 和参考图 `ax_ref` 的 x 和 y 轴限制
    ax_test.set_xlim([-1, i + 1])
    ax_test.set_ylim([-1, i + 1])
    ax_ref.set_xlim([-1, i + 1])
    ax_ref.set_ylim([-1, i + 1])


# 定义测试函数 `test_wedge_movement`
def test_wedge_movement():
    # 定义参数字典 `param_dict`，包含圆楔图形的不同属性及其初始值、新值和设置方法
    param_dict = {'center': ((0, 0), (1, 1), 'set_center'),
                  'r': (5, 8, 'set_radius'),
                  'width': (2, 3, 'set_width'),
                  'theta1': (0, 30, 'set_theta1'),
                  'theta2': (45, 50, 'set_theta2')}

    # 根据 `param_dict` 创建初始参数字典 `init_args`
    init_args = {k: v[0] for k, v in param_dict.items()}

    # 创建圆楔对象 `w`，初始参数使用 `init_args`
    w = mpatches.Wedge(**init_args)

    # 遍历参数字典 `param_dict`
    for attr, (old_v, new_v, func) in param_dict.items():
        # 断言圆楔对象 `w` 的当前属性值等于初始值 `old_v`
        assert getattr(w, attr) == old_v
        # 调用圆楔对象 `w` 的设置方法 `func`，将属性 `attr` 设置为新值 `new_v`
        getattr(w, func)(new_v)
        # 再次断言圆楔对象 `w` 的属性值等于新值 `new_v`
        assert getattr(w, attr) == new_v


# 使用装饰器 `@image_comparison`，比较生成的图像和参考图像，移除文本部分，设置公差值
@image_comparison(['wedge_range'], remove_text=True,
                  tol=0.009 if platform.machine() == 'arm64' else 0)
# 定义测试函数 `test_wedge_range`
def test_wedge_range():
    # 在当前图形上创建坐标轴 `ax`
    ax = plt.axes()

    # 定义角度 `t1` 作为常数
    t1 = 2.313869244286224

    # 定义多组不同的起始角度和结束角度列表 `args`
    args = [[52.31386924, 232.31386924],
            [52.313869244286224, 232.31386924428622],
            [t1, t1 + 180.0],
            [0, 360],
            [90, 90 + 360],
            [-180, 180],
            [0, 380],
            [45, 46],
            [46, 45]]

    # 遍历参数列表 `args`
    for i, (theta1, theta2) in enumerate(args):
        # 计算当前索引 `i` 的 x 和 y 坐标
        x = i % 3
        y = i // 3

        # 创建圆楔对象 `wedge`，设置其位置、半径、起始角度和结束角度，填充颜色为空，边缘颜色为黑色，线宽为 3
        wedge = mpatches.Wedge((x * 3, y * 3), 1, theta1, theta2,
                               facecolor='none', edgecolor='k', lw=3)

        # 将圆楔对象 `wedge` 添加到坐标轴 `ax` 上
        ax.add_artist(wedge)

    # 设置坐标轴 `ax` 的 x 和 y 轴限制
    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 9)


# 定义测试函数 `test_patch_str`，检查补丁对象的字符串表示是否正常工作
def test_patch_str():
    """
    Check that patches have nice and working `str` representation.

    Note that the logic is that `__str__` is defined such that:
    str(eval(str(p))) == str(p)
    """
    # 创建圆形补丁对象 `p`，设置其位置 `(1, 2)` 和半径 `3`
    p = mpatches.Circle(xy=(1, 2), radius=3)
    # 断言圆形补丁对象 `p` 的字符串表示等于给定字符串
    assert str(p) == 'Circle(xy=(1, 2), radius=3)'

    # 创建椭圆补丁对象 `p`，设置其位置 `(1, 2)`、宽度 `3`、高度 `4` 和角度 `5`
    p = mpatches.Ellipse(xy=(1, 2), width=3, height=4, angle=5)
    # 断言椭圆补丁对象 `p` 的字符串表示等于给定字符串
    assert str(p) == 'Ellipse(xy=(1, 2), width=3, height=4, angle=5)'

    # 创建矩形补丁对象 `p`，设置其位置 `(1, 2)`、宽度 `3`、高度 `4` 和角度 `5`
    p = mpatches.Rectangle(xy=(1, 2), width=3, height=4, angle=5)
    # 断言矩形补丁对象 `p` 的字符串表示等于给定字符串
    assert str(p) == 'Rectangle(xy=(1, 2), width=3, height=4, angle=5)'

    # 创建圆楔补丁对象 `p`，设置其中心 `(1, 2)`、半径 `3`、起始角度 `4`、结束角度 `5` 和宽度 `6`
    p = mpatches.Wedge(center=(1, 2), r=3, theta1=4, theta2=5, width=6)
    # 断言圆楔补丁对象 `p` 的字符串表示等于给定字符串
    assert str(p) == 'Wedge(center=(1, 2), r=3, theta1=4, theta2=5, width=6)'
    # 创建一个椭圆弧对象，指定其位置、宽度、高度、角度、起始角度和结束角度
    p = mpatches.Arc(xy=(1, 2), width=3, height=4, angle=5, theta1=6, theta2=7)
    # 期望的字符串表示形式
    expected = 'Arc(xy=(1, 2), width=3, height=4, angle=5, theta1=6, theta2=7)'
    # 断言椭圆弧对象的字符串表示是否与期望的一致
    assert str(p) == expected
    
    # 创建一个环状图形对象，指定其位置、内外半径、宽度和角度
    p = mpatches.Annulus(xy=(1, 2), r=(3, 4), width=1, angle=2)
    # 期望的字符串表示形式
    expected = "Annulus(xy=(1, 2), r=(3, 4), width=1, angle=2)"
    # 断言环状图形对象的字符串表示是否与期望的一致
    assert str(p) == expected
    
    # 创建一个正多边形对象，指定其中心位置、边数和半径
    p = mpatches.RegularPolygon((1, 2), 20, radius=5)
    # 断言正多边形对象的字符串表示是否与期望的一致
    assert str(p) == "RegularPolygon((1, 2), 20, radius=5, orientation=0)"
    
    # 创建一个圆形多边形对象，指定其位置、半径和分辨率
    p = mpatches.CirclePolygon(xy=(1, 2), radius=5, resolution=20)
    # 断言圆形多边形对象的字符串表示是否与期望的一致
    assert str(p) == "CirclePolygon((1, 2), radius=5, resolution=20)"
    
    # 创建一个带边框样式的矩形对象，指定其位置、宽度和高度
    p = mpatches.FancyBboxPatch((1, 2), width=3, height=4)
    # 断言带边框样式矩形对象的字符串表示是否与期望的一致
    assert str(p) == "FancyBboxPatch((1, 2), width=3, height=4)"
    
    # 创建一个路径对象，指定其顶点坐标和是否闭合
    path = mpath.Path([(1, 2), (2, 2), (1, 2)], closed=True)
    # 创建一个基于路径对象的路径补丁对象
    p = mpatches.PathPatch(path)
    # 断言路径补丁对象的字符串表示是否与期望的一致
    assert str(p) == "PathPatch3((1, 2) ...)"
    
    # 创建一个空的多边形对象
    p = mpatches.Polygon(np.empty((0, 2)))
    # 断言空多边形对象的字符串表示是否与期望的一致
    assert str(p) == "Polygon0()"
    
    # 创建一个多边形对象，指定其顶点坐标数据
    data = [[1, 2], [2, 2], [1, 2]]
    p = mpatches.Polygon(data)
    # 断言多边形对象的字符串表示是否与期望的一致
    assert str(p) == "Polygon3((1, 2) ...)"
    
    # 创建一个带箭头的路径补丁对象，指定其基于的路径对象
    p = mpatches.FancyArrowPatch(path=path)
    # 断言带箭头的路径补丁对象的字符串表示的前27个字符是否与期望的一致
    assert str(p)[:27] == "FancyArrowPatch(Path(array("
    
    # 创建一个带箭头的路径补丁对象，指定其起始点和终点位置
    p = mpatches.FancyArrowPatch((1, 2), (3, 4))
    # 断言带箭头的路径补丁对象的字符串表示是否与期望的一致
    assert str(p) == "FancyArrowPatch((1, 2)->(3, 4))"
    
    # 创建一个连接补丁对象，指定其起始点、终点和连接类型
    p = mpatches.ConnectionPatch((1, 2), (3, 4), 'data')
    # 断言连接补丁对象的字符串表示是否与期望的一致
    assert str(p) == "ConnectionPatch((1, 2), (3, 4))"
    
    # 创建一个阴影对象，指定其基于的补丁对象和阴影的偏移量
    s = mpatches.Shadow(p, 1, 1)
    # 断言阴影对象的字符串表示是否与期望的一致
    assert str(s) == "Shadow(ConnectionPatch((1, 2), (3, 4)))"
@image_comparison(['multi_color_hatch'], remove_text=True, style='default')
def test_multi_color_hatch():
    fig, ax = plt.subplots()  # 创建一个新的图形和子图对象

    rects = ax.bar(range(5), range(1, 6))  # 在子图上创建一个包含五个条形图的对象
    for i, rect in enumerate(rects):
        rect.set_facecolor('none')  # 设置条形图的填充颜色为透明
        rect.set_edgecolor(f'C{i}')  # 设置条形图的边框颜色为'C{i}'
        rect.set_hatch('/')  # 设置条形图的填充图案为斜线'/'

    ax.autoscale_view()  # 自动调整坐标轴的视图范围
    ax.autoscale(False)  # 关闭自动调整坐标轴的功能

    for i in range(5):
        with mpl.style.context({'hatch.color': f'C{i}'}):  # 使用特定的样式上下文，设置斜线图案的颜色为'C{i}'
            r = Rectangle((i - .8 / 2, 5), .8, 1, hatch='//', fc='none')  # 在子图上创建一个矩形对象，设置斜线图案和填充颜色为透明
        ax.add_patch(r)  # 将矩形对象添加到子图中


@image_comparison(['units_rectangle.png'])
def test_units_rectangle():
    import matplotlib.testing.jpl_units as U
    U.register()  # 注册单位

    p = mpatches.Rectangle((5*U.km, 6*U.km), 1*U.km, 2*U.km)  # 创建一个矩形对象，使用注册的单位进行设置

    fig, ax = plt.subplots()  # 创建一个新的图形和子图对象
    ax.add_patch(p)  # 将矩形对象添加到子图中
    ax.set_xlim([4*U.km, 7*U.km])  # 设置子图的 x 轴范围，使用注册的单位
    ax.set_ylim([5*U.km, 9*U.km])  # 设置子图的 y 轴范围，使用注册的单位


@image_comparison(['connection_patch.png'], style='mpl20', remove_text=True,
                  tol=0.024 if platform.machine() == 'arm64' else 0)
def test_connection_patch():
    fig, (ax1, ax2) = plt.subplots(1, 2)  # 创建一个包含两个子图的图形对象

    con = mpatches.ConnectionPatch(xyA=(0.1, 0.1), xyB=(0.9, 0.9),
                                   coordsA='data', coordsB='data',
                                   axesA=ax2, axesB=ax1,
                                   arrowstyle="->")  # 创建一个连接补丁对象，连接两个子图的特定数据点

    ax2.add_artist(con)  # 在第二个子图上添加连接补丁对象

    xyA = (0.6, 1.0)  # 在轴坐标中指定点的位置
    xyB = (0.0, 0.2)  # x 轴坐标中的点，y 轴数据坐标中的点
    coordsA = "axes fraction"  # 指定 xyA 使用的坐标系
    coordsB = ax2.get_yaxis_transform()  # 获取第二个子图的 y 轴转换
    con = mpatches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA,
                                   coordsB=coordsB, arrowstyle="-")  # 创建另一个连接补丁对象，连接指定的点
    ax2.add_artist(con)  # 在第二个子图上添加连接补丁对象


@check_figures_equal(extensions=["png"])
def test_connection_patch_fig(fig_test, fig_ref):
    # Test that connection patch can be added as figure artist, and that figure
    # pixels count negative values from the top right corner (this API may be
    # changed in the future).
    ax1, ax2 = fig_test.subplots(1, 2)  # 在测试图形中创建包含两个子图的对象
    con = mpatches.ConnectionPatch(
        xyA=(.3, .2), coordsA="data", axesA=ax1,
        xyB=(-30, -20), coordsB="figure pixels",
        arrowstyle="->", shrinkB=5)  # 创建一个连接补丁对象，连接数据点和图形像素

    fig_test.add_artist(con)  # 在测试图形中添加连接补丁对象

    ax1, ax2 = fig_ref.subplots(1, 2)  # 在参考图形中创建包含两个子图的对象
    bb = fig_ref.bbox  # 获取参考图形的边界框
    # Necessary so that pixel counts match on both sides.
    plt.rcParams["savefig.dpi"] = plt.rcParams["figure.dpi"]  # 设置保存图形时的 DPI 以匹配像素计数

    con = mpatches.ConnectionPatch(
        xyA=(.3, .2), coordsA="data", axesA=ax1,
        xyB=(bb.width - 30, bb.height - 20), coordsB="figure pixels",
        arrowstyle="->", shrinkB=5)  # 创建另一个连接补丁对象，连接数据点和图形像素
    fig_ref.add_artist(con)  # 在参考图形中添加连接补丁对象


def test_datetime_rectangle():
    # Check that creating a rectangle with timedeltas doesn't fail
    from datetime import datetime, timedelta

    start = datetime(2017, 1, 1, 0, 0, 0)  # 创建一个起始日期时间对象
    delta = timedelta(seconds=16)  # 创建一个时间增量对象
    patch = mpatches.Rectangle((start, 0), delta, 1)  # 创建一个矩形对象，使用日期时间和时间增量对象作为参数

    fig, ax = plt.subplots()  # 创建一个新的图形和子图对象
    ax.add_patch(patch)  # 将矩形对象添加到子图中


def test_datetime_datetime_fails():
    pass  # 空函数，未实现内容
    # 导入 datetime 模块中的 datetime 类
    from datetime import datetime
    
    # 创建一个 datetime 对象，表示起始时间为2017年1月1日 00:00:00
    start = datetime(2017, 1, 1, 0, 0, 0)
    
    # 创建一个 datetime 时间间隔对象，表示从1970年1月1日至1970年1月5日的时间间隔
    # 如果单位设置错误，将导致时间间隔为5天
    dt_delta = datetime(1970, 1, 5)
    
    # 使用 pytest 的上下文管理器，检查是否会抛出 TypeError 异常
    with pytest.raises(TypeError):
        # 尝试创建一个矩形对象，左下角顶点为 (start, 0)，宽度为 dt_delta，高度为 1
        mpatches.Rectangle((start, 0), dt_delta, 1)
    
    # 使用 pytest 的上下文管理器，检查是否会抛出 TypeError 异常
    with pytest.raises(TypeError):
        # 尝试创建一个矩形对象，左下角顶点为 (0, start)，宽度为 1，高度为 dt_delta
        mpatches.Rectangle((0, start), 1, dt_delta)
def test_contains_point():
    # 创建一个椭圆对象，中心为 (0.5, 0.5)，长轴半径为 0.5，短轴半径为 1.0
    ell = mpatches.Ellipse((0.5, 0.5), 0.5, 1.0)
    # 待测试的点列表
    points = [(0.0, 0.5), (0.2, 0.5), (0.25, 0.5), (0.5, 0.5)]
    # 获取椭圆的路径对象
    path = ell.get_path()
    # 获取椭圆的变换对象
    transform = ell.get_transform()
    # 处理椭圆的半径，此处为私有方法，传入 None 则自动计算
    radius = ell._process_radius(None)
    # 期望的结果，使用路径对象判断每个点是否在椭圆内
    expected = np.array([path.contains_point(point,
                                             transform,
                                             radius) for point in points])
    # 实际结果，直接调用椭圆对象的 contains_point 方法
    result = np.array([ell.contains_point(point) for point in points])
    # 断言：期望结果与实际结果相同
    assert np.all(result == expected)


def test_contains_points():
    # 创建一个椭圆对象，中心为 (0.5, 0.5)，长轴半径为 0.5，短轴半径为 1.0
    ell = mpatches.Ellipse((0.5, 0.5), 0.5, 1.0)
    # 待测试的点列表
    points = [(0.0, 0.5), (0.2, 0.5), (0.25, 0.5), (0.5, 0.5)]
    # 获取椭圆的路径对象
    path = ell.get_path()
    # 获取椭圆的变换对象
    transform = ell.get_transform()
    # 处理椭圆的半径，此处为私有方法，传入 None 则自动计算
    radius = ell._process_radius(None)
    # 期望的结果，使用路径对象判断所有点是否在椭圆内
    expected = path.contains_points(points, transform, radius)
    # 实际结果，直接调用椭圆对象的 contains_points 方法
    result = ell.contains_points(points)
    # 断言：期望结果与实际结果相同
    assert np.all(result == expected)


# Currently fails with pdf/svg, probably because some parts assume a dpi of 72.
@check_figures_equal(extensions=["png"])
def test_shadow(fig_test, fig_ref):
    # 阴影的起始位置
    xy = np.array([.2, .3])
    # 阴影的偏移量
    dxy = np.array([.1, .2])
    # 调整保存图像的 dpi 设置为 "figure"
    plt.rcParams["savefig.dpi"] = "figure"
    # 测试图像
    a1 = fig_test.subplots()
    # 创建一个矩形对象，位于指定位置，宽度为 0.5，高度为 0.5
    rect = mpatches.Rectangle(xy=xy, width=.5, height=.5)
    # 创建矩形对象的阴影，指定偏移量
    shadow = mpatches.Shadow(rect, ox=dxy[0], oy=dxy[1])
    # 将矩形和阴影添加到测试图像中
    a1.add_patch(rect)
    a1.add_patch(shadow)
    # 参考图像
    a2 = fig_ref.subplots()
    # 创建一个矩形对象，位于指定位置，宽度为 0.5，高度为 0.5
    rect = mpatches.Rectangle(xy=xy, width=.5, height=.5)
    # 创建矩形对象的阴影，根据图像 dpi 调整偏移量
    shadow = mpatches.Rectangle(
        xy=xy + fig_ref.dpi / 72 * dxy, width=.5, height=.5,
        fc=np.asarray(mcolors.to_rgb(rect.get_facecolor())) * .3,
        ec=np.asarray(mcolors.to_rgb(rect.get_facecolor())) * .3,
        alpha=.5)
    # 将矩形和阴影添加到参考图像中
    a2.add_patch(shadow)
    a2.add_patch(rect)


def test_fancyarrow_units():
    from datetime import datetime
    # 简单测试，检查 FancyArrowPatch 是否支持单位
    dtime = datetime(2000, 1, 1)
    fig, ax = plt.subplots()
    # 创建一个箭头对象，起始位置为 (0, dtime)，结束位置为 (0.01, dtime)
    arrow = FancyArrowPatch((0, dtime), (0.01, dtime))


def test_fancyarrow_setdata():
    fig, ax = plt.subplots()
    # 创建一个箭头对象，起始位置为 (0, 0)，方向为 (10, 10)，头部长度为 5，头部宽度为 1，宽度为 0.5
    arrow = ax.arrow(0, 0, 10, 10, head_length=5, head_width=1, width=.5)
    # 期望的第一组顶点坐标
    expected1 = np.array(
      [[13.54, 13.54],
       [10.35,  9.65],
       [10.18,  9.82],
       [0.18, -0.18],
       [-0.18,  0.18],
       [9.82, 10.18],
       [9.65, 10.35],
       [13.54, 13.54]]
    )
    # 断言：顶点坐标是否接近期望值
    assert np.allclose(expected1, np.round(arrow.verts, 2))

    # 期望的第二组顶点坐标
    expected2 = np.array(
      [[16.71, 16.71],
       [16.71, 15.29],
       [16.71, 15.29],
       [1.71,  0.29],
       [0.29,  1.71],
       [15.29, 16.71],
       [15.29, 16.71],
       [16.71, 16.71]]
    )
    # 更新箭头对象的数据
    arrow.set_data(
        x=1, y=1, dx=15, dy=15, width=2, head_width=2, head_length=1
    )
    # 断言：顶点坐标是否接近期望值
    assert np.allclose(expected2, np.round(arrow.verts, 2))


@image_comparison(["large_arc.svg"], style="mpl20")
def test_large_arc():
    # 创建包含两个子图的图形对象
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # 设置圆弧的中心点坐标和直径
    x = 210
    y = -2115
    diameter = 4261
    # 在每个子图上绘制圆弧
    for ax in [ax1, ax2]:
        # 创建圆弧对象并添加到子图中
        a = Arc((x, y), diameter, diameter, lw=2, color='k')
        ax.add_patch(a)
        # 设置子图不显示坐标轴
        ax.set_axis_off()
        # 设置子图纵横比为相等
        ax.set_aspect('equal')
    # 强制第一个子图显示高精度案例
    ax1.set_xlim(7, 8)
    ax1.set_ylim(5, 6)

    # 强制第二个子图显示低精度案例
    ax2.set_xlim(-25000, 18000)
    ax2.set_ylim(-20000, 6600)


@image_comparison(["all_quadrants_arcs.svg"], style="mpl20")
def test_rotated_arcs():
    # 创建包含4个子图的图形对象
    fig, ax_arr = plt.subplots(2, 2, squeeze=False, figsize=(10, 10))

    scale = 10_000_000
    diag_centers = ((-1, -1), (-1, 1), (1, 1), (1, -1))
    on_axis_centers = ((0, 1), (1, 0), (0, -1), (-1, 0))
    skews = ((2, 2), (2, 1/10), (2,  1/100), (2, 1/1000))

    # 遍历每个子图和相应的偏移参数
    for ax, (sx, sy) in zip(ax_arr.ravel(), skews):
        k = 0
        # 遍历每个子图中的每个圆弧
        for prescale, centers in zip((1 - .0001, (1 - .0001) / np.sqrt(2)),
                                      (on_axis_centers, diag_centers)):
            for j, (x_sign, y_sign) in enumerate(centers, start=k):
                # 创建旋转后的圆弧对象并添加到子图中
                a = Arc(
                    (x_sign * scale * prescale,
                     y_sign * scale * prescale),
                    scale * sx,
                    scale * sy,
                    lw=4,
                    color=f"C{j}",
                    zorder=1 + j,
                    angle=np.rad2deg(np.arctan2(y_sign, x_sign)) % 360,
                    label=f'big {j}',
                    gid=f'big {j}'
                )
                ax.add_patch(a)

            k = j+1
        # 设置子图的坐标限制
        ax.set_xlim(-scale / 4000, scale / 4000)
        ax.set_ylim(-scale / 4000, scale / 4000)
        # 添加水平和垂直参考线
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")
        # 设置子图不显示坐标轴
        ax.set_axis_off()
        # 设置子图纵横比为相等
        ax.set_aspect("equal")


def test_fancyarrow_shape_error():
    # 测试箭头的形状错误情况
    with pytest.raises(ValueError, match="Got unknown shape: 'foo'"):
        FancyArrow(0, 0, 0.2, 0.2, shape='foo')


@pytest.mark.parametrize('fmt, match', (
    ("foo", "Unknown style: 'foo'"),
    ("Round,foo", "Incorrect style argument: 'Round,foo'"),
))
def test_boxstyle_errors(fmt, match):
    # 测试框样式错误情况
    with pytest.raises(ValueError, match=match):
        BoxStyle(fmt)


@image_comparison(baseline_images=['annulus'], extensions=['png'])
def test_annulus():
    # 测试环形图形的创建和绘制
    fig, ax = plt.subplots()
    cir = Annulus((0.5, 0.5), 0.2, 0.05, fc='g')        # 圆形环形
    ell = Annulus((0.5, 0.5), (0.5, 0.3), 0.1, 45,      # 椭圆形环形
                  fc='m', ec='b', alpha=0.5, hatch='xxx')
    ax.add_patch(cir)
    ax.add_patch(ell)
    # 设置图形纵横比为相等
    ax.set_aspect('equal')


@image_comparison(baseline_images=['annulus'], extensions=['png'])
def test_annulus_setters():
    # 测试环形图形设置器的创建和绘制
    fig, ax = plt.subplots()
    cir = Annulus((0., 0.), 0.2, 0.01, fc='g')   # 圆形环形
    ell = Annulus((0., 0.), (1, 2), 0.1, 0,      # 椭圆形环形
                  fc='m', ec='b', alpha=0.5, hatch='xxx')
    ax.add_patch(cir)
    ax.add_patch(ell)
    # 设置图形纵横比为相等
    ax.set_aspect('equal')
    # 设置圆的中心坐标为 (0.5, 0.5)
    cir.center = (0.5, 0.5)
    # 设置圆的半径为 0.2
    cir.radii = 0.2
    # 设置圆的线宽为 0.05
    
    # 设置椭圆的中心坐标为 (0.5, 0.5)
    ell.center = (0.5, 0.5)
    # 设置椭圆的半径为 (长半轴 0.5, 短半轴 0.3)
    ell.radii = (0.5, 0.3)
    # 设置椭圆的线宽为 0.1
    # 设置椭圆的旋转角度为 45 度
    ell.angle = 45
# 使用装饰器将该函数标记为图像比较测试，并指定基线图像和扩展名为PNG
@image_comparison(baseline_images=['annulus'], extensions=['png'])
# 定义测试函数，测试Annulus类的设置方法
def test_annulus_setters2():

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    # 创建圆形环的Annulus对象，指定中心和内外半径，填充颜色为绿色
    cir = Annulus((0., 0.), 0.2, 0.01, fc='g')   # circular annulus

    # 创建椭圆形环的Annulus对象，指定中心、内外半径、角度等参数，并设置颜色、边框颜色、透明度、填充样式
    ell = Annulus((0., 0.), (1, 2), 0.1, 0,
                  fc='m', ec='b', alpha=0.5, hatch='xxx')   # elliptical

    # 将圆形环和椭圆形环添加到坐标轴对象
    ax.add_patch(cir)
    ax.add_patch(ell)

    # 设置坐标轴纵横比为相等
    ax.set_aspect('equal')

    # 修改圆形环的中心点坐标
    cir.center = (0.5, 0.5)

    # 设置圆形环的半径为0.2
    cir.set_semimajor(0.2)
    cir.set_semiminor(0.2)

    # 断言圆形环的半径是否为(0.2, 0.2)
    assert cir.radii == (0.2, 0.2)

    # 修改圆形环的宽度为0.05
    cir.width = 0.05

    # 修改椭圆形环的中心点坐标
    ell.center = (0.5, 0.5)

    # 设置椭圆形环的主半径为0.5，副半径为0.3
    ell.set_semimajor(0.5)
    ell.set_semiminor(0.3)

    # 断言椭圆形环的半径是否为(0.5, 0.3)
    assert ell.radii == (0.5, 0.3)

    # 修改椭圆形环的宽度为0.1
    ell.width = 0.1

    # 设置椭圆形环的角度为45度
    ell.angle = 45


# 定义测试函数，测试多边形的退化情况
def test_degenerate_polygon():
    # 创建一个点的列表
    point = [0, 0]

    # 创建一个包含该点的边界框对象，获取其范围的正确值
    correct_extents = Bbox([point, point]).extents

    # 断言使用该点创建的多边形对象的范围是否与正确值相等
    assert np.all(Polygon([point]).get_extents().extents == correct_extents)


# 使用参数化测试装饰器，测试颜色覆盖时的警告
@pytest.mark.parametrize('kwarg', ('edgecolor', 'facecolor'))
def test_color_override_warning(kwarg):
    # 使用pytest的warns上下文管理器，捕获UserWarning，并检查警告消息内容
    with pytest.warns(UserWarning,
                      match="Setting the 'color' property will override "
                            "the edgecolor or facecolor properties."):
        # 创建Patch对象，指定颜色为黑色，并通过kwarg参数设置边框或填充颜色为黑色
        Patch(color='black', **{kwarg: 'black'})


# 定义测试函数，测试空顶点的多边形
def test_empty_verts():
    # 创建一个空顶点的多边形对象
    poly = Polygon(np.zeros((0, 2)))

    # 断言该多边形对象的顶点列表是否为空
    assert poly.get_verts() == []


# 定义测试函数，测试默认的抗锯齿设置
def test_default_antialiased():
    # 创建一个Patch对象
    patch = Patch()

    # 设置Patch对象的抗锯齿属性为与全局配置的相反值
    patch.set_antialiased(not rcParams['patch.antialiased'])

    # 断言Patch对象的抗锯齿属性是否与设置值相同
    assert patch.get_antialiased() == (not rcParams['patch.antialiased'])

    # 检查设置为None时抗锯齿属性是否被重置为默认状态
    patch.set_antialiased(None)
    assert patch.get_antialiased() == rcParams['patch.antialiased']


# 定义测试函数，测试默认的线条样式设置
def test_default_linestyle():
    # 创建一个Patch对象
    patch = Patch()

    # 设置Patch对象的线条样式为虚线
    patch.set_linestyle('--')

    # 将Patch对象的线条样式设置为None
    patch.set_linestyle(None)

    # 断言Patch对象的线条样式是否被重置为实线
    assert patch.get_linestyle() == 'solid'


# 定义测试函数，测试默认的端点样式设置
def test_default_capstyle():
    # 创建一个Patch对象
    patch = Patch()

    # 断言Patch对象的端点样式是否为默认的平头
    assert patch.get_capstyle() == 'butt'


# 定义测试函数，测试默认的连接样式设置
def test_default_joinstyle():
    # 创建一个Patch对象
    patch = Patch()

    # 断言Patch对象的连接样式是否为默认的斜接
    assert patch.get_joinstyle() == 'miter'


# 使用图像比较测试装饰器，测试自动缩放弧线的情况
@image_comparison(["autoscale_arc"], extensions=['png', 'svg'])
def test_autoscale_arc():
    # 创建包含3个子图的图形对象
    fig, axs = plt.subplots(1, 3, figsize=(4, 1))

    # 创建多组弧线对象，并将它们添加到对应的子图中
    arc_lists = (
        [Arc((0, 0), 1, 1, theta1=0, theta2=90)],
        [Arc((0.5, 0.5), 1.5, 0.5, theta1=10, theta2=20)],
        [Arc((0.5, 0.5), 1.5, 0.5, theta1=10, theta2=20),
         Arc((0.5, 0.5), 2.5, 0.5, theta1=110, theta2=120),
         Arc((0.5, 0.5), 3.5, 0.5, theta1=210, theta2=220),
         Arc((0.5, 0.5), 4.5, 0.5, theta1=310, theta2=320)])

    for ax, arcs in zip(axs, arc_lists):
        for arc in arcs:
            ax.add_patch(arc)  # 将弧线对象添加到子图中
        ax.autoscale()  # 自动缩放子图


# 使用图像比较装饰器和图像检查装饰器，测试弧线集合的情况
@check_figures_equal(extensions=["png", 'svg', 'pdf', 'eps'])
def test_arc_in_collection(fig_test, fig_ref):
    # 创建两个包含相同弧线的图形对象
    arc1 = Arc([.5, .5], .5, 1, theta1=0, theta2=60, angle=20)
    arc2 = Arc([.5, .5], .5, 1, theta1=0, theta2=60, angle=20)
    # 创建一个 PatchCollection 对象 col，包含单个 patch arc2，用于绘制图形边缘
    col = mcollections.PatchCollection(patches=[arc2], facecolors='none', edgecolors='k')
    
    # 在 fig_ref 对象的子图上添加一个 patch 对象 arc1，用于绘制图形
    fig_ref.subplots().add_patch(arc1)
    
    # 在 fig_test 对象的子图上添加之前创建的 PatchCollection 对象 col，用于绘制图形边缘
    fig_test.subplots().add_collection(col)
# 使用装饰器检查图形是否相等，支持的文件扩展名包括 "png", 'svg', 'pdf', 'eps'
@check_figures_equal(extensions=["png", 'svg', 'pdf', 'eps'])
def test_modifying_arc(fig_test, fig_ref):
    # 创建一个弧形对象 arc1，位于中心 [.5, .5]，半径为 .5，起始角度为 0 度，结束角度为 60 度，旋转角度为 20 度
    arc1 = Arc([.5, .5], .5, 1, theta1=0, theta2=60, angle=20)
    # 创建另一个弧形对象 arc2，位于中心 [.5, .5]，半径为 1.5，起始角度为 0 度，结束角度为 60 度，旋转角度为 10 度
    arc2 = Arc([.5, .5], 1.5, 1, theta1=0, theta2=60, angle=10)
    # 将 arc1 添加到参考图形的子图中
    fig_ref.subplots().add_patch(arc1)
    # 将 arc2 添加到测试图形的子图中
    fig_test.subplots().add_patch(arc2)
    # 设置 arc2 的宽度为 .5
    arc2.set_width(.5)
    # 设置 arc2 的旋转角度为 20 度
    arc2.set_angle(20)


def test_arrow_set_data():
    # 创建一个新的图形 fig 和其对应的坐标轴 ax
    fig, ax = plt.subplots()
    # 创建一个箭头对象 arrow，起点坐标 (2, 0)，方向向上，长度为 10
    arrow = mpl.patches.Arrow(2, 0, 0, 10)
    
    # 预期的箭头顶点坐标，通过四舍五入确保精度为小数点后两位
    expected1 = np.array(
       [[1.9,  0.],
        [2.1, -0.],
        [2.1, 8.],
        [2.3, 8.],
        [2., 10.],
        [1.7, 8.],
        [1.9, 8.],
        [1.9, 0.]]
    )
    # 断言箭头的顶点坐标是否与预期值非常接近
    assert np.allclose(expected1, np.round(arrow.get_verts(), 2))
    
    # 预期的箭头顶点坐标，通过四舍五入确保精度为小数点后两位
    expected2 = np.array(
        [[0.39, 0.04],
         [0.61, -0.04],
         [3.01, 6.36],
         [3.24, 6.27],
         [3.5, 8.],
         [2.56, 6.53],
         [2.79, 6.44],
         [0.39, 0.04]]
    )
    # 使用 set_data 方法重新设置箭头的位置、方向、宽度等属性
    arrow.set_data(x=.5, dx=3, dy=8, width=1.2)
    # 断言箭头的顶点坐标是否与预期值非常接近
    assert np.allclose(expected2, np.round(arrow.get_verts(), 2))
```