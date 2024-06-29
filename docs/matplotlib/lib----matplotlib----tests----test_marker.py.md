# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_marker.py`

```py
import numpy as np  # 导入 NumPy 库，用于数值计算
import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块，用于绘图
from matplotlib import markers  # 导入 Matplotlib 的 markers 模块，用于处理标记样式
from matplotlib.path import Path  # 导入 Matplotlib 的 Path 类，用于定义路径
from matplotlib.testing.decorators import check_figures_equal  # 导入 Matplotlib 的测试装饰器，用于比较图形是否相等
from matplotlib.transforms import Affine2D  # 导入 Matplotlib 的 Affine2D 类，用于仿射变换

import pytest  # 导入 Pytest 库，用于编写和运行测试用例


def test_marker_fillstyle():
    # 测试标记样式对象的填充样式
    marker_style = markers.MarkerStyle(marker='o', fillstyle='none')
    assert marker_style.get_fillstyle() == 'none'  # 确保填充样式为 'none'
    assert not marker_style.is_filled()  # 确保标记没有填充


@pytest.mark.parametrize('marker', [
    'o',  # 圆圈标记
    'x',  # X 标记
    '',  # 空字符串标记
    'None',  # 空标记
    r'$\frac{1}{2}$',  # LaTeX 数学公式标记
    "$\u266B$",  # Unicode 符号标记
    1,  # 数字标记（实际上是非法的，应该抛出异常）
    markers.TICKLEFT,  # Matplotlib 预定义的标记样式之一
    [[-1, 0], [1, 0]],  # 自定义路径标记
    np.array([[-1, 0], [1, 0]]),  # NumPy 数组表示的自定义路径标记
    Path([[0, 0], [1, 0]], [Path.MOVETO, Path.LINETO]),  # Matplotlib Path 对象表示的路径标记
    (5, 0),  # 五边形标记
    (7, 1),  # 七角星标记
    (5, 2),  # 星号标记
    (5, 0, 10),  # 旋转了 10 度的五边形标记
    (7, 1, 10),  # 旋转了 10 度的七角星标记
    (5, 2, 10),  # 旋转了 10 度的星号标记
    markers.MarkerStyle('o'),  # 标记样式对象
])
def test_markers_valid(marker):
    # 检查是否能正确创建各种有效的标记样式对象
    markers.MarkerStyle(marker)


@pytest.mark.parametrize('marker', [
    'square',  # 非法的字符串标记
    np.array([[-0.5, 0, 1, 2, 3]]),  # 非法的一维数组标记
    (1,),  # 非法的元组标记
    (5, 3),  # 非法的元组标记，第二个参数必须为 0、1 或 2
    (1, 2, 3, 4),  # 非法的元组标记
])
def test_markers_invalid(marker):
    with pytest.raises(ValueError):  # 确保抛出 ValueError 异常
        markers.MarkerStyle(marker)


class UnsnappedMarkerStyle(markers.MarkerStyle):
    """
    A MarkerStyle where the snap threshold is force-disabled.

    This is used to compare to polygon/star/asterisk markers which do not have
    any snap threshold set.
    """

    def _recache(self):
        super()._recache()
        self._snap_threshold = None  # 禁用自动捕捉阈值


@check_figures_equal()
def test_poly_marker(fig_test, fig_ref):
    ax_test = fig_test.add_subplot()
    ax_ref = fig_ref.add_subplot()

    # 注意，一些参考尺寸必须不同，因为它们具有单位*长度*，而多边形标记则内接于单位*半径*的圆中。
    # 这引入了 np.sqrt(2) 的因子，但由于尺寸是平方的，因此变为 2。
    size = 20**2

    # 方块标记
    ax_test.scatter([0], [0], marker=(4, 0, 45), s=size)
    ax_ref.scatter([0], [0], marker='s', s=size/2)

    # 菱形标记，带和不带旋转参数
    ax_test.scatter([1], [1], marker=(4, 0), s=size)
    ax_ref.scatter([1], [1], marker=UnsnappedMarkerStyle('D'), s=size/2)
    ax_test.scatter([1], [1.5], marker=(4, 0, 0), s=size)
    ax_ref.scatter([1], [1.5], marker=UnsnappedMarkerStyle('D'), s=size/2)

    # 五边形标记，带和不带旋转参数
    ax_test.scatter([2], [2], marker=(5, 0), s=size)
    ax_ref.scatter([2], [2], marker=UnsnappedMarkerStyle('p'), s=size)
    ax_test.scatter([2], [2.5], marker=(5, 0, 0), s=size)
    ax_ref.scatter([2], [2.5], marker=UnsnappedMarkerStyle('p'), s=size)

    # 六边形标记，带和不带旋转参数
    ax_test.scatter([3], [3], marker=(6, 0), s=size)
    ax_ref.scatter([3], [3], marker='h', s=size)
    # 在 ax_test 上绘制一个散点图，位置为 (3, 3.5)，使用自定义标记 (6, 0, 0)，大小为 size
    ax_test.scatter([3], [3.5], marker=(6, 0, 0), s=size)
    # 在 ax_ref 上绘制一个散点图，位置为 (3, 3.5)，使用标准标记 'h'，大小为 size
    ax_ref.scatter([3], [3.5], marker='h', s=size)

    # 在 ax_test 上绘制一个散点图，位置为 (4, 4)，使用自定义标记 (6, 0, 30)，大小为 size
    ax_test.scatter([4], [4], marker=(6, 0, 30), s=size)
    # 在 ax_ref 上绘制一个散点图，位置为 (4, 4)，使用标准标记 'H'，大小为 size
    ax_ref.scatter([4], [4], marker='H', s=size)

    # 在 ax_test 上绘制一个散点图，位置为 (5, 5)，使用自定义标记 (8, 0, 22.5)，大小为 size
    ax_test.scatter([5], [5], marker=(8, 0, 22.5), s=size)
    # 在 ax_ref 上绘制一个散点图，位置为 (5, 5)，使用自定义的未捕捉标记样式 '8'，大小为 size
    ax_ref.scatter([5], [5], marker=UnsnappedMarkerStyle('8'), s=size)

    # 设置 ax_test 的 x 和 y 轴限制范围
    ax_test.set(xlim=(-0.5, 5.5), ylim=(-0.5, 5.5))
    # 设置 ax_ref 的 x 和 y 轴限制范围
    ax_ref.set(xlim=(-0.5, 5.5), ylim=(-0.5, 5.5))
# 定义一个测试函数，用于测试星号标记的绘制效果
def test_star_marker():
    # 我们没有严格等效于这个标记的东西，所以这里只做一个简单的烟雾测试。
    size = 20**2

    # 创建一个图形和一个轴对象
    fig, ax = plt.subplots()
    # 绘制一个使用特定星号标记的散点图，并设置大小为 size
    ax.scatter([0], [0], marker=(5, 1), s=size)
    ax.scatter([1], [1], marker=(5, 1, 0), s=size)
    # 设置轴的界限
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 1.5))


# 星号标记实际上是一个带有内部圆的星形，因此其端点是角，且有轻微的倒角。
# 参考标记只是单线段而没有角，因此没有倒角，我们需要增加一点容差。
@check_figures_equal(tol=1.45)
def test_asterisk_marker(fig_test, fig_ref, request):
    # 在测试和参考图上各添加一个子图
    ax_test = fig_test.add_subplot()
    ax_ref = fig_ref.add_subplot()

    # 由于参考标记使用的是单位长度，而星号标记是内切于单位半径圆内的，所以需要乘以 np.sqrt(2) 的因子，
    # 但由于大小是平方的，所以结果是 2。
    size = 20**2

    # 定义一个函数来绘制参考标记
    def draw_ref_marker(y, style, size):
        # 正如上面所述，每条线段被加倍。由于反锯齿处理，这些加倍的线段在 .png 结果中会略有不同。
        ax_ref.scatter([y], [y], marker=UnsnappedMarkerStyle(style), s=size)
        if request.getfixturevalue('ext') == 'png':
            ax_ref.scatter([y], [y], marker=UnsnappedMarkerStyle(style),
                           s=size)

    # 绘制加号标记
    ax_test.scatter([0], [0], marker=(4, 2), s=size)
    draw_ref_marker(0, '+', size)
    ax_test.scatter([0.5], [0.5], marker=(4, 2, 0), s=size)
    draw_ref_marker(0.5, '+', size)

    # 绘制叉号标记
    ax_test.scatter([1], [1], marker=(4, 2, 45), s=size)
    draw_ref_marker(1, 'x', size/2)

    # 设置测试和参考图的轴界限
    ax_test.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    ax_ref.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))


# 简单数学文本标记不完全是一个圆，因此这不是一个完美的匹配，但它足够接近以确认基于文本的标记正确居中。
# 但我们仍然需要一个小容差来处理这种差异。
@check_figures_equal(extensions=['png'], tol=1.86)
def test_text_marker(fig_ref, fig_test):
    # 在参考和测试图上各添加一个子图
    ax_ref = fig_ref.add_subplot()
    ax_test = fig_test.add_subplot()

    # 在参考图上绘制一个圆形数学文本标记
    ax_ref.plot(0, 0, marker=r'o', markersize=100, markeredgewidth=0)
    # 在测试图上绘制一个圆形数学文本标记
    ax_test.plot(0, 0, marker=r'$\bullet$', markersize=100, markeredgewidth=0)


@check_figures_equal()
def test_marker_clipping(fig_ref, fig_test):
    # 绘制多个标记可能会触发后端中的不同优化路径，因此比较单个标记与多个标记以确保它们被正确裁剪。
    marker_count = len(markers.MarkerStyle.markers)
    marker_size = 50
    ncol = 7
    nrow = marker_count // ncol + 1

    # 根据标记的数量动态设置参考图的大小
    width = 2 * marker_size * ncol
    height = 2 * marker_size * nrow * 2
    fig_ref.set_size_inches((width / fig_ref.dpi, height / fig_ref.dpi))
    # 添加一个坐标轴到参考图
    ax_ref = fig_ref.add_axes([0, 0, 1, 1])
    # 设置测试图的尺寸为给定宽度和高度（以英寸为单位）
    fig_test.set_size_inches((width / fig_test.dpi, height / fig_ref.dpi))
    # 在测试图上添加坐标轴，范围从(0,0)到(1,1)
    ax_test = fig_test.add_axes([0, 0, 1, 1])

    # 遍历所有标记样式，返回索引和标记
    for i, marker in enumerate(markers.MarkerStyle.markers):
        x = i % ncol  # 计算当前标记的 x 坐标位置
        y = i // ncol * 2  # 计算当前标记的 y 坐标位置（每行两个标记）

        # 在参考图上画单个标记线段
        ax_ref.plot([x, x], [y, y + 1], c='k', linestyle='-', lw=3)
        # 在参考图上画单个标记点
        ax_ref.plot(x, y, c='k',
                    marker=marker, markersize=marker_size, markeredgewidth=10,
                    fillstyle='full', markerfacecolor='white')
        # 在参考图上画单个标记点
        ax_ref.plot(x, y + 1, c='k',
                    marker=marker, markersize=marker_size, markeredgewidth=10,
                    fillstyle='full', markerfacecolor='white')

        # 在测试图上画多个标记（线段和点）
        ax_test.plot([x, x], [y, y + 1], c='k', linestyle='-', lw=3,
                     marker=marker, markersize=marker_size, markeredgewidth=10,
                     fillstyle='full', markerfacecolor='white')

    # 设置参考图的坐标轴范围
    ax_ref.set(xlim=(-0.5, ncol), ylim=(-0.5, 2 * nrow))
    # 设置测试图的坐标轴范围
    ax_test.set(xlim=(-0.5, ncol), ylim=(-0.5, 2 * nrow))
    # 关闭参考图的坐标轴
    ax_ref.axis('off')
    # 关闭测试图的坐标轴
    ax_test.axis('off')
# 测试初始化标记并转换为简单的加法操作
def test_marker_init_transforms():
    # 创建一个标记对象，形状为圆圈“o”
    marker = markers.MarkerStyle("o")
    # 创建一个仿射变换对象，平移(1, 1)
    t = Affine2D().translate(1, 1)
    # 使用给定的仿射变换对象初始化标记对象
    t_marker = markers.MarkerStyle("o", transform=t)
    # 断言：标记对象的仿射变换加上 t 等于 t_marker 对象的仿射变换
    assert marker.get_transform() + t == t_marker.get_transform()


# 测试初始化标记并设置连接风格
def test_marker_init_joinstyle():
    # 创建一个星形标记对象
    marker = markers.MarkerStyle("*")
    # 创建一个带有圆形连接风格的星形标记对象
    styled_marker = markers.MarkerStyle("*", joinstyle="round")
    # 断言：带有圆形连接风格的标记对象连接风格应为 "round"
    assert styled_marker.get_joinstyle() == "round"
    # 断言：普通星形标记对象连接风格不应为 "round"
    assert marker.get_joinstyle() != "round"


# 测试初始化标记并设置端点风格
def test_marker_init_captyle():
    # 创建一个星形标记对象
    marker = markers.MarkerStyle("*")
    # 创建一个带有圆形端点风格的星形标记对象
    styled_marker = markers.MarkerStyle("*", capstyle="round")
    # 断言：带有圆形端点风格的标记对象端点风格应为 "round"
    assert styled_marker.get_capstyle() == "round"
    # 断言：普通星形标记对象端点风格不应为 "round"
    assert marker.get_capstyle() != "round"


# 参数化测试：测试标记对象的仿射变换
@pytest.mark.parametrize("marker,transform,expected", [
    # 测试未经过变换的标记对象
    (markers.MarkerStyle("o"), Affine2D().translate(1, 1),
        Affine2D().translate(1, 1)),
    # 测试经过平移变换的标记对象
    (markers.MarkerStyle("o", transform=Affine2D().translate(1, 1)),
        Affine2D().translate(1, 1), Affine2D().translate(2, 2)),
    # 测试经过平移变换的复杂标记对象
    (markers.MarkerStyle("$|||$", transform=Affine2D().translate(1, 1)),
     Affine2D().translate(1, 1), Affine2D().translate(2, 2)),
    # 测试经过平移变换的特定标记对象
    (markers.MarkerStyle(
        markers.TICKLEFT, transform=Affine2D().translate(1, 1)),
        Affine2D().translate(1, 1), Affine2D().translate(2, 2)),
])
def test_marker_transformed(marker, transform, expected):
    # 对标记对象进行仿射变换
    new_marker = marker.transformed(transform)
    # 断言：新的标记对象不应与原标记对象相同
    assert new_marker is not marker
    # 断言：新标记对象的用户定义变换应与期望的变换相同
    assert new_marker.get_user_transform() == expected
    # 断言：原标记对象的用户定义变换不应与新标记对象的用户定义变换相同
    assert marker._user_transform is not new_marker._user_transform


# 测试旋转标记对象时的无效情况
def test_marker_rotated_invalid():
    # 创建一个圆圈形标记对象
    marker = markers.MarkerStyle("o")
    # 断言：尝试旋转标记对象时应引发 ValueError 异常
    with pytest.raises(ValueError):
        new_marker = marker.rotated()
    with pytest.raises(ValueError):
        new_marker = marker.rotated(deg=10, rad=10)


# 参数化测试：测试旋转标记对象
@pytest.mark.parametrize("marker,deg,rad,expected", [
    # 测试以角度旋转标记对象
    (markers.MarkerStyle("o"), 10, None, Affine2D().rotate_deg(10)),
    # 测试以弧度旋转标记对象
    (markers.MarkerStyle("o"), None, 0.01, Affine2D().rotate(0.01)),
    # 测试带有平移变换并以角度旋转标记对象
    (markers.MarkerStyle("o", transform=Affine2D().translate(1, 1)),
        10, None, Affine2D().translate(1, 1).rotate_deg(10)),
    # 测试带有平移变换并以弧度旋转标记对象
    (markers.MarkerStyle("o", transform=Affine2D().translate(1, 1)),
        None, 0.01, Affine2D().translate(1, 1).rotate(0.01)),
    # 测试带有平移变换的特定标记对象并以角度旋转
    (markers.MarkerStyle("$|||$", transform=Affine2D().translate(1, 1)),
      10, None, Affine2D().translate(1, 1).rotate_deg(10)),
    # 测试带有平移变换的特定标记对象并以角度旋转
    (markers.MarkerStyle(
        markers.TICKLEFT, transform=Affine2D().translate(1, 1)),
        10, None, Affine2D().translate(1, 1).rotate_deg(10)),
])
def test_marker_rotated(marker, deg, rad, expected):
    # 对标记对象进行旋转
    new_marker = marker.rotated(deg=deg, rad=rad)
    # 断言：新的标记对象不应与原标记对象相同
    assert new_marker is not marker
    # 断言：新标记对象的用户定义变换应与期望的变换相同
    assert new_marker.get_user_transform() == expected
    # 断言：原标记对象的用户定义变换不应与新标记对象的用户定义变换相同
    assert marker._user_transform is not new_marker._user_transform


# 测试缩放标记对象
def test_marker_scaled():
    # 创建一个数字 "1" 的标记对象
    marker = markers.MarkerStyle("1")
    # 对标记对象进行缩放
    new_marker = marker.scaled(2)
    # 断言：新的标记对象不应与原标记对象相同
    assert new_marker is not marker
    # 断言：检查新标记对象的用户变换是否为二维仿射变换的两倍缩放
    assert new_marker.get_user_transform() == Affine2D().scale(2)
    
    # 断言：检查原标记对象和新标记对象的用户变换对象是否不相同
    assert marker._user_transform is not new_marker._user_transform
    
    # 创建经过二倍横向和三倍纵向缩放的新标记对象
    new_marker = marker.scaled(2, 3)
    
    # 断言：检查新标记对象是否与原标记对象不同
    assert new_marker is not marker
    
    # 断言：检查新标记对象的用户变换是否为二维仿射变换的横向两倍和纵向三倍缩放
    assert new_marker.get_user_transform() == Affine2D().scale(2, 3)
    
    # 断言：检查原标记对象和新标记对象的用户变换对象是否不相同
    assert marker._user_transform is not new_marker._user_transform
    
    # 创建具有指定标记样式和二维仿射变换的原标记对象
    marker = markers.MarkerStyle("1", transform=Affine2D().translate(1, 1))
    
    # 创建经过二倍缩放的新标记对象
    new_marker = marker.scaled(2)
    
    # 断言：检查新标记对象是否与原标记对象不同
    assert new_marker is not marker
    
    # 创建期望的用户变换，先平移再横向二倍缩放
    expected = Affine2D().translate(1, 1).scale(2)
    
    # 断言：检查新标记对象的用户变换是否等于期望的用户变换
    assert new_marker.get_user_transform() == expected
    
    # 断言：检查原标记对象和新标记对象的用户变换对象是否不相同
    assert marker._user_transform is not new_marker._user_transform
# 定义一个名为 test_alt_transform 的函数，用于测试 MarkerStyle 类的 get_alt_transform 方法
def test_alt_transform():
    # 创建一个 MarkerStyle 对象 m1，指定标记样式为圆圈 "o"，方向为 "left"
    m1 = markers.MarkerStyle("o", "left")
    # 创建另一个 MarkerStyle 对象 m2，指定标记样式为圆圈 "o"，方向为 "left"，
    # 并且附带一个顺时针旋转 90 度的仿射变换 Affine2D().rotate_deg(90)
    m2 = markers.MarkerStyle("o", "left", Affine2D().rotate_deg(90))
    # 断言 m1 对象的 get_alt_transform 方法应用顺时针旋转 90 度后，结果应与 m2 对象的 get_alt_transform 方法的结果相等
    assert m1.get_alt_transform().rotate_deg(90) == m2.get_alt_transform()
```