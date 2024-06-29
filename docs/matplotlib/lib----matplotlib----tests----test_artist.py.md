# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_artist.py`

```py
# 导入所需的库
import io
from itertools import chain

import numpy as np

import pytest

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.artist as martist
import matplotlib.backend_bases as mbackend_bases
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison

# 定义测试函数：测试添加到具有不同变换规范的 Axes 中的图形块的行为
def test_patch_transform_of_none():
    # 创建一个新的 Axes 对象
    ax = plt.axes()
    # 设置坐标轴的范围
    ax.set_xlim(1, 3)
    ax.set_ylim(1, 3)

    # 在数据坐标 (2, 2) 处绘制一个椭圆，使用设备坐标来指定
    xy_data = (2, 2)
    xy_pix = ax.transData.transform(xy_data)

    # 不提供 transform 参数将椭圆放置在数据坐标中
    e = mpatches.Ellipse(xy_data, width=1, height=1, fc='yellow', alpha=0.5)
    ax.add_patch(e)
    assert e._transform == ax.transData

    # 提供 transform=None 将椭圆放置在设备坐标中
    e = mpatches.Ellipse(xy_pix, width=120, height=120, fc='coral',
                         transform=None, alpha=0.5)
    assert e.is_transform_set()
    ax.add_patch(e)
    assert isinstance(e._transform, mtransforms.IdentityTransform)

    # 提供 IdentityTransform 将椭圆放置在设备坐标中
    e = mpatches.Ellipse(xy_pix, width=100, height=100,
                         transform=mtransforms.IdentityTransform(), alpha=0.5)
    ax.add_patch(e)
    assert isinstance(e._transform, mtransforms.IdentityTransform)

    # 不提供 transform 参数，并且后续使用 get_transform 也不意味着 is_transform_set
    e = mpatches.Ellipse(xy_pix, width=120, height=120, fc='coral',
                         alpha=0.5)
    intermediate_transform = e.get_transform()
    assert not e.is_transform_set()
    ax.add_patch(e)
    assert e.get_transform() != intermediate_transform
    assert e.is_transform_set()
    assert e._transform == ax.transData


# 定义测试函数：测试添加到具有不同变换规范的 Axes 中的图形集合的行为
def test_collection_transform_of_none():
    # 创建一个新的 Axes 对象
    ax = plt.axes()
    # 设置坐标轴的范围
    ax.set_xlim(1, 3)
    ax.set_ylim(1, 3)

    # 在数据坐标 (2, 2) 处绘制一个椭圆，使用设备坐标来指定
    xy_data = (2, 2)
    xy_pix = ax.transData.transform(xy_data)

    # 不提供 transform 参数将集合放置在数据坐标中
    e = mpatches.Ellipse(xy_data, width=1, height=1)
    c = mcollections.PatchCollection([e], facecolor='yellow', alpha=0.5)
    ax.add_collection(c)
    # 集合应该处于数据坐标中
    assert c.get_offset_transform() + c.get_transform() == ax.transData

    # 提供 transform=None 将集合放置在设备坐标中
    e = mpatches.Ellipse(xy_pix, width=120, height=120)
    # 创建一个 PatchCollection 对象 c，其中包含一个椭圆 e，设置填充色为 'coral'，透明度为 0.5
    c = mcollections.PatchCollection([e], facecolor='coral',
                                     alpha=0.5)
    # 将 c 的坐标变换设置为 None，表示使用默认的坐标变换
    c.set_transform(None)
    # 将 PatchCollection 对象 c 添加到 ax（Axes 对象）中
    ax.add_collection(c)
    # 断言 c 的坐标变换是 IdentityTransform 对象
    assert isinstance(c.get_transform(), mtransforms.IdentityTransform)

    # 创建一个椭圆 e，位于 xy_pix 指定的位置，宽度为 100，高度为 100
    e = mpatches.Ellipse(xy_pix, width=100, height=100)
    # 创建一个 PatchCollection 对象 c，其中包含一个椭圆 e，使用 IdentityTransform 坐标变换，透明度为 0.5
    c = mcollections.PatchCollection([e],
                                     transform=mtransforms.IdentityTransform(),
                                     alpha=0.5)
    # 将 PatchCollection 对象 c 添加到 ax（Axes 对象）中
    ax.add_collection(c)
    # 断言 c 的偏移坐标变换是 IdentityTransform 对象
    assert isinstance(c.get_offset_transform(), mtransforms.IdentityTransform)
@image_comparison(["clip_path_clipping"], remove_text=True)
# 定义测试函数，比较生成图像和预期图像，移除文本后比较
def test_clipping():
    # 创建外部轮廓矩形的路径对象，深拷贝原型，顶点坐标乘以4并减去2
    exterior = mpath.Path.unit_rectangle().deepcopy()
    exterior.vertices *= 4
    exterior.vertices -= 2

    # 创建内部轮廓圆形的路径对象，深拷贝原型，翻转顶点顺序
    interior = mpath.Path.unit_circle().deepcopy()
    interior.vertices = interior.vertices[::-1]

    # 创建复合路径对象作为剪切路径，由外部和内部路径组成
    clip_path = mpath.Path.make_compound_path(exterior, interior)

    # 创建六角星形的路径对象，深拷贝原型，顶点坐标乘以2.6
    star = mpath.Path.unit_regular_star(6).deepcopy()
    star.vertices *= 2.6

    # 创建包含星形路径对象的路径集合对象，设置线宽5，边框颜色蓝色，填充颜色红色，透明度0.7，填充图案为星号
    col = mcollections.PathCollection([star], lw=5, edgecolor='blue',
                                      facecolor='red', alpha=0.7, hatch='*')
    # 设置路径集合对象的剪切路径为预定义的复合路径，转换坐标系为ax1的数据坐标系
    col.set_clip_path(clip_path, ax1.transData)
    # 将路径集合对象添加到ax1坐标系
    ax1.add_collection(col)

    # 创建矩形路径对象，设置线宽5，边框颜色蓝色，填充颜色红色，透明度0.7，填充图案为星号
    patch = mpatches.PathPatch(star, lw=5, edgecolor='blue', facecolor='red',
                               alpha=0.7, hatch='*')
    # 设置路径对象的剪切路径为预定义的复合路径，转换坐标系为ax2的数据坐标系
    patch.set_clip_path(clip_path, ax2.transData)
    # 将路径对象添加到ax2坐标系
    ax2.add_patch(patch)

    # 设置ax1坐标系的x和y轴限制
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])


@check_figures_equal(extensions=['png'])
# 定义检查函数，比较生成图和参考图是否相等，扩展名为PNG
def test_clipping_zoom(fig_test, fig_ref):
    # 将Axes添加到测试图形中，设置其位置和限制，确保剪切路径完全在图形之外不会中断剪切路径
    ax_test = fig_test.add_axes([0, 0, 1, 1])
    l, = ax_test.plot([-3, 3], [-3, 3])
    # 显式使用路径而不是矩形框进行剪切路径处理
    p = mpath.Path([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    p = mpatches.PathPatch(p, transform=ax_test.transData)
    l.set_clip_path(p)

    # 将Axes添加到参考图形中，设置其位置和限制
    ax_ref = fig_ref.add_axes([0, 0, 1, 1])
    ax_ref.plot([-3, 3], [-3, 3])

    # 设置参考图形的x和y轴限制
    ax_ref.set(xlim=(0.5, 0.75), ylim=(0.5, 0.75))
    # 设置测试图形的x和y轴限制
    ax_test.set(xlim=(0.5, 0.75), ylim=(0.5, 0.75))


def test_cull_markers():
    # 创建随机数据点
    x = np.random.random(20000)
    y = np.random.random(20000)

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制散点图
    ax.plot(x, y, 'k.')
    # 设置x轴限制
    ax.set_xlim(2, 3)

    # 创建内存中的PDF流
    pdf = io.BytesIO()
    # 将图形保存为PDF格式
    fig.savefig(pdf, format="pdf")
    # 断言PDF内容长度小于8000字节
    assert len(pdf.getvalue()) < 8000

    # 创建内存中的SVG流
    svg = io.BytesIO()
    # 将图形保存为SVG格式
    fig.savefig(svg, format="svg")
    # 断言SVG内容长度小于20000字节
    assert len(svg.getvalue()) < 20000


@image_comparison(['hatching'], remove_text=True, style='default')
# 定义测试函数，比较生成图和预期图是否相等，移除文本后比较，使用默认样式
def test_hatching():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots(1, 1)

    # 创建带有默认填充图案的矩形路径对象
    rect1 = mpatches.Rectangle((0, 0), 3, 4, hatch='/')
    # 将矩形路径对象添加到坐标轴
    ax.add_patch(rect1)

    # 创建带有默认填充图案的正多边形集合对象
    rect2 = mcollections.RegularPolyCollection(
        4, sizes=[16000], offsets=[(1.5, 6.5)], offset_transform=ax.transData,
        hatch='/')
    # 将正多边形集合对象添加到坐标轴
    ax.add_collection(rect2)

    # 创建带有指定边框颜色但默认填充图案的矩形路径对象
    rect3 = mpatches.Rectangle((4, 0), 3, 4, hatch='/', edgecolor='C1')
    # 将矩形路径对象添加到坐标轴
    ax.add_patch(rect3)

    # 创建带有指定边框颜色但默认填充图案的正多边形集合对象
    rect4 = mcollections.RegularPolyCollection(
        4, sizes=[16000], offsets=[(5.5, 6.5)], offset_transform=ax.transData,
        hatch='/', edgecolor='C1')
    # 将正多边形集合对象添加到坐标轴
    ax.add_collection(rect4)

    # 设置坐标轴的x和y轴限制
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)


def test_remove():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制imshow图像，显示36个元素的数组
    im = ax.imshow(np.arange(36).reshape(6, 6))
    # 创建一条新的折线，并将其赋值给变量 ln
    ln, = ax.plot(range(5))

    # 断言图形 fig 和轴 ax 都处于“stale”（过时）状态
    assert fig.stale
    assert ax.stale

    # 绘制画布，更新图形和轴的状态
    fig.canvas.draw()
    # 断言图形 fig 和轴 ax 不再处于“stale”状态
    assert not fig.stale
    assert not ax.stale
    # 断言新创建的折线 ln 也不再处于“stale”状态
    assert not ln.stale

    # 断言图像对象 im 存在于轴 ax 的鼠标悬停集合中
    assert im in ax._mouseover_set
    # 断言折线 ln 不在轴 ax 的鼠标悬停集合中
    assert ln not in ax._mouseover_set
    # 断言图像对象 im 的轴属性是 ax
    assert im.axes is ax

    # 移除图像对象 im 和折线 ln
    im.remove()
    ln.remove()

    # 遍历图像对象列表 [im, ln]
    for art in [im, ln]:
        # 断言图像对象 art 的轴属性为 None
        assert art.axes is None
        # 断言图像对象 art 的图形属性为 None
        assert art.figure is None

    # 断言图像对象 im 不再在轴 ax 的鼠标悬停集合中
    assert im not in ax._mouseover_set
    # 断言图形 fig 和轴 ax 再次处于“stale”状态
    assert fig.stale
    assert ax.stale
@image_comparison(["default_edges.png"], remove_text=True, style='default')
def test_default_edges():
    # 当这张测试图片重新生成时，请移除此行。
    # 设置全局配置以增加文本字距系数
    plt.rcParams['text.kerning_factor'] = 6

    # 创建一个包含4个子图的图形对象，并获取各个子图对象
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)

    # 在第一个子图上绘制两个数据集的散点图
    ax1.plot(np.arange(10), np.arange(10), 'x',
             np.arange(10) + 1, np.arange(10), 'o')
    
    # 在第二个子图上绘制柱状图，条边对齐
    ax2.bar(np.arange(10), np.arange(10), align='edge')
    
    # 在第三个子图上添加一个文本框，并设置坐标轴范围
    ax3.text(0, 0, "BOX", size=24, bbox=dict(boxstyle='sawtooth'))
    ax3.set_xlim((-1, 1))
    ax3.set_ylim((-1, 1))
    
    # 在第四个子图上绘制一个路径补丁对象，并添加到子图中
    pp1 = mpatches.PathPatch(
        mpath.Path([(0, 0), (1, 0), (1, 1), (0, 0)],
                   [mpath.Path.MOVETO, mpath.Path.CURVE3,
                    mpath.Path.CURVE3, mpath.Path.CLOSEPOLY]),
        fc="none", transform=ax4.transData)
    ax4.add_patch(pp1)


def test_properties():
    # 创建一个空的线条对象，并检查是否有警告被发出
    ln = mlines.Line2D([], [])
    ln.properties()  # Check that no warning is emitted.


def test_setp():
    # 检查空列表的设置
    plt.setp([])
    plt.setp([[]])

    # 检查任意可迭代对象的设置
    fig, ax = plt.subplots()
    lines1 = ax.plot(range(3))
    lines2 = ax.plot(range(3))
    # 设置多个线条对象的线宽为5
    martist.setp(chain(lines1, lines2), 'lw', 5)
    # 设置坐标轴边框线的颜色为绿色
    plt.setp(ax.spines.values(), color='green')

    # 检查 *file* 参数的设置
    sio = io.StringIO()
    plt.setp(lines1, 'zorder', file=sio)
    assert sio.getvalue() == '  zorder: float\n'


def test_None_zorder():
    fig, ax = plt.subplots()
    # 创建一个线条对象，并设置 zorder 为 None
    ln, = ax.plot(range(5), zorder=None)
    # 断言获取的 zorder 属性值等于 Line2D 类的 zorder 属性值
    assert ln.get_zorder() == mlines.Line2D.zorder
    # 设置新的 zorder 值，并进行断言检查
    ln.set_zorder(123456)
    assert ln.get_zorder() == 123456
    # 将 zorder 值设为 None，并进行断言检查
    ln.set_zorder(None)
    assert ln.get_zorder() == mlines.Line2D.zorder


@pytest.mark.parametrize('accept_clause, expected', [
    ('', 'unknown'),
    ("ACCEPTS: [ '-' | '--' | '-.' ]", "[ '-' | '--' | '-.' ]"),
    ('ACCEPTS: Some description.', 'Some description.'),
    ('.. ACCEPTS: Some description.', 'Some description.'),
    ('arg : int', 'int'),
    ('*arg : int', 'int'),
    ('arg : int\nACCEPTS: Something else.', 'Something else. '),
])
def test_artist_inspector_get_valid_values(accept_clause, expected):
    # 定义一个测试用的艺术家类，设置一个带有参数描述的方法文档字符串
    class TestArtist(martist.Artist):
        def set_f(self, arg):
            pass

    TestArtist.set_f.__doc__ = """
    Some text.

    %s
    """ % accept_clause
    # 使用艺术家检查器获取参数的有效值，并进行断言检查
    valid_values = martist.ArtistInspector(TestArtist).get_valid_values('f')
    assert valid_values == expected


def test_artist_inspector_get_aliases():
    # 测试 get_aliases 方法的返回格式和类型是否正确
    ai = martist.ArtistInspector(mlines.Line2D)
    aliases = ai.get_aliases()
    assert aliases["linewidth"] == {"lw"}


def test_set_alpha():
    # 创建一个艺术家对象，并使用 pytest 检查设置 alpha 参数时的异常情况
    art = martist.Artist()
    with pytest.raises(TypeError, match='^alpha must be numeric or None'):
        art.set_alpha('string')
    with pytest.raises(TypeError, match='^alpha must be numeric or None'):
        art.set_alpha([1, 2, 3])
    with pytest.raises(ValueError, match="outside 0-1 range"):
        art.set_alpha(1.1)
    # 使用 pytest 的上下文管理器来检测是否抛出指定类型的异常(ValueError)，并且异常消息要匹配给定的字符串 "outside 0-1 range"
    with pytest.raises(ValueError, match="outside 0-1 range"):
        # 调用 art 对象的 set_alpha 方法，并传入 np.nan 作为参数
        art.set_alpha(np.nan)
def test_set_alpha_for_array():
    # 创建一个Artist对象实例
    art = martist.Artist()
    
    # 测试设置数组的alpha值，预期会引发TypeError异常，匹配错误消息'^alpha must be numeric or None'
    with pytest.raises(TypeError, match='^alpha must be numeric or None'):
        art._set_alpha_for_array('string')
    
    # 测试设置数组的alpha值，预期会引发ValueError异常，匹配错误消息"outside 0-1 range"
    with pytest.raises(ValueError, match="outside 0-1 range"):
        art._set_alpha_for_array(1.1)
    
    # 测试设置数组的alpha值，预期会引发ValueError异常，因为值为np.nan，匹配错误消息"outside 0-1 range"
    with pytest.raises(ValueError, match="outside 0-1 range"):
        art._set_alpha_for_array(np.nan)
    
    # 测试设置数组的alpha值，预期会引发ValueError异常，因为数组包含超出0-1范围的值，匹配错误消息"alpha must be between 0 and 1"
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        art._set_alpha_for_array([0.5, 1.1])
    
    # 测试设置数组的alpha值，预期会引发ValueError异常，因为数组包含np.nan值，匹配错误消息"alpha must be between 0 and 1"
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        art._set_alpha_for_array([0.5, np.nan])


def test_callbacks():
    # 定义一个回调函数func，用于测试回调功能
    def func(artist):
        func.counter += 1

    # 设置func函数的计数器属性为0
    func.counter = 0

    # 创建一个Artist对象实例
    art = martist.Artist()
    
    # 添加回调函数func到Artist对象实例
    oid = art.add_callback(func)
    
    # 断言回调函数func的计数器为0
    assert func.counter == 0
    
    # 调用pchanged方法，应该触发回调函数
    art.pchanged()
    
    # 断言回调函数func的计数器为1
    assert func.counter == 1
    
    # 调用set_zorder方法，设置属性应该再次触发回调函数
    art.set_zorder(10)
    
    # 断言回调函数func的计数器为2
    assert func.counter == 2
    
    # 移除之前添加的回调函数
    art.remove_callback(oid)
    
    # 再次调用pchanged方法，不应该触发回调函数了
    art.pchanged()
    
    # 断言回调函数func的计数器仍为2
    assert func.counter == 2


def test_set_signature():
    """测试Artist子类自动生成的``set()``方法。"""
    # 定义一个名为MyArtist1的Artist子类
    class MyArtist1(martist.Artist):
        # 定义一个自定义的set_myparam1方法
        def set_myparam1(self, val):
            pass

    # 断言MyArtist1类有属性set，并且有'_autogenerated_signature'属性
    assert hasattr(MyArtist1.set, '_autogenerated_signature')
    
    # 断言MyArtist1类的set方法文档字符串中包含'myparam1'
    assert 'myparam1' in MyArtist1.set.__doc__

    # 定义一个名为MyArtist2的MyArtist1子类
    class MyArtist2(MyArtist1):
        # 定义一个自定义的set_myparam2方法
        def set_myparam2(self, val):
            pass

    # 断言MyArtist2类有属性set，并且有'_autogenerated_signature'属性
    assert hasattr(MyArtist2.set, '_autogenerated_signature')
    
    # 断言MyArtist2类的set方法文档字符串中包含'myparam1'和'myparam2'
    assert 'myparam1' in MyArtist2.set.__doc__
    assert 'myparam2' in MyArtist2.set.__doc__


def test_set_is_overwritten():
    """测试Artist子类中定义的set()方法不应该被覆盖。"""
    # 定义一个名为MyArtist3的Artist子类
    class MyArtist3(martist.Artist):
        # 定义一个自定义的set方法，文档字符串为"Not overwritten."
        def set(self, **kwargs):
            """Not overwritten."""

    # 断言MyArtist3类的set方法没有'_autogenerated_signature'属性
    assert not hasattr(MyArtist3.set, '_autogenerated_signature')
    
    # 断言MyArtist3类的set方法的文档字符串为"Not overwritten."
    assert MyArtist3.set.__doc__ == "Not overwritten."

    # 定义一个名为MyArtist4的MyArtist3子类
    class MyArtist4(MyArtist3):
        pass

    # 断言MyArtist4类的set方法与MyArtist3类的set方法是同一个对象
    assert MyArtist4.set is MyArtist3.set


def test_format_cursor_data_BoundaryNorm():
    """测试使用BoundaryNorm时的光标数据格式是否正确。"""
    # 创建一个3x3的空numpy数组X
    X = np.empty((3, 3))
    X[0, 0] = 0.9
    X[0, 1] = 0.99
    X[0, 2] = 0.999
    X[1, 0] = -1
    X[1, 1] = 0
    X[1, 2] = 1
    X[2, 0] = 0.09
    X[2, 1] = 0.009
    X[2, 2] = 0.0009

    # 将范围-1到1映射到0到256，步长为0.1
    fig, ax = plt.subplots()
    fig.suptitle("-1..1 to 0..256 in 0.1")
    
    # 创建一个BoundaryNorm对象norm
    norm = mcolors.BoundaryNorm(np.linspace(-1, 1, 20), 256)
    
    # 使用imshow方法在Axes上绘制图像
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)

    # 预期的标签列表
    labels_list = [
        "[0.9]",
        "[1.]",
        "[1.]",
        "[-1.0]",
        "[0.0]",
        "[1.0]",
        "[0.09]",
        "[0.009]",
        "[0.0009]",
    ]
    
    # 遍历X中的每个值v和标签列表中的每个标签label
    for v, label in zip(X.flat, labels_list):
        # 断言调用img的format_cursor_data方法返回的数据等于预期的标签
        assert img.format_cursor_data(v) == label

    # 关闭绘图对象
    plt.close()
    # 创建一个新的图形和子图对象
    fig, ax = plt.subplots()
    # 设置图形的总标题
    fig.suptitle("-1..1 to 0..256 in 0.01")
    # 获取名为 'RdBu_r' 的颜色映射对象，并对其进行重新采样，得到包含200个颜色的颜色映射对象
    cmap = mpl.colormaps['RdBu_r'].resampled(200)
    # 创建一个边界规范对象，用于映射从-1到1的值到0到256的颜色范围，共200个分段
    norm = mcolors.BoundaryNorm(np.linspace(-1, 1, 200), 200)
    # 在子图上绘制图像，使用给定的颜色映射和归一化规范
    img = ax.imshow(X, cmap=cmap, norm=norm)

    # 定义标签列表，用于断言每个像素值的格式化输出是否正确
    labels_list = [
        "[0.90]",
        "[0.99]",
        "[1.0]",
        "[-1.00]",
        "[0.00]",
        "[1.00]",
        "[0.09]",
        "[0.009]",
        "[0.0009]",
    ]
    # 遍历图像中每个像素值及其对应的标签，确保断言通过
    for v, label in zip(X.flat, labels_list):
        # 检查每个像素值的格式化输出是否与预期的标签一致
        assert img.format_cursor_data(v) == label

    # 关闭当前图形对象，清空图形窗口
    plt.close()

    # 创建一个新的图形和子图对象
    fig, ax = plt.subplots()
    # 设置图形的总标题
    fig.suptitle("-1..1 to 0..256 in 0.001")
    # 获取名为 'RdBu_r' 的颜色映射对象，并对其进行重新采样，得到包含2000个颜色的颜色映射对象
    cmap = mpl.colormaps['RdBu_r'].resampled(2000)
    # 创建一个边界规范对象，用于映射从-1到1的值到0到256的颜色范围，共2000个分段
    norm = mcolors.BoundaryNorm(np.linspace(-1, 1, 2000), 2000)
    # 在子图上绘制图像，使用给定的颜色映射和归一化规范
    img = ax.imshow(X, cmap=cmap, norm=norm)

    # 定义标签列表，用于断言每个像素值的格式化输出是否正确
    labels_list = [
        "[0.900]",
        "[0.990]",
        "[0.999]",
        "[-1.000]",
        "[0.000]",
        "[1.000]",
        "[0.090]",
        "[0.009]",
        "[0.0009]",
    ]
    # 遍历图像中每个像素值及其对应的标签，确保断言通过
    for v, label in zip(X.flat, labels_list):
        # 检查每个像素值的格式化输出是否与预期的标签一致
        assert img.format_cursor_data(v) == label

    # 关闭当前图形对象，清空图形窗口
    plt.close()

    # 创建一个新的数组 X，包含7行1列的空数组
    X = np.empty((7, 1))
    # 设置数组 X 中的值
    X[0] = -1.0
    X[1] = 0.0
    X[2] = 0.1
    X[3] = 0.5
    X[4] = 0.9
    X[5] = 1.0
    X[6] = 2.0

    # 定义标签列表，用于断言每个像素值的格式化输出是否正确
    labels_list = [
        "[-1.0]",
        "[0.0]",
        "[0.1]",
        "[0.5]",
        "[0.9]",
        "[1.0]",
        "[2.0]",
    ]

    # 创建一个新的图形和子图对象
    fig, ax = plt.subplots()
    # 设置图形的总标题
    fig.suptitle("noclip, neither")
    # 创建一个边界规范对象，将0到1的范围映射到256个颜色，不进行剪裁，也不进行扩展
    norm = mcolors.BoundaryNorm(
        np.linspace(0, 1, 4, endpoint=True), 256, clip=False, extend='neither')
    # 在子图上绘制图像，使用 'RdBu_r' 的颜色映射和定义好的归一化规范
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    # 遍历图像中每个像素值及其对应的标签，确保断言通过
    for v, label in zip(X.flat, labels_list):
        # 检查每个像素值的格式化输出是否与预期的标签一致
        assert img.format_cursor_data(v) == label

    # 关闭当前图形对象，清空图形窗口
    plt.close()

    # 创建一个新的图形和子图对象
    fig, ax = plt.subplots()
    # 设置图形的总标题
    fig.suptitle("noclip, min")
    # 创建一个边界规范对象，将0到1的范围映射到256个颜色，不进行剪裁，扩展到最小值
    norm = mcolors.BoundaryNorm(
        np.linspace(0, 1, 4, endpoint=True), 256, clip=False, extend='min')
    # 在子图上绘制图像，使用 'RdBu_r' 的颜色映射和定义好的归一化规范
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    # 遍历图像中每个像素值及其对应的标签，确保断言通过
    for v, label in zip(X.flat, labels_list):
        # 检查每个像素值的格式化输出是否与预期的标签一致
        assert img.format_cursor_data(v) == label

    # 关闭当前图形对象，清空图形窗口
    plt.close()

    # 创建一个新的图形和子图对象
    fig, ax = plt.subplots()
    # 设置图形的总标题
    fig.suptitle("noclip, max")
    # 创建一个边界规范对象，将0到1的范围映射到256个颜色，不进行剪裁，扩展到最大值
    norm = mcolors.BoundaryNorm(
        np.linspace(0, 1, 4, endpoint=True), 256, clip=False, extend='max')
    # 在子图上绘制图像，使用 'RdBu_r' 的颜色映射和定义好的归一化规范
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    # 遍历图像中每个像素值及其对应的标签，确保断言通过
    for v, label in zip(X.flat, labels_list):
        # 检查每个像素值的格式化输出是否与预期的标签一致
        assert img.format_cursor_data(v) == label

    # 关闭当前图形对象，清空图形窗口
    plt.close()

    # 创建一个新的图形和子图对象
    fig, ax = plt.subplots()
    # 设置图形的总标题
    fig.suptitle("noclip, both")
    # 创建一个边界归一化对象，将值范围 [0, 1] 等分为 4 个区间，用于颜色映射的归一化
    norm = mcolors.BoundaryNorm(
        np.linspace(0, 1, 4, endpoint=True), 256, clip=False, extend='both')
    # 使用归一化对象将二维数组 X 显示为图像，并指定颜色映射为 'RdBu_r'
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    # 对 X 扁平化后的每个值 v，与 labels_list 中的标签一一对应进行断言验证
    for v, label in zip(X.flat, labels_list):
        # 检查图像对象 img 是否正确格式化光标数据 v，并与预期标签 label 相符
        assert img.format_cursor_data(v) == label

    # 关闭当前图形
    plt.close()

    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 设置图形的总标题为 "clip, neither"
    fig.suptitle("clip, neither")
    # 创建另一个边界归一化对象，将值范围 [0, 1] 等分为 4 个区间，且开启剪裁功能，扩展模式为 'neither'
    norm = mcolors.BoundaryNorm(
        np.linspace(0, 1, 4, endpoint=True), 256, clip=True, extend='neither')
    # 使用新的归一化对象将二维数组 X 显示为图像，并指定颜色映射为 'RdBu_r'
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    # 对 X 扁平化后的每个值 v，与 labels_list 中的标签一一对应进行断言验证
    for v, label in zip(X.flat, labels_list):
        # 检查图像对象 img 是否正确格式化光标数据 v，并与预期标签 label 相符
        assert img.format_cursor_data(v) == label

    # 关闭当前图形
    plt.close()
def test_auto_no_rasterize():
    # 定义一个名为 Gen1 的类，继承自 martist.Artist
    class Gen1(martist.Artist):
        ...

    # 断言 Gen1 类的 __dict__ 中包含 'draw' 属性
    assert 'draw' in Gen1.__dict__
    # 断言 Gen1 类的 'draw' 属性与 Gen1.draw 相同
    assert Gen1.__dict__['draw'] is Gen1.draw

    # 定义一个名为 Gen2 的类，继承自 Gen1 类
    class Gen2(Gen1):
        ...

    # 断言 Gen2 类的 __dict__ 中不包含 'draw' 属性
    assert 'draw' not in Gen2.__dict__
    # 断言 Gen2 类的 'draw' 方法与 Gen1 类的 'draw' 方法相同
    assert Gen2.draw is Gen1.draw


def test_draw_wraper_forward_input():
    # 定义一个名为 TestKlass 的类，继承自 martist.Artist
    class TestKlass(martist.Artist):
        # 定义 draw 方法，接受 renderer 和 extra 两个参数，返回 extra
        def draw(self, renderer, extra):
            return extra

    # 创建 TestKlass 类的实例对象 art
    art = TestKlass()
    # 创建 mbackend_bases.RendererBase 类的实例对象 renderer
    renderer = mbackend_bases.RendererBase()

    # 断言调用 art 对象的 draw 方法，传入 renderer 和 'aardvark' 参数，返回 'aardvark'
    assert 'aardvark' == art.draw(renderer, 'aardvark')
    # 断言调用 art 对象的 draw 方法，传入 renderer 参数和命名参数 extra='aardvark'，返回 'aardvark'
    assert 'aardvark' == art.draw(renderer, extra='aardvark')
```