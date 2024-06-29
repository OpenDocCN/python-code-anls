# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_colors.py`

```
# 导入必要的模块和库
import copy  # 导入 copy 模块，用于复制对象
import itertools  # 导入 itertools 模块，用于迭代工具函数
import unittest.mock  # 导入 unittest.mock 模块，用于单元测试的模拟对象

from io import BytesIO  # 从 io 模块中导入 BytesIO 类，用于操作二进制数据的内存缓冲区
import numpy as np  # 导入 NumPy 库，用于科学计算
from PIL import Image  # 从 PIL 库中导入 Image 模块，用于图像处理
import pytest  # 导入 pytest 库，用于编写和运行测试
import base64  # 导入 base64 模块，用于编码解码 base64 数据

from numpy.testing import assert_array_equal, assert_array_almost_equal  # 从 numpy.testing 模块中导入数组相等性的断言函数

from matplotlib import cbook, cm  # 导入 matplotlib 库中的 cbook 和 cm 模块
import matplotlib  # 导入 matplotlib 库
import matplotlib as mpl  # 导入 matplotlib 库并使用 mpl 别名
import matplotlib.colors as mcolors  # 导入 matplotlib 库中的 colors 模块，用于颜色处理
import matplotlib.colorbar as mcolorbar  # 导入 matplotlib 库中的 colorbar 模块，用于颜色条
import matplotlib.pyplot as plt  # 导入 matplotlib 库中的 pyplot 模块，并使用 plt 别名
import matplotlib.scale as mscale  # 导入 matplotlib 库中的 scale 模块，用于比例尺
from matplotlib.rcsetup import cycler  # 从 matplotlib.rcsetup 模块中导入 cycler 对象
from matplotlib.testing.decorators import image_comparison, check_figures_equal  # 导入测试装饰器
from matplotlib.colors import is_color_like, to_rgba_array  # 导入颜色处理相关函数

@pytest.mark.parametrize('N, result', [
    (5, [1, .6, .2, .1, 0]),
    (2, [1, 0]),
    (1, [0]),
])
def test_create_lookup_table(N, result):
    # 定义测试数据
    data = [(0.0, 1.0, 1.0), (0.5, 0.2, 0.2), (1.0, 0.0, 0.0)]
    # 断言 _create_lookup_table 方法的返回结果与预期结果相等
    assert_array_almost_equal(mcolors._create_lookup_table(N, data), result)


@pytest.mark.parametrize("dtype", [np.uint8, int, np.float16, float])
def test_index_dtype(dtype):
    # 使用 uint8 进行索引操作，验证其有效性
    cm = mpl.colormaps["viridis"]
    assert_array_equal(cm(dtype(0)), cm(0))


def test_resampled():
    """
    GitHub issue #6025 pointed to incorrect ListedColormap.resampled;
    here we test the method for LinearSegmentedColormap as well.
    """
    n = 101
    # 创建颜色列表
    colorlist = np.empty((n, 4), float)
    colorlist[:, 0] = np.linspace(0, 1, n)
    colorlist[:, 1] = 0.2
    colorlist[:, 2] = np.linspace(1, 0, n)
    colorlist[:, 3] = 0.7
    # 使用 colorlist 创建 LinearSegmentedColormap 对象
    lsc = mcolors.LinearSegmentedColormap.from_list('lsc', colorlist)
    # 使用 colorlist 创建 ListedColormap 对象
    lc = mcolors.ListedColormap(colorlist)
    # 设置一些错误值进行测试
    for cmap in [lsc, lc]:
        cmap.set_under('r')
        cmap.set_over('g')
        cmap.set_bad('b')
    # 对 LinearSegmentedColormap 和 ListedColormap 进行重新采样
    lsc3 = lsc.resampled(3)
    lc3 = lc.resampled(3)
    # 预期的重新采样结果
    expected = np.array([[0.0, 0.2, 1.0, 0.7],
                         [0.5, 0.2, 0.5, 0.7],
                         [1.0, 0.2, 0.0, 0.7]], float)
    # 断言重新采样后的结果与预期结果相等
    assert_array_almost_equal(lsc3([0, 0.5, 1]), expected)
    assert_array_almost_equal(lc3([0, 0.5, 1]), expected)
    # 检查 over/under/bad 值是否正确复制
    assert_array_almost_equal(lsc(np.inf), lsc3(np.inf))
    assert_array_almost_equal(lsc(-np.inf), lsc3(-np.inf))
    assert_array_almost_equal(lsc(np.nan), lsc3(np.nan))
    assert_array_almost_equal(lc(np.inf), lc3(np.inf))
    assert_array_almost_equal(lc(-np.inf), lc3(-np.inf))
    assert_array_almost_equal(lc(np.nan), lc3(np.nan))


def test_colormaps_get_cmap():
    cr = mpl.colormaps

    # 检查字符串和 Colormap 是否相等
    assert cr.get_cmap('plasma') == cr["plasma"]
    assert cr.get_cmap(cr["magma"]) == cr["magma"]

    # 检查默认值
    assert cr.get_cmap(None) == cr[mpl.rcParams['image.cmap']]

    # 检查无效名称时是否引发 ValueError
    bad_cmap = 'AardvarksAreAwkward'
    with pytest.raises(ValueError, match=bad_cmap):
        cr.get_cmap(bad_cmap)

    # 检查错误类型时是否引发 TypeError
    # 使用 pytest 框架中的 `raises` 方法，验证下面的代码块是否会引发指定的异常类型(TypeError)，
    # 并且异常的错误信息需要匹配给定的字符串('object')
    with pytest.raises(TypeError, match='object'):
        # 调用 cr 对象的 get_cmap 方法，并传入一个空对象作为参数，期望引发异常
        cr.get_cmap(object())
# 测试重新注册内置颜色映射是否引发值错误异常
def test_double_register_builtin_cmap():
    # 定义颜色映射名称
    name = "viridis"
    # 定义异常匹配字符串，用于验证异常信息
    match = f"Re-registering the builtin cmap {name!r}."
    # 使用 pytest 检查是否抛出值错误异常，并验证异常信息
    with pytest.raises(ValueError, match=match):
        matplotlib.colormaps.register(mpl.colormaps[name], name=name, force=True)


# 测试颜色映射对象的复制和操作
def test_colormap_copy():
    # 获取预定义的颜色映射对象
    cmap = plt.cm.Reds
    # 复制颜色映射对象
    copied_cmap = copy.copy(cmap)
    # 在忽略无效值的情况下，对复制的颜色映射对象进行操作
    with np.errstate(invalid='ignore'):
        ret1 = copied_cmap([-1, 0, .5, 1, np.nan, np.inf])
    # 再次复制颜色映射对象
    cmap2 = copy.copy(copied_cmap)
    # 修改第二个复制对象的无效值表示
    cmap2.set_bad('g')
    # 在忽略无效值的情况下，再次对第一个复制的颜色映射对象进行操作
    with np.errstate(invalid='ignore'):
        ret2 = copied_cmap([-1, 0, .5, 1, np.nan, np.inf])
    # 断言两次操作的结果相等
    assert_array_equal(ret1, ret2)
    # 使用 .copy 方法再次复制颜色映射对象
    cmap = plt.cm.Reds
    copied_cmap = cmap.copy()
    # 在忽略无效值的情况下，对复制的颜色映射对象进行操作
    with np.errstate(invalid='ignore'):
        ret1 = copied_cmap([-1, 0, .5, 1, np.nan, np.inf])
    # 再次复制颜色映射对象
    cmap2 = copy.copy(copied_cmap)
    # 修改第二个复制对象的无效值表示
    cmap2.set_bad('g')
    # 在忽略无效值的情况下，再次对第一个复制的颜色映射对象进行操作
    with np.errstate(invalid='ignore'):
        ret2 = copied_cmap([-1, 0, .5, 1, np.nan, np.inf])
    # 断言两次操作的结果相等
    assert_array_equal(ret1, ret2)


# 测试颜色映射对象的相等性和修改
def test_colormap_equals():
    # 获取预定义的颜色映射对象
    cmap = mpl.colormaps["plasma"]
    # 复制颜色映射对象
    cm_copy = cmap.copy()
    # 断言复制后的对象不是同一个原始对象
    assert cm_copy is not cmap
    # 断言复制后的对象数据与原始对象相等
    assert cm_copy == cmap
    # 修改复制对象的无效值表示
    cm_copy.set_bad('y')
    # 断言修改后的对象与原始对象不相等
    assert cm_copy != cmap
    # 缩小复制对象的查找表，确保即使大小不同也能比较相等性
    cm_copy._lut = cm_copy._lut[:10, :]
    assert cm_copy != cmap
    # 测试不同名称但查找表相同的颜色映射对象是否相等
    cm_copy = cmap.copy()
    cm_copy.name = "Test"
    assert cm_copy == cmap
    # 测试颜色条的扩展属性是否影响对象的相等性
    cm_copy = cmap.copy()
    cm_copy.colorbar_extend = not cmap.colorbar_extend
    assert cm_copy != cmap


# 测试颜色映射对象在非本机字节顺序数组输入情况下的映射
def test_colormap_endian():
    """
    GitHub issue #1005: a bug in putmask caused erroneous
    mapping of 1.0 when input from a non-native-byteorder
    array.
    """
    # 获取预定义的颜色映射对象
    cmap = mpl.colormaps["jet"]
    # 定义测试数据数组，包含各种无效值
    a = [-0.5, 0, 0.5, 1, 1.5, np.nan]
    # 遍历不同数据类型的测试数据
    for dt in ["f2", "f4", "f8"]:
        # 创建本机字节顺序和非本机字节顺序的数组，并处理无效值
        anative = np.ma.masked_invalid(np.array(a, dtype=dt))
        aforeign = anative.byteswap().view(anative.dtype.newbyteorder())
        # 断言两种不同字节顺序的数组经颜色映射后结果相同
        assert_array_equal(cmap(anative), cmap(aforeign))


# 测试颜色映射对象对无效值的处理
def test_colormap_invalid():
    """
    GitHub issue #9892: Handling of nan's were getting mapped to under
    rather than bad. This tests to make sure all invalid values
    (-inf, nan, inf) are mapped respectively to (under, bad, over).
    """
    # 获取预定义的颜色映射对象
    cmap = mpl.colormaps["plasma"]
    # 定义包含各种无效值的数组
    x = np.array([-np.inf, -1, 0, np.nan, .7, 2, np.inf])
    # 预期输出的数组，用于测试 colormap 函数对输入数据 x 的处理结果
    expected = np.array([[0.050383, 0.029803, 0.527975, 1.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.,       0.,       0.,       0.],
                         [0.949217, 0.517763, 0.295662, 1.],
                         [0.940015, 0.975158, 0.131326, 1.],
                         [0.940015, 0.975158, 0.131326, 1.]])
    # 使用 assert_array_equal 函数验证 colormap(cmap) 对输入 x 的输出是否符合预期
    assert_array_equal(cmap(x), expected)

    # 测试带掩码值的表示，(-inf, inf) 被掩码处理
    expected = np.array([[0.,       0.,       0.,       0.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.,       0.,       0.,       0.],
                         [0.949217, 0.517763, 0.295662, 1.],
                         [0.940015, 0.975158, 0.131326, 1.],
                         [0.,       0.,       0.,       0.]])
    # 使用 assert_array_equal 函数验证 colormap(cmap) 对带有无效值掩码的输入 np.ma.masked_invalid(x) 的输出是否符合预期
    assert_array_equal(cmap(np.ma.masked_invalid(x)), expected)

    # 测试标量表示
    # 验证 colormap(-np.inf) 和 colormap(0) 的输出是否相等
    assert_array_equal(cmap(-np.inf), cmap(0))
    # 验证 colormap(np.inf) 和 colormap(1.0) 的输出是否相等
    assert_array_equal(cmap(np.inf), cmap(1.0))
    # 验证 colormap(np.nan) 的输出是否为 [0., 0., 0., 0.]
    assert_array_equal(cmap(np.nan), [0., 0., 0., 0.])
def test_colormap_return_types():
    """
    Make sure that tuples are returned for scalar input and
    that the proper shapes are returned for ndarrays.
    """
    # 获取"plasma"色图对象
    cmap = mpl.colormaps["plasma"]

    # 测试返回类型和形状

    # 对于标量输入，应返回长度为4的元组
    assert isinstance(cmap(0.5), tuple)
    assert len(cmap(0.5)) == 4

    # 输入数组应返回形状为 x.shape + (4,) 的ndarray
    x = np.ones(4)
    assert cmap(x).shape == x.shape + (4,)

    # 多维数组输入
    x2d = np.zeros((2, 2))
    assert cmap(x2d).shape == x2d.shape + (4,)


def test_BoundaryNorm():
    """
    GitHub issue #1258: interpolation was failing with numpy
    1.7 pre-release.
    """

    boundaries = [0, 1.1, 2.2]
    vals = [-1, 0, 1, 2, 2.2, 4]

    # 不使用插值
    expected = [-1, 0, 0, 1, 2, 2]
    ncolors = len(boundaries) - 1
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)

    # ncolors != len(boundaries) - 1 会触发插值
    expected = [-1, 0, 0, 2, 3, 3]
    ncolors = len(boundaries)
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)

    # 单一区域并使用插值
    expected = [-1, 1, 1, 1, 3, 3]
    bn = mcolors.BoundaryNorm([0, 2.2], ncolors)
    assert_array_equal(bn(vals), expected)

    # 更多边界用于第三种颜色
    boundaries = [0, 1, 2, 3]
    vals = [-1, 0.1, 1.1, 2.2, 4]
    ncolors = 5
    expected = [-1, 0, 2, 4, 5]
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)

    # 标量输入不应引发错误，并应返回一个标量
    boundaries = [0, 1, 2]
    vals = [-1, 0.1, 1.1, 2.2]
    bn = mcolors.BoundaryNorm(boundaries, 2)
    expected = [-1, 0, 1, 2]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)

    # 使用插值的情况
    bn = mcolors.BoundaryNorm(boundaries, 3)
    expected = [-1, 0, 2, 3]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)

    # 裁剪
    bn = mcolors.BoundaryNorm(boundaries, 3, clip=True)
    expected = [0, 0, 2, 2]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)

    # 掩码数组
    boundaries = [0, 1.1, 2.2]
    vals = np.ma.masked_invalid([-1., np.nan, 0, 1.4, 9])

    # 不使用插值
    ncolors = len(boundaries) - 1
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    expected = np.ma.masked_array([-1, -99, 0, 1, 2], mask=[0, 1, 0, 0, 0])
    assert_array_equal(bn(vals), expected)

    # 使用插值
    bn = mcolors.BoundaryNorm(boundaries, len(boundaries))
    # 创建预期的掩码数组，包含特定的值和掩码
    expected = np.ma.masked_array([-1, -99, 0, 2, 3], mask=[0, 1, 0, 0, 0])
    # 断言 bn(vals) 函数的输出与预期结果相等
    assert_array_equal(bn(vals), expected)

    # 非平凡的掩码数组测试
    vals = np.ma.masked_invalid([np.inf, np.nan])
    # 断言 bn(vals) 的掩码属性全为 True
    assert np.all(bn(vals).mask)
    vals = np.ma.masked_invalid([np.inf])
    # 断言 bn(vals) 的掩码属性全为 True
    assert np.all(bn(vals).mask)

    # 测试不兼容的 extend 和 clip 参数组合
    with pytest.raises(ValueError, match="not compatible"):
        mcolors.BoundaryNorm(np.arange(4), 5, extend='both', clip=True)

    # ncolors 参数过小的测试
    with pytest.raises(ValueError, match="ncolors must equal or exceed"):
        mcolors.BoundaryNorm(np.arange(4), 2)

    with pytest.raises(ValueError, match="ncolors must equal or exceed"):
        mcolors.BoundaryNorm(np.arange(4), 3, extend='min')

    with pytest.raises(ValueError, match="ncolors must equal or exceed"):
        mcolors.BoundaryNorm(np.arange(4), 4, extend='both')

    # 测试 extend 关键字，包含插值（大型 colormap）
    bounds = [1, 2, 3]
    cmap = mpl.colormaps['viridis']
    mynorm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    refnorm = mcolors.BoundaryNorm([0] + bounds + [4], cmap.N)
    x = np.random.randn(100) * 10 + 2
    ref = refnorm(x)
    ref[ref == 0] = -1
    ref[ref == cmap.N - 1] = cmap.N
    # 断言 mynorm(x) 的输出与 ref 相等
    assert_array_equal(mynorm(x), ref)

    # 不使用插值的情况
    cmref = mcolors.ListedColormap(['blue', 'red'])
    cmref.set_over('black')
    cmref.set_under('white')
    cmshould = mcolors.ListedColormap(['white', 'blue', 'red', 'black'])

    # 断言 cmref 的 over 颜色与 'black' 相同
    assert mcolors.same_color(cmref.get_over(), 'black')
    # 断言 cmref 的 under 颜色与 'white' 相同
    assert mcolors.same_color(cmref.get_under(), 'white')

    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='both')
    # 断言 mynorm 的 vmin 属性与 refnorm 的 vmin 属性相等
    assert mynorm.vmin == refnorm.vmin
    # 断言 mynorm 的 vmax 属性与 refnorm 的 vmax 属性相等
    assert mynorm.vmax == refnorm.vmax

    # 测试边界值处理：下界
    assert mynorm(bounds[0] - 0.1) == -1  # under
    # 测试边界值处理：第一个区间
    assert mynorm(bounds[0] + 0.1) == 1   # first bin -> second color
    # 测试边界值处理：倒数第二个颜色
    assert mynorm(bounds[-1] - 0.1) == cmshould.N - 2  # next-to-last color
    # 测试边界值处理：上界
    assert mynorm(bounds[-1] + 0.1) == cmshould.N  # over

    x = [-1, 1.2, 2.3, 9.6]
    # 断言 cmshould(mynorm(x)) 的输出与 cmshould([0, 1, 2, 3]) 相等
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2, 3]))
    x = np.random.randn(100) * 10 + 2
    # 断言 cmshould(mynorm(x)) 的输出与 cmref(refnorm(x)) 相等
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))

    # 只有最小值的情况
    cmref = mcolors.ListedColormap(['blue', 'red'])
    cmref.set_under('white')
    cmshould = mcolors.ListedColormap(['white', 'blue', 'red'])

    # 断言 cmref 的 under 颜色与 'white' 相同
    assert mcolors.same_color(cmref.get_under(), 'white')

    # 断言 cmref 的颜色数等于 2
    assert cmref.N == 2
    # 断言 cmshould 的颜色数等于 3
    assert cmshould.N == 3
    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='min')
    # 断言 mynorm 的 vmin 属性与 refnorm 的 vmin 属性相等
    assert mynorm.vmin == refnorm.vmin
    # 断言 mynorm 的 vmax 属性与 refnorm 的 vmax 属性相等
    assert mynorm.vmax == refnorm.vmax
    x = [-1, 1.2, 2.3]
    # 断言 cmshould(mynorm(x)) 的输出与 cmshould([0, 1, 2]) 相等
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2]))
    x = np.random.randn(100) * 10 + 2
    # 断言 cmshould(mynorm(x)) 的输出与 cmref(refnorm(x)) 相等
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))

    # 只有最大值的情况
    # 创建一个颜色映射对象 cmref，其中包含两种颜色：蓝色和红色
    cmref = mcolors.ListedColormap(['blue', 'red'])
    
    # 设置 cmref 的“超过范围”颜色为黑色
    cmref.set_over('black')
    
    # 创建另一个颜色映射对象 cmshould，包含三种颜色：蓝色、红色和黑色
    cmshould = mcolors.ListedColormap(['blue', 'red', 'black'])
    
    # 断言检查 cmref 的超过颜色是否为黑色
    assert mcolors.same_color(cmref.get_over(), 'black')
    
    # 断言检查 cmref 的颜色映射中颜色的数量是否为 2
    assert cmref.N == 2
    
    # 断言检查 cmshould 的颜色映射中颜色的数量是否为 3
    assert cmshould.N == 3
    
    # 使用给定的边界值 bounds 和颜色映射的数量 cmref.N 创建一个 BoundaryNorm 对象 refnorm
    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    
    # 使用给定的边界值 bounds、颜色映射的数量 cmshould.N 和 extend 参数 'max' 创建一个 BoundaryNorm 对象 mynorm
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='max')
    
    # 断言检查 mynorm 的最小值 vmin 是否与 refnorm 的最小值 vmin 相等
    assert mynorm.vmin == refnorm.vmin
    
    # 断言检查 mynorm 的最大值 vmax 是否与 refnorm 的最大值 vmax 相等
    assert mynorm.vmax == refnorm.vmax
    
    # 创建一个包含浮点数的列表 x
    x = [1.2, 2.3, 4]
    
    # 断言检查 cmshould 应用 mynorm 后输出的颜色结果是否与手动计算的颜色结果相等
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2]))
    
    # 生成一个包含 100 个随机数的 NumPy 数组 x，随机数服从标准正态分布乘以 10 并加上 2
    x = np.random.randn(100) * 10 + 2
    
    # 断言检查 cmshould 应用 mynorm 后输出的颜色结果是否与 cmref 应用 refnorm 后输出的颜色结果相等
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))
def test_CenteredNorm():
    np.random.seed(0)  # 设置随机种子为0，确保结果可重复

    # Assert equivalence to symmetrical Normalize.
    x = np.random.normal(size=100)  # 生成一个包含100个正态分布随机数的数组
    x_maxabs = np.max(np.abs(x))  # 计算x数组中绝对值的最大值
    norm_ref = mcolors.Normalize(vmin=-x_maxabs, vmax=x_maxabs)  # 创建一个以-x_maxabs到x_maxabs为范围的Normalize对象
    norm = mcolors.CenteredNorm()  # 创建一个CenteredNorm对象
    assert_array_almost_equal(norm_ref(x), norm(x))  # 断言使用两种方式进行归一化后的结果近似相等

    # Check that vcenter is in the center of vmin and vmax
    # when vcenter is set.
    vcenter = int(np.random.normal(scale=50))  # 生成一个均值为0，标准差为50的正态分布整数作为vcenter
    norm = mcolors.CenteredNorm(vcenter=vcenter)  # 创建一个指定vcenter的CenteredNorm对象
    norm.autoscale_None([1, 2])  # 自动设置范围，排除1和2
    assert norm.vmax + norm.vmin == 2 * vcenter  # 断言vmin和vmax之和等于2倍的vcenter

    # Check that halfrange can be set without setting vcenter and that it is
    # not reset through autoscale_None.
    norm = mcolors.CenteredNorm(halfrange=1.0)  # 创建一个指定halfrange的CenteredNorm对象
    norm.autoscale_None([1, 3000])  # 自动设置范围，排除1和3000
    assert norm.halfrange == 1.0  # 断言halfrange保持不变

    # Check that halfrange input works correctly.
    x = np.random.normal(size=10)  # 生成一个包含10个正态分布随机数的数组
    norm = mcolors.CenteredNorm(vcenter=0.5, halfrange=0.5)  # 创建一个指定vcenter和halfrange的CenteredNorm对象
    assert_array_almost_equal(x, norm(x))  # 断言使用指定的vcenter和halfrange后归一化的结果近似相等
    norm = mcolors.CenteredNorm(vcenter=1, halfrange=1)  # 创建一个指定vcenter和halfrange的CenteredNorm对象
    assert_array_almost_equal(x, 2 * norm(x))  # 断言使用指定的vcenter和halfrange后归一化的结果近似相等

    # Check that halfrange input works correctly and use setters.
    norm = mcolors.CenteredNorm()  # 创建一个CenteredNorm对象
    norm.vcenter = 2  # 设置vcenter为2
    norm.halfrange = 2  # 设置halfrange为2
    assert_array_almost_equal(x, 4 * norm(x))  # 断言使用指定的vcenter和halfrange后归一化的结果近似相等

    # Check that prior to adding data, setting halfrange first has same effect.
    norm = mcolors.CenteredNorm()  # 创建一个CenteredNorm对象
    norm.halfrange = 2  # 设置halfrange为2
    norm.vcenter = 2  # 设置vcenter为2
    assert_array_almost_equal(x, 4 * norm(x))  # 断言使用指定的vcenter和halfrange后归一化的结果近似相等

    # Check that manual change of vcenter adjusts halfrange accordingly.
    norm = mcolors.CenteredNorm()  # 创建一个CenteredNorm对象
    assert norm.vcenter == 0  # 断言vcenter初始为0
    # add data
    norm(np.linspace(-1.0, 0.0, 10))  # 向norm对象添加数据
    assert norm.vmax == 1.0  # 断言vmax为1.0
    assert norm.halfrange == 1.0  # 断言halfrange为1.0
    # set vcenter to 1, which should move the center but leave the
    # halfrange unchanged
    norm.vcenter = 1  # 设置vcenter为1，此操作应该移动中心但不改变halfrange
    assert norm.vmin == 0  # 断言vmin为0
    assert norm.vmax == 2  # 断言vmax为2
    assert norm.halfrange == 1  # 断言halfrange为1

    # Check setting vmin directly updates the halfrange and vmax, but
    # leaves vcenter alone
    norm.vmin = -1  # 直接设置vmin，更新halfrange和vmax，但不改变vcenter
    assert norm.halfrange == 2  # 断言halfrange为2
    assert norm.vmax == 3  # 断言vmax为3
    assert norm.vcenter == 1  # 断言vcenter为1

    # also check vmax updates
    norm.vmax = 2  # 检查vmax更新
    assert norm.halfrange == 1  # 断言halfrange为1
    assert norm.vmin == 0  # 断言vmin为0
    assert norm.vcenter == 1  # 断言vcenter为1


@pytest.mark.parametrize("vmin,vmax", [[-1, 2], [3, 1]])
def test_lognorm_invalid(vmin, vmax):
    # Check that invalid limits in LogNorm error
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)  # 创建一个LogNorm对象，检查是否会引发错误
    with pytest.raises(ValueError):  # 使用pytest断言引发了ValueError错误
        norm(1)
    with pytest.raises(ValueError):  # 使用pytest断言引发了ValueError错误
        norm.inverse(1)


def test_LogNorm():
    """
    LogNorm ignored clip, now it has the same
    behavior as Normalize, e.g., values > vmax are bigger than 1
    without clip, with clip they are 1.
    """
    ln = mcolors.LogNorm(clip=True, vmax=5)  # 创建一个LogNorm对象，设置clip为True，vmax为5
    assert_array_equal(ln([1, 6]), [0, 1.0])  # 断言对于给定的输入，LogNorm对象的处理结果符合预期


def test_LogNorm_inverse():
    """
    Test that lists work, and that the inverse works
    """
    # 创建一个对数标准化器 `norm`，指定范围为 vmin=0.1 到 vmax=10
    norm = mcolors.LogNorm(vmin=0.1, vmax=10)
    
    # 断言：验证对数标准化器 `norm` 对输入 [0.5, 0.4] 的输出准确性
    assert_array_almost_equal(norm([0.5, 0.4]), [0.349485, 0.30103])
    
    # 断言：验证对数标准化器 `norm` 的逆操作对 [0.349485, 0.30103] 的输出准确性
    assert_array_almost_equal([0.5, 0.4], norm.inverse([0.349485, 0.30103]))
    
    # 断言：验证对数标准化器 `norm` 对 0.4 的输出准确性
    assert_array_almost_equal(norm(0.4), [0.30103])
    
    # 断言：验证对数标准化器 `norm` 的逆操作对 [0.30103] 的输出准确性
    assert_array_almost_equal([0.4], norm.inverse([0.30103]))
def test_PowerNorm():
    # 检查指数为1时，与普通线性归一化结果相同。同时隐式检查从第一个数组输入自动初始化vmin/vmax。
    a = np.array([0, 0.5, 1, 1.5], dtype=float)
    # 创建指数为1的PowerNorm对象
    pnorm = mcolors.PowerNorm(1)
    # 创建普通Normalize对象
    norm = mcolors.Normalize()
    # 断言数组a经过普通归一化和指数归一化后的结果几乎相等
    assert_array_almost_equal(norm(a), pnorm(a))

    a = np.array([-0.5, 0, 2, 4, 8], dtype=float)
    expected = [-1/16, 0, 1/16, 1/4, 1]
    # 创建指数为2的PowerNorm对象，并指定vmin和vmax
    pnorm = mcolors.PowerNorm(2, vmin=0, vmax=8)
    # 断言数组a经过指数归一化后的结果与预期结果几乎相等
    assert_array_almost_equal(pnorm(a), expected)
    # 断言单个元素经过指数归一化后的结果与预期结果相等
    assert pnorm(a[0]) == expected[0]
    assert pnorm(a[2]) == expected[2]
    # 检查逆运算
    a_roundtrip = pnorm.inverse(pnorm(a))
    # 断言逆运算后的结果与原始输入数组a几乎相等
    assert_array_almost_equal(a, a_roundtrip)
    # PowerNorm逆运算会添加一个掩码，检查掩码是否正确
    assert_array_equal(a_roundtrip.mask, np.zeros(a.shape, dtype=bool))

    # Clip = True
    a = np.array([-0.5, 0, 1, 8, 16], dtype=float)
    expected = [0, 0, 0, 1, 1]
    # 创建指数为2的PowerNorm对象，并指定vmin和vmax，并设置clip=True
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=True)
    # 断言数组a经过指数归一化后的结果与预期结果几乎相等
    assert_array_almost_equal(pnorm(a), expected)
    assert pnorm(a[0]) == expected[0]
    assert pnorm(a[-1]) == expected[-1]
    # 创建指数为2的PowerNorm对象，并指定vmin和vmax，clip=False
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=False)
    # 断言在调用时使用clip=True时，数组a经过指数归一化后的结果与预期结果几乎相等
    assert_array_almost_equal(pnorm(a, clip=True), expected)
    assert pnorm(a[0], clip=True) == expected[0]
    assert pnorm(a[-1], clip=True) == expected[-1]

    # 检查clip=True保留掩码值
    a = np.ma.array([5, 2], mask=[True, False])
    out = pnorm(a, clip=True)
    assert_array_equal(out.mask, [True, False])


def test_PowerNorm_translation_invariance():
    a = np.array([0, 1/2, 1], dtype=float)
    expected = [0, 1/8, 1]
    # 创建指数为3的PowerNorm对象，并指定vmin和vmax为0和1
    pnorm = mcolors.PowerNorm(vmin=0, vmax=1, gamma=3)
    # 断言数组a经过指数归一化后的结果与预期结果几乎相等
    assert_array_almost_equal(pnorm(a), expected)
    # 创建指数为3的PowerNorm对象，并指定vmin和vmax为-2和-1
    pnorm = mcolors.PowerNorm(vmin=-2, vmax=-1, gamma=3)
    # 断言数组(a - 2)经过指数归一化后的结果与预期结果几乎相等
    assert_array_almost_equal(pnorm(a - 2), expected)


def test_powernorm_cbar_limits():
    fig, ax = plt.subplots()
    vmin, vmax = 300, 1000
    data = np.arange(10*10).reshape(10, 10) + vmin
    # 创建imshow对象，并指定使用gamma为0.2的PowerNorm对象作为归一化方法
    im = ax.imshow(data, norm=mcolors.PowerNorm(gamma=0.2, vmin=vmin, vmax=vmax))
    # 创建colorbar
    cbar = fig.colorbar(im)
    # 断言colorbar的y轴限制与指定的vmin和vmax相同
    assert cbar.ax.get_ylim() == (vmin, vmax)


def test_Normalize():
    # 创建普通Normalize对象
    norm = mcolors.Normalize()
    vals = np.arange(-10, 10, 1, dtype=float)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)

    # 处理整数输入的情况（在计算最大值和最小值时不会溢出，例如127-(-128)）
    vals = np.array([-128, 127], dtype=np.int8)
    # 创建Normalize对象，并指定vmin和vmax
    norm = mcolors.Normalize(vals.min(), vals.max())
    # 断言数组vals经过Normalize对象处理后的结果与预期结果相等
    assert_array_equal(norm(vals), [0, 1])

    # 不会在长双精度浮点数（例如Linux上的float128）上丢失精度
    # 对于数组输入...
    vals = np.array([1.2345678901, 9.8765432109], dtype=np.longdouble)
    # 创建Normalize对象，并指定vmin和vmax
    norm = mcolors.Normalize(vals[0], vals[1])
    # 断言norm对象处理vals后的数据类型为longdouble
    assert norm(vals).dtype == np.longdouble
    # 使用 assert_array_equal 函数检查 vals 经过 norm 函数归一化后的结果是否与 [0, 1] 相等
    assert_array_equal(norm(vals), [0, 1])
    # 对标量值进行相同的归一化检查
    eps = np.finfo(np.longdouble).resolution
    # 使用 np.finfo(np.longdouble).resolution 获取 longdouble 类型的机器精度
    norm = plt.Normalize(1, 1 + 100 * eps)
    # 使用 plt.Normalize 创建一个以 1 为基准值，范围为 [1, 1 + 100 * eps] 的归一化对象
    # 当 longdouble 是扩展精度（80 位）时，此操作将精确返回 0.5
    # 当 longdouble 是四倍精度（128 位）时，此操作将返回接近 0.5 的值
    assert_array_almost_equal(norm(1 + 50 * eps), 0.5, decimal=3)
    # 使用 assert_array_almost_equal 函数检查 norm 对象对 1 + 50 * eps 的归一化结果是否接近于 0.5，精确到小数点后三位
def test_FuncNorm():
    # 定义一个函数变量 forward，用于计算输入值的平方
    def forward(x):
        return (x**2)
    
    # 定义一个函数变量 inverse，用于计算输入值的平方根
    def inverse(x):
        return np.sqrt(x)

    # 创建一个 FuncNorm 对象 norm，传入 forward 和 inverse 函数，并设置范围为 [0, 10]
    norm = mcolors.FuncNorm((forward, inverse), vmin=0, vmax=10)
    
    # 预期输出结果
    expected = np.array([0, 0.25, 1])
    # 输入值
    input = np.array([0, 5, 10])
    
    # 断言 norm(input) 的输出与预期结果 expected 相近
    assert_array_almost_equal(norm(input), expected)
    # 断言 norm 的逆操作 norm.inverse(expected) 与 input 相近
    assert_array_almost_equal(norm.inverse(expected), input)

    # 重新定义 forward 函数，用于计算输入值的以 10 为底的对数
    def forward(x):
        return np.log10(x)
    
    # 重新定义 inverse 函数，用于计算输入值的 10 的幂
    def inverse(x):
        return 10**x
    
    # 创建一个新的 FuncNorm 对象 norm，传入重新定义后的 forward 和 inverse 函数，并设置范围为 [0.1, 10]
    norm = mcolors.FuncNorm((forward, inverse), vmin=0.1, vmax=10)
    
    # 创建一个 LogNorm 对象 lognorm，范围也为 [0.1, 10]
    lognorm = mcolors.LogNorm(vmin=0.1, vmax=10)
    
    # 断言 norm([0.2, 5, 10]) 的输出与 lognorm([0.2, 5, 10]) 相近
    assert_array_almost_equal(norm([0.2, 5, 10]), lognorm([0.2, 5, 10]))
    # 断言 norm 的逆操作 norm.inverse([0.2, 5, 10]) 与 lognorm 的逆操作 lognorm.inverse([0.2, 5, 10]) 相近
    assert_array_almost_equal(norm.inverse([0.2, 5, 10]), lognorm.inverse([0.2, 5, 10]))


def test_TwoSlopeNorm_autoscale():
    # 创建一个 TwoSlopeNorm 对象 norm，设置中心点为 20
    norm = mcolors.TwoSlopeNorm(vcenter=20)
    
    # 自动调整范围，基于给定的数据 [10, 20, 30, 40]
    norm.autoscale([10, 20, 30, 40])
    
    # 断言 norm 的最小值 vmin 为 10.0
    assert norm.vmin == 10.
    # 断言 norm 的最大值 vmax 为 40.0
    assert norm.vmax == 40.


def test_TwoSlopeNorm_autoscale_None_vmin():
    # 创建一个 TwoSlopeNorm 对象 norm，设置中心点为 2，最小值为 0，最大值未指定
    norm = mcolors.TwoSlopeNorm(2, vmin=0, vmax=None)
    
    # 自动调整范围，基于给定的数据 [1, 2, 3, 4, 5]
    norm.autoscale_None([1, 2, 3, 4, 5])
    
    # 断言 norm(5) 的输出为 1
    assert norm(5) == 1
    # 断言 norm 的最大值 vmax 为 5
    assert norm.vmax == 5


def test_TwoSlopeNorm_autoscale_None_vmax():
    # 创建一个 TwoSlopeNorm 对象 norm，设置中心点为 2，最小值未指定，最大值为 10
    norm = mcolors.TwoSlopeNorm(2, vmin=None, vmax=10)
    
    # 自动调整范围，基于给定的数据 [1, 2, 3, 4, 5]
    norm.autoscale_None([1, 2, 3, 4, 5])
    
    # 断言 norm(1) 的输出为 0
    assert norm(1) == 0
    # 断言 norm 的最小值 vmin 为 1
    assert norm.vmin == 1


def test_TwoSlopeNorm_scale():
    # 创建一个 TwoSlopeNorm 对象 norm，设置中心点为 2
    norm = mcolors.TwoSlopeNorm(2)
    
    # 断言 norm 的 scaled() 方法返回 False
    assert norm.scaled() is False
    # 调用 norm([1, 2, 3, 4])，对 norm 进行缩放
    norm([1, 2, 3, 4])
    # 断言 norm 的 scaled() 方法返回 True
    assert norm.scaled() is True


def test_TwoSlopeNorm_scaleout_center():
    # 创建一个 TwoSlopeNorm 对象 norm，设置中心点为 0
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    
    # 调用 norm([0, 1, 2, 3, 5])，检查 norm 的 vmin 是否不超过 vcenter
    norm([0, 1, 2, 3, 5])
    
    # 断言 norm 的最小值 vmin 为 -5
    assert norm.vmin == -5
    # 断言 norm 的最大值 vmax 为 5
    assert norm.vmax == 5


def test_TwoSlopeNorm_scaleout_center_max():
    # 创建一个 TwoSlopeNorm 对象 norm，设置中心点为 0
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    
    # 调用 norm([0, -1, -2, -3, -5])，检查 norm 的 vmax 是否不低于 vcenter
    norm([0, -1, -2, -3, -5])
    
    # 断言 norm 的最大值 vmax 为 5
    assert norm.vmax == 5
    # 断言 norm 的最小值 vmin 为 -5
    assert norm.vmin == -5


def test_TwoSlopeNorm_Even():
    # 创建一个 TwoSlopeNorm 对象 norm，设置范围为 [-1, 0, 4]，中心点为 0
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=4)
    
    # 创建输入值数组 vals
    vals = np.array([-1.0, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0])
    # 预期输出结果数组 expected
    expected = np.array([0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
    
    # 断言 norm(vals) 的输出与预期结果 expected 相等
    assert_array_equal(norm(vals), expected)


def test_TwoSlopeNorm_Odd():
    # 创建一个 TwoSlopeNorm 对象 norm，设置范围为 [-2, 0, 5]，中心点为 0
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=5)
    
    # 创建输入值数组 vals
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # 预期输出结果数组 expected
    expected = np.array([0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # 断言 norm(vals) 的输出与预期结果 expected 相等
    assert_array_equal(norm(vals), expected)


def test_TwoSlopeNorm_VminEqualsVcenter():
    # 当 vmin 等于 vcenter 时，应该引发 ValueError 异常
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=-2, vcenter=-2, vmax=2)


def test_TwoSlopeNorm_VmaxEqualsVcenter():
    # 当 vmax 等于 vcenter 时，应该引发 ValueError 异常
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=-2, vcenter=2, vmax=2)


def test_TwoSlopeNorm_VminGTVcenter():
    # 当 vmin 大于 vcenter 时，应该引发 ValueError 异常
    with pytest.raises
# 测试函数，验证当 vcenter 大于 vmax 时，TwoSlopeNorm 是否会引发 ValueError 异常
def test_TwoSlopeNorm_VcenterGTVmax():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=10, vcenter=25, vmax=20)

# 测试函数，验证在没有设置合适的缩放因子时，TwoSlopeNorm 的逆操作是否会引发 ValueError 异常
def test_TwoSlopeNorm_premature_scaling():
    norm = mcolors.TwoSlopeNorm(vcenter=2)
    with pytest.raises(ValueError):
        norm.inverse(np.array([0.1, 0.5, 0.9]))

# 测试 SymLogNorm 的行为，包括正常使用、逆操作、标量测试和遮罩测试
def test_SymLogNorm():
    norm = mcolors.SymLogNorm(3, vmax=5, linscale=1.2, base=np.e)
    vals = np.array([-30, -1, 2, 6], dtype=float)
    normed_vals = norm(vals)
    expected = [0., 0.53980074, 0.826991, 1.02758204]
    assert_array_almost_equal(normed_vals, expected)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)

    # 确保指定 vmin 时的结果与上述相同
    norm = mcolors.SymLogNorm(3, vmin=-30, vmax=5, linscale=1.2, base=np.e)
    normed_vals = norm(vals)
    assert_array_almost_equal(normed_vals, expected)

    # 更易于检查的测试
    norm = mcolors.SymLogNorm(1, vmin=-np.e**3, vmax=np.e**3, base=np.e)
    nn = norm([-np.e**3, -np.e**2, -np.e**1, -1,
              0, 1, np.e**1, np.e**2, np.e**3])
    xx = np.array([0., 0.109123, 0.218246, 0.32737, 0.5, 0.67263,
                   0.781754, 0.890877, 1.])
    assert_array_almost_equal(nn, xx)
    norm = mcolors.SymLogNorm(1, vmin=-10**3, vmax=10**3, base=10)
    nn = norm([-10**3, -10**2, -10**1, -1,
              0, 1, 10**1, 10**2, 10**3])
    xx = np.array([0., 0.121622, 0.243243, 0.364865, 0.5, 0.635135,
                   0.756757, 0.878378, 1.])
    assert_array_almost_equal(nn, xx)

# 测试未调用 SymLogNorm 时的 ColorbarBase
def test_SymLogNorm_colorbar():
    norm = mcolors.SymLogNorm(0.1, vmin=-1, vmax=1, linscale=1, base=np.e)
    fig = plt.figure()
    mcolorbar.ColorbarBase(fig.add_subplot(), norm=norm)
    plt.close(fig)

# 测试 SymLogNorm，确保其不会在零标签上添加子刻度
def test_SymLogNorm_single_zero():
    fig = plt.figure()
    norm = mcolors.SymLogNorm(1e-5, vmin=-1, vmax=1, base=np.e)
    cbar = mcolorbar.ColorbarBase(fig.add_subplot(), norm=norm)
    ticks = cbar.get_ticks()
    assert np.count_nonzero(ticks == 0) <= 1
    plt.close(fig)

# AsinhNorm 的测试类，包含初始化和规范化测试
class TestAsinhNorm:
    """
    Tests for `~.colors.AsinhNorm`
    """

    # 测试初始化
    def test_init(self):
        norm0 = mcolors.AsinhNorm()
        assert norm0.linear_width == 1

        norm5 = mcolors.AsinhNorm(linear_width=5)
        assert norm5.linear_width == 5

    # 测试规范化操作
    def test_norm(self):
        norm = mcolors.AsinhNorm(2, vmin=-4, vmax=4)
        vals = np.arange(-3.5, 3.5, 10)
        normed_vals = norm(vals)
        asinh2 = np.arcsinh(2)

        expected = (2 * np.arcsinh(vals / 2) + 2 * asinh2) / (4 * asinh2)
        assert_array_almost_equal(normed_vals, expected)

# 内部函数，用于检查给定规范化的逆操作是否正常工作
def _inverse_tester(norm_instance, vals):
    """
    Checks if the inverse of the given normalization is working.
    """
    # 使用断言来验证标准化对象的逆转换是否几乎等于原始数据
    assert_array_almost_equal(norm_instance.inverse(norm_instance(vals)), vals)
def _scalar_tester(norm_instance, vals):
    """
    Checks if scalars and arrays are handled the same way.
    Tests only for float.
    """
    # Apply the normalization instance to each value in the 'vals' list of floats
    scalar_result = [norm_instance(float(v)) for v in vals]
    # Assert that the array 'scalar_result' is almost equal to the result of applying
    # the normalization instance to the entire 'vals' array
    assert_array_almost_equal(scalar_result, norm_instance(vals))


def _mask_tester(norm_instance, vals):
    """
    Checks mask handling
    """
    # Create a masked array from 'vals'
    masked_array = np.ma.array(vals)
    # Mask the first element of the masked array
    masked_array[0] = np.ma.masked
    # Assert that the mask of the masked array returned by the normalization instance
    # is equal to the mask of the original masked array
    assert_array_equal(masked_array.mask, norm_instance(masked_array).mask)


@image_comparison(['levels_and_colors.png'])
def test_cmap_and_norm_from_levels_and_colors():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    # Generate a 7x7 data array ranging from -2 to 4
    data = np.linspace(-2, 4, 49).reshape(7, 7)
    # Define levels and corresponding colors for the colormap
    levels = [-1, 2, 2.5, 3]
    colors = ['red', 'green', 'blue', 'yellow', 'black']
    extend = 'both'
    # Generate a colormap and normalization instance from levels and colors
    cmap, norm = mcolors.from_levels_and_colors(levels, colors, extend=extend)

    # Create a plot with color mesh using the generated colormap and normalization
    ax = plt.axes()
    m = plt.pcolormesh(data, cmap=cmap, norm=norm)
    plt.colorbar(m)

    # Hide the axes labels (but not the colorbar ones, as they are useful)
    ax.tick_params(labelleft=False, labelbottom=False)


@image_comparison(baseline_images=['boundarynorm_and_colorbar'],
                  extensions=['png'], tol=1.0)
def test_boundarynorm_and_colorbarbase():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    # Create a figure and define axes with specified dimensions
    fig = plt.figure()
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
    ax3 = fig.add_axes([0.05, 0.15, 0.9, 0.15])

    # Set the colormap and bounds
    bounds = [-1, 2, 5, 7, 12, 15]
    cmap = mpl.colormaps['viridis']

    # Default behavior
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    # Create a colorbar with base on the first set of axes
    cb1 = mcolorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, extend='both',
                                 orientation='horizontal', spacing='uniform')

    # New behavior
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    # Create a colorbar with base on the second set of axes
    cb2 = mcolorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                                 orientation='horizontal')

    # User can still force to any extend='' if really needed
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    # Create a colorbar with base on the third set of axes
    cb3 = mcolorbar.ColorbarBase(ax3, cmap=cmap, norm=norm,
                                 extend='neither', orientation='horizontal')


def test_cmap_and_norm_from_levels_and_colors2():
    levels = [-1, 2, 2.5, 3]
    colors = ['red', (0, 1, 0), 'blue', (0.5, 0.5, 0.5), (0.0, 0.0, 0.0, 1.0)]
    clr = mcolors.to_rgba_array(colors)
    bad = (0.1, 0.1, 0.1, 0.1)
    no_color = (0.0, 0.0, 0.0, 0.0)
    masked_value = 'masked_value'

    # Define the test values which are of interest.
    # Note: levels are lev[i] <= v < lev[i+1]
    # 定义测试数据，每个元组包含一个扩展类型、一个索引和一个预期结果字典
    tests = [('both', None, {-2: clr[0],        # 扩展为both，索引为None时的预期结果字典
                             -1: clr[1],
                             2: clr[2],
                             2.25: clr[2],
                             3: clr[4],
                             3.5: clr[4],
                             masked_value: bad}),

             ('min', -1, {-2: clr[0],          # 扩展为min，索引为-1时的预期结果字典
                          -1: clr[1],
                          2: clr[2],
                          2.25: clr[2],
                          3: no_color,
                          3.5: no_color,
                          masked_value: bad}),

             ('max', -1, {-2: no_color,        # 扩展为max，索引为-1时的预期结果字典
                          -1: clr[0],
                          2: clr[1],
                          2.25: clr[1],
                          3: clr[3],
                          3.5: clr[3],
                          masked_value: bad}),

             ('neither', -2, {-2: no_color,    # 扩展为neither，索引为-2时的预期结果字典
                              -1: clr[0],
                              2: clr[1],
                              2.25: clr[1],
                              3: no_color,
                              3.5: no_color,
                              masked_value: bad}),
             ]

    # 对每个测试用例进行迭代
    for extend, i1, cases in tests:
        # 使用给定的级别和颜色创建颜色映射和归一化器
        cmap, norm = mcolors.from_levels_and_colors(levels, colors[0:i1],
                                                    extend=extend)
        # 设置无效值颜色
        cmap.set_bad(bad)
        # 对于每个数据值及其预期颜色进行断言
        for d_val, expected_color in cases.items():
            if d_val == masked_value:
                d_val = np.ma.array([1], mask=True)  # 如果数据值为masked_value，则创建一个带有掩码的numpy数组
            else:
                d_val = [d_val]  # 否则，将数据值包装为列表
            # 断言实际计算的颜色与预期颜色相等
            assert_array_equal(expected_color, cmap(norm(d_val))[0],
                               f'With extend={extend!r} and data '
                               f'value={d_val!r}')

    # 使用pytest检查是否引发了值错误异常
    with pytest.raises(ValueError):
        mcolors.from_levels_and_colors(levels, colors)
# 定义测试函数，验证 RGB 和 HSV 之间的往返转换
def test_rgb_hsv_round_trip():
    # 遍历不同形状的数组
    for a_shape in [(500, 500, 3), (500, 3), (1, 3), (3,)]:
        # 设置随机数种子为0，确保可重复的随机数生成
        np.random.seed(0)
        # 生成指定形状的随机数组
        tt = np.random.random(a_shape)
        # 断言 RGB 转 HSV 再转回 RGB 后结果与原始数据几乎相等
        assert_array_almost_equal(
            tt, mcolors.hsv_to_rgb(mcolors.rgb_to_hsv(tt)))
        # 断言 HSV 转 RGB 再转回 HSV 后结果与原始数据几乎相等
        assert_array_almost_equal(
            tt, mcolors.rgb_to_hsv(mcolors.hsv_to_rgb(tt)))


def test_autoscale_masked():
    # 测试 issue #2336。之前完全遮蔽的数据会触发 ValueError
    data = np.ma.masked_all((12, 20))
    # 使用伪彩色图绘制遮蔽后的数据
    plt.pcolor(data)
    # 绘制图形


@image_comparison(['light_source_shading_topo.png'])
def test_light_source_topo_surface():
    """
    使用不同的垂直夸张和混合模式对 DEM 进行阴影处理。
    """
    # 获取示例数据中的地形数据
    dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
    elev = dem['elevation']
    dx, dy = dem['dx'], dem['dy']
    # 根据经纬度范围计算真实的像素大小，用于准确的垂直夸张
    dx = 111320.0 * dx * np.cos(dem['ymin'])
    dy = 111320.0 * dy

    # 创建光源对象，设置光源方位角和高度角
    ls = mcolors.LightSource(315, 45)
    cmap = cm.gist_earth

    # 创建子图 3x3 的图表
    fig, axs = plt.subplots(nrows=3, ncols=3)
    # 在每个子图中，使用不同的混合模式和垂直夸张值来阴影处理 DEM
    for row, mode in zip(axs, ['hsv', 'overlay', 'soft']):
        for ax, ve in zip(row, [0.1, 1, 10]):
            rgb = ls.shade(elev, cmap, vert_exag=ve, dx=dx, dy=dy,
                           blend_mode=mode)
            ax.imshow(rgb)
            ax.set(xticks=[], yticks=[])


def test_light_source_shading_default():
    """
    默认情况下使用 "hsv" 混合模式的数组比较测试。
    确保默认结果在没有警告的情况下不会更改。
    """
    # 创建二维坐标网格
    y, x = np.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    # 计算高度数据
    z = 10 * np.cos(x**2 + y**2)

    cmap = plt.cm.copper
    # 创建光源对象，设置光源方位角和高度角
    ls = mcolors.LightSource(315, 45)
    # 使用光源对象对高度数据进行阴影处理，使用铜色调色板
    rgb = ls.shade(z, cmap)

    # 结果存储为转置后的数据，并四舍五入以便更紧凑地显示...
    # 期望的 RGB 数组，包含了多个颜色通道的值
    expect = np.array(
        # 第一个通道的 RGB 值
        [[[0.00, 0.45, 0.90, 0.90, 0.82, 0.62, 0.28, 0.00],
          [0.45, 0.94, 0.99, 1.00, 1.00, 0.96, 0.65, 0.17],
          [0.90, 0.99, 1.00, 1.00, 1.00, 1.00, 0.94, 0.35],
          [0.90, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.49],
          [0.82, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.41],
          [0.62, 0.96, 1.00, 1.00, 1.00, 1.00, 0.90, 0.07],
          [0.28, 0.65, 0.94, 1.00, 1.00, 0.90, 0.35, 0.01],
          [0.00, 0.17, 0.35, 0.49, 0.41, 0.07, 0.01, 0.00]],
    
         # 第二个通道的 RGB 值
         [[0.00, 0.28, 0.59, 0.72, 0.62, 0.40, 0.18, 0.00],
          [0.28, 0.78, 0.93, 0.92, 0.83, 0.66, 0.39, 0.11],
          [0.59, 0.93, 0.99, 1.00, 0.92, 0.75, 0.50, 0.21],
          [0.72, 0.92, 1.00, 0.99, 0.93, 0.76, 0.51, 0.18],
          [0.62, 0.83, 0.92, 0.93, 0.87, 0.68, 0.42, 0.08],
          [0.40, 0.66, 0.75, 0.76, 0.68, 0.52, 0.23, 0.02],
          [0.18, 0.39, 0.50, 0.51, 0.42, 0.23, 0.00, 0.00],
          [0.00, 0.11, 0.21, 0.18, 0.08, 0.02, 0.00, 0.00]],
    
         # 第三个通道的 RGB 值
         [[0.00, 0.18, 0.38, 0.46, 0.39, 0.26, 0.11, 0.00],
          [0.18, 0.50, 0.70, 0.75, 0.64, 0.44, 0.25, 0.07],
          [0.38, 0.70, 0.91, 0.98, 0.81, 0.51, 0.29, 0.13],
          [0.46, 0.75, 0.98, 0.96, 0.84, 0.48, 0.22, 0.12],
          [0.39, 0.64, 0.81, 0.84, 0.71, 0.31, 0.11, 0.05],
          [0.26, 0.44, 0.51, 0.48, 0.31, 0.10, 0.03, 0.01],
          [0.11, 0.25, 0.29, 0.22, 0.11, 0.03, 0.00, 0.00],
          [0.00, 0.07, 0.13, 0.12, 0.05, 0.01, 0.00, 0.00]],
    
         # 第四个通道的 RGB 值，全为 1.0，表示不透明的白色
         [[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]]
         ]).T
    
    # 使用 numpy.testing 包中的 assert_array_almost_equal 函数，比较 rgb 和 expect 数组的近似相等性
    assert_array_almost_equal(rgb, expect, decimal=2)
def test_light_source_shading_empty_mask():
    # 创建一个二维坐标网格，范围为[-1.2, 1.2]，分成8等份，用于生成 x 和 y 坐标
    y, x = np.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    # 根据 x 和 y 的值计算 z0，表示一个函数在二维平面上的数值
    z0 = 10 * np.cos(x**2 + y**2)
    # 创建一个 masked array，即带有遮罩的数组，初始遮罩条件与 z0 一致
    z1 = np.ma.array(z0)

    # 使用 copper 颜色映射创建颜色映射对象
    cmap = plt.cm.copper
    # 创建一个光源对象，设置光源位置和方向
    ls = mcolors.LightSource(315, 45)
    # 对 z0 应用光照效果，得到 RGB 彩色图像
    rgb0 = ls.shade(z0, cmap)
    # 对 z1 应用光照效果，得到 RGB 彩色图像
    rgb1 = ls.shade(z1, cmap)

    # 断言 rgb0 和 rgb1 的值几乎相等
    assert_array_almost_equal(rgb0, rgb1)


# Numpy 1.9.1 fixed a bug in masked arrays which resulted in
# additional elements being masked when calculating the gradient thus
# the output is different with earlier numpy versions.
def test_light_source_masked_shading():
    """
    Array comparison test for a surface with a masked portion. Ensures that
    we don't wind up with "fringes" of odd colors around masked regions.
    """
    # 创建一个二维坐标网格，范围为[-1.2, 1.2]，分成8等份，用于生成 x 和 y 坐标
    y, x = np.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    # 根据 x 和 y 的值计算 z，表示一个函数在二维平面上的数值
    z = 10 * np.cos(x**2 + y**2)

    # 将 z 中大于 9.9 的值遮罩起来，不参与后续计算和显示
    z = np.ma.masked_greater(z, 9.9)

    # 使用 copper 颜色映射创建颜色映射对象
    cmap = plt.cm.copper
    # 创建一个光源对象，设置光源位置和方向
    ls = mcolors.LightSource(315, 45)
    # 对 z 应用光照效果，得到 RGB 彩色图像
    rgb = ls.shade(z, cmap)

    # 期望的结果，为了更紧凑的显示，结果已经进行了转置和四舍五入处理
    expect = np.array(
        [[[0.00, 0.46, 0.91, 0.91, 0.84, 0.64, 0.29, 0.00],
          [0.46, 0.96, 1.00, 1.00, 1.00, 0.97, 0.67, 0.18],
          [0.91, 1.00, 1.00, 1.00, 1.00, 1.00, 0.96, 0.36],
          [0.91, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 0.51],
          [0.84, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 0.44],
          [0.64, 0.97, 1.00, 1.00, 1.00, 1.00, 0.94, 0.09],
          [0.29, 0.67, 0.96, 1.00, 1.00, 0.94, 0.38, 0.01],
          [0.00, 0.18, 0.36, 0.51, 0.44, 0.09, 0.01, 0.00]],

         [[0.00, 0.29, 0.61, 0.75, 0.64, 0.41, 0.18, 0.00],
          [0.29, 0.81, 0.95, 0.93, 0.85, 0.68, 0.40, 0.11],
          [0.61, 0.95, 1.00, 0.78, 0.78, 0.77, 0.52, 0.22],
          [0.75, 0.93, 0.78, 0.00, 0.00, 0.78, 0.54, 0.19],
          [0.64, 0.85, 0.78, 0.00, 0.00, 0.78, 0.45, 0.08],
          [0.41, 0.68, 0.77, 0.78, 0.78, 0.55, 0.25, 0.02],
          [0.18, 0.40, 0.52, 0.54, 0.45, 0.25, 0.00, 0.00],
          [0.00, 0.11, 0.22, 0.19, 0.08, 0.02, 0.00, 0.00]],

         [[0.00, 0.19, 0.39, 0.48, 0.41, 0.26, 0.12, 0.00],
          [0.19, 0.52, 0.73, 0.78, 0.66, 0.46, 0.26, 0.07],
          [0.39, 0.73, 0.95, 0.50, 0.50, 0.53, 0.30, 0.14],
          [0.48, 0.78, 0.50, 0.00, 0.00, 0.50, 0.23, 0.12],
          [0.41, 0.66, 0.50, 0.00, 0.00, 0.50, 0.11, 0.05],
          [0.26, 0.46, 0.53, 0.50, 0.50, 0.11, 0.03, 0.01],
          [0.12, 0.26, 0.30, 0.23, 0.11, 0.03, 0.00, 0.00],
          [0.00, 0.07, 0.14, 0.12, 0.05, 0.01, 0.00, 0.00]],

         [[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]],
    # 使用 NumPy 库中的 assert_array_almost_equal 函数来比较两个数组 rgb 和 expect 是否几乎相等
    assert_array_almost_equal(rgb, expect, decimal=2)
def test_light_source_hillshading():
    """
    Compare the current hillshading method against one that should be
    mathematically equivalent. Illuminates a cone from a range of angles.
    """

    def alternative_hillshade(azimuth, elev, z):
        # Convert azimuth and elevation to Cartesian coordinates
        illum = _sph2cart(*_azimuth2math(azimuth, elev))
        illum = np.array(illum)

        # Compute gradients of z (elevation data)
        dy, dx = np.gradient(-z)
        dy = -dy
        dz = np.ones_like(dy)
        normals = np.dstack([dx, dy, dz])
        normals /= np.linalg.norm(normals, axis=2)[..., None]

        # Compute intensity of illumination
        intensity = np.tensordot(normals, illum, axes=(2, 0))
        intensity -= intensity.min()
        intensity /= np.ptp(intensity)
        return intensity

    # Generate grid coordinates
    y, x = np.mgrid[5:0:-1, :5]
    z = -np.hypot(x - x.mean(), y - y.mean())

    # Test hillshading for various azimuth and elevation angles
    for az, elev in itertools.product(range(0, 390, 30), range(0, 105, 15)):
        ls = mcolors.LightSource(az, elev)
        h1 = ls.hillshade(z)
        h2 = alternative_hillshade(az, elev, z)
        assert_array_almost_equal(h1, h2)


def test_light_source_planar_hillshading():
    """
    Ensure that the illumination intensity is correct for planar surfaces.
    """

    def plane(azimuth, elevation, x, y):
        """
        Create a plane whose normal vector is at the given azimuth and
        elevation.
        """
        # Convert azimuth and elevation to Cartesian coordinates
        theta, phi = _azimuth2math(azimuth, elevation)
        a, b, c = _sph2cart(theta, phi)
        z = -(a*x + b*y) / c
        return z

    def angled_plane(azimuth, elevation, angle, x, y):
        """
        Create a plane whose normal vector is at an angle from the given
        azimuth and elevation.
        """
        # Adjust elevation and azimuth based on angle
        elevation = elevation + angle
        if elevation > 90:
            azimuth = (azimuth + 180) % 360
            elevation = (90 - elevation) % 90
        return plane(azimuth, elevation, x, y)

    # Generate grid coordinates
    y, x = np.mgrid[5:0:-1, :5]

    # Test planar hillshading for various azimuth, elevation, and angle combinations
    for az, elev in itertools.product(range(0, 390, 30), range(0, 105, 15)):
        ls = mcolors.LightSource(az, elev)

        # Make a plane at a range of angles to the illumination
        for angle in range(0, 105, 15):
            z = angled_plane(az, elev, angle, x, y)
            h = ls.hillshade(z)
            assert_array_almost_equal(h, np.cos(np.radians(angle)))


def test_color_names():
    # Test conversion of color names to hexadecimal values
    assert mcolors.to_hex("blue") == "#0000ff"
    assert mcolors.to_hex("xkcd:blue") == "#0343df"
    assert mcolors.to_hex("tab:blue") == "#1f77b4"


def _sph2cart(theta, phi):
    """
    Convert spherical coordinates (theta, phi) to Cartesian coordinates.
    """
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return x, y, z


def _azimuth2math(azimuth, elevation):
    """
    Convert azimuth (clockwise-from-north) and elevation (up-from-horizontal)
    to mathematical conventions (theta, phi).
    """
    theta = np.radians((90 - azimuth) % 360)
    phi = np.radians(90 - elevation)
    return theta, phi


def test_pandas_iterable(pd):
    # Ensure that using a list or series for colormaps yields equivalent results
    # i.e., the series isn't seen as a single color
    lst = ['red', 'blue', 'green']
    # 使用列表 lst 创建一个 Pandas Series 对象 s
    s = pd.Series(lst)
    
    # 使用列表 lst 创建一个包含 5 个颜色的 colormap 对象 cm1
    cm1 = mcolors.ListedColormap(lst, N=5)
    
    # 使用 Pandas Series 对象 s 创建一个包含 5 个颜色的 colormap 对象 cm2
    cm2 = mcolors.ListedColormap(s, N=5)
    
    # 断言两个 colormap 对象的颜色列表是否相等
    assert_array_equal(cm1.colors, cm2.colors)
# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_colormap_reversing 参数化，参数是排序后的所有 matplotlib 颜色映射的名称
@pytest.mark.parametrize('name', sorted(mpl.colormaps()))
def test_colormap_reversing(name):
    """
    检查颜色映射及其反转后的 _lut 数据是否几乎相同。
    """
    # 获取指定名称的颜色映射对象
    cmap = mpl.colormaps[name]
    # 获取该颜色映射对象的反转版本
    cmap_r = cmap.reversed()
    # 如果反转版本尚未初始化，则进行初始化
    if not cmap_r._isinit:
        cmap._init()
        cmap_r._init()
    # 断言颜色映射对象的 _lut 数据的部分与反转版本的 _lut 数据的部分是否几乎相等
    assert_array_almost_equal(cmap._lut[:-3], cmap_r._lut[-4::-1])
    # 断言特定边界值情况下颜色映射和其反转版本的输出是否几乎相等
    assert_array_almost_equal(cmap(-np.inf), cmap_r(np.inf))
    assert_array_almost_equal(cmap(np.inf), cmap_r(-np.inf))
    assert_array_almost_equal(cmap(np.nan), cmap_r(np.nan))


# 测试函数，用于验证颜色是否包含 alpha 通道
def test_has_alpha_channel():
    # 断言具有 RGBA 表示的颜色元组 (0, 0, 0, 0) 是否包含 alpha 通道
    assert mcolors._has_alpha_channel((0, 0, 0, 0))
    # 断言具有 RGBA 表示的列表 [1, 1, 1, 1] 是否包含 alpha 通道
    assert mcolors._has_alpha_channel([1, 1, 1, 1])
    # 断言字符串 'blue' 不包含 alpha 通道（4 字符的字符串）
    assert not mcolors._has_alpha_channel('blue')  # 4-char string!
    # 断言字符串 '0.25' 不包含 alpha 通道
    assert not mcolors._has_alpha_channel('0.25')
    # 断言字符串 'r' 不包含 alpha 通道
    assert not mcolors._has_alpha_channel('r')
    # 断言具有 RGB 表示的颜色元组 (1, 0, 0) 不包含 alpha 通道
    assert not mcolors._has_alpha_channel((1, 0, 0))


# 测试函数，用于验证 matplotlib 色彩转换
def test_cn():
    # 设置当前轴的属性循环，指定颜色循环为 ['blue', 'r']
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
                                                    ['blue', 'r'])
    # 断言颜色 'C0' 转换为十六进制后是否为 '#0000ff'
    assert mcolors.to_hex("C0") == '#0000ff'
    # 断言颜色 'C1' 转换为十六进制后是否为 '#ff0000'

    assert mcolors.to_hex("C1") == '#ff0000'

    # 修改当前轴的属性循环，指定颜色循环为 ['xkcd:blue', 'r']
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
                                                    ['xkcd:blue', 'r'])
    # 断言颜色 'C0' 转换为十六进制后是否为 '#0343df'
    assert mcolors.to_hex("C0") == '#0343df'
    # 断言颜色 'C1' 转换为十六进制后是否为 '#ff0000'
    assert mcolors.to_hex("C1") == '#ff0000'
    # 断言颜色 'C10' 转换为十六进制后是否为 '#0343df'
    assert mcolors.to_hex("C10") == '#0343df'
    # 断言颜色 'C11' 转换为十六进制后是否为 '#ff0000'

    # 修改当前轴的属性循环，指定颜色循环为 ['8e4585', 'r']
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['8e4585', 'r'])

    # 断言颜色 'C0' 转换为十六进制后是否为 '#8e4585'
    assert mcolors.to_hex("C0") == '#8e4585'
    # 如果 '8e4585' 在解析为十六进制颜色之前被解析为浮点数，将被视为一个非常大的数值。
    # 这种情况绝不能发生。
    assert mcolors.to_rgb("C0")[0] != np.inf


# 测试函数，用于验证颜色转换和处理
def test_conversions():
    # 断言将 "none" 转换为 RGBA 数组后是否返回 (0, 0, 0, 0)
    assert_array_equal(mcolors.to_rgba_array("none"), np.zeros((0, 4)))
    # 断言将空列表转换为 RGBA 数组后是否返回 (0, 0, 0, 0)
    assert_array_equal(mcolors.to_rgba_array([]), np.zeros((0, 4)))
    # 断言将灰度级列表 [".2", ".5", ".8"] 转换为 RGBA 数组后是否与单独转换结果堆叠的数组相等
    assert_array_equal(
        mcolors.to_rgba_array([".2", ".5", ".8"]),
        np.vstack([mcolors.to_rgba(c) for c in [".2", ".5", ".8"]]))
    # 断言将 RGB 颜色 (1, 1, 1) 转换为带有 0.5 透明度的 RGBA 颜色是否正确
    assert mcolors.to_rgba((1, 1, 1), .5) == (1, 1, 1, .5)
    # 断言将灰度级 ".1" 转换为带有 0.5 透明度的 RGBA 颜色是否正确
    assert mcolors.to_rgba(".1", .5) == (.1, .1, .1, .5)
    # 断言将 RGB 颜色 (.7, .7, .7) 转换为十六进制后是否为 "#b2b2b2"
    assert mcolors.to_hex((.7, .7, .7)) == "#b2b2b2"
    # 断言将十六进制颜色 "#1234abcd" 转换为 RGBA 后再转换回十六进制时是否与原始值相等
    hex_color = "#1234abcd"
    assert mcolors.to_hex(mcolors.to_rgba(hex_color), keep_alpha=True) == \
        hex_color


# 测试函数，用于验证带掩码数据的颜色转换
def test_conversions_masked():
    # 创建包含掩码数据的 numpy masked array
    x1 = np.ma.array(['k', 'b'], mask=[True, False])
    x2 = np.ma.array([[0, 0, 0, 1], [0, 0, 1, 1]])
    x2[0] = np.ma.masked
    # 断言掩码值 'k' 转换为 RGBA 颜色后是否为 (0, 0, 0, 0)
    assert mcolors.to_rgba(x1[0]) == (0, 0, 0, 0)
    # 使用 assert_array_equal 函数比较 mcolors.to_rgba_array(x1) 的返回值与预期结果是否相等
    assert_array_equal(mcolors.to_rgba_array(x1),
                       [[0, 0, 0, 0], [0, 0, 1, 1]])
    # 使用 assert_array_equal 函数比较 mcolors.to_rgba_array(x2) 的返回值与 mcolors.to_rgba_array(x1) 的返回值是否相等
    assert_array_equal(mcolors.to_rgba_array(x2), mcolors.to_rgba_array(x1))
# 测试将单个颜色名称转换为 RGBA 数组
def test_to_rgba_array_single_str():
    # 断言单个颜色名称转换为 RGBA 数组是否正确
    assert_array_equal(mcolors.to_rgba_array("red"), [(1, 0, 0, 1)])

    # 当输入的是单个字符颜色序列时，预期会引发 ValueError 异常，并匹配给定错误信息
    with pytest.raises(ValueError,
                       match="'rgb' is not a valid color value."):
        array = mcolors.to_rgba_array("rgb")


# 测试将包含两个颜色名称的元组转换为 RGBA 数组
def test_to_rgba_array_2tuple_str():
    # 预期的结果是一个包含两个 RGBA 数组的 numpy 数组
    expected = np.array([[0, 0, 0, 1], [1, 1, 1, 1]])
    assert_array_equal(mcolors.to_rgba_array(("k", "w")), expected)


# 测试带有 alpha 值数组的颜色转换
def test_to_rgba_array_alpha_array():
    # 当颜色数量与 alpha 数组的长度不匹配时，预期会引发 ValueError 异常
    with pytest.raises(ValueError, match="The number of colors must match"):
        mcolors.to_rgba_array(np.ones((5, 3), float), alpha=np.ones((2,)))
    # 指定 alpha 值数组，验证转换后的 RGBA 数组的 alpha 值是否正确
    alpha = [0.5, 0.6]
    c = mcolors.to_rgba_array(np.ones((2, 3), float), alpha=alpha)
    assert_array_equal(c[:, 3], alpha)
    c = mcolors.to_rgba_array(['r', 'g'], alpha=alpha)
    assert_array_equal(c[:, 3], alpha)


# 测试接受颜色和 alpha 值元组作为输入的颜色转换
def test_to_rgba_array_accepts_color_alpha_tuple():
    # 验证将 ('black', 0.9) 转换为 RGBA 数组是否正确
    assert_array_equal(
        mcolors.to_rgba_array(('black', 0.9)),
        [[0, 0, 0, 0.9]])


# 测试显式指定 alpha 值覆盖元组中指定的 alpha 值
def test_to_rgba_array_explicit_alpha_overrides_tuple_alpha():
    # 验证将 ('black', 0.9) 并显式指定 alpha 值为 0.5 后的 RGBA 数组是否正确
    assert_array_equal(
        mcolors.to_rgba_array(('black', 0.9), alpha=0.5),
        [[0, 0, 0, 0.5]])


# 测试接受颜色和 alpha 值元组列表或数组作为输入的颜色转换
def test_to_rgba_array_accepts_color_alpha_tuple_with_multiple_colors():
    # 验证将颜色数组或序列与 alpha 值 0.2 结合转换为 RGBA 数组是否正确
    color_array = np.array([[1., 1., 1., 1.], [0., 0., 1., 0.]])
    assert_array_equal(
        mcolors.to_rgba_array((color_array, 0.2)),
        [[1., 1., 1., 0.2], [0., 0., 1., 0.2]])

    color_sequence = [[1., 1., 1., 1.], [0., 0., 1., 0.]]
    assert_array_equal(
        mcolors.to_rgba_array((color_sequence, 0.4)),
        [[1., 1., 1., 0.4], [0., 0., 1., 0.4]])


# 测试当颜色元组中的 alpha 值超出合法范围时是否会引发异常
def test_to_rgba_array_error_with_color_invalid_alpha_tuple():
    with pytest.raises(ValueError, match="'alpha' must be between 0 and 1,"):
        mcolors.to_rgba_array(('black', 2.0))


# 使用参数化测试验证颜色和 alpha 值元组的输入是否正确转换为 RGBA
@pytest.mark.parametrize('rgba_alpha',
                         [('white', 0.5), ('#ffffff', 0.5), ('#ffffff00', 0.5),
                          ((1.0, 1.0, 1.0, 1.0), 0.5)])
def test_to_rgba_accepts_color_alpha_tuple(rgba_alpha):
    assert mcolors.to_rgba(rgba_alpha) == (1, 1, 1, 0.5)


# 测试显式指定 alpha 值是否会覆盖颜色和 alpha 值元组中指定的 alpha 值
def test_to_rgba_explicit_alpha_overrides_tuple_alpha():
    assert mcolors.to_rgba(('red', 0.1), alpha=0.9) == (1, 0, 0, 0.9)


# 测试当颜色元组中的 alpha 值超出合法范围时是否会引发异常
def test_to_rgba_error_with_color_invalid_alpha_tuple():
    with pytest.raises(ValueError, match="'alpha' must be between 0 and 1"):
        mcolors.to_rgba(('blue', 2.0))


# 使用参数化测试验证 ScalarMappable 对象将颜色数组转换为 RGBA 数组的正确性
@pytest.mark.parametrize("bytes", (True, False))
def test_scalarmappable_to_rgba(bytes):
    sm = cm.ScalarMappable()
    alpha_1 = 255 if bytes else 1

    # uint8 RGBA
    x = np.ones((2, 3, 4), dtype=np.uint8)
    expected = x.copy() if bytes else x.astype(np.float32)/255
    np.testing.assert_almost_equal(sm.to_rgba(x, bytes=bytes), expected)
    # uint8 RGB
    expected[..., 3] = alpha_1
    np.testing.assert_almost_equal(sm.to_rgba(x[..., :3], bytes=bytes), expected)
    # uint8 masked RGBA
    # 创建一个使用给定数组 `x` 创建的屏蔽数组，初始屏蔽所有元素
    xm = np.ma.masked_array(x, mask=np.zeros_like(x))
    # 将屏蔽数组中第一个元素的屏蔽位设置为 True
    xm.mask[0, 0, 0] = True
    # 如果 `bytes` 为真，则复制数组 `x`；否则将 `x` 转换为 `np.float32` 类型并归一化到 [0, 1]
    expected = x.copy() if bytes else x.astype(np.float32)/255
    # 将预期数组中第一个像素的 alpha 通道值设置为 0
    expected[0, 0, 3] = 0
    # 使用 `sm.to_rgba` 函数将屏蔽数组 `xm` 转换为 RGBA 表示，与预期值进行近似相等性检查
    np.testing.assert_almost_equal(sm.to_rgba(xm, bytes=bytes), expected)
    
    # uint8 masked RGB
    # 将预期数组的所有像素的 alpha 通道值设置为 `alpha_1`
    expected[..., 3] = alpha_1
    # 将预期数组中第一个像素的 alpha 通道值设置为 0
    expected[0, 0, 3] = 0
    # 使用 `sm.to_rgba` 函数将屏蔽数组 `xm` 的 RGB 部分转换为 RGBA 表示，与预期值进行近似相等性检查
    np.testing.assert_almost_equal(sm.to_rgba(xm[..., :3], bytes=bytes), expected)
    
    # float RGBA
    # 创建一个包含全部元素为 0.5 的浮点型数组 `x`
    x = np.ones((2, 3, 4), dtype=float) * 0.5
    # 如果 `bytes` 为真，则将数组 `x` 缩放到 [0, 255] 范围并转换为 `np.uint8` 类型；否则复制 `x`
    expected = (x * 255).astype(np.uint8) if bytes else x.copy()
    # 使用 `sm.to_rgba` 函数将数组 `x` 转换为 RGBA 表示，与预期值进行近似相等性检查
    np.testing.assert_almost_equal(sm.to_rgba(x, bytes=bytes), expected)
    
    # float RGB
    # 将预期数组的所有像素的 alpha 通道值设置为 `alpha_1`
    expected[..., 3] = alpha_1
    # 使用 `sm.to_rgba` 函数将数组 `x` 的 RGB 部分转换为 RGBA 表示，与预期值进行近似相等性检查
    np.testing.assert_almost_equal(sm.to_rgba(x[..., :3], bytes=bytes), expected)
    
    # float masked RGBA
    # 创建一个使用给定数组 `x` 创建的屏蔽数组，初始屏蔽所有元素
    xm = np.ma.masked_array(x, mask=np.zeros_like(x))
    # 将屏蔽数组中第一个元素的屏蔽位设置为 True
    xm.mask[0, 0, 0] = True
    # 如果 `bytes` 为真，则将数组 `x` 缩放到 [0, 255] 范围并转换为 `np.uint8` 类型；否则复制 `x`
    expected = (x * 255).astype(np.uint8) if bytes else x.copy()
    # 将预期数组中第一个像素的 alpha 通道值设置为 0
    expected[0, 0, 3] = 0
    # 使用 `sm.to_rgba` 函数将屏蔽数组 `xm` 转换为 RGBA 表示，与预期值进行近似相等性检查
    np.testing.assert_almost_equal(sm.to_rgba(xm, bytes=bytes), expected)
    
    # float masked RGB
    # 将预期数组的所有像素的 alpha 通道值设置为 `alpha_1`
    expected[..., 3] = alpha_1
    # 将预期数组中第一个像素的 alpha 通道值设置为 0
    expected[0, 0, 3] = 0
    # 使用 `sm.to_rgba` 函数将屏蔽数组 `xm` 的 RGB 部分转换为 RGBA 表示，与预期值进行近似相等性检查
    np.testing.assert_almost_equal(sm.to_rgba(xm[..., :3], bytes=bytes), expected)
@pytest.mark.parametrize("bytes", (True, False))
# 使用 pytest 的参数化装饰器，定义一个名为 test_scalarmappable_nan_to_rgba 的测试函数，参数为 bytes
def test_scalarmappable_nan_to_rgba(bytes):
    sm = cm.ScalarMappable()
    # 创建一个 ScalarMappable 对象

    # RGBA
    x = np.ones((2, 3, 4), dtype=float) * 0.5
    # 创建一个形状为 (2, 3, 4) 的浮点数数组 x，所有元素初始化为 0.5
    x[0, 0, 0] = np.nan
    # 将 x 的第一个元素设为 NaN
    expected = x.copy()
    # 复制 x 到 expected
    expected[0, 0, :] = 0
    # 将 expected 的第一个元素的 RGBA 值设为 0
    if bytes:
        expected = (expected * 255).astype(np.uint8)
        # 如果 bytes 为 True，则将 expected 数组乘以 255 并转换为无符号字节类型
    np.testing.assert_almost_equal(sm.to_rgba(x, bytes=bytes), expected)
    # 使用 ScalarMappable 对象的 to_rgba 方法处理数组 x，与期望结果 expected 近似比较
    assert np.any(np.isnan(x))  # Input array should not be changed
    # 断言 x 中至少有一个 NaN 值，验证输入数组不应被更改

    # RGB
    expected[..., 3] = 255 if bytes else 1
    # 将 expected 数组的所有行、所有列、第四个通道的值设为 255（如果 bytes 为 True），否则设为 1
    expected[0, 0, 3] = 0
    # 将 expected 数组的第一个元素的第四个通道的值设为 0
    np.testing.assert_almost_equal(sm.to_rgba(x[..., :3], bytes=bytes), expected)
    # 使用 ScalarMappable 对象的 to_rgba 方法处理数组 x 的前三个通道，与期望结果 expected 近似比较
    assert np.any(np.isnan(x))  # Input array should not be changed
    # 断言 x 中至少有一个 NaN 值，验证输入数组不应被更改

    # Out-of-range fail
    x[1, 0, 0] = 42
    # 修改 x 数组的第二行第一列第一通道的值为 42
    with pytest.raises(ValueError, match='0..1 range'):
        sm.to_rgba(x[..., :3], bytes=bytes)
    # 使用 ScalarMappable 对象的 to_rgba 方法处理数组 x 的前三个通道，预期会引发 ValueError 异常，异常消息包含 '0..1 range'
    # 调用 mcolors 模块中的 same_color 函数，断言 ['red', 'blue'] 和 ['r', 'b'] 是相同颜色
    assert mcolors.same_color(['red', 'blue'], ['r', 'b'])
    
    # 调用 mcolors 模块中的 same_color 函数，断言 'none' 和 'none' 是相同颜色
    assert mcolors.same_color('none', 'none')
    
    # 调用 mcolors 模块中的 same_color 函数，断言 'none' 和 'red' 不是相同颜色
    assert not mcolors.same_color('none', 'red')
    
    # 使用 pytest 的异常断言，验证调用 mcolors 模块中的 same_color 函数时，传入参数不合法（期望引发 ValueError 异常）
    with pytest.raises(ValueError):
        mcolors.same_color(['r', 'g', 'b'], ['r'])
    
    # 使用 pytest 的异常断言，验证调用 mcolors 模块中的 same_color 函数时，传入参数不合法（期望引发 ValueError 异常）
    with pytest.raises(ValueError):
        mcolors.same_color(['red', 'green'], 'none')
def test_hex_shorthand_notation():
    # 测试同色函数是否正确处理十六进制简写表示法的颜色
    assert mcolors.same_color("#123", "#112233")
    assert mcolors.same_color("#123a", "#112233aa")


def test_repr_png():
    # 获取'viridis'色图的 PNG 表示
    cmap = mpl.colormaps['viridis']
    png = cmap._repr_png_()
    # 确保 PNG 数据非空
    assert len(png) > 0
    # 从 PNG 数据创建图像对象
    img = Image.open(BytesIO(png))
    # 确保图像宽度和高度大于零
    assert img.width > 0
    assert img.height > 0
    # 确保图像中包含特定文本信息
    assert 'Title' in img.text
    assert 'Description' in img.text
    assert 'Author' in img.text
    assert 'Software' in img.text


def test_repr_html():
    # 获取'viridis'色图的 HTML 表示
    cmap = mpl.colormaps['viridis']
    html = cmap._repr_html_()
    # 确保 HTML 数据非空
    assert len(html) > 0
    # 获取色图的 PNG 表示
    png = cmap._repr_png_()
    # 确保 PNG 数据在 HTML 中以 Base64 编码形式存在
    assert base64.b64encode(png).decode('ascii') in html
    # 确保 HTML 以 '<div' 开头和'</div>'结尾
    assert html.startswith('<div')
    assert html.endswith('</div>')


def test_get_under_over_bad():
    # 测试获取色图 'viridis' 中的特殊值：下限、上限和不合法值
    cmap = mpl.colormaps['viridis']
    assert_array_equal(cmap.get_under(), cmap(-np.inf))
    assert_array_equal(cmap.get_over(), cmap(np.inf))
    assert_array_equal(cmap.get_bad(), cmap(np.nan))


@pytest.mark.parametrize('kind', ('over', 'under', 'bad'))
def test_non_mutable_get_values(kind):
    # 测试获取色图 'viridis' 中特定类型值的不可变性
    cmap = copy.copy(mpl.colormaps['viridis'])
    # 获取初始值
    init_value = getattr(cmap, f'get_{kind}')()
    # 设置特定类型的值为黑色
    getattr(cmap, f'set_{kind}')('k')
    # 获取设置后的值
    black_value = getattr(cmap, f'get_{kind}')()
    # 确保黑色值为 [0, 0, 0, 1]
    assert np.all(black_value == [0, 0, 0, 1])
    # 确保设置前后值不同
    assert not np.all(init_value == black_value)


def test_colormap_alpha_array():
    # 测试色图 'viridis' 处理带有 alpha 通道数组的数据
    cmap = mpl.colormaps['viridis']
    vals = [-1, 0.5, 2]  # under, valid, over
    # 确保当 alpha 是数组形式但是不支持时会抛出 ValueError 异常
    with pytest.raises(ValueError, match="alpha is array-like but"):
        cmap(vals, alpha=[1, 1, 1, 1])
    # 设置 alpha 通道数组并检查结果
    alpha = np.array([0.1, 0.2, 0.3])
    c = cmap(vals, alpha=alpha)
    assert_array_equal(c[:, -1], alpha)
    c = cmap(vals, alpha=alpha, bytes=True)
    assert_array_equal(c[:, -1], (alpha * 255).astype(np.uint8))


def test_colormap_bad_data_with_alpha():
    # 测试色图 'viridis' 处理带有不合法数据和 alpha 通道的情况
    cmap = mpl.colormaps['viridis']
    # 确保处理 np.nan 时返回黑色 (0, 0, 0, 0)
    c = cmap(np.nan, alpha=0.5)
    assert c == (0, 0, 0, 0)
    c = cmap([0.5, np.nan], alpha=0.5)
    assert_array_equal(c[1], (0, 0, 0, 0))
    c = cmap([0.5, np.nan], alpha=[0.1, 0.2])
    assert_array_equal(c[1], (0, 0, 0, 0))
    c = cmap([[np.nan, 0.5], [0, 0]], alpha=0.5)
    assert_array_equal(c[0, 0], (0, 0, 0, 0))
    c = cmap([[np.nan, 0.5], [0, 0]], alpha=np.full((2, 2), 0.5))
    assert_array_equal(c[0, 0], (0, 0, 0, 0))


def test_2d_to_rgba():
    # 测试颜色数组转换成 RGBA 表示，1维和2维应该一致
    color = np.array([0.1, 0.2, 0.3])
    rgba_1d = mcolors.to_rgba(color.reshape(-1))
    rgba_2d = mcolors.to_rgba(color.reshape((1, -1)))
    assert rgba_1d == rgba_2d


def test_set_dict_to_rgba():
    # downstream libraries do this...
    # note we can't test this because it is not well-ordered
    # so just smoketest:
    # 将颜色字典转换为 RGBA 数组
    colors = {(0, .5, 1), (1, .2, .5), (.4, 1, .2)}
    res = mcolors.to_rgba_array(colors)
    palette = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}
    res = mcolors.to_rgba_array(palette.values())
    exp = np.eye(3)
    np.testing.assert_array_almost_equal(res[:, :-1], exp)
def test_norm_deepcopy():
    # 创建一个对数标准化对象
    norm = mcolors.LogNorm()
    # 设置最小值为0.0002
    norm.vmin = 0.0002
    # 深度复制标准化对象norm
    norm2 = copy.deepcopy(norm)
    # 断言深度复制后的最小值与原始对象相同
    assert norm2.vmin == norm.vmin
    # 断言深度复制后的_scale属性是LogScale类型的实例
    assert isinstance(norm2._scale, mscale.LogScale)
    
    # 创建一个普通标准化对象
    norm = mcolors.Normalize()
    # 设置最小值为0.0002
    norm.vmin = 0.0002
    # 再次深度复制标准化对象norm
    norm2 = copy.deepcopy(norm)
    # 断言深度复制后的_scale属性为None
    assert norm2._scale is None
    # 断言深度复制后的最小值与原始对象相同
    assert norm2.vmin == norm.vmin


def test_norm_callback():
    # 创建一个模拟对象increment
    increment = unittest.mock.Mock(return_value=None)

    # 创建一个普通标准化对象
    norm = mcolors.Normalize()
    # 将increment连接到标准化对象的'changed'信号
    norm.callbacks.connect('changed', increment)
    # 断言increment的调用次数为0，因为还未更新任何内容
    assert increment.call_count == 0

    # 修改vmin和vmax来测试回调函数
    norm.vmin = 1
    assert increment.call_count == 1
    norm.vmax = 5
    assert increment.call_count == 2
    # 如果设置为相同的值，则回调不应该被调用
    norm.vmin = 1
    assert increment.call_count == 2
    norm.vmax = 5
    assert increment.call_count == 2

    # 我们只希望autoscale()调用发送一个更新信号
    increment.call_count = 0
    norm.autoscale([0, 1, 2])
    assert increment.call_count == 1


def test_scalarmappable_norm_update():
    # 创建一个普通标准化对象
    norm = mcolors.Normalize()
    # 创建一个标量映射对象sm，使用'plasma'颜色映射
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='plasma')
    # 初始时sm没有stale属性，将其设置为False
    sm.stale = False
    # 更新vmin/vmax后，标量映射应该变为stale
    norm.vmin = 5
    assert sm.stale
    sm.stale = False
    norm.vmax = 5
    assert sm.stale
    sm.stale = False
    norm.clip = True
    assert sm.stale
    
    # 切换到CenteredNorm和TwoSlopeNorm来测试它们
    # 同时确保直接更新norm和使用set_norm都会更新Norm回调
    norm = mcolors.CenteredNorm()
    sm.norm = norm
    sm.stale = False
    norm.vcenter = 1
    assert sm.stale
    
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    sm.set_norm(norm)
    sm.stale = False
    norm.vcenter = 1
    assert sm.stale


@check_figures_equal()
def test_norm_update_figs(fig_test, fig_ref):
    # 在参考图上添加子图ax_ref
    ax_ref = fig_ref.add_subplot()
    # 在测试图上添加子图ax_test
    ax_test = fig_test.add_subplot()

    z = np.arange(100).reshape((10, 10))
    # 在ax_ref上显示图像，使用范围为[10, 90]的Normalize对象
    ax_ref.imshow(z, norm=mcolors.Normalize(10, 90))

    # 先创建一个Normalize对象，设置不同的限制，然后在添加到图上后再更新
    norm = mcolors.Normalize(0, 1)
    ax_test.imshow(z, norm=norm)
    # 强制进行初始绘制，确保不是已经过时的状态
    fig_test.canvas.draw()
    # 更新norm的vmin和vmax为10和90
    norm.vmin, norm.vmax = 10, 90


def test_make_norm_from_scale_name():
    # 使用make_norm_from_scale创建一个LogitScale对应的Normalize对象
    logitnorm = mcolors.make_norm_from_scale(
        mscale.LogitScale, mcolors.Normalize)
    # 断言logitnorm的名称和限定名称都是"LogitScaleNorm"
    assert logitnorm.__name__ == logitnorm.__qualname__ == "LogitScaleNorm"


def test_color_sequences():
    # 基本访问，断言plt.color_sequences与matplotlib.color_sequences是相同的注册表
    assert plt.color_sequences is matplotlib.color_sequences  # same registry
    # 断言检查 plt.color_sequences 中的颜色序列是否按预期顺序排列
    assert list(plt.color_sequences) == [
        'tab10', 'tab20', 'tab20b', 'tab20c', 'Pastel1', 'Pastel2', 'Paired',
        'Accent', 'Dark2', 'Set1', 'Set2', 'Set3']
    
    # 断言检查 'tab10' 颜色序列的长度是否为 10
    assert len(plt.color_sequences['tab10']) == 10
    
    # 断言检查 'tab20' 颜色序列的长度是否为 20
    assert len(plt.color_sequences['tab20']) == 20
    
    # 创建一个包含预定义颜色 'tab:blue' 到 'tab:cyan' 的列表 tab_colors
    tab_colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    # 遍历 'tab10' 颜色序列和 tab_colors 列表，确保颜色匹配
    for seq_color, tab_color in zip(plt.color_sequences['tab10'], tab_colors):
        assert mcolors.same_color(seq_color, tab_color)
    
    # 使用 pytest 检查注册 'tab10' 序列时是否引发 ValueError 异常，且异常信息包含 "reserved name"
    with pytest.raises(ValueError, match="reserved name"):
        plt.color_sequences.register('tab10', ['r', 'g', 'b'])
    
    # 使用 pytest 检查注册 'invalid' 序列时是否引发 ValueError 异常，且异常信息包含 "not a valid color specification"
    with pytest.raises(ValueError, match="not a valid color specification"):
        plt.color_sequences.register('invalid', ['not a color'])
    
    # 定义一个 RGB 颜色列表 rgb_colors
    rgb_colors = ['r', 'g', 'b']
    
    # 注册 'rgb' 序列为 rgb_colors
    plt.color_sequences.register('rgb', rgb_colors)
    
    # 断言检查 'rgb' 序列是否与预期的 rgb_colors 相符
    assert plt.color_sequences['rgb'] == ['r', 'g', 'b']
    
    # 确保修改 rgb_colors 不会影响注册的 'rgb' 序列
    rgb_colors.append('c')
    assert plt.color_sequences['rgb'] == ['r', 'g', 'b']
    
    # 确保通过 plt.color_sequences['rgb'] 返回的列表是一个副本，不会因修改而改变注册的 'rgb' 序列
    plt.color_sequences['rgb'].append('c')
    assert plt.color_sequences['rgb'] == ['r', 'g', 'b']
    
    # 反注册 'rgb' 序列
    plt.color_sequences.unregister('rgb')
    
    # 使用 pytest 确保 'rgb' 序列已经被反注册，会引发 KeyError 异常
    with pytest.raises(KeyError):
        plt.color_sequences['rgb']  # rgb is gone
    
    # 多次反注册 'rgb' 序列不会引发异常
    plt.color_sequences.unregister('rgb')
    
    # 使用 pytest 确保无法反注册内置序列 'tab10'，会引发 ValueError 异常
    with pytest.raises(ValueError, match="Cannot unregister builtin"):
        plt.color_sequences.unregister('tab10')
def test_cm_set_cmap_error():
    # 创建一个标量映射对象
    sm = cm.ScalarMappable()
    # 指定一个几乎肯定不存在的颜色映射名
    bad_cmap = 'AardvarksAreAwkward'
    # 使用 pytest 的断言检查是否会引发 ValueError，并匹配指定错误信息
    with pytest.raises(ValueError, match=bad_cmap):
        sm.set_cmap(bad_cmap)


def test_set_cmap_mismatched_name():
    # 从 matplotlib 中获取名为 "viridis" 的颜色映射，并添加异常值
    cmap = matplotlib.colormaps["viridis"].with_extremes(over='r')
    # 将该颜色映射注册为不同的名称
    cmap.name = "test-cmap"
    matplotlib.colormaps.register(name='wrong-cmap', cmap=cmap)

    # 设置当前图形的颜色映射为 "wrong-cmap"
    plt.set_cmap("wrong-cmap")
    # 获取当前图形的颜色映射，期望返回刚注册的 cmap
    cmap_returned = plt.get_cmap("wrong-cmap")
    # 使用断言检查返回的颜色映射是否与注册的 cmap 相同
    assert cmap_returned == cmap
    # 使用断言检查返回的颜色映射的名称是否为 "wrong-cmap"
    assert cmap_returned.name == "wrong-cmap"


def test_cmap_alias_names():
    # 使用断言检查 "gray" 是否为原始名称
    assert matplotlib.colormaps["gray"].name == "gray"  # original
    # 使用断言检查 "grey" 是否为 "gray" 的别名
    assert matplotlib.colormaps["grey"].name == "grey"  # alias


def test_to_rgba_array_none_color_with_alpha_param():
    # 对于颜色为 "none" 的情况，期望的有效 alpha 值始终为 0，实现颜色的消失
    # 即使指定了显式的 alpha 值，也应该被忽略
    c = ["blue", "none"]
    alpha = [1, 1]
    # 使用断言检查转换后的 RGBA 数组是否符合预期
    assert_array_equal(
        to_rgba_array(c, alpha), [[0., 0., 1., 1.], [0., 0., 0., 0.]]
    )


@pytest.mark.parametrize('input, expected',
                         [('red', True),
                          (('red', 0.5), True),
                          (('red', 2), False),
                          (['red', 0.5], False),
                          (('red', 'blue'), False),
                          (['red', 'blue'], False),
                          ('C3', True),
                          (('C3', 0.5), True)])
def test_is_color_like(input, expected):
    # 使用断言检查 is_color_like 函数对输入的颜色值是否返回预期的结果
    assert is_color_like(input) is expected
```