# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_streamplot.py`

```
# 导入 NumPy 库，并将其命名为 np
import numpy as np
# 导入 assert_array_almost_equal 函数，用于测试数组近似相等性
from numpy.testing import assert_array_almost_equal
# 导入 pytest 库，用于编写和运行测试
import pytest
# 导入 matplotlib.pyplot 库，并将其命名为 plt，用于绘图
import matplotlib.pyplot as plt
# 导入 image_comparison 装饰器，用于比较图像输出
from matplotlib.testing.decorators import image_comparison
# 导入 matplotlib.transforms 模块，用于图形变换操作
import matplotlib.transforms as mtransforms


# 定义函数 velocity_field，生成二维速度场的坐标和分量
def velocity_field():
    # 创建 Y, X 网格，范围为 [-3, 3]，分别包含100个和200个点
    Y, X = np.mgrid[-3:3:100j, -3:3:200j]
    # 计算速度场 U 和 V
    U = -1 - X**2 + Y
    V = 1 + X - Y**2
    return X, Y, U, V


# 定义函数 swirl_velocity_field，生成漩涡流速度场的坐标和分量
def swirl_velocity_field():
    # 创建 x 和 y 线性空间，范围为 [-3, 3]，包含200个和100个点
    x = np.linspace(-3., 3., 200)
    y = np.linspace(-3., 3., 100)
    # 创建 X, Y 网格
    X, Y = np.meshgrid(x, y)
    # 设置参数 a = 0.1
    a = 0.1
    # 计算漩涡流速度场 U 和 V
    U = np.cos(a) * (-Y) - np.sin(a) * X
    V = np.sin(a) * (-Y) + np.cos(a) * X
    return x, y, U, V


# 使用 image_comparison 装饰器定义测试函数 test_startpoints
@image_comparison(['streamplot_startpoints'], remove_text=True, style='mpl20',
                  extensions=['png'])
def test_startpoints():
    # 调用 velocity_field 函数，获取速度场的坐标和分量
    X, Y, U, V = velocity_field()
    # 创建起始点网格，5x5 的网格覆盖整个速度场
    start_x, start_y = np.meshgrid(np.linspace(X.min(), X.max(), 5),
                                   np.linspace(Y.min(), Y.max(), 5))
    start_points = np.column_stack([start_x.ravel(), start_y.ravel()])
    # 绘制流线图，起始点为 start_points
    plt.streamplot(X, Y, U, V, start_points=start_points)
    # 绘制起始点
    plt.plot(start_x, start_y, 'ok')


# 使用 image_comparison 装饰器定义测试函数 test_colormap
@image_comparison(['streamplot_colormap'], remove_text=True, style='mpl20',
                  tol=0.022)
def test_colormap():
    # 调用 velocity_field 函数，获取速度场的坐标和分量
    X, Y, U, V = velocity_field()
    # 绘制流线图，颜色基于速度分量 U，使用 autumn 颜色映射，密度为 0.6，线宽为 2
    plt.streamplot(X, Y, U, V, color=U, density=0.6, linewidth=2,
                   cmap=plt.cm.autumn)
    # 添加颜色条
    plt.colorbar()


# 使用 image_comparison 装饰器定义测试函数 test_linewidth
@image_comparison(['streamplot_linewidth'], remove_text=True, style='mpl20',
                  tol=0.004)
def test_linewidth():
    # 调用 velocity_field 函数，获取速度场的坐标和分量
    X, Y, U, V = velocity_field()
    # 计算速度大小
    speed = np.hypot(U, V)
    # 根据速度大小计算线宽
    lw = 5 * speed / speed.max()
    # 创建子图对象 ax
    ax = plt.figure().subplots()
    # 绘制流线图，密度为 [0.5, 1]，颜色为黑色，线宽为 lw
    ax.streamplot(X, Y, U, V, density=[0.5, 1], color='k', linewidth=lw)


# 使用 image_comparison 装饰器定义测试函数 test_masks_and_nans
@image_comparison(['streamplot_masks_and_nans'],
                  remove_text=True, style='mpl20')
def test_masks_and_nans():
    # 调用 velocity_field 函数，获取速度场的坐标和分量
    X, Y, U, V = velocity_field()
    # 创建布尔类型的掩码数组 mask
    mask = np.zeros(U.shape, dtype=bool)
    # 将部分区域设为 True，表示掩盖该区域的流线
    mask[40:60, 80:120] = 1
    # 将部分速度场设置为 NaN
    U[:20, :40] = np.nan
    # 使用掩码数组创建掩盖了 NaN 值的 U 数组
    U = np.ma.array(U, mask=mask)
    # 创建子图对象 ax
    ax = plt.figure().subplots()
    # 绘制流线图，颜色基于 U 数组，使用蓝色色调
    with np.errstate(invalid='ignore'):
        ax.streamplot(X, Y, U, V, color=U, cmap=plt.cm.Blues)


# 使用 image_comparison 装饰器定义测试函数 test_maxlength
@image_comparison(['streamplot_maxlength.png'],
                  remove_text=True, style='mpl20', tol=0.302)
def test_maxlength():
    # 调用 swirl_velocity_field 函数，获取漩涡流速度场的坐标和分量
    x, y, U, V = swirl_velocity_field()
    # 创建子图对象 ax
    ax = plt.figure().subplots()
    # 绘制漩涡流的流线图，最大长度为 10，起始点为 [0., 1.5]
    ax.streamplot(x, y, U, V, maxlength=10., start_points=[[0., 1.5]],
                  linewidth=2, density=2)
    # 断言坐标轴的限制范围
    assert ax.get_xlim()[-1] == ax.get_ylim()[-1] == 3
    # 兼容旧测试图像的设置
    ax.set(xlim=(None, 3.2555988021882305), ylim=(None, 3.078326760195413))


# 使用 image_comparison 装饰器定义测试函数 test_maxlength_no_broken
@image_comparison(['streamplot_maxlength_no_broken.png'],
                  remove_text=True, style='mpl20', tol=0.302)
def test_maxlength_no_broken():
    # 调用 swirl_velocity_field 函数，获取漩涡流速度场的坐标和分量
    x, y, U, V = swirl_velocity_field()
    # 创建子图对象 ax
    ax = plt.figure().subplots()
    # 绘制漩涡流的流线图，最大长度为 10，起始点为 [0., 1.5]，不断裂的流线
    ax.streamplot(x, y, U, V, maxlength=10., start_points=[[0., 1.5]],
                  linewidth=2, density=2, broken_streamlines=False)
    # 断言检查 x 轴和 y 轴的最大限制是否为 3
    assert ax.get_xlim()[-1] == ax.get_ylim()[-1] == 3
    # 为了兼容旧的测试图像，设置新的 x 轴和 y 轴的限制
    ax.set(xlim=(None, 3.2555988021882305), ylim=(None, 3.078326760195413))
# 使用图像比较测试装饰器，比较生成的图像与预期图像是否一致
@image_comparison(['streamplot_direction.png'],
                  remove_text=True, style='mpl20', tol=0.073)
def test_direction():
    # 获取旋涡流场的坐标和速度场数据
    x, y, U, V = swirl_velocity_field()
    # 绘制流线图，设置反向积分方向、最大长度为1.5、起始点为[1.5, 0.]
    plt.streamplot(x, y, U, V, integration_direction='backward',
                   maxlength=1.5, start_points=[[1.5, 0.]],
                   linewidth=2, density=2)


def test_streamplot_limits():
    # 在图上创建坐标轴
    ax = plt.axes()
    # 在指定范围内生成均匀分布的数据点
    x = np.linspace(-5, 10, 20)
    y = np.linspace(-2, 4, 10)
    # 生成网格，用于绘制矢量场
    y, x = np.meshgrid(y, x)
    # 创建仿射变换对象，用于在数据坐标系上进行平移
    trans = mtransforms.Affine2D().translate(25, 32) + ax.transData
    # 绘制矢量场，设置变换为之前定义的仿射变换
    plt.barbs(x, y, np.sin(x), np.cos(y), transform=trans)
    # 断言计算的数据界限与原始数据的界限大致相等
    # 这是因为在更新数据界限时考虑了整个路径。
    assert_array_almost_equal(ax.dataLim.bounds, (20, 30, 15, 6),
                              decimal=1)


def test_streamplot_grid():
    u = np.ones((2, 2))
    v = np.zeros((2, 2))

    # 测试行和列数相同的情况
    x = np.array([[10, 20], [10, 30]])
    y = np.array([[10, 10], [20, 20]])
    with pytest.raises(ValueError, match="The rows of 'x' must be equal"):
        plt.streamplot(x, y, u, v)

    x = np.array([[10, 20], [10, 20]])
    y = np.array([[10, 10], [20, 30]])
    with pytest.raises(ValueError, match="The columns of 'y' must be equal"):
        plt.streamplot(x, y, u, v)

    # 绘制流线图，行和列数相同
    x = np.array([[10, 20], [10, 20]])
    y = np.array([[10, 10], [20, 20]])
    plt.streamplot(x, y, u, v)

    # 测试最大维度
    x = np.array([0, 10])
    y = np.array([[[0, 10]]])
    with pytest.raises(ValueError, match="'y' can have at maximum "
                                         "2 dimensions"):
        plt.streamplot(x, y, u, v)

    # 测试等间距性
    u = np.ones((3, 3))
    v = np.zeros((3, 3))
    x = np.array([0, 10, 20])
    y = np.array([0, 10, 30])
    with pytest.raises(ValueError, match="'y' values must be equally spaced"):
        plt.streamplot(x, y, u, v)

    # 测试严格递增
    x = np.array([0, 20, 40])
    y = np.array([0, 20, 10])
    with pytest.raises(ValueError, match="'y' must be strictly increasing"):
        plt.streamplot(x, y, u, v)


def test_streamplot_inputs():  # test no exception occurs.
    # 绘制流线图，使用完全掩码数据
    plt.streamplot(np.arange(3), np.arange(3),
                   np.full((3, 3), np.nan), np.full((3, 3), np.nan),
                   color=np.random.rand(3, 3))
    # 绘制流线图，使用数组
    plt.streamplot(range(3), range(3),
                   np.random.rand(3, 3), np.random.rand(3, 3))
```