# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_quiver.py`

```
# 导入 platform 和 sys 模块
import platform
import sys

# 导入 numpy 库并命名为 np
import numpy as np
# 导入 pytest 库
import pytest

# 从 matplotlib 库中导入 pyplot 模块
from matplotlib import pyplot as plt
# 从 matplotlib.testing.decorators 模块中导入 image_comparison 装饰器
from matplotlib.testing.decorators import image_comparison


# 定义函数 draw_quiver，绘制矢量场图
def draw_quiver(ax, **kwargs):
    # 创建网格 X, Y
    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, 1),
                       np.arange(0, 2 * np.pi, 1))
    # 计算矢量场的 U 和 V 分量
    U = np.cos(X)
    V = np.sin(Y)

    # 在指定坐标轴上绘制矢量场
    Q = ax.quiver(U, V, **kwargs)
    return Q


# 使用 pytest 装饰器标记的测试函数，检测矢量场内存泄漏
@pytest.mark.skipif(platform.python_implementation() != 'CPython',
                    reason='Requires CPython')
def test_quiver_memory_leak():
    # 创建图形和坐标轴
    fig, ax = plt.subplots()

    # 绘制矢量场并获取返回的对象 Q
    Q = draw_quiver(ax)
    # 记录 Q 的 X 属性
    ttX = Q.X
    # 移除对象 Q
    Q.remove()

    # 删除 Q 对象的引用
    del Q

    # 断言 ttX 的引用计数为 2
    assert sys.getrefcount(ttX) == 2


# 使用 pytest 装饰器标记的测试函数，检测矢量场关键字参数的内存泄漏
@pytest.mark.skipif(platform.python_implementation() != 'CPython',
                    reason='Requires CPython')
def test_quiver_key_memory_leak():
    # 创建图形和坐标轴
    fig, ax = plt.subplots()

    # 绘制矢量场并获取返回的对象 Q
    Q = draw_quiver(ax)

    # 在坐标轴上添加矢量场关键信息
    qk = ax.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$',
                      labelpos='W',
                      fontproperties={'weight': 'bold'})
    
    # 断言 qk 对象的引用计数为 3
    assert sys.getrefcount(qk) == 3
    
    # 移除对象 qk
    qk.remove()
    
    # 再次断言 qk 对象的引用计数为 2
    assert sys.getrefcount(qk) == 2


# 检测 quiver 函数的参数个数是否符合预期
def test_quiver_number_of_args():
    X = [1, 2]
    # 断言调用 plt.quiver(X) 时抛出 TypeError 异常
    with pytest.raises(
            TypeError,
            match='takes from 2 to 5 positional arguments but 1 were given'):
        plt.quiver(X)
    
    # 断言调用 plt.quiver(X, X, X, X, X, X) 时抛出 TypeError 异常
    with pytest.raises(
            TypeError,
            match='takes from 2 to 5 positional arguments but 6 were given'):
        plt.quiver(X, X, X, X, X, X)


# 检测 quiver 函数的参数大小是否符合预期
def test_quiver_arg_sizes():
    X2 = [1, 2]
    X3 = [1, 2, 3]
    
    # 断言调用 plt.quiver(X2, X3, X2, X2) 时抛出 ValueError 异常
    with pytest.raises(
            ValueError, match=('X and Y must be the same size, but '
                               'X.size is 2 and Y.size is 3.')):
        plt.quiver(X2, X3, X2, X2)
    
    # 断言调用 plt.quiver(X2, X2, X3, X2) 时抛出 ValueError 异常
    with pytest.raises(
            ValueError, match=('Argument U has a size 3 which does not match '
                               '2, the number of arrow positions')):
        plt.quiver(X2, X2, X3, X2)
    
    # 断言调用 plt.quiver(X2, X2, X2, X3) 时抛出 ValueError 异常
    with pytest.raises(
            ValueError, match=('Argument V has a size 3 which does not match '
                               '2, the number of arrow positions')):
        plt.quiver(X2, X2, X2, X3)
    
    # 断言调用 plt.quiver(X2, X2, X2, X2, X3) 时抛出 ValueError 异常
    with pytest.raises(
            ValueError, match=('Argument C has a size 3 which does not match '
                               '2, the number of arrow positions')):
        plt.quiver(X2, X2, X2, X2, X3)


# 检测绘制矢量场时是否产生警告
def test_no_warnings():
    # 创建图形和坐标轴
    fig, ax = plt.subplots()
    # 创建 X, Y 网格
    X, Y = np.meshgrid(np.arange(15), np.arange(10))
    U = V = np.ones_like(X)
    # 创建随机角度数据
    phi = (np.random.rand(15, 10) - .5) * 150
    # 绘制矢量场
    ax.quiver(X, Y, U, V, angles=phi)
    # 绘制完成后绘图区域刷新，检查是否有警告产生
    fig.canvas.draw()  # Check that no warning is emitted.


# 检测当 headlength 设为 0 时是否产生警告
def test_zero_headlength():
    # 基于 Doug McNeil 的报告进行测试
    # https://discourse.matplotlib.org/t/quiver-warnings/16722
    # 创建图形和坐标轴
    fig, ax = plt.subplots()
    # 创建 X, Y 网格
    X, Y = np.meshgrid(np.arange(10), np.arange(10))
    U, V = np.cos(X), np.sin(Y)
    # 绘制矢量场，设置头部长度和头部轴长为 0
    ax.quiver(U, V, headlength=0, headaxislength=0)
    # 绘制完成后绘图区域刷新，检查是否有警告产生
    fig.canvas.draw()  # Check that no warning is emitted.
@image_comparison(['quiver_animated_test_image.png'])
# 定义一个测试函数，比较修复问题 #2616 的效果
def test_quiver_animate():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 调用 draw_quiver 函数，传入轴对象和 animated=True 参数，返回一个 quiver 对象
    Q = draw_quiver(ax, animated=True)
    # 在轴对象上绘制 quiver 的图例，指定位置和文本标签
    ax.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$',
                 labelpos='W', fontproperties={'weight': 'bold'})


@image_comparison(['quiver_with_key_test_image.png'])
# 定义一个测试函数，测试绘制带有图例的 quiver 图像
def test_quiver_with_key():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 设置轴的边界
    ax.margins(0.1)
    # 调用 draw_quiver 函数，传入轴对象，返回一个 quiver 对象
    Q = draw_quiver(ax)
    # 在轴对象上绘制带有图例的 quiver，指定位置、文本标签和字体属性
    ax.quiverkey(Q, 0.5, 0.95, 2,
                 r'$2\, \mathrm{m}\, \mathrm{s}^{-1}$',
                 angle=-10,
                 coordinates='figure',
                 labelpos='W',
                 fontproperties={'weight': 'bold', 'size': 'large'})


@image_comparison(['quiver_single_test_image.png'], remove_text=True)
# 定义一个测试函数，测试绘制单个箭头的 quiver 图像
def test_quiver_single():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 设置轴的边界
    ax.margins(0.1)
    # 在轴对象上绘制单个箭头的 quiver 图像
    ax.quiver([1], [1], [2], [2])


# 定义一个测试函数，测试 quiver 对象的复制行为
def test_quiver_copy():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 定义一个包含 u 和 v 数组的字典
    uv = dict(u=np.array([1.1]), v=np.array([2.0]))
    # 在轴对象上绘制 quiver 图像，传入位置和速度信息，返回一个 quiver 对象
    q0 = ax.quiver([1], [1], uv['u'], uv['v'])
    # 修改数组 v 中的第一个元素
    uv['v'][0] = 0
    # 断言 quiver 对象的第一个速度分量是否等于 2.0
    assert q0.V[0] == 2.0


@image_comparison(['quiver_key_pivot.png'], remove_text=True)
# 定义一个测试函数，测试在轴上绘制多个带箭头的 quiver 图像并设置图例
def test_quiver_key_pivot():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 使用 np.mgrid 定义网格 u 和 v
    u, v = np.mgrid[0:2*np.pi:10j, 0:2*np.pi:10j]

    # 在轴对象上绘制 quiver 图像，传入 u 和 v 数组，返回一个 quiver 对象
    q = ax.quiver(np.sin(u), np.cos(v))
    # 设置轴的 x 和 y 范围
    ax.set_xlim(-2, 11)
    ax.set_ylim(-2, 11)
    # 在轴对象上绘制多个带箭头的 quiver 图像的图例，指定位置和文本标签
    ax.quiverkey(q, 0.5, 1, 1, 'N', labelpos='N')
    ax.quiverkey(q, 1, 0.5, 1, 'E', labelpos='E')
    ax.quiverkey(q, 0.5, 0, 1, 'S', labelpos='S')
    ax.quiverkey(q, 0, 0.5, 1, 'W', labelpos='W')


@image_comparison(['quiver_key_xy.png'], remove_text=True)
# 定义一个测试函数，测试在轴上绘制 xy 单位的 quiver 图像并设置图例
def test_quiver_key_xy():
    # 定义数组 X 和 Y
    X = np.arange(8)
    Y = np.zeros(8)
    # 计算角度数组
    angles = X * (np.pi / 4)
    # 计算复数数组 uv
    uv = np.exp(1j * angles)
    U = uv.real
    V = uv.imag
    # 创建包含两个子图的图形对象
    fig, axs = plt.subplots(2)
    # 遍历两个子图，分别设置 x 和 y 范围
    for ax, angle_str in zip(axs, ('uv', 'xy')):
        ax.set_xlim(-1, 8)
        ax.set_ylim(-0.2, 0.2)
        # 在子图上绘制 xy 单位的 quiver 图像，传入位置和速度信息，返回一个 quiver 对象
        q = ax.quiver(X, Y, U, V, pivot='middle',
                      units='xy', width=0.05,
                      scale=2, scale_units='xy',
                      angles=angle_str)
        # 在子图上绘制多个带箭头的 quiver 图像的图例，指定位置、角度和颜色
        for x, angle in zip((0.2, 0.5, 0.8), (0, 45, 90)):
            ax.quiverkey(q, X=x, Y=0.8, U=1, angle=angle, label='', color='b')


@image_comparison(['barbs_test_image.png'], remove_text=True)
# 定义一个测试函数，测试在轴上绘制 barbs 图像
def test_barbs():
    # 创建 x 数组
    x = np.linspace(-5, 5, 5)
    # 创建网格 X 和 Y
    X, Y = np.meshgrid(x, x)
    # 计算速度数组 U 和 V
    U, V = 12*X, 12*Y
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴对象上绘制 barbs 图像，传入位置和速度信息，以及其他参数
    ax.barbs(X, Y, U, V, np.hypot(U, V), fill_empty=True, rounding=False,
             sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3),
             cmap='viridis')


@image_comparison(['barbs_pivot_test_image.png'], remove_text=True)
# 定义一个测试函数，测试在轴上绘制带箭头的 barbs 图像
def test_barbs_pivot():
    # 创建 x 数组
    x = np.linspace(-5, 5, 5)
    # 创建网格 X 和 Y
    X, Y = np.meshgrid(x, x)
    # 计算矢量场的起始点和方向，乘以12以增加箭头的长度
    U, V = 12*X, 12*Y
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制矢量场，箭头的起始点由(X, Y)确定，长度和方向由(U, V)决定
    # 使用空箭头填充空白区域，设置箭头的圆角为False，旋转中心为1.7倍箭头长度
    # 定义空箭头的大小为0.25，箭头之间的间距为0.2，箭头的高度为0.3
    ax.barbs(X, Y, U, V, fill_empty=True, rounding=False, pivot=1.7,
             sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3))
    # 在相同的坐标轴上绘制散点图，点的大小为49，颜色为黑色
    ax.scatter(X, Y, s=49, c='black')
# 使用 @image_comparison 装饰器比较图像是否匹配，并移除测试中的文本
@image_comparison(['barbs_test_flip.png'], remove_text=True)
# 定义测试函数 test_barbs_flip，用于测试带有 flip_barb 数组的 barbs 方法
def test_barbs_flip():
    """Test barbs with an array for flip_barb."""
    # 创建一个在[-5, 5]范围内均匀分布的数组 x
    x = np.linspace(-5, 5, 5)
    # 创建网格 X, Y
    X, Y = np.meshgrid(x, x)
    # 设置风向 U 和 V
    U, V = 12*X, 12*Y
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制风羽图，使用 fill_empty 和 rounding 参数，并根据 flip_barb 条件翻转风羽
    ax.barbs(X, Y, U, V, fill_empty=True, rounding=False, pivot=1.7,
             sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3),
             flip_barb=Y < 0)


# 定义测试函数 test_barb_copy
def test_barb_copy():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建数组 u 和 v，分别赋值为 [1.1] 和 [2.2]
    u = np.array([1.1])
    v = np.array([2.2])
    # 绘制风羽图并获取返回的对象 b0
    b0 = ax.barbs([1], [1], u, v)
    # 修改数组 u 的第一个元素为 0，并验证 b0 中的 u 值
    u[0] = 0
    assert b0.u[0] == 1.1
    # 修改数组 v 的第一个元素为 0，并验证 b0 中的 v 值
    v[0] = 0
    assert b0.v[0] == 2.2


# 定义测试函数 test_bad_masked_sizes，测试当给定不同大小的掩码数组时的错误处理
def test_bad_masked_sizes():
    # 创建数组 x 和 y
    x = np.arange(3)
    y = np.arange(3)
    # 创建掩码数组 u 和 v
    u = np.ma.array(15. * np.ones((4,)))
    v = np.ma.array(15. * np.ones_like(u))
    # 将 u 和 v 的第二个元素设为掩码
    u[1] = np.ma.masked
    v[1] = np.ma.masked
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 断言抛出 ValueError 异常，当使用不同大小的掩码数组绘制风羽图时
    with pytest.raises(ValueError):
        ax.barbs(x, y, u, v)


# 定义测试函数 test_angles_and_scale，测试角度数组和 scale_units 参数
def test_angles_and_scale():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建网格 X, Y，并设置单位向量 U, V
    X, Y = np.meshgrid(np.arange(15), np.arange(10))
    U = V = np.ones_like(X)
    # 创建随机角度数组 phi
    phi = (np.random.rand(15, 10) - .5) * 150
    # 绘制箭头图，使用角度数组 phi 和 scale_units 参数为 'xy'
    ax.quiver(X, Y, U, V, angles=phi, scale_units='xy')


# 使用 @image_comparison 装饰器比较图像是否匹配，移除测试中的文本
@image_comparison(['quiver_xy.png'], remove_text=True)
# 定义测试函数 test_quiver_xy，测试简单的箭头从西南到东北的情况
def test_quiver_xy():
    # 创建图形和坐标轴对象，设置纵横比为 'equal'
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'))
    # 绘制箭头，从 (0,0) 到 (1,1)，使用角度 'xy'，scale_units 'xy'，且比例为 1
    ax.quiver(0, 0, 1, 1, angles='xy', scale_units='xy', scale=1)
    # 设置 x 和 y 轴的限制
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    # 添加网格线
    ax.grid()


# 定义测试函数 test_quiverkey_angles，测试当给定角度数组时，quiverkey 绘制的箭头数量
def test_quiverkey_angles():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建网格 X, Y，并设置单位向量 U, V，以及角度数组 angles
    X, Y = np.meshgrid(np.arange(2), np.arange(2))
    U = V = angles = np.ones_like(X)
    # 绘制箭头图，并获取返回的对象 q
    q = ax.quiver(X, Y, U, V, angles=angles)
    # 在坐标 (1, 1) 处添加 quiverkey，设置长度为 2，标签为 'Label'
    qk = ax.quiverkey(q, 1, 1, 2, 'Label')
    # 绘制 quiverkey 后，检查箭头顶点列表的长度是否为 1
    fig.canvas.draw()
    assert len(qk.verts) == 1


# 定义测试函数 test_quiverkey_angles_xy_aitoff，测试非笛卡尔坐标系下，使用角度 'xy' 和 scale_units 'xy' 的情况
def test_quiverkey_angles_xy_aitoff():
    # GH 26316 和 GH 26748
    # 仅用于测试目的
    # 定义参数列表，测试非笛卡尔坐标系下，使用角度 'xy' 和 scale_units 'xy' 的情况
    kwargs_list = [
        {'angles': 'xy'},
        {'angles': 'xy', 'scale_units': 'xy'},
        {'scale_units': 'xy'}
    ]

    # 遍历参数列表
    for kwargs_dict in kwargs_list:
        # 创建角度数组 x 和 y
        x = np.linspace(-np.pi, np.pi, 11)
        y = np.ones_like(x) * np.pi / 6
        # 创建单位向量 vx 和 vy
        vx = np.zeros_like(x)
        vy = np.ones_like(x)

        # 创建图形对象，使用 'aitoff' 投影添加子图
        fig = plt.figure()
        ax = fig.add_subplot(projection='aitoff')
        # 绘制箭头图，并使用给定的参数字典 kwargs_dict
        q = ax.quiver(x, y, vx, vy, **kwargs_dict)
        # 在坐标 (0, 0) 处添加 quiverkey，设置长度为 1，标签为 '1 units'
        qk = ax.quiverkey(q, 0, 0, 1, '1 units')

        # 绘制 quiverkey 后，检查箭头顶点列表的长度是否为 1
        fig.canvas.draw()
        assert len(qk.verts) == 1


# 定义测试函数 test_quiverkey_angles_scale_units_cartesian，测试笛卡尔坐标系下，使用角度 'xy' 的情况
def test_quiverkey_angles_scale_units_cartesian():
    # GH 26316
    # 未完成的测试
    # 测试在普通笛卡尔坐标系中，当 angles='xy' 和/或 scale_units='xy' 时，是否只绘制一个箭头

    # 定义多个参数字典，用于测试不同的参数组合
    kwargs_list = [
        {'angles': 'xy'},
        {'angles': 'xy', 'scale_units': 'xy'},
        {'scale_units': 'xy'}
    ]

    # 遍历参数字典列表
    for kwargs_dict in kwargs_list:
        # 定义箭头起点 X、Y 坐标和箭头方向 U、V 分量
        X = [0, -1, 0]
        Y = [0, -1, 0]
        U = [1, -1, 1]
        V = [1, -1, 0]

        # 创建一个新的图形和坐标轴对象
        fig, ax = plt.subplots()

        # 绘制箭头图，使用给定的参数字典 kwargs_dict
        q = ax.quiver(X, Y, U, V, **kwargs_dict)

        # 在图中添加箭头图的图例
        ax.quiverkey(q, X=0.3, Y=1.1, U=1,
                     label='Quiver key, length = 1', labelpos='E')

        # 在图中添加箭头图的图例
        qk = ax.quiverkey(q, 0, 0, 1, '1 units')

        # 绘图更新画布
        fig.canvas.draw()

        # 断言箭头图例的顶点数量是否为1
        assert len(qk.verts) == 1
def test_quiver_setuvc_numbers():
    """Check that it is possible to set all arrow UVC to the same numbers"""
    
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()

    # 创建一个二维网格，X 和 Y 分别是 X 和 Y 轴的坐标矩阵
    X, Y = np.meshgrid(np.arange(2), np.arange(2))
    
    # 创建一个与 X 和 Y 相同形状的数组，填充为 1，作为箭头的速度分量
    U = V = np.ones_like(X)

    # 在坐标轴上绘制箭头图，使用 X、Y 作为位置坐标，U、V 作为箭头的速度分量
    q = ax.quiver(X, Y, U, V)
    
    # 设置所有箭头的 U、V、C（颜色）为 (0, 1)
    q.set_UVC(0, 1)
```