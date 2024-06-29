# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\mplot3d\tests\test_legend3d.py`

```py
# 导入platform模块，用于获取平台信息
import platform

# 导入numpy库，用于处理数组和数值计算
import numpy as np

# 导入matplotlib库，并从中导入所需的子模块和函数
import matplotlib as mpl
from matplotlib.colors import same_color
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

# 使用image_comparison装饰器比较生成的图像是否符合预期
@image_comparison(['legend_plot.png'], remove_text=True, style='mpl20')
def test_legend_plot():
    # 创建一个包含3D子图的图形对象fig和对应的坐标轴对象ax
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # 生成一个长度为10的数组x
    x = np.arange(10)
    # 在3D坐标轴上绘制以'o'标记的x与5-x关系，设置z方向为'y'，并添加图例标签'z=1'
    ax.plot(x, 5 - x, 'o', zdir='y', label='z=1')
    # 在3D坐标轴上绘制以'o'标记的x与x-5关系，设置z方向为'y'，并添加图例标签'z=-1'
    ax.plot(x, x - 5, 'o', zdir='y', label='z=-1')
    # 添加图例到图形对象中
    ax.legend()

# 使用image_comparison装饰器比较生成的图像是否符合预期
@image_comparison(['legend_bar.png'], remove_text=True, style='mpl20')
def test_legend_bar():
    # 创建一个包含3D子图的图形对象fig和对应的坐标轴对象ax
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # 生成一个长度为10的数组x
    x = np.arange(10)
    # 在3D坐标轴上绘制x与x关系的柱状图，设置z方向为'y'，对齐方式为'edge'，颜色为'm'
    b1 = ax.bar(x, x, zdir='y', align='edge', color='m')
    # 在3D坐标轴上绘制x与倒序x关系的柱状图，设置z方向为'x'，对齐方式为'edge'，颜色为'g'
    b2 = ax.bar(x, x[::-1], zdir='x', align='edge', color='g')
    # 添加包含b1和b2的图例到图形对象中，标签分别为'up'和'down'
    ax.legend([b1[0], b2[0]], ['up', 'down'])

# 使用image_comparison装饰器比较生成的图像是否符合预期，并设置特定的样式和误差容忍度
@image_comparison(['fancy.png'], remove_text=True, style='mpl20',
                  tol=0.011 if platform.machine() == 'arm64' else 0)
def test_fancy():
    # 创建一个包含3D子图的图形对象fig和对应的坐标轴对象ax
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # 在3D坐标轴上绘制以'o--'标记的np.arange(10)和全为5的数组，添加图例标签'line'
    ax.plot(np.arange(10), np.full(10, 5), np.full(10, 5), 'o--', label='line')
    # 在3D坐标轴上绘制散点图，x轴为np.arange(10)，y轴为倒序的np.arange(10, 0, -1)，添加图例标签'scatter'
    ax.scatter(np.arange(10), np.arange(10, 0, -1), label='scatter')
    # 在3D坐标轴上绘制误差线图，x轴为全为5的数组，y轴为np.arange(10)，x误差为0.5，z误差为0.5，添加图例标签'errorbar'
    ax.errorbar(np.full(10, 5), np.arange(10), np.full(10, 10),
                xerr=0.5, zerr=0.5, label='errorbar')
    # 添加位于左下角的图例，列数为2，标题为'My legend'，点数为1
    ax.legend(loc='lower left', ncols=2, title='My legend', numpoints=1)

# 定义测试函数test_linecollection_scaled_dashes
def test_linecollection_scaled_dashes():
    # 创建三个包含线段列表的变量lines1、lines2、lines3
    lines1 = [[(0, .5), (.5, 1)], [(.3, .6), (.2, .2)]]
    lines2 = [[[0.7, .2], [.8, .4]], [[.5, .7], [.6, .1]]]
    lines3 = [[[0.6, .2], [.8, .4]], [[.5, .7], [.1, .1]]]
    # 使用art3d.Line3DCollection创建三个线段集合变量lc1、lc2、lc3，分别指定不同的线型和线宽
    lc1 = art3d.Line3DCollection(lines1, linestyles="--", lw=3)
    lc2 = art3d.Line3DCollection(lines2, linestyles="-.")
    lc3 = art3d.Line3DCollection(lines3, linestyles=":", lw=.5)

    # 创建一个包含3D子图的图形对象fig和对应的坐标轴对象ax
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # 向坐标轴对象ax中添加三个线段集合lc1、lc2、lc3
    ax.add_collection(lc1)
    ax.add_collection(lc2)
    ax.add_collection(lc3)

    # 从坐标轴对象ax中获取图例，标签分别为'line1'、'line2'、'line 3'
    leg = ax.legend([lc1, lc2, lc3], ['line1', 'line2', 'line 3'])
    # 检查图例的线型是否与预期一致
    h1, h2, h3 = leg.legend_handles
    for oh, lh in zip((lc1, lc2, lc3), (h1, h2, h3)):
        assert oh.get_linestyles()[0] == lh._dash_pattern

# 定义测试函数test_handlerline3d
def test_handlerline3d():
    # 创建一个包含3D子图的图形对象fig和对应的坐标轴对象ax
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # 在坐标轴对象ax中绘制以'v'标记的散点图，坐标点为[0, 0]和[1, 1]
    ax.scatter([0, 1], [0, 1], marker="v")
    # 创建包含art3d.Line3D对象的列表handles，每个对象使用'v'标记
    handles = [art3d.Line3D([0], [0], [0], marker="v")]
    # 添加包含单个'v'标记对象的图例到坐标轴对象ax中，标签为"Aardvark"，点数为1
    leg = ax.legend(handles, ["Aardvark"], numpoints=1)
    # 检查图例的标记类型是否与预期一致
    assert handles[0].get_marker() == leg.legend_handles[0].get_marker()

# 定义测试函数test_contour_legend_elements
def test_contour_legend_elements():
    # 使用np.mgrid创建二维数组x和y，范围分别为1到9
    x, y = np.mgrid[1:10, 1:10]
    # 计算x * y的值作为高度h，生成包含'blue'、'#00FF00'、'red'的颜色列表
    h = x * y
    colors = ['blue', '#00FF00', 'red']

    # 创建一个包含3D子图的图形对象fig和对应的坐标轴对象ax
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # 在坐标轴对象ax中
    # 断言：验证 artists 列表中的所有元素是否都是 mpl.lines.Line2D 类型的对象
    assert all(isinstance(a, mpl.lines.Line2D) for a in artists)
    
    # 断言：验证 artists 列表中的每个元素与 colors 列表中对应位置的颜色是否相同
    assert all(same_color(a.get_color(), c)
               for a, c in zip(artists, colors))
def test_contourf_legend_elements():
    # 生成一个网格，x和y分别在1到10之间的值
    x, y = np.mgrid[1:10, 1:10]
    # 计算网格点对应的高度值
    h = x * y

    # 创建一个包含3D投影的图形和坐标轴对象
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # 绘制等高线图并填充颜色，指定绘制水平线的高度等级和颜色
    cs = ax.contourf(x, y, h, levels=[10, 30, 50],
                     colors=['#FFFF00', '#FF00FF', '#00FFFF'],
                     extend='both')
    # 设置颜色映射的高值颜色为红色
    cs.cmap.set_over('red')
    # 设置颜色映射的低值颜色为蓝色
    cs.cmap.set_under('blue')
    # 标记颜色映射已经改变
    cs.changed()
    # 获取图例中的元素和标签
    artists, labels = cs.legend_elements()
    # 断言标签与预期相符
    assert labels == ['$x \\leq -1e+250s$',
                      '$10.0 < x \\leq 30.0$',
                      '$30.0 < x \\leq 50.0$',
                      '$x > 1e+250s$']
    # 预期的颜色顺序
    expected_colors = ('blue', '#FFFF00', '#FF00FF', 'red')
    # 断言所有图例元素都是矩形对象
    assert all(isinstance(a, mpl.patches.Rectangle) for a in artists)
    # 断言所有图例元素的颜色与预期颜色相同
    assert all(same_color(a.get_facecolor(), c)
               for a, c in zip(artists, expected_colors))


def test_legend_Poly3dCollection():
    # 定义一个三维多边形集合对象
    verts = np.asarray([[0, 0, 0], [0, 1, 1], [1, 0, 1]])
    mesh = art3d.Poly3DCollection([verts], label="surface")

    # 创建一个包含3D投影的图形和坐标轴对象
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # 设置多边形集合的边缘颜色为黑色
    mesh.set_edgecolor('k')
    # 将多边形集合添加到3D坐标轴中
    handle = ax.add_collection3d(mesh)
    # 获取坐标轴的图例
    leg = ax.legend()
    # 断言图例中第一个句柄的面颜色与添加的集合面颜色相同
    assert (leg.legend_handles[0].get_facecolor()
            == handle.get_facecolor()).all()
```