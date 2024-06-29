# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_patheffects.py`

```py
# 导入 platform 模块，用于获取操作系统信息
import platform
# 导入 numpy 库，并使用 np 别名
import numpy as np
# 从 matplotlib.testing.decorators 模块导入 image_comparison 装饰器
from matplotlib.testing.decorators import image_comparison
# 导入 matplotlib.pyplot 库，并使用 plt 别名
import matplotlib.pyplot as plt
# 导入 matplotlib.patheffects 模块，并使用 path_effects 别名
import matplotlib.patheffects as path_effects
# 从 matplotlib.path 模块导入 Path 类
from matplotlib.path import Path
# 导入 matplotlib.patches 模块，并使用 patches 别名
import matplotlib.patches as patches
# 从 matplotlib.backend_bases 模块导入 RendererBase 类
from matplotlib.backend_bases import RendererBase
# 从 matplotlib.patheffects 模块导入 PathEffectRenderer 类
from matplotlib.patheffects import PathEffectRenderer


@image_comparison(['patheffect1'], remove_text=True)
def test_patheffect1():
    # 创建一个子图对象 ax1
    ax1 = plt.subplot()
    # 在 ax1 上显示一个简单的二维数组图像
    ax1.imshow([[1, 2], [2, 3]])
    # 在指定位置 (1., 1.) 处添加注释文本 "test"，设置注释的起始和终止点坐标为 (0., 0)
    # 同时添加带有箭头样式的路径效果，设置文本大小为 20，水平对齐方式为居中
    # 同时添加路径效果，包括一个带有描边效果的路径效果对象
    txt = ax1.annotate("test", (1., 1.), (0., 0),
                       arrowprops=dict(arrowstyle="->",
                                       connectionstyle="angle3", lw=2),
                       size=20, ha="center",
                       path_effects=[path_effects.withStroke(linewidth=3,
                                                             foreground="w")])
    # 设置箭头部分的路径效果，包括描边和正常效果
    txt.arrow_patch.set_path_effects([path_effects.Stroke(linewidth=5,
                                                          foreground="w"),
                                      path_effects.Normal()])

    # 在 ax1 上启用网格线，并设置路径效果，添加一个带有描边效果的路径效果对象
    pe = [path_effects.withStroke(linewidth=3, foreground="w")]
    ax1.grid(True, linestyle="-", path_effects=pe)


@image_comparison(['patheffect2'], remove_text=True, style='mpl20',
                  tol=0.06 if platform.machine() == 'arm64' else 0)
def test_patheffect2():
    # 创建一个子图对象 ax2
    ax2 = plt.subplot()
    # 创建一个简单的二维数组 arr，并在 ax2 上显示其插值图像
    arr = np.arange(25).reshape((5, 5))
    ax2.imshow(arr, interpolation='nearest')
    # 在 ax2 上绘制 arr 的等高线，并设置路径效果，添加一个带有描边效果的路径效果对象
    cntr = ax2.contour(arr, colors="k")
    cntr.set(path_effects=[path_effects.withStroke(linewidth=3, foreground="w")])

    # 添加等高线标签，并设置路径效果，添加一个带有描边效果的路径效果对象
    clbls = ax2.clabel(cntr, fmt="%2.0f", use_clabeltext=True)
    plt.setp(clbls,
             path_effects=[path_effects.withStroke(linewidth=3,
                                                   foreground="w")])


@image_comparison(['patheffect3'], tol=0.019 if platform.machine() == 'arm64' else 0)
def test_patheffect3():
    # 在图上绘制一条带有样式的曲线，设置线宽为 4
    p1, = plt.plot([1, 3, 5, 4, 3], 'o-b', lw=4)
    # 设置曲线对象 p1 的路径效果，包括简单的线阴影和正常效果
    p1.set_path_effects([path_effects.SimpleLineShadow(),
                         path_effects.Normal()])
    # 设置图的标题，添加一个带有描边效果的路径效果对象，设置描边线宽为 1，颜色为红色
    plt.title(
        r'testing$^{123}$',
        path_effects=[path_effects.withStroke(linewidth=1, foreground="r")])
    # 设置图例的样式，添加一个带有简单补丁阴影的路径效果对象
    leg = plt.legend([p1], [r'Line 1$^2$'], fancybox=True, loc='upper left')
    leg.legendPatch.set_path_effects([path_effects.withSimplePatchShadow()])

    # 在指定位置添加文本 'Drop test'，设置颜色为白色，并设置一个带有路径效果的文本框
    text = plt.text(2, 3, 'Drop test', color='white',
                    bbox={'boxstyle': 'circle,pad=0.1', 'color': 'red'})
    # 设置文本对象 text 的路径效果，包括描边和简单补丁阴影
    pe = [path_effects.Stroke(linewidth=3.75, foreground='k'),
          path_effects.withSimplePatchShadow((6, -3), shadow_rgbFace='blue')]
    text.set_path_effects(pe)
    # 设置文本框的路径效果，与文本相同
    text.get_bbox_patch().set_path_effects(pe)

    # 设置路径效果列表，包括一个路径补丁效果和一个路径补丁效果，设置偏移量、阴影颜色和面颜色
    pe = [path_effects.PathPatchEffect(offset=(4, -4), hatch='xxxx',
                                       facecolor='gray'),
          path_effects.PathPatchEffect(edgecolor='white', facecolor='black',
                                       lw=1.1)]
    # 获取当前图形对象，并在指定位置添加文本 'Hatch shadow'
    # 设置文本的字体大小为 75，字体粗细为 1000（最粗），垂直对齐方式为居中
    t = plt.gcf().text(0.02, 0.1, 'Hatch shadow', fontsize=75, weight=1000,
                       va='center')
    # 给文本对象应用路径效果（path effects）
    t.set_path_effects(pe)
# 生成一个名为 'test_patheffects_stroked_text' 的测试函数，比较其生成的图像与 'stroked_text.png' 的差异
@image_comparison(['stroked_text.png'])
def test_patheffects_stroked_text():
    # 定义多个文本块
    text_chunks = [
        'A B C D E F G H I J K L',
        'M N O P Q R S T U V W',
        'X Y Z a b c d e f g h i j',
        'k l m n o p q r s t u v',
        'w x y z 0123456789',
        r"!@#$%^&*()-=_+[]\;'",
        ',./{}|:"<>?'
    ]
    # 设置字体大小
    font_size = 50

    # 在绘图中创建一个坐标轴
    ax = plt.axes((0, 0, 1, 1))
    # 遍历文本块列表
    for i, chunk in enumerate(text_chunks):
        # 在坐标轴上绘制文本，设置位置、大小、颜色等属性
        text = ax.text(x=0.01, y=(0.9 - i * 0.13), s=chunk,
                       fontdict={'ha': 'left', 'va': 'center',
                                 'size': font_size, 'color': 'white'})
        # 对文本应用路径效果：描边和正常效果
        text.set_path_effects([path_effects.Stroke(linewidth=font_size / 10,
                                                   foreground='black'),
                               path_effects.Normal()])

    # 设置坐标轴的范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # 关闭坐标轴显示
    ax.axis('off')


# 定义一个名为 'test_PathEffect_points_to_pixels' 的测试函数
def test_PathEffect_points_to_pixels():
    # 创建 DPI 为 150 的图形对象
    fig = plt.figure(dpi=150)
    # 绘制一个简单的折线图
    p1, = plt.plot(range(10))
    # 设置折线的路径效果：简单线阴影和正常效果
    p1.set_path_effects([path_effects.SimpleLineShadow(),
                         path_effects.Normal()])
    # 获取图形渲染器
    renderer = fig.canvas.get_renderer()
    # 使用路径效果渲染器创建一个路径效果渲染对象
    pe_renderer = path_effects.PathEffectRenderer(
        p1.get_path_effects(), renderer)
    # 断言：使用路径效果渲染器后，点大小转换为像素大小的结果应该一致
    assert renderer.points_to_pixels(15) == pe_renderer.points_to_pixels(15)


# 定义一个名为 'test_SimplePatchShadow_offset' 的测试函数
def test_SimplePatchShadow_offset():
    # 创建一个简单阴影路径效果对象，偏移量为 (4, 5)
    pe = path_effects.SimplePatchShadow(offset=(4, 5))
    # 断言：检查简单阴影路径效果对象的偏移量是否为 (4, 5)
    assert pe._offset == (4, 5)


# 生成一个名为 'test_collection' 的测试函数，比较其生成的图像与 'collection' 的差异，容忍度为 0.03，风格为 'mpl20'
@image_comparison(['collection'], tol=0.03, style='mpl20')
def test_collection():
    # 创建一个网格
    x, y = np.meshgrid(np.linspace(0, 10, 150), np.linspace(-5, 5, 100))
    # 计算数据：正弦加余弦
    data = np.sin(x) + np.cos(y)
    # 绘制等高线图
    cs = plt.contour(data)
    # 设置等高线的路径效果：路径补丁效果和描边效果
    cs.set(path_effects=[
        path_effects.PathPatchEffect(edgecolor='black', facecolor='none', linewidth=12),
        path_effects.Stroke(linewidth=5)])
    # 对每个等高线标签应用路径效果：带描边的效果，并设置边框样式和颜色
    for text in plt.clabel(cs, colors='white'):
        text.set_path_effects([path_effects.withStroke(foreground='k',
                                                       linewidth=3)])
        text.set_bbox({'boxstyle': 'sawtooth', 'facecolor': 'none',
                       'edgecolor': 'blue'})


# 生成一个名为 'test_tickedstroke' 的测试函数，比较其生成的图像与 'tickedstroke' 的差异，移除文本，文件类型为 'png'，容忍度为 0.22
@image_comparison(['tickedstroke'], remove_text=True, extensions=['png'], tol=0.22)
def test_tickedstroke():
    # 创建一个包含 3 个子图的图形对象，大小为 12x4
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    # 创建一个单位圆的路径对象
    path = Path.unit_circle()
    # 创建一个路径补丁对象，应用路径效果：带刻度的描边效果，角度为 -90，间距为 10，长度为 1
    patch = patches.PathPatch(path, facecolor='none', lw=2, path_effects=[
        path_effects.withTickedStroke(angle=-90, spacing=10,
                                      length=1)])

    # 将路径补丁对象添加到第一个子图中
    ax1.add_patch(patch)
    # 设置第一个子图的坐标轴等比例显示
    ax1.axis('equal')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

    # 在第二个子图中绘制一条简单折线，设置路径效果：带刻度的描边效果，间距为 7，角度为 135
    ax2.plot([0, 1], [0, 1], label=' ',
             path_effects=[path_effects.withTickedStroke(spacing=7,
                                                         angle=135)])
    # 设置点数目为 101
    nx = 101
    # 生成一个包含 nx 个均匀间隔点的数组，范围从 0.0 到 1.0
    x = np.linspace(0.0, 1.0, nx)
    # 使用 sin 函数生成一个以 x 为输入的 y 值数组，其中的振幅为 0.3，频率为 8，然后加上 0.4
    y = 0.3 * np.sin(x * 8) + 0.4
    # 在 ax2 图中绘制 x 和 y 的图形，带有标签 ' '，并应用路径效果 path_effects.withTickedStroke()
    ax2.plot(x, y, label=' ', path_effects=[path_effects.withTickedStroke()])

    # 在 ax2 图中显示图例
    ax2.legend()

    # 设置 nx 和 ny 的值分别为 101 和 105
    nx = 101
    ny = 105

    # 设置调查向量 xvec 和 yvec，分别生成从 0.001 到 4.0 的具有 nx 和 ny 个均匀间隔点的数组
    xvec = np.linspace(0.001, 4.0, nx)
    yvec = np.linspace(0.001, 4.0, ny)

    # 使用 xvec 和 yvec 创建网格 x1 和 x2
    x1, x2 = np.meshgrid(xvec, yvec)

    # 计算要绘制的一些内容
    g1 = -(3 * x1 + x2 - 5.5)
    g2 = -(x1 + 2 * x2 - 4)
    g3 = .8 + x1 ** -3 - x2

    # 在 ax3 图中绘制 g1 的等高线，等高线值为 [0]，颜色为黑色，并应用路径效果 path_effects.withTickedStroke(angle=135)
    cg1 = ax3.contour(x1, x2, g1, [0], colors=('k',))
    cg1.set(path_effects=[path_effects.withTickedStroke(angle=135)])

    # 在 ax3 图中绘制 g2 的等高线，等高线值为 [0]，颜色为红色，并应用路径效果 path_effects.withTickedStroke(angle=60, length=2)
    cg2 = ax3.contour(x1, x2, g2, [0], colors=('r',))
    cg2.set(path_effects=[path_effects.withTickedStroke(angle=60, length=2)])

    # 在 ax3 图中绘制 g3 的等高线，等高线值为 [0]，颜色为蓝色，并应用路径效果 path_effects.withTickedStroke(spacing=7)
    cg3 = ax3.contour(x1, x2, g3, [0], colors=('b',))
    cg3.set(path_effects=[path_effects.withTickedStroke(spacing=7)])

    # 设置 ax3 图的 x 轴和 y 轴的显示范围为 0 到 4
    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 4)
@image_comparison(['spaces_and_newlines.png'], remove_text=True)
def test_patheffects_spaces_and_newlines():
    # 创建一个新的图形轴
    ax = plt.subplot()
    # 定义两个字符串，其中一个包含多个空格
    s1 = "         "
    # 另一个包含换行符，会导致显示问题
    s2 = "\nNewline also causes problems"
    # 在图形轴上创建两个文本对象，分别显示上述字符串
    text1 = ax.text(0.5, 0.75, s1, ha='center', va='center', size=20,
                    bbox={'color': 'salmon'})
    text2 = ax.text(0.5, 0.25, s2, ha='center', va='center', size=20,
                    bbox={'color': 'thistle'})
    # 设置文本对象的路径效果为普通效果
    text1.set_path_effects([path_effects.Normal()])
    text2.set_path_effects([path_effects.Normal()])


def test_patheffects_overridden_methods_open_close_group():
    # 自定义渲染器类，继承自RendererBase
    class CustomRenderer(RendererBase):
        def __init__(self):
            super().__init__()

        # 重写打开组的方法
        def open_group(self, s, gid=None):
            return "open_group overridden"

        # 重写关闭组的方法
        def close_group(self, s):
            return "close_group overridden"

    # 创建PathEffectRenderer对象，传入普通效果和自定义渲染器
    renderer = PathEffectRenderer([path_effects.Normal()], CustomRenderer())

    # 断言自定义渲染器的打开和关闭组方法返回预期的结果
    assert renderer.open_group('s') == "open_group overridden"
    assert renderer.close_group('s') == "close_group overridden"
```