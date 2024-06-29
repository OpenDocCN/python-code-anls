# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\tests\test_axis_artist.py`

```
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并简写为 plt
from matplotlib.testing.decorators import image_comparison  # 导入 matplotlib 的测试装饰器 image_comparison

from mpl_toolkits.axisartist import AxisArtistHelperRectlinear  # 导入 mpl_toolkits.axisartist 中的 AxisArtistHelperRectlinear
from mpl_toolkits.axisartist.axis_artist import (AxisArtist, AxisLabel,  # 导入 mpl_toolkits.axisartist.axis_artist 中的类
                                                 LabelBase, Ticks, TickLabels)


@image_comparison(['axis_artist_ticks.png'], style='default')  # 使用 image_comparison 装饰器，比较测试结果图像，命名为 'axis_artist_ticks.png'，使用 'default' 风格
def test_ticks():  # 定义测试函数 test_ticks
    fig, ax = plt.subplots()  # 创建一个图形和一个子图

    ax.xaxis.set_visible(False)  # 设置 x 轴不可见
    ax.yaxis.set_visible(False)  # 设置 y 轴不可见

    locs_angles = [((i / 10, 0.0), i * 30) for i in range(-1, 12)]  # 创建 locs_angles 列表，包含位置和角度的元组

    ticks_in = Ticks(ticksize=10, axis=ax.xaxis)  # 创建 x 轴内侧刻度对象 ticks_in
    ticks_in.set_locs_angles(locs_angles)  # 设置 ticks_in 的位置和角度
    ax.add_artist(ticks_in)  # 将 ticks_in 添加到子图中

    ticks_out = Ticks(ticksize=10, tick_out=True, color='C3', axis=ax.xaxis)  # 创建 x 轴外侧刻度对象 ticks_out
    ticks_out.set_locs_angles(locs_angles)  # 设置 ticks_out 的位置和角度
    ax.add_artist(ticks_out)  # 将 ticks_out 添加到子图中


@image_comparison(['axis_artist_labelbase.png'], style='default')  # 使用 image_comparison 装饰器，比较测试结果图像，命名为 'axis_artist_labelbase.png'，使用 'default' 风格
def test_labelbase():  # 定义测试函数 test_labelbase
    # 在重新生成此测试图像时，请删除此行。
    plt.rcParams['text.kerning_factor'] = 6  # 设置文本的字距因子为 6

    fig, ax = plt.subplots()  # 创建一个图形和一个子图

    ax.plot([0.5], [0.5], "o")  # 在子图中绘制一个圆点

    label = LabelBase(0.5, 0.5, "Test")  # 创建 LabelBase 对象 label，位于 (0.5, 0.5)，文本为 "Test"
    label._ref_angle = -90  # 设置 label 的参考角度为 -90
    label._offset_radius = 50  # 设置 label 的偏移半径为 50
    label.set_rotation(-90)  # 设置 label 的旋转角度为 -90
    label.set(ha="center", va="top")  # 设置 label 的水平对齐方式为居中，垂直对齐方式为顶部
    ax.add_artist(label)  # 将 label 添加到子图中


@image_comparison(['axis_artist_ticklabels.png'], style='default')  # 使用 image_comparison 装饰器，比较测试结果图像，命名为 'axis_artist_ticklabels.png'，使用 'default' 风格
def test_ticklabels():  # 定义测试函数 test_ticklabels
    # 在重新生成此测试图像时，请删除此行。
    plt.rcParams['text.kerning_factor'] = 6  # 设置文本的字距因子为 6

    fig, ax = plt.subplots()  # 创建一个图形和一个子图

    ax.xaxis.set_visible(False)  # 设置 x 轴不可见
    ax.yaxis.set_visible(False)  # 设置 y 轴不可见

    ax.plot([0.2, 0.4], [0.5, 0.5], "o")  # 在子图中绘制两个圆点

    ticks = Ticks(ticksize=10, axis=ax.xaxis)  # 创建 x 轴刻度对象 ticks
    ax.add_artist(ticks)  # 将 ticks 添加到子图中
    locs_angles_labels = [((0.2, 0.5), -90, "0.2"),  # locs_angles_labels 列表，包含位置、角度和标签的元组
                          ((0.4, 0.5), -120, "0.4")]
    tick_locs_angles = [(xy, a + 180) for xy, a, l in locs_angles_labels]  # 计算刻度位置和角度
    ticks.set_locs_angles(tick_locs_angles)  # 设置 ticks 的位置和角度

    ticklabels = TickLabels(axis_direction="left")  # 创建刻度标签对象 ticklabels，刻度标签方向为左侧
    ticklabels._locs_angles_labels = locs_angles_labels  # 设置 ticklabels 的位置、角度和标签
    ticklabels.set_pad(10)  # 设置 ticklabels 的填充
    ax.add_artist(ticklabels)  # 将 ticklabels 添加到子图中

    ax.plot([0.5], [0.5], "s")  # 在子图中绘制一个正方形点
    axislabel = AxisLabel(0.5, 0.5, "Test")  # 创建 AxisLabel 对象 axislabel，位于 (0.5, 0.5)，文本为 "Test"
    axislabel._offset_radius = 20  # 设置 axislabel 的偏移半径为 20
    axislabel._ref_angle = 0  # 设置 axislabel 的参考角度为 0
    axislabel.set_axis_direction("bottom")  # 设置 axislabel 的轴方向为底部
    ax.add_artist(axislabel)  # 将 axislabel 添加到子图中

    ax.set_xlim(0, 1)  # 设置 x 轴的显示范围为 0 到 1
    ax.set_ylim(0, 1)  # 设置 y 轴的显示范围为 0 到 1


@image_comparison(['axis_artist.png'], style='default')  # 使用 image_comparison 装饰器，比较测试结果图像，命名为 'axis_artist.png'，使用 'default' 风格
def test_axis_artist():  # 定义测试函数 test_axis_artist
    # 在重新生成此测试图像时，请删除此行。
    plt.rcParams['text.kerning_factor'] = 6  # 设置文本的字距因子为 6

    fig, ax = plt.subplots()  # 创建一个图形和一个子图

    ax.xaxis.set_visible(False)  # 设置 x 轴不可见
    ax.yaxis.set_visible(False)  # 设置 y 轴不可见

    for loc in ('left', 'right', 'bottom'):  # 遍历轴的位置：左、右、底部
        helper = AxisArtistHelperRectlinear.Fixed(ax, loc=loc)  # 创建辅助对象 helper，固定在指定位置 loc 上
        axisline = AxisArtist(ax, helper, offset=None, axis_direction=loc)  # 创建轴艺术家对象 axisline
        ax.add_artist(axisline)  # 将 axisline 添加到子图中

    # 设置底部 AxisArtist 的标签为 "TTT"
    axisline.set_label("TTT")
    axisline.major_ticks.set_tick_out(False)  # 设置主刻度线不突出显示
    axisline.label.set_pad(5)  # 设置标签的填充为 5

    ax.set_ylabel("Test")  # 设置 y 轴标签为 "Test"
```