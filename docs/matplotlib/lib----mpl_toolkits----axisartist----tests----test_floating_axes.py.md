# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\tests\test_floating_axes.py`

```py
# 导入必要的库
import numpy as np

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import matplotlib.projections as mprojections  # 导入 matplotlib 的 projections 子模块
import matplotlib.transforms as mtransforms  # 导入 matplotlib 的 transforms 子模块
from matplotlib.testing.decorators import image_comparison  # 从 matplotlib 的 testing.decorators 模块导入 image_comparison 装饰器
from mpl_toolkits.axisartist.axislines import Subplot  # 从 mpl_toolkits.axisartist.axislines 模块导入 Subplot 类
from mpl_toolkits.axisartist.floating_axes import (  # 导入 mpl_toolkits.axisartist.floating_axes 模块的 FloatingAxes 和 GridHelperCurveLinear
    FloatingAxes, GridHelperCurveLinear)
from mpl_toolkits.axisartist.grid_finder import FixedLocator  # 从 mpl_toolkits.axisartist.grid_finder 模块导入 FixedLocator 类
from mpl_toolkits.axisartist import angle_helper  # 从 mpl_toolkits.axisartist 模块导入 angle_helper 模块


def test_subplot():
    # 创建一个大小为 5x5 的新图形对象
    fig = plt.figure(figsize=(5, 5))
    # 在图形上创建一个 Subplot 对象，位置为 111
    ax = Subplot(fig, 111)
    # 将子图添加到图形中
    fig.add_subplot(ax)


# Rather high tolerance to allow ongoing work with floating axes internals;
# remove when image is regenerated.
@image_comparison(['curvelinear3.png'], style='default', tol=5)
def test_curvelinear3():
    # 创建一个大小为 5x5 的新图形对象
    fig = plt.figure(figsize=(5, 5))

    # 创建一个仿射变换对象 tr，缩放角度单位为弧度，同时创建极坐标变换对象
    tr = (mtransforms.Affine2D().scale(np.pi / 180, 1) +
          mprojections.PolarAxes.PolarTransform(apply_theta_transforms=False))
    
    # 创建一个 GridHelperCurveLinear 对象 grid_helper
    grid_helper = GridHelperCurveLinear(
        tr,
        extremes=(0, 360, 10, 3),  # 设置极径和角度的极值
        grid_locator1=angle_helper.LocatorDMS(15),  # 设置角度轴的刻度定位器为 15 度间隔
        grid_locator2=FixedLocator([2, 4, 6, 8, 10]),  # 设置径向轴的刻度定位器为固定值
        tick_formatter1=angle_helper.FormatterDMS(),  # 设置角度轴的刻度格式化器为 DMS 格式
        tick_formatter2=None)  # 不设置径向轴的刻度格式化器

    # 在图形中添加一个浮动坐标轴（FloatingAxes），使用上面创建的 grid_helper
    ax1 = fig.add_subplot(axes_class=FloatingAxes, grid_helper=grid_helper)

    # 设置径向轴的比例尺
    r_scale = 10
    tr2 = mtransforms.Affine2D().scale(1, 1 / r_scale) + tr
    grid_helper2 = GridHelperCurveLinear(
        tr2,
        extremes=(0, 360, 10 * r_scale, 3 * r_scale),
        grid_locator2=FixedLocator([30, 60, 90]))

    # 添加一个新的固定坐标轴到 ax1 的右侧
    ax1.axis["right"] = axis = grid_helper2.new_fixed_axis("right", axes=ax1)

    # 设置左侧坐标轴的标签文本为 "Test 1"
    ax1.axis["left"].label.set_text("Test 1")
    # 设置右侧坐标轴的标签文本为 "Test 2"
    ax1.axis["right"].label.set_text("Test 2")
    # 隐藏左右两侧坐标轴
    ax1.axis["left", "right"].set_visible(False)

    # 在 ax1 上创建一个新的浮动坐标轴，方向为底部（1），位置在 7 处
    axis = grid_helper.new_floating_axis(1, 7, axes=ax1,
                                         axis_direction="bottom")
    # 设置新浮动坐标轴的标签文本为 "z = ?"
    axis.label.set_text("z = ?")
    # 设置新浮动坐标轴的标签可见
    axis.label.set_visible(True)
    # 设置新浮动坐标轴的线条颜色为灰色
    axis.line.set_color("0.5")

    # 在 ax1 上获取辅助坐标系对象 ax2
    ax2 = ax1.get_aux_axes(tr)

    # 创建数据点坐标
    xx, yy = [67, 90, 75, 30], [2, 5, 8, 4]
    # 在 ax2 上绘制散点图
    ax2.scatter(xx, yy)
    # 在 ax2 上绘制连线图
    l, = ax2.plot(xx, yy, "k-")
    # 设置连线图的裁剪路径为 ax1 的补丁区域
    l.set_clip_path(ax1.patch)


# Rather high tolerance to allow ongoing work with floating axes internals;
# remove when image is regenerated.
@image_comparison(['curvelinear4.png'], style='default', tol=0.9)
def test_curvelinear4():
    # 设置全局参数，调整文本的字距因子
    plt.rcParams['text.kerning_factor'] = 6

    # 创建一个大小为 5x5 的新图形对象
    fig = plt.figure(figsize=(5, 5))

    # 创建一个仿射变换对象 tr，缩放角度单位为弧度，同时创建极坐标变换对象
    tr = (mtransforms.Affine2D().scale(np.pi / 180, 1) +
          mprojections.PolarAxes.PolarTransform(apply_theta_transforms=False))
    
    # 创建一个 GridHelperCurveLinear 对象 grid_helper
    grid_helper = GridHelperCurveLinear(
        tr,
        extremes=(120, 30, 10, 0),  # 设置极径和角度的极值
        grid_locator1=angle_helper.LocatorDMS(5),  # 设置角度轴的刻度定位器为 5 度间隔
        grid_locator2=FixedLocator([2, 4, 6, 8, 10]),  # 设置径向轴的刻度定位器为固定值
        tick_formatter1=angle_helper.FormatterDMS(),  # 设置角度轴的刻度格式化器为 DMS 格式
        tick_formatter2=None)  # 不设置径向轴的刻度格式化器
    # 在图形 fig 上添加一个子图 ax1，使用 FloatingAxes 类和指定的 grid_helper
    ax1 = fig.add_subplot(axes_class=FloatingAxes, grid_helper=grid_helper)
    # 清空 ax1 的内容，确保 clear() 方法也恢复了 ax1 的正确限制
    ax1.clear()

    # 设置 ax1 左侧轴的标签文本为 "Test 1"
    ax1.axis["left"].label.set_text("Test 1")
    # 设置 ax1 右侧轴的标签文本为 "Test 2"
    ax1.axis["right"].label.set_text("Test 2")
    # 隐藏 ax1 的顶部轴
    ax1.axis["top"].set_visible(False)

    # 在 ax1 上创建一个新的浮动轴，位置在底部，角度为 70 度，使用 grid_helper
    axis = grid_helper.new_floating_axis(1, 70, axes=ax1,
                                         axis_direction="bottom")
    # 将这个新轴添加到 ax1 的 "z" 轴位置
    ax1.axis["z"] = axis
    # 启用新轴的所有功能和标签显示
    axis.toggle(all=True, label=True)
    # 设置新轴标签的方向为顶部
    axis.label.set_axis_direction("top")
    # 设置新轴标签的文本为 "z = ?"
    axis.label.set_text("z = ?")
    # 显示新轴的标签
    axis.label.set_visible(True)
    # 设置新轴线的颜色为灰色 ("0.5")
    axis.line.set_color("0.5")

    # 在 ax1 上获取与变换 tr 相关的辅助轴 ax2
    ax2 = ax1.get_aux_axes(tr)

    # 定义数据点的 x 和 y 值
    xx, yy = [67, 90, 75, 30], [2, 5, 8, 4]
    # 在 ax2 上绘制散点图
    ax2.scatter(xx, yy)
    # 在 ax2 上绘制连接这些点的线条，并保存返回的线对象到 l
    l, = ax2.plot(xx, yy, "k-")
    # 设置线条 l 的剪切路径为 ax1 的补丁（边界）
    l.set_clip_path(ax1.patch)
# 定义一个测试函数，用于验证浮动轴上的轴方向是否正确传播
def test_axis_direction():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形对象上创建一个子图
    ax = Subplot(fig, 111)
    # 将子图添加到图形对象中
    fig.add_subplot(ax)
    # 在轴对象上创建一个新的浮动轴，设置其在第二个坐标轴上，值为0，并指定轴方向为左侧
    ax.axis['y'] = ax.new_floating_axis(nth_coord=1, value=0,
                                        axis_direction='left')
    # 断言验证浮动轴对象上的轴方向是否为左侧
    assert ax.axis['y']._axis_direction == 'left'
```