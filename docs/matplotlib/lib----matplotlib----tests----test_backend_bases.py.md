# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_bases.py`

```py
import importlib  # 导入 Python 的 importlib 库，用于动态加载模块

from matplotlib import path, transforms  # 从 matplotlib 库中导入 path 和 transforms 模块
from matplotlib.backend_bases import (  # 从 matplotlib.backend_bases 导入多个类和枚举
    FigureCanvasBase, KeyEvent, LocationEvent, MouseButton, MouseEvent,
    NavigationToolbar2, RendererBase)
from matplotlib.backend_tools import RubberbandBase  # 导入 matplotlib.backend_tools 中的 RubberbandBase 类
from matplotlib.figure import Figure  # 导入 matplotlib.figure 中的 Figure 类
from matplotlib.testing._markers import needs_pgf_xelatex  # 导入测试标记

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，并命名为 plt

import numpy as np  # 导入 numpy 库，并命名为 np
import pytest  # 导入 pytest 测试框架


_EXPECTED_WARNING_TOOLMANAGER = (  # 定义全局变量 _EXPECTED_WARNING_TOOLMANAGER，用于存储警告信息的正则表达式
    r"Treat the new Tool classes introduced in "
    r"v[0-9]*.[0-9]* as experimental for now; "
    "the API and rcParam may change in future versions.")


def test_uses_per_path():  # 定义测试函数 test_uses_per_path
    id = transforms.Affine2D()  # 创建 Affine2D 变换对象，并赋值给 id
    paths = [path.Path.unit_regular_polygon(i) for i in range(3, 7)]  # 创建一系列的正多边形路径，并存储在列表 paths 中
    tforms_matrices = [id.rotate(i).get_matrix().copy() for i in range(1, 5)]  # 创建一系列的变换矩阵，并存储在列表 tforms_matrices 中
    offsets = np.arange(20).reshape((10, 2))  # 创建一个 10x2 的数组 offsets，用于存储偏移量
    facecolors = ['red', 'green']  # 面部颜色列表
    edgecolors = ['red', 'green']  # 边缘颜色列表

    def check(master_transform, paths, all_transforms,
              offsets, facecolors, edgecolors):
        rb = RendererBase()  # 创建 RendererBase 实例 rb
        raw_paths = list(rb._iter_collection_raw_paths(  # 调用 RendererBase 的 _iter_collection_raw_paths 方法，获取原始路径列表
            master_transform, paths, all_transforms))
        gc = rb.new_gc()  # 调用 RendererBase 的 new_gc 方法，创建图形上下文对象 gc
        ids = [path_id for xo, yo, path_id, gc0, rgbFace in
               rb._iter_collection(  # 调用 RendererBase 的 _iter_collection 方法，迭代并收集路径信息
                   gc, range(len(raw_paths)), offsets,
                   transforms.AffineDeltaTransform(master_transform),
                   facecolors, edgecolors, [], [], [False],
                   [], 'screen')]
        uses = rb._iter_collection_uses_per_path(  # 调用 RendererBase 的 _iter_collection_uses_per_path 方法，计算路径的使用次数
            paths, all_transforms, offsets, facecolors, edgecolors)
        if raw_paths:
            seen = np.bincount(ids, minlength=len(raw_paths))  # 统计每个路径的出现次数
            assert set(seen).issubset([uses - 1, uses])  # 断言每个路径的出现次数符合预期

    check(id, paths, tforms_matrices, offsets, facecolors, edgecolors)  # 调用 check 函数进行测试
    check(id, paths[0:1], tforms_matrices, offsets, facecolors, edgecolors)
    check(id, [], tforms_matrices, offsets, facecolors, edgecolors)
    check(id, paths, tforms_matrices[0:1], offsets, facecolors, edgecolors)
    check(id, paths, [], offsets, facecolors, edgecolors)
    for n in range(0, offsets.shape[0]):
        check(id, paths, tforms_matrices, offsets[0:n, :],
              facecolors, edgecolors)
    check(id, paths, tforms_matrices, offsets, [], edgecolors)
    check(id, paths, tforms_matrices, offsets, facecolors, [])
    check(id, paths, tforms_matrices, offsets, [], [])
    check(id, paths, tforms_matrices, offsets, facecolors[0:1], edgecolors)


def test_canvas_ctor():  # 定义测试函数 test_canvas_ctor
    assert isinstance(FigureCanvasBase().figure, Figure)  # 断言 FigureCanvasBase().figure 是 Figure 类的实例


def test_get_default_filename():  # 定义测试函数 test_get_default_filename
    assert plt.figure().canvas.get_default_filename() == 'image.png'  # 断言绘图对象的默认文件名为 'image.png'


def test_canvas_change():  # 定义测试函数 test_canvas_change
    fig = plt.figure()  # 创建图形对象 fig
    canvas = FigureCanvasBase(fig)  # 使用 fig 创建 FigureCanvasBase 实例 canvas
    plt.close(fig)  # 关闭图形对象 fig
    assert not plt.fignum_exists(fig.number)  # 断言图形对象 fig 不再存在


@pytest.mark.backend('pdf')  # pytest 标记，指定测试使用的后端为 'pdf'
def test_non_gui_warning(monkeypatch):  # 定义测试函数 test_non_gui_warning，接受 monkeypatch 参数
    plt.subplots()  # 调用 plt.subplots() 函数，创建子图
    # 设置环境变量 DISPLAY 为 ":999"，用于模拟显示器在端口 999 上的显示
    monkeypatch.setenv("DISPLAY", ":999")
    
    # 使用 pytest 捕获 UserWarning 类型的警告信息，并记录到 rec 中
    with pytest.warns(UserWarning) as rec:
        # 显示当前绘图的内容
        plt.show()
        # 断言捕获的警告数量为 1
        assert len(rec) == 1
        # 断言特定警告消息在捕获的消息中出现
        assert ('FigureCanvasPdf is non-interactive, and thus cannot be shown'
                in str(rec[0].message))
    
    # 使用 pytest 捕获 UserWarning 类型的警告信息，并记录到 rec 中
    with pytest.warns(UserWarning) as rec:
        # 获取当前图形，并显示其内容
        plt.gcf().show()
        # 断言捕获的警告数量为 1
        assert len(rec) == 1
        # 断言特定警告消息在捕获的消息中出现
        assert ('FigureCanvasPdf is non-interactive, and thus cannot be shown'
                in str(rec[0].message))
def test_grab_clear():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 在图形画布上抓取鼠标，将鼠标抓取器设置为当前轴对象
    fig.canvas.grab_mouse(ax)
    # 断言当前图形的鼠标抓取器确实是当前轴对象
    assert fig.canvas.mouse_grabber == ax

    # 清空图形内容
    fig.clear()
    # 断言当前图形的鼠标抓取器已经被清空
    assert fig.canvas.mouse_grabber is None


@pytest.mark.parametrize(
    "x, y", [(42, 24), (None, 42), (None, None), (200, 100.01), (205.75, 2.0)])
def test_location_event_position(x, y):
    # LocationEvent 应该将其 x 和 y 参数转换为整数，除非它们为 None
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 创建一个 FigureCanvasBase 类的实例
    canvas = FigureCanvasBase(fig)
    # 创建一个 LocationEvent 实例，传入测试事件名称、画布对象以及 x 和 y 参数
    event = LocationEvent("test_event", canvas, x, y)

    # 如果 x 是 None，则断言事件的 x 属性也是 None
    if x is None:
        assert event.x is None
    else:
        # 否则，断言事件的 x 属性应该是 x 的整数值
        assert event.x == int(x)
        # 并且确保 event.x 是整数类型
        assert isinstance(event.x, int)

    # 如果 y 是 None，则断言事件的 y 属性也是 None
    if y is None:
        assert event.y is None
    else:
        # 否则，断言事件的 y 属性应该是 y 的整数值
        assert event.y == int(y)
        # 并且确保 event.y 是整数类型
        assert isinstance(event.y, int)

    # 如果 x 和 y 都不是 None，则进行以下断言
    if x is not None and y is not None:
        # 断言轴对象根据 x 和 y 坐标格式化后的输出字符串
        assert (ax.format_coord(x, y)
                == f"(x, y) = ({ax.format_xdata(x)}, {ax.format_ydata(y)})")
        # 将轴对象的坐标格式化函数设置为返回固定字符串 "foo"
        ax.fmt_xdata = ax.fmt_ydata = lambda x: "foo"
        # 再次断言格式化后的坐标输出
        assert ax.format_coord(x, y) == "(x, y) = (foo, foo)"


def test_location_event_position_twin():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 设置主轴对象的 x 和 y 范围
    ax.set(xlim=(0, 10), ylim=(0, 20))
    # 断言主轴对象在坐标 (5., 5.) 处的格式化输出字符串
    assert ax.format_coord(5., 5.) == "(x, y) = (5.00, 5.00)"
    # 创建一个与主轴垂直的双轴对象，并设置其 y 范围
    ax.twinx().set(ylim=(0, 40))
    # 再次断言主轴对象在坐标 (5., 5.) 处的格式化输出字符串，包含双轴对象的 y 轴坐标
    assert ax.format_coord(5., 5.) == "(x, y) = (5.00, 5.00) | (5.00, 10.0)"
    # 创建一个与主轴水平的双轴对象，并设置其 x 范围
    ax.twiny().set(xlim=(0, 5))
    # 最后断言主轴对象在坐标 (5., 5.) 处的格式化输出字符串，包含所有三个轴的坐标信息
    assert (ax.format_coord(5., 5.)
            == "(x, y) = (5.00, 5.00) | (5.00, 10.0) | (2.50, 5.00)")


def test_pick():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个文本标签，并设置其具有 picker 属性
    fig.text(.5, .5, "hello", ha="center", va="center", picker=True)
    # 绘制图形画布
    fig.canvas.draw()

    # 创建一个列表来存储挑选事件
    picks = []
    # 定义一个处理挑选事件的函数
    def handle_pick(event):
        # 断言事件的鼠标事件的键值为 "a"
        assert event.mouseevent.key == "a"
        # 将事件添加到 picks 列表中
        picks.append(event)
    # 将处理挑选事件的函数连接到图形画布的挑选事件上
    fig.canvas.mpl_connect("pick_event", handle_pick)

    # 模拟一个键盘按下事件，键值为 "a"
    KeyEvent("key_press_event", fig.canvas, "a")._process()
    # 模拟一个鼠标按下事件，位置为图形坐标 (.5, .5)
    MouseEvent("button_press_event", fig.canvas,
               *fig.transFigure.transform((.5, .5)),
               MouseButton.LEFT)._process()
    # 模拟一个键盘释放事件，键值为 "a"
    KeyEvent("key_release_event", fig.canvas, "a")._process()
    # 最后断言 picks 列表中事件的数量为 1
    assert len(picks) == 1


def test_interactive_zoom():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 设置轴对象的 x 轴比例为 "logit"
    ax.set(xscale="logit")
    # 断言轴对象当前的导航模式为 None
    assert ax.get_navigate_mode() is None

    # 创建一个导航工具栏对象，传入图形画布
    tb = NavigationToolbar2(fig.canvas)
    # 使用导航工具栏对象进行放大操作
    tb.zoom()
    # 断言轴对象当前的导航模式为 'ZOOM'
    assert ax.get_navigate_mode() == 'ZOOM'

    # 获取当前轴对象的 x 和 y 轴限制范围
    xlim0 = ax.get_xlim()
    ylim0 = ax.get_ylim()

    # 定义起始缩放的数据坐标范围和结束缩放的数据坐标范围
    d0 = (1e-6, 0.1)
    d1 = (1-1e-5, 0.8)
    # 将数据坐标范围转换为屏幕坐标范围
    s0 = ax.transData.transform(d0).astype(int)
    s1 = ax.transData.transform(d1).astype(int)

    # 模拟一个鼠标按下事件，起始位置为 s0，使用左键
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *s0, MouseButton.LEFT)
    fig.canvas.callbacks.process(start_event.name, start_event)
    # 创建一个鼠标事件对象，模拟左键释放操作，使用在图形画布上的特定位置 s1
    stop_event = MouseEvent(
        "button_release_event", fig.canvas, *s1, MouseButton.LEFT)
    # 处理停止事件，触发对应的回调函数
    fig.canvas.callbacks.process(stop_event.name, stop_event)
    # 断言当前 X 轴的范围是否等于起始事件到停止事件的 X 数据范围
    assert ax.get_xlim() == (start_event.xdata, stop_event.xdata)
    # 断言当前 Y 轴的范围是否等于起始事件到停止事件的 Y 数据范围
    assert ax.get_ylim() == (start_event.ydata, stop_event.ydata)

    # 缩小视图。
    # 创建一个鼠标事件对象，模拟右键按下操作，使用在图形画布上的特定位置 s1
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *s1, MouseButton.RIGHT)
    # 处理开始事件，触发对应的回调函数
    fig.canvas.callbacks.process(start_event.name, start_event)
    # 创建一个鼠标事件对象，模拟右键释放操作，使用在图形画布上的特定位置 s0
    stop_event = MouseEvent(
        "button_release_event", fig.canvas, *s0, MouseButton.RIGHT)
    # 处理停止事件，触发对应的回调函数
    fig.canvas.callbacks.process(stop_event.name, stop_event)
    # 断言当前 X 轴的范围是否接近于初始 X 范围 xlim0，绝对公差小于 1e-10
    assert ax.get_xlim() == pytest.approx(xlim0, rel=0, abs=1e-10)
    # 断言当前 Y 轴的范围是否接近于初始 Y 范围 ylim0，绝对公差小于 1e-10
    assert ax.get_ylim() == pytest.approx(ylim0, rel=0, abs=1e-10)

    # 调用图形工具栏的缩放功能
    tb.zoom()
    # 断言当前的导航模式是否为 None
    assert ax.get_navigate_mode() is None

    # 断言 X 和 Y 轴的自动缩放是否都关闭
    assert not ax.get_autoscalex_on() and not ax.get_autoscaley_on()
def test_widgetlock_zoompan():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制一条直线
    ax.plot([0, 1], [0, 1])
    # 将绘图区域的控件锁定到指定的轴
    fig.canvas.widgetlock(ax)
    # 创建一个导航工具栏对象
    tb = NavigationToolbar2(fig.canvas)
    # 使用工具栏对象进行放大操作
    tb.zoom()
    # 断言轴的导航模式为空
    assert ax.get_navigate_mode() is None
    # 使用工具栏对象进行平移操作
    tb.pan()
    # 再次断言轴的导航模式为空
    assert ax.get_navigate_mode() is None


@pytest.mark.parametrize("plot_func", ["imshow", "contourf"])
@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("tool,button,expected",
                         [("zoom", MouseButton.LEFT, (4, 6)),  # 放大
                          ("zoom", MouseButton.RIGHT, (-20, 30)),  # 缩小
                          ("pan", MouseButton.LEFT, (-2, 8)),
                          ("pan", MouseButton.RIGHT, (1.47, 7.78))])  # 平移
def test_interactive_colorbar(plot_func, orientation, tool, button, expected):
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 创建一个示例数据数组
    data = np.arange(12).reshape((4, 3))
    vmin0, vmax0 = 0, 10
    # 根据传入的绘图函数名调用相应的绘图方法，并获取返回的绘图对象
    coll = getattr(ax, plot_func)(data, vmin=vmin0, vmax=vmax0)

    # 在图形上添加一个颜色条
    cb = fig.colorbar(coll, ax=ax, orientation=orientation)
    if plot_func == "contourf":
        # 断言颜色条轴不可导航，并直接退出测试
        assert not cb.ax.get_navigate()
        return

    # 断言颜色条轴可导航
    assert cb.ax.get_navigate()

    # 设置放大/缩小操作的数据坐标范围
    vmin, vmax = 4, 6
    # 设置数据坐标对应的像素坐标，y 坐标不重要，只需在 0 到 1 之间
    # 这里将 d0 和 d1 设置为相同的 y 坐标，用于测试像素坐标的微小变化不会取消缩放
    d0 = (vmin, 0.5)
    d1 = (vmax, 0.5)
    # 如果方向是垂直的，则交换 d0 和 d1
    if orientation == "vertical":
        d0 = d0[::-1]
        d1 = d1[::-1]
    # 将数据坐标转换为屏幕坐标
    s0 = cb.ax.transData.transform(d0).astype(int)
    s1 = cb.ax.transData.transform(d1).astype(int)

    # 设置鼠标事件的开始和停止
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *s0, button)
    stop_event = MouseEvent(
        "button_release_event", fig.canvas, *s1, button)

    # 创建导航工具栏对象
    tb = NavigationToolbar2(fig.canvas)
    if tool == "zoom":
        # 使用工具栏进行放大操作
        tb.zoom()
        tb.press_zoom(start_event)
        tb.drag_zoom(stop_event)
        tb.release_zoom(stop_event)
    else:
        # 使用工具栏进行平移操作
        tb.pan()
        tb.press_pan(start_event)
        tb.drag_pan(stop_event)
        tb.release_pan(stop_event)

    # 断言颜色条的最小值和最大值与期望值接近（精度为 0.15）
    assert (cb.vmin, cb.vmax) == pytest.approx(expected, abs=0.15)


def test_toolbar_zoompan():
    # 断言在使用工具栏之前会发出预期的用户警告
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    # 获取当前轴对象
    ax = plt.gca()
    # 断言轴的导航模式为空
    assert ax.get_navigate_mode() is None
    # 通过工具管理器触发放大工具
    ax.figure.canvas.manager.toolmanager.trigger_tool('zoom')
    # 断言当前 axes 对象的导航模式是 "ZOOM"
    assert ax.get_navigate_mode() == "ZOOM"
    # 获取 axes 对应的 figure 的画布管理器，然后触发 "pan" 工具
    ax.figure.canvas.manager.toolmanager.trigger_tool('pan')
    # 断言当前 axes 对象的导航模式是 "PAN"
    assert ax.get_navigate_mode() == "PAN"
def test_toolbar_home_restores_autoscale():
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制简单的线性图
    ax.plot(range(11), range(11))

    # 创建一个导航工具栏对象，并调用缩放功能
    tb = NavigationToolbar2(fig.canvas)
    tb.zoom()

    # 切换到对数刻度
    KeyEvent("key_press_event", fig.canvas, "k", 100, 100)._process()
    KeyEvent("key_press_event", fig.canvas, "l", 100, 100)._process()
    # 断言自动缩放后的轴限制
    assert ax.get_xlim() == ax.get_ylim() == (1, 10)  # Autolimits excluding 0.

    # 切换回线性刻度
    KeyEvent("key_press_event", fig.canvas, "k", 100, 100)._process()
    KeyEvent("key_press_event", fig.canvas, "l", 100, 100)._process()
    # 断言恢复到线性刻度后的轴限制
    assert ax.get_xlim() == ax.get_ylim() == (0, 10)  # Autolimits.

    # 从 (2, 2) 到 (5, 5) 缩放视图
    start, stop = ax.transData.transform([(2, 2), (5, 5)])
    MouseEvent("button_press_event", fig.canvas, *start, MouseButton.LEFT)._process()
    MouseEvent("button_release_event", fig.canvas, *stop, MouseButton.LEFT)._process()
    # 返回到初始视图
    KeyEvent("key_press_event", fig.canvas, "h")._process()

    # 最终断言轴限制是否恢复到初始状态
    assert ax.get_xlim() == ax.get_ylim() == (0, 10)

    # 再次切换到对数刻度
    KeyEvent("key_press_event", fig.canvas, "k", 100, 100)._process()
    KeyEvent("key_press_event", fig.canvas, "l", 100, 100)._process()
    # 断言再次自动缩放后的轴限制
    assert ax.get_xlim() == ax.get_ylim() == (1, 10)  # Autolimits excluding 0.
    # 定义一个包含多个元组的列表，每个元组描述了一个移动操作
    [
        # 元组1: 没有操作类型，目标为固定点 (3.49, 12.49)，区域范围 (2.7, 11.7)
        (None, (0.2, 0.2), (3.49, 12.49), (2.7, 11.7)),
        # 元组2: 没有操作类型，目标为固定点 (3.49, 12.49)，区域范围 (0, 9)
        (None, (0.2, 0.5), (3.49, 12.49), (0, 9)),
        # 元组3: 没有操作类型，目标为固定点 (0, 9)，区域范围 (2.7, 11.7)
        (None, (0.5, 0.2), (0, 9), (2.7, 11.7)),
        # 元组4: 没有操作类型，目标为固定点 (0, 9)，区域范围 (0, 9)，这里没有移动
        (None, (0.5, 0.5), (0, 9), (0, 9)),  # No move
        # 元组5: 没有操作类型，目标为固定点 (-3.47, 5.53)，区域范围 (2.25, 11.25)
        (None, (0.8, 0.25), (-3.47, 5.53), (2.25, 11.25)),
        # 元组6: 没有操作类型，目标为固定点 (3.49, 12.49)，区域范围 (2.25, 11.25)
        (None, (0.2, 0.25), (3.49, 12.49), (2.25, 11.25)),
        # 元组7: 没有操作类型，目标为固定点 (-3.47, 5.53)，区域范围 (-3.14, 5.86)
        (None, (0.8, 0.85), (-3.47, 5.53), (-3.14, 5.86)),
        # 元组8: 没有操作类型，目标为固定点 (3.49, 12.49)，区域范围 (-3.14, 5.86)
        (None, (0.2, 0.85), (3.49, 12.49), (-3.14, 5.86)),
        # 元组9: 操作类型为 "shift"，目标为固定点 (3.49, 12.49)，区域范围 (0, 9)，这里对 x 进行了对齐
        ("shift", (0.2, 0.4), (3.49, 12.49), (0, 9)),  # snap to x
        # 元组10: 操作类型为 "shift"，目标为固定点 (0, 9)，区域范围 (2.7, 11.7)，这里对 y 进行了对齐
        ("shift", (0.4, 0.2), (0, 9), (2.7, 11.7)),  # snap to y
        # 元组11: 操作类型为 "shift"，目标为固定点 (3.49, 12.49)，区域范围 (3.49, 12.49)，这里对角线进行了对齐
        ("shift", (0.2, 0.25), (3.49, 12.49), (3.49, 12.49)),  # snap to diagonal
        # 元组12: 操作类型为 "shift"，目标为固定点 (-3.47, 5.53)，区域范围 (3.47, 12.47)，这里对角线进行了对齐
        ("shift", (0.8, 0.25), (-3.47, 5.53), (3.47, 12.47)),  # snap to diagonal
        # 元组13: 操作类型为 "shift"，目标为固定点 (-3.58, 5.41)，区域范围 (-3.58, 5.41)，这里对角线进行了对齐
        ("shift", (0.8, 0.9), (-3.58, 5.41), (-3.58, 5.41)),  # snap to diagonal
        # 元组14: 操作类型为 "shift"，目标为固定点 (3.49, 12.49)，区域范围 (-3.49, 5.51)，这里对角线进行了对齐
        ("shift", (0.2, 0.85), (3.49, 12.49), (-3.49, 5.51)),  # snap to diagonal
        # 元组15: 操作类型为 "x"，目标为固定点 (3.49, 12.49)，区域范围 (0, 9)，这里只移动了 x 轴
        ("x", (0.2, 0.1), (3.49, 12.49), (0, 9)),  # only x
        # 元组16: 操作类型为 "y"，目标为固定点 (0, 9)，区域范围 (2.7, 11.7)，这里只移动了 y 轴
        ("y", (0.1, 0.2), (0, 9), (2.7, 11.7)),  # only y
        # 元组17: 操作类型为 "control"，目标为固定点 (3.49, 12.49)，区域范围 (3.49, 12.49)，这里对角线进行了对齐
        ("control", (0.2, 0.2), (3.49, 12.49), (3.49, 12.49)),  # diagonal
        # 元组18: 操作类型为 "control"，目标为固定点 (2.72, 11.72)，区域范围 (2.72, 11.72)，这里对角线进行了对齐
        ("control", (0.4, 0.2), (2.72, 11.72), (2.72, 11.72)),  # diagonal
    ]
# 定义一个测试函数，用于交互式平移测试
def test_interactive_pan(key, mouseend, expectedxlim, expectedylim):
    # 创建一个新的图形和轴
    fig, ax = plt.subplots()
    # 在轴上绘制一个简单的折线图
    ax.plot(np.arange(10))
    # 断言轴启用了导航功能
    assert ax.get_navigate()
    # 设置轴的等比例缩放，以便更容易看到对角线的移动
    ax.set_aspect('equal')

    # 鼠标移动的起始点为固定值 (0.5, 0.5)
    mousestart = (0.5, 0.5)
    # 将起始点转换为屏幕坐标 ("s")。事件只能使用像素精度定义，因此需要将像素值四舍五入，并与xdata/ydata进行比较，
    # 它们接近但不等于d0/d1。
    sstart = ax.transData.transform(mousestart).astype(int)
    send = ax.transData.transform(mouseend).astype(int)

    # 设置鼠标移动事件
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *sstart, button=MouseButton.LEFT,
        key=key)
    stop_event = MouseEvent(
        "button_release_event", fig.canvas, *send, button=MouseButton.LEFT,
        key=key)

    # 创建一个导航工具栏对象
    tb = NavigationToolbar2(fig.canvas)
    # 启动平移操作
    tb.pan()
    # 模拟按下平移操作
    tb.press_pan(start_event)
    # 拖动进行平移操作
    tb.drag_pan(stop_event)
    # 释放平移操作
    tb.release_pan(stop_event)
    # 断言轴的x轴和y轴的范围与期望值非常接近，由于屏幕整数分辨率的限制，不会完全相等
    assert tuple(ax.get_xlim()) == pytest.approx(expectedxlim, abs=0.02)
    assert tuple(ax.get_ylim()) == pytest.approx(expectedylim, abs=0.02)


# 定义一个测试函数，测试工具管理器中的工具移除功能
def test_toolmanager_remove():
    # 断言在使用工具管理器的情况下会发出UserWarning警告，匹配_EXPECTED_WARNING_TOOLMANAGER
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    # 获取当前图形对象
    fig = plt.gcf()
    # 记录工具管理器中工具的初始数量
    initial_len = len(fig.canvas.manager.toolmanager.tools)
    # 断言'forward'工具存在于工具管理器中
    assert 'forward' in fig.canvas.manager.toolmanager.tools
    # 移除'forward'工具
    fig.canvas.manager.toolmanager.remove_tool('forward')
    # 断言工具管理器中的工具数量减少了一个
    assert len(fig.canvas.manager.toolmanager.tools) == initial_len - 1
    # 再次断言'forward'工具不再存在于工具管理器中
    assert 'forward' not in fig.canvas.manager.toolmanager.tools


# 定义一个测试函数，测试工具管理器中的工具获取功能
def test_toolmanager_get_tool():
    # 断言在使用工具管理器的情况下会发出UserWarning警告，匹配_EXPECTED_WARNING_TOOLMANAGER
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    # 获取当前图形对象
    fig = plt.gcf()
    # 获取名为'rubberband'的工具
    rubberband = fig.canvas.manager.toolmanager.get_tool('rubberband')
    # 断言rubberband是RubberbandBase的实例
    assert isinstance(rubberband, RubberbandBase)
    # 断言获取到的rubberband工具与它本身相同
    assert fig.canvas.manager.toolmanager.get_tool(rubberband) is rubberband
    # 断言尝试获取不存在的'foo'工具会发出警告
    with pytest.warns(UserWarning,
                      match="ToolManager does not control tool 'foo'"):
        assert fig.canvas.manager.toolmanager.get_tool('foo') is None
    # 关闭警告，并再次尝试获取'foo'工具，确认返回None
    assert fig.canvas.manager.toolmanager.get_tool('foo', warn=False) is None

    # 断言尝试触发不存在的'foo'工具会发出警告
    with pytest.warns(UserWarning,
                      match="ToolManager does not control tool 'foo'"):
        assert fig.canvas.manager.toolmanager.trigger_tool('foo') is None


# 定义一个测试函数，测试工具管理器中的键映射更新功能
def test_toolmanager_update_keymap():
    # 断言在使用工具管理器的情况下会发出UserWarning警告，匹配_EXPECTED_WARNING_TOOLMANAGER
    with pytest.warns(UserWarning, match=_EXPECTED_WARNING_TOOLMANAGER):
        plt.rcParams['toolbar'] = 'toolmanager'
    # 获取当前图形对象
    fig = plt.gcf()
    # 断言'v'键在'forward'工具的键映射中
    assert 'v' in fig.canvas.manager.toolmanager.get_tool_keymap('forward')
    # 使用 pytest 模块检测警告，确保警告信息包含特定字符串 "Key c changed from back to forward"
    with pytest.warns(UserWarning, match="Key c changed from back to forward"):
        # 更新图形对象的工具管理器中 'forward' 键的映射为 'c'
        fig.canvas.manager.toolmanager.update_keymap('forward', 'c')
    
    # 断言：检查图形对象的工具管理器中 'forward' 键的映射是否为 ['c']
    assert fig.canvas.manager.toolmanager.get_tool_keymap('forward') == ['c']
    
    # 使用 pytest 模块断言，确保更新图形对象的工具管理器中 'foo' 键时抛出 KeyError 异常，异常信息包含 "'foo' not in Tools"
    with pytest.raises(KeyError, match="'foo' not in Tools"):
        # 尝试更新图形对象的工具管理器中 'foo' 键的映射为 'c'
        fig.canvas.manager.toolmanager.update_keymap('foo', 'c')
# 使用 pytest.mark.parametrize 装饰器为 test_interactive_pan_zoom_events 函数的 tool 参数设置参数化测试，依次取值 "zoom" 和 "pan"
# 为 button 参数设置参数化测试，依次取值 MouseButton.LEFT 和 MouseButton.RIGHT
# 为 patch_vis 参数设置参数化测试，依次取值 True 和 False
# 为 forward_nav 参数设置参数化测试，依次取值 True、False 和 "auto"
# 为 t_s 参数设置参数化测试，依次取值 "twin" 和 "share"
@pytest.mark.parametrize("tool", ["zoom", "pan"])
@pytest.mark.parametrize("button", [MouseButton.LEFT, MouseButton.RIGHT])
@pytest.mark.parametrize("patch_vis", [True, False])
@pytest.mark.parametrize("forward_nav", [True, False, "auto"])
@pytest.mark.parametrize("t_s", ["twin", "share"])
def test_interactive_pan_zoom_events(tool, button, patch_vis, forward_nav, t_s):
    # 创建一个新的图形对象 fig 和一个底部轴对象 ax_b
    fig, ax_b = plt.subplots()
    # 在 fig 上添加一个子图 ax_t，位置 (2,2,1)，并设置绘图层级 zorder 为 99
    ax_t = fig.add_subplot(221, zorder=99)
    # 根据参数 forward_nav 设置 ax_t 是否启用前向导航事件
    ax_t.set_forward_navigation_events(forward_nav)
    # 根据参数 patch_vis 设置 ax_t 的补丁是否可见
    ax_t.patch.set_visible(patch_vis)

    # ----------------------------
    # 根据参数 t_s 的取值进行分支处理
    if t_s == "share":
        # 在 fig 上添加第二个子图 ax_t_twin，并与 ax_t 共享 x 和 y 轴
        ax_t_twin = fig.add_subplot(222)
        ax_t_twin.sharex(ax_t)
        ax_t_twin.sharey(ax_t)

        # 在 fig 上添加第三个子图 ax_b_twin，并与 ax_b 共享 x 和 y 轴
        ax_b_twin = fig.add_subplot(223)
        ax_b_twin.sharex(ax_b)
        ax_b_twin.sharey(ax_b)
    elif t_s == "twin":
        # 在 ax_t 上创建一个与之共享 x 轴的镜像轴对象 ax_t_twin
        ax_t_twin = ax_t.twinx()
        # 在 ax_b 上创建一个与之共享 x 轴的镜像轴对象 ax_b_twin
        ax_b_twin = ax_b.twinx()

    # 对 ax_t 进行标签设置
    ax_t.set_label("ax_t")
    # 设置 ax_t 的补丁颜色为半透明红色
    ax_t.patch.set_facecolor((1, 0, 0, 0.5))

    # 对 ax_t_twin 进行标签设置
    ax_t_twin.set_label("ax_t_twin")
    # 设置 ax_t_twin 的补丁颜色为红色
    ax_t_twin.patch.set_facecolor("r")

    # 对 ax_b 进行标签设置
    ax_b.set_label("ax_b")
    # 设置 ax_b 的补丁颜色为半透明蓝色
    ax_b.patch.set_facecolor((0, 0, 1, 0.5))

    # 对 ax_b_twin 进行标签设置
    ax_b_twin.set_label("ax_b_twin")
    # 设置 ax_b_twin 的补丁颜色为蓝色
    ax_b_twin.patch.set_facecolor("b")

    # ----------------------------

    # 设置初始轴限制
    init_xlim, init_ylim = (0, 10), (0, 10)
    # 循环设置 ax_t 和 ax_b 的初始 x 和 y 轴限制
    for ax in [ax_t, ax_b]:
        ax.set_xlim(*init_xlim)
        ax.set_ylim(*init_ylim)

    # 鼠标从数据坐标系 ax_t 的 (1,1) 到 (2,2) 移动
    xstart_t, xstop_t, ystart_t, ystop_t = 1, 2, 1, 2
    # 将数据坐标转换为屏幕坐标系并四舍五入到整数值，以便与事件的像素精度进行比较
    s0 = ax_t.transData.transform((xstart_t, ystart_t)).astype(int)
    s1 = ax_t.transData.transform((xstop_t, ystop_t)).astype(int)

    # 计算在底部轴的数据坐标系中鼠标移动的距离
    xstart_b, ystart_b = ax_b.transData.inverted().transform(s0)
    xstop_b, ystop_b = ax_b.transData.inverted().transform(s1)

    # 设置鼠标事件的起始和停止事件对象，类型为 MouseEvent，模拟按下和释放鼠标按键事件
    start_event = MouseEvent("button_press_event", fig.canvas, *s0, button)
    stop_event = MouseEvent("button_release_event", fig.canvas, *s1, button)

    # 创建一个 NavigationToolbar2 对象 tb，绑定到 fig.canvas 上
    tb = NavigationToolbar2(fig.canvas)
    `
        # 如果工具是 "zoom"，执行以下操作
        if tool == "zoom":
            # 根据按钮确定放大还是缩小方向
            direction = ("in" if button == 1 else "out")
    
            # 根据指定的矩形边界计算在目标轴(ax_t)上的视图限制
            xlim_t, ylim_t = ax_t._prepare_view_from_bbox([*s0, *s1], direction)
    
            # 如果目标轴支持前向导航事件
            if ax_t.get_forward_navigation_events() is True:
                # 计算基准轴(ax_b)上的视图限制
                xlim_b, ylim_b = ax_b._prepare_view_from_bbox([*s0, *s1], direction)
            # 如果目标轴不支持前向导航事件
            elif ax_t.get_forward_navigation_events() is False:
                # 使用初始限制
                xlim_b = init_xlim
                ylim_b = init_ylim
            else:
                # 如果目标轴的补丁不可见
                if not ax_t.patch.get_visible():
                    # 计算基准轴(ax_b)上的视图限制
                    xlim_b, ylim_b = ax_b._prepare_view_from_bbox([*s0, *s1], direction)
                else:
                    # 使用初始限制
                    xlim_b = init_xlim
                    ylim_b = init_ylim
    
            # 执行缩放操作
            tb.zoom()
            # 按下缩放事件
            tb.press_zoom(start_event)
            # 拖动缩放
            tb.drag_zoom(stop_event)
            # 释放缩放
            tb.release_zoom(stop_event)
    
            # 断言目标轴(ax_t)的X和Y轴限制是否符合预期
            assert ax_t.get_xlim() == pytest.approx(xlim_t, abs=0.15)
            assert ax_t.get_ylim() == pytest.approx(ylim_t, abs=0.15)
            # 断言基准轴(ax_b)的X和Y轴限制是否符合预期
            assert ax_b.get_xlim() == pytest.approx(xlim_b, abs=0.15)
            assert ax_b.get_ylim() == pytest.approx(ylim_b, abs=0.15)
    
            # 检查双轴是否正确触发
            assert ax_t.get_xlim() == pytest.approx(ax_t_twin.get_xlim(), abs=0.15)
            assert ax_b.get_xlim() == pytest.approx(ax_b_twin.get_xlim(), abs=0.15)
        
        # 如果工具不是 "zoom"
        else:
            # 调用 start_pan 方法以确保设置了 ax._pan_start
            ax_t.start_pan(*s0, button)
            # 获取在目标轴(ax_t)上平移时的X和Y轴限制
            xlim_t, ylim_t = ax_t._get_pan_points(button, None, *s1).T.astype(float)
            # 结束平移操作
            ax_t.end_pan()
    
            # 如果目标轴支持前向导航事件
            if ax_t.get_forward_navigation_events() is True:
                ax_b.start_pan(*s0, button)
                xlim_b, ylim_b = ax_b._get_pan_points(button, None, *s1).T.astype(float)
                ax_b.end_pan()
            # 如果目标轴不支持前向导航事件
            elif ax_t.get_forward_navigation_events() is False:
                xlim_b = init_xlim
                ylim_b = init_ylim
            else:
                # 如果目标轴的补丁不可见
                if not ax_t.patch.get_visible():
                    ax_b.start_pan(*s0, button)
                    xlim_b, ylim_b = ax_b._get_pan_points(button, None, *s1).T.astype(float)
                    ax_b.end_pan()
                else:
                    xlim_b = init_xlim
                    ylim_b = init_ylim
    
            # 执行平移操作
            tb.pan()
            # 按下平移事件
            tb.press_pan(start_event)
            # 拖动平移
            tb.drag_pan(stop_event)
            # 释放平移
            tb.release_pan(stop_event)
    
            # 断言目标轴(ax_t)的X和Y轴限制是否符合预期
            assert ax_t.get_xlim() == pytest.approx(xlim_t, abs=0.15)
            assert ax_t.get_ylim() == pytest.approx(ylim_t, abs=0.15)
            # 断言基准轴(ax_b)的X和Y轴限制是否符合预期
            assert ax_b.get_xlim() == pytest.approx(xlim_b, abs=0.15)
            assert ax_b.get_ylim() == pytest.approx(ylim_b, abs=0.15)
    
            # 检查双轴是否正确触发
            assert ax_t.get_xlim() == pytest.approx(ax_t_twin.get_xlim(), abs=0.15)
            assert ax_b.get_xlim() == pytest.approx(ax_b_twin.get_xlim(), abs=0.15)
```