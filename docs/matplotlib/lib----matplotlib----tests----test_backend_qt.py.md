# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_qt.py`

```py
# 导入必要的模块和库
import copy  # 导入copy模块，用于复制对象
import importlib  # 导入importlib模块，用于动态加载模块
import os  # 导入os模块，提供与操作系统交互的功能
import signal  # 导入signal模块，用于处理信号
import sys  # 导入sys模块，提供对解释器的访问

from datetime import date, datetime  # 从datetime模块导入date和datetime类
from unittest import mock  # 从unittest模块导入mock类

import pytest  # 导入pytest测试框架

import matplotlib  # 导入matplotlib库
from matplotlib import pyplot as plt  # 从matplotlib中导入pyplot模块和plt别名
from matplotlib._pylab_helpers import Gcf  # 从matplotlib._pylab_helpers中导入Gcf类
from matplotlib import _c_internal_utils  # 导入matplotlib内部工具模块_c_internal_utils

try:
    from matplotlib.backends.qt_compat import QtGui, QtWidgets  # 尝试导入Qt相关模块
    from matplotlib.backends.qt_editor import _formlayout  # 导入Qt编辑器相关模块
except ImportError:
    pytestmark = pytest.mark.skip('No usable Qt bindings')  # 如果导入失败，标记跳过这些测试

_test_timeout = 60  # 设置测试超时时间为60秒，对于较慢的架构是一个合理安全的值


@pytest.fixture
def qt_core(request):
    from matplotlib.backends.qt_compat import QtCore  # 导入Qt核心模块
    return QtCore  # 返回QtCore对象供测试使用


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_fig_close():
    """
    测试关闭图形窗口的功能
    """
    # 保存当前的Gcf.figs状态
    init_figs = copy.copy(Gcf.figs)

    # 使用pyplot接口创建一个图形
    fig = plt.figure()

    # 模拟用户通过调用底层Qt对象的close方法点击关闭按钮
    fig.canvas.manager.window.close()

    # 断言：检查是否已经移除了由plt.figure()添加的FigureManager的引用
    assert init_figs == Gcf.figs


@pytest.mark.parametrize(
    "qt_key, qt_mods, answer",
    [
        ("Key_A", ["ShiftModifier"], "A"),
        ("Key_A", [], "a"),
        ("Key_A", ["ControlModifier"], ("ctrl+a")),
        (
            "Key_Aacute",
            ["ShiftModifier"],
            "\N{LATIN CAPITAL LETTER A WITH ACUTE}",
        ),
        ("Key_Aacute", [], "\N{LATIN SMALL LETTER A WITH ACUTE}"),
        ("Key_Control", ["AltModifier"], ("alt+control")),
        ("Key_Alt", ["ControlModifier"], "ctrl+alt"),
        (
            "Key_Aacute",
            ["ControlModifier", "AltModifier", "MetaModifier"],
            ("ctrl+alt+meta+\N{LATIN SMALL LETTER A WITH ACUTE}"),
        ),
        # 目前不映射媒体键，这可能会在将来改变。这意味着回调函数永远不会触发
        ("Key_Play", [], None),
        ("Key_Backspace", [], "backspace"),
        (
            "Key_Backspace",
            ["ControlModifier"],
            "ctrl+backspace",
        ),
    ],
    ids=[
        'shift',
        'lower',
        'control',
        'unicode_upper',
        'unicode_lower',
        'alt_control',
        'control_alt',
        'modifier_order',
        'non_unicode_key',
        'backspace',
        'backspace_mod',
    ]
)
@pytest.mark.parametrize('backend', [
    # 注意：值无关紧要，重要的是标记
    pytest.param(
        'Qt5Agg',
        marks=pytest.mark.backend('Qt5Agg', skip_on_importerror=True)),
    pytest.param(
        'QtAgg',
        marks=pytest.mark.backend('QtAgg', skip_on_importerror=True)),
])
def test_correct_key(backend, qt_core, qt_key, qt_mods, answer, monkeypatch):
    """
    测试在特定的Qt后端下，正确处理按键事件的功能
    """
    """
    创建一个图形。
    发送一个key_press_event事件（使用非公开的，特定于qtX后端的API）。
    """
    Catch the event.
    Assert sent and caught keys are the same.
    """
    # 导入必要的库函数和模块
    from matplotlib.backends.qt_compat import _to_int, QtCore

    # 如果运行平台是 macOS，并且答案不为空，则替换特定的键名
    if sys.platform == "darwin" and answer is not None:
        answer = answer.replace("ctrl", "cmd")
        answer = answer.replace("control", "cmd")
        answer = answer.replace("meta", "ctrl")
    
    # 初始化结果为 None
    result = None

    # 初始化 Qt 的键盘修饰符为无修饰符
    qt_mod = QtCore.Qt.KeyboardModifier.NoModifier
    
    # 将传入的 qt_mods 中的修饰符名对应的 Qt.KeyboardModifier 加入 qt_mod 中
    for mod in qt_mods:
        qt_mod |= getattr(QtCore.Qt.KeyboardModifier, mod)

    # 定义一个模拟事件的类 _Event
    class _Event:
        def isAutoRepeat(self): return False
        def key(self): return _to_int(getattr(QtCore.Qt.Key, qt_key))

    # 使用 monkeypatch 设置 QtWidgets.QApplication 的 keyboardModifiers 方法，
    # 返回预设的 qt_mod 作为键盘修饰符
    monkeypatch.setattr(QtWidgets.QApplication, "keyboardModifiers",
                        lambda self: qt_mod)

    # 定义按键事件的处理函数 on_key_press
    def on_key_press(event):
        nonlocal result
        result = event.key

    # 获取 matplotlib 的 Figure 对象的 canvas
    qt_canvas = plt.figure().canvas

    # 连接 'key_press_event' 事件到 on_key_press 处理函数
    qt_canvas.mpl_connect('key_press_event', on_key_press)
    
    # 模拟一个按键事件，并传入 _Event 对象
    qt_canvas.keyPressEvent(_Event())
    
    # 断言捕获到的按键与预期的答案相同
    assert result == answer


这段代码主要是模拟捕获键盘事件，并通过断言确保捕获的按键与预期的按键相同。
# 使用 pytest 的装饰器标记此函数为一个测试函数，并指定其后端为 'QtAgg'，如果导入错误则跳过测试
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_device_pixel_ratio_change():
    """
    Make sure that if the pixel ratio changes, the figure dpi changes but the
    widget remains the same logical size.
    """

    # 模拟属性路径，用于后续的属性值替换
    prop = 'matplotlib.backends.backend_qt.FigureCanvasQT.devicePixelRatioF'
    with mock.patch(prop) as p:
        p.return_value = 3

        # 创建一个尺寸为 (5, 2)，dpi 为 120 的图形对象
        fig = plt.figure(figsize=(5, 2), dpi=120)
        qt_canvas = fig.canvas
        qt_canvas.show()

        def set_device_pixel_ratio(ratio):
            p.return_value = ratio

            # 这里的值无关紧要，因为无法模拟 C++ QScreen 对象，但可以覆盖其周围的功能包装器。
            # 发送此事件仅是为了以与正常情况下相同的方式触发 Matplotlib 中的 DPI 变化处理程序。
            screen.logicalDotsPerInchChanged.emit(96)

            # 绘制图形
            qt_canvas.draw()
            qt_canvas.flush_events()

            # 确保模拟操作成功
            assert qt_canvas.device_pixel_ratio == ratio

        # 显示 QtCanvas 管理器
        qt_canvas.manager.show()
        # 获取当前窗口大小
        size = qt_canvas.size()
        # 获取窗口句柄的屏幕对象
        screen = qt_canvas.window().windowHandle().screen()
        # 设置设备像素比为 3
        set_device_pixel_ratio(3)

        # DPI 和渲染器宽度/高度发生变化
        assert fig.dpi == 360
        assert qt_canvas.renderer.width == 1800
        assert qt_canvas.renderer.height == 720

        # 实际部件大小和图形逻辑大小不变。
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()

        # 设置设备像素比为 2
        set_device_pixel_ratio(2)

        # DPI 和渲染器宽度/高度发生变化
        assert fig.dpi == 240
        assert qt_canvas.renderer.width == 1200
        assert qt_canvas.renderer.height == 480

        # 实际部件大小和图形逻辑大小不变。
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()

        # 设置设备像素比为 1.5
        set_device_pixel_ratio(1.5)

        # DPI 和渲染器宽度/高度发生变化
        assert fig.dpi == 180
        assert qt_canvas.renderer.width == 900
        assert qt_canvas.renderer.height == 360

        # 实际部件大小和图形逻辑大小不变。
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()
    # 绘制一个简单的折线图，传入的参数是一个包含两个点的列表
    ax.plot([1, 2])
    
    # 显示一个单像素的图像，传入的参数是一个包含一个像素的二维列表
    ax.imshow([[1]])
    
    # 绘制散点图，点的坐标为 (0,0), (1,1), (2,2)，颜色根据点的位置从0到2变化
    ax.scatter(range(3), range(3), c=range(3))
    
    # 使用 mock 模块替换 matplotlib 的后端库中的 _exec 方法为一个什么都不做的 lambda 函数
    with mock.patch("matplotlib.backends.qt_compat._exec", lambda obj: None):
        # 在图形的画布管理器上调用工具栏的编辑参数功能
        fig.canvas.manager.toolbar.edit_parameters()
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
# 使用 pytest.mark.backend 装饰器标记测试用例的后端为 'QtAgg'，并且在遇到 ImportError 时跳过测试
def test_figureoptions_with_datetime_axes():
    # 创建一个新的图形和轴
    fig, ax = plt.subplots()
    # 创建一个包含日期时间数据的列表
    xydata = [
        datetime(year=2021, month=1, day=1),
        datetime(year=2021, month=2, day=1)
    ]
    # 在轴上绘制日期时间数据的图形
    ax.plot(xydata, xydata)
    # 使用 mock.patch 临时替换 matplotlib 后端的函数 _exec，lambda 函数什么也不做
    with mock.patch("matplotlib.backends.qt_compat._exec", lambda obj: None):
        # 在图形的画布管理器上调用工具栏的 edit_parameters 方法
        fig.canvas.manager.toolbar.edit_parameters()


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
# 使用 pytest.mark.backend 装饰器标记测试用例的后端为 'QtAgg'，并且在遇到 ImportError 时跳过测试
def test_double_resize():
    # 检查连续调整图形大小两次后窗口大小是否保持不变
    fig, ax = plt.subplots()
    # 绘制图形的画布
    fig.canvas.draw()
    # 获取图形画布管理器的窗口对象
    window = fig.canvas.manager.window

    w, h = 3, 2
    # 设置图形的尺寸为 w x h 英寸
    fig.set_size_inches(w, h)
    # 断言图形画布的宽度和高度是否与 dpi 相关联
    assert fig.canvas.width() == w * matplotlib.rcParams['figure.dpi']
    assert fig.canvas.height() == h * matplotlib.rcParams['figure.dpi']

    # 记录调整前的窗口宽度和高度
    old_width = window.width()
    old_height = window.height()

    # 再次设置图形的尺寸为 w x h 英寸
    fig.set_size_inches(w, h)
    # 断言窗口的宽度和高度是否与之前保持一致
    assert window.width() == old_width
    assert window.height() == old_height


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
# 使用 pytest.mark.backend 装饰器标记测试用例的后端为 'QtAgg'，并且在遇到 ImportError 时跳过测试
def test_canvas_reinit():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    called = False

    def crashing_callback(fig, stale):
        nonlocal called
        # 空闲时重新绘制图形画布
        fig.canvas.draw_idle()
        called = True

    # 创建一个新的图形和轴
    fig, ax = plt.subplots()
    # 将 crashing_callback 函数设置为图形的过期回调函数
    fig.stale_callback = crashing_callback
    # 创建 FigureCanvasQTAgg 类的实例，不应该引发异常
    canvas = FigureCanvasQTAgg(fig)
    # 设置图形为过期状态
    fig.stale = True
    # 断言回调函数是否被调用过
    assert called


@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
# 使用 pytest.mark.backend 装饰器标记测试用例的后端为 'Qt5Agg'，并且在遇到 ImportError 时跳过测试
def test_form_widget_get_with_datetime_and_date_fields():
    from matplotlib.backends.backend_qt import _create_qApp
    _create_qApp()

    # 创建包含日期时间字段和日期字段的表单列表
    form = [
        ("Datetime field", datetime(year=2021, month=3, day=11)),
        ("Date field", date(year=2021, month=3, day=11))
    ]
    # 创建 _formlayout.FormWidget 类的实例
    widget = _formlayout.FormWidget(form)
    # 初始化表单部件
    widget.setup()
    # 获取表单中字段的值
    values = widget.get()
    # 断言获取的值与预期的日期时间字段和日期字段值一致
    assert values == [
        datetime(year=2021, month=3, day=11),
        date(year=2021, month=3, day=11)
    ]


def _get_testable_qt_backends():
    envs = []
    for deps, env in [
            ([qt_api], {"MPLBACKEND": "qtagg", "QT_API": qt_api})
            for qt_api in ["PyQt6", "PySide6", "PyQt5", "PySide2"]
    ]:
        reason = None
        missing = [dep for dep in deps if not importlib.util.find_spec(dep)]
        if (sys.platform == "linux" and
                not _c_internal_utils.display_is_valid()):
            reason = "$DISPLAY and $WAYLAND_DISPLAY are unset"
        elif missing:
            reason = "{} cannot be imported".format(", ".join(missing))
        elif env["MPLBACKEND"] == 'macosx' and os.environ.get('TF_BUILD'):
            reason = "macosx backend fails on Azure"
        marks = []
        if reason:
            marks.append(pytest.mark.skip(
                reason=f"Skipping {env} because {reason}"))
        envs.append(pytest.param(env, marks=marks, id=str(env)))
    return envs


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
# 使用 pytest.mark.backend 装饰器标记测试用例的后端为 'QtAgg'，并且在遇到 ImportError 时跳过测试
# 定义一个用于测试的函数，覆盖 SIGINT 信号的处理方式，接受 Qt 的核心模块作为参数
def test_fig_sigint_override(qt_core):
    # 从 matplotlib 的 Qt5 后端导入 _BackendQT5 类
    from matplotlib.backends.backend_qt5 import _BackendQT5
    
    # 创建一个新的图形窗口
    plt.figure()
    
    # 变量用于从事件循环内部访问处理程序
    event_loop_handler = None
    
    # 定义一个回调函数，在事件循环中触发：保存 SIGINT 处理程序，然后退出
    def fire_signal_and_quit():
        # 保存事件循环的信号处理程序
        nonlocal event_loop_handler
        event_loop_handler = signal.getsignal(signal.SIGINT)
        
        # 请求退出事件循环
        qt_core.QCoreApplication.exit()
    
    # 设置定时器，以退出事件循环
    qt_core.QTimer.singleShot(0, fire_signal_and_quit)
    
    # 保存原始的 SIGINT 处理程序
    original_handler = signal.getsignal(signal.SIGINT)
    
    # 定义自定义的 SIGINT 处理程序，确保其正常工作
    def custom_handler(signum, frame):
        pass
    
    # 设置 SIGINT 的处理程序为自定义处理程序
    signal.signal(signal.SIGINT, custom_handler)
    
    try:
        # 调用 matplotlib 的 Qt 后端的主循环函数，这将设置 SIGINT 并启动 Qt 事件循环
        # (这会触发定时器并退出)，然后 mainloop() 会重置 SIGINT
        matplotlib.backends.backend_qt._BackendQT.mainloop()
        
        # 断言：事件循环期间的处理程序已更改
        # (无法直接比较函数的相等性)
        assert event_loop_handler != custom_handler
        
        # 断言：当前的信号处理程序与我们之前设置的相同
        assert signal.getsignal(signal.SIGINT) == custom_handler
        
        # 再次重复测试，以验证 SIG_DFL 和 SIG_IGN 不会被覆盖
        for custom_handler in (signal.SIG_DFL, signal.SIG_IGN):
            qt_core.QTimer.singleShot(0, fire_signal_and_quit)
            signal.signal(signal.SIGINT, custom_handler)
            
            _BackendQT5.mainloop()
            
            # 断言：事件循环处理程序与设置的处理程序相同
            assert event_loop_handler == custom_handler
            # 断言：当前的信号处理程序与设置的处理程序相同
            assert signal.getsignal(signal.SIGINT) == custom_handler
    
    finally:
        # 在测试结束后，将 SIGINT 处理程序重置为测试前的原始处理程序
        signal.signal(signal.SIGINT, original_handler)


# 使用 pytest 的 backend 标记来定义一个测试函数，测试 IPython 环境的子进程中的 matplotlib
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_ipython():
    # 从 matplotlib 的测试模块中导入在子进程中运行 IPython 的函数
    from matplotlib.testing import ipython_in_subprocess
    # 在子进程中运行 IPython，并验证环境是否正确配置
    ipython_in_subprocess("qt", {(8, 24): "qtagg", (8, 15): "QtAgg", (7, 0): "Qt5Agg"})
```