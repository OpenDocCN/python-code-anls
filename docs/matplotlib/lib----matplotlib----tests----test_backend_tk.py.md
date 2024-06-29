# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_tk.py`

```py
# 导入 functools 模块，用于创建装饰器
import functools
# 导入 importlib 模块，用于动态导入模块
import importlib
# 导入 os 模块，提供与操作系统交互的功能
import os
# 导入 platform 模块，用于访问底层操作系统的信息
import platform
# 导入 subprocess 模块，用于创建和管理子进程
import subprocess
# 导入 sys 模块，提供对解释器相关的操作
import sys

# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 matplotlib 中导入 _c_internal_utils 模块
from matplotlib import _c_internal_utils
# 从 matplotlib.testing 中导入 subprocess_run_helper 函数
from matplotlib.testing import subprocess_run_helper

# 设置测试的超时时间，以秒为单位
_test_timeout = 60  # A reasonably safe value for slower architectures.


def _isolated_tk_test(success_count, func=None):
    """
    A decorator to run *func* in a subprocess and assert that it prints
    "success" *success_count* times and nothing on stderr.

    TkAgg tests seem to have interactions between tests, so isolate each test
    in a subprocess. See GH#18261
    """

    # 如果 func 为 None，则返回一个偏函数
    if func is None:
        return functools.partial(_isolated_tk_test, success_count)

    # 如果环境变量中包含 MPL_TEST_ESCAPE_HATCH，则直接返回 func 函数
    if "MPL_TEST_ESCAPE_HATCH" in os.environ:
        # set in subprocess_run_helper() below
        return func

    # 定义测试函数的装饰器，检查 tkinter 是否可用
    @pytest.mark.skipif(
        not importlib.util.find_spec('tkinter'),
        reason="missing tkinter"
    )
    # 如果在 Linux 下，并且显示变量未设置，跳过测试
    @pytest.mark.skipif(
        sys.platform == "linux" and not _c_internal_utils.display_is_valid(),
        reason="$DISPLAY and $WAYLAND_DISPLAY are unset"
    )
    # 标记预期测试失败的情况
    @pytest.mark.xfail(
        ('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and
        sys.platform == 'darwin' and sys.version_info[:2] < (3, 11),
        reason='Tk version mismatch on Azure macOS CI'
    )
    # 使用 functools.wraps 包装 func 函数，确保保留原始函数的属性
    @functools.wraps(func)
    def test_func():
        # 尝试导入 tkinter 模块，若失败则跳过测试
        pytest.importorskip('tkinter')
        try:
            # 在子进程中运行 func 函数，并设置超时时间和额外的环境变量
            proc = subprocess_run_helper(
                func, timeout=_test_timeout, extra_env=dict(
                    MPLBACKEND="TkAgg", MPL_TEST_ESCAPE_HATCH="1"))
        except subprocess.TimeoutExpired:
            pytest.fail("Subprocess timed out")
        except subprocess.CalledProcessError as e:
            pytest.fail("Subprocess failed to test intended behavior\n"
                        + str(e.stderr))
        else:
            # 在 macOS 上可能会输出与 OpenGL 相关或权限相关的无关错误，这里忽略它们
            ignored_lines = ["OpenGL", "CFMessagePort: bootstrap_register",
                             "/usr/include/servers/bootstrap_defs.h"]
            assert not [line for line in proc.stderr.splitlines()
                        if all(msg not in line for msg in ignored_lines)]
            # 断言标准输出中 "success" 出现的次数等于 success_count
            assert proc.stdout.count("success") == success_count

    return test_func


# 使用装饰器 @_isolated_tk_test(success_count=6)，运行以下函数作为测试
def test_blit():
    # 导入 matplotlib.pyplot 模块的别名 plt
    import matplotlib.pyplot as plt
    # 导入 numpy 模块的别名 np
    import numpy as np
    # 导入 matplotlib.backends.backend_tkagg 模块，隐藏导入警告
    import matplotlib.backends.backend_tkagg  # noqa
    # 从 matplotlib.backends 中导入 _backend_tk 和 _tkagg 模块
    from matplotlib.backends import _backend_tk, _tkagg

    # 创建一个图形窗口和坐标轴对象
    fig, ax = plt.subplots()
    # 获取图像对象的 TK PhotoImage 对象
    photoimage = fig.canvas._tkphoto
    # 创建一个 4x4x4 的全 1 无符号整数数组
    data = np.ones((4, 4, 4), dtype=np.uint8)
    # 测试超出边界的 blitting 操作。
    # 定义多个超出边界的矩形框坐标作为测试数据
    bad_boxes = ((-1, 2, 0, 2),
                 (2, 0, 0, 2),
                 (1, 6, 0, 2),
                 (0, 2, -1, 2),
                 (0, 2, 2, 0),
                 (0, 2, 1, 6))
    # 遍历每个超出边界的矩形框
    for bad_box in bad_boxes:
        try:
            # 尝试在 TK PhotoImage 上进行 blitting 操作
            _tkagg.blit(
                photoimage.tk.interpaddr(), str(photoimage), data,
                _tkagg.TK_PHOTO_COMPOSITE_OVERLAY, (0, 1, 2, 3), bad_box)
        except ValueError:
            # 捕获异常并打印成功消息
            print("success")

    # 测试在已销毁的画布上进行 blitting 操作。
    # 关闭图形窗口 fig
    plt.close(fig)
    # 在已销毁的画布上尝试 blitting 操作
    _backend_tk.blit(photoimage, data, (0, 1, 2, 3))
@_isolated_tk_test(success_count=1)
# 使用装饰器标记测试函数，确保在 Tkinter 环境中运行时独立性和成功计数为 1

def test_figuremanager_preserves_host_mainloop():
    import tkinter
    import matplotlib.pyplot as plt
    success = []

    def do_plot():
        plt.figure()
        plt.plot([1, 2], [3, 5])
        plt.close()
        root.after(0, legitimate_quit)

    def legitimate_quit():
        root.quit()
        success.append(True)

    root = tkinter.Tk()
    # 在主 Tkinter 线程之后，安排绘图操作
    root.after(0, do_plot)
    root.mainloop()

    if success:
        print("success")


@pytest.mark.skipif(platform.python_implementation() != 'CPython',
                    reason='PyPy does not support Tkinter threading: '
                           'https://foss.heptapod.net/pypy/pypy/-/issues/1929')
@pytest.mark.flaky(reruns=3)
@_isolated_tk_test(success_count=1)
# 使用装饰器标记测试函数，确保在 Tkinter 环境中运行时独立性和成功计数为 1

def test_figuremanager_cleans_own_mainloop():
    import tkinter
    import time
    import matplotlib.pyplot as plt
    import threading
    from matplotlib.cbook import _get_running_interactive_framework

    root = tkinter.Tk()
    plt.plot([1, 2, 3], [1, 2, 5])

    def target():
        while not 'tk' == _get_running_interactive_framework():
            time.sleep(.01)
        plt.close()
        if show_finished_event.wait():
            print('success')

    show_finished_event = threading.Event()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    plt.show(block=True)  # 测试该函数是否会挂起
    show_finished_event.set()
    thread.join()


@pytest.mark.flaky(reruns=3)
@_isolated_tk_test(success_count=0)
# 使用装饰器标记测试函数，确保在 Tkinter 环境中运行时独立性和成功计数为 0

def test_never_update():
    import tkinter
    del tkinter.Misc.update
    del tkinter.Misc.update_idletasks

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.show(block=False)

    plt.draw()  # 测试 FigureCanvasTkAgg。
    fig.canvas.toolbar.configure_subplots()  # 测试 NavigationToolbar2Tk。
    # 测试 FigureCanvasTk 的 filter_destroy 回调
    fig.canvas.get_tk_widget().after(100, plt.close, fig)

    # 检查事件队列中是否有 update() 或 update_idletasks()，功能等同于 tkinter.Misc.update。
    plt.show(block=True)

    # 注意，异常会打印到 stderr；_isolated_tk_test 会检查它们。


@_isolated_tk_test(success_count=2)
# 使用装饰器标记测试函数，确保在 Tkinter 环境中运行时独立性和成功计数为 2

def test_missing_back_button():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

    class Toolbar(NavigationToolbar2Tk):
        # 仅显示我们需要的按钮。
        toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                     t[0] in ('Home', 'Pan', 'Zoom')]

    fig = plt.figure()
    print("success")
    Toolbar(fig.canvas, fig.canvas.manager.window)  # 这里不应该抛出异常。
    print("success")


@_isolated_tk_test(success_count=1)
# 使用装饰器标记测试函数，确保在 Tkinter 环境中运行时独立性和成功计数为 1

def test_canvas_focus():
    import tkinter as tk
    import matplotlib.pyplot as plt
    success = []
    def check_focus():
        tkcanvas = fig.canvas.get_tk_widget()
        # 获取 Tkinter Canvas 对象
        # 等待绘图窗口出现
        if not tkcanvas.winfo_viewable():
            tkcanvas.wait_visibility()
        # 确保画布拥有焦点，这样它可以接收键盘事件
        if tkcanvas.focus_lastfor() == tkcanvas:
            success.append(True)
        plt.close()
        root.destroy()

    # 创建 Tkinter 根窗口
    root = tk.Tk()
    # 创建 Matplotlib 图形对象
    fig = plt.figure()
    # 绘制简单的线图
    plt.plot([1, 2, 3])
    # 在主事件循环开始前，安排显示图形
    root.after(0, plt.show)
    # 100 毫秒后执行 check_focus 函数
    root.after(100, check_focus)
    # 开始 Tkinter 主事件循环
    root.mainloop()

    # 如果 success 列表非空，则打印成功消息
    if success:
        print("success")
# 使用装饰器 @_isolated_tk_test(success_count=2) 包装函数，用于 tkinter 的单元测试
@_isolated_tk_test(success_count=2)
# 定义一个名为 test_embedding 的函数
def test_embedding():
    # 导入 tkinter 库并重命名为 tk
    import tkinter as tk
    # 从 matplotlib 的后端 TkAgg 中导入 FigureCanvasTkAgg 和 NavigationToolbar2Tk
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    # 导入 matplotlib 的基础后端模块 key_press_handler 和 Figure
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.figure import Figure

    # 创建一个 Tkinter 的根窗口对象
    root = tk.Tk()

    # 定义一个内部函数 test_figure，用于创建并展示一个 matplotlib 图形
    def test_figure(master):
        # 创建一个空白的 Figure 对象
        fig = Figure()
        # 在 Figure 对象上添加一个子图
        ax = fig.add_subplot()
        # 在子图上绘制一条简单的折线
        ax.plot([1, 2, 3])

        # 创建一个基于 Tkinter 的画布，将 matplotlib 的图形嵌入其中
        canvas = FigureCanvasTkAgg(fig, master=master)
        # 绘制画布内容
        canvas.draw()
        # 监听画布的键盘事件，将事件处理函数设置为 key_press_handler
        canvas.mpl_connect("key_press_event", key_press_handler)
        # 将 Tkinter 的画布部件打包展示在主窗口中，扩展并填充整个可用空间
        canvas.get_tk_widget().pack(expand=True, fill="both")

        # 创建一个 matplotlib 的工具栏，并将其嵌入到 Tkinter 的主窗口中
        toolbar = NavigationToolbar2Tk(canvas, master, pack_toolbar=False)
        toolbar.pack(expand=True, fill="x")

        # 隐藏之前创建的 Tkinter 画布部件和工具栏
        canvas.get_tk_widget().forget()
        toolbar.forget()

    # 调用 test_figure 函数，将主窗口对象 root 传递给它
    test_figure(root)
    # 打印成功消息
    print("success")

    # 使用不同的背景和前景颜色设置 Tkinter 主窗口的调色板
    root.tk_setPalette(background="sky blue", selectColor="midnight blue",
                       foreground="white")
    # 再次调用 test_figure 函数，展示更新后的主窗口
    test_figure(root)
    # 打印成功消息
    print("success")
```