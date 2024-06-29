# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backends_interactive.py`

```
import functools
import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

from PIL import Image

import pytest

import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.testing import subprocess_run_helper as _run_helper


class _WaitForStringPopen(subprocess.Popen):
    """
    A Popen that passes flags that allow triggering KeyboardInterrupt.
    """

    def __init__(self, *args, **kwargs):
        # 根据操作系统设置不同的创建标志
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        # 调用父类构造函数，设置环境变量和输出参数
        super().__init__(
            *args, **kwargs,
            env={**os.environ, "MPLBACKEND": "Agg", "SOURCE_DATE_EPOCH": "0"},
            stdout=subprocess.PIPE, universal_newlines=True)

    def wait_for(self, terminator):
        """Read until the terminator is reached."""
        buf = ''
        while True:
            # 逐字符读取子进程的标准输出
            c = self.stdout.read(1)
            if not c:
                # 如果没有读取到字符，抛出运行时异常
                raise RuntimeError(
                    f'Subprocess died before emitting expected {terminator!r}')
            buf += c
            if buf.endswith(terminator):
                # 当读取的内容以终止符结束时返回
                return


# Minimal smoke-testing of the backends for which the dependencies are
# PyPI-installable on CI.  They are not available for all tested Python
# versions so we don't fail on missing backends.

@functools.lru_cache
def _get_available_interactive_backends():
    # 检测Linux平台且显示无效时设为True
    _is_linux_and_display_invalid = (sys.platform == "linux" and
                                     not _c_internal_utils.display_is_valid())
    # 初始化环境变量列表
    envs = []
    # 遍历依赖和环境字典的元组列表
    for deps, env in [
            # 构建基于不同Qt API的依赖和环境
            *([([qt_api],
                {"MPLBACKEND": "qtagg", "QT_API": qt_api})
               for qt_api in ["PyQt6", "PySide6", "PyQt5", "PySide2"]],
            # 构建基于Qt API和cairocffi的依赖和环境
            *([([qt_api, "cairocffi"],
                {"MPLBACKEND": "qtcairo", "QT_API": qt_api})
               for qt_api in ["PyQt6", "PySide6", "PyQt5", "PySide2"]],
            # 构建基于cairo和gi的依赖和环境
            *[(["cairo", "gi"], {"MPLBACKEND": f"gtk{version}{renderer}"})
              for version in [3, 4] for renderer in ["agg", "cairo"]],
            # 构建基于tkinter的依赖和环境
            (["tkinter"], {"MPLBACKEND": "tkagg"}),
            # 构建基于wx的依赖和环境
            (["wx"], {"MPLBACKEND": "wx"}),
            # 构建基于wx的依赖和环境
            (["wx"], {"MPLBACKEND": "wxagg"}),
            # 构建基于matplotlib.backends._macosx的依赖和环境
            (["matplotlib.backends._macosx"], {"MPLBACKEND": "macosx"}),


这段代码主要是导入模块和定义类、函数，还有一些环境变量和依赖项的设置，通过注释解释了每一行代码的作用。
    # 定义一个函数，返回一个环境变量列表，以及对应的pytest标记列表
    def pytest_generate_tests(metafunc):
        # 如果测试项不包含参数化，则退出
        if 'envs' not in metafunc.fixturenames:
            return
        # 定义环境变量列表
        envs = []
        # 遍历测试参数化用例
        for env in metafunc.config.getini('env'):
            # 获取依赖项列表
            deps = env.get('BACKEND_DEPS', '').split(',')
            reason = None
            # 获取未找到的依赖项列表
            missing = [dep for dep in deps if not importlib.util.find_spec(dep)]
            # 如果是Linux且显示无效，则设置reason
            if _is_linux_and_display_invalid:
                reason = "$DISPLAY and $WAYLAND_DISPLAY are unset"
            # 如果有未找到的依赖项，则设置reason
            elif missing:
                reason = "{} cannot be imported".format(", ".join(missing))
            # 如果MPLBACKEND为macosx且在Azure上，则设置reason
            elif env["MPLBACKEND"] == 'macosx' and os.environ.get('TF_BUILD'):
                reason = "macosx backend fails on Azure"
            # 如果MPLBACKEND以gtk开头，则尝试导入gi库，检查GTK版本
            elif env["MPLBACKEND"].startswith('gtk'):
                import gi  # type: ignore
                version = env["MPLBACKEND"][3]
                repo = gi.Repository.get_default()
                # 如果找不到指定版本的GTK绑定，则设置reason
                if f'{version}.0' not in repo.enumerate_versions('Gtk'):
                    reason = "no usable GTK bindings"
            marks = []
            # 如果存在reason，则添加pytest标记，跳过测试
            if reason:
                marks.append(pytest.mark.skip(reason=f"Skipping {env} because {reason}"))
            # 如果MPLBACKEND以wx开头且在macOS平台上，则添加xfail标记，忽略此测试
            elif env["MPLBACKEND"].startswith('wx') and sys.platform == 'darwin':
                marks.append(pytest.mark.xfail(reason='github #16849'))
            # 如果MPLBACKEND为tkagg且在Azure macOS CI环境下Tk版本不匹配，则添加xfail标记
            elif (env['MPLBACKEND'] == 'tkagg' and
                  ('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and
                  sys.platform == 'darwin' and
                  sys.version_info[:2] < (3, 11)
                  ):
                marks.append(pytest.mark.xfail(reason='Tk version mismatch on Azure macOS CI'))
            # 将当前环境变量和其对应的标记列表添加到环境变量列表中
            envs.append(({**env, 'BACKEND_DEPS': ','.join(deps)}, marks))
        # 返回环境变量列表
        return envs
# 返回一个包含可测试的交互式后端的列表。重新创建此列表是因为下面的调用者可能会修改标记。
def _get_testable_interactive_backends():
    return [
        # 使用 pytest.param 创建参数化的对象，其中包含环境变量和标记列表
        pytest.param({**env}, marks=[*marks],
                     id='-'.join(f'{k}={v}' for k, v in env.items()))
        for env, marks in _get_available_interactive_backends()]


# 检查当前是否运行在持续集成环境中
def is_ci_environment():
    # 常见的持续集成环境变量列表
    ci_environment_variables = [
        'CI',        # 通用的持续集成环境变量
        'CONTINUOUS_INTEGRATION',  # 通用的持续集成环境变量
        'TRAVIS',    # Travis CI
        'CIRCLECI',  # CircleCI
        'JENKINS',   # Jenkins
        'GITLAB_CI',  # GitLab CI
        'GITHUB_ACTIONS',  # GitHub Actions
        'TEAMCITY_VERSION'  # TeamCity
        # 需要的其他持续集成环境变量可以继续添加
    ]

    # 检查是否有任何一个环境变量被设置
    for env_var in ci_environment_variables:
        if os.getenv(env_var):
            return True

    return False


# 根据是否处于持续集成环境来设置合理的测试超时时间
_test_timeout = 120 if is_ci_environment() else 20


# 测试工具栏按钮图标在LA模式下的情况（GH问题25174）
def _test_toolbar_button_la_mode_icon(fig):
    # 在临时目录中创建一个LA模式的图标
    with tempfile.TemporaryDirectory() as tempdir:
        img = Image.new("LA", (26, 26))  # 创建一个LA模式的26x26大小的图像
        tmp_img_path = os.path.join(tempdir, "test_la_icon.png")  # 临时图像文件路径
        img.save(tmp_img_path)  # 保存图像到临时文件

        # 定义一个自定义工具类，使用临时图像作为图标
        class CustomTool(ToolToggleBase):
            image = tmp_img_path
            description = ""  # gtk3后端不允许为None

        # 获取图形管理器的工具管理器和工具栏，添加自定义工具
        toolmanager = fig.canvas.manager.toolmanager
        toolbar = fig.canvas.manager.toolbar
        toolmanager.add_tool("test", CustomTool)
        toolbar.add_tool("test", "group")


# 此函数的源代码会被提取出来在另一个进程中运行，因此必须是完全自包含的。
# 使用定时器不仅允许测试定时器（在其他后端上），而且在gtk3和wx中是必要的，
# 因为直接处理来自draw_event的KeyEvent("q")可能会导致在删除画布小部件过早时出现故障。
def _test_interactive_impl():
    import importlib.util
    import io
    import json
    import sys

    import pytest

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.backend_bases import KeyEvent

    # 更新matplotlib的全局配置参数
    mpl.rcParams.update({
        "webagg.open_in_browser": False,
        "webagg.port_retries": 1,
    })

    # 从命令行参数中更新matplotlib的配置
    mpl.rcParams.update(json.loads(sys.argv[1]))

    backend = plt.rcParams["backend"].lower()  # 获取小写的后端名
    # 检查后端是否以 "agg" 结尾且不以 "gtk" 或 "web" 开头
    if backend.endswith("agg") and not backend.startswith(("gtk", "web")):
        # 强制设置交互式框架
        plt.figure()

        # 检查是否无法切换到使用另一个交互式框架的后端，但可以切换到使用 cairo 而非 agg 的后端，
        # 或者非交互式后端。在第一种情况下，将 tkagg 作为“其他”交互式后端，因为它（基本上）保证已经存在。
        # 此外，不测试从 gtk3（因为此时尚未设置 Gtk.main_level()）和 webagg（不使用交互式框架）切换的情况。
        if backend != "tkagg":
            with pytest.raises(ImportError):
                mpl.use("tkagg", force=True)

        # 定义检查备用后端的函数
        def check_alt_backend(alt_backend):
            # 强制使用备用后端
            mpl.use(alt_backend, force=True)
            # 创建新的图形对象
            fig = plt.figure()
            # 断言图形对象的画布类型符合预期的后端模块
            assert (type(fig.canvas).__module__ ==
                    f"matplotlib.backends.backend_{alt_backend}")
            # 关闭所有图形窗口
            plt.close("all")

        # 如果 cairocffi 可用，则检查使用 cairo 的备用后端
        if importlib.util.find_spec("cairocffi"):
            check_alt_backend(backend[:-3] + "cairo")
        # 检查使用 svg 的备用后端
        check_alt_backend("svg")

    # 强制使用指定的后端
    mpl.use(backend, force=True)

    # 创建包含单个轴的新图形对象
    fig, ax = plt.subplots()

    # 断言图形对象的画布类型符合预期的后端模块
    assert type(fig.canvas).__module__ == f"matplotlib.backends.backend_{backend}"

    # 断言图形窗口的标题是 "Figure 1"
    assert fig.canvas.manager.get_window_title() == "Figure 1"

    # 如果使用 toolmanager 作为工具栏，则测试工具栏按钮图标 LA 模式
    if mpl.rcParams["toolbar"] == "toolmanager":
        _test_toolbar_button_la_mode_icon(fig)

    # 在轴上绘制简单的线条
    ax.plot([0, 1], [2, 3])

    # 如果存在工具栏，则绘制橡皮筋框
    if fig.canvas.toolbar:  # 即 toolbar2。
        fig.canvas.toolbar.draw_rubberband(None, 1., 1, 2., 2)

    # 创建新的定时器对象，测试浮点数是否会被转换为整数
    timer = fig.canvas.new_timer(1.)
    # 添加回调函数以处理按键事件
    timer.add_callback(KeyEvent("key_press_event", fig.canvas, "q")._process)
    # 在绘制事件发生时启动定时器
    fig.canvas.mpl_connect("draw_event", lambda event: timer.start())
    # 监听窗口关闭事件
    fig.canvas.mpl_connect("close_event", print)

    # 创建一个 BytesIO 对象，用于保存图形为 PNG 格式
    result = io.BytesIO()
    fig.savefig(result, format='png')

    # 显示图形窗口
    plt.show()

    # 确保窗口已经真正关闭
    plt.pause(0.5)

    # 测试在交互式窗口关闭后保存是否正常工作，但图形对象未被删除
    result_after = io.BytesIO()
    fig.savefig(result_after, format='png')

    # 如果后端不以 'qt5' 开头且操作系统是 macOS，则断言两次保存的结果是否一致
    if not backend.startswith('qt5') and sys.platform == 'darwin':
        # FIXME: 这应该在 macOS 上修复 Qt5 后在所有地方启用
        # 以避免不正确的调整大小。
        assert result.getvalue() == result_after.getvalue()
@pytest.mark.parametrize("env", _get_testable_interactive_backends())
# 使用 `_get_testable_interactive_backends()` 函数返回的参数化测试环境
@pytest.mark.parametrize("toolbar", ["toolbar2", "toolmanager"])
# 参数化测试工具栏，包括 'toolbar2' 和 'toolmanager'
@pytest.mark.flaky(reruns=3)
# 标记此测试为有可能失败且重试3次的测试

def test_interactive_backend(env, toolbar):
    # 如果环境变量中的 MPLBACKEND 是 'macosx'
    if env["MPLBACKEND"] == "macosx":
        # 如果工具栏是 'toolmanager'
        if toolbar == "toolmanager":
            pytest.skip("toolmanager is not implemented for macosx.")
            # 跳过测试，输出跳过信息
    # 如果环境变量中的 MPLBACKEND 是 'wx'
    if env["MPLBACKEND"] == "wx":
        pytest.skip("wx backend is deprecated; tests failed on appveyor")
        # 跳过测试，输出跳过信息

    try:
        # 运行 _run_helper 函数，测试交互式实现
        proc = _run_helper(
            _test_interactive_impl,
            json.dumps({"toolbar": toolbar}),
            timeout=_test_timeout,
            extra_env=env,
        )
    except subprocess.CalledProcessError as err:
        # 捕获 subprocess 调用错误，输出失败信息和错误信息
        pytest.fail(
            "Subprocess failed to test intended behavior\n"
            + str(err.stderr))
    # 断言 stdout 中 "CloseEvent" 出现的次数为 1
    assert proc.stdout.count("CloseEvent") == 1


def _test_thread_impl():
    from concurrent.futures import ThreadPoolExecutor
    # 导入 matplotlib 库
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.rcParams.update({
        "webagg.open_in_browser": False,
        "webagg.port_retries": 1,
    })
    # 更新 matplotlib 的配置参数

    # 测试在线程中创建和绘制艺术家不会崩溃
    # 没有其他保证！
    fig, ax = plt.subplots()
    # plt.pause 比 plt.show(block=False) 更需要 toolbar2-tkagg
    plt.pause(0.5)

    # 使用线程池执行 ax.plot([1, 3, 6])，并获取执行结果
    future = ThreadPoolExecutor().submit(ax.plot, [1, 3, 6])
    future.result()  # 加入线程，重新抛出任何异常。

    # 连接 fig.canvas 的 "close_event" 事件到 print 函数
    fig.canvas.mpl_connect("close_event", print)
    # 使用线程池执行 fig.canvas.draw()，并获取执行结果
    future = ThreadPoolExecutor().submit(fig.canvas.draw)
    plt.pause(0.5)  # 至少在 Tkagg 上的 flush_events 失败（bpo-41176）
    future.result()  # 加入线程，重新抛出任何异常。
    plt.close()  # 后端负责在这里刷新任何事件
    if plt.rcParams["backend"].lower().startswith("wx"):
        # TODO: 调试为什么 WX 只在 py >= 3.8 需要这个
        fig.canvas.flush_events()


_thread_safe_backends = _get_testable_interactive_backends()
# 获取可进行线程安全测试的后端列表
# 已知不安全的后端。如果开始通过测试，请移除 xfails！
for param in _thread_safe_backends:
    backend = param.values[0]["MPLBACKEND"]
    if "cairo" in backend:
        # Cairo 后端在图形上下文中保存 cairo_t，共享这些不是线程安全的。
        param.marks.append(
            pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == "wx":
        # WX 后端的标记 xfail
        param.marks.append(
            pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == "macosx":
        from packaging.version import parse
        mac_ver = platform.mac_ver()[0]
        # 注意，macOS Big Sur 是 11 和 10.16，取决于 Python 编译时的 SDK。
        if mac_ver and parse(mac_ver) < parse('10.16'):
            # 如果 macOS 版本小于 10.16，则标记为 xfail
            param.marks.append(
                pytest.mark.xfail(raises=subprocess.TimeoutExpired,
                                  strict=True))
    # 如果环境变量 QT_API 的值为 "PySide2"，则将一个 xfail 标记附加到 param.marks 列表中，用于处理 subprocess.CalledProcessError 异常
    elif param.values[0].get("QT_API") == "PySide2":
        param.marks.append(
            pytest.mark.xfail(raises=subprocess.CalledProcessError))

    # 如果后端为 "tkagg" 并且 Python 解释器不是 CPython，则将一个 xfail 标记附加到 param.marks 列表中，
    # 提示 PyPy 不支持 Tkinter 的多线程，同时标记为严格模式（strict=True）
    elif backend == "tkagg" and platform.python_implementation() != 'CPython':
        param.marks.append(
            pytest.mark.xfail(
                reason='PyPy does not support Tkinter threading: '
                       'https://foss.heptapod.net/pypy/pypy/-/issues/1929',
                strict=True))

    # 如果后端为 "tkagg"，并且满足以下条件：
    # - 存在环境变量 'TF_BUILD' 或 'GITHUB_ACTION'
    # - 运行平台为 macOS
    # - Python 版本低于 3.11
    # 则将一个 xfail 标记附加到 param.marks 列表中，指明在 Azure macOS CI 中 Tk 版本不匹配的问题
    elif (backend == 'tkagg' and
          ('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and
          sys.platform == 'darwin' and sys.version_info[:2] < (3, 11)):
        param.marks.append(  # https://github.com/actions/setup-python/issues/649
            pytest.mark.xfail('Tk version mismatch on Azure macOS CI'))
# 使用 pytest.mark.parametrize 将测试参数化，参数为 _thread_safe_backends 列表中的每个环境
# 使用 pytest.mark.flaky 标记此测试为 flaky，重新运行次数为 3
@pytest.mark.parametrize("env", _thread_safe_backends)
@pytest.mark.flaky(reruns=3)
def test_interactive_thread_safety(env):
    # 运行 _run_helper 函数执行 _test_thread_impl 测试，设置超时时间为 _test_timeout，传入额外的环境参数 env
    proc = _run_helper(_test_thread_impl, timeout=_test_timeout, extra_env=env)
    # 断言 proc.stdout 中包含 "CloseEvent" 的次数为 1
    assert proc.stdout.count("CloseEvent") == 1


# 定义测试函数 _impl_test_lazy_auto_backend_selection
def _impl_test_lazy_auto_backend_selection():
    import matplotlib
    import matplotlib.pyplot as plt
    # 只导入 pyplot 不足以触发后端的解析
    bk = matplotlib.rcParams._get('backend')
    # 断言 bk 不是字符串类型
    assert not isinstance(bk, str)
    # 断言 plt._backend_mod 为 None
    assert plt._backend_mod is None
    # 实际绘图后应当触发
    plt.plot(5)
    # 断言 plt._backend_mod 不为 None
    assert plt._backend_mod is not None
    # 再次获取后端设置 bk
    bk = matplotlib.rcParams._get('backend')
    # 断言 bk 是字符串类型
    assert isinstance(bk, str)


# 定义测试函数 test_lazy_auto_backend_selection
def test_lazy_auto_backend_selection():
    # 运行 _run_helper 执行 _impl_test_lazy_auto_backend_selection 测试，设置超时时间为 _test_timeout
    _run_helper(_impl_test_lazy_auto_backend_selection,
                timeout=_test_timeout)


# 定义测试函数 _implqt5agg
def _implqt5agg():
    import matplotlib.backends.backend_qt5agg  # noqa
    import sys
    # 断言 'PyQt6' 不在 sys.modules 中
    assert 'PyQt6' not in sys.modules
    # 断言 'pyside6' 不在 sys.modules 中
    assert 'pyside6' not in sys.modules
    # 断言 'PyQt5' 在 sys.modules 中或者 'pyside2' 在 sys.modules 中
    assert 'PyQt5' in sys.modules or 'pyside2' in sys.modules


# 定义测试函数 _implcairo
def _implcairo():
    import matplotlib.backends.backend_qt5cairo  # noqa
    import sys
    # 断言 'PyQt6' 不在 sys.modules 中
    assert 'PyQt6' not in sys.modules
    # 断言 'pyside6' 不在 sys.modules 中
    assert 'pyside6' not in sys.modules
    # 断言 'PyQt5' 在 sys.modules 中或者 'pyside2' 在 sys.modules 中
    assert 'PyQt5' in sys.modules or 'pyside2' in sys.modules


# 定义测试函数 _implcore
def _implcore():
    import matplotlib.backends.backend_qt5  # noqa
    import sys
    # 断言 'PyQt6' 不在 sys.modules 中
    assert 'PyQt6' not in sys.modules
    # 断言 'pyside6' 不在 sys.modules 中
    assert 'pyside6' not in sys.modules
    # 断言 'PyQt5' 在 sys.modules 中或者 'pyside2' 在 sys.modules 中
    assert 'PyQt5' in sys.modules or 'pyside2' in sys.modules


# 定义测试函数 test_qt5backends_uses_qt5
def test_qt5backends_uses_qt5():
    # 根据 ['PyQt5', 'pyside2'] 列表中的依赖项，筛选出 importlib.util.find_spec 不为 None 的项，存入 qt5_bindings 列表
    qt5_bindings = [
        dep for dep in ['PyQt5', 'pyside2']
        if importlib.util.find_spec(dep) is not None
    ]
    # 根据 ['PyQt6', 'pyside6'] 列表中的依赖项，筛选出 importlib.util.find_spec 不为 None 的项，存入 qt6_bindings 列表
    qt6_bindings = [
        dep for dep in ['PyQt6', 'pyside6']
        if importlib.util.find_spec(dep) is not None
    ]
    # 如果 qt5_bindings 或 qt6_bindings 中的任一列表为空，则跳过此测试
    if len(qt5_bindings) == 0 or len(qt6_bindings) == 0:
        pytest.skip('need both QT6 and QT5 bindings')
    # 运行 _run_helper 执行 _implqt5agg 测试，设置超时时间为 _test_timeout
    _run_helper(_implqt5agg, timeout=_test_timeout)
    # 如果 importlib.util.find_spec('pycairo') 不为 None，则运行 _run_helper 执行 _implcairo 测试，设置超时时间为 _test_timeout
    if importlib.util.find_spec('pycairo') is not None:
        _run_helper(_implcairo, timeout=_test_timeout)
    # 运行 _run_helper 执行 _implcore 测试，设置超时时间为 _test_timeout
    _run_helper(_implcore, timeout=_test_timeout)


# 定义测试函数 _impl_missing
def _impl_missing():
    import sys
    # 模拟未安装的情况
    sys.modules["PyQt6"] = None
    sys.modules["PyQt5"] = None
    sys.modules["PySide2"] = None
    sys.modules["PySide6"] = None

    import matplotlib.pyplot as plt
    # 使用 pytest.raises 断言抛出 ImportError 异常，并匹配错误消息中包含 "Failed to import any of the following Qt"
    with pytest.raises(ImportError, match="Failed to import any of the following Qt"):
        plt.switch_backend("qtagg")
    # 确保错误消息中不包含 'PySide6' 或 'PyQt6'
    with pytest.raises(ImportError, match="^(?:(?!(PySide6|PyQt6)).)*$"):
        plt.switch_backend("qt5agg")


# 定义测试函数 test_qt_missing
def test_qt_missing():
    # 运行 _run_helper 执行 _impl_missing 测试，设置超时时间为 _test_timeout
    _run_helper(_impl_missing, timeout=_test_timeout)


# 定义测试函数 _impl_test_cross_Qt_imports
def _impl_test_cross_Qt_imports():
    import importlib
    import sys
    import warnings

    _, host_binding, mpl_binding = sys.argv
    # 导入 mpl 绑定，强制使用该绑定
    # 导入指定的模块，并使用动态字符串构建模块名称
    importlib.import_module(f'{mpl_binding}.QtCore')

    # 导入指定的模块，并使用动态字符串构建模块名称
    mpl_binding_qwidgets = importlib.import_module(f'{mpl_binding}.QtWidgets')

    # 导入 matplotlib 的 Qt 后端模块
    import matplotlib.backends.backend_qt

    # 导入指定的模块，并使用动态字符串构建模块名称
    host_qwidgets = importlib.import_module(f'{host_binding}.QtWidgets')

    # 创建一个 Qt 应用程序对象，设置应用程序名称为 "mpl testing"
    host_app = host_qwidgets.QApplication(["mpl testing"])

    # 设置警告过滤器，当警告信息匹配特定的正则表达式时，将其视为错误
    warnings.filterwarnings("error", message=r".*Mixing Qt major.*", category=UserWarning)

    # 调用 matplotlib 的 Qt 后端模块的函数，确保创建 Qt 应用程序对象
    matplotlib.backends.backend_qt._create_qApp()
# 定义一个生成器函数，用于生成 Qt5 和 Qt6 组合的参数对
def qt5_and_qt6_pairs():
    # 检查并收集 Qt5 相关的库
    qt5_bindings = [
        dep for dep in ['PyQt5', 'PySide2']
        if importlib.util.find_spec(dep) is not None
    ]
    # 检查并收集 Qt6 相关的库
    qt6_bindings = [
        dep for dep in ['PyQt6', 'PySide6']
        if importlib.util.find_spec(dep) is not None
    ]
    
    # 如果 Qt5 或 Qt6 的绑定列表为空，则跳过测试并标记
    if len(qt5_bindings) == 0 or len(qt6_bindings) == 0:
        yield pytest.param(None, None,
                           marks=[pytest.mark.skip('need both QT6 and QT5 bindings')])
        return
    
    # 生成 Qt5 和 Qt6 组合的所有可能对
    for qt5 in qt5_bindings:
        for qt6 in qt6_bindings:
            for pair in ([qt5, qt6], [qt6, qt5]):
                yield pair


# 使用 pytest 的参数化装饰器，传入 qt5_and_qt6_pairs 函数生成的参数对
@pytest.mark.parametrize('host, mpl', [*qt5_and_qt6_pairs()])
def test_cross_Qt_imports(host, mpl):
    try:
        # 运行测试辅助函数 _impl_test_cross_Qt_imports，并获取进程对象
        proc = _run_helper(_impl_test_cross_Qt_imports, host, mpl,
                           timeout=_test_timeout)
    except subprocess.CalledProcessError as ex:
        # 如果进程发生错误，捕获 stderr
        stderr = ex.stderr
    else:
        # 如果进程正常运行，获取 stderr
        stderr = proc.stderr
    
    # 断言 stderr 中包含警告信息
    assert "Mixing Qt major versions may not work as expected." in stderr


# 标记为在特定环境下跳过测试的装饰器，检查是否在 Azure 环境中
@pytest.mark.skipif('TF_BUILD' in os.environ,
                    reason="this test fails an azure for unknown reasons")
# 标记为在特定操作系统平台下跳过测试的装饰器，仅适用于 Windows
@pytest.mark.skipif(sys.platform == "win32", reason="Cannot send SIGINT on Windows.")
def test_webagg():
    # 导入 tornado，如果不存在则跳过测试
    pytest.importorskip("tornado")
    # 启动子进程执行特定命令，运行 webagg 相关测试
    proc = subprocess.Popen(
        [sys.executable, "-c",
         inspect.getsource(_test_interactive_impl)
         + "\n_test_interactive_impl()", "{}"],
        env={**os.environ, "MPLBACKEND": "webagg", "SOURCE_DATE_EPOCH": "0"})
    # 构建 webagg 服务的 URL 地址
    url = f'http://{mpl.rcParams["webagg.address"]}:{mpl.rcParams["webagg.port"]}'
    # 设置超时时间
    timeout = time.perf_counter() + _test_timeout
    try:
        while True:
            try:
                # 检查服务器子进程是否存活
                retcode = proc.poll()
                assert retcode is None
                # 尝试连接 webagg 服务器
                conn = urllib.request.urlopen(url)
                break
            except urllib.error.URLError:
                # 如果连接失败，检查超时时间
                if time.perf_counter() > timeout:
                    pytest.fail("Failed to connect to the webagg server.")
                else:
                    continue
        conn.close()
        # 发送信号给子进程
        proc.send_signal(signal.SIGINT)
        # 等待子进程结束
        assert proc.wait(timeout=_test_timeout) == 0
    finally:
        # 如果子进程仍在运行，强制结束子进程
        if proc.poll() is None:
            proc.kill()


# 内部函数，用于模拟无头环境运行
def _lazy_headless():
    import os
    import sys
    
    # 读取命令行参数，解析后端和依赖项
    backend, deps = sys.argv[1:]
    deps = deps.split(',')
    
    # 移除环境变量 DISPLAY 和 WAYLAND_DISPLAY，使其看起来像是无头环境
    os.environ.pop('DISPLAY', None)
    os.environ.pop('WAYLAND_DISPLAY', None)
    
    # 确保所有依赖项都不在 sys.modules 中
    for dep in deps:
        assert dep not in sys.modules
    
    # 确保 matplotlib 使用 Agg 后端
    import matplotlib.pyplot as plt
    assert plt.get_backend() == 'agg'
    # 检查每个依赖项是否已经加载到 sys.modules 中
    for dep in deps:
        assert dep not in sys.modules

    # 确保所有依赖项都已经安装并加载
    for dep in deps:
        # 动态导入每个依赖项的模块
        importlib.import_module(dep)
        # 断言该依赖项现在在 sys.modules 中
        assert dep in sys.modules

    # 尝试切换绘图后端，并确保预期会抛出 ImportError 异常
    try:
        plt.switch_backend(backend)
    except ImportError:
        # 如果成功捕获到 ImportError 异常，则继续执行
        pass
    else:
        # 如果没有捕获到 ImportError 异常，则退出程序，返回状态码 1
        sys.exit(1)
# 标记测试为仅限 Linux 平台，如果不是 Linux 则跳过
@pytest.mark.skipif(sys.platform != "linux", reason="this a linux-only test")
# 使用参数化测试，参数为获取的可交互后端环境列表
@pytest.mark.parametrize("env", _get_testable_interactive_backends())
def test_lazy_linux_headless(env):
    # 运行测试辅助函数，执行懒加载无头模式
    proc = _run_helper(
        _lazy_headless,
        env.pop('MPLBACKEND'), env.pop("BACKEND_DEPS"),
        timeout=_test_timeout,
        extra_env={**env, 'DISPLAY': '', 'WAYLAND_DISPLAY': ''}
    )


def _test_number_of_draws_script():
    import matplotlib.pyplot as plt

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    # 创建线条对象，设置为动画模式，仅在显式请求时绘制图形
    ln, = ax.plot([0, 1], [1, 2], animated=True)

    # 显示图形窗口，但脚本继续执行
    plt.show(block=False)
    # 暂停一段时间
    plt.pause(0.3)
    # 连接到绘制事件，用于计算事件发生的次数
    fig.canvas.mpl_connect('draw_event', print)

    # 获取整个图形区域内的图像副本（不包括动画对象）
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    # 绘制动画对象，使用缓存的渲染器
    ax.draw_artist(ln)
    # 将结果显示到屏幕上
    fig.canvas.blit(fig.bbox)

    # 循环执行以下操作10次
    for j in range(10):
        # 恢复画布到之前保存的状态，屏幕未改变
        fig.canvas.restore_region(bg)
        # 在此处创建一个新的艺术家对象，这是 blitting 的糟糕使用方式，
        # 但对于测试来说很好，以确保不会创建过多的绘制操作
        ln, = ax.plot([0, 1], [1, 2])
        # 渲染艺术家对象，更新画布状态，但不更新屏幕
        ax.draw_artist(ln)
        # 将图像复制到 GUI 状态，但屏幕可能尚未更改
        fig.canvas.blit(fig.bbox)
        # 刷新任何挂起的 GUI 事件，必要时重新绘制屏幕
        fig.canvas.flush_events()

    # 让事件循环处理所有内容后再离开
    plt.pause(0.1)


# 获取可交互后端环境的测试列表
_blit_backends = _get_testable_interactive_backends()
# 遍历每个后端参数
for param in _blit_backends:
    backend = param.values[0]["MPLBACKEND"]
    # 如果后端为 gtk3cairo，添加跳过标记，因为不支持 blitting
    if backend == "gtk3cairo":
        param.marks.append(
            pytest.mark.skip("gtk3cairo does not support blitting"))
    # 如果后端为 gtk4cairo，添加跳过标记，因为不支持 blitting
    elif backend == "gtk4cairo":
        param.marks.append(
            pytest.mark.skip("gtk4cairo does not support blitting"))
    # 如果后端为 wx，添加跳过标记，因为不支持 blitting
    elif backend == "wx":
        param.marks.append(
            pytest.mark.skip("wx does not support blitting"))
    # 如果条件满足，则添加标记，表明在特定环境中测试可能失败
    elif (backend == 'tkagg' and
          ('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and
          sys.platform == 'darwin' and
          sys.version_info[:2] < (3, 11)
          ):
        param.marks.append(
            pytest.mark.xfail('Tk version mismatch on Azure macOS CI')
        )

# 使用参数化测试，参数为支持 blitting 的后端环境
@pytest.mark.parametrize("env", _blit_backends)
# 子进程可能在获取显示时遇到困难，因此重新运行几次
@pytest.mark.parametrize("env", _get_testable_interactive_backends())
# 使用参数化测试，env 是从 _get_testable_interactive_backends() 返回的每个环境变量字典
@pytest.mark.parametrize("target, kwargs", [
    ('show', {'block': True}),
    # 参数化测试的一部分，测试 matplotlib.pyplot 的 show 函数，传递 block=True 参数
    ('pause', {'interval': 1}),
    # 参数化测试的一部分，测试 matplotlib.pyplot 的 pause 函数，传递 interval=1 参数
])
def test_interactive_timers(env, target, kwargs):
    # 如果环境的 MPLBACKEND 是 'gtk3cairo' 且在 CI 中，跳过测试
    if env["MPLBACKEND"] == "gtk3cairo" and os.getenv("CI"):
        pytest.skip("gtk3cairo timers do not work in remote CI")
    # 如果环境的 MPLBACKEND 是 'wx'，跳过测试
    if env["MPLBACKEND"] == "wx":
        pytest.skip("wx backend is deprecated; tests failed on appveyor")
    # 运行 _impl_test_interactive_timers 函数作为 helper 函数，传递环境变量和超时参数
    _run_helper(_impl_test_interactive_timers,
                timeout=_test_timeout, extra_env=env)
    ('pause', {'interval': 10})


# 创建一个元组 ('pause', {'interval': 10})
# 元组中包含一个字符串 'pause' 和一个字典 {'interval': 10}
# 结束 test_other_signal_before_sigint 函数定义
])
# 定义 test_sigint 函数，接受 env, target, kwargs 三个参数
def test_sigint(env, target, kwargs):
    # 获取环境变量 MPLBACKEND 的值
    backend = env.get("MPLBACKEND")
    # 如果 backend 不是以 "qt" 或 "macosx" 开头，则跳过测试
    if not backend.startswith(("qt", "macosx")):
        pytest.skip("SIGINT currently only tested on qt and macosx")
    # 构建一个子进程对象，用于执行 _test_sigint_impl 函数的代码
    proc = _WaitForStringPopen(
        [sys.executable, "-c",
         inspect.getsource(_test_sigint_impl) +
         f"\n_test_sigint_impl({backend!r}, {target!r}, {kwargs!r})"])
    try:
        # 等待子进程输出 'DRAW'
        proc.wait_for('DRAW')
        # 与子进程通信，获取标准输出和错误输出，设置超时时间为 _test_timeout
        stdout, _ = proc.communicate(timeout=_test_timeout)
    except Exception:
        # 发生异常时终止子进程
        proc.kill()
        # 再次获取子进程的输出
        stdout, _ = proc.communicate()
        # 将异常向上抛出
        raise
    # 断言标准输出中包含 'SUCCESS'
    assert 'SUCCESS' in stdout


# 定义 _test_other_signal_before_sigint_impl 函数，接受 backend, target_name, kwargs 三个参数
def _test_other_signal_before_sigint_impl(backend, target_name, kwargs):
    # 导入必要的模块：signal, matplotlib.pyplot as plt
    import signal
    import matplotlib.pyplot as plt

    # 设置 matplotlib 使用的后端
    plt.switch_backend(backend)

    # 获取 plt 对象中的目标函数，如 'show' 或 'pause'
    target = getattr(plt, target_name)

    # 创建一个新的图形对象
    fig = plt.figure()
    # 连接 'draw_event' 事件，打印 'DRAW' 并刷新输出
    fig.canvas.mpl_connect('draw_event', lambda *args: print('DRAW', flush=True))

    # 创建一个新的定时器，间隔为1秒，单次触发
    timer = fig.canvas.new_timer(interval=1)
    timer.single_shot = True
    # 添加回调函数，打印 'SIGUSR1' 并刷新输出
    timer.add_callback(print, 'SIGUSR1', flush=True)

    # 自定义信号处理函数，启动定时器
    def custom_signal_handler(signum, frame):
        timer.start()
    # 设置 SIGUSR1 信号的处理函数为 custom_signal_handler
    signal.signal(signal.SIGUSR1, custom_signal_handler)

    try:
        # 调用目标函数，传入 kwargs 参数
        target(**kwargs)
    except KeyboardInterrupt:
        # 捕获 KeyboardInterrupt 异常，打印 'SUCCESS' 并刷新输出
        print('SUCCESS', flush=True)


# 标记 test_other_signal_before_sigint 函数，当平台为 Windows 时跳过测试
@pytest.mark.skipif(sys.platform == 'win32',
                    reason='No other signal available to send on Windows')
# 参数化 test_other_signal_before_sigint 函数的 env 参数
@pytest.mark.parametrize("env", _get_testable_interactive_backends())
# 参数化 test_other_signal_before_sigint 函数的 target, kwargs 参数
@pytest.mark.parametrize("target, kwargs", [
    ('show', {'block': True}),
    ('pause', {'interval': 10})
])
# 定义 test_other_signal_before_sigint 函数，接受 env, target, kwargs, request 四个参数
def test_other_signal_before_sigint(env, target, kwargs, request):
    # 获取环境变量 MPLBACKEND 的值
    backend = env.get("MPLBACKEND")
    # 如果 backend 不是以 "qt" 或 "macosx" 开头，则跳过测试
    if not backend.startswith(("qt", "macosx")):
        pytest.skip("SIGINT currently only tested on qt and macosx")
    # 如果 backend 是 "macosx"，为当前测试节点添加一个 xfail 标记，原因是 macosx 后端存在问题
    if backend == "macosx":
        request.node.add_marker(pytest.mark.xfail(reason="macosx backend is buggy"))
    # 如果平台是 macOS 并且目标是 "show"，则为当前测试节点添加一个 xfail 标记，原因是 Qt 后端在 macOS 上存在问题
    if sys.platform == "darwin" and target == "show":
        request.node.add_marker(
            pytest.mark.xfail(reason="Qt backend is buggy on macOS"))
    # 构建一个子进程对象，用于执行 _test_other_signal_before_sigint_impl 函数的代码
    proc = _WaitForStringPopen(
        [sys.executable, "-c",
         inspect.getsource(_test_other_signal_before_sigint_impl) +
         "\n_test_other_signal_before_sigint_impl("
            f"{backend!r}, {target!r}, {kwargs!r})"])
    try:
        # 等待子进程输出 'DRAW'
        proc.wait_for('DRAW')
        # 发送 SIGUSR1 信号给子进程
        os.kill(proc.pid, signal.SIGUSR1)
        # 等待子进程输出 'SIGUSR1'
        proc.wait_for('SIGUSR1')
        # 发送 SIGINT 信号给子进程
        os.kill(proc.pid, signal.SIGINT)
        # 与子进程通信，获取标准输出和错误输出，设置超时时间为 _test_timeout
        stdout, _ = proc.communicate(timeout=_test_timeout)
    except Exception:
        # 发生异常时终止子进程
        proc.kill()
        # 再次获取子进程的输出
        stdout, _ = proc.communicate()
        # 将异常向上抛出
        raise
    # 打印子进程的标准输出
    print(stdout)
    # 断言标准输出中包含 'SUCCESS'
    assert 'SUCCESS' in stdout
```