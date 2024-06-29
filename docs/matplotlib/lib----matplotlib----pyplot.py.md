# `D:\src\scipysrc\matplotlib\lib\matplotlib\pyplot.py`

```
# fmt: off

from __future__ import annotations  # 导入未来版本的注释支持，用于类型注释中的类型引用

from contextlib import AbstractContextManager, ExitStack  # 导入上下文管理相关模块
from enum import Enum  # 导入枚举类型支持
import functools  # 导入函数工具模块，支持高阶函数
import importlib  # 导入模块动态加载支持
import inspect  # 导入检查对象信息的模块
import logging  # 导入日志记录模块
import sys  # 导入系统相关模块
import threading  # 导入多线程支持模块
import time  # 导入时间处理模块
from typing import TYPE_CHECKING, cast, overload  # 导入类型提示相关支持

from cycler import cycler  # 导入用于生成循环对象的模块，用于定制图表样式

import matplotlib  # 导入 matplotlib 库
import matplotlib.colorbar  # 导入 matplotlib 颜色条模块
import matplotlib.image  # 导入 matplotlib 图像模块
from matplotlib import _api  # 导入 matplotlib 内部 API 支持
from matplotlib import (  # 导入 matplotlib 常用功能模块，用于图形的创建和风格设置
    cm as cm, get_backend as get_backend, rcParams as rcParams, style as style)
from matplotlib import _pylab_helpers  # 导入 matplotlib 辅助工具模块
from matplotlib import interactive  # 导入 matplotlib 交互式绘图支持模块
from matplotlib import cbook  # 导入 matplotlib 内部工具库
from matplotlib import _docstring  # 导入 matplotlib 文档字符串支持模块
from matplotlib.backend_bases import (  # 导入 matplotlib 图形后端基础类
    FigureCanvasBase, FigureManagerBase, MouseButton)
from matplotlib.figure import Figure, FigureBase, figaspect  # 导入 matplotlib 图形和图形基类
from matplotlib.gridspec import GridSpec, SubplotSpec  # 导入 matplotlib 网格布局和子图规范支持
from matplotlib import rcsetup, rcParamsDefault, rcParamsOrig  # 导入 matplotlib 配置参数设置
from matplotlib.artist import Artist  # 导入 matplotlib 艺术家基类
from matplotlib.axes import Axes  # 导入 matplotlib 坐标轴类
from matplotlib.axes import Subplot  # 导入 matplotlib 子图类
from matplotlib.backends import BackendFilter, backend_registry  # 导入 matplotlib 后端过滤和后端注册
from matplotlib.projections import PolarAxes  # 导入 matplotlib 极坐标轴支持
from matplotlib import mlab  # 导入 matplotlib 数据处理模块，用于数据处理
from matplotlib.scale import get_scale_names  # 导入 matplotlib 比例尺名称获取模块

from matplotlib.cm import _colormaps  # 导入 matplotlib 颜色映射模块，用于颜色处理
from matplotlib.colors import _color_sequences, Colormap  # 导入 matplotlib 颜色序列和颜色映射类

import numpy as np  # 导入 NumPy 库

if TYPE_CHECKING:  # 如果是类型检查模式
    from collections.abc import Callable, Hashable, Iterable, Sequence  # 导入集合抽象基类相关支持
    import datetime  # 导入日期时间模块
    import pathlib  # 导入路径处理模块
    import os  # 导入操作系统模块
    from typing import Any, BinaryIO, Literal, TypeVar  # 导入类型变量相关支持
    from typing_extensions import ParamSpec  # 导入参数规范支持

    import PIL.Image  # 导入 PIL 图像处理模块
    from numpy.typing import ArrayLike  # 导入 NumPy 类型提示支持

    import matplotlib.axes  # 导入 matplotlib 坐标轴模块
    import matplotlib.artist  # 导入 matplotlib 艺术家模块
    import matplotlib.backend_bases  # 导入 matplotlib 图形后端基础模块
    # 导入需要的模块和类，以下按照模块分组进行导入
    from matplotlib.axis import Tick  # 导入 Tick 类，用于绘制坐标轴刻度
    from matplotlib.axes._base import _AxesBase  # 导入 _AxesBase 类，提供基本的坐标轴功能
    from matplotlib.backend_bases import RendererBase, Event  # 导入 RendererBase 和 Event 类，提供后端基础和事件处理功能
    from matplotlib.cm import ScalarMappable  # 导入 ScalarMappable 类，用于与颜色映射相关的操作
    from matplotlib.contour import ContourSet, QuadContourSet  # 导入 ContourSet 和 QuadContourSet 类，用于绘制等高线和四边形等高线
    from matplotlib.collections import (  # 导入集合模块中的多个类
        Collection,  # 通用的集合基类
        LineCollection,  # 线集合类，用于绘制多条线段
        PolyCollection,  # 多边形集合类，用于绘制多边形
        PathCollection,  # 路径集合类，用于绘制路径
        EventCollection,  # 事件集合类，用于绘制事件
        QuadMesh,  # 四边形网格类，用于绘制四边形网格
    )
    from matplotlib.colorbar import Colorbar  # 导入 Colorbar 类，用于绘制颜色条
    from matplotlib.container import (  # 导入容器模块中的多个类
        BarContainer,  # 柱状图容器类
        ErrorbarContainer,  # 误差棒容器类
        StemContainer,  # 柱形图容器类
    )
    from matplotlib.figure import SubFigure  # 导入 SubFigure 类，表示图形中的子图
    from matplotlib.legend import Legend  # 导入 Legend 类，用于绘制图例
    from matplotlib.mlab import GaussianKDE  # 导入 GaussianKDE 类，用于高斯核密度估计
    from matplotlib.image import AxesImage, FigureImage  # 导入 AxesImage 和 FigureImage 类，用于处理图像数据
    from matplotlib.patches import (  # 导入补丁模块中的多个类
        FancyArrow,  # 花式箭头类
        StepPatch,  # 步进补丁类
        Wedge,  # 扇形补丁类
    )
    from matplotlib.quiver import (  # 导入 quiver 模块中的多个类
        Barbs,  # 刺类，用于绘制矢量图中的刺
        Quiver,  # 箭类，用于绘制矢量图中的箭头
        QuiverKey,  # 箭头标记类，用于矢量图中箭头的标记
    )
    from matplotlib.scale import ScaleBase  # 导入 ScaleBase 类，提供刻度标尺的基类
    from matplotlib.transforms import Transform, Bbox  # 导入 Transform 和 Bbox 类，提供坐标变换和包围盒的功能
    from matplotlib.typing import (  # 导入类型提示模块中的类型
        ColorType,  # 颜色类型
        LineStyleType,  # 线条风格类型
        MarkerType,  # 标记类型
        HashableList,  # 可哈希列表类型
    )
    from matplotlib.widgets import SubplotTool  # 导入 SubplotTool 类，提供子图工具
    
    _P = ParamSpec('_P')  # 创建 ParamSpec 对象 _P，用于参数规范
    _R = TypeVar('_R')  # 创建类型变量 _R，表示任意类型
    _T = TypeVar('_T')  # 创建类型变量 _T，表示任意类型
# We may not need the following imports here:
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D, AxLine
from matplotlib.text import Text, Annotation
from matplotlib.patches import Arrow, Circle, Rectangle  # noqa: F401
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, Slider, Widget  # noqa: F401

# Importing specific components from a custom module for type hints
from .ticker import (  # noqa: F401
    TickHelper, Formatter, FixedFormatter, NullFormatter, FuncFormatter,
    FormatStrFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent,
    LogFormatterMathtext, Locator, IndexLocator, FixedLocator, NullLocator,
    LinearLocator, LogLocator, AutoLocator, MultipleLocator, MaxNLocator)

# Logger setup
_log = logging.getLogger(__name__)


# Explicit rename instead of import-as for typing's sake.
# Function overloads to copy docstrings and decorators
@overload
def _copy_docstring_and_deprecators(
    method: Any,
    func: Literal[None] = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


@overload
def _copy_docstring_and_deprecators(
    method: Any, func: Callable[_P, _R]) -> Callable[_P, _R]: ...


def _copy_docstring_and_deprecators(
    method: Any,
    func: Callable[_P, _R] | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[_P, _R]:
    """
    Copy docstring and decorators from *method* to *func* if *func* is provided,
    or return a partial function for copying docstring and decorators otherwise.
    """
    if func is None:
        # Return a partial function that wraps _copy_docstring_and_deprecators
        return cast('Callable[[Callable[_P, _R]], Callable[_P, _R]]',
                    functools.partial(_copy_docstring_and_deprecators, method))
    
    # Initialize decorators list with docstring copying
    decorators: list[Callable[[Callable[_P, _R]], Callable[_P, _R]]] = [
        _docstring.copy(method)
    ]
    
    # Check and propagate any decorators from *method* to *func*
    while hasattr(method, "__wrapped__"):
        potential_decorator = _api.deprecation.DECORATORS.get(method)
        if potential_decorator:
            decorators.append(potential_decorator)
        method = method.__wrapped__
    
    # Apply decorators in reverse order to *func*
    for decorator in decorators[::-1]:
        func = decorator(func)
    
    # Add pyplot note to *func* indicating it wraps another function
    _add_pyplot_note(func, method)
    return func


# List of functions exempted from adding a pyplot note
_NO_PYPLOT_NOTE = [
    'FigureBase._gci',  # wrapped_func is private
    '_AxesBase._sci',   # wrapped_func is private
    'Artist.findobj',   # not a standard pyplot wrapper because it does not operate
                        # on the current Figure / Axes. Explanation of relation would
                        # be more complex and is not too important.
]


def _add_pyplot_note(func, wrapped_func):
    """
    Add a note to the docstring of *func* indicating it is a pyplot wrapper.

    The note is added to the "Notes" section of the docstring. If that does
    not exist, a "Notes" section is created. In numpydoc, the "Notes"
    section is the third last possible section, only potentially followed by
    "References" and "Examples".
    """
    if not func.__doc__:
        return  # If func has no docstring, return without modification

    qualname = wrapped_func.__qualname__
    if qualname in _NO_PYPLOT_NOTE:
        return  # Skip adding a note if in the exemption list

    # Add note about pyplot wrapper status to func's docstring
    # Details are added to the "Notes" section or created if absent
    ...
    # 检查被包装的函数是否是一个方法（即是否包含"."）
    wrapped_func_is_method = True
    # 如果限定名称中不包含"."，则被包装的函数不是方法
    if "." not in qualname:
        wrapped_func_is_method = False
        # 构建函数链接，格式为"{模块}.{限定名称}"
        link = f"{wrapped_func.__module__}.{qualname}"
    # 如果限定名称以"Axes."开头，如"Axes.plot"
    elif qualname.startswith("Axes."):
        # 构建函数链接，格式为".axes.{限定名称}"
        link = ".axes." + qualname
    # 如果限定名称以"_AxesBase."开头，如"_AxesBase.set_xlabel"
    elif qualname.startswith("_AxesBase."):
        # 构建函数链接，格式为".axes.Axes{去除前缀的限定名称}"
        link = ".axes.Axes" + qualname[9:]
    # 如果限定名称以"Figure."开头，如"Figure.figimage"
    elif qualname.startswith("Figure."):
        # 构建函数链接，格式为".{限定名称}"
        link = "." + qualname
    # 如果限定名称以"FigureBase."开头，如"FigureBase.gca"
    elif qualname.startswith("FigureBase."):
        # 构建函数链接，格式为".Figure{去除前缀的限定名称}"
        link = ".Figure" + qualname[10:]
    # 如果限定名称以"FigureCanvasBase."开头，如"FigureBaseCanvas.mpl_connect"
    elif qualname.startswith("FigureCanvasBase."):
        # 构建函数链接，格式为".{限定名称}"
        link = "." + qualname
    else:
        # 如果限定名称不属于预期的任何类别，引发运行时错误
        raise RuntimeError(f"Wrapped method from unexpected class: {qualname}")

    # 根据是否是方法构建不同的消息
    if wrapped_func_is_method:
        message = f"This is the :ref:`pyplot wrapper <pyplot_interface>` for `{link}`."
    else:
        message = f"This is equivalent to `{link}`."

    # 查找正确的插入位置：
    # - 已经存在"Notes"部分，可以在其中插入
    # - 或者在下一个现有部分之前创建一个新的"Notes"部分。在numpydoc中，"Notes"部分是倒数第三个可能存在的部分，
    #   可能会跟在"References"和"Examples"之后。
    # - 或者在末尾附加一个新的"Notes"部分。
    doc = inspect.cleandoc(func.__doc__)
    if "\nNotes\n-----" in doc:
        before, after = doc.split("\nNotes\n-----", 1)
    elif (index := doc.find("\nReferences\n----------")) != -1:
        before, after = doc[:index], doc[index:]
    elif (index := doc.find("\nExamples\n--------")) != -1:
        before, after = doc[:index], doc[index:]
    else:
        # 如果没有找到"Notes"、"References"或"Examples"，则将内容追加到末尾
        before = doc + "\n"
        after = ""

    # 更新函数的文档字符串，添加包装信息到"Notes"部分
    func.__doc__ = f"{before}\nNotes\n-----\n\n.. note::\n\n    {message}\n{after}"
## Global ##

# Enum defining states controlled by {un}install_repl_displayhook().
# Possible states: NONE, PLAIN, IPYTHON
_ReplDisplayHook = Enum("_ReplDisplayHook", ["NONE", "PLAIN", "IPYTHON"])

# Global variable controlling the current state of the REPL display hook
_REPL_DISPLAYHOOK = _ReplDisplayHook.NONE


def _draw_all_if_interactive() -> None:
    """
    Helper function to conditionally draw all figures if matplotlib is in interactive mode.
    """
    if matplotlib.is_interactive():
        draw_all()


def install_repl_displayhook() -> None:
    """
    Connect to the display hook of the current shell.

    The display hook gets called when the read-evaluate-print-loop (REPL) of
    the shell has finished the execution of a command. We use this callback
    to be able to automatically update a figure in interactive mode.

    This works both with IPython and with vanilla python shells.
    """
    global _REPL_DISPLAYHOOK

    if _REPL_DISPLAYHOOK is _ReplDisplayHook.IPYTHON:
        return

    # Check if IPython module is available
    mod_ipython = sys.modules.get("IPython")
    if not mod_ipython:
        # If IPython module is not available, set display hook to PLAIN
        _REPL_DISPLAYHOOK = _ReplDisplayHook.PLAIN
        return

    ip = mod_ipython.get_ipython()
    if not ip:
        # If IPython instance is not available, set display hook to PLAIN
        _REPL_DISPLAYHOOK = _ReplDisplayHook.PLAIN
        return

    # Register post_execute event to call _draw_all_if_interactive
    ip.events.register("post_execute", _draw_all_if_interactive)
    _REPL_DISPLAYHOOK = _ReplDisplayHook.IPYTHON

    if mod_ipython.version_info[:2] < (8, 24):
        # Use backend2gui to get IPython GUI name for eventloop integration
        from IPython.core.pylabtools import backend2gui
        ipython_gui_name = backend2gui.get(get_backend())
    else:
        # Resolve backend using backend_registry for IPython >= 8.24
        _, ipython_gui_name = backend_registry.resolve_backend(get_backend())
    
    # Trigger IPython's eventloop integration, if available
    if ipython_gui_name:
        ip.enable_gui(ipython_gui_name)


def uninstall_repl_displayhook() -> None:
    """
    Disconnect from the display hook of the current shell.
    """
    global _REPL_DISPLAYHOOK

    if _REPL_DISPLAYHOOK is _ReplDisplayHook.IPYTHON:
        # Unregister post_execute event from IPython if installed
        from IPython import get_ipython
        ip = get_ipython()
        ip.events.unregister("post_execute", _draw_all_if_interactive)

    # Reset display hook state to NONE
    _REPL_DISPLAYHOOK = _ReplDisplayHook.NONE


# Alias for _pylab_helpers.Gcf.draw_all function
draw_all = _pylab_helpers.Gcf.draw_all


# Ensure this appears in the pyplot docs.
@_copy_docstring_and_deprecators(matplotlib.set_loglevel)
def set_loglevel(*args, **kwargs) -> None:
    """
    Redirect call to matplotlib.set_loglevel function.
    """
    return matplotlib.set_loglevel(*args, **kwargs)


@_copy_docstring_and_deprecators(Artist.findobj)
def findobj(
    o: Artist | None = None,
    match: Callable[[Artist], bool] | type[Artist] | None = None,
    include_self: bool = True
) -> list[Artist]:
    """
    Find objects within the specified Artist or the current figure.

    Args:
        o: Optional Artist instance or None (defaults to current figure).
        match: Callable or type for matching criteria.
        include_self: Boolean indicating whether to include the Artist itself.

    Returns:
        List of matching Artist objects.
    """
    if o is None:
        o = gcf()  # Get current figure if not provided
    return o.findobj(match, include_self=include_self)


_backend_mod: type[matplotlib.backend_bases._Backend] | None = None
# 确保选择了一个后端并返回它。
# 目前这是私有的，但将来可能会公开。

def _get_backend_mod() -> type[matplotlib.backend_bases._Backend]:
    # 如果 _backend_mod 为空
    if _backend_mod is None:
        # 使用 rcParams._get("backend") 来避免通过后备逻辑（这将重新导入 pyplot，
        # 然后如果需要解析自动标记，则调用 switch_backend）
        switch_backend(rcParams._get("backend"))

    # 将 _backend_mod 转换为指定类型
    return cast(type[matplotlib.backend_bases._Backend], _backend_mod)


def switch_backend(newbackend: str) -> None:
    """
    设置 pyplot 的后端。

    只有在没有启动其他交互式后端的事件循环时才能切换到交互式后端。
    总是可以在非交互式后端之间切换。

    如果新后端与当前后端不同，则通过 ``plt.close('all')`` 关闭所有打开的图形。

    Parameters
    ----------
    newbackend : str
        要使用的后端的名称（不区分大小写）。
    """

    # 全局变量 _backend_mod
    global _backend_mod

    # 确保初始化已被提升，以便稍后分配给它
    import matplotlib.backends

    # 如果 newbackend 是 rcsetup._auto_backend_sentinel
    if newbackend is rcsetup._auto_backend_sentinel:
        # 获取当前正在运行的交互式框架
        current_framework = cbook._get_running_interactive_framework()

        # 如果当前框架存在并且可以找到与之对应的后端
        if (current_framework and
                (backend := backend_registry.backend_for_gui_framework(
                    current_framework))):
            candidates = [backend]
        else:
            candidates = []

        # 增加默认的后端候选项
        candidates += [
            "macosx", "qtagg", "gtk4agg", "gtk3agg", "tkagg", "wxagg"]

        # 不尝试使用基于 cairo 的后端，因为每个都有额外的依赖 (pycairo)，质量较差。
        for candidate in candidates:
            try:
                switch_backend(candidate)
            except ImportError:
                continue
            else:
                # 如果成功切换到候选后端，则更新 rcParamsOrig 的 'backend'
                rcParamsOrig['backend'] = candidate
                return
        else:
            # 切换到 Agg 应该总是成功；如果失败，让异常传播出去。
            switch_backend("agg")
            rcParamsOrig["backend"] = "agg"
            return

    # 必须在访问逻辑开关上进行转义
    old_backend = dict.__getitem__(rcParams, 'backend')

    # 加载新后端模块
    module = backend_registry.load_backend_module(newbackend)
    # 获取画布类
    canvas_class = module.FigureCanvas

    # 获取所需的交互式框架
    required_framework = canvas_class.required_interactive_framework
    # 如果指定了必需的框架，则检查当前正在运行的交互式框架
    current_framework = cbook._get_running_interactive_framework()
    
    # 如果当前框架存在，并且需要的框架也存在，并且它们不一致，则抛出 ImportError 异常
    if (current_framework and required_framework
            and current_framework != required_framework):
        raise ImportError(
            "Cannot load backend {!r} which requires the {!r} interactive "
            "framework, as {!r} is currently running".format(
                newbackend, required_framework, current_framework))

    # 从模块中获取 new_figure_manager() 和 show() 函数

    # 经典方式下，后端可以直接导出这些函数。这保证了向后兼容性。
    new_figure_manager = getattr(module, "new_figure_manager", None)
    show = getattr(module, "show", None)

    # 在这种经典的方法中，后端被实现为模块，但从 backend_bases._Backend 继承默认方法实现。
    # 这是通过创建一个从 backend_bases._Backend 继承并且其主体填充了模块全局变量的 "类" 实现的。
    class backend_mod(matplotlib.backend_bases._Backend):
        locals().update(vars(module))

    # 然而，定义 new_figure_manager 和 show 的较新方法是从 canvas 方法派生它们。
    # 在这种情况下，也要相应地更新 backend_mod；此外，禁用对 draw_if_interactive 的每个后端定制。
    if new_figure_manager is None:

        # 如果未定义 new_figure_manager，则定义一个给定图形的 new_figure_manager 函数
        def new_figure_manager_given_figure(num, figure):
            return canvas_class.new_manager(figure, num)

        # 定义一个新的 new_figure_manager 函数，接受 num 和其他参数，并返回一个新的 Figure 实例
        def new_figure_manager(num, *args, FigureClass=Figure, **kwargs):
            fig = FigureClass(*args, **kwargs)
            return new_figure_manager_given_figure(num, fig)

        # 定义一个 draw_if_interactive 函数，用于绘制交互式图形
        def draw_if_interactive() -> None:
            if matplotlib.is_interactive():
                manager = _pylab_helpers.Gcf.get_active()
                if manager:
                    manager.canvas.draw_idle()

        # 将新定义的函数赋给 backend_mod 对象的相应属性
        backend_mod.new_figure_manager_given_figure = (
            new_figure_manager_given_figure)
        backend_mod.new_figure_manager = (
            new_figure_manager)
        backend_mod.draw_if_interactive = (
            draw_if_interactive)

    # 如果 manager 显式地覆盖了 pyplot_show，则使用它，即使全局 show 已经存在，因为后者可能是为了向后兼容性而存在的。
    manager_class = getattr(canvas_class, "manager_class", None)
    # 由于 pyplot_show 是一个类方法，所以不能直接比较 manager_class.pyplot_show 和 FMB.pyplot_show，
    # 因为上述构造是绑定的类方法，总是不同（绑定到不同的类）。我们还必须使用 getattr_static 而不是 vars，
    # 因为 manager_class 可能没有 __dict__。
    manager_pyplot_show = inspect.getattr_static(manager_class, "pyplot_show", None)
    # 获取 FigureManagerBase 类中名为 "pyplot_show" 的静态属性（方法或函数），若不存在则为 None
    base_pyplot_show = inspect.getattr_static(FigureManagerBase, "pyplot_show", None)
    
    # 如果 show 为 None，或者 manager_pyplot_show 不为 None 且与 base_pyplot_show 不同
    if (show is None
            or (manager_pyplot_show is not None
                and manager_pyplot_show != base_pyplot_show)):
        # 如果 manager_pyplot_show 为假值（False、None、空字符串等）
        if not manager_pyplot_show:
            # 抛出 ValueError 异常，指示后端 newbackend 既未定义 FigureCanvas.manager_class，也未定义顶级 show 函数
            raise ValueError(
                f"Backend {newbackend} defines neither FigureCanvas.manager_class nor "
                f"a toplevel show function")
        
        # 将 cast('Any', manager_class).pyplot_show 赋值给 _pyplot_show，类型标注忽略错误
        _pyplot_show = cast('Any', manager_class).pyplot_show
        # 将 _pyplot_show 赋值给 backend_mod.show，类型标注忽略方法分配错误
        backend_mod.show = _pyplot_show  # type: ignore[method-assign]

    # 记录调试信息，指示加载的后端名称及版本号
    _log.debug("Loaded backend %s version %s.",
               newbackend, backend_mod.backend_version)

    # 如果 newbackend 是 "ipympl" 或 "widget"
    if newbackend in ("ipympl", "widget"):
        # ipympl < 0.9.4 期望 rcParams["backend"] 是完全限定的后端名称，如 "module://ipympl.backend_nbagg" 而不是 "ipympl" 或 "widget"
        import importlib.metadata as im
        from matplotlib import _parse_to_version_info  # type: ignore[attr-defined]
        
        try:
            # 获取 ipympl 模块的版本号
            module_version = im.version("ipympl")
            # 如果版本号小于 (0, 9, 4)
            if _parse_to_version_info(module_version) < (0, 9, 4):
                # 将 newbackend 设置为 "module://ipympl.backend_nbagg"
                newbackend = "module://ipympl.backend_nbagg"
        except im.PackageNotFoundError:
            pass

    # 将 newbackend 赋值给 rcParams['backend'] 和 rcParamsDefault['backend']
    rcParams['backend'] = rcParamsDefault['backend'] = newbackend
    
    # 将 backend_mod 赋值给 _backend_mod
    _backend_mod = backend_mod
    
    # 遍历函数名列表 ["new_figure_manager", "draw_if_interactive", "show"]
    for func_name in ["new_figure_manager", "draw_if_interactive", "show"]:
        # 将 backend_mod 中对应函数的签名赋值给 globals() 中同名函数的 __signature__ 属性
        globals()[func_name].__signature__ = inspect.signature(
            getattr(backend_mod, func_name))

    # 为了兼容性需求，需要保持对后端的全局引用
    # 参见 https://github.com/matplotlib/matplotlib/issues/6092
    matplotlib.backends.backend = newbackend  # type: ignore[attr-defined]

    # 如果 old_backend 和 newbackend 不相等
    if not cbook._str_equal(old_backend, newbackend):
        # 如果存在打开的图形窗口
        if get_fignums():
            # 发出警告，指示自动关闭在切换后端时不再支持，建议首先显式调用 plt.close('all') 关闭所有图形窗口
            _api.warn_deprecated("3.8", message=(
                "Auto-close()ing of figures upon backend switching is deprecated since "
                "%(since)s and will be removed %(removal)s.  To suppress this warning, "
                "explicitly call plt.close('all') first."))
        # 关闭所有图形窗口
        close("all")

    # 确保 REPL 显示钩子已安装，以便在需要时成为交互式显示的一部分
    install_repl_displayhook()
# 如果 GUI 在主线程之外启动，则发出警告
def _warn_if_gui_out_of_main_thread() -> None:
    warn = False
    # 获取当前后端的画布类
    canvas_class = cast(type[FigureCanvasBase], _get_backend_mod().FigureCanvas)
    # 检查当前后端是否需要交互框架
    if canvas_class.required_interactive_framework:
        # 如果 threading 模块有 get_native_id 方法
        if hasattr(threading, 'get_native_id'):
            # 比较本地线程 ID，因为即使 Python 级别的 Thread 对象匹配，
            # 底层的 OS 线程（真正重要的部分）在具有绿色线程的 Python 实现中可能是不同的。
            if threading.get_native_id() != threading.main_thread().native_id:
                warn = True
        else:
            # 对于没有本地 ID 的情况（主要是对于 PyPy），回退到 Python 级别的 Thread。
            if threading.current_thread() is not threading.main_thread():
                warn = True
    # 如果需要发出警告，则调用 _api.warn_external 方法
    if warn:
        _api.warn_external(
            "Starting a Matplotlib GUI outside of the main thread will likely "
            "fail.")


# 根据后端加载情况重写该函数的签名
def new_figure_manager(*args, **kwargs):
    """创建一个新的图形管理器实例。"""
    # 检查并警告如果 GUI 不在主线程中
    _warn_if_gui_out_of_main_thread()
    # 调用后端模块的 new_figure_manager 方法
    return _get_backend_mod().new_figure_manager(*args, **kwargs)


# 根据后端加载情况重写该函数的签名
def draw_if_interactive(*args, **kwargs):
    """
    如果处于交互模式，则重新绘制当前图形。

    .. warning::

        终端用户通常不需要调用此函数，因为交互模式会自动处理这一过程。
    """
    # 调用后端模块的 draw_if_interactive 方法
    return _get_backend_mod().draw_if_interactive(*args, **kwargs)


# 根据后端加载情况重写该函数的签名
def show(*args, **kwargs) -> None:
    """
    显示所有打开的图形。

    Parameters
    ----------
    block : bool, optional
        是否在所有图形关闭之前等待。

        如果为 `True`，则阻塞并运行 GUI 主循环，直到所有图形窗口关闭。

        如果为 `False`，则确保所有图形窗口显示，并立即返回。在此情况下，
        您需要确保事件循环正在运行以确保图形响应。

        在非交互模式下默认为 True，在交互模式下默认为 False
        （参见 `.pyplot.isinteractive`）。

    See Also
    --------
    ion : 启用交互模式，在每个绘图命令后显示/更新图形，因此不需要调用 ``show()``。
    ioff : 禁用交互模式。
    savefig : 将图形保存为图像文件而不是在屏幕上显示。

    Notes
    -----
    **同时保存图形文件并显示窗口**

    如果需要图像文件以及用户界面窗口，应在 `.pyplot.show` 之前使用 `.pyplot.savefig`。
    在（阻塞的）``show()`` 结束时，图形将关闭并从 pyplot 注销。

    """
    # `_warn_if_gui_out_of_main_thread()`：调用一个私有函数 `_warn_if_gui_out_of_main_thread()`，用于检测 GUI 是否在主线程之外运行，并可能发出警告。
    
    # 返回调用 `_get_backend_mod().show(*args, **kwargs)` 的结果。这个函数根据当前 matplotlib 后端来展示图形。
# 检查当前 matplotlib 是否处于交互模式，并返回布尔值结果
def isinteractive() -> bool:
    """
    Return whether plots are updated after every plotting command.

    The interactive mode is mainly useful if you build plots from the command
    line and want to see the effect of each command while you are building the
    figure.

    In interactive mode:

    - newly created figures will be shown immediately;
    - figures will automatically redraw on change;
    - `.pyplot.show` will not block by default.

    In non-interactive mode:

    - newly created figures and changes to figures will not be reflected until
      explicitly asked to be;
    - `.pyplot.show` will block by default.

    See Also
    --------
    ion : Enable interactive mode.
    ioff : Disable interactive mode.
    show : Show all figures (and maybe block).
    pause : Show all figures, and block for a time.
    """
    return matplotlib.is_interactive()


# Note: The return type of ioff being AbstractContextManager
# instead of ExitStack is deliberate.
# See https://github.com/matplotlib/matplotlib/issues/27659
# and https://github.com/matplotlib/matplotlib/pull/27667 for more info.
# 禁用交互模式
def ioff() -> AbstractContextManager:
    """
    Disable interactive mode.

    See `.pyplot.isinteractive` for more details.

    See Also
    --------
    ion : Enable interactive mode.
    isinteractive : Whether interactive mode is enabled.
    show : Show all figures (and maybe block).
    pause : Show all figures, and block for a time.

    Notes
    -----
    For a temporary change, this can be used as a context manager::

        # if interactive mode is on
        # then figures will be shown on creation
        plt.ion()
        # This figure will be shown immediately
        fig = plt.figure()

        with plt.ioff():
            # interactive mode will be off
            # figures will not automatically be shown
            fig2 = plt.figure()
            # ...

    To enable optional usage as a context manager, this function returns a
    context manager object, which is not intended to be stored or
    accessed by the user.
    """
    stack = ExitStack()
    stack.callback(ion if isinteractive() else ioff)
    matplotlib.interactive(False)
    uninstall_repl_displayhook()
    return stack


# Note: The return type of ion being AbstractContextManager
# instead of ExitStack is deliberate.
# See https://github.com/matplotlib/matplotlib/issues/27659
# and https://github.com/matplotlib/matplotlib/pull/27667 for more info.
# 启用交互模式
def ion() -> AbstractContextManager:
    """
    Enable interactive mode.

    See `.pyplot.isinteractive` for more details.

    See Also
    --------
    ioff : Disable interactive mode.
    isinteractive : Whether interactive mode is enabled.
    show : Show all figures (and maybe block).
    pause : Show all figures, and block for a time.

    Notes
    -----
    """
    # 创建并返回一个上下文管理器，用于启用交互模式
    stack = ExitStack()
    stack.callback(ion if isinteractive() else ioff)
    matplotlib.interactive(True)
    install_repl_displayhook()
    return stack
    # 创建一个 ExitStack 对象，用于管理上下文，确保退出时调用相应的回调函数
    stack = ExitStack()
    # 根据当前的交互模式设置回调函数，如果交互模式开启则使用 ion，否则使用 ioff
    stack.callback(ion if isinteractive() else ioff)
    # 设置 matplotlib 的交互模式为 True
    matplotlib.interactive(True)
    # 安装 REPL 显示钩子，用于在 REPL 中显示图形
    install_repl_displayhook()
    # 返回 ExitStack 对象，以便可以使用它作为上下文管理器
    return stack
# 运行 GUI 事件循环，如果有活动的图形，则在暂停之前更新并显示它
def pause(interval: float) -> None:
    # 获取当前活动的图形管理器
    manager = _pylab_helpers.Gcf.get_active()
    if manager is not None:
        # 获取画布对象
        canvas = manager.canvas
        # 如果画布上的图形已经过时，则强制重绘
        if canvas.figure.stale:
            canvas.draw_idle()
        # 显示所有图形，不阻塞当前线程
        show(block=False)
        # 启动画布上的事件循环，暂停 interval 秒
        canvas.start_event_loop(interval)
    else:
        # 如果没有活动的图形，则线程休眠 interval 秒
        time.sleep(interval)


# 将指定图形组的所有属性设置为给定值
@_copy_docstring_and_deprecators(matplotlib.rc)
def rc(group: str, **kwargs) -> None:
    matplotlib.rc(group, **kwargs)


# 返回一个上下文管理器，该管理器可用于设置临时的 rc 参数
@_copy_docstring_and_deprecators(matplotlib.rc_context)
def rc_context(
    rc: dict[str, Any] | None = None,
    fname: str | pathlib.Path | os.PathLike | None = None,
) -> AbstractContextManager[None]:
    return matplotlib.rc_context(rc, fname)


# 恢复默认的 rc 参数设置，并在交互模式下绘制所有图形
@_copy_docstring_and_deprecators(matplotlib.rcdefaults)
def rcdefaults() -> None:
    matplotlib.rcdefaults()
    if matplotlib.is_interactive():
        draw_all()


# 获取指定艺术对象的属性
@_copy_docstring_and_deprecators(matplotlib.artist.getp)
def getp(obj, *args, **kwargs):
    return matplotlib.artist.getp(obj, *args, **kwargs)


# 获取指定艺术对象的属性值
@_copy_docstring_and_deprecators(matplotlib.artist.get)
def get(obj, *args, **kwargs):
    return matplotlib.artist.get(obj, *args, **kwargs)


# 设置指定艺术对象的属性
@_copy_docstring_and_deprecators(matplotlib.artist.setp)
def setp(obj, *args, **kwargs):
    return matplotlib.artist.setp(obj, *args, **kwargs)


# 激活 xkcd 风格的绘图模式
def xkcd(
    scale: float = 1, length: float = 100, randomness: float = 2
) -> ExitStack:
    """
    Turn on `xkcd <https://xkcd.com/>`_ sketch-style drawing mode.

    This will only have an effect on things drawn after this function is called.

    For best results, install the `xkcd script <https://github.com/ipython/xkcd-font/>`_
    font; xkcd fonts are not packaged with Matplotlib.

    Parameters
    ----------
    scale : float, optional
        The amplitude of the wiggle perpendicular to the source line.
    length : float, optional
        The length of the wiggle along the line.
    randomness : float, optional
        The scale factor by which the length is shrunken or expanded.

    Notes
    -----
    This function works by a number of rcParams, so it will probably
    override others you have set before.

    If you want the effects of this function to be temporary, it can
    """
    # 返回一个上下文管理器，用于控制 xkcd 风格的绘图
    return matplotlib.xkcd(scale=scale, length=length, randomness=randomness)
    """
    # 这段代码用于配置 matplotlib 的 xkcd 模式，可以用作上下文管理器
    # 或非上下文管理器使用。

    # 如果 text.usetex 设置为 True，则无法使用 xkcd 模式
    if rcParams['text.usetex']:
        raise RuntimeError(
            "xkcd mode is not compatible with text.usetex = True")

    # 创建一个 ExitStack 对象，用于管理回调函数的堆栈
    stack = ExitStack()

    # 将当前的 rcParams 拷贝更新到回调函数中，保证退出时恢复原始状态
    stack.callback(dict.update, rcParams, rcParams.copy())  # type: ignore[arg-type]

    # 导入 patheffects 模块，设置 xkcd 模式下的绘图参数
    from matplotlib import patheffects
    rcParams.update({
        'font.family': ['xkcd', 'xkcd Script', 'Comic Neue', 'Comic Sans MS'],
        'font.size': 14.0,
        'path.sketch': (scale, length, randomness),
        'path.effects': [
            patheffects.withStroke(linewidth=4, foreground="w")],
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'figure.facecolor': 'white',
        'grid.linewidth': 0.0,
        'axes.grid': False,
        'axes.unicode_minus': False,
        'axes.edgecolor': 'black',
        'xtick.major.size': 8,
        'xtick.major.width': 3,
        'ytick.major.size': 8,
        'ytick.major.width': 3,
    })

    # 返回配置好的 stack 对象，用于管理 xkcd 模式的退出操作
    return stack
    ```
# 创建或激活一个新的图形对象，并返回该对象。如果指定的编号已存在，则激活该图形对象。
def figure(
    # 图形的唯一标识符，可以是整数、字符串、Figure对象或SubFigure对象，可选参数
    num: int | str | Figure | SubFigure | None = None,
    # 图形的尺寸，单位为英寸，可选参数，默认使用rc配置中的figure.figsize
    figsize: tuple[float, float] | None = None,
    # 图形的分辨率，单位为每英寸点数（DPI），可选参数，默认使用rc配置中的figure.dpi
    dpi: float | None = None,
    *,
    # 图形的背景颜色，可选参数，默认使用rc配置中的figure.facecolor
    facecolor: ColorType | None = None,
    # 图形边框的颜色，可选参数，默认使用rc配置中的figure.edgecolor
    edgecolor: ColorType | None = None,
    # 控制是否绘制图形的边框，可选参数，默认为True
    frameon: bool = True,
    # 图形对象的类别，必须是Figure的子类，默认为Figure
    FigureClass: type[Figure] = Figure,
    # 如果为True且图形已存在，则清空图形内容，可选参数，默认为False
    clear: bool = False,
    **kwargs
) -> Figure:
    """
    创建一个新的图形对象，或激活一个已存在的图形对象。

    参数
    ----------
    num : int 或 str 或 `.Figure` 或 `.SubFigure`, 可选
        图形的唯一标识符。

        如果具有该标识符的图形已存在，则激活该图形并返回。整数引用`Figure.number`属性，字符串引用图形标签。

        如果不存在具有该标识符的图形或未提供*num*，则创建一个新的图形，激活并返回。如果*num*为整数，则用于`Figure.number`属性，
        否则使用自动生成的整数值（从1开始，每创建一个新图形递增）。如果*num*为字符串，则设置图形标签和窗口标题为该值。如果*num*为
        `SubFigure`，则激活其父`Figure`。

    figsize : (float, float), 默认值：:rc:`figure.figsize`
        图形的宽度和高度，单位为英寸。

    dpi : float, 默认值：:rc:`figure.dpi`
        图形的分辨率，单位为每英寸点数（DPI）。

    facecolor : :mpltype:`color`, 默认值：:rc:`figure.facecolor`
        背景颜色。

    edgecolor : :mpltype:`color`, 默认值：:rc:`figure.edgecolor`
        边框颜色。

    frameon : bool, 默认值：True
        如果为False，则不绘制图形边框。

    FigureClass : `~matplotlib.figure.Figure`的子类
        如果设置，则创建此子类的实例，而不是普通的`.Figure`。

    clear : bool, 默认值：False
        如果为True且图形已存在，则清空图形内容。

    layout : {'constrained', 'compressed', 'tight', 'none', `.LayoutEngine`, None}, \
    """
    # 实现函数体的具体逻辑，创建或激活图形对象，并返回该对象
# 默认情况下为 None，控制图表元素的布局机制，以避免重叠的坐标轴装饰（标签、刻度等）。
# 注意，布局管理器可能会显著减慢图表显示速度。
#
# - 'constrained': 通过调整坐标轴大小以避免重叠的坐标轴装饰来进行布局。
#   能处理复杂的图表布局和色条，因此推荐使用。详见 :ref:`constrainedlayout_guide` 中的示例。
#
# - 'compressed': 使用与 'constrained' 相同的算法，但移除固定纵横比坐标轴之间的额外空间。
#   最适合简单的坐标轴网格布局。
#
# - 'tight': 使用紧凑布局机制。这是一个相对简单的算法，调整子图参数以避免装饰重叠。
#   详见 `.Figure.set_tight_layout` 获取更多细节。
#
# - 'none': 不使用布局引擎。
#
# - `.LayoutEngine` 的实例。内置的布局类有 `.ConstrainedLayoutEngine` 和 `.TightLayoutEngine`，
#   可以通过 'constrained' 和 'tight' 更轻松地访问。传递一个实例允许第三方提供自己的布局引擎。
#
# 如果未提供，则回退到使用参数 *tight_layout* 和 *constrained_layout*，包括它们的配置默认值
# :rc:`figure.autolayout` 和 :rc:`figure.constrained_layout.use`。
#
# **kwargs
# 附加的关键字参数传递给 `.Figure` 构造函数。
#
# Returns
# -------
# `~matplotlib.figure.Figure`
# 返回新创建的图形对象 `~matplotlib.figure.Figure`。
#
# Notes
# -----
# 新创建的图形对象通过当前后端提供的 `~.FigureCanvasBase.new_manager` 方法或 `new_figure_manager` 函数传递，
# 这些方法在图形上安装画布和管理器。
#
# 一旦完成上述步骤，将按顺序调用 :rc:`figure.hooks`，这些钩子允许对图形进行任意定制（例如附加回调函数）
# 或相关元素的定制（例如修改工具栏）。详见 :doc:`/gallery/user_interfaces/mplcvd` 中有关工具栏定制的示例。
#
# 如果您正在创建许多图形，请确保显式调用 `.pyplot.close` 关闭不再使用的图形，
# 因为这样可以使 pyplot 正确清理内存。
#
# `~matplotlib.rcParams` 定义了默认值，可以在 matplotlibrc 文件中进行修改。
"""
allnums = get_fignums()
    # 如果 num 是 FigureBase 的实例
    if isinstance(num, FigureBase):
        # 类型被进一步缩小为 `Figure | SubFigure`，通过输入和 isinstance 的结合
        # 检查图形的管理器是否为 None，如果是则抛出数值错误
        if num.canvas.manager is None:
            raise ValueError("The passed figure is not managed by pyplot")
        # 如果指定了任何 figsize、dpi、facecolor、edgecolor、not frameon 或 kwargs，
        # 并且 num.canvas.manager.num 在 allnums 中，则发出警告
        elif any([figsize, dpi, facecolor, edgecolor, not frameon, kwargs]) and num.canvas.manager.num in allnums:
            _api.warn_external(
                "Ignoring specified arguments in this call "
                f"because figure with num: {num.canvas.manager.num} already exists")
        # 设置活动的图形为 num.canvas.manager
        _pylab_helpers.Gcf.set_active(num.canvas.manager)
        # 返回 num 对应的图形对象
        return num.figure

    # 计算下一个可用的 num
    next_num = max(allnums) + 1 if allnums else 1
    # 初始化图形标签为空字符串
    fig_label = ''
    
    # 如果 num 为 None，则将其设置为下一个可用的 num
    if num is None:
        num = next_num
    else:
        # 如果指定了任何 figsize、dpi、facecolor、edgecolor、not frameon 或 kwargs，
        # 并且 num 在 allnums 中，则发出警告
        if any([figsize, dpi, facecolor, edgecolor, not frameon, kwargs]) and num in allnums:
            _api.warn_external(
                "Ignoring specified arguments in this call "
                f"because figure with num: {num} already exists")
        
        # 如果 num 是字符串类型
        if isinstance(num, str):
            # 将 fig_label 设置为 num
            fig_label = num
            # 获取所有图形标签
            all_labels = get_figlabels()
            # 如果 fig_label 不在 all_labels 中
            if fig_label not in all_labels:
                # 如果 fig_label 是 'all'，则发出警告
                if fig_label == 'all':
                    _api.warn_external("close('all') closes all existing figures.")
                # 将 num 设置为下一个可用的 num
                num = next_num
            else:
                # 获取 fig_label 在 all_labels 中的索引
                inum = all_labels.index(fig_label)
                # 将 num 设置为 allnums 中对应的值
                num = allnums[inum]
        else:
            # 将 num 转换为整数，对 num 参数进行基本验证
            num = int(num)  # crude validation of num argument

    # Type of "num" has narrowed to int, but mypy can't quite see it
    # 获取 num 对应的图形管理器
    manager = _pylab_helpers.Gcf.get_fig_manager(num)  # type: ignore[arg-type]
    # 如果未提供 manager 参数，则获取配置中的最大打开警告次数
    if manager is None:
        max_open_warning = rcParams['figure.max_open_warning']
        # 如果当前已打开的图形数量等于或超过最大警告次数，则发出运行时警告
        if len(allnums) == max_open_warning >= 1:
            _api.warn_external(
                # 提示用户已打开的图形数量超过设定的最大警告次数
                f"More than {max_open_warning} figures have been opened. "
                # 提醒用户通过 pyplot 接口创建的图形会一直保留，直到显式关闭，可能会占用过多内存
                f"Figures created through the pyplot interface "
                f"(`matplotlib.pyplot.figure`) are retained until explicitly "
                f"closed and may consume too much memory. (To control this "
                # 提示用户可以通过修改 rcParam `figure.max_open_warning` 来控制此警告
                f"warning, see the rcParam `figure.max_open_warning`). "
                # 建议用户考虑使用 `matplotlib.pyplot.close()` 来关闭不再需要的图形
                f"Consider using `matplotlib.pyplot.close()`.",
                RuntimeWarning)

        # 使用给定参数创建新的图形管理器对象
        manager = new_figure_manager(
            num, figsize=figsize, dpi=dpi,
            facecolor=facecolor, edgecolor=edgecolor, frameon=frameon,
            FigureClass=FigureClass, **kwargs)
        # 从图形管理器获取图形对象
        fig = manager.canvas.figure
        # 如果提供了图形标签，则设置图形的标签
        if fig_label:
            fig.set_label(fig_label)

        # 遍历配置中注册的所有 figure hooks，并调用对应的处理函数
        for hookspecs in rcParams["figure.hooks"]:
            module_name, dotted_name = hookspecs.split(":")
            obj: Any = importlib.import_module(module_name)
            for part in dotted_name.split("."):
                obj = getattr(obj, part)
            obj(fig)

        # 设置新创建的图形管理器为活动的图形管理器
        _pylab_helpers.Gcf._set_new_active_manager(manager)

        # 如果是交互式模式，确保后端支持在绘图命令中调用此函数以显示图形
        draw_if_interactive()

        # 如果当前的 REPL 显示钩子为普通模式，则设置图形的过时回调函数为自动交互绘制
        if _REPL_DISPLAYHOOK is _ReplDisplayHook.PLAIN:
            fig.stale_callback = _auto_draw_if_interactive

    # 如果 clear 参数为 True，则清除图形对象
    if clear:
        manager.canvas.figure.clear()

    # 返回图形管理器的图形对象
    return manager.canvas.figure
# 获取当前图形对象
def _auto_draw_if_interactive(fig, val):
    """
    An internal helper function for making sure that auto-redrawing
    works as intended in the plain python repl.

    Parameters
    ----------
    fig : Figure
        A figure object which is assumed to be associated with a canvas
    """
    # 如果启用了交互模式，并且画布没有在保存或空闲绘制状态下
    if (val and matplotlib.is_interactive()
            and not fig.canvas.is_saving()
            and not fig.canvas._is_idle_drawing):
        # 一些图形元素在绘制过程中可能会标记为过时（例如，绘制时计算轴位置和刻度标签）
        # 但这不应该触发重绘，因为当前的重绘已经将它们考虑在内
        with fig.canvas._idle_draw_cntx():
            fig.canvas.draw_idle()


def gcf() -> Figure:
    """
    Get the current figure.

    If there is currently no figure on the pyplot figure stack, a new one is
    created using `~.pyplot.figure()`.  (To test whether there is currently a
    figure on the pyplot figure stack, check whether `~.pyplot.get_fignums()`
    is empty.)
    """
    # 获取当前活跃的图形管理器
    manager = _pylab_helpers.Gcf.get_active()
    if manager is not None:
        return manager.canvas.figure
    else:
        return figure()


def fignum_exists(num: int | str) -> bool:
    """
    Return whether the figure with the given id exists.

    Parameters
    ----------
    num : int or str
        A figure identifier.

    Returns
    -------
    bool
        Whether or not a figure with id *num* exists.
    """
    # 检查指定编号的图形是否存在
    return (
        _pylab_helpers.Gcf.has_fignum(num)
        if isinstance(num, int)
        else num in get_figlabels()
    )


def get_fignums() -> list[int]:
    """Return a list of existing figure numbers."""
    # 返回已存在的图形编号列表（按升序排序）
    return sorted(_pylab_helpers.Gcf.figs)


def get_figlabels() -> list[Any]:
    """Return a list of existing figure labels."""
    # 获取所有图形管理器，并按编号排序，返回每个图形的标签
    managers = _pylab_helpers.Gcf.get_all_fig_managers()
    managers.sort(key=lambda m: m.num)
    return [m.canvas.figure.get_label() for m in managers]


def get_current_fig_manager() -> FigureManagerBase | None:
    """
    Return the figure manager of the current figure.

    The figure manager is a container for the actual backend-depended window
    that displays the figure on screen.

    If no current figure exists, a new one is created, and its figure
    manager is returned.

    Returns
    -------
    `.FigureManagerBase` or backend-dependent subclass thereof
    """
    # 返回当前图形的图形管理器，如果不存在则创建新的图形并返回其管理器
    return gcf().canvas.manager


@_copy_docstring_and_deprecators(FigureCanvasBase.mpl_connect)
def connect(s: str, func: Callable[[Event], Any]) -> int:
    # 连接指定事件类型到回调函数
    return gcf().canvas.mpl_connect(s, func)


@_copy_docstring_and_deprecators(FigureCanvasBase.mpl_disconnect)
def disconnect(cid: int) -> None:
    # 断开指定的连接
    gcf().canvas.mpl_disconnect(cid)


def close(fig: None | int | str | Figure | Literal["all"] = None) -> None:
    """
    Close a figure window.

    Parameters
    ----------
    fig : None, int, str, Figure, or 'all', optional
        The figure to close. Default is the current figure.
    """
    fig : None or int or str or `.Figure`
        The figure to close. There are a number of ways to specify this:

        - *None*: the current figure
        - `.Figure`: the given `.Figure` instance
        - ``int``: a figure number
        - ``str``: a figure name
        - 'all': all figures

    """
    # 如果 fig 参数为 None，则关闭当前活动的图形窗口
    if fig is None:
        # 获取当前活动的图形管理器实例
        manager = _pylab_helpers.Gcf.get_active()
        # 如果没有活动的图形管理器实例，则直接返回
        if manager is None:
            return
        else:
            # 销毁当前活动的图形管理器实例
            _pylab_helpers.Gcf.destroy(manager)
    
    # 如果 fig 参数为 'all'，则关闭所有的图形窗口
    elif fig == 'all':
        # 销毁所有的图形管理器实例
        _pylab_helpers.Gcf.destroy_all()
    
    # 如果 fig 参数为整数，则按照给定的图形编号关闭对应的图形窗口
    elif isinstance(fig, int):
        # 销毁指定编号的图形管理器实例
        _pylab_helpers.Gcf.destroy(fig)
    
    # 如果 fig 参数具有 int 属性，则假设它是一个 UUID 类型，使用其整数表示来关闭对应的图形窗口
    elif hasattr(fig, 'int'):
        # 使用 fig.int 的整数表示来销毁对应的图形管理器实例
        _pylab_helpers.Gcf.destroy(fig.int)
    
    # 如果 fig 参数为字符串，则假设它是一个图形名称，根据名称关闭对应的图形窗口
    elif isinstance(fig, str):
        # 获取所有图形的名称列表
        all_labels = get_figlabels()
        # 如果指定的图形名称在名称列表中，则找到其对应的图形编号并关闭对应的图形窗口
        if fig in all_labels:
            # 获取指定图形名称对应的图形编号
            num = get_fignums()[all_labels.index(fig)]
            # 销毁指定编号的图形管理器实例
            _pylab_helpers.Gcf.destroy(num)
    
    # 如果 fig 参数为 Figure 类的实例，则关闭该 Figure 对象对应的图形窗口
    elif isinstance(fig, Figure):
        # 销毁指定的 Figure 对象对应的图形管理器实例
        _pylab_helpers.Gcf.destroy_fig(fig)
    
    # 如果 fig 参数类型不符合预期，则抛出类型错误异常
    else:
        raise TypeError("close() argument must be a Figure, an int, a string, "
                        "or None, not %s" % type(fig))
def clf() -> None:
    """清除当前图形。"""
    gcf().clear()


def draw() -> None:
    """
    重新绘制当前图形。

    这用于更新已修改但未自动重新绘制的图形。如果处于交互模式（通过 `.ion()` 设置），这通常是很少需要的，
    但可能有修改图形状态而不标记为“过时”的方式。请将这些情况报告为错误。

    这等效于调用 ``fig.canvas.draw_idle()``，其中 ``fig`` 是当前图形。

    参见
    --------
    .FigureCanvasBase.draw_idle
    .FigureCanvasBase.draw
    """
    gcf().canvas.draw_idle()


@_copy_docstring_and_deprecators(Figure.savefig)
def savefig(*args, **kwargs) -> None:
    fig = gcf()
    # savefig 默认实现没有返回值，所以 mypy 不满意
    # 这里应该是因为子类可以返回值？
    res = fig.savefig(*args, **kwargs)  # type: ignore[func-returns-value]
    fig.canvas.draw_idle()  # 如果 'transparent=True'，需要这个来重置颜色。
    return res


## 将内容放入图形中 ##


def figlegend(*args, **kwargs) -> Legend:
    return gcf().legend(*args, **kwargs)
if Figure.legend.__doc__:
    figlegend.__doc__ = Figure.legend.__doc__ \
        .replace(" legend(", " figlegend(") \
        .replace("fig.legend(", "plt.figlegend(") \
        .replace("ax.plot(", "plt.plot(")


## 坐标轴 ##

@_docstring.dedent_interpd
def axes(
    arg: None | tuple[float, float, float, float] = None,
    **kwargs
) -> matplotlib.axes.Axes:
    """
    在当前图形中添加一个坐标轴并将其设置为当前坐标轴。

    调用签名::

        plt.axes()
        plt.axes(rect, projection=None, polar=False, **kwargs)
        plt.axes(ax)

    参数
    ----------
    arg : None 或 4 元组
        此函数的确切行为取决于类型：

        - *None*: 使用 ``subplot(**kwargs)`` 添加一个新的全窗口坐标轴。
        - 4 元组浮点数 *rect* = ``(left, bottom, width, height)``。
          使用 `~.Figure.add_axes` 在当前图形上添加尺寸为 *rect* 的新坐标轴，单位为标准化 (0, 1)。

    projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
'polar', 'rectilinear', str}, 可选
        `~.axes.Axes` 的投影类型。*str* 是自定义投影的名称，参见 `~matplotlib.projections`。默认值
        None 结果是 'rectilinear' 投影。

    polar : bool, 默认值：False
        如果为 True，则等同于 projection='polar'。

    sharex, sharey : `~matplotlib.axes.Axes`, 可选
        与 sharex 和/或 sharey 共享 x 或 y `~matplotlib.axis`。
        轴将具有与共享轴相同的限制、刻度和比例。

    label : str
        返回的坐标轴的标签。

    返回
    -------
    matplotlib.axes.Axes
    """
    pass  # 函数体暂时为空，实际功能由后续代码实现
    """
    获取当前图形对象，如果指定位置参数，则创建指定位置的轴对象；否则根据关键字参数创建子图轴对象。

    Parameters
    ----------
    arg : `~.axes.Axes`, optional
        如果为 None，则根据 kwargs 创建子图轴对象；否则，根据 arg 和 kwargs 创建轴对象。

    position : tuple, optional
        轴对象的位置参数，指定轴在图形中的位置。

    **kwargs
        返回的轴对象的关键字参数。如果使用直角坐标系，关键字参数可以在 `~.axes.Axes` 类中找到。
        如果使用极坐标系，可能还会有其他关键字参数，请参阅实际的轴对象类文档。

        %(Axes:kwdoc)s  # 此处是一个占位符，用于动态生成关键字参数的文档。

    Returns
    -------
    `~.axes.Axes`, or a subclass of `~.axes.Axes`
        返回的轴对象取决于所使用的投影。如果使用直角投影，则为 `~.axes.Axes` 类。
        如果使用极坐标投影，则为 `.projections.polar.PolarAxes` 类。

    See Also
    --------
    .Figure.add_axes
    .pyplot.subplot
    .Figure.add_subplot
    .Figure.subplots
    .pyplot.subplots

    Examples
    --------
    ::

        # 创建一个新的全窗口轴对象
        plt.axes()

        # 创建一个指定尺寸和灰色背景的新轴对象
        plt.axes((left, bottom, width, height), facecolor='grey')
    """
    fig = gcf()  # 获取当前图形对象
    pos = kwargs.pop('position', None)  # 弹出关键字参数中的 'position'，并赋值给 pos 变量
    if arg is None:
        if pos is None:
            return fig.add_subplot(**kwargs)  # 如果 arg 为 None，并且未指定位置，根据 kwargs 创建子图轴对象
        else:
            return fig.add_axes(pos, **kwargs)  # 如果 arg 为 None，但指定了位置 pos，使用 pos 和 kwargs 创建轴对象
    else:
        return fig.add_axes(arg, **kwargs)  # 如果 arg 不为 None，则使用 arg 和 kwargs 创建轴对象
# 从当前图中删除指定的 `~.axes.Axes` 对象（默认为当前 Axes）。
def delaxes(ax: matplotlib.axes.Axes | None = None) -> None:
    # 如果未指定 Axes 对象，则获取当前 Axes
    if ax is None:
        ax = gca()
    # 调用 `remove` 方法移除指定的 Axes 对象
    ax.remove()


# 设置当前 Axes 为指定的 Axes 对象 `ax`，并将当前图设置为 `ax` 所属的 Figure。
def sca(ax: Axes) -> None:
    """
    设置当前 Axes 为 `ax`，并将当前 Figure 设置为 `ax` 的父级。
    """
    # Mypy 视 `ax.figure` 为可能为 None，
    # 但在调用此函数时，`ax.figure` 不会为 None。
    # 此外，`Figure` 和 `FigureBase` 之间的细微差别是 mypy 捕获的。
    figure(ax.figure)  # type: ignore[arg-type]
    ax.figure.sca(ax)  # type: ignore[union-attr]


# 清空当前 Axes 的内容。
def cla() -> None:
    """清空当前 Axes 的内容。"""
    # 调用 `gca()` 获取当前 Axes，然后调用其 `cla()` 方法清空内容
    return gca().cla()


## More ways of creating Axes ##

# 添加一个 Axes 到当前图中，或检索一个已存在的 Axes。
@_docstring.dedent_interpd
def subplot(*args, **kwargs) -> Axes:
    """
    添加一个 Axes 到当前图中，或检索一个已存在的 Axes。

    这是 `.Figure.add_subplot` 的包装，提供了在使用隐式 API 时的额外行为（见注释部分）。

    调用签名::

       subplot(nrows, ncols, index, **kwargs)
       subplot(pos, **kwargs)
       subplot(**kwargs)
       subplot(ax)

    参数
    ----------
    *args : int, (int, int, *index*) 或 `.SubplotSpec`，默认：(1, 1, 1)
        Axes 的位置，可以是以下之一：

        - 三个整数（*nrows*，*ncols*，*index*）。Axes 将位于具有 *nrows* 行和 *ncols* 列的网格上的 *index* 位置。
          *index* 从左上角开始为 1，向右增加。*index* 也可以是一个二元组，指定（*first*，*last*）索引（基于 1，并包括 *last*），
          例如，``fig.add_subplot(3, 1, (1, 2))`` 创建一个跨越图的上部 2/3 的子图。
        - 一个 3 位整数。这些位被解释为分别给定的三个单个位数整数，即 ``fig.add_subplot(235)`` 等同于 ``fig.add_subplot(2, 3, 5)``。
          请注意，只有在子图不超过 9 个时才能使用此选项。
        - 一个 `.SubplotSpec`。

    projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
'polar', 'rectilinear', str}，可选
        Axes (`~.axes.Axes`) 的投影类型。*str* 是自定义投影的名称，请参见 `~matplotlib.projections`。
        默认为 None，将使用 'rectilinear' 投影。

    polar : bool，默认：False
        如果为 True，则等效于 projection='polar'。

    sharex, sharey : `~matplotlib.axes.Axes`，可选
        与 `sharex` 和/或 `sharey` 共享 x 或 y 轴 `~matplotlib.axis`。共享 Axes 的轴将具有相同的限制、刻度和比例。

    label : str
        返回的 Axes 的标签。

    返回
    -------
    返回添加或检索的 Axes 对象。
    """
    # 这里我们只会规范化 `polar=True` 和 `projection='polar'`，其余部分交给后续的代码处理。
    unset = object()
    # 从 `kwargs` 中获取 `projection` 参数的值，如果没有则使用 `unset` 作为默认值
    projection = kwargs.get('projection', unset)
    # 从 `kwargs` 中弹出 `polar` 参数的值，如果没有则使用 `unset` 作为默认值
    polar = kwargs.pop('polar', unset)
    # 如果极坐标未被设置且为真，则执行以下逻辑
    if polar is not unset and polar:
        # 如果用户提供的参数矛盾，抛出异常
        if projection is not unset and projection != 'polar':
            raise ValueError(
                f"polar={polar}, yet projection={projection!r}. "
                "Only one of these arguments should be supplied."
            )
        # 将投影设置为极坐标，并更新到关键字参数中
        kwargs['projection'] = projection = 'polar'

    # 如果 subplot 被调用时没有传入参数，则默认创建 subplot(1, 1, 1)
    if len(args) == 0:
        args = (1, 1, 1)

    # 检查是否传入了布尔类型作为第三个参数，这是为了防止误将 subplot(1, 2, False) 写成 subplots(1, 2, False)
    if len(args) >= 3 and isinstance(args[2], bool):
        # 发出警告，提醒用户可能使用错误的函数
        _api.warn_external("The subplot index argument to subplot() appears "
                           "to be a boolean. Did you intend to use "
                           "subplots()?")

    # 检查是否传入了 'nrows' 或 'ncols' 作为关键字参数，这些参数不适用于 subplot 函数
    if 'nrows' in kwargs or 'ncols' in kwargs:
        # 如果传入了不期望的关键字参数，抛出类型错误异常
        raise TypeError("subplot() got an unexpected keyword argument 'ncols' "
                        "and/or 'nrows'.  Did you intend to call subplots()?")

    # 获取当前的图形对象
    fig = gcf()

    # 首先查找是否存在符合条件的子图规格
    key = SubplotSpec._from_subplot_args(fig, args)

    for ax in fig.axes:
        # 如果找到位置匹配的 Axes 对象，则可以重用它，条件是未传入关键字参数或者传入的关键字参数和 Axes 类的初始化参数相同
        if (ax.get_subplotspec() == key
            and (kwargs == {}
                 or (ax._projection_init
                     == fig._process_projection_requirements(**kwargs)))):
            break
    else:
        # 遍历完所有已知的 Axes 对象都不匹配，因此创建一个新的 Axes 对象
        ax = fig.add_subplot(*args, **kwargs)

    # 将当前图形设置为操作的 Axes 对象
    fig.sca(ax)

    # 返回操作的 Axes 对象
    return ax
# 导入函数重载的装饰器，用于支持多种不同参数类型的函数定义
@overload
def subplots(
    nrows: Literal[1] = ...,
    ncols: Literal[1] = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: Literal[True] = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    **fig_kw
) -> tuple[Figure, Axes]:
    ...


# 导入函数重载的装饰器，用于支持多种不同参数类型的函数定义
@overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: Literal[False],
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    **fig_kw
) -> tuple[Figure, np.ndarray]:  # TODO numpy/numpy#24738
    ...


# 导入函数重载的装饰器，用于支持多种不同参数类型的函数定义
@overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: bool = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    **fig_kw
) -> tuple[Figure, Axes | np.ndarray]:
    ...


# 创建函数 subplots，用于生成包含子图网格的图形和图表
def subplots(
    nrows: int = 1, ncols: int = 1, *,
    sharex: bool | Literal["none", "all", "row", "col"] = False,
    sharey: bool | Literal["none", "all", "row", "col"] = False,
    squeeze: bool = True,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    subplot_kw: dict[str, Any] | None = None,
    gridspec_kw: dict[str, Any] | None = None,
    **fig_kw
) -> tuple[Figure, Any]:
    """
    创建图形和子图网格的实用包装器。

    此实用程序包装器使得一次调用即可方便地创建子图的常见布局，包括封装的图形对象。

    Parameters
    ----------
    nrows, ncols : int, default: 1
        子图网格的行数和列数。
    # sharex, sharey 控制子图之间 x 或 y 轴的共享方式
    sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
        Controls sharing of properties among x (*sharex*) or y (*sharey*)
        axes:
        
        - True or 'all': x- or y-axis will be shared among all subplots.
        - False or 'none': each subplot x- or y-axis will be independent.
        - 'row': each subplot row will share an x- or y-axis.
        - 'col': each subplot column will share an x- or y-axis.
        
        When subplots have a shared x-axis along a column, only the x tick
        labels of the bottom subplot are created. Similarly, when subplots
        have a shared y-axis along a row, only the y tick labels of the first
        column subplot are created. To later turn other subplots' ticklabels
        on, use `~matplotlib.axes.Axes.tick_params`.
        
        When subplots have a shared axis that has units, calling
        `.Axis.set_units` will update each axis with the new units.
        
        Note that it is not possible to unshare axes.

    # squeeze 控制是否从返回的 `~matplotlib.axes.Axes` 数组中挤出额外的维度
    squeeze : bool, default: True
        - If True, extra dimensions are squeezed out from the returned
          array of `~matplotlib.axes.Axes`:
          
          - if only one subplot is constructed (nrows=ncols=1), the
            resulting single Axes object is returned as a scalar.
          - for Nx1 or 1xM subplots, the returned object is a 1D numpy
            object array of Axes objects.
          - for NxM, subplots with N>1 and M>1 are returned as a 2D array.
          
        - If False, no squeezing at all is done: the returned Axes object is
          always a 2D array containing Axes instances, even if it ends up
          being 1x1.

    # width_ratios 定义列的相对宽度
    width_ratios : array-like of length *ncols*, optional
        Defines the relative widths of the columns. Each column gets a
        relative width of ``width_ratios[i] / sum(width_ratios)``.
        If not given, all columns will have the same width.  Equivalent
        to ``gridspec_kw={'width_ratios': [...]}``.

    # height_ratios 定义行的相对高度
    height_ratios : array-like of length *nrows*, optional
        Defines the relative heights of the rows. Each row gets a
        relative height of ``height_ratios[i] / sum(height_ratios)``.
        If not given, all rows will have the same height. Convenience
        for ``gridspec_kw={'height_ratios': [...]}``.

    # subplot_kw 传递给 `~matplotlib.figure.Figure.add_subplot` 调用的关键字参数
    subplot_kw : dict, optional
        Dict with keywords passed to the
        `~matplotlib.figure.Figure.add_subplot` call used to create each
        subplot.

    # gridspec_kw 传递给 `~matplotlib.gridspec.GridSpec` 构造函数的关键字参数
    gridspec_kw : dict, optional
        Dict with keywords passed to the `~matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.

    # **fig_kw 所有额外的关键字参数传递给 `.pyplot.figure` 调用
    **fig_kw
        All additional keyword arguments are passed to the
        `.pyplot.figure` call.

    # 返回一个 `.Figure` 对象
    Returns
    -------
    fig : `.Figure`
    # 使用传入的参数 `fig_kw` 创建一个新的 Figure 对象
    fig = figure(**fig_kw)
    
    # 在新创建的 Figure 对象上创建子图，根据指定的参数设置子图的行数 `nrows` 和列数 `ncols`，以及共享的 X 轴和 Y 轴设置 `sharex` 和 `sharey`。
    # 这些子图可以通过 `squeeze` 参数来控制生成的数组的维度。子图的样式可以通过 `subplot_kw` 参数来指定，如投影类型。
    # 可以通过 `gridspec_kw` 参数来进一步调整子图的布局，`height_ratios` 和 `width_ratios` 参数可以指定子图的高度和宽度比例。
    axs = fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
                       squeeze=squeeze, subplot_kw=subplot_kw,
                       gridspec_kw=gridspec_kw, height_ratios=height_ratios,
                       width_ratios=width_ratios)
    
    # 返回创建的 Figure 对象 `fig` 和生成的子图数组 `axs`
    return fig, axs
# subplot_mosaic 函数的重载，接受一个字符串参数 mosaic，表示布局结构，返回一个元组，包含一个 Figure 对象和一个字典，字典的键是字符串，值是 matplotlib.axes.Axes 对象
@overload
def subplot_mosaic(
    mosaic: str,
    *,
    sharex: bool = ...,
    sharey: bool = ...,
    width_ratios: ArrayLike | None = ...,
    height_ratios: ArrayLike | None = ...,
    empty_sentinel: str = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    per_subplot_kw: dict[str | tuple[str, ...], dict[str, Any]] | None = ...,
    **fig_kw: Any
) -> tuple[Figure, dict[str, matplotlib.axes.Axes]]: ...


# subplot_mosaic 函数的重载，接受一个嵌套列表参数 mosaic，表示布局结构，返回一个元组，包含一个 Figure 对象和一个字典，字典的键是 Hashable 类型的列表元素，值是 matplotlib.axes.Axes 对象
@overload
def subplot_mosaic(
    mosaic: list[HashableList[_T]],
    *,
    sharex: bool = ...,
    sharey: bool = ...,
    width_ratios: ArrayLike | None = ...,
    height_ratios: ArrayLike | None = ...,
    empty_sentinel: _T = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    per_subplot_kw: dict[_T | tuple[_T, ...], dict[str, Any]] | None = ...,
    **fig_kw: Any
) -> tuple[Figure, dict[_T, matplotlib.axes.Axes]]: ...


# subplot_mosaic 函数的重载，接受一个嵌套列表参数 mosaic，表示布局结构，返回一个元组，包含一个 Figure 对象和一个字典，字典的键是 Hashable 类型的元素，值是 matplotlib.axes.Axes 对象
@overload
def subplot_mosaic(
    mosaic: list[HashableList[Hashable]],
    *,
    sharex: bool = ...,
    sharey: bool = ...,
    width_ratios: ArrayLike | None = ...,
    height_ratios: ArrayLike | None = ...,
    empty_sentinel: Any = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    per_subplot_kw: dict[Hashable | tuple[Hashable, ...], dict[str, Any]] | None = ...,
    **fig_kw: Any
) -> tuple[Figure, dict[Hashable, matplotlib.axes.Axes]]: ...


# subplot_mosaic 函数的主体实现，根据不同的参数类型构建基于 ASCII 艺术或嵌套列表的 Axes 布局
def subplot_mosaic(
    mosaic: str | list[HashableList[_T]] | list[HashableList[Hashable]],
    *,
    sharex: bool = False,
    sharey: bool = False,
    width_ratios: ArrayLike | None = None,
    height_ratios: ArrayLike | None = None,
    empty_sentinel: Any = '.',
    subplot_kw: dict[str, Any] | None = None,
    gridspec_kw: dict[str, Any] | None = None,
    per_subplot_kw: dict[str | tuple[str, ...], dict[str, Any]] |
                    dict[_T | tuple[_T, ...], dict[str, Any]] |
                    dict[Hashable | tuple[Hashable, ...], dict[str, Any]] | None = None,
    **fig_kw: Any
) -> tuple[Figure, dict[str, matplotlib.axes.Axes]] | \
     tuple[Figure, dict[_T, matplotlib.axes.Axes]] | \
     tuple[Figure, dict[Hashable, matplotlib.axes.Axes]]:
    """
    根据 ASCII 艺术或嵌套列表构建 Axes 布局的辅助函数。

    这是一个用于视觉上构建复杂 GridSpec 布局的辅助函数。

    详细 API 文档和示例见 :ref:`mosaic`。

    参数
    ----------
    mosaic : str | list[HashableList[_T]] | list[HashableList[Hashable]]
        定义布局结构的字符串或嵌套列表。

    sharex : bool, optional, 默认为 False
        是否共享 x 轴。

    sharey : bool, optional, 默认为 False
        是否共享 y 轴。

    width_ratios : ArrayLike or None, optional, 默认为 None
        子图列的宽度比例。

    height_ratios : ArrayLike or None, optional, 默认为 None
        子图行的高度比例。

    empty_sentinel : Any, optional, 默认为 '.'
        在布局定义中表示空白区域的标记。

    subplot_kw : dict[str, Any] or None, optional, 默认为 None
        应用于每个子图的关键字参数。

    gridspec_kw : dict[str, Any] or None, optional, 默认为 None
        GridSpec 的关键字参数。

    per_subplot_kw : dict[str | tuple[str, ...], dict[str, Any]] or
                     dict[_T | tuple[_T, ...], dict[str, Any]] or
                     dict[Hashable | tuple[Hashable, ...], dict[str, Any]] or None, optional, 默认为 None
        每个子图的特定关键字参数，可以是字符串键或元组键。

    **fig_kw : Any
        其它 Figure 对象的关键字参数。

    返回
    -------
    tuple
        一个包含 Figure 对象和字典的元组，字典的键为字符串，值为 matplotlib.axes.Axes 对象，
        或者键为 _T 类型或 Hashable 类型，值为 matplotlib.axes.Axes 对象。
    """
    # mosaic 是一个用于描述子图排列方式的数据结构，可以是字符串或嵌套的列表。
    # 如果是字符串，每个字符代表一个子图位置；如果是列表的列表，则可以创建嵌套布局。
    mosaic : list of list of {hashable or nested} or str

        # 例如，下面的布局：
        # x = [['A panel', 'A panel', 'edge'],
        #      ['C panel', '.',       'edge']]
        
        # 将产生四个子图：
        
        # - 'A panel'，占据第一行，跨越前两列
        # - 'edge'，占据两行，在右侧
        # - 'C panel'，占据左下角的一个单元格
        # - 一个空白，占据底部中间的一个单元格

        # 布局中的任何条目都可以是相同形式的列表的列表，以创建嵌套布局。

        # 如果输入是字符串，则必须是以下形式：
        # '''
        # AAE
        # C.E
        # '''

        # 每个字符代表一个列，每行代表一行子图。
        # 这种格式只允许单字符的子图标签，不允许嵌套，但非常简洁。

    # sharex 和 sharey 控制是否共享子图的 x 轴和 y 轴。
    # 如果为 True，则所有子图共享 x 轴（sharex=True）或 y 轴（sharey=True）。
    # 在这种情况下，刻度标签的可见性和轴单位的行为与 `subplots` 相同。
    # 如果为 False，则每个子图的 x 轴或 y 轴是独立的。
    sharex, sharey : bool, default: False

    # width_ratios 是一个数组，用于指定各列的相对宽度。
    # 每列的宽度比例为 ``width_ratios[i] / sum(width_ratios)``。
    # 如果不提供，所有列将具有相同的宽度。
    # 这是对 ``gridspec_kw={'width_ratios': [...]} 的方便写法。
    width_ratios : array-like of length *ncols*, optional

    # height_ratios 是一个数组，用于指定各行的相对高度。
    # 每行的高度比例为 ``height_ratios[i] / sum(height_ratios)``。
    # 如果不提供，所有行将具有相同的高度。
    # 这是对 ``gridspec_kw={'height_ratios': [...]} 的方便写法。
    height_ratios : array-like of length *nrows*, optional

    # empty_sentinel 是在布局中表示“留空此处”的条目。
    # 默认为 ``'.'``。注意，如果 *layout* 是一个字符串，则通过 `inspect.cleandoc` 处理，
    # 以去除前导空格，这可能会干扰使用空格作为空标志的功能。

    empty_sentinel : object, optional

    # subplot_kw 是一个字典，包含传递给 `.Figure.add_subplot` 调用的关键字参数，
    # 用于创建每个子图。这些值可以被 *per_subplot_kw* 中的值覆盖。
    subplot_kw : dict, optional
    per_subplot_kw : dict, optional
        一个字典，将Axes标识符或标识符元组映射到关键字参数字典，这些参数将传递给`.Figure.add_subplot`调用，用于创建每个子图。这些字典中的值优先于*subplot_kw*中的值。

        如果*mosaic*是一个字符串，因此所有键都是单个字符，则可以使用单个字符串而不是元组作为键；即``"AB"``等效于``("A", "B")``。

        .. versionadded:: 3.7

    gridspec_kw : dict, optional
        传递给`.GridSpec`构造函数的关键字字典，用于创建子图所放置的网格。

    **fig_kw
        所有额外的关键字参数都将传递给`.pyplot.figure`调用。

    Returns
    -------
    fig : `.Figure`
       新的图形对象

    dict[label, Axes]
       一个字典，将标签映射到Axes对象。Axes对象的顺序是它们在总布局中位置的从左到右和从上到下的顺序。

    """
    fig = figure(**fig_kw)  # 创建一个新的图形对象，使用传入的关键字参数fig_kw

    ax_dict = fig.subplot_mosaic(  # type: ignore[misc]
        mosaic,  # type: ignore[arg-type]
        sharex=sharex, sharey=sharey,  # 设置共享的x轴和y轴
        height_ratios=height_ratios, width_ratios=width_ratios,  # 设置子图的高度比例和宽度比例
        subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,  # 传递给子图的关键字参数和网格参数
        empty_sentinel=empty_sentinel,
        per_subplot_kw=per_subplot_kw,  # type: ignore[arg-type]
    )
    return fig, ax_dict  # 返回新创建的图形对象fig和Axes对象字典ax_dict
# 创建一个在规则网格中特定位置的子图。

def subplot2grid(
    shape: tuple[int, int], loc: tuple[int, int],
    rowspan: int = 1, colspan: int = 1,
    fig: Figure | None = None,
    **kwargs
) -> matplotlib.axes.Axes:
    """
    Create a subplot at a specific location inside a regular grid.

    Parameters
    ----------
    shape : (int, int)
        Number of rows and columns of the grid in which to place axis.
    loc : (int, int)
        Row number and column number of the axis location within the grid.
    rowspan : int, default: 1
        Number of rows for the axis to span downwards.
    colspan : int, default: 1
        Number of columns for the axis to span to the right.
    fig : `.Figure`, optional
        Figure to place the subplot in. Defaults to the current figure.
    **kwargs
        Additional keyword arguments are handed to `~.Figure.add_subplot`.

    Returns
    -------
    `~.axes.Axes`

        The Axes of the subplot. The returned Axes can actually be an instance
        of a subclass, such as `.projections.polar.PolarAxes` for polar
        projections.

    Notes
    -----
    The following call ::

        ax = subplot2grid((nrows, ncols), (row, col), rowspan, colspan)

    is identical to ::

        fig = gcf()
        gs = fig.add_gridspec(nrows, ncols)
        ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
    """
    # 如果未指定 fig 参数，则使用当前的图形对象
    if fig is None:
        fig = gcf()
    # 从 shape 参数中获取行数和列数
    rows, cols = shape
    # 确保网格规范存在，获取网格规范对象
    gs = GridSpec._check_gridspec_exists(fig, rows, cols)
    # 创建一个新的子图规范对象，指定位置和跨度
    subplotspec = gs.new_subplotspec(loc, rowspan=rowspan, colspan=colspan)
    # 将子图添加到图形对象中，并传递其他关键字参数
    return fig.add_subplot(subplotspec, **kwargs)


# 创建并返回一个与现有 x 轴共享的第二个 Axes 对象，其刻度在右侧

def twinx(ax: matplotlib.axes.Axes | None = None) -> _AxesBase:
    """
    Make and return a second Axes that shares the *x*-axis.  The new Axes will
    overlay *ax* (or the current Axes if *ax* is *None*), and its ticks will be
    on the right.

    Examples
    --------
    :doc:`/gallery/subplots_axes_and_figures/two_scales`
    """
    # 如果未提供 ax 参数，则使用当前的坐标轴
    if ax is None:
        ax = gca()
    # 创建一个新的共享 x 轴的 Axes 对象
    ax1 = ax.twinx()
    return ax1


# 创建并返回一个与现有 y 轴共享的第二个 Axes 对象，其刻度在顶部

def twiny(ax: matplotlib.axes.Axes | None = None) -> _AxesBase:
    """
    Make and return a second Axes that shares the *y*-axis.  The new Axes will
    overlay *ax* (or the current Axes if *ax* is *None*), and its ticks will be
    on the top.

    Examples
    --------
    :doc:`/gallery/subplots_axes_and_figures/two_scales`
    """
    # 如果未提供 ax 参数，则使用当前的坐标轴
    if ax is None:
        ax = gca()
    # 创建一个新的共享 y 轴的 Axes 对象
    ax1 = ax.twiny()
    return ax1


# 为指定图形对象启动一个子图工具窗口

def subplot_tool(targetfig: Figure | None = None) -> SubplotTool | None:
    """
    Launch a subplot tool window for a figure.

    Returns
    -------
    `matplotlib.widgets.SubplotTool`
    """
    # 如果未指定 targetfig 参数，则使用当前的图形对象
    if targetfig is None:
        targetfig = gcf()
    # 获取图形对象的画布管理器的工具栏对象
    tb = targetfig.canvas.manager.toolbar  # type: ignore[union-attr]
    # 检查工具栏对象是否具有配置子图的方法
    if hasattr(tb, "configure_subplots"):  # toolbar2
        from matplotlib.backend_bases import NavigationToolbar2
        # 调用配置子图方法并返回结果
        return cast(NavigationToolbar2, tb).configure_subplots()
    # 如果 traceback 对象 tb 具有属性 "trigger_tool"，表明使用的是 toolmanager
    elif hasattr(tb, "trigger_tool"):  # toolmanager
        # 从 matplotlib.backend_bases 模块中导入 ToolContainerBase 类
        from matplotlib.backend_bases import ToolContainerBase
        # 将 tb 强制类型转换为 ToolContainerBase 类型，并调用 trigger_tool 方法
        cast(ToolContainerBase, tb).trigger_tool("subplots")
        # 返回空值 None，表示处理完成
        return None
    # 如果没有 "trigger_tool" 属性，则抛出 ValueError 异常
    else:
        # 抛出异常，指明 subplot_tool 只能用于带有关联工具栏的图形
        raise ValueError("subplot_tool can only be launched for figures with "
                         "an associated toolbar")
# 获取当前图表的坐标轴对象
ax = gca()

# 如果未指定参数，则返回当前 x 轴的限制范围
if not ticks and not labels:
    return ax.get_xlim()

# 否则，根据传入的参数设置新的 x 轴限制范围，并返回设置后的结果
ret = ax.set_xlim(*args, **kwargs)
return ret



# 获取当前图表的坐标轴对象
ax = gca()

# 如果未指定参数，则返回当前 y 轴的限制范围
if not ticks and not labels:
    return ax.get_ylim()

# 否则，根据传入的参数设置新的 y 轴限制范围，并返回设置后的结果
ret = ax.set_ylim(*args, **kwargs)
return ret



# 获取当前图表的坐标轴对象
ax = gca()

# 如果未指定 ticks 和 labels 参数，则返回当前 x 轴刻度位置和标签
if not ticks and not labels:
    return ax.get_xticks(minor=minor), ax.get_xticklabels(minor=minor, **kwargs)

# 否则，根据传入的 ticks 和 labels 参数设置新的 x 轴刻度位置和标签，并返回设置后的结果
ret = ax.set_xticks(ticks, minor=minor)
ax.set_xticklabels(labels, **kwargs)
return ret
    # 获取当前的坐标轴对象
    ax = gca()
    
    # 定义 locs 变量，用于存储 x 轴刻度的位置信息，类型为 Tick 对象列表或 NumPy 数组
    locs: list[Tick] | np.ndarray
    
    # 如果未传入 ticks 参数，则获取主要或次要刻度的位置信息
    if ticks is None:
        locs = ax.get_xticks(minor=minor)
        # 如果 labels 参数不为 None，则抛出类型错误，因为 labels 参数只能在设置 ticks 参数时一同传入
        if labels is not None:
            raise TypeError("xticks(): Parameter 'labels' can't be set "
                            "without setting 'ticks'")
    else:
        # 如果传入了 ticks 参数，则设置主要或次要刻度的位置信息
        locs = ax.set_xticks(ticks, minor=minor)
    
    # 定义 labels_out 变量，用于存储 x 轴刻度的标签文本对象列表
    labels_out: list[Text] = []
    
    # 如果 labels 参数为 None，则获取主要或次要刻度的标签文本对象，并根据 kwargs 更新每个标签的属性
    if labels is None:
        labels_out = ax.get_xticklabels(minor=minor)
        for l in labels_out:
            l._internal_update(kwargs)
    else:
        # 如果传入了 labels 参数，则设置主要或次要刻度的标签文本，并根据 kwargs 更新标签的属性
        labels_out = ax.set_xticklabels(labels, minor=minor, **kwargs)
    
    # 返回 locs（刻度位置信息列表）和 labels_out（标签文本对象列表）
    return locs, labels_out
def yticks(
    ticks: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
    *,
    minor: bool = False,
    **kwargs
) -> tuple[list[Tick] | np.ndarray, list[Text]]:
    """
    Get or set the current tick locations and labels of the y-axis.

    Pass no arguments to return the current values without modifying them.

    Parameters
    ----------
    ticks : array-like, optional
        The list of ytick locations.  Passing an empty list removes all yticks.
    labels : array-like, optional
        The labels to place at the given *ticks* locations.  This argument can
        only be passed if *ticks* is passed as well.
    minor : bool, default: False
        If ``False``, get/set the major ticks/labels; if ``True``, the minor
        ticks/labels.
    **kwargs
        `.Text` properties can be used to control the appearance of the labels.

        .. warning::

            This only sets the properties of the current ticks, which is
            only sufficient if you either pass *ticks*, resulting in a
            fixed list of ticks, or if the plot is static.

            Ticks are not guaranteed to be persistent. Various operations
            can create, delete and modify the Tick instances. There is an
            imminent risk that these settings can get lost if you work on
            the figure further (including also panning/zooming on a
            displayed figure).

            Use `~.pyplot.tick_params` instead if possible.

    Returns
    -------
    locs
        The list of ytick locations.
    labels
        The list of ylabel `.Text` objects.

    Notes
    -----
    Calling this function with no arguments (e.g. ``yticks()``) is the pyplot
    equivalent of calling `~.Axes.get_yticks` and `~.Axes.get_yticklabels` on
    the current Axes.
    Calling this function with arguments is the pyplot equivalent of calling
    `~.Axes.set_yticks` and `~.Axes.set_yticklabels` on the current Axes.

    Examples
    --------
    >>> locs, labels = yticks()  # Get the current locations and labels.
    >>> yticks(np.arange(0, 1, step=0.2))  # Set label locations.
    >>> yticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
    >>> yticks([0, 1, 2], ['January', 'February', 'March'],
    ...        rotation=45)  # Set text labels and properties.
    >>> yticks([])  # Disable yticks.
    """
    # 获取当前的坐标轴对象
    ax = gca()

    # 定义存储刻度位置的变量，类型为 Tick 对象列表或 ndarray
    locs: list[Tick] | np.ndarray
    # 如果 ticks 参数为 None，则获取主要或次要刻度的位置
    if ticks is None:
        locs = ax.get_yticks(minor=minor)
        # 如果 labels 参数不为 None，则抛出异常，因为不能单独设置 labels 而不设置 ticks
        if labels is not None:
            raise TypeError("yticks(): Parameter 'labels' can't be set "
                            "without setting 'ticks'")
    else:
        # 否则设置 y 轴刻度为指定的 ticks 值，可以选择设置为主要或次要刻度
        locs = ax.set_yticks(ticks, minor=minor)

    # 定义存储刻度标签的列表
    labels_out: list[Text] = []
    # 如果 labels 参数为 None，则获取主要或次要刻度的标签，并更新标签的属性
    if labels is None:
        labels_out = ax.get_yticklabels(minor=minor)
        for l in labels_out:
            l._internal_update(kwargs)
    else:
        # 否则设置 y 轴刻度标签为指定的 labels 值，并设置其属性
        labels_out = ax.set_yticklabels(labels, minor=minor, **kwargs)

    # 返回刻度位置和刻度标签的元组
    return locs, labels_out
    radii: ArrayLike | None = None,
    labels: Sequence[str | Text] | None = None,
    angle: float | None = None,
    fmt: str | None = None,
    **kwargs


    # 定义函数参数：radii，可以是类数组类型或者空值（默认为None）
    radii: ArrayLike | None = None,
    # 定义函数参数：labels，可以是字符串序列或文本序列，或者空值（默认为None）
    labels: Sequence[str | Text] | None = None,
    # 定义函数参数：angle，可以是浮点数或空值（默认为None）
    angle: float | None = None,
    # 定义函数参数：fmt，可以是字符串或空值（默认为None）
    fmt: str | None = None,
    # 定义额外的关键字参数kwargs，可以接收任意额外的参数
    **kwargs
def rgrids(
    radii: tuple[float], 
    labels: tuple[str] | None = None, 
    angle: float = 22.5, 
    fmt: str | None = None, 
    **kwargs
) -> tuple[list[Line2D], list[Text]]:
    """
    Get or set the radial gridlines on the current polar plot.

    Call signatures::

     lines, labels = rgrids()
     lines, labels = rgrids(radii, labels=None, angle=22.5, fmt=None, **kwargs)

    When called with no arguments, `.rgrids` simply returns the tuple
    (*lines*, *labels*). When called with arguments, the labels will
    appear at the specified radial distances and angle.

    Parameters
    ----------
    radii : tuple with floats
        The radii for the radial gridlines

    labels : tuple with strings or None
        The labels to use at each radial gridline. The
        `matplotlib.ticker.ScalarFormatter` will be used if None.

    angle : float
        The angular position of the radius labels in degrees.

    fmt : str or None
        Format string used in `matplotlib.ticker.FormatStrFormatter`.
        For example '%f'.

    Returns
    -------
    lines : list of `.lines.Line2D`
        The radial gridlines.

    labels : list of `.text.Text`
        The tick labels.

    Other Parameters
    ----------------
    **kwargs
        *kwargs* are optional `.Text` properties for the labels.

    See Also
    --------
    .pyplot.thetagrids
    .projections.polar.PolarAxes.set_rgrids
    .Axis.get_gridlines
    .Axis.get_ticklabels

    Examples
    --------
    ::

      # set the locations of the radial gridlines
      lines, labels = rgrids( (0.25, 0.5, 1.0) )

      # set the locations and labels of the radial gridlines
      lines, labels = rgrids( (0.25, 0.5, 1.0), ('Tom', 'Dick', 'Harry' ))
    """
    # 获取当前坐标轴对象
    ax = gca()
    # 检查当前坐标轴是否为极坐标系，若不是则抛出运行时错误
    if not isinstance(ax, PolarAxes):
        raise RuntimeError('rgrids only defined for polar Axes')
    # 如果所有参数都未指定且kwargs为空，则返回当前坐标轴的径向网格线和刻度标签
    if all(p is None for p in [radii, labels, angle, fmt]) and not kwargs:
        lines_out: list[Line2D] = ax.yaxis.get_gridlines()
        labels_out: list[Text] = ax.yaxis.get_ticklabels()
    # 否则，调用坐标轴的设置径向网格线方法，设置指定的参数并返回结果
    elif radii is not None:
        lines_out, labels_out = ax.set_rgrids(
            radii, labels=labels, angle=angle, fmt=fmt, **kwargs)
    else:
        # 如果radii为None但其他参数有值，则抛出类型错误
        raise TypeError("'radii' cannot be None when other parameters are passed")
    return lines_out, labels_out



def thetagrids(
    angles: ArrayLike | None = None,
    labels: Sequence[str | Text] | None = None,
    fmt: str | None = None,
    **kwargs
) -> tuple[list[Line2D], list[Text]]:
    """
    Get or set the theta gridlines on the current polar plot.

    Call signatures::

     lines, labels = thetagrids()
     lines, labels = thetagrids(angles, labels=None, fmt=None, **kwargs)

    When called with no arguments, `.thetagrids` simply returns the tuple
    (*lines*, *labels*). When called with arguments, the labels will
    appear at the specified angles.

    Parameters
    ----------
    angles : tuple with floats, degrees
        The angles of the theta gridlines.

    labels : sequence of strings or None
        The labels to use at each theta gridline.

    fmt : str or None
        Format string used in `matplotlib.ticker.FormatStrFormatter`.
        For example '%f'.

    Returns
    -------
    lines : list of `.lines.Line2D`
        The theta gridlines.

    labels : list of `.text.Text`
        The tick labels.

    Other Parameters
    ----------------
    **kwargs
        *kwargs* are optional `.Text` properties for the labels.

    See Also
    --------
    .pyplot.rgrids
    .projections.polar.PolarAxes.set_thetagrids
    .Axis.get_gridlines
    .Axis.get_ticklabels

    Examples
    --------
    ::

      # set the locations of the theta gridlines
      lines, labels = thetagrids( (0, 45, 90, 135, 180) )

      # set the locations and labels of the theta gridlines
      lines, labels = thetagrids( (0, 45, 90, 135, 180), ('N', 'NE', 'E', 'SE', 'S') )
    """
    # 获取当前坐标轴对象
    ax = gca()
    # 检查当前坐标轴是否为极坐标系，若不是则抛出运行时错误
    if not isinstance(ax, PolarAxes):
        raise RuntimeError('thetagrids only defined for polar Axes')
    # 如果所有参数都未指定且kwargs为空，则返回当前坐标轴的角度网格线和刻度标签
    if all(p is None for p in [angles, labels, fmt]) and not kwargs:
        lines_out: list[Line2D] = ax.xaxis.get_gridlines()
        labels_out: list[Text] = ax.xaxis.get_ticklabels()
    # 否则，调用坐标轴的设置角度网格线方法，设置指定的参数并返回结果
    else:
        lines_out, labels_out = ax.set_thetagrids(
            angles, labels=labels, fmt=fmt, **kwargs)
    return lines_out, labels_out
    # 获取当前的坐标轴对象
    ax = gca()
    # 如果坐标轴不是极坐标轴类型，则抛出异常
    if not isinstance(ax, PolarAxes):
        raise RuntimeError('thetagrids only defined for polar Axes')
    # 如果 angles, labels, fmt 均为 None，并且没有额外的关键字参数，则获取默认的角度网格线和标签
    if all(param is None for param in [angles, labels, fmt]) and not kwargs:
        # 获取当前坐标轴的角度刻度线
        lines_out: list[Line2D] = ax.xaxis.get_ticklines()
        # 获取当前坐标轴的角度刻度标签
        labels_out: list[Text] = ax.xaxis.get_ticklabels()
    # 否则，如果 angles 不是 None，则设置自定义的角度网格线和标签
    elif angles is None:
        raise TypeError("'angles' cannot be None when other parameters are passed")
    else:
        # 调用坐标轴对象的 set_thetagrids 方法设置自定义的角度网格线和标签
        lines_out, labels_out = ax.set_thetagrids(angles,
                                                  labels=labels, fmt=fmt,
                                                  **kwargs)
    # 返回生成的角度网格线和标签列表
    return lines_out, labels_out
# 将该函数标记为即将弃用，预计在版本 3.7 中弃用
@_api.deprecated("3.7", pending=True)
# 定义函数 `get_plot_commands()`，返回所有绘图命令的排序列表
def get_plot_commands() -> list[str]:
    # 定义非绘图命令的集合
    NON_PLOT_COMMANDS = {
        'connect', 'disconnect', 'get_current_fig_manager', 'ginput',
        'new_figure_manager', 'waitforbuttonpress'}
    # 返回所有不属于非绘图命令的绘图命令列表
    return [name for name in _get_pyplot_commands()
            if name not in NON_PLOT_COMMANDS]


# 定义函数 `_get_pyplot_commands()`，返回所有绘图命令的列表
def _get_pyplot_commands() -> list[str]:
    # 定义需要排除的命令集合，包括色图设置函数和带有前置下划线的私有函数
    exclude = {'colormaps', 'colors', 'get_plot_commands', *colormaps}
    # 获取定义 `get_plot_commands` 的模块对象
    this_module = inspect.getmodule(get_plot_commands)
    # 返回排序后的全局函数名称列表，不包括以下划线开头的私有函数和排除列表中的函数
    return sorted(
        name for name, obj in globals().items()
        if not name.startswith('_') and name not in exclude
           and inspect.isfunction(obj)
           and inspect.getmodule(obj) is this_module)


## Plotting part 1: manually generated functions and wrappers ##


# 将该函数的文档字符串和弃用标记从 `Figure.colorbar` 复制过来
def colorbar(
    mappable: ScalarMappable | None = None,
    cax: matplotlib.axes.Axes | None = None,
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None,
    **kwargs
) -> Colorbar:
    # 如果未提供 `mappable` 参数，则使用 `gci()` 函数获取一个可映射对象
    if mappable is None:
        mappable = gci()
        # 如果未找到可映射对象，则引发运行时错误
        if mappable is None:
            raise RuntimeError('No mappable was found to use for colorbar '
                               'creation. First define a mappable such as '
                               'an image (with imshow) or a contour set ('
                               'with contourf).')
    # 调用 `gcf().colorbar()` 创建颜色条对象，并返回结果
    ret = gcf().colorbar(mappable, cax=cax, ax=ax, **kwargs)
    return ret


# 定义函数 `clim(vmin: float | None = None, vmax: float | None = None)`，设置当前图像的颜色限制
def clim(vmin: float | None = None, vmax: float | None = None) -> None:
    """
    Set the color limits of the current image.

    If either *vmin* or *vmax* is None, the image min/max respectively
    will be used for color scaling.

    If you want to set the clim of multiple images, use
    `~.ScalarMappable.set_clim` on every image, for example::

      for im in gca().get_images():
          im.set_clim(0, 0.5)

    """
    # 获取当前图像对象
    im = gci()
    # 如果未找到图像对象，则引发运行时错误
    if im is None:
        raise RuntimeError('You must first define an image, e.g., with imshow')

    # 设置图像对象的颜色限制
    im.set_clim(vmin, vmax)


# 定义函数 `get_cmap(name: Colormap | str | None = None, lut: int | None = None)`，获取颜色映射对象
def get_cmap(name: Colormap | str | None = None, lut: int | None = None) -> Colormap:
    """
    Get a colormap instance, defaulting to rc values if *name* is None.

    Parameters
    ----------
    name : `~matplotlib.colors.Colormap` or str or None, default: None
        If a `.Colormap` instance, it will be returned. Otherwise, the name of
        a colormap known to Matplotlib, which will be resampled by *lut*. The
        default, None, means :rc:`image.cmap`.

    """
    # 如果传入的 name 参数为 None，则使用默认的 rcParams['image.cmap'] 作为 colormap 名称
    if name is None:
        name = rcParams['image.cmap']
    
    # 如果 name 已经是 Colormap 的实例，则直接返回 name
    if isinstance(name, Colormap):
        return name
    
    # 使用 _api 检查 name 是否在已排序的 _colormaps 列表中
    _api.check_in_list(sorted(_colormaps), name=name)
    
    # 如果 lut 参数为 None，则直接返回 _colormaps 中对应名称的 Colormap 对象
    if lut is None:
        return _colormaps[name]
    else:
        # 否则，对 _colormaps[name] 的 Colormap 对象进行 resampled 操作，使其具有 lut 个条目的查找表
        return _colormaps[name].resampled(lut)
# 设置默认的色彩映射，并将其应用于当前的图像（如果存在）
def set_cmap(cmap: Colormap | str) -> None:
    # 使用给定的色彩映射名称或实例获取对应的 Colormap 对象
    cmap = get_cmap(cmap)
    
    # 设置全局的图像属性，将色彩映射设置为指定的映射名称
    rc('image', cmap=cmap.name)
    
    # 获取当前的图像对象
    im = gci()
    
    # 如果当前图像对象不为空，则设置其色彩映射为指定的 cmap
    if im is not None:
        im.set_cmap(cmap)


# 读取图像文件，并返回对应的 NumPy 数组
@_copy_docstring_and_deprecators(matplotlib.image.imread)
def imread(
        fname: str | pathlib.Path | BinaryIO, format: str | None = None
) -> np.ndarray:
    return matplotlib.image.imread(fname, format)


# 保存图像数组到指定文件中
@_copy_docstring_and_deprecators(matplotlib.image.imsave)
def imsave(
    fname: str | os.PathLike | BinaryIO, arr: ArrayLike, **kwargs
) -> None:
    matplotlib.image.imsave(fname, arr, **kwargs)


# 在新的图像窗口中显示一个二维数组作为矩阵
def matshow(A: ArrayLike, fignum: None | int = None, **kwargs) -> AxesImage:
    """
    The origin is set at the upper left hand corner.
    The indexing is ``(row, column)`` so that the first index runs vertically
    and the second index runs horizontally in the figure:
    
    ... 略去部分详细文档注释以保持简洁 ...
    
    Parameters
    ----------
    A : 2D array-like
        The matrix to be displayed.
    
    fignum : None or int
        If *None*, create a new, appropriately sized figure window.
        
        If 0, use the current Axes (creating one if there is none, without ever
        adjusting the figure size).
        
        Otherwise, create a new Axes on the figure with the given number
        (creating it at the appropriate size if it does not exist, but not
        adjusting the figure size otherwise).  Note that this will be drawn on
        top of any preexisting Axes on the figure.
    
    Returns
    -------
    `~matplotlib.image.AxesImage`
    
    Other Parameters
    ----------------
    **kwargs : `~matplotlib.axes.Axes.imshow` arguments
    """
    # 将输入的数组转换为 NumPy 数组
    A = np.asanyarray(A)
    
    # 如果 fignum 为 0，则获取当前的轴对象
    if fignum == 0:
        ax = gca()
    else:
        # 否则，根据数组的实际纵横比例创建新的图像窗口
        fig = figure(fignum, figsize=figaspect(A))
        ax = fig.add_axes((0.15, 0.09, 0.775, 0.775))
    
    # 在指定轴上显示矩阵 A，并传递额外的关键字参数
    im = ax.matshow(A, **kwargs)
    
    # 设置科学记数法显示图像
    sci(im)
    
    return im


# 创建极坐标图
def polar(*args, **kwargs) -> list[Line2D]:
    """
    Make a polar plot.

    call signature::

      polar(theta, r, **kwargs)

    Multiple *theta*, *r* arguments are supported, with format strings, as in
    `plot`.
    """
    # 如果存在轴对象，则检查它是否具有极坐标投影
    # 检查当前图形的坐标轴是否存在
    if gcf().get_axes():
        # 获取当前图形的当前坐标轴
        ax = gca()
        # 如果当前坐标轴不是极坐标轴（PolarAxes的实例），则发出警告
        if not isinstance(ax, PolarAxes):
            _api.warn_external('Trying to create polar plot on an Axes '
                               'that does not have a polar projection.')
    else:
        # 如果当前图形没有坐标轴，则创建一个极坐标轴
        ax = axes(projection="polar")
    
    # 返回根据传入参数绘制的图形对象
    return ax.plot(*args, **kwargs)
# 如果 rcParams['backend_fallback'] 为真，并且请求的是交互式后端，则忽略 rcParams['backend']，
# 强制选择与当前运行的交互式框架兼容的后端。
if (rcParams["backend_fallback"]
        and rcParams._get_backend_or_none() in (  # 指定类型忽略[attr-defined]
            set(backend_registry.list_builtin(BackendFilter.INTERACTIVE)) -
            {'webagg', 'nbagg'})
        and cbook._get_running_interactive_framework()):
    rcParams._set("backend", rcsetup._auto_backend_sentinel)

# fmt: on

################# REMAINING CONTENT GENERATED BY boilerplate.py ##############


# boilerplate.py 自动生成的内容。请勿编辑，否则更改将会丢失。

@_copy_docstring_and_deprecators(Figure.figimage)
def figimage(
    X: ArrayLike,
    xo: int = 0,
    yo: int = 0,
    alpha: float | None = None,
    norm: str | Normalize | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    origin: Literal["upper", "lower"] | None = None,
    resize: bool = False,
    **kwargs,
) -> FigureImage:
    # 调用当前图形的 figimage 方法，用于在图形上绘制图像
    return gcf().figimage(
        X,
        xo=xo,
        yo=yo,
        alpha=alpha,
        norm=norm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        resize=resize,
        **kwargs,
)


# boilerplate.py 自动生成的内容。请勿编辑，否则更改将会丢失。

@_copy_docstring_and_deprecators(Figure.text)
def figtext(
    x: float, y: float, s: str, fontdict: dict[str, Any] | None = None, **kwargs
) -> Text:
    # 调用当前图形的 text 方法，在指定位置添加文本
    return gcf().text(x, y, s, fontdict=fontdict, **kwargs)


# boilerplate.py 自动生成的内容。请勿编辑，否则更改将会丢失。

@_copy_docstring_and_deprecators(Figure.gca)
def gca() -> Axes:
    # 调用当前图形的 gca 方法，获取当前坐标轴对象
    return gcf().gca()


# boilerplate.py 自动生成的内容。请勿编辑，否则更改将会丢失。

@_copy_docstring_and_deprecators(Figure._gci)
def gci() -> ScalarMappable | None:
    # 调用当前图形的 _gci 方法，获取当前颜色映射对象或者 None
    return gcf()._gci()


# boilerplate.py 自动生成的内容。请勿编辑，否则更改将会丢失。

@_copy_docstring_and_deprecators(Figure.ginput)
def ginput(
    n: int = 1,
    timeout: float = 30,
    show_clicks: bool = True,
    mouse_add: MouseButton = MouseButton.LEFT,
    mouse_pop: MouseButton = MouseButton.RIGHT,
    mouse_stop: MouseButton = MouseButton.MIDDLE,
) -> list[tuple[int, int]]:
    # 调用当前图形的 ginput 方法，用于获取用户的鼠标输入
    return gcf().ginput(
        n=n,
        timeout=timeout,
        show_clicks=show_clicks,
        mouse_add=mouse_add,
        mouse_pop=mouse_pop,
        mouse_stop=mouse_stop,
    )


# boilerplate.py 自动生成的内容。请勿编辑，否则更改将会丢失。

@_copy_docstring_and_deprecators(Figure.subplots_adjust)
def subplots_adjust(
    left: float | None = None,
    bottom: float | None = None,
    right: float | None = None,
    top: float | None = None,
    wspace: float | None = None,
    hspace: float | None = None,
) -> None:
    # 调用当前图形的 subplots_adjust 方法，调整子图的布局参数
    pass  # 无返回值
    # 调整当前图形的子图布局参数，包括左边界(left)、底边界(bottom)、右边界(right)、顶边界(top)、水平空白间距(wspace)、垂直空白间距(hspace)
    gcf().subplots_adjust(
        left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace
    )
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用 Figure 对象的 suptitle 方法，添加总标题，并将所有附加参数传递给该方法
@_copy_docstring_and_deprecators(Figure.suptitle)
def suptitle(t: str, **kwargs) -> Text:
    return gcf().suptitle(t, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用 Figure 对象的 tight_layout 方法，调整图表布局，根据提供的参数进行设置
def tight_layout(
    *,
    pad: float = 1.08,
    h_pad: float | None = None,
    w_pad: float | None = None,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    gcf().tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用 Figure 对象的 waitforbuttonpress 方法，等待用户按下按钮或者超时
def waitforbuttonpress(timeout: float = -1) -> None | bool:
    return gcf().waitforbuttonpress(timeout=timeout)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用 Axes 对象的 acorr 方法，绘制自相关图，并将提供的参数传递给该方法
def acorr(
    x: ArrayLike, *, data=None, **kwargs
) -> tuple[np.ndarray, np.ndarray, LineCollection | Line2D, Line2D | None]:
    return gca().acorr(x, **({"data": data} if data is not None else {}), **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用 Axes 对象的 angle_spectrum 方法，绘制角频谱图，并将提供的参数传递给该方法
def angle_spectrum(
    x: ArrayLike,
    Fs: float | None = None,
    Fc: int | None = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    *,
    data=None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, Line2D]:
    return gca().angle_spectrum(
        x,
        Fs=Fs,
        Fc=Fc,
        window=window,
        pad_to=pad_to,
        sides=sides,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用 Axes 对象的 annotate 方法，添加注释到指定坐标点，并将提供的参数传递给该方法
def annotate(
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float] | None = None,
    xycoords: str
    | Artist
    | Transform
    | Callable[[RendererBase], Bbox | Transform]
    | tuple[float, float] = "data",
    textcoords: str
    | Artist
    | Transform
    | Callable[[RendererBase], Bbox | Transform]
    | tuple[float, float]
    | None = None,
    arrowprops: dict[str, Any] | None = None,
    annotation_clip: bool | None = None,
    **kwargs,
) -> Annotation:
    return gca().annotate(
        text,
        xy,
        xytext=xytext,
        xycoords=xycoords,
        textcoords=textcoords,
        arrowprops=arrowprops,
        annotation_clip=annotation_clip,
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用 Axes 对象的 arrow 方法，在指定坐标点之间绘制箭头，并将提供的参数传递给该方法
def arrow(x: float, y: float, dx: float, dy: float, **kwargs) -> FancyArrow:
    return gca().arrow(x, y, dx, dy, **kwargs)
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.autoscale复制文档字符串和过时警告装饰器，并应用于当前函数
@_copy_docstring_and_deprecators(Axes.autoscale)
def autoscale(
    enable: bool = True,
    axis: Literal["both", "x", "y"] = "both",
    tight: bool | None = None,
) -> None:
    # 调用当前图形的gca()方法，自动缩放坐标轴
    gca().autoscale(enable=enable, axis=axis, tight=tight)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.axhline复制文档字符串和过时警告装饰器，并应用于当前函数
@_copy_docstring_and_deprecators(Axes.axhline)
def axhline(y: float = 0, xmin: float = 0, xmax: float = 1, **kwargs) -> Line2D:
    # 在当前图形的坐标系中添加水平线
    return gca().axhline(y=y, xmin=xmin, xmax=xmax, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.axhspan复制文档字符串和过时警告装饰器，并应用于当前函数
@_copy_docstring_and_deprecators(Axes.axhspan)
def axhspan(
    ymin: float, ymax: float, xmin: float = 0, xmax: float = 1, **kwargs
) -> Rectangle:
    # 在当前图形的坐标系中添加水平跨度区域
    return gca().axhspan(ymin, ymax, xmin=xmin, xmax=xmax, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.axis复制文档字符串和过时警告装饰器，并应用于当前函数
@_copy_docstring_and_deprecators(Axes.axis)
def axis(
    arg: tuple[float, float, float, float] | bool | str | None = None,
    /,
    *,
    emit: bool = True,
    **kwargs,
) -> tuple[float, float, float, float]:
    # 获取或设置当前图形的坐标轴属性
    return gca().axis(arg, emit=emit, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.axline复制文档字符串和过时警告装饰器，并应用于当前函数
@_copy_docstring_and_deprecators(Axes.axline)
def axline(
    xy1: tuple[float, float],
    xy2: tuple[float, float] | None = None,
    *,
    slope: float | None = None,
    **kwargs,
) -> AxLine:
    # 在当前图形的坐标系中添加线条
    return gca().axline(xy1, xy2=xy2, slope=slope, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.axvline复制文档字符串和过时警告装饰器，并应用于当前函数
@_copy_docstring_and_deprecators(Axes.axvline)
def axvline(x: float = 0, ymin: float = 0, ymax: float = 1, **kwargs) -> Line2D:
    # 在当前图形的坐标系中添加垂直线
    return gca().axvline(x=x, ymin=ymin, ymax=ymax, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.axvspan复制文档字符串和过时警告装饰器，并应用于当前函数
@_copy_docstring_and_deprecators(Axes.axvspan)
def axvspan(
    xmin: float, xmax: float, ymin: float = 0, ymax: float = 1, **kwargs
) -> Rectangle:
    # 在当前图形的坐标系中添加垂直跨度区域
    return gca().axvspan(xmin, xmax, ymin=ymin, ymax=ymax, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.bar复制文档字符串和过时警告装饰器，并应用于当前函数
@_copy_docstring_and_deprecators(Axes.bar)
def bar(
    x: float | ArrayLike,
    height: float | ArrayLike,
    width: float | ArrayLike = 0.8,
    bottom: float | ArrayLike | None = None,
    *,
    align: Literal["center", "edge"] = "center",
    data=None,
    **kwargs,
) -> BarContainer:
    # 在当前图形的坐标系中添加条形图
    return gca().bar(
        x,
        height,
        width=width,
        bottom=bottom,
        align=align,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.barbs复制文档字符串和过时警告装饰器，并应用于当前函数
@_copy_docstring_and_deprecators(Axes.barbs)
def barbs(*args, data=None, **kwargs) -> Barbs:
    # 在当前图形的坐标系中添加箭头图
    return gca().barbs(*args, **({"data": data} if data is not None else {}), **kwargs)
# 使用 _copy_docstring_and_deprecators 装饰器，将 barh 函数的文档字符串和废弃警告拷贝到当前函数
@_copy_docstring_and_deprecators(Axes.barh)
def barh(
    # y 参数可以是浮点数或类数组对象，表示条形图的 y 坐标
    y: float | ArrayLike,
    # width 参数可以是浮点数或类数组对象，表示条形图的宽度
    width: float | ArrayLike,
    # height 参数可以是浮点数或类数组对象，表示条形图的高度，默认为 0.8
    height: float | ArrayLike = 0.8,
    # left 参数可以是浮点数、类数组对象或 None，表示条形图的左侧位置
    left: float | ArrayLike | None = None,
    # align 参数必须为 "center" 或 "edge"，表示条形图的对齐方式
    *,
    align: Literal["center", "edge"] = "center",
    # data 参数默认为 None，表示附加数据
    data=None,
    **kwargs,  # 其他关键字参数
) -> BarContainer:  # 函数返回一个 BarContainer 对象
    # 调用 gca() 获取当前轴对象，然后调用其 barh 方法绘制水平条形图
    return gca().barh(
        y,
        width,
        height=height,
        left=left,
        align=align,
        **({"data": data} if data is not None else {}),  # 如果 data 不为 None，则传递 data 参数
        **kwargs,  # 传递其他关键字参数
    )


# 由 boilerplate.py 自动生成。编辑将导致更改丢失。
# 使用 _copy_docstring_and_deprecators 装饰器，将 bar_label 函数的文档字符串和废弃警告拷贝到当前函数
@_copy_docstring_and_deprecators(Axes.bar_label)
def bar_label(
    # container 参数为 BarContainer 对象，表示条形图容器
    container: BarContainer,
    # labels 参数可以是类数组对象或 None，表示条形图标签
    labels: ArrayLike | None = None,
    *,
    # fmt 参数可以是字符串或格式化函数，用于标签的格式化
    fmt: str | Callable[[float], str] = "%g",
    # label_type 参数必须为 "center" 或 "edge"，表示标签的位置
    label_type: Literal["center", "edge"] = "edge",
    # padding 参数为浮点数，表示标签与条形图之间的间距
    padding: float = 0,
    **kwargs,  # 其他关键字参数
) -> list[Annotation]:  # 函数返回 Annotation 对象的列表
    # 调用 gca() 获取当前轴对象，然后调用其 bar_label 方法添加条形图标签
    return gca().bar_label(
        container,
        labels=labels,
        fmt=fmt,
        label_type=label_type,
        padding=padding,
        **kwargs,  # 传递其他关键字参数
    )


# 由 boilerplate.py 自动生成。编辑将导致更改丢失。
# 使用 _copy_docstring_and_deprecators 装饰器，将 boxplot 函数的文档字符串和废弃警告拷贝到当前函数
@_copy_docstring_and_deprecators(Axes.boxplot)
def boxplot(
    # x 参数可以是类数组对象或类数组对象的序列，表示箱线图的数据
    x: ArrayLike | Sequence[ArrayLike],
    # notch 参数为布尔值或 None，表示箱线图是否使用凹口
    notch: bool | None = None,
    # sym 参数为字符串或 None，表示箱线图异常值的符号
    sym: str | None = None,
    # vert 参数为布尔值或 None，表示箱线图的方向
    vert: bool | None = None,
    # orientation 参数必须为 "vertical" 或 "horizontal"，表示箱线图的方向
    orientation: Literal["vertical", "horizontal"] = "vertical",
    # whis 参数可以是浮点数、浮点数元组或 None，表示箱线图的触须范围
    whis: float | tuple[float, float] | None = None,
    # positions 参数可以是类数组对象或 None，表示箱线图的位置
    positions: ArrayLike | None = None,
    # widths 参数可以是浮点数、类数组对象或 None，表示箱线图箱体的宽度
    widths: float | ArrayLike | None = None,
    # patch_artist 参数为布尔值或 None，表示是否使用填充箱线图
    patch_artist: bool | None = None,
    # bootstrap 参数为整数或 None，表示计算箱线图置信区间时使用的 bootstrap 方法
    bootstrap: int | None = None,
    # usermedians 参数可以是类数组对象或 None，表示箱线图的用户指定中位数
    usermedians: ArrayLike | None = None,
    # conf_intervals 参数可以是类数组对象或 None，表示箱线图的置信区间
    conf_intervals: ArrayLike | None = None,
    # meanline 参数为布尔值或 None，表示是否绘制箱线图的均值线
    meanline: bool | None = None,
    # showmeans 参数为布尔值或 None，表示是否显示箱线图的均值点
    showmeans: bool | None = None,
    # showcaps 参数为布尔值或 None，表示是否显示箱线图的触顶
    showcaps: bool | None = None,
    # showbox 参数为布尔值或 None，表示是否显示箱线图的箱体
    showbox: bool | None = None,
    # showfliers 参数为布尔值或 None，表示是否显示箱线图的异常值
    showfliers: bool | None = None,
    # boxprops 参数为字典对象或 None，表示箱线图的箱体属性
    boxprops: dict[str, Any] | None = None,
    # tick_labels 参数为字符串序列或 None，表示箱线图的刻度标签
    tick_labels: Sequence[str] | None = None,
    # flierprops 参数为字典对象或 None，表示箱线图的异常值属性
    flierprops: dict[str, Any] | None = None,
    # medianprops 参数为字典对象或 None，表示箱线图的中位数属性
    medianprops: dict[str, Any] | None = None,
    # meanprops 参数为字典对象或 None，表示箱线图的均值属性
    meanprops: dict[str, Any] | None = None,
    # capprops 参数为字典对象或 None，表示箱线图的触顶属性
    capprops: dict[str, Any] | None = None,
    # whiskerprops 参数为字典对象或 None，表示箱线图的触须属性
    whiskerprops: dict[str, Any] | None = None,
    # manage_ticks 参数为布尔值，表示是否管理箱线图的刻度
    manage_ticks: bool = True,
    # autorange 参数为布尔值，表示是否自动调整箱线图的范围
    autorange: bool = False,
    # zorder 参数可以是浮点数或 None，表示箱线图的 z 轴顺序
    zorder: float | None = None,
    # capwidths 参数可以是浮点数、类数组对象或 None，表示箱线图的触顶宽度
    capwidths: float | ArrayLike | None = None,
    # label 参数为字符串序列或 None，表示箱线图的标签
    label: Sequence[str] | None = None,
    *,
    data=None,  # data 参数默认为 None，表示附加数据
) -> dict[str, Any]:  # 函数返回一个包含各种箱线图属性的字典
    # 调用当前图形的 gca() 方法获取当前轴对象，然后调用其 boxplot 方法绘制箱线图
    return gca().boxplot(
        x,  # x 轴上的数据，通常是要绘制的数据集
        notch=notch,  # 是否绘制缺口箱线图（notched box plot），默认为 False
        sym=sym,  # 离群值的表示方式，可以是字符串（如 '+'）或者列表（每个数据集对应一个符号）
        vert=vert,  # 箱线图的方向，True 表示垂直箱线图，False 表示水平箱线图
        orientation=orientation,  # 箱线图的方向，'vertical' 或 'horizontal'
        whis=whis,  # 确定箱须长度的因子
        positions=positions,  # 箱线图的位置
        widths=widths,  # 箱线图的宽度
        patch_artist=patch_artist,  # 是否给箱体上色，默认为 False
        bootstrap=bootstrap,  # 是否使用 bootstrap 方法计算置信区间，默认为 None 或 False
        usermedians=usermedians,  # 自定义中位数的位置
        conf_intervals=conf_intervals,  # 自定义置信区间
        meanline=meanline,  # 是否显示均值线
        showmeans=showmeans,  # 是否显示均值点
        showcaps=showcaps,  # 是否显示箱线图的边缘线
        showbox=showbox,  # 是否显示箱体
        showfliers=showfliers,  # 是否显示离群值
        boxprops=boxprops,  # 箱体的属性设置
        tick_labels=tick_labels,  # 箱线图的刻度标签
        flierprops=flierprops,  # 离群值的属性设置
        medianprops=medianprops,  # 中位数线的属性设置
        meanprops=meanprops,  # 均值线的属性设置
        capprops=capprops,  # 箱线图边缘线的属性设置
        whiskerprops=whiskerprops,  # 箱须的属性设置
        manage_ticks=manage_ticks,  # 是否管理刻度
        autorange=autorange,  # 是否自动调整轴范围
        zorder=zorder,  # 图层顺序
        capwidths=capwidths,  # 箱线图边缘线的宽度
        label=label,  # 箱线图的标签
        **({"data": data} if data is not None else {}),  # 传递额外的参数，如数据集（如果提供了的话）
    )
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.broken_barh复制文档字符串和过时功能
def broken_barh(
    xranges: Sequence[tuple[float, float]],
    yrange: tuple[float, float],
    *,
    data=None,
    **kwargs,
) -> PolyCollection:
    # 调用当前的坐标轴对象，并调用其broken_barh方法，传递参数和关键字参数
    return gca().broken_barh(
        xranges, yrange, **({"data": data} if data is not None else {}), **kwargs
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.clabel复制文档字符串和过时功能
def clabel(CS: ContourSet, levels: ArrayLike | None = None, **kwargs) -> list[Text]:
    # 调用当前的坐标轴对象，并调用其clabel方法，传递参数和关键字参数
    return gca().clabel(CS, levels=levels, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.cohere复制文档字符串和过时功能
def cohere(
    x: ArrayLike,
    y: ArrayLike,
    NFFT: int = 256,
    Fs: float = 2,
    Fc: int = 0,
    detrend: Literal["none", "mean", "linear"]
    | Callable[[ArrayLike], ArrayLike] = mlab.detrend_none,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike = mlab.window_hanning,
    noverlap: int = 0,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] = "default",
    scale_by_freq: bool | None = None,
    *,
    data=None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    # 调用当前的坐标轴对象，并调用其cohere方法，传递参数和关键字参数
    return gca().cohere(
        x,
        y,
        NFFT=NFFT,
        Fs=Fs,
        Fc=Fc,
        detrend=detrend,
        window=window,
        noverlap=noverlap,
        pad_to=pad_to,
        sides=sides,
        scale_by_freq=scale_by_freq,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.contour复制文档字符串和过时功能
def contour(*args, data=None, **kwargs) -> QuadContourSet:
    # 调用当前的坐标轴对象，并调用其contour方法，传递参数和关键字参数
    __ret = gca().contour(
        *args, **({"data": data} if data is not None else {}), **kwargs
    )
    # 如果返回对象包含数据属性，则调用sci函数
    if __ret._A is not None:  # type: ignore[attr-defined]
        sci(__ret)
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.contourf复制文档字符串和过时功能
def contourf(*args, data=None, **kwargs) -> QuadContourSet:
    # 调用当前的坐标轴对象，并调用其contourf方法，传递参数和关键字参数
    __ret = gca().contourf(
        *args, **({"data": data} if data is not None else {}), **kwargs
    )
    # 如果返回对象包含数据属性，则调用sci函数
    if __ret._A is not None:  # type: ignore[attr-defined]
        sci(__ret)
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes.csd复制文档字符串和过时功能
def csd(
    x: ArrayLike,
    y: ArrayLike,
    NFFT: int | None = None,
    Fs: float | None = None,
    Fc: int | None = None,
    detrend: Literal["none", "mean", "linear"]
    | Callable[[ArrayLike], ArrayLike]
    | None = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    noverlap: int | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    scale_by_freq: bool | None = None,
    *,
    data=None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    # 调用当前的坐标轴对象，并调用其csd方法，传递参数和关键字参数
    return gca().csd(
        x,
        y,
        NFFT=NFFT,
        Fs=Fs,
        Fc=Fc,
        detrend=detrend,
        window=window,
        noverlap=noverlap,
        pad_to=pad_to,
        sides=sides,
        scale_by_freq=scale_by_freq,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    # 声明一个变量 return_line，其类型可以是 bool 或者 None
    return_line: bool | None = None,
    # '*' 表示之后的参数必须以关键字形式传入
    # 声明一个参数 data，默认值为 None
    *,
    # **kwargs 表示接受任意数量的关键字参数，将它们存储在一个字典中
    data=None,
    **kwargs,
# 返回当前图形的坐标轴对象，并调用其 `csd` 方法计算交叉谱密度（CSD）
def csd(
    x: ArrayLike,
    y: ArrayLike,
    NFFT: int = 256,
    Fs: float = 2,
    Fc: float = 0,
    detrend: Union[bool, Callable[[np.ndarray], np.ndarray]] = False,
    window: Union[str, Tuple[float, ...], np.ndarray] = 'hann',
    noverlap: int = 128,
    pad_to: Optional[int] = None,
    sides: str = 'default',
    scale_by_freq: bool = True,
    return_line: bool = False,
    **({"data": data} if data is not None else {}),
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Line2D]:
    return gca().csd(
        x,
        y,
        NFFT=NFFT,
        Fs=Fs,
        Fc=Fc,
        detrend=detrend,
        window=window,
        noverlap=noverlap,
        pad_to=pad_to,
        sides=sides,
        scale_by_freq=scale_by_freq,
        return_line=return_line,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# 由 boilerplate.py 自动生成。不要编辑，否则更改将会丢失。
# 返回当前图形的坐标轴对象，并调用其 `ecdf` 方法生成经验累积分布函数（ECDF）
@_copy_docstring_and_deprecators(Axes.ecdf)
def ecdf(
    x: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    complementary: bool = False,
    orientation: Literal["vertical", "horizonatal"] = "vertical",
    compress: bool = False,
    data=None,
    **kwargs,
) -> Line2D:
    return gca().ecdf(
        x,
        weights=weights,
        complementary=complementary,
        orientation=orientation,
        compress=compress,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# 由 boilerplate.py 自动生成。不要编辑，否则更改将会丢失。
# 返回当前图形的坐标轴对象，并调用其 `errorbar` 方法绘制误差线图
@_copy_docstring_and_deprecators(Axes.errorbar)
def errorbar(
    x: float | ArrayLike,
    y: float | ArrayLike,
    yerr: float | ArrayLike | None = None,
    xerr: float | ArrayLike | None = None,
    fmt: str = "",
    ecolor: ColorType | None = None,
    elinewidth: float | None = None,
    capsize: float | None = None,
    barsabove: bool = False,
    lolims: bool | ArrayLike = False,
    uplims: bool | ArrayLike = False,
    xlolims: bool | ArrayLike = False,
    xuplims: bool | ArrayLike = False,
    errorevery: int | tuple[int, int] = 1,
    capthick: float | None = None,
    *,
    data=None,
    **kwargs,
) -> ErrorbarContainer:
    return gca().errorbar(
        x,
        y,
        yerr=yerr,
        xerr=xerr,
        fmt=fmt,
        ecolor=ecolor,
        elinewidth=elinewidth,
        capsize=capsize,
        barsabove=barsabove,
        lolims=lolims,
        uplims=uplims,
        xlolims=xlolims,
        xuplims=xuplims,
        errorevery=errorevery,
        capthick=capthick,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# 由 boilerplate.py 自动生成。不要编辑，否则更改将会丢失。
# 返回当前图形的坐标轴对象，并调用其 `eventplot` 方法绘制事件图
@_copy_docstring_and_deprecators(Axes.eventplot)
def eventplot(
    positions: ArrayLike | Sequence[ArrayLike],
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    lineoffsets: float | Sequence[float] = 1,
    linelengths: float | Sequence[float] = 1,
    linewidths: float | Sequence[float] | None = None,
    colors: ColorType | Sequence[ColorType] | None = None,
    alpha: float | Sequence[float] | None = None,
    linestyles: LineStyleType | Sequence[LineStyleType] = "solid",
    *,
    data=None,
    **kwargs,
) -> EventCollection:
    return gca().eventplot(
        positions,
        orientation=orientation,
        lineoffsets=lineoffsets,
        linelengths=linelengths,
        linewidths=linewidths,
        colors=colors,
        alpha=alpha,
        linestyles=linestyles,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    # 调用 gca() 获取当前图形的坐标轴对象，并在其上绘制事件图
    return gca().eventplot(
        positions,                  # 事件的位置坐标，可以是一个列表或数组
        orientation=orientation,    # 事件图的方向，可以是 'horizontal' 或 'vertical'
        lineoffsets=lineoffsets,    # 每个事件行的偏移量，可以是一个列表或数组
        linelengths=linelengths,    # 每个事件行的长度，可以是一个标量或数组
        linewidths=linewidths,      # 每条事件线的宽度，可以是一个标量或数组
        colors=colors,              # 每条事件线的颜色，可以是一个颜色字符串或颜色序列
        alpha=alpha,                # 事件线的透明度，0（完全透明）到1（完全不透明）
        linestyles=linestyles,      # 每条事件线的线型，可以是一个线型字符串或线型序列
        **({"data": data} if data is not None else {}),  # 如果提供了数据，则作为关键字参数传递
        **kwargs,                   # 其它可能的关键字参数，传递给 eventplot 函数
    )
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 根据 Axes.fill 方法生成的装饰器，用于填充多边形区域
@_copy_docstring_and_deprecators(Axes.fill)
def fill(*args, data=None, **kwargs) -> list[Polygon]:
    # 获取当前图表的坐标轴对象，并调用其 fill 方法进行填充操作
    return gca().fill(*args, **({"data": data} if data is not None else {}), **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 根据 Axes.fill_between 方法生成的装饰器，用于在两个曲线之间填充区域
@_copy_docstring_and_deprecators(Axes.fill_between)
def fill_between(
    x: ArrayLike,
    y1: ArrayLike | float,
    y2: ArrayLike | float = 0,
    where: Sequence[bool] | None = None,
    interpolate: bool = False,
    step: Literal["pre", "post", "mid"] | None = None,
    *,
    data=None,
    **kwargs,
) -> PolyCollection:
    # 获取当前图表的坐标轴对象，并调用其 fill_between 方法进行填充区域操作
    return gca().fill_between(
        x,
        y1,
        y2=y2,
        where=where,
        interpolate=interpolate,
        step=step,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 根据 Axes.fill_betweenx 方法生成的装饰器，用于在两个曲线之间填充水平区域
@_copy_docstring_and_deprecators(Axes.fill_betweenx)
def fill_betweenx(
    y: ArrayLike,
    x1: ArrayLike | float,
    x2: ArrayLike | float = 0,
    where: Sequence[bool] | None = None,
    step: Literal["pre", "post", "mid"] | None = None,
    interpolate: bool = False,
    *,
    data=None,
    **kwargs,
) -> PolyCollection:
    # 获取当前图表的坐标轴对象，并调用其 fill_betweenx 方法进行填充水平区域操作
    return gca().fill_betweenx(
        y,
        x1,
        x2=x2,
        where=where,
        step=step,
        interpolate=interpolate,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 根据 Axes.grid 方法生成的装饰器，用于显示坐标轴网格线
@_copy_docstring_and_deprecators(Axes.grid)
def grid(
    visible: bool | None = None,
    which: Literal["major", "minor", "both"] = "major",
    axis: Literal["both", "x", "y"] = "both",
    **kwargs,
) -> None:
    # 获取当前图表的坐标轴对象，并调用其 grid 方法进行网格线显示设置
    gca().grid(visible=visible, which=which, axis=axis, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 根据 Axes.hexbin 方法生成的装饰器，用于绘制六边形网格图
@_copy_docstring_and_deprecators(Axes.hexbin)
def hexbin(
    x: ArrayLike,
    y: ArrayLike,
    C: ArrayLike | None = None,
    gridsize: int | tuple[int, int] = 100,
    bins: Literal["log"] | int | Sequence[float] | None = None,
    xscale: Literal["linear", "log"] = "linear",
    yscale: Literal["linear", "log"] = "linear",
    extent: tuple[float, float, float, float] | None = None,
    cmap: str | Colormap | None = None,
    norm: str | Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float | None = None,
    linewidths: float | None = None,
    edgecolors: Literal["face", "none"] | ColorType = "face",
    reduce_C_function: Callable[[np.ndarray | list[float]], float] = np.mean,
    mincnt: int | None = None,
    marginals: bool = False,
    *,
    data=None,
    **kwargs,
) -> PolyCollection:
    # 获取当前图表的坐标轴对象，并调用其 hexbin 方法进行六边形网格图绘制
    return gca().hexbin(
        x,
        y,
        C=C,
        gridsize=gridsize,
        bins=bins,
        xscale=xscale,
        yscale=yscale,
        extent=extent,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
        reduce_C_function=reduce_C_function,
        mincnt=mincnt,
        marginals=marginals,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    # 调用当前图形的坐标轴对象，并在其上创建二维 hexbin 图
    __ret = gca().hexbin(
        x,
        y,
        C=C,
        gridsize=gridsize,
        bins=bins,
        xscale=xscale,
        yscale=yscale,
        extent=extent,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
        reduce_C_function=reduce_C_function,
        mincnt=mincnt,
        marginals=marginals,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    # 在当前坐标轴对象上绘制科学记数法的图形
    sci(__ret)
    # 返回 hexbin 函数的结果，即绘制的图形对象
    return __ret
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 通过装饰器复制 Axes.hist 的文档字符串和过时警告
@_copy_docstring_and_deprecators(Axes.hist)
# 绘制直方图的函数，返回数据数组、边界数组和条形容器或多边形对象列表的元组
def hist(
    x: ArrayLike | Sequence[ArrayLike],  # x 数据可以是数组或数组序列
    bins: int | Sequence[float] | str | None = None,  # 直方图的箱数或边界值序列或字符串，可选，默认为 None
    range: tuple[float, float] | None = None,  # 数据的范围或 None
    density: bool = False,  # 是否绘制密度直方图，默认为 False
    weights: ArrayLike | None = None,  # 数据权重，默认为 None
    cumulative: bool | float = False,  # 是否绘制累积直方图，默认为 False
    bottom: ArrayLike | float | None = None,  # 条形底部位置，默认为 None
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "bar",  # 直方图类型，默认为 "bar"
    align: Literal["left", "mid", "right"] = "mid",  # 条形对齐方式，默认为 "mid"
    orientation: Literal["vertical", "horizontal"] = "vertical",  # 绘制方向，默认为 "vertical"
    rwidth: float | None = None,  # 条形宽度比例或 None，默认为 None
    log: bool = False,  # 是否使用对数坐标，默认为 False
    color: ColorType | Sequence[ColorType] | None = None,  # 条形颜色或颜色序列，默认为 None
    label: str | Sequence[str] | None = None,  # 条形标签或标签序列，默认为 None
    stacked: bool = False,  # 是否堆叠条形，默认为 False
    *,
    data=None,  # 数据参数，可选，默认为 None
    **kwargs,  # 其他关键字参数
) -> tuple[
    np.ndarray | list[np.ndarray],  # 数据数组或数组列表
    np.ndarray,  # 边界数组
    BarContainer | Polygon | list[BarContainer | Polygon],  # 条形容器或多边形对象列表
]:
    return gca().hist(
        x,
        bins=bins,
        range=range,
        density=density,
        weights=weights,
        cumulative=cumulative,
        bottom=bottom,
        histtype=histtype,
        align=align,
        orientation=orientation,
        rwidth=rwidth,
        log=log,
        color=color,
        label=label,
        stacked=stacked,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 通过装饰器复制 Axes.stairs 的文档字符串和过时警告
@_copy_docstring_and_deprecators(Axes.stairs)
# 绘制阶梯图的函数，返回阶梯图对象 StepPatch
def stairs(
    values: ArrayLike,  # 数据值数组
    edges: ArrayLike | None = None,  # 边缘数组或 None，默认为 None
    *,
    orientation: Literal["vertical", "horizontal"] = "vertical",  # 绘制方向，默认为 "vertical"
    baseline: float | ArrayLike | None = 0,  # 基线位置，默认为 0 或 None
    fill: bool = False,  # 是否填充阶梯图，默认为 False
    data=None,  # 数据参数，可选，默认为 None
    **kwargs,  # 其他关键字参数
) -> StepPatch:  # 返回 StepPatch 对象
    return gca().stairs(
        values,
        edges=edges,
        orientation=orientation,
        baseline=baseline,
        fill=fill,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 通过装饰器复制 Axes.hist2d 的文档字符串和过时警告
@_copy_docstring_and_deprecators(Axes.hist2d)
# 绘制二维直方图的函数，返回数据数组、x 边界数组、y 边界数组和 QuadMesh 对象的元组
def hist2d(
    x: ArrayLike,  # x 数据数组
    y: ArrayLike,  # y 数据数组
    bins: None | int | tuple[int, int] | ArrayLike | tuple[ArrayLike, ArrayLike] = 10,  # 箱数或边界数组或箱数组元组，默认为 10
    range: ArrayLike | None = None,  # 数据范围或 None
    density: bool = False,  # 是否绘制密度直方图，默认为 False
    weights: ArrayLike | None = None,  # 数据权重，默认为 None
    cmin: float | None = None,  # 最小颜色映射值或 None
    cmax: float | None = None,  # 最大颜色映射值或 None
    *,
    data=None,  # 数据参数，可选，默认为 None
    **kwargs,  # 其他关键字参数
) -> tuple[np.ndarray, np.ndarray, np.ndarray, QuadMesh]:  # 返回数据数组、x 边界数组、y 边界数组和 QuadMesh 对象的元组
    __ret = gca().hist2d(
        x,
        y,
        bins=bins,
        range=range,
        density=density,
        weights=weights,
        cmin=cmin,
        cmax=cmax,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    sci(__ret[-1])  # 对 QuadMesh 对象进行科学记数法处理
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 通过装饰器复制 Axes.hlines 的文档字符串和过时警告
@_copy_docstring_and_deprecators(Axes.hlines)
# 绘制水平线的函数，返回 StepPatch 对象
def hlines(
    y: float | ArrayLike,  # y 坐标或坐标数组
    xmin: float | ArrayLike,  # 最小 x 坐标或坐标数组
    xmax: float | ArrayLike,
    # xmax 是一个参数，可以是 float 类型或者类似数组的数据类型（ArrayLike）

    colors: ColorType | Sequence[ColorType] | None = None,
    # colors 是一个参数，可以是 ColorType 类型，或者 ColorType 类型的序列，或者 None，默认为 None

    linestyles: LineStyleType = "solid",
    # linestyles 是一个参数，可以是 LineStyleType 类型，默认为 "solid"

    label: str = "",
    # label 是一个参数，是一个字符串类型，默认为空字符串

    *,
    # * 后面的参数是命名关键字参数（keyword-only arguments），需要以关键字形式传入

    data=None,
    # data 是一个参数，默认为 None

    **kwargs,
    # **kwargs 是接受所有额外的关键字参数，并将它们存储在 kwargs 字典中
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.

# 使用当前的坐标轴对象（gca()）绘制水平线集合，并返回 LineCollection 对象
def hlines(
    y,  # 水平线的 y 坐标或坐标数组
    xmin,  # 水平线的起始 x 坐标
    xmax,  # 水平线的结束 x 坐标
    colors=None,  # 水平线的颜色
    linestyles='solid',  # 水平线的线型
    label=None,  # 水平线的标签
    **({"data": data} if data is not None else {}),  # 如果提供了 data 参数，则传递给函数
    **kwargs,  # 其他可选参数
) -> LineCollection:  # 返回一个 LineCollection 对象
    return gca().hlines(
        y,
        xmin,
        xmax,
        colors=colors,
        linestyles=linestyles,
        label=label,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 根据提供的参数绘制图像，并返回 AxesImage 对象
@_copy_docstring_and_deprecators(Axes.imshow)
def imshow(
    X: ArrayLike | PIL.Image.Image,  # 要显示的图像数据或 PIL 图像对象
    cmap: str | Colormap | None = None,  # 颜色映射或 None（使用默认映射）
    norm: str | Normalize | None = None,  # 归一化方式或 None（使用默认方式）
    *,
    aspect: Literal["equal", "auto"] | float | None = None,  # 图像显示的纵横比
    interpolation: str | None = None,  # 插值方式或 None（使用默认插值）
    alpha: float | ArrayLike | None = None,  # 图像的透明度
    vmin: float | None = None,  # 最小值限制或 None（自动确定）
    vmax: float | None = None,  # 最大值限制或 None（自动确定）
    origin: Literal["upper", "lower"] | None = None,  # 坐标原点的位置
    extent: tuple[float, float, float, float] | None = None,  # 图像的坐标范围
    interpolation_stage: Literal["data", "rgba"] | None = None,  # 插值阶段
    filternorm: bool = True,  # 是否应用标准化过滤器
    filterrad: float = 4.0,  # 过滤器的半径
    resample: bool | None = None,  # 是否重新采样图像
    url: str | None = None,  # 图像相关的 URL
    data=None,  # 额外的数据参数
    **kwargs,  # 其他可选参数
) -> AxesImage:  # 返回一个 AxesImage 对象
    __ret = gca().imshow(
        X,
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        interpolation=interpolation,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        extent=extent,
        interpolation_stage=interpolation_stage,
        filternorm=filternorm,
        filterrad=filterrad,
        resample=resample,
        url=url,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    sci(__ret)  # 在图像上设置科学计数法标签
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 添加图例到当前坐标轴对象，并返回 Legend 对象
@_copy_docstring_and_deprecators(Axes.legend)
def legend(*args, **kwargs) -> Legend:  # 接受任意参数，返回一个 Legend 对象
    return gca().legend(*args, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 配置坐标轴的定位器参数
@_copy_docstring_and_deprecators(Axes.locator_params)
def locator_params(
    axis: Literal["both", "x", "y"] = "both",  # 指定定位器参数应用的轴
    tight: bool | None = None,  # 是否启用紧凑布局
    **kwargs,  # 其他可选参数
) -> None:  # 无返回值
    gca().locator_params(axis=axis, tight=tight, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 绘制双对数坐标图，并返回 Line2D 对象列表
@_copy_docstring_and_deprecators(Axes.loglog)
def loglog(*args, **kwargs) -> list[Line2D]:  # 接受任意参数，返回 Line2D 对象列表
    return gca().loglog(*args, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 计算信号的幅度谱，并返回幅度谱、频率和 Line2D 对象
@_copy_docstring_and_deprecators(Axes.magnitude_spectrum)
def magnitude_spectrum(
    x: ArrayLike,  # 输入信号数据
    Fs: float | None = None,  # 信号的采样率或 None（自动确定）
    Fc: int | None = None,  # 信号的截止频率或 None（自动确定）
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,  # 窗函数或窗函数数组或 None（不使用窗函数）
    pad_to: int | None = None,  # 扩展到的长度或 None（不扩展）
    sides: Literal["default", "onesided", "twosided"] | None = None,  # 频谱的边界条件
    scale: Literal["default", "linear", "dB"] | None = None,  # 幅度谱的显示方式
    *,
    data=None,  # 额外的数据参数
    **kwargs,  # 其他可选参数
) -> tuple[np.ndarray, np.ndarray, Line2D]:  # 返回幅度谱、频率数组和 Line2D 对象
    # 调用 gca() 函数获取当前坐标轴对象，并计算其幅度谱
    return gca().magnitude_spectrum(
        x,                  # 输入信号的数据
        Fs=Fs,              # 采样频率
        Fc=Fc,              # 截止频率
        window=window,      # 窗函数类型
        pad_to=pad_to,      # 补零长度
        sides=sides,        # 输出谱的类型（双边或单边）
        scale=scale,        # 谱的缩放方式
        **({"data": data} if data is not None else {}),  # 如果有额外的数据参数，则传递进去
        **kwargs,           # 其他的关键字参数
    )
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes类中复制文档字符串和过时警告，并应用于margins函数
def margins(
    *margins: float,
    x: float | None = None,
    y: float | None = None,
    tight: bool | None = True,
) -> tuple[float, float] | None:
    # 调用当前坐标轴的margins方法，并将参数传递进去
    return gca().margins(*margins, x=x, y=y, tight=tight)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes类中复制文档字符串和过时警告，并应用于minorticks_off函数
def minorticks_off() -> None:
    # 调用当前坐标轴的minorticks_off方法，关闭次刻度
    gca().minorticks_off()


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes类中复制文档字符串和过时警告，并应用于minorticks_on函数
def minorticks_on() -> None:
    # 调用当前坐标轴的minorticks_on方法，打开次刻度
    gca().minorticks_on()


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes类中复制文档字符串和过时警告，并应用于pcolor函数
def pcolor(
    *args: ArrayLike,
    shading: Literal["flat", "nearest", "auto"] | None = None,
    alpha: float | None = None,
    norm: str | Normalize | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    data=None,
    **kwargs,
) -> Collection:
    # 调用当前坐标轴的pcolor方法，并传递相应参数
    __ret = gca().pcolor(
        *args,
        shading=shading,
        alpha=alpha,
        norm=norm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    # 对返回的对象应用科学计数法标注
    sci(__ret)
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes类中复制文档字符串和过时警告，并应用于pcolormesh函数
def pcolormesh(
    *args: ArrayLike,
    alpha: float | None = None,
    norm: str | Normalize | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    shading: Literal["flat", "nearest", "gouraud", "auto"] | None = None,
    antialiased: bool = False,
    data=None,
    **kwargs,
) -> QuadMesh:
    # 调用当前坐标轴的pcolormesh方法，并传递相应参数
    __ret = gca().pcolormesh(
        *args,
        alpha=alpha,
        norm=norm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading=shading,
        antialiased=antialiased,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    # 对返回的对象应用科学计数法标注
    sci(__ret)
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 从Axes类中复制文档字符串和过时警告，并应用于phase_spectrum函数
def phase_spectrum(
    x: ArrayLike,
    Fs: float | None = None,
    Fc: int | None = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    *,
    data=None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, Line2D]:
    # 调用当前坐标轴的phase_spectrum方法，并传递相应参数
    return gca().phase_spectrum(
        x,
        Fs=Fs,
        Fc=Fc,
        window=window,
        pad_to=pad_to,
        sides=sides,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.

# 使用 _copy_docstring_and_deprecators 装饰器，将 Axes.pie 方法的文档字符串和废弃警告复制到当前函数
@_copy_docstring_and_deprecators(Axes.pie)
def pie(
    # 接受数组类型的输入参数 x，表示饼图的数据
    x: ArrayLike,
    # 可选参数，控制各扇区是否分离的数组或者 None
    explode: ArrayLike | None = None,
    # 可选参数，指定各扇区的标签序列或者 None
    labels: Sequence[str] | None = None,
    # 可选参数，指定扇区的颜色，可以是颜色类型、颜色序列或者 None
    colors: ColorType | Sequence[ColorType] | None = None,
    # 可选参数，指定自动标注百分比的格式字符串或者回调函数
    autopct: str | Callable[[float], str] | None = None,
    # 可选参数，指定百分比标签距离圆心的距离
    pctdistance: float = 0.6,
    # 可选参数，指定是否显示阴影效果
    shadow: bool = False,
    # 可选参数，指定标签距离扇区的距离倍率或者 None
    labeldistance: float | None = 1.1,
    # 可选参数，指定起始角度，默认为 0 度（从正 x 轴开始逆时针）
    startangle: float = 0,
    # 可选参数，指定饼图半径，默认为 1（单位圆）
    radius: float = 1,
    # 可选参数，指定扇区绘制方向，True 表示逆时针，False 表示顺时针
    counterclock: bool = True,
    # 可选参数，指定扇形属性的字典，控制扇区的绘制属性
    wedgeprops: dict[str, Any] | None = None,
    # 可选参数，指定文本属性的字典，控制标签文本的属性
    textprops: dict[str, Any] | None = None,
    # 可选参数，指定饼图的中心位置，默认为 (0, 0)
    center: tuple[float, float] = (0, 0),
    # 可选参数，指定是否显示饼图的框架
    frame: bool = False,
    # 可选参数，指定是否旋转标签以避免重叠
    rotatelabels: bool = False,
    # 命名关键字参数，指定是否归一化数据，默认为 True
    *,
    normalize: bool = True,
    # 可选参数，指定用于填充扇区的图案字符串或者字符串序列
    hatch: str | Sequence[str] | None = None,
    # 可选参数，传递额外数据，如果不为 None 则作为关键字参数传递给绘图方法
    data=None,
) -> tuple[list[Wedge], list[Text]] | tuple[list[Wedge], list[Text], list[Text]]:
    # 调用当前坐标轴对象的 pie 方法，绘制饼图，并返回绘制结果
    return gca().pie(
        x,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=autopct,
        pctdistance=pctdistance,
        shadow=shadow,
        labeldistance=labeldistance,
        startangle=startangle,
        radius=radius,
        counterclock=counterclock,
        wedgeprops=wedgeprops,
        textprops=textprops,
        center=center,
        frame=frame,
        rotatelabels=rotatelabels,
        normalize=normalize,
        hatch=hatch,
        **({"data": data} if data is not None else {}),
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.

# 使用 _copy_docstring_and_deprecators 装饰器，将 Axes.plot 方法的文档字符串和废弃警告复制到当前函数
@_copy_docstring_and_deprecators(Axes.plot)
def plot(
    # 可变位置参数，接受浮点数、数组类型或字符串
    *args: float | ArrayLike | str,
    # 可选参数，控制 x 轴的缩放
    scalex: bool = True,
    # 可选参数，控制 y 轴的缩放
    scaley: bool = True,
    # 可选参数，传递额外数据，如果不为 None 则作为关键字参数传递给绘图方法
    data=None,
    # 关键字参数，传递给底层绘图方法的其余参数
    **kwargs,
) -> list[Line2D]:
    # 调用当前坐标轴对象的 plot 方法，绘制折线图，并返回绘制结果
    return gca().plot(
        *args,
        scalex=scalex,
        scaley=scaley,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.

# 使用 _copy_docstring_and_deprecators 装饰器，将 Axes.plot_date 方法的文档字符串和废弃警告复制到当前函数
@_copy_docstring_and_deprecators(Axes.plot_date)
def plot_date(
    # 接受数组类型的 x 坐标数据
    x: ArrayLike,
    # 接受数组类型的 y 坐标数据
    y: ArrayLike,
    # 可选参数，指定绘制日期的格式字符串，默认为 "o"
    fmt: str = "o",
    # 可选参数，指定时区信息或者 None
    tz: str | datetime.tzinfo | None = None,
    # 可选参数，指定 x 轴是否使用日期格式
    xdate: bool = True,
    # 可选参数，指定 y 轴是否使用日期格式
    ydate: bool = False,
    # 命名关键字参数，传递额外数据，如果不为 None 则作为关键字参数传递给绘图方法
    *,
    data=None,
    # 关键字参数，传递给底层绘图方法的其余参数
    **kwargs,
) -> list[Line2D]:
    # 调用当前坐标轴对象的 plot_date 方法，绘制日期折线图，并返回绘制结果
    return gca().plot_date(
        x,
        y,
        fmt=fmt,
        tz=tz,
        xdate=xdate,
        ydate=ydate,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.

# 使用 _copy_docstring_and_deprecators 装饰器，将 Axes.psd 方法的文档字符串和废弃警告复制到当前函数
@_copy_docstring_and_deprecators(Axes.psd)
def psd(
    # 接受数组类型的输入参数 x，表示信号数据
    x: ArrayLike,
    # 可选参数，指定 FFT 窗口大小或者 None
    NFFT: int | None = None,
    # 可选参数，指定信号的采样频率或者 None
    Fs: float | None = None,
    # 可选参数，指定信号的中心频率或者 None
    Fc: int | None = None,
    # 可选参数，指定去趋势的方法，可以是字符串或者回调函数
    detrend: Literal["none", "mean", "linear"]
    | Callable[[ArrayLike], ArrayLike]
    | None = None,
    # 可选参数，指定窗口函数，可以是函数或者数组类型或者 None
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    # 可选参数，指定重叠部分的数量或者 None
    noverlap: int | None = None,
    # 可选参数，指定输出长度的填充方式或者 None
    pad_to: int | None = None,
    # 可
def psd(
    x,
    NFFT=None,
    Fs=None,
    Fc=None,
    detrend=None,
    window=None,
    noverlap=0,
    pad_to=None,
    sides='default',
    scale_by_freq=None,
    return_line=False,
    data=None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Line2D]:
    __ret = gca().psd(
        x,
        NFFT=NFFT,
        Fs=Fs,
        Fc=Fc,
        detrend=detrend,
        window=window,
        noverlap=noverlap,
        pad_to=pad_to,
        sides=sides,
        scale_by_freq=scale_by_freq,
        return_line=return_line,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    sci(__ret)
    return __ret


根据给定的参数计算信号的功率谱密度（PSD），返回频谱和功率密度数组。



@_copy_docstring_and_deprecators(Axes.quiver)
def quiver(*args, data=None, **kwargs) -> Quiver:
    __ret = gca().quiver(
        *args, **({"data": data} if data is not None else {}), **kwargs
    )
    sci(__ret)
    return __ret


使用当前的坐标轴创建一个矢量场图（quiver plot），根据传入的参数绘制箭头。可选地使用额外的数据坐标系。



@_copy_docstring_and_deprecators(Axes.quiverkey)
def quiverkey(
    Q: Quiver, X: float, Y: float, U: float, label: str, **kwargs
) -> QuiverKey:
    return gca().quiverkey(Q, X, Y, U, label, **kwargs)


为矢量场图（quiver plot）添加一个箭头图例，指定箭头的位置、大小和标签。



@_copy_docstring_and_deprecators(Axes.scatter)
def scatter(
    x: float | ArrayLike,
    y: float | ArrayLike,
    s: float | ArrayLike | None = None,
    c: ArrayLike | Sequence[ColorType] | ColorType | None = None,
    marker: MarkerType | None = None,
    cmap: str | Colormap | None = None,
    norm: str | Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float | None = None,
    linewidths: float | Sequence[float] | None = None,
    *,
    edgecolors: Literal["face", "none"] | ColorType | Sequence[ColorType] | None = None,
    plotnonfinite: bool = False,
    data=None,
    **kwargs,
) -> PathCollection:
    __ret = gca().scatter(
        x,
        y,
        s=s,
        c=c,
        marker=marker,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
        plotnonfinite=plotnonfinite,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )
    sci(__ret)
    return __ret


在当前的坐标轴上绘制散点图，根据给定的参数绘制散点。可选地使用额外的数据坐标系。



@_copy_docstring_and_deprecators(Axes.semilogx)
def semilogx(*args, **kwargs) -> list[Line2D]:
    return gca().semilogx(*args, **kwargs)


在对数刻度下绘制 x 轴的线图。使用给定的参数绘制线条。



@_copy_docstring_and_deprecators(Axes.semilogy)
def semilogy(*args, **kwargs) -> list[Line2D]:
    return gca().semilogy(*args, **kwargs)


在对数刻度下绘制 y 轴的线图。使用给定的参数绘制线条。



@_copy_docstring_and_deprecators(Axes.specgram)
def specgram(
    x: ArrayLike,
    NFFT: int | None = None,
    Fs: float | None = None,
    Fc: int | None = None,
    detrend: Literal["none", "mean", "linear"]
    | Callable[[ArrayLike], ArrayLike]
    | None = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,


计算信号的谱图（spectrogram），显示信号的频谱随时间的变化。参数包括输入信号、窗口大小等。
    noverlap: int | None = None,
    cmap: str | Colormap | None = None,
    xextent: tuple[float, float] | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    scale_by_freq: bool | None = None,
    mode: Literal["default", "psd", "magnitude", "angle", "phase"] | None = None,
    scale: Literal["default", "linear", "dB"] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    data=None,
    **kwargs,



    noverlap: int | None = None,  # 滑动窗口重叠量，可以是整数或None
    cmap: str | Colormap | None = None,  # 颜色映射，可以是字符串、Colormap对象或None
    xextent: tuple[float, float] | None = None,  # x轴范围，由两个浮点数组成的元组或None
    pad_to: int | None = None,  # FFT结果的填充长度，可以是整数或None
    sides: Literal["default", "onesided", "twosided"] | None = None,  # FFT的侧面类型，可以是默认、单侧或双侧，或None
    scale_by_freq: bool | None = None,  # 是否按频率缩放，可以是布尔值或None
    mode: Literal["default", "psd", "magnitude", "angle", "phase"] | None = None,  # 绘图模式，可以是默认、功率谱密度、幅度、角度、相位，或None
    scale: Literal["default", "linear", "dB"] | None = None,  # 比例尺，可以是默认、线性、分贝，或None
    vmin: float | None = None,  # 数据的最小值，可以是浮点数或None
    vmax: float | None = None,  # 数据的最大值，可以是浮点数或None
    *,  # 标记位置，表示后面的参数只能通过关键字传递
    data=None,  # 用于绘图的数据，默认为None
    **kwargs,  # 其他关键字参数，用于传递给绘图函数的额外选项
# 返回一个元组，包含四个元素：np.ndarray，np.ndarray，np.ndarray，AxesImage
def __ret = gca().specgram(
    x,  # 输入信号 x
    NFFT=NFFT,  # FFT窗口大小
    Fs=Fs,  # 采样频率
    Fc=Fc,  # 信号中心频率
    detrend=detrend,  # 是否去趋势
    window=window,  # 窗函数
    noverlap=noverlap,  # 重叠窗口长度
    cmap=cmap,  # 颜色映射
    xextent=xextent,  # x轴范围
    pad_to=pad_to,  # 是否填充至指定长度
    sides=sides,  # 显示双侧频谱或单侧频谱
    scale_by_freq=scale_by_freq,  # 是否按频率放大
    mode=mode,  # 绘制模式
    scale=scale,  # 缩放比例
    vmin=vmin,  # 颜色映射的最小值
    vmax=vmax,  # 颜色映射的最大值
    **({"data": data} if data is not None else {}),  # 如果提供了数据，则传递给specgram函数
    **kwargs,  # 其他关键字参数传递给specgram函数
)
sci(__ret[-1])  # 科学记数法显示结果的最后一个元素
return __ret



# 使用 gca().spy 绘制矩阵 Z 的稀疏热图，并返回绘制的对象
@_copy_docstring_and_deprecators(Axes.spy)
def spy(
    Z: ArrayLike,  # 输入的数组或类数组对象
    precision: float | Literal["present"] = 0,  # 控制像素显示的精度或特殊的"present"值
    marker: str | None = None,  # 点的标记类型
    markersize: float | None = None,  # 点的大小
    aspect: Literal["equal", "auto"] | float | None = "equal",  # 纵横比或"equal"，"auto"
    origin: Literal["upper", "lower"] = "upper",  # 原点位置，"upper" 或 "lower"
    **kwargs,  # 其他关键字参数传递给 gca().spy 函数
) -> AxesImage:  # 返回值为 AxesImage 对象
    __ret = gca().spy(
        Z,
        precision=precision,
        marker=marker,
        markersize=markersize,
        aspect=aspect,
        origin=origin,
        **kwargs,
    )
    if isinstance(__ret, cm.ScalarMappable):
        sci(__ret)  # 如果返回对象是 ScalarMappable 类型，则使用科学记数法显示
    return __ret



# 使用 gca().stackplot 绘制堆叠图，并返回绘制的对象
@_copy_docstring_and_deprecators(Axes.stackplot)
def stackplot(
    x,  # x 轴数据
    *args,  # 可变位置参数，用于堆叠图的数据
    labels=(),  # 图例标签
    colors=None,  # 堆叠区域的颜色
    hatch=None,  # 填充的图案样式
    baseline="zero",  # 堆叠基线
    data=None,  # 数据对象
    **kwargs  # 其他关键字参数传递给 gca().stackplot 函数
):
    return gca().stackplot(
        x,
        *args,
        labels=labels,
        colors=colors,
        hatch=hatch,
        baseline=baseline,
        **({"data": data} if data is not None else {}),  # 如果提供了数据，则传递给 gca().stackplot 函数
        **kwargs,
    )



# 使用 gca().stem 绘制柴火图，并返回绘制的对象
@_copy_docstring_and_deprecators(Axes.stem)
def stem(
    *args: ArrayLike | str,  # 可变位置参数，用于柴火图的数据
    linefmt: str | None = None,  # 线条格式
    markerfmt: str | None = None,  # 标记格式
    basefmt: str | None = None,  # 基线格式
    bottom: float = 0,  # 基线的位置
    label: str | None = None,  # 标签
    orientation: Literal["vertical", "horizontal"] = "vertical",  # 方向
    data=None,  # 数据对象
) -> StemContainer:  # 返回值为 StemContainer 对象
    return gca().stem(
        *args,
        linefmt=linefmt,
        markerfmt=markerfmt,
        basefmt=basefmt,
        bottom=bottom,
        label=label,
        orientation=orientation,
        **({"data": data} if data is not None else {}),  # 如果提供了数据，则传递给 gca().stem 函数
    )



# 使用 gca().step 绘制步进图，并返回绘制的对象列表
@_copy_docstring_and_deprecators(Axes.step)
def step(
    x: ArrayLike,  # x 轴数据
    y: ArrayLike,  # y 轴数据
    *args,  # 可变位置参数，用于步进图的其他数据
    where: Literal["pre", "post", "mid"] = "pre",  # 绘制方式
    data=None,  # 数据对象
    **kwargs,  # 其他关键字参数传递给 gca().step 函数
) -> list[Line2D]:  # 返回值为 Line2D 对象列表
    return gca().step(
        x,
        y,
        *args,
        where=where,
        **({"data": data} if data is not None else {}),  # 如果提供了数据，则传递给 gca().step 函数
        **kwargs,
    )



# 使用 gca().streamplot 绘制流线图，并返回绘制的对象
@_copy_docstring_and_deprecators(Axes.streamplot)
def streamplot(
    x,  # x 轴数据
    y,  # y 轴数据
    u,  # x 方向的速度分量
    v,  # y 方向的速度分量
    density=1,  # 控制箭头的密度
    linewidth=None,  # 箭头的线宽
    color=None,  # 箭头的颜色
    **kwargs,  # 其他关键字参数传递给 gca().streamplot 函数
):
    return gca().streamplot(
        x,
        y,
        u,
        v,
        density=density,
        linewidth=linewidth,
        color=color,
        **kwargs,
    )
    cmap=None,  # 颜色映射，用于定义箭头的颜色，默认为None
    norm=None,  # 用于映射颜色数据的归一化对象，默认为None
    arrowsize=1,  # 箭头大小的比例因子，默认为1
    arrowstyle="-|>",  # 箭头样式，可以是预定义的字符串形式，默认为"-|>"
    minlength=0.1,  # 最小长度，用于控制最小的流线长度，默认为0.1
    transform=None,  # 坐标系的转换对象，默认为None
    zorder=None,  # 图层顺序，控制绘图对象的前后顺序，默认为None
    start_points=None,  # 流线的起始点列表，默认为None
    maxlength=4.0,  # 最大长度，用于控制最大的流线长度，默认为4.0
    integration_direction="both",  # 积分方向，可以是"both"（双向）或"forward"（向前），"backward"（向后）之一，默认为"both"
    broken_streamlines=True,  # 是否允许断裂的流线，默认为True
    *,  # 从这里开始是强制关键字参数，后续参数必须以关键字形式传递
    data=None,  # 用户提供的额外数据，用于流线绘制的数据源，默认为None
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用当前的坐标轴对象绘制流场图
def streamplot(
    x,
    y,
    u,
    v,
    density=None,
    linewidth=None,
    color=None,
    cmap=None,
    norm=None,
    arrowsize=None,
    arrowstyle=None,
    minlength=None,
    transform=None,
    zorder=None,
    start_points=None,
    maxlength=None,
    integration_direction='both',
    broken_streamlines=False,
    **({"data": data} if data is not None else {}),
):
    # 调用当前坐标轴的streamplot方法，并返回结果
    __ret = gca().streamplot(
        x,
        y,
        u,
        v,
        density=density,
        linewidth=linewidth,
        color=color,
        cmap=cmap,
        norm=norm,
        arrowsize=arrowsize,
        arrowstyle=arrowstyle,
        minlength=minlength,
        transform=transform,
        zorder=zorder,
        start_points=start_points,
        maxlength=maxlength,
        integration_direction=integration_direction,
        broken_streamlines=broken_streamlines,
        **({"data": data} if data is not None else {}),
    )
    # 将生成的线对象添加到科学计数法的视图中
    sci(__ret.lines)
    # 返回streamplot方法的结果
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用当前的坐标轴对象绘制表格
def table(
    cellText=None,
    cellColours=None,
    cellLoc="right",
    colWidths=None,
    rowLabels=None,
    rowColours=None,
    rowLoc="left",
    colLabels=None,
    colColours=None,
    colLoc="center",
    loc="bottom",
    bbox=None,
    edges="closed",
    **kwargs,
):
    # 调用当前坐标轴的table方法，并返回结果
    return gca().table(
        cellText=cellText,
        cellColours=cellColours,
        cellLoc=cellLoc,
        colWidths=colWidths,
        rowLabels=rowLabels,
        rowColours=rowColours,
        rowLoc=rowLoc,
        colLabels=colLabels,
        colColours=colColours,
        colLoc=colLoc,
        loc=loc,
        bbox=bbox,
        edges=edges,
        **kwargs,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用当前的坐标轴对象添加文本
def text(
    x: float, y: float, s: str, fontdict: dict[str, Any] | None = None, **kwargs
) -> Text:
    # 调用当前坐标轴的text方法，并返回文本对象
    return gca().text(x, y, s, fontdict=fontdict, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置当前坐标轴的刻度参数
def tick_params(axis: Literal["both", "x", "y"] = "both", **kwargs) -> None:
    # 调用当前坐标轴的tick_params方法，设置刻度参数
    gca().tick_params(axis=axis, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置当前坐标轴的刻度标签格式
def ticklabel_format(
    *,
    axis: Literal["both", "x", "y"] = "both",
    style: Literal["", "sci", "scientific", "plain"] | None = None,
    scilimits: tuple[int, int] | None = None,
    useOffset: bool | float | None = None,
    useLocale: bool | None = None,
    useMathText: bool | None = None,
) -> None:
    # 调用当前坐标轴的ticklabel_format方法，设置刻度标签格式
    gca().ticklabel_format(
        axis=axis,
        style=style,
        scilimits=scilimits,
        useOffset=useOffset,
        useLocale=useLocale,
        useMathText=useMathText,
    )


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用当前的坐标轴对象绘制三角形等高线图
def tricontour(*args, **kwargs):
    # 调用当前坐标轴的tricontour方法，并返回结果
    __ret = gca().tricontour(*args, **kwargs)
    # 如果等高线图返回的对象具有数据数组，则将其添加到科学计数法的视图中
    if __ret._A is not None:  # type: ignore[attr-defined]
        sci(__ret)
    # 返回tricontour方法的结果
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 使用当前的坐标轴对象填充三角形等高线图
def tricontourf(*args, **kwargs):
    # 调用当前坐标轴的tricontourf方法，并返回结果
    return gca().tricontourf(*args, **kwargs)
# 调用当前轴对象的 `tricontourf` 方法，绘制三角形等高线填充图，并返回结果
def tricontourf(*args, **kwargs):
    __ret = gca().tricontourf(*args, **kwargs)
    # 如果返回对象的 `_A` 属性不为 None，则调用 `sci` 函数进行科学计数标记
    if __ret._A is not None:  # type: ignore[attr-defined]
        sci(__ret)
    return __ret


# 通过调用当前轴对象的 `tripcolor` 方法绘制三角形颜色填充图
# 自动生成的代码，不要编辑，否则更改将会丢失
@_copy_docstring_and_deprecators(Axes.tripcolor)
def tripcolor(
    *args,
    alpha=1.0,
    norm=None,
    cmap=None,
    vmin=None,
    vmax=None,
    shading="flat",
    facecolors=None,
    **kwargs,
):
    __ret = gca().tripcolor(
        *args,
        alpha=alpha,
        norm=norm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading=shading,
        facecolors=facecolors,
        **kwargs,
    )
    # 调用 `sci` 函数进行科学计数标记
    sci(__ret)
    return __ret


# 自动生成的代码，不要编辑，否则更改将会丢失
@_copy_docstring_and_deprecators(Axes.triplot)
def triplot(*args, **kwargs):
    # 调用当前轴对象的 `triplot` 方法，绘制三角形图
    return gca().triplot(*args, **kwargs)


# 自动生成的代码，不要编辑，否则更改将会丢失
@_copy_docstring_and_deprecators(Axes.violinplot)
def violinplot(
    dataset: ArrayLike | Sequence[ArrayLike],
    positions: ArrayLike | None = None,
    vert: bool | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    widths: float | ArrayLike = 0.5,
    showmeans: bool = False,
    showextrema: bool = True,
    showmedians: bool = False,
    quantiles: Sequence[float | Sequence[float]] | None = None,
    points: int = 100,
    bw_method: Literal["scott", "silverman"]
    | float
    | Callable[[GaussianKDE], float]
    | None = None,
    side: Literal["both", "low", "high"] = "both",
    *,
    data=None,
) -> dict[str, Collection]:
    # 调用当前轴对象的 `violinplot` 方法，绘制小提琴图，并返回结果
    return gca().violinplot(
        dataset,
        positions=positions,
        vert=vert,
        orientation=orientation,
        widths=widths,
        showmeans=showmeans,
        showextrema=showextrema,
        showmedians=showmedians,
        quantiles=quantiles,
        points=points,
        bw_method=bw_method,
        side=side,
        **({"data": data} if data is not None else {}),
    )


# 自动生成的代码，不要编辑，否则更改将会丢失
@_copy_docstring_and_deprecators(Axes.vlines)
def vlines(
    x: float | ArrayLike,
    ymin: float | ArrayLike,
    ymax: float | ArrayLike,
    colors: ColorType | Sequence[ColorType] | None = None,
    linestyles: LineStyleType = "solid",
    label: str = "",
    *,
    data=None,
    **kwargs,
) -> LineCollection:
    # 调用当前轴对象的 `vlines` 方法，绘制垂直线，并返回结果
    return gca().vlines(
        x,
        ymin,
        ymax,
        colors=colors,
        linestyles=linestyles,
        label=label,
        **({"data": data} if data is not None else {}),
        **kwargs,
    )


# 自动生成的代码，不要编辑，否则更改将会丢失
@_copy_docstring_and_deprecators(Axes.xcorr)
def xcorr(
    x: ArrayLike,
    y: ArrayLike,
    normed: bool = True,
    detrend: Callable[[ArrayLike], ArrayLike] = mlab.detrend_none,
    usevlines: bool = True,
    maxlags: int = 10,
    *,
    data=None,
) -> dict[str, Collection]:
    # 调用当前轴对象的 `xcorr` 方法，计算并绘制两个序列的相关性，并返回结果
    return gca().xcorr(
        x,
        y,
        normed=normed,
        detrend=detrend,
        usevlines=usevlines,
        maxlags=maxlags,
        **({"data": data} if data is not None else {}),
    )
    data=None,
    **kwargs,
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.

# 设置默认的 colormap 为 'cool'
def cool() -> None:
    """
    Set the colormap to 'cool'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
    设置默认的 colormap 为 'cool'
    set_cmap("cool")
    # 设置默认的颜色映射为 "cool"，同时如果有当前图像，也会将其颜色映射设置为 "cool"
    set_cmap("cool")
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'copper'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def copper() -> None:
    set_cmap("copper")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'flag'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def flag() -> None:
    set_cmap("flag")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'gray'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def gray() -> None:
    set_cmap("gray")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'hot'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def hot() -> None:
    set_cmap("hot")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'hsv'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def hsv() -> None:
    set_cmap("hsv")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'jet'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def jet() -> None:
    set_cmap("jet")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'pink'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def pink() -> None:
    set_cmap("pink")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'prism'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def prism() -> None:
    set_cmap("prism")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'spring'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def spring() -> None:
    set_cmap("spring")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色图为 'summer'。

# 这会改变默认色图以及当前图像的色图（如果有的话）。更多信息请参见 `help(colormaps)`。
def summer() -> None:
    set_cmap("summer")
    # 设置默认色彩映射为 "summer"，并且如果当前存在图像，也会将其色彩映射修改为 "summer"
    set_cmap("summer")
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色彩映射为 'winter'。
# 这会改变默认的色彩映射以及当前图像的色彩映射（如果有的话）。
# 更多信息请参考 `help(colormaps)`。
def winter() -> None:
    set_cmap("winter")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色彩映射为 'magma'。
# 这会改变默认的色彩映射以及当前图像的色彩映射（如果有的话）。
# 更多信息请参考 `help(colormaps)`。
def magma() -> None:
    set_cmap("magma")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色彩映射为 'inferno'。
# 这会改变默认的色彩映射以及当前图像的色彩映射（如果有的话）。
# 更多信息请参考 `help(colormaps)`。
def inferno() -> None:
    set_cmap("inferno")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色彩映射为 'plasma'。
# 这会改变默认的色彩映射以及当前图像的色彩映射（如果有的话）。
# 更多信息请参考 `help(colormaps)`。
def plasma() -> None:
    set_cmap("plasma")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色彩映射为 'viridis'。
# 这会改变默认的色彩映射以及当前图像的色彩映射（如果有的话）。
# 更多信息请参考 `help(colormaps)`。
def viridis() -> None:
    set_cmap("viridis")


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# 设置色彩映射为 'nipy_spectral'。
# 这会改变默认的色彩映射以及当前图像的色彩映射（如果有的话）。
# 更多信息请参考 `help(colormaps)`。
def nipy_spectral() -> None:
    set_cmap("nipy_spectral")
```