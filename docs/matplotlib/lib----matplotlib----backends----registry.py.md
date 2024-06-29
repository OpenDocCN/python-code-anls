# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\registry.py`

```py
from enum import Enum  # 导入枚举类型模块

import importlib  # 导入动态导入模块


class BackendFilter(Enum):  # 定义枚举类型 BackendFilter
    """
    Filter used with :meth:`~matplotlib.backends.registry.BackendRegistry.list_builtin`

    .. versionadded:: 3.9
    """
    INTERACTIVE = 0  # 枚举成员：交互式后端
    NON_INTERACTIVE = 1  # 枚举成员：非交互式后端


class BackendRegistry:  # 定义后端注册类 BackendRegistry
    """
    Registry of backends available within Matplotlib.

    This is the single source of truth for available backends.

    All use of ``BackendRegistry`` should be via the singleton instance
    ``backend_registry`` which can be imported from ``matplotlib.backends``.

    Each backend has a name, a module name containing the backend code, and an
    optional GUI framework that must be running if the backend is interactive.
    There are three sources of backends: built-in (source code is within the
    Matplotlib repository), explicit ``module://some.backend`` syntax (backend is
    obtained by loading the module), or via an entry point (self-registering
    backend in an external package).

    .. versionadded:: 3.9
    """

    # Mapping of built-in backend name to GUI framework, or "headless" for no
    # GUI framework. Built-in backends are those which are included in the
    # Matplotlib repo. A backend with name 'name' is located in the module
    # f"matplotlib.backends.backend_{name.lower()}"
    _BUILTIN_BACKEND_TO_GUI_FRAMEWORK = {
        "gtk3agg": "gtk3",  # GTK3 后端对应的 GUI 框架为 gtk3
        "gtk3cairo": "gtk3",  # GTK3 Cairo 后端对应的 GUI 框架为 gtk3
        "gtk4agg": "gtk4",  # GTK4 后端对应的 GUI 框架为 gtk4
        "gtk4cairo": "gtk4",  # GTK4 Cairo 后端对应的 GUI 框架为 gtk4
        "macosx": "macosx",  # macOS 后端对应的 GUI 框架为 macosx
        "nbagg": "nbagg",  # nbagg 后端对应的 GUI 框架为 nbagg
        "notebook": "nbagg",  # notebook 后端对应的 GUI 框架为 nbagg
        "qtagg": "qt",  # qtagg 后端对应的 GUI 框架为 qt
        "qtcairo": "qt",  # qtcairo 后端对应的 GUI 框架为 qt
        "qt5agg": "qt5",  # qt5agg 后端对应的 GUI 框架为 qt5
        "qt5cairo": "qt5",  # qt5cairo 后端对应的 GUI 框架为 qt5
        "tkagg": "tk",  # tkagg 后端对应的 GUI 框架为 tk
        "tkcairo": "tk",  # tkcairo 后端对应的 GUI 框架为 tk
        "webagg": "webagg",  # webagg 后端对应的 GUI 框架为 webagg
        "wx": "wx",  # wx 后端对应的 GUI 框架为 wx
        "wxagg": "wx",  # wxagg 后端对应的 GUI 框架为 wx
        "wxcairo": "wx",  # wxcairo 后端对应的 GUI 框架为 wx
        "agg": "headless",  # agg 后端为无 GUI 框架的 headless 后端
        "cairo": "headless",  # cairo 后端为无 GUI 框架的 headless 后端
        "pdf": "headless",  # pdf 后端为无 GUI 框架的 headless 后端
        "pgf": "headless",  # pgf 后端为无 GUI 框架的 headless 后端
        "ps": "headless",  # ps 后端为无 GUI 框架的 headless 后端
        "svg": "headless",  # svg 后端为无 GUI 框架的 headless 后端
        "template": "headless",  # template 后端为无 GUI 框架的 headless 后端
    }

    # Reverse mapping of gui framework to preferred built-in backend.
    _GUI_FRAMEWORK_TO_BACKEND = {
        "gtk3": "gtk3agg",  # gtk3 对应的首选内置后端为 gtk3agg
        "gtk4": "gtk4agg",  # gtk4 对应的首选内置后端为 gtk4agg
        "headless": "agg",  # headless 对应的首选内置后端为 agg
        "macosx": "macosx",  # macosx 对应的首选内置后端为 macosx
        "qt": "qtagg",  # qt 对应的首选内置后端为 qtagg
        "qt5": "qt5agg",  # qt5 对应的首选内置后端为 qt5agg
        "qt6": "qtagg",  # qt6 对应的首选内置后端为 qtagg
        "tk": "tkagg",  # tk 对应的首选内置后端为 tkagg
        "wx": "wxagg",  # wx 对应的首选内置后端为 wxagg
    }
    def __init__(self):
        # 当首次需要时才加载入口点。
        self._loaded_entry_points = False

        # 非内建后端到GUI框架的映射，动态添加自入口点和matplotlib.use("module://some.backend")格式。
        # 新条目有一个"unknown"的GUI框架，在首次需要时通过调用 _get_gui_framework_by_loading 确定。
        self._backend_to_gui_framework = {}

        # 后端名称到模块名称的映射，对于不同于"f"matplotlib.backends.backend_{backend_name.lower()}的情况。
        # 这些要么是为了向后兼容而硬编码的，要么是从入口点或"module://some.backend"语法加载的。
        self._name_to_module = {
            "notebook": "nbagg",
        }

    def _backend_module_name(self, backend):
        if backend.startswith("module://"):
            return backend[9:]

        # 返回包含指定后端的模块名称。
        # 不检查后端是否有效，请使用 is_valid_backend 进行检查。
        backend = backend.lower()

        # 检查是否有特定名称到模块的映射。
        backend = self._name_to_module.get(backend, backend)

        return (backend[9:] if backend.startswith("module://")
                else f"matplotlib.backends.backend_{backend}")

    def _clear(self):
        # 清除所有动态添加的数据，仅用于测试。
        self.__init__()

    def _ensure_entry_points_loaded(self):
        # 如果尚未加载入口点，则加载入口点。
        if not self._loaded_entry_points:
            entries = self._read_entry_points()
            self._validate_and_store_entry_points(entries)
            self._loaded_entry_points = True

    def _get_gui_framework_by_loading(self, backend):
        # 通过加载其模块并读取 FigureCanvas.required_interactive_framework 属性来确定后端的GUI框架。
        # 如果没有GUI框架，则返回"headless"。
        module = self.load_backend_module(backend)
        canvas_class = module.FigureCanvas
        return canvas_class.required_interactive_framework or "headless"
    def _read_entry_points(self):
        # 读取自我声明为 Matplotlib 后端的模块的入口点。
        # 期望类似于 matplotlib-inline 的入口点（在 pyproject.toml 格式中）：
        #   [project.entry-points."matplotlib.backend"]
        #   inline = "matplotlib_inline.backend_inline"
        import importlib.metadata as im  # 导入导入元数据模块
        import sys  # 导入 sys 模块

        # Python 3.10 之前版本没有 entry_points 组关键字
        group = "matplotlib.backend"
        if sys.version_info >= (3, 10):
            entry_points = im.entry_points(group=group)  # 获取特定组的入口点
        else:
            entry_points = im.entry_points().get(group, ())  # 获取所有入口点并选择特定组的入口点
        entries = [(entry.name, entry.value) for entry in entry_points]

        # 为了向后兼容，如果安装了 matplotlib-inline 和/或 ipympl 但版本太旧而不包含入口点，则创建它们。
        # 在此函数中不直接导入 ipympl，因为这会调用 matplotlib.use()。
        def backward_compatible_entry_points(
                entries, module_name, threshold_version, names, target):
            from matplotlib import _parse_to_version_info
            try:
                module_version = im.version(module_name)
                if _parse_to_version_info(module_version) < threshold_version:
                    for name in names:
                        entries.append((name, target))
            except im.PackageNotFoundError:
                pass

        names = [entry[0] for entry in entries]
        if "inline" not in names:
            backward_compatible_entry_points(
                entries, "matplotlib_inline", (0, 1, 7), ["inline"],
                "matplotlib_inline.backend_inline")
        if "ipympl" not in names:
            backward_compatible_entry_points(
                entries, "ipympl", (0, 9, 4), ["ipympl", "widget"],
                "ipympl.backend_nbagg")

        return entries
    # 验证并存储入口点，以便可以通过 matplotlib.use() 正常使用它们。
    # 入口点名称不能以 module:// 格式开头，不能覆盖内置后端名称，并且不能有多个
    # 具有相同名称但不同模块的入口点。允许具有相同名称和值的多个入口点（有时会在我们的控制之外发生，
    # 参见 https://github.com/matplotlib/matplotlib/issues/28367）。
    for name, module in set(entries):
        # 将入口点名称转换为小写
        name = name.lower()
        # 如果入口点名称以 "module://" 开头，引发运行时错误
        if name.startswith("module://"):
            raise RuntimeError(
                f"Entry point name '{name}' cannot start with 'module://'")
        # 如果入口点名称在内置后端名称字典中
        if name in self._BUILTIN_BACKEND_TO_GUI_FRAMEWORK:
            raise RuntimeError(f"Entry point name '{name}' is a built-in backend")
        # 如果入口点名称已经存在于已知的后端到 GUI 框架映射中
        if name in self._backend_to_gui_framework:
            raise RuntimeError(f"Entry point name '{name}' duplicated")

        # 将入口点名称与其模块连接成 module:// 模块形式，并存储在 _name_to_module 字典中
        self._name_to_module[name] = "module://" + module
        # 尚不知道后端 GUI 框架，只在必要时确定。
        self._backend_to_gui_framework[name] = "unknown"

def backend_for_gui_framework(self, framework):
    """
    返回与指定 GUI 框架对应的后端名称。

    Parameters
    ----------
    framework : str
        GUI 框架，例如 "qt"。

    Returns
    -------
    str or None
        后端名称，如果未识别 GUI 框架则返回 None。
    """
    return self._GUI_FRAMEWORK_TO_BACKEND.get(framework.lower())
    def is_valid_backend(self, backend):
        """
        Return True if the backend name is valid, False otherwise.

        A backend name is valid if it is one of the built-in backends or has been
        dynamically added via an entry point. Those beginning with ``module://`` are
        always considered valid and are added to the current list of all backends
        within this function.

        Even if a name is valid, it may not be importable or usable. This can only be
        determined by loading and using the backend module.

        Parameters
        ----------
        backend : str
            Name of backend.

        Returns
        -------
        bool
            True if backend is valid, False otherwise.
        """
        # 如果 backend 不是以 "module://" 开头，则转换为小写
        if not backend.startswith("module://"):
            backend = backend.lower()

        # 将长名称转换为其简化形式，以确保向后兼容性
        backwards_compat = {
            "module://ipympl.backend_nbagg": "widget",
            "module://matplotlib_inline.backend_inline": "inline",
        }
        backend = backwards_compat.get(backend, backend)

        # 检查 backend 是否是内置后端或已经动态添加的后端
        if (backend in self._BUILTIN_BACKEND_TO_GUI_FRAMEWORK or
                backend in self._backend_to_gui_framework):
            return True

        # 如果 backend 是以 "module://" 开头，则添加到 _backend_to_gui_framework 中并返回 True
        if backend.startswith("module://"):
            self._backend_to_gui_framework[backend] = "unknown"
            return True

        # 如果需要，加载 entry points 并检查 backend 是否存在于 _backend_to_gui_framework 中
        self._ensure_entry_points_loaded()
        if backend in self._backend_to_gui_framework:
            return True

        # 如果以上条件都不满足，则返回 False
        return False

    def list_all(self):
        """
        Return list of all known backends.

        These include built-in backends and those obtained at runtime either from entry
        points or explicit ``module://some.backend`` syntax.

        Entry points will be loaded if they haven't been already.

        Returns
        -------
        list of str
            Backend names.
        """
        # 确保已加载 entry points
        self._ensure_entry_points_loaded()
        # 返回所有已知后端的列表，包括内置后端和运行时获取的后端
        return [*self.list_builtin(), *self._backend_to_gui_framework]
    def list_builtin(self, filter_=None):
        """
        Return list of backends that are built into Matplotlib.

        Parameters
        ----------
        filter_ : `~.BackendFilter`, optional
            Filter to apply to returned backends. For example, to return only
            non-interactive backends use `.BackendFilter.NON_INTERACTIVE`.

        Returns
        -------
        list of str
            Backend names.
        """
        # 如果指定了交互式后端过滤器，返回不是"headless"的后端列表
        if filter_ == BackendFilter.INTERACTIVE:
            return [k for k, v in self._BUILTIN_BACKEND_TO_GUI_FRAMEWORK.items()
                    if v != "headless"]
        # 如果指定了非交互式后端过滤器，返回"headless"的后端列表
        elif filter_ == BackendFilter.NON_INTERACTIVE:
            return [k for k, v in self._BUILTIN_BACKEND_TO_GUI_FRAMEWORK.items()
                    if v == "headless"]

        # 否则返回所有内置后端的列表
        return [*self._BUILTIN_BACKEND_TO_GUI_FRAMEWORK]

    def list_gui_frameworks(self):
        """
        Return list of GUI frameworks used by Matplotlib backends.

        Returns
        -------
        list of str
            GUI framework names.
        """
        # 返回所有不是"headless"的 GUI 框架列表
        return [k for k in self._GUI_FRAMEWORK_TO_BACKEND if k != "headless"]

    def load_backend_module(self, backend):
        """
        Load and return the module containing the specified backend.

        Parameters
        ----------
        backend : str
            Name of backend to load.

        Returns
        -------
        Module
            Module containing backend.
        """
        # 根据后端名称获取模块名称
        module_name = self._backend_module_name(backend)
        # 动态导入并返回包含指定后端的模块
        return importlib.import_module(module_name)
    def resolve_backend(self, backend):
        """
        Return the backend and GUI framework for the specified backend name.

        If the GUI framework is not yet known then it will be determined by loading the
        backend module and checking the ``FigureCanvas.required_interactive_framework``
        attribute.

        This function only loads entry points if they have not already been loaded and
        the backend is not built-in and not of ``module://some.backend`` format.

        Parameters
        ----------
        backend : str or None
            Name of backend, or None to use the default backend.

        Returns
        -------
        backend : str
            The backend name.
        framework : str or None
            The GUI framework, which will be None for a backend that is non-interactive.
        """
        # 如果 backend 是字符串类型
        if isinstance(backend, str):
            # 如果 backend 不以 "module://" 开头，则将其转换为小写
            if not backend.startswith("module://"):
                backend = backend.lower()
        else:  # 如果 backend 可能是 _auto_backend_sentinel 或 None
            # 使用当前正在运行的 backend...
            from matplotlib import get_backend
            backend = get_backend()

        # 检查是否已知 backend（内置或动态加载）
        gui = (self._BUILTIN_BACKEND_TO_GUI_FRAMEWORK.get(backend) or
               self._backend_to_gui_framework.get(backend))

        # 检查是否为 "module://something" 形式的 backend
        if gui is None and isinstance(backend, str) and backend.startswith("module://"):
            gui = "unknown"

        # 检查是否为可能的 entry point
        if gui is None and not self._loaded_entry_points:
            self._ensure_entry_points_loaded()
            gui = self._backend_to_gui_framework.get(backend)

        # 如果 backend 已知但其 gui framework 未知
        if gui == "unknown":
            gui = self._get_gui_framework_by_loading(backend)
            self._backend_to_gui_framework[backend] = gui

        # 如果 gui 仍然为 None，则抛出 RuntimeError
        if gui is None:
            raise RuntimeError(f"'{backend}' is not a recognised backend name")

        # 返回 backend 和 gui framework（如果不是 "headless" 则返回）
        return backend, gui if gui != "headless" else None
    def resolve_gui_or_backend(self, gui_or_backend):
        """
        Return the backend and GUI framework for the specified string that may be
        either a GUI framework or a backend name, tested in that order.

        This is for use with the IPython %matplotlib magic command which may be a GUI
        framework such as ``%matplotlib qt`` or a backend name such as
        ``%matplotlib qtagg``.

        This function only loads entry points if they have not already been loaded and
        the backend is not built-in and not of ``module://some.backend`` format.

        Parameters
        ----------
        gui_or_backend : str or None
            Name of GUI framework or backend, or None to use the default backend.

        Returns
        -------
        backend : str
            The backend name.
        framework : str or None
            The GUI framework, which will be None for a backend that is non-interactive.
        """
        # 如果 gui_or_backend 不是以 "module://" 开头，则将其转换为小写
        if not gui_or_backend.startswith("module://"):
            gui_or_backend = gui_or_backend.lower()

        # 首先检查它是否是 GUI 框架名称
        backend = self.backend_for_gui_framework(gui_or_backend)
        # 如果找到了对应的 backend，则返回该 backend 和 gui_or_backend
        if backend is not None:
            return backend, gui_or_backend if gui_or_backend != "headless" else None

        # 如果不是 GUI 框架名称，则检查是否是后端名称
        try:
            return self.resolve_backend(gui_or_backend)
        except Exception:  # 如果发生异常（如 KeyError），则抛出 RuntimeError
            raise RuntimeError(
                f"'{gui_or_backend} is not a recognised GUI loop or backend name")
# 创建一个单例模式的后端注册表实例
backend_registry = BackendRegistry()
```