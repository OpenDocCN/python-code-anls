# `D:\src\scipysrc\matplotlib\lib\matplotlib\style\core.py`

```
"""
Core functions and attributes for the matplotlib style library:

``use``
    Select style sheet to override the current matplotlib settings.
``context``
    Context manager to use a style sheet temporarily.
``available``
    List available style sheets.
``library``
    A dictionary of style names and matplotlib settings.
"""

import contextlib  # 导入上下文管理工具模块
import logging  # 导入日志记录模块
import os  # 导入操作系统功能模块
from pathlib import Path  # 导入路径操作模块
import sys  # 导入系统模块
import warnings  # 导入警告模块

if sys.version_info >= (3, 10):
    import importlib.resources as importlib_resources  # Python 3.10 及以上版本使用的资源管理模块
else:
    # 对于 Python 3.9 及以下版本，使用 importlib.resources 时的兼容性说明
    import importlib_resources

import matplotlib as mpl  # 导入 matplotlib 库
from matplotlib import _api, _docstring, _rc_params_in_file, rcParamsDefault  # 导入 matplotlib 内部 API

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

__all__ = ['use', 'context', 'available', 'library', 'reload_library']  # 模块中公开的接口列表

BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), 'stylelib')  # 基础风格库路径，获取 matplotlib 的数据路径并与 'stylelib' 目录组合成完整路径
# 用户风格库路径列表，通常是用户配置路径下的 'stylelib' 目录
USER_LIBRARY_PATHS = [os.path.join(mpl.get_configdir(), 'stylelib')]
STYLE_EXTENSION = 'mplstyle'  # 风格文件扩展名
# 不应用于风格的 rcParams 列表
STYLE_BLACKLIST = {
    'interactive', 'backend', 'webagg.port', 'webagg.address',
    'webagg.port_retries', 'webagg.open_in_browser', 'backend_fallback',
    'toolbar', 'timezone', 'figure.max_open_warning',
    'figure.raise_window', 'savefig.directory', 'tk.window_focus',
    'docstring.hardcopy', 'date.epoch'}

@_docstring.Substitution(
    "\n".join(map("- {}".format, sorted(STYLE_BLACKLIST, key=str.lower)))
)
def use(style):
    """
    Use Matplotlib style settings from a style specification.

    The style name of 'default' is reserved for reverting back to
    the default style settings.

    .. note::

       This updates the `.rcParams` with the settings from the style.
       `.rcParams` not defined in the style are kept.

    Parameters
    ----------
    style : str, dict, Path or list

        A style specification. Valid options are:

        str
            - One of the style names in `.style.available` (a builtin style or
              a style installed in the user library path).

            - A dotted name of the form "package.style_name"; in that case,
              "package" should be an importable Python package name, e.g. at
              ``/path/to/package/__init__.py``; the loaded style file is
              ``/path/to/package/style_name.mplstyle``.  (Style files in
              subpackages are likewise supported.)

            - The path or URL to a style file, which gets loaded by
              `.rc_params_from_file`.

        dict
            A mapping of key/value pairs for `matplotlib.rcParams`.

        Path
            The path to a style file, which gets loaded by
            `.rc_params_from_file`.

        list
            A list of style specifiers (str, Path or dict), which are applied
            from first to last in the list.

    Notes
    -----
    """
    """
    The following function applies a specified style configuration to Matplotlib's runtime parameters (`rcParams`).
    
    If the provided `style` argument is a string, Path object, or dictionary, it processes it accordingly. Multiple styles
    can be provided as a list. It also includes handling for certain style aliases and deprecated styles.
    
    The function iterates through each specified style, applies it to `rcParams`, and updates the runtime configuration
    accordingly. It filters out parameters that are not related to styling and logs warnings for these exclusions.
    
    Parameters:
    - style: Can be a string, Path object, dictionary, or list thereof, representing the style configuration.
    
    """
    
    if isinstance(style, (str, Path)) or hasattr(style, 'keys'):
        # If `style` is a single string, Path, or dictionary, convert it to a list with one element.
        styles = [style]
    else:
        styles = style
    
    # Alias for certain styles to standard names
    style_alias = {'mpl20': 'default', 'mpl15': 'classic'}
    
    for style in styles:
        if isinstance(style, str):
            # Resolve aliases to their respective standard names
            style = style_alias.get(style, style)
            
            if style == "default":
                # Apply default style configuration from `rcParamsDefault`, excluding certain parameters.
                with _api.suppress_matplotlib_deprecation_warning():
                    style = {k: rcParamsDefault[k] for k in rcParamsDefault
                             if k not in STYLE_BLACKLIST}
            elif style in library:
                # Apply a style from a predefined library.
                style = library[style]
            elif "." in style:
                # Attempt to load a style from a dotted package.module_name or file path.
                pkg, _, name = style.rpartition(".")
                try:
                    path = (importlib_resources.files(pkg)
                            / f"{name}.{STYLE_EXTENSION}")
                    style = _rc_params_in_file(path)
                except (ModuleNotFoundError, OSError, TypeError) as exc:
                    # Handle potential errors when loading the style file or package.
                    pass
        
        if isinstance(style, (str, Path)):
            # Load style configuration from a file (specified by path or string).
            try:
                style = _rc_params_in_file(style)
            except OSError as err:
                # Raise an error if the specified style cannot be loaded.
                raise OSError(
                    f"{style!r} is not a valid package style, path of style "
                    f"file, URL of style file, or library style name (library "
                    f"styles are listed in `style.available`)") from err
        
        # Filter out parameters in `style` that are not related to styling.
        filtered = {}
        for k in style:
            if k in STYLE_BLACKLIST:
                # Warn about parameters that are not related to styling.
                _api.warn_external(
                    f"Style includes a parameter, {k!r}, that is not "
                    f"related to style.  Ignoring this parameter.")
            else:
                filtered[k] = style[k]
        
        # Update `rcParams` with the filtered style configuration.
        mpl.rcParams.update(filtered)
# 定义上下文管理器，用于临时应用样式设置
@contextlib.contextmanager
def context(style, after_reset=False):
    """
    Context manager for using style settings temporarily.

    Parameters
    ----------
    style : str, dict, Path or list
        A style specification. Valid options are:

        str
            - One of the style names in `.style.available` (a builtin style or
              a style installed in the user library path).

            - A dotted name of the form "package.style_name"; in that case,
              "package" should be an importable Python package name, e.g. at
              ``/path/to/package/__init__.py``; the loaded style file is
              ``/path/to/package/style_name.mplstyle``.  (Style files in
              subpackages are likewise supported.)

            - The path or URL to a style file, which gets loaded by
              `.rc_params_from_file`.
        dict
            A mapping of key/value pairs for `matplotlib.rcParams`.

        Path
            The path to a style file, which gets loaded by
            `.rc_params_from_file`.

        list
            A list of style specifiers (str, Path or dict), which are applied
            from first to last in the list.

    after_reset : bool
        If True, apply style after resetting settings to their defaults;
        otherwise, apply style on top of the current settings.
    """
    # 进入 matplotlib 的上下文环境
    with mpl.rc_context():
        # 如果设置了 after_reset 参数，重置到默认设置
        if after_reset:
            mpl.rcdefaults()
        # 应用指定的样式
        use(style)
        # 通过 yield 将控制权交给调用方，在这期间执行相关操作
        yield


def update_user_library(library):
    """Update style library with user-defined rc files."""
    # 遍历用户样式库路径，更新主样式库
    for stylelib_path in map(os.path.expanduser, USER_LIBRARY_PATHS):
        # 读取样式目录中的样式文件并更新到 library 中
        styles = read_style_directory(stylelib_path)
        update_nested_dict(library, styles)
    return library


def read_style_directory(style_dir):
    """Return dictionary of styles defined in *style_dir*."""
    # 初始化一个空字典用于存储样式
    styles = dict()
    # 遍历指定样式目录下的所有样式文件
    for path in Path(style_dir).glob(f"*.{STYLE_EXTENSION}"):
        # 使用 _rc_params_in_file 函数读取样式文件中的参数
        with warnings.catch_warnings(record=True) as warns:
            styles[path.stem] = _rc_params_in_file(path)
        # 遍历警告信息并记录到日志中
        for w in warns:
            _log.warning('In %s: %s', path, w.message)
    return styles


def update_nested_dict(main_dict, new_dict):
    """
    Update nested dict (only level of nesting) with new values.

    Unlike `dict.update`, this assumes that the values of the parent dict are
    dicts (or dict-like), so you shouldn't replace the nested dict if it
    already exists. Instead you should update the sub-dict.
    """
    # 更新主字典中的嵌套字典的值
    for name, rc_dict in new_dict.items():
        main_dict.setdefault(name, {}).update(rc_dict)
    return main_dict


# 加载基础样式库
# =============
_base_library = read_style_directory(BASE_LIBRARY_PATH)
# 初始化样式库字典
library = {}
# 初始化可用样式列表
available = []


def reload_library():
    """Reload the style library."""
    # 清空当前样式库
    library.clear()
    # 更新基础样式库到主样式库
    library.update(update_user_library(_base_library))
    # 将样式库中的样式名按字母顺序存储到可用样式列表中
    available[:] = sorted(library.keys())


# 刷新样式库
reload_library()
```