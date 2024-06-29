# `D:\src\scipysrc\matplotlib\lib\matplotlib\__init__.py`

```py
"""
An object-oriented plotting library.

A procedural interface is provided by the companion pyplot module,
which may be imported directly, e.g.::

    import matplotlib.pyplot as plt

or using ipython::

    ipython

at your terminal, followed by::

    In [1]: %matplotlib
    In [2]: import matplotlib.pyplot as plt

at the ipython shell prompt.

For the most part, direct use of the explicit object-oriented library is
encouraged when programming; the implicit pyplot interface is primarily for
working interactively. The exceptions to this suggestion are the pyplot
functions `.pyplot.figure`, `.pyplot.subplot`, `.pyplot.subplots`, and
`.pyplot.savefig`, which can greatly simplify scripting.  See
:ref:`api_interfaces` for an explanation of the tradeoffs between the implicit
and explicit interfaces.

Modules include:

:mod:`matplotlib.axes`
    The `~.axes.Axes` class.  Most pyplot functions are wrappers for
    `~.axes.Axes` methods.  The axes module is the highest level of OO
    access to the library.

:mod:`matplotlib.figure`
    The `.Figure` class.

:mod:`matplotlib.artist`
    The `.Artist` base class for all classes that draw things.

:mod:`matplotlib.lines`
    The `.Line2D` class for drawing lines and markers.

:mod:`matplotlib.patches`
    Classes for drawing polygons.

:mod:`matplotlib.text`
    The `.Text` and `.Annotation` classes.

:mod:`matplotlib.image`
    The `.AxesImage` and `.FigureImage` classes.

:mod:`matplotlib.collections`
    Classes for efficient drawing of groups of lines or polygons.

:mod:`matplotlib.colors`
    Color specifications and making colormaps.

:mod:`matplotlib.cm`
    Colormaps, and the `.ScalarMappable` mixin class for providing color
    mapping functionality to other classes.

:mod:`matplotlib.ticker`
    Calculation of tick mark locations and formatting of tick labels.

:mod:`matplotlib.backends`
    A subpackage with modules for various GUI libraries and output formats.

The base matplotlib namespace includes:

`~matplotlib.rcParams`
    Default configuration settings; their defaults may be overridden using
    a :file:`matplotlibrc` file.

`~matplotlib.use`
    Setting the Matplotlib backend.  This should be called before any
    figure is created, because it is not possible to switch between
    different GUI backends after that.

The following environment variables can be used to customize the behavior:

:envvar:`MPLBACKEND`
    This optional variable can be set to choose the Matplotlib backend. See
    :ref:`what-is-a-backend`.

:envvar:`MPLCONFIGDIR`
    This is the directory used to store user customizations to
    Matplotlib, as well as some caches to improve performance. If
    :envvar:`MPLCONFIGDIR` is not defined, :file:`{HOME}/.config/matplotlib`
    and :file:`{HOME}/.cache/matplotlib` are used on Linux, and
    :file:`{HOME}/.matplotlib` on other platforms, if they are
    writable. Otherwise, the Python standard library's `tempfile.gettempdir`
"""
    is used to find a base directory in which the :file:`matplotlib`
    subdirectory is created.


# 这行代码用于寻找一个基础目录，在该目录下创建 :file:`matplotlib` 子目录。
"""
Matplotlib was initially written by John D. Hunter (1968-2012) and is now
developed and maintained by a host of others.

Occasionally the internal documentation (python docstrings) will refer
to MATLAB®, a registered trademark of The MathWorks, Inc.
"""

__all__ = [
    "__bibtex__",
    "__version__",
    "__version_info__",
    "set_loglevel",
    "ExecutableNotFoundError",
    "get_configdir",
    "get_cachedir",
    "get_data_path",
    "matplotlib_fname",
    "MatplotlibDeprecationWarning",
    "RcParams",
    "rc_params",
    "rc_params_from_file",
    "rcParamsDefault",
    "rcParams",
    "rcParamsOrig",
    "defaultParams",
    "rc",
    "rcdefaults",
    "rc_file_defaults",
    "rc_file",
    "rc_context",
    "use",
    "get_backend",
    "interactive",
    "is_interactive",
    "colormaps",
    "color_sequences",
]

import atexit
from collections import namedtuple  # 导入 namedtuple 类
from collections.abc import MutableMapping  # 导入 MutableMapping 抽象基类
import contextlib  # 导入 contextlib 模块
import functools  # 导入 functools 模块
import importlib  # 导入 importlib 模块
import inspect  # 导入 inspect 模块
from inspect import Parameter  # 从 inspect 模块导入 Parameter 类
import locale  # 导入 locale 模块
import logging  # 导入 logging 模块
import os  # 导入 os 模块
from pathlib import Path  # 从 pathlib 模块导入 Path 类
import pprint  # 导入 pprint 模块
import re  # 导入 re 模块
import shutil  # 导入 shutil 模块
import subprocess  # 导入 subprocess 模块
import sys  # 导入 sys 模块
import tempfile  # 导入 tempfile 模块

from packaging.version import parse as parse_version  # 从 packaging.version 模块导入 parse 函数并重命名为 parse_version

# cbook must import matplotlib only within function
# definitions, so it is safe to import from it here.
from . import _api, _version, cbook, _docstring, rcsetup  # 从当前包中导入 _api, _version, cbook, _docstring, rcsetup 模块
from matplotlib.cbook import sanitize_sequence  # 从 matplotlib.cbook 模块导入 sanitize_sequence 函数
from matplotlib._api import MatplotlibDeprecationWarning  # 从 matplotlib._api 模块导入 MatplotlibDeprecationWarning 类
from matplotlib.rcsetup import cycler  # 从 matplotlib.rcsetup 模块导入 cycler 函数并忽略 F401 错误

_log = logging.getLogger(__name__)  # 获取当前模块的 logger 对象

__bibtex__ = r"""@Article{Hunter:2007,
  Author    = {Hunter, J. D.},
  Title     = {Matplotlib: A 2D graphics environment},
  Journal   = {Computing in Science \& Engineering},
  Volume    = {9},
  Number    = {3},
  Pages     = {90--95},
  abstract  = {Matplotlib is a 2D graphics package used for Python
  for application development, interactive scripting, and
  publication-quality image generation across user
  interfaces and operating systems.},
  publisher = {IEEE COMPUTER SOC},
  year      = 2007
}"""

# modelled after sys.version_info
_VersionInfo = namedtuple('_VersionInfo',
                          'major, minor, micro, releaselevel, serial')


def _parse_to_version_info(version_str):
    """
    Parse a version string to a namedtuple analogous to sys.version_info.

    See:
    https://packaging.pypa.io/en/latest/version.html#packaging.version.parse
    https://docs.python.org/3/library/sys.html#sys.version_info
    """
    v = parse_version(version_str)  # 解析版本字符串为 Version 对象
    if v.pre is None and v.post is None and v.dev is None:
        return _VersionInfo(v.major, v.minor, v.micro, 'final', 0)  # 如果版本为稳定版本，则返回 _VersionInfo 对象
    elif v.dev is not None:
        return _VersionInfo(v.major, v.minor, v.micro, 'alpha', v.dev)  # 如果版本为开发版本，则返回 _VersionInfo 对象
    elif v.pre is not None:
        # 如果版本号有预发行标识符
        releaselevel = {
            'a': 'alpha',
            'b': 'beta',
            'rc': 'candidate'}.get(v.pre[0], 'alpha')
        # 根据预发行标识符确定发行级别，如果未知则默认为 alpha
        return _VersionInfo(v.major, v.minor, v.micro, releaselevel, v.pre[1])
    else:
        # 如果版本号没有预发行标识符，使用后续发行的猜测下一个开发版本方案（来自 setuptools_scm）
        return _VersionInfo(v.major, v.minor, v.micro + 1, 'alpha', v.post)
def _get_version():
    """Return the version string used for __version__."""
    # 只有在真正需要时才调用 git 子进程，例如在 matplotlib 的 git 仓库中但不是浅克隆时，
    # 比如 CI 使用的那些浅克隆会触发 setuptools_scm 的警告。
    root = Path(__file__).resolve().parents[2]
    if ((root / ".matplotlib-repo").exists()
            and (root / ".git").exists()
            and not (root / ".git/shallow").exists()):
        try:
            import setuptools_scm
        except ImportError:
            pass
        else:
            return setuptools_scm.get_version(
                root=root,
                version_scheme="release-branch-semver",
                local_scheme="node-and-date",
                fallback_version=_version.version,
            )
    # 如果不在仓库中或 setuptools_scm 不可用，则从 _version.py 文件获取版本。
    return _version.version


@_api.caching_module_getattr
class __getattr__:
    __version__ = property(lambda self: _get_version())
    __version_info__ = property(
        lambda self: _parse_to_version_info(self.__version__))


def _check_versions():

    # 快速修复以确保在导入 kiwisolver 之前加载 Microsoft Visual C++ redistributable
    from . import ft2font  # noqa: F401

    for modname, minver in [
            ("cycler", "0.10"),
            ("dateutil", "2.7"),
            ("kiwisolver", "1.3.1"),
            ("numpy", "1.23"),
            ("pyparsing", "2.3.1"),
    ]:
        module = importlib.import_module(modname)
        if parse_version(module.__version__) < parse_version(minver):
            raise ImportError(f"Matplotlib requires {modname}>={minver}; "
                              f"you have {module.__version__}")


_check_versions()


# 装饰器确保每次都返回相同的处理程序，并且仅附加一次。
@functools.cache
def _ensure_handler():
    """
    第一次调用此函数时，使用与 logging.basicConfig 相同的格式向 Matplotlib 根记录器附加 StreamHandler。

    每次调用此函数时都返回此处理程序。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    _log.addHandler(handler)
    return handler


def set_loglevel(level):
    """
    配置 Matplotlib 的日志级别。

    Matplotlib 使用标准库的 `logging` 框架在根记录器 'matplotlib' 下。这是一个帮助函数：

    - 设置 Matplotlib 根记录器的级别
    - 设置根记录器处理程序的级别，如果处理程序尚不存在，则创建处理程序

    通常，应该调用 ``set_loglevel("info")`` 或 ``set_loglevel("debug")`` 以获取额外的调试信息。

    安装自己日志处理程序的用户或应用程序可能希望直接操作 ``logging.getLogger('matplotlib')`` 而不是使用此函数。
    """
    # 设置全局日志级别为指定的级别
    _log.setLevel(level.upper())
    # 确保存在并返回当前的日志处理器（handler），然后设置其日志级别为指定的级别
    _ensure_handler().setLevel(level.upper())
# 装饰器函数，用于记录函数的返回值并进行缓存
def _logged_cached(fmt, func=None):
    # 如果 func 参数为 None，则返回实际的装饰器函数
    if func is None:
        return functools.partial(_logged_cached, fmt)

    called = False  # 记录函数是否被调用过的标志
    ret = None  # 保存函数的返回值

    @functools.wraps(func)
    def wrapper(**kwargs):
        nonlocal called, ret
        # 如果函数尚未被调用过
        if not called:
            ret = func(**kwargs)  # 调用函数并保存返回值
            called = True  # 设置调用标志为 True
            _log.debug(fmt, ret)  # 记录返回值到日志中
        return ret  # 直接返回保存的返回值

    return wrapper  # 返回装饰后的函数


# 命名元组，用于表示可执行文件的信息
_ExecInfo = namedtuple("_ExecInfo", "executable raw_version version")


class ExecutableNotFoundError(FileNotFoundError):
    """
    Matplotlib 可选依赖项中的某个可执行文件未找到时抛出的错误。
    """
    pass


@functools.cache
def _get_executable_info(name):
    """
    获取 Matplotlib 可选依赖项中某个可执行文件的版本信息。

    .. warning::
       此函数支持查询的可执行文件列表是根据 Matplotlib 的内部需求设定的，可能会随时更改。

    Parameters
    ----------
    name : str
        要查询的可执行文件。目前支持以下值："dvipng", "gs", "inkscape", "magick", "pdftocairo", "pdftops"。
        此列表可能会随时更改。

    Returns
    -------
    tuple
        命名元组，包含字段 ``executable``（`str` 类型）和 ``version``
        （`packaging.Version` 类型，如果无法确定版本则为 ``None``）。

    Raises
    ------
    ExecutableNotFoundError
        如果未找到可执行文件或其版本过旧，低于 Matplotlib 支持的最低版本。
        也可以通过将其添加到 :envvar:`_MPLHIDEEXECUTABLES` 环境变量中（以逗号分隔的列表），
        在调用此函数之前设置，从而“隐藏”某个可执行文件，以便于调试。
    ValueError
        如果可执行文件不是我们知道如何查询的类型。
    """
    # 定义一个函数impl，接受参数args（命令及其参数列表）、regex（用于匹配版本号的正则表达式）、min_ver（最小版本号要求，默认为None）、ignore_exit_code（是否忽略退出码，默认为False）
    def impl(args, regex, min_ver=None, ignore_exit_code=False):
        try:
            # 执行子进程，捕获其标准输出和标准错误输出，返回输出内容的字符串形式
            output = subprocess.check_output(
                args, stderr=subprocess.STDOUT,
                text=True, errors="replace")
        except subprocess.CalledProcessError as _cpe:
            # 如果子进程返回非零退出码且ignore_exit_code为True，则使用其输出作为output
            if ignore_exit_code:
                output = _cpe.output
            else:
                # 否则抛出ExecutableNotFoundError异常，传递_cpe作为原因
                raise ExecutableNotFoundError(str(_cpe)) from _cpe
        except OSError as _ose:
            # 捕获操作系统错误，抛出ExecutableNotFoundError异常，传递_ose作为原因
            raise ExecutableNotFoundError(str(_ose)) from _ose
        
        # 在output中搜索匹配regex的内容
        match = re.search(regex, output)
        if match:
            # 如果找到匹配项，提取第一个捕获组作为原始版本号
            raw_version = match.group(1)
            # 使用parse_version函数解析原始版本号
            version = parse_version(raw_version)
            # 如果设置了min_ver且当前版本小于min_ver，则抛出ExecutableNotFoundError异常
            if min_ver is not None and version < parse_version(min_ver):
                raise ExecutableNotFoundError(
                    f"You have {args[0]} version {version} but the minimum "
                    f"version supported by Matplotlib is {min_ver}")
            # 返回_ExecInfo对象，包含可执行文件名、原始版本号和解析后的版本号
            return _ExecInfo(args[0], raw_version, version)
        else:
            # 如果未找到匹配项，抛出ExecutableNotFoundError异常，指示无法确定可执行文件的版本
            raise ExecutableNotFoundError(
                f"Failed to determine the version of {args[0]} from "
                f"{' '.join(args)}, which output {output}")

    # 如果name出现在环境变量"_MPLHIDEEXECUTABLES"中，则抛出ExecutableNotFoundError异常，指示该可执行文件被隐藏
    if name in os.environ.get("_MPLHIDEEXECUTABLES", "").split(","):
        raise ExecutableNotFoundError(f"{name} was hidden")

    # 根据name的值执行相应的处理逻辑
    if name == "dvipng":
        # 对dvipng执行impl函数调用，返回其版本信息，要求版本号至少为1.6
        return impl(["dvipng", "-version"], "(?m)^dvipng(?: .*)? (.+)", "1.6")
    elif name == "gs":
        # 对Ghostscript进行处理，尝试多个可能的可执行文件名称列表
        execs = (["gswin32c", "gswin64c", "mgs", "gs"]  # "mgs" for miktex.
                 if sys.platform == "win32" else
                 ["gs"])
        for e in execs:
            try:
                # 使用impl函数调用执行Ghostscript相关命令，要求版本号至少为9
                return impl([e, "--version"], "(.*)", "9")
            except ExecutableNotFoundError:
                pass
        # 如果所有尝试均失败，抛出ExecutableNotFoundError异常，指示无法找到Ghostscript安装
        message = "Failed to find a Ghostscript installation"
        raise ExecutableNotFoundError(message)
    elif name == "inkscape":
        try:
            # 尝试以无界面模式获取Inkscape版本信息，适用于版本<1.0的情况
            return impl(["inkscape", "--without-gui", "-V"],
                        "Inkscape ([^ ]*)")
        except ExecutableNotFoundError:
            pass  # 忽略异常继续执行
        # 如果上述方法失败，尝试不带--without-gui选项获取版本信息，适用于版本>=1.0的情况
        return impl(["inkscape", "-V"], "Inkscape ([^ ]*)")
    elif name == "magick":
        # 如果 name 是 "magick"，则进入条件判断
        if sys.platform == "win32":
            # 检查注册表，避免与 Windows 内置的 convert.exe 混淆
            import winreg
            binpath = ""
            # 尝试不同的标志打开注册表键
            for flag in [0, winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY]:
                try:
                    with winreg.OpenKeyEx(
                            winreg.HKEY_LOCAL_MACHINE,
                            r"Software\Imagemagick\Current",
                            0, winreg.KEY_QUERY_VALUE | flag) as hkey:
                        # 获取注册表键 "BinPath" 的值
                        binpath = winreg.QueryValueEx(hkey, "BinPath")[0]
                except OSError:
                    pass
            path = None
            # 如果找到了可执行文件路径，则在其中查找 convert.exe 或 magick.exe
            if binpath:
                for name in ["convert.exe", "magick.exe"]:
                    candidate = Path(binpath, name)
                    if candidate.exists():
                        path = str(candidate)
                        break
            # 如果未找到路径，则抛出 ExecutableNotFoundError 异常
            if path is None:
                raise ExecutableNotFoundError(
                    "Failed to find an ImageMagick installation")
        else:
            # 对于非 Windows 系统，设定默认路径为 "convert"
            path = "convert"
        # 调用 impl 函数，执行命令 ["path", "--version"]，匹配版本信息
        info = impl([path, "--version"], r"^Version: ImageMagick (\S*)")
        # 如果版本是 "7.0.10-34"，抛出 ExecutableNotFoundError 异常
        if info.raw_version == "7.0.10-34":
            raise ExecutableNotFoundError(
                f"You have ImageMagick {info.version}, which is unsupported")
        # 返回版本信息
        return info
    elif name == "pdftocairo":
        # 如果 name 是 "pdftocairo"，则调用 impl 函数执行命令 ["pdftocairo", "-v"]，匹配版本信息
        return impl(["pdftocairo", "-v"], "pdftocairo version (.*)")
    elif name == "pdftops":
        # 如果 name 是 "pdftops"，调用 impl 函数执行命令 ["pdftops", "-v"]，匹配版本信息
        info = impl(["pdftops", "-v"], "^pdftops version (.*)",
                    ignore_exit_code=True)
        # 如果 info 存在且版本不在支持的范围内，则抛出 ExecutableNotFoundError 异常
        if info and not (
                3 <= info.version.major or
                parse_version("0.9") <= info.version < parse_version("1.0")):
            raise ExecutableNotFoundError(
                f"You have pdftops version {info.version} but the minimum "
                f"version supported by Matplotlib is 3.0")
        # 返回版本信息
        return info
    else:
        # 如果 name 不是已知的可执行文件，则抛出 ValueError 异常
        raise ValueError(f"Unknown executable: {name!r}")
# 返回 XDG 配置目录路径，根据 XDG 基础目录规范确定：
# https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
def _get_xdg_config_dir():
    return os.environ.get('XDG_CONFIG_HOME') or str(Path.home() / ".config")


# 返回 XDG 缓存目录路径，根据 XDG 基础目录规范确定：
# https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
def _get_xdg_cache_dir():
    return os.environ.get('XDG_CACHE_HOME') or str(Path.home() / ".cache")


# 根据所提供的 XDG 基础目录获取配置或缓存目录路径
def _get_config_or_cache_dir(xdg_base_getter):
    # 获取 MPLCONFIGDIR 环境变量设置的目录路径
    configdir = os.environ.get('MPLCONFIGDIR')
    if configdir:
        configdir = Path(configdir).resolve()
    # 如果未设置 MPLCONFIGDIR 环境变量，根据操作系统判断设定默认目录
    elif sys.platform.startswith(('linux', 'freebsd')):
        # 在 Linux 和 FreeBSD 系统上，根据 XDG 规范获取目录，以 matplotlib 子目录存放
        configdir = Path(xdg_base_getter(), "matplotlib")
    else:
        # 在其他平台，默认使用用户主目录下的 .matplotlib 目录
        configdir = Path.home() / ".matplotlib"
    
    try:
        # 创建配置或缓存目录，如果已存在则忽略，同时确保创建父级目录
        configdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    else:
        # 如果目录可写且为目录类型，则返回该目录的字符串路径
        if os.access(str(configdir), os.W_OK) and configdir.is_dir():
            return str(configdir)
    
    # 如果无法创建或无法访问配置或缓存目录，则创建临时目录
    try:
        tmpdir = tempfile.mkdtemp(prefix="matplotlib-")
    except OSError as exc:
        # 抛出异常，指示无法创建临时目录
        raise OSError(
            f"Matplotlib requires access to a writable cache directory, but the "
            f"default path ({configdir}) is not a writable directory, and a temporary "
            f"directory could not be created; set the MPLCONFIGDIR environment "
            f"variable to a writable directory") from exc
    
    # 将 MPLCONFIGDIR 环境变量设置为临时目录路径，并注册退出时删除临时目录的操作
    os.environ["MPLCONFIGDIR"] = tmpdir
    atexit.register(shutil.rmtree, tmpdir)
    
    # 发出警告日志，说明 Matplotlib 创建了临时缓存目录
    _log.warning(
        "Matplotlib created a temporary cache directory at %s because the default path "
        "(%s) is not a writable directory; it is highly recommended to set the "
        "MPLCONFIGDIR environment variable to a writable directory, in particular to "
        "speed up the import of Matplotlib and to better support multiprocessing.",
        tmpdir, configdir)
    
    # 返回临时目录的路径作为配置或缓存目录的最终选择
    return tmpdir


# 装饰器函数，用于记录日志，返回配置目录路径
@_logged_cached('CONFIGDIR=%s')
def get_configdir():
    """
    Return the string path of the configuration directory.

    The directory is chosen as follows:

    1. If the MPLCONFIGDIR environment variable is supplied, choose that.
    2. On Linux, follow the XDG specification and look first in
       ``$XDG_CONFIG_HOME``, if defined, or ``$HOME/.config``.  On other
       platforms, choose ``$HOME/.matplotlib``.
    3. If the chosen directory exists and is writable, use that as the
       configuration directory.
    4. Else, create a temporary directory, and use it as the configuration
       directory.
    """
    return _get_config_or_cache_dir(_get_xdg_config_dir)


# 装饰器函数，用于记录日志，返回缓存目录路径
@_logged_cached('CACHEDIR=%s')
def get_cachedir():
    # 这里的具体实现在上面的 `_get_config_or_cache_dir` 中完成了
    # 因此在此不需要再次注释
    pass
    """
    返回缓存目录的字符串路径。
    
    查找目录的方法与 `get_configdir` 相同，只是使用 `$XDG_CACHE_HOME`/`$HOME/.cache` 替代。
    """
    return _get_config_or_cache_dir(_get_xdg_cache_dir)
# 使用装饰器 `_logged_cached`，记录并缓存获取 Matplotlib 数据路径的函数
@_logged_cached('matplotlib data path: %s')
# 定义获取 Matplotlib 数据路径的函数
def get_data_path():
    """Return the path to Matplotlib data."""
    # 返回当前文件路径加上子目录 'mpl-data' 的路径作为 Matplotlib 数据路径
    return str(Path(__file__).with_name("mpl-data"))


# 定义获取 Matplotlib 配置文件路径的函数
def matplotlib_fname():
    """
    Get the location of the config file.

    The file location is determined in the following order

    - ``$PWD/matplotlibrc``
    - ``$MATPLOTLIBRC`` if it is not a directory
    - ``$MATPLOTLIBRC/matplotlibrc``
    - ``$MPLCONFIGDIR/matplotlibrc``
    - On Linux,
        - ``$XDG_CONFIG_HOME/matplotlib/matplotlibrc`` (if ``$XDG_CONFIG_HOME``
          is defined)
        - or ``$HOME/.config/matplotlib/matplotlibrc`` (if ``$XDG_CONFIG_HOME``
          is not defined)
    - On other platforms,
      - ``$HOME/.matplotlib/matplotlibrc`` if ``$HOME`` is defined
    - Lastly, it looks in ``$MATPLOTLIBDATA/matplotlibrc``, which should always
      exist.
    """

    # 定义生成配置文件候选路径的生成器函数
    def gen_candidates():
        # 返回相对路径 'matplotlibrc' 作为第一个候选项
        yield 'matplotlibrc'
        try:
            # 尝试获取环境变量 MATPLOTLIBRC 的值
            matplotlibrc = os.environ['MATPLOTLIBRC']
        except KeyError:
            pass
        else:
            # 如果成功获取 MATPLOTLIBRC 环境变量，返回其值及其拼接的路径作为候选项
            yield matplotlibrc
            yield os.path.join(matplotlibrc, 'matplotlibrc')
        # 返回配置文件所在配置目录的路径作为候选项
        yield os.path.join(get_configdir(), 'matplotlibrc')
        # 返回 Matplotlib 数据路径中的 'matplotlibrc' 文件路径作为最后一个候选项
        yield os.path.join(get_data_path(), 'matplotlibrc')

    # 遍历所有候选路径，返回第一个存在且不是目录的文件路径作为配置文件路径
    for fname in gen_candidates():
        if os.path.exists(fname) and not os.path.isdir(fname):
            return fname

    # 如果找不到配置文件，抛出运行时错误
    raise RuntimeError("Could not find matplotlibrc file; your Matplotlib "
                       "install is broken")


# rcParams 被弃用并自动映射到另一个键。
# 值是元组 (版本号, 新名称, f_old2new, f_new2old)。
_deprecated_map = {}

# rcParams 被弃用；一些可以手动映射到另一个键。
# 值是元组 (版本号, 新名称或 None)。
_deprecated_ignore_map = {}

# rcParams 被弃用；可以使用 None 来抑制警告；实际上仍然列在 rcParams 中。
# 值是元组 (版本号,)。
_deprecated_remain_as_none = {}


# 使用 `_docstring.Substitution` 替换字符串中的占位符
@_docstring.Substitution(
    "\n".join(map("- {}".format, sorted(rcsetup._validators, key=str.lower)))
)
# RcParams 类，继承自 MutableMapping 和 dict，用于存储和验证配置参数
class RcParams(MutableMapping, dict):
    """
    A dict-like key-value store for config parameters, including validation.

    Validating functions are defined and associated with rc parameters in
    :mod:`matplotlib.rcsetup`.

    The list of rcParams is:

    %s

    See Also
    --------
    :ref:`customizing-with-matplotlibrc-files`
    """

    # 设置验证函数列表为 rcsetup 模块中的验证器列表
    validate = rcsetup._validators

    # 在初始化时验证传入的值
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
    def _set(self, key, val):
        """
        直接写入数据，跳过废弃和验证逻辑。

        Notes
        -----
        作为最终用户或下游库，几乎总是应该使用 ``rcParams[key] = val`` 而不是 ``_set()``。

        仅有极少数特殊情况需要直接访问数据。
        这些情况以前使用 ``dict.__setitem__(rcParams, key, val)``，现在已被废弃并由 ``rcParams._set(key, val)`` 替代。

        即使是私有方法，我们也保证 ``rcParams._set`` 的 API 稳定性，
        即它遵循 Matplotlib 的 API 和废弃策略。

        :meta public:
        """
        dict.__setitem__(self, key, val)

    def _get(self, key):
        """
        直接读取数据，跳过废弃、后端和验证逻辑。

        Notes
        -----
        作为最终用户或下游库，几乎总是应该使用 ``val = rcParams[key]`` 而不是 ``_get()``。

        仅有极少数特殊情况需要直接访问数据。
        这些情况以前使用 ``dict.__getitem__(rcParams, key, val)``，现在已被废弃并由 ``rcParams._get(key)`` 替代。

        即使是私有方法，我们也保证 ``rcParams._get`` 的 API 稳定性，
        即它遵循 Matplotlib 的 API 和废弃策略。

        :meta public:
        """
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        """
        设置字典中的键值对，并处理废弃的键名和特定的值。

        如果键名在 _deprecated_map 中，会根据映射替换键名和值。
        如果键名在 _deprecated_remain_as_none 中，会发出警告并保持值为 None。
        如果键名在 _deprecated_ignore_map 中，会忽略该键的操作。
        如果键名为 'backend'，且值为 rcsetup._auto_backend_sentinel，则检查当前是否已存在 'backend' 键。
        尝试使用键名对应的验证函数验证值，如果验证失败则抛出 ValueError。
        最终通过 self._set 方法将键值对写入字典中。

        如果键名不在字典中，抛出 KeyError 异常并提示该键名不是有效的 rc 参数。

        """
        try:
            if key in _deprecated_map:
                version, alt_key, alt_val, inverse_alt = _deprecated_map[key]
                _api.warn_deprecated(
                    version, name=key, obj_type="rcparam", alternative=alt_key)
                key = alt_key
                val = alt_val(val)
            elif key in _deprecated_remain_as_none and val is not None:
                version, = _deprecated_remain_as_none[key]
                _api.warn_deprecated(version, name=key, obj_type="rcparam")
            elif key in _deprecated_ignore_map:
                version, alt_key = _deprecated_ignore_map[key]
                _api.warn_deprecated(
                    version, name=key, obj_type="rcparam", alternative=alt_key)
                return
            elif key == 'backend':
                if val is rcsetup._auto_backend_sentinel:
                    if 'backend' in self:
                        return
            try:
                cval = self.validate[key](val)
            except ValueError as ve:
                raise ValueError(f"Key {key}: {ve}") from None
            self._set(key, cval)
        except KeyError as err:
            raise KeyError(
                f"{key} is not a valid rc parameter (see rcParams.keys() for "
                f"a list of valid parameters)") from err
    def __getitem__(self, key):
        # 检查是否为已废弃的参数，如果是，则发出警告并返回替代参数的值
        if key in _deprecated_map:
            version, alt_key, alt_val, inverse_alt = _deprecated_map[key]
            _api.warn_deprecated(
                version, name=key, obj_type="rcparam", alternative=alt_key)
            return inverse_alt(self._get(alt_key))

        # 检查是否为忽略的已废弃参数，如果是，则发出警告并返回替代参数的值（如果有的话）
        elif key in _deprecated_ignore_map:
            version, alt_key = _deprecated_ignore_map[key]
            _api.warn_deprecated(
                version, name=key, obj_type="rcparam", alternative=alt_key)
            return self._get(alt_key) if alt_key else None

        # 如果是 "backend" 参数且当前对象是全局的 rcParams 对象，则进行特殊处理
        # 用于设置自动后端时，触发后端自动选择
        elif key == "backend" and self is globals().get("rcParams"):
            val = self._get(key)
            if val is rcsetup._auto_backend_sentinel:
                from matplotlib import pyplot as plt
                plt.switch_backend(rcsetup._auto_backend_sentinel)

        # 其他情况直接返回参数的值
        return self._get(key)

    def _get_backend_or_none(self):
        """获取请求的后端，如果有的话，但不触发解析。"""
        backend = self._get("backend")
        return None if backend is rcsetup._auto_backend_sentinel else backend

    def __repr__(self):
        """返回该 RcParams 实例的可打印表示形式。"""
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        with _api.suppress_matplotlib_deprecation_warning():
            # 使用 pprint 格式化字典，保持良好的缩进和宽度
            repr_split = pprint.pformat(dict(self), indent=1,
                                        width=80 - indent).split('\n')
        repr_indented = ('\n' + ' ' * indent).join(repr_split)
        return f'{class_name}({repr_indented})'

    def __str__(self):
        """返回该 RcParams 实例的字符串表示形式，按键排序输出。"""
        return '\n'.join(map('{0[0]}: {0[1]}'.format, sorted(self.items())))

    def __iter__(self):
        """按键排序后，迭代该 RcParams 实例的键。"""
        with _api.suppress_matplotlib_deprecation_warning():
            yield from sorted(dict.__iter__(self))

    def __len__(self):
        """返回该 RcParams 实例中键的数量。"""
        return dict.__len__(self)

    def find_all(self, pattern):
        """
        返回匹配给定模式的子集 RcParams 字典。

        使用 re.search 进行匹配。

        .. 注意::

            对返回的字典的更改不会传播到父 RcParams 字典中。

        """
        pattern_re = re.compile(pattern)
        return RcParams((key, value)
                        for key, value in self.items()
                        if pattern_re.search(key))

    def copy(self):
        """复制此 RcParams 实例。"""
        rccopy = RcParams()
        for k in self:  # 跳过已废弃和重新验证的参数
            rccopy._set(k, self._get(k))
        return rccopy
# 根据默认的 Matplotlib 配置文件构造一个 `RcParams` 实例
def rc_params(fail_on_error=False):
    # 调用 rc_params_from_file 函数，从默认的 Matplotlib 配置文件中获取 RcParams 实例
    return rc_params_from_file(matplotlib_fname(), fail_on_error)


# 使用 functools.cache 装饰器缓存函数结果，提高性能
@functools.cache
def _get_ssl_context():
    try:
        import certifi
    except ImportError:
        _log.debug("Could not import certifi.")
        return None
    import ssl
    # 创建默认的 SSL 上下文，使用 certifi 提供的 CA 证书路径
    return ssl.create_default_context(cafile=certifi.where())


# 使用 contextlib.contextmanager 装饰器定义上下文管理器，用于打开文件或 URL
@contextlib.contextmanager
def _open_file_or_url(fname):
    if (isinstance(fname, str)
            and fname.startswith(('http://', 'https://', 'ftp://', 'file:'))):
        import urllib.request
        # 获取 SSL 上下文
        ssl_ctx = _get_ssl_context()
        if ssl_ctx is None:
            _log.debug(
                "Could not get certifi ssl context, https may not work."
            )
        # 使用 urllib.request.urlopen 打开 URL，使用 ssl_ctx 进行 HTTPS 请求
        with urllib.request.urlopen(fname, context=ssl_ctx) as f:
            # 使用生成器表达式生成 UTF-8 解码后的行对象，作为 yield 的返回值
            yield (line.decode('utf-8') for line in f)
    else:
        # 扩展用户路径中的波浪号（~），打开本地文件
        fname = os.path.expanduser(fname)
        # 打开本地文件，使用 UTF-8 编码
        with open(fname, encoding='utf-8') as f:
            # 返回文件对象 f 作为 yield 的返回值
            yield f


# 定义一个函数，从文件中构造 `RcParams` 实例
def _rc_params_in_file(fname, transform=lambda x: x, fail_on_error=False):
    """
    Construct a `RcParams` instance from file *fname*.

    Unlike `rc_params_from_file`, the configuration class only contains the
    parameters specified in the file (i.e. default values are not filled in).

    Parameters
    ----------
    fname : path-like
        The loaded file.
    transform : callable, default: the identity function
        A function called on each individual line of the file to transform it,
        before further parsing.
    fail_on_error : bool, default: False
        Whether invalid entries should result in an exception or a warning.
    """
    import matplotlib as mpl
    # 临时存储解析后的配置项
    rc_temp = {}
    # 使用 _open_file_or_url 函数打开 fname 对应的文件
    with _open_file_or_url(fname) as fd:
        try:
            for line_no, line in enumerate(fd, 1):
                # 对每一行应用 transform 函数进行转换
                line = transform(line)
                # 去除每一行的注释部分
                strippedline = cbook._strip_comment(line)
                # 如果剩余部分为空，则跳过该行
                if not strippedline:
                    continue
                # 以 ':' 分割键值对
                tup = strippedline.split(':', 1)
                # 如果分割结果不是两部分，则警告缺少冒号的情况，并跳过该行
                if len(tup) != 2:
                    _log.warning('Missing colon in file %r, line %d (%r)',
                                 fname, line_no, line.rstrip('\n'))
                    continue
                key, val = tup
                key = key.strip()
                val = val.strip()
                # 如果值以双引号开头和结尾，则去除双引号
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]  # 去除双引号
                # 如果键已存在于 rc_temp 中，则警告重复键的情况
                if key in rc_temp:
                    _log.warning('Duplicate key in file %r, line %d (%r)',
                                 fname, line_no, line.rstrip('\n'))
                # 将键值对加入 rc_temp 字典中
                rc_temp[key] = (val, line, line_no)
        except UnicodeDecodeError:
            # 如果文件无法以 UTF-8 解码，则警告并抛出异常
            _log.warning('Cannot decode configuration file %r as utf-8.',
                         fname)
            raise

    # 创建一个 RcParams 实例作为配置
    config = RcParams()
    # 遍历 rc_temp 字典中的每个键值对，其中键为 key，值为元组 (val, line, line_no)
    for key, (val, line, line_no) in rc_temp.items():
        # 检查当前 key 是否在 rcsetup._validators 中
        if key in rcsetup._validators:
            # 如果设置了 fail_on_error 标志，尝试将 val 转换为适当的类型或者抛出异常
            if fail_on_error:
                config[key] = val  # 尝试将 val 转换为适当的类型或者抛出异常
            else:
                try:
                    config[key] = val  # 尝试将 val 转换为适当的类型或者跳过处理
                except Exception as msg:
                    # 在出现异常时记录警告信息，指出文件、行号、行内容及异常消息
                    _log.warning('Bad value in file %r, line %d (%r): %s',
                                 fname, line_no, line.rstrip('\n'), msg)
        # 如果 key 在 _deprecated_ignore_map 中，则进行相应的处理
        elif key in _deprecated_ignore_map:
            # 获取对应的版本号和替代键名
            version, alt_key = _deprecated_ignore_map[key]
            # 发出警告，指出被弃用的设置及替代建议
            _api.warn_deprecated(
                version, name=key, alternative=alt_key, obj_type='rcparam',
                addendum="Please update your matplotlibrc.")
        else:
            # 特殊处理 __version__，通过访问模块级别的 __getattr__ 触发属性查找
            version = ('main' if '.post' in mpl.__version__
                       else f'v{mpl.__version__}')
            # 记录警告信息，指出当前版本信息
            _log.warning('Matplotlib version is %s', version)
# 当文件 %(fname)s 的第 %(line_no)s 行出现 %(key)s 键错误时，输出错误信息，指引用户更新 matplotlibrc 文件
Bad key %(key)s in file %(fname)s, line %(line_no)s (%(line)r)
You probably need to get an updated matplotlibrc file from
https://github.com/matplotlib/matplotlib/blob/%(version)s/lib/matplotlib/mpl-data/matplotlibrc
or from the matplotlib source distribution""",
                         dict(key=key, fname=fname, line_no=line_no,
                              line=line.rstrip('\n'), version=version))

def rc_params_from_file(fname, fail_on_error=False, use_default_template=True):
    """
    从文件 *fname* 构建 `RcParams` 对象。

    Parameters
    ----------
    fname : str or path-like
        Matplotlib rc 设置文件。
    fail_on_error : bool
        如果为 True，解析器在无法转换参数时会引发错误。
    use_default_template : bool
        如果为 True，在使用给定文件中的参数更新默认参数之前初始化。
        如果为 False，配置类仅包含文件中指定的参数。（用于更新字典）

    Returns
    -------
    config : RcParams
        从文件中读取的配置参数。
    """
    # 从文件中获取配置参数
    config_from_file = _rc_params_in_file(fname, fail_on_error=fail_on_error)

    if not use_default_template:
        return config_from_file

    # 使用默认参数模板与从文件中读取的参数合并成新的配置对象
    with _api.suppress_matplotlib_deprecation_warning():
        config = RcParams({**rcParamsDefault, **config_from_file})

    # 如果配置了 LaTeX 前导代码，则记录警告信息
    if "".join(config['text.latex.preamble']):
        _log.info("""
*****************************************************************
You have the following UNSUPPORTED LaTeX preamble customizations:
%s
Please do not ask for support with these customizations active.
*****************************************************************
""", '\n'.join(config['text.latex.preamble']))

    # 记录调试信息，标记已加载的 rc 文件
    _log.debug('loaded rc file %s', fname)

    return config


# 构建全局实例时，需要通过显式调用超类（dict.update，dict.items）执行某些更新操作，
# 以避免触发 _auto_backend_sentinel 的解析。
rcParamsDefault = _rc_params_in_file(
    cbook._get_data_path("matplotlibrc"),
    # 去除行首注释
    transform=lambda line: line[1:] if line.startswith("#") else line,
    fail_on_error=True)

# 将硬编码的默认值更新到 rcParamsDefault 中
dict.update(rcParamsDefault, rcsetup._hardcoded_defaults)

# 如果 matplotlibrc 文件的默认值字典中没有 backend 条目，则填充 _auto_backend_sentinel。
# 如果有 backend 条目，则不填充。
dict.setdefault(rcParamsDefault, "backend", rcsetup._auto_backend_sentinel)

# 创建 RcParams 的全局实例
rcParams = RcParams()

# 更新 rcParams 以包含 rcParamsDefault 中的所有条目
dict.update(rcParams, dict.items(rcParamsDefault))

# 从 matplotlib_fname() 函数返回的文件中更新 rcParams
dict.update(rcParams, _rc_params_in_file(matplotlib_fname()))

# 备份 rcParams 的原始副本
rcParamsOrig = rcParams.copy()

# 这也检查所有 rcParams 是否确实列在模板中。
with _api.suppress_matplotlib_deprecation_warning():
    pass  # 确保所有 rcParams 确实列在模板中。
    # 将 rcsetup.defaultParams 赋值给 defaultParams，同时保留向后兼容性。
    defaultParams = rcsetup.defaultParams = {
        # 对于已废弃的 rcParams，我们希望进行解析，但不包括 "backend"...
        key: [(rcsetup._auto_backend_sentinel if key == "backend" else
               rcParamsDefault[key]),
              validator]
        # 使用 rcsetup._validators.items() 中的键值对进行循环迭代
        for key, validator in rcsetup._validators.items()}
# 如果 rcParams 中 axes.formatter.use_locale 为 True，则设置当前 locale 使用系统默认设置
if rcParams['axes.formatter.use_locale']:
    locale.setlocale(locale.LC_ALL, '')

# 定义函数 rc，用于设置当前的 .rcParams 参数配置
def rc(group, **kwargs):
    """
    设置当前的 `.rcParams` 参数配置。*group* 是参数组的名称，例如，
    对于 `lines.linewidth`，组名是 `lines`，对于 `axes.facecolor`，组名是 `axes`，以此类推。
    *group* 也可以是一组组名的列表或元组，例如（*xtick*, *ytick*）。
    *kwargs* 是一个属性名/值的字典，例如::

      rc('lines', linewidth=2, color='r')

    设置当前的 `.rcParams` 等效于::

      rcParams['lines.linewidth'] = 2
      rcParams['lines.color'] = 'r'

    下列别名可用于简化输入，方便交互式用户：

    =====   =================
    别名     属性
    =====   =================
    'lw'    'linewidth'
    'ls'    'linestyle'
    'c'     'color'
    'fc'    'facecolor'
    'ec'    'edgecolor'
    'mew'   'markeredgewidth'
    'aa'    'antialiased'
    =====   =================

    因此，上述调用可以简写为::

          rc('lines', lw=2, c='r')

    注意可以使用 Python 的 kwargs 字典功能存储默认参数字典。例如，可以如下自定义
    字体参数配置::

      font = {'family' : 'monospace',
              'weight' : 'bold',
              'size'   : 'larger'}
      rc('font', **font)  # 将字体字典作为 kwargs 传入

    这使得可以轻松地在多个配置之间切换。使用 ``matplotlib.style.use('default')`` 或
    :func:`~matplotlib.rcdefaults` 来恢复默认的 `.rcParams` 设置。

    注意
    -----
    使用正常的字典接口也可以实现类似的功能，即 ``rcParams.update({"lines.linewidth": 2, ...})``
    （但 `rcParams.update` 不支持别名或分组）。
    """

    # 别名字典，用于将简写映射为完整的参数名
    aliases = {
        'lw':  'linewidth',
        'ls':  'linestyle',
        'c':   'color',
        'fc':  'facecolor',
        'ec':  'edgecolor',
        'mew': 'markeredgewidth',
        'aa':  'antialiased',
    }

    # 如果 group 是字符串，则转换为元组形式
    if isinstance(group, str):
        group = (group,)
    
    # 遍历每个组名
    for g in group:
        # 遍历 kwargs 中的每个属性名和值
        for k, v in kwargs.items():
            # 获取别名对应的完整参数名，如果没有别名则使用原始的属性名
            name = aliases.get(k) or k
            key = f'{g}.{name}'  # 组合成完整的参数键名
            try:
                rcParams[key] = v  # 设置对应的参数值到 rcParams 中
            except KeyError as err:
                raise KeyError(('Unrecognized key "%s" for group "%s" and '
                                'name "%s"') % (key, g, name)) from err


def rcdefaults():
    """
    从 Matplotlib 的内部默认样式中恢复 `.rcParams`。

    样式黑名单中的 `.rcParams`（在 `matplotlib.style.core.STYLE_BLACKLIST` 中定义）
    不会被更新。

    参见
    --------
    matplotlib.rc_file_defaults
        从最初由 Matplotlib 加载的 rc 文件中恢复 `.rcParams`。
    matplotlib.style.use
        使用特定的样式文件。调用 ``style.use('default')`` 来恢复默认样式。
    """
    # 使用 `_api.suppress_matplotlib_deprecation_warning()` 上下文管理器来抑制 matplotlib 废弃警告
    with _api.suppress_matplotlib_deprecation_warning():
        # 从 `style.core` 模块导入 STYLE_BLACKLIST
        from .style.core import STYLE_BLACKLIST
        # 清空当前的 rcParams 参数设置
        rcParams.clear()
        # 更新 rcParams 参数设置，仅保留 rcParamsDefault 中不在 STYLE_BLACKLIST 中的项
        rcParams.update({k: v for k, v in rcParamsDefault.items()
                         if k not in STYLE_BLACKLIST})
# 恢复 `.rcParams` 到由 Matplotlib 加载的原始 rc 文件的状态。
# 在这个过程中，不会更新风格黑名单中的 `.rcParams`（定义在 `matplotlib.style.core.STYLE_BLACKLIST` 中）。
def rc_file_defaults():
    """
    Restore the `.rcParams` from the original rc file loaded by Matplotlib.

    Style-blacklisted `.rcParams` (defined in
    ``matplotlib.style.core.STYLE_BLACKLIST``) are not updated.
    """
    # Deprecation warnings were already handled when creating rcParamsOrig, no
    # need to reemit them here.
    # 使用 `_api.suppress_matplotlib_deprecation_warning()` 来抑制已经在创建 `rcParamsOrig` 时处理过的弃用警告。
    with _api.suppress_matplotlib_deprecation_warning():
        # 从 `.style.core` 模块导入 `STYLE_BLACKLIST`
        from .style.core import STYLE_BLACKLIST
        # 更新 `.rcParams`，排除掉在 `STYLE_BLACKLIST` 中的键
        rcParams.update({k: rcParamsOrig[k] for k in rcParamsOrig
                         if k not in STYLE_BLACKLIST})


def rc_file(fname, *, use_default_template=True):
    """
    Update `.rcParams` from file.

    Style-blacklisted `.rcParams` (defined in
    ``matplotlib.style.core.STYLE_BLACKLIST``) are not updated.

    Parameters
    ----------
    fname : str or path-like
        A file with Matplotlib rc settings.

    use_default_template : bool
        If True, initialize with default parameters before updating with those
        in the given file. If False, the current configuration persists
        and only the parameters specified in the file are updated.
    """
    # Deprecation warnings were already handled in rc_params_from_file, no need
    # to reemit them here.
    # 使用 `_api.suppress_matplotlib_deprecation_warning()` 来抑制已经在 `rc_params_from_file` 中处理过的弃用警告。
    with _api.suppress_matplotlib_deprecation_warning():
        # 从 `.style.core` 模块导入 `STYLE_BLACKLIST`
        from .style.core import STYLE_BLACKLIST
        # 从文件中读取 rc 参数
        rc_from_file = rc_params_from_file(
            fname, use_default_template=use_default_template)
        # 更新 `.rcParams`，排除掉在 `STYLE_BLACKLIST` 中的键
        rcParams.update({k: rc_from_file[k] for k in rc_from_file
                         if k not in STYLE_BLACKLIST})


@contextlib.contextmanager
def rc_context(rc=None, fname=None):
    """
    Return a context manager for temporarily changing rcParams.

    The :rc:`backend` will not be reset by the context manager.

    rcParams changed both through the context manager invocation and
    in the body of the context will be reset on context exit.

    Parameters
    ----------
    rc : dict
        The rcParams to temporarily set.
    fname : str or path-like
        A file with Matplotlib rc settings. If both *fname* and *rc* are given,
        settings from *rc* take precedence.

    See Also
    --------
    :ref:`customizing-with-matplotlibrc-files`

    Examples
    --------
    Passing explicit values via a dict::

        with mpl.rc_context({'interactive': False}):
            fig, ax = plt.subplots()
            ax.plot(range(3), range(3))
            fig.savefig('example.png')
            plt.close(fig)

    Loading settings from a file::

         with mpl.rc_context(fname='print.rc'):
             plt.plot(x, y)  # uses 'print.rc'

    Setting in the context body::

        with mpl.rc_context():
            # will be reset
            mpl.rcParams['lines.linewidth'] = 5
            plt.plot(x, y)

    """
    # 备份当前的 rcParams 到 `orig` 变量，并删除 `backend` 键
    orig = dict(rcParams.copy())
    del orig['backend']
    try:
        # 如果指定了 `fname`，从文件中更新 rcParams
        if fname:
            rc_file(fname)
        # 如果指定了 `rc`，使用它更新 rcParams
        if rc:
            rcParams.update(rc)
        # 进入上下文管理器
        yield
    finally:
        dict.update(rcParams, orig)  # 在 finally 块中，使用 dict.update() 方法将 orig 中的键值对更新到 rcParams 中，恢复原始的参数设置。
# 选择用于渲染和 GUI 整合的后端。

def use(backend, *, force=True):
    # 验证并获取有效的后端名称
    name = validate_backend(backend)
    # 不要过早解析 "auto" 后端设置
    if rcParams._get_backend_or_none() == name:
        # 如果请求的后端已经设置，无需执行任何操作
        pass
    else:
        # 如果 pyplot 尚未导入，则不导入它。这样做可能会在我们有机会更改到用户刚请求的后端之前，
        # 触发 `plt.switch_backend` 切换到 _default_ 后端
        plt = sys.modules.get('matplotlib.pyplot')
        # 如果 pyplot 已导入，则尝试切换后端
        if plt is not None:
            try:
                # 在此处进行导入检查，以便在用户未安装支持所选后端的库时重新引发异常。
                plt.switch_backend(name)
            except ImportError:
                if force:
                    raise
        # 如果未导入 pyplot，则可以设置 rcParam 值，
        # 这将在用户最终导入 pyplot 时得到尊重
        else:
            rcParams['backend'] = backend
    # 如果用户请求了特定的后端，则不帮助回退到备用后端
    rcParams['backend_fallback'] = False


# 如果存在 MPLBACKEND 环境变量，则设置 rcParams['backend'] 为其值
if os.environ.get('MPLBACKEND'):
    rcParams['backend'] = os.environ.get('MPLBACKEND')


# 返回当前后端的名称。
def get_backend():
    # 返回全局绘图参数（matplotlib）中的后端设置值
    return rcParams['backend']
def interactive(b):
    """
    Set whether to redraw after every plotting command (e.g. `.pyplot.xlabel`).
    """
    # 设置是否在每个绘图命令后重绘
    rcParams['interactive'] = b


def is_interactive():
    """
    Return whether to redraw after every plotting command.

    .. note::

        This function is only intended for use in backends. End users should
        use `.pyplot.isinteractive` instead.
    """
    # 返回是否在每个绘图命令后重绘的设置值
    return rcParams['interactive']


def _val_or_rc(val, rc_name):
    """
    If *val* is None, return ``mpl.rcParams[rc_name]``, otherwise return val.
    """
    # 如果 *val* 为 None，则返回 mpl.rcParams[rc_name]，否则返回 val
    return val if val is not None else rcParams[rc_name]


def _init_tests():
    # The version of FreeType to install locally for running the tests. This must match
    # the value in `meson.build`.
    LOCAL_FREETYPE_VERSION = '2.6.1'

    # 导入 ft2font 模块并检查本地安装的 FreeType 版本是否匹配测试需求
    from matplotlib import ft2font
    if (ft2font.__freetype_version__ != LOCAL_FREETYPE_VERSION or
            ft2font.__freetype_build_type__ != 'local'):
        _log.warning(
            "Matplotlib is not built with the correct FreeType version to run tests.  "
            "Rebuild without setting system-freetype=true in Meson setup options.  "
            "Expect many image comparison failures below.  "
            "Expected freetype version %s.  "
            "Found freetype version %s.  "
            "Freetype build type is %slocal.",
            LOCAL_FREETYPE_VERSION,
            ft2font.__freetype_version__,
            "" if ft2font.__freetype_build_type__ == 'local' else "not ")


def _replacer(data, value):
    """
    Either returns ``data[value]`` or passes ``data`` back, converts either to
    a sequence.
    """
    try:
        # 如果 value 是字符串，则尝试使用 __getitem__ 获取 data 中的对应值
        if isinstance(value, str):
            value = data[value]
    except Exception:
        # 如果键不存在或发生异常，则静默地返回 value
        pass
    return sanitize_sequence(value)


def _label_from_arg(y, default_name):
    try:
        # 尝试返回 y 的名称属性
        return y.name
    except AttributeError:
        # 如果 y 没有名称属性，则返回默认名称或 None
        if isinstance(default_name, str):
            return default_name
    return None


def _add_data_doc(docstring, replace_names):
    """
    Add documentation for a *data* field to the given docstring.

    Parameters
    ----------
    docstring : str
        The input docstring.
    replace_names : list of str or None
        The list of parameter names which arguments should be replaced by
        ``data[name]`` (if ``data[name]`` does not throw an exception).  If
        None, replacement is attempted for all arguments.

    Returns
    -------
    str
        The augmented docstring.
    """
    if (docstring is None
            or replace_names is not None and len(replace_names) == 0):
        return docstring
    docstring = inspect.cleandoc(docstring)

    data_doc = ("""\
    If given, all parameters also accept a string ``s``, which is
    interpreted as ``data[s]`` if ``s`` is a key in ``data``."""
                if replace_names is None else f"""\

    # 如果提供了 replace_names 参数且不为空列表，则替换文档字符串中的参数名为 `data[name]`
    docstring += "\n\n"
    return docstring
    If given, the following parameters also accept a string ``s``, which is
    interpreted as ``data[s]`` if ``s`` is a key in ``data``:

    {', '.join(map('*{}*'.format, replace_names))}""")
    # 使用字符串替换而不是格式化具有以下优点
    # 1) 简化缩进处理
    # 2) 避免在文档字符串中出现格式化字符 '{', '%' 的问题

    # 如果日志级别为 DEBUG 及以下
    if _log.level <= logging.DEBUG:
        # test_data_parameter_replacement() 用于测试这些日志消息
        # 确保消息和测试保持同步
        if "data : indexable object, optional" not in docstring:
            _log.debug("data parameter docstring error: no data parameter")
        if 'DATA_PARAMETER_PLACEHOLDER' not in docstring:
            _log.debug("data parameter docstring error: missing placeholder")

    # 返回替换了 'DATA_PARAMETER_PLACEHOLDER' 的文档字符串
    return docstring.replace('    DATA_PARAMETER_PLACEHOLDER', data_doc)
# 定义一个装饰器函数，用于给函数添加一个名为 'data' 的关键字参数
def _preprocess_data(func=None, *, replace_names=None, label_namer=None):
    """
    A decorator to add a 'data' kwarg to a function.

    When applied::

        @_preprocess_data()
        def func(ax, *args, **kwargs): ...

    the signature is modified to ``decorated(ax, *args, data=None, **kwargs)``
    with the following behavior:

    - if called with ``data=None``, forward the other arguments to ``func``;
    - otherwise, *data* must be a mapping; for any argument passed in as a
      string ``name``, replace the argument by ``data[name]`` (if this does not
      throw an exception), then forward the arguments to ``func``.

    In either case, any argument that is a `MappingView` is also converted to a
    list.

    Parameters
    ----------
    replace_names : list of str or None, default: None
        The list of parameter names for which lookup into *data* should be
        attempted. If None, replacement is attempted for all arguments.
    label_namer : str, default: None
        If set e.g. to "namer" (which must be a kwarg in the function's
        signature -- not as ``**kwargs``), if the *namer* argument passed in is
        a (string) key of *data* and no *label* kwarg is passed, then use the
        (string) value of the *namer* as *label*. ::

            @_preprocess_data(label_namer="foo")
            def func(foo, label=None): ...

            func("key", data={"key": value})
            # is equivalent to
            func.__wrapped__(value, label="key")
    """

    if func is None:  # Return the actual decorator.
        # 返回实际的装饰器函数
        return functools.partial(
            _preprocess_data,
            replace_names=replace_names, label_namer=label_namer)

    # 获取函数的签名信息
    sig = inspect.signature(func)
    varargs_name = None
    varkwargs_name = None
    arg_names = []
    params = list(sig.parameters.values())

    # 遍历函数参数，处理可变位置参数和可变关键字参数
    for p in params:
        if p.kind is Parameter.VAR_POSITIONAL:
            varargs_name = p.name
        elif p.kind is Parameter.VAR_KEYWORD:
            varkwargs_name = p.name
        else:
            arg_names.append(p.name)

    # 添加一个名为 'data' 的关键字参数到函数签名中
    data_param = Parameter("data", Parameter.KEYWORD_ONLY, default=None)
    if varkwargs_name:
        params.insert(-1, data_param)
    else:
        params.append(data_param)

    # 使用更新后的参数列表创建新的函数签名
    new_sig = sig.replace(parameters=params)

    # 移除第一个 "ax" / self 参数后的其余参数作为有效参数列表
    arg_names = arg_names[1:]

    # 检查参数名的有效性，确保 replace_names 中的参数名都在有效参数列表中，或者函数有可变关键字参数
    assert {*arg_names}.issuperset(replace_names or []) or varkwargs_name, (
        "Matplotlib internal error: invalid replace_names "
        f"({replace_names!r}) for {func.__name__!r}")

    # 检查 label_namer 是否为 None 或者是函数签名中的有效参数名之一
    assert label_namer is None or label_namer in arg_names, (
        "Matplotlib internal error: invalid label_namer "
        f"({label_namer!r}) for {func.__name__!r}")

    # 返回经过装饰器装饰后的函数，保留原始函数的信息
    @functools.wraps(func)
    def inner(ax, *args, data=None, **kwargs):
        # 如果没有传入数据，则直接调用原函数
        if data is None:
            return func(
                ax,
                *map(sanitize_sequence, args),
                **{k: sanitize_sequence(v) for k, v in kwargs.items()})

        # 使用新的签名绑定参数和关键字参数
        bound = new_sig.bind(ax, *args, **kwargs)
        
        # 自动获取标签名称
        auto_label = (bound.arguments.get(label_namer)
                      or bound.kwargs.get(label_namer))

        # 遍历绑定的参数和关键字参数
        for k, v in bound.arguments.items():
            # 如果参数名为 varkwargs_name，则遍历其字典值，替换指定名称的数据
            if k == varkwargs_name:
                for k1, v1 in v.items():
                    if replace_names is None or k1 in replace_names:
                        v[k1] = _replacer(data, v1)
            # 如果参数名为 varargs_name，则替换其元组值中的数据
            elif k == varargs_name:
                if replace_names is None:
                    bound.arguments[k] = tuple(_replacer(data, v1) for v1 in v)
            # 对于其它参数名，则替换其数值
            else:
                if replace_names is None or k in replace_names:
                    bound.arguments[k] = _replacer(data, v)

        # 更新绑定后的参数和关键字参数
        new_args = bound.args
        new_kwargs = bound.kwargs

        # 将参数和关键字参数合并到一个字典中
        args_and_kwargs = {**bound.arguments, **bound.kwargs}
        # 如果存在标签生成器并且字典中未包含"label"键，则从标签生成器中生成标签并添加到关键字参数中
        if label_namer and "label" not in args_and_kwargs:
            new_kwargs["label"] = _label_from_arg(
                args_and_kwargs.get(label_namer), auto_label)

        # 调用原函数，传入更新后的参数和关键字参数
        return func(*new_args, **new_kwargs)

    # 将函数的文档字符串更新为包含数据替换的描述
    inner.__doc__ = _add_data_doc(inner.__doc__, replace_names)
    # 将函数的签名更新为新的签名
    inner.__signature__ = new_sig
    # 返回内部函数 inner
    return inner
# 使用调试日志记录交互状态是否为交互式
_log.debug('interactive is %s', is_interactive())
# 使用调试日志记录平台信息
_log.debug('platform is %s', sys.platform)

# 解决方案：必须推迟导入 colormaps 直到加载完 rcParams，因为 colormap 的创建依赖于 rcParams
# 导入 matplotlib.cm 中的 _colormaps 作为 colormaps，并禁止 pylint 错误 E402
from matplotlib.cm import _colormaps as colormaps  # noqa: E402
# 导入 matplotlib.colors 中的 _color_sequences 作为 color_sequences，并禁止 pylint 错误 E402
from matplotlib.colors import _color_sequences as color_sequences  # noqa: E402
```