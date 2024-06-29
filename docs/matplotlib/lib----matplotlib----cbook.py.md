# `D:\src\scipysrc\matplotlib\lib\matplotlib\cbook.py`

```
"""
A collection of utility functions and classes.  Originally, many
(but not all) were from the Python Cookbook -- hence the name cbook.
"""

import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref

import numpy as np

try:
    from numpy.exceptions import VisibleDeprecationWarning  # numpy >= 1.25
except ImportError:
    from numpy import VisibleDeprecationWarning

import matplotlib
from matplotlib import _api, _c_internal_utils


def _get_running_interactive_framework():
    """
    Return the interactive framework whose event loop is currently running, if
    any, or "headless" if no event loop can be started, or None.

    Returns
    -------
    Optional[str]
        One of the following values: "qt", "gtk3", "gtk4", "wx", "tk",
        "macosx", "headless", ``None``.
    """
    # 检查当前是否有 PyQt6、PySide6、PyQt5、PySide2 中的任何一个模块已经导入并且存在 QApplication 实例
    QtWidgets = (
        sys.modules.get("PyQt6.QtWidgets")
        or sys.modules.get("PySide6.QtWidgets")
        or sys.modules.get("PyQt5.QtWidgets")
        or sys.modules.get("PySide2.QtWidgets")
    )
    if QtWidgets and QtWidgets.QApplication.instance():
        return "qt"
    
    # 检查是否有 gi.repository.Gtk 模块已经导入
    Gtk = sys.modules.get("gi.repository.Gtk")
    if Gtk:
        # 如果 Gtk 主版本号为 4，则进一步检查 GLib.main_depth() 是否大于 0
        if Gtk.MAJOR_VERSION == 4:
            from gi.repository import GLib
            if GLib.main_depth():
                return "gtk4"
        # 如果 Gtk 主版本号为 3，则检查 Gtk.main_level() 是否大于 0
        if Gtk.MAJOR_VERSION == 3 and Gtk.main_level():
            return "gtk3"
    
    # 检查是否有 wx 模块已经导入并且存在 wx.App 实例
    wx = sys.modules.get("wx")
    if wx and wx.GetApp():
        return "wx"
    
    # 检查是否有 tkinter 模块已经导入并且存在与 tkinter.mainloop 或 tkinter.Misc.mainloop 相关联的帧
    tkinter = sys.modules.get("tkinter")
    if tkinter:
        codes = {tkinter.mainloop.__code__, tkinter.Misc.mainloop.__code__}
        for frame in sys._current_frames().values():
            while frame:
                if frame.f_code in codes:
                    return "tk"
                frame = frame.f_back
        # 释放帧的引用，防止循环引用
        del frame
    
    # 检查是否有 matplotlib.backends._macosx 模块已经导入并且事件循环正在运行
    macosx = sys.modules.get("matplotlib.backends._macosx")
    if macosx and macosx.event_loop_is_running():
        return "macosx"
    
    # 检查显示是否有效，如果无法确定，则返回 "headless"
    if not _c_internal_utils.display_is_valid():
        return "headless"
    
    # 如果以上条件都不满足，则返回 None
    return None


def _exception_printer(exc):
    """
    Print the exception traceback if an interactive framework is running; otherwise, raise the exception.

    Parameters
    ----------
    exc : Exception
        The exception to be printed.

    Returns
    -------
    None
    """
    # 如果当前没有交互式框架运行或者是 headless 模式，则抛出异常
    if _get_running_interactive_framework() in ["headless", None]:
        raise exc
    else:
        # 否则打印异常的堆栈信息
        traceback.print_exc()


class _StrongRef:
    """
    Wrapper similar to a weakref, but keeping a strong reference to the object.
    """

    def __init__(self, obj):
        self._obj = obj

    def __call__(self):
        return self._obj

    def __eq__(self, other):
        return isinstance(other, _StrongRef) and self._obj == other._obj

    def __hash__(self):
        return hash(self._obj)


def _weak_or_strong_ref(func, callback):
    """
    Return either a weak reference or a strong reference to the function object.

    Parameters
    ----------
    func : callable
        The function object to be referenced.
    callback : callable
        The callback function.

    Returns
    -------
    callable
        Either a weak reference or a strong reference to the function object.
    """
    """
    尝试返回一个 `WeakMethod` 对象来包装函数 *func*，如果不可能，则返回一个 `_StrongRef` 对象。
    """
    # 尝试使用 weakref 创建一个 WeakMethod 对象来弱引用函数 func，并指定回调函数 callback
    try:
        return weakref.WeakMethod(func, callback)
    # 如果类型错误发生，则说明无法创建 WeakMethod 对象，返回一个 _StrongRef 对象来强引用函数 func
    except TypeError:
        return _StrongRef(func)
    """
    Handle registering, processing, blocking, and disconnecting
    for a set of signals and callbacks:

        >>> def oneat(x):
        ...     print('eat', x)
        >>> def ondrink(x):
        ...     print('drink', x)

        >>> from matplotlib.cbook import CallbackRegistry
        >>> callbacks = CallbackRegistry()

        >>> id_eat = callbacks.connect('eat', oneat)
        >>> id_drink = callbacks.connect('drink', ondrink)

        >>> callbacks.process('drink', 123)
        drink 123
        >>> callbacks.process('eat', 456)
        eat 456
        >>> callbacks.process('be merry', 456)   # nothing will be called

        >>> callbacks.disconnect(id_eat)
        >>> callbacks.process('eat', 456)        # nothing will be called

        >>> with callbacks.blocked(signal='drink'):
        ...     callbacks.process('drink', 123)  # nothing will be called
        >>> callbacks.process('drink', 123)
        drink 123

    In practice, one should always disconnect all callbacks when they are
    no longer needed to avoid dangling references (and thus memory leaks).
    However, real code in Matplotlib rarely does so, and due to its design,
    it is rather difficult to place this kind of code.  To get around this,
    and prevent this class of memory leaks, we instead store weak references
    to bound methods only, so when the destination object needs to die, the
    CallbackRegistry won't keep it alive.

    Parameters
    ----------
    exception_handler : callable, optional
       If not None, *exception_handler* must be a function that takes an
       `Exception` as single parameter.  It gets called with any `Exception`
       raised by the callbacks during `CallbackRegistry.process`, and may
       either re-raise the exception or handle it in another manner.

       The default handler prints the exception (with `traceback.print_exc`) if
       an interactive event loop is running; it re-raises the exception if no
       interactive event loop is running.

    signals : list, optional
        If not None, *signals* is a list of signals that this registry handles:
        attempting to `process` or to `connect` to a signal not in the list
        throws a `ValueError`.  The default, None, does not restrict the
        handled signals.
    """

    # We maintain two mappings:
    #   callbacks: signal -> {cid -> weakref-to-callback}
    #   _func_cid_map: signal -> {weakref-to-callback -> cid}

    def __init__(self, exception_handler=_exception_printer, *, signals=None):
        # Copy the signals list if provided, else set it to None
        self._signals = None if signals is None else list(signals)  # Copy it.
        # Set the exception handler function
        self.exception_handler = exception_handler
        # Initialize an empty dictionary to store callbacks for each signal
        self.callbacks = {}
        # Initialize a generator to generate unique callback IDs
        self._cid_gen = itertools.count()
        # Initialize an empty dictionary to map weak references to callback IDs
        self._func_cid_map = {}
        # A set to keep track of callback IDs that need to be pickled
        self._pickled_cids = set()
    # 返回对象的状态的字典表示
    def __getstate__(self):
        return {
            **vars(self),  # 将对象的所有实例变量展开为字典的一部分
            # 一般情况下，回调函数可能无法被序列化，因此我们会丢弃它们，
            # 除非 self._pickled_cids 指示要保留。
            "callbacks": {s: {cid: proxy() for cid, proxy in d.items()
                              if cid in self._pickled_cids}
                          for s, d in self.callbacks.items()},
            # 在 __setstate__ 中更容易从 callbacks 重建此值。
            "_func_cid_map": None,
            "_cid_gen": next(self._cid_gen)  # 生成下一个回调 ID
        }

    # 设置对象的状态，使用传入的状态字典
    def __setstate__(self, state):
        cid_count = state.pop('_cid_gen')  # 弹出回调 ID 计数器
        vars(self).update(state)  # 更新对象的实例变量
        # 重建回调函数字典
        self.callbacks = {
            s: {cid: _weak_or_strong_ref(func, self._remove_proxy)
                for cid, func in d.items()}
            for s, d in self.callbacks.items()}
        # 重建回调函数到 CID 的映射字典
        self._func_cid_map = {
            s: {proxy: cid for cid, proxy in d.items()}
            for s, d in self.callbacks.items()}
        # 重建回调 ID 生成器
        self._cid_gen = itertools.count(cid_count)

    # 连接信号和回调函数
    def connect(self, signal, func):
        """Register *func* to be called when signal *signal* is generated."""
        if self._signals is not None:
            _api.check_in_list(self._signals, signal=signal)  # 检查信号是否在预期的列表中
        self._func_cid_map.setdefault(signal, {})  # 设置默认的信号到回调 ID 映射
        proxy = _weak_or_strong_ref(func, self._remove_proxy)  # 创建回调函数的弱引用或强引用
        if proxy in self._func_cid_map[signal]:
            return self._func_cid_map[signal][proxy]  # 如果已经存在相同的回调函数，则返回对应的 CID
        cid = next(self._cid_gen)  # 生成新的回调 ID
        self._func_cid_map[signal][proxy] = cid  # 将回调函数和 CID 添加到映射中
        self.callbacks.setdefault(signal, {})  # 如果信号不存在于回调函数字典中，则设置默认值
        self.callbacks[signal][cid] = proxy  # 将回调函数和其 CID 添加到回调函数字典中
        return cid  # 返回分配给回调函数的 CID

    # 类似于 `.connect`，但在序列化和反序列化时保留回调函数
    # 目前仅供内部使用
    def _connect_picklable(self, signal, func):
        cid = self.connect(signal, func)  # 连接信号和回调函数
        self._pickled_cids.add(cid)  # 将 CID 添加到可序列化 CID 集合中
        return cid  # 返回分配给回调函数的 CID

    # 在这一点上保留对 sys.is_finalizing 的引用，因为此时 sys 可能已经被清除
    # 无法适当地撤销弱引用
    def _remove_proxy(self, proxy, *, _is_finalizing=sys.is_finalizing):
        if _is_finalizing():
            # 在这一点上弱引用无法适当地撤销
            return
        for signal, proxy_to_cid in list(self._func_cid_map.items()):
            cid = proxy_to_cid.pop(proxy, None)  # 从映射中删除对应的回调函数和 CID
            if cid is not None:
                del self.callbacks[signal][cid]  # 从回调函数字典中删除对应的回调函数和 CID
                self._pickled_cids.discard(cid)  # 从可序列化 CID 集合中删除 CID
                break
        else:
            # 如果未找到对应的弱引用，则直接返回
            return
        # 清理空的字典
        if len(self.callbacks[signal]) == 0:
            del self.callbacks[signal]  # 删除空的回调函数字典条目
            del self._func_cid_map[signal]  # 删除空的映射条目
    def disconnect(self, cid):
        """
        Disconnect the callback registered with callback id *cid*.

        No error is raised if such a callback does not exist.
        """
        # Remove the callback id from the set of pickled callback ids
        self._pickled_cids.discard(cid)
        
        # Clean up callbacks associated with the given cid
        for signal, cid_to_proxy in list(self.callbacks.items()):
            proxy = cid_to_proxy.pop(cid, None)
            if proxy is not None:
                break
        else:
            # If no matching proxy found for the cid, return without errors
            return
        
        # Remove the proxy from the _func_cid_map associated with the signal
        proxy_to_cid = self._func_cid_map[signal]
        for current_proxy, current_cid in list(proxy_to_cid.items()):
            if current_cid == cid:
                assert proxy is current_proxy
                del proxy_to_cid[current_proxy]
        
        # Clean up empty dicts in callbacks
        if len(self.callbacks[signal]) == 0:
            del self.callbacks[signal]
            del self._func_cid_map[signal]

    def process(self, s, *args, **kwargs):
        """
        Process signal *s*.

        All of the functions registered to receive callbacks on *s* will be
        called with ``*args`` and ``**kwargs``.
        """
        # Check if _signals is defined and validate the signal
        if self._signals is not None:
            _api.check_in_list(self._signals, signal=s)
        
        # Iterate over a list of references to functions for the given signal
        for ref in list(self.callbacks.get(s, {}).values()):
            func = ref()
            if func is not None:
                try:
                    # Call the function with provided arguments and keyword arguments
                    func(*args, **kwargs)
                # Handle exceptions other than KeyboardInterrupt, SystemExit, and GeneratorExit
                except Exception as exc:
                    if self.exception_handler is not None:
                        self.exception_handler(exc)
                    else:
                        raise

    @contextlib.contextmanager
    def blocked(self, *, signal=None):
        """
        Block callback signals from being processed.

        A context manager to temporarily block/disable callback signals
        from being processed by the registered listeners.

        Parameters
        ----------
        signal : str, optional
            The callback signal to block. The default is to block all signals.
        """
        # Save a reference to the original callbacks dictionary
        orig = self.callbacks
        try:
            if signal is None:
                # If signal is None, empty out the callbacks dictionary
                self.callbacks = {}
            else:
                # If signal is specified, create a new dictionary excluding that signal
                self.callbacks = {k: orig[k] for k in orig if k != signal}
            # Yield control back to the caller
            yield
        finally:
            # Restore the original callbacks dictionary after context manager exits
            self.callbacks = orig
class silent_list(list):
    """
    A list with a short ``repr()``.

    This is meant to be used for a homogeneous list of artists, so that they
    don't cause long, meaningless output.

    Instead of ::

        [<matplotlib.lines.Line2D object at 0x7f5749fed3c8>,
         <matplotlib.lines.Line2D object at 0x7f5749fed4e0>,
         <matplotlib.lines.Line2D object at 0x7f5758016550>]

    one will get ::

        <a list of 3 Line2D objects>

    If ``self.type`` is None, the type name is obtained from the first item in
    the list (if any).
    """

    def __init__(self, type, seq=None):
        # Initialize the silent_list object with a type and optional sequence
        self.type = type
        if seq is not None:
            # Extend the list with elements from seq if provided
            self.extend(seq)

    def __repr__(self):
        # Return a shortened representation of the list
        if self.type is not None or len(self) != 0:
            tp = self.type if self.type is not None else type(self[0]).__name__
            return f"<a list of {len(self)} {tp} objects>"
        else:
            return "<an empty list>"


def _local_over_kwdict(
        local_var, kwargs, *keys,
        warning_cls=_api.MatplotlibDeprecationWarning):
    # Manage keyword arguments, preferring local_var over kwargs
    out = local_var
    for key in keys:
        kwarg_val = kwargs.pop(key, None)
        if kwarg_val is not None:
            if out is None:
                out = kwarg_val
            else:
                # Issue a warning about ignored keyword arguments
                _api.warn_external(f'"{key}" keyword argument will be ignored',
                                   warning_cls)
    return out


def strip_math(s):
    """
    Remove latex formatting from mathtext.

    Only handles fully math and fully non-math strings.
    """
    # Remove LaTeX formatting from the string s
    if len(s) >= 2 and s[0] == s[-1] == "$":
        s = s[1:-1]
        for tex, plain in [
                (r"\times", "x"),  # Specifically for Formatter support.
                (r"\mathdefault", ""),
                (r"\rm", ""),
                (r"\cal", ""),
                (r"\tt", ""),
                (r"\it", ""),
                ("\\", ""),
                ("{", ""),
                ("}", ""),
        ]:
            s = s.replace(tex, plain)
    return s


def _strip_comment(s):
    """Strip everything from the first unquoted #."""
    pos = 0
    while True:
        quote_pos = s.find('"', pos)
        hash_pos = s.find('#', pos)
        if quote_pos < 0:
            # Return the string up to the first unquoted '#' or the entire string if no '#'
            without_comment = s if hash_pos < 0 else s[:hash_pos]
            return without_comment.strip()
        elif 0 <= hash_pos < quote_pos:
            # Return the string up to the first unquoted '#'
            return s[:hash_pos].strip()
        else:
            # Move past the closing quote to continue searching
            closing_quote_pos = s.find('"', quote_pos + 1)
            if closing_quote_pos < 0:
                raise ValueError(
                    f"Missing closing quote in: {s!r}. If you need a double-"
                    'quote inside a string, use escaping: e.g. "the \" char"')
            pos = closing_quote_pos + 1  # move past closing quote


def is_writable_file_like(obj):
    """Return whether *obj* looks like a file object with a *write* method."""
    # Check if obj has a callable 'write' attribute
    return callable(getattr(obj, 'write', None))


def file_requires_unicode(x):
    """
    # 尝试向给定的可写文件类对象写入空字节串，以检测是否支持 Unicode 写入
    try:
        x.write(b'')
    # 捕获可能的 TypeError 异常，表明对象不支持二进制写入，可能需要 Unicode
    except TypeError:
        # 如果捕获到 TypeError 异常，则返回 True，表示需要使用 Unicode 写入
        return True
    else:
        # 如果未捕获到异常，则返回 False，表示对象支持二进制写入，不需要使用 Unicode
        return False
# 将路径转换为打开的文件句柄或通过文件样式对象传递。

# 考虑使用 `open_file_cm` 替代，因为它可以更轻松地关闭新创建的文件对象。

def to_filehandle(fname, flag='r', return_opened=False, encoding=None):
    """
    Convert a path to an open file handle or pass-through a file-like object.

    Parameters
    ----------
    fname : str or path-like or file-like
        如果是 `str` 或 `os.PathLike`，则使用 *flag* 和 *encoding* 指定的标志打开文件。
        如果是文件样式对象，则直接传递。
    flag : str, default: 'r'
        当 *fname* 是 `str` 或 `os.PathLike` 时传递给 `open` 的模式参数；如果 *fname* 是文件样式对象，则忽略。
    return_opened : bool, default: False
        如果为 True，则返回文件对象和一个布尔值，指示是否是新文件（调用者需要关闭）。如果为 False，则仅返回新文件对象。
    encoding : str or None, default: None
        当 *fname* 是 `str` 或 `os.PathLike` 时传递给 `open` 的编码参数；如果 *fname* 是文件样式对象，则忽略。

    Returns
    -------
    fh : file-like
    opened : bool
        仅当 *return_opened* 为 True 时返回 *opened*。
    """
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)  # 如果是 PathLike 对象，将其转换为路径字符串
    if isinstance(fname, str):
        if fname.endswith('.gz'):
            fh = gzip.open(fname, flag)  # 如果文件名以 .gz 结尾，使用 gzip 打开
        elif fname.endswith('.bz2'):
            # Python 可能没有编译支持 bz2，
            # 直到需要时才引入模块
            import bz2
            fh = bz2.BZ2File(fname, flag)  # 如果文件名以 .bz2 结尾，使用 bz2 打开
        else:
            fh = open(fname, flag, encoding=encoding)  # 否则按指定的 flag 和 encoding 打开文件
        opened = True  # 标记文件已打开
    elif hasattr(fname, 'seek'):
        fh = fname  # 如果 fname 具有 seek 属性，认为它是一个文件样式对象，直接使用它
        opened = False  # 标记文件未打开
    else:
        raise ValueError('fname must be a PathLike or file handle')  # 抛出异常，fname 必须是 PathLike 或文件句柄
    if return_opened:
        return fh, opened  # 如果需要返回是否已打开的标志，则一起返回
    return fh  # 否则只返回文件对象


def open_file_cm(path_or_file, mode="r", encoding=None):
    r"""Pass through file objects and context-manage path-likes."""
    fh, opened = to_filehandle(path_or_file, mode, True, encoding)
    return fh if opened else contextlib.nullcontext(fh)


def is_scalar_or_string(val):
    """Return whether the given object is a scalar or string like."""
    return isinstance(val, str) or not np.iterable(val)


@_api.delete_parameter(
    "3.8", "np_load", alternative="open(get_sample_data(..., asfileobj=False))")
def get_sample_data(fname, asfileobj=True, *, np_load=True):
    """
    Return a sample data file.  *fname* is a path relative to the
    :file:`mpl-data/sample_data` directory.  If *asfileobj* is `True`
    return a file object, otherwise just a file path.

    Sample data files are stored in the 'mpl-data/sample_data' directory within
    the Matplotlib package.

    If the filename ends in .gz, the file is implicitly ungzipped.  If the
    filename ends with .npy or .npz, and *asfileobj* is `True`, the file is
    loaded with `numpy.load`.
    """
    # 构造数据路径，使用_get_data_path函数获取完整文件路径
    path = _get_data_path('sample_data', fname)
    # 如果需要返回文件对象
    if asfileobj:
        # 获取文件后缀并转换为小写
        suffix = path.suffix.lower()
        # 如果是.gz压缩文件，返回gzip打开的文件对象
        if suffix == '.gz':
            return gzip.open(path)
        # 如果是.npy或者.npz文件
        elif suffix in ['.npy', '.npz']:
            # 如果需要使用numpy加载
            if np_load:
                return np.load(path)
            else:
                # 否则返回二进制只读文件对象
                return path.open('rb')
        # 如果是.csv, .xrc或者.txt文件，返回文本文件读取对象
        elif suffix in ['.csv', '.xrc', '.txt']:
            return path.open('r')
        else:
            # 否则返回二进制只读文件对象
            return path.open('rb')
    else:
        # 否则返回文件路径的字符串形式
        return str(path)
# 返回指向 Matplotlib 提供的资源文件的 `pathlib.Path` 对象路径
def _get_data_path(*args):
    """
    Return the `pathlib.Path` to a resource file provided by Matplotlib.

    ``*args`` specify a path relative to the base data path.
    """
    return Path(matplotlib.get_data_path(), *args)


# 生成一个展开嵌套容器的生成器
def flatten(seq, scalarp=is_scalar_or_string):
    """
    Return a generator of flattened nested containers.

    For example:

        >>> from matplotlib.cbook import flatten
        >>> l = (('John', ['Hunter']), (1, 23), [[([42, (5, 23)], )]])
        >>> print(list(flatten(l)))
        ['John', 'Hunter', 1, 23, 42, 5, 23]

    By: Composite of Holger Krekel and Luther Blissett
    From: https://code.activestate.com/recipes/121294/
    and Recipe 1.12 in cookbook
    """
    # 遍历序列中的每个元素
    for item in seq:
        # 如果元素是标量或字符串，或者为 None，则直接生成
        if scalarp(item) or item is None:
            yield item
        else:
            # 如果元素是容器，则递归展开其元素并生成
            yield from flatten(item, scalarp)


# 使用 @_api.deprecated 装饰器标记为在版本 3.8 后废弃
@_api.deprecated("3.8")
class Stack:
    """
    Stack of elements with a movable cursor.

    Mimics home/back/forward in a web browser.
    """

    def __init__(self, default=None):
        # 初始化空栈并设置默认元素
        self.clear()
        self._default = default

    def __call__(self):
        """Return the current element, or None."""
        # 如果栈为空，则返回默认元素，否则返回当前位置的元素
        if not self._elements:
            return self._default
        else:
            return self._elements[self._pos]

    def __len__(self):
        # 返回栈的元素个数
        return len(self._elements)

    def __getitem__(self, ind):
        # 获取栈中指定位置的元素
        return self._elements[ind]

    def forward(self):
        """Move the position forward and return the current element."""
        # 将位置向前移动一位，并返回当前位置的元素
        self._pos = min(self._pos + 1, len(self._elements) - 1)
        return self()

    def back(self):
        """Move the position back and return the current element."""
        # 将位置向后移动一位，并返回当前位置的元素
        if self._pos > 0:
            self._pos -= 1
        return self()

    def push(self, o):
        """
        Push *o* to the stack at current position.  Discard all later elements.

        *o* is returned.
        """
        # 将元素 o 推入栈中当前位置，丢弃后续所有元素
        self._elements = self._elements[:self._pos + 1] + [o]
        self._pos = len(self._elements) - 1
        return self()

    def home(self):
        """
        Push the first element onto the top of the stack.

        The first element is returned.
        """
        # 将第一个元素推入栈顶，并返回该元素
        if not self._elements:
            return
        self.push(self._elements[0])
        return self()

    def empty(self):
        """Return whether the stack is empty."""
        # 返回栈是否为空
        return len(self._elements) == 0

    def clear(self):
        """Empty the stack."""
        # 清空栈
        self._pos = -1
        self._elements = []
    # 将对象 *o* 的所有引用移到堆栈顶部，并返回该对象。

    # 如果 *o* 不在堆栈中，则抛出 ValueError 异常。
    def bubble(self, o):
        if o not in self._elements:
            raise ValueError('Given element not contained in the stack')
        
        # 备份当前堆栈元素
        old_elements = self._elements.copy()
        
        # 清空堆栈
        self.clear()
        
        # 收集所有与 *o* 相同的元素，将它们推入堆栈顶部
        top_elements = []
        for elem in old_elements:
            if elem == o:
                top_elements.append(elem)
            else:
                self.push(elem)
        
        # 将 *o* 推入堆栈顶部，根据它在堆栈中的出现次数重复推入
        for _ in top_elements:
            self.push(o)
        
        # 返回 *o* 对象
        return o

    # 从堆栈中移除对象 *o*。

    # 如果 *o* 不在堆栈中，则抛出 ValueError 异常。
    def remove(self, o):
        if o not in self._elements:
            raise ValueError('Given element not contained in the stack')
        
        # 备份当前堆栈元素
        old_elements = self._elements.copy()
        
        # 清空堆栈
        self.clear()
        
        # 将除了 *o* 以外的所有元素推入堆栈
        for elem in old_elements:
            if elem != o:
                self.push(elem)
class _Stack:
    """
    Stack of elements with a movable cursor.

    Mimics home/back/forward in a web browser.
    """

    def __init__(self):
        # 初始化栈的位置为-1（空栈）和空元素列表
        self._pos = -1
        self._elements = []

    def clear(self):
        """Empty the stack."""
        # 清空栈，将位置重置为-1并清空元素列表
        self._pos = -1
        self._elements = []

    def __call__(self):
        """Return the current element, or None."""
        # 返回当前位置的元素，如果栈为空则返回None
        return self._elements[self._pos] if self._elements else None

    def __len__(self):
        # 返回栈中元素的数量
        return len(self._elements)

    def __getitem__(self, ind):
        # 获取指定位置的元素
        return self._elements[ind]

    def forward(self):
        """Move the position forward and return the current element."""
        # 将位置向前移动一步，并返回当前位置的元素
        self._pos = min(self._pos + 1, len(self._elements) - 1)
        return self()

    def back(self):
        """Move the position back and return the current element."""
        # 将位置向后移动一步，并返回当前位置的元素
        self._pos = max(self._pos - 1, 0)
        return self()

    def push(self, o):
        """
        Push *o* to the stack after the current position, and return *o*.

        Discard all later elements.
        """
        # 将元素 *o* 推入栈中当前位置后面，并返回 *o*
        # 同时丢弃后面的所有元素
        self._elements[self._pos + 1:] = [o]
        self._pos = len(self._elements) - 1
        return o

    def home(self):
        """
        Push the first element onto the top of the stack.

        The first element is returned.
        """
        # 将第一个元素推入栈顶，并返回该元素
        return self.push(self._elements[0]) if self._elements else None


def safe_masked_invalid(x, copy=False):
    x = np.array(x, subok=True, copy=copy)
    if not x.dtype.isnative:
        # 如果数组不是本机字节顺序，则进行字节交换以转换为本机字节顺序
        x = x.byteswap(inplace=copy).view(x.dtype.newbyteorder('N'))
    try:
        # 尝试创建一个掩码，标记所有非有限数值为无效
        xm = np.ma.masked_where(~(np.isfinite(x)), x, copy=False)
    except TypeError:
        return x
    return xm


def print_cycles(objects, outstream=sys.stdout, show_progress=False):
    """
    Print loops of cyclic references in the given *objects*.

    It is often useful to pass in ``gc.garbage`` to find the cycles that are
    preventing some objects from being garbage collected.

    Parameters
    ----------
    objects
        A list of objects to find cycles in.
    outstream
        The stream for output.
    show_progress : bool
        If True, print the number of objects reached as they are found.
    """
    import gc
    # 在给定对象中打印循环引用的环，通常用于查找阻止某些对象被垃圾回收的循环引用
    # 可以传入 gc.garbage 来查找这些循环引用
    def print_path(path):
        # 遍历路径中的每一步骤
        for i, step in enumerate(path):
            # 下一个步骤的索引，使用取模运算实现循环
            next = path[(i + 1) % len(path)]

            # 输出当前步骤的类型信息
            outstream.write("   %s -- " % type(step))
            # 如果当前步骤是字典类型
            if isinstance(step, dict):
                # 遍历字典中的键值对
                for key, val in step.items():
                    # 如果值是下一个步骤，则输出键作为索引
                    if val is next:
                        outstream.write(f"[{key!r}]")
                        break
                    # 如果键是下一个步骤，则输出键值对
                    if key is next:
                        outstream.write(f"[{key}] = {val!r}")
                        break
            # 如果当前步骤是列表类型
            elif isinstance(step, list):
                # 输出下一个步骤在列表中的索引
                outstream.write("[%d]" % step.index(next))
            # 如果当前步骤是元组类型
            elif isinstance(step, tuple):
                # 输出元组类型的标记
                outstream.write("( tuple )")
            else:
                # 其他类型直接输出表示
                outstream.write(repr(step))
            # 输出箭头连接符号
            outstream.write(" ->\n")
        # 输出换行符号
        outstream.write("\n")

    def recurse(obj, start, all, current_path):
        # 如果需要展示进度，则输出当前处理的对象总数
        if show_progress:
            outstream.write("%d\r" % len(all))

        # 将当前对象加入到已处理的对象字典中
        all[id(obj)] = None

        # 获取当前对象的所有引用对象
        referents = gc.get_referents(obj)
        for referent in referents:
            # 如果找到回到起点的路径，则打印路径信息
            if referent is start:
                print_path(current_path)

            # 不递归处理原始对象列表或临时引用的对象
            elif referent is objects or isinstance(referent, types.FrameType):
                continue

            # 如果未曾处理过该引用对象，则递归处理
            elif id(referent) not in all:
                recurse(referent, start, all, current_path + [obj])

    # 遍历给定的对象列表
    for obj in objects:
        # 输出当前处理的对象信息
        outstream.write(f"Examining: {obj!r}\n")
        # 从当前对象开始递归处理其引用对象
        recurse(obj, obj, {}, [])
    """
    A disjoint-set data structure.

    Objects can be joined using :meth:`join`, tested for connectedness
    using :meth:`joined`, and all disjoint sets can be retrieved by
    using the object as an iterator.

    The objects being joined must be hashable and weak-referenceable.

    Examples
    --------
    >>> from matplotlib.cbook import Grouper
    >>> class Foo:
    ...     def __init__(self, s):
    ...         self.s = s
    ...     def __repr__(self):
    ...         return self.s
    ...
    >>> a, b, c, d, e, f = [Foo(x) for x in 'abcdef']
    >>> grp = Grouper()
    >>> grp.join(a, b)
    >>> grp.join(b, c)
    >>> grp.join(d, e)
    >>> list(grp)
    [[a, b, c], [d, e]]
    >>> grp.joined(a, b)
    True
    >>> grp.joined(a, c)
    True
    >>> grp.joined(a, d)
    False
    """

    def __init__(self, init=()):
        # 初始化一个弱引用字典，将每个元素映射到其自身的弱引用集合
        self._mapping = weakref.WeakKeyDictionary(
            {x: weakref.WeakSet([x]) for x in init})
        # 初始化一个空的弱引用字典，用于保持元素的顺序
        self._ordering = weakref.WeakKeyDictionary()
        # 遍历初始元素，将其加入到顺序字典中
        for x in init:
            if x not in self._ordering:
                self._ordering[x] = len(self._ordering)
        # 初始化下一个顺序号，用于简化序列化操作
        self._next_order = len(self._ordering)  # Plain int to simplify pickling.

    def __getstate__(self):
        # 返回当前对象的状态，将弱引用转换为强引用
        return {
            **vars(self),
            "_mapping": {k: set(v) for k, v in self._mapping.items()},
            "_ordering": {**self._ordering},
        }

    def __setstate__(self, state):
        # 从状态中恢复对象的属性
        vars(self).update(state)
        # 将强引用转换为弱引用
        self._mapping = weakref.WeakKeyDictionary(
            {k: weakref.WeakSet(v) for k, v in self._mapping.items()})
        self._ordering = weakref.WeakKeyDictionary(self._ordering)

    def __contains__(self, item):
        # 检查元素是否在映射中
        return item in self._mapping

    @_api.deprecated("3.8", alternative="none, you no longer need to clean a Grouper")
    def clean(self):
        """Clean dead weak references from the dictionary."""

    def join(self, a, *args):
        """
        Join given arguments into the same set.  Accepts one or more arguments.
        """
        mapping = self._mapping
        try:
            # 获取元素a的弱引用集合，若不存在则创建新的弱引用集合
            set_a = mapping[a]
        except KeyError:
            set_a = mapping[a] = weakref.WeakSet([a])
            # 若元素a未在顺序字典中，则加入并分配下一个顺序号
            self._ordering[a] = self._next_order
            self._next_order += 1
        for arg in args:
            try:
                # 获取参数arg的弱引用集合，若不存在则创建新的弱引用集合
                set_b = mapping[arg]
            except KeyError:
                set_b = mapping[arg] = weakref.WeakSet([arg])
                # 若参数arg未在顺序字典中，则加入并分配下一个顺序号
                self._ordering[arg] = self._next_order
                self._next_order += 1
            # 若set_b不等于set_a，则将set_b合并到set_a中，并更新映射关系
            if set_b is not set_a:
                if len(set_b) > len(set_a):
                    set_a, set_b = set_b, set_a
                set_a.update(set_b)
                for elem in set_b:
                    mapping[elem] = set_a
    # 判断元素 *a* 和 *b* 是否属于同一个集合
    def joined(self, a, b):
        """Return whether *a* and *b* are members of the same set."""
        return (self._mapping.get(a, object()) is self._mapping.get(b))

    # 从分组器中移除元素 *a*，如果元素不在分组器中则什么也不做
    def remove(self, a):
        """Remove *a* from the grouper, doing nothing if it is not there."""
        # 从映射中删除元素 *a*，如果不存在则返回 {a} 集合，并从集合中移除 *a*
        self._mapping.pop(a, {a}).remove(a)
        # 从排序映射中移除元素 *a*
        self._ordering.pop(a, None)

    # 迭代器，遍历每个不相交集合作为列表
    # 如果在调用 join() 方法期间调用，迭代器会变为无效状态
    def __iter__(self):
        """
        Iterate over each of the disjoint sets as a list.

        The iterator is invalid if interleaved with calls to join().
        """
        # 使用集合推导式创建一个包含唯一集合的字典
        unique_groups = {id(group): group for group in self._mapping.values()}
        # 对每个集合按照 _ordering 映射的值排序后进行迭代
        for group in unique_groups.values():
            yield sorted(group, key=self._ordering.__getitem__)

    # 返回与 *a* 相连接的所有项，包括 *a* 本身
    def get_siblings(self, a):
        """Return all of the items joined with *a*, including itself."""
        # 获取与 *a* 相连接的所有项，如果 *a* 不在映射中，则返回包含 *a* 的列表
        siblings = self._mapping.get(a, [a])
        # 根据 _ordering 映射的值排序返回结果
        return sorted(siblings, key=self._ordering.get)
class GrouperView:
    """Immutable view over a `.Grouper`."""

    # 初始化方法，接受一个 grouper 对象作为参数，并将其保存在 _grouper 属性中
    def __init__(self, grouper): self._grouper = grouper
    
    # 实现 __contains__ 方法，判断传入的 item 是否在 _grouper 中
    def __contains__(self, item): return item in self._grouper
    
    # 实现 __iter__ 方法，返回 _grouper 的迭代器
    def __iter__(self): return iter(self._grouper)
    
    # 返回 _grouper 对象的 joined 方法的结果
    def joined(self, a, b): return self._grouper.joined(a, b)
    
    # 返回 _grouper 对象的 get_siblings 方法的结果
    def get_siblings(self, a): return self._grouper.get_siblings(a)


def simple_linear_interpolation(a, steps):
    """
    Resample an array with ``steps - 1`` points between original point pairs.

    Along each column of *a*, ``(steps - 1)`` points are introduced between
    each original values; the values are linearly interpolated.

    Parameters
    ----------
    a : array, shape (n, ...)
        Input array with shape (n, ...).
    steps : int
        Number of steps for interpolation between each pair of points.

    Returns
    -------
    array
        Resampled array with shape ``((n - 1) * steps + 1, ...)``.
    """
    # Reshape input array a into a 2-dimensional array
    fps = a.reshape((len(a), -1))
    
    # Create an array of x positions for the original data points
    xp = np.arange(len(a)) * steps
    
    # Create an array of x positions for the interpolated data points
    x = np.arange((len(a) - 1) * steps + 1)
    
    # Perform linear interpolation along each column of fps
    return (np.column_stack([np.interp(x, xp, fp) for fp in fps.T])
            .reshape((len(x),) + a.shape[1:]))


def delete_masked_points(*args):
    """
    Find all masked and/or non-finite points in a set of arguments,
    and return the arguments with only the unmasked points remaining.

    Arguments can be in any of 5 categories:

    1) 1-D masked arrays
    2) 1-D ndarrays
    3) ndarrays with more than one dimension
    4) other non-string iterables
    5) anything else

    The first argument must be in one of the first four categories;
    any argument with a length differing from that of the first
    argument (and hence anything in category 5) then will be
    passed through unchanged.

    Masks are obtained from all arguments of the correct length
    in categories 1, 2, and 4; a point is bad if masked in a masked
    array or if it is a nan or inf.  No attempt is made to
    extract a mask from categories 2, 3, and 4 if `numpy.isfinite`
    does not yield a Boolean array.

    All input arguments that are not passed unchanged are returned
    as ndarrays after removing the points or rows corresponding to
    masks in any of the arguments.

    A vastly simpler version of this function was originally
    written as a helper for Axes.scatter().

    """
    # 如果没有参数传入，则返回空元组
    if not len(args):
        return ()
    
    # 如果第一个参数是标量或者字符串，则抛出异常
    if is_scalar_or_string(args[0]):
        raise ValueError("First argument must be a sequence")
    
    # 获取第一个参数的长度
    nrecs = len(args[0])
    
    # 存放处理后的参数的列表
    margs = []
    
    # 判断每个参数是否应该处理，初始化为 False 的列表
    seqlist = [False] * len(args)
    
    # 遍历参数列表
    for i, x in enumerate(args):
        # 检查是否为可迭代对象且长度与第一个参数相同
        if not isinstance(x, str) and np.iterable(x) and len(x) == nrecs:
            seqlist[i] = True
            # 如果是 masked array，则检查是否是 1 维的
            if isinstance(x, np.ma.MaskedArray):
                if x.ndim > 1:
                    raise ValueError("Masked arrays must be 1-D")
            else:
                # 转换为 ndarray
                x = np.asarray(x)
        # 将处理后的参数添加到 margs 中
        margs.append(x)
    
    # 存放所有掩码的列表，用来表示有效数据点的位置
    masks = []
    # 遍历 margs 列表，同时获取索引 i 和元素 x
    for i, x in enumerate(margs):
        # 如果 seqlist[i] 为真，则执行以下条件语句块
        if seqlist[i]:
            # 如果 x 的维度大于1，则跳过当前循环，不处理 nan 位置
            if x.ndim > 1:
                continue  # 不处理多维数组的情况，避免获取 nan 的位置信息
            # 如果 x 是 np.ma.MaskedArray 类型，则取反其掩码并添加到 masks 列表中
            if isinstance(x, np.ma.MaskedArray):
                masks.append(~np.ma.getmaskarray(x))  # 取反掩码
                xd = x.data  # 获取 MaskedArray 的数据部分
            else:
                xd = x  # 否则直接使用 x 的值
            try:
                # 尝试生成一个标记 x 中有限数值的掩码
                mask = np.isfinite(xd)
                # 如果 mask 是 np.ndarray 类型，则添加到 masks 列表中
                if isinstance(mask, np.ndarray):
                    masks.append(mask)
            except Exception:  # 捕获任何异常，暂时不处理
                pass
    # 如果 masks 列表不为空
    if len(masks):
        # 使用逻辑与操作对 masks 列表中的掩码进行合并
        mask = np.logical_and.reduce(masks)
        # 获取非零掩码的索引值
        igood = mask.nonzero()[0]
        # 如果有效索引值的数量小于 nrecs
        if len(igood) < nrecs:
            # 再次遍历 margs 列表
            for i, x in enumerate(margs):
                # 如果 seqlist[i] 为真，则执行以下条件语句块
                if seqlist[i]:
                    # 将 margs[i] 限制为有效索引 igood 对应的切片
                    margs[i] = x[igood]
    # 再次遍历 margs 列表
    for i, x in enumerate(margs):
        # 如果 seqlist[i] 为真且 x 是 np.ma.MaskedArray 类型
        if seqlist[i] and isinstance(x, np.ma.MaskedArray):
            # 将 margs[i] 中的掩码数据填充为其实际数值
            margs[i] = x.filled()
    # 返回处理后的 margs 列表
    return margs
# 组合多个输入参数，找出所有的掩码和/或非有限点，并返回带有共同掩码的掩码数组。

def _combine_masks(*args):
    """
    Find all masked and/or non-finite points in a set of arguments,
    and return the arguments as masked arrays with a common mask.

    Arguments can be in any of 5 categories:

    1) 1-D masked arrays
    2) 1-D ndarrays
    3) ndarrays with more than one dimension
    4) other non-string iterables
    5) anything else

    The first argument must be in one of the first four categories;
    any argument with a length differing from that of the first
    argument (and hence anything in category 5) then will be
    passed through unchanged.

    Masks are obtained from all arguments of the correct length
    in categories 1, 2, and 4; a point is bad if masked in a masked
    array or if it is a nan or inf.  No attempt is made to
    extract a mask from categories 2 and 4 if `numpy.isfinite`
    does not yield a Boolean array.  Category 3 is included to
    support RGB or RGBA ndarrays, which are assumed to have only
    valid values and which are passed through unchanged.

    All input arguments that are not passed unchanged are returned
    as masked arrays if any masked points are found, otherwise as
    ndarrays.

    """
    # 如果没有参数，则返回空元组
    if not len(args):
        return ()
    # 检查第一个参数是否为标量或字符串
    if is_scalar_or_string(args[0]):
        raise ValueError("First argument must be a sequence")
    # 记录第一个参数的长度
    nrecs = len(args[0])
    margs = []  # 输出参数；有些可能被修改。
    seqlist = [False] * len(args)  # 标志位：如果输出将被掩码，则为True。
    masks = []    # 掩码列表。
    for i, x in enumerate(args):
        # 如果参数是标量或字符串，或者其长度与第一个参数不同，则保持不变。
        if is_scalar_or_string(x) or len(x) != nrecs:
            margs.append(x)  # 保持不变。
        else:
            # 如果参数是MaskedArray且维度大于1，则引发异常。
            if isinstance(x, np.ma.MaskedArray) and x.ndim > 1:
                raise ValueError("Masked arrays must be 1-D")
            try:
                # 将参数转换为任意数组
                x = np.asanyarray(x)
            except (VisibleDeprecationWarning, ValueError):
                # NumPy 1.19 会对不规则数组引发警告，但我们希望在此处接受基本任何类型。
                x = np.asanyarray(x, dtype=object)
            # 如果参数是一维数组，则处理其无效数据并获取其掩码。
            if x.ndim == 1:
                x = safe_masked_invalid(x)
                seqlist[i] = True
                if np.ma.is_masked(x):
                    masks.append(np.ma.getmaskarray(x))
            margs.append(x)  # 可能已修改。
    # 如果存在掩码，则合并它们并将其应用到需要掩码的参数中。
    if len(masks):
        mask = np.logical_or.reduce(masks)
        for i, x in enumerate(margs):
            if seqlist[i]:
                margs[i] = np.ma.array(x, mask=mask)
    # 返回处理后的参数列表。
    return margs


def _broadcast_with_masks(*args, compress=False):
    """
    Broadcast inputs, combining all masked arrays.

    Parameters
    ----------
    *args : array-like
        The inputs to broadcast.
    compress : bool, default: False
        Whether to compress the masked arrays. If False, the masked values
        are replaced by NaNs.

    Returns
    -------
    list of array-like
        The broadcasted and masked inputs.
    """
    # 提取可能存在的掩码（mask）
    masks = [k.mask for k in args if isinstance(k, np.ma.MaskedArray)]
    # 广播以匹配形状
    bcast = np.broadcast_arrays(*args, *masks)
    # 提取输入数据（去除掩码后的数据）
    inputs = bcast[:len(args)]
    # 提取掩码数据
    masks = bcast[len(args):]
    if masks:
        # 将多个掩码合并为一个
        mask = np.logical_or.reduce(masks)
        # 如果指定压缩操作
        if compress:
            # 在输入数据上施加掩码并压缩
            inputs = [np.ma.array(k, mask=mask).compressed()
                      for k in inputs]
        else:
            # 在输入数据上施加掩码，转换为浮点型并填充 NaN 后展平
            inputs = [np.ma.array(k, mask=mask, dtype=float).filled(np.nan).ravel()
                      for k in inputs]
    else:
        # 没有掩码时，直接展平输入数据
        inputs = [np.ravel(k) for k in inputs]
    # 返回处理后的输入数据
    return inputs
def boxplot_stats(X, whis=1.5, bootstrap=None, labels=None, autorange=False):
    r"""
    Return a list of dictionaries of statistics used to draw a series of box
    and whisker plots using `~.Axes.bxp`.

    Parameters
    ----------
    X : array-like
        Data that will be represented in the boxplots. Should have 2 or
        fewer dimensions.

    whis : float or (float, float), default: 1.5
        The position of the whiskers.

        If a float, the lower whisker is at the lowest datum above
        ``Q1 - whis*(Q3-Q1)``, and the upper whisker at the highest datum below
        ``Q3 + whis*(Q3-Q1)``, where Q1 and Q3 are the first and third
        quartiles.  The default value of ``whis = 1.5`` corresponds to Tukey's
        original definition of boxplots.

        If a pair of floats, they indicate the percentiles at which to draw the
        whiskers (e.g., (5, 95)).  In particular, setting this to (0, 100)
        results in whiskers covering the whole range of the data.

        In the edge case where ``Q1 == Q3``, *whis* is automatically set to
        (0, 100) (cover the whole range of the data) if *autorange* is True.

        Beyond the whiskers, data are considered outliers and are plotted as
        individual points.

    bootstrap : int, optional
        Number of times the confidence intervals around the median
        should be bootstrapped (percentile method).

    labels : list of str, optional
        Labels for each dataset. Length must be compatible with
        dimensions of *X*.

    autorange : bool, optional (False)
        When `True` and the data are distributed such that the 25th and 75th
        percentiles are equal, ``whis`` is set to (0, 100) such that the
        whisker ends are at the minimum and maximum of the data.

    Returns
    -------
    list of dict
        A list of dictionaries containing the results for each column
        of data. Keys of each dictionary are the following:

        ========   ===================================
        Key        Value Description
        ========   ===================================
        label      tick label for the boxplot
        mean       arithmetic mean value
        med        50th percentile
        q1         first quartile (25th percentile)
        q3         third quartile (75th percentile)
        iqr        interquartile range
        cilo       lower notch around the median
        cihi       upper notch around the median
        whislo     end of the lower whisker
        whishi     end of the upper whisker
        fliers     outliers
        ========   ===================================

    Notes
    -----
    Non-bootstrapping approach to confidence interval uses Gaussian-based
    asymptotic approximation:

    .. math::

        \mathrm{med} \pm 1.57 \times \frac{\mathrm{iqr}}{\sqrt{N}}

    General approach from:
    McGill, R., Tukey, J.W., and Larsen, W.A. (1978) "Variations of
    """
    # 计算数据 X 的各种统计量并存储到字典列表中
    stats = []

    # 遍历数据的每一列
    for col in range(X.shape[1]):
        # 计算当前列的第一四分位数（Q1）
        q1 = np.percentile(X[:, col], 25)
        # 计算当前列的中位数（第二四分位数，即Q2）
        med = np.median(X[:, col])
        # 计算当前列的第三四分位数（Q3）
        q3 = np.percentile(X[:, col], 75)
        # 计算当前列的四分位距（IQR）
        iqr = q3 - q1

        if bootstrap is not None:
            # 使用自举法计算中位数周围的置信区间
            # 略去具体的自举法计算过程

            cilo = cihi = None
        else:
            # 使用 Gaussian-based asymptotic approximation 计算置信区间
            cilo = med - 1.57 * iqr / np.sqrt(X.shape[0])
            cihi = med + 1.57 * iqr / np.sqrt(X.shape[0])

        # 计算边缘的下限和上限
        if np.ndim(whis) == 0:
            loval = q1 - whis * iqr
            hival = q3 + whis * iqr
        else:
            loval = np.percentile(X[:, col], whis[0])
            hival = np.percentile(X[:, col], whis[1])

        # 计算异常值（fliers）
        fliers = X[(X[:, col] < loval) | (X[:, col] > hival), col]

        # 创建包含当前列统计信息的字典
        stat = {
            'label': labels[col] if labels is not None else None,
            'mean': np.mean(X[:, col]),
            'med': med,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'cilo': cilo,
            'cihi': cihi,
            'whislo': loval,
            'whishi': hival,
            'fliers': fliers
        }

        # 将当前列的统计信息字典加入到统计列表中
        stats.append(stat)

    # 返回包含所有列统计信息的列表
    return stats
    """
    根据"Comparing Location Estimates: Trimmed Means, Medians, and
    Boxplots", The American Statistician, 32:12-16.
    """

    # 定义一个内部函数，用于计算数据的中位数的置信区间
    def _bootstrap_median(data, N=5000):
        # 计算中位数的95%置信区间
        M = len(data)
        percentiles = [2.5, 97.5]

        # 从数据中随机采样生成 bootstrap 样本索引
        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
        # 计算每个 bootstrap 样本的中位数
        estimate = np.median(bsData, axis=1, overwrite_input=True)

        # 计算中位数的置信区间
        CI = np.percentile(estimate, percentiles)
        return CI

    # 定义一个内部函数，用于计算数据的置信区间
    def _compute_conf_interval(data, med, iqr, bootstrap):
        if bootstrap is not None:
            # 使用 bootstrap 方法估计 notch 位置的置信区间
            # 获取围绕中位数的置信区间
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:
            # 计算不使用 bootstrap 方法时的 notch 位置的置信区间
            N = len(data)
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)

        return notch_min, notch_max

    # 输出是一个字典列表
    bxpstats = []

    # 将 X 转换为列表的列表
    X = _reshape_2D(X, "X")

    # 计算 X 的列数
    ncols = len(X)
    if labels is None:
        # 如果标签为 None，则使用 None 重复
        labels = itertools.repeat(None)
    elif len(labels) != ncols:
        # 如果标签的长度与 X 的列数不兼容，则引发 ValueError
        raise ValueError("Dimensions of labels and X must be compatible")

    # 设置输入的 whis 参数
    input_whis = whis
    # 遍历输入数据 X 和对应的标签 labels，同时获取索引 ii 和元组 (x, label)
    for ii, (x, label) in enumerate(zip(X, labels)):

        # 创建空字典 stats，用于存储统计信息
        stats = {}

        # 如果标签不为空，则将其存入 stats 字典中
        if label is not None:
            stats['label'] = label

        # 将输入的 whis 值恢复为 input_whis 的值，以防它在循环中被更改
        whis = input_whis

        # 将当前的 stats 字典添加到 bxpstats 列表中
        bxpstats.append(stats)

        # 如果 x 的长度为 0，则跳过当前循环，设置 stats 字典中相关统计值为 NaN
        if len(x) == 0:
            stats['fliers'] = np.array([])
            stats['mean'] = np.nan
            stats['med'] = np.nan
            stats['q1'] = np.nan
            stats['q3'] = np.nan
            stats['iqr'] = np.nan
            stats['cilo'] = np.nan
            stats['cihi'] = np.nan
            stats['whislo'] = np.nan
            stats['whishi'] = np.nan
            continue

        # 将 x 转换为 masked array，并展开为一维数组
        x = np.ma.asarray(x)
        x = x.data[~x.mask].ravel()

        # 计算算术平均值并存入 stats 字典中
        stats['mean'] = np.mean(x)

        # 计算中位数和四分位数
        q1, med, q3 = np.percentile(x, [25, 50, 75])

        # 计算四分位距
        stats['iqr'] = q3 - q1

        # 如果四分位距为 0 且 autorange 为真，则更新 whis 的值
        if stats['iqr'] == 0 and autorange:
            whis = (0, 100)

        # 计算围绕中位数的置信区间，并存入 stats 字典中
        stats['cilo'], stats['cihi'] = _compute_conf_interval(
            x, med, stats['iqr'], bootstrap
        )

        # 计算最低/最高的非异常值
        if np.iterable(whis) and not isinstance(whis, str):
            loval, hival = np.percentile(x, whis)
        elif np.isreal(whis):
            loval = q1 - whis * stats['iqr']
            hival = q3 + whis * stats['iqr']
        else:
            raise ValueError('whis must be a float or list of percentiles')

        # 计算上边缘值
        wiskhi = x[x <= hival]
        if len(wiskhi) == 0 or np.max(wiskhi) < q3:
            stats['whishi'] = q3
        else:
            stats['whishi'] = np.max(wiskhi)

        # 计算下边缘值
        wisklo = x[x >= loval]
        if len(wisklo) == 0 or np.min(wisklo) > q1:
            stats['whislo'] = q1
        else:
            stats['whislo'] = np.min(wisklo)

        # 计算所有异常值，并存入 stats 字典中
        stats['fliers'] = np.concatenate([
            x[x < stats['whislo']],
            x[x > stats['whishi']],
        ])

        # 将 q1、med、q3 存入 stats 字典中
        stats['q1'], stats['med'], stats['q3'] = q1, med, q3

    # 返回 bxpstats 列表，其中包含了每个数据集 x 的统计信息
    return bxpstats
#: Maps short codes for line style to their full name used by backends.
ls_mapper = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
#: Maps full names for line styles used by backends to their short codes.
ls_mapper_r = {v: k for k, v in ls_mapper.items()}


def contiguous_regions(mask):
    """
    Return a list of (ind0, ind1) such that ``mask[ind0:ind1].all()`` is
    True and we cover all such regions.
    """
    # Convert the input mask into a NumPy boolean array
    mask = np.asarray(mask, dtype=bool)

    if not mask.size:
        return []

    # Find the indices where the mask changes value
    idx, = np.nonzero(mask[:-1] != mask[1:])
    idx += 1

    # Convert NumPy array indices to a list
    idx = idx.tolist()

    # Add the start and end indices if necessary to cover entire regions
    if mask[0]:
        idx = [0] + idx
    if mask[-1]:
        idx.append(len(mask))

    # Return a list of tuples representing contiguous True regions
    return list(zip(idx[::2], idx[1::2]))


def is_math_text(s):
    """
    Return whether the string *s* contains math expressions.

    This is done by checking whether *s* contains an even number of
    non-escaped dollar signs.
    """
    # Convert input to string
    s = str(s)
    # Count occurrences of '$' that are not escaped with '\$'
    dollar_count = s.count(r'$') - s.count(r'\$')
    # Determine if the count of '$' is even and greater than zero
    even_dollars = (dollar_count > 0 and dollar_count % 2 == 0)
    # Return whether the string contains math expressions
    return even_dollars


def _to_unmasked_float_array(x):
    """
    Convert a sequence to a float array; if input was a masked array, masked
    values are converted to nans.
    """
    # Check if input has attribute 'mask' indicating it's a masked array
    if hasattr(x, 'mask'):
        # Convert masked array to float array, replacing masked values with NaNs
        return np.ma.asarray(x, float).filled(np.nan)
    else:
        # Convert input to float array
        return np.asarray(x, float)


def _check_1d(x):
    """Convert scalars to 1D arrays; pass-through arrays as is."""
    # Unpack in case of e.g. Pandas or xarray object
    x = _unpack_to_numpy(x)
    # Check if input has necessary attributes or is scalar
    if (not hasattr(x, 'shape') or
            not hasattr(x, 'ndim') or
            len(x.shape) < 1):
        # Ensure input is at least 1-dimensional array
        return np.atleast_1d(x)
    else:
        # Return input array as-is if it meets criteria
        return x


def _reshape_2D(X, name):
    """
    Use Fortran ordering to convert ndarrays and lists of iterables to lists of
    1D arrays.

    Lists of iterables are converted by applying `numpy.asanyarray` to each of
    their elements.  1D ndarrays are returned in a singleton list containing
    them.  2D ndarrays are converted to the list of their *columns*.

    *name* is used to generate the error message for invalid inputs.
    """

    # Unpack in case of e.g. Pandas or xarray object
    X = _unpack_to_numpy(X)

    # Iterate over columns for ndarrays.
    # 检查 X 是否是 numpy 数组
    if isinstance(X, np.ndarray):
        # 如果是数组，转置为列向量
        X = X.T

        # 如果数组长度为0，返回包含一个空列表的列表
        if len(X) == 0:
            return [[]]
        # 如果数组是1维且元素是标量，直接返回包含 X 的列表
        elif X.ndim == 1 and np.ndim(X[0]) == 0:
            return [X]
        # 如果数组是1维或2维，先将其展平为1维
        elif X.ndim in [1, 2]:
            return [np.reshape(x, -1) for x in X]
        else:
            # 抛出值错误，要求 name 变量必须有2个或更少的维度
            raise ValueError(f'{name} must have 2 or fewer dimensions')

    # 迭代处理可迭代对象的列表
    if len(X) == 0:
        return [[]]

    result = []
    is_1d = True
    for xi in X:
        # 检查 xi 是否可迭代，但字符串视为单个元素
        if not isinstance(xi, str):
            try:
                iter(xi)
            except TypeError:
                pass
            else:
                is_1d = False
        # 将 xi 转换为 numpy 数组
        xi = np.asanyarray(xi)
        nd = np.ndim(xi)
        if nd > 1:
            # 抛出值错误，要求 name 变量必须有2个或更少的维度
            raise ValueError(f'{name} must have 2 or fewer dimensions')
        # 将 xi 展平为1维并添加到结果列表中
        result.append(xi.reshape(-1))

    if is_1d:
        # 如果结果是1维数组，直接返回包含结果的列表
        return [np.reshape(result, -1)]
    else:
        # 如果结果是2维数组或者1维数组的可迭代对象，返回结果列表
        return result
# 创建一个空列表，用于存储每个小提琴图的统计数据字典
vpstats = []

# 将输入的数据 X 转换成一个二维数据序列的列表
X = _reshape_2D(X, "X")

# 如果 quantiles 不为 None 并且长度不为 0，将其转换为与数据序列相同形状的二维数组
if quantiles is not None and len(quantiles) != 0:
    quantiles = _reshape_2D(quantiles, "quantiles")
# 否则，如果 quantiles 是 None 或空数组，则用空列表填充 quantiles
else:
    quantiles = [[]] * len(X)

# 检查 quantiles 的长度与数据序列 X 的长度是否一致，若不一致则抛出 ValueError 异常
if len(X) != len(quantiles):
    raise ValueError("List of violinplot statistics and quantiles values"
                     " must have the same length")

# 将 X 和 quantiles 进行逐对打包，以备后续使用
    # 遍历输入数据 X 和对应的分位数 quantiles
    for (x, q) in zip(X, quantiles):
        # 为当前分布创建结果字典
        stats = {}

        # 计算分布的基本统计信息
        min_val = np.min(x)  # 计算数据 x 的最小值
        max_val = np.max(x)  # 计算数据 x 的最大值
        quantile_val = np.percentile(x, 100 * q)  # 计算数据 x 的分位数值

        # 计算核密度估计值
        coords = np.linspace(min_val, max_val, points)  # 在最小值和最大值之间生成均匀间隔的坐标点
        stats['vals'] = method(x, coords)  # 使用指定方法计算数据 x 在坐标点 coords 处的值
        stats['coords'] = coords  # 将坐标点保存到结果字典中

        # 存储该分布的额外统计信息
        stats['mean'] = np.mean(x)  # 计算数据 x 的均值
        stats['median'] = np.median(x)  # 计算数据 x 的中位数
        stats['min'] = min_val  # 将最小值保存到结果字典中
        stats['max'] = max_val  # 将最大值保存到结果字典中
        stats['quantiles'] = np.atleast_1d(quantile_val)  # 将分位数值保存到结果字典中，确保其至少是一个数组

        # 将当前分布的统计信息添加到输出列表中
        vpstats.append(stats)

    # 返回所有分布的统计信息列表
    return vpstats
def pts_to_prestep(x, *args):
    """
    Convert continuous line to pre-steps.

    Given a set of ``N`` points, convert to ``2N - 1`` points, which when
    connected linearly give a step function which changes values at the
    beginning of the intervals.

    Parameters
    ----------
    x : array
        The x location of the steps. May be empty.

    y1, ..., yp : array
        y arrays to be turned into steps; all must be the same length as ``x``.

    Returns
    -------
    array
        The x and y values converted to steps in the same order as the input;
        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is
        length ``N``, each of these arrays will be length ``2N + 1``. For
        ``N=0``, the length will be 0.

    Examples
    --------
    >>> x_s, y1_s, y2_s = pts_to_prestep(x, y1, y2)
    """
    # Create an array to hold the step values, initialized with zeros
    steps = np.zeros((1 + len(args), max(2 * len(x) - 1, 0)))
    
    # Assign x values to every second element starting from the beginning
    steps[0, 0::2] = x
    
    # Assign values to create a step function where each pair of consecutive
    # elements are the same, starting from the second element
    steps[0, 1::2] = steps[0, 0:-2:2]
    
    # Assign y values (from *args) to every second element starting from the beginning
    steps[1:, 0::2] = args
    
    # Assign values to create a step function for each y array where each pair
    # of consecutive elements are the same, starting from the second element
    steps[1:, 1::2] = steps[1:, 2::2]
    
    # Return the array with x and y values converted to steps
    return steps


def pts_to_poststep(x, *args):
    """
    Convert continuous line to post-steps.

    Given a set of ``N`` points convert to ``2N + 1`` points, which when
    connected linearly give a step function which changes values at the end of
    the intervals.

    Parameters
    ----------
    x : array
        The x location of the steps. May be empty.

    y1, ..., yp : array
        y arrays to be turned into steps; all must be the same length as ``x``.

    Returns
    -------
    array
        The x and y values converted to steps in the same order as the input;
        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is
        length ``N``, each of these arrays will be length ``2N + 1``. For
        ``N=0``, the length will be 0.

    Examples
    --------
    >>> x_s, y1_s, y2_s = pts_to_poststep(x, y1, y2)
    """
    # Create an array to hold the step values, initialized with zeros
    steps = np.zeros((1 + len(args), max(2 * len(x) - 1, 0)))
    
    # Assign x values to every second element starting from the beginning
    steps[0, 0::2] = x
    
    # Assign values to create a step function where each pair of consecutive
    # elements are the same, starting from the second element
    steps[0, 1::2] = steps[0, 2::2]
    
    # Assign y values (from *args) to every second element starting from the beginning
    steps[1:, 0::2] = args
    
    # Assign values to create a step function for each y array where each pair
    # of consecutive elements are the same, starting from the first element
    steps[1:, 1::2] = steps[1:, 0:-2:2]
    
    # Return the array with x and y values converted to steps
    return steps


def pts_to_midstep(x, *args):
    """
    Convert continuous line to mid-steps.

    Given a set of ``N`` points convert to ``2N`` points which when connected
    linearly give a step function which changes values at the middle of the
    intervals.

    Parameters
    ----------
    x : array
        The x location of the steps. May be empty.

    y1, ..., yp : array
        y arrays to be turned into steps; all must be the same length as
        ``x``.

    Returns
    -------
    array
        The x and y values converted to steps in the same order as the input;
        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is
        length ``N``, each of these arrays will be length ``2N``.

    Examples
    --------
    """
    # Create an array to hold the step values, initialized with zeros
    steps = np.zeros((1 + len(args), max(2 * len(x), 0)))
    
    # Assign x values to every second element starting from the beginning
    steps[0, 0::2] = x
    
    # Assign values to create a step function where each pair of consecutive
    # elements are the same, starting from the second element
    steps[0, 1::2] = steps[0, 2::2]
    
    # Assign y values (from *args) to every second element starting from the beginning
    steps[1:, 0::2] = args
    
    # Assign values to create a step function for each y array where each pair
    # of consecutive elements are the same, starting from the first element
    steps[1:, 1::2] = steps[1:, 0:-2:2]
    
    # Return the array with x and y values converted to steps
    return steps
    # 调用函数 pts_to_midstep 处理输入的 x, y1, y2，获取返回的 x_s, y1_s, y2_s
    x_s, y1_s, y2_s = pts_to_midstep(x, y1, y2)
    # 创建一个形状为 (1 + len(args), 2 * len(x)) 的全零数组 steps
    steps = np.zeros((1 + len(args), 2 * len(x)))
    # 将输入的 x 转换为 NumPy 数组
    x = np.asanyarray(x)
    # 将 steps 数组的奇数列和偶数列赋值为相邻元素的均值，用于生成中间步骤
    steps[0, 1:-1:2] = steps[0, 2::2] = (x[:-1] + x[1:]) / 2
    # 将 steps 数组的第一列和最后一列分别设置为 x 的第一个元素和最后一个元素，即使输入为空也适用
    steps[0, :1] = x[:1]  # Also works for zero-sized input.
    steps[0, -1:] = x[-1:]
    # 将除第一行外的每行的偶数列设置为 args 的值，再将每行的奇数列设置为对应偶数列的值，用于处理其他参数的步骤
    steps[1:, 0::2] = args
    steps[1:, 1::2] = steps[1:, 0::2]
    # 返回生成的 steps 数组，其中包含了中间步骤的数据
    return steps
STEP_LOOKUP_MAP = {'default': lambda x, y: (x, y),  # 定义一个映射字典，将字符串键映射到函数对象
                   'steps': pts_to_prestep,       # 'steps'键映射到pts_to_prestep函数
                   'steps-pre': pts_to_prestep,    # 'steps-pre'键映射到pts_to_prestep函数
                   'steps-post': pts_to_poststep,  # 'steps-post'键映射到pts_to_poststep函数
                   'steps-mid': pts_to_midstep}   # 'steps-mid'键映射到pts_to_midstep函数


def index_of(y):
    """
    A helper function to create reasonable x values for the given *y*.

    This is used for plotting (x, y) if x values are not explicitly given.

    First try ``y.index`` (assuming *y* is a `pandas.Series`), if that
    fails, use ``range(len(y))``.

    This will be extended in the future to deal with more types of
    labeled data.

    Parameters
    ----------
    y : float or array-like

    Returns
    -------
    x, y : ndarray
       The x and y values to plot.
    """
    try:
        return y.index.to_numpy(), y.to_numpy()  # 尝试从y的索引和值中获取ndarray，用于绘图的x和y值
    except AttributeError:
        pass
    try:
        y = _check_1d(y)  # 检查y是否为一维数组
    except (VisibleDeprecationWarning, ValueError):
        # NumPy 1.19 will warn on ragged input, and we can't actually use it.
        pass
    else:
        return np.arange(y.shape[0], dtype=float), y  # 返回一个从0到y长度的浮点数类型的ndarray作为x值，y作为y值
    raise ValueError('Input could not be cast to an at-least-1D NumPy array')  # 如果无法转换成至少一维的NumPy数组则抛出异常


def safe_first_element(obj):
    """
    Return the first element in *obj*.

    This is a type-independent way of obtaining the first element,
    supporting both index access and the iterator protocol.
    """
    if isinstance(obj, collections.abc.Iterator):
        # needed to accept `array.flat` as input.
        # np.flatiter reports as an instance of collections.Iterator but can still be
        # indexed via []. This has the side effect of re-setting the iterator, but
        # that is acceptable.
        try:
            return obj[0]  # 如果obj是迭代器，则尝试获取第一个元素
        except TypeError:
            pass
        raise RuntimeError("matplotlib does not support generators as input")  # 抛出运行时异常，matplotlib不支持生成器作为输入
    return next(iter(obj))  # 否则，使用迭代器协议获取obj的下一个元素


def _safe_first_finite(obj):
    """
    Return the first finite element in *obj* if one is available and skip_nonfinite is
    True. Otherwise, return the first element.

    This is a method for internal use.

    This is a type-independent way of obtaining the first finite element, supporting
    both index access and the iterator protocol.
    """
    def safe_isfinite(val):
        if val is None:
            return False
        try:
            return math.isfinite(val)  # 检查val是否是有限数
        except (TypeError, ValueError):
            # if the outer object is 2d, then val is a 1d array, and
            # - math.isfinite(numpy.zeros(3)) raises TypeError
            # - math.isfinite(torch.zeros(3)) raises ValueError
            pass
        try:
            return np.isfinite(val) if np.isscalar(val) else True  # 检查val是否是有限数，处理标量和数组的情况
        except TypeError:
            # This is something that NumPy cannot make heads or tails of,
            # assume "finite"
            return True

    if isinstance(obj, np.flatiter):
        # TODO do the finite filtering on this
        return obj[0]  # 如果obj是NumPy的flatiter类型，则返回第一个元素
    # 如果对象是迭代器的实例（但不是可迭代对象），抛出运行时错误
    elif isinstance(obj, collections.abc.Iterator):
        # 抛出异常，指明 matplotlib 不支持生成器作为输入
        raise RuntimeError("matplotlib does not support generators as input")
    # 对于其他类型的可迭代对象
    else:
        # 遍历可迭代对象中的每个元素
        for val in obj:
            # 检查元素是否为有限数值（安全性检查）
            if safe_isfinite(val):
                # 如果是有限数值，返回该元素
                return val
        # 如果没有找到有限数值的元素，则返回可迭代对象的第一个元素（安全操作）
        return safe_first_element(obj)
# 将 dictview 对象转换为列表，其他输入不做修改直接返回
def sanitize_sequence(data):
    return (list(data) if isinstance(data, collections.abc.MappingView)
            else data)


# 规范化关键字参数的辅助函数
def normalize_kwargs(kw, alias_mapping=None):
    """
    Helper function to normalize kwarg inputs.

    Parameters
    ----------
    kw : dict or None
        A dict of keyword arguments.  None is explicitly supported and treated
        as an empty dict, to support functions with an optional parameter of
        the form ``props=None``.

    alias_mapping : dict or Artist subclass or Artist instance, optional
        A mapping between a canonical name to a list of aliases, in order of
        precedence from lowest to highest.

        If the canonical value is not in the list it is assumed to have the
        highest priority.

        If an Artist subclass or instance is passed, use its properties alias
        mapping.

    Raises
    ------
    TypeError
        To match what Python raises if invalid arguments/keyword arguments are
        passed to a callable.
    """

    # 引入 matplotlib 的 Artist 类
    from matplotlib.artist import Artist

    # 如果 kw 为 None，则返回空字典
    if kw is None:
        return {}

    # 处理 alias_mapping 的默认值
    if alias_mapping is None:
        alias_mapping = {}
    # 如果 alias_mapping 是 Artist 类或其实例，则使用其属性的别名映射
    elif (isinstance(alias_mapping, type) and issubclass(alias_mapping, Artist)
          or isinstance(alias_mapping, Artist)):
        alias_mapping = getattr(alias_mapping, "_alias_map", {})

    # 创建从别名到规范名称的映射
    to_canonical = {alias: canonical
                    for canonical, alias_list in alias_mapping.items()
                    for alias in alias_list}
    canonical_to_seen = {}
    ret = {}  # 输出字典

    # 遍历 kw 中的键值对
    for k, v in kw.items():
        canonical = to_canonical.get(k, k)
        # 检查规范名称是否已经在 canonical_to_seen 中出现过
        if canonical in canonical_to_seen:
            # 如果出现重复的别名和规范名称，则抛出 TypeError
            raise TypeError(f"Got both {canonical_to_seen[canonical]!r} and "
                            f"{k!r}, which are aliases of one another")
        canonical_to_seen[canonical] = k
        ret[canonical] = v

    return ret


# 上下文管理器，用于锁定给定路径
@contextlib.contextmanager
def _lock_path(path):
    """
    Context manager for locking a path.

    Usage::

        with _lock_path(path):
            ...

    Another thread or process that attempts to lock the same path will wait
    until this context manager is exited.

    The lock is implemented by creating a temporary file in the parent
    directory, so that directory must exist and be writable.
    """
    # 将路径转换为 Path 对象
    path = Path(path)
    # 构建锁文件的路径，名称为原始文件名 + ".matplotlib-lock"
    lock_path = path.with_name(path.name + ".matplotlib-lock")
    retries = 50
    sleeptime = 0.1
    # 尝试获取锁文件，最多重试 50 次，每次间隔 0.1 秒
    for _ in range(retries):
        try:
            with lock_path.open("xb"):
                break
        except FileExistsError:
            time.sleep(sleeptime)
    else:
        # 如果超过重试次数仍未获取到锁文件，则抛出 TimeoutError
        raise TimeoutError("""\
Lock error: Matplotlib failed to acquire the following lock file:
    {}
This maybe due to another process holding this lock file.  If you are sure no
""".format(lock_path))
# 删除文件并重新尝试运行，显示带有锁定路径的消息。
    other Matplotlib process is running, remove this file and try again.""".format(
        lock_path))
# 尝试执行代码块
    try:
        # 返回上下文管理的内容
        yield
# 最终执行的路径删除代码块
    finally:
        lock_path.unlink()


def _topmost_artist(
        artists,
        # 缓存最大值的功能
        _cached_max=functools.partial(max, key=operator.attrgetter("zorder"))):
    """
    获取列表中最顶部的艺术家。

    如果有多个最顶部的艺术家，返回最后一个，因为它将绘制在其他艺术家之上。
    `max`函数在存在多个最大值时返回第一个，因此需要以反向顺序迭代列表。
    """
    return _cached_max(reversed(artists))


def _str_equal(obj, s):
    """
    返回 *obj* 是否是与字符串 *s* 相等的字符串。

    此辅助函数仅用于处理 *obj* 是numpy数组的情况，
    因为这种情况下，简单的 ``obj == s`` 将返回一个数组，不能在布尔上下文中使用。
    """
    return isinstance(obj, str) and obj == s


def _str_lower_equal(obj, s):
    """
    返回 *obj* 是否是在小写情况下与字符串 *s* 相等的字符串。

    此辅助函数仅用于处理 *obj* 是numpy数组的情况，
    因为这种情况下，简单的 ``obj == s`` 将返回一个数组，不能在布尔上下文中使用。
    """
    return isinstance(obj, str) and obj.lower() == s


def _array_perimeter(arr):
    """
    获取数组 *arr* 的周边元素。

    参数
    ----------
    arr : ndarray, shape (M, N)
        输入数组

    返回
    -------
    ndarray, shape (2*(M - 1) + 2*(N - 1),)
        数组的周边元素::

           [arr[0, 0], ..., arr[0, -1], ..., arr[-1, -1], ..., arr[-1, 0], ...]

    示例
    --------
    >>> i, j = np.ogrid[:3, :4]
    >>> a = i*10 + j
    >>> a
    array([[ 0,  1,  2,  3],
           [10, 11, 12, 13],
           [20, 21, 22, 23]])
    >>> _array_perimeter(a)
    array([ 0,  1,  2,  3, 13, 23, 22, 21, 20, 10])
    """
    # 注意使用Python的半开区间范围来避免重复角点
    forward = np.s_[0:-1]      # [0 ... -1)
    backward = np.s_[-1:0:-1]  # [-1 ... 0)
    return np.concatenate((
        arr[0, forward],
        arr[forward, -1],
        arr[-1, backward],
        arr[backward, 0],
    ))


def _unfold(arr, axis, size, step):
    """
    添加一个额外维度，包含沿 *axis* 的滑动窗口。

    所有窗口大小为 *size*，从每 *step* 元素开始。

    参数
    ----------
    arr : ndarray, shape (N_1, ..., N_k)
        输入数组
    axis : int
        提取窗口的轴
    size : int
        窗口的大小
    step : int
        后续窗口中第一个元素之间的步长。

    返回
    -------
    ndarray, shape (N_1, ..., 1 + (N_axis-size)/step, ..., N_k, size)

    示例
    --------
    >>> i, j = np.ogrid[:3, :7]
    >>> a = i*10 + j
    >>> a
    # 定义了一个三维数组的示例
    array([[ 0,  1,  2,  3,  4,  5,  6],
           [10, 11, 12, 13, 14, 15, 16],
           [20, 21, 22, 23, 24, 25, 26]])
    
    # 调用 _unfold 函数对数组进行展开操作，指定展开的轴为第 1 轴，大小为 3，步长为 2
    >>> _unfold(a, axis=1, size=3, step=2)
    
    # 返回的结果是一个三维数组，每个子数组从原始数组中截取并展开的部分
    array([[[ 0,  1,  2],
            [ 2,  3,  4],
            [ 4,  5,  6]],
    
           [[10, 11, 12],
            [12, 13, 14],
            [14, 15, 16]],
    
           [[20, 21, 22],
            [22, 23, 24],
            [24, 25, 26]]])
    
    """
    # 计算新数组的形状，增加一个维度用于展开
    new_shape = [*arr.shape, size]
    # 计算新数组的步长，增加一个维度用于展开
    new_strides = [*arr.strides, arr.strides[axis]]
    # 根据指定的轴、大小和步长计算新数组在指定轴上的形状调整
    new_shape[axis] = (new_shape[axis] - size) // step + 1
    # 根据指定的轴和步长调整新数组在指定轴上的步长
    new_strides[axis] = new_strides[axis] * step
    # 使用 NumPy 提供的 stride_tricks 库中的 as_strided 函数，按照新的形状和步长生成展开后的数组视图，设置为不可写
    return np.lib.stride_tricks.as_strided(arr,
                                           shape=new_shape,
                                           strides=new_strides,
                                           writeable=False)
def _array_patch_perimeters(x, rstride, cstride):
    """
    Extract perimeters of patches from *arr*.

    Extracted patches are of size (*rstride* + 1) x (*cstride* + 1) and
    share perimeters with their neighbors. The ordering of the vertices matches
    that returned by ``_array_perimeter``.

    Parameters
    ----------
    x : ndarray, shape (N, M)
        Input array
    rstride : int
        Vertical (row) stride between corresponding elements of each patch
    cstride : int
        Horizontal (column) stride between corresponding elements of each patch

    Returns
    -------
    ndarray, shape (N/rstride * M/cstride, 2 * (rstride + cstride))
    """
    assert rstride > 0 and cstride > 0  # 确保 rstride 和 cstride 大于零
    assert (x.shape[0] - 1) % rstride == 0  # 确保行数减1对 rstride 取模为零
    assert (x.shape[1] - 1) % cstride == 0  # 确保列数减1对 cstride 取模为零

    # We build up each perimeter from four half-open intervals. Here is an
    # illustrated explanation for rstride == cstride == 3
    #
    #       T T T R
    #       L     R
    #       L     R
    #       L B B B
    #
    # where T means that this element will be in the top array, R for right,
    # B for bottom and L for left. Each of the arrays below has a shape of:
    #
    #    (number of perimeters that can be extracted vertically,
    #     number of perimeters that can be extracted horizontally,
    #     cstride for top and bottom and rstride for left and right)
    #
    # Note that _unfold doesn't incur any memory copies, so the only costly
    # operation here is the np.concatenate.
    
    # 从输入数组 x 中提取每个 patch 的周长，每个 patch 大小为 (*rstride* + 1) x (*cstride* + 1)
    top = _unfold(x[:-1:rstride, :-1], 1, cstride, cstride)
    bottom = _unfold(x[rstride::rstride, 1:], 1, cstride, cstride)[..., ::-1]
    right = _unfold(x[:-1, cstride::cstride], 0, rstride, rstride)
    left = _unfold(x[1:, :-1:cstride], 0, rstride, rstride)[..., ::-1]

    # 将提取的周长组合成一个数组，每个周长包含 top, right, bottom, left 四个部分
    return (np.concatenate((top, right, bottom, left), axis=2)
              .reshape(-1, 2 * (rstride + cstride)))


@contextlib.contextmanager
def _setattr_cm(obj, **kwargs):
    """
    Temporarily set some attributes; restore original state at context exit.
    """
    sentinel = object()
    origs = {}
    # 遍历关键字参数kwargs中的每个属性名attr
    for attr in kwargs:
        # 获取对象obj中属性attr的原始值orig，如果属性不存在则使用sentinel作为默认值
        orig = getattr(obj, attr, sentinel)
        # 如果属性attr存在于对象obj的实例字典中，或者orig值为sentinel
        if attr in obj.__dict__ or orig is sentinel:
            # 如果从实例字典中获取属性，或者对象不具有该属性，我们可以信任上述原始值
            origs[attr] = orig
        else:
            # 如果属性不在实例字典中，则它必须是来自类级别的
            cls_orig = getattr(type(obj), attr)
            # 如果处理的是一个属性（但不是一般的描述符），我们希望将原始值设置回去
            if isinstance(cls_orig, property):
                origs[attr] = orig
            # 否则，我们将在实例字典级别阴影来自MRO（方法解析顺序）更高级别的属性
            # 我们假设我们可以使用delattr(obj, attr)来清理自己的后续操作
            # 如果使用非property的自定义描述符实现了__set__（而__delete__不像堆栈那样操作），这段代码可能会失败
            # 然而，这是一个内部工具，我们目前没有任何自定义描述符。
            else:
                origs[attr] = sentinel

    try:
        # 针对kwargs中的每个属性和对应的值，将值设置为对象obj的属性
        for attr, val in kwargs.items():
            setattr(obj, attr, val)
        # 执行yield语句，暂时性地返回控制权给调用者
        yield
    finally:
        # 恢复所有被更改的属性到它们的原始状态
        for attr, orig in origs.items():
            # 如果原始值是sentinel，删除对象obj的属性attr
            if orig is sentinel:
                delattr(obj, attr)
            # 否则，将原始值orig重新设置为对象obj的属性attr的值
            else:
                setattr(obj, attr, orig)
class _OrderedSet(collections.abc.MutableSet):
    # 定义一个有序集合类，继承自MutableSet抽象基类
    def __init__(self):
        # 初始化方法，创建一个空的有序字典
        self._od = collections.OrderedDict()

    def __contains__(self, key):
        # 判断键是否在有序字典中
        return key in self._od

    def __iter__(self):
        # 返回有序字典的迭代器
        return iter(self._od)

    def __len__(self):
        # 返回有序字典中键的数量
        return len(self._od)

    def add(self, key):
        # 向有序字典中添加键，若键已存在则先删除再添加
        self._od.pop(key, None)
        self._od[key] = None

    def discard(self, key):
        # 从有序字典中移除指定的键
        self._od.pop(key, None)


# Agg's buffers are unmultiplied RGBA8888, which neither PyQt<=5.1 nor cairo
# support; however, both do support premultiplied ARGB32.


def _premultiplied_argb32_to_unmultiplied_rgba8888(buf):
    """
    Convert a premultiplied ARGB32 buffer to an unmultiplied RGBA8888 buffer.
    """
    # 使用.take()确保结果是C连续的数组
    rgba = np.take(
        buf,
        [2, 1, 0, 3] if sys.byteorder == "little" else [1, 2, 3, 0], axis=2)
    # 提取RGB部分
    rgb = rgba[..., :-1]
    alpha = rgba[..., -1]
    # 反预乘alpha通道，使用cairo-png.c中相同的公式
    mask = alpha != 0
    for channel in np.rollaxis(rgb, -1):
        channel[mask] = (
            (channel[mask].astype(int) * 255 + alpha[mask] // 2)
            // alpha[mask])
    return rgba


def _unmultiplied_rgba8888_to_premultiplied_argb32(rgba8888):
    """
    Convert an unmultiplied RGBA8888 buffer to a premultiplied ARGB32 buffer.
    """
    # 根据系统字节顺序选择对应通道顺序
    if sys.byteorder == "little":
        argb32 = np.take(rgba8888, [2, 1, 0, 3], axis=2)
        rgb24 = argb32[..., :-1]
        alpha8 = argb32[..., -1:]
    else:
        argb32 = np.take(rgba8888, [3, 0, 1, 2], axis=2)
        alpha8 = argb32[..., :1]
        rgb24 = argb32[..., 1:]
    # 当alpha通道不完全不透明时进行预乘操作，这里的转换需要在整数缓冲区中进行
    if alpha8.min() != 0xff:
        np.multiply(rgb24, alpha8 / 0xff, out=rgb24, casting="unsafe")
    return argb32


def _get_nonzero_slices(buf):
    """
    Return the bounds of the nonzero region of a 2D array as a pair of slices.

    ``buf[_get_nonzero_slices(buf)]`` is the smallest sub-rectangle in *buf*
    that encloses all non-zero entries in *buf*.  If *buf* is fully zero, then
    ``(slice(0, 0), slice(0, 0))`` is returned.
    """
    # 找到二维数组中非零区域的边界，返回作为两个切片的元组
    x_nz, = buf.any(axis=0).nonzero()
    y_nz, = buf.any(axis=1).nonzero()
    if len(x_nz) and len(y_nz):
        l, r = x_nz[[0, -1]]
        b, t = y_nz[[0, -1]]
        return slice(b, t + 1), slice(l, r + 1)
    else:
        return slice(0, 0), slice(0, 0)


def _pformat_subprocess(command):
    """Pretty-format a subprocess command for printing/logging purposes."""
    # 格式化子进程命令，便于打印和记录日志
    return (command if isinstance(command, str)
            else " ".join(shlex.quote(os.fspath(arg)) for arg in command))


def _check_and_log_subprocess(command, logger, **kwargs):
    """
    Run *command*, returning its stdout output if it succeeds.
    """
    # 运行给定的命令，并在成功时返回其标准输出
    # 在 DEBUG 级别记录要执行的子进程命令
    logger.debug('%s', _pformat_subprocess(command))
    # 运行子进程命令，捕获输出
    proc = subprocess.run(command, capture_output=True, **kwargs)
    # 如果子进程返回非零状态码，则处理失败情况
    if proc.returncode:
        stdout = proc.stdout
        # 如果 stdout 是字节类型，则解码为字符串
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        stderr = proc.stderr
        # 如果 stderr 是字节类型，则解码为字符串
        if isinstance(stderr, bytes):
            stderr = stderr.decode()
        # 抛出运行时异常，包含失败命令、stdout 和 stderr 输出
        raise RuntimeError(
            f"The command\n"
            f"    {_pformat_subprocess(command)}\n"
            f"failed and generated the following output:\n"
            f"{stdout}\n"
            f"and the following error:\n"
            f"{stderr}")
    # 如果有 stdout 输出，则在 DEBUG 级别记录
    if proc.stdout:
        logger.debug("stdout:\n%s", proc.stdout)
    # 如果有 stderr 输出，则在 DEBUG 级别记录
    if proc.stderr:
        logger.debug("stderr:\n%s", proc.stderr)
    # 返回子进程的 stdout 输出
    return proc.stdout
# 执行 Matplotlib 创建新 GUI 应用程序时的操作系统相关设置
def _setup_new_guiapp():
    """
    Perform OS-dependent setup when Matplotlib creates a new GUI application.
    """
    try:
        # 尝试获取当前进程的显式应用程序用户模型 ID
        _c_internal_utils.Win32_GetCurrentProcessExplicitAppUserModelID()
    except OSError:
        # 如果失败，设置当前进程的显式应用程序用户模型 ID 为 "matplotlib"
        _c_internal_utils.Win32_SetCurrentProcessExplicitAppUserModelID(
            "matplotlib")


# 格式化数字，保留指定精度的小数，移除尾部的零及可能的小数点
def _format_approx(number, precision):
    """
    Format the number with at most the number of decimals given as precision.
    Remove trailing zeros and possibly the decimal point.
    """
    return f'{number:.{precision}f}'.rstrip('0').rstrip('.') or '0'


# 计算数值的有效数字位数，基于给定的误差值
def _g_sig_digits(value, delta):
    """
    Return the number of significant digits to %g-format *value*, assuming that
    it is known with an error of *delta*.
    """
    if delta == 0:
        # 当 delta 为 0 时，可能是因为在尝试格式化微小范围内的值，替换为与最接近浮点数的距离
        delta = abs(np.spacing(value))
    # 如果 value 是有限数，则计算其有效数字位数
    return max(
        0,
        (math.floor(math.log10(abs(value))) + 1 if value else 1)
        - math.floor(math.log10(delta))) if math.isfinite(value) else 0


# 将 Unicode 键或 X keysym 转换为 Matplotlib 键名
def _unikey_or_keysym_to_mplkey(unikey, keysym):
    """
    Convert a Unicode key or X keysym to a Matplotlib key name.

    The Unicode key is checked first; this avoids having to list most printable
    keysyms such as ``EuroSign``.
    """
    # 对于非打印字符，例如在 gtk3 中传递 "\0" 或 tk 中传递空字符串
    if unikey and unikey.isprintable():
        return unikey
    key = keysym.lower()
    if key.startswith("kp_"):  # 处理键盘键 (keypad_x，包括 kp_enter)
        key = key[3:]
    if key.startswith("page_"):  # 处理页面键 (page_{up,down})
        key = key.replace("page_", "page")
    if key.endswith(("_l", "_r")):  # 处理左右修饰键 (alt_l, ctrl_l, shift_l)
        key = key[:-2]
    if sys.platform == "darwin" and key == "meta":
        # macOS 上的 meta 键应该被报告为 command 键
        key = "cmd"
    # 映射特定键名到 Matplotlib 的命名规则
    key = {
        "return": "enter",
        "prior": "pageup",  # 在 tk 中使用的键名
        "next": "pagedown",  # 在 tk 中使用的键名
    }.get(key, key)
    return key


# 创建一个能够生成从 mixin 继承的可 pickle 类的工厂函数
@functools.cache
def _make_class_factory(mixin_class, fmt, attr_name=None):
    """
    Return a function that creates picklable classes inheriting from a mixin.

    After ::

        factory = _make_class_factory(FooMixin, fmt, attr_name)
        FooAxes = factory(Axes)

    ``Foo`` is a class that inherits from ``FooMixin`` and ``Axes`` and **is
    """
    @functools.cache
    # 使用 functools.cache 装饰器，将函数 class_factory 转换为带有缓存功能的函数
    def class_factory(axes_class):
        # 如果 axes_class 已经是 mixin_class 的子类，则直接返回 axes_class
        if issubclass(axes_class, mixin_class):
            return axes_class

        # 参数命名为 "axes_class" 是为了向后兼容，但实际上它只是一个基类；没有使用轴的语义。
        base_class = axes_class

        # 定义一个新的子类 subcls，它继承自 mixin_class 和 base_class
        class subcls(mixin_class, base_class):
            # 比 "__module__ = 'matplotlib.cbook'" 更好的近似值。
            __module__ = mixin_class.__module__

            # 定义 __reduce__ 方法，用于对象的序列化和反序列化
            def __reduce__(self):
                return (_picklable_class_constructor,
                        (mixin_class, fmt, attr_name, base_class),
                        self.__getstate__())

        # 设置子类的 __name__ 和 __qualname__ 属性，格式化使用 base_class 的类名
        subcls.__name__ = subcls.__qualname__ = fmt.format(base_class.__name__)
        # 如果 attr_name 不为 None，则将 base_class 存储在 subcls 的 attr_name 属性中
        if attr_name is not None:
            setattr(subcls, attr_name, base_class)
        # 返回创建的子类 subcls
        return subcls

    # 设置 class_factory 函数的 __module__ 属性，保持和 mixin_class.__module__ 相同
    class_factory.__module__ = mixin_class.__module__
    # 返回经过装饰器缓存的 class_factory 函数
    return class_factory
# 定义一个内部辅助函数，用于创建可序列化类
def _picklable_class_constructor(mixin_class, fmt, attr_name, base_class):
    # 调用 _make_class_factory 函数，创建一个类工厂
    factory = _make_class_factory(mixin_class, fmt, attr_name)
    # 使用工厂创建一个新类，并返回其 __new__ 方法的结果
    cls = factory(base_class)
    return cls.__new__(cls)


# 检查 'x' 是否为 PyTorch Tensor 的内部函数
def _is_torch_array(x):
    try:
        # 故意不导入 torch。如果有人创建了一个 torch 数组，torch 应该已经在 sys.modules 中
        return isinstance(x, sys.modules['torch'].Tensor)
    except Exception:  # 捕获可能的异常，比如 TypeError、KeyError、AttributeError 等
        # 我们尝试访问导入模块的属性，可能会有任意用户代码，因此故意捕获所有异常
        return False


# 检查 'x' 是否为 JAX Array 的内部函数
def _is_jax_array(x):
    try:
        # 故意不导入 jax。如果有人创建了一个 jax 数组，jax 应该已经在 sys.modules 中
        return isinstance(x, sys.modules['jax'].Array)
    except Exception:  # 捕获可能的异常，比如 TypeError、KeyError、AttributeError 等
        # 我们尝试访问导入模块的属性，可能会有任意用户代码，因此故意捕获所有异常
        return False


# 检查 'x' 是否为 TensorFlow Tensor 或 Variable 的内部函数
def _is_tensorflow_array(x):
    try:
        # 故意不导入 TensorFlow。如果有人创建了一个 TensorFlow 数组，TensorFlow 应该已经在 sys.modules 中
        # 使用 `is_tensor` 来不依赖于 TensorFlow 数组的类结构，因为 `tf.Variables` 不是 `tf.Tensor` 的实例
        # （它们可以相互转换）
        return isinstance(x, sys.modules['tensorflow'].is_tensor(x))
    except Exception:  # 捕获可能的异常，比如 TypeError、KeyError、AttributeError 等
        # 我们尝试访问导入模块的属性，可能会有任意用户代码，因此故意捕获所有异常
        return False


# 从例如 pandas 和 xarray 对象中提取数据的内部辅助函数
def _unpack_to_numpy(x):
    if isinstance(x, np.ndarray):
        # 如果是 numpy 数组，直接返回
        return x
    if hasattr(x, 'to_numpy'):
        # 假设任何 to_numpy() 方法实际上返回一个 numpy 数组
        return x.to_numpy()
    if hasattr(x, 'values'):
        xtmp = x.values
        # 例如，字典有一个 'values' 属性，但它不是一个属性，因此在这种情况下我们不希望返回一个函数
        if isinstance(xtmp, np.ndarray):
            return xtmp
    # 检查输入变量 x 是否是 Torch、JAX 或 TensorFlow 的数组之一
    # 使用 np.asarray() 转换而不是显式调用 __array__() 方法，
    # 后者只是众多方法之一，并且是最后的手段，详见：
    # https://numpy.org/devdocs/user/basics.interoperability.html#using-arbitrary-objects-in-numpy
    # 因此，让数组自行判断最佳的转换方式
    xtmp = np.asarray(x)

    # 如果 np.asarray 方法未来不返回 numpy 数组
    if isinstance(xtmp, np.ndarray):
        # 返回转换后的 numpy 数组
        return xtmp
    # 返回原始输入 x
    return x
def _auto_format_str(fmt, value):
    """
    Apply *value* to the format string *fmt*.

    This works both with unnamed %-style formatting and
    unnamed {}-style formatting. %-style formatting has priority.
    If *fmt* is %-style formattable that will be used. Otherwise,
    {}-formatting is applied. Strings without formatting placeholders
    are passed through as is.

    Examples
    --------
    >>> _auto_format_str('%.2f m', 0.2)
    '0.20 m'
    >>> _auto_format_str('{} m', 0.2)
    '0.2 m'
    >>> _auto_format_str('const', 0.2)
    'const'
    >>> _auto_format_str('%d or {}', 0.2)
    '0 or {}'
    """
    try:
        # 尝试使用 %-style 格式化字符串 fmt，并传入 value 进行格式化
        return fmt % (value,)
    except (TypeError, ValueError):
        # 如果 fmt 不支持 %-style 格式化，则使用 {}-style 格式化
        return fmt.format(value)
```