# `.\numpy\numpy\lib\_utils_impl.py`

```py
# 导入必要的模块
import os  # 导入操作系统模块
import sys  # 导入系统模块
import textwrap  # 导入文本包装模块
import types  # 导入类型模块
import re  # 导入正则表达式模块
import warnings  # 导入警告模块
import functools  # 导入函数工具模块
import platform  # 导入平台信息模块

# 导入 numpy 相关模块
from numpy._core import ndarray
from numpy._utils import set_module
import numpy as np

# 定义 __all__ 列表，指定导出的公共接口
__all__ = [
    'get_include', 'info', 'show_runtime'
]

# 设置模块为 numpy
@set_module('numpy')
def show_runtime():
    """
    打印系统中各种资源的信息，包括可用的内部支持和正在使用的 BLAS/LAPACK 库

    .. versionadded:: 1.24.0

    See Also
    --------
    show_config : 显示 NumPy 构建时系统中的库信息。

    Notes
    -----
    1. 使用 `threadpoolctl <https://pypi.org/project/threadpoolctl/>`_ 库获取信息（如果可用）。
    2. SIMD 相关信息从 ``__cpu_features__``, ``__cpu_baseline__`` 和 ``__cpu_dispatch__`` 获取。

    """
    # 导入必要的模块和函数
    from numpy._core._multiarray_umath import (
        __cpu_features__, __cpu_baseline__, __cpu_dispatch__
    )
    from pprint import pprint
    
    # 构建配置信息列表
    config_found = [{
        "numpy_version": np.__version__,  # NumPy 版本信息
        "python": sys.version,  # Python 版本信息
        "uname": platform.uname(),  # 系统平台信息
    }]
    
    features_found, features_not_found = [], []
    # 检查 SIMD 特性，记录支持和不支持的特性
    for feature in __cpu_dispatch__:
        if __cpu_features__[feature]:
            features_found.append(feature)
        else:
            features_not_found.append(feature)
    
    # 添加 SIMD 扩展信息到配置列表中
    config_found.append({
        "simd_extensions": {
            "baseline": __cpu_baseline__,  # 基线 SIMD 支持情况
            "found": features_found,  # 支持的 SIMD 特性列表
            "not_found": features_not_found  # 不支持的 SIMD 特性列表
        }
    })
    
    try:
        # 尝试导入 threadpoolctl 并获取线程池信息
        from threadpoolctl import threadpool_info
        config_found.extend(threadpool_info())
    except ImportError:
        # 如果 threadpoolctl 未安装，显示警告信息
        print("WARNING: `threadpoolctl` not found in system!"
              " Install it by `pip install threadpoolctl`."
              " Once installed, try `np.show_runtime` again"
              " for more detailed build information")
    
    # 使用 pprint 打印配置信息
    pprint(config_found)


# 设置模块为 numpy
@set_module('numpy')
def get_include():
    """
    返回包含 NumPy \\*.h 头文件的目录。

    需要编译依赖 NumPy 的扩展模块可能需要使用此函数找到合适的包含目录。

    Notes
    -----
    在使用 ``setuptools`` 时，例如在 ``setup.py`` 中::

        import numpy as np
        ...
        Extension('extension_name', ...
                  include_dirs=[np.get_include()])
        ...

    在 NumPy 2.0 中引入了 CLI 工具 ``numpy-config``，对于除了 ``setuptools`` 外的构建系统，推荐使用它::

        $ numpy-config --cflags
        -I/path/to/site-packages/numpy/_core/include

        # 或者依赖 pkg-config:
        $ export PKG_CONFIG_PATH=$(numpy-config --pkgconfigdir)
        $ pkg-config --cflags
        -I/path/to/site-packages/numpy/_core/include

    Examples
    --------
    >>> np.get_include()
    '.../site-packages/numpy/core/include'  # 可能会有所不同

    """
    import numpy  # 导入 NumPy 模块
    # 检查 numpy 模块的 show_config 属性是否为 None
    if numpy.show_config is None:
        # 如果为 None，则表示代码运行在 numpy 源代码目录中
        # 构造包含 numpy 核心头文件的路径
        d = os.path.join(os.path.dirname(numpy.__file__), '_core', 'include')
    else:
        # 如果 show_config 不为 None，则表示使用已安装的 numpy 核心头文件
        # 导入 numpy._core 模块
        import numpy._core as _core
        # 构造包含 numpy 核心头文件的路径
        d = os.path.join(os.path.dirname(_core.__file__), 'include')
    # 返回最终确定的头文件路径
    return d
class _Deprecate:
    """
    Decorator class to deprecate old functions.

    Refer to `deprecate` for details.

    See Also
    --------
    deprecate

    """

    def __init__(self, old_name=None, new_name=None, message=None):
        # 初始化方法，设置被弃用函数的旧名称、新名称和消息
        self.old_name = old_name
        self.new_name = new_name
        self.message = message

    def __call__(self, func, *args, **kwargs):
        """
        Decorator call.  Refer to ``deprecate``.

        """
        # 获取被弃用函数的旧名称、新名称和消息
        old_name = self.old_name
        new_name = self.new_name
        message = self.message

        # 如果旧名称未提供，则使用函数的名称
        if old_name is None:
            old_name = func.__name__
        # 如果新名称未提供，则生成默认的弃用文档信息
        if new_name is None:
            depdoc = "`%s` is deprecated!" % old_name
        else:
            depdoc = "`%s` is deprecated, use `%s` instead!" % \
                     (old_name, new_name)

        # 如果有提供消息，则添加到弃用文档信息中
        if message is not None:
            depdoc += "\n" + message

        # 包装被弃用函数，发出 DeprecationWarning 警告
        @functools.wraps(func)
        def newfunc(*args, **kwds):
            warnings.warn(depdoc, DeprecationWarning, stacklevel=2)
            return func(*args, **kwds)

        # 将新函数名称设置为旧名称
        newfunc.__name__ = old_name
        # 处理函数的文档字符串，将弃用信息添加到文档中
        doc = func.__doc__
        if doc is None:
            doc = depdoc
        else:
            lines = doc.expandtabs().split('\n')
            indent = _get_indent(lines[1:])
            if lines[0].lstrip():
                # 缩进原始文档的第一行，以便让 inspect.cleandoc() 去除文档字符串的缩进
                doc = indent * ' ' + doc
            else:
                # 删除与 cleandoc() 相同的前导空行
                skip = len(lines[0]) + 1
                for line in lines[1:]:
                    if len(line) > indent:
                        break
                    skip += len(line) + 1
                doc = doc[skip:]
            # 将弃用信息缩进，并与原始文档组合成新的文档字符串
            depdoc = textwrap.indent(depdoc, ' ' * indent)
            doc = '\n\n'.join([depdoc, doc])
        # 将处理后的文档字符串设置为新函数的文档
        newfunc.__doc__ = doc

        return newfunc


def _get_indent(lines):
    """
    Determines the leading whitespace that could be removed from all the lines.
    """
    # 确定可以从所有行中移除的前导空白
    indent = sys.maxsize
    for line in lines:
        content = len(line.lstrip())
        if content:
            indent = min(indent, len(line) - content)
    if indent == sys.maxsize:
        indent = 0
    return indent


def deprecate(*args, **kwargs):
    """
    Issues a DeprecationWarning, adds warning to `old_name`'s
    docstring, rebinds ``old_name.__name__`` and returns the new
    function object.

    This function may also be used as a decorator.

    .. deprecated:: 2.0
        Use `~warnings.warn` with :exc:`DeprecationWarning` instead.

    Parameters
    ----------
    func : function
        The function to be deprecated.
    old_name : str, optional
        The name of the function to be deprecated. Default is None, in
        which case the name of `func` is used.

    """
    # 定义函数的新名称，默认为 None，若不提供则显示旧名称已废弃的警告信息
    new_name : str, optional
        函数的新名称。默认为 None，如果提供了新名称，则显示旧名称已废弃，建议使用新名称的警告信息。
    message : str, optional
        废弃函数的额外解释说明。会显示在文档字符串中的警告信息之后。

    Returns
    -------
    old_func : function
        废弃的函数对象。

    Examples
    --------
    注意，`olduint` 在打印废弃警告后返回一个值：

    >>> olduint = np.lib.utils.deprecate(np.uint)
    DeprecationWarning: `uint64` is deprecated! # 可能会有所不同
    >>> olduint(6)
    6

    """
    # `deprecate` 可以作为函数或装饰器运行
    # 如果作为函数运行，则初始化装饰器类并执行其 __call__ 方法

    # 在 NumPy 2.0 中废弃，日期为 2023-07-11
    warnings.warn(
        "`deprecate` 已被废弃，请使用 `warn` 与 `DeprecationWarning` 替代。"
        "（在 NumPy 2.0 中废弃）",
        DeprecationWarning,
        stacklevel=2
    )

    if args:
        fn = args[0]
        args = args[1:]

        return _Deprecate(*args, **kwargs)(fn)
    else:
        return _Deprecate(*args, **kwargs)
# Deprecate a function with a specified message.
def deprecate_with_doc(msg):
    """
    Deprecates a function and includes the deprecation in its docstring.

    .. deprecated:: 2.0
        Use `~warnings.warn` with :exc:`DeprecationWarning` instead.

    This function is used as a decorator. It returns an object that can be
    used to issue a DeprecationWarning, by passing the to-be decorated
    function as argument, this adds warning to the to-be decorated function's
    docstring and returns the new function object.

    See Also
    --------
    deprecate : Decorate a function such that it issues a
                :exc:`DeprecationWarning`

    Parameters
    ----------
    msg : str
        Additional explanation of the deprecation. Displayed in the
        docstring after the warning.

    Returns
    -------
    obj : object

    """

    # Issue a warning about the deprecation.
    warnings.warn(
        "`deprecate` is deprecated, "
        "use `warn` with `DeprecationWarning` instead. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    # Return an object to manage the deprecation.
    return _Deprecate(message=msg)


#-----------------------------------------------------------------------------


# NOTE: pydoc defines a help function which works similarly to this
# except it uses a pager to take over the screen.

# Combine name and arguments and split into multiple lines of specified width.
# End lines on a comma and begin argument list indented with the rest of the arguments.
def _split_line(name, arguments, width):
    firstwidth = len(name)
    k = firstwidth
    newstr = name
    sepstr = ", "
    arglist = arguments.split(sepstr)
    for argument in arglist:
        if k == firstwidth:
            addstr = ""
        else:
            addstr = sepstr
        k = k + len(argument) + len(addstr)
        if k > width:
            k = firstwidth + 1 + len(argument)
            newstr = newstr + ",\n" + " "*(firstwidth+2) + argument
        else:
            newstr = newstr + addstr + argument
    return newstr


_namedict = None
_dictlist = None

# Traverse all module directories underneath globals
# to see if something is defined.
def _makenamedict(module='numpy'):
    # Import the specified module and initialize dictionaries.
    module = __import__(module, globals(), locals(), [])
    thedict = {module.__name__: module.__dict__}
    dictlist = [module.__name__]
    totraverse = [module.__dict__]
    while True:
        if len(totraverse) == 0:
            break
        # Traverse through module dictionaries.
        thisdict = totraverse.pop(0)
        for x in thisdict.keys():
            if isinstance(thisdict[x], types.ModuleType):
                # Check if the item is a module and add it to dictionaries for traversal.
                modname = thisdict[x].__name__
                if modname not in dictlist:
                    moddict = thisdict[x].__dict__
                    dictlist.append(modname)
                    totraverse.append(moddict)
                    thedict[modname] = moddict
    return thedict, dictlist


def _info(obj, output=None):
    """Provide information about ndarray obj.

    Parameters
    ----------
    # 定义一个 lambda 函数 bp，用于返回其参数本身，不做任何改变
    bp = lambda x: x
    # 获取对象 obj 的类名
    cls = getattr(obj, '__class__', type(obj))
    # 获取类名的字符串表示，或者使用类本身的名字
    nm = getattr(cls, '__name__', cls)
    # 获取对象的步幅（strides）
    strides = obj.strides
    # 获取对象元素的字节顺序
    endian = obj.dtype.byteorder

    # 如果输出对象 output 为 None，则将其设为标准输出流 sys.stdout
    if output is None:
        output = sys.stdout

    # 打印对象的类名到输出流
    print("class: ", nm, file=output)
    # 打印对象的形状（shape）到输出流
    print("shape: ", obj.shape, file=output)
    # 打印对象的步幅（strides）到输出流
    print("strides: ", strides, file=output)
    # 打印对象每个元素的字节大小（itemsize）到输出流
    print("itemsize: ", obj.itemsize, file=output)
    # 打印对象是否按照特定对齐方式对齐（aligned）到输出流
    print("aligned: ", bp(obj.flags.aligned), file=output)
    # 打印对象是否是连续存储（contiguous）的到输出流
    print("contiguous: ", bp(obj.flags.contiguous), file=output)
    # 打印对象是否按照 Fortran 顺序存储（fortran）到输出流
    print("fortran: ", obj.flags.fortran, file=output)
    # 打印对象的数据指针的十六进制表示和额外信息（如果有的话）到输出流
    print(
        "data pointer: %s%s" % (hex(obj.ctypes._as_parameter_.value), extra),
        file=output
        )
    # 打印对象的字节顺序（byteorder）到输出流
    print("byteorder: ", end=' ', file=output)
    # 根据对象的字节顺序不同打印不同的表示到输出流，并确定是否需要字节交换
    if endian in ['|', '=']:
        print("%s%s%s" % (tic, sys.byteorder, tic), file=output)
        byteswap = False
    elif endian == '>':
        print("%sbig%s" % (tic, tic), file=output)
        byteswap = sys.byteorder != "big"
    else:
        print("%slittle%s" % (tic, tic), file=output)
        byteswap = sys.byteorder != "little"
    # 打印对象是否需要字节交换的布尔值到输出流
    print("byteswap: ", bp(byteswap), file=output)
    # 打印对象的数据类型（dtype）到输出流
    print("type: %s" % obj.dtype, file=output)
@set_module('numpy')
def info(object=None, maxwidth=76, output=None, toplevel='numpy'):
    """
    Get help information for an array, function, class, or module.

    Parameters
    ----------
    object : object or str, optional
        Input object or name to get information about. If `object` is
        an `ndarray` instance, information about the array is printed.
        If `object` is a numpy object, its docstring is given. If it is
        a string, available modules are searched for matching objects.
        If None, information about `info` itself is returned.
    maxwidth : int, optional
        Printing width.
    output : file like object, optional
        File like object that the output is written to, default is
        ``None``, in which case ``sys.stdout`` will be used.
        The object has to be opened in 'w' or 'a' mode.
    toplevel : str, optional
        Start search at this level.

    Notes
    -----
    When used interactively with an object, ``np.info(obj)`` is equivalent
    to ``help(obj)`` on the Python prompt or ``obj?`` on the IPython
    prompt.

    Examples
    --------
    >>> np.info(np.polyval) # doctest: +SKIP
       polyval(p, x)
         Evaluate the polynomial p at x.
         ...

    When using a string for `object` it is possible to get multiple results.

    >>> np.info('fft') # doctest: +SKIP
         *** Found in numpy ***
    Core FFT routines
    ...
         *** Found in numpy.fft ***
     fft(a, n=None, axis=-1)
    ...
         *** Repeat reference found in numpy.fft.fftpack ***
         *** Total of 3 references found. ***

    When the argument is an array, information about the array is printed.

    >>> a = np.array([[1 + 2j, 3, -4], [-5j, 6, 0]], dtype=np.complex64)
    >>> np.info(a)
    class:  ndarray
    shape:  (2, 3)
    strides:  (24, 8)
    itemsize:  8
    aligned:  True
    contiguous:  True
    fortran:  False
    data pointer: 0x562b6e0d2860  # may vary
    byteorder:  little
    byteswap:  False
    type: complex64

    """
    # 定义全局变量 _namedict 和 _dictlist
    global _namedict, _dictlist
    # 为了加快 numpy 导入时间，进行局部导入
    import pydoc
    import inspect

    # 如果 object 拥有 _ppimport_importer 或者 _ppimport_module 属性，则重新赋值为 _ppimport_module
    if (hasattr(object, '_ppimport_importer') or
           hasattr(object, '_ppimport_module')):
        object = object._ppimport_module
    # 如果 object 拥有 _ppimport_attr 属性，则重新赋值为 _ppimport_attr
    elif hasattr(object, '_ppimport_attr'):
        object = object._ppimport_attr

    # 如果 output 参数为 None，则设置 output 为 sys.stdout
    if output is None:
        output = sys.stdout

    # 如果 object 为 None，则调用 info 函数来获取关于 info 函数本身的信息
    if object is None:
        info(info)
    # 如果 object 是 ndarray 的实例，则调用 _info 函数打印关于该数组的信息到指定的 output
    elif isinstance(object, ndarray):
        _info(object, output=output)
    # 如果 object 是字符串类型，则执行以下操作
    elif isinstance(object, str):
        # 如果 _namedict 为空，则调用 _makenamedict 函数生成 _namedict 和 _dictlist
        if _namedict is None:
            _namedict, _dictlist = _makenamedict(toplevel)
        # 初始化找到的对象数量为 0
        numfound = 0
        # 创建一个空列表来存储找到的对象
        objlist = []
        # 遍历 _dictlist 中的 namestr
        for namestr in _dictlist:
            try:
                # 尝试从 _namedict 中获取 namestr 下的 object 对象
                obj = _namedict[namestr][object]
                # 如果该对象已经在 objlist 中，则打印重复引用的警告信息到 output
                if id(obj) in objlist:
                    print("\n     "
                          "*** Repeat reference found in %s *** " % namestr,
                          file=output
                          )
                else:
                    # 否则将该对象的 id 添加到 objlist 中，并打印找到对象的信息到 output
                    objlist.append(id(obj))
                    print("     *** Found in %s ***" % namestr, file=output)
                    # 调用 info 函数打印对象的详细信息到 output
                    info(obj)
                    # 打印分隔线到 output
                    print("-"*maxwidth, file=output)
                # 增加找到的对象数量计数
                numfound += 1
            except KeyError:
                # 如果在 _namedict 中找不到相应的对象，则继续循环
                pass
        # 如果未找到任何对象，则打印未找到帮助信息到 output
        if numfound == 0:
            print("Help for %s not found." % object, file=output)
        else:
            # 否则打印找到的对象总数到 output
            print("\n     "
                  "*** Total of %d references found. ***" % numfound,
                  file=output
                  )

    # 如果 object 是函数或方法，则执行以下操作
    elif inspect.isfunction(object) or inspect.ismethod(object):
        # 获取对象的名称
        name = object.__name__
        try:
            # 尝试获取对象的参数签名，并转换为字符串
            arguments = str(inspect.signature(object))
        except Exception:
            # 如果获取参数签名失败，则将参数字符串设为 "()"
            arguments = "()"

        # 如果对象名称和参数签名的长度超过最大宽度，则将其拆分为符合最大宽度的字符串
        if len(name + arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        # 打印对象名称及参数信息到 output
        print(" " + argstr + "\n", file=output)
        # 打印对象的文档字符串到 output
        print(inspect.getdoc(object), file=output)

    # 如果 object 是类，则执行以下操作
    elif inspect.isclass(object):
        # 获取类的名称
        name = object.__name__
        try:
            # 尝试获取类的参数签名，并转换为字符串
            arguments = str(inspect.signature(object))
        except Exception:
            # 如果获取参数签名失败，则将参数字符串设为 "()"
            arguments = "()"

        # 如果类名称和参数签名的长度超过最大宽度，则将其拆分为符合最大宽度的字符串
        if len(name + arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        # 打印类名称及参数信息到 output
        print(" " + argstr + "\n", file=output)
        # 获取类的文档字符串
        doc1 = inspect.getdoc(object)
        if doc1 is None:
            # 如果类没有文档字符串，但有 __init__ 方法，则打印 __init__ 方法的文档字符串到 output
            if hasattr(object, '__init__'):
                print(inspect.getdoc(object.__init__), file=output)
        else:
            # 否则打印类的文档字符串到 output
            print(inspect.getdoc(object), file=output)

        # 获取类的所有方法
        methods = pydoc.allmethods(object)

        # 过滤出所有公共方法（不以 '_' 开头的方法）
        public_methods = [meth for meth in methods if meth[0] != '_']
        if public_methods:
            # 如果有公共方法，则打印 "Methods:" 到 output
            print("\n\nMethods:\n", file=output)
            # 遍历每个公共方法，并打印方法名及方法的简短描述到 output
            for meth in public_methods:
                thisobj = getattr(object, meth, None)
                if thisobj is not None:
                    # 获取方法的简短描述
                    methstr, other = pydoc.splitdoc(
                            inspect.getdoc(thisobj) or "None"
                            )
                # 打印方法名及其简短描述到 output
                print("  %s  --  %s" % (meth, methstr), file=output)

    # 如果 object 有文档字符串，则打印其文档字符串到 output
    elif hasattr(object, '__doc__'):
        print(inspect.getdoc(object), file=output)
def safe_eval(source):
    """
    Protected string evaluation.

    .. deprecated:: 2.0
        Use `ast.literal_eval` instead.

    Evaluate a string containing a Python literal expression without
    allowing the execution of arbitrary non-literal code.

    .. warning::

        This function is identical to :py:meth:`ast.literal_eval` and
        has the same security implications.  It may not always be safe
        to evaluate large input strings.

    Parameters
    ----------
    source : str
        The string to evaluate.

    Returns
    -------
    obj : object
       The result of evaluating `source`.

    Raises
    ------
    SyntaxError
        If the code has invalid Python syntax, or if it contains
        non-literal code.

    Examples
    --------
    >>> np.safe_eval('1')
    1
    >>> np.safe_eval('[1, 2, 3]')
    [1, 2, 3]
    >>> np.safe_eval('{"foo": ("bar", 10.0)}')
    {'foo': ('bar', 10.0)}

    >>> np.safe_eval('import os')
    Traceback (most recent call last):
      ...
    SyntaxError: invalid syntax

    >>> np.safe_eval('open("/home/user/.ssh/id_dsa").read()')
    Traceback (most recent call last):
      ...
    ValueError: malformed node or string: <_ast.Call object at 0x...>

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`safe_eval` is deprecated. Use `ast.literal_eval` instead. "
        "Be aware of security implications, such as memory exhaustion "
        "based attacks (deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    # Local import to speed up numpy's import time.
    import ast
    # 使用 ast 模块的 literal_eval 方法安全地评估字符串表达式
    return ast.literal_eval(source)


def _median_nancheck(data, result, axis):
    """
    Utility function to check median result from data for NaN values at the end
    and return NaN in that case. Input result can also be a MaskedArray.

    Parameters
    ----------
    data : array
        Sorted input data to median function
    result : Array or MaskedArray
        Result of median function.
    axis : int
        Axis along which the median was computed.

    Returns
    -------
    result : scalar or ndarray
        Median or NaN in axes which contained NaN in the input.  If the input
        was an array, NaN will be inserted in-place.  If a scalar, either the
        input itself or a scalar NaN.
    """
    if data.size == 0:
        return result
    # 获取数据中最后一个元素的潜在 NaN 值
    potential_nans = data.take(-1, axis=axis)
    # 检查潜在 NaN 值是否是 MaskedArray，并处理为普通的 ndarray
    n = np.isnan(potential_nans)
    if np.ma.isMaskedArray(n):
        n = n.filled(False)

    # 如果没有 NaN 值，则直接返回结果
    if not n.any():
        return result

    # 如果结果是 numpy 的标量类型，则直接返回潜在 NaN 值
    if isinstance(result, np.generic):
        return potential_nans

    # 否则复制 NaN 值（如果存在）
    # 将 potential_nans 中的值复制到 result 中，仅在 where 中对应的位置为 True 时进行复制
    np.copyto(result, potential_nans, where=n)
    # 返回复制后的 result 数组作为函数的结果
    return result
# 返回当前构建支持的 CPU 特性的字符串表示

def _opt_info():
    """
    Returns a string containing the CPU features supported
    by the current build.

    The format of the string can be explained as follows:
        - Dispatched features supported by the running machine end with `*`.
        - Dispatched features not supported by the running machine
          end with `?`.
        - Remaining features represent the baseline.

    Returns:
        str: A formatted string indicating the supported CPU features.
    """
    # 导入必要的模块和变量
    from numpy._core._multiarray_umath import (
        __cpu_features__, __cpu_baseline__, __cpu_dispatch__
    )

    # 如果没有基准特性和调度特性，则返回空字符串
    if len(__cpu_baseline__) == 0 and len(__cpu_dispatch__) == 0:
        return ''

    # 初始化启用特性的字符串为基准特性
    enabled_features = ' '.join(__cpu_baseline__)

    # 遍历调度特性，根据实际机器支持情况添加 '*' 或 '?'
    for feature in __cpu_dispatch__:
        if __cpu_features__[feature]:
            enabled_features += f" {feature}*"
        else:
            enabled_features += f" {feature}?"

    return enabled_features

# 返回不包含元数据的 dtype，如果包含元数据则返回其复制品

def drop_metadata(dtype, /):
    """
    Returns the dtype unchanged if it contained no metadata or a copy of the
    dtype if it (or any of its structure dtypes) contained metadata.

    This utility is used by `np.save` and `np.savez` to drop metadata before
    saving.

    .. note::

        Due to its limitation this function may move to a more appropriate
        home or change in the future and is considered semi-public API only.

    .. warning::

        This function does not preserve more strange things like record dtypes
        and user dtypes may simply return the wrong thing.  If you need to be
        sure about the latter, check the result with:
        ``np.can_cast(new_dtype, dtype, casting="no")``.

    """
    # 如果 dtype 包含字段信息
    if dtype.fields is not None:
        found_metadata = dtype.metadata is not None

        names = []
        formats = []
        offsets = []
        titles = []
        # 遍历 dtype 的字段信息
        for name, field in dtype.fields.items():
            # 递归调用 drop_metadata 处理字段的 dtype
            field_dt = drop_metadata(field[0])
            # 如果字段 dtype 发生变化，设置 found_metadata 为 True
            if field_dt is not field[0]:
                found_metadata = True

            names.append(name)
            formats.append(field_dt)
            offsets.append(field[1])
            titles.append(None if len(field) < 3 else field[2])

        # 如果没有发现元数据，则返回原始的 dtype
        if not found_metadata:
            return dtype

        # 构建新的 dtype 结构体
        structure = dict(
            names=names, formats=formats, offsets=offsets, titles=titles,
            itemsize=dtype.itemsize)

        # NOTE: Could pass (dtype.type, structure) to preserve record dtypes...
        return np.dtype(structure, align=dtype.isalignedstruct)
    # 如果 dtype 是子数组 dtype
    elif dtype.subdtype is not None:
        # 获取子数组 dtype 和形状
        subdtype, shape = dtype.subdtype
        # 递归调用 drop_metadata 处理子数组 dtype
        new_subdtype = drop_metadata(subdtype)
        # 如果原始 dtype 没有元数据且子数组 dtype 没有变化，则返回原始 dtype
        if dtype.metadata is None and new_subdtype is subdtype:
            return dtype

        # 构建新的子数组 dtype
        return np.dtype((new_subdtype, shape))
    else:
        # 如果不是结构化数据类型，即普通的非结构化数据类型
        # 检查数据类型是否有元数据
        if dtype.metadata is None:
            # 如果没有元数据，直接返回该数据类型
            return dtype
        # 对于没有元数据的数据类型，需要注意 `dt.str` 并不能完全回路（round-trip），例如对于用户定义的数据类型。
        # 返回该数据类型的字符串表示的 NumPy 数据类型
        return np.dtype(dtype.str)
```