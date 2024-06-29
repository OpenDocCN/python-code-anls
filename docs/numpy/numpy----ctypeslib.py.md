# `.\numpy\numpy\ctypeslib.py`

```py
"""
============================
``ctypes`` Utility Functions
============================

See Also
--------
load_library : Load a C library.
ndpointer : Array restype/argtype with verification.
as_ctypes : Create a ctypes array from an ndarray.
as_array : Create an ndarray from a ctypes array.

References
----------
.. [1] "SciPy Cookbook: ctypes", https://scipy-cookbook.readthedocs.io/items/Ctypes.html

Examples
--------
Load the C library:

>>> _lib = np.ctypeslib.load_library('libmystuff', '.')     #doctest: +SKIP

Our result type, an ndarray that must be of type double, be 1-dimensional
and is C-contiguous in memory:

>>> array_1d_double = np.ctypeslib.ndpointer(
...                          dtype=np.double,
...                          ndim=1, flags='CONTIGUOUS')    #doctest: +SKIP

Our C-function typically takes an array and updates its values
in-place.  For example::

    void foo_func(double* x, int length)
    {
        int i;
        for (i = 0; i < length; i++) {
            x[i] = i*i;
        }
    }

We wrap it using:

>>> _lib.foo_func.restype = None                      #doctest: +SKIP
>>> _lib.foo_func.argtypes = [array_1d_double, c_int] #doctest: +SKIP

Then, we're ready to call ``foo_func``:

>>> out = np.empty(15, dtype=np.double)
>>> _lib.foo_func(out, len(out))                #doctest: +SKIP

"""
__all__ = ['load_library', 'ndpointer', 'c_intp', 'as_ctypes', 'as_array',
           'as_ctypes_type']

import os                                       # 导入操作系统接口模块
from numpy import (                             # 从 numpy 库导入以下模块：
    integer, ndarray, dtype as _dtype, asarray, frombuffer
)
from numpy._core.multiarray import _flagdict, flagsobj  # 导入 numpy 的内部多维数组相关模块

try:
    import ctypes                               # 尝试导入 ctypes 库
except ImportError:
    ctypes = None

if ctypes is None:                              # 如果 ctypes 库不可用，定义一个 _dummy 函数来抛出 ImportError
    def _dummy(*args, **kwds):
        """
        Dummy object that raises an ImportError if ctypes is not available.

        Raises
        ------
        ImportError
            If ctypes is not available.

        """
        raise ImportError("ctypes is not available.")
    load_library = _dummy                        # 定义 load_library, as_ctypes, as_array 为 _dummy 函数
    as_ctypes = _dummy
    as_array = _dummy
    from numpy import intp as c_intp            # 从 numpy 导入 intp 并命名为 c_intp
    _ndptr_base = object                        # _ndptr_base 设为 Python 对象
else:
    import numpy._core._internal as nic          # 导入 numpy 内部的 _internal 模块并命名为 nic
    c_intp = nic._getintp_ctype()               # 获取 ctypes 中的 c_intp 类型
    del nic                                     # 删除 nic 引用，释放内存
    _ndptr_base = ctypes.c_void_p               # 设置 _ndptr_base 为 ctypes 的 c_void_p 类型

    # Adapted from Albert Strasheim
    def load_library(libname, loader_path):
        """
        It is possible to load a library using

        >>> lib = ctypes.cdll[<full_path_name>] # doctest: +SKIP

        But there are cross-platform considerations, such as library file extensions,
        plus the fact Windows will just load the first library it finds with that name.
        NumPy supplies the load_library function as a convenience.

        .. versionchanged:: 1.20.0
            Allow libname and loader_path to take any
            :term:`python:path-like object`.

        Parameters
        ----------
        libname : path-like
            Name of the library, which can have 'lib' as a prefix,
            but without an extension.
        loader_path : path-like
            Where the library can be found.

        Returns
        -------
        ctypes.cdll[libpath] : library object
           A ctypes library object

        Raises
        ------
        OSError
            If there is no library with the expected extension, or the
            library is defective and cannot be loaded.
        """
        # Convert path-like objects into strings
        libname = os.fsdecode(libname)  # 解码并获取库名称的字符串表示
        loader_path = os.fsdecode(loader_path)  # 解码并获取加载路径的字符串表示

        ext = os.path.splitext(libname)[1]  # 获取库名称的文件扩展名
        if not ext:
            import sys
            import sysconfig
            # 尝试使用平台特定的库文件名加载库，否则默认为libname.[so|dll|dylib]。
            # 有时这些文件在非Linux平台上会构建错误。
            base_ext = ".so"
            if sys.platform.startswith("darwin"):
                base_ext = ".dylib"
            elif sys.platform.startswith("win"):
                base_ext = ".dll"
            libname_ext = [libname + base_ext]
            so_ext = sysconfig.get_config_var("EXT_SUFFIX")
            if not so_ext == base_ext:
                libname_ext.insert(0, libname + so_ext)
        else:
            libname_ext = [libname]

        loader_path = os.path.abspath(loader_path)  # 获取加载路径的绝对路径
        if not os.path.isdir(loader_path):
            libdir = os.path.dirname(loader_path)  # 获取加载路径的父目录
        else:
            libdir = loader_path

        for ln in libname_ext:
            libpath = os.path.join(libdir, ln)  # 组合库文件的完整路径
            if os.path.exists(libpath):  # 检查库文件是否存在
                try:
                    return ctypes.cdll[libpath]  # 尝试加载库文件并返回 ctypes 库对象
                except OSError:
                    ## defective lib file
                    raise  # 抛出异常，说明库文件有问题
        ## if no successful return in the libname_ext loop:
        raise OSError("no file with expected extension")  # 如果在 libname_ext 循环中没有成功返回，则抛出异常
def _num_fromflags(flaglist):
    # 初始化一个计数器
    num = 0
    # 遍历传入的标志列表，将每个标志对应的值累加到计数器中
    for val in flaglist:
        num += _flagdict[val]
    # 返回累加后的结果作为标志的数字表示
    return num

_flagnames = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE',
              'OWNDATA', 'WRITEBACKIFCOPY']
def _flags_fromnum(num):
    # 初始化一个空列表用来存储结果
    res = []
    # 遍历已定义的所有标志名称
    for key in _flagnames:
        # 获取当前标志对应的数值
        value = _flagdict[key]
        # 检查当前标志是否在给定的数字表示中
        if (num & value):
            # 如果在其中，则将该标志名称添加到结果列表中
            res.append(key)
    # 返回所有匹配的标志名称列表
    return res


class _ndptr(_ndptr_base):
    @classmethod
    def from_param(cls, obj):
        # 检查传入的对象是否为 ndarray 类型
        if not isinstance(obj, ndarray):
            raise TypeError("argument must be an ndarray")
        # 如果定义了特定的数据类型，检查传入数组是否符合要求
        if cls._dtype_ is not None \
               and obj.dtype != cls._dtype_:
            raise TypeError("array must have data type %s" % cls._dtype_)
        # 如果定义了特定的维度数，检查传入数组是否符合要求
        if cls._ndim_ is not None \
               and obj.ndim != cls._ndim_:
            raise TypeError("array must have %d dimension(s)" % cls._ndim_)
        # 如果定义了特定的形状，检查传入数组是否符合要求
        if cls._shape_ is not None \
               and obj.shape != cls._shape_:
            raise TypeError("array must have shape %s" % str(cls._shape_))
        # 如果定义了特定的标志，检查传入数组是否符合要求
        if cls._flags_ is not None \
               and ((obj.flags.num & cls._flags_) != cls._flags_):
            raise TypeError("array must have flags %s" %
                    _flags_fromnum(cls._flags_))
        # 返回传入数组的 ctypes 对象
        return obj.ctypes


class _concrete_ndptr(_ndptr):
    """
    Like _ndptr, but with `_shape_` and `_dtype_` specified.

    Notably, this means the pointer has enough information to reconstruct
    the array, which is not generally true.
    """
    def _check_retval_(self):
        """
        This method is called when this class is used as the .restype
        attribute for a shared-library function, to automatically wrap the
        pointer into an array.
        """
        # 返回指针指向的数据作为 ndarray 对象
        return self.contents

    @property
    def contents(self):
        """
        Get an ndarray viewing the data pointed to by this pointer.

        This mirrors the `contents` attribute of a normal ctypes pointer
        """
        # 构建完整的数据类型描述
        full_dtype = _dtype((self._dtype_, self._shape_))
        # 根据完整的数据类型描述创建对应的 ctypes 类型
        full_ctype = ctypes.c_char * full_dtype.itemsize
        # 将当前指针对象转换为指向完整 ctypes 类型的指针，并获取其内容
        buffer = ctypes.cast(self, ctypes.POINTER(full_ctype)).contents
        # 将 ctypes 缓冲区转换为 ndarray，并去掉多余的维度
        return frombuffer(buffer, dtype=full_dtype).squeeze(axis=0)


# Factory for an array-checking class with from_param defined for
#  use with ctypes argtypes mechanism
_pointer_type_cache = {}
def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    """
    Array-checking restype/argtypes.

    An ndpointer instance is used to describe an ndarray in restypes
    and argtypes specifications.  This approach is more flexible than
    using, for example, ``POINTER(c_double)``, since several restrictions
    can be specified, which are verified upon calling the ctypes function.
    These include data type, number of dimensions, shape and flags.  If a
    given array does not satisfy the specified restrictions,
    a ``TypeError`` is raised.

    Parameters
    ----------
    """
    # 此函数主要用于创建一个描述 ndarray 的类型，检查其数据类型、维度、形状和标志
    pass
    # data-type 数据类型，可选参数
    dtype : data-type, optional
    # int 整数，可选参数
    ndim : int, optional
    # tuple of ints 整数元组，可选参数
    shape : tuple of ints, optional
    # str or tuple of str 字符串或字符串元组，数组标志；可以是以下一项或多项：
    # C_CONTIGUOUS / C / CONTIGUOUS
    # F_CONTIGUOUS / F / FORTRAN
    # OWNDATA / O
    # WRITEABLE / W
    # ALIGNED / A
    # WRITEBACKIFCOPY / X

    # 返回
    -------
    # ndpointer 类型对象
    klass : ndpointer type object
    # 类型对象，是一个包含 dtype、ndim、shape 和 flags 信息的 `_ndtpr` 实例。

    # 异常
    ------
    # TypeError
    # 如果给定的数组不满足指定的限制条件。

    # 示例
    --------
    # 将 clib.somefunc 的 argtypes 设置为 [np.ctypeslib.ndpointer(dtype=np.float64,
    #                                                  ndim=1,
    #                                                  flags='C_CONTIGUOUS')]
    ... #doctest: +SKIP
    # 调用 clib.somefunc，传入 np.array([1, 2, 3], dtype=np.float64) 作为参数
    ... #doctest: +SKIP
    """

    # 将 dtype 标准化为 Optional[dtype]
    if dtype is not None:
        dtype = _dtype(dtype)

    # 将 flags 标准化为 Optional[int]
    num = None
    if flags is not None:
        if isinstance(flags, str):
            flags = flags.split(',')
        elif isinstance(flags, (int, integer)):
            num = flags
            flags = _flags_fromnum(num)
        elif isinstance(flags, flagsobj):
            num = flags.num
            flags = _flags_fromnum(num)
        if num is None:
            try:
                flags = [x.strip().upper() for x in flags]
            except Exception as e:
                raise TypeError("invalid flags specification") from e
            num = _num_fromflags(flags)

    # 将 shape 标准化为 Optional[tuple]
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError:
            # 单个整数 -> 转为 1 元组
            shape = (shape,)

    # 缓存键，包含 dtype、ndim、shape 和 num
    cache_key = (dtype, ndim, shape, num)

    try:
        # 尝试从缓存中获取 _pointer_type_cache 中的值
        return _pointer_type_cache[cache_key]
    except KeyError:
        pass

    # 为新类型生成一个名称
    if dtype is None:
        name = 'any'
    elif dtype.names is not None:
        name = str(id(dtype))
    else:
        name = dtype.str
    if ndim is not None:
        name += "_%dd" % ndim
    if shape is not None:
        name += "_"+"x".join(str(x) for x in shape)
    if flags is not None:
        name += "_"+"_".join(flags)

    # 如果 dtype 和 shape 都不为 None，则基于 _concrete_ndptr
    # 否则基于 _ndptr
    if dtype is not None and shape is not None:
        base = _concrete_ndptr
    else:
        base = _ndptr

    # 创建一个新类型 klass，类型名为 'ndpointer_%s' % name
    klass = type("ndpointer_%s"%name, (base,),
                 {"_dtype_": dtype,
                  "_shape_" : shape,
                  "_ndim_" : ndim,
                  "_flags_" : num})
    # 将 klass 存储到 _pointer_type_cache 中，使用 cache_key 作为键
    _pointer_type_cache[cache_key] = klass
    return klass
if ctypes is not None:
    # 定义函数 _ctype_ndarray，用于创建给定元素类型和形状的 ndarray
    def _ctype_ndarray(element_type, shape):
        """ Create an ndarray of the given element type and shape """
        # 反向遍历形状，逐步构建元素类型
        for dim in shape[::-1]:
            element_type = dim * element_type
            # 防止类型名称包含 np.ctypeslib
            element_type.__module__ = None
        return element_type


    # 定义函数 _get_scalar_type_map，返回将本机字节序标量 dtype 映射到 ctypes 类型的字典
    def _get_scalar_type_map():
        """
        Return a dictionary mapping native endian scalar dtype to ctypes types
        """
        ct = ctypes
        # 定义简单的 ctypes 类型列表
        simple_types = [
            ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
            ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
            ct.c_float, ct.c_double,
            ct.c_bool,
        ]
        # 返回字典，映射 dtype 到对应的 ctypes 类型
        return {_dtype(ctype): ctype for ctype in simple_types}


    # 获取本机字节序标量 dtype 到 ctypes 类型的映射
    _scalar_type_map = _get_scalar_type_map()


    # 定义函数 _ctype_from_dtype_scalar，根据 dtype 返回对应的 ctypes 类型
    def _ctype_from_dtype_scalar(dtype):
        # 确保将 `=` 转换为本机字节序的 <, >, 或 |
        dtype_with_endian = dtype.newbyteorder('S').newbyteorder('S')
        dtype_native = dtype.newbyteorder('=')
        try:
            # 根据本机字节序的 dtype 获取对应的 ctypes 类型
            ctype = _scalar_type_map[dtype_native]
        except KeyError as e:
            # 抛出异常，表示无法转换该 dtype 到 ctypes 类型
            raise NotImplementedError(
                "Converting {!r} to a ctypes type".format(dtype)
            ) from None

        # 根据 dtype 的字节序调整 ctypes 类型
        if dtype_with_endian.byteorder == '>':
            ctype = ctype.__ctype_be__
        elif dtype_with_endian.byteorder == '<':
            ctype = ctype.__ctype_le__

        return ctype


    # 定义函数 _ctype_from_dtype_subarray，根据 dtype 的子数组返回对应的 ctypes 类型
    def _ctype_from_dtype_subarray(dtype):
        # 获取元素 dtype 和形状
        element_dtype, shape = dtype.subdtype
        # 根据元素 dtype 获取对应的 ctypes 类型
        ctype = _ctype_from_dtype(element_dtype)
        # 使用 _ctype_ndarray 创建 ndarray 类型，并返回
        return _ctype_ndarray(ctype, shape)
    # 根据结构化数据类型（dtype）创建对应的 ctypes 类型，支持嵌套结构和数组
    def _ctype_from_dtype_structured(dtype):
        # 提取每个字段的偏移量信息
        field_data = []
        for name in dtype.names:
            # 获取字段的数据类型和偏移量
            field_dtype, offset = dtype.fields[name][:2]
            # 将字段信息存入列表
            field_data.append((offset, name, _ctype_from_dtype(field_dtype)))
    
        # ctypes 不关心字段的顺序，按偏移量排序字段信息
        field_data = sorted(field_data, key=lambda f: f[0])
    
        # 如果有多个字段且所有字段的偏移量均为 0，则为联合体（union）
        if len(field_data) > 1 and all(offset == 0 for offset, name, ctype in field_data):
            # 初始化联合体的大小和字段列表
            size = 0
            _fields_ = []
            for offset, name, ctype in field_data:
                _fields_.append((name, ctype))
                size = max(size, ctypes.sizeof(ctype))
    
            # 如果结构体的总大小与 dtype 中定义的大小不一致，则添加填充字段
            if dtype.itemsize != size:
                _fields_.append(('', ctypes.c_char * dtype.itemsize))
    
            # 手动插入了填充字段，因此总是设置 `_pack_` 为 1
            return type('union', (ctypes.Union,), dict(
                _fields_=_fields_,
                _pack_=1,
                __module__=None,
            ))
        else:
            last_offset = 0
            _fields_ = []
            for offset, name, ctype in field_data:
                # 计算字段之间的填充空间
                padding = offset - last_offset
                if padding < 0:
                    raise NotImplementedError("Overlapping fields")
                if padding > 0:
                    _fields_.append(('', ctypes.c_char * padding))
    
                _fields_.append((name, ctype))
                last_offset = offset + ctypes.sizeof(ctype)
    
            # 计算最后一个字段之后的填充空间
            padding = dtype.itemsize - last_offset
            if padding > 0:
                _fields_.append(('', ctypes.c_char * padding))
    
            # 手动插入了填充字段，因此总是设置 `_pack_` 为 1
            return type('struct', (ctypes.Structure,), dict(
                _fields_=_fields_,
                _pack_=1,
                __module__=None,
            ))
    
    
    def _ctype_from_dtype(dtype):
        # 如果数据类型具有字段信息，则调用 _ctype_from_dtype_structured 处理
        if dtype.fields is not None:
            return _ctype_from_dtype_structured(dtype)
        # 如果数据类型具有子数据类型信息，则调用 _ctype_from_dtype_subarray 处理
        elif dtype.subdtype is not None:
            return _ctype_from_dtype_subarray(dtype)
        # 否则，将数据类型视为标量，调用 _ctype_from_dtype_scalar 处理
        else:
            return _ctype_from_dtype_scalar(dtype)
    def as_ctypes_type(dtype):
        r"""
        Convert a dtype into a ctypes type.

        Parameters
        ----------
        dtype : dtype
            The dtype to convert

        Returns
        -------
        ctype
            A ctype scalar, union, array, or struct

        Raises
        ------
        NotImplementedError
            If the conversion is not possible

        Notes
        -----
        This function does not losslessly round-trip in either direction.

        ``np.dtype(as_ctypes_type(dt))`` will:

        - insert padding fields
        - reorder fields to be sorted by offset
        - discard field titles

        ``as_ctypes_type(np.dtype(ctype))`` will:

        - discard the class names of `ctypes.Structure`\ s and
          `ctypes.Union`\ s
        - convert single-element `ctypes.Union`\ s into single-element
          `ctypes.Structure`\ s
        - insert padding fields

        """
        # 调用内部函数 _ctype_from_dtype 进行 dtype 到 ctypes 类型的转换
        return _ctype_from_dtype(_dtype(dtype))


    def as_array(obj, shape=None):
        """
        Create a numpy array from a ctypes array or POINTER.

        The numpy array shares the memory with the ctypes object.

        The shape parameter must be given if converting from a ctypes POINTER.
        The shape parameter is ignored if converting from a ctypes array
        """
        if isinstance(obj, ctypes._Pointer):
            # 如果 obj 是 ctypes._Pointer 类型，则将其转换为指定 shape 的数组
            # 如果 shape 为 None，则抛出 TypeError 异常
            if shape is None:
                raise TypeError(
                    'as_array() requires a shape argument when called on a '
                    'pointer')
            # 构造指向 obj 的指针类型 p_arr_type
            p_arr_type = ctypes.POINTER(_ctype_ndarray(obj._type_, shape))
            # 使用 ctypes.cast 将 obj 转换为 p_arr_type 指向的内容（数组）
            obj = ctypes.cast(obj, p_arr_type).contents

        # 调用 asarray 函数，返回 obj 的 numpy 数组表示
        return asarray(obj)


    def as_ctypes(obj):
        """Create and return a ctypes object from a numpy array.  Actually
        anything that exposes the __array_interface__ is accepted."""
        # 获取 obj 的 __array_interface__
        ai = obj.__array_interface__
        # 如果数组是 strided arrays，则抛出 TypeError 异常
        if ai["strides"]:
            raise TypeError("strided arrays not supported")
        # 如果 __array_interface__ 的版本不是 3，则抛出 TypeError 异常
        if ai["version"] != 3:
            raise TypeError("only __array_interface__ version 3 supported")
        # 获取数组的数据地址和 readonly 属性
        addr, readonly = ai["data"]
        # 如果数组是只读的，则抛出 TypeError 异常
        if readonly:
            raise TypeError("readonly arrays unsupported")

        # 根据 ai["typestr"] 调用 as_ctypes_type 函数转换为对应的 ctypes 标量类型
        ctype_scalar = as_ctypes_type(ai["typestr"])
        # 构造一个 ctypes 对象，类型是 _ctype_ndarray(ctype_scalar, ai["shape"])
        result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
        # 使用 from_address 方法从地址 addr 创建 result_type 类型的对象 result
        result = result_type.from_address(addr)
        # 将 obj 保存在 result 的 __keep 属性中
        result.__keep = obj
        return result
```