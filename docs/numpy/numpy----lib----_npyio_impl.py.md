# `.\numpy\numpy\lib\_npyio_impl.py`

```py
"""
IO related functions.
"""
# 引入必要的库和模块
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式操作的支持
import functools  # 提供高阶函数的支持
import itertools  # 提供迭代工具函数的支持
import warnings  # 提供警告管理的功能
import weakref  # 提供弱引用对象的支持
import contextlib  # 提供上下文管理工具的支持
import operator  # 提供各种运算符相关的函数

from operator import itemgetter, index as opindex, methodcaller  # 导入特定的操作符和函数
from collections.abc import Mapping  # 导入抽象基类 Mapping

import pickle  # 提供对象序列化和反序列化的功能

import numpy as np  # 导入 NumPy 库
from . import format  # 导入当前包中的 format 模块
from ._datasource import DataSource  # 导入 DataSource 类
from numpy._core import overrides  # 导入 NumPy 中的 overrides 模块
from numpy._core.multiarray import packbits, unpackbits  # 导入 NumPy 中的 packbits 和 unpackbits 函数
from numpy._core._multiarray_umath import _load_from_filelike  # 导入 NumPy 中的 _load_from_filelike 函数
from numpy._core.overrides import set_array_function_like_doc, set_module  # 导入 NumPy 中的函数
from ._iotools import (  # 导入当前包中的 _iotools 模块的多个函数和类
    LineSplitter, NameValidator, StringConverter, ConverterError,
    ConverterLockError, ConversionWarning, _is_string_like,
    has_nested_fields, flatten_dtype, easy_dtype, _decode_line
    )
from numpy._utils import asunicode, asbytes  # 导入 NumPy 中的字符串处理函数


__all__ = [
    'savetxt', 'loadtxt', 'genfromtxt', 'load', 'save', 'savez',
    'savez_compressed', 'packbits', 'unpackbits', 'fromregex'
    ]


# 创建一个偏函数，用于将模块设置为 'numpy' 的数组函数分发
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


class BagObj:
    """
    BagObj(obj)

    Convert attribute look-ups to getitems on the object passed in.

    Parameters
    ----------
    obj : class instance
        Object on which attribute look-up is performed.

    Examples
    --------
    >>> from numpy.lib._npyio_impl import BagObj as BO
    >>> class BagDemo:
    ...     def __getitem__(self, key): # An instance of BagObj(BagDemo)
    ...                                 # will call this method when any
    ...                                 # attribute look-up is required
    ...         result = "Doesn't matter what you want, "
    ...         return result + "you're gonna get this"
    ...
    >>> demo_obj = BagDemo()
    >>> bagobj = BO(demo_obj)
    >>> bagobj.hello_there
    "Doesn't matter what you want, you're gonna get this"
    >>> bagobj.I_can_be_anything
    "Doesn't matter what you want, you're gonna get this"

    """

    def __init__(self, obj):
        # 使用弱引用将对象 obj 封装成 BagObj 的私有属性 _obj
        self._obj = weakref.proxy(obj)

    def __getattribute__(self, key):
        try:
            # 尝试从私有属性 _obj 中获取键为 key 的值
            return object.__getattribute__(self, '_obj')[key]
        except KeyError:
            # 如果键不存在，则抛出 AttributeError
            raise AttributeError(key) from None

    def __dir__(self):
        """
        Enables dir(bagobj) to list the files in an NpzFile.

        This also enables tab-completion in an interpreter or IPython.
        """
        # 返回私有属性 _obj 的所有键，以实现 dir(bagobj) 的功能
        return list(object.__getattribute__(self, '_obj').keys())


def zipfile_factory(file, *args, **kwargs):
    """
    Create a ZipFile.

    Allows for Zip64, and the `file` argument can accept file, str, or
    pathlib.Path objects. `args` and `kwargs` are passed to the zipfile.ZipFile
    constructor.
    """
    # 如果 file 没有 read 方法，则将其视为路径，并将其转换为 str 类型
    if not hasattr(file, 'read'):
        file = os.fspath(file)
    import zipfile  # 动态导入 zipfile 模块
    kwargs['allowZip64'] = True  # 设置允许 Zip64
    # 创建并返回一个 ZipFile 对象，传入 file、args 和 kwargs
    return zipfile.ZipFile(file, *args, **kwargs)
# 设置模块名称为'numpy.lib.npyio'，用于标识这个类所属的模块
@set_module('numpy.lib.npyio')
# 定义一个类 NpzFile，它继承自 Mapping（映射类型），表明它是一个类字典的对象
class NpzFile(Mapping):
    """
    NpzFile(fid)

    A dictionary-like object with lazy-loading of files in the zipped
    archive provided on construction.

    `NpzFile` is used to load files in the NumPy ``.npz`` data archive
    format. It assumes that files in the archive have a ``.npy`` extension,
    other files are ignored.

    The arrays and file strings are lazily loaded on either
    getitem access using ``obj['key']`` or attribute lookup using
    ``obj.f.key``. A list of all files (without ``.npy`` extensions) can
    be obtained with ``obj.files`` and the ZipFile object itself using
    ``obj.zip``.

    Attributes
    ----------
    files : list of str
        List of all files in the archive with a ``.npy`` extension.
    zip : ZipFile instance
        The ZipFile object initialized with the zipped archive.
    f : BagObj instance
        An object on which attribute can be performed as an alternative
        to getitem access on the `NpzFile` instance itself.
    allow_pickle : bool, optional
        Allow loading pickled data. Default: False

        .. versionchanged:: 1.16.3
            Made default False in response to CVE-2019-6446.

    pickle_kwargs : dict, optional
        Additional keyword arguments to pass on to pickle.load.
        These are only useful when loading object arrays saved on
        Python 2 when using Python 3.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.
        This option is ignored when `allow_pickle` is passed.  In that case
        the file is by definition trusted and the limit is unnecessary.

    Parameters
    ----------
    fid : file, str, or pathlib.Path
        The zipped archive to open. This is either a file-like object
        or a string containing the path to the archive.
    own_fid : bool, optional
        Whether NpzFile should close the file handle.
        Requires that `fid` is a file-like object.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)
    >>> np.savez(outfile, x=x, y=y)
    >>> _ = outfile.seek(0)

    >>> npz = np.load(outfile)
    >>> isinstance(npz, np.lib.npyio.NpzFile)
    True
    >>> npz
    NpzFile 'object' with keys: x, y
    >>> sorted(npz.files)
    ['x', 'y']
    >>> npz['x']  # getitem access
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> npz.f.x  # attribute lookup
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    # 初始化类属性
    # 设置 zip 属性为 None，表示尚未初始化 ZipFile 对象
    zip = None
    # 设置 fid 属性为 None，表示尚未初始化文件对象
    fid = None
    # 设置 _MAX_REPR_ARRAY_COUNT 属性为 5，用于控制数组在输出时的最大展示数量
    _MAX_REPR_ARRAY_COUNT = 5
    # 初始化方法，用于设置对象的初始状态
    def __init__(self, fid, own_fid=False, allow_pickle=False,
                 pickle_kwargs=None, *,
                 max_header_size=format._MAX_HEADER_SIZE):
        # 延迟导入 zipfile，因为 zipfile 依赖 gzip，后者是标准库的可选组件
        _zip = zipfile_factory(fid)
        # 获取 ZIP 文件中所有文件的文件名列表
        self._files = _zip.namelist()
        # 初始化空列表，用于存储除去后缀 ".npy" 之后的文件名
        self.files = []
        # 是否允许使用 pickle 序列化
        self.allow_pickle = allow_pickle
        # 设置最大头部大小
        self.max_header_size = max_header_size
        # pickle 序列化的参数
        self.pickle_kwargs = pickle_kwargs
        # 遍历文件名列表，去除 ".npy" 后缀并添加到 self.files 中
        for x in self._files:
            if x.endswith('.npy'):
                self.files.append(x[:-4])
            else:
                self.files.append(x)
        # 将 zipfile 对象赋值给 self.zip
        self.zip = _zip
        # 创建 BagObj 对象并赋值给 self.f
        self.f = BagObj(self)
        # 如果 own_fid 为 True，则将 fid 赋值给 self.fid
        if own_fid:
            self.fid = fid

    # 进入上下文管理器时调用，返回对象自身
    def __enter__(self):
        return self

    # 退出上下文管理器时调用，关闭文件
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # 关闭文件的方法
    def close(self):
        """
        Close the file.

        """
        # 如果 self.zip 不为 None，则关闭 self.zip
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        # 如果 self.fid 不为 None，则关闭 self.fid
        if self.fid is not None:
            self.fid.close()
            self.fid = None
        # 将 self.f 置为 None，打破引用循环
        self.f = None  # break reference cycle

    # 对象被删除时调用，关闭文件
    def __del__(self):
        self.close()

    # 实现 Mapping ABC 的迭代方法
    def __iter__(self):
        return iter(self.files)

    # 实现 Mapping ABC 的长度方法
    def __len__(self):
        return len(self.files)

    # 实现 Mapping ABC 的获取元素方法
    def __getitem__(self, key):
        # FIXME: This seems like it will copy strings around
        #   more than is strictly necessary.  The zipfile
        #   will read the string and then
        #   the format.read_array will copy the string
        #   to another place in memory.
        #   It would be better if the zipfile could read
        #   (or at least uncompress) the data
        #   directly into the array memory.
        # 检查 key 是否在 self._files 中或者在 self.files 中
        member = False
        if key in self._files:
            member = True
        elif key in self.files:
            member = True
            key += '.npy'
        # 如果 key 是成员，则打开对应的字节流
        if member:
            bytes = self.zip.open(key)
            # 读取格式的魔术前缀
            magic = bytes.read(len(format.MAGIC_PREFIX))
            bytes.close()
            # 如果魔术前缀匹配，则调用 format.read_array 读取数组数据
            if magic == format.MAGIC_PREFIX:
                bytes = self.zip.open(key)
                return format.read_array(bytes,
                                         allow_pickle=self.allow_pickle,
                                         pickle_kwargs=self.pickle_kwargs,
                                         max_header_size=self.max_header_size)
            else:
                # 否则直接读取文件内容
                return self.zip.read(key)
        else:
            # 如果 key 不是成员，则抛出 KeyError
            raise KeyError(f"{key} is not a file in the archive")

    # 实现 Mapping ABC 的成员检查方法
    def __contains__(self, key):
        return (key in self._files or key in self.files)
    # 返回对象的字符串表示形式，用于显示对象的信息
    def __repr__(self):
        # 如果 self.fid 是字符串，则使用它作为文件名，否则尝试获取 self.fid 的 name 属性作为文件名，如果都不存在则使用默认值 "object"
        if isinstance(self.fid, str):
            filename = self.fid
        else:
            filename = getattr(self.fid, "name", "object")

        # 获取数组的名称列表，最多显示 self._MAX_REPR_ARRAY_COUNT 个数组的名称
        array_names = ', '.join(self.files[:self._MAX_REPR_ARRAY_COUNT])
        if len(self.files) > self._MAX_REPR_ARRAY_COUNT:
            array_names += "..."
        
        # 返回对象的字符串表示形式，包括文件名和数组名称列表
        return f"NpzFile {filename!r} with keys: {array_names}"

    # 解决 Mapping 方法中文档字符串的问题
    # 这些方法包含 `->` 符号，可能会导致 sphinx-docs 对类型注释的解释出现问题。详见 gh-25964
    def get(self, key, default=None, /):
        """
        返回键为 k 的值，如果 k 存在于 D 中；否则返回 d，默认为 None。
        """
        return Mapping.get(self, key, default)

    def items(self):
        """
        返回提供视图的类似集合的对象，该视图显示 D 的项
        """
        return Mapping.items(self)

    def keys(self):
        """
        返回提供视图的类似集合的对象，该视图显示 D 的键
        """
        return Mapping.keys(self)

    def values(self):
        """
        返回提供视图的类似集合的对象，该视图显示 D 的值
        """
        return Mapping.values(self)
# 设置函数装饰器，将模块名设置为'numpy'
@set_module('numpy')
# 定义函数load，用于从.npy、.npz或pickle文件中加载数组或对象
def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True,
         encoding='ASCII', *, max_header_size=format._MAX_HEADER_SIZE):
    """
    Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    .. warning:: Loading files that contain object arrays uses the ``pickle``
                 module, which is not secure against erroneous or maliciously
                 constructed data. Consider passing ``allow_pickle=False`` to
                 load data that is known not to contain object arrays for the
                 safer handling of untrusted sources.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the
        ``seek()`` and ``read()`` methods and must always
        be opened in binary mode.  Pickled files require that the
        file-like object support the ``readline()`` method as well.
    mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, then memory-map the file, using the given mode (see
        `numpy.memmap` for a detailed description of the modes).  A
        memory-mapped array is kept on disk. However, it can be accessed
        and sliced like any ndarray.  Memory mapping is especially useful
        for accessing small fragments of large files without reading the
        entire file into memory.
    allow_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files. Reasons for
        disallowing pickles include security, as loading pickled data can
        execute arbitrary code. If pickles are disallowed, loading object
        arrays will fail. Default: False

        .. versionchanged:: 1.16.3
            Made default False in response to CVE-2019-6446.

    fix_imports : bool, optional
        Only useful when loading Python 2 generated pickled files on Python 3,
        which includes npy/npz files containing object arrays. If `fix_imports`
        is True, pickle will try to map the old Python 2 names to the new names
        used in Python 3.
    encoding : str, optional
        What encoding to use when reading Python 2 strings. Only useful when
        loading Python 2 generated pickled files in Python 3, which includes
        npy/npz files containing object arrays. Values other than 'latin1',
        'ASCII', and 'bytes' are not allowed, as they can corrupt numerical
        data. Default: 'ASCII'
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.
        This option is ignored when `allow_pickle` is passed.  In that case
        the file is by definition trusted and the limit is unnecessary.

    Returns
    -------
    """
    result : array, tuple, dict, etc.
        存储在文件中的数据。对于 `.npz` 文件，返回的 NpzFile 类的实例必须关闭，以避免泄漏文件描述符。

    Raises
    ------
    OSError
        如果输入文件不存在或无法读取。
    UnpicklingError
        如果 `allow_pickle=True`，但文件无法作为 pickle 加载。
    ValueError
        文件包含对象数组，但给定了 `allow_pickle=False`。
    EOFError
        在同一文件句柄上多次调用 `np.load` 时，如果已经读取了所有数据。

    See Also
    --------
    save, savez, savez_compressed, loadtxt
    memmap : 创建一个内存映射到存储在磁盘上的数组。
    lib.format.open_memmap : 创建或加载一个内存映射的 `.npy` 文件。

    Notes
    -----
    - 如果文件包含 pickle 数据，则返回存储在 pickle 中的对象。
    - 如果文件是 `.npy` 文件，则返回单个数组。
    - 如果文件是 `.npz` 文件，则返回类似字典的对象，包含 `{filename: array}` 键值对，每个键值对对应存档中的一个文件。
    - 如果文件是 `.npz` 文件，则返回的值支持上下文管理器协议，类似于 open 函数的用法::

        with load('foo.npz') as data:
            a = data['a']

      当退出 'with' 块时关闭底层文件描述符。

    Examples
    --------
    将数据存储到磁盘并再次加载：

    >>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
    >>> np.load('/tmp/123.npy')
    array([[1, 2, 3],
           [4, 5, 6]])

    将压缩数据存储到磁盘并再次加载：

    >>> a=np.array([[1, 2, 3], [4, 5, 6]])
    >>> b=np.array([1, 2])
    >>> np.savez('/tmp/123.npz', a=a, b=b)
    >>> data = np.load('/tmp/123.npz')
    >>> data['a']
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> data['b']
    array([1, 2])
    >>> data.close()

    内存映射存储的数组，然后直接从磁盘访问第二行：

    >>> X = np.load('/tmp/123.npy', mmap_mode='r')
    >>> X[1, :]
    memmap([4, 5, 6])
    # 检查给定的编码是否在支持的列表中，如果不在则抛出错误
    if encoding not in ('ASCII', 'latin1', 'bytes'):
        # 对于 pickle 的 'encoding' 值也影响到 NumPy 数组序列化的二进制数据的加载编码。
        # Pickle 不会将编码信息传递给 NumPy。NumPy 的反序列化代码在预期应该是二进制的位置上，
        # 假定 unicode 数据是 'latin1' 编码。'bytes' 也是安全的，就像 'ASCII' 一样。
        #
        # 其它编码值可能会破坏二进制数据，因此我们有意禁止它们。出于同样的原因，不公开 'errors='
        # 参数，因为除了 'strict' 结果外，其它值也可能会悄悄地破坏数值数据。
        raise ValueError("encoding must be 'ASCII', 'latin1', or 'bytes'")

    # 准备用于 pickle 的关键字参数字典
    pickle_kwargs = dict(encoding=encoding, fix_imports=fix_imports)

    # 使用 ExitStack 来管理资源，确保文件句柄被适当关闭
    with contextlib.ExitStack() as stack:
        # 如果 file 对象具有 'read' 属性，则直接使用它作为文件句柄
        if hasattr(file, 'read'):
            fid = file
            own_fid = False
        else:
            # 否则，打开文件并作为文件句柄使用，确保在离开作用域时关闭
            fid = stack.enter_context(open(os.fspath(file), "rb"))
            own_fid = True

        # 用于识别 NumPy 二进制文件和 pickle 的代码
        _ZIP_PREFIX = b'PK\x03\x04'
        _ZIP_SUFFIX = b'PK\x05\x06'  # 空 zip 文件以此开头
        N = len(format.MAGIC_PREFIX)
        # 读取文件开头的 N 个字节作为 magic 数据
        magic = fid.read(N)
        if not magic:
            raise EOFError("No data left in file")
        # 如果文件大小小于 N，则确保不会超出文件开头的位置
        fid.seek(-min(N, len(magic)), 1)  # 回退

        # 判断文件类型，是否为 zip 文件
        if magic.startswith(_ZIP_PREFIX) or magic.startswith(_ZIP_SUFFIX):
            # zip 文件（假设是 .npz）
            # 可能将文件所有权转移给 NpzFile
            stack.pop_all()
            ret = NpzFile(fid, own_fid=own_fid, allow_pickle=allow_pickle,
                          pickle_kwargs=pickle_kwargs,
                          max_header_size=max_header_size)
            return ret
        elif magic == format.MAGIC_PREFIX:
            # .npy 文件
            if mmap_mode:
                if allow_pickle:
                    max_header_size = 2**64
                return format.open_memmap(file, mode=mmap_mode,
                                          max_header_size=max_header_size)
            else:
                return format.read_array(fid, allow_pickle=allow_pickle,
                                         pickle_kwargs=pickle_kwargs,
                                         max_header_size=max_header_size)
        else:
            # 尝试解析为 pickle 格式
            if not allow_pickle:
                raise ValueError("Cannot load file containing pickled data "
                                 "when allow_pickle=False")
            try:
                return pickle.load(fid, **pickle_kwargs)
            except Exception as e:
                raise pickle.UnpicklingError(
                    f"Failed to interpret file {file!r} as a pickle") from e
# 定义一个保存函数，根据数组和文件路径或对象，将数组以 NumPy 的 .npy 格式保存到文件中
@array_function_dispatch(_save_dispatcher)
def save(file, arr, allow_pickle=True, fix_imports=np._NoValue):
    """
    Save an array to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        File or filename to which the data is saved. If file is a file-object,
        then the filename is unchanged.  If file is a string or Path,
        a ``.npy`` extension will be appended to the filename if it does not
        already have one.
    arr : array_like
        Array data to be saved.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for
        disallowing pickles include security (loading pickled data can execute
        arbitrary code) and portability (pickled objects may not be loadable
        on different Python installations, for example if the stored objects
        require libraries that are not available, and not all pickled data is
        compatible between different versions of Python).
        Default: True
    fix_imports : bool, optional
        The `fix_imports` flag is deprecated and has no effect.

        .. deprecated:: 2.1
            This flag is ignored since NumPy 1.17 and was only needed to
            support loading some files in Python 2 written in Python 3.

    See Also
    --------
    savez : Save several arrays into a ``.npz`` archive
    savetxt, load

    Notes
    -----
    For a description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.

    Any data saved to the file is appended to the end of the file.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = np.arange(10)
    >>> np.save(outfile, x)

    >>> _ = outfile.seek(0) # Only needed to simulate closing & reopening file
    >>> np.load(outfile)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


    >>> with open('test.npy', 'wb') as f:
    ...     np.save(f, np.array([1, 2]))
    ...     np.save(f, np.array([1, 3]))
    >>> with open('test.npy', 'rb') as f:
    ...     a = np.load(f)
    ...     b = np.load(f)
    >>> print(a, b)
    # [1 2] [1 3]
    """
    
    # 如果 fix_imports 不是默认值 np._NoValue，则发出警告提示，该标志已弃用
    if fix_imports is not np._NoValue:
        warnings.warn(
            "The 'fix_imports' flag is deprecated and has no effect. "
            "(Deprecated in NumPy 2.1)",
            DeprecationWarning, stacklevel=2)
    
    # 根据 file 是否具有 'write' 属性来确定文件上下文
    if hasattr(file, 'write'):
        file_ctx = contextlib.nullcontext(file)
    else:
        # 将 file 转换为文件路径字符串，确保文件名以 .npy 结尾
        file = os.fspath(file)
        if not file.endswith('.npy'):
            file = file + '.npy'
        # 打开文件以写入二进制模式
        file_ctx = open(file, "wb")

    # 使用文件上下文来打开文件，并将 arr 转换为 NumPy 数组，调用 format.write_array 写入数据
    with file_ctx as fid:
        arr = np.asanyarray(arr)
        format.write_array(fid, arr, allow_pickle=allow_pickle,
                           pickle_kwargs=dict(fix_imports=fix_imports))


# 定义一个保存函数的分发器，将文件和其他参数传递给 save 函数
def _save_dispatcher(file, arr, allow_pickle=None, fix_imports=None):
    return (arr,)


# 定义一个保存函数的分发器，将文件和其他参数传递给 save 函数，并返回除文件外的所有参数
def _savez_dispatcher(file, *args, **kwds):
    yield from args
    # 使用生成器语法，逐个产出关键字参数字典 `kwds` 中的值
    yield from kwds.values()
@array_function_dispatch(_savez_dispatcher)
# 使用装饰器进行函数分派，确保正确调用与保存相关的函数

def savez(file, *args, **kwds):
    """Save several arrays into a single file in uncompressed ``.npz`` format.

    Provide arrays as keyword arguments to store them under the
    corresponding name in the output file: ``savez(fn, x=x, y=y)``.

    If arrays are specified as positional arguments, i.e., ``savez(fn,
    x, y)``, their names will be `arr_0`, `arr_1`, etc.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        Either the filename (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the filename if it is not
        already there.
    args : Arguments, optional
        Arrays to save to the file. Please use keyword arguments (see
        `kwds` below) to assign names to arrays.  Arrays specified as
        args will be named "arr_0", "arr_1", and so on.
    kwds : Keyword arguments, optional
        Arrays to save to the file. Each array will be saved to the
        output file with its corresponding keyword name.

    Returns
    -------
    None

    See Also
    --------
    save : Save a single array to a binary file in NumPy format.
    savetxt : Save an array to a file as plain text.
    savez_compressed : Save several arrays into a compressed ``.npz`` archive

    Notes
    -----
    The ``.npz`` file format is a zipped archive of files named after the
    variables they contain.  The archive is not compressed and each file
    in the archive contains one variable in ``.npy`` format. For a
    description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.

    When opening the saved ``.npz`` file with `load` a `~lib.npyio.NpzFile`
    object is returned. This is a dictionary-like object which can be queried
    for its list of arrays (with the ``.files`` attribute), and for the arrays
    themselves.

    Keys passed in `kwds` are used as filenames inside the ZIP archive.
    Therefore, keys should be valid filenames; e.g., avoid keys that begin with
    ``/`` or contain ``.``.

    When naming variables with keyword arguments, it is not possible to name a
    variable ``file``, as this would cause the ``file`` argument to be defined
    twice in the call to ``savez``.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)

    Using `savez` with \\*args, the arrays are saved with default names.

    >>> np.savez(outfile, x, y)
    >>> _ = outfile.seek(0) # Only needed to simulate closing & reopening file
    >>> npzfile = np.load(outfile)
    >>> npzfile.files
    ['arr_0', 'arr_1']
    >>> npzfile['arr_0']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Using `savez` with \\**kwds, the arrays are saved with the keyword names.

    >>> outfile = TemporaryFile()
    >>> np.savez(outfile, x=x, y=y)
    >>> _ = outfile.seek(0)
    """
    # 在给定文件中保存多个数组，使用未压缩的 ``.npz`` 格式

    # 如果未指定文件扩展名为 ``.npz``，则自动添加扩展名
    # 将输入参数中的数组以对应的关键字名保存到输出文件中
    # 当使用位置参数指定数组时，它们将被命名为 "arr_0", "arr_1" 等
    # 参数 `kwds` 中的关键字作为文件名保存在 ZIP 归档中
    # 注意避免使用包含特殊字符如 `/` 或 `.` 开始的键名
    # 在使用关键字参数命名变量时，不能使用变量名 `file`，否则会导致 `file` 参数在调用 `savez` 时被重定义
    pass
    # 使用 numpy 的 load 函数加载指定的 NPZ 文件并返回一个 npzfile 对象
    >>> npzfile = np.load(outfile)
    # 对 npzfile 对象的文件列表进行排序并返回
    >>> sorted(npzfile.files)
    # 访问 npzfile 对象中键为 'x' 的数组数据
    >>> npzfile['x']
    # 输出数组 'x' 中的内容
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    # 调用 _savez 函数，将参数 file、args、kwds 和 False 传递给它
    _savez(file, args, kwds, False)
# 导入必要的 zipfile 模块，用于处理压缩文件
import zipfile

# 如果传入的文件参数没有 'write' 方法，则将其转换为文件路径字符串
if not hasattr(file, 'write'):
    file = os.fspath(file)
    # 如果文件路径不以 '.npz' 结尾，则添加 '.npz' 后缀
    if not file.endswith('.npz'):
        file = file + '.npz'

# 将关键字参数（即传入的数组）存储到 namedict 字典中
namedict = kwds
    # 遍历参数列表并为每个参数创建一个键，格式为'arr_i'，其中i是参数的索引
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        # 检查当前键是否已存在于命名字典中，如果存在则抛出数值错误异常
        if key in namedict.keys():
            raise ValueError(
                "Cannot use un-named variables and keyword %s" % key)
        # 将当前键值对添加到命名字典中
        namedict[key] = val

    # 根据压缩选项确定压缩类型：如果压缩为True，则使用ZIP_DEFLATED进行压缩，否则使用ZIP_STORED
    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    # 使用工厂函数创建一个zip文件对象，指定文件名和打开模式为写入，同时指定压缩类型
    zipf = zipfile_factory(file, mode="w", compression=compression)

    # 遍历命名字典中的每个键值对
    for key, val in namedict.items():
        # 为当前值生成文件名，形式为'arr_i.npy'
        fname = key + '.npy'
        # 将值转换为NumPy数组（如果尚未是数组的话）
        val = np.asanyarray(val)
        # 使用zip文件对象的open方法打开当前文件名对应的文件，以写入模式，强制使用ZIP64格式
        with zipf.open(fname, 'w', force_zip64=True) as fid:
            # 使用格式化对象的write_array方法将数组写入到文件中
            format.write_array(fid, val,
                               allow_pickle=allow_pickle,
                               pickle_kwargs=pickle_kwargs)

    # 关闭zip文件对象，完成写入操作
    zipf.close()
# 确保参数 ndmin 在 _ensure_ndmin_ndarray 函数中被支持
def _ensure_ndmin_ndarray_check_param(ndmin):
    """Just checks if the param ndmin is supported on
        _ensure_ndmin_ndarray. It is intended to be used as
        verification before running anything expensive.
        e.g. loadtxt, genfromtxt
    """
    # 检查 ndmin 参数的合法性
    if ndmin not in [0, 1, 2]:
        raise ValueError(f"Illegal value of ndmin keyword: {ndmin}")


def _ensure_ndmin_ndarray(a, *, ndmin: int):
    """This is a helper function of loadtxt and genfromtxt to ensure
        proper minimum dimension as requested

        ndim : int. Supported values 1, 2, 3
                    ^^ whenever this changes, keep in sync with
                       _ensure_ndmin_ndarray_check_param
    """
    # 验证数组至少具有 `ndmin` 指定的维度
    # 调整数组的大小和形状，去除多余的维度
    if a.ndim > ndmin:
        a = np.squeeze(a)
    # 确保数组至少具有指定的最小维度
    # - 为了奇怪的情况，如 ndmin=1 且 a.squeeze().ndim=0
    if a.ndim < ndmin:
        if ndmin == 1:
            a = np.atleast_1d(a)
        elif ndmin == 2:
            a = np.atleast_2d(a).T

    return a


# loadtxt 一次读取的行数，可以为了测试目的进行重写
_loadtxt_chunksize = 50000


def _check_nonneg_int(value, name="argument"):
    try:
        operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _preprocess_comments(iterable, comments, encoding):
    """
    Generator that consumes a line iterated iterable and strips out the
    multiple (or multi-character) comments from lines.
    This is a pre-processing step to achieve feature parity with loadtxt
    (we assume that this feature is a nieche feature).
    """
    for line in iterable:
        if isinstance(line, bytes):
            # 需要在此处处理转换，否则分割将会失败
            line = line.decode(encoding)

        for c in comments:
            line = line.split(c, 1)[0]

        yield line


# 遇到参数化数据类型时一次读取的行数
_loadtxt_chunksize = 50000


def _read(fname, *, delimiter=',', comment='#', quote='"',
          imaginary_unit='j', usecols=None, skiplines=0,
          max_rows=None, converters=None, ndmin=None, unpack=False,
          dtype=np.float64, encoding=None):
    r"""
    从文本文件中读取一个 NumPy 数组。
    这是 loadtxt 的辅助函数。

    Parameters
    ----------
    fname : file, str, or pathlib.Path
        要读取的文件名或文件路径。
    delimiter : str, optional
        文件中字段的分隔符。
        默认为逗号 ','。如果为 None，则任何空白序列都视为分隔符。
    # comment 参数：用于指定注释的起始字符或字符串序列，从该字符到行尾的文本将被忽略。
    # 可选参数，可以使用 None 禁用所有注释功能。
    comment : str or sequence of str or None, optional

    # quote 参数：用于引用字符串字段的字符。默认为双引号（'"'）。
    # 可选参数，使用 None 禁用引号支持。
    quote : str or None, optional

    # imaginary_unit 参数：表示虚数单位 `sqrt(-1)` 的字符。默认为 'j'。
    # 可选参数，表示一个字符串。
    imaginary_unit : str, optional

    # usecols 参数：一个一维整数数组，指定要包含在数组中的文件列号。
    # 如果未提供此值，则使用文件中的所有列。
    # 可选参数，表示一个数组。
    usecols : array_like, optional

    # skiplines 参数：在解释文件中的数据之前要跳过的行数。
    # 可选参数，表示一个整数。
    skiplines : int, optional

    # max_rows 参数：要读取的数据的最大行数。默认为读取整个文件。
    # 可选参数，表示一个整数。
    max_rows : int, optional

    # converters 参数：一个字典或可调用对象，用于将所有列的字符串解析为所需的值。
    # 或者，将列号映射到解析函数的字典。
    # 默认为 None。例如，``converters = {0: datestr2num}`` 将日期字符串转换为数字。
    # 可选参数，表示一个字典或函数。
    converters : dict or callable, optional

    # ndmin 参数：返回数组的最小维数。允许的值为 0、1 或 2。默认为 0。
    # 可选参数，表示一个整数。
    ndmin : int, optional

    # unpack 参数：如果为 True，则返回的数组会被转置，允许使用 ``x, y, z = read(...)`` 解包。
    # 当与结构化数据类型一起使用时，为每个字段返回数组。默认为 False。
    # 可选参数，表示一个布尔值。
    unpack : bool, optional

    # dtype 参数：一个 NumPy dtype 实例，可以是结构化的 dtype，用于映射到文件的列。
    # 可选参数，表示一个 NumPy 数据类型。
    dtype : numpy data type

    # encoding 参数：用于解码输入文件的编码。特殊值 'bytes'（默认）启用与 `converters` 的向后兼容行为，
    # 确保输入到转换函数的是编码字节对象。
    # 如果 encoding 是 ``'bytes'`` 或 ``None``，则使用默认系统编码。
    # 可选参数，表示一个字符串。
    encoding : str, optional

    # 返回值：一个 NumPy 数组。
    # 返回的数组根据参数的配置和文件内容而定。
    Returns
    -------
    ndarray
        NumPy array.
    """
    # 处理特殊的 'bytes' 编码关键字
    byte_converters = False
    if encoding == 'bytes':
        encoding = None
        byte_converters = True

    # 如果未提供 dtype 参数，则抛出类型错误异常
    if dtype is None:
        raise TypeError("a dtype must be provided.")
    
    # 将 dtype 转换为 NumPy 的数据类型实例
    dtype = np.dtype(dtype)

    # 用于对象块间通过 dtype 读取
    read_dtype_via_object_chunks = None
    # 如果 dtype 的类型是 'SUM' 并且满足以下条件之一：
    # dtype 等于 "S0"、"U0"、"M8" 或 'm8'，则表示这是一个旧的“灵活”dtype。
    # 目前核心代码中不真正支持参数化的 dtype（核心中没有 dtype 探测步骤），
    # 但为了向后兼容性，我们必须支持这些类型。
    if dtype.kind in 'SUM' and (
            dtype == "S0" or dtype == "U0" or dtype == "M8" or dtype == 'm8'):
        # 将 dtype 保存到 read_dtype_via_object_chunks 变量中
        read_dtype_via_object_chunks = dtype
        # 将 dtype 设置为通用的 object 类型
        dtype = np.dtype(object)

    # 如果 usecols 不为 None，则进行以下处理：
    if usecols is not None:
        # 尝试将 usecols 转换为整数列表，如果无法转换，则将其视为单个整数
        try:
            usecols = list(usecols)
        except TypeError:
            usecols = [usecols]

    # 检查和确保 ndmin 的值符合数组的最小维度要求
    _ensure_ndmin_ndarray_check_param(ndmin)

    # 如果 comment 为 None，则将 comments 设置为 None；否则进行如下处理：
    if comment is None:
        comments = None
    else:
        # 假设 comments 是一个字符串序列，如果其中包含空字符串，则抛出异常
        if "" in comment:
            raise ValueError(
                "comments cannot be an empty string. Use comments=None to "
                "disable comments."
            )
        # 将 comments 转换为元组类型
        comments = tuple(comment)
        # 将 comment 设置为 None，表示不使用单个字符作为 comment
        comment = None
        # 如果 comments 长度为 0，则将其设置为 None，表示不需要注释
        if len(comments) == 0:
            comments = None  # 没有任何注释
        # 如果 comments 长度为 1，则进一步判断：
        elif len(comments) == 1:
            # 如果只有一个注释，并且该注释只有一个字符，则正常解析即可处理
            if isinstance(comments[0], str) and len(comments[0]) == 1:
                comment = comments[0]
                comments = None
        else:
            # 如果有多个注释字符，则进行输入验证
            if delimiter in comments:
                raise TypeError(
                    f"Comment characters '{comments}' cannot include the "
                    f"delimiter '{delimiter}'"
                )

    # 现在 comment 可能是一个 1 或 0 个字符的字符串，或者是一个元组
    if comments is not None:
        # 注意：早期版本支持两个字符的注释（并且可以扩展到多个字符），我们假设这种情况不常见，因此不做优化处理。
        if quote is not None:
            raise ValueError(
                "when multiple comments or a multi-character comment is "
                "given, quotes are not supported.  In this case quotechar "
                "must be set to None.")

    # 检查虚数单位 imaginary_unit 的长度是否为 1
    if len(imaginary_unit) != 1:
        raise ValueError('len(imaginary_unit) must be 1.')

    # 检查 skiplines 是否为非负整数
    _check_nonneg_int(skiplines)
    # 如果 max_rows 不为 None，则检查其是否为非负整数；否则将 max_rows 设置为 -1，表示读取整个文件。
    if max_rows is not None:
        _check_nonneg_int(max_rows)
    else:
        max_rows = -1  # 将 -1 传递给 C 代码表示“读取整个文件”。

    # 创建一个空的上下文管理器作为文件句柄的关闭上下文，默认 filelike 为 False
    fh_closing_ctx = contextlib.nullcontext()
    filelike = False
    try:
        # 如果 fname 是 os.PathLike 类型，将其转换为字符串路径
        if isinstance(fname, os.PathLike):
            fname = os.fspath(fname)
        
        # 如果 fname 是字符串类型
        if isinstance(fname, str):
            # 使用 numpy 的 _datasource 模块打开文件以供读取，并指定以文本模式打开
            fh = np.lib._datasource.open(fname, 'rt', encoding=encoding)
            
            # 如果 encoding 参数为 None，则尝试从文件句柄中获取编码，否则使用 'latin1' 编码
            if encoding is None:
                encoding = getattr(fh, 'encoding', 'latin1')
            
            # 使用 contextlib.closing 包装文件句柄，以便在退出上下文时自动关闭
            fh_closing_ctx = contextlib.closing(fh)
            
            # 将数据指向文件句柄
            data = fh
            
            # 标记文件类型为类文件对象
            filelike = True
        else:
            # 如果 fname 不是字符串类型
            
            # 如果 encoding 参数为 None，则尝试从 fname 对象中获取编码，否则使用 'latin1' 编码
            if encoding is None:
                encoding = getattr(fname, 'encoding', 'latin1')
            
            # 将数据指向 fname 对象的迭代器
            data = iter(fname)
    
    # 捕获 TypeError 异常
    except TypeError as e:
        # 抛出 ValueError 异常，说明传入的 fname 参数类型不正确
        raise ValueError(
            f"fname must be a string, filehandle, list of strings,\n"
            f"or generator. Got {type(fname)} instead.") from e
    with fh_closing_ctx:
        # 使用上下文管理器 fh_closing_ctx 进行文件句柄的安全关闭操作

        if comments is not None:
            # 如果 comments 参数不为 None，则需要预处理数据中的注释信息
            if filelike:
                # 如果 filelike 标志为 True，则将 data 转换为迭代器
                data = iter(data)
                filelike = False

            # 对数据进行预处理，处理其中的注释信息和编码
            data = _preprocess_comments(data, comments, encoding)

        if read_dtype_via_object_chunks is None:
            # 如果 read_dtype_via_object_chunks 为 None，则直接从文件对象中加载数据到 arr
            arr = _load_from_filelike(
                data, delimiter=delimiter, comment=comment, quote=quote,
                imaginary_unit=imaginary_unit,
                usecols=usecols, skiplines=skiplines, max_rows=max_rows,
                converters=converters, dtype=dtype,
                encoding=encoding, filelike=filelike,
                byte_converters=byte_converters)

        else:
            # 如果 read_dtype_via_object_chunks 不为 None，则使用对象数组的方式读取文件，并转换为指定的 dtype
            # 这种方法确保正确发现字符串长度和日期时间单位（例如 `arr.astype()`）
            # 由于分块处理，某些错误报告可能不太清晰，目前如此。
            if filelike:
                # 如果 filelike 标志为 True，则无法在从文件中读取时使用分块处理
                data = iter(data)

            c_byte_converters = False
            if read_dtype_via_object_chunks == "S":
                # 如果 read_dtype_via_object_chunks 是 "S"，则使用 latin1 编码而不是 ascii
                c_byte_converters = True

            chunks = []
            while max_rows != 0:
                # 循环读取数据直到达到 max_rows 为止
                if max_rows < 0:
                    chunk_size = _loadtxt_chunksize
                else:
                    chunk_size = min(_loadtxt_chunksize, max_rows)

                # 从文件对象中加载数据块并转换为指定的 dtype
                next_arr = _load_from_filelike(
                    data, delimiter=delimiter, comment=comment, quote=quote,
                    imaginary_unit=imaginary_unit,
                    usecols=usecols, skiplines=skiplines, max_rows=max_rows,
                    converters=converters, dtype=dtype,
                    encoding=encoding, filelike=filelike,
                    byte_converters=byte_converters,
                    c_byte_converters=c_byte_converters)

                # 在此处进行类型转换。我们希望这样做对于大文件更好，因为存储更紧凑。
                # 可以适应（原则上连接可以进行类型转换）。
                chunks.append(next_arr.astype(read_dtype_via_object_chunks))

                skiprows = 0  # 只需对第一个块进行跳过行数操作
                if max_rows >= 0:
                    max_rows -= chunk_size
                if len(next_arr) < chunk_size:
                    # 请求的数据量少于块大小，则表示已经完成读取。
                    break

            # 至少需要一个块，但如果为空，则最后一个块可能有错误的形状。
            if len(chunks) > 1 and len(chunks[-1]) == 0:
                del chunks[-1]
            if len(chunks) == 1:
                arr = chunks[0]
            else:
                # 将所有块连接成一个数组
                arr = np.concatenate(chunks, axis=0)

    # 注意：对于结构化的 dtype，ndmin 的功能如广告所述，但通常情况下...
    # 确保数组至少具有指定的最小维数，同时保持数组的原始结构维度。
    # 如果数组本身是一维的，则 ndmin=2 将添加一个额外的维度，即使没有进行挤压操作。
    # 在某些情况下，使用 `squeeze=False` 可能是一个更好的解决方案（pandas 使用挤压操作）。
    arr = _ensure_ndmin_ndarray(arr, ndmin=ndmin)

    # 检查数组的形状是否非空
    if arr.shape:
        # 如果数组的第一个维度长度为 0，则发出警告，指出输入数据不包含任何内容。
        warnings.warn(
            f'loadtxt: input contained no data: "{fname}"',
            category=UserWarning,
            stacklevel=3
        )

    # 如果设置了 unpack 参数
    if unpack:
        # 获取数组的数据类型
        dt = arr.dtype
        # 如果数据类型具有字段名（即结构化数组）
        if dt.names is not None:
            # 对于结构化数组，返回每个字段的数组
            return [arr[field] for field in dt.names]
        else:
            # 对于非结构化数组，返回转置后的数组
            return arr.T
    else:
        # 如果未设置 unpack 参数，直接返回原始数组
        return arr
# 设置装饰器，使该函数的文档类似于数组函数的文档
# 设置函数的模块名称为 'numpy'
@set_array_function_like_doc
@set_module('numpy')
# 定义函数 loadtxt，用于从文本文件加载数据
def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding=None, max_rows=None, *, quotechar=None,
            like=None):
    r"""
    从文本文件加载数据。

    Parameters
    ----------
    fname : file, str, pathlib.Path, list of str, generator
        要读取的文件、文件名、列表或生成器。如果文件名的扩展名为 ``.gz`` 或 ``.bz2``，则首先解压文件。
        注意，生成器必须返回字节或字符串。列表中的字符串或生成器生成的字符串将被视为行。
    dtype : data-type, optional
        结果数组的数据类型；默认为 float。如果这是结构化数据类型，则结果数组将是一维的，并且每行将被解释为数组的一个元素。
        在这种情况下，使用的列数必须与数据类型中的字段数相匹配。
    comments : str or sequence of str or None, optional
        用于指示注释开始的字符或字符列表。None 表示没有注释。为了向后兼容，字节字符串将解码为 'latin1'。默认为 '#'.
    delimiter : str, optional
        用于分隔值的字符。为了向后兼容，字节字符串将解码为 'latin1'。默认为空白字符。

        .. versionchanged:: 1.23.0
           仅支持单个字符分隔符。不能使用换行符作为分隔符。

    converters : dict or callable, optional
        自定义值解析的转换器函数。如果 `converters` 是可调用的，则该函数将应用于所有列；否则，它必须是将列号映射到解析器函数的字典。
        更多详细信息请参见示例。
        默认值为 None。

        .. versionchanged:: 1.23.0
           添加了将单个可调用函数传递给所有列的功能。

    skiprows : int, optional
        跳过前 `skiprows` 行，包括注释；默认为 0。
    usecols : int or sequence, optional
        要读取的列，从 0 开始计数。例如，``usecols = (1,4,5)`` 将提取第 2、第 5 和第 6 列。
        默认为 None，表示读取所有列。

        .. versionchanged:: 1.11.0
            当需要读取单个列时，可以使用整数而不是元组。例如，``usecols = 3`` 读取第四列，与 ``usecols = (3,)`` 的效果相同。

    unpack : bool, optional
        如果为 True，则返回的数组进行转置，以便可以使用 ``x, y, z = loadtxt(...)`` 进行拆包。
        当与结构化数据类型一起使用时，为每个字段返回数组。
        默认为 False。
    ndmin : int, optional
        # 返回的数组至少有 `ndmin` 维度。否则，将压缩单维轴。
        Legal values: 0 (default), 1 or 2.
        
        .. versionadded:: 1.6.0
    encoding : str, optional
        # 用于解码输入文件的编码。不适用于输入流。
        特殊值 'bytes' 启用向后兼容的工作，确保尽可能以字节数组形式接收结果，并将 'latin1' 编码的字符串传递给转换器。
        覆盖此值以接收 Unicode 数组，并将字符串作为输入传递给转换器。如果设置为 None，则使用系统默认值。默认值为 'bytes'。
        
        .. versionadded:: 1.14.0
        .. versionchanged:: 2.0
            在 NumPy 2 之前，默认为 ``'bytes'``，用于 Python 2 兼容性。现在默认为 ``None``。
    max_rows : int, optional
        # 在跳过 `skiprows` 行后，读取 `max_rows` 行内容。默认是读取所有行。
        注意，不计入 `max_rows` 的空行（如空行和注释行），而这些行在 `skiprows` 中计数。

        .. versionadded:: 1.16.0

        .. versionchanged:: 1.23.0
            不计入不包含数据的行，包括注释行（例如以 '#' 开头或通过 `comments` 指定的行），以计算 `max_rows`。
    quotechar : unicode character or None, optional
        # 用于表示引用项的起始和结束的字符。
        在引用项内部忽略定界符或注释字符的出现。默认值为 ``quotechar=None``，表示禁用引用支持。

        如果在引用字段内找到两个连续的 `quotechar` 实例，则第一个将被视为转义字符。请参见示例。

        .. versionadded:: 1.23.0
    ${ARRAY_FUNCTION_LIKE}
        # 与数组函数类似的功能。

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        # 从文本文件读取的数据。

    See Also
    --------
    load, fromstring, fromregex
    genfromtxt : 以指定方式处理缺失值的加载数据。
    scipy.io.loadmat : 读取 MATLAB 数据文件

    Notes
    -----
    # 此函数旨在快速读取格式简单的文件。
    `genfromtxt` 函数提供更复杂的处理，例如具有缺失值的行。

    输入文本文件中的每行必须具有相同数量的值，才能读取所有值。
    如果所有行的值数量不同，则可以通过 `usecols` 指定列来读取最多 n 列（其中 n 是所有行中最少的值的数量）。

    .. versionadded:: 1.10.0

    Python 的 float.hex 方法生成的字符串可用作浮点数的输入。

    Examples
    --------
    # 示例
    >>> from io import StringIO   # 导入StringIO模块，用于创建类似文件对象的字符串IO
    >>> c = StringIO("0 1\n2 3")
    >>> np.loadtxt(c)
    array([[0., 1.],             # 从StringIO对象c中加载数据，生成一个2x2的数组
           [2., 3.]])
    
    >>> d = StringIO("M 21 72\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([(b'M', 21, 72.), (b'F', 35, 58.)],    # 从StringIO对象d中加载数据，生成一个结构化数组，指定字段名和数据类型
          dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])
    
    >>> c = StringIO("1,0,2\n3,0,4")
    >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
    >>> x
    array([1., 3.])             # 从StringIO对象c中加载数据，仅使用第0列和第2列数据，分别赋给x和y
    >>> y
    array([2., 4.])
    
    The `converters` argument is used to specify functions to preprocess the
    text prior to parsing. `converters` can be a dictionary that maps
    preprocessing functions to each column:
    
    >>> s = StringIO("1.618, 2.296\n3.141, 4.669\n")
    >>> conv = {
    ...     0: lambda x: np.floor(float(x)),  # 定义第0列数据的转换函数，向下取整
    ...     1: lambda x: np.ceil(float(x)),   # 定义第1列数据的转换函数，向上取整
    ... }
    >>> np.loadtxt(s, delimiter=",", converters=conv)
    array([[1., 3.],             # 从StringIO对象s中加载数据，并应用转换函数conv对每列进行预处理
           [3., 5.]])
    
    `converters` can be a callable instead of a dictionary, in which case it
    is applied to all columns:
    
    >>> s = StringIO("0xDE 0xAD\n0xC0 0xDE")
    >>> import functools
    >>> conv = functools.partial(int, base=16)
    >>> np.loadtxt(s, converters=conv)
    array([[222., 173.],         # 从StringIO对象s中加载数据，使用int(base=16)转换器对所有数据进行16进制转换
           [192., 222.]])
    
    This example shows how `converters` can be used to convert a field
    with a trailing minus sign into a negative number.
    
    >>> s = StringIO("10.01 31.25-\n19.22 64.31\n17.57- 63.94")
    >>> def conv(fld):
    ...     return -float(fld[:-1]) if fld.endswith("-") else float(fld)
    ...
    >>> np.loadtxt(s, converters=conv)
    array([[ 10.01, -31.25],     # 从StringIO对象s中加载数据，使用自定义转换器conv处理以减号结尾的字段
           [ 19.22,  64.31],
           [-17.57,  63.94]])
    
    Using a callable as the converter can be particularly useful for handling
    values with different formatting, e.g. floats with underscores:
    
    >>> s = StringIO("1 2.7 100_000")
    >>> np.loadtxt(s, converters=float)
    array([1.e+00, 2.7e+00, 1.e+05])    # 从StringIO对象s中加载数据，直接使用float转换器处理所有数据
    
    This idea can be extended to automatically handle values specified in
    many different formats, such as hex values:
    
    >>> def conv(val):
    ...     try:
    ...         return float(val)
    ...     except ValueError:
    ...         return float.fromhex(val)
    >>> s = StringIO("1, 2.5, 3_000, 0b4, 0x1.4000000000000p+2")
    >>> np.loadtxt(s, delimiter=",", converters=conv)
    array([1.0e+00, 2.5e+00, 3.0e+03, 1.8e+02, 5.0e+00])   # 从StringIO对象s中加载数据，并根据转换器conv处理不同格式的数据
    
    Or a format where the ``-`` sign comes after the number:
    
    >>> s = StringIO("10.01 31.25-\n19.22 64.31\n17.57- 63.94")
    >>> conv = lambda x: -float(x[:-1]) if x.endswith("-") else float(x)
    >>> np.loadtxt(s, converters=conv)
    array([[ 10.01, -31.25],     # 从StringIO对象s中加载数据，使用lambda表达式conv作为转换器处理以减号结尾的字段
           [ 19.22,  64.31],
           [-17.57,  63.94]])
    
    Support for quoted fields is enabled with the `quotechar` parameter.
    # 如果传入了 `like` 参数，则调用 `_loadtxt_with_like` 函数进行加载数据，该函数会根据 `like` 参数的设置来读取数据文件
    if like is not None:
        return _loadtxt_with_like(
            like, fname, dtype=dtype, comments=comments, delimiter=delimiter,
            converters=converters, skiprows=skiprows, usecols=usecols,
            unpack=unpack, ndmin=ndmin, encoding=encoding,
            max_rows=max_rows
        )

    # 如果 `delimiter` 参数是字节类型，则解码为 Latin1 字符串
    if isinstance(delimiter, bytes):
        delimiter.decode("latin1")

    # 如果 `dtype` 参数未指定，则默认使用 np.float64 数据类型
    if dtype is None:
        dtype = np.float64

    # 将 `comments` 参数赋值给 `comment` 变量，用于控制字符类型转换的便利性
    comment = comments
    # 如果 `comment` 参数不为空，则根据其类型进行解码处理，确保统一为字符串列表
    if comment is not None:
        if isinstance(comment, (str, bytes)):
            comment = [comment]
        comment = [
            x.decode('latin1') if isinstance(x, bytes) else x for x in comment]

    # 如果 `delimiter` 参数是字节类型，则解码为 Latin1 字符串
    if isinstance(delimiter, bytes):
        delimiter = delimiter.decode('latin1')

    # 调用 `_read` 函数从文件中读取数据到数组 `arr` 中，传入各种参数进行控制
    arr = _read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
                converters=converters, skiplines=skiprows, usecols=usecols,
                unpack=unpack, ndmin=ndmin, encoding=encoding,
                max_rows=max_rows, quote=quotechar)

    # 返回读取到的数组 `arr`
    return arr
# 将 loadtxt 函数作为 array_function_dispatch 的结果传递给 _loadtxt_with_like 变量
_loadtxt_with_like = array_function_dispatch()(loadtxt)

# 定义 _savetxt_dispatcher 函数，用于分派 savetxt 函数的参数
def _savetxt_dispatcher(fname, X, fmt=None, delimiter=None, newline=None,
                        header=None, footer=None, comments=None,
                        encoding=None):
    # 返回一个包含 X 的元组
    return (X,)

# 使用 array_function_dispatch 装饰器将 _savetxt_dispatcher 函数作为 savetxt 的分派函数
@array_function_dispatch(_savetxt_dispatcher)
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """
    Save an array to a text file.

    Parameters
    ----------
    fname : filename, file handle or pathlib.Path
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    X : 1D or 2D array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs, optional
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored. For complex `X`, the legal options
        for `fmt` are:

        * a single specifier, ``fmt='%.4e'``, resulting in numbers formatted
          like ``' (%s+%sj)' % (fmt, fmt)``
        * a full string specifying every real and imaginary part, e.g.
          ``' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'`` for 3 columns
        * a list of specifiers, one per column - in this case, the real
          and imaginary part must have separate specifiers,
          e.g. ``['%.3e + %.3ej', '(%.15e%+.15ej)']`` for 2 columns
    delimiter : str, optional
        String or character separating columns.
    newline : str, optional
        String or character separating lines.

        .. versionadded:: 1.5.0
    header : str, optional
        String that will be written at the beginning of the file.

        .. versionadded:: 1.7.0
    footer : str, optional
        String that will be written at the end of the file.

        .. versionadded:: 1.7.0
    comments : str, optional
        String that will be prepended to the ``header`` and ``footer`` strings,
        to mark them as comments. Default: '# ',  as expected by e.g.
        ``numpy.loadtxt``.

        .. versionadded:: 1.7.0
    encoding : {None, str}, optional
        Encoding used to encode the outputfile. Does not apply to output
        streams. If the encoding is something other than 'bytes' or 'latin1'
        you will not be able to load the file in NumPy versions < 1.14. Default
        is 'latin1'.

        .. versionadded:: 1.14.0


    See Also
    --------
    save : Save an array to a binary file in NumPy ``.npy`` format
    savez : Save several arrays into an uncompressed ``.npz`` archive
    savez_compressed : Save several arrays into a compressed ``.npz`` archive

    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):
    """
    own_fh = False
    如果 fname 是 os.PathLike 的实例，转换为路径字符串
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    如果 fname 类型类似字符串：
        # 创建一个空文件
        open(fname, 'wt').close()
        使用 np.lib._datasource 打开 fname 文件进行写入操作，指定编码为 encoding
        fh = np.lib._datasource.open(fname, 'wt', encoding=encoding)
        own_fh = True
    否则，如果 fname 具有 'write' 属性：
        # 包装以处理字节输出流
        fh = WriteWrap(fname, encoding or 'latin1')
    else:
        raise ValueError('fname must be a string or file handle')

    try:
        X = np.asarray(X)

        # Handle 1-dimensional arrays
        # 处理一维数组
        if X.ndim == 0 or X.ndim > 2:
            raise ValueError(
                "Expected 1D or 2D array, got %dD array instead" % X.ndim)
        elif X.ndim == 1:
            # Common case -- 1d array of numbers
            # 常见情况 -- 一维数字数组
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1

            # Complex dtype -- each field indicates a separate column
            # 复杂数据类型 -- 每个字段表示一个单独的列
            else:
                ncol = len(X.dtype.names)
        else:
            ncol = X.shape[1]

        iscomplex_X = np.iscomplexobj(X)
        # `fmt` can be a string with multiple insertion points or a
        # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
        # `fmt` 可以是包含多个插入点的字符串或格式列表。例如 '%10.5f\t%10d' 或 ('%10.5f', '$10d')
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
            format = delimiter.join(fmt)
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count('%')
            error = ValueError('fmt has wrong number of %% formats:  %s' % fmt)
            if n_fmt_chars == 1:
                if iscomplex_X:
                    fmt = [' (%s+%sj)' % (fmt, fmt), ] * ncol
                else:
                    fmt = [fmt, ] * ncol
                format = delimiter.join(fmt)
            elif iscomplex_X and n_fmt_chars != (2 * ncol):
                raise error
            elif ((not iscomplex_X) and n_fmt_chars != ncol):
                raise error
            else:
                format = fmt
        else:
            raise ValueError('invalid fmt: %r' % (fmt,))

        if len(header) > 0:
            header = header.replace('\n', '\n' + comments)
            fh.write(comments + header + newline)
        if iscomplex_X:
            for row in X:
                row2 = []
                for number in row:
                    row2.append(number.real)
                    row2.append(number.imag)
                s = format % tuple(row2) + newline
                fh.write(s.replace('+-', '-'))
        else:
            for row in X:
                try:
                    v = format % tuple(row) + newline
                except TypeError as e:
                    raise TypeError("Mismatch between array dtype ('%s') and "
                                    "format specifier ('%s')"
                                    % (str(X.dtype), format)) from e
                fh.write(v)

        if len(footer) > 0:
            footer = footer.replace('\n', '\n' + comments)
            fh.write(comments + footer + newline)
    finally:
        if own_fh:
            fh.close()
# 设置当前模块为 'numpy'
@set_module('numpy')
# 定义函数 fromregex，用于从文本文件中使用正则表达式解析并构建结构化数组
def fromregex(file, regexp, dtype, encoding=None):
    r"""
    Construct an array from a text file, using regular expression parsing.

    The returned array is always a structured array, and is constructed from
    all matches of the regular expression in the file. Groups in the regular
    expression are converted to fields of the structured array.

    Parameters
    ----------
    file : file, str, or pathlib.Path
        Filename or file object to read.

        .. versionchanged:: 1.22.0
            Now accepts `os.PathLike` implementations.
    regexp : str or regexp
        Regular expression used to parse the file.
        Groups in the regular expression correspond to fields in the dtype.
    dtype : dtype or list of dtypes
        Dtype for the structured array; must be a structured datatype.
    encoding : str, optional
        Encoding used to decode the inputfile. Does not apply to input streams.

        .. versionadded:: 1.14.0

    Returns
    -------
    output : ndarray
        The output array, containing the part of the content of `file` that
        was matched by `regexp`. `output` is always a structured array.

    Raises
    ------
    TypeError
        When `dtype` is not a valid dtype for a structured array.

    See Also
    --------
    fromstring, loadtxt

    Notes
    -----
    Dtypes for structured arrays can be specified in several forms, but all
    forms specify at least the data type and field name. For details see
    `basics.rec`.

    Examples
    --------
    >>> from io import StringIO
    >>> text = StringIO("1312 foo\n1534  bar\n444   qux")

    >>> regexp = r"(\d+)\s+(...)"  # match [digits, whitespace, anything]
    >>> output = np.fromregex(text, regexp,
    ...                       [('num', np.int64), ('key', 'S3')])
    >>> output
    array([(1312, b'foo'), (1534, b'bar'), ( 444, b'qux')],
          dtype=[('num', '<i8'), ('key', 'S3')])
    >>> output['num']
    array([1312, 1534,  444])

    """
    # 检查 file 是否具有 read 方法，如果没有则将其视为文件路径，并以指定的编码打开文件
    own_fh = False
    if not hasattr(file, "read"):
        file = os.fspath(file)
        file = np.lib._datasource.open(file, 'rt', encoding=encoding)
        own_fh = True
    try:
        # 如果 dtype 不是 np.dtype 对象，则尝试将其转换为 np.dtype 对象
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        
        # 如果 dtype 不是结构化数据类型，则抛出 TypeError 异常
        if dtype.names is None:
            raise TypeError('dtype must be a structured datatype.')

        # 读取文件内容到变量 content
        content = file.read()

        # 如果 content 是 bytes 类型，并且 regexp 是字符串类型，则将 regexp 转换为字节类型
        if isinstance(content, bytes) and isinstance(regexp, str):
            regexp = asbytes(regexp)

        # 如果 regexp 没有 match 方法，则将其编译为正则表达式对象
        if not hasattr(regexp, 'match'):
            regexp = re.compile(regexp)
        
        # 使用正则表达式在 content 中查找所有匹配的序列，保存到 seq 中
        seq = regexp.findall(content)
        
        # 如果 seq 不为空且第一个元素不是元组，则创建新的数组作为单一数据类型的结构化数组
        if seq and not isinstance(seq[0], tuple):
            # 创建新的数据类型为第一个字段的数据类型
            newdtype = np.dtype(dtype[dtype.names[0]])
            # 使用新的数据类型创建数组 output
            output = np.array(seq, dtype=newdtype)
            # 将 output 的数据类型重新解释为原始的结构化数据类型 dtype
            output.dtype = dtype
        else:
            # 否则直接创建结构化数组 output
            output = np.array(seq, dtype=dtype)

        # 返回结果 output
        return output
    finally:
        # 如果 own_fh 为 True，则关闭文件对象 file
        if own_fh:
            file.close()
# 设置一个装饰器函数，使其生成的数组可以像文档一样
@set_array_function_like_doc
# 设置模块名称为'numpy'
@set_module('numpy')
# 从文本文件中加载数据，处理缺失值
def genfromtxt(fname, dtype=float, comments='#', delimiter=None,
               skip_header=0, skip_footer=0, converters=None,
               missing_values=None, filling_values=None, usecols=None,
               names=None, excludelist=None,
               deletechars=''.join(sorted(NameValidator.defaultdeletechars)),
               replace_space='_', autostrip=False, case_sensitive=True,
               defaultfmt="f%i", unpack=None, usemask=False, loose=True,
               invalid_raise=True, max_rows=None, encoding=None,
               *, ndmin=0, like=None):
    """
    Load data from a text file, with missing values handled as specified.

    Each line past the first `skip_header` lines is split at the `delimiter`
    character, and characters following the `comments` character are discarded.

    Parameters
    ----------
    fname : file, str, pathlib.Path, list of str, generator
        File, filename, list, or generator to read.  If the filename
        extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note
        that generators must return bytes or strings. The strings
        in a list or produced by a generator are treated as lines.
    dtype : dtype, optional
        Data type of the resulting array.
        If None, the dtypes will be determined by the contents of each
        column, individually.
    comments : str, optional
        The character used to indicate the start of a comment.
        All the characters occurring on a line after a comment are discarded.
    delimiter : str, int, or sequence, optional
        The string used to separate values.  By default, any consecutive
        whitespaces act as delimiter.  An integer or sequence of integers
        can also be provided as width(s) of each field.
    skiprows : int, optional
        `skiprows` was removed in numpy 1.10. Please use `skip_header` instead.
    skip_header : int, optional
        The number of lines to skip at the beginning of the file.
    skip_footer : int, optional
        The number of lines to skip at the end of the file.
    converters : variable, optional
        The set of functions that convert the data of a column to a value.
        The converters can also be used to provide a default value
        for missing data: ``converters = {3: lambda s: float(s or 0)}``.

... (以下代码省略)
    usecols : sequence, optional
        # 定义一个可选参数，用于指定要读取的列的序列，其中第一个列为0。
        # 例如，使用 usecols = (1, 4, 5) 将提取第2、第5和第6列。
    names : {None, True, str, sequence}, optional
        # 如果 names 为 True，则从 skip_header 后的第一行读取字段名称。
        # 此行可以选择以注释分隔符开头。注释分隔符之前的任何内容都将被丢弃。
        # 如果 names 是一个序列或逗号分隔的单个字符串，则这些名称将用于定义结构化 dtype 的字段名称。
        # 如果 names 是 None，则使用 dtype 字段的名称（如果有）。
    excludelist : sequence, optional
        # 要排除的名称列表。此列表将附加到默认列表 ['return','file','print']。
        # 排除的名称将附加下划线，例如，'file' 将变为 'file_'。
    deletechars : str, optional
        # 要从名称中删除的无效字符的字符串。
    defaultfmt : str, optional
        # 用于定义默认字段名称的格式，例如 "f%i" 或 "f_%02i"。
    autostrip : bool, optional
        # 是否自动删除变量名称中的空格。
    replace_space : char, optional
        # 用于替换变量名称中空格的字符。默认为 '_'。
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        # 如果为 True，则字段名称区分大小写。
        # 如果为 False 或 'upper'，字段名称转换为大写。
        # 如果为 'lower'，字段名称转换为小写。
    unpack : bool, optional
        # 如果为 True，则返回的数组是转置的，因此可以使用 ``x, y, z = genfromtxt(...)`` 来解包。
        # 当与结构化数据类型一起使用时，为每个字段返回数组。
        # 默认为 False。
    usemask : bool, optional
        # 如果为 True，则返回一个掩码数组。
        # 如果为 False，则返回一个常规数组。
    loose : bool, optional
        # 如果为 True，则不会为无效值引发错误。
    invalid_raise : bool, optional
        # 如果为 True，则在检测到列数不一致时引发异常。
        # 如果为 False，则发出警告并跳过有问题的行。
    max_rows : int, optional
        # 要读取的最大行数。不能与 skip_footer 同时使用。
        # 如果给定，值必须至少为1。默认为读取整个文件。

        .. versionadded:: 1.10.0
    # `encoding`参数：用于解码输入文件的编码方式。当`fname`是文件对象时不适用。
    # 特殊值'bytes'保证在可能的情况下接收字节数组，并将latin1编码的字符串传递给转换器，以确保向后兼容性。
    # 将此值覆盖为接收Unicode数组并将字符串传递给转换器。如果设为None，则使用系统默认值。
    # 默认值为'bytes'。

    # `ndmin`参数：与`loadtxt`函数相同的参数。

    # 返回值：返回从文本文件中读取的数据。如果`usemask`为True，则返回一个掩码数组。

    # 参见：numpy.loadtxt函数，在没有数据缺失时的等效函数。

    # 注意：
    # * 当空格用作分隔符或没有给定分隔符输入时，两个字段之间不应有缺失数据。
    # * 当变量被命名（通过灵活的dtype或`names`序列）时，文件中不应有任何标题（否则会引发ValueError异常）。
    # * 默认情况下，单个值不会被去除空格。使用自定义转换器时，请确保该函数删除空格。
    # * 由于dtype的发现，自定义转换器可能会收到意外的值。

    # 参考资料：
    # [1] NumPy用户指南，`I/O with NumPy`章节。
    # 设置分隔符，指定为列表 [1, 3, 5]
    ...     delimiter=[1,3,5])
    # 示例数据
    >>> data
    # 创建一个数组，包含整数、浮点数和字符串，每列的数据类型分别为 '<i8'、'<f8' 和 '<U5'
    array((1, 1.3, 'abcde'),
          dtype=[('intvar', '<i8'), ('fltvar', '<f8'), ('strvar', '<U5')])

    # 一个演示用的例子，展示如何添加注释

    >>> f = StringIO('''
    ... text,# of chars
    ... hello world,11
    ... numpy,5''')
    # 使用 StringIO 创建一个包含文本数据的对象 f，每行使用逗号分隔
    >>> np.genfromtxt(f, dtype='S12,S12', delimiter=',')
    # 从文本文件中加载数据到 NumPy 数组，指定每列的数据类型为 'S12'，使用逗号作为分隔符
    array([(b'text', b''), (b'hello world', b'11'), (b'numpy', b'5')],
      dtype=[('f0', 'S12'), ('f1', 'S12')])

    """

    # 如果提供了 like 参数，则调用 _genfromtxt_with_like 函数进行处理
    if like is not None:
        return _genfromtxt_with_like(
            like, fname, dtype=dtype, comments=comments, delimiter=delimiter,
            skip_header=skip_header, skip_footer=skip_footer,
            converters=converters, missing_values=missing_values,
            filling_values=filling_values, usecols=usecols, names=names,
            excludelist=excludelist, deletechars=deletechars,
            replace_space=replace_space, autostrip=autostrip,
            case_sensitive=case_sensitive, defaultfmt=defaultfmt,
            unpack=unpack, usemask=usemask, loose=loose,
            invalid_raise=invalid_raise, max_rows=max_rows, encoding=encoding,
            ndmin=ndmin,
        )

    # 检查 ndmin 参数，确保其为整数且合理
    _ensure_ndmin_ndarray_check_param(ndmin)

    # 如果指定了 max_rows 参数，则进行相关的验证
    if max_rows is not None:
        # 如果同时指定了 skip_footer 参数，则抛出 ValueError
        if skip_footer:
            raise ValueError(
                    "The keywords 'skip_footer' and 'max_rows' can not be "
                    "specified at the same time.")
        # 如果 max_rows 小于 1，则抛出 ValueError
        if max_rows < 1:
            raise ValueError("'max_rows' must be at least 1.")

    # 如果 usemask 参数为 True，则导入 MaskedArray 和 make_mask_descr 函数
    if usemask:
        from numpy.ma import MaskedArray, make_mask_descr

    # 检查 converters 参数是否为字典类型，如果不是则抛出 TypeError
    # 用户自定义的转换器字典
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        raise TypeError(
            "The input argument 'converter' should be a valid dictionary "
            "(got '%s' instead)" % type(user_converters))

    # 如果 encoding 设置为 'bytes'，则将其重置为 None，并设置 byte_converters 为 True
    # 否则将 byte_converters 设置为 False
    if encoding == 'bytes':
        encoding = None
        byte_converters = True
    else:
        byte_converters = False

    # 初始化文件句柄 fid，使用 np.lib._datasource.open 打开文件，并使用 contextlib.closing 确保在使用后关闭
    # 如果 fname 是 os.PathLike 类型，则转换为字符串路径
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    # 如果 fname 是字符串类型，则直接打开文件并设置为文本模式，指定编码为 encoding
    # 否则直接使用 fname 作为文件句柄，使用 contextlib.nullcontext 确保在不需要时正确关闭
    if isinstance(fname, str):
        fid = np.lib._datasource.open(fname, 'rt', encoding=encoding)
        fid_ctx = contextlib.closing(fid)
    else:
        fid = fname
        fid_ctx = contextlib.nullcontext(fid)
    try:
        # 将 fid 转换为迭代器 fhd
        fhd = iter(fid)
    except TypeError as e:
        # 如果无法转换为迭代器，则抛出 TypeError，提示 fname 的类型错误
        raise TypeError(
            "fname must be a string, a filehandle, a sequence of strings,\n"
            f"or an iterator of strings. Got {type(fname)} instead."
        ) from e
    # 更新转换器（如果需要）
    # 如果数据类型未指定，则尝试对每列应用转换器
    if dtype is None:
        # 遍历转换器列表中的每个转换器及其索引
        for (i, converter) in enumerate(converters):
            # 从行中获取当前列的数据
            current_column = [itemgetter(i)(_m) for _m in rows]
            try:
                # 尝试升级当前列的数据
                converter.iterupgrade(current_column)
            except ConverterLockError:
                # 如果转换器被锁定，则生成错误消息
                errmsg = "Converter #%i is locked and cannot be upgraded: " % i
                # 获取当前列的所有值
                current_column = map(itemgetter(i), rows)
                # 遍历处理每个值
                for (j, value) in enumerate(current_column):
                    try:
                        # 尝试对当前值进行升级
                        converter.upgrade(value)
                    except (ConverterError, ValueError):
                        # 如果升级过程中发生错误，则记录错误消息
                        errmsg += "(occurred line #%i for value '%s')"
                        errmsg %= (j + 1 + skip_header, value)
                        # 抛出转换器错误
                        raise ConverterError(errmsg)

    # 检查是否存在无效值
    nbinvalid = len(invalid)
    if nbinvalid > 0:
        # 计算出处理后的行数
        nbrows = len(rows) + nbinvalid - skip_footer
        # 构造错误消息模板
        template = "    Line #%%i (got %%i columns instead of %i)" % nbcols
        if skip_footer > 0:
            # 计算需要跳过的无效值数量
            nbinvalid_skipped = len([_ for _ in invalid
                                     if _[0] > nbrows + skip_header])
            # 移除超出范围的无效值
            invalid = invalid[:nbinvalid - nbinvalid_skipped]
            # 调整要跳过的尾部行数
            skip_footer -= nbinvalid_skipped
# 如果指定了 skip_footer 参数，则减少 nbrows 的数量，以排除尾部行数
nbrows -= skip_footer

# 如果存在无效数据条目，生成错误消息列表 errmsg
errmsg = [template % (i, nb)
          for (i, nb) in invalid]

# 如果 skip_footer 大于 0，则删除 rows 列表中的最后 skip_footer 行数据
if skip_footer > 0:
    rows = rows[:-skip_footer]
    # 如果使用掩码，则同样删除掩码列表中的最后 skip_footer 行数据
    if usemask:
        masks = masks[:-skip_footer]

# 根据 loose 参数决定采用宽松转换还是严格转换方式处理 rows 中的每个值
if loose:
    # 使用 _loose_call 方法对每列数据应用相应的转换器 conv
    rows = list(
        zip(*[[conv._loose_call(_r) for _r in map(itemgetter(i), rows)]
              for (i, conv) in enumerate(converters)]))
else:
    # 使用 _strict_call 方法对每列数据应用相应的转换器 conv
    rows = list(
        zip(*[[conv._strict_call(_r) for _r in map(itemgetter(i), rows)]
              for (i, conv) in enumerate(converters)]))

# 将转换后的数据列表赋值给 data 变量
data = rows
    if dtype is None:
        # 如果未指定数据类型，则从转换器的类型中获取数据类型
        column_types = [conv.type for conv in converters]
        # 找到包含字符串的列的索引...
        strcolidx = [i for (i, v) in enumerate(column_types)
                     if v == np.str_]

        if byte_converters and strcolidx:
            # 将字符串转换回字节以保持向后兼容性
            warnings.warn(
                "Reading unicode strings without specifying the encoding "
                "argument is deprecated. Set the encoding, use None for the "
                "system default.",
                np.exceptions.VisibleDeprecationWarning, stacklevel=2)

            # 定义函数，用于编码 Unicode 列
            def encode_unicode_cols(row_tup):
                row = list(row_tup)
                for i in strcolidx:
                    row[i] = row[i].encode('latin1')
                return tuple(row)

            try:
                # 尝试对数据中的 Unicode 列进行编码转换
                data = [encode_unicode_cols(r) for r in data]
            except UnicodeEncodeError:
                pass
            else:
                # 如果成功转换，则更新对应列的类型为字节类型
                for i in strcolidx:
                    column_types[i] = np.bytes_

        # 更新字符串类型的长度为正确的长度
        sized_column_types = column_types[:]
        for i, col_type in enumerate(column_types):
            if np.issubdtype(col_type, np.character):
                # 计算每列中最大的字符数目
                n_chars = max(len(row[i]) for row in data)
                sized_column_types[i] = (col_type, n_chars)

        if names is None:
            # 如果未指定列名，则根据数据的统一类型（尺寸调整前）创建基础数据类型
            base = {
                c_type
                for c, c_type in zip(converters, column_types)
                if c._checked}
            if len(base) == 1:
                uniform_type, = base
                (ddtype, mdtype) = (uniform_type, bool)
            else:
                # 使用默认格式生成列名和数据类型的元组列表
                ddtype = [(defaultfmt % i, dt)
                          for (i, dt) in enumerate(sized_column_types)]
                if usemask:
                    # 如果使用掩码，则生成掩码列名和布尔类型的元组列表
                    mdtype = [(defaultfmt % i, bool)
                              for (i, dt) in enumerate(sized_column_types)]
        else:
            # 如果指定了列名，则使用指定的列名和长度调整后的数据类型创建元组列表
            ddtype = list(zip(names, sized_column_types))
            mdtype = list(zip(names, [bool] * len(sized_column_types)))
        # 根据指定的数据类型创建 NumPy 数组
        output = np.array(data, dtype=ddtype)
        if usemask:
            # 如果使用掩码，则根据掩码数据类型创建 NumPy 数组
            outputmask = np.array(masks, dtype=mdtype)
    else:
        # 如果需要，覆盖初始的数据类型名称
        if names and dtype.names is not None:
            dtype.names = names
        # Case 1. We have a structured type
        if len(dtype_flat) > 1:
            # 嵌套的数据类型，例如[('a', int), ('b', [('b0', int), ('b1', 'f4')])]
            # 首先，使用扁平化的数据类型创建数组：
            # [('a', int), ('b1', int), ('b2', float)]
            # 然后，使用指定的数据类型查看数组。
            if 'O' in (_.char for _ in dtype_flat):
                if has_nested_fields(dtype):
                    raise NotImplementedError(
                        "Nested fields involving objects are not supported...")
                else:
                    output = np.array(data, dtype=dtype)
            else:
                rows = np.array(data, dtype=[('', _) for _ in dtype_flat])
                output = rows.view(dtype)
            # 现在，以相同的方式处理行掩码
            if usemask:
                rowmasks = np.array(
                    masks, dtype=np.dtype([('', bool) for t in dtype_flat]))
                # 构建新的数据类型描述
                mdtype = make_mask_descr(dtype)
                outputmask = rowmasks.view(mdtype)
        # Case #2. We have a basic dtype
        else:
            # 我们使用了一些用户定义的转换器
            if user_converters:
                ishomogeneous = True
                descr = []
                for i, ttype in enumerate([conv.type for conv in converters]):
                    # 保持当前转换器的数据类型
                    if i in user_converters:
                        ishomogeneous &= (ttype == dtype.type)
                        if np.issubdtype(ttype, np.character):
                            ttype = (ttype, max(len(row[i]) for row in data))
                        descr.append(('', ttype))
                    else:
                        descr.append(('', dtype))
                # 所以我们改变了数据类型？
                if not ishomogeneous:
                    # 我们有多个字段
                    if len(descr) > 1:
                        dtype = np.dtype(descr)
                    # 我们只有一个字段：如果不需要，去掉名称。
                    else:
                        dtype = np.dtype(ttype)
            #
            output = np.array(data, dtype)
            if usemask:
                if dtype.names is not None:
                    mdtype = [(_, bool) for _ in dtype.names]
                else:
                    mdtype = bool
                outputmask = np.array(masks, dtype=mdtype)
    # 尝试处理我们遗漏的缺失数据
    names = output.dtype.names
    # 如果 usemask 为真且 names 不为空，则执行以下操作
    if usemask and names:
        # 对于 names 和 converters 中的每一对 (name, conv)，执行以下循环
        for (name, conv) in zip(names, converters):
            # 从 conv.missing_values 中获取非空的缺失值列表
            missing_values = [conv(_) for _ in conv.missing_values if _ != '']
            # 对于每个缺失值 mval，在 outputmask[name] 上设置相应的位
            for mval in missing_values:
                outputmask[name] |= (output[name] == mval)
    
    # 构造最终的数组
    if usemask:
        # 将 output 转换为 MaskedArray 类型
        output = output.view(MaskedArray)
        # 将 outputmask 赋值给 output 的掩码属性
        output._mask = outputmask

    # 确保 output 至少是 ndmin 维度的 ndarray
    output = _ensure_ndmin_ndarray(output, ndmin=ndmin)

    # 如果 unpack 为真，则根据 names 的情况返回相应的结果
    if unpack:
        if names is None:
            # 如果 names 为空，则返回 output 的转置
            return output.T
        elif len(names) == 1:
            # 如果 names 中只有一个元素，则返回该元素对应的数据
            # 对单一名称的 dtype 也进行压缩
            return output[names[0]]
        else:
            # 对于具有多个字段的结构化数组，返回每个字段的数组
            return [output[field] for field in names]
    
    # 如果不需要 unpack，则直接返回 output
    return output
# 使用 `array_function_dispatch()` 装饰 `genfromtxt` 函数，并将其赋值给 `_genfromtxt_with_like`
_genfromtxt_with_like = array_function_dispatch()(genfromtxt)


def recfromtxt(fname, **kwargs):
    """
    从文件中加载 ASCII 数据，并以记录数组的形式返回。

    如果 `usemask=False`，则返回标准的 `recarray`，
    如果 `usemask=True`，则返回一个 MaskedRecords 数组。

    .. deprecated:: 2.0
        使用 `numpy.genfromtxt` 替代。

    Parameters
    ----------
    fname, kwargs : 输入参数的描述，请参见 `genfromtxt`。

    See Also
    --------
    numpy.genfromtxt : 通用函数

    Notes
    -----
    默认情况下，`dtype` 是 None，这意味着输出数组的数据类型将从数据中确定。
    """

    # 在 NumPy 2.0 中弃用，2023-07-11
    warnings.warn(
        "`recfromtxt` 已弃用，"
        "请使用 `numpy.genfromtxt` 替代。"
        "（在 NumPy 2.0 中弃用）",
        DeprecationWarning,
        stacklevel=2
    )

    # 设置 `dtype` 为 None，如适用
    kwargs.setdefault("dtype", None)
    # 获取 `usemask` 参数，默认为 False
    usemask = kwargs.get('usemask', False)
    # 调用 `genfromtxt` 函数加载数据
    output = genfromtxt(fname, **kwargs)
    # 根据 `usemask` 参数选择输出的数据类型
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output


def recfromcsv(fname, **kwargs):
    """
    从逗号分隔的文件中加载 ASCII 数据。

    返回的数组是一个记录数组（如果 `usemask=False`，参见 `recarray`）或一个掩码记录数组（如果 `usemask=True`，
    参见 `ma.mrecords.MaskedRecords`）。

    .. deprecated:: 2.0
        使用 `numpy.genfromtxt` 并将逗号作为 `delimiter` 替代。

    Parameters
    ----------
    fname, kwargs : 输入参数的描述，请参见 `genfromtxt`。

    See Also
    --------
    numpy.genfromtxt : 用于加载 ASCII 数据的通用函数。

    Notes
    -----
    默认情况下，`dtype` 是 None，这意味着输出数组的数据类型将从数据中确定。
    """

    # 在 NumPy 2.0 中弃用，2023-07-11
    warnings.warn(
        "`recfromcsv` 已弃用，"
        "请使用 `numpy.genfromtxt` 并将逗号作为 `delimiter` 替代。"
        "（在 NumPy 2.0 中弃用）",
        DeprecationWarning,
        stacklevel=2
    )

    # 设置用于 CSV 导入的 genfromtxt 的默认 kwargs
    kwargs.setdefault("case_sensitive", "lower")
    kwargs.setdefault("names", True)
    kwargs.setdefault("delimiter", ",")
    kwargs.setdefault("dtype", None)
    # 调用 `genfromtxt` 函数加载数据
    output = genfromtxt(fname, **kwargs)

    # 根据 `usemask` 参数选择输出的数据类型
    usemask = kwargs.get("usemask", False)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output
```