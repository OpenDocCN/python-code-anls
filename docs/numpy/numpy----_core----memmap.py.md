# `.\numpy\numpy\_core\memmap.py`

```
# 从 contextlib 模块中导入 nullcontext 上下文管理器，用于创建一个空的上下文
from contextlib import nullcontext
# 导入 operator 模块，用于一些操作符的函数形式
import operator
# 导入 numpy 库，并将其作为 np 别名引入
import numpy as np
# 从 .._utils 中导入 set_module 函数
from .._utils import set_module
# 从当前包的 numeric 模块中导入 uint8, ndarray, dtype 类型
from .numeric import uint8, ndarray, dtype

# 设置 __all__ 变量，指定导出的模块成员
__all__ = ['memmap']

# 将 dtype 别名赋值给 dtypedescr 变量
dtypedescr = dtype
# 定义有效的文件访问模式列表
valid_filemodes = ["r", "c", "r+", "w+"]
# 定义可写入的文件访问模式列表
writeable_filemodes = ["r+", "w+"]

# 创建一个字典，将不同的模式字符串映射到对应的文件访问模式
mode_equivalents = {
    "readonly":"r",
    "copyonwrite":"c",
    "readwrite":"r+",
    "write":"w+"
    }

# 使用装饰器 set_module('numpy') 将 memmap 类设置为 numpy 模块的一部分
@set_module('numpy')
# 定义 memmap 类，继承自 ndarray 类
class memmap(ndarray):
    """Create a memory-map to an array stored in a *binary* file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  NumPy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.

    This subclass of ndarray has some unpleasant interactions with
    some operations, because it doesn't quite fit properly as a subclass.
    An alternative to using this subclass is to create the ``mmap``
    object yourself, then create an ndarray with ndarray.__new__ directly,
    passing the object created in its 'buffer=' parameter.

    This class may at some point be turned into a factory function
    which returns a view into an mmap buffer.

    Flush the memmap instance to write the changes to the file. Currently there
    is no API to close the underlying ``mmap``. It is tricky to ensure the
    resource is actually closed, since it may be shared between different
    memmap instances.


    Parameters
    ----------
    filename : str, file-like object, or pathlib.Path instance
        The file name or file object to be used as the array data buffer.
    dtype : data-type, optional
        The data-type used to interpret the file contents.
        Default is `uint8`.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:

        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        |      | If ``mode == 'w+'`` then `shape` must also be specified.    |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+

        Default is 'r+'.
    """
    # 类文档字符串，说明了 memmap 类的作用和特性
    pass
    # offset : int, optional
    #     数据在文件中的偏移量。如果 `offset` 是以字节为单位的，通常应该是 `dtype` 的字节大小的倍数。
    #     当 `mode != 'r'` 时，即使偏移量超出文件末尾也是有效的；文件将被扩展以容纳额外的数据。
    #     默认情况下，即使 `filename` 是文件指针 `fp` 且 `fp.tell() != 0`，`memmap` 也将从文件的开头开始。
    shape : int or sequence of ints, optional
        # 数组的期望形状。如果 `mode == 'r'` 并且 `offset` 后的剩余字节数不是 `dtype` 的字节大小的倍数，
        # 则必须指定 `shape`。默认情况下，返回的数组将是一维的，元素数量由文件大小和数据类型确定。

        .. versionchanged:: 2.0
            # shape 参数现在可以是任何整数序列类型，先前只限于元组和整数。
    
    order : {'C', 'F'}, optional
        # 指定 ndarray 内存布局的顺序：
        # :term:`行主序`，即 C 风格，或 :term:`列主序`，即 Fortran 风格。
        # 仅当形状大于一维时才会起作用。默认顺序是 'C'。

    Attributes
    ----------
    filename : str or pathlib.Path instance
        # 映射文件的路径。
    offset : int
        # 文件中的偏移位置。
    mode : str
        # 文件模式。

    Methods
    -------
    flush
        # 将内存中的任何更改刷新到磁盘文件。
        # 删除 memmap 对象时，会先调用 flush 来将更改写入磁盘。

    See also
    --------
    lib.format.open_memmap : 创建或加载一个内存映射的 `.npy` 文件。

    Notes
    -----
    # memmap 对象可用于任何接受 ndarray 的地方。
    # 给定一个 memmap `fp`，`isinstance(fp, numpy.ndarray)` 返回 `True`。

    # 在 32 位系统上，内存映射文件不能超过 2GB。

    # 当 memmap 导致文件在文件系统中被创建或扩展到当前大小之外时，新部分的内容是未指定的。
    # 在具有 POSIX 文件系统语义的系统上，扩展部分将填充为零字节。

    Examples
    --------
    >>> data = np.arange(12, dtype='float32')
    >>> data.resize((3,4))

    # 此示例使用临时文件，以便 doctest 不会将文件写入您的目录。实际应使用一个正常的文件名。

    >>> from tempfile import mkdtemp
    >>> import os.path as path
    >>> filename = path.join(mkdtemp(), 'newfile.dat')

    # 使用与数据匹配的 dtype 和 shape 创建一个 memmap：

    >>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
    >>> fp
    memmap([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]], dtype=float32)

    # 将数据写入 memmap 数组：

    >>> fp[:] = data[:]
    >>> fp
    __array_priority__ = -100.0
    # 设置数组优先级，用于指定数组对象之间操作的优先级
    
    def __array_finalize__(self, obj):
        # 如果 obj 是 memmap 类型且与当前数组共享内存，则继承相关属性
        if hasattr(obj, '_mmap') and np.may_share_memory(self, obj):
            self._mmap = obj._mmap
            self.filename = obj.filename
            self.offset = obj.offset
            self.mode = obj.mode
        else:
            # 否则，置空相关属性
            self._mmap = None
            self.filename = None
            self.offset = None
            self.mode = None
    
    def flush(self):
        """
        将数组的任何更改写入到磁盘文件中。
    
        详细信息请参考 `memmap`。
    
        参数
        ----------
        None
    
        参见
        --------
        memmap
        """
        # 如果数组基于其他对象，并且该对象具有 flush 方法，则调用其 flush 方法
        if self.base is not None and hasattr(self.base, 'flush'):
            self.base.flush()
    # 覆盖父类方法`__array_wrap__`，处理数组的包装
    def __array_wrap__(self, arr, context=None, return_scalar=False):
        # 调用父类的`__array_wrap__`方法，传入数组`arr`和上下文`context`
        arr = super().__array_wrap__(arr, context)

        # 如果`self`与`arr`相同，或者`self`不是`memmap`类型，则直接返回`arr`
        if self is arr or type(self) is not memmap:
            return arr

        # 如果需要返回标量而不是0维的`memmap`，例如在`np.sum`中`axis=None`的情况
        if return_scalar:
            return arr[()]

        # 否则返回`arr`的`np.ndarray`视图
        return arr.view(np.ndarray)

    # 覆盖父类方法`__getitem__`，处理数组的索引操作
    def __getitem__(self, index):
        # 调用父类的`__getitem__`方法，传入索引`index`，获取结果`res`
        res = super().__getitem__(index)
        
        # 如果`res`是`memmap`类型且没有内存映射对象`_mmap`，则返回`ndarray`类型的视图
        if type(res) is memmap and res._mmap is None:
            return res.view(type=ndarray)
        
        # 否则直接返回`res`
        return res
```