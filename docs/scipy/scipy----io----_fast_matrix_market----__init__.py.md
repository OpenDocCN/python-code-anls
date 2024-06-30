# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\__init__.py`

```
# Copyright (C) 2022-2023 Adam Lugowski. All rights reserved.
# Use of this source code is governed by the BSD 2-clause license found in
# the LICENSE.txt file.
# SPDX-License-Identifier: BSD-2-Clause
"""
Matrix Market I/O with a C++ backend.
See http://math.nist.gov/MatrixMarket/formats.html
for information about the Matrix Market format.

.. versionadded:: 1.12.0
"""
import io  # 导入 io 模块，用于处理文件流和缓冲区
import os  # 导入 os 模块，提供操作系统相关的功能

import numpy as np  # 导入 NumPy 库，用于数值计算
import scipy.sparse  # 导入 SciPy 稀疏矩阵模块
from scipy.io import _mmio  # 导入 SciPy 内部的 Matrix Market I/O 模块

__all__ = ['mminfo', 'mmread', 'mmwrite']  # 定义模块的公开接口列表

PARALLELISM = 0  # 并行度设为 0，表示使用系统中的 CPU 核心数
"""
Number of threads that `mmread()` and `mmwrite()` use.
0 means number of CPUs in the system.
Use `threadpoolctl` to set this value.
"""

ALWAYS_FIND_SYMMETRY = False  # 是否总是查找对称性，默认为 False

_field_to_dtype = {
    "integer": "int64",
    "unsigned-integer": "uint64",
    "real": "float64",
    "complex": "complex",
    "pattern": "float64",
}
# 字段类型到 NumPy 数据类型的映射字典

def _fmm_version():
    from . import _fmm_core
    return _fmm_core.__version__
# 返回 _fmm_core 模块的版本信息

# Register with threadpoolctl, if available
try:
    import threadpoolctl  # 尝试导入 threadpoolctl 模块

    class _FMMThreadPoolCtlController(threadpoolctl.LibController):
        user_api = "scipy"
        internal_api = "scipy_mmio"

        filename_prefixes = ("_fmm_core",)

        def get_num_threads(self):
            global PARALLELISM
            return PARALLELISM

        def set_num_threads(self, num_threads):
            global PARALLELISM
            PARALLELISM = num_threads

        def get_version(self):
            return _fmm_version

        def set_additional_attributes(self):
            pass

    threadpoolctl.register(_FMMThreadPoolCtlController)
# 注册自定义的 LibController 类用于控制线程池数量
except (ImportError, AttributeError):
    # threadpoolctl not installed or version too old
    pass
# 如果无法导入 threadpoolctl 模块或者版本过旧，则捕获异常并忽略

class _TextToBytesWrapper(io.BufferedReader):
    """
    Convert a TextIOBase string stream to a byte stream.
    """

    def __init__(self, text_io_buffer, encoding=None, errors=None, **kwargs):
        super().__init__(text_io_buffer, **kwargs)
        self.encoding = encoding or text_io_buffer.encoding or 'utf-8'
        self.errors = errors or text_io_buffer.errors or 'strict'

    def __del__(self):
        # do not close the wrapped stream
        self.detach()

    def _encoding_call(self, method_name, *args, **kwargs):
        raw_method = getattr(self.raw, method_name)
        val = raw_method(*args, **kwargs)
        return val.encode(self.encoding, errors=self.errors)
    # 定义私有方法 _encoding_call，将从文本流中读取的数据编码成字节流

    def read(self, size=-1):
        return self._encoding_call('read', size)
    # 重载 read 方法，使用 _encoding_call 方法读取数据并返回编码后的字节流

    def read1(self, size=-1):
        return self._encoding_call('read1', size)
    # 重载 read1 方法，使用 _encoding_call 方法读取数据并返回编码后的字节流

    def peek(self, size=-1):
        return self._encoding_call('peek', size)
    # 重载 peek 方法，使用 _encoding_call 方法读取数据并返回编码后的字节流
    # 随机定位不被允许，因为字节偏移和字符偏移之间的转换不是平凡的，
    # 可能导致字节偏移正好落在一个字符之内。
    if offset == 0 and whence == 0 or \
       offset == 0 and whence == 2:
        # 如果偏移量为0且起始点或结束点，则允许定位
        super().seek(offset, whence)
    else:
        # 放弃任何其他的定位操作
        # 在此应用中，这可能发生在 pystreambuf 在 sync() 期间进行 seek 操作时，
        # 这种情况可能发生在关闭部分读取的流时。
        # 例如，当 mminfo() 只读取头部然后退出时。
        pass
# 从模块中导入_fmm_core，用于与底层功能交互
from . import _fmm_core

# 根据传入的游标对象读取 MatrixMarket 数组主体部分
def _read_body_array(cursor):
    # 创建一个与 header 形状相同的全零数组，数据类型根据字段类型动态确定
    vals = np.zeros(cursor.header.shape, dtype=_field_to_dtype.get(cursor.header.field))
    # 调用底层函数_read_body_array读取数组主体数据
    _fmm_core.read_body_array(cursor, vals)
    return vals

# 根据传入的游标对象读取 MatrixMarket 坐标部分
def _read_body_coo(cursor, generalize_symmetry=True):
    # 从模块中导入_fmm_core，用于与底层功能交互
    from . import _fmm_core

    # 默认使用 int32 作为索引数据类型，如果行数或列数超过 int32 范围，则使用 int64
    index_dtype = "int32"
    if cursor.header.nrows >= 2**31 or cursor.header.ncols >= 2**31:
        index_dtype = "int64"

    # 创建三个全零数组，用于存储坐标数据和值数据，数据类型根据字段类型动态确定
    i = np.zeros(cursor.header.nnz, dtype=index_dtype)
    j = np.zeros(cursor.header.nnz, dtype=index_dtype)
    data = np.zeros(cursor.header.nnz, dtype=_field_to_dtype.get(cursor.header.field))

    # 调用底层函数_read_body_coo读取坐标数据和值数据
    _fmm_core.read_body_coo(cursor, i, j, data)

    # 如果 generalize_symmetry 为真且矩阵不是 general 对称的，则处理对称性
    if generalize_symmetry and cursor.header.symmetry != "general":
        off_diagonal_mask = (i != j)
        off_diagonal_rows = i[off_diagonal_mask]
        off_diagonal_cols = j[off_diagonal_mask]
        off_diagonal_data = data[off_diagonal_mask]

        # 根据矩阵的对称类型调整数据
        if cursor.header.symmetry == "skew-symmetric":
            off_diagonal_data *= -1
        elif cursor.header.symmetry == "hermitian":
            off_diagonal_data = off_diagonal_data.conjugate()

        # 扩展坐标和数据数组，添加非对角线元素
        i = np.concatenate((i, off_diagonal_cols))
        j = np.concatenate((j, off_diagonal_rows))
        data = np.concatenate((data, off_diagonal_data))

    # 返回元组，包含数据和坐标数组，以及 MatrixMarket 矩阵形状
    return (data, (i, j)), cursor.header.shape

# 获取用于读取的游标对象，支持不同的输入源和并行度选项
def _get_read_cursor(source, parallelism=None):
    # 从模块中导入_fmm_core，用于与底层功能交互
    from . import _fmm_core

    # 初始化一个需要关闭的流对象的变量
    ret_stream_to_close = None
    # 如果未提供并行度选项，则使用默认值 PARALLELISM
    if parallelism is None:
        parallelism = PARALLELISM

    try:
        # 尝试将 source 转换为文件路径字符串
        source = os.fspath(source)
        is_path = True  # 标记为文件路径
    except TypeError:
        is_path = False  # source 不是文件路径

    # 如果 source 是文件路径
    if is_path:
        path = str(source)
        # 根据文件扩展名选择合适的读取方式
        if path.endswith('.gz'):
            import gzip
            source = gzip.GzipFile(path, 'r')  # 使用 gzip 解压缩文件
            ret_stream_to_close = source  # 需要关闭的流对象为解压缩文件对象
        elif path.endswith('.bz2'):
            import bz2
            source = bz2.BZ2File(path, 'rb')  # 使用 bz2 解压缩文件
            ret_stream_to_close = source  # 需要关闭的流对象为解压缩文件对象
        else:
            return _fmm_core.open_read_file(path, parallelism), ret_stream_to_close

    # 如果 source 是流对象
    if hasattr(source, "read"):
        # 如果 source 是文本流对象，则转换为字节流对象
        if isinstance(source, io.TextIOBase):
            source = _TextToBytesWrapper(source)
        # 调用底层函数_open_read_stream打开读取流
        return _fmm_core.open_read_stream(source, parallelism), ret_stream_to_close
    else:
        # 抛出异常，表示未知的 source 类型
        raise TypeError("Unknown source type")

# 获取用于写入的游标对象，支持设置文件属性如对称性和精度
def _get_write_cursor(target, h=None, comment=None, parallelism=None,
                      symmetry="general", precision=None):
    # 从模块中导入_fmm_core，用于与底层功能交互
    from . import _fmm_core

    # 如果未提供并行度选项，则使用默认值 PARALLELISM
    if parallelism is None:
        parallelism = PARALLELISM
    # 如果未提供注释，则使用空字符串
    if comment is None:
        comment = ''
    # 如果未提供对称性选项，则默认为 "general"
    if symmetry is None:
        symmetry = "general"
    # 如果未提供精度选项，则默认为 -1
    if precision is None:
        precision = -1
    # 如果 h 为空值，则调用 _fmm_core.header 函数生成头部信息，包括注释和对称性
    if not h:
        h = _fmm_core.header(comment=comment, symmetry=symmetry)

    try:
        # 尝试将 target 转换为文件路径的字符串表示形式
        target = os.fspath(target)
        # 如果成功转换，表示 target 是文件路径
        # 调用 _fmm_core.open_write_file 打开并写入文件，传入文件路径、头部信息、并行性和精度参数
        return _fmm_core.open_write_file(str(target), h, parallelism, precision)
    except TypeError:
        pass

    # 如果 target 没有 "write" 属性，即不是文件路径字符串，检查其是否具有 "write" 方法
    if hasattr(target, "write"):
        # 如果是流对象
        if isinstance(target, io.TextIOBase):
            # 如果是文本模式的流对象，抛出类型错误
            raise TypeError("target stream must be open in binary mode.")
        # 调用 _fmm_core.open_write_stream 打开并写入流对象，传入流对象、头部信息、并行性和精度参数
        return _fmm_core.open_write_stream(target, h, parallelism, precision)
    else:
        # 如果 target 既不是文件路径字符串，也没有 "write" 方法，抛出类型错误
        raise TypeError("Unknown source object")
def _apply_field(data, field, no_pattern=False):
    """
    Ensure that ``data.dtype`` is compatible with the specified MatrixMarket field type.

    Parameters
    ----------
    data : ndarray
        Input array.

    field : str
        Matrix Market field, such as 'real', 'complex', 'integer', 'pattern'.

    no_pattern : bool, optional
        Whether an empty array may be returned for a 'pattern' field.

    Returns
    -------
    data : ndarray
        Input data if no conversion necessary, or a converted version
    """

    # 如果 field 为 None，则直接返回原始数据
    if field is None:
        return data
    # 如果 field 为 "pattern"，根据 no_pattern 参数决定返回空数组或者原始数据
    if field == "pattern":
        if no_pattern:
            return data
        else:
            return np.zeros(0)

    # 根据 field 获取对应的数据类型
    dtype = _field_to_dtype.get(field, None)
    # 如果未找到对应的数据类型，则抛出 ValueError 异常
    if dtype is None:
        raise ValueError("Invalid field.")

    # 将输入数据转换为指定的数据类型并返回
    return np.asarray(data, dtype=dtype)


def _validate_symmetry(symmetry):
    """
    Check that the symmetry parameter is one that MatrixMarket allows..
    """
    # 如果 symmetry 为 None，则返回 "general"
    if symmetry is None:
        return "general"

    # 将 symmetry 转换为小写字符串
    symmetry = str(symmetry).lower()
    # 定义允许的对称性列表
    symmetries = ["general", "symmetric", "skew-symmetric", "hermitian"]
    # 如果 symmetry 不在允许的对称性列表中，则抛出 ValueError 异常
    if symmetry not in symmetries:
        raise ValueError("Invalid symmetry. Must be one of: " + ", ".join(symmetries))

    # 返回符合要求的对称性字符串
    return symmetry


def mmread(source):
    """
    Reads the contents of a Matrix Market file-like 'source' into a matrix.

    Parameters
    ----------
    source : str or file-like
        Matrix Market filename (extensions .mtx, .mtz.gz)
        or open file-like object.

    Returns
    -------
    a : ndarray or coo_matrix
        Dense or sparse matrix depending on the matrix format in the
        Matrix Market file.

    Notes
    -----
    .. versionchanged:: 1.12.0
        C++ implementation.

    Examples
    --------
    >>> from io import StringIO
    >>> from scipy.io import mmread

    >>> text = '''%%MatrixMarket matrix coordinate real general
    ...  5 5 7
    ...  2 3 1.0
    ...  3 4 2.0
    ...  3 5 3.0
    ...  4 1 4.0
    ...  4 2 5.0
    ...  4 3 6.0
    ...  4 4 7.0
    ... '''

    ``mmread(source)`` returns the data as sparse matrix in COO format.

    >>> m = mmread(StringIO(text))
    >>> m
    <COOrdinate sparse matrix of dtype 'float64'
        with 7 stored elements and shape (5, 5)>
    >>> m.toarray()
    array([[0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 2., 3.],
           [4., 5., 6., 7., 0.],
           [0., 0., 0., 0., 0.]])

    This method is threaded.
    The default number of threads is equal to the number of CPUs in the system.
    Use `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ to override:

    >>> import threadpoolctl
    >>>
    >>> with threadpoolctl.threadpool_limits(limits=2):
    ...     m = mmread(StringIO(text))

    """
    # 获取读取数据的游标和需要关闭的流
    cursor, stream_to_close = _get_read_cursor(source)
    # 检查游标的头部格式是否为 "array"
    if cursor.header.format == "array":
        # 如果是数组格式，则调用函数 _read_body_array() 读取数据
        mat = _read_body_array(cursor)
        # 如果有需要关闭的流对象，则关闭之
        if stream_to_close:
            stream_to_close.close()
        # 返回读取的数组数据
        return mat
    else:
        # 如果头部格式不是 "array"，则导入稀疏矩阵的库
        from scipy.sparse import coo_matrix
        # 调用函数 _read_body_coo() 读取 COO 格式的数据，允许对称性泛化
        triplet, shape = _read_body_coo(cursor, generalize_symmetry=True)
        # 如果有需要关闭的流对象，则关闭之
        if stream_to_close:
            stream_to_close.close()
        # 使用读取的数据创建 COO 矩阵对象，并返回
        return coo_matrix(triplet, shape=shape)
# 将稀疏或密集的二维数组 `a` 写入到 Matrix Market 类型的文件 `target` 中
def mmwrite(target, a, comment=None, field=None, precision=None, symmetry="AUTO"):
    """
    Writes the sparse or dense array `a` to Matrix Market file-like `target`.

    Parameters
    ----------
    target : str or file-like
        Matrix Market 文件名（扩展名为 .mtx）或者已打开的文件对象。
    a : array like
        稀疏或密集的二维数组。
    comment : str, optional
        添加到 Matrix Market 文件开头的注释。
    field : None or str, optional
        可选的值为 'real', 'complex', 'pattern', 或者 'integer'。
    precision : None or int, optional
        显示实数或复数值的位数。
    symmetry : None or str, optional
        可选的值为 'AUTO', 'general', 'symmetric', 'skew-symmetric', 或 'hermitian'。
        如果 symmetry 为 None，则根据数组 `a` 的值确定对称性类型。
        如果 symmetry 为 'AUTO'，则由 mmwrite 自行决定或设定数组 `a` 的对称性类型为 'general'。

    Returns
    -------
    None

    Notes
    -----
    .. versionchanged:: 1.12.0
        C++ implementation.

    Examples
    --------
    >>> from io import BytesIO
    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix
    >>> from scipy.io import mmwrite

    将一个小的 NumPy 数组写入到 Matrix Market 文件中，文件格式为 ``'array'``。

    >>> a = np.array([[1.0, 0, 0, 0], [0, 2.5, 0, 6.25]])
    >>> target = BytesIO()
    >>> mmwrite(target, a)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    2 4
    1
    0
    0
    2.5
    0
    0
    0
    6.25

    添加一个注释到输出文件，并设置精度为 3。

    >>> target = BytesIO()
    >>> mmwrite(target, a, comment='\n Some test data.\n', precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    % Some test data.
    %
    2 4
    1.00e+00
    0.00e+00
    0.00e+00
    2.50e+00
    0.00e+00
    0.00e+00
    0.00e+00
    6.25e+00

    在调用 `mmwrite` 前将数组转换为稀疏矩阵。这会导致输出格式为 ``'coordinate'`` 而不是 ``'array'``。

    >>> target = BytesIO()
    >>> mmwrite(target, coo_matrix(a), precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix coordinate real general
    %
    2 4 3
    1 1 1.00e+00
    2 2 2.50e+00
    2 4 6.25e+00

    将一个复数的 Hermitian 数组写入到 Matrix Market 文件中。请注意，实际写入文件的只有六个值；其余的值由对称性推导得出。

    >>> z = np.array([[3, 1+2j, 4-3j], [1-2j, 1, -5j], [4+3j, 5j, 2.5]])
    >>> z
    array([[ 3. +0.j,  1. +2.j,  4. -3.j],
           [ 1. -2.j,  1. +0.j, -0. -5.j],
           [ 4. +3.j,  0. +5.j,  2.5+0.j]])

    >>> target = BytesIO()
    >>> mmwrite(target, z, precision=2)
    >>> print(target.getvalue().decode('latin1'))

"""
    %%MatrixMarket matrix array complex hermitian
    %
    3 3
    3.0e+00 0.0e+00
    1.0e+00 -2.0e+00
    4.0e+00 3.0e+00
    1.0e+00 0.0e+00
    0.0e+00 5.0e+00
    2.5e+00 0.0e+00
    
    This method is threaded.
    The default number of threads is equal to the number of CPUs in the system.
    Use `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ to override:
    
    >>> import threadpoolctl
    >>>
    >>> target = BytesIO()
    >>> with threadpoolctl.threadpool_limits(limits=2):
    ...     mmwrite(target, a)
    
    """
    # 导入本地模块 `_fmm_core`
    from . import _fmm_core
    
    # 如果 `a` 是列表、元组或者有 `__array__` 属性，则转换为 NumPy 数组
    if isinstance(a, list) or isinstance(a, tuple) or hasattr(a, "__array__"):
        a = np.asarray(a)
    
    # 如果 `symmetry` 参数为 "AUTO"
    if symmetry == "AUTO":
        # 如果总是查找对称性或者数组形状的最大值小于 100，则设为无对称性
        if ALWAYS_FIND_SYMMETRY or (hasattr(a, "shape") and max(a.shape) < 100):
            symmetry = None
        else:
            # 否则设为一般对称性
            symmetry = "general"
    
    # 如果 `symmetry` 参数为 `None`，通过 `_mmio.MMFile()._get_symmetry(a)` 获取对称性
    if symmetry is None:
        symmetry = _mmio.MMFile()._get_symmetry(a)
    
    # 确保对称性合法性
    symmetry = _validate_symmetry(symmetry)
    
    # 获取写入游标，包括评论、精度、对称性等参数
    cursor = _get_write_cursor(target, comment=comment,
                               precision=precision, symmetry=symmetry)
    
    # 如果 `a` 是 NumPy 数组
    if isinstance(a, np.ndarray):
        # 写入密集的 NumPy 数组
        a = _apply_field(a, field, no_pattern=True)
        _fmm_core.write_body_array(cursor, a)
    
    # 如果 `a` 是稀疏的 SciPy 矩阵
    elif scipy.sparse.issparse(a):
        # 转换为 COO 格式
        a = a.tocoo()
    
        # 如果指定了对称性且不是一般对称性
        if symmetry is not None and symmetry != "general":
            # 对称矩阵只指定对角线以下的元素，确保矩阵满足此要求
            from scipy.sparse import coo_array
            lower_triangle_mask = a.row >= a.col
            a = coo_array((a.data[lower_triangle_mask],
                          (a.row[lower_triangle_mask],
                           a.col[lower_triangle_mask])), shape=a.shape)
    
        # 应用字段，并写入 COO 格式的数据
        data = _apply_field(a.data, field)
        _fmm_core.write_body_coo(cursor, a.shape, a.row, a.col, data)
    
    else:
        # 抛出未知矩阵类型的异常
        raise ValueError("unknown matrix type: %s" % type(a))
# 返回 Matrix Market 文件或类似对象的大小和存储参数信息

def mminfo(source):
    """
    Return size and storage parameters from Matrix Market file-like 'source'.
    
    Parameters
    ----------
    source : str or file-like
        Matrix Market 文件名（扩展名为 .mtx）或打开的类文件对象

    Returns
    -------
    rows : int
        矩阵的行数。
    cols : int
        矩阵的列数。
    entries : int
        稀疏矩阵的非零条目数，或者对于密集矩阵是 rows * cols。
    format : str
        数据格式，可以是 'coordinate' 或 'array'。
    field : str
        数据域，可以是 'real', 'complex', 'pattern', 或 'integer'。
    symmetry : str
        对称性，可以是 'general', 'symmetric', 'skew-symmetric', 或 'hermitian'。

    Notes
    -----
    .. versionchanged:: 1.12.0
        使用 C++ 实现。

    Examples
    --------
    >>> from io import StringIO
    >>> from scipy.io import mminfo

    >>> text = '''%%MatrixMarket matrix coordinate real general
    ...  5 5 7
    ...  2 3 1.0
    ...  3 4 2.0
    ...  3 5 3.0
    ...  4 1 4.0
    ...  4 2 5.0
    ...  4 3 6.0
    ...  4 4 7.0
    ... '''

    ``mminfo(source)`` 返回源文件的行数、列数、格式、数据域和对称性属性。

    >>> mminfo(StringIO(text))
    (5, 5, 7, 'coordinate', 'real', 'general')
    """
    # 获得读取光标和需关闭的流对象
    cursor, stream_to_close = _get_read_cursor(source, 1)
    # 获取头部信息
    h = cursor.header
    # 关闭光标
    cursor.close()
    # 如果需要关闭流对象，则关闭之
    if stream_to_close:
        stream_to_close.close()
    # 返回头部的行数、列数、非零条目数、数据格式、数据域和对称性属性
    return h.nrows, h.ncols, h.nnz, h.format, h.field, h.symmetry
```