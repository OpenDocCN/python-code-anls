# `D:\src\scipysrc\scipy\scipy\io\_harwell_boeing\hb.py`

```
"""
Implementation of Harwell-Boeing read/write.

At the moment not the full Harwell-Boeing format is supported. Supported
features are:

    - assembled, non-symmetric, real matrices
    - integer for pointer/indices
    - exponential format for float values, and int format

"""
# TODO:
#   - Add more support (symmetric/complex matrices, non-assembled matrices ?)

# XXX: reading is reasonably efficient (>= 85 % is in numpy.fromstring), but
# takes a lot of memory. Being faster would require compiled code.
# write is not efficient. Although not a terribly exciting task,
# having reusable facilities to efficiently read/write fortran-formatted files
# would be useful outside this module.

# 导入警告模块
import warnings

# 导入 numpy 库并将其命名为 np
import numpy as np
# 导入 scipy.sparse 模块中的 csc_matrix 类
from scipy.sparse import csc_matrix
# 导入 _fortran_format_parser 模块中的 FortranFormatParser, IntFormat, ExpFormat 类
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat

# 导出的模块内容列表
__all__ = ["hb_read", "hb_write"]


# 自定义异常类 MalformedHeader，用于处理格式错误的头部信息
class MalformedHeader(Exception):
    pass


# 自定义警告类 LineOverflow，用于处理行溢出警告
class LineOverflow(Warning):
    pass


# 定义函数 _nbytes_full，计算按给定的解析后的 Fortran 格式读取全部完整行所需的字节数
def _nbytes_full(fmt, nlines):
    """Return the number of bytes to read to get every full lines for the
    given parsed fortran format."""
    return (fmt.repeat * fmt.width + 1) * (nlines - 1)


# 定义类 HBInfo，用于处理 Harwell-Boeing 格式的信息
class HBInfo:
    @classmethod
    def from_data(cls, m, title="Default title", key="0", mxtype=None, fmt=None):
        """Create a HBInfo instance from an existing sparse matrix.

        Parameters
        ----------
        m : sparse matrix
            the HBInfo instance will derive its parameters from m
        title : str
            Title to put in the HB header
        key : str
            Key
        mxtype : HBMatrixType
            type of the input matrix
        fmt : dict
            not implemented

        Returns
        -------
        hb_info : HBInfo instance
        """
        # Convert the sparse matrix 'm' to Compressed Sparse Column format
        m = m.tocsc(copy=False)

        # Extract necessary attributes from the sparse matrix 'm'
        pointer = m.indptr   # Array of indices into the indices and data arrays
        indices = m.indices  # Array of column indices for each non-zero element
        values = m.data      # Array of non-zero elements in the matrix

        # Get dimensions and other properties from the sparse matrix 'm'
        nrows, ncols = m.shape     # Number of rows and columns
        nnon_zeros = m.nnz         # Number of non-zero elements

        if fmt is None:
            # Determine the appropriate format for pointer indices
            # +1 because HB uses one-based indexing (Fortran)
            pointer_fmt = IntFormat.from_number(np.max(pointer+1))
            # Determine the appropriate format for indices
            indices_fmt = IntFormat.from_number(np.max(indices+1))

            # Determine the appropriate format for values based on dtype
            if values.dtype.kind in np.typecodes["AllFloat"]:
                values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
            elif values.dtype.kind in np.typecodes["AllInteger"]:
                values_fmt = IntFormat.from_number(-np.max(np.abs(values)))
            else:
                message = f"type {values.dtype.kind} not implemented yet"
                raise NotImplementedError(message)
        else:
            # Raise an error if the 'fmt' argument is provided (not implemented)
            raise NotImplementedError("fmt argument not supported yet.")

        if mxtype is None:
            # Determine matrix type if not provided
            if not np.isrealobj(values):
                raise ValueError("Complex values not supported yet")
            if values.dtype.kind in np.typecodes["AllInteger"]:
                tp = "integer"
            elif values.dtype.kind in np.typecodes["AllFloat"]:
                tp = "real"
            else:
                raise NotImplementedError("type %s for values not implemented"
                                          % values.dtype)
            # Set default mxtype based on determined type
            mxtype = HBMatrixType(tp, "unsymmetric", "assembled")
        else:
            # Raise an error if 'mxtype' argument is provided (not handled yet)
            raise ValueError("mxtype argument not handled yet.")

        # Function to calculate number of lines needed for a given format
        def _nlines(fmt, size):
            nlines = size // fmt.repeat
            if nlines * fmt.repeat != size:
                nlines += 1
            return nlines

        # Calculate number of lines needed for each format
        pointer_nlines = _nlines(pointer_fmt, pointer.size)
        indices_nlines = _nlines(indices_fmt, indices.size)
        values_nlines = _nlines(values_fmt, values.size)

        # Calculate total number of lines needed
        total_nlines = pointer_nlines + indices_nlines + values_nlines

        # Return an instance of HBInfo with calculated parameters
        return cls(title, key,
            total_nlines, pointer_nlines, indices_nlines, values_nlines,
            mxtype, nrows, ncols, nnon_zeros,
            pointer_fmt.fortran_format, indices_fmt.fortran_format,
            values_fmt.fortran_format)

    @classmethod
    def dump(self):
        """返回此实例对应的头部信息字符串。"""
        # 创建一个列表，包含标题和键的左对齐字符串，总共长度为80个字符
        header = [self.title.ljust(72) + self.key.ljust(8)]

        # 添加包含总行数、指针行数、索引行数和值行数的格式化字符串
        header.append("%14d%14d%14d%14d" %
                      (self.total_nlines, self.pointer_nlines,
                       self.indices_nlines, self.values_nlines))

        # 添加包含最大类型的格式化字符串，左对齐14个字符，以及行数、列数、非零元素数和一个空值
        header.append("%14s%14d%14d%14d%14d" %
                      (self.mxtype.fortran_format.ljust(14), self.nrows,
                       self.ncols, self.nnon_zeros, 0))

        # 获取指针格式、索引格式和值格式的Fortran格式化字符串，每个左对齐16个字符和20个字符
        pffmt = self.pointer_format.fortran_format
        iffmt = self.indices_format.fortran_format
        vffmt = self.values_format.fortran_format

        # 添加指针、索引和值的格式化字符串，每个左对齐16个字符和20个字符
        header.append("%16s%16s%20s" %
                      (pffmt.ljust(16), iffmt.ljust(16), vffmt.ljust(20)))

        # 将列表中的所有字符串连接成一个以换行符分隔的字符串，并返回
        return "\n".join(header)
# 定义一个内部函数，用于将输入的值转换为整数
def _expect_int(value, msg=None):
    try:
        # 尝试将输入的值转换为整数并返回
        return int(value)
    except ValueError as e:
        # 如果转换失败，则抛出 ValueError 异常，并根据情况提供自定义消息
        if msg is None:
            msg = "Expected an int, got %s"
        raise ValueError(msg % value) from e


# 定义一个内部函数，用于从给定内容中读取稀疏矩阵数据并返回压缩稀疏列 (CSC) 格式的矩阵对象
def _read_hb_data(content, header):
    # XXX: 这里可以考虑减少内存使用（字符串拼接）
    
    # 读取指针数据并形成字符串
    ptr_string = "".join([content.read(header.pointer_nbytes_full),
                           content.readline()])
    # 将指针数据转换为 numpy 数组
    ptr = np.fromstring(ptr_string,
            dtype=int, sep=' ')

    # 读取索引数据并形成字符串
    ind_string = "".join([content.read(header.indices_nbytes_full),
                       content.readline()])
    # 将索引数据转换为 numpy 数组
    ind = np.fromstring(ind_string,
            dtype=int, sep=' ')

    # 读取数值数据并形成字符串
    val_string = "".join([content.read(header.values_nbytes_full),
                          content.readline()])
    # 将数值数据转换为 numpy 数组，使用指定的数据类型
    val = np.fromstring(val_string,
            dtype=header.values_dtype, sep=' ')

    try:
        # 使用 CSC 矩阵格式创建稀疏矩阵对象，并返回
        return csc_matrix((val, ind-1, ptr-1),
                          shape=(header.nrows, header.ncols))
    except ValueError as e:
        # 如果出现异常，则重新抛出该异常
        raise e


# 定义一个内部函数，用于将稀疏矩阵数据写入指定的文件对象中
def _write_data(m, fid, header):
    # 将输入的稀疏矩阵对象转换为压缩稀疏列格式，如果可能，不进行复制
    m = m.tocsc(copy=False)

    # 定义一个内部函数，用于将数组数据按照指定的格式写入文件中
    def write_array(f, ar, nlines, fmt):
        # ar_nlines 表示完整行数，n 是每行的项目数，ffmt 是 Fortran 格式

        # 转换为 Python 的格式
        pyfmt = fmt.python_format
        # 将格式扩展到整行数据
        pyfmt_full = pyfmt * fmt.repeat

        # 对每个数组写入，首先写入完整行，最后处理部分行
        full = ar[:(nlines - 1) * fmt.repeat]
        for row in full.reshape((nlines-1, fmt.repeat)):
            f.write(pyfmt_full % tuple(row) + "\n")
        nremain = ar.size - full.size
        if nremain > 0:
            f.write((pyfmt * nremain) % tuple(ar[ar.size - nremain:]) + "\n")

    # 将头部信息写入文件
    fid.write(header.dump())
    fid.write("\n")
    # 对指针、索引和数值数组分别调用写入函数，进行写入操作
    # +1 是因为 Fortran 使用基于 1 的索引
    write_array(fid, m.indptr+1, header.pointer_nlines,
                header.pointer_format)
    write_array(fid, m.indices+1, header.indices_nlines,
                header.indices_format)
    write_array(fid, m.data, header.values_nlines,
                header.values_format)


class HBMatrixType:
    """用于保存矩阵类型信息的类。"""
    # q2f* 将限定名称翻译为 Fortran 字符
    _q2f_type = {
        "real": "R",
        "complex": "C",
        "pattern": "P",
        "integer": "I",
    }
    # 将限定结构名称翻译为 Fortran 字符
    _q2f_structure = {
            "symmetric": "S",
            "unsymmetric": "U",
            "hermitian": "H",
            "skewsymmetric": "Z",
            "rectangular": "R"
    }
    # 将存储方式翻译为 Fortran 字符
    _q2f_storage = {
        "assembled": "A",
        "elemental": "E",
    }

    # 反向映射，将 Fortran 字符转换回限定名称
    _f2q_type = {j: i for i, j in _q2f_type.items()}
    _f2q_structure = {j: i for i, j in _q2f_structure.items()}
    _f2q_storage = {j: i for i, j in _q2f_storage.items()}

    @classmethod
    # 根据 Fortran 格式字符串创建 HBMatrixType 类的实例
    def from_fortran(cls, fmt):
        # 检查格式字符串长度是否为 3，若不是则抛出异常
        if not len(fmt) == 3:
            raise ValueError("Fortran format for matrix type should be 3 "
                             "characters long")
        try:
            # 根据格式字符串中的字符获取对应的值类型、结构和存储方式
            value_type = cls._f2q_type[fmt[0]]
            structure = cls._f2q_structure[fmt[1]]
            storage = cls._f2q_storage[fmt[2]]
            # 使用获取的值类型、结构和存储方式创建 HBMatrixType 实例并返回
            return cls(value_type, structure, storage)
        except KeyError as e:
            # 若发生 KeyError，则表示格式字符串中有未识别的字符，抛出相应异常
            raise ValueError("Unrecognized format %s" % fmt) from e

    # 初始化 HBMatrixType 类的实例
    def __init__(self, value_type, structure, storage="assembled"):
        self.value_type = value_type
        self.structure = structure
        self.storage = storage

        # 检查值类型、结构和存储方式是否在预定义的映射中，若不在则抛出相应异常
        if value_type not in self._q2f_type:
            raise ValueError("Unrecognized type %s" % value_type)
        if structure not in self._q2f_structure:
            raise ValueError("Unrecognized structure %s" % structure)
        if storage not in self._q2f_storage:
            raise ValueError("Unrecognized storage %s" % storage)

    # 属性装饰器，返回当前实例的 Fortran 格式字符串表示
    @property
    def fortran_format(self):
        return self._q2f_type[self.value_type] + \
               self._q2f_structure[self.structure] + \
               self._q2f_storage[self.storage]

    # 返回实例的字符串表示形式，用于打印对象时的显示
    def __repr__(self):
        return f"HBMatrixType({self.value_type}, {self.structure}, {self.storage})"
class HBFile:
    def __init__(self, file, hb_info=None):
        """Create a HBFile instance.

        Parameters
        ----------
        file : file-object
            File object representing the Harwell-Boeing file.
            StringIO works as well.
        hb_info : HBInfo, optional
            Optional meta-data for the HB file. If not provided, it is inferred
            from the file.

        Notes
        -----
        If hb_info is provided, the file should be writable.
        """
        self._fid = file
        if hb_info is None:
            # If no hb_info is provided, infer it from the file
            self._hb_info = HBInfo.from_file(file)
        else:
            # If hb_info is provided, use it directly
            self._hb_info = hb_info

    @property
    def title(self):
        """Property: Title of the HBFile."""
        return self._hb_info.title

    @property
    def key(self):
        """Property: Key of the HBFile."""
        return self._hb_info.key

    @property
    def type(self):
        """Property: Type of the HBFile (value type)."""
        return self._hb_info.mxtype.value_type

    @property
    def structure(self):
        """Property: Structure of the HBFile (matrix structure)."""
        return self._hb_info.mxtype.structure

    @property
    def storage(self):
        """Property: Storage of the HBFile (matrix storage)."""
        return self._hb_info.mxtype.storage

    def read_matrix(self):
        """Read matrix data from the HBFile."""
        return _read_hb_data(self._fid, self._hb_info)

    def write_matrix(self, m):
        """Write matrix data to the HBFile.

        Parameters
        ----------
        m : sparse-matrix
            The sparse matrix to be written.
        """
        return _write_data(m, self._fid, self._hb_info)


def hb_read(path_or_open_file):
    """Read HB-format file.

    Parameters
    ----------
    path_or_open_file : path-like or file-like
        If a file-like object, it is used as-is. Otherwise, it is opened
        before reading.

    Returns
    -------
    data : scipy.sparse.csc_matrix instance
        The data read from the HB file as a sparse matrix.

    Notes
    -----
    At the moment not the full Harwell-Boeing format is supported. Supported
    features are:

        - assembled, non-symmetric, real matrices
        - integer for pointer/indices
        - exponential format for float values, and int format

    Examples
    --------
    We can read and write a harwell-boeing format file:

    >>> from scipy.io import hb_read, hb_write
    >>> from scipy.sparse import csr_array, eye
    >>> data = csr_array(eye(3))  # create a sparse array
    >>> hb_write("data.hb", data)  # write a hb file
    >>> print(hb_read("data.hb"))  # read a hb file
    (np.int32(0), np.int32(0))    1.0
    (np.int32(1), np.int32(1))    1.0
    (np.int32(2), np.int32(2))    1.0
    """
    def _get_matrix(fid):
        """Internal function to get matrix from HBFile."""
        hb = HBFile(fid)
        return hb.read_matrix()

    if hasattr(path_or_open_file, 'read'):
        # If path_or_open_file is already a file-like object, use it directly
        return _get_matrix(path_or_open_file)
    else:
        # If path_or_open_file is a path-like object, open it and then read
        with open(path_or_open_file) as f:
            return _get_matrix(f)


def hb_write(path_or_open_file, m, hb_info=None):
    """Write HB-format file.

    Parameters
    ----------
    path_or_open_file : path-like or file-like
        If a file-like object, it is used as-is. Otherwise, it is opened
        before writing.
    m : sparse-matrix
        The sparse matrix to write to the HB file.
    hb_info : HBInfo, optional
        Meta-data for the HB file.

    Returns
    -------
    None

    Notes
    -----

    """
    m = m.tocsc(copy=False)

将稀疏矩阵 `m` 转换为压缩列格式 (CSC)，并且在转换过程中不进行复制。


    if hb_info is None:
        hb_info = HBInfo.from_data(m)

如果 `hb_info` 为空，则通过 `HBInfo.from_data(m)` 创建一个 `HBInfo` 对象，其中包含矩阵 `m` 的信息。


    def _set_matrix(fid):
        hb = HBFile(fid, hb_info)
        return hb.write_matrix(m)

定义内部函数 `_set_matrix(fid)`，接受文件标识符 `fid` 作为参数。创建 `HBFile` 对象 `hb`，使用给定的 `hb_info` 和文件标识符来初始化。然后调用 `hb.write_matrix(m)` 将矩阵 `m` 写入到 `HBFile` 对象 `hb` 所代表的文件中。


    if hasattr(path_or_open_file, 'write'):
        return _set_matrix(path_or_open_file)

如果 `path_or_open_file` 具有 `write` 属性，说明它是一个可写的文件对象，直接调用 `_set_matrix(path_or_open_file)` 将矩阵写入该文件对象中，并返回结果。


    else:
        with open(path_or_open_file, 'w') as f:
            return _set_matrix(f)

否则，假设 `path_or_open_file` 是一个文件路径，以写入模式打开该文件 `f`，然后调用 `_set_matrix(f)` 将矩阵写入文件 `f` 中，并返回结果。
```