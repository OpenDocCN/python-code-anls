# `D:\src\scipysrc\scipy\scipy\io\_fortran.py`

```
"""
Module to read / write Fortran unformatted sequential files.

This is in the spirit of code written by Neil Martinsen-Burrell and Joe Zuntz.

"""
# 引入警告模块，用于可能的警告通知
import warnings
# 引入 NumPy 库，用于处理数值和数组操作
import numpy as np

# 定义模块中公开的类和异常
__all__ = ['FortranFile', 'FortranEOFError', 'FortranFormattingError']


class FortranEOFError(TypeError, OSError):
    """Indicates that the file ended properly.

    This error descends from TypeError because the code used to raise
    TypeError (and this was the only way to know that the file had
    ended) so users might have ``except TypeError:``.

    """
    pass


class FortranFormattingError(TypeError, OSError):
    """Indicates that the file ended mid-record.

    Descends from TypeError for backward compatibility.

    """
    pass


class FortranFile:
    """
    A file object for unformatted sequential files from Fortran code.

    Parameters
    ----------
    filename : file or str
        Open file object or filename.
    mode : {'r', 'w'}, optional
        Read-write mode, default is 'r'.
    header_dtype : dtype, optional
        Data type of the header. Size and endianness must match the input/output file.

    Notes
    -----
    These files are broken up into records of unspecified types. The size of
    each record is given at the start (although the size of this header is not
    standard) and the data is written onto disk without any formatting. Fortran
    compilers supporting the BACKSPACE statement will write a second copy of
    the size to facilitate backwards seeking.

    This class only supports files written with both sizes for the record.
    It also does not support the subrecords used in Intel and gfortran compilers
    for records which are greater than 2GB with a 4-byte header.

    An example of an unformatted sequential file in Fortran would be written as::

        OPEN(1, FILE=myfilename, FORM='unformatted')

        WRITE(1) myvariable

    Since this is a non-standard file format, whose contents depend on the
    compiler and the endianness of the machine, caution is advised. Files from
    gfortran 4.8.0 and gfortran 4.1.2 on x86_64 are known to work.

    Consider using Fortran direct-access files or files from the newer Stream
    I/O, which can be easily read by `numpy.fromfile`.

    Examples
    --------
    To create an unformatted sequential Fortran file:

    >>> from scipy.io import FortranFile
    >>> import numpy as np
    >>> f = FortranFile('test.unf', 'w')
    >>> f.write_record(np.array([1,2,3,4,5], dtype=np.int32))
    >>> f.write_record(np.linspace(0,1,20).reshape((5,4)).T)
    >>> f.close()

    To read this file:

    >>> f = FortranFile('test.unf', 'r')
    >>> print(f.read_ints(np.int32))
    [1 2 3 4 5]
    >>> print(f.read_reals(float).reshape((5,4), order="F"))
    """

    def __init__(self, filename, mode='r', header_dtype=None):
        # 初始化方法，接受文件名或文件对象以及读写模式和头部数据类型
        pass

    def close(self):
        # 关闭文件对象的方法
        pass

    def read_ints(self, dtype):
        # 读取整数数组的方法，根据给定的数据类型
        pass

    def read_reals(self, dtype):
        # 读取实数数组的方法，根据给定的数据类型
        pass

    def write_record(self, data):
        # 写入记录的方法，根据给定的数据
        pass
    """
    Initialize a FortranFile object to read/write Fortran-style binary files.
    
    Parameters:
    - filename : str or file-like object
        The name of the file or a file-like object to read from/write to.
    - mode : str, optional
        The mode to open the file in ('r' for read, 'w' for write).
    - header_dtype : dtype, optional
        The data type of the header, defaults to np.uint32.
    
    Raises:
    - ValueError
        If header_dtype is None or mode is invalid.
    - UserWarning
        If header_dtype is not unsigned.
    - TypeError
        If filename is not a valid file-like object or string.
    
    Notes:
    - Initializes a FortranFile object to handle Fortran-style binary files.
    - Checks and sets the header data type.
    - Handles file opening based on the mode provided.
    """
    def __init__(self, filename, mode='r', header_dtype=np.uint32):
        if header_dtype is None:
            raise ValueError('Must specify dtype')
    
        header_dtype = np.dtype(header_dtype)
        if header_dtype.kind != 'u':
            warnings.warn("Given a dtype which is not unsigned.", stacklevel=2)
    
        if mode not in 'rw' or len(mode) != 1:
            raise ValueError('mode must be either r or w')
    
        if hasattr(filename, 'seek'):
            self._fp = filename
        else:
            self._fp = open(filename, '%sb' % mode)
    
        self._header_dtype = header_dtype
    
    
    """
    Read the size of the next record from the file.
    
    Parameters:
    - eof_ok : bool, optional
        Flag indicating if end-of-file is acceptable.
    
    Returns:
    - n : int
        The size of the next record.
    
    Raises:
    - FortranEOFError
        If end-of-file occurs at the end of the record and eof_ok is False.
    - FortranFormattingError
        If end-of-file occurs in the middle of the record size.
    
    Notes:
    - Reads the size of the next record using the header data type.
    - Handles end-of-file conditions based on the eof_ok flag.
    """
    def _read_size(self, eof_ok=False):
        n = self._header_dtype.itemsize
        b = self._fp.read(n)
        if (not b) and eof_ok:
            raise FortranEOFError("End of file occurred at end of record")
        elif len(b) < n:
            raise FortranFormattingError(
                "End of file in the middle of the record size")
        return int(np.frombuffer(b, dtype=self._header_dtype, count=1)[0])
    
    
    """
    Write a record (including sizes) to the file.
    
    Parameters:
    - *items : array_like
        The data arrays to write.
    
    Notes:
    - Writes data items to a Fortran-style binary file.
    - Uses the header data type to determine the size of the records.
    - Formats the data correctly for Fortran programs.
    """
    def write_record(self, *items):
        items = tuple(np.asarray(item) for item in items)
        total_size = sum(item.nbytes for item in items)
    
        nb = np.array([total_size], dtype=self._header_dtype)
    
        nb.tofile(self._fp)
        for item in items:
            item.tofile(self._fp)
        nb.tofile(self._fp)
    
    
    """
    Reads a record of a given type from the file, defaulting to an integer type.
    
    Parameters:
    - dtype : dtype, optional
        Data type specifying the size and endianness of the data.
    
    Returns:
    - data : ndarray
        A 1-D array object containing the read data.
    
    See Also:
    - read_reals
    - read_record
    
    Notes:
    - Reads a record of data from a Fortran-style binary file.
    - Supports reading different data types by specifying the dtype parameter.
    """
    def read_ints(self, dtype='i4'):
        return self.read_record(dtype)
    # 从文件中读取给定类型的记录，默认为浮点数（Fortran 中的 real*8）。
    def read_reals(self, dtype='f8'):
        """
        Reads a record of a given type from the file, defaulting to a floating
        point number (``real*8`` in Fortran).

        Parameters
        ----------
        dtype : dtype, optional
            Data type specifying the size and endianness of the data.

        Returns
        -------
        data : ndarray
            A 1-D array object.

        See Also
        --------
        read_ints
        read_record

        """
        # 调用 read_record 方法读取数据并返回
        return self.read_record(dtype)

    # 关闭文件。在关闭后，调用此对象的其它方法是不支持的。
    def close(self):
        """
        Closes the file. It is unsupported to call any other methods off this
        object after closing it. Note that this class supports the 'with'
        statement in modern versions of Python, to call this automatically

        """
        # 调用文件对象的 _fp 属性的 close 方法关闭文件
        self._fp.close()

    # 实现上下文管理器的 __enter__ 方法
    def __enter__(self):
        return self

    # 实现上下文管理器的 __exit__ 方法
    def __exit__(self, type, value, tb):
        # 调用 close 方法关闭文件
        self.close()
```