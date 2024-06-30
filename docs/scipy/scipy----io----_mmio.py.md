# `D:\src\scipysrc\scipy\scipy\io\_mmio.py`

```
"""
  Matrix Market I/O in Python.
  See http://math.nist.gov/MatrixMarket/formats.html
  for information about the Matrix Market format.
"""
#
# Author: Pearu Peterson <pearu@cens.ioc.ee>
# Created: October, 2004
#
# References:
#  http://math.nist.gov/MatrixMarket/
#
import os

import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
                   ones, can_cast)

from scipy.sparse import coo_matrix, issparse

__all__ = ['mminfo', 'mmread', 'mmwrite', 'MMFile']


# -----------------------------------------------------------------------------
def asstr(s):
    # 将输入的字节流或字符串转换为字符串（使用 latin1 编码）
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)


def mminfo(source):
    """
    Return size and storage parameters from Matrix Market file-like 'source'.

    Parameters
    ----------
    source : str or file-like
        Matrix Market filename (extension .mtx) or open file-like object

    Returns
    -------
    rows : int
        Number of matrix rows.
    cols : int
        Number of matrix columns.
    entries : int
        Number of non-zero entries of a sparse matrix
        or rows*cols for a dense matrix.
    format : str
        Either 'coordinate' or 'array'.
    field : str
        Either 'real', 'complex', 'pattern', or 'integer'.
    symmetry : str
        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.

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


    ``mminfo(source)`` returns the number of rows, number of columns,
    format, field type and symmetry attribute of the source file.

    >>> mminfo(StringIO(text))
    (5, 5, 7, 'coordinate', 'real', 'general')
    """
    return MMFile.info(source)

# -----------------------------------------------------------------------------


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
    <5x5 sparse matrix of type '<class 'numpy.float64'>'
    with 7 stored elements in COOrdinate format>
    >>> m.A
    # 创建一个二维数组，表示一个稀疏矩阵，包含了特定的数值
    array([[0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 2., 3.],
           [4., 5., 6., 7., 0.],
           [0., 0., 0., 0., 0.]])
    """
    # 调用 MMFile 类的实例化对象 MMFile()，并调用其 read 方法读取数据源的内容
    return MMFile().read(source)
# -----------------------------------------------------------------------------
# 定义函数 mmwrite，用于将稀疏或密集的二维数组 a 写入 Matrix Market 文件中
# target : str or file-like
#     Matrix Market 文件名（扩展名为 .mtx）或者已打开的文件对象
# a : array like
#     稀疏或密集的二维数组
# comment : str, optional
#     要添加到 Matrix Market 文件开头的注释信息
# field : None or str, optional
#     数据类型，可以是 'real', 'complex', 'pattern', 或 'integer'
# precision : None or int, optional
#     显示实数或复数值的小数位数
# symmetry : None or str, optional
#     矩阵的对称性，可以是 'general', 'symmetric', 'skew-symmetric', 或 'hermitian'
#     如果 symmetry 为 None，则根据数组 a 的值确定对称性类型
def mmwrite(target, a, comment='', field=None, precision=None, symmetry=None):
    r"""
    将稀疏或密集数组 `a` 写入 Matrix Market 文件样式的 `target`。

    参数
    ----------
    target : str or file-like
        Matrix Market 文件名（扩展名 .mtx）或者已打开的文件对象。
    a : array like
        稀疏或密集的二维数组。
    comment : str, optional
        要添加到 Matrix Market 文件开头的注释。
    field : None or str, optional
        数据类型，可以是 'real', 'complex', 'pattern' 或 'integer'。
    precision : None or int, optional
        实数或复数值的显示小数位数。
    symmetry : None or str, optional
        矩阵的对称性，可以是 'general', 'symmetric', 'skew-symmetric' 或 'hermitian'。
        如果 symmetry 为 None，则根据数组 a 的值确定对称性类型。

    返回
    -------
    None
    ```
    %%MatrixMarket matrix array complex hermitian
    %
    # MatrixMarket 格式声明，表示接下来的数据是复数埃尔米特矩阵
    3 3
    # 3行3列的矩阵
    3.00e+00 0.00e+00
    # 第一行第一列元素为 3.00+0.00i
    1.00e+00 -2.00e+00
    # 第一行第二列元素为 1.00-2.00i
    4.00e+00 3.00e+00
    # 第一行第三列元素为 4.00+3.00i
    1.00e+00 0.00e+00
    # 第二行第一列元素为 1.00+0.00i
    0.00e+00 5.00e+00
    # 第二行第二列元素为 0.00+5.00i
    2.50e+00 0.00e+00
    # 第二行第三列元素为 2.50+0.00i
    
    """
    MMFile().write(target, a, comment, field, precision, symmetry)
    # 这部分是一个多行注释，描述了某个函数 MMFile().write() 的使用方式和参数
###############################################################################
class MMFile:
    __slots__ = ('_rows',
                 '_cols',
                 '_entries',
                 '_format',
                 '_field',
                 '_symmetry')

    @property
    def rows(self):
        # 返回 MMFile 对象的行数属性
        return self._rows

    @property
    def cols(self):
        # 返回 MMFile 对象的列数属性
        return self._cols

    @property
    def entries(self):
        # 返回 MMFile 对象的条目数属性
        return self._entries

    @property
    def format(self):
        # 返回 MMFile 对象的格式属性
        return self._format

    @property
    def field(self):
        # 返回 MMFile 对象的字段类型属性
        return self._field

    @property
    def symmetry(self):
        # 返回 MMFile 对象的对称性属性
        return self._symmetry

    @property
    def has_symmetry(self):
        # 检查 MMFile 对象是否具有对称性
        return self._symmetry in (self.SYMMETRY_SYMMETRIC,
                                  self.SYMMETRY_SKEW_SYMMETRIC,
                                  self.SYMMETRY_HERMITIAN)

    # format values
    FORMAT_COORDINATE = 'coordinate'
    FORMAT_ARRAY = 'array'
    FORMAT_VALUES = (FORMAT_COORDINATE, FORMAT_ARRAY)

    @classmethod
    def _validate_format(self, format):
        # 静态方法：验证给定的格式是否在允许的格式列表中
        if format not in self.FORMAT_VALUES:
            msg = f'unknown format type {format}, must be one of {self.FORMAT_VALUES}'
            raise ValueError(msg)

    # field values
    FIELD_INTEGER = 'integer'
    FIELD_UNSIGNED = 'unsigned-integer'
    FIELD_REAL = 'real'
    FIELD_COMPLEX = 'complex'
    FIELD_PATTERN = 'pattern'
    FIELD_VALUES = (FIELD_INTEGER, FIELD_UNSIGNED, FIELD_REAL, FIELD_COMPLEX,
                    FIELD_PATTERN)

    @classmethod
    def _validate_field(self, field):
        # 静态方法：验证给定的字段类型是否在允许的字段类型列表中
        if field not in self.FIELD_VALUES:
            msg = f'unknown field type {field}, must be one of {self.FIELD_VALUES}'
            raise ValueError(msg)

    # symmetry values
    SYMMETRY_GENERAL = 'general'
    SYMMETRY_SYMMETRIC = 'symmetric'
    SYMMETRY_SKEW_SYMMETRIC = 'skew-symmetric'
    SYMMETRY_HERMITIAN = 'hermitian'
    SYMMETRY_VALUES = (SYMMETRY_GENERAL, SYMMETRY_SYMMETRIC,
                       SYMMETRY_SKEW_SYMMETRIC, SYMMETRY_HERMITIAN)

    @classmethod
    def _validate_symmetry(self, symmetry):
        # 静态方法：验证给定的对称性类型是否在允许的对称性类型列表中
        if symmetry not in self.SYMMETRY_VALUES:
            raise ValueError(f'unknown symmetry type {symmetry}, '
                             f'must be one of {self.SYMMETRY_VALUES}')

    DTYPES_BY_FIELD = {FIELD_INTEGER: 'intp',
                       FIELD_UNSIGNED: 'uint64',
                       FIELD_REAL: 'd',
                       FIELD_COMPLEX: 'D',
                       FIELD_PATTERN: 'd'}

    # -------------------------------------------------------------------------
    @staticmethod
    def reader():
        # 静态方法：用于读取 MMFile 对象的数据
        pass

    # -------------------------------------------------------------------------
    @staticmethod
    def writer():
        # 静态方法：用于将数据写入 MMFile 对象
        pass

    # -------------------------------------------------------------------------
    @classmethod
    def info(self, source):
        """
        Return size, storage parameters from Matrix Market file-like 'source'.

        Parameters
        ----------
        source : str or file-like
            Matrix Market filename (extension .mtx) or open file-like object

        Returns
        -------
        rows : int
            Number of matrix rows.
        cols : int
            Number of matrix columns.
        entries : int
            Number of non-zero entries of a sparse matrix
            or rows*cols for a dense matrix.
        format : str
            Either 'coordinate' or 'array'.
        field : str
            Either 'real', 'complex', 'pattern', or 'integer'.
        symmetry : str
            Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
        """

        # Open the source and prepare to read from it
        stream, close_it = self._open(source)

        try:

            # read and validate header line
            line = stream.readline()
            # Split the line and strip each part, converting to string
            mmid, matrix, format, field, symmetry = \
                (asstr(part.strip()) for part in line.split())
            # Check if the file is in Matrix Market format
            if not mmid.startswith('%%MatrixMarket'):
                raise ValueError('source is not in Matrix Market format')
            # Validate the header line
            if not matrix.lower() == 'matrix':
                raise ValueError("Problem reading file header: " + line)

            # http://math.nist.gov/MatrixMarket/formats.html
            # Determine and normalize the format type
            if format.lower() == 'array':
                format = self.FORMAT_ARRAY
            elif format.lower() == 'coordinate':
                format = self.FORMAT_COORDINATE

            # skip comments
            # Skip lines that start with '%' (comments in Matrix Market format)
            while line:
                if line.lstrip() and line.lstrip()[0] in ['%', 37]:
                    line = stream.readline()
                else:
                    break

            # skip empty lines
            # Skip over any empty lines encountered
            while not line.strip():
                line = stream.readline()

            # Parse the line to get matrix dimensions and entries count
            split_line = line.split()
            if format == self.FORMAT_ARRAY:
                if not len(split_line) == 2:
                    raise ValueError("Header line not of length 2: " +
                                     line.decode('ascii'))
                # Extract rows and columns for array format
                rows, cols = map(int, split_line)
                entries = rows * cols
            else:
                if not len(split_line) == 3:
                    raise ValueError("Header line not of length 3: " +
                                     line.decode('ascii'))
                # Extract rows, columns, and entries for coordinate format
                rows, cols, entries = map(int, split_line)

            # Return the extracted information
            return (rows, cols, entries, format, field.lower(),
                    symmetry.lower())

        finally:
            # Close the stream if it was opened within this function
            if close_it:
                stream.close()

    # -------------------------------------------------------------------------
    @staticmethod
    # 返回一个用于读取的打开文件流，基于给定的源

    # 如果源是文件名，则尝试打开它（在尝试带有mtx和gzipped mtx扩展名的情况下）
    # 否则，直接返回源。

    # Parameters
    # ----------
    # filespec : str or file-like
    #     文件名或类似文件的对象的字符串表示
    # mode : str, optional
    #     打开文件的模式，如果`filespec`是文件名时使用

    # Returns
    # -------
    # fobj : file-like
    #     打开的文件对象
    # close_it : bool
    #     如果调用函数应该在完成时关闭此文件，则为True；否则为false。
    def _open(filespec, mode='rb'):
        """ Return an open file stream for reading based on source.

        If source is a file name, open it (after trying to find it with mtx and
        gzipped mtx extensions). Otherwise, just return source.

        Parameters
        ----------
        filespec : str or file-like
            String giving file name or file-like object
        mode : str, optional
            Mode with which to open file, if `filespec` is a file name.

        Returns
        -------
        fobj : file-like
            Open file-like object.
        close_it : bool
            True if the calling function should close this file when done,
            false otherwise.
        """
        # 如果'filespec'是路径类（str、pathlib.Path、os.DirEntry或实现了'__fspath__'方法的其他类），
        # 尝试将其转换为str。如果抛出'TypeError'异常，则假定它是一个打开的文件句柄，直接返回它。
        try:
            filespec = os.fspath(filespec)
        except TypeError:
            return filespec, False

        # 现在'filespec'肯定是一个str了

        # 以读取模式打开文件
        if mode[0] == 'r':

            # 确定带有扩展名的文件名
            if not os.path.isfile(filespec):
                if os.path.isfile(filespec+'.mtx'):
                    filespec = filespec + '.mtx'
                elif os.path.isfile(filespec+'.mtx.gz'):
                    filespec = filespec + '.mtx.gz'
                elif os.path.isfile(filespec+'.mtx.bz2'):
                    filespec = filespec + '.mtx.bz2'
            # 打开文件名
            if filespec.endswith('.gz'):
                import gzip
                stream = gzip.open(filespec, mode)
            elif filespec.endswith('.bz2'):
                import bz2
                stream = bz2.BZ2File(filespec, 'rb')
            else:
                stream = open(filespec, mode)

        # 以写入模式打开文件
        else:
            if filespec[-4:] != '.mtx':
                filespec = filespec + '.mtx'
            stream = open(filespec, mode)

        return stream, True

    # -------------------------------------------------------------------------
    @staticmethod
    # 定义一个私有方法 `_get_symmetry`，用于确定给定矩阵 `a` 的对称性类型
    def _get_symmetry(a):
        # 获取矩阵的行数和列数
        m, n = a.shape
        # 如果行数不等于列数，则返回一般对称性类型
        if m != n:
            return MMFile.SYMMETRY_GENERAL
        # 初始化对称性类型标志
        issymm = True
        isskew = True
        # 检查是否为复数类型
        isherm = a.dtype.char in 'FD'

        # 处理稀疏输入
        if issparse(a):
            # 转换为COO格式
            a = a.tocoo()
            (row, col) = a.nonzero()
            # 检查下三角和上三角非零元素数量是否相等
            if (row < col).sum() != (row > col).sum():
                return MMFile.SYMMETRY_GENERAL

            # 转换为DOK格式
            a = a.todok()

            # 定义对称对的迭代器
            def symm_iterator():
                for ((i, j), aij) in a.items():
                    if i > j:
                        aji = a[j, i]
                        yield (aij, aji, False)
                    elif i == j:
                        yield (aij, aij, True)

        # 处理非稀疏输入
        else:
            # 定义对称对的迭代器
            def symm_iterator():
                for j in range(n):
                    for i in range(j, n):
                        aij, aji = a[i][j], a[j][i]
                        yield (aij, aji, i == j)

        # 检查对称性
        # 返回 aij, aji, 是否对角线元素
        for (aij, aji, is_diagonal) in symm_iterator():
            # 如果是Skew对称且为对角线元素且aij不为0，则不是Skew对称
            if isskew and is_diagonal and aij != 0:
                isskew = False
            else:
                # 如果是Symm对称且aij不等于aji，则不是Symm对称
                if issymm and aij != aji:
                    issymm = False
                # 忽略溢出错误
                with np.errstate(over="ignore"):
                    # 如果是Skew对称且aij不等于-aji，则不是Skew对称
                    if isskew and aij != -aji:
                        isskew = False
                # 如果是Herm对称且aij不等于aji的共轭，则不是Herm对称
                if isherm and aij != conj(aji):
                    isherm = False
            # 如果不是Symm、Skew或者Herm对称，则中断循环
            if not (issymm or isskew or isherm):
                break

        # 返回对称性类型值
        if issymm:
            return MMFile.SYMMETRY_SYMMETRIC
        if isskew:
            return MMFile.SYMMETRY_SKEW_SYMMETRIC
        if isherm:
            return MMFile.SYMMETRY_HERMITIAN
        return MMFile.SYMMETRY_GENERAL

    # -------------------------------------------------------------------------
    # 静态方法：根据给定的字段和精度返回格式模板
    @staticmethod
    def _field_template(field, precision):
        return {MMFile.FIELD_REAL: '%%.%ie\n' % precision,
                MMFile.FIELD_INTEGER: '%i\n',
                MMFile.FIELD_UNSIGNED: '%u\n',
                MMFile.FIELD_COMPLEX: '%%.%ie %%.%ie\n' %
                    (precision, precision)
                }.get(field, None)

    # -------------------------------------------------------------------------
    # 初始化方法：根据传入的关键字参数初始化对象属性
    def __init__(self, **kwargs):
        self._init_attrs(**kwargs)

    # -------------------------------------------------------------------------
    def read(self, source):
        """
        从 Matrix Market 文件或类似的源（文件名或打开的文件对象）读取内容到矩阵中。

        Parameters
        ----------
        source : str or file-like
            Matrix Market 文件名（扩展名为 .mtx, .mtz.gz）或打开的文件对象。

        Returns
        -------
        a : ndarray or coo_matrix
            根据 Matrix Market 文件中的矩阵格式返回稠密或稀疏矩阵。
        """
        # 打开源并获取流对象及关闭标志
        stream, close_it = self._open(source)

        try:
            # 解析文件头部信息
            self._parse_header(stream)
            # 解析文件主体内容并返回相应的矩阵表示
            return self._parse_body(stream)

        finally:
            # 如果需要关闭流对象，则关闭它
            if close_it:
                stream.close()

    # -------------------------------------------------------------------------
    def write(self, target, a, comment='', field=None, precision=None,
              symmetry=None):
        """
        将稀疏或稠密数组 `a` 写入 Matrix Market 类似的目标文件。

        Parameters
        ----------
        target : str or file-like
            Matrix Market 文件名（扩展名为 .mtx）或打开的文件对象。
        a : array like
            稀疏或稠密的二维数组。
        comment : str, optional
            要添加到 Matrix Market 文件开头的注释。
        field : None or str, optional
            可以是 'real', 'complex', 'pattern', 或 'integer' 中的一个。
        precision : None or int, optional
            显示实数或复数值的小数位数。
        symmetry : None or str, optional
            可以是 'general', 'symmetric', 'skew-symmetric', 或 'hermitian' 中的一个。
            如果 symmetry 为 None，则根据数组 `a` 的值确定其对称性类型。
        """

        # 打开目标文件并获取流对象及关闭标志
        stream, close_it = self._open(target, 'wb')

        try:
            # 将数组 `a` 写入到流对象中
            self._write(stream, a, comment, field, precision, symmetry)

        finally:
            # 如果需要关闭流对象，则关闭它；否则刷新流
            if close_it:
                stream.close()
            else:
                stream.flush()

    # -------------------------------------------------------------------------
    def _init_attrs(self, **kwargs):
        """
        使用对应的关键字参数值或默认值（None）初始化每个属性。
        """

        # 获取类的所有属性名
        attrs = self.__class__.__slots__
        # 将属性名中的首字母去除，得到公共属性名列表
        public_attrs = [attr[1:] for attr in attrs]
        # 找出传入的无效关键字参数
        invalid_keys = set(kwargs.keys()) - set(public_attrs)

        if invalid_keys:
            # 如果有无效的关键字参数，则抛出 ValueError 异常
            raise ValueError('''发现 {} 个无效的关键字参数，请只使用 {}'''.format(tuple(invalid_keys),
                                             public_attrs))

        # 根据关键字参数初始化各个属性
        for attr in attrs:
            setattr(self, attr, kwargs.get(attr[1:], None))

    # -------------------------------------------------------------------------
    # 定义一个方法 `_parse_header`，接受参数 `stream`，用于解析文件头部信息
    def _parse_header(self, stream):
        # 调用类方法 `info` 分析 `stream`，返回行数、列数、条目数、格式、字段和对称性信息
        rows, cols, entries, format, field, symmetry = \
            self.__class__.info(stream)
        # 调用对象的 `_init_attrs` 方法，初始化对象的属性，
        # 将解析得到的行数、列数、条目数、格式、字段和对称性信息传递给相应的属性
        self._init_attrs(rows=rows, cols=cols, entries=entries, format=format,
                         field=field, symmetry=symmetry)

    # -------------------------------------------------------------------------
    #  ------------------------------------------------------------------------
def _is_fromfile_compatible(stream):
    """
    Check whether `stream` is compatible with numpy.fromfile.

    Passing a gzipped file object to ``fromfile/fromstring`` doesn't work with
    Python 3.
    """

    # 创建一个空列表，用于存储不兼容的流类
    bad_cls = []

    # 尝试导入 gzip 模块并添加 GzipFile 到不兼容的类列表
    try:
        import gzip
        bad_cls.append(gzip.GzipFile)
    except ImportError:
        pass

    # 尝试导入 bz2 模块并添加 BZ2File 到不兼容的类列表
    try:
        import bz2
        bad_cls.append(bz2.BZ2File)
    except ImportError:
        pass

    # 将列表转换为元组，以便进行类型检查
    bad_cls = tuple(bad_cls)

    # 检查传入的流对象是否不是不兼容类的实例，返回检查结果
    return not isinstance(stream, bad_cls)
```