# `D:\src\scipysrc\scipy\scipy\io\matlab\_mio4.py`

```
# 导入系统、警告和数学模块
import sys
import warnings
import math

# 导入 NumPy 和 SciPy 稀疏矩阵模块
import numpy as np
import scipy.sparse

# 导入 MATLAB 文件读取相关模块和函数
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
                      convert_dtypes, arr_to_chars, arr_dtype_number)

# 导入 MATLAB 文件写入相关工具函数
from ._mio_utils import squeeze_element, chars_to_strings

# 导入 functools 中的 reduce 函数
from functools import reduce

# 指定模块中公开的类和函数
__all__ = [
    'MatFile4Reader', 'MatFile4Writer', 'SYS_LITTLE_ENDIAN',
    'VarHeader4', 'VarReader4', 'VarWriter4', 'arr_to_2d', 'mclass_info',
    'mdtypes_template', 'miDOUBLE', 'miINT16', 'miINT32', 'miSINGLE',
    'miUINT16', 'miUINT8', 'mxCHAR_CLASS', 'mxFULL_CLASS', 'mxSPARSE_CLASS',
    'np_to_mtypes', 'order_codes'
]

# 系统字节顺序是否为小端
SYS_LITTLE_ENDIAN = sys.byteorder == 'little'

# MATLAB 文件格式中数据类型的常量定义
miDOUBLE = 0
miSINGLE = 1
miINT32 = 2
miINT16 = 3
miUINT16 = 4
miUINT8 = 5

# MATLAB 文件格式中数据类型与 NumPy dtype 的映射模板
mdtypes_template = {
    miDOUBLE: 'f8',
    miSINGLE: 'f4',
    miINT32: 'i4',
    miINT16: 'i2',
    miUINT16: 'u2',
    miUINT8: 'u1',
    'header': [('mopt', 'i4'),
               ('mrows', 'i4'),
               ('ncols', 'i4'),
               ('imagf', 'i4'),
               ('namlen', 'i4')],
    'U1': 'U1',
}

# NumPy dtype 到 MATLAB 文件格式数据类型的映射
np_to_mtypes = {
    'f8': miDOUBLE,
    'c32': miDOUBLE,
    'c24': miDOUBLE,
    'c16': miDOUBLE,
    'f4': miSINGLE,
    'c8': miSINGLE,
    'i4': miINT32,
    'i2': miINT16,
    'u2': miUINT16,
    'u1': miUINT8,
    'S1': miUINT8,
}

# MATLAB 矩阵类别的常量定义
mxFULL_CLASS = 0
mxCHAR_CLASS = 1
mxSPARSE_CLASS = 2

# MATLAB 文件格式中字节顺序代码与符号的映射
order_codes = {
    0: '<',
    1: '>',
    2: 'VAX D-float',  # !
    3: 'VAX G-float',
    4: 'Cray',  # !!
}

# MATLAB 矩阵类别与描述字符串的映射
mclass_info = {
    mxFULL_CLASS: 'double',
    mxCHAR_CLASS: 'char',
    mxSPARSE_CLASS: 'sparse',
}


class VarHeader4:
    # MATLAB 4 文件格式中变量的特性定义
    is_logical = False  # 逻辑类型标志位
    is_global = False  # 全局变量标志位

    def __init__(self,
                 name,
                 dtype,
                 mclass,
                 dims,
                 is_complex):
        self.name = name  # 变量名
        self.dtype = dtype  # 数据类型
        self.mclass = mclass  # 矩阵类别
        self.dims = dims  # 维度信息
        self.is_complex = is_complex  # 是否复数


class VarReader4:
    ''' 用于读取 MATLAB 4 文件格式变量的类 '''

    def __init__(self, file_reader):
        self.file_reader = file_reader  # 文件读取器
        self.mat_stream = file_reader.mat_stream  # 文件流
        self.dtypes = file_reader.dtypes  # 数据类型
        self.chars_as_strings = file_reader.chars_as_strings  # 字符串处理标志
        self.squeeze_me = file_reader.squeeze_me  # 数据压缩标志
    def read_header(self):
        ''' Read and return header for variable '''
        # 从 mat_stream 中读取指定类型的数据结构，解析为字典
        data = read_dtype(self.mat_stream, self.dtypes['header'])
        # 从 mat_stream 中读取变量名，并去除末尾的空字节
        name = self.mat_stream.read(int(data['namlen'])).strip(b'\x00')
        # 检查数据头部的 mopt 字段是否在有效范围内
        if data['mopt'] < 0 or data['mopt'] > 5000:
            raise ValueError('Mat 4 mopt wrong format, byteswapping problem?')
        # 解析 mopt 字段，获取其中的 M 值（表示字节顺序），以及剩余的部分
        M, rest = divmod(data['mopt'], 1000)  # order code
        # 如果 M 值不在支持的范围内，发出警告
        if M not in (0, 1):
            warnings.warn("We do not support byte ordering '%s'; returned "
                          "data may be corrupt" % order_codes[M],
                          UserWarning, stacklevel=3)
        # 解析剩余部分，获取 O 值（未使用应为 0）
        O, rest = divmod(rest, 100)  # unused, should be 0
        # 如果 O 值不为 0，则抛出错误
        if O != 0:
            raise ValueError('O in MOPT integer should be 0, wrong format?')
        # 解析剩余部分，获取 P 值（数据类型代码）和 T 值（矩阵类型代码）
        P, rest = divmod(rest, 10)  # data type code e.g miDOUBLE (see above)
        T = rest  # matrix type code e.g., mxFULL_CLASS (see above)
        # 将数据头部中的 mrows 和 ncols 字段解析为维度元组 dims
        dims = (data['mrows'], data['ncols'])
        # 检查 imagf 字段是否为 1，确定是否为复数类型数据
        is_complex = data['imagf'] == 1
        # 根据 P 值获取数据类型
        dtype = self.dtypes[P]
        # 返回解析后的变量头部信息对象 VarHeader4
        return VarHeader4(
            name,
            dtype,
            T,
            dims,
            is_complex)

    def array_from_header(self, hdr, process=True):
        # 获取变量头部的类别码 mclass
        mclass = hdr.mclass
        # 根据 mclass 类别码选择相应的数组读取方法
        if mclass == mxFULL_CLASS:
            arr = self.read_full_array(hdr)
        elif mclass == mxCHAR_CLASS:
            arr = self.read_char_array(hdr)
            # 如果需要处理且字符数组需要作为字符串处理，则转换为字符串数组
            if process and self.chars_as_strings:
                arr = chars_to_strings(arr)
        elif mclass == mxSPARSE_CLASS:
            # 对于稀疏矩阵，直接返回读取的稀疏数组
            return self.read_sparse_array(hdr)
        else:
            # 如果没有适合 mclass 的读取器，抛出类型错误
            raise TypeError('No reader for class code %s' % mclass)
        # 如果需要处理且需要压缩数组，则进行压缩处理
        if process and self.squeeze_me:
            return squeeze_element(arr)
        # 否则直接返回读取的数组
        return arr
    def read_sub_array(self, hdr, copy=True):
        ''' Mat4 read using header `hdr` dtype and dims
        
        Parameters
        ----------
        hdr : object
           object with attributes ``dtype``, ``dims``. dtype is assumed to be
           the correct endianness
        copy : bool, optional
           copies array before return if True (default True)
           (buffer is usually read only)
        
        Returns
        -------
        arr : ndarray
            of dtype given by `hdr` ``dtype`` and shape given by `hdr` ``dims``
        '''
        # 计算需要读取的字节数
        dt = hdr.dtype
        dims = hdr.dims
        num_bytes = dt.itemsize
        for d in dims:
            num_bytes *= d
        # 从文件流中读取指定字节数的数据到缓冲区
        buffer = self.mat_stream.read(int(num_bytes))
        if len(buffer) != num_bytes:
            # 如果读取的字节数与预期不符，抛出数值错误
            raise ValueError("Not enough bytes to read matrix '%s'; is this "
                             "a badly-formed file? Consider listing matrices "
                             "with `whosmat` and loading named matrices with "
                             "`variable_names` kwarg to `loadmat`" % hdr.name)
        # 使用缓冲区数据创建 NumPy 数组
        arr = np.ndarray(shape=dims,
                         dtype=dt,
                         buffer=buffer,
                         order='F')
        if copy:
            # 如果需要复制数组，则进行复制操作
            arr = arr.copy()
        return arr

    def read_full_array(self, hdr):
        ''' Full (rather than sparse) matrix getter
        
        Read matrix (array) can be real or complex
        
        Parameters
        ----------
        hdr : ``VarHeader4`` instance
        
        Returns
        -------
        arr : ndarray
            complex array if ``hdr.is_complex`` is True, otherwise a real
            numeric array
        '''
        if hdr.is_complex:
            # 如果是复数数组，避免复制数组以节省内存
            res = self.read_sub_array(hdr, copy=False)
            res_j = self.read_sub_array(hdr, copy=False)
            # 返回复数数组
            return res + (res_j * 1j)
        # 否则返回实数数组
        return self.read_sub_array(hdr)

    def read_char_array(self, hdr):
        ''' latin-1 text matrix (char matrix) reader
        
        Parameters
        ----------
        hdr : ``VarHeader4`` instance
        
        Returns
        -------
        arr : ndarray
            with dtype 'U1', shape given by `hdr` ``dims``
        '''
        # 读取字符数组并转换为 uint8 类型
        arr = self.read_sub_array(hdr).astype(np.uint8)
        # 将 uint8 数组转换为 Latin-1 编码的字符串
        S = arr.tobytes().decode('latin-1')
        # 使用字符串数据创建 Unicode 字符数组的副本并返回
        return np.ndarray(shape=hdr.dims,
                          dtype=np.dtype('U1'),
                          buffer=np.array(S)).copy()
    def read_sparse_array(self, hdr):
        ''' Read and return sparse matrix type
        
        Parameters
        ----------
        hdr : ``VarHeader4`` instance
            包含有关稀疏矩阵的元数据的对象
        
        Returns
        -------
        arr : ``scipy.sparse.coo_matrix``
            使用从稀疏矩阵数据中读取的数据类型为 ``float``，形状从数据中获取的稀疏矩阵
        
        Notes
        -----
        MATLAB 4 实数稀疏数组以 N+1 行 3 列的数组格式保存，其中 N 是非零值的数量。第一列值 [0:N] 是每个非零值的（基于 1 的）行索引，
        第二列 [0:N] 是列索引，第三列 [0:N] 是（实数）值。行的最后值 [-1,0:2] 分别是输出矩阵的形状的 mrows 和 ncols 值。值列的最后一个值是填充的 0。
        头部的 mrows 和 ncols 值给出了存储矩阵的形状，这里为 [N+1, 3]。复数数据以 4 列矩阵保存，第四列包含虚部；值列的最后一个值再次为 0。
        复数稀疏数据的头部没有设置 ``imagf`` 字段为 True；数据为复数只能通过有 4 个存储列来检测。

        '''
        res = self.read_sub_array(hdr)
        tmp = res[:-1,:]  # 从结果中获取所有行，但是去掉最后一行
        dims = (int(res[-1,0]), int(res[-1,1]))  # 从最后一行中获取形状信息，并转换为整数
        I = np.ascontiguousarray(tmp[:,0],dtype='intc')  # 将第一列视为连续的整数数组，修正字节顺序
        J = np.ascontiguousarray(tmp[:,1],dtype='intc')  # 将第二列视为连续的整数数组，修正字节顺序
        I -= 1  # 为了从 1 开始的索引进行减一修正
        J -= 1  # 为了从 1 开始的索引进行减一修正
        if res.shape[1] == 3:
            V = np.ascontiguousarray(tmp[:,2],dtype='float')  # 如果结果的列数为 3，则将第三列视为连续的浮点数数组
        else:
            V = np.ascontiguousarray(tmp[:,2],dtype='complex')  # 否则将第三列视为连续的复数数组
            V.imag = tmp[:,3]  # 设置虚部为结果的第四列
        return scipy.sparse.coo_matrix((V,(I,J)), dims)  # 返回以 (V, (I, J)) 形式构建的 scipy 稀疏 COO 矩阵，指定形状为 dims
    def shape_from_header(self, hdr):
        '''Read the shape of the array described by the header.
        The file position after this call is unspecified.
        '''
        # 从头部信息中获取数组的形状
        mclass = hdr.mclass
        # 根据不同的类别码确定形状的读取方式
        if mclass == mxFULL_CLASS:
            # 如果是完整类别码，直接从维度信息中读取形状
            shape = tuple(map(int, hdr.dims))
        elif mclass == mxCHAR_CLASS:
            # 如果是字符类别码，也直接从维度信息中读取形状
            shape = tuple(map(int, hdr.dims))
            # 如果需要将字符视为字符串，则在形状中去除最后一个维度
            if self.chars_as_strings:
                shape = shape[:-1]
        elif mclass == mxSPARSE_CLASS:
            # 如果是稀疏类别码，需要处理数据类型和维度信息
            dt = hdr.dtype
            dims = hdr.dims

            # 确保维度信息是有效的二维数组
            if not (len(dims) == 2 and dims[0] >= 1 and dims[1] >= 1):
                return ()

            # 读取稀疏矩阵的行数和列数
            self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
            rows = np.ndarray(shape=(), dtype=dt,
                              buffer=self.mat_stream.read(dt.itemsize))
            self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
            cols = np.ndarray(shape=(), dtype=dt,
                              buffer=self.mat_stream.read(dt.itemsize))

            # 形状为行数和列数的元组
            shape = (int(rows), int(cols))
        else:
            # 如果类别码无法识别，则引发类型错误
            raise TypeError('No reader for class code %s' % mclass)

        # 如果需要压缩形状（去除维度为1的部分），则进行压缩处理
        if self.squeeze_me:
            shape = tuple([x for x in shape if x != 1])
        
        # 返回最终确定的数组形状
        return shape
# MatFile4Reader 类，用于读取 Mat4 格式的文件
class MatFile4Reader(MatFileReader):
    ''' Mat4 文件的读取器 '''

    @docfiller
    def __init__(self, mat_stream, *args, **kwargs):
        ''' 初始化 Matlab 4 文件读取器

        %(matstream_arg)s
        %(load_args)s
        '''
        # 调用父类的初始化方法
        super().__init__(mat_stream, *args, **kwargs)
        # 初始化矩阵读取器为 None
        self._matrix_reader = None

    def guess_byte_order(self):
        ''' 猜测字节顺序

        通过读取文件流的第一个整型值来猜测文件的字节顺序
        '''
        # 将文件流的指针移动到开头
        self.mat_stream.seek(0)
        # 读取第一个整型值
        mopt = read_dtype(self.mat_stream, np.dtype('i4'))
        # 再次将文件流的指针移动到开头
        self.mat_stream.seek(0)
        # 根据读取的值来判断字节顺序
        if mopt == 0:
            return '<'
        if mopt < 0 or mopt > 5000:
            # 数字必须已经进行了字节交换
            return SYS_LITTLE_ENDIAN and '>' or '<'
        # 没有进行字节交换
        return SYS_LITTLE_ENDIAN and '<' or '>'

    def initialize_read(self):
        ''' 初始化读取过程

        根据 self 中的参数设置读取器
        '''
        # 将数据类型转换为当前字节顺序
        self.dtypes = convert_dtypes(mdtypes_template, self.byte_order)
        # 初始化矩阵读取器
        self._matrix_reader = VarReader4(self)

    def read_var_header(self):
        ''' 读取并返回变量头部信息及下一个位置

        Returns
        -------
        header : object
           可传递给 self.read_var_array 的对象，具有属性 `name` 和 `is_global`
        next_position : int
           下一个变量在流中的位置
        '''
        # 从矩阵读取器中读取头部信息
        hdr = self._matrix_reader.read_header()
        # 计算数组元素个数的快速乘积
        n = reduce(lambda x, y: x*y, hdr.dims, 1)  # 快速乘积
        # 计算剩余的字节数
        remaining_bytes = hdr.dtype.itemsize * n
        # 如果是复数且不是稀疏矩阵，则剩余字节数需要乘以2
        if hdr.is_complex and not hdr.mclass == mxSPARSE_CLASS:
            remaining_bytes *= 2
        # 计算下一个变量在流中的位置
        next_position = self.mat_stream.tell() + remaining_bytes
        return hdr, next_position

    def read_var_array(self, header, process=True):
        ''' 根据头部信息读取数组

        Parameters
        ----------
        header : header object
           定义变量头部的对象，包含各种字段
        process : {True, False}, optional
           如果为 True，则在加载数组时应用递归后处理。

        Returns
        -------
        arr : array
           根据 `process` 应用或未应用后处理的数组
        '''
        # 从矩阵读取器中读取给定头部的数组
        return self._matrix_reader.array_from_header(header, process)
    def get_variables(self, variable_names=None):
        ''' get variables from stream as dictionary

        Parameters
        ----------
        variable_names : None or str or sequence of str, optional
            variable name, or sequence of variable names to get from Mat file /
            file stream. If None, then get all variables in file.
        '''
        # 如果 variable_names 是字符串，转换成包含单个元素的列表
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        # 如果 variable_names 不是 None，且不是字符串，转换成列表形式
        elif variable_names is not None:
            variable_names = list(variable_names)
        # 将文件流的位置调整到起始位置
        self.mat_stream.seek(0)
        # 设置变量读取的初始化条件
        self.initialize_read()
        # 创建空字典来存储变量名和变量数据
        mdict = {}
        # 循环直到流的末尾
        while not self.end_of_stream():
            # 读取变量的头部信息和下一个变量的位置
            hdr, next_position = self.read_var_header()
            # 将变量名转换为字符串（如果为 None 则设为 'None'）
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            # 如果 variable_names 不为空且当前变量名不在 variable_names 中，则跳过当前变量
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            # 将变量名和对应的数据存入字典 mdict
            mdict[name] = self.read_var_array(hdr)
            # 将流的位置调整到下一个变量的位置
            self.mat_stream.seek(next_position)
            # 如果 variable_names 不为空，则移除已经获取的变量名
            if variable_names is not None:
                variable_names.remove(name)
                # 如果 variable_names 已经为空，则跳出循环
                if len(variable_names) == 0:
                    break
        # 返回存储变量名和数据的字典 mdict
        return mdict

    def list_variables(self):
        ''' list variables from stream '''
        # 将文件流的位置调整到起始位置
        self.mat_stream.seek(0)
        # 设置变量读取的初始化条件
        self.initialize_read()
        # 创建空列表来存储变量信息
        vars = []
        # 循环直到流的末尾
        while not self.end_of_stream():
            # 读取变量的头部信息和下一个变量的位置
            hdr, next_position = self.read_var_header()
            # 将变量名转换为字符串（如果为 None 则设为 'None'）
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            # 从头部信息中获取变量的形状信息
            shape = self._matrix_reader.shape_from_header(hdr)
            # 根据变量的类型码获取变量的信息（例如类型名称）
            info = mclass_info.get(hdr.mclass, 'unknown')
            # 将变量名、形状和类型信息组成元组，添加到 vars 列表中
            vars.append((name, shape, info))

            # 将流的位置调整到下一个变量的位置
            self.mat_stream.seek(next_position)
        # 返回包含所有变量信息的列表 vars
        return vars
def arr_to_2d(arr, oned_as='row'):
    ''' Make ``arr`` exactly two dimensional

    If `arr` has more than 2 dimensions, raise a ValueError

    Parameters
    ----------
    arr : array
        Input array to be reshaped
    oned_as : {'row', 'column'}, optional
       Whether to reshape 1-D vectors as row vectors or column vectors.
       See documentation for ``matdims`` for more detail

    Returns
    -------
    arr2d : array
       2-D version of the input array
    '''
    # Determine the desired dimensions for the array
    dims = matdims(arr, oned_as)
    # Check if the array has more than 2 dimensions
    if len(dims) > 2:
        raise ValueError('Matlab 4 files cannot save arrays with more than '
                         '2 dimensions')
    # Reshape the input array to exactly 2 dimensions based on calculated dimensions
    return arr.reshape(dims)


class VarWriter4:
    def __init__(self, file_writer):
        # Initialize with the file stream from the given file writer
        self.file_stream = file_writer.file_stream
        # Set the 'oned_as' attribute from the file writer
        self.oned_as = file_writer.oned_as

    def write_bytes(self, arr):
        # Write the bytes representation of the array to the file stream
        self.file_stream.write(arr.tobytes(order='F'))

    def write_string(self, s):
        # Write the string 's' to the file stream
        self.file_stream.write(s)

    def write_header(self, name, shape, P=miDOUBLE, T=mxFULL_CLASS, imagf=0):
        ''' Write header for given data options

        Parameters
        ----------
        name : str
            Name of the variable
        shape : sequence
            Shape of array as it will be read in Matlab
        P : int, optional
            Code for mat4 data type, one of ``miDOUBLE, miSINGLE, miINT32,
            miINT16, miUINT16, miUINT8``
        T : int, optional
            Code for mat4 matrix class, one of ``mxFULL_CLASS, mxCHAR_CLASS,
            mxSPARSE_CLASS``
        imagf : int, optional
            Flag indicating complex data

        '''
        # Create an empty header using predefined template for Matlab data
        header = np.empty((), mdtypes_template['header'])
        # Determine byte order and other flags for Matlab format
        M = not SYS_LITTLE_ENDIAN
        O = 0
        # Construct the 'mopt' field of the header
        header['mopt'] = (M * 1000 +
                          O * 100 +
                          P * 10 +
                          T)
        # Set the number of rows and columns in the header
        header['mrows'] = shape[0]
        header['ncols'] = shape[1]
        # Set the imaginary flag in the header
        header['imagf'] = imagf
        # Set the length of the variable name field in the header
        header['namlen'] = len(name) + 1
        # Write the header bytes to the file stream
        self.write_bytes(header)
        # Prepare the variable name string and write it to the file stream
        data = name + '\0'
        self.write_string(data.encode('latin1'))
    def write(self, arr, name):
        ''' Write matrix `arr`, with name `name`

        Parameters
        ----------
        arr : array_like
           array to write
        name : str
           name in matlab workspace
        '''
        # 检查 arr 是否为稀疏矩阵，因为 np.asarray 对 scipy.sparse 返回对象数组
        if scipy.sparse.issparse(arr):
            # 如果是稀疏矩阵，则调用 write_sparse 方法
            self.write_sparse(arr, name)
            return
        # 将 arr 转换为 NumPy 数组
        arr = np.asarray(arr)
        # 获取数组的数据类型
        dt = arr.dtype
        # 如果数据类型不是本机字节顺序，则转换为本机字节顺序
        if not dt.isnative:
            arr = arr.astype(dt.newbyteorder('='))
        # 获取数据类型的类型
        dtt = dt.type
        # 如果数据类型是对象数组，则抛出类型错误
        if dtt is np.object_:
            raise TypeError('Cannot save object arrays in Mat4')
        # 如果数据类型是 void 类型，则抛出类型错误
        elif dtt is np.void:
            raise TypeError('Cannot save void type arrays')
        # 如果数据类型是字符串类型（str_ 或 bytes_），则调用 write_char 方法
        elif dtt in (np.str_, np.bytes_):
            self.write_char(arr, name)
            return
        # 否则调用 write_numeric 方法
        self.write_numeric(arr, name)

    def write_numeric(self, arr, name):
        # 将 arr 转换为二维数组
        arr = arr_to_2d(arr, self.oned_as)
        # 检查是否为复数数组
        imagf = arr.dtype.kind == 'c'
        try:
            # 根据数据类型获取对应的 Matlab 类型
            P = np_to_mtypes[arr.dtype.str[1:]]
        except KeyError:
            # 如果未找到对应的 Matlab 类型，根据 imagf 决定转换为复数或浮点数类型
            if imagf:
                arr = arr.astype('c128')
            else:
                arr = arr.astype('f8')
            P = miDOUBLE
        # 调用 write_header 方法写入头部信息
        self.write_header(name,
                          arr.shape,
                          P=P,
                          T=mxFULL_CLASS,
                          imagf=imagf)
        # 如果是复数数组，分别写入实部和虚部；否则直接写入数组数据
        if imagf:
            self.write_bytes(arr.real)
            self.write_bytes(arr.imag)
        else:
            self.write_bytes(arr)

    def write_char(self, arr, name):
        # 如果数组类型为字符串（str_）且字符大小不是 Unicode 1 大小
        if arr.dtype.type == np.str_ and arr.dtype.itemsize != np.dtype('U1').itemsize:
            # 将字符串数组转换为字符数组
            arr = arr_to_chars(arr)
        # 将 arr 转换为二维数组
        arr = arr_to_2d(arr, self.oned_as)
        # 获取数组的维度
        dims = arr.shape
        # 调用 write_header 方法写入头部信息
        self.write_header(
            name,
            dims,
            P=miUINT8,
            T=mxCHAR_CLASS)
        # 如果数组类型是 Unicode 类型
        if arr.dtype.kind == 'U':
            # 将 Unicode 编码为 Latin-1
            n_chars = math.prod(dims)
            st_arr = np.ndarray(shape=(),
                                dtype=arr_dtype_number(arr, n_chars),
                                buffer=arr)
            st = st_arr.item().encode('latin-1')
            arr = np.ndarray(shape=dims, dtype='S1', buffer=st)
        # 调用 write_bytes 方法写入数组数据
        self.write_bytes(arr)
    # 定义一个方法，用于将稀疏矩阵写入到某个名称对应的文件中
    def write_sparse(self, arr, name):
        ''' Sparse matrices are 2-D

        See docstring for VarReader4.read_sparse_array
        '''
        # 将稀疏矩阵 arr 转换为 COO 格式（行索引，列索引，值）
        A = arr.tocoo()  # convert to sparse COO format (ijv)
        # 判断数组 A 的数据类型是否为复数类型
        imagf = A.dtype.kind == 'c'
        # 创建一个用于存储 COO 格式数据的数组 ijv，初始大小为 (A.nnz + 1, 3+imagf)
        ijv = np.zeros((A.nnz + 1, 3+imagf), dtype='f8')
        # 将 A 的行索引复制到 ijv 的前 A.nnz 行的第一列
        ijv[:-1,0] = A.row
        # 将 A 的列索引复制到 ijv 的前 A.nnz 行的第二列
        ijv[:-1,1] = A.col
        # 将 ijv 的前 A.nnz 行的第一列和第二列加一，实现从零到一的索引转换为从一到二的索引
        ijv[:-1,0:2] += 1  # 1 based indexing
        # 如果数据类型为复数类型
        if imagf:
            # 将 A 的实部复制到 ijv 的前 A.nnz 行的第三列
            ijv[:-1,2] = A.data.real
            # 将 A 的虚部复制到 ijv 的前 A.nnz 行的第四列
            ijv[:-1,3] = A.data.imag
        else:
            # 将 A 的数据复制到 ijv 的前 A.nnz 行的第三列
            ijv[:-1,2] = A.data
        # 将 A 的形状信息复制到 ijv 的最后一行的前两列
        ijv[-1,0:2] = A.shape
        # 调用对象的 write_header 方法，写入文件头信息
        self.write_header(
            name,
            ijv.shape,
            P=miDOUBLE,
            T=mxSPARSE_CLASS)
        # 调用对象的 write_bytes 方法，写入二进制数据 ijv
        self.write_bytes(ijv)
    # 定义 MatFile4Writer 类，用于写入 MATLAB 4 格式文件
    ''' Class for writing matlab 4 format files '''
    def __init__(self, file_stream, oned_as=None):
        # 初始化方法，接受一个文件流和一个可选的参数 oned_as
        self.file_stream = file_stream
        # 如果 oned_as 为 None，则默认设置为 'row'
        if oned_as is None:
            oned_as = 'row'
        self.oned_as = oned_as
        self._matrix_writer = None

    def put_variables(self, mdict, write_header=None):
        ''' Write variables in `mdict` to stream
        
        Parameters
        ----------
        mdict : mapping
           mapping with method ``items`` return name, contents pairs
           where ``name`` which will appeak in the matlab workspace in
           file load, and ``contents`` is something writeable to a
           matlab file, such as a NumPy array.
        write_header : {None, True, False}
           If True, then write the matlab file header before writing the
           variables. If None (the default) then write the file header
           if we are at position 0 in the stream. By setting False
           here, and setting the stream position to the end of the file,
           you can append variables to a matlab file
        '''
        # 对于 MATLAB 4 格式文件，没有文件头，因此忽略 write_header 参数。
        # 这个参数是为了与 MATLAB 5 版本的方法兼容而存在的。
        self._matrix_writer = VarWriter4(self)
        # 遍历 mdict 中的每个变量名和变量内容，并使用 _matrix_writer 将其写入流中
        for name, var in mdict.items():
            self._matrix_writer.write(var, name)
```