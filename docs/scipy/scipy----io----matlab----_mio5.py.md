# `D:\src\scipysrc\scipy\scipy\io\matlab\_mio5.py`

```
# 导入必要的库和模块
import math  # 导入数学库
import os    # 导入操作系统功能库
import time  # 导入时间处理库
import sys   # 导入系统相关库
import zlib  # 导入数据压缩库

# 导入字节流处理模块
from io import BytesIO

# 导入警告处理模块
import warnings
# 导入 NumPy 库，使用 np 作为别名
import numpy as np

# 导入 SciPy 稀疏矩阵模块
import scipy.sparse

# 从 _byteordercodes 模块中导入 native_code 和 swapped_code 常量
from ._byteordercodes import native_code, swapped_code

# 从 _miobase 模块中导入多个函数和异常类
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
                      arr_to_chars, arr_dtype_number, MatWriteError,
                      MatReadError, MatReadWarning)

# 从 _mio5_utils 模块中导入 VarReader5 类
from ._mio5_utils import VarReader5

# 从 _mio5_params 模块中导入多个常量和对象
from ._mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
                          NP_TO_MXTYPES, miCOMPRESSED, miMATRIX, miINT8,
                          miUTF8, miUINT32, mxCELL_CLASS, mxSTRUCT_CLASS,
                          mxOBJECT_CLASS, mxCHAR_CLASS, mxSPARSE_CLASS,
                          mxDOUBLE_CLASS, mclass_info, mat_struct)

# 从 _streams 模块中导入 ZlibInputStream 类
from ._streams import ZlibInputStream


def _has_struct(elem):
    """Determine if elem is an array and if first array item is a struct."""
    # 检查 elem 是否为 ndarray 类型，且其第一个元素是否为 mat_struct 类型
    return (isinstance(elem, np.ndarray) and (elem.size > 0) and (elem.ndim > 0) and
            isinstance(elem[0], mat_struct))


def _inspect_cell_array(ndarray):
    """Construct lists from cell arrays (loaded as numpy ndarrays), recursing
    into items if they contain mat_struct objects."""
    # 从单元数组（加载为 numpy ndarray）构建列表，并在包含 mat_struct 对象的情况下递归处理子项
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, mat_struct):
            elem_list.append(_matstruct_to_dict(sub_elem))
        elif _has_struct(sub_elem):
            elem_list.append(_inspect_cell_array(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list


def _matstruct_to_dict(matobj):
    """Construct nested dicts from mat_struct objects."""
    # 从 mat_struct 对象构建嵌套字典
    d = {}
    for f in matobj._fieldnames:
        elem = matobj.__dict__[f]
        if isinstance(elem, mat_struct):
            d[f] = _matstruct_to_dict(elem)
        elif _has_struct(elem):
            d[f] = _inspect_cell_array(elem)
        else:
            d[f] = elem
    return d


def _simplify_cells(d):
    """Convert mat objects in dict to nested dicts."""
    # 将字典中的 mat 对象转换为嵌套字典
    for key in d:
        if isinstance(d[key], mat_struct):
            d[key] = _matstruct_to_dict(d[key])
        elif _has_struct(d[key]):
            d[key] = _inspect_cell_array(d[key])
    return d


class MatFile5Reader(MatFileReader):
    ''' Reader for Mat 5 mat files
    Adds the following attribute to base class

    uint16_codec - char codec to use for uint16 char arrays
        (defaults to system default codec)

    Uses variable reader that has the following standard interface (see
    abstract class in ``miobase``::

       __init__(self, file_reader)
       read_header(self)
       array_from_header(self)

    and added interface::

       set_stream(self, stream)
       read_full_tag(self)

    '''
    # Mat 5 格式 .mat 文件的读取器
    @docfiller
    def __init__(self,
                 mat_stream,
                 byte_order=None,
                 mat_dtype=False,
                 squeeze_me=False,
                 chars_as_strings=True,
                 matlab_compatible=False,
                 struct_as_record=True,
                 verify_compressed_data_integrity=True,
                 uint16_codec=None,
                 simplify_cells=False):
        '''Initializer for matlab 5 file format reader
        
        Initializes an instance of the MATLAB 5 file format reader with various options.

        Parameters:
        mat_stream : file-like object
            Stream object representing the MATLAB file contents.
        byte_order : {'<', '>', None}, optional
            Byte order specifier for reading data (little-endian, big-endian, or None for default).
        mat_dtype : bool, optional
            Flag indicating whether to apply MATLAB data type interpretation.
        squeeze_me : bool, optional
            Flag indicating whether to squeeze singleton dimensions.
        chars_as_strings : bool, optional
            Flag indicating whether to interpret char arrays as strings.
        matlab_compatible : bool, optional
            Flag indicating compatibility mode for MATLAB.
        struct_as_record : bool, optional
            Flag indicating whether to treat MATLAB structs as records.
        verify_compressed_data_integrity : bool, optional
            Flag indicating whether to verify integrity of compressed data.
        uint16_codec : {None, string}, optional
            Codec to use for uint16 char arrays (e.g., 'utf-8'). Defaults to system default.
        simplify_cells : bool, optional
            Flag indicating whether to simplify cell arrays.
        '''
        super().__init__(
            mat_stream,
            byte_order,
            mat_dtype,
            squeeze_me,
            chars_as_strings,
            matlab_compatible,
            struct_as_record,
            verify_compressed_data_integrity,
            simplify_cells)
        # Set uint16 codec if not provided
        if not uint16_codec:
            uint16_codec = sys.getdefaultencoding()
        self.uint16_codec = uint16_codec
        # Initialize placeholders for readers
        self._file_reader = None
        self._matrix_reader = None

    def guess_byte_order(self):
        ''' Guess byte order based on MATLAB file header.

        This method examines the first few bytes of the MATLAB file to determine
        the byte order used for data encoding.

        Returns:
        str:
            '<' for little-endian or '>' for big-endian, based on the file header.
        '''
        self.mat_stream.seek(126)
        mi = self.mat_stream.read(2)
        self.mat_stream.seek(0)
        return mi == b'IM' and '<' or '>'

    def read_file_header(self):
        ''' Read the header information from a MATLAB file.

        This method reads the header section of a MATLAB file, extracting
        information such as file description and version.

        Returns:
        dict:
            A dictionary containing '__header__' and '__version__' information.
        '''
        hdict = {}
        hdr_dtype = MDTYPES[self.byte_order]['dtypes']['file_header']
        hdr = read_dtype(self.mat_stream, hdr_dtype)
        hdict['__header__'] = hdr['description'].item().strip(b' \t\n\000')
        v_major = hdr['version'] >> 8
        v_minor = hdr['version'] & 0xFF
        hdict['__version__'] = '%d.%d' % (v_major, v_minor)
        return hdict

    def initialize_read(self):
        ''' Initialize readers for reading MATLAB variables.

        This method sets up two readers: one for the top-level stream (`_file_reader`)
        and another for matrix streams (`_matrix_reader`). These readers are essential
        for parsing and interpreting MATLAB data structures during reading.

        '''
        # Initialize reader for top level stream
        self._file_reader = VarReader5(self)
        # Initialize reader for matrix streams
        self._matrix_reader = VarReader5(self)
    def read_var_header(self):
        ''' 
        Read header, return header, next position
        
        Header has to define at least .name and .is_global
        
        Parameters
        ----------
        None
        
        Returns
        -------
        header : object
           object that can be passed to self.read_var_array, and that
           has attributes .name and .is_global
        next_position : int
           position in stream of next variable
        '''
        # 从文件读取标签的类型和字节数
        mdtype, byte_count = self._file_reader.read_full_tag()
        # 如果未读取任何字节，则抛出值错误异常
        if not byte_count > 0:
            raise ValueError("Did not read any bytes")
        # 计算下一个变量的流位置
        next_pos = self.mat_stream.tell() + byte_count
        # 如果标签类型为 miCOMPRESSED
        if mdtype == miCOMPRESSED:
            # 从压缩数据创建新的流
            stream = ZlibInputStream(self.mat_stream, byte_count)
            # 设置矩阵读取器的流为新创建的压缩流
            self._matrix_reader.set_stream(stream)
            # 是否验证压缩数据完整性的标志
            check_stream_limit = self.verify_compressed_data_integrity
            # 从压缩流中读取标签类型和字节数
            mdtype, byte_count = self._matrix_reader.read_full_tag()
        else:
            # 如果不是压缩类型，则不需要验证流限制
            check_stream_limit = False
            # 设置矩阵读取器的流为原始流
            self._matrix_reader.set_stream(self.mat_stream)
        # 如果标签类型不是 miMATRIX，则抛出类型错误异常
        if not mdtype == miMATRIX:
            raise TypeError('Expecting miMATRIX type here, got %d' % mdtype)
        # 从流中读取头部信息
        header = self._matrix_reader.read_header(check_stream_limit)
        # 返回读取的头部信息和下一个变量的流位置
        return header, next_pos

    def read_var_array(self, header, process=True):
        ''' 
        Read array, given `header`
        
        Parameters
        ----------
        header : header object
           object with fields defining variable header
        process : {True, False} bool, optional
           If True, apply recursive post-processing during loading of
           array.
        
        Returns
        -------
        arr : array
           array with post-processing applied or not according to
           `process`.
        '''
        # 从头部对象读取数组，如果 process 为 True，则进行递归后处理
        return self._matrix_reader.array_from_header(header, process)
    def get_variables(self, variable_names=None):
        ''' get variables from stream as dictionary
        
        variable_names   - optional list of variable names to get
        
        If variable_names is None, then get all variables in file
        '''
        # 如果 variable_names 是字符串，则转换为列表
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        # 如果 variable_names 不为 None，则将其转换为列表形式
        elif variable_names is not None:
            variable_names = list(variable_names)

        # 将文件指针移至流的起始位置
        self.mat_stream.seek(0)
        # 调用初始化读取函数，使用 self 中的所有参数
        self.initialize_read()
        # 读取文件头部信息，并返回字典 mdict
        mdict = self.read_file_header()
        # 将 '__globals__' 初始化为空列表
        mdict['__globals__'] = []
        
        # 当流未结束时循环读取变量
        while not self.end_of_stream():
            # 读取变量头部信息及下一个位置
            hdr, next_position = self.read_var_header()
            # 解码变量名为字符串（使用 'latin1' 编码）
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            
            # 如果变量名已存在于 mdict 中，发出警告
            if name in mdict:
                msg = (
                    f'Duplicate variable name "{name}" in stream'
                    " - replacing previous with new\nConsider"
                    "scipy.io.matlab._mio5.varmats_from_mat to split "
                    "file into single variable files"
                )
                warnings.warn(msg, MatReadWarning, stacklevel=2)
            
            # 如果变量名为空字符串，表示可能是 MATLAB 7 函数工作空间
            if name == '':
                name = '__function_workspace__'
                process = False  # 保持原始状态，避免 mat_dtype 处理问题
            else:
                process = True
            
            # 如果指定了 variable_names 并且当前变量名不在列表中，则跳过
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            
            try:
                # 尝试读取变量数组数据
                res = self.read_var_array(hdr, process)
            except MatReadError as err:
                # 如果读取出错，则发出警告
                warnings.warn(
                    f'Unreadable variable "{name}", because "{err}"',
                    Warning, stacklevel=2)
                res = "Read error: %s" % err
            
            # 将文件指针移至下一个位置
            self.mat_stream.seek(next_position)
            # 将变量名及其对应的数据存入 mdict
            mdict[name] = res
            
            # 如果变量被标记为全局变量，则添加到 '__globals__' 列表中
            if hdr.is_global:
                mdict['__globals__'].append(name)
            
            # 如果指定了 variable_names，则移除已读取的变量名
            if variable_names is not None:
                variable_names.remove(name)
                # 如果 variable_names 列表为空，则跳出循环
                if len(variable_names) == 0:
                    break
        
        # 如果需要简化单元格结构，则返回简化后的 mdict
        if self.simplify_cells:
            return _simplify_cells(mdict)
        else:
            return mdict
    # 定义一个方法，用于列出从流中读取的变量
    def list_variables(self):
        ''' list variables from stream '''
        # 将流的读取位置设置为起始位置
        self.mat_stream.seek(0)
        # 调用初始化读取方法，使用self中的所有参数来配置读取对象
        self.initialize_read()
        # 读取文件头信息
        self.read_file_header()
        # 初始化一个空列表用于存储变量信息
        vars = []
        # 循环直到流的末尾
        while not self.end_of_stream():
            # 读取变量头信息及下一个变量的位置
            hdr, next_position = self.read_var_header()
            # 解码变量名，如果为None则设置为'None'
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            # 如果变量名为空字符串，表示可能是MATLAB 7的函数工作空间
            if name == '':
                name = '__function_workspace__'

            # 从头信息中获取变量的形状
            shape = self._matrix_reader.shape_from_header(hdr)
            # 如果变量标记为逻辑类型
            if hdr.is_logical:
                info = 'logical'
            else:
                # 否则根据MATLAB类型码获取类型信息，如果未知则标记为'unknown'
                info = mclass_info.get(hdr.mclass, 'unknown')
            # 将变量名、形状和类型信息作为元组添加到变量列表中
            vars.append((name, shape, info))

            # 设置流的读取位置到下一个变量的起始位置
            self.mat_stream.seek(next_position)
        
        # 返回包含所有变量信息的列表
        return vars
def varmats_from_mat(file_obj):
    """ 从 mat 5 文件中提取变量作为 mat 文件对象的序列

    这在处理包含难以读取变量的 mat 文件时非常有用。此函数以原始形式提取变量，并将其未读取的形式放回文件流中，以供保存或读取。另一个用途是处理存在多个同名变量的情况，此函数会返回这些重复变量，而标准读取器会覆盖返回字典中的重复变量。

    参数
    ----------
    file_obj : file-like
        包含 mat 文件的文件对象

    返回
    -------
    named_mats : list
        包含元组 (name, BytesIO) 的列表，其中 BytesIO 是文件流对象，包含单个变量的 mat 文件内容。BytesIO 包含原始头部和单个变量的字符串。如果 var_file_obj 是单个 BytesIO 实例，则可以像这样保存为 mat 文件：
        open('test.mat', 'wb').write(var_file_obj.read())

    示例
    --------
    >>> import scipy.io
    >>> import numpy as np
    >>> from io import BytesIO
    >>> from scipy.io.matlab._mio5 import varmats_from_mat
    >>> mat_fileobj = BytesIO()
    >>> scipy.io.savemat(mat_fileobj, {'b': np.arange(10), 'a': 'a string'})
    >>> varmats = varmats_from_mat(mat_fileobj)
    >>> sorted([name for name, str_obj in varmats])
    ['a', 'b']
    """
    # 使用 MatFile5Reader 类读取文件对象
    rdr = MatFile5Reader(file_obj)
    # 重置文件对象的位置到起始位置
    file_obj.seek(0)
    # 读取顶层文件头的原始数据
    hdr_len = MDTYPES[native_code]['dtypes']['file_header'].itemsize
    raw_hdr = file_obj.read(hdr_len)
    # 初始化变量读取过程
    file_obj.seek(0)
    rdr.initialize_read()
    rdr.read_file_header()
    next_position = file_obj.tell()
    named_mats = []
    while not rdr.end_of_stream():
        start_position = next_position
        # 读取变量头部和下一个变量的位置
        hdr, next_position = rdr.read_var_header()
        # 解码变量名
        name = 'None' if hdr.name is None else hdr.name.decode('latin1')
        # 读取原始变量字符串
        file_obj.seek(start_position)
        byte_count = next_position - start_position
        var_str = file_obj.read(byte_count)
        # 写入到 BytesIO 对象中
        out_obj = BytesIO()
        out_obj.write(raw_hdr)
        out_obj.write(var_str)
        out_obj.seek(0)
        named_mats.append((name, out_obj))
    return named_mats


class EmptyStructMarker:
    """ 表示输出中存在空 matlab 结构体的类 """


def to_writeable(source):
    ''' 将输入对象 ``source`` 转换为可写入的对象

    参数
    ----------
    source : object
        输入对象

    返回
    -------
    '''
    arr : None or ndarray or EmptyStructMarker
        如果 `source` 不能转换为可以写入 matfile 的内容，则返回 None。
        如果 `source` 等同于空字典，则返回 `EmptyStructMarker`。
        否则，返回将 `source` 转换为适合写入 matfile 的 ndarray 的结果。

    '''
    # 如果 `source` 已经是 ndarray，则直接返回它
    if isinstance(source, np.ndarray):
        return source
    # 如果 `source` 是 None，则返回 None
    if source is None:
        return None
    # 如果 `source` 具有 `__array__` 属性，则将其转换为 ndarray 返回
    if hasattr(source, "__array__"):
        return np.asarray(source)
    
    # 对于实现了映射的对象
    is_mapping = (hasattr(source, 'keys') and hasattr(source, 'values') and
                  hasattr(source, 'items'))
    
    # 对于 NumPy 的通用类型
    if isinstance(source, np.generic):
        # NumPy 标量永远不是映射（PyPy 的问题的回避）
        pass
    # 对于不是映射但有 `__dict__` 属性的对象
    elif not is_mapping and hasattr(source, '__dict__'):
        # 将对象的 `__dict__` 转换为字典，但忽略以 `_` 开头的键
        source = {key: value for key, value in source.__dict__.items()
                  if not key.startswith('_')}
        is_mapping = True
    
    # 如果 `source` 是映射对象
    if is_mapping:
        dtype = []
        values = []
        # 遍历映射对象的键值对
        for field, value in source.items():
            # 如果键是字符串且不以数字或下划线开头，则将其作为结构体的字段名
            if (isinstance(field, str) and
                    field[0] not in '_0123456789'):
                dtype.append((str(field), object))
                values.append(value)
        # 如果有符合条件的字段，则返回其构成的 ndarray
        if dtype:
            return np.array([tuple(values)], dtype)
        else:
            # 否则返回 `EmptyStructMarker`
            return EmptyStructMarker
    
    # 尝试将 `source` 转换为 ndarray
    try:
        narr = np.asanyarray(source)
    except ValueError:
        narr = np.asanyarray(source, dtype=object)
    
    # 如果转换后的 ndarray 类型为 object 并且形状为 ()，且与原始 `source` 相等，则返回 None
    if narr.dtype.type in (object, np.object_) and \
       narr.shape == () and narr == source:
        return None
    
    # 否则返回转换后的 ndarray
    return narr
# 用于方便写入程序的本地字节顺序数据类型
NDT_FILE_HDR = MDTYPES[native_code]['dtypes']['file_header']
# 完整标签的本地字节顺序数据类型
NDT_TAG_FULL = MDTYPES[native_code]['dtypes']['tag_full']
# 小数据标签的本地字节顺序数据类型
NDT_TAG_SMALL = MDTYPES[native_code]['dtypes']['tag_smalldata']
# 数组标志的本地字节顺序数据类型
NDT_ARRAY_FLAGS = MDTYPES[native_code]['dtypes']['array_flags']

class VarWriter5:
    ''' 通用的 Matlab 矩阵写入类 '''
    # 创建一个空的 NumPy 数组作为 mat_tag，并使用 NDT_TAG_FULL 数据类型
    mat_tag = np.zeros((), NDT_TAG_FULL)
    mat_tag['mdtype'] = miMATRIX

    def __init__(self, file_writer):
        # 初始化时，从文件写入器获取文件流
        self.file_stream = file_writer.file_stream
        # 从文件写入器获取 Unicode 字符串设置
        self.unicode_strings = file_writer.unicode_strings
        # 从文件写入器获取长字段名称设置
        self.long_field_names = file_writer.long_field_names
        # 从文件写入器获取 one-dimensional 设置
        self.oned_as = file_writer.oned_as
        # 用于顶层写入的变量名，初始为 None
        self._var_name = None
        # 用于顶层写入的变量是否为全局变量，默认为 False
        self._var_is_global = False

    def write_bytes(self, arr):
        # 将数组 arr 转换为字节流并写入文件流
        self.file_stream.write(arr.tobytes(order='F'))

    def write_string(self, s):
        # 将字符串 s 写入文件流
        self.file_stream.write(s)

    def write_element(self, arr, mdtype=None):
        ''' 写入标签和数据 '''
        if mdtype is None:
            mdtype = NP_TO_MTYPES[arr.dtype.str[1:]]
        # 如果数组的字节顺序不是本地字节顺序，则进行调整
        if arr.dtype.byteorder == swapped_code:
            arr = arr.byteswap().view(arr.dtype.newbyteorder())
        byte_count = arr.size * arr.itemsize
        # 如果字节数小于等于 4，则使用 write_smalldata_element 方法
        if byte_count <= 4:
            self.write_smalldata_element(arr, mdtype, byte_count)
        else:
            # 否则使用 write_regular_element 方法
            self.write_regular_element(arr, mdtype, byte_count)

    def write_smalldata_element(self, arr, mdtype, byte_count):
        # 写入带有嵌入数据的标签
        tag = np.zeros((), NDT_TAG_SMALL)
        tag['byte_count_mdtype'] = (byte_count << 16) + mdtype
        # 将数组 arr 转换为字节流并写入标签的 data 字段
        tag['data'] = arr.tobytes(order='F')
        self.write_bytes(tag)

    def write_regular_element(self, arr, mdtype, byte_count):
        # 写入标签和数据
        tag = np.zeros((), NDT_TAG_FULL)
        tag['mdtype'] = mdtype
        tag['byte_count'] = byte_count
        self.write_bytes(tag)  # 写入标签
        self.write_bytes(arr)  # 写入数据
        # 填充到下一个 64 位边界
        bc_mod_8 = byte_count % 8
        if bc_mod_8:
            self.file_stream.write(b'\x00' * (8 - bc_mod_8))
    def write_header(self,
                     shape,
                     mclass,
                     is_complex=False,
                     is_logical=False,
                     nzmax=0):
        ''' Write header for given data options
        shape : sequence
           array shape
        mclass      - mat5 matrix class
        is_complex  - True if matrix is complex
        is_logical  - True if matrix is logical
        nzmax        - max non zero elements for sparse arrays

        We get the name and the global flag from the object, and reset
        them to defaults after we've used them
        '''
        # 从对象存储中获取变量名和是否全局标志，并在使用后重置为默认值
        name = self._var_name
        is_global = self._var_is_global
        # 初始化顶层矩阵标签，并记录位置
        self._mat_tag_pos = self.file_stream.tell()
        self.write_bytes(self.mat_tag)
        # 写入数组标志 (复数、全局、逻辑、类别、nzmax)
        af = np.zeros((), NDT_ARRAY_FLAGS)
        af['data_type'] = miUINT32
        af['byte_count'] = 8
        flags = is_complex << 3 | is_global << 2 | is_logical << 1
        af['flags_class'] = mclass | flags << 8
        af['nzmax'] = nzmax
        self.write_bytes(af)
        # 写入数组形状
        self.write_element(np.array(shape, dtype='i4'))
        # 写入变量名
        name = np.asarray(name)
        if name == '':  # 空字符串以零结尾
            self.write_smalldata_element(name, miINT8, 0)
        else:
            self.write_element(name, miINT8)
        # 重置对象存储的变量名和全局标志为默认值
        self._var_name = ''
        self._var_is_global = False

    def update_matrix_tag(self, start_pos):
        curr_pos = self.file_stream.tell()
        self.file_stream.seek(start_pos)
        byte_count = curr_pos - start_pos - 8
        if byte_count >= 2**32:
            raise MatWriteError("Matrix too large to save with Matlab "
                                "5 format")
        self.mat_tag['byte_count'] = byte_count
        self.write_bytes(self.mat_tag)
        self.file_stream.seek(curr_pos)

    def write_top(self, arr, name, is_global):
        """ Write variable at top level of mat file

        Parameters
        ----------
        arr : array_like
            array-like object to create writer for
        name : str, optional
            name as it will appear in matlab workspace
            default is empty string
        is_global : {False, True}, optional
            whether variable will be global on load into matlab
        """
        # 这些在顶层头部写入前设置，在相同写入结束后取消设置，因为它们对于较低级别不适用
        self._var_is_global = is_global
        self._var_name = name
        # 写入头部和数据
        self.write(arr)
    def write(self, arr):
        ''' Write `arr` to stream at top and sub levels

        Parameters
        ----------
        arr : array_like
            array-like object to create writer for
        '''
        # 存储当前文件流位置，以便后续更新矩阵标签
        mat_tag_pos = self.file_stream.tell()

        # 首先检查是否为稀疏矩阵
        if scipy.sparse.issparse(arr):
            # 如果是稀疏矩阵，调用写稀疏矩阵的方法
            self.write_sparse(arr)
            # 更新矩阵标签
            self.update_matrix_tag(mat_tag_pos)
            return

        # 尝试转换非数组对象
        narr = to_writeable(arr)
        if narr is None:
            raise TypeError(f'Could not convert {arr} (type {type(arr)}) to array')

        # 处理特定类型的对象
        if isinstance(narr, MatlabObject):
            self.write_object(narr)
        elif isinstance(narr, MatlabFunction):
            raise MatWriteError('Cannot write matlab functions')
        elif narr is EmptyStructMarker:  # 空结构体数组
            self.write_empty_struct()
        elif narr.dtype.fields:  # 结构体数组
            self.write_struct(narr)
        elif narr.dtype.hasobject:  # 单元数组
            self.write_cells(narr)
        elif narr.dtype.kind in ('U', 'S'):
            # 字符数组处理
            if self.unicode_strings:
                codec = 'UTF8'
            else:
                codec = 'ascii'
            self.write_char(narr, codec)
        else:
            # 数值数组处理
            self.write_numeric(narr)

        # 更新矩阵标签
        self.update_matrix_tag(mat_tag_pos)

    def write_numeric(self, arr):
        # 检查数组是否为复数类型
        imagf = arr.dtype.kind == 'c'
        # 检查数组是否为逻辑类型
        logif = arr.dtype.kind == 'b'

        try:
            # 获取 MATLAB 类型对应的类型码
            mclass = NP_TO_MXTYPES[arr.dtype.str[1:]]
        except KeyError:
            # 没有匹配的 MATLAB 类型，可能是复杂类型如 complex256 / float128 / float96
            # 将数据转换为 complex128 / float64 类型
            if imagf:
                arr = arr.astype('c128')
            elif logif:
                arr = arr.astype('i1')  # 应只包含 0 或 1
            else:
                arr = arr.astype('f8')
            # 设置为默认的双精度浮点类型
            mclass = mxDOUBLE_CLASS

        # 写入数据头部信息
        self.write_header(matdims(arr, self.oned_as),
                          mclass,
                          is_complex=imagf,
                          is_logical=logif)

        # 如果是复数数组，分别写入实部和虚部
        if imagf:
            self.write_element(arr.real)
            self.write_element(arr.imag)
        else:
            # 否则直接写入数组元素
            self.write_element(arr)
    def write_char(self, arr, codec='ascii'):
        ''' Write string array `arr` with given `codec`
        '''
        # Check if the array `arr` is empty or contains only empty strings
        if arr.size == 0 or np.all(arr == ''):
            # Handle special case where `arr` is empty or contains only empty strings
            # This is necessary due to how Matlab treats empty and zero-padded strings
            shape = (0,) * np.max([arr.ndim, 2])
            # Write header for empty array shape with character class
            self.write_header(shape, mxCHAR_CLASS)
            # Write small data element for empty array
            self.write_smalldata_element(arr, miUTF8, 0)
            return
        
        # Convert `arr` to a character array
        arr = arr_to_chars(arr)
        
        # Write the shape directly because the character recoding may change the length
        shape = arr.shape
        self.write_header(shape, mxCHAR_CLASS)
        
        # Check if `arr` dtype is Unicode and not empty
        if arr.dtype.kind == 'U' and arr.size:
            # Concatenate all characters into one long string
            # Transpose `arr` to flatten it and ensure Fortran order for byte writing
            n_chars = math.prod(shape)
            st_arr = np.ndarray(shape=(),
                                dtype=arr_dtype_number(arr, n_chars),
                                buffer=arr.T.copy())  # Fortran order
            # Encode concatenated string with specified codec to get byte string
            st = st_arr.item().encode(codec)
            # Reconstruct as 1-D byte array
            arr = np.ndarray(shape=(len(st),),
                             dtype='S1',
                             buffer=st)
        
        # Write the final element
        self.write_element(arr, mdtype=miUTF8)

    def write_sparse(self, arr):
        ''' Sparse matrices are 2D
        '''
        # Convert `arr` to Compressed Sparse Column format
        A = arr.tocsc()
        # Ensure row indices are sorted, as required by MATLAB
        A.sort_indices()
        # Check if matrix elements are complex or logical
        is_complex = (A.dtype.kind == 'c')
        is_logical = (A.dtype.kind == 'b')
        # Number of non-zero elements in `A`
        nz = A.nnz
        # Write header for sparse matrix, specifying dimensions and class
        self.write_header(matdims(arr, self.oned_as),
                          mxSPARSE_CLASS,
                          is_complex=is_complex,
                          is_logical=is_logical,
                          # Ensure MATLAB can load file even with nzmax as 0
                          nzmax=1 if nz == 0 else nz)
        # Write elements: indices of non-zero elements
        self.write_element(A.indices.astype('i4'))
        # Write elements: index pointers for column starts
        self.write_element(A.indptr.astype('i4'))
        # Write elements: real parts of non-zero data
        self.write_element(A.data.real)
        # Write imaginary parts if matrix is complex
        if is_complex:
            self.write_element(A.data.imag)
    # 写入 MATLAB 的细胞数组（cell array），包括数据维度和类别信息
    def write_cells(self, arr):
        self.write_header(matdims(arr, self.oned_as),
                          mxCELL_CLASS)
        # 将输入数组转换为至少是二维的，按列主序（column-major）展开
        A = np.atleast_2d(arr).flatten('F')
        # 遍历展开后的数组元素，并依次写入
        for el in A:
            self.write(el)

    # 写入空结构体数据
    def write_empty_struct(self):
        self.write_header((1, 1), mxSTRUCT_CLASS)
        # 在示例 MATLAB 结构体中，字段名长度最大为1
        self.write_element(np.array(1, dtype=np.int32))
        # 字段名元素为空
        self.write_element(np.array([], dtype=np.int8))

    # 写入结构体数据
    def write_struct(self, arr):
        self.write_header(matdims(arr, self.oned_as),
                          mxSTRUCT_CLASS)
        # 写入结构体的具体项
        self._write_items(arr)

    # 写入结构体的具体项
    def _write_items(self, arr):
        # 写入字段名
        fieldnames = [f[0] for f in arr.dtype.descr]
        # 计算字段名的最大长度，并确保不超过设定的最大长度
        length = max([len(fieldname) for fieldname in fieldnames])+1
        max_length = (self.long_field_names and 64) or 32
        if length > max_length:
            raise ValueError("Field names are restricted to %d characters" %
                             (max_length-1))
        # 写入字段名长度信息
        self.write_element(np.array([length], dtype='i4'))
        # 将字段名以指定长度写入，数据类型为 'S<length>'
        self.write_element(
            np.array(fieldnames, dtype='S%d' % (length)),
            mdtype=miINT8)
        # 展开数组为至少二维的列主序，并依次写入每个字段的数据
        A = np.atleast_2d(arr).flatten('F')
        for el in A:
            for f in fieldnames:
                self.write(el[f])

    # 写入对象数据，类似于写入结构体，但使用不同的 mx 类型，并在头部之后添加额外的 classname 元素
    def write_object(self, arr):
        '''Same as writing structs, except different mx class, and extra
        classname element after header
        '''
        self.write_header(matdims(arr, self.oned_as),
                          mxOBJECT_CLASS)
        # 写入 classname 元素
        self.write_element(np.array(arr.classname, dtype='S'),
                           mdtype=miINT8)
        # 写入对象的具体项
        self._write_items(arr)
# MatFile5Writer 类，用于写入 mat5 格式的文件
class MatFile5Writer:
    ''' Class for writing mat5 files '''

    @docfiller
    # 初始化方法，接收多个参数来配置文件写入行为
    def __init__(self, file_stream,
                 do_compression=False,
                 unicode_strings=False,
                 global_vars=None,
                 long_field_names=False,
                 oned_as='row'):
        ''' Initialize writer for matlab 5 format files

        Parameters
        ----------
        %(do_compression)s
            是否进行压缩，默认为 False
        %(unicode_strings)s
            是否使用 Unicode 字符串，默认为 False
        global_vars : None or sequence of strings, optional
            需要标记为 Matlab 全局变量的变量名列表
        %(long_fields)s
            是否使用长字段名，默认为 False
        %(oned_as)s
            数组的方向，默认为 'row'
        '''
        # 设置文件流
        self.file_stream = file_stream
        # 是否进行压缩
        self.do_compression = do_compression
        # 是否使用 Unicode 字符串
        self.unicode_strings = unicode_strings
        # 如果有指定全局变量，则设置全局变量列表；否则为空列表
        if global_vars:
            self.global_vars = global_vars
        else:
            self.global_vars = []
        # 是否使用长字段名
        self.long_field_names = long_field_names
        # 数组方向设置
        self.oned_as = oned_as
        # 矩阵写入器初始化为 None
        self._matrix_writer = None

    # 写入文件头部信息的方法
    def write_file_header(self):
        # 创建一个空的 numpy 数组作为文件头部信息
        hdr = np.zeros((), NDT_FILE_HDR)
        # 设置文件描述信息，包括 Matlab 版本和创建时间
        hdr['description'] = (f'MATLAB 5.0 MAT-file Platform: {os.name}, '
                              f'Created on: {time.asctime()}')
        # 设置版本号
        hdr['version'] = 0x0100
        # 设置字节序测试值，使用小端序列化 'MI'
        hdr['endian_test'] = np.ndarray(shape=(),
                                      dtype='S2',
                                      buffer=np.uint16(0x4d49))
        # 将文件头部信息写入到文件流中
        self.file_stream.write(hdr.tobytes())
    def put_variables(self, mdict, write_header=None):
        ''' Write variables in `mdict` to stream

        Parameters
        ----------
        mdict : mapping
           mapping with method ``items`` returns name, contents pairs where
           ``name`` which will appear in the matlab workspace in file load, and
           ``contents`` is something writeable to a matlab file, such as a NumPy
           array.
        write_header : {None, True, False}, optional
           If True, then write the matlab file header before writing the
           variables. If None (the default) then write the file header
           if we are at position 0 in the stream. By setting False
           here, and setting the stream position to the end of the file,
           you can append variables to a matlab file
        '''
        # 如果未指定是否写入文件头并且流的位置在文件开头，则写入文件头
        if write_header is None:
            write_header = self.file_stream.tell() == 0
        if write_header:
            # 调用写入文件头的方法
            self.write_file_header()
        
        # 初始化矩阵写入器
        self._matrix_writer = VarWriter5(self)
        
        # 遍历给定的字典 mdict，包含变量名和内容
        for name, var in mdict.items():
            # 如果变量名以下划线开头，则跳过该变量
            if name[0] == '_':
                continue
            
            # 检查变量名是否在全局变量列表中
            is_global = name in self.global_vars
            
            # 如果开启了数据压缩
            if self.do_compression:
                # 创建一个字节流对象
                stream = BytesIO()
                # 将文件流设置为字节流对象
                self._matrix_writer.file_stream = stream
                # 使用矩阵写入器写入数据
                self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
                # 压缩字节流内容
                out_str = zlib.compress(stream.getvalue())
                # 创建一个标签数组对象
                tag = np.empty((), NDT_TAG_FULL)
                # 设置标签对象的元数据类型为压缩类型
                tag['mdtype'] = miCOMPRESSED
                # 设置标签对象的字节长度为压缩后内容的长度
                tag['byte_count'] = len(out_str)
                # 将标签对象的字节表示写入文件流
                self.file_stream.write(tag.tobytes())
                # 将压缩后的内容写入文件流
                self.file_stream.write(out_str)
            else:  # 如果没有开启压缩
                # 使用矩阵写入器直接写入数据
                self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
```