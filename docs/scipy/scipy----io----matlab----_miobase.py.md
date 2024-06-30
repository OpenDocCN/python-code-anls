# `D:\src\scipysrc\scipy\scipy\io\matlab\_miobase.py`

```
# 导入所需模块和包
import numpy as np  # 导入 NumPy 库
from scipy._lib import doccer  # 导入 doccer 模块，用于文档字符串处理

from . import _byteordercodes as boc  # 导入本地模块 _byteordercodes

# 定义公开的类和异常列表
__all__ = [
    'MatReadError', 'MatReadWarning', 'MatWriteError',
]

# 定义 MATLAB 文件读取时可能出现的异常类
class MatReadError(Exception):
    """Exception indicating a read issue."""

# 定义 MATLAB 文件写入时可能出现的异常类
class MatWriteError(Exception):
    """Exception indicating a write issue."""

# 定义 MATLAB 文件读取时的警告类
class MatReadWarning(UserWarning):
    """Warning class for read issues."""

# 定义文档字符串字典，用于填充函数和类的文档
doc_dict = \
    {'file_arg':
         '''file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True) Can also pass open file-like object.''',
     'append_arg':
         '''appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present. Default is True.''',
     'load_args':
         '''byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).''',
     'struct_arg':
         '''struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of SciPy version 0.7.x (returning
   numpy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.''',
     'matstream_arg':
         '''mat_stream : file-like
   Object with file API, open for reading.''',
     'long_fields':
         '''long_field_names : bool, optional
   * False - maximum field name length in a structure is 31 characters
     which is the documented maximum length. This is the default.
   * True - maximum field name length in a structure is 63 characters
     which works for MATLAB 7.6''',
     'do_compression':
         '''do_compression : bool, optional
   Whether to compress matrices on write. Default is False.''',
     'oned_as':
         '''oned_as : {'row', 'column'}, optional
   If 'column', write 1-D NumPy arrays as column vectors.
   If 'row', write 1D NumPy arrays as row vectors.''',
     'unicode_strings':
         '''unicode_strings : bool, optional
   If True, write strings as Unicode, else MATLAB usual encoding.'''}

# 使用 doccer 模块填充文档字符串的内容
docfiller = doccer.filldoc(doc_dict)

'''
'''
# 将给定的数据类型映射转换为指定字节顺序的新数据类型映射

Parameters
----------
dtype_template : mapping
   包含从``np.dtype(val)``返回numpy数据类型的映射
order_code : str
   可以用于``dtype.newbyteorder()``的顺序代码

Returns
-------
dtypes : mapping
   映射，其中的值已被替换为``np.dtype(val).newbyteorder(order_code)``后的结果

'''
def convert_dtypes(dtype_template, order_code):
    dtypes = dtype_template.copy()
    for k in dtypes:
        dtypes[k] = np.dtype(dtypes[k]).newbyteorder(order_code)
    return dtypes


'''
# 读取已知类型的字节流数据的通用函数

Parameters
----------
mat_stream : file_like object
    MATLAB (tm) mat 文件流
a_dtype : dtype
    要读取的数组的数据类型。假定`a_dtype`的字节顺序是正确的。

Returns
-------
arr : ndarray
    从流中读取的数据类型为`a_dtype`的数组。

'''
def read_dtype(mat_stream, a_dtype):
    num_bytes = a_dtype.itemsize
    arr = np.ndarray(shape=(),
                     dtype=a_dtype,
                     buffer=mat_stream.read(num_bytes),
                     order='F')
    return arr


'''
# 根据文件名返回主要版本和次要版本的元组，依赖于MAT文件类型

其中：

#. 0,x -> 版本 4 格式的MAT文件
#. 1,x -> 版本 5 格式的MAT文件
#. 2,x -> 版本 7.3 格式的MAT文件（HDF格式）

Parameters
----------
file_name : str
   MAT文件的名称（如果appendmat==True，则不需要.mat扩展名）。也可以传递打开的类文件对象。

'''
def matfile_version(file_name, *, appendmat=True):
    pass
    # appendmat 参数用于指定是否在文件名末尾添加 .mat 扩展名（如果尚未存在）。默认为 True。
    # 返回值为 MATLAB 文件格式的主要版本号和次要版本号。
    # 如果文件为空，则引发 MatReadError 异常。
    # 如果 matfile 版本未知，则引发 ValueError 异常。
    # 注意：该函数会将文件读取指针设置为 0。
    
    from ._mio import _open_file_context
    # 使用 _open_file_context 函数打开文件并返回文件对象，利用 with 语句进行资源管理
    with _open_file_context(file_name, appendmat=appendmat) as fileobj:
        # 调用 _get_matfile_version 函数，传入文件对象 fileobj，获取 MATLAB 文件的版本信息
        return _get_matfile_version(fileobj)
# 获取当前脚本的matfile_version，并将其赋值给get_matfile_version
get_matfile_version = matfile_version


# 从文件对象中获取MAT文件的版本信息
def _get_matfile_version(fileobj):
    # 将文件对象的读取位置移动到文件开头
    fileobj.seek(0)
    # 读取前4个字节，用于判断MAT文件的版本
    mopt_bytes = fileobj.read(4)
    # 如果读取的字节长度为0，表示MAT文件为空，抛出MatReadError异常
    if len(mopt_bytes) == 0:
        raise MatReadError("Mat file appears to be empty")
    # 将读取的字节数据转换为一个包含4个无符号整数的NumPy数组
    mopt_ints = np.ndarray(shape=(4,), dtype=np.uint8, buffer=mopt_bytes)
    # 如果数组中包含0，表示MAT文件是Mat4格式，返回(0, 0)
    if 0 in mopt_ints:
        fileobj.seek(0)
        return (0, 0)
    # 对于Mat5或7.3格式，需要读取文件头部的整数信息
    # 字节124到128包含一个版本整数和一个大小端测试字符串
    fileobj.seek(124)
    tst_str = fileobj.read(4)
    fileobj.seek(0)
    # 解析版本信息，确定主版本号和次版本号
    maj_ind = int(tst_str[2] == b'I'[0])
    maj_val = int(tst_str[maj_ind])
    min_val = int(tst_str[1 - maj_ind])
    ret = (maj_val, min_val)
    # 如果主版本号是1或2，则返回版本信息
    if maj_val in (1, 2):
        return ret
    # 否则抛出异常，表示未知的MAT文件类型和版本
    raise ValueError('Unknown mat file type, version {}, {}'.format(*ret))


# 确定给定数组的等效MATLAB维度
def matdims(arr, oned_as='column'):
    """
    Determine equivalent MATLAB dimensions for given array

    Parameters
    ----------
    arr : ndarray
        Input array
    oned_as : {'column', 'row'}, optional
        Whether 1-D arrays are returned as MATLAB row or column matrices.
        Default is 'column'.

    Returns
    -------
    dims : tuple
        Shape tuple, in the form MATLAB expects it.

    Notes
    -----
    We had to decide what shape a 1 dimensional array would be by
    default. ``np.atleast_2d`` thinks it is a row vector. The
    default for a vector in MATLAB (e.g., ``>> 1:12``) is a row vector.

    Versions of scipy up to and including 0.11 resulted (accidentally)
    in 1-D arrays being read as column vectors. For the moment, we
    maintain the same tradition here.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.io.matlab._miobase import matdims
    >>> matdims(np.array(1)) # NumPy scalar
    (1, 1)
    >>> matdims(np.array([1])) # 1-D array, 1 element
    (1, 1)
    >>> matdims(np.array([1,2])) # 1-D array, 2 elements
    (2, 1)
    >>> matdims(np.array([[2],[3]])) # 2-D array, column vector
    (2, 1)
    >>> matdims(np.array([[2,3]])) # 2-D array, row vector
    (1, 2)
    >>> matdims(np.array([[[2,3]]])) # 3-D array, rowish vector
    (1, 1, 2)
    >>> matdims(np.array([])) # empty 1-D array
    (0, 0)
    >>> matdims(np.array([[]])) # empty 2-D array
    (0, 0)
    >>> matdims(np.array([[[]]])) # empty 3-D array
    (0, 0, 0)

    Optional argument flips 1-D shape behavior.

    >>> matdims(np.array([1,2]), 'row') # 1-D array, 2 elements
    (1, 2)

    The argument has to make sense though

    >>> matdims(np.array([1,2]), 'bizarre')
    Traceback (most recent call last):
       ...
    ValueError: 1-D option "bizarre" is strange

    """
    # 获取数组的形状
    shape = arr.shape
    # 如果数组是标量，返回(1, 1)
    if shape == ():
        return (1, 1)
    # 如果形状是一维的情况下进行判断
    if len(shape) == 1:
        # 如果一维数组长度为0，返回一个元组 (0, 0)
        if shape[0] == 0:
            return (0, 0)
        # 如果指定将一维数组视作列向量，返回形状元组末尾增加一个维度1的元组
        elif oned_as == 'column':
            return shape + (1,)
        # 如果指定将一维数组视作行向量，返回一个元组，在形状元组开头增加一个维度1
        elif oned_as == 'row':
            return (1,) + shape
        # 如果指定的视图选项不是 'column' 或 'row'，抛出数值错误异常
        else:
            raise ValueError('1-D option "%s" is strange' % oned_as)
    
    # 返回形状，此处表示形状不是一维的情况
    return shape
# 定义一个抽象类，定义了用于读取变量的接口
class MatVarReader:
    ''' Abstract class defining required interface for var readers'''
    def __init__(self, file_reader):
        # 初始化方法，未实现具体功能
        pass

    def read_header(self):
        ''' Returns header '''
        # 未实现的方法，应该返回头部信息
        pass

    def array_from_header(self, header):
        ''' Reads array given header '''
        # 未实现的方法，根据给定的头部信息读取数组
        pass


class MatFileReader:
    """ Base object for reading mat files

    To make this class functional, you will need to override the
    following methods:

    matrix_getter_factory   - gives object to fetch next matrix from stream
    guess_byte_order        - guesses file byte order from file
    """

    @docfiller
    def __init__(self, mat_stream,
                 byte_order=None,
                 mat_dtype=False,
                 squeeze_me=False,
                 chars_as_strings=True,
                 matlab_compatible=False,
                 struct_as_record=True,
                 verify_compressed_data_integrity=True,
                 simplify_cells=False):
        '''
        Initializer for mat file reader

        mat_stream : file-like
            object with file API, open for reading
    %(load_args)s
        '''
        # 初始化方法，设置 mat 文件流和一些参数
        # 如果没有指定字节顺序，尝试猜测文件的字节顺序
        if not byte_order:
            byte_order = self.guess_byte_order()
        else:
            byte_order = boc.to_numpy_code(byte_order)
        self.mat_stream = mat_stream
        self.dtypes = {}  # 数据类型的字典
        self.byte_order = byte_order  # 文件的字节顺序
        self.struct_as_record = struct_as_record  # 是否将结构体作为记录处理
        # 根据 matlab_compatible 参数设置一些选项
        if matlab_compatible:
            self.set_matlab_compatible()
        else:
            self.squeeze_me = squeeze_me
            self.chars_as_strings = chars_as_strings
            self.mat_dtype = mat_dtype
        self.verify_compressed_data_integrity = verify_compressed_data_integrity
        self.simplify_cells = simplify_cells
        # 如果启用了 simplify_cells 参数，设置 squeeze_me 为 True，struct_as_record 为 False
        if simplify_cells:
            self.squeeze_me = True
            self.struct_as_record = False

    def set_matlab_compatible(self):
        ''' Sets options to return arrays as MATLAB loads them '''
        # 设置选项，使得返回的数组与 MATLAB 加载的方式兼容
        self.mat_dtype = True
        self.squeeze_me = False
        self.chars_as_strings = False

    def guess_byte_order(self):
        ''' As we do not know what file type we have, assume native '''
        # 因为不知道文件类型，假设使用本地的字节顺序
        return boc.native_code

    def end_of_stream(self):
        # 检查是否到达流的末尾
        b = self.mat_stream.read(1)
        curpos = self.mat_stream.tell()
        self.mat_stream.seek(curpos-1)
        return len(b) == 0


def arr_dtype_number(arr, num):
    ''' Return dtype for given number of items per element'''
    # 返回给定每个元素项目数的数据类型
    return np.dtype(arr.dtype.str[:2] + str(num))


def arr_to_chars(arr):
    ''' Convert string array to char array '''
    # 将字符串数组转换为字符数组
    dims = list(arr.shape)
    if not dims:
        dims = [1]
    dims.append(int(arr.dtype.str[2:]))
    arr = np.ndarray(shape=dims,
                     dtype=arr_dtype_number(arr, 1),
                     buffer=arr)
    empties = [arr == np.array('', dtype=arr.dtype)]
    if not np.any(empties):
        return arr
    arr = arr.copy()
    # 使用元组作为字典的键，将空位置列表 `empties` 转换为元组并在数组 `arr` 中设置为空格字符 ' '
    arr[tuple(empties)] = ' '
    # 返回更新后的数组 `arr`
    return arr
```