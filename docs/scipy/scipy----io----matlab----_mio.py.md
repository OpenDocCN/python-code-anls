# `D:\src\scipysrc\scipy\scipy\io\matlab\_mio.py`

```
"""
Module for reading and writing matlab (TM) .mat files
"""
# Authors: Travis Oliphant, Matthew Brett

# 导入所需模块
from contextlib import contextmanager

# 从内部模块中导入函数和类
from ._miobase import _get_matfile_version, docfiller
from ._mio4 import MatFile4Reader, MatFile4Writer
from ._mio5 import MatFile5Reader, MatFile5Writer

# 模块中公开的函数和类
__all__ = ['loadmat', 'savemat', 'whosmat']


@contextmanager
def _open_file_context(file_like, appendmat, mode='rb'):
    """
    Context manager for opening file-like objects.

    Parameters
    ----------
    file_like : file-like object or str
        File to open or already opened file-like object.
    appendmat : bool
        Whether to append '.mat' to the file name if not already present.
    mode : str, optional
        Mode in which to open the file ('rb' for read binary by default).

    Yields
    ------
    f : file-like object
        Opened file-like object.
    """
    # 调用_open_file函数获取文件对象
    f, opened = _open_file(file_like, appendmat, mode)
    try:
        yield f
    finally:
        # 如果文件是由本函数打开的，则关闭它
        if opened:
            f.close()


def _open_file(file_like, appendmat, mode='rb'):
    """
    Open `file_like` and return as file-like object. First, check if object is
    already file-like; if so, return it as-is. Otherwise, try to pass it
    to open(). If that fails, and `file_like` is a string, and `appendmat` is true,
    append '.mat' and try again.

    Parameters
    ----------
    file_like : file-like object or str
        File to open or already opened file-like object.
    appendmat : bool
        Whether to append '.mat' to the file name if not already present.
    mode : str, optional
        Mode in which to open the file ('rb' for read binary by default).

    Returns
    -------
    f : file-like object
        Opened file-like object.
    opened : bool
        Indicates whether the file was opened by this function.
    """
    reqs = {'read'} if set(mode) & set('r+') else set()
    if set(mode) & set('wax+'):
        reqs.add('write')
    # 如果file_like已经是文件对象的一部分，则直接返回
    if reqs.issubset(dir(file_like)):
        return file_like, False

    try:
        # 尝试打开文件，如果失败且file_like是字符串，且appendmat为True，则加上'.mat'后缀再次尝试
        return open(file_like, mode), True
    except OSError as e:
        # 可能的异常情况处理，如文件未找到
        if isinstance(file_like, str):
            if appendmat and not file_like.endswith('.mat'):
                file_like += '.mat'
            return open(file_like, mode), True
        else:
            raise OSError(
                'Reader needs file name or open file-like object'
            ) from e


@docfiller
def mat_reader_factory(file_name, appendmat=True, **kwargs):
    """
    Create reader for matlab .mat format files.

    Parameters
    ----------
    file_name : str
        Name of the mat file (do not need .mat extension if
        appendmat==True). Can also pass open file-like object.
    appendmat : bool, optional
        Whether to append the .mat extension to the end of the given
        filename, if not already present. Default is True.
    **kwargs : keyword arguments
        Additional arguments to pass to the specific MatFileReader.

    Returns
    -------
    matreader : MatFileReader object
        Initialized instance of MatFileReader class matching the mat file
        type detected in `filename`.
    file_opened : bool
        Whether the file was opened by this routine.
    """
    # 获取文件的字节流和打开状态
    byte_stream, file_opened = _open_file(file_name, appendmat)
    # 获取.mat文件的版本信息
    mjv, mnv = _get_matfile_version(byte_stream)
    # 根据.mat文件版本选择合适的读取器
    if mjv == 0:
        return MatFile4Reader(byte_stream, **kwargs), file_opened
    elif mjv == 1:
        return MatFile5Reader(byte_stream, **kwargs), file_opened
    elif mjv == 2:
        raise NotImplementedError('Please use HDF reader for matlab v7.3 '
                                  'files, e.g. h5py')
    else:
        raise TypeError('Did not recognize version %s' % mjv)


@docfiller
def loadmat(file_name, mdict=None, appendmat=True, **kwargs):
    """
    Load MATLAB file.

    Parameters
    ----------
    file_name : str
        Name of the mat file (do not need .mat extension if
        appendmat==True). Can also pass open file-like object.
    mdict : dict, optional
        Dictionary in which to insert matfile variables.
    appendmat : bool, optional
        True to append the .mat extension to the end of the given
        filename, if not already present. Default is True.
    **kwargs : keyword arguments
        Additional arguments to pass to the specific MatFileReader.

    Notes
    -----
    This function loads data from a MATLAB .mat file into Python variables.
    """
    # byte_order 参数，指定字节顺序，默认为 None，表示从 mat 文件中猜测字节顺序
       None by default, implying byte order guessed from mat
       file. Otherwise can be one of ('native', '=', 'little', '<',
       'BIG', '>').
    # mat_dtype 参数，如果为 True，返回的数组与 MATLAB 中加载时的 dtype 相同
       If True, return arrays in same dtype as would be loaded into
       MATLAB (instead of the dtype with which they are saved).
    # squeeze_me 参数，指定是否压缩单位矩阵的维度
       Whether to squeeze unit matrix dimensions or not.
    # chars_as_strings 参数，指定是否将字符数组转换为字符串数组
       Whether to convert char arrays to string arrays.
    # matlab_compatible 参数，返回 MATLAB 加载的矩阵形式
       Returns matrices as would be loaded by MATLAB (implies
       squeeze_me=False, chars_as_strings=False, mat_dtype=True,
       struct_as_record=True).
    # struct_as_record 参数，指定是否将 MATLAB 结构体加载为 NumPy 记录数组
       Whether to load MATLAB structs as NumPy record arrays, or as
       old-style NumPy arrays with dtype=object. Setting this flag to
       False replicates the behavior of scipy version 0.7.x (returning
       NumPy object arrays). The default setting is True, because it
       allows easier round-trip load and save of MATLAB files.
    # verify_compressed_data_integrity 参数，指定是否验证 MATLAB 文件中压缩序列的长度
        Whether the length of compressed sequences in the MATLAB file
        should be checked, to ensure that they are not longer than we expect.
        It is advisable to enable this (the default) because overlong
        compressed sequences in MATLAB files generally indicate that the
        files have experienced some sort of corruption.
    # variable_names 参数，指定要从文件中读取的 MATLAB 变量名称序列
        If None (the default) - read all variables in file. Otherwise,
        `variable_names` should be a sequence of strings, giving names of the
        MATLAB variables to read from the file. The reader will skip any
        variable with a name not in this sequence, possibly saving some read
        processing.
    # simplify_cells 参数，指定是否返回简化的字典结构（对包含单元数组的 mat 文件有用）
        If True, return a simplified dict structure (which is useful if the mat
        file contains cell arrays). Note that this only affects the structure
        of the result and not its contents (which is identical for both output
        structures). If True, this automatically sets `struct_as_record` to
        False and `squeeze_me` to True, which is required to simplify cells.

    # 返回值
    mat_dict : dict
       dictionary with variable names as keys, and loaded matrices as
       values.

    # 注意事项
    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

    You will need an HDF5 Python library to read MATLAB 7.3 format mat
    files. Because SciPy does not supply one, we do not implement the
    HDF5 / 7.3 interface here.

    # 示例
    >>> from os.path import dirname, join as pjoin
    >>> import scipy.io as sio

    Get the filename for an example .mat file from the tests/data directory.

    >>> data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
    # 构建.mat文件的完整路径，用于后续加载
    mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')

    # 加载.mat文件内容
    mat_contents = sio.loadmat(mat_fname)

    # 结果是一个字典，每个变量对应一个键值对
    >>> sorted(mat_contents.keys())
    ['__globals__', '__header__', '__version__', 'testdouble']
    >>> mat_contents['testdouble']
    array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,
            3.92699082, 4.71238898, 5.49778714, 6.28318531]])

    # 默认情况下，SciPy将MATLAB结构读取为结构化的NumPy数组，dtype字段为'object'类型，
    # 名称对应于MATLAB结构字段名称。通过设置`struct_as_record=False`可禁用此功能。

    # 获取包含MATLAB结构'teststruct'的示例.mat文件的文件名，并加载内容
    matstruct_fname = pjoin(data_dir, 'teststruct_7.4_GLNX86.mat')
    matstruct_contents = sio.loadmat(matstruct_fname)
    teststruct = matstruct_contents['teststruct']

    # 结构化数组的dtype是MATLAB结构的大小，不是任何特定字段中元素的数量。
    # shape默认为2维，除非设置了可选参数`squeeze_me=True`，此时所有长度为1的维度都会被移除。
    >>> teststruct.size
    1
    >>> teststruct.shape
    (1, 1)

    # 获取MATLAB结构中第一个元素的'stringfield'
    >>> teststruct[0, 0]['stringfield']
    array(['Rats live on no evil star.'],
      dtype='<U26')

    # 获取'doublefield'的第一个元素
    >>> teststruct['doublefield'][0, 0]
    array([[ 1.41421356,  2.71828183,  3.14159265]])

    # 加载MATLAB结构，将长度为1的维度压缩掉，并获取'complexfield'的项
    matstruct_squeezed = sio.loadmat(matstruct_fname, squeeze_me=True)
    >>> matstruct_squeezed['teststruct'].shape
    ()
    >>> matstruct_squeezed['teststruct']['complexfield'].shape
    ()
    >>> matstruct_squeezed['teststruct']['complexfield'].item()
    array([ 1.41421356+1.41421356j,  2.71828183+2.71828183j,
        3.14159265+3.14159265j])

    # 从kwargs中弹出'variable_names'，默认为None
    variable_names = kwargs.pop('variable_names', None)

    # 使用_open_file_context打开文件，获取MR和matfile_dict
    with _open_file_context(file_name, appendmat) as f:
        MR, _ = mat_reader_factory(f, **kwargs)
        matfile_dict = MR.get_variables(variable_names)

    # 如果mdict不为None，则更新mdict，否则将mdict设为matfile_dict
    if mdict is not None:
        mdict.update(matfile_dict)
    else:
        mdict = matfile_dict

    # 返回更新后的mdict
    return mdict
# 定义一个函数，用于将给定的字典中的变量保存到 MATLAB 格式的 .mat 文件中
@docfiller
def savemat(file_name, mdict,
            appendmat=True,
            format='5',
            long_field_names=False,
            do_compression=False,
            oned_as='row'):
    """
    Save a dictionary of names and arrays into a MATLAB-style .mat file.

    This saves the array objects in the given dictionary to a MATLAB-
    style .mat file.

    Parameters
    ----------
    file_name : str or file-like object
        Name of the .mat file (.mat extension not needed if ``appendmat ==
        True``).
        Can also pass open file_like object.
    mdict : dict
        Dictionary from which to save matfile variables.
    appendmat : bool, optional
        True (the default) to append the .mat extension to the end of the
        given filename, if not already present.
    format : {'5', '4'}, string, optional
        '5' (the default) for MATLAB 5 and up (to 7.2),
        '4' for MATLAB 4 .mat files.
    long_field_names : bool, optional
        False (the default) - maximum field name length in a structure is
        31 characters which is the documented maximum length.
        True - maximum field name length in a structure is 63 characters
        which works for MATLAB 7.6+.
    do_compression : bool, optional
        Whether or not to compress matrices on write. Default is False.
    oned_as : {'row', 'column'}, optional
        If 'column', write 1-D NumPy arrays as column vectors.
        If 'row', write 1-D NumPy arrays as row vectors.

    Examples
    --------
    >>> from scipy.io import savemat
    >>> import numpy as np
    >>> a = np.arange(20)
    >>> mdic = {"a": a, "label": "experiment"}
    >>> mdic
    {'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19]),
    'label': 'experiment'}
    >>> savemat("matlab_matrix.mat", mdic)
    """
    # 打开文件的上下文管理器，准备写入 MATLAB 格式的 .mat 文件
    with _open_file_context(file_name, appendmat, 'wb') as file_stream:
        # 根据指定的格式选择合适的写入器
        if format == '4':
            # 对于格式为 '4'，使用MatFile4Writer，如果长字段名称选项为True，则报错
            if long_field_names:
                message = "Long field names are not available for version 4 files"
                raise ValueError(message)
            MW = MatFile4Writer(file_stream, oned_as)
        elif format == '5':
            # 对于格式为 '5'，使用MatFile5Writer，并根据参数设置是否压缩、使用Unicode字符串等
            MW = MatFile5Writer(file_stream,
                                do_compression=do_compression,
                                unicode_strings=True,
                                long_field_names=long_field_names,
                                oned_as=oned_as)
        else:
            # 如果格式不是 '4' 或 '5'，则抛出错误
            raise ValueError("Format should be '4' or '5'")
        # 将字典中的变量写入 .mat 文件
        MW.put_variables(mdict)


# 定义一个函数，用于列出 MATLAB 文件中的变量信息
@docfiller
def whosmat(file_name, appendmat=True, **kwargs):
    """
    List variables inside a MATLAB file.

    Parameters
    ----------
    %(file_arg)s
    %(append_arg)s
    %(load_args)s
    %(struct_arg)s

    Returns
    -------

    """
    with _open_file_context(file_name, appendmat) as f:
        # 使用 _open_file_context 函数打开文件，获取文件对象 f
        ML, file_opened = mat_reader_factory(f, **kwargs)
        # 使用 mat_reader_factory 函数读取文件内容，返回 ML 对象和文件打开状态
        variables = ML.list_variables()
        # 调用 ML 对象的 list_variables 方法，获取文件中的变量列表
    return variables
```