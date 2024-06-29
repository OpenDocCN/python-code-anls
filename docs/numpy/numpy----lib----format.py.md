# `.\numpy\numpy\lib\format.py`

```
"""
Binary serialization

NPY format
==========

A simple format for saving numpy arrays to disk with the full
information about them.

The ``.npy`` format is the standard binary file format in NumPy for
persisting a *single* arbitrary NumPy array on disk. The format stores all
of the shape and dtype information necessary to reconstruct the array
correctly even on another machine with a different architecture.
The format is designed to be as simple as possible while achieving
its limited goals.

The ``.npz`` format is the standard format for persisting *multiple* NumPy
arrays on disk. A ``.npz`` file is a zip file containing multiple ``.npy``
files, one for each array.

Capabilities
------------

- Can represent all NumPy arrays including nested record arrays and
  object arrays.

- Represents the data in its native binary form.

- Supports Fortran-contiguous arrays directly.

- Stores all of the necessary information to reconstruct the array
  including shape and dtype on a machine of a different
  architecture.  Both little-endian and big-endian arrays are
  supported, and a file with little-endian numbers will yield
  a little-endian array on any machine reading the file. The
  types are described in terms of their actual sizes. For example,
  if a machine with a 64-bit C "long int" writes out an array with
  "long ints", a reading machine with 32-bit C "long ints" will yield
  an array with 64-bit integers.

- Is straightforward to reverse engineer. Datasets often live longer than
  the programs that created them. A competent developer should be
  able to create a solution in their preferred programming language to
  read most ``.npy`` files that they have been given without much
  documentation.

- Allows memory-mapping of the data. See `open_memmap`.

- Can be read from a filelike stream object instead of an actual file.

- Stores object arrays, i.e. arrays containing elements that are arbitrary
  Python objects. Files with object arrays are not to be mmapable, but
  can be read and written to disk.

Limitations
-----------

- Arbitrary subclasses of numpy.ndarray are not completely preserved.
  Subclasses will be accepted for writing, but only the array data will
  be written out. A regular numpy.ndarray object will be created
  upon reading the file.

.. warning::

  Due to limitations in the interpretation of structured dtypes, dtypes
  with fields with empty names will have the names replaced by 'f0', 'f1',
  etc. Such arrays will not round-trip through the format entirely
  accurately. The data is intact; only the field names will differ. We are
  working on a fix for this. This fix will not require a change in the
  file format. The arrays with such structures can still be saved and
  restored, and the correct dtype may be restored by using the
  ``loadedarray.view(correct_dtype)`` method.

File extensions
---------------

We recommend using the ``.npy`` and ``.npz`` extensions for files saved
"""

# 以上是关于 NPY 和 NPZ 文件格式的详细说明和建议。这段注释描述了 NumPy 序列化的二进制格式，包括其设计目标、功能、限制和文件扩展建议。
# The version 1.0 format of the NumPy .npy file format specification.
# This section describes the structure and contents of .npy files in version 1.0.

# The first 6 bytes of the .npy file are a magic string: exactly ``\\x93NUMPY``.
# This magic string identifies the file as a NumPy .npy format file.

# The next byte (7th byte overall) is an unsigned byte representing the major version number of the file format.
# For version 1.0, this byte would be ``\\x01``.

# The following byte (8th byte overall) is an unsigned byte representing the minor version number of the file format.
# For version 1.0, this byte would be ``\\x00``.

# The next 2 bytes (9th and 10th bytes overall) form a little-endian unsigned short integer,
# which indicates the length of the header data (HEADER_LEN).

# The next HEADER_LEN bytes (starting from the 11th byte overall) constitute the header data,
# describing the array's format. It is an ASCII string that contains a Python literal expression of a dictionary.
# This header is terminated by a newline (``\\n``) and padded with spaces (``\\x20``) for alignment purposes.

# The dictionary in the header contains three keys:
# - "descr": Describes the data type of the array.
# - "fortran_order": Indicates whether the array data is Fortran-contiguous.
# - "shape": Specifies the shape of the array.

# For clarity and consistency, the keys in the dictionary are sorted alphabetically.

# Following the header data is the array data itself. If the data type (`dtype`) of the array contains
# Python objects (`dtype.hasobject` is True), the data is serialized using Python's pickle format.
# Otherwise, the data consists of contiguous bytes of the array, either in C-contiguous or Fortran-contiguous order.

# Consumers of the .npy file can calculate the size of the array data by multiplying the number of elements
# (determined by the shape) by `dtype.itemsize`.

# The version numbering of the .npy file format is independent of the NumPy library version numbering,
# ensuring backward and forward compatibility with the `numpy.io` module.
Format Version 3.0
------------------

This version replaces the ASCII string (which in practice was latin1) with
a utf8-encoded string, so supports structured types with any unicode field
names.

Notes
-----
The ``.npy`` format, including motivation for creating it and a comparison of
alternatives, is described in the
:doc:`"npy-format" NEP <neps:nep-0001-npy-format>`, however details have
evolved with time and this document is more current.

"""
import io
import os
import pickle
import warnings

import numpy
from numpy.lib._utils_impl import drop_metadata


__all__ = []

# 预期的键集合，用于检查数据类型描述字典的完整性
EXPECTED_KEYS = {'descr', 'fortran_order', 'shape'}
# 魔术前缀，用于识别.npy文件的起始标志
MAGIC_PREFIX = b'\x93NUMPY'
# 魔术字符串的长度
MAGIC_LEN = len(MAGIC_PREFIX) + 2
# 数组的对齐方式，默认为64，通常是2的幂，介于16到4096之间
ARRAY_ALIGN = 64
# 用于读取npz文件的缓冲区大小，以字节为单位
BUFFER_SIZE = 2**18
# 允许在64位系统中某个轴向上的地址空间内进行增长
GROWTH_AXIS_MAX_DIGITS = 21  # = len(str(8*2**64-1)) hypothetical int1 dtype

# 版本1.0和2.0之间的区别是头部长度由2字节（H）扩展为4字节（I），以支持大型结构化数组的存储
# 版本信息与对应的头部格式
_header_size_info = {
    (1, 0): ('<H', 'latin1'),
    (2, 0): ('<I', 'latin1'),
    (3, 0): ('<I', 'utf8'),
}

# Python的literal_eval函数在处理大输入时并不安全，因为解析可能会变慢甚至导致解释器崩溃。
# 这是一个任意设置的低限，应该在实践中是安全的。
_MAX_HEADER_SIZE = 10000

def _check_version(version):
    """
    检查给定的文件格式版本是否受支持。

    Parameters
    ----------
    version : tuple
        文件格式的主次版本号组成的元组

    Raises
    ------
    ValueError
        如果版本不是(1,0)，(2,0)或(3,0)中的一个
    """
    if version not in [(1, 0), (2, 0), (3, 0), None]:
        msg = "we only support format version (1,0), (2,0), and (3,0), not %s"
        raise ValueError(msg % (version,))

def magic(major, minor):
    """
    返回给定文件格式版本的魔术字符串。

    Parameters
    ----------
    major : int in [0, 255]
        主版本号，应在0到255之间
    minor : int in [0, 255]
        次版本号，应在0到255之间

    Returns
    -------
    magic : str
        魔术字符串，用于表示文件格式版本

    Raises
    ------
    ValueError
        如果版本号超出范围
    """
    if major < 0 or major > 255:
        raise ValueError("major version must be 0 <= major < 256")
    if minor < 0 or minor > 255:
        raise ValueError("minor version must be 0 <= minor < 256")
    return MAGIC_PREFIX + bytes([major, minor])

def read_magic(fp):
    """
    读取文件中的魔术字符串，获取文件格式的版本信息。

    Parameters
    ----------
    fp : filelike object
        文件对象或类似文件对象

    Returns
    -------
    major : int
        主版本号
    minor : int
        次版本号
    """
    magic_str = _read_bytes(fp, MAGIC_LEN, "magic string")
    if magic_str[:-2] != MAGIC_PREFIX:
        msg = "the magic string is not correct; expected %r, got %r"
        raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
    major, minor = magic_str[-2:]
    return major, minor

def dtype_to_descr(dtype):
    """
    从dtype对象获取可序列化的描述符。

    Parameters
    ----------
    dtype : dtype object
        数据类型对象

    Returns
    -------
    descr : str
        序列化后的描述符字符串
    """
    # .descr属性不能通过dtype()构造函数完全回转
    # 简单类型（如dtype('float32')）有
    `
        """
        a descr which looks like a record array with one field with '' as
        a name. The dtype() constructor interprets this as a request to give
        a default name.  Instead, we construct descriptor that can be passed to
        dtype().
    
        Parameters
        ----------
        dtype : dtype
            The dtype of the array that will be written to disk.
    
        Returns
        -------
        descr : object
            An object that can be passed to `numpy.dtype()` in order to
            replicate the input dtype.
    
        """
        # 注意：drop_metadata 可能不会返回正确的 dtype，例如对于用户自定义的 dtype。在这种情况下，我们下面的代码也会失败。
        new_dtype = drop_metadata(dtype)
        # 如果 drop_metadata 返回的 dtype 与原始的 dtype 不同，发出警告。
        if new_dtype is not dtype:
            warnings.warn("metadata on a dtype is not saved to an npy/npz. "
                          "Use another format (such as pickle) to store it.",
                          UserWarning, stacklevel=2)
        # 如果 dtype 具有字段名，则返回该字段的描述符。
        if dtype.names is not None:
            # 这是一个记录数组。.descr 是合适的。XXX: 像填充字节这样的字段名称为空的部分仍然会被处理。这需要在 dtype() 的 C 实现中修复。
            return dtype.descr
        # 如果 dtype 不是遗留的，并且被认为是用户自定义的 dtype。
        elif not type(dtype)._legacy:
            # 这必须是用户定义的 dtype，因为 numpy 在公共 API 中还没有暴露任何非遗留 dtype。
            #
            # 非遗留 dtype 尚未具有 __array_interface__ 支持。作为一种权宜之计，我们使用 pickle 来保存数组，并且误导性地声称 dtype 是对象类型。
            # 当加载数组时，descriptor 会随数组一起反序列化，并且头部的对象 dtype 会被丢弃。
            #
            # 未来的 NEP 应该定义一种序列化he "
                          "pickle protocol. Loading this file requires "
                          "allow_pickle=True to be set.",
                          UserWarning, stacklevel=2)
            return "|O"
        else:
            # 如果以上条件都不符合，则返回 dtype 的字符串表示形式。
            return dtype.str
def _wrap_header(header, version):
    """
    Takes a stringified header, and attaches the prefix and padding to it
    """
    # 确保版本信息不为空
    assert version is not None
    # 使用指定版本的格式和编码获取格式化字符串和编码方式
    fmt, encoding = _header_size_info[version]
    # 将头部字符串编码为指定编码方式的字节流
    header = header.encode(encoding)
    # 计算头部字符串长度加上一个字节的空位
    hlen = len(header) + 1
    # 计算需要填充的空白长度，使得 MAGIC_LEN、fmt 的结构体大小、hlen 加上 padlen 后能够被 ARRAY_ALIGN 整除
    padlen = ARRAY_ALIGN - ((MAGIC_LEN + struct.calcsize(fmt) + hlen) % ARRAY_ALIGN)
    
    try:
        # 生成包含魔数和头部长度的前缀数据
        header_prefix = magic(*version) + struct.pack(fmt, hlen + padlen)
    except struct.error:
        # 如果生成头部数据时发生结构错误，抛出 ValueError 异常
        msg = "Header length {} too big for version={}".format(hlen, version)
        raise ValueError(msg) from None
    
    # 使用空格和换行符填充头部数据，以便使魔数字符串、头部长度短整型和头部数据都能在 ARRAY_ALIGN 字节边界上对齐。
    # 这样做支持在像 Linux 这样的系统上内存映射对齐为 ARRAY_ALIGN 的数据类型，
    # 其中 mmap() 的偏移量必须是页面对齐的（即文件的开头）。
    return header_prefix + header + b' '*padlen + b'\n'
# 从文件头部读取数组的版本信息，封装了版本选择的逻辑
def _wrap_header_guess_version(header):
    """
    Like `_wrap_header`, but chooses an appropriate version given the contents
    """
    try:
        # 尝试使用 (1, 0) 版本封装头部信息
        return _wrap_header(header, (1, 0))
    except ValueError:
        pass

    try:
        # 尝试使用 (2, 0) 版本封装头部信息
        ret = _wrap_header(header, (2, 0))
    except UnicodeEncodeError:
        pass
    else:
        # 如果成功，给出警告：格式为 2.0 的存储数组只能被 NumPy >= 1.9 读取
        warnings.warn("Stored array in format 2.0. It can only be"
                      "read by NumPy >= 1.9", UserWarning, stacklevel=2)
        return ret

    # 尝试使用 (3, 0) 版本封装头部信息
    header = _wrap_header(header, (3, 0))
    # 给出警告：格式为 3.0 的存储数组只能被 NumPy >= 1.17 读取
    warnings.warn("Stored array in format 3.0. It can only be "
                  "read by NumPy >= 1.17", UserWarning, stacklevel=2)
    return header


def _write_array_header(fp, d, version=None):
    """ Write the header for an array and returns the version used

    Parameters
    ----------
    fp : filelike object
        文件对象，用于写入头部信息
    d : dict
        包含了适合写入文件头部的字符串表示的条目
    version : tuple or None
        版本号，None 表示使用最旧兼容版本。如果提供了具体版本号且格式不支持，则会引发 ValueError。
        默认: None
    """
    header = ["{"]
    for key, value in sorted(d.items()):
        # 在这里需要使用 repr，因为读取时需要 eval
        header.append("'%s': %s, " % (key, repr(value)))
    header.append("}")
    header = "".join(header)

    # 添加一些空余空间，以便可以在原地修改数组头部信息，例如在末尾追加数据时改变数组大小
    shape = d['shape']
    header += " " * ((GROWTH_AXIS_MAX_DIGITS - len(repr(
        shape[-1 if d['fortran_order'] else 0]
    ))) if len(shape) > 0 else 0)

    if version is None:
        # 根据内容推测适合的版本号
        header = _wrap_header_guess_version(header)
    else:
        # 使用指定版本号封装头部信息
        header = _wrap_header(header, version)
    fp.write(header)


def write_array_header_1_0(fp, d):
    """ Write the header for an array using the 1.0 format.

    Parameters
    ----------
    fp : filelike object
        文件对象，用于写入头部信息
    d : dict
        包含了适合写入文件头部的字符串表示的条目
    """
    _write_array_header(fp, d, (1, 0))


def write_array_header_2_0(fp, d):
    """ Write the header for an array using the 2.0 format.
        The 2.0 format allows storing very large structured arrays.

    .. versionadded:: 1.9.0

    Parameters
    ----------
    fp : filelike object
        文件对象，用于写入头部信息
    d : dict
        包含了适合写入文件头部的字符串表示的条目
    """
    _write_array_header(fp, d, (2, 0))


def read_array_header_1_0(fp, max_header_size=_MAX_HEADER_SIZE):
    """
    Read an array header from a filelike object using the 1.0 file format
    version.

    This will leave the file object located just after the header.

    Parameters
    ----------
    fp : filelike object
        文件对象，用于读取头部信息
    max_header_size : int, optional
        最大头部大小限制，默认为 _MAX_HEADER_SIZE
    """
    # fp: 类文件对象
    #     文件对象或类似文件的对象，具有 `.read()` 方法。
    
    # 返回值
    # -------
    # shape: 元组，包含整数
    #     数组的形状。
    # fortran_order: 布尔值
    #     如果数组数据是 C 连续或 Fortran 连续，则将其直接写出。否则，在写出之前将使其连续。
    # dtype: dtype
    #     文件数据的数据类型。
    # max_header_size: 整数，可选
    #     头部的最大允许大小。大型头部可能不安全加载，因此需要显式传递较大的值。
    #     参见 :py:func:`ast.literal_eval()` 获取详细信息。
    
    # 异常
    # ------
    # ValueError
    #     如果数据无效。
    
    """
    通过调用 _read_array_header 函数读取数组头部信息，
    传递文件对象 fp 和版本号 (1, 0)，同时可以指定最大头部大小 max_header_size。
    """
    return _read_array_header(
            fp, version=(1, 0), max_header_size=max_header_size)
# 从给定的文件对象中读取数组头部信息，使用版本为 2.0 的文件格式。
def read_array_header_2_0(fp, max_header_size=_MAX_HEADER_SIZE):
    """
    Read an array header from a filelike object using the 2.0 file format
    version.

    This will leave the file object located just after the header.

    .. versionadded:: 1.9.0

    Parameters
    ----------
    fp : filelike object
        A file object or something with a `.read()` method like a file.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.

    Returns
    -------
    shape : tuple of int
        The shape of the array.
    fortran_order : bool
        The array data will be written out directly if it is either
        C-contiguous or Fortran-contiguous. Otherwise, it will be made
        contiguous before writing it out.
    dtype : dtype
        The dtype of the file's data.

    Raises
    ------
    ValueError
        If the data is invalid.

    """
    # 调用内部函数 _read_array_header，读取数组头部信息
    return _read_array_header(
            fp, version=(2, 0), max_header_size=max_header_size)


# 清理 npz 文件头部字符串中的 'L'，使得 Python 2 生成的头部可以在 Python 3 中读取
def _filter_header(s):
    """Clean up 'L' in npz header ints.

    Cleans up the 'L' in strings representing integers. Needed to allow npz
    headers produced in Python2 to be read in Python3.

    Parameters
    ----------
    s : string
        Npy file header.

    Returns
    -------
    header : str
        Cleaned up header.

    """
    # 导入 tokenize 和 StringIO，用于处理字符串中的 'L'
    import tokenize
    from io import StringIO

    # 生成字符串的 token 流
    tokens = []
    last_token_was_number = False
    for token in tokenize.generate_tokens(StringIO(s).readline):
        token_type = token[0]
        token_string = token[1]
        # 如果上一个 token 是数字且当前 token 是名字且为 'L'，则跳过
        if (last_token_was_number and
                token_type == tokenize.NAME and
                token_string == "L"):
            continue
        else:
            tokens.append(token)
        last_token_was_number = (token_type == tokenize.NUMBER)
    # 重新组合 token 流成为字符串头部
    return tokenize.untokenize(tokens)


def _read_array_header(fp, version, max_header_size=_MAX_HEADER_SIZE):
    """
    see read_array_header_1_0
    """
    # 读取一个无符号的小端 short int，它表示头部的长度
    import ast
    import struct
    hinfo = _header_size_info.get(version)
    if hinfo is None:
        raise ValueError("Invalid version {!r}".format(version))
    hlength_type, encoding = hinfo

    # 读取头部长度的字节流
    hlength_str = _read_bytes(fp, struct.calcsize(hlength_type), "array header length")
    # 解包得到头部的长度值
    header_length = struct.unpack(hlength_type, hlength_str)[0]
    # 读取指定长度的头部数据
    header = _read_bytes(fp, header_length, "array header")
    # 将头部数据解码成字符串
    header = header.decode(encoding)
    # 如果 header 的长度超过了最大允许的 header 大小，抛出 ValueError 异常
    if len(header) > max_header_size:
        raise ValueError(
            f"Header info length ({len(header)}) is large and may not be safe "
            "to load securely.\n"
            "To allow loading, adjust `max_header_size` or fully trust "
            "the `.npy` file using `allow_pickle=True`.\n"
            "For safety against large resource use or crashes, sandboxing "
            "may be necessary.")

    # 将 header 解析为 Python 字典对象 d，使用 ast.literal_eval 函数安全执行
    # header 是一个漂亮打印的字符串表示的 Python 字典，以 ARRAY_ALIGN 字节边界对齐
    # 字典的键是字符串
    try:
        d = ast.literal_eval(header)
    except SyntaxError as e:
        # 如果 header 解析失败，并且版本 <= (2, 0)，尝试使用 _filter_header 进行处理
        if version <= (2, 0):
            header = _filter_header(header)
            try:
                d = ast.literal_eval(header)
            except SyntaxError as e2:
                # 如果第二次解析仍然失败，则抛出详细的 ValueError 异常
                msg = "Cannot parse header: {!r}"
                raise ValueError(msg.format(header)) from e2
            else:
                # 发出警告，说明需要额外的头部解析，因为文件是在 Python 2 上创建的
                warnings.warn(
                    "Reading `.npy` or `.npz` file required additional "
                    "header parsing as it was created on Python 2. Save the "
                    "file again to speed up loading and avoid this warning.",
                    UserWarning, stacklevel=4)
        else:
            # 如果版本大于 (2, 0)，直接抛出详细的 ValueError 异常
            msg = "Cannot parse header: {!r}"
            raise ValueError(msg.format(header)) from e

    # 检查 d 是否为字典类型，如果不是则抛出 ValueError 异常
    if not isinstance(d, dict):
        msg = "Header is not a dictionary: {!r}"
        raise ValueError(msg.format(d))

    # 检查 d 的键集合是否与 EXPECTED_KEYS 一致，如果不一致则抛出 ValueError 异常
    if EXPECTED_KEYS != d.keys():
        keys = sorted(d.keys())
        msg = "Header does not contain the correct keys: {!r}"
        raise ValueError(msg.format(keys))

    # 对 shape、fortran_order 和 descr 进行合理性检查
    # 检查 shape 是否为元组且元素是否全为整数，如果不是则抛出 ValueError 异常
    if (not isinstance(d['shape'], tuple) or
            not all(isinstance(x, int) for x in d['shape'])):
        msg = "shape is not valid: {!r}"
        raise ValueError(msg.format(d['shape']))
    
    # 检查 fortran_order 是否为布尔型，如果不是则抛出 ValueError 异常
    if not isinstance(d['fortran_order'], bool):
        msg = "fortran_order is not a valid bool: {!r}"
        raise ValueError(msg.format(d['fortran_order']))
    
    # 尝试将 descr 转换为有效的 dtype 描述符，如果失败则抛出 ValueError 异常
    try:
        dtype = descr_to_dtype(d['descr'])
    except TypeError as e:
        msg = "descr is not a valid dtype descriptor: {!r}"
        raise ValueError(msg.format(d['descr'])) from e

    # 返回解析后的有效数据：shape、fortran_order 和 dtype
    return d['shape'], d['fortran_order'], dtype
# 检查并确保版本号是有效的
_check_version(version)
# 写入数组的头部信息到文件对象中，根据数组生成头部数据
_write_array_header(fp, header_data_from_array_1_0(array), version)

if array.itemsize == 0:
    # 如果数组的元素字节大小为0，则缓冲区大小设为0
    buffersize = 0
else:
    # 否则，将缓冲区大小设置为16 MiB，以隐藏Python循环开销
    buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)

dtype_class = type(array.dtype)

if array.dtype.hasobject or not dtype_class._legacy:
    # 如果数组包含Python对象或者其dtype不是传统的（legacy），则无法直接写出数据，需要使用pickle进行序列化
    if not allow_pickle:
        # 如果不允许使用pickle，并且数组包含对象，则抛出异常
        if array.dtype.hasobject:
            raise ValueError("Object arrays cannot be saved when "
                             "allow_pickle=False")
        # 如果dtype不是传统的，并且不允许使用pickle，则抛出异常
        if not dtype_class._legacy:
            raise ValueError("User-defined dtypes cannot be saved "
                             "when allow_pickle=False")
    # 如果未提供pickle_kwargs，则初始化为空字典
    if pickle_kwargs is None:
        pickle_kwargs = {}
    # 使用pickle将数组数据写入文件对象
    pickle.dump(array, fp, protocol=4, **pickle_kwargs)
elif array.flags.f_contiguous and not array.flags.c_contiguous:
    # 如果数组是Fortran顺序存储（列优先），且不是C顺序存储（行优先）
    if isfileobj(fp):
        # 如果文件对象是真实的文件对象，则直接将数组的转置写入文件
        array.T.tofile(fp)
    else:
        # 否则，使用nditer迭代器按块写入数组数据到文件对象
        for chunk in numpy.nditer(
                array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                buffersize=buffersize, order='F'):
            fp.write(chunk.tobytes('C'))
    else:
        # 如果文件对象不是普通文件，检查是否是文件对象
        if isfileobj(fp):
            # 如果是文件对象，将数组内容写入文件对象
            array.tofile(fp)
        else:
            # 如果文件对象不是普通文件，迭代数组的每个块并写入文件对象
            for chunk in numpy.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='C'):
                fp.write(chunk.tobytes('C'))
# 从一个NPY文件中读取一个数组

def read_array(fp, allow_pickle=False, pickle_kwargs=None, *,
               max_header_size=_MAX_HEADER_SIZE):
    """
    Read an array from an NPY file.

    Parameters
    ----------
    fp : file_like object
        If this is not a real file object, then this may take extra memory
        and time.
    allow_pickle : bool, optional
        Whether to allow writing pickled data. Default: False

        .. versionchanged:: 1.16.3
            Made default False in response to CVE-2019-6446.

    pickle_kwargs : dict
        Additional keyword arguments to pass to pickle.load. These are only
        useful when loading object arrays saved on Python 2 when using
        Python 3.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.
        This option is ignored when `allow_pickle` is passed.  In that case
        the file is by definition trusted and the limit is unnecessary.

    Returns
    -------
    array : ndarray
        The array from the data on disk.

    Raises
    ------
    ValueError
        If the data is invalid, or allow_pickle=False and the file contains
        an object array.

    """

    # 如果允许使用pickle，则忽略max_header_size限制，因为此时输入被视为完全可信任的
    if allow_pickle:
        max_header_size = 2**64

    # 读取文件的魔数版本号
    version = read_magic(fp)
    # 检查文件版本是否符合要求
    _check_version(version)
    # 读取数组的形状、Fortran顺序和数据类型信息
    shape, fortran_order, dtype = _read_array_header(
            fp, version, max_header_size=max_header_size)

    # 计算数组中元素的总数
    if len(shape) == 0:
        count = 1
    else:
        count = numpy.multiply.reduce(shape, dtype=numpy.int64)

    # 现在读取实际的数据
    if dtype.hasobject:
        # 如果数组包含Python对象，则需要反序列化数据
        if not allow_pickle:
            raise ValueError("Object arrays cannot be loaded when "
                             "allow_pickle=False")
        # 如果pickle_kwargs为None，则初始化为空字典
        if pickle_kwargs is None:
            pickle_kwargs = {}
        try:
            # 使用pickle加载数据
            array = pickle.load(fp, **pickle_kwargs)
        except UnicodeError as err:
            # 如果出现UnicodeError异常，则提供更友好的错误消息
            raise UnicodeError("Unpickling a python object failed: %r\n"
                               "You may need to pass the encoding= option "
                               "to numpy.load" % (err,)) from err
    # 如果不满足以上条件，进入这个分支
    else:
        # 如果传入的文件对象是文件对象（通过isfileobj()函数判断）
        if isfileobj(fp):
            # 可以使用快速的fromfile()函数来读取数据
            array = numpy.fromfile(fp, dtype=dtype, count=count)
        else:
            # 如果不是真正的文件对象，需要以占用大量内存的方式读取数据
            # crc32 模块对于大于 2 ** 32 字节的读取会失败，
            # 打破了从 gzip 流中读取大数据的功能。将读取分块为 BUFFER_SIZE 字节，
            # 以避免问题并减少读取时的内存开销。在非分块情况下，count < max_read_count，
            # 因此只进行一次读取。

            # 使用 np.ndarray 而不是 np.empty，因为后者不能正确实例化零宽度字符串的数据类型；参见
            # https://github.com/numpy/numpy/pull/6430
            array = numpy.ndarray(count, dtype=dtype)

            if dtype.itemsize > 0:
                # 如果 dtype.itemsize == 0，则无需再读取
                max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dtype.itemsize)

                # 按块读取数据
                for i in range(0, count, max_read_count):
                    read_count = min(max_read_count, count - i)
                    read_size = int(read_count * dtype.itemsize)
                    data = _read_bytes(fp, read_size, "array data")
                    array[i:i+read_count] = numpy.frombuffer(data, dtype=dtype,
                                                             count=read_count)

        # 如果需要按 Fortran 顺序重新排列数组
        if fortran_order:
            array.shape = shape[::-1]
            array = array.transpose()
        else:
            # 按指定形状设置数组形状
            array.shape = shape

    # 返回最终的数组
    return array
# 定义一个函数，用于以内存映射方式打开 .npy 文件，并返回内存映射数组对象。
def open_memmap(filename, mode='r+', dtype=None, shape=None,
                fortran_order=False, version=None, *,
                max_header_size=_MAX_HEADER_SIZE):
    """
    Open a .npy file as a memory-mapped array.

    This may be used to read an existing file or create a new one.

    Parameters
    ----------
    filename : str or path-like
        The name of the file on disk.  This may *not* be a file-like
        object.
    mode : str, optional
        The mode in which to open the file; the default is 'r+'.  In
        addition to the standard file modes, 'c' is also accepted to mean
        "copy on write."  See `memmap` for the available mode strings.
    dtype : data-type, optional
        The data type of the array if we are creating a new file in "write"
        mode, if not, `dtype` is ignored.  The default value is None, which
        results in a data-type of `float64`.
    shape : tuple of int
        The shape of the array if we are creating a new file in "write"
        mode, in which case this parameter is required.  Otherwise, this
        parameter is ignored and is thus optional.
    fortran_order : bool, optional
        Whether the array should be Fortran-contiguous (True) or
        C-contiguous (False, the default) if we are creating a new file in
        "write" mode.
    version : tuple of int (major, minor) or None
        If the mode is a "write" mode, then this is the version of the file
        format used to create the file.  None means use the oldest
        supported version that is able to store the data.  Default: None
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.

    Returns
    -------
    marray : memmap
        The memory-mapped array.

    Raises
    ------
    ValueError
        If the data or the mode is invalid.
    OSError
        If the file is not found or cannot be opened correctly.

    See Also
    --------
    numpy.memmap

    """
    # 检查 filename 是否为文件对象，如果是则抛出错误，因为内存映射不能使用现有文件句柄。
    if isfileobj(filename):
        raise ValueError("Filename must be a string or a path-like object."
                         "  Memmap cannot use existing file handles.")
    if 'w' in mode:
        # 如果 'w' 在 mode 中，表示我们正在创建文件，而不是读取它。
        # 检查是否需要创建文件的版本。
        _check_version(version)
        
        # 确保给定的 dtype 是真正的 dtype 对象，而不仅仅是可以解释为 dtype 对象的内容。
        dtype = numpy.dtype(dtype)
        
        # 如果 dtype 包含 Python 对象，则不能进行内存映射。
        if dtype.hasobject:
            msg = "Array can't be memory-mapped: Python objects in dtype."
            raise ValueError(msg)
        
        # 构建描述文件头所需的字典。
        d = dict(
            descr=dtype_to_descr(dtype),  # 将 dtype 转换为描述器
            fortran_order=fortran_order,  # 是否按 Fortran 顺序存储
            shape=shape,  # 数组的形状
        )
        
        # 如果执行到这里，应该可以安全地创建文件。
        with open(os.fspath(filename), mode+'b') as fp:
            _write_array_header(fp, d, version)  # 写入数组头信息
            offset = fp.tell()  # 记录当前文件指针位置
    
    else:
        # 否则，读取文件的头部信息。
        with open(os.fspath(filename), 'rb') as fp:
            version = read_magic(fp)  # 读取文件的魔数（magic number）
            _check_version(version)  # 检查文件版本是否合适

            # 从文件中读取数组的形状、存储顺序和数据类型。
            shape, fortran_order, dtype = _read_array_header(
                    fp, version, max_header_size=max_header_size)
            
            # 如果 dtype 包含 Python 对象，则不能进行内存映射。
            if dtype.hasobject:
                msg = "Array can't be memory-mapped: Python objects in dtype."
                raise ValueError(msg)
            
            offset = fp.tell()  # 记录当前文件指针位置
    
    # 根据 fortran_order 确定数组的存储顺序。
    if fortran_order:
        order = 'F'
    else:
        order = 'C'
    
    # 如果 mode 是 'w+'，需将其修改为 'r+'，因为已经向文件写入数据。
    if mode == 'w+':
        mode = 'r+'
    
    # 创建内存映射数组对象。
    marray = numpy.memmap(filename, dtype=dtype, shape=shape, order=order,
        mode=mode, offset=offset)
    
    return marray
# 从文件对象 `fp` 中读取指定大小 `size` 的数据，直到读取完为止。
# 如果在读取完 `size` 字节之前遇到 EOF，则抛出 ValueError 异常。
# 对于非阻塞对象，仅支持继承自 io 对象的情况。

def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    # 初始化一个空的 bytes 对象，用于存储读取的数据
    data = bytes()
    while True:
        # 对于 io 文件（在 Python3 中是默认的），如果读取到末尾返回 None 或抛出异常
        # 对于 Python2 的文件对象，可能会截断数据，无法处理非阻塞情况
        try:
            # 尝试从文件对象 `fp` 中读取剩余未读取的部分（直到 `size - len(data)` 字节）
            r = fp.read(size - len(data))
            # 将读取到的数据追加到 `data` 中
            data += r
            # 如果读取返回空数据（EOF）或者已经读取了 `size` 字节，停止循环
            if len(r) == 0 or len(data) == size:
                break
        except BlockingIOError:
            pass
    # 检查实际读取的数据长度是否等于指定的 `size`
    if len(data) != size:
        # 如果实际读取长度不等于 `size`，抛出异常，显示预期读取的字节数和实际读取的字节数
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        # 如果读取长度等于 `size`，返回读取到的数据
        return data


def isfileobj(f):
    # 检查对象 `f` 是否是文件对象（FileIO、BufferedReader、BufferedWriter 的实例）
    if not isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter)):
        return False
    try:
        # 尝试获取文件对象的 `fileno()` 方法，如果包装了 BytesIO 等对象可能会引发 OSError 异常
        f.fileno()
        return True
    except OSError:
        return False
```