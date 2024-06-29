# `D:\src\scipysrc\pandas\pandas\io\sas\sas_xport.py`

```
"""
Read a SAS XPort format file into a Pandas DataFrame.

Based on code from Jack Cushman (github.com/jcushman/xport).

The file format is defined here:

https://support.sas.com/content/dam/SAS/support/en/technical-papers/record-layout-of-a-sas-version-5-or-6-data-set-in-sas-transport-xport-format.pdf
"""

# 导入必要的库和模块
from __future__ import annotations

from datetime import datetime  # 导入 datetime 模块中的 datetime 类
import struct  # 导入 struct 模块，用于处理二进制数据的解析和打包
from typing import TYPE_CHECKING  # 导入 TYPE_CHECKING 用于类型检查的模块
import warnings  # 导入 warnings 模块，用于发出警告

import numpy as np  # 导入 NumPy 库
# 导入 Pandas 库，使用 pandas.util._decorators 的 Appender 装饰器
from pandas.util._decorators import Appender
# 导入 Pandas 库，使用 pandas.util._exceptions 的 find_stack_level 函数
from pandas.util._exceptions import find_stack_level

import pandas as pd  # 导入 Pandas 库
# 从 pandas.io.common 模块中导入 get_handle 函数
from pandas.io.common import get_handle
# 从 pandas.io.sas.sasreader 模块中导入 SASReader 类
from pandas.io.sas.sasreader import SASReader

if TYPE_CHECKING:
    # 如果 TYPE_CHECKING 为真，则导入以下类型
    from pandas._typing import (
        CompressionOptions,  # 压缩选项
        DatetimeNaTType,  # DatetimeNaT 类型
        FilePath,  # 文件路径类型
        ReadBuffer,  # 读取缓冲区类型
    )

_correct_line1 = (
    "HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"
    "000000000000000000000000000000  "
)
_correct_header1 = (
    "HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000"
)
_correct_header2 = (
    "HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!"
    "000000000000000000000000000000  "
)
_correct_obs_header = (
    "HEADER RECORD*******OBS     HEADER RECORD!!!!!!!"
    "000000000000000000000000000000  "
)
_fieldkeys = [
    "ntype",  # 类型
    "nhfun",  # hfun
    "field_length",  # 字段长度
    "nvar0",  # var0
    "name",  # 名称
    "label",  # 标签
    "nform",  # form
    "nfl",  # fl
    "num_decimals",  # 小数位数
    "nfj",  # fj
    "nfill",  # fill
    "niform",  # iform
    "nifl",  # ifl
    "nifd",  # ifd
    "npos",  # pos
    "_",  # 占位符
]

_base_params_doc = """\
Parameters
----------
filepath_or_buffer : str or file-like object
    Path to SAS file or object implementing binary read method."""
_params2_doc = """\
index : identifier of index column
    Identifier of column that should be used as index of the DataFrame.
encoding : str
    Encoding for text data.
chunksize : int
    Read file `chunksize` lines at a time, returns iterator."""
_format_params_doc = """\
format : str
    File format, only `xport` is currently supported."""
_iterator_doc = """\
iterator : bool, default False
    Return XportReader object for reading file incrementally."""

# 函数文档字符串，描述了 read_sas 函数的使用方法和返回值
_read_sas_doc = f"""Read a SAS file into a DataFrame.

{_base_params_doc}
{_format_params_doc}
{_params2_doc}
{_iterator_doc}

Returns
-------
DataFrame or XportReader

Examples
--------
Read a SAS Xport file:

>>> df = pd.read_sas('filename.XPT')

Read a Xport file in 10,000 line chunks:

>>> itr = pd.read_sas('filename.XPT', chunksize=10000)
>>> for chunk in itr:
>>>     do_something(chunk)

"""

# XportReader 类的文档字符串，描述了其参数和属性
_xport_reader_doc = f"""\
Class for reading SAS Xport files.

{_base_params_doc}
{_params2_doc}

Attributes
----------
member_info : list
    Contains information about the file
fields : list
    Contains information about the variables in the file
"""

# read 方法的文档字符串，描述了其参数和返回值
_read_method_doc = """\
Read observations from SAS Xport file, returning as data frame.

Parameters
----------
nrows : int
    Number of rows to read from data file; if None, read whole
    file.

Returns
-------
"""
# A DataFrame.
"""

def _parse_date(datestr: str) -> DatetimeNaTType:
    """Given a date in xport format, return Python date."""
    try:
        # e.g. "16FEB11:10:07:55"
        # 将给定的日期字符串解析为 Python 的日期对象
        return datetime.strptime(datestr, "%d%b%y:%H:%M:%S")
    except ValueError:
        # 如果解析失败，则返回 Pandas 提供的 NaT（Not a Time）表示缺失日期
        return pd.NaT


def _split_line(s: str, parts):
    """
    Parameters
    ----------
    s: str
        Fixed-length string to split
    parts: list of (name, length) pairs
        Used to break up string, name '_' will be filtered from output.

    Returns
    -------
    Dict of name:contents of string at given location.
    """
    out = {}
    start = 0
    for name, length in parts:
        # 按照指定长度分割固定长度的字符串 s，并去除两端的空格，生成字典
        out[name] = s[start : start + length].strip()
        start += length
    del out["_"]  # 删除字典中键为 "_" 的项
    return out


def _handle_truncated_float_vec(vec, nbytes):
    # This feature is not well documented, but some SAS XPORT files
    # have 2-7 byte "truncated" floats.  To read these truncated
    # floats, pad them with zeros on the right to make 8 byte floats.
    #
    # References:
    # https://github.com/jcushman/xport/pull/3
    # The R "foreign" library

    if nbytes != 8:
        # 将长度为 nbytes 的向量 vec 扩展为长度为 8 的字节数组，以便处理截断的浮点数
        vec1 = np.zeros(len(vec), np.dtype("S8"))
        dtype = np.dtype(f"S{nbytes},S{8 - nbytes}")
        vec2 = vec1.view(dtype=dtype)
        vec2["f0"] = vec
        return vec2

    return vec


def _parse_float_vec(vec):
    """
    Parse a vector of float values representing IBM 8 byte floats into
    native 8 byte floats.
    """
    dtype = np.dtype(">u4,>u4")
    vec1 = vec.view(dtype=dtype)
    xport1 = vec1["f0"]
    xport2 = vec1["f1"]

    # Start by setting first half of ieee number to first half of IBM
    # number sans exponent
    ieee1 = xport1 & 0x00FFFFFF

    # The fraction bit to the left of the binary point in the ieee
    # format was set and the number was shifted 0, 1, 2, or 3
    # places. This will tell us how to adjust the ibm exponent to be a
    # power of 2 ieee exponent and how to shift the fraction bits to
    # restore the correct magnitude.
    shift = np.zeros(len(vec), dtype=np.uint8)
    shift[np.where(xport1 & 0x00200000)] = 1
    shift[np.where(xport1 & 0x00400000)] = 2
    shift[np.where(xport1 & 0x00800000)] = 3

    # shift the ieee number down the correct number of places then
    # set the second half of the ieee number to be the second half
    # of the ibm number shifted appropriately, ored with the bits
    # from the first half that would have been shifted in if we
    # could shift a double. All we are worried about are the low
    # order 3 bits of the first half since we're only shifting by
    # 1, 2, or 3.
    ieee1 >>= shift
    ieee2 = (xport2 >> shift) | ((xport1 & 0x00000007) << (29 + (3 - shift)))

    # clear the 1 bit to the left of the binary point
    ieee1 &= 0xFFEFFFFF

    # set the exponent of the ieee number to be the actual exponent
    # plus the shift count + 1023. Or this into the first half of the
    # ieee1 represents the first part of the IEEE floating-point number in IBM format
    # The IBM exponent is excess 64, but is adjusted by 65 due to a conversion adjustment
    # During conversion, the exponent is incremented by 1 and the fraction bits are shifted 4 positions right of the radix point.
    # (The addition of >> 24 is necessary because C treats & 0x7f as 0x7f000000, which Python does not)
    ieee1 |= ((((((xport1 >> 24) & 0x7F) - 65) << 2) + shift + 1023) << 20) | (
        xport1 & 0x80000000
    )

    # Create a numpy array 'ieee' with a structured dtype of two big-endian unsigned 4-byte integers
    ieee = np.empty((len(ieee1),), dtype=">u4,>u4")
    # Assign 'ieee1' to the 'f0' field and 'ieee2' to the 'f1' field of 'ieee'
    ieee["f0"] = ieee1
    ieee["f1"] = ieee2
    # Convert the structured array 'ieee' into a view of a big-endian double precision floating-point dtype
    ieee = ieee.view(dtype=">f8")
    # Convert 'ieee' to a standard float64 array
    ieee = ieee.astype("f8")

    # Return the resulting IEEE floating-point numbers
    return ieee
class XportReader(SASReader):
    __doc__ = _xport_reader_doc  # 设置类的文档字符串为预定义的文档字符串变量

    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        index=None,
        encoding: str | None = "ISO-8859-1",
        chunksize: int | None = None,
        compression: CompressionOptions = "infer",
    ) -> None:
        self._encoding = encoding  # 初始化编码方式
        self._lines_read = 0  # 初始化已读取的行数
        self._index = index  # 初始化索引
        self._chunksize = chunksize  # 初始化块大小

        self.handles = get_handle(
            filepath_or_buffer,
            "rb",
            encoding=encoding,
            is_text=False,
            compression=compression,
        )  # 获取处理文件或缓冲区的句柄
        self.filepath_or_buffer = self.handles.handle  # 获取文件路径或缓冲区的处理句柄

        try:
            self._read_header()  # 尝试读取文件头部信息
        except Exception:
            self.close()  # 如果出现异常，关闭处理句柄
            raise

    def close(self) -> None:
        self.handles.close()  # 关闭处理句柄

    def _get_row(self):
        return self.filepath_or_buffer.read(80).decode()  # 读取文件或缓冲区的下一行，解码为字符串

    def __next__(self) -> pd.DataFrame:
        return self.read(nrows=self._chunksize or 1)  # 返回下一个数据块作为 Pandas 的 DataFrame

    def _record_count(self) -> int:
        """
        Get number of records in file.

        This is maybe suboptimal because we have to seek to the end of
        the file.

        Side effect: returns file position to record_start.
        """
        self.filepath_or_buffer.seek(0, 2)  # 将文件或缓冲区指针移动到末尾
        total_records_length = self.filepath_or_buffer.tell() - self.record_start  # 计算记录的总长度

        if total_records_length % 80 != 0:
            warnings.warn(
                "xport file may be corrupted.",
                stacklevel=find_stack_level(),
            )  # 如果总记录长度不是80的倍数，发出警告可能是损坏的文件

        if self.record_length > 80:
            self.filepath_or_buffer.seek(self.record_start)  # 如果记录长度大于80，将文件或缓冲区指针移动到记录开始位置
            return total_records_length // self.record_length  # 返回记录总数

        self.filepath_or_buffer.seek(-80, 2)  # 否则，将文件或缓冲区指针移动到倒数第80个字节
        last_card_bytes = self.filepath_or_buffer.read(80)  # 读取最后一张卡片的字节数据
        last_card = np.frombuffer(last_card_bytes, dtype=np.uint64)  # 将字节数据转换为 numpy 数组

        # 8 byte blank
        ix = np.flatnonzero(last_card == 2314885530818453536)  # 查找数组中等于给定值的索引位置

        if len(ix) == 0:
            tail_pad = 0
        else:
            tail_pad = 8 * len(ix)  # 计算尾部填充的字节数

        self.filepath_or_buffer.seek(self.record_start)  # 将文件或缓冲区指针移动到记录开始位置

        return (total_records_length - tail_pad) // self.record_length  # 返回记录总数

    def get_chunk(self, size: int | None = None) -> pd.DataFrame:
        """
        Reads lines from Xport file and returns as dataframe

        Parameters
        ----------
        size : int, defaults to None
            Number of lines to read.  If None, reads whole file.

        Returns
        -------
        DataFrame
        """
        if size is None:
            size = self._chunksize  # 如果未指定大小，使用初始化时的块大小
        return self.read(nrows=size)  # 返回读取的数据块作为 Pandas 的 DataFrame
    # 定义一个方法，用于检测缺失的双精度值
    def _missing_double(self, vec):
        # 将输入向量视为一个结构化数组，每个元素包含4个字段：每个字段的数据类型分别为：无符号字节、无符号字节、无符号16位整数和无符号32位整数
        v = vec.view(dtype="u1,u1,u2,u4")
        # 计算缺失值条件：所有字段f1、f2、f3都等于0
        miss = (v["f1"] == 0) & (v["f2"] == 0) & (v["f3"] == 0)
        # 计算第二个缺失值条件：字段f0的范围在0x41到0x5A之间、等于0x5F或等于0x2E
        miss1 = (
            ((v["f0"] >= 0x41) & (v["f0"] <= 0x5A))
            | (v["f0"] == 0x5F)
            | (v["f0"] == 0x2E)
        )
        # 合并两个条件，得到最终的缺失值标记
        miss &= miss1
        return miss

    # 使用装饰器将方法添加到类中，文档字符串来自_read_method_doc
    @Appender(_read_method_doc)
    # 定义一个读取方法，返回一个Pandas数据框，可选参数nrows指定要读取的行数，默认为全部行数
    def read(self, nrows: int | None = None) -> pd.DataFrame:
        # 如果未指定要读取的行数，默认读取全部行数self.nobs
        if nrows is None:
            nrows = self.nobs

        # 计算要读取的行数read_lines，最大不超过剩余未读行数self.nobs - self._lines_read
        read_lines = min(nrows, self.nobs - self._lines_read)
        # 计算要读取的字节数read_len，每行数据长度乘以要读取的行数read_lines
        read_len = read_lines * self.record_length
        # 如果读取长度小于等于0，关闭资源并引发StopIteration异常
        if read_len <= 0:
            self.close()
            raise StopIteration
        # 从文件路径或缓冲区中读取指定长度的原始数据raw
        raw = self.filepath_or_buffer.read(read_len)
        # 使用NumPy从缓冲区中读取数据，按照指定的数据类型_dtype解析，读取行数为read_lines
        data = np.frombuffer(raw, dtype=self._dtype, count=read_lines)

        # 初始化一个空字典，用于存储DataFrame的列数据
        df_data = {}
        # 遍历所有列索引j及其对应的列名x
        for j, x in enumerate(self.columns):
            # 获取当前列的数据向量vec
            vec = data["s" + str(j)]
            # 获取当前列的数据类型ntype
            ntype = self.fields[j]["ntype"]
            # 如果数据类型为"numeric"
            if ntype == "numeric":
                # 处理截断的浮点数向量，根据字段长度self.fields[j]["field_length"]
                vec = _handle_truncated_float_vec(vec, self.fields[j]["field_length"])
                # 检测缺失的双精度值，生成缺失值标记miss
                miss = self._missing_double(vec)
                # 解析浮点数向量，得到浮点数值v
                v = _parse_float_vec(vec)
                # 将缺失的位置设置为NaN
                v[miss] = np.nan
            # 如果数据类型为"char"
            elif self.fields[j]["ntype"] == "char":
                # 去除每个字符元素的末尾空格，生成字符串列表v
                v = [y.rstrip() for y in vec]
                # 如果有编码方式_encoding，使用指定编码解码字符串列表v
                if self._encoding is not None:
                    v = [y.decode(self._encoding) for y in v]

            # 将列名x及其对应的数据v更新到df_data字典中
            df_data.update({x: v})
        # 根据df_data创建一个Pandas数据框df

        # 如果索引_index为None，设置默认数值索引
        if self._index is None:
            df.index = pd.Index(range(self._lines_read, self._lines_read + read_lines))
        # 否则使用索引_index设置数据框的索引
        else:
            df = df.set_index(self._index)

        # 更新已读行数_lines_read
        self._lines_read += read_lines

        # 返回创建的数据框df
        return df
```