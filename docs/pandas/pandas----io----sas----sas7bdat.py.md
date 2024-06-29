# `D:\src\scipysrc\pandas\pandas\io\sas\sas7bdat.py`

```
# 从未来导入类型注释
from __future__ import annotations

# 导入日期时间模块
from datetime import datetime
# 导入系统模块
import sys
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入 numpy 库
import numpy as np

# 从 pandas 库的内部字节交换模块中导入函数
from pandas._libs.byteswap import (
    read_double_with_byteswap,
    read_float_with_byteswap,
    read_uint16_with_byteswap,
    read_uint32_with_byteswap,
    read_uint64_with_byteswap,
)
# 从 pandas 库的内部 SAS 模块中导入 Parser 和 get_subheader_index 函数
from pandas._libs.sas import (
    Parser,
    get_subheader_index,
)
# 从 pandas 库的时间序列转换模块中导入 cast_from_unit_vectorized 函数
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
# 从 pandas 库的错误模块中导入 EmptyDataError 异常类
from pandas.errors import EmptyDataError

# 导入 pandas 库，并将其命名为 pd
import pandas as pd
# 从 pandas 库中导入 DataFrame 和 Timestamp 类
from pandas import (
    DataFrame,
    Timestamp,
)

# 从 pandas.io.common 模块中导入 get_handle 函数
from pandas.io.common import get_handle
# 导入 pandas.io.sas.sas_constants 模块，并将其命名为 const
import pandas.io.sas.sas_constants as const
# 从 pandas.io.sas.sasreader 模块中导入 SASReader 类
from pandas.io.sas.sasreader import SASReader

# 如果类型检查开启，则从 pandas._typing 模块中导入指定类型
if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
    )

# 定义 Unix 时间原点为 1970-01-01
_unix_origin = Timestamp("1970-01-01")
# 定义 SAS 时间原点为 1960-01-01
_sas_origin = Timestamp("1960-01-01")


def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    """
    Convert to Timestamp if possible, otherwise to datetime.datetime.
    SAS float64 lacks precision for more than ms resolution so the fit
    to datetime.datetime is ok.

    Parameters
    ----------
    sas_datetimes : {Series, Sequence[float]}
       Dates or datetimes in SAS
    unit : {'d', 's'}
       "d" if the floats represent dates, "s" for datetimes

    Returns
    -------
    Series
       Series of datetime64 dtype or datetime.datetime.
    """
    # 计算时间差，将 SAS 时间转换为 Unix 时间
    td = (_sas_origin - _unix_origin).as_unit("s")
    if unit == "s":
        # 将秒单位的 SAS 时间转换为毫秒，并添加时间差
        millis = cast_from_unit_vectorized(
            sas_datetimes._values, unit="s", out_unit="ms"
        )
        dt64ms = millis.view("M8[ms]") + td
        return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
    else:
        # 将天单位的 SAS 时间转换为秒，并添加时间差
        vals = np.array(sas_datetimes, dtype="M8[D]") + td
        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)


# 列对象，用于表示 SAS 数据文件中的列信息
class _Column:
    col_id: int
    name: str | bytes
    label: str | bytes
    format: str | bytes
    ctype: bytes
    length: int

    def __init__(
        self,
        col_id: int,
        # 当 convert_header_text 为 False 时，这些可能是字节类型
        name: str | bytes,
        label: str | bytes,
        format: str | bytes,
        ctype: bytes,
        length: int,
    ) -> None:
        # 初始化列对象的各个属性
        self.col_id = col_id
        self.name = name
        self.label = label
        self.format = format
        self.ctype = ctype
        self.length = length


# SAS7BDATReader 类，用于读取 SAS7BDAT 格式的 SAS 数据文件
class SAS7BDATReader(SASReader):
    """
    This class represents a SAS data file in SAS7BDAT format.
    """
    # 读取 SAS7BDAT 格式的 SAS 文件。

    Parameters
    ----------
    path_or_buf : str or buffer
        SAS 文件的路径或指向 SAS 文件内容的文件对象。
    index : str or None, optional
        用作索引的列标识符，默认为 None。
    convert_dates : bool, optional
        尝试将日期转换为 Pandas 的 datetime 值，默认为 True。
        注意，某些罕见的 SAS 日期格式可能不受支持。
    blank_missing : bool, optional
        将空字符串转换为缺失值（SAS 使用空格表示缺失的字符变量），默认为 True。
    chunksize : int or None, optional
        返回 SAS7BDATReader 对象以进行迭代，每次返回给定行数的数据块，默认为 None。
    encoding : str or None, optional
        字符串编码，根据 Python 标准编码。当 encoding='infer' 时，尝试从文件头检测编码；
        当 encoding=None 时，保留数据为二进制格式。
    convert_text : bool, optional
        如果为 False，则文本变量保持原始字节形式，默认为 True。
    convert_header_text : bool, optional
        如果为 False，则头部文本（包括列名）保持原始字节形式，默认为 True。
    """

    _int_length: int
    _cached_page: bytes | None

    def __init__(
        self,
        path_or_buf: Union[str, ReadBuffer[bytes]],
        index=None,
        convert_dates: bool = True,
        blank_missing: bool = True,
        chunksize: Optional[int] = None,
        encoding: Optional[str] = None,
        convert_text: bool = True,
        convert_header_text: bool = True,
        compression: CompressionOptions = "infer",
    # 初始化函数，设置对象的各种属性
    def __init__(
        self, index: int, convert_dates: bool, blank_missing: bool,
        chunksize: int, encoding: str, convert_text: bool,
        convert_header_text: bool
    ) -> None:
        # 设置对象的索引属性
        self.index = index
        # 设置是否转换日期的属性
        self.convert_dates = convert_dates
        # 设置是否空白缺失的属性
        self.blank_missing = blank_missing
        # 设置块大小的属性
        self.chunksize = chunksize
        # 设置编码方式的属性
        self.encoding = encoding
        # 设置是否转换文本的属性
        self.convert_text = convert_text
        # 设置是否转换头部文本的属性
        self.convert_header_text = convert_header_text

        # 设置默认编码方式为 latin-1
        self.default_encoding = "latin-1"
        # 设置压缩为空字节串
        self.compression = b""
        # 初始化列名的原始字节串列表
        self.column_names_raw: list[bytes] = []
        # 初始化列名的字符串列表
        self.column_names: list[str | bytes] = []
        # 初始化列格式的字符串列表
        self.column_formats: list[str | bytes] = []
        # 初始化列对象的列表
        self.columns: list[_Column] = []

        # 初始化当前页面数据子标题指针的列表
        self._current_page_data_subheader_pointers: list[tuple[int, int]] = []
        # 初始化缓存页面为空
        self._cached_page = None
        # 初始化列数据长度的列表
        self._column_data_lengths: list[int] = []
        # 初始化列数据偏移量的列表
        self._column_data_offsets: list[int] = []
        # 初始化列类型的字节串列表
        self._column_types: list[bytes] = []

        # 初始化文件中当前行的索引
        self._current_row_in_file_index = 0
        # 初始化页面中当前行的索引
        self._current_row_on_page_index = 0
        # 初始化文件中当前行的索引（重复定义，可能是笔误）

        # 调用 get_handle 函数获取文件句柄，用于读取二进制数据
        self.handles = get_handle(
            path_or_buf, "rb", is_text=False, compression=compression
        )

        # 将文件句柄保存到对象的 _path_or_buf 属性中
        self._path_or_buf = self.handles.handle

        # 定义用于处理子标题的方法列表，按照 const.SASIndex 的顺序排列
        self._subheader_processors = [
            self._process_rowsize_subheader,  # 处理行大小子标题的方法
            self._process_columnsize_subheader,  # 处理列大小子标题的方法
            self._process_subheader_counts,  # 处理子标题计数的方法
            self._process_columntext_subheader,  # 处理列文本子标题的方法
            self._process_columnname_subheader,  # 处理列名子标题的方法
            self._process_columnattributes_subheader,  # 处理列属性子标题的方法
            self._process_format_subheader,  # 处理格式子标题的方法
            self._process_columnlist_subheader,  # 处理列列表子标题的方法
            None,  # 数据子标题暂未实现
        ]

        try:
            # 获取属性信息
            self._get_properties()
            # 解析元数据信息
            self._parse_metadata()
        except Exception:
            # 出现异常时关闭对象
            self.close()
            raise

    # 返回列数据长度的 numpy int64 数组
    def column_data_lengths(self) -> np.ndarray:
        """Return a numpy int64 array of the column data lengths"""
        return np.asarray(self._column_data_lengths, dtype=np.int64)

    # 返回列数据偏移量的 numpy int64 数组
    def column_data_offsets(self) -> np.ndarray:
        """Return a numpy int64 array of the column offsets"""
        return np.asarray(self._column_data_offsets, dtype=np.int64)

    # 返回列类型的 numpy 字符数组，表示为 's'（字符串）或 'd'（双精度）
    def column_types(self) -> np.ndarray:
        """
        Returns a numpy character array of the column types:
           s (string) or d (double)
        """
        return np.asarray(self._column_types, dtype=np.dtype("S1"))

    # 关闭对象的方法
    def close(self) -> None:
        # 调用文件句柄的关闭方法
        self.handles.close()
    def _get_properties(self) -> None:
        # 检查魔数
        self._path_or_buf.seek(0)
        # 从文件或缓冲区读取288字节作为缓存页面
        self._cached_page = self._path_or_buf.read(288)
        if self._cached_page[0 : len(const.magic)] != const.magic:
            # 如果魔数不匹配，则抛出值错误异常（可能不是SAS文件）
            raise ValueError("magic number mismatch (not a SAS file?)")

        # 获取对齐信息
        buf = self._read_bytes(const.align_1_offset, const.align_1_length)
        if buf == const.u64_byte_checker_value:
            # 如果缓冲区内容与64位对齐值匹配，则设置为64位模式
            self.U64 = True
            self._int_length = 8
            self._page_bit_offset = const.page_bit_offset_x64
            self._subheader_pointer_length = const.subheader_pointer_length_x64
        else:
            # 否则设置为32位模式
            self.U64 = False
            self._page_bit_offset = const.page_bit_offset_x86
            self._subheader_pointer_length = const.subheader_pointer_length_x86
            self._int_length = 4
        buf = self._read_bytes(const.align_2_offset, const.align_2_length)
        if buf == const.align_1_checker_value:
            align1 = const.align_2_value
        else:
            align1 = 0

        # 获取字节顺序信息
        buf = self._read_bytes(const.endianness_offset, const.endianness_length)
        if buf == b"\x01":
            # 如果字节顺序标志为1，则设置字节顺序为小端（"<"），并根据系统字节顺序设置需要字节交换标志
            self.byte_order = "<"
            self.need_byteswap = sys.byteorder == "big"
        else:
            # 否则设置字节顺序为大端（">"），并根据系统字节顺序设置需要字节交换标志
            self.byte_order = ">"
            self.need_byteswap = sys.byteorder == "little"

        # 获取编码信息
        buf = self._read_bytes(const.encoding_offset, const.encoding_length)[0]
        if buf in const.encoding_names:
            # 如果编码标志在已知编码名称列表中，则推断出编码名称，并在编码为"infer"时设置编码
            self.inferred_encoding = const.encoding_names[buf]
            if self.encoding == "infer":
                self.encoding = self.inferred_encoding
        else:
            # 否则标记编码为未知（代码值为buf）
            self.inferred_encoding = f"unknown (code={buf})"

        # 时间戳为1960年1月1日的时刻
        epoch = datetime(1960, 1, 1)
        x = self._read_float(
            const.date_created_offset + align1, const.date_created_length
        )
        # 计算创建日期时间戳，并转换为datetime类型
        self.date_created = epoch + pd.to_timedelta(x, unit="s")
        x = self._read_float(
            const.date_modified_offset + align1, const.date_modified_length
        )
        # 计算修改日期时间戳，并转换为datetime类型
        self.date_modified = epoch + pd.to_timedelta(x, unit="s")

        # 读取头部长度
        self.header_length = self._read_uint(
            const.header_size_offset + align1, const.header_size_length
        )

        # 将剩余的头部数据读入缓存页面
        buf = self._path_or_buf.read(self.header_length - 288)
        self._cached_page += buf
        # 检查缓存页面长度是否等于头部长度，若不等则抛出值错误异常
        if len(self._cached_page) != self.header_length:  # type: ignore[arg-type]
            raise ValueError("The SAS7BDAT file appears to be truncated.")

        # 获取页面长度信息
        self._page_length = self._read_uint(
            const.page_size_offset + align1, const.page_size_length
        )
    # 返回下一个数据帧对象，按照指定的行数读取数据
    def __next__(self) -> DataFrame:
        # 调用read方法读取数据帧，行数为self.chunksize或者1（取较大值）
        da = self.read(nrows=self.chunksize or 1)
        # 如果返回的数据帧为空，则关闭当前对象并抛出StopIteration异常
        if da.empty:
            self.close()
            raise StopIteration
        # 返回读取到的数据帧对象
        return da

    # 读取给定宽度的单精度或双精度浮点数
    def _read_float(self, offset: int, width: int) -> float:
        # 断言_cached_page不为None
        assert self._cached_page is not None
        # 根据宽度选择读取相应字节数的浮点数，如果需要进行字节交换则进行处理
        if width == 4:
            return read_float_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        elif width == 8:
            return read_double_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        else:
            # 如果宽度不是4或8，则关闭当前对象并抛出值错误异常
            self.close()
            raise ValueError("invalid float width")

    # 读取给定宽度的单精度无符号整数
    def _read_uint(self, offset: int, width: int) -> int:
        # 断言_cached_page不为None
        assert self._cached_page is not None
        # 根据宽度选择读取相应字节数的无符号整数，如果需要进行字节交换则进行处理
        if width == 1:
            return self._read_bytes(offset, 1)[0]
        elif width == 2:
            return read_uint16_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        elif width == 4:
            return read_uint32_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        elif width == 8:
            return read_uint64_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        else:
            # 如果宽度不是1、2、4或8，则关闭当前对象并抛出值错误异常
            self.close()
            raise ValueError("invalid int width")

    # 从_cached_page中读取指定偏移和长度的字节数据
    def _read_bytes(self, offset: int, length: int):
        # 断言_cached_page不为None
        assert self._cached_page is not None
        # 如果偏移加长度超过_cached_page的长度，则关闭当前对象并抛出值错误异常
        if offset + length > len(self._cached_page):
            self.close()
            raise ValueError("The cached page is too small.")
        # 返回从偏移开始，指定长度的字节数据
        return self._cached_page[offset : offset + length]

    # 解析元数据信息
    def _parse_metadata(self) -> None:
        done = False
        while not done:
            # 从_path_or_buf中读取_page_length长度的数据到_cached_page
            self._cached_page = self._path_or_buf.read(self._page_length)
            # 如果读取到的_cached_page长度小于等于0，则跳出循环
            if len(self._cached_page) <= 0:
                break
            # 如果读取到的_cached_page长度不等于_page_length，则抛出值错误异常
            if len(self._cached_page) != self._page_length:
                raise ValueError("Failed to read a meta data page from the SAS file.")
            # 处理读取到的元数据页
            done = self._process_page_meta()

    # 处理页面元数据
    def _process_page_meta(self) -> bool:
        # 读取当前页面头部信息
        self._read_page_header()
        # 页面类型列表，包括元数据类型和混合类型
        pt = const.page_meta_types + [const.page_amd_type, const.page_mix_type]
        # 如果当前页面类型在pt列表中，则处理页面元数据
        if self._current_page_type in pt:
            self._process_page_metadata()
        # 判断当前页面类型是否为数据页或混合页，或者当前页面数据子标题指针列表不为空
        is_data_page = self._current_page_type == const.page_data_type
        is_mix_page = self._current_page_type == const.page_mix_type
        # 返回判断结果，只要is_data_page、is_mix_page或当前页面数据子标题指针列表不为空，即返回True，否则返回False
        return bool(
            is_data_page
            or is_mix_page
            or self._current_page_data_subheader_pointers != []
        )
    # 读取页面头部信息的方法
    def _read_page_header(self) -> None:
        # 获取页面位偏移量
        bit_offset = self._page_bit_offset
        # 计算页面类型在位偏移上的具体位置
        tx = const.page_type_offset + bit_offset
        # 读取页面类型，并应用位掩码进行处理
        self._current_page_type = (
            self._read_uint(tx, const.page_type_length) & const.page_type_mask2
        )
        # 计算块计数在位偏移上的具体位置，并读取块计数
        tx = const.block_count_offset + bit_offset
        self._current_page_block_count = self._read_uint(tx, const.block_count_length)
        # 计算子头部计数在位偏移上的具体位置，并读取子头部计数
        tx = const.subheader_count_offset + bit_offset
        self._current_page_subheaders_count = self._read_uint(
            tx, const.subheader_count_length
        )

    # 处理页面元数据的方法
    def _process_page_metadata(self) -> None:
        # 获取页面位偏移量
        bit_offset = self._page_bit_offset

        # 遍历每个子头部
        for i in range(self._current_page_subheaders_count):
            # 计算子头部指针数组在位偏移上的具体位置
            offset = const.subheader_pointers_offset + bit_offset
            total_offset = offset + self._subheader_pointer_length * i

            # 读取子头部的偏移量
            subheader_offset = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length

            # 读取子头部的长度
            subheader_length = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length

            # 读取子头部的压缩标志
            subheader_compression = self._read_uint(total_offset, 1)
            total_offset += 1

            # 读取子头部的类型标志
            subheader_type = self._read_uint(total_offset, 1)

            # 如果子头部长度为0或者子头部被截断，则继续下一次循环
            if (
                subheader_length == 0
                or subheader_compression == const.truncated_subheader_id
            ):
                continue

            # 读取子头部的签名字节序列
            subheader_signature = self._read_bytes(subheader_offset, self._int_length)
            # 根据子头部签名获取对应的子头部处理器
            subheader_index = get_subheader_index(subheader_signature)
            subheader_processor = self._subheader_processors[subheader_index]

            # 如果未找到对应的子头部处理器，则进行进一步检查
            if subheader_processor is None:
                f1 = subheader_compression in (const.compressed_subheader_id, 0)
                f2 = subheader_type == const.compressed_subheader_type
                # 如果满足压缩条件，则将子头部指针添加到当前页面数据的子头部指针数组中
                if self.compression and f1 and f2:
                    self._current_page_data_subheader_pointers.append(
                        (subheader_offset, subheader_length)
                    )
                else:
                    # 否则关闭当前处理并引发异常
                    self.close()
                    raise ValueError(
                        f"Unknown subheader signature {subheader_signature}"
                    )
            else:
                # 否则调用对应的子头部处理器进行处理
                subheader_processor(subheader_offset, subheader_length)
    # 处理行大小子标题信息
    def _process_rowsize_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length  # 获取整数长度
        lcs_offset = offset  # 初始化 LCS 偏移量
        lcp_offset = offset  # 初始化 LCP 偏移量
        if self.U64:
            lcs_offset += 682  # 如果是 U64，调整 LCS 偏移量
            lcp_offset += 706  # 如果是 U64，调整 LCP 偏移量
        else:
            lcs_offset += 354  # 如果不是 U64，调整 LCS 偏移量
            lcp_offset += 378  # 如果不是 U64，调整 LCP 偏移量

        # 读取行长度
        self.row_length = self._read_uint(
            offset + const.row_length_offset_multiplier * int_len,
            int_len,
        )
        # 读取行数
        self.row_count = self._read_uint(
            offset + const.row_count_offset_multiplier * int_len,
            int_len,
        )
        # 读取列数（部分1）
        self.col_count_p1 = self._read_uint(
            offset + const.col_count_p1_multiplier * int_len, int_len
        )
        # 读取列数（部分2）
        self.col_count_p2 = self._read_uint(
            offset + const.col_count_p2_multiplier * int_len, int_len
        )
        # 读取混合页的行数
        mx = const.row_count_on_mix_page_offset_multiplier * int_len
        self._mix_page_row_count = self._read_uint(offset + mx, int_len)
        # 读取 LCS
        self._lcs = self._read_uint(lcs_offset, 2)
        # 读取 LCP
        self._lcp = self._read_uint(lcp_offset, 2)

    # 处理列大小子标题信息
    def _process_columnsize_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length  # 获取整数长度
        offset += int_len  # 调整偏移量
        # 读取列数
        self.column_count = self._read_uint(offset, int_len)
        # 如果列数之和与总列数不匹配，输出警告信息
        if self.col_count_p1 + self.col_count_p2 != self.column_count:
            print(
                f"Warning: column count mismatch ({self.col_count_p1} + "
                f"{self.col_count_p2} != {self.column_count})\n"
            )

    # 未知用途
    def _process_subheader_counts(self, offset: int, length: int) -> None:
        pass
    # 将偏移量增加整数长度，用于指向文本块大小字段的位置
    offset += self._int_length
    # 读取文本块大小的无符号整数值
    text_block_size = self._read_uint(offset, const.text_block_size_length)

    # 从指定偏移量读取指定长度的字节数据
    buf = self._read_bytes(offset, text_block_size)
    # 截取并去除字节流末尾的空字节和空字符，作为列名的原始字节序列
    cname_raw = buf[0:text_block_size].rstrip(b"\x00 ")
    # 将处理后的列名原始字节序列添加到列名原始列表中
    self.column_names_raw.append(cname_raw)

    # 如果列名原始列表中只有一个元素
    if len(self.column_names_raw) == 1:
        # 初始化压缩字面量为空字节串
        compression_literal = b""
        # 遍历预定义的压缩字面量列表
        for cl in const.compression_literals:
            # 如果列名原始字节序列中包含当前压缩字面量
            if cl in cname_raw:
                # 设置当前压缩字面量为找到的压缩字面量
                compression_literal = cl
        # 将压缩字面量赋值给数据对象的压缩属性
        self.compression = compression_literal
        # 将偏移量减去整数长度
        offset -= self._int_length

        # 根据数据对象的U64属性选择合适的偏移量值
        offset1 = offset + 16
        if self.U64:
            offset1 += 4

        # 从指定偏移量读取指定长度的字节数据
        buf = self._read_bytes(offset1, self._lcp)
        # 截取并去除字节流末尾的空字节，作为压缩字面量
        compression_literal = buf.rstrip(b"\x00")
        # 如果压缩字面量为空字节串
        if compression_literal == b"":
            # 将数据对象的_lcs属性设置为0
            self._lcs = 0
            # 根据数据对象的U64属性选择合适的偏移量值
            offset1 = offset + 32
            if self.U64:
                offset1 += 4
            # 从指定偏移量读取指定长度的字节数据
            buf = self._read_bytes(offset1, self._lcp)
            # 将读取的字节数据作为创建过程函数名称存储到数据对象的creator_proc属性中
            self.creator_proc = buf[0 : self._lcp]
        # 如果压缩字面量等于预定义的RLE压缩字面量
        elif compression_literal == const.rle_compression:
            # 根据数据对象的U64属性选择合适的偏移量值
            offset1 = offset + 40
            if self.U64:
                offset1 += 4
            # 从指定偏移量读取指定长度的字节数据
            buf = self._read_bytes(offset1, self._lcp)
            # 将读取的字节数据作为创建过程函数名称存储到数据对象的creator_proc属性中
            self.creator_proc = buf[0 : self._lcp]
        # 如果_lcs属性值大于0
        elif self._lcs > 0:
            # 将_lcp属性设置为0
            self._lcp = 0
            # 根据数据对象的U64属性选择合适的偏移量值
            offset1 = offset + 16
            if self.U64:
                offset1 += 4
            # 从指定偏移量读取指定长度的字节数据
            buf = self._read_bytes(offset1, self._lcs)
            # 将读取的字节数据作为创建过程函数名称存储到数据对象的creator_proc属性中

            self.creator_proc = buf[0 : self._lcp]
    # 如果数据对象具有creator_proc属性
    if hasattr(self, "creator_proc"):
        # 使用_convert_header_text方法处理creator_proc属性的文本内容
        self.creator_proc = self._convert_header_text(self.creator_proc)
    # 处理列名子标题的方法，根据偏移量和长度操作
    def _process_columnname_subheader(self, offset: int, length: int) -> None:
        # 获取整数长度
        int_len = self._int_length
        # 偏移量增加整数长度
        offset += int_len
        # 计算列名指针数量
        column_name_pointers_count = (length - 2 * int_len - 12) // 8
        # 遍历列名指针数量
        for i in range(column_name_pointers_count):
            # 计算文本子标题位置
            text_subheader = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_text_subheader_offset
            )
            # 计算列名偏移量位置
            col_name_offset = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_offset_offset
            )
            # 计算列名长度位置
            col_name_length = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_length_offset
            )

            # 读取文本子标题索引
            idx = self._read_uint(
                text_subheader, const.column_name_text_subheader_length
            )
            # 读取列名偏移量
            col_offset = self._read_uint(
                col_name_offset, const.column_name_offset_length
            )
            # 读取列名长度
            col_len = self._read_uint(col_name_length, const.column_name_length_length)

            # 获取列名原始数据
            name_raw = self.column_names_raw[idx]
            # 提取具体列名
            cname = name_raw[col_offset : col_offset + col_len]
            # 将处理后的列名添加到列名列表
            self.column_names.append(self._convert_header_text(cname))

    # 处理列属性子标题的方法，根据偏移量和长度操作
    def _process_columnattributes_subheader(self, offset: int, length: int) -> None:
        # 获取整数长度
        int_len = self._int_length
        # 计算列属性向量数量
        column_attributes_vectors_count = (length - 2 * int_len - 12) // (int_len + 8)
        # 遍历列属性向量数量
        for i in range(column_attributes_vectors_count):
            # 计算列数据偏移量位置
            col_data_offset = (
                offset + int_len + const.column_data_offset_offset + i * (int_len + 8)
            )
            # 计算列数据长度位置
            col_data_len = (
                offset
                + 2 * int_len
                + const.column_data_length_offset
                + i * (int_len + 8)
            )
            # 计算列类型位置
            col_types = (
                offset + 2 * int_len + const.column_type_offset + i * (int_len + 8)
            )

            # 读取列数据偏移量
            x = self._read_uint(col_data_offset, int_len)
            # 将列数据偏移量添加到列数据偏移量列表
            self._column_data_offsets.append(x)

            # 读取列数据长度
            x = self._read_uint(col_data_len, const.column_data_length_length)
            # 将列数据长度添加到列数据长度列表
            self._column_data_lengths.append(x)

            # 读取列类型
            x = self._read_uint(col_types, const.column_type_length)
            # 根据读取的列类型值（1 或 0）添加 'd' 或 's' 到列类型列表
            self._column_types.append(b"d" if x == 1 else b"s")

    # 处理列列表子标题的方法，根据偏移量和长度操作
    def _process_columnlist_subheader(self, offset: int, length: int) -> None:
        # 未知用途，暂时不做处理
        # unknown purpose, do nothing for now
        pass
    # 处理格式子标题的方法，从指定偏移量和长度开始
    def _process_format_subheader(self, offset: int, length: int) -> None:
        # 计算整数长度
        int_len = self._int_length
        # 计算文本子标题格式的偏移量
        text_subheader_format = (
            offset + const.column_format_text_subheader_index_offset + 3 * int_len
        )
        # 计算列格式偏移量
        col_format_offset = offset + const.column_format_offset_offset + 3 * int_len
        # 计算列格式长度
        col_format_len = offset + const.column_format_length_offset + 3 * int_len
        # 计算文本子标题标签的偏移量
        text_subheader_label = (
            offset + const.column_label_text_subheader_index_offset + 3 * int_len
        )
        # 计算列标签偏移量
        col_label_offset = offset + const.column_label_offset_offset + 3 * int_len
        # 计算列标签长度
        col_label_len = offset + const.column_label_length_offset + 3 * int_len

        # 读取文本子标题格式索引
        x = self._read_uint(
            text_subheader_format, const.column_format_text_subheader_index_length
        )
        # 确保索引不超过列名列表的长度
        format_idx = min(x, len(self.column_names_raw) - 1)

        # 读取列格式起始位置
        format_start = self._read_uint(
            col_format_offset, const.column_format_offset_length
        )
        # 读取列格式长度
        format_len = self._read_uint(col_format_len, const.column_format_length_length)

        # 读取文本子标题标签索引
        label_idx = self._read_uint(
            text_subheader_label, const.column_label_text_subheader_index_length
        )
        # 确保索引不超过列名列表的长度
        label_idx = min(label_idx, len(self.column_names_raw) - 1)

        # 读取列标签起始位置
        label_start = self._read_uint(
            col_label_offset, const.column_label_offset_length
        )
        # 读取列标签长度
        label_len = self._read_uint(col_label_len, const.column_label_length_length)

        # 获取列标签的名称
        label_names = self.column_names_raw[label_idx]
        # 转换列标签的文本
        column_label = self._convert_header_text(
            label_names[label_start : label_start + label_len]
        )
        # 获取列格式的名称
        format_names = self.column_names_raw[format_idx]
        # 转换列格式的文本
        column_format = self._convert_header_text(
            format_names[format_start : format_start + format_len]
        )
        # 当前列的序号
        current_column_number = len(self.columns)

        # 创建列对象
        col = _Column(
            current_column_number,
            self.column_names[current_column_number],
            column_label,
            column_format,
            self._column_types[current_column_number],
            self._column_data_lengths[current_column_number],
        )

        # 将列格式添加到列表中
        self.column_formats.append(column_format)
        # 将列对象添加到列列表中
        self.columns.append(col)
    # 读取数据，返回一个 DataFrame 对象
    def read(self, nrows: int | None = None) -> DataFrame:
        # 如果未指定行数且存在分块大小，则设定为分块大小
        if (nrows is None) and (self.chunksize is not None):
            nrows = self.chunksize
        # 否则，如果未指定行数，则设定为文件中的总行数
        elif nrows is None:
            nrows = self.row_count

        # 如果列类型列表为空，关闭文件并抛出异常
        if len(self._column_types) == 0:
            self.close()
            raise EmptyDataError("No columns to parse from file")

        # 如果指定的行数大于0且当前文件中的行索引超过文件总行数，返回一个空的 DataFrame
        if nrows > 0 and self._current_row_in_file_index >= self.row_count:
            return DataFrame()

        # 将行数限制为当前文件行数与文件中当前行索引之差的较小值
        nrows = min(nrows, self.row_count - self._current_row_in_file_index)

        # 统计列类型列表中 b"d" 和 b"s" 的数量
        nd = self._column_types.count(b"d")
        ns = self._column_types.count(b"s")

        # 创建字符串数据的空数组
        self._string_chunk = np.empty((ns, nrows), dtype=object)
        # 创建字节数据的空数组
        self._byte_chunk = np.zeros((nd, 8 * nrows), dtype=np.uint8)

        # 初始化当前数据块中的行索引
        self._current_row_in_chunk_index = 0
        # 创建解析器对象，并读取指定行数的数据
        p = Parser(self)
        p.read(nrows)

        # 将数据块转换为 DataFrame 对象
        rslt = self._chunk_to_dataframe()
        # 如果存在索引列，则将结果 DataFrame 设置索引
        if self.index is not None:
            rslt = rslt.set_index(self.index)

        # 返回结果 DataFrame 对象
        return rslt

    # 读取文件的下一个页面数据
    def _read_next_page(self):
        # 清空当前页面数据的子标题指针列表
        self._current_page_data_subheader_pointers = []
        # 从路径或缓冲区中读取指定页面长度的数据
        self._cached_page = self._path_or_buf.read(self._page_length)
        # 如果读取的页面数据长度小于等于0，则返回 True
        if len(self._cached_page) <= 0:
            return True
        # 如果读取的页面数据长度与指定页面长度不相等，则关闭文件并抛出异常
        elif len(self._cached_page) != self._page_length:
            self.close()
            msg = (
                "failed to read complete page from file (read "
                f"{len(self._cached_page):d} of {self._page_length:d} bytes)"
            )
            raise ValueError(msg)

        # 读取页面头部信息
        self._read_page_header()
        # 如果当前页面类型属于常量中的页面元数据类型，则处理页面元数据
        if self._current_page_type in const.page_meta_types:
            self._process_page_metadata()

        # 如果当前页面类型不属于常量中的页面元数据类型或数据类型，则继续读取下一个页面
        if self._current_page_type not in const.page_meta_types + [
            const.page_data_type,
            const.page_mix_type,
        ]:
            return self._read_next_page()

        # 返回 False 表示未读取完整页面数据
        return False
    # 将当前数据块转换为 DataFrame 格式的方法
    def _chunk_to_dataframe(self) -> DataFrame:
        # 获取当前数据块在文件中的行索引范围
        n = self._current_row_in_chunk_index
        m = self._current_row_in_file_index
        ix = range(m - n, m)
        rslt = {}  # 初始化结果字典

        js, jb = 0, 0  # 初始化字符串和数值列索引

        # 遍历所有列
        for j in range(self.column_count):
            name = self.column_names[j]  # 获取列名

            # 处理数值列
            if self._column_types[j] == b"d":
                # 从字节块中读取数值列数据，并转换为指定格式
                col_arr = self._byte_chunk[jb, :].view(dtype=self.byte_order + "d")
                rslt[name] = pd.Series(col_arr, dtype=np.float64, index=ix, copy=False)

                # 如果需要转换日期
                if self.convert_dates:
                    # 根据列格式转换日期时间
                    if self.column_formats[j] in const.sas_date_formats:
                        rslt[name] = _convert_datetimes(rslt[name], "d")
                    elif self.column_formats[j] in const.sas_datetime_formats:
                        rslt[name] = _convert_datetimes(rslt[name], "s")
                jb += 1

            # 处理字符串列
            elif self._column_types[j] == b"s":
                # 从字节块中读取字符串列数据，并解码
                rslt[name] = pd.Series(self._string_chunk[js, :], index=ix, copy=False)

                # 如果需要转换文本且有编码方式
                if self.convert_text and (self.encoding is not None):
                    rslt[name] = self._decode_string(rslt[name].str)
                js += 1

            # 处理未知列类型
            else:
                # 关闭数据源并引发错误
                self.close()
                raise ValueError(f"unknown column type {self._column_types[j]!r}")

        # 构建 DataFrame 对象并返回，使用列名和行索引
        df = DataFrame(rslt, columns=self.column_names, index=ix, copy=False)
        return df

    # 解码字节为字符串的方法
    def _decode_string(self, b):
        return b.decode(self.encoding or self.default_encoding)

    # 根据需要转换标题文本的方法，返回字符串或字节码
    def _convert_header_text(self, b: bytes) -> str | bytes:
        if self.convert_header_text:
            return self._decode_string(b)  # 如果需要转换，则解码为字符串
        else:
            return b  # 否则直接返回原始字节码
```