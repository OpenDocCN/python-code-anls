# `D:\src\scipysrc\pandas\pandas\io\formats\format.py`

```
"""
Internal module for formatting output data in csv, html, xml,
and latex files. This module also applies to display formatting.
"""

# 从未来版本导入注解模块，用于类型提示
from __future__ import annotations

# 导入抽象基类中的类型
from collections.abc import (
    Callable,        # 可调用对象类型
    Generator,       # 生成器类型
    Hashable,        # 可散列对象类型
    Mapping,         # 映射类型
    Sequence,        # 序列类型
)

# 导入上下文管理器模块
from contextlib import contextmanager

# 导入CSV模块中的QUOTE_NONE常量
from csv import QUOTE_NONE

# 导入Decimal类
from decimal import Decimal

# 导入偏函数模块
from functools import partial

# 导入字符串IO模块
from io import StringIO

# 导入数学模块
import math

# 导入正则表达式模块
import re

# 导入终端尺寸获取函数
from shutil import get_terminal_size

# 导入类型提示相关模块
from typing import (
    TYPE_CHECKING,   # 类型检查标志
    Any,             # 任意类型
    Final,           # 最终变量类型
    cast,            # 类型转换函数
)

# 导入NumPy库
import numpy as np

# 从pandas配置模块中导入获取和设置选项的函数
from pandas._config.config import (
    get_option,      # 获取配置选项值的函数
    set_option,      # 设置配置选项值的函数
)

# 从pandas库中导入C语言实现的函数和数据结构
from pandas._libs import lib

# 从pandas缺失值处理模块中导入NA常量
from pandas._libs.missing import NA

# 从pandas时间序列模块中导入时间相关类型
from pandas._libs.tslibs import (
    NaT,             # 不确定或缺失的时间戳
    Timedelta,       # 时间差类型
    Timestamp,       # 时间戳类型
)

# 从pandas时间序列模块中导入NaTType常量
from pandas._libs.tslibs.nattype import NaTType

# 从pandas核心数据类型模块中导入通用数据类型判断函数
from pandas.core.dtypes.common import (
    is_complex_dtype,     # 是否为复数类型判断函数
    is_float,             # 是否为浮点数类型判断函数
    is_integer,           # 是否为整数类型判断函数
    is_list_like,         # 是否为列表类型判断函数
    is_numeric_dtype,     # 是否为数值类型判断函数
    is_scalar,            # 是否为标量类型判断函数
)

# 从pandas核心数据类型模块中导入具体数据类型
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,     # 分类数据类型
    DatetimeTZDtype,      # 带时区的日期时间类型
    ExtensionDtype,       # 扩展数据类型
)

# 从pandas核心数据类型模块中导入缺失值处理函数
from pandas.core.dtypes.missing import (
    isna,                 # 是否为缺失值判断函数
    notna,                # 是否为非缺失值判断函数
)

# 从pandas核心数组模块中导入具体数组类型
from pandas.core.arrays import (
    Categorical,          # 分类数据数组类型
    DatetimeArray,        # 日期时间数组类型
    ExtensionArray,       # 扩展数组类型
    TimedeltaArray,       # 时间差数组类型
)

# 从pandas核心数组字符串类型模块中导入StringDtype类型
from pandas.core.arrays.string_ import StringDtype

# 从pandas核心基类模块中导入PandasObject类
from pandas.core.base import PandasObject

# 导入pandas核心公共函数模块
import pandas.core.common as com

# 从pandas核心索引模块中导入具体索引类型
from pandas.core.indexes.api import (
    Index,                 # 基本索引类型
    MultiIndex,            # 多重索引类型
    PeriodIndex,           # 周期索引类型
    ensure_index,          # 确保索引函数
)

# 从pandas核心日期时间索引模块中导入具体索引类型
from pandas.core.indexes.datetimes import DatetimeIndex

# 从pandas核心时间差索引模块中导入具体索引类型
from pandas.core.indexes.timedeltas import TimedeltaIndex

# 从pandas核心重塑模块中导入连接函数
from pandas.core.reshape.concat import concat

# 从pandas IO模块中导入常用函数
from pandas.io.common import (
    check_parent_directory,   # 检查父目录是否存在函数
    stringify_path,           # 转换路径为字符串函数
)

# 从pandas IO格式化模块中导入打印函数
from pandas.io.formats import printing

# 如果是类型检查模式，则进一步导入类型相关的类和函数
if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,           # 数组类型
        Axes,                # 轴类型
        ColspaceArgType,     # 列空间参数类型
        ColspaceType,        # 列空间类型
        CompressionOptions,  # 压缩选项类型
        FilePath,            # 文件路径类型
        FloatFormatType,     # 浮点格式类型
        FormattersType,      # 格式化器类型
        IndexLabel,          # 索引标签类型
        SequenceNotStr,      # 非字符串序列类型
        StorageOptions,      # 存储选项类型
        WriteBuffer,         # 写入缓冲类型
    )

    from pandas import (
        DataFrame,           # 数据框类型
        Series,              # 系列类型
    )
# 定义常见的文档字符串模板，描述了函数的各种参数及其默认值
common_docstring: Final = """
        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        columns : array-like, optional, default None
            The subset of columns to write. Writes all columns by default.
        col_space : %(col_space_type)s, optional
            %(col_space)s.
        header : %(header_type)s, optional
            %(header)s.
        index : bool, optional, default True
            Whether to print index (row) labels.
        na_rep : str, optional, default 'NaN'
            String representation of ``NaN`` to use.
        formatters : list, tuple or dict of one-param. functions, optional
            Formatter functions to apply to columns' elements by position or
            name.
            The result of each function must be a unicode string.
            List/tuple must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns' elements if they are
            floats. This function must return a unicode string and will be
            applied only to the non-``NaN`` elements, with ``NaN`` being
            handled by ``na_rep``.
        sparsify : bool, optional, default True
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row.
        index_names : bool, optional, default True
            Prints the names of the indexes.
        justify : str, default None
            How to justify the column labels. If None uses the option from
            the print configuration (controlled by set_option), 'right' out
            of the box. Valid values are

            * left
            * right
            * center
            * justify
            * justify-all
            * start
            * end
            * inherit
            * match-parent
            * initial
            * unset.
        max_rows : int, optional
            Maximum number of rows to display in the console.
        max_cols : int, optional
            Maximum number of columns to display in the console.
        show_dimensions : bool, default False
            Display DataFrame dimensions (number of rows by number of columns).
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
"""

# 定义有效的列标签对齐方式的元组
VALID_JUSTIFY_PARAMETERS = (
    "left",
    "right",
    "center",
    "justify",
    "justify-all",
    "start",
    "end",
    "inherit",
    "match-parent",
    "initial",
    "unset",
)

# 定义返回文档字符串模板，描述了函数的返回值
return_docstring: Final = """
        Returns
        -------
        str or None
            If buf is None, returns the result as a string. Otherwise returns
            None.
"""


class SeriesFormatter:
    """
    Implement the main logic of Series.to_string, which underlies
    Series.__repr__.
    """
    def __init__(
        self,
        series: Series,
        *,
        length: bool | str = True,  # 初始化方法，接受一个Series作为参数，并可选地接受多个命名关键字参数
        header: bool = True,  # 默认为True，控制是否包含列名
        index: bool = True,  # 默认为True，控制是否包含行索引
        na_rep: str = "NaN",  # 默认为"NaN"，用于表示缺失值的字符串
        name: bool = False,  # 默认为False，控制是否包含Series的名称
        float_format: str | None = None,  # 浮点数格式化字符串，默认为None，将根据设置获取
        dtype: bool = True,  # 默认为True，控制是否显示数据类型
        max_rows: int | None = None,  # 最大显示行数，默认为None，即显示所有行
        min_rows: int | None = None,  # 最小显示行数，默认为None，即根据需要自动调整
    ) -> None:
        self.series = series  # 将传入的Series赋值给实例变量self.series
        self.buf = StringIO()  # 创建一个StringIO对象，用于存储输出的字符串
        self.name = name  # 将是否包含Series名称的设置赋值给self.name
        self.na_rep = na_rep  # 将缺失值表示字符串的设置赋值给self.na_rep
        self.header = header  # 将是否包含列名的设置赋值给self.header
        self.length = length  # 将是否显示长度信息的设置赋值给self.length
        self.index = index  # 将是否包含行索引的设置赋值给self.index
        self.max_rows = max_rows  # 将最大显示行数的设置赋值给self.max_rows
        self.min_rows = min_rows  # 将最小显示行数的设置赋值给self.min_rows

        if float_format is None:
            float_format = get_option("display.float_format")  # 如果未设置浮点数格式化字符串，则根据全局设置获取

        self.float_format = float_format  # 将浮点数格式化字符串的设置赋值给self.float_format
        self.dtype = dtype  # 将是否显示数据类型的设置赋值给self.dtype
        self.adj = printing.get_adjustment()  # 调用printing模块的get_adjustment方法获取调整参数

        self._chk_truncate()  # 调用内部方法_chk_truncate进行截断检查和处理

    def _chk_truncate(self) -> None:
        self.tr_row_num: int | None  # 定义实例变量self.tr_row_num，表示截断的行数，可能为整数或None

        min_rows = self.min_rows  # 将最小显示行数赋值给局部变量min_rows
        max_rows = self.max_rows  # 将最大显示行数赋值给局部变量max_rows

        # 根据最大显示行数确定是否需要截断，实际截断的行数由min_rows确定
        is_truncated_vertically = max_rows and (len(self.series) > max_rows)

        series = self.series  # 将实例变量self.series赋值给局部变量series

        if is_truncated_vertically:
            max_rows = cast(int, max_rows)  # 将最大显示行数转换为整数类型

            if min_rows:
                # 如果设置了min_rows（不为None或0），将max_rows设置为两者的最小值
                max_rows = min(min_rows, max_rows)

            if max_rows == 1:
                row_num = max_rows  # 如果最大显示行数为1，则row_num为1
                series = series.iloc[:max_rows]  # 对series进行切片，仅保留前max_rows行
            else:
                row_num = max_rows // 2  # 否则，row_num为max_rows的一半
                # 对series进行切片，保留前row_num行和后row_num行，并合并为一个新的series
                series = concat((series.iloc[:row_num], series.iloc[-row_num:]))

            self.tr_row_num = row_num  # 将截断的行数赋值给self.tr_row_num
        else:
            self.tr_row_num = None  # 如果不需要截断，则将self.tr_row_num设置为None

        self.tr_series = series  # 将经过截断处理后的series赋值给self.tr_series
        self.is_truncated_vertically = is_truncated_vertically  # 将是否垂直截断的布尔值赋值给self.is_truncated_vertically
    # 获取页脚信息的方法，返回一个字符串
    def _get_footer(self) -> str:
        # 获取系列的名称
        name = self.series.name
        # 初始化页脚字符串为空
        footer = ""

        # 获取系列的索引
        index = self.series.index
        # 如果索引是日期时间索引、周期索引或时间增量索引，并且有定义频率
        if (
            isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex))
            and index.freq is not None
        ):
            # 添加频率信息到页脚字符串
            footer += f"Freq: {index.freqstr}"

        # 如果系列有名称并且不是 False
        if self.name is not False and name is not None:
            # 如果页脚字符串不为空，则添加逗号和空格
            if footer:
                footer += ", "

            # 获取格式化后的系列名称并添加到页脚字符串
            series_name = printing.pprint_thing(name, escape_chars=("\t", "\r", "\n"))
            footer += f"Name: {series_name}"

        # 如果需要显示长度信息或者根据条件截断了垂直显示
        if self.length is True or (
            self.length == "truncate" and self.is_truncated_vertically
        ):
            # 如果页脚字符串不为空，则添加逗号和空格
            if footer:
                footer += ", "
            # 添加长度信息到页脚字符串
            footer += f"Length: {len(self.series)}"

        # 如果需要显示数据类型信息并且数据类型不是 False 也不是 None
        if self.dtype is not False and self.dtype is not None:
            # 获取数据类型名称并添加到页脚字符串
            dtype_name = getattr(self.tr_series.dtype, "name", None)
            if dtype_name:
                if footer:
                    footer += ", "
                footer += f"dtype: {printing.pprint_thing(dtype_name)}"

        # 如果系列的数据类型是分类数据类型
        if isinstance(self.tr_series.dtype, CategoricalDtype):
            # 获取级别信息并添加到页脚字符串的新行
            level_info = self.tr_series._values._get_repr_footer()
            if footer:
                footer += "\n"
            footer += level_info

        # 将页脚字符串转换为字符串并返回
        return str(footer)

    # 获取格式化后的值的方法，返回一个字符串列表
    def _get_formatted_values(self) -> list[str]:
        # 调用 format_array 函数获取格式化后的数组值
        return format_array(
            self.tr_series._values,  # 传递系列的值作为格式化数组的输入
            None,  # 不传递列名
            float_format=self.float_format,  # 浮点数格式
            na_rep=self.na_rep,  # 缺失值的替代表示
            leading_space=self.index,  # 是否在索引前面加空格
        )
    # 将对象转换为字符串表示形式，返回格式化后的字符串
    def to_string(self) -> str:
        # 获取对象的时间序列数据
        series = self.tr_series
        # 获取对象的页脚信息
        footer = self._get_footer()

        # 如果时间序列数据为空
        if len(series) == 0:
            # 返回空时间序列对象的字符串表示形式
            return f"{type(self.series).__name__}([], {footer})"

        # 获取时间序列的索引
        index = series.index
        # 检查索引是否有名称
        have_header = _has_names(index)

        # 如果索引是多重索引对象
        if isinstance(index, MultiIndex):
            # 格式化多重索引，包括名称，调整其显示格式
            fmt_index = index._format_multi(include_names=True, sparsify=None)
            # 获取调整对象，用于适当对齐格式化后的索引
            adj = printing.get_adjustment()
            # 将格式化后的索引适当连接，以便显示在三列中，并按行分割
            fmt_index = adj.adjoin(2, *fmt_index).split("\n")
        else:
            # 格式化平坦索引，包括名称
            fmt_index = index._format_flat(include_name=True)
        
        # 获取格式化后的数值数据
        fmt_values = self._get_formatted_values()

        # 如果对象被垂直截断
        if self.is_truncated_vertically:
            # 初始行头行数为0
            n_header_rows = 0
            # 获取截断的行号
            row_num = self.tr_row_num
            # 强制转换行号为整数
            row_num = cast(int, row_num)
            # 获取格式化值的最后一行宽度
            width = self.adj.len(fmt_values[row_num - 1])
            # 如果宽度大于3，则省略号为"..."
            if width > 3:
                dot_str = "..."
            else:
                dot_str = ".."
            # 使用居中模式，对省略号进行对齐
            dot_str = self.adj.justify([dot_str], width, mode="center")[0]
            # 将省略号插入格式化值的指定行
            fmt_values.insert(row_num + n_header_rows, dot_str)
            # 在格式化索引中的指定行后插入空字符串

            fmt_index.insert(row_num + 1, "")

        # 如果存在索引
        if self.index:
            # 将格式化后的索引和数值数据连接成三列，并按行显示
            result = self.adj.adjoin(3, *[fmt_index[1:], fmt_values])
        else:
            # 否则，仅将格式化后的数值数据连接成三列显示
            result = self.adj.adjoin(3, fmt_values)

        # 如果有头部信息并且头部信息存在于索引中
        if self.header and have_header:
            # 在结果的开头添加格式化后的索引第一行
            result = fmt_index[0] + "\n" + result

        # 如果存在页脚信息，则在结果的末尾添加页脚信息
        if footer:
            result += "\n" + footer

        # 返回结果的字符串表示形式
        return str("".join(result))
# 返回用于 repr(dataFrame) 调用的参数字典，供 DataFrame.to_string 使用
def get_dataframe_repr_params() -> dict[str, Any]:
    """Get the parameters used to repr(dataFrame) calls using DataFrame.to_string.

    Supplying these parameters to DataFrame.to_string is equivalent to calling
    ``repr(DataFrame)``. This is useful if you want to adjust the repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame([[1, 2], [3, 4]])
    >>> repr_params = pd.io.formats.format.get_dataframe_repr_params()
    >>> repr(df) == df.to_string(**repr_params)
    True
    """
    # 导入控制台模块以获取控制台大小
    from pandas.io.formats import console

    # 根据显示设置决定线宽度
    if get_option("display.expand_frame_repr"):
        line_width, _ = console.get_console_size()
    else:
        line_width = None
    # 返回 DataFrame.to_string 方法所需的显示参数字典
    return {
        "max_rows": get_option("display.max_rows"),
        "min_rows": get_option("display.min_rows"),
        "max_cols": get_option("display.max_columns"),
        "max_colwidth": get_option("display.max_colwidth"),
        "show_dimensions": get_option("display.show_dimensions"),
        "line_width": line_width,
    }


# 返回用于 repr(Series) 调用的参数字典，供 Series.to_string 使用
def get_series_repr_params() -> dict[str, Any]:
    """Get the parameters used to repr(Series) calls using Series.to_string.

    Supplying these parameters to Series.to_string is equivalent to calling
    ``repr(series)``. This is useful if you want to adjust the series repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> ser = pd.Series([1, 2, 3, 4])
    >>> repr_params = pd.io.formats.format.get_series_repr_params()
    >>> repr(ser) == ser.to_string(**repr_params)
    True
    """
    # 获取终端窗口大小
    width, height = get_terminal_size()
    # 获取最大行数选项
    max_rows_opt = get_option("display.max_rows")
    # 计算最大行数和最小行数
    max_rows = height if max_rows_opt == 0 else max_rows_opt
    min_rows = height if max_rows_opt == 0 else get_option("display.min_rows")

    # 返回 Series.to_string 方法所需的显示参数字典
    return {
        "name": True,
        "dtype": True,
        "min_rows": min_rows,
        "max_rows": max_rows,
        "length": get_option("display.show_dimensions"),
    }


class DataFrameFormatter:
    """
    Class for processing dataframe formatting options and data.

    Used by DataFrame.to_string, which backs DataFrame.__repr__.
    """

    # 类的文档字符串，包含通用文档和返回文档
    __doc__ = __doc__ if __doc__ else ""
    __doc__ += common_docstring + return_docstring
    def __init__(
        self,
        frame: DataFrame,
        columns: Axes | None = None,
        col_space: ColspaceArgType | None = None,
        header: bool | SequenceNotStr[str] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: FormattersType | None = None,
        justify: str | None = None,
        float_format: FloatFormatType | None = None,
        sparsify: bool | None = None,
        index_names: bool = True,
        max_rows: int | None = None,
        min_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: bool | str = False,
        decimal: str = ".",
        bold_rows: bool = False,
        escape: bool = True,
    ) -> None:
        """
        Initialize a PrettyTable object with various formatting options.

        Args:
            frame (DataFrame): The DataFrame to display.
            columns (Axes | None, optional): Columns to display. Defaults to None.
            col_space (ColspaceArgType | None, optional): Column spacing options. Defaults to None.
            header (bool | SequenceNotStr[str], optional): Whether to display headers. Defaults to True.
            index (bool, optional): Whether to display index. Defaults to True.
            na_rep (str, optional): String representation for NaN values. Defaults to "NaN".
            formatters (FormattersType | None, optional): Custom formatters for data display. Defaults to None.
            justify (str | None, optional): Justification for data display. Defaults to None.
            float_format (FloatFormatType | None, optional): Formatting options for floats. Defaults to None.
            sparsify (bool | None, optional): Whether to sparsify the display. Defaults to None.
            index_names (bool, optional): Whether to show index names. Defaults to True.
            max_rows (int | None, optional): Maximum number of rows to display. Defaults to None.
            min_rows (int | None, optional): Minimum number of rows to display. Defaults to None.
            max_cols (int | None, optional): Maximum number of columns to display. Defaults to None.
            show_dimensions (bool | str, optional): Whether to show dimensions. Defaults to False.
            decimal (str, optional): Decimal separator. Defaults to ".".
            bold_rows (bool, optional): Whether to display rows in bold. Defaults to False.
            escape (bool, optional): Whether to escape special characters. Defaults to True.
        """
        self.frame = frame
        self.columns = self._initialize_columns(columns)  # Initialize columns for display
        self.col_space = self._initialize_colspace(col_space)  # Initialize column spacing
        self.header = header  # Set header display option
        self.index = index  # Set index display option
        self.na_rep = na_rep  # Set NaN representation
        self.formatters = self._initialize_formatters(formatters)  # Initialize data formatters
        self.justify = self._initialize_justify(justify)  # Set data justification
        self.float_format = float_format  # Set float formatting
        self.sparsify = self._initialize_sparsify(sparsify)  # Set sparsity option
        self.show_index_names = index_names  # Set index names display option
        self.decimal = decimal  # Set decimal separator
        self.bold_rows = bold_rows  # Set bold rows option
        self.escape = escape  # Set escape special characters option
        self.max_rows = max_rows  # Set maximum rows to display
        self.min_rows = min_rows  # Set minimum rows to display
        self.max_cols = max_cols  # Set maximum columns to display
        self.show_dimensions = show_dimensions  # Set show dimensions option

        self.max_cols_fitted = self._calc_max_cols_fitted()  # Calculate maximum fitted columns
        self.max_rows_fitted = self._calc_max_rows_fitted()  # Calculate maximum fitted rows

        self.tr_frame = self.frame  # Initialize transformed frame with original frame
        self.truncate()  # Truncate frame for display
        self.adj = printing.get_adjustment()  # Get adjustment for printing

    def get_strcols(self) -> list[list[str]]:
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        strcols = self._get_strcols_without_index()  # Get string columns without index

        if self.index:
            str_index = self._get_formatted_index(self.tr_frame)  # Format index for display
            strcols.insert(0, str_index)  # Insert formatted index at the beginning

        return strcols  # Return list of string columns

    @property
    def should_show_dimensions(self) -> bool:
        """
        Determine if dimensions should be shown based on user options.
        """
        return self.show_dimensions is True or (
            self.show_dimensions == "truncate" and self.is_truncated
        )

    @property
    def is_truncated(self) -> bool:
        """
        Check if the DataFrame is truncated either horizontally or vertically.
        """
        return bool(self.is_truncated_horizontally or self.is_truncated_vertically)

    @property
    def is_truncated_horizontally(self) -> bool:
        """
        Check if the DataFrame is truncated horizontally.
        """
        return bool(self.max_cols_fitted and (len(self.columns) > self.max_cols_fitted))

    @property
    def is_truncated_vertically(self) -> bool:
        """
        Check if the DataFrame is truncated vertically.
        """
        return bool(self.max_rows_fitted and (len(self.frame) > self.max_rows_fitted))

    @property
    def dimensions_info(self) -> str:
        """
        Return information about the dimensions of the DataFrame.
        """
        return f"\n\n[{len(self.frame)} rows x {len(self.frame.columns)} columns]"

    @property
    def has_index_names(self) -> bool:
        """
        Check if the DataFrame's index has names.
        """
        return _has_names(self.frame.index)
    # 检查数据框是否具有列名
    def has_column_names(self) -> bool:
        return _has_names(self.frame.columns)

    # 检查是否显示行索引名称
    @property
    def show_row_idx_names(self) -> bool:
        return all((self.has_index_names, self.index, self.show_index_names))

    # 检查是否显示列索引名称
    @property
    def show_col_idx_names(self) -> bool:
        return all((self.has_column_names, self.show_index_names, self.header))

    # 计算最大显示行数
    @property
    def max_rows_displayed(self) -> int:
        return min(self.max_rows or len(self.frame), len(self.frame))

    # 初始化稀疏显示选项
    def _initialize_sparsify(self, sparsify: bool | None) -> bool:
        if sparsify is None:
            return get_option("display.multi_sparse")
        return sparsify

    # 初始化格式化选项
    def _initialize_formatters(
        self, formatters: FormattersType | None
    ) -> FormattersType:
        if formatters is None:
            return {}
        elif len(self.frame.columns) == len(formatters) or isinstance(formatters, dict):
            return formatters
        else:
            raise ValueError(
                f"Formatters length({len(formatters)}) should match "
                f"DataFrame number of columns({len(self.frame.columns)})"
            )

    # 初始化列标题对齐方式
    def _initialize_justify(self, justify: str | None) -> str:
        if justify is None:
            return get_option("display.colheader_justify")
        else:
            return justify

    # 初始化数据框的列
    def _initialize_columns(self, columns: Axes | None) -> Index:
        if columns is not None:
            cols = ensure_index(columns)
            self.frame = self.frame[cols]
            return cols
        else:
            return self.frame.columns

    # 初始化列间空白的显示选项
    def _initialize_colspace(self, col_space: ColspaceArgType | None) -> ColspaceType:
        result: ColspaceType

        if col_space is None:
            result = {}
        elif isinstance(col_space, (int, str)):
            result = {"": col_space}
            result.update({column: col_space for column in self.frame.columns})
        elif isinstance(col_space, Mapping):
            for column in col_space.keys():
                if column not in self.frame.columns and column != "":
                    raise ValueError(
                        f"Col_space is defined for an unknown column: {column}"
                    )
            result = col_space
        else:
            if len(self.frame.columns) != len(col_space):
                raise ValueError(
                    f"Col_space length({len(col_space)}) should match "
                    f"DataFrame number of columns({len(self.frame.columns)})"
                )
            result = dict(zip(self.frame.columns, col_space))
        return result

    # 计算适应屏幕的最大列数
    def _calc_max_cols_fitted(self) -> int | None:
        """Number of columns fitting the screen."""
        if not self._is_in_terminal():
            return self.max_cols

        width, _ = get_terminal_size()
        if self._is_screen_narrow(width):
            return width
        else:
            return self.max_cols
    def _calc_max_rows_fitted(self) -> int | None:
        """Calculate the maximum number of rows that can fit the screen."""
        max_rows: int | None  # Declare max_rows variable as integer or None

        if self._is_in_terminal():  # Check if output is intended for terminal
            _, height = get_terminal_size()  # Get terminal dimensions
            if self.max_rows == 0:
                # Calculate rows available for actual data after accounting for auxiliary rows
                return height - self._get_number_of_auxiliary_rows()

            if self._is_screen_short(height):
                max_rows = height  # Set max_rows to terminal height if screen is short
            else:
                max_rows = self.max_rows  # Use configured max_rows if screen height is sufficient
        else:
            max_rows = self.max_rows  # Use configured max_rows if not in terminal

        return self._adjust_max_rows(max_rows)  # Adjust and return the computed max_rows

    def _adjust_max_rows(self, max_rows: int | None) -> int | None:
        """Adjust the maximum number of rows based on display logic."""
        if max_rows:
            if (len(self.frame) > max_rows) and self.min_rows:
                # If the frame exceeds max_rows and min_rows is set, adjust max_rows
                max_rows = min(self.min_rows, max_rows)
        return max_rows  # Return adjusted max_rows

    def _is_in_terminal(self) -> bool:
        """Check if the output is intended for a terminal."""
        return bool(self.max_cols == 0 or self.max_rows == 0)  # Returns True if output is to terminal

    def _is_screen_narrow(self, max_width) -> bool:
        """Check if the screen width is narrow."""
        return bool(self.max_cols == 0 and len(self.frame.columns) > max_width)  # Returns True if screen is narrow

    def _is_screen_short(self, max_height) -> bool:
        """Check if the screen height is short."""
        return bool(self.max_rows == 0 and len(self.frame) > max_height)  # Returns True if screen is short

    def _get_number_of_auxiliary_rows(self) -> int:
        """Calculate the number of rows occupied by prompt, dots, and dimension info."""
        dot_row = 1  # Dot row count
        prompt_row = 1  # Prompt row count
        num_rows = dot_row + prompt_row  # Initial number of rows

        if self.show_dimensions:
            num_rows += len(self.dimensions_info.splitlines())  # Add rows for dimension info lines

        if self.header:
            num_rows += 1  # Add one row for header if present

        return num_rows  # Return total number of auxiliary rows

    def truncate(self) -> None:
        """
        Truncate the frame horizontally or vertically if necessary.
        """
        if self.is_truncated_horizontally:
            self._truncate_horizontally()  # Truncate horizontally if flagged

        if self.is_truncated_vertically:
            self._truncate_vertically()  # Truncate vertically if flagged
    # 水平方向截断操作，删除不需要显示的列并调整格式化器

    # 断言确保最大适配列数已定义
    assert self.max_cols_fitted is not None
    # 计算要保留的列数为最大适配列数的一半
    col_num = self.max_cols_fitted // 2
    # 如果要保留的列数大于等于1
    if col_num >= 1:
        # 从左边和右边分别选择一半的列
        left = self.tr_frame.iloc[:, :col_num]
        right = self.tr_frame.iloc[:, -col_num:]
        # 合并左右两部分列
        self.tr_frame = concat((left, right), axis=1)

        # 如果格式化器是列表或元组类型
        if isinstance(self.formatters, (list, tuple)):
            # 截取格式化器的左右各一半，重新组成格式化器列表
            self.formatters = [
                *self.formatters[:col_num],
                *self.formatters[-col_num:],
            ]
    else:
        # 如果要保留的列数小于1，则按最大列数进行截断
        col_num = cast(int, self.max_cols)
        self.tr_frame = self.tr_frame.iloc[:, :col_num]
    
    # 更新保留的列数
    self.tr_col_num = col_num

    # 垂直方向截断操作，删除不需要显示的行

    # 断言确保最大适配行数已定义
    assert self.max_rows_fitted is not None
    # 计算要保留的行数为最大适配行数的一半
    row_num = self.max_rows_fitted // 2
    # 如果要保留的行数大于等于1
    if row_num >= 1:
        _len = len(self.tr_frame)
        # 构建索引，选择前一半和后一半的行
        _slice = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
        self.tr_frame = self.tr_frame.iloc[_slice]
    else:
        # 如果要保留的行数小于1，则按最大行数进行截断
        row_num = cast(int, self.max_rows)
        self.tr_frame = self.tr_frame.iloc[:row_num, :]

    # 更新保留的行数
    self.tr_row_num = row_num
    # 返回一个包含没有索引的字符串列的列表的列表
    def _get_strcols_without_index(self) -> list[list[str]]:
        # 初始化一个空的二维列表用于存储字符串列
        strcols: list[list[str]] = []

        # 如果 header 不是类列表且为空，则处理每列数据
        if not is_list_like(self.header) and not self.header:
            # 遍历每列数据的索引和值
            for i, c in enumerate(self.tr_frame):
                # 对当前列进行格式化
                fmt_values = self.format_col(i)
                # 将格式化后的值调整为固定宽度
                fmt_values = _make_fixed_width(
                    strings=fmt_values,
                    justify=self.justify,
                    minimum=int(self.col_space.get(c, 0)),
                    adj=self.adj,
                )
                # 将格式化后的列添加到结果列表中
                strcols.append(fmt_values)
            # 返回格式化后的所有列
            return strcols

        # 如果 header 是类列表，则进行类型转换
        if is_list_like(self.header):
            self.header = cast(list[str], self.header)
            # 如果 header 的长度与列数不匹配，则抛出异常
            if len(self.header) != len(self.columns):
                raise ValueError(
                    f"Writing {len(self.columns)} cols "
                    f"but got {len(self.header)} aliases"
                )
            # 初始化字符串列，每个列包含一个标签
            str_columns = [[label] for label in self.header]
        else:
            # 否则获取格式化后的列标签
            str_columns = self._get_formatted_column_labels(self.tr_frame)

        # 如果设置了显示行索引名称，则在每个列标签后添加空字符串
        if self.show_row_idx_names:
            for x in str_columns:
                x.append("")

        # 对每列数据进行处理
        for i, c in enumerate(self.tr_frame):
            # 获取当前列的列标签
            cheader = str_columns[i]
            # 计算当前列标签的最大宽度
            header_colwidth = max(
                int(self.col_space.get(c, 0)), *(self.adj.len(x) for x in cheader)
            )
            # 对当前列进行格式化
            fmt_values = self.format_col(i)
            # 将格式化后的列值调整为固定宽度
            fmt_values = _make_fixed_width(
                fmt_values, self.justify, minimum=header_colwidth, adj=self.adj
            )

            # 计算格式化后列值的最大长度
            max_len = max(*(self.adj.len(x) for x in fmt_values), header_colwidth)
            # 对列标签进行调整对齐
            cheader = self.adj.justify(cheader, max_len, mode=self.justify)
            # 将调整后的列标签与格式化后的列值合并并添加到结果列表中
            strcols.append(cheader + fmt_values)

        # 返回格式化后的所有列
        return strcols

    # 格式化指定索引列的数据并返回格式化后的列表
    def format_col(self, i: int) -> list[str]:
        # 获取表格数据
        frame = self.tr_frame
        # 获取指定列的格式化器
        formatter = self._get_formatter(i)
        # 对指定列进行格式化并返回格式化后的列表
        return format_array(
            frame.iloc[:, i]._values,
            formatter,
            float_format=self.float_format,
            na_rep=self.na_rep,
            space=self.col_space.get(frame.columns[i]),
            decimal=self.decimal,
            leading_space=self.index,
        )

    # 获取指定列的格式化器并返回
    def _get_formatter(self, i: str | int) -> Callable | None:
        # 如果格式化器是列表或元组，则根据索引返回对应格式化器
        if isinstance(self.formatters, (list, tuple)):
            if is_integer(i):
                i = cast(int, i)
                return self.formatters[i]
            else:
                return None
        else:
            # 否则如果索引是整数且不在列索引中，则返回格式化器
            if is_integer(i) and i not in self.columns:
                i = self.columns[i]
            return self.formatters.get(i, None)
    # 返回一个格式化后的列标签列表，用于打印或导出到不同格式的文档
    def _get_formatted_column_labels(self, frame: DataFrame) -> list[list[str]]:
        # 导入需要的函数库
        from pandas.core.indexes.multi import sparsify_labels

        # 获取数据框的列对象
        columns = frame.columns

        # 如果列是多级索引
        if isinstance(columns, MultiIndex):
            # 格式化多级索引的列标签，保留稀疏性（根据sparsify标志），不包括名称
            fmt_columns = columns._format_multi(sparsify=False, include_names=False)
            # 如果需要稀疏化标签且fmt_columns非空，则执行稀疏化操作
            if self.sparsify and len(fmt_columns):
                fmt_columns = sparsify_labels(fmt_columns)

            # 转换为字符串列表的列表，每个内部列表表示一个级别的列标签
            str_columns = [list(x) for x in zip(*fmt_columns)]
        else:
            # 格式化扁平化的列标签，不包括名称
            fmt_columns = columns._format_flat(include_name=False)
            # 根据列的数据类型判断是否需要填充空格
            str_columns = [
                [
                    " " + x
                    if not self._get_formatter(i) and is_numeric_dtype(dtype)
                    else x
                ]
                for i, (x, dtype) in enumerate(zip(fmt_columns, self.frame.dtypes))
            ]
        return str_columns

    # 返回格式化后的索引标签列表，用于打印或导出到不同格式的文档
    def _get_formatted_index(self, frame: DataFrame) -> list[str]:
        # 注意：此函数只被to_string()和to_latex()使用，不会被to_html()使用，因此可以在这里安全地进行col_space的强制类型转换
        col_space = {k: cast(int, v) for k, v in self.col_space.items()}
        # 获取数据框的索引对象和列对象
        index = frame.index
        columns = frame.columns
        # 获取索引格式化的函数对象
        fmt = self._get_formatter("__index__")

        # 如果索引是多级索引
        if isinstance(index, MultiIndex):
            # 格式化多级索引的索引标签，根据sparsify和show_row_idx_names设置，使用指定的formatter
            fmt_index = index._format_multi(
                sparsify=self.sparsify,
                include_names=self.show_row_idx_names,
                formatter=fmt,
            )
        else:
            # 格式化扁平化的索引标签，根据show_row_idx_names设置，使用指定的formatter
            fmt_index = [
                index._format_flat(include_name=self.show_row_idx_names, formatter=fmt)
            ]

        # 对格式化后的索引标签进行固定宽度调整
        fmt_index = [
            tuple(
                _make_fixed_width(
                    list(x), justify="left", minimum=col_space.get("", 0), adj=self.adj
                )
            )
            for x in fmt_index
        ]

        # 使用调整器对象将格式化后的索引标签进行合并并分隔成行
        adjoined = self.adj.adjoin(1, *fmt_index).split("\n")

        # 如果需要显示列名，则将列名转换为字符串列表；否则使用空字符串列表
        if self.show_col_idx_names:
            col_header = [str(x) for x in self._get_column_name_list()]
        else:
            col_header = [""] * columns.nlevels

        # 如果需要标题行，则返回列头和合并后的索引标签行；否则返回合并后的索引标签行
        if self.header:
            return col_header + adjoined
        else:
            return adjoined

    # 返回格式化后的列名列表，用于打印或导出到不同格式的文档
    def _get_column_name_list(self) -> list[Hashable]:
        # 初始化列名列表
        names: list[Hashable] = []
        # 获取数据框的列对象
        columns = self.frame.columns
        # 如果列是多级索引，则将每个级别的名称（若存在）加入到列表中
        if isinstance(columns, MultiIndex):
            names.extend("" if name is None else name for name in columns.names)
        else:
            # 否则将列名称（若存在）加入到列表中
            names.append("" if columns.name is None else columns.name)
        return names
    """Class for creating dataframe output in multiple formats.

    Called in pandas.core.generic.NDFrame:
        - to_csv
        - to_latex

    Called in pandas.DataFrame:
        - to_html
        - to_string

    Parameters
    ----------
    fmt : DataFrameFormatter
        Formatter with the formatting options.
    """

    def __init__(self, fmt: DataFrameFormatter) -> None:
        """
        Initialize the DataFrameRenderer instance.

        Parameters
        ----------
        fmt : DataFrameFormatter
            The formatter object defining formatting options.
        """
        self.fmt = fmt

    def to_html(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        encoding: str | None = None,
        classes: str | list | tuple | None = None,
        notebook: bool = False,
        border: int | bool | None = None,
        table_id: str | None = None,
        render_links: bool = False,
    ) -> str | None:
        """
        Render a DataFrame to an HTML table.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding : str, default “utf-8”
            Set character encoding.
        classes : str or list-like
            Classes to include in the `class` attribute of the opening
            ``<table>`` tag, in addition to the default "dataframe".
        notebook : {True, False}, optional, default False
            Whether the generated HTML is for IPython Notebook.
        border : int or bool or None
            A ``border=border`` attribute is included in the opening
            ``<table>`` tag. Default ``pd.options.display.html.border``.
        table_id : str, optional
            A CSS id included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.

        Returns
        -------
        str or None
            If `buf` is None, returns the rendered HTML table as a string;
            otherwise, saves the HTML to the specified buffer.

        Notes
        -----
        This method imports necessary formatters and determines the appropriate
        formatter based on the `notebook` parameter.
        """
        from pandas.io.formats.html import (
            HTMLFormatter,
            NotebookFormatter,
        )

        # Determine the formatter class based on the notebook flag
        Klass = NotebookFormatter if notebook else HTMLFormatter

        # Instantiate the selected formatter with provided options
        html_formatter = Klass(
            self.fmt,
            classes=classes,
            border=border,
            table_id=table_id,
            render_links=render_links,
        )

        # Generate the HTML string representation of the DataFrame
        string = html_formatter.to_string()

        # Save the generated HTML to the specified buffer or return as string
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        encoding: str | None = None,
        line_width: int | None = None,
    ) -> str | None:
        """
        Render a DataFrame to a console-friendly tabular output.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding : str, default “utf-8”
            Set character encoding.
        line_width : int, optional
            Width of the display. If None, use the detected terminal width.

        Returns
        -------
        str or None
            If `buf` is None, returns the rendered tabular output as a string;
            otherwise, saves the output to the specified buffer.

        Notes
        -----
        This method renders the DataFrame to a tabular text format suitable for
        console output, using specified formatting options.
        """
    def to_csv(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        encoding: str | None = None,
        sep: str = ",",
        columns: Sequence[Hashable] | None = None,
        index_label: IndexLabel | None = None,
        mode: str = "w",
        compression: CompressionOptions = "infer",
        quoting: int | None = None,
        quotechar: str = '"',
        lineterminator: str | None = None,
        chunksize: int | None = None,
        date_format: str | None = None,
        doublequote: bool = True,
        escapechar: str | None = None,
        errors: str = "strict",
        storage_options: StorageOptions | None = None,
    ) -> str | None:
        """
        Render dataframe as comma-separated file.
        """
        # 导入 CSV 格式化工具类
        from pandas.io.formats.csvs import CSVFormatter
    
        # 如果 path_or_buf 为 None，则创建一个 StringIO 缓冲区对象
        if path_or_buf is None:
            created_buffer = True
            path_or_buf = StringIO()
        else:
            created_buffer = False
    
        # 创建 CSV 格式化对象
        csv_formatter = CSVFormatter(
            path_or_buf=path_or_buf,
            lineterminator=lineterminator,
            sep=sep,
            encoding=encoding,
            errors=errors,
            compression=compression,
            quoting=quoting,
            cols=columns,
            index_label=index_label,
            mode=mode,
            chunksize=chunksize,
            quotechar=quotechar,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            storage_options=storage_options,
            formatter=self.fmt,
        )
        # 调用 CSV 格式化对象的保存方法
        csv_formatter.save()
    
        # 如果创建了缓冲区，则将其内容返回为字符串并关闭
        if created_buffer:
            assert isinstance(path_or_buf, StringIO)
            content = path_or_buf.getvalue()
            path_or_buf.close()
            return content
    
        # 如果没有创建缓冲区，则返回 None
        return None
# ----------------------------------------------------------------------
# Array formatters

# 格式化数组以便打印
def format_array(
    values: ArrayLike,
    formatter: Callable | None,
    float_format: FloatFormatType | None = None,
    na_rep: str = "NaN",
    digits: int | None = None,
    space: str | int | None = None,
    justify: str = "right",
    decimal: str = ".",
    leading_space: bool | None = True,
    quoting: int | None = None,
    fallback_formatter: Callable | None = None,
) -> list[str]:
    """
    Format an array for printing.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray  # 待打印的数组，可以是 NumPy 数组或扩展数组
    formatter  # 格式化函数，用于格式化数组中的每个元素
    float_format  # 浮点数格式化字符串
    na_rep  # 代表缺失值的字符串
    digits  # 浮点数格式化的小数位数
    space  # 列与列之间的空格数或空格字符串
    justify  # 数字对齐方式，默认右对齐
    decimal  # 小数点的字符表示
    leading_space : bool, optional, default True
        Whether the array should be formatted with a leading space.
        When an array as a column of a Series or DataFrame, we do want
        the leading space to pad between columns.

        When formatting an Index subclass
        (e.g. IntervalIndex._get_values_for_csv), we don't want the
        leading space since it should be left-aligned.
    fallback_formatter  # 备用的格式化函数

    Returns
    -------
    List[str]  # 返回格式化后的字符串列表
    """
    fmt_klass: type[_GenericArrayFormatter]

    # ----------------------------------------------------------------------
    # Serialization function

# ----------------------------------------------------------------------
# Serialization function

# 执行序列化。如果 buf 为 None，则写入 buf 中；如果 buf 不为 None，则作为字符串返回。
def save_to_buffer(
    string: str,
    buf: FilePath | WriteBuffer[str] | None = None,
    encoding: str | None = None,
) -> str | None:
    """
    Perform serialization. Write to buf or return as string if buf is None.

    Parameters
    ----------
    string : str  # 待序列化的字符串
    buf : FilePath | WriteBuffer[str] | None, optional  # 文件路径或写入缓冲区对象，如果为 None 则直接返回字符串
    encoding : str | None, optional  # 字符编码，如果为 None 则默认为 utf-8

    Returns
    -------
    str | None  # 如果 buf 为 None，则返回序列化后的字符串；否则返回 None
    """
    with _get_buffer(buf, encoding=encoding) as fd:  # 使用上下文管理器获取合适的缓冲区对象
        fd.write(string)  # 将字符串写入缓冲区

        if buf is None:
            # error: "WriteBuffer[str]" has no attribute "getvalue"
            return fd.getvalue()  # type: ignore[attr-defined]  # 如果 buf 为 None，则返回缓冲区中的数据
        return None  # 如果 buf 不为 None，则返回 None


@contextmanager
def _get_buffer(
    buf: FilePath | WriteBuffer[str] | None, encoding: str | None = None
) -> Generator[WriteBuffer[str], None, None] | Generator[StringIO, None, None]:
    """
    Context manager to open, yield and close buffer for filenames or Path-like
    objects, otherwise yield buf unchanged.

    Parameters
    ----------
    buf : FilePath | WriteBuffer[str] | None  # 文件路径或写入缓冲区对象，如果为 None 则创建一个 StringIO 对象
    encoding : str | None, optional  # 字符编码，如果为 None 则默认为 utf-8

    Yields
    ------
    Generator[WriteBuffer[str], None, None] | Generator[StringIO, None, None]
        返回写入缓冲区对象的生成器，或者直接返回 buf 对象
    """
    if buf is not None:
        buf = stringify_path(buf)  # 如果 buf 不为 None，则将其转换为字符串表示形式
    else:
        buf = StringIO()  # 如果 buf 为 None，则创建一个 StringIO 对象

    if encoding is None:
        encoding = "utf-8"  # 如果编码未指定，则默认使用 utf-8
    elif not isinstance(buf, str):
        raise ValueError("buf is not a file name and encoding is specified.")  # 如果 buf 不是文件名且指定了编码，则引发异常

    if hasattr(buf, "write"):
        # Incompatible types in "yield" (actual type "Union[str, WriteBuffer[str],
        # StringIO]", expected type "Union[WriteBuffer[str], StringIO]")
        yield buf  # type: ignore[misc]  # 如果 buf 有 write 方法，则直接返回 buf 对象
    elif isinstance(buf, str):
        check_parent_directory(str(buf))  # 检查父目录是否存在，如果不存在则创建
        with open(buf, "w", encoding=encoding, newline="") as f:
            # GH#30034 open instead of codecs.open prevents a file leak
            #  if we have an invalid encoding argument.
            # newline="" is needed to roundtrip correctly on
            #  windows test_to_latex_filename
            yield f  # 返回打开的文件对象
    else:
        raise TypeError("buf is not a file name and it has no write method")  # 如果 buf 不是文件名且没有 write 方法，则引发类型错误异常
    # 如果 values 的数据类型是 np.datetime64 且具有时区信息 ("M")，则选择 DatetimeArray 类型的日期时间格式化器
    if lib.is_np_dtype(values.dtype, "M"):
        fmt_klass = _Datetime64Formatter
        # 将 values 强制类型转换为 DatetimeArray
        values = cast(DatetimeArray, values)
    
    # 如果 values 的数据类型是 DatetimeTZDtype 类型，则选择带时区信息的日期时间格式化器
    elif isinstance(values.dtype, DatetimeTZDtype):
        fmt_klass = _Datetime64TZFormatter
        # 将 values 强制类型转换为 DatetimeArray
        values = cast(DatetimeArray, values)
    
    # 如果 values 的数据类型是 np.timedelta64 类型 ("m")，则选择 TimedeltaArray 类型的时间差格式化器
    elif lib.is_np_dtype(values.dtype, "m"):
        fmt_klass = _Timedelta64Formatter
        # 将 values 强制类型转换为 TimedeltaArray
        values = cast(TimedeltaArray, values)
    
    # 如果 values 的数据类型是 ExtensionDtype 类型，则选择 ExtensionArray 格式化器
    elif isinstance(values.dtype, ExtensionDtype):
        fmt_klass = _ExtensionArrayFormatter
    
    # 如果 values 的数据类型是复数类型 ("fc")，则选择浮点数数组格式化器
    elif lib.is_np_dtype(values.dtype, "fc"):
        fmt_klass = FloatArrayFormatter
    
    # 如果 values 的数据类型是无符号整数类型 ("iu")，则选择整数数组格式化器
    elif lib.is_np_dtype(values.dtype, "iu"):
        fmt_klass = _IntArrayFormatter
    
    # 如果 values 的数据类型不属于以上任何一种情况，则选择通用数组格式化器
    else:
        fmt_klass = _GenericArrayFormatter

    # 如果未指定 space 参数，则默认设置为 12
    if space is None:
        space = 12

    # 如果未指定 float_format 参数，则从全局配置中获取默认的浮点数格式化字符串
    if float_format is None:
        float_format = get_option("display.float_format")

    # 如果未指定 digits 参数，则从全局配置中获取默认的显示精度
    if digits is None:
        digits = get_option("display.precision")

    # 创建格式化对象 fmt_obj，根据之前选择的格式化器及各种参数进行初始化
    fmt_obj = fmt_klass(
        values,
        digits=digits,
        na_rep=na_rep,
        float_format=float_format,
        formatter=formatter,
        space=space,
        justify=justify,
        decimal=decimal,
        leading_space=leading_space,
        quoting=quoting,
        fallback_formatter=fallback_formatter,
    )

    # 返回格式化对象 fmt_obj 的结果
    return fmt_obj.get_result()
class _GenericArrayFormatter:
    # 定义一个名为 _GenericArrayFormatter 的类，用于格式化数组或类数组结构
    def __init__(
        self,
        values: ArrayLike,
        digits: int = 7,
        formatter: Callable | None = None,
        na_rep: str = "NaN",
        space: str | int = 12,
        float_format: FloatFormatType | None = None,
        justify: str = "right",
        decimal: str = ".",
        quoting: int | None = None,
        fixed_width: bool = True,
        leading_space: bool | None = True,
        fallback_formatter: Callable | None = None,
    ) -> None:
        # 初始化方法，接受多个参数用于配置格式化器对象
        self.values = values  # 保存传入的数据 values
        self.digits = digits  # 小数点后的位数，默认为 7
        self.na_rep = na_rep  # 在数据缺失时显示的字符串，默认为 "NaN"
        self.space = space  # 列的宽度，默认为 12
        self.formatter = formatter  # 格式化函数，可选
        self.float_format = float_format  # 浮点数格式化选项，可选
        self.justify = justify  # 对齐方式，默认右对齐
        self.decimal = decimal  # 小数点符号，默认为 "."
        self.quoting = quoting  # 引用风格，可选
        self.fixed_width = fixed_width  # 是否固定列宽，默认为 True
        self.leading_space = leading_space  # 是否在数字前面加空格，可选
        self.fallback_formatter = fallback_formatter  # 备用格式化函数，可选

    def get_result(self) -> list[str]:
        # 返回处理后的结果列表，调用内部方法 _format_strings() 格式化数据
        fmt_values = self._format_strings()
        return _make_fixed_width(fmt_values, self.justify)
    # 定义一个方法 _format_strings，返回类型为 list[str]
    def _format_strings(self) -> list[str]:
        # 如果 self.float_format 为 None，则获取默认的浮点数格式
        if self.float_format is None:
            float_format = get_option("display.float_format")
            # 如果默认浮点数格式也为 None，则获取显示精度并创建浮点数格式化函数
            if float_format is None:
                precision = get_option("display.precision")
                float_format = lambda x: _trim_zeros_single_float(
                    f"{x: .{precision:d}f}"
                )
        else:
            # 否则使用指定的浮点数格式
            float_format = self.float_format

        # 如果定义了 self.formatter，则使用它作为格式化函数
        if self.formatter is not None:
            formatter = self.formatter
        # 否则如果定义了 self.fallback_formatter，则使用它作为格式化函数
        elif self.fallback_formatter is not None:
            formatter = self.fallback_formatter
        # 否则创建一个默认的格式化函数 partial(printing.pprint_thing)
        else:
            quote_strings = self.quoting is not None and self.quoting != QUOTE_NONE
            formatter = partial(
                printing.pprint_thing,
                escape_chars=("\t", "\r", "\n"),
                quote_strings=quote_strings,
            )

        # 定义内部方法 _format(x)，根据不同情况格式化输入 x
        def _format(x):
            # 如果定义了 self.na_rep 且 x 是空标量值，则根据类型返回特定字符串
            if self.na_rep is not None and is_scalar(x) and isna(x):
                if x is None:
                    return "None"
                elif x is NA:
                    return str(NA)
                elif x is NaT or isinstance(x, (np.datetime64, np.timedelta64)):
                    return "NaT"
                return self.na_rep
            # 如果 x 是 Pandas 对象，则返回其字符串表示形式
            elif isinstance(x, PandasObject):
                return str(x)
            # 如果 x 是字符串类型，则返回其表达式形式
            elif isinstance(x, StringDtype):
                return repr(x)
            else:
                # 否则返回使用 formatter 格式化后的字符串表示
                return str(formatter(x))

        # 获取待格式化的值 vals
        vals = self.values
        # 如果 vals 不是 ndarray，则抛出类型错误
        if not isinstance(vals, np.ndarray):
            raise TypeError(
                "ExtensionArray formatting should use _ExtensionArrayFormatter"
            )
        # 推断 vals 中元素的类型为浮点数
        inferred = lib.map_infer(vals, is_float)
        # 判断 vals 是否为浮点数类型数组
        is_float_type = (
            inferred
            # vals 可能有 2 或更多维度，检查所有非空元素
            & np.all(notna(vals), axis=tuple(range(1, len(vals.shape))))
        )
        # 如果未指定 leading_space，则根据浮点数类型判断是否需要前导空格
        leading_space = self.leading_space
        if leading_space is None:
            leading_space = is_float_type.any()

        # 初始化格式化后的值列表 fmt_values
        fmt_values = []
        # 遍历 vals 中的每个元素 v，并根据其类型和前导空格需求进行格式化
        for i, v in enumerate(vals):
            if (not is_float_type[i] or self.formatter is not None) and leading_space:
                # 如果不是浮点数类型或已定义 formatter，则在前面添加空格后格式化
                fmt_values.append(f" {_format(v)}")
            elif is_float_type[i]:
                # 如果是浮点数类型，则应用指定的浮点数格式进行格式化
                fmt_values.append(float_format(v))
            else:
                # 否则根据 leading_space 的值确定是否包含空格，并格式化
                if leading_space is False:
                    # 明确指定为 False，因此默认情况下包含空格
                    tpl = "{v}"
                else:
                    tpl = " {v}"
                fmt_values.append(tpl.format(v=_format(v)))

        # 返回格式化后的值列表 fmt_values
        return fmt_values
# FloatArrayFormatter 类继承自 _GenericArrayFormatter 类
class FloatArrayFormatter(_GenericArrayFormatter):
    
    # 初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 如果 float_format 不为 None 而 formatter 为 None，则进行以下操作
        # GH21625, GH22270 是 GitHub issue 的引用
        self.fixed_width = False
        if self.float_format is not None and self.formatter is None:
            # 如果 float_format 是可调用的函数，则将其设置为 formatter，同时将 float_format 置为 None
            self.formatter = self.float_format
            self.float_format = None

    # _value_formatter 方法定义，接受两个参数，float_format 和 threshold
    def _value_formatter(
        self,
        float_format: FloatFormatType | None = None,
        threshold: float | None = None,
    ) -> Callable:
        """Returns a function to be applied on each value to format it"""
        # 如果 float_format 参数为 None，则使用 self.float_format
        if float_format is None:
            float_format = self.float_format

        # 定义一个基础的格式化函数 base_formatter，根据是否为 NaN，返回格式化后的字符串或者 self.na_rep
        if float_format:

            def base_formatter(v):
                assert float_format is not None  # for mypy
                # 如果 v 不为 NaN，则调用 float_format 进行格式化，否则返回 self.na_rep
                return (
                    float_format(value=v)  # type: ignore[operator,call-arg]
                    if notna(v)
                    else self.na_rep
                )

        else:

            def base_formatter(v):
                # 如果 v 不为 NaN，则返回 str(v)，否则返回 self.na_rep
                return str(v) if notna(v) else self.na_rep

        # 如果 self.decimal 不为 "."，定义一个 decimal_formatter 函数，
        # 将 base_formatter 返回的字符串中的 "." 替换为 self.decimal，只替换一次
        if self.decimal != ".":

            def decimal_formatter(v):
                return base_formatter(v).replace(".", self.decimal, 1)

        else:
            # 否则，decimal_formatter 与 base_formatter 相同
            decimal_formatter = base_formatter

        # 如果 threshold 为 None，直接返回 decimal_formatter 函数
        if threshold is None:
            return decimal_formatter

        # 定义一个 formatter 函数，根据值的绝对值与 threshold 的比较来选择使用 decimal_formatter 还是返回 0.0 的格式化结果
        def formatter(value):
            if notna(value):
                if abs(value) > threshold:
                    return decimal_formatter(value)
                else:
                    return decimal_formatter(0.0)
            else:
                return self.na_rep

        return formatter

    # _format_strings 方法定义，返回结果为字符串列表，调用了 get_result_as_array 方法
    def _format_strings(self) -> list[str]:
        return list(self.get_result_as_array())


# _IntArrayFormatter 类继承自 _GenericArrayFormatter 类
class _IntArrayFormatter(_GenericArrayFormatter):
    
    # _format_strings 方法定义，返回结果为格式化后的字符串列表
    def _format_strings(self) -> list[str]:
        # 如果 leading_space 为 False，定义一个 lambda 函数 formatter_str，用于格式化整数为字符串
        if self.leading_space is False:
            formatter_str = lambda x: f"{x:d}".format(x=x)
        else:
            # 否则，定义一个 lambda 函数 formatter_str，格式化整数为字符串，有一个空格
            formatter_str = lambda x: f"{x: d}".format(x=x)
        
        # formatter 变量为 self.formatter 或者 formatter_str
        formatter = self.formatter or formatter_str
        # 格式化 self.values 中的每一个值，结果为 fmt_values 列表
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values


# _Datetime64Formatter 类继承自 _GenericArrayFormatter 类，values 类型为 DatetimeArray
class _Datetime64Formatter(_GenericArrayFormatter):
    values: DatetimeArray  # values 属性为 DatetimeArray 类型
    # 初始化方法，接受DatetimeArray类型的values作为参数，以及可选的nat_rep和date_format参数
    # 调用父类的初始化方法，将values和额外的关键字参数传递给父类
    super().__init__(values, **kwargs)
    # 设置实例属性nat_rep，用于表示缺失日期时间的字符串表示，默认为"NaT"
    self.nat_rep = nat_rep
    # 设置实例属性date_format，用于指定日期时间格式化的格式，如果未指定则为None
    self.date_format = date_format

def _format_strings(self) -> list[str]:
    """we by definition have DO NOT have a TZ"""
    # 获取实例属性values，即DatetimeArray类型的数据
    values = self.values

    # 如果实例属性formatter不为None，说明有格式化函数
    if self.formatter is not None:
        # 对values中的每个元素应用formatter函数进行格式化，返回格式化后的列表
        return [self.formatter(x) for x in values]

    # 如果formatter为None，则调用values的_format_native_types方法对其进行格式化
    fmt_values = values._format_native_types(
        na_rep=self.nat_rep, date_format=self.date_format
    )
    # 将格式化后的结果转换为列表并返回
    return fmt_values.tolist()
class _ExtensionArrayFormatter(_GenericArrayFormatter):
    values: ExtensionArray

    # 从 ExtensionArray 继承并添加格式化字符串的功能
    def _format_strings(self) -> list[str]:
        values = self.values

        formatter = self.formatter
        fallback_formatter = None
        if formatter is None:
            # 如果未指定 formatter，则使用 values 的默认格式化方法
            fallback_formatter = values._formatter(boxed=True)

        if isinstance(values, Categorical):
            # 如果 values 是 Categorical 类型，获取其内部值以保留 tzinfo 信息
            array = values._internal_get_values()
        else:
            # 否则将 values 转换为 object 类型的 numpy 数组
            array = np.asarray(values, dtype=object)

        # 调用 format_array 函数格式化数组
        fmt_values = format_array(
            array,
            formatter,
            float_format=self.float_format,
            na_rep=self.na_rep,
            digits=self.digits,
            space=self.space,
            justify=self.justify,
            decimal=self.decimal,
            leading_space=self.leading_space,
            quoting=self.quoting,
            fallback_formatter=fallback_formatter,
        )
        return fmt_values


def format_percentiles(
    percentiles: np.ndarray | Sequence[float],
) -> list[str]:
    """
    Outputs rounded and formatted percentiles.

    Parameters
    ----------
    percentiles : list-like, containing floats from interval [0,1]

    Returns
    -------
    formatted : list of strings

    Notes
    -----
    Rounding precision is chosen so that: (1) if any two elements of
    ``percentiles`` differ, they remain different after rounding
    (2) no entry is *rounded* to 0% or 100%.
    Any non-integer is always rounded to at least 1 decimal place.

    Examples
    --------
    Keeps all entries different after rounding:

    >>> format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
    ['1.999%', '2.001%', '50%', '66.667%', '99.99%']

    No element is rounded to 0% or 100% (unless already equal to it).
    Duplicates are allowed:

    >>> format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])
    ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']
    """
    percentiles = np.asarray(percentiles)

    # 检查 percentiles 是否为数值类型且在 [0,1] 区间内
    if (
        not is_numeric_dtype(percentiles)
        or not np.all(percentiles >= 0)
        or not np.all(percentiles <= 1)
    ):
        raise ValueError("percentiles should all be in the interval [0,1]")

    # 将 percentiles 转换为百分比形式
    percentiles = 100 * percentiles
    # 获取百分比的精度
    prec = get_precision(percentiles)
    # 对 percentiles 进行四舍五入并转换为整数类型
    percentiles_round_type = percentiles.round(prec).astype(int)

    # 判断是否所有元素都为整数
    int_idx = np.isclose(percentiles_round_type, percentiles)

    if np.all(int_idx):
        # 如果所有元素都是整数，则直接转换为字符串形式并加上百分号返回
        out = percentiles_round_type.astype(str)
        return [i + "%" for i in out]

    # 获取唯一的百分比值
    unique_pcts = np.unique(percentiles)
    # 获取唯一值的精度
    prec = get_precision(unique_pcts)
    # 初始化输出数组
    out = np.empty_like(percentiles, dtype=object)
    # 对整数元素进行处理
    out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)

    # 对非整数元素进行处理
    out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
    return [i + "%" for i in out]


def get_precision(array: np.ndarray | Sequence[float]) -> int:
    # 返回数组中元素的小数点位数的最大值
    # 取数组的第一个元素，如果大于0，则赋值给 to_begin，否则设为 None
    to_begin = array[0] if array[0] > 0 else None
    
    # 取数组的最后一个元素，如果小于100，则计算其与100的差值赋值给 to_end，否则设为 None
    to_end = 100 - array[-1] if array[-1] < 100 else None
    
    # 使用 numpy 的差分函数计算数组的一阶差分，并指定起始和结束值
    diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
    
    # 对差分结果取绝对值
    diff = abs(diff)
    
    # 计算差分结果的最小值的对数（以10为底），取负数并向下取整，作为精度的候选值
    prec = -np.floor(np.log10(np.min(diff))).astype(int)
    
    # 确保精度至少为1
    prec = max(1, prec)
    
    # 返回计算出的精度值
    return prec
# 定义一个函数 `_format_datetime64`，用于格式化 datetime64 对象为字符串
def _format_datetime64(x: NaTType | Timestamp, nat_rep: str = "NaT") -> str:
    # 如果 x 是 NaT 类型，返回指定的字符串表示 NaT
    if x is NaT:
        return nat_rep

    # Timestamp.__str__ 会回退到 datetime.datetime.__str__ 方法，使用 isoformat(sep=' ') 格式化为字符串，而不是使用 strftime（更快速）
    return str(x)


# 定义另一个函数 `_format_datetime64_dateonly`，用于只显示日期部分的 datetime64 对象的格式化
def _format_datetime64_dateonly(
    x: NaTType | Timestamp,
    nat_rep: str = "NaT",
    date_format: str | None = None,
) -> str:
    # 如果 x 是 NaT 类型，返回指定的字符串表示 NaT
    if isinstance(x, NaTType):
        return nat_rep

    # 如果提供了 date_format 参数，则使用 strftime 方法格式化日期
    if date_format:
        return x.strftime(date_format)
    else:
        # 否则使用 Timestamp._date_repr 方法显示日期（依赖于字符串格式化，比 strftime 更快）
        return x._date_repr


# 定义函数 get_format_datetime64，返回一个格式化器函数，接受 datetime64 作为输入并返回字符串
def get_format_datetime64(
    is_dates_only: bool, nat_rep: str = "NaT", date_format: str | None = None
) -> Callable:
    """Return a formatter callable taking a datetime64 as input and providing
    a string as output"""

    # 如果 is_dates_only 为 True，则返回一个 lambda 函数调用 `_format_datetime64_dateonly`
    if is_dates_only:
        return lambda x: _format_datetime64_dateonly(
            x, nat_rep=nat_rep, date_format=date_format
        )
    else:
        # 否则返回一个 lambda 函数调用 `_format_datetime64`
        return lambda x: _format_datetime64(x, nat_rep=nat_rep)


# 定义一个类 `_Datetime64TZFormatter`，继承自 `_Datetime64Formatter`
class _Datetime64TZFormatter(_Datetime64Formatter):
    values: DatetimeArray

    # 定义方法 `_format_strings`，返回格式化后的字符串列表
    def _format_strings(self) -> list[str]:
        """we by definition have a TZ"""
        # 获取是否只有日期信息的标志
        ido = self.values._is_dates_only
        # 将数据转换为对象数组
        values = self.values.astype(object)
        # 获取格式化器函数，如果未指定则使用 get_format_datetime64 函数获取
        formatter = self.formatter or get_format_datetime64(
            ido, date_format=self.date_format
        )
        # 对每个值应用格式化器函数，得到格式化后的值列表
        fmt_values = [formatter(x) for x in values]

        return fmt_values


# 定义一个类 `_Timedelta64Formatter`，继承自 `_GenericArrayFormatter`
class _Timedelta64Formatter(_GenericArrayFormatter):
    values: TimedeltaArray

    def __init__(
        self,
        values: TimedeltaArray,
        nat_rep: str = "NaT",
        **kwargs,
    ) -> None:
        # TODO: nat_rep is never passed, na_rep is.
        # 调用父类构造方法，初始化数据
        super().__init__(values, **kwargs)
        # 设置本地属性 nat_rep 为指定的字符串表示 NaT

        self.nat_rep = nat_rep

    # 定义方法 `_format_strings`，返回格式化后的字符串列表
    def _format_strings(self) -> list[str]:
        # 获取格式化器函数，如果未指定则使用 get_format_timedelta64 函数获取
        formatter = self.formatter or get_format_timedelta64(
            self.values, nat_rep=self.nat_rep, box=False
        )
        # 对每个值应用格式化器函数，得到格式化后的值列表
        return [formatter(x) for x in self.values]


# 定义函数 get_format_timedelta64，返回一个格式化器函数，接受 timedelta 数组作为输入并返回字符串
def get_format_timedelta64(
    values: TimedeltaArray,
    nat_rep: str | float = "NaT",
    box: bool = False,
) -> Callable:
    """
    Return a formatter function for a range of timedeltas.
    These will all have the same format argument

    If box, then show the return in quotes
    """
    # 获取是否仅包含日期信息的标志
    even_days = values._is_dates_only

    # 如果 even_days 为 True，则 format 设为 None
    if even_days:
        format = None
    else:
        # 否则设为 "long"
        format = "long"

    # 定义内部格式化器函数 `_formatter`，接受一个值作为输入并返回格式化后的字符串
    def _formatter(x):
        # 如果 x 是 None 或者是标量且为 NA，则返回指定的字符串表示 NaT
        if x is None or (is_scalar(x) and isna(x)):
            return nat_rep

        # 如果 x 不是 Timedelta 对象，则将其转换为 Timedelta 对象
        if not isinstance(x, Timedelta):
            x = Timedelta(x)

        # 调用 Timedelta._repr_base 方法以指定的格式返回基本表示形式（依赖于字符串格式化，比 strftime 更快）
        result = x._repr_base(format=format)
        # 如果 box 参数为 True，则将结果用单引号包裹
        if box:
            result = f"'{result}'"
        return result

    return _formatter


# 定义一个函数 `_make_fixed_width`，接受一个字符串列表和一个对齐方式参数，并不返回值
def _make_fixed_width(
    strings: list[str],
    justify: str = "right",
):
    # minimum 是一个可选的整数参数，可以为 None
    minimum: int | None = None,
    # adj 是一个可选的 _TextAdjustment 类型参数，可以为 None
    adj: printing._TextAdjustment | None = None,
def _trim_zeros_float(
    str_floats: ArrayLike | list[str], decimal: str = "."
) -> list[str]:
    """
    Trims the maximum number of trailing zeros equally from
    all numbers containing decimals, leaving just one if
    necessary.
    """
    # 初始化一个变量用于存储处理后的字符串数组
    trimmed = str_floats
    # 构造一个正则表达式，用于匹配形如 "123.456" 的数字字符串
    number_regex = re.compile(rf"^\s*[\+-]?[0-9]+\{decimal}[0-9]*$")

    def is_number_with_decimal(x) -> bool:
        # 检查给定的字符串是否匹配定义好的数字正则表达式
        return re.match(number_regex, x) is not None
    # 定义一个函数，用于确定是否需要对字符串数组进行修剪
    def should_trim(values: ArrayLike | list[str]) -> bool:
        """
        Determine if an array of strings should be trimmed.

        Returns True if all numbers containing decimals (defined by the
        above regular expression) within the array end in a zero, otherwise
        returns False.
        """
        # 从输入数组中筛选出所有包含小数的数字字符串
        numbers = [x for x in values if is_number_with_decimal(x)]
        # 返回是否有满足条件的数字，并且这些数字是否都以'0'结尾
        return len(numbers) > 0 and all(x.endswith("0") for x in numbers)

    # 当需要修剪的数组满足 should_trim 函数的条件时，执行修剪操作
    while should_trim(trimmed):
        # 对于包含小数的数字字符串，去掉最后一个字符（即小数点后的数字）
        trimmed = [x[:-1] if is_number_with_decimal(x) else x for x in trimmed]

    # 在需要的情况下，在小数点后面添加一个'0'
    result = [
        x + "0" if is_number_with_decimal(x) and x.endswith(decimal) else x
        for x in trimmed
    ]
    # 返回处理后的结果数组
    return result
# 检查索引对象是否具有命名，返回布尔值
def _has_names(index: Index) -> bool:
    if isinstance(index, MultiIndex):
        # 如果是 MultiIndex 类型，则调用 com.any_not_none 方法检查是否有任何非空的命名
        return com.any_not_none(*index.names)
    else:
        # 否则，直接检查索引对象的 name 属性是否不为 None
        return index.name is not None


class EngFormatter:
    """
    Formats float values according to engineering format.

    Based on matplotlib.ticker.EngFormatter
    """

    # 工程符号的 SI 前缀定义
    ENG_PREFIXES = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "u",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
    }

    def __init__(
        self, accuracy: int | None = None, use_eng_prefix: bool = False
    ) -> None:
        # 初始化 EngFormatter 对象
        self.accuracy = accuracy  # 设置精度
        self.use_eng_prefix = use_eng_prefix  # 是否使用工程符号前缀

    def __call__(self, num: float) -> str:
        """
        Formats a number in engineering notation, appending a letter
        representing the power of 1000 of the original number. Some examples:
        >>> format_eng = EngFormatter(accuracy=0, use_eng_prefix=True)
        >>> format_eng(0)
        ' 0'
        >>> format_eng = EngFormatter(accuracy=1, use_eng_prefix=True)
        >>> format_eng(1_000_000)
        ' 1.0M'
        >>> format_eng = EngFormatter(accuracy=2, use_eng_prefix=False)
        >>> format_eng("-1e-6")
        '-1.00E-06'

        @param num: the value to represent
        @type num: either a numeric value or a string that can be converted to
                   a numeric value (as per decimal.Decimal constructor)

        @return: engineering formatted string
        """
        # 将输入的数值转换为 Decimal 类型
        dnum = Decimal(str(num))

        # 处理特殊值
        if Decimal.is_nan(dnum):
            return "NaN"

        if Decimal.is_infinite(dnum):
            return "inf"

        # 处理负数情况
        sign = 1
        if dnum < 0:  # pragma: no cover
            sign = -1
            dnum = -dnum

        # 计算指数部分
        if dnum != 0:
            pow10 = Decimal(int(math.floor(dnum.log10() / 3) * 3))
        else:
            pow10 = Decimal(0)

        # 将指数限制在 ENG_PREFIXES 定义的范围内
        pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
        pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
        int_pow10 = int(pow10)

        # 根据设置决定使用工程符号前缀还是 E- 或 E+ 表示
        if self.use_eng_prefix:
            prefix = self.ENG_PREFIXES[int_pow10]
        elif int_pow10 < 0:
            prefix = f"E-{-int_pow10:02d}"
        else:
            prefix = f"E+{int_pow10:02d}"

        # 计算有效数字部分
        mant = sign * dnum / (10**pow10)

        # 根据精度格式化输出字符串
        if self.accuracy is None:  # pragma: no cover
            format_str = "{mant: g}{prefix}"
        else:
            format_str = f"{{mant: .{self.accuracy:d}f}}{{prefix}}"

        # 格式化最终输出结果
        formatted = format_str.format(mant=mant, prefix=prefix)

        return formatted


def set_eng_float_format(accuracy: int = 3, use_eng_prefix: bool = False) -> None:
    """
    Format float representation in DataFrame with SI notation.

    Parameters
    ----------
    accuracy : int, optional
        The number of digits after the decimal point (default is 3).
    use_eng_prefix : bool, optional
        Whether to use SI engineering prefixes (default is False).

    """
    # 设置工具库 Pandas 的浮点数显示格式为工程记数法
    accuracy : int, default 3
        Number of decimal digits after the floating point.
    use_eng_prefix : bool, default False
        Whether to represent a value with SI prefixes.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> df = pd.DataFrame([1e-9, 1e-3, 1, 1e3, 1e6])
    >>> df
                  0
    0  1.000000e-09
    1  1.000000e-03
    2  1.000000e+00
    3  1.000000e+03
    4  1.000000e+06
    
    >>> pd.set_eng_float_format(accuracy=1)
    >>> df
               0
    0  1.0E-09
    1  1.0E-03
    2  1.0E+00
    3  1.0E+03
    4  1.0E+06
    
    >>> pd.set_eng_float_format(use_eng_prefix=True)
    >>> df
              0
    0  1.000n
    1  1.000m
    2   1.000
    3  1.000k
    4  1.000M
    
    >>> pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)
    >>> df
            0
    0  1.0n
    1  1.0m
    2   1.0
    3  1.0k
    4  1.0M
    
    >>> pd.set_option("display.float_format", None)  # unset option
    """
    使用工程记数法格式化 Pandas 显示的浮点数
    """
    set_option("display.float_format", EngFormatter(accuracy, use_eng_prefix))
# 为给定的多层级列表计算每个层级中每个索引的长度，并返回一个列表，其中包含多个字典。
def get_level_lengths(
    levels: Any, sentinel: bool | object | str = ""
) -> list[dict[int, int]]:
    """
    对于每个层级中的每个索引，返回索引的长度。

    Parameters
    ----------
    levels : list of lists
        每个层级的值组成的列表。
    sentinel : string, optional
        表示不会在其上开始新索引的值。

    Returns
    -------
    返回字典的列表。每个层级返回一个索引映射，其中键为行中的索引，值为索引的长度。
    """
    # 如果 levels 列表为空，则返回空列表
    if len(levels) == 0:
        return []

    # 控制标志列表，用于跟踪每个列的状态
    control = [True] * len(levels[0])

    # 结果列表
    result = []
    # 遍历每个层级
    for level in levels:
        last_index = 0

        # 长度字典，用于存储每个索引的长度
        lengths = {}
        # 遍历层级中的每个索引
        for i, key in enumerate(level):
            # 如果控制标志为真且当前索引值等于 sentinel，则跳过
            if control[i] and key == sentinel:
                pass
            else:
                # 将控制标志设为假，记录上一个索引到当前索引之间的长度
                control[i] = False
                lengths[last_index] = i - last_index
                last_index = i

        # 记录最后一个索引到末尾的长度
        lengths[last_index] = len(level) - last_index

        # 将长度字典添加到结果列表中
        result.append(lengths)

    # 返回最终的结果列表
    return result


# 向缓冲区中写入字符串列表作为行
def buffer_put_lines(buf: WriteBuffer[str], lines: list[str]) -> None:
    """
    将行追加到缓冲区中。

    Parameters
    ----------
    buf
        要写入的缓冲区
    lines
        要追加的行。
    """
    # 如果 lines 中有任何一个元素是字符串，则将所有元素转换为字符串
    if any(isinstance(x, str) for x in lines):
        lines = [str(x) for x in lines]
    # 将行连接成一个字符串，并写入缓冲区
    buf.write("\n".join(lines))
```