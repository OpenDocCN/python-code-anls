# `D:\src\scipysrc\pandas\pandas\io\excel\_util.py`

```
# 从__future__模块导入annotations功能，用于类型提示的支持
from __future__ import annotations

# 导入collections.abc模块中的若干抽象基类
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    MutableMapping,
    Sequence,
)

# 导入typing模块中的一些类型相关的工具
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    overload,
)

# 从pandas.compat._optional模块导入import_optional_dependency函数
from pandas.compat._optional import import_optional_dependency

# 从pandas.core.dtypes.common模块导入is_integer和is_list_like函数
from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)

# 如果当前处于类型检查状态(TYPE_CHECKING为True)，则导入ExcelWriter类
if TYPE_CHECKING:
    from pandas.io.excel._base import ExcelWriter

    # 定义ExcelWriter_t为ExcelWriter类型的类型变量
    ExcelWriter_t = type[ExcelWriter]

# 创建一个空的可变映射_writers，用于存储ExcelWriter类型的对象
_writers: MutableMapping[str, ExcelWriter_t] = {}


def register_writer(klass: ExcelWriter_t) -> None:
    """
    将引擎注册到excel写入器注册表中。

    您必须使用此方法与``to_excel``集成。

    Parameters
    ----------
    klass : ExcelWriter
        ExcelWriter类型的引擎对象
    """
    # 如果klass不可调用，则抛出值错误
    if not callable(klass):
        raise ValueError("Can only register callables as engines")
    # 获取引擎的名称
    engine_name = klass._engine
    # 将引擎对象存储在_writers字典中，以引擎名称作为键
    _writers[engine_name] = klass


def get_default_engine(ext: str, mode: Literal["reader", "writer"] = "reader") -> str:
    """
    返回给定扩展名的默认读取器/写入器。

    Parameters
    ----------
    ext : str
        要获取默认引擎的Excel文件扩展名。
    mode : str {'reader', 'writer'}
        是否获取读取或写入的默认引擎。
        可以是'reader'或'writer'

    Returns
    -------
    str
        扩展名的默认引擎。
    """
    # 定义默认的读取器映射
    _default_readers = {
        "xlsx": "openpyxl",
        "xlsm": "openpyxl",
        "xlsb": "pyxlsb",
        "xls": "xlrd",
        "ods": "odf",
    }
    # 定义默认的写入器映射
    _default_writers = {
        "xlsx": "openpyxl",
        "xlsm": "openpyxl",
        "xlsb": "pyxlsb",
        "ods": "odf",
    }
    # 断言模式为'reader'或'writer'
    assert mode in ["reader", "writer"]
    # 如果模式为'writer'
    if mode == "writer":
        # 如果安装了xlsxwriter，则优先使用它
        xlsxwriter = import_optional_dependency("xlsxwriter", errors="warn")
        if xlsxwriter:
            _default_writers["xlsx"] = "xlsxwriter"
        # 返回指定扩展名的默认写入引擎
        return _default_writers[ext]
    else:
        # 返回指定扩展名的默认读取引擎
        return _default_readers[ext]


def get_writer(engine_name: str) -> ExcelWriter_t:
    """
    获取指定引擎名称的Excel写入器。

    Parameters
    ----------
    engine_name : str
        要获取的Excel写入器引擎名称。

    Returns
    -------
    ExcelWriter_t
        ExcelWriter_t类型的对象，对应于指定引擎名称。

    Raises
    ------
    ValueError
        如果找不到指定名称的Excel写入器。
    """
    try:
        # 尝试从_writers字典中获取指定名称的写入器对象
        return _writers[engine_name]
    except KeyError as err:
        # 如果找不到，则抛出值错误，并将原始错误包装其中
        raise ValueError(f"No Excel writer '{engine_name}'") from err


def _excel2num(x: str) -> int:
    """
    将类似于'AB'的Excel列名转换为基于0的列索引。

    Parameters
    ----------
    x : str
        要转换为基于0的列索引的Excel列名。

    Returns
    -------
    num : int
        与列名对应的列索引。

    Raises
    ------
    ValueError
        如果Excel列名的某个部分无效。
    """
    index = 0

    # 遍历去除空格后的大写列名字符串x的每个字符
    for c in x.upper().strip():
        # 获取字符的Unicode码点
        cp = ord(c)

        # 如果码点不在'A'到'Z'之间，则抛出值错误
        if cp < ord("A") or cp > ord("Z"):
            raise ValueError(f"Invalid column name: {x}")

        # 计算基于0的列索引
        index = index * 26 + cp - ord("A") + 1

    # 返回计算得到的列索引值
    return index
    # 返回 index 减去 1 的结果
    return index - 1
# 将区域字符串转换为列索引列表

def _range2cols(areas: str) -> list[int]:
    """
    Convert comma separated list of column names and ranges to indices.

    Parameters
    ----------
    areas : str
        A string containing a sequence of column ranges (or areas).

    Returns
    -------
    cols : list
        A list of 0-based column indices.

    Examples
    --------
    >>> _range2cols("A:E")
    [0, 1, 2, 3, 4]
    >>> _range2cols("A,C,Z:AB")
    [0, 2, 25, 26, 27]
    """
    cols: list[int] = []  # 初始化一个空列表用于存放列索引

    for rng in areas.split(","):  # 对于以逗号分隔的区域字符串进行迭代处理
        if ":" in rng:  # 如果区域字符串包含冒号，表示是一个范围
            rngs = rng.split(":")
            # 扩展cols列表，将范围内的列索引加入列表中
            cols.extend(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
        else:
            # 否则直接将单个列索引加入cols列表中
            cols.append(_excel2num(rng))

    return cols


@overload
def maybe_convert_usecols(usecols: str | list[int]) -> list[int]: ...


@overload
def maybe_convert_usecols(usecols: list[str]) -> list[str]: ...


@overload
def maybe_convert_usecols(usecols: usecols_func) -> usecols_func: ...


@overload
def maybe_convert_usecols(usecols: None) -> None: ...


def maybe_convert_usecols(
    usecols: str | list[int] | list[str] | usecols_func | None,
) -> None | list[int] | list[str] | usecols_func:
    """
    Convert `usecols` into a compatible format for parsing in `parsers.py`.

    Parameters
    ----------
    usecols : object
        The use-columns object to potentially convert.

    Returns
    -------
    converted : object
        The compatible format of `usecols`.
    """
    if usecols is None:
        return usecols  # 如果usecols为None，直接返回None

    if is_integer(usecols):
        raise ValueError(
            "Passing an integer for `usecols` is no longer supported.  "
            "Please pass in a list of int from 0 to `usecols` inclusive instead."
        )

    if isinstance(usecols, str):
        return _range2cols(usecols)  # 如果usecols是字符串，则调用_range2cols函数进行转换

    return usecols  # 否则直接返回usecols本身，已经是合适的格式


@overload
def validate_freeze_panes(freeze_panes: tuple[int, int]) -> Literal[True]: ...


@overload
def validate_freeze_panes(freeze_panes: None) -> Literal[False]: ...


def validate_freeze_panes(freeze_panes: tuple[int, int] | None) -> bool:
    """
    Validate the freeze panes tuple format.

    Parameters
    ----------
    freeze_panes : tuple[int, int] | None
        The tuple representing freeze panes coordinates.

    Returns
    -------
    bool
        True if freeze_panes is valid, False otherwise.
    """
    if freeze_panes is not None:
        if len(freeze_panes) == 2 and all(
            isinstance(item, int) for item in freeze_panes
        ):
            return True  # 如果freeze_panes是长度为2且所有元素都是整数，则返回True

        raise ValueError(
            "freeze_panes must be of form (row, column) "
            "where row and column are integers"
        )

    # 如果freeze_panes未指定，返回False，表示不应用到输出表格中
    return False


def fill_mi_header(
    row: list[Hashable], control_row: list[bool]
) -> tuple[list[Hashable], list[bool]]:
    """
    Forward fill blank entries in row but only inside the same parent index.

    Used for creating headers in Multiindex.

    Parameters
    ----------
    row : list
        List of items in a single row.
    control_row : list
        List indicating where forward fill should apply.

    Returns
    -------
    tuple
        Tuple containing the filled row and control_row after modification.
    """
    # control_row 是一个布尔值列表，用于确定特定列是否与上一个值在同一父索引下。
    # 这有助于阻止空单元格在不同索引之间的传播。
    last = row[0]  # 将第一个元素设为初始值
    for i in range(1, len(row)):  # 循环遍历除第一个元素外的所有元素
        if not control_row[i]:  # 如果 control_row[i] 为 False
            last = row[i]  # 更新 last 为当前元素的值

        if row[i] == "" or row[i] is None:  # 如果当前元素为空字符串或 None
            row[i] = last  # 将当前元素设为 last 的值（用于填充空单元格）
        else:
            control_row[i] = False  # 将 control_row[i] 设为 False，表示该列不再与上一个值在同一父索引下
            last = row[i]  # 更新 last 为当前元素的值

    # 返回更新后的行和修改后的 control_row 列表
    return row, control_row
# 弹出用于 MultiIndex 解析的表头名称

def pop_header_name(
    row: list[Hashable], index_col: int | Sequence[int]
) -> tuple[Hashable | None, list[Hashable]]:
    """
    Pop the header name for MultiIndex parsing.

    Parameters
    ----------
    row : list
        要解析表头名称的数据行。
    index_col : int, list
        数据的索引列。假定为非空。

    Returns
    -------
    header_name : Hashable or None
        提取的表头名称。
    trimmed_row : list
        原始数据行删除表头名称后的结果。
    """
    # 弹出表头名称并填充为空白。
    if is_list_like(index_col):
        assert isinstance(index_col, Iterable)
        i = max(index_col)
    else:
        assert not isinstance(index_col, Iterable)
        i = index_col

    header_name = row[i]
    header_name = None if header_name == "" else header_name

    return header_name, row[:i] + [""] + row[i + 1 :]


def combine_kwargs(engine_kwargs: dict[str, Any] | None, kwargs: dict) -> dict:
    """
    用于合并后端引擎的两个 kwargs 源。

    使用 kwargs 已经被弃用，此函数仅用于 1.3 版本，并应在 1.4/2.0 版本中移除。
    此外 _base.ExcelWriter.__new__ 确保 engine_kwargs 或 kwargs 必须分别为 None 或空。

    Parameters
    ----------
    engine_kwargs: dict
        要传递给引擎的 kwargs。
    kwargs: dict
        要传递给引擎的 kwargs（已弃用）。

    Returns
    -------
    engine_kwargs 与 kwargs 结合后的结果。
    """
    if engine_kwargs is None:
        result = {}
    else:
        result = engine_kwargs.copy()
    result.update(kwargs)
    return result
```