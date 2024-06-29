# `D:\src\scipysrc\pandas\pandas\io\parsers\python_parser.py`

```
# 从未来版本导入注释语法以支持类型提示的类型注释
from __future__ import annotations

# 导入 collections 模块的 abc 和 defaultdict 类
from collections import (
    abc,
    defaultdict,
)

# 导入 csv 模块用于 CSV 文件处理
import csv

# 导入 StringIO 类用于在内存中操作字符串
from io import StringIO

# 导入 re 模块用于正则表达式操作
import re

# 导入 TYPE_CHECKING 用于类型检查
from typing import (
    IO,
    TYPE_CHECKING,
    DefaultDict,
    Literal,
    cast,
)

# 导入警告模块，用于处理警告信息
import warnings

# 导入 numpy 库
import numpy as np

# 导入 pandas 库中的相关模块和类
from pandas._libs import lib
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
    is_numeric_dtype,
)
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
    dedup_names,
    is_potential_multi_index,
)
from pandas.io.parsers.base_parser import (
    ParserBase,
    parser_defaults,
)

# 如果是类型检查状态，导入额外的类型
if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
        Mapping,
        Sequence,
    )
    from pandas._typing import (
        ArrayLike,
        ReadCsvBuffer,
        Scalar,
        T,
    )
    from pandas import (
        Index,
        MultiIndex,
        Series,
    )

# BOM 字符（字节顺序标记）
# 文件开头的特殊字符，用于指示文件的字节顺序
# 这个标记会影响解析，需要在处理时移除
_BOM = "\ufeff"


class PythonParser(ParserBase):
    _no_thousands_columns: set[int]

    @cache_readonly
    def num(self) -> re.Pattern:
        # 创建用于匹配数字的正则表达式模式对象
        decimal = re.escape(self.decimal)
        if self.thousands is None:
            regex = rf"^[\-\+]?[0-9]*({decimal}[0-9]*)?([0-9]?(E|e)\-?[0-9]+)?$"
        else:
            thousands = re.escape(self.thousands)
            regex = (
                rf"^[\-\+]?([0-9]+{thousands}|[0-9])*({decimal}[0-9]*)?"
                rf"([0-9]?(E|e)\-?[0-9]+)?$"
            )
        return re.compile(regex)
    # 定义一个私有方法 `_make_reader`，接受一个文件对象或者读取 CSV 缓冲区的输入，返回一个迭代器，迭代的元素是字符串列表
    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]) -> Iterator[list[str]]:
        # 获取分隔符
        sep = self.delimiter
        
        # 如果分隔符为 None 或者长度为 1，则执行以下操作
        if sep is None or len(sep) == 1:
            # 如果设置了行终止符，则抛出异常，因为目前不支持自定义行终止符
            if self.lineterminator:
                raise ValueError(
                    "Custom line terminators not supported in python parser (yet)"
                )
            
            # 定义一个内部类 MyDialect，继承自 csv.Dialect
            class MyDialect(csv.Dialect):
                delimiter = self.delimiter
                quotechar = self.quotechar
                escapechar = self.escapechar
                doublequote = self.doublequote
                skipinitialspace = self.skipinitialspace
                quoting = self.quoting
                lineterminator = "\n"

            # 实例化 MyDialect 类
            dia = MyDialect

            # 如果分隔符不为 None，则将 MyDialect 的分隔符设置为 sep
            if sep is not None:
                dia.delimiter = sep
            else:
                # 尝试从第一个有效行中猜测分隔符，即不是注释行且不在跳过行中
                line = f.readline()
                lines = self._check_comments([[line]])[0]
                while self.skipfunc(self.pos) or not lines:
                    self.pos += 1
                    line = f.readline()
                    lines = self._check_comments([[line]])[0]
                lines_str = cast(list[str], lines)

                # 因为 `line` 是一个字符串，`lines` 将会是一个只包含一个字符串的列表
                line = lines_str[0]

                self.pos += 1
                self.line_pos += 1
                # 使用 csv.Sniffer 自动检测分隔符
                sniffed = csv.Sniffer().sniff(line)
                dia.delimiter = sniffed.delimiter

                # 注意: 在这里编码是不相关的
                # 使用 MyDialect 解析 `line`，并将结果扩展到 `self.buf` 中
                line_rdr = csv.reader(StringIO(line), dialect=dia)
                self.buf.extend(list(line_rdr))

            # 注意: 在这里编码是不相关的
            # 使用 MyDialect 创建一个 CSV 读取器，使用给定的文件对象 `f`，严格模式为 True
            reader = csv.reader(f, dialect=dia, strict=True)

        else:
            # 如果分隔符不为单字符，则执行以下操作
            # 定义一个内部函数 `_read`，逐行读取文件对象 `f` 中的内容，并使用正则表达式 `sep` 分割每行
            def _read():
                line = f.readline()
                pat = re.compile(sep)

                yield pat.split(line.strip())

                for line in f:
                    yield pat.split(line.strip())

            # 使用 `_read` 函数创建一个自定义的读取器
            reader = _read()

        # 返回最终的读取器对象
        return reader
    # 针对索引行获取行内容，处理可能的 StopIteration 异常
    try:
        content = self._get_lines(rows)
    except StopIteration:
        # 如果是首个数据块，返回空列表；否则关闭流并向上抛出异常
        if self._first_chunk:
            content = []
        else:
            self.close()
            raise

    # 标记已完成首次读取，下次将会触发 StopIteration
    self._first_chunk = False

    index: Index | None
    # 使用原始列名创建列序列
    columns: Sequence[Hashable] = list(self.orig_names)
    if not len(content):  # pragma: no cover
        # 如果内容为空，生成一个具有正确元数据的 DataFrame
        # 错误：无法确定 'index_col' 的类型
        names = dedup_names(
            self.orig_names,
            is_potential_multi_index(
                self.orig_names,
                self.index_col,  # type: ignore[has-type]
            ),
        )
        # 获取空元数据信息
        index, columns, col_dict = self._get_empty_meta(
            names,
            self.dtype,
        )
        # 可能创建多重索引列
        conv_columns = self._maybe_make_multi_index_columns(columns, self.col_names)
        return index, conv_columns, col_dict

    # 处理索引名称在内容中的新样式
    count_empty_content_vals = count_empty_vals(content[0])
    indexnamerow = None
    if self.has_index_names and count_empty_content_vals == len(columns):
        indexnamerow = content[0]
        content = content[1:]

    # 将行数据转换为列数据
    alldata = self._rows_to_cols(content)
    # 排除隐式索引并更新列名
    data, columns = self._exclude_implicit_index(alldata)

    # 转换数据格式
    conv_data = self._convert_data(data)
    conv_data = self._do_date_conversions(columns, conv_data)

    # 创建索引和结果列
    index, result_columns = self._make_index(
        conv_data, alldata, columns, indexnamerow
    )

    return index, result_columns, conv_data


def _exclude_implicit_index(
    self,
    alldata: list[np.ndarray],
) -> tuple[Mapping[Hashable, np.ndarray], Sequence[Hashable]]:
    # 错误：无法确定 'index_col' 的类型
    names = dedup_names(
        self.orig_names,
        is_potential_multi_index(
            self.orig_names,
            self.index_col,  # type: ignore[has-type]
        ),
    )

    offset = 0
    if self._implicit_index:
        # 错误：无法确定 'index_col' 的类型
        offset = len(self.index_col)  # type: ignore[has-type]

    len_alldata = len(alldata)
    # 检查数据长度与名称匹配
    self._check_data_length(names, alldata)

    return {
        name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata
    }, names


# 旧版接口
def get_chunk(
    self, size: int | None = None
) -> tuple[
    Index | None,
    Sequence[Hashable] | MultiIndex,
    Mapping[Hashable, ArrayLike | Series],
]:
    if size is None:
        # 错误：无法确定 "PythonParser" 的属性 "chunksize"
        size = self.chunksize  # type: ignore[attr-defined]
    return self.read(rows=size)
    def _convert_data(
        self,
        data: Mapping[Hashable, np.ndarray],
    ) -> Mapping[Hashable, ArrayLike]:
        # 应用转换器
        clean_conv = self._clean_mapping(self.converters)
        # 清理数据类型映射
        clean_dtypes = self._clean_mapping(self.dtype)

        # 应用缺失值处理
        clean_na_values = {}
        clean_na_fvalues = {}

        if isinstance(self.na_values, dict):
            # 遍历每列的缺失值设置
            for col in self.na_values:
                if col is not None:
                    na_value = self.na_values[col]
                    na_fvalue = self.na_fvalues[col]

                    # 如果列标识为整数且不在原始列名中，转换为原始列名
                    if isinstance(col, int) and col not in self.orig_names:
                        col = self.orig_names[col]

                    # 添加到清理后的缺失值字典中
                    clean_na_values[col] = na_value
                    clean_na_fvalues[col] = na_fvalue
        else:
            # 如果缺失值设置不是字典，直接赋值
            clean_na_values = self.na_values
            clean_na_fvalues = self.na_fvalues

        # 调用方法，将数据转换为 NumPy 数组形式
        return self._convert_to_ndarrays(
            data,
            clean_na_values,
            clean_na_fvalues,
            clean_conv,
            clean_dtypes,
        )

    @cache_readonly
    def _have_mi_columns(self) -> bool:
        # 如果未指定表头，则返回 False
        if self.header is None:
            return False

        header = self.header
        # 如果表头是列表、元组或 NumPy 数组，则判断其长度是否大于 1
        if isinstance(header, (list, tuple, np.ndarray)):
            return len(header) > 1
        else:
            return False

    def _infer_columns(
        self,
    @cache_readonly
    def _header_line(self):
        # 存储行以便在 _get_index_name 中重用
        if self.header is not None:
            return None

        try:
            # 尝试从缓冲区获取行
            line = self._buffered_line()
        except StopIteration as err:
            # 如果缓冲区为空且没有指定列名，则引发异常
            if not self.names:
                raise EmptyDataError("No columns to parse from file") from err

            # 否则，使用已知的列名
            line = self.names[:]
        return line

    def _handle_usecols(
        self,
        columns: list[list[Scalar | None]],
        usecols_key: list[Scalar | None],
        num_original_columns: int,
        """
        Sets self._col_indices

        usecols_key is used if there are string usecols.
        """
        # 声明变量 col_indices，可能是整数集合或列表
        col_indices: set[int] | list[int]
        # 如果 self.usecols 不为空，则处理 usecols
        if self.usecols is not None:
            # 如果 usecols 是可调用对象，则通过 _evaluate_usecols 方法获取列索引
            if callable(self.usecols):
                col_indices = self._evaluate_usecols(self.usecols, usecols_key)
            # 如果 usecols 中包含字符串，则处理每个字符串
            elif any(isinstance(u, str) for u in self.usecols):
                # 如果 columns 多于一个，则抛出 ValueError
                if len(columns) > 1:
                    raise ValueError(
                        "If using multiple headers, usecols must be integers."
                    )
                # 初始化列索引列表
                col_indices = []
                # 遍历 usecols 中的每个元素
                for col in self.usecols:
                    # 如果元素是字符串，则尝试获取其在 usecols_key 中的索引
                    if isinstance(col, str):
                        try:
                            col_indices.append(usecols_key.index(col))
                        except ValueError:
                            # 如果未找到对应的列名，则调用 _validate_usecols_names 方法验证 usecols
                            self._validate_usecols_names(self.usecols, usecols_key)
                    # 如果元素不是字符串，则直接加入列索引列表
                    else:
                        col_indices.append(col)
            # 如果 usecols 中不包含字符串，直接赋值给 col_indices
            else:
                # 检查 usecols 中是否有超出原始列数的索引
                missing_usecols = [
                    col for col in self.usecols if col >= num_original_columns
                ]
                # 如果有超出范围的索引，则抛出 ParserError 异常
                if missing_usecols:
                    raise ParserError(
                        "Defining usecols with out-of-bounds indices is not allowed. "
                        f"{missing_usecols} are out-of-bounds.",
                    )
                col_indices = self.usecols

            # 根据 col_indices 从 columns 中选取对应的列，并更新 columns
            columns = [
                [n for i, n in enumerate(column) if i in col_indices]
                for column in columns
            ]
            # 将 col_indices 排序后赋值给 self._col_indices
            self._col_indices = sorted(col_indices)
        # 返回处理后的 columns 列表
        return columns

    def _buffered_line(self) -> list[Scalar]:
        """
        Return a line from buffer, filling buffer if required.
        """
        # 如果 buf 中有数据，则返回 buf 中的第一行
        if len(self.buf) > 0:
            return self.buf[0]
        else:
            # 否则调用 _next_line 方法返回下一行数据
            return self._next_line()
    # 检查是否存在 BOM 字符，并在必要时移除。如果第一个元素之后存在引号，也移除引号，因为引号技术上位于名称开头而不是中间。
    def _check_for_bom(self, first_row: list[Scalar]) -> list[Scalar]:
        """
        Checks whether the file begins with the BOM character.
        If it does, remove it. In addition, if there is quoting
        in the field subsequent to the BOM, remove it as well
        because it technically takes place at the beginning of
        the name, not the middle of it.
        """
        # first_row will be a list, so we need to check
        # that that list is not empty before proceeding.
        if not first_row:
            return first_row

        # The first element of this row is the one that could have the
        # BOM that we want to remove. Check that the first element is a
        # string before proceeding.
        if not isinstance(first_row[0], str):
            return first_row

        # Check that the string is not empty, as that would
        # obviously not have a BOM at the start of it.
        if not first_row[0]:
            return first_row

        # Since the string is non-empty, check that it does
        # in fact begin with a BOM.
        first_elt = first_row[0][0]
        if first_elt != _BOM:
            return first_row

        first_row_bom = first_row[0]
        new_row: str

        if len(first_row_bom) > 1 and first_row_bom[1] == self.quotechar:
            start = 2
            quote = first_row_bom[1]
            end = first_row_bom[2:].index(quote) + 2

            # Extract the data between the quotation marks
            new_row = first_row_bom[start:end]

            # Extract any remaining data after the second
            # quotation mark.
            if len(first_row_bom) > end + 1:
                new_row += first_row_bom[end + 1:]

        else:
            # No quotation so just remove BOM from first element
            new_row = first_row_bom[1:]

        new_row_list: list[Scalar] = [new_row]
        return new_row_list + first_row[1:]

    # 检查一行数据是否为空。
    def _is_line_empty(self, line: Sequence[Scalar]) -> bool:
        """
        Check if a line is empty or not.

        Parameters
        ----------
        line : str, array-like
            The line of data to check.

        Returns
        -------
        boolean : Whether or not the line is empty.
        """
        return not line or all(not x for x in line)
    # 返回类型为 list[Scalar] 的下一行数据
    def _next_line(self) -> list[Scalar]:
        # 如果 self.data 是一个列表
        if isinstance(self.data, list):
            # 跳过由 self.skipfunc 指定的位置，直到不需要跳过为止
            while self.skipfunc(self.pos):
                # 如果 self.pos 超出了列表长度，则跳出循环
                if self.pos >= len(self.data):
                    break
                # 移动到下一个位置
                self.pos += 1

            while True:
                try:
                    # 检查并返回不包含注释的行，取列表中第 self.pos 位置的行
                    line = self._check_comments([self.data[self.pos]])[0]
                    # 移动到下一个位置
                    self.pos += 1
                    # 如果不跳过空白行，并且当前行为空或不为空行
                    if not self.skip_blank_lines and (
                        self._is_line_empty(self.data[self.pos - 1]) or line
                    ):
                        break
                    # 如果跳过空白行
                    if self.skip_blank_lines:
                        # 移除空行，并获取第一行
                        ret = self._remove_empty_lines([line])
                        if ret:
                            line = ret[0]
                            break
                except IndexError as err:
                    # 如果索引错误，抛出 StopIteration 异常
                    raise StopIteration from err
        else:
            # 如果 self.data 不是列表
            while self.skipfunc(self.pos):
                # 移动到下一个位置
                self.pos += 1
                # 调用迭代器的下一个元素
                next(self.data)

            while True:
                # 获取迭代器的下一个行
                orig_line = self._next_iter_line(row_num=self.pos + 1)
                # 移动到下一个位置
                self.pos += 1

                if orig_line is not None:
                    # 检查并返回不包含注释的行
                    line = self._check_comments([orig_line])[0]

                    # 如果跳过空白行
                    if self.skip_blank_lines:
                        # 移除空行，并获取第一行
                        ret = self._remove_empty_lines([line])

                        if ret:
                            line = ret[0]
                            break
                    # 如果不跳过空白行，并且当前行为空或不为空行
                    elif self._is_line_empty(orig_line) or line:
                        break

        # 如果 self.pos 为 1，则当前行可能包含 BOM（文件开头的字节顺序标记）
        if self.pos == 1:
            # 检查并移除 BOM
            line = self._check_for_bom(line)

        # 增加行数计数器
        self.line_pos += 1
        # 将当前行添加到缓冲区
        self.buf.append(line)
        # 返回当前行
        return line

    # 提示用户关于格式错误的警告信息
    def _alert_malformed(self, msg: str, row_num: int) -> None:
        """
        Alert a user about a malformed row, depending on value of
        `self.on_bad_lines` enum.

        If `self.on_bad_lines` is ERROR, the alert will be `ParserError`.
        If `self.on_bad_lines` is WARN, the alert will be printed out.

        Parameters
        ----------
        msg: str
            The error message to display.
        row_num: int
            The row number where the parsing error occurred.
            Because this row number is displayed, we 1-index,
            even though we 0-index internally.
        """
        # 如果设置为处理错误行的枚举值为 ERROR，则抛出 ParserError 异常
        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
            raise ParserError(msg)
        # 如果设置为处理错误行的枚举值为 WARN，则发出警告信息
        if self.on_bad_lines == self.BadLineHandleMethod.WARN:
            warnings.warn(
                f"Skipping line {row_num}: {msg}\n",
                ParserWarning,
                stacklevel=find_stack_level(),
            )
    def _next_iter_line(self, row_num: int) -> list[Scalar] | None:
        """
        Wrapper around iterating through `self.data` (CSV source).

        When a CSV error is raised, we check for specific
        error messages that allow us to customize the
        error message displayed to the user.

        Parameters
        ----------
        row_num: int
            The row number of the line being parsed.
        """
        try:
            assert not isinstance(self.data, list)
            # Attempt to fetch the next line from self.data
            line = next(self.data)
            # lie about list[str] vs list[Scalar] to minimize ignores
            return line  # type: ignore[return-value]
        except csv.Error as e:
            # Handle CSV errors
            if self.on_bad_lines in (
                self.BadLineHandleMethod.ERROR,
                self.BadLineHandleMethod.WARN,
            ):
                msg = str(e)

                # Customize error message for specific error conditions
                if "NULL byte" in msg or "line contains NUL" in msg:
                    msg = (
                        "NULL byte detected. This byte "
                        "cannot be processed in Python's "
                        "native csv library at the moment, "
                        "so please pass in engine='c' instead"
                    )

                # Append information about skipped footer rows to the error message
                if self.skipfooter > 0:
                    reason = (
                        "Error could possibly be due to "
                        "parsing errors in the skipped footer rows "
                        "(the skipfooter keyword is only applied "
                        "after Python's csv library has parsed "
                        "all rows)."
                    )
                    msg += ". " + reason

                # Raise alert about malformed data with the constructed error message
                self._alert_malformed(msg, row_num)
            return None

    def _check_comments(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        """
        Checks for and removes comments from lines of data based on self.comment.

        Parameters
        ----------
        lines: list[list[Scalar]]
            List of lines where comments need to be checked and potentially removed.

        Returns
        -------
        list[list[Scalar]]
            Processed lines with comments removed where applicable.
        """
        if self.comment is None:
            return lines
        ret = []
        for line in lines:
            rl = []
            for x in line:
                if (
                    not isinstance(x, str)
                    or self.comment not in x
                    or x in self.na_values
                ):
                    rl.append(x)
                else:
                    # Remove comment and append the cleaned segment to rl
                    x = x[: x.find(self.comment)]
                    if len(x) > 0:
                        rl.append(x)
                    break
            ret.append(rl)
        return ret
    def _remove_empty_lines(self, lines: list[list[T]]) -> list[list[T]]:
        """
        Iterate through the lines and remove any that are
        either empty or contain only one whitespace value

        Parameters
        ----------
        lines : list of list of Scalars
            The array of lines that we are to filter.

        Returns
        -------
        filtered_lines : list of list of Scalars
            The same array of lines with the "empty" ones removed.
        """
        # 生成一个新的列表，其中排除了空行和只包含一个空格值的行
        ret = [
            line
            for line in lines
            if (
                len(line) > 1
                or len(line) == 1
                and (not isinstance(line[0], str) or line[0].strip())
            )
        ]
        return ret

    def _check_thousands(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        """
        Check if the 'thousands' attribute is set. If not, return the lines unchanged.
        Otherwise, replace occurrences of 'thousands' with an empty string.

        Parameters
        ----------
        lines : list of list of Scalar
            The lines to be processed.

        Returns
        -------
        list of list of Scalar
            Processed lines with 'thousands' replaced if applicable.
        """
        if self.thousands is None:
            return lines

        # 调用私有方法来替换 lines 中的 'thousands' 字符串为空字符串
        return self._search_replace_num_columns(
            lines=lines, search=self.thousands, replace=""
        )

    def _search_replace_num_columns(
        self, lines: list[list[Scalar]], search: str, replace: str
    ) -> list[list[Scalar]]:
        """
        Search and replace occurrences of a specific string ('search') with another string ('replace')
        in each line of lines.

        Parameters
        ----------
        lines : list of list of Scalar
            The lines to be processed.
        search : str
            The string to search for.
        replace : str
            The string to replace 'search' with.

        Returns
        -------
        list of list of Scalar
            Processed lines with replacements made as specified.
        """
        ret = []
        for line in lines:
            rl = []
            for i, x in enumerate(line):
                # 检查是否需要替换 'search'，如果不需要则保留原值
                if (
                    not isinstance(x, str)
                    or search not in x
                    or i in self._no_thousands_columns
                    or not self.num.search(x.strip())
                ):
                    rl.append(x)
                else:
                    rl.append(x.replace(search, replace))
            ret.append(rl)
        return ret

    def _check_decimal(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        """
        Check if the 'decimal' attribute matches the default parser decimal.
        If it does, return the lines unchanged. Otherwise, replace occurrences
        of 'decimal' with a dot ('.').

        Parameters
        ----------
        lines : list of list of Scalar
            The lines to be processed.

        Returns
        -------
        list of list of Scalar
            Processed lines with 'decimal' replaced if applicable.
        """
        if self.decimal == parser_defaults["decimal"]:
            return lines

        # 调用私有方法来替换 lines 中的 'decimal' 字符串为点号 '.'
        return self._search_replace_num_columns(
            lines=lines, search=self.decimal, replace="."
        )

    def _clear_buffer(self) -> None:
        """
        Clear the internal buffer 'buf'.
        """
        self.buf = []

    def _get_index_name(
        self,
    ) -> tuple[Sequence[Hashable] | None, list[Hashable], list[Hashable]]:
        """
        尝试多种情况来获取行：

        0) 在第 0 行和第 1 行有表头，并且它们的总长度等于下一行的长度。
        将第 0 行视为列名，第 1 行视为索引
        1) 寻找隐式索引：第 1 行的列数多于第 0 行。如果为真，则假定第 1 行列出索引列，第 0 行列出普通列。
        2) 如果列出了索引，则从列中获取索引。
        """
        columns: Sequence[Hashable] = self.orig_names
        orig_names = list(columns)
        columns = list(columns)

        line: list[Scalar] | None
        if self._header_line is not None:
            line = self._header_line
        else:
            try:
                line = self._next_line()
            except StopIteration:
                line = None

        next_line: list[Scalar] | None
        try:
            next_line = self._next_line()
        except StopIteration:
            next_line = None

        # 隐式设定 index_col=0 因为列名少一个
        implicit_first_cols = 0
        if line is not None:
            # 留在 0，#2442
            # Case 1
            # 错误：无法确定 'index_col' 的类型
            index_col = self.index_col  # type: ignore[has-type]
            if index_col is not False:
                implicit_first_cols = len(line) - self.num_original_columns

            # Case 0
            if (
                next_line is not None
                and self.header is not None
                and index_col is not False
            ):
                if len(next_line) == len(line) + self.num_original_columns:
                    # 列名和索引名在不同行上
                    self.index_col = list(range(len(line)))
                    self.buf = self.buf[1:]

                    for c in reversed(line):
                        columns.insert(0, c)

                    # 更新原始名称列表以包含所有索引。
                    orig_names = list(columns)
                    self.num_original_columns = len(columns)
                    return line, orig_names, columns

        if implicit_first_cols > 0:
            # Case 1
            self._implicit_index = True
            if self.index_col is None:
                self.index_col = list(range(implicit_first_cols))

            index_name = None

        else:
            # Case 2
            (index_name, _, self.index_col) = self._clean_index_names(
                columns, self.index_col
            )

        return index_name, orig_names, columns
    # 定义一个方法 `_get_lines`，返回一个列表，其中每个元素是一个包含标量的列表
    def _get_lines(self, rows: int | None = None) -> list[list[Scalar]]:
        # 将对象属性 `buf` 赋值给局部变量 `lines`
        lines = self.buf
        new_rows = None

        # 如果参数 `rows` 不为 None
        if rows is not None:
            # 如果当前缓冲区的行数大于或等于参数 `rows`
            if len(self.buf) >= rows:
                # 将前 `rows` 行赋值给 `new_rows`，同时更新缓冲区剩余的行数到 `self.buf`
                new_rows, self.buf = self.buf[:rows], self.buf[rows:]
            else:
                # 如果缓冲区的行数不足以满足参数 `rows`，则计算需要额外获取的行数
                rows -= len(self.buf)

        # 如果 `new_rows` 仍为 None
        if new_rows is None:
            # 如果对象属性 `data` 是列表类型
            if isinstance(self.data, list):
                # 如果当前位置 `pos` 超出了数据列表的长度，抛出 StopIteration 异常
                if self.pos > len(self.data):
                    raise StopIteration
                # 如果 `rows` 为 None，则获取从当前位置到列表末尾的所有行
                if rows is None:
                    new_rows = self.data[self.pos :]
                    new_pos = len(self.data)
                else:
                    # 否则获取从当前位置到 `pos + rows` 位置的行
                    new_rows = self.data[self.pos : self.pos + rows]
                    new_pos = self.pos + rows

                # 调用 `_remove_skipped_rows` 方法，移除跳过的行
                new_rows = self._remove_skipped_rows(new_rows)
                # 将新获取的行添加到 `lines` 中
                lines.extend(new_rows)
                # 更新当前位置 `pos` 为 `new_pos`
                self.pos = new_pos

            else:
                # 如果对象属性 `data` 不是列表类型，则初始化空列表 `new_rows`
                new_rows = []
                try:
                    # 如果 `rows` 不为 None，则迭代处理数据生成器的行
                    if rows is not None:
                        row_index = 0
                        row_ct = 0
                        offset = self.pos if self.pos is not None else 0
                        # 循环直到获取到 `rows` 行数据或者迭代结束
                        while row_ct < rows:
                            new_row = next(self.data)
                            # 如果不跳过当前行，则累计有效行数 `row_ct`
                            if not self.skipfunc(offset + row_index):
                                row_ct += 1
                            row_index += 1
                            # 将新行添加到 `new_rows` 列表中
                            new_rows.append(new_row)

                        # 获取 `new_rows` 的长度
                        len_new_rows = len(new_rows)
                        # 调用 `_remove_skipped_rows` 方法，移除跳过的行
                        new_rows = self._remove_skipped_rows(new_rows)
                        # 将新获取的行添加到 `lines` 中
                        lines.extend(new_rows)
                    else:
                        # 如果 `rows` 为 None，则循环直到迭代结束
                        rows = 0
                        while True:
                            # 调用 `_next_iter_line` 方法获取下一行数据
                            next_row = self._next_iter_line(row_num=self.pos + rows + 1)
                            rows += 1
                            # 如果获取到新行，则将其添加到 `new_rows` 列表中
                            if next_row is not None:
                                new_rows.append(next_row)
                        # 获取 `new_rows` 的长度
                        len_new_rows = len(new_rows)

                except StopIteration:
                    # 处理 StopIteration 异常，获取 `new_rows` 的长度
                    len_new_rows = len(new_rows)
                    # 调用 `_remove_skipped_rows` 方法，移除跳过的行
                    new_rows = self._remove_skipped_rows(new_rows)
                    # 将新获取的行添加到 `lines` 中
                    lines.extend(new_rows)
                    # 如果 `lines` 为空，则重新抛出异常
                    if len(lines) == 0:
                        raise
                # 更新当前位置 `pos` 为 `pos + len_new_rows`
                self.pos += len_new_rows

            # 清空缓冲区
            self.buf = []
        else:
            # 否则将 `lines` 赋值为 `new_rows`
            lines = new_rows

        # 如果 `skipfooter` 属性为真，则移除末尾的指定行数 `skipfooter`
        if self.skipfooter:
            lines = lines[: -self.skipfooter]

        # 调用 `_check_comments` 方法，检查并处理注释行
        lines = self._check_comments(lines)
        # 如果 `skip_blank_lines` 属性为真，则移除空行
        if self.skip_blank_lines:
            lines = self._remove_empty_lines(lines)
        # 调用 `_check_thousands` 方法，检查并处理千位分隔符
        lines = self._check_thousands(lines)
        # 调用 `_check_decimal` 方法，检查并处理小数点格式
        return self._check_decimal(lines)
    # 如果存在跳过行的条件（self.skiprows 不为空），则根据条件跳过部分行
    def _remove_skipped_rows(self, new_rows: list[list[Scalar]]) -> list[list[Scalar]]:
        if self.skiprows:
            # 使用列表推导式过滤掉应跳过的行，条件由 self.skipfunc(i + self.pos) 决定
            return [
                row for i, row in enumerate(new_rows) if not self.skipfunc(i + self.pos)
            ]
        # 如果没有跳过行的条件，则直接返回新行列表 new_rows
        return new_rows

    # 设置不处理千分位格式的列索引集合
    def _set_no_thousand_columns(self) -> set[int]:
        # 初始化一个空集合，用于存储不处理千分位格式的列的索引
        no_thousands_columns: set[int] = set()
        
        # 如果同时定义了列和日期解析选项（self.columns 和 self.parse_dates 均不为空）
        if self.columns and self.parse_dates:
            # 断言确保列索引已经初始化
            assert self._col_indices is not None
            # 调用内部方法 _set_noconvert_dtype_columns，确定不需要转换数据类型的列索引集合
            no_thousands_columns = self._set_noconvert_dtype_columns(
                self._col_indices, self.columns
            )
        
        # 如果同时定义了列和数据类型选项（self.columns 和 self.dtype 均不为空）
        if self.columns and self.dtype:
            # 再次断言确保列索引已经初始化
            assert self._col_indices is not None
            
            # 遍历列索引和列名
            for i, col in zip(self._col_indices, self.columns):
                # 如果数据类型不是字典或者不是数值类型，则将该列索引添加到 no_thousands_columns 集合中
                if not isinstance(self.dtype, dict) and not is_numeric_dtype(
                    self.dtype
                ):
                    no_thousands_columns.add(i)
                
                # 如果数据类型是字典，并且列名在字典中，并且数据类型不是数值或布尔类型，则将该列索引添加到 no_thousands_columns 集合中
                if (
                    isinstance(self.dtype, dict)
                    and col in self.dtype
                    and (
                        not is_numeric_dtype(self.dtype[col])
                        or is_bool_dtype(self.dtype[col])
                    )
                ):
                    no_thousands_columns.add(i)
        
        # 返回不处理千分位格式的列索引集合
        return no_thousands_columns
class FixedWidthReader(abc.Iterator):
    """
    A reader of fixed-width lines.
    """

    def __init__(
        self,
        f: IO[str] | ReadCsvBuffer[str],
        colspecs: list[tuple[int, int]] | Literal["infer"],
        delimiter: str | None,
        comment: str | None,
        skiprows: set[int] | None = None,
        infer_nrows: int = 100,
    ) -> None:
        # 初始化方法，设置对象的初始属性
        self.f = f  # 文件对象或 CSV 缓冲对象
        self.buffer: Iterator | None = None  # 缓冲区初始化为空
        self.delimiter = "\r\n" + delimiter if delimiter else "\n\r\t "  # 行分隔符
        self.comment = comment  # 注释字符
        # 如果 colspecs 是 "infer"，则调用 detect_colspecs 推断列宽
        if colspecs == "infer":
            self.colspecs = self.detect_colspecs(
                infer_nrows=infer_nrows, skiprows=skiprows
            )
        else:
            self.colspecs = colspecs  # 否则使用给定的列宽列表

        # 检查列宽的类型是否正确
        if not isinstance(self.colspecs, (tuple, list)):
            raise TypeError(
                "column specifications must be a list or tuple, "
                f"input was a {type(colspecs).__name__}"
            )

        for colspec in self.colspecs:
            # 检查每个列宽规范是否为有效的元组或列表
            if not (
                isinstance(colspec, (tuple, list))
                and len(colspec) == 2
                and isinstance(colspec[0], (int, np.integer, type(None)))
                and isinstance(colspec[1], (int, np.integer, type(None)))
            ):
                raise TypeError(
                    "Each column specification must be "
                    "2 element tuple or list of integers"
                )

    def get_rows(self, infer_nrows: int, skiprows: set[int] | None = None) -> list[str]:
        """
        Read rows from self.f, skipping as specified.

        We distinguish buffer_rows (the first <= infer_nrows
        lines) from the rows returned to detect_colspecs
        because it's simpler to leave the other locations
        with skiprows logic alone than to modify them to
        deal with the fact we skipped some rows here as
        well.

        Parameters
        ----------
        infer_nrows : int
            Number of rows to read from self.f, not counting
            rows that are skipped.
        skiprows: set, optional
            Indices of rows to skip.

        Returns
        -------
        detect_rows : list of str
            A list containing the rows to read.

        """
        if skiprows is None:
            skiprows = set()  # 如果 skiprows 为 None，则初始化为空集合
        buffer_rows = []  # 初始化缓冲行列表
        detect_rows = []  # 初始化检测行列表
        for i, row in enumerate(self.f):
            if i not in skiprows:
                detect_rows.append(row)  # 如果行索引不在 skiprows 中，则添加到 detect_rows
            buffer_rows.append(row)  # 将所有行都添加到 buffer_rows
            if len(detect_rows) >= infer_nrows:
                break
        self.buffer = iter(buffer_rows)  # 将 buffer_rows 转换为迭代器赋给 self.buffer
        return detect_rows  # 返回 detect_rows 列表

    def detect_colspecs(
        self, infer_nrows: int = 100, skiprows: set[int] | None = None
    ) -> list[tuple[int, int]]:
        """
        Detect column specifications from the first `infer_nrows`
        lines of `self.f`, skipping rows as specified.

        Parameters
        ----------
        infer_nrows : int, optional
            Number of rows to read from self.f for inference.
        skiprows : set of int, optional
            Indices of rows to skip.

        Returns
        -------
        colspecs : list of tuple of int
            List of detected column specifications (start, end).

        """
        # 如果 skiprows 为 None，则初始化为空集合
        if skiprows is None:
            skiprows = set()
        colspecs = []  # 初始化列宽度列表
        for row in self.f:
            if len(colspecs) >= infer_nrows:
                break
            if len(skiprows) > 0 and i in skiprows:
                continue
            # 探测每行的列宽度，添加到 colspecs 中
            start = 0
            for width in self.colspecs:
                end = start + width
                colspecs.append((start, end))
                start = end
        return colspecs  # 返回探测到的列宽度列表
    ) -> list[tuple[int, int]]:
        # 定义正则表达式的分隔符转义，用于匹配行内容
        delimiters = "".join([rf"\{x}" for x in self.delimiter])
        # 创建正则表达式模式，匹配非分隔符字符的连续序列
        pattern = re.compile(f"([^{delimiters}]+)")
        # 调用实例方法获取数据行，推断行数，并跳过指定行数
        rows = self.get_rows(infer_nrows, skiprows)
        # 如果未获取到任何行数据，则抛出空数据异常
        if not rows:
            raise EmptyDataError("No rows from which to infer column width")
        # 计算所有行中最长行的长度
        max_len = max(map(len, rows))
        # 创建一个长度为最长行加一的整数数组，初始值为零
        mask = np.zeros(max_len + 1, dtype=int)
        # 如果存在注释符号，则在每行中删除注释符号及其后的内容
        if self.comment is not None:
            rows = [row.partition(self.comment)[0] for row in rows]
        # 对每一行应用正则表达式，标记出匹配的字符范围
        for row in rows:
            for m in pattern.finditer(row):
                mask[m.start() : m.end()] = 1
        # 将标记数组向右移动一位，第一个元素设为零
        shifted = np.roll(mask, 1)
        shifted[0] = 0
        # 找出边界变化的位置，构建成成对的边界
        edges = np.where((mask ^ shifted) == 1)[0]
        edge_pairs = list(zip(edges[::2], edges[1::2]))
        # 返回成对的边界列表
        return edge_pairs

    def __next__(self) -> list[str]:
        # 如果缓冲区不为空，尝试从缓冲区获取下一行数据
        if self.buffer is not None:
            try:
                line = next(self.buffer)
            except StopIteration:
                # 缓冲区数据已耗尽，将缓冲区置空，从文件中获取下一行数据
                self.buffer = None
                line = next(self.f)  # type: ignore[arg-type]
        else:
            # 从文件中获取下一行数据
            line = next(self.f)  # type: ignore[arg-type]
        # 返回处理后的行数据，根据列范围切片并去除分隔符空格
        return [line[from_:to].strip(self.delimiter) for (from_, to) in self.colspecs]
class FixedWidthFieldParser(PythonParser):
    """
    Specialization that Converts fixed-width fields into DataFrames.
    See PythonParser for details.
    """

    def __init__(self, f: ReadCsvBuffer[str], **kwds) -> None:
        """
        Initializes the FixedWidthFieldParser object.

        Parameters
        ----------
        f : ReadCsvBuffer[str]
            The input buffer containing fixed-width fields.
        **kwds : keyword arguments
            Additional parameters including 'colspecs' and 'infer_nrows'.
        """
        # Support iterators, convert to a list.
        self.colspecs = kwds.pop("colspecs")
        self.infer_nrows = kwds.pop("infer_nrows")
        PythonParser.__init__(self, f, **kwds)

    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]) -> FixedWidthReader:
        """
        Creates a FixedWidthReader object.

        Parameters
        ----------
        f : IO[str] | ReadCsvBuffer[str]
            Input stream or buffer containing fixed-width fields.

        Returns
        -------
        FixedWidthReader
            An instance of FixedWidthReader initialized with parameters.
        """
        return FixedWidthReader(
            f,
            self.colspecs,
            self.delimiter,
            self.comment,
            self.skiprows,
            self.infer_nrows,
        )

    def _remove_empty_lines(self, lines: list[list[T]]) -> list[list[T]]:
        """
        Removes empty lines from the list of lines.

        Parameters
        ----------
        lines : list[list[T]]
            List of lines where each line is a list of elements.

        Returns
        -------
        list[list[T]]
            List of non-empty lines.

        Notes
        -----
        With fixed-width fields, empty lines become arrays of empty strings.
        """
        return [
            line
            for line in lines
            if any(not isinstance(e, str) or e.strip() for e in line)
        ]


def count_empty_vals(vals) -> int:
    """
    Counts the number of empty or None values in a sequence.

    Parameters
    ----------
    vals : iterable
        Sequence of values to be checked.

    Returns
    -------
    int
        Number of empty or None values in 'vals'.
    """
    return sum(1 for v in vals if v == "" or v is None)


def _validate_skipfooter_arg(skipfooter: int) -> int:
    """
    Validate the 'skipfooter' parameter.

    Checks whether 'skipfooter' is a non-negative integer.
    Raises a ValueError if that is not the case.

    Parameters
    ----------
    skipfooter : int
        The number of rows to skip at the end of the file.

    Returns
    -------
    int
        The original input 'skipfooter' if it's valid.

    Raises
    ------
    ValueError
        If 'skipfooter' is not a non-negative integer.
    """
    if not is_integer(skipfooter):
        raise ValueError("skipfooter must be an integer")

    if skipfooter < 0:
        raise ValueError("skipfooter cannot be negative")

    # Incompatible return value type (got "Union[int, integer[Any]]", expected "int")
    return skipfooter  # type: ignore[return-value]
```