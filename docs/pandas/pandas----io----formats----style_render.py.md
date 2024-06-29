# `D:\src\scipysrc\pandas\pandas\io\formats\style_render.py`

```
from __future__ import annotations  # 引入未来版本的类型注解支持

from collections import defaultdict  # 引入 defaultdict 支持默认值的字典
from collections.abc import (  # 引入抽象基类中的 Callable 和 Sequence
    Callable,
    Sequence,
)
from functools import partial  # 引入 partial 函数用于创建偏函数
import re  # 引入正则表达式模块
from typing import (  # 引入类型提示中的各种类型
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Optional,
    TypedDict,
    Union,
)
from uuid import uuid4  # 引入 uuid4 用于生成 UUID

import numpy as np  # 引入 NumPy 数学库

from pandas._config import get_option  # 从 pandas._config 模块中导入 get_option 函数

from pandas._libs import lib  # 引入 pandas 库的 C 库支持
from pandas.compat._optional import import_optional_dependency  # 引入可选依赖项导入函数

from pandas.core.dtypes.common import (  # 引入 pandas 核心数据类型中的常见函数
    is_complex,
    is_float,
    is_integer,
)
from pandas.core.dtypes.generic import ABCSeries  # 引入 pandas 通用数据类型的抽象基类

from pandas import (  # 从 pandas 中导入 DataFrame、Index 等数据结构和函数
    DataFrame,
    Index,
    IndexSlice,
    MultiIndex,
    Series,
    isna,
)
from pandas.api.types import is_list_like  # 导入判断对象是否列表样式的函数
import pandas.core.common as com  # 导入 pandas 核心公共模块

if TYPE_CHECKING:
    from pandas._typing import (  # 如果在类型检查模式下，从 pandas._typing 中导入 Axis 和 Level 类型
        Axis,
        Level,
    )
jinja2 = import_optional_dependency("jinja2", extra="DataFrame.style requires jinja2.")  # 导入可选依赖项 jinja2
from markupsafe import escape as escape_html  # 导入 markupsafe 库中的 escape 函数（jinja2 的依赖）

BaseFormatter = Union[str, Callable]  # 定义 BaseFormatter 类型为字符串或可调用对象的联合类型
ExtFormatter = Union[BaseFormatter, dict[Any, Optional[BaseFormatter]]]  # 定义 ExtFormatter 类型为 BaseFormatter 或包含 BaseFormatter 的字典
CSSPair = tuple[str, Union[str, float]]  # 定义 CSSPair 类型为包含两个元素的元组，元素分别为字符串和字符串或浮点数的联合类型
CSSList = list[CSSPair]  # 定义 CSSList 类型为 CSSPair 元组的列表
CSSProperties = Union[str, CSSList]  # 定义 CSSProperties 类型为字符串或 CSSList 的联合类型

class CSSDict(TypedDict):  # 定义 CSSDict 类型为 TypedDict
    selector: str  # CSS 选择器的字符串类型键
    props: CSSProperties  # CSS 属性的 CSSProperties 类型值

CSSStyles = list[CSSDict]  # 定义 CSSStyles 类型为 CSSDict 字典的列表
Subset = Union[slice, Sequence, Index]  # 定义 Subset 类型为 slice、Sequence 或 Index 的联合类型

class StylerRenderer:  # 定义 StylerRenderer 类
    """
    Base class to process rendering a Styler with a specified jinja2 template.
    """

    loader = jinja2.PackageLoader("pandas", "io/formats/templates")  # 设置 jinja2 模板的加载器
    env = jinja2.Environment(loader=loader, trim_blocks=True)  # 创建 jinja2 环境对象，设置模板块间的空白处理
    template_html = env.get_template("html.tpl")  # 获取 HTML 模板
    template_html_table = env.get_template("html_table.tpl")  # 获取 HTML 表格模板
    template_html_style = env.get_template("html_style.tpl")  # 获取 HTML 样式模板
    template_latex = env.get_template("latex.tpl")  # 获取 LaTeX 模板
    template_string = env.get_template("string.tpl")  # 获取字符串模板

    def __init__(  # 定义初始化方法
        self,
        data: DataFrame | Series,  # 参数 data 可以是 DataFrame 或 Series
        uuid: str | None = None,  # 参数 uuid 可以是字符串或 None
        uuid_len: int = 5,  # 参数 uuid_len 默认为 5 的整数
        table_styles: CSSStyles | None = None,  # 参数 table_styles 可以是 CSSStyles 类型或 None
        table_attributes: str | None = None,  # 参数 table_attributes 可以是字符串或 None
        caption: str | tuple | list | None = None,  # 参数 caption 可以是字符串、元组、列表或 None
        cell_ids: bool = True,  # 参数 cell_ids 默认为 True 的布尔类型
        precision: int | None = None,  # 参数 precision 可以是整数或 None
    def _render(  # 定义 _render 方法
        self,
        sparse_index: bool,  # 参数 sparse_index 是布尔类型
        sparse_columns: bool,  # 参数 sparse_columns 是布尔类型
        max_rows: int | None = None,  # 参数 max_rows 可以是整数或 None
        max_cols: int | None = None,  # 参数 max_cols 可以是整数或 None
        blank: str = "",  # 参数 blank 默认为空字符串
    ):
        """
        Computes and applies styles and then generates the general render dicts.

        Also extends the `ctx` and `ctx_index` attributes with those of concatenated
        stylers for use within `_translate_latex`
        """
        # 调用 _compute 方法，计算并应用样式
        self._compute()
        # 初始化一个空列表 dxs 用于存储渲染后的结果
        dxs = []
        # 获取当前对象的索引长度
        ctx_len = len(self.index)
        # 遍历 self.concatenated 中的每个对象
        for i, concatenated in enumerate(self.concatenated):
            # 将当前对象的一些属性传递给 concatenated 对象
            concatenated.hide_index_ = self.hide_index_
            concatenated.hidden_columns = self.hidden_columns
            # 构建 foot 变量，用于生成 CSS 样式的后缀
            foot = f"{self.css['foot']}{i}"
            # 设置 concatenated 对象的 CSS 样式
            concatenated.css = {
                **self.css,
                "data": f"{foot}_data",
                "row_heading": f"{foot}_row_heading",
                "row": f"{foot}_row",
                "foot": f"{foot}_foot",
            }
            # 调用 concatenated 对象的 _render 方法进行渲染
            dx = concatenated._render(
                sparse_index, sparse_columns, max_rows, max_cols, blank
            )
            # 将渲染后的结果添加到 dxs 列表中
            dxs.append(dx)

            # 将 concatenated 对象的 ctx 字典中的项添加到 self.ctx 中
            for (r, c), v in concatenated.ctx.items():
                self.ctx[(r + ctx_len, c)] = v
            # 将 concatenated 对象的 ctx_index 字典中的项添加到 self.ctx_index 中
            for (r, c), v in concatenated.ctx_index.items():
                self.ctx_index[(r + ctx_len, c)] = v

            # 更新 ctx_len，累加当前 concatenated 对象的索引长度
            ctx_len += len(concatenated.index)

        # 调用 _translate 方法进行最终的转换
        d = self._translate(
            sparse_index, sparse_columns, max_rows, max_cols, blank, dxs
        )
        # 返回渲染后的结果字典 d
        return d

    def _render_html(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        **kwargs,
    ) -> str:
        """
        Renders the ``Styler`` including all applied styles to HTML.
        Generates a dict with necessary kwargs passed to jinja2 template.
        """
        # 调用 _render 方法进行渲染，生成渲染结果字典 d
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols, "&nbsp;")
        # 更新 d 字典，添加额外的 kwargs
        d.update(kwargs)
        # 使用 template_html 对象渲染 HTML，并返回结果字符串
        return self.template_html.render(
            **d,
            html_table_tpl=self.template_html_table,
            html_style_tpl=self.template_html_style,
        )

    def _render_latex(
        self, sparse_index: bool, sparse_columns: bool, clines: str | None, **kwargs
    ) -> str:
        """
        Render a Styler in latex format
        """
        # 调用 _render 方法进行渲染，生成渲染结果字典 d
        d = self._render(sparse_index, sparse_columns, None, None)
        # 调用 _translate_latex 方法将渲染结果转换为 LaTeX 格式
        self._translate_latex(d, clines=clines)
        # 设置 template_latex 对象的全局函数，用于处理 LaTeX 表格样式
        self.template_latex.globals["parse_wrap"] = _parse_latex_table_wrapping
        self.template_latex.globals["parse_table"] = _parse_latex_table_styles
        self.template_latex.globals["parse_cell"] = _parse_latex_cell_styles
        self.template_latex.globals["parse_header"] = _parse_latex_header_span
        # 更新 d 字典，添加额外的 kwargs
        d.update(kwargs)
        # 使用 template_latex 对象渲染 LaTeX，并返回结果字符串
        return self.template_latex.render(**d)

    def _render_string(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        **kwargs,
    ) -> str:
        """
        Render a Styler in string format
        """
        # 调用 _render 方法生成样式渲染的字典 d
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols)
        # 更新字典 d，添加任意额外的关键字参数
        d.update(kwargs)
        # 使用模板字符串渲染并返回最终样式字符串
        return self.template_string.render(**d)

    def _compute(self):
        """
        Execute the style functions built up in `self._todo`.

        Relies on the conventions that all style functions go through
        .apply or .map. The append styles to apply as tuples of

        (application method, *args, **kwargs)
        """
        # 清空上下文环境的数据
        self.ctx.clear()
        self.ctx_index.clear()
        self.ctx_columns.clear()
        r = self
        # 遍历 self._todo 中的函数、参数和关键字参数，并依次执行
        for func, args, kwargs in self._todo:
            r = func(self)(*args, **kwargs)
        # 返回执行结果 r
        return r

    def _translate(
        self,
        sparse_index: bool,
        sparse_cols: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        blank: str = "&nbsp;",
        dxs: list[dict] | None = None,
    def _translate_header(self, sparsify_cols: bool, max_cols: int):
        """
        Build each <tr> within table <head> as a list

        Using the structure:
             +----------------------------+---------------+---------------------------+
             |  index_blanks ...          | column_name_0 |  column_headers (level_0) |
          1) |       ..                   |       ..      |             ..            |
             |  index_blanks ...          | column_name_n |  column_headers (level_n) |
             +----------------------------+---------------+---------------------------+
          2) |  index_names (level_0 to level_n) ...      | column_blanks ...         |
             +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        sparsify_cols : bool
            Whether column_headers section will add colspan attributes (>1) to elements.
        max_cols : int
            Maximum number of columns to render. If exceeded will contain `...` filler.

        Returns
        -------
        head : list
            The associated HTML elements needed for template rendering.
        """
        # Calculate the lengths of columns based on MultiIndex structure
        col_lengths = _get_level_lengths(
            self.columns, sparsify_cols, max_cols, self.hidden_columns
        )

        # Convert column labels to list format
        clabels = self.data.columns.tolist()
        if self.data.columns.nlevels == 1:
            clabels = [[x] for x in clabels]
        clabels = list(zip(*clabels))

        head = []
        # 1) Generate column header rows
        for r, hide in enumerate(self.hide_columns_):
            if hide or not clabels:
                continue

            # Generate HTML elements for a column header row
            header_row = self._generate_col_header_row(
                (r, clabels), max_cols, col_lengths
            )
            head.append(header_row)

        # 2) Generate index names row if conditions are met
        if (
            self.data.index.names
            and com.any_not_none(*self.data.index.names)
            and not all(self.hide_index_)
            and not self.hide_index_names
        ):
            # Generate HTML elements for index names row
            index_names_row = self._generate_index_names_row(
                clabels, max_cols, col_lengths
            )
            head.append(index_names_row)

        return head

    def _generate_col_header_row(
        self, iter: Sequence, max_cols: int, col_lengths: dict
    ):
        """
        Generate HTML elements for a column header row.

        Parameters
        ----------
        iter : Sequence
            Iterator containing information about columns.
        max_cols : int
            Maximum number of columns to render.
        col_lengths : dict
            Dictionary containing lengths of columns based on MultiIndex structure.

        Returns
        -------
        list
            HTML elements representing the column header row.
        """
        # Implementation details for generating column header row
        pass

    def _generate_index_names_row(
        self, iter: Sequence, max_cols: int, col_lengths: dict
    ):
        """
        Generate HTML elements for an index names row.

        Parameters
        ----------
        iter : Sequence
            Iterator containing information about columns.
        max_cols : int
            Maximum number of columns to render.
        col_lengths : dict
            Dictionary containing lengths of columns based on MultiIndex structure.

        Returns
        -------
        list
            HTML elements representing the index names row.
        """
        # Implementation details for generating index names row
        pass
        """
        生成包含索引名称的行

         +----------------------------+---------------+---------------------------+
         |  index_names (level_0 to level_n) ...      | column_blanks ...         |
         +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            外部作用域中的循环变量
        max_cols : int
            允许的列数上限

        Returns
        -------
        list of elements
            返回一个元素列表
        """

        # 将传入的 iter 参数赋值给 clabels
        clabels = iter

        # 生成索引名称的列表
        index_names = [
            _element(
                "th",
                f"{self.css['index_name']} {self.css['level']}{c}",
                # 如果索引名称为 None，则使用空值样式
                self.css["blank_value"] if name is None else name,
                # 检查是否隐藏该索引列
                not self.hide_index_[c],
                # 如果存在索引名称，应用显示函数
                display_value=(
                    None if name is None else self._display_funcs_index_names[c](name)
                ),
            )
            # 遍历数据对象的索引名称列表，使用 enumerate 获取索引和名称
            for c, name in enumerate(self.data.index.names)
        ]

        # 初始化空的列名列表
        column_blanks: list = []
        # 可见列的计数器
        visible_col_count: int = 0
        # 如果 clabels 非空
        if clabels:
            # 获取最后一个级别的索引列数
            last_level = self.columns.nlevels - 1  # 使用最后一个级别，因为不会稀疏化
            # 遍历最后一个级别的列标签和值
            for c, value in enumerate(clabels[last_level]):
                # 检查表头元素是否可见
                header_element_visible = _is_visible(c, last_level, col_lengths)
                if header_element_visible:
                    visible_col_count += 1
                # 检查是否需要修剪列数，并进行相应处理
                if self._check_trim(
                    visible_col_count,
                    max_cols,
                    column_blanks,
                    "th",
                    f"{self.css['blank']} {self.css['col']}{c} {self.css['col_trim']}",
                    self.css["blank_value"],
                ):
                    break

                # 添加空白列
                column_blanks.append(
                    _element(
                        "th",
                        f"{self.css['blank']} {self.css['col']}{c}",
                        self.css["blank_value"],
                        # 检查列是否在隐藏列中
                        c not in self.hidden_columns,
                    )
                )

        # 返回索引名称列表和空白列列表的组合
        return index_names + column_blanks
    def _translate_body(self, idx_lengths: dict, max_rows: int, max_cols: int):
        """
        Build each <tr> within table <body> as a list

        Use the following structure:
          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column   ...     |
          +--------------------------------------------+---------------------------+

        Also add elements to the cellstyle_map for more efficient grouped elements in
        <style></style> block

        Parameters
        ----------
        idx_lengths : dict
            Dictionary containing the lengths of indices.
        max_rows : int
            Maximum number of rows to render.
        max_cols : int
            Maximum number of columns to render.

        Returns
        -------
        body : list
            The associated HTML elements needed for template rendering.
        """
        # Get the list of row labels from the data's index
        rlabels = self.data.index.tolist()
        
        # Convert row labels to a list of lists if the index is not a MultiIndex
        if not isinstance(self.data.index, MultiIndex):
            rlabels = [[x] for x in rlabels]

        # Initialize an empty list to store the body of the table
        body: list = []
        
        # Initialize a counter for visible rows
        visible_row_count: int = 0
        
        # Iterate through enumerated rows in the data, skipping hidden rows
        for r, row_tup in [
            z for z in enumerate(self.data.itertuples()) if z[0] not in self.hidden_rows
        ]:
            visible_row_count += 1
            
            # Check if trimming of rows is needed based on maximum rows allowed
            if self._check_trim(
                visible_row_count,
                max_rows,
                body,
                "row",
            ):
                break

            # Generate HTML for the current body row
            body_row = self._generate_body_row(
                (r, row_tup, rlabels), max_cols, idx_lengths
            )
            
            # Append the generated body row to the body list
            body.append(body_row)
        
        # Return the completed body list
        return body

    def _check_trim(
        self,
        count: int,
        max: int,
        obj: list,
        element: str,
        css: str | None = None,
        value: str = "...",
    ) -> bool:
        """
        Indicates whether to break render loops and append a trimming indicator

        Parameters
        ----------
        count : int
            The loop count of previous visible items.
        max : int
            The allowable rendered items in the loop.
        obj : list
            The current render collection of the rendered items.
        element : str
            The type of element to append in the case a trimming indicator is needed.
        css : str, optional
            The css to add to the trimming indicator element.
        value : str, optional
            The value of the elements display if necessary.

        Returns
        -------
        result : bool
            Whether a trimming element was required and appended.
        """
        # Check if the current count exceeds the maximum allowed
        if count > max:
            # Append a trimmed row element if the element type is "row"
            if element == "row":
                obj.append(self._generate_trimmed_row(max))
            else:
                # Append a generic trimmed element with specified CSS and value
                obj.append(_element(element, css, value, True, attributes=""))
            return True
        
        # Return False if no trimming was needed
        return False
    def _generate_trimmed_row(self, max_cols: int) -> list:
        """
        当渲染行数过多时，生成包含 "..." 的修剪行。

        Parameters
        ----------
        max_cols : int
            允许的最大列数

        Returns
        -------
        list of elements
            包含修剪行和数据的元素列表
        """
        # 生成索引标题的元素列表，每个元素都是一个 _element 对象
        index_headers = [
            _element(
                "th",
                (
                    f"{self.css['row_heading']} {self.css['level']}{c} "
                    f"{self.css['row_trim']}"
                ),
                "...",
                not self.hide_index_[c],  # 如果隐藏标志为 False，则显示该元素
                attributes="",  # 元素的额外属性为空字符串
            )
            for c in range(self.data.index.nlevels)  # 遍历数据的索引层级
        ]

        data: list = []  # 初始化数据列表为空
        visible_col_count: int = 0  # 可见列计数器初始化为0
        for c, _ in enumerate(self.columns):  # 遍历数据的列
            data_element_visible = c not in self.hidden_columns  # 检查数据元素是否可见
            if data_element_visible:
                visible_col_count += 1  # 如果数据元素可见，则增加可见列计数

            # 检查是否需要修剪行
            if self._check_trim(
                visible_col_count,
                max_cols,
                data,
                "td",
                f"{self.css['data']} {self.css['row_trim']} {self.css['col_trim']}",
            ):
                break  # 如果需要修剪，则终止循环

            # 构建数据元素，并添加到数据列表中
            data.append(
                _element(
                    "td",
                    f"{self.css['data']} {self.css['col']}{c} {self.css['row_trim']}",
                    "...",  # 使用 "..." 表示修剪的数据
                    data_element_visible,  # 数据元素是否可见
                    attributes="",  # 元素的额外属性为空字符串
                )
            )

        return index_headers + data  # 返回索引标题和数据元素组成的列表



    def _generate_body_row(
        self,
        iter: tuple,
        max_cols: int,
        idx_lengths: dict,
    ):
        """
        Generate a body row for the rendered table.

        Parameters
        ----------
        iter : tuple
            Tuple containing data elements for the row.
        max_cols : int
            Maximum number of columns allowed.
        idx_lengths : dict
            Dictionary containing lengths of indices.

        Returns
        -------
        None
        """



    def format(
        self,
        formatter: ExtFormatter | None = None,
        subset: Subset | None = None,
        na_rep: str | None = None,
        precision: int | None = None,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
        hyperlinks: str | None = None,
    ):
        """
        Format the DataFrame using specified formatting options.

        Parameters
        ----------
        formatter : ExtFormatter or None, optional
            Custom formatter object.
        subset : Subset or None, optional
            Subset of columns to format.
        na_rep : str or None, optional
            String representation of NA/null values.
        precision : int or None, optional
            Number of decimal places.
        decimal : str, optional
            Decimal separator.
        thousands : str or None, optional
            Thousands separator.
        escape : str or None, optional
            String to escape special characters.
        hyperlinks : str or None, optional
            URL pattern for hyperlinks.

        Returns
        -------
        None
        """



    def format_index(
        self,
        formatter: ExtFormatter | None = None,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        na_rep: str | None = None,
        precision: int | None = None,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
        hyperlinks: str | None = None,
    ):
        """
        Format the index of the DataFrame using specified options.

        Parameters
        ----------
        formatter : ExtFormatter or None, optional
            Custom formatter object.
        axis : Axis, optional
            The axis to format (0 for rows, 1 for columns).
        level : Level or list[Level] or None, optional
            Levels of the MultiIndex to format.
        na_rep : str or None, optional
            String representation of NA/null values.
        precision : int or None, optional
            Number of decimal places.
        decimal : str, optional
            Decimal separator.
        thousands : str or None, optional
            Thousands separator.
        escape : str or None, optional
            String to escape special characters.
        hyperlinks : str or None, optional
            URL pattern for hyperlinks.

        Returns
        -------
        None
        """



    def relabel_index(
        self,
        labels: Sequence | Index,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
    ):
        """
        Relabel the index of the DataFrame.

        Parameters
        ----------
        labels : Sequence or Index
            New labels for the index.
        axis : Axis, optional
            The axis to relabel (0 for rows, 1 for columns).
        level : Level or list[Level] or None, optional
            Levels of the MultiIndex to relabel.

        Returns
        -------
        None
        """



    def format_index_names(
        self,
        formatter: ExtFormatter | None = None,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        na_rep: str | None = None,
        precision: int | None = None,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
        hyperlinks: str | None = None,
    ):
        """
        Format the names of index levels in the DataFrame.

        Parameters
        ----------
        formatter : ExtFormatter or None, optional
            Custom formatter object.
        axis : Axis, optional
            The axis to format (0 for rows, 1 for columns).
        level : Level or list[Level] or None, optional
            Levels of the MultiIndex to format.
        na_rep : str or None, optional
            String representation of NA/null values.
        precision : int or None, optional
            Number of decimal places.
        decimal : str, optional
            Decimal separator.
        thousands : str or None, optional
            Thousands separator.
        escape : str or None, optional
            String to escape special characters.
        hyperlinks : str or None, optional
            URL pattern for hyperlinks.

        Returns
        -------
        None
        """
# 定义一个函数 _element，用于生成一个包含 <td></td> 或 <th></th> 元素信息的容器字典
def _element(
    html_element: str,
    html_class: str | None,
    value: Any,
    is_visible: bool,
    **kwargs,
) -> dict:
    """
    Template to return container with information for a <td></td> or <th></th> element.
    """
    # 如果 kwargs 中不存在 "display_value" 键或其对应值为 None，则将 value 赋给 "display_value"
    if "display_value" not in kwargs or kwargs["display_value"] is None:
        kwargs["display_value"] = value
    # 返回包含元素类型、值、类名、可见性和其他关键字参数的字典
    return {
        "type": html_element,
        "value": value,
        "class": html_class,
        "is_visible": is_visible,
        **kwargs,
    }


# 定义一个函数 _get_trimming_maximums，用于递归减少行和列的数量以满足最大元素限制
def _get_trimming_maximums(
    rn,
    cn,
    max_elements,
    max_rows=None,
    max_cols=None,
    scaling_factor: float = 0.8,
) -> tuple[int, int]:
    """
    Recursively reduce the number of rows and columns to satisfy max elements.

    Parameters
    ----------
    rn, cn : int
        The number of input rows / columns
    max_elements : int
        The number of allowable elements
    max_rows, max_cols : int, optional
        Directly specify an initial maximum rows or columns before compression.
    scaling_factor : float
        Factor at which to reduce the number of rows / columns to fit.

    Returns
    -------
    rn, cn : tuple
        New rn and cn values that satisfy the max_elements constraint
    """

    # 内部函数，根据缩放因子减少行和列的数量
    def scale_down(rn, cn):
        if cn >= rn:
            return rn, int(cn * scaling_factor)
        else:
            return int(rn * scaling_factor), cn

    # 如果指定了 max_rows，则限制 rn 不超过 max_rows
    if max_rows:
        rn = max_rows if rn > max_rows else rn
    # 如果指定了 max_cols，则限制 cn 不超过 max_cols
    if max_cols:
        cn = max_cols if cn > max_cols else cn

    # 当行数乘以列数大于 max_elements 时，循环调用 scale_down 函数减少行和列的数量
    while rn * cn > max_elements:
        rn, cn = scale_down(rn, cn)

    # 返回满足 max_elements 约束的新 rn 和 cn 值的元组
    return rn, cn


# 定义一个函数 _get_level_lengths，用于查找每个元素的级别长度
def _get_level_lengths(
    index: Index,
    sparsify: bool,
    max_index: int,
    hidden_elements: Sequence[int] | None = None,
):
    """
    Given an index, find the level length for each element.

    Parameters
    ----------
    index : Index
        Index or columns to determine lengths of each element
    sparsify : bool
        Whether to hide or show each distinct element in a MultiIndex
    max_index : int
        The maximum number of elements to analyse along the index due to trimming
    hidden_elements : sequence of int
        Index positions of elements hidden from display in the index affecting
        length

    Returns
    -------
    Dict :
        Result is a dictionary of (level, initial_position): span
    """
    # 如果 index 是 MultiIndex 类型，则格式化多级索引
    if isinstance(index, MultiIndex):
        levels = index._format_multi(sparsify=lib.no_default, include_names=False)
    else:
        levels = index._format_flat(include_name=False)

    # 如果 hidden_elements 为 None，则初始化为空列表
    if hidden_elements is None:
        hidden_elements = []

    lengths = {}
    # 如果 index 不是 MultiIndex 类型，则遍历 levels 中的元素，生成长度字典
    if not isinstance(index, MultiIndex):
        for i, value in enumerate(levels):
            if i not in hidden_elements:
                lengths[(0, i)] = 1
        return lengths
    for i, lvl in enumerate(levels):
        visible_row_count = 0  # 用于由于显示限制而中断循环
        for j, row in enumerate(lvl):
            if visible_row_count > max_index:
                break  # 如果可见行计数超过最大索引，则跳出循环
            if not sparsify:
                # 如果不稀疏化，则长度始终为1，因为没有聚合。
                if j not in hidden_elements:
                    lengths[(i, j)] = 1  # 设置长度为1
                    visible_row_count += 1  # 增加可见行计数
            elif (row is not lib.no_default) and (j not in hidden_elements):
                # 如果该元素没有被稀疏化，则必须是节的开始
                last_label = j
                lengths[(i, last_label)] = 1  # 设置长度为1
                visible_row_count += 1  # 增加可见行计数
            elif row is not lib.no_default:
                # 即使上述被隐藏了，也要跟踪它，以防长度大于1且后续元素可见
                last_label = j
                lengths[(i, last_label)] = 0  # 设置长度为0
            elif j not in hidden_elements:
                # 那么元素必须是稀疏化部分的一部分并且可见
                visible_row_count += 1  # 增加可见行计数
                if visible_row_count > max_index:
                    break  # 如果达到渲染限制，则不添加长度
                if lengths[(i, last_label)] == 0:
                    # 如果上一个迭代是节的开头但被隐藏了，则偏移
                    last_label = j
                    lengths[(i, last_label)] = 1  # 设置长度为1
                else:
                    # 否则添加到上一个迭代
                    lengths[(i, last_label)] += 1

    non_zero_lengths = {
        element: length for element, length in lengths.items() if length >= 1
    }
    
    return non_zero_lengths  # 返回长度大于等于1的元素及其长度的字典
def _is_visible(idx_row, idx_col, lengths) -> bool:
    """
    Determine if a specific cell (idx_row, idx_col) is visible based on lengths.

    Parameters
    ----------
    idx_row : int
        Row index of the cell.
    idx_col : int
        Column index of the cell.
    lengths : dict
        Dictionary containing visibility information.

    Returns
    -------
    bool
        True if the cell is visible, False otherwise.
    """
    return (idx_col, idx_row) in lengths


def format_table_styles(styles: CSSStyles) -> CSSStyles:
    """
    Formats CSS styles to ensure each selector applies to only one element.

    Parameters
    ----------
    styles : list
        List of dictionaries specifying CSS styles.

    Returns
    -------
    list
        List of dictionaries with separated CSS selectors and corresponding properties.
    """
    return [
        {"selector": selector, "props": css_dict["props"]}
        for css_dict in styles
        for selector in css_dict["selector"].split(",")
    ]


def _default_formatter(x: Any, precision: int, thousands: bool = False) -> Any:
    """
    Format the display of a value based on its type and formatting options.

    Parameters
    ----------
    x : Any
        Input variable to be formatted.
    precision : int
        Floating point precision used if `x` is float or complex.
    thousands : bool, default False
        Whether to group digits with thousands separated by ",".

    Returns
    -------
    Any
        Formatted value matching input type, or string if input is float or complex with grouping.
    """
    if is_float(x) or is_complex(x):
        return f"{x:,.{precision}f}" if thousands else f"{x:.{precision}f}"
    elif is_integer(x):
        return f"{x:,}" if thousands else str(x)
    return x


def _wrap_decimal_thousands(
    formatter: Callable, decimal: str, thousands: str | None
) -> Callable:
    """
    Wraps a formatting function to handle decimal and thousands separators.

    Parameters
    ----------
    formatter : Callable
        Formatting function for float, complex, or integer values.
    decimal : str
        Decimal separator string.
    thousands : str or None
        Thousands separator string.

    Returns
    -------
    Callable
        Wrapper function that applies the specified formatting with separators.
    """

    def wrapper(x):
        if is_float(x) or is_integer(x) or is_complex(x):
            if decimal != "." and thousands is not None and thousands != ",":
                return (
                    formatter(x)
                    .replace(",", "§_§-")  # temporary string to avoid clash
                    .replace(".", decimal)
                    .replace("§_§-", thousands)
                )
            elif decimal != "." and (thousands is None or thousands == ","):
                return formatter(x).replace(".", decimal)
            elif decimal == "." and thousands is not None and thousands != ",":
                return formatter(x).replace(",", thousands)
        return formatter(x)

    return wrapper


def _str_escape(x, escape):
    """
    Escape a string based on specified method.

    Parameters
    ----------
    x : Any
        Input variable, typically a string.
    escape : str
        Escape method ('html', 'latex', 'latex-math').

    Returns
    -------
    Any
        Escaped string if `x` is a string and escape method is valid, else returns input.
    
    Raises
    ------
    ValueError
        If escape method is not one of {'html', 'latex', 'latex-math'}.
    """
    if isinstance(x, str):
        if escape == "html":
            return escape_html(x)
        elif escape == "latex":
            return _escape_latex(x)
        elif escape == "latex-math":
            return _escape_latex_math(x)
        else:
            raise ValueError(
                f"`escape` only permitted in {{'html', 'latex', 'latex-math'}}, got {escape}"
            )
    return x


def _render_href(x, format):
    """
    Detects URLs in text and converts them into HTML hyperlinks using specified format.

    Parameters
    ----------
    x : Any
        Input variable, typically a string.
    format : str
        Format to convert detected URLs into hyperlinks.

    Returns
    -------
    None
        This function directly modifies `x` to convert URLs to hyperlinks in place.
    """
    # 检查变量 x 是否是字符串类型
    if isinstance(x, str):
        # 根据格式选择超链接的模板字符串
        if format == "html":
            href = '<a href="{0}" target="_blank">{0}</a>'
        elif format == "latex":
            href = r"\href{{{0}}}{{{0}}}"
        else:
            # 如果格式不是 'html' 或 'latex'，则抛出数值错误异常
            raise ValueError("``hyperlinks`` format can only be 'html' or 'latex'")
        
        # 匹配 URL 或者网址的正则表达式模式
        pat = r"((http|ftp)s?:\/\/|www.)[\w/\-?=%.:@]+\.[\w/\-&?=%.,':;~!@#$*()\[\]]+"
        
        # 使用正则表达式替换 x 中匹配的 URL 或网址为超链接格式
        return re.sub(pat, lambda m: href.format(m.group(0)), x)
    
    # 如果 x 不是字符串类型，则直接返回 x
    return x
def _maybe_wrap_formatter(
    formatter: BaseFormatter | None = None,
    na_rep: str | None = None,
    precision: int | None = None,
    decimal: str = ".",
    thousands: str | None = None,
    escape: str | None = None,
    hyperlinks: str | None = None,
) -> Callable:
    """
    Allows formatters to be expressed as str, callable or None, where None returns
    a default formatting function. wraps with na_rep, and precision where they are
    available.
    """
    # 根据输入的 formatter 类型，确定初始函数 func_0
    if isinstance(formatter, str):
        func_0 = lambda x: formatter.format(x)  # 如果 formatter 是 str，则使用格式化字符串函数
    elif callable(formatter):
        func_0 = formatter  # 如果 formatter 是 callable，则直接使用其作为函数 func_0
    elif formatter is None:
        precision = (
            get_option("styler.format.precision") if precision is None else precision
        )
        # 如果 formatter 是 None，则使用默认工厂函数 _default_formatter 创建 func_0
        func_0 = partial(
            _default_formatter, precision=precision, thousands=(thousands is not None)
        )
    else:
        raise TypeError(f"'formatter' expected str or callable, got {type(formatter)}")

    # 如果 escape 参数不为 None，则用 _str_escape 函数替换 func_0
    if escape is not None:
        func_1 = lambda x: func_0(_str_escape(x, escape=escape))
    else:
        func_1 = func_0

    # 如果 decimal 不是默认值 '.'，或者 thousands 不是 None 且不是默认值 ','，
    # 则用 _wrap_decimal_thousands 函数替换 func_1
    if decimal != "." or (thousands is not None and thousands != ","):
        func_2 = _wrap_decimal_thousands(func_1, decimal=decimal, thousands=thousands)
    else:
        func_2 = func_1

    # 如果 hyperlinks 参数不为 None，则用 _render_href 函数替换 func_2
    if hyperlinks is not None:
        func_3 = lambda x: func_2(_render_href(x, format=hyperlinks))
    else:
        func_3 = func_2

    # 如果 na_rep 参数为 None，则直接返回 func_3；否则根据 isna(x) 的值决定返回 na_rep 或 func_3
    if na_rep is None:
        return func_3
    else:
        return lambda x: na_rep if (isna(x) is True) else func_3(x)


def non_reducing_slice(slice_: Subset):
    """
    Ensure that a slice doesn't reduce to a Series or Scalar.

    Any user-passed `subset` should have this called on it
    to make sure we're always working with DataFrames.
    """
    # 默认使用列切片，类似 DataFrame
    # ['A', 'B'] -> IndexSlices[:, ['A', 'B']]
    kinds = (ABCSeries, np.ndarray, Index, list, str)
    if isinstance(slice_, kinds):
        slice_ = IndexSlice[:, slice_]

    def pred(part) -> bool:
        """
        Returns
        -------
        bool
            True if slice does *not* reduce,
            False if `part` is a tuple.
        """
        # 当 part 是 tuple 时，如果任何一个元素是 slice 或者 list-like，则返回 True，表示切片不会减少
        if isinstance(part, tuple):
            # GH#39421 检查是否为子切片：
            return any((isinstance(s, slice) or is_list_like(s)) for s in part)
        else:
            # 当 part 是 slice 或者 list-like 时，返回 True，表示切片不会减少
            return isinstance(part, slice) or is_list_like(part)
    # 如果 `slice_` 不是类似列表的对象（即不可迭代的对象），进入条件判断
    if not is_list_like(slice_):
        # 如果 `slice_` 不是切片对象，则将其包装成一个二维列表，例如 df.loc[1]
        slice_ = [[slice_]]
    else:
        # 否则，如果 `slice_` 是类似列表的对象（可迭代的对象），进入这个分支
        # 使用列表推导式，对 `slice_` 中的每个元素进行判断和处理
        slice_ = [p if pred(p) else [p] for p in slice_]  # type: ignore[union-attr]
    # 将处理后的 `slice_` 转换成元组并返回
    return tuple(slice_)
# 如果输入的 style 是字符串，则尝试将其转换为元组格式的列表，每个元组表示一个 CSS 属性和值
def maybe_convert_css_to_tuples(style: CSSProperties) -> CSSList:
    if isinstance(style, str):
        # 根据分号分割字符串为属性-值对，并移除空白项
        s = style.split(";")
        try:
            # 使用冒号分割每个属性-值对，并去除空白字符，构建元组列表
            return [
                (x.split(":")[0].strip(), x.split(":")[1].strip())
                for x in s
                if x.strip() != ""
            ]
        except IndexError as err:
            # 如果格式不正确，抛出 ValueError 异常
            raise ValueError(
                "Styles supplied as string must follow CSS rule formats, "
                f"for example 'attr: val;'. '{style}' was given."
            ) from err
    # 如果不是字符串，则直接返回原始 style 对象
    return style


# 将给定的 level 参数重新格式化为一个整数列表，用于在 hide_index 或 hide_columns 中使用
def refactor_levels(
    level: Level | list[Level] | None,
    obj: Index,
) -> list[int]:
    if level is None:
        # 如果 level 为 None，则生成一个包含所有层级的整数列表
        levels_: list[int] = list(range(obj.nlevels))
    elif isinstance(level, int):
        # 如果 level 是整数，则将其包装为列表
        levels_ = [level]
    elif isinstance(level, str):
        # 如果 level 是字符串，则获取其对应的层级编号
        levels_ = [obj._get_level_number(level)]
    elif isinstance(level, list):
        # 如果 level 是列表，则逐个处理其元素，将字符串转换为层级编号
        levels_ = [
            obj._get_level_number(lev) if not isinstance(lev, int) else lev
            for lev in level
        ]
    else:
        # 如果 level 类型不正确，则抛出 ValueError 异常
        raise ValueError("`level` must be of type `int`, `str` or list of such")
    return levels_


class Tooltips:
    """
    An extension to ``Styler`` that allows for and manipulates tooltips on hover
    of ``<td>`` cells in the HTML result.

    Parameters
    ----------
    css_name: str, default "pd-t"
        Name of the CSS class that controls visualisation of tooltips.
    css_props: list-like, default; see Notes
        List of (attr, value) tuples defining properties of the CSS class.
    tooltips: DataFrame, default empty
        DataFrame of strings aligned with underlying Styler data for tooltip
        display.
    as_title_attribute: bool, default False
        Flag to use title attribute based tooltips (True) or <span> based
        tooltips (False).
        Add the tooltip text as title attribute to resultant <td> element. If
        True, no CSS is generated and styling effects do not apply.

    Notes
    -----
    The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

    Hidden visibility is a key prerequisite to the hover functionality, and should
    always be included in any manual properties specification.
    """
    def __init__(
        self,
        css_props: CSSProperties = [  # noqa: B006
            ("visibility", "hidden"),  # 默认CSS属性列表，设置元素隐藏
            ("position", "absolute"),  # 设置元素绝对定位
            ("z-index", 1),  # 设置元素层级
            ("background-color", "black"),  # 设置背景颜色为黑色
            ("color", "white"),  # 设置文本颜色为白色
            ("transform", "translate(-20px, -20px)"),  # 设置元素平移
        ],
        css_name: str = "pd-t",  # CSS类名，默认为"pd-t"
        tooltips: DataFrame = DataFrame(),  # 初始化空的DataFrame用于存储工具提示信息
        as_title_attribute: bool = False,  # 是否将工具提示信息作为标题属性
    ) -> None:
        self.class_name = css_name  # 将CSS类名存储在实例变量中
        self.class_properties = css_props  # 将CSS属性列表存储在实例变量中
        self.tt_data = tooltips  # 将工具提示信息DataFrame存储在实例变量中
        self.table_styles: CSSStyles = []  # 初始化表格样式为空列表
        self.as_title_attribute = as_title_attribute  # 将是否作为标题属性的标志存储在实例变量中

    @property
    def _class_styles(self):
        """
        Combine the ``_Tooltips`` CSS class name and CSS properties to the format
        required to extend the underlying ``Styler`` `table_styles` to allow
        tooltips to render in HTML.

        Returns
        -------
        styles : List
        """
        return [
            {
                "selector": f".{self.class_name}",  # 创建选择器，选择CSS类名对应的元素
                "props": maybe_convert_css_to_tuples(self.class_properties),  # 调用函数将CSS属性转换为元组形式
            }
        ]

    def _pseudo_css(
        self, uuid: str, name: str, row: int, col: int, text: str
    ) -> list[CSSDict]:
        """
        For every table data-cell that has a valid tooltip (not None, NaN or
        empty string) must create two pseudo CSS entries for the specific
        <td> element id which are added to overall table styles:
        an on hover visibility change and a content change
        dependent upon the user's chosen display string.

        For example:
            [{"selector": "T__row1_col1:hover .pd-t",
             "props": [("visibility", "visible")]},
            {"selector": "T__row1_col1 .pd-t::after",
             "props": [("content", "Some Valid Text String")]}]

        Parameters
        ----------
        uuid: str
            The uuid of the Styler instance
        name: str
            The css-name of the class used for styling tooltips
        row : int
            The row index of the specified tooltip string data
        col : int
            The col index of the specified tooltip string data
        text : str
            The textual content of the tooltip to be displayed in HTML.

        Returns
        -------
        pseudo_css : List
        """
        selector_id = "#T_" + uuid + "_row" + str(row) + "_col" + str(col)  # 构建特定元素ID选择器
        return [
            {
                "selector": selector_id + f":hover .{name}",  # 悬停时使CSS类名的元素可见
                "props": [("visibility", "visible")],
            },
            {
                "selector": selector_id + f" .{name}::after",  # 在特定元素后面添加文本内容
                "props": [("content", f'"{text}"')],
            },
        ]
    def _translate(self, styler: StylerRenderer, d: dict):
        """
        Mutate the render dictionary to allow for tooltips:

        - Add ``<span>`` HTML element to each data cells ``display_value``. Ignores
          headers.
        - Add table level CSS styles to control pseudo classes.

        Parameters
        ----------
        styler_data : DataFrame
            Underlying ``Styler`` DataFrame used for reindexing.
        uuid : str
            The underlying ``Styler`` uuid for CSS id.
        d : dict
            The dictionary prior to final render

        Returns
        -------
        render_dict : Dict
        """
        # Reindex the self.tt_data DataFrame to match the structure of styler.data
        self.tt_data = self.tt_data.reindex_like(styler.data)
        
        # If self.tt_data is empty, return the original dictionary d
        if self.tt_data.empty:
            return d

        # Create a mask to identify cells with NaN values or empty strings
        mask = (self.tt_data.isna()) | (self.tt_data.eq(""))  # empty string = no tooltip
        
        # Conditional block for adding tooltips via pseudo CSS and <span> elements
        if not self.as_title_attribute:
            # Initialize table_styles list to store CSS styles
            name = self.class_name
            self.table_styles = [
                style
                for sublist in [
                    # Generate pseudo CSS for each cell with non-empty tooltips
                    self._pseudo_css(
                        styler.uuid, name, i, j, str(self.tt_data.iloc[i, j])
                    )
                    for i in range(len(self.tt_data.index))
                    for j in range(len(self.tt_data.columns))
                    if not (
                        mask.iloc[i, j]
                        or i in styler.hidden_rows
                        or j in styler.hidden_columns
                    )
                ]
                for style in sublist
            ]

            # Add span class to every cell with non-empty tooltip
            if self.table_styles:
                for row in d["body"]:
                    for item in row:
                        if item["type"] == "td":
                            item["display_value"] = (
                                str(item["display_value"])
                                + f'<span class="{self.class_name}"></span>'
                            )
                d["table_styles"].extend(self._class_styles)
                d["table_styles"].extend(self.table_styles)
        
        # Conditional block for adding tooltips as "title" attribute on <td> elements
        else:
            index_offset = self.tt_data.index.nlevels
            body = d["body"]
            for i in range(len(self.tt_data.index)):
                for j in range(len(self.tt_data.columns)):
                    if (
                        not mask.iloc[i, j]
                        or i in styler.hidden_rows
                        or j in styler.hidden_columns
                    ):
                        row = body[i]
                        item = row[j + index_offset]
                        value = self.tt_data.iloc[i, j]
                        item["attributes"] += f' title="{value}"'
        
        # Return the mutated render dictionary
        return d
# 判断是否需要将 LaTeX {tabular} 包装在 {table} 环境中
def _parse_latex_table_wrapping(table_styles: CSSStyles, caption: str | None) -> bool:
    """
    Indicate whether LaTeX {tabular} should be wrapped with a {table} environment.

    Parses the `table_styles` and detects any selectors which must be included outside
    of {tabular}, i.e. indicating that wrapping must occur, and therefore return True,
    or if a caption exists and requires similar.
    """
    # 忽略的选择器将与 {tabular} 一起包含，因此不需要包装
    IGNORED_WRAPPERS = ["toprule", "midrule", "bottomrule", "column_format"]
    return (
        table_styles is not None
        and any(d["selector"] not in IGNORED_WRAPPERS for d in table_styles)
    ) or caption is not None


def _parse_latex_table_styles(table_styles: CSSStyles, selector: str) -> str | None:
    """
    Return the first 'props' 'value' from ``tables_styles`` identified by ``selector``.

    Examples
    --------
    >>> table_styles = [
    ...     {"selector": "foo", "props": [("attr", "value")]},
    ...     {"selector": "bar", "props": [("attr", "overwritten")]},
    ...     {"selector": "bar", "props": [("a1", "baz"), ("a2", "ignore")]},
    ... ]
    >>> _parse_latex_table_styles(table_styles, selector="bar")
    'baz'

    Notes
    -----
    The replacement of "§" with ":" is to avoid the CSS problem where ":" has structural
    significance and cannot be used in LaTeX labels, but is often required by them.
    """
    for style in table_styles[::-1]:  # 从最近应用的样式开始遍历
        if style["selector"] == selector:
            return str(style["props"][0][1]).replace("§", ":")
    return None


def _parse_latex_cell_styles(
    latex_styles: CSSList, display_value: str, convert_css: bool = False
) -> str:
    r"""
    Mutate the ``display_value`` string including LaTeX commands from ``latex_styles``.

    This method builds a recursive latex chain of commands based on the
    CSSList input, nested around ``display_value``.

    If a CSS style is given as ('<command>', '<options>') this is translated to
    '\<command><options>{display_value}', and this value is treated as the
    display value for the next iteration.

    The most recent style forms the inner component, for example for styles:
    `[('c1', 'o1'), ('c2', 'o2')]` this returns: `\c1o1{\c2o2{display_value}}`

    Sometimes latex commands have to be wrapped with curly braces in different ways:
    We create some parsing flags to identify the different behaviours:

     - `--rwrap`        : `\<command><options>{<display_value>}`
     - `--wrap`         : `{\<command><options> <display_value>}`
     - `--nowrap`       : `\<command><options> <display_value>`
     - `--lwrap`        : `{\<command><options>} <display_value>`
     - `--dwrap`        : `{\<command><options>}{<display_value>}`

    For example for styles:
    `[('c1', 'o1--wrap'), ('c2', 'o2')]` this returns: `{\c1o1 \c2o2{display_value}}
    """
    # 如果 convert_css 为真，则调用 _parse_latex_css_conversion 函数处理 latex_styles
    if convert_css:
        latex_styles = _parse_latex_css_conversion(latex_styles)
    # 遍历 latex_styles 列表，倒序处理最近的样式
    for command, options in latex_styles[::-1]:  # in reverse for most recent style
        # 定义不同格式化选项的字典
        formatter = {
            "--wrap": f"{{\\{command}--to_parse {display_value}}}",
            "--nowrap": f"\\{command}--to_parse {display_value}",
            "--lwrap": f"{{\\{command}--to_parse}} {display_value}",
            "--rwrap": f"\\{command}--to_parse{{{display_value}}}",
            "--dwrap": f"{{\\{command}--to_parse}}{{{display_value}}}",
        }
        # 将显示值更新为添加命令和选项后的字符串
        display_value = f"\\{command}{options} {display_value}"
        # 遍历格式化选项列表，根据选项内容调整显示值
        for arg in ["--nowrap", "--wrap", "--lwrap", "--rwrap", "--dwrap"]:
            # 如果选项字符串中包含当前的格式化选项
            if arg in str(options):
                # 使用对应的格式化字符串替换显示值，调用 _parse_latex_options_strip 处理选项值
                display_value = formatter[arg].replace(
                    "--to_parse", _parse_latex_options_strip(value=options, arg=arg)
                )
                # 只处理第一个匹配到的选项，因此结束循环
                break  # only ever one purposeful entry
    # 返回最终处理后的显示值
    return display_value
def _parse_latex_header_span(
    cell: dict[str, Any],
    multirow_align: str,
    multicol_align: str,
    wrap: bool = False,
    convert_css: bool = False,
) -> str:
    r"""
    Refactor the cell `display_value` if a 'colspan' or 'rowspan' attribute is present.

    'rowspan' and 'colspan' do not occur simultaneouly. If they are detected then
    the `display_value` is altered to a LaTeX `multirow` or `multicol` command
    respectively, with the appropriate cell-span.

    ``wrap`` is used to enclose the `display_value` in braces which is needed for
    column headers using an siunitx package.

    Requires the package {multirow}, whereas multicol support is usually built in
    to the {tabular} environment.

    Examples
    --------
    >>> cell = {"cellstyle": "", "display_value": "text", "attributes": 'colspan="3"'}
    >>> _parse_latex_header_span(cell, "t", "c")
    '\\multicolumn{3}{c}{text}'
    """
    # 解析并处理单元格样式，将 CSS 样式转换为 LaTeX 格式
    display_val = _parse_latex_cell_styles(
        cell["cellstyle"], cell["display_value"], convert_css
    )
    if "attributes" in cell:
        attrs = cell["attributes"]
        # 检查是否存在列合并属性 'colspan="n"'
        if 'colspan="' in attrs:
            # 提取并解析列合并数目
            colspan = attrs[attrs.find('colspan="') + 9 :]  # len('colspan="') = 9
            colspan = int(colspan[: colspan.find('"')])
            # 根据 multicol_align 参数生成 LaTeX 多列命令
            if "naive-l" == multicol_align:
                out = f"{{{display_val}}}" if wrap else f"{display_val}"
                blanks = " & {}" if wrap else " &"
                return out + blanks * (colspan - 1)
            elif "naive-r" == multicol_align:
                out = f"{{{display_val}}}" if wrap else f"{display_val}"
                blanks = "{} & " if wrap else "& "
                return blanks * (colspan - 1) + out
            # 生成 LaTeX 多列命令，指定列数和对齐方式
            return f"\\multicolumn{{{colspan}}}{{{multicol_align}}}{{{display_val}}}"
        # 检查是否存在行合并属性 'rowspan="n"'
        elif 'rowspan="' in attrs:
            # 如果 multirow_align 为 "naive"，直接返回显示值 display_val
            if multirow_align == "naive":
                return display_val
            # 提取并解析行合并数目
            rowspan = attrs[attrs.find('rowspan="') + 9 :]
            rowspan = int(rowspan[: rowspan.find('"')])
            # 生成 LaTeX 多行命令，指定行数和内容对齐方式
            return f"\\multirow[{multirow_align}]{{{rowspan}}}{{*}}{{{display_val}}}"
    # 如果 wrap 参数为 True，使用大括号包裹 display_val
    if wrap:
        return f"{{{display_val}}}"
    else:
        return display_val


def _parse_latex_options_strip(value: str | float, arg: str) -> str:
    """
    Strip a css_value which may have latex wrapping arguments, css comment identifiers,
    and whitespaces, to a valid string for latex options parsing.

    For example: 'red /* --wrap */  ' --> 'red'
    """
    # 移除 value 中的 arg 参数，以及 CSS 注释标识符，并去除首尾空白字符
    return str(value).replace(arg, "").replace("/*", "").replace("*/", "").strip()


def _parse_latex_css_conversion(styles: CSSList) -> CSSList:
    """
    Convert CSS (attribute,value) pairs to equivalent LaTeX (command,options) pairs.

    Ignore conversion if tagged with `--latex` option, skipped if no conversion found.
    """

    def font_weight(value, arg) -> tuple[str, str] | None:
        # 如果 value 为 "bold" 或 "bolder"，返回 LaTeX 字体加粗命令及相应参数
        if value in ("bold", "bolder"):
            return "bfseries", f"{arg}"
        return None
    def font_style(value, arg) -> tuple[str, str] | None:
        # 如果值为 "italic"，返回 ("itshape", value)
        if value == "italic":
            return "itshape", f"{arg}"
        # 如果值为 "oblique"，返回 ("slshape", value)
        if value == "oblique":
            return "slshape", f"{arg}"
        # 否则返回 None
        return None

    def color(value, user_arg, command, comm_arg):
        """
        CSS colors have 5 formats to process:

         - 6 digit hex code: "#ff23ee"     --> [HTML]{FF23EE}
         - 3 digit hex code: "#f0e"        --> [HTML]{FF00EE}
         - rgba: rgba(128, 255, 0, 0.5)    --> [rgb]{0.502, 1.000, 0.000}
         - rgb: rgb(128, 255, 0,)          --> [rbg]{0.502, 1.000, 0.000}
         - string: red                     --> {red}

        Additionally rgb or rgba can be expressed in % which is also parsed.
        """
        # 如果用户参数不为空，则使用用户参数，否则使用命令参数
        arg = user_arg if user_arg != "" else comm_arg

        # 如果值以 "#" 开头且长度为 7，则颜色是 6 位十六进制代码
        if value[0] == "#" and len(value) == 7:
            return command, f"[HTML]{{{value[1:].upper()}}}{arg}"
        # 如果值以 "#" 开头且长度为 4，则颜色是 3 位十六进制代码
        if value[0] == "#" and len(value) == 4:
            val = f"{value[1].upper()*2}{value[2].upper()*2}{value[3].upper()*2}"
            return command, f"[HTML]{{{val}}}{arg}"
        # 如果值以 "rgb" 开头，则颜色是 rgb 或 rgba 格式
        elif value[:3] == "rgb":
            # 提取红色分量并转换为小数
            r = re.findall("(?<=\\()[0-9\\s%]+(?=,)", value)[0].strip()
            r = float(r[:-1]) / 100 if "%" in r else int(r) / 255
            # 提取绿色分量并转换为小数
            g = re.findall("(?<=,)[0-9\\s%]+(?=,)", value)[0].strip()
            g = float(g[:-1]) / 100 if "%" in g else int(g) / 255
            # 如果是 rgba 格式，提取蓝色分量
            if value[3] == "a":
                b = re.findall("(?<=,)[0-9\\s%]+(?=,)", value)[1].strip()
            else:
                b = re.findall("(?<=,)[0-9\\s%]+(?=\\))", value)[0].strip()
            b = float(b[:-1]) / 100 if "%" in b else int(b) / 255
            return command, f"[rgb]{{{r:.3f}, {g:.3f}, {b:.3f}}}{arg}"
        else:
            # 如果颜色是字符串形式的命名颜色，直接返回
            return command, f"{{{value}}}{arg}"

    CONVERTED_ATTRIBUTES: dict[str, Callable] = {
        "font-weight": font_weight,
        "background-color": partial(color, command="cellcolor", comm_arg="--lwrap"),
        "color": partial(color, command="color", comm_arg=""),
        "font-style": font_style,
    }

    latex_styles: CSSList = []
    # 遍历样式列表，处理特定的 LaTeX 样式
    for attribute, value in styles:
        if isinstance(value, str) and "--latex" in value:
            # 将样式添加到 LaTeX 样式列表，移除 '--latex'
            latex_styles.append((attribute, value.replace("--latex", "")))
        if attribute in CONVERTED_ATTRIBUTES:
            arg = ""
            # 检查样式中是否包含特定的 LaTeX 选项
            for x in ["--wrap", "--nowrap", "--lwrap", "--dwrap", "--rwrap"]:
                if x in str(value):
                    # 解析并获取 LaTeX 选项，同时更新样式值
                    arg, value = x, _parse_latex_options_strip(value, x)
                    break
            # 转换并获取 LaTeX 样式
            latex_style = CONVERTED_ATTRIBUTES[attribute](value, arg)
            if latex_style is not None:
                # 如果转换成功，则将结果添加到 LaTeX 样式列表
                latex_styles.extend([latex_style])
    # 返回最终的 LaTeX 样式列表
    return latex_styles
def _escape_latex(s: str) -> str:
    r"""
    Replace the characters ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``,
    ``~``, ``^``, and ``\`` in the string with LaTeX-safe sequences.

    Use this if you need to display text that might contain such characters in LaTeX.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    return (
        s.replace("\\", "ab2§=§8yz")  # 替换反斜杠为罕见字符串，以避免与最终转换冲突
        .replace("ab2§=§8yz ", "ab2§=§8yz\\space ")  # 因为 \backslash 会吞掉空格
        .replace("&", "\\&")  # 替换 & 为 \&
        .replace("%", "\\%")  # 替换 % 为 \%
        .replace("$", "\\$")  # 替换 $ 为 \$
        .replace("#", "\\#")  # 替换 # 为 \#
        .replace("_", "\\_")  # 替换 _ 为 \_
        .replace("{", "\\{")  # 替换 { 为 \{
        .replace("}", "\\}")  # 替换 } 为 \}
        .replace("~ ", "~\\space ")  # 因为 \textasciitilde 会吞掉空格
        .replace("~", "\\textasciitilde ")  # 替换 ~ 为 \textasciitilde
        .replace("^ ", "^\\space ")  # 因为 \textasciicircum 会吞掉空格
        .replace("^", "\\textasciicircum ")  # 替换 ^ 为 \textasciicircum
        .replace("ab2§=§8yz", "\\textbackslash ")  # 将罕见字符串替换回 \textbackslash
    )


def _math_mode_with_dollar(s: str) -> str:
    r"""
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``$`` and end with ``$``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace(r"\$", r"rt8§=§7wz")  # 替换 \$ 为罕见字符串，以避免干扰
    pattern = re.compile(r"\$.*?\$")
    pos = 0
    ps = pattern.search(s, pos)
    res = []
    while ps:
        res.append(_escape_latex(s[pos : ps.span()[0]]))  # 调用 _escape_latex 处理非数学模式部分
        res.append(ps.group())  # 保留数学模式部分
        pos = ps.span()[1]
        ps = pattern.search(s, pos)

    res.append(_escape_latex(s[pos : len(s)]))  # 处理最后的非数学模式部分
    return "".join(res).replace(r"rt8§=§7wz", r"\$")  # 将罕见字符串替换回 \$


def _math_mode_with_parentheses(s: str) -> str:
    r"""
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``\(`` and end with ``\)``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace(r"\(", r"LEFT§=§6yzLEFT").replace(r"\)", r"RIGHTab5§=§RIGHT")
    res = []
    # 使用正则表达式按照指定的模式分割字符串s，生成列表item
    for item in re.split(r"LEFT§=§6yz|ab5§=§RIGHT", s):
        # 如果item以"LEFT"开头并以"RIGHT"结尾，替换其中的"LEFT"和"RIGHT"为"\("和"\)"
        if item.startswith("LEFT") and item.endswith("RIGHT"):
            res.append(item.replace("LEFT", r"\(").replace("RIGHT", r"\)"))
        # 如果item中同时包含"LEFT"和"RIGHT"，调用_escape_latex函数转义并替换其中的"LEFT"和"RIGHT"为"\("和"\)"
        elif "LEFT" in item and "RIGHT" in item:
            res.append(
                _escape_latex(item).replace("LEFT", r"\(").replace("RIGHT", r"\)")
            )
        # 否则，调用_escape_latex函数转义item，并将其中的"LEFT"和"RIGHT"替换为"\textbackslash ("和"\textbackslash )"
        else:
            res.append(
                _escape_latex(item)
                .replace("LEFT", r"\textbackslash (")
                .replace("RIGHT", r"\textbackslash )")
            )
    # 将列表res中的所有字符串连接成一个字符串，并返回结果
    return "".join(res)
def _escape_latex_math(s: str) -> str:
    r"""
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which either are surrounded
    by two characters ``$`` or start with the character ``\(`` and end with ``\)``,
    are preserved without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    # 替换所有的 "\$" 为 "rt8§=§7wz"，以便稍后处理
    s = s.replace(r"\$", r"rt8§=§7wz")
    # 寻找第一个匹配 "$...$" 的子字符串
    ps_d = re.compile(r"\$.*?\$").search(s, 0)
    # 寻找第一个匹配 "\(...\)" 的子字符串
    ps_p = re.compile(r"\(.*?\)").search(s, 0)
    # 存储匹配到的模式的起始位置
    mode = []
    if ps_d:
        mode.append(ps_d.span()[0])
    if ps_p:
        mode.append(ps_p.span()[0])
    # 如果没有匹配到任何模式，调用 _escape_latex 函数处理字符串
    if len(mode) == 0:
        return _escape_latex(s.replace(r"rt8§=§7wz", r"\$"))
    # 如果第一个匹配的模式是 "$" 开始的，则调用 _math_mode_with_dollar 函数处理
    if s[mode[0]] == r"$":
        return _math_mode_with_dollar(s.replace(r"rt8§=§7wz", r"\$"))
    # 如果第一个匹配的模式是 "\(" 开始的，则调用 _math_mode_with_parentheses 函数处理
    if s[mode[0] - 1 : mode[0] + 1] == r"\(":
        return _math_mode_with_parentheses(s.replace(r"rt8§=§7wz", r"\$"))
    # 否则调用 _escape_latex 函数处理字符串
    else:
        return _escape_latex(s.replace(r"rt8§=§7wz", r"\$"))
```