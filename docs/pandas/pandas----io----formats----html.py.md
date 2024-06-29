# `D:\src\scipysrc\pandas\pandas\io\formats\html.py`

```
"""
Module for formatting output data in HTML.
"""

from __future__ import annotations

from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    cast,
)

from pandas._config import get_option

from pandas._libs import lib

from pandas import (
    MultiIndex,
    option_context,
)

from pandas.io.common import is_url
from pandas.io.formats.format import (
    DataFrameFormatter,
    get_level_lengths,
)
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
    )


class HTMLFormatter:
    """
    Internal class for formatting output data in html.
    This class is intended for shared functionality between
    DataFrame.to_html() and DataFrame._repr_html_().
    Any logic in common with other output formatting methods
    should ideally be inherited from classes in format.py
    and this class responsible for only producing html markup.
    """

    indent_delta: Final = 2

    def __init__(
        self,
        formatter: DataFrameFormatter,
        classes: str | list[str] | tuple[str, ...] | None = None,
        border: int | bool | None = None,
        table_id: str | None = None,
        render_links: bool = False,
    ) -> None:
        # 初始化方法，接受DataFrameFormatter对象、CSS类、表格边框、表格ID和是否渲染链接作为参数
        self.fmt = formatter
        self.classes = classes
        self.frame = self.fmt.frame  # 获取格式化器中的数据框对象
        self.columns = self.fmt.tr_frame.columns  # 获取数据框的列名
        self.elements: list[str] = []  # 初始化一个空列表，用于存储生成的HTML元素
        self.bold_rows = self.fmt.bold_rows  # 获取是否加粗行的标志
        self.escape = self.fmt.escape  # 获取是否转义HTML的标志
        self.show_dimensions = self.fmt.show_dimensions  # 获取是否显示维度的标志
        if border is None or border is True:
            border = cast(int, get_option("display.html.border"))
        elif not border:
            border = None

        self.border = border  # 设置表格边框的属性
        self.table_id = table_id  # 设置表格ID的属性
        self.render_links = render_links  # 设置是否渲染链接的属性

        self.col_space = {}  # 初始化列宽度的字典
        is_multi_index = isinstance(self.columns, MultiIndex)
        for column, value in self.fmt.col_space.items():
            col_space_value = f"{value}px" if isinstance(value, int) else value
            self.col_space[column] = col_space_value
            # 处理多级索引的情况，将列名展平并添加到列宽度字典中
            if is_multi_index and isinstance(column, tuple):
                for column_index in column:
                    self.col_space[str(column_index)] = col_space_value

    def to_string(self) -> str:
        # 将生成的HTML元素列表转换为字符串输出
        lines = self.render()
        if any(isinstance(x, str) for x in lines):
            lines = [str(x) for x in lines]
        return "\n".join(lines)

    def render(self) -> list[str]:
        # 渲染HTML表格并生成HTML元素列表
        self._write_table()

        if self.should_show_dimensions:  # 如果需要显示维度信息
            by = chr(215)  # × 符号
            self.write(
                f"<p>{len(self.frame)} rows {by} {len(self.frame.columns)} columns</p>"
            )

        return self.elements  # 返回生成的HTML元素列表
    # 返回一个布尔值，指示是否应该显示维度信息，基于格式设置对象的相应属性
    def should_show_dimensions(self) -> bool:
        return self.fmt.should_show_dimensions

    # 返回一个布尔值，指示是否应该显示行索引名称，基于格式设置对象的相应属性
    @property
    def show_row_idx_names(self) -> bool:
        return self.fmt.show_row_idx_names

    # 返回一个布尔值，指示是否应该显示列索引名称，基于格式设置对象的相应属性
    @property
    def show_col_idx_names(self) -> bool:
        return self.fmt.show_col_idx_names

    # 返回一个整数，表示行级别的数量，根据格式设置对象的索引属性决定
    @property
    def row_levels(self) -> int:
        if self.fmt.index:
            # 如果正在显示行索引，则返回数据帧索引的级别数
            return self.frame.index.nlevels
        elif self.show_col_idx_names:
            # 如果列索引名称正在显示，则返回1，参见gh-22579的说明
            # 当列索引命名时，即使是标准索引，也会发生列不对齐。
            # 如果不显示行索引，则需要在数据帧值之前包含一个空白单元格列。
            return 1
        # 如果不显示行索引，则返回0
        return 0

    # 返回一个迭代器，提供格式化后的列名
    def _get_columns_formatted_values(self) -> Iterable:
        return self.columns

    # 返回一个布尔值，指示数据是否被截断显示，基于格式设置对象的相应属性
    @property
    def is_truncated(self) -> bool:
        return self.fmt.is_truncated

    # 返回一个整数，表示数据帧的列数，基于格式设置对象的相应属性
    @property
    def ncols(self) -> int:
        return len(self.fmt.tr_frame.columns)

    # 将格式化后的字符串写入内部列表，带有指定的缩进
    def write(self, s: Any, indent: int = 0) -> None:
        rs = pprint_thing(s)  # 格式化输入数据为字符串
        self.elements.append(" " * indent + rs)  # 将格式化后的字符串加入元素列表，带有指定的缩进

    # 写入一个格式化的<th>单元格，可以设置最小宽度和标签
    def write_th(
        self, s: Any, header: bool = False, indent: int = 0, tags: str | None = None
    ) -> None:
        """
        Method for writing a formatted <th> cell.

        If col_space is set on the formatter then that is used for
        the value of min-width.

        Parameters
        ----------
        s : object
            The data to be written inside the cell.
        header : bool, default False
            Set to True if the <th> is for use inside <thead>.  This will
            cause min-width to be set if there is one.
        indent : int, default 0
            The indentation level of the cell.
        tags : str, default None
            Tags to include in the cell.

        Returns
        -------
        A written <th> cell.
        """
        col_space = self.col_space.get(s, None)  # 获取给定数据s对应的列空间值

        if header and col_space is not None:
            tags = tags or ""
            tags += f'style="min-width: {col_space};"'  # 如果是表头且有列空间值，则设置最小宽度样式

        self._write_cell(s, kind="th", indent=indent, tags=tags)  # 调用内部方法写入<th>单元格

    # 写入一个格式化的<td>单元格，可以设置标签和缩进
    def write_td(self, s: Any, indent: int = 0, tags: str | None = None) -> None:
        self._write_cell(s, kind="td", indent=indent, tags=tags)  # 调用内部方法写入<td>单元格

    # 内部方法：写入一个格式化的单元格，可以指定类型、缩进和标签
    def _write_cell(
        self, s: Any, kind: str = "td", indent: int = 0, tags: str | None = None
    ):
        # 实现写入单元格的具体逻辑，供write_th和write_td方法调用
        pass  # 在这个注释中可以写明具体的实现逻辑，但是不要改变代码结构
    ) -> None:
        # 如果传入了标签参数，则使用指定的标签创建开始标签
        if tags is not None:
            start_tag = f"<{kind} {tags}>"
        else:
            # 否则只使用类型创建开始标签
            start_tag = f"<{kind}>"

        # 如果需要转义文本
        if self.escape:
            # 定义转义字符字典，防止 & 被重复转义
            esc = {"&": r"&amp;", "<": r"&lt;", ">": r"&gt;"}
        else:
            esc = {}

        # 使用 pprint_thing 函数处理输入的对象 s，并去除两侧空白字符
        rs = pprint_thing(s, escape_chars=esc).strip()

        # 如果需要渲染链接且 rs 是一个 URL
        if self.render_links and is_url(rs):
            # 对 rs 进行非转义处理，并添加到链接标签中
            rs_unescaped = pprint_thing(s, escape_chars={}).strip()
            start_tag += f'<a href="{rs_unescaped}" target="_blank">'
            end_a = "</a>"
        else:
            end_a = ""

        # 写入最终的标签字符串，包括开始标签、处理后的内容 rs、链接结束标签和结束标签
        self.write(f"{start_tag}{rs}{end_a}</{kind}>", indent)

    def write_tr(
        self,
        line: Iterable,
        indent: int = 0,
        indent_delta: int = 0,
        header: bool = False,
        align: str | None = None,
        tags: dict[int, str] | None = None,
        nindex_levels: int = 0,
    ) -> None:
        # 如果 tags 参数为 None，则将其初始化为空字典
        if tags is None:
            tags = {}

        # 根据 align 参数确定是否为居中对齐，写入相应的开始行标签
        if align is None:
            self.write("<tr>", indent)
        else:
            self.write(f'<tr style="text-align: {align};">', indent)
        indent += indent_delta

        # 遍历传入的 line 参数，同时处理对应的 tags 参数
        for i, s in enumerate(line):
            val_tag = tags.get(i, None)
            # 如果是表头或者需要加粗的行索引
            if header or (self.bold_rows and i < nindex_levels):
                # 调用 write_th 方法写入表头单元格
                self.write_th(s, indent=indent, header=header, tags=val_tag)
            else:
                # 否则调用 write_td 方法写入数据单元格
                self.write_td(s, indent, tags=val_tag)

        indent -= indent_delta
        # 写入结束行标签
        self.write("</tr>", indent)

    def _write_table(self, indent: int = 0) -> None:
        _classes = ["dataframe"]  # 默认的 CSS 类名为 "dataframe"
        # 获取是否使用 MathJax 的选项
        use_mathjax = get_option("display.html.use_mathjax")
        # 如果不使用 MathJax，则将 "tex2jax_ignore" 类名加入到 _classes 中
        if not use_mathjax:
            _classes.append("tex2jax_ignore")
        # 如果定义了自定义的 CSS 类名，则将其加入到 _classes 中
        if self.classes is not None:
            if isinstance(self.classes, str):
                self.classes = self.classes.split()
            if not isinstance(self.classes, (list, tuple)):
                raise TypeError(
                    "classes must be a string, list, "
                    f"or tuple, not {type(self.classes)}"
                )
            _classes.extend(self.classes)

        # 如果未定义表格的 ID，则将 id_section 置为空字符串
        if self.table_id is None:
            id_section = ""
        else:
            id_section = f' id="{self.table_id}"'

        # 如果未定义表格的边框属性，则将 border_attr 置为空字符串
        if self.border is None:
            border_attr = ""
        else:
            border_attr = f' border="{self.border}"'

        # 写入表格的开始标签，包括边框属性、CSS 类名和 ID
        self.write(
            f'<table{border_attr} class="{" ".join(_classes)}"{id_section}>',
            indent,
        )

        # 如果设置了表格头部格式化或者显示行索引名称，则调用 _write_header 方法写入表头
        if self.fmt.header or self.show_row_idx_names:
            self._write_header(indent + self.indent_delta)

        # 调用 _write_body 方法写入表格的主体内容
        self._write_body(indent + self.indent_delta)

        # 写入表格的结束标签
        self.write("</table>", indent)
    # 写入普通行数据到表格主体
    def _write_regular_rows(
        self, fmt_values: Mapping[int, list[str]], indent: int
    ) -> None:
        self.write("<tbody>", indent)  # 在给定的缩进位置写入<tbody>标签，表示表格主体开始
        fmt_values = self._get_formatted_values()  # 获取格式化后的数值字典

        # 写入值
        if self.fmt.index and isinstance(self.frame.index, MultiIndex):
            # 如果需要索引并且索引是多级索引，则调用函数写入层次化行
            self._write_hierarchical_rows(fmt_values, indent + self.indent_delta)
        else:
            # 否则调用函数写入普通行
            self._write_regular_rows(fmt_values, indent + self.indent_delta)

        self.write("</tbody>", indent)  # 在给定的缩进位置写入</tbody>标签，表示表格主体结束
    # 定义方法，用于绘制格式化的表格行
    ) -> None:
        # 获取水平截断状态
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        # 获取垂直截断状态
        is_truncated_vertically = self.fmt.is_truncated_vertically

        # 获取表格行数
        nrows = len(self.fmt.tr_frame)

        # 如果存在索引列
        if self.fmt.index:
            # 获取索引格式化函数
            fmt = self.fmt._get_formatter("__index__")
            # 如果格式化函数不为空
            if fmt is not None:
                # 对索引列进行格式化映射
                index_values = self.fmt.tr_frame.index.map(fmt)
            else:
                # 仅在非多级索引时执行到这里
                index_values = self.fmt.tr_frame.index._format_flat(include_name=False)

        # 初始化行列表
        row: list[str] = []

        # 遍历每一行
        for i in range(nrows):
            # 如果垂直截断开启且当前行为截断行
            if is_truncated_vertically and i == (self.fmt.tr_row_num):
                # 创建由省略号组成的分隔行
                str_sep_row = ["..."] * len(row)
                # 调用写入方法输出分隔行
                self.write_tr(
                    str_sep_row,
                    indent,
                    self.indent_delta,
                    tags=None,
                    nindex_levels=self.row_levels,
                )

            # 清空当前行列表
            row = []

            # 如果存在索引列
            if self.fmt.index:
                # 添加当前行的索引值
                row.append(index_values[i])

            # 如果需要显示列索引名称
            elif self.show_col_idx_names:
                # 在数据单元格之前添加空单元格
                row.append("")

            # 向当前行列表中添加格式化后的数据值
            row.extend(fmt_values[j][i] for j in range(self.ncols))

            # 如果开启了水平截断
            if is_truncated_horizontally:
                # 计算添加省略号的列索引
                dot_col_ix = self.fmt.tr_col_num + self.row_levels
                # 在相应位置插入省略号
                row.insert(dot_col_ix, "...")

            # 调用写入方法输出当前行
            self.write_tr(
                row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels
            )

    # 定义方法，用于写入层次化的行数据
    def _write_hierarchical_rows(
        self, fmt_values: Mapping[int, list[str]], indent: int
class NotebookFormatter(HTMLFormatter):
    """
    Internal class for formatting output data in html for display in Jupyter
    Notebooks. This class is intended for functionality specific to
    DataFrame._repr_html_() and DataFrame.to_html(notebook=True)
    """

    # 返回一个字典，包含格式化后的列数据
    def _get_formatted_values(self) -> dict[int, list[str]]:
        return {i: self.fmt.format_col(i) for i in range(self.ncols)}

    # 返回一个列表，包含格式化后的列数据，适用于非多重索引情况
    def _get_columns_formatted_values(self) -> list[str]:
        # only reached with non-Multi Index
        return self.columns._format_flat(include_name=False)

    # 输出样式信息到HTML中，使用"scoped"属性来限定样式的作用范围仅限于当前元素
    def write_style(self) -> None:
        template_first = """\
            <style scoped>"""
        template_last = """\
            </style>"""
        template_select = """\
                .dataframe %s {
                    %s: %s;
                }"""
        element_props = [
            ("tbody tr th:only-of-type", "vertical-align", "middle"),
            ("tbody tr th", "vertical-align", "top"),
        ]
        if isinstance(self.columns, MultiIndex):
            element_props.append(("thead tr th", "text-align", "left"))
            if self.show_row_idx_names:
                element_props.append(
                    ("thead tr:last-of-type th", "text-align", "right")
                )
        else:
            element_props.append(("thead th", "text-align", "right"))
        template_mid = "\n\n".join(template_select % t for t in element_props)
        # 组合最终的样式模板
        template = dedent(f"{template_first}\n{template_mid}\n{template_last}")
        # 将样式模板写入当前对象
        self.write(template)

    # 渲染HTML内容，并返回元素列表
    def render(self) -> list[str]:
        # 将<div>标签写入HTML
        self.write("<div>")
        # 写入样式信息到HTML
        self.write_style()
        # 调用父类方法渲染HTML内容
        super().render()
        # 写入</div>标签到HTML
        self.write("</div>")
        # 返回元素列表
        return self.elements
```