# `D:\src\scipysrc\pandas\pandas\io\formats\style.py`

```
    """
    Module for applying conditional formatting to DataFrames and Series.
    """

    # 用于将 DataFrame 和 Series 进行条件格式化的模块

    from __future__ import annotations

    import copy
    from functools import partial
    import operator
    from typing import (
        TYPE_CHECKING,
        overload,
    )

    import numpy as np

    from pandas._config import get_option

    from pandas.compat._optional import import_optional_dependency
    from pandas.util._decorators import (
        Substitution,
        doc,
    )

    import pandas as pd
    from pandas import (
        IndexSlice,
        RangeIndex,
    )
    import pandas.core.common as com
    from pandas.core.frame import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame
    from pandas.core.shared_docs import _shared_docs

    from pandas.io.formats.format import save_to_buffer

    # 导入 jinja2 用于 DataFrame.style 需要的可选依赖
    jinja2 = import_optional_dependency("jinja2", extra="DataFrame.style requires jinja2.")

    from pandas.io.formats.style_render import (
        CSSProperties,
        CSSStyles,
        ExtFormatter,
        StylerRenderer,
        Subset,
        Tooltips,
        format_table_styles,
        maybe_convert_css_to_tuples,
        non_reducing_slice,
        refactor_levels,
    )

    if TYPE_CHECKING:
        from collections.abc import (
            Callable,
            Hashable,
            Sequence,
        )

        from matplotlib.colors import Colormap

        from pandas._typing import (
            Any,
            Axis,
            AxisInt,
            Concatenate,
            ExcelWriterMergeCells,
            FilePath,
            IndexLabel,
            IntervalClosedType,
            Level,
            P,
            QuantileInterpolation,
            Scalar,
            Self,
            StorageOptions,
            T,
            WriteBuffer,
            WriteExcelBuffer,
        )

        from pandas import ExcelWriter


    ####
    # Shared Doc Strings

    # Subset 参数的文档字符串
    subset_args = """subset : label, array-like, IndexSlice, optional
                A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
                or single key, to `DataFrame.loc[:, <subset>]` where the columns are
                prioritised, to limit ``data`` to *before* applying the function."""

    # Props 参数的文档字符串
    properties_args = """props : str, default None
               CSS properties to use for highlighting. If ``props`` is given, ``color``
               is not used."""

    # Color 参数的文档字符串
    coloring_args = """color : str, default '{default}'
               Background color to use for highlighting."""

    # Buf 参数的文档字符串
    buffering_args = """buf : str, path object, file-like object, optional
             String, path object (implementing ``os.PathLike[str]``), or file-like
             object implementing a string ``write()`` function. If ``None``, the result is
             returned as a string."""

    # Encoding 参数的文档字符串
    encoding_args = """encoding : str, optional
                  Character encoding setting for file output (and meta tags if available).
                  Defaults to ``pandas.options.styler.render.encoding`` value of "utf-8"."""

    #
    ###

    # Styler 类，继承自 StylerRenderer 类
    class Styler(StylerRenderer):
        r"""
        Helps style a DataFrame or Series according to the data with HTML and CSS.

        Parameters
        ----------
        data : Series or DataFrame
            Data to be styled - either a Series or DataFrame.
        """

        # 帮助使用 HTML 和 CSS 样式化 DataFrame 或 Series 的类
    # precision : int, optional
    #     控制浮点数的显示精度。如果未指定，则默认使用 pandas.options.styler.format.precision。

    # table_styles : list-like, default None
    #     用于定义表格样式的列表。每个元素是一个字典 {selector: (attr, value)}，具体说明见备注。

    # uuid : str, default None
    #     用于避免 CSS 冲突的唯一标识符，通常自动生成。

    # caption : str, tuple, default None
    #     表格的标题字符串。在 LaTeX 双标题中，使用元组。

    # table_attributes : str, default None
    #     出现在表格开头的 <table> 标签中的其他属性，通常自动包括 id 属性。

    # cell_ids : bool, default True
    #     如果为 True，则每个单元格会有一个 id 属性，格式为 T_<uuid>_row<num_row>_col<num_col>，
    #     其中 <uuid> 是唯一标识符，<num_row> 是行号，<num_col> 是列号。

    # na_rep : str, optional
    #     用于表示缺失值的字符串。如果未指定，将使用 pandas.options.styler.format.na_rep。

    # uuid_len : int, default 5
    #     如果未指定 uuid，将以十六进制字符形式生成的 uuid 长度，范围在 [0, 32] 之间。

    # decimal : str, optional
    #     用作浮点数、复数和整数的小数分隔符。如果未指定，将使用 pandas.options.styler.format.decimal。

    # thousands : str, optional, default None
    #     用作浮点数、复数和整数的千位分隔符。如果未指定，将使用 pandas.options.styler.format.thousands。

    # escape : str, optional
    #     控制单元格显示字符串中的特殊字符转义方式：
    #     - 'html'：用于 HTML 安全序列替换字符 &、<、>、' 和 "。
    #     - 'latex'：用于 LaTeX 安全序列替换字符 &、%、$、#、_、{、}、~、^ 和 \。
    #     - 'latex-math'：类似于 'latex' 模式，但对于数学子字符串，可以用 $ 包围或以 \( 开头和 \) 结尾。
    #     如果未指定，将使用 pandas.options.styler.format.escape。

    # formatter : str, callable, dict, optional
    #     用于定义如何显示值的对象。参见 Styler.format。如果未指定，将使用 pandas.options.styler.format.formatter。

    # Attributes
    # ----------
    # env : Jinja2 jinja2.Environment
    #     Jinja2 的环境对象，用于渲染模板。

    # template_html : Jinja2 Template
    #     HTML 渲染模板。

    # template_html_table : Jinja2 Template
    #     HTML 表格渲染模板。

    # template_html_style : Jinja2 Template
    #     HTML 样式渲染模板。

    # template_latex : Jinja2 Template
    #     LaTeX 渲染模板。

    # loader : Jinja2 Loader
    #     Jinja2 的加载器对象，用于加载模板。

    # See Also
    # --------
    # 其它相关信息（略）。
    # 定义一个类 `Styler`，用于为 DataFrame 和 Series 构建样式化的 HTML 表示。
    # 支持的参数包括：
    # - data: DataFrame 或 Series 对象，用于构建样式化表格的数据
    # - precision: int 或 None，控制数值显示的精度
    # - table_styles: CSS 样式表，用于自定义表格的外观
    # - uuid: str 或 None，用于唯一标识表格的 UUID
    # - caption: str、tuple 或 list 或 None，表格的标题
    # - table_attributes: str 或 None，表格的 HTML 属性
    # - cell_ids: bool，默认为 True，是否生成单元格的 ID
    # - na_rep: str 或 None，用于替换缺失值的字符串表示
    # - uuid_len: int，默认为 5，UUID 的长度
    # - decimal: str 或 None，数值的小数点符号
    # - thousands: str 或 None，千分位分隔符
    # - escape: str 或 None，用于避免 HTML 注入的转义方式
    # - formatter: ExtFormatter 或 None，格式化器对象
    def __init__(
        self,
        data: DataFrame | Series,
        precision: int | None = None,
        table_styles: CSSStyles | None = None,
        uuid: str | None = None,
        caption: str | tuple | list | None = None,
        table_attributes: str | None = None,
        cell_ids: bool = True,
        na_rep: str | None = None,
        uuid_len: int = 5,
        decimal: str | None = None,
        thousands: str | None = None,
        escape: str | None = None,
        formatter: ExtFormatter | None = None,
    ) -> None:
        """
        Initialize the Styler object with specified parameters.

        Args:
            data : DataFrame
                The data to be styled.
            uuid : Optional[str]
                A unique identifier for the Styler object.
            uuid_len : int
                Length of the UUID.
            table_styles : List[Dict[str, Union[str, Dict[str, str]]]]
                List of dictionaries specifying CSS styles for the table.
            table_attributes : Dict[str, str]
                Additional attributes to be applied to the table tag.
            caption : str
                Caption for the table.
            cell_ids : bool
                Whether to include cell IDs.
            precision : int
                Precision for floating point numbers.
        """

        super().__init__(
            data=data,
            uuid=uuid,
            uuid_len=uuid_len,
            table_styles=table_styles,
            table_attributes=table_attributes,
            caption=caption,
            cell_ids=cell_ids,
            precision=precision,
        )

        # validate ordered args
        thousands = thousands or get_option("styler.format.thousands")
        decimal = decimal or get_option("styler.format.decimal")
        na_rep = na_rep or get_option("styler.format.na_rep")
        escape = escape or get_option("styler.format.escape")
        formatter = formatter or get_option("styler.format.formatter")
        # precision is handled by superclass as default for performance

        # Apply formatting options to the Styler object
        self.format(
            formatter=formatter,
            precision=precision,
            na_rep=na_rep,
            escape=escape,
            decimal=decimal,
            thousands=thousands,
        )

    def _repr_html_(self) -> str | None:
        """
        Returns the HTML representation of the Styler object for Jupyter notebook display.

        Returns:
            str | None:
                HTML representation of the Styler object if rendering in HTML format,
                otherwise None.
        """
        if get_option("styler.render.repr") == "html":
            return self.to_html()
        return None

    def _repr_latex_(self) -> str | None:
        """
        Returns the LaTeX representation of the Styler object.

        Returns:
            str | None:
                LaTeX representation of the Styler object if rendering in LaTeX format,
                otherwise None.
        """
        if get_option("styler.render.repr") == "latex":
            return self.to_latex()
        return None

    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = None,
        css_class: str | None = None,
        as_title_attribute: bool = False,
    ):
        """
        Sets tooltips for the Styler object.

        Args:
            ttips : DataFrame
                DataFrame containing tooltips.
            props : CSSProperties | None, optional
                CSS properties for tooltips.
            css_class : str | None, optional
                CSS class for tooltips.
            as_title_attribute : bool, optional
                Whether to use the tooltip as the title attribute.

        Returns:
            None
        """

    @doc(
        NDFrame.to_excel,
        klass="Styler",
        storage_options=_shared_docs["storage_options"],
        storage_options_versionadded="1.5.0",
        extra_parameters="",
    )
    def to_excel(
        self,
        excel_writer: FilePath | WriteExcelBuffer | ExcelWriter,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: str | None = None,
        merge_cells: ExcelWriterMergeCells = True,
        encoding: str | None = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
    ):
        """
        Exports the styled DataFrame to an Excel file.

        Args:
            excel_writer : FilePath | WriteExcelBuffer | ExcelWriter
                File path, buffer, or ExcelWriter object to which the data is written.
            sheet_name : str, optional
                Name of the sheet within the Excel file.
            na_rep : str, optional
                String representation of NaN values.
            float_format : str | None, optional
                Format string for floating point numbers.
            columns : Sequence[Hashable] | None, optional
                Columns to include in the export.
            header : Sequence[Hashable] | bool, optional
                Row to use for the column labels.
            index : bool, optional
                Whether to include the index.
            index_label : IndexLabel | None, optional
                Column label for index column(s).
            startrow : int, optional
                Upper left cell row to dump data frame.
            startcol : int, optional
                Upper left cell column to dump data frame.
            engine : str | None, optional
                Write engine to use.
            merge_cells : ExcelWriterMergeCells, optional
                Whether to merge cells in Excel.
            encoding : str | None, optional
                Encoding of the resulting Excel file.
            inf_rep : str, optional
                Representation of infinite values.
            verbose : bool, optional
                Whether to display verbose output.
            freeze_panes : tuple[int, int] | None, optional
                Specify rows and columns to freeze in Excel.
            storage_options : StorageOptions | None, optional
                Options for storing data.

        Returns:
            None
        """
    ) -> None:
        # 导入 ExcelFormatter 类
        from pandas.io.formats.excel import ExcelFormatter
        
        # 创建 ExcelFormatter 的实例
        formatter = ExcelFormatter(
            self,
            na_rep=na_rep,
            cols=columns,
            header=header,
            float_format=float_format,
            index=index,
            index_label=index_label,
            merge_cells=merge_cells,
            inf_rep=inf_rep,
        )
        
        # 调用 ExcelFormatter 实例的 write 方法，将数据写入 Excel 文件
        formatter.write(
            excel_writer,
            sheet_name=sheet_name,
            startrow=startrow,
            startcol=startcol,
            freeze_panes=freeze_panes,
            engine=engine,
            storage_options=storage_options,
        )

    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool | None = ...,
        clines: str | None = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str | None = ...,
        multicol_align: str | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> None: ...

    @overload
    def to_latex(
        self,
        buf: None = ...,
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool | None = ...,
        clines: str | None = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str | None = ...,
        multicol_align: str | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> str: ...

    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        column_format: str | None = None,
        position: str | None = None,
        position_float: str | None = None,
        hrules: bool | None = None,
        clines: str | None = None,
        label: str | None = None,
        caption: str | tuple | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        multirow_align: str | None = None,
        multicol_align: str | None = None,
        siunitx: bool = False,
        environment: str | None = None,
        encoding: str | None = None,
        convert_css: bool = False,
    @overload
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        bold_headers: bool = ...,
        caption: str | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...,
        **kwargs,
    ) -> None:
    ```
    # 定义方法 `to_html`，用于将数据转换为 HTML 格式
    # 接受参数：
    # - `buf`: 文件路径或字符串写入缓冲区
    # - `table_uuid`: 表格的唯一标识符，可选
    # - `table_attributes`: 表格的属性字符串，可选
    # - `sparse_index`: 是否使用稀疏索引布尔值，可选
    # - `sparse_columns`: 是否使用稀疏列布尔值，可选
    # - `bold_headers`: 是否加粗表头布尔值，默认为真
    # - `caption`: 表格标题，可选
    # - `max_rows`: 最大行数限制，可选
    # - `max_columns`: 最大列数限制，可选
    # - `encoding`: 编码格式，可选
    # - `doctype_html`: 是否输出 HTML 文档类型声明，默认为假
    # - `exclude_styles`: 是否排除样式信息，默认为假
    # - `**kwargs`: 其他未列出的关键字参数

    @overload
    def to_html(
        self,
        buf: None = ...,
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        bold_headers: bool = ...,
        caption: str | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...,
        **kwargs,
    ) -> str:
    ```
    # 方法重载：`to_html` 的返回类型为字符串的版本
    # 参数与上一个定义的方法相同，除了 `buf` 参数允许为 `None`
    # 返回值为生成的 HTML 字符串

    @Substitution(buf=buffering_args, encoding=encoding_args)
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        table_uuid: str | None = None,
        table_attributes: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        bold_headers: bool = False,
        caption: str | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        encoding: str | None = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
        **kwargs,
    ):
    ```
    # 带参数替换装饰器 `Substitution` 用于生成文档
    # 定义方法 `to_html` 的实现，接受更多参数的版本，允许 `buf` 为 `None`
    # 参数与之前的定义相同，具体含义参考上述注释

    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        delimiter: str = ...,
    ) -> None:
    ```
    # 定义方法 `to_string`，用于将数据转换为字符串格式
    # 接受参数：
    # - `buf`: 文件路径或字符串写入缓冲区
    # - `encoding`: 编码格式，可选
    # - `sparse_index`: 是否使用稀疏索引布尔值，可选
    # - `sparse_columns`: 是否使用稀疏列布尔值，可选
    # - `max_rows`: 最大行数限制，可选
    # - `max_columns`: 最大列数限制，可选
    # - `delimiter`: 列之间的分隔符，默认为空格

    @overload
    def to_string(
        self,
        buf: None = ...,
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        delimiter: str = ...,
    ) -> str:
    ```
    # 方法重载：`to_string` 的返回类型为字符串的版本
    # 参数与上一个定义的方法相同，除了 `buf` 参数允许为 `None`
    # 返回值为生成的字符串

    @Substitution(buf=buffering_args, encoding=encoding_args)
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        delimiter: str = " ",
    ):
    ```
    # 带参数替换装饰器 `Substitution` 用于生成文档
    # 定义方法 `to_string` 的实现，接受更多参数的版本，允许 `buf` 为 `None`
    # 参数与之前的定义相同，具体含义参考上述注释
    ) -> str | None:
        """
        Write Styler to a file, buffer or string in text format.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        %(buf)s
            A buffer or file-like object to write the styled output. If None, the styled
            output is returned as a string.
        %(encoding)s
            The encoding to be used if `buf` is not None.
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows``, which is None.
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).
        delimiter : str, default single space
            The separator between data elements.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.to_string()
        ' A B\\n0 1 3\\n1 2 4\\n'
        """
        # 创建一个当前对象的深度拷贝
        obj = self._copy(deepcopy=True)

        # 如果 sparse_index 为 None，则使用默认的 styler.sparse.index 设置
        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        
        # 如果 sparse_columns 为 None，则使用默认的 styler.sparse.columns 设置
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        # 调用对象的 _render_string 方法，生成文本格式的样式化输出
        text = obj._render_string(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
            delimiter=delimiter,
        )

        # 将生成的文本保存到缓冲区或文件，或者作为字符串返回
        return save_to_buffer(
            text, buf=buf, encoding=(encoding if buf is not None else None)
        )
    def set_td_classes(self, classes: DataFrame) -> Styler:
        """
        Set the ``class`` attribute of ``<td>`` HTML elements.

        Parameters
        ----------
        classes : DataFrame
            DataFrame containing strings that will be translated to CSS classes,
            mapped by identical column and index key values that must exist on the
            underlying Styler data. None, NaN values, and empty strings will
            be ignored and not affect the rendered HTML.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_table_attributes: Set the table attributes added to the ``<table>``
            HTML element.

        Notes
        -----
        Can be used in combination with ``Styler.set_table_styles`` to define an
        internal CSS solution without reference to external CSS files.

        Examples
        --------
        >>> df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        >>> classes = pd.DataFrame(
        ...     [["min-val red", "", "blue"], ["red", None, "blue max-val"]],
        ...     index=df.index,
        ...     columns=df.columns,
        ... )
        >>> df.style.set_td_classes(classes)  # doctest: +SKIP

        Using `MultiIndex` columns and a `classes` `DataFrame` as a subset of the
        underlying,

        >>> df = pd.DataFrame(
        ...     [[1, 2], [3, 4]],
        ...     index=["a", "b"],
        ...     columns=[["level0", "level0"], ["level1a", "level1b"]],
        ... )
        >>> classes = pd.DataFrame(
        ...     ["min-val"], index=["a"], columns=[["level0"], ["level1a"]]
        ... )
        >>> df.style.set_td_classes(classes)  # doctest: +SKIP

        Form of the output with new additional css classes,

        >>> from pandas.io.formats.style import Styler
        >>> df = pd.DataFrame([[1]])
        >>> css = pd.DataFrame([["other-class"]])
        >>> s = Styler(df, uuid="_", cell_ids=False).set_td_classes(css)
        >>> s.hide(axis=0).to_html()  # doctest: +SKIP
        '<style type="text/css"></style>'
        '<table id="T__">'
        '  <thead>'
        '    <tr><th class="col_heading level0 col0" >0</th></tr>'
        '  </thead>'
        '  <tbody>'
        '    <tr><td class="data row0 col0 other-class" >1</td></tr>'
        '  </tbody>'
        '</table>'
        """
        # 检查 classes 的索引和列是否唯一，否则抛出错误
        if not classes.index.is_unique or not classes.columns.is_unique:
            raise KeyError(
                "Classes render only if `classes` has unique index and columns."
            )
        # 将 classes 重新索引以匹配 Styler 对象的数据结构
        classes = classes.reindex_like(self.data)

        # 遍历 classes 中的每一行和每个值，并更新对应单元格的 CSS 类
        for r, row_tup in enumerate(classes.itertuples()):
            for c, value in enumerate(row_tup[1:]):
                # 如果值不是 NaN 或空字符串，则更新对应单元格的 CSS 类
                if not (pd.isna(value) or value == ""):
                    self.cell_context[(r, c)] = str(value)

        # 返回更新后的 Styler 对象
        return self
    def _update_ctx(self, attrs: DataFrame) -> None:
        """
        Update the state of the ``Styler`` for data cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : DataFrame
            should contain strings of '<property>: <value>;<prop2>: <val2>'
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        """
        # 检查索引和列是否唯一，否则抛出异常
        if not self.index.is_unique or not self.columns.is_unique:
            raise KeyError(
                "`Styler.apply` and `.map` are not compatible "
                "with non-unique index or columns."
            )

        # 遍历属性DataFrame的列
        for cn in attrs.columns:
            # 获取当前列的位置索引
            j = self.columns.get_loc(cn)
            # 获取当前列的Series对象
            ser = attrs[cn]
            # 遍历Series对象的索引和值
            for rn, c in ser.items():
                # 如果值为空或者是NaN，则跳过
                if not c or pd.isna(c):
                    continue
                # 将字符串样式转换为元组列表
                css_list = maybe_convert_css_to_tuples(c)
                # 获取行索引i
                i = self.index.get_loc(rn)
                # 将样式信息添加到self.ctx的对应位置
                self.ctx[(i, j)].extend(css_list)

    def _update_ctx_header(self, attrs: DataFrame, axis: AxisInt) -> None:
        """
        Update the state of the ``Styler`` for header cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : Series
            Should contain strings of '<property>: <value>;<prop2>: <val2>', and an
            integer index.
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        axis : int
            Identifies whether the ctx object being updated is the index or columns
        """
        # 遍历属性DataFrame的列
        for j in attrs.columns:
            # 获取当前列的Series对象
            ser = attrs[j]
            # 遍历Series对象的索引和值
            for i, c in ser.items():
                # 如果值为空，则跳过
                if not c:
                    continue
                # 将字符串样式转换为元组列表
                css_list = maybe_convert_css_to_tuples(c)
                # 根据axis参数确定更新self.ctx_index或self.ctx_columns
                if axis == 0:
                    self.ctx_index[(i, j)].extend(css_list)
                else:
                    self.ctx_columns[(j, i)].extend(css_list)
    def _copy(self, deepcopy: bool = False) -> Styler:
        """
        Copies a Styler, allowing for deepcopy or shallow copy

        Copying a Styler aims to recreate a new Styler object which contains the same
        data and styles as the original.

        Data dependent attributes [copied and NOT exported]:
          - formatting (._display_funcs)
          - hidden index values or column values (.hidden_rows, .hidden_columns)
          - tooltips
          - cell_context (cell css classes)
          - ctx (cell css styles)
          - caption
          - concatenated stylers

        Non-data dependent attributes [copied and exported]:
          - css
          - hidden index state and hidden columns state (.hide_index_, .hide_columns_)
          - table_attributes
          - table_styles
          - applied styles (_todo)

        """
        # 创建一个新的 Styler 对象，根据当前对象的数据初始化
        styler = type(self)(
            self.data,  # 使用当前对象的数据初始化新对象的 'data', 'columns', 'index' 属性（浅拷贝）
        )
        shallow = [  # 简单的字符串或布尔值属性，进行浅拷贝
            "hide_index_",
            "hide_columns_",
            "hide_column_names",
            "hide_index_names",
            "table_attributes",
            "cell_ids",
            "caption",
            "uuid",
            "uuid_len",
            "template_latex",  # 如果有自定义模板，则也会被复制
            "template_html_style",
            "template_html_table",
            "template_html",
        ]
        deep = [  # 嵌套的列表或字典属性，进行深拷贝
            "css",
            "concatenated",
            "_display_funcs",
            "_display_funcs_index",
            "_display_funcs_columns",
            "_display_funcs_index_names",
            "_display_funcs_column_names",
            "hidden_rows",
            "hidden_columns",
            "ctx",
            "ctx_index",
            "ctx_columns",
            "cell_context",
            "_todo",
            "table_styles",
            "tooltips",
        ]

        # 对于浅拷贝的属性，将当前对象的对应属性值复制给新对象的对应属性
        for attr in shallow:
            setattr(styler, attr, getattr(self, attr))

        # 对于深拷贝的属性，根据 deepcopy 参数选择进行深拷贝或浅拷贝
        for attr in deep:
            val = getattr(self, attr)
            setattr(styler, attr, copy.deepcopy(val) if deepcopy else val)

        # 返回复制后的 Styler 对象
        return styler

    # 创建一个浅拷贝的 Styler 对象
    def __copy__(self) -> Styler:
        return self._copy(deepcopy=False)

    # 创建一个深拷贝的 Styler 对象
    def __deepcopy__(self, memo) -> Styler:
        return self._copy(deepcopy=True)
    # 生成一个新的 Styler 实例，作为清除样式后的备份副本
    clean_copy = Styler(self.data, uuid=self.uuid)
    # 获取 clean_copy 对象中所有非方法属性的列表
    clean_attrs = [a for a in clean_copy.__dict__ if not callable(a)]
    # 获取当前对象（self）中所有非方法属性的列表，可能还包括其他属性
    self_attrs = [a for a in self.__dict__ if not callable(a)]  # maybe more attrs
    # 将 clean_copy 中的所有属性复制到当前对象（self）中
    for attr in clean_attrs:
        setattr(self, attr, getattr(clean_copy, attr))
    # 删除当前对象（self）中存在但在 clean_copy 中不存在的属性
    for attr in set(self_attrs).difference(clean_attrs):
        delattr(self, attr)
    ) -> Styler:
        subset = slice(None) if subset is None else subset
        # 如果 subset 为 None，则使用 slice(None)，否则保持 subset 不变
        subset = non_reducing_slice(subset)
        # 对 subset 进行非缩减的切片处理
        data = self.data.loc[subset]
        # 根据 subset 获取相应的数据子集

        if data.empty:
            result = DataFrame()
            # 如果数据子集为空，则返回一个空的 DataFrame
        elif axis is None:
            result = func(data, **kwargs)
            # 如果 axis 为 None，则直接应用 func 函数到 data 上
            if not isinstance(result, DataFrame):
                if not isinstance(result, np.ndarray):
                    raise TypeError(
                        f"Function {func!r} must return a DataFrame or ndarray "
                        f"when passed to `Styler.apply` with axis=None"
                    )
                if data.shape != result.shape:
                    raise ValueError(
                        f"Function {func!r} returned ndarray with wrong shape.\n"
                        f"Result has shape: {result.shape}\n"
                        f"Expected shape: {data.shape}"
                    )
                result = DataFrame(result, index=data.index, columns=data.columns)
                # 如果 result 不是 DataFrame，则尝试将其转换为 DataFrame，保持索引和列名与 data 相同
        else:
            axis = self.data._get_axis_number(axis)
            # 获取 axis 对应的轴向编号
            if axis == 0:
                result = data.apply(func, axis=0, **kwargs)
                # 如果 axis 是 0，则沿着列方向应用 func 函数到 data 上
            else:
                result = data.T.apply(func, axis=0, **kwargs).T  # see GH 42005
                # 否则，沿着行方向应用 func 函数到 data 转置后的数据上，并将结果再次转置

        if isinstance(result, Series):
            raise ValueError(
                f"Function {func!r} resulted in the apply method collapsing to a "
                f"Series.\nUsually, this is the result of the function returning a "
                f"single value, instead of list-like."
            )
            # 如果结果是 Series，则抛出 ValueError，提示 func 函数返回了单一值而不是类似列表的结果

        msg = (
            f"Function {func!r} created invalid {{0}} labels.\nUsually, this is "
            f"the result of the function returning a "
            f"{'Series' if axis is not None else 'DataFrame'} which contains invalid "
            f"labels, or returning an incorrectly shaped, list-like object which "
            f"cannot be mapped to labels, possibly due to applying the function along "
            f"the wrong axis.\n"
            f"Result {{0}} has shape: {{1}}\n"
            f"Expected {{0}} shape:   {{2}}"
        )
        if not all(result.index.isin(data.index)):
            raise ValueError(msg.format("index", result.index.shape, data.index.shape))
            # 如果 result 的索引不完全包含在 data 的索引中，则抛出 ValueError，提示索引标签无效
        if not all(result.columns.isin(data.columns)):
            raise ValueError(
                msg.format("columns", result.columns.shape, data.columns.shape)
            )
            # 如果 result 的列名不完全包含在 data 的列名中，则抛出 ValueError，提示列名标签无效

        self._update_ctx(result)
        # 更新上下文环境，将 result 设置为当前 Styler 对象的结果
        return self
        # 返回当前 Styler 对象自身
    ) -> Styler:
        # 确定要操作的轴向，0表示行轴（index），1表示列轴（columns）
        axis = self.data._get_axis_number(axis)
        # 根据轴向选择要操作的对象，如果axis为0，则选择行索引(self.index)，否则选择列索引(self.columns)
        obj = self.index if axis == 0 else self.columns

        # 将给定的level参数重构为与obj兼容的级别列表
        levels_ = refactor_levels(level, obj)
        # 使用obj中的数据创建一个DataFrame，并仅选择指定的levels_列
        data = DataFrame(obj.to_list()).loc[:, levels_]

        # 根据method参数的值选择不同的方法对data进行处理
        if method == "apply":
            # 对data应用指定的func函数，沿指定的axis轴向执行，并传递额外的kwargs参数
            result = data.apply(func, axis=0, **kwargs)
        elif method == "map":
            # 使用func函数映射data中的每个元素，并传递额外的kwargs参数
            result = data.map(func, **kwargs)

        # 更新Styler对象的上下文头部信息，传入处理后的结果result和操作的轴向axis
        self._update_ctx_header(result, axis)
        # 返回更新后的Styler对象本身
        return self

    @doc(
        this="apply",
        wise="level-wise",
        alt="map",
        altwise="elementwise",
        func="take a Series and return a string array of the same length",
        input_note="the index as a Series, if an Index, or a level of a MultiIndex",
        output_note="an identically sized array of CSS styles as strings",
        var="label",
        ret='np.where(label == "B", "background-color: yellow;", "")',
        ret2='["background-color: yellow;" if "x" in v else "" for v in label]',
    )
    # 定义apply_index方法，用于根据给定的func函数对索引数据进行样式应用
    def apply_index(
        self,
        func: Callable,
        axis: AxisInt | str = 0,
        level: Level | list[Level] | None = None,
        **kwargs,
    ) -> Styler:
        """
        Apply a CSS-styling function to the index or column headers, {wise}.

        Updates the HTML representation with the result.

        .. versionadded:: 1.4.0

        .. versionadded:: 2.1.0
           Styler.applymap_index was deprecated and renamed to Styler.map_index.

        Parameters
        ----------
        func : function
            ``func`` should {func}.
            # 接受一个函数作为参数，该函数应该 {func}。
        axis : {{0, 1, "index", "columns"}}
            The headers over which to apply the function.
            # 指定应用函数的方向，可以是 0（行）、1（列）、"index"（索引）、"columns"（列）之一。
        level : int, str, list, optional
            If index is MultiIndex the level(s) over which to apply the function.
            # 如果索引是多级索引，指定应用函数的层级（单个层级、多个层级的列表或者 None）。
        **kwargs : dict
            Pass along to ``func``.
            # 其他传递给 ``func`` 的关键字参数。

        Returns
        -------
        Styler
            # 返回一个 Styler 对象。

        See Also
        --------
        Styler.{alt}_index: Apply a CSS-styling function to headers {altwise}.
            # 参见 Styler.{alt}_index：将 CSS 样式函数应用于头部 {altwise}。
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.
            # 参见 Styler.apply：逐列、逐行或整体应用 CSS 样式函数。
        Styler.map: Apply a CSS-styling function elementwise.
            # 参见 Styler.map：逐元素应用 CSS 样式函数。

        Notes
        -----
        Each input to ``func`` will be {input_note}. The output of ``func`` should be
        {output_note}, in the format 'attribute: value; attribute2: value2; ...'
        or, if nothing is to be applied to that element, an empty string or ``None``.
        # ``func`` 的每个输入应该是 {input_note}。``func`` 的输出应该是 {output_note}，
        # 格式为 'attribute: value; attribute2: value2; ...'，
        # 或者如果不应用任何样式，则返回空字符串或 ``None``。

        Examples
        --------
        Basic usage to conditionally highlight values in the index.

        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"])
        >>> def color_b({var}):
        ...     return {ret}
        >>> df.style.{this}_index(color_b)  # doctest: +SKIP
        # 基本用法：条件性地高亮索引中的值。

        .. figure:: ../../_static/style/appmaphead1.png

        Selectively applying to specific levels of MultiIndex columns.

        >>> midx = pd.MultiIndex.from_product([["ix", "jy"], [0, 1], ["x3", "z4"]])
        >>> df = pd.DataFrame([np.arange(8)], columns=midx)
        >>> def highlight_x({var}):
        ...     return {ret2}
        >>> df.style.{this}_index(
        ...     highlight_x, axis="columns", level=[0, 2])  # doctest: +SKIP
        # 选择性地应用于多级索引列的特定层级。

        .. figure:: ../../_static/style/appmaphead2.png
        """
        self._todo.append(
            (
                lambda instance: getattr(instance, "_apply_index"),
                (func, axis, level, "apply"),
                kwargs,
            )
        )
        # 将待办事项添加到对象的私有属性 _todo 中，包括函数、方向、层级和应用类型的信息。
        return self

    @doc(
        apply_index,
        this="map",
        wise="elementwise",
        alt="apply",
        altwise="level-wise",
        func="take a scalar and return a string",
        input_note="an index value, if an Index, or a level value of a MultiIndex",
        output_note="CSS styles as a string",
        var="label",
        ret='"background-color: yellow;" if label == "B" else None',
        ret2='"background-color: yellow;" if "x" in label else None',
    )
    def map_index(
        self,
        func: Callable,
        axis: AxisInt | str = 0,
        level: Level | list[Level] | None = None,
        **kwargs,
        # 根据索引或列标题应用 CSS 样式函数 {wise}。

        Updates the HTML representation with the result.
        # 使用结果更新 HTML 表示。

        .. versionadded:: 1.4.0

        .. versionadded:: 2.1.0
           Styler.applymap_index 被弃用并重命名为 Styler.map_index。

        Parameters
        ----------
        func : function
            ``func`` 应 {func}。
        axis : {{0, 1, "index", "columns"}}
    @Substitution(subset=subset_args)
    # 使用装饰器Substitution，替换subset参数为subset_args中定义的值
    def map(self, func: Callable, subset: Subset | None = None, **kwargs) -> Styler:
        """
        Apply a CSS-styling function elementwise.

        Updates the HTML representation with the result.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a string.
        %(subset)s
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.map_index: Apply a CSS-styling function to headers elementwise.
        Styler.apply_index: Apply a CSS-styling function to headers level-wise.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Notes
        -----
        The elements of the output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or,
        if nothing is to be applied to that element, an empty string or ``None``.

        Examples
        --------
        >>> def color_negative(v, color):
        ...     return f"color: {color};" if v < 0 else None
        >>> df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        >>> df.style.map(color_negative, color="red")  # doctest: +SKIP

        Using ``subset`` to restrict application to a single column or multiple columns

        >>> df.style.map(color_negative, color="red", subset="A")
        ... # doctest: +SKIP
        >>> df.style.map(color_negative, color="red", subset=["A", "B"])
        ... # doctest: +SKIP

        Using a 2d input to ``subset`` to select rows in addition to columns

        >>> df.style.map(
        ...     color_negative, color="red", subset=([0, 1, 2], slice(None))
        ... )  # doctest: +SKIP
        >>> df.style.map(color_negative, color="red", subset=(slice(0, 5, 2), "A"))
        ... # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        # 将要执行的操作添加到self._todo列表中，包括函数_apply_index和其参数(func, subset)，以及kwargs
        self._todo.append(
            (lambda instance: getattr(instance, "_map"), (func, subset), kwargs)
        )
        # 返回当前对象实例，用于链式调用
        return self
    def set_table_attributes(self, attributes: str) -> Styler:
        """
        Set the table attributes added to the ``<table>`` HTML element.

        These are items in addition to automatic (by default) ``id`` attribute.

        Parameters
        ----------
        attributes : str
            String containing additional attributes for the HTML ``<table>`` tag.

        Returns
        -------
        Styler
            Returns the current Styler object to allow method chaining.

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_td_classes: Set the DataFrame of strings added to the ``class``
            attribute of ``<td>`` HTML elements.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4))
        >>> df.style.set_table_attributes('class="pure-table"')  # doctest: +SKIP
        # ... <table class="pure-table"> ...
        """
        self.table_attributes = attributes
        return self

    def export(self) -> dict[str, Any]:
        """
        Export the styles applied to the current Styler.

        Can be applied to a second Styler with ``Styler.use``.

        Returns
        -------
        dict
            A dictionary containing non-data dependent attributes of the current Styler.

        See Also
        --------
        Styler.use: Set the styles on the current Styler.
        Styler.copy: Create a copy of the current Styler.

        Notes
        -----
        This method is designed to copy non-data dependent attributes of
        one Styler to another. It differs from ``Styler.copy`` where data and
        data dependent attributes are also copied.

        The following items are exported since they are not generally data dependent:

          - Styling functions added by the ``apply`` and ``map``
          - Whether axes and names are hidden from the display, if unambiguous.
          - Table attributes
          - Table styles

        The following attributes are considered data dependent and therefore not
        exported:

          - Caption
          - UUID
          - Tooltips
          - Any hidden rows or columns identified by Index labels
          - Any formatting applied using ``Styler.format``
          - Any CSS classes added using ``Styler.set_td_classes``

        Examples
        --------

        >>> styler = pd.DataFrame([[1, 2], [3, 4]]).style
        >>> styler2 = pd.DataFrame([[9, 9, 9]]).style
        >>> styler.hide(axis=0).highlight_max(axis=1)  # doctest: +SKIP
        >>> export = styler.export()
        >>> styler2.use(export)  # doctest: +SKIP
        """
        return {
            "apply": copy.copy(self._todo),
            "table_attributes": self.table_attributes,
            "table_styles": copy.copy(self.table_styles),
            "hide_index": all(self.hide_index_),
            "hide_columns": all(self.hide_columns_),
            "hide_index_names": self.hide_index_names,
            "hide_column_names": self.hide_column_names,
            "css": copy.copy(self.css),
        }
    def use(self, styles: dict[str, Any]) -> Styler:
        """
        Set the styles on the current Styler.

        Possibly uses styles from ``Styler.export``.

        Parameters
        ----------
        styles : dict(str, Any)
            List of attributes to add to Styler. Dict keys should contain only:
              - "apply": list of styler functions, typically added with ``apply`` or
                ``map``.
              - "table_attributes": HTML attributes, typically added with
                ``set_table_attributes``.
              - "table_styles": CSS selectors and properties, typically added with
                ``set_table_styles``.
              - "hide_index":  whether the index is hidden, typically added with
                ``hide_index``, or a boolean list for hidden levels.
              - "hide_columns": whether column headers are hidden, typically added with
                ``hide_columns``, or a boolean list for hidden levels.
              - "hide_index_names": whether index names are hidden.
              - "hide_column_names": whether column header names are hidden.
              - "css": the css class names used.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.export : Export the non data dependent attributes to the current Styler.

        Examples
        --------

        >>> styler = pd.DataFrame([[1, 2], [3, 4]]).style
        >>> styler2 = pd.DataFrame([[9, 9, 9]]).style
        >>> styler.hide(axis=0).highlight_max(axis=1)  # doctest: +SKIP
        >>> export = styler.export()
        >>> styler2.use(export)  # doctest: +SKIP
        """
        # 将 "apply" 中的样式函数添加到待处理列表中
        self._todo.extend(styles.get("apply", []))
        
        # 合并并设置表格的 HTML 属性
        table_attributes: str = self.table_attributes or ""
        obj_table_atts: str = (
            "" if styles.get("table_attributes") is None else str(styles.get("table_attributes"))
        )
        self.set_table_attributes((table_attributes + " " + obj_table_atts).strip())
        
        # 设置表格的 CSS 样式
        if styles.get("table_styles"):
            self.set_table_styles(styles.get("table_styles"), overwrite=False)

        # 处理索引和列的隐藏设置
        for obj in ["index", "columns"]:
            hide_obj = styles.get("hide_" + obj)
            if hide_obj is not None:
                if isinstance(hide_obj, bool):
                    # 如果 hide_obj 是布尔值，则设置相应级别的隐藏状态
                    n = getattr(self, obj).nlevels
                    setattr(self, "hide_" + obj + "_", [hide_obj] * n)
                else:
                    # 否则，设置特定级别的隐藏状态
                    setattr(self, "hide_" + obj + "_", hide_obj)

        # 设置是否隐藏索引和列的名称
        self.hide_index_names = styles.get("hide_index_names", False)
        self.hide_column_names = styles.get("hide_column_names", False)
        
        # 设置样式中的 CSS 类名
        if styles.get("css"):
            self.css = styles.get("css")  # type: ignore[assignment]
        
        # 返回当前的 Styler 对象
        return self
    def set_uuid(self, uuid: str) -> Styler:
        """
        Set the uuid applied to ``id`` attributes of HTML elements.

        Parameters
        ----------
        uuid : str
            The unique identifier to be applied to HTML element IDs.

        Returns
        -------
        Styler
            Returns the current Styler instance for method chaining.

        Notes
        -----
        Almost all HTML elements within the table, including the table itself (`<table>`),
        are assigned IDs following the format ``T_uuid_<extra>``, where ``<extra>`` typically
        provides a more specific identifier like ``row1_col2``.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["c1", "c2"])

        You can get the `id` attributes with the following:

        >>> print(df.style.to_html())  # doctest: +SKIP

        To add a title to column `c1`, its `id` would be T_20a7d_level0_col0:

        >>> df.style.set_uuid("T_20a7d_level0_col0").set_caption("Test")
        ... # doctest: +SKIP

        Please see:
        `Table visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        self.uuid = uuid
        return self

    def set_caption(self, caption: str | tuple | list) -> Styler:
        """
        Set the text added to a ``<caption>`` HTML element.

        Parameters
        ----------
        caption : str, tuple, list
            For HTML output, use the string input. For LaTeX, the string input provides
            the main caption, and a tuple can provide full and short captions.

        Returns
        -------
        Styler
            Returns the current Styler instance for method chaining.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.set_caption("test")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        msg = "`caption` must be either a string or 2-tuple of strings."
        if isinstance(caption, (list, tuple)):
            if (
                len(caption) != 2
                or not isinstance(caption[0], str)
                or not isinstance(caption[1], str)
            ):
                raise ValueError(msg)
        elif not isinstance(caption, str):
            raise ValueError(msg)
        self.caption = caption
        return self

    def set_sticky(
        self,
        axis: Axis = 0,
        pixel_size: int | None = None,
        levels: Level | list[Level] | None = None,
    ):
        """
        Set sticky behavior for table headers.

        Parameters
        ----------
        axis : int, default 0
            Axis along which to set sticky behavior.
        pixel_size : int or None, optional
            Size in pixels for the sticky behavior.
        levels : int, list of int, or None, optional
            Level or levels where sticky behavior is applied.

        This method modifies the behavior of table headers to remain visible as
        the table is scrolled.

        Returns
        -------
        None
        """
        # Implementation details are not provided; this method only sets attributes.

    def set_table_styles(
        self,
        table_styles: dict[Any, CSSStyles] | CSSStyles | None = None,
        axis: AxisInt = 0,
        overwrite: bool = True,
        css_class_names: dict[str, str] | None = None,
    ):
        """
        Set custom styles for the table.

        Parameters
        ----------
        table_styles : dict[Any, CSSStyles] or CSSStyles or None, optional
            Custom CSS styles to apply to the table.
        axis : int, default 0
            Axis along which to apply the styles.
        overwrite : bool, default True
            Whether to overwrite existing styles.
        css_class_names : dict[str, str] or None, optional
            Mapping of class names to CSS styles.

        Returns
        -------
        None
        """
        # Implementation details are not provided; this method only sets attributes.

    def hide(
        self,
        subset: Subset | None = None,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        names: bool = False,
    ):
        """
        Hide elements of the styled table.

        Parameters
        ----------
        subset : subset or None, optional
            Subset of elements to hide.
        axis : int, default 0
            Axis along which to hide elements.
        level : int, list of int, or None, optional
            Level or levels where hiding applies.
        names : bool, default False
            Whether to hide elements by name.

        Returns
        -------
        None
        """
        # Implementation details are not provided; this method only sets attributes.
    # -----------------------------------------------------------------------
    # 定义一个方法用于获取默认的数值列子集的布尔掩码。
    # 返回一个布尔掩码，指示 `self.data` 中数值列的位置。
    # 选择掩码而不是列名也适用于布尔列标签（GH47838）。

    def _get_numeric_subset_default(self):
        return self.data.columns.isin(self.data.select_dtypes(include=np.number))

    @doc(
        name="background",
        alt="text",
        image_prefix="bg",
        text_threshold="""text_color_threshold : float or int\n
            Luminance threshold for determining text color in [0, 1]. Facilitates text\n
            visibility across varying background colors. All text is dark if 0, and\n
            light if 1, defaults to 0.408.""",
    )
    @Substitution(subset=subset_args)
    # 定义一个方法用于应用背景渐变样式到数据框，返回样式对象。
    def background_gradient(
        self,
        cmap: str | Colormap = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        text_color_threshold: float = 0.408,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ):
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()
        
        # 应用背景渐变样式函数 `_background_gradient` 到数据框，返回样式对象。
        return self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
            text_only=True,
        )

    @doc(
        background_gradient,
        name="text",
        alt="background",
        image_prefix="tg",
        text_threshold="",
    )
    # 定义一个方法用于应用文本渐变样式到数据框，返回样式对象。
    def text_gradient(
        self,
        cmap: str | Colormap = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ) -> Styler:
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()

        # 应用背景渐变样式函数 `_background_gradient` 到数据框，返回样式对象，只针对文本渐变。
        return self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
            text_only=True,
        )

    @Substitution(subset=subset_args)
    def set_properties(self, subset: Subset | None = None, **kwargs) -> Styler:
        """
        Set defined CSS-properties to each ``<td>`` HTML element for the given subset.

        Parameters
        ----------
        %(subset)s
            Specifies the subset of data to apply the CSS properties to.
        **kwargs : dict
            A dictionary of property, value pairs to be set for each cell.

        Returns
        -------
        Styler
            A Styler object with updated CSS properties.

        Notes
        -----
        This is a convenience method which wraps the :meth:`Styler.map` method, applying
        CSS properties independently of the data.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4))
        >>> df.style.set_properties(color="white", align="right")  # doctest: +SKIP
        >>> df.style.set_properties(**{"background-color": "yellow"})  # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        # Generate a string of CSS property-value pairs
        values = "".join([f"{p}: {v};" for p, v in kwargs.items()])
        # Apply the generated CSS properties using Styler.map method
        return self.map(lambda x: values, subset=subset)

    @Substitution(subset=subset_args)
    def bar(
        self,
        subset: Subset | None = None,
        axis: Axis | None = 0,
        *,
        color: str | list | tuple | None = None,
        cmap: Any | None = None,
        width: float = 100,
        height: float = 100,
        align: str | float | Callable = "mid",
        vmin: float | None = None,
        vmax: float | None = None,
        props: str = "width: 10em;",
    ):
        """
        Create a bar chart representation for the DataFrame.

        Parameters
        ----------
        %(subset)s
            Specifies the subset of data to create the bar chart for.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            The axis to use for plotting.
        color : str, list, tuple, or None, optional
            Color or list of colors for bars.
        cmap : Any, optional
            Colormap to use for mapping the color.
        width : float, default 100
            Width of the bar.
        height : float, default 100
            Height of the bar.
        align : {'mid', 'edge'} or float or callable, default 'mid'
            Alignment of the bars relative to the x coordinates.
        vmin : float or None, optional
            Minimum value to anchor colormap.
        vmax : float or None, optional
            Maximum value to anchor colormap.
        props : str, default 'width: 10em;'
            Additional CSS properties for styling the bar chart.

        Returns
        -------
        None
            The method directly plots the bar chart.

        See Also
        --------
        Styler.barh: Horizontal bar chart representation.
        Styler.scatter: Scatter plot representation.
        Styler.line: Line plot representation.
        Styler.pie: Pie chart representation.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.bar(subset=['A', 'B'], color='lightblue', width=50)  # doctest: +SKIP
        """
        pass  # This method is intended to create a bar chart, but the implementation is not provided.

    @Substitution(
        subset=subset_args,
        props=properties_args,
        color=coloring_args.format(default="red"),
    )
    def highlight_null(
        self,
        color: str = "red",
        subset: Subset | None = None,
        props: str | None = None,
    ) -> Styler:
        """
        Highlight missing values with a specific style.

        Parameters
        ----------
        %(color)s
            Color to be used for highlighting missing values.
        %(subset)s
            Specifies the subset of data to highlight.
        %(props)s
            Additional CSS properties for customizing the highlight style.

        Returns
        -------
        Styler
            A Styler object with applied highlight styles.

        See Also
        --------
        Styler.highlight_max: Highlight the maximum value with a style.
        Styler.highlight_min: Highlight the minimum value with a style.
        Styler.highlight_between: Highlight values within a specified range with a style.
        Styler.highlight_quantile: Highlight values based on quantiles with a style.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, None], "B": [3, 4]})
        >>> df.style.highlight_null(color="yellow")  # doctest: +SKIP

        Please refer to the `Table Visualization <../../user_guide/style.ipynb>`_
        for more examples.
        """
        # Define a function to apply the highlight style based on missing values
        def f(data: DataFrame, props: str) -> np.ndarray:
            return np.where(pd.isna(data).to_numpy(), props, "")

        # If props is not provided, use a default background-color style with the specified color
        if props is None:
            props = f"background-color: {color};"
        # Apply the highlight function to the Styler object using apply method
        return self.apply(f, axis=None, subset=subset, props=props)
    @Substitution(
        subset=subset_args,  # 定义了 subset 参数，用于指定要高亮的子集
        color=coloring_args.format(default="yellow"),  # 定义了 color 参数，默认为黄色
        props=properties_args,  # 定义了 props 参数，用于指定附加的样式属性
    )
    def highlight_max(
        self,
        subset: Subset | None = None,  # subset 参数，指定要高亮的子集，默认为 None
        color: str = "yellow",  # color 参数，指定高亮颜色，默认为黄色
        axis: Axis | None = 0,  # axis 参数，指定操作轴，默认为 0
        props: str | None = None,  # props 参数，指定附加的样式属性，默认为 None
    ) -> Styler:
        """
        Highlight the maximum with a style.

        Parameters
        ----------
        %(subset)s  # subset 参数的描述
        %(color)s  # color 参数的描述
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(props)s  # props 参数的描述

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [2, 1], "B": [3, 4]})
        >>> df.style.highlight_max(color="yellow")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """

        if props is None:  # 如果 props 参数为 None，则设置默认的背景颜色样式
            props = f"background-color: {color};"
        return self.apply(
            partial(_highlight_value, op="max"),  # 使用 partial 函数将 _highlight_value 函数应用到 Styler 上，操作为最大值
            axis=axis,  # 指定操作轴
            subset=subset,  # 指定要操作的子集
            props=props,  # 指定附加的样式属性
        )

    @Substitution(
        subset=subset_args,  # 定义了 subset 参数，用于指定要高亮的子集
        color=coloring_args.format(default="yellow"),  # 定义了 color 参数，默认为黄色
        props=properties_args,  # 定义了 props 参数，用于指定附加的样式属性
    )
    def highlight_min(
        self,
        subset: Subset | None = None,  # subset 参数，指定要高亮的子集，默认为 None
        color: str = "yellow",  # color 参数，指定高亮颜色，默认为黄色
        axis: Axis | None = 0,  # axis 参数，指定操作轴，默认为 0
        props: str | None = None,  # props 参数，指定附加的样式属性，默认为 None
    @Substitution(
        subset=subset_args,
        color=coloring_args.format(default="yellow"),
        props=properties_args,
    )
    # 使用指定的参数来高亮DataFrame中介于指定分位数范围内的值，可设置颜色和其他属性
    def highlight_quantile(
        self,
        subset: Subset | None = None,  # 可选，要高亮的子集
        color: str = "yellow",  # 高亮的颜色，默认为黄色
        axis: Axis | None = 0,  # 操作的轴：0表示按列，1表示按行，None表示整个DataFrame
        q_left: float = 0.0,  # 左分位数，默认为0.0
        q_right: float = 1.0,  # 右分位数，默认为1.0
        interpolation: QuantileInterpolation = "linear",  # 插值方法，默认为线性插值
        inclusive: IntervalClosedType = "both",  # 包含端点的类型，默认为两端都包含
        props: str | None = None,  # 额外的CSS样式属性，如果为None，则使用默认的背景颜色
    ):
        """
        Highlight values defined by a quantile with a style.

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(props)s

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_min: Highlight the minimum with a style.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [2, 1], "B": [3, 4]})
        >>> df.style.highlight_quantile(q_left=0.1, q_right=0.9, color="yellow")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        
        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            partial(_highlight_value, op="quantile", q_left=q_left, q_right=q_right),
            axis=axis,
            subset=subset,
            props=props,
        )
    ) -> type[Styler]:
        """
        Factory function for creating a subclass of ``Styler``.

        Uses custom templates and Jinja environment.

        .. versionchanged:: 1.3.0

        Parameters
        ----------
        searchpath : str or list
            Path or paths of directories containing the templates.
        html_table : str
            Name of your custom template to replace the html_table template.

            .. versionadded:: 1.3.0

        html_style : str
            Name of your custom template to replace the html_style template.

            .. versionadded:: 1.3.0

        Returns
        -------
        MyStyler : subclass of Styler
            Has the correct ``env``,``template_html``, ``template_html_table`` and
            ``template_html_style`` class attributes set.

        Examples
        --------
        >>> from pandas.io.formats.style import Styler
        >>> EasyStyler = Styler.from_custom_template(
        ...     "path/to/template",
        ...     "template.tpl",
        ... )  # doctest: +SKIP
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> EasyStyler(df)  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        loader = jinja2.ChoiceLoader([jinja2.FileSystemLoader(searchpath), cls.loader])

        # mypy doesn't like dynamically-defined classes
        # error: Variable "cls" is not valid as a type
        # error: Invalid base class "cls"
        # 定义一个名为 MyStyler 的内部类，继承自参数传入的 cls 类（通常是 Styler 类）
        class MyStyler(cls):  # type: ignore[valid-type,misc]
            # 创建一个 Jinja2 环境，并使用 ChoiceLoader 加载器
            env = jinja2.Environment(loader=loader)
            # 如果传入了 html_table 参数，则使用环境加载对应的模板
            if html_table:
                template_html_table = env.get_template(html_table)
            # 如果传入了 html_style 参数，则使用环境加载对应的模板
            if html_style:
                template_html_style = env.get_template(html_style)

        # 返回定义好的 MyStyler 类作为工厂函数的结果
        return MyStyler

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    
    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
def _validate_apply_axis_arg(
    arg: NDFrame | Sequence | np.ndarray,
    arg_name: str,
    dtype: Any | None,
    data: NDFrame,
) -> np.ndarray:
    """
    对于应用类型的方法，当 ``axis=None`` 时，``data`` 被创建为 DataFrame；当 ``axis=[1,0]`` 时，创建为 Series。
    在某些操作中，需要确保 ``arg`` 和 ``data`` 具有兼容的形状，否则会抛出异常。

    Parameters
    ----------
    arg : sequence, Series or DataFrame
        用户输入的参数
    arg_name : string
        参数的名称，用于错误消息
    dtype : numpy dtype, optional
        如果提供，强制的 numpy 数据类型
    data : Series or DataFrame
        Styler 数据的子集，对其执行操作

    Returns
    -------
    ndarray
    """
    dtype = {"dtype": dtype} if dtype else {}
    # 如果 arg 是 Series 而 data 是 DataFrame，抛出 ValueError
    if isinstance(arg, Series) and isinstance(data, DataFrame):
        raise ValueError(
            f"'{arg_name}' 是一个 Series，但是操作的基础数据是一个 DataFrame，因为 'axis=None'"
        )
    # 如果 arg 是 DataFrame 而 data 是 Series，抛出 ValueError
    if isinstance(arg, DataFrame) and isinstance(data, Series):
        raise ValueError(
            f"'{arg_name}' 是一个 DataFrame，但是操作的基础数据是一个 Series，因为 'axis in [0,1]'"
        )
    # 如果 arg 是 Series 或 DataFrame，则根据 data 进行索引 / 列对齐，并转换为 numpy 数组
    if isinstance(arg, (Series, DataFrame)):
        arg = arg.reindex_like(data).to_numpy(**dtype)
    else:
        # 否则将 arg 转换为 numpy 数组，并确保其为 np.ndarray 类型
        arg = np.asarray(arg, **dtype)
        assert isinstance(arg, np.ndarray)  # mypy 要求
        # 检查输入是否为有效形状
        if arg.shape != data.shape:
            raise ValueError(
                f"提供的 '{arg_name}' 的形状与所选 'axis' 上的数据不匹配：得到 {arg.shape}，期望 {data.shape}"
            )
    return arg


def _background_gradient(
    data,
    cmap: str | Colormap = "PuBu",
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: float | None = None,
    vmax: float | None = None,
    gmap: Sequence | np.ndarray | DataFrame | Series | None = None,
    text_only: bool = False,
) -> list[str] | DataFrame:
    """
    根据数据或渐变映射为范围内的背景着色
    """
    # 如果 gmap 为 None，则使用数据创建渐变映射
    if gmap is None:
        gmap = data.to_numpy(dtype=float, na_value=np.nan)
    else:
        # 否则验证 gmap 是否与基础数据兼容
        gmap = _validate_apply_axis_arg(gmap, "gmap", float, data)

    # 计算数据的最小值和最大值
    smin = np.nanmin(gmap) if vmin is None else vmin
    smax = np.nanmax(gmap) if vmax is None else vmax
    rng = smax - smin
    _matplotlib = import_optional_dependency(
        "matplotlib", extra="Styler.background_gradient requires matplotlib."
    )
    # 创建正规化对象，用于颜色映射
    norm = _matplotlib.colors.Normalize(smin - (rng * low), smax + (rng * high))
    # 如果未提供颜色映射，则使用默认的 matplotlib 配置中的颜色映射来生成 RGBA 数组
    if cmap is None:
        rgbas = _matplotlib.colormaps[_matplotlib.rcParams["image.cmap"]](norm(gmap))
    else:
        # 使用指定的颜色映射来生成 RGBA 数组
        rgbas = _matplotlib.colormaps.get_cmap(cmap)(norm(gmap))

    # 定义一个函数，计算给定颜色的相对亮度
    def relative_luminance(rgba) -> float:
        """
        Calculate relative luminance of a color.

        The calculation adheres to the W3C standards
        (https://www.w3.org/WAI/GL/wiki/Relative_luminance)

        Parameters
        ----------
        rgba : tuple
            RGB or RGBA tuple representing a color

        Returns
        -------
        float
            The relative luminance as a value from 0 to 1
        """
        # 解构 RGB(A) 元组，根据公式计算相对亮度
        r, g, b = (
            x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
            for x in rgba[:3]
        )
        # 返回相对亮度值
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    # 定义一个函数，生成 CSS 样式字符串
    def css(rgba, text_only) -> str:
        """
        Generate CSS style string for a given color.

        Parameters
        ----------
        rgba : tuple
            RGB or RGBA tuple representing a color
        text_only : bool
            Flag indicating whether to generate only text color CSS

        Returns
        -------
        str
            CSS style string
        """
        if not text_only:
            # 计算颜色的相对亮度，根据阈值决定文本颜色
            dark = relative_luminance(rgba) < text_color_threshold
            text_color = "#f1f1f1" if dark else "#000000"
            # 返回包含背景颜色和文本颜色的 CSS 样式字符串
            return (
                f"background-color: {_matplotlib.colors.rgb2hex(rgba)};"
                f"color: {text_color};"
            )
        else:
            # 返回只包含文本颜色的 CSS 样式字符串
            return f"color: {_matplotlib.colors.rgb2hex(rgba)};"

    # 如果数据是一维的，返回每个颜色对应的 CSS 样式字符串组成的列表
    if data.ndim == 1:
        return [css(rgba, text_only) for rgba in rgbas]
    else:
        # 如果数据是二维的，返回一个 DataFrame，其中每个单元格包含颜色对应的 CSS 样式字符串
        return DataFrame(
            [[css(rgba, text_only) for rgba in row] for row in rgbas],
            index=data.index,
            columns=data.columns,
        )
# 生成 CSS 样式字符串的数组，根据数据值在给定范围内的条件
def _highlight_between(
    data: NDFrame,  # 数据对象，可以是 DataFrame 或 Series
    props: str,  # CSS 样式属性字符串
    left: Scalar | Sequence | np.ndarray | NDFrame | None = None,  # 左边界值，可以是标量、序列或数组，默认为 None
    right: Scalar | Sequence | np.ndarray | NDFrame | None = None,  # 右边界值，可以是标量、序列或数组，默认为 None
    inclusive: bool | str = True,  # 是否包含边界值，可以是布尔值或字符串，默认为 True
) -> np.ndarray:
    """
    Return an array of css props based on condition of data values within given range.
    根据数据值在给定范围内的条件返回一个 CSS 样式属性数组。
    """
    if np.iterable(left) and not isinstance(left, str):
        # 如果 left 是可迭代对象且不是字符串，则验证并应用左边界参数
        left = _validate_apply_axis_arg(left, "left", None, data)

    if np.iterable(right) and not isinstance(right, str):
        # 如果 right 是可迭代对象且不是字符串，则验证并应用右边界参数
        right = _validate_apply_axis_arg(right, "right", None, data)

    # 根据 inclusive 参数获取正确的比较操作函数对
    if inclusive == "both":
        ops = (operator.ge, operator.le)  # 包含两个边界
    elif inclusive == "neither":
        ops = (operator.gt, operator.lt)  # 不包含任何边界
    elif inclusive == "left":
        ops = (operator.ge, operator.lt)  # 包含左边界，不包含右边界
    elif inclusive == "right":
        ops = (operator.gt, operator.le)  # 包含右边界，不包含左边界
    else:
        raise ValueError(
            f"'inclusive' values can be 'both', 'left', 'right', or 'neither' "
            f"got {inclusive}"
        )

    g_left = (
        ops[0](data, left)  # type: ignore[arg-type] 根据 ops[0] 执行比较操作，如果 left 不为 None
        if left is not None
        else np.full(data.shape, True, dtype=bool)  # 否则返回一个全为 True 的布尔数组
    )
    if isinstance(g_left, (DataFrame, Series)):
        g_left = g_left.where(pd.notna(g_left), False)  # 将 NaN 替换为 False

    l_right = (
        ops[1](data, right)  # type: ignore[arg-type] 根据 ops[1] 执行比较操作，如果 right 不为 None
        if right is not None
        else np.full(data.shape, True, dtype=bool)  # 否则返回一个全为 True 的布尔数组
    )
    if isinstance(l_right, (DataFrame, Series)):
        l_right = l_right.where(pd.notna(l_right), False)  # 将 NaN 替换为 False

    return np.where(g_left & l_right, props, "")  # 根据条件 g_left 和 l_right 返回 props 或空字符串的数组
    """
    Draw bar chart in data cells using HTML CSS linear gradient.

    Parameters
    ----------
    data : Series or DataFrame
        Underlying subset of Styler data on which operations are performed.
    align : str in {"left", "right", "mid", "zero", "mean"}, int, float, callable
        Method for how bars are structured or scalar value of center point.
    colors : list-like of str
        Two listed colors as string in valid CSS.
    width : float in [0,1]
        The percentage of the cell, measured from left, where drawn bars will reside.
    height : float in [0,1]
        The percentage of the cell's height where drawn bars will reside, centrally
        aligned.
    vmin : float, optional
        Overwrite the minimum value of the window.
    vmax : float, optional
        Overwrite the maximum value of the window.
    base_css : str
        Additional CSS that is included in the cell before bars are drawn.
    """

    def css_bar(start: float, end: float, color: str) -> str:
        """
        Generate CSS code to draw a bar from start to end in a table cell.

        Uses linear-gradient.

        Parameters
        ----------
        start : float
            Relative positional start of bar coloring in [0,1]
        end : float
            Relative positional end of the bar coloring in [0,1]
        color : str
            CSS valid color to apply.

        Returns
        -------
        str : The CSS applicable to the cell.

        Notes
        -----
        Uses `base_css` from outer scope.
        """
        # 将外部作用域的 `base_css` 赋值给本地变量 `cell_css`
        cell_css = base_css
        # 如果结束位置大于开始位置
        if end > start:
            # 拼接 CSS 字符串，使用线性渐变进行背景绘制
            cell_css += "background: linear-gradient(90deg,"
            # 如果起始位置大于 0，则添加透明背景色段和实际颜色段
            if start > 0:
                cell_css += f" transparent {start*100:.1f}%, {color} {start*100:.1f}%,"
            # 添加实际颜色段和透明背景色段
            cell_css += f" {color} {end*100:.1f}%, transparent {end*100:.1f}%)"
        # 返回生成的 CSS 字符串
        return cell_css
    def css_calc(x, left: float, right: float, align: str, color: str | list | tuple):
        """
        Return the correct CSS for bar placement based on calculated values.
    
        Parameters
        ----------
        x : float
            Value which determines the bar placement.
        left : float
            Value marking the left side of calculation, usually minimum of data.
        right : float
            Value marking the right side of the calculation, usually maximum of data
            (left < right).
        align : {"left", "right", "zero", "mid"}
            How the bars will be positioned.
            "left", "right", "zero" can be used with any values for ``left``, ``right``.
            "mid" can only be used where ``left <= 0`` and ``right >= 0``.
            "zero" is used to specify a center when all values ``x``, ``left``,
            ``right`` are translated, e.g. by say a mean or median.
    
        Returns
        -------
        str : Resultant CSS with linear gradient.
    
        Notes
        -----
        Uses ``colors``, ``width`` and ``height`` from outer scope.
        """
        if pd.isna(x):
            return base_css  # 返回预设的基本 CSS 样式
    
        if isinstance(color, (list, tuple)):
            color = color[0] if x < 0 else color[1]  # 根据 x 的正负选择颜色
        assert isinstance(color, str)  # 确保 color 是字符串类型，mypy 的重新定义
    
        x = left if x < left else x  # 如果 x 小于 left，则使用 left
        x = right if x > right else x  # 如果 x 大于 right，则使用 right，修剪数据以适应窗口范围
    
        start: float = 0  # 初始化起始位置
        end: float = 1  # 初始化结束位置
    
        if align == "left":
            # 所有比例从左侧开始测量，位于 left 和 right 之间
            end = (x - left) / (right - left)
    
        elif align == "right":
            # 所有比例从右侧开始测量，位于 left 和 right 之间
            start = (x - left) / (right - left)
    
        else:
            z_frac: float = 0.5  # 在 left-right 范围内基于零的位置比例
            if align == "zero":
                # 所有比例从零点中心开始测量
                limit: float = max(abs(left), abs(right))
                left, right = -limit, limit  # 将 left 和 right 重置为对称范围
            elif align == "mid":
                # 从零点开始向左或向右绘制柱状图，中心在中点
                mid: float = (left + right) / 2
                z_frac = (
                    -mid / (right - left) + 0.5 if mid < 0 else -left / (right - left)
                )
    
            if x < 0:
                start, end = (x - left) / (right - left), z_frac
            else:
                start, end = z_frac, (x - left) / (right - left)
    
        # 调用 css_bar 函数生成 CSS 样式，并根据 height 调整背景大小
        ret = css_bar(start * width, end * width, color)
        if height < 1 and "background: linear-gradient(" in ret:
            return (
                ret + f" no-repeat center; background-size: 100% {height * 100:.1f}%;"
            )
        else:
            return ret
    
    values = data.to_numpy()
    # 处理 np.nanmin/np.nanmax 无法处理 pd.NA 的问题的巧妙方法
    left = np.nanmin(data.min(skipna=True)) if vmin is None else vmin
    # 如果未指定最小值（vmin），则计算数据中的最小值，忽略NaN值，否则使用指定的vmin作为最小值
    right = np.nanmax(data.max(skipna=True)) if vmax is None else vmax
    # 如果未指定最大值（vmax），则计算数据中的最大值，忽略NaN值，否则使用指定的vmax作为最大值
    z: float = 0  # 调整数据的偏移量，用于数据平移

    if align == "mid":
        # 如果对齐方式为"mid"
        if left >= 0:
            # 如果最小值大于等于0，"mid"的行为将等同于"left"
            align, left = "left", 0 if vmin is None else vmin
        elif right <= 0:
            # 如果最大值小于等于0，"mid"的行为将等同于"right"
            align, right = "right", 0 if vmax is None else vmax
    elif align == "mean":
        # 如果对齐方式为"mean"
        z, align = np.nanmean(values), "zero"
        # z取数据的均值，对齐方式设为"zero"
    elif callable(align):
        # 如果对齐方式是一个可调用对象
        z, align = align(values), "zero"
        # z取调用对象的返回值，对齐方式设为"zero"
    elif isinstance(align, (float, int)):
        # 如果对齐方式是浮点数或整数
        z, align = float(align), "zero"
        # z取对齐方式，设为"zero"
    elif align not in ("left", "right", "zero"):
        # 如果对齐方式不在预期的值中
        raise ValueError(
            "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or be a "
            "value defining the center line or a callable that returns a float"
        )
        # 抛出数值错误，说明对齐方式应为指定的几种字符串或者定义中心线的值，或者可返回浮点数的可调用对象

    rgbas = None
    if cmap is not None:
        # 如果指定了色彩映射
        # 导入matplotlib库
        _matplotlib = import_optional_dependency(
            "matplotlib", extra="Styler.bar requires matplotlib."
        )
        cmap = (
            _matplotlib.colormaps[cmap]
            if isinstance(cmap, str)
            else cmap  # 假定为Colormap实例，如文档所述
        )
        norm = _matplotlib.colors.Normalize(left, right)
        # 根据指定的最小值和最大值创建归一化对象
        rgbas = cmap(norm(values))
        # 使用归一化后的数据应用色彩映射
        if data.ndim == 1:
            rgbas = [_matplotlib.colors.rgb2hex(rgba) for rgba in rgbas]
            # 如果数据是一维的，转换颜色表示为十六进制字符串
        else:
            rgbas = [
                [_matplotlib.colors.rgb2hex(rgba) for rgba in row] for row in rgbas
            ]
            # 如果数据是二维的，转换每行的颜色表示为十六进制字符串列表

    assert isinstance(align, str)  # mypy: should now be in [left, right, mid, zero]
    # 断言对齐方式现在应为左、右、中或零中的一种字符串

    if data.ndim == 1:
        # 如果数据是一维的
        return [
            css_calc(
                x - z, left - z, right - z, align, colors if rgbas is None else rgbas[i]
            )
            for i, x in enumerate(values)
        ]
        # 返回每个值的CSS计算结果，用于样式化单个条目
    else:
        # 如果数据是二维的
        return np.array(
            [
                [
                    css_calc(
                        x - z,
                        left - z,
                        right - z,
                        align,
                        colors if rgbas is None else rgbas[i][j],
                    )
                    for j, x in enumerate(row)
                ]
                for i, row in enumerate(values)
            ]
        )
        # 返回二维数组，其中每个元素都经过CSS计算，用于样式化多个条目
```