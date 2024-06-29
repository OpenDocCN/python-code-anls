# `D:\src\scipysrc\pandas\pandas\io\formats\excel.py`

```
    def __init__(
        self,
        row: int,
        col: int,
        val,
        style: dict | None,
        css_styles: dict[tuple[int, int], list[tuple[str, Any]]] | None,
        css_row: int,
        css_col: int,
        css_converter: Callable | None,
        **kwargs,
    ) -> None:
        if css_styles and css_converter:
            # 如果存在 CSS 样式和转换器，则从 CSS 样式中提取唯一的声明，并转换为 frozenset 进行缓存
            declaration_dict = {
                prop.lower(): val for prop, val in css_styles[css_row, css_col]
            }
            unique_declarations = frozenset(declaration_dict.items())
            # 使用 CSS 转换器将唯一的声明转换为 Excel 样式
            style = css_converter(unique_declarations)

        # 调用父类的构造函数初始化 ExcelCell 对象
        super().__init__(row=row, col=col, val=val, style=style, **kwargs)


这段代码是一个子类 `CssExcelCell` 的构造函数。它继承自 `ExcelCell` 类，用于表示Excel表格中的单元格，包含行号、列号、值、样式等属性。在初始化时，根据传入的 CSS 样式和转换器，将 CSS 样式中的属性转换为适用于 Excel 的样式。
    # CSSResolver class used for computing CSS styles
    compute_css = CSSResolver()

    # Constructor initializing the object with optional inherited CSS declarations
    def __init__(self, inherited: str | None = None) -> None:
        # If inherited CSS is provided, compute its resolved form
        if inherited is not None:
            self.inherited = self.compute_css(inherited)
        else:
            self.inherited = None
        # functools.cache decorator applied to _call_uncached to cache results
        # Avoids unnecessary recalculations when __call__ is invoked with the same input
        self._call_cached = functools.cache(self._call_uncached)

    def __call__(
        self, declarations: str | frozenset[tuple[str, str]]
    ) -> dict[str, dict[str, str]]:
        """
        Convert CSS declarations to ExcelWriter style.

        Parameters
        ----------
        declarations : str | frozenset[tuple[str, str]]
            CSS string or set of CSS declaration tuples.
            e.g. "font-weight: bold; background: blue" or
            {("font-weight", "bold"), ("background", "blue")}

        Returns
        -------
        xlstyle : dict
            A style as interpreted by ExcelWriter when found in
            ExcelCell.style.
        """
        # Calls _call_cached which internally resolves CSS declarations to ExcelWriter style
        return self._call_cached(declarations)

    def _call_uncached(
        self, declarations: str | frozenset[tuple[str, str]]
    ) -> dict[str, dict[str, str]]:
        """
        Helper method to convert CSS declarations to ExcelWriter style.

        Parameters and Returns are the same as in __call__.
        """
        # Placeholder for the actual implementation of converting CSS to ExcelWriter style
        pass
    ) -> dict[str, dict[str, str]]:
        properties = self.compute_css(declarations, self.inherited)
        return self.build_xlstyle(properties)

    def build_xlstyle(self, props: Mapping[str, str]) -> dict[str, dict[str, str]]:
        out = {
            "alignment": self.build_alignment(props),  # 构建样式的对齐部分
            "border": self.build_border(props),        # 构建样式的边框部分
            "fill": self.build_fill(props),            # 构建样式的填充部分
            "font": self.build_font(props),            # 构建样式的字体部分
            "number_format": self.build_number_format(props),  # 构建样式的数字格式部分
        }

        # TODO: 处理单元格的宽度和高度：需要在 pandas.io.excel 中添加支持

        def remove_none(d: dict[str, str | None]) -> None:
            """递归移除值为 None 的键"""
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, dict):
                    remove_none(v)
                    if not v:
                        del d[k]

        remove_none(out)  # 移除 out 字典中值为 None 的键
        return out

    def build_alignment(self, props: Mapping[str, str]) -> dict[str, bool | str | None]:
        # TODO: 处理文本缩进和左侧内边距 -> alignment.indent
        return {
            "horizontal": props.get("text-align"),                  # 水平对齐方式
            "vertical": self._get_vertical_alignment(props),        # 垂直对齐方式
            "wrap_text": self._get_is_wrap_text(props),             # 文本是否自动换行
        }

    def _get_vertical_alignment(self, props: Mapping[str, str]) -> str | None:
        vertical_align = props.get("vertical-align")
        if vertical_align:
            return self.VERTICAL_MAP.get(vertical_align)  # 获取垂直对齐方式的映射值
        return None

    def _get_is_wrap_text(self, props: Mapping[str, str]) -> bool | None:
        if props.get("white-space") is None:
            return None
        return bool(props["white-space"] not in ("nowrap", "pre", "pre-line"))  # 检查文本是否需要自动换行

    def build_border(
        self, props: Mapping[str, str]
    ) -> dict[str, dict[str, str | None]]:
        return {
            side: {
                "style": self._border_style(                         # 构建边框样式
                    props.get(f"border-{side}-style"),
                    props.get(f"border-{side}-width"),
                    self.color_to_excel(props.get(f"border-{side}-color")),
                ),
                "color": self.color_to_excel(props.get(f"border-{side}-color")),  # 边框颜色
            }
            for side in ["top", "right", "bottom", "left"]  # 遍历四个边框：上、右、下、左
        }

    def _border_style(
        self, style: str | None, width: str | None, color: str | None
    ) -> str | None:
    ) -> str | None:
        # 将样式和宽度转换为 OpenXML 格式，可能的取值包括：
        #       'dashDot'
        #       'dashDotDot'
        #       'dashed'
        #       'dotted'
        #       'double'
        #       'hair'
        #       'medium'
        #       'mediumDashDot'
        #       'mediumDashDotDot'
        #       'mediumDashed'
        #       'slantDashDot'
        #       'thick'
        #       'thin'
        if width is None and style is None and color is None:
            # 如果宽度、样式和颜色都为 None，则移除样式字典中的 "border"
            return None

        if width is None and style is None:
            # 如果宽度和样式都为 None，则保留样式字典中的 "border"
            return "none"

        if style in ("none", "hidden"):
            # 如果样式为 "none" 或 "hidden"，则返回 "none"
            return "none"

        width_name = self._get_width_name(width)
        if width_name is None:
            # 如果宽度名为 None，则返回 "none"
            return "none"

        if style in (None, "groove", "ridge", "inset", "outset", "solid"):
            # 如果样式为 None 或者是一些未处理的样式，返回宽度名
            return width_name

        if style == "double":
            # 如果样式为 "double"，返回 "double"
            return "double"
        if style == "dotted":
            if width_name in ("hair", "thin"):
                # 如果宽度名为 "hair" 或 "thin"，返回 "dotted"
                return "dotted"
            return "mediumDashDotDot"
        if style == "dashed":
            if width_name in ("hair", "thin"):
                # 如果宽度名为 "hair" 或 "thin"，返回 "dashed"
                return "dashed"
            return "mediumDashed"
        elif style in self.BORDER_STYLE_MAP:
            # 如果样式在 BORDER_STYLE_MAP 中定义（Excel 特定样式），返回对应值
            return self.BORDER_STYLE_MAP[style]
        else:
            warnings.warn(
                f"Unhandled border style format: {style!r}",
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            # 对于未处理的样式，返回 "none"
            return "none"

    def _get_width_name(self, width_input: str | None) -> str | None:
        # 将宽度字符串转换为对应的宽度名称
        width = self._width_to_float(width_input)
        if width < 1e-5:
            return None
        elif width < 1.3:
            return "thin"
        elif width < 2.8:
            return "medium"
        return "thick"

    def _width_to_float(self, width: str | None) -> float:
        # 将宽度字符串转换为浮点数
        if width is None:
            width = "2pt"
        return self._pt_to_float(width)

    def _pt_to_float(self, pt_string: str) -> float:
        # 将表示以 pt 结尾的字符串转换为浮点数
        assert pt_string.endswith("pt")
        return float(pt_string.rstrip("pt"))

    def build_fill(self, props: Mapping[str, str]):
        # 构建填充样式
        # TODO: 可能允许特殊属性如 -excel-pattern-bgcolor 和 -excel-pattern-type
        fill_color = props.get("background-color")
        if fill_color not in (None, "transparent", "none"):
            # 如果背景颜色不为 None、"transparent" 或 "none"，返回实体颜色和实体类型为 "solid" 的字典
            return {"fgColor": self.color_to_excel(fill_color), "patternType": "solid"}

    def build_number_format(self, props: Mapping[str, str]) -> dict[str, str | None]:
        # 构建数字格式
        fc = props.get("number-format")
        fc = fc.replace("§", ";") if isinstance(fc, str) else fc
        return {"format_code": fc}

    def build_font(
        self, props: Mapping[str, str]
    ) -> dict[str, bool | float | str | None]:
        # 获取字体名称列表
        font_names = self._get_font_names(props)
        # 获取文本装饰风格
        decoration = self._get_decoration(props)
        # 构建并返回包含字体属性的字典
        return {
            "name": font_names[0] if font_names else None,  # 获取第一个字体名称或返回 None
            "family": self._select_font_family(font_names),  # 选择字体家族
            "size": self._get_font_size(props),  # 获取字体大小
            "bold": self._get_is_bold(props),  # 检查是否为粗体
            "italic": self._get_is_italic(props),  # 检查是否为斜体
            "underline": ("single" if "underline" in decoration else None),  # 检查是否有下划线装饰
            "strike": ("line-through" in decoration) or None,  # 检查是否有删除线装饰
            "color": self.color_to_excel(props.get("color")),  # 将颜色转换为 Excel 兼容格式
            # 根据阴影效果设置值，如果阴影非零则为 True，否则为 None
            "shadow": self._get_shadow(props),
        }

    def _get_is_bold(self, props: Mapping[str, str]) -> bool | None:
        # 获取字体粗细信息
        weight = props.get("font-weight")
        if weight:
            return self.BOLD_MAP.get(weight)
        return None

    def _get_is_italic(self, props: Mapping[str, str]) -> bool | None:
        # 获取字体是否为斜体信息
        font_style = props.get("font-style")
        if font_style:
            return self.ITALIC_MAP.get(font_style)
        return None

    def _get_decoration(self, props: Mapping[str, str]) -> Sequence[str]:
        # 获取文本装饰信息列表
        decoration = props.get("text-decoration")
        if decoration is not None:
            return decoration.split()
        else:
            return ()

    def _get_underline(self, decoration: Sequence[str]) -> str | None:
        # 检查是否有下划线装饰
        if "underline" in decoration:
            return "single"
        return None

    def _get_shadow(self, props: Mapping[str, str]) -> bool | None:
        # 检查是否有文本阴影效果
        if "text-shadow" in props:
            return bool(re.search("^[^#(]*[1-9]", props["text-shadow"]))
        return None

    def _get_font_names(self, props: Mapping[str, str]) -> Sequence[str]:
        # 从样式属性中获取字体名称列表
        font_names_tmp = re.findall(
            r"""(?x)
            (
            "(?:[^"]|\\")+"
            |
            '(?:[^']|\\')+'
            |
            [^'",]+
            )(?=,|\s*$)
        """,
            props.get("font-family", ""),
        )

        font_names = []
        # 处理获取到的字体名称
        for name in font_names_tmp:
            if name[:1] == '"':
                name = name[1:-1].replace('\\"', '"')
            elif name[:1] == "'":
                name = name[1:-1].replace("\\'", "'")
            else:
                name = name.strip()
            if name:
                font_names.append(name)
        return font_names

    def _get_font_size(self, props: Mapping[str, str]) -> float | None:
        # 获取字体大小信息
        size = props.get("font-size")
        if size is None:
            return size
        return self._pt_to_float(size)

    def _select_font_family(self, font_names: Sequence[str]) -> int | None:
        # 选择适合的字体家族
        family = None
        for name in font_names:
            family = self.FAMILY_MAP.get(name)
            if family:
                break

        return family
    # 将颜色字符串转换成 Excel 格式的颜色代码，如果输入为 None 则返回 None
    def color_to_excel(self, val: str | None) -> str | None:
        # 如果输入值为 None，则直接返回 None
        if val is None:
            return None
        
        # 如果颜色字符串以 '#' 开头，则认为是十六进制颜色，需要转换成 Excel 格式
        if self._is_hex_color(val):
            return self._convert_hex_to_excel(val)
        
        try:
            # 尝试从预定义的颜色名称映射中获取对应的 Excel 颜色代码
            return self.NAMED_COLORS[val]
        except KeyError:
            # 如果颜色不在预定义列表中，则发出警告，并指明警告的具体内容和级别
            warnings.warn(
                f"Unhandled color format: {val!r}",
                CSSWarning,
                stacklevel=find_stack_level(),
            )
        # 对于无法处理的颜色格式，返回 None
        return None

    # 检查颜色字符串是否是十六进制表示
    def _is_hex_color(self, color_string: str) -> bool:
        return bool(color_string.startswith("#"))

    # 将十六进制颜色字符串转换成 Excel 格式的颜色代码
    def _convert_hex_to_excel(self, color_string: str) -> str:
        # 去除颜色字符串开头的 '#' 符号
        code = color_string.lstrip("#")
        # 如果是简写形式的颜色（例如 #RGB），则将其扩展成全格式（例如 #RRGGBB）
        if self._is_shorthand_color(color_string):
            return (code[0] * 2 + code[1] * 2 + code[2] * 2).upper()
        else:
            # 如果是全格式的十六进制颜色，则直接转换为大写形式返回
            return code.upper()

    # 检查颜色字符串是否是简写形式（三位十六进制表示）
    def _is_shorthand_color(self, color_string: str) -> bool:
        """Check if color code is shorthand.

        #FFF is a shorthand as opposed to full #FFFFFF.
        """
        # 去除颜色字符串开头的 '#' 符号
        code = color_string.lstrip("#")
        # 如果颜色代码长度为 3，则认为是简写形式
        if len(code) == 3:
            return True
        # 如果颜色代码长度为 6，则认为是全格式
        elif len(code) == 6:
            return False
        else:
            # 如果长度既不是 3 也不是 6，则抛出异常，说明颜色格式不符合预期
            raise ValueError(f"Unexpected color {color_string}")
    """
    Class for formatting a DataFrame to a list of ExcelCells,

    Parameters
    ----------
    df : DataFrame or Styler
        要格式化的 DataFrame 或 Styler 对象
    na_rep: na representation
        缺失值的表示方式
    float_format : str, default None
        浮点数的格式化字符串
    cols : sequence, optional
        要写入的列名序列
    header : bool or sequence of str, default True
        是否写入列名。如果给定字符串序列，假定为列名的别名
    index : bool, default True
        是否输出行名（索引）
    index_label : str or sequence, default None
        索引列的列标签。如果为 None，并且 `header` 和 `index` 均为 True，则使用索引名称。
        如果 DataFrame 使用 MultiIndex，则应给出一个序列。
    merge_cells : bool or 'columns', default False
        如果为 True，则格式化 MultiIndex 列标题和层次行为合并单元格。
        如果为 'columns'，则仅合并 MultiIndex 列标题。
        .. versionchanged:: 3.0.0
            添加了 'columns' 选项。
    inf_rep : str, default `'inf'`
        表示 Excel 中无法表示的 np.inf 值的字符串表示方式。
        对于 -inf，将在前面添加一个 '-' 符号。
    style_converter : callable, optional
        将 Styler 样式（CSS）转换为 ExcelWriter 样式的可调用函数。
        默认为 ``CSSToExcelConverter()``。
        其签名应为 css_declarations string -> excel style。
        仅对正文单元格调用此函数。
    """

    max_rows = 2**20
    max_cols = 2**14

    def __init__(
        self,
        df,
        na_rep: str = "",
        float_format: str | None = None,
        cols: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        merge_cells: ExcelWriterMergeCells = False,
        inf_rep: str = "inf",
        style_converter: Callable | None = None,
    ):
        """
        初始化方法，用于设置 Excel 格式化器的参数。

        Parameters
        ----------
        df : DataFrame or Styler
            要格式化的 DataFrame 或 Styler 对象
        na_rep: str, optional
            缺失值的表示方式，默认为空字符串
        float_format : str, optional
            浮点数的格式化字符串，默认为 None
        cols : sequence, optional
            要写入的列名序列，默认为 None
        header : bool or sequence of str, default True
            是否写入列名。如果给定字符串序列，假定为列名的别名
        index : bool, optional
            是否输出行名（索引），默认为 True
        index_label : str or sequence, optional
            索引列的列标签。如果为 None，并且 `header` 和 `index` 均为 True，则使用索引名称。
            如果 DataFrame 使用 MultiIndex，则应给出一个序列。
        merge_cells : bool or 'columns', optional
            是否合并单元格以格式化 MultiIndex 列标题和层次行，默认为 False。
            如果为 'columns'，则仅合并 MultiIndex 列标题。
            .. versionchanged:: 3.0.0
                添加了 'columns' 选项。
        inf_rep : str, optional
            表示 Excel 中无法表示的 np.inf 值的字符串表示方式，默认为 "inf"。
            对于 -inf，将在前面添加一个 '-' 符号。
        style_converter : callable, optional
            将 Styler 样式（CSS）转换为 ExcelWriter 样式的可调用函数。
            默认为 ``CSSToExcelConverter()``。
            其签名应为 css_declarations string -> excel style。
            仅对正文单元格调用此函数。
        """
    ) -> None:
        # 初始化行计数器
        self.rowcounter = 0
        # 设置缺失值的替代符号
        self.na_rep = na_rep
        # 如果传入的 df 不是 DataFrame 对象，则假设传入的是 Styler 对象
        if not isinstance(df, DataFrame):
            # 将传入的 Styler 对象赋值给 self.styler
            self.styler = df
            # 计算应用的样式
            self.styler._compute()  # calculate applied styles
            # 获取 Styler 对象的数据部分
            df = df.data
            # 如果未提供 style_converter，则默认使用 CSSToExcelConverter()
            if style_converter is None:
                style_converter = CSSToExcelConverter()
            # 设置 style_converter 属性为传入的值
            self.style_converter: Callable | None = style_converter
        else:
            # 如果传入的是 DataFrame 对象，则设置 self.styler 为 None
            self.styler = None
            # 设置 style_converter 为 None
            self.style_converter = None
        # 将处理后的 df 赋值给实例变量 self.df
        self.df = df
        # 如果指定了列名 cols
        if cols is not None:
            # 检查所有指定列名是否都存在于 DataFrame 的列中
            if not len(Index(cols).intersection(df.columns)):
                # 如果有列名不存在于 DataFrame 中，则抛出 KeyError
                raise KeyError("passes columns are not ALL present dataframe")

            # 检查指定列名的个数是否与集合长度相等
            if len(Index(cols).intersection(df.columns)) != len(set(cols)):
                # 如果不相等，则抛出 KeyError
                raise KeyError("Not all names specified in 'columns' are found")

            # 重新索引 DataFrame，仅保留指定的列
            self.df = df.reindex(columns=cols)

        # 将 DataFrame 的列名赋值给实例变量 self.columns
        self.columns = self.df.columns
        # 设置浮点数格式化的格式
        self.float_format = float_format
        # 设置是否包含索引
        self.index = index
        # 设置索引标签
        self.index_label = index_label
        # 设置是否包含表头
        self.header = header

        # 检查 merge_cells 是否为布尔值或者 "columns"
        if not isinstance(merge_cells, bool) and merge_cells != "columns":
            # 如果不是合法的值，则抛出 ValueError
            raise ValueError(f"Unexpected value for {merge_cells=}.")
        # 设置是否合并单元格的选项
        self.merge_cells = merge_cells
        # 设置无穷大替代符号
        self.inf_rep = inf_rep

    def _format_value(self, val):
        # 如果 val 是标量且是缺失值
        if is_scalar(val) and missing.isna(val):
            # 使用指定的缺失值替代符号
            val = self.na_rep
        # 如果 val 是浮点数
        elif is_float(val):
            # 如果 val 是正无穷大
            if missing.isposinf_scalar(val):
                # 使用正无穷大替代符号
                val = self.inf_rep
            # 如果 val 是负无穷大
            elif missing.isneginf_scalar(val):
                # 使用负无穷大替代符号
                val = f"-{self.inf_rep}"
            # 如果设置了浮点数格式化的格式
            elif self.float_format is not None:
                # 格式化浮点数的值
                val = float(self.float_format % val)
        # 如果 val 具有 tzinfo 属性（即包含时区信息）
        if getattr(val, "tzinfo", None) is not None:
            # 抛出 ValueError，因为 Excel 不支持带有时区的日期时间
            raise ValueError(
                "Excel does not support datetimes with "
                "timezones. Please ensure that datetimes "
                "are timezone unaware before writing to Excel."
            )
        # 返回格式化后的值
        return val
    # 定义一个方法 `_format_header_mi`，返回一个 Excel 单元格的可迭代对象
    def _format_header_mi(self) -> Iterable[ExcelCell]:
        # 检查列的层级数是否大于1
        if self.columns.nlevels > 1:
            # 如果没有设置索引且写入 Excel 文件，则抛出未实现错误
            if not self.index:
                raise NotImplementedError(
                    "Writing to Excel with MultiIndex columns and no "
                    "index ('index'=False) is not yet implemented."
                )

        # 如果没有设置别名或者未定义表头，则直接返回
        if not (self._has_aliases or self.header):
            return

        # 获取列对象
        columns = self.columns
        # 格式化多级索引的字符串表示形式，根据需要合并单元格
        level_strs = columns._format_multi(
            sparsify=self.merge_cells in {True, "columns"}, include_names=False
        )
        # 获取每个级别字符串的长度信息
        level_lengths = get_level_lengths(level_strs)
        # 列偏移量初始化为0，行号初始化为0
        coloffset = 0
        lnum = 0

        # 如果设置了索引且索引是 MultiIndex 类型
        if self.index and isinstance(self.df.index, MultiIndex):
            coloffset = self.df.index.nlevels - 1

        # 如果需要合并单元格
        if self.merge_cells in {True, "columns"}:
            # 将多级索引格式化为合并单元格
            for lnum, name in enumerate(columns.names):
                yield ExcelCell(
                    row=lnum,
                    col=coloffset,
                    val=name,
                    style=None,
                )

            # 遍历级别长度、级别值和级别代码的组合
            for lnum, (spans, levels, level_codes) in enumerate(
                zip(level_lengths, columns.levels, columns.codes)
            ):
                # 取出级别对应的值
                values = levels.take(level_codes)
                # 遍历每个值和其对应的跨度
                for i, span_val in spans.items():
                    # 计算合并单元格的起始和结束位置
                    mergestart, mergeend = None, None
                    if span_val > 1:
                        mergestart, mergeend = lnum, coloffset + i + span_val
                    yield CssExcelCell(
                        row=lnum,
                        col=coloffset + i + 1,
                        val=values[i],
                        style=None,
                        css_styles=getattr(self.styler, "ctx_columns", None),
                        css_row=lnum,
                        css_col=i,
                        css_converter=self.style_converter,
                        mergestart=mergestart,
                        mergeend=mergeend,
                    )
        else:
            # 使用点号格式化表示级别的旧版格式
            for i, values in enumerate(zip(*level_strs)):
                v = ".".join(map(pprint_thing, values))
                yield CssExcelCell(
                    row=lnum,
                    col=coloffset + i + 1,
                    val=v,
                    style=None,
                    css_styles=getattr(self.styler, "ctx_columns", None),
                    css_row=lnum,
                    css_col=i,
                    css_converter=self.style_converter,
                )

        # 更新行计数器
        self.rowcounter = lnum
    # 生成普通格式的表头单元格迭代器
    def _format_header_regular(self) -> Iterable[ExcelCell]:
        # 如果存在别名或者已有表头
        if self._has_aliases or self.header:
            coloffset = 0

            # 如果存在索引列
            if self.index:
                coloffset = 1
                # 如果数据框索引是多级索引
                if isinstance(self.df.index, MultiIndex):
                    coloffset = len(self.df.index.names)

            # 获取列名列表
            colnames = self.columns
            # 如果存在别名，则使用别名替换表头
            if self._has_aliases:
                self.header = cast(Sequence, self.header)
                # 检查别名列表长度与列名列表长度是否一致
                if len(self.header) != len(self.columns):
                    raise ValueError(
                        f"Writing {len(self.columns)} cols "
                        f"but got {len(self.header)} aliases"
                    )
                colnames = self.header

            # 遍历列名列表，生成对应的 CssExcelCell 对象
            for colindex, colname in enumerate(colnames):
                yield CssExcelCell(
                    row=self.rowcounter,
                    col=colindex + coloffset,
                    val=colname,
                    style=None,
                    css_styles=getattr(self.styler, "ctx_columns", None),
                    css_row=0,
                    css_col=colindex,
                    css_converter=self.style_converter,
                )

    # 格式化表头的主方法，返回合并后的单元格迭代器
    def _format_header(self) -> Iterable[ExcelCell]:
        gen: Iterable[ExcelCell]

        # 如果数据框的列是多级索引
        if isinstance(self.columns, MultiIndex):
            gen = self._format_header_mi()  # 调用多级索引格式化方法
        else:
            gen = self._format_header_regular()  # 调用普通格式化方法

        gen2: Iterable[ExcelCell] = ()

        # 如果数据框有索引名称
        if self.df.index.names:
            # 创建包含索引名称的行列表
            row = [x if x is not None else "" for x in self.df.index.names] + [
                ""
            ] * len(self.columns)
            # 如果所有索引名称均不为空，则生成 ExcelCell 对象的迭代器
            if functools.reduce(lambda x, y: x and y, (x != "" for x in row)):
                gen2 = (
                    ExcelCell(self.rowcounter, colindex, val, None)
                    for colindex, val in enumerate(row)
                )
                self.rowcounter += 1
        # 返回合并后的迭代器
        return itertools.chain(gen, gen2)

    # 格式化表体的方法，返回表体的单元格迭代器
    def _format_body(self) -> Iterable[ExcelCell]:
        # 如果数据框的索引是多级索引
        if isinstance(self.df.index, MultiIndex):
            return self._format_hierarchical_rows()  # 调用多级索引行格式化方法
        else:
            return self._format_regular_rows()  # 调用普通行格式化方法
    def _format_regular_rows(self) -> Iterable[ExcelCell]:
        # 如果存在别名或者有表头，增加行计数器
        if self._has_aliases or self.header:
            self.rowcounter += 1

        # 是否输出索引和索引标签？
        if self.index:
            # 检查索引标签是否存在
            # 如果是列表，则取第一个，因为这不是多重索引
            if self.index_label and isinstance(
                self.index_label, (list, tuple, np.ndarray, Index)
            ):
                index_label = self.index_label[0]
            # 如果是字符串，直接使用
            elif self.index_label and isinstance(self.index_label, str):
                index_label = self.index_label
            else:
                index_label = self.df.index.names[0]

            # 如果列是多重索引，则增加行计数器
            if isinstance(self.columns, MultiIndex):
                self.rowcounter += 1

            # 如果索引标签存在且表头不是 False，则生成 ExcelCell 对象并返回
            if index_label and self.header is not False:
                yield ExcelCell(self.rowcounter - 1, 0, index_label, None)

            # 写入索引值
            index_values = self.df.index
            if isinstance(self.df.index, PeriodIndex):
                index_values = self.df.index.to_timestamp()

            # 遍历索引值，生成 CssExcelCell 对象并返回
            for idx, idxval in enumerate(index_values):
                yield CssExcelCell(
                    row=self.rowcounter + idx,
                    col=0,
                    val=idxval,
                    style=None,
                    css_styles=getattr(self.styler, "ctx_index", None),
                    css_row=idx,
                    css_col=0,
                    css_converter=self.style_converter,
                )
            coloffset = 1
        else:
            coloffset = 0

        # 调用 _generate_body 方法生成主体内容并返回
        yield from self._generate_body(coloffset)

    @property
    def _has_aliases(self) -> bool:
        """Whether the aliases for column names are present."""
        # 返回列名是否存在别名的布尔值
        return is_list_like(self.header)

    def _generate_body(self, coloffset: int) -> Iterable[ExcelCell]:
        # 逐列生成数据框数据的主体部分
        for colidx in range(len(self.columns)):
            series = self.df.iloc[:, colidx]
            for i, val in enumerate(series):
                yield CssExcelCell(
                    row=self.rowcounter + i,
                    col=colidx + coloffset,
                    val=val,
                    style=None,
                    css_styles=getattr(self.styler, "ctx", None),
                    css_row=i,
                    css_col=colidx,
                    css_converter=self.style_converter,
                )

    def get_formatted_cells(self) -> Iterable[ExcelCell]:
        # 返回格式化后的单元格对象的生成器
        for cell in itertools.chain(self._format_header(), self._format_body()):
            cell.val = self._format_value(cell.val)
            yield cell

    @doc(storage_options=_shared_docs["storage_options"])
    def write(
        self,
        writer: FilePath | WriteExcelBuffer | ExcelWriter,  # 定义函数参数 writer，可以是文件路径、文件对象或 ExcelWriter 对象
        sheet_name: str = "Sheet1",  # 指定数据框将被写入的工作表名称，默认为 "Sheet1"
        startrow: int = 0,  # 指定数据框在工作表中的起始行索引，默认为 0
        startcol: int = 0,  # 指定数据框在工作表中的起始列索引，默认为 0
        freeze_panes: tuple[int, int] | None = None,  # 指定要冻结的行列数元组，默认为 None
        engine: str | None = None,  # 指定用于写入的引擎类型，如果 writer 是文件路径时使用，默认为 None
        storage_options: StorageOptions | None = None,  # 存储选项，用于引擎中的存储配置，默认为 None
        engine_kwargs: dict | None = None,  # 传递给 Excel 引擎的额外关键字参数，默认为 None
    ) -> None:
        """
        writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame
        startrow :
            upper left cell row to dump data frame
        startcol :
            upper left cell column to dump data frame
        freeze_panes : tuple of integer (length 2), default None
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen
        engine : string, default None
            write engine to use if writer is a path - you can also set this
            via the options ``io.excel.xlsx.writer``,
            or ``io.excel.xlsm.writer``.

        {storage_options}

        engine_kwargs: dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        from pandas.io.excel import ExcelWriter  # 导入 ExcelWriter 类

        num_rows, num_cols = self.df.shape  # 获取数据框的行数和列数
        if num_rows > self.max_rows or num_cols > self.max_cols:  # 如果数据框大小超出最大允许大小
            raise ValueError(  # 抛出值错误异常
                f"This sheet is too large! Your sheet size is: {num_rows}, {num_cols} "
                f"Max sheet size is: {self.max_rows}, {self.max_cols}"
            )

        if engine_kwargs is None:  # 如果引擎关键字参数为空
            engine_kwargs = {}  # 设置为空字典

        formatted_cells = self.get_formatted_cells()  # 调用获取格式化单元格的方法，返回格式化后的单元格数据
        if isinstance(writer, ExcelWriter):  # 如果 writer 是 ExcelWriter 类型的实例
            need_save = False  # 不需要保存文件
        else:
            writer = ExcelWriter(  # 创建一个 ExcelWriter 对象
                writer,
                engine=engine,
                storage_options=storage_options,
                engine_kwargs=engine_kwargs,
            )
            need_save = True  # 需要保存文件

        try:
            writer._write_cells(  # 调用 writer 对象的 _write_cells 方法，写入单元格数据
                formatted_cells,
                sheet_name,
                startrow=startrow,
                startcol=startcol,
                freeze_panes=freeze_panes,
            )
        finally:
            # make sure to close opened file handles
            if need_save:  # 如果需要保存文件
                writer.close()  # 关闭 writer 对象
```