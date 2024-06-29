# `D:\src\scipysrc\pandas\pandas\io\excel\_odswriter.py`

```
    def _write_cells(
        self,
        cells: list[ExcelCell],
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
    ) -> None:
        """
        Write cells to a specific sheet in the workbook.

        Args:
            cells: List of ExcelCell objects representing cells to write.
            sheet_name: Optional name of the sheet to write cells to.
            startrow: Starting row index for writing cells (default is 0).
            startcol: Starting column index for writing cells (default is 0).
            freeze_panes: Optional tuple indicating the row and column to freeze panes.

        Returns:
            None
        """
        # Iterate over provided cells and write them to the specified sheet
        for cell in cells:
            # Calculate the target row and column indices for each cell
            row = startrow + cell.row
            col = startcol + cell.col
            # Validate and set the cell value in the sheet
            self.sheets[sheet_name].setCellText(
                cell.row, cell.col, str(cell.value)
            )
        
        # Validate and apply freeze panes if specified
        if freeze_panes is not None:
            validate_freeze_panes(self.sheets[sheet_name], *freeze_panes)
    ) -> None:
        """
        Write the frame cells using odf
        """
        from odf.table import (
            Table,              # 导入表格类
            TableCell,          # 导入单元格类
            TableRow,           # 导入行类
        )
        from odf.text import P   # 导入段落类

        sheet_name = self._get_sheet_name(sheet_name)  # 获取有效的表格名称
        assert sheet_name is not None  # 断言确保表格名称不为空

        if sheet_name in self.sheets:   # 如果表格名称已存在于self.sheets中
            wks = self.sheets[sheet_name]  # 则使用已存在的表格对象
        else:
            wks = Table(name=sheet_name)   # 否则创建新的表格对象
            self.book.spreadsheet.addElement(wks)  # 将新表格对象添加到文档中

        if validate_freeze_panes(freeze_panes):  # 如果冻结窗格有效
            freeze_panes = cast(tuple[int, int], freeze_panes)  # 将冻结窗格转换为元组类型
            self._create_freeze_panes(sheet_name, freeze_panes)  # 创建冻结窗格

        for _ in range(startrow):   # 根据起始行添加空行到表格中
            wks.addElement(TableRow())

        rows: DefaultDict = defaultdict(TableRow)   # 使用默认字典存储行对象
        col_count: DefaultDict = defaultdict(int)   # 使用默认字典存储列计数

        for cell in sorted(cells, key=lambda cell: (cell.row, cell.col)):
            # 只有在行为空时才添加空单元格
            if not col_count[cell.row]:
                for _ in range(startcol):
                    rows[cell.row].addElement(TableCell())

            # 如果需要，填充空单元格
            for _ in range(cell.col - col_count[cell.row]):
                rows[cell.row].addElement(TableCell())
                col_count[cell.row] += 1

            pvalue, tc = self._make_table_cell(cell)   # 创建表格单元格和段落对象
            rows[cell.row].addElement(tc)   # 将单元格添加到行中
            col_count[cell.row] += 1
            p = P(text=pvalue)   # 创建段落对象
            tc.addElement(p)     # 将段落对象添加到单元格对象中

        # 将所有行添加到表格中
        if len(rows) > 0:
            for row_nr in range(max(rows.keys()) + 1):
                wks.addElement(rows[row_nr])

    def _make_table_cell_attributes(self, cell: ExcelCell) -> dict[str, int | str]:
        """Convert cell attributes to OpenDocument attributes

        Parameters
        ----------
        cell : ExcelCell
            Spreadsheet cell data

        Returns
        -------
        attributes : Dict[str, Union[int, str]]
            Dictionary with attributes and attribute values
        """
        attributes: dict[str, int | str] = {}   # 初始化属性字典
        style_name = self._process_style(cell.style)   # 处理单元格样式
        if style_name is not None:
            attributes["stylename"] = style_name   # 如果样式名不为空，则添加样式名属性
        if cell.mergestart is not None and cell.mergeend is not None:
            attributes["numberrowsspanned"] = max(1, cell.mergestart)   # 添加行合并跨度属性
            attributes["numbercolumnsspanned"] = cell.mergeend   # 添加列合并跨度属性
        return attributes   # 返回处理后的属性字典
    # 将 Excel 单元格数据转换为 OpenDocument 电子表格的单元格

    # 定义_make_table_cell方法，接受一个ExcelCell对象作为参数，并返回一个元组
    def _make_table_cell(self, cell: ExcelCell) -> tuple[object, Any]:
        """Convert cell data to an OpenDocument spreadsheet cell

        Parameters
        ----------
        cell : ExcelCell
            Spreadsheet cell data

        Returns
        -------
        pvalue, cell : Tuple[str, TableCell]
            Display value, Cell value
        """
        
        # 导入所需的 TableCell 类
        from odf.table import TableCell

        # 根据传入的 cell 参数生成单元格的属性
        attributes = self._make_table_cell_attributes(cell)
        
        # 调用 _value_with_fmt 方法，获取值和格式
        val, fmt = self._value_with_fmt(cell.val)
        
        # 初始化 pvalue 和 value 变量为 val 的值
        pvalue = value = val
        
        # 如果 val 是布尔类型
        if isinstance(val, bool):
            # 将布尔值转换为小写字符串和大写字符串作为显示值
            value = str(val).lower()
            pvalue = str(val).upper()
            # 返回 pvalue 和一个 TableCell 对象，表示布尔类型的单元格
            return (
                pvalue,
                TableCell(
                    valuetype="boolean",
                    booleanvalue=value,
                    attributes=attributes,
                ),
            )
        
        # 如果 val 是 datetime.datetime 类型
        elif isinstance(val, datetime.datetime):
            # 快速格式化日期时间为 ISO 格式
            value = val.isoformat()
            # 使用本地化相关的慢速格式化为本地时间格式
            pvalue = val.strftime("%c")
            # 返回 pvalue 和一个 TableCell 对象，表示日期时间类型的单元格
            return (
                pvalue,
                TableCell(valuetype="date", datevalue=value, attributes=attributes),
            )
        
        # 如果 val 是 datetime.date 类型
        elif isinstance(val, datetime.date):
            # 快速格式化日期为 ISO 格式
            value = f"{val.year}-{val.month:02d}-{val.day:02d}"
            # 使用本地化相关的慢速格式化为本地日期格式
            pvalue = val.strftime("%x")
            # 返回 pvalue 和一个 TableCell 对象，表示日期类型的单元格
            return (
                pvalue,
                TableCell(valuetype="date", datevalue=value, attributes=attributes),
            )
        
        # 如果 val 是字符串类型
        elif isinstance(val, str):
            # 返回 pvalue 和一个 TableCell 对象，表示字符串类型的单元格
            return (
                pvalue,
                TableCell(
                    valuetype="string",
                    stringvalue=value,
                    attributes=attributes,
                ),
            )
        
        # 对于其他类型的值，假定为数值类型
        else:
            # 返回 pvalue 和一个 TableCell 对象，表示数值类型的单元格
            return (
                pvalue,
                TableCell(
                    valuetype="float",
                    value=value,
                    attributes=attributes,
                ),
            )

    # 定义 _process_style 方法的重载，接受一个字典类型的 style 参数，返回字符串类型的结果
    @overload
    def _process_style(self, style: dict[str, Any]) -> str: ...

    # 定义 _process_style 方法的重载，接受一个空值 None 的 style 参数，返回空值 None
    @overload
    def _process_style(self, style: None) -> None: ...
    def _process_style(self, style: dict[str, Any] | None) -> str | None:
        """Convert a style dictionary to a OpenDocument style sheet

        Parameters
        ----------
        style : Dict
            Style dictionary

        Returns
        -------
        style_key : str
            Unique style key for later reference in sheet
        """
        # 导入所需的 OpenDocument 样式相关模块
        from odf.style import (
            ParagraphProperties,
            Style,
            TableCellProperties,
            TextProperties,
        )

        # 如果 style 为 None，则直接返回 None
        if style is None:
            return None
        
        # 将 style 字典转换为 JSON 格式的字符串，作为唯一的样式键
        style_key = json.dumps(style)
        
        # 如果样式键已经存在于 _style_dict 中，直接返回对应的样式名称
        if style_key in self._style_dict:
            return self._style_dict[style_key]
        
        # 生成新的样式名称，以 "pd" 开头，后面跟着当前样式字典数量加一的数字
        name = f"pd{len(self._style_dict)+1}"
        
        # 将生成的样式名称和样式键存入 _style_dict 中
        self._style_dict[style_key] = name
        
        # 创建一个新的 OpenDocument 样式对象，类型为 table-cell
        odf_style = Style(name=name, family="table-cell")
        
        # 处理字体设置
        if "font" in style:
            font = style["font"]
            if font.get("bold", False):
                # 如果字体设置为加粗，则添加加粗属性
                odf_style.addElement(TextProperties(fontweight="bold"))
        
        # 处理边框设置
        if "borders" in style:
            borders = style["borders"]
            for side, thickness in borders.items():
                # 将边框粗细映射为对应的 OpenDocument 格式，添加到样式中
                thickness_translation = {"thin": "0.75pt solid #000000"}
                odf_style.addElement(
                    TableCellProperties(
                        attributes={f"border{side}": thickness_translation[thickness]}
                    )
                )
        
        # 处理对齐设置
        if "alignment" in style:
            alignment = style["alignment"]
            horizontal = alignment.get("horizontal")
            if horizontal:
                # 如果有水平对齐设置，则添加段落属性
                odf_style.addElement(ParagraphProperties(textalign=horizontal))
            vertical = alignment.get("vertical")
            if vertical:
                # 如果有垂直对齐设置，则添加单元格属性
                odf_style.addElement(TableCellProperties(verticalalign=vertical))
        
        # 将创建的样式对象添加到文档的样式表中
        self.book.styles.addElement(odf_style)
        
        # 返回新创建的样式名称
        return name
    ) -> None:
        """
        Create freeze panes in the sheet.

        Parameters
        ----------
        sheet_name : str
            Name of the spreadsheet
        freeze_panes : tuple of (int, int)
            Freeze pane location x and y
        """
        # 导入所需的配置模块
        from odf.config import (
            ConfigItem,
            ConfigItemMapEntry,
            ConfigItemMapIndexed,
            ConfigItemMapNamed,
            ConfigItemSet,
        )

        # 创建一个视图设置的配置项集合
        config_item_set = ConfigItemSet(name="ooo:view-settings")
        # 将配置项集合添加到文档对象中
        self.book.settings.addElement(config_item_set)

        # 创建一个索引映射的配置项
        config_item_map_indexed = ConfigItemMapIndexed(name="Views")
        # 将索引映射配置项添加到配置项集合中
        config_item_set.addElement(config_item_map_indexed)

        # 创建一个映射条目的配置项
        config_item_map_entry = ConfigItemMapEntry()
        # 将映射条目配置项添加到索引映射配置项中
        config_item_map_indexed.addElement(config_item_map_entry)

        # 创建一个命名映射的配置项
        config_item_map_named = ConfigItemMapNamed(name="Tables")
        # 将命名映射配置项添加到映射条目配置项中
        config_item_map_entry.addElement(config_item_map_named)

        # 创建具有给定名称的新映射条目配置项
        config_item_map_entry = ConfigItemMapEntry(name=sheet_name)
        # 将新映射条目配置项添加到命名映射配置项中
        config_item_map_named.addElement(config_item_map_entry)

        # 添加水平分割模式配置项
        config_item_map_entry.addElement(
            ConfigItem(name="HorizontalSplitMode", type="short", text="2")
        )
        # 添加垂直分割模式配置项
        config_item_map_entry.addElement(
            ConfigItem(name="VerticalSplitMode", type="short", text="2")
        )
        # 添加水平分割位置配置项
        config_item_map_entry.addElement(
            ConfigItem(
                name="HorizontalSplitPosition", type="int", text=str(freeze_panes[0])
            )
        )
        # 添加垂直分割位置配置项
        config_item_map_entry.addElement(
            ConfigItem(
                name="VerticalSplitPosition", type="int", text=str(freeze_panes[1])
            )
        )
        # 添加右侧位置配置项
        config_item_map_entry.addElement(
            ConfigItem(name="PositionRight", type="int", text=str(freeze_panes[0]))
        )
        # 添加底部位置配置项
        config_item_map_entry.addElement(
            ConfigItem(name="PositionBottom", type="int", text=str(freeze_panes[1]))
        )
```