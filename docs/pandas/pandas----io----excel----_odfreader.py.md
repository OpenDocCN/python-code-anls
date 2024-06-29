# `D:\src\scipysrc\pandas\pandas\io\excel\_odfreader.py`

```
    @property
    def _workbook_class(self) -> type[OpenDocument]:
        # 返回 OpenDocument 类型，用于表示 ODF 文件的工作簿
        from odf.opendocument import OpenDocument
        return OpenDocument

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs
    ) -> OpenDocument:
        # 载入指定路径或缓冲区中的 ODF 文档，返回 OpenDocument 实例
        from odf.opendocument import load
        return load(filepath_or_buffer, **engine_kwargs)

    @property
    def empty_value(self) -> str:
        """Property for compat with other readers."""
        # 返回一个空字符串，用于与其他读取器的兼容性
        return ""

    @property
    def sheet_names(self) -> list[str]:
        """Return a list of sheet names present in the document"""
        # 返回文档中存在的表单名称列表
        from odf.table import Table
        tables = self.book.getElementsByType(Table)
        return [t.getAttribute("name") for t in tables]

    def get_sheet_by_index(self, index: int):
        # 根据索引获取表单对象
        from odf.table import Table
        self.raise_if_bad_sheet_by_index(index)
        tables = self.book.getElementsByType(Table)
        return tables[index]

    def get_sheet_by_name(self, name: str):
        # 根据名称获取表单对象
        from odf.table import Table
        self.raise_if_bad_sheet_by_name(name)
        tables = self.book.getElementsByType(Table)
        for table in tables:
            if table.getAttribute("name") == name:
                return table
        # 如果找不到对应名称的表单，则关闭文档并抛出 ValueError 异常
        self.close()
        raise ValueError(f"sheet {name} not found")

    def get_sheet_data(
        self, sheet, file_rows_needed: int | None = None
    ):
        # 获取指定表单的数据
    ) -> list[list[Scalar | NaTType]]:
        """
        Parse an ODF Table into a list of lists
        """
        # 导入所需的 ODF 表格相关类
        from odf.table import (
            CoveredTableCell,
            TableCell,
            TableRow,
        )

        # 获取覆盖单元格和普通单元格的 QName
        covered_cell_name = CoveredTableCell().qname
        table_cell_name = TableCell().qname
        cell_names = {covered_cell_name, table_cell_name}

        # 获取表格中的所有行对象
        sheet_rows = sheet.getElementsByType(TableRow)
        empty_rows = 0  # 记录空行数
        max_row_len = 0  # 记录最大行长度

        # 初始化表格列表
        table: list[list[Scalar | NaTType]] = []

        # 遍历每一行
        for sheet_row in sheet_rows:
            empty_cells = 0  # 记录空单元格数
            table_row: list[Scalar | NaTType] = []  # 初始化当前行的列表

            # 遍历行中的每个单元格
            for sheet_cell in sheet_row.childNodes:
                # 检查单元格是否属于需要处理的类型
                if hasattr(sheet_cell, "qname") and sheet_cell.qname in cell_names:
                    if sheet_cell.qname == table_cell_name:
                        value = self._get_cell_value(sheet_cell)  # 获取单元格的值
                    else:
                        value = self.empty_value  # 单元格为空时的值

                    column_repeat = self._get_column_repeat(sheet_cell)  # 获取列重复次数

                    # 如果单元格值为空值，则递增空单元格计数
                    if value == self.empty_value:
                        empty_cells += column_repeat
                    else:
                        # 将空单元格填充到表格行中，然后填充具体数值
                        table_row.extend([self.empty_value] * empty_cells)
                        empty_cells = 0
                        table_row.extend([value] * column_repeat)

            # 更新最大行长度
            if max_row_len < len(table_row):
                max_row_len = len(table_row)

            # 获取行重复次数
            row_repeat = self._get_row_repeat(sheet_row)
            if len(table_row) == 0:
                empty_rows += row_repeat
            else:
                # 如果行不为空，则添加空行或者具体行数据到表格中
                table.extend([[self.empty_value]] * empty_rows)
                empty_rows = 0
                table.extend(table_row for _ in range(row_repeat))
            if file_rows_needed is not None and len(table) >= file_rows_needed:
                break

        # 将表格调整为方形（每行等长）
        for row in table:
            if len(row) < max_row_len:
                row.extend([self.empty_value] * (max_row_len - len(row)))

        # 返回最终的表格数据
        return table

    def _get_row_repeat(self, row) -> int:
        """
        Return number of times this row was repeated
        Repeating an empty row appeared to be a common way
        of representing sparse rows in the table.
        """
        # 导入命名空间以便获取行重复次数
        from odf.namespaces import TABLENS

        # 获取行重复次数，并转换为整数
        return int(row.attributes.get((TABLENS, "number-rows-repeated"), 1))

    def _get_column_repeat(self, cell) -> int:
        # 导入命名空间以便获取列重复次数
        from odf.namespaces import TABLENS

        # 获取列重复次数，并转换为整数
        return int(cell.attributes.get((TABLENS, "number-columns-repeated"), 1))
    def _get_cell_value(self, cell) -> Scalar | NaTType:
        # 导入所需的命名空间，用于处理 OpenDocument 格式
        from odf.namespaces import OFFICENS
        
        # 检查单元格内容是否为 "#N/A"，如果是则返回 NaN
        if str(cell) == "#N/A":
            return np.nan
        
        # 获取单元格的值类型
        cell_type = cell.attributes.get((OFFICENS, "value-type"))
        
        # 处理布尔类型的单元格值
        if cell_type == "boolean":
            if str(cell) == "TRUE":
                return True
            return False
        
        # 处理未指定类型的单元格值，返回空值
        if cell_type is None:
            return self.empty_value
        
        # 处理浮点数类型的单元格值
        elif cell_type == "float":
            # 从单元格属性中获取浮点数值并转换为 float 类型
            cell_value = float(cell.attributes.get((OFFICENS, "value")))
            val = int(cell_value)
            # 如果浮点数可以转换为整数，则返回整数值，否则返回浮点数值
            if val == cell_value:
                return val
            return cell_value
        
        # 处理百分比类型的单元格值
        elif cell_type == "percentage":
            cell_value = cell.attributes.get((OFFICENS, "value"))
            return float(cell_value)
        
        # 处理字符串类型的单元格值
        elif cell_type == "string":
            return self._get_cell_string_value(cell)
        
        # 处理货币类型的单元格值
        elif cell_type == "currency":
            cell_value = cell.attributes.get((OFFICENS, "value"))
            return float(cell_value)
        
        # 处理日期类型的单元格值
        elif cell_type == "date":
            cell_value = cell.attributes.get((OFFICENS, "date-value"))
            return pd.Timestamp(cell_value)
        
        # 处理时间类型的单元格值
        elif cell_type == "time":
            # 将单元格内容转换为时间戳并返回时间部分
            stamp = pd.Timestamp(str(cell))
            return cast(Scalar, stamp.time())
        
        # 处理其他未知类型的单元格值，引发异常
        else:
            self.close()
            raise ValueError(f"Unrecognized type {cell_type}")

    def _get_cell_string_value(self, cell) -> str:
        """
        Find and decode OpenDocument text:s tags that represent
        a run length encoded sequence of space characters.
        """
        # 导入所需的元素和命名空间，用于处理 OpenDocument 格式的文本内容
        from odf.element import Element
        from odf.namespaces import TEXTNS
        from odf.office import Annotation
        from odf.text import S
        
        # 定义所需的 QName（Qualified Name）
        office_annotation = Annotation().qname
        text_s = S().qname
        
        # 初始化字符串值列表
        value = []
        
        # 遍历单元格子节点的片段
        for fragment in cell.childNodes:
            if isinstance(fragment, Element):
                # 处理文本段落的情况，解码成一系列以空格字符编码的文本
                if fragment.qname == text_s:
                    spaces = int(fragment.attributes.get((TEXTNS, "c"), 1))
                    value.append(" " * spaces)
                # 忽略办公注解，继续处理下一个片段
                elif fragment.qname == office_annotation:
                    continue
                else:
                    # 递归处理嵌套片段，以处理多个空格的情况
                    value.append(self._get_cell_string_value(fragment))
            else:
                # 将非元素片段转换为字符串并去除换行符，添加到值列表中
                value.append(str(fragment).strip("\n"))
        
        # 将所有值列表中的字符串连接成一个字符串并返回
        return "".join(value)
```