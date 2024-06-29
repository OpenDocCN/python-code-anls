# `D:\src\scipysrc\pandas\pandas\io\excel\_xlrd.py`

```
    @property
    def _workbook_class(self) -> type[Book]:
        # 导入 xlrd 库中的 Book 类型，用于指定返回类型
        from xlrd import Book

        return Book

    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> Book:
        # 导入 xlrd 库中的 open_workbook 函数，根据输入参数加载工作簿数据
        from xlrd import open_workbook

        # 如果 filepath_or_buffer 有 read 方法，将其读取为数据，并传递给 open_workbook 函数
        if hasattr(filepath_or_buffer, "read"):
            data = filepath_or_buffer.read()
            return open_workbook(file_contents=data, **engine_kwargs)
        else:
            # 否则，将 filepath_or_buffer 作为文件路径传递给 open_workbook 函数
            return open_workbook(filepath_or_buffer, **engine_kwargs)

    @property
    def sheet_names(self):
        # 返回由当前工作簿对象包含的所有工作表名称组成的列表
        return self.book.sheet_names()

    def get_sheet_by_name(self, name):
        # 检查指定名称的工作表是否存在，若存在则返回该工作表对象
        self.raise_if_bad_sheet_by_name(name)
        return self.book.sheet_by_name(name)

    def get_sheet_by_index(self, index):
        # 检查指定索引的工作表是否存在，若存在则返回该工作表对象
        self.raise_if_bad_sheet_by_index(index)
        return self.book.sheet_by_index(index)

    def get_sheet_data(
        self, sheet, file_rows_needed: int | None = None
    ):
        # 获取指定工作表的数据，file_rows_needed 参数指示需要的行数，如果为 None，则返回所有行
    ) -> list[list[Scalar]]:
        # 导入 xlrd 库中需要的数据类型和函数
        from xlrd import (
            XL_CELL_BOOLEAN,
            XL_CELL_DATE,
            XL_CELL_ERROR,
            XL_CELL_NUMBER,
            xldate,
        )

        # 获取 Excel 表格的日期模式（1900 还是 1904 epoch）
        epoch1904 = self.book.datemode

        # 定义一个函数，用于解析单元格内容并转换为适合 Pandas 的对象
        def _parse_cell(cell_contents, cell_typ):
            """
            将单元格内容转换为适合 Pandas 使用的对象
            """
            if cell_typ == XL_CELL_DATE:
                # 使用 xlrd 中的日期处理方法 xldate.xldate_as_datetime 进行转换
                try:
                    cell_contents = xldate.xldate_as_datetime(cell_contents, epoch1904)
                except OverflowError:
                    return cell_contents

                # Excel 中日期和时间没有严格区分，这里如果是 epoch 日期，则视为时间
                year = (cell_contents.timetuple())[0:3]
                if (not epoch1904 and year == (1899, 12, 31)) or (
                    epoch1904 and year == (1904, 1, 1)
                ):
                    cell_contents = time(
                        cell_contents.hour,
                        cell_contents.minute,
                        cell_contents.second,
                        cell_contents.microsecond,
                    )

            elif cell_typ == XL_CELL_ERROR:
                # 如果是错误类型的单元格，将内容设为 NaN
                cell_contents = np.nan
            elif cell_typ == XL_CELL_BOOLEAN:
                # 如果是布尔类型的单元格，将内容转换为布尔值
                cell_contents = bool(cell_contents)
            elif cell_typ == XL_CELL_NUMBER:
                # 如果是数字类型的单元格，Excel 中的数字总是浮点数
                # 尝试将其转换为整数，如果转换后与原值相等，则设为整数
                if math.isfinite(cell_contents):
                    val = int(cell_contents)
                    if val == cell_contents:
                        cell_contents = val
            return cell_contents

        # 获取表格的行数
        nrows = sheet.nrows
        # 如果指定了需要的行数，取行数和指定行数的较小值
        if file_rows_needed is not None:
            nrows = min(nrows, file_rows_needed)
        
        # 遍历每一行，对每个单元格应用 _parse_cell 函数，生成解析后的数据列表
        return [
            [
                _parse_cell(value, typ)
                for value, typ in zip(sheet.row_values(i), sheet.row_types(i))
            ]
            for i in range(nrows)
        ]
```