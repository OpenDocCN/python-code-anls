# `D:\src\scipysrc\pandas\pandas\io\excel\_pyxlsb.py`

```
# pyright: reportMissingImports=false
# 导入未来版本的注解支持
from __future__ import annotations

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入依赖的库和函数
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

# 导入共享文档信息
from pandas.core.shared_docs import _shared_docs

# 导入基础的 Excel 读取类
from pandas.io.excel._base import BaseExcelReader

# 如果是类型检查模式，导入 Workbook 类
if TYPE_CHECKING:
    from pyxlsb import Workbook

    # 导入路径、读取缓冲、标量和存储选项类型
    from pandas._typing import (
        FilePath,
        ReadBuffer,
        Scalar,
        StorageOptions,
    )


# 继承自 BaseExcelReader 的 PyxlsbReader 类
class PyxlsbReader(BaseExcelReader["Workbook"]):
    @doc(storage_options=_shared_docs["storage_options"])
    # 初始化方法
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        """
        使用 pyxlsb 引擎的 Excel 读取器。

        Parameters
        ----------
        filepath_or_buffer : str, path object, or Workbook
            要解析的对象。
        {storage_options}
        engine_kwargs : dict, optional
            传递给 Excel 引擎的任意关键字参数。
        """
        # 导入可选依赖项 pyxlsb
        import_optional_dependency("pyxlsb")
        
        # 调用父类的初始化方法，设置 book 属性为加载的结果
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    # 属性方法，返回 Workbook 类型
    @property
    def _workbook_class(self) -> type[Workbook]:
        # 导入 pyxlsb 的 Workbook 类
        from pyxlsb import Workbook

        return Workbook

    # 加载工作簿的方法
    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs
    ) -> Workbook:
        # 导入 pyxlsb 的 open_workbook 函数
        from pyxlsb import open_workbook

        # TODO: 在缓冲区功能中的一个修改
        # 这可能需要对 Pyxlsb 库进行一些修改
        # 实际打开工作簿的工作在 xlsbpackage.py，大约第20行左右

        # 调用 open_workbook 方法打开文件或缓冲区，并传入引擎关键字参数
        return open_workbook(filepath_or_buffer, **engine_kwargs)

    # 返回工作表名列表的属性方法
    @property
    def sheet_names(self) -> list[str]:
        return self.book.sheets

    # 根据名称获取工作表的方法
    def get_sheet_by_name(self, name: str):
        # 检查工作表名称是否有效
        self.raise_if_bad_sheet_by_name(name)
        # 返回指定名称的工作表对象
        return self.book.get_sheet(name)

    # 根据索引获取工作表的方法
    def get_sheet_by_index(self, index: int):
        # 检查工作表索引是否有效
        self.raise_if_bad_sheet_by_index(index)
        # pyxlsb 的工作表索引从1开始，返回索引加一的工作表对象
        return self.book.get_sheet(index + 1)

    # 将单元格转换为标量值的方法
    def _convert_cell(self, cell) -> Scalar:
        # TODO: pyxlsb 中无法区分浮点数和日期时间类型
        # 这意味着无法从 xlsb 文件中读取日期时间类型
        if cell.v is None:
            return ""  # 避免未命名列显示为 Unnamed: i
        if isinstance(cell.v, float):
            val = int(cell.v)
            if val == cell.v:
                return val
            else:
                return float(cell.v)

        return cell.v
    def get_sheet_data(
        self,
        sheet,
        file_rows_needed: int | None = None,
    ) -> list[list[Scalar]]:
        # 初始化空数据列表
        data: list[list[Scalar]] = []
        # 初始上一个行号设为-1
        previous_row_number = -1
        # 当 sparse=True 时，行可能具有不同的长度，并且不返回空行。
        # 单元格是命名元组，包含行号、列号、值 (r, c, v)。
        for row in sheet.rows(sparse=True):
            # 获取当前行的行号
            row_number = row[0].r
            # 将每个单元格转换为内部格式
            converted_row = [self._convert_cell(cell) for cell in row]
            # 删除尾部的空元素
            while converted_row and converted_row[-1] == "":
                converted_row.pop()
            # 如果转换后的行不为空
            if converted_row:
                # 填充空数据以扩展行数至当前行和上一行之间的差
                data.extend([[]] * (row_number - previous_row_number - 1))
                # 添加当前转换后的行数据到数据列表中
                data.append(converted_row)
                # 更新上一个行号
                previous_row_number = row_number
            # 如果指定了文件所需的行数，并且数据长度已经达到或超过所需行数，跳出循环
            if file_rows_needed is not None and len(data) >= file_rows_needed:
                break
        # 如果数据不为空
        if data:
            # 计算数据中最大行的长度
            max_width = max(len(data_row) for data_row in data)
            # 如果数据中最小行的长度小于最大行的长度
            if min(len(data_row) for data_row in data) < max_width:
                # 创建空单元格列表
                empty_cell: list[Scalar] = [""]
                # 扩展每行数据以填充至最大宽度
                data = [
                    data_row + (max_width - len(data_row)) * empty_cell
                    for data_row in data
                ]
        # 返回处理后的数据列表
        return data
```