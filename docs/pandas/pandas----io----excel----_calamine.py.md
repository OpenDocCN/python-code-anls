# `D:\src\scipysrc\pandas\pandas\io\excel\_calamine.py`

```
# 从未来模块导入注解支持，用于在类型检查时引用当前模块的类型
from __future__ import annotations

# 导入日期时间相关模块
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
# 导入类型检查相关模块
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

# 导入可选依赖
from pandas.compat._optional import import_optional_dependency
# 导入文档装饰器
from pandas.util._decorators import doc

# 导入 pandas 库并简写为 pd
import pandas as pd
# 导入共享文档内容
from pandas.core.shared_docs import _shared_docs

# 导入 Excel 读取基类
from pandas.io.excel._base import BaseExcelReader

# 如果类型检查开启，则导入 CalamineSheet 和 CalamineWorkbook 类型
if TYPE_CHECKING:
    from python_calamine import (
        CalamineSheet,
        CalamineWorkbook,
    )

    # 导入其他类型别名
    from pandas._typing import (
        FilePath,
        NaTType,
        ReadBuffer,
        Scalar,
        StorageOptions,
    )

# 定义单元格值类型的别名
_CellValue = Union[int, float, str, bool, time, date, datetime, timedelta]


# 定义 CalamineReader 类，继承自 BaseExcelReader["CalamineWorkbook"]
class CalamineReader(BaseExcelReader["CalamineWorkbook"]):
    # 使用文档装饰器定义初始化方法，描述参数含义
    @doc(storage_options=_shared_docs["storage_options"])
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],  # 文件路径或缓冲区
        storage_options: StorageOptions | None = None,  # 存储选项
        engine_kwargs: dict | None = None,  # 引擎关键字参数
    ) -> None:
        """
        Reader using calamine engine (xlsx/xls/xlsb/ods).

        Parameters
        ----------
        filepath_or_buffer : str, path to be parsed or
            an open readable stream.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        # 导入 python_calamine 可选依赖
        import_optional_dependency("python_calamine")
        # 调用父类初始化方法
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    # 定义 _workbook_class 属性方法，返回 CalamineWorkbook 类型
    @property
    def _workbook_class(self) -> type[CalamineWorkbook]:
        # 导入 CalamineWorkbook 类
        from python_calamine import CalamineWorkbook

        return CalamineWorkbook

    # 定义 load_workbook 方法，加载工作簿
    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs: Any
    ) -> CalamineWorkbook:
        # 导入 load_workbook 方法
        from python_calamine import load_workbook

        # 调用 load_workbook 方法加载工作簿
        return load_workbook(
            filepath_or_buffer,
            **engine_kwargs,
        )

    # 定义 sheet_names 属性方法，返回所有工作表的名称列表
    @property
    def sheet_names(self) -> list[str]:
        # 导入 SheetTypeEnum 枚举
        from python_calamine import SheetTypeEnum

        # 返回所有工作表名称列表，仅包括工作表类型为 WorkSheet 的
        return [
            sheet.name
            for sheet in self.book.sheets_metadata
            if sheet.typ == SheetTypeEnum.WorkSheet
        ]

    # 定义 get_sheet_by_name 方法，根据名称获取工作表对象
    def get_sheet_by_name(self, name: str) -> CalamineSheet:
        # 如果名称无效则抛出异常
        self.raise_if_bad_sheet_by_name(name)
        # 返回指定名称的工作表对象
        return self.book.get_sheet_by_name(name)

    # 定义 get_sheet_by_index 方法，根据索引获取工作表对象
    def get_sheet_by_index(self, index: int) -> CalamineSheet:
        # 如果索引无效则抛出异常
        self.raise_if_bad_sheet_by_index(index)
        # 返回指定索引的工作表对象
        return self.book.get_sheet_by_index(index)

    # 定义 get_sheet_data 方法，获取工作表数据
    def get_sheet_data(
        self, sheet: CalamineSheet, file_rows_needed: int | None = None
    ):
    # 定义一个函数，该函数用于将单元格的值转换为标量、NaTType 或时间对象
    def _convert_cell(value: _CellValue) -> Scalar | NaTType | time:
        # 如果值是浮点数
        if isinstance(value, float):
            # 转换为整数
            val = int(value)
            # 如果转换后的整数与原始值相等，返回整数；否则返回原始浮点数
            if val == value:
                return val
            else:
                return value
        # 如果值是日期对象
        elif isinstance(value, date):
            # 转换为 Pandas 的时间戳对象
            return pd.Timestamp(value)
        # 如果值是时间间隔对象
        elif isinstance(value, timedelta):
            # 转换为 Pandas 的时间间隔对象
            return pd.Timedelta(value)
        # 如果值是时间对象
        elif isinstance(value, time):
            # 直接返回时间对象
            return value

        # 如果不属于上述类型，则直接返回原始值
        return value

    # 从工作表中读取数据，并且保留空区域，仅读取指定行数
    rows: list[list[_CellValue]] = sheet.to_python(
        skip_empty_area=False, nrows=file_rows_needed
    )
    # 对每一行中的每个单元格应用 _convert_cell 函数，将数据转换为适当的类型
    data = [[_convert_cell(cell) for cell in row] for row in rows]

    # 返回转换后的数据
    return data
```