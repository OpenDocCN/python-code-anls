# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\excel.py`

```
# 从 io 模块中导入 BytesIO 类，用于创建二进制数据的缓冲区
from io import BytesIO

# 导入 numpy 库，并使用 np 作为别名
import numpy as np

# 从 odf.opendocument 模块中导入 OpenDocumentSpreadsheet 类
from odf.opendocument import OpenDocumentSpreadsheet

# 从 odf.table 模块中导入 Table、TableCell 和 TableRow 类
from odf.table import (
    Table,
    TableCell,
    TableRow,
)

# 从 odf.text 模块中导入 P 类
from odf.text import P

# 从 pandas 库中导入 DataFrame、ExcelWriter、Index、date_range 和 read_excel 函数
from pandas import (
    DataFrame,
    ExcelWriter,
    Index,
    date_range,
    read_excel,
)


# 定义一个函数 _generate_dataframe，用于生成一个 DataFrame 对象并返回
def _generate_dataframe():
    # 设置 N 和 C 的值
    N = 2000
    C = 5
    # 创建一个 DataFrame 对象 df，填充随机数值，指定列名和索引
    df = DataFrame(
        np.random.randn(N, C),
        columns=[f"float{i}" for i in range(C)],
        index=date_range("20000101", periods=N, freq="h"),
    )
    # 向 df 添加一个名为 'object' 的列，使用 Index 对象包装字符串列表作为值
    df["object"] = Index([f"i-{i}" for i in range(N)], dtype=object)
    # 返回生成的 DataFrame 对象 df
    return df


# 定义一个类 WriteExcel，用于执行将 DataFrame 写入 Excel 的操作
class WriteExcel:
    # 定义参数 params 和 param_names
    params = ["openpyxl", "xlsxwriter"]
    param_names = ["engine"]

    # 定义 setup 方法，用于初始化数据
    def setup(self, engine):
        # 调用 _generate_dataframe 函数生成一个 DataFrame 对象并赋值给 self.df
        self.df = _generate_dataframe()

    # 定义 time_write_excel 方法，用于将 self.df 数据写入 Excel 文件
    def time_write_excel(self, engine):
        # 创建一个 BytesIO 对象 bio 用于写入 Excel 数据
        bio = BytesIO()
        bio.seek(0)
        # 使用 ExcelWriter 创建一个 Excel 写入对象 writer
        with ExcelWriter(bio, engine=engine) as writer:
            # 将 self.df 数据写入 Excel 文件的 Sheet1 页
            self.df.to_excel(writer, sheet_name="Sheet1")


# 定义一个类 WriteExcelStyled，用于将样式化的 DataFrame 数据写入 Excel
class WriteExcelStyled:
    # 定义参数 params 和 param_names
    params = ["openpyxl", "xlsxwriter"]
    param_names = ["engine"]

    # 定义 setup 方法，用于初始化数据
    def setup(self, engine):
        # 调用 _generate_dataframe 函数生成一个 DataFrame 对象并赋值给 self.df
        self.df = _generate_dataframe()

    # 定义 time_write_excel_style 方法，用于将样式化的 self.df 数据写入 Excel 文件
    def time_write_excel_style(self, engine):
        # 创建一个 BytesIO 对象 bio 用于写入 Excel 数据
        bio = BytesIO()
        bio.seek(0)
        # 使用 ExcelWriter 创建一个 Excel 写入对象 writer
        with ExcelWriter(bio, engine=engine) as writer:
            # 将 self.df 样式化后的数据写入 Excel 文件的 Sheet1 页
            df_style = self.df.style
            df_style.map(lambda x: "border: red 1px solid;")
            df_style.map(lambda x: "color: blue")
            df_style.map(lambda x: "border-color: green black", subset=["float1"])
            df_style.to_excel(writer, sheet_name="Sheet1")


# 定义一个类 ReadExcel，用于执行从 Excel 文件读取数据的操作
class ReadExcel:
    # 定义参数 params 和 param_names
    params = ["openpyxl", "odf"]
    param_names = ["engine"]

    # 定义 Excel 文件名和 ODF 文件名
    fname_excel = "spreadsheet.xlsx"
    fname_odf = "spreadsheet.ods"

    # 定义 _create_odf 方法，用于创建 ODF 文件
    def _create_odf(self):
        # 创建一个 OpenDocumentSpreadsheet 对象 doc
        doc = OpenDocumentSpreadsheet()
        # 创建一个名为 'Table1' 的表格对象 table
        table = Table(name="Table1")
        # 遍历 self.df 中的每一行数据
        for row in self.df.values:
            # 创建一个 TableRow 对象 tr
            tr = TableRow()
            # 遍历行中的每个值 val
            for val in row:
                # 创建一个带有字符串值的 TableCell 对象 tc
                tc = TableCell(valuetype="string")
                tc.addElement(P(text=val))
                # 将 tc 添加到 tr 中
                tr.addElement(tc)
            # 将 tr 添加到 table 中
            table.addElement(tr)

        # 将 table 添加到 doc.spreadsheet 中
        doc.spreadsheet.addElement(table)
        # 将 doc 保存为 self.fname_odf 文件
        doc.save(self.fname_odf)

    # 定义 setup_cache 方法，用于初始化数据和创建 Excel、ODF 文件
    def setup_cache(self):
        # 调用 _generate_dataframe 函数生成一个 DataFrame 对象并赋值给 self.df
        self.df = _generate_dataframe()

        # 将 self.df 数据写入 Excel 文件的 Sheet1 页
        self.df.to_excel(self.fname_excel, sheet_name="Sheet1")
        # 调用 _create_odf 方法创建 ODF 文件
        self._create_odf()

    # 定义 time_read_excel 方法，用于从 Excel 文件中读取数据
    def time_read_excel(self, engine):
        # 根据 engine 参数选择要读取的文件名 fname
        if engine == "odf":
            fname = self.fname_odf
        else:
            fname = self.fname_excel
        # 调用 read_excel 函数读取 fname 文件中的数据
        read_excel(fname, engine=engine)


# 定义一个类 ReadExcelNRows，继承自 ReadExcel 类，用于从 Excel 文件中读取部分数据
class ReadExcelNRows(ReadExcel):
    # 重写 time_read_excel 方法，用于从 Excel 文件中读取前 10 行数据
    def time_read_excel(self, engine):
        # 根据 engine 参数选择要读取的文件名 fname
        if engine == "odf":
            fname = self.fname_odf
        else:
            fname = self.fname_excel
        # 调用 read_excel 函数读取 fname 文件中的前 10 行数据
        read_excel(fname, engine=engine, nrows=10)

# 从 pandas_vb_common 模块中导入 setup 函数
from ..pandas_vb_common import setup  # noqa: F401 isort:skip
```