# `D:\src\scipysrc\pandas\pandas\tests\io\excel\test_writers.py`

```
from datetime import (
    date,  # 导入日期对象
    datetime,  # 导入日期时间对象
    timedelta,  # 导入时间间隔对象
)
from functools import partial  # 导入偏函数功能
from io import BytesIO  # 导入字节流对象
import os  # 导入操作系统相关功能
import re  # 导入正则表达式模块
import uuid  # 导入UUID生成模块

import numpy as np  # 导入NumPy库，重命名为np
import pytest  # 导入pytest测试框架

from pandas.compat._optional import import_optional_dependency  # 导入可选依赖
import pandas.util._test_decorators as td  # 导入测试装饰器
import pandas as pd  # 导入Pandas库，重命名为pd
from pandas import (
    DataFrame,  # 导入DataFrame类
    Index,  # 导入Index类
    MultiIndex,  # 导入MultiIndex类
    date_range,  # 导入日期范围生成函数
    option_context,  # 导入上下文管理器函数
)
import pandas._testing as tm  # 导入Pandas测试模块

from pandas.io.excel import (
    ExcelFile,  # 导入ExcelFile类
    ExcelWriter,  # 导入ExcelWriter类
    _OpenpyxlWriter,  # 导入OpenpyxlWriter类
    _XlsxWriter,  # 导入XlsxWriter类
    register_writer,  # 导入注册写入器函数
)
from pandas.io.excel._util import _writers  # 导入Excel写入器工具函数


def get_exp_unit(path: str) -> str:
    """
    根据文件路径判断文件类型，返回时间单位。
    如果文件路径以".ods"结尾，则返回's'；否则返回'us'。
    """
    if path.endswith(".ods"):
        return "s"
    return "us"


@pytest.fixture
def frame(float_frame):
    """
    Fixture函数：返回fixture "float_frame"的前十个条目。
    """
    return float_frame[:10]


@pytest.fixture(params=[True, False, "columns"])
def merge_cells(request):
    """
    Fixture函数：根据参数化请求返回True、False或者"columns"。
    """
    return request.param


@pytest.fixture
def tmp_excel(ext, tmp_path):
    """
    Fixture函数：创建临时Excel文件并返回其路径字符串。
    """
    tmp = tmp_path / f"{uuid.uuid4()}{ext}"
    tmp.touch()
    return str(tmp)


@pytest.fixture
def set_engine(engine, ext):
    """
    Fixture函数：设置Excel文件写入引擎。

    该fixture用于在每个测试用例中设置全局选项以指定使用的写入引擎。
    在测试执行后，将撤销对全局选项的更改。
    """
    option_name = f"io.excel.{ext.strip('.')}.writer"
    with option_context(option_name, engine):
        yield


@pytest.mark.parametrize(
    "ext",
    [
        pytest.param(".xlsx", marks=[td.skip_if_no("openpyxl"), td.skip_if_no("xlrd")]),
        pytest.param(".xlsm", marks=[td.skip_if_no("openpyxl"), td.skip_if_no("xlrd")]),
        pytest.param(
            ".xlsx", marks=[td.skip_if_no("xlsxwriter"), td.skip_if_no("xlrd")]
        ),
        pytest.param(".ods", marks=td.skip_if_no("odf")),
    ],
)
class TestRoundTrip:
    @pytest.mark.parametrize(
        "header,expected",
        [(None, [np.nan] * 4), (0, {"Unnamed: 0": [np.nan] * 3})],
    )
    def test_read_one_empty_col_no_header(self, tmp_excel, header, expected):
        """
        测试函数：读取Excel文件中的数据，验证空列和无标题情况下的预期结果。

        Parameters:
        - tmp_excel: 临时Excel文件路径
        - header: 标题行的处理方式，可以为None或0
        - expected: 预期的DataFrame结果
        """
        # 标记：引用GitHub问题编号12292
        # 准备数据
        filename = "no_header"
        df = DataFrame([["", 1, 100], ["", 2, 200], ["", 3, 300], ["", 4, 400]])

        # 将DataFrame写入临时Excel文件
        df.to_excel(tmp_excel, sheet_name=filename, index=False, header=False)

        # 从临时Excel文件中读取数据
        result = pd.read_excel(
            tmp_excel, sheet_name=filename, usecols=[0], header=header
        )
        expected = DataFrame(expected)

        # 断言：验证读取结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "header,expected_extra",
        [(None, [0]), (0, [])],
    )
    def test_read_one_empty_col_with_header(self, tmp_excel, header, expected_extra):
        # 定义测试函数，测试带有标题的情况下读取包含空列的数据
        filename = "with_header"
        # 创建包含空列的 DataFrame
        df = DataFrame([["", 1, 100], ["", 2, 200], ["", 3, 300], ["", 4, 400]])

        # 将 DataFrame 写入 Excel 文件，指定工作表名称和是否包含标题行
        df.to_excel(tmp_excel, sheet_name="with_header", index=False, header=True)
        
        # 从 Excel 文件读取数据，仅使用第一列，并根据 header 参数确定是否有标题行
        result = pd.read_excel(
            tmp_excel, sheet_name=filename, usecols=[0], header=header
        )
        
        # 创建预期的 DataFrame，与读取结果进行比较
        expected = DataFrame(expected_extra + [np.nan] * 4)
        tm.assert_frame_equal(result, expected)

    def test_set_column_names_in_parameter(self, tmp_excel):
        # GH 12870 : 传递与关键字参数名相关联的列名
        refdf = DataFrame([[1, "foo"], [2, "bar"], [3, "baz"]], columns=["a", "b"])

        # 使用 ExcelWriter 将数据写入 Excel 文件，分别包含和不包含标题行
        with ExcelWriter(tmp_excel) as writer:
            refdf.to_excel(writer, sheet_name="Data_no_head", header=False, index=False)
            refdf.to_excel(writer, sheet_name="Data_with_head", index=False)

        # 修改 DataFrame 的列名
        refdf.columns = ["A", "B"]

        # 使用 ExcelFile 读取 Excel 文件中的数据，分别设置不同的参数
        with ExcelFile(tmp_excel) as reader:
            xlsdf_no_head = pd.read_excel(
                reader, sheet_name="Data_no_head", header=None, names=["A", "B"]
            )
            xlsdf_with_head = pd.read_excel(
                reader,
                sheet_name="Data_with_head",
                index_col=None,
                names=["A", "B"],
            )

        # 断言读取的 DataFrame 是否与预期的 DataFrame 相等
        tm.assert_frame_equal(xlsdf_no_head, refdf)
        tm.assert_frame_equal(xlsdf_with_head, refdf)

    def test_creating_and_reading_multiple_sheets(self, tmp_excel):
        # see gh-9450
        #
        # 测试从运行时创建的包含多个工作表的 Excel 文件中读取多个工作表的数据
        def tdf(col_sheet_name):
            d, i = [11, 22, 33], [1, 2, 3]
            return DataFrame(d, i, columns=[col_sheet_name])

        sheets = ["AAA", "BBB", "CCC"]

        # 创建包含多个工作表的 DataFrame 字典
        dfs = [tdf(s) for s in sheets]
        dfs = dict(zip(sheets, dfs))

        # 使用 ExcelWriter 将多个工作表的数据写入 Excel 文件
        with ExcelWriter(tmp_excel) as ew:
            for sheetname, df in dfs.items():
                df.to_excel(ew, sheet_name=sheetname)

        # 从 Excel 文件读取多个工作表的数据，设置第一列为索引列
        dfs_returned = pd.read_excel(tmp_excel, sheet_name=sheets, index_col=0)

        # 断言每个读取的工作表的 DataFrame 是否与原始创建的 DataFrame 相等
        for s in sheets:
            tm.assert_frame_equal(dfs[s], dfs_returned[s])
    # 定义测试函数，用于测试读取具有多级索引和空级别的 Excel 文件情况
    def test_read_excel_multiindex_empty_level(self, tmp_excel):
        # 添加注释 "see gh-12453"，指明这个测试用例是为了解决 GitHub 上的 issue 12453
        # 创建一个 DataFrame 对象 df，包含多级列索引和一个空的级别
        df = DataFrame(
            {
                ("One", "x"): {0: 1},       # 列名 ("One", "x") 的值为 1
                ("Two", "X"): {0: 3},       # 列名 ("Two", "X") 的值为 3
                ("Two", "Y"): {0: 7},       # 列名 ("Two", "Y") 的值为 7
                ("Zero", ""): {0: 0},       # 列名 ("Zero", "") 的值为 0
            }
        )

        # 创建期望的 DataFrame 对象 expected，对应于 df 的结构，但是 "Zero" 列的空级别名被命名为 "Unnamed: 4_level_1"
        expected = DataFrame(
            {
                ("One", "x"): {0: 1},                       # 列名 ("One", "x") 的值为 1
                ("Two", "X"): {0: 3},                       # 列名 ("Two", "X") 的值为 3
                ("Two", "Y"): {0: 7},                       # 列名 ("Two", "Y") 的值为 7
                ("Zero", "Unnamed: 4_level_1"): {0: 0},     # 列名 ("Zero", "Unnamed: 4_level_1") 的值为 0
            }
        )

        # 将 DataFrame df 写入到临时 Excel 文件 tmp_excel 中
        df.to_excel(tmp_excel)
        # 使用 Pandas 的 read_excel 方法读取 tmp_excel 文件内容，指定了多级列索引和以第一列为索引列
        actual = pd.read_excel(tmp_excel, header=[0, 1], index_col=0)
        # 使用 Pandas 测试工具 assert_frame_equal 检查读取的 actual DataFrame 是否与期望的 expected DataFrame 相同
        tm.assert_frame_equal(actual, expected)

        # 创建另一个 DataFrame 对象 df，包含多级列索引，但没有空级别
        df = DataFrame(
            {
                ("Beg", ""): {0: 0},       # 列名 ("Beg", "") 的值为 0
                ("Middle", "x"): {0: 1},   # 列名 ("Middle", "x") 的值为 1
                ("Tail", "X"): {0: 3},     # 列名 ("Tail", "X") 的值为 3
                ("Tail", "Y"): {0: 7},     # 列名 ("Tail", "Y") 的值为 7
            }
        )

        # 创建期望的 DataFrame 对象 expected，对应于 df 的结构，但是 "Beg" 列的空级别名被命名为 "Unnamed: 1_level_1"
        expected = DataFrame(
            {
                ("Beg", "Unnamed: 1_level_1"): {0: 0},      # 列名 ("Beg", "Unnamed: 1_level_1") 的值为 0
                ("Middle", "x"): {0: 1},                    # 列名 ("Middle", "x") 的值为 1
                ("Tail", "X"): {0: 3},                      # 列名 ("Tail", "X") 的值为 3
                ("Tail", "Y"): {0: 7},                      # 列名 ("Tail", "Y") 的值为 7
            }
        )

        # 将 DataFrame df 写入到临时 Excel 文件 tmp_excel 中
        df.to_excel(tmp_excel)
        # 使用 Pandas 的 read_excel 方法读取 tmp_excel 文件内容，指定了多级列索引和以第一列为索引列
        actual = pd.read_excel(tmp_excel, header=[0, 1], index_col=0)
        # 使用 Pandas 测试工具 assert_frame_equal 检查读取的 actual DataFrame 是否与期望的 expected DataFrame 相同
        tm.assert_frame_equal(actual, expected)
        # see gh-4679
        # Empty name case current read in as
        # unnamed levels, not Nones.
        # 检查是否存在列索引名称或者列层级数小于等于1的情况，如果是，则设置检查标志为True
        check_names = bool(r_idx_names) or r_idx_levels <= 1

        # 如果列层级数为1，则创建一个包含单一列名的索引对象
        columns = Index(list("abcde"))
        else:
            # 否则，创建一个多层次索引对象，每层包含一个长度为5的列表作为数据，同时为每层设置名称
            columns = MultiIndex.from_arrays(
                [range(5) for _ in range(c_idx_levels)],
                names=[f"{c_idx_names}-{i}" for i in range(c_idx_levels)],
            )
        
        # 如果行层级数为1，则创建一个包含单一行索引的索引对象
        index = Index(list("ghijk"))
        else:
            # 否则，创建一个多层次索引对象，每层包含一个长度为5的列表作为数据，同时为每层设置名称
            index = MultiIndex.from_arrays(
                [range(5) for _ in range(r_idx_levels)],
                names=[f"{r_idx_names}-{i}" for i in range(r_idx_levels)],
            )
        
        # 创建一个数据帧对象，数据由5行5列的1.1倍数值矩阵组成，列索引为columns，行索引为index
        df = DataFrame(
            1.1 * np.ones((5, 5)),
            columns=columns,
            index=index,
        )
        
        # 将数据帧写入到临时Excel文件中
        df.to_excel(tmp_excel)

        # 从Excel文件中读取数据，设置行索引和列索引的层级结构，得到的结果存储在act变量中
        act = pd.read_excel(
            tmp_excel,
            index_col=list(range(r_idx_levels)),
            header=list(range(c_idx_levels)),
        )
        
        # 使用测试工具tm来比较预期的数据帧df和读取得到的数据帧act，如果check_names为True，则检查列名是否相同
        tm.assert_frame_equal(df, act, check_names=check_names)

        # 将数据帧第一行所有列的值设置为NaN（缺失值），然后将数据帧写入到临时Excel文件中
        df.iloc[0, :] = np.nan
        df.to_excel(tmp_excel)

        # 从更新后的Excel文件中再次读取数据，设置行索引和列索引的层级结构，得到的结果存储在act变量中
        act = pd.read_excel(
            tmp_excel,
            index_col=list(range(r_idx_levels)),
            header=list(range(c_idx_levels)),
        )
        
        # 再次使用测试工具tm来比较更新后的数据帧df和读取得到的数据帧act，检查列名是否相同
        tm.assert_frame_equal(df, act, check_names=check_names)

        # 将数据帧最后一行所有列的值设置为NaN（缺失值），然后将数据帧写入到临时Excel文件中
        df.iloc[-1, :] = np.nan
        df.to_excel(tmp_excel)
        
        # 从更新后的Excel文件中再次读取数据，设置行索引和列索引的层级结构，得到的结果存储在act变量中
        act = pd.read_excel(
            tmp_excel,
            index_col=list(range(r_idx_levels)),
            header=list(range(c_idx_levels)),
        )
        
        # 第三次使用测试工具tm来比较更新后的数据帧df和读取得到的数据帧act，检查列名是否相同
        tm.assert_frame_equal(df, act, check_names=check_names)
    # 定义一个测试方法，用于测试多索引和时间间隔的日期时间
    def test_multiindex_interval_datetimes(self, tmp_excel):
        # GH 30986: GitHub issue编号，标识这段代码的相关问题
        # 创建一个多级索引对象，由两个数组组成：
        # - 第一个数组包含范围为 0 到 3 的整数
        # - 第二个数组包含时间间隔范围，从"2020-01-01"开始，每6个月一个时间间隔，总共4个间隔
        midx = MultiIndex.from_arrays(
            [
                range(4),
                pd.interval_range(
                    start=pd.Timestamp("2020-01-01"), periods=4, freq="6ME"
                ),
            ]
        )
        # 创建一个数据框，使用上述多级索引作为索引，数据是范围为0到3的整数
        df = DataFrame(range(4), index=midx)
        # 将数据框写入到 Excel 文件中
        df.to_excel(tmp_excel)
        # 从 Excel 文件中读取数据，指定多级索引的列为[0, 1]
        result = pd.read_excel(tmp_excel, index_col=[0, 1])
        # 创建预期的数据框，使用多级索引对象作为索引，数据由以下两个数组组成：
        # - 第一个数组是范围为0到3的整数
        # - 第二个数组是包含时间间隔字符串的数组，表示每个时间间隔的开始和结束日期
        expected = DataFrame(
            range(4),
            MultiIndex.from_arrays(
                [
                    range(4),
                    [
                        "(2020-01-31 00:00:00, 2020-07-31 00:00:00]",
                        "(2020-07-31 00:00:00, 2021-01-31 00:00:00]",
                        "(2021-01-31 00:00:00, 2021-07-31 00:00:00]",
                        "(2021-07-31 00:00:00, 2022-01-31 00:00:00]",
                    ],
                ]
            ),
        )
        # 使用测试框架的函数来比较实际结果和预期结果的数据框是否相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "engine,ext",
    [  # 定义参数化测试的参数列表，包括不同的引擎和文件扩展名
        pytest.param(
            "openpyxl",
            ".xlsx",
            marks=[td.skip_if_no("openpyxl"), td.skip_if_no("xlrd")],  # 添加标记以跳过缺少的依赖项
        ),
        pytest.param(
            "openpyxl",
            ".xlsm",
            marks=[td.skip_if_no("openpyxl"), td.skip_if_no("xlrd")],  # 同样添加标记以跳过缺少的依赖项
        ),
        pytest.param(
            "xlsxwriter",
            ".xlsx",
            marks=[td.skip_if_no("xlsxwriter"), td.skip_if_no("xlrd")],  # 添加标记以跳过缺少的依赖项
        ),
        pytest.param("odf", ".ods", marks=td.skip_if_no("odf")),  # 添加标记以跳过缺少的依赖项
    ],
)
@pytest.mark.usefixtures("set_engine")
class TestExcelWriter:
    def test_excel_sheet_size(self, tmp_excel):
        # GH 26080
        breaking_row_count = 2**20 + 1  # 定义一个破坏性行数，用于测试
        breaking_col_count = 2**14 + 1  # 定义一个破坏性列数，用于测试
        # 用两个数组来防止在测试过程中出现内存问题
        row_arr = np.zeros(shape=(breaking_row_count, 1))  # 创建一个指定形状的全零数组
        col_arr = np.zeros(shape=(1, breaking_col_count))  # 创建一个指定形状的全零数组
        row_df = DataFrame(row_arr)  # 将数组转换为 DataFrame 对象
        col_df = DataFrame(col_arr)  # 将数组转换为 DataFrame 对象

        msg = "sheet is too large"  # 定义错误消息字符串
        with pytest.raises(ValueError, match=msg):  # 检查是否会引发 ValueError 异常，并匹配错误消息
            row_df.to_excel(tmp_excel)  # 将 DataFrame 写入 Excel，并预期引发异常

        with pytest.raises(ValueError, match=msg):  # 检查是否会引发 ValueError 异常，并匹配错误消息
            col_df.to_excel(tmp_excel)  # 将 DataFrame 写入 Excel，并预期引发异常

    def test_excel_sheet_by_name_raise(self, tmp_excel):
        gt = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))  # 创建一个指定形状的 DataFrame
        gt.to_excel(tmp_excel)  # 将 DataFrame 写入 Excel

        with ExcelFile(tmp_excel) as xl:  # 使用 ExcelFile 打开 Excel 文件
            df = pd.read_excel(xl, sheet_name=0, index_col=0)  # 从 Excel 文件中读取指定的 sheet

        tm.assert_frame_equal(gt, df)  # 使用测试工具验证 DataFrame 的相等性

        msg = "Worksheet named '0' not found"  # 定义错误消息字符串
        with pytest.raises(ValueError, match=msg):  # 检查是否会引发 ValueError 异常，并匹配错误消息
            pd.read_excel(xl, "0")  # 从 Excel 文件中读取指定名称的 sheet，并预期引发异常

    def test_excel_writer_context_manager(self, frame, tmp_excel):
        with ExcelWriter(tmp_excel) as writer:  # 使用 ExcelWriter 打开 Excel 文件进行写入操作
            frame.to_excel(writer, sheet_name="Data1")  # 将 DataFrame 写入 Excel 中的指定 sheet
            frame2 = frame.copy()  # 复制 DataFrame 对象
            frame2.columns = frame.columns[::-1]  # 修改列的顺序
            frame2.to_excel(writer, sheet_name="Data2")  # 将修改后的 DataFrame 写入 Excel 中的指定 sheet

        with ExcelFile(tmp_excel) as reader:  # 使用 ExcelFile 打开 Excel 文件进行读取操作
            found_df = pd.read_excel(reader, sheet_name="Data1", index_col=0)  # 从 Excel 中读取指定的 sheet
            found_df2 = pd.read_excel(reader, sheet_name="Data2", index_col=0)  # 从 Excel 中读取指定的 sheet

            tm.assert_frame_equal(found_df, frame)  # 使用测试工具验证 DataFrame 的相等性
            tm.assert_frame_equal(found_df2, frame2)  # 使用测试工具验证 DataFrame 的相等性
    def test_roundtrip(self, frame, tmp_excel):
        # 复制数据框以确保原始数据不变
        frame = frame.copy()
        # 将前5行"A"列的值设为NaN
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan

        # 将数据框写入Excel文件，指定工作表名称为"test1"
        frame.to_excel(tmp_excel, sheet_name="test1")
        # 将数据框写入Excel文件，指定工作表名称为"test1"，仅包含"A"和"B"列
        frame.to_excel(tmp_excel, sheet_name="test1", columns=["A", "B"])
        # 将数据框写入Excel文件，指定工作表名称为"test1"，不包含表头
        frame.to_excel(tmp_excel, sheet_name="test1", header=False)
        # 将数据框写入Excel文件，指定工作表名称为"test1"，不包含索引
        frame.to_excel(tmp_excel, sheet_name="test1", index=False)

        # 测试数据框的往返写入和读取
        frame.to_excel(tmp_excel, sheet_name="test1")
        # 从Excel文件中读取数据，指定工作表名称为"test1"，以第一列作为索引列
        recons = pd.read_excel(tmp_excel, sheet_name="test1", index_col=0)
        # 使用测试框架比较原始数据框和重新构建的数据框
        tm.assert_frame_equal(frame, recons)

        # 将数据框写入Excel文件，指定工作表名称为"test1"，不包含索引
        frame.to_excel(tmp_excel, sheet_name="test1", index=False)
        # 从Excel文件中读取数据，指定工作表名称为"test1"，不设置索引列
        recons = pd.read_excel(tmp_excel, sheet_name="test1", index_col=None)
        # 将重新构建的数据框的索引设置为与原始数据框相同
        recons.index = frame.index
        # 使用测试框架比较原始数据框和重新构建的数据框
        tm.assert_frame_equal(frame, recons)

        # 将数据框写入Excel文件，指定工作表名称为"test1"，将NaN值表示为"NA"
        frame.to_excel(tmp_excel, sheet_name="test1", na_rep="NA")
        # 从Excel文件中读取数据，指定工作表名称为"test1"，以第一列作为索引列，将"NA"值解析为NaN
        recons = pd.read_excel(tmp_excel, sheet_name="test1", index_col=0, na_values=["NA"])
        # 使用测试框架比较原始数据框和重新构建的数据框
        tm.assert_frame_equal(frame, recons)

        # GH 3611：处理GH 3611问题，将NaN值表示为"88"
        frame.to_excel(tmp_excel, sheet_name="test1", na_rep="88")
        # 从Excel文件中读取数据，指定工作表名称为"test1"，以第一列作为索引列，将"88"值解析为NaN
        recons = pd.read_excel(tmp_excel, sheet_name="test1", index_col=0, na_values=["88"])
        # 使用测试框架比较原始数据框和重新构建的数据框
        tm.assert_frame_equal(frame, recons)

        # 再次处理GH 3611问题，将NaN值表示为"88"，包括整数和浮点数
        frame.to_excel(tmp_excel, sheet_name="test1", na_rep="88")
        # 从Excel文件中读取数据，指定工作表名称为"test1"，以第一列作为索引列，将"88"值和"88.0"值解析为NaN
        recons = pd.read_excel(tmp_excel, sheet_name="test1", index_col=0, na_values=[88, 88.0])
        # 使用测试框架比较原始数据框和重新构建的数据框
        tm.assert_frame_equal(frame, recons)

        # GH 6573：处理GH 6573问题，将整个数据框写入名为"Sheet1"的工作表
        frame.to_excel(tmp_excel, sheet_name="Sheet1")
        # 从Excel文件中读取数据，将第一列作为索引列
        recons = pd.read_excel(tmp_excel, index_col=0)
        # 使用测试框架比较原始数据框和重新构建的数据框
        tm.assert_frame_equal(frame, recons)

        # 将整个数据框写入名为"0"的工作表
        frame.to_excel(tmp_excel, sheet_name="0")
        # 从Excel文件中读取数据，将第一列作为索引列
        recons = pd.read_excel(tmp_excel, index_col=0)
        # 使用测试框架比较原始数据框和重新构建的数据框
        tm.assert_frame_equal(frame, recons)

        # GH 8825：测试Pandas Series应提供to_excel方法
        # 将数据框的"A"列转换为Series，并将其写入Excel文件
        s = frame["A"]
        s.to_excel(tmp_excel)
        # 从Excel文件中读取数据，将第一列作为索引列
        recons = pd.read_excel(tmp_excel, index_col=0)
        # 使用测试框架比较原始Series和重新构建的数据框
        tm.assert_frame_equal(s.to_frame(), recons)

    def test_mixed(self, frame, tmp_excel):
        # 复制数据框以确保原始数据不变
        mixed_frame = frame.copy()
        # 在数据框中添加新列"foo"，所有行的值均为"bar"
        mixed_frame["foo"] = "bar"

        # 将整个数据框写入名为"test1"的工作表
        mixed_frame.to_excel(tmp_excel, sheet_name="test1")
        # 使用ExcelFile类打开Excel文件，通过上下文管理器读取数据，指定工作表名称为"test1"，将第一列作为索引列
        with ExcelFile(tmp_excel) as reader:
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0)
        # 使用测试框架比较原始数据框和重新构建的数据框
        tm.assert_frame_equal(mixed_frame, recons)
    # 测试处理时间序列框架的函数
    def test_ts_frame(self, tmp_excel):
        # 获取表达单位
        unit = get_exp_unit(tmp_excel)
        # 创建一个随机数据帧
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )

        # 将数据帧索引转换为 DatetimeIndex 类型，不设置频率
        index = pd.DatetimeIndex(np.asarray(df.index), freq=None)
        df.index = index

        # 创建预期的数据帧
        expected = df[:]
        # 根据给定的单位调整预期数据帧的索引
        expected.index = expected.index.as_unit(unit)

        # 将数据帧写入到 Excel 文件中的 "test1" 工作表
        df.to_excel(tmp_excel, sheet_name="test1")
        # 使用 ExcelFile 打开 Excel 文件
        with ExcelFile(tmp_excel) as reader:
            # 从 "test1" 工作表中读取数据重构成 recons 数据帧
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0)
        # 使用断言验证预期数据帧与重构数据帧是否相等
        tm.assert_frame_equal(expected, recons)

    # 测试包含 NaN 值的基本功能
    def test_basics_with_nan(self, frame, tmp_excel):
        # 复制数据帧
        frame = frame.copy()
        # 将数据帧中前 5 行 "A" 列的值设置为 NaN
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan
        # 将数据帧写入到 Excel 文件中的 "test1" 工作表
        frame.to_excel(tmp_excel, sheet_name="test1")
        # 指定写入特定列 "A" 和 "B" 到 "test1" 工作表
        frame.to_excel(tmp_excel, sheet_name="test1", columns=["A", "B"])
        # 写入数据帧到 "test1" 工作表，但不包括表头
        frame.to_excel(tmp_excel, sheet_name="test1", header=False)
        # 写入数据帧到 "test1" 工作表，但不包括索引
        frame.to_excel(tmp_excel, sheet_name="test1", index=False)

    # 测试整数类型
    @pytest.mark.parametrize("np_type", [np.int8, np.int16, np.int32, np.int64])
    def test_int_types(self, np_type, tmp_excel):
        # 测试从 Excel 读取 np.int 值时返回整数而不是浮点数
        df = DataFrame(
            np.random.default_rng(2).integers(-10, 10, size=(10, 2)), dtype=np_type
        )
        # 将数据帧写入到 Excel 文件中的 "test1" 工作表
        df.to_excel(tmp_excel, sheet_name="test1")

        # 使用 ExcelFile 打开 Excel 文件
        with ExcelFile(tmp_excel) as reader:
            # 从 "test1" 工作表中读取数据重构成 recons 数据帧
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0)

        # 将数据帧转换为 np.int64 类型
        int_frame = df.astype(np.int64)
        # 使用断言验证整数化后的数据帧与重构数据帧是否相等
        tm.assert_frame_equal(int_frame, recons)

        # 直接从 Excel 文件中读取数据帧，并使用断言验证整数化后的数据帧与读取的数据帧是否相等
        recons2 = pd.read_excel(tmp_excel, sheet_name="test1", index_col=0)
        tm.assert_frame_equal(int_frame, recons2)

    # 测试浮点数类型
    @pytest.mark.parametrize("np_type", [np.float16, np.float32, np.float64])
    def test_float_types(self, np_type, tmp_excel):
        # 测试从 Excel 读取 np.float 值时返回浮点数
        df = DataFrame(np.random.default_rng(2).random(10), dtype=np_type)
        # 将数据帧写入到 Excel 文件中的 "test1" 工作表
        df.to_excel(tmp_excel, sheet_name="test1")

        # 使用 ExcelFile 打开 Excel 文件
        with ExcelFile(tmp_excel) as reader:
            # 从 "test1" 工作表中读取数据重构成 recons 数据帧，并转换为指定类型 np_type
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0).astype(
                np_type
            )

        # 使用断言验证原始数据帧与重构数据帧是否相等
        tm.assert_frame_equal(df, recons)

    # 测试布尔类型
    def test_bool_types(self, tmp_excel):
        # 测试从 Excel 读取 np.bool_ 值时返回布尔值
        df = DataFrame([1, 0, True, False], dtype=np.bool_)
        # 将数据帧写入到 Excel 文件中的 "test1" 工作表
        df.to_excel(tmp_excel, sheet_name="test1")

        # 使用 ExcelFile 打开 Excel 文件
        with ExcelFile(tmp_excel) as reader:
            # 从 "test1" 工作表中读取数据重构成 recons 数据帧，并转换为指定类型 np.bool_
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0).astype(
                np.bool_
            )

        # 使用断言验证原始数据帧与重构数据帧是否相等
        tm.assert_frame_equal(df, recons)
    def test_inf_roundtrip(self, tmp_excel):
        # 创建包含无穷大和负无穷大的数据框
        df = DataFrame([(1, np.inf), (2, 3), (5, -np.inf)])
        # 将数据框写入 Excel 文件中的一个工作表
        df.to_excel(tmp_excel, sheet_name="test1")

        # 使用 ExcelFile 对象读取 Excel 文件
        with ExcelFile(tmp_excel) as reader:
            # 从指定工作表读取数据到 recons 变量
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0)

        # 断言原始数据框和重新构建的数据框相等
        tm.assert_frame_equal(df, recons)

    def test_sheets(self, frame, tmp_excel):
        # freq 不会回转
        # 获取时间单位
        unit = get_exp_unit(tmp_excel)
        # 创建一个包含随机数据的时间序列数据框
        tsframe = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )

        # 将时间索引转换为给定的单位
        index = pd.DatetimeIndex(np.asarray(tsframe.index), freq=None)
        tsframe.index = index

        # 创建预期的数据框副本
        expected = tsframe[:]
        expected.index = expected.index.as_unit(unit)

        # 复制输入数据框，并将部分数据设置为 NaN
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan

        # 将数据框写入 Excel 文件的多个工作表
        frame.to_excel(tmp_excel, sheet_name="test1")
        frame.to_excel(tmp_excel, sheet_name="test1", columns=["A", "B"])
        frame.to_excel(tmp_excel, sheet_name="test1", header=False)
        frame.to_excel(tmp_excel, sheet_name="test1", index=False)

        # 测试写入到不同工作表
        with ExcelWriter(tmp_excel) as writer:
            frame.to_excel(writer, sheet_name="test1")
            tsframe.to_excel(writer, sheet_name="test2")
        with ExcelFile(tmp_excel) as reader:
            # 从 Excel 文件中读取特定工作表的数据，并进行断言比较
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0)
            tm.assert_frame_equal(frame, recons)
            recons = pd.read_excel(reader, sheet_name="test2", index_col=0)
        # 断言预期的数据框和重新构建的数据框相等
        tm.assert_frame_equal(expected, recons)
        # 断言 Excel 文件包含两个工作表
        assert 2 == len(reader.sheet_names)
        assert "test1" == reader.sheet_names[0]
        assert "test2" == reader.sheet_names[1]

    def test_colaliases(self, frame, tmp_excel):
        # 复制输入数据框，并将部分数据设置为 NaN
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan

        # 将数据框写入 Excel 文件的一个工作表
        frame.to_excel(tmp_excel, sheet_name="test1")
        frame.to_excel(tmp_excel, sheet_name="test1", columns=["A", "B"])
        frame.to_excel(tmp_excel, sheet_name="test1", header=False)
        frame.to_excel(tmp_excel, sheet_name="test1", index=False)

        # 使用列别名写入数据框到 Excel 文件的一个工作表
        col_aliases = Index(["AA", "X", "Y", "Z"])
        frame.to_excel(tmp_excel, sheet_name="test1", header=col_aliases)
        with ExcelFile(tmp_excel) as reader:
            # 从 Excel 文件中读取特定工作表的数据，并进行断言比较
            rs = pd.read_excel(reader, sheet_name="test1", index_col=0)
        # 创建预期的数据框副本，将列名转换为列别名
        xp = frame.copy()
        xp.columns = col_aliases
        # 断言预期的数据框和重新构建的数据框相等
        tm.assert_frame_equal(xp, rs)
    # 定义测试函数，用于测试带索引标签的 Excel 表格往返操作
    def test_roundtrip_indexlabels(self, merge_cells, frame, tmp_excel):
        # 复制数据帧以防止修改原始数据
        frame = frame.copy()
        # 将第一列前五行设置为 NaN
        frame.iloc[:5, frame.columns.get_loc("A")] = np.nan
        
        # 将数据帧写入 Excel 文件，指定工作表名为 "test1"
        frame.to_excel(tmp_excel, sheet_name="test1")
        # 再次将数据帧写入同一 Excel 文件的 "test1" 工作表，仅包括列 "A" 和 "B"
        frame.to_excel(tmp_excel, sheet_name="test1", columns=["A", "B"])
        # 将数据帧写入 Excel 文件的 "test1" 工作表，但不包括表头
        frame.to_excel(tmp_excel, sheet_name="test1", header=False)
        # 将数据帧写入 Excel 文件的 "test1" 工作表，不包括索引
        frame.to_excel(tmp_excel, sheet_name="test1", index=False)

        # 创建随机数据帧，内容为正态分布的随机数，并写入 Excel 文件的 "test1" 工作表，设置索引标签为 ["test"]，并根据参数 merge_cells 合并单元格
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2))) >= 0
        df.to_excel(
            tmp_excel, sheet_name="test1", index_label=["test"], merge_cells=merge_cells
        )
        # 使用 ExcelFile 对象读取 Excel 文件，获取工作表 "test1" 的内容，设置第一列为索引，并将数据类型转换为 np.int64
        with ExcelFile(tmp_excel) as reader:
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0).astype(
                np.int64
            )
        # 设置原始数据帧的索引名称为 ["test"]，并断言与重构数据帧的索引名称相同
        df.index.names = ["test"]
        assert df.index.names == recons.index.names

        # 创建随机数据帧，并写入 Excel 文件的 "test1" 工作表，设置多个索引标签为 ["test", "dummy", "dummy2"]，并根据参数 merge_cells 合并单元格
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2))) >= 0
        df.to_excel(
            tmp_excel,
            sheet_name="test1",
            index_label=["test", "dummy", "dummy2"],
            merge_cells=merge_cells,
        )
        # 使用 ExcelFile 对象读取 Excel 文件，获取工作表 "test1" 的内容，设置第一列为索引，并将数据类型转换为 np.int64
        with ExcelFile(tmp_excel) as reader:
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0).astype(
                np.int64
            )
        # 设置原始数据帧的索引名称为 ["test"]，并断言与重构数据帧的索引名称相同
        df.index.names = ["test"]
        assert df.index.names == recons.index.names

        # 创建随机数据帧，并写入 Excel 文件的 "test1" 工作表，设置索引标签为 "test"，并根据参数 merge_cells 合并单元格
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2))) >= 0
        df.to_excel(
            tmp_excel, sheet_name="test1", index_label="test", merge_cells=merge_cells
        )
        # 使用 ExcelFile 对象读取 Excel 文件，获取工作表 "test1" 的内容，设置第一列为索引，并将数据类型转换为 np.int64
        with ExcelFile(tmp_excel) as reader:
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0).astype(
                np.int64
            )
        # 设置原始数据帧的索引名称为 ["test"]，并使用断言比较两个数据帧是否相等
        df.index.names = ["test"]
        tm.assert_frame_equal(df, recons.astype(bool))

        # 将数据帧写入 Excel 文件的 "test1" 工作表，仅包括列 "A", "B", "C", "D"，不包括索引，并根据参数 merge_cells 合并单元格
        frame.to_excel(
            tmp_excel,
            sheet_name="test1",
            columns=["A", "B", "C", "D"],
            index=False,
            merge_cells=merge_cells,
        )
        # 将数据帧的 ["A", "B"] 列设置为新的索引（与列 "C", "D" 在同一行）
        df = frame.copy()
        df = df.set_index(["A", "B"])

        # 使用 ExcelFile 对象读取 Excel 文件，获取工作表 "test1" 的内容，设置前两列为多级索引
        with ExcelFile(tmp_excel) as reader:
            recons = pd.read_excel(reader, sheet_name="test1", index_col=[0, 1])
        # 使用断言比较两个数据帧是否相等
        tm.assert_frame_equal(df, recons)

    # 定义测试函数，用于测试带索引名称的 Excel 表格往返操作
    def test_excel_roundtrip_indexname(self, merge_cells, tmp_excel):
        # 创建随机数据帧，并设置索引名称为 "foo"
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        df.index.name = "foo"

        # 将数据帧写入 Excel 文件，并根据参数 merge_cells 合并单元格
        df.to_excel(tmp_excel, merge_cells=merge_cells)

        # 使用 ExcelFile 对象读取 Excel 文件，获取第一个工作表的内容，并将第一列设置为索引
        with ExcelFile(tmp_excel) as xf:
            result = pd.read_excel(xf, sheet_name=xf.sheet_names[0], index_col=0)

        # 使用断言比较读取的数据帧与原始数据帧是否相等
        tm.assert_frame_equal(result, df)
        # 断言读取数据帧的索引名称为 "foo"
        assert result.index.name == "foo"
    def test_excel_roundtrip_datetime(self, merge_cells, tmp_excel):
        # 测试 Excel 往返日期时间处理

        # 获取临时 Excel 文件的日期单位
        unit = get_exp_unit(tmp_excel)

        # 创建一个 DataFrame，填充随机标准正态分布的数据
        tsframe = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )

        # 将 DataFrame 的索引转换为 DatetimeIndex，去除频率信息
        index = pd.DatetimeIndex(np.asarray(tsframe.index), freq=None)
        tsframe.index = index

        # 复制 DataFrame
        tsf = tsframe.copy()

        # 将索引中的日期时间转换为日期
        tsf.index = [x.date() for x in tsframe.index]

        # 将 DataFrame 写入到 Excel 文件中，包含合并单元格选项
        tsf.to_excel(tmp_excel, sheet_name="test1", merge_cells=merge_cells)

        # 使用 ExcelFile 打开临时 Excel 文件作为 reader
        with ExcelFile(tmp_excel) as reader:
            # 从 Excel 文件中读取名为 "test1" 的 sheet 数据到 recons
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0)

        # 设置期望值为 tsframe 的复制
        expected = tsframe[:]

        # 将期望值的索引转换为指定单位的时间戳
        expected.index = expected.index.as_unit(unit)

        # 使用断言函数比较期望值和从 Excel 中恢复的数据
        tm.assert_frame_equal(expected, recons)

    def test_excel_date_datetime_format(self, ext, tmp_excel, tmp_path):
        # 查看 gh-4133
        #
        # Excel 输出格式字符串

        # 获取临时 Excel 文件的日期单位
        unit = get_exp_unit(tmp_excel)

        # 创建一个 DataFrame，包含日期和日期时间数据
        df = DataFrame(
            [
                [date(2014, 1, 31), date(1999, 9, 24)],
                [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)],
            ],
            index=["DATE", "DATETIME"],
            columns=["X", "Y"],
        )

        # 创建期望的 DataFrame，将其数据类型转换为指定单位的时间戳
        df_expected = DataFrame(
            [
                [datetime(2014, 1, 31), datetime(1999, 9, 24)],
                [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)],
            ],
            index=["DATE", "DATETIME"],
            columns=["X", "Y"],
        )
        df_expected = df_expected.astype(f"M8[{unit}]")

        # 创建一个临时文件名
        filename2 = tmp_path / f"tmp2{ext}"
        filename2.touch()

        # 使用 ExcelWriter 写入 df 到 tmp_excel 中的 "test1" sheet
        with ExcelWriter(tmp_excel) as writer1:
            df.to_excel(writer1, sheet_name="test1")

        # 使用 ExcelWriter 写入 df 到 filename2 中的 "test1" sheet，指定日期和日期时间格式
        with ExcelWriter(
            filename2,
            date_format="DD.MM.YYYY",
            datetime_format="DD.MM.YYYY HH-MM-SS",
        ) as writer2:
            df.to_excel(writer2, sheet_name="test1")

        # 使用 ExcelFile 打开临时 Excel 文件作为 reader1
        with ExcelFile(tmp_excel) as reader1:
            # 从 Excel 文件中读取名为 "test1" 的 sheet 数据到 rs1
            rs1 = pd.read_excel(reader1, sheet_name="test1", index_col=0)

        # 使用 ExcelFile 打开 filename2 作为 reader2
        with ExcelFile(filename2) as reader2:
            # 从 filename2 中读取名为 "test1" 的 sheet 数据到 rs2
            rs2 = pd.read_excel(reader2, sheet_name="test1", index_col=0)

        # 将 rs2 的数据类型转换为指定单位的时间戳
        rs2 = rs2.astype(f"M8[{unit}]")

        # 使用断言函数比较 rs1 和 rs2 的数据
        tm.assert_frame_equal(rs1, rs2)

        # 由于读取器为日期返回 datetime 对象，因此需要使用 df_expected 来检查结果
        tm.assert_frame_equal(rs2, df_expected)
    def test_to_excel_interval_no_labels(self, tmp_excel, using_infer_string):
        # see gh-19242
        #
        # Test writing Interval without labels.
        
        # 创建一个包含随机整数数据的 DataFrame，数据范围在 [-10, 10] 之间
        df = DataFrame(
            np.random.default_rng(2).integers(-10, 10, size=(20, 1)), dtype=np.int64
        )
        expected = df.copy()  # 复制 DataFrame
        
        # 将 df 中的第一列数据分成 10 个区间，并将结果存入 "new" 列
        df["new"] = pd.cut(df[0], 10)
        
        # 用相同的方法处理 expected DataFrame，并将结果转换为字符串类型（根据使用的条件）
        expected["new"] = pd.cut(expected[0], 10).astype(
            str if not using_infer_string else "string[pyarrow_numpy]"
        )
        
        # 将 DataFrame 写入 Excel 文件 tmp_excel 中的 "test1" 工作表
        df.to_excel(tmp_excel, sheet_name="test1")
        
        # 使用 ExcelFile 打开 tmp_excel，读取 "test1" 工作表的内容到 recons DataFrame
        with ExcelFile(tmp_excel) as reader:
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0)
        
        # 断言 expected 和 recons 是否相等
        tm.assert_frame_equal(expected, recons)

    def test_to_excel_interval_labels(self, tmp_excel):
        # see gh-19242
        #
        # Test writing Interval with labels.
        
        # 创建一个包含随机整数数据的 DataFrame，数据范围在 [-10, 10] 之间
        df = DataFrame(
            np.random.default_rng(2).integers(-10, 10, size=(20, 1)), dtype=np.int64
        )
        expected = df.copy()  # 复制 DataFrame
        
        # 将 df 中的第一列数据分成 10 个区间，并用指定标签命名这些区间
        intervals = pd.cut(
            df[0], 10, labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        )
        df["new"] = intervals
        
        # 将 expected DataFrame 中的 "new" 列设为与 intervals 一致的 Series
        expected["new"] = pd.Series(list(intervals))
        
        # 将 DataFrame 写入 Excel 文件 tmp_excel 中的 "test1" 工作表
        df.to_excel(tmp_excel, sheet_name="test1")
        
        # 使用 ExcelFile 打开 tmp_excel，读取 "test1" 工作表的内容到 recons DataFrame
        with ExcelFile(tmp_excel) as reader:
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0)
        
        # 断言 expected 和 recons 是否相等
        tm.assert_frame_equal(expected, recons)

    def test_to_excel_timedelta(self, tmp_excel):
        # see gh-19242, gh-9155
        #
        # Test writing timedelta to xls.
        
        # 创建一个包含随机整数数据的 DataFrame，数据范围在 [-10, 10] 之间，列名为 "A"
        df = DataFrame(
            np.random.default_rng(2).integers(-10, 10, size=(20, 1)),
            columns=["A"],
            dtype=np.int64,
        )
        expected = df.copy()  # 复制 DataFrame
        
        # 将 df["A"] 列中的整数转换为 timedelta 类型的数据，存入 "new" 列
        df["new"] = df["A"].apply(lambda x: timedelta(seconds=x))
        
        # 将 expected["A"] 列中的整数转换为相应的秒数经过一天的 timedelta，存入 "new" 列
        expected["new"] = expected["A"].apply(
            lambda x: timedelta(seconds=x).total_seconds() / 86400
        )
        
        # 将 DataFrame 写入 Excel 文件 tmp_excel 中的 "test1" 工作表
        df.to_excel(tmp_excel, sheet_name="test1")
        
        # 使用 ExcelFile 打开 tmp_excel，读取 "test1" 工作表的内容到 recons DataFrame
        with ExcelFile(tmp_excel) as reader:
            recons = pd.read_excel(reader, sheet_name="test1", index_col=0)
        
        # 断言 expected 和 recons 是否相等
        tm.assert_frame_equal(expected, recons)

    def test_to_excel_periodindex(self, tmp_excel):
        # xp has a PeriodIndex
        
        # 创建一个包含标准正态分布数据的 DataFrame，行为日期范围，列为 ABCD
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )
        
        # 对 df 进行 "ME" 频率重采样，并转换为 PeriodIndex
        xp = df.resample("ME").mean().to_period("M")
        
        # 将 xp DataFrame 写入 Excel 文件 tmp_excel 中的 "sht1" 工作表
        xp.to_excel(tmp_excel, sheet_name="sht1")
        
        # 使用 ExcelFile 打开 tmp_excel，读取 "sht1" 工作表的内容到 rs DataFrame
        with ExcelFile(tmp_excel) as reader:
            rs = pd.read_excel(reader, sheet_name="sht1", index_col=0)
        
        # 断言 xp 和 rs（转换为 PeriodIndex 后）是否相等
        tm.assert_frame_equal(xp, rs.to_period("M"))
    # 定义一个测试方法，测试多级索引情况下的数据写入 Excel
    def test_to_excel_multiindex(self, merge_cells, frame, tmp_excel):
        # 创建一个二维数组作为新的多级索引
        arrays = np.arange(len(frame.index) * 2, dtype=np.int64).reshape(2, -1)
        # 从数组创建一个多级索引对象，指定索引的名称为 "first" 和 "second"
        new_index = MultiIndex.from_arrays(arrays, names=["first", "second"])
        # 将数据框的索引设置为新创建的多级索引
        frame.index = new_index

        # 将数据框的内容写入 Excel 文件中的 "test1" 工作表，不包括表头
        frame.to_excel(tmp_excel, sheet_name="test1", header=False)
        # 再次将数据框的内容写入 Excel 文件中的 "test1" 工作表，仅包括列 "A" 和 "B"
        frame.to_excel(tmp_excel, sheet_name="test1", columns=["A", "B"])

        # 再来一次往返写入
        # 将数据框的内容写入 Excel 文件中的 "test1" 工作表，根据 merge_cells 参数决定是否合并单元格
        frame.to_excel(tmp_excel, sheet_name="test1", merge_cells=merge_cells)
        # 使用 ExcelFile 对象读取刚刚写入的 Excel 文件
        with ExcelFile(tmp_excel) as reader:
            # 从 Excel 文件中读取数据，指定 "test1" 工作表为数据框的索引列
            df = pd.read_excel(reader, sheet_name="test1", index_col=[0, 1])
        # 使用断言比较写入前后的数据框是否相同
        tm.assert_frame_equal(frame, df)

    # GH13511
    # 定义一个测试方法，测试带有 NaN 标签的多级索引情况下的数据写入 Excel
    def test_to_excel_multiindex_nan_label(self, merge_cells, tmp_excel):
        # 创建一个数据框，包含带有 NaN 值的列 "A"、"B" 和随机生成的列 "C"
        df = DataFrame(
            {
                "A": [None, 2, 3],
                "B": [10, 20, 30],
                "C": np.random.default_rng(2).random(3),
            }
        )
        # 将数据框的索引设置为列 "A" 和 "B" 的多级索引
        df = df.set_index(["A", "B"])

        # 将数据框的内容写入 Excel 文件中，根据 merge_cells 参数决定是否合并单元格
        df.to_excel(tmp_excel, merge_cells=merge_cells)
        # 从 Excel 文件中读取数据，指定列 "A" 和 "B" 为数据框的索引列
        df1 = pd.read_excel(tmp_excel, index_col=[0, 1])
        # 使用断言比较写入前后的数据框是否相同
        tm.assert_frame_equal(df, df1)

    # Test for Issue 11328. If column indices are integers, make
    # sure they are handled correctly for either setting of
    # merge_cells
    # 定义一个测试方法，测试当列索引为整数时，无论 merge_cells 设置如何，都能正确处理
    def test_to_excel_multiindex_cols(self, merge_cells, frame, tmp_excel):
        # 创建一个二维数组作为新的多级索引
        arrays = np.arange(len(frame.index) * 2, dtype=np.int64).reshape(2, -1)
        # 从数组创建一个多级索引对象，指定索引的名称为 "first" 和 "second"
        new_index = MultiIndex.from_arrays(arrays, names=["first", "second"])
        # 将数据框的索引设置为新创建的多级索引
        frame.index = new_index

        # 创建一个新的列索引，包含元组对 (40, 1), (40, 2), (50, 1), (50, 2)
        new_cols_index = MultiIndex.from_tuples([(40, 1), (40, 2), (50, 1), (50, 2)])
        # 将数据框的列索引设置为新创建的多级索引
        frame.columns = new_cols_index
        # 初始化 header 变量为整数列表 [0, 1]，根据 merge_cells 参数决定是否重新赋值为 0
        header = [0, 1]
        if not merge_cells:
            header = 0

        # 再来一次往返写入
        # 将数据框的内容写入 Excel 文件中的 "test1" 工作表，根据 merge_cells 参数决定是否合并单元格
        frame.to_excel(tmp_excel, sheet_name="test1", merge_cells=merge_cells)
        # 使用 ExcelFile 对象读取刚刚写入的 Excel 文件
        with ExcelFile(tmp_excel) as reader:
            # 从 Excel 文件中读取数据，根据 header 参数确定是否使用多级列索引
            df = pd.read_excel(
                reader, sheet_name="test1", header=header, index_col=[0, 1]
            )
        # 如果 merge_cells 参数为 False，则需要重新格式化列索引
        if not merge_cells:
            fm = frame.columns._format_multi(sparsify=False, include_names=False)
            frame.columns = [".".join(map(str, q)) for q in zip(*fm)]
        # 使用断言比较写入前后的数据框是否相同
        tm.assert_frame_equal(frame, df)
    # 测试带有多级索引和日期的情况
    def test_to_excel_multiindex_dates(self, merge_cells, tmp_excel):
        # 从临时 Excel 文件中获取实验单元
        unit = get_exp_unit(tmp_excel)
        # 创建一个随机数据框，包含5行4列的标准正态分布随机数
        tsframe = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )
        # 将数据框的索引转换为多级索引
        tsframe.index = MultiIndex.from_arrays(
            [
                tsframe.index.as_unit(unit),  # 使用实验单元重新设定索引的第一级
                np.arange(len(tsframe.index), dtype=np.int64),  # 创建第二级索引
            ],
            names=["time", "foo"],  # 设置索引名称
        )

        # 将数据框写入到临时 Excel 文件的指定表单中，可以选择合并单元格
        tsframe.to_excel(tmp_excel, sheet_name="test1", merge_cells=merge_cells)
        # 使用 ExcelFile 对象读取临时 Excel 文件中的指定表单
        with ExcelFile(tmp_excel) as reader:
            # 重新读取 Excel 文件中的数据，并将其设定为多级索引
            recons = pd.read_excel(reader, sheet_name="test1", index_col=[0, 1])

        # 使用测试工具验证数据框与重新读取的数据框是否相等
        tm.assert_frame_equal(tsframe, recons)
        # 验证重新读取的数据框的索引名称是否与预期相符
        assert recons.index.names == ("time", "foo")

    # 测试写入和重新读取不带索引的多级索引数据
    def test_to_excel_multiindex_no_write_index(self, tmp_excel):
        # 初始非多级索引数据框
        frame1 = DataFrame({"a": [10, 20], "b": [30, 40], "c": [50, 60]})

        # 复制数据框并创建多级索引
        frame2 = frame1.copy()
        multi_index = MultiIndex.from_tuples([(70, 80), (90, 100)])
        frame2.index = multi_index

        # 将带有多级索引的数据框写入 Excel，不包含索引
        frame2.to_excel(tmp_excel, sheet_name="test1", index=False)

        # 重新读取 Excel 中的数据
        with ExcelFile(tmp_excel) as reader:
            frame3 = pd.read_excel(reader, sheet_name="test1")

        # 验证重新读取的数据框是否与初始数据框相等
        tm.assert_frame_equal(frame1, frame3)

    # 测试写入和重新读取空的多级索引数据框
    def test_to_excel_empty_multiindex(self, tmp_excel):
        # 创建预期结果为一个空的数据框
        expected = DataFrame([], columns=[0, 1, 2])

        # 创建一个空的多级索引数据框，并将其写入 Excel
        df = DataFrame([], index=MultiIndex.from_tuples([], names=[0, 1]), columns=[2])
        df.to_excel(tmp_excel, sheet_name="test1")

        # 从 Excel 文件中重新读取数据，并验证是否与预期相等
        with ExcelFile(tmp_excel) as reader:
            result = pd.read_excel(reader, sheet_name="test1")
        tm.assert_frame_equal(
            result, expected, check_index_type=False, check_dtype=False
        )

    # 测试写入和重新读取空的多级索引数据框，同时设置列和行索引
    def test_to_excel_empty_multiindex_both_axes(self, tmp_excel):
        # 创建一个空的多级索引数据框
        df = DataFrame(
            [],
            index=MultiIndex.from_tuples([], names=[0, 1]),
            columns=MultiIndex.from_tuples([("A", "B")]),
        )
        # 将数据框写入 Excel 文件中
        df.to_excel(tmp_excel)
        # 从 Excel 文件中读取数据，并指定列和行索引，验证是否与原数据框相等
        result = pd.read_excel(tmp_excel, header=[0, 1], index_col=[0, 1])
        tm.assert_frame_equal(result, df)
    def test_to_excel_float_format(self, tmp_excel):
        # 创建一个包含浮点数数据的 DataFrame
        df = DataFrame(
            [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            index=["A", "B"],
            columns=["X", "Y", "Z"],
        )
        # 将 DataFrame 写入 Excel 文件，设置浮点数格式为两位小数
        df.to_excel(tmp_excel, sheet_name="test1", float_format="%.2f")

        # 使用 ExcelFile 打开临时 Excel 文件并读取特定工作表到 DataFrame
        with ExcelFile(tmp_excel) as reader:
            result = pd.read_excel(reader, sheet_name="test1", index_col=0)

        # 创建期望的 DataFrame，保留两位小数的浮点数格式
        expected = DataFrame(
            [[0.12, 0.23, 0.57], [12.32, 123123.20, 321321.20]],
            index=["A", "B"],
            columns=["X", "Y", "Z"],
        )
        # 使用测试工具（tm.assert_frame_equal）比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_to_excel_output_encoding(self, tmp_excel):
        # 创建包含 Unicode 字符的 DataFrame
        df = DataFrame(
            [["\u0192", "\u0193", "\u0194"], ["\u0195", "\u0196", "\u0197"]],
            index=["A\u0192", "B"],
            columns=["X\u0193", "Y", "Z"],
        )

        # 将 DataFrame 写入 Excel 文件，不指定浮点数格式或工作表名称
        df.to_excel(tmp_excel, sheet_name="TestSheet")

        # 从临时 Excel 文件读取数据到 DataFrame，并与原始 DataFrame 比较
        result = pd.read_excel(tmp_excel, sheet_name="TestSheet", index_col=0)
        tm.assert_frame_equal(result, df)

    def test_to_excel_unicode_filename(self, ext, tmp_path):
        # 创建一个具有 Unicode 文件名的 Excel 文件
        filename = tmp_path / f"\u0192u.{ext}"
        filename.touch()

        # 创建包含浮点数数据的 DataFrame
        df = DataFrame(
            [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            index=["A", "B"],
            columns=["X", "Y", "Z"],
        )
        # 将 DataFrame 写入具有 Unicode 文件名的 Excel 文件，设置浮点数格式为两位小数
        df.to_excel(filename, sheet_name="test1", float_format="%.2f")

        # 使用 ExcelFile 打开具有 Unicode 文件名的 Excel 文件并读取特定工作表到 DataFrame
        with ExcelFile(filename) as reader:
            result = pd.read_excel(reader, sheet_name="test1", index_col=0)

        # 创建期望的 DataFrame，保留两位小数的浮点数格式
        expected = DataFrame(
            [[0.12, 0.23, 0.57], [12.32, 123123.20, 321321.20]],
            index=["A", "B"],
            columns=["X", "Y", "Z"],
        )
        # 使用测试工具（tm.assert_frame_equal）比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("use_headers", [True, False])
    @pytest.mark.parametrize("r_idx_nlevels", [1, 2, 3])
    @pytest.mark.parametrize("c_idx_nlevels", [1, 2, 3])
    def test_excel_010_hemstring(
        self, merge_cells, c_idx_nlevels, r_idx_nlevels, use_headers, tmp_excel
        ):
        # 参数化测试用例，具体测试逻辑在其他方法中实现
        ):
            # 定义一个内部函数 `roundtrip`，用于将数据写入临时 Excel 文件并进行读取
            def roundtrip(data, header=True, parser_hdr=0, index=True):
                # 将数据写入临时 Excel 文件
                data.to_excel(
                    tmp_excel, header=header, merge_cells=merge_cells, index=index
                )

                # 使用 `ExcelFile` 打开临时 Excel 文件
                with ExcelFile(tmp_excel) as xf:
                    # 读取 Excel 文件的第一个表格数据作为 DataFrame
                    return pd.read_excel(
                        xf, sheet_name=xf.sheet_names[0], header=parser_hdr
                    )

            # 基本测试
            # 根据是否使用标题确定解析器标题
            parser_header = 0 if use_headers else None
            # 调用 `roundtrip` 函数，将 DataFrame([0]) 传入，返回结果赋给 `res`
            res = roundtrip(DataFrame([0]), use_headers, parser_header)

            # 断言返回的 DataFrame 形状为 (1, 2)
            assert res.shape == (1, 2)
            # 断言返回的 DataFrame 的第一行第一列不是 NaN
            assert res.iloc[0, 0] is not np.nan

            # 更复杂的带多级索引的测试
            nrows = 5
            ncols = 3

            # 限制在版本 0.10 中的功能
            # 在版本 0.11 解决之前覆盖 gh-2370

            # 如果列的索引级别为 1
            if c_idx_nlevels == 1:
                # 创建一个包含对象类型列名的 Index
                columns = Index([f"a-{i}" for i in range(ncols)], dtype=object)
            else:
                # 使用多个数组创建 MultiIndex，指定每级别的名称
                columns = MultiIndex.from_arrays(
                    [range(ncols) for _ in range(c_idx_nlevels)],
                    names=[f"i-{i}" for i in range(c_idx_nlevels)],
                )

            # 如果行的索引级别为 1
            if r_idx_nlevels == 1:
                # 创建一个包含对象类型索引名的 Index
                index = Index([f"b-{i}" for i in range(nrows)], dtype=object)
            else:
                # 使用多个数组创建 MultiIndex，指定每级别的名称
                index = MultiIndex.from_arrays(
                    [range(nrows) for _ in range(r_idx_nlevels)],
                    names=[f"j-{i}" for i in range(r_idx_nlevels)],
                )

            # 创建一个包含全部为 1 的 DataFrame，指定列和行索引
            df = DataFrame(
                np.ones((nrows, ncols)),
                columns=columns,
                index=index,
            )

            # 如果列的索引级别大于 1，执行以下操作
            if c_idx_nlevels > 1:
                # 抛出未实现错误，匹配消息字符串 `msg`
                msg = (
                    "Writing to Excel with MultiIndex columns and no index "
                    "\\('index'=False\\) is not yet implemented."
                )
                with pytest.raises(NotImplementedError, match=msg):
                    # 调用 `roundtrip` 函数，预期引发 NotImplementError
                    roundtrip(df, use_headers, index=False)
            else:
                # 否则，调用 `roundtrip` 函数，传入 `df` 和 `use_headers`，返回结果赋给 `res`
                res = roundtrip(df, use_headers)

                # 如果使用标题
                if use_headers:
                    # 断言返回的 DataFrame 形状为 (nrows, ncols + r_idx_nlevels)
                    assert res.shape == (nrows, ncols + r_idx_nlevels)
                else:
                    # 否则，第一行作为列名，断言返回的 DataFrame 形状为 (nrows - 1, ncols + r_idx_nlevels)
                    assert res.shape == (nrows - 1, ncols + r_idx_nlevels)

                # 没有 NaN 值的断言
                for r in range(len(res.index)):
                    for c in range(len(res.columns)):
                        assert res.iloc[r, c] is not np.nan
    def test_duplicated_columns(self, tmp_excel):
        # see gh-5235
        # 创建一个 DataFrame，包含重复列名 "B"
        df = DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["A", "B", "B"])
        # 将 DataFrame 写入 Excel 文件
        df.to_excel(tmp_excel, sheet_name="test1")
        # 创建期望的 DataFrame，自动解决重复列名问题，改为 "B.1"
        expected = DataFrame(
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["A", "B", "B.1"]
        )

        # 默认情况下进行列名重命名
        # 从 Excel 文件读取数据作为结果
        result = pd.read_excel(tmp_excel, sheet_name="test1", index_col=0)
        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)

        # see gh-11007, gh-10970
        # 创建一个 DataFrame，包含交换列名 "A" 和 "B" 的情况
        df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["A", "B", "A", "B"])
        # 将 DataFrame 写入 Excel 文件
        df.to_excel(tmp_excel, sheet_name="test1")

        # 从 Excel 文件读取数据作为结果
        result = pd.read_excel(tmp_excel, sheet_name="test1", index_col=0)
        # 创建期望的 DataFrame，自动解决重复列名问题，改为 "A.1" 和 "B.1"
        expected = DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]], columns=["A", "B", "A.1", "B.1"]
        )
        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)

        # see gh-10982
        # 将 DataFrame 写入 Excel 文件，不包含索引和列名
        df.to_excel(tmp_excel, sheet_name="test1", index=False, header=False)
        # 从 Excel 文件读取数据作为结果，不使用头部行作为列名
        result = pd.read_excel(tmp_excel, sheet_name="test1", header=None)

        # 创建期望的 DataFrame
        expected = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    def test_swapped_columns(self, tmp_excel):
        # Test for issue #5427.
        # 创建一个 DataFrame，将列 "A" 和 "B" 交换后写入 Excel 文件
        write_frame = DataFrame({"A": [1, 1, 1], "B": [2, 2, 2]})
        write_frame.to_excel(tmp_excel, sheet_name="test1", columns=["B", "A"])

        # 从 Excel 文件读取数据作为结果
        read_frame = pd.read_excel(tmp_excel, sheet_name="test1", header=0)

        # 断言读取结果中的列 "A" 与写入时的列 "A" 相等
        tm.assert_series_equal(write_frame["A"], read_frame["A"])
        # 断言读取结果中的列 "B" 与写入时的列 "B" 相等
        tm.assert_series_equal(write_frame["B"], read_frame["B"])

    def test_invalid_columns(self, tmp_excel):
        # see gh-10982
        # 创建一个 DataFrame，尝试写入不存在的列名 "C"
        write_frame = DataFrame({"A": [1, 1, 1], "B": [2, 2, 2]})

        # 断言写入过程中会抛出 KeyError 异常，匹配错误信息 "Not all names specified"
        with pytest.raises(KeyError, match="Not all names specified"):
            write_frame.to_excel(tmp_excel, sheet_name="test1", columns=["B", "C"])

        # 断言写入过程中会抛出 KeyError 异常，匹配错误信息 "'passes columns are not ALL present dataframe'"
        with pytest.raises(
            KeyError, match="'passes columns are not ALL present dataframe'"
        ):
            write_frame.to_excel(tmp_excel, sheet_name="test1", columns=["C", "D"])

    @pytest.mark.parametrize(
        "to_excel_index,read_excel_index_col",
        [
            (True, 0),  # 在写入文件时包含索引
            (False, None),  # 在写入文件时不包含索引
        ],
    )
    def test_write_subset_columns(
        self, tmp_excel, to_excel_index, read_excel_index_col
    ):
        # GH 31677
        # 创建一个 DataFrame，仅写入部分列 "A" 和 "B" 到 Excel 文件
        write_frame = DataFrame({"A": [1, 1, 1], "B": [2, 2, 2], "C": [3, 3, 3]})
        write_frame.to_excel(
            tmp_excel,
            sheet_name="col_subset_bug",
            columns=["A", "B"],
            index=to_excel_index,
        )

        # 从 Excel 文件读取部分列 "A" 和 "B" 的数据作为期望结果
        expected = write_frame[["A", "B"]]
        read_frame = pd.read_excel(
            tmp_excel, sheet_name="col_subset_bug", index_col=read_excel_index_col
        )

        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(expected, read_frame)
    def test_comment_arg(self, tmp_excel):
        # see gh-18735
        #
        # Test the comment argument functionality to pd.read_excel.

        # 创建要读取的文件。
        df = DataFrame({"A": ["one", "#one", "one"], "B": ["two", "two", "#two"]})
        df.to_excel(tmp_excel, sheet_name="test_c")

        # 读取文件，不使用 comment 参数。
        result1 = pd.read_excel(tmp_excel, sheet_name="test_c", index_col=0)

        # 修改结果中的部分数据为 None。
        result1.iloc[1, 0] = None
        result1.iloc[1, 1] = None
        result1.iloc[2, 1] = None

        # 使用 comment 参数读取文件。
        result2 = pd.read_excel(
            tmp_excel, sheet_name="test_c", comment="#", index_col=0
        )
        # 比较两次读取的结果是否相等。
        tm.assert_frame_equal(result1, result2)

    def test_comment_default(self, tmp_excel):
        # Re issue #18735
        # Test the comment argument default to pd.read_excel

        # 创建要读取的文件。
        df = DataFrame({"A": ["one", "#one", "one"], "B": ["two", "two", "#two"]})
        df.to_excel(tmp_excel, sheet_name="test_c")

        # 使用默认的 comment 参数和显式指定 comment=None 读取文件。
        result1 = pd.read_excel(tmp_excel, sheet_name="test_c")
        result2 = pd.read_excel(tmp_excel, sheet_name="test_c", comment=None)
        # 比较两次读取的结果是否相等。
        tm.assert_frame_equal(result1, result2)

    def test_comment_used(self, tmp_excel):
        # see gh-18735
        #
        # Test the comment argument is working as expected when used.

        # 创建要读取的文件。
        df = DataFrame({"A": ["one", "#one", "one"], "B": ["two", "two", "#two"]})
        df.to_excel(tmp_excel, sheet_name="test_c")

        # 使用 comment 参数读取文件，并与手动生成的预期输出比较。
        expected = DataFrame({"A": ["one", None, "one"], "B": ["two", None, None]})
        result = pd.read_excel(tmp_excel, sheet_name="test_c", comment="#", index_col=0)
        # 比较读取的结果是否与预期相等。
        tm.assert_frame_equal(result, expected)

    def test_comment_empty_line(self, tmp_excel):
        # Re issue #18735
        # Test that pd.read_excel ignores commented lines at the end of file

        # 创建要写入的文件。
        df = DataFrame({"a": ["1", "#2"], "b": ["2", "3"]})
        df.to_excel(tmp_excel, index=False)

        # 测试 pd.read_excel 是否忽略文件末尾的注释行。
        expected = DataFrame({"a": [1], "b": [2]})
        result = pd.read_excel(tmp_excel, comment="#")
        # 比较读取的结果是否与预期相等。
        tm.assert_frame_equal(result, expected)
    def test_datetimes(self, tmp_excel):
        # 测试写入和读取日期时间数据，针对问题 #9139。（参考 #9185）
        # 获取日期时间单位
        unit = get_exp_unit(tmp_excel)
        # 创建一个日期时间列表
        datetimes = [
            datetime(2013, 1, 13, 1, 2, 3),
            datetime(2013, 1, 13, 2, 45, 56),
            datetime(2013, 1, 13, 4, 29, 49),
            datetime(2013, 1, 13, 6, 13, 42),
            datetime(2013, 1, 13, 7, 57, 35),
            datetime(2013, 1, 13, 9, 41, 28),
            datetime(2013, 1, 13, 11, 25, 21),
            datetime(2013, 1, 13, 13, 9, 14),
            datetime(2013, 1, 13, 14, 53, 7),
            datetime(2013, 1, 13, 16, 37, 0),
            datetime(2013, 1, 13, 18, 20, 52),
        ]

        # 创建一个 DataFrame 包含日期时间数据列 "A"
        write_frame = DataFrame({"A": datetimes})
        # 将 DataFrame 写入到 Excel 文件中
        write_frame.to_excel(tmp_excel, sheet_name="Sheet1")
        # 从 Excel 文件中读取数据到 DataFrame
        read_frame = pd.read_excel(tmp_excel, sheet_name="Sheet1", header=0)

        # 期望的数据类型转换为特定日期时间单位
        expected = write_frame.astype(f"M8[{unit}]")
        # 断言写入的数据列 "A" 和读取的数据列 "A" 相等
        tm.assert_series_equal(expected["A"], read_frame["A"])

    def test_bytes_io(self, engine):
        # 见问题 gh-7074
        # 使用 BytesIO 来模拟文件操作
        with BytesIO() as bio:
            # 创建一个随机数据的 DataFrame
            df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))

            # 显式传递 engine 参数，因为没有文件路径可以推断
            with ExcelWriter(bio, engine=engine) as writer:
                # 将 DataFrame 写入 ExcelWriter 对象中
                df.to_excel(writer)

            # 将指针移到文件开头
            bio.seek(0)
            # 从 BytesIO 中重新读取 Excel 数据到 DataFrame
            reread_df = pd.read_excel(bio, index_col=0)
            # 断言原始 DataFrame 和重新读取的 DataFrame 相等
            tm.assert_frame_equal(df, reread_df)

    def test_engine_kwargs(self, engine, tmp_excel):
        # GH#52368
        # 创建一个包含字典元素的 DataFrame
        df = DataFrame([{"A": 1, "B": 2}, {"A": 3, "B": 4}])

        # 不同引擎的错误消息定义
        msgs = {
            "odf": r"OpenDocumentSpreadsheet() got an unexpected keyword "
            r"argument 'foo'",
            "openpyxl": r"Workbook.__init__() got an unexpected keyword argument 'foo'",
            "xlsxwriter": r"Workbook.__init__() got an unexpected keyword argument 'foo'",
        }

        # 处理 openpyxl 引擎错误消息的变化（写入和追加模式）
        if engine == "openpyxl" and not os.path.exists(tmp_excel):
            msgs["openpyxl"] = (
                r"load_workbook() got an unexpected keyword argument 'foo'"
            )

        # 使用 pytest 来断言特定引擎会抛出预期的 TypeError 异常
        with pytest.raises(TypeError, match=re.escape(msgs[engine])):
            df.to_excel(
                tmp_excel,
                engine=engine,
                engine_kwargs={"foo": "bar"},
            )
    # 测试函数，用于验证将 DataFrame 写入 Excel 并从 Excel 读取的功能
    def test_write_lists_dict(self, tmp_excel):
        # 创建一个包含不同数据类型的 DataFrame
        df = DataFrame(
            {
                "mixed": ["a", ["b", "c"], {"d": "e", "f": 2}],
                "numeric": [1, 2, 3.0],
                "str": ["apple", "banana", "cherry"],
            }
        )
        # 将 DataFrame 写入 Excel 文件的指定工作表
        df.to_excel(tmp_excel, sheet_name="Sheet1")
        # 从 Excel 读取指定工作表的数据到 DataFrame
        read = pd.read_excel(tmp_excel, sheet_name="Sheet1", header=0, index_col=0)

        # 创建预期的 DataFrame 副本
        expected = df.copy()
        # 将混合列转换为字符串
        expected.mixed = expected.mixed.apply(str)
        # 将数字列转换为 int64 类型
        expected.numeric = expected.numeric.astype("int64")

        # 断言读取的 DataFrame 和预期的 DataFrame 是否相等
        tm.assert_frame_equal(read, expected)

    # 测试函数，验证将 DataFrame 写入 Excel 并从 Excel 读取的功能
    def test_render_as_column_name(self, tmp_excel):
        # 创建一个包含特定列名的 DataFrame
        df = DataFrame({"render": [1, 2], "data": [3, 4]})
        # 将 DataFrame 写入 Excel 文件的指定工作表
        df.to_excel(tmp_excel, sheet_name="Sheet1")
        # 从 Excel 读取指定工作表的数据到 DataFrame
        read = pd.read_excel(tmp_excel, "Sheet1", index_col=0)

        # 预期读取的结果应与原始 DataFrame 相等
        expected = df
        tm.assert_frame_equal(read, expected)

    # 测试函数，验证在读取 Excel 时设置 true 和 false 值的选项
    def test_true_and_false_value_options(self, tmp_excel):
        # 创建一个包含对象类型数据的 DataFrame
        df = DataFrame([["foo", "bar"]], columns=["col1", "col2"], dtype=object)
        # 使用上下文管理器设置选项以确保不进行静默转换
        with option_context("future.no_silent_downcasting", True):
            # 预期的 DataFrame 将字符串 "foo" 转换为 True，"bar" 转换为 False，并转换为布尔型
            expected = df.replace({"foo": True, "bar": False}).astype("bool")

        # 将 DataFrame 写入 Excel 文件
        df.to_excel(tmp_excel)
        # 从 Excel 读取数据到 DataFrame，设置 true 和 false 值的映射
        read_frame = pd.read_excel(
            tmp_excel, true_values=["foo"], false_values=["bar"], index_col=0
        )
        tm.assert_frame_equal(read_frame, expected)

    # 测试函数，验证在写入 Excel 时设置冻结窗格的功能
    def test_freeze_panes(self, tmp_excel):
        # 创建一个包含两列数据的 DataFrame
        expected = DataFrame([[1, 2], [3, 4]], columns=["col1", "col2"])
        # 将 DataFrame 写入 Excel 文件的指定工作表，并设置冻结窗格在第一行第一列
        expected.to_excel(tmp_excel, sheet_name="Sheet1", freeze_panes=(1, 1))

        # 从 Excel 读取指定工作表的数据到 DataFrame
        result = pd.read_excel(tmp_excel, index_col=0)
        tm.assert_frame_equal(result, expected)

    # 测试函数，验证使用部分路径函数写入和读取 Excel 的功能
    def test_path_path_lib(self, engine, ext):
        # 创建一个包含特定数据的 DataFrame
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 创建部分函数用于将 DataFrame 写入 Excel 文件
        writer = partial(df.to_excel, engine=engine)

        # 创建部分函数用于从 Excel 读取数据到 DataFrame
        reader = partial(pd.read_excel, index_col=0)
        # 执行路径函数往返测试，将 DataFrame 写入和从 Excel 读取，结果应与原始 DataFrame 相等
        result = tm.round_trip_pathlib(writer, reader, path=f"foo{ext}")
        tm.assert_frame_equal(result, df)

    # 测试函数，验证在处理合并单元格和自定义对象时的功能
    def test_merged_cell_custom_objects(self, tmp_excel):
        # 创建一个包含 MultiIndex 的 DataFrame
        mi = MultiIndex.from_tuples(
            [
                (pd.Period("2018"), pd.Period("2018Q1")),
                (pd.Period("2018"), pd.Period("2018Q2")),
            ]
        )
        expected = DataFrame(np.ones((2, 2), dtype="int64"), columns=mi)
        # 将 DataFrame 写入 Excel 文件
        expected.to_excel(tmp_excel)
        # 从 Excel 读取数据到 DataFrame，设置行和列头
        result = pd.read_excel(tmp_excel, header=[0, 1], index_col=0)
        # 需要将 PeriodIndexes 转换为标准 Indexes 以便进行相等断言
        expected.columns = expected.columns.set_levels(
            [[str(i) for i in mi.levels[0]], [str(i) for i in mi.levels[1]]],
            level=[0, 1],
        )
        tm.assert_frame_equal(result, expected)
    @pytest.mark.parametrize("dtype", [None, object])
    def test_raise_when_saving_timezones(self, dtype, tz_aware_fixture, tmp_excel):
        # 标记该测试函数参数化，测试不同的数据类型（None 或 object）
        # GH 27008, GH 7056
        # 获取时区感知对象
        tz = tz_aware_fixture
        # 创建带有时区信息的时间戳数据
        data = pd.Timestamp("2019", tz=tz)
        # 使用给定的数据类型创建数据帧
        df = DataFrame([data], dtype=dtype)
        # 断言在保存到 Excel 时会引发 ValueError，并匹配指定的错误信息
        with pytest.raises(ValueError, match="Excel does not support"):
            df.to_excel(tmp_excel)

        # 将时间戳转换为 Python 的 datetime 对象
        data = data.to_pydatetime()
        # 使用给定的数据类型创建数据帧
        df = DataFrame([data], dtype=dtype)
        # 断言在保存到 Excel 时会引发 ValueError，并匹配指定的错误信息
        with pytest.raises(ValueError, match="Excel does not support"):
            df.to_excel(tmp_excel)

    def test_excel_duplicate_columns_with_names(self, tmp_excel):
        # GH#39695
        # 创建包含重复列名的数据帧
        df = DataFrame({"A": [0, 1], "B": [10, 11]})
        # 将数据帧保存为 Excel，指定列名
        df.to_excel(tmp_excel, columns=["A", "B", "A"], index=False)

        # 从保存的 Excel 中读取数据
        result = pd.read_excel(tmp_excel)
        # 创建预期的数据帧，处理重复列名
        expected = DataFrame([[0, 10, 0], [1, 11, 1]], columns=["A", "B", "A.1"])
        # 断言读取的结果与预期的数据帧相等
        tm.assert_frame_equal(result, expected)

    def test_if_sheet_exists_raises(self, tmp_excel):
        # GH 40230
        # 定义错误信息
        msg = "if_sheet_exists is only valid in append mode (mode='a')"

        # 使用 pytest 断言，在创建 ExcelWriter 时应该引发 ValueError，并匹配指定的错误信息
        with pytest.raises(ValueError, match=re.escape(msg)):
            ExcelWriter(tmp_excel, if_sheet_exists="replace")

    def test_excel_writer_empty_frame(self, engine, tmp_excel):
        # GH#45793
        # 使用引擎创建 ExcelWriter 对象，将空数据帧保存到 Excel
        with ExcelWriter(tmp_excel, engine=engine) as writer:
            DataFrame().to_excel(writer)
        # 从保存的 Excel 中读取数据
        result = pd.read_excel(tmp_excel)
        # 创建预期的空数据帧
        expected = DataFrame()
        # 断言读取的结果与预期的数据帧相等
        tm.assert_frame_equal(result, expected)

    def test_to_excel_empty_frame(self, engine, tmp_excel):
        # GH#45793
        # 将空数据帧保存到 Excel，使用指定的引擎
        DataFrame().to_excel(tmp_excel, engine=engine)
        # 从保存的 Excel 中读取数据
        result = pd.read_excel(tmp_excel)
        # 创建预期的空数据帧
        expected = DataFrame()
        # 断言读取的结果与预期的数据帧相等
        tm.assert_frame_equal(result, expected)

    def test_to_excel_raising_warning_when_cell_character_exceed_limit(self):
        # GH#56954
        # 创建包含超过限制字符的数据帧
        df = DataFrame({"A": ["a" * 32768]})
        # 定义警告信息的正则表达式
        msg = r"Cell contents too long \(32768\), truncated to 32767 characters"
        # 使用 pytest 断言，执行操作时应产生 UserWarning，并匹配指定的警告信息
        with tm.assert_produces_warning(
            UserWarning, match=msg, raise_on_extra_warnings=False
        ):
            # 创建字节流缓冲区
            buf = BytesIO()
            # 将数据帧保存到 Excel 中
            df.to_excel(buf)
# 定义一个测试类 TestExcelWriterEngineTests
class TestExcelWriterEngineTests:
    
    # 使用 pytest 的参数化装饰器，为后续的测试方法提供参数
    @pytest.mark.parametrize(
        "klass,ext",
        [
            # 参数化测试用例，使用 _XlsxWriter 类和 .xlsx 扩展名，添加一个标记以跳过测试（如果没有安装 xlsxwriter）
            pytest.param(_XlsxWriter, ".xlsx", marks=td.skip_if_no("xlsxwriter")),
            # 参数化测试用例，使用 _OpenpyxlWriter 类和 .xlsx 扩展名，添加一个标记以跳过测试（如果没有安装 openpyxl）
            pytest.param(_OpenpyxlWriter, ".xlsx", marks=td.skip_if_no("openpyxl")),
        ],
    )
    # 定义测试方法 test_ExcelWriter_dispatch，接受 klass、ext 和 tmp_excel 参数
    def test_ExcelWriter_dispatch(self, klass, ext, tmp_excel):
        # 在 ExcelWriter 对象的上下文中，使用 tmp_excel 创建一个 writer 对象
        with ExcelWriter(tmp_excel) as writer:
            # 如果 ext 是 .xlsx 并且检测到安装了 xlsxwriter
            if ext == ".xlsx" and bool(
                import_optional_dependency("xlsxwriter", errors="ignore")
            ):
                # 断言 writer 是 _XlsxWriter 类的实例
                assert isinstance(writer, _XlsxWriter)
            else:
                # 断言 writer 是 klass 类的实例（_XlsxWriter 或 _OpenpyxlWriter）
                assert isinstance(writer, klass)

    # 定义测试方法 test_ExcelWriter_dispatch_raises
    def test_ExcelWriter_dispatch_raises(self):
        # 在 ExcelWriter 对象的上下文中，使用文件名为 "nothing" 的 ExcelWriter 引发 ValueError 异常，异常信息包含 "No engine"
        with pytest.raises(ValueError, match="No engine"):
            ExcelWriter("nothing")

    # 定义测试方法 test_register_writer，接受 tmp_path 参数
    def test_register_writer(self, tmp_path):
        # 定义 DummyClass 类，继承自 ExcelWriter
        class DummyClass(ExcelWriter):
            # 设置类变量
            called_save = False
            called_write_cells = False
            called_sheets = False
            _supported_extensions = ("xlsx", "xls")
            _engine = "dummy"

            # 定义 book 方法，未实现具体功能
            def book(self):
                pass

            # 定义 _save 方法，将 called_save 设置为 True
            def _save(self):
                type(self).called_save = True

            # 定义 _write_cells 方法，将 called_write_cells 设置为 True
            def _write_cells(self, *args, **kwargs):
                type(self).called_write_cells = True

            # 定义 sheets 属性，将 called_sheets 设置为 True
            @property
            def sheets(self):
                type(self).called_sheets = True

            # 定义类方法 assert_called_and_reset，断言 called_save 为 True，called_write_cells 为 True，called_sheets 为 False，并将它们重置为 False
            @classmethod
            def assert_called_and_reset(cls):
                assert cls.called_save
                assert cls.called_write_cells
                assert not cls.called_sheets
                cls.called_save = False
                cls.called_write_cells = False

        # 将 DummyClass 注册为 ExcelWriter 的一个写入器
        register_writer(DummyClass)

        # 在选项上下文中，设置 "io.excel.xlsx.writer" 为 "dummy"
        with option_context("io.excel.xlsx.writer", "dummy"):
            # 创建文件路径为 tmp_path / "something.xlsx" 的 ExcelWriter 对象，并使用 writer 进行断言
            filepath = tmp_path / "something.xlsx"
            filepath.touch()
            with ExcelWriter(filepath) as writer:
                assert isinstance(writer, DummyClass)
            
            # 创建一个 DataFrame df，写入到 filepath 对应的 Excel 文件中，然后调用 DummyClass 的 assert_called_and_reset 类方法
            df = DataFrame(
                ["a"],
                columns=Index(["b"], name="foo"),
                index=Index(["c"], name="bar"),
            )
            df.to_excel(filepath)
            DummyClass.assert_called_and_reset()

        # 创建文件路径为 tmp_path / "something2.xlsx" 的 ExcelWriter 对象，并使用 writer 进行断言
        filepath2 = tmp_path / "something2.xlsx"
        filepath2.touch()
        df.to_excel(filepath2, engine="dummy")
        DummyClass.assert_called_and_reset()


# 使用 td.skip_if_no 装饰器，如果没有安装 "xlrd" 或 "openpyxl" 则跳过测试
@td.skip_if_no("xlrd")
@td.skip_if_no("openpyxl")
# 定义测试类 TestFSPath
class TestFSPath:
    # 定义测试方法 test_excelfile_fspath，接受 tmp_path 参数
    def test_excelfile_fspath(self, tmp_path):
        # 创建文件路径为 tmp_path / "foo.xlsx" 的 Excel 文件，并将 DataFrame df 写入到该文件中
        path = tmp_path / "foo.xlsx"
        path.touch()
        df = DataFrame({"A": [1, 2]})
        df.to_excel(path)
        
        # 在 ExcelFile 对象的上下文中，将 os.fspath(xl) 的结果与 path 的字符串形式进行断言
        with ExcelFile(path) as xl:
            result = os.fspath(xl)
        assert result == str(path)
    # 定义一个测试方法，测试 ExcelWriter 对象的文件路径操作
    def test_excelwriter_fspath(self, tmp_path):
        # 创建一个临时文件路径对象，指向 "foo.xlsx"
        path = tmp_path / "foo.xlsx"
        # 创建实际文件 "foo.xlsx"
        path.touch()
        # 使用 ExcelWriter 打开文件路径 path
        with ExcelWriter(path) as writer:
            # 断言 ExcelWriter 对象的文件路径转换为字符串后与 path 的字符串形式相等
            assert os.fspath(writer) == str(path)
@pytest.mark.parametrize("klass", _writers.values())
def test_subclass_attr(klass):
    # 使用 pytest 的参数化装饰器，循环测试所有 ExcelWriter 的子类
    # 测试条件：确保 ExcelWriter 的子类没有公共属性（issue 49602）
    
    # 获取 ExcelWriter 类的所有非私有属性名集合
    attrs_base = {name for name in dir(ExcelWriter) if not name.startswith("_")}
    
    # 获取当前子类 klass 的所有非私有属性名集合
    attrs_klass = {name for name in dir(klass) if not name.startswith("_")}
    
    # 断言：ExcelWriter 类的属性与当前子类的属性集合应完全一致
    assert not attrs_base.symmetric_difference(attrs_klass)
```