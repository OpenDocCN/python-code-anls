# `D:\src\scipysrc\pandas\pandas\tests\io\sas\test_sas7bdat.py`

```
# 导入必要的模块和库
import contextlib
from datetime import datetime
import io
import os
from pathlib import Path

import numpy as np
import pytest

# 导入 pandas 库及其相关模块和类
import pandas as pd
import pandas._testing as tm

# 导入 SAS 数据读取相关模块
from pandas.io.sas.sas7bdat import SAS7BDATReader


# 创建一个 pytest 的 fixture，用于返回数据路径
@pytest.fixture
def dirpath(datapath):
    return datapath("io", "sas", "data")


# 创建一个 pytest 的参数化 fixture，返回测试数据和索引
@pytest.fixture(params=[(1, range(1, 16)), (2, [16])])
def data_test_ix(request, dirpath):
    i, test_ix = request.param
    # 构建文件名
    fname = os.path.join(dirpath, f"test_sas7bdat_{i}.csv")
    # 从 CSV 文件中读取数据到 DataFrame
    df = pd.read_csv(fname)
    # 定义时间戳的起始日期
    epoch = datetime(1960, 1, 1)
    # 将 Column4 的时间间隔转换为日期
    t1 = pd.to_timedelta(df["Column4"], unit="D")
    df["Column4"] = (epoch + t1).astype("M8[s]")
    # 将 Column12 的时间间隔转换为日期
    t2 = pd.to_timedelta(df["Column12"], unit="D")
    df["Column12"] = (epoch + t2).astype("M8[s]")
    # 将所有 np.int64 类型的列转换为 np.float64
    for k in range(df.shape[1]):
        col = df.iloc[:, k]
        if col.dtype == np.int64:
            df.isetitem(k, df.iloc[:, k].astype(np.float64))
    # 返回处理后的 DataFrame 和测试索引
    return df, test_ix


# 定义一个测试类 TestSAS7BDAT
# 这个类用于测试从 SAS7BDAT 文件读取数据的各种情况
class TestSAS7BDAT:
    
    # 标记为 pytest 的慢速测试
    @pytest.mark.slow
    # 测试从文件中读取 SAS7BDAT 数据
    def test_from_file(self, dirpath, data_test_ix):
        expected, test_ix = data_test_ix
        for k in test_ix:
            # 构建文件路径
            fname = os.path.join(dirpath, f"test{k}.sas7bdat")
            # 从 SAS7BDAT 文件中读取数据到 DataFrame
            df = pd.read_sas(fname, encoding="utf-8")
            # 断言读取的 DataFrame 与期望的 DataFrame 相等
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    # 测试从缓冲区中读取 SAS7BDAT 数据
    def test_from_buffer(self, dirpath, data_test_ix):
        expected, test_ix = data_test_ix
        for k in test_ix:
            # 构建文件路径
            fname = os.path.join(dirpath, f"test{k}.sas7bdat")
            # 打开文件并读取其二进制内容
            with open(fname, "rb") as f:
                byts = f.read()
            # 将二进制内容封装成 BytesIO 对象
            buf = io.BytesIO(byts)
            # 使用 pd.read_sas 从缓冲区读取数据到 DataFrame
            with pd.read_sas(
                buf, format="sas7bdat", iterator=True, encoding="utf-8"
            ) as rdr:
                df = rdr.read()
            # 断言读取的 DataFrame 与期望的 DataFrame 相等
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    # 测试使用迭代器从 SAS7BDAT 文件中读取数据
    def test_from_iterator(self, dirpath, data_test_ix):
        expected, test_ix = data_test_ix
        for k in test_ix:
            # 构建文件路径
            fname = os.path.join(dirpath, f"test{k}.sas7bdat")
            # 使用 pd.read_sas 的迭代器模式读取部分数据到 DataFrame
            with pd.read_sas(fname, iterator=True, encoding="utf-8") as rdr:
                df = rdr.read(2)
                # 断言读取的部分 DataFrame 与期望的部分 DataFrame 相等
                tm.assert_frame_equal(df, expected.iloc[0:2, :])
                df = rdr.read(3)
                # 断言读取的部分 DataFrame 与期望的部分 DataFrame 相等
                tm.assert_frame_equal(df, expected.iloc[2:5, :])

    @pytest.mark.slow
    # 测试使用 Pathlib.Path 对象从 SAS7BDAT 文件中读取数据
    def test_path_pathlib(self, dirpath, data_test_ix):
        expected, test_ix = data_test_ix
        for k in test_ix:
            # 构建文件路径
            fname = Path(os.path.join(dirpath, f"test{k}.sas7bdat"))
            # 使用 pd.read_sas 从 Pathlib.Path 对象读取数据到 DataFrame
            df = pd.read_sas(fname, encoding="utf-8")
            # 断言读取的 DataFrame 与期望的 DataFrame 相等
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    # 参数化测试：测试不同的 chunksize 和文件索引 k 的组合
    @pytest.mark.parametrize("chunksize", (3, 5, 10, 11))
    @pytest.mark.parametrize("k", range(1, 17))
    # 定义一个测试函数，用于迭代读取指定路径下的 SAS 数据集文件，并统计数据行数
    def test_iterator_loop(self, dirpath, k, chunksize):
        # 构建要读取的文件路径
        fname = os.path.join(dirpath, f"test{k}.sas7bdat")
        # 使用 pandas 的 read_sas 函数以迭代器方式读取 SAS 数据集文件
        with pd.read_sas(fname, chunksize=chunksize, encoding="utf-8") as rdr:
            y = 0
            # 迭代读取每个 chunk
            for x in rdr:
                # 累加当前 chunk 的行数到 y 变量
                y += x.shape[0]
        # 断言最终累加的行数等于 rdr 对象中的行数
        assert y == rdr.row_count

    # 定义另一个测试函数，用于验证在使用迭代方式读取时是否会读取超过文件中的行数
    def test_iterator_read_too_much(self, dirpath):
        # 构建要读取的文件路径
        fname = os.path.join(dirpath, "test1.sas7bdat")
        # 第一次使用迭代方式读取，尝试读取比文件行数多 20 行的数据
        with pd.read_sas(
            fname, format="sas7bdat", iterator=True, encoding="utf-8"
        ) as rdr:
            d1 = rdr.read(rdr.row_count + 20)

        # 第二次使用迭代方式读取，同样尝试读取比文件行数多 20 行的数据
        with pd.read_sas(fname, iterator=True, encoding="utf-8") as rdr:
            d2 = rdr.read(rdr.row_count + 20)
        # 使用测试工具比较两次读取结果是否相等
        tm.assert_frame_equal(d1, d2)
# 定义函数以测试不同的编码选项
def test_encoding_options(datapath):
    # 拼接数据路径和文件名，获取 SAS 数据文件的完整路径
    fname = datapath("io", "sas", "data", "test1.sas7bdat")
    
    # 使用默认编码读取 SAS 数据文件，生成 DataFrame df1
    df1 = pd.read_sas(fname)
    
    # 使用 UTF-8 编码读取 SAS 数据文件，生成 DataFrame df2
    df2 = pd.read_sas(fname, encoding="utf-8")
    
    # 尝试将 df1 中每列数据解码为 UTF-8 编码（如果列是字符串类型）
    for col in df1.columns:
        try:
            df1[col] = df1[col].str.decode("utf-8")
        except AttributeError:
            pass
    
    # 断言两个 DataFrame df1 和 df2 相等
    tm.assert_frame_equal(df1, df2)
    
    # 使用 SAS7BDATReader 打开 SAS 数据文件，关闭时确保不转换头部文本
    with contextlib.closing(SAS7BDATReader(fname, convert_header_text=False)) as rdr:
        # 使用 SAS7BDATReader 读取数据，生成 DataFrame df3
        df3 = rdr.read()
    
    # 逐列断言 df1 和 df3 的列名相等（需解码）
    for x, y in zip(df1.columns, df3.columns):
        assert x == y.decode()


# 定义函数以测试推断编码选项
def test_encoding_infer(datapath):
    # 拼接数据路径和文件名，获取 SAS 数据文件的完整路径
    fname = datapath("io", "sas", "data", "test1.sas7bdat")
    
    # 使用推断编码打开 SAS 数据文件的迭代器，确保编码为 cp1252
    with pd.read_sas(fname, encoding="infer", iterator=True) as df1_reader:
        # 断言推断的编码是否为 cp1252
        assert df1_reader.inferred_encoding == "cp1252"
        # 从迭代器中读取数据，生成 DataFrame df1
        df1 = df1_reader.read()
    
    # 使用 cp1252 编码打开 SAS 数据文件的迭代器，生成 DataFrame df2
    with pd.read_sas(fname, encoding="cp1252", iterator=True) as df2_reader:
        df2 = df2_reader.read()
    
    # 断言两个 DataFrame df1 和 df2 相等
    tm.assert_frame_equal(df1, df2)


# 定义函数以测试产品销售数据的读取
def test_productsales(datapath):
    # 拼接数据路径和文件名，获取产品销售数据的 SAS 数据文件的完整路径
    fname = datapath("io", "sas", "data", "productsales.sas7bdat")
    
    # 使用 UTF-8 编码读取 SAS 数据文件，生成 DataFrame df
    df = pd.read_sas(fname, encoding="utf-8")
    
    # 拼接数据路径和文件名，获取产品销售数据的 CSV 文件的完整路径
    fname = datapath("io", "sas", "data", "productsales.csv")
    
    # 使用默认设置读取 CSV 文件，指定 MONTH 列为日期类型，生成 DataFrame df0
    df0 = pd.read_csv(fname, parse_dates=["MONTH"])
    
    # 将特定列转换为 float64 类型
    vn = ["ACTUAL", "PREDICT", "QUARTER", "YEAR"]
    df0[vn] = df0[vn].astype(np.float64)
    
    # 将 MONTH 列转换为秒级精度的日期时间类型
    df0["MONTH"] = df0["MONTH"].astype("M8[s]")
    
    # 断言两个 DataFrame df 和 df0 相等
    tm.assert_frame_equal(df, df0)


# 定义函数以测试特定数据集的读取
def test_12659(datapath):
    # 拼接数据路径和文件名，获取特定数据集的 SAS 数据文件的完整路径
    fname = datapath("io", "sas", "data", "test_12659.sas7bdat")
    
    # 使用默认设置读取 SAS 数据文件，生成 DataFrame df
    df = pd.read_sas(fname)
    
    # 拼接数据路径和文件名，获取特定数据集的 CSV 文件的完整路径
    fname = datapath("io", "sas", "data", "test_12659.csv")
    
    # 使用默认设置读取 CSV 文件，生成 DataFrame df0
    df0 = pd.read_csv(fname)
    
    # 将 df0 中的所有列转换为 float64 类型
    df0 = df0.astype(np.float64)
    
    # 断言两个 DataFrame df 和 df0 相等
    tm.assert_frame_equal(df, df0)


# 定义函数以测试航空数据集的读取
def test_airline(datapath):
    # 拼接数据路径和文件名，获取航空数据集的 SAS 数据文件的完整路径
    fname = datapath("io", "sas", "data", "airline.sas7bdat")
    
    # 使用默认设置读取 SAS 数据文件，生成 DataFrame df
    df = pd.read_sas(fname)
    
    # 拼接数据路径和文件名，获取航空数据集的 CSV 文件的完整路径
    fname = datapath("io", "sas", "data", "airline.csv")
    
    # 使用默认设置读取 CSV 文件，生成 DataFrame df0
    df0 = pd.read_csv(fname)
    
    # 将 df0 中的所有列转换为 float64 类型
    df0 = df0.astype(np.float64)
    
    # 断言两个 DataFrame df 和 df0 相等
    tm.assert_frame_equal(df, df0)


# 使用 pytest 标记，如果在 WASM 环境下则跳过测试日期时间处理
@pytest.mark.skipif(WASM, reason="Pyodide/WASM has 32-bitness")
def test_date_time(datapath):
    # 拼接数据路径和文件名，获取日期时间数据的 SAS 数据文件的完整路径
    fname = datapath("io", "sas", "data", "datetime.sas7bdat")
    
    # 使用默认设置读取 SAS 数据文件，生成 DataFrame df
    df = pd.read_sas(fname)
    
    # 拼接数据路径和文件名，获取日期时间数据的 CSV 文件的完整路径
    fname = datapath("io", "sas", "data", "datetime.csv")
    
    # 使用默认设置读取 CSV 文件，指定多列为日期时间类型，生成 DataFrame df0
    df0 = pd.read_csv(
        fname, parse_dates=["Date1", "Date2", "DateTime", "DateTimeHi", "Taiw"]
    )
    
    # 将 df 中第四列的时间戳四舍五入到微秒级
    df[df.columns[3]] = df.iloc[:, 3].dt.round("us")
    
    # 将 df0 中的多列转换为秒级或毫秒级日期时间类型
    df0["Date1"] = df0["Date1"].astype("M8[s]")
    df0["Date2"] = df0["Date2"].astype("M8[s]")
    df0["DateTime"] = df0["DateTime"].astype("M8[ms]")
    df0["Taiw"] = df0["Taiw"].astype("M8[s]")
    # 将 DataFrame 列 "DateTimeHi" 转换为纳秒精度的日期时间类型，并四舍五入到毫秒
    res = df0["DateTimeHi"].astype("M8[us]").dt.round("ms")
    # 将四舍五入后的日期时间类型再转换回毫秒精度
    df0["DateTimeHi"] = res.astype("M8[ms]")

    # 如果不是64位系统，则进行以下操作
    if not IS64:
        # 在第一行的 "DateTimeHi" 列上增加1毫秒
        df0.loc[0, "DateTimeHi"] += np.timedelta64(1, "ms")
        # 在第二行和第三行的 "DateTimeHi" 列上减少1毫秒
        df0.loc[[2, 3], "DateTimeHi"] -= np.timedelta64(1, "ms")
    
    # 使用 pandas.testing 模块比较两个 DataFrame df 和 df0 是否相等
    tm.assert_frame_equal(df, df0)
@pytest.mark.parametrize("column", ["WGT", "CYL"])
# 使用 pytest 的参数化功能，依次测试 "WGT" 和 "CYL" 两列数据
def test_compact_numerical_values(datapath, column):
    # 回归测试 #21616
    fname = datapath("io", "sas", "data", "cars.sas7bdat")
    # 使用 pandas 读取 SAS 数据文件，指定编码为 "latin-1"
    df = pd.read_sas(fname, encoding="latin-1")
    # cars.sas7bdat 中的 CYL 和 WGT 列宽度小于 8，并且只包含整数值
    # 测试 pandas 读取时不会通过添加小数位来损坏数字
    result = df[column]
    expected = df[column].round()
    # 使用 pytest 的测试工具，确保 result 和 expected Series 相等，精确检查
    tm.assert_series_equal(result, expected, check_exact=True)


def test_many_columns(datapath):
    # PR #22628：测试在更多地方查找列信息
    fname = datapath("io", "sas", "data", "many_columns.sas7bdat")

    # 使用 pandas 读取 SAS 数据文件，指定编码为 "latin-1"
    df = pd.read_sas(fname, encoding="latin-1")

    fname = datapath("io", "sas", "data", "many_columns.csv")
    # 使用 pandas 读取 CSV 文件，指定编码为 "latin-1"
    df0 = pd.read_csv(fname, encoding="latin-1")
    # 使用 pytest 的测试工具，确保 df 和 df0 DataFrame 相等
    tm.assert_frame_equal(df, df0)


def test_inconsistent_number_of_rows(datapath):
    # PR #22628：回归测试 #16615
    fname = datapath("io", "sas", "data", "load_log.sas7bdat")
    # 使用 pandas 读取 SAS 数据文件，指定编码为 "latin-1"
    df = pd.read_sas(fname, encoding="latin-1")
    # 断言 DataFrame df 的长度为 2097
    assert len(df) == 2097


def test_zero_variables(datapath):
    # PR #18184：检查 SAS 文件是否有零个变量
    fname = datapath("io", "sas", "data", "zero_variables.sas7bdat")
    # 使用 pytest 来确保读取 SAS 文件时会引发 EmptyDataError 异常，且异常信息匹配指定字符串
    with pytest.raises(EmptyDataError, match="No columns to parse from file"):
        pd.read_sas(fname)


def test_zero_rows(datapath):
    # GH 18198：检查处理零行数据的情况
    fname = datapath("io", "sas", "data", "zero_rows.sas7bdat")
    # 使用 pandas 读取 SAS 数据文件，无异常
    result = pd.read_sas(fname)
    # 创建预期 DataFrame，然后确保 result 和 expected DataFrame 相等
    expected = pd.DataFrame([{"char_field": "a", "num_field": 1.0}]).iloc[:0]
    tm.assert_frame_equal(result, expected)


def test_corrupt_read(datapath):
    # BUG #35566：不关心具体的读取失败细节，重要的是读取资源后应进行清理
    fname = datapath("io", "sas", "data", "corrupt.sas7bdat")
    msg = "'SAS7BDATReader' object has no attribute 'row_count'"
    # 使用 pytest 来确保读取 SAS 文件时会引发 AttributeError 异常，且异常信息匹配指定字符串
    with pytest.raises(AttributeError, match=msg):
        pd.read_sas(fname)


@pytest.mark.xfail(WASM, reason="failing with currently set tolerances on WASM")
# 使用 pytest 的 xfail 标记，对于 WASM 平台当前设置的容差值，该测试预期会失败
def test_max_sas_date(datapath):
    # GH 20927：最大的 SAS 日期时间在数据集中是 31DEC9999:23:59:59.999
    # 但是由于 buggy sas7bdat 模块，实际读取可能是 29DEC9999:23:59:59.998993
    # 查看 GH#56014 讨论关于正确的 "expected" 结果
    fname = datapath("io", "sas", "data", "max_sas_date.sas7bdat")
    # 使用 pandas 读取 SAS 数据文件，指定编码为 "iso-8859-1"
    df = pd.read_sas(fname, encoding="iso-8859-1")
    # 创建一个预期的 Pandas DataFrame 对象，包含不同数据类型的列
    expected = pd.DataFrame(
        {
            "text": ["max", "normal"],  # 文本列包含两个字符串
            "dt_as_float": [253717747199.999, 1880323199.999],  # 浮点数表示的日期时间
            "dt_as_dt": np.array(
                [
                    datetime(9999, 12, 29, 23, 59, 59, 999000),  # datetime 对象数组，精确到毫秒
                    datetime(2019, 8, 1, 23, 59, 59, 999000),
                ],
                dtype="M8[ms]",  # 指定 datetime 数组的数据类型为毫秒级别
            ),
            "date_as_float": [2936547.0, 21762.0],  # 浮点数表示的日期
            "date_as_date": np.array(
                [
                    datetime(9999, 12, 29),  # datetime 对象数组，精确到秒
                    datetime(2019, 8, 1),
                ],
                dtype="M8[s]",  # 指定 datetime 数组的数据类型为秒级别
            ),
        },
        columns=["text", "dt_as_float", "dt_as_dt", "date_as_float", "date_as_date"],  # 指定列的顺序
    )

    if not IS64:
        # 如果不是64位系统，执行以下操作，通常在持续集成环境中会出现这种情况
        # 从 "dt_as_dt" 列中减去一毫秒的时间间隔
        expected.loc[:, "dt_as_dt"] -= np.timedelta64(1, "ms")

    # 使用测试工具（test tool）比较 DataFrame df 和预期的 DataFrame expected 是否相等
    tm.assert_frame_equal(df, expected)
@pytest.mark.xfail(WASM, reason="failing with currently set tolerances on WASM")
def test_max_sas_date_iterator(datapath):
    # 标记此测试为预期失败（expected failure）在 WASM 平台上，因为当前容忍度设置不兼容
    # GH 20927: 当作为迭代器调用时，只返回日期大于 pd.Timestamp.max 的那些块作为 datetime.datetime
    # 如果发生这种情况，整个块将作为 datetime.datetime 返回
    col_order = ["text", "dt_as_float", "dt_as_dt", "date_as_float", "date_as_date"]
    fname = datapath("io", "sas", "data", "max_sas_date.sas7bdat")
    results = []
    for df in pd.read_sas(fname, encoding="iso-8859-1", chunksize=1):
        # GH 19732: 从 SAS 导入的时间戳将导致浮点错误
        df.reset_index(inplace=True, drop=True)
        results.append(df)
    expected = [
        pd.DataFrame(
            {
                "text": ["max"],
                "dt_as_float": [253717747199.999],
                "dt_as_dt": np.array(
                    [datetime(9999, 12, 29, 23, 59, 59, 999000)], dtype="M8[ms]"
                ),
                "date_as_float": [2936547.0],
                "date_as_date": np.array([datetime(9999, 12, 29)], dtype="M8[s]"),
            },
            columns=col_order,
        ),
        pd.DataFrame(
            {
                "text": ["normal"],
                "dt_as_float": [1880323199.999],
                "dt_as_dt": np.array(["2019-08-01 23:59:59.999"], dtype="M8[ms]"),
                "date_as_float": [21762.0],
                "date_as_date": np.array(["2019-08-01"], dtype="M8[s]"),
            },
            columns=col_order,
        ),
    ]
    if not IS64:
        # 在 CI 中得到的结果没有明显的原因
        expected[0].loc[0, "dt_as_dt"] -= np.timedelta64(1, "ms")
        expected[1].loc[0, "dt_as_dt"] -= np.timedelta64(1, "ms")

    tm.assert_frame_equal(results[0], expected[0])
    tm.assert_frame_equal(results[1], expected[1])


@pytest.mark.skipif(WASM, reason="Pyodide/WASM has 32-bitness")
def test_null_date(datapath):
    # 测试跳过，如果在 WASM 平台上，因为 Pyodide/WASM 是 32 位系统
    fname = datapath("io", "sas", "data", "dates_null.sas7bdat")
    df = pd.read_sas(fname, encoding="utf-8")

    expected = pd.DataFrame(
        {
            "datecol": np.array(
                [
                    datetime(9999, 12, 29),
                    np.datetime64("NaT"),
                ],
                dtype="M8[s]",
            ),
            "datetimecol": np.array(
                [
                    datetime(9999, 12, 29, 23, 59, 59, 999000),
                    np.datetime64("NaT"),
                ],
                dtype="M8[ms]",
            ),
        },
    )
    if not IS64:
        # 在 CI 中得到的结果没有明显的原因
        expected.loc[0, "datetimecol"] -= np.timedelta64(1, "ms")
    tm.assert_frame_equal(df, expected)


def test_meta2_page(datapath):
    # GH 35545: 读取测试文件中的数据框
    fname = datapath("io", "sas", "data", "test_meta2_page.sas7bdat")
    df = pd.read_sas(fname)
    assert len(df) == 1000


@pytest.mark.parametrize(
    "test_file, override_offset, override_value, expected_msg",
    [
        # 测试用例1：文件名为 "test2.sas7bdat"，覆盖偏移量为 0x10000 + 55229，覆盖值为 0x80 | 0x0F，期望输出为 "Out of bounds"
        ("test2.sas7bdat", 0x10000 + 55229, 0x80 | 0x0F, "Out of bounds"),
        # 测试用例2：文件名为 "test2.sas7bdat"，覆盖偏移量为 0x10000 + 55229，覆盖值为 0x10，期望输出为 "unknown control byte"
        ("test2.sas7bdat", 0x10000 + 55229, 0x10, "unknown control byte"),
        # 测试用例3：文件名为 "test3.sas7bdat"，覆盖偏移量为 118170，覆盖值为 184，期望输出为 "Out of bounds"
        ("test3.sas7bdat", 118170, 184, "Out of bounds"),
    ],
# 定义一个测试函数，用于测试 RLE/RDC 解压缩中的异常情况
def test_rle_rdc_exceptions(
    datapath, test_file, override_offset, override_value, expected_msg
):
    """Errors in RLE/RDC decompression should propagate."""
    # 使用指定的数据路径和测试文件名打开文件，并以二进制形式读取内容
    with open(datapath("io", "sas", "data", test_file), "rb") as fd:
        # 将读取的内容转换为字节数组
        data = bytearray(fd.read())
    # 在指定偏移处覆盖原数据值为指定的新值
    data[override_offset] = override_value
    # 使用 pytest 断言捕获异常，检查异常消息是否符合预期
    with pytest.raises(Exception, match=expected_msg):
        # 使用 Pandas 的 read_sas 方法读取修改后的数据流，期望抛出异常
        pd.read_sas(io.BytesIO(data), format="sas7bdat")


# 定义一个测试函数，用于测试包含 0x40 控制字节的情况
def test_0x40_control_byte(datapath):
    # GH 31243
    # 构建包含 0x40 控制字节的 SAS 数据文件路径，并使用 Pandas 读取
    fname = datapath("io", "sas", "data", "0x40controlbyte.sas7bdat")
    df = pd.read_sas(fname, encoding="ascii")
    # 构建包含 0x40 控制字节的 CSV 文件路径，并使用 Pandas 读取
    fname = datapath("io", "sas", "data", "0x40controlbyte.csv")
    df0 = pd.read_csv(fname, dtype="object")
    # 使用 Pandas 的 assert_frame_equal 方法比较两个数据框是否相等
    tm.assert_frame_equal(df, df0)


# 定义一个测试函数，用于测试包含 0x00 控制字节的情况
def test_0x00_control_byte(datapath):
    # GH 47099
    # 构建包含 0x00 控制字节的 SAS 压缩文件路径，并使用 Pandas 读取第一个 chunk
    fname = datapath("io", "sas", "data", "0x00controlbyte.sas7bdat.bz2")
    df = next(pd.read_sas(fname, chunksize=11_000))
    # 使用断言检查数据框的形状是否符合预期
    assert df.shape == (11_000, 20)
```