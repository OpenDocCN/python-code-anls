# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_round_trip.py`

```
# 导入必要的模块和库
import datetime  # 导入处理日期和时间的模块
import re  # 导入正则表达式模块

import numpy as np  # 导入数值计算库numpy
import pytest  # 导入单元测试框架pytest

from pandas._libs.tslibs import Timestamp  # 导入时间戳相关库
from pandas.compat import is_platform_windows  # 导入兼容性库

import pandas as pd  # 导入数据分析工具pandas
from pandas import (  # 从pandas导入多个子模块和函数
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    _testing as tm,
    bdate_range,
    date_range,
    read_hdf,
)
from pandas.tests.io.pytables.common import (  # 导入pytables通用测试相关函数
    _maybe_remove,
    ensure_clean_store,
)
from pandas.util import _test_decorators as td  # 导入测试相关的装饰器

pytestmark = pytest.mark.single_cpu  # 设置pytest标记，单CPU执行


def test_conv_read_write():
    with tm.ensure_clean() as path:  # 使用tm.ensure_clean()确保路径干净

        def roundtrip(key, obj, **kwargs):
            obj.to_hdf(path, key=key, **kwargs)  # 将对象写入HDF文件
            return read_hdf(path, key)  # 从HDF文件读取对象

        o = Series(  # 创建Series对象o
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        tm.assert_series_equal(o, roundtrip("series", o))  # 断言写入和读取后的Series对象相等

        o = Series(range(10), dtype="float64", index=[f"i_{i}" for i in range(10)])  # 创建带有字符串索引的Series对象o
        tm.assert_series_equal(o, roundtrip("string_series", o))  # 断言写入和读取后的Series对象相等

        o = DataFrame(  # 创建DataFrame对象o
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        tm.assert_frame_equal(o, roundtrip("frame", o))  # 断言写入和读取后的DataFrame对象相等

        # table
        df = DataFrame({"A": range(5), "B": range(5)})  # 创建DataFrame对象df
        df.to_hdf(path, key="table", append=True)  # 将df对象追加写入HDF文件中的表格
        result = read_hdf(path, "table", where=["index>2"])  # 从HDF文件读取表格，并根据条件筛选数据
        tm.assert_frame_equal(df[df.index > 2], result)  # 断言读取结果与条件筛选后的df相等


def test_long_strings(setup_path):
    # GH6166
    data = ["a" * 50] * 10  # 创建包含长字符串的数据列表
    df = DataFrame({"a": data}, index=data)  # 创建包含长字符串的DataFrame对象df

    with ensure_clean_store(setup_path) as store:  # 使用ensure_clean_store确保存储区干净
        store.append("df", df, data_columns=["a"])  # 向存储区追加df对象，并指定数据列为"a"

        result = store.select("df")  # 从存储区中选择名称为"df"的数据
        tm.assert_frame_equal(df, result)  # 断言读取结果与df相等


def test_api(tmp_path, setup_path):
    # GH4584
    # 当to_hdf不接受同时传入append和format参数时的API问题
    path = tmp_path / setup_path  # 设置临时路径和安装路径

    df = DataFrame(range(20))  # 创建包含20个整数的DataFrame对象df
    df.iloc[:10].to_hdf(path, key="df", append=True, format="table")  # 将df的前10行写入HDF文件，格式为表格，追加写入
    df.iloc[10:].to_hdf(path, key="df", append=True, format="table")  # 将df的后10行写入HDF文件，格式为表格，追加写入
    tm.assert_frame_equal(read_hdf(path, "df"), df)  # 断言读取结果与df相等

    # append设置为False
    df.iloc[:10].to_hdf(path, key="df", append=False, format="table")  # 将df的前10行写入HDF文件，格式为表格，覆盖写入
    df.iloc[10:].to_hdf(path, key="df", append=True, format="table")  # 将df的后10行追加写入HDF文件，格式为表格
    tm.assert_frame_equal(read_hdf(path, "df"), df)  # 断言读取结果与df相等


def test_api_append(tmp_path, setup_path):
    path = tmp_path / setup_path  # 设置临时路径和安装路径

    df = DataFrame(range(20))  # 创建包含20个整数的DataFrame对象df
    df.iloc[:10].to_hdf(path, key="df", append=True)  # 将df的前10行追加写入HDF文件
    df.iloc[10:].to_hdf(path, key="df", append=True, format="table")  # 将df的后10行追加写入HDF文件，格式为表格
    tm.assert_frame_equal(read_hdf(path, "df"), df)  # 断言读取结果与df相等

    # append设置为False，格式为表格
    df.iloc[:10].to_hdf(path, key="df", append=False, format="table")  # 将df的前10行覆盖写入HDF文件，格式为表格
    df.iloc[10:].to_hdf(path, key="df", append=True)  # 将df的后10行追加写入HDF文件
    tm.assert_frame_equal(read_hdf(path, "df"), df)  # 断言读取结果与df相等


def test_api_2(tmp_path, setup_path):
    path = tmp_path / setup_path  # 设置临时路径和安装路径

    df = DataFrame(range(20))  # 创建包含20个整数的DataFrame对象df
    # 使用 pandas 将 DataFrame 写入 HDF5 文件，指定键为 "df"，不追加数据，使用固定格式存储
    df.to_hdf(path, key="df", append=False, format="fixed")
    # 读取指定路径下键为 "df" 的 HDF5 文件内容，与原 DataFrame 进行比较验证是否相等
    tm.assert_frame_equal(read_hdf(path, "df"), df)
    
    # 使用 pandas 将 DataFrame 写入 HDF5 文件，指定键为 "df"，不追加数据，使用浮点格式存储
    df.to_hdf(path, key="df", append=False, format="f")
    # 读取指定路径下键为 "df" 的 HDF5 文件内容，与原 DataFrame 进行比较验证是否相等
    tm.assert_frame_equal(read_hdf(path, "df"), df)
    
    # 使用 pandas 将 DataFrame 写入 HDF5 文件，指定键为 "df"，不追加数据，自动选择格式存储
    df.to_hdf(path, key="df", append=False)
    # 读取指定路径下键为 "df" 的 HDF5 文件内容，与原 DataFrame 进行比较验证是否相等
    tm.assert_frame_equal(read_hdf(path, "df"), df)
    
    # 使用 pandas 将 DataFrame 写入 HDF5 文件，指定键为 "df"，追加数据到已有文件，自动选择格式存储
    df.to_hdf(path, key="df")
    # 读取指定路径下键为 "df" 的 HDF5 文件内容，与原 DataFrame 进行比较验证是否相等
    tm.assert_frame_equal(read_hdf(path, "df"), df)
    
    # 使用 ensure_clean_store 上下文管理器创建一个清理过的存储环境，并指定存储路径 setup_path
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含 0 到 19 的 DataFrame
        df = DataFrame(range(20))
        
        # 清除存储中的 "df" 数据
        _maybe_remove(store, "df")
        # 将 DataFrame 的前 10 行追加到存储中的 "df"，使用表格格式存储
        store.append("df", df.iloc[:10], append=True, format="table")
        # 将 DataFrame 的后 10 行追加到存储中的 "df"，使用表格格式存储
        store.append("df", df.iloc[10:], append=True, format="table")
        # 从存储中选择键为 "df" 的内容，与原 DataFrame 进行比较验证是否相等
        tm.assert_frame_equal(store.select("df"), df)
    
        # append 参数设置为 False
        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=False, format="table")
        store.append("df", df.iloc[10:], append=True, format="table")
        tm.assert_frame_equal(store.select("df"), df)
    
        # 测试不同的存储格式
        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=False, format="table")
        store.append("df", df.iloc[10:], append=True, format="table")
        tm.assert_frame_equal(store.select("df"), df)
    
        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=False, format="table")
        store.append("df", df.iloc[10:], append=True, format=None)
        tm.assert_frame_equal(store.select("df"), df)
# 测试处理 API 不合法情况的函数
def test_api_invalid(tmp_path, setup_path):
    # 创建一个 DataFrame 对象，包含从 0 到 119 的数字，以及相关设置
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),  # 使用 numpy 生成一组数据，reshape 成 30 行 4 列的形状
        columns=Index(list("ABCD"), dtype=object),  # 指定列索引为 A、B、C、D，并指定数据类型为对象
        index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 指定行索引为类似 i-0、i-1 的格式，并指定数据类型为对象
    )

    msg = "Can only append to Tables"  # 设置错误消息字符串

    # 使用 pytest 断言捕获 ValueError 异常，并验证是否匹配特定消息
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key="df", append=True, format="f")

    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key="df", append=True, format="fixed")

    msg = r"invalid HDFStore format specified \[foo\]"  # 设置另一个错误消息字符串

    # 使用 pytest 断言捕获 TypeError 异常，并验证是否匹配特定消息
    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key="df", append=True, format="foo")

    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key="df", append=False, format="foo")

    # 文件路径为空字符串时，设置文件不存在的错误消息
    path = ""
    msg = f"File {path} does not exist"

    # 使用 pytest 断言捕获 FileNotFoundError 异常，并验证是否匹配特定消息
    with pytest.raises(FileNotFoundError, match=msg):
        read_hdf(path, "df")


# 测试获取函数的功能
def test_get(setup_path):
    # 在确保存储区域干净的前提下，使用 store 对象进行测试
    with ensure_clean_store(setup_path) as store:
        # 向存储中添加一个名为 "a" 的 Series 对象，数据为浮点数序列，索引为日期范围
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        left = store.get("a")  # 使用 get 方法获取键为 "a" 的对象
        right = store["a"]  # 直接访问键为 "a" 的对象
        tm.assert_series_equal(left, right)  # 使用 pandas 测试模块验证两个 Series 对象是否相等

        left = store.get("/a")  # 使用 get 方法获取键为 "/a" 的对象
        right = store["/a"]  # 直接访问键为 "/a" 的对象
        tm.assert_series_equal(left, right)  # 使用 pandas 测试模块验证两个 Series 对象是否相等

        # 使用 pytest 断言捕获 KeyError 异常，并验证是否匹配特定消息
        with pytest.raises(KeyError, match="'No object named b in the file'"):
            store.get("b")


# 测试处理整数索引的函数
def test_put_integer(setup_path):
    # 创建一个具有随机数据的 DataFrame 对象，使用 _check_roundtrip 函数验证存取过程
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 100)))
    _check_roundtrip(df, tm.assert_frame_equal, setup_path)


# 测试表值及数据类型的往返存取函数
def test_table_values_dtypes_roundtrip(setup_path):
    # 使用 ensure_clean_store 函数确保 store 在测试前是干净的状态
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含一列浮点数的 DataFrame 对象 df1
        df1 = DataFrame({"a": [1, 2, 3]}, dtype="f8")
        # 将 df1 添加到 store 中，键名为 "df_f8"
        store.append("df_f8", df1)
        # 断言 df1 的数据类型与 store 中 "df_f8" 的数据类型相等
        tm.assert_series_equal(df1.dtypes, store["df_f8"].dtypes)

        # 创建一个包含一列整数的 DataFrame 对象 df2
        df2 = DataFrame({"a": [1, 2, 3]}, dtype="i8")
        # 将 df2 添加到 store 中，键名为 "df_i8"
        store.append("df_i8", df2)
        # 断言 df2 的数据类型与 store 中 "df_i8" 的数据类型相等
        tm.assert_series_equal(df2.dtypes, store["df_i8"].dtypes)

        # 当数据类型不兼容时引发 ValueError 异常
        msg = re.escape(
            "invalid combination of [values_axes] on appending data "
            "[name->values_block_0,cname->values_block_0,"
            "dtype->float64,kind->float,shape->(1, 3)] vs "
            "current table [name->values_block_0,"
            "cname->values_block_0,dtype->int64,kind->integer,"
            "shape->None]"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df_i8", df1)

        # 检查创建、存储和检索 float32 类型数据（实际上有些繁琐）
        df1 = DataFrame(np.array([[1], [2], [3]], dtype="f4"), columns=["A"])
        # 将 df1 添加到 store 中，键名为 "df_f4"
        store.append("df_f4", df1)
        # 断言 df1 的数据类型与 store 中 "df_f4" 的数据类型相等
        tm.assert_series_equal(df1.dtypes, store["df_f4"].dtypes)
        # 断言 df1 的第一列数据类型为 "float32"
        assert df1.dtypes.iloc[0] == "float32"

        # 检查混合数据类型的情况
        df1 = DataFrame(
            {
                c: Series(np.random.default_rng(2).integers(5), dtype=c)
                for c in ["float32", "float64", "int32", "int64", "int16", "int8"]
            }
        )
        df1["string"] = "foo"
        df1["float322"] = 1.0
        df1["float322"] = df1["float322"].astype("float32")
        df1["bool"] = df1["float32"] > 0
        df1["time_s_1"] = Timestamp("20130101")
        df1["time_s_2"] = Timestamp("20130101 00:00:00")
        df1["time_ms"] = Timestamp("20130101 00:00:00.000")
        df1["time_ns"] = Timestamp("20130102 00:00:00.000000000")

        # 将 df1 添加到 store 中，键名为 "df_mixed_dtypes1"
        store.append("df_mixed_dtypes1", df1)
        # 选择 "df_mixed_dtypes1" 并检查其数据类型的计数分布
        result = store.select("df_mixed_dtypes1").dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        # 期望的数据类型计数结果
        expected = Series(
            {
                "float32": 2,
                "float64": 1,
                "int32": 1,
                "bool": 1,
                "int16": 1,
                "int8": 1,
                "int64": 1,
                "object": 1,
                "datetime64[s]": 2,
                "datetime64[ms]": 1,
                "datetime64[ns]": 1,
            },
            name="count",
        )
        # 对结果和期望进行排序
        result = result.sort_index()
        expected = expected.sort_index()
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)
@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
# 使用 pytest 的装饰器标记，忽略特定警告类别的警告信息
def test_series(setup_path):
    # 创建一个 Series 对象，包含整数范围为 [0, 9]，数据类型为 float64，索引为以 "i_" 开头的字符串列表
    s = Series(range(10), dtype="float64", index=[f"i_{i}" for i in range(10)])
    # 调用 _check_roundtrip 函数，验证序列相等性，使用 setup_path 作为路径参数
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)

    # 创建另一个 Series 对象，包含从 0 到 9 的浮点数，索引为从 "2020-01-01" 开始的 10 个日期
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    # 再次调用 _check_roundtrip 函数，验证序列相等性，使用 setup_path 作为路径参数
    _check_roundtrip(ts, tm.assert_series_equal, path=setup_path)

    # 创建一个新的 Series 对象，其索引为 ts.index 的对象
    ts2 = Series(ts.index, Index(ts.index, dtype=object))
    # 调用 _check_roundtrip 函数，验证序列相等性，使用 setup_path 作为路径参数
    _check_roundtrip(ts2, tm.assert_series_equal, path=setup_path)

    # 创建另一个新的 Series 对象，其值为 ts.values，索引为 ts.index 转换为对象数组
    ts3 = Series(ts.values, Index(np.asarray(ts.index, dtype=object), dtype=object))
    # 再次调用 _check_roundtrip 函数，验证序列相等性，使用 setup_path 作为路径参数，不检查索引类型
    _check_roundtrip(
        ts3, tm.assert_series_equal, path=setup_path, check_index_type=False
    )


def test_float_index(setup_path):
    # GH #454
    # 创建一个 Series 对象，包含使用标准正态分布生成的随机浮点数，索引为使用标准正态分布生成的随机浮点数
    index = np.random.default_rng(2).standard_normal(10)
    s = Series(np.random.default_rng(2).standard_normal(10), index=index)
    # 调用 _check_roundtrip 函数，验证序列相等性，使用 setup_path 作为路径参数
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)


def test_tuple_index(setup_path, performance_warning):
    # GH #492
    # 创建一个 DataFrame 对象，包含使用标准正态分布生成的随机数据，行索引为 idx，列索引为 col
    col = np.arange(10)
    idx = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    data = np.random.default_rng(2).standard_normal(30).reshape((3, 10))
    DF = DataFrame(data, index=idx, columns=col)

    # 使用 pytest 的上下文管理，检查性能警告
    with tm.assert_produces_warning(performance_warning):
        # 调用 _check_roundtrip 函数，验证 DataFrame 相等性，使用 setup_path 作为路径参数
        _check_roundtrip(DF, tm.assert_frame_equal, path=setup_path)


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
# 使用 pytest 的装饰器标记，忽略特定警告类别的警告信息
def test_index_types(setup_path):
    values = np.random.default_rng(2).standard_normal(2)

    # 定义一个 lambda 函数，用于比较两个 Series 对象的相等性，检查索引类型
    func = lambda lhs, rhs: tm.assert_series_equal(lhs, rhs, check_index_type=True)

    # 创建多个 Series 对象，每个对象包含随机值和不同类型的索引
    ser = Series(values, [0, "y"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.datetime.today(), 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, ["y", 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.date.today(), "a"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [0, "y"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.datetime.today(), 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, ["y", 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.date.today(), "a"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [1.23, "b"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [1, 1.53])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [1, 5])
    _check_roundtrip(ser, func, path=setup_path)

    # 创建一个 Series 对象，包含使用 DatetimeIndex 的随机日期索引和随机值
    dti = DatetimeIndex(["2012-01-01", "2012-01-02"], dtype="M8[ns]")
    ser = Series(values, index=dti)
    _check_roundtrip(ser, func, path=setup_path)

    # 将 Series 对象的索引转换为秒级别的单位
    ser.index = ser.index.as_unit("s")
    _check_roundtrip(ser, func, path=setup_path)


def test_timeseries_preepoch(setup_path, request):
    # 创建一个工作日日期范围对象，从 "1/1/1940" 到 "1/1/1960"
    dr = bdate_range("1/1/1940", "1/1/1960")
    # 使用 NumPy 生成指定长度的随机标准正态分布数据，并将其封装成 Pandas 的 Series 对象
    ts = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
    try:
        # 调用 _check_roundtrip 函数，验证 Series 对象 ts 能否与另一个 Series 对象通过指定的方法相等
        _check_roundtrip(ts, tm.assert_series_equal, path=setup_path)
    except OverflowError:
        # 如果发生溢出错误，则检查操作系统是否为 Windows
        if is_platform_windows():
            # 在 Windows 平台上应用一个标记，表明这个测试在某些 Windows 平台上已知会失败
            request.applymarker(
                pytest.mark.xfail("known failure on some windows platforms")
            )
        # 将溢出错误重新引发，以便上层处理
        raise
@pytest.mark.parametrize(
    # 参数化测试，compression 可以是 False 或者 True，根据操作系统选择性跳过压缩测试
    "compression", [False, pytest.param(True, marks=td.skip_if_windows)]
)
def test_frame(compression, setup_path):
    # 创建一个 DataFrame，包含 30 行 4 列的数据，列标签为 'A', 'B', 'C', 'D'，行标签为 'i-0' 到 'i-29'
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    # 在 DataFrame 中随机设置一些缺失值
    df.iloc[0, 0] = np.nan
    df.iloc[5, 3] = np.nan

    # 调用 _check_roundtrip_table 函数，验证数据框的序列化和反序列化是否正常，压缩方式由 compression 参数指定
    _check_roundtrip_table(
        df, tm.assert_frame_equal, path=setup_path, compression=compression
    )

    # 调用 _check_roundtrip 函数，同样验证数据框的序列化和反序列化是否正常，压缩方式由 compression 参数指定
    _check_roundtrip(
        df, tm.assert_frame_equal, path=setup_path, compression=compression
    )

    # 创建一个新的 DataFrame，包含 10 行 4 列的随机数据，列标签为 'A', 'B', 'C', 'D'，行标签为 '2000-01-01' 至 '2000-01-14'
    tdf = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    # 调用 _check_roundtrip 函数，验证新数据框的序列化和反序列化是否正常，压缩方式由 compression 参数指定
    _check_roundtrip(
        tdf, tm.assert_frame_equal, path=setup_path, compression=compression
    )

    # 使用 ensure_clean_store 函数确保存储路径 setup_path 下的数据存储环境干净
    with ensure_clean_store(setup_path) as store:
        # 在数据存储中添加一个名为 "df" 的 DataFrame，并且在存储中设置 "df" 的引用
        df["foo"] = np.random.default_rng(2).standard_normal(len(df))
        store["df"] = df
        recons = store["df"]
        # 断言重新构建的 DataFrame recons 的数据管理器是否已经被整合
        assert recons._mgr.is_consolidated()

    # 对空的 DataFrame 切片进行序列化和反序列化测试，验证是否正常处理空数据
    _check_roundtrip(df[:0], tm.assert_frame_equal, path=setup_path)


def test_empty_series_frame(setup_path):
    # 创建一个空的 Series s0
    s0 = Series(dtype=object)
    # 创建一个命名为 "myseries" 的空 Series s1
    s1 = Series(name="myseries", dtype=object)
    # 创建一个空的 DataFrame df0
    df0 = DataFrame()
    # 创建一个指定索引为 ['a', 'b', 'c'] 的空 DataFrame df1
    df1 = DataFrame(index=["a", "b", "c"])
    # 创建一个指定列标签为 ['d', 'e', 'f'] 的空 DataFrame df2
    df2 = DataFrame(columns=["d", "e", "f"])

    # 对各个空数据结构进行序列化和反序列化测试，验证是否正常处理空数据
    _check_roundtrip(s0, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(s1, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(df0, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)


@pytest.mark.parametrize(
    # 参数化测试，dtype 可以是 np.int64, np.float64, object, 'm8[ns]', 'M8[ns]' 中的一种
    "dtype", [np.int64, np.float64, object, "m8[ns]", "M8[ns]"]
)
def test_empty_series(dtype, setup_path):
    # 根据指定的 dtype 创建一个空 Series s
    s = Series(dtype=dtype)
    # 对空 Series 进行序列化和反序列化测试，验证是否正常处理空数据
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)


def test_can_serialize_dates(setup_path):
    # 创建一个 DataFrame，包含随机数值，索引为日期范围内的工作日
    rng = [x.date() for x in bdate_range("1/1/2000", "1/30/2000")]
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )

    # 对日期型数据进行序列化和反序列化测试，验证是否正常处理日期类型数据
    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)


def test_store_hierarchical(setup_path, multiindex_dataframe_random_data):
    # 使用多级索引的随机数据创建一个 DataFrame frame
    frame = multiindex_dataframe_random_data

    # 对多级索引 DataFrame 进行序列化和反序列化测试，验证是否正常处理层次化数据
    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame.T, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame["A"], tm.assert_series_equal, path=setup_path)

    # 确认存储中正确保存了数据框的名称
    with ensure_clean_store(setup_path) as store:
        store["frame"] = frame
        recons = store["frame"]
        tm.assert_frame_equal(recons, frame)
# 定义测试函数，用于测试混合数据的存储
def test_store_mixed(compression, setup_path):
    # 定义内部函数 _make_one，用于创建一个DataFrame对象
    def _make_one():
        # 创建一个包含特定数据的DataFrame对象
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),  # 创建一个30行4列的DataFrame，数据为1.1倍递增数组
            columns=Index(list("ABCD"), dtype=object),  # 列标签为 ['A', 'B', 'C', 'D']，数据类型为对象类型
            index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 行标签为 ['i-0', 'i-1', ..., 'i-29']，数据类型为对象类型
        )
        # 添加额外的列到DataFrame中
        df["obj1"] = "foo"  # 添加名为 'obj1' 的列，数据为字符串 "foo"
        df["obj2"] = "bar"  # 添加名为 'obj2' 的列，数据为字符串 "bar"
        df["bool1"] = df["A"] > 0  # 添加名为 'bool1' 的列，数据为 'A' 列中大于0的布尔值
        df["bool2"] = df["B"] > 0  # 添加名为 'bool2' 的列，数据为 'B' 列中大于0的布尔值
        df["int1"] = 1  # 添加名为 'int1' 的列，数据为整数1
        df["int2"] = 2  # 添加名为 'int2' 的列，数据为整数2
        # 对DataFrame进行整理并返回
        return df._consolidate()

    # 创建两个DataFrame对象
    df1 = _make_one()
    df2 = _make_one()

    # 调用 _check_roundtrip 函数，测试 DataFrame 对象 df1 和 df2 的存储和读取是否相等
    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)

    # 在确保存储路径干净的情况下，使用 ensure_clean_store 上下文管理器
    with ensure_clean_store(setup_path) as store:
        # 存储 df1 到 store 中，并断言存储和读取的数据是否相等
        store["obj"] = df1
        tm.assert_frame_equal(store["obj"], df1)
        # 存储 df2 到 store 中，并断言存储和读取的数据是否相等
        store["obj"] = df2
        tm.assert_frame_equal(store["obj"], df2)

    # 检查是否可以存储 Series 的所有这些类型
    _check_roundtrip(
        df1["obj1"],
        tm.assert_series_equal,
        path=setup_path,
        compression=compression,
    )
    _check_roundtrip(
        df1["bool1"],
        tm.assert_series_equal,
        path=setup_path,
        compression=compression,
    )
    _check_roundtrip(
        df1["int1"],
        tm.assert_series_equal,
        path=setup_path,
        compression=compression,
    )


# 定义内部函数 _check_roundtrip，用于测试对象的存储和读取是否正确
def _check_roundtrip(obj, comparator, path, compression=False, **kwargs):
    options = {}
    if compression:
        options["complib"] = "blosc"  # 如果使用压缩，选择 blosc 压缩算法

    # 在 ensure_clean_store 上下文管理器中，以写入模式打开 store，并使用指定的选项
    with ensure_clean_store(path, "w", **options) as store:
        # 存储 obj 到 store 中
        store["obj"] = obj
        # 从 store 中读取存储的对象
        retrieved = store["obj"]
        # 使用给定的比较器比较读取到的对象和原始对象
        comparator(retrieved, obj, **kwargs)


# 定义内部函数 _check_roundtrip_table，用于测试表格对象的存储和读取是否正确
def _check_roundtrip_table(obj, comparator, path, compression=False):
    options = {}
    if compression:
        options["complib"] = "blosc"  # 如果使用压缩，选择 blosc 压缩算法

    # 在 ensure_clean_store 上下文管理器中，以写入模式打开 store，并使用指定的选项
    with ensure_clean_store(path, "w", **options) as store:
        # 将 obj 以表格格式存储到 store 中
        store.put("obj", obj, format="table")
        # 从 store 中读取存储的对象
        retrieved = store["obj"]
        # 使用给定的比较器比较读取到的对象和原始对象
        comparator(retrieved, obj)


# 定义测试函数，测试Unicode索引的存储和读取是否正确
def test_unicode_index(setup_path):
    unicode_values = ["\u03c3", "\u03c3\u03c3"]  # 定义包含Unicode值的列表

    # 创建一个Series对象，索引为Unicode值列表，数据为随机生成的标准正态分布数据
    s = Series(
        np.random.default_rng(2).standard_normal(len(unicode_values)),
        unicode_values,
    )
    # 调用 _check_roundtrip 函数，测试 Series 对象 s 的存储和读取是否相等
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)


# 定义测试函数，测试长编码的Unicode值的存储和读取是否正确
def test_unicode_longer_encoded(setup_path):
    # GH 11234
    char = "\u0394"  # 定义一个Unicode字符
    # 创建一个包含Unicode列的DataFrame对象
    df = DataFrame({"A": [char]})
    # 在确保存储路径干净的情况下，使用 ensure_clean_store 上下文管理器
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame df 以表格格式存储到 store 中，使用UTF-8编码
        store.put("df", df, format="table", encoding="utf-8")
        # 从 store 中获取存储的对象
        result = store.get("df")
        # 断言从 store 中获取的对象和原始对象 df 是否相等
        tm.assert_frame_equal(result, df)

    # 创建一个包含长编码Unicode列的DataFrame对象
    df = DataFrame({"A": ["a", char], "B": ["b", "b"]})
    # 在确保存储路径干净的情况下，使用 ensure_clean_store 上下文管理器
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame df 以表格格式存储到 store 中，使用UTF-8编码
        store.put("df", df, format="table", encoding="utf-8")
        # 从 store 中获取存储的对象
        result = store.get("df")
        # 断言从 store 中获取的对象和原始对象 df 是否相等
        tm.assert_frame_equal(result, df)


# 定义测试函数，测试包含不同类型数据的DataFrame的存储和读取是否正确
def test_store_datetime_mixed(setup_path):
    # 创建一个包含不同类型数据的DataFrame对象
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})
    # 创建一个 Series 对象 ts，其中包含从 0 到 9 的浮点数，索引为从 "2020-01-01" 开始的连续 10 个日期
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    
    # 将 Series ts 的前三个索引（日期）赋值给 DataFrame df 的新列 "d"
    df["d"] = ts.index[:3]
    
    # 调用 _check_roundtrip 函数，检查 DataFrame df 是否与另一个 DataFrame 通过某种方式相等，
    # 使用 tm.assert_frame_equal 进行比较，同时将路径 setup_path 传递给函数
    _check_roundtrip(df, tm.assert_frame_equal, path=setup_path)
# 测试数据框的写入和读取是否相等
def test_round_trip_equals(tmp_path, setup_path):
    # 创建一个包含两列的数据框
    df = DataFrame({"B": [1, 2], "A": ["x", "y"]})

    # 设置文件路径
    path = tmp_path / setup_path
    # 将数据框写入 HDF 文件
    df.to_hdf(path, key="df", format="table")
    # 从 HDF 文件中读取数据框
    other = read_hdf(path, "df")
    # 检查两个数据框是否相等
    tm.assert_frame_equal(df, other)
    # 断言两个数据框是否相等
    assert df.equals(other)
    assert other.equals(df)


# 测试推断字符串列
def test_infer_string_columns(tmp_path, setup_path):
    # 检查是否存在 pyarrow 模块，如果不存在则跳过测试
    pytest.importorskip("pyarrow")
    # 设置文件路径
    path = tmp_path / setup_path
    # 设置将来推断字符串的选项为 True
    with pd.option_context("future.infer_string", True):
        # 创建一个数据框，包含四列和十行，并设置索引
        df = DataFrame(1, columns=list("ABCD"), index=list(range(10))).set_index(
            ["A", "B"]
        )
        # 复制数据框
        expected = df.copy()
        # 将数据框写入 HDF 文件
        df.to_hdf(path, key="df", format="table")

        # 从 HDF 文件中读取数据框
        result = read_hdf(path, "df")
        # 检查读取的数据框是否与预期相等
        tm.assert_frame_equal(result, expected)
```