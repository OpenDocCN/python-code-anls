# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_astype.py`

```
# 导入pickle模块，用于序列化和反序列化Python对象
import pickle

# 导入NumPy库，用于科学计算，支持多维数组和矩阵运算
import numpy as np

# 导入pytest模块，用于编写和运行测试用例
import pytest

# 从pandas.compat.pyarrow模块中导入pa_version_under12p0，用于检查pyarrow版本是否小于12.0
from pandas.compat.pyarrow import pa_version_under12p0

# 导入pandas.util._test_decorators模块中的td别名，用于测试装饰器
import pandas.util._test_decorators as td

# 从pandas库中导入DataFrame、Series、Timestamp、date_range等类和函数
from pandas import (
    DataFrame,
    Series,
    Timestamp,
    date_range,
)

# 导入pandas._testing模块，用于测试支持函数和类
import pandas._testing as tm

# 从pandas.tests.copy_view.util模块中导入get_array函数
from pandas.tests.copy_view.util import get_array


def test_astype_single_dtype():
    # 创建DataFrame对象df，包含三列：a为整数列表，b为整数列表，c为浮点数1.5
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": 1.5})
    # 复制df，保存原始数据到df_orig
    df_orig = df.copy()
    # 将df的数据类型全部转换为float64，并保存为df2
    df2 = df.astype("float64")

    # 断言df2中的列"c"与df中的列"c"共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 断言df2中的列"a"与df中的列"a"不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改df2中的元素触发"c"列的写时复制
    df2.iloc[0, 2] = 5.5
    # 断言df2与df_orig相等
    assert not np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)

    # 再次修改原始df不影响df2的结果
    df2 = df.astype("float64")
    df.iloc[0, 2] = 5.5
    tm.assert_frame_equal(df2, df_orig.astype("float64"))


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize("new_dtype", ["int64", "Int64", "int64[pyarrow]"])
def test_astype_avoids_copy(dtype, new_dtype):
    # 如果new_dtype为"int64[pyarrow]"，则检查并跳过，确保pyarrow已导入
    if new_dtype == "int64[pyarrow]":
        pytest.importorskip("pyarrow")
    # 创建DataFrame对象df，包含一列"a"，数据类型为dtype指定
    df = DataFrame({"a": [1, 2, 3]}, dtype=dtype)
    # 复制df，保存原始数据到df_orig
    df_orig = df.copy()
    # 将df的数据类型全部转换为new_dtype，并保存为df2
    df2 = df.astype(new_dtype)
    # 断言df2中的列"a"与df中的列"a"共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改df2中的元素触发"a"列的写时复制
    df2.iloc[0, 0] = 10
    # 断言df2与df_orig相等
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)

    # 再次修改原始df不影响df2的结果
    df2 = df.astype(new_dtype)
    df.iloc[0, 0] = 100
    tm.assert_frame_equal(df2, df_orig.astype(new_dtype))


@pytest.mark.parametrize("dtype", ["float64", "int32", "Int32", "int32[pyarrow]"])
def test_astype_different_target_dtype(dtype):
    # 如果dtype为"int32[pyarrow]"，则检查并跳过，确保pyarrow已导入
    if dtype == "int32[pyarrow]":
        pytest.importorskip("pyarrow")
    # 创建DataFrame对象df，包含一列"a"，数据类型为默认的dtype
    df = DataFrame({"a": [1, 2, 3]})
    # 复制df，保存原始数据到df_orig
    df_orig = df.copy()
    # 将df的数据类型全部转换为dtype，并保存为df2
    df2 = df.astype(dtype)

    # 断言df2中的列"a"与df中的列"a"不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言df2是否有参考
    assert df2._mgr._has_no_reference(0)

    # 修改df2中的元素，断言df与df_orig相等
    df2.iloc[0, 0] = 5
    tm.assert_frame_equal(df, df_orig)

    # 再次修改原始df不影响df2的结果
    df2 = df.astype(dtype)
    df.iloc[0, 0] = 100
    tm.assert_frame_equal(df2, df_orig.astype(dtype))


def test_astype_numpy_to_ea():
    # 创建Series对象ser，包含整数列表[1, 2, 3]
    ser = Series([1, 2, 3])
    # 将ser的数据类型转换为"Int64"，保存为result
    result = ser.astype("Int64")
    # 断言ser与result共享内存
    assert np.shares_memory(get_array(ser), get_array(result))


@pytest.mark.parametrize(
    "dtype, new_dtype", [("object", "string"), ("string", "object")]
)
def test_astype_string_and_object(dtype, new_dtype):
    # 创建DataFrame对象df，包含一列"a"，数据类型为dtype指定
    df = DataFrame({"a": ["a", "b", "c"]}, dtype=dtype)
    # 复制df，保存原始数据到df_orig
    df_orig = df.copy()
    # 将df的数据类型全部转换为new_dtype，并保存为df2
    df2 = df.astype(new_dtype)
    # 断言df2中的列"a"与df中的列"a"共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改df2中的元素
    df2.iloc[0, 0] = "x"
    # 断言df与df_orig相等
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "dtype, new_dtype", [("object", "string"), ("string", "object")]


# 定义一个包含元组的列表，每个元组包含两个字符串
"dtype, new_dtype", [("object", "string"), ("string", "object")]


这行代码定义了一个包含两个元组的列表。每个元组包含两个字符串，第一个字符串表示当前数据类型（dtype），第二个字符串表示转换后的新数据类型（new_dtype）。
# 定义一个函数，用于测试将 DataFrame 的某列转换为指定类型后是否会更新原始 DataFrame
def test_astype_string_and_object_update_original(dtype, new_dtype):
    # 创建一个 DataFrame，包含一列字符串类型的数据
    df = DataFrame({"a": ["a", "b", "c"]}, dtype=dtype)
    # 使用 astype 方法将 DataFrame 的数据类型转换为新的数据类型
    df2 = df.astype(new_dtype)
    # 复制 df2，以备后续比较
    df_orig = df2.copy()
    # 断言转换后的 DataFrame 的某列与原始 DataFrame 的某列共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改原始 DataFrame 的某个元素
    df.iloc[0, 0] = "x"
    # 断言修改原始 DataFrame 后，与复制的 df2 仍相等
    tm.assert_frame_equal(df2, df_orig)


# 定义一个函数，用于测试在使用 pickle 进行序列化和反序列化过程中，转换为字符串类型是否会影响原始数据
def test_astype_string_copy_on_pickle_roundrip():
    # 创建一个 Series，包含对象类型的数组
    base = Series(np.array([(1, 2), None, 1], dtype="object"))
    # 使用 pickle 进行深拷贝
    base_copy = pickle.loads(pickle.dumps(base))
    # 将拷贝后的 Series 转换为字符串类型
    base_copy.astype(str)
    # 断言转换后的 Series 与原始 Series 相等
    tm.assert_series_equal(base, base_copy)


# 装饰器函数，用于测试在使用 pyarrow 时，转换为字符串类型是否会影响只读数组
@td.skip_if_no("pyarrow")
def test_astype_string_read_only_on_pickle_roundrip():
    # 创建一个 Series，包含对象类型的数组
    base = Series(np.array([(1, 2), None, 1], dtype="object"))
    # 使用 pickle 进行深拷贝
    base_copy = pickle.loads(pickle.dumps(base))
    # 将拷贝后的 Series 设置为只读
    base_copy._values.flags.writeable = False
    # 将只读的 Series 转换为 pyarrow 的字符串类型
    base_copy.astype("string[pyarrow]")
    # 断言转换后的 Series 与原始 Series 相等
    tm.assert_series_equal(base, base_copy)


# 定义一个函数，用于测试将 DataFrame 的多列转换为指定类型后的内存共享情况
def test_astype_dict_dtypes():
    # 创建一个 DataFrame，包含多列数据
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": Series([1.5, 1.5, 1.5], dtype="float64")}
    )
    # 复制原始 DataFrame
    df_orig = df.copy()
    # 使用字典指定将某些列转换为特定数据类型
    df2 = df.astype({"a": "float64", "c": "float64"})

    # 断言转换后的 DataFrame 的某列与原始 DataFrame 的某列共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # 断言转换后的 DataFrame 的某列与原始 DataFrame 的某列不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改转换后的 DataFrame 的某个元素
    df2.iloc[0, 2] = 5.5
    # 断言修改转换后的 DataFrame 后，与原始 DataFrame 的某列不再共享内存
    assert not np.shares_memory(get_array(df2, "c"), get_array(df, "c"))

    df2.iloc[0, 1] = 10
    # 断言修改转换后的 DataFrame 后，与原始 DataFrame 的某列不再共享内存
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # 断言转换后的 DataFrame 与原始 DataFrame 相等
    tm.assert_frame_equal(df, df_orig)


# 定义一个函数，用于测试将 DataFrame 的日期时间列转换为不同分辨率的影响
def test_astype_different_datetime_resos():
    # 创建一个 DataFrame，包含日期时间列
    df = DataFrame({"a": date_range("2019-12-31", periods=2, freq="D")})
    # 将日期时间列转换为毫秒级的日期时间
    result = df.astype("datetime64[ms]")

    # 断言转换后的 DataFrame 的日期时间列与原始 DataFrame 的日期时间列不共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    # 断言结果 DataFrame 中的数据块没有引用
    assert result._mgr._has_no_reference(0)


# 定义一个函数，用于测试将 DataFrame 的时区日期时间列转换为不同时区的影响
def test_astype_different_timezones():
    # 创建一个 DataFrame，包含时区日期时间列
    df = DataFrame(
        {"a": date_range("2019-12-31", periods=5, freq="D", tz="US/Pacific")}
    )
    # 将时区日期时间列转换为欧洲柏林时区的日期时间
    result = df.astype("datetime64[ns, Europe/Berlin]")
    # 断言结果 DataFrame 中的数据块有引用
    assert not result._mgr._has_no_reference(0)
    # 断言转换后的 DataFrame 的日期时间列与原始 DataFrame 的日期时间列共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))


# 定义一个函数，用于测试将 DataFrame 的时区日期时间列转换为不同时区和不同分辨率的影响
def test_astype_different_timezones_different_reso():
    # 创建一个 DataFrame，包含时区日期时间列
    df = DataFrame(
        {"a": date_range("2019-12-31", periods=5, freq="D", tz="US/Pacific")}
    )
    # 将时区日期时间列转换为欧洲柏林时区的毫秒级日期时间
    result = df.astype("datetime64[ms, Europe/Berlin]")
    # 断言结果 DataFrame 中的数据块没有引用
    assert result._mgr._has_no_reference(0)
    # 断言转换后的 DataFrame 的日期时间列与原始 DataFrame 的日期时间列不共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))


# 定义一个函数，用于测试将 DataFrame 的时间戳列转换为 Arrow 格式的时间戳的影响
def test_astype_arrow_timestamp():
    # 检查是否导入了 pyarrow 库，如果未导入则跳过测试
    pytest.importorskip("pyarrow")
    # 创建一个 DataFrame 对象 `df`，其中包含一个名为 "a" 的列，列的值是两个特定的时间戳
    df = DataFrame(
        {
            "a": [
                Timestamp("2020-01-01 01:01:01.000001"),
                Timestamp("2020-01-01 01:01:01.000001"),
            ]
        },
        dtype="M8[ns]",  # 设置列的数据类型为纳秒精度的时间戳
    )
    
    # 将 DataFrame `df` 的数据类型转换为 `timestamp[ns][pyarrow]`，并将结果存储在 `result` 中
    result = df.astype("timestamp[ns][pyarrow]")
    
    # 断言条件：确保 `result` 的第一个元素的内部管理器 `_mgr` 不是无引用状态
    assert not result._mgr._has_no_reference(0)
    
    # 如果当前的 PyArrow 版本小于 12.0
    if pa_version_under12p0:
        # 断言条件：确保原始 DataFrame `df` 的列 "a" 与 `result` 的列 "a" 不共享内存
        assert not np.shares_memory(
            get_array(df, "a"), get_array(result, "a")._pa_array
        )
    else:
        # 断言条件：确保原始 DataFrame `df` 的列 "a" 与 `result` 的列 "a" 共享内存
        assert np.shares_memory(get_array(df, "a"), get_array(result, "a")._pa_array)
# 定义一个函数用于测试数据类型转换，推断对象类型
def test_convert_dtypes_infer_objects():
    # 创建一个包含字符串的序列
    ser = Series(["a", "b", "c"])
    # 复制序列以备后用
    ser_orig = ser.copy()
    # 调用convert_dtypes方法进行数据类型转换，禁用整数、布尔、浮点数和字符串的转换
    result = ser.convert_dtypes(
        convert_integer=False,
        convert_boolean=False,
        convert_floating=False,
        convert_string=False,
    )

    # 断言：验证结果序列与原始序列共享内存
    assert np.shares_memory(get_array(ser), get_array(result))
    # 修改结果序列的第一个元素为"x"
    result.iloc[0] = "x"
    # 断言：验证修改后的结果序列与原始序列相等
    tm.assert_series_equal(ser, ser_orig)


# 定义一个函数用于测试数据类型转换
def test_convert_dtypes():
    # 创建一个包含多列数据的数据帧
    df = DataFrame({"a": ["a", "b"], "b": [1, 2], "c": [1.5, 2.5], "d": [True, False]})
    # 复制数据帧以备后用
    df_orig = df.copy()
    # 调用convert_dtypes方法进行数据类型转换
    df2 = df.convert_dtypes()

    # 断言：验证转换后的数据帧与原始数据帧在列"a"上共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：验证转换后的数据帧与原始数据帧在列"d"上共享内存
    assert np.shares_memory(get_array(df2, "d"), get_array(df, "d"))
    # 断言：验证转换后的数据帧与原始数据帧在列"b"上共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # 断言：验证转换后的数据帧与原始数据帧在列"c"上共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 修改转换后的数据帧的第一行第一列的值为"x"
    df2.iloc[0, 0] = "x"
    # 断言：验证修改后的数据帧与原始数据帧相等
    tm.assert_frame_equal(df, df_orig)
```