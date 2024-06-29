# `D:\src\scipysrc\pandas\pandas\tests\arrays\masked\test_arrow_compat.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 测试框架，用于单元测试

import pandas as pd  # 导入 Pandas 数据分析库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块

pytestmark = pytest.mark.filterwarnings(  # 设置 Pytest 标记，忽略特定警告信息
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

pa = pytest.importorskip("pyarrow")  # 导入 Pyarrow 库，如果未安装则跳过执行

from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask  # 导入 Pandas Arrow 工具函数

arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES]  # 创建包含不同整数类型数据的数组列表
arrays += [pd.array([0.1, 0.2, 0.3, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES]  # 添加包含不同浮点数类型数据的数组列表
arrays += [pd.array([True, False, True, None], dtype="boolean")]  # 添加布尔类型数据的数组

@pytest.fixture(params=arrays, ids=[a.dtype.name for a in arrays])  # 定义测试用例参数化的 Fixture
def data(request):
    """
    Fixture returning parametrized array from given dtype, including integer,
    float and boolean
    """
    return request.param  # 返回 Fixture 参数化的数据

def test_arrow_array(data):
    arr = pa.array(data)  # 使用 Pyarrow 将数据转换为 Arrow 数组
    expected = pa.array(
        data.to_numpy(object, na_value=None),  # 将 Pandas 数组转换为 NumPy 数组对象
        type=pa.from_numpy_dtype(data.dtype.numpy_dtype),  # 根据 NumPy 数据类型创建 Pyarrow 类型对象
    )
    assert arr.equals(expected)  # 断言 Arrow 数组与预期结果相等

def test_arrow_roundtrip(data):
    df = pd.DataFrame({"a": data})  # 创建包含数据列的 Pandas 数据帧
    table = pa.table(df)  # 使用 Pyarrow 创建表格
    assert table.field("a").type == str(data.dtype.numpy_dtype)  # 断言表格中字段的类型与数据类型相匹配

    result = table.to_pandas()  # 将 Pyarrow 表格转换为 Pandas 数据帧
    assert result["a"].dtype == data.dtype  # 断言转换后的数据帧列的数据类型与原始数据类型相同
    tm.assert_frame_equal(result, df)  # 使用 Pandas 测试工具断言转换后的数据帧与原始数据帧相等

def test_dataframe_from_arrow_types_mapper():
    def types_mapper(arrow_type):
        if pa.types.is_boolean(arrow_type):
            return pd.BooleanDtype()  # 如果 Arrow 类型为布尔类型，返回 Pandas 布尔类型
        elif pa.types.is_integer(arrow_type):
            return pd.Int64Dtype()  # 如果 Arrow 类型为整数类型，返回 Pandas Int64 类型

    bools_array = pa.array([True, None, False], type=pa.bool_())  # 创建布尔类型的 Arrow 数组
    ints_array = pa.array([1, None, 2], type=pa.int64())  # 创建整数类型的 Arrow 数组
    small_ints_array = pa.array([-1, 0, 7], type=pa.int8())  # 创建小整数类型的 Arrow 数组
    record_batch = pa.RecordBatch.from_arrays(
        [bools_array, ints_array, small_ints_array], ["bools", "ints", "small_ints"]  # 从数组创建记录批次
    )
    result = record_batch.to_pandas(types_mapper=types_mapper)  # 使用类型映射器将 Arrow 记录批次转换为 Pandas 数据帧
    bools = pd.Series([True, None, False], dtype="boolean")  # 创建预期的布尔类型 Series
    ints = pd.Series([1, None, 2], dtype="Int64")  # 创建预期的整数类型 Series
    small_ints = pd.Series([-1, 0, 7], dtype="Int64")  # 创建预期的整数类型 Series
    expected = pd.DataFrame({"bools": bools, "ints": ints, "small_ints": small_ints})  # 创建预期的数据帧
    tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试工具断言结果数据帧与预期数据帧相等

def test_arrow_load_from_zero_chunks(data):
    # GH-41040

    df = pd.DataFrame({"a": data[0:0]})  # 创建空数据帧
    table = pa.table(df)  # 使用 Pyarrow 创建表格
    assert table.field("a").type == str(data.dtype.numpy_dtype)  # 断言表格中字段的类型与数据类型相匹配
    table = pa.table(
        [pa.chunked_array([], type=table.field("a").type)], schema=table.schema  # 创建包含空块数组的表格
    )
    result = table.to_pandas()  # 将 Pyarrow 表格转换为 Pandas 数据帧
    assert result["a"].dtype == data.dtype  # 断言转换后的数据帧列的数据类型与原始数据类型相同
    tm.assert_frame_equal(result, df)  # 使用 Pandas 测试工具断言转换后的数据帧与原始数据帧相等

def test_arrow_from_arrow_uint():
    # https://github.com/pandas-dev/pandas/issues/31896
    # possible mismatch in types

    dtype = pd.UInt32Dtype()  # 创建 Pandas UInt32 类型
    result = dtype.__from_arrow__(pa.array([1, 2, 3, 4, None], type="int64"))  # 将 Arrow 数组转换为 Pandas 扩展数组
    expected = pd.array([1, 2, 3, 4, None], dtype="UInt32")  # 创建预期的 Pandas 扩展数组

    tm.assert_extension_array_equal(result, expected)  # 使用 Pandas 测试工具断言扩展数组相等

def test_arrow_sliced(data):
    pass  # 占位符测试函数，不执行任何操作
    # 创建一个 Pandas 数据帧，列名为 'a'，数据为传入的变量 data
    df = pd.DataFrame({"a": data})
    # 使用 Pandas Arrow 库创建一个 Arrow 表格
    table = pa.table(df)
    # 从 Arrow 表格中切片获取第 2 行到最后一行的数据，转换为 Pandas 数据帧
    result = table.slice(2, None).to_pandas()
    # 创建预期的 Pandas 数据帧，包含从第 2 行到最后一行的数据，并重新设置索引
    expected = df.iloc[2:].reset_index(drop=True)
    # 使用测试框架检查切片后的结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)

    # 处理数据帧中的缺失值，用 data[0] 填充
    df2 = df.fillna(data[0])
    # 使用 Pandas Arrow 库创建一个 Arrow 表格
    table = pa.table(df2)
    # 从 Arrow 表格中切片获取第 2 行到最后一行的数据，转换为 Pandas 数据帧
    result = table.slice(2, None).to_pandas()
    # 创建预期的 Pandas 数据帧，包含从第 2 行到最后一行的数据，并重新设置索引
    expected = df2.iloc[2:].reset_index(drop=True)
    # 使用测试框架检查切片后的结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)
@pytest.fixture
def np_dtype_to_arrays(any_real_numpy_dtype):
    """
    Fixture returning actual and expected dtype, pandas and numpy arrays and
    mask from a given numpy dtype
    """
    # 将任意真实的 numpy 数据类型转换为 numpy.dtype 对象
    np_dtype = np.dtype(any_real_numpy_dtype)
    # 使用 numpy 数据类型创建 pyarrow 中的数据类型对象
    pa_type = pa.from_numpy_dtype(np_dtype)

    # 创建包含特定数据的 pyarrow 数组，包括一个未指定值，以创建位掩码缓冲区
    pa_array = pa.array([0, 1, 2, None], type=pa_type)
    # 创建预期的 numpy 数组，只比较前三个值，因为位掩码的最后一个值不指定
    np_expected = np.array([0, 1, 2], dtype=np_dtype)
    # 创建预期的位掩码数组，最后一个值为 False
    mask_expected = np.array([True, True, True, False])
    return np_dtype, pa_array, np_expected, mask_expected


def test_pyarrow_array_to_numpy_and_mask(np_dtype_to_arrays):
    """
    Test conversion from pyarrow array to numpy array.

    Modifies the pyarrow buffer to contain padding and offset, which are
    considered valid buffers by pyarrow.

    Also tests empty pyarrow arrays with non empty buffers.
    See https://github.com/pandas-dev/pandas/issues/40896
    """
    # 解包 fixture 提供的返回值
    np_dtype, pa_array, np_expected, mask_expected = np_dtype_to_arrays
    # 转换 pyarrow 数组为 numpy 数组和掩码数组
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array, np_dtype)
    # 断言前三个值的 numpy 数组与预期值相等
    tm.assert_numpy_array_equal(data[:3], np_expected)
    # 断言掩码数组与预期掩码数组相等
    tm.assert_numpy_array_equal(mask, mask_expected)

    # 获取 pyarrow 数组的缓冲区
    mask_buffer = pa_array.buffers()[0]
    data_buffer = pa_array.buffers()[1]
    data_buffer_bytes = pa_array.buffers()[1].to_pybytes()

    # 向缓冲区添加尾部填充
    data_buffer_trail = pa.py_buffer(data_buffer_bytes + b"\x00")
    # 创建带有填充的新的 pyarrow 数组
    pa_array_trail = pa.Array.from_buffers(
        type=pa_array.type,
        length=len(pa_array),
        buffers=[mask_buffer, data_buffer_trail],
        offset=pa_array.offset,
    )
    # 验证新的 pyarrow 数组的有效性
    pa_array_trail.validate()
    # 再次转换为 numpy 数组和掩码数组，并进行断言
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_trail, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)

    # 向缓冲区添加偏移量
    offset = b"\x00" * (pa_array.type.bit_width // 8)
    data_buffer_offset = pa.py_buffer(offset + data_buffer_bytes)
    mask_buffer_offset = pa.py_buffer(b"\x0e")
    # 创建带有偏移量的新的 pyarrow 数组
    pa_array_offset = pa.Array.from_buffers(
        type=pa_array.type,
        length=len(pa_array),
        buffers=[mask_buffer_offset, data_buffer_offset],
        offset=pa_array.offset + 1,
    )
    # 验证新的 pyarrow 数组的有效性
    pa_array_offset.validate()
    # 再次转换为 numpy 数组和掩码数组，并进行断言
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)

    # 空数组
    np_expected_empty = np.array([], dtype=np_dtype)
    mask_expected_empty = np.array([], dtype=np.bool_)

    # 创建长度为 0 的新的 pyarrow 数组
    pa_array_offset = pa.Array.from_buffers(
        type=pa_array.type,
        length=0,
        buffers=[mask_buffer, data_buffer],
        offset=pa_array.offset,
    )
    # 验证新的 pyarrow 数组的有效性
    pa_array_offset.validate()
    # 使用 pyarrow_array_to_numpy_and_mask 函数将 pyarrow 数组转换为 NumPy 数组和掩码数组
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    # 使用测试框架中的 assert_numpy_array_equal 函数，验证 data 数组的前三个元素是否与 np_expected_empty 数组相等
    tm.assert_numpy_array_equal(data[:3], np_expected_empty)
    # 使用测试框架中的 assert_numpy_array_equal 函数，验证 mask 数组是否与 mask_expected_empty 数组相等
    tm.assert_numpy_array_equal(mask, mask_expected_empty)
@pytest.mark.parametrize(
    "arr", [pa.nulls(10), pa.chunked_array([pa.nulls(4), pa.nulls(6)])]
)
# 使用 pytest 的参数化装饰器，为测试函数 test_from_arrow_null 注入不同的参数 arr，
# 分别是长度为 10 的空值数组和包含两个部分的 chunked 数组
def test_from_arrow_null(data, arr):
    # 调用 data 对象的 dtype 属性的 __from_arrow__ 方法，将 arr 转换为相应数据类型
    res = data.dtype.__from_arrow__(arr)
    # 断言结果 res 中所有值都是缺失值
    assert res.isna().all()
    # 断言结果 res 的长度为 10
    assert len(res) == 10


def test_from_arrow_type_error(data):
    # 确保当传入错误的数组类型时，__from_arrow__ 方法抛出 TypeError

    # 创建一个包含 data 数据的 pyarrow 数组，然后将其强制转换为字符串类型数组
    arr = pa.array(data).cast("string")
    # 使用 pytest.raises 断言，期望捕获到 TypeError 异常
    with pytest.raises(TypeError, match=None):
        # 执行 data.dtype 的 __from_arrow__ 方法，传入错误类型的数组 arr
        # 我们只验证抛出 TypeError 异常，而不验证具体的错误信息内容
        data.dtype.__from_arrow__(arr)
```