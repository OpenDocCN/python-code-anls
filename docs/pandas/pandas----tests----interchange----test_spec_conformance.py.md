# `D:\src\scipysrc\pandas\pandas\tests\interchange\test_spec_conformance.py`

```
"""
A verbatim copy (vendored) of the spec tests.
Taken from https://github.com/data-apis/dataframe-api
"""

# 导入必要的库
import ctypes  # 导入 ctypes 库
import math  # 导入 math 库

import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 pandas 库并使用别名 pd

# 定义测试数据生成器
@pytest.fixture
def df_from_dict():
    def maker(dct, is_categorical=False):
        # 使用给定字典创建 DataFrame 对象
        df = pd.DataFrame(dct)
        # 如果指定为分类数据，则将 DataFrame 转换为分类类型
        return df.astype("category") if is_categorical else df

    return maker

# 参数化测试，包括不同的测试数据和对应的标识符
@pytest.mark.parametrize(
    "test_data",
    [
        {"a": ["foo", "bar"], "b": ["baz", "qux"]},  # 字符串数据
        {"a": [1.5, 2.5, 3.5], "b": [9.2, 10.5, 11.8]},  # 浮点数数据
        {"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]},  # 整数数据
    ],
    ids=["str_data", "float_data", "int_data"],  # 对应的测试数据标识符
)
def test_only_one_dtype(test_data, df_from_dict):
    columns = list(test_data.keys())
    df = df_from_dict(test_data)
    dfX = df.__dataframe__()

    column_size = len(test_data[columns[0]])
    for column in columns:
        colX = dfX.get_column_by_name(column)
        # 断言每列的空值数量为 0
        assert colX.null_count == 0
        # 断言空值数量的数据类型为整数
        assert isinstance(colX.null_count, int)
        # 断言每列的大小与测试数据中对应列的大小相等
        assert colX.size() == column_size
        # 断言每列的偏移量为 0
        assert colX.offset == 0

# 测试混合数据类型的处理
def test_mixed_dtypes(df_from_dict):
    df = df_from_dict(
        {
            "a": [1, 2, 3],  # 整数类型
            "b": [3, 4, 5],  # 整数类型
            "c": [1.5, 2.5, 3.5],  # 浮点数类型
            "d": [9, 10, 11],  # 整数类型
            "e": [True, False, True],  # 布尔类型
            "f": ["a", "", "c"],  # 字符串类型
        }
    )
    dfX = df.__dataframe__()
    # 每列的数据类型的含义解释需要参考规范；我们无法在这里导入规范，因为此文件可能被任何地方复制；
    # dtype[0] 的值在上面已经解释过了

    columns = {"a": 0, "b": 0, "c": 2, "d": 0, "e": 20, "f": 21}

    for column, kind in columns.items():
        colX = dfX.get_column_by_name(column)
        # 断言每列的空值数量为 0
        assert colX.null_count == 0
        # 断言空值数量的数据类型为整数
        assert isinstance(colX.null_count, int)
        # 断言每列的大小为 3
        assert colX.size() == 3
        # 断言每列的偏移量为 0
        assert colX.offset == 0
        # 断言每列的数据类型的第一个元素与预期的类型相符
        assert colX.dtype[0] == kind

    # 断言特定列的数据类型的第二个元素为 64
    assert dfX.get_column_by_name("c").dtype[1] == 64

# 测试包含 NaN 值的浮点数列
def test_na_float(df_from_dict):
    df = df_from_dict({"a": [1.0, math.nan, 2.0]})
    dfX = df.__dataframe__()
    colX = dfX.get_column_by_name("a")
    # 断言列中的空值数量为 1
    assert colX.null_count == 1
    # 断言空值数量的数据类型为整数
    assert isinstance(colX.null_count, int)

# 测试非分类数据列的处理
def test_noncategorical(df_from_dict):
    df = df_from_dict({"a": [1, 2, 3]})
    dfX = df.__dataframe__()
    colX = dfX.get_column_by_name("a")
    # 使用 pytest 检查非分类列描述时抛出的类型错误异常
    with pytest.raises(TypeError, match=".*categorical.*"):
        colX.describe_categorical

# 测试分类数据列的处理
def test_categorical(df_from_dict):
    df = df_from_dict(
        {"weekday": ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", "Sat", "Sun"]},
        is_categorical=True,  # 指定该列为分类数据
    )

    colX = df.__dataframe__().get_column_by_name("weekday")
    categorical = colX.describe_categorical
    # 断言分类数据的排序属性为布尔类型
    assert isinstance(categorical["is_ordered"], bool)
    # 断言语句，用于检查变量 `categorical["is_dictionary"]` 是否为布尔类型
    assert isinstance(categorical["is_dictionary"], bool)
```python`
def test_dataframe(df_from_dict):
    # 使用字典数据创建 DataFrame 对象
    df = df_from_dict(
        {"x": [True, True, False], "y": [1, 2, 0], "z": [9.2, 10.5, 11.8]}
    )
    # 获取 DataFrame 对象的底层数据结构
    dfX = df.__dataframe__()

    # 检查 DataFrame 中的列数是否为 3
    assert dfX.num_columns() == 3
    # 检查 DataFrame 中的行数是否为 3
    assert dfX.num_rows() == 3
    # 检查 DataFrame 中的块数是否为 1
    assert dfX.num_chunks() == 1
    # 检查列名是否为 ["x", "y", "z"]
    assert list(dfX.column_names()) == ["x", "y", "z"]
    # 检查选择列（0, 2）后的列名是否等于选择列名（"x", "z"）后的列名
    assert list(dfX.select_columns((0, 2)).column_names()) == list(
        dfX.select_columns_by_name(("x", "z")).column_names()
    )


@pytest.mark.parametrize(["size", "n_chunks"], [(10, 3), (12, 3), (12, 5)])
# 使用参数化测试，测试不同数据大小和块数的 DataFrame
def test_df_get_chunks(size, n_chunks, df_from_dict):
    # 创建一个包含指定大小的整数序列的 DataFrame
    df = df_from_dict({"x": list(range(size))})
    # 获取 DataFrame 的底层数据结构
    dfX = df.__dataframe__()
    # 获取数据块列表
    chunks = list(dfX.get_chunks(n_chunks))
    # 检查块的数量是否等于指定的块数
    assert len(chunks) == n_chunks
    # 检查所有块的行数之和是否等于数据大小
    assert sum(chunk.num_rows() for chunk in chunks) == size


@pytest.mark.parametrize(["size", "n_chunks"], [(10, 3), (12, 3), (12, 5)])
# 使用参数化测试，测试不同数据大小和块数的列数据块
def test_column_get_chunks(size, n_chunks, df_from_dict):
    # 创建一个包含指定大小的整数序列的 DataFrame
    df = df_from_dict({"x": list(range(size))})
    # 获取 DataFrame 的底层数据结构
    dfX = df.__dataframe__()
    # 获取列数据块列表
    chunks = list(dfX.get_column(0).get_chunks(n_chunks))
    # 检查块的数量是否等于指定的块数
    assert len(chunks) == n_chunks
    # 检查所有块的大小之和是否等于数据大小
    assert sum(chunk.size() for chunk in chunks) == size


def test_get_columns(df_from_dict):
    # 创建包含两列数据的 DataFrame
    df = df_from_dict({"a": [0, 1], "b": [2.5, 3.5]})
    # 获取 DataFrame 的底层数据结构
    dfX = df.__dataframe__()
    # 遍历 DataFrame 的所有列
    for colX in dfX.get_columns():
        # 检查列的大小是否为 2
        assert colX.size() == 2
        # 检查列的块数是否为 1
        assert colX.num_chunks() == 1
    # 检查第一列的数据类型是否为整数（0）
    assert dfX.get_column(0).dtype[0] == 0  # INT
    # 检查第二列的数据类型是否为浮点数（2）
    assert dfX.get_column(1).dtype[0] == 2  # FLOAT


def test_buffer(df_from_dict):
    # 创建包含指定数据的 DataFrame
    arr = [0, 1, -1]
    df = df_from_dict({"a": arr})
    # 获取 DataFrame 的底层数据结构
    dfX = df.__dataframe__()
    # 获取第一列的对象
    colX = dfX.get_column(0)
    # 获取列的缓冲区
    bufX = colX.get_buffers()

    # 获取数据缓冲区和数据类型
    dataBuf, dataDtype = bufX["data"]

    # 检查数据缓冲区的大小是否大于 0
    assert dataBuf.bufsize > 0
    # 检查数据缓冲区的指针是否不为 0
    assert dataBuf.ptr != 0
    # 获取数据缓冲区的设备信息
    device, _ = dataBuf.__dlpack_device__()

    # 检查数据类型的第一个字节是否为整数（0）
    assert dataDtype[0] == 0  # INT

    if device == 1:  # CPU-only as we're going to directly read memory here
        # 获取数据位宽
        bitwidth = dataDtype[1]
        # 根据位宽选择相应的 ctypes 类型
        ctype = {
            8: ctypes.c_int8,
            16: ctypes.c_int16,
            32: ctypes.c_int32,
            64: ctypes.c_int64,
        }[bitwidth]

        # 遍历数据数组，检查缓冲区中的值是否与预期值匹配
        for idx, truth in enumerate(arr):
            val = ctype.from_address(dataBuf.ptr + idx * (bitwidth // 8)).value
            assert val == truth, f"Buffer at index {idx} mismatch"
```