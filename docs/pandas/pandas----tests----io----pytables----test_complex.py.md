# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_complex.py`

```
# 导入必要的库
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

import pandas as pd  # 导入Pandas库，用于数据操作和分析
from pandas import (  # 从Pandas中导入DataFrame和Series类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入Pandas的测试工具模块
from pandas.tests.io.pytables.common import ensure_clean_store  # 导入确保清理存储的函数

from pandas.io.pytables import read_hdf  # 从Pandas的PyTables IO模块中导入read_hdf函数


def test_complex_fixed(tmp_path, setup_path):
    # 创建一个复数类型为np.complex64的DataFrame对象
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)).astype(np.complex64),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    # 构建临时路径
    path = tmp_path / setup_path
    # 将DataFrame对象写入HDF5文件中，键名为"df"
    df.to_hdf(path, key="df")
    # 从HDF5文件中重新读取名为"df"的数据集
    reread = read_hdf(path, "df")
    # 断言两个DataFrame对象是否相等
    tm.assert_frame_equal(df, reread)

    # 创建一个复数类型为np.complex128的新DataFrame对象
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)).astype(np.complex128),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    # 再次构建临时路径
    path = tmp_path / setup_path
    # 将新的DataFrame对象写入HDF5文件中，键名为"df"
    df.to_hdf(path, key="df")
    # 从HDF5文件中重新读取名为"df"的数据集
    reread = read_hdf(path, "df")
    # 再次断言两个DataFrame对象是否相等
    tm.assert_frame_equal(df, reread)


def test_complex_table(tmp_path, setup_path):
    # 创建一个复数类型为np.complex64的DataFrame对象
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)).astype(np.complex64),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    # 构建临时路径
    path = tmp_path / setup_path
    # 将DataFrame对象以"table"格式写入HDF5文件中，键名为"df"
    df.to_hdf(path, key="df", format="table")
    # 从HDF5文件中重新读取名为"df"的数据集
    reread = read_hdf(path, key="df")
    # 断言两个DataFrame对象是否相等
    tm.assert_frame_equal(df, reread)

    # 创建一个复数类型为np.complex128的新DataFrame对象
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)).astype(np.complex128),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    # 再次构建临时路径
    path = tmp_path / setup_path
    # 将新的DataFrame对象以"table"格式写入HDF5文件中，键名为"df"，写入模式为"overwrite"
    df.to_hdf(path, key="df", format="table", mode="w")
    # 从HDF5文件中重新读取名为"df"的数据集
    reread = read_hdf(path, "df")
    # 再次断言两个DataFrame对象是否相等
    tm.assert_frame_equal(df, reread)


def test_complex_mixed_fixed(tmp_path, setup_path):
    # 创建包含复数类型为np.complex64和np.complex128的混合DataFrame对象
    complex64 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex64
    )
    complex128 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex128
    )
    df = DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "d"],
            "C": complex64,
            "D": complex128,
            "E": [1.0, 2.0, 3.0, 4.0],
        },
        index=list("abcd"),
    )

    # 构建临时路径
    path = tmp_path / setup_path
    # 将DataFrame对象写入HDF5文件中，键名为"df"
    df.to_hdf(path, key="df")
    # 从HDF5文件中重新读取名为"df"的数据集
    reread = read_hdf(path, "df")
    # 断言两个DataFrame对象是否相等
    tm.assert_frame_equal(df, reread)


def test_complex_mixed_table(tmp_path, setup_path):
    # 创建包含复数类型为np.complex64和np.complex128的混合DataFrame对象
    complex64 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex64
    )
    complex128 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex128
    )
    df = DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "d"],
            "C": complex64,
            "D": complex128,
            "E": [1.0, 2.0, 3.0, 4.0],
        },
        index=list("abcd"),
    )

    # 使用ensure_clean_store函数确保存储的清理，同时创建一个临时存储区域
    with ensure_clean_store(setup_path) as store:
        # 将DataFrame对象追加到存储区域中，设置数据列为["A", "B"]
        store.append("df", df, data_columns=["A", "B"])
        # 从存储区域中选择名为"df"且"A>2"的数据集
        result = store.select("df", where="A>2")
        # 断言筛选后的DataFrame对象是否与预期的部分相等
        tm.assert_frame_equal(df.loc[df.A > 2], result)

    # 构建临时路径
    path = tmp_path / setup_path
    # 将 DataFrame 对象 df 写入 HDF 文件，使用指定的路径、键和表格格式
    df.to_hdf(path, key="df", format="table")
    # 从指定的 HDF 文件中重新读取数据，使用指定的路径和键
    reread = read_hdf(path, "df")
    # 使用测试工具检查 df 和从 HDF 文件中重新读取的数据 reread 是否相等
    tm.assert_frame_equal(df, reread)
# 测试复杂数据类型在不同维度下的固定格式写入和读取
def test_complex_across_dimensions_fixed(tmp_path, setup_path):
    # 创建复数类型的 numpy 数组
    complex128 = np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
    # 创建 Pandas Series 对象，并指定索引
    s = Series(complex128, index=list("abcd"))
    # 创建 Pandas DataFrame 对象，其中列 'A' 和 'B' 使用相同的 Series
    df = DataFrame({"A": s, "B": s})

    # 定义待测试的对象和比较函数列表
    objs = [s, df]
    comps = [tm.assert_series_equal, tm.assert_frame_equal]
    # 遍历对象列表和比较函数列表，逐个执行以下操作：
    for obj, comp in zip(objs, comps):
        # 组合临时路径和设置路径
        path = tmp_path / setup_path
        # 将对象以固定格式写入 HDF5 文件
        obj.to_hdf(path, key="obj", format="fixed")
        # 从 HDF5 文件中重新读取数据
        reread = read_hdf(path, "obj")
        # 使用比较函数比较原始对象和重新读取的对象
        comp(obj, reread)


# 测试复杂数据类型在不同维度下的表格格式写入和读取
def test_complex_across_dimensions(tmp_path, setup_path):
    # 创建复数类型的 numpy 数组
    complex128 = np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
    # 创建 Pandas Series 对象，并指定索引
    s = Series(complex128, index=list("abcd"))
    # 创建 Pandas DataFrame 对象，其中列 'A' 和 'B' 使用相同的 Series
    df = DataFrame({"A": s, "B": s})

    # 组合临时路径和设置路径
    path = tmp_path / setup_path
    # 将 DataFrame 以表格格式写入 HDF5 文件
    df.to_hdf(path, key="obj", format="table")
    # 从 HDF5 文件中重新读取数据
    reread = read_hdf(path, "obj")
    # 使用测试框架中的函数比较原始 DataFrame 和重新读取的 DataFrame
    tm.assert_frame_equal(df, reread)


# 测试当表格中包含复杂数据类型时的索引错误处理
def test_complex_indexing_error(setup_path):
    # 创建复数类型的 numpy 数组
    complex128 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex128
    )
    # 创建包含复数列的 Pandas DataFrame 对象，并指定索引
    df = DataFrame(
        {"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"], "C": complex128},
        index=list("abcd"),
    )

    # 定义错误消息内容
    msg = (
        "Columns containing complex values can be stored "
        "but cannot be indexed when using table format. "
        "Either use fixed format, set index=False, "
        "or do not include the columns containing complex "
        "values to data_columns when initializing the table."
    )

    # 使用带清理存储的上下文环境来处理 HDF5 存储
    with ensure_clean_store(setup_path) as store:
        # 预期抛出 TypeError 异常，且异常信息匹配设定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 将 DataFrame 追加到存储中，指定包含复数列 'C' 作为数据列
            store.append("df", df, data_columns=["C"])


# 测试当 Series 包含复杂数据类型时的错误处理
def test_complex_series_error(tmp_path, setup_path):
    # 创建复数类型的 numpy 数组
    complex128 = np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
    # 创建 Pandas Series 对象，并指定索引
    s = Series(complex128, index=list("abcd"))

    # 定义错误消息内容
    msg = (
        "Columns containing complex values can be stored "
        "but cannot be indexed when using table format. "
        "Either use fixed format, set index=False, "
        "or do not include the columns containing complex "
        "values to data_columns when initializing the table."
    )

    # 组合临时路径和设置路径
    path = tmp_path / setup_path
    # 预期抛出 TypeError 异常，且异常信息匹配设定的错误消息
    with pytest.raises(TypeError, match=msg):
        # 将 Series 以表格格式写入 HDF5 文件
        s.to_hdf(path, key="obj", format="t")

    # 将 Series 以表格格式写入 HDF5 文件，不包含索引
    s.to_hdf(path, key="obj", format="t", index=False)
    # 从 HDF5 文件中重新读取数据
    reread = read_hdf(path, "obj")
    # 使用测试框架中的函数比较原始 Series 和重新读取的 Series
    tm.assert_series_equal(s, reread)


# 测试在存储中追加复杂数据类型的 DataFrame
def test_complex_append(setup_path):
    # 创建包含复数列的 Pandas DataFrame 对象和随机数据列
    df = DataFrame(
        {
            "a": np.random.default_rng(2).standard_normal(100).astype(np.complex128),
            "b": np.random.default_rng(2).standard_normal(100),
        }
    )

    # 使用带清理存储的上下文环境来处理 HDF5 存储
    with ensure_clean_store(setup_path) as store:
        # 向存储中追加 DataFrame，指定 'b' 列作为数据列
        store.append("df", df, data_columns=["b"])
        # 再次向存储中追加 DataFrame
        store.append("df", df)
        # 从存储中选择数据集 'df' 并比较与合并后的 DataFrame
        result = store.select("df")
        tm.assert_frame_equal(pd.concat([df, df], axis=0), result)
```