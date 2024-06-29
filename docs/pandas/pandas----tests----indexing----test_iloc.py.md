# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_iloc.py`

```
"""test positional based indexing with iloc"""

# 导入所需的模块和类
from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
    NA,
    Categorical,
    CategoricalDtype,
    DataFrame,
    Index,
    Interval,
    NaT,
    Series,
    Timestamp,
    array,
    concat,
    date_range,
    interval_range,
    isna,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises

# 用于匹配 numpy 报错信息的正则表达式
_slice_iloc_msg = re.escape(
    "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) "
    "and integer or boolean arrays are valid indices"
)


class TestiLoc:
    # 参数化测试用例，测试不同的索引方式
    @pytest.mark.parametrize("key", [2, -1, [0, 1, 2]])
    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcd"), dtype=object),
            Index([2, 4, "null", 8], dtype=object),
            date_range("20130101", periods=4),
            Index(range(0, 8, 2), dtype=np.float64),
            Index([]),
        ],
    )
    def test_iloc_getitem_int_and_list_int(self, key, frame_or_series, index, request):
        # 创建数据对象
        obj = frame_or_series(range(len(index)), index=index)
        # 调用辅助函数进行索引测试，并验证是否引发 IndexError 异常
        check_indexing_smoketest_or_raises(
            obj,
            "iloc",
            key,
            fails=IndexError,
        )

        # 数组索引的情况下，确保单个索引器返回正确的类型


class TestiLocBaseIndependent:
    """Tests Independent Of Base Class"""

    # 参数化测试用例，测试不同的索引方式
    @pytest.mark.parametrize(
        "key",
        [
            slice(None),
            slice(3),
            range(3),
            [0, 1, 2],
            Index(range(3)),
            np.asarray([0, 1, 2]),
        ],
    )
    def test_iloc_setitem_fullcol_categorical(self, indexer_li, key):
        # 创建包含对象的 DataFrame
        frame = DataFrame({0: range(3)}, dtype=object)

        # 创建分类数据
        cat = Categorical(["alpha", "beta", "gamma"])

        # 确保 DataFrame 能够容纳分类数据
        assert frame._mgr.blocks[0]._can_hold_element(cat)

        # 复制 DataFrame
        df = frame.copy()
        orig_vals = df.values

        # 使用索引器设置数据
        indexer_li(df)[key, 0] = cat

        # 预期的 DataFrame
        expected = DataFrame({0: cat}).astype(object)
        assert np.shares_memory(df[0].values, orig_vals)

        # 断言 DataFrame 相等
        tm.assert_frame_equal(df, expected)

        # 检查我们没有对分类数据的视图
        df.iloc[0, 0] = "gamma"
        assert cat[0] != "gamma"

        # 在旧版中（"split" 路径），我们总是覆盖列。从 2.0 开始，我们正确地写入列，保留对象数据类型。
        frame = DataFrame({0: np.array([0, 1, 2], dtype=object), 1: range(3)})
        df = frame.copy()
        indexer_li(df)[key, 0] = cat
        expected = DataFrame({0: Series(cat.astype(object), dtype=object), 1: range(3)})
        tm.assert_frame_equal(df, expected)
    # 测试函数：测试在 inplace 操作中设置 iloc 的行为
    def test_iloc_setitem_ea_inplace(self, frame_or_series, index_or_series_or_array):
        # GH#38952 情况：未设置整列
        # 创建不含 NA 值的整数数组
        arr = array([1, 2, 3, 4])
        # 根据输入创建 DataFrame 或 Series 对象
        obj = frame_or_series(arr.to_numpy("i8"))

        # 根据对象类型选择操作的值
        if frame_or_series is Series:
            values = obj.values
        else:
            values = obj._mgr.blocks[0].values

        # 根据对象类型设置 iloc 的值
        if frame_or_series is Series:
            obj.iloc[:2] = index_or_series_or_array(arr[2:])
        else:
            obj.iloc[:2, 0] = index_or_series_or_array(arr[2:])

        # 预期的结果
        expected = frame_or_series(np.array([3, 4, 3, 4], dtype="i8"))
        # 断言结果与预期结果相等
        tm.assert_equal(obj, expected)

        # 检查是否真正进行了 inplace 操作
        if frame_or_series is Series:
            assert obj.values is not values
            assert np.shares_memory(obj.values, values)
        else:
            assert np.shares_memory(obj[0].values, values)

    # 测试函数：测试是否为标量访问
    def test_is_scalar_access(self):
        # GH#32085 对于带有重复索引的索引，_is_scalar_access 不影响
        index = Index([1, 2, 1])
        ser = Series(range(3), index=index)

        # 断言是否为标量访问
        assert ser.iloc._is_scalar_access((1,))

        df = ser.to_frame()
        # 断言是否为标量访问
        assert df.iloc._is_scalar_access((1, 0))

    @pytest.mark.parametrize("index,columns", [(np.arange(20), list("ABCDE"))])
    @pytest.mark.parametrize(
        "index_vals,column_vals",
        [
            ([slice(None), ["A", "D"]]),  # 用于测试非整数索引的情况
            (["1", "2"], slice(None)),    # 用于测试非整数索引的情况
            ([datetime(2019, 1, 1)], slice(None)),  # 用于测试非整数索引的情况
        ],
    )
    # 测试函数：测试 iloc 中非整数索引会引发错误
    def test_iloc_non_integer_raises(self, index, columns, index_vals, column_vals):
        # GH 25753
        # 创建具有指定形状的 DataFrame，用于测试
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), len(columns))),
            index=index,
            columns=columns,
        )
        # 期望引发的错误信息
        msg = ".iloc requires numeric indexers, got"
        # 断言引发特定类型的错误
        with pytest.raises(IndexError, match=msg):
            df.iloc[index_vals, column_vals]

    # 测试函数：测试 iloc 中使用无效标量索引会引发 TypeError
    def test_iloc_getitem_invalid_scalar(self, frame_or_series):
        # GH 21982
        # 创建包含数字的 DataFrame 对象
        obj = DataFrame(np.arange(100).reshape(10, 10))
        # 调用 get_obj 方法，获取适当的对象
        obj = tm.get_obj(obj, frame_or_series)

        # 断言使用无效标量索引会引发 TypeError 错误
        with pytest.raises(TypeError, match="Cannot index by location index"):
            obj.iloc["a"]

    # 测试函数：测试在数组操作中，负索引不会改变原始数组的内容
    def test_iloc_array_not_mutating_negative_indices(self):
        # GH 21867
        # 创建一个包含负数的数组副本
        array_with_neg_numbers = np.array([1, 2, -1])
        array_copy = array_with_neg_numbers.copy()
        # 创建一个 DataFrame 对象
        df = DataFrame(
            {"A": [100, 101, 102], "B": [103, 104, 105], "C": [106, 107, 108]},
            index=[1, 2, 3],
        )
        # 使用 iloc 进行数组操作
        df.iloc[array_with_neg_numbers]
        # 断言数组操作后数组内容不变
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)
        # 使用 iloc 进行列操作
        df.iloc[:, array_with_neg_numbers]
        # 再次断言数组操作后数组内容不变
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)
    def test_iloc_getitem_neg_int_can_reach_first_index(self):
        # GH10547 and GH10779
        # negative integers should be able to reach index 0
        # 创建一个包含两列的DataFrame对象，列"A"包含值[2, 3, 5]，列"B"包含值[7, 11, 13]
        df = DataFrame({"A": [2, 3, 5], "B": [7, 11, 13]}
        
        # 从DataFrame中选择"A"列，返回一个Series对象
        s = df["A"]

        # 从DataFrame中选择第一行作为期望结果，保存为Series对象expected
        expected = df.iloc[0]
        # 使用负整数索引-3从DataFrame中获取第一行，保存为Series对象result
        result = df.iloc[-3]
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

        # 从DataFrame中选择第一行作为期望结果，保存为DataFrame对象expected
        expected = df.iloc[[0]]
        # 使用负整数索引-3从DataFrame中获取第一行，保存为DataFrame对象result
        result = df.iloc[[-3]]
        # 断言DataFrame对象result与期望对象expected相等
        tm.assert_frame_equal(result, expected)

        # 从Series对象s中选择第一个元素作为期望结果，保存为expected
        expected = s.iloc[0]
        # 使用负整数索引-3从Series对象s中获取第一个元素，保存为result
        assert result == expected

        # 从Series对象s中选择第一个元素作为期望结果，保存为expected
        expected = s.iloc[[0]]
        # 使用负整数索引-3从Series对象s中获取第一个元素，保存为result
        tm.assert_series_equal(result, expected)

        # 检查GH10547中突出显示的长度为1的Series情况
        # 创建一个包含单个元素"a"的Series对象，指定index为["A"]
        expected = Series(["a"], index=["A"])
        # 使用负整数索引-1从Series对象expected中获取元素，保存为result
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_dups(self):
        # GH 6766
        # 创建包含两行的DataFrame对象，第一行{"A": None, "B": 1}，第二行{"A": 2, "B": 2}
        df1 = DataFrame([{"A": None, "B": 1}, {"A": 2, "B": 2}])
        # 创建包含两行的DataFrame对象，第一行{"A": 3, "B": 3}，第二行{"A": 4, "B": 4}
        df2 = DataFrame([{"A": 3, "B": 3}, {"A": 4, "B": 4}])
        # 沿轴1（列方向）连接df1和df2，创建一个新的DataFrame对象df
        df = concat([df1, df2], axis=1)

        # 使用交叉索引（iloc）获取第一行第一列的元素，保存为result
        result = df.iloc[0, 0]
        # 断言结果是否为NaN
        assert isna(result)

        # 使用交叉索引（iloc）获取第一行所有列的数据，保存为result
        result = df.iloc[0, :]
        # 创建一个期望的Series对象，包含NaN、1、3、3四个元素，指定index为["A", "B", "A", "B"]，name为0
        expected = Series([np.nan, 1, 3, 3], index=["A", "B", "A", "B"], name=0)
        # 断言结果Series对象result与期望Series对象expected相等
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_array(self):
        # 创建包含三行的DataFrame对象，每行包含三列数据
        df = DataFrame([
            {"A": 1, "B": 2, "C": 3},
            {"A": 100, "B": 200, "C": 300},
            {"A": 1000, "B": 2000, "C": 3000},
        ])

        # 使用索引数组[0]从DataFrame中获取第一行数据，保存为DataFrame对象expected
        expected = DataFrame([{"A": 1, "B": 2, "C": 3}])
        # 断言结果DataFrame对象df.iloc[[0]]与期望DataFrame对象expected相等
        tm.assert_frame_equal(df.iloc[[0]], expected)

        # 使用索引数组[0, 1]从DataFrame中获取第一行和第二行数据，保存为DataFrame对象expected
        expected = DataFrame([
            {"A": 1, "B": 2, "C": 3},
            {"A": 100, "B": 200, "C": 300}
        ])
        # 断言结果DataFrame对象df.iloc[[0, 1]]与期望DataFrame对象expected相等
        tm.assert_frame_equal(df.iloc[[0, 1]], expected)

        # 使用行索引数组[0, 2]和列索引数组[1, 2]从DataFrame中获取指定行和列的数据，保存为DataFrame对象result
        expected = DataFrame([
            {"B": 2, "C": 3},
            {"B": 2000, "C": 3000}
        ], index=[0, 2])
        result = df.iloc[[0, 2], [1, 2]]
        # 断言结果DataFrame对象result与期望DataFrame对象expected相等
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_bool(self):
        # 创建包含三行的DataFrame对象，每行包含三列数据
        df = DataFrame([
            {"A": 1, "B": 2, "C": 3},
            {"A": 100, "B": 200, "C": 300},
            {"A": 1000, "B": 2000, "C": 3000},
        ])

        # 使用布尔索引[True, True, False]从DataFrame中选择前两行数据，保存为DataFrame对象result
        expected = DataFrame([
            {"A": 1, "B": 2, "C": 3},
            {"A": 100, "B": 200, "C": 300}
        ])
        result = df.iloc[[True, True, False]]
        # 断言结果DataFrame对象result与期望DataFrame对象expected相等
        tm.assert_frame_equal(result, expected)

        # 使用lambda函数和布尔条件x.index % 2 == 0从DataFrame中选择符合条件的行，保存为DataFrame对象result
        expected = DataFrame([
            {"A": 1, "B": 2, "C": 3},
            {"A": 1000, "B": 2000, "C": 3000}
        ], index=[0, 2])
        result = df.iloc[lambda x: x.index % 2 == 0]
        # 断言结果DataFrame对象result与期望DataFrame对象expected相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("index", [[True, False], [True, False, True, False]])
    # 定义一个测试函数，用于测试通过布尔索引访问 Series 时的行为
    def test_iloc_getitem_bool_diff_len(self, index):
        # 创建一个包含整数的 Series 对象
        s = Series([1, 2, 3])
        # 生成一条关于错误长度的消息，用于测试断言
        msg = f"Boolean index has wrong length: {len(index)} instead of {len(s)}"
        # 使用 pytest 来验证预期的 IndexError 是否会被触发，并匹配指定的错误消息
        with pytest.raises(IndexError, match=msg):
            # 尝试通过 iloc 访问 Series，使用给定的布尔索引
            s.iloc[index]

    # 定义测试函数，用于测试通过切片访问 DataFrame 的行为
    def test_iloc_getitem_slice(self):
        # 创建一个 DataFrame 对象，包含了三个字典类型的数据行
        df = DataFrame(
            [
                {"A": 1, "B": 2, "C": 3},
                {"A": 100, "B": 200, "C": 300},
                {"A": 1000, "B": 2000, "C": 3000},
            ]
        )

        # 预期的 DataFrame，包含前两行的数据
        expected = DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 100, "B": 200, "C": 300}])
        # 使用 iloc 进行切片操作，获取前两行的结果
        result = df.iloc[:2]
        tm.assert_frame_equal(result, expected)

        # 预期的 DataFrame，包含第二行的前两列数据
        expected = DataFrame([{"A": 100, "B": 200}], index=[1])
        # 使用 iloc 进行切片操作，获取指定行列范围的结果
        result = df.iloc[1:2, 0:2]
        tm.assert_frame_equal(result, expected)

        # 预期的 DataFrame，只包含第一列和第三列的数据
        expected = DataFrame(
            [{"A": 1, "C": 3}, {"A": 100, "C": 300}, {"A": 1000, "C": 3000}]
        )
        # 使用 iloc 通过传入函数进行列索引选择，获取结果
        result = df.iloc[:, lambda df: [0, 2]]
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，用于测试包含重复列名的 DataFrame 的 iloc 操作
    def test_iloc_getitem_slice_dups(self):
        # 创建包含随机数据的两个 DataFrame 对象
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=["A", "A", "B", "B"],
        )
        df2 = DataFrame(
            np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
            columns=["A", "C"],
        )

        # 在列方向上连接两个 DataFrame 对象
        df = concat([df1, df2], axis=1)
        tm.assert_frame_equal(df.iloc[:, :4], df1)
        tm.assert_frame_equal(df.iloc[:, 4:], df2)

        # 在列方向上连接两个 DataFrame 对象（顺序相反）
        df = concat([df2, df1], axis=1)
        tm.assert_frame_equal(df.iloc[:, :2], df2)
        tm.assert_frame_equal(df.iloc[:, 2:], df1)

        # 预期的 DataFrame，包含第一列和第三列的数据（df1 的第一列重复）
        exp = concat([df2, df1.iloc[:, [0]]], axis=1)
        tm.assert_frame_equal(df.iloc[:, 0:3], exp)

        # 在行方向上连接两个相同的 DataFrame 对象
        df = concat([df, df], axis=0)
        tm.assert_frame_equal(df.iloc[0:10, :2], df2)
        tm.assert_frame_equal(df.iloc[0:10, 2:], df1)
        tm.assert_frame_equal(df.iloc[10:, :2], df2)
        tm.assert_frame_equal(df.iloc[10:, 2:], df1)

    # 定义测试函数，用于测试通过 iloc 进行赋值操作的行为
    def test_iloc_setitem(self):
        # 创建一个包含随机数据的 DataFrame 对象，具有自定义的索引和列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=np.arange(0, 8, 2),
            columns=np.arange(0, 12, 3),
        )

        # 使用 iloc 将指定位置的元素设置为特定值
        df.iloc[1, 1] = 1
        result = df.iloc[1, 1]
        assert result == 1

        # 使用 iloc 将指定列范围内的所有元素设置为零
        df.iloc[:, 2:3] = 0
        expected = df.iloc[:, 2:3]
        result = df.iloc[:, 2:3]
        tm.assert_frame_equal(result, expected)

        # 创建一个 Series 对象，并通过 iloc 对其进行切片和赋值操作
        s = Series(0, index=[4, 5, 6])
        s.iloc[1:2] += 1
        expected = Series([0, 1, 0], index=[4, 5, 6])
        tm.assert_series_equal(s, expected)
    def test_iloc_setitem_axis_argument(self):
        # 测试 iloc 的 axis 参数设置
        df = DataFrame([[6, "c", 10], [7, "d", 11], [8, "e", 12]])
        # 将第一列转换为对象类型
        df[1] = df[1].astype(object)
        expected = DataFrame([[6, "c", 10], [7, "d", 11], [5, 5, 5]])
        # 将期望结果的第一列也转换为对象类型
        expected[1] = expected[1].astype(object)
        # 使用 iloc 根据 axis=0 修改第三行的数据为 5
        df.iloc(axis=0)[2] = 5
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[6, "c", 10], [7, "d", 11], [8, "e", 12]])
        df[1] = df[1].astype(object)
        expected = DataFrame([[6, "c", 5], [7, "d", 5], [8, "e", 5]])
        expected[1] = expected[1].astype(object)
        # 使用 iloc 根据 axis=1 修改第三列的数据为 5
        df.iloc(axis=1)[2] = 5
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_list(self):
        # 使用 iloc 设置列表形式的项
        df = DataFrame(
            np.arange(9).reshape((3, 3)), index=["A", "B", "C"], columns=["A", "B", "C"]
        )
        # 使用 iloc 选择特定的行和列
        df.iloc[[0, 1], [1, 2]]
        # 将选择的行和列的值增加 100
        df.iloc[[0, 1], [1, 2]] += 100

        expected = DataFrame(
            np.array([0, 101, 102, 3, 104, 105, 6, 7, 8]).reshape((3, 3)),
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_pandas_object(self):
        # 测试 iloc 设置 pandas 对象
        s_orig = Series([0, 1, 2, 3])
        expected = Series([0, -1, -2, 3])

        s = s_orig.copy()
        # 使用 iloc 根据 Series 设置值
        s.iloc[Series([1, 2])] = [-1, -2]
        tm.assert_series_equal(s, expected)

        s = s_orig.copy()
        # 使用 iloc 根据 Index 设置值
        s.iloc[Index([1, 2])] = [-1, -2]
        tm.assert_series_equal(s, expected)

    def test_iloc_setitem_dups(self):
        # 测试 iloc 处理重复值
        # 使用 iloc 和 mask 从另一个 iloc 对齐处理
        df1 = DataFrame([{"A": None, "B": 1}, {"A": 2, "B": 2}])
        df2 = DataFrame([{"A": 3, "B": 3}, {"A": 4, "B": 4}])
        df = concat([df1, df2], axis=1)

        expected = df.fillna(3)
        # 选取需要填充的位置
        inds = np.isnan(df.iloc[:, 0])
        mask = inds[inds].index
        # 使用 iloc 根据 mask 修改数据
        df.iloc[mask, 0] = df.iloc[mask, 2]
        tm.assert_frame_equal(df, expected)

        # 删除跨块的重复列
        expected = DataFrame({0: [1, 2], 1: [3, 4]})
        expected.columns = ["B", "B"]
        del df["A"]
        tm.assert_frame_equal(df, expected)

        # 将结果赋值回自身
        df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]
        tm.assert_frame_equal(df, expected)

        # 反向操作 x 2
        df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
        df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
        tm.assert_frame_equal(df, expected)
    # 定义测试方法，用于测试在 DataFrame 中存在重复列的情况下，使用 iloc 进行赋值操作
    def test_iloc_setitem_frame_duplicate_columns_multiple_blocks(self):
        # 创建一个 DataFrame，包含重复列 "B"
        df = DataFrame([[0, 1], [2, 3]], columns=["B", "B"])

        # 将第一列的整数转换为浮点数，直接修改原数据
        df.iloc[:, 0] = df.iloc[:, 0].astype("f8")
        # 检查 DataFrame 内部数据块数量是否为1
        assert len(df._mgr.blocks) == 1

        # 如果赋值的浮点数超出了原整数列的容量，会产生警告，并执行强制类型转换
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.iloc[:, 0] = df.iloc[:, 0] + 0.5
        # 检查 DataFrame 内部数据块数量是否为2
        assert len(df._mgr.blocks) == 2

        # 创建一个预期结果的 DataFrame 副本
        expected = df.copy()

        # 将选定的区域重新赋值为自身（无实际修改效果，用于测试目的）
        df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]

        # 使用测试工具函数检查两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    # TODO: GH#27620 this test used to compare iloc against ix; check if this
    #  is redundant with another test comparing iloc against loc
    # 定义测试方法，用于测试 DataFrame 的 iloc 获取操作
    def test_iloc_getitem_frame(self):
        # 创建一个随机数据填充的 DataFrame，包含10行4列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            index=range(0, 20, 2),
            columns=range(0, 8, 2),
        )

        # 使用 iloc 获取第2行数据，与 loc 方法获取第4行数据进行比较
        result = df.iloc[2]
        exp = df.loc[4]
        tm.assert_series_equal(result, exp)

        # 使用 iloc 获取第(2, 2)个元素，与 loc 方法获取第(4, 4)个元素进行比较
        result = df.iloc[2, 2]
        exp = df.loc[4, 4]
        assert result == exp

        # 使用 iloc 进行切片操作，获取第4到第8行数据，与 loc 方法获取第8到第14行数据进行比较
        result = df.iloc[4:8]
        expected = df.loc[8:14]
        tm.assert_frame_equal(result, expected)

        # 使用 iloc 获取特定列范围的数据，与 loc 方法获取相同范围的列进行比较
        result = df.iloc[:, 2:3]
        expected = df.loc[:, 4:5]
        tm.assert_frame_equal(result, expected)

        # 使用 iloc 根据列表指定特定行的数据，与 loc 方法获取相同行进行比较
        result = df.iloc[[0, 1, 3]]
        expected = df.loc[[0, 2, 6]]
        tm.assert_frame_equal(result, expected)

        # 使用 iloc 根据列表同时指定特定行和列的数据，与 loc 方法获取相同的行列进行比较
        result = df.iloc[[0, 1, 3], [0, 1]]
        expected = df.loc[[0, 2, 6], [0, 2]]
        tm.assert_frame_equal(result, expected)

        # 使用 iloc 根据负索引指定特定行和列的数据，与 loc 方法获取相同的行列进行比较
        result = df.iloc[[-1, 1, 3], [-1, 1]]
        expected = df.loc[[18, 2, 6], [6, 2]]
        tm.assert_frame_equal(result, expected)

        # 使用 iloc 根据包含重复索引的列表指定特定行和列的数据，与 loc 方法获取相同的行列进行比较
        result = df.iloc[[-1, -1, 1, 3], [-1, 1]]
        expected = df.loc[[18, 18, 2, 6], [6, 2]]
        tm.assert_frame_equal(result, expected)

        # 使用 iloc 根据类似索引对象获取特定行的数据，与 loc 方法获取相同行进行比较
        s = Series(index=range(1, 5), dtype=object)
        result = df.iloc[s.index]
        expected = df.loc[[2, 4, 6, 8]]
        tm.assert_frame_equal(result, expected)
    `
        # 定义一个测试函数，用于测试 DataFrame 的 iloc 属性的行为
        def test_iloc_getitem_labelled_frame(self):
            # 尝试使用带标签的数据框
            # 创建一个包含随机标准正态分布数据的 DataFrame，使用字母作为索引和列名
            df = DataFrame(
                np.random.default_rng(2).standard_normal((10, 4)),
                index=list("abcdefghij"),
                columns=list("ABCD"),
            )
    
            # 获取 iloc 的结果，比较是否与 loc 方法获取的结果一致
            result = df.iloc[1, 1]
            exp = df.loc["b", "B"]
            assert result == exp
    
            # 获取 iloc 的切片结果，比较是否与 loc 方法获取的结果一致
            result = df.iloc[:, 2:3]
            expected = df.loc[:, ["C"]]
            tm.assert_frame_equal(result, expected)
    
            # 使用负索引进行获取
            result = df.iloc[-1, -1]
            exp = df.loc["j", "D"]
            assert result == exp
    
            # 测试超出范围的索引，预期会触发 IndexError 异常
            msg = "index 5 is out of bounds for axis 0 with size 4|index out of bounds"
            with pytest.raises(IndexError, match=msg):
                df.iloc[10, 5]
    
            # 尝试使用标签进行索引，预期会触发 ValueError 异常
            msg = (
                r"Location based indexing can only have \[integer, integer "
                r"slice \(START point is INCLUDED, END point is EXCLUDED\), "
                r"listlike of integers, boolean array\] types"
            )
            with pytest.raises(ValueError, match=msg):
                df.iloc["j", "D"]
    
        # 测试 iloc 的获取和文档问题
        def test_iloc_getitem_doc_issue(self):
            # 多轴切片问题，单个块表现
            # 在 GH 6059 中曝光
    
            # 创建一个包含随机标准正态分布数据的 DataFrame
            arr = np.random.default_rng(2).standard_normal((6, 4))
            index = date_range("20130101", periods=6)
            columns = list("ABCD")
            df = DataFrame(arr, index=index, columns=columns)
    
            # 定义 ref_locs
    
            # 描述数据框 df 的统计信息
            df.describe()
    
            # 获取 iloc 的切片结果，比较是否与预期的 DataFrame 结果一致
            result = df.iloc[3:5, 0:2]
    
            expected = DataFrame(arr[3:5, 0:2], index=index[3:5], columns=columns[0:2])
            tm.assert_frame_equal(result, expected)
    
            # 处理重复列名的情况
            df.columns = list("aaaa")
            result = df.iloc[3:5, 0:2]
    
            expected = DataFrame(arr[3:5, 0:2], index=index[3:5], columns=list("aa"))
            tm.assert_frame_equal(result, expected)
    
            # 相关处理
            arr = np.random.default_rng(2).standard_normal((6, 4))
            index = list(range(0, 12, 2))
            columns = list(range(0, 8, 2))
            df = DataFrame(arr, index=index, columns=columns)
    
            # 获取 df 数据框的内部块信息
            df._mgr.blocks[0].mgr_locs
    
            # 获取 iloc 的切片结果，比较是否与预期的 DataFrame 结果一致
            result = df.iloc[1:5, 2:4]
            expected = DataFrame(arr[1:5, 2:4], index=index[1:5], columns=columns[2:4])
            tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试 DataFrame 的 iloc 设置单个元素的操作
    def test_iloc_setitem_series(self):
        # 创建一个 10x4 的 DataFrame，填充随机标准正态分布的数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            index=list("abcdefghij"),
            columns=list("ABCD"),
        )

        # 将第 (1, 1) 位置的元素设为 1
        df.iloc[1, 1] = 1
        # 获取第 (1, 1) 位置的元素值
        result = df.iloc[1, 1]
        # 断言结果是否为 1
        assert result == 1

        # 将所有行的第 2 到 3 列（从零开始索引）的元素设为 0
        df.iloc[:, 2:3] = 0
        # 获取设定后的预期结果（所有行的第 2 列）
        expected = df.iloc[:, 2:3]
        # 获取实际结果
        result = df.iloc[:, 2:3]
        # 断言 DataFrame 是否符合预期
        tm.assert_frame_equal(result, expected)

        # 创建一个 Series，包含 10 个随机标准正态分布的元素，索引为偶数
        s = Series(np.random.default_rng(2).standard_normal(10), index=range(0, 20, 2))

        # 将索引为 1 的元素设为 1
        s.iloc[1] = 1
        # 获取索引为 1 的元素值
        result = s.iloc[1]
        # 断言结果是否为 1
        assert result == 1

        # 将前 4 个元素设为 0
        s.iloc[:4] = 0
        # 获取设定后的预期结果（前 4 个元素）
        expected = s.iloc[:4]
        # 获取实际结果
        result = s.iloc[:4]
        # 断言 Series 是否符合预期
        tm.assert_series_equal(result, expected)

        # 创建一个包含六个元素的 Series，初始值均为 -1
        s = Series([-1] * 6)
        # 将索引为 0, 2, 4 的元素分别设为 0, 2, 4
        s.iloc[0::2] = [0, 2, 4]
        # 将索引为 1, 3, 5 的元素分别设为 1, 3, 5
        s.iloc[1::2] = [1, 3, 5]
        # 获取结果 Series
        result = s
        # 创建预期的 Series，包含 0 到 5 的连续整数
        expected = Series([0, 1, 2, 3, 4, 5])
        # 断言 Series 是否符合预期
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，用于测试 DataFrame 的 iloc 设置嵌套列表的操作
    def test_iloc_setitem_list_of_lists(self):
        # GH 7551
        # 测试在混合和单一数据类型框架中，列表列表的设置是否正确

        # 创建一个 DataFrame，包含两列 A 和 B，A 列为整数 0 到 4，B 列为整数 5 到 9
        df = DataFrame(
            {"A": np.arange(5, dtype="int64"), "B": np.arange(5, 10, dtype="int64")}
        )
        # 将第 2 到 3 行的元素设为 [[10, 11], [12, 13]]
        df.iloc[2:4] = [[10, 11], [12, 13]]
        # 创建预期的 DataFrame
        expected = DataFrame({"A": [0, 1, 10, 12, 4], "B": [5, 6, 11, 13, 9]})
        # 断言 DataFrame 是否符合预期
        tm.assert_frame_equal(df, expected)

        # 创建一个 DataFrame，包含两列 A 和 B，A 列为字符 'a' 到 'e'，B 列为整数 5 到 9
        df = DataFrame(
            {"A": ["a", "b", "c", "d", "e"], "B": np.arange(5, 10, dtype="int64")}
        )
        # 将第 2 到 3 行的元素设为 [['x', 11], ['y', 13]]
        df.iloc[2:4] = [["x", 11], ["y", 13]]
        # 创建预期的 DataFrame
        expected = DataFrame({"A": ["a", "b", "x", "y", "e"], "B": [5, 6, 11, 13, 9]})
        # 断言 DataFrame 是否符合预期
        tm.assert_frame_equal(df, expected)

    # 使用 pytest 的参数化标记，测试针对单个标量索引的 iloc 设置操作
    @pytest.mark.parametrize("indexer", [[0], slice(None, 1, None), np.array([0])])
    @pytest.mark.parametrize("value", [["Z"], np.array(["Z"])])
    def test_iloc_setitem_with_scalar_index(self, indexer, value):
        # GH #19474
        # 测试类似 "df.iloc[0, [0]] = ['Z']" 的赋值应逐元素进行评估，而不使用 "setter('A', ['Z'])"。

        # 将 DataFrame 的列 A 类型设置为对象类型，以避免设置 'Z' 时的类型提升
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"]).astype({"A": object})
        # 将第 (0, 0) 位置的元素设为 'Z'
        df.iloc[0, indexer] = value
        # 获取设置后的第 (0, 0) 位置的元素值
        result = df.iloc[0, 0]
        # 断言结果是否为标量并且值为 "Z"
        assert is_scalar(result) and result == "Z"

    # 忽略 UserWarning 类型的警告
    @pytest.mark.filterwarnings("ignore::UserWarning")
    # 定义一个测试方法，用于测试在使用 Series 的布尔掩码时的 iloc 操作
    def test_iloc_mask(self):
        # 创建一个 DataFrame 包含整数列表，以及作为索引的字母列表
        df = DataFrame(list(range(5)), index=list("ABCDE"), columns=["a"])
        # 创建一个布尔掩码，表示列 'a' 中的偶数位置
        mask = df.a % 2 == 0
        # 预期引发 ValueError，因为 iLocation 基于布尔索引不能使用可索引对象作为掩码
        msg = "iLocation based boolean indexing cannot use an indexable as a mask"
        with pytest.raises(ValueError, match=msg):
            df.iloc[mask]
        
        # 调整掩码的索引，以解决前述问题
        mask.index = range(len(mask))
        # 预期引发 NotImplementedError，因为 iLocation 基于整数类型的布尔索引不可用
        msg = "iLocation based boolean indexing on an integer type is not available"
        with pytest.raises(NotImplementedError, match=msg):
            df.iloc[mask]

        # 使用 ndarray 作为掩码，预期结果与原始 DataFrame 相等
        result = df.iloc[np.array([True] * len(mask), dtype=bool)]
        tm.assert_frame_equal(result, df)

        # 创建一个 DataFrame 包含 locs 和 nums 列，以及以 nums 为索引的字符串列表
        locs = np.arange(4)
        nums = 2**locs
        reps = [bin(num) for num in nums]
        df = DataFrame({"locs": locs, "nums": nums}, index=reps)

        # 预期结果字典，键为元组 (索引类型, 方法)，值为对应的字符串
        expected = {
            (None, ""): "0b1100",
            (None, ".loc"): "0b1100",
            (None, ".iloc"): "0b1100",
            ("index", ""): "0b11",
            ("index", ".loc"): "0b11",
            ("index", ".iloc"): (
                "iLocation based boolean indexing cannot use an indexable as a mask"
            ),
            ("locs", ""): "Unalignable boolean Series provided as indexer "
            "(index of the boolean Series and of the indexed "
            "object do not match).",
            ("locs", ".loc"): "Unalignable boolean Series provided as indexer "
            "(index of the boolean Series and of the "
            "indexed object do not match).",
            ("locs", ".iloc"): (
                "iLocation based boolean indexing on an "
                "integer type is not available"
            ),
        }

        # 遍历预期结果字典，测试 reindex 操作使用布尔掩码时的 UserWarnings
        for idx in [None, "index", "locs"]:
            # 创建布尔掩码，表示 nums 列中大于 2 的值
            mask = (df.nums > 2).values
            if idx:
                # 根据 idx 获取相应的索引对象，并反转其顺序
                mask_index = getattr(df, idx)[::-1]
                # 将 Series 对象重新赋值为经过索引反转的布尔掩码
                mask = Series(mask, index=list(mask_index))
            for method in ["", ".loc", ".iloc"]:
                try:
                    # 根据 method 选择相应的访问器对象
                    if method:
                        accessor = getattr(df, method[1:])
                    else:
                        accessor = df
                    # 计算 accessor 对象中符合布尔掩码条件的 nums 列的和，并转换为二进制字符串
                    answer = str(bin(accessor[mask]["nums"].sum()))
                except (ValueError, IndexingError, NotImplementedError) as err:
                    # 如果出现异常，则将异常信息转换为字符串
                    answer = str(err)

                # 获取预期结果字典中对应键的值
                key = (
                    idx,
                    method,
                )
                r = expected.get(key)
                # 检查实际结果与预期结果是否一致，如不一致则引发 AssertionError
                if r != answer:
                    raise AssertionError(
                        f"[{key}] does not match [{answer}], received [{r}]"
                    )
    def test_iloc_non_unique_indexing(self):
        # GH 4017, non-unique indexing (on the axis)
        # 创建一个包含"A"和"B"两列，每列有3000个0.1和1的DataFrame
        df = DataFrame({"A": [0.1] * 3000, "B": [1] * 3000)
        # 创建一个索引数组，范围是0到2970，步长为99
        idx = np.arange(30) * 99
        # 从df中使用iloc方法按照idx数组进行索引，得到期望的DataFrame
        expected = df.iloc[idx]

        # 将df按照2倍和3倍进行concat操作，创建df3
        df3 = concat([df, 2 * df, 3 * df])
        # 从df3中使用iloc方法按照idx数组进行索引，得到结果DataFrame
        result = df3.iloc[idx]

        # 使用测试工具tm.assert_frame_equal比较result和expected，确保它们相等
        tm.assert_frame_equal(result, expected)

        # 创建一个包含"A"和"B"两列，每列有1000个0.1和1的DataFrame
        df2 = DataFrame({"A": [0.1] * 1000, "B": [1] * 1000})
        # 将df2按照2倍和3倍进行concat操作，覆盖原有的df2
        df2 = concat([df2, 2 * df2, 3 * df2])

        # 使用pytest工具验证在df2上使用loc方法按照idx数组会引发KeyError异常，并且异常信息包含"not in index"
        with pytest.raises(KeyError, match="not in index"):
            df2.loc[idx]

    def test_iloc_empty_list_indexer_is_ok(self):
        # 创建一个5行2列，元素全为1的DataFrame，行索引为'i-0'到'i-4'，列索引为'i-0'和'i-1'
        df = DataFrame(
            np.ones((5, 2)),
            index=Index([f"i-{i}" for i in range(5)], name="a"),
            columns=Index([f"i-{i}" for i in range(2)], name="a"),
        )
        # 使用tm.assert_frame_equal比较垂直方向空索引和列范围为0的切片，确保它们相等
        tm.assert_frame_equal(
            df.iloc[:, []],
            df.iloc[:, :0],
            check_index_type=True,
            check_column_type=True,
        )
        # 使用tm.assert_frame_equal比较水平方向空索引和行范围为0的切片，确保它们相等
        tm.assert_frame_equal(
            df.iloc[[], :],
            df.iloc[:0, :],
            check_index_type=True,
            check_column_type=True,
        )
        # 使用tm.assert_frame_equal比较水平方向空索引和行范围为0的切片，确保它们相等
        tm.assert_frame_equal(
            df.iloc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True
        )

    def test_identity_slice_returns_new_object(self):
        # GH13873
        # 创建一个包含列'a'，元素为[1, 2, 3]的DataFrame
        original_df = DataFrame({"a": [1, 2, 3]})
        # 使用.iloc[:]对original_df进行切片操作，得到sliced_df
        sliced_df = original_df.iloc[:]
        # 断言sliced_df与original_df不是同一个对象
        assert sliced_df is not original_df

        # 断言列'a'的数据是浅拷贝关系
        assert np.shares_memory(original_df["a"], sliced_df["a"])

        # 使用.loc[:, 'a']进行设置操作会就地修改sliced_df和original_df，这取决于Copy-on-Write（CoW）
        original_df.loc[:, "a"] = [4, 4, 4]
        # 断言sliced_df的列'a'仍然是[1, 2, 3]
        assert (sliced_df["a"] == [1, 2, 3]).all()

        # 创建一个包含[1, 2, 3, 4, 5, 6]的Series
        original_series = Series([1, 2, 3, 4, 5, 6])
        # 使用.iloc[:]对original_series进行切片操作，得到sliced_series
        sliced_series = original_series.iloc[:]
        # 断言sliced_series与original_series不是同一个对象
        assert sliced_series is not original_series

        # 断言sliced_series也是浅拷贝关系
        original_series[:3] = [7, 8, 9]
        # 浅拷贝没有更新（CoW）
        assert all(sliced_series[:3] == [1, 2, 3])

    def test_indexing_zerodim_np_array(self):
        # GH24919
        # 创建一个包含[[1, 2], [3, 4]]的DataFrame
        df = DataFrame([[1, 2], [3, 4]])
        # 使用.iloc[np.array(0)]对df进行索引，得到结果Series
        result = df.iloc[np.array(0)]
        # 创建一个名称为0的Series，元素为[1, 2]
        s = Series([1, 2], name=0)
        # 使用测试工具tm.assert_series_equal比较result和s，确保它们相等
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self):
        # GH24919
        # 创建一个包含[1, 2]的Series
        s = Series([1, 2])
        # 使用.iloc[np.array(0)]对s进行索引，得到结果1
        result = s.iloc[np.array(0)]
        # 断言result等于1
        assert result == 1
    def test_iloc_setitem_categorical_updates_inplace(self):
        # 创建一个混合数据类型的分类变量
        cat = Categorical(["A", "B", "C"])
        # 创建一个包含分类变量和整数的数据框
        df = DataFrame({1: cat, 2: [1, 2, 3]}, copy=False)

        # 断言数据框的第一列与分类变量共享内存
        assert tm.shares_memory(df[1], cat)

        # 使用 iloc 操作，将数据框的第一列赋值为分类变量的逆序，此操作在原地修改值
        df.iloc[:, 0] = cat[::-1]

        # 再次断言数据框的第一列与分类变量共享内存
        assert tm.shares_memory(df[1], cat)
        # 创建一个预期的分类变量，逆序排序，指定类别顺序
        expected = Categorical(["C", "B", "A"], categories=["A", "B", "C"])
        # 断言分类变量与预期值相等
        tm.assert_categorical_equal(cat, expected)

    def test_iloc_with_boolean_operation(self):
        # GH 20627
        # 创建一个包含 NaN 值的数据框
        result = DataFrame([[0, 1], [2, 3], [4, 5], [6, np.nan]])
        # 使用 iloc 和布尔运算符，选择索引小于等于 2 的行，并将其乘以 2
        result.iloc[result.index <= 2] *= 2
        # 创建一个预期的数据框
        expected = DataFrame([[0, 2], [4, 6], [8, 10], [6, np.nan]])
        # 断言结果数据框与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 再次使用 iloc 和布尔运算符，选择索引大于 2 的行，并将其乘以 2
        result.iloc[result.index > 2] *= 2
        # 更新预期的数据框
        expected = DataFrame([[0, 2], [4, 6], [8, 10], [12, np.nan]])
        # 断言结果数据框与更新后的预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 使用 iloc 和布尔数组选择行，将其乘以 2
        result.iloc[[True, True, False, False]] *= 2
        # 更新预期的数据框
        expected = DataFrame([[0, 4], [8, 12], [8, 10], [12, np.nan]])
        # 断言结果数据框与更新后的预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 使用 iloc 和布尔数组选择行，将其除以 2
        result.iloc[[False, False, True, True]] /= 2
        # 更新预期的数据框
        expected = DataFrame([[0, 4.0], [8, 12.0], [4, 5.0], [6, np.nan]])
        # 断言结果数据框与更新后的预期数据框相等
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_singlerow_slice_categoricaldtype_gives_series(self):
        # GH#29521
        # 创建一个包含分类变量的数据框
        df = DataFrame({"x": Categorical("a b c d e".split())})
        # 使用 iloc 提取第一行数据，返回一个系列对象
        result = df.iloc[0]
        # 创建一个原始的分类变量
        raw_cat = Categorical(["a"], categories=["a", "b", "c", "d", "e"])
        # 创建一个预期的系列对象，指定了索引名称和数据类型
        expected = Series(raw_cat, index=["x"], name=0, dtype="category")

        # 断言提取的系列对象与预期的系列对象相等
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_categorical_values(self):
        # GH#14580
        # 在包含分类数据的系列上使用 iloc() 方法进行测试

        # 创建一个包含整数的系列，并转换为分类数据类型
        ser = Series([1, 2, 3]).astype("category")

        # 使用 iloc() 方法进行切片操作
        result = ser.iloc[0:2]
        # 创建一个预期的系列对象，指定了分类数据类型和类别
        expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
        # 断言提取的系列对象与预期的系列对象相等
        tm.assert_series_equal(result, expected)

        # 使用 iloc() 方法提取指定索引列表的数据
        result = ser.iloc[[0, 1]]
        # 再次创建一个预期的系列对象，指定了分类数据类型和类别
        expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
        # 断言提取的系列对象与预期的系列对象相等
        tm.assert_series_equal(result, expected)

        # 使用 iloc() 方法和布尔数组提取数据
        result = ser.iloc[[True, False, False]]
        # 创建一个预期的系列对象，指定了分类数据类型和类别
        expected = Series([1]).astype(CategoricalDtype([1, 2, 3]))
        # 断言提取的系列对象与预期的系列对象相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("value", [None, NaT, np.nan])
    def test_iloc_setitem_td64_values_cast_na(self, value):
        # GH#18586
        # 创建一个时间增量数据类型的系列
        series = Series([0, 1, 2], dtype="timedelta64[ns]")
        # 使用 iloc() 方法将系列的第一个值设置为指定的值（如 None、NaT 或 np.nan）
        series.iloc[0] = value
        # 创建一个预期的系列对象，将第一个值转换为 NaT
        expected = Series([NaT, 1, 2], dtype="timedelta64[ns]")
        # 断言结果的系列对象与预期的系列对象相等
        tm.assert_series_equal(series, expected)

    @pytest.mark.parametrize("not_na", [Interval(0, 1), "a", 1.0])
    # 测试在混合了 NaN 和区间值的情况下设置元素
    def test_setitem_mix_of_nan_and_interval(self, not_na, nulls_fixture):
        # GH#27937
        # 创建分类数据类型，指定分类值列表
        dtype = CategoricalDtype(categories=[not_na])
        # 创建包含空值的 Series 对象，使用指定的数据类型
        ser = Series(
            [nulls_fixture, nulls_fixture, nulls_fixture, nulls_fixture], dtype=dtype
        )
        # 使用 iloc 方法设置前三个元素的值
        ser.iloc[:3] = [nulls_fixture, not_na, nulls_fixture]
        # 期望的 Series 对象
        exp = Series([nulls_fixture, not_na, nulls_fixture, nulls_fixture], dtype=dtype)
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(ser, exp)

    # 测试在空 DataFrame 上使用三维 ndarray 设置元素时是否会引发异常
    def test_iloc_setitem_empty_frame_raises_with_3d_ndarray(self):
        # 创建空 Index 对象
        idx = Index([])
        # 创建指定大小的随机数 DataFrame 对象，使用标准正态分布
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), len(idx))),
            index=idx,
            columns=idx,
        )
        # 创建一个三维 ndarray
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        # 设置异常消息
        msg = f"Cannot set values with ndim > {obj.ndim}"
        # 使用 pytest 检查是否会引发 ValueError 异常，且异常消息匹配预期消息
        with pytest.raises(ValueError, match=msg):
            obj.iloc[nd3] = 0

    # 测试在使用 iloc 获取元素时读取只读数据的情况
    def test_iloc_getitem_read_only_values(self, indexer_li):
        # GH#10043 这是一个基本的 iloc 测试，同时也测试 loc
        # 创建一个可读写的二维数组
        rw_array = np.eye(10)
        rw_df = DataFrame(rw_array)

        # 创建一个只读的二维数组
        ro_array = np.eye(10)
        ro_array.setflags(write=False)
        ro_df = DataFrame(ro_array)

        # 使用自定义的 indexer_li 函数分别获取 rw_df 和 ro_df 中指定索引的数据，并断言它们相等
        tm.assert_frame_equal(
            indexer_li(rw_df)[[1, 2, 3]], indexer_li(ro_df)[[1, 2, 3]]
        )
        # 使用自定义的 indexer_li 函数分别获取 rw_df 和 ro_df 中指定索引的数据，并断言它们相等
        tm.assert_frame_equal(indexer_li(rw_df)[[1]], indexer_li(ro_df)[[1]])
        # 使用自定义的 indexer_li 函数分别获取 rw_df 和 ro_df 中指定列索引的数据，并断言它们相等
        tm.assert_series_equal(indexer_li(rw_df)[1], indexer_li(ro_df)[1])
        # 使用自定义的 indexer_li 函数分别获取 rw_df 和 ro_df 中指定行范围的数据，并断言它们相等
        tm.assert_frame_equal(indexer_li(rw_df)[1:3], indexer_li(ro_df)[1:3])

    # 测试在使用只读数组的 iloc 操作时是否会引发 TypeError 异常
    def test_iloc_getitem_readonly_key(self):
        # GH#17192 使用只读数组的 iloc 操作引发 TypeError 异常
        # 创建包含单列数据的 DataFrame 对象
        df = DataFrame({"data": np.ones(100, dtype="float64")})
        # 创建一个只读的索引数组
        indices = np.array([1, 3, 6])
        indices.flags.writeable = False

        # 使用只读索引数组进行 iloc 操作，获取结果并与预期结果进行比较
        result = df.iloc[indices]
        expected = df.loc[[1, 3, 6]]
        tm.assert_frame_equal(result, expected)

        # 使用只读索引数组进行 iloc 操作，获取单列 Series 结果并与预期结果进行比较
        result = df["data"].iloc[indices]
        expected = df["data"].loc[[1, 3, 6]]
        tm.assert_series_equal(result, expected)

    # 测试在 DataFrame 中使用 iloc 将 Series 赋值给单个单元格时的情况
    def test_iloc_assign_series_to_df_cell(self):
        # GH 37593
        # 创建只包含一列的空 DataFrame 对象
        df = DataFrame(columns=["a"], index=[0])
        # 将 Series 对象赋值给 DataFrame 中的一个单元格
        df.iloc[0, 0] = Series([1, 2, 3])
        # 期望的 DataFrame 对象
        expected = DataFrame({"a": [Series([1, 2, 3])]}, columns=["a"], index=[0])
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(df, expected)

    # 使用 pytest 的参数化标记，测试在使用 bool 索引器时是否正确设置元素
    @pytest.mark.parametrize("klass", [list, np.array])
    def test_iloc_setitem_bool_indexer(self, klass):
        # GH#36741
        # 创建包含 flag 和 value 两列数据的 DataFrame 对象
        df = DataFrame({"flag": ["x", "y", "z"], "value": [1, 3, 4]})
        # 创建一个布尔类型的索引器
        indexer = klass([True, False, False])
        # 使用 iloc 方法根据索引器设置 value 列中的元素值
        df.iloc[indexer, 1] = df.iloc[indexer, 1] * 2
        # 期望的 DataFrame 对象
        expected = DataFrame({"flag": ["x", "y", "z"], "value": [2, 3, 4]})
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(df, expected)

    # 使用 pytest 的参数化标记，测试在使用索引器时是否正确设置元素
    @pytest.mark.parametrize("indexer", [[1], slice(1, 2)])
    def test_iloc_setitem_pure_position_based(self, indexer):
        # GH#22046
        # 创建包含两列的 DataFrame df1
        df1 = DataFrame({"a2": [11, 12, 13], "b2": [14, 15, 16]})
        # 创建包含三列的 DataFrame df2
        df2 = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        # 将 df1 的第一列数据设置到 df2 的所有行的 indexer 列上
        df2.iloc[:, indexer] = df1.iloc[:, [0]]
        # 创建预期结果的 DataFrame
        expected = DataFrame({"a": [1, 2, 3], "b": [11, 12, 13], "c": [7, 8, 9]})
        # 断言 df2 和预期结果相等
        tm.assert_frame_equal(df2, expected)

    def test_iloc_setitem_dictionary_value(self):
        # GH#37728
        # 创建包含两列的 DataFrame df
        df = DataFrame({"x": [1, 2], "y": [2, 2]})
        # 创建一个字典 rhs
        rhs = {"x": 9, "y": 99}
        # 将 rhs 的值设置到 df 的第二行
        df.iloc[1] = rhs
        # 创建预期结果的 DataFrame
        expected = DataFrame({"x": [1, 9], "y": [2, 99]})
        # 断言 df 和预期结果相等
        tm.assert_frame_equal(df, expected)

        # GH#38335 同样的操作，混合类型
        # 创建包含两列的 DataFrame df
        df = DataFrame({"x": [1, 2], "y": [2.0, 2.0]})
        # 将 rhs 的值设置到 df 的第二行
        df.iloc[1] = rhs
        # 创建预期结果的 DataFrame
        expected = DataFrame({"x": [1, 9], "y": [2.0, 99.0]})
        # 断言 df 和预期结果相等
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_float_duplicates(self):
        # 创建一个 3x3 的随机数 DataFrame df
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=[0.1, 0.2, 0.2],
            columns=list("abc"),
        )
        # 选择 df 中索引为 0.2 的所有行和列
        expect = df.iloc[1:]
        # 断言 df 中索引为 0.2 的行与预期结果相等
        tm.assert_frame_equal(df.loc[0.2], expect)

        # 选择 df 中索引为 0.2 的所有行中 "a" 列
        expect = df.iloc[1:, 0]
        # 断言 df 中索引为 0.2 的行的 "a" 列与预期结果相等
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

        # 修改 df 的索引
        df.index = [1, 0.2, 0.2]
        # 选择 df 中索引为 0.2 的所有行和列
        expect = df.iloc[1:]
        # 断言 df 中索引为 0.2 的行与预期结果相等
        tm.assert_frame_equal(df.loc[0.2], expect)

        # 选择 df 中索引为 0.2 的所有行中 "a" 列
        expect = df.iloc[1:, 0]
        # 断言 df 中索引为 0.2 的行的 "a" 列与预期结果相等
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

        # 创建一个 4x3 的随机数 DataFrame df
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)),
            index=[1, 0.2, 0.2, 1],
            columns=list("abc"),
        )
        # 选择 df 中索引为 0.2 的行，排除第一行和最后一行
        expect = df.iloc[1:-1]
        # 断言 df 中索引为 0.2 的行与预期结果相等
        tm.assert_frame_equal(df.loc[0.2], expect)

        # 选择 df 中索引为 0.2 的行中 "a" 列，排除第一行和最后一行
        expect = df.iloc[1:-1, 0]
        # 断言 df 中索引为 0.2 的行的 "a" 列与预期结果相等
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

        # 修改 df 的索引
        df.index = [0.1, 0.2, 2, 0.2]
        # 选择 df 中索引为 0.2 和最后一行的所有列
        expect = df.iloc[[1, -1]]
        # 断言 df 中索引为 0.2 的行与预期结果相等
        tm.assert_frame_equal(df.loc[0.2], expect)

        # 选择 df 中索引为 0.2 和最后一行的 "a" 列
        expect = df.iloc[[1, -1], 0]
        # 断言 df 中索引为 0.2 的行的 "a" 列与预期结果相等
        tm.assert_series_equal(df.loc[0.2, "a"], expect)
    def test_iloc_setitem_custom_object(self):
        # 定义一个自定义对象 TO
        class TO:
            def __init__(self, value) -> None:
                self.value = value

            def __str__(self) -> str:
                return f"[{self.value}]"

            __repr__ = __str__

            def __eq__(self, other) -> bool:
                return self.value == other.value

            def view(self):
                return self

        # 创建一个空的 DataFrame，指定行索引和列名
        df = DataFrame(index=[0, 1], columns=[0])
        # 在指定位置 [1, 0] 处设置自定义对象 TO(1)
        df.iloc[1, 0] = TO(1)
        # 再次在同一位置 [1, 0] 处设置自定义对象 TO(2)
        df.iloc[1, 0] = TO(2)

        # 创建期望的 DataFrame 结果，与设置后的 df 进行比较
        result = DataFrame(index=[0, 1], columns=[0])
        result.iloc[1, 0] = TO(2)

        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, df)

        # 即使将值设置回去，仍然保持对象的数据类型
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = TO(1)
        # 将指定位置 [1, 0] 处设置为 np.nan
        df.iloc[1, 0] = np.nan
        result = DataFrame(index=[0, 1], columns=[0])

        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, df)

    def test_iloc_getitem_with_duplicates(self):
        # 创建一个包含随机数据的 DataFrame，指定列和索引含有重复值
        df = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            columns=list("ABC"),
            index=list("aab"),
        )

        # 获取索引为 0 的行，预期返回一个 Series 对象
        result = df.iloc[0]
        assert isinstance(result, Series)
        # 使用测试工具比较 Series 的值是否接近 DataFrame 第一行的值
        tm.assert_almost_equal(result.values, df.values[0])

        # 对转置后的 DataFrame 进行行切片，预期返回一个 Series 对象
        result = df.T.iloc[:, 0]
        assert isinstance(result, Series)
        # 使用测试工具比较 Series 的值是否接近 DataFrame 第一列的值
        tm.assert_almost_equal(result.values, df.values[0])

    def test_iloc_getitem_with_duplicates2(self):
        # GH#2259 测试案例
        # 创建一个具有重复列名的 DataFrame
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1, 1, 2])
        # 使用 iloc 获取所有行和第一列的子 DataFrame
        result = df.iloc[:, [0]]
        # 使用 take 方法获取期望的 DataFrame
        expected = df.take([0], axis=1)
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_iloc_interval(self):
        # GH#17130 测试案例
        # 创建一个带有 Interval 键的 DataFrame
        df = DataFrame({Interval(1, 2): [1, 2]})

        # 使用 iloc 获取索引为 0 的行，预期返回一个 Series 对象
        result = df.iloc[0]
        expected = Series({Interval(1, 2): 1}, name=0)
        # 使用测试工具比较两个 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 使用 iloc 获取所有行和第一列的 Series 对象
        result = df.iloc[:, 0]
        expected = Series([1, 2], name=Interval(1, 2))
        # 使用测试工具比较两个 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 复制 DataFrame 并在第一列上增加 1
        result = df.copy()
        result.iloc[:, 0] += 1
        expected = DataFrame({Interval(1, 2): [2, 3]})
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("indexing_func", [list, np.array])
    @pytest.mark.parametrize("rhs_func", [list, np.array])
    def test_loc_setitem_boolean_list(self, rhs_func, indexing_func):
        # GH#20438 测试案例，专门测试列表键而不是类似数组的键

        # 创建一个 Series 对象
        ser = Series([0, 1, 2])
        # 使用 iloc 根据布尔列表索引设置值
        ser.iloc[indexing_func([True, False, True])] = rhs_func([5, 10])
        expected = Series([5, 1, 10])
        # 使用测试工具比较两个 Series 是否相等
        tm.assert_series_equal(ser, expected)

        # 创建一个 DataFrame 对象
        df = DataFrame({"a": [0, 1, 2]})
        # 使用 iloc 根据布尔列表索引设置值
        df.iloc[indexing_func([True, False, True])] = rhs_func([[5], [10]])
        expected = DataFrame({"a": [5, 1, 10]})
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)
    # 定义一个测试函数，测试 DataFrame 的 iloc 操作，负步长切片索引
    def test_iloc_getitem_slice_negative_step_ea_block(self):
        # 创建一个包含整数列 'A' 的 DataFrame，数据类型为 Int64
        df = DataFrame({"A": [1, 2, 3]}, dtype="Int64")

        # 执行 iloc 操作，选择所有行和倒序的所有列，期望结果应该与原始 DataFrame 相同
        res = df.iloc[:, ::-1]
        tm.assert_frame_equal(res, df)

        # 向 DataFrame 添加新列 'B'，并执行相同的 iloc 操作
        res = df.iloc[:, ::-1]
        # 创建期望的 DataFrame，包含列 'B' 和列 'A'，与原始 DataFrame 的数据相同
        expected = DataFrame({"B": df["B"], "A": df["A"]})
        tm.assert_frame_equal(res, expected)

    # 定义一个测试函数，测试 DataFrame 的 iloc 设置操作，向二维 ndarray 赋值
    def test_iloc_setitem_2d_ndarray_into_ea_block(self):
        # 创建一个包含类别列 'status' 的 DataFrame
        df = DataFrame({"status": ["a", "b", "c"]}, dtype="category")
        # 使用 iloc 将二维 ndarray 赋值给指定位置
        df.iloc[np.array([0, 1]), np.array([0])] = np.array([["a"], ["a"]])

        # 创建期望的 DataFrame，修改了指定位置后 'status' 列的值
        expected = DataFrame({"status": ["a", "a", "c"]}, dtype=df["status"].dtype)
        tm.assert_frame_equal(df, expected)

    # 定义一个测试函数，测试 DataFrame 的 iloc 获取单个整数索引的操作
    def test_iloc_getitem_int_single_ea_block_view(self):
        # 创建一个 DataFrame，从 interval_range 的值创建
        arr = interval_range(1, 10.0)._values
        df = DataFrame(arr)

        # 使用 iloc 获取单个整数索引，创建一个 Series，应该是 DataFrame 数据的视图
        ser = df.iloc[2]

        # 如果 ser 是视图，则修改 arr[2] 应该也会修改 ser[0] 的值
        assert arr[2] != arr[-1]  # 否则后续测试无意义
        arr[2] = arr[-1]
        assert ser[0] == arr[-1]

    # 定义一个测试函数，测试 DataFrame 的 iloc 设置多列转换为 datetime
    def test_iloc_setitem_multicolumn_to_datetime(self):
        # 创建一个 DataFrame，包含列 'A' 和 'B'，其中 'A' 包含日期字符串
        df = DataFrame({"A": ["2022-01-01", "2022-01-02"], "B": ["2021", "2022"]})

        # 使用 iloc 将一个新的 DataFrame 赋值给列 'A'，包含 datetime 对象
        df.iloc[:, [0]] = DataFrame({"A": to_datetime(["2021", "2022"])})

        # 创建期望的 DataFrame，其中 'A' 列转换为 datetime 对象，'B' 列保持不变
        expected = DataFrame(
            {
                "A": [
                    Timestamp("2021-01-01 00:00:00"),
                    Timestamp("2022-01-01 00:00:00"),
                ],
                "B": ["2021", "2022"],
            }
        )
        tm.assert_frame_equal(df, expected, check_dtype=False)
class TestILocErrors:
    # NB: this test should work for _any_ Series we can pass as
    #  series_with_simple_index
    # 测试类用于测试针对任何可以作为 series_with_simple_index 传递的 Series 的情况

    def test_iloc_float_raises(self, series_with_simple_index, frame_or_series):
        # GH#4892
        # float_indexers should raise exceptions
        # on appropriate Index types & accessors
        # this duplicates the code below
        # but is specifically testing for the error
        # message
        # 测试使用浮点数索引引发异常，对应适当的索引类型和访问器，这段代码重复了下面的代码，但专门测试错误消息

        obj = series_with_simple_index
        if frame_or_series is DataFrame:
            obj = obj.to_frame()

        msg = "Cannot index by location index with a non-integer key"
        with pytest.raises(TypeError, match=msg):
            obj.iloc[3.0]

        with pytest.raises(IndexError, match=_slice_iloc_msg):
            obj.iloc[3.0] = 0

    def test_iloc_getitem_setitem_fancy_exceptions(self, float_frame):
        # 测试通过iloc进行getitem和setitem操作时引发的复杂异常

        with pytest.raises(IndexingError, match="Too many indexers"):
            float_frame.iloc[:, :, :]

        with pytest.raises(IndexError, match="too many indices for array"):
            # GH#32257 we let numpy do validation, get their exception
            # GH#32257 让numpy进行验证，获取它们的异常
            float_frame.iloc[:, :, :] = 1

    def test_iloc_frame_indexer(self):
        # GH#39004
        # 测试DataFrame的索引器在.iloc中的应用，预期会引发TypeError异常

        df = DataFrame({"a": [1, 2, 3]})
        indexer = DataFrame({"a": [True, False, True]})
        msg = "DataFrame indexer for .iloc is not supported. Consider using .loc"
        with pytest.raises(TypeError, match=msg):
            df.iloc[indexer] = 1

        msg = (
            "DataFrame indexer is not allowed for .iloc\n"
            "Consider using .loc for automatic alignment."
        )
        with pytest.raises(IndexError, match=msg):
            df.iloc[indexer]


class TestILocSetItemDuplicateColumns:
    def test_iloc_setitem_scalar_duplicate_columns(self):
        # GH#15686, duplicate columns and mixed dtype
        # 测试.iloc设置标量值到具有重复列和混合dtype的DataFrame中

        df1 = DataFrame([{"A": None, "B": 1}, {"A": 2, "B": 2}])
        df2 = DataFrame([{"A": 3, "B": 3}, {"A": 4, "B": 4}])
        df = concat([df1, df2], axis=1)
        df.iloc[0, 0] = -1

        assert df.iloc[0, 0] == -1
        assert df.iloc[0, 2] == 3
        assert df.dtypes.iloc[2] == np.int64

    def test_iloc_setitem_list_duplicate_columns(self):
        # GH#22036 setting with same-sized list
        # 测试使用相同大小的列表进行.iloc设置，处理具有重复列的情况

        df = DataFrame([[0, "str", "str2"]], columns=["a", "b", "b"])
        df.iloc[:, 2] = ["str3"]

        expected = DataFrame([[0, "str", "str3"]], columns=["a", "b", "b"])
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_series_duplicate_columns(self):
        # 测试使用.iloc设置Series到具有重复列的DataFrame中

        df = DataFrame(
            np.arange(8, dtype=np.int64).reshape(2, 4), columns=["A", "B", "A", "B"]
        )
        df.iloc[:, 0] = df.iloc[:, 0].astype(np.float64)
        assert df.dtypes.iloc[2] == np.int64

    @pytest.mark.parametrize(
        ["dtypes", "init_value", "expected_value"],
        [("int64", "0", 0), ("float", "1.2", 1.2)],
    )
    # 定义一个测试方法，用于测试处理重复列名情况下的iloc设置项数据类型
    def test_iloc_setitem_dtypes_duplicate_columns(
        self, dtypes, init_value, expected_value
    ):
        # 创建一个DataFrame对象，包含一行数据和重复的列名，数据类型为object
        df = DataFrame(
            [[init_value, "str", "str2"]], columns=["a", "b", "b"], dtype=object
        )

        # 使用iloc方法选择所有行的第一列，并将其转换为指定的数据类型dtypes
        # 由于GH#45333的执行，这里直接在原地设置值，保持object数据类型不变
        df.iloc[:, 0] = df.iloc[:, 0].astype(dtypes)

        # 创建预期的DataFrame对象，包含预期的值和列名，数据类型为object
        expected_df = DataFrame(
            [[expected_value, "str", "str2"]],
            columns=["a", "b", "b"],
            dtype=object,
        )
        # 使用测试模块中的方法assert_frame_equal来比较df和expected_df是否相等
        tm.assert_frame_equal(df, expected_df)
class TestILocCallable:
    # 测试 DataFrame 的 iloc 方法中使用 callable 对象的行为
    def test_frame_iloc_getitem_callable(self):
        # GH#11485：GitHub issue #11485，用于跟踪相关问题

        # 创建一个 DataFrame 对象 df，包含两列："X" 和 "Y"，以及自定义索引列表
        df = DataFrame({"X": [1, 2, 3, 4], "Y": list("aabb")}, index=list("ABCD"))

        # 使用 lambda 函数返回指定位置的行数据
        res = df.iloc[lambda x: [1, 3]]
        # 使用 tm.assert_frame_equal 检查 res 是否与 df.iloc[[1, 3]] 相等
        tm.assert_frame_equal(res, df.iloc[[1, 3]])

        # 使用 lambda 函数返回指定位置的行和所有列数据
        res = df.iloc[lambda x: [1, 3], :]
        # 使用 tm.assert_frame_equal 检查 res 是否与 df.iloc[[1, 3], :] 相等
        tm.assert_frame_equal(res, df.iloc[[1, 3], :])

        # 使用 lambda 函数返回指定位置的行和指定列索引 0 的数据
        res = df.iloc[lambda x: [1, 3], lambda x: 0]
        # 使用 tm.assert_series_equal 检查 res 是否与 df.iloc[[1, 3], 0] 相等
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])

        # 使用 lambda 函数返回指定位置的行和列索引列表 [0] 的数据
        res = df.iloc[lambda x: [1, 3], lambda x: [0]]
        # 使用 tm.assert_frame_equal 检查 res 是否与 df.iloc[[1, 3], [0]] 相等
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

        # 混合使用列表和 lambda 函数返回指定位置的行和指定列索引 0 的数据
        res = df.iloc[[1, 3], lambda x: 0]
        # 使用 tm.assert_series_equal 检查 res 是否与 df.iloc[[1, 3], 0] 相等
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])

        # 混合使用列表和 lambda 函数返回指定位置的行和列索引列表 [0] 的数据
        res = df.iloc[[1, 3], lambda x: [0]]
        # 使用 tm.assert_frame_equal 检查 res 是否与 df.iloc[[1, 3], [0]] 相等
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

        # 使用 lambda 函数返回指定位置的行和指定列索引 0 的数据
        res = df.iloc[lambda x: [1, 3], 0]
        # 使用 tm.assert_series_equal 检查 res 是否与 df.iloc[[1, 3], 0] 相等
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])

        # 使用 lambda 函数返回指定位置的行和列索引列表 [0] 的数据
        res = df.iloc[lambda x: [1, 3], [0]]
        # 使用 tm.assert_frame_equal 检查 res 是否与 df.iloc[[1, 3], [0]] 相等
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

    # 测试 DataFrame 的 iloc 方法中使用 callable 对象进行设置的行为
    def test_frame_iloc_setitem_callable(self):
        # GH#11485：GitHub issue #11485，用于跟踪相关问题

        # 创建一个 DataFrame 对象 df，包含两列："X" 和 "Y"，以及自定义索引列表
        df = DataFrame(
            {"X": [1, 2, 3, 4], "Y": Series(list("aabb"), dtype=object)},
            index=list("ABCD"),
        )

        # 复制 df，然后使用 lambda 函数设置指定位置的行为 0
        res = df.copy()
        res.iloc[lambda x: [1, 3]] = 0
        # 创建期望的 DataFrame 对象 exp，然后使用 tm.assert_frame_equal 检查 res 是否与 exp 相等
        exp = df.copy()
        exp.iloc[[1, 3]] = 0
        tm.assert_frame_equal(res, exp)

        # 复制 df，然后使用 lambda 函数设置指定位置的行为 -1
        res = df.copy()
        res.iloc[lambda x: [1, 3], :] = -1
        # 创建期望的 DataFrame 对象 exp，然后使用 tm.assert_frame_equal 检查 res 是否与 exp 相等
        exp = df.copy()
        exp.iloc[[1, 3], :] = -1
        tm.assert_frame_equal(res, exp)

        # 复制 df，然后使用 lambda 函数设置指定位置的行和列索引 0 的值为 5
        res = df.copy()
        res.iloc[lambda x: [1, 3], lambda x: 0] = 5
        # 创建期望的 DataFrame 对象 exp，然后使用 tm.assert_frame_equal 检查 res 是否与 exp 相等
        exp = df.copy()
        exp.iloc[[1, 3], 0] = 5
        tm.assert_frame_equal(res, exp)

        # 复制 df，然后使用 lambda 函数设置指定位置的行和列索引列表 [0] 的值为 25
        res = df.copy()
        res.iloc[lambda x: [1, 3], lambda x: [0]] = 25
        # 创建期望的 DataFrame 对象 exp，然后使用 tm.assert_frame_equal 检查 res 是否与 exp 相等
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = 25
        tm.assert_frame_equal(res, exp)

        # 混合使用列表和 lambda 函数设置指定位置的行和列索引 0 的值为 -3
        res = df.copy()
        res.iloc[[1, 3], lambda x: 0] = -3
        # 创建期望的 DataFrame 对象 exp，然后使用 tm.assert_frame_equal 检查 res 是否与 exp 相等
        exp = df.copy()
        exp.iloc[[1, 3], 0] = -3
        tm.assert_frame_equal(res, exp)

        # 混合使用列表和 lambda 函数设置指定位置的行和列索引列表 [0] 的值为 -5
        res = df.copy()
        res.iloc[[1, 3], lambda x: [0]] = -5
        # 创建期望的 DataFrame 对象 exp，然后使用 tm.assert_frame_equal 检查 res 是否与 exp 相等
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = -5
        tm.assert_frame_equal(res, exp)

        # 使用 lambda 函数设置指定位置的行和列索引 0 的值为 10
        res = df.copy()
        res.iloc[lambda x: [1, 3], 0] = 10
        # 创建期望的 DataFrame 对象 exp，然后使用 tm.assert_frame_equal 检查 res 是否与 exp 相等
        exp = df.copy()
        exp.iloc[[1, 3], 0] = 10
        tm.assert_frame_equal(res, exp)

        # 使用 lambda 函数设置指定位置的行和列索引列表 [0] 的值为 [-5, -5]
        res = df.copy()
        res.iloc[lambda x: [1, 3], [0]] = [-5, -5]
        # 创建期望的 DataFrame 对象 exp，然后使用 tm.assert_frame_equal 检查 res 是否与 exp 相等
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = [-5, -5]
        tm.assert_frame_equal(res, exp)
    # 定义一个测试函数，测试 Series 对象的 iloc 功能
    def test_iloc(self):
        # 创建一个包含随机标准正态分布数据的 Series 对象，索引为偶数序列
        ser = Series(
            np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
        )
        # 备份原始的 Series 对象
        ser_original = ser.copy()

        # 遍历 Series 对象的长度
        for i in range(len(ser)):
            # 使用 iloc 获取第 i 个位置的值
            result = ser.iloc[i]
            # 使用索引操作获取第 i 个位置对应的值
            exp = ser[ser.index[i]]
            # 使用断言验证结果与期望值的近似性
            tm.assert_almost_equal(result, exp)

        # 通过传递一个切片进行测试
        result = ser.iloc[slice(1, 3)]
        # 使用 loc 方法获取期望结果的切片
        expected = ser.loc[2:4]
        # 使用断言验证 Series 对象的相等性
        tm.assert_series_equal(result, expected)

        # 测试切片是否为视图
        with tm.assert_produces_warning(None):
            # 确保不会产生虚假的 FutureWarning
            result[:] = 0
        # 使用断言验证修改后的 Series 是否与原始备份相等
        tm.assert_series_equal(ser, ser_original)

        # 使用整数列表进行测试
        result = ser.iloc[[0, 2, 3, 4, 5]]
        # 使用 reindex 方法获取期望的 Series 对象
        expected = ser.reindex(ser.index[[0, 2, 3, 4, 5]])
        # 使用断言验证 Series 对象的相等性
        tm.assert_series_equal(result, expected)

    # 测试在索引非唯一的情况下的 iloc 获取操作
    def test_iloc_getitem_nonunique(self):
        # 创建一个具有重复索引的 Series 对象
        ser = Series([0, 1, 2], index=[0, 1, 0])
        # 使用 iloc 获取第二个位置的值，应该返回 2
        assert ser.iloc[2] == 2

    # 测试纯位置基础的 iloc 设置项
    def test_iloc_setitem_pure_position_based(self):
        # GH#22046
        # 创建两个 Series 对象
        ser1 = Series([1, 2, 3])
        ser2 = Series([4, 5, 6], index=[1, 0, 2])
        # 使用 iloc 进行切片赋值
        ser1.iloc[1:3] = ser2.iloc[1:3]
        # 创建期望的 Series 对象
        expected = Series([1, 5, 6])
        # 使用断言验证 Series 对象的相等性
        tm.assert_series_equal(ser1, expected)

    # 测试在存在 nullable int64 类型且大小为 1 的 NaN 时的 iloc 功能
    def test_iloc_nullable_int64_size_1_nan(self):
        # GH 31861
        # 创建一个包含 NaN 的 DataFrame 对象
        result = DataFrame({"a": ["test"], "b": [np.nan]})
        # 验证未来可能的警告是否会被触发
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 尝试将列 'b' 转换为 Int64 类型
            result.loc[:, "b"] = result.loc[:, "b"].astype("Int64")
        # 创建期望的 DataFrame 对象
        expected = DataFrame({"a": ["test"], "b": array([NA], dtype="Int64")})
        # 使用断言验证 DataFrame 对象的相等性
        tm.assert_frame_equal(result, expected)
```