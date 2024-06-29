# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_indexing.py`

```
# 导入所需的模块和库
import array  # 导入 array 模块
from datetime import datetime  # 从 datetime 模块导入 datetime 类
import re  # 导入 re 模块，用于正则表达式操作
import weakref  # 导入 weakref 模块，用于弱引用支持

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 测试框架

from pandas._config import using_pyarrow_string_dtype  # 从 pandas._config 模块导入 using_pyarrow_string_dtype
from pandas.errors import IndexingError  # 从 pandas.errors 模块导入 IndexingError 异常类

from pandas.core.dtypes.common import (  # 从 pandas.core.dtypes.common 模块导入以下函数
    is_float_dtype,  # 检查是否为浮点数类型
    is_integer_dtype,  # 检查是否为整数类型
    is_object_dtype,  # 检查是否为对象类型
)

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
from pandas import (  # 从 pandas 导入以下类和函数
    DataFrame,  # 数据帧类
    Index,  # 索引类
    NaT,  # 代表缺失日期时间值的对象
    Series,  # 系列类
    date_range,  # 创建日期范围
    offsets,  # 偏移量函数
    timedelta_range,  # 创建时间间隔范围
)
import pandas._testing as tm  # 导入 pandas._testing 模块，并使用 tm 别名
from pandas.tests.indexing.common import _mklbl  # 从 pandas.tests.indexing.common 模块导入 _mklbl 函数
from pandas.tests.indexing.test_floats import gen_obj  # 从 pandas.tests.indexing.test_floats 模块导入 gen_obj 函数

# ------------------------------------------------------------------------
# 索引测试用例


class TestFancy:
    """纯粹的获取/设置项目和高级索引"""

    def test_setitem_ndarray_1d(self):
        # GH5508
        # 使用 1 维 ndarray 进行设置项目

        # 创建具有整数索引的 DataFrame 对象
        df = DataFrame(index=Index(np.arange(1, 11), dtype=np.int64))
        # 向 df 添加名为 'foo' 的列，填充为全零的浮点数数组
        df["foo"] = np.zeros(10, dtype=np.float64)
        # 向 df 添加名为 'bar' 的列，填充为全零的复数数值数组
        df["bar"] = np.zeros(10, dtype=complex)

        # 设置不合法的操作，应引发 ValueError 异常
        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            # 使用数组设置索引范围为 [2:5] 的 'bar' 列
            df.loc[df.index[2:5], "bar"] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])

        # 设置合法的操作，更新索引范围为 [2:6] 的 'bar' 列
        df.loc[df.index[2:6], "bar"] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])

        # 验证结果是否符合预期
        result = df.loc[df.index[2:6], "bar"]
        expected = Series(
            [2.33j, 1.23 + 0.1j, 2.2, 1.0], index=[3, 4, 5, 6], name="bar"
        )
        tm.assert_series_equal(result, expected)

    def test_setitem_ndarray_1d_2(self):
        # GH5508
        # 使用 1 维 ndarray 进行设置项目，第二个测试用例

        # 创建具有整数索引的 DataFrame 对象
        df = DataFrame(index=Index(np.arange(1, 11)))
        # 向 df 添加名为 'foo' 的列，填充为全零的浮点数数组
        df["foo"] = np.zeros(10, dtype=np.float64)
        # 向 df 添加名为 'bar' 的列，填充为全零的复数数值数组
        df["bar"] = np.zeros(10, dtype=complex)

        # 设置不合法的操作，应引发 ValueError 异常
        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            # 使用数组设置索引范围为 [2:5] 的 df
            df[2:5] = np.arange(1, 4) * 1j

    @pytest.mark.filterwarnings(
        "ignore:Series.__getitem__ treating keys as positions is deprecated:"
        "FutureWarning"
    )
    # 定义一个测试方法，用于测试对三维 ndarray 进行索引操作的异常情况
    def test_getitem_ndarray_3d(self, index, frame_or_series, indexer_sli):
        # GH 25567
        # 根据输入的 frame_or_series 和 index 生成相应的对象
        obj = gen_obj(frame_or_series, index)
        # 使用 indexer_sli 对 obj 进行索引操作
        idxr = indexer_sli(obj)
        # 创建一个形状为 (2, 2, 2)，元素取值范围在 [0, 5) 的三维随机 ndarray
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        # 初始化消息列表
        msgs = []
        # 如果 frame_or_series 是 Series，并且 indexer_sli 是 tm.setitem 或 tm.iloc
        if frame_or_series is Series and indexer_sli in [tm.setitem, tm.iloc]:
            # 添加维度数量不匹配的错误消息
            msgs.append(r"Wrong number of dimensions. values.ndim > ndim \[3 > 1\]")
        # 如果 frame_or_series 是 Series 或者 indexer_sli 是 tm.iloc
        if frame_or_series is Series or indexer_sli is tm.iloc:
            # 添加缓冲区维度不匹配的错误消息
            msgs.append(r"Buffer has wrong number of dimensions \(expected 1, got 3\)")
        # 如果 indexer_sli 是 tm.loc 或者 (frame_or_series 是 Series 并且 indexer_sli 是 tm.setitem)
        if indexer_sli is tm.loc or (frame_or_series is Series and indexer_sli is tm.setitem):
            # 添加多维键无法索引的错误消息
            msgs.append("Cannot index with multidimensional key")
        # 如果 frame_or_series 是 DataFrame 并且 indexer_sli 是 tm.setitem
        if frame_or_series is DataFrame and indexer_sli is tm.setitem:
            # 添加索引数据必须是一维的错误消息
            msgs.append("Index data must be 1-dimensional")
        # 如果 index 是 pd.IntervalIndex 类型并且 indexer_sli 是 tm.iloc
        if isinstance(index, pd.IntervalIndex) and indexer_sli is tm.iloc:
            # 添加索引数据必须是一维的错误消息
            msgs.append("Index data must be 1-dimensional")
        # 如果 index 是 (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex) 中的一种
        if isinstance(index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex)):
            # 添加数据必须是一维的错误消息
            msgs.append("Data must be 1-dimensional")
        # 如果 index 的长度为 0 或者 index 是 pd.MultiIndex 类型
        if len(index) == 0 or isinstance(index, pd.MultiIndex):
            # 添加位置索引超出范围的错误消息
            msgs.append("positional indexers are out-of-bounds")
        # 如果 index 的类型是 Index 并且 index._values 不是 ndarray 类型
        if type(index) is Index and not isinstance(index._values, np.ndarray):
            # 添加值必须是一维数组的错误消息
            msgs.append("values must be a 1D array")
            # 添加只处理一维数组的错误消息
            msgs.append("only handle 1-dimensional arrays")

        # 将消息列表中的消息用 "|" 连接成一个字符串
        msg = "|".join(msgs)

        # 设置可能的错误类型
        potential_errors = (IndexError, ValueError, NotImplementedError)
        # 使用 pytest 的 raises 方法，期待捕获到 potential_errors 中的异常，并且异常消息要匹配 msg
        with pytest.raises(potential_errors, match=msg):
            # 对 nd3 进行索引操作
            idxr[nd3]

    # 使用 pytest 的 mark.filterwarnings 忽略特定警告
    @pytest.mark.filterwarnings(
        "ignore:Series.__setitem__ treating keys as positions is deprecated:"
        "FutureWarning"
    )
    # 定义一个测试方法，用于测试对三维 ndarray 进行赋值操作的异常情况
    def test_setitem_ndarray_3d(self, index, frame_or_series, indexer_sli):
        # GH 25567
        # 根据输入的 frame_or_series 和 index 生成相应的对象
        obj = gen_obj(frame_or_series, index)
        # 使用 indexer_sli 对 obj 进行索引操作
        idxr = indexer_sli(obj)
        # 创建一个形状为 (2, 2, 2)，元素取值范围在 [0, 5) 的三维随机 ndarray
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        # 如果 indexer_sli 是 tm.iloc
        if indexer_sli is tm.iloc:
            # 设置错误类型为 ValueError
            err = ValueError
            # 设置错误消息，指示不能使用大于 obj.ndim 的维度进行赋值
            msg = f"Cannot set values with ndim > {obj.ndim}"
        else:
            # 设置错误类型为 ValueError
            err = ValueError
            # 将多个错误消息用 "|" 连接成一个字符串
            msg = "|".join(
                [
                    r"Buffer has wrong number of dimensions \(expected 1, got 3\)",
                    "Cannot set values with ndim > 1",
                    "Index data must be 1-dimensional",
                    "Data must be 1-dimensional",
                    "Array conditional must be same shape as self",
                ]
            )

        # 使用 pytest 的 raises 方法，期待捕获到 err 类型的异常，并且异常消息要匹配 msg
        with pytest.raises(err, match=msg):
            # 对 nd3 进行赋值操作
            idxr[nd3] = 0
    # 测试获取零维数组项的情况
    def test_getitem_ndarray_0d(self):
        # GH#24924
        # 创建一个包含单个元素 0 的 NumPy 数组作为键
        key = np.array(0)

        # 创建一个包含两行两列的 DataFrame
        df = DataFrame([[1, 2], [3, 4]])
        # 使用数组作为键，获取 DataFrame 中的列
        result = df[key]
        # 期望的结果是包含索引为 0 的一列数据的 Series
        expected = Series([1, 3], name=0)
        tm.assert_series_equal(result, expected)

        # 创建一个包含两个元素的 Series
        ser = Series([1, 2])
        # 使用数组作为键，获取 Series 中的元素
        result = ser[key]
        # 断言结果应为索引为 0 的元素，即 1
        assert result == 1

    # 测试 np.inf 作为索引时的数据类型提升
    def test_inf_upcast(self):
        # GH 16957
        # 我们应该能够使用 np.inf 作为键
        # np.inf 应该导致索引转换为浮点数

        # 创建一个只有一列的 DataFrame
        df = DataFrame(columns=[0])
        # 在行索引为 1 和 2 的位置插入数据
        df.loc[1] = 1
        df.loc[2] = 2
        # 在索引为 np.inf 的位置插入数据 3
        df.loc[np.inf] = 3

        # 确保我们可以查找到值
        assert df.loc[np.inf, 0] == 3

        # 获取 DataFrame 的索引
        result = df.index
        # 期望的索引应包含 1, 2, np.inf，数据类型为 np.float64
        expected = Index([1, 2, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    # 测试设置元素数据类型提升
    def test_setitem_dtype_upcast(self):
        # GH3216
        # 创建一个包含字典的 DataFrame
        df = DataFrame([{"a": 1}, {"a": 3, "b": 2}])
        # 设置列 'c' 的值为 NaN
        df["c"] = np.nan
        # 断言列 'c' 的数据类型为 np.float64
        assert df["c"].dtype == np.float64

        # 使用字符串 "foo" 设置某个元素，预期会产生未来警告
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            df.loc[0, "c"] = "foo"
        # 期望的 DataFrame 包含 'a', 'b', 'c' 三列，其中 'c' 列的元素类型为对象类型 Series
        expected = DataFrame(
            {"a": [1, 3], "b": [np.nan, 2], "c": Series(["foo", np.nan], dtype=object)}
        )
        tm.assert_frame_equal(df, expected)

    # 使用参数化测试来测试元素数据类型提升的情况
    @pytest.mark.parametrize("val", [3.14, "wxyz"])
    def test_setitem_dtype_upcast2(self, val):
        # GH10280
        # 创建一个 2x3 的整数类型 DataFrame
        df = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3),
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )

        # 复制 DataFrame 的左侧部分
        left = df.copy()
        # 使用参数化的值设置某个元素，预期会产生未来警告
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            left.loc["a", "bar"] = val
        # 期望的 DataFrame 包含 'foo', 'bar', 'baz' 三列
        right = DataFrame(
            [[0, val, 2], [3, 4, 5]],
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )

        tm.assert_frame_equal(left, right)
        # 确保 'foo', 'baz' 列的数据类型仍为整数类型
        assert is_integer_dtype(left["foo"])
        assert is_integer_dtype(left["baz"])

    # 测试元素数据类型提升的情况
    def test_setitem_dtype_upcast3(self):
        # 创建一个浮点数类型的 DataFrame
        left = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3) / 10.0,
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )
        # 使用字符串 "wxyz" 设置某个元素，预期会产生未来警告
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            left.loc["a", "bar"] = "wxyz"

        # 期望的 DataFrame 包含 'foo', 'bar', 'baz' 三列
        right = DataFrame(
            [[0, "wxyz", 0.2], [0.3, 0.4, 0.5]],
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )

        tm.assert_frame_equal(left, right)
        # 确保 'foo', 'baz' 列的数据类型仍为浮点数类型
        assert is_float_dtype(left["foo"])
        assert is_float_dtype(left["baz"])
    # 测试函数：使用“花式索引”处理重复列名情况
    def test_dups_fancy_indexing(self):
        # GH 3455

        # 创建一个包含单位矩阵的 DataFrame，列名为["a", "a", "b"]
        df = DataFrame(np.eye(3), columns=["a", "a", "b"])
        # 对 DataFrame 进行“花式索引”，选择列["b", "a"]，并获取其列名
        result = df[["b", "a"]].columns
        # 预期的结果是一个 Index 对象，包含列名["b", "a", "a"]
        expected = Index(["b", "a", "a"])
        # 使用测试框架验证 result 是否与 expected 相等
        tm.assert_index_equal(result, expected)

    # 测试函数：处理不同数据类型间的“花式索引”
    def test_dups_fancy_indexing_across_dtypes(self):
        # across dtypes

        # 创建一个包含不同数据类型的 DataFrame
        df = DataFrame([[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]], columns=list("aaaaaaa"))
        # 创建一个期望的 DataFrame，与 df 结构相同
        result = DataFrame([[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]])
        # 将 result 的列名设为与 df 相同的列名，GH#3468
        result.columns = list("aaaaaaa")

        # GH#3509 对使用重复列名进行索引的情况进行测试
        df.iloc[:, 4]  # 烟雾测试，选择索引为4的列
        result.iloc[:, 4]  # 烟雾测试，选择索引为4的列

        # 使用测试框架验证 df 是否与 result 相等
        tm.assert_frame_equal(df, result)

    # 测试函数：处理未按顺序选择的重复列名情况
    def test_dups_fancy_indexing_not_in_order(self):
        # GH 3561, dups not in selected order

        # 创建一个包含不同索引的 DataFrame
        df = DataFrame(
            {"test": [5, 7, 9, 11], "test1": [4.0, 5, 6, 7], "other": list("abcd")},
            index=["A", "A", "B", "C"],
        )
        # 选择特定行索引
        rows = ["C", "B"]
        # 创建预期结果的 DataFrame
        expected = DataFrame(
            {"test": [11, 9], "test1": [7.0, 6], "other": ["d", "c"]}, index=rows
        )
        # 使用 loc 方法根据行索引选择子集
        result = df.loc[rows]
        # 使用测试框架验证 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

        # 再次使用 loc 方法，但将行索引封装在 Index 对象中
        result = df.loc[Index(rows)]
        # 使用测试框架验证 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

        # 使用不在索引中的行索引列表进行测试，预期会引发 KeyError 异常
        rows = ["C", "B", "E"]
        with pytest.raises(KeyError, match="not in index"):
            df.loc[rows]

        # 见 GH5553，确保使用正确的索引器
        rows = ["F", "G", "H", "C", "B", "E"]
        with pytest.raises(KeyError, match="not in index"):
            df.loc[rows]

    # 测试函数：仅包含缺失标签的“花式索引”
    def test_dups_fancy_indexing_only_missing_label(self, using_infer_string):
        # List containing only missing label

        # 创建一个具有随机数据的 DataFrame，使用列表作为索引
        dfnu = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), index=list("AABCD")
        )
        # 如果使用 infer_string，则预期引发 KeyError 异常，匹配特定错误信息
        if using_infer_string:
            with pytest.raises(
                KeyError,
                match=re.escape(
                    "\"None of [Index(['E'], dtype='string')] are in the [index]\""
                ),
            ):
                dfnu.loc[["E"]]
        else:
            # 否则，预期引发 KeyError 异常，匹配特定错误信息
            with pytest.raises(
                KeyError,
                match=re.escape(
                    "\"None of [Index(['E'], dtype='object')] are in the [index]\""
                ),
            ):
                dfnu.loc[["E"]]

    # 测试函数：处理含有缺失标签的“花式索引”
    @pytest.mark.parametrize("vals", [[0, 1, 2], list("abc")])
    def test_dups_fancy_indexing_missing_label(self, vals):
        # GH 4619; duplicate indexer with missing label

        # 创建一个包含单列的 DataFrame
        df = DataFrame({"A": vals})
        # 预期引发 KeyError 异常，因为选择的索引中包含不在 DataFrame 索引中的标签
        with pytest.raises(KeyError, match="not in index"):
            df.loc[[0, 8, 0]]

    # 测试函数：处理非唯一索引器和非唯一选择器的情况
    def test_dups_fancy_indexing_non_unique(self):
        # non unique with non unique selector

        # 创建一个包含重复索引和非唯一选择器的 DataFrame
        df = DataFrame({"test": [5, 7, 9, 11]}, index=["A", "A", "B", "C"])
        # 预期引发 KeyError 异常，因为选择的索引中包含不在 DataFrame 索引中的标签
        with pytest.raises(KeyError, match="not in index"):
            df.loc[["A", "A", "E"]]
    # 定义一个测试函数，用于测试 DataFrame 的索引操作中的重复值处理
    def test_dups_fancy_indexing2(self):
        # GH 5835 表示 GitHub 上的 issue 编号，此处关注重复的索引和缺失值的处理
        # 创建一个 5x5 的 DataFrame，数据由标准正态分布随机生成，列名包括 "A", "B", "B", "B", "A"
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 5)),
            columns=["A", "B", "B", "B", "A"],
        )

        # 使用 pytest 检测是否会抛出 KeyError，并且错误消息中包含 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            # 尝试在 DataFrame 中使用 loc 方法，选择所有行和列 ["A", "B", "C"]
            df.loc[:, ["A", "B", "C"]]

    # 定义另一个测试函数，用于测试 DataFrame 的多轴索引操作
    def test_dups_fancy_indexing3(self):
        # GH 6504 表示 GitHub 上的 issue 编号，关注多轴索引
        # 创建一个 9x2 的 DataFrame，数据由标准正态分布随机生成，行索引为 [1, 1, 1, 2, 2, 2, 3, 3, 3]，列名为 ["a", "b"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((9, 2)),
            index=[1, 1, 1, 2, 2, 2, 3, 3, 3],
            columns=["a", "b"],
        )

        # 期望结果是选择 DataFrame 的前 6 行
        expected = df.iloc[0:6]
        # 使用 loc 方法选择行索引为 [1, 2] 的数据
        result = df.loc[[1, 2]]
        # 使用 assert_frame_equal 检查结果是否与期望一致
        tm.assert_frame_equal(result, expected)

        # 期望结果是整个 DataFrame
        expected = df
        # 使用 loc 方法选择所有行和列 ["a", "b"]
        result = df.loc[:, ["a", "b"]]
        # 使用 assert_frame_equal 检查结果是否与期望一致
        tm.assert_frame_equal(result, expected)

        # 期望结果是选择 DataFrame 的前 6 行，并且只包括列 ["a", "b"]
        expected = df.iloc[0:6, :]
        # 使用 loc 方法选择行索引为 [1, 2] 的数据，并且只包括列 ["a", "b"]
        result = df.loc[[1, 2], ["a", "b"]]
        # 使用 assert_frame_equal 检查结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，用于测试 Series 的索引操作中的重复整数索引情况
    def test_duplicate_int_indexing(self, indexer_sl):
        # GH 17347 表示 GitHub 上的 issue 编号，关注重复整数索引的处理
        # 创建一个 Series，索引为 [1, 1, 3]，值为 [0, 1, 2]
        ser = Series(range(3), index=[1, 1, 3])
        # 期望结果是一个 Series，索引为 [1, 1]，值为 [0, 1]
        expected = Series(range(2), index=[1, 1])
        # 使用 indexer_sl 函数对 ser 进行索引操作，选择索引为 [1] 的数据
        result = indexer_sl(ser)[[1]]
        # 使用 assert_series_equal 检查结果是否与期望一致
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，用于测试 DataFrame 在混合索引操作时的 bug
    def test_indexing_mixed_frame_bug(self):
        # GH3492 表示 GitHub 上的 issue 编号，关注 DataFrame 在混合索引操作时的 bug
        # 创建一个包含两列 "a" 和 "b" 的 DataFrame，索引为 {1: "aaa", 2: "bbb", 3: "ccc"} 和 {1: 111, 2: 222, 3: 333}
        df = DataFrame(
            {"a": {1: "aaa", 2: "bbb", 3: "ccc"}, "b": {1: 111, 2: 222, 3: 333}}
        )

        # 这行代码正常工作，正确创建了新列 "test"
        df["test"] = df["a"].apply(lambda x: "_" if x == "aaa" else x)

        # 这行代码不工作，即列 "test" 没有被正确修改
        idx = df["test"] == "_"
        temp = df.loc[idx, "a"].apply(lambda x: "-----" if x == "aaa" else x)
        df.loc[idx, "test"] = temp
        # 使用 assert 检查特定位置的值是否符合预期
        assert df.iloc[0, 2] == "-----"

    # 定义一个测试函数，用于测试 DataFrame 在多类型列表索引访问时的异常处理
    def test_multitype_list_index_access(self):
        # GH 10610 表示 GitHub 上的 issue 编号，关注多类型列表索引访问的异常处理
        # 创建一个 10x5 的 DataFrame，数据由标准均匀分布随机生成，列名为 ["a", 20, 21, 22, 23]
        df = DataFrame(
            np.random.default_rng(2).random((10, 5)), columns=["a"] + [20, 21, 22, 23]
        )

        # 使用 pytest 检测是否会抛出 KeyError，并且错误消息中包含 "'[26, -8] not in index'"
        with pytest.raises(KeyError, match=re.escape("'[26, -8] not in index'")):
            # 尝试访问 DataFrame 中不存在的索引 [22, 26, -8]
            df[[22, 26, -8]]
        
        # 使用 assert 检查 DataFrame 中第 21 列的行数是否等于总行数
        assert df[21].shape[0] == df.shape[0]
    # 定义一个测试函数，用于测试设置索引为NaN的情况
    def test_set_index_nan(self):
        # GH 3586，标识 GitHub issue 编号
        # 创建一个 DataFrame 对象，包含多个列和行，其中包含了NaN值
        df = DataFrame(
            {
                "PRuid": {
                    17: "nonQC",
                    18: "nonQC",
                    19: "nonQC",
                    20: "10",
                    21: "11",
                    22: "12",
                    23: "13",
                    24: "24",
                    25: "35",
                    26: "46",
                    27: "47",
                    28: "48",
                    29: "59",
                    30: "10",
                },
                "QC": {
                    17: 0.0,
                    18: 0.0,
                    19: 0.0,
                    20: np.nan,
                    21: np.nan,
                    22: np.nan,
                    23: np.nan,
                    24: 1.0,
                    25: np.nan,
                    26: np.nan,
                    27: np.nan,
                    28: np.nan,
                    29: np.nan,
                    30: np.nan,
                },
                "data": {
                    17: 7.9544899999999998,
                    18: 8.0142609999999994,
                    19: 7.8591520000000008,
                    20: 0.86140349999999999,
                    21: 0.87853110000000001,
                    22: 0.8427041999999999,
                    23: 0.78587700000000005,
                    24: 0.73062459999999996,
                    25: 0.81668560000000001,
                    26: 0.81927080000000008,
                    27: 0.80705009999999999,
                    28: 0.81440240000000008,
                    29: 0.80140849999999997,
                    30: 0.81307740000000006,
                },
                "year": {
                    17: 2006,
                    18: 2007,
                    19: 2008,
                    20: 1985,
                    21: 1985,
                    22: 1985,
                    23: 1985,
                    24: 1985,
                    25: 1985,
                    26: 1985,
                    27: 1985,
                    28: 1985,
                    29: 1985,
                    30: 1986,
                },
            }
        ).reset_index()

        # 设置索引为["year", "PRuid", "QC"]，然后重置索引，再根据原始列顺序重新排列列
        result = (
            df.set_index(["year", "PRuid", "QC"])
            .reset_index()
            .reindex(columns=df.columns)
        )
        # 断言结果 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(result, df)

    # 标记为预期失败的测试用例，原因是无法对 Arrow 字符串进行乘法操作
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't multiply arrow strings"
    )
    def test_multi_assign(self):
        # GH 3626, an assignment of a sub-df to a df
        # set float64 to avoid upcast when setting nan

        # 创建一个 DataFrame 对象 df，包含四列数据，其中 col2 列被设置为 float64 类型以避免在设置 nan 时的数据类型转换
        df = DataFrame(
            {
                "FC": ["a", "b", "a", "b", "a", "b"],
                "PF": [0, 0, 0, 0, 1, 1],
                "col1": list(range(6)),
                "col2": list(range(6, 12)),
            }
        ).astype({"col2": "float64"})

        # 将 df 的第二行第一列设置为 np.nan
        df.iloc[1, 0] = np.nan

        # 复制 df 到 df2
        df2 = df.copy()

        # 创建一个布尔掩码，用来选择 df2 中 FC 列非 NaN 值的行
        mask = ~df2.FC.isna()

        # 定义要更新的列名列表
        cols = ["col1", "col2"]

        # 将 df2 中符合 mask 条件的行，按元素乘以2并赋值给 dft
        dft = df2 * 2

        # 将 dft 的第四行第四列设置为 np.nan
        dft.iloc[3, 3] = np.nan

        # 创建一个期望的 DataFrame 对象 expected，用于与 df2 进行比较
        expected = DataFrame(
            {
                "FC": ["a", "b", "a", "b", "a", "b"],
                "PF": [0, 0, 0, 0, 1, 1],
                "col1": Series([0, 1, 4, 6, 8, 10]),
                "col2": [12, 7, 16, np.nan, 20, 22],
            }
        )

        # 将 dft 中符合 mask 条件的行和列 cols 的数据更新到 df2 中
        df2.loc[mask, cols] = dft.loc[mask, cols]

        # 断言 df2 与期望的结果 expected 相等
        tm.assert_frame_equal(df2, expected)

        # 用 ndarray 更新 df2 的操作
        # 强制转换为 float64 类型，因为 values 具有 float64 dtype
        # GH 14001
        expected = DataFrame(
            {
                "FC": ["a", "b", "a", "b", "a", "b"],
                "PF": [0, 0, 0, 0, 1, 1],
                "col1": [0, 1, 4, 6, 8, 10],
                "col2": [12, 7, 16, np.nan, 20, 22],
            }
        )

        # 复制 df 到 df2
        df2 = df.copy()

        # 用 dft.loc[mask, cols].values 更新 df2 中符合 mask 条件的行和列 cols 的数据
        df2.loc[mask, cols] = dft.loc[mask, cols].values

        # 断言 df2 与期望的结果 expected 相等
        tm.assert_frame_equal(df2, expected)
    def test_string_slice_empty(self):
        # 测试用例名称：test_string_slice_empty
        # Issue编号：GH 14424

        # 创建一个空的DataFrame对象
        df = DataFrame()
        # 断言DataFrame的索引不全为日期
        assert not df.index._is_all_dates
        # 使用pytest断言，预期会引发KeyError，并匹配包含"'2011'"的异常信息
        with pytest.raises(KeyError, match="'2011'"):
            df["2011"]

        # 使用pytest断言，预期会引发KeyError，并匹配以"^0$"开头的异常信息
        with pytest.raises(KeyError, match="^0$"):
            df.loc["2011", 0]

    def test_astype_assignment(self, using_infer_string):
        # 测试用例名称：test_astype_assignment
        # Issue编号：GH4312 (iloc)

        # 创建一个包含指定数据的DataFrame对象
        df_orig = DataFrame(
            [["1", "2", "3", ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )

        # 复制DataFrame对象，以便后续操作不影响原始数据
        df = df_orig.copy()

        # 根据GH#45333的规定，在2.0版本中，此设置尝试原地进行，因此保留对象dtype为object
        # 将第0列到第1列（不包含）的数据转换为np.int64类型
        df.iloc[:, 0:2] = df.iloc[:, 0:2].astype(np.int64)
        # 创建预期的DataFrame对象
        expected = DataFrame(
            [[1, 2, "3", ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        # 根据using_infer_string的值判断是否需要额外转换列A和列B为object类型
        if not using_infer_string:
            expected["A"] = expected["A"].astype(object)
            expected["B"] = expected["B"].astype(object)
        # 使用tm模块断言两个DataFrame对象是否相等
        tm.assert_frame_equal(df, expected)

        # GH5702 (loc)
        df = df_orig.copy()
        # 将"A"列数据转换为np.int64类型
        df.loc[:, "A"] = df.loc[:, "A"].astype(np.int64)
        # 创建预期的DataFrame对象
        expected = DataFrame(
            [[1, "2", "3", ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        # 根据using_infer_string的值判断是否需要额外转换列A为object类型
        if not using_infer_string:
            expected["A"] = expected["A"].astype(object)
        # 使用tm模块断言两个DataFrame对象是否相等
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        # 将"B"和"C"列的数据转换为np.int64类型
        df.loc[:, ["B", "C"]] = df.loc[:, ["B", "C"]].astype(np.int64)
        # 创建预期的DataFrame对象
        expected = DataFrame(
            [["1", 2, 3, ".4", 5, 6.0, "foo"]], columns=list("ABCDEFG")
        )
        # 根据using_infer_string的值判断是否需要额外转换列B和列C为object类型
        if not using_infer_string:
            expected["B"] = expected["B"].astype(object)
            expected["C"] = expected["C"].astype(object)
        # 使用tm模块断言两个DataFrame对象是否相等
        tm.assert_frame_equal(df, expected)

    def test_astype_assignment_full_replacements(self):
        # 测试用例名称：test_astype_assignment_full_replacements
        # 描述：全量替换/无NaN值

        # 创建一个包含指定数据的DataFrame对象
        df = DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})

        # 根据GH#45333的规定，在2.0版本中，此赋值操作原地进行，因此保留float64类型
        # 将"A"列数据转换为np.int64类型
        df.iloc[:, 0] = df["A"].astype(np.int64)
        # 创建预期的DataFrame对象
        expected = DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
        # 使用tm模块断言两个DataFrame对象是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个包含指定数据的DataFrame对象
        df = DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
        # 将"A"列数据转换为np.int64类型
        df.loc[:, "A"] = df["A"].astype(np.int64)
        # 使用tm模块断言两个DataFrame对象是否相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer", [tm.getitem, tm.loc])
    # 定义一个测试方法，用于测试索引类型的强制转换
    def test_index_type_coercion(self, indexer):
        # GH 11836
        # 如果我们有一个索引类型，并用看起来与之相同的值设置它，但实际上并不相同
        # （例如使用浮点数或字符串 '0' 进行设置）
        # 那么我们需要强制转换为对象类型

        # 对于整数索引
        for s in [Series(range(5)), Series(range(5), index=range(1, 6))]:
            # 断言索引是整数类型
            assert is_integer_dtype(s.index)

            # 创建 s 的副本 s2
            s2 = s.copy()
            # 使用 indexer 函数在 s2 上设置索引为 0.1 的位置为 0
            indexer(s2)[0.1] = 0
            # 断言现在 s2 的索引是浮点数类型
            assert is_float_dtype(s2.index)
            # 断言 indexer 函数在 s2 上索引为 0.1 的位置等于 0
            assert indexer(s2)[0.1] == 0

            # 创建 s 的副本 s2
            s2 = s.copy()
            # 使用 indexer 函数在 s2 上设置索引为 0.0 的位置为 0
            indexer(s2)[0.0] = 0
            # 如果原始索引中没有 0，则将其添加到期望的索引中
            exp = s.index
            if 0 not in s:
                exp = Index(s.index.tolist() + [0])
            # 断言 s2 的索引与期望的索引相等
            tm.assert_index_equal(s2.index, exp)

            # 创建 s 的副本 s2
            s2 = s.copy()
            # 使用 indexer 函数在 s2 上设置索引为 "0" 的位置为 0
            indexer(s2)["0"] = 0
            # 断言现在 s2 的索引是对象类型
            assert is_object_dtype(s2.index)

        # 对于浮点数索引
        for s in [Series(range(5), index=np.arange(5.0))]:
            # 断言索引是浮点数类型
            assert is_float_dtype(s.index)

            # 创建 s 的副本 s2
            s2 = s.copy()
            # 使用 indexer 函数在 s2 上设置索引为 0.1 的位置为 0
            indexer(s2)[0.1] = 0
            # 断言现在 s2 的索引是浮点数类型
            assert is_float_dtype(s2.index)
            # 断言 indexer 函数在 s2 上索引为 0.1 的位置等于 0
            assert indexer(s2)[0.1] == 0

            # 创建 s 的副本 s2
            s2 = s.copy()
            # 使用 indexer 函数在 s2 上设置索引为 0.0 的位置为 0
            indexer(s2)[0.0] = 0
            # 断言 s2 的索引与 s 的索引相等
            tm.assert_index_equal(s2.index, s.index)

            # 创建 s 的副本 s2
            s2 = s.copy()
            # 使用 indexer 函数在 s2 上设置索引为 "0" 的位置为 0
            indexer(s2)["0"] = 0
            # 断言现在 s2 的索引是对象类型
            assert is_object_dtype(s2.index)
class TestMisc:
    # 测试将浮点数作为列索引转换为混合索引
    def test_float_index_to_mixed(self):
        # 创建一个包含两列随机数的 DataFrame，列索引为浮点数
        df = DataFrame(
            {
                0.0: np.random.default_rng(2).random(10),
                1.0: np.random.default_rng(2).random(10),
            }
        )
        # 添加新列"a"，并赋值为10
        df["a"] = 10

        # 创建预期的 DataFrame，包含三列：0.0列、1.0列和常数列"a"
        expected = DataFrame({0.0: df[0.0], 1.0: df[1.0], "a": [10] * 10})
        # 使用测试工具函数检验预期结果与实际结果是否相等
        tm.assert_frame_equal(expected, df)

    # 测试在具有浮点数索引的 DataFrame 中进行非标量赋值
    def test_float_index_non_scalar_assignment(self):
        # 创建一个具有浮点数索引的 DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        # 将前两行的数据赋值为1
        df.loc[df.index[:2]] = 1
        # 创建预期的 DataFrame，对应的行数据部分被修改为1
        expected = DataFrame({"a": [1, 1, 3], "b": [1, 1, 5]}, index=df.index)
        # 使用测试工具函数检验预期结果与实际结果是否相等
        tm.assert_frame_equal(expected, df)

    # 测试使用 loc 进行完整索引视图的设置
    def test_loc_setitem_fullindex_views(self):
        # 创建一个具有浮点数索引的 DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        # 复制 DataFrame
        df2 = df.copy()
        # 使用 loc 对整个索引视图进行设置
        df.loc[df.index] = df.loc[df.index]
        # 使用测试工具函数检验预期结果与实际结果是否相等
        tm.assert_frame_equal(df, df2)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set int into string")
    # 测试右手边数据的对齐性
    def test_rhs_alignment(self):
        # GH8258, 测试确保行和列与所分配的内容对齐。覆盖统一数据类型和多类型情况
        def run_tests(df, rhs, right_loc, right_iloc):
            # 标签、索引、切片
            lbl_one, idx_one, slice_one = list("bcd"), [1, 2, 3], slice(1, 4)
            lbl_two, idx_two, slice_two = ["joe", "jolie"], [1, 2], slice(1, 3)

            # 复制 DataFrame
            left = df.copy()
            # 使用 loc 设置左侧 DataFrame 的部分数据
            left.loc[lbl_one, lbl_two] = rhs
            # 使用测试工具函数检验预期结果与实际结果是否相等
            tm.assert_frame_equal(left, right_loc)

            # 复制 DataFrame
            left = df.copy()
            # 使用 iloc 设置左侧 DataFrame 的部分数据
            left.iloc[idx_one, idx_two] = rhs
            # 使用测试工具函数检验预期结果与实际结果是否相等
            tm.assert_frame_equal(left, right_iloc)

            # 复制 DataFrame
            left = df.copy()
            # 使用 iloc 设置左侧 DataFrame 的部分数据
            left.iloc[slice_one, slice_two] = rhs
            # 使用测试工具函数检验预期结果与实际结果是否相等
            tm.assert_frame_equal(left, right_iloc)

        # 创建一个 5x4 的整数型 DataFrame
        xs = np.arange(20).reshape(5, 4)
        cols = ["jim", "joe", "jolie", "joline"]
        df = DataFrame(xs, columns=cols, index=list("abcde"), dtype="int64")

        # 右侧数据; 排列索引并乘以-2
        rhs = -2 * df.iloc[3:0:-1, 2:0:-1]

        # 预期的 "right" 结果; 将数据乘以-2
        right_iloc = df.copy()
        right_iloc["joe"] = [1, 14, 10, 6, 17]
        right_iloc["jolie"] = [2, 13, 9, 5, 18]
        right_iloc.iloc[1:4, 1:3] *= -2
        right_loc = df.copy()
        right_loc.iloc[1:4, 1:3] *= -2

        # 使用统一的数据类型运行测试
        run_tests(df, rhs, right_loc, right_iloc)

        # 使数据帧成为多类型并重新运行测试
        for frame in [df, rhs, right_loc, right_iloc]:
            frame["joe"] = frame["joe"].astype("float64")
            frame["jolie"] = frame["jolie"].map(lambda x: f"@{x}")
        right_iloc["joe"] = [1.0, "@-28", "@-20", "@-12", 17.0]
        right_iloc["jolie"] = ["@2", -26.0, -18.0, -10.0, "@18"]
        # 使用测试工具函数检验预期结果与实际结果是否相等，同时检查警告信息
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            run_tests(df, rhs, right_loc, right_iloc)
    @pytest.mark.parametrize(
        "idx", [_mklbl("A", 20), np.arange(20) + 100, np.linspace(100, 150, 20)]
    )
    def test_str_label_slicing_with_negative_step(self, idx):
        SLC = pd.IndexSlice  # 定义索引切片对象SLC

        idx = Index(idx)  # 将idx转换为索引对象Index
        ser = Series(np.arange(20), index=idx)  # 创建一个Series，使用np.arange(20)作为数据，idx作为索引
        tm.assert_indexing_slices_equivalent(ser, SLC[idx[9] :: -1], SLC[9::-1])  # 断言索引切片操作的等效性
        tm.assert_indexing_slices_equivalent(ser, SLC[: idx[9] : -1], SLC[:8:-1])  # 断言索引切片操作的等效性
        tm.assert_indexing_slices_equivalent(
            ser, SLC[idx[13] : idx[9] : -1], SLC[13:8:-1]  # 断言索引切片操作的等效性
        )
        tm.assert_indexing_slices_equivalent(ser, SLC[idx[9] : idx[13] : -1], SLC[:0])  # 断言索引切片操作的等效性

    def test_slice_with_zero_step_raises(self, index, indexer_sl, frame_or_series):
        obj = frame_or_series(np.arange(len(index)), index=index)  # 创建DataFrame或Series对象
        with pytest.raises(ValueError, match="slice step cannot be zero"):  # 断言抛出 ValueError 异常
            indexer_sl(obj)[::0]  # 使用索引器 indexer_sl 进行切片操作，步长为0，预期会抛出异常

    def test_loc_setitem_indexing_assignment_dict_already_exists(self):
        index = Index([-5, 0, 5], name="z")  # 创建索引对象Index
        df = DataFrame({"x": [1, 2, 6], "y": [2, 2, 8]}, index=index)  # 创建DataFrame对象df
        expected = df.copy()  # 复制DataFrame对象df为expected
        rhs = {"x": 9, "y": 99}  # 定义右侧字典rhs
        df.loc[5] = rhs  # 使用loc方法进行索引赋值操作
        expected.loc[5] = [9, 99]  # 期望的索引赋值结果
        tm.assert_frame_equal(df, expected)  # 断言DataFrame对象df与expected相等

        # GH#38335 same thing, mixed dtypes
        df = DataFrame({"x": [1, 2, 6], "y": [2.0, 2.0, 8.0]}, index=index)  # 创建DataFrame对象df，包含混合数据类型
        df.loc[5] = rhs  # 使用loc方法进行索引赋值操作
        expected = DataFrame({"x": [1, 2, 9], "y": [2.0, 2.0, 99.0]}, index=index)  # 期望的DataFrame对象expected
        tm.assert_frame_equal(df, expected)  # 断言DataFrame对象df与expected相等

    def test_iloc_getitem_indexing_dtypes_on_empty(self):
        # Check that .iloc returns correct dtypes GH9983
        df = DataFrame({"a": [1, 2, 3], "b": ["b", "b2", "b3"]})  # 创建DataFrame对象df
        df2 = df.iloc[[], :]  # 使用iloc方法进行索引操作，得到空的DataFrame对象df2

        assert df2.loc[:, "a"].dtype == np.int64  # 断言df2的"a"列数据类型为np.int64
        tm.assert_series_equal(df2.loc[:, "a"], df2.iloc[:, 0])  # 断言df2的"a"列与第一列的Series相等

    @pytest.mark.parametrize("size", [5, 999999, 1000000])
    def test_loc_range_in_series_indexing(self, size):
        # range can cause an indexing error
        # GH 11652
        s = Series(index=range(size), dtype=np.float64)  # 创建具有指定大小和数据类型的Series对象s
        s.loc[range(1)] = 42  # 使用loc方法进行索引赋值操作
        tm.assert_series_equal(s.loc[range(1)], Series(42.0, index=[0]))  # 断言Series对象s的索引范围为[0]，数据为42.0

        s.loc[range(2)] = 43  # 使用loc方法进行索引赋值操作
        tm.assert_series_equal(s.loc[range(2)], Series(43.0, index=[0, 1]))  # 断言Series对象s的索引范围为[0, 1]，数据为43.0

    def test_partial_boolean_frame_indexing(self):
        # GH 17170
        df = DataFrame(
            np.arange(9.0).reshape(3, 3), index=list("abc"), columns=list("ABC")
        )  # 创建DataFrame对象df
        index_df = DataFrame(1, index=list("ab"), columns=list("AB"))  # 创建索引DataFrame对象index_df
        result = df[index_df.notnull()]  # 使用布尔DataFrame索引df的行
        expected = DataFrame(
            np.array([[0.0, 1.0, np.nan], [3.0, 4.0, np.nan], [np.nan] * 3]),
            index=list("abc"),
            columns=list("ABC"),
        )  # 期望的DataFrame对象expected
        tm.assert_frame_equal(result, expected)  # 断言DataFrame对象result与expected相等
    # 测试函数，用于验证在不产生引用循环的情况下的行为
    def test_no_reference_cycle(self):
        # 创建一个包含两列的 DataFrame 对象
        df = DataFrame({"a": [0, 1], "b": [2, 3]})
        # 对于每个指定的方法名，通过 getattr 动态获取方法的引用并调用
        for name in ("loc", "iloc", "at", "iat"):
            getattr(df, name)
        # 创建 DataFrame 对象的弱引用 wr
        wr = weakref.ref(df)
        # 删除原始的 DataFrame 对象 df
        del df
        # 使用弱引用 wr 获取原始对象，确认原始对象已被删除
        assert wr() is None

    # 测试函数，验证在包含 NaN 值的 Series 上进行标签索引的行为
    def test_label_indexing_on_nan(self, nulls_fixture):
        # 创建一个包含 NaN 值的 Series 对象
        df = Series([1, "{1,2}", 1, nulls_fixture])
        # 调用 value_counts 方法，计算每个值的出现次数，包括 NaN 值
        vc = df.value_counts(dropna=False)
        # 使用 loc 方法，在计数结果中查找 nulls_fixture 的计数值
        result1 = vc.loc[nulls_fixture]
        # 直接通过中括号语法，在计数结果中查找 nulls_fixture 的计数值
        result2 = vc[nulls_fixture]

        expected = 1
        # 断言通过 loc 方法得到的计数值与期望值一致
        assert result1 == expected
        # 断言通过中括号语法得到的计数值与期望值一致
        assert result2 == expected
class TestDataframeNoneCoercion:
    EXPECTED_SINGLE_ROW_RESULTS = [
        # 对于数值系列，应该强制转换为 NaN。
        ([1, 2, 3], [np.nan, 2, 3], FutureWarning),
        ([1.0, 2.0, 3.0], [np.nan, 2.0, 3.0], None),
        # 对于日期时间系列，应该强制转换为 NaT。
        (
            [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)],
            None,
        ),
        # 对于对象类型，应该保留 None 值。
        (["foo", "bar", "baz"], [None, "bar", "baz"], None),
    ]

    @pytest.mark.parametrize("expected", EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_loc(self, expected):
        start_data, expected_result, warn = expected

        # 创建起始数据框架，包含单列 "foo"，填充起始数据
        start_dataframe = DataFrame({"foo": start_data})
        # 使用 loc 方法将第一行 "foo" 列设为 None
        start_dataframe.loc[0, ["foo"]] = None

        # 创建预期数据框架，与预期结果对应
        expected_dataframe = DataFrame({"foo": expected_result})
        # 使用测试框架中的 assert_frame_equal 方法比较起始和预期数据框架
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize("expected", EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_setitem_and_dataframe(self, expected):
        start_data, expected_result, warn = expected

        # 创建起始数据框架，包含单列 "foo"，填充起始数据
        start_dataframe = DataFrame({"foo": start_data})
        # 使用布尔索引和 setitem 操作将符合条件的行设置为 None
        start_dataframe[start_dataframe["foo"] == start_dataframe["foo"][0]] = None

        # 创建预期数据框架，与预期结果对应
        expected_dataframe = DataFrame({"foo": expected_result})
        # 使用测试框架中的 assert_frame_equal 方法比较起始和预期数据框架
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize("expected", EXPECTED_SINGLE_ROW_RESULTS)
    def test_none_coercion_loc_and_dataframe(self, expected):
        start_data, expected_result, warn = expected

        # 创建起始数据框架，包含单列 "foo"，填充起始数据
        start_dataframe = DataFrame({"foo": start_data})
        # 使用 loc 方法和布尔索引将符合条件的行设置为 None
        start_dataframe.loc[start_dataframe["foo"] == start_dataframe["foo"][0]] = None

        # 创建预期数据框架，与预期结果对应
        expected_dataframe = DataFrame({"foo": expected_result})
        # 使用测试框架中的 assert_frame_equal 方法比较起始和预期数据框架
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    def test_none_coercion_mixed_dtypes(self):
        # 创建包含不同数据类型的起始数据框架
        start_dataframe = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.0, 2.0, 3.0],
                "c": [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                "d": ["a", "b", "c"],
            }
        )
        # 使用 iloc 方法将第一行数据设置为 None
        start_dataframe.iloc[0] = None

        # 创建预期数据框架，与预期结果对应
        exp = DataFrame(
            {
                "a": [np.nan, 2, 3],
                "b": [np.nan, 2.0, 3.0],
                "c": [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)],
                "d": [None, "b", "c"],
            }
        )
        # 使用测试框架中的 assert_frame_equal 方法比较起始和预期数据框架
        tm.assert_frame_equal(start_dataframe, exp)
    # 测试函数，用于测试设置 DatetimeArray 中单个字符串标量的情况
    def test_setitem_dt64_string_scalar(self, tz_naive_fixture, indexer_sli):
        # 从 tz_naive_fixture 中获取时区信息
        tz = tz_naive_fixture

        # 创建一个包含三个日期的 DatetimeIndex，带有时区信息
        dti = date_range("2016-01-01", periods=3, tz=tz)
        # 将 DatetimeIndex 转换为 Series
        ser = Series(dti.copy(deep=True))

        # 获取 Series 内部的值数组
        values = ser._values

        # 新值为字符串 "2018-01-01"，验证是否可以设置该值
        newval = "2018-01-01"
        values._validate_setitem_value(newval)

        # 使用索引器 indexer_sli 获取索引并设置新值
        indexer_sli(ser)[0] = newval

        # 如果时区为 None，则验证 Series 的 dtype 与原始 DatetimeIndex 的 dtype 相同，
        # 并且 Series 的值数组指向同一个 ndarray 对象
        if tz is None:
            # TODO(EA2D): we can make this no-copy in tz-naive case too
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            # 否则，验证 Series 的值数组与原始值数组是同一个对象
            assert ser._values is values

    # 测试函数，用于测试设置 DatetimeArray 中多个字符串值的情况
    @pytest.mark.parametrize("box", [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize(
        "key", [[0, 1], slice(0, 2), np.array([True, True, False])]
    )
    def test_setitem_dt64_string_values(self, tz_naive_fixture, indexer_sli, key, box):
        # 从 tz_naive_fixture 中获取时区信息
        tz = tz_naive_fixture

        # 如果 key 是 slice 且 indexer_sli 是 tm.loc，则将 key 重新设置为 slice(0, 1)
        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)

        # 创建一个包含三个日期的 DatetimeIndex，带有时区信息
        dti = date_range("2016-01-01", periods=3, tz=tz)
        # 将 DatetimeIndex 转换为 Series
        ser = Series(dti.copy(deep=True))

        # 获取 Series 内部的值数组
        values = ser._values

        # 新值为字符串数组 ["2019-01-01", "2010-01-02"]，验证是否可以设置这些值
        newvals = box(["2019-01-01", "2010-01-02"])
        values._validate_setitem_value(newvals)

        # 使用索引器 indexer_sli 获取索引并设置新值
        indexer_sli(ser)[key] = newvals

        # 如果时区为 None，则验证 Series 的 dtype 与原始 DatetimeIndex 的 dtype 相同，
        # 并且 Series 的值数组指向同一个 ndarray 对象
        if tz is None:
            # TODO(EA2D): we can make this no-copy in tz-naive case too
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            # 否则，验证 Series 的值数组与原始值数组是同一个对象
            assert ser._values is values

    # 测试函数，用于测试设置 TimedeltaArray 中单个标量的情况
    @pytest.mark.parametrize("scalar", ["3 Days", offsets.Hour(4)])
    def test_setitem_td64_scalar(self, indexer_sli, scalar):
        # 创建一个包含三个时间差的 TimedeltaIndex
        tdi = timedelta_range("1 Day", periods=3)
        # 将 TimedeltaIndex 转换为 Series
        ser = Series(tdi.copy(deep=True))

        # 获取 Series 内部的值数组
        values = ser._values

        # 验证是否可以设置标量值 scalar
        values._validate_setitem_value(scalar)

        # 使用索引器 indexer_sli 获取索引并设置新值
        indexer_sli(ser)[0] = scalar

        # 验证 Series 的值数组指向同一个 ndarray 对象
        assert ser._values._ndarray is values._ndarray

    # 测试函数，用于测试设置 TimedeltaArray 中多个字符串值的情况
    @pytest.mark.parametrize("box", [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize(
        "key", [[0, 1], slice(0, 2), np.array([True, True, False])]
    )
    def test_setitem_td64_string_values(self, indexer_sli, key, box):
        # 如果 key 是 slice 且 indexer_sli 是 tm.loc，则将 key 重新设置为 slice(0, 1)
        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)

        # 创建一个包含三个时间差的 TimedeltaIndex
        tdi = timedelta_range("1 Day", periods=3)
        # 将 TimedeltaIndex 转换为 Series
        ser = Series(tdi.copy(deep=True))

        # 获取 Series 内部的值数组
        values = ser._values

        # 新值为字符串数组 ["10 Days", "44 hours"]，验证是否可以设置这些值
        newvals = box(["10 Days", "44 hours"])
        values._validate_setitem_value(newvals)

        # 使用索引器 indexer_sli 获取索引并设置新值
        indexer_sli(ser)[key] = newvals

        # 验证 Series 的值数组指向同一个 ndarray 对象
        assert ser._values._ndarray is values._ndarray
def test_extension_array_cross_section():
    # 定义测试函数：验证对同类扩展数组的切片应该返回扩展数组
    df = DataFrame(
        {
            "A": pd.array([1, 2], dtype="Int64"),
            "B": pd.array([3, 4], dtype="Int64"),
        },
        index=["a", "b"],
    )
    # 期望结果：从DataFrame中取出索引为'a'的行，期望得到一个Series
    expected = Series(pd.array([1, 3], dtype="Int64"), index=["A", "B"], name="a")
    # 实际结果：从DataFrame中取出索引为'a'的行
    result = df.loc["a"]
    # 断言：验证实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)

    # 实际结果：从DataFrame中按照位置取出第一个行
    result = df.iloc[0]
    # 断言：验证实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)


def test_extension_array_cross_section_converts():
    # 当所有数值列都是数值时，返回数值Series
    df = DataFrame(
        {
            "A": pd.array([1, 2], dtype="Int64"),
            "B": np.array([1, 2], dtype="int64"),
        },
        index=["a", "b"],
    )
    # 实际结果：从DataFrame中取出索引为'a'的行
    result = df.loc["a"]
    # 期望结果：预期得到一个包含数值的Series
    expected = Series([1, 1], dtype="Int64", index=["A", "B"], name="a")
    # 断言：验证实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)

    # 实际结果：从DataFrame中按照位置取出第一个行
    result = df.iloc[0]
    # 断言：验证实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)

    # 当列类型混合时，返回对象Series
    df = DataFrame(
        {"A": pd.array([1, 2], dtype="Int64"), "B": np.array(["a", "b"])},
        index=["a", "b"],
    )
    # 实际结果：从DataFrame中取出索引为'a'的行
    result = df.loc["a"]
    # 期望结果：预期得到一个包含对象类型数据的Series
    expected = Series([1, "a"], dtype=object, index=["A", "B"], name="a")
    # 断言：验证实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)

    # 实际结果：从DataFrame中按照位置取出第一个行
    result = df.iloc[0]
    # 断言：验证实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, keys",
    [(Series([10]), (0, 0)), (Series([1, 2, 3], index=list("abc")), (0, 1))],
)
def test_ser_tup_indexer_exceeds_dimensions(ser, keys, indexer_li):
    # GH#13831：测试当索引器超过序列维度时应抛出异常
    exp_err, exp_msg = IndexingError, "Too many indexers"
    # 断言：确保抛出预期异常和消息
    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys]

    if indexer_li == tm.iloc:
        # 对于 iloc.__setitem__，让 numpy 处理错误报告。
        exp_err, exp_msg = IndexError, "too many indices for array"

    # 断言：确保抛出预期异常和消息
    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys] = 0


def test_ser_list_indexer_exceeds_dimensions(indexer_li):
    # GH#13831：确保当使用元组索引器超过序列维度时抛出异常，但使用列表时不抛出异常。
    ser = Series([10])
    # 实际结果：使用索引器获取两次索引为0的元素
    res = indexer_li(ser)[[0, 0]]
    # 期望结果：预期得到一个包含两个索引为0的元素的Series
    exp = Series([10, 10], index=Index([0, 0]))
    # 断言：验证实际结果与期望结果是否相等
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize(
    "value", [(0, 1), [0, 1], np.array([0, 1]), array.array("b", [0, 1])]
)
def test_scalar_setitem_with_nested_value(value):
    # 对于数值数据，我们尝试展开并因长度不匹配而引发异常
    df = DataFrame({"A": [1, 2, 3]})
    msg = "|".join(
        [
            "Must have equal len keys and value",
            "setting an array element with a sequence",
        ]
    )
    # 断言：确保在设置DataFrame元素时抛出预期异常和消息
    with pytest.raises(ValueError, match=msg):
        df.loc[0, "B"] = value

    # TODO 对于对象类型数据，也会出现类似情况，但我们是否应该保留嵌套数据并设置为这种形式？
    # 创建一个 Pandas DataFrame 对象，包含两列："A" 和 "B"
    df = DataFrame({"A": [1, 2, 3], "B": np.array([1, "a", "b"], dtype=object)})
    
    # 使用 pytest 检查是否会抛出 ValueError 异常，并验证异常信息中包含指定的匹配字符串
    with pytest.raises(ValueError, match="Must have equal len keys and value"):
        # 尝试修改 DataFrame 中第一行第二列 "B" 的值为变量 value
        df.loc[0, "B"] = value
        
    # 如果 value 是 numpy 数组
    # assert 语句用于检查条件是否为真，如果条件为假，则抛出 AssertionError 异常
    # 如果条件为真，程序继续执行
    if isinstance(value, np.ndarray):
        # 检查 DataFrame 中第一行第二列 "B" 的值是否与 value 中所有元素相等
        assert (df.loc[0, "B"] == value).all()
    else:
        # 检查 DataFrame 中第一行第二列 "B" 的值是否等于 value
        assert df.loc[0, "B"] == value
# 使用 pytest 的参数化装饰器 @pytest.mark.parametrize，为函数 test_scalar_setitem_series_with_nested_value 添加多组参数进行测试
@pytest.mark.parametrize(
    "value", [(0, 1), [0, 1], np.array([0, 1]), array.array("b", [0, 1])]
)
# 定义测试函数 test_scalar_setitem_series_with_nested_value，接受参数 value 和 indexer_sli
def test_scalar_setitem_series_with_nested_value(value, indexer_sli):
    # 对于数值数据，尝试解包并且在长度不匹配时抛出异常
    ser = Series([1, 2, 3])
    # 使用 pytest 的上下文管理器检查是否抛出 ValueError 异常，并匹配指定的错误信息
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        indexer_sli(ser)[0] = value

    # 对于对象类型的数据，保留嵌套数据并进行设置
    ser = Series([1, "a", "b"], dtype=object)
    indexer_sli(ser)[0] = value
    # 如果 value 是 numpy 数组，则断言序列的第一个元素与 value 相等
    if isinstance(value, np.ndarray):
        assert (ser.loc[0] == value).all()
    else:
        assert ser.loc[0] == value


# 使用 pytest 的参数化装饰器 @pytest.mark.parametrize，为函数 test_scalar_setitem_with_nested_value_length1 添加多组参数进行测试
@pytest.mark.parametrize(
    "value", [(0.0,), [0.0], np.array([0.0]), array.array("d", [0.0])]
)
# 定义测试函数 test_scalar_setitem_with_nested_value_length1，接受参数 value
def test_scalar_setitem_with_nested_value_length1(value):
    # https://github.com/pandas-dev/pandas/issues/46268

    # 对于数值数据，将长度为 1 的数组分配给标量位置时会被解包
    df = DataFrame({"A": [1, 2, 3]})
    df.loc[0, "B"] = value
    expected = DataFrame({"A": [1, 2, 3], "B": [0.0, np.nan, np.nan]})
    # 使用 pandas 的 tm.assert_frame_equal 函数断言 DataFrame df 与预期的 expected 是否相等
    tm.assert_frame_equal(df, expected)

    # 对于对象类型的数据，保留嵌套数据
    df = DataFrame({"A": [1, 2, 3], "B": np.array([1, "a", "b"], dtype=object)})
    df.loc[0, "B"] = value
    # 如果 value 是 numpy 数组，则断言 DataFrame 中指定位置的值与 value 相等
    if isinstance(value, np.ndarray):
        assert (df.loc[0, "B"] == value).all()
    else:
        assert df.loc[0, "B"] == value


# 使用 pytest 的参数化装饰器 @pytest.mark.parametrize，为函数 test_scalar_setitem_series_with_nested_value_length1 添加多组参数进行测试
@pytest.mark.parametrize(
    "value", [(0.0,), [0.0], np.array([0.0]), array.array("d", [0.0])]
)
# 定义测试函数 test_scalar_setitem_series_with_nested_value_length1，接受参数 value 和 indexer_sli
def test_scalar_setitem_series_with_nested_value_length1(value, indexer_sli):
    # 对于数值数据，将长度为 1 的数组分配给标量位置时会被解包
    # TODO this only happens in case of ndarray, should we make this consistent
    # for all list-likes? (as happens for DataFrame.(i)loc, see test above)
    ser = Series([1.0, 2.0, 3.0])
    # 如果 value 是 numpy 数组，则将其分配给序列的第一个元素
    if isinstance(value, np.ndarray):
        indexer_sli(ser)[0] = value
        expected = Series([0.0, 2.0, 3.0])
        # 使用 pandas 的 tm.assert_series_equal 函数断言序列 ser 与预期的 expected 是否相等
        tm.assert_series_equal(ser, expected)
    else:
        # 如果 value 不是 numpy 数组，则期望抛出 ValueError 异常，并匹配指定的错误信息
        with pytest.raises(
            ValueError, match="setting an array element with a sequence"
        ):
            indexer_sli(ser)[0] = value

    # 对于对象类型的数据，保留嵌套数据
    ser = Series([1, "a", "b"], dtype=object)
    indexer_sli(ser)[0] = value
    # 如果 value 是 numpy 数组，则断言序列的第一个元素与 value 相等
    if isinstance(value, np.ndarray):
        assert (ser.loc[0] == value).all()
    else:
        assert ser.loc[0] == value


# 定义测试函数 test_object_dtype_series_set_series_element，测试设置 Series 元素为 Series 类型的值
def test_object_dtype_series_set_series_element():
    # GH 48933
    # 创建一个对象类型的 Series，指定索引为 ["a", "b"]
    s1 = Series(dtype="O", index=["a", "b"])

    # 将空的 Series 分配给 s1 的索引为 "a" 的位置
    s1["a"] = Series()
    # 使用 pandas 的 tm.assert_series_equal 函数断言 s1 中索引为 "a" 的值与空的 Series 是否相等
    tm.assert_series_equal(s1.loc["a"], Series())

    # 将空的 Series 分配给 s1 的索引为 "b" 的位置
    s1.loc["b"] = Series()
    # 使用 pandas 的 tm.assert_series_equal 函数断言 s1 中索引为 "b" 的值与空的 Series 是否相等

    # 创建一个对象类型的 Series，指定索引为 ["a", "b"]
    s2 = Series(dtype="O", index=["a", "b"])

    # 将空的 Series 分配给 s2 的索引为 1 的位置（"b"）
    s2.iloc[1] = Series()
    # 使用 pandas 的 tm.assert_series_equal 函数断言 s2 中索引为 1 的值与空的 Series 是否相等
    tm.assert_series_equal(s2.iloc[1], Series())
```