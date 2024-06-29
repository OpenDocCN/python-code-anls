# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_indexing.py`

```
from datetime import timedelta  # 导入 timedelta 类用于处理时间间隔
import re  # 导入 re 模块用于正则表达式操作

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库

from pandas._libs import index as libindex  # 导入 pandas 库中的索引模块
from pandas.errors import InvalidIndexError  # 导入 pandas 中的异常类

import pandas as pd  # 导入 pandas 库并简称为 pd
from pandas import (  # 从 pandas 中导入多个对象
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    date_range,
)
import pandas._testing as tm  # 导入 pandas 测试模块

class TestSliceLocs:
    def test_slice_locs_partial(self, idx):
        sorted_idx, _ = idx.sortlevel(0)  # 对传入的 idx 对象按第一级别进行排序

        result = sorted_idx.slice_locs(("foo", "two"), ("qux", "one"))  # 计算指定边界的切片位置
        assert result == (1, 5)  # 断言切片位置是否符合预期

        result = sorted_idx.slice_locs(None, ("qux", "one"))  # 计算从开始到指定边界的切片位置
        assert result == (0, 5)  # 断言切片位置是否符合预期

        result = sorted_idx.slice_locs(("foo", "two"), None)  # 计算从指定边界到末尾的切片位置
        assert result == (1, len(sorted_idx))  # 断言切片位置是否符合预期

        result = sorted_idx.slice_locs("bar", "baz")  # 计算指定边界的切片位置
        assert result == (2, 4)  # 断言切片位置是否符合预期

    def test_slice_locs(self):
        df = DataFrame(  # 创建 DataFrame 对象
            np.random.default_rng(2).standard_normal((50, 4)),  # 使用随机数填充数据
            columns=Index(list("ABCD"), dtype=object),  # 设置列索引
            index=date_range("2000-01-01", periods=50, freq="B"),  # 设置时间索引
        )
        stacked = df.stack()  # 将 DataFrame 堆叠成 Series
        idx = stacked.index  # 获取堆叠后的 Series 的索引

        slob = slice(*idx.slice_locs(df.index[5], df.index[15]))  # 根据时间索引切片范围
        sliced = stacked[slob]  # 对堆叠后的 Series 进行切片操作
        expected = df[5:16].stack()  # 期望的切片结果
        tm.assert_almost_equal(sliced.values, expected.values)  # 断言切片后的值与期望值接近

        slob = slice(  # 创建切片对象
            *idx.slice_locs(
                df.index[5] + timedelta(seconds=30),  # 起始时间加上30秒
                df.index[15] - timedelta(seconds=30),  # 结束时间减去30秒
            )
        )
        sliced = stacked[slob]  # 对堆叠后的 Series 进行切片操作
        expected = df[6:15].stack()  # 期望的切片结果
        tm.assert_almost_equal(sliced.values, expected.values)  # 断言切片后的值与期望值接近

    def test_slice_locs_with_type_mismatch(self):
        df = DataFrame(  # 创建 DataFrame 对象
            np.random.default_rng(2).standard_normal((10, 4)),  # 使用随机数填充数据
            columns=Index(list("ABCD"), dtype=object),  # 设置列索引
            index=date_range("2000-01-01", periods=10, freq="B"),  # 设置时间索引
        )
        stacked = df.stack()  # 将 DataFrame 堆叠成 Series
        idx = stacked.index  # 获取堆叠后的 Series 的索引
        with pytest.raises(TypeError, match="^Level type mismatch"):  # 捕获预期的类型不匹配异常
            idx.slice_locs((1, 3))  # 尝试计算不匹配类型的切片位置
        with pytest.raises(TypeError, match="^Level type mismatch"):  # 捕获预期的类型不匹配异常
            idx.slice_locs(df.index[5] + timedelta(seconds=30), (5, 2))  # 尝试计算不匹配类型的切片位置
        df = DataFrame(  # 创建新的 DataFrame 对象
            np.ones((5, 5)),  # 使用值为1的数据填充
            index=Index([f"i-{i}" for i in range(5)], name="a"),  # 设置索引为字符串
            columns=Index([f"i-{i}" for i in range(5)], name="a"),  # 设置列名为字符串
        )
        stacked = df.stack()  # 将 DataFrame 堆叠成 Series
        idx = stacked.index  # 获取堆叠后的 Series 的索引
        with pytest.raises(TypeError, match="^Level type mismatch"):  # 捕获预期的类型不匹配异常
            idx.slice_locs(timedelta(seconds=30))  # 尝试计算不匹配类型的切片位置
        # TODO: Try creating a UnicodeDecodeError in exception message
        with pytest.raises(TypeError, match="^Level type mismatch"):  # 捕获预期的类型不匹配异常
            idx.slice_locs(df.index[1], (16, "a"))  # 尝试计算不匹配类型的切片位置
    # 定义测试方法，验证未排序的 MultiIndex 的 slice_locs 方法行为
    def test_slice_locs_not_sorted(self):
        # 创建一个 MultiIndex 对象，包含三个层级，每个层级有不同的索引值和代码
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        # 定义匹配错误消息的正则表达式，捕获 KeyError 异常
        msg = "[Kk]ey length.*greater than MultiIndex lexsort depth"
        # 断言在调用 slice_locs 方法时会抛出 KeyError 异常，并匹配错误消息
        with pytest.raises(KeyError, match=msg):
            index.slice_locs((1, 0, 1), (2, 1, 0))

        # 调用 sortlevel 方法，对 MultiIndex 进行排序，返回排序后的索引和一个不使用的变量
        sorted_index, _ = index.sortlevel(0)
        # 调用 slice_locs 方法，验证其功能正常
        sorted_index.slice_locs((1, 0, 1), (2, 1, 0))

    # 定义测试方法，验证 MultiIndex 的 slice_locs 方法在索引不包含的情况下的行为
    def test_slice_locs_not_contained(self):
        # 创建 MultiIndex 对象，包含两个层级，每个层级有不同的级别和代码
        index = MultiIndex(
            levels=[[0, 2, 4, 6], [0, 2, 4]],
            codes=[[0, 0, 0, 1, 1, 2, 3, 3, 3], [0, 1, 2, 1, 2, 2, 0, 1, 2]],
        )

        # 调用 slice_locs 方法，验证指定索引范围的结果是否符合预期
        result = index.slice_locs((1, 0), (5, 2))
        assert result == (3, 6)

        result = index.slice_locs(1, 5)
        assert result == (3, 6)

        result = index.slice_locs((2, 2), (5, 2))
        assert result == (3, 6)

        result = index.slice_locs(2, 5)
        assert result == (3, 6)

        result = index.slice_locs((1, 0), (6, 3))
        assert result == (3, 8)

        result = index.slice_locs(-1, 10)
        assert result == (0, len(index))

    # 使用 pytest 的参数化装饰器，定义多个测试用例，验证 MultiIndex 的 slice_locs 方法处理缺失值的行为
    @pytest.mark.parametrize(
        "index_arr,expected,start_idx,end_idx",
        [
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, None),
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, "b"),
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, ("b", "e")),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), None),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), "c"),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), ("c", "e")),
        ],
    )
    # 定义测试方法，验证 MultiIndex 的 slice_locs 方法在包含缺失值的情况下的行为
    def test_slice_locs_with_missing_value(
        self, index_arr, expected, start_idx, end_idx
    ):
        # 创建 MultiIndex 对象，从给定的索引数组创建
        idx = MultiIndex.from_arrays(index_arr)
        # 调用 slice_locs 方法，验证其处理缺失值的结果是否符合预期
        result = idx.slice_locs(start=start_idx, end=end_idx)
        assert result == expected
class TestPutmask:
    def test_putmask_with_wrong_mask(self, idx):
        # GH18368

        # 错误消息定义
        msg = "putmask: mask and data must be the same size"
        # 测试用例：当索引 idx 的长度比掩码长度多1时，应引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) + 1, np.bool_), 1)

        # 测试用例：当索引 idx 的长度比掩码长度少1时，应引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) - 1, np.bool_), 1)

        # 测试用例：当掩码不是布尔数组时，应引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            idx.putmask("foo", 1)

    def test_putmask_multiindex_other(self):
        # GH#43212 `value` is also a MultiIndex

        # 创建左侧的 MultiIndex 对象
        left = MultiIndex.from_tuples([(np.nan, 6), (np.nan, 6), ("a", 4)])
        # 创建右侧的 MultiIndex 对象
        right = MultiIndex.from_tuples([("a", 1), ("a", 1), ("d", 1)])
        # 创建布尔掩码数组
        mask = np.array([True, True, False])

        # 对左侧 MultiIndex 对象应用掩码操作，结果保存在 result 中
        result = left.putmask(mask, right)

        # 期望的结果 MultiIndex 对象
        expected = MultiIndex.from_tuples([right[0], right[1], left[2]])
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype(self, any_numeric_ea_dtype):
        # GH#49830

        # 创建具有指定数据类型的 MultiIndex 对象
        midx = MultiIndex.from_arrays(
            [pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]]
        )
        # 创建另一个具有相同数据类型的 MultiIndex 对象
        midx2 = MultiIndex.from_arrays(
            [pd.Series([5, 6, 7], dtype=any_numeric_ea_dtype), [-1, -2, -3]]
        )
        # 对 midx 应用掩码操作，结果保存在 result 中
        result = midx.putmask([True, False, False], midx2)
        # 期望的结果 MultiIndex 对象
        expected = MultiIndex.from_arrays(
            [pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]]
        )
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype_shorter_value(self, any_numeric_ea_dtype):
        # GH#49830

        # 创建具有指定数据类型的 MultiIndex 对象
        midx = MultiIndex.from_arrays(
            [pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]]
        )
        # 创建具有相同数据类型的较短 MultiIndex 对象
        midx2 = MultiIndex.from_arrays(
            [pd.Series([5], dtype=any_numeric_ea_dtype), [-1]]
        )
        # 对 midx 应用掩码操作，结果保存在 result 中
        result = midx.putmask([True, False, False], midx2)
        # 期望的结果 MultiIndex 对象
        expected = MultiIndex.from_arrays(
            [pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]]
        )
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)
    ````
        # 定义一个测试方法，用于测试 MultiIndex 对象的 get_indexer 方法
        def test_get_indexer(self):
            # 创建一个主轴索引，包含从 0 到 3 的整数
            major_axis = Index(np.arange(4))
            # 创建一个次轴索引，包含从 0 到 1 的整数
            minor_axis = Index(np.arange(2))
    
            # 创建一个包含主轴编码的 NumPy 数组
            major_codes = np.array([0, 0, 1, 2, 2, 3, 3], dtype=np.intp)
            # 创建一个包含次轴编码的 NumPy 数组
            minor_codes = np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.intp)
    
            # 使用主轴索引和次轴索引创建一个 MultiIndex 对象
            index = MultiIndex(
                levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
            )
            # 使用切片方式创建 idx1，包含 index 的前 5 个元素
            idx1 = index[:5]
            # 使用列表索引方式创建 idx2，包含 index 中的第 1、3、5 个元素
            idx2 = index[[1, 3, 5]]
    
            # 使用 idx2 中的元素获取 idx1 的索引，结果存储在 r1 中
            r1 = idx1.get_indexer(idx2)
            # 断言 r1 与预期结果 [1, 3, -1] 几乎相等
            tm.assert_almost_equal(r1, np.array([1, 3, -1], dtype=np.intp))
    
            # 使用 idx1 获取 idx2 的索引，采用填充方法 "pad"，结果存储在 r1 中
            r1 = idx2.get_indexer(idx1, method="pad")
            # 创建预期结果数组 e1
            e1 = np.array([-1, 0, 0, 1, 1], dtype=np.intp)
            # 断言 r1 与预期结果 e1 几乎相等
            tm.assert_almost_equal(r1, e1)
    
            # 使用 idx2 获取 idx1 逆序的索引，采用填充方法 "pad"，结果存储在 r2 中
            r2 = idx2.get_indexer(idx1[::-1], method="pad")
            # 断言 r2 与预期结果 e1 逆序相等
            tm.assert_almost_equal(r2, e1[::-1])
    
            # 使用 idx2 获取 idx1 的索引，采用填充方法 "ffill"，结果存储在 rffill1 中
            rffill1 = idx2.get_indexer(idx1, method="ffill")
            # 断言 r1 与 rffill1 几乎相等
            tm.assert_almost_equal(r1, rffill1)
    
            # 使用 idx2 获取 idx1 的索引，采用填充方法 "backfill"，结果存储在 r1 中
            r1 = idx2.get_indexer(idx1, method="backfill")
            # 创建预期结果数组 e1
            e1 = np.array([0, 0, 1, 1, 2], dtype=np.intp)
            # 断言 r1 与预期结果 e1 几乎相等
            tm.assert_almost_equal(r1, e1)
    
            # 使用 idx2 获取 idx1 逆序的索引，采用填充方法 "backfill"，结果存储在 r2 中
            r2 = idx2.get_indexer(idx1[::-1], method="backfill")
            # 断言 r2 与预期结果 e1 逆序相等
            tm.assert_almost_equal(r2, e1[::-1])
    
            # 使用 idx2 获取 idx1 的索引，采用填充方法 "bfill"，结果存储在 rbfill1 中
            rbfill1 = idx2.get_indexer(idx1, method="bfill")
            # 断言 r1 与 rbfill1 几乎相等
            tm.assert_almost_equal(r1, rbfill1)
    
            # 对非 MultiIndex 对象（idx2.values）使用 idx1 的 get_indexer 方法，结果存储在 r1 中
            r1 = idx1.get_indexer(idx2.values)
            # 获取 idx1 对 idx2 的预期结果
            rexp1 = idx1.get_indexer(idx2)
            # 断言 r1 与 rexp1 几乎相等
            tm.assert_almost_equal(r1, rexp1)
    
            # 使用 idx1 获取列表 [1, 2, 3] 的索引，结果存储在 r1 中
            r1 = idx1.get_indexer([1, 2, 3])
            # 断言 r1 的每个元素为 [-1, -1, -1]
            assert (r1 == [-1, -1, -1]).all()
    
            # 创建包含重复值的 Index 对象 idx1 和 idx2
            idx1 = Index(list(range(10)) + list(range(10)))
            idx2 = Index(list(range(20)))
    
            # 创建错误消息
            msg = "Reindexing only valid with uniquely valued Index objects"
            # 断言使用 idx2 对 idx1 的索引会抛出 InvalidIndexError 异常，且异常消息匹配 msg
            with pytest.raises(InvalidIndexError, match=msg):
                idx1.get_indexer(idx2)
    
        # 定义一个测试方法，用于测试 MultiIndex 对象的 get_indexer 方法的 nearest 方法
        def test_get_indexer_nearest(self):
            # 创建一个 MultiIndex 对象 midx
            midx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
            # 创建错误消息
            msg = (
                "method='nearest' not implemented yet for MultiIndex; "
                "see GitHub issue 9365"
            )
            # 断言调用 midx 的 get_indexer 方法，使用方法 "nearest" 会抛出 NotImplementedError 异常，且异常消息匹配 msg
            with pytest.raises(NotImplementedError, match=msg):
                midx.get_indexer(["a"], method="nearest")
            
            # 创建错误消息
            msg = "tolerance not implemented yet for MultiIndex"
            # 断言调用 midx 的 get_indexer 方法，使用方法 "pad" 和 tolerance=2 会抛出 NotImplementedError 异常，且异常消息匹配 msg
            with pytest.raises(NotImplementedError, match=msg):
                midx.get_indexer(["a"], method="pad", tolerance=2)
    
        # 定义一个测试方法，用于测试 MultiIndex 对象的 get_indexer 方法的 categorical 和 time 情况
        def test_get_indexer_categorical_time(self):
            # 创建一个 MultiIndex 对象 midx，包含分类和时间序列的组合
            midx = MultiIndex.from_product(
                [
                    Categorical(["a", "b", "c"]),
                    Categorical(date_range("2012-01-01", periods=3, freq="h")),
                ]
            )
            # 获取 midx 对 midx 的索引，结果存储在 result 中
            result = midx.get_indexer(midx)
            # 断言 result 与从 0 到 8 的整数数组相等
            tm.assert_numpy_array_equal(result, np.arange(9, dtype=np.intp))
    @pytest.mark.parametrize(
        "index_arr,labels,expected",
        [  # 使用 pytest 的 parametrize 装饰器，定义测试参数化输入
            (
                [[1, np.nan, 2], [3, 4, 5]],  # 第一个测试参数：二维数组作为索引数组
                [1, np.nan, 2],  # 第一个测试参数：标签数组
                np.array([-1, -1, -1], dtype=np.intp),  # 第一个测试参数：期望输出结果
            ),
            ([[1, np.nan, 2], [3, 4, 5]], [(np.nan, 4)], np.array([1], dtype=np.intp)),
            ([[1, 2, 3], [np.nan, 4, 5]], [(1, np.nan)], np.array([0], dtype=np.intp)),
            (
                [[1, 2, 3], [np.nan, 4, 5]],  # 最后一个测试参数：二维数组作为索引数组
                [np.nan, 4, 5],  # 最后一个测试参数：标签数组
                np.array([-1, -1, -1], dtype=np.intp),  # 最后一个测试参数：期望输出结果
            ),
        ],
    )
    def test_get_indexer_with_missing_value(self, index_arr, labels, expected):
        # issue 19132
        # 测试用例名称和编号，解决缺失值问题
        idx = MultiIndex.from_arrays(index_arr)  # 使用 index_arr 创建 MultiIndex 对象
        result = idx.get_indexer(labels)  # 调用 MultiIndex 对象的 get_indexer 方法
        tm.assert_numpy_array_equal(result, expected)  # 使用 tm.assert_numpy_array_equal 检查结果是否符合预期

    def test_get_indexer_methods(self):
        # https://github.com/pandas-dev/pandas/issues/29896
        # 测试不同方法获取索引器的行为是否正确
        # 确保在多级索引中，使用不同的填充方法（无填充、反向填充、前向填充）获取索引器的行为正确
        #
        # 可视化显示的 MultiIndex 如下：
        # mult_idx_1:
        #  0: -1 0
        #  1:    2
        #  2:    3
        #  3:    4
        #  4:  0 0
        #  5:    2
        #  6:    3
        #  7:    4
        #  8:  1 0
        #  9:    2
        # 10:    3
        # 11:    4
        #
        # mult_idx_2:
        #  0: 0 1
        #  1:   3
        #  2:   4
        mult_idx_1 = MultiIndex.from_product([[-1, 0, 1], [0, 2, 3, 4]])  # 创建第一个 MultiIndex 对象
        mult_idx_2 = MultiIndex.from_product([[0], [1, 3, 4]])  # 创建第二个 MultiIndex 对象

        indexer = mult_idx_1.get_indexer(mult_idx_2)  # 获取第二个 MultiIndex 在第一个 MultiIndex 中的索引器
        expected = np.array([-1, 6, 7], dtype=indexer.dtype)  # 预期的索引器结果
        tm.assert_almost_equal(expected, indexer)  # 使用 tm.assert_almost_equal 检查结果是否接近预期

        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="backfill")  # 使用反向填充方法获取索引器
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)  # 预期的反向填充索引器结果
        tm.assert_almost_equal(expected, backfill_indexer)  # 使用 tm.assert_almost_equal 检查结果是否接近预期

        # 确保 legacy "bfill" 选项与 "backfill" 功能完全相同
        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="bfill")
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="pad")  # 使用前向填充方法获取索引器
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)  # 预期的前向填充索引器结果
        tm.assert_almost_equal(expected, pad_indexer)  # 使用 tm.assert_almost_equal 检查结果是否接近预期

        # 确保 legacy "ffill" 选项与 "pad" 功能完全相同
        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="ffill")
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)
    @pytest.mark.parametrize("method", ["pad", "ffill", "backfill", "bfill", "nearest"])
    def test_get_indexer_methods_raise_for_non_monotonic(self, method):
        # 使用 pytest.mark.parametrize 装饰器为 test_get_indexer_methods_raise_for_non_monotonic 方法参数化，测试不同的 method 值
        # method 参数可以取 "pad", "ffill", "backfill", "bfill", "nearest" 中的一个
        # 53452
        # 为测试标识符，可能是一个内部参考编号或者问题编号，用于跟踪问题或变更
        mi = MultiIndex.from_arrays([[0, 4, 2], [0, 4, 2]])
        # 创建一个 MultiIndex 对象 mi，从给定的数组列表创建，数组包含两个子数组作为层级索引
        if method == "nearest":
            # 如果 method 参数为 "nearest"，则抛出 NotImplementedError 异常
            err = NotImplementedError
            # 错误类型设为 NotImplementedError
            msg = "not implemented yet for MultiIndex"
            # 异常消息为 "not implemented yet for MultiIndex"
        else:
            # 如果 method 参数不为 "nearest"，则抛出 ValueError 异常
            err = ValueError
            # 错误类型设为 ValueError
            msg = "index must be monotonic increasing or decreasing"
            # 异常消息为 "index must be monotonic increasing or decreasing"
        with pytest.raises(err, match=msg):
            # 使用 pytest.raises 检查是否抛出特定类型和消息的异常
            mi.get_indexer([(1, 1)], method=method)
            # 调用 MultiIndex 对象的 get_indexer 方法，测试抛出异常的情况

    def test_get_indexer_crossing_levels(self):
        # 测试 MultiIndex 的 get_indexer 方法在处理多层索引时的交叉情况
        # https://github.com/pandas-dev/pandas/issues/29896
        # 参考 GitHub 上的具体问题或讨论，说明这个测试场景
        # tests a corner case with get_indexer() with MultiIndexes where, when we
        # need to "carry" across levels, proper tuple ordering is respected
        #
        # the MultiIndexes used in this test, visually, are:
        #  0: 1 1 1 1
        #  1:       2
        #  2:     2 1
        #  3:       2
        #  4: 1 2 1 1
        #  5:       2
        #  6:     2 1
        #  7:       2
        #  8: 2 1 1 1
        #  9:       2
        # 10:     2 1
        # 11:       2
        # 12: 2 2 1 1
        # 13:       2
        # 14:     2 1
        # 15:       2
        mult_idx_1 = MultiIndex.from_product([[1, 2]] * 4)
        # 创建一个 MultiIndex 对象 mult_idx_1，从多个子数组的笛卡尔积中生成，每个子数组包含 [1, 2]
        mult_idx_2 = MultiIndex.from_tuples([(1, 3, 2, 2), (2, 3, 2, 2)])
        # 创建另一个 MultiIndex 对象 mult_idx_2，从给定的元组列表中生成，每个元组代表一个层级索引

        # show the tuple orderings, which get_indexer() should respect
        # 断言检查 get_indexer 方法在处理时是否遵循正确的元组顺序

        assert mult_idx_1[7] < mult_idx_2[0] < mult_idx_1[8]
        # 检查索引位置 7 的 mult_idx_1 小于 mult_idx_2 的索引位置 0，且 mult_idx_2 的索引位置 0 小于 mult_idx_1 的索引位置 8
        assert mult_idx_1[-1] < mult_idx_2[1]
        # 检查 mult_idx_1 的最后一个索引小于 mult_idx_2 的索引位置 1

        indexer = mult_idx_1.get_indexer(mult_idx_2)
        # 调用 mult_idx_1 的 get_indexer 方法，返回 mult_idx_2 在 mult_idx_1 中的索引
        expected = np.array([-1, -1], dtype=indexer.dtype)
        # 期望的索引结果数组，初始化为 -1，与 indexer 的数据类型一致
        tm.assert_almost_equal(expected, indexer)
        # 使用测试工具 assert_almost_equal 检查 indexer 是否与期望的结果数组 expected 几乎相等

        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="bfill")
        # 使用 "bfill" 方法调用 mult_idx_1 的 get_indexer 方法，返回 mult_idx_2 在 mult_idx_1 中的后向填充索引
        expected = np.array([8, -1], dtype=backfill_indexer.dtype)
        # 期望的后向填充索引结果数组，第一个索引为 8，第二个索引为 -1，与 backfill_indexer 的数据类型一致
        tm.assert_almost_equal(expected, backfill_indexer)
        # 使用测试工具 assert_almost_equal 检查 backfill_indexer 是否与期望的结果数组 expected 几乎相等

        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="ffill")
        # 使用 "ffill" 方法调用 mult_idx_1 的 get_indexer 方法，返回 mult_idx_2 在 mult_idx_1 中的前向填充索引
        expected = np.array([7, 15], dtype=pad_indexer.dtype)
        # 期望的前向填充索引结果数组，第一个索引为 7，第二个索引为 15，与 pad_indexer 的数据类型一致
        tm.assert_almost_equal(expected, pad_indexer)
        # 使用测试工具 assert_almost_equal 检查 pad_indexer 是否与期望的结果数组 expected 几乎相等

    def test_get_indexer_kwarg_validation(self):
        # GH#41918
        # 参考 GitHub 上的具体问题或讨论，说明这个测试场景
        mi = MultiIndex.from_product([range(3), ["A", "B"]])
        # 创建一个 MultiIndex 对象 mi，从两个迭代器的笛卡尔积中生成，第一个迭代器是 range(3)，第二个迭代器是 ["A", "B"]

        msg = "limit argument only valid if doing pad, backfill or nearest"
        # 错误消息，限制参数仅在执行 pad、backfill 或 nearest 时有效
        with pytest.raises(ValueError, match=msg):
            # 使用 pytest.raises 检查是否抛出特定类型和消息的异常
            mi.get_indexer(mi[:-1], limit=4)
            # 调用 MultiIndex 对象的 get_indexer 方法，测试抛出异常的情况，传递了不适用的 limit 参数

        msg = "tolerance argument only valid if doing pad, backfill or nearest"
        # 错误消息，容差参数仅在执行 pad、backfill 或 nearest 时有效
        with pytest.raises(ValueError, match=msg):
            # 使用 pytest.raises 检查是否抛出特定类型和消息的异常
            mi.get_indexer(mi[:-1], tolerance="piano")
            # 调用 MultiIndex 对象的 get_indexer 方法，测试抛出异常的情况，传递了不适用的 tolerance 参数
    # 定义一个测试方法，用于测试处理 NaN 值索引的情况
    def test_get_indexer_nan(self):
        # GH#37222 表示这段代码关联到GitHub上的 issue #37222
        # 创建一个多级索引 idx1，包含两级，第一级为 ["A"]，第二级为 [1.0, 2.0]，并分别命名为 "id1" 和 "id2"
        idx1 = MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"])
        # 创建另一个多级索引 idx2，与 idx1 结构相似，但第二级包含了 NaN 值，第一级仍为 ["A"]，命名为 "id1" 和 "id2"
        idx2 = MultiIndex.from_product([["A"], [np.nan, 2.0]], names=["id1", "id2"])
        # 预期的结果是一个包含两个元素的 NumPy 数组，内容为 [-1, 1]
        expected = np.array([-1, 1])
        # 使用 idx1 在 idx2 上调用 get_indexer 方法，将结果存储在 result 变量中
        result = idx2.get_indexer(idx1)
        # 使用测试工具函数 tm.assert_numpy_array_equal 检查 result 是否等于预期结果 expected，忽略数据类型检查
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)
        # 使用 idx2 在 idx1 上调用 get_indexer 方法，将结果再次存储在 result 变量中
        result = idx1.get_indexer(idx2)
        # 再次使用测试工具函数检查 result 是否等于预期结果 expected
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)
def test_getitem(idx):
    # scalar，测试索引器对标量的操作
    assert idx[2] == ("bar", "one")

    # slice，测试索引器对切片的操作
    result = idx[2:5]
    expected = idx[[2, 3, 4]]
    assert result.equals(expected)

    # boolean，测试索引器对布尔数组的操作
    result = idx[[True, False, True, False, True, True]]
    result2 = idx[np.array([True, False, True, False, True, True])]
    expected = idx[[0, 2, 4, 5]]
    assert result.equals(expected)
    assert result2.equals(expected)


def test_getitem_group_select(idx):
    # 对索引对象进行排序，并获取特定元素的位置
    sorted_idx, _ = idx.sortlevel(0)
    assert sorted_idx.get_loc("baz") == slice(3, 4)
    assert sorted_idx.get_loc("foo") == slice(0, 2)


@pytest.mark.parametrize("box", [list, Index])
def test_getitem_bool_index_all(box):
    # GH#22533，测试索引器对全为True或全为False的布尔数组的操作
    ind1 = box([True] * 5)
    idx = MultiIndex.from_tuples([(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)])
    tm.assert_index_equal(idx[ind1], idx)

    ind2 = box([True, False, True, False, False])
    expected = MultiIndex.from_tuples([(10, 1), (30, 3)])
    tm.assert_index_equal(idx[ind2], expected)


@pytest.mark.parametrize("box", [list, Index])
def test_getitem_bool_index_single(box):
    # GH#22533，测试索引器对单个True或单个False的布尔数组的操作
    ind1 = box([True])
    idx = MultiIndex.from_tuples([(10, 1)])
    tm.assert_index_equal(idx[ind1], idx)

    ind2 = box([False])
    expected = MultiIndex(
        levels=[np.array([], dtype=np.int64), np.array([], dtype=np.int64)],
        codes=[[], []],
    )
    tm.assert_index_equal(idx[ind2], expected)


class TestGetLoc:
    def test_get_loc(self, idx):
        # 测试获取指定元素的位置
        assert idx.get_loc(("foo", "two")) == 1
        assert idx.get_loc(("baz", "two")) == 3
        with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
            idx.get_loc(("bar", "two"))
        with pytest.raises(KeyError, match=r"^'quux'$"):
            idx.get_loc("quux")

        # 3 levels，测试多层索引对象的位置获取
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        with pytest.raises(KeyError, match=r"^\(1, 1\)$"):
            index.get_loc((1, 1))
        assert index.get_loc((2, 0)) == slice(3, 5)

    def test_get_loc_duplicates(self):
        # 测试处理重复值的情况下获取位置
        index = Index([2, 2, 2, 2])
        result = index.get_loc(2)
        expected = slice(0, 4)
        assert result == expected

        index = Index(["c", "a", "a", "b", "b"])
        rs = index.get_loc("c")
        xp = 0
        assert rs == xp

        with pytest.raises(KeyError, match="2"):
            index.get_loc(2)
    # 定义测试方法，用于测试 MultiIndex 的 get_loc_level 方法
    def test_get_loc_level(self):
        # 创建一个 MultiIndex 对象，包含三个层级，每个层级是一个 Index 对象
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            # 指定每个层级的编码，用来指定层级中各元素的位置
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        # 调用 MultiIndex 的 get_loc_level 方法，获取指定标签的位置和新的索引
        loc, new_index = index.get_loc_level((0, 1))
        # 预期的结果是一个切片对象，表示位置 1 到 2（不包含）
        expected = slice(1, 2)
        # 根据预期的切片对象，获取对应的索引，并移除两个层级的标签
        exp_index = index[expected].droplevel(0).droplevel(0)
        # 断言获取的位置和新索引是否符合预期
        assert loc == expected
        assert new_index.equals(exp_index)

        # 重复上述步骤，测试不同的标签组合
        loc, new_index = index.get_loc_level((0, 1, 0))
        # 期望的位置是 1
        expected = 1
        assert loc == expected
        assert new_index is None

        # 使用 pytest 检查是否会引发 KeyError 异常，且异常消息与给定正则表达式匹配
        with pytest.raises(KeyError, match=r"^\(2, 2\)$"):
            index.get_loc_level((2, 2))
        
        # GH 22221: 未使用的标签
        with pytest.raises(KeyError, match=r"^2$"):
            # 删除标签为 2 的层级，然后调用 get_loc_level 方法
            index.drop(2).get_loc_level(2)
        
        # 在未排序的层级上使用未使用的标签：
        with pytest.raises(KeyError, match=r"^2$"):
            # 在层级为 2 的情况下，删除标签为 1 的元素，然后调用 get_loc_level 方法
            index.drop(1, level=2).get_loc_level(2, level=2)

        # 创建另一个 MultiIndex 对象，包含两个层级，第一个层级是一个列表，第二个层级是一个范围列表
        index = MultiIndex(
            levels=[[2000], list(range(4))],
            codes=[np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3])],
        )
        # 调用 MultiIndex 的 get_loc_level 方法，使用标签 (2000, slice(None, None)) 获取位置和新索引
        result, new_index = index.get_loc_level((2000, slice(None, None)))
        # 期望的结果是一个全范围的切片对象
        expected = slice(None, None)
        # 断言获取的结果和新索引是否符合预期，新索引移除第一个层级的标签
        assert result == expected
        assert new_index.equals(index.droplevel(0))

    # 使用 pytest 的参数化标记，测试多个数据类型的情况
    @pytest.mark.parametrize("dtype1", [int, float, bool, str])
    @pytest.mark.parametrize("dtype2", [int, float, bool, str])
    def test_get_loc_multiple_dtypes(self, dtype1, dtype2):
        # GH 18520
        # 创建包含两个层级的列表，每个层级使用不同的数据类型
        levels = [np.array([0, 1]).astype(dtype1), np.array([0, 1]).astype(dtype2)]
        # 使用 from_product 方法创建 MultiIndex 对象
        idx = MultiIndex.from_product(levels)
        # 断言获取 idx[2] 的位置是否等于 2
        assert idx.get_loc(idx[2]) == 2

    # 使用 pytest 的参数化标记，测试隐式类型转换的情况
    @pytest.mark.parametrize("level", [0, 1])
    @pytest.mark.parametrize("dtypes", [[int, float], [float, int]])
    def test_get_loc_implicit_cast(self, level, dtypes):
        # GH 18818, GH 15994 : as flat index, cast int to float and vice-versa
        # 创建包含两个层级的列表，每个层级都是包含字符串 "a" 和 "b" 的列表
        levels = [["a", "b"], ["c", "d"]]
        # 指定要查找的关键字
        key = ["b", "d"]
        # 获取层级和关键字的数据类型
        lev_dtype, key_dtype = dtypes
        # 将层级中第 level 层的数据类型转换为 lev_dtype
        levels[level] = np.array([0, 1], dtype=lev_dtype)
        # 将关键字中第 level 位置的元素转换为 key_dtype 类型的数据
        key[level] = key_dtype(1)
        # 使用 from_product 方法创建 MultiIndex 对象
        idx = MultiIndex.from_product(levels)
        # 断言获取关键字 key 的位置是否等于 3
        assert idx.get_loc(tuple(key)) == 3

    # 使用 pytest 的参数化标记，测试不同的数据类型情况
    @pytest.mark.parametrize("dtype", [bool, object])
    def test_get_loc_cast_bool(self, dtype):
        # GH 19086 : int is casted to bool, but not vice-versa (for object dtype)
        # 对于对象类型，将整数强制转换为布尔值，但反之不转换。
        # 当 dtype 是 bool 类型时，不做任何方向的强制转换。
        levels = [Index([False, True], dtype=dtype), np.arange(2, dtype="int64")]
        # 创建一个多级索引对象 idx，由 levels 中的元素构成
        idx = MultiIndex.from_product(levels)

        if dtype is bool:
            # 断言在布尔类型的情况下，获取 (0, 1) 应抛出 KeyError 异常，匹配指定的正则表达式
            with pytest.raises(KeyError, match=r"^\(0, 1\)$"):
                assert idx.get_loc((0, 1)) == 1
            # 断言在布尔类型的情况下，获取 (1, 0) 应抛出 KeyError 异常，匹配指定的正则表达式
            with pytest.raises(KeyError, match=r"^\(1, 0\)$"):
                assert idx.get_loc((1, 0)) == 2
        else:
            # 使用 Python 对象比较，其中 0 == False，1 == True
            assert idx.get_loc((0, 1)) == 1
            assert idx.get_loc((1, 0)) == 2

        # 断言获取 (False, True) 应抛出 KeyError 异常，匹配指定的正则表达式
        with pytest.raises(KeyError, match=r"^\(False, True\)$"):
            idx.get_loc((False, True))
        # 断言获取 (True, False) 应抛出 KeyError 异常，匹配指定的正则表达式
        with pytest.raises(KeyError, match=r"^\(True, False\)$"):
            idx.get_loc((True, False))

    @pytest.mark.parametrize("level", [0, 1])
    def test_get_loc_nan(self, level, nulls_fixture):
        # GH 18485 : NaN in MultiIndex
        # 创建一个多级索引对象 idx，其中包含 NaN 值
        levels = [["a", "b"], ["c", "d"]]
        key = ["b", "d"]
        # 将 levels[level] 和 key[level] 替换为 nulls_fixture 的类型
        levels[level] = np.array([0, nulls_fixture], dtype=type(nulls_fixture))
        key[level] = nulls_fixture
        idx = MultiIndex.from_product(levels)
        # 断言获取 tuple(key) 对应的位置为 3
        assert idx.get_loc(tuple(key)) == 3

    def test_get_loc_missing_nan(self):
        # GH 8569
        # 创建一个多级索引对象 idx，其中包含指定数组
        idx = MultiIndex.from_arrays([[1.0, 2.0], [3.0, 4.0]])
        # 断言获取 1 的位置是一个 slice 对象
        assert isinstance(idx.get_loc(1), slice)
        # 断言获取 3 应抛出 KeyError 异常，匹配指定的正则表达式
        with pytest.raises(KeyError, match=r"^3$"):
            idx.get_loc(3)
        # 断言获取 np.nan 应抛出 KeyError 异常，匹配指定的正则表达式
        with pytest.raises(KeyError, match=r"^nan$"):
            idx.get_loc(np.nan)
        # 断言获取 [np.nan] 应抛出 InvalidIndexError 异常，匹配指定的正则表达式
        with pytest.raises(InvalidIndexError, match=r"\[nan\]"):
            # 当列表类/非可散列类型引发 TypeError
            idx.get_loc([np.nan])

    def test_get_loc_with_values_including_missing_values(self):
        # issue 19132
        # 创建一个多级索引对象 idx，其中包含 np.nan 值
        idx = MultiIndex.from_product([[np.nan, 1]] * 2)
        # 断言获取 np.nan 的位置应为预期的切片对象
        expected = slice(0, 2, None)
        assert idx.get_loc(np.nan) == expected

        idx = MultiIndex.from_arrays([[np.nan, 1, 2, np.nan]])
        # 断言获取 np.nan 的位置应为预期的布尔数组
        expected = np.array([True, False, False, True])
        tm.assert_numpy_array_equal(idx.get_loc(np.nan), expected)

        idx = MultiIndex.from_product([[np.nan, 1]] * 3)
        # 断言获取 (np.nan, 1) 的位置应为预期的切片对象
        expected = slice(2, 4, None)
        assert idx.get_loc((np.nan, 1)) == expected

    def test_get_loc_duplicates2(self):
        # TODO: de-duplicate with test_get_loc_duplicates above?
        # 创建一个多级索引对象 index，指定其层级和编码
        index = MultiIndex(
            levels=[["D", "B", "C"], [0, 26, 27, 37, 57, 67, 75, 82]],
            codes=[[0, 0, 0, 1, 2, 2, 2, 2, 2, 2], [1, 3, 4, 6, 0, 2, 2, 3, 5, 7]],
            names=["tag", "day"],
        )

        # 断言获取 "D" 的位置为预期的切片对象
        assert index.get_loc("D") == slice(0, 3)
    # 测试函数：检查在超过字典排序深度时获取位置的行为
    def test_get_loc_past_lexsort_depth(self, performance_warning):
        # 创建一个多级索引对象 idx
        idx = MultiIndex(
            levels=[["a"], [0, 7], [1]],
            codes=[[0, 0], [1, 0], [0, 0]],
            names=["x", "y", "z"],
            sortorder=0,
        )
        # 设置要查找的键值对 key
        key = ("a", 7)

        # 使用断言确保产生性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行 get_loc 方法来获取 key 的位置
            # PerformanceWarning: indexing past lexsort depth may impact performance
            result = idx.get_loc(key)

        # 断言获取的位置结果是一个切片对象，表示找到了匹配项
        assert result == slice(0, 1, None)

    # 测试函数：测试 MultiIndex 对象中使用空列表作为参数时是否能引发异常
    def test_multiindex_get_loc_list_raises(self):
        # 创建一个 MultiIndex 对象 idx，从元组列表创建
        idx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        # 定义匹配的错误消息正则表达式
        msg = r"\[\]"

        # 使用 pytest.raises 检查是否引发了 InvalidIndexError 异常，并匹配指定的错误消息
        with pytest.raises(InvalidIndexError, match=msg):
            # 调用 get_loc 方法，传入空列表作为参数
            idx.get_loc([])

    # 测试函数：测试在 MultiIndex 对象中使用嵌套元组作为键时是否能引发 KeyError 异常
    def test_get_loc_nested_tuple_raises_keyerror(self):
        # 从产品集合创建 MultiIndex 对象 mi
        mi = MultiIndex.from_product([range(3), range(4), range(5), range(6)])
        # 定义一个嵌套的元组作为要查找的键 key
        key = ((2, 3, 4), "foo")

        # 使用 pytest.raises 检查是否引发了 KeyError 异常，并匹配包含键信息的错误消息
        with pytest.raises(KeyError, match=re.escape(str(key))):
            # 调用 get_loc 方法，传入嵌套元组作为参数
            mi.get_loc(key)
class TestWhere:
    # 测试 where 方法在 MultiIndex 上的行为
    def test_where(self):
        # 创建一个 MultiIndex 对象，包含两个元组
        i = MultiIndex.from_tuples([("A", 1), ("A", 2)])

        # 准备一个异常消息，用于匹配 pytest.raises 中的异常
        msg = r"\.where is not supported for MultiIndex operations"
        # 使用 pytest 的异常断言，验证 where 方法在 MultiIndex 上抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=msg):
            i.where(True)

    # 测试 where 方法在 MultiIndex 上处理 array-like 对象的行为
    def test_where_array_like(self, listlike_box):
        # 创建一个 MultiIndex 对象，包含两个元组
        mi = MultiIndex.from_tuples([("A", 1), ("A", 2)])
        # 准备一个条件列表
        cond = [False, True]
        # 准备一个异常消息，用于匹配 pytest.raises 中的异常
        msg = r"\.where is not supported for MultiIndex operations"
        # 使用 pytest 的异常断言，验证 where 方法在 MultiIndex 上抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=msg):
            # 调用 where 方法，传入 array-like 对象作为条件
            mi.where(listlike_box(cond))


class TestContains:
    # 测试 MultiIndex 对象的 contains 方法在顶层元素上的行为
    def test_contains_top_level(self):
        # 创建一个 MultiIndex 对象，包含两层索引，每层两个元素
        midx = MultiIndex.from_product([["A", "B"], [1, 2]])
        # 验证元素 "A" 在 MultiIndex 中
        assert "A" in midx
        # 验证元素 "A" 不在 MultiIndex 的 _engine 属性中
        assert "A" not in midx._engine

    # 测试包含 NaT 的 MultiIndex 对象的 contains 方法的行为
    def test_contains_with_nat(self):
        # 创建一个包含 NaT 的 MultiIndex 对象
        mi = MultiIndex(
            levels=[["C"], date_range("2012-01-01", periods=5)],
            codes=[[0, 0, 0, 0, 0, 0], [-1, 0, 1, 2, 3, 4]],
            names=[None, "B"],
        )
        # 验证特定元组是否在 MultiIndex 中
        assert ("C", pd.Timestamp("2012-01-01")) in mi
        # 遍历 MultiIndex 的所有值，验证每个值是否在 MultiIndex 中
        for val in mi.values:
            assert val in mi

    # 测试 MultiIndex 对象的 contains 方法在一般情况下的行为
    def test_contains(self, idx):
        # 验证特定元组是否在 MultiIndex 中
        assert ("foo", "two") in idx
        # 验证特定元组是否不在 MultiIndex 中
        assert ("bar", "two") not in idx
        # 验证 None 不在 MultiIndex 中
        assert None not in idx

    # 测试包含缺失值的 MultiIndex 对象的 contains 方法的行为
    def test_contains_with_missing_value(self):
        # 创建包含 NaN 的 MultiIndex 对象
        idx = MultiIndex.from_arrays([[1, np.nan, 2]])
        # 验证 NaN 在 MultiIndex 中
        assert np.nan in idx

        # 创建包含 NaN 的 MultiIndex 对象
        idx = MultiIndex.from_arrays([[1, 2], [np.nan, 3]])
        # 验证 NaN 不在 MultiIndex 中
        assert np.nan not in idx
        # 验证特定元组是否在 MultiIndex 中
        assert (1, np.nan) in idx

    # 测试 dropped 状态的 MultiIndex 对象的 contains 方法的行为
    def test_multiindex_contains_dropped(self):
        # 创建一个 MultiIndex 对象，包含两个维度的笛卡尔积
        idx = MultiIndex.from_product([[1, 2], [3, 4]])
        # 验证特定值是否在 MultiIndex 中
        assert 2 in idx
        # 对 MultiIndex 对象调用 drop 方法，删除特定值
        idx = idx.drop(2)

        # 验证被删除的值依然存在于 MultiIndex 的 levels 属性中
        assert 2 in idx.levels[0]
        # 验证被删除的值不再存在于 MultiIndex 中
        assert 2 not in idx

        # 验证字符串值是否在 MultiIndex 中
        idx = MultiIndex.from_product([["a", "b"], ["c", "d"]])
        assert "a" in idx
        # 对 MultiIndex 对象调用 drop 方法，删除特定字符串值
        idx = idx.drop("a")
        # 验证被删除的字符串值依然存在于 MultiIndex 的 levels 属性中
        assert "a" in idx.levels[0]
        # 验证被删除的字符串值不再存在于 MultiIndex 中
        assert "a" not in idx

    # 测试包含 timedelta64 类型的 MultiIndex 对象的 contains 方法的行为
    def test_contains_td64_level(self):
        # 创建一个 timedelta 的时间范围
        tx = pd.timedelta_range("09:30:00", "16:00:00", freq="30 min")
        # 创建一个 MultiIndex 对象，包含时间范围和索引的数组
        idx = MultiIndex.from_arrays([tx, np.arange(len(tx))])
        # 验证特定时间在 MultiIndex 中
        assert tx[0] in idx
        # 验证特定字符串不在 MultiIndex 中
        assert "element_not_exit" not in idx
        # 验证特定时间字符串在 MultiIndex 中
        assert "0 day 09:30:00" in idx

    # 测试包含大规模 MultiIndex 对象的 contains 方法的行为
    def test_large_mi_contains(self, monkeypatch):
        # 设置 libindex 模块的 _SIZE_CUTOFF 属性
        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 10)
            # 创建一个 MultiIndex 对象，包含两个数组
            result = MultiIndex.from_arrays([range(10), range(10)])
            # 验证特定元组不在 MultiIndex 中
            assert (10, 0) not in result
    # 创建一个多级索引对象，其中包含以下几个级别的数据：
    # - 第一个级别是从 "2019-01-01T00:15:33" 开始的时间序列，每小时一个时间点，命名为 "date"
    # - 第二个级别只包含一个值 "x"
    # - 第三个级别只包含一个值 3
    idx = MultiIndex.from_product(
        [
            date_range("2019-01-01T00:15:33", periods=100, freq="h", name="date"),
            ["x"],
            [3],
        ]
    )
    
    # 根据上面创建的多级索引 idx 和一个包含递增整数的数组，创建一个 DataFrame 对象
    df = DataFrame({"foo": np.arange(len(idx))}, idx)
    
    # 从 DataFrame df 中选择指定的数据子集：
    # - 使用 pd.IndexSlice 对象指定选择条件：
    #   - 第一个级别时间从 "2019-1-2" 开始到最后一个时间点
    #   - 第二个级别为 "x"
    #   - 第三个级别为任意值
    # - 最后选择列 "foo"
    result = df.loc[pd.IndexSlice["2019-1-2":, "x", :], "foo"]
    
    # 创建另一个多级索引对象，其中包含以下几个级别的数据：
    # - 第一个级别从 "2019-01-02T00:15:33" 开始到 "2019-01-05T03:15:33" 结束的时间序列，每小时一个时间点，命名为 "date"
    # - 第二个级别只包含一个值 "x"
    # - 第三个级别只包含一个值 3
    qidx = MultiIndex.from_product(
        [
            date_range(
                start="2019-01-02T00:15:33",
                end="2019-01-05T03:15:33",
                freq="h",
                name="date",
            ),
            ["x"],
            [3],
        ]
    )
    
    # 创建一个预期的 Series 对象，数据为递增整数，索引为 qidx，名称为 "foo"
    should_be = pd.Series(data=np.arange(24, len(qidx) + 24), index=qidx, name="foo")
    
    # 使用 assert_series_equal 函数比较 result 和 should_be 两个 Series 对象是否相等
    tm.assert_series_equal(result, should_be)
@pytest.mark.parametrize(
    "index_arr,expected,target,algo",
    [
        # 参数化测试用例：测试不同输入情况下的期望结果
        ([[np.nan, "a", "b"], ["c", "d", "e"]], 0, np.nan, "left"),  # 第一个测试用例
        ([[np.nan, "a", "b"], ["c", "d", "e"]], 1, (np.nan, "c"), "right"),  # 第二个测试用例
        ([["a", "b", "c"], ["d", np.nan, "d"]], 1, ("b", np.nan), "left"),  # 第三个测试用例
    ],
)
def test_get_slice_bound_with_missing_value(index_arr, expected, target, algo):
    # issue 19132
    # 创建 MultiIndex 对象，用给定的 index_arr 数组作为数据源
    idx = MultiIndex.from_arrays(index_arr)
    # 调用被测试的方法，获取切片边界的结果
    result = idx.get_slice_bound(target, side=algo)
    # 断言获取的结果与期望值一致
    assert result == expected


@pytest.mark.parametrize(
    "index_arr,expected,start_idx,end_idx",
    [
        # 参数化测试用例：测试不同输入情况下的期望结果
        ([[np.nan, 1, 2], [3, 4, 5]], slice(0, 2, None), np.nan, 1),  # 第一个测试用例
        ([[np.nan, 1, 2], [3, 4, 5]], slice(0, 3, None), np.nan, (2, 5)),  # 第二个测试用例
        ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), 3),  # 第三个测试用例
        ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), (3, 5)),  # 第四个测试用例
    ],
)
def test_slice_indexer_with_missing_value(index_arr, expected, start_idx, end_idx):
    # issue 19132
    # 创建 MultiIndex 对象，用给定的 index_arr 数组作为数据源
    idx = MultiIndex.from_arrays(index_arr)
    # 调用被测试的方法，获取切片索引器的结果
    result = idx.slice_indexer(start=start_idx, end=end_idx)
    # 断言获取的结果与期望值一致
    assert result == expected


@pytest.mark.parametrize(
    "N, expected_dtype",
    [
        # 参数化测试用例：测试不同输入情况下的期望结果
        (1, "uint8"),   # 当 N=1 时的预期数据类型
        (2, "uint16"),  # 当 N=2 时的预期数据类型
        (4, "uint32"),  # 当 N=4 时的预期数据类型
        (8, "uint64"),  # 当 N=8 时的预期数据类型
        (10, "object"), # 当 N=10 时的预期数据类型
    ],
)
def test_pyint_engine(N, expected_dtype):
    # GH#18519 : when combinations of codes cannot be represented in 64
    # bits, the index underlying the MultiIndex engine works with Python
    # integers, rather than uint64.
    # 根据不同的 N 值构造不同的 keys 列表，用于测试 MultiIndex 对象的引擎值的数据类型
    keys = [
        tuple(arr)
        for arr in [
            [0] * 4 * N,
            [1] * 4 * N,
            [np.nan] * N + [0] * 3 * N,
            [0] * N + [1] * 3 * N,
            [np.nan] * N + [1] * 2 * N + [0] * N,
        ]
    ]
    # 创建 MultiIndex 对象，用 keys 列表作为数据源
    index = MultiIndex.from_tuples(keys)
    # 断言 MultiIndex 对象的引擎值的数据类型与期望的数据类型一致
    assert index._engine.values.dtype == expected_dtype

    # 对每个 key_value 进行测试
    for idx, key_value in enumerate(keys):
        # 断言获取的 key_value 在 MultiIndex 对象中的位置索引等于 idx
        assert index.get_loc(key_value) == idx

        # 生成一个数组，包含从 0 到 idx 的整数值，作为预期的结果
        expected = np.arange(idx + 1, dtype=np.intp)
        # 获取 key_value 列表的索引器结果
        result = index.get_indexer([keys[i] for i in expected])
        # 使用断言函数检查 result 和 expected 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 对于缺失的 key_value，预期结果中的第一个索引为 -1，其余依次为 0 到 len(keys)-1
    idces = range(len(keys))
    expected = np.array([-1] + list(idces), dtype=np.intp)
    # 创建一个缺失的 key_value
    missing = tuple([0, 1, 0, 1] * N)
    # 获取 key_value 列表的索引器结果
    result = index.get_indexer([missing] + [keys[i] for i in idces])
    # 使用断言函数检查 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)
    "keys,expected",
    [
        # 第一组测试用例：slice(None) 表示取所有元素，[5, 4] 表示选择索引为 5 和 4 的元素，期望输出为 [1, 0]
        ((slice(None), [5, 4]), [1, 0]),
        # 第二组测试用例：slice(None) 表示取所有元素，[4, 5] 表示选择索引为 4 和 5 的元素，期望输出为 [0, 1]
        ((slice(None), [4, 5]), [0, 1]),
        # 第三组测试用例：[True, False, True] 表示选择为 True 的元素对应的索引，[4, 6] 表示选择索引为 4 和 6 的元素，期望输出为 [0, 2]
        (([True, False, True], [4, 6]), [0, 2]),
        # 第四组测试用例：[True, False, True] 表示选择为 True 的元素对应的索引，[6, 4] 表示选择索引为 6 和 4 的元素，期望输出为 [0, 2]
        (([True, False, True], [6, 4]), [0, 2]),
        # 第五组测试用例：2 表示直接选择索引为 2 的元素，[4, 5] 表示选择索引为 4 和 5 的元素，期望输出为 [0, 1]
        ((2, [4, 5]), [0, 1]),
        # 第六组测试用例：2 表示直接选择索引为 2 的元素，[5, 4] 表示选择索引为 5 和 4 的元素，期望输出为 [1, 0]
        ((2, [5, 4]), [1, 0]),
        # 第七组测试用例：[2] 表示选择索引为 2 的元素，[4, 5] 表示选择索引为 4 和 5 的元素，期望输出为 [0, 1]
        (([2], [4, 5]), [0, 1]),
        # 第八组测试用例：[2] 表示选择索引为 2 的元素，[5, 4] 表示选择索引为 5 和 4 的元素，期望输出为 [1, 0]
        (([2], [5, 4]), [1, 0]),
    ],
def test_get_locs_reordering(keys, expected):
    # 测试函数用例，测试 MultiIndex 对象的 get_locs 方法重新排序功能
    # GH48384 是 GitHub 上的 issue 编号，可能是与此函数相关的问题追踪号
    # 创建一个 MultiIndex 对象，从多个数组中构建索引
    idx = MultiIndex.from_arrays(
        [
            [2, 2, 1],  # 第一个级别的索引值
            [4, 5, 6],  # 第二个级别的索引值
        ]
    )
    # 调用 MultiIndex 对象的 get_locs 方法，获取给定 keys 的位置索引
    result = idx.get_locs(keys)
    # 期望结果，将 expected 转换为 np.intp 类型的 numpy 数组
    expected = np.array(expected, dtype=np.intp)
    # 使用测试框架的函数验证 result 与 expected 是否相等
    tm.assert_numpy_array_equal(result, expected)


def test_get_indexer_for_multiindex_with_nans(nulls_fixture):
    # 测试函数用例，测试带有 NaN 值的 MultiIndex 对象的 get_indexer 方法
    # GH37222 是 GitHub 上的 issue 编号，可能是与此函数相关的问题追踪号
    # 创建两个 MultiIndex 对象，分别表示带有 NaN 值和不带 NaN 值的情况
    idx1 = MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"])
    idx2 = MultiIndex.from_product([["A"], [nulls_fixture, 2.0]], names=["id1", "id2"])

    # 调用 idx2 的 get_indexer 方法，获取 idx1 在 idx2 中的索引位置
    result = idx2.get_indexer(idx1)
    # 期望结果，将 [-1, 1] 转换为 np.intp 类型的 numpy 数组
    expected = np.array([-1, 1], dtype=np.intp)
    # 使用测试框架的函数验证 result 与 expected 是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 调用 idx1 的 get_indexer 方法，获取 idx2 在 idx1 中的索引位置
    result = idx1.get_indexer(idx2)
    # 期望结果，将 [-1, 1] 转换为 np.intp 类型的 numpy 数组
    expected = np.array([-1, 1], dtype=np.intp)
    # 使用测试框架的函数验证 result 与 expected 是否相等
    tm.assert_numpy_array_equal(result, expected)
```