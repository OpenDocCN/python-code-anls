# `D:\src\scipysrc\pandas\pandas\tests\test_sorting.py`

```
from collections import defaultdict  # 导入 defaultdict 类，用于创建默认字典
from datetime import datetime  # 导入 datetime 类，用于处理日期时间相关操作
from itertools import product  # 导入 product 函数，用于计算多个可迭代对象的笛卡尔积

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 导入 pandas 库中的多个模块和类
    NA,  # 导入 NA 常量，表示缺失值
    DataFrame,  # 导入 DataFrame 类，用于处理表格数据
    MultiIndex,  # 导入 MultiIndex 类，用于多级索引
    Series,  # 导入 Series 类，表示一维标记数组
    array,  # 导入 array 函数，用于创建数组
    concat,  # 导入 concat 函数，用于连接对象
    merge,  # 导入 merge 函数，用于数据库风格的合并操作
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块
from pandas.core.algorithms import safe_sort  # 导入安全排序算法
import pandas.core.common as com  # 导入 pandas 核心通用模块
from pandas.core.sorting import (  # 导入排序相关函数和类
    _decons_group_index,  # 导入 _decons_group_index 函数，用于解构组索引
    get_group_index,  # 导入 get_group_index 函数，用于获取组索引
    is_int64_overflow_possible,  # 导入 is_int64_overflow_possible 函数，用于检查是否可能发生 int64 溢出
    lexsort_indexer,  # 导入 lexsort_indexer 函数，用于多级索引的词典序排序
    nargsort,  # 导入 nargsort 函数，用于对数组进行排序
)


@pytest.fixture
def left_right():
    # 设置低端、高端和 n 的值，用于生成随机 DataFrame
    low, high, n = -1 << 10, 1 << 10, 1 << 20
    # 创建左侧 DataFrame，包含随机整数列和总和列
    left = DataFrame(
        np.random.default_rng(2).integers(low, high, (n, 7)), columns=list("ABCDEFG")
    )
    left["left"] = left.sum(axis=1)  # 计算各行总和，并添加到 DataFrame
    # 从左侧 DataFrame 中随机抽样生成右侧 DataFrame
    right = left.sample(
        frac=1, random_state=np.random.default_rng(2), ignore_index=True
    )
    right.columns = right.columns[:-1].tolist() + ["right"]  # 修改右侧 DataFrame 列名
    right["right"] *= -1  # 将右侧 DataFrame 中的 "right" 列乘以 -1
    return left, right  # 返回左右两个 DataFrame 对象


class TestSorting:
    @pytest.mark.slow
    def test_int64_overflow(self):
        # 构造用于测试的数组 A 和 B
        B = np.concatenate((np.arange(1000), np.arange(1000), np.arange(500)))
        A = np.arange(2500)
        # 创建包含随机数据和指定列的 DataFrame 对象 df
        df = DataFrame(
            {
                "A": A,
                "B": B,
                "C": A,
                "D": B,
                "E": A,
                "F": B,
                "G": A,
                "H": B,
                "values": np.random.default_rng(2).standard_normal(2500),
            }
        )

        lg = df.groupby(["A", "B", "C", "D", "E", "F", "G", "H"])  # 按指定列分组并计算总和
        rg = df.groupby(["H", "G", "F", "E", "D", "C", "B", "A"])  # 按逆序列分组并计算总和

        left = lg.sum()["values"]  # 对左侧分组后的数据进行求和
        right = rg.sum()["values"]  # 对右侧分组后的数据进行求和

        exp_index, _ = left.index.sortlevel()  # 排序左侧分组后的索引
        tm.assert_index_equal(left.index, exp_index)  # 断言左侧索引与预期索引相等

        exp_index, _ = right.index.sortlevel(0)  # 排序右侧分组后的索引
        tm.assert_index_equal(right.index, exp_index)  # 断言右侧索引与预期索引相等

        tups = list(map(tuple, df[["A", "B", "C", "D", "E", "F", "G", "H"]].values))  # 将指定列的值转换为元组列表
        tups = com.asarray_tuplesafe(tups)  # 将元组列表转换为安全的 NumPy 数组

        expected = df.groupby(tups).sum()["values"]  # 根据元组分组并计算总和

        for k, v in expected.items():
            assert left[k] == right[k[::-1]]  # 断言左侧与右侧对应位置的值相等
            assert left[k] == v  # 断言左侧对应位置的值与期望值相等
        assert len(left) == len(right)  # 断言左右两侧的长度相等

    def test_int64_overflow_groupby_large_range(self):
        # GH9096，测试较大范围的分组
        values = range(55109)  # 创建一个范围为 55109 的值列表
        data = DataFrame.from_dict({"a": values, "b": values, "c": values, "d": values})  # 根据值列表创建 DataFrame
        grouped = data.groupby(["a", "b", "c", "d"])  # 按指定列进行分组
        assert len(grouped) == len(values)  # 断言分组后的数量与值列表的长度相等

    @pytest.mark.parametrize("agg", ["mean", "median"])
    # 定义一个测试方法，用于测试在大型混洗数据帧上分组和聚合的情况
    def test_int64_overflow_groupby_large_df_shuffled(self, agg):
        # 使用 NumPy 的随机数生成器创建一个指定范围内的整数数组
        rs = np.random.default_rng(2)
        arr = rs.integers(-1 << 12, 1 << 12, (1 << 15, 5))
        # 从现有数组中随机选择索引，增加一些重复行
        i = rs.choice(len(arr), len(arr) * 4)
        arr = np.vstack((arr, arr[i]))

        # 对数组的行进行随机排列（洗牌）
        i = rs.permutation(len(arr))
        arr = arr[i]

        # 创建一个 Pandas 数据帧，列名为 'abcde'，并添加两列 'jim' 和 'joe'
        df = DataFrame(arr, columns=list("abcde"))
        df["jim"], df["joe"] = np.zeros((2, len(df)))

        # 按列 'abcde' 分组数据帧
        gr = df.groupby(list("abcde"))

        # 断言检查是否存在可能的 int64 溢出情况
        assert is_int64_overflow_possible(
            tuple(ping.ngroups for ping in gr._grouper.groupings)
        )

        # 使用唯一的数组拆分为五部分，创建一个多级索引
        mi = MultiIndex.from_arrays(
            [ar.ravel() for ar in np.array_split(np.unique(arr, axis=0), 5, axis=1)],
            names=list("abcde"),
        )

        # 创建一个指定索引和列的数据帧，索引按顺序排序
        res = DataFrame(
            np.zeros((len(mi), 2)), columns=["jim", "joe"], index=mi
        ).sort_index()

        # 断言检查使用指定聚合函数后的数据帧是否与预期结果相等
        tm.assert_frame_equal(getattr(gr, agg)(), res)

    @pytest.mark.parametrize(
        "order, na_position, exp",
        [
            # 参数化测试用例1：升序排序，缺失值置于最后，期望结果 exp
            [
                True,
                "last",
                list(range(5, 105)) + list(range(5)) + list(range(105, 110)),
            ],
            # 参数化测试用例2：升序排序，缺失值置于最前，期望结果 exp
            [
                True,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(5, 105)),
            ],
            # 参数化测试用例3：降序排序，缺失值置于最后，期望结果 exp
            [
                False,
                "last",
                list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110)),
            ],
            # 参数化测试用例4：降序排序，缺失值置于最前，期望结果 exp
            [
                False,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1)),
            ],
        ],
    )
    # 定义一个测试方法，测试 lexsort_indexer 函数的排序功能
    def test_lexsort_indexer(self, order, na_position, exp):
        keys = [[np.nan] * 5 + list(range(100)) + [np.nan] * 5]
        # 调用 lexsort_indexer 函数，得到排序后的索引结果
        result = lexsort_indexer(keys, orders=order, na_position=na_position)
        # 断言检查排序后的结果是否与期望结果 exp 相等
        tm.assert_numpy_array_equal(result, np.array(exp, dtype=np.intp))

    @pytest.mark.parametrize(
        "ascending, na_position, exp",
        [
            # 参数化测试用例1：升序排序，缺失值置于最后，期望结果 exp
            [
                True,
                "last",
                list(range(5, 105)) + list(range(5)) + list(range(105, 110)),
            ],
            # 参数化测试用例2：升序排序，缺失值置于最前，期望结果 exp
            [
                True,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(5, 105)),
            ],
            # 参数化测试用例3：降序排序，缺失值置于最后，期望结果 exp
            [
                False,
                "last",
                list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110)),
            ],
            # 参数化测试用例4：降序排序，缺失值置于最前，期望结果 exp
            [
                False,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1)),
            ],
        ],
    )
    def test_nargsort(self, ascending, na_position, exp):
        # 创建一个包含 NaN 值的对象数组，长度为 5，其后是一个包含 0 到 99 的整数序列，再接着长度为 5 的 NaN 值数组
        items = np.array([np.nan] * 5 + list(range(100)) + [np.nan] * 5, dtype="O")

        # mergesort 是最难正确实现的排序算法，因为我们希望它是稳定的。

        # 根据 numpy/core/tests/test_multiarray 中的注释，"""为了检查实际算法，排序的项目数必须大于 ~50，因为对于小数组，快速排序和归并排序会切换到插入排序。"""

        # 使用 nargsort 函数对 items 进行排序，指定排序算法为 mergesort，指定升序或降序（由参数 ascending 控制），并指定 NaN 的排列位置（由参数 na_position 控制）
        result = nargsort(
            items, kind="mergesort", ascending=ascending, na_position=na_position
        )
        # 使用 tm.assert_numpy_array_equal 检查排序结果是否与期望的 exp 数组相等（忽略数据类型）
        tm.assert_numpy_array_equal(result, np.array(exp), check_dtype=False)
class TestMerge:
    # 测试整数溢出外部合并情况
    def test_int64_overflow_outer_merge(self):
        # #2690, 组合爆炸
        # 创建一个包含随机标准正态分布数据的 DataFrame，1000行7列
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 7)),
            columns=list("ABCDEF") + ["G1"],
        )
        # 创建另一个包含随机标准正态分布数据的 DataFrame，1000行7列
        df2 = DataFrame(
            np.random.default_rng(3).standard_normal((1000, 7)),
            columns=list("ABCDEF") + ["G2"],
        )
        # 对 df1 和 df2 进行外部合并
        result = merge(df1, df2, how="outer")
        # 断言合并后的结果行数为 2000
        assert len(result) == 2000

    @pytest.mark.slow
    # 测试整数溢出检查列求和情况
    def test_int64_overflow_check_sum_col(self, left_right):
        left, right = left_right

        # 对 left 和 right 进行外部合并
        out = merge(left, right, how="outer")
        # 断言合并后的行数等于 left 的行数
        assert len(out) == len(left)
        # 断言 out 的 "left" 列等于 "right" 列的负数（忽略列名检查）
        tm.assert_series_equal(out["left"], -out["right"], check_names=False)
        # 计算 out 前两列之外的每行数据的和
        result = out.iloc[:, :-2].sum(axis=1)
        # 断言 out 的 "left" 列等于 result（忽略列名检查）
        tm.assert_series_equal(out["left"], result, check_names=False)
        # 断言 result 的列名为 None
        assert result.name is None

    @pytest.mark.slow
    # 测试整数溢出如何合并情况
    def test_int64_overflow_how_merge(self, left_right, join_type):
        left, right = left_right

        # 对 left 和 right 进行外部合并
        out = merge(left, right, how="outer")
        # 对合并后的结果按照列名排序
        out.sort_values(out.columns.tolist(), inplace=True)
        # 将索引设置为从 0 开始的连续整数
        out.index = np.arange(len(out))
        # 使用 tm.assert_frame_equal 断言 out 等于按照指定合并类型（sort=True）再次合并的结果
        tm.assert_frame_equal(out, merge(left, right, how=join_type, sort=True))

    @pytest.mark.slow
    # 测试整数溢出排序为假的顺序情况
    def test_int64_overflow_sort_false_order(self, left_right):
        left, right = left_right

        # 检查左连接时，sort=False 时保持左框架顺序的合并
        out = merge(left, right, how="left", sort=False)
        # 使用 tm.assert_frame_equal 断言 left 等于 out 的左框架列（按列名列表排序）
        tm.assert_frame_equal(left, out[left.columns.tolist()])

        # 检查右连接时，sort=False 时保持右框架顺序的合并
        out = merge(right, left, how="left", sort=False)
        # 使用 tm.assert_frame_equal 断言 right 等于 out 的右框架列（按列名列表排序）
        tm.assert_frame_equal(right, out[right.columns.tolist()])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "codes_list, shape",
        [
            [
                # 定义包含多个整数数组的列表，每个数组使用 tile 和 astype 生成
                [
                    np.tile([0, 1, 2, 3, 0, 1, 2, 3], 100).astype(np.int64),
                    np.tile([0, 2, 4, 3, 0, 1, 2, 3], 100).astype(np.int64),
                    np.tile([5, 1, 0, 2, 3, 0, 5, 4], 100).astype(np.int64),
                ],
                # 定义数组的形状
                (4, 5, 6),
            ],
            [
                [
                    # 使用 arange 生成的整数数组，并进行 tile 操作
                    np.tile(np.arange(10000, dtype=np.int64), 5),
                    np.tile(np.arange(10000, dtype=np.int64), 5),
                ],
                # 定义数组的形状
                (10000, 10000),
            ],
        ],
    )
    # 测试解构函数的情况
    def test_decons(codes_list, shape):
        # 获取组索引
        group_index = get_group_index(codes_list, shape, sort=True, xnull=True)
        # 使用组索引进行解构
        codes_list2 = _decons_group_index(group_index, shape)

        # 使用循环逐个断言 codes_list 和 codes_list2 中的数组相等
        for a, b in zip(codes_list, codes_list2):
            tm.assert_numpy_array_equal(a, b)


class TestSafeSort:
    @pytest.mark.parametrize(
        "arg, exp",
        [
            # 定义整数列表的参数化测试
            [[3, 1, 2, 0, 4], [0, 1, 2, 3, 4]],
            # 定义对象数组的参数化测试
            [
                np.array(list("baaacb"), dtype=object),
                np.array(list("aaabbc"), dtype=object),
            ],
            # 定义空列表的参数化测试
            [[], []],
        ],
    )
    # 定义一个测试函数，用于测试 safe_sort 函数的基本排序功能
    def test_basic_sort(self, arg, exp):
        # 调用 safe_sort 函数对传入的 numpy 数组进行安全排序
        result = safe_sort(np.array(arg))
        # 将期望结果转换为 numpy 数组
        expected = np.array(exp)
        # 使用 pytest 框架的 assert 函数检查计算结果与期望结果是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 使用 pytest.mark.parametrize 装饰器标记参数化测试，测试代码和期望结果
    @pytest.mark.parametrize("verify", [True, False])
    @pytest.mark.parametrize(
        "codes, exp_codes",
        [
            # 第一组参数化测试，输入为 [0, 1, 1, 2, 3, 0, -1, 4]，期望输出为 [3, 1, 1, 2, 0, 3, -1, 4]
            [[0, 1, 1, 2, 3, 0, -1, 4], [3, 1, 1, 2, 0, 3, -1, 4]],
            # 第二组参数化测试，空输入，期望输出也为空列表
            [[], []],
        ],
    )
    # 定义测试函数，测试 safe_sort 函数对特定 codes 的排序功能
    def test_codes(self, verify, codes, exp_codes):
        # 定义输入的 values 数组
        values = np.array([3, 1, 2, 0, 4])
        # 定义期望的 values 数组结果
        expected = np.array([0, 1, 2, 3, 4])

        # 调用 safe_sort 函数，传入 values 和 codes，同时开启 NA 哨兵标志和验证标志
        result, result_codes = safe_sort(
            values, codes, use_na_sentinel=True, verify=verify
        )
        # 将期望的 codes 转换为 numpy 数组
        expected_codes = np.array(exp_codes, dtype=np.intp)
        # 使用 pytest 框架的 assert 函数检查计算结果与期望结果是否相等
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    # 定义测试函数，测试 safe_sort 函数对超出界限的 codes 的处理功能
    def test_codes_out_of_bound(self):
        # 定义输入的 values 数组
        values = np.array([3, 1, 2, 0, 4])
        # 定义期望的 values 数组结果
        expected = np.array([0, 1, 2, 3, 4])

        # 定义超出界限的 codes 列表
        codes = [0, 101, 102, 2, 3, 0, 99, 4]
        # 调用 safe_sort 函数，传入 values 和 codes，同时开启 NA 哨兵标志
        result, result_codes = safe_sort(values, codes, use_na_sentinel=True)
        # 定义期望的 codes 数组结果，包含替代值 -1 表示超出界限的索引
        expected_codes = np.array([3, -1, -1, 2, 0, 3, -1, 4], dtype=np.intp)
        # 使用 pytest 框架的 assert 函数检查计算结果与期望结果是否相等
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    # 定义测试函数，测试 safe_sort 函数对混合整数和对象的排序功能
    def test_mixed_integer(self):
        # 定义混合类型的 values 数组
        values = np.array(["b", 1, 0, "a", 0, "b"], dtype=object)
        # 调用 safe_sort 函数对 values 进行排序
        result = safe_sort(values)
        # 定义期望的排序结果数组
        expected = np.array([0, 0, 1, "a", "b", "b"], dtype=object)
        # 使用 pytest 框架的 assert 函数检查计算结果与期望结果是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试函数，测试 safe_sort 函数对混合整数和 codes 的排序功能
    def test_mixed_integer_with_codes(self):
        # 定义混合类型的 values 数组
        values = np.array(["b", 1, 0, "a"], dtype=object)
        # 定义 codes 列表
        codes = [0, 1, 2, 3, 0, -1, 1]
        # 调用 safe_sort 函数，传入 values 和 codes
        result, result_codes = safe_sort(values, codes)
        # 定义期望的 values 和 codes 的排序结果数组
        expected = np.array([0, 1, "a", "b"], dtype=object)
        expected_codes = np.array([3, 1, 0, 2, 3, -1, 1], dtype=np.intp)
        # 使用 pytest 框架的 assert 函数检查计算结果与期望结果是否相等
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    # 定义测试函数，测试 safe_sort 函数对无法排序的数据类型抛出异常的功能
    def test_unsortable(self):
        # GH 13714
        # 定义无法排序的对象数组，包含 datetime 对象
        arr = np.array([1, 2, datetime.now(), 0, 3], dtype=object)
        # 定义预期的错误信息模式
        msg = "'[<>]' not supported between instances of .*"
        # 使用 pytest 框架的 assert 函数检查是否抛出预期的 TypeError 异常，并匹配错误信息模式
        with pytest.raises(TypeError, match=msg):
            safe_sort(arr)

    # 使用 pytest.mark.parametrize 装饰器标记参数化测试，测试 safe_sort 函数对异常输入的处理功能
    @pytest.mark.parametrize(
        "arg, codes, err, msg",
        [
            # 第一组参数化测试，输入为单个整数，期望抛出 TypeError 异常，错误信息为 "Only np.ndarray, ExtensionArray, and Index"
            [1, None, TypeError, "Only np.ndarray, ExtensionArray, and Index"],
            # 第二组参数化测试，codes 输入为整数，期望抛出 TypeError 异常，错误信息为 "Only list-like objects or None"
            [np.array([0, 1, 2]), 1, TypeError, "Only list-like objects or None"],
            # 第三组参数化测试，codes 中包含重复值，期望抛出 ValueError 异常，错误信息为 "values should be unique"
            [np.array([0, 1, 2, 1]), [0, 1], ValueError, "values should be unique"],
        ],
    )
    # 定义测试函数，测试 safe_sort 函数对异常输入的处理功能
    def test_exceptions(self, arg, codes, err, msg):
        # 使用 pytest 框架的 assert 函数检查是否抛出预期的异常类型，并匹配错误信息
        with pytest.raises(err, match=msg):
            safe_sort(values=arg, codes=codes)
    # 定义一个测试方法，用于测试扩展数组的排序功能
    def test_extension_array(self, arg, exp):
        # 使用给定的参数创建一个 Int64 类型的数组
        a = array(arg, dtype="Int64")
        # 调用 safe_sort 函数对数组进行安全排序，并将结果保存在 result 变量中
        result = safe_sort(a)
        # 创建一个预期的 Int64 类型数组
        expected = array(exp, dtype="Int64")
        # 使用 pytest 的断言方法，比较 result 和 expected 是否相等
        tm.assert_extension_array_equal(result, expected)

    # 使用 pytest 的参数化标记，定义另一个测试方法，用于测试带有编码的扩展数组
    @pytest.mark.parametrize("verify", [True, False])
    def test_extension_array_codes(self, verify):
        # 使用给定的参数创建一个 Int64 类型的数组
        a = array([1, 3, 2], dtype="Int64")
        # 调用 safe_sort 函数对数组进行安全排序，并将结果保存在 result 和 codes 变量中
        # 使用额外的参数指定排序顺序和 NA 哨兵的使用，并根据 verify 参数进行验证
        result, codes = safe_sort(a, [0, 1, -1, 2], use_na_sentinel=True, verify=verify)
        # 创建预期的值数组和编码数组
        expected_values = array([1, 2, 3], dtype="Int64")
        expected_codes = np.array([0, 2, -1, 1], dtype=np.intp)
        # 使用 pytest 的断言方法，比较 result 和 expected_values 是否相等
        tm.assert_extension_array_equal(result, expected_values)
        # 使用 pytest 的断言方法，比较 codes 和 expected_codes 是否相等
        tm.assert_numpy_array_equal(codes, expected_codes)
# 测试混合字符串和空值的情况，使用指定的空值数据进行测试
def test_mixed_str_null(nulls_fixture):
    # 创建包含字符串和空值的 NumPy 数组
    values = np.array(["b", nulls_fixture, "a", "b"], dtype=object)
    # 对数组进行安全排序，保留空值的位置
    result = safe_sort(values)
    # 创建预期的结果数组，保持空值的位置
    expected = np.array(["a", "b", "b", nulls_fixture], dtype=object)
    # 使用测试工具函数验证 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


# 测试安全排序在多级索引上的行为
def test_safe_sort_multiindex():
    # GH#48412 表示 GitHub 上的 issue 编号
    # 创建包含整数和空值的 Series 对象，数据类型为 'Int64'
    arr1 = Series([2, 1, NA, NA], dtype="Int64")
    # 创建普通的整数列表
    arr2 = [2, 1, 3, 3]
    # 创建多级索引对象，使用上述两个数组作为层级
    midx = MultiIndex.from_arrays([arr1, arr2])
    # 对多级索引对象进行安全排序
    result = safe_sort(midx)
    # 创建预期的多级索引对象，保留空值的位置
    expected = MultiIndex.from_arrays(
        [Series([1, 2, NA, NA], dtype="Int64"), [1, 2, 3, 3]]
    )
    # 使用测试工具函数验证两个多级索引对象是否相等
    tm.assert_index_equal(result, expected)
```