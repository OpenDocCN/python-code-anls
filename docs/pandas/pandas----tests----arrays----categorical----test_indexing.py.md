# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_indexing.py`

```
import math  # 导入math模块，提供数学函数支持

import numpy as np  # 导入numpy库，提供多维数组和矩阵计算功能
import pytest  # 导入pytest库，用于编写和执行测试用例

from pandas import (  # 从pandas库中导入以下子模块和类
    NA,  # 表示缺失值的常量
    Categorical,  # 用于处理分类数据的类
    CategoricalIndex,  # 分类索引对象
    Index,  # 一般性索引对象
    Interval,  # 表示时间间隔的类
    IntervalIndex,  # 时间间隔索引对象
    NaT,  # 表示缺失的时间戳值
    PeriodIndex,  # 表示时期的索引对象
    Series,  # 表示一维标签化数据结构
    Timedelta,  # 表示时间差的类
    Timestamp,  # 表示时间戳的类
)
import pandas._testing as tm  # 导入pandas的测试辅助工具
import pandas.core.common as com  # 导入pandas的核心通用工具

class TestCategoricalIndexingWithFactor:
    def test_getitem(self):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        assert factor[0] == "a"  # 断言第一个元素为'a'
        assert factor[-1] == "c"  # 断言最后一个元素为'c'

        subf = factor[[0, 1, 2]]  # 选择索引为0、1、2的子集
        tm.assert_numpy_array_equal(subf._codes, np.array([0, 1, 1], dtype=np.int8))  # 断言子集的编码与预期相等

        subf = factor[np.asarray(factor) == "c"]  # 根据条件选择元素为'c'的子集
        tm.assert_numpy_array_equal(subf._codes, np.array([2, 2, 2], dtype=np.int8))  # 断言子集的编码与预期相等

    def test_setitem(self):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        # int/positional
        c = factor.copy()
        c[0] = "b"  # 将第一个元素改为'b'
        assert c[0] == "b"  # 断言第一个元素为'b'
        c[-1] = "a"  # 将最后一个元素改为'a'
        assert c[-1] == "a"  # 断言最后一个元素为'a'

        # boolean
        c = factor.copy()
        indexer = np.zeros(len(c), dtype="bool")  # 创建布尔索引数组
        indexer[0] = True  # 设置第一个元素为True
        indexer[-1] = True  # 设置最后一个元素为True
        c[indexer] = "c"  # 根据布尔索引数组将选定位置元素改为'c'
        expected = Categorical(["c", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        tm.assert_categorical_equal(c, expected)  # 断言修改后的c与预期的expected相等

    @pytest.mark.parametrize("categories", [None, ["b", "a"]])
    def test_setitem_same_but_unordered(self, categories):
        # GH-24142
        other = Categorical(["b", "a"], categories=categories)  # 创建新的分类变量other
        target = Categorical(["a", "b"], categories=["a", "b"])  # 创建目标分类变量target
        mask = np.array([True, False])  # 创建布尔掩码数组
        target[mask] = other[mask]  # 将other中符合掩码的元素赋值给target相应位置
        expected = Categorical(["b", "b"], categories=["a", "b"])  # 预期的分类变量expected
        tm.assert_categorical_equal(target, expected)  # 断言target与expected相等

    @pytest.mark.parametrize(
        "other",
        [
            Categorical(["b", "a"], categories=["b", "a", "c"]),
            Categorical(["b", "a"], categories=["a", "b", "c"]),
            Categorical(["a", "a"], categories=["a"]),
            Categorical(["b", "b"], categories=["b"]),
        ],
    )
    def test_setitem_different_unordered_raises(self, other):
        # GH-24142
        target = Categorical(["a", "b"], categories=["a", "b"])  # 创建目标分类变量target
        mask = np.array([True, False])  # 创建布尔掩码数组
        msg = "Cannot set a Categorical with another, without identical categories"
        with pytest.raises(TypeError, match=msg):  # 断言抛出TypeError异常，且异常消息匹配msg
            target[mask] = other[mask]  # 尝试将other中符合掩码的元素赋值给target

    @pytest.mark.parametrize(
        "other",
        [
            Categorical(["b", "a"]),  # 创建新的分类变量other
            Categorical(["b", "a"], categories=["b", "a"], ordered=True),  # 创建新的有序分类变量other
            Categorical(["b", "a"], categories=["a", "b", "c"], ordered=True),  # 创建新的有序分类变量other
        ],
    )
    # 定义一个测试方法，用于验证设置相同顺序时引发异常的情况
    def test_setitem_same_ordered_raises(self, other):
        # GH-24142: GitHub issue编号，标识此测试案例
        # 创建一个有序的分类对象，包含字符串列表作为初始数据和类别，并标记为有序
        target = Categorical(["a", "b"], categories=["a", "b"], ordered=True)
        # 创建一个布尔数组作为掩码
        mask = np.array([True, False])
        # 当尝试用不同的分类对象的掩码项设置目标对象时，期望引发 TypeError 异常，异常消息为指定的信息
        msg = "Cannot set a Categorical with another, without identical categories"
        with pytest.raises(TypeError, match=msg):
            target[mask] = other[mask]

    # 定义一个测试方法，用于验证设置元组元素时的行为
    def test_setitem_tuple(self):
        # GH#20439: GitHub issue编号，标识此测试案例
        # 创建一个包含元组的分类对象，作为初始数据
        cat = Categorical([(0, 1), (0, 2), (0, 1)])

        # 此操作不应该引发异常，将第一个元组项设置为第二个元组项
        cat[1] = cat[0]
        # 断言第一个元组项是否被正确设置为(0, 1)
        assert cat[1] == (0, 1)

    # 定义一个测试方法，用于验证设置类似列表的输入时的行为
    def test_setitem_listlike(self):
        # GH#9469: GitHub issue编号，标识此测试案例
        # 使用随机整数初始化一个分类对象，并将其转换为 int8 类型
        cat = Categorical(
            np.random.default_rng(2).integers(0, 5, size=150000).astype(np.int8)
        ).add_categories([-1000])
        # 创建一个索引器数组，指定要更新的位置
        indexer = np.array([100000]).astype(np.int64)
        # 使用索引器将特定位置的值设置为 -1000
        cat[indexer] = -1000

        # 在此处断言代码的结果
        # 验证特定索引位置的结果是否被正确映射到 -1000 这个类别
        result = cat.codes[np.array([100000]).astype(np.int64)]
        tm.assert_numpy_array_equal(result, np.array([5], dtype="int8"))
    # 定义一个测试类 TestCategoricalIndexing，用于测试分类索引操作
class TestCategoricalIndexing:
    
    # 定义一个测试方法 test_getitem_slice，测试切片操作
    def test_getitem_slice(self):
        # 创建一个 Categorical 对象，包含字符串列表作为分类数据
        cat = Categorical(["a", "b", "c", "d", "a", "b", "c"])
        # 进行单元素索引操作，获取索引为3的元素
        sliced = cat[3]
        # 断言切片结果为字符串 "d"
        assert sliced == "d"

        # 进行切片操作，获取索引为3到4的子集
        sliced = cat[3:5]
        # 创建预期的 Categorical 对象，包含子集 ["d", "a"]，并指定类别列表为 ["a", "b", "c", "d"]
        expected = Categorical(["d", "a"], categories=["a", "b", "c", "d"])
        # 使用测试框架断言两个 Categorical 对象相等
        tm.assert_categorical_equal(sliced, expected)

    # 定义一个测试方法 test_getitem_listlike，测试列表型索引操作
    def test_getitem_listlike(self):
        # GH 9469
        # properly coerce the input indexers
        
        # 使用 NumPy 随机数生成器创建一个随机整数数组，作为 Categorical 对象的数据
        c = Categorical(
            np.random.default_rng(2).integers(0, 5, size=150000).astype(np.int8)
        )
        # 通过索引数组获取对应的 codes 数组
        result = c.codes[np.array([100000]).astype(np.int64)]
        # 创建预期的索引数组，并获取其 codes 数组
        expected = c[np.array([100000]).astype(np.int64)].codes
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法 test_periodindex，测试 PeriodIndex 相关操作
    def test_periodindex(self):
        # 创建一个 PeriodIndex 对象 idx1，包含日期字符串数组，频率为月份
        idx1 = PeriodIndex(
            ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"],
            freq="M",
        )

        # 将 PeriodIndex 对象 idx1 转换为 Categorical 对象
        cat1 = Categorical(idx1)
        # 执行字符串转换操作
        str(cat1)
        # 创建预期的 codes 数组 exp_arr 和索引对象 exp_idx
        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
        exp_idx = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")
        # 使用测试框架断言 Categorical 对象的 codes 数组和类别索引与预期相等
        tm.assert_numpy_array_equal(cat1._codes, exp_arr)
        tm.assert_index_equal(cat1.categories, exp_idx)

        # 创建一个新的 PeriodIndex 对象 idx2，包含日期字符串数组，频率为月份，并指定有序
        idx2 = PeriodIndex(
            ["2014-03", "2014-03", "2014-02", "2014-01", "2014-03", "2014-01"],
            freq="M",
        )
        # 将 PeriodIndex 对象 idx2 转换为有序的 Categorical 对象
        cat2 = Categorical(idx2, ordered=True)
        # 执行字符串转换操作
        str(cat2)
        # 创建预期的 codes 数组 exp_arr 和索引对象 exp_idx2
        exp_arr = np.array([2, 2, 1, 0, 2, 0], dtype=np.int8)
        exp_idx2 = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")
        # 使用测试框架断言有序的 Categorical 对象的 codes 数组和类别索引与预期相等
        tm.assert_numpy_array_equal(cat2._codes, exp_arr)
        tm.assert_index_equal(cat2.categories, exp_idx2)

        # 创建一个新的 PeriodIndex 对象 idx3，包含日期字符串数组，频率为月份，并指定有序
        idx3 = PeriodIndex(
            [
                "2013-12",
                "2013-11",
                "2013-10",
                "2013-09",
                "2013-08",
                "2013-07",
                "2013-05",
            ],
            freq="M",
        )
        # 将 PeriodIndex 对象 idx3 转换为有序的 Categorical 对象
        cat3 = Categorical(idx3, ordered=True)
        # 创建预期的 codes 数组 exp_arr 和索引对象 exp_idx
        exp_arr = np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
        exp_idx = PeriodIndex(
            [
                "2013-05",
                "2013-07",
                "2013-08",
                "2013-09",
                "2013-10",
                "2013-11",
                "2013-12",
            ],
            freq="M",
        )
        # 使用测试框架断言有序的 Categorical 对象的 codes 数组和类别索引与预期相等
        tm.assert_numpy_array_equal(cat3._codes, exp_arr)
        tm.assert_index_equal(cat3.categories, exp_idx)

    # 使用 pytest 的参数化装饰器，定义一个测试方法 test_periodindex_on_null_types，测试空值类型的 PeriodIndex 操作
    @pytest.mark.parametrize(
        "null_val",
        [None, np.nan, NaT, NA, math.nan, "NaT", "nat", "NAT", "nan", "NaN", "NAN"],
    )
    def test_periodindex_on_null_types(self, null_val):
        # GH 46673
        # 创建一个 PeriodIndex 对象 result，包含日期字符串数组和空值类型，频率为天
        result = PeriodIndex(["2022-04-06", "2022-04-07", null_val], freq="D")
        # 创建预期的 PeriodIndex 对象 expected，将空值类型替换为 NaT
        expected = PeriodIndex(["2022-04-06", "2022-04-07", "NaT"], dtype="period[D]")
        # 使用断言检查 result 中索引为2的值是否为 NaT
        assert result[2] is NaT
        # 使用测试框架断言两个 PeriodIndex 对象相等
        tm.assert_index_equal(result, expected)

    # 使用 pytest 的参数化装饰器，定义一个测试方法 test_periodindex_on_null_types，测试空值类型的 PeriodIndex 操作
    @pytest.mark.parametrize("new_categories", [[1, 2, 3, 4], [1, 2]])
    # 定义测试方法：测试当新分类长度不正确时是否会引发异常
    def test_categories_assignments_wrong_length_raises(self, new_categories):
        # 创建一个包含重复项的分类对象
        cat = Categorical(["a", "b", "c", "a"])
        # 定义异常消息
        msg = (
            "new categories need to have the same number of items "
            "as the old categories!"
        )
        # 使用 pytest 的断言检查是否会抛出 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            cat.rename_categories(new_categories)

    # 测试参数化：不同排序和唯一性组合的情况
    @pytest.mark.parametrize(
        "idx_values", [[1, 2, 3, 4], [1, 3, 2, 4], [1, 3, 3, 4], [1, 2, 2, 4]]
    )
    # 测试参数化：不同缺失和唯一性组合的情况
    @pytest.mark.parametrize("key_values", [[1, 2], [1, 5], [1, 1], [5, 5]])
    @pytest.mark.parametrize("key_class", [Categorical, CategoricalIndex])
    @pytest.mark.parametrize("dtype", [None, "category", "key"])
    # 定义测试方法：测试获取非唯一索引器的情况
    def test_get_indexer_non_unique(self, idx_values, key_values, key_class, dtype):
        # GH 21448
        # 创建一个指定值和分类的对象
        key = key_class(key_values, categories=range(1, 5))

        # 如果 dtype 为 "key"，则将其设置为 key 的数据类型
        if dtype == "key":
            dtype = key.dtype

        # 创建索引对象 idx
        idx = Index(idx_values, dtype=dtype)
        # 调用 idx 对象的方法，获取非唯一索引器
        expected, exp_miss = idx.get_indexer_non_unique(key_values)
        result, res_miss = idx.get_indexer_non_unique(key)

        # 使用 pandas 测试工具（tm）来断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(expected, result)
        tm.assert_numpy_array_equal(exp_miss, res_miss)

        # 获取唯一值索引器并进行断言
        exp_unique = idx.unique().get_indexer(key_values)
        res_unique = idx.unique().get_indexer(key)
        tm.assert_numpy_array_equal(res_unique, exp_unique)

    # 定义测试方法：测试 where 方法中未观察到值为 NaN 的情况
    def test_where_unobserved_nan(self):
        # 创建一个包含分类数据的 Series 对象
        ser = Series(Categorical(["a", "b"]))
        # 调用 where 方法
        result = ser.where([True, False])
        # 创建期望的 Series 对象
        expected = Series(Categorical(["a", None], categories=["a", "b"]))
        # 使用 pandas 测试工具（tm）来断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 全部为 NA 的情况
        ser = Series(Categorical(["a", "b"]))
        result = ser.where([False, False])
        expected = Series(Categorical([None, None], categories=["a", "b"]))
        tm.assert_series_equal(result, expected)

    # 定义测试方法：测试 where 方法中未观察到分类的情况
    def test_where_unobserved_categories(self):
        # 创建一个指定分类和分类索引的 Series 对象
        ser = Series(Categorical(["a", "b", "c"], categories=["d", "c", "b", "a"]))
        # 调用 where 方法
        result = ser.where([True, True, False], other="b")
        # 创建期望的 Series 对象
        expected = Series(Categorical(["a", "b", "b"], categories=ser.cat.categories))
        # 使用 pandas 测试工具（tm）来断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法：测试 where 方法中其他分类的情况
    def test_where_other_categorical(self):
        # 创建一个指定分类和分类索引的 Series 对象
        ser = Series(Categorical(["a", "b", "c"], categories=["d", "c", "b", "a"]))
        # 创建另一个分类对象
        other = Categorical(["b", "c", "a"], categories=["a", "c", "b", "d"])
        # 调用 where 方法
        result = ser.where([True, False, True], other)
        # 创建期望的 Series 对象
        expected = Series(Categorical(["a", "c", "c"], dtype=ser.dtype))
        # 使用 pandas 测试工具（tm）来断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法：测试 where 方法中设置新分类时是否会引发异常
    def test_where_new_category_raises(self):
        # 创建一个包含分类数据的 Series 对象
        ser = Series(Categorical(["a", "b", "c"]))
        # 定义异常消息
        msg = "Cannot setitem on a Categorical with a new category"
        # 使用 pytest 的断言检查是否会抛出 TypeError 异常，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            ser.where([True, False, True], "d")
    # 定义一个测试方法，用于测试条件不匹配时是否会引发异常
    def test_where_ordered_differs_rasies(self):
        # 创建一个序列对象，其中包含有序分类数据，原始顺序为["a", "b", "c"]，类别定义为["d", "c", "b", "a"]
        ser = Series(
            Categorical(["a", "b", "c"], categories=["d", "c", "b", "a"], ordered=True)
        )
        # 创建另一个分类对象，其中包含有序分类数据，原始顺序为["b", "c", "a"]，类别定义为["a", "c", "b", "d"]
        other = Categorical(
            ["b", "c", "a"], categories=["a", "c", "b", "d"], ordered=True
        )
        # 使用 pytest 断言语法，在执行 ser.where(...) 操作时，预期会引发 TypeError 异常，
        # 并且异常消息中包含 "without identical categories"
        with pytest.raises(TypeError, match="without identical categories"):
            # 在 ser 序列中根据条件 [True, False, True] 执行 where 操作，使用 other 对象作为替代值
            ser.where([True, False, True], other)
class TestContains:
    # 测试类 TestContains，用于测试 Categorical 类的包含操作

    def test_contains(self):
        # 测试方法 test_contains，验证 Categorical 对象的包含性质

        # 创建 Categorical 对象，指定数据和类别
        cat = Categorical(list("aabbca"), categories=list("cab"))

        # 断言 'b' 在 cat 中
        assert "b" in cat
        # 断言 'z' 不在 cat 中
        assert "z" not in cat
        # 断言 NaN 不在 cat 中
        assert np.nan not in cat

        # 使用 pytest 检查预期异常
        with pytest.raises(TypeError, match="unhashable type: 'list'"):
            assert [1] in cat

        # 断言整数 0 和 1 不在 cat 中
        assert 0 not in cat
        assert 1 not in cat

        # 添加 NaN 到数据中，重新创建 Categorical 对象
        cat = Categorical(list("aabbca") + [np.nan], categories=list("cab"))
        # 断言 NaN 在 cat 中
        assert np.nan in cat

    @pytest.mark.parametrize(
        "item, expected",
        [
            (Interval(0, 1), True),
            (1.5, True),
            (Interval(0.5, 1.5), False),
            ("a", False),
            (Timestamp(1), False),
            (Timedelta(1), False),
        ],
        ids=str,
    )
    def test_contains_interval(self, item, expected):
        # 测试方法 test_contains_interval，验证 Categorical 对象在区间和非区间对象中的包含性质

        # 创建 Categorical 对象，基于 IntervalIndex
        cat = Categorical(IntervalIndex.from_breaks(range(3)))
        # 检查 item 是否在 cat 中，并与预期结果比较
        result = item in cat
        assert result is expected

    def test_contains_list(self):
        # 测试方法 test_contains_list，验证 Categorical 对象处理列表的包含性质

        # 创建 Categorical 对象，数据为 [1, 2, 3]
        cat = Categorical([1, 2, 3])

        # 断言 'a' 不在 cat 中
        assert "a" not in cat

        # 使用 pytest 检查预期异常，列表 ["a"] 不可哈希
        with pytest.raises(TypeError, match="unhashable type"):
            ["a"] in cat

        # 使用 pytest 检查预期异常，列表 ["a", "b"] 不可哈希
        with pytest.raises(TypeError, match="unhashable type"):
            ["a", "b"] in cat


@pytest.mark.parametrize("index", [True, False])
def test_mask_with_boolean(index):
    # 参数化测试方法 test_mask_with_boolean，验证 Series 对象通过布尔类型 CategoricalIndex 进行掩码操作

    # 创建 Series 对象，数据为 [0, 1, 2]
    ser = Series(range(3))
    # 创建 Categorical 对象，数据为 [True, False, True]
    idx = Categorical([True, False, True])

    # 如果 index 为 True，则将 idx 转换为 CategoricalIndex 类型
    if index:
        idx = CategoricalIndex(idx)

    # 断言 idx 是布尔类型索引器
    assert com.is_bool_indexer(idx)

    # 使用 idx 对 ser 进行掩码操作，检查结果是否与预期相同
    result = ser[idx]
    expected = ser[idx.astype("object")]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("index", [True, False])
def test_mask_with_boolean_na_treated_as_false(index):
    # 参数化测试方法 test_mask_with_boolean_na_treated_as_false，验证处理布尔类型 CategoricalIndex 中的 NaN 作为 False 的情况

    # 创建 Series 对象，数据为 [0, 1, 2]
    ser = Series(range(3))
    # 创建 Categorical 对象，数据为 [True, False, None]
    idx = Categorical([True, False, None])

    # 如果 index 为 True，则将 idx 转换为 CategoricalIndex 类型
    if index:
        idx = CategoricalIndex(idx)

    # 使用 idx 对 ser 进行掩码操作，检查处理 NaN 后的结果是否与预期相同
    result = ser[idx]
    expected = ser[idx.fillna(False)]

    tm.assert_series_equal(result, expected)


@pytest.fixture
def non_coercible_categorical(monkeypatch):
    """
    Monkeypatch Categorical.__array__ to ensure no implicit conversion.

    Raises
    ------
    ValueError
        When Categorical.__array__ is called.
    """

    # 用于测试的固件 non_coercible_categorical，确保 Categorical.__array__ 不会进行隐式转换

    # 定义 array 方法，当调用 Categorical.__array__ 时抛出 ValueError 异常
    def array(self, dtype=None):
        raise ValueError("I cannot be converted.")

    # 使用 monkeypatch 修改 Categorical 类的 __array__ 方法为 array
    with monkeypatch.context() as m:
        m.setattr(Categorical, "__array__", array)
        yield


def test_series_at():
    # 测试方法 test_series_at，验证 Series 对象的索引操作

    # 创建 Categorical 对象，数据为 ["a", "b", "c"]
    arr = Categorical(["a", "b", "c"])
    # 创建 Series 对象，以 arr 作为数据
    ser = Series(arr)

    # 获取 ser 的索引为 0 的元素，断言结果与预期相同
    result = ser.at[0]
    assert result == "a"
```