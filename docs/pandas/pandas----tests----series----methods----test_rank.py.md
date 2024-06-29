# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_rank.py`

```
# 导入必要的库和模块
from itertools import chain  # 导入 chain 函数，用于扁平化多个可迭代对象
import operator  # 导入 operator 模块，用于函数操作符的函数形式

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas._libs.algos import (  # 从 pandas 库中导入算法相关的模块
    Infinity,  # 正无穷的表示
    NegInfinity,  # 负无穷的表示
)
import pandas.util._test_decorators as td  # 导入 pandas 测试装饰器模块

from pandas import (  # 从 pandas 库中导入多个对象
    NA,  # 表示缺失数据的常量
    NaT,  # 表示缺失时间的常量
    Series,  # 表示一维数据结构的对象
    Timestamp,  # 表示时间戳的对象
    date_range,  # 生成日期范围的函数
)
import pandas._testing as tm  # 导入 pandas 测试工具模块
from pandas.api.types import CategoricalDtype  # 导入 pandas 类型模块中的分类数据类型

# 定义一个 Pytest 的 fixture，返回一个 Series 对象
@pytest.fixture
def ser():
    return Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])

# 定义一个 Pytest 的 fixture，包含不同的参数化组合
@pytest.fixture(
    params=[
        ["average", np.array([1.5, 5.5, 7.0, 3.5, np.nan, 3.5, 1.5, 8.0, np.nan, 5.5])],
        ["min", np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5])],
        ["max", np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6])],
        ["first", np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6])],
        ["dense", np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])],
    ]
)
def results(request):
    return request.param  # 返回参数化的结果

# 定义一个 Pytest 的 fixture，包含不同的数据类型参数化
@pytest.fixture(
    params=[
        "object",
        "float64",
        "int64",
        "Float64",
        "Int64",
        pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("int64[pyarrow]", marks=td.skip_if_no("pyarrow")),
    ]
)
def dtype(request):
    return request.param  # 返回数据类型参数

# 定义一个测试类 TestSeriesRank，用于测试 Series 的排名功能
class TestSeriesRank:
    def test_rank_nullable_integer(self):
        # 测试用例 GH 56976，测试排名时处理可空整数的情况
        exp = Series([np.nan, 2, np.nan, 3, 3, 2, 3, 1])
        exp = exp.astype("Int64")  # 将 Series 转换为可空整数类型
        result = exp.rank(na_option="keep")  # 对 Series 进行排名操作，保留缺失值

        expected = Series([np.nan, 2.5, np.nan, 5.0, 5.0, 2.5, 5.0, 1.0])

        tm.assert_series_equal(result, expected)  # 使用测试工具模块检查结果是否与预期相符

    def test_rank_signature(self):
        s = Series([0, 1])
        s.rank(method="average")  # 测试排名方法为平均值的情况
        msg = "No axis named average for object type Series"
        with pytest.raises(ValueError, match=msg):  # 检查是否抛出 ValueError 异常，并匹配特定消息
            s.rank("average")  # 尝试使用字符串参数调用 rank 方法

    @pytest.mark.parametrize("dtype", [None, object])
    def test_rank_tie_methods(self, ser, results, dtype):
        method, exp = results
        ser = ser if dtype is None else ser.astype(dtype)  # 根据参数化的数据类型转换 Series
        result = ser.rank(method=method)  # 使用指定的方法对 Series 进行排名
        tm.assert_series_equal(result, Series(exp))  # 使用测试工具模块检查结果是否与预期相符

    @pytest.mark.parametrize("na_option", ["top", "bottom", "keep"])
    @pytest.mark.parametrize(
        "dtype, na_value, pos_inf, neg_inf",
        [
            ("object", None, Infinity(), NegInfinity()),  # 测试对象类型为 object 时的无穷大和无穷小
            ("float64", np.nan, np.inf, -np.inf),  # 测试 float64 类型时的无穷大和无穷小
            ("Float64", NA, np.inf, -np.inf),  # 测试 Float64 类型时的无穷大和无穷小
            pytest.param(
                "float64[pyarrow]",
                NA,
                np.inf,
                -np.inf,
                marks=td.skip_if_no("pyarrow"),  # 使用 pytest 装饰器跳过没有 pyarrow 的测试
            ),
        ],
    )
    def test_rank_tie_methods_on_infs_nans(
        self, rank_method, na_option, ascending, dtype, na_value, pos_inf, neg_inf
    ):
        # 检查是否可以导入 pytest
        pytest.importorskip("scipy")
        # 根据 dtype 参数设置预期的数据类型
        if dtype == "float64[pyarrow]":
            if rank_method == "average":
                exp_dtype = "float64[pyarrow]"
            else:
                exp_dtype = "uint64[pyarrow]"
        else:
            exp_dtype = "float64"

        # 设置测试中的数据块大小
        chunk = 3
        # 创建包含负无穷、缺失值和正无穷的输入数组
        in_arr = [neg_inf] * chunk + [na_value] * chunk + [pos_inf] * chunk
        # 将输入数组转换为 pandas Series 对象
        iseries = Series(in_arr, dtype=dtype)
        # 设置预期的排名结果字典
        exp_ranks = {
            "average": ([2, 2, 2], [5, 5, 5], [8, 8, 8]),
            "min": ([1, 1, 1], [4, 4, 4], [7, 7, 7]),
            "max": ([3, 3, 3], [6, 6, 6], [9, 9, 9]),
            "first": ([1, 2, 3], [4, 5, 6], [7, 8, 9]),
            "dense": ([1, 1, 1], [2, 2, 2], [3, 3, 3]),
        }
        # 根据排名方法选择预期的排名结果
        ranks = exp_ranks[rank_method]
        # 根据缺失值处理选项设置期望的排序顺序
        if na_option == "top":
            order = [ranks[1], ranks[0], ranks[2]]
        elif na_option == "bottom":
            order = [ranks[0], ranks[2], ranks[1]]
        else:
            order = [ranks[0], [np.nan] * chunk, ranks[1]]
        # 如果需要降序排列，则反转排序顺序
        expected = order if ascending else order[::-1]
        # 展开排序顺序并转换为列表
        expected = list(chain.from_iterable(expected))
        # 计算 pandas Series 对象的排名结果
        result = iseries.rank(
            method=rank_method, na_option=na_option, ascending=ascending
        )
        # 断言排名结果与预期结果的一致性
        tm.assert_series_equal(result, Series(expected, dtype=exp_dtype))

    def test_rank_desc_mix_nans_infs(self):
        # GH 19538
        # 检查混合 NaN 和 Inf 时的降序排名
        iseries = Series([1, np.nan, np.inf, -np.inf, 25])
        # 计算 Series 对象的降序排名
        result = iseries.rank(ascending=False)
        # 设置预期的结果 Series 对象
        exp = Series([3, np.nan, 1, 4, 2], dtype="float64")
        # 断言排名结果与预期结果的一致性
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "op, value",
        [
            [operator.add, 0],
            [operator.add, 1e6],
            [operator.mul, 1e-6],
        ],
    )
    def test_rank_methods_series(self, rank_method, op, value):
        # 检查是否可以导入 scipy.stats
        sp_stats = pytest.importorskip("scipy.stats")

        # 生成随机标准正态分布数据
        xs = np.random.default_rng(2).standard_normal(9)
        # 添加重复项以增加数据复杂性
        xs = np.concatenate([xs[i:] for i in range(0, 9, 2)])
        np.random.default_rng(2).shuffle(xs)

        # 创建与数据长度相同的索引
        index = [chr(ord("a") + i) for i in range(len(xs))]
        # 根据操作符和数值对数据进行操作
        vals = op(xs, value)
        # 创建 pandas Series 对象
        ts = Series(vals, index=index)
        # 计算 Series 对象的排名结果
        result = ts.rank(method=rank_method)
        # 使用 scipy.stats 中的 rankdata 函数计算预期的排名结果
        sprank = sp_stats.rankdata(
            vals, rank_method if rank_method != "first" else "ordinal"
        )
        # 将预期的排名结果转换为 pandas Series 对象并指定数据类型
        expected = Series(sprank, index=index).astype("float64")
        # 断言排名结果与预期结果的一致性
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, exp",
        [
            ([1], [1]),
            ([2], [1]),
            ([0], [1]),
            ([2, 2], [1, 1]),
            ([1, 2, 3], [1, 2, 3]),
            ([4, 2, 1], [3, 2, 1]),
            ([1, 1, 5, 5, 3], [1, 1, 3, 3, 2]),
            ([-5, -4, -3, -2, -1], [1, 2, 3, 4, 5]),
        ],
    )
    # 测试稠密排名方法
    def test_rank_dense_method(self, dtype, ser, exp):
        # 将输入的序列转换为指定的数据类型
        s = Series(ser).astype(dtype)
        # 计算稠密排名结果
        result = s.rank(method="dense")
        # 将期望的结果序列转换为与计算结果相同的数据类型
        expected = Series(exp).astype(result.dtype)
        # 断言计算结果与期望结果相等
        tm.assert_series_equal(result, expected)

    # 测试降序排名
    def test_rank_descending(self, ser, results, dtype):
        # 获取结果的方法和未使用的结果
        method, _ = results
        # 根据数据类型处理输入序列
        if "i" in dtype:
            s = ser.dropna()
        else:
            s = ser.astype(dtype)

        # 计算降序排名结果
        res = s.rank(ascending=False)
        # 计算期望的降序排名结果
        expected = (s.max() - s).rank()
        # 断言计算结果与期望结果相等
        tm.assert_series_equal(res, expected)

        # 根据给定的方法计算降序排名结果
        expected = (s.max() - s).rank(method=method)
        res2 = s.rank(method=method, ascending=False)
        # 断言计算结果与期望结果相等
        tm.assert_series_equal(res2, expected)

    # 测试整数类型数据的排名
    def test_rank_int(self, ser, results):
        # 获取结果的方法和期望结果
        method, exp = results
        # 删除缺失值并转换为整数类型
        s = ser.dropna().astype("i8")

        # 计算排名结果
        result = s.rank(method=method)
        # 删除期望结果中的缺失值并调整索引
        expected = Series(exp).dropna()
        expected.index = result.index
        # 断言计算结果与期望结果相等
        tm.assert_series_equal(result, expected)

    # 测试对象类型序列的排名（bug修复）
    def test_rank_object_bug(self):
        # GH 13445
        # 对对象类型序列进行简单测试
        Series([np.nan] * 32).astype(object).rank(ascending=True)
        Series([np.nan] * 32).astype(object).rank(ascending=False)

    # 测试排名是否会就地修改序列
    def test_rank_modify_inplace(self):
        # GH 18521
        # 检查排名操作不会改变原序列
        s = Series([Timestamp("2017-01-05 10:20:27.569000"), NaT])
        expected = s.copy()

        s.rank()
        result = s
        # 断言计算结果与原序列相等
        tm.assert_series_equal(result, expected)

    # 测试处理极小值时的排名计算
    def test_rank_ea_small_values(self):
        # GH#52471
        # 创建包含极小值的序列，并使用最小值方法计算排名
        ser = Series(
            [5.4954145e29, -9.791984e-21, 9.3715776e-26, NA, 1.8790257e-28],
            dtype="Float64",
        )
        result = ser.rank(method="min")
        expected = Series([4, 1, 3, np.nan, 2])
        # 断言计算结果与期望结果相等
        tm.assert_series_equal(result, expected)
# 定义了一个名为 test_rank_dense_pct 的测试函数，用于测试 Series 对象的 rank 方法在 method='dense', pct=True 时的行为
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),  # 测试当输入为 [1] 时的预期输出为 [1.0]
        ([1, 2], [1.0 / 2, 2.0 / 2]),  # 测试当输入为 [1, 2] 时的预期输出为 [1.0/2, 2.0/2]
        ([2, 2], [1.0, 1.0]),  # 测试当输入为 [2, 2] 时的预期输出为 [1.0, 1.0]
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),  # 测试当输入为 [1, 2, 3] 时的预期输出为 [1.0/3, 2.0/3, 3.0/3]
        ([1, 2, 2], [1.0 / 2, 2.0 / 2, 2.0 / 2]),  # 测试当输入为 [1, 2, 2] 时的预期输出为 [1.0/2, 2.0/2, 2.0/2]
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),  # 测试当输入为 [4, 2, 1] 时的预期输出为 [3.0/3, 2.0/3, 1.0/3]
        ([1, 1, 5, 5, 3], [1.0 / 3, 1.0 / 3, 3.0 / 3, 3.0 / 3, 2.0 / 3]),  # 测试当输入为 [1, 1, 5, 5, 3] 时的预期输出为 [1.0/3, 1.0/3, 3.0/3, 3.0/3, 2.0/3]
        ([1, 1, 3, 3, 5, 5], [1.0 / 3, 1.0 / 3, 2.0 / 3, 2.0 / 3, 3.0 / 3, 3.0 / 3]),  # 测试当输入为 [1, 1, 3, 3, 5, 5] 时的预期输出为 [1.0/3, 1.0/3, 2.0/3, 2.0/3, 3.0/3, 3.0/3]
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),  # 测试当输入为 [-5, -4, -3, -2, -1] 时的预期输出为 [1.0/5, 2.0/5, 3.0/5, 4.0/5, 5.0/5]
    ],
)
def test_rank_dense_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)  # 创建 Series 对象，将输入 ser 转换为指定的数据类型 dtype
    result = s.rank(method="dense", pct=True)  # 调用 Series 的 rank 方法，使用 method='dense', pct=True 进行排名计算
    expected = Series(exp).astype(result.dtype)  # 将预期输出 exp 转换为和 result 相同的数据类型
    tm.assert_series_equal(result, expected)  # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等

# 定义了一个名为 test_rank_min_pct 的测试函数，用于测试 Series 对象的 rank 方法在 method='min', pct=True 时的行为
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),  # 同上，测试各种输入情况的预期输出
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0 / 2, 1.0 / 2]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.0 / 3, 2.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 5, 1.0 / 5, 4.0 / 5, 4.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 6, 1.0 / 6, 3.0 / 6, 3.0 / 6, 5.0 / 6, 5.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_min_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)
    result = s.rank(method="min", pct=True)
    expected = Series(exp).astype(result.dtype)
    tm.assert_series_equal(result, expected)

# 定义了一个名为 test_rank_max_pct 的测试函数，用于测试 Series 对象的 rank 方法在 method='max', pct=True 时的行为
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),  # 同上，测试各种输入情况的预期输出
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0, 1.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 3.0 / 3, 3.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [2.0 / 5, 2.0 / 5, 5.0 / 5, 5.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [2.0 / 6, 2.0 / 6, 4.0 / 6, 4.0 / 6, 6.0 / 6, 6.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_max_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)
    result = s.rank(method="max", pct=True)
    expected = Series(exp).astype(result.dtype)
    tm.assert_series_equal(result, expected)

# 定义了一个名为 test_rank_average_pct 的测试函数，用于测试 Series 对象的 rank 方法在 method='average', pct=True 时的行为
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),  # 同上，测试各种输入情况的预期输出
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.5 / 2, 1.5 / 2]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.
    # 将预期结果转换为与结果相同的数据类型，并创建 Series 对象
    expected = Series(exp).astype(result.dtype)
    # 使用测试框架中的函数来比较结果和预期结果的 Series 对象是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),  # 参数化测试用例：输入为 [1]，期望输出为 [1.0]
        ([1, 2], [1.0 / 2, 2.0 / 2]),  # 输入为 [1, 2]，期望输出为 [0.5, 1.0]
        ([2, 2], [1.0 / 2, 2.0 / 2.0]),  # 输入为 [2, 2]，期望输出为 [0.5, 1.0]
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),  # 输入为 [1, 2, 3]，期望输出为 [0.333, 0.666, 1.0]
        ([1, 2, 2], [1.0 / 3, 2.0 / 3, 3.0 / 3]),  # 输入为 [1, 2, 2]，期望输出为 [0.333, 0.666, 1.0]
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),  # 输入为 [4, 2, 1]，期望输出为 [1.0, 0.666, 0.333]
        ([1, 1, 5, 5, 3], [1.0 / 5, 2.0 / 5, 4.0 / 5, 5.0 / 5, 3.0 / 5]),  # 输入为 [1, 1, 5, 5, 3]，期望输出为 [0.2, 0.4, 0.8, 1.0, 0.6]
        ([1, 1, 3, 3, 5, 5], [1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6, 6.0 / 6]),  # 输入为 [1, 1, 3, 3, 5, 5]，期望输出为 [0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),  # 输入为 [-5, -4, -3, -2, -1]，期望输出为 [0.2, 0.4, 0.6, 0.8, 1.0]
    ],
)
def test_rank_first_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)  # 将输入列表 ser 转换为指定的 dtype 类型的 Series 对象
    result = s.rank(method="first", pct=True)  # 对 Series s 进行排名计算，使用 "first" 方法并返回百分比排名
    expected = Series(exp).astype(result.dtype)  # 将期望的输出列表 exp 转换为与 result 相同的数据类型的 Series 对象
    tm.assert_series_equal(result, expected)  # 使用测试框架中的方法比较 result 和 expected 是否相等


@pytest.mark.single_cpu
def test_pct_max_many_rows():
    # GH 18271
    s = Series(np.arange(2**24 + 1))  # 创建一个包含 2^24 + 1 个元素的 Series 对象
    result = s.rank(pct=True).max()  # 计算 Series s 的百分比排名，并返回最大值
    assert result == 1  # 断言最大值是否等于 1
```