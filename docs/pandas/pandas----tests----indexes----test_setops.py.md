# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_setops.py`

```
"""
The tests in this package are to ensure the proper resultant dtypes of
set operations.
"""

# 导入必要的库和模块
from datetime import datetime
import operator

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

from pandas._libs import lib  # 导入 pandas 的底层库

from pandas.core.dtypes.cast import find_common_type  # 导入用于查找公共数据类型的函数

from pandas import (  # 导入 pandas 中的多个子模块和类
    CategoricalDtype,
    CategoricalIndex,
    DatetimeTZDtype,
    Index,
    MultiIndex,
    PeriodDtype,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm  # 导入 pandas 测试工具模块
from pandas.api.types import (  # 导入 pandas 的数据类型检查函数
    is_signed_integer_dtype,
    pandas_dtype,
)


def equal_contents(arr1, arr2) -> bool:
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)  # 检查 arr1 和 arr2 的唯一元素集合是否相等


@pytest.fixture(
    params=tm.ALL_REAL_NUMPY_DTYPES
    + [
        "object",
        "category",
        "datetime64[ns]",
        "timedelta64[ns]",
    ]
)
def any_dtype_for_small_pos_integer_indexes(request):
    """
    Dtypes that can be given to an Index with small positive integers.

    This means that for any dtype `x` in the params list, `Index([1, 2, 3], dtype=x)` is
    valid and gives the correct Index (sub-)class.
    """
    return request.param  # 返回可以用于具有小正整数的索引的数据类型


@pytest.fixture
def index_flat2(index_flat):
    return index_flat  # 返回扁平化的索引


def test_union_same_types(index):
    # Union with a non-unique, non-monotonic index raises error
    # Only needed for bool index factory
    idx1 = index.sort_values()  # 对索引进行排序
    idx2 = index.sort_values()  # 对索引进行排序
    assert idx1.union(idx2).dtype == idx1.dtype  # 断言联合后的索引数据类型与原始索引一致


def test_union_different_types(index_flat, index_flat2, request):
    # This test only considers combinations of indices
    # GH 23525
    idx1 = index_flat  # 使用第一个索引
    idx2 = index_flat2  # 使用第二个索引

    if (
        not idx1.is_unique
        and not idx2.is_unique
        and idx1.dtype.kind == "i"
        and idx2.dtype.kind == "b"
    ) or (
        not idx2.is_unique
        and not idx1.is_unique
        and idx2.dtype.kind == "i"
        and idx1.dtype.kind == "b"
    ):
        # Each condition had idx[1|2].is_monotonic_decreasing
        # but failed when e.g.
        # idx1 = Index(
        # [True, True, True, True, True, True, True, True, False, False], dtype='bool'
        # )
        # idx2 = Index([0, 0, 1, 1, 2, 2], dtype='int64')
        mark = pytest.mark.xfail(
            reason="GH#44000 True==1", raises=ValueError, strict=False
        )
        request.applymarker(mark)  # 应用标记来标记失败的测试用例

    common_dtype = find_common_type([idx1.dtype, idx2.dtype])  # 查找两个索引的公共数据类型

    warn = None
    msg = "'<' not supported between"
    if not len(idx1) or not len(idx2):
        pass
    elif (idx1.dtype.kind == "c" and (not lib.is_np_dtype(idx2.dtype, "iufc"))) or (
        idx2.dtype.kind == "c" and (not lib.is_np_dtype(idx1.dtype, "iufc"))
    ):
        # complex objects non-sortable
        warn = RuntimeWarning  # 复杂对象不可排序时发出警告
    elif (
        isinstance(idx1.dtype, PeriodDtype) and isinstance(idx2.dtype, CategoricalDtype)
        # 检查是否有 idx1 是 PeriodDtype 而 idx2 是 CategoricalDtype 的情况
    ) or (
        isinstance(idx2.dtype, PeriodDtype) and isinstance(idx1.dtype, CategoricalDtype)
    ):
        # 如果 idx1 和 idx2 中任意一个的 dtype 是 PeriodDtype，并且另一个是 CategoricalDtype，则发出 FutureWarning
        warn = FutureWarning
        msg = r"PeriodDtype\[B\] is deprecated"
        mark = pytest.mark.xfail(
            reason="Warning not produced on all builds",
            raises=AssertionError,
            strict=False,
        )
        # 将 mark 应用到当前测试请求中
        request.applymarker(mark)

    any_uint64 = np.uint64 in (idx1.dtype, idx2.dtype)
    # 检查 idx1 或 idx2 是否包含 np.uint64 类型的数据
    idx1_signed = is_signed_integer_dtype(idx1.dtype)
    # 检查 idx1 的 dtype 是否是有符号整数类型
    idx2_signed = is_signed_integer_dtype(idx2.dtype)
    # 检查 idx2 的 dtype 是否是有符号整数类型

    # 对 idx1 和 idx2 进行排序以确保索引的非唯一性和非单调性不会引发错误
    idx1 = idx1.sort_values()
    idx2 = idx2.sort_values()

    # 使用上下文管理器 assert_produces_warning 检查是否产生特定的警告信息
    with tm.assert_produces_warning(warn, match=msg):
        res1 = idx1.union(idx2)
        res2 = idx2.union(idx1)

    if any_uint64 and (idx1_signed or idx2_signed):
        # 如果 idx1 或 idx2 包含 uint64 类型，并且其中至少一个是有符号整数类型，则检查结果的 dtype 是否为对象类型
        assert res1.dtype == np.dtype("O")
        assert res2.dtype == np.dtype("O")
    else:
        # 否则，检查结果的 dtype 是否与 common_dtype 相同
        assert res1.dtype == common_dtype
        assert res2.dtype == common_dtype
@pytest.mark.parametrize(
    "idx1,idx2",
    [  # 参数化测试用例，包括不同的索引对象组合
        (Index(np.arange(5), dtype=np.int64), RangeIndex(5)),  # 第一组测试参数
        (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.int64)),  # 第二组测试参数
        (Index(np.arange(5), dtype=np.float64), RangeIndex(5)),  # 第三组测试参数
        (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.uint64)),  # 第四组测试参数
    ],
)
def test_compatible_inconsistent_pairs(idx1, idx2):
    # GH 23525
    # 对于给定的索引对象，测试其 union 方法的结果
    res1 = idx1.union(idx2)
    # 对于给定的索引对象，测试其 union 方法的结果（交换参数顺序）
    res2 = idx2.union(idx1)

    # 断言 union 方法的结果数据类型在输入索引对象的数据类型集合中
    assert res1.dtype in (idx1.dtype, idx2.dtype)
    assert res2.dtype in (idx1.dtype, idx2.dtype)


@pytest.mark.parametrize(
    "left, right, expected",
    [  # 参数化测试用例，包括不同的数据类型组合
        ("int64", "int64", "int64"),  # 整数类型的测试
        ("int64", "uint64", "object"),  # 整数和无符号整数类型的测试
        ("int64", "float64", "float64"),  # 整数和浮点数类型的测试
        ("uint64", "float64", "float64"),  # 无符号整数和浮点数类型的测试
        ("uint64", "uint64", "uint64"),  # 两个无符号整数类型的测试
        ("float64", "float64", "float64"),  # 浮点数类型的测试
        ("datetime64[ns]", "int64", "object"),  # 日期时间和整数类型的测试
        ("datetime64[ns]", "uint64", "object"),  # 日期时间和无符号整数类型的测试
        ("datetime64[ns]", "float64", "object"),  # 日期时间和浮点数类型的测试
        ("datetime64[ns, CET]", "int64", "object"),  # 含时区信息的日期时间和整数类型的测试
        ("datetime64[ns, CET]", "uint64", "object"),  # 含时区信息的日期时间和无符号整数类型的测试
        ("datetime64[ns, CET]", "float64", "object"),  # 含时区信息的日期时间和浮点数类型的测试
        ("Period[D]", "int64", "object"),  # 时期类型和整数类型的测试
        ("Period[D]", "uint64", "object"),  # 时期类型和无符号整数类型的测试
        ("Period[D]", "float64", "object"),  # 时期类型和浮点数类型的测试
    ],
)
@pytest.mark.parametrize("names", [("foo", "foo", "foo"), ("foo", "bar", None)])
def test_union_dtypes(left, right, expected, names):
    left = pandas_dtype(left)
    right = pandas_dtype(right)
    # 使用指定数据类型和名称创建两个空索引对象
    a = Index([], dtype=left, name=names[0])
    b = Index([], dtype=right, name=names[1])
    # 执行索引对象的 union 方法，断言结果的数据类型和名称
    result = a.union(b)
    assert result.dtype == expected
    assert result.name == names[2]

    # 测试交集操作的名称保留
    # TODO: 确定所需的数据类型；是否要求其交换性？
    result = a.intersection(b)
    assert result.name == names[2]


@pytest.mark.parametrize("values", [[1, 2, 2, 3], [3, 3]])
def test_intersection_duplicates(values):
    # GH#31326
    # 创建索引对象并执行交集操作，断言结果与预期相等
    a = Index(values)
    b = Index([3, 3])
    result = a.intersection(b)
    expected = Index([3])
    tm.assert_index_equal(result, expected)


class TestSetOps:
    # Set operation tests shared by all indexes in the `index` fixture
    @pytest.mark.parametrize("case", [0.5, "xxx"])
    @pytest.mark.parametrize(
        "method", ["intersection", "union", "difference", "symmetric_difference"]
    )
    def test_set_ops_error_cases(self, case, method, index):
        # 非可迭代输入
        # 断言调用特定索引对象的集合操作方法（如交集、并集等）时抛出类型错误异常
        msg = "Input must be Index or array-like"
        with pytest.raises(TypeError, match=msg):
            getattr(index, method)(case)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 定义一个测试方法，用于测试索引对象的交集功能
    def test_intersection_base(self, index):
        # 如果索引对象是 CategoricalIndex 类型，则跳过测试，并给出相应的提示信息
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"Not relevant for {type(index).__name__}")

        # 获取索引对象的前五个唯一值作为第一个集合
        first = index[:5].unique()
        # 获取索引对象的前三个唯一值作为第二个集合
        second = index[:3].unique()
        # 计算第一个集合和第二个集合的交集
        intersect = first.intersection(second)
        # 断言计算得到的交集与第二个集合相等
        tm.assert_index_equal(intersect, second)

        # 如果索引对象的数据类型是 DatetimeTZDtype，则不适用下面的测试
        if isinstance(index.dtype, DatetimeTZDtype):
            # 下面的测试将会被跳过，因为 second.values 会去除时区信息
            return

        # GH#10149
        # 准备不同情况的测试数据
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            # 计算第一个集合和当前情况下的 case 的交集
            result = first.intersection(case)
            # 断言交集的内容与第二个集合相同
            assert equal_contents(result, second)

        # 如果索引对象是 MultiIndex 类型
        if isinstance(index, MultiIndex):
            # 准备错误消息
            msg = "other must be a MultiIndex or a list of tuples"
            # 确保在尝试用非 MultiIndex 对象进行交集计算时会抛出 TypeError 异常
            with pytest.raises(TypeError, match=msg):
                first.intersection([1, 2, 3])

    # 标记此测试方法以忽略特定警告
    @pytest.mark.filterwarnings(
        "ignore:Falling back on a non-pyarrow:pandas.errors.PerformanceWarning"
    )
    # 标记此测试方法以忽略特定警告，使用正则表达式匹配警告消息
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 定义一个测试方法，用于测试索引对象的并集功能
    def test_union_base(self, index):
        # 确保索引对象的元素是唯一的
        index = index.unique()
        # 从索引对象中获取第三个元素之后的所有元素作为第一个集合
        first = index[3:]
        # 从索引对象中获取前五个元素作为第二个集合
        second = index[:5]
        # 获取整个索引对象
        everything = index

        # 计算第一个集合和第二个集合的并集
        union = first.union(second)
        # 断言计算得到的并集排序后与整个索引对象排序后相等
        tm.assert_index_equal(union.sort_values(), everything.sort_values())

        # 如果索引对象的数据类型是 DatetimeTZDtype，则不适用下面的测试
        if isinstance(index.dtype, DatetimeTZDtype):
            # 下面的测试将会被跳过，因为 second.values 会去除时区信息
            return

        # GH#10149
        # 准备不同情况的测试数据
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            # 计算第一个集合和当前情况下的 case 的并集
            result = first.union(case)
            # 断言并集的内容与整个索引对象相同
            assert equal_contents(result, everything)

        # 如果索引对象是 MultiIndex 类型
        if isinstance(index, MultiIndex):
            # 准备错误消息
            msg = "other must be a MultiIndex or a list of tuples"
            # 确保在尝试用非 MultiIndex 对象进行并集计算时会抛出 TypeError 异常
            with pytest.raises(TypeError, match=msg):
                first.union([1, 2, 3])
    def test_difference_base(self, sort, index):
        # 从索引的第二个元素开始创建一个新的集合first
        first = index[2:]
        # 从索引的第一个元素到第四个元素创建一个新的集合second
        second = index[:4]
        # 如果索引推断类型为布尔类型
        if index.inferred_type == "boolean":
            # 计算first和second集合的差集，存储在answer中
            # TODO: 需要确认对于索引固定装置是否有关于此处不适用的假设？
            answer = set(first).difference(set(second))
        # 如果索引是CategoricalIndex的实例
        elif isinstance(index, CategoricalIndex):
            # 将answer设置为一个空列表
            answer = []
        else:
            # 否则，将answer设置为索引从第四个元素开始的所有元素组成的集合
            answer = index[4:]
        # 计算first集合与second集合的差集，按sort的规则排序，存储在result中
        result = first.difference(second, sort)
        # 断言result与answer的内容相等
        assert equal_contents(result, answer)

        # GH#10149
        # 创建一个包含不同形式second的案例列表
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        # 对于每个案例
        for case in cases:
            # 计算first集合与当前案例的差集，按sort的规则排序，存储在result中
            result = first.difference(case, sort)
            # 断言result与answer的内容相等
            assert equal_contents(result, answer)

        # 如果索引是MultiIndex的实例
        if isinstance(index, MultiIndex):
            # 设置错误消息
            msg = "other must be a MultiIndex or a list of tuples"
            # 使用pytest断言应该引发TypeError，并匹配msg消息
            with pytest.raises(TypeError, match=msg):
                # 计算first集合与列表[1, 2, 3]的差集，按sort的规则排序
                first.difference([1, 2, 3], sort)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.filterwarnings(
        "ignore:Falling back on a non-pyarrow:pandas.errors.PerformanceWarning"
    )
    def test_symmetric_difference(self, index):
        # 如果索引是CategoricalIndex的实例，则跳过测试
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"Not relevant for {type(index).__name__}")
        # 如果索引长度小于2，则跳过测试
        if len(index) < 2:
            pytest.skip("Too few values for test")
        # 如果索引的第一个元素在其余元素中，或者最后一个元素在其余元素中，则跳过测试
        if index[0] in index[1:] or index[-1] in index[:-1]:
            # 索引装置例如一个布尔索引，不满足此条件，另一个示例是[0, 0, 1, 1, 2, 2]
            pytest.skip("Index values do not satisfy test condition.")

        # 从索引的第二个元素开始创建一个新的集合first
        first = index[1:]
        # 从索引的第一个元素到倒数第二个元素创建一个新的集合second
        second = index[:-1]
        # 将索引的第一个和最后一个元素作为答案集合answer
        answer = index[[0, -1]]
        # 计算first集合与second集合的对称差集，按值排序后存储在result中
        result = first.symmetric_difference(second)
        # 断言result按值排序后与answer按值排序后内容相等
        tm.assert_index_equal(result.sort_values(), answer.sort_values())

        # GH#10149
        # 创建一个包含不同形式second的案例列表
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        # 对于每个案例
        for case in cases:
            # 计算first集合与当前案例的对称差集，存储在result中
            result = first.symmetric_difference(case)
            # 断言result与answer的内容相等
            assert equal_contents(result, answer)

        # 如果索引是MultiIndex的实例
        if isinstance(index, MultiIndex):
            # 设置错误消息
            msg = "other must be a MultiIndex or a list of tuples"
            # 使用pytest断言应该引发TypeError，并匹配msg消息
            with pytest.raises(TypeError, match=msg):
                # 计算first集合与列表[1, 2, 3]的对称差集
                first.symmetric_difference([1, 2, 3])

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_corner_union(self, index_flat, fname, sname, expected_name):
        # GH#9943, GH#9862
        # Test unions with various name combinations
        # Do not test MultiIndex or repeats
        # 检验不同名称组合的联合操作
        # 不测试多重索引或重复的情况

        if not index_flat.is_unique:
            # 如果索引不是唯一的，则获取其唯一值作为新索引
            index = index_flat.unique()
        else:
            index = index_flat

        # Test copy.union(copy)
        # 测试复制的索引对象与复制的索引对象的联合操作
        first = index.copy().set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test copy.union(empty)
        # 测试复制的索引对象与空索引对象的联合操作
        first = index.copy().set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test empty.union(copy)
        # 测试空索引对象与复制的索引对象的联合操作
        first = index.drop(index).set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test empty.union(empty)
        # 测试空索引对象与空索引对象的联合操作
        first = index.drop(index).set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_union_unequal(self, index_flat, fname, sname, expected_name):
        if not index_flat.is_unique:
            # 如果索引不是唯一的，则获取其唯一值作为新索引
            index = index_flat.unique()
        else:
            index = index_flat

        # test copy.union(subset) - need sort for unicode and string
        # 测试复制的索引对象与子集索引对象的联合操作 - 需要对 Unicode 和字符串进行排序
        first = index.copy().set_names(fname)
        second = index[1:].set_names(sname)
        union = first.union(second).sort_values()
        expected = index.set_names(expected_name).sort_values()
        tm.assert_index_equal(union, expected)
    def test_corner_intersect(self, index_flat, fname, sname, expected_name):
        # GH#35847
        # Test intersections with various name combinations
        # 检查索引是否唯一，如果不唯一则去重
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # Test copy.intersection(copy)
        # 复制索引并设置名称为fname
        first = index.copy().set_names(fname)
        # 复制索引并设置名称为sname
        second = index.copy().set_names(sname)
        # 计算两个索引的交集
        intersect = first.intersection(second)
        # 复制索引并设置预期名称为expected_name
        expected = index.copy().set_names(expected_name)
        # 断言交集结果与预期结果相等
        tm.assert_index_equal(intersect, expected)

        # Test copy.intersection(empty)
        # 复制索引并设置名称为fname
        first = index.copy().set_names(fname)
        # 从索引中删除所有元素，并设置名称为sname
        second = index.drop(index).set_names(sname)
        # 计算两个索引的交集
        intersect = first.intersection(second)
        # 复制索引并设置预期名称为expected_name
        expected = index.drop(index).set_names(expected_name)
        # 断言交集结果与预期结果相等
        tm.assert_index_equal(intersect, expected)

        # Test empty.intersection(copy)
        # 从索引中删除所有元素，并设置名称为fname
        first = index.drop(index).set_names(fname)
        # 复制索引并设置名称为sname
        second = index.copy().set_names(sname)
        # 计算两个索引的交集
        intersect = first.intersection(second)
        # 复制索引并设置预期名称为expected_name
        expected = index.drop(index).set_names(expected_name)
        # 断言交集结果与预期结果相等
        tm.assert_index_equal(intersect, expected)

        # Test empty.intersection(empty)
        # 从索引中删除所有元素，并设置名称为fname
        first = index.drop(index).set_names(fname)
        # 从索引中删除所有元素，并设置名称为sname
        second = index.drop(index).set_names(sname)
        # 计算两个索引的交集
        intersect = first.intersection(second)
        # 复制索引并设置预期名称为expected_name
        expected = index.drop(index).set_names(expected_name)
        # 断言交集结果与预期结果相等
        tm.assert_index_equal(intersect, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_intersect_unequal(self, index_flat, fname, sname, expected_name):
        # 检查索引是否唯一，如果不唯一则去重
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # test copy.intersection(subset) - need sort for unicode and string
        # 复制索引并设置名称为fname
        first = index.copy().set_names(fname)
        # 从索引中选取子集并设置名称为sname，需要排序以处理Unicode和字符串
        second = index[1:].set_names(sname)
        # 计算两个索引的交集并排序
        intersect = first.intersection(second).sort_values()
        # 复制索引并设置预期名称为expected_name，并排序
        expected = index[1:].set_names(expected_name).sort_values()
        # 断言交集结果与预期结果相等
        tm.assert_index_equal(intersect, expected)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_intersection_name_retention_with_nameless(self, index):
        # 如果索引是MultiIndex，则重命名其层级；否则，重命名为"foo"
        if isinstance(index, MultiIndex):
            index = index.rename(list(range(index.nlevels)))
        else:
            index = index.rename("foo")

        other = np.asarray(index)

        # 计算索引与其他数组的交集
        result = index.intersection(other)
        # 断言结果的名称与索引的名称相同
        assert result.name == index.name

        # 空的其他数组，但是类型相同
        # 计算索引与空的其他数组的交集
        result = index.intersection(other[:0])
        # 断言结果的名称与索引的名称相同
        assert result.name == index.name

        # 空的self索引
        # 计算空的索引与其他数组的交集
        result = index[:0].intersection(other)
        # 断言结果的名称与索引的名称相同
        assert result.name == index.name
    # GH#20040
    # 如果从一个集合中取出它自己的差集，需要保持索引类型的一致性
    def test_difference_preserves_type_empty(self, index, sort):
        # 如果索引不是唯一的，跳过测试
        if not index.is_unique:
            pytest.skip("Not relevant since index is not unique")
        # 计算索引和自身的差集，并根据排序标志进行排序
        result = index.difference(index, sort=sort)
        # 期望的结果是一个空的索引
        expected = index[:0]
        # 断言计算结果与期望结果相等
        tm.assert_index_equal(result, expected, exact=True)

    # GH#20040
    # 测试索引重命名后差集的保留性
    def test_difference_name_retention_equals(self, index, names):
        # 如果索引是 MultiIndex 类型，则对名称进行重复以匹配索引的层级数
        if isinstance(index, MultiIndex):
            names = [[x] * index.nlevels for x in names]
        # 对索引进行重命名操作
        index = index.rename(names[0])
        other = index.rename(names[1])

        # 断言重命名后的索引相等
        assert index.equals(other)

        # 计算索引与重命名后索引的差集
        result = index.difference(other)
        # 期望的结果是一个空的索引，并且使用给定的名称重命名
        expected = index[:0].rename(names[2])
        # 断言计算结果与期望结果相等
        tm.assert_index_equal(result, expected)

    # GH#20040
    # 测试空索引与自身的差集是否等同于索引与自身的差集
    def test_intersection_difference_match_empty(self, index, sort):
        # 如果索引不是唯一的，跳过测试
        if not index.is_unique:
            pytest.skip("Not relevant because index is not unique")
        # 计算索引与空索引的交集
        inter = index.intersection(index[:0])
        # 计算索引与自身的差集，并根据排序标志进行排序
        diff = index.difference(index, sort=sort)
        # 断言交集结果与差集结果相等
        tm.assert_index_equal(inter, diff, exact=True)
# 忽略特定警告消息，用于测试目的，过滤掉有关于期望行为的警告
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
# 忽略特定警告消息，用于测试目的，过滤掉有关于性能警告的警告
@pytest.mark.filterwarnings(
    "ignore:Falling back on a non-pyarrow:pandas.errors.PerformanceWarning"
)
# 参数化测试方法，对指定的方法（交集、并集、差集、对称差集）进行多次测试
@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_setop_with_categorical(index_flat, sort, method):
    # 将MultiIndex测试分开处理，详见tests.indexes.multi.test_setops
    # 使用传入的index_flat作为测试用的index
    index = index_flat

    # 将index转换为category类型，并赋值给other
    other = index.astype("category")
    # 如果index是RangeIndex类型，则exact为"equiv"，否则为True
    exact = "equiv" if isinstance(index, RangeIndex) else True

    # 对index和other执行指定方法（如交集、并集等），并将结果赋给result和expected
    result = getattr(index, method)(other, sort=sort)
    expected = getattr(index, method)(index, sort=sort)
    # 断言结果是否相等
    tm.assert_index_equal(result, expected, exact=exact)

    # 对index和other的前5个元素执行指定方法，将结果赋给result和expected
    result = getattr(index, method)(other[:5], sort=sort)
    expected = getattr(index, method)(index[:5], sort=sort)
    # 断言结果是否相等
    tm.assert_index_equal(result, expected, exact=exact)


def test_intersection_duplicates_all_indexes(index):
    # GH#38743
    # 如果index为空，则跳过该测试
    if index.empty:
        pytest.skip("Not relevant for empty Index")

    # 将index赋值给idx
    idx = index
    # 从idx中选择部分元素组成idx_non_unique
    idx_non_unique = idx[[0, 0, 1, 2]]

    # 断言idx和idx_non_unique的交集是否与idx_non_unique和idx的交集相等
    assert idx.intersection(idx_non_unique).equals(idx_non_unique.intersection(idx))
    # 断言idx和idx_non_unique的交集是否唯一
    assert idx.intersection(idx_non_unique).is_unique


def test_union_duplicate_index_subsets_of_each_other(
    any_dtype_for_small_pos_integer_indexes,
):
    # GH#31326
    # 使用any_dtype_for_small_pos_integer_indexes作为数据类型
    dtype = any_dtype_for_small_pos_integer_indexes
    # 创建Index对象a和b
    a = Index([1, 2, 2, 3], dtype=dtype)
    b = Index([3, 3, 4], dtype=dtype)

    # 创建期望的结果Index对象expected
    expected = Index([1, 2, 2, 3, 3, 4], dtype=dtype)
    # 如果a是CategoricalIndex类型，则修改expected
    if isinstance(a, CategoricalIndex):
        expected = Index([1, 2, 2, 3, 3, 4])
    # 对a和b执行union操作，将结果赋给result
    result = a.union(b)
    # 断言结果是否与期望相等
    tm.assert_index_equal(result, expected)
    # 对a和b执行union操作（不排序），将结果赋给result
    result = a.union(b, sort=False)
    # 断言结果是否与期望相等
    tm.assert_index_equal(result, expected)


def test_union_with_duplicate_index_and_non_monotonic(
    any_dtype_for_small_pos_integer_indexes,
):
    # GH#36289
    # 使用any_dtype_for_small_pos_integer_indexes作为数据类型
    dtype = any_dtype_for_small_pos_integer_indexes
    # 创建Index对象a和b
    a = Index([1, 0, 0], dtype=dtype)
    b = Index([0, 1], dtype=dtype)
    # 创建期望的结果Index对象expected
    expected = Index([0, 0, 1], dtype=dtype)

    # 对a和b执行union操作，将结果赋给result
    result = a.union(b)
    # 断言结果是否与期望相等
    tm.assert_index_equal(result, expected)

    # 对b和a执行union操作，将结果赋给result
    result = b.union(a)
    # 断言结果是否与期望相等
    tm.assert_index_equal(result, expected)


def test_union_duplicate_index_different_dtypes():
    # GH#36289
    # 创建Index对象a和b，分别包含整数和字符串
    a = Index([1, 2, 2, 3])
    b = Index(["1", "0", "0"])
    # 创建期望的结果Index对象expected
    expected = Index([1, 2, 2, 3, "1", "0", "0"])
    # 对a和b执行union操作（不排序），将结果赋给result
    result = a.union(b, sort=False)
    # 断言结果是否与期望相等
    tm.assert_index_equal(result, expected)


def test_union_same_value_duplicated_in_both():
    # GH#36289
    # 创建Index对象a和b，包含重复的值
    a = Index([0, 0, 1])
    b = Index([0, 0, 1, 2])
    # 对a和b执行union操作，将结果赋给result
    result = a.union(b)
    # 创建期望的结果Index对象expected
    expected = Index([0, 0, 1, 2])
    # 断言结果是否与期望相等
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dup", [1, np.nan])
def test_union_nan_in_both(dup):
    # GH#36289
    # 创建包含NaN和整数的Index对象a和b
    a = Index([np.nan, 1, 2, 2])
    b = Index([np.nan, dup, 1, 2])
    # 对a和b执行union操作（不排序），将结果赋给result
    result = a.union(b, sort=False)
    # 创建期望的结果Index对象expected
    expected = Index([np.nan, dup, 1.0, 2.0, 2.0])
    # 断言结果是否与期望相等
    tm.assert_index_equal(result, expected)
    # 使用测试工具tm来比较result和expected的索引是否相等
    tm.assert_index_equal(result, expected)
````
def test_union_rangeindex_sort_true():
    # 测试 RangeIndex 的 union 操作，设置 sort=True 参数
    idx1 = RangeIndex(1, 100, 6)  # 创建一个 RangeIndex，从 1 到 100，步长为 6
    idx2 = RangeIndex(1, 50, 3)   # 创建另一个 RangeIndex，从 1 到 50，步长为 3
    result = idx1.union(idx2, sort=True)  # 执行 union 操作，排序结果
    expected = Index(
        [
            1,
            4,
            7,
            10,
            13,
            16,
            19,
            22,
            25,
            28,
            31,
            34,
            37,
            40,
            43,
            46,
            49,
            55,
            61,
            67,
            73,
            79,
            85,
            91,
            97,
        ]
    )  # 定义期望的结果 Index 对象
    tm.assert_index_equal(result, expected)  # 验证结果是否与期望一致


def test_union_with_duplicate_index_not_subset_and_non_monotonic(
    any_dtype_for_small_pos_integer_indexes,
):
    # 测试包含重复索引的 union 操作，且不属于子集，且不是单调的情况
    dtype = any_dtype_for_small_pos_integer_indexes  # 获取数据类型
    a = Index([1, 0, 2], dtype=dtype)  # 创建第一个 Index 对象
    b = Index([0, 0, 1], dtype=dtype)  # 创建第二个 Index 对象
    expected = Index([0, 0, 1, 2], dtype=dtype)  # 定义期望的结果 Index 对象
    if isinstance(a, CategoricalIndex):
        expected = Index([0, 0, 1, 2])  # 如果是 CategoricalIndex，调整期望结果

    result = a.union(b)  # 执行 union 操作
    tm.assert_index_equal(result, expected)  # 验证结果是否与期望一致

    result = b.union(a)  # 执行 b 与 a 的 union 操作
    tm.assert_index_equal(result, expected)  # 验证结果是否与期望一致


def test_union_int_categorical_with_nan():
    # 测试整数索引和包含 NaN 的 CategoricalIndex 的 union 操作
    ci = CategoricalIndex([1, 2, np.nan])  # 创建包含 NaN 的 CategoricalIndex
    assert ci.categories.dtype.kind == "i"  # 确保类别数据类型为整数

    idx = Index([1, 2])  # 创建一个整数 Index 对象

    result = idx.union(ci)  # 执行 Index 与 CategoricalIndex 的 union 操作
    expected = Index([1, 2, np.nan], dtype=np.float64)  # 定义期望的结果 Index 对象，数据类型为 float64
    tm.assert_index_equal(result, expected)  # 验证结果是否与期望一致

    result = ci.union(idx)  # 执行 CategoricalIndex 与 Index 的 union 操作
    tm.assert_index_equal(result, expected)  # 验证结果是否与期望一致


class TestSetOpsUnsorted:
    # 测试集合操作，未排序情况下的情况
    def test_intersect_str_dates(self):
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]  # 创建包含日期的列表

        index1 = Index(dt_dates, dtype=object)  # 创建第一个 Index 对象，数据类型为 object
        index2 = Index(["aa"], dtype=object)  # 创建第二个 Index 对象，包含字符串数据
        result = index2.intersection(index1)  # 执行 intersection 操作

        expected = Index([], dtype=object)  # 定义期望的结果 Index 对象，空集合
        tm.assert_index_equal(result, expected)  # 验证结果是否与期望一致

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_intersection(self, index, sort):
        first = index[:20]  # 获取 index 的前 20 个元素
        second = index[:10]  # 获取 index 的前 10 个元素
        intersect = first.intersection(second, sort=sort)  # 执行 intersection 操作，排序由 sort 参数决定
        if sort in (None, False):
            tm.assert_index_equal(intersect.sort_values(), second.sort_values())  # 如果未排序，验证排序后的结果与期望一致
        else:
            tm.assert_index_equal(intersect, second)  # 如果已排序，验证结果是否与期望一致

        # Corner cases
        inter = first.intersection(first, sort=sort)  # 交集操作的特殊情况：与自身的交集
        assert inter is first  # 验证结果是否与第一个 index 相同

    @pytest.mark.parametrize(
        "index2_name,keeps_name",
        [
            ("index", True),  # 保留相同名称
            ("other", False),  # 不保留不同名称
            (None, False),
        ],
    )
    def test_intersection_name_preservation(self, index2_name, keeps_name, sort):
        # 创建第二个索引对象，使用给定的名称
        index2 = Index([3, 4, 5, 6, 7], name=index2_name)
        # 创建第一个索引对象，默认名称为 "index"
        index1 = Index([1, 2, 3, 4, 5], name="index")
        # 预期的交集结果，包含公共元素 [3, 4, 5]
        expected = Index([3, 4, 5])
        # 计算两个索引对象的交集，并根据需要进行排序
        result = index1.intersection(index2, sort)

        # 如果需要保持名称一致，则将预期结果的名称设置为 "index"
        if keeps_name:
            expected.name = "index"

        # 断言交集结果的名称与预期相同
        assert result.name == expected.name
        # 断言交集结果与预期结果相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize(
        "first_name,second_name,expected_name",
        [("A", "A", "A"), ("A", "B", None), (None, "B", None)],
    )
    def test_intersection_name_preservation2(
        self, index, first_name, second_name, expected_name, sort
    ):
        # 获取第一个和第二个索引对象的子集
        first = index[5:20]
        second = index[:10]
        # 设置第一个和第二个索引对象的名称
        first.name = first_name
        second.name = second_name
        # 计算两个索引对象的交集，并根据需要进行排序
        intersect = first.intersection(second, sort=sort)
        # 断言交集结果的名称与预期相同
        assert intersect.name == expected_name

    def test_chained_union(self, sort):
        # Chained unions handles names correctly
        # 创建三个索引对象，分别指定名称
        i1 = Index([1, 2], name="i1")
        i2 = Index([5, 6], name="i2")
        i3 = Index([3, 4], name="i3")
        # 进行链式联合操作，并根据需要进行排序
        union = i1.union(i2.union(i3, sort=sort), sort=sort)
        # 预期的联合结果
        expected = i1.union(i2, sort=sort).union(i3, sort=sort)
        # 断言联合结果与预期结果相等
        tm.assert_index_equal(union, expected)

        # 创建三个索引对象，分别指定名称
        j1 = Index([1, 2], name="j1")
        j2 = Index([], name="j2")
        j3 = Index([], name="j3")
        # 进行链式联合操作，并根据需要进行排序
        union = j1.union(j2.union(j3, sort=sort), sort=sort)
        # 预期的联合结果
        expected = j1.union(j2, sort=sort).union(j3, sort=sort)
        # 断言联合结果与预期结果相等
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union(self, index, sort):
        # 获取第一个和第二个索引对象的子集
        first = index[5:20]
        second = index[:10]
        everything = index[:20]

        # 计算两个索引对象的联合，并根据需要进行排序
        union = first.union(second, sort=sort)
        # 根据排序需求断言联合结果
        if sort in (None, False):
            tm.assert_index_equal(union.sort_values(), everything.sort_values())
        else:
            tm.assert_index_equal(union, everything)

    @pytest.mark.parametrize("klass", [np.array, Series, list])
    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union_from_iterables(self, index, klass, sort):
        # GH#10149
        # 获取第一个和第二个索引对象的子集
        first = index[5:20]
        second = index[:10]
        everything = index[:20]

        # 将第二个索引对象转换为指定类的对象，并计算它与第一个索引对象的联合
        case = klass(second.values)
        result = first.union(case, sort=sort)
        # 根据排序需求断言联合结果
        if sort in (None, False):
            tm.assert_index_equal(result.sort_values(), everything.sort_values())
        else:
            tm.assert_index_equal(result, everything)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    # 在给定索引的切片中选取从第5到第20个元素，并赋值给变量first
    first = index[5:20]

    # 使用first集合自身的并集（union），根据参数sort的值进行排序或不排序
    union = first.union(first, sort=sort)
    # 检查是否满足集合的恒等性，即在sort为True时，返回的union与first不是同一个对象
    assert (union is first) is (not sort)

    # 对于空的Index集合，因为数据类型不一致，将两个对象都重新转换为dtype('O')
    union = first.union(Index([], dtype=first.dtype), sort=sort)
    # 检查是否满足集合的恒等性，即在sort为True时，返回的union与first不是同一个对象
    assert (union is first) is (not sort)

    # 创建一个空的Index集合，并与first集合进行并集操作，根据sort参数决定是否排序
    union = Index([], dtype=first.dtype).union(first, sort=sort)
    # 检查是否满足集合的恒等性，即在sort为True时，返回的union与first不是同一个对象
    assert (union is first) is (not sort)

@pytest.mark.parametrize("index", ["string"], indirect=True)
@pytest.mark.parametrize("second_name,expected", [(None, None), ("name", "name")])
def test_difference_name_preservation(self, index, second_name, expected, sort):
    # 从给定索引中选取第5到第20个元素，并赋值给变量first
    first = index[5:20]
    # 从给定索引中选取前10个元素，并赋值给变量second
    second = index[:10]
    # 从给定索引中选取第10到第20个元素，并赋值给变量answer
    answer = index[10:20]

    # 为first集合设置名称为"name"
    first.name = "name"
    # 从first集合中减去second集合的元素，根据sort参数决定是否排序
    result = first.difference(second, sort=sort)

    # 如果sort为True，验证result与answer集合相等
    if sort is True:
        tm.assert_index_equal(result, answer)
    else:
        # 否则，验证经过排序后的result与answer相等，并保持名称一致
        answer.name = second_name
        tm.assert_index_equal(result.sort_values(), answer.sort_values())

    # 如果expected为None，验证result的名称为空
    if expected is None:
        assert result.name is None
    else:
        # 否则，验证result的名称与expected相等
        assert result.name == expected

def test_difference_empty_arg(self, index, sort):
    # 创建index的副本，并从中选取第5到第20个元素，然后设置名称为"name"，赋值给变量first
    first = index.copy()
    first = first[5:20]
    first.name = "name"
    # 从first集合中减去空集合，根据sort参数决定是否排序
    result = first.difference([], sort)
    # 创建index中第5到第20个唯一元素的集合，并设置名称为"name"，赋值给变量expected
    expected = index[5:20].unique()
    expected.name = "name"
    # 验证result与expected集合相等
    tm.assert_index_equal(result, expected)

def test_difference_should_not_compare(self):
    # 创建包含重复元素的Index集合left和包含单个元素的Index集合right
    left = Index([1, 1])
    right = Index([True])
    # 从left集合中减去right集合的元素，赋值给变量result
    result = left.difference(right)
    # 创建包含单个元素的Index集合expected
    expected = Index([1])
    # 验证result与expected集合相等
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize("index", ["string"], indirect=True)
def test_difference_identity(self, index, sort):
    # 从给定索引中选取第5到第20个元素，并赋值给变量first
    first = index[5:20]
    # 为first集合设置名称为"name"
    first.name = "name"
    # 从first集合中减去first集合的元素，根据sort参数决定是否排序
    result = first.difference(first, sort)

    # 验证result的长度为0，并且其名称与first的名称相等
    assert len(result) == 0
    assert result.name == first.name

@pytest.mark.parametrize("index", ["string"], indirect=True)
def test_difference_sort(self, index, sort):
    # 从给定索引中选取第5到第20个元素，并赋值给变量first
    first = index[5:20]
    # 从给定索引中选取前10个元素，并赋值给变量second
    second = index[:10]

    # 从first集合中减去second集合的元素，根据sort参数决定是否排序
    result = first.difference(second, sort)
    # 创建index中第10到第20个元素的集合，并赋值给变量expected
    expected = index[10:20]

    # 如果sort为None，对expected进行排序
    if sort is None:
        expected = expected.sort_values()

    # 验证result与expected集合相等
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize("opname", ["difference", "symmetric_difference"])
    # 定义一个测试方法，用于测试无法比较的情况下的操作
    def test_difference_incomparable(self, opname):
        # 创建包含不同类型数据的索引 a 和 b
        a = Index([3, Timestamp("2000"), 1])
        b = Index([2, Timestamp("1999"), 1])
        # 创建一个操作对象，使用 operator.methodcaller 方法调用 opname 操作并应用于 b
        op = operator.methodcaller(opname, b)

        # 断言操作会产生 RuntimeWarning 警告，并匹配指定的警告消息
        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            # 执行操作 op 在索引 a 上，得到结果 result
            result = op(a)
        # 预期的结果索引
        expected = Index([3, Timestamp("2000"), 2, Timestamp("1999")])
        # 如果操作名称为 "difference"，则预期结果只包含前两个元素
        if opname == "difference":
            expected = expected[:2]
        # 断言结果 result 与预期结果 expected 相等
        tm.assert_index_equal(result, expected)

        # 创建一个操作对象，使用 operator.methodcaller 方法调用 opname 操作并应用于 b，同时禁用排序
        op = operator.methodcaller(opname, b, sort=False)
        # 执行操作 op 在索引 a 上，得到结果 result
        result = op(a)
        # 断言结果 result 与预期结果 expected 相等
        tm.assert_index_equal(result, expected)

    # 使用参数化装饰器，定义一个测试方法，测试无法比较的情况下的操作（设置为 True 的情况）
    @pytest.mark.parametrize("opname", ["difference", "symmetric_difference"])
    def test_difference_incomparable_true(self, opname):
        # 创建包含不同类型数据的索引 a 和 b
        a = Index([3, Timestamp("2000"), 1])
        b = Index([2, Timestamp("1999"), 1])
        # 创建一个操作对象，使用 operator.methodcaller 方法调用 opname 操作并应用于 b，并强制排序
        op = operator.methodcaller(opname, b, sort=True)

        # 预期的错误消息
        msg = "'<' not supported between instances of 'Timestamp' and 'int'"
        # 断言操作会产生 TypeError 异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 执行操作 op 在索引 a 上，期望抛出 TypeError
            op(a)

    # 定义一个测试方法，测试多重索引的对称差异操作
    def test_symmetric_difference_mi(self, sort):
        # 创建两个多重索引对象
        index1 = MultiIndex.from_tuples(zip(["foo", "bar", "baz"], [1, 2, 3]))
        index2 = MultiIndex.from_tuples([("foo", 1), ("bar", 3)])
        # 执行 index1 和 index2 的对称差异操作，并根据 sort 参数进行排序
        result = index1.symmetric_difference(index2, sort=sort)
        # 预期的多重索引结果
        expected = MultiIndex.from_tuples([("bar", 2), ("baz", 3), ("bar", 3)])
        # 如果 sort 参数为 None，则对预期结果进行排序
        if sort is None:
            expected = expected.sort_values()
        # 断言结果 result 与预期结果 expected 相等
        tm.assert_index_equal(result, expected)

    # 使用参数化装饰器，定义一个测试方法，测试对称差异操作中处理缺失值的情况
    @pytest.mark.parametrize(
        "index2,expected",
        [
            ([0, 1, np.nan], [2.0, 3.0, 0.0]),
            ([0, 1], [np.nan, 2.0, 3.0, 0.0]),
        ],
    )
    def test_symmetric_difference_missing(self, index2, expected, sort):
        # 创建索引对象 index2 和预期的索引对象 expected
        index2 = Index(index2)
        expected = Index(expected)
        # 创建索引对象 index1，包含整数和 NaN 值
        index1 = Index([1, np.nan, 2, 3])

        # 执行 index1 和 index2 的对称差异操作，并根据 sort 参数进行排序
        result = index1.symmetric_difference(index2, sort=sort)
        # 如果 sort 参数为 None，则对预期结果进行排序
        if sort is None:
            expected = expected.sort_values()
        # 断言结果 result 与预期结果 expected 相等
        tm.assert_index_equal(result, expected)
    # 定义一个测试函数，用于测试非索引的对称差集操作
    def test_symmetric_difference_non_index(self, sort):
        # 创建第一个索引对象，包含整数元素，指定名称为 "index1"
        index1 = Index([1, 2, 3, 4], name="index1")
        # 创建一个包含整数元素的 NumPy 数组作为第二个索引对象
        index2 = np.array([2, 3, 4, 5])
        # 创建预期结果的索引对象，包含元素 [1, 5]，名称为 "index1"
        expected = Index([1, 5], name="index1")
        # 对第一个索引对象和第二个数组执行对称差集操作，根据参数 sort 进行排序
        result = index1.symmetric_difference(index2, sort=sort)
        # 如果 sort 是 None 或 True，则断言结果与预期相等
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            # 否则，断言排序后的结果与预期相等
            tm.assert_index_equal(result.sort_values(), expected)
        # 断言结果对象的名称为 "index1"
        assert result.name == "index1"

        # 对第一个索引对象和第二个数组执行对称差集操作，并指定结果对象的名称为 "new_name"
        result = index1.symmetric_difference(index2, result_name="new_name", sort=sort)
        # 更新预期结果对象的名称为 "new_name"
        expected.name = "new_name"
        # 如果 sort 是 None 或 True，则断言结果与更新后的预期相等
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            # 否则，断言排序后的结果与更新后的预期相等
            tm.assert_index_equal(result.sort_values(), expected)
        # 断言更新后结果对象的名称为 "new_name"

    # 定义一个测试函数，用于测试包含任意数值型和 Arrow 数据类型的索引对象的并集操作
    def test_union_ea_dtypes(self, any_numeric_ea_and_arrow_dtype):
        # 创建包含数值型和 Arrow 数据类型的索引对象，元素为 [1, 2, 3]
        idx = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
        # 创建第二个相同类型的索引对象，元素为 [3, 4, 5]
        idx2 = Index([3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
        # 对这两个索引对象执行并集操作
        result = idx.union(idx2)
        # 创建预期结果的索引对象，包含元素 [1, 2, 3, 4, 5]，并且类型与输入的类型相同
        expected = Index([1, 2, 3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)

    # 定义一个测试函数，用于测试包含任意字符串数据类型的索引对象的并集操作
    def test_union_string_array(self, any_string_dtype):
        # 创建包含字符串类型的索引对象，元素为 ["a"]
        idx1 = Index(["a"], dtype=any_string_dtype)
        # 创建第二个相同类型的索引对象，元素为 ["b"]
        idx2 = Index(["b"], dtype=any_string_dtype)
        # 对这两个索引对象执行并集操作
        result = idx1.union(idx2)
        # 创建预期结果的索引对象，包含元素 ["a", "b"]，并且类型与输入的类型相同
        expected = Index(["a", "b"], dtype=any_string_dtype)
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)
```