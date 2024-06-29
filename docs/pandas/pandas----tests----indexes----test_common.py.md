# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_common.py`

```
"""
Collection of tests asserting things that should be true for
any index subclass except for MultiIndex. Makes use of the `index_flat`
fixture defined in pandas/conftest.py.
"""

# 导入必要的模块和库
from copy import (
    copy,  # 导入 copy 函数
    deepcopy,  # 导入 deepcopy 函数
)
import re  # 导入 re 模块

import numpy as np  # 导入 numpy 库并命名为 np
import pytest  # 导入 pytest 测试框架

from pandas.compat import IS64  # 导入 IS64 变量
from pandas.compat.numpy import np_version_gte1p25  # 导入 np_version_gte1p25 变量

from pandas.core.dtypes.common import (
    is_integer_dtype,  # 导入 is_integer_dtype 函数
    is_numeric_dtype,  # 导入 is_numeric_dtype 函数
)

import pandas as pd  # 导入 pandas 库并命名为 pd
from pandas import (
    CategoricalIndex,  # 导入 CategoricalIndex 类
    MultiIndex,  # 导入 MultiIndex 类
    PeriodIndex,  # 导入 PeriodIndex 类
    RangeIndex,  # 导入 RangeIndex 类
)
import pandas._testing as tm  # 导入 pandas 测试模块


class TestCommon:
    @pytest.mark.parametrize("name", [None, "new_name"])
    def test_to_frame(self, name, index_flat):
        # see GH#15230, GH#22580
        idx = index_flat  # 使用 index_flat fixture 创建 idx 变量

        if name:
            idx_name = name  # 如果 name 存在，则使用 name 作为 idx_name
        else:
            idx_name = idx.name or 0  # 否则使用 idx 的名称或默认值 0 作为 idx_name

        df = idx.to_frame(name=idx_name)  # 将 idx 转换为 DataFrame 对象

        assert df.index is idx  # 断言 DataFrame 的索引与 idx 是同一个对象
        assert len(df.columns) == 1  # 断言 DataFrame 的列数为 1
        assert df.columns[0] == idx_name  # 断言 DataFrame 的列名为 idx_name

        df = idx.to_frame(index=False, name=idx_name)  # 将 idx 转换为 DataFrame 对象，不包括索引

        assert df.index is not idx  # 断言 DataFrame 的索引不是 idx

    def test_droplevel(self, index_flat):
        # GH 21115
        # MultiIndex is tested separately in test_multi.py
        index = index_flat  # 使用 index_flat fixture 创建 index 变量

        assert index.droplevel([]).equals(index)  # 断言删除空的级别后，index 仍然相等

        for level in [index.name, [index.name]]:
            if isinstance(index.name, tuple) and level is index.name:
                # GH 21121 : droplevel with tuple name
                continue
            msg = (
                "Cannot remove 1 levels from an index with 1 levels: at least one "
                "level must be left."
            )
            with pytest.raises(ValueError, match=msg):
                index.droplevel(level)

        for level in "wrong", ["wrong"]:
            with pytest.raises(
                KeyError,
                match=r"'Requested level \(wrong\) does not match index name \(None\)'",
            ):
                index.droplevel(level)

    def test_constructor_non_hashable_name(self, index_flat):
        # GH 20527
        index = index_flat  # 使用 index_flat fixture 创建 index 变量

        message = "Index.name must be a hashable type"
        renamed = [["1"]]

        # With .rename()
        with pytest.raises(TypeError, match=message):
            index.rename(name=renamed)

        # With .set_names()
        with pytest.raises(TypeError, match=message):
            index.set_names(names=renamed)

    def test_constructor_unwraps_index(self, index_flat):
        a = index_flat  # 使用 index_flat fixture 创建 a 变量
        # Passing dtype is necessary for Index([True, False], dtype=object)
        #  case.
        b = type(a)(a, dtype=a.dtype)  # 创建 b 变量，并传递 dtype 参数以处理特定情况
        tm.assert_equal(a._data, b._data)  # 使用测试模块 tm 断言 a 和 b 的数据相等

    def test_to_flat_index(self, index_flat):
        # 22866
        index = index_flat  # 使用 index_flat fixture 创建 index 变量

        result = index.to_flat_index()  # 将 index 转换为平坦的索引

        tm.assert_index_equal(result, index)  # 使用测试模块 tm 断言 result 和 index 相等
    # 测试设置索引名称的方法，用于 MultiIndex 的单元测试
    def test_set_name_methods(self, index_flat):
        # 复制索引以便后续测试
        index = index_flat
        # 新的索引名称
        new_name = "This is the new name for this index"

        # 保存原始索引名称
        original_name = index.name
        # 使用新名称设置索引，并返回新的索引对象
        new_ind = index.set_names([new_name])
        # 断言新索引的名称等于新名称
        assert new_ind.name == new_name
        # 断言原始索引的名称未变
        assert index.name == original_name
        # 在原地重命名索引，返回值应为 None
        res = index.rename(new_name, inplace=True)
        assert res is None
        # 断言索引的名称已经更新为新名称
        assert index.name == new_name
        # 断言索引的所有名称为一个包含新名称的列表
        assert index.names == [new_name]
        # 使用 pytest 断言，设置名称时如果 level 参数不为 None 应抛出 ValueError 异常
        with pytest.raises(ValueError, match="Level must be None"):
            index.set_names("a", level=0)

        # 在原地重命名索引，但是只保留元组和其他容器的名称不变
        name = ("A", "B")
        index.rename(name, inplace=True)
        assert index.name == name
        assert index.names == [name]

    # 标记为预期失败的测试用例，测试设置单一标签名称且没有 level 参数时的情况
    @pytest.mark.xfail
    def test_set_names_single_label_no_level(self, index_flat):
        # 使用 pytest 断言，设置名称时应该抛出 TypeError 异常，提示需要 list-like 类型
        with pytest.raises(TypeError, match="list-like"):
            index_flat.set_names("a")

    # 测试复制和深复制操作
    def test_copy_and_deepcopy(self, index_flat):
        # 复制索引以便后续测试
        index = index_flat

        # 对于每个复制函数（copy 和 deepcopy）
        for func in (copy, deepcopy):
            # 使用函数复制索引，并断言复制后的对象与原始对象不同
            idx_copy = func(index)
            assert idx_copy is not index
            # 断言复制后的对象与原始对象相等
            assert idx_copy.equals(index)

        # 深复制索引，并设置新的名称为 "banana"
        new_copy = index.copy(deep=True, name="banana")
        # 断言新复制对象的名称为 "banana"
        assert new_copy.name == "banana"

    # 测试复制索引并设置名称的情况
    def test_copy_name(self, index_flat):
        # GH#12309: 检查初始化时传递的 "name" 参数是否被正确应用
        index = index_flat

        # 使用 "copy=True" 和指定的名称 "mario" 复制索引
        first = type(index)(index, copy=True, name="mario")
        # 使用 "copy=False" 复制索引
        second = type(first)(first, copy=False)

        # 即使 "copy=False"，我们也希望得到一个新的对象
        assert first is not second
        # 使用 tm.assert_index_equal() 断言两个索引对象相等
        tm.assert_index_equal(first, second)

        # 验证索引对象与原始索引相等
        assert index.equals(first)

        # 断言第一个索引对象的名称为 "mario"
        assert first.name == "mario"
        # 断言第二个索引对象的名称为 "mario"
        assert second.name == "mario"

        # TODO: 属于系列算术测试？
        # 创建两个系列对象，并进行乘法运算
        s1 = pd.Series(2, index=first)
        s2 = pd.Series(3, index=second[:-1])
        # 查看 GH#13365
        s3 = s1 * s2
        # 断言结果系列的索引名称为 "mario"
        assert s3.index.name == "mario"

    # 另一个测试复制并设置名称的情况
    def test_copy_name2(self, index_flat):
        # GH#35592
        index = index_flat

        # 断言复制后的索引对象名称为 "mario"
        assert index.copy(name="mario").name == "mario"

        # 使用 pytest 断言，如果新名称的长度不为 1，则应抛出 ValueError 异常
        with pytest.raises(ValueError, match="Length of new names must be 1, got 2"):
            index.copy(name=["mario", "luigi"])

        # 使用 pytest 断言，如果名称不是可散列类型，应抛出 TypeError 异常
        msg = f"{type(index).__name__}.name must be a hashable type"
        with pytest.raises(TypeError, match=msg):
            index.copy(name=[["mario"]])
    # 定义测试方法，测试索引的唯一性（对扁平化索引进行测试）
    def test_unique_level(self, index_flat):
        # 不对 MultiIndex 进行测试（因为它们是分开测试的）
        index = index_flat

        # GH 17896
        # 期望结果是去除重复值后的索引
        expected = index.drop_duplicates()
        
        # 对指定的多个级别进行唯一性测试
        for level in [0, index.name, None]:
            result = index.unique(level=level)
            # 使用测试框架验证结果索引与期望的索引是否相等
            tm.assert_index_equal(result, expected)

        # 测试超出索引级别范围的情况
        msg = "Too many levels: Index has only 1 level, not 4"
        with pytest.raises(IndexError, match=msg):
            index.unique(level=3)

        # 测试请求不存在的索引级别的情况
        msg = (
            rf"Requested level \(wrong\) does not match index name "
            rf"\({re.escape(index.name.__repr__())}\)"
        )
        with pytest.raises(KeyError, match=msg):
            index.unique(level="wrong")

    # 定义另一个测试方法，测试索引的唯一性
    def test_unique(self, index_flat):
        # 单独对 MultiIndex 进行测试
        index = index_flat
        
        # 如果索引长度为零，则跳过测试
        if not len(index):
            pytest.skip("Skip check for empty Index and MultiIndex")

        # 从索引中选取重复值作为测试目标
        idx = index[[0] * 5]
        idx_unique = index[[0]]

        # 确保 `idx_unique` 是唯一的且不包含 NaN
        assert idx_unique.is_unique is True
        try:
            assert idx_unique.hasnans is False
        except NotImplementedError:
            pass

        # 进行唯一性测试，并验证结果与 `idx_unique` 是否相等
        result = idx.unique()
        tm.assert_index_equal(result, idx_unique)

        # 如果索引不能包含 NaN，则跳过 NaN 检查
        if not index._can_hold_na:
            pytest.skip("Skip na-check if index cannot hold na")

        # 创建一个包含 NaN 的索引值数组
        vals = index._values[[0] * 5]
        vals[0] = np.nan

        # 选择部分唯一值，并创建包含 NaN 的索引副本
        vals_unique = vals[:2]
        idx_nan = index._shallow_copy(vals)
        idx_unique_nan = index._shallow_copy(vals_unique)
        assert idx_unique_nan.is_unique is True

        # 确保索引和包含 NaN 的索引的数据类型与原索引一致
        assert idx_nan.dtype == index.dtype
        assert idx_unique_nan.dtype == index.dtype

        # 期望的唯一索引是包含 NaN 的索引
        expected = idx_unique_nan
        # 对多个索引执行唯一性测试，并使用测试框架验证结果
        for pos, i in enumerate([idx_nan, idx_unique_nan]):
            result = i.unique()
            tm.assert_index_equal(result, expected)

    # 使用 pytest 的标记忽略特定的警告信息
    @pytest.mark.filterwarnings("ignore:Period with BDay freq:FutureWarning")
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 定义一个单元测试方法，用于测试搜索排序功能，针对单调性索引
    def test_searchsorted_monotonic(self, index_flat, request):
        # GH17271: 该注释指出此处可能是针对 GitHub 上的 issue 或 bug 的修复或测试
        index = index_flat
        # 如果索引是区间索引（IntervalIndex），则标记测试为预期失败，并添加原因说明
        if isinstance(index, pd.IntervalIndex):
            mark = pytest.mark.xfail(
                reason="IntervalIndex.searchsorted does not support Interval arg",
                raises=NotImplementedError,
            )
            request.applymarker(mark)

        # 如果索引为空，则跳过测试，因为无法测试空索引
        if index.empty:
            pytest.skip("Skip check for empty Index")
        # 获取索引的第一个值作为测试值
        value = index[0]

        # 确定预期的搜索结果（处理右侧重复值）
        expected_left, expected_right = 0, (index == value).argmin()
        if expected_right == 0:
            # 如果所有值都相同，则预期右侧结果应为索引长度
            expected_right = len(index)

        # 如果索引是单调递增或单调递减的
        if index.is_monotonic_increasing or index.is_monotonic_decreasing:
            # 测试单调性索引的左侧搜索
            ssm_left = index._searchsorted_monotonic(value, side="left")
            assert expected_left == ssm_left

            # 测试单调性索引的右侧搜索
            ssm_right = index._searchsorted_monotonic(value, side="right")
            assert expected_right == ssm_right

            # 使用普通搜索方法测试左侧搜索
            ss_left = index.searchsorted(value, side="left")
            assert expected_left == ss_left

            # 使用普通搜索方法测试右侧搜索
            ss_right = index.searchsorted(value, side="right")
            assert expected_right == ss_right
        else:
            # 对于非单调性的索引，应当抛出 ValueError 异常
            msg = "index must be monotonic increasing or decreasing"
            with pytest.raises(ValueError, match=msg):
                index._searchsorted_monotonic(value, side="left")

    # 添加一个标记忽略警告，特指 PeriodDtype[B] 的过时警告
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 定义一个测试方法，用于测试去除重复项功能，对于扁平化索引（index_flat），和保留策略（keep）进行测试
    def test_drop_duplicates(self, index_flat, keep):
        # 对于多级索引（MultiIndex），进行单独测试
        index = index_flat
        
        # 如果索引是 RangeIndex 类型，则跳过测试，因为 RangeIndex 无法包含重复项
        if isinstance(index, RangeIndex):
            pytest.skip(
                "RangeIndex is tested in test_drop_duplicates_no_duplicates "
                "as it cannot hold duplicates"
            )
        
        # 如果索引长度为 0，则跳过测试，因为空索引无法包含重复项
        if len(index) == 0:
            pytest.skip(
                "empty index is tested in test_drop_duplicates_no_duplicates "
                "as it cannot hold duplicates"
            )

        # 创建一个唯一的索引
        holder = type(index)
        unique_values = list(set(index))
        dtype = index.dtype if is_numeric_dtype(index) else None
        unique_idx = holder(unique_values, dtype=dtype)

        # 创建一个含有重复项的索引
        n = len(unique_idx)
        duplicated_selection = np.random.default_rng(2).choice(n, int(n * 1.5))
        idx = holder(unique_idx.values[duplicated_selection])

        # 对于 Series.duplicated 进行单独测试
        expected_duplicated = (
            pd.Series(duplicated_selection).duplicated(keep=keep).values
        )
        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected_duplicated)

        # 对于 Series.drop_duplicates 进行单独测试
        expected_dropped = holder(pd.Series(idx).drop_duplicates(keep=keep))
        tm.assert_index_equal(idx.drop_duplicates(keep=keep), expected_dropped)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 定义一个测试方法，用于测试没有重复项的情况（test_drop_duplicates_no_duplicates）
    def test_drop_duplicates_no_duplicates(self, index_flat):
        # 对于多级索引（MultiIndex），进行单独测试
        index = index_flat

        # 如果索引是 RangeIndex 类型，则唯一索引即为该索引本身，因为 RangeIndex 无法包含重复项
        if isinstance(index, RangeIndex):
            unique_idx = index
        else:
            holder = type(index)
            unique_values = list(set(index))
            dtype = index.dtype if is_numeric_dtype(index) else None
            unique_idx = holder(unique_values, dtype=dtype)

        # 验证唯一索引中没有重复项
        expected_duplicated = np.array([False] * len(unique_idx), dtype="bool")
        tm.assert_numpy_array_equal(unique_idx.duplicated(), expected_duplicated)
        
        # 对唯一索引进行去重操作，并验证结果索引与原索引相等
        result_dropped = unique_idx.drop_duplicates()
        tm.assert_index_equal(result_dropped, unique_idx)
        
        # 验证返回的去重后索引对象是原索引对象的浅拷贝
        assert result_dropped is not unique_idx

    # 定义一个测试方法，用于测试 inplace 参数为 True 时的情况（test_drop_duplicates_inplace）
    def test_drop_duplicates_inplace(self, index):
        msg = r"drop_duplicates\(\) got an unexpected keyword argument"
        
        # 当 inplace 参数为 True 时，验证会抛出 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            index.drop_duplicates(inplace=True)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 测试是否存在重复值的方法，针对扁平化索引进行测试
    def test_has_duplicates(self, index_flat):
        # MultiIndex 在以下测试中单独进行测试：
        #   tests/indexes/multi/test_unique_and_duplicates.
        
        # 将传入的索引赋值给 index 变量
        index = index_flat
        
        # 获取索引对象的类型
        holder = type(index)
        
        # 如果索引长度为零或者是 RangeIndex 类型，则跳过测试
        if not len(index) or isinstance(index, RangeIndex):
            # MultiIndex 在以下测试中单独进行测试：
            #   tests/indexes/multi/test_unique_and_duplicates.
            # RangeIndex 根据定义是唯一的索引
            pytest.skip("跳过空索引、MultiIndex和RangeIndex的检查")

        # 创建一个新的索引对象，使用第一个元素重复 5 次
        idx = holder([index[0]] * 5)
        
        # 断言该索引对象不是唯一的
        assert idx.is_unique is False
        
        # 断言该索引对象存在重复值
        assert idx.has_duplicates is True

    # 参数化测试方法，测试数据类型转换时是否保留名称
    @pytest.mark.parametrize(
        "dtype",
        ["int64", "uint64", "float64", "category", "datetime64[ns]", "timedelta64[ns]"],
    )
    def test_astype_preserves_name(self, index, dtype):
        # 参考 GitHub 上的 issue，确保转换时保留名称
        # https://github.com/pandas-dev/pandas/issues/32013
        
        # 如果索引是 MultiIndex 类型，则设置每级索引的名称为 "idx0", "idx1", ...
        if isinstance(index, MultiIndex):
            index.names = ["idx" + str(i) for i in range(index.nlevels)]
        else:
            # 否则设置索引的名称为 "idx"
            index.name = "idx"

        # 初始化警告信息变量
        warn = None
        
        # 如果索引的数据类型是复数类型且目标转换类型为 float64、int64、uint64，则警告丢弃虚部
        if index.dtype.kind == "c" and dtype in ["float64", "int64", "uint64"]:
            # 如果 NumPy 版本 >= 1.25，则使用 np.exceptions.ComplexWarning
            if np_version_gte1p25:
                warn = np.exceptions.ComplexWarning
            else:
                # 否则使用 np.ComplexWarning
                warn = np.ComplexWarning

        # 判断是否是 pyarrow 字符串类型并且目标类型为 category
        is_pyarrow_str = str(index.dtype) == "string[pyarrow]" and dtype == "category"
        
        try:
            # 尝试进行类型转换，捕获警告信息
            with tm.assert_produces_warning(
                warn,
                raise_on_extra_warnings=is_pyarrow_str,
                check_stacklevel=False,
            ):
                result = index.astype(dtype)
        except (ValueError, TypeError, NotImplementedError, SystemError):
            return

        # 如果索引是 MultiIndex 类型，则断言转换后的索引的名称与原索引的名称相同
        if isinstance(index, MultiIndex):
            assert result.names == index.names
        else:
            # 否则断言转换后的索引的名称与原索引的名称相同
            assert result.name == index.name

    # 测试方法，检查索引是否存在 NaN 值并返回对应的布尔值
    def test_hasnans_isnans(self, index_flat):
        # GH#11343，添加了 hasnans / isnans 的测试
        index = index_flat

        # 创建索引的深拷贝，命名为 idx
        idx = index.copy(deep=True)

        # 创建期望的布尔数组，表示索引中不包含 NaN 值
        expected = np.array([False] * len(idx), dtype=bool)

        # 断言索引的 _isnan 属性与预期数组相等
        tm.assert_numpy_array_equal(idx._isnan, expected)
        
        # 断言索引不包含 NaN 值
        assert idx.hasnans is False

        # 再次创建索引的深拷贝，命名为 idx
        idx = index.copy(deep=True)
        
        # 获取索引的值数组
        values = idx._values

        # 如果索引长度为 0，则直接返回
        if len(index) == 0:
            return
        # 如果索引的数据类型是整数类型，则直接返回
        elif is_integer_dtype(index.dtype):
            return
        # 如果索引的数据类型是布尔类型
        elif index.dtype == bool:
            # 设置索引值数组的第二个元素为 NaN，此时将会被转换为 True
            # 所以直接返回
            return

        # 将索引值数组的第二个元素设置为 NaN
        values[1] = np.nan

        # 使用索引的类型创建新的索引对象，其值为修改后的 values 数组
        idx = type(index)(values)

        # 创建预期的布尔数组，表示索引中存在 NaN 值
        expected = np.array([False] * len(idx), dtype=bool)
        expected[1] = True
        
        # 断言索引的 _isnan 属性与预期数组相等
        tm.assert_numpy_array_equal(idx._isnan, expected)
        
        # 断言索引包含 NaN 值
        assert idx.hasnans is True
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
# 使用 pytest 的 mark.filterwarnings 忽略特定的警告信息

@pytest.mark.parametrize("na_position", [None, "middle"])
# 使用 pytest 的 mark.parametrize 注入不同的 na_position 参数进行参数化测试
def test_sort_values_invalid_na_position(index_with_missing, na_position):
    # 测试函数：test_sort_values_invalid_na_position
    # 用于测试在排序中使用无效的 na_position 参数时是否引发 ValueError 异常

    with pytest.raises(ValueError, match=f"invalid na_position: {na_position}"):
        # 使用 pytest.raises 检查是否引发 ValueError 异常，并匹配异常信息
        index_with_missing.sort_values(na_position=na_position)


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize("na_position", ["first", "last"])
# 使用 pytest 的 mark.parametrize 注入不同的 na_position 参数进行参数化测试
def test_sort_values_with_missing(index_with_missing, na_position, request):
    # 测试函数：test_sort_values_with_missing
    # 用于测试 sort_values 函数在处理缺失值时的行为，根据不同的 na_position 参数排序

    # GH 35584. Test that sort_values works with missing values,
    # sort non-missing and place missing according to na_position
    # GH 35584. 测试 sort_values 函数在处理缺失值时的行为，对非缺失值排序，并根据 na_position 参数处理缺失值

    if isinstance(index_with_missing, CategoricalIndex):
        # 如果 index_with_missing 是 CategoricalIndex 类型，标记为预期失败（xfail）
        request.applymarker(
            pytest.mark.xfail(
                reason="missing value sorting order not well-defined", strict=False
            )
        )

    missing_count = np.sum(index_with_missing.isna())
    # 统计缺失值的数量
    not_na_vals = index_with_missing[index_with_missing.notna()].values
    # 获取非缺失值的数组
    sorted_values = np.sort(not_na_vals)
    # 对非缺失值进行排序

    if na_position == "first":
        sorted_values = np.concatenate([[None] * missing_count, sorted_values])
        # 如果 na_position 为 "first"，则将缺失值放在排序结果的开头
    else:
        sorted_values = np.concatenate([sorted_values, [None] * missing_count])
        # 否则将缺失值放在排序结果的末尾

    # 显式指定需要的 dtype，适用于由 EA 支持的 Index 类型，如 IntegerArray
    expected = type(index_with_missing)(sorted_values, dtype=index_with_missing.dtype)

    result = index_with_missing.sort_values(na_position=na_position)
    # 调用 sort_values 函数进行排序

    tm.assert_index_equal(result, expected)
    # 使用 tm.assert_index_equal 断言结果与预期相等


def test_sort_values_natsort_key():
    # 测试函数：test_sort_values_natsort_key
    # GH#56081

    def split_convert(s):
        return tuple(int(x) for x in s.split("."))
        # 定义一个函数 split_convert，将字符串 s 按 "." 分割并转换为元组

    idx = pd.Index(["1.9", "2.0", "1.11", "1.10"])
    # 创建一个包含字符串的 pandas Index 对象
    expected = pd.Index(["1.9", "1.10", "1.11", "2.0"])
    # 创建预期的排序结果的 pandas Index 对象

    result = idx.sort_values(key=lambda x: tuple(map(split_convert, x)))
    # 使用 sort_values 函数进行排序，根据 split_convert 函数的结果排序

    tm.assert_index_equal(result, expected)
    # 使用 tm.assert_index_equal 断言结果与预期相等


def test_ndarray_compat_properties(index):
    # 测试函数：test_ndarray_compat_properties

    if isinstance(index, PeriodIndex) and not IS64:
        pytest.skip("Overflow")
        # 如果 index 是 PeriodIndex 类型且不是 IS64，跳过测试（标记为 pytest.skip）

    idx = index
    # 将传入的 index 参数赋值给 idx 变量

    assert idx.T.equals(idx)
    # 断言索引的转置是否等于自身
    assert idx.transpose().equals(idx)
    # 断言索引的转置是否等于自身

    values = idx.values
    # 获取索引的值数组

    assert idx.shape == values.shape
    # 断言索引的形状与其值数组的形状相同
    assert idx.ndim == values.ndim
    # 断言索引的维度与其值数组的维度相同
    assert idx.size == values.size
    # 断言索引的大小与其值数组的大小相同

    if not isinstance(index, (RangeIndex, MultiIndex)):
        # 如果索引不是 RangeIndex 或 MultiIndex 类型
        assert idx.nbytes == values.nbytes
        # 断言索引的字节数与其值数组的字节数相同

    # test for validity
    # 验证索引的有效性
    idx.nbytes
    idx.values.nbytes


def test_compare_read_only_array():
    # 测试函数：test_compare_read_only_array
    # GH#57130

    arr = np.array([], dtype=object)
    # 创建一个空的 numpy 数组，dtype 为 object
    arr.flags.writeable = False
    # 将数组的写入标志设置为 False，使其为只读数组
    idx = pd.Index(arr)
    # 使用该数组创建一个 pandas Index 对象

    result = idx > 69
    # 比较索引对象中的值是否大于 69
    assert result.dtype == bool
    # 断言比较结果的 dtype 是否为布尔类型


def test_to_frame_column_rangeindex():
    # 测试函数：test_to_frame_column_rangeindex

    idx = pd.Index([1])
    # 创建一个包含整数 1 的 pandas Index 对象

    result = idx.to_frame().columns
    # 将索引对象转换为 DataFrame，并获取其列

    expected = RangeIndex(1)
    # 创建一个预期的 RangeIndex 对象，包含一个整数 1

    tm.assert_index_equal(result, expected, exact=True)
    # 使用 tm.assert_index_equal 断言结果与预期相等，严格匹配


def test_to_frame_name_tuple_multiindex():
    # 测试函数：test_to_frame_name_tuple_multiindex

    idx = pd.Index([1])
    # 创建一个包含整数 1 的 pandas Index 对象

    result = idx.to_frame(name=(1, 2))
    # 将索引对象转换为 DataFrame，并指定名称为元组 (1, 2)
    # 创建一个预期的 Pandas DataFrame 对象，包含单个数据值为 1，列索引为 MultiIndex 对象的数组，行索引为变量 idx 所表示的索引
    expected = pd.DataFrame([1], columns=MultiIndex.from_arrays([[1], [2]]), index=idx)
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
```