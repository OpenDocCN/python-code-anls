# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_loc.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas.errors import IndexingError  # 导入 pandas 的索引错误模块

import pandas as pd  # 导入 pandas 库，并命名为 pd
from pandas import (  # 从 pandas 中导入以下对象：
    DataFrame,  # 数据框对象
    Index,  # 索引对象
    MultiIndex,  # 多重索引对象
    Series,  # 系列对象
)
import pandas._testing as tm  # 导入 pandas 的测试模块


@pytest.fixture  # 定义一个 pytest 的测试夹具
def frame_random_data_integer_multi_index():
    levels = [[0, 1], [0, 1, 2]]  # 多重索引的层级
    codes = [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]  # 多重索引的编码
    index = MultiIndex(levels=levels, codes=codes)  # 创建多重索引对象
    return DataFrame(np.random.default_rng(2).standard_normal((6, 2)), index=index)
    # 返回一个带有随机数据的数据框，索引为多重索引对象


class TestMultiIndexLoc:  # 定义测试类 TestMultiIndexLoc
    def test_loc_setitem_frame_with_multiindex(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data  # 获取多重索引数据框
        frame.loc[("bar", "two"), "B"] = 5  # 设置特定位置的值为 5
        assert frame.loc[("bar", "two"), "B"] == 5  # 断言确保值已经设置成功

        # with integer labels
        df = frame.copy()  # 复制数据框
        df.columns = list(range(3))  # 修改列名为整数序列
        df.loc[("bar", "two"), 1] = 7  # 设置特定位置的值为 7
        assert df.loc[("bar", "two"), 1] == 7  # 断言确保值已经设置成功

    def test_loc_getitem_general(self, performance_warning, any_real_numpy_dtype):
        # GH#2817
        dtype = any_real_numpy_dtype  # 获取任意的 NumPy 数据类型
        data = {  # 定义数据字典
            "amount": {0: 700, 1: 600, 2: 222, 3: 333, 4: 444},
            "col": {0: 3.5, 1: 3.5, 2: 4.0, 3: 4.0, 4: 4.0},
            "num": {0: 12, 1: 11, 2: 12, 3: 12, 4: 12},
        }
        df = DataFrame(data)  # 创建数据框
        df = df.astype({"col": dtype, "num": dtype})  # 将列转换为指定的数据类型
        df = df.set_index(keys=["col", "num"])  # 设置列为索引
        key = 4.0, 12  # 定义一个键

        with tm.assert_produces_warning(performance_warning):  # 断言产生性能警告
            tm.assert_frame_equal(df.loc[key], df.iloc[2:])  # 断言两个数据框相等

        # this is ok
        return_value = df.sort_index(inplace=True)  # 按索引排序，原地排序
        assert return_value is None  # 断言返回值为 None
        res = df.loc[key]  # 获取特定索引位置的数据

        # col has float dtype, result should be float64 Index
        col_arr = np.array([4.0] * 3, dtype=dtype)  # 创建浮点数组
        year_arr = np.array([12] * 3, dtype=dtype)  # 创建整数数组
        index = MultiIndex.from_arrays([col_arr, year_arr], names=["col", "num"])  # 根据数组创建多重索引
        expected = DataFrame({"amount": [222, 333, 444]}, index=index)  # 创建预期的数据框
        tm.assert_frame_equal(res, expected)  # 断言两个数据框相等

    def test_loc_getitem_multiindex_missing_label_raises(self):
        # GH#21593
        df = DataFrame(  # 创建数据框
            np.random.default_rng(2).standard_normal((3, 3)),  # 随机数据
            columns=[[2, 2, 4], [6, 8, 10]],  # 列的多重索引
            index=[[4, 4, 8], [8, 10, 12]],  # 行的多重索引
        )

        with pytest.raises(KeyError, match=r"^2$"):  # 断言捕获 KeyError 异常
            df.loc[2]  # 尝试使用不存在的标签进行索引

    def test_loc_getitem_list_of_tuples_with_multiindex(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        ser = multiindex_year_month_day_dataframe_random_data["A"]  # 获取系列对象
        expected = ser.reindex(ser.index[49:51])  # 重新索引预期结果
        result = ser.loc[[(2000, 3, 10), (2000, 3, 13)]]  # 使用元组列表进行索引
        tm.assert_series_equal(result, expected)  # 断言两个系列对象相等
    # 测试函数，验证 loc 方法在处理 Series 对象时的行为
    def test_loc_getitem_series(self):
        # GH14730
        # 测试用例：将 Series 对象作为键传递给 loc 方法，该 Series 对象带有 MultiIndex
        index = MultiIndex.from_product([[1, 2, 3], ["A", "B", "C"]])
        x = Series(index=index, data=range(9), dtype=np.float64)
        y = Series([1, 3])
        expected = Series(
            data=[0, 1, 2, 6, 7, 8],
            index=MultiIndex.from_product([[1, 3], ["A", "B", "C"]]),
            dtype=np.float64,
        )
        result = x.loc[y]
        tm.assert_series_equal(result, expected)

        result = x.loc[[1, 3]]
        tm.assert_series_equal(result, expected)

        # GH15424
        # 测试用例：使用带有索引的 Series 对象作为键传递给 loc 方法
        y1 = Series([1, 3], index=[1, 2])
        result = x.loc[y1]
        tm.assert_series_equal(result, expected)

        # 测试用例：传递一个空的 Series 对象
        empty = Series(data=[], dtype=np.float64)
        expected = Series(
            [],
            index=MultiIndex(levels=index.levels, codes=[[], []], dtype=np.float64),
            dtype=np.float64,
        )
        result = x.loc[empty]
        tm.assert_series_equal(result, expected)

    # 测试函数，验证 loc 方法在处理数组时的行为
    def test_loc_getitem_array(self):
        # GH15434
        # 测试用例：将数组作为键传递给 loc 方法，该数组带有 MultiIndex
        index = MultiIndex.from_product([[1, 2, 3], ["A", "B", "C"]])
        x = Series(index=index, data=range(9), dtype=np.float64)
        y = np.array([1, 3])
        expected = Series(
            data=[0, 1, 2, 6, 7, 8],
            index=MultiIndex.from_product([[1, 3], ["A", "B", "C"]]),
            dtype=np.float64,
        )
        result = x.loc[y]
        tm.assert_series_equal(result, expected)

        # 空数组的测试用例:
        empty = np.array([])
        expected = Series(
            [],
            index=MultiIndex(levels=index.levels, codes=[[], []], dtype=np.float64),
            dtype="float64",
        )
        result = x.loc[empty]
        tm.assert_series_equal(result, expected)

        # 0 维数组（标量）的测试用例:
        scalar = np.int64(1)
        expected = Series(data=[0, 1, 2], index=["A", "B", "C"], dtype=np.float64)
        result = x.loc[scalar]
        tm.assert_series_equal(result, expected)

    # 测试函数，验证 loc 方法在处理多级索引标签时的行为
    def test_loc_multiindex_labels(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=[["i", "i", "j"], ["A", "A", "B"]],
            index=[["i", "i", "j"], ["X", "X", "Y"]],
        )

        # 测试用例：选取第一和第二行
        expected = df.iloc[[0, 1]].droplevel(0)
        result = df.loc["i"]
        tm.assert_frame_equal(result, expected)

        # 测试用例：选取第三列（最后一列）
        expected = df.iloc[:, [2]].droplevel(0, axis=1)
        result = df.loc[:, "j"]
        tm.assert_frame_equal(result, expected)

        # 测试用例：选取右下角的单元格
        expected = df.iloc[[2], [2]].droplevel(0).droplevel(0, axis=1)
        result = df.loc["j"].loc[:, "j"]
        tm.assert_frame_equal(result, expected)

        # 测试用例：使用元组进行选择
        expected = df.iloc[[0, 1]]
        result = df.loc[("i", "X")]
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试 MultiIndex 下的 loc 方法，验证定位和比较数据框的行为
    def test_loc_multiindex_ints(self):
        # 创建一个 DataFrame，填充随机标准正态分布的数据，设置列和行的多级索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=[[2, 2, 4], [6, 8, 10]],
            index=[[4, 4, 8], [8, 10, 12]],
        )
        # 期望的结果是通过 iloc 取第一行和第二行，然后去掉第一级索引
        expected = df.iloc[[0, 1]].droplevel(0)
        # 使用 loc 方法获取索引为 4 的所有行数据
        result = df.loc[4]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和期望，确认它们相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试在 MultiIndex 中使用 loc 方法时，对于缺失标签会抛出 KeyError 的情况
    def test_loc_multiindex_missing_label_raises(self):
        # 创建一个 DataFrame，填充随机标准正态分布的数据，设置列和行的多级索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=[[2, 2, 4], [6, 8, 10]],
            index=[[4, 4, 8], [8, 10, 12]],
        )

        # 使用 pytest 的 raises 方法捕获 KeyError 异常，并验证异常信息匹配 "^2$"
        with pytest.raises(KeyError, match=r"^2$"):
            df.loc[2]

    # 使用 pytest 的 parametrize 装饰器定义一个参数化测试方法，用于测试 MultiIndex 中列表缺失标签时的情况
    @pytest.mark.parametrize("key, pos", [([2, 4], [0, 1]), ([2], []), ([2, 3], [])])
    def test_loc_multiindex_list_missing_label(self, key, pos):
        # GH 27148 - lists with missing labels _do_ raise
        # 创建一个 DataFrame，填充随机标准正态分布的数据，设置列和行的多级索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=[[2, 2, 4], [6, 8, 10]],
            index=[[4, 4, 8], [8, 10, 12]],
        )

        # 使用 pytest 的 raises 方法捕获 KeyError 异常，并验证异常信息包含 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            df.loc[key]

    # 定义一个测试方法，用于测试 MultiIndex 中过多维度时的异常抛出情况
    def test_loc_multiindex_too_many_dims_raises(self):
        # GH 14885
        # 创建一个 Series，设置一个三级 MultiIndex
        s = Series(
            range(8),
            index=MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]]),
        )

        # 使用 pytest 的 raises 方法捕获 KeyError 异常，并验证异常信息匹配 "^\('a', 'b'\)$"
        with pytest.raises(KeyError, match=r"^\('a', 'b'\)$"):
            s.loc["a", "b"]
        
        # 使用 pytest 的 raises 方法捕获 KeyError 异常，并验证异常信息匹配 "^\('a', 'd', 'g'\)$"
        with pytest.raises(KeyError, match=r"^\('a', 'd', 'g'\)$"):
            s.loc["a", "d", "g"]
        
        # 使用 pytest 的 raises 方法捕获 IndexingError 异常，并验证异常信息包含 "Too many indexers"
        with pytest.raises(IndexingError, match="Too many indexers"):
            s.loc["a", "d", "g", "j"]

    # 定义一个测试方法，用于测试 MultiIndex 中索引器为 None 时的情况
    def test_loc_multiindex_indexer_none(self):
        # GH6788
        # 创建一个多级索引的 DataFrame，填充随机标准正态分布的数据
        # 设置一个 attributes 和 attribute_values 的多级索引
        attributes = ["Attribute" + str(i) for i in range(1)]
        attribute_values = ["Value" + str(i) for i in range(5)]

        index = MultiIndex.from_product([attributes, attribute_values])
        df = 0.1 * np.random.default_rng(2).standard_normal((10, 1 * 5)) + 0.5
        df = DataFrame(df, columns=index)
        # 使用 loc 方法获取所有行的特定列
        result = df[attributes]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和原始 DataFrame，确认它们相等
        tm.assert_frame_equal(result, df)

        # GH 7349
        # 创建一个 Series，设置一个二级 MultiIndex
        df = DataFrame(
            np.arange(12).reshape(-1, 1),
            index=MultiIndex.from_product([[1, 2, 3, 4], [1, 2, 3]]),
        )

        # 使用 loc 方法获取特定索引的数据，并与期望结果进行比较
        expected = df.loc[([1, 2],), :]
        result = df.loc[[1, 2]]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和期望，确认它们相等
        tm.assert_frame_equal(result, expected)
    def test_loc_multiindex_incomplete(self):
        # GH 7399
        # 处理不完整的索引器
        s = Series(
            np.arange(15, dtype="int64"),
            MultiIndex.from_product([range(5), ["a", "b", "c"]]),
        )
        expected = s.loc[:, "a":"c"]  # 期望的结果是选择所有行和列从 'a' 到 'c'

        result = s.loc[0:4, "a":"c"]  # 使用整数切片和列切片选择数据
        tm.assert_series_equal(result, expected)

        result = s.loc[:4, "a":"c"]  # 使用省略号和列切片选择数据
        tm.assert_series_equal(result, expected)

        result = s.loc[0:, "a":"c"]  # 使用整数切片和省略号和列切片选择数据
        tm.assert_series_equal(result, expected)

        # GH 7400
        # 使用多级索引进行索引，当使用列表索引器时跳过错误元素
        s = Series(
            np.arange(15, dtype="int64"),
            MultiIndex.from_product([range(5), ["a", "b", "c"]]),
        )
        expected = s.iloc[[6, 7, 8, 12, 13, 14]]  # 期望的结果是选择特定的索引位置

        result = s.loc[2:4:2, "a":"c"]  # 使用整数切片和列切片选择数据
        tm.assert_series_equal(result, expected)

    def test_get_loc_single_level(self):
        single_level = MultiIndex(
            levels=[["foo", "bar", "baz", "qux"]], codes=[[0, 1, 2, 3]], names=["first"]
        )
        s = Series(
            np.random.default_rng(2).standard_normal(len(single_level)),
            index=single_level,
        )
        for k in single_level.values:
            s[k]

    def test_loc_getitem_int_slice(self):
        # GH 3053
        # loc 应该像标签切片一样处理整数切片

        index = MultiIndex.from_product([[6, 7, 8], ["a", "b"]])
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 6)), index, index)
        result = df.loc[6:8, :]  # 使用整数切片和所有列选择数据
        expected = df  # 期望结果是整个 DataFrame
        tm.assert_frame_equal(result, expected)

        index = MultiIndex.from_product([[10, 20, 30], ["a", "b"]])
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 6)), index, index)
        result = df.loc[20:30, :]  # 使用整数切片和所有列选择数据
        expected = df.iloc[2:]  # 期望结果是从第三行开始的 DataFrame
        tm.assert_frame_equal(result, expected)

        # 文档示例
        result = df.loc[10, :]  # 使用标签和所有列选择数据
        expected = df.iloc[0:2]  # 期望结果是前两行的 DataFrame
        expected.index = ["a", "b"]  # 重设索引
        tm.assert_frame_equal(result, expected)

        result = df.loc[:, 10]  # 使用所有行和标签选择数据
        expected = df[10]  # 期望结果是特定标签列的 Series
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer_type_1", (list, tuple, set, slice, np.ndarray, Series, Index)
    )
    @pytest.mark.parametrize(
        "indexer_type_2", (list, tuple, set, slice, np.ndarray, Series, Index)
    )
    def test_loc_getitem_nested_indexer(self, indexer_type_1, indexer_type_2):
        # GH #19686
        # .loc should work with nested indexers which can be
        # any list-like objects (see `is_list_like` (`pandas.api.types`)) or slices

        # 定义一个函数，用于将嵌套的索引器转换为相应类型的索引器
        def convert_nested_indexer(indexer_type, keys):
            if indexer_type == np.ndarray:
                return np.array(keys)
            if indexer_type == slice:
                return slice(*keys)
            return indexer_type(keys)

        # 创建两个列表
        a = [10, 20, 30]  # 列表 a 包含整数 10, 20, 30
        b = [1, 2, 3]  # 列表 b 包含整数 1, 2, 3

        # 用 MultiIndex.from_product 创建一个多级索引对象 index
        index = MultiIndex.from_product([a, b])

        # 用 DataFrame 创建一个数据帧 df，其数据为递增的整数，索引为 index，列名为 "Data"
        df = DataFrame(
            np.arange(len(index), dtype="int64"), index=index, columns=["Data"]
        )

        # 定义一个元组 keys，其中包含两个子元组，每个子元组包含两个整数
        keys = ([10, 20], [2, 3])

        # 定义一个元组 types，包含两种索引器类型
        types = (indexer_type_1, indexer_type_2)

        # 检查所有组合的嵌套对象索引器是否有效
        indexer = tuple(
            convert_nested_indexer(indexer_type, k)
            for indexer_type, k in zip(types, keys)
        )

        # 如果任一索引器类型是 set，则引发 TypeError 异常，匹配字符串 "as an indexer is not supported"
        if indexer_type_1 is set or indexer_type_2 is set:
            with pytest.raises(TypeError, match="as an indexer is not supported"):
                df.loc[indexer, "Data"]
            return
        else:
            # 否则，将 df 根据 indexer 和列名 "Data" 进行索引，并将结果赋给 result
            result = df.loc[indexer, "Data"]

        # 创建一个预期的 Series 对象 expected，包含预期的数据和索引
        expected = Series(
            [1, 2, 4, 5], name="Data", index=MultiIndex.from_product(keys)
        )

        # 使用 tm.assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    def test_multiindex_loc_one_dimensional_tuple(self, frame_or_series):
        # GH#37711
        # 创建一个 MultiIndex 对象 mi，包含两个元组
        mi = MultiIndex.from_tuples([("a", "A"), ("b", "A")])
        
        # 根据 frame_or_series 创建一个对象 obj，索引为 mi，数据为 [1, 2]
        obj = frame_or_series([1, 2], index=mi)
        
        # 使用 loc 方法设置索引为 ("a",) 的行数据为 0
        obj.loc[("a",)] = 0
        
        # 创建一个预期的对象 expected，与 obj 相同索引，数据为 [0, 2]
        expected = frame_or_series([0, 2], index=mi)
        
        # 使用 tm.assert_equal 检查 obj 和 expected 是否相等
        tm.assert_equal(obj, expected)

    @pytest.mark.parametrize("indexer", [("a",), ("a")])
    def test_multiindex_one_dimensional_tuple_columns(self, indexer):
        # GH#37711
        # 创建一个 MultiIndex 对象 mi，包含两个元组
        mi = MultiIndex.from_tuples([("a", "A"), ("b", "A")])
        
        # 使用 DataFrame 创建一个对象 obj，数据为 [1, 2]，索引为 mi
        obj = DataFrame([1, 2], index=mi)
        
        # 使用 loc 方法设置索引为 indexer，列为所有列的数据为 0
        obj.loc[indexer, :] = 0
        
        # 创建一个预期的对象 expected，与 obj 相同索引，数据为 [0, 2]
        expected = DataFrame([0, 2], index=mi)
        
        # 使用 tm.assert_frame_equal 检查 obj 和 expected 是否相等
        tm.assert_frame_equal(obj, expected)

    @pytest.mark.parametrize(
        "indexer, exp_value", [(slice(None), 1.0), ((1, 2), np.nan)]
    )
    def test_multiindex_setitem_columns_enlarging(self, indexer, exp_value):
        # GH#39147
        # 创建一个 MultiIndex 对象 mi，包含两个元组
        mi = MultiIndex.from_tuples([(1, 2), (3, 4)])
        
        # 使用 DataFrame 创建一个对象 df，数据为 [[1, 2], [3, 4]]，索引为 mi，列名为 ["a", "b"]
        df = DataFrame([[1, 2], [3, 4]], index=mi, columns=["a", "b"])
        
        # 使用 loc 方法设置索引为 indexer，列为 ["c", "d"] 的数据为 1.0
        df.loc[indexer, ["c", "d"]] = 1.0
        
        # 创建一个预期的对象 expected，与 df 相同索引，列名为 ["a", "b", "c", "d"]，根据 exp_value 填充
        expected = DataFrame(
            [[1, 2, 1.0, 1.0], [3, 4, exp_value, exp_value]],
            index=mi,
            columns=["a", "b", "c", "d"],
        )
        
        # 使用 tm.assert_frame_equal 检查 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)
    def test_multiindex_setitem_axis_set(self):
        # GH#58116
        # 创建一个日期范围，从"2001-01-01"开始，每日频率，共两个周期
        dates = pd.date_range("2001-01-01", freq="D", periods=2)
        # 创建一个标识符列表
        ids = ["i1", "i2", "i3"]
        # 用日期和标识符列表创建一个多级索引对象
        index = MultiIndex.from_product([dates, ids], names=["date", "identifier"])
        # 创建一个数据框，索引为上述创建的多级索引，列为["A", "B"]，所有值初始化为0.0
        df = DataFrame(0.0, index=index, columns=["A", "B"])
        # 使用loc(axis=0)来选择行索引轴上的多级索引标签，将其值设置为None
        df.loc(axis=0)["2001-01-01", ["i1", "i3"]] = None

        # 创建预期的数据框，索引和列与df相同，但将一部分值设置为None
        expected = DataFrame(
            [
                [None, None],
                [0.0, 0.0],
                [None, None],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            index=index,
            columns=["A", "B"],
        )

        # 使用测试工具比较df和预期结果的内容是否相等
        tm.assert_frame_equal(df, expected)

    def test_sorted_multiindex_after_union(self):
        # GH#44752
        # 创建一个日期和索引的笛卡尔积，结果为多级索引对象midx
        midx = MultiIndex.from_product(
            [pd.date_range("20110101", periods=2), Index(["a", "b"])]
        )
        # 创建两个系列对象，索引为midx
        ser1 = Series(1, index=midx)
        ser2 = Series(1, index=midx[:2])
        # 将两个系列对象合并成一个数据框df，沿列轴合并
        df = pd.concat([ser1, ser2], axis=1)
        # 复制df作为预期结果
        expected = df.copy()
        # 使用loc来选择"2011-01-01"到"2011-01-02"之间的行
        result = df.loc["2011-01-01":"2011-01-02"]
        # 使用测试工具比较result和预期结果的内容是否相等
        tm.assert_frame_equal(result, expected)

        # 创建一个数据框，键为0和1，值分别为ser1和ser2
        df = DataFrame({0: ser1, 1: ser2})
        # 使用loc来选择"2011-01-01"到"2011-01-02"之间的行
        result = df.loc["2011-01-01":"2011-01-02"]
        # 使用测试工具比较result和预期结果的内容是否相等
        tm.assert_frame_equal(result, expected)

        # 将ser2重新索引为ser1的索引，并与ser1合并成数据框df
        df = pd.concat([ser1, ser2.reindex(ser1.index)], axis=1)
        # 使用loc来选择"2011-01-01"到"2011-01-02"之间的行
        result = df.loc["2011-01-01":"2011-01-02"]
        # 使用测试工具比较result和预期结果的内容是否相等
        tm.assert_frame_equal(result, expected)

    def test_loc_no_second_level_index(self):
        # GH#43599
        # 创建一个数据框，索引为三级多级索引，每级索引包含不同的字母
        df = DataFrame(
            index=MultiIndex.from_product([list("ab"), list("cd"), list("e")]),
            columns=["Val"],
        )
        # 使用loc和切片对象来选择第二级索引为"c"的所有行
        res = df.loc[np.s_[:, "c", :]]
        # 创建预期的数据框，索引为只包含"a"和"b"的第一级索引和"e"的第三级索引
        expected = DataFrame(
            index=MultiIndex.from_product([list("ab"), list("e")]), columns=["Val"]
        )
        # 使用测试工具比较res和预期结果的内容是否相等
        tm.assert_frame_equal(res, expected)

    def test_loc_multi_index_key_error(self):
        # GH 51892
        # 创建一个数据框，使用元组作为列名和对应的值
        df = DataFrame(
            {
                (1, 2): ["a", "b", "c"],
                (1, 3): ["d", "e", "f"],
                (2, 2): ["g", "h", "i"],
                (2, 4): ["j", "k", "l"],
            }
        )
        # 使用pytest来断言在访问(1, 4)位置时会引发KeyError异常
        with pytest.raises(KeyError, match=r"(1, 4)"):
            df.loc[0, (1, 4)]
@pytest.mark.parametrize(
    "indexer, pos",
    [
        ([], []),  # 空列表，预期正常情况
        (["A"], slice(3)),  # 包含单个索引"A"，使用切片对象slice(3)
        (["A", "D"], []),  # "D" 不在索引中，预期引发异常
        (["D", "E"], []),  # 索引中无值，预期引发异常
        (["D"], []),  # 同上，但使用单个项目列表: GH 27148
        (pd.IndexSlice[:, ["foo"]], slice(2, None, 3)),  # 多层索引切片，包含字符串列表
        (pd.IndexSlice[:, ["foo", "bah"]], slice(2, None, 3)),  # 多层索引切片，包含多个字符串列表
    ],
)
def test_loc_getitem_duplicates_multiindex_missing_indexers(indexer, pos):
    # GH 7866
    # 多层索引切片，处理缺少的索引
    idx = MultiIndex.from_product(
        [["A", "B", "C"], ["foo", "bar", "baz"]], names=["one", "two"]
    )
    ser = Series(np.arange(9, dtype="int64"), index=idx).sort_index()
    expected = ser.iloc[pos]

    if expected.size == 0 and indexer != []:
        # 如果期望结果为空且索引器不为空，则期望引发 KeyError 异常
        with pytest.raises(KeyError, match=str(indexer)):
            ser.loc[indexer]
    elif indexer == (slice(None), ["foo", "bah"]):
        # 如果索引器包含'bah'，但'bah'不在索引的层级中，则期望引发 KeyError 异常
        with pytest.raises(KeyError, match="'bah'"):
            ser.loc[indexer]
    else:
        # 其他情况下，获取结果并与预期进行比较
        result = ser.loc[indexer]
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns_indexer", [([], slice(None)), (["foo"], [])])
def test_loc_getitem_duplicates_multiindex_empty_indexer(columns_indexer):
    # GH 8737
    # 多层索引情况下的空索引器
    multi_index = MultiIndex.from_product((["foo", "bar", "baz"], ["alpha", "beta"]))
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 6)),
        index=range(5),
        columns=multi_index,
    )
    df = df.sort_index(level=0, axis=1)

    expected = DataFrame(index=range(5), columns=multi_index.reindex([])[0])
    result = df.loc[:, columns_indexer]
    tm.assert_frame_equal(result, expected)


def test_loc_getitem_duplicates_multiindex_non_scalar_type_object():
    # GH 7914
    # 多层索引中处理非标量对象类型
    df = DataFrame(
        [[np.mean, np.median], ["mean", "median"]],
        columns=MultiIndex.from_tuples([("functs", "mean"), ("functs", "median")]),
        index=["function", "name"],
    )
    result = df.loc["function", ("functs", "mean")]
    expected = np.mean
    assert result == expected


def test_loc_getitem_tuple_plus_slice():
    # GH 671
    # 使用元组加切片作为索引
    df = DataFrame(
        {
            "a": np.arange(10),
            "b": np.arange(10),
            "c": np.random.default_rng(2).standard_normal(10),
            "d": np.random.default_rng(2).standard_normal(10),
        }
    ).set_index(["a", "b"])
    expected = df.loc[0, 0]
    result = df.loc[(0, 0), :]
    tm.assert_series_equal(result, expected)


def test_loc_getitem_int(frame_random_data_integer_multi_index):
    # 使用整数作为索引
    df = frame_random_data_integer_multi_index
    result = df.loc[1]
    expected = df[-3:]
    expected.index = expected.index.droplevel(0)
    tm.assert_frame_equal(result, expected)


def test_loc_getitem_int_raises_exception(frame_random_data_integer_multi_index):
    # 期望引发异常的整数索引情况
    pass  # 空测试函数，仅用于标记预期的行为
    # 将变量 df 指定为 frame_random_data_integer_multi_index 的引用
    df = frame_random_data_integer_multi_index
    # 使用 pytest 的上下文管理器检查是否引发 KeyError 异常，并匹配错误消息以 "^3$" 开头的模式
    with pytest.raises(KeyError, match=r"^3$"):
        # 尝试使用 loc 方法访问 DataFrame df 的索引为 3 的行
        df.loc[3]
def test_loc_getitem_lowerdim_corner(multiindex_dataframe_random_data):
    # 使用提供的随机数据创建多索引数据框
    df = multiindex_dataframe_random_data

    # 测试设置 - 检查键不在数据框中引发 KeyError 异常
    with pytest.raises(KeyError, match=r"^\('bar', 'three'\)$"):
        df.loc[("bar", "three"), "B"]

    # 理论上应该在排序空间中插入???? （这里的注释是不完整的，可以忽略）
    df.loc[("bar", "three"), "B"] = 0
    expected = 0
    result = df.sort_index().loc[("bar", "three"), "B"]
    assert result == expected


def test_loc_setitem_single_column_slice():
    # 案例来自 https://github.com/pandas-dev/pandas/issues/27841
    df = DataFrame(
        "string",
        index=list("abcd"),
        columns=MultiIndex.from_product([["Main"], ("another", "one")]),
    )
    df["labels"] = "a"
    df.loc[:, "labels"] = df.index
    tm.assert_numpy_array_equal(np.asarray(df["labels"]), np.asarray(df.index))

    # 测试非对象块的情况
    df = DataFrame(
        np.nan,
        index=range(4),
        columns=MultiIndex.from_tuples([("A", "1"), ("A", "2"), ("B", "1")]),
    )
    expected = df.copy()
    df.loc[:, "B"] = np.arange(4)
    expected.iloc[:, 2] = np.arange(4)
    tm.assert_frame_equal(df, expected)


def test_loc_nan_multiindex(using_infer_string):
    # GH 5286
    tups = [
        ("Good Things", "C", np.nan),
        ("Good Things", "R", np.nan),
        ("Bad Things", "C", np.nan),
        ("Bad Things", "T", np.nan),
        ("Okay Things", "N", "B"),
        ("Okay Things", "N", "D"),
        ("Okay Things", "B", np.nan),
        ("Okay Things", "D", np.nan),
    ]
    # 创建具有多级索引的数据框，并初始化其内容为全 1
    df = DataFrame(
        np.ones((8, 4)),
        columns=Index(["d1", "d2", "d3", "d4"]),
        index=MultiIndex.from_tuples(tups, names=["u1", "u2", "u3"]),
    )
    # 使用 loc 获取 "Good Things" 的数据，然后进一步获取 "C" 的数据
    result = df.loc["Good Things"].loc["C"]
    # 创建预期的数据框，仅包含一行全 1
    expected = DataFrame(
        np.ones((1, 4)),
        index=Index(
            [np.nan],
            dtype="object" if not using_infer_string else "string[pyarrow_numpy]",
            name="u3",
        ),
        columns=Index(["d1", "d2", "d3", "d4"]),
    )
    tm.assert_frame_equal(result, expected)


def test_loc_period_string_indexing():
    # GH 9892
    # 创建一个时期范围和一个索引
    a = pd.period_range("2013Q1", "2013Q4", freq="Q")
    i = (1111, 2222, 3333)
    idx = MultiIndex.from_product((a, i), names=("Period", "CVR"))
    # 创建一个多索引数据框，索引为时期和索引值的笛卡尔积
    df = DataFrame(
        index=idx,
        columns=(
            "OMS",
            "OMK",
            "RES",
            "DRIFT_IND",
            "OEVRIG_IND",
            "FIN_IND",
            "VARE_UD",
            "LOEN_UD",
            "FIN_UD",
        ),
    )
    # 使用 loc 获取特定索引处的数据
    result = df.loc[("2013Q1", 1111), "OMS"]

    # 通过完全匹配字符串解析，确保不是切片
    alt = df.loc[(a[0], 1111), "OMS"]
    assert np.isnan(alt)

    # 因为字符串的分辨率匹配，这是一个精确查找，而不是切片
    assert np.isnan(result)

    alt = df.loc[("2013Q1", 1111), "OMS"]
    assert np.isnan(alt)


def test_loc_datetime_mask_slicing():
    # GH 16699
    # 创建一个日期时间索引
    dt_idx = pd.to_datetime(["2017-05-04", "2017-05-05"])
    # 使用 MultiIndex.from_product 创建多级索引，其中索引值为 dt_idx 的笛卡尔积，索引级别命名为 "Idx1", "Idx2"
    m_idx = MultiIndex.from_product([dt_idx, dt_idx], names=["Idx1", "Idx2"])
    
    # 使用 DataFrame 创建数据框，数据由二维列表提供，索引为 m_idx，列名为 "C1", "C2"
    df = DataFrame(
        data=[[1, 2], [3, 4], [5, 6], [7, 6]], index=m_idx, columns=["C1", "C2"]
    )
    
    # 从数据框 df 中选择满足条件的行，条件为第一个索引为 dt_idx[0]，第二个索引为 df 索引第二级大于 "2017-05-04" 的行，选择列 "C1"
    result = df.loc[(dt_idx[0], (df.index.get_level_values(1) > "2017-05-04")), "C1"]
    
    # 创建预期的 Series 对象，包含单个值 3，名称为 "C1"，索引为 MultiIndex，包含单个元组 (pd.Timestamp("2017-05-04"), pd.Timestamp("2017-05-05"))，索引级别命名为 "Idx1", "Idx2"
    expected = Series(
        [3],
        name="C1",
        index=MultiIndex.from_tuples(
            [(pd.Timestamp("2017-05-04"), pd.Timestamp("2017-05-05"))],
            names=["Idx1", "Idx2"],
        ),
    )
    
    # 使用 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
# 测试函数：测试针对包含时间戳的 Series 使用 loc 和 MultiIndex 的切片功能
def test_loc_datetime_series_tuple_slicing():
    # 创建一个时间戳对象
    date = pd.Timestamp("2000")
    # 创建一个 Series 对象，设置值为 1，使用 MultiIndex 作为索引，包含一个时间戳，命名为 "a" 和 "b"
    ser = Series(
        1,
        index=MultiIndex.from_tuples([("a", date)], names=["a", "b"]),
        name="c",
    )
    # 使用 loc 方法，通过切片方式选择所有列和特定的时间戳，并赋值给 result 变量
    result = ser.loc[:, [date]]
    # 断言 result 和 ser 的内容相等
    tm.assert_series_equal(result, ser)


# 测试函数：测试针对 MultiIndex 使用 loc 方法进行索引
def test_loc_with_mi_indexer():
    # 创建一个 DataFrame 对象，数据包含作者和价格，使用 MultiIndex 作为索引，命名为 "index" 和 "date"
    df = DataFrame(
        data=[["a", 1], ["a", 0], ["b", 1], ["c", 2]],
        index=MultiIndex.from_tuples(
            [(0, 1), (1, 0), (1, 1), (1, 1)], names=["index", "date"]
        ),
        columns=["author", "price"],
    )
    # 创建一个 MultiIndex 对象，包含两个元组作为索引，命名为 "index" 和 "date"
    idx = MultiIndex.from_tuples([(0, 1), (1, 1)], names=["index", "date"])
    # 使用 loc 方法，通过 idx 索引选择所有列，并赋值给 result 变量
    result = df.loc[idx, :]
    # 创建一个期望的 DataFrame 对象，包含特定作者和价格，使用相同的 MultiIndex 作为索引，命名为 "index" 和 "date"
    expected = DataFrame(
        [["a", 1], ["b", 1], ["c", 2]],
        index=MultiIndex.from_tuples([(0, 1), (1, 1), (1, 1)], names=["index", "date"]),
        columns=["author", "price"],
    )
    # 断言 result 和 expected 的内容相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试在 MultiIndex 的第一个级别命名为 0 时使用 loc 方法
def test_loc_mi_with_level1_named_0():
    # 创建一个包含时区的日期范围对象
    dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")

    # 创建一个 Series 对象，使用日期范围对象作为索引
    ser = Series(range(3), index=dti)
    # 将 Series 转换为 DataFrame 对象，并添加一个列，包含相同的日期范围对象
    df = ser.to_frame()
    df[1] = dti

    # 使用 set_index 方法，在第一个级别上追加一个索引
    df2 = df.set_index(0, append=True)
    # 断言索引的名称是 (None, 0)
    assert df2.index.names == (None, 0)
    # 使用 get_loc 方法，对第一个日期范围对象进行检索（仅为验证）
    df2.index.get_loc(dti[0])  # smoke test

    # 使用 loc 方法，选择特定的日期范围对象，并赋值给 result 变量
    result = df2.loc[dti[0]]
    # 创建一个期望的 DataFrame 对象，包含相同的内容，但移除了第一个级别的索引
    expected = df2.iloc[[0]].droplevel(None)
    # 断言 result 和 expected 的内容相等
    tm.assert_frame_equal(result, expected)

    # 创建一个 Series 对象，包含在 df2 中的第二列
    ser2 = df2[1]
    # 断言 Series 的索引名称是 (None, 0)
    assert ser2.index.names == (None, 0)

    # 使用 loc 方法，选择特定的日期范围对象，并赋值给 result 变量
    result = ser2.loc[dti[0]]
    # 创建一个期望的 Series 对象，包含相同的内容，但移除了第一个级别的索引
    expected = ser2.iloc[[0]].droplevel(None)
    # 断言 result 和 expected 的内容相等
    tm.assert_series_equal(result, expected)


# 测试函数：测试使用 loc 方法对包含字符串切片的 DataFrame 进行索引
def test_getitem_str_slice():
    # 创建一个 DataFrame 对象，包含时间、股票代码、买入价格和卖出价格等信息
    df = DataFrame(
        [
            ["20160525 13:30:00.023", "MSFT", "51.95", "51.95"],
            ["20160525 13:30:00.048", "GOOG", "720.50", "720.93"],
            ["20160525 13:30:00.076", "AAPL", "98.55", "98.56"],
            ["20160525 13:30:00.131", "AAPL", "98.61", "98.62"],
            ["20160525 13:30:00.135", "MSFT", "51.92", "51.95"],
            ["20160525 13:30:00.135", "AAPL", "98.61", "98.62"],
        ],
        columns="time,ticker,bid,ask".split(","),
    )
    # 使用 set_index 方法，将列 "ticker" 和 "time" 设置为索引，并按照索引进行排序
    df2 = df.set_index(["ticker", "time"]).sort_index()

    # 使用 loc 方法，通过多级索引选择特定股票代码和时间范围，并移除第一个级别的索引
    res = df2.loc[("AAPL", slice("2016-05-25 13:30:00")), :].droplevel(0)
    # 创建一个期望的 DataFrame 对象，包含相同的内容
    expected = df2.loc["AAPL"].loc[slice("2016-05-25 13:30:00"), :]
    # 断言 res 和 expected 的内容相等
    tm.assert_frame_equal(res, expected)


# 测试函数：测试在包含三个级别的 PeriodIndex 中使用 loc 方法
def test_3levels_leading_period_index():
    # 创建一个 PeriodIndex 对象，包含日期时间和频率信息
    pi = pd.PeriodIndex(
        ["20181101 1100", "20181101 1200", "20181102 1300", "20181102 1400"],
        name="datetime",
        freq="D",
    )
    # 创建一个 Series 对象，包含浮点数值，使用 MultiIndex 作为索引
    lev2 = ["A", "A", "Z", "W"]
    lev3 = ["B", "C", "Q", "F"]
    mi = MultiIndex.from_arrays([pi, lev2, lev3])

    # 使用 loc 方法，通过多级索引选择特定的日期时间，并赋值给 result 变量
    ser = Series(range(4), index=mi, dtype=np.float64)
    result = ser.loc[(pi[0], "A", "B")]
    # 断言 result 的值等于 0.0
    assert result == 0.0
    # 测试缺失的键引发 KeyError 而不是 TypeError
    def test_missing_keys_raises_keyerror(self):
        # 创建一个包含12个元素的二维数组 DataFrame，指定列名为 ["A", "B", "C"]
        df = DataFrame(np.arange(12).reshape(4, 3), columns=["A", "B", "C"])
        # 在 DataFrame 上设置复合索引 ["A", "B"]
        df2 = df.set_index(["A", "B"])

        # 使用 pytest 断言捕获 KeyError 异常，匹配错误信息包含 "1"
        with pytest.raises(KeyError, match="1"):
            # 尝试访问复合索引 (1, 6) 的元素
            df2.loc[(1, 6)]

    # 测试缺失的键引发 KeyError 而不是 "IndexingError: Too many indexers"
    def test_missing_key_raises_keyerror2(self):
        # 创建一个包含两级索引的 Series
        ser = Series(-1, index=MultiIndex.from_product([[0, 1]] * 2))

        # 使用 pytest 断言捕获 KeyError 异常，匹配错误信息为 "\(0, 3\)"
        with pytest.raises(KeyError, match=r"\(0, 3\)"):
            # 尝试访问索引 (0, 3) 的元素
            ser.loc[0, 3]

    # 测试组合键的缺失引发特定 KeyError
    def test_missing_key_combination(self):
        # 创建一个包含多级索引的 DataFrame
        mi = MultiIndex.from_arrays(
            [
                np.array(["a", "a", "b", "b"]),
                np.array(["1", "2", "2", "3"]),
                np.array(["c", "d", "c", "d"]),
            ],
            names=["one", "two", "three"],
        )
        # 创建一个随机值填充的 DataFrame，使用 mi 作为索引
        df = DataFrame(np.random.default_rng(2).random((4, 3)), index=mi)
        
        # 构造用于匹配的错误消息
        msg = r"\('b', '1', slice\(None, None, None\)\)"
        
        # 使用 pytest 断言捕获 KeyError 异常，匹配错误信息为 msg
        with pytest.raises(KeyError, match=msg):
            # 尝试使用复合键 ("b", "1", slice(None)) 访问 DataFrame
            df.loc[("b", "1", slice(None)), :]
        
        # 使用 pytest 断言捕获 KeyError 异常，匹配错误信息为 "\(b', '1'\)"
        with pytest.raises(KeyError, match=r"\('b', '1'\)"):
            # 尝试获取索引 ("b", "1") 的位置信息
            df.index.get_locs(("b", "1"))
# 测试从多重索引的 DataFrame 中获取单列 Series 的元素，通过多个索引值获取对应的结果
def test_getitem_loc_commutability(multiindex_year_month_day_dataframe_random_data):
    df = multiindex_year_month_day_dataframe_random_data
    # 获取列"A"的 Series
    ser = df["A"]
    # 使用多个索引值(2000, 5)获取 Series 中的元素
    result = ser[2000, 5]
    # 使用 loc 方法根据多重索引(2000, 5)获取 DataFrame 中的元素"A"的值
    expected = df.loc[2000, 5]["A"]
    # 使用测试框架检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


# 测试在包含 NaN 的 DataFrame 中使用 loc 方法获取数据
def test_loc_with_nan():
    # GH: 27104
    # 创建一个包含 NaN 值的 DataFrame，并设置索引为 ["ind1", "ind2"]
    df = DataFrame(
        {"col": [1, 2, 5], "ind1": ["a", "d", np.nan], "ind2": [1, 4, 5]}
    ).set_index(["ind1", "ind2"])
    # 使用 loc 方法获取索引为 ["a"] 的部分 DataFrame
    result = df.loc[["a"]]
    # 创建预期的 DataFrame，索引使用 MultiIndex，并使用测试框架检查结果
    expected = DataFrame(
        {"col": [1]}, index=MultiIndex.from_tuples([("a", 1)], names=["ind1", "ind2"])
    )
    tm.assert_frame_equal(result, expected)

    # 使用 loc 方法获取索引为 "a" 的部分 DataFrame，并使用测试框架检查结果
    result = df.loc["a"]
    expected = DataFrame({"col": [1]}, index=Index([1], name="ind2"))
    tm.assert_frame_equal(result, expected)


# 测试在 MultiIndex DataFrame 中使用 loc 方法获取不存在的元组
def test_getitem_non_found_tuple():
    # GH: 25236
    # 创建一个 MultiIndex DataFrame，并设置索引为 ["a", "b", "c"]
    df = DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"]).set_index(
        ["a", "b", "c"]
    )
    # 使用 pytest 框架检查是否会引发 KeyError，匹配错误信息为 "(2.0, 2.0, 3.0)"
    with pytest.raises(KeyError, match=r"\(2\.0, 2\.0, 3\.0\)"):
        df.loc[(2.0, 2.0, 3.0)]


# 测试在 DateTime 索引中使用 loc 方法获取元素
def test_get_loc_datetime_index():
    # GH#24263
    # 创建一个日期范围索引
    index = pd.date_range("2001-01-01", periods=100)
    # 创建一个 MultiIndex，包含日期范围索引
    mi = MultiIndex.from_arrays([index])
    # 检查 get_loc 方法对 Index 和 MultiIndex 的匹配情况
    assert mi.get_loc("2001-01") == slice(0, 31, None)
    assert index.get_loc("2001-01") == slice(0, 31, None)

    # 获取经过步长为 2 的 MultiIndex 中 "2001-01" 的 loc
    loc = mi[::2].get_loc("2001-01")
    expected = index[::2].get_loc("2001-01")
    assert loc == expected

    # 获取重复两次的 MultiIndex 中 "2001-01" 的 loc
    loc = mi.repeat(2).get_loc("2001-01")
    expected = index.repeat(2).get_loc("2001-01")
    assert loc == expected

    # 获取追加两次的 MultiIndex 中 "2001-01" 的 loc
    loc = mi.append(mi).get_loc("2001-01")
    expected = index.append(index).get_loc("2001-01")
    # TODO: standardize return type for MultiIndex.get_loc
    # 使用测试框架检查结果是否相等
    tm.assert_numpy_array_equal(loc.nonzero()[0], expected)


# 测试在 MultiIndex DataFrame 中使用 loc 方法设置 indexer 的不同顺序的数据
def test_loc_setitem_indexer_differently_ordered():
    # GH#34603
    # 创建一个 MultiIndex，包含 ["a", "b"] 和 [0, 1]
    mi = MultiIndex.from_product([["a", "b"], [0, 1]])
    # 创建一个 DataFrame，并使用 mi 作为索引
    df = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=mi)

    # 创建一个索引为 ("a", [1, 0]) 的 indexer，设置对应位置的值为 np.array([[9, 10], [11, 12]])
    indexer = ("a", [1, 0])
    df.loc[indexer, :] = np.array([[9, 10], [11, 12]])
    # 创建预期的 DataFrame，并使用测试框架检查结果
    expected = DataFrame([[11, 12], [9, 10], [5, 6], [7, 8]], index=mi)
    tm.assert_frame_equal(df, expected)


# 测试在 MultiIndex DataFrame 中使用 loc 方法获取 indexer 顺序不同的切片数据
def test_loc_getitem_index_differently_ordered_slice_none():
    # GH#31330
    # 创建一个 MultiIndex DataFrame，并设置复合索引
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
        columns=["a", "b"],
    )
    # 使用 loc 方法获取索引为 (slice(None), [2, 1]) 的部分 DataFrame
    result = df.loc[(slice(None), [2, 1]), :]
    # 创建预期的 DataFrame，并使用测试框架检查结果
    expected = DataFrame(
        [[3, 4], [7, 8], [1, 2], [5, 6]],
        index=[["a", "b", "a", "b"], [2, 2, 1, 1]],
        columns=["a", "b"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("indexer", [[1, 2, 7, 6, 2, 3, 8, 7], [1, 2, 7, 6, 3, 8]])
# 测试在 MultiIndex DataFrame 中使用 loc 方法获取 indexer 顺序不同且包含重复数据的切片数据
def test_loc_getitem_index_differently_ordered_slice_none_duplicates(indexer):
    # GH#40978
    # 在这里只是定义了参数化的测试用例，详细测试逻辑不在这里展开
    # 创建一个 DataFrame 对象，包含一个由值 1 组成的列表，行索引为 MultiIndex 对象
    # MultiIndex 由元组构成，每个元组包含两个整数
    df = DataFrame(
        [1] * 8,  # 列表中的每个元素都是整数 1，共 8 个元素，对应 DataFrame 的 8 行
        index=MultiIndex.from_tuples(
            [(1, 1), (1, 2), (1, 7), (1, 6), (2, 2), (2, 3), (2, 8), (2, 7)]
        ),  # 创建一个 MultiIndex 对象作为行索引，包含了 8 个元组
        columns=["a"],  # 指定 DataFrame 只有一列，列名为 "a"
    )
    # 使用切片对象 indexer 选择 DataFrame 的子集，存储到 result 中
    result = df.loc[(slice(None), indexer), :]
    # 创建一个预期的 DataFrame 对象，也包含一个由值 1 组成的列表
    # 行索引是一个包含两个子列表的列表，每个子列表表示 MultiIndex 的各级索引
    expected = DataFrame(
        [1] * 8,
        index=[[1, 1, 2, 1, 2, 1, 2, 2], [1, 2, 2, 7, 7, 6, 3, 8]],
        columns=["a"],
    )
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 使用 df.index.isin 方法选择包含在 indexer 中的子集，存储到 result 中
    result = df.loc[df.index.isin(indexer, level=1), :]
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 df 是否相等
    tm.assert_frame_equal(result, df)
def test_loc_getitem_drops_levels_for_one_row_dataframe():
    # GH#10521 "x" and "z" are both scalar indexing, so those levels are dropped
    # 创建一个多级索引对象 `mi`，包含三个级别的标签 ["a", "b", "c"]
    mi = MultiIndex.from_arrays([["x"], ["y"], ["z"]], names=["a", "b", "c"])
    # 创建一个DataFrame `df`，包含一行数据 {"d": [0]}，使用 `mi` 作为索引
    df = DataFrame({"d": [0]}, index=mi)
    # 期望的DataFrame `expected` 是 `df` 去除第一级和第三级索引后的结果
    expected = df.droplevel([0, 2])
    # 使用 `.loc` 方法选择 DataFrame `df` 中 "x" 行，":" 表示选择所有第二级索引，"z" 表示选择第三级索引为 "z"
    result = df.loc["x", :, "z"]
    # 断言 `result` 与 `expected` 相等
    tm.assert_frame_equal(result, expected)

    # 创建一个Series `ser`，包含一个数据 [0]，使用 `mi` 作为索引
    ser = Series([0], index=mi)
    # 使用 `.loc` 方法选择 Series `ser` 中 "x" 行，":" 表示选择所有第二级索引，"z" 表示选择第三级索引为 "z"
    result = ser.loc["x", :, "z"]
    # 期望的Series `expected` 是包含一个数据 [0]，索引为 ["y"]，名为 "b"
    expected = Series([0], index=Index(["y"], name="b"))
    # 断言 `result` 与 `expected` 相等
    tm.assert_series_equal(result, expected)


def test_mi_columns_loc_list_label_order():
    # GH 10710
    # 创建一个多级列索引 `cols`，包含两个级别，第一级为 ["A", "B", "C"]，第二级为 [1, 2]
    cols = MultiIndex.from_product([["A", "B", "C"], [1, 2]])
    # 创建一个DataFrame `df`，形状为 (5, 6)，所有元素为0，列索引为 `cols`
    df = DataFrame(np.zeros((5, 6)), columns=cols)
    # 使用 `.loc` 方法选择所有行 ":"，列索引选择顺序为 ["B", "A"] 的列
    result = df.loc[:, ["B", "A"]]
    # 期望的DataFrame `expected` 形状为 (5, 4)，所有元素为0，列索引为 [("B", 1), ("B", 2), ("A", 1), ("A", 2)]
    expected = DataFrame(
        np.zeros((5, 4)),
        columns=MultiIndex.from_tuples([("B", 1), ("B", 2), ("A", 1), ("A", 2)]),
    )
    # 断言 `result` 与 `expected` 相等
    tm.assert_frame_equal(result, expected)


def test_mi_partial_indexing_list_raises():
    # GH 13501
    # 创建一个DataFrame `frame`，形状为 (4, 3)，包含索引和列名
    frame = DataFrame(
        np.arange(12).reshape((4, 3)),
        index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
        columns=[["Ohio", "Ohio", "Colorado"], ["Green", "Red", "Green"]],
    )
    # 设置索引名和列名
    frame.index.names = ["key1", "key2"]
    frame.columns.names = ["state", "color"]
    # 使用 `.loc` 方法选择索引为 ["b", 2]，列名为 "Colorado" 的元素，预期会引发 KeyError
    with pytest.raises(KeyError, match="\\[2\\] not in index"):
        frame.loc[["b", 2], "Colorado"]


def test_mi_indexing_list_nonexistent_raises():
    # GH 15452
    # 创建一个Series `s`，包含4个数据，使用乘积形成的多级索引
    s = Series(range(4), index=MultiIndex.from_product([[1, 2], ["a", "b"]]))
    # 使用 `.loc` 方法选择索引为 ["not", "found"] 的元素，预期会引发 KeyError
    with pytest.raises(KeyError, match="\\['not' 'found'\\] not in index"):
        s.loc[["not", "found"]]


def test_mi_add_cell_missing_row_non_unique():
    # GH 16018
    # 创建一个DataFrame `result`，形状为 (4, 4)，包含指定的数据和索引
    result = DataFrame(
        [[1, 2, 5, 6], [3, 4, 7, 8]],
        index=["a", "a"],
        columns=MultiIndex.from_product([[1, 2], ["A", "B"]]),
    )
    # 向 `result` 添加一行索引为 "c"，数据为 [-1]
    result.loc["c"] = -1
    # 修改 `result` 中索引为 "c"，列索引为 (1, "A") 的元素为 3
    result.loc["c", (1, "A")] = 3
    # 添加一行索引为 "d"，列索引为 (1, "A") 的元素为 3
    result.loc["d", (1, "A")] = 3
    # 期望的DataFrame `expected` 包含指定的数据和索引
    expected = DataFrame(
        [
            [1.0, 2.0, 5.0, 6.0],
            [3.0, 4.0, 7.0, 8.0],
            [3.0, -1.0, -1, -1],
            [3.0, np.nan, np.nan, np.nan],
        ],
        index=["a", "a", "c", "d"],
        columns=MultiIndex.from_product([[1, 2], ["A", "B"]]),
    )
    # 断言 `result` 与 `expected` 相等
    tm.assert_frame_equal(result, expected)


def test_loc_get_scalar_casting_to_float():
    # GH#41369
    # 创建一个DataFrame `df`，包含两列 "a" 和 "b"，使用多级索引
    df = DataFrame(
        {"a": 1.0, "b": 2}, index=MultiIndex.from_arrays([[3], [4]], names=["c", "d"])
    )
    # 使用 `.loc` 方法选择索引为 (3, 4)，列名为 "b" 的元素，期望结果为2，且类型为 np.int64
    result = df.loc[(3, 4), "b"]
    assert result == 2
    assert isinstance(result, np.int64)
    # 使用 `.loc` 方法选择索引为 [(3, 4)]，列名为 "b" 的元素的第一个值，期望结果为2，且类型为 np.int64
    result = df.loc[[(3, 4)], "b"].iloc[0]
    assert result == 2
    assert isinstance(result, np.int64)


def test_loc_empty_single_selector_with_names():
    # GH 19517
    # 创建一个多级索引 `idx`，包含两个级别，第一级为 ["a", "b"]，第二级为 ["A", "B"]，名称分别为 1 和 0
    idx = MultiIndex.from_product([["a", "b"], ["A", "B"]], names=[1, 0])
    # 创建一个Series `s2`，索引为 `idx`，数据类型为 np.float64
    s2 = Series(index=idx, dtype=np.float64)
    # 使用 `.loc` 方法选择索引为 "a" 的元素，期望结果为包含两个 NaN 值的Series，索引为 ["A", "B"]，名称为 0
    result = s2.loc["a"]
    expected = Series([np.nan, np.nan], index=Index(["A", "B"], name=0))
    # 断言 `result` 与 `expected` 相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试在特定情况下是否会触发 KeyError 异常
def test_loc_keyerror_rightmost_key_missing():
    # GH 20951
    # 创建一个 DataFrame 对象，包含三列数据，其中列 "A" 和 "B" 作为索引
    df = DataFrame(
        {
            "A": [100, 100, 200, 200, 300, 300],
            "B": [10, 10, 20, 21, 31, 33],
            "C": range(6),
        }
    )
    # 将 "A" 和 "B" 列设为 DataFrame 的索引
    df = df.set_index(["A", "B"])
    # 使用 pytest 模块的 raises 函数检查是否会抛出 KeyError 异常，并匹配异常消息 "^1$"
    with pytest.raises(KeyError, match="^1$"):
        # 在 DataFrame 中使用 loc 方法来访问索引为 (100, 1) 的行，预期会抛出 KeyError 异常
        df.loc[(100, 1)]


# 定义一个测试函数，用于测试 MultiIndex 下的 Series 对象的 loc 方法
def test_multindex_series_loc_with_tuple_label():
    # GH#43908
    # 创建一个 MultiIndex 对象，包含两个元组作为索引标签
    mi = MultiIndex.from_tuples([(1, 2), (3, (4, 5))])
    # 创建一个 Series 对象，指定索引为 mi，数值为 [1, 2]
    ser = Series([1, 2], index=mi)
    # 使用 Series 对象的 loc 方法，通过元组标签 (3, (4, 5)) 来访问对应的数值
    result = ser.loc[(3, (4, 5))]
    # 断言访问结果应为 2
    assert result == 2
```