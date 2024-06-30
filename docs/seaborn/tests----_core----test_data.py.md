# `D:\src\scipysrc\seaborn\tests\_core\test_data.py`

```
import functools  # 导入 functools 模块，用于创建部分函数
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理和分析

import pytest  # 导入 pytest 测试框架
from numpy.testing import assert_array_equal  # 导入 NumPy 的数组相等断言函数
from pandas.testing import assert_series_equal  # 导入 Pandas 的序列相等断言函数

from seaborn._core.data import PlotData  # 从 seaborn 库中导入 PlotData 类

# 使用 functools 创建一个偏函数 assert_vector_equal，用于比较 Pandas Series 是否相等，忽略列名
assert_vector_equal = functools.partial(assert_series_equal, check_names=False)

# 定义一个测试类 TestPlotData
class TestPlotData:

    # pytest 的 fixture，用于返回长格式数据的变量字典
    @pytest.fixture
    def long_variables(self):
        variables = dict(x="x", y="y", color="a", size="z", style="s_cat")
        return variables

    # 测试函数：测试命名向量的情况
    def test_named_vectors(self, long_df, long_variables):
        # 创建 PlotData 对象 p，传入长格式数据和长格式变量
        p = PlotData(long_df, long_variables)
        # 断言 p 的源数据是 long_df
        assert p.source_data is long_df
        # 断言 p 的源变量是 long_variables
        assert p.source_vars is long_variables
        # 遍历 long_variables 的键值对，断言 p 的命名与向量的映射关系
        for key, val in long_variables.items():
            assert p.names[key] == val
            # 使用 assert_vector_equal 检查 p 的 frame[key] 与 long_df[val] 是否相等

    # 测试函数：测试命名向量和给定向量的情况
    def test_named_and_given_vectors(self, long_df, long_variables):
        # 修改 long_variables 中的部分键值对
        long_variables["y"] = long_df["b"]
        long_variables["size"] = long_df["z"].to_numpy()

        # 创建 PlotData 对象 p，传入修改后的长格式数据和变量
        p = PlotData(long_df, long_variables)

        # 使用 assert_vector_equal 检查 p 的 frame[key] 与 long_df[long_variables[key]] 是否相等
        assert_vector_equal(p.frame["color"], long_df[long_variables["color"]])
        assert_vector_equal(p.frame["y"], long_df["b"])
        assert_vector_equal(p.frame["size"], long_df["z"])

        # 断言 p 的命名与向量的映射关系
        assert p.names["color"] == long_variables["color"]
        assert p.names["y"] == "b"
        assert p.names["size"] is None

        # 断言 p 的 ids 与向量的映射关系
        assert p.ids["color"] == long_variables["color"]
        assert p.ids["y"] == "b"
        assert p.ids["size"] == id(long_variables["size"])

    # 测试函数：测试将索引作为变量的情况
    def test_index_as_variable(self, long_df, long_variables):
        # 创建一个新的索引对象 index
        index = pd.Index(np.arange(len(long_df)) * 2 + 10, name="i", dtype=int)
        long_variables["x"] = "i"
        # 创建 PlotData 对象 p，传入设定索引后的长格式数据和变量
        p = PlotData(long_df.set_index(index), long_variables)

        # 断言 p 的命名和 ids 与 "x" 相等，且 frame["x"] 等于索引 Series
        assert p.names["x"] == p.ids["x"] == "i"
        assert_vector_equal(p.frame["x"], pd.Series(index, index))

    # 测试函数：测试将多级索引作为变量的情况
    def test_multiindex_as_variables(self, long_df, long_variables):
        # 创建两个新的索引对象 index_i 和 index_j
        index_i = pd.Index(np.arange(len(long_df)) * 2 + 10, name="i", dtype=int)
        index_j = pd.Index(np.arange(len(long_df)) * 3 + 5, name="j", dtype=int)
        index = pd.MultiIndex.from_arrays([index_i, index_j])
        long_variables.update({"x": "i", "y": "j"})

        # 创建 PlotData 对象 p，传入设定多级索引后的长格式数据和变量
        p = PlotData(long_df.set_index(index), long_variables)

        # 使用 assert_vector_equal 检查 p 的 frame[key] 与对应的索引 Series 是否相等
        assert_vector_equal(p.frame["x"], pd.Series(index_i, index))
        assert_vector_equal(p.frame["y"], pd.Series(index_j, index))

    # 测试函数：测试将整数作为变量键的情况
    def test_int_as_variable_key(self, rng):
        # 创建一个包含随机数的 DataFrame 对象 df
        df = pd.DataFrame(rng.uniform(size=(10, 3)))

        var = "x"
        key = 2

        # 创建 PlotData 对象 p，传入包含整数键的 DataFrame 和变量字典
        p = PlotData(df, {var: key})
        # 使用 assert_vector_equal 检查 p 的 frame[var] 与 df[key] 是否相等
        assert_vector_equal(p.frame[var], df[key])
        # 断言 p 的命名和 ids 与 key 的字符串表示相等
        assert p.names[var] == p.ids[var] == str(key)

    # 测试函数：测试将整数作为变量值的情况
    def test_int_as_variable_value(self, long_df):
        # 创建 PlotData 对象 p，传入包含整数变量值的长格式数据和变量字典
        p = PlotData(long_df, {"x": 0, "y": "y"})
        # 使用 assert 检查 p 的 frame["x"] 全部为 0
        assert (p.frame["x"] == 0).all()
        # 断言 p 的命名为 None，ids 与整数变量值 0 的 id() 相等
        assert p.names["x"] is None
        assert p.ids["x"] == id(0)
    # 测试函数：使用元组作为变量键
    def test_tuple_as_variable_key(self, rng):
        # 创建多级索引的 DataFrame
        cols = pd.MultiIndex.from_product([("a", "b", "c"), ("x", "y")])
        df = pd.DataFrame(rng.uniform(size=(10, 6)), columns=cols)

        # 定义变量和键
        var = "color"
        key = ("b", "y")
        
        # 创建 PlotData 对象，并传入 DataFrame 和变量及其对应的键
        p = PlotData(df, {var: key})
        
        # 断言：验证获取列数据是否正确
        assert_vector_equal(p.frame[var], df[key])
        
        # 断言：验证变量名、ID 是否与键对应
        assert p.names[var] == p.ids[var] == str(key)

    # 测试函数：使用字典作为数据输入
    def test_dict_as_data(self, long_dict, long_variables):
        # 创建 PlotData 对象，并传入长字典和长变量字典
        p = PlotData(long_dict, long_variables)
        
        # 断言：验证源数据是否正确传递
        assert p.source_data is long_dict
        
        # 遍历长变量字典，验证每个变量名对应的 Series 数据是否正确
        for key, val in long_variables.items():
            assert_vector_equal(p.frame[key], pd.Series(long_dict[val]))

    # 测试函数：测试不同类型的向量数据
    @pytest.mark.parametrize(
        "vector_type",
        ["series", "numpy", "list"],
    )
    def test_vectors_various_types(self, long_df, long_variables, vector_type):
        # 根据长 DataFrame 和长变量字典创建变量字典
        variables = {key: long_df[val] for key, val in long_variables.items()}
        
        # 根据向量类型进行适当的数据转换
        if vector_type == "numpy":
            variables = {key: val.to_numpy() for key, val in variables.items()}
        elif vector_type == "list":
            variables = {key: val.to_list() for key, val in variables.items()}

        # 创建 PlotData 对象，并传入 None 和变量字典
        p = PlotData(None, variables)

        # 断言：验证变量名列表是否匹配长变量字典的键列表
        assert list(p.names) == list(long_variables)
        
        # 根据向量类型进行不同的断言
        if vector_type == "series":
            # 断言：验证源变量是否正确传递
            assert p.source_vars is variables
            # 断言：验证变量名、ID 是否与变量名、值的名称相同
            assert p.names == p.ids == {key: val.name for key, val in variables.items()}
        else:
            # 断言：验证变量名是否与变量字典的键对应，值是否为 None
            assert p.names == {key: None for key in variables}
            # 断言：验证变量 ID 是否与其值的 ID 对应
            assert p.ids == {key: id(val) for key, val in variables.items()}

        # 遍历长变量字典，根据向量类型进行不同的断言
        for key, val in long_variables.items():
            if vector_type == "series":
                # 断言：验证 DataFrame 列数据是否与长 DataFrame 中的 Series 数据一致
                assert_vector_equal(p.frame[key], long_df[val])
            else:
                # 断言：验证 DataFrame 列数据是否与长 DataFrame 中的数组数据一致
                assert_array_equal(p.frame[key], long_df[val])

    # 测试函数：测试变量值为 None 的情况
    def test_none_as_variable_value(self, long_df):
        # 创建 PlotData 对象，并传入长 DataFrame 和包含 "x" 和 None 的变量字典
        p = PlotData(long_df, {"x": "z", "y": None})
        
        # 断言：验证 DataFrame 列名是否为 ["x"]
        assert list(p.frame.columns) == ["x"]
        
        # 断言：验证变量名、ID 是否与变量名、值相对应
        assert p.names == p.ids == {"x": "z"}

    # 测试函数：测试 DataFrame 和向量长度不匹配的情况
    def test_frame_and_vector_mismatched_lengths(self, long_df):
        # 创建长度为长 DataFrame 两倍的向量
        vector = np.arange(len(long_df) * 2)
        
        # 断言：验证创建 PlotData 对象时会抛出 ValueError 异常
        with pytest.raises(ValueError):
            PlotData(long_df, {"x": "x", "y": vector})

    # 测试函数：测试空数据输入的情况
    @pytest.mark.parametrize(
        "arg", [{}, pd.DataFrame()],
    )
    def test_empty_data_input(self, arg):
        # 创建 PlotData 对象，并传入空数据和空变量字典
        p = PlotData(arg, {})
        
        # 断言：验证 DataFrame 是否为空
        assert p.frame.empty
        
        # 断言：验证变量名是否为空
        assert not p.names

        # 如果输入不是 DataFrame，再次创建 PlotData 对象，并传入 dict(x=arg, y=arg)
        if not isinstance(arg, pd.DataFrame):
            p = PlotData(None, dict(x=arg, y=arg))
            
            # 断言：验证 DataFrame 是否为空
            assert p.frame.empty
            
            # 断言：验证变量名是否为空
            assert not p.names
    # 定义一个测试方法，用于测试索引对齐时将 Series 转换为 DataFrame
    def test_index_alignment_series_to_dataframe(self):

        # 创建列表 x 和对应的索引 x_index
        x = [1, 2, 3]
        x_index = pd.Index(x, dtype=int)

        # 创建列表 y_values 和对应的索引 y_index，并用其创建 Series 对象 y
        y_values = [3, 4, 5]
        y_index = pd.Index(y_values, dtype=int)
        y = pd.Series(y_values, y_index, name="y")

        # 创建 DataFrame 对象 data，包含列 'x'，并以 x_index 为索引
        data = pd.DataFrame(dict(x=x), index=x_index)

        # 创建 PlotData 对象 p，用 data 中的 'x' 列和给定的 Series 对象 y
        p = PlotData(data, {"x": "x", "y": y})

        # 期望的 x 列和 y 列的结果
        x_col_expected = pd.Series([1, 2, 3, np.nan, np.nan], np.arange(1, 6))
        y_col_expected = pd.Series([np.nan, np.nan, 3, 4, 5], np.arange(1, 6))

        # 断言 p 对象的 frame 属性中 'x' 列和 'y' 列与期望值相等
        assert_vector_equal(p.frame["x"], x_col_expected)
        assert_vector_equal(p.frame["y"], y_col_expected)

    # 定义一个测试方法，用于测试两个 Series 之间的索引对齐
    def test_index_alignment_between_series(self):

        # 创建列表 x_index 和 x_values，并用其创建 Series 对象 x
        x_index = [1, 2, 3]
        x_values = [10, 20, 30]
        x = pd.Series(x_values, x_index, name="x")

        # 创建列表 y_index 和 y_values，并用其创建 Series 对象 y
        y_index = [3, 4, 5]
        y_values = [300, 400, 500]
        y = pd.Series(y_values, y_index, name="y")

        # 创建 PlotData 对象 p，用 x 和 y 对象
        p = PlotData(None, {"x": x, "y": y})

        # 期望的索引和 x 列、y 列的结果
        idx_expected = [1, 2, 3, 4, 5]
        x_col_expected = pd.Series([10, 20, 30, np.nan, np.nan], idx_expected)
        y_col_expected = pd.Series([np.nan, np.nan, 300, 400, 500], idx_expected)

        # 断言 p 对象的 frame 属性中 'x' 列和 'y' 列与期望值相等
        assert_vector_equal(p.frame["x"], x_col_expected)
        assert_vector_equal(p.frame["y"], y_col_expected)

    # 定义一个测试方法，测试当键值不在数据中时是否引发异常
    def test_key_not_in_data_raises(self, long_df):

        # 设置变量 var 和 key
        var = "x"
        key = "what"

        # 期望的错误消息
        msg = f"Could not interpret value `{key}` for `{var}`. An entry with this name"

        # 使用 pytest 的断言检查 PlotData 构造函数是否引发 ValueError 异常，并验证错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            PlotData(long_df, {var: key})

    # 定义一个测试方法，测试当键值没有对应数据时是否引发异常
    def test_key_with_no_data_raises(self):

        # 设置变量 var 和 key
        var = "x"
        key = "what"

        # 期望的错误消息
        msg = f"Could not interpret value `{key}` for `{var}`. Value is a string,"

        # 使用 pytest 的断言检查 PlotData 构造函数是否引发 ValueError 异常，并验证错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            PlotData(None, {var: key})

    # 定义一个测试方法，测试当数据向量长度不匹配时是否引发异常
    def test_data_vector_different_lengths_raises(self, long_df):

        # 创建一个长度为 (long_df 的长度 - 5) 的 ndarray 向量 vector
        vector = np.arange(len(long_df) - 5)

        # 期望的错误消息
        msg = "Length of ndarray vectors must match length of `data`"

        # 使用 pytest 的断言检查 PlotData 构造函数是否引发 ValueError 异常，并验证错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            PlotData(long_df, {"y": vector})

    # 定义一个测试方法，测试当变量未定义时是否引发异常
    def test_undefined_variables_raise(self, long_df):

        # 使用 pytest 的断言检查 PlotData 构造函数是否引发 ValueError 异常，验证错误消息是否匹配
        with pytest.raises(ValueError):
            PlotData(long_df, dict(x="not_in_df"))

        with pytest.raises(ValueError):
            PlotData(long_df, dict(x="x", y="not_in_df"))

        with pytest.raises(ValueError):
            PlotData(long_df, dict(x="x", y="y", color="not_in_df"))

    # 定义一个测试方法，测试包含操作的行为
    def test_contains_operation(self, long_df):

        # 创建 PlotData 对象 p，用 long_df 的 'y' 列作为 'x' 列，'a' 列作为 'color' 列
        p = PlotData(long_df, {"x": "y", "color": long_df["a"]})

        # 使用 pytest 的 assert 来检查是否包含特定变量
        assert "x" in p
        assert "y" not in p
        assert "color" in p

    # 定义一个测试方法，测试合并添加变量的行为
    def test_join_add_variable(self, long_df):

        # 定义两个字典 v1 和 v2，分别表示需要添加到 PlotData 对象的变量和键
        v1 = {"x": "x", "y": "f"}
        v2 = {"color": "a"}

        # 创建 PlotData 对象 p1，用 long_df 和 v1
        p1 = PlotData(long_df, v1)

        # 使用 p1 的 join 方法将 v2 合并进来，得到 p2
        p2 = p1.join(None, v2)

        # 遍历 v1 和 v2 的键值对，验证它们是否在 p2 中并且其值和 long_df 中对应列的值相等
        for var, key in dict(**v1, **v2).items():
            assert var in p2
            assert p2.names[var] == key
            assert_vector_equal(p2.frame[var], long_df[key])
    # 测试用例：测试替换变量的情况
    def test_join_replace_variable(self, long_df):

        # 定义两个字典，分别包含初始变量和要替换的变量映射关系
        v1 = {"x": "x", "y": "y"}
        v2 = {"y": "s"}

        # 创建 PlotData 对象 p1，使用长数据框和初始变量字典
        p1 = PlotData(long_df, v1)
        # 使用 p1 对象进行 join 操作，替换变量为 v2 中的映射关系，得到 p2 对象
        p2 = p1.join(None, v2)

        # 复制初始变量字典并更新为替换后的变量字典
        variables = v1.copy()
        variables.update(v2)

        # 遍历更新后的变量字典
        for var, key in variables.items():
            # 断言变量 var 存在于 p2 中
            assert var in p2
            # 断言 p2 中变量 var 对应的名称为 key
            assert p2.names[var] == key
            # 断言 p2 中变量 var 对应的数据与长数据框中 key 列相等
            assert_vector_equal(p2.frame[var], long_df[key])

    # 测试用例：测试移除变量的情况
    def test_join_remove_variable(self, long_df):

        # 定义初始变量字典和要移除的变量名
        variables = {"x": "x", "y": "f"}
        drop_var = "y"

        # 创建 PlotData 对象 p1，使用长数据框和初始变量字典
        p1 = PlotData(long_df, variables)
        # 使用 p1 对象进行 join 操作，移除 drop_var 变量，得到 p2 对象
        p2 = p1.join(None, {drop_var: None})

        # 断言 drop_var 存在于 p1 中
        assert drop_var in p1
        # 断言 drop_var 不在 p2 中
        assert drop_var not in p2
        # 断言 drop_var 不在 p2 的数据框中
        assert drop_var not in p2.frame
        # 断言 drop_var 不在 p2 的变量名字典中
        assert drop_var not in p2.names

    # 测试用例：测试替换和添加变量的各种情况
    def test_join_all_operations(self, long_df):

        # 定义初始变量字典 v1 和要替换或添加的变量字典 v2
        v1 = {"x": "x", "y": "y", "color": "a"}
        v2 = {"y": "s", "size": "s", "color": None}

        # 创建 PlotData 对象 p1，使用长数据框和初始变量字典 v1
        p1 = PlotData(long_df, v1)
        # 使用 p1 对象进行 join 操作，替换或添加变量为 v2 中的映射关系，得到 p2 对象
        p2 = p1.join(None, v2)

        # 遍历 v2 中的变量映射关系
        for var, key in v2.items():
            # 如果 key 为 None，断言变量 var 不在 p2 中
            if key is None:
                assert var not in p2
            else:
                # 否则，断言 p2 中变量 var 的名称为 key
                assert p2.names[var] == key
                # 断言 p2 中变量 var 的数据与长数据框中 key 列相等
                assert_vector_equal(p2.frame[var], long_df[key])

    # 测试用例：测试在包含全部数据的情况下进行替换和添加变量
    def test_join_all_operations_same_data(self, long_df):

        # 定义初始变量字典 v1 和要替换或添加的变量字典 v2
        v1 = {"x": "x", "y": "y", "color": "a"}
        v2 = {"y": "s", "size": "s", "color": None}

        # 创建 PlotData 对象 p1，使用长数据框和初始变量字典 v1
        p1 = PlotData(long_df, v1)
        # 使用 p1 对象进行 join 操作，替换或添加变量为 v2 中的映射关系，得到 p2 对象
        p2 = p1.join(long_df, v2)

        # 遍历 v2 中的变量映射关系
        for var, key in v2.items():
            # 如果 key 为 None，断言变量 var 不在 p2 中
            if key is None:
                assert var not in p2
            else:
                # 否则，断言 p2 中变量 var 的名称为 key
                assert p2.names[var] == key
                # 断言 p2 中变量 var 的数据与长数据框中 key 列相等
                assert_vector_equal(p2.frame[var], long_df[key])

    # 测试用例：测试添加新数据并替换或添加变量的情况
    def test_join_add_variable_new_data(self, long_df):

        # 从长数据框中抽取部分数据
        d1 = long_df[["x", "y"]]
        d2 = long_df[["a", "s"]]

        # 定义初始变量字典 v1 和要替换或添加的变量字典 v2
        v1 = {"x": "x", "y": "y"}
        v2 = {"color": "a"}

        # 创建 PlotData 对象 p1，使用 d1 数据和初始变量字典 v1
        p1 = PlotData(d1, v1)
        # 使用 p1 对象进行 join 操作，将 d2 数据和 v2 中的变量映射关系替换或添加到 p1 中，得到 p2 对象
        p2 = p1.join(d2, v2)

        # 合并 v1 和 v2 得到所有的变量映射关系
        for var, key in dict(**v1, **v2).items():
            # 断言 p2 中变量 var 的名称为 key
            assert p2.names[var] == key
            # 断言 p2 中变量 var 的数据与长数据框中 key 列相等
            assert_vector_equal(p2.frame[var], long_df[key])

    # 测试用例：测试替换已存在数据并替换或添加变量的情况
    def test_join_replace_variable_new_data(self, long_df):

        # 从长数据框中抽取部分数据
        d1 = long_df[["x", "y"]]
        d2 = long_df[["a", "s"]]

        # 定义初始变量字典 v1 和要替换或添加的变量字典 v2
        v1 = {"x": "x", "y": "y"}
        v2 = {"x": "a"}

        # 创建 PlotData 对象 p1，使用 d1 数据和初始变量字典 v1
        p1 = PlotData(d1, v1)
        # 使用 p1 对象进行 join 操作，将 d2 数据和 v2 中的变量映射关系替换或添加到 p1 中，得到 p2 对象
        p2 = p1.join(d2, v2)

        # 合并 v1 和 v2 得到所有的变量映射关系
        variables = v1.copy()
        variables.update(v2)

        # 遍历所有的变量映射关系
        for var, key in variables.items():
            # 断言 p2 中变量 var 的名称为 key
            assert p2.names[var] == key
            # 断言 p2 中变量 var 的数据与长数据框中 key 列相等
            assert_vector_equal(p2.frame[var], long_df[key])
    # 测试函数：测试在不同索引处连接数据并添加变量
    def test_join_add_variable_different_index(self, long_df):
        # 取长数据框的前70行作为d1
        d1 = long_df.iloc[:70]
        # 取长数据框的从第30行开始到末尾的数据作为d2
        d2 = long_df.iloc[30:]

        # 定义变量字典v1和v2
        v1 = {"x": "a"}
        v2 = {"y": "z"}

        # 创建PlotData对象p1，使用d1和v1作为参数
        p1 = PlotData(d1, v1)
        # 调用p1的join方法，将d2和v2作为参数，返回一个新的PlotData对象p2
        p2 = p1.join(d2, v2)

        # 从v1和v2中分别解包得到(var1, key1)和(var2, key2)
        (var1, key1), = v1.items()
        (var2, key2), = v2.items()

        # 断言：验证p2中d1索引处var1列的值与d1中key1列的值相等
        assert_vector_equal(p2.frame.loc[d1.index, var1], d1[key1])
        # 断言：验证p2中d2索引处var2列的值与d2中key2列的值相等
        assert_vector_equal(p2.frame.loc[d2.index, var2], d2[key2])

        # 断言：验证p2中d2索引与d1索引的差集处var1列的所有值是否都是缺失值NaN
        assert p2.frame.loc[d2.index.difference(d1.index), var1].isna().all()
        # 断言：验证p2中d1索引与d2索引的差集处var2列的所有值是否都是缺失值NaN
        assert p2.frame.loc[d1.index.difference(d2.index), var2].isna().all()

    # 测试函数：测试在不同索引处连接数据并替换变量
    def test_join_replace_variable_different_index(self, long_df):
        # 取长数据框的前70行作为d1
        d1 = long_df.iloc[:70]
        # 取长数据框的从第30行开始到末尾的数据作为d2
        d2 = long_df.iloc[30:]

        # 定义变量名var和两个键名k1, k2
        var = "x"
        k1, k2 = "a", "z"
        # 定义变量字典v1和v2，将var分别映射到k1和k2
        v1 = {var: k1}
        v2 = {var: k2}

        # 创建PlotData对象p1，使用d1和v1作为参数
        p1 = PlotData(d1, v1)
        # 调用p1的join方法，将d2和v2作为参数，返回一个新的PlotData对象p2
        p2 = p1.join(d2, v2)

        # 从v1和v2中分别解包得到(var1, key1)和(var2, key2)
        (var1, key1), = v1.items()
        (var2, key2), = v2.items()

        # 断言：验证p2中d2索引处var列的值与d2中k2列的值相等
        assert_vector_equal(p2.frame.loc[d2.index, var], d2[k2])
        # 断言：验证p2中d1索引与d2索引的差集处var列的所有值是否都是缺失值NaN
        assert p2.frame.loc[d1.index.difference(d2.index), var].isna().all()

    # 测试函数：测试在子集数据上继承变量
    def test_join_subset_data_inherit_variables(self, long_df):
        # 从长数据框中选取满足条件long_df["a"] == "b"的子数据框sub_df
        sub_df = long_df[long_df["a"] == "b"]

        # 定义变量名var为"y"，并创建PlotData对象p1，使用长数据框和{var: var}作为参数
        var = "y"
        p1 = PlotData(long_df, {var: var})
        # 调用p1的join方法，将sub_df和None作为参数，返回一个新的PlotData对象p2
        p2 = p1.join(sub_df, None)

        # 断言：验证p2中sub_df索引处var列的值与sub_df中var列的值相等
        assert_vector_equal(p2.frame.loc[sub_df.index, var], sub_df[var])
        # 断言：验证p2中长数据框索引与sub_df索引的差集处var列的所有值是否都是缺失值NaN
        assert p2.frame.loc[long_df.index.difference(sub_df.index), var].isna().all()

    # 测试函数：测试从原始数据继承多个变量
    def test_join_multiple_inherits_from_orig(self, rng):
        # 创建包含两列(a和b)的100行随机正态分布数据框d1
        d1 = pd.DataFrame(dict(a=rng.normal(0, 1, 100), b=rng.normal(0, 1, 100)))
        # 创建只包含一列(a)的100行随机正态分布数据框d2
        d2 = pd.DataFrame(dict(a=rng.normal(0, 1, 100)))

        # 创建PlotData对象p，使用d1和{"x": "a"}作为参数，并依次调用join方法，添加变量"y": "a"和None
        p = PlotData(d1, {"x": "a"}).join(d2, {"y": "a"}).join(None, {"y": "a"})
        # 断言：验证p中frame中"x"列的值与d1中"a"列的值相等
        assert_vector_equal(p.frame["x"], d1["a"])
        # 断言：验证p中frame中"y"列的值与d1中"a"列的值相等
        assert_vector_equal(p.frame["y"], d1["a"])

    # 测试函数：测试异常类型处理
    def test_bad_type(self, flat_list):
        # 定义错误信息
        err = "Data source must be a DataFrame or Mapping"
        # 使用pytest的raises装饰器，断言初始化PlotData对象时，传入flat_list和空字典会抛出TypeError异常
        with pytest.raises(TypeError, match=err):
            PlotData(flat_list, {})

    # 跳过测试：测试数据交换功能，前提条件是pd.api中有interchange属性
    @pytest.mark.skipif(
        condition=not hasattr(pd.api, "interchange"),
        reason="Tests behavior assuming support for dataframe interchange"
    )
    def test_data_interchange(self, mock_long_df, long_df):
        # 定义变量字典variables
        variables = {"x": "x", "y": "z", "color": "a"}
        # 创建PlotData对象p，使用mock_long_df和variables作为参数
        p = PlotData(mock_long_df, variables)
        # 遍历variables中的每个变量var和对应的列名col，断言p中frame中var列的值与long_df中col列的值相等
        for var, col in variables.items():
            assert_vector_equal(p.frame[var], long_df[col])

        # 创建PlotData对象p，使用mock_long_df和{**variables, "color": long_df["a"]}作为参数
        p = PlotData(mock_long_df, {**variables, "color": long_df["a"]})
        # 遍历variables中的每个变量var和对应的列名col，断言p中frame中var列的值与long_df中col列的值相等
        for var, col in variables.items():
            assert_vector_equal(p.frame[var], long_df[col])
    # 定义一个测试方法，用于测试数据交换失败的情况，使用 mock_long_df 作为模拟的长数据框架对象
    def test_data_interchange_failure(self, mock_long_df):

        # 模拟将 _data 属性设置为 None，破坏 __dataframe__() 方法
        mock_long_df._data = None  # Break __dataframe__()

        # 使用 pytest 的上下文管理器，检查是否抛出 RuntimeError 异常，并验证异常消息中包含指定字符串
        with pytest.raises(RuntimeError, match="Encountered an exception"):
            # 创建 PlotData 对象，传入 mock_long_df 和 {"x": "x"} 作为参数
            PlotData(mock_long_df, {"x": "x"})

    # 使用 pytest.mark.skipif 装饰器标记这个测试方法，条件是 pd.api 中存在 "interchange" 属性
    @pytest.mark.skipif(
        condition=hasattr(pd.api, "interchange"),
        reason="Tests graceful failure without support for dataframe interchange"
    )
    # 定义一个测试方法，用于测试数据交换支持情况的测试，使用 mock_long_df 作为模拟的长数据框架对象
    def test_data_interchange_support_test(self, mock_long_df):

        # 使用 pytest 的上下文管理器，检查是否抛出 TypeError 异常，并验证异常消息中包含指定字符串
        with pytest.raises(TypeError, match="Support for non-pandas DataFrame"):
            # 创建 PlotData 对象，传入 mock_long_df 和 {"x": "x"} 作为参数
            PlotData(mock_long_df, {"x": "x"})
```