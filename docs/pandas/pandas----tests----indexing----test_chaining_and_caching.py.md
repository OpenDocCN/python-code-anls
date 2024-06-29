# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_chaining_and_caching.py`

```
# 从字符串模块导入ascii_letters常量，包含所有大小写字母
from string import ascii_letters

# 导入numpy库，并重命名为np
import numpy as np

# 导入pytest库
import pytest

# 导入pandas库，并重命名为pd
import pandas as pd

# 从pandas库中导入特定模块
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
    option_context,
)

# 导入pandas内部测试模块，重命名为tm
import pandas._testing as tm

# 错误消息字符串赋值给msg变量
msg = "A value is trying to be set on a copy of a slice from a DataFrame"


# 定义一个测试类TestCaching
class TestCaching:

    # 定义测试方法test_slice_consolidate_invalidate_item_cache
    def test_slice_consolidate_invalidate_item_cache(self):
        # 进入上下文管理器，设置chained_assignment为None，解决链式赋值问题
        with option_context("chained_assignment", None):
            # 创建一个DataFrame对象，包含两列'aa'和'bb'
            df = DataFrame({"aa": np.arange(5), "bb": [2.2] * 5})

            # 向DataFrame中添加一个新列'cc'，并初始化为0.0
            df["cc"] = 0.0

            # 缓存对'bb'列的引用
            df["bb"]

            # 尝试对错误的Series进行赋值
            with tm.raises_chained_assignment_error():
                df["bb"].iloc[0] = 0.17
            
            # 断言：'bb'列的第一个元素仍为2.2
            tm.assert_almost_equal(df["bb"][0], 2.2)

    # 使用pytest的参数化装饰器，测试方法接收do_ref参数
    @pytest.mark.parametrize("do_ref", [True, False])
    def test_setitem_cache_updating(self, do_ref):
        # GH 5424
        # 定义一个包含字符串的列表cont
        cont = ["one", "two", "three", "four", "five", "six", "seven"]

        # 创建一个DataFrame对象，包含三列'a', 'b', 'c'
        df = DataFrame({"a": cont, "b": cont[3:] + cont[:3], "c": np.arange(7)})

        # 引用缓存中的值
        if do_ref:
            df.loc[0, "c"]

        # 设置新值
        df.loc[7, "c"] = 1

        # 断言：新设置的值被成功更新
        assert df.loc[0, "c"] == 0.0
        assert df.loc[7, "c"] == 1.0

    # 测试方法：验证使用切片进行赋值时是否正确更新缓存
    def test_setitem_cache_updating_slices(self):
        # GH 7084
        # 创建期望的DataFrame对象
        expected = DataFrame(
            {"A": [600, 600, 600]}, index=date_range("5/7/2014", "5/9/2014")
        )

        # 创建一个初始的DataFrame对象out
        out = DataFrame({"A": [0, 0, 0]}, index=date_range("5/7/2014", "5/9/2014"))

        # 创建一个新的DataFrame对象df
        df = DataFrame({"C": ["A", "A", "A"], "D": [100, 200, 300]})

        # 遍历df，更新out中的值
        six = Timestamp("5/7/2014")
        eix = Timestamp("5/9/2014")
        for ix, row in df.iterrows():
            out.loc[six:eix, row["C"]] = out.loc[six:eix, row["C"]] + row["D"]

        # 断言：out与期望的DataFrame对象expected相等
        tm.assert_frame_equal(out, expected)
        tm.assert_series_equal(out["A"], expected["A"])

        # 尝试使用链式索引进行赋值
        # 这种方式会抛出chained_assignment_error
        out = DataFrame({"A": [0, 0, 0]}, index=date_range("5/7/2014", "5/9/2014"))
        out_original = out.copy()
        for ix, row in df.iterrows():
            v = out[row["C"]][six:eix] + row["D"]
            with tm.raises_chained_assignment_error():
                out[row["C"]][six:eix] = v

        # 断言：out与初始的out_original相等
        tm.assert_frame_equal(out, out_original)
        tm.assert_series_equal(out["A"], out_original["A"])

        # 使用.loc方法更新值，而不是链式索引
        out = DataFrame({"A": [0, 0, 0]}, index=date_range("5/7/2014", "5/9/2014"))
        for ix, row in df.iterrows():
            out.loc[six:eix, row["C"]] += row["D"]

        # 断言：out与期望的DataFrame对象expected相等
        tm.assert_frame_equal(out, expected)
        tm.assert_series_equal(out["A"], expected["A"])
    # 定义一个测试函数，验证修改 Series 后是否清除了父 DataFrame 的缓存
    def test_altering_series_clears_parent_cache(self):
        # GH #33675：GitHub 上的 issue 编号，用于跟踪问题

        # 创建一个 DataFrame 包含两行两列的数据，指定行索引和列名
        df = DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["A", "B"])
        
        # 从 DataFrame 中选择列 "A"，返回一个 Series 对象
        ser = df["A"]

        # 向 Series 添加新条目会导致其底层数组发生变化，因此需要清除 df._item_cache 中的 "A" 条目
        ser["c"] = 5
        
        # 验证 Series 的长度是否为 3
        assert len(ser) == 3
        
        # 验证 df["A"] 和 ser 不是同一个对象
        assert df["A"] is not ser
        
        # 验证 df["A"] 的长度仍然是 2
        assert len(df["A"]) == 2
class TestChaining:
    # 测试链式赋值中设置错误的情况
    def test_setitem_chained_setfault(self):
        # GH6026
        # 数据集
        data = ["right", "left", "left", "left", "right", "left", "timeout"]

        # 创建 DataFrame 对象，包含名为 "response" 的列，数据来自于 numpy 数组
        df = DataFrame({"response": np.array(data)})
        # 创建条件掩码，标识响应列中值为 "timeout" 的位置
        mask = df.response == "timeout"
        # 断言在链式赋值错误抛出异常的情况下，DataFrame 的 "response" 列不变
        with tm.raises_chained_assignment_error():
            df.response[mask] = "none"
        # 断言 DataFrame 的内容与原始数据一致
        tm.assert_frame_equal(df, DataFrame({"response": data}))

        # 创建记录数组，用于创建 DataFrame
        recarray = np.rec.fromarrays([data], names=["response"])
        df = DataFrame(recarray)
        mask = df.response == "timeout"
        with tm.raises_chained_assignment_error():
            df.response[mask] = "none"
        tm.assert_frame_equal(df, DataFrame({"response": data}))

        # 创建包含两个响应列的 DataFrame 对象
        df = DataFrame({"response": data, "response1": data})
        # 复制 DataFrame 以备后用
        df_original = df.copy()
        mask = df.response == "timeout"
        with tm.raises_chained_assignment_error():
            df.response[mask] = "none"
        tm.assert_frame_equal(df, df_original)

        # GH 6056
        # 期望的 DataFrame 对象
        expected = DataFrame({"A": [np.nan, "bar", "bah", "foo", "bar"]})
        # 创建 DataFrame 对象，包含名为 "A" 的列，数据来自于 numpy 数组
        df = DataFrame({"A": np.array(["foo", "bar", "bah", "foo", "bar"])})
        # 在链式赋值错误抛出异常的情况下，尝试将第一个元素设置为 NaN
        with tm.raises_chained_assignment_error():
            df["A"].iloc[0] = np.nan
        # 断言 DataFrame 的头部数据与期望一致
        expected = DataFrame({"A": ["foo", "bar", "bah", "foo", "bar"]})
        result = df.head()
        tm.assert_frame_equal(result, expected)

        # 创建 DataFrame 对象，包含名为 "A" 的列，数据来自于 numpy 数组
        df = DataFrame({"A": np.array(["foo", "bar", "bah", "foo", "bar"])})
        # 在链式赋值错误抛出异常的情况下，尝试将第一个元素设置为 NaN
        with tm.raises_chained_assignment_error():
            df.A.iloc[0] = np.nan
        result = df.head()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.arm_slow
    # 测试检测链式赋值错误的情况
    def test_detect_chained_assignment(self):
        with option_context("chained_assignment", "raise"):
            # 在链式赋值设为 "raise" 的上下文中操作
            # 创建 DataFrame 对象，包含 2x2 的整数数据，列名为 "A" 和 "B"
            df = DataFrame(
                np.arange(4).reshape(2, 2), columns=list("AB"), dtype="int64"
            )
            # 复制原始 DataFrame 以备后用
            df_original = df.copy()

            # 在链式赋值错误抛出异常的情况下，尝试将第一行 "A" 列设置为 -5
            with tm.raises_chained_assignment_error():
                df["A"][0] = -5
            # 在链式赋值错误抛出异常的情况下，尝试将第二行 "A" 列设置为 -6
            with tm.raises_chained_assignment_error():
                df["A"][1] = -6
            # 断言 DataFrame 的内容与原始数据一致
            tm.assert_frame_equal(df, df_original)

    @pytest.mark.arm_slow
    # 测试检测链式赋值错误的情况
    def test_detect_chained_assignment_raises(self):
        # 在链式赋值设为 "raise" 的上下文中操作
        # 创建 DataFrame 对象，包含两列： "A" 列为整数序列， "B" 列为浮点数序列
        df = DataFrame(
            {
                "A": Series(range(2), dtype="int64"),
                "B": np.array(np.arange(2, 4), dtype=np.float64),
            }
        )
        # 复制原始 DataFrame 以备后用
        df_original = df.copy()
        # 在链式赋值错误抛出异常的情况下，尝试将第一行 "A" 列设置为 -5
        with tm.raises_chained_assignment_error():
            df["A"][0] = -5
        # 在链式赋值错误抛出异常的情况下，尝试将第二行 "A" 列设置为 -6
        with tm.raises_chained_assignment_error():
            df["A"][1] = -6
        # 断言 DataFrame 的内容与原始数据一致
        tm.assert_frame_equal(df, df_original)
    def test_detect_chained_assignment_fails(self):
        # 定义一个测试函数，用于检测连锁赋值失败的情况

        # 创建一个 DataFrame 对象，包含两列："A" 和 "B"
        df = DataFrame(
            {
                "A": Series(range(2), dtype="int64"),
                "B": np.array(np.arange(2, 4), dtype=np.float64),
            }
        )

        # 使用 pytest 检测是否会抛出连锁赋值错误
        with tm.raises_chained_assignment_error():
            # 尝试进行连锁赋值操作
            df.loc[0]["A"] = -5

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_doc_example(self):
        # 文档示例测试函数

        # 创建一个 DataFrame 对象，包含两列："a" 和 "c"
        df = DataFrame(
            {
                "a": ["one", "one", "two", "three", "two", "one", "six"],
                "c": Series(range(7), dtype="int64"),
            }
        )

        # 根据条件筛选行索引
        indexer = df.a.str.startswith("o")

        # 使用 pytest 检测是否会抛出连锁赋值错误
        with tm.raises_chained_assignment_error():
            # 尝试进行连锁赋值操作
            df[indexer]["c"] = 42

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_object_dtype(self):
        # 检测对象类型的连锁赋值情况

        # 创建一个 DataFrame 对象，包含两列："A" 和 "B"
        df = DataFrame(
            {"A": Series(["aaa", "bbb", "ccc"], dtype=object), "B": [1, 2, 3]}
        )

        # 备份原始 DataFrame
        df_original = df.copy()

        # 使用 pytest 检测是否会抛出连锁赋值错误
        with tm.raises_chained_assignment_error():
            # 尝试进行连锁赋值操作
            df["A"][0] = 111

        # 检查操作后 DataFrame 是否保持不变
        tm.assert_frame_equal(df, df_original)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_is_copy_pickle(self, temp_file):
        # 检测从 pickle 文件中读取后的连锁赋值情况

        # 创建一个 DataFrame 对象，包含一列："A"
        df = DataFrame({"A": [1, 2]})

        # 将 DataFrame 对象保存到 pickle 文件中
        path = str(temp_file)
        df.to_pickle(path)

        # 从 pickle 文件中读取 DataFrame 对象
        df2 = pd.read_pickle(path)

        # 进行列赋值操作
        df2["B"] = df2["A"]
        df2["B"] = df2["A"]

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_str(self):
        # 检测字符串处理中的连锁赋值情况

        # 生成随机索引
        idxs = np.random.default_rng(2).integers(len(ascii_letters), size=(100, 2))
        idxs.sort(axis=1)
        strings = [ascii_letters[x[0] : x[1]] for x in idxs]

        # 创建一个 DataFrame 对象，包含一列："letters"
        df = DataFrame(strings, columns=["letters"])

        # 根据条件筛选行索引
        indexer = df.letters.apply(lambda x: len(x) > 10)

        # 尝试进行连锁赋值操作
        df.loc[indexer, "letters"] = df.loc[indexer, "letters"].apply(str.lower)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_sorting(self):
        # 检测排序操作中的连锁赋值情况

        # 创建一个 DataFrame 对象，包含四列随机标准正态分布的数据
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

        # 对第一列进行排序，并获取排序后的 Series 对象
        ser = df.iloc[:, 0].sort_values()

        # 使用 pytest 检测两个 Series 对象是否相等
        tm.assert_series_equal(ser, df.iloc[:, 0].sort_values())
        tm.assert_series_equal(ser, df[0].sort_values())

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_false_positives(self):
        # 检测连锁赋值误报的情况

        # 创建一个 DataFrame 对象，包含两列："column1" 和 "column2"
        df = DataFrame({"column1": ["a", "a", "a"], "column2": [4, 8, 9]})
        
        # 转换 DataFrame 对象为字符串
        str(df)

        # 修改 "column1" 列的值
        df["column1"] = df["column1"] + "b"
        str(df)

        # 根据条件筛选行
        df = df[df["column2"] != 8]
        str(df)

        # 再次修改 "column1" 列的值
        df["column1"] = df["column1"] + "c"
        str(df)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_undefined_column(self):
        # 测试未定义列的链式赋值错误检测
        # 创建一个包含从0到8的整数的 DataFrame，列名为 "count"
        df = DataFrame(np.arange(0, 9), columns=["count"])
        # 向 DataFrame 添加一个名为 "group" 的列，所有值设为 "b"
        df["group"] = "b"
        # 备份原始 DataFrame
        df_original = df.copy()
        # 使用 pytest 的 raises_chained_assignment_error 上下文管理器
        with tm.raises_chained_assignment_error():
            # 尝试对 df 的切片进行链式赋值，会引发异常
            df.iloc[0:5]["group"] = "a"
        # 检查修改后的 df 是否与原始备份相等
        tm.assert_frame_equal(df, df_original)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_changing_dtype(self):
        # 测试改变数据类型时的链式赋值错误检测
        # 创建一个包含不同数据类型的 DataFrame
        df = DataFrame(
            {
                "A": date_range("20130101", periods=5),
                "B": np.random.default_rng(2).standard_normal(5),
                "C": np.arange(5, dtype="int64"),
                "D": ["a", "b", "c", "d", "e"],
            }
        )
        # 备份原始 DataFrame
        df_original = df.copy()

        with tm.raises_chained_assignment_error():
            # 尝试修改 df 的某个元素的数据类型，会引发异常
            df.loc[2]["D"] = "foo"
        with tm.raises_chained_assignment_error():
            # 尝试将 df 的某个整数列修改为字符串类型，会引发异常
            df.loc[2]["C"] = "foo"
        # 检查修改后的 df 是否与原始备份相等
        tm.assert_frame_equal(df, df_original)
        # TODO: 当 PDEP-6 生效时，使用 tm.raises_chained_assignment_error()
        with tm.raises_chained_assignment_error(
            extra_warnings=(FutureWarning,), extra_match=(None,)
        ):
            # 通过索引修改 df 的某个元素为字符串，会引发异常
            df["C"][2] = "foo"
        # 再次检查修改后的 df 是否与原始备份相等
        tm.assert_frame_equal(df, df_original)

    def test_setting_with_copy_bug(self):
        # 测试在副本上操作时的设置错误
        # 创建一个包含不同类型数据的 DataFrame
        df = DataFrame(
            {"a": list(range(4)), "b": list("ab.."), "c": ["a", "b", np.nan, "d"]}
        )
        # 备份原始 DataFrame
        df_original = df.copy()
        # 创建一个布尔掩码，标识列 'c' 中的 NaN 值
        mask = pd.isna(df.c)
        with tm.raises_chained_assignment_error():
            # 尝试对 df 的 'c' 列的 NaN 值赋值 'b' 列的对应值，会引发异常
            df[["c"]][mask] = df[["b"]][mask]
        # 检查修改后的 df 是否与原始备份相等
        tm.assert_frame_equal(df, df_original)

    def test_setting_with_copy_bug_no_warning(self):
        # 测试不会触发警告的设置错误
        # 创建一个 DataFrame，并获取其 'x' 列的子集
        # GH 8730
        df1 = DataFrame({"x": Series(["a", "b", "c"]), "y": Series(["d", "e", "f"])})
        df2 = df1[["x"]]

        # 这里不应该引发异常
        df2["y"] = ["g", "h", "i"]

    def test_detect_chained_assignment_warnings_errors(self):
        # 测试链式赋值警告和错误的检测
        # 创建一个包含字符串和整数的 DataFrame
        df = DataFrame({"A": ["aaa", "bbb", "ccc"], "B": [1, 2, 3]})
        with tm.raises_chained_assignment_error():
            # 尝试修改 df 的 'A' 列的第一个元素为整数，会引发异常
            df.loc[0]["A"] = 111

    @pytest.mark.parametrize("rhs", [3, DataFrame({0: [1, 2, 3, 4]})])
    def test_detect_chained_assignment_warning_stacklevel(self, rhs):
        # 测试链式赋值警告的堆栈级别
        # GH#42570
        # 创建一个 5x5 的 DataFrame
        df = DataFrame(np.arange(25).reshape(5, 5))
        # 备份原始 DataFrame
        df_original = df.copy()
        # 对 DataFrame 进行切片操作
        chained = df.loc[:3]
        # 将切片后的 DataFrame 的第二列赋值为 rhs
        chained[2] = rhs
        # 检查修改后的 df 是否与原始备份相等
        tm.assert_frame_equal(df, df_original)
    def test_chained_getitem_with_lists(self):
        # GH6394
        # 对于嵌套列表的链式索引，从0.12版本开始出现的回归问题

        # 创建一个包含两列的DataFrame，每列都是包含3个元素的numpy数组的列表
        df = DataFrame({"A": 5 * [np.zeros(3)], "B": 5 * [np.ones(3)]})
        
        # 通过链式索引获取预期的值，将结果与预期进行比较
        expected = df["A"].iloc[2]
        result = df.loc[2, "A"]
        tm.assert_numpy_array_equal(result, expected)
        
        # 通过链式索引再次获取预期的值，将结果与预期进行比较
        result2 = df.iloc[2]["A"]
        tm.assert_numpy_array_equal(result2, expected)
        
        # 通过链式索引获取预期的值，将结果与预期进行比较
        result3 = df["A"].loc[2]
        tm.assert_numpy_array_equal(result3, expected)
        
        # 直接通过链式索引获取预期的值，将结果与预期进行比较
        result4 = df["A"].iloc[2]
        tm.assert_numpy_array_equal(result4, expected)

    def test_cache_updating(self):
        # GH 4939, 确保在设置项目时更新缓存

        # 创建一个包含10行4列全为零的DataFrame，列名为['A', 'B', 'C', 'D']
        df = DataFrame(
            np.zeros((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
        )
        
        # 缓存'A'列的Series
        df["A"]
        
        # 在DataFrame末尾添加一行，用第一行的数据填充，并断言新行在'A'和'B'列的索引中
        df.loc["Hello Friend"] = df.iloc[0]
        assert "Hello Friend" in df["A"].index
        assert "Hello Friend" in df["B"].index

    def test_cache_updating2(self):
        # 10264
        # 创建一个5行5列，数据类型为int64的DataFrame，列名为['a', 'b', 'c', 'd', 'e']，行索引为[0, 1, 2, 3, 4]

        df = DataFrame(
            np.zeros((5, 5), dtype="int64"),
            columns=["a", "b", "c", "d", "e"],
            index=range(5),
        )
        
        # 在DataFrame中添加一列'f'，并初始化为0
        df["f"] = 0
        
        # 复制DataFrame以备后用
        df_orig = df.copy()
        
        # 使用pytest断言，在'f'列的第四行赋值1会引发ValueError异常
        with pytest.raises(ValueError, match="read-only"):
            df.f.values[3] = 1
        
        # 检查DataFrame是否保持不变
        tm.assert_frame_equal(df, df_orig)

    def test_iloc_setitem_chained_assignment(self):
        # GH#3970
        # 使用option_context关闭链式赋值警告，然后测试其行为

        with option_context("chained_assignment", None):
            # 创建一个包含两列的DataFrame，一列是0到4的整数，另一列是5个2.2的列表
            df = DataFrame({"aa": range(5), "bb": [2.2] * 5})
            
            # 在DataFrame中添加一个新列'cc'，并初始化为0.0
            df["cc"] = 0.0
            
            # 创建一个包含True的列表，长度与DataFrame的行数相同
            ck = [True] * len(df)
            
            # 使用tm.raises_chained_assignment_error()上下文管理器，测试在链式赋值时修改'bb'列第一个元素会引发异常
            with tm.raises_chained_assignment_error():
                df["bb"].iloc[0] = 0.13
            
            # 测试此查询不会破坏链式设置为0.15
            df.iloc[ck]
            
            # 再次测试在链式赋值时修改'bb'列第一个元素会引发异常
            with tm.raises_chained_assignment_error():
                df["bb"].iloc[0] = 0.15
            
            # 断言'bb'列第一个元素仍为2.2
            assert df["bb"].iloc[0] == 2.2

    def test_getitem_loc_assignment_slice_state(self):
        # GH 13569
        # 测试在使用loc索引赋值时，如果索引不存在则不会更改DataFrame

        # 创建一个包含一列'a'，值为[10, 20, 30]的DataFrame
        df = DataFrame({"a": [10, 20, 30]})
        
        # 使用tm.raises_chained_assignment_error()上下文管理器，测试在索引不存在时通过loc索引赋值不会更改DataFrame
        with tm.raises_chained_assignment_error():
            df["a"].loc[4] = 40
        
        # 断言DataFrame保持不变
        tm.assert_frame_equal(df, DataFrame({"a": [10, 20, 30]}))
        
        # 断言Series 'df["a"]' 与预期相同
        tm.assert_series_equal(df["a"], Series([10, 20, 30], name="a"))
```