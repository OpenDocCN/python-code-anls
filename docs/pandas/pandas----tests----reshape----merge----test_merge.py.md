# `D:\src\scipysrc\pandas\pandas\tests\reshape\merge\test_merge.py`

```
    # 导入必要的日期时间相关模块
    from datetime import (
        date,
        datetime,
        timedelta,
    )
    # 导入正则表达式模块
    import re

    # 导入第三方数值计算库numpy并简称为np
    import numpy as np
    # 导入测试框架pytest
    import pytest

    # 从pandas包中导入常用数据类型判断函数
    from pandas.core.dtypes.common import (
        is_object_dtype,
        is_string_dtype,
    )
    # 从pandas包中导入分类数据类型
    from pandas.core.dtypes.dtypes import CategoricalDtype

    # 导入pandas并简称为pd
    import pandas as pd
    # 从pandas包中导入多个常用数据结构和索引类型
    from pandas import (
        Categorical,
        CategoricalIndex,
        DataFrame,
        DatetimeIndex,
        Index,
        IntervalIndex,
        MultiIndex,
        PeriodIndex,
        RangeIndex,
        Series,
        TimedeltaIndex,
    )
    # 导入pandas测试模块，并简称为tm
    import pandas._testing as tm
    # 从pandas包中导入数据合并相关函数
    from pandas.core.reshape.concat import concat
    from pandas.core.reshape.merge import (
        MergeError,
        merge,
    )


    # 定义生成测试数据的函数
    def get_test_data(ngroups=8, n=50):
        # 生成一组唯一的组号列表
        unique_groups = list(range(ngroups))
        # 重复生成数据，以填充到指定数量
        arr = np.asarray(np.tile(unique_groups, n // ngroups))

        # 如果生成的数组长度小于指定长度，则补充额外的数据
        if len(arr) < n:
            arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])

        # 使用随机数种子为2对数组进行洗牌操作
        np.random.default_rng(2).shuffle(arr)
        return arr


    # 定义用于数据框测试的pytest fixture
    @pytest.fixture
    def dfs_for_indicator():
        # 创建两个数据框，用于数据合并测试
        df1 = DataFrame({"col1": [0, 1], "col_conflict": [1, 2], "col_left": ["a", "b"]})
        df2 = DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col_conflict": [1, 2, 3, 4, 5],
                "col_right": [2, 2, 2, 2, 2],
            }
        )
        return df1, df2


    # 定义数据合并测试类
    class TestMerge:
        # 定义pytest fixture，生成用于测试的数据框
        @pytest.fixture
        def df(self):
            # 创建包含随机数据的数据框
            df = DataFrame(
                {
                    "key1": get_test_data(),
                    "key2": get_test_data(),
                    "data1": np.random.default_rng(2).standard_normal(50),
                    "data2": np.random.default_rng(2).standard_normal(50),
                }
            )

            # 选择key2列大于1的行构成新的数据框
            df = df[df["key2"] > 1]
            return df

        # 定义pytest fixture，生成第二个用于测试的数据框
        @pytest.fixture
        def df2(self):
            # 创建包含随机数据的数据框，特定长度和组数
            return DataFrame(
                {
                    "key1": get_test_data(n=10),
                    "key2": get_test_data(ngroups=4, n=10),
                    "value": np.random.default_rng(2).standard_normal(10),
                }
            )

        # 定义pytest fixture，生成用于左连接测试的数据框
        @pytest.fixture
        def left(self):
            # 创建包含随机数据的数据框，包括key和v1列
            return DataFrame(
                {
                    "key": ["a", "b", "c", "d", "e", "e", "a"],
                    "v1": np.random.default_rng(2).standard_normal(7),
                }
            )

        # 定义测试内连接的方法
        def test_merge_inner_join_empty(self):
            # 测试空数据框与非空数据框的内连接情况，应返回空数据框
            df_empty = DataFrame()
            df_a = DataFrame({"a": [1, 2]}, index=[0, 1], dtype="int64")
            result = merge(df_empty, df_a, left_index=True, right_index=True)
            expected = DataFrame({"a": []}, dtype="int64")
            tm.assert_frame_equal(result, expected)

        # 定义测试普通合并的方法
        def test_merge_common(self, df, df2):
            # 测试两个数据框的普通合并结果是否与预期一致
            joined = merge(df, df2)
            exp = merge(df, df2, on=["key1", "key2"])
            tm.assert_frame_equal(joined, exp)
    def test_merge_non_string_columns(self):
        # 此测试用例来自于 GitHub 的一个 issue 讨论，验证处理非字符串列名的方法
        left = DataFrame(
            {0: [1, 0, 1, 0], 1: [0, 1, 0, 0], 2: [0, 0, 2, 0], 3: [1, 0, 0, 3]}
        )

        right = left.astype(float)
        expected = left
        # 执行 merge 操作，将 left 和 right 合并
        result = merge(left, right)
        # 使用测试框架检查预期输出和实际输出是否一致
        tm.assert_frame_equal(expected, result)

    def test_merge_index_as_on_arg(self, df, df2):
        # GH14355
        # 使用 "key1" 列作为索引来合并 left 和 right DataFrame
        left = df.set_index("key1")
        right = df2.set_index("key1")
        # 执行 merge 操作，使用 "key1" 作为连接键
        result = merge(left, right, on="key1")
        # 预期结果是使用 "key1" 列作为索引后合并 df 和 df2 后的结果
        expected = merge(df, df2, on="key1").set_index("key1")
        tm.assert_frame_equal(result, expected)

    def test_merge_index_singlekey_right_vs_left(self):
        left = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )

        # 左连接测试
        merged1 = merge(
            left, right, left_on="key", right_index=True, how="left", sort=False
        )
        # 右连接测试
        merged2 = merge(
            right, left, right_on="key", left_index=True, how="right", sort=False
        )
        # 使用测试框架验证左连接和右连接的结果是否一致
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

        # 排序后的左连接测试
        merged1 = merge(
            left, right, left_on="key", right_index=True, how="left", sort=True
        )
        # 排序后的右连接测试
        merged2 = merge(
            right, left, right_on="key", left_index=True, how="right", sort=True
        )
        # 使用测试框架验证排序后左连接和右连接的结果是否一致
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

    def test_merge_index_singlekey_inner(self):
        left = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )

        # 内连接测试
        result = merge(left, right, left_on="key", right_index=True, how="inner")
        # 预期结果是左表和右表按 "key" 列内连接后的结果
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected)

        # 右表作为主表的内连接测试
        result = merge(right, left, right_on="key", left_index=True, how="inner")
        # 预期结果是左表和右表按 "key" 列内连接后的结果
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected.loc[:, result.columns])
    # 测试未正确指定合并条件时是否触发异常，使用 pytest 的 raises 方法检查异常类型和匹配消息
    def test_merge_misspecified(self, df, df2, left):
        # 创建右侧 DataFrame，包含随机生成的数据列"v2"，指定索引顺序
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )
        # 设置异常消息内容
        msg = "Must pass right_on or right_index=True"
        # 检查是否抛出 MergeError 异常，消息需匹配预设的 msg
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, left_index=True)
        # 设置异常消息内容
        msg = "Must pass left_on or left_index=True"
        # 检查是否抛出 MergeError 异常，消息需匹配预设的 msg
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, right_index=True)

        # 设置异常消息内容
        msg = (
            'Can only pass argument "on" OR "left_on" and "right_on", not '
            "a combination of both"
        )
        # 检查是否抛出 MergeError 异常，消息需匹配预设的 msg
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, left, left_on="key", on="key")

        # 设置异常消息内容
        msg = r"len\(right_on\) must equal len\(left_on\)"
        # 检查是否抛出 ValueError 异常，消息需匹配预设的 msg
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on=["key1"], right_on=["key1", "key2"])

    # 测试混淆的索引和合并键参数是否触发异常，使用 pytest 的 raises 方法检查异常类型和匹配消息
    def test_index_and_on_parameters_confusion(self, df, df2):
        # 设置异常消息内容
        msg = "right_index parameter must be of type bool, not <class 'list'>"
        # 检查是否抛出 ValueError 异常，消息需匹配预设的 msg
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=False,
                right_index=["key1", "key2"],
            )
        # 设置异常消息内容
        msg = "left_index parameter must be of type bool, not <class 'list'>"
        # 检查是否抛出 ValueError 异常，消息需匹配预设的 msg
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=["key1", "key2"],
                right_index=False,
            )
        # 检查是否抛出 ValueError 异常，消息需匹配预设的 msg
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=["key1", "key2"],
                right_index=["key1", "key2"],
            )

    # 测试合并操作中重叠键的情况，验证合并结果是否符合预期
    def test_merge_overlap(self, left):
        # 执行基于"key"列的左右合并
        merged = merge(left, left, on="key")
        # 计算预期合并后的行数，应为左侧"key"值计数的平方和
        exp_len = (left["key"].value_counts() ** 2).sum()
        # 断言合并后的行数是否符合预期
        assert len(merged) == exp_len
        # 断言合并后的结果中是否包含"v1_x"列
        assert "v1_x" in merged
        # 断言合并后的结果中是否包含"v1_y"列
        assert "v1_y" in merged
    def test_intelligently_handle_join_key():
        # 创建左侧 DataFrame，包含 'key' 和 'value' 列，用于合并
        left = DataFrame(
            {"key": [1, 1, 2, 2, 3], "value": list(range(5))}, columns=["value", "key"]
        )
        # 创建右侧 DataFrame，包含 'key' 和 'rvalue' 列，用于合并
        right = DataFrame({"key": [1, 1, 2, 3, 4, 5], "rvalue": list(range(6))})

        # 使用 'key' 列进行外连接，合并左右 DataFrame
        joined = merge(left, right, on="key", how="outer")
        # 创建预期的合并后 DataFrame，包含 'value', 'key', 'rvalue' 列
        expected = DataFrame(
            {
                "key": [1, 1, 1, 1, 2, 2, 3, 4, 5],
                "value": np.array([0, 0, 1, 1, 2, 3, 4, np.nan, np.nan]),
                "rvalue": [0, 1, 0, 1, 2, 2, 3, 4, 5],
            },
            columns=["value", "key", "rvalue"],
        )
        # 断言合并后的 DataFrame 与预期结果相等
        tm.assert_frame_equal(joined, expected)
    def test_merge_join_key_dtype_cast(self):
        # #8596
        # 创建第一个数据帧 df1，包含 "key" 列为整数 1，"v1" 列为整数 10
        df1 = DataFrame({"key": [1], "v1": [10]})
        # 创建第二个数据帧 df2，包含 "key" 列为整数 2，"v1" 列为整数 20
        df2 = DataFrame({"key": [2], "v1": [20]})
        # 将 df1 和 df2 进行外连接，结果存储在 df 中
        df = merge(df1, df2, how="outer")
        # 断言合并后的 df 中 "key" 列的数据类型为 int64
        assert df["key"].dtype == "int64"

        # 创建第一个数据帧 df1，包含 "key" 列为布尔值 True，"v1" 列为整数 1
        df1 = DataFrame({"key": [True], "v1": [1]})
        # 创建第二个数据帧 df2，包含 "key" 列为布尔值 False，"v1" 列为整数 0
        df2 = DataFrame({"key": [False], "v1": [0]})
        # 将 df1 和 df2 进行外连接，结果存储在 df 中
        df = merge(df1, df2, how="outer")

        # GH13169
        # GH#40073
        # 断言合并后的 df 中 "key" 列的数据类型为布尔值 bool
        assert df["key"].dtype == "bool"

        # 创建第一个数据帧 df1，包含 "val" 列为整数 1
        df1 = DataFrame({"val": [1]})
        # 创建第二个数据帧 df2，包含 "val" 列为整数 2
        df2 = DataFrame({"val": [2]})
        # 创建左连接键 lkey 和右连接键 rkey
        lkey = np.array([1])
        rkey = np.array([2])
        # 将 df1 和 df2 根据 lkey 和 rkey 进行外连接，结果存储在 df 中
        df = merge(df1, df2, left_on=lkey, right_on=rkey, how="outer")
        # 断言合并后的 df 中 "key_0" 列的数据类型为 int 类型
        assert df["key_0"].dtype == np.dtype(int)

    def test_handle_join_key_pass_array(self):
        # 创建左侧数据帧 left，包含 "key" 列和 "value" 列，值分别为 [1, 1, 2, 2, 3] 和 [0, 1, 2, 3, 4]
        left = DataFrame(
            {"key": [1, 1, 2, 2, 3], "value": np.arange(5)},
            columns=["value", "key"],
            dtype="int64",
        )
        # 创建右侧数据帧 right，包含 "rvalue" 列，值为 [0, 1, 2, 3, 4, 5]
        right = DataFrame({"rvalue": np.arange(6)}, dtype="int64")
        # 创建连接键 key，值为 [1, 1, 2, 3, 4, 5]
        key = np.array([1, 1, 2, 3, 4, 5], dtype="int64")

        # 将 left 和 right 根据 "key" 列和 key 进行外连接，结果存储在 merged 中
        merged = merge(left, right, left_on="key", right_on=key, how="outer")
        # 将 right 和 left 根据 key 和 "key" 列进行外连接，结果存储在 merged2 中
        merged2 = merge(right, left, left_on=key, right_on="key", how="outer")

        # 断言 merged 和 merged2 中的 "key" 列相等
        tm.assert_series_equal(merged["key"], merged2["key"])
        # 断言 merged 中的 "key" 列所有值均非空
        assert merged["key"].notna().all()
        # 断言 merged2 中的 "key" 列所有值均非空
        assert merged2["key"].notna().all()

        # 创建左侧数据帧 left，包含 "value" 列，值为 [0, 1, 2, 3, 4]
        left = DataFrame({"value": np.arange(5)}, columns=["value"])
        # 创建右侧数据帧 right，包含 "rvalue" 列，值为 [0, 1, 2, 3, 4, 5]
        right = DataFrame({"rvalue": np.arange(6)})
        # 创建左连接键 lkey，值为 [1, 1, 2, 2, 3]
        lkey = np.array([1, 1, 2, 2, 3])
        # 创建右连接键 rkey，值为 [1, 1, 2, 3, 4, 5]
        rkey = np.array([1, 1, 2, 3, 4, 5])

        # 将 left 和 right 根据 lkey 和 rkey 进行外连接，结果存储在 merged 中
        merged = merge(left, right, left_on=lkey, right_on=rkey, how="outer")
        # 创建预期结果 Series，包含 [1, 1, 1, 1, 2, 2, 3, 4, 5]，数据类型为 int，名称为 "key_0"
        expected = Series([1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=int, name="key_0")
        # 断言 merged 中的 "key_0" 列与预期结果相等
        tm.assert_series_equal(merged["key_0"], expected)

        # 创建左侧数据帧 left，包含 "value" 列，值为 [0, 1, 2]
        left = DataFrame({"value": np.arange(3)})
        # 创建右侧数据帧 right，包含 "rvalue" 列，值为 [0, 1, 2, 3, 4, 5]
        right = DataFrame({"rvalue": np.arange(6)})

        # 创建连接键 key，值为 [0, 1, 1, 2, 2, 3]，数据类型为 int64
        key = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        # 将 left 和 right 根据 left 的索引和 key 进行外连接，结果存储在 merged 中
        merged = merge(left, right, left_index=True, right_on=key, how="outer")
        # 断言 merged 中的 "key_0" 列与 Series key 相等，名称为 "key_0"
        tm.assert_series_equal(merged["key_0"], Series(key, name="key_0"))

    def test_no_overlap_more_informative_error(self):
        # 获取当前时间并赋值给变量 dt
        dt = datetime.now()
        # 创建数据帧 df1，包含 "x" 列为字符串 "a"，索引为当前时间 dt
        df1 = DataFrame({"x": ["a"]}, index=[dt])

        # 创建数据帧 df2，包含 "y" 列为字符串列表 ["b", "c"]，索引为当前时间 dt 和 dt
        df2 = DataFrame({"y": ["b", "c"]}, index=[dt, dt])

        # 创建错误消息 msg，描述无公共列可进行合并的情况
        msg = (
            "No common columns to perform merge on. "
            f"Merge options: left_on={None}, right_on={None}, "
            f"left_index={False}, right_index={False}"
        )

        # 使用 pytest 框架断言 merge 操作会引发 MergeError，并匹配预期的错误消息
        with pytest.raises(MergeError, match=msg):
            merge(df1, df2)
    def test_merge_non_unique_indexes(self):
        # 创建几个日期时间对象
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        dt4 = datetime(2012, 5, 4)

        # 创建 DataFrame df1，包含单个索引为 dt 的列 'x'
        df1 = DataFrame({"x": ["a"]}, index=[dt])
        
        # 创建 DataFrame df2，包含索引为 dt 重复的列 'y' 包含两个值 'b', 'c'
        df2 = DataFrame({"y": ["b", "c"]}, index=[dt, dt])
        
        # 调用 _check_merge 函数，对 df1 和 df2 进行合并测试
        _check_merge(df1, df2)

        # Not monotonic
        # 创建 df1，包含索引为 dt2, dt, dt4 的列 'x'
        df1 = DataFrame({"x": ["a", "b", "q"]}, index=[dt2, dt, dt4])
        
        # 创建 df2，包含索引为 dt3, dt3, dt2, dt2, dt, dt 的列 'y' 包含六个值 'c', 'd', 'e', 'f', 'g', 'h'
        df2 = DataFrame({"y": ["c", "d", "e", "f", "g", "h"]}, index=[dt3, dt3, dt2, dt2, dt, dt])
        
        # 调用 _check_merge 函数，对 df1 和 df2 进行合并测试
        _check_merge(df1, df2)

        # 创建 df1，包含索引为 dt, dt 的列 'x'，相同索引值
        df1 = DataFrame({"x": ["a", "b"]}, index=[dt, dt])
        
        # 创建 df2，包含索引为 dt, dt 的列 'y'，相同索引值
        df2 = DataFrame({"y": ["c", "d"]}, index=[dt, dt])
        
        # 调用 _check_merge 函数，对 df1 和 df2 进行合并测试
        _check_merge(df1, df2)

    def test_merge_non_unique_index_many_to_many(self):
        # 创建几个日期时间对象
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        
        # 创建 df1，包含索引为 dt2, dt2, dt, dt 的列 'x'，有重复索引
        df1 = DataFrame({"x": ["a", "b", "c", "d"]}, index=[dt2, dt2, dt, dt])
        
        # 创建 df2，包含索引为 dt2, dt2, dt3, dt, dt 的列 'y'，有重复索引
        df2 = DataFrame({"y": ["e", "f", "g", " h", "i"]}, index=[dt2, dt2, dt3, dt, dt])
        
        # 调用 _check_merge 函数，对 df1 和 df2 进行合并测试
        _check_merge(df1, df2)

    def test_left_merge_empty_dataframe(self):
        # 创建包含键为 [1] 和值为 [2] 的 DataFrame left
        left = DataFrame({"key": [1], "value": [2]})
        
        # 创建一个空的 DataFrame right，仅包含键列
        right = DataFrame({"key": []})
        
        # 对 left 和 right 进行左连接，使用键 "key"，方式为左连接
        result = merge(left, right, on="key", how="left")
        
        # 断言结果与 left 相等
        tm.assert_frame_equal(result, left)
        
        # 对 right 和 left 进行右连接，使用键 "key"，方式为右连接
        result = merge(right, left, on="key", how="right")
        
        # 断言结果与 left 相等
        tm.assert_frame_equal(result, left)

    def test_merge_empty_dataframe(self, index, join_type):
        # GH52777
        # 创建一个空的 DataFrame left，使用指定的索引
        left = DataFrame([], index=index[:0])
        
        # 创建一个与 left 相同的空 DataFrame right
        right = left.copy()
        
        # 对 left 和 right 进行连接，方式由 join_type 指定
        result = left.join(right, how=join_type)
        
        # 断言结果与 left 相等
        tm.assert_frame_equal(result, left)

    @pytest.mark.parametrize(
        "kwarg",
        [
            {"left_index": True, "right_index": True},
            {"left_index": True, "right_on": "x"},
            {"left_on": "a", "right_index": True},
            {"left_on": "a", "right_on": "x"},
        ],
    )
    def test_merge_left_empty_right_empty(self, join_type, kwarg):
        # GH 10824
        # 创建一个列为 ["a", "b", "c"] 的空 DataFrame left
        left = DataFrame(columns=["a", "b", "c"])
        
        # 创建一个列为 ["x", "y", "z"] 的空 DataFrame right
        right = DataFrame(columns=["x", "y", "z"])
        
        # 创建一个预期结果 DataFrame，包含左右 DataFrame 的所有列
        exp_in = DataFrame(columns=["a", "b", "c", "x", "y", "z"], dtype=object)
        
        # 对 left 和 right 进行连接，方式由 join_type 指定，使用 kwarg 指定的参数
        result = merge(left, right, how=join_type, **kwarg)
        
        # 断言结果与预期结果 exp_in 相等
        tm.assert_frame_equal(result, exp_in)
    # 定义一个测试函数，用于测试左表为空而右表不为空的情况
    def test_merge_left_empty_right_notempty(self):
        # GH 10824
        # 创建一个空的 DataFrame，列名为 ["a", "b", "c"]
        left = DataFrame(columns=["a", "b", "c"])
        # 创建一个非空的 DataFrame，包含三行数据和列名 ["x", "y", "z"]
        right = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["x", "y", "z"])

        # 期望的输出结果，合并后的 DataFrame
        exp_out = DataFrame(
            {
                "a": np.array([np.nan] * 3, dtype=object),
                "b": np.array([np.nan] * 3, dtype=object),
                "c": np.array([np.nan] * 3, dtype=object),
                "x": [1, 4, 7],
                "y": [2, 5, 8],
                "z": [3, 6, 9],
            },
            columns=["a", "b", "c", "x", "y", "z"],
        )
        # 期望的输入结果，空的 DataFrame，保持数据类型
        exp_in = exp_out[0:0]  # make empty DataFrame keeping dtype

        # 定义内部函数 check1，用于检查合并结果是否与期望结果相等
        def check1(exp, kwarg):
            # 使用内部参数进行左连接和内连接的合并，并进行结果比较
            result = merge(left, right, how="inner", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="left", **kwarg)
            tm.assert_frame_equal(result, exp)

        # 定义内部函数 check2，用于检查合并结果是否与期望结果相等
        def check2(exp, kwarg):
            # 使用内部参数进行右连接和外连接的合并，并进行结果比较
            result = merge(left, right, how="right", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="outer", **kwarg)
            tm.assert_frame_equal(result, exp)

        # 遍历不同的参数组合进行测试
        for kwarg in [
            {"left_index": True, "right_index": True},  # 左右表使用索引进行连接
            {"left_index": True, "right_on": "x"},  # 左表使用索引，右表使用指定列名进行连接
        ]:
            check1(exp_in, kwarg)  # 检查内连接和左连接
            check2(exp_out, kwarg)  # 检查右连接和外连接

        # 使用指定的参数组合进行测试，左表使用列 "a"，右表使用索引进行连接
        kwarg = {"left_on": "a", "right_index": True}
        check1(exp_in, kwarg)  # 检查内连接和左连接
        exp_out["a"] = [0, 1, 2]  # 修改期望输出结果中的列 "a"
        check2(exp_out, kwarg)  # 检查右连接和外连接

        # 使用指定的参数组合进行测试，左表使用列 "a"，右表使用列 "x" 进行连接
        kwarg = {"left_on": "a", "right_on": "x"}
        check1(exp_in, kwarg)  # 检查内连接和左连接
        exp_out["a"] = np.array([np.nan] * 3, dtype=object)  # 修改期望输出结果中的列 "a"
        check2(exp_out, kwarg)  # 检查右连接和外连接
    # 定义一个测试方法，测试左侧 DataFrame 非空，右侧 DataFrame 为空的情况
    def test_merge_left_notempty_right_empty(self):
        # GH 10824
        # 创建一个包含数据的左侧 DataFrame，列名为 ["a", "b", "c"]
        left = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])
        # 创建一个空的右侧 DataFrame，列名为 ["x", "y", "z"]
        right = DataFrame(columns=["x", "y", "z"])

        # 期望的输出 DataFrame，包含合并后的结果，列为 ["a", "b", "c", "x", "y", "z"]
        exp_out = DataFrame(
            {
                "a": [1, 4, 7],
                "b": [2, 5, 8],
                "c": [3, 6, 9],
                "x": np.array([np.nan] * 3, dtype=object),
                "y": np.array([np.nan] * 3, dtype=object),
                "z": np.array([np.nan] * 3, dtype=object),
            },
            columns=["a", "b", "c", "x", "y", "z"],
        )
        # 用于保持空的 DataFrame，保留其数据类型
        exp_in = exp_out[0:0]  # make empty DataFrame keeping dtype
        # 将空 DataFrame 的索引转换为 object 类型
        exp_in.index = exp_in.index.astype(object)

        # 定义一个内部函数 check1，用于检查不同合并方式（inner 和 right）的结果
        def check1(exp, kwarg):
            # 执行 inner 和 right 合并，检查结果是否与期望一致
            result = merge(left, right, how="inner", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="right", **kwarg)
            tm.assert_frame_equal(result, exp)

        # 定义一个内部函数 check2，用于检查不同合并方式（left 和 outer）的结果
        def check2(exp, kwarg):
            # 执行 left 和 outer 合并，检查结果是否与期望一致
            result = merge(left, right, how="left", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="outer", **kwarg)
            tm.assert_frame_equal(result, exp)

            # 对于以下不同的合并参数，依次执行 check1 和 check2 方法
            for kwarg in [
                {"left_index": True, "right_index": True},
                {"left_index": True, "right_on": "x"},
                {"left_on": "a", "right_index": True},
                {"left_on": "a", "right_on": "x"},
            ]:
                check1(exp_in, kwarg)
                check2(exp_out, kwarg)

    # 使用 pytest.mark.parametrize 标记的测试参数化方法，测试不同数据类型的 Series
    @pytest.mark.parametrize(
        "series_of_dtype",
        [
            Series([1], dtype="int64"),
            Series([1], dtype="Int64"),
            Series([1.23]),
            Series(["foo"]),
            Series([True]),
            Series([pd.Timestamp("2018-01-01")]),
            Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
        ],
    )
    # 使用 pytest.mark.parametrize 标记的测试参数化方法，测试不同数据类型的 Series
    @pytest.mark.parametrize(
        "series_of_dtype2",
        [
            Series([1], dtype="int64"),
            Series([1], dtype="Int64"),
            Series([1.23]),
            Series(["foo"]),
            Series([True]),
            Series([pd.Timestamp("2018-01-01")]),
            Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
        ],
    )
    # 定义一个测试方法，用于测试空数据框的合并操作
    def test_merge_empty_frame(self, series_of_dtype, series_of_dtype2):
        # GH 25183
        # 创建一个包含指定数据列的 DataFrame 对象
        df = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype2},
            columns=["key", "value"],
        )
        # 创建一个空的 DataFrame，只保留列名
        df_empty = df[:0]
        # 创建预期的 DataFrame，包含指定列的 Series 对象
        expected = DataFrame(
            {
                "key": Series(dtype=df.dtypes["key"]),
                "value_x": Series(dtype=df.dtypes["value"]),
                "value_y": Series(dtype=df.dtypes["value"]),
            },
            columns=["key", "value_x", "value_y"],
        )
        # 执行空数据框与原数据框的合并操作
        actual = df_empty.merge(df, on="key")
        # 使用测试框架中的方法比较实际结果和预期结果的数据框
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "series_of_dtype",
        [
            Series([1], dtype="int64"),
            Series([1], dtype="Int64"),
            Series([1.23]),
            Series(["foo"]),
            Series([True]),
            Series([pd.Timestamp("2018-01-01")]),
            Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
        ],
    )
    @pytest.mark.parametrize(
        "series_of_dtype_all_na",
        [
            Series([np.nan], dtype="Int64"),
            Series([np.nan], dtype="float"),
            Series([np.nan], dtype="object"),
            Series([pd.NaT]),
        ],
    )
    # 定义一个测试方法，用于测试包含所有值为 NA 的列的合并操作
    def test_merge_all_na_column(self, series_of_dtype, series_of_dtype_all_na):
        # GH 25183
        # 创建左侧和右侧数据框，每个都包含指定的数据列
        df_left = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype_all_na},
            columns=["key", "value"],
        )
        df_right = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype_all_na},
            columns=["key", "value"],
        )
        # 创建预期的 DataFrame，包含指定列的 Series 对象
        expected = DataFrame(
            {
                "key": series_of_dtype,
                "value_x": series_of_dtype_all_na,
                "value_y": series_of_dtype_all_na,
            },
            columns=["key", "value_x", "value_y"],
        )
        # 执行左右数据框的合并操作
        actual = df_left.merge(df_right, on="key")
        # 使用测试框架中的方法比较实际结果和预期结果的数据框
        tm.assert_frame_equal(actual, expected)
    def test_merge_nosort(self):
        # 定义一个测试函数，用于测试不排序的数据合并操作

        # 创建一个包含多个变量的字典数据集合
        d = {
            "var1": np.random.default_rng(2).integers(0, 10, size=10),
            "var2": np.random.default_rng(2).integers(0, 10, size=10),
            "var3": [
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2010, 2, 3),
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2012, 4, 3),
                datetime(2012, 3, 4),
                datetime(2008, 5, 1),
                datetime(2010, 2, 3),
                datetime(2012, 2, 3),
            ],
        }
        # 根据字典创建一个数据框
        df = DataFrame.from_dict(d)
        
        # 提取 df 中 var3 列的唯一值
        var3 = df.var3.unique()
        # 对 var3 列进行排序
        var3 = np.sort(var3)
        
        # 创建一个新的数据框，包含 var3 和随机生成的 var8 列
        new = DataFrame.from_dict(
            {"var3": var3, "var8": np.random.default_rng(2).random(7)}
        )

        # 执行不排序的数据框合并操作，得到结果 result
        result = df.merge(new, on="var3", sort=False)
        # 期望的合并结果，使用 merge 函数生成 exp
        exp = merge(df, new, on="var3", sort=False)
        
        # 使用测试框架检查 result 和 exp 是否相等
        tm.assert_frame_equal(result, exp)

        # 断言 df.var3 的唯一值与 result.var3 的唯一值是否全部相等
        assert (df.var3.unique() == result.var3.unique()).all()

    @pytest.mark.parametrize(
        ("sort", "values"), [(False, [1, 1, 0, 1, 1]), (True, [0, 1, 1, 1, 1])]
    )
    @pytest.mark.parametrize("how", ["left", "right"])
    def test_merge_same_order_left_right(self, sort, values, how):
        # 定义一个测试函数，用于测试相同顺序下左右合并操作

        # 创建一个包含单列 'a' 的数据框
        df = DataFrame({"a": [1, 0, 1]})

        # 执行左右合并操作，根据参数 how 和 sort 进行不同方式的合并
        result = df.merge(df, on="a", how=how, sort=sort)
        # 期望的合并结果，创建一个包含 values 值的数据框
        expected = DataFrame(values, columns=["a"])
        # 使用测试框架检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_merge_nan_right(self):
        # 定义一个测试函数，用于测试右侧数据含 NaN 值的合并操作

        # 创建两个包含不同列的数据框
        df1 = DataFrame({"i1": [0, 1], "i2": [0, 1]})
        df2 = DataFrame({"i1": [0], "i3": [0]})
        
        # 执行基于 'i1' 列的合并操作，并指定右侧数据框的列名后缀
        result = df1.join(df2, on="i1", rsuffix="_")
        
        # 期望的合并结果，使用 DataFrame 构造函数创建一个期望的数据框
        expected = (
            DataFrame(
                {
                    "i1": {0: 0.0, 1: 1},
                    "i2": {0: 0, 1: 1},
                    "i1_": {0: 0, 1: np.nan},
                    "i3": {0: 0.0, 1: np.nan},
                    None: {0: 0, 1: 0},
                },
                columns=Index(["i1", "i2", "i1_", "i3", None], dtype=object),
            )
            .set_index(None)
            .reset_index()[["i1", "i2", "i1_", "i3"]]
        )
        
        # 修改 result 列名的数据类型为 object，并使用测试框架检查 result 和 expected 是否相等
        result.columns = result.columns.astype("object")
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_merge_nan_right2(self):
        # 定义一个测试函数，用于测试右侧数据含 NaN 值的合并操作（第二种情况）

        # 创建两个包含不同列的数据框
        df1 = DataFrame({"i1": [0, 1], "i2": [0.5, 1.5]})
        df2 = DataFrame({"i1": [0], "i3": [0.7]})
        
        # 执行基于 'i1' 列的合并操作，并指定右侧数据框的列名后缀
        result = df1.join(df2, rsuffix="_", on="i1")
        
        # 期望的合并结果，使用 DataFrame 构造函数创建一个期望的数据框
        expected = DataFrame(
            {
                "i1": {0: 0, 1: 1},
                "i1_": {0: 0.0, 1: np.nan},
                "i2": {0: 0.5, 1: 1.5},
                "i3": {0: 0.69999999999999996, 1: np.nan},
            }
        )[["i1", "i2", "i1_", "i3"]]
        
        # 使用测试框架检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
    )
    def test_merge_type(self, df, df2):
        # 定义一个特定的 DataFrame 子类 NotADataFrame
        class NotADataFrame(DataFrame):
            # 定义 _constructor 属性，返回自身类型 NotADataFrame
            @property
            def _constructor(self):
                return NotADataFrame

        # 创建 NotADataFrame 的实例 nad，传入 df 数据
        nad = NotADataFrame(df)
        # 使用 merge 方法将 df2 与 nad 合并，按照 "key1" 列进行合并
        result = nad.merge(df2, on="key1")

        # 断言 result 的类型是 NotADataFrame
        assert isinstance(result, NotADataFrame)

    def test_join_append_timedeltas(self):
        # 处理 timedelta64 在 join/merge 操作中的问题
        # GitHub issue 5695

        # 创建包含日期时间和时间增量的 DataFrame d
        d = DataFrame.from_dict(
            {"d": [datetime(2013, 11, 5, 5, 56)], "t": [timedelta(0, 22500)]}
        )
        # 创建一个空 DataFrame df
        df = DataFrame(columns=list("dt"))
        # 将 d 连接到 df，忽略索引，生成新的 DataFrame
        df = concat([df, d], ignore_index=True)
        # 将 d 连接到 df，忽略索引，生成新的 DataFrame result
        result = concat([df, d], ignore_index=True)
        # 创建期望的 DataFrame expected
        expected = DataFrame(
            {
                "d": [datetime(2013, 11, 5, 5, 56), datetime(2013, 11, 5, 5, 56)],
                "t": [timedelta(0, 22500), timedelta(0, 22500)],
            },
            dtype=object,
        )
        # 使用测试框架的 assert_frame_equal 方法比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_join_append_timedeltas2(self):
        # 处理 timedelta64 在 join/merge 操作中的问题
        # GitHub issue 5695

        # 创建包含时间增量的 DataFrame lhs 和 rhs
        td = np.timedelta64(300000000)
        lhs = DataFrame(Series([td, td], index=["A", "B"]))
        rhs = DataFrame(Series([td], index=["A"]))

        # 使用 join 方法将 rhs 连接到 lhs 的左侧，使用左侧列的 rsuffix 为 "r"
        result = lhs.join(rhs, rsuffix="r", how="left")
        # 创建期望的 DataFrame expected
        expected = DataFrame(
            {
                "0": Series([td, td], index=list("AB")),
                "0r": Series([td, pd.NaT], index=list("AB")),
            }
        )
        # 使用测试框架的 assert_frame_equal 方法比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_other_datetime_unit(self, unit):
        # 处理不同时间单位的问题
        # GitHub issue 13389

        # 创建一个包含实体 ID 的 DataFrame df1
        df1 = DataFrame({"entity_id": [101, 102]})
        # 创建一个包含空值的 Series ser，使用实体 ID 作为索引，名称为 "days"
        ser = Series([None, None], index=[101, 102], name="days")

        # 根据给定的时间单位构造相应的 dtype
        dtype = f"datetime64[{unit}]"

        if unit in ["D", "h", "m"]:
            # 如果单位是 "D", "h", "m"，则将其转换为最近支持的单位 "seconds"
            exp_dtype = "datetime64[s]"
        else:
            exp_dtype = dtype
        # 将 ser 转换为指定 dtype 的 DataFrame df2，并命名为 "days"
        df2 = ser.astype(exp_dtype).to_frame("days")
        # 断言 df2 中 "days" 列的 dtype 是否为 exp_dtype
        assert df2["days"].dtype == exp_dtype

        # 使用 merge 方法将 df1 和 df2 按实体 ID 列进行合并，生成 result
        result = df1.merge(df2, left_on="entity_id", right_index=True)

        # 创建一个包含 "nat" 值的数组 days，使用 exp_dtype 作为 dtype
        days = np.array(["nat", "nat"], dtype=exp_dtype)
        days = pd.core.arrays.DatetimeArray._simple_new(days, dtype=days.dtype)
        # 创建期望的 DataFrame exp
        exp = DataFrame(
            {
                "entity_id": [101, 102],
                "days": days,
            },
            columns=["entity_id", "days"],
        )
        # 断言 exp 中 "days" 列的 dtype 是否为 exp_dtype，并比较 result 和 exp 是否相等
        assert exp["days"].dtype == exp_dtype
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    # 测试不同的时间单位是否能正确处理
    def test_other_timedelta_unit(self, unit):
        # GH 13389
        # 创建包含两列的 DataFrame
        df1 = DataFrame({"entity_id": [101, 102]})
        # 创建包含两个空值的 Series，并指定索引
        ser = Series([None, None], index=[101, 102], name="days")

        # 根据时间单位构建数据类型
        dtype = f"m8[{unit}]"
        if unit in ["D", "h", "m"]:
            # 如果时间单位为天、小时或分钟，则抛出异常
            msg = "Supported resolutions are 's', 'ms', 'us', 'ns'"
            with pytest.raises(ValueError, match=msg):
                ser.astype(dtype)

            # 将 Series 转换为秒单位的时间类型
            df2 = ser.astype("m8[s]").to_frame("days")
        else:
            # 否则将 Series 转换为指定时间单位的时间类型
            df2 = ser.astype(dtype).to_frame("days")
            assert df2["days"].dtype == dtype

        # 将两个 DataFrame 进行合并
        result = df1.merge(df2, left_on="entity_id", right_index=True)

        # 创建预期的 DataFrame
        exp = DataFrame(
            {"entity_id": [101, 102], "days": np.array(["nat", "nat"], dtype=dtype)},
            columns=["entity_id", "days"],
        )
        # 检查合并后的结果是否与预期一致
        tm.assert_frame_equal(result, exp)

    # 测试重叠列的错误消息
    def test_overlapping_columns_error_message(self):
        # 创建包含重叠列的两个 DataFrame
        df = DataFrame({"key": [1, 2, 3], "v1": [4, 5, 6], "v2": [7, 8, 9]})
        df2 = DataFrame({"key": [1, 2, 3], "v1": [4, 5, 6], "v2": [7, 8, 9]})

        # 修改列名使其重叠
        df.columns = ["key", "foo", "foo"]
        df2.columns = ["key", "bar", "bar"]
        # 创建预期的 DataFrame
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "v1": [4, 5, 6],
                "v2": [7, 8, 9],
                "v3": [4, 5, 6],
                "v4": [7, 8, 9],
            }
        )
        expected.columns = ["key", "foo", "foo", "bar", "bar"]
        # 检查合并后的结果是否与预期一致
        tm.assert_frame_equal(merge(df, df2), expected)

        # #2649, #10639
        # 修改列名使其重叠
        df2.columns = ["key1", "foo", "foo"]
        # 检查是否会抛出合并错误
        msg = r"Data columns not unique: Index\(\['foo'\], dtype='object|string'\)"
        with pytest.raises(MergeError, match=msg):
            merge(df, df2)

    # 测试在带有时区的 datetime64 上进行合并
    def test_merge_on_datetime64tz(self):
        # GH11405
        # 创建带有时区信息的两个 DataFrame
        left = DataFrame(
            {
                "key": pd.date_range("20151010", periods=2, tz="US/Eastern"),
                "value": [1, 2],
            }
        )
        right = DataFrame(
            {
                "key": pd.date_range("20151011", periods=3, tz="US/Eastern"),
                "value": [1, 2, 3],
            }
        )

        # 创建预期的 DataFrame
        expected = DataFrame(
            {
                "key": pd.date_range("20151010", periods=4, tz="US/Eastern"),
                "value_x": [1, 2, np.nan, np.nan],
                "value_y": [np.nan, 1, 2, 3],
            }
        )
        # 执行合并操作并检查结果是否与预期一致
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，用于测试处理带有 datetime64tz 类型值的数据合并情况
    def test_merge_datetime64tz_values(self):
        # 创建左侧 DataFrame，包含 key 列和带有 US/Eastern 时区的日期范围
        left = DataFrame(
            {
                "key": [1, 2],
                "value": pd.date_range("20151010", periods=2, tz="US/Eastern"),
            }
        )
        # 创建右侧 DataFrame，包含 key 列和带有 US/Eastern 时区的日期范围
        right = DataFrame(
            {
                "key": [2, 3],
                "value": pd.date_range("20151011", periods=2, tz="US/Eastern"),
            }
        )
        # 创建预期的结果 DataFrame，包含合并后的 key 列和 value_x, value_y 列
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "value_x": list(pd.date_range("20151010", periods=2, tz="US/Eastern")) + [pd.NaT],
                "value_y": [pd.NaT] + list(pd.date_range("20151011", periods=2, tz="US/Eastern")),
            }
        )
        # 执行合并操作，将左右 DataFrame 按 key 列进行外连接
        result = merge(left, right, on="key", how="outer")
        # 断言合并后的结果与预期的结果相等
        tm.assert_frame_equal(result, expected)
        # 断言合并后的结果中 value_x 列的数据类型为 datetime64[ns, US/Eastern]
        assert result["value_x"].dtype == "datetime64[ns, US/Eastern]"
        # 断言合并后的结果中 value_y 列的数据类型为 datetime64[ns, US/Eastern]

        assert result["value_y"].dtype == "datetime64[ns, US/Eastern]"

    # 定义测试函数，用于测试处理带有 datetime64tz 类型值的数据合并情况（处理空的情况）
    def test_merge_on_datetime64tz_empty(self):
        # 创建具有 'UTC' 时区的 DatetimeIndex 类型数据类型
        dtz = pd.DatetimeTZDtype(tz="UTC")
        # 创建右侧 DataFrame，包含 date, value, date2 列，使用上述的 DatetimeIndex 类型
        right = DataFrame(
            {
                "date": DatetimeIndex(["2018"], dtype=dtz),
                "value": [4.0],
                "date2": DatetimeIndex(["2019"], dtype=dtz),
            },
            columns=["date", "value", "date2"],
        )
        # 创建左侧 DataFrame，为空
        left = right[:0]
        # 执行左右 DataFrame 按 date 列进行合并操作
        result = left.merge(right, on="date")
        # 创建预期的结果 DataFrame，包含 date, value_x, date2_x, value_y, date2_y 列
        expected = DataFrame(
            {
                "date": Series(dtype=dtz),
                "value_x": Series(dtype=float),
                "date2_x": Series(dtype=dtz),
                "value_y": Series(dtype=float),
                "date2_y": Series(dtype=dtz),
            },
            columns=["date", "value_x", "date2_x", "value_y", "date2_y"],
        )
        # 断言合并后的结果与预期的结果相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，用于测试处理带有 dst 转换的 datetime64tz 类型值的数据合并情况
    def test_merge_datetime64tz_with_dst_transition(self):
        # 创建 DataFrame df1，包含 'Europe/Madrid' 时区的日期范围和值为 1 的 value 列
        df1 = DataFrame(
            pd.date_range("2017-10-29 01:00", periods=4, freq="h", tz="Europe/Madrid"),
            columns=["date"],
        )
        df1["value"] = 1
        # 创建 DataFrame df2，包含特定日期和值为 2 的 value 列，并进行时区处理
        df2 = DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2017-10-29 03:00:00",
                        "2017-10-29 04:00:00",
                        "2017-10-29 05:00:00",
                    ]
                ),
                "value": 2,
            }
        )
        df2["date"] = df2["date"].dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid")
        # 执行 df1 和 df2 按 date 列进行外连接的合并操作
        result = merge(df1, df2, how="outer", on="date")
        # 创建预期的结果 DataFrame，包含合并后的 date 列和 value_x, value_y 列
        expected = DataFrame(
            {
                "date": pd.date_range(
                    "2017-10-29 01:00", periods=7, freq="h", tz="Europe/Madrid"
                ),
                "value_x": [1] * 4 + [np.nan] * 3,
                "value_y": [np.nan] * 4 + [2] * 3,
            }
        )
        # 断言合并后的结果与预期的结果相等
        tm.assert_frame_equal(result, expected)
    def test_merge_non_unique_period_index(self):
        # 测试非唯一周期索引合并
        # 创建一个包含16个月的周期索引
        index = pd.period_range("2016-01-01", periods=16, freq="M")
        # 根据周期索引创建DataFrame，列名为'pnum'
        df = DataFrame(list(range(len(index))), index=index, columns=["pnum"])
        # 将DataFrame df与自身合并
        df2 = concat([df, df])
        # 使用周期索引进行内连接合并
        result = df.merge(df2, left_index=True, right_index=True, how="inner")
        # 创建预期的DataFrame，使用重复的值填充
        expected = DataFrame(
            np.tile(np.arange(16, dtype=np.int64).repeat(2).reshape(-1, 1), 2),
            columns=["pnum_x", "pnum_y"],
            index=df2.sort_index().index,
        )
        # 断言结果与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_merge_on_periods(self):
        # 测试基于周期的合并
        # 创建左侧DataFrame，包含周期索引和值
        left = DataFrame(
            {"key": pd.period_range("20151010", periods=2, freq="D"), "value": [1, 2]}
        )
        # 创建右侧DataFrame，包含周期索引和值
        right = DataFrame(
            {
                "key": pd.period_range("20151011", periods=3, freq="D"),
                "value": [1, 2, 3],
            }
        )

        # 创建预期的DataFrame，包含周期索引和值
        expected = DataFrame(
            {
                "key": pd.period_range("20151010", periods=4, freq="D"),
                "value_x": [1, 2, np.nan, np.nan],
                "value_y": [np.nan, 1, 2, 3],
            }
        )
        # 执行基于key列的外连接合并
        result = merge(left, right, on="key", how="outer")
        # 断言结果与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_merge_period_values(self):
        # 测试周期值的合并
        # 创建左侧DataFrame，包含键和周期值
        left = DataFrame(
            {"key": [1, 2], "value": pd.period_range("20151010", periods=2, freq="D")}
        )
        # 创建右侧DataFrame，包含键和周期值
        right = DataFrame(
            {"key": [2, 3], "value": pd.period_range("20151011", periods=2, freq="D")}
        )

        # 创建左侧和右侧预期的周期范围
        exp_x = pd.period_range("20151010", periods=2, freq="D")
        exp_y = pd.period_range("20151011", periods=2, freq="D")
        # 创建预期的DataFrame，包含左侧和右侧的值
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "value_x": list(exp_x) + [pd.NaT],
                "value_y": [pd.NaT] + list(exp_y),
            }
        )
        # 执行基于key列的外连接合并
        result = merge(left, right, on="key", how="outer")
        # 断言结果与预期DataFrame相等
        tm.assert_frame_equal(result, expected)
        # 断言结果的'value_x'列和'value_y'列的数据类型为周期类型
        assert result["value_x"].dtype == "Period[D]"
        assert result["value_y"].dtype == "Period[D]"
    def test_indicator(self, dfs_for_indicator):
        # PR #10054. xref #7412 and closes #8790.
        # 从传入的 dfs_for_indicator 中解包出 df1 和 df2
        df1, df2 = dfs_for_indicator
        # 复制 df1，以便后续比较是否有副作用
        df1_copy = df1.copy()

        # 复制 df2，以便后续比较是否有副作用
        df2_copy = df2.copy()

        # 创建一个 DataFrame df_result，包含了几列数据，用于与合并后的数据框比较
        df_result = DataFrame(
            {
                "col1": [0, 1, 2, 3, 4, 5],
                "col_conflict_x": [1, 2, np.nan, np.nan, np.nan, np.nan],
                "col_left": ["a", "b", np.nan, np.nan, np.nan, np.nan],
                "col_conflict_y": [np.nan, 1, 2, 3, 4, 5],
                "col_right": [np.nan, 2, 2, 2, 2, 2],
            }
        )

        # 添加 '_merge' 列到 df_result 中，使用 Categorical 类别指示合并结果
        df_result["_merge"] = Categorical(
            [
                "left_only",
                "both",
                "right_only",
                "right_only",
                "right_only",
                "right_only",
            ],
            categories=["left_only", "right_only", "both"],
        )

        # 调整 df_result 的列顺序
        df_result = df_result[
            [
                "col1",
                "col_conflict_x",
                "col_left",
                "col_conflict_y",
                "col_right",
                "_merge",
            ]
        ]

        # 使用 merge 函数将 df1 和 df2 按照 'col1' 列进行外连接，指示合并情况
        test = merge(df1, df2, on="col1", how="outer", indicator=True)
        # 断言 test 和 df_result 相等
        tm.assert_frame_equal(test, df_result)

        # 使用 DataFrame 的 merge 方法进行相同的外连接，指示合并情况
        test = df1.merge(df2, on="col1", how="outer", indicator=True)
        # 断言 test 和 df_result 相等
        tm.assert_frame_equal(test, df_result)

        # 检查 df1 和 df2 是否有副作用
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)

        # 创建一个带有自定义列名 '_merge' 的 df_result_custom_name，用于与合并结果比较
        df_result_custom_name = df_result
        df_result_custom_name = df_result_custom_name.rename(
            columns={"_merge": "custom_name"}
        )

        # 使用 merge 函数进行外连接，并指定自定义的指示列名
        test_custom_name = merge(
            df1, df2, on="col1", how="outer", indicator="custom_name"
        )
        # 断言 test_custom_name 和 df_result_custom_name 相等
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)

        # 使用 DataFrame 的 merge 方法进行相同的外连接，并指定自定义的指示列名
        test_custom_name = df1.merge(
            df2, on="col1", how="outer", indicator="custom_name"
        )
        # 断言 test_custom_name 和 df_result_custom_name 相等
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)

    def test_merge_indicator_arg_validation(self, dfs_for_indicator):
        # 检查 indicator 参数只接受字符串或布尔值
        df1, df2 = dfs_for_indicator

        msg = "indicator option can only accept boolean or string arguments"
        # 使用 pytest 的断言检查当传入非字符串或布尔值时是否会抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on="col1", how="outer", indicator=5)
        with pytest.raises(ValueError, match=msg):
            df1.merge(df2, on="col1", how="outer", indicator=5)
    # 测试合并后结果的完整性
    def test_merge_indicator_result_integrity(self, dfs_for_indicator):
        # 从传入的数据框中获取两个数据框 df1 和 df2
        df1, df2 = dfs_for_indicator

        # 测试在左连接模式下，使用指示器列检查是否所有行都不是右侧独有的
        test2 = merge(df1, df2, on="col1", how="left", indicator=True)
        assert (test2._merge != "right_only").all()
        # 使用 DataFrame 的 merge 方法进行相同的测试
        test2 = df1.merge(df2, on="col1", how="left", indicator=True)
        assert (test2._merge != "right_only").all()

        # 测试在右连接模式下，使用指示器列检查是否所有行都不是左侧独有的
        test3 = merge(df1, df2, on="col1", how="right", indicator=True)
        assert (test3._merge != "left_only").all()
        # 使用 DataFrame 的 merge 方法进行相同的测试
        test3 = df1.merge(df2, on="col1", how="right", indicator=True)
        assert (test3._merge != "left_only").all()

        # 测试在内连接模式下，使用指示器列检查是否所有行都是左右均有的
        test4 = merge(df1, df2, on="col1", how="inner", indicator=True)
        assert (test4._merge == "both").all()
        # 使用 DataFrame 的 merge 方法进行相同的测试
        test4 = df1.merge(df2, on="col1", how="inner", indicator=True)
        assert (test4._merge == "both").all()

    # 测试使用指示器时，是否会检测到无效的情况
    def test_merge_indicator_invalid(self, dfs_for_indicator):
        # 从传入的数据框中获取 df1
        df1, _ = dfs_for_indicator

        # 遍历不合法的指示器列名称，测试是否会抛出 ValueError 异常
        for i in ["_right_indicator", "_left_indicator", "_merge"]:
            # 创建包含不合法列名的数据框 df_badcolumn
            df_badcolumn = DataFrame({"col1": [1, 2], i: [2, 2]})

            # 构造预期的异常消息
            msg = (
                "Cannot use `indicator=True` option when data contains a "
                f"column named {i}|"
                "Cannot use name of an existing column for indicator column"
            )
            # 使用 pytest 的断言检查是否会抛出预期异常消息的 ValueError 异常
            with pytest.raises(ValueError, match=msg):
                merge(df1, df_badcolumn, on="col1", how="outer", indicator=True)
            with pytest.raises(ValueError, match=msg):
                df1.merge(df_badcolumn, on="col1", how="outer", indicator=True)

        # 检查自定义指示器列名与已存在列名的冲突
        df_badcolumn = DataFrame({"col1": [1, 2], "custom_column_name": [2, 2]})

        # 构造预期的异常消息
        msg = "Cannot use name of an existing column for indicator column"
        # 使用 pytest 的断言检查是否会抛出预期异常消息的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            merge(
                df1,
                df_badcolumn,
                on="col1",
                how="outer",
                indicator="custom_column_name",
            )
        with pytest.raises(ValueError, match=msg):
            df1.merge(
                df_badcolumn, on="col1", how="outer", indicator="custom_column_name"
            )
    def test_merge_indicator_multiple_columns(self):
        # 合并多列
        # 创建 DataFrame df3，包含列 col1 和 col2
        df3 = DataFrame({"col1": [0, 1], "col2": ["a", "b"]})

        # 创建 DataFrame df4，包含列 col1 和 col2
        df4 = DataFrame({"col1": [1, 1, 3], "col2": ["b", "x", "y"]})

        # 手动编码的期望结果，包含列 col1、col2 和 _merge
        hand_coded_result = DataFrame(
            {"col1": [0, 1, 1, 3], "col2": ["a", "b", "x", "y"]}
        )
        hand_coded_result["_merge"] = Categorical(
            ["left_only", "both", "right_only", "right_only"],
            categories=["left_only", "right_only", "both"],
        )

        # 测试 merge 函数，合并 df3 和 df4，根据 col1 和 col2 外连接，并添加 _merge 指示列
        test5 = merge(df3, df4, on=["col1", "col2"], how="outer", indicator=True)
        # 使用 tm.assert_frame_equal 检查 test5 和手动编码的结果是否相等
        tm.assert_frame_equal(test5, hand_coded_result)

        # 测试 DataFrame 的 merge 方法，合并 df3 和 df4，根据 col1 和 col2 外连接，并添加 _merge 指示列
        test5 = df3.merge(df4, on=["col1", "col2"], how="outer", indicator=True)
        # 使用 tm.assert_frame_equal 检查 test5 和手动编码的结果是否相等
        tm.assert_frame_equal(test5, hand_coded_result)

    def test_merge_two_empty_df_no_division_error(self):
        # GH17776, PR #17846
        # 创建空 DataFrame a，列名为 'a', 'b', 'c'
        a = DataFrame({"a": [], "b": [], "c": []})
        # 在进行 merge 操作时，设置 numpy 的 divide 错误状态为 raise
        with np.errstate(divide="raise"):
            # 对两个空 DataFrame a 进行 merge 操作，根据列 'a' 和 'b' 进行合并
            merge(a, a, on=("a", "b"))

    @pytest.mark.parametrize("how", ["right", "outer"])
    @pytest.mark.parametrize(
        "index,expected_index",
        [
            # 分类索引的测试用例
            (
                CategoricalIndex([1, 2, 4]),
                CategoricalIndex([1, 2, 4, None, None, None]),
            ),
            # 日期时间索引的测试用例
            (
                DatetimeIndex(
                    ["2001-01-01", "2002-02-02", "2003-03-03"], dtype="M8[ns]"
                ),
                DatetimeIndex(
                    ["2001-01-01", "2002-02-02", "2003-03-03", pd.NaT, pd.NaT, pd.NaT],
                    dtype="M8[ns]",
                ),
            ),
            # 所有实数类型的索引测试用例
            *[
                (
                    Index([1, 2, 3], dtype=dtyp),
                    Index([1, 2, 3, None, None, None], dtype=np.float64),
                )
                for dtyp in tm.ALL_REAL_NUMPY_DTYPES
            ],
            # 区间索引的测试用例
            (
                IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4)]),
                IntervalIndex.from_tuples(
                    [(1, 2), (2, 3), (3, 4), np.nan, np.nan, np.nan]
                ),
            ),
            # 周期索引的测试用例
            (
                PeriodIndex(["2001-01-01", "2001-01-02", "2001-01-03"], freq="D"),
                PeriodIndex(
                    ["2001-01-01", "2001-01-02", "2001-01-03", pd.NaT, pd.NaT, pd.NaT],
                    freq="D",
                ),
            ),
            # 时间增量索引的测试用例
            (
                TimedeltaIndex(["1D", "2D", "3D"]),
                TimedeltaIndex(["1D", "2D", "3D", pd.NaT, pd.NaT, pd.NaT]),
            ),
        ],
    )
    def test_merge_on_index_with_more_values(self, how, index, expected_index):
        # GH 24212
        # pd.merge gets [0, 1, 2, -1, -1, -1] as left_indexer, ensure that
        # -1 is interpreted as a missing value instead of the last element
        
        # 创建第一个 DataFrame，包含列 'a' 和 'key'，使用给定的索引
        df1 = DataFrame({"a": [0, 1, 2], "key": [0, 1, 2]}, index=index)
        
        # 创建第二个 DataFrame，包含列 'b'，没有指定索引
        df2 = DataFrame({"b": [0, 1, 2, 3, 4, 5]})
        
        # 使用 pd.merge 将两个 DataFrame 合并，根据左侧的 'key' 列和右侧的索引进行合并
        result = df1.merge(df2, left_on="key", right_index=True, how=how)
        
        # 创建预期的 DataFrame，包含所有可能的合并结果，包括 NaN 值
        expected = DataFrame(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [np.nan, 3, 3],
                [np.nan, 4, 4],
                [np.nan, 5, 5],
            ],
            columns=["a", "key", "b"],
        )
        
        # 将预期结果根据预期索引设置为新索引
        expected.set_index(expected_index, inplace=True)
        
        # 使用测试工具比较实际结果和预期结果
        tm.assert_frame_equal(result, expected)

    def test_merge_right_index_right(self):
        # Note: the expected output here is probably incorrect.
        # See https://github.com/pandas-dev/pandas/issues/17257 for more.
        # We include this as a regression test for GH-24897.
        
        # 创建左侧 DataFrame，包含列 'a' 和 'key'
        left = DataFrame({"a": [1, 2, 3], "key": [0, 1, 1]})
        
        # 创建右侧 DataFrame，包含列 'b'
        right = DataFrame({"b": [1, 2, 3]})
        
        # 创建预期的合并结果 DataFrame，包含所有可能的合并结果，包括 NaN 值
        expected = DataFrame(
            {"a": [1, 2, 3, None], "key": [0, 1, 1, 2], "b": [1, 2, 2, 3]},
            columns=["a", "key", "b"],
            index=[0, 1, 2, np.nan],
        )
        
        # 使用 pd.merge 将左侧 DataFrame 和右侧 DataFrame 按照左侧的 'key' 列和右侧的索引进行合并
        result = left.merge(right, left_on="key", right_index=True, how="right")
        
        # 使用测试工具比较实际结果和预期结果
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("how", ["left", "right"])
    def test_merge_preserves_row_order(self, how):
        # GH 27453
        
        # 创建左侧 DataFrame，包含列 'animal' 和 'max_speed'
        left_df = DataFrame({"animal": ["dog", "pig"], "max_speed": [40, 11]})
        
        # 创建右侧 DataFrame，包含列 'animal' 和 'max_speed'
        right_df = DataFrame({"animal": ["quetzal", "pig"], "max_speed": [80, 11]})
        
        # 使用 pd.merge 按照列 'animal' 和 'max_speed' 进行合并，保持行顺序
        result = left_df.merge(right_df, on=["animal", "max_speed"], how=how)
        
        # 根据不同的合并方式选择预期的 DataFrame
        if how == "right":
            expected = DataFrame({"animal": ["quetzal", "pig"], "max_speed": [80, 11]})
        else:
            expected = DataFrame({"animal": ["dog", "
    def test_merge_readonly(self):
        # 测试用例函数：test_merge_readonly，用于验证只读模式下的合并操作

        # 创建第一个数据框，包含20个元素的二维数组，重塑为4行5列，列名为["a", "b", "c", "d", "e"]
        data1 = DataFrame(
            np.arange(20).reshape((4, 5)) + 1, columns=["a", "b", "c", "d", "e"]
        )
        
        # 创建第二个数据框，包含20个元素的二维数组，重塑为5行4列，列名为["a", "b", "x", "y"]
        data2 = DataFrame(
            np.arange(20).reshape((5, 4)) + 1, columns=["a", "b", "x", "y"]
        )

        # 将第一个数据框中每个块的值设为只读
        for block in data1._mgr.blocks:
            block.values.flags.writeable = False
        
        # 合并第一个数据框和第二个数据框
        data1.merge(data2)  # 执行合并操作，预期不会报错
def _check_merge(x, y):
    # 循环遍历不同的合并方式和排序方式
    for how in ["inner", "left", "outer"]:
        for sort in [True, False]:
            # 执行数据框合并操作，根据当前循环的合并方式和排序方式
            result = x.join(y, how=how, sort=sort)

            # 生成预期的合并结果，同时重置索引，以便与 result 进行比较
            expected = merge(x.reset_index(), y.reset_index(), how=how, sort=sort)
            expected = expected.set_index("index")

            # TODO check_names on merge?
            # 使用测试框架检查 result 是否与 expected 相等，忽略列名检查
            tm.assert_frame_equal(result, expected, check_names=False)


class TestMergeDtypes:
    @pytest.mark.parametrize("dtype", [object, "category"])
    def test_different(self, dtype):
        # 创建左侧数据框，包含不同类型的列数据
        left = DataFrame(
            {
                "A": ["foo", "bar"],
                "B": Series(["foo", "bar"]).astype("category"),
                "C": [1, 2],
                "D": [1.0, 2.0],
                "E": Series([1, 2], dtype="uint64"),
                "F": Series([1, 2], dtype="int32"),
            }
        )
        # 创建右侧数据框，只包含一列 A，数据类型为指定的 dtype
        right_vals = Series(["foo", "bar"], dtype=dtype)
        right = DataFrame({"A": right_vals})

        # GH 9780
        # 允许在对象和类别列上进行合并，并将类别列转换为对象类型
        result = merge(left, right, on="A")
        # 断言合并后的列 A 的数据类型为对象类型或字符串类型
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize("d2", [np.int64, np.float64, np.float32, np.float16])
    def test_join_multi_dtypes(self, any_int_numpy_dtype, d2):
        dtype1 = np.dtype(any_int_numpy_dtype)
        dtype2 = np.dtype(d2)

        # 创建左侧数据框，包含多种数据类型的列
        left = DataFrame(
            {
                "k1": np.array([0, 1, 2] * 8, dtype=dtype1),
                "k2": ["foo", "bar"] * 12,
                "v": np.array(np.arange(24), dtype=np.int64),
            }
        )

        # 创建右侧数据框，包含一列 v2，索引为 MultiIndex
        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": np.array([5, 7], dtype=dtype2)}, index=index)

        # 执行基于多列合并的操作，左侧的列 k1 和 k2 与右侧的索引进行合并
        result = left.join(right, on=["k1", "k2"])

        # 复制左侧数据框作为预期结果
        expected = left.copy()

        # 根据右侧数据类型的 kind 属性，确定预期结果中 v2 列的数据类型
        if dtype2.kind == "i":
            dtype2 = np.dtype("float64")
        expected["v2"] = np.array(np.nan, dtype=dtype2)
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        # 使用测试框架检查 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

        # 再次执行基于多列合并的操作，并进行排序
        result = left.join(right, on=["k1", "k2"], sort=True)
        expected.sort_values(["k1", "k2"], kind="mergesort", inplace=True)

        # 使用测试框架检查排序后的 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "int_vals, float_vals, exp_vals",
        [
            ([1, 2, 3], [1.0, 2.0, 3.0], {"X": [1, 2, 3], "Y": [1.0, 2.0, 3.0]}),
            ([1, 2, 3], [1.0, 3.0], {"X": [1, 3], "Y": [1.0, 3.0]}),
            ([1, 2], [1.0, 2.0, 3.0], {"X": [1, 2], "Y": [1.0, 2.0]}),
        ],
    )
    def test_merge_on_ints_floats(self, int_vals, float_vals, exp_vals):
        # GH 16572
        # 检查当在整数和浮点数列上进行合并时，确保浮点数列不会被转换为对象类型
        A = DataFrame({"X": int_vals})
        B = DataFrame({"Y": float_vals})
        expected = DataFrame(exp_vals)

        # 根据指定的左右列名进行数据框合并，生成结果数据框
        result = A.merge(B, left_on="X", right_on="Y")
        # 断言结果数据框与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 根据指定的左右列名进行数据框合并，生成结果数据框，并且只保留指定的列顺序
        result = B.merge(A, left_on="Y", right_on="X")
        # 断言结果数据框与预期数据框的部分列相等
        tm.assert_frame_equal(result, expected[["Y", "X"]])

    def test_merge_key_dtype_cast(self):
        # GH 17044
        # 检查在合并键的数据类型转换时的行为
        df1 = DataFrame({"key": [1.0, 2.0], "v1": [10, 20]}, columns=["key", "v1"])
        df2 = DataFrame({"key": [2], "v2": [200]}, columns=["key", "v2"])
        # 使用左连接方式，根据指定的键合并两个数据框
        result = df1.merge(df2, on="key", how="left")
        # 创建预期的数据框
        expected = DataFrame(
            {"key": [1.0, 2.0], "v1": [10, 20], "v2": [np.nan, 200.0]},
            columns=["key", "v1", "v2"],
        )
        # 断言结果数据框与预期数据框相等
        tm.assert_frame_equal(result, expected)

    def test_merge_on_ints_floats_warning(self):
        # GH 16572
        # 在整数和浮点数列上合并时，如果浮点数的值与其整数表示不完全相等，则会产生警告
        A = DataFrame({"X": [1, 2, 3]})
        B = DataFrame({"Y": [1.1, 2.5, 3.0]})
        expected = DataFrame({"X": [3], "Y": [3.0]})

        msg = "the float values are not equal to their int representation"
        # 使用上下文管理器检查是否产生特定类型和匹配消息的警告
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = A.merge(B, left_on="X", right_on="Y")
            # 断言结果数据框与预期数据框相等
            tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(UserWarning, match=msg):
            result = B.merge(A, left_on="Y", right_on="X")
            # 断言结果数据框与预期数据框的部分列相等
            tm.assert_frame_equal(result, expected[["Y", "X"]])

        # 测试当浮点数列包含 NaN 时不会产生警告
        B = DataFrame({"Y": [np.nan, np.nan, 3.0]})

        with tm.assert_produces_warning(None):
            result = B.merge(A, left_on="Y", right_on="X")
            # 断言结果数据框与预期数据框的部分列相等
            tm.assert_frame_equal(result, expected[["Y", "X"]])

    def test_merge_incompat_infer_boolean_object(self):
        # GH21119: bool + object bool merge OK
        # 检查布尔类型和对象类型布尔类型的合并
        df1 = DataFrame({"key": Series([True, False], dtype=object)})
        df2 = DataFrame({"key": [True, False]})

        expected = DataFrame({"key": [True, False]}, dtype=object)
        # 根据指定的键合并两个数据框
        result = merge(df1, df2, on="key")
        # 断言结果数据框与预期数据框相等
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, on="key")
        # 断言结果数据框与预期数据框相等
        tm.assert_frame_equal(result, expected)
    def test_merge_incompat_infer_boolean_object_with_missing(self):
        # GH21119: bool + object bool merge OK
        # with missing value
        # 创建包含布尔值和对象布尔值（含缺失值）的数据框 df1
        df1 = DataFrame({"key": Series([True, False, np.nan], dtype=object)})
        # 创建包含布尔值的数据框 df2
        df2 = DataFrame({"key": [True, False]})
        # 期望的合并结果，数据类型为对象
        expected = DataFrame({"key": [True, False]}, dtype=object)
        # 对 df1 和 df2 按照 'key' 列进行合并
        result = merge(df1, df2, on="key")
        # 断言合并结果与期望结果相等
        tm.assert_frame_equal(result, expected)
        # 再次合并 df2 和 df1 按照 'key' 列
        result = merge(df2, df1, on="key")
        # 断言合并结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "df1_vals, df2_vals",
        [
            # merge on category coerces to object
            # 合并分类数据到对象类型
            ([0, 1, 2], Series(["a", "b", "a"]).astype("category")),
            ([0.0, 1.0, 2.0], Series(["a", "b", "a"]).astype("category")),
            # no not infer
            # 不推断的情况
            ([0, 1], Series([False, True], dtype=object)),
            ([0, 1], Series([False, True], dtype=bool)),
        ],
    )
    def test_merge_incompat_dtypes_are_ok(self, df1_vals, df2_vals):
        # these are explicitly allowed incompat merges, that pass thru
        # the result type is dependent on if the values on the rhs are
        # inferred, otherwise these will be coerced to object
        # 这些是显式允许的不兼容合并，结果类型取决于右侧值是否推断，否则将被强制转换为对象

        # 创建包含 df1_vals 的数据框 df1
        df1 = DataFrame({"A": df1_vals})
        # 创建包含 df2_vals 的数据框 df2
        df2 = DataFrame({"A": df2_vals})

        # 对 df1 和 df2 按照 'A' 列进行合并
        result = merge(df1, df2, on=["A"])
        # 断言合并结果的 'A' 列数据类型为对象
        assert is_object_dtype(result.A.dtype)
        # 再次对 df2 和 df1 按照 'A' 列进行合并
        result = merge(df2, df1, on=["A"])
        # 断言合并结果的 'A' 列数据类型为对象或字符串
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize(
        "df1_vals, df2_vals",
        [
            # do not infer to numeric
            # 不推断为数值类型
            (Series([1, 2], dtype="uint64"), ["a", "b", "c"]),
            (Series([1, 2], dtype="int32"), ["a", "b", "c"]),
            ([0, 1, 2], ["0", "1", "2"]),
            ([0.0, 1.0, 2.0], ["0", "1", "2"]),
            (
                pd.date_range("1/1/2011", periods=2, freq="D"),
                ["2011-01-01", "2011-01-02"],
            ),
            (pd.date_range("1/1/2011", periods=2, freq="D"), [0, 1]),
            (pd.date_range("1/1/2011", periods=2, freq="D"), [0.0, 1.0]),
            (
                pd.date_range("20130101", periods=3),
                pd.date_range("20130101", periods=3, tz="US/Eastern"),
            ),
        ],
    )
    # 定义一个测试方法，用于测试在不兼容的数据类型上进行合并时是否正确引发 ValueError
    def test_merge_incompat_dtypes_error(self, df1_vals, df2_vals):
        # GH 9780, GH 15800
        # 提到 GitHub 问题编号，表明此测试的相关问题
        # 当用户尝试在不兼容的数据类型上进行合并时引发 ValueError（例如，对象和整数/浮点数）

        # 创建包含 'A' 列的两个 DataFrame 对象 df1 和 df2
        df1 = DataFrame({"A": df1_vals})
        df2 = DataFrame({"A": df2_vals})

        # 生成错误消息，说明正在尝试在不兼容的数据类型上合并 'A' 列
        msg = (
            f"You are trying to merge on {df1['A'].dtype} and {df2['A'].dtype} "
            "columns for key 'A'. If you wish to proceed you should use pd.concat"
        )
        msg = re.escape(msg)  # 对消息进行正则表达式转义处理，以确保匹配精确度
        # 使用 pytest 检查是否引发了 ValueError，并验证错误消息是否匹配预期
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on=["A"])

        # 检查在交换数据框顺序时是否仍然引发错误
        msg = (
            f"You are trying to merge on {df2['A'].dtype} and {df1['A'].dtype} "
            "columns for key 'A'. If you wish to proceed you should use pd.concat"
        )
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df2, df1, on=["A"])

        # 检查在多列上合并时是否仍然引发错误
        # 错误消息应提到第一个不兼容列
        if len(df1_vals) == len(df2_vals):
            # 在 df1 和 df2 中，列 A 具有兼容（相同）的数据类型
            # 列 B 和 C 在 df1 和 df2 中具有不兼容的数据类型
            df3 = DataFrame({"A": df2_vals, "B": df1_vals, "C": df1_vals})
            df4 = DataFrame({"A": df2_vals, "B": df2_vals, "C": df2_vals})

            # 检查当合并所有列 A、B 和 C 时是否正确引发错误
            # 错误消息应提到键 'B'
            msg = (
                f"You are trying to merge on {df3['B'].dtype} and {df4['B'].dtype} "
                "columns for key 'B'. If you wish to proceed you should use pd.concat"
            )
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4)

            # 检查当合并列 A 和 C 时是否正确引发错误
            # 错误消息应提到键 'C'
            msg = (
                f"You are trying to merge on {df3['C'].dtype} and {df4['C'].dtype} "
                "columns for key 'C'. If you wish to proceed you should use pd.concat"
            )
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4, on=["A", "C"])
    # 定义一个测试方法，用于测试合并具有任意数值类型的数据框，根据指定的合并方式和预期数据
    def test_merge_EA_dtype(self, any_numeric_ea_dtype, how, expected_data):
        # GH#40073
        # 创建第一个数据框，包含一个数值列"id"，使用指定的数据类型any_numeric_ea_dtype
        d1 = DataFrame([(1,)], columns=["id"], dtype=any_numeric_ea_dtype)
        # 创建第二个数据框，也包含一个数值列"id"，数据类型同样为any_numeric_ea_dtype
        d2 = DataFrame([(2,)], columns=["id"], dtype=any_numeric_ea_dtype)
        # 执行数据框合并操作，根据指定的合并方式how
        result = merge(d1, d2, how=how)
        # 创建预期的数据框索引，长度为expected_data的长度
        exp_index = RangeIndex(len(expected_data))
        # 创建预期的数据框，数据来源于expected_data，列名为"id"，数据类型为any_numeric_ea_dtype
        expected = DataFrame(
            expected_data, index=exp_index, columns=["id"], dtype=any_numeric_ea_dtype
        )
        # 使用测试框架中的方法，断言两个数据框是否相等
        tm.assert_frame_equal(result, expected)

    # 使用参数化标记定义一个测试方法，用于测试合并具有任意字符串类型的数据框，根据指定的合并方式和预期数据
    @pytest.mark.parametrize(
        "expected_data, how",
        [
            (["a", "b"], "outer"),
            ([], "inner"),
            (["b"], "right"),
            (["a"], "left"),
        ],
    )
    def test_merge_string_dtype(self, how, expected_data, any_string_dtype):
        # GH#40073
        # 创建第一个数据框，包含一个字符串列"id"，使用指定的数据类型any_string_dtype
        d1 = DataFrame([("a",)], columns=["id"], dtype=any_string_dtype)
        # 创建第二个数据框，也包含一个字符串列"id"，数据类型同样为any_string_dtype
        d2 = DataFrame([("b",)], columns=["id"], dtype=any_string_dtype)
        # 执行数据框合并操作，根据指定的合并方式how
        result = merge(d1, d2, how=how)
        # 创建预期的数据框索引，长度为expected_data的长度
        exp_idx = RangeIndex(len(expected_data))
        # 创建预期的数据框，数据来源于expected_data，列名为"id"，数据类型为any_string_dtype
        expected = DataFrame(
            expected_data, index=exp_idx, columns=["id"], dtype=any_string_dtype
        )
        # 使用测试框架中的方法，断言两个数据框是否相等
        tm.assert_frame_equal(result, expected)

    # 使用参数化标记定义一个测试方法，用于测试合并具有布尔类型的数据框，根据指定的合并方式和预期数据
    @pytest.mark.parametrize(
        "how, expected_data",
        [
            ("inner", [[True, 1, 4], [False, 5, 3]]),
            ("outer", [[False, 5, 3], [True, 1, 4]]),
            ("left", [[True, 1, 4], [False, 5, 3]]),
            ("right", [[False, 5, 3], [True, 1, 4]]),
        ],
    )
    def test_merge_bool_dtype(self, how, expected_data):
        # GH#40073
        # 创建第一个数据框，包含列"A"和"B"，数据类型为布尔型
        df1 = DataFrame({"A": [True, False], "B": [1, 5]})
        # 创建第二个数据框，包含列"A"和"C"，数据类型为布尔型
        df2 = DataFrame({"A": [False, True], "C": [3, 4]})
        # 执行数据框合并操作，根据指定的合并方式how
        result = merge(df1, df2, how=how)
        # 创建预期的数据框，数据来源于expected_data，列名为"A", "B", "C"
        expected = DataFrame(expected_data, columns=["A", "B", "C"])
        # 使用测试框架中的方法，断言两个数据框是否相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试带字符串数据类型的数据合并操作
    def test_merge_ea_with_string(self, join_type, string_dtype):
        # 避免在多级索引上使用 `assign` 方法（GH 43734）
        
        # 创建 DataFrame df1，使用 pd.StringDtype 类型作为数据类型
        df1 = DataFrame(
            data={
                ("lvl0", "lvl1-a"): ["1", "2", "3", "4", None],
                ("lvl0", "lvl1-b"): ["4", "5", "6", "7", "8"],
            },
            dtype=pd.StringDtype(),
        )
        # 备份 df1 副本
        df1_copy = df1.copy()
        
        # 创建 DataFrame df2，使用参数传入的 string_dtype 类型作为数据类型
        df2 = DataFrame(
            data={
                ("lvl0", "lvl1-a"): ["1", "2", "3", pd.NA, "5"],
                ("lvl0", "lvl1-c"): ["7", "8", "9", pd.NA, "11"],
            },
            dtype=string_dtype,
        )
        # 备份 df2 副本
        df2_copy = df2.copy()
        
        # 将 df1 和 df2 按照指定的多级索引列 "lvl0/lvl1-a" 进行合并，合并方式由 join_type 决定
        merged = merge(left=df1, right=df2, on=[("lvl0", "lvl1-a")], how=join_type)

        # 检查 df1 和 df2 是否未发生改变
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)

        # 检查合并后数据框的预期数据类型
        expected = Series(
            [np.dtype("O"), pd.StringDtype(), np.dtype("O")],
            index=MultiIndex.from_tuples(
                [("lvl0", "lvl1-a"), ("lvl0", "lvl1-b"), ("lvl0", "lvl1-c")]
            ),
        )
        tm.assert_series_equal(merged.dtypes, expected)

    @pytest.mark.parametrize(
        "left_empty, how, exp",
        [
            (False, "left", "left"),
            (False, "right", "empty"),
            (False, "inner", "empty"),
            (False, "outer", "left"),
            (False, "cross", "empty_cross"),
            (True, "left", "empty"),
            (True, "right", "right"),
            (True, "inner", "empty"),
            (True, "outer", "right"),
            (True, "cross", "empty_cross"),
        ],
    )
    # 定义一个参数化测试方法，用于测试空数据框的合并操作
    def test_merge_empty(self, left_empty, how, exp):
        # 创建左侧数据框 left，包含列 "A" 和 "B"
        left = DataFrame({"A": [2, 1], "B": [3, 4]})
        # 创建右侧数据框 right，包含列 "A" 和 "C"，数据类型为 "int64"
        right = DataFrame({"A": [1], "C": [5]}, dtype="int64")

        # 根据 left_empty 参数决定是否将 left 数据框截取为空
        if left_empty:
            left = left.head(0)
        else:
            right = right.head(0)

        # 执行左右数据框的合并，合并方式由 how 参数决定
        result = left.merge(right, how=how)

        # 根据 exp 参数选择预期的合并结果数据框
        if exp == "left":
            expected = DataFrame({"A": [2, 1], "B": [3, 4], "C": [np.nan, np.nan]})
        elif exp == "right":
            expected = DataFrame({"A": [1], "B": [np.nan], "C": [5]})
        elif exp == "empty":
            expected = DataFrame(columns=["A", "B", "C"], dtype="int64")
        elif exp == "empty_cross":
            expected = DataFrame(columns=["A_x", "B", "A_y", "C"], dtype="int64")

        # 如果合并方式为 "outer"，按照列 "A" 的值排序结果数据框 expected
        if how == "outer":
            expected = expected.sort_values("A", ignore_index=True)

        # 断言合并结果与预期结果相等
        tm.assert_frame_equal(result, expected)
@pytest.fixture
def left():
    # 返回一个包含随机选择的 "foo" 和 "bar" 字符串的 Series 对象，转换为分类类型
    return DataFrame(
        {
            "X": Series(
                np.random.default_rng(2).choice(["foo", "bar"], size=(10,))
            ).astype(CategoricalDtype(["foo", "bar"])),
            "Y": np.random.default_rng(2).choice(["one", "two", "three"], size=(10,)),
        }
    )


@pytest.fixture
def right():
    # 返回一个包含 "foo" 和 "bar" 字符串的 Series 对象，转换为分类类型，以及一个整数列表的 DataFrame 对象
    return DataFrame(
        {
            "X": Series(["foo", "bar"]).astype(CategoricalDtype(["foo", "bar"])),
            "Z": [1, 2],
        }
    )


class TestMergeCategorical:
    def test_identical(self, left, using_infer_string):
        # 在相同的列 'X' 上合并，应保持数据类型
        merged = merge(left, left, on="X")
        # 获取合并后的数据类型并排序
        result = merged.dtypes.sort_index()
        # 如果不使用 'infer_string'，则数据类型为对象 'O'，否则为字符串 'string'
        dtype = np.dtype("O") if not using_infer_string else "string"
        # 期望的数据类型 Series
        expected = Series(
            [CategoricalDtype(categories=["foo", "bar"]), dtype, dtype],
            index=["X", "Y_x", "Y_y"],
        )
        # 断言合并后的数据类型与期望的数据类型相等
        tm.assert_series_equal(result, expected)

    def test_basic(self, left, right, using_infer_string):
        # 'X' 列具有匹配的分类数据类型，因此应保持合并后的列
        merged = merge(left, right, on="X")
        # 获取合并后的数据类型并排序
        result = merged.dtypes.sort_index()
        # 如果不使用 'infer_string'，则数据类型为对象 'O'，否则为字符串 'string'
        dtype = np.dtype("O") if not using_infer_string else "string"
        # 期望的数据类型 Series
        expected = Series(
            [
                CategoricalDtype(categories=["foo", "bar"]),
                dtype,
                np.dtype("int64"),
            ],
            index=["X", "Y", "Z"],
        )
        # 断言合并后的数据类型与期望的数据类型相等
        tm.assert_series_equal(result, expected)
    def test_merge_categorical(self):
        # GH 9426
        # 定义右侧数据框
        right = DataFrame(
            {
                "c": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"},
                "d": {0: "null", 1: "null", 2: "null", 3: "null", 4: "null"},
            }
        )
        # 定义左侧数据框
        left = DataFrame(
            {
                "a": {0: "f", 1: "f", 2: "f", 3: "f", 4: "f"},
                "b": {0: "g", 1: "g", 2: "g", 3: "g", 4: "g"},
            }
        )
        # 合并左右数据框，按左侧'b'列和右侧'c'列合并，使用左连接方式
        df = merge(left, right, how="left", left_on="b", right_on="c")

        # object-object 类型的预期结果
        expected = df.copy()

        # object-cat 类型的合并，注意传播类别
        cright = right.copy()
        cright["d"] = cright["d"].astype("category")
        result = merge(left, cright, how="left", left_on="b", right_on="c")
        expected["d"] = expected["d"].astype(CategoricalDtype(["null"]))
        tm.assert_frame_equal(result, expected)

        # cat-object 类型的合并
        cleft = left.copy()
        cleft["b"] = cleft["b"].astype("category")
        result = merge(cleft, cright, how="left", left_on="b", right_on="c")
        tm.assert_frame_equal(result, expected)

        # cat-cat 类型的合并
        cright = right.copy()
        cright["d"] = cright["d"].astype("category")
        cleft = left.copy()
        cleft["b"] = cleft["b"].astype("category")
        result = merge(cleft, cright, how="left", left_on="b", right_on="c")
        tm.assert_frame_equal(result, expected)

    def tests_merge_categorical_unordered_equal(self):
        # GH-19551
        # 创建第一个数据框，包含分类列 'Foo' 和对应左侧列 'Left'
        df1 = DataFrame(
            {
                "Foo": Categorical(["A", "B", "C"], categories=["A", "B", "C"]),
                "Left": ["A0", "B0", "C0"],
            }
        )

        # 创建第二个数据框，包含分类列 'Foo' 和对应右侧列 'Right'
        df2 = DataFrame(
            {
                "Foo": Categorical(["C", "B", "A"], categories=["C", "B", "A"]),
                "Right": ["C1", "B1", "A1"],
            }
        )
        # 根据 'Foo' 列合并两个数据框
        result = merge(df1, df2, on=["Foo"])
        # 预期结果数据框，包含 'Foo' 列、'Left' 列和 'Right' 列
        expected = DataFrame(
            {
                "Foo": Categorical(["A", "B", "C"]),
                "Left": ["A0", "B0", "C0"],
                "Right": ["A1", "B1", "C1"],
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    # 定义一个测试函数，测试在具有无序分类索引的情况下多级索引的合并
    def test_multiindex_merge_with_unordered_categoricalindex(self, ordered):
        # GH 36973
        # 创建一个分类数据类型，包括两个类别："P2" 和 "P1"
        pcat = CategoricalDtype(categories=["P2", "P1"], ordered=ordered)
        # 创建第一个数据帧 df1，设置多级索引为 ["id", "p"]，其中 "p" 列使用之前定义的分类类型 pcat
        df1 = DataFrame(
            {
                "id": ["C", "C", "D"],
                "p": Categorical(["P2", "P1", "P2"], dtype=pcat),
                "a": [0, 1, 2],
            }
        ).set_index(["id", "p"])
        # 创建第二个数据帧 df2，设置多级索引为 ["id", "p"]，其中 "p" 列也使用之前定义的分类类型 pcat
        df2 = DataFrame(
            {
                "id": ["A", "C", "C"],
                "p": Categorical(["P2", "P2", "P1"], dtype=pcat),
                "d1": [10, 11, 12],
            }
        ).set_index(["id", "p"])
        # 使用 merge 函数将 df1 和 df2 按照左右索引合并，方式为左连接（left join）
        result = merge(df1, df2, how="left", left_index=True, right_index=True)
        # 创建预期的合并结果数据帧 expected，设置多级索引为 ["id", "p"]，同时包含 "a" 和 "d1" 列
        expected = DataFrame(
            {
                "id": ["C", "C", "D"],
                "p": Categorical(["P2", "P1", "P2"], dtype=pcat),
                "a": [0, 1, 2],
                "d1": [11.0, 12.0, np.nan],
            }
        ).set_index(["id", "p"])
        # 使用 assert_frame_equal 函数断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)

    # 定义另一个测试函数，测试在使用 merge 函数时保留非合并列的数据类型和分类信息
    def test_other_columns(self, left, right, using_infer_string):
        # 非合并的列尽可能保留其原始数据类型
        # 将 right 数据帧的 "Z" 列转换为分类类型
        right = right.assign(Z=right.Z.astype("category"))

        # 使用 merge 函数将 left 和 right 数据帧按照 "X" 列合并
        merged = merge(left, right, on="X")
        # 获取合并后数据帧的列数据类型，并按索引排序
        result = merged.dtypes.sort_index()
        # 根据使用推断字符串的标志位确定预期的数据类型
        dtype = np.dtype("O") if not using_infer_string else "string"
        # 创建预期的列数据类型 Series，包含 ["X", "Y", "Z"] 列的数据类型信息
        expected = Series(
            [
                CategoricalDtype(categories=["foo", "bar"]),
                dtype,
                CategoricalDtype(categories=[1, 2]),
            ],
            index=["X", "Y", "Z"],
        )
        # 使用 assert_series_equal 函数断言 result 和 expected 列数据类型相等
        tm.assert_series_equal(result, expected)

        # 断言 left 数据帧的 "X" 列的分类信息与 merged 后的 "X" 列相匹配
        assert left.X.values._categories_match_up_to_permutation(merged.X.values)
        # 断言 right 数据帧的 "Z" 列的分类信息与 merged 后的 "Z" 列相匹配
        assert right.Z.values._categories_match_up_to_permutation(merged.Z.values)

    # 使用 pytest.mark.parametrize 标记参数化测试函数，测试在合并列具有不同数据类型时的情况
    @pytest.mark.parametrize(
        "change",
        [
            lambda x: x,
            lambda x: x.astype(CategoricalDtype(["foo", "bar", "bah"])),
            lambda x: x.astype(CategoricalDtype(ordered=True)),
        ],
    )
    def test_dtype_on_merged_different(
        self, change, join_type, left, right, using_infer_string
    ):
        # 我们的合并列 "X" 现在具有两种不同的数据类型，因此结果必须是对象类型（"O"）
        X = change(right.X.astype("object"))
        # 将 right 数据帧的 "X" 列更新为 X
        right = right.assign(X=X)
        # 断言 left 数据帧的 "X" 列是分类数据类型
        assert isinstance(left.X.values.dtype, CategoricalDtype)
        # 不再断言 left.X 和 right.X 的分类信息是否匹配

        # 使用 merge 函数将 left 和 right 数据帧按照 "X" 列进行合并，指定合并方式为 join_type
        merged = merge(left, right, on="X", how=join_type)

        # 获取合并后数据帧的列数据类型，并按索引排序
        result = merged.dtypes.sort_index()
        # 根据使用推断字符串的标志位确定预期的数据类型
        dtype = np.dtype("O") if not using_infer_string else "string"
        # 创建预期的列数据类型 Series，包含 ["X", "Y", "Z"] 列的数据类型信息
        expected = Series([dtype, dtype, np.dtype("int64")], index=["X", "Y", "Z"])
        # 使用 assert_series_equal 函数断言 result 和 expected 列数据类型相等
        tm.assert_series_equal(result, expected)
    def test_self_join_multiple_categories(self):
        # GH 16767
        # GH 16767 is the GitHub issue number related to this test case.
        # non-duplicates should work with multiple categories
        # Create a multiplier for data generation.
        m = 5
        # Generate a DataFrame with columns 'a', 'b', 'c', and 'd' using list comprehension.
        df = DataFrame(
            {
                "a": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] * m,
                "b": ["t", "w", "x", "y", "z"] * 2 * m,
                "c": [
                    letter
                    for each in ["m", "n", "u", "p", "o"]
                    for letter in [each] * 2 * m
                ],
                "d": [
                    letter
                    for each in [
                        "aa",
                        "bb",
                        "cc",
                        "dd",
                        "ee",
                        "ff",
                        "gg",
                        "hh",
                        "ii",
                        "jj",
                    ]
                    for letter in [each] * m
                ],
            }
        )

        # change them all to categorical variables
        # Convert all columns in the DataFrame to categorical data type.
        df = df.apply(lambda x: x.astype("category"))

        # self-join should equal ourselves
        # Perform a self-join on the DataFrame 'df' based on all columns.
        result = merge(df, df, on=list(df.columns))

        # Assert that the resulting DataFrame 'result' is equal to the original 'df'.
        tm.assert_frame_equal(result, df)

    def test_dtype_on_categorical_dates(self):
        # GH 16900
        # GH 16900 is the GitHub issue number related to this test case.
        # dates should not be coerced to ints
        # Create a DataFrame 'df' with columns 'date' and 'num2'.
        df = DataFrame(
            [[date(2001, 1, 1), 1.1], [date(2001, 1, 2), 1.3]], columns=["date", "num2"]
        )
        # Convert the 'date' column in 'df' to categorical data type.
        df["date"] = df["date"].astype("category")

        # Create another DataFrame 'df2' with columns 'date' and 'num4'.
        df2 = DataFrame(
            [[date(2001, 1, 1), 1.3], [date(2001, 1, 3), 1.4]], columns=["date", "num4"]
        )
        # Convert the 'date' column in 'df2' to categorical data type.
        df2["date"] = df2["date"].astype("category")

        # Create an expected DataFrame 'expected_outer' for the outer merge result.
        expected_outer = DataFrame(
            [
                [pd.Timestamp("2001-01-01").date(), 1.1, 1.3],
                [pd.Timestamp("2001-01-02").date(), 1.3, np.nan],
                [pd.Timestamp("2001-01-03").date(), np.nan, 1.4],
            ],
            columns=["date", "num2", "num4"],
        )
        # Perform an outer merge of 'df' and 'df2' based on the 'date' column.
        result_outer = merge(df, df2, how="outer", on=["date"])
        # Assert that the resulting DataFrame 'result_outer' matches 'expected_outer'.
        tm.assert_frame_equal(result_outer, expected_outer)

        # Create an expected DataFrame 'expected_inner' for the inner merge result.
        expected_inner = DataFrame(
            [[pd.Timestamp("2001-01-01").date(), 1.1, 1.3]],
            columns=["date", "num2", "num4"],
        )
        # Perform an inner merge of 'df' and 'df2' based on the 'date' column.
        result_inner = merge(df, df2, how="inner", on=["date"])
        # Assert that the resulting DataFrame 'result_inner' matches 'expected_inner'.
        tm.assert_frame_equal(result_inner, expected_inner)

    @pytest.mark.parametrize("ordered", [True, False])
    @pytest.mark.parametrize(
        "category_column,categories,expected_categories",
        [
            ([False, True, True, False], [True, False], [True, False]),
            ([2, 1, 1, 2], [1, 2], [1, 2]),
            (["False", "True", "True", "False"], ["True", "False"], ["True", "False"]),
        ],
    )
    def test_merging_with_bool_or_int_cateorical_column(
        self, category_column, categories, expected_categories, ordered
    ):
        # This is a parameterized test function to test merging behavior with categorical data.
    def test_merge_with_categorical_column(self):
        # GH 17187
        # 创建一个包含 id 和分类列的 DataFrame
        df1 = DataFrame({"id": [1, 2, 3, 4], "cat": category_column})
        # 将分类列转换为指定的分类类型
        df1["cat"] = df1["cat"].astype(CategoricalDtype(categories, ordered=ordered))
        # 创建另一个包含 id 和数值列的 DataFrame
        df2 = DataFrame({"id": [2, 4], "num": [1, 9]})
        # 使用 merge 方法将两个 DataFrame 按照 id 列合并
        result = df1.merge(df2)
        # 创建预期结果的 DataFrame，包含 id、预期的分类列和数值列
        expected = DataFrame({"id": [2, 4], "cat": expected_categories, "num": [1, 9]})
        # 将预期的分类列转换为指定的分类类型
        expected["cat"] = expected["cat"].astype(
            CategoricalDtype(categories, ordered=ordered)
        )
        # 使用 assert_frame_equal 方法比较预期结果和实际结果是否相等
        tm.assert_frame_equal(expected, result)

    def test_merge_on_int_array(self):
        # GH 23020
        # 创建一个包含整数和 NaN 的 Series 的 DataFrame
        df = DataFrame({"A": Series([1, 2, np.nan], dtype="Int64"), "B": 1})
        # 使用 merge 方法根据列 A 进行合并
        result = merge(df, df, on="A")
        # 创建预期的 DataFrame，包含列 A、两个 B 列
        expected = DataFrame(
            {"A": Series([1, 2, np.nan], dtype="Int64"), "B_x": 1, "B_y": 1}
        )
        # 使用 assert_frame_equal 方法比较预期结果和实际结果是否相等
        tm.assert_frame_equal(result, expected)
class TestMergeOnIndexes:
    @pytest.mark.parametrize(
        "how, sort, expected",
        [
            ("inner", False, DataFrame({"a": [20, 10], "b": [200, 100]}, index=[2, 1])),
            ("inner", True, DataFrame({"a": [10, 20], "b": [100, 200]}, index=[1, 2])),
            (
                "left",
                False,
                DataFrame({"a": [20, 10, 0], "b": [200, 100, np.nan]}, index=[2, 1, 0]),
            ),
            (
                "left",
                True,
                DataFrame({"a": [0, 10, 20], "b": [np.nan, 100, 200]}, index=[0, 1, 2]),
            ),
            (
                "right",
                False,
                DataFrame(
                    {"a": [np.nan, 10, 20], "b": [300, 100, 200]}, index=[3, 1, 2]
                ),
            ),
            (
                "right",
                True,
                DataFrame(
                    {"a": [10, 20, np.nan], "b": [100, 200, 300]}, index=[1, 2, 3]
                ),
            ),
            (
                "outer",
                False,
                DataFrame(
                    {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                    index=[0, 1, 2, 3],
                ),
            ),
            (
                "outer",
                True,
                DataFrame(
                    {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                    index=[0, 1, 2, 3],
                ),
            ),
        ],
    )
    def test_merge_on_indexes(self, how, sort, expected):
        left_df = DataFrame({"a": [20, 10, 0]}, index=[2, 1, 0])
        right_df = DataFrame({"b": [300, 100, 200]}, index=[3, 1, 2])
        result = merge(
            left_df, right_df, left_index=True, right_index=True, how=how, sort=sort
        )
        # 断言结果与期望是否相等
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        Index([1, 2, 4], dtype=dtyp, name="index_col")
        for dtyp in tm.ALL_REAL_NUMPY_DTYPES
    ]
    + [
        CategoricalIndex(["A", "B", "C"], categories=["A", "B", "C"], name="index_col"),
        RangeIndex(start=0, stop=3, name="index_col"),
        DatetimeIndex(["2018-01-01", "2018-01-02", "2018-01-03"], name="index_col"),
    ],
    ids=lambda x: f"{type(x).__name__}[{x.dtype}]",
)
def test_merge_index_types(index):
    # gh-20777
    # assert key access is consistent across index types
    # 创建左右两个数据框，使用不同的索引类型进行合并
    left = DataFrame({"left_data": [1, 2, 3]}, index=index)
    right = DataFrame({"right_data": [1.0, 2.0, 3.0]}, index=index)

    # 执行合并操作
    result = left.merge(right, on=["index_col"])

    # 创建期望的合并结果
    expected = DataFrame(
        {"left_data": [1, 2, 3], "right_data": [1.0, 2.0, 3.0]}, index=index
    )
    # 断言合并结果与期望是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "on,left_on,right_on,left_index,right_index,nm",
    # 定义一个包含多个元组的列表，每个元组描述了一组条件参数
    [
        # 第一个元组: ("outer", "inner") 作为第一个参数，其余参数为 None 或者 False，最后一个参数为 "B"
        (["outer", "inner"], None, None, False, False, "B"),
        # 第二个元组: 所有参数为 None，除了倒数第二个参数为 True，倒数第一个参数为 "B"
        (None, None, None, True, True, "B"),
        # 第三个元组: ["outer", "inner"] 作为第二个参数，最后一个参数为 "B"
        (None, ["outer", "inner"], None, False, True, "B"),
        # 第四个元组: ["outer", "inner"] 作为第三个参数，倒数第二个参数为 True
        (None, None, ["outer", "inner"], True, False, "B"),
        # 第五个元组: ("outer", "inner") 作为第一个参数，最后一个参数为 None
        (["outer", "inner"], None, None, False, False, None),
        # 第六个元组: 所有参数为 None，除了倒数第二个参数为 True，倒数第一个参数为 None
        (None, None, None, True, True, None),
        # 第七个元组: ["outer", "inner"] 作为第二个参数，倒数第一个参数为 None
        (None, ["outer", "inner"], None, False, True, None),
        # 第八个元组: ["outer", "inner"] 作为第三个参数，倒数第二个参数为 True，最后一个参数为 None
        (None, None, ["outer", "inner"], True, False, None),
    ],
# 定义了一个测试函数，用于测试 merge 方法的多种情况
def test_merge_series(on, left_on, right_on, left_index, right_index, nm):
    # GH 21220
    # 创建一个 DataFrame a，包含列 "A"，使用 MultiIndex 建立索引
    a = DataFrame(
        {"A": [1, 2, 3, 4]},
        index=MultiIndex.from_product([["a", "b"], [0, 1]], names=["outer", "inner"]),
    )
    
    # 创建一个 Series b，包含数值和 MultiIndex 索引，并指定名称 nm
    b = Series(
        [1, 2, 3, 4],
        index=MultiIndex.from_product([["a", "b"], [1, 2]], names=["outer", "inner"]),
        name=nm,
    )
    
    # 创建预期结果的 DataFrame expected，包含列 "A" 和 "B"，使用 MultiIndex 建立索引
    expected = DataFrame(
        {"A": [2, 4], "B": [1, 3]},
        index=MultiIndex.from_product([["a", "b"], [1]], names=["outer", "inner"]),
    )
    
    # 如果 nm 不为 None，则进行 merge 操作，并断言结果与预期结果相等
    if nm is not None:
        result = merge(
            a,
            b,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
        )
        tm.assert_frame_equal(result, expected)
    else:
        # 如果 nm 为 None，则测试应抛出 ValueError 异常
        msg = "Cannot merge a Series without a name"
        with pytest.raises(ValueError, match=msg):
            result = merge(
                a,
                b,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
            )


# 定义了一个测试函数，用于测试在多级索引下使用 merge 方法时的情况
def test_merge_series_multilevel():
    # GH#47946
    # GH 40993: For raising, enforced in 2.0
    # 创建一个 DataFrame a，包含列 "A"，使用 MultiIndex 建立索引
    a = DataFrame(
        {"A": [1, 2, 3, 4]},
        index=MultiIndex.from_product([["a", "b"], [0, 1]], names=["outer", "inner"]),
    )
    
    # 创建一个 Series b，包含数值和 MultiIndex 索引，且使用元组作为名称
    b = Series(
        [1, 2, 3, 4],
        index=MultiIndex.from_product([["a", "b"], [1, 2]], names=["outer", "inner"]),
        name=("B", "C"),
    )
    
    # 使用 pytest 来断言 merge 操作会抛出 MergeError 异常，且异常信息包含指定字符串
    with pytest.raises(
        MergeError, match="Not allowed to merge between different levels"
    ):
        merge(a, b, on=["outer", "inner"])


# 使用 pytest 的参数化测试功能，测试 merge 方法在不同参数设置下的情况
@pytest.mark.parametrize(
    "col1, col2, kwargs, expected_cols",
    [
        (0, 0, {"suffixes": ("", "_dup")}, ["0", "0_dup"]),
        (0, 0, {"suffixes": (None, "_dup")}, [0, "0_dup"]),
        (0, 0, {"suffixes": ("_x", "_y")}, ["0_x", "0_y"]),
        (0, 0, {"suffixes": ["_x", "_y"]}, ["0_x", "0_y"]),
        ("a", 0, {"suffixes": (None, "_y")}, ["a", 0]),
        (0.0, 0.0, {"suffixes": ("_x", None)}, ["0.0_x", 0.0]),
        ("b", "b", {"suffixes": (None, "_y")}, ["b", "b_y"]),
        ("a", "a", {"suffixes": ("_x", None)}, ["a_x", "a"]),
        ("a", "b", {"suffixes": ("_x", None)}, ["a", "b"]),
        ("a", "a", {"suffixes": (None, "_x")}, ["a", "a_x"]),
        (0, 0, {"suffixes": ("_a", None)}, ["0_a", 0]),
        ("a", "a", {}, ["a_x", "a_y"]),
        (0, 0, {}, ["0_x", "0_y"]),
    ],
)
def test_merge_suffix(col1, col2, kwargs, expected_cols):
    # issue: 24782
    # 创建两个 DataFrame a 和 b，每个都包含一个列，用于测试 merge 方法在不同设置下的行为
    a = DataFrame({col1: [1, 2, 3]})
    b = DataFrame({col2: [4, 5, 6]})
    
    # 创建预期结果的 DataFrame expected，列名由参数 expected_cols 给出
    expected = DataFrame([[1, 4], [2, 5], [3, 6]], columns=expected_cols)
    
    # 使用 merge 方法进行合并，然后断言结果与预期结果相等
    result = a.merge(b, left_index=True, right_index=True, **kwargs)
    tm.assert_frame_equal(result, expected)
    
    # 同样使用顶层的 merge 函数进行测试，断言结果与预期结果相等
    result = merge(a, b, left_index=True, right_index=True, **kwargs)
    # 使用测试框架中的函数来比较两个数据帧是否相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器，为 test_merge_duplicate_suffix 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "how,expected",
    [
        (
            "right",
            {"A": [100, 200, 300], "B1": [60, 70, np.nan], "B2": [600, 700, 800]},
        ),
        (
            "outer",
            {
                "A": [1, 100, 200, 300],
                "B1": [80, 60, 70, np.nan],
                "B2": [np.nan, 600, 700, 800],
            },
        ),
    ],
)
# 定义用于测试 DataFrame 合并操作的函数 test_merge_duplicate_suffix
def test_merge_duplicate_suffix(how, expected):
    # 创建左侧 DataFrame
    left_df = DataFrame({"A": [100, 200, 1], "B": [60, 70, 80]})
    # 创建右侧 DataFrame
    right_df = DataFrame({"A": [100, 200, 300], "B": [600, 700, 800]})
    # 执行合并操作，根据指定的列 A，使用 how 参数进行合并，suffixes 参数设置了后缀 "_x"，"_x"
    result = merge(left_df, right_df, on="A", how=how, suffixes=("_x", "_x"))
    # 根据预期结果 expected 创建 DataFrame 对象
    expected = DataFrame(expected)
    # 设置预期 DataFrame 的列名为 ["A", "B_x", "B_x"]
    expected.columns = ["A", "B_x", "B_x"]
    # 使用 pytest 框架提供的 assert_frame_equal 方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器，为 test_merge_suffix_error 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "col1, col2, suffixes",
    [("a", "a", (None, None)), ("a", "a", ("", None)), (0, 0, (None, ""))],
)
# 定义用于测试 DataFrame 合并操作抛出异常情况的函数 test_merge_suffix_error
def test_merge_suffix_error(col1, col2, suffixes):
    # 创建第一个 DataFrame a，列名为 col1，包含数据 [1, 2, 3]
    a = DataFrame({col1: [1, 2, 3]})
    # 创建第二个 DataFrame b，列名为 col2，包含数据 [3, 4, 5]
    b = DataFrame({col2: [3, 4, 5]})
    # 执行合并操作，设置 left_index=True 和 right_index=True，同时指定 suffixes 参数
    # 期望此处抛出 ValueError 异常，错误信息为 "columns overlap but no suffix specified"
    msg = "columns overlap but no suffix specified"
    with pytest.raises(ValueError, match=msg):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)


# 使用 pytest.mark.parametrize 装饰器，为 test_merge_suffix_raises 函数定义多组参数化测试用例
@pytest.mark.parametrize("suffixes", [{"left", "right"}, {"left": 0, "right": 0}])
# 定义用于测试 DataFrame 合并操作抛出异常情况的函数 test_merge_suffix_raises
def test_merge_suffix_raises(suffixes):
    # 创建第一个 DataFrame a，列名为 "a"，包含数据 [1, 2, 3]
    a = DataFrame({"a": [1, 2, 3]})
    # 创建第二个 DataFrame b，列名为 "b"，包含数据 [3, 4, 5]
    b = DataFrame({"b": [3, 4, 5]})
    # 执行合并操作，设置 left_index=True 和 right_index=True，同时指定 suffixes 参数
    # 期望此处抛出 TypeError 异常，错误信息为 "Passing 'suffixes' as a"
    with pytest.raises(TypeError, match="Passing 'suffixes' as a"):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)


# 使用 pytest.mark.parametrize 装饰器，为 test_merge_suffix_length_error 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "col1, col2, suffixes, msg",
    [
        ("a", "a", ("a", "b", "c"), r"too many values to unpack \(expected 2\)"),
        ("a", "a", tuple("a"), r"not enough values to unpack \(expected 2, got 1\)"),
    ],
)
# 定义用于测试 DataFrame 合并操作抛出异常情况的函数 test_merge_suffix_length_error
def test_merge_suffix_length_error(col1, col2, suffixes, msg):
    # 创建第一个 DataFrame a，列名为 col1，包含数据 [1, 2, 3]
    a = DataFrame({col1: [1, 2, 3]})
    # 创建第二个 DataFrame b，列名为 col2，包含数据 [3, 4, 5]
    b = DataFrame({col2: [3, 4, 5]})
    # 执行合并操作，设置 left_index=True 和 right_index=True，同时指定 suffixes 参数
    # 期望此处抛出 ValueError 异常，错误信息为 msg
    with pytest.raises(ValueError, match=msg):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)


# 使用 pytest.mark.parametrize 装饰器，为 test_merge_equal_cat_dtypes 函数定义多组参数化测试用例
@pytest.mark.parametrize("cat_dtype", ["one", "two"])
@pytest.mark.parametrize("reverse", [True, False])
# 定义用于测试 DataFrame 合并操作的分类数据类型情况的函数 test_merge_equal_cat_dtypes
def test_merge_equal_cat_dtypes(cat_dtype, reverse):
    # 定义分类数据类型的字典 cat_dtypes
    cat_dtypes = {
        "one": CategoricalDtype(categories=["a", "b", "c"], ordered=False),
        "two": CategoricalDtype(categories=["a", "b", "c"], ordered=False),
    }
    # 创建第一个 DataFrame df1，包含列 "foo" 和 "left"，其中 "foo" 使用 cat_dtypes[cat_dtype] 类型
    df1 = DataFrame(
        {"foo": Series(["a", "b", "c"]).astype(cat_dtypes["one"]), "left": [1, 2, 3]}
    ).set_index("foo")

    # 创建数据列表 data_foo 和 data_right
    data_foo = ["a", "b", "c"]
    data_right = [1, 2, 3]

    # 如果 reverse 为 True，则反转 data_foo 和 data_right
    if reverse:
        data_foo.reverse()
        data_right.reverse()

    # 创建第二个 DataFrame df2，包含列 "foo" 和 "right"，其中 "foo" 使用 cat_dtypes[cat_dtype] 类型
    df2 = DataFrame(
        {"foo": Series(data_foo).astype(cat_dtypes[cat_dtype]), "right": data_right}
    ).set_index("foo")

    # 执行合并操作，设置 left_index=True 和 right_index=True
    result = df1.merge(df2, left_index=True, right_index=True)
    # 创建预期的 DataFrame 对象，包括列 "left"、"right" 和 "foo"
    # 其中 "foo" 列通过 Series 创建，并转换为指定的分类数据类型
    expected = DataFrame(
        {
            "left": [1, 2, 3],
            "right": [1, 2, 3],
            "foo": Series(["a", "b", "c"]).astype(cat_dtypes["one"]),
        }
    ).set_index("foo")

    # 使用测试框架中的 assert_frame_equal 方法比较结果 DataFrame 和预期的 DataFrame
    tm.assert_frame_equal(result, expected)
def test_merge_equal_cat_dtypes2():
    # 根据 issue 编号 gh-22501
    # 定义一个非排序的分类数据类型，包含三个类别：'a', 'b', 'c'
    cat_dtype = CategoricalDtype(categories=["a", "b", "c"], ordered=False)

    # 创建第一个DataFrame，设置索引为'foo'列的值，其中'foo'列的类型为上面定义的分类类型
    df1 = DataFrame(
        {"foo": Series(["a", "b"]).astype(cat_dtype), "left": [1, 2]}
    ).set_index("foo")

    # 创建第二个DataFrame，同样设置索引为'foo'列的值，'foo'列类型为之前定义的分类类型
    df2 = DataFrame(
        {"foo": Series(["a", "b", "c"]).astype(cat_dtype), "right": [3, 2, 1]}
    ).set_index("foo")

    # 对两个DataFrame进行按索引合并
    result = df1.merge(df2, left_index=True, right_index=True)

    # 创建预期结果的DataFrame，设置索引为'foo'列的值，'foo'列类型为之前定义的分类类型
    expected = DataFrame(
        {"left": [1, 2], "right": [3, 2], "foo": Series(["a", "b"]).astype(cat_dtype)}
    ).set_index("foo")

    # 使用测试工具比较实际结果和预期结果的DataFrame是否相等
    tm.assert_frame_equal(result, expected)


def test_merge_on_cat_and_ext_array():
    # 根据 issue 编号 GH 28668
    # 创建包含区间数据类型的DataFrame
    right = DataFrame(
        {"a": Series([pd.Interval(0, 1), pd.Interval(1, 2)], dtype="interval")}
    )
    # 复制右侧DataFrame并将'a'列转换为分类数据类型
    left = right.copy()
    left["a"] = left["a"].astype("category")

    # 使用'on'参数在'a'列上进行内连接合并左右两个DataFrame
    result = merge(left, right, how="inner", on="a")
    # 复制右侧DataFrame作为预期结果
    expected = right.copy()

    # 使用测试工具比较实际结果和预期结果的DataFrame是否相等
    tm.assert_frame_equal(result, expected)


def test_merge_multiindex_columns():
    # 根据 issue 编号 Issue #28518
    # 验证合并两个DataFrame后得到预期的标签
    # 此问题的原始原因来自于 bug lexsort_depth，已在 test_lexsort_depth 中测试过

    # 创建多级索引
    letters = ["a", "b", "c", "d"]
    numbers = ["1", "2", "3"]
    index = MultiIndex.from_product((letters, numbers), names=["outer", "inner"])

    # 创建空的DataFrame，并设置'id'列
    frame_x = DataFrame(columns=index)
    frame_x["id"] = ""
    frame_y = DataFrame(columns=index)
    frame_y["id"] = ""

    # 在'id'列上进行合并，并指定后缀
    result = frame_x.merge(frame_y, on="id", suffixes=("_x", "_y"))

    # 构造预期结果的索引
    tuples = [(letter + "_x", num) for letter in letters for num in numbers]
    tuples += [("id", "")]
    tuples += [(letter + "_y", num) for letter in letters for num in numbers]
    expected_index = MultiIndex.from_tuples(tuples, names=["outer", "inner"])

    # 根据预期索引创建DataFrame作为预期结果
    expected = DataFrame(columns=expected_index)

    # 使用测试工具比较实际结果和预期结果的DataFrame是否相等，忽略数据类型检查
    tm.assert_frame_equal(result, expected, check_dtype=False)


def test_merge_datetime_upcast_dtype():
    # 根据 issue 链接 https://github.com/pandas-dev/pandas/issues/31208
    # 创建两个DataFrame
    df1 = DataFrame({"x": ["a", "b", "c"], "y": ["1", "2", "4"]})
    df2 = DataFrame(
        {"y": ["1", "2", "3"], "z": pd.to_datetime(["2000", "2001", "2002"])}
    )

    # 在'y'列上进行左连接合并，并指定左侧DataFrame的'z'列为预期结果的'z'列
    result = merge(df1, df2, how="left", on="y")

    # 创建预期结果的DataFrame，包括'x'、'y'和'z'列
    expected = DataFrame(
        {
            "x": ["a", "b", "c"],
            "y": ["1", "2", "4"],
            "z": pd.to_datetime(["2000", "2001", "NaT"]),
        }
    )

    # 使用测试工具比较实际结果和预期结果的DataFrame是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("n_categories", [5, 128])
def test_categorical_non_unique_monotonic(n_categories):
    # 根据 issue 编号 GH 28189
    # 当 n_categories 为 5 时，测试 libjoin 中的 int8 情况，
    # 当 n_categories 为 128 时，测试 int16 情况。

    # 创建分类索引，包含指定数量的类别
    left_index = CategoricalIndex([0] + list(range(n_categories)))
    df1 = DataFrame(range(n_categories + 1), columns=["value"], index=left_index)
    # 创建一个 DataFrame df2，包含一个值为6的单元格，列名为"value"，索引为CategoricalIndex([0])，分类列表从0到n_categories-1
    df2 = DataFrame(
        [[6]],
        columns=["value"],
        index=CategoricalIndex([0], categories=list(range(n_categories))),
    )

    # 使用左连接（left join）合并 df1 和 df2，使用它们的索引作为连接键
    result = merge(df1, df2, how="left", left_index=True, right_index=True)

    # 创建一个期望的 DataFrame expected，根据给定的规则，列名为"value_x"和"value_y"，索引为 left_index
    expected = DataFrame(
        [[i, 6.0] if i < 2 else [i, np.nan] for i in range(n_categories + 1)],
        columns=["value_x", "value_y"],
        index=left_index,
    )

    # 使用测试工具（test tool）tm.assert_frame_equal 比较期望的 DataFrame 和实际得到的结果 result
    tm.assert_frame_equal(expected, result)
# 定义测试函数，用于测试多级分类数据的合并和连接操作
def test_merge_join_categorical_multiindex():
    # From issue 16627
    # 创建第一个DataFrame a，包含一个分类列"Cat1"和一个整数列"Int1"
    a = {
        "Cat1": Categorical(["a", "b", "a", "c", "a", "b"], ["a", "b", "c"]),
        "Int1": [0, 1, 0, 1, 0, 0],
    }
    a = DataFrame(a)

    # 创建第二个DataFrame b，包含一个分类列"Cat"、一个整数列"Int"和一个因子列"Factor"，并将"Cat"和"Int"设为索引
    b = {
        "Cat": Categorical(["a", "b", "c", "a", "b", "c"], ["a", "b", "c"]),
        "Int": [0, 0, 0, 1, 1, 1],
        "Factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    }
    b = DataFrame(b).set_index(["Cat", "Int"])["Factor"]

    # 使用merge函数将DataFrame a和重设索引后的DataFrame b按照"Cat1"和"Int1"列左连接，赋值给expected
    expected = merge(
        a,
        b.reset_index(),
        left_on=["Cat1", "Int1"],
        right_on=["Cat", "Int"],
        how="left",
    )
    # 从expected中删除"Cat"和"Int"列，axis=1表示删除列
    expected = expected.drop(["Cat", "Int"], axis=1)
    # 使用join方法将DataFrame a和DataFrame b按照"Cat1"和"Int1"列连接，赋值给result
    result = a.join(b, on=["Cat1", "Int1"])
    # 使用assert_frame_equal函数断言expected和result是否相等
    tm.assert_frame_equal(expected, result)

    # Same test, but with ordered categorical
    # 创建具有有序分类的新的DataFrame a，指定分类顺序为["b", "a", "c"]
    a = {
        "Cat1": Categorical(
            ["a", "b", "a", "c", "a", "b"], ["b", "a", "c"], ordered=True
        ),
        "Int1": [0, 1, 0, 1, 0, 0],
    }
    a = DataFrame(a)

    # 创建具有有序分类的新的DataFrame b，指定分类顺序为["b", "a", "c"]
    b = {
        "Cat": Categorical(
            ["a", "b", "c", "a", "b", "c"], ["b", "a", "c"], ordered=True
        ),
        "Int": [0, 0, 0, 1, 1, 1],
        "Factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    }
    b = DataFrame(b).set_index(["Cat", "Int"])["Factor"]

    # 使用merge函数将DataFrame a和重设索引后的DataFrame b按照"Cat1"和"Int1"列左连接，赋值给expected
    expected = merge(
        a,
        b.reset_index(),
        left_on=["Cat1", "Int1"],
        right_on=["Cat", "Int"],
        how="left",
    )
    # 从expected中删除"Cat"和"Int"列，axis=1表示删除列
    expected = expected.drop(["Cat", "Int"], axis=1)
    # 使用join方法将DataFrame a和DataFrame b按照"Cat1"和"Int1"列连接，赋值给result
    result = a.join(b, on=["Cat1", "Int1"])
    # 使用assert_frame_equal函数断言expected和result是否相等
    tm.assert_frame_equal(expected, result)


# 定义测试函数，用于测试合并和连接操作时的错误报告机制
@pytest.mark.parametrize("func", ["merge", "merge_asof"])
@pytest.mark.parametrize(
    ("kwargs", "err_msg"),
    [
        # 测试左连接时传递"left_on"和"left_index"同时出现的错误报告
        ({"left_on": "a", "left_index": True}, ["left_on", "left_index"]),
        # 测试右连接时传递"right_on"和"right_index"同时出现的错误报告
        ({"right_on": "a", "right_index": True}, ["right_on", "right_index"]),
    ],
)
def test_merge_join_cols_error_reporting_duplicates(func, kwargs, err_msg):
    # GH: 16228
    # 创建左侧DataFrame，包含列"a"和"b"
    left = DataFrame({"a": [1, 2], "b": [3, 4]})
    # 创建右侧DataFrame，包含列"a"和"c"
    right = DataFrame({"a": [1, 1], "c": [5, 6]})
    # 期望的错误消息，匹配正则表达式rf'Can only pass argument "{err_msg[0]}" OR "{err_msg[1]}" not both\.'
    msg = rf'Can only pass argument "{err_msg[0]}" OR "{err_msg[1]}" not both\.'
    # 使用pytest.raises断言在调用merge函数时会抛出MergeError，并匹配msg错误消息
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, **kwargs)


# 定义测试函数，用于测试合并和连接操作时的错误报告机制
@pytest.mark.parametrize("func", ["merge", "merge_asof"])
@pytest.mark.parametrize(
    ("kwargs", "err_msg"),
    [
        # 测试左连接时未传递"right_on"或"right_index"导致的错误报告
        ({"left_on": "a"}, ["right_on", "right_index"]),
        # 测试右连接时未传递"left_on"或"left_index"导致的错误报告
        ({"right_on": "a"}, ["left_on", "left_index"]),
    ],
)
def test_merge_join_cols_error_reporting_missing(func, kwargs, err_msg):
    # GH: 16228
    # 创建左侧DataFrame，包含列"a"和"b"
    left = DataFrame({"a": [1, 2], "b": [3, 4]})
    # 创建右侧DataFrame，包含列"a"和"c"
    right = DataFrame({"a": [1, 1], "c": [5, 6]})
    # 期望的错误消息，匹配正则表达式rf'Must pass "{err_msg[0]}" OR "{err_msg[1]}"\.'
    msg = rf'Must pass "{err_msg[0]}" OR "{err_msg[1]}"\.'
    # 使用pytest.raises断言在调用merge函数时会抛出MergeError，并匹配msg错误消息
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, **kwargs)


# 定义测试函数，用于测试合并和连接操作时的错误报告机制
@pytest.mark.parametrize("func", ["merge", "merge_asof"])
@pytest.mark.parametrize(
    "kwargs",
    [
        # 测试右连接时只传递"right_index"参数的错误报告
        {"right_index": True},
        # 测试左连接时只传递"left_index"参数的错误报告
        {"left_index": True},
    ],
)
def test_merge_join_cols_error_reporting_on_and_index(func, kwargs):
    # GH: 16228
    # 创建左侧和右侧的 DataFrame
    left = DataFrame({"a": [1, 2], "b": [3, 4]})
    right = DataFrame({"a": [1, 1], "c": [5, 6]})
    # 定义错误消息
    msg = (
        r'Can only pass argument "on" OR "left_index" '
        r'and "right_index", not a combination of both\.'
    )
    # 断言调用指定函数时抛出 MergeError 异常，并匹配预期的错误消息
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, on="a", **kwargs)


def test_merge_right_left_index():
    # GH#38616
    # 创建左侧和右侧的 DataFrame
    left = DataFrame({"x": [1, 1], "z": ["foo", "foo"]})
    right = DataFrame({"x": [1, 1], "z": ["foo", "foo"]})
    # 执行合并操作，指定右侧使用索引，左侧使用列 'x' 进行合并
    result = merge(left, right, how="right", left_index=True, right_on="x")
    # 创建预期的合并结果 DataFrame
    expected = DataFrame(
        {
            "x": [1, 1],
            "x_x": [1, 1],
            "z_x": ["foo", "foo"],
            "x_y": [1, 1],
            "z_y": ["foo", "foo"],
        }
    )
    # 断言实际结果与预期结果相等
    tm.assert_frame_equal(result, expected)


def test_merge_result_empty_index_and_on():
    # GH#33814
    # 创建左侧和右侧的 DataFrame，并设置索引
    df1 = DataFrame({"a": [1], "b": [2]}).set_index(["a", "b"])
    df2 = DataFrame({"b": [1]}).set_index(["b"])
    # 创建预期的空 DataFrame，设置相同的索引
    expected = DataFrame({"a": [], "b": []}, dtype=np.int64).set_index(["a", "b"])
    # 执行基于左侧列 'b' 和右侧索引的合并
    result = merge(df1, df2, left_on=["b"], right_index=True)
    # 断言实际结果与预期结果相等
    tm.assert_frame_equal(result, expected)
    # 执行基于右侧列 'b' 和左侧索引的合并
    result = merge(df2, df1, left_index=True, right_on=["b"])
    # 断言实际结果与预期结果相等
    tm.assert_frame_equal(result, expected)


def test_merge_suffixes_produce_dup_columns_raises():
    # GH#22818; Enforced in 2.0
    # 创建左侧和右侧的 DataFrame
    left = DataFrame({"a": [1, 2, 3], "b": 1, "b_x": 2})
    right = DataFrame({"a": [1, 2, 3], "b": 2})
    # 断言合并时使用后缀会导致重复列的 MergeError 异常
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(left, right, on="a")
    # 断言合并时使用后缀会导致重复列的 MergeError 异常
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(right, left, on="a", suffixes=("_y", "_x"))


def test_merge_duplicate_columns_with_suffix_no_warning():
    # GH#22818
    # 创建左侧包含重复列名的 DataFrame
    left = DataFrame([[1, 1, 1], [2, 2, 2]], columns=["a", "b", "b"])
    right = DataFrame({"a": [1, 3], "b": 2})
    # 执行合并操作，不会因为源 DataFrame 存在重复列而引发警告
    result = merge(left, right, on="a")
    # 创建预期的合并结果 DataFrame
    expected = DataFrame([[1, 1, 1, 2]], columns=["a", "b_x", "b_x", "b_y"])
    # 断言实际结果与预期结果相等
    tm.assert_frame_equal(result, expected)


def test_merge_duplicate_columns_with_suffix_causing_another_duplicate_raises():
    # GH#22818, Enforced in 2.0
    # 创建左侧包含重复列名并可能导致后缀冲突的 DataFrame
    left = DataFrame([[1, 1, 1, 1], [2, 2, 2, 2]], columns=["a", "b", "b", "b_x"])
    right = DataFrame({"a": [1, 3], "b": 2})
    # 断言合并时使用后缀会导致重复列的 MergeError 异常
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(left, right, on="a")


def test_merge_string_float_column_result():
    # GH 13353
    # 创建具有字符串和浮点数列名的 DataFrame
    df1 = DataFrame([[1, 2], [3, 4]], columns=Index(["a", 114.0]))
    df2 = DataFrame([[9, 10], [11, 12]], columns=["x", "y"])
    # 执行内连接合并，使用索引进行合并
    result = merge(df2, df1, how="inner", left_index=True, right_index=True)
    # 创建一个期望的 DataFrame 对象，包含两行数据：
    # 第一行是 [9, 10, 1, 2]，第二行是 [11, 12, 3, 4]
    # 指定列的名称为 ["x", "y", "a", 114.0]
    expected = DataFrame(
        [[9, 10, 1, 2], [11, 12, 3, 4]], columns=Index(["x", "y", "a", 114.0])
    )
    
    # 使用测试框架中的方法 tm.assert_frame_equal 比较 result 和 expected 两个 DataFrame 对象
    tm.assert_frame_equal(result, expected)
# 测试函数，验证在左侧索引类型不匹配的情况下是否会引发 MergeError 异常
def test_mergeerror_on_left_index_mismatched_dtypes():
    # GH 22449
    # 创建包含单个数据 "X" 的 DataFrame，列名为 "C"，索引为 22
    df_1 = DataFrame(data=["X"], columns=["C"], index=[22])
    # 创建包含单个数据 "X" 的 DataFrame，列名为 "C"，索引为 999
    df_2 = DataFrame(data=["X"], columns=["C"], index=[999])
    # 使用 pytest 断言检查 merge 函数调用时是否会引发 MergeError，并且错误信息包含指定文本
    with pytest.raises(MergeError, match="Can only pass argument"):
        merge(df_1, df_2, on=["C"], left_index=True)


# 测试函数，验证在左侧索引为 CategoricalIndex 类型时不会引发异常
def test_merge_on_left_categoricalindex():
    # GH#48464 don't raise when left_on is a CategoricalIndex
    # 创建一个 CategoricalIndex，包含范围为 [0, 1, 2] 的索引
    ci = CategoricalIndex(range(3))
    
    # 创建一个 DataFrame，包含两列 "A" 和 "B"，数据为 ci 和范围为 [0, 1, 2] 的整数
    right = DataFrame({"A": ci, "B": range(3)})
    # 创建一个 DataFrame，包含一列 "C"，数据为范围为 [3, 4, 5] 的整数
    left = DataFrame({"C": range(3, 6)})
    
    # 调用 merge 函数，左侧使用 left DataFrame，右侧使用 right DataFrame，
    # 左侧关联键为 ci，右侧关联键为 "A"
    res = merge(left, right, left_on=ci, right_on="A")
    # 期望的结果 DataFrame，左侧关联键为 ci._data，右侧关联键为 "A"
    expected = merge(left, right, left_on=ci._data, right_on="A")
    # 使用 tm.assert_frame_equal 检查 res 和 expected 是否相等
    tm.assert_frame_equal(res, expected)


# 测试函数，验证在使用 NaN 进行外连接时的 merge 函数行为
def test_merge_outer_with_NaN(dtype):
    # GH#43550
    # 创建一个 DataFrame，包含两列 "key" 和 "col1"，数据类型为 dtype
    left = DataFrame({"key": [1, 2], "col1": [1, 2]}, dtype=dtype)
    # 创建一个 DataFrame，包含两列 "key" 和 "col2"，其中 "key" 的值为 NaN，数据类型为 dtype
    right = DataFrame({"key": [np.nan, np.nan], "col2": [3, 4]}, dtype=dtype)
    # 执行外连接，左侧使用 left DataFrame，右侧使用 right DataFrame，关联键为 "key"
    result = merge(left, right, on="key", how="outer")
    # 期望的结果 DataFrame，包含三列 "key"、"col1"、"col2"，其中 "key" 包括 NaN 值
    expected = DataFrame(
        {
            "key": [1, 2, np.nan, np.nan],
            "col1": [1, 2, np.nan, np.nan],
            "col2": [np.nan, np.nan, 3, 4],
        },
        dtype=dtype,
    )
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 交换左右 DataFrame 进行外连接
    result = merge(right, left, on="key", how="outer")
    # 期望的结果 DataFrame，包含三列 "key"、"col2"、"col1"，其中 "key" 包括 NaN 值
    expected = DataFrame(
        {
            "key": [1, 2, np.nan, np.nan],
            "col2": [np.nan, np.nan, 3, 4],
            "col1": [1, 2, np.nan, np.nan],
        },
        dtype=dtype,
    )
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证在左右 DataFrame 使用不同索引名称时的 merge 函数行为
def test_merge_different_index_names():
    # GH#45094
    # 创建一个 DataFrame，包含一列 "a"，索引名称为 "c"
    left = DataFrame({"a": [1]}, index=Index([1], name="c"))
    # 创建一个 DataFrame，包含一列 "a"，索引名称为 "d"
    right = DataFrame({"a": [1]}, index=Index([1], name="d"))
    # 执行 merge 操作，左侧使用 left DataFrame，右侧使用 right DataFrame，
    # 左侧关联键为 "c"，右侧关联键为 "d"
    result = merge(left, right, left_on="c", right_on="d")
    # 期望的结果 DataFrame，包含两列 "a_x" 和 "a_y"
    expected = DataFrame({"a_x": [1], "a_y": 1})
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证在不同数据类型的情况下 merge 函数的行为
def test_merge_ea(any_numeric_ea_dtype, join_type):
    # GH#44240
    # 创建一个 DataFrame，包含两列 "a" 和 "b"，数据类型为 any_numeric_ea_dtype
    left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
    # 创建一个 DataFrame，包含两列 "a" 和 "c"，数据类型为 any_numeric_ea_dtype
    right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype)
    # 执行 merge 操作，左侧使用 left DataFrame，右侧使用 right DataFrame，连接类型为 join_type
    result = left.merge(right, how=join_type)
    # 期望的结果 DataFrame，包含三列 "a"、"b"、"c"，数据类型为 any_numeric_ea_dtype
    expected = DataFrame({"a": [1, 2, 3], "b": 1, "c": 2}, dtype=any_numeric_ea_dtype)
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证在合并不同数据类型和非 ea 类型时的 merge 函数行为
def test_merge_ea_and_non_ea(any_numeric_ea_dtype, join_type):
    # GH#44240
    # 创建一个 DataFrame，包含两列 "a" 和 "b"，数据类型为 any_numeric_ea_dtype
    left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
    # 创建一个 DataFrame，包含两列 "a" 和 "c"，数据类型为 any_numeric_ea_dtype.lower()
    right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype.lower())
    # 执行 merge 操作，左侧使用 left DataFrame，右侧使用 right DataFrame，连接类型为 join_type
    result = left.merge(right, how=join_type)
    # 期望的结果 DataFrame，包含三列 "a"、"b"、"c"，数据类型分别为 any_numeric_ea_dtype 和 any_numeric_ea_dtype.lower()
    expected = DataFrame(
        {
            "a": Series([1, 2, 3], dtype=any_numeric_ea_dtype),
            "b": Series([1, 1, 1], dtype=any_numeric_ea_dtype),
            "c": Series([2, 2, 2], dtype=any_numeric_ea_dtype.lower()),
        }
    )
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器指定参数 dtype 为 "int64" 和 "int64[pyarrow]"
@pytest.mark.parametrize("dtype", ["int64", "int64[pyarrow]"])
# 定义一个测试函数，用于测试合并 Arrow 和 NumPy 的数据类型
def test_merge_arrow_and_numpy_dtypes(dtype):
    # 检查是否能够导入 pyarrow 库，否则跳过测试
    pytest.importorskip("pyarrow")
    # 创建一个包含整数列 'a' 的 DataFrame，数据类型为传入的 dtype
    df = DataFrame({"a": [1, 2]}, dtype=dtype)
    # 创建另一个 DataFrame，包含整数列 'a'，数据类型为 "int64[pyarrow]"
    df2 = DataFrame({"a": [1, 2]}, dtype="int64[pyarrow]")
    # 将 df 和 df2 进行合并
    result = df.merge(df2)
    # 期望的合并结果应该与 df 相同
    expected = df.copy()
    # 使用测试框架检查结果是否符合预期
    tm.assert_frame_equal(result, expected)

    # 将 df2 和 df 进行合并
    result = df2.merge(df)
    # 期望的合并结果应该与 df2 相同
    expected = df2.copy()
    # 使用测试框架检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


# 使用参数化装饰器定义测试函数，测试不同时间分辨率的日期时间合并
@pytest.mark.parametrize("tz", [None, "America/Chicago"])
def test_merge_datetime_different_resolution(tz, join_type):
    # 创建包含三个带时区信息的时间戳的 DataFrame
    vals = [
        pd.Timestamp(2023, 5, 12, tz=tz),
        pd.Timestamp(2023, 5, 13, tz=tz),
        pd.Timestamp(2023, 5, 14, tz=tz),
    ]
    df1 = DataFrame({"t": vals[:2], "a": [1.0, 2.0]})
    # 将 df1 中的 't' 列时间单位转换为纳秒
    df1["t"] = df1["t"].dt.as_unit("ns")
    df2 = DataFrame({"t": vals[1:], "b": [1.0, 2.0]})
    # 将 df2 中的 't' 列时间单位转换为秒
    df2["t"] = df2["t"].dt.as_unit("s")

    # 创建期望的结果 DataFrame
    expected = DataFrame({"t": vals, "a": [1.0, 2.0, np.nan], "b": [np.nan, 1.0, 2.0]})
    expected["t"] = expected["t"].dt.as_unit("ns")
    
    # 根据不同的 join_type 设置期望的结果
    if join_type == "inner":
        expected = expected.iloc[[1]].reset_index(drop=True)
    elif join_type == "left":
        expected = expected.iloc[[0, 1]]
    elif join_type == "right":
        expected = expected.iloc[[1, 2]].reset_index(drop=True)

    # 执行 df1 和 df2 的合并操作，根据 join_type 指定的合并方式
    result = df1.merge(df2, on="t", how=join_type)
    # 使用测试框架检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试在单层多索引下的 DataFrame 合并
def test_merge_multiindex_single_level():
    # 创建包含一个列 'col' 的 DataFrame
    df = DataFrame({"col": ["A", "B"]})
    # 创建一个包含数据为 100 的 'b' 列的 DataFrame，使用多索引，索引名为 'col'
    df2 = DataFrame(
        data={"b": [100]},
        index=MultiIndex.from_tuples([("A",), ("C",)], names=["col"]),
    )
    # 创建期望的结果 DataFrame
    expected = DataFrame({"col": ["A", "B"], "b": [100, np.nan]})

    # 执行 df 和 df2 的左连接操作，以左表的 'col' 列和右表的索引进行匹配
    result = df.merge(df2, left_on=["col"], right_index=True, how="left")
    # 使用测试框架检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


# 使用参数化装饰器定义测试函数，测试不同合并方式的组合
@pytest.mark.parametrize("on_index", [True, False])
@pytest.mark.parametrize("left_unique", [True, False])
@pytest.mark.parametrize("left_monotonic", [True, False])
@pytest.mark.parametrize("right_unique", [True, False])
@pytest.mark.parametrize("right_monotonic", [True, False])
def test_merge_combinations(
    join_type,
    sort,
    on_index,
    left_unique,
    left_monotonic,
    right_unique,
    right_monotonic,
):
    # 设置合并的方式
    how = join_type
    # 创建左表的键值列表
    left = [2, 3]
    if left_unique:
        left.append(4 if left_monotonic else 1)
    else:
        left.append(3 if left_monotonic else 2)

    # 创建右表的键值列表
    right = [2, 3]
    if right_unique:
        right.append(4 if right_monotonic else 1)
    else:
        right.append(3 if right_monotonic else 2)

    # 创建左表和右表的 DataFrame
    left = DataFrame({"key": left})
    right = DataFrame({"key": right})

    # 如果 on_index 为 True，则设置左表和右表的索引为 'key'
    if on_index:
        left = left.set_index("key")
        right = right.set_index("key")
        on_kwargs = {"left_index": True, "right_index": True}
    else:
        on_kwargs = {"on": "key"}

    # 执行左表和右表的合并操作，根据指定的合并方式和其他参数
    result = merge(left, right, how=how, sort=sort, **on_kwargs)

    # 如果 on_index 为 True，则将左表和右表重置索引为默认索引
    if on_index:
        left = left.reset_index()
        right = right.reset_index()
    # 如果连接方式是 "left", "right", 或者 "inner" 中的一种
    if how in ["left", "right", "inner"]:
        # 如果连接方式是 "left" 或者 "inner"
        if how in ["left", "inner"]:
            # 设置预期值、另一侧数据、另一侧数据唯一值
            expected, other, other_unique = left, right, right_unique
        else:
            # 设置预期值、另一侧数据、另一侧数据唯一值
            expected, other, other_unique = right, left, left_unique
        
        # 如果连接方式是 "inner"
        if how == "inner":
            # 找出左右两侧数据中键的交集
            keep_values = set(left["key"].values).intersection(right["key"].values)
            # 创建布尔掩码，用于筛选预期数据中在交集中的键
            keep_mask = expected["key"].isin(keep_values)
            expected = expected[keep_mask]  # 更新预期数据为交集中的数据
        
        # 如果需要排序
        if sort:
            expected = expected.sort_values("key")  # 根据键值排序
        
        # 如果另一侧数据中的键不唯一
        if not other_unique:
            # 计算另一侧数据中每个键的重复次数
            other_value_counts = other["key"].value_counts()
            # 根据预期数据中的键重新索引，填充缺失值为1
            repeats = other_value_counts.reindex(expected["key"].values, fill_value=1)
            repeats = repeats.astype(np.intp)  # 转换为整数类型
            # 根据重复次数重复预期数据中的键，并转换为数据帧形式
            expected = expected["key"].repeat(repeats.values)
            expected = expected.to_frame()
    
    # 如果连接方式是 "outer"
    elif how == "outer":
        # 计算左右两侧数据中每个键的出现次数
        left_counts = left["key"].value_counts()
        right_counts = right["key"].value_counts()
        # 计算预期数据中每个键的预期出现次数
        expected_counts = left_counts.mul(right_counts, fill_value=1)
        expected_counts = expected_counts.astype(np.intp)  # 转换为整数类型
        # 根据预期出现次数重复预期数据中的键，并转换为数据帧形式
        expected = expected_counts.index.values.repeat(expected_counts.values)
        expected = DataFrame({"key": expected})
        expected = expected.sort_values("key")  # 根据键值排序
    
    # 如果按索引连接
    if on_index:
        expected = expected.set_index("key")  # 将 "key" 列设置为索引
    else:
        expected = expected.reset_index(drop=True)  # 重置索引为默认数值索引
    
    # 使用测试框架验证结果和预期数据框是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试合并包含整数和浮点数的 Pandas DataFrame 的行为
def test_merge_ea_int_and_float_numpy():
    # GH#46178，引用 GitHub 上的 issue 编号
    # 创建包含一个浮点数和一个空值的 DataFrame，数据类型为 pd.Int64Dtype
    df1 = DataFrame([1.0, np.nan], dtype=pd.Int64Dtype())
    # 创建包含一个浮点数的 DataFrame
    df2 = DataFrame([1.5])
    # 创建一个预期的空 DataFrame，列名为 [0]，数据类型为 "Int64"
    expected = DataFrame(columns=[0], dtype="Int64")

    # 断言在合并 df1 和 df2 时会产生 UserWarning 警告，警告信息包含 "You are merging"
    with tm.assert_produces_warning(UserWarning, match="You are merging"):
        result = df1.merge(df2)
    # 断言合并的结果与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)

    # 同样断言在合并 df2 和 df1 时会产生 UserWarning 警告，警告信息包含 "You are merging"
    with tm.assert_produces_warning(UserWarning, match="You are merging"):
        result = df2.merge(df1)
    # 断言合并的结果与预期的 DataFrame 相等，并且将结果转换为 "float64" 数据类型
    tm.assert_frame_equal(result, expected.astype("float64"))

    # 修改 df2 的内容为包含一个整数的 DataFrame
    df2 = DataFrame([1.0])
    # 创建一个预期的 DataFrame，包含一个整数值，列名为 [0]，数据类型为 "Int64"
    expected = DataFrame([1], columns=[0], dtype="Int64")
    # 断言合并 df1 和 df2 的结果与预期的 DataFrame 相等
    result = df1.merge(df2)
    tm.assert_frame_equal(result, expected)

    # 断言合并 df2 和 df1 的结果与预期的 DataFrame 相等，并且将结果转换为 "float64" 数据类型
    result = df2.merge(df1)
    tm.assert_frame_equal(result, expected.astype("float64"))


# 定义一个测试函数，测试 Pandas DataFrame 在使用 Arrow 字符串索引进行合并的行为
def test_merge_arrow_string_index(any_string_dtype):
    # GH#54894，引用 GitHub 上的 issue 编号
    # 确保 pyarrow 库已导入，如果未导入则跳过该测试
    pytest.importorskip("pyarrow")
    # 创建一个包含字符串类型的 DataFrame，列名为 "a"
    left = DataFrame({"a": ["a", "b"]}, dtype=any_string_dtype)
    # 创建一个包含整数值的 DataFrame，索引为包含字符串的 Index
    right = DataFrame({"b": 1}, index=Index(["a", "c"], dtype=any_string_dtype))
    # 执行左连接，以 "left_on" 参数指定左侧 DataFrame 的列名，"right_index=True" 表示右侧使用索引进行合并
    result = left.merge(right, left_on="a", right_index=True, how="left")
    # 创建一个预期的 DataFrame，包含列 "a" 和 "b"，其中右侧匹配不到的行填充为 NaN
    expected = DataFrame(
        {"a": Series(["a", "b"], dtype=any_string_dtype), "b": [1, np.nan]}
    )
    # 断言合并的结果与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试合并空 DataFrame 列顺序的行为
@pytest.mark.parametrize("left_empty", [True, False])
@pytest.mark.parametrize("right_empty", [True, False])
def test_merge_empty_frames_column_order(left_empty, right_empty):
    # GH 51929，引用 GitHub 上的 issue 编号
    # 创建一个包含整数值的 DataFrame，包含列 "A" 和 "B"
    df1 = DataFrame(1, index=[0], columns=["A", "B"])
    # 创建一个包含整数值的 DataFrame，包含列 "A"、"C" 和 "D"
    df2 = DataFrame(1, index=[0], columns=["A", "C", "D"])

    # 如果 left_empty 为 True，则将 df1 截取为空 DataFrame
    if left_empty:
        df1 = df1.iloc[:0]
    # 如果 right_empty 为 True，则将 df2 截取为空 DataFrame
    if right_empty:
        df2 = df2.iloc[:0]

    # 执行外连接（outer join），合并 df1 和 df2 的结果
    result = merge(df1, df2, on=["A"], how="outer")
    # 创建一个预期的 DataFrame，包含列 "A"、"B"、"C" 和 "D"
    expected = DataFrame(1, index=[0], columns=["A", "B", "C", "D"])
    # 根据 left_empty 和 right_empty 的组合设置预期的结果
    if left_empty and right_empty:
        expected = expected.iloc[:0]
    elif left_empty:
        expected["B"] = np.nan
    elif right_empty:
        expected[["C", "D"]] = np.nan
    # 断言合并的结果与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试合并包含 datetime 和 timedelta 的 Pandas DataFrame 的行为
@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_merge_datetime_and_timedelta(how):
    # 创建包含一个 datetime64[ns] 类型列的 DataFrame
    left = DataFrame({"key": Series([1, None], dtype="datetime64[ns]")})
    # 创建包含一个 timedelta64[ns] 类型列的 DataFrame
    right = DataFrame({"key": Series([1], dtype="timedelta64[ns]")})

    # 构建 ValueError 的错误信息
    msg = (
        f"You are trying to merge on {left['key'].dtype} and {right['key'].dtype} "
        "columns for key 'key'. If you wish to proceed you should use pd.concat"
    )
    # 断言在合并时会抛出 ValueError 异常，异常信息匹配预定义的错误信息
    with pytest.raises(ValueError, match=re.escape(msg)):
        left.merge(right, on="key", how=how)

    # 构建 ValueError 的错误信息（反向合并）
    msg = (
        f"You are trying to merge on {right['key'].dtype} and {left['key'].dtype} "
        "columns for key 'key'. If you wish to proceed you should use pd.concat"
    )
    # 断言在合并时会抛出 ValueError 异常，异常信息匹配预定义的错误信息
    with pytest.raises(ValueError, match=re.escape(msg)):
        right.merge(left, on="key", how=how)
```