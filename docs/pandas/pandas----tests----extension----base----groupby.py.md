# `D:\src\scipysrc\pandas\pandas\tests\extension\base\groupby.py`

```
# 导入正则表达式模块
import re

# 导入 pytest 测试框架
import pytest

# 从 pandas 库中导入常用数据类型检查函数
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

# 导入 pandas 库，并使用 pd 别名
import pandas as pd

# 导入 pandas 内部测试工具
import pandas._testing as tm

# 使用 pytest 标记忽略特定的警告
@pytest.mark.filterwarnings(
    "ignore:The default of observed=False is deprecated:FutureWarning"
)
# 定义一个基础的 Groupby 测试类
class BaseGroupbyTests:
    """Groupby-specific tests."""

    # 测试组别功能的 Grouper 对象
    def test_grouping_grouper(self, data_for_grouping):
        # 创建一个 DataFrame 对象 df，包含列 A 和 B
        df = pd.DataFrame(
            {
                "A": pd.Series(
                    ["B", "B", None, None, "A", "A", "B", "C"], dtype=object
                ),
                "B": data_for_grouping,
            }
        )
        # 获取按 A 列分组后的第一个 grouper 对象
        gr1 = df.groupby("A")._grouper.groupings[0]
        # 获取按 B 列分组后的第一个 grouper 对象
        gr2 = df.groupby("B")._grouper.groupings[0]

        # 断言 grouper 对象中的 grouping_vector 与 df 的 A 列值相等
        tm.assert_numpy_array_equal(gr1.grouping_vector, df.A.values)
        # 断言 grouper 对象中的 grouping_vector 与 data_for_grouping 相等
        tm.assert_extension_array_equal(gr2.grouping_vector, data_for_grouping)

    # 使用参数化测试不同的 as_index 值
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        # 创建一个 DataFrame 对象 df，包含列 A 和 B
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})

        # 检查 data_for_grouping 的数据类型是否为布尔型
        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # 如果是布尔型，只保留前面的数据，根据文档字符串条件
            df = df.iloc[:-1]

        # 对 B 列进行分组并计算 A 列的平均值
        result = df.groupby("B", as_index=as_index).A.mean()
        # 使用 factorize 函数获取唯一值和索引
        _, uniques = pd.factorize(data_for_grouping, sort=True)

        # 预期的 A 列平均值列表
        exp_vals = [3.0, 1.0, 4.0]
        if is_bool:
            exp_vals = exp_vals[:-1]
        if as_index:
            # 如果 as_index 为 True，创建预期的 Series 对象
            index = pd.Index(uniques, name="B")
            expected = pd.Series(exp_vals, index=index, name="A")
            # 断言结果与预期相等
            tm.assert_series_equal(result, expected)
        else:
            # 如果 as_index 为 False，创建预期的 DataFrame 对象
            expected = pd.DataFrame({"B": uniques, "A": exp_vals})
            # 断言结果 DataFrame 与预期 DataFrame 相等
            tm.assert_frame_equal(result, expected)

    # 测试基于扩展类型的 groupby 聚合
    def test_groupby_agg_extension(self, data_for_grouping):
        # 创建一个 DataFrame 对象 df，包含列 A 和 B
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})

        # 选择出符合条件的行并设置 A 列为索引
        expected = df.iloc[[0, 2, 4, 7]]
        expected = expected.set_index("A")

        # 对 A 列进行分组并对 B 列应用 "first" 聚合函数
        result = df.groupby("A").agg({"B": "first"})
        # 断言结果 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 对 A 列进行分组并应用 "first" 聚合函数
        result = df.groupby("A").agg("first")
        # 断言结果 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 对 A 列进行分组并应用 first() 方法
        result = df.groupby("A").first()
        # 断言结果 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于对数据进行分组，不排序
    def test_groupby_extension_no_sort(self, data_for_grouping):
        # 创建包含两列的 DataFrame，其中一列为整数，另一列为传入的数据
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        
        # 检查数据是否为布尔类型
        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # 如果数据是布尔类型，移除最后一行数据
            # 这里假设最后一个条目的 c==b （参见 data_for_grouping 的文档字符串）
            df = df.iloc[:-1]
        
        # 对 DataFrame 按 B 列进行分组，计算 A 列的平均值，结果不排序
        result = df.groupby("B", sort=False).A.mean()
        
        # 使用 pd.factorize 函数得到数据的唯一值及其对应的索引
        _, index = pd.factorize(data_for_grouping, sort=False)
        
        # 创建一个索引对象，以数据的唯一值命名为 B
        index = pd.Index(index, name="B")
        
        # 预期的 A 列的数值，根据数据类型是否为布尔类型来确定
        exp_vals = [1.0, 3.0, 4.0]
        if is_bool:
            exp_vals = exp_vals[:-1]
        
        # 创建预期的结果 Series，包含预期的 A 列数值及其对应的索引
        expected = pd.Series(exp_vals, index=index, name="A")
        
        # 断言计算结果与预期结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，对数据进行分组并进行变换操作
    def test_groupby_extension_transform(self, data_for_grouping):
        # 检查数据是否为布尔类型
        is_bool = data_for_grouping.dtype._is_boolean

        # 从 data_for_grouping 中选取非空值，创建包含两列的 DataFrame
        valid = data_for_grouping[~data_for_grouping.isna()]
        df = pd.DataFrame({"A": [1, 1, 3, 3, 1, 4], "B": valid})
        
        # 再次检查数据是否为布尔类型
        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # 如果数据是布尔类型，移除最后一行数据
            # 这里假设最后一个条目的 c==b （参见 data_for_grouping 的文档字符串）
            df = df.iloc[:-1]
        
        # 对 DataFrame 按 B 列进行分组，计算 A 列的长度
        result = df.groupby("B").A.transform(len)
        
        # 创建预期的结果 Series，包含预期的 A 列长度及其对应的名称
        expected = pd.Series([3, 3, 2, 2, 3, 1], name="A")
        if is_bool:
            expected = expected[:-1]
        
        # 断言计算结果与预期结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，对数据进行分组并应用自定义操作
    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        # 创建包含两列的 DataFrame，其中一列为整数，另一列为传入的数据
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        
        # 设置警告消息内容
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        
        # 断言产生警告，警告类型为 DeprecationWarning，消息内容匹配设定的 msg
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 对 DataFrame 按 B 列进行分组，并应用自定义的操作函数 groupby_apply_op
            df.groupby("B", group_keys=False, observed=False).apply(groupby_apply_op)
        
        # 对 DataFrame 按 B 列进行分组，并对 A 列应用自定义的操作函数 groupby_apply_op
        df.groupby("B", group_keys=False, observed=False).A.apply(groupby_apply_op)
        
        # 再次设置警告消息内容
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        
        # 断言产生警告，警告类型为 DeprecationWarning，消息内容匹配设定的 msg
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 对 DataFrame 按 A 列进行分组，并应用自定义的操作函数 groupby_apply_op
            df.groupby("A", group_keys=False, observed=False).apply(groupby_apply_op)
        
        # 对 DataFrame 按 A 列进行分组，并对 B 列应用自定义的操作函数 groupby_apply_op

    # 定义一个测试函数，对数据进行分组，并应用匿名函数获取数据数组
    def test_groupby_apply_identity(self, data_for_grouping):
        # 创建包含两列的 DataFrame，其中一列为整数，另一列为传入的数据
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        
        # 对 DataFrame 按 A 列进行分组，并应用匿名函数获取 B 列的数据数组
        result = df.groupby("A").B.apply(lambda x: x.array)
        
        # 创建预期的结果 Series，包含按 A 列分组后的 B 列数据数组
        expected = pd.Series(
            [
                df.B.iloc[[0, 1, 6]].array,
                df.B.iloc[[2, 3]].array,
                df.B.iloc[[4, 5]].array,
                df.B.iloc[[7]].array,
            ],
            index=pd.Index([1, 2, 3, 4], name="A"),
            name="B",
        )
        
        # 断言计算结果与预期结果是否相等
        tm.assert_series_equal(result, expected)
    # 定义一个测试函数，用于测试在数据分组时，处理不同数据类型的情况
    def test_in_numeric_groupby(self, data_for_grouping):
        # 创建一个 Pandas 数据框，包含三列数据：A 列有重复数字作为分组依据，B 列包含输入的数据，C 列全为1
        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 4],
                "B": data_for_grouping,
                "C": [1, 1, 1, 1, 1, 1, 1, 1],
            }
        )

        # 获取数据列的数据类型
        dtype = data_for_grouping.dtype
        # 检查数据类型是否为数值型、布尔型、decimal 或字符串型，或者包含 'm' 类型（特指 duration 类型）
        if (
            is_numeric_dtype(dtype)
            or is_bool_dtype(dtype)
            or dtype.name == "decimal"
            or is_string_dtype(dtype)
            or is_object_dtype(dtype)
            or dtype.kind == "m"  # 特别指定的 duration 类型（pyarrow）
        ):
            # 如果数据类型满足条件，期望的结果为包含 "B" 和 "C" 的索引
            expected = pd.Index(["B", "C"])
            # 对数据框按列 A 进行分组，然后求和并获取结果的列索引
            result = df.groupby("A").sum().columns
        else:
            # 如果数据类型不满足条件，期望的结果为只包含 "C" 的索引
            expected = pd.Index(["C"])

            # 准备匹配的错误消息，用于测试在此条件下是否会引发 TypeError 异常
            msg = "|".join(
                [
                    # period 类型不支持 sum 操作
                    "does not support sum operations",
                    # datetime 类型不支持 'sum' 操作
                    "does not support operation 'sum'",
                    # 其他所有类型的错误消息，指明聚合函数失败的原因和数据类型
                    re.escape(f"agg function failed [how->sum,dtype->{dtype}"),
                ]
            )
            # 使用 pytest 检查在求和操作时是否引发了预期的 TypeError 异常，并匹配错误消息
            with pytest.raises(TypeError, match=msg):
                df.groupby("A").sum()
            # 对数据框按列 A 进行分组，只对数值类型进行求和并获取结果的列索引
            result = df.groupby("A").sum(numeric_only=True).columns
        
        # 使用 Pandas 的 assert_index_equal 函数比较实际结果和期望结果的索引是否一致
        tm.assert_index_equal(result, expected)
```