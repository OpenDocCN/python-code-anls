# `D:\src\scipysrc\pandas\pandas\tests\generic\test_finalize.py`

```
"""
An exhaustive list of pandas methods exercising NDFrame.__finalize__.
"""

# 引入所需的库和模块
import operator  # 运算符模块，用于操作函数
import re  # 正则表达式模块

import numpy as np  # 数值计算库
import pytest  # 测试框架

import pandas as pd  # 数据处理和分析库

# TODO: 待完成列表
# * 二元方法（mul, div, 等）
# * 二元输出（align, 等）
# * 顶层方法（concat, merge, get_dummies, 等）
# * 窗口方法
# * 累积减少方法

# 定义 pytest 标记，标记为“未实现”
not_implemented_mark = pytest.mark.xfail(reason="not implemented")

# 创建一个多重索引对象
mi = pd.MultiIndex.from_product([["a", "b"], [0, 1]], names=["A", "B"])

# 创建测试数据
frame_data = ({"A": [1]},)  # DataFrame 数据
frame_mi_data = ({"A": [1, 2, 3, 4]}, mi)  # 多重索引 DataFrame 数据

# 所有测试方法的列表
_all_methods = [
    # 以下是一系列测试方法的元组，每个元组包含：
    # - 类型构造器：Series 或 DataFrame
    # - 构造器参数
    # - 使用 operator.methodcaller 创建的调用对象，设置属性为此值
    (pd.Series, ([0],), operator.methodcaller("take", [])),  # Series.take 方法
    (pd.Series, ([0],), operator.methodcaller("__getitem__", [True])),  # Series.__getitem__ 方法
    (pd.Series, ([0],), operator.methodcaller("repeat", 2)),  # Series.repeat 方法
    (pd.Series, ([0],), operator.methodcaller("reset_index")),  # Series.reset_index 方法
    (pd.Series, ([0],), operator.methodcaller("reset_index", drop=True)),  # Series.reset_index 方法，drop=True
    (pd.Series, ([0],), operator.methodcaller("to_frame")),  # Series.to_frame 方法
    (pd.Series, ([0, 0],), operator.methodcaller("drop_duplicates")),  # Series.drop_duplicates 方法
    (pd.Series, ([0, 0],), operator.methodcaller("duplicated")),  # Series.duplicated 方法
    (pd.Series, ([0, 0],), operator.methodcaller("round")),  # Series.round 方法
    (pd.Series, ([0, 0],), operator.methodcaller("rename", lambda x: x + 1)),  # Series.rename 方法，接受 lambda 函数
    (pd.Series, ([0, 0],), operator.methodcaller("rename", "name")),  # Series.rename 方法，接受字符串参数
    (pd.Series, ([0, 0],), operator.methodcaller("set_axis", ["a", "b"])),  # Series.set_axis 方法
    (pd.Series, ([0, 0],), operator.methodcaller("reindex", [1, 0])),  # Series.reindex 方法
    (pd.Series, ([0, 0],), operator.methodcaller("drop", [0])),  # Series.drop 方法
    (pd.Series, (pd.array([0, pd.NA]),), operator.methodcaller("fillna", 0)),  # Series.fillna 方法
    (pd.Series, ([0, 0],), operator.methodcaller("replace", {0: 1})),  # Series.replace 方法
    (pd.Series, ([0, 0],), operator.methodcaller("shift")),  # Series.shift 方法
    (pd.Series, ([0, 0],), operator.methodcaller("isin", [0, 1])),  # Series.isin 方法
    (pd.Series, ([0, 0],), operator.methodcaller("between", 0, 2)),  # Series.between 方法
    (pd.Series, ([0, 0],), operator.methodcaller("isna")),  # Series.isna 方法
    (pd.Series, ([0, 0],), operator.methodcaller("isnull")),  # Series.isnull 方法
    (pd.Series, ([0, 0],), operator.methodcaller("notna")),  # Series.notna 方法
    (pd.Series, ([0, 0],), operator.methodcaller("notnull")),  # Series.notnull 方法
    (pd.Series, ([1],), operator.methodcaller("add", pd.Series([1]))),  # Series.add 方法
    # TODO: mul, div, etc.  # 待完成：mul, div 等方法
    (
        pd.Series,
        ([0], pd.period_range("2000", periods=1)),
        operator.methodcaller("to_timestamp"),  # Series.to_timestamp 方法
    ),
    (
        pd.Series,
        ([0], pd.date_range("2000", periods=1)),
        operator.methodcaller("to_period"),  # Series.to_period 方法
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("dot", pd.DataFrame(index=["A"])),  # DataFrame.dot 方法
        ),
        marks=pytest.mark.xfail(reason="Implement binary finalize"),  # 标记为“未实现”
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("transpose")),  # DataFrame.transpose 方法
    (pd.DataFrame, frame_data, operator.methodcaller("__getitem__", "A")),  # DataFrame.__getitem__ 方法
]
    (pd.DataFrame, frame_data, operator.methodcaller("__getitem__", ["A"])),
    # 调用 DataFrame 的 __getitem__ 方法，获取列名为 "A" 的列数据

    (pd.DataFrame, frame_data, operator.methodcaller("__getitem__", np.array([True]))),
    # 调用 DataFrame 的 __getitem__ 方法，使用布尔数组作为索引，获取对应位置为 True 的行数据

    (pd.DataFrame, ({("A", "a"): [1]},), operator.methodcaller("__getitem__", ["A"])),
    # 在包含元组的字典中，使用元组索引获取列名为 "A" 的列数据

    (pd.DataFrame, frame_data, operator.methodcaller("query", "A == 1")),
    # 使用 query 方法查询 DataFrame，选择满足条件 "A == 1" 的行

    (pd.DataFrame, frame_data, operator.methodcaller("eval", "A + 1", engine="python")),
    # 使用 eval 方法在 Python 引擎中计算表达式 "A + 1"，并返回结果

    (pd.DataFrame, frame_data, operator.methodcaller("select_dtypes", include="int")),
    # 使用 select_dtypes 方法选择指定数据类型（这里是整数）的列

    (pd.DataFrame, frame_data, operator.methodcaller("assign", b=1)),
    # 使用 assign 方法添加新列 "b"，并赋值为 1

    (pd.DataFrame, frame_data, operator.methodcaller("set_axis", ["A"])),
    # 使用 set_axis 方法设置 DataFrame 的轴标签为 ["A"]

    (pd.DataFrame, frame_data, operator.methodcaller("reindex", [0, 1])),
    # 使用 reindex 方法重新索引 DataFrame，只保留索引为 0 和 1 的行

    (pd.DataFrame, frame_data, operator.methodcaller("drop", columns=["A"])),
    # 使用 drop 方法删除指定列名为 "A" 的列

    (pd.DataFrame, frame_data, operator.methodcaller("drop", index=[0])),
    # 使用 drop 方法删除指定索引为 0 的行

    (pd.DataFrame, frame_data, operator.methodcaller("rename", columns={"A": "a"})),
    # 使用 rename 方法重命名列名 "A" 为 "a"

    (pd.DataFrame, frame_data, operator.methodcaller("rename", index=lambda x: x)),
    # 使用 rename 方法对索引进行重命名，使用 lambda 函数保持不变

    (pd.DataFrame, frame_data, operator.methodcaller("fillna", "A")),
    # 使用 fillna 方法填充缺失值，列 "A" 使用指定的填充值

    (pd.DataFrame, frame_data, operator.methodcaller("set_index", "A")),
    # 使用 set_index 方法将列 "A" 设置为索引

    (pd.DataFrame, frame_data, operator.methodcaller("reset_index")),
    # 使用 reset_index 方法重置索引，将现有索引转换为列，并重新生成默认整数索引

    (pd.DataFrame, frame_data, operator.methodcaller("isna")),
    # 使用 isna 方法检查 DataFrame 中的缺失值，返回布尔值 DataFrame

    (pd.DataFrame, frame_data, operator.methodcaller("isnull")),
    # 使用 isnull 方法检查 DataFrame 中的缺失值，返回布尔值 DataFrame

    (pd.DataFrame, frame_data, operator.methodcaller("notna")),
    # 使用 notna 方法检查 DataFrame 中的非缺失值，返回布尔值 DataFrame

    (pd.DataFrame, frame_data, operator.methodcaller("notnull")),
    # 使用 notnull 方法检查 DataFrame 中的非缺失值，返回布尔值 DataFrame

    (pd.DataFrame, frame_data, operator.methodcaller("dropna")),
    # 使用 dropna 方法删除包含缺失值的行或列

    (pd.DataFrame, frame_data, operator.methodcaller("drop_duplicates")),
    # 使用 drop_duplicates 方法删除重复的行

    (pd.DataFrame, frame_data, operator.methodcaller("duplicated")),
    # 使用 duplicated 方法标记重复的行，并返回布尔值 Series

    (pd.DataFrame, frame_data, operator.methodcaller("sort_values", by="A")),
    # 使用 sort_values 方法按照列 "A" 的值排序 DataFrame

    (pd.DataFrame, frame_data, operator.methodcaller("sort_index")),
    # 使用 sort_index 方法按照索引排序 DataFrame

    (pd.DataFrame, frame_data, operator.methodcaller("nlargest", 1, "A")),
    # 使用 nlargest 方法获取按列 "A" 最大的前 1 个值对应的行

    (pd.DataFrame, frame_data, operator.methodcaller("nsmallest", 1, "A")),
    # 使用 nsmallest 方法获取按列 "A" 最小的前 1 个值对应的行

    (pd.DataFrame, frame_mi_data, operator.methodcaller("swaplevel")),
    # 使用 swaplevel 方法交换 MultiIndex 的级别

    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("add", pd.DataFrame(*frame_data)),
    ),
    # 使用 add 方法将两个 DataFrame 对象逐元素相加

    # TODO: div, mul, etc. (未实现的部分，用于提醒添加其他操作符的功能)

    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("combine", pd.DataFrame(*frame_data), operator.add),
    ),
    # 使用 combine 方法将两个 DataFrame 对象逐元素地组合

    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("combine_first", pd.DataFrame(*frame_data)),
    ),
    # 使用 combine_first 方法将当前 DataFrame 的缺失值用另一个 DataFrame 的对应值填充

    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("update", pd.DataFrame(*frame_data)),
        ),
        marks=not_implemented_mark,
    ),
    # 使用 update 方法将另一个 DataFrame 中的非 NA 值更新到当前 DataFrame 中

    (pd.DataFrame, frame_data, operator.methodcaller("pivot", columns="A")),
    # 使用 pivot 方法根据列 "A" 的值重塑 DataFrame

    (
        pd.DataFrame,
        ({"A": [1], "B": [1]},),
        operator.methodcaller("pivot_table", columns="A"),
    ),
    # 使用 pivot_table 方法根据列 "A" 创建透视表
    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 pivot_table 方法，按列'A'进行透视，聚合函数包括'mean'和'sum'
    (
        pd.DataFrame,
        ({"A": [1], "B": [1]},),
        operator.methodcaller("pivot_table", columns="A", aggfunc=["mean", "sum"]),
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 stack 方法
    (pd.DataFrame, frame_data, operator.methodcaller("stack")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 explode 方法，指定列'A'进行展开
    (pd.DataFrame, frame_data, operator.methodcaller("explode", "A")),

    # 创建一个包含 DataFrame 类型、一个多级索引数据帧和一个方法调用器的元组，该方法调用器调用 unstack 方法
    (pd.DataFrame, frame_mi_data, operator.methodcaller("unstack")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 melt 方法，设置'id_vars'为['A']，'value_vars'为['B']
    (
        pd.DataFrame,
        ({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]},),
        operator.methodcaller("melt", id_vars=["A"], value_vars=["B"]),
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 map 方法，映射函数为 lambda 函数
    (pd.DataFrame, frame_data, operator.methodcaller("map", lambda x: x)),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 merge 方法，与给定的 DataFrame 合并
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("merge", pd.DataFrame({"A": [1]})),
        ),
        marks=not_implemented_mark,  # 添加标记，表示该功能未实现
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 round 方法，保留两位小数
    (pd.DataFrame, frame_data, operator.methodcaller("round", 2)),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 corr 方法，计算相关系数
    (pd.DataFrame, frame_data, operator.methodcaller("corr")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 cov 方法，计算协方差
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("cov")),
        marks=[
            pytest.mark.filterwarnings("ignore::RuntimeWarning"),  # 添加标记，忽略运行时警告
        ],
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 corrwith 方法，计算与另一个 DataFrame 的相关系数
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("corrwith", pd.DataFrame(*frame_data)),
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 count 方法，计算非空元素的数量
    (pd.DataFrame, frame_data, operator.methodcaller("count")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 nunique 方法，计算唯一值的数量
    (pd.DataFrame, frame_data, operator.methodcaller("nunique")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 idxmin 方法，返回最小值的索引
    (pd.DataFrame, frame_data, operator.methodcaller("idxmin")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 idxmax 方法，返回最大值的索引
    (pd.DataFrame, frame_data, operator.methodcaller("idxmax")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 mode 方法，计算众数
    (pd.DataFrame, frame_data, operator.methodcaller("mode")),

    # 创建一个包含 Series 类型、一个列表和一个方法调用器的元组，该方法调用器调用 mode 方法，计算众数
    (pd.Series, [0], operator.methodcaller("mode")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 median 方法，计算中位数
    (pd.DataFrame, frame_data, operator.methodcaller("median")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 quantile 方法，计算分位数，仅对数值类型有效
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("quantile", numeric_only=True),
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 quantile 方法，计算指定分位数，仅对数值类型有效
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("quantile", q=[0.25, 0.75], numeric_only=True),
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 quantile 方法，计算分位数，支持时间类型
    (
        pd.DataFrame,
        ({"A": [pd.Timedelta(days=1), pd.Timedelta(days=2)]},),
        operator.methodcaller("quantile", numeric_only=False),
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 quantile 方法，计算分位数，仅对数值类型有效
    (
        pd.DataFrame,
        ({"A": [np.datetime64("2022-01-01"), np.datetime64("2022-01-02")]},),
        operator.methodcaller("quantile", numeric_only=True),
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 to_timestamp 方法，将时间数据转换为时间戳
    (
        pd.DataFrame,
        ({"A": [1]}, [pd.Period("2000", "D")]),
        operator.methodcaller("to_timestamp"),
    ),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器的元组，该方法调用器调用 to_period 方法，将时间戳数据转换为周期
    (
        pd.DataFrame,
        ({"A": [1]}, [pd.Timestamp("2000")]),
        operator.methodcaller("to_period", freq="D"),
    ),

    # 创建一个包含 DataFrame 类型、一个多级索引数据帧和一个方法调用器的元组，该方法调用器调用 isin 方法，检查元素是否在给定值中
    (pd.DataFrame, frame_mi_data, operator.methodcaller("isin", [1])),

    # 创建一个包含 DataFrame 类型、一个多级索引数据帧和一个方法调用器的元组，该方法调用器调用 isin 方法，检查元素是否在给定 Series 中
    (pd.DataFrame, frame_mi_data, operator.methodcaller("isin", pd.Series([1]))),

    # 创建一个包含 DataFrame 类型、一个多级索引数据帧和一个方法调用器的元组，该方法调用器调用 isin 方法，检查元素是否在给定 DataFrame 中
    (
        pd.DataFrame,
        frame_mi_data,
        operator.methodcaller("isin", pd.DataFrame({"A": [1]})),
    ),

    # 创建一个包含 DataFrame 类型、一个多级索引数据帧和一个方法调用器的元组，该方法调用器调用 droplevel 方法，删除指定级别的索引
    (pd.DataFrame, frame_mi_data, operator.methodcaller("droplevel", "A")),

    # 创建一个包含 DataFrame 类型、一个数据帧和一个方法调用器
    # 对于列进行压缩，否则会得到一个标量值
    (pd.DataFrame, frame_data, operator.methodcaller("squeeze", axis="columns")),
    # 对于 Series，进行压缩操作，即将单维度的 Series 转换为标量值
    (pd.Series, ([1, 2],), operator.methodcaller("squeeze")),
    # 对于 Series，重命名索引为 'a'
    (pd.Series, ([1, 2],), operator.methodcaller("rename_axis", index="a")),
    # 对于 DataFrame，重命名列索引为 'a'
    (pd.DataFrame, frame_data, operator.methodcaller("rename_axis", columns="a")),

    # 一元操作
    (pd.DataFrame, frame_data, operator.neg),  # DataFrame 中的元素取负值
    (pd.Series, [1], operator.neg),  # Series 中的元素取负值
    (pd.DataFrame, frame_data, operator.pos),  # DataFrame 中的元素取正值
    (pd.Series, [1], operator.pos),  # Series 中的元素取正值
    (pd.DataFrame, frame_data, operator.inv),  # DataFrame 中的元素按位取反
    (pd.Series, [1], operator.inv),  # Series 中的元素按位取反
    (pd.DataFrame, frame_data, abs),  # DataFrame 中的元素取绝对值
    (pd.Series, [1], abs),  # Series 中的元素取绝对值
    (pd.DataFrame, frame_data, round),  # DataFrame 中的元素四舍五入
    (pd.Series, [1], round),  # Series 中的元素四舍五入

    # 调用操作方法
    (pd.DataFrame, frame_data, operator.methodcaller("take", [0, 0])),  # DataFrame 中按照索引列表取值
    (pd.DataFrame, frame_mi_data, operator.methodcaller("xs", "a")),  # 从多级索引的 DataFrame 中取出特定索引的行
    (pd.Series, (1, mi), operator.methodcaller("xs", "a")),  # 从多级索引的 Series 中取出特定索引的值
    (pd.DataFrame, frame_data, operator.methodcaller("get", "A")),  # 获取 DataFrame 中特定列的 Series
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("reindex_like", pd.DataFrame({"A": [1, 2, 3]})),
    ),  # 根据给定 DataFrame 重新索引当前 DataFrame
    (
        pd.Series,
        frame_data,
        operator.methodcaller("reindex_like", pd.Series([0, 1, 2])),
    ),  # 根据给定 Series 重新索引当前 Series

    # 添加前缀和后缀
    (pd.DataFrame, frame_data, operator.methodcaller("add_prefix", "_")),  # 为 DataFrame 的列名添加前缀 '_'
    (pd.DataFrame, frame_data, operator.methodcaller("add_suffix", "_")),  # 为 DataFrame 的列名添加后缀 '_'
    (pd.Series, (1, ["a", "b"]), operator.methodcaller("add_prefix", "_")),  # 为 Series 的索引添加前缀 '_'
    (pd.Series, (1, ["a", "b"]), operator.methodcaller("add_suffix", "_")),  # 为 Series 的索引添加后缀 '_'

    # 排序和选择
    (pd.Series, ([3, 2],), operator.methodcaller("sort_values")),  # 对 Series 进行排序
    (pd.Series, ([1] * 10,), operator.methodcaller("head")),  # 获取 Series 的前几行
    (pd.DataFrame, ({"A": [1] * 10},), operator.methodcaller("head")),  # 获取 DataFrame 的前几行
    (pd.Series, ([1] * 10,), operator.methodcaller("tail")),  # 获取 Series 的后几行
    (pd.DataFrame, ({"A": [1] * 10},), operator.methodcaller("tail")),  # 获取 DataFrame 的后几行

    # 随机抽样
    (pd.Series, ([1, 2],), operator.methodcaller("sample", n=2, replace=True)),  # 对 Series 进行随机抽样
    (pd.DataFrame, (frame_data,), operator.methodcaller("sample", n=2, replace=True)),  # 对 DataFrame 进行随机抽样

    # 类型转换和数据操作
    (pd.Series, ([1, 2],), operator.methodcaller("astype", float)),  # 将 Series 中的元素类型转换为 float
    (pd.DataFrame, frame_data, operator.methodcaller("astype", float)),  # 将 DataFrame 中的元素类型转换为 float
    (pd.Series, ([1, 2],), operator.methodcaller("copy")),  # 复制 Series
    (pd.DataFrame, frame_data, operator.methodcaller("copy")),  # 复制 DataFrame
    (pd.Series, ([1, 2], None, object), operator.methodcaller("infer_objects")),  # 推断 Series 中的对象类型
    (
        pd.DataFrame,
        ({"A": np.array([1, 2], dtype=object)},),
        operator.methodcaller("infer_objects"),
    ),  # 推断 DataFrame 中的对象类型
    (pd.Series, ([1, 2],), operator.methodcaller("convert_dtypes")),  # 将 Series 转换为适当的类型
    (pd.DataFrame, frame_data, operator.methodcaller("convert_dtypes")),  # 将 DataFrame 转换为适当的类型

    # 缺失值处理
    (pd.Series, ([1, None, 3],), operator.methodcaller("interpolate")),  # 对 Series 中的缺失值进行插值处理
    (pd.DataFrame, ({"A": [1, None, 3]},), operator.methodcaller("interpolate")),  # 对 DataFrame 中的缺失值进行插值处理

    # 值范围限制
    (pd.Series, ([1, 2],), operator.methodcaller("clip", lower=1)),  # 将 Series 中小于 lower 的值设为 lower
    (pd.DataFrame, frame_data, operator.methodcaller("clip", lower=1)),  # 将 DataFrame 中小于 lower 的值设为 lower
    # 创建一个元组，包含一个 pandas Series 和相关参数，以及一个调用特定方法的函数
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),  # 创建一个包含日期范围的元组作为参数
        operator.methodcaller("asfreq", "h"),   # 创建一个方法调用对象，调用 asfreq("h")
    ),
    # 创建一个元组，包含一个 pandas DataFrame 和相关参数，以及一个调用特定方法的函数
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),  # 创建一个包含数据和日期范围的元组作为参数
        operator.methodcaller("asfreq", "h"),   # 创建一个方法调用对象，调用 asfreq("h")
    ),
    # 类似地，以下每个元组都包含一个 pandas 对象（Series 或 DataFrame）、参数元组和方法调用对象
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("at_time", "12:00"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("at_time", "12:00"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("between_time", "12:00", "13:00"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("between_time", "12:00", "13:00"),
    ),
    # ...
    # 余下的每个元组都代表一个 pandas 对象（Series 或 DataFrame）、参数元组和方法调用对象
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.Series 的对象
    # - 包含单个列表 [1] 的元组（作为参数传递给方法调用器）
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "cummax" 方法的对象
    (pd.Series, ([1],), operator.methodcaller("cummax")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "cummax" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("cummax")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.Series 的对象
    # - 包含单个列表 [1] 的元组（作为参数传递给方法调用器）
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "cumprod" 方法的对象
    (pd.Series, ([1],), operator.methodcaller("cumprod")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "cumprod" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("cumprod")),
    
    # Reductions（减少操作）
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "any" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("any")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "all" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("all")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "min" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("min")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "max" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("max")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "sum" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("sum")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "std" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("std")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "mean" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("mean")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "prod" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("prod")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "sem" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("sem")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "skew" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("skew")),
    
    # 创建一个元组列表，每个元组包含以下内容：
    # - 类型为 pd.DataFrame 的对象
    # - frame_data 变量的内容作为数据
    # - 使用 operator 模块的 methodcaller 方法创建一个调用 "kurt" 方法的对象
    (pd.DataFrame, frame_data, operator.methodcaller("kurt")),
# 结束了上一个测试函数的定义块

def idfn(x):
    # 创建正则表达式对象，用于匹配单引号间的内容
    xpr = re.compile(r"'(.*)?'")
    # 在参数 x 的字符串表示中搜索匹配项
    m = xpr.search(str(x))
    # 如果找到匹配项，则返回匹配到的内容
    if m:
        return m.group(1)
    else:
        # 如果没有找到匹配项，则返回参数 x 的字符串表示
        return str(x)


@pytest.mark.parametrize("ndframe_method", _all_methods, ids=lambda x: idfn(x[-1]))
def test_finalize_called(ndframe_method):
    # 解包参数：类、初始化参数、方法名
    cls, init_args, method = ndframe_method
    # 创建类实例
    ndframe = cls(*init_args)

    # 设置实例属性 attrs 为字典 {"a": 1}
    ndframe.attrs = {"a": 1}
    # 调用指定的方法，并获取结果
    result = method(ndframe)

    # 断言结果的 attrs 属性等于 {"a": 1}
    assert result.attrs == {"a": 1}


@not_implemented_mark
def test_finalize_called_eval_numexpr():
    # 导入 numexpr 模块，如果不存在则跳过测试
    pytest.importorskip("numexpr")
    # 创建 DataFrame 实例
    df = pd.DataFrame({"A": [1, 2]})
    # 设置实例属性 attrs["A"] 为 1
    df.attrs["A"] = 1
    # 使用 numexpr 引擎计算表达式 "A + 1" 的结果
    result = df.eval("A + 1", engine="numexpr")
    # 断言结果的 attrs 属性等于 {"A": 1}
    assert result.attrs == {"A": 1}


# ----------------------------------------------------------------------------
# 二元操作


@pytest.mark.parametrize("annotate", ["left", "right", "both"])
@pytest.mark.parametrize(
    "args",
    [
        (1, pd.Series([1])),
        (1, pd.DataFrame({"A": [1]})),
        (pd.Series([1]), 1),
        (pd.DataFrame({"A": [1]}), 1),
        (pd.Series([1]), pd.Series([1])),
        (pd.DataFrame({"A": [1]}), pd.DataFrame({"A": [1]})),
        (pd.Series([1]), pd.DataFrame({"A": [1]})),
        (pd.DataFrame({"A": [1]}), pd.Series([1])),
    ],
    ids=lambda x: f"({type(x[0]).__name__},{type(x[1]).__name__})",
)
def test_binops(request, args, annotate, all_binary_operators):
    # 此处生成了 624 个测试... 是否有必要？
    left, right = args
    # 如果 left 是 DataFrame 或 Series 类型，则将其 attrs 属性设置为空字典
    if isinstance(left, (pd.DataFrame, pd.Series)):
        left.attrs = {}
    # 如果 right 是 DataFrame 或 Series 类型，则将其 attrs 属性设置为空字典
    if isinstance(right, (pd.DataFrame, pd.Series)):
        right.attrs = {}

    # 如果 annotate 为 "left" 并且 left 是 int 类型，则跳过测试
    if annotate == "left" and isinstance(left, int):
        pytest.skip("left is an int and doesn't support .attrs")
    # 如果 annotate 为 "right" 并且 right 是 int 类型，则跳过测试
    if annotate == "right" and isinstance(right, int):
        pytest.skip("right is an int and doesn't support .attrs")
    # 检查左右操作数是否都不是整数，并且 annotate 不等于 "both"
    if not (isinstance(left, int) or isinstance(right, int)) and annotate != "both":
        # 检查 all_binary_operators 函数对象名是否不是以 "r" 开头
        if not all_binary_operators.__name__.startswith("r"):
            # 如果 annotate 为 "right" 并且 left 和 right 类型相同
            if annotate == "right" and isinstance(left, type(right)):
                # 应用 pytest.mark.xfail 标记，标记为预期失败，给出失败原因
                request.applymarker(
                    pytest.mark.xfail(
                        reason=f"{all_binary_operators} doesn't work when right has "
                        f"attrs and both are {type(left)}"
                    )
                )
            # 如果 left 和 right 类型不同
            if not isinstance(left, type(right)):
                # 如果 annotate 为 "left" 并且 left 是 pandas Series 对象
                if annotate == "left" and isinstance(left, pd.Series):
                    # 应用 pytest.mark.xfail 标记，标记为预期失败，给出失败原因
                    request.applymarker(
                        pytest.mark.xfail(
                            reason=f"{all_binary_operators} doesn't work when the "
                            "objects are different Series has attrs"
                        )
                    )
                # 如果 annotate 为 "right" 并且 right 是 pandas Series 对象
                elif annotate == "right" and isinstance(right, pd.Series):
                    # 应用 pytest.mark.xfail 标记，标记为预期失败，给出失败原因
                    request.applymarker(
                        pytest.mark.xfail(
                            reason=f"{all_binary_operators} doesn't work when the "
                            "objects are different Series has attrs"
                        )
                    )
        else:
            # 如果 annotate 为 "left" 并且 left 和 right 类型相同
            if annotate == "left" and isinstance(left, type(right)):
                # 应用 pytest.mark.xfail 标记，标记为预期失败，给出失败原因
                request.applymarker(
                    pytest.mark.xfail(
                        reason=f"{all_binary_operators} doesn't work when left has "
                        f"attrs and both are {type(left)}"
                    )
                )
            # 如果 left 和 right 类型不同
            if not isinstance(left, type(right)):
                # 如果 annotate 为 "right" 并且 right 是 pandas Series 对象
                if annotate == "right" and isinstance(right, pd.Series):
                    # 应用 pytest.mark.xfail 标记，标记为预期失败，给出失败原因
                    request.applymarker(
                        pytest.mark.xfail(
                            reason=f"{all_binary_operators} doesn't work when the "
                            "objects are different Series has attrs"
                        )
                    )
                # 如果 annotate 为 "left" 并且 left 是 pandas Series 对象
                elif annotate == "left" and isinstance(left, pd.Series):
                    # 应用 pytest.mark.xfail 标记，标记为预期失败，给出失败原因
                    request.applymarker(
                        pytest.mark.xfail(
                            reason=f"{all_binary_operators} doesn't work when the "
                            "objects are different Series has attrs"
                        )
                    )

    # 如果 annotate 为 "left" 或 "both" 并且 left 不是整数
    if annotate in {"left", "both"} and not isinstance(left, int):
        # 给 left 对象添加 attrs 属性，赋值为 {"a": 1}
        left.attrs = {"a": 1}
    
    # 如果 annotate 为 "right" 或 "both" 并且 right 不是整数
    if annotate in {"right", "both"} and not isinstance(right, int):
        # 给 right 对象添加 attrs 属性，赋值为 {"a": 1}
        right.attrs = {"a": 1}

    # 检查 all_binary_operators 是否是以下比较操作符之一
    is_cmp = all_binary_operators in [
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
    ]
    # 如果 is_cmp 为 True 并且 left 是 pandas DataFrame 对象，right 是 pandas Series 对象
    if is_cmp and isinstance(left, pd.DataFrame) and isinstance(right, pd.Series):
        # 在 pandas 2.0 中移除了比较时的静默对齐操作，参考 GitHub 问题 #28759
        # 对 left 和 right 进行列对齐，axis=1 表示按列对齐
        left, right = left.align(right, axis=1)
    # 如果 is_cmp 为真，并且 left 是 pd.Series 类型，right 是 pd.DataFrame 类型，则执行对齐操作
    elif is_cmp and isinstance(left, pd.Series) and isinstance(right, pd.DataFrame):
        # 将 right 和 left 按照列对齐
        right, left = right.align(left, axis=1)

    # 对 left 和 right 执行所有的二元操作，并将结果存储在 result 中
    result = all_binary_operators(left, right)
    # 断言结果对象 result 的 attrs 属性等于 {"a": 1}，否则会引发 AssertionError
    assert result.attrs == {"a": 1}
# ----------------------------------------------------------------------------
# Accessors

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，用于测试字符串方法
@pytest.mark.parametrize(
    "method",
    [
        # 使用 operator.methodcaller 创建字符串方法调用器，对字符串进行 capitalize 操作
        operator.methodcaller("capitalize"),
        # 对字符串进行 casefold 操作，将字符串转换为小写并处理大小写折叠
        operator.methodcaller("casefold"),
        # 调用 cat 方法将字符串连接字符 'a'，类似于字符串拼接操作
        operator.methodcaller("cat", ["a"]),
        # 检查字符串中是否包含字符 'a'
        operator.methodcaller("contains", "a"),
        # 统计字符串中字符 'a' 的出现次数
        operator.methodcaller("count", "a"),
        # 使用 utf-8 编码对字符串进行编码
        operator.methodcaller("encode", "utf-8"),
        # 检查字符串是否以字符 'a' 结尾
        operator.methodcaller("endswith", "a"),
        # 从字符串中提取匹配正则表达式 r"(\w)(\d)" 的部分
        operator.methodcaller("extract", r"(\w)(\d)"),
        # 从字符串中提取匹配正则表达式 r"(\w)(\d)" 的部分，不进行扩展
        operator.methodcaller("extract", r"(\w)(\d)", expand=False),
        # 查找字符 'a' 在字符串中的位置
        operator.methodcaller("find", "a"),
        # 查找字符串中所有出现的字符 'a'，返回列表
        operator.methodcaller("findall", "a"),
        # 获取字符串的索引为 0 的字符
        operator.methodcaller("get", 0),
        # 查找字符串中字符 'a' 的索引位置
        operator.methodcaller("index", "a"),
        # 返回字符串的长度
        operator.methodcaller("len"),
        # 将字符串左对齐，并使用空格填充到长度 4
        operator.methodcaller("ljust", 4),
        # 将字符串转换为小写
        operator.methodcaller("lower"),
        # 去除字符串左侧的空白字符
        operator.methodcaller("lstrip"),
        # 对字符串进行匹配，使用正则表达式 r"\w"
        operator.methodcaller("match", r"\w"),
        # 标准化字符串，使用 NFC 规范
        operator.methodcaller("normalize", "NFC"),
        # 在字符串的左侧填充空格，使其长度为 4
        operator.methodcaller("pad", 4),
        # 将字符串以字符 'a' 分割成三部分
        operator.methodcaller("partition", "a"),
        # 将字符串重复两次
        operator.methodcaller("repeat", 2),
        # 将字符串中的字符 'a' 替换为字符 'b'
        operator.methodcaller("replace", "a", "b"),
        # 从右侧查找字符 'a' 在字符串中的位置
        operator.methodcaller("rfind", "a"),
        # 从右侧查找字符 'a' 在字符串中的索引位置
        operator.methodcaller("rindex", "a"),
        # 将字符串右对齐，并使用空格填充到长度 4
        operator.methodcaller("rjust", 4),
        # 将字符串以字符 'a' 分割成三部分，从右侧开始
        operator.methodcaller("rpartition", "a"),
        # 去除字符串右侧的空白字符
        operator.methodcaller("rstrip"),
        # 返回字符串的切片，从索引 4 开始
        operator.methodcaller("slice", 4),
        # 将字符串的第一个字符替换为 'a'
        operator.methodcaller("slice_replace", 1, repl="a"),
        # 检查字符串是否以字符 'a' 开头
        operator.methodcaller("startswith", "a"),
        # 去除字符串两侧的空白字符
        operator.methodcaller("strip"),
        # 将字符串中的大小写互换
        operator.methodcaller("swapcase"),
        # 使用字典 {"a": "b"} 对字符串进行翻译
        operator.methodcaller("translate", {"a": "b"}),
        # 将字符串转换为大写
        operator.methodcaller("upper"),
        # 将字符串按长度 4 进行换行
        operator.methodcaller("wrap", 4),
        # 在字符串的左侧用零填充到长度 4
        operator.methodcaller("zfill", 4),
        # 检查字符串是否为字母数字组合
        operator.methodcaller("isalnum"),
        # 检查字符串是否全为字母
        operator.methodcaller("isalpha"),
        # 检查字符串是否全为数字
        operator.methodcaller("isdigit"),
        # 检查字符串是否全为空白字符
        operator.methodcaller("isspace"),
        # 检查字符串是否全为小写字母
        operator.methodcaller("islower"),
        # 检查字符串是否全为大写字母
        operator.methodcaller("isupper"),
        # 检查字符串是否符合标题格式
        operator.methodcaller("istitle"),
        # 检查字符串是否全为数字字符
        operator.methodcaller("isnumeric"),
        # 检查字符串是否全为十进制数字
        operator.methodcaller("isdecimal"),
        # 对字符串进行独热编码，生成哑变量
        operator.methodcaller("get_dummies"),
    ],
    ids=idfn,
)
# 定义测试函数 test_string_method，对参数中的字符串方法进行测试
def test_string_method(method):
    # 创建包含字符串 "a1" 的 pandas Series 对象 s
    s = pd.Series(["a1"])
    # 为 Series 对象设置属性 attrs，赋值为 {"a": 1}
    s.attrs = {"a": 1}
    # 调用指定的字符串方法 method 对 Series 对象中的字符串进行操作
    result = method(s.str)
    # 断言操作结果的 attrs 属性与预期的 {"a": 1} 相等
    assert result.attrs == {"a": 1}


# 使用 pytest.mark.parametrize 装饰器定义参数化测试，用于测试日期时间方法
@pytest.mark.parametrize(
    "method",
    [
        # 使用 operator.methodcaller 创建日期时间方法调用器，转换为 Period
        operator.methodcaller("to_period"),
        # 将日期时间转换为指定时区 "CET" 的本地时间
        operator.methodcaller("tz_localize", "CET"),
        # 标准化日期时间
        operator.methodcaller("normalize"),
        # 将日期时间格式化为年份 "%Y"
        operator.methodcaller("strftime", "%Y"),
        # 对日期时间进行舍入，单位为小时 "h"
        operator.methodcaller("round", "h"),
        # 对日期时间进行向下取整，单位为小时 "h"
        operator.methodcaller("floor", "h"),
        # 对日期时间进行向上取整，单位为小时 "h"
        operator.methodcaller("ceil", "h"),
        # 获取月份名称
        operator.methodcaller("month_name"),
        # 获取星期几的名称
        operator.methodcaller("day_name"),
    ],
    ids=idfn,
)
# 定义测试函数 test_datetime_method，对参数中的日期时间方法进行测试
def test_datetime_method(method):
    # 创建一个 Pandas Series 对象，其中包含从 "2000" 年开始的四个日期时间戳
    s = pd.Series(pd.date_range("2000", periods=4))
    
    # 为 Pandas Series 对象 s 添加属性，这里设置属性字典为 {"a": 1}
    s.attrs = {"a": 1}
    
    # 调用对象 s 的 dt 属性，假设这里 method 是一个可以接受 Pandas Series.dt 的方法
    result = method(s.dt)
    
    # 断言语句，验证 result 对象的 attrs 属性是否等于 {"a": 1}
    assert result.attrs == {"a": 1}
@pytest.mark.parametrize(
    "attr",
    [
        "date",  # 日期属性
        "time",  # 时间属性
        "timetz",  # 带时区的时间属性
        "year",  # 年份属性
        "month",  # 月份属性
        "day",  # 天属性
        "hour",  # 小时属性
        "minute",  # 分钟属性
        "second",  # 秒属性
        "microsecond",  # 微秒属性
        "nanosecond",  # 纳秒属性
        "dayofweek",  # 星期几属性
        "day_of_week",  # 星期几属性（别名）
        "dayofyear",  # 年内的第几天属性
        "day_of_year",  # 年内的第几天属性（别名）
        "quarter",  # 季度属性
        "is_month_start",  # 是否为月初属性
        "is_month_end",  # 是否为月末属性
        "is_quarter_start",  # 是否为季度初属性
        "is_quarter_end",  # 是否为季度末属性
        "is_year_start",  # 是否为年初属性
        "is_year_end",  # 是否为年末属性
        "is_leap_year",  # 是否为闰年属性
        "daysinmonth",  # 当月天数属性
        "days_in_month",  # 当月天数属性（别名）
    ],
)
def test_datetime_property(attr):
    # 创建一个包含日期范围的 Series 对象
    s = pd.Series(pd.date_range("2000", periods=4))
    # 设置自定义属性
    s.attrs = {"a": 1}
    # 获取指定属性的结果
    result = getattr(s.dt, attr)
    # 断言结果的自定义属性为预期值
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "attr", ["days", "seconds", "microseconds", "nanoseconds", "components"]
)
def test_timedelta_property(attr):
    # 创建一个包含时间差范围的 Series 对象
    s = pd.Series(pd.timedelta_range("2000", periods=4))
    # 设置自定义属性
    s.attrs = {"a": 1}
    # 获取指定属性的结果
    result = getattr(s.dt, attr)
    # 断言结果的自定义属性为预期值
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize("method", [operator.methodcaller("total_seconds")])
def test_timedelta_methods(method):
    # 创建一个包含时间差范围的 Series 对象
    s = pd.Series(pd.timedelta_range("2000", periods=4))
    # 设置自定义属性
    s.attrs = {"a": 1}
    # 执行指定的方法
    result = method(s.dt)
    # 断言结果的自定义属性为预期值
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("add_categories", ["c"]),  # 添加分类
        operator.methodcaller("as_ordered"),  # 转换为有序分类
        operator.methodcaller("as_unordered"),  # 转换为无序分类
        lambda x: getattr(x, "codes"),  # 获取分类的编码
        operator.methodcaller("remove_categories", "a"),  # 移除指定的分类
        operator.methodcaller("remove_unused_categories"),  # 移除未使用的分类
        operator.methodcaller("rename_categories", {"a": "A", "b": "B"}),  # 重命名分类
        operator.methodcaller("reorder_categories", ["b", "a"]),  # 重新排序分类
        operator.methodcaller("set_categories", ["A", "B"]),  # 设置分类
    ],
)
@not_implemented_mark
def test_categorical_accessor(method):
    # 创建一个包含分类数据的 Series 对象
    s = pd.Series(["a", "b"], dtype="category")
    # 设置自定义属性
    s.attrs = {"a": 1}
    # 执行指定的方法
    result = method(s.cat)
    # 断言结果的自定义属性为预期值
    assert result.attrs == {"a": 1}


# ----------------------------------------------------------------------------
# Groupby


@pytest.mark.parametrize(
    "obj", [pd.Series([0, 0]), pd.DataFrame({"A": [0, 1], "B": [1, 2]})]
)
@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("sum"),  # 求和操作
        lambda x: x.apply(lambda y: y),  # 应用函数
        lambda x: x.agg("sum"),  # 聚合求和
        lambda x: x.agg("mean"),  # 聚合求均值
        lambda x: x.agg("median"),  # 聚合求中位数
    ],
)
def test_groupby_finalize(obj, method):
    # 设置自定义属性
    obj.attrs = {"a": 1}
    # 执行指定的方法，对数据进行聚合操作
    result = method(obj.groupby([0, 0], group_keys=False))
    # 断言结果的自定义属性为预期值
    assert result.attrs == {"a": 1}
    [
        # 对输入的数据框进行聚合操作，计算列的总和和非缺失值的数量
        lambda x: x.agg(["sum", "count"]),
        # 对输入的数据框进行聚合操作，计算列的标准差
        lambda x: x.agg("std"),
        # 对输入的数据框进行聚合操作，计算列的方差
        lambda x: x.agg("var"),
        # 对输入的数据框进行聚合操作，计算列的标准误差
        lambda x: x.agg("sem"),
        # 对输入的数据框进行聚合操作，计算分组后每组的大小
        lambda x: x.agg("size"),
        # 对输入的数据框进行聚合操作，计算每列的开盘、最高、最低和收盘价
        lambda x: x.agg("ohlc"),
    ],
# 在这个函数定义前添加一个标记，表示它还没有被实现
@not_implemented_mark
def test_groupby_finalize_not_implemented(obj, method):
    # 给对象设置属性字典，包含键'a'和对应值1
    obj.attrs = {"a": 1}
    # 调用方法处理传入的对象，使用groupby方法传入参数[0, 0]
    result = method(obj.groupby([0, 0]))
    # 断言结果对象的属性字典应与原始设置一致
    assert result.attrs == {"a": 1}

def test_finalize_frame_series_name():
    # 添加注释，指向 GitHub 上的相关讨论或者修复
    # https://github.com/pandas-dev/pandas/pull/37186/files#r506978889
    # 确保不复制DataFrame列的名称到Series对象中。
    # 创建一个DataFrame，包含一个名为'name'的列，各自包含1和2的值。
    df = pd.DataFrame({"name": [1, 2]})
    # 对Series对象调用__finalize__方法，传入DataFrame对象df作为参数。
    result = pd.Series([1, 2]).__finalize__(df)
    # 断言结果Series对象的名称应该是None
    assert result.name is None
```