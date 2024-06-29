# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_indexing.py`

```
# Test GroupBy._positional_selector positional grouped indexing GH#42864

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，用于数据分析和操作
import pandas._testing as tm  # 导入 Pandas 内部测试模块

# 使用 pytest.mark.parametrize 装饰器定义多组参数化测试数据
@pytest.mark.parametrize(
    "arg, expected_rows",  # 参数名称和预期结果
    [
        [0, [0, 1, 4]],    # 测试参数为单个整数 0
        [2, [5]],          # 测试参数为单个整数 2
        [5, []],           # 测试参数为单个整数 5
        [-1, [3, 4, 7]],   # 测试参数为单个负整数 -1
        [-2, [1, 6]],      # 测试参数为单个负整数 -2
        [-6, []],          # 测试参数为单个负整数 -6
    ],
)
def test_int(slice_test_df, slice_test_grouped, arg, expected_rows):
    # 测试单个整数参数的情况
    result = slice_test_grouped._positional_selector[arg]
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)


def test_slice(slice_test_df, slice_test_grouped):
    # 测试单个切片参数的情况
    result = slice_test_grouped._positional_selector[0:3:2]
    expected = slice_test_df.iloc[[0, 1, 4, 5]]

    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器定义多组参数化测试数据
@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [[0, 2], [0, 1, 4, 5]],               # 测试参数为整数列表
        [[0, 2, -1], [0, 1, 3, 4, 5, 7]],     # 测试参数为包含负整数的整数列表
        [range(0, 3, 2), [0, 1, 4, 5]],       # 测试参数为 range 对象
        [{0, 2}, [0, 1, 4, 5]],               # 测试参数为集合对象
    ],
    ids=[
        "list",         # 测试参数为列表的情况
        "negative",     # 测试参数为负数的情况
        "range",        # 测试参数为 range 对象的情况
        "set",          # 测试参数为集合的情况
    ],
)
def test_list(slice_test_df, slice_test_grouped, arg, expected_rows):
    # 测试整数列表和整数值可迭代对象的情况
    result = slice_test_grouped._positional_selector[arg]
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)


def test_ints(slice_test_df, slice_test_grouped):
    # 测试整数元组的情况
    result = slice_test_grouped._positional_selector[0, 2, -1]
    expected = slice_test_df.iloc[[0, 1, 3, 4, 5, 7]]

    tm.assert_frame_equal(result, expected)


def test_slices(slice_test_df, slice_test_grouped):
    # 测试切片元组的情况
    result = slice_test_grouped._positional_selector[:2, -2:]
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4, 6, 7]]

    tm.assert_frame_equal(result, expected)


def test_mix(slice_test_df, slice_test_grouped):
    # 测试整数和切片混合的元组情况
    result = slice_test_grouped._positional_selector[0, 1, -2:]
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4, 6, 7]]

    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器定义多组参数化测试数据
@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [0, [0, 1, 4]],                               # 测试参数为单个整数 0
        [[0, 2, -1], [0, 1, 3, 4, 5, 7]],              # 测试参数为整数列表
        [(slice(None, 2), slice(-2, None)), [0, 1, 2, 3, 4, 6, 7]],  # 测试参数为切片元组
    ],
)
def test_as_index(slice_test_df, arg, expected_rows):
    # 测试默认的 as_index 行为
    result = slice_test_df.groupby("Group", sort=False)._positional_selector[arg]
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)


def test_doc_examples():
    # 测试文档中的示例
    df = pd.DataFrame(
        [["a", 1], ["a", 2], ["a", 3], ["b", 4], ["b", 5]], columns=["A", "B"]
    )

    grouped = df.groupby("A", as_index=False)

    result = grouped._positional_selector[1:2]
    expected = pd.DataFrame([["a", 2], ["b", 5]], columns=["A", "B"], index=[1, 4])
    # 比较两个 Pandas 数据帧 `result` 和 `expected` 是否相等
    tm.assert_frame_equal(result, expected)
    
    # 从 `grouped._positional_selector` 中选择位置为 (1, -1) 的元素赋值给 `result`
    result = grouped._positional_selector[1, -1]
    
    # 创建一个预期的 Pandas 数据帧 `expected`，包含指定数据、列和索引
    expected = pd.DataFrame(
        [["a", 2], ["a", 3], ["b", 5]], columns=["A", "B"], index=[1, 2, 4]
    )
    
    # 比较 `result` 和 `expected` 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个名为 test_multiindex 的测试函数，用于测试多索引在文档中提到的使用案例
def test_multiindex():
    # 定义一个内部函数 _make_df_from_data，根据给定数据生成 DataFrame 对象
    def _make_df_from_data(data):
        # 初始化一个空字典 rows 用于存储行数据
        rows = {}
        # 遍历日期数据
        for date in data:
            # 遍历每个日期下的数据级别
            for level in data[date]:
                # 将每个级别的数据组成字典并加入到 rows 中，使用 (date, level[0]) 作为索引
                rows[(date, level[0])] = {"A": level[1], "B": level[2]}
        
        # 使用 from_dict 方法将字典数据转换为 DataFrame 对象，索引方式为 "index"
        df = pd.DataFrame.from_dict(rows, orient="index")
        # 设置 DataFrame 的索引名称为 ("Date", "Item")
        df.index.names = ("Date", "Item")
        return df

    # 使用默认的随机数生成器创建 rng 对象
    rng = np.random.default_rng(2)
    # 设定日期数为 100
    ndates = 100
    # 设定条目数为 20
    nitems = 20
    # 创建日期范围，从 "20130101" 开始，周期为 ndates 天，频率为每天 ("D")
    dates = pd.date_range("20130101", periods=ndates, freq="D")
    # 创建条目列表，命名格式为 "item i"，i 从 0 到 nitems-1
    items = [f"item {i}" for i in range(nitems)]

    # 初始化一个空字典 multiindex_data 用于存储多索引数据
    multiindex_data = {}
    # 遍历日期列表
    for date in dates:
        # 计算每个日期的条目数，范围为 nitems 到 nitems-12 之间的随机整数
        nitems_for_date = nitems - rng.integers(0, 12)
        # 针对每个条目生成随机数据，并按照第二个元素排序
        levels = [
            (item, rng.integers(0, 10000) / 100, rng.integers(0, 10000) / 100)
            for item in items[:nitems_for_date]
        ]
        levels.sort(key=lambda x: x[1])
        # 将排序后的数据加入到 multiindex_data 字典中
        multiindex_data[date] = levels

    # 调用 _make_df_from_data 函数，根据 multiindex_data 创建 DataFrame 对象 df
    df = _make_df_from_data(multiindex_data)
    # 对 df 进行分组，并选择每组中第三个到倒数第三个元素
    result = df.groupby("Date", as_index=False).nth(slice(3, -3))

    # 创建一个切片字典 sliced，仅包含每个日期的第三个到倒数第三个元素
    sliced = {date: values[3:-3] for date, values in multiindex_data.items()}
    # 使用 _make_df_from_data 函数创建期望的 DataFrame 对象 expected
    expected = _make_df_from_data(sliced)

    # 使用 pytest 模块的 assert_frame_equal 函数比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器为 test_against_head_and_tail 函数添加参数化测试
@pytest.mark.parametrize("arg", [1, 5, 30, 1000, -1, -5, -30, -1000])
@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("simulated", [True, False])
# 定义测试函数 test_against_head_and_tail，用于比较测试结果与分组头尾结果的一致性
def test_against_head_and_tail(arg, method, simulated):
    # 设定分组数为 100
    n_groups = 100
    # 每组的行数为 30
    n_rows_per_group = 30

    # 创建数据字典 data，包含 "group" 和 "value" 两列数据
    data = {
        "group": [
            f"group {g}" for j in range(n_rows_per_group) for g in range(n_groups)
        ],
        "value": [
            f"group {g} row {j}"
            for j in range(n_rows_per_group)
            for g in range(n_groups)
        ],
    }
    # 使用数据字典创建 DataFrame 对象 df
    df = pd.DataFrame(data)
    # 根据 "group" 列进行分组，as_index=False 表示不保留索引
    grouped = df.groupby("group", as_index=False)
    # 根据参数 arg 确定选择的行数，若 arg >= 0 则选择前 arg 行，否则选择后 -arg 行
    size = arg if arg >= 0 else n_rows_per_group + arg

    # 根据 method 参数选择是使用头部还是尾部的数据
    if method == "head":
        # 选择分组的前 arg 行数据
        result = grouped._positional_selector[:arg]

        # 如果 simulated 为 True，则按照模拟的方式生成预期结果 expected
        if simulated:
            # 计算选择的行索引列表 indices
            indices = [
                j * n_groups + i
                for j in range(size)
                for i in range(n_groups)
                if j * n_groups + i < n_groups * n_rows_per_group
            ]
            # 根据索引列表生成预期的 DataFrame 对象 expected
            expected = df.iloc[indices]

        else:
            # 否则使用 pandas 的 head 方法生成预期结果 expected
            expected = grouped.head(arg)

    else:
        # 选择分组的后 arg 行数据
        result = grouped._positional_selector[-arg:]

        # 如果 simulated 为 True，则按照模拟的方式生成预期结果 expected
        if simulated:
            # 计算选择的行索引列表 indices
            indices = [
                (n_rows_per_group + j - size) * n_groups + i
                for j in range(size)
                for i in range(n_groups)
                if (n_rows_per_group + j - size) * n_groups + i >= 0
            ]
            # 根据索引列表生成预期的 DataFrame 对象 expected
            expected = df.iloc[indices]

        else:
            # 否则使用 pandas 的 tail 方法生成预期结果 expected
            expected = grouped.tail(arg)

    # 使用 pytest 模块的 assert_frame_equal 函数比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 为 start 和 stop 参数添加参数化测试
@pytest.mark.parametrize("start", [None, 0, 1, 10, -1, -10])
@pytest.mark.parametrize("stop", [None, 0, 1, 10, -1, -10])
@pytest.mark.parametrize("step", [None, 1, 5])
# 使用 pytest 的 parametrize 装饰器，对 test_against_df_iloc 函数进行多组参数化测试
def test_against_df_iloc(start, stop, step):
    # 测试单个分组是否与 DataFrame.iloc 方法结果一致
    n_rows = 30

    data = {
        "group": ["group 0"] * n_rows,
        "value": list(range(n_rows)),
    }
    # 创建 DataFrame 对象
    df = pd.DataFrame(data)
    # 按照 "group" 列进行分组
    grouped = df.groupby("group", as_index=False)

    # 使用 grouped 对象的 _positional_selector 属性进行切片操作
    result = grouped._positional_selector[start:stop:step]
    # 期望的结果由 DataFrame 的 iloc 方法得出
    expected = df.iloc[start:stop:step]

    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


def test_series():
    # 测试分组后的 Series 对象
    ser = pd.Series([1, 2, 3, 4, 5], index=["a", "a", "a", "b", "b"])
    # 按照索引的第一级别进行分组
    grouped = ser.groupby(level=0)
    # 使用 grouped 对象的 _positional_selector 属性进行切片操作
    result = grouped._positional_selector[1:2]
    # 期望的结果由具体切片操作得出的 Series 对象
    expected = pd.Series([2, 5], index=["a", "b"])

    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
# 使用 pytest 的 parametrize 装饰器，对 test_step 函数进行多组参数化测试
def test_step(step):
    # 测试不同步长的切片操作
    data = [["x", f"x{i}"] for i in range(5)]
    data += [["y", f"y{i}"] for i in range(4)]
    data += [["z", f"z{i}"] for i in range(3)]
    # 创建 DataFrame 对象
    df = pd.DataFrame(data, columns=["A", "B"])

    # 按照 "A" 列进行分组
    grouped = df.groupby("A", as_index=False)

    # 使用 grouped 对象的 _positional_selector 属性进行切片操作
    result = grouped._positional_selector[::step]

    # 期望的结果由具体切片操作得出的 DataFrame 对象
    data = [["x", f"x{i}"] for i in range(0, 5, step)]
    data += [["y", f"y{i}"] for i in range(0, 4, step)]
    data += [["z", f"z{i}"] for i in range(0, 3, step)]

    index = [0 + i for i in range(0, 5, step)]
    index += [5 + i for i in range(0, 4, step)]
    index += [9 + i for i in range(0, 3, step)]

    expected = pd.DataFrame(data, columns=["A", "B"], index=index)

    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


def test_columns_on_iter():
    # GitHub 问题编号 #44821
    df = pd.DataFrame({k: range(10) for k in "ABC"})

    # 按照 "A" 列进行分组，并选择特定的列
    cols = ["A", "B"]
    for _, dg in df.groupby(df.A < 4)[cols]:
        # 断言分组后的 DataFrame 的列与预期的列相等
        tm.assert_index_equal(dg.columns, pd.Index(cols))
        # 确保 "C" 列不在分组后的 DataFrame 的列中
        assert "C" not in dg.columns


@pytest.mark.parametrize("func", [list, pd.Index, pd.Series, np.array])
# 使用 pytest 的 parametrize 装饰器，对 test_groupby_duplicated_columns 函数进行多组参数化测试
def test_groupby_duplicated_columns(func):
    # GitHub 问题编号 #44924
    df = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 3],
            "C": ["G", "G"],
        }
    )
    # 按照 "C" 列进行分组，并选择特定的列
    result = df.groupby("C")[func(["A", "B", "A"])].mean()

    expected = pd.DataFrame(
        [[1.5, 3.0, 1.5]], columns=["A", "B", "A"], index=pd.Index(["G"], name="C")
    )

    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


def test_groupby_get_nonexisting_groups():
    # GitHub 问题编号 #32492
    df = pd.DataFrame(
        data={
            "A": ["a1", "a2", None],
            "B": ["b1", "b2", "b1"],
            "val": [1, 2, 3],
        }
    )
    # 按照 ["A", "B"] 多列进行分组
    grps = df.groupby(by=["A", "B"])

    # 使用 get_group 方法获取不存在的组合
    msg = "('a2', 'b1')"
    with pytest.raises(KeyError, match=msg):
        grps.get_group(("a2", "b1"))
```