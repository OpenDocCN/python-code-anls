# `D:\src\scipysrc\pandas\pandas\tests\generic\test_label_or_level_utils.py`

```
# 导入pytest库，用于编写和运行测试
import pytest

# 从pandas库中导入array_equivalent函数
from pandas.core.dtypes.missing import array_equivalent

# 从pandas库中导入并命名为pd，用于数据处理和分析
import pandas as pd


# Fixtures（测试夹具）
# ========

# 创建一个返回具有列'L1'、'L2'和'L3'的DataFrame的测试夹具
@pytest.fixture
def df():
    """DataFrame with columns 'L1', 'L2', and 'L3'"""
    return pd.DataFrame({"L1": [1, 2, 3], "L2": [11, 12, 13], "L3": ["A", "B", "C"]})


# 创建一个带有列或索引级别'L1'、'L2'和'L3'的DataFrame的测试夹具，根据传入的参数动态设置索引级别
@pytest.fixture(params=[[], ["L1"], ["L1", "L2"], ["L1", "L2", "L3"]])
def df_levels(request, df):
    """DataFrame with columns or index levels 'L1', 'L2', and 'L3'"""
    levels = request.param

    # 如果levels不为空，则将DataFrame按levels设置为索引
    if levels:
        df = df.set_index(levels)

    return df


# 创建一个具有级别'L1'和'L2'以及标签'L1'和'L3'的DataFrame的测试夹具
@pytest.fixture
def df_ambig(df):
    """DataFrame with levels 'L1' and 'L2' and labels 'L1' and 'L3'"""
    df = df.set_index(["L1", "L2"])

    # 将'L1'列的值设置为'L3'列的值
    df["L1"] = df["L3"]

    return df


# Test is label/level reference
# =============================

# 获取DataFrame索引级别和列标签的期望值
def get_labels_levels(df_levels):
    expected_labels = list(df_levels.columns)
    expected_levels = [name for name in df_levels.index.names if name is not None]
    return expected_labels, expected_levels


# 断言标签引用的辅助函数
def assert_label_reference(frame, labels, axis):
    for label in labels:
        assert frame._is_label_reference(label, axis=axis)
        assert not frame._is_level_reference(label, axis=axis)
        assert frame._is_label_or_level_reference(label, axis=axis)


# 断言级别引用的辅助函数
def assert_level_reference(frame, levels, axis):
    for level in levels:
        assert frame._is_level_reference(level, axis=axis)
        assert not frame._is_label_reference(level, axis=axis)
        assert frame._is_label_or_level_reference(level, axis=axis)


# DataFrame
# ---------

# 测试DataFrame在简单情况下是否是级别或标签引用
def test_is_level_or_label_reference_df_simple(df_levels, axis):
    axis = df_levels._get_axis_number(axis)
    # 计算预期的标签和级别
    expected_labels, expected_levels = get_labels_levels(df_levels)

    # 如果axis为1，则转置DataFrame
    if axis == 1:
        df_levels = df_levels.T

    # 执行断言检查
    assert_level_reference(df_levels, expected_levels, axis=axis)
    assert_label_reference(df_levels, expected_labels, axis=axis)


# 测试DataFrame在有歧义的情况下是否是级别引用
def test_is_level_reference_df_ambig(df_ambig, axis):
    axis = df_ambig._get_axis_number(axis)

    # 如果axis为1，则转置DataFrame
    if axis == 1:
        df_ambig = df_ambig.T

    # DataFrame同时具有在轴上的级别和命名为'L1'的标签，因此'L1'应引用标签而不是级别
    assert_label_reference(df_ambig, ["L1"], axis=axis)

    # DataFrame在轴上具有名为'L2'的级别且不含歧义，因此'L2'是级别引用
    assert_level_reference(df_ambig, ["L2"], axis=axis)

    # DataFrame具有名为'L3'的列且不是级别引用
    assert_label_reference(df_ambig, ["L3"], axis=axis)


# Series
# ------

# 测试Series在简单情况下是否是级别引用
def test_is_level_reference_series_simple_axis0(df):
    # 使用'L1'作为索引创建Series
    s = df.set_index("L1").L2
    assert_level_reference(s, ["L1"], axis=0)
    assert not s._is_level_reference("L2")

    # 使用'L1'和'L2'作为索引创建Series
    s = df.set_index(["L1", "L2"]).L3
    # 对 Series 对象 s 进行多级索引的级别验证，期望索引包含 "L1" 和 "L2" 两个级别，沿着轴 0 进行检查
    assert_level_reference(s, ["L1", "L2"], axis=0)
    
    # 检查 Series 对象 s 是否不包含名为 "L3" 的级别引用，返回布尔结果
    assert not s._is_level_reference("L3")
# 定义测试函数，用于测试 Series 对象是否正确处理引用级别时的错误情况
def test_is_level_reference_series_axis1_error(df):
    # 将 DataFrame 按照 L1 列设置为索引，并取其 L2 列作为 Series 对象 s
    s = df.set_index("L1").L2

    # 使用 pytest 检查是否抛出 ValueError 异常，并验证异常消息中是否包含 "No axis named 1"
    with pytest.raises(ValueError, match="No axis named 1"):
        # 在 axis=1 的情况下尝试检查引用级别，预期会抛出异常
        s._is_level_reference("L1", axis=1)


# Test _check_label_or_level_ambiguity_df
# =======================================


# DataFrame
# ---------
# 测试 DataFrame 对象在检查标签或级别模糊性时的行为
def test_check_label_or_level_ambiguity_df(df_ambig, axis):
    # 获取轴的编号
    axis = df_ambig._get_axis_number(axis)
    
    # 如果 axis == 1，则转置 DataFrame
    if axis == 1:
        df_ambig = df_ambig.T
        msg = "'L1' is both a column level and an index label"
    else:
        msg = "'L1' is both an index level and a column label"
    
    # 使用 pytest 检查是否抛出 ValueError 异常，并验证异常消息是否与预期的 msg 匹配
    with pytest.raises(ValueError, match=msg):
        # 检查 DataFrame df_ambig 中标签或级别 'L1' 是否模糊
        df_ambig._check_label_or_level_ambiguity("L1", axis=axis)

    # 检查 DataFrame df_ambig 中级别 'L2' 是否模糊，预期不会抛出异常
    df_ambig._check_label_or_level_ambiguity("L2", axis=axis)

    # 检查 DataFrame df_ambig 中标签 'L3' 是否模糊，预期不会抛出异常
    assert not df_ambig._check_label_or_level_ambiguity("L3", axis=axis)


# Series
# ------
# 测试 Series 对象在检查标签或级别模糊性时的行为
def test_check_label_or_level_ambiguity_series(df):
    # 由于 Series 没有列，因此引用永远不会模糊

    # 将 DataFrame 按照 L1 列设置为索引，并取其 L2 列作为 Series 对象 s
    s = df.set_index("L1").L2

    # 检查 Series 对象 s 中标签或级别 'L1' 是否模糊，预期不会抛出异常
    s._check_label_or_level_ambiguity("L1", axis=0)
    
    # 检查 Series 对象 s 中标签或级别 'L2' 是否模糊，预期不会抛出异常
    s._check_label_or_level_ambiguity("L2", axis=0)

    # 将 DataFrame 按照 ['L1', 'L2'] 列设置为索引，并取其 L3 列作为 Series 对象 s
    s = df.set_index(["L1", "L2"]).L3

    # 检查 Series 对象 s 中标签或级别 'L1' 是否模糊，预期不会抛出异常
    s._check_label_or_level_ambiguity("L1", axis=0)
    
    # 检查 Series 对象 s 中标签或级别 'L2' 是否模糊，预期不会抛出异常
    s._check_label_or_level_ambiguity("L2", axis=0)
    
    # 检查 Series 对象 s 中标签或级别 'L3' 是否模糊，预期不会抛出异常
    s._check_label_or_level_ambiguity("L3", axis=0)


def test_check_label_or_level_ambiguity_series_axis1_error(df):
    # 将 DataFrame 按照 L1 列设置为索引，并取其 L2 列作为 Series 对象 s
    s = df.set_index("L1").L2

    # 使用 pytest 检查是否抛出 ValueError 异常，并验证异常消息中是否包含 "No axis named 1"
    with pytest.raises(ValueError, match="No axis named 1"):
        # 在 axis=1 的情况下尝试检查标签或级别模糊性，预期会抛出异常
        s._check_label_or_level_ambiguity("L1", axis=1)


# Test _get_label_or_level_values
# ===============================
# 定义用于断言 DataFrame 中标签值或级别值的辅助函数
def assert_label_values(frame, labels, axis):
    # 获取轴的编号
    axis = frame._get_axis_number(axis)
    
    # 遍历每个标签
    for label in labels:
        # 如果轴是 0，则获取标签 label 对应的值
        if axis == 0:
            expected = frame[label]._values
        else:
            expected = frame.loc[label]._values
        
        # 获取 DataFrame 中标签或级别 label 的值
        result = frame._get_label_or_level_values(label, axis=axis)
        # 断言预期值与实际值是否相等
        assert array_equivalent(expected, result)


# 定义用于断言 DataFrame 中级别值的辅助函数
def assert_level_values(frame, levels, axis):
    # 获取轴的编号
    axis = frame._get_axis_number(axis)
    
    # 遍历每个级别
    for level in levels:
        # 如果轴是 0，则获取级别 level 对应的索引级别值
        if axis == 0:
            expected = frame.index.get_level_values(level=level)._values
        else:
            expected = frame.columns.get_level_values(level=level)._values
        
        # 获取 DataFrame 中级别 level 的值
        result = frame._get_label_or_level_values(level, axis=axis)
        # 断言预期值与实际值是否相等
        assert array_equivalent(expected, result)


# DataFrame
# ---------
# 测试 DataFrame 对象在获取简单标签或级别值时的行为
def test_get_label_or_level_values_df_simple(df_levels, axis):
    # 计算预期的标签和级别
    expected_labels, expected_levels = get_labels_levels(df_levels)
    # 获取指定轴(axis)的编号
    axis = df_levels._get_axis_number(axis)

    # 如果 axis 为 1，则转置数据框(df_levels)
    if axis == 1:
        df_levels = df_levels.T

    # 对数据框(df_levels)进行标签值的断言检查，确保符合预期的标签值
    assert_label_values(df_levels, expected_labels, axis=axis)

    # 对数据框(df_levels)进行层级值的断言检查，确保符合预期的层级值
    assert_level_values(df_levels, expected_levels, axis=axis)
# 定义函数，用于测试处理包含模糊标签或级别的 DataFrame
def test_get_label_or_level_values_df_ambig(df_ambig, axis):
    # 获取指定轴的编号
    axis = df_ambig._get_axis_number(axis)
    # 如果 axis == 1，则转置数据框
    if axis == 1:
        df_ambig = df_ambig.T

    # 断言数据框具有名为 "L2" 的轴上级别，并且该级别不模糊
    assert_level_values(df_ambig, ["L2"], axis=axis)

    # 断言数据框具有名为 "L3" 的轴外标签，并且该标签不模糊
    assert_label_values(df_ambig, ["L3"], axis=axis)


# 定义函数，用于测试处理包含重复标签的 DataFrame
def test_get_label_or_level_values_df_duplabels(df, axis):
    # 将数据框设置以 "L1" 作为索引
    df = df.set_index(["L1"])
    # 合并 df 和 df["L2"] 列，生成包含重复标签的数据框
    df_duplabels = pd.concat([df, df["L2"]], axis=1)
    # 获取指定轴的编号
    axis = df_duplabels._get_axis_number(axis)
    # 如果 axis == 1，则转置数据框
    if axis == 1:
        df_duplabels = df_duplabels.T

    # 断言数据框具有不模糊的级别 'L1'
    assert_level_values(df_duplabels, ["L1"], axis=axis)

    # 断言数据框具有唯一的标签 'L3'
    assert_label_values(df_duplabels, ["L3"], axis=axis)

    # 断言数据框具有重复的标签 'L2'
    if axis == 0:
        expected_msg = "The column label 'L2' is not unique"
    else:
        expected_msg = "The index label 'L2' is not unique"

    # 使用 pytest 断言预期的 ValueError 异常和匹配的错误消息
    with pytest.raises(ValueError, match=expected_msg):
        assert_label_values(df_duplabels, ["L2"], axis=axis)


# Series
# ------
# 定义函数，用于测试处理 Series 在轴 0 上的标签或级别值
def test_get_label_or_level_values_series_axis0(df):
    # 创建以 "L1" 为索引的 Series
    s = df.set_index("L1").L2
    # 断言 Series 具有轴 0 上的级别 'L1'
    assert_level_values(s, ["L1"], axis=0)

    # 创建以 ["L1", "L2"] 为索引的 Series
    s = df.set_index(["L1", "L2"]).L3
    # 断言 Series 具有轴 0 上的级别 'L1' 和 'L2'
    assert_level_values(s, ["L1", "L2"], axis=0)


# 定义函数，用于测试处理 Series 在轴 1 上的错误情况
def test_get_label_or_level_values_series_axis1_error(df):
    # 创建以 "L1" 为索引的 Series
    s = df.set_index("L1").L2

    # 使用 pytest 断言预期的 ValueError 异常和匹配的错误消息
    with pytest.raises(ValueError, match="No axis named 1"):
        s._get_label_or_level_values("L1", axis=1)


# Test _drop_labels_or_levels
# ===========================
# 定义函数，用于断言在指定轴上删除标签后的数据框的状态
def assert_labels_dropped(frame, labels, axis):
    # 获取指定轴的编号
    axis = frame._get_axis_number(axis)
    # 遍历每个标签
    for label in labels:
        # 调用 _drop_labels_or_levels 方法，删除指定标签的行或列
        df_dropped = frame._drop_labels_or_levels(label, axis=axis)

        # 根据轴的方向进行断言
        if axis == 0:
            # 断言删除前数据框的列中包含指定标签，而删除后数据框的列中不包含该标签
            assert label in frame.columns
            assert label not in df_dropped.columns
        else:
            # 断言删除前数据框的索引中包含指定标签，而删除后数据框的索引中不包含该标签
            assert label in frame.index
            assert label not in df_dropped.index


# 定义函数，用于断言在指定轴上删除级别后的数据框的状态
def assert_levels_dropped(frame, levels, axis):
    # 获取指定轴的编号
    axis = frame._get_axis_number(axis)
    # 遍历每个级别
    for level in levels:
        # 调用 _drop_labels_or_levels 方法，删除指定级别的行或列
        df_dropped = frame._drop_labels_or_levels(level, axis=axis)

        # 根据轴的方向进行断言
        if axis == 0:
            # 断言删除前数据框的索引名称中包含指定级别，而删除后数据框的索引名称中不包含该级别
            assert level in frame.index.names
            assert level not in df_dropped.index.names
        else:
            # 断言删除前数据框的列名称中包含指定级别，而删除后数据框的列名称中不包含该级别
            assert level in frame.columns.names
            assert level not in df_dropped.columns.names


# DataFrame
# ---------
# 定义函数，用于测试处理 DataFrame 上删除标签或级别的操作
def test_drop_labels_or_levels_df(df_levels, axis):
    # 计算预期的标签和级别
    expected_labels, expected_levels = get_labels_levels(df_levels)

    # 获取指定轴的编号
    axis = df_levels._get_axis_number(axis)
    # 如果 axis == 1，则转置数据框
    if axis == 1:
        df_levels = df_levels.T

    # 执行断言检查
    # 检查数据框中删除标签后的结果是否符合预期
    assert_labels_dropped(df_levels, expected_labels, axis=axis)
    
    # 检查数据框中删除层级后的结果是否符合预期
    assert_levels_dropped(df_levels, expected_levels, axis=axis)
    
    # 使用 pytest 来确保在调用 _drop_labels_or_levels 方法时抛出 ValueError 异常，并且异常消息包含指定的错误信息
    with pytest.raises(ValueError, match="not valid labels or levels"):
        df_levels._drop_labels_or_levels("L4", axis=axis)
# Series
# ------
# 测试函数，用于测试删除 Series 中标签或级别的功能
def test_drop_labels_or_levels_series(df):
    # 将 DataFrame 按照列 'L1' 设置为索引，然后取出 'L2' 列作为 Series
    s = df.set_index("L1").L2
    # 调用 assert_levels_dropped 函数，验证在轴 0 上删除了 'L1' 这个级别
    assert_levels_dropped(s, ["L1"], axis=0)

    # 使用 pytest 断言捕获 ValueError 异常，确保试图删除不存在的标签或级别时会引发异常
    with pytest.raises(ValueError, match="not valid labels or levels"):
        s._drop_labels_or_levels("L4", axis=0)

    # 将 DataFrame 按照列 'L1' 和 'L2' 设置为复合索引，然后取出 'L3' 列作为 Series
    s = df.set_index(["L1", "L2"]).L3
    # 再次调用 assert_levels_dropped 函数，验证在轴 0 上删除了 'L1' 和 'L2' 这两个级别
    assert_levels_dropped(s, ["L1", "L2"], axis=0)

    # 使用 pytest 断言捕获 ValueError 异常，确保试图删除不存在的标签或级别时会引发异常
    with pytest.raises(ValueError, match="not valid labels or levels"):
        s._drop_labels_or_levels("L4", axis=0)
```