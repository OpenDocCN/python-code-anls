# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_na_indexing.py`

```
# 导入pytest库，用于测试框架
import pytest

# 导入pandas库，并导入测试工具模块
import pandas as pd
import pandas._testing as tm

# 使用pytest的@parametrize装饰器进行参数化测试
@pytest.mark.parametrize(
    "values, dtype",
    [
        ([], "object"),                         # 空列表作为值，数据类型为object
        ([1, 2, 3], "int64"),                   # 整数列表作为值，数据类型为int64
        ([1.0, 2.0, 3.0], "float64"),           # 浮点数列表作为值，数据类型为float64
        (["a", "b", "c"], "object"),            # 字符串列表作为值，数据类型为object
        (["a", "b", "c"], "string"),            # 字符串列表作为值，数据类型为string
        ([1, 2, 3], "datetime64[ns]"),          # 整数列表作为值，数据类型为datetime64[ns]
        ([1, 2, 3], "datetime64[ns, CET]"),     # 整数列表作为值，数据类型为datetime64[ns, CET]
        ([1, 2, 3], "timedelta64[ns]"),         # 整数列表作为值，数据类型为timedelta64[ns]
        (["2000", "2001", "2002"], "Period[D]"),# 字符串列表作为值，数据类型为Period[D]
        ([1, 0, 3], "Sparse"),                  # 整数列表作为值，数据类型为Sparse
        ([pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(3, 4)], "interval"),  # 区间列表作为值，数据类型为interval
    ],
)
# 对mask进行参数化，包括三种情况的布尔值列表
@pytest.mark.parametrize(
    "mask", [[True, False, False], [True, True, True], [False, False, False]]
)
# 对indexer_class进行参数化，包括四种不同的索引器类型
@pytest.mark.parametrize("indexer_class", [list, pd.array, pd.Index, pd.Series])
# 对frame进行参数化，包括True和False两种情况
@pytest.mark.parametrize("frame", [True, False])
# 定义测试函数test_series_mask_boolean，测试Series对象的布尔掩码功能
def test_series_mask_boolean(values, dtype, mask, indexer_class, frame):
    # 如果values的长度小于3，则使用前三个字母作为索引
    index = ["a", "b", "c"][: len(values)]
    # 根据values的长度截取mask列表
    mask = mask[: len(values)]

    # 创建Series对象obj，根据参数dtype和index
    obj = pd.Series(values, dtype=dtype, index=index)
    
    # 如果frame为True
    if frame:
        # 如果values的长度为0，则创建一个空的DataFrame对象
        if len(values) == 0:
            obj = pd.DataFrame(dtype=dtype, index=index)
        else:
            # 否则将obj转换为DataFrame对象
            obj = obj.to_frame()

    # 根据indexer_class类型进行不同的处理
    if indexer_class is pd.array:
        # 如果indexer_class是pd.array，则将mask转换为布尔类型的pd.array
        mask = pd.array(mask, dtype="boolean")
    elif indexer_class is pd.Series:
        # 如果indexer_class是pd.Series，则创建一个带有index的布尔类型Series
        mask = pd.Series(mask, index=obj.index, dtype="boolean")
    else:
        # 否则，使用indexer_class来处理mask列表
        mask = indexer_class(mask)

    # 计算期望的结果
    expected = obj[mask]

    # 根据掩码选取的结果
    result = obj[mask]
    # 使用测试工具tm.assert_equal检查结果是否与期望相同
    tm.assert_equal(result, expected)

    # 如果indexer_class是pd.Series
    if indexer_class is pd.Series:
        # 创建一个错误信息，用于检查索引位置的布尔索引不能使用可索引对象作为掩码
        msg = "iLocation based boolean indexing cannot use an indexable as a mask"
        # 使用pytest的raises函数检查是否引发了预期的ValueError，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            result = obj.iloc[mask]
    else:
        # 否则，使用iloc进行索引操作
        result = obj.iloc[mask]
        # 使用测试工具tm.assert_equal检查结果是否与期望相同
        tm.assert_equal(result, expected)

    # 使用loc进行索引操作
    result = obj.loc[mask]
    # 使用测试工具tm.assert_equal检查结果是否与期望相同
    tm.assert_equal(result, expected)


# 定义测试函数test_na_treated_as_false，测试缺失值被视为False的情况
def test_na_treated_as_false(frame_or_series, indexer_sli):
    # 创建一个包含整数1, 2, 3的对象obj
    obj = frame_or_series([1, 2, 3])

    # 创建一个包含True, False, None的布尔类型pd.array作为掩码
    mask = pd.array([True, False, None], dtype="boolean")

    # 计算结果，根据掩码进行选择
    result = indexer_sli(obj)[mask]
    # 计算期望的结果，对掩码进行填充，将None替换为False
    expected = indexer_sli(obj)[mask.fillna(False)]

    # 使用测试工具tm.assert_equal检查结果是否与期望相同
    tm.assert_equal(result, expected)
```