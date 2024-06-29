# `D:\src\scipysrc\pandas\pandas\tests\extension\array_with_attr\test_array_with_attr.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 测试模块
from pandas.tests.extension.array_with_attr import FloatAttrArray  # 导入自定义的 FloatAttrArray 类


def test_concat_with_all_na():
    # 测试合并/连接时，确保列数组的属性在重新索引数组时得到保留
    # 使用 FloatAttrArray 创建带有属性的浮点数组
    arr = FloatAttrArray(np.array([np.nan, np.nan], dtype="float64"), attr="test")

    # 创建 DataFrame df1 包含列 'col' 和 'key'，其中 'col' 是 arr，'key' 是 [0, 1]
    df1 = pd.DataFrame({"col": arr, "key": [0, 1]})
    # 创建 DataFrame df2 包含列 'key' 和 'col2'，其中 'key' 是 [0, 1]，'col2' 是 [1, 2]
    df2 = pd.DataFrame({"key": [0, 1], "col2": [1, 2]})
    # 使用 'key' 列进行合并 df1 和 df2，并将结果保存在 result 中
    result = pd.merge(df1, df2, on="key")
    # 创建期望的 DataFrame expected，包含 'col' (来自 arr)、'key' 和 'col2'
    expected = pd.DataFrame({"col": arr, "key": [0, 1], "col2": [1, 2]})
    # 断言 result 和 expected 相等
    tm.assert_frame_equal(result, expected)
    # 断言 result 的 'col' 列的数组属性为 "test"
    assert result["col"].array.attr == "test"

    # 重新设置 df2，'key' 列中的值为 [0, 2]，'col2' 列为 [1, 2]
    df2 = pd.DataFrame({"key": [0, 2], "col2": [1, 2]})
    # 再次使用 'key' 列进行合并 df1 和 df2，并将结果保存在 result 中
    result = pd.merge(df1, df2, on="key")
    # 创建期望的 DataFrame expected，包含 'col' (从 arr 中取出第一个元素)、'key' 和 'col2'
    expected = pd.DataFrame({"col": arr.take([0]), "key": [0], "col2": [1]})
    # 断言 result 和 expected 相等
    tm.assert_frame_equal(result, expected)
    # 断言 result 的 'col' 列的数组属性为 "test"
    assert result["col"].array.attr == "test"

    # 使用 concat 将 df1 和 df2 按 'key' 列连接，并将结果保存在 result 中
    result = pd.concat([df1.set_index("key"), df2.set_index("key")], axis=1)
    # 创建期望的 DataFrame expected，包含 'col' (从 arr 中取出 [0, 1, -1] 元素)、'col2' 和 'key' 列
    expected = pd.DataFrame(
        {"col": arr.take([0, 1, -1]), "col2": [1, np.nan, 2], "key": [0, 1, 2]}
    ).set_index("key")
    # 断言 result 和 expected 相等
    tm.assert_frame_equal(result, expected)
    # 断言 result 的 'col' 列的数组属性为 "test"
    assert result["col"].array.attr == "test"
```