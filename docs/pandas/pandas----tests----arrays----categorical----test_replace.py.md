# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_replace.py`

```
import pytest  # 导入 pytest 库

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
from pandas import Categorical  # 从 pandas 库中导入 Categorical 类
import pandas._testing as tm  # 导入 pandas 内部测试模块


@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器定义参数化测试
    "to_replace,value,expected",  # 参数化的参数列表
    [
        # one-to-one 替换
        (4, 1, [1, 2, 3]),  # 当 to_replace 是 4，替换为 1，期望结果是 [1, 2, 3]
        (3, 1, [1, 2, 1]),  # 当 to_replace 是 3，替换为 1，期望结果是 [1, 2, 1]
        # many-to-one 替换
        ((5, 6), 2, [1, 2, 3]),  # 当 to_replace 是 (5, 6)，替换为 2，期望结果是 [1, 2, 3]
        ((3, 2), 1, [1, 1, 1]),  # 当 to_replace 是 (3, 2)，替换为 1，期望结果是 [1, 1, 1]
    ],
)
def test_replace_categorical_series(to_replace, value, expected):
    # GH 31720
    ser = pd.Series([1, 2, 3], dtype="category")  # 创建一个分类类型的 Series 对象
    result = ser.replace(to_replace, value)  # 对 Series 对象进行替换操作
    expected = pd.Series(Categorical(expected, categories=[1, 2, 3]))  # 创建期望的 Series 对象
    tm.assert_series_equal(result, expected)  # 使用测试模块中的 assert_series_equal 函数进行结果比较


@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器定义参数化测试
    "to_replace,value",  # 参数化的参数列表
    [
        # one-to-one 替换
        (3, 5),  # 当 to_replace 是 3，替换为 5
        # many-to-one 替换
        ((3, 2), 5),  # 当 to_replace 是 (3, 2)，替换为 5
    ],
)
def test_replace_categorical_series_new_category_raises(to_replace, value):
    # GH 31720
    ser = pd.Series([1, 2, 3], dtype="category")  # 创建一个分类类型的 Series 对象
    with pytest.raises(  # 使用 pytest 的 raises 函数检测异常
        TypeError, match="Cannot setitem on a Categorical with a new category"
    ):
        ser.replace(to_replace, value)  # 尝试替换操作，期望引发 TypeError 异常


def test_replace_maintain_ordering():
    # GH51016
    dtype = pd.CategoricalDtype([0, 1, 2], ordered=True)  # 创建有序分类类型
    ser = pd.Series([0, 1, 2], dtype=dtype)  # 创建一个使用有序分类类型的 Series 对象
    result = ser.replace(0, 2)  # 替换 Series 中的值
    expected = pd.Series([2, 1, 2], dtype=dtype)  # 创建期望的 Series 对象
    tm.assert_series_equal(expected, result, check_category_order=True)  # 使用测试模块中的 assert_series_equal 函数进行结果比较


def test_replace_categorical_ea_dtype():
    # GH49404
    cat = Categorical(pd.array(["a", "b", "c"], dtype="string"))  # 创建一个字符串类型的分类对象
    result = pd.Series(cat).replace(["a", "b"], ["c", "c"])._values  # 对分类对象进行替换操作，并获取其值
    expected = Categorical(  # 创建期望的分类对象
        pd.array(["c"] * 3, dtype="string"),  # 所有值替换为 "c"
        categories=pd.array(["a", "b", "c"], dtype="string"),  # 分类的类别保持不变
    )
    tm.assert_categorical_equal(result, expected)  # 使用测试模块中的 assert_categorical_equal 函数进行结果比较


def test_replace_categorical_ea_dtype_different_cats_raises():
    # GH49404
    cat = Categorical(pd.array(["a", "b"], dtype="string"))  # 创建一个字符串类型的分类对象
    with pytest.raises(  # 使用 pytest 的 raises 函数检测异常
        TypeError, match="Cannot setitem on a Categorical with a new category"
    ):
        pd.Series(cat).replace(["a", "b"], ["c", pd.NA])  # 尝试替换操作，期望引发 TypeError 异常
```