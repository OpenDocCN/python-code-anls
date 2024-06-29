# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_take.py`

```
import numpy as np  # 导入NumPy库，用于处理数组和数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas import Categorical  # 从pandas库中导入Categorical类，用于处理分类数据
import pandas._testing as tm  # 导入pandas._testing模块，提供测试辅助函数

@pytest.fixture(params=[True, False])
def allow_fill(request):
    """Fixture providing a boolean 'allow_fill' parameter for Categorical.take"""
    return request.param  # 返回参数化的allow_fill值，用于测试用例

class TestTake:
    # 测试类用于测试Categorical.take方法

    # https://github.com/pandas-dev/pandas/issues/20664
    def test_take_default_allow_fill(self):
        cat = Categorical(["a", "b"])  # 创建一个包含两个分类值的Categorical对象
        with tm.assert_produces_warning(None):  # 确保没有产生警告
            result = cat.take([0, -1])  # 调用take方法取出指定索引位置的值
        assert result.equals(cat)  # 检查结果是否与原始Categorical对象相等

    def test_take_positive_no_warning(self):
        cat = Categorical(["a", "b"])  # 创建一个包含两个分类值的Categorical对象
        with tm.assert_produces_warning(None):  # 确保没有产生警告
            cat.take([0, 0])  # 调用take方法取出指定索引位置的值

    def test_take_bounds(self, allow_fill):
        # https://github.com/pandas-dev/pandas/issues/20664
        cat = Categorical(["a", "b", "a"])  # 创建一个包含三个分类值的Categorical对象
        if allow_fill:
            msg = "indices are out-of-bounds"  # 如果允许填充，期望的错误消息
        else:
            msg = "index 4 is out of bounds for( axis 0 with)? size 3"  # 如果不允许填充，期望的错误消息
        with pytest.raises(IndexError, match=msg):  # 检查是否抛出预期的IndexError异常
            cat.take([4, 5], allow_fill=allow_fill)  # 调用take方法尝试取出超出边界的索引

    def test_take_empty(self, allow_fill):
        # https://github.com/pandas-dev/pandas/issues/20664
        cat = Categorical([], categories=["a", "b"])  # 创建一个空的Categorical对象，指定分类列表
        if allow_fill:
            msg = "indices are out-of-bounds"  # 如果允许填充，期望的错误消息
        else:
            msg = "cannot do a non-empty take from an empty axes"  # 如果不允许填充，期望的错误消息
        with pytest.raises(IndexError, match=msg):  # 检查是否抛出预期的IndexError异常
            cat.take([0], allow_fill=allow_fill)  # 调用take方法尝试从空对象中取值

    def test_positional_take(self, ordered):
        cat = Categorical(["a", "a", "b", "b"], categories=["b", "a"], ordered=ordered)
        result = cat.take([0, 1, 2], allow_fill=False)
        expected = Categorical(
            ["a", "a", "b"], categories=cat.categories, ordered=ordered
        )
        tm.assert_categorical_equal(result, expected)

    def test_positional_take_unobserved(self, ordered):
        cat = Categorical(["a", "b"], categories=["a", "b", "c"], ordered=ordered)
        result = cat.take([1, 0], allow_fill=False)
        expected = Categorical(["b", "a"], categories=cat.categories, ordered=ordered)
        tm.assert_categorical_equal(result, expected)

    def test_take_allow_fill(self):
        # https://github.com/pandas-dev/pandas/issues/23296
        cat = Categorical(["a", "a", "b"])  # 创建一个包含三个分类值的Categorical对象
        result = cat.take([0, -1, -1], allow_fill=True)  # 使用allow_fill=True参数调用take方法
        expected = Categorical(["a", np.nan, np.nan], categories=["a", "b"])  # 预期的结果Categorical对象
        tm.assert_categorical_equal(result, expected)  # 检查实际结果与预期结果是否相等

    def test_take_fill_with_negative_one(self):
        # -1 was a category
        cat = Categorical([-1, 0, 1])  # 创建一个包含整数分类值的Categorical对象
        result = cat.take([0, -1, 1], allow_fill=True, fill_value=-1)  # 使用填充值调用take方法
        expected = Categorical([-1, -1, 0], categories=[-1, 0, 1])  # 预期的结果Categorical对象
        tm.assert_categorical_equal(result, expected)  # 检查实际结果与预期结果是否相等
    # 定义测试函数，用于测试在填充值情况下的 `take` 方法
    def test_take_fill_value(self):
        # 针对 GitHub 上的特定问题链接：https://github.com/pandas-dev/pandas/issues/23296
        # 创建一个包含值为 ["a", "b", "c"] 的分类变量对象
        cat = Categorical(["a", "b", "c"])
        # 使用 `take` 方法从分类变量中获取指定索引处的值，当索引为负数时使用填充值 "a"，允许填充操作
        result = cat.take([0, 1, -1], fill_value="a", allow_fill=True)
        # 期望的结果是一个包含值为 ["a", "b", "a"] 的分类变量对象，同时指定了类别为 ["a", "b", "c"]
        expected = Categorical(["a", "b", "a"], categories=["a", "b", "c"])
        # 使用测试框架的断言方法来比较 `result` 和 `expected` 是否相等
        tm.assert_categorical_equal(result, expected)

    # 定义测试函数，用于测试在填充值情况下的 `take` 方法并期望引发异常
    def test_take_fill_value_new_raises(self):
        # 针对 GitHub 上的特定问题链接：https://github.com/pandas-dev/pandas/issues/23296
        # 创建一个包含值为 ["a", "b", "c"] 的分类变量对象
        cat = Categorical(["a", "b", "c"])
        # 准备一个正则表达式模式，用于匹配异常消息，表明不允许向包含新类别的分类变量中插入新值 "d"
        xpr = r"Cannot setitem on a Categorical with a new category \(d\)"
        # 使用 pytest 框架的 `raises` 方法来断言执行 `take` 方法时是否会引发 TypeError 异常，并匹配指定的异常消息模式
        with pytest.raises(TypeError, match=xpr):
            cat.take([0, 1, -1], fill_value="d", allow_fill=True)
```