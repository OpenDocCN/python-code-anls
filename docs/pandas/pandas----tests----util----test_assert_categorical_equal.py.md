# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_categorical_equal.py`

```
# 导入 pytest 测试框架
import pytest

# 从 pandas 库中导入 Categorical 类
from pandas import Categorical

# 导入 pandas 测试模块
import pandas._testing as tm

# 使用 pytest.mark.parametrize 装饰器，定义参数化测试函数 test_categorical_equal
@pytest.mark.parametrize("c", [None, [1, 2, 3, 4, 5]])
def test_categorical_equal(c):
    # 创建 Categorical 对象 c，指定其 categories 参数
    c = Categorical([1, 2, 3, 4], categories=c)
    # 使用 pandas 测试模块中的 assert_categorical_equal 方法，比较 c 和 c 自身，期望相等
    tm.assert_categorical_equal(c, c)

# 使用 pytest.mark.parametrize 装饰器，定义参数化测试函数 test_categorical_equal_order_mismatch
@pytest.mark.parametrize("check_category_order", [True, False])
def test_categorical_equal_order_mismatch(check_category_order):
    # 创建两个不同顺序的 Categorical 对象 c1 和 c2
    c1 = Categorical([1, 2, 3, 4], categories=[1, 2, 3, 4])
    c2 = Categorical([1, 2, 3, 4], categories=[4, 3, 2, 1])
    kwargs = {"check_category_order": check_category_order}

    # 如果 check_category_order 为 True，定义错误信息 msg，并期望抛出 AssertionError
    if check_category_order:
        msg = """Categorical\\.categories are different

Categorical\\.categories values are different \\(100\\.0 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[4, 3, 2, 1\\], dtype='int64'\\)"""
        with pytest.raises(AssertionError, match=msg):
            tm.assert_categorical_equal(c1, c2, **kwargs)
    # 否则，直接比较 c1 和 c2，期望相等
    else:
        tm.assert_categorical_equal(c1, c2, **kwargs)

# 定义测试函数 test_categorical_equal_categories_mismatch
def test_categorical_equal_categories_mismatch():
    # 定义错误信息 msg
    msg = """Categorical\\.categories are different

Categorical\\.categories values are different \\(25\\.0 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 3, 5\\], dtype='int64'\\)"""

    # 创建两个不同 categories 的 Categorical 对象 c1 和 c2，并期望抛出 AssertionError
    c1 = Categorical([1, 2, 3, 4])
    c2 = Categorical([1, 2, 3, 5])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)

# 定义测试函数 test_categorical_equal_codes_mismatch
def test_categorical_equal_codes_mismatch():
    # 定义 categories 列表和错误信息 msg
    categories = [1, 2, 3, 4]
    msg = """Categorical\\.codes are different

Categorical\\.codes values are different \\(50\\.0 %\\)
\\[left\\]:  \\[0, 1, 3, 2\\]
\\[right\\]: \\[0, 1, 2, 3\\]"""

    # 创建两个不同 codes 的 Categorical 对象 c1 和 c2，并期望抛出 AssertionError
    c1 = Categorical([1, 2, 4, 3], categories=categories)
    c2 = Categorical([1, 2, 3, 4], categories=categories)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)

# 定义测试函数 test_categorical_equal_ordered_mismatch
def test_categorical_equal_ordered_mismatch():
    # 定义数据列表和错误信息 msg
    data = [1, 2, 3, 4]
    msg = """Categorical are different

Attribute "ordered" are different
\\[left\\]:  False
\\[right\\]: True"""

    # 创建两个不同 ordered 的 Categorical 对象 c1 和 c2，并期望抛出 AssertionError
    c1 = Categorical(data, ordered=False)
    c2 = Categorical(data, ordered=True)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)

# 使用 pytest.mark.parametrize 装饰器，定义参数化测试函数 test_categorical_equal_object_override
@pytest.mark.parametrize("obj", ["index", "foo", "pandas"])
def test_categorical_equal_object_override(obj):
    # 定义数据列表和错误信息 msg
    data = [1, 2, 3, 4]
    msg = f"""{obj} are different

Attribute "ordered" are different
\\[left\\]:  False
\\[right\\]: True"""

    # 创建两个不同 ordered 的 Categorical 对象 c1 和 c2，并期望抛出 AssertionError
    c1 = Categorical(data, ordered=False)
    c2 = Categorical(data, ordered=True)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2, obj=obj)
```