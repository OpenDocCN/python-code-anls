# `D:\src\scipysrc\sympy\sympy\core\tests\test_sorting.py`

```
from sympy.core.sorting import default_sort_key, ordered  # 导入排序相关的函数和类
from sympy.testing.pytest import raises  # 导入测试框架中的异常检测函数

from sympy.abc import x  # 导入符号变量 x


def test_default_sort_key():
    func = lambda x: x  # 定义一个简单的 lambda 函数 func
    assert sorted([func, x, func], key=default_sort_key) == [func, func, x]  # 对列表进行排序，并使用 default_sort_key 作为排序关键字

    class C:
        def __repr__(self):
            return 'x.y'  # 定义一个类 C，其 __repr__ 方法返回字符串 'x.y'
    func = C()  # 创建类 C 的一个实例 func
    assert sorted([x, func], key=default_sort_key) == [func, x]  # 对列表进行排序，并使用 default_sort_key 作为排序关键字


def test_ordered():
    # Issue 7210 - this had been failing with python2/3 problems
    assert (list(ordered([{1:3, 2:4, 9:10}, {1:3}])) == \
               [{1: 3}, {1: 3, 2: 4, 9: 10}])  # 测试 ordered 函数对字典列表进行排序的正确性
    # warnings should not be raised for identical items
    l = [1, 1]  # 创建一个列表 l 包含重复的元素
    assert list(ordered(l, warn=True)) == l  # 测试 ordered 函数对具有相同元素的列表不会引发警告
    l = [[1], [2], [1]]  # 创建一个包含列表的列表 l
    assert list(ordered(l, warn=True)) == [[1], [1], [2]]  # 测试 ordered 函数对列表的列表按特定规则排序
    raises(ValueError, lambda: list(ordered(['a', 'ab'], keys=[lambda x: x[0]],
        default=False, warn=True)))  # 测试 ordered 函数在特定条件下是否引发 ValueError 异常
```