# `D:\src\scipysrc\sympy\sympy\tensor\array\tests\test_ndim_array.py`

```
from sympy.testing.pytest import raises
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.tensor.array import Array
from sympy.tensor.array.dense_ndim_array import (
    ImmutableDenseNDimArray, MutableDenseNDimArray)
from sympy.tensor.array.sparse_ndim_array import (
    ImmutableSparseNDimArray, MutableSparseNDimArray)

from sympy.abc import x, y

# 可变数组类型的列表
mutable_array_types = [
    MutableDenseNDimArray,
    MutableSparseNDimArray
]

# 所有数组类型的列表
array_types = [
    ImmutableDenseNDimArray,
    ImmutableSparseNDimArray,
    MutableDenseNDimArray,
    MutableSparseNDimArray
]

# 测试负索引的数组功能
def test_array_negative_indices():
    for ArrayType in array_types:
        # 创建一个测试用的数组对象
        test_array = ArrayType([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        # 断言负索引能正确获取列的最后一个元素
        assert test_array[:, -1] == Array([5, 10])
        assert test_array[:, -2] == Array([4, 9])
        assert test_array[:, -3] == Array([3, 8])
        assert test_array[:, -4] == Array([2, 7])
        assert test_array[:, -5] == Array([1, 6])
        # 断言正索引能正确获取指定位置的列
        assert test_array[:, 0] == Array([1, 6])
        assert test_array[:, 1] == Array([2, 7])
        assert test_array[:, 2] == Array([3, 8])
        assert test_array[:, 3] == Array([4, 9])
        assert test_array[:, 4] == Array([5, 10])

        # 使用 lambda 函数和 raises 函数检查索引超出范围时是否会抛出 ValueError 异常
        raises(ValueError, lambda: test_array[:, -6])
        raises(ValueError, lambda: test_array[-3, :])

        # 断言能正确获取二维数组的最后一个元素
        assert test_array[-1, -1] == 10


# 测试问题编号 18361 的数组简化功能
def test_issue_18361():
    A = Array([sin(2 * x) - 2 * sin(x) * cos(x)])
    B = Array([sin(x)**2 + cos(x)**2, 0])
    C = Array([(x + x**2)/(x*sin(y)**2 + x*cos(y)**2), 2*sin(x)*cos(x)])
    assert simplify(A) == Array([0])
    assert simplify(B) == Array([1, 0])
    assert simplify(C) == Array([x + 1, sin(2*x)])


# 测试问题编号 20222 的数组操作异常
def test_issue_20222():
    A = Array([[1, 2], [3, 4]])
    B = Matrix([[1,2],[3,4]])
    raises(TypeError, lambda: A - B)


# 测试问题编号 17851 的空数组初始化和属性检查
def test_issue_17851():
    for array_type in array_types:
        A = array_type([])
        assert isinstance(A, array_type)
        assert A.shape == (0,)
        assert list(A) == []


# 测试问题编号 18715 和 17851 的可变数组修改功能
def test_issue_and_18715():
    for array_type in mutable_array_types:
        A = array_type([0, 1, 2])
        A[0] += 5
        assert A[0] == 5
```