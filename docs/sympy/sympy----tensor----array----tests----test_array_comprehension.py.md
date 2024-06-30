# `D:\src\scipysrc\sympy\sympy\tensor\array\tests\test_array_comprehension.py`

```
# 从 sympy.tensor.array.array_comprehension 模块导入 ArrayComprehension 和 ArrayComprehensionMap 类
from sympy.tensor.array.array_comprehension import ArrayComprehension, ArrayComprehensionMap
# 从 sympy.tensor.array 模块导入 ImmutableDenseNDimArray 类
from sympy.tensor.array import ImmutableDenseNDimArray
# 从 sympy.abc 模块导入 i, j, k, l 符号变量
from sympy.abc import i, j, k, l
# 从 sympy.testing.pytest 模块导入 raises 函数
from sympy.testing.pytest import raises
# 从 sympy.matrices 模块导入 Matrix 类
from sympy.matrices import Matrix


# 定义测试函数 test_array_comprehension
def test_array_comprehension():
    # 创建 ArrayComprehension 对象 a，表示一个 3x3 的二维数组，元素为 i*j，其中 i 范围是 1 到 3，j 范围是 2 到 4
    a = ArrayComprehension(i*j, (i, 1, 3), (j, 2, 4))
    # 创建 ArrayComprehension 对象 b，表示一个一维数组，元素为 i，i 的范围是 1 到 j+1
    b = ArrayComprehension(i, (i, 1, j+1))
    # 创建 ArrayComprehension 对象 c，表示一个 3x4x5x6 的四维数组，元素为 i+j+k+l，各变量范围分别是 (i, 1, 2), (j, 1, 3), (k, 1, 4), (l, 1, 5)
    c = ArrayComprehension(i+j+k+l, (i, 1, 2), (j, 1, 3), (k, 1, 4), (l, 1, 5))
    # 创建 ArrayComprehension 对象 d，表示一个一维数组，元素为 k，k 的范围是 1 到 5
    d = ArrayComprehension(k, (i, 1, 5))
    # 创建 ArrayComprehension 对象 e，表示一个一维数组，元素为 i，i 的范围是 k+1 到 k+5
    e = ArrayComprehension(i, (j, k+1, k+5))
    
    # 断言语句，验证数组 a 调用 doit() 方法后转换为列表形式等于特定的二维列表
    assert a.doit().tolist() == [[2, 3, 4], [4, 6, 8], [6, 9, 12]]
    # 断言语句，验证数组 a 的形状为 (3, 3)
    assert a.shape == (3, 3)
    # 断言语句，验证数组 a 的形状是否为数值形式
    assert a.is_shape_numeric == True
    # 断言语句，验证数组 a 转换为列表形式等于特定的二维列表
    assert a.tolist() == [[2, 3, 4], [4, 6, 8], [6, 9, 12]]
    # 断言语句，验证数组 a 转换为 Matrix 对象后等于特定的 Matrix 对象
    assert a.tomatrix() == Matrix([
                           [2, 3, 4],
                           [4, 6, 8],
                           [6, 9, 12]])
    # 断言语句，验证数组 a 的长度为 9
    assert len(a) == 9
    # 断言语句，验证 b 调用 doit() 方法后返回的对象类型是 ArrayComprehension
    assert isinstance(b.doit(), ArrayComprehension)
    # 断言语句，验证数组 a 调用 doit() 方法后返回的对象类型是 ImmutableDenseNDimArray
    assert isinstance(a.doit(), ImmutableDenseNDimArray)
    # 断言语句，验证 b 替换 j 为 3 后的结果等于 ArrayComprehension(i, (i, 1, 4))
    assert b.subs(j, 3) == ArrayComprehension(i, (i, 1, 4))
    # 断言语句，验证数组 b 的自由符号集合为 {j}
    assert b.free_symbols == {j}
    # 断言语句，验证数组 b 的形状为 (j+1,)
    assert b.shape == (j + 1,)
    # 断言语句，验证数组 b 的秩为 1
    assert b.rank() == 1
    # 断言语句，验证数组 b 的形状是否为数值形式为 False
    assert b.is_shape_numeric == False
    # 断言语句，验证数组 c 的自由符号集合为空集合
    assert c.free_symbols == set()
    # 断言语句，验证数组 c 的函数表达式为 i + j + k + l
    assert c.function == i + j + k + l
    # 断言语句，验证数组 c 的限制条件为 ((i, 1, 2), (j, 1, 3), (k, 1, 4), (l, 1, 5))
    assert c.limits == ((i, 1, 2), (j, 1, 3), (k, 1, 4), (l, 1, 5))
    # 断言语句，验证数组 c 调用 doit() 方法后转换为列表形式等于特定的四维列表
    assert c.doit().tolist() == [[[[4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11]],
                                  [[5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12]],
                                  [[6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13]]],
                                 [[[5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12]],
                                  [[6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13]],
                                  [[7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13], [10, 11, 12, 13, 14]]]]
    # 断言语句，验证数组 c 的自由符号集合为空集合
    assert c.free_symbols == set()
    # 断言语句，验证数组 c 的变量列表为 [i, j, k, l]
    assert c.variables == [i, j, k, l]
    # 断言语句，验证数组 c 的绑定符号列表为 [i, j, k, l]
    assert c.bound_symbols == [i, j, k, l]
    # 断言语句，验证数组 d 调用 doit() 方法后转换为列表形式等于特定的一维列表，元素全为 k
    assert d.doit().tolist() == [k, k, k, k, k]
    # 断言语句，验证数组 e 的长度为 5
    assert len(e) == 5
    # 断言语句，验证调用 ArrayComprehension 构造函数时，给定的参数不合法会抛出 TypeError 异常
    raises(TypeError, lambda: ArrayComprehension(i*j, (i, 1, 3), (j, 2, [1, 3, 2])))
    # 断言语句，验证调用 ArrayComprehension 构造函数时，给定的参数不
    # 确保数组 `a` 转换为列表后与预期的值相等
    assert a.tolist() == [2, 3, 4, 5, 6]
    # 确保数组 `a` 的长度为 5
    assert len(a) == 5
    # 确保调用 `a.doit()` 返回的对象是 ImmutableDenseNDimArray 类型的实例
    assert isinstance(a.doit(), ImmutableDenseNDimArray)
    # 使用 lambda 表达式创建 ArrayComprehensionMap 对象 `expr`，其表达式为 `i+1`，`i` 的范围是 1 到 `k`
    expr = ArrayComprehensionMap(lambda i: i+1, (i, 1, k))
    # 确保 `expr.doit()` 返回的结果与 `expr` 相等
    assert expr.doit() == expr
    # 确保将 `k` 替换为 4 后，`expr.subs(k, 4)` 返回的结果与预期的 ArrayComprehensionMap 对象相等
    assert expr.subs(k, 4) == ArrayComprehensionMap(lambda i: i+1, (i, 1, 4))
    # 确保将 `k` 替换为 4 后，`expr.subs(k, 4).doit()` 返回的 ImmutableDenseNDimArray 对象与预期的数组相等
    assert expr.subs(k, 4).doit() == ImmutableDenseNDimArray([2, 3, 4, 5])
    # 使用多个范围参数创建 ArrayComprehensionMap 对象 `b`，其表达式为 `i+1`，范围分别是 (i, 1, 2), (i, 1, 3), (i, 1, 4), (i, 1, 5)
    b = ArrayComprehensionMap(lambda i: i+1, (i, 1, 2), (i, 1, 3), (i, 1, 4), (i, 1, 5))
    # 确保调用 `b.doit()` 后得到的结果转换为列表后与预期的多维数组相等
    assert b.doit().tolist() == [[[[2, 3, 4, 5, 6], [3, 5, 7, 9, 11], [4, 7, 10, 13, 16], [5, 9, 13, 17, 21]],
                                  [[3, 5, 7, 9, 11], [5, 9, 13, 17, 21], [7, 13, 19, 25, 31], [9, 17, 25, 33, 41]],
                                  [[4, 7, 10, 13, 16], [7, 13, 19, 25, 31], [10, 19, 28, 37, 46], [13, 25, 37, 49, 61]]],
                                 [[[3, 5, 7, 9, 11], [5, 9, 13, 17, 21], [7, 13, 19, 25, 31], [9, 17, 25, 33, 41]],
                                  [[5, 9, 13, 17, 21], [9, 17, 25, 33, 41], [13, 25, 37, 49, 61], [17, 33, 49, 65, 81]],
                                  [[7, 13, 19, 25, 31], [13, 25, 37, 49, 61], [19, 37, 55, 73, 91], [25, 49, 73, 97, 121]]]]
    
    # 测试 lambda 表达式的相关功能
    # 确保使用 lambda 表达式 `lambda: 3` 创建的 ArrayComprehensionMap 对象调用 `doit()` 后转换为列表后与预期的结果相等
    assert ArrayComprehensionMap(lambda: 3, (i, 1, 5)).doit().tolist() == [3, 3, 3, 3, 3]
    # 确保使用 lambda 表达式 `lambda i: i+1` 创建的 ArrayComprehensionMap 对象调用 `doit()` 后转换为列表后与预期的结果相等
    assert ArrayComprehensionMap(lambda i: i+1, (i, 1, 5)).doit().tolist() == [2, 3, 4, 5, 6]
    # 确保尝试创建无效的 ArrayComprehensionMap 对象时抛出 ValueError 异常
    raises(ValueError, lambda: ArrayComprehensionMap(i*j, (i, 1, 3), (j, 2, 4)))
    # 使用 lambda 表达式 `lambda i, j: i+j` 创建 ArrayComprehensionMap 对象 `a`
    a = ArrayComprehensionMap(lambda i, j: i+j, (i, 1, 5))
    # 确保调用 `a.doit()` 时抛出 ValueError 异常
    raises(ValueError, lambda: a.doit())
```