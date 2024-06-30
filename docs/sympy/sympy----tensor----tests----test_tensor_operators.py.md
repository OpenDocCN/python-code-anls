# `D:\src\scipysrc\sympy\sympy\tensor\tests\test_tensor_operators.py`

```
# 导入 sympy 库中的 sin 和 cos 函数
from sympy import sin, cos
# 导入 sympy.testing.pytest 模块中的 raises 函数，用于测试异常情况
from sympy.testing.pytest import raises

# 导入 sympy.tensor.toperators 模块中的 PartialDerivative 类，用于求偏导数
from sympy.tensor.toperators import PartialDerivative
# 导入 sympy.tensor.tensor 模块中的 TensorIndexType、tensor_indices、TensorHead、tensor_heads 类和函数
from sympy.tensor.tensor import (TensorIndexType,
                                 tensor_indices,
                                 TensorHead, tensor_heads)
# 导入 sympy.core.numbers 模块中的 Rational 类，用于处理有理数
from sympy.core.numbers import Rational
# 导入 sympy.core.symbol 模块中的 symbols 函数，用于创建符号变量
from sympy.core.symbol import symbols
# 导入 sympy.matrices.dense 模块中的 diag 函数，用于创建对角矩阵
from sympy.matrices.dense import diag
# 导入 sympy.tensor.array 模块中的 Array 类，用于处理多维数组
from sympy.tensor.array import Array

# 导入 sympy.core.random 模块中的 randint 函数，用于生成随机整数
from sympy.core.random import randint

# 创建一个张量索引类型对象 L
L = TensorIndexType("L")
# 创建多个张量索引对象 i, j, k, m, m1, m2, m3, m4
i, j, k, m, m1, m2, m3, m4 = tensor_indices("i j k m m1 m2 m3 m4", L)
# 创建一个张量索引对象 i0
i0 = tensor_indices("i0", L)
# 创建两个张量索引对象 L_0, L_1
L_0, L_1 = tensor_indices("L_0 L_1", L)

# 创建四个张量头对象 A, B, C, D
A, B, C, D = tensor_heads("A B C D", [L])

# 创建一个张量头对象 H
H = TensorHead("H", [L, L])


# 定义测试函数，用于测试无效的偏导数张量的价值
def test_invalid_partial_derivative_valence():
    raises(ValueError, lambda: PartialDerivative(C(j), D(-j)))
    raises(ValueError, lambda: PartialDerivative(C(-j), D(j)))


# 定义测试函数，用于测试张量的偏导数
def test_tensor_partial_deriv():
    # 测试偏导数的扁平化性质
    expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    assert expr == PartialDerivative(A(i), A(j), A(k))
    assert expr.expr == A(i)
    assert expr.variables == (A(j), A(k))
    assert expr.get_indices() == [i, -j, -k]
    assert expr.get_free_indices() == [i, -j, -k]

    expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(i))
    assert expr.expr == A(L_0)
    assert expr.variables == (A(j), A(L_0))

    expr1 = PartialDerivative(A(i), A(j))
    assert expr1.expr == A(i)
    assert expr1.variables == (A(j),)

    expr2 = A(i)*PartialDerivative(H(k, -i), A(j))
    assert expr2.get_indices() == [L_0, k, -L_0, -j]

    expr2b = A(i)*PartialDerivative(H(k, -i), A(-j))
    assert expr2b.get_indices() == [L_0, k, -L_0, j]

    expr3 = A(i)*PartialDerivative(B(k)*C(-i) + 3*H(k, -i), A(j))
    assert expr3.get_indices() == [L_0, k, -L_0, -j]

    expr4 = (A(i) + B(i))*PartialDerivative(C(j), D(j))
    assert expr4.get_indices() == [i, L_0, -L_0]

    expr4b = (A(i) + B(i))*PartialDerivative(C(-j), D(-j))
    assert expr4b.get_indices() == [i, -L_0, L_0]

    expr5 = (A(i) + B(i))*PartialDerivative(C(-i), D(j))
    assert expr5.get_indices() == [L_0, -L_0, -j]


# 定义测试函数，用于替换数组中的偏导数张量
def test_replace_arrays_partial_derivative():

    # 创建符号变量 x, y, z, t
    x, y, z, t = symbols("x y z t")

    expr = PartialDerivative(A(i), B(j))
    # 替换张量 A(i) 和 B(j) 的值为数组，并验证结果
    repl = expr.replace_with_arrays({A(i): [sin(x)*cos(y), x**3*y**2], B(i): [x, y]})
    assert repl == Array([[cos(x)*cos(y), -sin(x)*sin(y)], [3*x**2*y**2, 2*x**3*y]])
    # 替换张量 A(i) 和 B(j) 的值为数组，并指定替换的索引，验证结果
    repl = expr.replace_with_arrays({A(i): [sin(x)*cos(y), x**3*y**2], B(i): [x, y]}, [-j, i])
    assert repl == Array([[cos(x)*cos(y), 3*x**2*y**2], [-sin(x)*sin(y), 2*x**3*y]])

    # 验证偏导数表达式对应的自由索引和所有索引
    expr = PartialDerivative(A(i), A(-j))
    assert expr.get_free_indices() == [i, j]
    assert expr.get_indices() == [i, j]
    # 替换张量 A(i) 的值为数组，并使用对角矩阵替换张量 L 的值，验证结果
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, [i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, [i, j]) == Array([[1, 0], [0, -1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, 1)}, [i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, -1)}, [i, j]) == Array([[1, 0], [0, -1]])

    # 创建一个 PartialDerivative 对象，计算 A(i) 对 A(j) 的偏导数表达式
    expr = PartialDerivative(A(i), A(j))
    # 获取偏导数表达式中的自由指标
    assert expr.get_free_indices() == [i, -j]
    # 获取偏导数表达式中的所有指标
    assert expr.get_indices() == [i, -j]
    # 使用指定的替换字典替换偏导数表达式中的数组，并断言结果
    assert expr.replace_with_arrays({A(i): [x, y]}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, 1)}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, -1)}, [i, -j]) == Array([[1, 0], [0, 1]])

    # 创建一个 PartialDerivative 对象，计算 A(-i) 对 A(-j) 的偏导数表达式
    expr = PartialDerivative(A(-i), A(-j))
    # 获取偏导数表达式中的自由指标
    assert expr.get_free_indices() == [-i, j]
    # 获取偏导数表达式中的所有指标
    assert expr.get_indices() == [-i, j]
    # 使用指定的替换字典替换偏导数表达式中的数组，并断言结果
    assert expr.replace_with_arrays({A(-i): [x, y]}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, 1)}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, -1)}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, [-i, j]) == Array([[1, 0], [0, 1]])

    # 创建一个 PartialDerivative 对象，计算 A(i) 对 A(i) 的偏导数表达式
    expr = PartialDerivative(A(i), A(i))
    # 获取偏导数表达式中的自由指标
    assert expr.get_free_indices() == []
    # 获取偏导数表达式中的所有指标
    assert expr.get_indices() == [L_0, -L_0]
    # 使用指定的替换字典替换偏导数表达式中的数组，并断言结果
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, []) == 2
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, []) == 2

    # 创建一个 PartialDerivative 对象，计算 A(-i) 对 A(-i) 的偏导数表达式
    expr = PartialDerivative(A(-i), A(-i))
    # 获取偏导数表达式中的自由指标
    assert expr.get_free_indices() == []
    # 获取偏导数表达式中的所有指标
    assert expr.get_indices() == [-L_0, L_0]
    # 使用指定的替换字典替换偏导数表达式中的数组，并断言结果
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, []) == 2
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, []) == 2

    # 创建一个 PartialDerivative 对象，计算 (H(i, j) + H(j, i)) 对 A(i) 的偏导数表达式
    expr = PartialDerivative(H(i, j) + H(j, i), A(i))
    # 获取偏导数表达式中的所有指标
    assert expr.get_indices() == [L_0, j, -L_0]
    # 获取偏导数表达式中的自由指标
    assert expr.get_free_indices() == [j]

    # 创建一个 PartialDerivative 对象，计算 (H(i, j) + H(j, i)) 对 A(k) * B(-i) 的偏导数表达式
    expr = PartialDerivative(H(i, j) + H(j, i), A(k))*B(-i)
    # 获取偏导数表达式中的所有指标
    assert expr.get_indices() == [L_0, j, -k, -L_0]
    # 获取偏导数表达式中的自由指标
    assert expr.get_free_indices() == [j, -k]

    # 创建一个 PartialDerivative 对象，计算 A(i) * (H(-i, j) + H(j, -i)) 对 A(j) 的偏导数表达式
    expr = PartialDerivative(A(i)*(H(-i, j) + H(j, -i)), A(j))
    # 获取偏导数表达式中的所有指标
    assert expr.get_indices() == [L_0, -L_0, L_1, -L_1]
    # 获取偏导数表达式中的自由指标
    assert expr.get_free_indices() == []

    # 添加一个表达式到 A(j)*A(-j) + expr
    expr = A(j)*A(-j) + expr
    # 获取表达式中的所有指标
    assert expr.get_indices() == [L_0, -L_0, L_1, -L_1]
    # 获取表达式中的自由指标
    assert expr.get_free_indices() == []

    # 创建一个复杂的表达式 A(i)*(B(j)*PartialDerivative(C(-j), D(i)) + C(j)*PartialDerivative(D(-j), B(i)))
    expr = A(i)*(B(j)*PartialDerivative(C(-j), D(i)) + C(j)*PartialDerivative(D(-j), B(i)))
    # 获取表达式中的所有指标
    assert expr.get_indices() == [L_0, L_1, -L_1, -L_0]
    # 获取表达式中的自由指标
    assert expr.get_free_indices() == []

    # 创建一个表达式 A(i)*PartialDerivative(C(-j), D(i))
    expr = A(i)*PartialDerivative(C(-j), D(i))
    # 断言：验证表达式对象 `expr` 的索引是否与给定列表相同，期望结果为 [L_0, -j, -L_0]
    assert expr.get_indices() == [L_0, -j, -L_0]
    
    # 断言：验证表达式对象 `expr` 的自由索引是否与给定列表相同，期望结果为 [-j]
    assert expr.get_free_indices() == [-j]
def test_expand_partial_derivative_sum_rule():
    tau = symbols("tau")

    # 创建局部导数表达式 PartialDerivative(A(i), tau)
    expr1aa = PartialDerivative(A(i), tau)

    # 断言表达式的展开结果是否等于原表达式 PartialDerivative(A(i), tau)
    assert expr1aa._expand_partial_derivative() == PartialDerivative(A(i), tau)

    # 创建局部导数表达式 PartialDerivative(A(i) + B(i), tau)
    expr1ab = PartialDerivative(A(i) + B(i), tau)

    # 断言表达式的展开结果是否等于 A(i) 和 B(i) 的部分导数之和
    assert (expr1ab._expand_partial_derivative() ==
            PartialDerivative(A(i), tau) +
            PartialDerivative(B(i), tau))

    # 创建局部导数表达式 PartialDerivative(A(i) + B(i) + C(i), tau)
    expr1ac = PartialDerivative(A(i) + B(i) + C(i), tau)

    # 断言表达式的展开结果是否等于 A(i)、B(i) 和 C(i) 的部分导数之和
    assert (expr1ac._expand_partial_derivative() ==
            PartialDerivative(A(i), tau) +
            PartialDerivative(B(i), tau) +
            PartialDerivative(C(i), tau))

    # 创建局部导数表达式 PartialDerivative(A(i), D(j))
    expr1ba = PartialDerivative(A(i), D(j))

    # 断言表达式的展开结果是否等于原表达式 PartialDerivative(A(i), D(j))
    assert expr1ba._expand_partial_derivative() ==\
        PartialDerivative(A(i), D(j))

    # 创建局部导数表达式 PartialDerivative(A(i) + B(i), D(j))
    expr1bb = PartialDerivative(A(i) + B(i), D(j))

    # 断言表达式的展开结果是否等于 A(i) 和 B(i) 的部分导数之和
    assert (expr1bb._expand_partial_derivative() ==
            PartialDerivative(A(i), D(j)) +
            PartialDerivative(B(i), D(j)))

    # 创建局部导数表达式 PartialDerivative(A(i) + B(i) + C(i), D(j))
    expr1bc = PartialDerivative(A(i) + B(i) + C(i), D(j))

    # 断言表达式的展开结果是否等于 A(i)、B(i) 和 C(i) 的部分导数之和
    assert expr1bc._expand_partial_derivative() ==\
        PartialDerivative(A(i), D(j))\
        + PartialDerivative(B(i), D(j))\
        + PartialDerivative(C(i), D(j))

    # 创建局部导数表达式 PartialDerivative(A(i), H(j, k))
    expr1ca = PartialDerivative(A(i), H(j, k))

    # 断言表达式的展开结果是否等于原表达式 PartialDerivative(A(i), H(j, k))
    assert expr1ca._expand_partial_derivative() ==\
        PartialDerivative(A(i), H(j, k))

    # 创建局部导数表达式 PartialDerivative(A(i) + B(i), H(j, k))
    expr1cb = PartialDerivative(A(i) + B(i), H(j, k))

    # 断言表达式的展开结果是否等于 A(i) 和 B(i) 的部分导数之和
    assert (expr1cb._expand_partial_derivative() ==
            PartialDerivative(A(i), H(j, k))
            + PartialDerivative(B(i), H(j, k)))

    # 创建局部导数表达式 PartialDerivative(A(i) + B(i) + C(i), H(j, k))
    expr1cc = PartialDerivative(A(i) + B(i) + C(i), H(j, k))

    # 断言表达式的展开结果是否等于 A(i)、B(i) 和 C(i) 的部分导数之和
    assert (expr1cc._expand_partial_derivative() ==
            PartialDerivative(A(i), H(j, k))
            + PartialDerivative(B(i), H(j, k))
            + PartialDerivative(C(i), H(j, k)))

    # 创建局部导数表达式 PartialDerivative(A(i), (D(j), H(k, m)))
    expr1da = PartialDerivative(A(i), (D(j), H(k, m)))

    # 断言表达式的展开结果是否等于原表达式 PartialDerivative(A(i), (D(j), H(k, m)))
    assert expr1da._expand_partial_derivative() ==\
        PartialDerivative(A(i), (D(j), H(k, m)))

    # 创建局部导数表达式 PartialDerivative(A(i) + B(i), (D(j), H(k, m)))
    expr1db = PartialDerivative(A(i) + B(i), (D(j), H(k, m)))

    # 断言表达式的展开结果是否等于 A(i) 和 B(i) 的部分导数之和
    assert expr1db._expand_partial_derivative() ==\
        PartialDerivative(A(i), (D(j), H(k, m)))\
        + PartialDerivative(B(i), (D(j), H(k, m)))

    # 创建局部导数表达式 PartialDerivative(A(i) + B(i) + C(i), (D(j), H(k, m)))
    expr1dc = PartialDerivative(A(i) + B(i) + C(i), (D(j), H(k, m)))

    # 断言表达式的展开结果是否等于 A(i)、B(i) 和 C(i) 的部分导数之和
    assert expr1dc._expand_partial_derivative() ==\
        PartialDerivative(A(i), (D(j), H(k, m)))\
        + PartialDerivative(B(i), (D(j), H(k, m)))\
        + PartialDerivative(C(i), (D(j), H(k, m)))


def test_expand_partial_derivative_constant_factor_rule():
    nneg = randint(0, 1000)
    pos = randint(1, 1000)
    neg = -randint(1, 1000)

    # 创建常数系数乘以局部导数表达式 PartialDerivative(nneg*A(i), D(j))
    expr2a = PartialDerivative(nneg*A(i), D(j))

    # 断言表达式的展开结果是否等于常数系数乘以 A(i) 的部分导数
    assert expr2a._expand_partial_derivative() ==\
        nneg*PartialDerivative(A(i), D(j))
    # 创建一个偏导数表达式，对应于 neg * A(i) 对 D(j) 的偏导数
    expr2b = PartialDerivative(neg*A(i), D(j))
    # 断言表达式展开后的结果应为 neg * PartialDerivative(A(i), D(j))
    assert expr2b._expand_partial_derivative() ==\
        neg*PartialDerivative(A(i), D(j))

    # 创建一个偏导数表达式，对应于 c1 * A(i) 对 D(j) 的偏导数
    expr2ca = PartialDerivative(c1*A(i), D(j))
    # 断言表达式展开后的结果应为 c1 * PartialDerivative(A(i), D(j))
    assert expr2ca._expand_partial_derivative() ==\
        c1*PartialDerivative(A(i), D(j))

    # 创建一个偏导数表达式，对应于 c2 * A(i) 对 D(j) 的偏导数
    expr2cb = PartialDerivative(c2*A(i), D(j))
    # 断言表达式展开后的结果应为 c2 * PartialDerivative(A(i), D(j))
    assert expr2cb._expand_partial_derivative() ==\
        c2*PartialDerivative(A(i), D(j))

    # 创建一个偏导数表达式，对应于 c3 * A(i) 对 D(j) 的偏导数
    expr2cc = PartialDerivative(c3*A(i), D(j))
    # 断言表达式展开后的结果应为 c3 * PartialDerivative(A(i), D(j))
    assert expr2cc._expand_partial_derivative() ==\
        c3*PartialDerivative(A(i), D(j))
# 定义测试函数，用于测试偏导数的展开规则
def test_expand_partial_derivative_full_linearity():
    # 生成随机的非负整数
    nneg = randint(0, 1000)
    # 生成随机的正整数
    pos = randint(1, 1000)
    # 生成随机的负整数
    neg = -randint(1, 1000)

    # 创建有理数对象 c1, c2, c3
    c1 = Rational(nneg, pos)
    c2 = Rational(neg, pos)
    c3 = Rational(nneg, neg)

    # 创建一个 PartialDerivative 对象 p，计算其值并断言结果为真
    p = PartialDerivative(42, D(j))
    assert p and not p._expand_partial_derivative()

    # 测试线性展开
    expr3a = PartialDerivative(nneg*A(i) + pos*B(i), D(j))
    assert expr3a._expand_partial_derivative() ==\
        nneg*PartialDerivative(A(i), D(j))\
        + pos*PartialDerivative(B(i), D(j))

    expr3b = PartialDerivative(nneg*A(i) + neg*B(i), D(j))
    assert expr3b._expand_partial_derivative() ==\
        nneg*PartialDerivative(A(i), D(j))\
        + neg*PartialDerivative(B(i), D(j))

    expr3c = PartialDerivative(neg*A(i) + pos*B(i), D(j))
    assert expr3c._expand_partial_derivative() ==\
        neg*PartialDerivative(A(i), D(j))\
        + pos*PartialDerivative(B(i), D(j))

    # 测试带有有理数系数的情况
    expr3d = PartialDerivative(c1*A(i) + c2*B(i), D(j))
    assert expr3d._expand_partial_derivative() ==\
        c1*PartialDerivative(A(i), D(j))\
        + c2*PartialDerivative(B(i), D(j))

    expr3e = PartialDerivative(c2*A(i) + c1*B(i), D(j))
    assert expr3e._expand_partial_derivative() ==\
        c2*PartialDerivative(A(i), D(j))\
        + c1*PartialDerivative(B(i), D(j))

    expr3f = PartialDerivative(c2*A(i) + c3*B(i), D(j))
    assert expr3f._expand_partial_derivative() ==\
        c2*PartialDerivative(A(i), D(j))\
        + c3*PartialDerivative(B(i), D(j))

    expr3g = PartialDerivative(c3*A(i) + c2*B(i), D(j))
    assert expr3g._expand_partial_derivative() ==\
        c3*PartialDerivative(A(i), D(j))\
        + c2*PartialDerivative(B(i), D(j))

    expr3h = PartialDerivative(c3*A(i) + c1*B(i), D(j))
    assert expr3h._expand_partial_derivative() ==\
        c3*PartialDerivative(A(i), D(j))\
        + c1*PartialDerivative(B(i), D(j))

    expr3i = PartialDerivative(c1*A(i) + c3*B(i), D(j))
    assert expr3i._expand_partial_derivative() ==\
        c1*PartialDerivative(A(i), D(j))\
        + c3*PartialDerivative(B(i), D(j))


# 定义测试函数，用于测试偏导数的乘积法则
def test_expand_partial_derivative_product_rule():
    # 测试乘积法则
    expr4a = PartialDerivative(A(i)*B(j), D(k))

    assert expr4a._expand_partial_derivative() == \
        PartialDerivative(A(i), D(k))*B(j)\
        + A(i)*PartialDerivative(B(j), D(k))

    expr4b = PartialDerivative(A(i)*B(j)*C(k), D(m))
    assert expr4b._expand_partial_derivative() ==\
        PartialDerivative(A(i), D(m))*B(j)*C(k)\
        + A(i)*PartialDerivative(B(j), D(m))*C(k)\
        + A(i)*B(j)*PartialDerivative(C(k), D(m))

    expr4c = PartialDerivative(A(i)*B(j), C(k), D(m))
    assert expr4c._expand_partial_derivative() ==\
        PartialDerivative(A(i), C(k), D(m))*B(j) \
        + PartialDerivative(A(i), C(k))*PartialDerivative(B(j), D(m))\
        + PartialDerivative(A(i), D(m))*PartialDerivative(B(j), C(k))\
        + A(i)*PartialDerivative(B(j), C(k), D(m))
# 定义测试函数，用于评估部分导数表达式关于符号的求导

tau, alpha = symbols("tau alpha")
# 定义符号变量 tau 和 alpha

expr1 = PartialDerivative(tau**alpha, tau)
# 创建对 tau**alpha 关于 tau 的部分导数表达式
assert expr1._perform_derivative() == alpha * 1 / tau * tau ** alpha
# 断言部分导数表达式的计算结果是否正确

expr2 = PartialDerivative(2*tau + 3*tau**4, tau)
# 创建对 2*tau + 3*tau**4 关于 tau 的部分导数表达式
assert expr2._perform_derivative() == 2 + 12 * tau ** 3
# 断言部分导数表达式的计算结果是否正确

expr3 = PartialDerivative(2*tau + 3*tau**4, alpha)
# 创建对 2*tau + 3*tau**4 关于 alpha 的部分导数表达式
assert expr3._perform_derivative() == 0
# 断言部分导数表达式的计算结果是否正确


# 定义测试函数，用于评估单个张量关于标量的部分导数表达式

tau, mu = symbols("tau mu")
# 定义符号变量 tau 和 mu

expr = PartialDerivative(tau**mu, tau)
# 创建对 tau**mu 关于 tau 的部分导数表达式
assert expr._perform_derivative() == mu*tau**mu/tau
# 断言部分导数表达式的计算结果是否正确

expr1a = PartialDerivative(A(i), tau)
# 创建对 A(i) 关于 tau 的部分导数表达式
assert expr1a._perform_derivative() == 0
# 断言部分导数表达式的计算结果是否正确

expr1b = PartialDerivative(A(-i), tau)
# 创建对 A(-i) 关于 tau 的部分导数表达式
assert expr1b._perform_derivative() == 0
# 断言部分导数表达式的计算结果是否正确

expr2a = PartialDerivative(H(i, j), tau)
# 创建对 H(i, j) 关于 tau 的部分导数表达式
assert expr2a._perform_derivative() == 0
# 断言部分导数表达式的计算结果是否正确

expr2b = PartialDerivative(H(i, -j), tau)
# 创建对 H(i, -j) 关于 tau 的部分导数表达式
assert expr2b._perform_derivative() == 0
# 断言部分导数表达式的计算结果是否正确

expr2c = PartialDerivative(H(-i, j), tau)
# 创建对 H(-i, j) 关于 tau 的部分导数表达式
assert expr2c._perform_derivative() == 0
# 断言部分导数表达式的计算结果是否正确

expr2d = PartialDerivative(H(-i, -j), tau)
# 创建对 H(-i, -j) 关于 tau 的部分导数表达式
assert expr2d._perform_derivative() == 0
# 断言部分导数表达式的计算结果是否正确


# 定义测试函数，用于评估一阶张量关于张量的部分导数表达式

expr1 = PartialDerivative(A(i), A(j))
# 创建对 A(i) 关于 A(j) 的部分导数表达式
assert expr1._perform_derivative() - L.delta(i, -j) == 0
# 断言部分导数表达式的计算结果是否正确

expr2 = PartialDerivative(A(i), A(-j))
# 创建对 A(i) 关于 A(-j) 的部分导数表达式
assert expr2._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, j) == 0
# 断言部分导数表达式的计算结果是否正确

expr3 = PartialDerivative(A(-i), A(-j))
# 创建对 A(-i) 关于 A(-j) 的部分导数表达式
assert expr3._perform_derivative() - L.delta(-i, j) == 0
# 断言部分导数表达式的计算结果是否正确

expr4 = PartialDerivative(A(-i), A(j))
# 创建对 A(-i) 关于 A(j) 的部分导数表达式
assert expr4._perform_derivative() - L.metric(-i, -L_0) * L.delta(L_0, -j) == 0
# 断言部分导数表达式的计算结果是否正确

expr5 = PartialDerivative(A(i), B(j))
expr6 = PartialDerivative(A(i), C(j))
expr7 = PartialDerivative(A(i), D(j))
expr8 = PartialDerivative(A(i), H(j, k))
# 创建对 A(i) 关于其他张量的部分导数表达式
assert expr5._perform_derivative() == 0
assert expr6._perform_derivative() == 0
assert expr7._perform_derivative() == 0
assert expr8._perform_derivative() == 0
# 断言各部分导数表达式的计算结果是否正确

expr9 = PartialDerivative(A(i), A(i))
# 创建对 A(i) 关于 A(i) 的部分导数表达式
assert expr9._perform_derivative() - L.delta(L_0, -L_0) == 0
# 断言部分导数表达式的计算结果是否正确

expr10 = PartialDerivative(A(-i), A(-i))
# 创建对 A(-i) 关于 A(-i) 的部分导数表达式
assert expr10._perform_derivative() - L.delta(-L_0, L_0) == 0
# 断言部分导数表达式的计算结果是否正确


# 定义测试函数，用于评估二阶张量关于张量的部分导数表达式

expr1 = PartialDerivative(H(i, j), H(m, m1))
# 创建对 H(i, j) 关于 H(m, m1) 的部分导数表达式
assert expr1._perform_derivative() - L.delta(i, -m) * L.delta(j, -m1) == 0
# 断言部分导数表达式的计算结果是否正确

expr2 = PartialDerivative(H(i, j), H(-m, m1))
# 创建对 H(i, j) 关于 H(-m, m1) 的部分导数表达式
assert expr2._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, m) * L.delta(j, -m1) == 0
# 断言部分导数表达式的计算结果是否正确

expr3 = PartialDerivative(H(i, j), H(m, -m1))
# 创建对 H(i, j) 关于 H(m, -m1) 的部分导数表达式
assert expr3._perform_derivative() - L.delta(i, -m) * L.metric(j, L_0) * L.delta(-L_0, m1) == 0
# 断言部分导数表达式的计算结果是否正确

expr4 = PartialDerivative(H(i, j), H(-m, -m1))
# 创建对 H(i, j) 关于 H(-m, -m1) 的部分导数表达式
assert expr4._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, m) * L.metric(j, L_1) * L.delta(-L_1, m1) == 0
# 断言部分导数表达式的计算结果是否正确
    # 创建表达式 expr1a，表示 A(i) 对 A(i) 的偏导数
    expr1a = PartialDerivative(A(i), A(i))
    # 创建表达式 expr1b，表示 A(i) 对 A(k) 的偏导数
    expr1b = PartialDerivative(A(i), A(k))
    # 创建表达式 expr1c，表示 L.delta(-i, k) * A(i) 对 A(k) 的偏导数
    expr1c = PartialDerivative(L.delta(-i, k) * A(i), A(k))

    # 断言：expr1a 对 A(i) 的偏导数减去 L.delta(-i, k) * expr1b 对 A(k) 的偏导数与零张量的收缩等于零
    assert (expr1a._perform_derivative()
            - (L.delta(-i, k) * expr1b._perform_derivative())).contract_delta(L.delta) == 0

    # 断言：expr1a 对 A(i) 的偏导数减去 expr1c 对 A(k) 的偏导数与零张量的收缩等于零
    assert (expr1a._perform_derivative()
            - expr1c._perform_derivative()).contract_delta(L.delta) == 0

    # 创建表达式 expr2a，表示 H(i, j) 对 H(i, j) 的偏导数
    expr2a = PartialDerivative(H(i, j), H(i, j))
    # 创建表达式 expr2b，表示 H(i, j) 对 H(k, m) 的偏导数
    expr2b = PartialDerivative(H(i, j), H(k, m))
    # 创建表达式 expr2c，表示 L.delta(-i, k) * L.delta(-j, m) * H(i, j) 对 H(k, m) 的偏导数
    expr2c = PartialDerivative(L.delta(-i, k) * L.delta(-j, m) * H(i, j), H(k, m))

    # 断言：expr2a 对 H(i, j) 的偏导数减去 L.delta(-i, k) * L.delta(-j, m) * expr2b 对 H(k, m) 的偏导数与零张量的收缩等于零
    assert (expr2a._perform_derivative()
            - (L.delta(-i, k) * L.delta(-j, m) * expr2b._perform_derivative())).contract_delta(L.delta) == 0

    # 断言：expr2a 对 H(i, j) 的偏导数减去 expr2c 对 H(k, m) 的偏导数与零张量的收缩等于零
    assert (expr2a._perform_derivative()
            - expr2c._perform_derivative()).contract_delta(L.delta) == 0
# 定义测试函数，用于评估偏导数表达式1的计算
def test_eval_partial_derivative_expr1():

    # 定义符号变量 tau 和 alpha
    tau, alpha = symbols("tau alpha")

    # 定义基础表达式1，包含多个项
    base_expr1 = A(i)*H(-i, j) + A(i)*A(-i)*A(j) + tau**alpha*A(j)

    # 计算关于张量 H(k, m) 的偏导数
    tensor_derivative = PartialDerivative(base_expr1, H(k, m))._perform_derivative()

    # 计算关于向量 A(k) 的偏导数
    vector_derivative = PartialDerivative(base_expr1, A(k))._perform_derivative()

    # 计算关于标量 tau 的偏导数
    scalar_derivative = PartialDerivative(base_expr1, tau)._perform_derivative()

    # 断言：张量偏导数结果应该为零
    assert (tensor_derivative - A(L_0)*L.metric(-L_0, -L_1)*L.delta(L_1, -k)*L.delta(j, -m)) == 0

    # 断言：向量偏导数结果应该为零
    assert (vector_derivative - (tau**alpha*L.delta(j, -k) +
        L.delta(L_0, -k)*A(-L_0)*A(j) +
        A(L_0)*L.metric(-L_0, -L_1)*L.delta(L_1, -k)*A(j) +
        A(L_0)*A(-L_0)*L.delta(j, -k) +
        L.delta(L_0, -k)*H(-L_0, j))).expand() == 0

    # 断言：向量偏导数经度规和 delta 缩并后的结果应为零
    assert (vector_derivative.contract_metric(L.metric).contract_delta(L.delta) -
        (tau**alpha*L.delta(j, -k) + A(L_0)*A(-L_0)*L.delta(j, -k) + H(-k, j) + 2*A(j)*A(-k))).expand() == 0

    # 断言：标量偏导数结果应为零
    assert scalar_derivative - alpha*1/tau*tau**alpha*A(j) == 0


# 定义测试函数，用于评估混合标量和张量表达式2的偏导数计算
def test_eval_partial_derivative_mixed_scalar_tensor_expr2():

    # 定义符号变量 tau 和 alpha
    tau, alpha = symbols("tau alpha")

    # 定义基础表达式2，包含多个项
    base_expr2 = A(i)*A(-i) + tau**2

    # 计算关于向量 A(k) 的偏导数
    vector_expression = PartialDerivative(base_expr2, A(k))._perform_derivative()

    # 断言：向量偏导数结果应为零
    assert  (vector_expression -
        (L.delta(L_0, -k)*A(-L_0) + A(L_0)*L.metric(-L_0, -L_1)*L.delta(L_1, -k))).expand() == 0

    # 计算关于标量 tau 的偏导数
    scalar_expression = PartialDerivative(base_expr2, tau)._perform_derivative()

    # 断言：标量偏导数结果应为 2*tau
    assert scalar_expression == 2*tau
```