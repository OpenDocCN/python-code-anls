# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_qexpr.py`

```
# 从 sympy.core.numbers 模块中导入 Integer 类
from sympy.core.numbers import Integer
# 从 sympy.core.symbol 模块中导入 Symbol 类
from sympy.core.symbol import Symbol
# 从 sympy.physics.quantum.qexpr 模块中导入 QExpr 类和 _qsympify_sequence 函数
from sympy.physics.quantum.qexpr import QExpr, _qsympify_sequence
# 从 sympy.physics.quantum.hilbert 模块中导入 HilbertSpace 类
from sympy.physics.quantum.hilbert import HilbertSpace
# 从 sympy.core.containers 模块中导入 Tuple 类
from sympy.core.containers import Tuple

# 创建名为 x 的符号变量
x = Symbol('x')
# 创建名为 y 的符号变量
y = Symbol('y')

# 定义名为 test_qexpr_new 的测试函数
def test_qexpr_new():
    # 创建 QExpr 对象 q，参数为 0
    q = QExpr(0)
    # 断言 q 对象的 label 属性为 (0,)
    assert q.label == (0,)
    # 断言 q 对象的 hilbert_space 属性为 HilbertSpace() 的实例
    assert q.hilbert_space == HilbertSpace()
    # 断言 q 对象的 is_commutative 属性为 False
    assert q.is_commutative is False

    # 创建 QExpr 对象 q，参数为 0, 1
    q = QExpr(0, 1)
    # 断言 q 对象的 label 属性为 (Integer(0), Integer(1))
    assert q.label == (Integer(0), Integer(1))

    # 调用 QExpr._new_rawargs 方法，参数为 HilbertSpace(), Integer(0), Integer(1)
    q = QExpr._new_rawargs(HilbertSpace(), Integer(0), Integer(1))
    # 断言 q 对象的 label 属性为 (Integer(0), Integer(1))
    assert q.label == (Integer(0), Integer(1))
    # 断言 q 对象的 hilbert_space 属性为 HilbertSpace() 的实例
    assert q.hilbert_space == HilbertSpace()


# 定义名为 test_qexpr_commutative 的测试函数
def test_qexpr_commutative():
    # 创建 QExpr 对象 q1，参数为 x
    q1 = QExpr(x)
    # 创建 QExpr 对象 q2，参数为 y
    q2 = QExpr(y)
    # 断言 q1 对象的 is_commutative 属性为 False
    assert q1.is_commutative is False
    # 断言 q2 对象的 is_commutative 属性为 False
    assert q2.is_commutative is False
    # 断言 q1*q2 不等于 q2*q1
    assert q1*q2 != q2*q1

    # 调用 QExpr._new_rawargs 方法，参数为 Integer(0), Integer(1), HilbertSpace()
    q = QExpr._new_rawargs(Integer(0), Integer(1), HilbertSpace())
    # 断言 q 对象的 is_commutative 属性为 False
    assert q.is_commutative is False


# 定义名为 test_qexpr_commutative_free_symbols 的测试函数
def test_qexpr_commutative_free_symbols():
    # 创建 QExpr 对象 q1，参数为 x
    q1 = QExpr(x)
    # 弹出 q1 对象的 free_symbols 集合中的元素，并断言其 is_commutative 属性为 False
    assert q1.free_symbols.pop().is_commutative is False

    # 创建 QExpr 对象 q2，参数为 'q2'
    q2 = QExpr('q2')
    # 弹出 q2 对象的 free_symbols 集合中的元素，并断言其 is_commutative 属性为 False
    assert q2.free_symbols.pop().is_commutative is False


# 定义名为 test_qexpr_subs 的测试函数
def test_qexpr_subs():
    # 创建 QExpr 对象 q1，参数为 x, y
    q1 = QExpr(x, y)
    # 断言对 q1 对象执行 subs(x, y) 操作后得到的结果与 QExpr(y, y) 相等
    assert q1.subs(x, y) == QExpr(y, y)
    # 断言对 q1 对象执行 subs({x: 1, y: 2}) 操作后得到的结果与 QExpr(1, 2) 相等
    assert q1.subs({x: 1, y: 2}) == QExpr(1, 2)


# 定义名为 test_qsympify 的测试函数
def test_qsympify():
    # 断言 _qsympify_sequence([[1, 2], [1, 3]]) 的结果为 (Tuple(1, 2), Tuple(1, 3))
    assert _qsympify_sequence([[1, 2], [1, 3]]) == (Tuple(1, 2), Tuple(1, 3))
    # 断言 _qsympify_sequence(([1, 2, [3, 4, [2, ]], 1], 3)) 的结果为 (Tuple(1, 2, Tuple(3, 4, Tuple(2,)), 1), 3)
    assert _qsympify_sequence(([1, 2, [3, 4, [2, ]], 1], 3)) == \
        (Tuple(1, 2, Tuple(3, 4, Tuple(2,)), 1), 3)
    # 断言 _qsympify_sequence((1,)) 的结果为 (1,)
    assert _qsympify_sequence((1,)) == (1,)
```