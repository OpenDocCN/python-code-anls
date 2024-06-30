# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_domainscalar.py`

```
from sympy.testing.pytest import raises  # 导入 raises 函数，用于测试是否抛出指定异常

from sympy.core.symbol import S  # 导入 S 符号
from sympy.polys import ZZ, QQ  # 导入整数和有理数环
from sympy.polys.matrices.domainscalar import DomainScalar  # 导入域标量类
from sympy.polys.matrices.domainmatrix import DomainMatrix  # 导入域矩阵类


def test_DomainScalar___new__():
    raises(TypeError, lambda: DomainScalar(ZZ(1), QQ))  # 断言调用 DomainScalar 构造函数会抛出 TypeError 异常
    raises(TypeError, lambda: DomainScalar(ZZ(1), 1))   # 断言调用 DomainScalar 构造函数会抛出 TypeError 异常


def test_DomainScalar_new():
    A = DomainScalar(ZZ(1), ZZ)  # 创建一个域标量对象 A
    B = A.new(ZZ(4), ZZ)         # 使用 A 调用 new 方法创建新的域标量对象 B
    assert B == DomainScalar(ZZ(4), ZZ)  # 断言 B 等于指定的域标量对象


def test_DomainScalar_repr():
    A = DomainScalar(ZZ(1), ZZ)  # 创建一个域标量对象 A
    assert repr(A) in {'1', 'mpz(1)'}  # 断言 A 的 repr 字符串在指定的集合中


def test_DomainScalar_from_sympy():
    expr = S(1)  # 创建一个 SymPy 的符号表达式
    B = DomainScalar.from_sympy(expr)  # 调用 from_sympy 方法创建一个域标量对象 B
    assert B == DomainScalar(ZZ(1), ZZ)  # 断言 B 等于指定的域标量对象


def test_DomainScalar_to_sympy():
    B = DomainScalar(ZZ(1), ZZ)  # 创建一个域标量对象 B
    expr = B.to_sympy()  # 调用 to_sympy 方法将域标量对象转换为 SymPy 符号表达式
    assert expr.is_Integer and expr == 1  # 断言 expr 是整数且等于 1


def test_DomainScalar_to_domain():
    A = DomainScalar(ZZ(1), ZZ)  # 创建一个域标量对象 A
    B = A.to_domain(QQ)  # 调用 to_domain 方法将 A 转换为指定域 QQ 的域标量对象 B
    assert B == DomainScalar(QQ(1), QQ)  # 断言 B 等于指定的域标量对象


def test_DomainScalar_convert_to():
    A = DomainScalar(ZZ(1), ZZ)  # 创建一个域标量对象 A
    B = A.convert_to(QQ)  # 调用 convert_to 方法将 A 转换为指定域 QQ 的域标量对象 B
    assert B == DomainScalar(QQ(1), QQ)  # 断言 B 等于指定的域标量对象


def test_DomainScalar_unify():
    A = DomainScalar(ZZ(1), ZZ)  # 创建一个域标量对象 A
    B = DomainScalar(QQ(2), QQ)  # 创建一个域标量对象 B
    A, B = A.unify(B)  # 调用 unify 方法统一 A 和 B 的域
    assert A.domain == B.domain == QQ  # 断言 A 和 B 的域均为 QQ


def test_DomainScalar_add():
    A = DomainScalar(ZZ(1), ZZ)  # 创建一个域标量对象 A
    B = DomainScalar(QQ(2), QQ)  # 创建一个域标量对象 B
    assert A + B == DomainScalar(QQ(3), QQ)  # 断言 A 加 B 的结果等于指定的域标量对象

    raises(TypeError, lambda: A + 1.5)  # 断言调用 A + 1.5 会抛出 TypeError 异常


def test_DomainScalar_sub():
    A = DomainScalar(ZZ(1), ZZ)  # 创建一个域标量对象 A
    B = DomainScalar(QQ(2), QQ)  # 创建一个域标量对象 B
    assert A - B == DomainScalar(QQ(-1), QQ)  # 断言 A 减 B 的结果等于指定的域标量对象

    raises(TypeError, lambda: A - 1.5)  # 断言调用 A - 1.5 会抛出 TypeError 异常


def test_DomainScalar_mul():
    A = DomainScalar(ZZ(1), ZZ)  # 创建一个域标量对象 A
    B = DomainScalar(QQ(2), QQ)  # 创建一个域标量对象 B
    dm = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)  # 创建一个域矩阵对象 dm
    assert A * B == DomainScalar(QQ(2), QQ)  # 断言 A 乘 B 的结果等于指定的域标量对象
    assert A * dm == dm  # 断言 A 乘 dm 的结果等于 dm
    assert B * 2 == DomainScalar(QQ(4), QQ)  # 断言 B 乘以 2 的结果等于指定的域标量对象

    raises(TypeError, lambda: A * 1.5)  # 断言调用 A * 1.5 会抛出 TypeError 异常


def test_DomainScalar_floordiv():
    A = DomainScalar(ZZ(-5), ZZ)  # 创建一个域标量对象 A
    B = DomainScalar(QQ(2), QQ)  # 创建一个域标量对象 B
    assert A // B == DomainScalar(QQ(-5, 2), QQ)  # 断言 A 地板除以 B 的结果等于指定的域标量对象
    C = DomainScalar(ZZ(2), ZZ)  # 创建一个域标量对象 C
    assert A // C == DomainScalar(ZZ(-3), ZZ)  # 断言 A 地板除以 C 的结果等于指定的域标量对象

    raises(TypeError, lambda: A // 1.5)  # 断言调用 A // 1.5 会抛出 TypeError 异常


def test_DomainScalar_mod():
    A = DomainScalar(ZZ(5), ZZ)  # 创建一个域标量对象 A
    B = DomainScalar(QQ(2), QQ)  # 创建一个域标量对象 B
    assert A % B == DomainScalar(QQ(0), QQ)  # 断言 A 取模 B 的结果等于指定的域标量对象
    C = DomainScalar(ZZ(2), ZZ)  # 创建一个域标量对象 C
    assert A % C == DomainScalar(ZZ(1), ZZ)  # 断言 A 取模 C 的结果等于指定的域标量对象

    raises(TypeError, lambda: A % 1.5)  # 断言调用 A % 1.5 会抛出 TypeError 异常


def test_DomainScalar_divmod():
    A = DomainScalar(ZZ(5), ZZ)  # 创建一个域标量对象 A
    B = DomainScalar(QQ(2), QQ)  # 创建一个域标量对象 B
    assert divmod(A, B) == (DomainScalar(QQ(5, 2), QQ), DomainScalar(QQ(0), QQ))  # 断言 A 和 B 的 divmod 结果等于指定的域标量对象组成的元组
    C = DomainScalar(ZZ(2), ZZ)  # 创建一个域标量对象 C
    assert divmod(A, C) == (DomainScalar(ZZ(2), ZZ), DomainScalar(ZZ(1), ZZ))  # 断言 A 和 C 的 divmod 结果等于指定的域标量对象组成的元组

    raises(TypeError, lambda: divmod(A, 1.5))  # 断言调用 divmod(A, 1.5) 会抛出 TypeError 异常


def test_DomainScalar_pow():
    A = DomainScalar(ZZ(-5), ZZ)  # 创建一个域标量对象 A
    B = A**(2)  # 对 A 进行指数运算
    assert B == DomainScalar(ZZ(25), ZZ)  # 断言 B 等于指定的域标
# 定义测试函数，用于测试 DomainScalar 类的正数加法功能
def test_DomainScalar_pos():
    # 创建两个 DomainScalar 对象 A 和 B，分别赋值为 QQ(2)
    A = DomainScalar(QQ(2), QQ)
    B = DomainScalar(QQ(2), QQ)
    # 断言 A 的正数值应该等于 B
    assert +A == B


# 定义测试函数，用于测试 DomainScalar 类的负数取反功能
def test_DomainScalar_neg():
    # 创建两个 DomainScalar 对象 A 和 B，分别赋值为 QQ(2) 和 QQ(-2)
    A = DomainScalar(QQ(2), QQ)
    B = DomainScalar(QQ(-2), QQ)
    # 断言 A 的负数值应该等于 B
    assert -A == B


# 定义测试函数，用于测试 DomainScalar 类的相等性判断功能
def test_DomainScalar_eq():
    # 创建 DomainScalar 对象 A，赋值为 QQ(2)
    A = DomainScalar(QQ(2), QQ)
    # 断言 A 应该等于自身
    assert A == A
    # 创建 DomainScalar 对象 B，赋值为 ZZ(-5)
    B = DomainScalar(ZZ(-5), ZZ)
    # 断言 A 应该不等于 B
    assert A != B
    # 创建 DomainScalar 对象 C，赋值为 ZZ(2)
    C = DomainScalar(ZZ(2), ZZ)
    # 断言 A 应该不等于 C
    assert A != C
    # 创建一个普通列表 D，断言 A 应该不等于 D
    D = [1]
    assert A != D


# 定义测试函数，用于测试 DomainScalar 类的零判断功能
def test_DomainScalar_isZero():
    # 创建 DomainScalar 对象 A，赋值为 ZZ(0)
    A = DomainScalar(ZZ(0), ZZ)
    # 断言 A 的 is_zero() 方法应返回 True
    assert A.is_zero() == True
    # 创建 DomainScalar 对象 B，赋值为 ZZ(1)
    B = DomainScalar(ZZ(1), ZZ)
    # 断言 B 的 is_zero() 方法应返回 False
    assert B.is_zero() == False


# 定义测试函数，用于测试 DomainScalar 类的单位元判断功能
def test_DomainScalar_isOne():
    # 创建 DomainScalar 对象 A，赋值为 ZZ(1)
    A = DomainScalar(ZZ(1), ZZ)
    # 断言 A 的 is_one() 方法应返回 True
    assert A.is_one() == True
    # 创建 DomainScalar 对象 B，赋值为 ZZ(0)
    B = DomainScalar(ZZ(0), ZZ)
    # 断言 B 的 is_one() 方法应返回 False
    assert B.is_one() == False
```