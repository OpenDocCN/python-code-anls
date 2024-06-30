# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_commutator.py`

```
from sympy.core.numbers import Integer  # 导入整数类型 Integer
from sympy.core.symbol import symbols  # 导入符号变量 symbols

from sympy.physics.quantum.dagger import Dagger  # 导入共轭算符 Dagger
from sympy.physics.quantum.commutator import Commutator as Comm  # 导入对易子 Comm
from sympy.physics.quantum.operator import Operator  # 导入算符 Operator


a, b, c = symbols('a,b,c')  # 定义符号变量 a, b, c
n = symbols('n', integer=True)  # 定义整数类型的符号变量 n
A, B, C, D = symbols('A,B,C,D', commutative=False)  # 定义非交换符号变量 A, B, C, D


def test_commutator():
    c = Comm(A, B)  # 计算算符 A 和 B 的对易子，并赋给变量 c
    assert c.is_commutative is False  # 断言 c 不是可交换的
    assert isinstance(c, Comm)  # 断言 c 是 Comm 类的实例
    assert c.subs(A, C) == Comm(C, B)  # 断言将 A 替换为 C 后，与 Comm(C, B) 相等


def test_commutator_identities():
    assert Comm(a*A, b*B) == a*b*Comm(A, B)  # 断言对易子的线性性质
    assert Comm(A, A) == 0  # 断言任意算符与自身的对易子为零
    assert Comm(a, b) == 0  # 断言两个常数的对易子为零
    assert Comm(A, B) == -Comm(B, A)  # 断言对易子的反交换性质
    assert Comm(A, B).doit() == A*B - B*A  # 断言对易子的求值结果
    assert Comm(A, B*C).expand(commutator=True) == Comm(A, B)*C + B*Comm(A, C)  # 断言对易子的分配性质
    assert Comm(A*B, C*D).expand(commutator=True) == \
        A*C*Comm(B, D) + A*Comm(B, C)*D + C*Comm(A, D)*B + Comm(A, C)*D*B  # 断言多个算符的对易子的展开
    assert Comm(A, B**2).expand(commutator=True) == Comm(A, B)*B + B*Comm(A, B)  # 断言对易子与幂次算符的展开
    assert Comm(A**2, C**2).expand(commutator=True) == \
        Comm(A*B, C*D).expand(commutator=True).replace(B, A).replace(D, C) == \
        A*C*Comm(A, C) + A*Comm(A, C)*C + C*Comm(A, C)*A + Comm(A, C)*C*A  # 断言复杂对易子的展开和替换
    assert Comm(A, C**-2).expand(commutator=True) == \
        Comm(A, (1/C)*(1/D)).expand(commutator=True).replace(D, C)  # 断言对易子与倒数的展开和替换
    assert Comm(A + B, C + D).expand(commutator=True) == \
        Comm(A, C) + Comm(A, D) + Comm(B, C) + Comm(B, D)  # 断言对易子的加法性质
    assert Comm(A, B + C).expand(commutator=True) == Comm(A, B) + Comm(A, C)  # 断言对易子的分配性质
    assert Comm(A**n, B).expand(commutator=True) == Comm(A**n, B)  # 断言对易子与幂次算符的展开

    e = Comm(A, Comm(B, C)) + Comm(B, Comm(C, A)) + Comm(C, Comm(A, B))
    assert e.doit().expand() == 0  # 断言复合对易子的求值结果为零


def test_commutator_dagger():
    comm = Comm(A*B, C)  # 计算算符 A*B 和 C 的对易子，并赋给变量 comm
    assert Dagger(comm).expand(commutator=True) == \
        - Comm(Dagger(B), Dagger(C))*Dagger(A) - \
        Dagger(B)*Comm(Dagger(A), Dagger(C))  # 断言算符共轭的对易子的展开结果


class Foo(Operator):

    def _eval_commutator_Bar(self, bar):
        return Integer(0)  # 定义 Foo 类的算符对易子计算方法


class Bar(Operator):
    pass  # 定义 Bar 类


class Tam(Operator):

    def _eval_commutator_Foo(self, foo):
        return Integer(1)  # 定义 Tam 类的算符对易子计算方法


def test_eval_commutator():
    F = Foo('F')  # 创建 Foo 类的实例 F
    B = Bar('B')  # 创建 Bar 类的实例 B
    T = Tam('T')  # 创建 Tam 类的实例 T
    assert Comm(F, B).doit() == 0  # 断言 Foo 类和 Bar 类的对易子结果为零
    assert Comm(B, F).doit() == 0  # 断言 Bar 类和 Foo 类的对易子结果为零
    assert Comm(F, T).doit() == -1  # 断言 Foo 类和 Tam 类的对易子结果为 -1
    assert Comm(T, F).doit() == 1  # 断言 Tam 类和 Foo 类的对易子结果为 1
    assert Comm(B, T).doit() == B*T - T*B  # 断言 Bar 类和 Tam 类的对易子的展开结果
    assert Comm(F**2, B).expand(commutator=True).doit() == 0  # 断言 Foo 类的平方与 Bar 类的对易子的展开结果为零
    assert Comm(F**2, T).expand(commutator=True).doit() == -2*F  # 断言 Foo 类的平方与 Tam 类的对易子的展开结果为 -2*F
    assert Comm(F, T**2).expand(commutator=True).doit() == -2*T  # 断言 Foo 类和 Tam 类的平方的对易子的展开结果为 -2*T
    assert Comm(T**2, F).expand(commutator=True).doit() == 2*T  # 断言 Tam 类的平方和 Foo 类的对易子的展开结果为 2*T
    assert Comm(T**2, F**3).expand(commutator=True).doit() == 2*F*T*F + 2*F**2*T + 2*T*F**2  # 断言复杂对易子的展开结果
```