# `D:\src\scipysrc\sympy\sympy\physics\quantum\commutator.py`

```
"""The commutator: [A,B] = A*B - B*A."""

# 导入所需模块和类
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm

# 导入量子力学相关模块
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import Operator

# 设置该模块可以导出的公共接口
__all__ = [
    'Commutator'
]

#-----------------------------------------------------------------------------
# Commutator
#-----------------------------------------------------------------------------


class Commutator(Expr):
    """The standard commutator, in an unevaluated state.

    Explanation
    ===========

    Evaluating a commutator is defined [1]_ as: ``[A, B] = A*B - B*A``. This
    class returns the commutator in an unevaluated form. To evaluate the
    commutator, use the ``.doit()`` method.

    Canonical ordering of a commutator is ``[A, B]`` for ``A < B``. The
    arguments of the commutator are put into canonical order using ``__cmp__``.
    If ``B < A``, then ``[B, A]`` is returned as ``-[A, B]``.

    Parameters
    ==========

    A : Expr
        The first argument of the commutator [A,B].
    B : Expr
        The second argument of the commutator [A,B].

    Examples
    ========

    >>> from sympy.physics.quantum import Commutator, Dagger, Operator
    >>> from sympy.abc import x, y
    >>> A = Operator('A')
    >>> B = Operator('B')
    >>> C = Operator('C')

    Create a commutator and use ``.doit()`` to evaluate it:

    >>> comm = Commutator(A, B)
    >>> comm
    [A,B]
    >>> comm.doit()
    A*B - B*A

    The commutator orders it arguments in canonical order:

    >>> comm = Commutator(B, A); comm
    -[A,B]

    Commutative constants are factored out:

    >>> Commutator(3*x*A, x*y*B)
    3*x**2*y*[A,B]

    Using ``.expand(commutator=True)``, the standard commutator expansion rules
    can be applied:

    >>> Commutator(A+B, C).expand(commutator=True)
    [A,C] + [B,C]
    >>> Commutator(A, B+C).expand(commutator=True)
    [A,B] + [A,C]
    >>> Commutator(A*B, C).expand(commutator=True)
    [A,C]*B + A*[B,C]
    >>> Commutator(A, B*C).expand(commutator=True)
    [A,B]*C + B*[A,C]

    Adjoint operations applied to the commutator are properly applied to the
    arguments:

    >>> Dagger(Commutator(A, B))
    -[Dagger(A),Dagger(B)]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Commutator
    """
    
    # 设置该类为非交换类
    is_commutative = False

    # 定义构造函数，创建一个未评估的交换子对象
    def __new__(cls, A, B):
        r = cls.eval(A, B)
        if r is not None:
            return r
        obj = Expr.__new__(cls, A, B)
        return obj

    @classmethod
    # 定义一个类方法 eval，用于计算两个表达式 a 和 b 的 commutator（交换子）
    def eval(cls, a, b):
        # 如果 a 或 b 为空，则返回零
        if not (a and b):
            return S.Zero
        # 如果 a 等于 b，则返回零
        if a == b:
            return S.Zero
        # 如果 a 或 b 是可交换的，则返回零
        if a.is_commutative or b.is_commutative:
            return S.Zero

        # 分解 a 和 b 成可交换部分和不可交换部分
        ca, nca = a.args_cnc()
        cb, ncb = b.args_cnc()
        c_part = ca + cb
        # 如果存在可交换部分，则计算 commutator 的乘积
        if c_part:
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))

        # 规范化参数顺序
        # 如果 a 比 b 小，则返回负一乘以 commutator 的计算结果
        if a.compare(b) == 1:
            return S.NegativeOne * cls(b, a)

    # 定义一个私有方法 _expand_pow，用于处理幂次展开
    def _expand_pow(self, A, B, sign):
        # 获取幂次
        exp = A.exp
        # 如果幂次不是整数或不是常数或绝对值小于等于 1，则直接返回
        if not exp.is_integer or not exp.is_constant() or abs(exp) <= 1:
            return self
        # 获取基数
        base = A.base
        # 如果幂次为负数，则将基数改为其倒数，幂次改为正数
        if exp.is_negative:
            base = A.base ** -1
            exp = -exp
        # 计算 commutator
        comm = Commutator(base, B).expand(commutator=True)

        # 计算幂次展开的结果
        result = base ** (exp - 1) * comm
        for i in range(1, exp):
            result += base ** (exp - 1 - i) * comm * base ** i
        return sign * result.expand()
    # 定义一个方法用于计算展开交换子的表达式
    def _eval_expand_commutator(self, **hints):
        # 获取第一个参数
        A = self.args[0]
        # 获取第二个参数
        B = self.args[1]

        # 如果第一个参数是加法表达式
        if isinstance(A, Add):
            # [A + B, C]  ->  [A, C] + [B, C]
            sargs = []
            # 遍历加法表达式中的每个项
            for term in A.args:
                # 计算交换子 [term, B]
                comm = Commutator(term, B)
                # 如果交换子仍然是 Commutator 类型，则递归展开
                if isinstance(comm, Commutator):
                    comm = comm._eval_expand_commutator()
                # 将计算得到的交换子加入到列表中
                sargs.append(comm)
            # 返回展开后的加法表达式
            return Add(*sargs)
        # 如果第二个参数是加法表达式
        elif isinstance(B, Add):
            # [A, B + C]  ->  [A, B] + [A, C]
            sargs = []
            # 遍历加法表达式中的每个项
            for term in B.args:
                # 计算交换子 [A, term]
                comm = Commutator(A, term)
                # 如果交换子仍然是 Commutator 类型，则递归展开
                if isinstance(comm, Commutator):
                    comm = comm._eval_expand_commutator()
                # 将计算得到的交换子加入到列表中
                sargs.append(comm)
            # 返回展开后的加法表达式
            return Add(*sargs)
        # 如果第一个参数是乘法表达式
        elif isinstance(A, Mul):
            # [A*B, C] -> A*[B, C] + [A, C]*B
            # 提取乘法表达式的因子
            a = A.args[0]
            b = Mul(*A.args[1:])
            c = B
            # 计算交换子 [B, C] 和 [A, C]
            comm1 = Commutator(b, c)
            comm2 = Commutator(a, c)
            # 如果交换子仍然是 Commutator 类型，则递归展开
            if isinstance(comm1, Commutator):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, Commutator):
                comm2 = comm2._eval_expand_commutator()
            # 计算展开后的乘法表达式
            first = Mul(a, comm1)
            second = Mul(comm2, b)
            return Add(first, second)
        # 如果第二个参数是乘法表达式
        elif isinstance(B, Mul):
            # [A, B*C] -> [A, B]*C + B*[A, C]
            # 提取乘法表达式的因子
            a = A
            b = B.args[0]
            c = Mul(*B.args[1:])
            # 计算交换子 [A, B] 和 [A, C]
            comm1 = Commutator(a, b)
            comm2 = Commutator(a, c)
            # 如果交换子仍然是 Commutator 类型，则递归展开
            if isinstance(comm1, Commutator):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, Commutator):
                comm2 = comm2._eval_expand_commutator()
            # 计算展开后的乘法表达式
            first = Mul(comm1, c)
            second = Mul(b, comm2)
            return Add(first, second)
        # 如果第一个参数是幂次表达式
        elif isinstance(A, Pow):
            # [A**n, C] -> A**(n - 1)*[A, C] + A**(n - 2)*[A, C]*A + ... + [A, C]*A**(n-1)
            # 调用内部方法展开幂次表达式的交换子
            return self._expand_pow(A, B, 1)
        # 如果第二个参数是幂次表达式
        elif isinstance(B, Pow):
            # [A, C**n] -> C**(n - 1)*[C, A] + C**(n - 2)*[C, A]*C + ... + [C, A]*C**(n-1)
            # 调用内部方法展开幂次表达式的交换子
            return self._expand_pow(B, A, -1)

        # 如果没有改变，直接返回自身
        return self

    # 定义一个方法用于执行交换子运算
    def doit(self, **hints):
        """ Evaluate commutator """
        # 获取第一个参数
        A = self.args[0]
        # 获取第二个参数
        B = self.args[1]
        # 如果两个参数都是操作符类型
        if isinstance(A, Operator) and isinstance(B, Operator):
            try:
                # 尝试计算交换子 [A, B]
                comm = A._eval_commutator(B, **hints)
            except NotImplementedError:
                try:
                    # 如果失败，则尝试计算交换子 [-B, A]
                    comm = -1*B._eval_commutator(A, **hints)
                except NotImplementedError:
                    # 如果都失败，返回 None
                    comm = None
            # 如果计算成功，继续进行 doit 操作
            if comm is not None:
                return comm.doit(**hints)
        # 如果不满足上述条件，返回默认交换子计算公式的结果
        return (A*B - B*A).doit(**hints)

    # 定义一个方法用于计算伴随交换子
    def _eval_adjoint(self):
        # 返回 A 和 B 的伴随交换子
        return Commutator(Dagger(self.args[1]), Dagger(self.args[0]))
    # 返回一个字符串表示，表示对象的类名和其两个参数的打印形式
    def _sympyrepr(self, printer, *args):
        return "%s(%s,%s)" % (
            self.__class__.__name__, printer._print(
                self.args[0]), printer._print(self.args[1])
        )

    # 返回一个字符串表示，表示对象两个参数的打印形式
    def _sympystr(self, printer, *args):
        return "[%s,%s]" % (
            printer._print(self.args[0]), printer._print(self.args[1]))

    # 返回一个漂亮的打印形式，使用打印机对象打印第一个参数，并用逗号分隔第二个参数
    def _pretty(self, printer, *args):
        pform = printer._print(self.args[0], *args)
        pform = prettyForm(*pform.right(prettyForm(',')))
        pform = prettyForm(*pform.right(printer._print(self.args[1], *args)))
        pform = prettyForm(*pform.parens(left='[', right=']'))
        return pform

    # 返回 LaTeX 格式的字符串表示，使用打印机对象打印所有参数
    def _latex(self, printer, *args):
        return "\\left[%s,%s\\right]" % tuple([
            printer._print(arg, *args) for arg in self.args])
```