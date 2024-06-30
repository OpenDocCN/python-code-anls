# `D:\src\scipysrc\sympy\sympy\sandbox\indexed_integrals.py`

```
# 从 sympy.tensor 模块导入 Indexed 类，用于处理带索引的变量
from sympy.tensor import Indexed
# 从 sympy.core.containers 模块导入 Tuple 类，用于处理元组类型
from sympy.core.containers import Tuple
# 从 sympy.core.symbol 模块导入 Dummy 类，用于创建虚拟符号
from sympy.core.symbol import Dummy
# 从 sympy.core.sympify 模块导入 sympify 函数，用于将输入转换为 SymPy 对象
from sympy.core.sympify import sympify
# 从 sympy.integrals.integrals 模块导入 Integral 类，用于处理积分运算
from sympy.integrals.integrals import Integral


class IndexedIntegral(Integral):
    """
    Experimental class to test integration by indexed variables.

    Usage is analogue to ``Integral``, it simply adds awareness of
    integration over indices.

    Contraction of non-identical index symbols referring to the same
    ``IndexedBase`` is not yet supported.

    Examples
    ========

    >>> from sympy.sandbox.indexed_integrals import IndexedIntegral
    >>> from sympy import IndexedBase, symbols
    >>> A = IndexedBase('A')
    >>> i, j = symbols('i j', integer=True)
    >>> ii = IndexedIntegral(A[i], A[i])
    >>> ii
    Integral(_A[i], _A[i])
    >>> ii.doit()
    A[i]**2/2

    If the indices are different, indexed objects are considered to be
    different variables:

    >>> i2 = IndexedIntegral(A[j], A[i])
    >>> i2
    Integral(A[j], _A[i])
    >>> i2.doit()
    A[i]*A[j]
    """

    def __new__(cls, function, *limits, **assumptions):
        # 处理限制条件，将涉及索引的限制条件转换成虚拟符号替换字典
        repl, limits = IndexedIntegral._indexed_process_limits(limits)
        # 将输入函数转换为 SymPy 对象
        function = sympify(function)
        # 使用虚拟符号替换字典替换函数中的 Indexed 对象
        function = function.xreplace(repl)
        # 调用 Integral 类的构造函数创建对象
        obj = Integral.__new__(cls, function, *limits, **assumptions)
        # 记录虚拟符号替换字典
        obj._indexed_repl = repl
        # 创建反向的虚拟符号替换字典
        obj._indexed_reverse_repl = {val: key for key, val in repl.items()}
        return obj

    def doit(self):
        # 调用父类 Integral 的 doit 方法执行积分运算
        res = super().doit()
        # 使用反向虚拟符号替换字典，替换积分结果中的虚拟符号为 Indexed 对象
        return res.xreplace(self._indexed_reverse_repl)

    @staticmethod
    def _indexed_process_limits(limits):
        # 初始化虚拟符号替换字典和新的限制条件列表
        repl = {}
        newlimits = []
        # 遍历限制条件列表
        for i in limits:
            # 检查限制条件是否为元组或列表
            if isinstance(i, (tuple, list, Tuple)):
                v = i[0]  # 取出第一个元素作为变量
                vrest = i[1:]  # 取出剩余元素作为其余限制条件
            else:
                v = i  # 将限制条件作为变量
                vrest = ()  # 其余限制条件为空元组
            # 检查变量是否为 Indexed 对象
            if isinstance(v, Indexed):
                # 如果该 Indexed 对象还未在替换字典中，为其创建新的虚拟符号
                if v not in repl:
                    r = Dummy(str(v))
                    repl[v] = r
                # 将新的限制条件加入到新的限制条件列表中，用新的虚拟符号替换 Indexed 对象
                newlimits.append((r,) + vrest)
            else:
                # 如果变量不是 Indexed 对象，直接加入到新的限制条件列表中
                newlimits.append(i)
        # 返回虚拟符号替换字典和处理后的限制条件列表
        return repl, newlimits
```