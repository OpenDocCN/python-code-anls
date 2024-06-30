# `D:\src\scipysrc\sympy\sympy\physics\quantum\trace.py`

```
# 从 sympy.core.add 模块导入 Add 类
# 用于表示 Sympy 表达式中的加法操作
from sympy.core.add import Add

# 从 sympy.core.containers 模块导入 Tuple 类
# 用于表示 Sympy 中的元组容器
from sympy.core.containers import Tuple

# 从 sympy.core.expr 模块导入 Expr 类
# 用于表示 Sympy 中的表达式
from sympy.core.expr import Expr

# 从 sympy.core.mul 模块导入 Mul 类
# 用于表示 Sympy 表达式中的乘法操作
from sympy.core.mul import Mul

# 从 sympy.core.power 模块导入 Pow 类
# 用于表示 Sympy 表达式中的指数操作
from sympy.core.power import Pow

# 从 sympy.core.sorting 模块导入 default_sort_key 函数
# 用于排序 Sympy 对象的默认键
from sympy.core.sorting import default_sort_key

# 从 sympy.core.sympify 模块导入 sympify 函数
# 用于将输入转换为 Sympy 表达式
from sympy.core.sympify import sympify

# 从 sympy.matrices 模块导入 Matrix 类
# 用于表示 Sympy 中的矩阵对象
from sympy.matrices import Matrix


def _is_scalar(e):
    """ Helper method used in Tr

    Explanation
    ===========
    
    This function checks if the input expression 'e' is a scalar.
    It uses sympify to ensure 'e' is converted to a Sympy expression.
    Returns True if 'e' is an integer, float, rational, number,
    or a commutative symbol; otherwise, returns False.

    """

    e = sympify(e)
    if isinstance(e, Expr):
        if (e.is_Integer or e.is_Float or
            e.is_Rational or e.is_Number or
            (e.is_Symbol and e.is_commutative)
                ):
            return True

    return False


def _cycle_permute(l):
    """ Cyclic permutations based on canonical ordering

    Explanation
    ===========

    This method performs cyclic permutations on the list 'l' based
    on canonical ordering using ASCII values. It duplicates 'l' for
    easier processing, finds the minimum item, and creates sublists
    based on occurrences of the minimum item. It compares these
    sublists lexicographically to determine the ordered list 'ordered_l'.

    TODO: Handle condition such as symbols have subscripts/superscripts
    in case of lexicographic sort

    """

    if len(l) == 1:
        return l

    min_item = min(l, key=default_sort_key)
    indices = [i for i, x in enumerate(l) if x == min_item]

    le = list(l)
    le.extend(l)  # duplicate and extend string for easy processing

    indices.append(len(l) + indices[0])

    sublist = [[le[indices[i]:indices[i + 1]]] for i in
               range(len(indices) - 1)]

    idx = sublist.index(min(sublist))
    ordered_l = le[indices[idx]:indices[idx] + len(l)]

    return ordered_l


def _rearrange_args(l):
    """ this just moves the last arg to first position
     to enable expansion of args
     A,B,A ==> A**2,B

    Explanation
    ===========

    This function rearranges the list 'l' such that the last argument
    is moved to the first position. It facilitates the expansion of
    arguments in a mathematical expression. Returns the rearranged list.

    """

    if len(l) == 1:
        return l

    x = list(l[-1:])
    x.extend(l[0:-1])
    return Mul(*x).args


class Tr(Expr):
    """ Generic Trace operation than can trace over:

    a) SymPy matrix
    b) operators
    c) outer products

    Parameters
    ==========
    o : operator, matrix, expr
    i : tuple/list indices (optional)

    Examples
    ========

    # TODO: Need to handle printing

    a) Trace(A+B) = Tr(A) + Tr(B)
    b) Trace(scalar*Operator) = scalar*Trace(Operator)

    >>> from sympy.physics.quantum.trace import Tr
    >>> from sympy import symbols, Matrix
    >>> a, b = symbols('a b', commutative=True)
    >>> A, B = symbols('A B', commutative=False)
    >>> Tr(a*A,[2])
    a*Tr(A)
    >>> m = Matrix([[1,2],[1,1]])
    >>> Tr(m)
    2

    """
    def __new__(cls, *args):
        """ 构造一个 Trace 对象。

        Parameters
        ==========
        args = SymPy 表达式
        indices = 如果有索引，可以是元组或列表，可选的

        """

        # 如果参数长度为 2
        if (len(args) == 2):
            # 如果第二个参数不是列表、元组或 Tuple 类型，则将其转换为 Tuple
            if not isinstance(args[1], (list, Tuple, tuple)):
                indices = Tuple(args[1])
            else:
                indices = Tuple(*args[1])

            expr = args[0]
        # 如果参数长度为 1
        elif (len(args) == 1):
            indices = Tuple()  # 索引为空元组
            expr = args[0]
        else:
            # 抛出错误，Tr 的参数应为形如 (expr[, [indices]]) 的形式
            raise ValueError("Arguments to Tr should be of form "
                             "(expr[, [indices]])")

        # 根据表达式的类型进行不同的处理
        if isinstance(expr, Matrix):
            return expr.trace()  # 对于 Matrix 类型，返回其迹
        elif hasattr(expr, 'trace') and callable(expr.trace):
            # 对于具有 trace() 方法的对象（如 numpy），返回其迹
            return expr.trace()
        elif isinstance(expr, Add):
            # 对于加法表达式，递归地计算每个子表达式的 Trace
            return Add(*[Tr(arg, indices) for arg in expr.args])
        elif isinstance(expr, Mul):
            c_part, nc_part = expr.args_cnc()
            if len(nc_part) == 0:
                return Mul(*c_part)  # 如果非交换部分为空，则返回交换部分的乘积
            else:
                # 创建一个新的 Trace 对象，处理非交换部分
                obj = Expr.__new__(cls, Mul(*nc_part), indices)
                # 避免即使 len(c_part)==0 也返回缓存实例的情况
                return Mul(*c_part) * obj if len(c_part) > 0 else obj
        elif isinstance(expr, Pow):
            if (_is_scalar(expr.args[0]) and
                    _is_scalar(expr.args[1])):
                return expr  # 如果是幂操作且两个参数都是标量，则返回原表达式
            else:
                # 创建一个新的 Trace 对象，处理幂操作
                return Expr.__new__(cls, expr, indices)
        else:
            if (_is_scalar(expr)):
                return expr  # 如果表达式是标量，则直接返回

            # 创建一个新的 Trace 对象，处理一般表达式
            return Expr.__new__(cls, expr, indices)

    @property
    def kind(self):
        expr = self.args[0]
        expr_kind = expr.kind
        return expr_kind.element_kind  # 返回表达式的元素类型

    def doit(self, **hints):
        """ 执行迹运算。

        # TODO: 当前版本忽略了为部分迹设置的索引。

        >>> from sympy.physics.quantum.trace import Tr
        >>> from sympy.physics.quantum.operator import OuterProduct
        >>> from sympy.physics.quantum.spin import JzKet, JzBra
        >>> t = Tr(OuterProduct(JzKet(1,1), JzBra(1,1)))
        >>> t.doit()
        1

        """
        if hasattr(self.args[0], '_eval_trace'):
            return self.args[0]._eval_trace(indices=self.args[1])

        return self  # 如果对象没有 _eval_trace 方法，返回对象本身

    @property
    def is_number(self):
        # TODO : 改进此实现
        return True  # 总是返回 True，暂未实现具体逻辑

    # TODO: 检查是否需要 permute 方法
    # 以及它是否需要返回一个新实例
    # 定义一个方法用于对参数进行循环排列

    def permute(self, pos):
        """ Permute the arguments cyclically.

        Parameters
        ==========

        pos : integer, if positive, shift-right, else shift-left
            位置参数：如果为正数，向右移动；如果为负数，向左移动

        Examples
        ========

        >>> from sympy.physics.quantum.trace import Tr
        >>> from sympy import symbols
        >>> A, B, C, D = symbols('A B C D', commutative=False)
        >>> t = Tr(A*B*C*D)
        >>> t.permute(2)
        Tr(C*D*A*B)
        >>> t.permute(-2)
        Tr(C*D*A*B)

        """
        # 根据参数正负值确定移动的位数
        if pos > 0:
            pos = pos % len(self.args[0].args)
        else:
            pos = -(abs(pos) % len(self.args[0].args))

        # 根据确定的位移，重新排列参数列表
        args = list(self.args[0].args[-pos:] + self.args[0].args[0:-pos])

        # 返回重新排列后的表达式
        return Tr(Mul(*(args)))

    def _hashable_content(self):
        # 如果参数是乘法类型，对参数进行循环排列和重新排列
        if isinstance(self.args[0], Mul):
            args = _cycle_permute(_rearrange_args(self.args[0].args))
        else:
            args = [self.args[0]]

        # 返回一个元组，其中包含重新排列后的参数和第二个参数
        return tuple(args) + (self.args[1], )
```