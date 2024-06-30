# `D:\src\scipysrc\sympy\sympy\physics\paulialgebra.py`

```
"""
This module implements Pauli algebra by subclassing Symbol. Only algebraic
properties of Pauli matrices are used (we do not use the Matrix class).

See the documentation to the class Pauli for examples.

References
==========

.. [1] https://en.wikipedia.org/wiki/Pauli_matrices
"""

# 导入必要的模块和类
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.symbol import Symbol
from sympy.physics.quantum import TensorProduct

# 定义可以导出的公共函数和类
__all__ = ['evaluate_pauli_product']


def delta(i, j):
    """
    Returns 1 if ``i == j``, else 0.

    This is used in the multiplication of Pauli matrices.

    Examples
    ========

    >>> from sympy.physics.paulialgebra import delta
    >>> delta(1, 1)
    1
    >>> delta(2, 3)
    0
    """
    # 检查索引是否相等，返回相应的值
    if i == j:
        return 1
    else:
        return 0


def epsilon(i, j, k):
    """
    Return 1 if i,j,k is equal to (1,2,3), (2,3,1), or (3,1,2);
    -1 if ``i``,``j``,``k`` is equal to (1,3,2), (3,2,1), or (2,1,3);
    else return 0.

    This is used in the multiplication of Pauli matrices.

    Examples
    ========

    >>> from sympy.physics.paulialgebra import epsilon
    >>> epsilon(1, 2, 3)
    1
    >>> epsilon(1, 3, 2)
    -1
    """
    # 检查给定的三元组是否在指定的模式中，返回相应的值
    if (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2)):
        return 1
    elif (i, j, k) in ((1, 3, 2), (3, 2, 1), (2, 1, 3)):
        return -1
    else:
        return 0


class Pauli(Symbol):
    """
    The class representing algebraic properties of Pauli matrices.

    Explanation
    ===========

    The symbol used to display the Pauli matrices can be changed with an
    optional parameter ``label="sigma"``. Pauli matrices with different
    ``label`` attributes cannot multiply together.

    If the left multiplication of symbol or number with Pauli matrix is needed,
    please use parentheses  to separate Pauli and symbolic multiplication
    (for example: 2*I*(Pauli(3)*Pauli(2))).

    Another variant is to use evaluate_pauli_product function to evaluate
    the product of Pauli matrices and other symbols (with commutative
    multiply rules).

    See Also
    ========

    evaluate_pauli_product

    Examples
    ========

    >>> from sympy.physics.paulialgebra import Pauli
    >>> Pauli(1)
    sigma1
    >>> Pauli(1)*Pauli(2)
    I*sigma3
    >>> Pauli(1)*Pauli(1)
    1
    >>> Pauli(3)**4
    1
    >>> Pauli(1)*Pauli(2)*Pauli(3)
    I

    >>> from sympy.physics.paulialgebra import Pauli
    >>> Pauli(1, label="tau")
    tau1
    >>> Pauli(1)*Pauli(2, label="tau")
    sigma1*tau2
    >>> Pauli(1, label="tau")*Pauli(2, label="tau")
    I*tau3

    >>> from sympy import I
    >>> I*(Pauli(2)*Pauli(3))
    -sigma1

    >>> from sympy.physics.paulialgebra import evaluate_pauli_product
    >>> f = I*Pauli(2)*Pauli(3)
    >>> f
    I*sigma2*sigma3
    >>> evaluate_pauli_product(f)
    -sigma1
    """

    __slots__ = ("i", "label")
    # 创建一个新的 Pauli 对象实例
    def __new__(cls, i, label="sigma"):
        # 检查输入的 Pauli 索引是否有效
        if i not in [1, 2, 3]:
            raise IndexError("Invalid Pauli index")
        # 调用父类 Symbol 的 __new__ 方法创建新的符号对象实例
        obj = Symbol.__new__(cls, "%s%d" %(label,i), commutative=False, hermitian=True)
        obj.i = i
        obj.label = label
        return obj

    # 返回用于创建对象的参数元组和空字典
    def __getnewargs_ex__(self):
        return (self.i, self.label), {}

    # 返回一个可哈希内容的元组，用于对象的哈希计算
    def _hashable_content(self):
        return (self.i, self.label)

    # FIXME 不适用于 -I*Pauli(2)*Pauli(3) 情况
    def __mul__(self, other):
        # 如果 other 是 Pauli 对象
        if isinstance(other, Pauli):
            j = self.i
            k = other.i
            jlab = self.label
            klab = other.label

            # 如果两个 Pauli 对象的标签相同
            if jlab == klab:
                return delta(j, k) \
                    + I*epsilon(j, k, 1)*Pauli(1,jlab) \
                    + I*epsilon(j, k, 2)*Pauli(2,jlab) \
                    + I*epsilon(j, k, 3)*Pauli(3,jlab)
        # 否则调用父类的 __mul__ 方法
        return super().__mul__(other)

    # 计算对象的指数幂操作
    def _eval_power(b, e):
        # 如果指数 e 是正整数
        if e.is_Integer and e.is_positive:
            return super().__pow__(int(e) % 2)
def evaluate_pauli_product(arg):
    '''Help function to evaluate Pauli matrices product
    with symbolic objects.

    Parameters
    ==========

    arg: symbolic expression that contains Pauli matrices

    Examples
    ========

    >>> from sympy.physics.pauli import Pauli, evaluate_pauli_product
    >>> from sympy import I
    >>> evaluate_pauli_product(I*Pauli(1)*Pauli(2))
    -sigma3

    >>> from sympy.abc import x
    >>> evaluate_pauli_product(x**2*Pauli(2)*Pauli(1))
    -I*x**2*sigma3
    '''
    
    # 初始化开始和结束的变量为参数本身
    start = arg
    end = arg

    # 如果参数是幂次运算且底数是 Pauli 对象，且指数是奇数，则返回该 Pauli 对象；否则返回 1
    if isinstance(arg, Pow) and isinstance(arg.args[0], Pauli):
        if arg.args[1].is_odd:
            return arg.args[0]
        else:
            return 1

    # 如果参数是加法表达式，则对其中每一部分递归调用 evaluate_pauli_product，并返回它们的和
    if isinstance(arg, Add):
        return Add(*[evaluate_pauli_product(part) for part in arg.args])

    # 如果参数是张量积表达式，则对其中每一部分递归调用 evaluate_pauli_product，并返回它们的张量积
    if isinstance(arg, TensorProduct):
        return TensorProduct(*[evaluate_pauli_product(part) for part in arg.args])

    # 如果参数不是乘法表达式，则直接返回参数
    elif not(isinstance(arg, Mul)):
        return arg

    # 当开始不等于结束时继续循环处理参数
    while not start == end or start == arg and end == arg:
        start = end

        # 将开始项分解为系数和乘积项
        tmp = start.as_coeff_mul()
        sigma_product = 1  # 初始化 Pauli 矩阵的乘积为单位元素
        com_product = 1    # 初始化可交换项的乘积为单位元素
        keeper = 1         # 初始化保留项为单位元素

        # 遍历乘积项的每一个元素
        for el in tmp[1]:
            if isinstance(el, Pauli):
                sigma_product *= el  # 如果是 Pauli 对象，则乘到 sigma_product 中
            elif not el.is_commutative:
                if isinstance(el, Pow) and isinstance(el.args[0], Pauli):
                    if el.args[1].is_odd:
                        sigma_product *= el.args[0]  # 如果是 Pauli 对象的奇次幂，则乘到 sigma_product 中
                elif isinstance(el, TensorProduct):
                    keeper = keeper * sigma_product * \
                        TensorProduct(
                            *[evaluate_pauli_product(part) for part in el.args]
                        )  # 如果是张量积表达式，则处理每一部分并乘到 keeper 中
                    sigma_product = 1  # 重置 sigma_product
                else:
                    keeper = keeper * sigma_product * el  # 否则乘到 keeper 中
                    sigma_product = 1  # 重置 sigma_product
            else:
                com_product *= el  # 如果是可交换项，则乘到 com_product 中
        end = tmp[0] * keeper * sigma_product * com_product  # 更新结束项
        if end == arg:
            break  # 如果结束项等于参数则结束循环
    return end  # 返回最终的结束项
```