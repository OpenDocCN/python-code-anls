# `D:\src\scipysrc\sympy\sympy\physics\quantum\tensorproduct.py`

```
# 导入所需模块和类
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sympify import sympify
from sympy.matrices.dense import DenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix as ImmutableMatrix
from sympy.printing.pretty.stringpict import prettyForm

# 导入量子力学相关模块和类
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray,
    scipy_sparse_matrix,
    matrix_tensor_product
)
from sympy.physics.quantum.trace import Tr

# 将当前模块中的指定类或函数导出
__all__ = [
    'TensorProduct',
    'tensor_product_simp'
]

#-----------------------------------------------------------------------------
# Tensor product
#-----------------------------------------------------------------------------

# 控制是否将张量积状态打印为组合的 Bra/Ket 还是作为不同 Bra/Ket 的显式张量积的标志
_combined_printing = False


def combined_tensor_printing(combined):
    """设置全局标志，控制是否将状态的张量积打印为组合的 Bra/Ket 或作为不同 Bra/Ket 的显式张量积。

    Parameters
    ----------
    combined : bool
        当为 True 时，将张量积状态组合成一个 Bra/Ket；当为 False 时，使用显式的张量积符号分隔每个 Bra/Ket。
    """
    global _combined_printing
    _combined_printing = combined


class TensorProduct(Expr):
    """两个或多个参数的张量积。

    对于矩阵，使用 ``matrix_tensor_product`` 计算 Kronecker 或张量积矩阵。对于其他对象，返回一个符号化的 ``TensorProduct`` 实例。
    张量积是在量子力学中主要用于操作符和状态的非交换乘法。

    当前，张量积区分可交换和不可交换的参数。可交换的参数假定为标量，并从 ``TensorProduct`` 中分离出来。不可交换的参数保留在结果的 ``TensorProduct`` 中。

    Parameters
    ==========

    args : tuple
        要进行张量积的对象序列。

    Examples
    ========

    从简单的 SymPy 矩阵开始的张量积示例::

        >>> from sympy import Matrix
        >>> from sympy.physics.quantum import TensorProduct

        >>> m1 = Matrix([[1,2],[3,4]])
        >>> m2 = Matrix([[1,0],[0,1]])
        >>> TensorProduct(m1, m2)
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2],
        [3, 0, 4, 0],
        [0, 3, 0, 4]])
        >>> TensorProduct(m2, m1)
        Matrix([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 3, 4]])
    ```
    # 是否为交换的（非交换）
    is_commutative = False

    # 构造函数，用于创建新的 TensorProduct 实例
    def __new__(cls, *args):
        # 如果第一个参数是矩阵或者其他特定类型，则调用 matrix_tensor_product 函数
        if isinstance(args[0], (Matrix, ImmutableMatrix, numpy_ndarray,
                                scipy_sparse_matrix)):
            return matrix_tensor_product(*args)
        # 将参数符号化，并展平成常数部分和非交换部分
        c_part, new_args = cls.flatten(sympify(args))
        # 将常数部分合并成一个乘法表达式
        c_part = Mul(*c_part)
        # 根据参数数量返回不同的计算结果
        if len(new_args) == 0:
            return c_part
        elif len(new_args) == 1:
            return c_part * new_args[0]
        else:
            tp = Expr.__new__(cls, *new_args)
            return c_part * tp

    @classmethod
    def flatten(cls, args):
        # TODO: 禁止嵌套的 TensorProducts
        # 分别收集常数部分和非交换部分
        c_part = []
        nc_parts = []
        for arg in args:
            # 获取每个参数的常数部分和非交换部分
            cp, ncp = arg.args_cnc()
            # 将常数部分添加到列表中
            c_part.extend(list(cp))
            # 创建乘法表达式并添加到非交换部分列表中
            nc_parts.append(Mul._from_args(ncp))
        return c_part, nc_parts

    # 计算伴随操作（共轭转置）
    def _eval_adjoint(self):
        # 对每个参数应用 Dagger 操作，并返回新的 TensorProduct 实例
        return TensorProduct(*[Dagger(i) for i in self.args])

    # 重写函数，根据指定的规则对参数进行重写并展开成乘积
    def _eval_rewrite(self, rule, args, **hints):
        return TensorProduct(*args).expand(tensorproduct=True)

    # 返回 TensorProduct 实例的字符串表示
    def _sympystr(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            # 如果参数是 Add、Pow 或 Mul 类型，则在打印时加上括号
            if isinstance(self.args[i], (Add, Pow, Mul)):
                s = s + '('
            s = s + printer._print(self.args[i])
            if isinstance(self.args[i], (Add, Pow, Mul)):
                s = s + ')'
            if i != length - 1:
                s = s + 'x'
        return s
    # 定义一个私有方法 `_pretty`，用于美化打印输出
    def _pretty(self, printer, *args):

        # 如果 `_combined_printing` 为真，并且所有参数都是 `Ket` 或 `Bra` 类型
        if (_combined_printing and
                (all(isinstance(arg, Ket) for arg in self.args) or
                 all(isinstance(arg, Bra) for arg in self.args))):

            # 获取参数列表的长度
            length = len(self.args)
            # 用打印机对象打印空字符串，得到初始形式
            pform = printer._print('', *args)
            # 遍历参数列表
            for i in range(length):
                # 使用打印机对象打印空字符串，得到下一个形式
                next_pform = printer._print('', *args)
                # 获取当前参数的子参数列表的长度
                length_i = len(self.args[i].args)
                # 遍历当前参数的子参数列表
                for j in range(length_i):
                    # 使用打印机对象打印当前参数的子参数，并得到其形式
                    part_pform = printer._print(self.args[i].args[j], *args)
                    # 更新下一个形式，将当前参数的子参数形式添加到右侧
                    next_pform = prettyForm(*next_pform.right(part_pform))
                    # 如果不是最后一个子参数，添加逗号和空格到右侧
                    if j != length_i - 1:
                        next_pform = prettyForm(*next_pform.right(', '))

                # 如果当前参数的子参数数量大于1，使用大括号包裹下一个形式
                if len(self.args[i].args) > 1:
                    next_pform = prettyForm(
                        *next_pform.parens(left='{', right='}')
                    )
                # 更新总形式，将下一个形式添加到右侧
                pform = prettyForm(*pform.right(next_pform))
                # 如果不是最后一个参数，添加逗号和空格到右侧
                if i != length - 1:
                    pform = prettyForm(*pform.right(',' + ' '))

            # 更新总形式，将第一个参数的左括号添加到左侧，将最后一个参数的右括号添加到右侧
            pform = prettyForm(*pform.left(self.args[0].lbracket))
            pform = prettyForm(*pform.right(self.args[0].rbracket))
            # 返回最终的形式
            return pform

        # 如果不满足上述条件，则执行以下操作
        length = len(self.args)
        # 用打印机对象打印空字符串，得到初始形式
        pform = printer._print('', *args)
        # 遍历参数列表
        for i in range(length):
            # 使用打印机对象打印当前参数，并得到其形式
            next_pform = printer._print(self.args[i], *args)
            # 如果当前参数是 `Add` 或 `Mul` 类型，使用括号包裹当前形式
            if isinstance(self.args[i], (Add, Mul)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            # 更新总形式，将当前形式添加到右侧
            pform = prettyForm(*pform.right(next_pform))
            # 如果不是最后一个参数，根据打印机是否使用 Unicode 添加乘号或 `x` 和空格到右侧
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right('x' + ' '))
        # 返回最终的形式
        return pform
    # 定义一个私有方法 `_latex`，用于生成表示 LaTeX 格式的字符串
    def _latex(self, printer, *args):

        # 如果启用了组合打印，并且所有参数都是 Ket 或者所有参数都是 Bra
        if (_combined_printing and
                (all(isinstance(arg, Ket) for arg in self.args) or
                 all(isinstance(arg, Bra) for arg in self.args))):
            
            # 定义一个内部函数，用于包装标签，处理多标签情况
            def _label_wrap(label, nlabels):
                return label if nlabels == 1 else r"\left\{%s\right\}" % label

            # 将参数的打印标签格式化为 LaTeX 格式的字符串
            s = r", ".join([_label_wrap(arg._print_label_latex(printer, *args),
                                        len(arg.args)) for arg in self.args])

            # 返回格式化后的 LaTeX 字符串，包含左右边界
            return r"{%s%s%s}" % (self.args[0].lbracket_latex, s,
                                  self.args[0].rbracket_latex)

        # 计算参数的个数
        length = len(self.args)
        s = ''
        # 遍历每个参数
        for i in range(length):
            # 如果参数是 Add 或者 Mul 类型之一，添加左括号
            if isinstance(self.args[i], (Add, Mul)):
                s = s + '\\left('
            # 用大括号包裹参数的打印结果，以便 matplotlib 正确渲染 LaTeX
            s = s + '{' + printer._print(self.args[i], *args) + '}'
            # 如果参数是 Add 或者 Mul 类型之一，添加右括号
            if isinstance(self.args[i], (Add, Mul)):
                s = s + '\\right)'
            # 如果不是最后一个参数，添加张量积符号
            if i != length - 1:
                s = s + '\\otimes '
        # 返回最终生成的 LaTeX 格式字符串
        return s

    # 定义一个方法 `doit`，对张量积进行求值
    def doit(self, **hints):
        return TensorProduct(*[item.doit(**hints) for item in self.args])

    # 定义一个方法 `_eval_expand_tensorproduct`，用于在加法中展开张量积
    def _eval_expand_tensorproduct(self, **hints):
        """Distribute TensorProducts across addition."""
        args = self.args
        add_args = []
        # 遍历参数列表
        for i in range(len(args)):
            # 如果当前参数是 Add 类型
            if isinstance(args[i], Add):
                # 将 Add 参数分解为单独的 TensorProduct 对象
                for aa in args[i].args:
                    tp = TensorProduct(*args[:i] + (aa,) + args[i + 1:])
                    c_part, nc_part = tp.args_cnc()
                    # 检查非合并部分是否包含单个 TensorProduct 对象，需要展开
                    if len(nc_part) == 1 and isinstance(nc_part[0], TensorProduct):
                        nc_part = (nc_part[0]._eval_expand_tensorproduct(), )
                    # 将展开后的部分添加到列表中
                    add_args.append(Mul(*c_part)*Mul(*nc_part))
                break

        # 如果有展开后的部分，则返回加法结果
        if add_args:
            return Add(*add_args)
        else:
            return self

    # 定义一个方法 `_eval_trace`，计算张量积的迹
    def _eval_trace(self, **kwargs):
        # 获取指定的索引列表
        indices = kwargs.get('indices', None)
        # 对张量积进行简化处理
        exp = tensor_product_simp(self)

        # 如果索引列表为空或者长度为 0
        if indices is None or len(indices) == 0:
            # 返回每个参数的迹的乘积
            return Mul(*[Tr(arg).doit() for arg in exp.args])
        else:
            # 返回根据索引计算的迹的乘积或者保持原值
            return Mul(*[Tr(value).doit() if idx in indices else value
                         for idx, value in enumerate(exp.args)])
# 简化具有张量积的乘法表达式
def tensor_product_simp_Mul(e):
    """Simplify a Mul with TensorProducts.

    Current the main use of this is to simplify a ``Mul`` of ``TensorProduct``s
    to a ``TensorProduct`` of ``Muls``. It currently only works for relatively
    simple cases where the initial ``Mul`` only has scalars and raw
    ``TensorProduct``s, not ``Add``, ``Pow``, ``Commutator``s of
    ``TensorProduct``s.

    Parameters
    ==========

    e : Expr
        A ``Mul`` of ``TensorProduct``s to be simplified.

    Returns
    =======

    e : Expr
        A ``TensorProduct`` of ``Mul``s.

    Examples
    ========

    This is an example of the type of simplification that this function
    performs::

        >>> from sympy.physics.quantum.tensorproduct import \
                    tensor_product_simp_Mul, TensorProduct
        >>> from sympy import Symbol
        >>> A = Symbol('A', commutative=False)
        >>> B = Symbol('B', commutative=False)
        >>> C = Symbol('C', commutative=False)
        >>> D = Symbol('D', commutative=False)
        >>> e = TensorProduct(A, B) * TensorProduct(C, D)
        >>> e
        AxB*CxD
        >>> tensor_product_simp_Mul(e)
        (A*C)x(B*D)

    """
    # 如果输入表达式不是 Mul 类型，则直接返回
    if not isinstance(e, Mul):
        return e
    # 将表达式 e 拆分为系数部分和非系数部分
    c_part, nc_part = e.args_cnc()
    # 计算非系数部分的数量
    n_nc = len(nc_part)
    # 如果非系数部分为空，则返回原始表达式
    if n_nc == 0:
        return e
    # 如果非系数部分只有一个元素，并且是 Pow 类型，则对其进行处理并返回处理结果
    elif n_nc == 1:
        if isinstance(nc_part[0], Pow):
            return Mul(*c_part) * tensor_product_simp_Pow(nc_part[0])
        # 否则直接返回原始表达式
        return e
    # 如果表达式 e 包含 TensorProduct
    elif e.has(TensorProduct):
        # 当前处理的部分是非交换部分的第一个元素
        current = nc_part[0]
        # 如果当前元素不是 TensorProduct 类型，则进行处理
        if not isinstance(current, TensorProduct):
            # 如果当前元素是 Pow 类型
            if isinstance(current, Pow):
                # 如果 Pow 的基是 TensorProduct 类型，则简化处理
                if isinstance(current.base, TensorProduct):
                    current = tensor_product_simp_Pow(current)
            else:
                # 如果以上条件均不满足，则引发类型错误异常
                raise TypeError('TensorProduct expected, got: %r' % current)
        
        # 计算当前 TensorProduct 中的项数
        n_terms = len(current.args)
        # 复制当前的参数列表
        new_args = list(current.args)
        
        # 遍历非交换部分的其余部分
        for next in nc_part[1:]:
            # TODO: 在此处检查 next 和 current 的希尔伯特空间
            
            # 如果 next 是 TensorProduct 类型
            if isinstance(next, TensorProduct):
                # 如果 next 的项数与当前项数不同，则引发量子错误异常
                if n_terms != len(next.args):
                    raise QuantumError(
                        'TensorProducts of different lengths: %r and %r' %
                        (current, next)
                    )
                
                # 对每个位置的项进行乘积操作
                for i in range(len(new_args)):
                    new_args[i] = new_args[i] * next.args[i]
            
            # 如果 next 不是 TensorProduct 类型
            else:
                # 如果 next 是 Pow 类型
                if isinstance(next, Pow):
                    # 如果 Pow 的基是 TensorProduct 类型，则简化处理
                    if isinstance(next.base, TensorProduct):
                        new_tp = tensor_product_simp_Pow(next)
                        # 对每个位置的项进行乘积操作
                        for i in range(len(new_args)):
                            new_args[i] = new_args[i] * new_tp.args[i]
                    else:
                        # 如果 Pow 的基不是 TensorProduct 类型，则引发类型错误异常
                        raise TypeError('TensorProduct expected, got: %r' % next)
                else:
                    # 如果 next 既不是 TensorProduct 类型也不是 Pow 类型，则引发类型错误异常
                    raise TypeError('TensorProduct expected, got: %r' % next)
            
            # 将当前处理的部分更新为下一个部分
            current = next
        
        # 返回重新组合后的 Mul 结果乘以新的 TensorProduct
        return Mul(*c_part) * TensorProduct(*new_args)
    
    # 如果表达式 e 包含 Pow
    elif e.has(Pow):
        # 对非交换部分中的每个 Pow 进行简化处理
        new_args = [ tensor_product_simp_Pow(nc) for nc in nc_part ]
        # 返回简化后的 Mul 结果乘以新的 TensorProduct
        return tensor_product_simp_Mul(Mul(*c_part) * TensorProduct(*new_args))
    
    # 如果以上条件均不满足，则直接返回原始表达式 e
    else:
        return e
# 定义函数，用于简化并合并 TensorProducts
def tensor_product_simp(e, **hints):
    """Try to simplify and combine TensorProducts.

    In general this will try to pull expressions inside of ``TensorProducts``.
    It currently only works for relatively simple cases where the products have
    only scalars, raw ``TensorProducts``, not ``Add``, ``Pow``, ``Commutators``
    of ``TensorProducts``. It is best to see what it does by showing examples.

    Examples
    ========

    >>> from sympy.physics.quantum import tensor_product_simp
    >>> from sympy.physics.quantum import TensorProduct
    >>> from sympy import Symbol
    >>> A = Symbol('A', commutative=False)
    >>> B = Symbol('B', commutative=False)
    >>> C = Symbol('C', commutative=False)
    >>> D = Symbol('D', commutative=False)

    First see what happens to products of tensor products:

    >>> e = TensorProduct(A,B)*TensorProduct(C,D)
    >>> e
    AxB*CxD
    >>> tensor_product_simp(e)
    (A*C)x(B*D)

    This is the core logic of this function, and it works inside, powers, sums,
    commutators and anticommutators as well:

    >>> tensor_product_simp(e**2)
    (A*C)x(B*D)**2

    """

    # 如果表达式是加法，则逐个简化其成员并返回加法结果
    if isinstance(e, Add):
        return Add(*[tensor_product_simp(arg) for arg in e.args])
    # 如果表达式是幂运算，则根据基是不是 TensorProduct 分别处理
    elif isinstance(e, Pow):
        if isinstance(e.base, TensorProduct):
            return tensor_product_simp_Pow(e)
        else:
            return tensor_product_simp(e.base) ** e.exp
    # 如果表达式是乘法，则调用处理乘法的函数
    elif isinstance(e, Mul):
        return tensor_product_simp_Mul(e)
    # 如果表达式是交换子，则逐个简化其成员并返回交换子
    elif isinstance(e, Commutator):
        return Commutator(*[tensor_product_simp(arg) for arg in e.args])
    # 如果表达式是反交换子，则逐个简化其成员并返回反交换子
    elif isinstance(e, AntiCommutator):
        return AntiCommutator(*[tensor_product_simp(arg) for arg in e.args])
    # 对于其他情况，保持不变直接返回
    else:
        return e
```