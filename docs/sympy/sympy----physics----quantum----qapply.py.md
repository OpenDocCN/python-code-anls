# `D:\src\scipysrc\sympy\sympy\physics\quantum\qapply.py`

```
# 导入必要的模块和类
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sympify import sympify

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import OuterProduct, Operator
from sympy.physics.quantum.state import State, KetBase, BraBase, Wavefunction
from sympy.physics.quantum.tensorproduct import TensorProduct

__all__ = [
    'qapply'
]

#-----------------------------------------------------------------------------
# 主要代码
#-----------------------------------------------------------------------------

def qapply(e, **options):
    """将运算符应用于量子表达式中的态。

    Parameters
    ==========

    e : Expr
        包含运算符和态的表达式。将递归地查找操作符作用在符号态上。
    options : dict
        一个包含键值对的字典，确定如何执行运算符的操作。

        有效的选项包括：

        * ``dagger``: 尝试将Dagger运算符应用到左边（默认为False）。
        * ``ip_doit``: 当遇到内积时，调用``.doit()``方法（默认为True）。

    Returns
    =======

    e : Expr
        原始表达式，但运算符已应用于态。

    Examples
    ========

        >>> from sympy.physics.quantum import qapply, Ket, Bra
        >>> b = Bra('b')
        >>> k = Ket('k')
        >>> A = k * b
        >>> A
        |k><b|
        >>> qapply(A * b.dual / (b * b.dual))
        |k>
        >>> qapply(k.dual * A / (k.dual * k), dagger=True)
        <b|
        >>> qapply(k.dual * A / (k.dual * k))
        <k|*|k><b|/<k|k>
    """
    from sympy.physics.quantum.density import Density

    # 获取选项中的Dagger标志，默认为False
    dagger = options.get('dagger', False)

    # 如果表达式e为零，则返回零
    if e == 0:
        return S.Zero

    # 将表达式e展开到最简形式，包括所有的交换子和张量积
    e = e.expand(commutator=True, tensorproduct=True)

    # 如果e是KetBase的实例，则直接返回它
    if isinstance(e, KetBase):
        return e

    # 如果e是Add(a, b, c, ...)的实例，则递归地对每个项应用qapply
    # 返回Add(qapply(a), qapply(b), ...)
    # 如果表达式 e 是一个加法操作（Add），则进行如下处理：
    elif isinstance(e, Add):
        # 初始化结果为0
        result = 0
        # 遍历加法操作的每一个参数
        for arg in e.args:
            # 对每个参数递归调用 qapply，并累加结果
            result += qapply(arg, **options)
        # 对最终累加得到的结果进行展开（expand）
        return result.expand()

    # 如果表达式 e 是一个密度算子（Density），则进行如下处理：
    elif isinstance(e, Density):
        # 对每一个 (state, prob) 组合，递归调用 qapply，并构造新的参数列表
        new_args = [(qapply(state, **options), prob) for (state, prob) in e.args]
        # 返回一个新的 Density 对象，使用新参数列表
        return Density(*new_args)

    # 如果表达式 e 是一个张量积（TensorProduct），则进行如下处理：
    elif isinstance(e, TensorProduct):
        # 对张量积的每一个元素，递归调用 qapply，并构造新的张量积对象
        return TensorProduct(*[qapply(t, **options) for t in e.args])

    # 如果表达式 e 是一个幂次操作（Pow），则进行如下处理：
    elif isinstance(e, Pow):
        # 对幂次操作的底数递归调用 qapply，并对结果取幂次操作的指数次方
        return qapply(e.base, **options)**e.exp

    # 如果表达式 e 是一个乘法操作（Mul），则进行如下处理：
    elif isinstance(e, Mul):
        # 将乘法操作的参数分成常数部分和非常数部分
        c_part, nc_part = e.args_cnc()
        # 将常数部分和非常数部分各自构造成乘法对象
        c_mul = Mul(*c_part)
        nc_mul = Mul(*nc_part)
        # 如果非常数部分是一个乘法操作，则调用 qapply_Mul 处理
        if isinstance(nc_mul, Mul):
            result = c_mul*qapply_Mul(nc_mul, **options)
        else:
            # 否则，对非常数部分递归调用 qapply 处理
            result = c_mul*qapply(nc_mul, **options)
        # 如果结果等于原始表达式 e，并且需要使用 dagger 操作，则返回 dagger 处理后的结果
        if result == e and dagger:
            return Dagger(qapply_Mul(Dagger(e), **options))
        else:
            # 否则，直接返回处理后的结果
            return result

    # 对于所有其他情况（如 State, Operator, Commutator, InnerProduct, OuterProduct），不需要进一步操作
    else:
        # 直接返回表达式 e 自身
        return e
def qapply_Mul(e, **options):
    ip_doit = options.get('ip_doit', True)  # 获取参数字典中的 ip_doit 参数，默认为 True

    args = list(e.args)  # 将表达式 e 的参数列表化

    # 如果参数个数小于等于1或者 e 不是 Mul 类型，则无需处理，直接返回 e
    if len(args) <= 1 or not isinstance(e, Mul):
        return e
    rhs = args.pop()  # 弹出最后一个参数作为 rhs
    lhs = args.pop()  # 弹出倒数第二个参数作为 lhs

    # 如果 lhs 或 rhs 不是 Wavefunction，并且 sympify(rhs).is_commutative 为真，或者 lhs 不是 Wavefunction 并且 sympify(lhs).is_commutative 为真，则无需处理，直接返回 e
    if (not isinstance(rhs, Wavefunction) and sympify(rhs).is_commutative) or \
            (not isinstance(lhs, Wavefunction) and sympify(lhs).is_commutative):
        return e

    # 如果 lhs 是 Pow 类型且指数 exp 是整数，则将 lhs.base ** (lhs.exp - 1) 添加到 args 中，并将 lhs.base 赋值给 lhs
    if isinstance(lhs, Pow) and lhs.exp.is_Integer:
        args.append(lhs.base**(lhs.exp - 1))
        lhs = lhs.base

    # 如果 lhs 是 OuterProduct 类型，则将 lhs.ket 添加到 args 中，并将 lhs.bra 赋值给 lhs
    if isinstance(lhs, OuterProduct):
        args.append(lhs.ket)
        lhs = lhs.bra

    # 如果 lhs 是 Commutator 或 AntiCommutator 类型，则调用 lhs.doit() 处理
    if isinstance(lhs, (Commutator, AntiCommutator)):
        comm = lhs.doit()
        if isinstance(comm, Add):
            # 返回 qapply(...) 对象，通过 e.func(...) 加上 comm.args[0] 和 rhs 组成的新表达式，以及 e.func(...) 加上 comm.args[1] 和 rhs 组成的新表达式
            return qapply(
                e.func(*(args + [comm.args[0], rhs])) +
                e.func(*(args + [comm.args[1], rhs])),
                **options
            )
        else:
            # 返回 qapply(...) 对象，通过 e.func(...) 加上 args、comm 和 rhs 组成的新表达式
            return qapply(e.func(*args)*comm*rhs, **options)

    # 如果 lhs 和 rhs 都是 TensorProduct 类型，并且每个元素是 Operator、State、Mul、Pow 或 1，并且长度相同，则对每对 lhs.args[n]*rhs.args[n] 调用 qapply(...)，最后用 TensorProduct 包装并展开
    if isinstance(lhs, TensorProduct) and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in lhs.args) and \
            isinstance(rhs, TensorProduct) and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in rhs.args) and \
            len(lhs.args) == len(rhs.args):
        result = TensorProduct(*[qapply(lhs.args[n]*rhs.args[n], **options) for n in range(len(lhs.args))]).expand(tensorproduct=True)
        # 返回 qapply_Mul(...) 对象，通过 e.func(...) 加上 result 组成的新表达式
        return qapply_Mul(e.func(*args), **options)*result

    # 尝试实际应用操作符并构建内积
    try:
        result = lhs._apply_operator(rhs, **options)
    except NotImplementedError:
        result = None

    if result is None:
        _apply_right = getattr(rhs, '_apply_from_right_to', None)
        if _apply_right is not None:
            try:
                result = _apply_right(lhs, **options)
            except NotImplementedError:
                result = None

    if result is None:
        # 如果 lhs 是 BraBase 类型并且 rhs 是 KetBase 类型，则创建 InnerProduct 对象，如果 ip_doit 为真则执行 doit()
        if isinstance(lhs, BraBase) and isinstance(rhs, KetBase):
            result = InnerProduct(lhs, rhs)
            if ip_doit:
                result = result.doit()

    # 如果 result 等于 0，则返回 S.Zero
    if result == 0:
        return S.Zero
    elif result is None:
        if len(args) == 0:
            # 如果 args 为空，则返回 e
            return e
        else:
            # 否则返回 qapply_Mul(...) 对象，通过 e.func(...) 加上 args 和 lhs 组成的新表达式，再乘以 rhs
            return qapply_Mul(e.func(*(args + [lhs])), **options)*rhs
    elif isinstance(result, InnerProduct):
        # 如果 result 是 InnerProduct 类型，则返回 result 乘以 qapply_Mul(...) 对象
        return result*qapply_Mul(e.func(*args), **options)
    else:  # 如果结果是一个标量乘以 Mul、Add 或 TensorProduct 对象
        # 对表达式 e 中的函数应用 qapply，并乘以 result
        return qapply(e.func(*args)*result, **options)
```