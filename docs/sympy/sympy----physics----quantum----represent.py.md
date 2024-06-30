# `D:\src\scipysrc\sympy\sympy\physics\quantum\represent.py`

```
# 从 sympy.core.add 模块导入 Add 类
from sympy.core.add import Add
# 从 sympy.core.expr 模块导入 Expr 类
from sympy.core.expr import Expr
# 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.numbers 模块导入 I
from sympy.core.numbers import I
# 从 sympy.core.power 模块导入 Pow 类
from sympy.core.power import Pow
# 从 sympy.integrals.integrals 模块导入 integrate 函数
from sympy.integrals.integrals import integrate
# 从 sympy.physics.quantum.dagger 模块导入 Dagger 类
from sympy.physics.quantum.dagger import Dagger
# 从 sympy.physics.quantum.commutator 模块导入 Commutator 类
from sympy.physics.quantum.commutator import Commutator
# 从 sympy.physics.quantum.anticommutator 模块导入 AntiCommutator 类
from sympy.physics.quantum.anticommutator import AntiCommutator
# 从 sympy.physics.quantum.innerproduct 模块导入 InnerProduct 类
from sympy.physics.quantum.innerproduct import InnerProduct
# 从 sympy.physics.quantum.qexpr 模块导入 QExpr 类
from sympy.physics.quantum.qexpr import QExpr
# 从 sympy.physics.quantum.tensorproduct 模块导入 TensorProduct 类
from sympy.physics.quantum.tensorproduct import TensorProduct
# 从 sympy.physics.quantum.matrixutils 模块导入 flatten_scalar 函数
from sympy.physics.quantum.matrixutils import flatten_scalar
# 从 sympy.physics.quantum.state 模块导入 KetBase, BraBase, StateBase 类
from sympy.physics.quantum.state import KetBase, BraBase, StateBase
# 从 sympy.physics.quantum.operator 模块导入 Operator, OuterProduct 类
from sympy.physics.quantum.operator import Operator, OuterProduct
# 从 sympy.physics.quantum.qapply 模块导入 qapply 函数
from sympy.physics.quantum.qapply import qapply
# 从 sympy.physics.quantum.operatorset 模块导入 operators_to_state, state_to_operators 函数
from sympy.physics.quantum.operatorset import operators_to_state, state_to_operators

# 导出的符号列表，限制只导出以下符号
__all__ = [
    'represent',
    'rep_innerproduct',
    'rep_expectation',
    'integrate_result',
    'get_basis',
    'enumerate_states'
]

#-----------------------------------------------------------------------------
# Represent
#-----------------------------------------------------------------------------


def _sympy_to_scalar(e):
    """Convert from a SymPy scalar to a Python scalar."""
    # 如果 e 是 SymPy 的表达式
    if isinstance(e, Expr):
        # 如果 e 是整数类型
        if e.is_Integer:
            return int(e)
        # 如果 e 是浮点数类型
        elif e.is_Float:
            return float(e)
        # 如果 e 是有理数类型
        elif e.is_Rational:
            return float(e)
        # 如果 e 是数值或数值符号或复数单位 I
        elif e.is_Number or e.is_NumberSymbol or e == I:
            return complex(e)
    # 若不符合以上条件，则抛出类型错误异常
    raise TypeError('Expected number, got: %r' % e)


def represent(expr, **options):
    """Represent the quantum expression in the given basis.

    In quantum mechanics abstract states and operators can be represented in
    various basis sets. Under this operation the follow transforms happen:

    * Ket -> column vector or function
    * Bra -> row vector of function
    * Operator -> matrix or differential operator

    This function is the top-level interface for this action.

    This function walks the SymPy expression tree looking for ``QExpr``
    instances that have a ``_represent`` method. This method is then called
    and the object is replaced by the representation returned by this method.
    By default, the ``_represent`` method will dispatch to other methods
    that handle the representation logic for a particular basis set. The
    naming convention for these methods is the following::

        def _represent_FooBasis(self, e, basis, **options)

    This function will have the logic for representing instances of its class
    in the basis set having a class named ``FooBasis``.

    Parameters
    ==========

    expr  : Expr
        The expression to represent.
    """
    # 函数用于在给定的基础上表示量子表达式
    pass
    format = options.get('format', 'sympy')
    # 从 options 字典中获取 'format' 键对应的值，默认为 'sympy'
    if format == 'numpy':
        # 如果 format 的取值为 'numpy'，则导入 numpy 库
        import numpy as np
    # 检查表达式是否为 QExpr 类型且不是 OuterProduct 类型
    if isinstance(expr, QExpr) and not isinstance(expr, OuterProduct):
        # 设置选项中的 replace_none 为 False
        options['replace_none'] = False
        # 获取表达式的基矢量
        temp_basis = get_basis(expr, **options)
        # 如果临时基矢量不为 None，则更新选项中的 basis
        if temp_basis is not None:
            options['basis'] = temp_basis
        try:
            # 尝试调用表达式的 _represent 方法进行表示
            return expr._represent(**options)
        except NotImplementedError as strerr:
            # 如果不存在 _represent_FOO 方法，则映射到适当的基态并尝试其它表示方法
            options['replace_none'] = True

            # 如果表达式是 KetBase 或 BraBase 类型
            if isinstance(expr, (KetBase, BraBase)):
                try:
                    # 调用 rep_innerproduct 方法进行表示
                    return rep_innerproduct(expr, **options)
                except NotImplementedError:
                    # 如果操作也失败，则抛出之前的 NotImplementedError
                    raise NotImplementedError(strerr)
            # 如果表达式是 Operator 类型
            elif isinstance(expr, Operator):
                try:
                    # 调用 rep_expectation 方法进行表示
                    return rep_expectation(expr, **options)
                except NotImplementedError:
                    # 如果操作也失败，则抛出之前的 NotImplementedError
                    raise NotImplementedError(strerr)
            else:
                # 如果是其它类型的表达式，则抛出之前的 NotImplementedError
                raise NotImplementedError(strerr)

    # 如果表达式是 Add 类型
    elif isinstance(expr, Add):
        # 计算第一个参数的表示结果
        result = represent(expr.args[0], **options)
        # 对于剩余的参数，依次计算并累加表示结果
        for args in expr.args[1:]:
            # scipy.sparse 不支持 +=，因此这里使用 plain = 赋值
            result = result + represent(args, **options)
        return result

    # 如果表达式是 Pow 类型
    elif isinstance(expr, Pow):
        # 将表达式转换为底数和指数
        base, exp = expr.as_base_exp()
        # 如果格式是 'numpy' 或 'scipy.sparse'，则将指数转换为标量
        if format in ('numpy', 'scipy.sparse'):
            exp = _sympy_to_scalar(exp)
        # 计算底数的表示结果
        base = represent(base, **options)
        # 如果格式是 'scipy.sparse' 且指数为负数
        if format == 'scipy.sparse' and exp < 0:
            # 导入 scipy.sparse.linalg 中的 inv 函数
            from scipy.sparse.linalg import inv
            exp = - exp
            # 对 csr 格式的底数求逆
            base = inv(base.tocsc()).tocsr()
        # 如果格式是 'numpy'，则返回底数的 exp 次幂
        if format == 'numpy':
            return np.linalg.matrix_power(base, exp)
        # 否则返回底数的 exp 次幂
        return base ** exp

    # 如果表达式是 TensorProduct 类型
    elif isinstance(expr, TensorProduct):
        # 对表达式中的每个参数进行表示计算
        new_args = [represent(arg, **options) for arg in expr.args]
        return TensorProduct(*new_args)

    # 如果表达式是 Dagger 类型
    elif isinstance(expr, Dagger):
        # 对 Dagger 类型的参数进行表示计算
        return Dagger(represent(expr.args[0], **options))

    # 如果表达式是 Commutator 类型
    elif isinstance(expr, Commutator):
        # 获取表达式的第一个和第二个参数
        A = expr.args[0]
        B = expr.args[1]
        # 计算 Mul(A, B) - Mul(B, A) 的表示结果
        return represent(Mul(A, B) - Mul(B, A), **options)

    # 如果表达式是 AntiCommutator 类型
    elif isinstance(expr, AntiCommutator):
        # 获取表达式的第一个和第二个参数
        A = expr.args[0]
        B = expr.args[1]
        # 计算 Mul(A, B) + Mul(B, A) 的表示结果
        return represent(Mul(A, B) + Mul(B, A), **options)

    # 如果表达式是 InnerProduct 类型
    elif isinstance(expr, InnerProduct):
        # 计算 Mul(expr.bra, expr.ket) 的表示结果
        return represent(Mul(expr.bra, expr.ket), **options)

    # 如果表达式既不是 Mul 类型也不是 OuterProduct 类型
    elif not isinstance(expr, (Mul, OuterProduct)):
        # 对于 'numpy' 和 'scipy.sparse' 格式，只能处理数值前因子
        if format in ('numpy', 'scipy.sparse'):
            # 将表达式转换为标量
            return _sympy_to_scalar(expr)
        # 否则直接返回表达式本身
        return expr

    # 如果表达式既不是 Mul 类型也不是 OuterProduct 类型，则抛出类型错误
    if not isinstance(expr, (Mul, OuterProduct)):
        raise TypeError('Mul expected, got: %r' % expr)

    # 如果选项中存在 "index" 键，则递增其值
    if "index" in options:
        options["index"] += 1
    # 如果条件不满足（即expr不是Operator类型），设置options字典中的index为1
    else:
        options["index"] = 1

    # 如果options字典中没有"unities"键，则将其初始化为空列表
    if "unities" not in options:
        options["unities"] = []

    # 对表达式的最后一个参数进行表示，使用给定的选项
    result = represent(expr.args[-1], **options)
    # 将表达式的最后一个参数存储在last_arg中
    last_arg = expr.args[-1]

    # 对除了最后一个参数之外的参数进行反向遍历
    for arg in reversed(expr.args[:-1]):
        # 如果最后一个参数是Operator类型
        if isinstance(last_arg, Operator):
            # 增加options字典中的index计数
            options["index"] += 1
            # 将当前index添加到unities列表中
            options["unities"].append(options["index"])
        # 如果最后一个参数是BraBase类型，且当前参数是KetBase类型
        elif isinstance(last_arg, BraBase) and isinstance(arg, KetBase):
            # 增加options字典中的index计数
            options["index"] += 1
        # 如果最后一个参数是KetBase类型，且当前参数是Operator类型
        elif isinstance(last_arg, KetBase) and isinstance(arg, Operator):
            # 将当前index添加到unities列表中
            options["unities"].append(options["index"])
        # 如果最后一个参数是KetBase类型，且当前参数是BraBase类型
        elif isinstance(last_arg, KetBase) and isinstance(arg, BraBase):
            # 将当前index添加到unities列表中
            options["unities"].append(options["index"])

        # 对当前参数进行表示，使用给定的选项
        next_arg = represent(arg, **options)
        # 如果format为'numpy'且next_arg是np.ndarray类型
        if format == 'numpy' and isinstance(next_arg, np.ndarray):
            # 必须使用np.matmul进行两个np.ndarray的矩阵乘法
            result = np.matmul(next_arg, result)
        else:
            # 否则，使用标准乘法运算
            result = next_arg * result
        # 更新last_arg为当前参数
        last_arg = arg

    # 当内积为向量时，所有三种矩阵格式都创建1x1矩阵。在这些情况下，我们返回一个标量。
    result = flatten_scalar(result)

    # 将结果集成到expr中，使用给定的选项
    result = integrate_result(expr, result, **options)

    # 返回最终结果
    return result
# 定义函数，计算给定表达式的内积表示，如 `<x'|x>`
def rep_innerproduct(expr, **options):
    """
    Returns an innerproduct like representation (e.g. ``<x'|x>``) for the
    given state.

    Attempts to calculate inner product with a bra from the specified
    basis. Should only be passed an instance of KetBase or BraBase

    Parameters
    ==========

    expr : KetBase or BraBase
        The expression to be represented

    Examples
    ========

    >>> from sympy.physics.quantum.represent import rep_innerproduct
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> rep_innerproduct(XKet())
    DiracDelta(x - x_1)
    >>> rep_innerproduct(XKet(), basis=PxOp())
    sqrt(2)*exp(-I*px_1*x/hbar)/(2*sqrt(hbar)*sqrt(pi))
    >>> rep_innerproduct(PxKet(), basis=XOp())
    sqrt(2)*exp(I*px*x_1/hbar)/(2*sqrt(hbar)*sqrt(pi))

    """

    # 检查传入的表达式是否为 KetBase 或 BraBase 的实例
    if not isinstance(expr, (KetBase, BraBase)):
        raise TypeError("expr passed is not a Bra or Ket")

    # 根据选项获取表达式的基态
    basis = get_basis(expr, **options)

    # 检查获取的基态是否为 StateBase 的实例
    if not isinstance(basis, StateBase):
        raise NotImplementedError("Can't form this representation!")

    # 如果选项中没有指定索引，则设置默认索引为 1
    if "index" not in options:
        options["index"] = 1

    # 枚举基态的状态
    basis_kets = enumerate_states(basis, options["index"], 2)

    # 根据表达式的类型确定 bra 和 ket
    if isinstance(expr, BraBase):
        bra = expr
        ket = (basis_kets[1] if basis_kets[0].dual == expr else basis_kets[0])
    else:
        bra = (basis_kets[1].dual if basis_kets[0]
               == expr else basis_kets[0].dual)
        ket = expr

    # 计算 bra 和 ket 的内积
    prod = InnerProduct(bra, ket)
    result = prod.doit()

    # 获取格式选项，返回表达式的格式化表示
    format = options.get('format', 'sympy')
    return expr._format_represent(result, format)
    """
    This function integrates over any unities that may have been
    inserted into the quantum expression and returns the result.
    It uses the interval of the Hilbert space of the basis state
    passed to it in order to figure out the limits of integration.
    The unities option must be
    specified for this to work.

    Note: This is mostly used internally by represent(). Examples are
    given merely to show the use cases.

    Parameters
    ==========

    orig_expr : quantum expression
        The original expression which was to be represented

    result: Expr
        The resulting representation that we wish to integrate over

    Examples
    ========

    >>> from sympy import symbols, DiracDelta
    >>> from sympy.physics.quantum.represent import integrate_result
    >>> from sympy.physics.quantum.cartesian import XOp, XKet
    >>> x_ket = XKet()
    >>> X_op = XOp()
    >>> x, x_1, x_2 = symbols('x, x_1, x_2')
    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2))
    x*DiracDelta(x - x_1)*DiracDelta(x_1 - x_2)
    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2),
    ...     unities=[1])
    x*DiracDelta(x - x_2)

    """
    # 如果结果不是符号表达式，则直接返回结果
    if not isinstance(result, Expr):
        return result

    # 设置选项中的 'replace_none' 为 True
    options['replace_none'] = True
    # 如果选项中没有 'basis'，则从原始表达式中获取最后一个参数作为基态，并获取对应的基础
    if "basis" not in options:
        arg = orig_expr.args[-1]
        options["basis"] = get_basis(arg, **options)
    # 如果选项中的 'basis' 不是 StateBase 的实例，则重新获取基础
    elif not isinstance(options["basis"], StateBase):
        options["basis"] = get_basis(orig_expr, **options)

    # 弹出选项中的 'basis'，并赋值给变量 basis
    basis = options.pop("basis", None)

    # 如果 basis 为 None，则直接返回结果
    if basis is None:
        return result

    # 弹出选项中的 'unities'，并赋值给变量 unities
    unities = options.pop("unities", [])

    # 如果 unities 的长度为 0，则直接返回结果
    if len(unities) == 0:
        return result

    # 枚举基态中的状态，并获取其坐标
    kets = enumerate_states(basis, unities)
    coords = [k.label[0] for k in kets]

    # 对于每个坐标，如果在结果的自由符号中，则进行积分替换
    for coord in coords:
        if coord in result.free_symbols:
            # TODO: Add support for sets of operators
            # 将基态转换为操作符，并获取其希尔伯特空间的区间
            basis_op = state_to_operators(basis)
            start = basis_op.hilbert_space.interval.start
            end = basis_op.hilbert_space.interval.end
            # 对结果进行积分替换
            result = integrate(result, (coord, start, end))

    # 返回积分后的结果
    return result
# 返回与指定基础相关的基础状态实例，基础通过参数 options 指定。如果未指定基础，则尝试根据给定表达式形成默认基础状态。
def get_basis(expr, *, basis=None, replace_none=True, **options):
    """
    Returns a basis state instance corresponding to the basis specified in
    options=s. If no basis is specified, the function tries to form a default
    basis state of the given expression.

    There are three behaviors:

    1. The basis specified in options is already an instance of StateBase. If
       this is the case, it is simply returned. If the class is specified but
       not an instance, a default instance is returned.

    2. The basis specified is an operator or set of operators. If this
       is the case, the operator_to_state mapping method is used.

    3. No basis is specified. If expr is a state, then a default instance of
       its class is returned.  If expr is an operator, then it is mapped to the
       corresponding state.  If it is neither, then we cannot obtain the basis
       state.

    If the basis cannot be mapped, then it is not changed.

    This will be called from within represent, and represent will
    only pass QExpr's.

    TODO (?): Support for Muls and other types of expressions?

    Parameters
    ==========

    expr : Operator or StateBase
        Expression whose basis is sought

    Examples
    ========

    >>> from sympy.physics.quantum.represent import get_basis
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> x = XKet()
    >>> X = XOp()
    >>> get_basis(x)
    |x>
    >>> get_basis(X)
    |x>
    >>> get_basis(x, basis=PxOp())
    |px>
    >>> get_basis(x, basis=PxKet)
    |px>

    """

    # 如果未指定基础且不替换为 None，则返回 None
    if basis is None and not replace_none:
        return None

    # 如果未指定基础
    if basis is None:
        # 如果表达式是 KetBase 的实例，则返回其类的默认实例
        if isinstance(expr, KetBase):
            return _make_default(expr.__class__)
        # 如果表达式是 BraBase 的实例，则返回其对偶类的默认实例
        elif isinstance(expr, BraBase):
            return _make_default(expr.dual_class())
        # 如果表达式是 Operator 的实例，则将其映射为相应的状态实例
        elif isinstance(expr, Operator):
            state_inst = operators_to_state(expr)
            return (state_inst if state_inst is not None else None)
        else:
            return None
    # 如果指定的基础是 Operator 或其子类，则将其映射为相应的状态实例
    elif (isinstance(basis, Operator) or
          (not isinstance(basis, StateBase) and issubclass(basis, Operator))):
        state = operators_to_state(basis)
        if state is None:
            return None
        elif isinstance(state, StateBase):
            return state
        else:
            return _make_default(state)
    # 如果指定的基础是 StateBase 的实例，则直接返回
    elif isinstance(basis, StateBase):
        return basis
    # 如果指定的基础是 StateBase 的子类，则返回其类的默认实例
    elif issubclass(basis, StateBase):
        return _make_default(basis)
    else:
        return None


def _make_default(expr):
    # XXX: Catching TypeError like this is a bad way of distinguishing
    # instances from classes. The logic using this function should be
    # rewritten somehow.
    # 尝试创建实例，如果无法创建则返回表达式本身
    try:
        expr = expr()
    except TypeError:
        return expr

    return expr


def enumerate_states(*args, **options):
    """
    Returns instances of the given state with dummy indices appended

    Operates in two different modes:
    
    ...
    # 两个参数被传递给该函数。第一个参数是要进行索引的基本状态，第二个参数是要附加的索引列表。
    
    # 三个参数被传递给该函数。第一个参数仍然是要进行索引的基本状态。第二个参数是开始计数的起始索引。最后一个参数是您希望接收的 ket 的数量。
    
    # 尝试调用 state._enumerate_state 方法。如果调用失败，则返回一个空列表。
    
    state = args[0]
    
    if len(args) not in (2, 3):
        raise NotImplementedError("Wrong number of arguments!")
    
    if not isinstance(state, StateBase):
        raise TypeError("First argument is not a state!")
    
    if len(args) == 3:
        num_states = args[2]
        options['start_index'] = args[1]
    else:
        num_states = len(args[1])
        options['index_list'] = args[1]
    
    try:
        ret = state._enumerate_state(num_states, **options)
    except NotImplementedError:
        ret = []
    
    return ret
```