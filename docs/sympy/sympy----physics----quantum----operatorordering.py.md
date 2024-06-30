# `D:\src\scipysrc\sympy\sympy\physics\quantum\operatorordering.py`

```
# 导入警告模块
import warnings

# 从Sympy库中导入具体模块和类
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.physics.quantum import Commutator, AntiCommutator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

# 模块中可以被外部调用的函数和类的列表
__all__ = [
    'normal_order',
    'normal_ordered_form'
]

# 定义一个辅助函数，用于将幂表达式扩展为乘法表达式，以便于正规序处理
def _expand_powers(factors):
    """
    Helper function for normal_ordered_form and normal_order: Expand a
    power expression to a multiplication expression so that that the
    expression can be handled by the normal ordering functions.
    """
    new_factors = []
    # 遍历因子列表中的每个因子
    for factor in factors.args:
        # 如果因子是幂对象，并且指数部分是整数且大于0
        if (isinstance(factor, Pow)
                and isinstance(factor.args[1], Integer)
                and factor.args[1] > 0):
            # 将幂展开为多个因子
            for n in range(factor.args[1]):
                new_factors.append(factor.args[0])
        else:
            # 否则将因子直接添加到新因子列表中
            new_factors.append(factor)

    return new_factors

# 定义一个辅助函数，用于将乘法表达式中的因子重新排列为正规序形式
def _normal_ordered_form_factor(product, independent=False, recursive_limit=10,
                                _recursive_depth=0):
    """
    Helper function for normal_ordered_form_factor: Write multiplication
    expression with bosonic or fermionic operators on normally ordered form,
    using the bosonic and fermionic commutation relations. The resulting
    operator expression is equivalent to the argument, but will in general be
    a sum of operator products instead of a simple product.
    """
    # 扩展幂因子为乘法表达式
    factors = _expand_powers(product)

    # 初始化新因子列表和计数器
    new_factors = []
    n = 0
    # 当 n 小于因子列表长度减一时，执行循环
    while n < len(factors) - 1:
        # 取出当前因子和下一个因子
        current, next = factors[n], factors[n + 1]
        # 如果当前因子或下一个因子不是 FermionOp 或 BosonOp 类型之一，将当前因子添加到新因子列表，并继续下一轮循环
        if any(not isinstance(f, (FermionOp, BosonOp)) for f in (current, next)):
            new_factors.append(current)
            n += 1
            continue

        # 构建两个键来比较当前因子和下一个因子的优先级
        key_1 = (current.is_annihilation, str(current.name))
        key_2 = (next.is_annihilation, str(next.name))

        # 如果 key_1 小于等于 key_2，则将当前因子添加到新因子列表，并继续下一轮循环
        if key_1 <= key_2:
            new_factors.append(current)
            n += 1
            continue

        # 如果 key_1 大于 key_2，则跳过下一个因子，将当前因子的索引 n 向后移动两步
        n += 2
        # 如果当前因子是湮灭算符而下一个因子不是，则根据它们的类型执行不同的操作
        if current.is_annihilation and not next.is_annihilation:
            if isinstance(current, BosonOp) and isinstance(next, BosonOp):
                # 如果两个 BosonOp 的第一个参数不相同，根据独立性决定是否创建一个交换子，然后将结果添加到新因子列表
                if current.args[0] != next.args[0]:
                    if independent:
                        c = 0
                    else:
                        c = Commutator(current, next)
                    new_factors.append(next * current + c)
                else:
                    new_factors.append(next * current + 1)
            elif isinstance(current, FermionOp) and isinstance(next, FermionOp):
                # 如果两个 FermionOp 的第一个参数不相同，根据独立性决定是否创建一个反对易子，然后将结果添加到新因子列表
                if current.args[0] != next.args[0]:
                    if independent:
                        c = 0
                    else:
                        c = AntiCommutator(current, next)
                    new_factors.append(-next * current + c)
                else:
                    new_factors.append(-next * current + 1)
        elif (current.is_annihilation == next.is_annihilation and
              isinstance(current, FermionOp) and isinstance(next, FermionOp)):
            # 如果两个 FermionOp 都是湮灭算符或者都是创生算符，则将它们相乘的结果添加到新因子列表
            new_factors.append(-next * current)
        else:
            # 否则，将两个因子相乘的结果添加到新因子列表
            new_factors.append(next * current)

    # 如果 n 等于因子列表长度减一，则将最后一个因子添加到新因子列表
    if n == len(factors) - 1:
        new_factors.append(factors[-1])

    # 如果新因子列表和原因子列表相等，则直接返回 product
    if new_factors == factors:
        return product
    else:
        # 否则，将新因子列表组成的乘积表达式进行展开，并调用 normal_ordered_form 函数进行规范排序处理
        expr = Mul(*new_factors).expand()
        return normal_ordered_form(expr,
                                   recursive_limit=recursive_limit,
                                   _recursive_depth=_recursive_depth + 1,
                                   independent=independent)
# 定义一个名为 _normal_ordered_form_terms 的函数，用于将表达式中的每个项按正规次序进行排序处理。
# 参数:
#   expr: 表达式，可能包含加法和乘法操作
#   independent: 布尔值，默认为 False，指定操作符是否在不同的希尔伯特空间中
#   recursive_limit: 整数，默认为 10，递归调用函数的最大次数限制
#   _recursive_depth: 整数，默认为 0，当前递归深度
def _normal_ordered_form_terms(expr, independent=False, recursive_limit=10,
                               _recursive_depth=0):
    """
    Helper function for normal_ordered_form: loop through each term in an
    addition expression and call _normal_ordered_form_factor to perform the
    factor to an normally ordered expression.
    """

    # 创建一个新列表用于存放处理后的项
    new_terms = []
    # 遍历表达式中的每个项
    for term in expr.args:
        # 如果当前项是乘法表达式
        if isinstance(term, Mul):
            # 调用 _normal_ordered_form_factor 函数对该项进行正规排序处理
            new_term = _normal_ordered_form_factor(
                term, recursive_limit=recursive_limit,
                _recursive_depth=_recursive_depth, independent=independent)
            # 将处理后的项添加到新列表中
            new_terms.append(new_term)
        else:
            # 如果当前项不是乘法表达式，则直接添加到新列表中
            new_terms.append(term)

    # 将处理后的新项组合成一个加法表达式并返回
    return Add(*new_terms)


# 定义一个名为 normal_ordered_form 的函数，用于将表达式转换为正规次序形式
# 参数:
#   expr: 表达式，可能包含加法和乘法操作
#   independent: 布尔值，默认为 False，指定操作符是否在不同的希尔伯特空间中
#   recursive_limit: 整数，默认为 10，递归调用函数的最大次数限制
#   _recursive_depth: 整数，默认为 0，当前递归深度
def normal_ordered_form(expr, independent=False, recursive_limit=10,
                        _recursive_depth=0):
    """Write an expression with bosonic or fermionic operators on normal
    ordered form, where each term is normally ordered. Note that this
    normal ordered form is equivalent to the original expression.

    Parameters
    ==========

    expr : expression
        The expression write on normal ordered form.
    independent : bool (default False)
        Whether to consider operator with different names as operating in
        different Hilbert spaces. If False, the (anti-)commutation is left
        explicit.
    recursive_limit : int (default 10)
        The number of allowed recursive applications of the function.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.operatorordering import normal_ordered_form
    >>> a = BosonOp("a")
    >>> normal_ordered_form(a * Dagger(a))
    1 + Dagger(a)*a
    """

    # 如果递归深度超过了递归限制，则发出警告并返回原始表达式
    if _recursive_depth > recursive_limit:
        warnings.warn("Too many recursions, aborting")
        return expr

    # 如果表达式是加法表达式，则调用 _normal_ordered_form_terms 函数进行处理
    if isinstance(expr, Add):
        return _normal_ordered_form_terms(expr,
                                          recursive_limit=recursive_limit,
                                          _recursive_depth=_recursive_depth,
                                          independent=independent)
    # 如果表达式是乘法表达式，则调用 _normal_ordered_form_factor 函数进行处理
    elif isinstance(expr, Mul):
        return _normal_ordered_form_factor(expr,
                                           recursive_limit=recursive_limit,
                                           _recursive_depth=_recursive_depth,
                                           independent=independent)
    else:
        # 如果表达式不是加法或乘法表达式，则直接返回原始表达式
        return expr


# 定义一个名为 _normal_order_factor 的函数，用于对乘法表达式进行正规排序处理
# 参数:
#   product: 乘法表达式，可能包含多个因子
#   recursive_limit: 整数，默认为 10，递归调用函数的最大次数限制
#   _recursive_depth: 整数，默认为 0，当前递归深度
def _normal_order_factor(product, recursive_limit=10, _recursive_depth=0):
    """
    Helper function for normal_order: Normal order a multiplication expression
    with bosonic or fermionic operators. In general the resulting operator
    expression will not be equivalent to original product.
    """

    # 展开乘法表达式中的因子
    factors = _expand_powers(product)

    n = 0
    new_factors = []
    # 当前索引 n 小于 factors 列表长度减一时，执行循环
    while n < len(factors) - 1:

        # 检查 factors[n] 是否为 BosonOp 类型且为湮灭算符
        if (isinstance(factors[n], BosonOp) and
                factors[n].is_annihilation):
            # 如果是玻色子算符
            if not isinstance(factors[n + 1], BosonOp):
                # 下一个因子不是玻色子算符，将当前因子添加到新列表
                new_factors.append(factors[n])
            else:
                # 下一个因子是玻色子算符
                if factors[n + 1].is_annihilation:
                    # 下一个因子也是湮灭算符，将当前因子添加到新列表
                    new_factors.append(factors[n])
                else:
                    # 下一个因子不是湮灭算符但属于同一种类，合并两个因子并添加到新列表
                    new_factors.append(factors[n + 1] * factors[n])
                    # 增加索引以跳过已处理的下一个因子
                    n += 1

        # 检查 factors[n] 是否为 FermionOp 类型且为湮灭算符
        elif (isinstance(factors[n], FermionOp) and
              factors[n].is_annihilation):
            # 如果是费米子算符
            if not isinstance(factors[n + 1], FermionOp):
                # 下一个因子不是费米子算符，将当前因子添加到新列表
                new_factors.append(factors[n])
            else:
                # 下一个因子是费米子算符
                if factors[n + 1].is_annihilation:
                    # 下一个因子也是湮灭算符，将当前因子添加到新列表
                    new_factors.append(factors[n])
                else:
                    # 下一个因子不是湮灭算符但属于同一种类，合并两个因子并添加到新列表
                    new_factors.append(-factors[n + 1] * factors[n])
                    # 增加索引以跳过已处理的下一个因子
                    n += 1

        else:
            # 如果当前因子不是特定类型的算符，直接添加到新列表
            new_factors.append(factors[n])

        # 增加索引以处理下一个因子
        n += 1

    # 处理最后一个因子，若 n 等于 factors 列表长度减一，则将其添加到新列表
    if n == len(factors) - 1:
        new_factors.append(factors[-1])

    # 检查新生成的因子列表是否与原列表相同
    if new_factors == factors:
        return product
    else:
        # 使用 Mul 类将新因子列表展开成表达式
        expr = Mul(*new_factors).expand()
        # 递归调用 normal_order 函数，对表达式进行归一化处理
        return normal_order(expr,
                            recursive_limit=recursive_limit,
                            _recursive_depth=_recursive_depth + 1)
# 用于正规序排列表达式中的每个项，通过调用 _normal_order_factor 函数对因子进行正规序处理
def _normal_order_terms(expr, recursive_limit=10, _recursive_depth=0):
    """
    Helper function for normal_order: look through each term in an addition
    expression and call _normal_order_factor to perform the normal ordering
    on the factors.
    """

    new_terms = []
    for term in expr.args:
        if isinstance(term, Mul):
            # 对乘法项调用 _normal_order_factor 函数进行正规序处理
            new_term = _normal_order_factor(term,
                                            recursive_limit=recursive_limit,
                                            _recursive_depth=_recursive_depth)
            new_terms.append(new_term)
        else:
            new_terms.append(term)

    # 返回由处理后的新项组成的加法表达式
    return Add(*new_terms)


def normal_order(expr, recursive_limit=10, _recursive_depth=0):
    """Normal order an expression with bosonic or fermionic operators. Note
    that this normal order is not equivalent to the original expression, but
    the creation and annihilation operators in each term in expr is reordered
    so that the expression becomes normal ordered.

    Parameters
    ==========

    expr : expression
        The expression to normal order.

    recursive_limit : int (default 10)
        The number of allowed recursive applications of the function.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.operatorordering import normal_order
    >>> a = BosonOp("a")
    >>> normal_order(a * Dagger(a))
    Dagger(a)*a
    """
    # 如果表达式是加法，则调用 _normal_order_terms 函数进行正规序处理
    if isinstance(expr, Add):
        return _normal_order_terms(expr, recursive_limit=recursive_limit,
                                   _recursive_depth=_recursive_depth)
    # 如果表达式是乘法，则调用 _normal_order_factor 函数进行正规序处理
    elif isinstance(expr, Mul):
        return _normal_order_factor(expr, recursive_limit=recursive_limit,
                                    _recursive_depth=_recursive_depth)
    else:
        # 如果表达式不是加法也不是乘法，直接返回表达式本身
        return expr
```