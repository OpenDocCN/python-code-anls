# `D:\src\scipysrc\sympy\sympy\codegen\algorithms.py`

```
# 从 sympy 库中导入所需的模块和类
from sympy.core.containers import Tuple
from sympy.core.numbers import oo
from sympy.core.relational import (Gt, Lt)
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import And
from sympy.codegen.ast import (
    Assignment, AddAugmentedAssignment, break_, CodeBlock, Declaration, FunctionDefinition,
    Print, Return, Scope, While, Variable, Pointer, real
)
from sympy.codegen.cfunctions import isnan

""" This module collects functions for constructing ASTs representing algorithms. """

# 定义一个函数，用于生成 Newton-Raphson 方法的抽象语法树（AST）
def newtons_method(expr, wrt, atol=1e-12, delta=None, *, rtol=4e-16, debug=False,
                   itermax=None, counter=None, delta_fn=lambda e, x: -e/e.diff(x),
                   cse=False, handle_nan=None,
                   bounds=None):
    """ Generates an AST for Newton-Raphson method (a root-finding algorithm).

    Explanation
    ===========

    Returns an abstract syntax tree (AST) based on ``sympy.codegen.ast`` for Netwon's
    method of root-finding.

    Parameters
    ==========

    expr : expression
        The expression whose root is to be found.
    wrt : Symbol
        Symbol with respect to which differentiation is performed.
    atol : number or expression
        Absolute tolerance (stopping criterion)
    rtol : number or expression
        Relative tolerance (stopping criterion)
    delta : Symbol
        Symbol for the step size; if None, a Dummy symbol is created.
    debug : bool
        Whether to print convergence information during iterations
    itermax : number or expr
        Maximum number of iterations.
    counter : Symbol
        Symbol for the iteration counter; if None, a Dummy symbol is created.
    delta_fn: Callable[[Expr, Symbol], Expr]
        Function to compute the step size in each iteration.
    cse: bool
        Perform common sub-expression elimination on delta expression
    handle_nan: Token
        How to handle occurrence of not-a-number (NaN).
    bounds: Optional[tuple[Expr, Expr]]
        Perform optimization within bounds

    Examples
    ========

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.ast import Assignment
    >>> from sympy.codegen.algorithms import newtons_method
    >>> x, dx, atol = symbols('x dx atol')
    >>> expr = cos(x) - x**3
    >>> algo = newtons_method(expr, x, atol=atol, delta=dx)
    >>> algo.has(Assignment(dx, -expr/expr.diff(x)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Newton%27s_method

    """

    # 如果 delta 参数为 None，则创建一个 Dummy 符号作为步长
    if delta is None:
        delta = Dummy()
        Wrapper = Scope
        name_d = 'delta'
    else:
        Wrapper = lambda x: x
        name_d = delta.name

    # 计算步长表达式，根据是否进行常用子表达式消除决定如何处理
    delta_expr = delta_fn(expr, wrt)
    if cse:
        from sympy.simplify.cse_main import cse
        cses, (red,) = cse([delta_expr.factor()])
        whl_bdy = [Assignment(dum, sub_e) for dum, sub_e in cses]
        whl_bdy += [Assignment(delta, red)]
    # 如果 handle_nan 参数不为 None，则创建一个处理 NaN 的 While 循环体
    if handle_nan is not None:
        whl_bdy += [While(isnan(delta), CodeBlock(handle_nan, break_))]
    
    # 将 AddAugmentedAssignment 对象添加到循环体中，用于增加 wrt 变量的值
    whl_bdy += [AddAugmentedAssignment(wrt, delta)]
    
    # 如果 bounds 参数不为 None，则创建一个将 wrt 变量限制在指定范围内的 Assignment 对象
    if bounds is not None:
        whl_bdy += [Assignment(wrt, Min(Max(wrt, bounds[0]), bounds[1]))]
    
    # 如果 debug 参数为 True，则创建一个打印 wrt 和 delta 变量值的 Print 对象
    if debug:
        prnt = Print([wrt, delta], r"{}=%12.5g {}=%12.5g\n".format(wrt.name, name_d))
        whl_bdy += [prnt]
    
    # 创建一个要求条件，要求 Abs(delta) 大于 atol + rtol*Abs(wrt) 的 Gt 对象
    req = Gt(Abs(delta), atol + rtol*Abs(wrt))
    
    # 创建一个声明 delta 变量为无穷大的 Declaration 对象，并添加到 declars 列表中
    declars = [Declaration(Variable(delta, type=real, value=oo))]
    
    # 如果 itermax 参数不为 None，则创建一个计数器变量 counter，并将其添加到 declars 列表中
    if itermax is not None:
        counter = counter or Dummy(integer=True)
        v_counter = Variable.deduced(counter, 0)
        declars.append(Declaration(v_counter))
        whl_bdy.append(AddAugmentedAssignment(counter, 1))
        req = And(req, Lt(counter, itermax))
    
    # 创建一个 While 对象，该对象的循环条件为 req，循环体为 whl_bdy 中的代码块
    whl = While(req, CodeBlock(*whl_bdy))
    
    # 创建一个代码块 blck，包含 declars 中声明的变量和可能的 debug 打印语句以及循环对象 whl
    blck = declars
    if debug:
        blck.append(Print([wrt], r"{}=%12.5g\n".format(wrt.name)))
    blck += [whl]
    
    # 返回一个 Wrapper 对象，该对象包含整个 blck 代码块
    return Wrapper(CodeBlock(*blck))
# 定义一个函数，用于将输入参数转换为符号对象
def _symbol_of(arg):
    if isinstance(arg, Declaration):
        # 如果参数是 Declaration 类型，则取其变量符号作为结果
        arg = arg.variable.symbol
    elif isinstance(arg, Variable):
        # 如果参数是 Variable 类型，则取其符号作为结果
        arg = arg.symbol
    return arg


# 创建一个实现牛顿-拉夫逊方法的函数的抽象语法树（AST）
def newtons_method_function(expr, wrt, params=None, func_name="newton", attrs=Tuple(), *, delta=None, **kwargs):
    """ Generates an AST for a function implementing the Newton-Raphson method.

    Parameters
    ==========

    expr : expression
        待优化的表达式
    wrt : Symbol
        优化的变量，即微分对象
    params : iterable of symbols
        表达式中出现的常数符号，在迭代过程中将作为生成函数的参数
    func_name : str
        生成函数的名称
    attrs : Tuple
        传递给 FunctionDefinition 的属性实例
    \\*\\*kwargs :
        传递给 :func:`sympy.codegen.algorithms.newtons_method` 的关键字参数

    Examples
    ========

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.algorithms import newtons_method_function
    >>> from sympy.codegen.pyutils import render_as_module
    >>> x = symbols('x')
    >>> expr = cos(x) - x**3
    >>> func = newtons_method_function(expr, x)
    >>> py_mod = render_as_module(func)  # 将生成的源代码作为字符串
    >>> namespace = {}
    >>> exec(py_mod, namespace, namespace)
    >>> res = eval('newton(0.5)', namespace)
    >>> abs(res - 0.865474033102) < 1e-12
    True

    See Also
    ========

    sympy.codegen.algorithms.newtons_method

    """
    # 如果 params 为空，则默认将 wrt 作为参数
    if params is None:
        params = (wrt,)
    
    # 如果 params 中有 Pointer 类型的符号，替换为指针的符号表示
    pointer_subs = {p.symbol: Symbol('(*%s)' % p.symbol.name)
                    for p in params if isinstance(p, Pointer)}
    
    # 如果 delta 未指定，则为 wrt 创建一个新的符号作为步长
    if delta is None:
        delta = Symbol('d_' + wrt.name)
        if expr.has(delta):
            delta = None  # 将使用 Dummy 作为步长符号
    
    # 调用 newtons_method 函数生成算法的表达式，并替换指针符号
    algo = newtons_method(expr, wrt, delta=delta, **kwargs).xreplace(pointer_subs)
    
    # 如果 algo 是 Scope 类型，则获取其主体部分作为算法体
    if isinstance(algo, Scope):
        algo = algo.body
    
    # 检查表达式中是否有不在 params 中的自由符号
    not_in_params = expr.free_symbols.difference({_symbol_of(p) for p in params})
    if not_in_params:
        raise ValueError("Missing symbols in params: %s" % ', '.join(map(str, not_in_params)))
    
    # 创建函数定义所需的变量声明
    declars = tuple(Variable(p, real) for p in params)
    
    # 将算法体和返回语句组成函数体的代码块
    body = CodeBlock(algo, Return(wrt))
    
    # 返回函数定义对象
    return FunctionDefinition(real, func_name, declars, body, attrs=attrs)
```