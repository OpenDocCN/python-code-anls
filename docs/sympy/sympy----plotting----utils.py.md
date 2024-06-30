# `D:\src\scipysrc\sympy\sympy\plotting\utils.py`

```
# 导入所需的符号计算模块中的具体类
from sympy.core.containers import Tuple
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import BooleanFunction
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import FiniteSet
from sympy.tensor.indexed import Indexed


def _get_free_symbols(exprs):
    """Returns the free symbols of a symbolic expression.

    If the expression contains any of these elements, assume that they are
    the "free symbols" of the expression:

    * indexed objects
    * applied undefined function (useful for sympy.physics.mechanics module)
    """
    # 如果参数不是列表、元组或集合，则将其转换为列表
    if not isinstance(exprs, (list, tuple, set)):
        exprs = [exprs]
    # 如果所有表达式都是可调用的，则返回空集合
    if all(callable(e) for e in exprs):
        return set()

    # 收集所有表达式中的 Indexed 对象
    free = set().union(*[e.atoms(Indexed) for e in exprs])
    # 收集所有表达式中的 AppliedUndef 对象
    free = free.union(*[e.atoms(AppliedUndef) for e in exprs])
    # 如果以上两步没有收集到任何自由符号，则返回表达式中所有的自由符号
    return free or set().union(*[e.free_symbols for e in exprs])


def extract_solution(set_sol, n=10):
    """Extract numerical solutions from a set solution (computed by solveset,
    linsolve, nonlinsolve). Often, it is not trivial do get something useful
    out of them.

    Parameters
    ==========

    n : int, optional
        In order to replace ImageSet with FiniteSet, an iterator is created
        for each ImageSet contained in `set_sol`, starting from 0 up to `n`.
        Default value: 10.
    """
    # 查找解集中的所有 ImageSet 对象
    images = set_sol.find(ImageSet)
    # 对每个 ImageSet 对象进行处理
    for im in images:
        # 创建一个迭代器来逐个获取 ImageSet 中的元素，最多获取 n 个元素
        it = iter(im)
        s = FiniteSet(*[next(it) for _ in range(0, n)])
        # 将原解集中的 ImageSet 对象替换为 FiniteSet 对象
        set_sol = set_sol.subs(im, s)
    return set_sol


def _plot_sympify(args):
    """This function recursively loop over the arguments passed to the plot
    functions: the sympify function will be applied to all arguments except
    those of type string/dict.

    Generally, users can provide the following arguments to a plot function:

    expr, range1 [tuple, opt], ..., label [str, opt], rendering_kw [dict, opt]

    `expr, range1, ...` can be sympified, whereas `label, rendering_kw` can't.
    In particular, whenever a special character like $, {, }, ... is used in
    the `label`, sympify will raise an error.
    """
    # 如果参数是 Expr 类型，则直接返回
    if isinstance(args, Expr):
        return args

    # 如果参数是列表，则逐个处理列表中的元素
    args = list(args)
    for i, a in enumerate(args):
        # 如果元素是列表或元组，则递归调用 _plot_sympify 处理其内部元素
        if isinstance(a, (list, tuple)):
            args[i] = Tuple(*_plot_sympify(a), sympify=False)
        # 如果元素不是字符串、字典，也不是可调用对象（并且不是向量），则对其进行 sympify 处理
        elif not (isinstance(a, (str, dict)) or callable(a)
                  or ((a.__class__.__name__ == "Vector") and not isinstance(a, Basic))):
            args[i] = sympify(a)
    return args
# 定义一个函数用于创建变量范围的元组，如果未指定范围则使用默认范围 (-10, 10)
get_default_range = lambda symbol: Tuple(symbol, -10, 10)

# 创建一个函数用于提取表达式中的自由符号
free_symbols = _get_free_symbols(exprs)

# 如果参数 params 存在，则从自由符号集合中排除已在 params 中提供的符号
if params is not None:
    free_symbols = free_symbols.difference(params.keys())

# 检查自由符号的数量是否与所需的自由符号数量 npar 相符，否则引发 ValueError 异常
if len(free_symbols) > npar:
    raise ValueError(
        "Too many free symbols.\n"
        + "Expected {} free symbols.\n".format(npar)
        + "Received {}: {}".format(len(free_symbols), free_symbols)
    )

# 检查用户提供的范围数量是否超过了所需的范围数量 npar，否则引发 ValueError 异常
if len(ranges) > npar:
    raise ValueError(
        "Too many ranges. Received %s, expected %s" % (len(ranges), npar))

# 检查用户提供的范围中是否存在相同的符号，如果存在则引发 ValueError 异常
rfs = set().union([r[0] for r in ranges])
if len(rfs) != len(ranges):
    raise ValueError("Multiple ranges with the same symbol")

# 如果用户提供的范围数量少于所需的范围数量 npar，则为每个缺失的自由符号添加默认范围
if len(ranges) < npar:
    symbols = free_symbols.difference(rfs)
    if symbols != set():
        for s in symbols:
            ranges.append(get_default_range(s))
    # 如果仍有空位，则用 Dummy 符号填充
    for i in range(npar - len(ranges)):
        ranges.append(get_default_range(Dummy()))

# 如果自由符号的数量等于所需的自由符号数量 npar，则检查用户提供的范围是否与自由符号匹配
if len(free_symbols) == npar:
    rfs = set().union([r[0] for r in ranges])
    if len(free_symbols.difference(rfs)) > 0:
        raise ValueError(
            "Incompatible free symbols of the expressions with "
            "the ranges.\n"
            + "Free symbols in the expressions: {}\n".format(free_symbols)
            + "Free symbols in the ranges: {}".format(rfs)
        )

# 返回经过处理的范围列表
return ranges
    # 检查返回值 r 是否为元组类型
    return (
        # 判断 r 是否为元组类型
        isinstance(r, Tuple)
        # 检查元组长度是否为 3
        and (len(r) == 3)
        # 确保第二个元素不是字符串且是数字类型
        and (not isinstance(r.args[1], str)) and r.args[1].is_number
        # 确保第三个元素不是字符串且是数字类型
        and (not isinstance(r.args[2], str)) and r.args[2].is_number
    )
# 接受任意数量的参数，并对其进行解包，前提是这些参数已经被 _plot_sympify() 和/或 _check_arguments() 处理过
def _unpack_args(*args):
    # 将 args 中符合条件的部分提取出来作为 ranges 列表，这些部分是表示范围的元组或列表
    ranges = [t for t in args if _is_range(t)]
    # 将 args 中的字符串类型的参数提取出来作为 labels 列表，这些是用于标签的字符串
    labels = [t for t in args if isinstance(t, str)]
    # 如果 labels 非空，则取第一个元素作为标签；否则 label 为 None
    label = None if not labels else labels[0]
    # 将 args 中的字典类型参数提取出来作为 rendering_kw，这些是用于渲染的关键字参数
    rendering_kw = [t for t in args if isinstance(t, dict)]
    # 如果 rendering_kw 非空，则取第一个元素作为 rendering_kw；否则 rendering_kw 为 None
    rendering_kw = None if not rendering_kw else rendering_kw[0]
    # NOTE: 为什么是 None？因为 args 可能已经被 _check_arguments 预处理，所以 None 可能表示 rendering_kw
    # 判断 args 中哪些元素是表达式，构建 exprs 列表，这些元素是可以绘制的表达式
    results = [not (_is_range(a) or isinstance(a, (str, dict)) or (a is None)) for a in args]
    exprs = [a for a, b in zip(args, results) if b]
    # 返回表达式列表、范围列表、标签和渲染关键字参数
    return exprs, ranges, label, rendering_kw


# 检查参数并将其转换为形如 (exprs, ranges, label, rendering_kw) 的元组
def _check_arguments(args, nexpr, npar, **kwargs):
    # args: 绘图函数接收到的参数
    # nexpr: 组成要绘制表达式的子表达式数量
    # npar: 绘图函数需要的自由符号数量
    # **kwargs: 传递给绘图函数的关键字参数，用于验证是否提供了 "params"
    """
    if not args:
        如果参数列表为空，则返回空列表
        return []
    output = []
    初始化空的输出列表
    params = kwargs.get("params", None)
    从关键字参数中获取"params"对应的值，如果不存在则为None

    if all(isinstance(a, (Expr, Relational, BooleanFunction)) for a in args[:nexpr]):
        如果参数列表中前nexpr个参数都是 SymPy 的表达式、关系或布尔函数：
        # 在这种情况下，使用单个绘图命令，我们要绘制：
        #   1. 一个表达式
        #   2. 多个表达式在相同的范围内

        exprs, ranges, label, rendering_kw = _unpack_args(*args)
        调用 _unpack_args 函数解包参数，得到表达式列表、范围、标签和渲染关键字
        free_symbols = set().union(*[e.free_symbols for e in exprs])
        获取所有表达式中的自由符号的并集
        ranges = _create_ranges(exprs, ranges, npar, label, params)
        调用 _create_ranges 函数生成表达式的范围列表，用于绘图，考虑到标签和参数

        if nexpr > 1:
            如果表达式数量大于1：
            # 对于 plot_parametric 或 plot3d_parametric_line，会定义一个曲线的2或3个表达式。将它们组合在一起。
            if len(exprs) == nexpr:
                如果表达式的数量等于 nexpr：
                exprs = (tuple(exprs),)
            将表达式列表组合成一个元组

        for expr in exprs:
            遍历表达式列表
            # 需要这个 if-else 来处理 plot/plot3d 和 plot_parametric/plot3d_parametric_line 两种情况
            is_expr = isinstance(expr, (Expr, Relational, BooleanFunction))
            判断当前表达式是否为 SymPy 的表达式、关系或布尔函数类型
            e = (expr,) if is_expr else expr
            如果是，将其封装为元组；否则直接使用原始表达式
            output.append((*e, *ranges, label, rendering_kw))
            将表达式、范围、标签和渲染关键字添加到输出列表中
    """
    else:
        # 如果进入这个分支，表示需要绘制多个表达式，每个表达式都有自己的范围。
        # 每个要绘制的“表达式”具有以下形式：(expr, range, label)，其中label是可选的

        # 解包参数，获取表达式、范围、标签和渲染关键字
        _, ranges, labels, rendering_kw = _unpack_args(*args)
        labels = [labels] if labels else []

        # 计算表达式的数量
        n = (len(ranges) + len(labels) +
            (len(rendering_kw) if rendering_kw is not None else 0))
        new_args = args[:-n] if n > 0 else args

        # 此时，new_args可能是[expr]的形式，但我需要它是[[expr]]的形式，以便能够循环遍历
        # [expr, range [opt], label [opt]]
        if not isinstance(new_args[0], (list, tuple, Tuple)):
            new_args = [new_args]

        # 每个参数的形式是(expr1, expr2, ..., range1 [可选], ...,
        #   label [可选], rendering_kw [可选])
        for arg in new_args:
            # 查找“局部”范围和标签。如果不存在，则使用“全局”的范围和标签。
            l = [a for a in arg if isinstance(a, str)]
            if not l:
                l = labels
            r = [a for a in arg if _is_range(a)]
            if not r:
                r = ranges.copy()
            rend_kw = [a for a in arg if isinstance(a, dict)]
            rend_kw = rendering_kw if len(rend_kw) == 0 else rend_kw[0]

            # 注意：arg = arg[:nexpr]可能会引发异常，如果使用了lambda函数，改为执行以下操作：
            arg = [arg[i] for i in range(nexpr)]
            free_symbols = set()
            if all(not callable(a) for a in arg):
                free_symbols = free_symbols.union(*[a.free_symbols for a in arg])
            if len(r) != npar:
                r = _create_ranges(arg, r, npar, "", params)

            label = None if not l else l[0]
            output.append((*arg, *r, label, rend_kw))
    return output
```