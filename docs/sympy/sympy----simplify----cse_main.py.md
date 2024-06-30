# `D:\src\scipysrc\sympy\sympy\simplify\cse_main.py`

```
# 导入必要的模块和函数
from collections import defaultdict  # 导入defaultdict类，用于创建默认值为list的字典

from sympy.core import Basic, Mul, Add, Pow, sympify  # 导入SymPy核心模块的相关类和函数
from sympy.core.containers import Tuple, OrderedSet  # 导入SymPy核心容器模块的相关类
from sympy.core.exprtools import factor_terms  # 导入因式分解函数
from sympy.core.singleton import S  # 导入SymPy单例模块的S对象
from sympy.core.sorting import ordered  # 导入排序函数
from sympy.core.symbol import symbols, Symbol  # 导入符号类和符号生成函数
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,  # 导入矩阵相关类
                            SparseMatrix, ImmutableSparseMatrix)
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,  # 导入矩阵表达式相关类和函数
                                        MatAdd, MatPow, Inverse)
from sympy.matrices.expressions.matexpr import MatrixElement  # 导入矩阵元素类
from sympy.polys.rootoftools import RootOf  # 导入根式的工具函数
from sympy.utilities.iterables import (numbered_symbols, sift,  # 导入迭代工具函数
                                       topological_sort, iterable)

from . import cse_opts  # 导入本地的cse_opts模块

# 定义一对(preprocessor, postprocessor)元组，用于常见的优化操作
basic_optimizations = [(cse_opts.sub_pre, cse_opts.sub_post),
                       (factor_terms, None)]

def reps_toposort(r):
    """对替换列表``r``进行排序，使得(k1, v1)在(k2, v2)之前出现，
    如果k2在v1的自由符号中出现。这种顺序与cse函数返回其结果的顺序相匹配
    （因此，为了在替换选项中使用这些替换，反转排序顺序是合理的）。

    Examples
    ========

    >>> from sympy.simplify.cse_main import reps_toposort
    >>> from sympy.abc import x, y
    >>> from sympy import Eq
    >>> for l, r in reps_toposort([(x, y + 1), (y, 2)]):
    ...     print(Eq(l, r))
    ...
    Eq(y, 2)
    Eq(x, y + 1)

    """
    r = sympify(r)  # 将输入的替换列表r转换为SymPy表达式
    E = []
    for c1, (k1, v1) in enumerate(r):
        for c2, (k2, v2) in enumerate(r):
            if k1 in v2.free_symbols:  # 如果k1出现在v2的自由符号中
                E.append((c1, c2))
    return [r[i] for i in topological_sort((range(len(r)), E))]

def cse_separate(r, e):
    """从表达式e中移出形式为(symbol, expr)的表达式，并使用reps_toposort对替换r进行排序。

    Examples
    ========

    >>> from sympy.simplify.cse_main import cse_separate
    >>> from sympy.abc import x, y, z
    >>> from sympy import cos, exp, cse, Eq, symbols  # 导入 sympy 库中的相关函数和类
    >>> x0, x1 = symbols('x:2')  # 定义符号变量 x0 和 x1
    >>> eq = (x + 1 + exp((x + 1)/(y + 1)) + cos(y + 1))  # 定义一个复杂的表达式 eq
    >>> cse([eq, Eq(x, z + 1), z - 2], postprocess=cse_separate) in [  # 对 eq, x=z+1 和 z-2 进行公共子表达式消除 (CSE)
    ... [[(x0, y + 1), (x, z + 1), (x1, x + 1)],  # 第一个结果：公共子表达式消除的结果 [(x0, y+1), (x, z+1), (x1, x+1)]
    ...  [x1 + exp(x1/x0) + cos(x0), z - 2]],  # 对应的简化后的表达式
    ... [[(x1, y + 1), (x, z + 1), (x0, x + 1)],  # 第二个结果：公共子表达式消除的结果 [(x1, y+1), (x, z+1), (x0, x+1)]
    ...  [x0 + exp(x0/x1) + cos(x1), z - 2]]]  # 对应的简化后的表达式
    ...
    True  # 验证以上两个结果是否在列表中，返回 True 表示验证通过
    """
    d = sift(e, lambda w: w.is_Equality and w.lhs.is_Symbol)  # 使用 sift 函数筛选出 e 中所有左侧为符号的等式
    r = r + [w.args for w in d[True]]  # 将所有满足条件的等式左侧符号添加到 r 列表中
    e = d[False]  # 更新 e，移除所有满足条件的等式左侧符号
    return [reps_toposort(r), e]  # 返回经过拓扑排序后的 r 列表和更新后的 e
# 对于给定的 CSE 结果 `r` 和表达式列表 `e`，返回一组元组 `(a, b)`，其中 `a` 是一个符号，`b` 是一个表达式或者 None。
# 当一个符号在后续表达式中不再需要时，其值为 None。
def cse_release_variables(r, e):
    if not r:
        # 如果 `r` 为空，则直接返回原始输入 `r` 和 `e`
        return r, e

    # 将 `r` 中的符号和表达式分开
    s, p = zip(*r)
    
    # 生成一个新的符号列表 `esyms`，用于替代表达式中的符号
    esyms = symbols('_:%d' % len(e))
    syms = list(esyms)
    
    s = list(s)
    in_use = set(s)
    p = list(p)
    
    # 根据子表达式数量的多少，对 `e` 进行排序，以便优先处理具有较多子表达式的表达式
    e = [(e[i], syms[i]) for i in range(len(e))]
    e, syms = zip(*sorted(e,
        key=lambda x: -sum(p[s.index(i)].count_ops()
        for i in x[0].free_symbols & in_use)))
    syms = list(syms)
    p += e
    rv = []
    i = len(p) - 1
    
    # 反向处理表达式列表 `p`
    while i >= 0:
        _p = p.pop()
        c = in_use & _p.free_symbols
        if c:  # 如果存在被使用的符号，则按照字符串排序后加入结果列表 `rv`
            rv.extend([(s, None) for s in sorted(c, key=str)])
        if i >= len(r):
            rv.append((syms.pop(), _p))
        else:
            rv.append((s[i], _p))
        in_use -= c
        i -= 1
    
    # 反转结果列表 `rv` 并返回
    rv.reverse()
    return rv, esyms


# ====end of cse postprocess idioms===========================


# 对表达式进行预处理，以优化常见子表达式消除
def preprocess_for_cse(expr, optimizations):
    """ Preprocess an expression to optimize for common subexpression
    elimination.

    Parameters
    ==========

    expr : SymPy expression
        The target expression to optimize.
    optimizations : list of (callable, callable) pairs
        The (preprocessor, postprocessor) pairs.

    Returns
    =======

    expr : SymPy expression
        The transformed expression.
    """
    for pre, post in optimizations:
        if pre is not None:
            expr = pre(expr)
    return expr


# 对经过常见子表达式消除处理后的表达式进行后处理，返回 SymPy 的规范形式
def postprocess_for_cse(expr, optimizations):
    """Postprocess an expression after common subexpression elimination to
    return the expression to canonical SymPy form.

    Parameters
    ==========

    expr : SymPy expression
        The target expression to transform.
    optimizations : list of (callable, callable) pairs, optional
        The (preprocessor, postprocessor) pairs.  The postprocessors will be
        applied in reversed order to undo the effects of the preprocessors
        correctly.

    Returns
    =======

    expr : SymPy expression
        The transformed expression.
    """
    pass  # 这里需要实现对表达式的后处理，以返回 SymPy 的规范形式
    # 对给定的 SymPy 表达式应用一系列优化操作，这些操作被存储在 optimizations 列表中
    for pre, post in reversed(optimizations):
        # 反向遍历优化操作，pre 是预处理函数，post 是后处理函数
        if post is not None:
            # 如果后处理函数不为空，应用后处理函数 post 到表达式 expr 上
            expr = post(expr)
    # 返回经过所有优化操作处理后的表达式
    return expr
    """
    A class which manages a mapping from functions to arguments and an inverse
    mapping from arguments to functions.
    """

    def __init__(self, funcs):
        # 初始化空字典，用于存储每个值的值编号
        self.value_numbers = {}
        # 初始化空列表，用于按值编号存储值本身
        self.value_number_to_value = []

        # 初始化两个空列表，用于存储参数到函数集合的映射和函数到参数集合的映射
        self.arg_to_funcset = []
        self.func_to_argset = []

        # 遍历传入的函数列表
        for func_i, func in enumerate(funcs):
            # 初始化有序集合，用于存储当前函数的参数的值编号
            func_argset = OrderedSet()

            # 遍历当前函数的参数列表
            for func_arg in func.args:
                # 获取参数的值编号，如果参数已存在则返回现有的编号，否则添加新的编号
                arg_number = self.get_or_add_value_number(func_arg)
                # 将参数的值编号添加到当前函数的参数集合中
                func_argset.add(arg_number)
                # 将当前函数的索引添加到参数到函数集合的对应位置
                self.arg_to_funcset[arg_number].add(func_i)

            # 将当前函数的参数集合添加到函数到参数集合的列表中
            self.func_to_argset.append(func_argset)

    def get_args_in_value_order(self, argset):
        """
        Return the list of arguments in sorted order according to their value
        numbers.
        """
        # 根据值编号的顺序返回参数列表
        return [self.value_number_to_value[argn] for argn in sorted(argset)]

    def get_or_add_value_number(self, value):
        """
        Return the value number for the given argument.
        """
        # 获取当前已有的值编号数量
        nvalues = len(self.value_numbers)
        # 获取值对应的值编号，如果值尚未存在则添加新的值编号，并将值添加到值列表中
        value_number = self.value_numbers.setdefault(value, nvalues)
        if value_number == nvalues:
            self.value_number_to_value.append(value)
            self.arg_to_funcset.append(OrderedSet())
        return value_number

    def stop_arg_tracking(self, func_i):
        """
        Remove the function func_i from the argument to function mapping.
        """
        # 遍历函数 func_i 对应的参数集合
        for arg in self.func_to_argset[func_i]:
            # 从参数到函数集合中移除函数 func_i 的索引
            self.arg_to_funcset[arg].remove(func_i)
    def get_common_arg_candidates(self, argset, min_func_i=0):
        """
        返回一个字典，其键为函数编号。字典中的条目表示该函数与参数集合 ``argset`` 共享的参数数量。
        所有键的值至少为 ``min_func_i``。
        """
        # 创建一个默认值为0的默认字典，用于存储函数编号与共同参数数量的映射关系
        count_map = defaultdict(lambda: 0)
        
        # 如果参数集合为空，则直接返回空的 count_map
        if not argset:
            return count_map

        # 生成一个包含每个参数对应函数集合的列表
        funcsets = [self.arg_to_funcset[arg] for arg in argset]
        
        # 优化：单独处理最大的函数集合，避免重复处理
        largest_funcset = max(funcsets, key=len)

        # 遍历除了最大函数集合之外的其他函数集合
        for funcset in funcsets:
            if largest_funcset is funcset:
                continue
            # 对于每个函数编号，如果大于等于 min_func_i，则增加其在 count_map 中的计数
            for func_i in funcset:
                if func_i >= min_func_i:
                    count_map[func_i] += 1

        # 选择较小的容器（count_map, largest_funcset）来进行迭代，以减少迭代次数
        (smaller_funcs_container,
         larger_funcs_container) = sorted(
                 [largest_funcset, count_map],
                 key=len)

        # 遍历较小的函数集合容器
        for func_i in smaller_funcs_container:
            # 如果该函数编号不在 count_map 中，则跳过
            if count_map[func_i] < 1:
                continue

            # 如果该函数编号也在较大的函数集合容器中，则增加其在 count_map 中的计数
            if func_i in larger_funcs_container:
                count_map[func_i] += 1

        # 过滤掉 count_map 中值小于2的条目，并返回结果字典
        return {k: v for k, v in count_map.items() if v >= 2}

    def get_subset_candidates(self, argset, restrict_to_funcset=None):
        """
        返回一个函数集合，其中每个函数的参数列表包含 ``argset``，可选地仅包含 ``restrict_to_funcset`` 中的函数。
        """
        # 获取参数集合的迭代器
        iarg = iter(argset)

        # 初始化一个有序集合，包含第一个参数对应的函数集合
        indices = OrderedSet(
            fi for fi in self.arg_to_funcset[next(iarg)])

        # 如果指定了限制函数集合，则筛选出交集
        if restrict_to_funcset is not None:
            indices &= restrict_to_funcset

        # 对于剩余的参数，继续筛选交集函数集合
        for arg in iarg:
            indices &= self.arg_to_funcset[arg]

        # 返回结果函数集合
        return indices

    def update_func_argset(self, func_i, new_argset):
        """
        更新函数的参数集合。
        """
        # 将新参数集合转换为有序集合
        new_args = OrderedSet(new_argset)
        # 获取原始参数集合
        old_args = self.func_to_argset[func_i]

        # 从参数对应函数集合中移除删除的参数
        for deleted_arg in old_args - new_args:
            self.arg_to_funcset[deleted_arg].remove(func_i)
        # 将新增的参数添加到参数对应的函数集合中
        for added_arg in new_args - old_args:
            self.arg_to_funcset[added_arg].add(func_i)

        # 清空函数到参数集合的映射，更新为新的参数集合
        self.func_to_argset[func_i].clear()
        self.func_to_argset[func_i].update(new_args)
class Unevaluated:

    def __init__(self, func, args):
        self.func = func  # 初始化函数对象
        self.args = args  # 初始化函数参数列表

    def __str__(self):
        return "Uneval<{}>({})".format(
                self.func, ", ".join(str(a) for a in self.args))
        # 返回描述该对象的字符串表示形式

    def as_unevaluated_basic(self):
        return self.func(*self.args, evaluate=False)
        # 调用函数对象，并以未求值状态返回其结果

    @property
    def free_symbols(self):
        return set().union(*[a.free_symbols for a in self.args])
        # 返回该对象中所有自由符号的集合

    __repr__ = __str__


def match_common_args(func_class, funcs, opt_subs):
    """
    Recognize and extract common subexpressions of function arguments within a
    set of function calls. For instance, for the following function calls::

        x + z + y
        sin(x + y)

    this will extract a common subexpression of `x + y`::

        w = x + y
        w + z
        sin(w)

    The function we work with is assumed to be associative and commutative.

    Parameters
    ==========

    func_class: class
        The function class (e.g. Add, Mul)
        函数类别，例如加法、乘法等

    funcs: list of functions
        A list of function calls.
        函数调用列表

    opt_subs: dict
        A dictionary of substitutions which this function may update.
        可能更新的替换字典
    """

    # Sort to ensure that whole-function subexpressions come before the items
    # that use them.
    funcs = sorted(funcs, key=lambda f: len(f.args))
    # 按照函数参数长度对函数调用列表进行排序
    arg_tracker = FuncArgTracker(funcs)
    # 使用函数调用列表初始化函数参数追踪器对象

    changed = OrderedSet()
    # 创建一个有序集合对象，用于存储变化的元素
    # 遍历函数列表 `funcs` 中的每一个索引 `i`
    for i in range(len(funcs)):
        # 获取从当前函数 `i` 开始，与其它函数共享的常见参数候选项的计数
        common_arg_candidates_counts = arg_tracker.get_common_arg_candidates(
                arg_tracker.func_to_argset[i], min_func_i=i + 1)

        # 将共享参数候选项按匹配大小排序，以便先尝试组合较小的匹配
        common_arg_candidates = OrderedSet(sorted(
                common_arg_candidates_counts.keys(),
                key=lambda k: (common_arg_candidates_counts[k], k)))

        # 循环处理所有的共享参数候选项
        while common_arg_candidates:
            # 从共享参数候选项中弹出一个候选项 `j`
            j = common_arg_candidates.pop(last=False)

            # 计算函数 `i` 和函数 `j` 共同的参数集合
            com_args = arg_tracker.func_to_argset[i].intersection(
                    arg_tracker.func_to_argset[j])

            # 如果共同参数数量少于等于1，则跳过当前迭代
            if len(com_args) <= 1:
                # 这种情况可能发生在先前的迭代中，一组共同参数已经被合并了
                continue

            # 对所有集合，用函数在其上的运算来替换共同的符号，以允许递归匹配

            # 计算函数 `i` 中不属于共同参数的差集 `diff_i`
            diff_i = arg_tracker.func_to_argset[i].difference(com_args)
            if diff_i:
                # 创建一个未评估的 `com_func` 对象，以允许递归匹配
                com_func = Unevaluated(
                        func_class, arg_tracker.get_args_in_value_order(com_args))
                # 获取或添加 `com_func` 的数值编号
                com_func_number = arg_tracker.get_or_add_value_number(com_func)
                # 更新函数 `i` 的参数集合，将差集和 `com_func_number` 加入
                arg_tracker.update_func_argset(i, diff_i | OrderedSet([com_func_number]))
                # 将函数 `i` 标记为已更改
                changed.add(i)
            else:
                # 将整个表达式视为一个公共子表达式 (CSE)
                #
                # 这个操作的原因有些微妙。在 `tree_cse()` 中，`to_eliminate` 只包含
                # 那些出现超过一次的表达式。问题在于未评估的表达式与评估后的等效表达式不相等。
                # 因此，如果我们使用未评估的版本，`tree_cse()` 将不会将 `funcs[i]` 标记为 CSE。
                com_func_number = arg_tracker.get_or_add_value_number(funcs[i])

            # 计算函数 `j` 中不属于共同参数的差集 `diff_j`
            diff_j = arg_tracker.func_to_argset[j].difference(com_args)
            # 更新函数 `j` 的参数集合，将差集和 `com_func_number` 加入
            arg_tracker.update_func_argset(j, diff_j | OrderedSet([com_func_number]))
            # 将函数 `j` 标记为已更改
            changed.add(j)

            # 对于能够包含共同参数 `com_args` 的所有函数的子集候选项 `k`
            for k in arg_tracker.get_subset_candidates(
                    com_args, common_arg_candidates):
                # 计算函数 `k` 中不属于共同参数的差集 `diff_k`
                diff_k = arg_tracker.func_to_argset[k].difference(com_args)
                # 更新函数 `k` 的参数集合，将差集和 `com_func_number` 加入
                arg_tracker.update_func_argset(k, diff_k | OrderedSet([com_func_number]))
                # 将函数 `k` 标记为已更改
                changed.add(k)

        # 如果函数 `i` 已经在本轮迭代中被更改过
        if i in changed:
            # 将优化替换映射加入 `opt_subs` 中，以将 `funcs[i]` 替换为未评估的 `com_func`
            opt_subs[funcs[i]] = Unevaluated(func_class,
                arg_tracker.get_args_in_value_order(arg_tracker.func_to_argset[i]))

        # 停止跟踪函数 `i` 的参数
        arg_tracker.stop_arg_tracking(i)
def opt_cse(exprs, order='canonical'):
    """Find optimization opportunities in Adds, Muls, Pows and negative
    coefficient Muls.

    Parameters
    ==========

    exprs : list of SymPy expressions
        The expressions to optimize.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.

    Returns
    =======

    opt_subs : dictionary of expression substitutions
        The expression substitutions which can be useful to optimize CSE.

    Examples
    ========

    >>> from sympy.simplify.cse_main import opt_cse
    >>> from sympy.abc import x
    >>> opt_subs = opt_cse([x**-2])
    >>> k, v = list(opt_subs.keys())[0], list(opt_subs.values())[0]
    >>> print((k, v.as_unevaluated_basic()))
    (x**(-2), 1/(x**2))
    """
    opt_subs = {}  # 存储优化的表达式替换

    adds = OrderedSet()  # 用于存储加法操作的集合，保持顺序
    muls = OrderedSet()  # 用于存储乘法操作的集合，保持顺序

    seen_subexp = set()  # 记录已经处理过的子表达式
    collapsible_subexp = set()  # 可折叠的子表达式集合

    def _find_opts(expr):
        # 递归函数，用于查找优化机会的表达式

        if not isinstance(expr, (Basic, Unevaluated)):
            return

        if expr.is_Atom or expr.is_Order:
            return

        if iterable(expr):
            list(map(_find_opts, expr))
            return

        if expr in seen_subexp:
            return expr
        seen_subexp.add(expr)

        list(map(_find_opts, expr.args))

        if not isinstance(expr, MatrixExpr) and expr.could_extract_minus_sign():
            # 处理可能提取负号的表达式
            if isinstance(expr, Add):
                neg_expr = Add(*(-i for i in expr.args))
            else:
                neg_expr = -expr

            if not neg_expr.is_Atom:
                opt_subs[expr] = Unevaluated(Mul, (S.NegativeOne, neg_expr))
                seen_subexp.add(neg_expr)
                expr = neg_expr

        if isinstance(expr, (Mul, MatMul)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                muls.add(expr)

        elif isinstance(expr, (Add, MatAdd)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                adds.add(expr)

        elif isinstance(expr, Inverse):
            # 不处理 `Inverse` 作为 `MatPow`
            pass

        elif isinstance(expr, (Pow, MatPow)):
            base, exp = expr.base, expr.exp
            if exp.could_extract_minus_sign():
                opt_subs[expr] = Unevaluated(Pow, (Pow(base, -exp), -1))

    for e in exprs:
        if isinstance(e, (Basic, Unevaluated)):
            _find_opts(e)

    # 处理具有单个参数的多变量操作的折叠
    edges = [(s, s.args[0]) for s in collapsible_subexp
             if s.args[0] in collapsible_subexp]
    # 对拓扑排序的结果进行反向迭代处理
    for e in reversed(topological_sort((collapsible_subexp, edges))):
        # 将每个节点的优化子表达式存储到字典 opt_subs 中，如果已存在则保留原始值
        opt_subs[e] = opt_subs.get(e.args[0], e.args[0])

    # 将乘法操作符分解为可交换的部分
    commutative_muls = OrderedSet()
    # 遍历每个乘法操作
    for m in muls:
        # 将乘法表达式 m 拆分为可交换部分 c 和不可交换部分 nc
        c, nc = m.args_cnc(cset=False)
        # 如果存在可交换部分
        if c:
            # 创建只包含可交换部分的新乘法表达式 c_mul
            c_mul = m.func(*c)
            # 如果还存在不可交换部分
            if nc:
                # 如果可交换部分 c_mul 等于 1，则新对象直接为不可交换部分的乘法表达式
                if c_mul == 1:
                    new_obj = m.func(*nc)
                else:
                    # 否则根据原始乘法类型创建新对象，包含可交换部分和不可交换部分
                    if isinstance(m, MatMul):
                        new_obj = m.func(c_mul, *nc, evaluate=False)
                    else:
                        new_obj = m.func(c_mul, m.func(*nc), evaluate=False)
                # 将原始乘法表达式 m 映射到新对象，用于后续替换优化
                opt_subs[m] = new_obj
            # 如果可交换部分超过一个元素，则将 c_mul 添加到可交换乘法集合中
            if len(c) > 1:
                commutative_muls.add(c_mul)

    # 匹配加法操作中的公共参数，并进行优化替换
    match_common_args(Add, adds, opt_subs)
    # 匹配乘法操作中的可交换乘法集合中的公共参数，并进行优化替换
    match_common_args(Mul, commutative_muls, opt_subs)

    # 返回优化后的乘法表达式的映射结果
    return opt_subs
    ## 找到重复的子表达式

    # 要消除的重复子表达式集合
    to_eliminate = set()

    # 已经看到的子表达式集合
    seen_subexp = set()

    # 被排除的符号集合
    excluded_symbols = set()

    def _find_repeated(expr):
        # 如果表达式不是 SymPy 的基本对象或 Unevaluated 对象，则返回
        if not isinstance(expr, (Basic, Unevaluated)):
            return

        # 如果表达式是 RootOf 类型，则返回
        if isinstance(expr, RootOf):
            return

        # 如果表达式是基本对象，并且是原子、阶数或者特定矩阵类型，则处理
        if isinstance(expr, Basic) and (
                expr.is_Atom or
                expr.is_Order or
                isinstance(expr, (MatrixSymbol, MatrixElement))):
            # 如果是符号，则将其名称添加到排除符号集合中
            if expr.is_Symbol:
                excluded_symbols.add(expr.name)
            return

        # 如果表达式可迭代，则处理其参数
        if iterable(expr):
            args = expr
        else:
            # 检查是否已经见过该子表达式
            if expr in seen_subexp:
                # 对于每个要忽略的符号，如果在表达式的自由符号中，则不处理
                for ign in ignore:
                    if ign in expr.free_symbols:
                        break
                else:
                    # 将该表达式添加到要消除的集合中
                    to_eliminate.add(expr)
                    return

            # 将当前表达式添加到已见过的子表达式集合中
            seen_subexp.add(expr)

            # 如果表达式在 opt_subs 中，则用其替换当前表达式
            if expr in opt_subs:
                expr = opt_subs[expr]

            # 获取表达式的参数列表
            args = expr.args

        # 递归处理参数列表中的每个参数
        list(map(_find_repeated, args))

    # 遍历所有输入的表达式列表
    for e in exprs:
        if isinstance(e, Basic):
            # 对于每个基本表达式，执行重复子表达式查找
            _find_repeated(e)

    ## 重建表达式树

    # 从生成器中移除与表达式中已有名称冲突的符号
    symbols = (_ for _ in symbols if _.name not in excluded_symbols)

    # 替换列表，用于存储将被替换的子表达式
    replacements = []

    # 替换字典，用于存储最终的替换映射
    subs = {}
    # 定义一个内部函数 _rebuild，用于重建表达式中的部分元素
    def _rebuild(expr):
        # 如果表达式不是 Basic 或 Unevaluated 类型，则直接返回
        if not isinstance(expr, (Basic, Unevaluated)):
            return expr

        # 如果表达式没有子元素，则直接返回
        if not expr.args:
            return expr

        # 如果表达式是可迭代的，则递归重建其子元素
        if iterable(expr):
            new_args = [_rebuild(arg) for arg in expr.args]
            return expr.func(*new_args)

        # 如果表达式在 subs 字典中，则返回其对应的替换值
        if expr in subs:
            return subs[expr]

        # 保存原始表达式，如果它在 opt_subs 字典中，则尝试替换为 opt_subs 中的值
        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]

        # 如果启用了 order，并且表达式是乘法或者矩阵乘法类型，则按顺序处理其参数，以确保替换顺序与哈希无关
        if order != 'none':
            if isinstance(expr, (Mul, MatMul)):
                c, nc = expr.args_cnc()
                if c == [1]:
                    args = nc
                else:
                    args = list(ordered(c)) + nc
            elif isinstance(expr, (Add, MatAdd)):
                args = list(ordered(expr.args))
            else:
                args = expr.args
        else:
            args = expr.args

        # 对表达式的参数递归调用 _rebuild 函数
        new_args = list(map(_rebuild, args))

        # 如果表达式是 Unevaluated 类型，或者参数有变化，则构造新的表达式
        if isinstance(expr, Unevaluated) or new_args != args:
            new_expr = expr.func(*new_args)
        else:
            new_expr = expr

        # 如果原始表达式需要消除，则生成新的符号进行替换
        if orig_expr in to_eliminate:
            try:
                sym = next(symbols)
            except StopIteration:
                raise ValueError("Symbols iterator ran out of symbols.")

            # 如果原始表达式是矩阵表达式，则创建一个对应形状的新符号
            if isinstance(orig_expr, MatrixExpr):
                sym = MatrixSymbol(sym.name, orig_expr.rows,
                    orig_expr.cols)

            # 将原始表达式与新表达式的替换关系存入 subs 字典，并记录替换的元组
            subs[orig_expr] = sym
            replacements.append((sym, new_expr))
            return sym

        else:
            return new_expr

    # 创建一个空列表 reduced_exprs，用于存储处理后的表达式
    reduced_exprs = []

    # 遍历输入的表达式列表 exprs
    for e in exprs:
        # 如果表达式是 Basic 类型，则调用 _rebuild 函数进行重建
        if isinstance(e, Basic):
            reduced_e = _rebuild(e)
        else:
            reduced_e = e
        # 将处理后的表达式添加到 reduced_exprs 列表中
        reduced_exprs.append(reduced_e)

    # 返回替换列表 replacements 和处理后的表达式列表 reduced_exprs
    return replacements, reduced_exprs
# 对给定表达式进行公共子表达式消除（CSE）操作，减少重复计算，提高效率
def cse(exprs, symbols=None, optimizations=None, postprocess=None,
        order='canonical', ignore=(), list=True):
    """ Perform common subexpression elimination on an expression.

    Parameters
    ==========

    exprs : list of SymPy expressions, or a single SymPy expression
        要减少的表达式列表，或者单个 SymPy 表达式。
    symbols : infinite iterator yielding unique Symbols
        用于标识提取出的公共子表达式的唯一符号的无限迭代器。推荐使用 `numbered_symbols` 生成器。默认情况下是形如 "x0", "x1" 等的符号流。必须是一个无限迭代器。
    optimizations : list of (callable, callable) pairs
        外部优化函数的预处理器和后处理器对。可选地可以传递 'basic' 以使用预定义的基本优化。旧实现中默认使用这些“基本”优化，但在处理大型表达式时可能非常慢。现在，默认不进行预处理或后处理优化。
    postprocess : function
        接受 cse 的两个返回值，并返回所需的 cse 输出形式的函数。例如，如果希望反转替换，则可以使用 lambda 函数 `lambda r, e: return reversed(r), e`。
    order : string, 'none' or 'canonical'
        用于处理 Mul 和 Add 参数的顺序。如果设置为 'canonical'，参数将以规范顺序排序。如果设置为 'none'，排序将更快，但依赖于表达式的哈希值，因此是机器相关和可变的。在处理速度为关键的大型表达式时，建议使用 order='none'。
    ignore : iterable of Symbols
        忽略任何包含在 `ignore` 中的符号的替换。
    list : bool, (default True)
        返回表达式列表，否则返回与输入相同类型的表达式（当为 False 时）。

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        所有被替换的公共子表达式。较早出现的子表达式可能会出现在后面的子表达式中。
    reduced_exprs : list of SymPy expressions
        所有替换后的简化表达式。

    Examples
    ========

    >>> from sympy import cse, SparseMatrix
    >>> from sympy.abc import x, y, z, w
    >>> cse(((w + x + y + z)*(w + y + z))/(w + x)**3)
    ([(x0, y + z), (x1, w + x)], [(w + x0)*(x0 + x1)/x1**3])


    List of expressions with recursive substitutions:

    >>> m = SparseMatrix([x + y, x + y + z])
    >>> cse([(x+y)**2, x + y + z, y + z, x + z + y, m])
    ([(x0, x + y), (x1, x0 + z)], [x0**2, x1, y + z, x1, Matrix([
    [x0],
    [x1]])])

    Note: the type and mutability of input matrices is retained.

    >>> isinstance(_[1][-1], SparseMatrix)
    True

    The user may disallow substitutions containing certain symbols:
    """
    The `cse` function performs Common Subexpression Elimination (CSE) on a list of mathematical expressions.
    
    The function handles various options for optimization and postprocessing of the expressions.
    
    :param exprs: The expressions to perform CSE on, can be a single expression or a list of expressions.
    :param symbols: Symbols to use for the CSE process, defaults to `None` which generates new symbols.
    :param optimizations: Optional list of optimizations to apply during preprocessing.
    :param postprocess: Optional function for postprocessing the CSE results.
    :param order: Optional parameter to specify the order of optimizations.
    :param ignore: Symbols to ignore during CSE.
    :param list: Flag indicating whether to force the result to be a list even if exprs is a single expression.
    
    :return: Depending on the `postprocess` function, returns either a tuple of replacements and reduced expressions,
             or the result of applying `postprocess` to these values.
    """
    
    if not list:
        # If list flag is not set, perform CSE without forcing list output
        return _cse_homogeneous(exprs,
            symbols=symbols, optimizations=optimizations,
            postprocess=postprocess, order=order, ignore=ignore)
    
    if isinstance(exprs, (int, float)):
        # Convert exprs to SymPy expression if it's a numerical value
        exprs = sympify(exprs)
    
    # Ensure exprs is iterable even if it's a single expression
    if isinstance(exprs, (Basic, MatrixBase)):
        exprs = [exprs]
    
    copy = exprs
    temp = []
    for e in exprs:
        if isinstance(e, (Matrix, ImmutableMatrix)):
            # Flatten Matrix or ImmutableMatrix to Tuple of elements
            temp.append(Tuple(*e.flat()))
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            # Convert SparseMatrix or ImmutableSparseMatrix to Tuple of (key, value) pairs
            temp.append(Tuple(*e.todok().items()))
        else:
            temp.append(e)
    exprs = temp
    del temp
    
    if optimizations is None:
        optimizations = []
    elif optimizations == 'basic':
        optimizations = basic_optimizations
    
    # Preprocess expressions to optimize for CSE
    reduced_exprs = [preprocess_for_cse(e, optimizations) for e in exprs]
    
    if symbols is None:
        symbols = numbered_symbols(cls=Symbol)
    else:
        # Ensure symbols is an iterator
        symbols = iter(symbols)
    
    # Perform optimization substitutions
    opt_subs = opt_cse(reduced_exprs, order)
    
    # Apply tree-based CSE algorithm
    replacements, reduced_exprs = tree_cse(reduced_exprs, symbols, opt_subs,
                                           order, ignore)
    
    # Postprocess the results to return expressions to canonical form
    exprs = copy
    replacements = [(sym, postprocess_for_cse(subtree, optimizations))
                    for sym, subtree in replacements]
    reduced_exprs = [postprocess_for_cse(e, optimizations)
                     for e in reduced_exprs]
    
    # Convert reduced expressions back to original type (Matrix or SparseMatrix)
    for i, e in enumerate(exprs):
        if isinstance(e, (Matrix, ImmutableMatrix)):
            reduced_exprs[i] = Matrix(e.rows, e.cols, reduced_exprs[i])
            if isinstance(e, ImmutableMatrix):
                reduced_exprs[i] = reduced_exprs[i].as_immutable()
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            m = SparseMatrix(e.rows, e.cols, {})
            for k, v in reduced_exprs[i]:
                m[k] = v
            if isinstance(e, ImmutableSparseMatrix):
                m = m.as_immutable()
            reduced_exprs[i] = m
    
    if postprocess is None:
        return replacements, reduced_exprs
    
    # Apply user-defined postprocessing function to the results
    return postprocess(replacements, reduced_exprs)
def _cse_homogeneous(exprs, **kwargs):
    """
    Same as ``cse`` but the ``reduced_exprs`` are returned
    with the same type as ``exprs`` or a sympified version of the same.

    Parameters
    ==========

    exprs : an Expr, iterable of Expr or dictionary with Expr values
        the expressions in which repeated subexpressions will be identified
    kwargs : additional arguments for the ``cse`` function

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        All of the common subexpressions that were replaced. Subexpressions
        earlier in this list might show up in subexpressions later in this
        list.
    reduced_exprs : list of SymPy expressions
        The reduced expressions with all of the replacements above.

    Examples
    ========

    >>> from sympy.simplify.cse_main import cse
    >>> from sympy import cos, Tuple, Matrix
    >>> from sympy.abc import x
    >>> output = lambda x: type(cse(x, list=False)[1])
    >>> output(1)
    <class 'sympy.core.numbers.One'>
    >>> output('cos(x)')
    <class 'str'>
    >>> output(cos(x))
    cos
    >>> output(Tuple(1, x))
    <class 'sympy.core.containers.Tuple'>
    >>> output(Matrix([[1,0], [0,1]]))
    <class 'sympy.matrices.dense.MutableDenseMatrix'>
    >>> output([1, x])
    <class 'list'>
    >>> output((1, x))
    <class 'tuple'>
    >>> output({1, x})
    <class 'set'>
    """
    # 如果 exprs 是字符串类型，则将其 sympify 后再进行 homogeneous 处理
    if isinstance(exprs, str):
        replacements, reduced_exprs = _cse_homogeneous(
            sympify(exprs), **kwargs)
        return replacements, repr(reduced_exprs)
    
    # 如果 exprs 是列表、元组或集合类型，则直接调用 cse 处理
    if isinstance(exprs, (list, tuple, set)):
        replacements, reduced_exprs = cse(exprs, **kwargs)
        return replacements, type(exprs)(reduced_exprs)
    
    # 如果 exprs 是字典类型，则按顺序处理字典的键值对
    if isinstance(exprs, dict):
        keys = list(exprs.keys())  # 为了保证元素顺序，获取键列表
        replacements, values = cse([exprs[k] for k in keys], **kwargs)
        reduced_exprs = dict(zip(keys, values))
        return replacements, reduced_exprs

    try:
        replacements, (reduced_exprs,) = cse(exprs, **kwargs)
    except TypeError:  # 捕获可能出现的 TypeError 异常，如 'mpf' 对象
        return [], exprs
    else:
        return replacements, reduced_exprs
```