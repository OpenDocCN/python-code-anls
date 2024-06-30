# `D:\src\scipysrc\sympy\sympy\assumptions\satask.py`

```
"""
Module to evaluate the proposition with assumptions using SAT algorithm.
"""

# 导入必要的模块和函数
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.assumptions.ask_generated import get_all_known_matrix_facts, get_all_known_number_facts
from sympy.assumptions.assume import global_assumptions, AppliedPredicate
from sympy.assumptions.sathandlers import class_fact_registry
from sympy.core import oo
from sympy.logic.inference import satisfiable
from sympy.assumptions.cnf import CNF, EncodedCNF
from sympy.matrices.kind import MatrixKind


def satask(proposition, assumptions=True, context=global_assumptions,
        use_known_facts=True, iterations=oo):
    """
    Function to evaluate the proposition with assumptions using SAT algorithm.

    This function extracts every fact relevant to the expressions composing
    proposition and assumptions. For example, if a predicate containing
    ``Abs(x)`` is proposed, then ``Q.zero(Abs(x)) | Q.positive(Abs(x))``
    will be found and passed to SAT solver because ``Q.nonnegative`` is
    registered as a fact for ``Abs``.

    Proposition is evaluated to ``True`` or ``False`` if the truth value can be
    determined. If not, ``None`` is returned.

    Parameters
    ==========

    proposition : Any boolean expression.
        Proposition which will be evaluated to boolean value.

    assumptions : Any boolean expression, optional.
        Local assumptions to evaluate the *proposition*.

    context : AssumptionsContext, optional.
        Default assumptions to evaluate the *proposition*. By default,
        this is ``sympy.assumptions.global_assumptions`` variable.

    use_known_facts : bool, optional.
        If ``True``, facts from ``sympy.assumptions.ask_generated``
        module are passed to SAT solver as well.

    iterations : int, optional.
        Number of times that relevant facts are recursively extracted.
        Default is infinite times until no new fact is found.

    Returns
    =======

    ``True``, ``False``, or ``None``

    Examples
    ========

    >>> from sympy import Abs, Q
    >>> from sympy.assumptions.satask import satask
    >>> from sympy.abc import x
    >>> satask(Q.zero(Abs(x)), Q.zero(x))
    True

    """
    # 将命题和反命题转换为 CNF（合取范式）
    props = CNF.from_prop(proposition)
    _props = CNF.from_prop(~proposition)

    # 将假设转换为 CNF
    assumptions = CNF.from_prop(assumptions)

    # 将上下文转换为 CNF
    context_cnf = CNF()
    if context:
        context_cnf = context_cnf.extend(context)

    # 获取所有相关事实
    sat = get_all_relevant_facts(props, assumptions, context_cnf,
        use_known_facts=use_known_facts, iterations=iterations)

    # 将假设加入事实库
    sat.add_from_cnf(assumptions)
    if context:
        sat.add_from_cnf(context_cnf)

    # 检查命题的可满足性
    return check_satisfiability(props, _props, sat)


def check_satisfiability(prop, _prop, factbase):
    # 复制事实库
    sat_true = factbase.copy()
    sat_false = factbase.copy()

    # 将命题和反命题加入事实库
    sat_true.add_from_cnf(prop)
    sat_false.add_from_cnf(_prop)

    # 判断命题是否可满足
    can_be_true = satisfiable(sat_true)
    # 调用 satisfiable 函数，检查 sat_false 是否可满足
    can_be_false = satisfiable(sat_false)

    # 如果 sat_false 和 sat_true 都可满足，则返回 None
    if can_be_true and can_be_false:
        return None

    # 如果 sat_true 可满足而 sat_false 不可满足，则返回 True
    if can_be_true and not can_be_false:
        return True

    # 如果 sat_true 不可满足而 sat_false 可满足，则返回 False
    if not can_be_true and can_be_false:
        return False

    # 如果 sat_true 和 sat_false 都不可满足，则抛出 ValueError 异常
    # 提示有矛盾的假设，建议运行额外检查以确定不一致的组合
    raise ValueError("Inconsistent assumptions")
# 从给定的命题 *proposition* 中提取所有谓词的参数表达式
def extract_predargs(proposition, assumptions=None, context=None):
    """
    Extract every expression in the argument of predicates from *proposition*,
    *assumptions* and *context*.

    Parameters
    ==========

    proposition : sympy.assumptions.cnf.CNF
        The main proposition from which predicates are extracted.

    assumptions : sympy.assumptions.cnf.CNF, optional.
        Assumptions that provide additional context.

    context : sympy.assumptions.cnf.CNF, optional.
        Contextual information generated from assumptions.

    Examples
    ========

    >>> from sympy import Q, Abs
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.satask import extract_predargs
    >>> from sympy.abc import x, y
    >>> props = CNF.from_prop(Q.zero(Abs(x*y)))
    >>> assump = CNF.from_prop(Q.zero(x) & Q.zero(y))
    >>> extract_predargs(props, assump)
    {x, y, Abs(x*y)}

    """
    # 获取命题中所有符号
    req_keys = find_symbols(proposition)
    # 获取命题中所有谓词
    keys = proposition.all_predicates()
    
    # XXX: We need this since True/False are not Basic
    # 由于 True 和 False 不是 Basic 类型，所以我们需要进行特殊处理
    lkeys = set()
    if assumptions:
        # 将假设中的所有谓词添加到 lkeys 中
        lkeys |= assumptions.all_predicates()
    if context:
        # 将上下文中的所有谓词添加到 lkeys 中
        lkeys |= context.all_predicates()

    # 排除 S.true 和 S.false，因为它们不是需要的谓词
    lkeys = lkeys - {S.true, S.false}
    
    # 用于迭代的临时变量
    tmp_keys = None
    # 当 tmp_keys 不为空集时执行循环
    while tmp_keys != set():
        tmp = set()
        # 遍历 lkeys 中的每个元素
        for l in lkeys:
            # 查找当前谓词 l 中的符号
            syms = find_symbols(l)
            # 如果 syms 和 req_keys 有交集，则将 syms 添加到 tmp 中
            if (syms & req_keys) != set():
                tmp |= syms
        # 更新 tmp_keys
        tmp_keys = tmp - req_keys
        # 将 tmp_keys 合并到 req_keys 中
        req_keys |= tmp_keys
    
    # 将 lkeys 中包含的带有相关符号的谓词添加到 keys 中
    keys |= {l for l in lkeys if find_symbols(l) & req_keys != set()}
    
    # 存储提取到的表达式的集合
    exprs = set()
    # 遍历 keys 中的每个谓词
    for key in keys:
        # 如果当前谓词是 AppliedPredicate 类型
        if isinstance(key, AppliedPredicate):
            # 将谓词的参数添加到 exprs 中
            exprs |= set(key.arguments)
        else:
            # 否则直接将当前谓词添加到 exprs 中
            exprs.add(key)
    
    # 返回提取到的所有表达式的集合
    return exprs


# 在给定的 pred 中查找所有 Symbol 对象
def find_symbols(pred):
    """
    Find every :obj:`~.Symbol` in *pred*.

    Parameters
    ==========

    pred : sympy.assumptions.cnf.CNF, or any Expr.
        The expression or CNF object to search for symbols.

    """
    if isinstance(pred, CNF):
        symbols = set()
        # 遍历 pred 中的所有谓词，并将找到的符号添加到 symbols 中
        for a in pred.all_predicates():
            symbols |= find_symbols(a)
        return symbols
    # 如果 pred 不是 CNF 对象，则直接返回其中的 Symbol 对象
    return pred.atoms(Symbol)


# 从表达式集合中提取相关的事实
def get_relevant_clsfacts(exprs, relevant_facts=None):
    """
    Extract relevant facts from the items in *exprs*. Facts are defined in
    ``assumptions.sathandlers`` module.

    This function is recursively called by ``get_all_relevant_facts()``.

    Parameters
    ==========

    exprs : set
        Expressions whose relevant facts are searched.

    relevant_facts : sympy.assumptions.cnf.CNF, optional.
        Pre-discovered relevant facts.

    Returns
    =======

    exprs : set
        Candidates for next relevant fact searching.

    relevant_facts : sympy.assumptions.cnf.CNF
        Updated relevant facts.

    Examples
    ========

    Here, we will see how facts relevant to ``Abs(x*y)`` are recursively
    extracted. On the first run, set containing the expression is passed
    without pre-discovered relevant facts. The result is a set containing
    candidates for next run, and ``CNF()`` instance containing facts
    which are relevant to ``Abs`` and its argument.

    """
    # 暂未提供详细实现，此处不需要添加更多的注释
    pass
    # 导入 Sympy 中的 Abs 函数
    >>> from sympy import Abs
    # 导入 Sympy 中获取相关分类事实的函数
    >>> from sympy.assumptions.satask import get_relevant_clsfacts
    # 导入 Sympy 中的符号 x 和 y
    >>> from sympy.abc import x, y
    # 创建一个包含 Abs(x*y) 的表达式集合
    >>> exprs = {Abs(x*y)}
    # 调用 get_relevant_clsfacts 函数获取表达式集合和相关事实
    >>> exprs, facts = get_relevant_clsfacts(exprs)
    # 打印表达式集合
    >>> exprs
    {x*y}
    # 打印 facts 中的 clauses，跳过 doctest 检查
    >>> facts.clauses #doctest: +SKIP
    {frozenset({Literal(Q.odd(Abs(x*y)), False), Literal(Q.odd(x*y), True)}),
    frozenset({Literal(Q.zero(Abs(x*y)), False), Literal(Q.zero(x*y), True)}),
    frozenset({Literal(Q.even(Abs(x*y)), False), Literal(Q.even(x*y), True)}),
    frozenset({Literal(Q.zero(Abs(x*y)), True), Literal(Q.zero(x*y), False)}),
    frozenset({Literal(Q.even(Abs(x*y)), False),
                Literal(Q.odd(Abs(x*y)), False),
                Literal(Q.odd(x*y), True)}),
    frozenset({Literal(Q.even(Abs(x*y)), False),
                Literal(Q.even(x*y), True),
                Literal(Q.odd(Abs(x*y)), False)}),
    frozenset({Literal(Q.positive(Abs(x*y)), False),
                Literal(Q.zero(Abs(x*y)), False)})}

    # 将第一次运行的结果传递给第二次运行，并获取更新后的表达式集合和事实
    We pass the first run's results to the second run, and get the expressions
    for next run and updated facts.

    # 再次调用 get_relevant_clsfacts 函数，传入更新后的 facts 参数，获取表达式集合
    >>> exprs, facts = get_relevant_clsfacts(exprs, relevant_facts=facts)
    # 打印更新后的表达式集合
    >>> exprs
    {x, y}

    # 在最后一次运行中，不再返回任何候选，因此我们知道成功获取了所有相关事实。
    On final run, no more candidate is returned thus we know that all
    relevant facts are successfully retrieved.

    # 再次调用 get_relevant_clsfacts 函数，传入更新后的 facts 参数，获取表达式集合
    >>> exprs, facts = get_relevant_clsfacts(exprs, relevant_facts=facts)
    # 打印更新后的表达式集合，此时应为空集
    >>> exprs
    set()

    """
    # 如果 relevant_facts 为空，则初始化为 CNF 对象
    if not relevant_facts:
        relevant_facts = CNF()

    # 创建一个新的表达式集合
    newexprs = set()
    # 遍历每个表达式中的每个表达式
    for expr in exprs:
        # 调用 class_fact_registry 函数获取该表达式的分类事实
        for fact in class_fact_registry(expr):
            # 将事实转换为 CNF 格式
            newfact = CNF.to_CNF(fact)
            # 更新 relevant_facts，将新的事实与之前的相关事实取交集
            relevant_facts = relevant_facts._and(newfact)
            # 遍历新事实中的所有谓词
            for key in newfact.all_predicates():
                # 如果谓词是 AppliedPredicate 类型，则将其参数加入 newexprs 集合
                if isinstance(key, AppliedPredicate):
                    newexprs |= set(key.arguments)

    # 返回更新后的表达式集合减去原始表达式集合的结果，以及更新后的 relevant_facts
    return newexprs - exprs, relevant_facts
def get_all_relevant_facts(proposition, assumptions, context,
        use_known_facts=True, iterations=oo):
    """
    Extract all relevant facts from *proposition* and *assumptions*.

    This function extracts the facts by recursively calling
    ``get_relevant_clsfacts()``. Extracted facts are converted to
    ``EncodedCNF`` and returned.

    Parameters
    ==========

    proposition : sympy.assumptions.cnf.CNF
        CNF generated from proposition expression.

    assumptions : sympy.assumptions.cnf.CNF
        CNF generated from assumption expression.

    context : sympy.assumptions.cnf.CNF
        CNF generated from assumptions context.

    use_known_facts : bool, optional.
        If ``True``, facts from ``sympy.assumptions.ask_generated``
        module are encoded as well.

    iterations : int, optional.
        Number of times that relevant facts are recursively extracted.
        Default is infinite times until no new fact is found.

    Returns
    =======

    sympy.assumptions.cnf.EncodedCNF
        Encoded CNF containing all relevant facts.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.satask import get_all_relevant_facts
    >>> from sympy.abc import x, y
    >>> props = CNF.from_prop(Q.nonzero(x*y))
    >>> assump = CNF.from_prop(Q.nonzero(x))
    >>> context = CNF.from_prop(Q.nonzero(y))
    >>> get_all_relevant_facts(props, assump, context) #doctest: +SKIP
    <sympy.assumptions.cnf.EncodedCNF at 0x7f09faa6ccd0>

    """
    # 初始化计数器
    i = 0
    # 创建一个空的 CNF 对象来存储相关的事实
    relevant_facts = CNF()
    # 创建一个空集合来存储所有的表达式
    all_exprs = set()
    # 循环直到达到迭代次数或不再发现新的事实
    while True:
        # 在第一次迭代时提取谓词参数
        if i == 0:
            exprs = extract_predargs(proposition, assumptions, context)
        # 将当前迭代中提取的表达式添加到所有表达式的集合中
        all_exprs |= exprs
        # 提取相关的分类事实并更新相关事实集合
        exprs, relevant_facts = get_relevant_clsfacts(exprs, relevant_facts)
        # 增加迭代次数计数器
        i += 1
        # 如果达到指定的迭代次数，则终止循环
        if i >= iterations:
            break
        # 如果没有更多新的表达式提取，则终止循环
        if not exprs:
            break
    # 如果使用已知事实
    if use_known_facts:
        # 创建一个空的 CNF 对象来存储已知事实
        known_facts_CNF = CNF()

        # 如果表达式列表中有任何数值矩阵类型的表达式，添加所有已知的矩阵事实
        if any(expr.kind == MatrixKind(NumberKind) for expr in all_exprs):
            known_facts_CNF.add_clauses(get_all_known_matrix_facts())

        # 检查表达式列表中是否有 NumberKind 或 UndefinedKind 类型的表达式，添加所有已知的数值事实
        if any(((expr.kind == NumberKind) or (expr.kind == UndefinedKind)) for expr in all_exprs):
            known_facts_CNF.add_clauses(get_all_known_number_facts())

        # 创建一个编码后的 CNF 对象，并从已知事实 CNF 转换而来
        kf_encoded = EncodedCNF()
        kf_encoded.from_cnf(known_facts_CNF)

        # 定义一个函数来转换文字（literal），根据 delta 进行偏移
        def translate_literal(lit, delta):
            if lit > 0:
                return lit + delta
            else:
                return lit - delta

        # 定义一个函数来转换数据，将每个子句中的文字按 delta 进行偏移
        def translate_data(data, delta):
            return [{translate_literal(i, delta) for i in clause} for clause in data]

        # 初始化数据和符号列表
        data = []
        symbols = []
        n_lit = len(kf_encoded.symbols)

        # 对所有表达式进行迭代，为每个表达式创建符号，并将编码后的数据进行偏移
        for i, expr in enumerate(all_exprs):
            symbols += [pred(expr) for pred in kf_encoded.symbols]
            data += translate_data(kf_encoded.data, i * n_lit)

        # 创建符号到编码值的映射字典
        encoding = dict(list(zip(symbols, range(1, len(symbols)+1))))

        # 创建一个编码后的 CNF 上下文对象，使用已偏移的数据和符号编码
        ctx = EncodedCNF(data, encoding)
    else:
        # 如果不使用已知事实，则创建一个空的编码后的 CNF 上下文对象
        ctx = EncodedCNF()

    # 向上下文对象添加来自 relevant_facts 的 CNF 数据
    ctx.add_from_cnf(relevant_facts)

    # 返回创建的上下文对象
    return ctx
```