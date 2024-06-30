# `D:\src\scipysrc\sympy\sympy\assumptions\lra_satask.py`

```
# 从 sympy.assumptions.assume 模块导入全局假设
from sympy.assumptions.assume import global_assumptions
# 从 sympy.assumptions.cnf 模块导入 CNF 和 EncodedCNF 类
from sympy.assumptions.cnf import CNF, EncodedCNF
# 从 sympy.assumptions.ask 模块导入 Q 对象
from sympy.assumptions.ask import Q
# 从 sympy.logic.inference 模块导入 satisfiable 函数
from sympy.logic.inference import satisfiable
# 从 sympy.logic.algorithms.lra_theory 模块导入 UnhandledInput 和 ALLOWED_PRED 常量
from sympy.logic.algorithms.lra_theory import UnhandledInput, ALLOWED_PRED
# 从 sympy.matrices.kind 模块导入 MatrixKind 类
from sympy.matrices.kind import MatrixKind
# 从 sympy.core.kind 模块导入 NumberKind 类
from sympy.core.kind import NumberKind
# 从 sympy.assumptions.assume 模块导入 AppliedPredicate 类
from sympy.assumptions.assume import AppliedPredicate
# 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.singleton 模块导入 S 对象
from sympy.core.singleton import S

# 定义函数 lra_satask，用于使用 SAT 算法和线性实数算术理论求解假设下的命题
def lra_satask(proposition, assumptions=True, context=global_assumptions):
    """
    Function to evaluate the proposition with assumptions using SAT algorithm
    in conjunction with an Linear Real Arithmetic theory solver.

    Used to handle inequalities. Should eventually be depreciated and combined
    into satask, but infinity handling and other things need to be implemented
    before that can happen.
    """
    # 将命题 proposition 转换为 CNF 格式
    props = CNF.from_prop(proposition)
    _props = CNF.from_prop(~proposition)

    # 将假设 assumptions 转换为 CNF 格式
    cnf = CNF.from_prop(assumptions)
    assumptions = EncodedCNF()
    assumptions.from_cnf(cnf)

    # 将 context 转换为 CNF 格式
    context_cnf = CNF()
    if context:
        context_cnf = context_cnf.extend(context)

    assumptions.add_from_cnf(context_cnf)

    # 调用 check_satisfiability 函数检查命题的可满足性
    return check_satisfiability(props, _props, assumptions)

# WHITE_LIST 是一个包含可以处理的谓词集合，包括 ALLOWED_PRED 中的内容和一些 Q 模块定义的谓词
WHITE_LIST = ALLOWED_PRED | {Q.positive, Q.negative, Q.zero, Q.nonzero, Q.nonpositive, Q.nonnegative,
                                            Q.extended_positive, Q.extended_negative, Q.extended_nonpositive,
                                            Q.extended_negative, Q.extended_nonzero, Q.negative_infinite,
                                            Q.positive_infinite}

# 定义函数 check_satisfiability，用于检查命题的可满足性
def check_satisfiability(prop, _prop, factbase):
    sat_true = factbase.copy()
    sat_false = factbase.copy()
    sat_true.add_from_cnf(prop)
    sat_false.add_from_cnf(_prop)

    # 从 sat_true 中获取所有的谓词和表达式
    all_pred, all_exprs = get_all_pred_and_expr_from_enc_cnf(sat_true)

    # 遍历所有的谓词，如果谓词不在 WHITE_LIST 中且不是 Q.ne，则抛出 UnhandledInput 异常
    for pred in all_pred:
        if pred.function not in WHITE_LIST and pred.function != Q.ne:
            raise UnhandledInput(f"LRASolver: {pred} is an unhandled predicate")
    
    # 遍历所有的表达式，如果表达式的类型是 MatrixKind，则抛出 UnhandledInput 异常
    for expr in all_exprs:
        if expr.kind == MatrixKind(NumberKind):
            raise UnhandledInput(f"LRASolver: {expr} is of MatrixKind")
        if expr == S.NaN:
            raise UnhandledInput("LRASolver: nan")

    # 转换旧的假设为谓词并添加到 sat_true 和 sat_false 中，同时检查未处理的谓词
    # 对于从所有表达式中提取的前提，逐个处理
    for assm in extract_pred_from_old_assum(all_exprs):
        # 获取当前已有编码的长度
        n = len(sat_true.encoding)
        # 如果前提不在真集合的编码中，将其添加进去，并分配一个新的编码
        if assm not in sat_true.encoding:
            sat_true.encoding[assm] = n + 1
        # 将真集合的编码添加到数据中
        sat_true.data.append([sat_true.encoding[assm]])

        # 获取当前已有编码的长度
        n = len(sat_false.encoding)
        # 如果前提不在假集合的编码中，将其添加进去，并分配一个新的编码
        if assm not in sat_false.encoding:
            sat_false.encoding[assm] = n + 1
        # 将假集合的编码添加到数据中
        sat_false.data.append([sat_false.encoding[assm]])

    # 对真集合进行预处理
    sat_true = _preprocess(sat_true)
    # 对假集合进行预处理
    sat_false = _preprocess(sat_false)

    # 判断真集合是否可满足，使用线性实数算术理论（LRA）
    can_be_true = satisfiable(sat_true, use_lra_theory=True) is not False
    # 判断假集合是否可满足，使用线性实数算术理论（LRA）
    can_be_false = satisfiable(sat_false, use_lra_theory=True) is not False

    # 如果既可以为真也可以为假，返回空
    if can_be_true and can_be_false:
        return None

    # 如果可以为真且不能为假，返回True
    if can_be_true and not can_be_false:
        return True

    # 如果不可以为真且可以为假，返回False
    if not can_be_true and can_be_false:
        return False

    # 如果既不能为真也不能为假，抛出数值错误
    if not can_be_true and not can_be_false:
        raise ValueError("Inconsistent assumptions")
# 对传入的编码 CNF 进行预处理，返回只包含 Q.eq, Q.gt, Q.lt, Q.ge 和 Q.le 谓词的编码 CNF。

def _preprocess(enc_cnf):
    """
    Returns an encoded cnf with only Q.eq, Q.gt, Q.lt,
    Q.ge, and Q.le predicate.

    Converts every unequality into a disjunction of strict
    inequalities. For example, x != 3 would become
    x < 3 OR x > 3.

    Also converts all negated Q.ne predicates into
    equalities.
    """

    # 复制编码 CNF，确保不会改变原始数据
    enc_cnf = enc_cnf.copy()
    # 当前编码计数器初始化为1
    cur_enc = 1
    # 反向编码字典，用于根据值查找键
    rev_encoding = {value: key for key, value in enc_cnf.encoding.items()}

    # 新的编码字典和处理后的数据列表
    new_encoding = {}
    new_data = []

    # 遍历每个子句
    for clause in enc_cnf.data:
        new_clause = []
        # 遍历每个文字
        for lit in clause:
            if lit == 0:
                # 如果文字为0，则添加到新子句并更新新编码字典
                new_clause.append(lit)
                new_encoding[lit] = False
                continue

            # 获取文字对应的谓词
            prop = rev_encoding[abs(lit)]
            # 判断文字是否为负数
            negated = lit < 0
            # 根据文字的正负性确定符号（正数为1，负数为-1）
            sign = (lit > 0) - (lit < 0)

            # 将谓词转换为二元关系
            prop = _pred_to_binrel(prop)

            # 如果谓词不是 AppliedPredicate 类型
            if not isinstance(prop, AppliedPredicate):
                # 如果谓词不在新编码字典中，则添加，并更新编码计数器
                if prop not in new_encoding:
                    new_encoding[prop] = cur_enc
                    cur_enc += 1
                lit = new_encoding[prop]
                new_clause.append(sign * lit)
                continue

            # 处理 Q.ne 谓词的否定情况
            if negated and prop.function == Q.eq:
                negated = False
                prop = Q.ne(*prop.arguments)

            # 如果谓词是 Q.ne
            if prop.function == Q.ne:
                arg1, arg2 = prop.arguments
                if negated:
                    # 将 Q.ne 的否定转换为 Q.eq，并添加到新编码字典中
                    new_prop = Q.eq(arg1, arg2)
                    if new_prop not in new_encoding:
                        new_encoding[new_prop] = cur_enc
                        cur_enc += 1

                    new_enc = new_encoding[new_prop]
                    new_clause.append(new_enc)
                    continue
                else:
                    # 将 Q.ne 谓词转换为严格不等的两个谓词，并添加到新编码字典中
                    new_props = (Q.gt(arg1, arg2), Q.lt(arg1, arg2))
                    for new_prop in new_props:
                        if new_prop not in new_encoding:
                            new_encoding[new_prop] = cur_enc
                            cur_enc += 1

                        new_enc = new_encoding[new_prop]
                        new_clause.append(new_enc)
                    continue

            # 处理 Q.eq 谓词的否定情况
            if prop.function == Q.eq and negated:
                assert False

            # 如果谓词不在新编码字典中，则添加，并更新编码计数器
            if prop not in new_encoding:
                new_encoding[prop] = cur_enc
                cur_enc += 1
            new_clause.append(new_encoding[prop] * sign)

        # 将处理后的子句添加到新数据列表中
        new_data.append(new_clause)

    # 断言新编码字典的大小至少为当前编码计数器减一
    assert len(new_encoding) >= cur_enc - 1

    # 创建并返回新的 EncodedCNF 对象
    enc_cnf = EncodedCNF(new_data, new_encoding)
    return enc_cnf


def _pred_to_binrel(pred):
    # 如果谓词不是 AppliedPredicate 类型，则直接返回
    if not isinstance(pred, AppliedPredicate):
        return pred

    # 如果谓词在 pred_to_pos_neg_zero 字典中，则根据对应的函数转换为对应的二元关系
    if pred.function in pred_to_pos_neg_zero:
        f = pred_to_pos_neg_zero[pred.function]
        if f is False:
            return False
        pred = f(pred.arguments[0])

    # 返回转换后的谓词
    return pred
    # 根据预测函数的不同情况，转换预测表达式为相应的数学比较表达式
    if pred.function == Q.positive:
        # 如果预测函数为正数判断，将预测表达式转换为大于比较
        pred = Q.gt(pred.arguments[0], 0)
    elif pred.function == Q.negative:
        # 如果预测函数为负数判断，将预测表达式转换为小于比较
        pred = Q.lt(pred.arguments[0], 0)
    elif pred.function == Q.zero:
        # 如果预测函数为零判断，将预测表达式转换为等于比较
        pred = Q.eq(pred.arguments[0], 0)
    elif pred.function == Q.nonpositive:
        # 如果预测函数为非正数判断，将预测表达式转换为小于等于比较
        pred = Q.le(pred.arguments[0], 0)
    elif pred.function == Q.nonnegative:
        # 如果预测函数为非负数判断，将预测表达式转换为大于等于比较
        pred = Q.ge(pred.arguments[0], 0)
    elif pred.function == Q.nonzero:
        # 如果预测函数为非零判断，将预测表达式转换为不等于比较
        pred = Q.ne(pred.arguments[0], 0)
    
    # 返回转换后的预测表达式
    return pred
# 定义一个映射，将扩展的谓词映射到标准的谓词
pred_to_pos_neg_zero = {
    Q.extended_positive: Q.positive,  # 扩展正数谓词映射到正数谓词
    Q.extended_negative: Q.negative,  # 扩展负数谓词映射到负数谓词
    Q.extended_nonpositive: Q.nonpositive,  # 扩展非正数谓词映射到非正数谓词
    Q.extended_nonzero: Q.nonzero,  # 扩展非零谓词映射到非零谓词
    Q.negative_infinite: False,  # 负无穷谓词映射到False
    Q.positive_infinite: False  # 正无穷谓词映射到False
}

# 从编码 CNF 表达式中获取所有的谓词和表达式
def get_all_pred_and_expr_from_enc_cnf(enc_cnf):
    all_exprs = set()
    all_pred = set()
    for pred in enc_cnf.encoding.keys():
        if isinstance(pred, AppliedPredicate):
            all_pred.add(pred)  # 将应用谓词添加到谓词集合中
            all_exprs.update(pred.arguments)  # 更新表达式集合，包含谓词的所有参数

    return all_pred, all_exprs

# 从旧假设中提取谓词
def extract_pred_from_old_assum(all_exprs):
    """
    Returns a list of relevant new assumption predicate
    based on any old assumptions.

    Raises an UnhandledInput exception if any of the assumptions are
    unhandled.

    Ignored predicate:
    - commutative
    - complex
    - algebraic
    - transcendental
    - extended_real
    - real
    - all matrix predicate
    - rational
    - irrational

    Example
    =======
    >>> from sympy.assumptions.lra_satask import extract_pred_from_old_assum
    >>> from sympy import symbols
    >>> x, y = symbols("x y", positive=True)
    >>> extract_pred_from_old_assum([x, y, 2])
    [Q.positive(x), Q.positive(y)]
    """
    ret = []
    for expr in all_exprs:
        if not hasattr(expr, "free_symbols"):
            continue
        if len(expr.free_symbols) == 0:
            continue

        if expr.is_real is not True:
            raise UnhandledInput(f"LRASolver: {expr} must be real")
        # test for I times imaginary variable; such expressions are considered real
        if isinstance(expr, Mul) and any(arg.is_real is not True for arg in expr.args):
            raise UnhandledInput(f"LRASolver: {expr} must be real")

        if expr.is_integer == True and expr.is_zero != True:
            raise UnhandledInput(f"LRASolver: {expr} is an integer")
        if expr.is_integer == False:
            raise UnhandledInput(f"LRASolver: {expr} can't be an integer")
        if expr.is_rational == False:
            raise UnhandledInput(f"LRASolver: {expr} is irational")

        if expr.is_zero:
            ret.append(Q.zero(expr))  # 如果表达式为零，添加零谓词
        elif expr.is_positive:
            ret.append(Q.positive(expr))  # 如果表达式为正数，添加正数谓词
        elif expr.is_negative:
            ret.append(Q.negative(expr))  # 如果表达式为负数，添加负数谓词
        elif expr.is_nonzero:
            ret.append(Q.nonzero(expr))  # 如果表达式为非零，添加非零谓词
        elif expr.is_nonpositive:
            ret.append(Q.nonpositive(expr))  # 如果表达式为非正数，添加非正数谓词
        elif expr.is_nonnegative:
            ret.append(Q.nonnegative(expr))  # 如果表达式为非负数，添加非负数谓词

    return ret
```