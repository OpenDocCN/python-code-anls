# `D:\src\scipysrc\sympy\sympy\simplify\combsimp.py`

```
# 从 sympy 库中导入 Mul 类，用于处理乘法表达式
from sympy.core import Mul
# 从 sympy 库中导入 count_ops 函数，用于计算表达式中的操作数
from sympy.core.function import count_ops
# 从 sympy 库中导入 preorder_traversal 和 bottom_up 函数，用于遍历表达式树
from sympy.core.traversal import preorder_traversal, bottom_up
# 从 sympy.functions.combinatorial.factorials 模块导入 binomial 和 factorial 函数
from sympy.functions.combinatorial.factorials import binomial, factorial
# 从 sympy.functions 模块导入 gamma 函数
from sympy.functions import gamma
# 从 sympy.simplify.gammasimp 模块导入 gammasimp 和 _gammasimp 函数
from sympy.simplify.gammasimp import gammasimp, _gammasimp

# 从 sympy.utilities.timeutils 模块导入 timethis 装饰器函数
from sympy.utilities.timeutils import timethis


# 使用 timethis 装饰器来定义 combsimp 函数，并添加 'combsimp' 标签
@timethis('combsimp')
def combsimp(expr):
    r"""
    Simplify combinatorial expressions.

    Explanation
    ===========

    This function takes as input an expression containing factorials,
    binomials, Pochhammer symbol and other "combinatorial" functions,
    and tries to minimize the number of those functions and reduce
    the size of their arguments.

    The algorithm works by rewriting all combinatorial functions as
    gamma functions and applying gammasimp() except simplification
    steps that may make an integer argument non-integer. See docstring
    of gammasimp for more information.

    Then it rewrites expression in terms of factorials and binomials by
    rewriting gammas as factorials and converting (a+b)!/a!b! into
    binomials.

    If expression has gamma functions or combinatorial functions
    with non-integer argument, it is automatically passed to gammasimp.

    Examples
    ========

    >>> from sympy.simplify import combsimp
    >>> from sympy import factorial, binomial, symbols
    >>> n, k = symbols('n k', integer = True)

    >>> combsimp(factorial(n)/factorial(n - 3))
    n*(n - 2)*(n - 1)
    >>> combsimp(binomial(n+1, k+1)/binomial(n, k))
    (n + 1)/(k + 1)

    """

    # 将表达式中的 gamma 函数重写为一般形式，piecewise=False 表示不生成分段函数
    expr = expr.rewrite(gamma, piecewise=False)
    # 检查表达式中是否存在 gamma 函数，并且其参数不是整数
    if any(isinstance(node, gamma) and not node.args[0].is_integer
           for node in preorder_traversal(expr)):
        # 如果存在非整数参数的 gamma 函数，调用 gammasimp 进行简化
        return gammasimp(expr);

    # 将表达式应用 _gammasimp 函数，转换为因子和二项式的形式
    expr = _gammasimp(expr, as_comb=True)
    # 将表达式中的 gamma 函数重写为因子阶乘的形式
    expr = _gamma_as_comb(expr)
    return expr


# 定义 _gamma_as_comb 辅助函数
def _gamma_as_comb(expr):
    """
    Helper function for combsimp.

    Rewrites expression in terms of factorials and binomials
    """

    # 将表达式重写为因子阶乘的形式
    expr = expr.rewrite(factorial)
    return expr
    # 定义函数 f(rv)，处理表达式 rv 中的阶乘项的乘法因子化简
    def f(rv):
        # 如果 rv 不是乘法表达式，则直接返回 rv
        if not rv.is_Mul:
            return rv
        # 将 rv 转换为幂次字典形式
        rvd = rv.as_powers_dict()
        # 初始化分子和分母的阶乘参数列表
        nd_fact_args = [[], []]  # numerator, denominator

        # 遍历 rv 中的每个键
        for k in rvd:
            # 如果键是阶乘对象且对应的幂次是整数
            if isinstance(k, factorial) and rvd[k].is_Integer:
                # 如果幂次为正数，将对应的基数重复添加到分子的参数列表中
                if rvd[k].is_positive:
                    nd_fact_args[0].extend([k.args[0]] * rvd[k])
                # 如果幂次为负数，将对应的基数重复添加到分母的参数列表中
                else:
                    nd_fact_args[1].extend([k.args[0]] * -rvd[k])
                # 将当前阶乘项在 rvd 中的幂次置为 0
                rvd[k] = 0
        
        # 如果分子或分母的阶乘参数列表为空，则返回原始表达式 rv
        if not nd_fact_args[0] or not nd_fact_args[1]:
            return rv

        # 标记是否有成功匹配的情况
        hit = False
        # 遍历分子和分母的阶乘参数列表
        for m in range(2):
            i = 0
            # 遍历当前列表中的每个元素
            while i < len(nd_fact_args[m]):
                ai = nd_fact_args[m][i]
                # 在当前列表中寻找可以与当前元素 ai 相加得到分子和分母中的元素
                for j in range(i + 1, len(nd_fact_args[m])):
                    aj = nd_fact_args[m][j]

                    sum = ai + aj
                    # 如果找到了可以匹配的元素
                    if sum in nd_fact_args[1 - m]:
                        hit = True

                        # 从分子或分母的列表中移除匹配的元素
                        nd_fact_args[1 - m].remove(sum)
                        del nd_fact_args[m][j]
                        del nd_fact_args[m][i]

                        # 根据 ai 和 aj 的复杂性选择较小的元素创建二项式对象
                        rvd[binomial(sum, ai if count_ops(ai) <
                                count_ops(aj) else aj)] += (
                                -1 if m == 0 else 1)
                        break
                else:
                    i += 1

        # 如果存在匹配情况，返回简化后的乘法表达式
        if hit:
            return Mul(*([k**rvd[k] for k in rvd] + [factorial(k)
                    for k in nd_fact_args[0]]))/Mul(*[factorial(k)
                    for k in nd_fact_args[1]])
        # 否则返回原始表达式 rv
        return rv

    # 对表达式 expr 应用自底向上的方式进行变换，使用函数 f 进行处理
    return bottom_up(expr, f)
```