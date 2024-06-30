# `D:\src\scipysrc\sympy\sympy\simplify\sqrtdenest.py`

```
from sympy.core import Add, Expr, Mul, S, sympify
# 导入必要的类和函数，用于符号计算
from sympy.core.function import _mexpand, count_ops, expand_mul
# 导入符号计算函数，用于展开和计算操作数
from sympy.core.sorting import default_sort_key
# 导入排序函数，用于默认排序键
from sympy.core.symbol import Dummy
# 导入符号类，用于创建临时符号
from sympy.functions import root, sign, sqrt
# 导入函数，包括平方根函数等
from sympy.polys import Poly, PolynomialError
# 导入多项式相关类和异常

def is_sqrt(expr):
    """Return True if expr is a sqrt, otherwise False."""
    # 检查表达式是否为平方根表达式
    return expr.is_Pow and expr.exp.is_Rational and abs(expr.exp) is S.Half

def sqrt_depth(p) -> int:
    """Return the maximum depth of any square root argument of p."""
    # 返回表达式 p 中任意平方根参数的最大深度
    if p is S.ImaginaryUnit:
        return 1
    if p.is_Atom:
        return 0
    if p.is_Add or p.is_Mul:
        return max(sqrt_depth(x) for x in p.args)
    if is_sqrt(p):
        return sqrt_depth(p.base) + 1
    return 0

def is_algebraic(p):
    """Return True if p is comprised of only Rationals or square roots
    of Rationals and algebraic operations."""
    # 判断表达式 p 是否仅由有理数或有理数平方根以及代数运算构成
    if p.is_Rational:
        return True
    elif p.is_Atom:
        return False
    elif is_sqrt(p) or p.is_Pow and p.exp.is_Integer:
        return is_algebraic(p.base)
    elif p.is_Add or p.is_Mul:
        return all(is_algebraic(x) for x in p.args)
    else:
        return False

def _subsets(n):
    """
    Returns all possible subsets of the set (0, 1, ..., n-1) except the
    empty set, listed in reversed lexicographical order according to binary
    representation, so that the case of the fourth root is treated last.
    """
    # 返回集合 (0, 1, ..., n-1) 的所有可能非空子集，按照二进制表示的逆字典序排列
    if n == 1:
        a = [[1]]
    elif n == 2:
        a = [[1, 0], [0, 1], [1, 1]]
    elif n == 3:
        a = [[1, 0, 0], [0, 1, 0], [1, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    else:
        b = _subsets(n - 1)
        a0 = [x + [0] for x in b]
        a1 = [x + [1] for x in b]
        a = a0 + [[0]*(n - 1) + [1]] + a1
    return a

def sqrtdenest(expr, max_iter=3):
    """Denests sqrts in an expression that contain other square roots
    if possible, otherwise returns the expr unchanged. This is based on the
    algorithms of [1].
    """
    # 对包含其他平方根的表达式进行可能的平方根解套，否则返回原始表达式
    # 基于参考文献[1]的算法
    # 对给定表达式进行展开和乘法展开
    expr = expand_mul(expr)
    
    # 迭代求解并且对平方根进行去嵌套操作
    for i in range(max_iter):
        # 调用_sqrtdenest0函数对表达式进行平方根去嵌套操作
        z = _sqrtdenest0(expr)
        # 如果表达式没有变化，则停止迭代并返回结果
        if expr == z:
            return expr
        # 更新表达式为去嵌套后的结果，继续迭代
        expr = z
    
    # 返回最终的去嵌套后的表达式结果
    return expr
# 定义函数 _sqrt_match，用于处理形如 a + b*sqrt(r) 的表达式 p，并返回 [a, b, r] 的列表，其中 sqrt(r) 的深度最大。
def _sqrt_match(p):
    """Return [a, b, r] for p.match(a + b*sqrt(r)) where, in addition to
    matching, sqrt(r) also has then maximal sqrt_depth among addends of p.

    Examples
    ========

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrt_match
    >>> _sqrt_match(1 + sqrt(2) + sqrt(2)*sqrt(3) +  2*sqrt(1+sqrt(5)))
    [1 + sqrt(2) + sqrt(6), 2, 1 + sqrt(5)]
    """
    # 导入 split_surds 函数，用于处理根式的分割
    from sympy.simplify.radsimp import split_surds

    # 将表达式 p 扩展成多项式
    p = _mexpand(p)
    # 如果 p 是一个数值，则返回 (p, 0, 0)
    if p.is_Number:
        res = (p, S.Zero, S.Zero)
    # 如果 p 是一个加法表达式
    elif p.is_Add:
        # 对加法表达式的参数进行排序
        pargs = sorted(p.args, key=default_sort_key)
        # 创建平方项列表
        sqargs = [x**2 for x in pargs]
        # 如果所有平方项都是有理数且为正数
        if all(sq.is_Rational and sq.is_positive for sq in sqargs):
            # 调用 split_surds 函数分割根式并返回结果
            r, b, a = split_surds(p)
            res = a, b, r
            return list(res)
        # 使过程变得规范化，将参数包含在元组中以确保选择具有给定深度的最大参数
        v = [(sqrt_depth(x), x, i) for i, x in enumerate(pargs)]
        # 选择具有最大深度的参数
        nmax = max(v, key=default_sort_key)
        if nmax[0] == 0:
            res = []
        else:
            # 选择 r
            depth, _, i = nmax
            r = pargs.pop(i)
            v.pop(i)
            b = S.One
            # 如果 r 是乘法表达式
            if r.is_Mul:
                bv = []
                rv = []
                for x in r.args:
                    # 将深度小于当前深度的参数添加到 bv 中
                    if sqrt_depth(x) < depth:
                        bv.append(x)
                    else:
                        rv.append(x)
                b = Mul._from_args(bv)
                r = Mul._from_args(rv)
            # 收集包含 r 的项
            a1 = []
            b1 = [b]
            for x in v:
                if x[0] < depth:
                    a1.append(x[1])
                else:
                    x1 = x[1]
                    if x1 == r:
                        b1.append(1)
                    else:
                        if x1.is_Mul:
                            x1args = list(x1.args)
                            if r in x1args:
                                x1args.remove(r)
                                b1.append(Mul(*x1args))
                            else:
                                a1.append(x[1])
                        else:
                            a1.append(x[1])
            a = Add(*a1)
            b = Add(*b1)
            res = (a, b, r**2)
    else:
        # 将 p 分解为乘法系数和平方根
        b, r = p.as_coeff_Mul()
        # 如果 r 是平方根，则返回 (0, b, r**2)
        if is_sqrt(r):
            res = (S.Zero, b, r**2)
        else:
            res = []
    return list(res)
    # 检查表达式是否是平方根表达式
    if is_sqrt(expr):
        # 将表达式转换为分子和分母
        n, d = expr.as_numer_denom()
        # 如果分母为1，表示分子是一个平方根
        if d is S.One:
            # 如果分子是一个加法表达式
            if n.base.is_Add:
                # 对加法表达式的参数进行排序
                args = sorted(n.base.args, key=default_sort_key)
                # 如果参数大于2且所有参数的平方均为整数
                if len(args) > 2 and all((x**2).is_Integer for x in args):
                    try:
                        # 尝试对分子进行平方根去嵌套处理
                        return _sqrtdenest_rec(n)
                    except SqrtdenestStopIteration:
                        pass
                # 对加法表达式的各项进行扩展并求平方根
                expr = sqrt(_mexpand(Add(*[_sqrtdenest0(x) for x in args])))
            # 对处理后的表达式继续进行平方根去嵌套处理
            return _sqrtdenest1(expr)
        else:
            # 对分子和分母分别进行平方根去嵌套处理
            n, d = [_sqrtdenest0(i) for i in (n, d)]
            return n/d

    # 如果表达式是加法表达式
    if isinstance(expr, Add):
        # 分别保存系数和参数
        cs = []
        args = []
        for arg in expr.args:
            c, a = arg.as_coeff_Mul()
            cs.append(c)
            args.append(a)

        # 如果所有系数是有理数且所有参数是平方根表达式
        if all(c.is_Rational for c in cs) and all(is_sqrt(arg) for arg in args):
            # 对有理数系数和参数是平方根的项进行合并处理
            return _sqrt_ratcomb(cs, args)

    # 如果表达式是一般的表达式对象
    if isinstance(expr, Expr):
        # 对表达式的参数递归进行平方根去嵌套处理
        args = expr.args
        if args:
            return expr.func(*[_sqrtdenest0(a) for a in args])
    # 返回未经处理的原始表达式
    return expr
def _sqrtdenest1(expr, denester=True):
    """Return denested expr after denesting with simpler methods or, that
    failing, using the denester."""

    # 导入 radsimp 函数，用于简化表达式
    from sympy.simplify.simplify import radsimp

    # 如果表达式不是平方根表达式，直接返回原表达式
    if not is_sqrt(expr):
        return expr

    # 提取平方根的底数
    a = expr.base

    # 如果底数是原子（Atom），则直接返回原表达式
    if a.is_Atom:
        return expr

    # 尝试匹配底数，得到匹配结果（a, b, r）
    val = _sqrt_match(a)

    # 如果没有匹配结果，返回原表达式
    if not val:
        return expr

    # 解构匹配结果
    a, b, r = val

    # 尝试快速数值化简
    d2 = _mexpand(a**2 - b**2*r)
    # 检查 d2 是否为有理数
    if d2.is_Rational:
        # 如果 d2 是正数
        if d2.is_positive:
            # 尝试用 _sqrt_numeric_denest 处理，返回处理结果
            z = _sqrt_numeric_denest(a, b, r, d2)
            # 如果结果不为 None，则直接返回结果
            if z is not None:
                return z
        else:
            # 处理负数情况，即开四次方根的情况
            # sqrtdenest(sqrt(3 + 2*sqrt(3))) =
            # sqrt(2)*3**(1/4)/2 + sqrt(2)*3**(3/4)/2
            # 计算 -d2*r，并展开
            dr2 = _mexpand(-d2*r)
            # 计算 dr 为其平方根
            dr = sqrt(dr2)
            # 如果 dr 是有理数
            if dr.is_Rational:
                # 尝试用 _sqrt_numeric_denest 处理，返回处理结果
                z = _sqrt_numeric_denest(_mexpand(b*r), a, r, dr2)
                # 如果结果不为 None，则返回处理结果并开四次方根
                if z is not None:
                    return z/root(r, 4)

    else:
        # 如果 d2 不是有理数，尝试用 _sqrt_symbolic_denest 处理，返回处理结果
        z = _sqrt_symbolic_denest(a, b, r)
        # 如果结果不为 None，则返回处理结果
        if z is not None:
            return z

    # 如果没有 denester 或者表达式不是代数式，则直接返回表达式
    if not denester or not is_algebraic(expr):
        return expr

    # 尝试用 sqrt_biquadratic_denest 处理表达式，返回处理结果
    res = sqrt_biquadratic_denest(expr, a, b, r, d2)
    # 如果结果存在，则返回处理结果
    if res:
        return res

    # 调用 _denester 处理表达式，处理结果存入 z
    av0 = [a, b, r, d2]
    z = _denester([radsimp(expr**2)], av0, 0, sqrt_depth(expr))[0]
    # 如果 av0[1] 是 None，则返回原始表达式
    if av0[1] is None:
        return expr
    # 如果 z 不为 None
    if z is not None:
        # 如果 z 和 expr 的开方深度相同，并且 z 的操作数比 expr 多，则返回 expr
        if sqrt_depth(z) == sqrt_depth(expr) and count_ops(z) > count_ops(expr):
            return expr
        # 否则返回 z
        return z
    # 如果 z 是 None，则返回原始表达式
    return expr
def _sqrt_symbolic_denest(a, b, r):
    """Given an expression, sqrt(a + b*sqrt(b)), return the denested
    expression or None.

    Explanation
    ===========

    If r = ra + rb*sqrt(rr), try replacing sqrt(rr) in ``a`` with
    (y**2 - ra)/rb, and if the result is a quadratic, ca*y**2 + cb*y + cc, and
    (cb + b)**2 - 4*ca*cc is 0, then sqrt(a + b*sqrt(r)) can be rewritten as
    sqrt(ca*(sqrt(r) + (cb + b)/(2*ca))**2).

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import _sqrt_symbolic_denest, sqrtdenest
    >>> from sympy import sqrt, Symbol
    >>> from sympy.abc import x

    >>> a, b, r = 16 - 2*sqrt(29), 2, -10*sqrt(29) + 55
    >>> _sqrt_symbolic_denest(a, b, r)
    sqrt(11 - 2*sqrt(29)) + sqrt(5)

    If the expression is numeric, it will be simplified:

    >>> w = sqrt(sqrt(sqrt(3) + 1) + 1) + 1 + sqrt(2)
    >>> sqrtdenest(sqrt((w**2).expand()))
    1 + sqrt(2) + sqrt(1 + sqrt(1 + sqrt(3)))

    Otherwise, it will only be simplified if assumptions allow:

    >>> w = w.subs(sqrt(3), sqrt(x + 3))
    >>> sqrtdenest(sqrt((w**2).expand()))
    sqrt((sqrt(sqrt(sqrt(x + 3) + 1) + 1) + 1 + sqrt(2))**2)

    Notice that the argument of the sqrt is a square. If x is made positive
    then the sqrt of the square is resolved:

    >>> _.subs(x, Symbol('x', positive=True))
    sqrt(sqrt(sqrt(x + 3) + 1) + 1) + 1 + sqrt(2)
    """

    # 将输入的参数转换为 SymPy 的表达式
    a, b, r = map(sympify, (a, b, r))
    
    # 对 r 进行匹配，获取其形式为 ra + rb*sqrt(rr) 的部分
    rval = _sqrt_match(r)
    if not rval:
        return None
    ra, rb, rr = rval
    
    # 如果 rb 非零，尝试用 (y**2 - ra)/rb 替换 a 中的 sqrt(rr)
    if rb:
        y = Dummy('y', positive=True)
        try:
            # 将 a 中的 sqrt(rr) 替换为 (y**2 - ra)/rb，得到新的表达式 newa
            newa = Poly(a.subs(sqrt(rr), (y**2 - ra)/rb), y)
        except PolynomialError:
            return None
        # 如果 newa 是二次多项式
        if newa.degree() == 2:
            ca, cb, cc = newa.all_coeffs()
            cb += b
            # 检查条件 (cb + b)**2 - 4*ca*cc 是否为 0
            if _mexpand(cb**2 - 4*ca*cc).equals(0):
                # 重写 sqrt(a + b*sqrt(r)) 为 sqrt(ca*(sqrt(r) + (cb + b)/(2*ca))**2)
                z = sqrt(ca*(sqrt(r) + cb/(2*ca))**2)
                if z.is_number:
                    z = _mexpand(Mul._from_args(z.as_content_primitive()))
                return z


def _sqrt_numeric_denest(a, b, r, d2):
    r"""Helper that denest
    $\sqrt{a + b \sqrt{r}}, d^2 = a^2 - b^2 r > 0$

    If it cannot be denested, it returns ``None``.
    """
    d = sqrt(d2)
    s = a + d
    # sqrt_depth(res) <= sqrt_depth(s) + 1
    # sqrt_depth(expr) = sqrt_depth(r) + 2
    # there is denesting if sqrt_depth(s) + 1 < sqrt_depth(r) + 2
    # if s**2 is Number there is a fourth root
    if sqrt_depth(s) < sqrt_depth(r) + 1 or (s**2).is_Rational:
        s1, s2 = sign(s), sign(b)
        if s1 == s2 == -1:
            s1 = s2 = 1
        # 计算 denest 结果
        res = (s1 * sqrt(a + d) + s2 * sqrt(a - d)) * sqrt(2) / 2
        return res.expand()


def sqrt_biquadratic_denest(expr, a, b, r, d2):
    """denest expr = sqrt(a + b*sqrt(r))
    where a, b, r are linear combinations of square roots of
    positive rationals on the rationals (SQRR) and r > 0, b != 0,
    d2 = a**2 - b**2*r > 0

    If it cannot denest it returns None.

    Explanation
    ===========
    """
    Search for a solution A of type SQRR of the biquadratic equation
    4*A**4 - 4*a*A**2 + b**2*r = 0                               (1)
    sqd = sqrt(a**2 - b**2*r)
    Choosing the sqrt to be positive, the possible solutions are
    A = sqrt(a/2 +/- sqd/2)
    Since a, b, r are SQRR, then a**2 - b**2*r is a SQRR,
    so if sqd can be denested, it is done by
    _sqrtdenest_rec, and the result is a SQRR.
    Similarly for A.
    Examples of solutions (in both cases a and sqd are positive):

      Example of expr with solution sqrt(a/2 + sqd/2) but not
      solution sqrt(a/2 - sqd/2):
      expr = sqrt(-sqrt(15) - sqrt(2)*sqrt(-sqrt(5) + 5) - sqrt(3) + 8)
      a = -sqrt(15) - sqrt(3) + 8; sqd = -2*sqrt(5) - 2 + 4*sqrt(3)

      Example of expr with solution sqrt(a/2 - sqd/2) but not
      solution sqrt(a/2 + sqd/2):
      w = 2 + r2 + r3 + (1 + r3)*sqrt(2 + r2 + 5*r3)
      expr = sqrt((w**2).expand())
      a = 4*sqrt(6) + 8*sqrt(2) + 47 + 28*sqrt(3)
      sqd = 29 + 20*sqrt(3)

    Define B = b/2*A; eq.(1) implies a = A**2 + B**2*r; then
    expr**2 = a + b*sqrt(r) = (A + B*sqrt(r))**2

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrt_match, sqrt_biquadratic_denest
    >>> z = sqrt((2*sqrt(2) + 4)*sqrt(2 + sqrt(2)) + 5*sqrt(2) + 8)
    >>> a, b, r = _sqrt_match(z**2)
    >>> d2 = a**2 - b**2*r
    >>> sqrt_biquadratic_denest(z, a, b, r, d2)
    sqrt(2) + sqrt(sqrt(2) + 2) + 2
    ```
def _denester(nested, av0, h, max_depth_level):
    """Denests a list of expressions that contain nested square roots.

    Explanation
    ===========

    Algorithm based on <http://www.almaden.ibm.com/cs/people/fagin/symb85.pdf>.

    It is assumed that all of the elements of 'nested' share the same
    bottom-level radicand. (This is stated in the paper, on page 177, in
    the paragraph immediately preceding the algorithm.)

    When evaluating all of the arguments in parallel, the bottom-level
    radicand only needs to be denested once. This means that calling
    _denester with x arguments results in a recursive invocation with x+1
    arguments; hence _denester has polynomial complexity.

    However, if the arguments were evaluated separately, each call would
    result in two recursive invocations, and the algorithm would have
    exponential complexity.

    This is discussed in the paper in the middle paragraph of page 179.
    """
    # 导入符号计算库中的根简化函数
    from sympy.simplify.simplify import radsimp
    # 如果递归深度超过最大深度限制，则返回空结果
    if h > max_depth_level:
        return None, None
    # 如果 av0 的第二个元素为 None，则返回空结果
    if av0[1] is None:
        return None, None
    # 如果 av0 的第一个元素为 None，并且 nested 中所有元素都是数字，则表示没有嵌套的参数
    if (av0[0] is None and
            all(n.is_Number for n in nested)):  # no arguments are nested
        # 遍历 nested 的所有子集 'f'
        for f in _subsets(len(nested)):  # test subset 'f' of nested
            # 构建子集 'f' 对应的乘积，并展开
            p = _mexpand(Mul(*[nested[i] for i in range(len(f)) if f[i]]))
            # 如果 f 中 1 的个数大于 1 且最后一个元素为真，则取反 p
            if f.count(1) > 1 and f[-1]:
                p = -p
            # 计算 p 的平方根
            sqp = sqrt(p)
            # 如果 sqp 是有理数，则返回其平方根以及对应的子集 'f'
            if sqp.is_Rational:
                return sqp, f  # got a perfect square so return its square root.
        # 如果没有找到完全平方数，则返回前一次调用的根被放入的项
        return sqrt(nested[-1]), [0]*len(nested)
    else:
        # 初始化变量 R 为 None
        R = None
        # 如果 av0 的第一个元素不为 None
        if av0[0] is not None:
            # 将 av0 的前两个元素组成一个列表作为 values
            values = [av0[:2]]
            # 将 av0 的第三个元素赋值给 R
            R = av0[2]
            # 将 av0 的第四个和 R 组成一个列表作为 nested2
            nested2 = [av0[3], R]
            # 将 av0 的第一个元素设为 None
            av0[0] = None
        else:
            # 通过筛选不为 None 的表达式生成 values 列表
            values = list(filter(None, [_sqrt_match(expr) for expr in nested]))
            # 遍历 values 中的每个元素
            for v in values:
                # 如果 v 的第三个元素存在
                if v[2]:  # Since if b=0, r is not defined
                    # 如果 R 已经定义，检查 R 是否与 v 的第三个元素相等
                    if R is not None:
                        # 如果不相等，将 av0 的第二个元素设为 None 并返回 None, None
                        if R != v[2]:
                            av0[1] = None
                            return None, None
                    else:
                        # 将 R 设为 v 的第三个元素
                        R = v[2]
            # 如果 R 仍然是 None
            if R is None:
                # 返回前一次调用的根数
                return sqrt(nested[-1]), [0]*len(nested)
            # 将 values 中每个元素的第一项平方后减去 R 乘以第二项的平方，再加上 R，生成 nested2 列表
            nested2 = [_mexpand(v[0]**2) - _mexpand(R*v[1]**2) for v in values] + [R]
        # 调用 _denester 函数处理 nested2，av0，h+1，max_depth_level 参数
        d, f = _denester(nested2, av0, h + 1, max_depth_level)
        # 如果 f 为 False，返回 None, None
        if not f:
            return None, None
        # 如果 nested 中所有 f[i] 均为 False
        if not any(f[i] for i in range(len(nested))):
            # 取 values 列表的最后一个元素
            v = values[-1]
            # 返回 v 的第一项加上 v 的第二项乘以 d 的平方根，以及 f
            return sqrt(v[0] + _mexpand(v[1]*d)), f
        else:
            # 创建 p，该变量为 nested 中所有 f[i] 为 True 的元素的乘积
            p = Mul(*[nested[i] for i in range(len(nested)) if f[i]])
            # 调用 _sqrt_match 函数处理 p，将结果赋值给 v
            v = _sqrt_match(p)
            # 如果 f 中包含 1，并且 f 中 1 的索引小于 nested 的长度减 1，且 f 的最后一个元素为 True
            if 1 in f and f.index(1) < len(nested) - 1 and f[len(nested) - 1]:
                # 修改 v 的第一项和第二项为它们的相反数
                v[0] = -v[0]
                v[1] = -v[1]
            # 如果 f 的最后一个元素为 False，表示解中存在平方根
            if not f[len(nested)]:  # Solution denests with square roots
                # 计算 v 的第一项加上 d 的扩展结果
                vad = _mexpand(v[0] + d)
                # 如果 vad 小于等于 0，返回前一次调用的根数和全 0 的列表
                if vad <= 0:
                    return sqrt(nested[-1]), [0]*len(nested)
                # 如果 vad 的平方根深度小于等于 R 的平方根深度加 1，或者 vad 的平方是一个数字
                if not(sqrt_depth(vad) <= sqrt_depth(R) + 1 or
                       (vad**2).is_Number):
                    # 将 av0 的第二个元素设为 None 并返回 None, None
                    av0[1] = None
                    return None, None

                # 计算 sqrt(vad) 的平方根简化结果
                sqvad = _sqrtdenest1(sqrt(vad), denester=False)
                # 如果 sqrt(sqvad) 的平方根深度小于等于 R 的平方根深度加 1
                if not (sqrt_depth(sqvad) <= sqrt_depth(R) + 1):
                    # 将 av0 的第二个元素设为 None 并返回 None, None
                    av0[1] = None
                    return None, None
                # 计算 sqvad 的倒数的根式简化结果
                sqvad1 = radsimp(1/sqvad)
                # 计算 _mexpand(sqvad/sqrt(2) + v[1]*sqrt(R)*sqvad1/sqrt(2)) 的结果，并返回该值和 f
                res = _mexpand(sqvad/sqrt(2) + (v[1]*sqrt(R)*sqvad1/sqrt(2)))
                return res, f

                      #          sign(v[1])*sqrt(_mexpand(v[1]**2*R*vad1/2))), f
            else:  # Solution requires a fourth root
                # 计算 v[1]*R 的扩展结果加上 d
                s2 = _mexpand(v[1]*R) + d
                # 如果 s2 小于等于 0，返回前一次调用的根数和全 0 的列表
                if s2 <= 0:
                    return sqrt(nested[-1]), [0]*len(nested)
                # 计算 R 的第四个根和 sqrt(s2) 的值
                FR, s = root(_mexpand(R), 4), sqrt(s2)
                # 计算 s/(sqrt(2)*FR) + v[0]*FR/(sqrt(2)*s) 的结果，并返回该值和 f
                return _mexpand(s/(sqrt(2)*FR) + v[0]*FR/(sqrt(2)*s)), f
# 定义函数 `_sqrt_ratcomb`，用于将有理组合的根式解套
"""Denest rational combinations of radicals.

Based on section 5 of [1].

Examples
========

>>> from sympy import sqrt
>>> from sympy.simplify.sqrtdenest import sqrtdenest
>>> z = sqrt(1+sqrt(3)) + sqrt(3+3*sqrt(3)) - sqrt(10+6*sqrt(3))
>>> sqrtdenest(z)
0
"""
def _sqrt_ratcomb(cs, args):
    # 导入 `radsimp` 函数用于对根式进行简化
    from sympy.simplify.radsimp import radsimp

    # 检查是否存在可以解套的根式对
    def find(a):
        n = len(a)
        for i in range(n - 1):
            for j in range(i + 1, n):
                # 获取第 i 和第 j 个根式的基数
                s1 = a[i].base
                s2 = a[j].base
                # 计算它们的乘积
                p = _mexpand(s1 * s2)
                # 对乘积的平方根进行解套
                s = sqrtdenest(sqrt(p))
                # 如果解套后的结果不等于原来的平方根，则返回解套结果及其索引
                if s != sqrt(p):
                    return s, i, j

    # 调用 find 函数查找可解套的根式对，并获取返回的结果
    indices = find(args)
    # 如果没有找到可解套的根式对，则返回根据系数 cs 和根式 args 计算的加法表达式
    if indices is None:
        return Add(*[c * arg for c, arg in zip(cs, args)])

    # 解包找到的解套根式对的结果及其索引
    s, i1, i2 = indices

    # 弹出系数 cs 中第 i2 个元素，并弹出根式 args 中第 i2 个元素
    c2 = cs.pop(i2)
    args.pop(i2)
    # 获取根式 args 中第 i1 个元素
    a1 = args[i1]

    # 将第 i1 个根式替换为 s/a1.base 的有理化简结果
    cs[i1] += radsimp(c2 * s / a1.base)

    # 递归调用 _sqrt_ratcomb 函数，继续解套剩余的根式组合
    return _sqrt_ratcomb(cs, args)
```