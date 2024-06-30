# `D:\src\scipysrc\sympy\sympy\polys\numberfields\galoisgroups.py`

```
"""
Compute Galois groups of polynomials.

We use algorithms from [1], with some modifications to use lookup tables for
resolvents.

References
==========

.. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.

"""

from collections import defaultdict  # 导入 defaultdict 类用于创建默认字典
import random  # 导入 random 模块提供随机数功能

from sympy.core.symbol import Dummy, symbols  # 导入 Dummy 符号和 symbols 函数
from sympy.ntheory.primetest import is_square  # 导入 is_square 函数用于检查平方数
from sympy.polys.domains import ZZ  # 导入整数环 ZZ
from sympy.polys.densebasic import dup_random  # 导入 dup_random 函数用于生成随机多项式
from sympy.polys.densetools import dup_eval  # 导入 dup_eval 函数用于多项式求值
from sympy.polys.euclidtools import dup_discriminant  # 导入 dup_discriminant 函数用于计算判别式
from sympy.polys.factortools import dup_factor_list, dup_irreducible_p  # 导入多项式因式分解工具函数
from sympy.polys.numberfields.galois_resolvents import (
    GaloisGroupException, get_resolvent_by_lookup, define_resolvents,
    Resolvent,
)  # 导入 Galois 群相关的异常类和解决方案函数
from sympy.polys.numberfields.utilities import coeff_search  # 导入 coeff_search 函数
from sympy.polys.polytools import (Poly, poly_from_expr,
                                   PolificationFailed, ComputationFailed)  # 导入多项式相关工具函数
from sympy.polys.sqfreetools import dup_sqf_p  # 导入 dup_sqf_p 函数用于最小平方自由分解
from sympy.utilities import public  # 导入 public 装饰器

class MaxTriesException(GaloisGroupException):
    ...


def tschirnhausen_transformation(T, max_coeff=10, max_tries=30, history=None,
                                 fixed_order=True):
    r"""
    Given a univariate, monic, irreducible polynomial over the integers, find
    another such polynomial defining the same number field.

    Explanation
    ===========

    See Alg 6.3.4 of [1].

    Parameters
    ==========

    T : Poly
        The given polynomial
    max_coeff : int
        When choosing a transformation as part of the process,
        keep the coeffs between plus and minus this.
    max_tries : int
        Consider at most this many transformations.
    history : set, None, optional (default=None)
        Pass a set of ``Poly.rep``'s in order to prevent any of these
        polynomials from being returned as the polynomial ``U`` i.e. the
        transformation of the given polynomial *T*. The given poly *T* will
        automatically be added to this set, before we try to find a new one.
    fixed_order : bool, default True
        If ``True``, work through candidate transformations A(x) in a fixed
        order, from small coeffs to large, resulting in deterministic behavior.
        If ``False``, the A(x) are chosen randomly, while still working our way
        up from small coefficients to larger ones.

    Returns
    =======

    Pair ``(A, U)``

        ``A`` and ``U`` are ``Poly``, ``A`` is the
        transformation, and ``U`` is the transformed polynomial that defines
        the same number field as *T*. The polynomial ``A`` maps the roots of
        *T* to the roots of ``U``.

    Raises
    ======

    MaxTriesException
        if could not find a polynomial before exceeding *max_tries*.

    """
    X = Dummy('X')  # 创建一个虚拟符号 X
    n = T.degree()  # 获取多项式 T 的次数
    if history is None:
        history = set()  # 如果 history 为 None，则创建一个空集合
    history.add(T.rep)  # 将多项式 T 的表示式添加到历史记录集合中
    if fixed_order:
        # 如果指定了固定的顺序，初始化系数生成器为空字典，设置初始的次数和系数和
        coeff_generators = {}
        deg_coeff_sum = 3
        current_degree = 2

    def get_coeff_generator(degree):
        # 返回给定次数的系数生成器，如果不存在则创建一个新的并保存在字典中
        gen = coeff_generators.get(degree, coeff_search(degree, 1))
        coeff_generators[degree] = gen
        return gen

    for i in range(max_tries):

        # We never use linear A(x), since applying a fixed linear transformation
        # to all roots will only multiply the discriminant of T by a square
        # integer. This will change nothing important. In particular, if disc(T)
        # was zero before, it will still be zero now, and typically we apply
        # the transformation in hopes of replacing T by a squarefree poly.
        
        # 我们从不使用线性的 A(x)，因为对所有的根施加固定的线性变换只会将 T 的判别式乘以一个平方整数。
        # 这不会改变任何重要的东西。特别地，如果 disc(T) 在之前是零，那么现在仍然是零，通常我们应用这种变换，
        # 是希望用一个无平方因子的多项式替换 T。

        if fixed_order:
            # 如果指定了固定顺序，根据当前的次数和最大系数和移动通过恒定和的线。
            # 首先是 d + c = 3，对应 (d, c) = (2, 1)。
            # 然后是 d + c = 4，对应 (d, c) = (3, 1), (2, 2)。
            # 然后是 d + c = 5，对应 (d, c) = (4, 1), (3, 2), (2, 3)，以此类推。
            # 对于给定的 (d, c)，我们在移动到下一个之前穿过所有最大为 c 的系数集合。
            gen = get_coeff_generator(current_degree)
            coeffs = next(gen)
            m = max(abs(c) for c in coeffs)
            if current_degree + m > deg_coeff_sum:
                if current_degree == 2:
                    deg_coeff_sum += 1
                    current_degree = deg_coeff_sum - 1
                else:
                    current_degree -= 1
                gen = get_coeff_generator(current_degree)
                coeffs = next(gen)
            a = [ZZ(1)] + [ZZ(c) for c in coeffs]

        else:
            # 我们使用逐步增加的系数界限，直到指定的最大值，因为使用较小的系数更有可能成功。
            # 每个系数界限有五次尝试的机会，然后增加。
            C = min(i//5 + 1, max_coeff)
            d = random.randint(2, n - 1)
            a = dup_random(d, -C, C, ZZ)

        A = Poly(a, T.gen)
        U = Poly(T.resultant(X - A), X)
        if U.rep not in history and dup_sqf_p(U.rep.to_list(), ZZ):
            return A, U
    raise MaxTriesException
def _galois_group_degree_3(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 3.

    Explanation
    ===========

    Uses Prop 6.3.5 of [1].

    """
    from sympy.combinatorics.galois import S3TransitiveSubgroups
    # 如果输入 T 是一个多项式（Poly）对象，则计算其判别式；否则，使用 dup_discriminant 函数计算
    d = T.discriminant() if isinstance(T, Poly) else dup_discriminant(T, ZZ)
    # 检查判别式是否为完全平方数
    return ((S3TransitiveSubgroups.A3, True) if has_square_disc(T)
            else (S3TransitiveSubgroups.S3, False))


def _galois_group_degree_4_root_approx(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 4.

    Explanation
    ===========

    Follows Alg 6.3.7 of [1], using a pure root approximation approach.

    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.galois import S4TransitiveSubgroups

    X = symbols('X0 X1 X2 X3')
    # 我们首先考虑形式 F = X0*X2 + X1*X3 的解决方案
    # 并且群 G = S4。在这种情况下，稳定子群 H 是 D4 = < (0123), (02) >，
    # 并且 G/H 的一组代表是 {I, (01), (03)}
    F1 = X[0]*X[2] + X[1]*X[3]
    s1 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 3)
    ]
    # 计算解决方案 R1，使用给定的变量 X 和代表集合 s1
    R1 = Resolvent(F1, X, s1)

    # 在算法的后半部分（如果我们达到那里），我们使用另一个形式和一组余类代表。
    # 但是，我们可能需要首先对它们进行排列，所以现在不能形成它们的解决方案。
    F2_pre = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[0]**2
    s2_pre = [
        Permutation(3),
        Permutation(3)(0, 2)
    ]
    
    # 创建一个空集合，用于记录历史记录
    history = set()
    # 进行最多 max_tries 次尝试
    for i in range(max_tries):
        if i > 0:
            # 如果正在重试，则需要一个新的多项式 T。
            _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                                history=history,
                                                fixed_order=not randomize)

        # 计算 R1 在多项式 T 上的评估结果及其整数根信息
        R_dup, _, i0 = R1.eval_for_poly(T, find_integer_root=True)
        # 如果 R 不是平方自由的，则必须重试。
        if not dup_sqf_p(R_dup, ZZ):
            continue

        # 根据 [1] 中的命题 6.3.1，若 disc(T) 是平方数，则 Gal(T) 包含在 A4 中。
        sq_disc = has_square_disc(T)

        if i0 is None:
            # 根据 [1] 中的定理 6.3.3，Gal(T) 不与我们选择的 D4 的任何子群共轭。
            # 这意味着 Gal(T) 要么是 A4，要么是 S4。
            return ((S4TransitiveSubgroups.A4, True) if sq_disc
                    else (S4TransitiveSubgroups.S4, False))

        # Gal(T) 与 H = D4 的某个子群共轭，因此它要么是 V，要么是 C4 或 D4 本身。

        if sq_disc:
            # C4 和 D4 都不包含在 A4 中，因此 Gal(T) 必须是 V。
            return (S4TransitiveSubgroups.V, True)

        # Gal(T) 只能是 D4 或 C4。
        # 现在我们将使用第二个解析式，其中 G 是 Gal(T) 包含的 D4 的共轭。
        # 为了确定正确的共轭，我们需要找到对应于我们找到的整数根的置换。
        sigma = s1[i0]
        # 应用 sigma 意味着对 F 的参数进行置换，并共轭余元的集合代表。
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma * tau * sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        R_dup, _, _ = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)
        # 如果 d 为零（R 有重复根），则必须重试。
        if d == 0:
            continue
        if is_square(d):
            return (S4TransitiveSubgroups.C4, False)
        else:
            return (S4TransitiveSubgroups.D4, False)

    # 如果达到最大尝试次数仍未成功，则抛出最大尝试异常
    raise MaxTriesException
# 计算一个四次多项式的 Galois 群
def _galois_group_degree_4_lookup(T, max_tries=30, randomize=False):
    """
    根据 Alg 6.3.6 [1] 算法计算四次多项式的 Galois 群，但使用了结果系数查找。

    Parameters
    ----------
    T : Polynomial
        输入的四次多项式
    max_tries : int, optional
        最大尝试次数，默认为30
    randomize : bool, optional
        是否随机化，默认为False

    Returns
    -------
    tuple
        返回 Galois 群和一个布尔值，指示是否具有平方判别式
    """
    from sympy.combinatorics.galois import S4TransitiveSubgroups

    # 用于记录历史操作，防止重复
    history = set()
    
    # 最多尝试 max_tries 次
    for i in range(max_tries):
        # 通过查找得到一个结果多项式 R_dup
        R_dup = get_resolvent_by_lookup(T, 0)
        # 检查 R_dup 是否为平方自由的
        if dup_sqf_p(R_dup, ZZ):
            break
        # 如果不是平方自由的，进行 Tschirnhausen 变换
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        # 如果达到最大尝试次数仍未成功，抛出 MaxTriesException 异常
        raise MaxTriesException

    # 计算 R_dup 的不可约因子列表 fl
    fl = dup_factor_list(R_dup, ZZ)
    # 提取不可约因子的次数信息，组成一个有序列表 L
    L = sorted(sum([
        [len(r) - 1] * e for r, e in fl[1]
    ], []))

    # 根据 L 的值返回相应的 Galois 群和布尔值
    if L == [6]:
        return ((S4TransitiveSubgroups.A4, True) if has_square_disc(T)
            else (S4TransitiveSubgroups.S4, False))

    if L == [1, 1, 4]:
        return (S4TransitiveSubgroups.C4, False)

    if L == [2, 2, 2]:
        return (S4TransitiveSubgroups.V, True)

    # 断言 L 为 [2, 4]
    assert L == [2, 4]
    return (S4TransitiveSubgroups.D4, False)


# 计算一个五次多项式的 Galois 群，使用混合方法
def _galois_group_degree_5_hybrid(T, max_tries=30, randomize=False):
    """
    根据 Alg 6.3.9 [1] 算法计算五次多项式的 Galois 群，使用混合方法，结合了结果系数查找和根的近似。

    Parameters
    ----------
    T : Polynomial
        输入的五次多项式
    max_tries : int, optional
        最大尝试次数，默认为30
    randomize : bool, optional
        是否随机化，默认为False
    """
    from sympy.combinatorics.galois import S5TransitiveSubgroups
    from sympy.combinatorics.permutations import Permutation

    # 定义符号变量 X0, X1, X2, X3, X4
    X5 = symbols("X0,X1,X2,X3,X4")
    # 定义解决方案的列表
    res = define_resolvents()
    # 提取第一个解决方案 F51，其余部分暂不使用
    F51, _, s51 = res[(5, 1)]
    # 将 F51 转换为表达式，使用 X5 作为变量
    F51 = F51.as_expr(*X5)
    # 创建 F51 的解决方案 R51
    R51 = Resolvent(F51, X5, s51)

    # 用于记录历史操作，防止重复
    history = set()
    # 标志，指示是否已经达到第二阶段
    reached_second_stage = False
    # 尝试最大次数次数循环
    for i in range(max_tries):
        # 如果不是第一次尝试，则应用特定的 Tschirnhausen 变换
        if i > 0:
            _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                                history=history,
                                                fixed_order=not randomize)
        # 获取第一个 resolvent R51_dup
        R51_dup = get_resolvent_by_lookup(T, 1)
        # 如果 R51_dup 不是平方自由的，则继续下一次循环
        if not dup_sqf_p(R51_dup, ZZ):
            continue

        # 第一阶段
        # 如果尚未进入第二阶段，则需要测试群是否为 S5、A5 或 M20
        if not reached_second_stage:
            # 检查是否存在平方判别式
            sq_disc = has_square_disc(T)

            if dup_irreducible_p(R51_dup, ZZ):
                # 如果 R51_dup 是不可约的，根据平方判别式返回 S5 或 A5
                return ((S5TransitiveSubgroups.A5, True) if sq_disc
                        else (S5TransitiveSubgroups.S5, False))

            if not sq_disc:
                # 如果没有平方判别式，则返回 M20
                return (S5TransitiveSubgroups.M20, False)

        # 第二阶段
        reached_second_stage = True
        # R51 必须有 T 的整数根
        # 为了选择第二个 resolvent，需要知道 F51 的哪个共轭是根
        rounded_roots = R51.round_roots_to_integers_for_poly(T)
        # 这些是整数，是 R51 的根的候选者，找到第一个真正是根的
        for permutation_index, candidate_root in rounded_roots.items():
            if not dup_eval(R51_dup, candidate_root, ZZ):
                break

        # 设置 X 为 X5
        X = X5
        # 计算 F2_pre
        F2_pre = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[4]**2 + X[4]*X[0]**2
        # 设置 s2_pre 为特定的置换
        s2_pre = [
            Permutation(4),
            Permutation(4)(0, 1)(2, 4)
        ]

        # 设置 i0 为 permutation_index
        i0 = permutation_index
        # 取出 s51 中的置换 sigma
        sigma = s51[i0]
        # 计算 F2，并使用 sigma 将 X 映射到对应的置换下
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        # 计算 s2，这里是 sigma * tau * sigma 的形式
        s2 = [sigma*tau*sigma for tau in s2_pre]
        # 计算 Resolvent R2
        R2 = Resolvent(F2, X, s2)
        # 计算 R_dup，并获取其对 T 的求值
        R_dup, _, _ = R2.eval_for_poly(T)
        # 计算 R_dup 的判别式 d
        d = dup_discriminant(R_dup, ZZ)

        # 如果 d 等于 0，则继续下一次循环
        if d == 0:
            continue
        # 如果 d 是完全平方数，则返回 C5
        if is_square(d):
            return (S5TransitiveSubgroups.C5, True)
        else:
            # 否则返回 D5
            return (S5TransitiveSubgroups.D5, True)

    # 如果达到最大尝试次数仍未找到结果，则抛出 MaxTriesException
    raise MaxTriesException
def _galois_group_degree_5_lookup_ext_factor(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 5.

    Explanation
    ===========

    Based on Alg 6.3.9 of [1], but uses resolvent coeff lookup, plus
    factorization over an algebraic extension.

    """
    from sympy.combinatorics.galois import S5TransitiveSubgroups

    # 将输入的多项式 T 存储在 _T 中，以备后用
    _T = T

    # 初始化一个空集合，用于记录历史尝试过的多项式
    history = set()
    # 尝试最多 max_tries 次，进行以下操作
    for i in range(max_tries):
        # 获取通过查找得到的共轭分解多项式 R_dup
        R_dup = get_resolvent_by_lookup(T, 1)
        # 检查 R_dup 是否是平方自由的
        if dup_sqf_p(R_dup, ZZ):
            # 如果是平方自由的，跳出循环
            break
        # 否则，进行特希尔豪森变换，更新 T，并更新历史记录
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        # 如果达到最大尝试次数仍未找到符合条件的 R_dup，则抛出异常
        raise MaxTriesException

    # 检查 T 是否有平方判别式
    sq_disc = has_square_disc(T)

    # 如果 R_dup 是不可约的
    if dup_irreducible_p(R_dup, ZZ):
        # 如果 T 有平方判别式，返回 A5 组及 True
        return ((S5TransitiveSubgroups.A5, True) if sq_disc
                # 否则返回 S5 组及 False
                else (S5TransitiveSubgroups.S5, False))

    # 如果 T 没有平方判别式，返回 M20 组及 False
    if not sq_disc:
        return (S5TransitiveSubgroups.M20, False)

    # 如果程序执行到这里，Gal(T) 只能是 D5 或者 C5
    # 但是为了 Gal(T) 有阶数 5，T 必须在通过添加其一个根后的扩域中已经完全分解
    fl = Poly(_T, domain=ZZ.alg_field_from_poly(_T)).factor_list()[1]
    # 如果多项式的因子个数为 5，则返回 C5 组及 True
    if len(fl) == 5:
        return (S5TransitiveSubgroups.C5, True)
    # 否则返回 D5 组及 True
    else:
        return (S5TransitiveSubgroups.D5, True)


def _galois_group_degree_6_lookup(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 6.

    Explanation
    ===========

    Based on Alg 6.3.10 of [1], but uses resolvent coeff lookup.

    """
    from sympy.combinatorics.galois import S6TransitiveSubgroups

    # 第一个共轭分解多项式的初始化
    history = set()
    # 尝试最多 max_tries 次，进行以下操作
    for i in range(max_tries):
        # 获取通过查找得到的共轭分解多项式 R_dup
        R_dup = get_resolvent_by_lookup(T, 1)
        # 检查 R_dup 是否是平方自由的
        if dup_sqf_p(R_dup, ZZ):
            # 如果是平方自由的，跳出循环
            break
        # 否则，进行特希尔豪森变换，更新 T，并更新历史记录
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        # 如果达到最大尝试次数仍未找到符合条件的 R_dup，则抛出异常
        raise MaxTriesException

    # 对 R_dup 进行因子分解
    fl = dup_factor_list(R_dup, ZZ)

    # 将因子按照次数分组
    factors_by_deg = defaultdict(list)
    for r, _ in fl[1]:
        factors_by_deg[len(r) - 1].append(r)

    # 对分组后的因子按次数排序
    L = sorted(sum([
        [d] * len(ff) for d, ff in factors_by_deg.items()
    ], []))

    # 检查多项式 T 是否有平方判别式
    T_has_sq_disc = has_square_disc(T)

    # 如果 L 是 [1, 2, 3]，表明因子分组为 [1, 2, 3]
    if L == [1, 2, 3]:
        # 获取第一个次数为 3 的因子 f1
        f1 = factors_by_deg[3][0]
        # 如果 f1 有平方判别式，返回 C6 组及 False
        return ((S6TransitiveSubgroups.C6, False) if has_square_disc(f1)
                # 否则返回 D6 组及 False
                else (S6TransitiveSubgroups.D6, False))

    # 如果 L 是 [3, 3]，表明因子分组为 [3, 3]
    elif L == [3, 3]:
        # 获取第一个和第二个次数为 3 的因子 f1 和 f2
        f1, f2 = factors_by_deg[3]
        # 检查 f1 或 f2 是否有平方判别式
        any_square = has_square_disc(f1) or has_square_disc(f2)
        # 如果任意一个有平方判别式，返回 G18 组及 False
        return ((S6TransitiveSubgroups.G18, False) if any_square
                # 否则返回 G36m 组及 False
                else (S6TransitiveSubgroups.G36m, False))
    elif L == [2, 4]:
        # 如果 L 等于 [2, 4]
        if T_has_sq_disc:
            # 如果 T 有平方的不可约项，则返回 (S6TransitiveSubgroups.S4p, True)
            return (S6TransitiveSubgroups.S4p, True)
        else:
            # 否则，从 factors_by_deg[4] 中获取第一个因子
            f1 = factors_by_deg[4][0]
            # 根据 f1 是否有平方的不可约项，返回不同的结果
            return ((S6TransitiveSubgroups.A4xC2, False) if has_square_disc(f1)
                    else (S6TransitiveSubgroups.S4xC2, False))

    elif L == [1, 1, 4]:
        # 如果 L 等于 [1, 1, 4]
        return ((S6TransitiveSubgroups.A4, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.S4m, False))

    elif L == [1, 5]:
        # 如果 L 等于 [1, 5]
        return ((S6TransitiveSubgroups.PSL2F5, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.PGL2F5, False))

    elif L == [1, 1, 1, 3]:
        # 如果 L 等于 [1, 1, 1, 3]
        return (S6TransitiveSubgroups.S3, False)

    assert L == [6]
    # 如果 L 不等于 [6]，则触发断言错误

    # Second resolvent:
    # 第二解析式：

    history = set()
    # 创建一个空集合 history，用于存储历史记录
    for i in range(max_tries):
        # 迭代 max_tries 次数
        R_dup = get_resolvent_by_lookup(T, 2)
        # 通过查找函数获取 T 的二次解析式并赋值给 R_dup
        if dup_sqf_p(R_dup, ZZ):
            # 如果 R_dup 是平方自由的
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
        # 否则，通过特兴豪森变换更新 T
    else:
        # 如果 for 循环正常退出（未触发 break），则抛出 MaxTriesException 异常
        raise MaxTriesException

    T_has_sq_disc = has_square_disc(T)
    # 检查 T 是否有平方的不可约项并赋值给 T_has_sq_disc

    if dup_irreducible_p(R_dup, ZZ):
        # 如果 R_dup 是不可约的
        return ((S6TransitiveSubgroups.A6, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.S6, False))
        # 返回 (S6TransitiveSubgroups.A6, True) 或 (S6TransitiveSubgroups.S6, False)
    else:
        # 否则（R_dup 可约）
        return ((S6TransitiveSubgroups.G36p, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.G72, False))
        # 返回 (S6TransitiveSubgroups.G36p, True) 或 (S6TransitiveSubgroups.G72, False)
# 定义一个公共函数，计算多项式 *f* 的 Galois 群，支持最高度为6的多项式。

@public
def galois_group(f, *gens, by_name=False, max_tries=30, randomize=False, **args):
    r"""
    计算多项式 *f* 的 Galois 群，支持最高度为6的多项式。

    Examples
    ========

    >>> from sympy import galois_group
    >>> from sympy.abc import x
    >>> f = x**4 + 1
    >>> G, alt = galois_group(f)
    >>> print(G)
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    返回的群对象和一个布尔值，指示是否包含在交错群 $A_n$ 中，其中 $n$ 是 *T* 的次数。
    这些信息可以帮助确定群的性质：

    >>> alt
    True
    >>> G.order()
    4

    可以选择通过名称返回群对象：

    >>> G_name, _ = galois_group(f, by_name=True)
    >>> print(G_name)
    S4TransitiveSubgroups.V

    然后可以通过名称的 ``get_perm_group()`` 方法获取群对象：

    >>> G_name.get_perm_group()
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    群的名称是 :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups` 等枚举类的值。

    Parameters
    ==========

    f : Expr
        要确定 Galois 群的既约多项式，系数为 :ref:`ZZ` 或 :ref:`QQ`。
    gens : optional list of symbols
        用于将 *f* 转换为 Poly 的符号列表，将传递给 :py:func:`~.poly_from_expr` 函数。
    by_name : bool, default False
        如果为 ``True``，则返回群对象的名称。否则返回 :py:class:`~.PermutationGroup`。
    max_tries : int, default 30
        在涉及生成 Tschirnhausen 变换的步骤中，最多尝试此次数。
    randomize : bool, default False
        如果为 ``True``，则在生成 Tschirnhausen 变换时使用随机系数。
        否则按固定顺序尝试变换。两种方法都从小系数和次数开始，逐步增加。
    args : optional
        用于将 *f* 转换为 Poly 的参数，将传递给 :py:func:`~.poly_from_expr` 函数。

    Returns
    =======

    Pair ``(G, alt)``
        第一个元素 ``G`` 表示 Galois 群。如果 *by_name* 是 ``True``，则它是
        :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups` 等枚举类的实例，
        否则是 :py:class:`~.PermutationGroup`。

        第二个元素是一个布尔值，指示群是否包含在交错群 $A_n$ 中（$n$ 是 *T* 的次数）。

    Raises
    ======

    ValueError
        如果 *f* 的次数不受支持。
    """
    # 定义文档字符串，描述了函数的作用、参数、异常情况和相关参考信息
    MaxTriesException
        if could not complete before exceeding *max_tries* in those steps
        that involve generating Tschirnhausen transformations.

    See Also
    ========

    .Poly.galois_group

    """
    # 初始化参数，如果未提供生成器(gens)和参数(args)，则设为空列表和空字典
    gens = gens or []
    args = args or {}

    try:
        # 尝试将表达式 f 转化为多项式 F，并返回结果及选项信息
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 若转化失败，则抛出计算失败异常，附带失败原因
        raise ComputationFailed('galois_group', 1, exc)

    # 调用多项式 F 的 galois_group 方法，计算其 Galois 群
    return F.galois_group(by_name=by_name, max_tries=max_tries,
                          randomize=randomize)
```