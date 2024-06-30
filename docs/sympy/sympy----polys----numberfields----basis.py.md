# `D:\src\scipysrc\sympy\sympy\polys\numberfields\basis.py`

```
"""Computing integral bases for number fields. """

# 导入必要的模块和类
from sympy.polys.polytools import Poly
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.utilities.decorator import public
from .modules import ModuleEndomorphism, ModuleHomomorphism, PowerBasis
from .utilities import extract_fundamental_discriminant

# 定义一个私有函数，应用 Dedekind 判据来扩展给定素数 p 相对于多项式 T 的序
def _apply_Dedekind_criterion(T, p):
    r"""
    Apply the "Dedekind criterion" to test whether the order needs to be
    enlarged relative to a given prime *p*.
    """
    x = T.gen
    T_bar = Poly(T, modulus=p)  # 构造在模 p 下的多项式 T_bar
    lc, fl = T_bar.factor_list()  # 对 T_bar 进行因式分解
    assert lc == 1  # 确保最高次项系数为 1
    g_bar = Poly(1, x, modulus=p)  # 构造 x 模 p 的单位多项式
    for ti_bar, _ in fl:
        g_bar *= ti_bar  # 计算 T_bar 的不可约因子的乘积 g_bar
    h_bar = T_bar // g_bar  # 计算 T_bar 除以 g_bar 的商 h_bar
    g = Poly(g_bar, domain=ZZ)  # 将 g_bar 转换为整数多项式
    h = Poly(h_bar, domain=ZZ)  # 将 h_bar 转换为整数多项式
    f = (g * h - T) // p  # 计算 (g * h - T) // p
    f_bar = Poly(f, modulus=p)  # 计算 (g * h - T) 模 p 的结果
    Z_bar = f_bar
    for b in [g_bar, h_bar]:
        Z_bar = Z_bar.gcd(b)  # 计算 Z_bar 和 g_bar, h_bar 的最大公因子
    U_bar = T_bar // Z_bar  # 计算 T_bar 除以 Z_bar 的商 U_bar
    m = Z_bar.degree()  # 计算 Z_bar 的次数
    return U_bar, m  # 返回 U_bar 和 m

# 计算给定模块 H 和素数 p 下的零根式
def nilradical_mod_p(H, p, q=None):
    r"""
    Compute the nilradical mod *p* for a given order *H*, and prime *p*.

    Explanation
    ===========

    This is the ideal $I$ in $H/pH$ consisting of all elements some positive
    power of which is zero in this quotient ring, i.e. is a multiple of *p*.

    Parameters
    ==========

    H : :py:class:`~.Submodule`
        The given order.
    p : int
        The rational prime.
    q : int, optional
        If known, the smallest power of *p* that is $>=$ the dimension of *H*.
        If not provided, we compute it here.

    Returns
    =======

    :py:class:`~.Module` representing the nilradical mod *p* in *H*.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.
    (See Lemma 6.1.6.)

    """
    n = H.n  # 获取模块 H 的维度
    if q is None:
        q = p
        while q < n:
            q *= p  # 计算大于等于模块 H 维度的 p 的最小幂次 q
    phi = ModuleEndomorphism(H, lambda x: x**q)  # 定义模块的自同态 phi
    return phi.kernel(modulus=p)  # 返回 phi 的核，即模 p 下的零根式

# 执行 Round Two 算法的第二次扩展
def _second_enlargement(H, p, q):
    r"""
    Perform the second enlargement in the Round Two algorithm.
    """
    Ip = nilradical_mod_p(H, p, q=q)  # 计算模 p 下的零根式 Ip
    B = H.parent.submodule_from_matrix(H.matrix * Ip.matrix, denom=H.denom)  # 从矩阵计算模块 B
    C = B + p*H  # 计算模块 C
    E = C.endomorphism_ring()  # 计算 C 的自同态环
    phi = ModuleHomomorphism(H, E, lambda x: E.inner_endomorphism(x))  # 定义模块同态 phi
    gamma = phi.kernel(modulus=p)  # 计算 phi 的核 gamma
    G = H.parent.submodule_from_matrix(H.matrix * gamma.matrix, denom=H.denom * p)  # 计算模块 G
    H1 = G + H  # 计算模块 H1
    return H1, Ip  # 返回 H1 和 Ip

# 公开函数，执行 Zassenhaus 的 Round 2 算法
@public
def round_two(T, radicals=None):
    r"""
    Zassenhaus's "Round 2" algorithm.

    Explanation
    ===========

    Carry out Zassenhaus's "Round 2" algorithm on an irreducible polynomial
    *T* over :ref:`ZZ` or :ref:`QQ`. This computes an integral basis and the
    discriminant for the field $K = \mathbb{Q}[x]/(T(x))$.

    Alternatively, you may pass an :py:class:`~.AlgebraicField` instance, in
    K = None
    # 初始化变量 K，用于存储代数域对象或者置空（None）

    if isinstance(T, AlgebraicField):
        # 检查 T 是否为 AlgebraicField 类型的对象
        K, T = T, T.ext.minpoly_of_element()
        # 如果是，则将 K 设置为 T，同时将 T 设置为 T 的扩展的最小元素的最小多项式
    # 检查多项式环 T 的性质：必须是单变量且是不可约的，定义域必须是 ZZ 或 QQ
    if (   not T.is_univariate
        or not T.is_irreducible
        or T.domain not in [ZZ, QQ]):
        raise ValueError('Round 2 requires an irreducible univariate polynomial over ZZ or QQ.')
    
    # 将多项式 T 转换为首一多项式，并通过缩放根的方式实现
    T, _ = T.make_monic_over_integers_by_scaling_roots()
    
    # 计算多项式 T 的次数
    n = T.degree()
    
    # 计算多项式 T 的判别式
    D = T.discriminant()
    
    # 将判别式 D 转换为 ZZ 类型的整数
    D_modulus = ZZ.from_sympy(abs(D))
    
    # 从判别式 D 中提取基本判别式和对应的 F 值
    _, F = extract_fundamental_discriminant(D)
    
    # 使用多项式环 T 或 K 来创建幂基础 Ztheta
    Ztheta = PowerBasis(K or T)
    
    # 计算幂基础 Ztheta 的整体子模
    H = Ztheta.whole_submodule()
    
    # 初始化 nilrad 为 None
    nilrad = None
    
    # 进行主循环，处理每一个 F 值
    while F:
        # 从 F 中弹出一个素数和其对应的指数
        p, e = F.popitem()
        
        # 应用 Dedekind 判据来计算 U_bar 和 m
        U_bar, m = _apply_Dedekind_criterion(T, p)
        
        # 如果 m 等于 0，则继续下一轮循环
        if m == 0:
            continue
        
        # 将 U_bar 转换为 ZZ 域的多项式，并从中创建元素 U
        U = Ztheta.element_from_poly(Poly(U_bar, domain=ZZ))
        
        # 对当前的整体子模 H 进行模 p 的 HNF 扩展
        H = H.add(U // p * H, hnf_modulus=D_modulus)
        
        # 如果 e 小于等于 m，则继续下一轮循环
        if e <= m:
            continue
        
        # 计算 q，即 p 的幂次
        q = p
        while q < n:
            q *= p
        
        # 进行第二次扩展，并获取扩展后的 H1 和 nilrad
        H1, nilrad = _second_enlargement(H, p, q)
        
        # 反复进行第二次扩展，直到 H1 等于 H
        while H1 != H:
            H = H1
            H1, nilrad = _second_enlargement(H, p, q)
    
    # 注意：我们仅存储所有 nilradicals mod p 中的最后一个。这是因为除非对整个积分基础计算，
    # 否则可能不准确。（换句话说，如果在将 H 传递给 `_second_enlargement` 时，它不已经等于 ZK，
    # 那么我们无法信任如上计算的 nilradical。）例如：如果 T(x) = x ** 3 + 15 * x ** 2 - 9 * x + 13，
    # 那么 F 可以被 2、3 和 7 整除，计算出的模 2 的 nilradical 可能不准确，对于完整的、最大的秩 ZK。
    if nilrad is not None and isinstance(radicals, dict):
        radicals[p] = nilrad
    
    # 将 H 赋值给 ZK
    ZK = H
    
    # 预设已知的昂贵布尔属性为真：
    ZK._starts_with_unity = True
    ZK._is_sq_maxrank_HNF = True
    
    # 计算 dK，即 D 乘以 ZK 矩阵行列式的平方，再除以 ZK 的分母的 2*n 次方
    dK = (D * ZK.matrix.det() ** 2) // ZK.denom ** (2 * n)
    
    # 返回 ZK 和 dK
    return ZK, dK
```