# `D:\src\scipysrc\sympy\sympy\combinatorics\homomorphisms.py`

```
import itertools  # 导入 itertools 库，用于生成迭代器和组合
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation  # 导入有关自由群的类和函数
from sympy.combinatorics.free_groups import FreeGroup  # 导入自由群类
from sympy.combinatorics.perm_groups import PermutationGroup  # 导入排列群类
from sympy.core.intfunc import igcd  # 导入整数最大公约数函数
from sympy.functions.combinatorial.numbers import totient  # 导入计算欧拉函数的函数
from sympy.core.singleton import S  # 导入 SymPy 单例 S

class GroupHomomorphism:
    '''
    A class representing group homomorphisms. Instantiate using `homomorphism()`.

    References
    ==========

    .. [1] Holt, D., Eick, B. and O'Brien, E. (2005). Handbook of computational group theory.

    '''

    def __init__(self, domain, codomain, images):
        self.domain = domain  # 设置域
        self.codomain = codomain  # 设置余域
        self.images = images  # 设置映射关系
        self._inverses = None  # 初始化逆映射为 None
        self._kernel = None  # 初始化核为 None
        self._image = None  # 初始化像为 None

    def _invs(self):
        '''
        Return a dictionary with `{gen: inverse}` where `gen` is a rewriting
        generator of `codomain` (e.g. strong generator for permutation groups)
        and `inverse` is an element of its preimage

        '''
        image = self.image()  # 获取映射的像
        inverses = {}  # 初始化逆映射字典
        for k in list(self.images.keys()):  # 遍历映射关系中的键
            v = self.images[k]  # 获取映射值
            if not (v in inverses or v.is_identity):  # 如果映射值不在逆映射中且不是恒等元素
                inverses[v] = k  # 添加到逆映射字典中
        if isinstance(self.codomain, PermutationGroup):  # 如果余域是排列群
            gens = image.strong_gens  # 获取强发生集合
        else:
            gens = image.generators  # 否则获取生成器集合
        for g in gens:  # 遍历生成器集合
            if g in inverses or g.is_identity:  # 如果生成器已经在逆映射中或者是恒等元素
                continue  # 继续下一轮循环
            w = self.domain.identity  # 获取域的单位元素
            if isinstance(self.codomain, PermutationGroup):  # 如果余域是排列群
                parts = image._strong_gens_slp[g][::-1]  # 获取强发生生成器的逆序生成器
            else:
                parts = g  # 否则直接使用生成器
            for s in parts:  # 遍历生成器的部分
                if s in inverses:  # 如果部分在逆映射中
                    w = w * inverses[s]  # 更新逆映射
                else:
                    w = w * inverses[s ** -1] ** -1  # 更新逆映射
            inverses[g] = w  # 添加生成器到逆映射字典

        return inverses  # 返回逆映射字典
    def invert(self, g):
        '''
        Return an element of the preimage of ``g`` or of each element
        of ``g`` if ``g`` is a list.

        Explanation
        ===========

        If the codomain is an FpGroup, the inverse for equal
        elements might not always be the same unless the FpGroup's
        rewriting system is confluent. However, making a system
        confluent can be time-consuming. If it's important, try
        `self.codomain.make_confluent()` first.

        '''
        # 导入必要的模块和类
        from sympy.combinatorics import Permutation
        from sympy.combinatorics.free_groups import FreeGroupElement

        # 如果 g 是 Permutation 或 FreeGroupElement 类型
        if isinstance(g, (Permutation, FreeGroupElement)):
            # 如果 codomain 是 FpGroup，对 g 进行约化
            if isinstance(self.codomain, FpGroup):
                g = self.codomain.reduce(g)
            
            # 如果尚未计算过逆元素的字典，进行计算
            if self._inverses is None:
                self._inverses = self._invs()
            
            # 获取映像图像
            image = self.image()
            # 初始元素为域的单位元素
            w = self.domain.identity
            
            # 如果 codomain 是 PermutationGroup
            if isinstance(self.codomain, PermutationGroup):
                # 生成器的乘积与 g 的逆序
                gens = image.generator_product(g)[::-1]
            else:
                gens = g
            
            # 遍历生成器列表
            for i in range(len(gens)):
                s = gens[i]
                # 如果 s 是单位元素，继续下一个循环
                if s.is_identity:
                    continue
                # 如果 s 在已计算的逆元素字典中
                if s in self._inverses:
                    w = w * self._inverses[s]
                else:
                    w = w * self._inverses[s**-1]**-1
            
            # 返回计算结果 w
            return w
        
        # 如果 g 是列表类型
        elif isinstance(g, list):
            # 对列表中每个元素递归调用 invert 方法，并返回结果列表
            return [self.invert(e) for e in g]

    def kernel(self):
        '''
        Compute the kernel of `self`.

        '''
        # 如果 kernel 尚未计算，则进行计算
        if self._kernel is None:
            self._kernel = self._compute_kernel()
        
        # 返回 kernel
        return self._kernel

    def _compute_kernel(self):
        # 获取域 G 和其阶 G_order
        G = self.domain
        G_order = G.order()
        
        # 如果 G_order 是无穷的，抛出未实现错误
        if G_order is S.Infinity:
            raise NotImplementedError(
                "Kernel computation is not implemented for infinite groups")
        
        # 初始化生成器列表为空
        gens = []
        
        # 如果域 G 是 PermutationGroup 类型
        if isinstance(G, PermutationGroup):
            # 创建仅包含单位元素的置换群 K
            K = PermutationGroup(G.identity)
        else:
            # 对于 FpGroup，创建一个正规子群 K
            K = FpSubgroup(G, gens, normal=True)
        
        # 计算映像的阶 i
        i = self.image().order()
        
        # 当 K 的阶乘以 i 不等于 G 的阶时，进行循环
        while K.order() * i != G_order:
            # 在域 G 中随机选择元素 r
            r = G.random()
            # 计算元素 k，其等于 r 乘以其逆的逆元素
            k = r * self.invert(self(r))**-1
            
            # 如果 k 不在 K 中，则添加到生成器列表 gens 中
            if k not in K:
                gens.append(k)
                # 如果 G 是 PermutationGroup 类型，则更新 K
                if isinstance(G, PermutationGroup):
                    K = PermutationGroup(gens)
                else:
                    K = FpSubgroup(G, gens, normal=True)
        
        # 返回计算得到的 kernel K
        return K
    def image(self):
        '''
        Compute the image of `self`.
        计算 `self` 的像。

        '''
        if self._image is None:
            values = list(set(self.images.values()))
            if isinstance(self.codomain, PermutationGroup):
                self._image = self.codomain.subgroup(values)
            else:
                self._image = FpSubgroup(self.codomain, values)
        return self._image

    def _apply(self, elem):
        '''
        Apply `self` to `elem`.
        将 `self` 应用于 `elem`。

        '''
        if elem not in self.domain:
            if isinstance(elem, (list, tuple)):
                return [self._apply(e) for e in elem]
            raise ValueError("The supplied element does not belong to the domain")
        if elem.is_identity:
            return self.codomain.identity
        else:
            images = self.images
            value = self.codomain.identity
            if isinstance(self.domain, PermutationGroup):
                gens = self.domain.generator_product(elem, original=True)
                for g in gens:
                    if g in self.images:
                        value = images[g]*value
                    else:
                        value = images[g**-1]**-1*value
            else:
                i = 0
                for _, p in elem.array_form:
                    if p < 0:
                        g = elem[i]**-1
                    else:
                        g = elem[i]
                    value = value*images[g]**p
                    i += abs(p)
        return value

    def __call__(self, elem):
        return self._apply(elem)

    def is_injective(self):
        '''
        Check if the homomorphism is injective
        检查同态映射是否是单射（单射性质）。

        '''
        return self.kernel().order() == 1

    def is_surjective(self):
        '''
        Check if the homomorphism is surjective
        检查同态映射是否是满射（满射性质）。

        '''
        im = self.image().order()
        oth = self.codomain.order()
        if im is S.Infinity and oth is S.Infinity:
            return None
        else:
            return im == oth

    def is_isomorphism(self):
        '''
        Check if `self` is an isomorphism.
        检查 `self` 是否为同构映射。

        '''
        return self.is_injective() and self.is_surjective()

    def is_trivial(self):
        '''
        Check is `self` is a trivial homomorphism, i.e. all elements
        are mapped to the identity.
        检查 `self` 是否是平凡同态映射，即所有元素都映射到恒等元素。

        '''
        return self.image().order() == 1

    def compose(self, other):
        '''
        Return the composition of `self` and `other`, i.e.
        the homomorphism phi such that for all g in the domain
        of `other`, phi(g) = self(other(g))
        返回 `self` 和 `other` 的复合映射，即对于 `other` 域中的所有 g，
        phi(g) = self(other(g))

        '''
        if not other.image().is_subgroup(self.domain):
            raise ValueError("The image of `other` must be a subgroup of "
                    "the domain of `self`")
        images = {g: self(other(g)) for g in other.images}
        return GroupHomomorphism(other.domain, self.codomain, images)
    def restrict_to(self, H):
        '''
        将同态映射限制到域的子群 `H` 上，并返回限制后的结果。
        '''
        # 检查参数 H 是否为 PermutationGroup 类型且是域 self.domain 的子群
        if not isinstance(H, PermutationGroup) or not H.is_subgroup(self.domain):
            raise ValueError("Given H is not a subgroup of the domain")
        domain = H
        # 生成字典 images，包含 H 的生成元 g 及其映射值 self(g)
        images = {g: self(g) for g in H.generators}
        # 返回一个新的 GroupHomomorphism 对象，表示限制后的同态映射
        return GroupHomomorphism(domain, self.codomain, images)

    def invert_subgroup(self, H):
        '''
        返回域的子群，该子群是同态映射图像的逆像子群 `H`。

        '''
        # 检查 H 是否为 self.image() 的子群
        if not H.is_subgroup(self.image()):
            raise ValueError("Given H is not a subgroup of the image")
        gens = []
        # 创建包含映像恒等元素的 PermutationGroup 对象 P
        P = PermutationGroup(self.image().identity)
        # 对于 H 的生成元 h，计算其逆元素 h_i，并添加到 gens 中
        for h in H.generators:
            h_i = self.invert(h)
            if h_i not in P:
                gens.append(h_i)
                P = PermutationGroup(gens)
            # 对于每个核心的生成元 k，检查 k*h_i 是否在 P 中，若不在则添加
            for k in self.kernel().generators:
                if k*h_i not in P:
                    gens.append(k*h_i)
                    P = PermutationGroup(gens)
        # 返回包含 gens 中所有元素的 PermutationGroup 对象 P
        return P
# 创建一个群同态映射（如果可能），从群“domain”到群“codomain”，其定义基于domain的生成元“gens”的映像。
# “gens”和“images”可以是相同大小的列表或元组。如果“gens”是群生成元的真子集，
# 未指定的生成元将映射到恒等元素。如果未指定映像，则创建平凡同态映射。
# 如果给定的生成元映像不定义同态映射，则会引发异常。
# 如果“check”为False，则不检查给定的映像是否实际定义了同态映射。
def homomorphism(domain, codomain, gens, images=(), check=True):
    if not isinstance(domain, (PermutationGroup, FpGroup, FreeGroup)):
        raise TypeError("The domain must be a group")
    if not isinstance(codomain, (PermutationGroup, FpGroup, FreeGroup)):
        raise TypeError("The codomain must be a group")

    # 获取domain的生成元
    generators = domain.generators
    # 检查给定的生成元是否都属于domain的生成元
    if not all(g in generators for g in gens):
        raise ValueError("The supplied generators must be a subset of the domain's generators")
    # 检查给定的映像是否都属于codomain
    if not all(g in codomain for g in images):
        raise ValueError("The images must be elements of the codomain")

    # 如果指定了映像，则映像的数量必须与生成元的数量相等
    if images and len(images) != len(gens):
        raise ValueError("The number of images must be equal to the number of generators")

    # 将gens和images转换为列表形式，并确保images的长度为generators的长度，未指定的映像将为恒等元素
    gens = list(gens)
    images = list(images)
    images.extend([codomain.identity]*(len(generators)-len(images)))
    gens.extend([g for g in generators if g not in gens])
    # 创建生成元到映像的字典
    images = dict(zip(gens, images))

    # 如果需要检查且给定的映像不定义同态映射，则引发异常
    if check and not _check_homomorphism(domain, codomain, images):
        raise ValueError("The given images do not define a homomorphism")
    
    # 返回群同态映射对象
    return GroupHomomorphism(domain, codomain, images)


# 检查给定的生成元映像是否定义了同态映射。
def _check_homomorphism(domain, codomain, images):
    # 获取domain的表示并获取相关的生成元和关系
    pres = domain if hasattr(domain, 'relators') else domain.presentation()
    rels = pres.relators
    gens = pres.generators
    # 获取生成元的符号标识
    symbols = [g.ext_rep[0] for g in gens]
    # 创建符号到domain生成元的映射
    symbols_to_domain_generators = dict(zip(symbols, domain.generators))
    # 获取codomain的恒等元素
    identity = codomain.identity

    # 定义计算单词映像的内部函数
    def _image(r):
        w = identity
        # 遍历关系的符号形式，计算生成元的映像
        for symbol, power in r.array_form:
            g = symbols_to_domain_generators[symbol]
            w *= images[g]**power
        return w
    # 遍历关系集合rels中的每一个关系r
    for r in rels:
        # 如果codomain是FpGroup类型的实例
        if isinstance(codomain, FpGroup):
            # 计算r的映像_image(r)，并与identity进行比较
            s = codomain.equals(_image(r), identity)
            # 如果无法确定它们是否相等
            if s is None:
                # 仅在无法确定等式真实性时尝试使重写系统confluent
                success = codomain.make_confluent()
                # 重新计算r的映像并与identity比较
                s = codomain.equals(_image(r), identity)
                # 如果仍然无法确定并且重写系统confluent操作不成功
                if s is None and not success:
                    # 抛出运行时错误，指示无法确定映像是否定义同态
                    raise RuntimeError("Can't determine if the images "
                        "define a homomorphism. Try increasing "
                        "the maximum number of rewriting rules "
                        "(group._rewriting_system.set_max(new_value); "
                        "the current value is stored in group._rewriting"
                        "_system.maxeqns)")
        else:
            # 如果codomain不是FpGroup类型的实例，则直接检查r的映像是否是identity
            s = _image(r).is_identity
        # 如果s为False，即r的映像不是identity，则返回False
        if not s:
            return False
    # 如果所有rels中的关系r的映像都是identity，则返回True
    return True
# 计算由置换群 `group` 在集合 `omega` 上诱导的同态映射，该映射在作用下保持封闭性
def orbit_homomorphism(group, omega):
    # 导入必要的模块和类
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.named_groups import SymmetricGroup

    # 创建目标群，其阶数与 omega 的长度相同
    codomain = SymmetricGroup(len(omega))
    # 获取目标群的单位元素
    identity = codomain.identity

    # 将 omega 转换为列表形式
    omega = list(omega)

    # 生成置换群 `group` 中每个生成元的像，即生成元的置换对应的映射
    images = {g: identity * Permutation([omega.index(o ^ g) for o in omega]) for g in group.generators}

    # 对置换群 `group` 进行 Schreier Sims 算法处理
    group._schreier_sims(base=omega)

    # 创建群同态映射对象 H，其中包含了群 `group` 到目标群 `codomain` 的映射 `images`
    H = GroupHomomorphism(group, codomain, images)

    # 根据基本稳定子的数量与 omega 的长度比较，确定同态映射 H 的核
    if len(group.basic_stabilizers) > len(omega):
        H._kernel = group.basic_stabilizers[len(omega)]
    else:
        H._kernel = PermutationGroup([group.identity])

    return H

# 计算由置换群 `group` 在块系统 `blocks` 上诱导的同态映射
def block_homomorphism(group, blocks):
    # 导入必要的模块和类
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.named_groups import SymmetricGroup

    # 获取块系统的长度
    n = len(blocks)

    # 初始化变量 m, p, b
    m = 0
    p = []
    b = [None] * n

    # 构建块系统的编号及其代表元
    for i in range(n):
        if blocks[i] == i:
            p.append(i)
            b[i] = m
            m += 1

    # 为每个块编号，将 blocks[i] 所属的块编号存储在 b[i] 中
    for i in range(n):
        b[i] = b[blocks[i]]

    # 创建目标群，其阶数与块系统中块的数量相同
    codomain = SymmetricGroup(m)
    # 目标群中的单位置换
    identity = range(m)

    # 生成置换群 `group` 中每个生成元的像，即生成元的置换对应的映射
    images = {g: Permutation([b[p[i] ^ g] for i in identity]) for g in group.generators}

    # 创建群同态映射对象 H，其中包含了群 `group` 到目标群 `codomain` 的映射 `images`
    H = GroupHomomorphism(group, codomain, images)

    return H

# 计算两个给定群之间的同构映射
def group_isomorphism(G, H, isomorphism=True):
    '''
    计算两个给定群之间的同构映射。

    Parameters
    ==========

    G : 一个有限的 `FpGroup` 或 `PermutationGroup`。
        第一个群。

    H : 一个有限的 `FpGroup` 或 `PermutationGroup`。
        第二个群。

    isomorphism : bool
        用于避免在用户仅想检查群之间是否存在同构时进行同态映射的计算。

    Returns
    =======

    如果 isomorphism = False -- 返回一个布尔值。
    如果 isomorphism = True  -- 返回一个布尔值和 `G` 到 `H` 之间的同构映射。

    Examples
    ========

    >>> from sympy.combinatorics import free_group, Permutation
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> from sympy.combinatorics.fp_groups import FpGroup
    >>> from sympy.combinatorics.homomorphisms import group_isomorphism
    '''
    >>> from sympy.combinatorics.named_groups import DihedralGroup, AlternatingGroup

    >>> D = DihedralGroup(8)  # 创建一个阶为8的二面体群D
    >>> p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)  # 创建一个排列p，包含8个元素
    >>> P = PermutationGroup(p)  # 使用排列p创建置换群P
    >>> group_isomorphism(D, P)  # 检查群D和群P之间是否同构，返回结果为(False, None)

    >>> F, a, b = free_group("a, b")  # 创建自由群F，包含生成元a和b
    >>> G = FpGroup(F, [a**3, b**3, (a*b)**2])  # 使用自由群F和定义的关系创建群G
    >>> H = AlternatingGroup(4)  # 创建一个阶为4的交错群H
    >>> (check, T) = group_isomorphism(G, H)  # 检查群G和群H之间是否同构，并返回检查结果和同构映射T
    >>> check  # 输出检查结果
    True
    >>> T(b*a*b**-1*a**-1*b**-1)  # 对同构映射T应用一个置换，返回置换结果(0 2 3)

    Notes
    =====

    Uses the approach suggested by Robert Tarjan to compute the isomorphism between two groups.
    First, the generators of ``G`` are mapped to the elements of ``H`` and
    we check if the mapping induces an isomorphism.

    '''
    if not isinstance(G, (PermutationGroup, FpGroup)):
        raise TypeError("The group must be a PermutationGroup or an FpGroup")
    if not isinstance(H, (PermutationGroup, FpGroup)):
        raise TypeError("The group must be a PermutationGroup or an FpGroup")

    if isinstance(G, FpGroup) and isinstance(H, FpGroup):
        G = simplify_presentation(G)  # 简化群G的表示
        H = simplify_presentation(H)  # 简化群H的表示
        # 两个具有相同生成元的无限FpGroup在它们的关系相同但顺序不同时是同构的
        if G.generators == H.generators and sorted(G.relators) == sorted(H.relators):
            if not isomorphism:
                return True
            return (True, homomorphism(G, H, G.generators, H.generators))

    # `_H`是与`H`同构的置换群。
    _H = H
    g_order = G.order()  # 计算群G的阶数
    h_order = H.order()  # 计算群H的阶数

    if g_order is S.Infinity:
        raise NotImplementedError("Isomorphism methods are not implemented for infinite groups.")

    if isinstance(H, FpGroup):
        if h_order is S.Infinity:
            raise NotImplementedError("Isomorphism methods are not implemented for infinite groups.")
        _H, h_isomorphism = H._to_perm_group()  # 将FpGroup H转换为对应的置换群和同构映射

    if g_order != h_order or G.is_abelian != H.is_abelian:
        if not isomorphism:
            return False
        return (False, None)

    if not isomorphism:
        # 对于同阶循环数不同的两个群，它们之间是同构的
        n = g_order
        if gcd(n, totient(n)) == 1:
            return True

    # 将`G`的生成元与`_H`的子集进行匹配
    gens = list(G.generators)
    for subset in itertools.permutations(_H, len(gens)):
        images = list(subset)
        images.extend([_H.identity] * (len(G.generators) - len(images)))
        _images = dict(zip(gens, images))
        if _check_homomorphism(G, _H, _images):  # 检查是否存在同态映射
            if isinstance(H, FpGroup):
                images = h_isomorphism.invert(images)  # 对于FpGroup H，反转映射
            T = homomorphism(G, H, G.generators, images, check=False)  # 创建同构映射T
            if T.is_isomorphism():  # 验证是否为同构映射
                # 这是一个有效的同构映射
                if not isomorphism:
                    return True
                return (True, T)
    # 如果没有给定同构关系（isomorphism），则返回 False
    if not isomorphism:
        # 返回一个元组，第一个元素为 False，第二个元素为 None
        return False
    # 否则返回一个包含 False 和 None 的元组
    return (False, None)
# 定义函数 is_isomorphic，用于检查两个群体是否同构（即结构上相同）
'''
Check if the groups are isomorphic to each other

Parameters
==========

G : A finite ``FpGroup`` or a ``PermutationGroup``
    第一个群体，可以是有限 FpGroup 或置换群（PermutationGroup）。

H : A finite ``FpGroup`` or a ``PermutationGroup``
    第二个群体，可以是有限 FpGroup 或置换群（PermutationGroup）。

Returns
=======

boolean
'''
# 调用 group_isomorphism 函数来检查 G 和 H 是否同构，isomorphism=False 表示只检查结构是否相同
return group_isomorphism(G, H, isomorphism=False)
```