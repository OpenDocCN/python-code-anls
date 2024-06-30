# `D:\src\scipysrc\sympy\sympy\combinatorics\tensor_can.py`

```
# 导入需要的符号计算模块中的类和函数
from sympy.combinatorics.permutations import Permutation, _af_rmul, \
    _af_invert, _af_new
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
    _orbit_transversal
from sympy.combinatorics.util import _distribute_gens_by_base, \
    _orbits_transversals_from_bsgs

"""
    张量规范化的参考文献：

    [1] R. Portugal "Algorithmic simplification of tensor expressions",
        J. Phys. A 32 (1999) 7779-7789

    [2] R. Portugal, B.F. Svaiter "Group-theoretic Approach for Symbolic
        Tensor Manipulation: I. Free Indices"
        arXiv:math-ph/0107031v1

    [3] L.R.U. Manssur, R. Portugal "Group-theoretic Approach for Symbolic
        Tensor Manipulation: II. Dummy Indices"
        arXiv:math-ph/0107032v1

    [4] xperm.c part of XPerm written by J. M. Martin-Garcia
        http://www.xact.es/index.html
"""


def dummy_sgs(dummies, sym, n):
    """
    返回虚指标的强生成器列表。

    Parameters
    ==========

    dummies : List of dummy indices.
        `dummies[2k], dummies[2k+1]` are paired indices.
        In base form, the dummy indices are always in
        consecutive positions.
        虚指标列表。`dummies[2k], dummies[2k+1]` 表示成对的虚指标。
        在基础形式中，虚指标总是连续的。

    sym : symmetry under interchange of contracted dummies::
        * None  no symmetry
        * 0     commuting
        * 1     anticommuting
        虚指标在交换时的对称性:
        * None  无对称性
        * 0     交换
        * 1     反交换

    n : number of indices
        指标的数量

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import dummy_sgs
    >>> dummy_sgs(list(range(2, 8)), 0, 8)
    [[0, 1, 3, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 5, 4, 6, 7, 8, 9],
     [0, 1, 2, 3, 4, 5, 7, 6, 8, 9], [0, 1, 4, 5, 2, 3, 6, 7, 8, 9],
     [0, 1, 2, 3, 6, 7, 4, 5, 8, 9]]
    """
    # 如果虚指标列表长度大于总指标数量，抛出异常
    if len(dummies) > n:
        raise ValueError("List too large")
    res = []
    # 如果有对称性，处理协变和逆变指标的交换
    if sym is not None:
        for j in dummies[::2]:
            a = list(range(n + 2))
            if sym == 1:
                a[n] = n + 1
                a[n + 1] = n
            a[j], a[j + 1] = a[j + 1], a[j]
            res.append(a)
    # 重命名虚指标
    for j in dummies[:-3:2]:
        a = list(range(n + 2))
        a[j:j + 4] = a[j + 2], a[j + 3], a[j], a[j + 1]
        res.append(a)
    return res


def _min_dummies(dummies, sym, indices):
    """
    返回在虚指标组的群中指标轨道的最小元素列表。
    详见 ``double_coset_can_rep`` 的描述，关于 ``dummies`` 和 ``sym`` 的信息。
    ``indices`` 是初始的虚指标列表。

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import _min_dummies
    >>> _min_dummies([list(range(2, 8))], [0], list(range(10)))
    [0, 1, 2, 2, 2, 2, 2, 2, 8, 9]
    """
    num_types = len(sym)
    # 计算每个虚指标组的最小元素
    m = [min(dx) if dx else None for dx in dummies]
    res = indices[:]
    for i in range(num_types):
        for c, i in enumerate(indices):
            for j in range(num_types):
                if i in dummies[j]:
                    res[c] = m[j]
                    break
    return res
# 返回满足条件 s[h[b]] == j 的代表 h
def _trace_S(s, j, b, S_cosets):
    """
    Return the representative h satisfying s[h[b]] == j

    If there is not such a representative return None
    """
    # 遍历 S_cosets[b] 中的每个 h
    for h in S_cosets[b]:
        # 如果找到满足条件的 h，则返回该 h
        if s[h[b]] == j:
            return h
    # 如果找不到满足条件的 h，则返回 None
    return None


# 返回满足条件 h[gj] == p_i 的代表 h
def _trace_D(gj, p_i, Dxtrav):
    """
    Return the representative h satisfying h[gj] == p_i

    If there is not such a representative return None
    """
    # 遍历 Dxtrav 中的每个 h
    for h in Dxtrav:
        # 如果找到满足条件的 h，则返回该 h
        if h[gj] == p_i:
            return h
    # 如果找不到满足条件的 h，则返回 None
    return None


# 从 dumx 中移除 p0
def _dumx_remove(dumx, dumx_flat, p0):
    """
    remove p0 from dumx
    """
    res = []
    # 遍历 dumx 中的每个 dx
    for dx in dumx:
        # 如果 p0 不在 dx 中，则保留该 dx
        if p0 not in dx:
            res.append(dx)
            continue
        # 如果 p0 在 dx 中，找到 p0 对应的配对元素 p0_paired
        k = dx.index(p0)
        if k % 2 == 0:
            p0_paired = dx[k + 1]
        else:
            p0_paired = dx[k - 1]
        # 从 dx 和 dumx_flat 中移除 p0 和 p0_paired
        dx.remove(p0)
        dx.remove(p0_paired)
        dumx_flat.remove(p0)
        dumx_flat.remove(p0_paired)
        # 将修改后的 dx 添加到结果列表 res 中
        res.append(dx)


# 将 transversal 转换为 coset 表示形式
def transversal2coset(size, base, transversal):
    a = []
    j = 0
    # 遍历 size 的范围
    for i in range(size):
        # 如果 i 在 base 中，添加 transversal[j].values() 的排序后结果到 a 中
        if i in base:
            a.append(sorted(transversal[j].values()))
            j += 1
        else:
            # 否则添加 [list(range(size))] 到 a 中
            a.append([list(range(size))])
    # 从 a 的末尾向前检查，去除末尾为 [list(range(size))] 的元素
    j = len(a) - 1
    while a[j] == [list(range(size))]:
        j -= 1
    return a[:j + 1]


# 实现 Butler-Portugal 算法进行张量的标准化
def double_coset_can_rep(dummies, sym, b_S, sgens, S_transversals, g):
    r"""
    Butler-Portugal algorithm for tensor canonicalization with dummy indices.

    Parameters
    ==========

      dummies
        list of lists of dummy indices,
        one list for each type of index;
        the dummy indices are put in order contravariant, covariant
        [d0, -d0, d1, -d1, ...].

      sym
        list of the symmetries of the index metric for each type.

      possible symmetries of the metrics
              * 0     symmetric
              * 1     antisymmetric
              * None  no symmetry

      b_S
        base of a minimal slot symmetry BSGS.

      sgens
        generators of the slot symmetry BSGS.

      S_transversals
        transversals for the slot BSGS.

      g
        permutation representing the tensor.

    Returns
    =======

    Return 0 if the tensor is zero, else return the array form of
    the permutation representing the canonical form of the tensor.

    Notes
    =====

    A tensor with dummy indices can be represented in a number
    of equivalent ways which typically grows exponentially with
    the number of indices. To be able to establish if two tensors
    with many indices are equal becomes computationally very slow
    in absence of an efficient algorithm.

    The Butler-Portugal algorithm [3] is an efficient algorithm to
    put tensors in canonical form, solving the above problem.

    Portugal observed that a tensor can be represented by a permutation,
    and that the class of tensors equivalent to it under slot and dummy
    symmetries is equivalent to the double coset `D*g*S`
    """
    pass
    # 根据 Butler 算法寻找双余类的代表，找到张量的规范形式
    # 该算法中使用的排列乘法约定为 (p*q)(i) = p[q[i]]，与 Permutation 类中使用的相反

    # 假设 g 是一个以数组形式表示的排列；一个具有指标 ind 的张量可以写成
    # t = T(ind[g[0]], ..., ind[g[n-1]])
    # 其中 n = len(ind)；
    # g 的大小为 n + 2，最后两个索引是张量符号的标志（在 [4] 中引入的技巧）。

    # 槽对称变换 s 是作用在槽上的排列
    # t -> T(ind[(g*s)[0]], ..., ind[(g*s)[n-1]])

    # 哑符号变换作用于指标 ind
    # t -> T(ind[(d*g)[0]], ..., ind[(d*g)[n-1]])

    # 仅关注张量在这些对称性下的变换，可以用 g 来表示张量，其变换为
    # g -> d*g*s，因此它属于余类 D*g*S，或者说属于由槽和哑符号允许的所有排列组成的集合。

    # 让我们通过一个例子解释这些约定。

    # 给定一个张量 T^{d3 d2 d1}{}_{d1 d2 d3}，具有槽对称性
    # T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}
    # T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}
    # 和对称度量，找到与之等价的张量，其在指标顺序上最低：
    # 字典序顺序 `d1, d2, d3`，且先是逆变指标，后是协变指标；这就是张量的规范形式。

    # 规范形式是 `-T^{d1 d2 d3}{}_{d1 d2 d3}`
    # 通过 `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}` 获得。

    # 要将此问题转换为该函数的输入，请使用以下索引名称的排序
    # (- 表示协变的简写) `d1, -d1, d2, -d2, d3, -d3`

    # `T^{d3 d2 d1}{}_{d1 d2 d3}` 对应于 `g = [4, 2, 0, 1, 3, 5, 6, 7]`
    # 其中最后两个索引是为了表示符号

    # `sgens = [Permutation(0, 2)(6, 7), Permutation(0, 4)(6, 7)]`

    # sgens[0] 是槽对称性 `- (0, 2)`
    # `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`

    # sgens[1] 是槽对称性 `- (0, 4)`
    # `T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}`

    # 哑符号群 D 由强基生成元生成
    # `[(0, 1), (2, 3), (4, 5), (0, 2)(1, 3), (0, 4)(1, 5)]`
    # 其中前三个交换同一索引的协变和逆变位置 (d1 <-> -d1)，后两个交换哑符号索引本身 (d1 <-> d2)。

    # 哑符号从左边作用
    # `d = [1, 0, 2, 3, 4, 5, 6, 7]`  交换 `d1 <-> -d1`
    # `T^{d3 d2 d1}{}_{d1 d2 d3} == T^{d3 d2}{}_{d1}{}^{d1}{}_{d2 d3}`
    `g=[4, 2, 0, 1, 3, 5, 6, 7]  -> [4, 2, 1, 0, 3, 5, 6, 7] = _af_rmul(d, g)`
    which differs from `_af_rmul(g, d)`.


    # 定义变量 g，表示一个具体的置换，将索引位置从 [4, 2, 0, 1, 3, 5, 6, 7] 映射为 [4, 2, 1, 0, 3, 5, 6, 7]
    # 调用 _af_rmul(d, g) 时，此置换应用在 d 上，得到一个新的张量或结果
    # 调用 _af_rmul(g, d) 时，此置换应用在 g 上，得到一个不同的张量或结果



    The slot symmetry acts from the right
    `s = [2, 1, 0, 3, 4, 5, 7, 6]`  exchanges slots 0 and 2 and changes sign


    # 定义变量 s，表示另一个具体的置换，将索引位置从 [2, 1, 0, 3, 4, 5, 7, 6] 映射为 [2, 1, 0, 3, 4, 5, 7, 6]
    # 此置换交换了索引位置 0 和 2，并且改变了符号（可能是指线性代数中的乘法操作）



    `T^{d3 d2 d1}{}_{d1 d2 d3} == -T^{d1 d2 d3}{}_{d1 d2 d3}`


    # 表达式 T^{d3 d2 d1}{}_{d1 d2 d3} 等于 -T^{d1 d2 d3}{}_{d1 d2 d3}
    # 表示张量 T 在特定的对称性操作下具有特定的对称性质



    `g=[4,2,0,1,3,5,6,7]  -> [0, 2, 4, 1, 3, 5, 7, 6] = _af_rmul(g, s)`


    # 定义变量 g，表示一个具体的置换，将索引位置从 [4, 2, 0, 1, 3, 5, 6, 7] 映射为 [0, 2, 4, 1, 3, 5, 7, 6]
    # 调用 _af_rmul(g, s) 时，此置换应用在 g 上，得到一个新的张量或结果



    Example in which the tensor is zero, same slot symmetries as above:
    `T^{d2}{}_{d1 d3}{}^{d1 d3}{}_{d2}`


    # 展示了一个张量为零的例子，具有与前述相同的槽对称性
    # 表达式 T^{d2}{}_{d1 d3}{}^{d1 d3}{}_{d2} 表示了一个特定的张量表达式



    `= -T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`   under slot symmetry `-(0,4)`;


    # 在槽对称性 -(0,4) 下，表达式等于 -T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}
    # 表示了该张量在特定对称性操作下的结果



    `= T_{d3 d1}{}^{d3}{}^{d1 d2}{}_{d2}`    under slot symmetry `-(0,2)`;


    # 在槽对称性 -(0,2) 下，表达式等于 T_{d3 d1}{}^{d3}{}^{d1 d2}{}_{d2}
    # 表示了该张量在特定对称性操作下的结果



    `= T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`    symmetric metric;


    # 对称度量下，表达式等于 T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}
    # 表示了该张量在对称度量条件下的结果



    `= 0`  since two of these lines have tensors differ only for the sign.


    # 结果等于 0，因为这些行中的两个张量仅在符号上有所不同



    The double coset D*g*S consists of permutations `h = d*g*s` corresponding
    to equivalent tensors; if there are two `h` which are the same apart
    from the sign, return zero; otherwise
    choose as representative the tensor with indices
    ordered lexicographically according to `[d1, -d1, d2, -d2, d3, -d3]`
    that is ``rep = min(D*g*S) = min([d*g*s for d in D for s in S])``


    # 双陪集 D*g*S 包含了等效张量对应的置换 `h = d*g*s`；
    # 如果有两个 `h` 除了符号外是相同的，则返回零；否则选择按照 `[d1, -d1, d2, -d2, d3, -d3]` 字典序排序的张量作为代表
    # 即 `rep = min(D*g*S) = min([d*g*s for d in D for s in S])`



    The indices are fixed one by one; first choose the lowest index
    for slot 0, then the lowest remaining index for slot 1, etc.
    Doing this one obtains a chain of stabilizers


    # 索引逐个固定；首先选择槽 0 中的最低索引，然后选择槽 1 中剩余的最低索引，依此类推
    # 这样可以获得一系列的稳定子群（stabilizers）



    `S \rightarrow S_{b0} \rightarrow S_{b0,b1} \rightarrow \dots` and
    `D \rightarrow D_{p0} \rightarrow D_{p0,p1} \rightarrow \dots`


    # `S` 转向 `S_{b0}` 转向 `S_{b0,b1}` 等等，以及 `D` 转向 `D_{p0}` 转向 `D_{p0,p1}` 等等
    # 表示了基于对称群和稳定子群的链式操作



    where ``[b0, b1, ...] = range(b)`` is a base of the symmetric group;
    the strong base `b_S` of S is an ordered sublist of it;
    therefore it is sufficient to compute once the
    strong base generators of S using the Schreier-Sims algorithm;
    the stabilizers of the strong base generators are the
    strong base generators of the stabilizer subgroup.


    # `b0, b1, ...` 是对称群的一个基；S 的强基 `b_S` 是其有序子列表；
    # 因此，使用 Schreier-Sims 算法计算一次 S 的强基生成器就足够了；
    # 强基生成器的稳定子群也是稳定子群的强基生成器



    ``dbase = [p0, p1, ...]`` is not in general in lexicographic order,
    so that one must recompute the strong base generators each time;
    however this is trivial, there is no need to use the Schreier-Sims
    algorithm for D.


    # `dbase = [p0, p1, ...]` 通常不是按字典序排列的，
    # 因此每次必
    # 计算变量size为生成元g的大小
    size = g.size
    # 将生成元g表示为其数组形式
    g = g.array_form
    # 计算虚拟元素的数量，即size减去2
    num_dummies = size - 2
    # 生成虚拟元素的索引列表
    indices = list(range(num_dummies))
    # 检查是否所有的metric都带有对称性
    all_metrics_with_sym = not any(_ is None for _ in sym)
    # 计算对称性的类型数量
    num_types = len(sym)
    # 创建虚拟元素列表的副本
    dumx = dummies[:]
    # 将虚拟元素列表扁平化
    dumx_flat = []
    for dx in dumx:
        dumx_flat.extend(dx)
    # 复制变量b_S
    b_S = b_S[:]
    # 将sgens列表中的每个元素表示为其数组形式
    sgensx = [h._array_form for h in sgens]
    # 如果b_S不为空，则计算其对应的transversals
    if b_S:
        S_transversals = transversal2coset(size, b_S, S_transversals)
    # 计算D的强生成集合
    dsgsx = []
    for i in range(num_types):
        dsgsx.extend(dummy_sgs(dumx[i], sym[i], num_dummies))
    # 创建标识符列表
    idn = list(range(size))
    # TAB为包含条目(s, d, h)的列表，其中h = _af_rmuln(d,g,s)
    # 在下文中，d*g*s表示_af_rmuln(d,g,s)
    TAB = [(idn, idn, g)]
    # 返回TAB中第一个元素的最后一个元素（即h）
    return TAB[0][-1]
def _get_map_slots(size, fixed_slots):
    """
    Generate a mapping for slot indices.

    Parameters
    ==========
    size : int
        Size of the permutation.
    fixed_slots : list
        List of fixed slot indices.

    Returns
    =======
    res : list
        List representing the mapping of slot indices.

    Explanation
    ===========
    This function generates a mapping such that the indices in
    `fixed_slots` remain fixed, and other indices are mapped to
    consecutive integers starting from 0.

    Examples
    ========
    >>> _get_map_slots(5, [1, 3])
    [0, 1, 2, 1, 3]
    """
    res = list(range(size))
    pos = 0
    for i in range(size):
        if i in fixed_slots:
            continue
        res[i] = pos
        pos += 1
    return res



def _lift_sgens(size, fixed_slots, free, s):
    """
    Lift the action of s on the free indices.

    Parameters
    ==========
    size : int
        Size of the permutation.
    fixed_slots : list
        List of fixed slot indices.
    free : list
        List of free indices.
    s : object
        Permutation object representing the action.

    Explanation
    ===========
    This function modifies the permutation `s` such that it acts
    correctly on the indices specified by `free`, while keeping
    the indices in `fixed_slots` fixed according to their relative
    order in the permutation.

    Examples
    ========
    >>> _lift_sgens(5, [1, 3], [0, 2], Permutation([1, 0, 2, 3, 4]))
    None (This is a placeholder example; actual output depends on the function use.)
    """
    a = []
    j = k = 0
    fd = [y for _, y in sorted(zip(fixed_slots, free))]
    num_free = len(free)
    # 对于给定范围内的索引 i，依次处理列表操作
    for i in range(size):
        # 如果当前索引 i 存在于固定槽位列表中
        if i in fixed_slots:
            # 将固定数据列表中索引 k 的值添加到列表 a 中，并更新 k 索引以指向下一个元素
            a.append(fd[k])
            k += 1
        else:
            # 将自由槽位中的元素 s[j] 加上可用自由槽位的数量 num_free 后添加到列表 a 中，并更新 j 索引
            a.append(s[j] + num_free)
            j += 1
    # 返回处理后的列表 a
    return a
# 定义函数 canonicalize，用于规范化由张量组成的张量

def canonicalize(g, dummies, msym, *v):
    """
    canonicalize tensor formed by tensors

    Parameters
    ==========

    g : permutation representing the tensor
        张量的表示排列

    dummies : list representing the dummy indices
      it can be a list of dummy indices of the same type
      or a list of lists of dummy indices, one list for each
      type of index;
      the dummy indices must come after the free indices,
      and put in order contravariant, covariant
      [d0, -d0, d1,-d1,...]
        表示虚指标的列表，可以是相同类型的虚指标列表，也可以是多个类型的虚指标列表；
        虚指标必须在自由指标之后，并按逆变、协变顺序排序

    msym :  symmetry of the metric(s)
        it can be an integer or a list;
        in the first case it is the symmetry of the dummy index metric;
        in the second case it is the list of the symmetries of the
        index metric for each type
        指标度规的对称性，可以是整数或列表；
        在第一种情况下，它是虚指标度规的对称性；
        在第二种情况下，它是每种类型指标度规的对称性列表

    v : list, (base_i, gens_i, n_i, sym_i) for tensors of type `i`
        v：类型为 `i` 的张量的列表，每个元素是 (base_i, gens_i, n_i, sym_i)

    base_i, gens_i : BSGS for tensors of this type.
        The BSGS should have minimal base under lexicographic ordering;
        if not, an attempt is made do get the minimal BSGS;
        in case of failure,
        canonicalize_naive is used, which is much slower.
        base_i, gens_i：此类型张量的 BSGS（基与生成器系统）。
        BSGS 应按字典序具有最小基；
        如果没有，则尝试获取最小的 BSGS；
        如果失败，则使用 canonicalize_naive，这会更慢。

    n_i :    number of tensors of type `i`.
        n_i：类型为 `i` 的张量数量。

    sym_i :  symmetry under exchange of component tensors of type `i`.
        sym_i：在交换类型为 `i` 的分量张量时的对称性。

        Both for msym and sym_i the cases are
            * None  no symmetry
            * 0     commuting
            * 1     anticommuting
        对于 msym 和 sym_i，情况如下：
            * None  没有对称性
            * 0     交换
            * 1     反交换

    Returns
    =======

    0 if the tensor is zero, else return the array form of
    the permutation representing the canonical form of the tensor.
    如果张量为零则返回 0，否则返回表示张量规范形式的排列的数组形式。

    Algorithm
    =========

    First one uses canonical_free to get the minimum tensor under
    lexicographic order, using only the slot symmetries.
    首先使用 canonical_free 根据仅使用槽对称性的词典顺序获取最小张量。

    If the component tensors have not minimal BSGS, it is attempted
    to find it; if the attempt fails canonicalize_naive
    is used instead.
    如果组成张量的分量没有最小的 BSGS，则尝试找到它；如果尝试失败，则使用 canonicalize_naive。

    Compute the residual slot symmetry keeping fixed the free indices
    using tensor_gens(base, gens, list_free_indices, sym).
    使用 tensor_gens(base, gens, list_free_indices, sym) 计算保持自由指标不变的剩余槽对称性。

    Reduce the problem eliminating the free indices.
    减少问题，消除自由指标。

    Then use double_coset_can_rep and lift back the result reintroducing
    the free indices.
    然后使用 double_coset_can_rep 并通过重新引入自由指标来提升结果。

    Examples
    ========

    one type of index with commuting metric;

    `A_{a b}` and `B_{a b}` antisymmetric and commuting

    `T = A_{d0 d1} * B^{d0}{}_{d2} * B^{d2 d1}`

    `ord = [d0,-d0,d1,-d1,d2,-d2]` order of the indices

    g = [1, 3, 0, 5, 4, 2, 6, 7]

    `T_c = 0`
    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, canonicalize, bsgs_direct_product
    >>> from sympy.combinatorics import Permutation
    >>> base2a, gens2a = get_symmetric_group_sgs(2, 1)
    >>> t0 = (base2a, gens2a, 1, 0)
    >>> t1 = (base2a, gens2a, 2, 0)
    >>> g = Permutation([1, 3, 0, 5, 4, 2, 6, 7])
    >>> canonicalize(g, range(6), 0, t0, t1)
    0

    same as above, but with `B_{a b}` anticommuting

    `T_c = -A^{d0 d1} * B_{d0}{}^{d2} * B_{d1 d2}`

    can = [0,2,1,4,3,5,7,6]

    >>> t1 = (base2a, gens2a, 2, 1)
    >>> canonicalize(g, range(6), 0, t0, t1)
    [0, 2, 1, 4, 3, 5, 7, 6]
    """
    pass
    """
    two types of indices `[a,b,c,d,e,f]` and `[m,n]`, in this order,
    both with commuting metric

    `f^{a b c}` antisymmetric, commuting

    `A_{m a}` no symmetry, commuting

    `T = f^c{}_{d a} * f^f{}_{e b} * A_m{}^d * A^{m b} * A_n{}^a * A^{n e}`

    ord = [c,f,a,-a,b,-b,d,-d,e,-e,m,-m,n,-n]

    g = [0,7,3, 1,9,5, 11,6, 10,4, 13,2, 12,8, 14,15]

    The canonical tensor is
    `T_c = -f^{c a b} * f^{f d e} * A^m{}_a * A_{m d} * A^n{}_b * A_{n e}`

    can = [0,2,4, 1,6,8, 10,3, 11,7, 12,5, 13,9, 15,14]

    >>> base_f, gens_f = get_symmetric_group_sgs(3, 1)
    >>> base1, gens1 = get_symmetric_group_sgs(1)
    >>> base_A, gens_A = bsgs_direct_product(base1, gens1, base1, gens1)
    >>> t0 = (base_f, gens_f, 2, 0)
    >>> t1 = (base_A, gens_A, 4, 0)
    >>> dummies = [range(2, 10), range(10, 14)]
    >>> g = Permutation([0, 7, 3, 1, 9, 5, 11, 6, 10, 4, 13, 2, 12, 8, 14, 15])
    >>> canonicalize(g, dummies, [0, 0], t0, t1)
    [0, 2, 4, 1, 6, 8, 10, 3, 11, 7, 12, 5, 13, 9, 15, 14]
    """
    from sympy.combinatorics.testutil import canonicalize_naive
    
    # 检查 msym 的类型和值
    if not isinstance(msym, list):
        if msym not in (0, 1, None):
            raise ValueError('msym must be 0, 1 or None')
        num_types = 1
    else:
        num_types = len(msym)
        # 检查 msym 列表中的值是否合法
        if not all(msymx in (0, 1, None) for msymx in msym):
            raise ValueError('msym entries must be 0, 1 or None')
        # 检查 dummies 和 msym 的长度是否相等
        if len(dummies) != num_types:
            raise ValueError(
                'dummies and msym must have the same number of elements')
    
    # 获取生成器的大小
    size = g.size
    num_tensors = 0
    v1 = []
    
    # 遍历每个张量组件
    for base_i, gens_i, n_i, sym_i in v:
        # 检查 BSGS 是否是最小的；在 double_coset_can_rep 中使用该属性；
        # 如果不是最小的则使用 canonicalize_naive
        if not _is_minimal_bsgs(base_i, gens_i):
            mbsgs = get_minimal_bsgs(base_i, gens_i)
            if not mbsgs:
                can = canonicalize_naive(g, dummies, msym, *v)
                return can
            base_i, gens_i = mbsgs
        v1.append((base_i, gens_i, [[]] * n_i, sym_i))
        num_tensors += n_i
    
    # 如果只有一个类型且 msym 不是列表，则转换为列表形式
    if num_types == 1 and not isinstance(msym, list):
        dummies = [dummies]
        msym = [msym]
    
    # 将 dummies 展开成一维数组，并检查其有效性
    flat_dummies = []
    for dumx in dummies:
        flat_dummies.extend(dumx)
    if flat_dummies and flat_dummies != list(range(flat_dummies[0], flat_dummies[-1] + 1)):
        raise ValueError('dummies is not valid')
    
    # 计算张量的 slot 对称性
    size1, sbase, sgens = gens_products(*v1)
    if size != size1:
        raise ValueError(
            'g has size %d, generators have size %d' % (size, size1))
    
    # 找到自由指标
    free = [i for i in range(size - 2) if i not in flat_dummies]
    num_free = len(free)
    
    # 在 slot 对称性下找到最小张量 g1
    g1 = canonical_free(sbase, sgens, g, num_free)
    if not flat_dummies:
        return g1
    
    # 保存 g1 的符号
    sign = 0 if g1[-1] == size - 1 else 1

    # 保持自由指标不变
    # 确定 free_i，这是包含固定的张量槽位列表
    # 固定的槽位是被自由指标占据的张量槽位
    start = 0
    for i, (base_i, gens_i, n_i, sym_i) in enumerate(v):
        # 初始化空列表，用于存储自由指标的位置信息
        free_i = []
        # 计算每个分量张量的长度
        len_tens = gens_i[0].size - 2
        # 遍历当前分量张量的固定槽位
        for j in range(n_i):
            # 获取对应分量张量的元素
            h = g1[start:(start + len_tens)]
            fr = []
            # 在 h 中找到固定元素的位置，并存入 fr 列表
            for k in free:
                if k in h:
                    fr.append(h.index(k))
            free_i.append(fr)
            # 更新起始位置，准备处理下一个分量张量
            start += len_tens
        # 更新 v1 中的元组，将 free_i 加入其中
        v1[i] = (base_i, gens_i, free_i, sym_i)

    # 使用 v1 计算张量的 BSGS 分解，处理固定的自由指标
    size, sbase, sgens = gens_products(*v1)

    # 缩减排列，去除自由指标
    pos_free = [g1.index(x) for x in range(num_free)]
    size_red = size - num_free
    g1_red = [x - num_free for x in g1 if x in flat_dummies]
    if sign:
        g1_red.extend([size_red - 1, size_red - 2])
    else:
        g1_red.extend([size_red - 2, size_red - 1])
    # 获取映射表，将新的张量槽位映射到旧的槽位
    map_slots = _get_map_slots(size, pos_free)
    sbase_red = [map_slots[i] for i in sbase if i not in pos_free]
    sgens_red = [_af_new([map_slots[i] for i in y._array_form if i not in pos_free]) for y in sgens]
    dummies_red = [[x - num_free for x in y] for y in dummies]
    # 计算横截集
    transv_red = get_transversals(sbase_red, sgens_red)
    g1_red = _af_new(g1_red)
    # 计算双陪集的代表元素
    g2 = double_coset_can_rep(
        dummies_red, msym, sbase_red, sgens_red, transv_red, g1_red)
    if g2 == 0:
        return 0
    # 将结果升到具有自由指标的情况
    g3 = _lift_sgens(size, pos_free, free, g2)
    return g3
# 定义一个函数，计算两个排列的直积
def perm_af_direct_product(gens1, gens2, signed=True):
    # 将gens1和gens2转换为列表的列表
    gens1 = [list(x) for x in gens1]
    gens2 = [list(x) for x in gens2]
    # 如果signed为True，则设定s为2，否则为0
    s = 2 if signed else 0
    # 计算gens1和gens2的元素个数（不包括符号位）
    n1 = len(gens1[0]) - s
    n2 = len(gens2[0]) - s
    # 创建从0到n1-1和从n1到n1+n2-1的列表
    start = list(range(n1))
    end = list(range(n1, n1 + n2))
    # 如果signed为True，对gens1和gens2进行调整
    if signed:
        gens1 = [gen[:-2] + end + [gen[-2] + n2, gen[-1] + n2]
                 for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]
    else:
        gens1 = [gen + end for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]

    # 将gens1和gens2合并成结果列表res
    res = gens1 + gens2

    # 返回结果列表res
    return res


# 定义一个函数，计算两个BSGS（基数与强生成序列）的直积
def bsgs_direct_product(base1, gens1, base2, gens2, signed=True):
    # 如果signed为True，则设定s为2，否则为0
    s = 2 if signed else 0
    # 计算gens1的元素个数（不包括符号位）
    n1 = gens1[0].size - s
    # 将base1转换为列表
    base = list(base1)
    # 将base2的每个元素加上n1后加入base
    base += [x + n1 for x in base2]
    # 将gens1和gens2转换为数组形式的列表
    gens1 = [h._array_form for h in gens1]
    gens2 = [h._array_form for h in gens2]
    # 调用perm_af_direct_product计算gens1和gens2的直积
    gens = perm_af_direct_product(gens1, gens2, signed)
    # 计算gens的长度
    size = len(gens[0])
    # 创建一个标准的排列
    id_af = list(range(size))
    # 从gens中去除标准排列
    gens = [h for h in gens if h != id_af]
    # 如果gens为空，则用标准排列替代
    if not gens:
        gens = [id_af]
    # 返回base和gens的元素新排列
    return base, [_af_new(h) for h in gens]


# 定义一个函数，返回（反）对称张量的最小BSGS的基数和强生成序列
def get_symmetric_group_sgs(n, antisym=False):
    # 如果n为1，返回一个空列表和一个长度为3的新排列
    if n == 1:
        return [], [_af_new(list(range(3)))]
    # 生成包含n-1个排列的列表gens
    gens = [Permutation(n - 1)(i, i + 1)._array_form for i in range(n - 1)]
    # 如果antisym为0，则在gens的每个元素末尾加上n和n+1
    if antisym == 0:
        gens = [x + [n, n + 1] for x in gens]
    else:
        # 如果不满足条件，则生成新的列表gens，每个元素是原列表x加上两个数字[n+1, n]
        gens = [x + [n + 1, n] for x in gens]
    # 创建一个从0到n-2的列表，作为base
    base = list(range(n - 1))
    # 返回base和gens中每个元素经过_af_new处理后的结果组成的列表
    return base, [_af_new(h) for h in gens]
riemann_bsgs = [0, 2], [Permutation(0, 1)(4, 5), Permutation(2, 3)(4, 5),
                        Permutation(5)(0, 2)(1, 3)]


def get_transversals(base, gens):
    """
    Return transversals for the group with BSGS base, gens
    """
    if not base:
        return []
    # Distribute generators by base and calculate stabilizers
    stabs = _distribute_gens_by_base(base, gens)
    # Compute orbits and corresponding transversals from the BSGS
    orbits, transversals = _orbits_transversals_from_bsgs(base, stabs)
    # Convert transversals to a more readable format
    transversals = [{x: h._array_form for x, h in y.items()} for y in
                    transversals]
    return transversals


def _is_minimal_bsgs(base, gens):
    """
    Check if the BSGS has minimal base under lexicographic order.

    base, gens BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, _is_minimal_bsgs
    >>> _is_minimal_bsgs(*riemann_bsgs)
    True
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> _is_minimal_bsgs(*riemann_bsgs1)
    False
    """
    base1 = []
    sgs1 = gens[:]
    size = gens[0].size
    # Check each position in the base for minimality
    for i in range(size):
        if not all(h._array_form[i] == i for h in sgs1):
            base1.append(i)
            sgs1 = [h for h in sgs1 if h._array_form[i] == i]
    return base1 == base


def get_minimal_bsgs(base, gens):
    """
    Compute a minimal BSGS

    base, gens BSGS

    If base, gens is a minimal BSGS return it; else return None if unable to find one

    TODO: use baseswap if unable to find a minimal BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import get_minimal_bsgs
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> get_minimal_bsgs(*riemann_bsgs1)
    ([0, 2], [(0 1)(4 5), (5)(0 2)(1 3), (2 3)(4 5)])
    """
    # Create a PermutationGroup from generators
    G = PermutationGroup(gens)
    # Compute incremental Schreier-Sims algorithm for base and generators
    base, gens = G.schreier_sims_incremental()
    # Check if the resulting BSGS is minimal
    if not _is_minimal_bsgs(base, gens):
        return None
    return base, gens


def tensor_gens(base, gens, list_free_indices, sym=0):
    """
    Returns size, res_base, res_gens BSGS for n tensors of the
    same type.

    Explanation
    ===========

    base, gens BSGS for tensors of this type
    list_free_indices  list of the slots occupied by fixed indices
                       for each of the tensors

    sym symmetry under commutation of two tensors
    sym   None  no symmetry
    sym   0     commuting
    sym   1     anticommuting

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import tensor_gens, get_symmetric_group_sgs

    two symmetric tensors with 3 indices without free indices

    >>> base, gens = get_symmetric_group_sgs(3)
    >>> tensor_gens(base, gens, [[], []])
    (8, [0, 1, 3, 4], [(7)(0 1), (7)(1 2), (7)(3 4), (7)(4 5), (7)(0 3)(1 4)(2 5)])

    two symmetric tensors with 3 indices with free indices in slot 1 and 0
    """
    # Function implementation would go here, but it's not provided in the snippet
    pass
    # 定义一个函数 _get_bsgs，用于计算生成集 G 中自由指标 free_indices 的基和生成元
    def _get_bsgs(G, base, gens, free_indices):
        """
        返回 G.pointwise_stabilizer(free_indices) 的基本稳定子群链表
        """
        # 如果没有自由指标，则直接返回 base 和 gens 的副本
        if not free_indices:
            return base[:], gens[:]
        else:
            # 否则计算自由指标的点态稳定子群 H
            H = G.pointwise_stabilizer(free_indices)
            # 使用 H 的斯赫莱尔-辛姆斯增量算法获取基和生成元
            base, sgs = H.schreier_sims_incremental()
            return base, sgs

    # 如果 base 为空且 list_free_indices 中空列表的个数小于 2，则没有槽对称性
    if not base and list_free_indices.count([]) < 2:
        # 计算结果的大小 size
        n = len(list_free_indices)
        size = gens[0].size
        size = n * (size - 2) + 2
        # 返回 size、空列表和一个新的 anti-focus 列表
        return size, [], [_af_new(list(range(size)))]

    # 如果 list_free_indices 中有任何非空列表，则需要计算点态稳定子群 G
    if any(list_free_indices):
        G = PermutationGroup(gens)
    else:
        G = None

    # no_free 是一个列表的列表，用于存储没有固定指标的组分张量的指标
    no_free = []
    size = gens[0].size
    id_af = list(range(size))
    num_indices = size - 2
    # 如果第一个 list_free_indices 是空的，则将其索引加入 no_free
    if not list_free_indices[0]:
        no_free.append(list(range(num_indices)))
    # 使用 _get_bsgs 计算 list_free_indices[0] 的基和生成元，初始化结果 base 和 gens
    res_base, res_gens = _get_bsgs(G, base, gens, list_free_indices[0])
    # 遍历剩余的 list_free_indices
    for i in range(1, len(list_free_indices)):
        # 使用 _get_bsgs 计算每个 list_free_indices[i] 的基和生成元
        base1, gens1 = _get_bsgs(G, base, gens, list_free_indices[i])
        # 使用 bsgs_direct_product 将结果与之前的结果进行直积操作
        res_base, res_gens = bsgs_direct_product(res_base, res_gens,
                                                 base1, gens1, 1)
        # 如果 list_free_indices[i] 是空的，则将其索引加入 no_free
        if not list_free_indices[i]:
            no_free.append(list(range(size - 2, size - 2 + num_indices)))
        size += num_indices
    nr = size - 2
    # 从 res_gens 中移除表示单位置换的元素
    res_gens = [h for h in res_gens if h._array_form != id_af]
    
    # 如果 sym 为 None 或者 no_free 为空，则没有对称张量，直接返回结果
    if sym is None or not no_free:
        if not res_gens:
            res_gens = [_af_new(id_af)]
        return size, res_base, res_gens

    # 如果组分张量具有最小的 BSGS，则它们的直积 P 也具有最小的 BSGS，
    # 其槽对称群是 S = P*C，其中 C 是用于（反）对易组分张量的群
    base_comm = []
    for i in range(len(no_free) - 1):
        ind1 = no_free[i]
        ind2 = no_free[i + 1]
        a = list(range(ind1[0]))
        a.extend(ind2)
        a.extend(ind1)
        base_comm.append(ind1[0])
        a.extend(list(range(ind2[-1] + 1, nr)))
        if sym == 0:
            a.extend([nr, nr + 1])
        else:
            a.extend([nr + 1, nr])
        res_gens.append(_af_new(a))
    res_base = list(res_base)
    # 对每个元素进行处理，将不在res_base中的元素添加到res_base中
    for i in base_comm:
        # 检查当前元素是否已存在于res_base中，如果不存在则添加
        if i not in res_base:
            res_base.append(i)
    
    # 对结果集合res_base进行排序
    res_base.sort()
    
    # 如果结果生成器集合res_gens为空，则初始化为一个包含单个_af_new(id_af)结果的列表
    if not res_gens:
        res_gens = [_af_new(id_af)]
    
    # 返回计算得到的size、排序后的res_base和初始化后的res_gens
    return size, res_base, res_gens
# 定义一个函数 gens_products，用于生成多种类型张量的 BSGS（基本稳定子群系统）。

def gens_products(*v):
    """
    返回 n 种不同类型张量的 BSGS 大小、基、生成器。

    说明
    ===========
    
    v 是一个序列，包含了 (base_i, gens_i, free_i, sym_i) 元组，其中：
    base_i, gens_i 是类型为 `i` 的张量的 BSGS
    free_i 是每种类型 `i` 张量的固定插槽列表；如果有 `n_i` 个类型 `i` 的张量，且它们都没有固定插槽，则 `free = [[]]*n_i`
    sym_i 表示类型 `i` 的张量之间是否（反）对易，0 表示对易，1 表示反对易

    示例
    ========

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, gens_products
    >>> base, gens = get_symmetric_group_sgs(2)
    >>> gens_products((base, gens, [[], []], 0))
    (6, [0, 2], [(5)(0 1), (5)(2 3), (5)(0 2)(1 3)])
    >>> gens_products((base, gens, [[1], []], 0))
    (6, [2], [(5)(2 3)])
    """

    # 初始化结果的 BSGS 大小、基和生成器
    res_size, res_base, res_gens = tensor_gens(*v[0])
    
    # 对于每个类型的张量，依次计算其 BSGS 并进行直接积
    for i in range(1, len(v)):
        size, base, gens = tensor_gens(*v[i])
        res_base, res_gens = bsgs_direct_product(res_base, res_gens, base,
                                                 gens, 1)
    
    # 更新结果的 BSGS 大小为第一个生成器的大小
    res_size = res_gens[0].size
    
    # 生成一个标识排列
    id_af = list(range(res_size))
    
    # 筛选出非标识排列作为最终结果的生成器集合
    res_gens = [h for h in res_gens if h != id_af]
    
    # 如果没有非标识排列，则将标识排列作为生成器
    if not res_gens:
        res_gens = [id_af]
    
    # 返回最终的 BSGS 大小、基和生成器
    return res_size, res_base, res_gens
```