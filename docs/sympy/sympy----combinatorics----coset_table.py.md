# `D:\src\scipysrc\sympy\sympy\combinatorics\coset_table.py`

```
from sympy.combinatorics.free_groups import free_group  # 导入自由群的自由生成元函数
from sympy.printing.defaults import DefaultPrinting  # 导入默认打印设置

from itertools import chain, product  # 导入链和笛卡尔积函数
from bisect import bisect_left  # 导入二分查找函数

###############################################################################
#                           COSET TABLE                                       #
###############################################################################

class CosetTable(DefaultPrinting):
    """
    coset_table: 数学上的一个余陪表，用列表的列表表示
                 alpha: 数学上的一个余陪（精确地说，是一个活跃的余陪）
                        用一个介于1到n之间的整数表示
                        alpha属于c
                 x: 数学上的"A"中的一个元素（生成元及其逆元的集合），
                    用"FpGroupElement"表示
                 fp_grp: 以<X|R>为展示的有限呈现群
                 H: fp_grp的子群
                 注意：我们将H初始设定为仅仅是fp_grp生成元词语的列表。
                      因为“subgroup”方法尚未实现。
    """

    r"""

    Properties
    ==========

    [1] `0 \in \Omega` and `\tau(1) = \epsilon`
    [2] `\alpha^x = \beta \Leftrightarrow \beta^{x^{-1}} = \alpha`
    [3] If `\alpha^x = \beta`, then `H \tau(\alpha)x = H \tau(\beta)`
    [4] `\forall \alpha \in \Omega, 1^{\tau(\alpha)} = \alpha`

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"

    .. [2] John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
           Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490.
           "Implementation and Analysis of the Todd-Coxeter Algorithm"

    """
    # 余陪枚举中允许的余陪数量的默认限制。
    coset_table_max_limit = 4096000
    # 当前实例的余陪表限制
    coset_table_limit = None
    # 被清空的推断栈的最大大小
    max_stack_size = 100
    def __init__(self, fp_grp, subgroup, max_cosets=None):
        # 如果未提供 max_cosets 参数，则使用默认值 CosetTable.coset_table_max_limit
        if not max_cosets:
            max_cosets = CosetTable.coset_table_max_limit
        # 设置 fp_group 和 subgroup 属性
        self.fp_group = fp_grp
        self.subgroup = subgroup
        # 设置 coset_table_limit 属性为 max_cosets
        self.coset_table_limit = max_cosets
        # 初始化空列表 p，起始元素为 0
        self.p = [0]
        # 初始化列表 A，其中包含生成元及其逆元的扁平化列表
        self.A = list(chain.from_iterable((gen, gen**-1) \
                for gen in self.fp_group.generators))
        # 初始化二维列表 P，每个元素初始为 None，长度为 A 列表的长度
        self.P = [[None]*len(self.A)]
        # 初始化二维列表 table，每个元素初始为 None，长度为 A 列表的长度
        self.table = [[None]*len(self.A)]
        # 创建 A_dict 字典，将 A 列表中的元素映射到其索引
        self.A_dict = {x: self.A.index(x) for x in self.A}
        # 创建 A_dict_inv 字典，根据 A_dict 中的索引创建逆映射
        self.A_dict_inv = {}
        for x, index in self.A_dict.items():
            if index % 2 == 0:
                self.A_dict_inv[x] = self.A_dict[x] + 1
            else:
                self.A_dict_inv[x] = self.A_dict[x] - 1
        # 初始化 deduction_stack，用于存储推导过程中的元素
        self.deduction_stack = []
        # 根据 subgroup 属性，创建修改后的方法属性 _grp
        H = self.subgroup
        self._grp = free_group(', ' .join(["a_%d" % i for i in range(len(H))]))[0]
        # 再次初始化 P，这可能是重复代码，需要注意
        self.P = [[None]*len(self.A)]
        # 初始化 p_p 字典，可能用于存储特定属性
        self.p_p = {}

    @property
    def omega(self):
        """Set of live cosets. """
        # 返回 p 列表中值等于其索引的所有元素，形成 live cosets 集合
        return [coset for coset in range(len(self.p)) if self.p[coset] == coset]

    def copy(self):
        """
        Return a shallow copy of Coset Table instance ``self``.

        """
        # 创建当前对象的浅复制 self_copy
        self_copy = self.__class__(self.fp_group, self.subgroup)
        # 复制 table 列表的内容到 self_copy.table
        self_copy.table = [list(perm_rep) for perm_rep in self.table]
        # 复制 p 列表的内容到 self_copy.p
        self_copy.p = list(self.p)
        # 复制 deduction_stack 列表的内容到 self_copy.deduction_stack
        self_copy.deduction_stack = list(self.deduction_stack)
        return self_copy

    def __str__(self):
        # 返回对象的描述字符串，包括 fp_group 和 subgroup 的信息
        return "Coset Table on %s with %s as subgroup generators" \
                % (self.fp_group, self.subgroup)

    __repr__ = __str__

    @property
    def n(self):
        """The number `n` represents the length of the sublist containing the
        live cosets.

        """
        # 如果 table 为空，则返回 0
        if not self.table:
            return 0
        # 返回 live cosets 中最大值加 1，即为 n 的值
        return max(self.omega) + 1

    # Pg. 152 [1]
    def is_complete(self):
        r"""
        The coset table is called complete if it has no undefined entries
        on the live cosets; that is, `\alpha^x` is defined for all
        `\alpha \in \Omega` and `x \in A`.

        """
        # 检查 live cosets 中是否有未定义的条目，如果有，则返回 False；否则返回 True
        return not any(None in self.table[coset] for coset in self.omega)

    # Pg. 153 [1]
    def define(self, alpha, x, modified=False):
        r"""
        This routine is used in the relator-based strategy of Todd-Coxeter
        algorithm if some `\alpha^x` is undefined. We check whether there is
        space available for defining a new coset. If there is enough space
        then we remedy this by adjoining a new coset `\beta` to `\Omega`
        (i.e to set of live cosets) and put that equal to `\alpha^x`, then
        make an assignment satisfying Property[1]. If there is not enough space
        then we halt the Coset Table creation. The maximum amount of space that
        can be used by Coset Table can be manipulated using the class variable
        ``CosetTable.coset_table_max_limit``.

        See Also
        ========

        define_c

        """
        # 获取当前对象的A属性（集合）
        A = self.A
        # 获取当前对象的table属性（表）
        table = self.table
        # 获取当前表的长度
        len_table = len(table)
        # 检查是否已经达到了允许的最大余陪表大小
        if len_table >= self.coset_table_limit:
            # 如果超过了最大值，则抛出异常终止余陪表的进一步生成
            raise ValueError("the coset enumeration has defined more than "
                    "%s cosets. Try with a greater value max number of cosets "
                    % self.coset_table_limit)
        # 在表中添加新的行，初始化为None
        table.append([None]*len(A))
        # 将P中添加新的行，初始化为None
        self.P.append([None]*len(self.A))
        # beta是生成的新余陪
        beta = len_table
        # 将beta添加到p中
        self.p.append(beta)
        # 将alpha^x设置为beta
        table[alpha][self.A_dict[x]] = beta
        # 将beta^(x^-1)设置为alpha
        table[beta][self.A_dict_inv[x]] = alpha
        # 如果modified标志为True，则进一步设置属性P[alpha][x]和P[beta][x^-1]
        if modified:
            self.P[alpha][self.A_dict[x]] = self._grp.identity
            self.P[beta][self.A_dict_inv[x]] = self._grp.identity
            self.p_p[beta] = self._grp.identity

    def define_c(self, alpha, x):
        r"""
        A variation of ``define`` routine, described on Pg. 165 [1], used in
        the coset table-based strategy of Todd-Coxeter algorithm. It differs
        from ``define`` routine in that for each definition it also adds the
        tuple `(\alpha, x)` to the deduction stack.

        See Also
        ========

        define

        """
        # 获取当前对象的A属性（集合）
        A = self.A
        # 获取当前对象的table属性（表）
        table = self.table
        # 获取当前表的长度
        len_table = len(table)
        # 检查是否已经达到了允许的最大余陪表大小
        if len_table >= self.coset_table_limit:
            # 如果超过了最大值，则抛出异常终止余陪表的进一步生成
            raise ValueError("the coset enumeration has defined more than "
                    "%s cosets. Try with a greater value max number of cosets "
                    % self.coset_table_limit)
        # 在表中添加新的行，初始化为None
        table.append([None]*len(A))
        # beta是生成的新余陪
        beta = len_table
        # 将beta添加到p中
        self.p.append(beta)
        # 将alpha^x设置为beta
        table[alpha][self.A_dict[x]] = beta
        # 将beta^(x^-1)设置为alpha
        table[beta][self.A_dict_inv[x]] = alpha
        # 将(alpha, x)元组添加到推导栈中
        self.deduction_stack.append((alpha, x))
    # 定义一个名为 scan_c 的方法，用于执行扫描操作
    def scan_c(self, alpha, word):
        """
        A variation of ``scan`` routine, described on pg. 165 of [1], which
        puts at tuple, whenever a deduction occurs, to deduction stack.

        See Also
        ========

        scan, scan_check, scan_and_fill, scan_and_fill_c

        """
        # alpha 是一个整数，表示一个“余类”
        # 由于扫描可以出现两种情况：
        # 1. 当 alpha=0 且 w 在 Y 中（即 H 的生成集合）
        # 2. 当 alpha 在 Omega 中（余类的集合），w 在 R 中（关系式）
        
        # 获取类中保存的 A_dict 和 A_dict_inv 字典
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        # 获取类中保存的 table 属性
        table = self.table
        # 初始化变量 f 为 alpha
        f = alpha
        # 初始化变量 i 为 0
        i = 0
        # 获取变量 r 为 word 的长度
        r = len(word)
        # 初始化变量 b 为 alpha
        b = alpha
        # 初始化变量 j 为 r - 1
        j = r - 1
        
        # 当 i 小于等于 j 并且 table[f][A_dict[word[i]]] 不为 None 时
        while i <= j and table[f][A_dict[word[i]]] is not None:
            # 更新 f 为 table[f][A_dict[word[i]]]
            f = table[f][A_dict[word[i]]]
            # 更新 i 为 i + 1
            i += 1
        
        # 如果 i 大于 j
        if i > j:
            # 如果 f 不等于 b，则调用 coincidence_c 方法
            if f != b:
                self.coincidence_c(f, b)
            return
        
        # 当 j 大于等于 i 且 table[b][A_dict_inv[word[j]]] 不为 None 时
        while j >= i and table[b][A_dict_inv[word[j]]] is not None:
            # 更新 b 为 table[b][A_dict_inv[word[j]]]
            b = table[b][A_dict_inv[word[j]]]
            # 更新 j 为 j - 1
            j -= 1
        
        # 如果 j 小于 i
        if j < i:
            # 输出表示 f ~ b 的错误完成扫描
            # 运行“coincidence”例程
            self.coincidence_c(f, b)
        # 如果 j 等于 i
        elif j == i:
            # 进行推导过程
            table[f][A_dict[word[i]]] = b
            table[b][A_dict_inv[word[i]]] = f
            # 将推导元组 (f, word[i]) 添加到推导堆栈中
            self.deduction_stack.append((f, word[i]))
        # 否则扫描不完整，没有产生信息

    # alpha, beta coincide, i.e. alpha, beta represent the pair of cosets where
    # coincidence occurs
    # 在余类表格枚举方法中，变体的“coincidence”例程，用于添加新的余类到余类表中时，将其追加到“deduction_stack”中
    def coincidence_c(self, alpha, beta):
        """
        A variation of ``coincidence`` routine used in the coset-table based
        method of coset enumeration. The only difference being on addition of
        a new coset in coset table(i.e new coset introduction), then it is
        appended to ``deduction_stack``.

        See Also
        ========

        coincidence

        """
        A_dict = self.A_dict  # 获取 A 字典
        A_dict_inv = self.A_dict_inv  # 获取 A 逆字典
        table = self.table  # 获取表格
        # behaves as a queue
        q = []  # 初始化队列 q
        self.merge(alpha, beta, q)  # 调用 merge 方法初始化队列
        while len(q) > 0:
            gamma = q.pop(0)  # 从队列中取出下一个 gamma
            for x in A_dict:
                delta = table[gamma][A_dict[x]]  # 获取表格中对应的 delta
                if delta is not None:
                    table[delta][A_dict_inv[x]] = None  # 清空对应的表格项
                    # only line of difference from ``coincidence`` routine
                    self.deduction_stack.append((delta, x**-1))  # 将 (delta, x^-1) 加入 deduction_stack
                    mu = self.rep(gamma)  # 获取 gamma 的代表元 mu
                    nu = self.rep(delta)  # 获取 delta 的代表元 nu
                    if table[mu][A_dict[x]] is not None:
                        self.merge(nu, table[mu][A_dict[x]], q)  # 调用 merge 方法更新队列
                    elif table[nu][A_dict_inv[x]] is not None:
                        self.merge(mu, table[nu][A_dict_inv[x]], q)  # 调用 merge 方法更新队列
                    else:
                        table[mu][A_dict[x]] = nu  # 更新表格
                        table[nu][A_dict_inv[x]] = mu  # 更新表格

    # 在低指数子群算法中使用
    def scan_check(self, alpha, word):
        r"""
        Another version of ``scan`` routine, described on, it checks whether
        `\alpha` scans correctly under `word`, it is a straightforward
        modification of ``scan``. ``scan_check`` returns ``False`` (rather than
        calling ``coincidence``) if the scan completes incorrectly; otherwise
        it returns ``True``.

        See Also
        ========

        scan, scan_c, scan_and_fill, scan_and_fill_c

        """
        # alpha is an integer representing a "coset"
        # since scanning can be in two cases
        # 1. for alpha=0 and w in Y (i.e generating set of H)
        # 2. alpha in Omega (set of live cosets), w in R (relators)
        A_dict = self.A_dict  # 获取 A 字典
        A_dict_inv = self.A_dict_inv  # 获取 A 逆字典
        table = self.table  # 获取表格
        f = alpha  # 设置 f 为 alpha
        i = 0  # 初始化 i
        r = len(word)  # 获取 word 的长度
        b = alpha  # 设置 b 为 alpha
        j = r - 1  # 设置 j 为 word 的最后一个索引
        while i <= j and table[f][A_dict[word[i]]] is not None:
            f = table[f][A_dict[word[i]]]  # 更新 f
            i += 1  # 更新 i
        if i > j:
            return f == b  # 返回是否 f 等于 b
        while j >= i and table[b][A_dict_inv[word[j]]] is not None:
            b = table[b][A_dict_inv[word[j]]]  # 更新 b
            j -= 1  # 更新 j
        if j < i:
            # we have an incorrect completed scan with coincidence f ~ b
            # return False, instead of calling coincidence routine
            return False  # 返回扫描完成错误
        elif j == i:
            # deduction process
            table[f][A_dict[word[i]]] = b  # 更新表格
            table[b][A_dict_inv[word[i]]] = f  # 更新表格
        return True  # 返回扫描正确
    def merge(self, k, lamda, q, w=None, modified=False):
        """
        Merge two classes with representatives ``k`` and ``lamda``, described
        on Pg. 157 [1] (for pseudocode), start by putting ``p[k] = lamda``.
        It is more efficient to choose the new representative from the larger
        of the two classes being merged, i.e larger among ``k`` and ``lamda``.
        procedure ``merge`` performs the merging operation, adds the deleted
        class representative to the queue ``q``.

        Parameters
        ==========

        'k', 'lamda' being the two class representatives to be merged.

        Notes
        =====

        Pg. 86-87 [1] contains a description of this method.

        See Also
        ========

        coincidence, rep

        """
        # 获取当前对象的属性 self.p，即类别的代表元素字典
        p = self.p
        # 获取当前对象的属性 self.rep，即类别的代表元素计算函数
        rep = self.rep
        # 调用 rep 函数，计算类别 k 的代表元素 phi
        phi = rep(k, modified=modified)
        # 调用 rep 函数，计算类别 lamda 的代表元素 psi
        psi = rep(lamda, modified=modified)
        # 如果 phi 和 psi 不相等，执行合并操作
        if phi != psi:
            # 取 phi 和 psi 中较小的值作为 mu
            mu = min(phi, psi)
            # 取 phi 和 psi 中较大的值作为 v
            v = max(phi, psi)
            # 将 p[v] 设为 mu，即将类别 v 的代表元素设为 mu
            p[v] = mu
            # 如果设置了 modified 参数
            if modified:
                # 根据不同情况更新 self.p_p 中的值
                if v == phi:
                    self.p_p[phi] = self.p_p[k]**-1*w*self.p_p[lamda]
                else:
                    self.p_p[psi] = self.p_p[lamda]**-1*w**-1*self.p_p[k]
            # 将 v 添加到队列 q 中，表示类别 v 的代表元素被合并删除
            q.append(v)
    def rep(self, k, modified=False):
        r"""
        Parameters
        ==========

        `k \in [0 \ldots n-1]`, as for ``self`` only array ``p`` is used
        参数：
            `k \in [0 \ldots n-1]`，对于 ``self``，仅使用数组 ``p``

        Returns
        =======

        Representative of the class containing ``k``.
        返回：
            包含 ``k`` 的等价类的代表。

        Returns the representative of `\sim` class containing ``k``, it also
        makes some modification to array ``p`` of ``self`` to ease further
        computations, described on Pg. 157 [1].
        返回包含 ``k`` 的 `\sim` 类的代表，并对 ``self`` 的数组 ``p`` 进行一些修改，
        以便进一步计算，详见第157页 [1]。

        The information on classes under `\sim` is stored in array `p` of
        ``self`` argument, which will always satisfy the property:

        `p[\alpha] \sim \alpha` and `p[\alpha]=\alpha \iff \alpha=rep(\alpha)`
        `\forall \in [0 \ldots n-1]`.
        `\sim` 下的类的信息存储在 ``self`` 参数的数组 `p` 中，其总是满足以下性质：

        `p[\alpha] \sim \alpha` 和 `p[\alpha]=\alpha \iff \alpha=rep(\alpha)`
        对于所有的 $\alpha \in [0 \ldots n-1]$。

        So, for `\alpha \in [0 \ldots n-1]`, we find `rep(self, \alpha)` by
        continually replacing `\alpha` by `p[\alpha]` until it becomes
        constant (i.e satisfies `p[\alpha] = \alpha`):
        因此，对于 `\alpha \in [0 \ldots n-1]`，我们通过不断用 `p[\alpha]` 替换 `\alpha`，
        直到它变成常数（即满足 `p[\alpha] = \alpha`）来找到 `rep(self, \alpha)`。

        To increase the efficiency of later ``rep`` calculations, whenever we
        find `rep(self, \alpha)=\beta`, we set
        `p[\gamma] = \beta \forall \gamma \in p-chain` from `\alpha` to `\beta`
        为了增加后续 ``rep`` 计算的效率，每当我们找到 `rep(self, \alpha)=\beta` 时，
        我们设置 `p[\gamma] = \beta \forall \gamma \in p-chain`，其中 `p-chain` 是从 `\alpha` 到 `\beta` 的链条。

        Notes
        =====

        ``rep`` routine is also described on Pg. 85-87 [1] in Atkinson's
        algorithm, this results from the fact that ``coincidence`` routine
        introduces functionality similar to that introduced by the
        ``minimal_block`` routine on Pg. 85-87 [1].
        注：
            ``rep`` 程序在第85-87页 [1] 中也有描述，这源于 ``coincidence`` 程序引入了类似于第85-87页 [1] 中 ``minimal_block`` 程序引入的功能。

        See Also
        ========

        coincidence, merge
        参见：
            coincidence, merge

        """
        p = self.p  # 获取数组 `p`
        lamda = k   # 初始化 lamda 为 k
        rho = p[lamda]  # 初始化 rho 为 p[lamda]
        if modified:
            s = p[:]  # 如果 modified 为 True，则复制数组 `p` 到 s
        while rho != lamda:
            if modified:
                s[rho] = lamda  # 如果 modified 为 True，则设置 s[rho] = lamda
            lamda = rho  # 更新 lamda 为 rho
            rho = p[lamda]  # 更新 rho 为 p[lamda]
        if modified:
            rho = s[lamda]  # 如果 modified 为 True，则将 rho 更新为 s[lamda]
            while rho != k:
                mu = rho  # 设置 mu 为 rho
                rho = s[mu]  # 更新 rho 为 s[mu]
                p[rho] = lamda  # 设置 p[rho] = lamda
                self.p_p[rho] = self.p_p[rho]*self.p_p[mu]  # 更新 self.p_p[rho]
        else:
            mu = k  # 设置 mu 为 k
            rho = p[mu]  # 设置 rho 为 p[mu]
            while rho != lamda:
                p[mu] = lamda  # 设置 p[mu] = lamda
                mu = rho  # 更新 mu 为 rho
                rho = p[mu]  # 更新 rho 为 p[mu]
        return lamda  # 返回 lamda
    def coincidence(self, alpha, beta, w=None, modified=False):
        r"""
        The third situation described in ``scan`` routine is handled by this
        routine, described on Pg. 156-161 [1].

        The unfortunate situation when the scan completes but not correctly,
        then ``coincidence`` routine is run. i.e when for some `i` with
        `1 \le i \le r+1`, we have `w=st` with `s = x_1 x_2 \dots x_{i-1}`,
        `t = x_i x_{i+1} \dots x_r`, and `\beta = \alpha^s` and
        `\gamma = \alpha^{t-1}` are defined but unequal. This means that
        `\beta` and `\gamma` represent the same coset of `H` in `G`. Described
        on Pg. 156 [1]. ``rep``

        See Also
        ========

        scan

        """
        # 获取当前对象中的 A_dict 属性，这是一个字典
        A_dict = self.A_dict
        # 获取当前对象中的 A_dict_inv 属性，这也是一个字典
        A_dict_inv = self.A_dict_inv
        # 获取当前对象中的 table 属性，这可能是一个表格或其他数据结构
        table = self.table
        # 创建一个空列表 q，作为队列使用
        q = []
        # 如果 modified 参数为 True，则调用 modified_merge 方法，否则调用 merge 方法
        if modified:
            self.modified_merge(alpha, beta, w, q)
        else:
            self.merge(alpha, beta, q)
        # 当队列 q 非空时执行循环
        while len(q) > 0:
            # 从队列头部取出一个元素作为 gamma
            gamma = q.pop(0)
            # 对 A_dict 中的每个元素 x 执行循环
            for x in A_dict:
                # 查找 table 中 gamma 对应的 A_dict[x] 位置上的值，保存为 delta
                delta = table[gamma][A_dict[x]]
                # 如果 delta 不为 None
                if delta is not None:
                    # 将 table 中 delta 对应的 A_dict_inv[x] 位置置为 None
                    table[delta][A_dict_inv[x]] = None
                    # 分别计算 gamma 和 delta 的代表元 mu 和 nu
                    mu = self.rep(gamma, modified=modified)
                    nu = self.rep(delta, modified=modified)
                    # 如果 table 中 mu 对应的 A_dict[x] 位置不为 None
                    if table[mu][A_dict[x]] is not None:
                        # 如果 modified 参数为 True
                        if modified:
                            # 计算 v 的值
                            v = self.p_p[delta]**-1*self.P[gamma][self.A_dict[x]]**-1
                            v = v*self.p_p[gamma]*self.P[mu][self.A_dict[x]]
                            # 调用 modified_merge 方法
                            self.modified_merge(nu, table[mu][self.A_dict[x]], v, q)
                        else:
                            # 否则调用 merge 方法
                            self.merge(nu, table[mu][A_dict[x]], q)
                    # 如果 table 中 nu 对应的 A_dict_inv[x] 位置不为 None
                    elif table[nu][A_dict_inv[x]] is not None:
                        # 如果 modified 参数为 True
                        if modified:
                            # 计算 v 的值
                            v = self.p_p[gamma]**-1*self.P[gamma][self.A_dict[x]]
                            v = v*self.p_p[delta]*self.P[mu][self.A_dict_inv[x]]
                            # 调用 modified_merge 方法
                            self.modified_merge(mu, table[nu][self.A_dict_inv[x]], v, q)
                        else:
                            # 否则调用 merge 方法
                            self.merge(mu, table[nu][A_dict_inv[x]], q)
                    else:
                        # 将 table 中 mu 对应的 A_dict[x] 位置置为 nu
                        table[mu][A_dict[x]] = nu
                        # 将 table 中 nu 对应的 A_dict_inv[x] 位置置为 mu
                        table[nu][A_dict_inv[x]] = mu
                        # 如果 modified 参数为 True
                        if modified:
                            # 计算 v 的值
                            v = self.p_p[gamma]**-1*self.P[gamma][self.A_dict[x]]*self.p_p[delta]
                            # 将 P 中 mu 对应的 A_dict[x] 位置置为 v
                            self.P[mu][self.A_dict[x]] = v
                            # 将 P 中 nu 对应的 A_dict_inv[x] 位置置为 v 的逆
                            self.P[nu][self.A_dict_inv[x]] = v**-1

    # 用于 HLT 策略中的方法
    # A modified version of the `scan` routine used in coset enumeration,
    # designed to ensure completion of scanning by filling gaps in relators
    # or subgroup generators.

    def scan_and_fill(self, alpha, word):
        """
        A modified version of ``scan`` routine used in the relator-based
        method of coset enumeration, described on pg. 162-163 [1], which
        follows the idea that whenever the procedure is called and the scan
        is incomplete then it makes new definitions to enable the scan to
        complete; i.e it fills in the gaps in the scan of the relator or
        subgroup generator.
        """
        self.scan(alpha, word, fill=True)

    def scan_and_fill_c(self, alpha, word):
        """
        A modified version of ``scan`` routine, described on Pg. 165 second
        para. [1], with modification similar to that of ``scan_anf_fill`` the
        only difference being it calls the coincidence procedure used in the
        coset-table based method i.e. the routine ``coincidence_c`` is used.

        See Also
        ========

        scan, scan_and_fill
        """
        # Initialize variables from instance attributes
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        r = len(word)
        f = alpha
        i = 0
        b = alpha
        j = r - 1

        # loop until it has filled the alpha row in the table.
        while True:
            # do the forward scanning
            while i <= j and table[f][A_dict[word[i]]] is not None:
                f = table[f][A_dict[word[i]]]
                i += 1
            if i > j:
                # If forward scan completed, check for coincidence
                if f != b:
                    self.coincidence_c(f, b)
                return
            # forward scan was incomplete, scan backwards
            while j >= i and table[b][A_dict_inv[word[j]]] is not None:
                b = table[b][A_dict_inv[word[j]]]
                j -= 1
            if j < i:
                self.coincidence_c(f, b)
            elif j == i:
                # Define entries in the table and push to deduction stack
                table[f][A_dict[word[i]]] = b
                table[b][A_dict_inv[word[i]]] = f
                self.deduction_stack.append((f, word[i]))
            else:
                # Define entry in the table
                self.define_c(f, word[i])

    # method used in the HLT strategy
    def look_ahead(self):
        """
        When combined with the HLT method this is known as HLT+Lookahead
        method of coset enumeration, described on pg. 164 [1]. Whenever
        ``define`` aborts due to lack of space available this procedure is
        executed. This routine helps in recovering space resulting from
        "coincidence" of cosets.
        """
        R = self.fp_group.relators
        p = self.p
        # complete scan all relators under all cosets(obviously live)
        # without making new definitions
        for beta in self.omega:
            for w in R:
                self.scan(beta, w)
                if p[beta] < beta:
                    break
    def process_deductions(self, R_c_x, R_c_x_inv):
        """
        Processes the deductions that have been pushed onto ``deduction_stack``,
        described on Pg. 166 [1] and is used in coset-table based enumeration.

        See Also
        ========

        deduction_stack

        """
        p = self.p  # 获取类实例的属性 self.p，表示群的生成器
        table = self.table  # 获取类实例的属性 self.table，表示余类表
        while len(self.deduction_stack) > 0:  # 当 deduction_stack 不为空时循环执行
            if len(self.deduction_stack) >= CosetTable.max_stack_size:  # 如果 deduction_stack 的长度超过最大堆栈大小
                self.look_ahead()  # 调用类的方法 look_ahead，处理后续推导
                del self.deduction_stack[:]  # 清空 deduction_stack 列表
                continue  # 继续下一次循环
            else:
                alpha, x = self.deduction_stack.pop()  # 从 deduction_stack 中弹出元素 alpha 和 x
                if p[alpha] == alpha:  # 如果 p[alpha] 等于 alpha
                    for w in R_c_x:  # 遍历 R_c_x 中的元素 w
                        self.scan_c(alpha, w)  # 调用类的方法 scan_c，扫描并处理 alpha 的余类
                        if p[alpha] < alpha:  # 如果 p[alpha] 小于 alpha
                            break  # 跳出 for 循环
            beta = table[alpha][self.A_dict[x]]  # 获取表 table 中 alpha 行、A_dict[x] 列的元素 beta
            if beta is not None and p[beta] == beta:  # 如果 beta 不为 None 且 p[beta] 等于 beta
                for w in R_c_x_inv:  # 遍历 R_c_x_inv 中的元素 w
                    self.scan_c(beta, w)  # 调用类的方法 scan_c，扫描并处理 beta 的余类
                    if p[beta] < beta:  # 如果 p[beta] 小于 beta
                        break  # 跳出 for 循环

    def process_deductions_check(self, R_c_x, R_c_x_inv):
        """
        A variation of ``process_deductions``, this calls ``scan_check``
        wherever ``process_deductions`` calls ``scan``, described on Pg. [1].

        See Also
        ========

        process_deductions

        """
        table = self.table  # 获取类实例的属性 self.table，表示余类表
        while len(self.deduction_stack) > 0:  # 当 deduction_stack 不为空时循环执行
            alpha, x = self.deduction_stack.pop()  # 从 deduction_stack 中弹出元素 alpha 和 x
            if not all(self.scan_check(alpha, w) for w in R_c_x):  # 如果 R_c_x 中的所有元素对于 alpha 都通过 scan_check 方法
                return False  # 返回 False
            beta = table[alpha][self.A_dict[x]]  # 获取表 table 中 alpha 行、A_dict[x] 列的元素 beta
            if beta is not None:
                if not all(self.scan_check(beta, w) for w in R_c_x_inv):  # 如果 R_c_x_inv 中的所有元素对于 beta 都通过 scan_check 方法
                    return False  # 返回 False
        return True  # 返回 True

    def switch(self, beta, gamma):
        r"""Switch the elements `\beta, \gamma \in \Omega` of ``self``, used
        by the ``standardize`` procedure, described on Pg. 167 [1].

        See Also
        ========

        standardize

        """
        A = self.A  # 获取类实例的属性 self.A，表示一个集合
        A_dict = self.A_dict  # 获取类实例的属性 self.A_dict，表示 A 中元素到索引的映射
        table = self.table  # 获取类实例的属性 self.table，表示余类表
        for x in A:  # 遍历集合 A 中的元素 x
            z = table[gamma][A_dict[x]]  # 获取表 table 中 gamma 行、A_dict[x] 列的元素 z
            table[gamma][A_dict[x]] = table[beta][A_dict[x]]  # 将表 table 中 beta 行、A_dict[x] 列的元素复制到 gamma 行、A_dict[x] 列
            table[beta][A_dict[x]] = z  # 将 z 赋值给表 table 中 beta 行、A_dict[x] 列的元素
            for alpha in range(len(self.p)):  # 遍历类实例属性 self.p 的长度范围
                if self.p[alpha] == alpha:  # 如果 self.p[alpha] 等于 alpha
                    if table[alpha][A_dict[x]] == beta:  # 如果表 table 中 alpha 行、A_dict[x] 列的元素为 beta
                        table[alpha][A_dict[x]] = gamma  # 将 gamma 赋值给表 table 中 alpha 行、A_dict[x] 列的元素
                    elif table[alpha][A_dict[x]] == gamma:  # 如果表 table 中 alpha 行、A_dict[x] 列的元素为 gamma
                        table[alpha][A_dict[x]] = beta  # 将 beta 赋值给表 table 中 alpha 行、A_dict[x] 列的元素
    # 标准化余类表格
    def standardize(self):
        r"""
        如果在余类表中按顺序浏览余类，并且在每个余类中通过生成器映射时（忽略生成器的逆映射），余类按照整数 `0, 1, \dots, n` 的顺序出现，则余类表被称为标准化的。
        "Standardize" 重新排列 `\Omega` 的元素，以便通过首先按 `\Omega` 元素然后按 A 元素的顺序扫描余类表，余类按升序出现。
        ``standardize()`` 在枚举结束时用于对余类进行排列，使它们以某种标准顺序出现。

        Notes
        =====

        过程在第 167-168 页[1]描述，还利用了 ``switch`` 程序用更小的整数值替换。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
        >>> F, x, y = free_group("x, y")

        # 来自 [1] 的示例 5.3
        >>> f = FpGroup(F, [x**2*y**2, x**3*y**5])
        >>> C = coset_enumeration_r(f, [])
        >>> C.compress()
        >>> C.table
        [[1, 3, 1, 3], [2, 0, 2, 0], [3, 1, 3, 1], [0, 2, 0, 2]]
        >>> C.standardize()
        >>> C.table
        [[1, 2, 1, 2], [3, 0, 3, 0], [0, 3, 0, 3], [2, 1, 2, 1]]

        """
        A = self.A  # 获取生成器集合 A
        A_dict = self.A_dict  # 获取生成器到索引的映射字典 A_dict
        gamma = 1  # 初始化 gamma 为 1
        # 遍历元素 alpha 属于范围 [0, n) 和 A 中的生成器 x 的笛卡尔积
        for alpha, x in product(range(self.n), A):
            beta = self.table[alpha][A_dict[x]]  # 获取余类表中 alpha 行 A_dict[x] 列的值
            if beta >= gamma:  # 如果 beta 大于等于 gamma
                if beta > gamma:  # 如果 beta 大于 gamma
                    self.switch(gamma, beta)  # 调用 switch 方法，交换 gamma 和 beta
                gamma += 1  # gamma 自增 1
                if gamma == self.n:  # 如果 gamma 等于 n
                    return  # 返回

    # 压缩余类表
    def compress(self):
        """移除余类表中的非活跃余类，详见第 167 页 [1]。

        """
        gamma = -1  # 初始化 gamma 为 -1
        A = self.A  # 获取生成器集合 A
        A_dict = self.A_dict  # 获取生成器到索引的映射字典 A_dict
        A_dict_inv = self.A_dict_inv  # 获取生成器到逆映射索引的映射字典 A_dict_inv
        table = self.table  # 获取余类表
        chi = tuple([i for i in range(len(self.p)) if self.p[i] != i])  # 创建 chi 元组，包含 self.p 中非标准位置的索引
        for alpha in self.omega:  # 遍历 omega 中的元素 alpha
            gamma += 1  # gamma 自增 1
            if gamma != alpha:  # 如果 gamma 不等于 alpha
                # 在余类表中用 gamma 替换 alpha
                for x in A:  # 遍历生成器集合 A
                    beta = table[alpha][A_dict[x]]  # 获取余类表中 alpha 行 A_dict[x] 列的值
                    table[gamma][A_dict[x]] = beta  # 将 beta 放入表格中 gamma 行 A_dict[x] 列
                    table[beta][A_dict_inv[x]] == gamma  # 将 gamma 放入表格中 beta 行 A_dict_inv[x] 列
        # 所有表格中的余类都是活跃余类
        self.p = list(range(gamma + 1))  # 重新定义 self.p 为 gamma + 1 的列表
        # 删除无用列
        del table[len(self.p):]  # 删除表格中长度大于 gamma + 1 的部分
        # 重新定义值
        for row in table:  # 遍历表格中的每一行
            for j in range(len(self.A)):  # 遍历生成器集合 A 的长度
                row[j] -= bisect_left(chi, row[j])  # row[j] 减去 chi 中比 row[j] 小的第一个位置的索引
    def conjugates(self, R):
        # 将给定的关系列表 R 中的每个关系 rel 进行循环共轭处理，并将逆关系也加入列表
        R_c = list(chain.from_iterable((rel.cyclic_conjugates(), \
                (rel**-1).cyclic_conjugates()) for rel in R))
        # 初始化一个空集合 R_set 来存储所有循环共轭类
        R_set = set()
        # 将所有循环共轭类中的元素合并到 R_set 中
        for conjugate in R_c:
            R_set = R_set.union(conjugate)
        # 初始化一个空列表 R_c_list 来存储生成的循环共轭类列表
        R_c_list = []
        # 对于自由生成元素集合 self.A 中的每个元素 x
        for x in self.A:
            # 找出以 x 开头的所有循环共轭类的子集，存入 r 中
            r = {word for word in R_set if word[0] == x}
            # 将 r 添加到 R_c_list 中
            R_c_list.append(r)
            # 从 R_set 中移除已处理的 r
            R_set.difference_update(r)
        # 返回生成的循环共轭类列表 R_c_list
        return R_c_list

    def coset_representative(self, coset):
        '''
        计算给定陪集的陪集代表元。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
        >>> F, x, y = free_group("x, y")
        >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
        >>> C = coset_enumeration_r(f, [x])
        >>> C.compress()
        >>> C.table
        [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]
        >>> C.coset_representative(0)
        <identity>
        >>> C.coset_representative(1)
        y
        >>> C.coset_representative(2)
        y**-1

        '''
        # 对于自由生成元素集合 self.A 中的每个元素 x
        for x in self.A:
            # gamma 是在表 self.table 中对应于给定陪集 coset 的元素
            gamma = self.table[coset][self.A_dict[x]]
            # 如果 coset 是 0，则返回自由群的单位元素
            if coset == 0:
                return self.fp_group.identity
            # 如果 gamma 比 coset 小，则递归计算 gamma 的代表元，并乘以 x 的逆
            if gamma < coset:
                return self.coset_representative(gamma)*x**-1

    ##############################
    #      Modified Methods      #
    ##############################

    def modified_define(self, alpha, x):
        r"""
        定义一个从 [1..n] 到 A* 的函数 p_p，
        作为修改后的陪集表的额外组成部分。

        Parameters
        ==========

        \alpha \in \Omega
        x \in A*

        See Also
        ========

        define

        """
        self.define(alpha, x, modified=True)

    def modified_scan(self, alpha, w, y, fill=False):
        r"""
        Parameters
        ==========
        \alpha \in \Omega
        w \in A*
        y \in (YUY^-1)
        fill -- 当设置为 True 时，使用 modified_scan_and_fill。

        See Also
        ========

        scan
        """
        self.scan(alpha, w, y=y, fill=fill, modified=True)

    def modified_scan_and_fill(self, alpha, w, y):
        self.modified_scan(alpha, w, y, fill=True)

    def modified_merge(self, k, lamda, w, q):
        r"""
        Parameters
        ==========

        'k', 'lamda' -- 要合并的两个类代表元。
        q -- 长度为 l 的队列，要从 `\Omega` * 中删除的元素。
        w -- (YUY^-1) 中的单词

        See Also
        ========

        merge
        """
        self.merge(k, lamda, q, w=w, modified=True)

    def modified_rep(self, k):
        r"""
        Parameters
        ==========

        `k \in [0 \ldots n-1]`

        See Also
        ========

        rep
        """
        self.rep(k, modified=True)
    # 定义一个方法 `modified_coincidence`，该方法接受三个参数
    # - `alpha`: 表示一个符号 `\alpha`，属于集合 `\Omega`
    # - `beta`: 表示一个符号 `\beta`，属于集合 `\Omega`
    # - `w`: 表示一个符号 `w`，它是来自集合 `Y` 或其逆集合 `Y^{-1}` 的元素

    # 该方法没有返回值，其作用是调用 `coincidence` 方法，传入参数：
    # - `alpha` 和 `beta` 作为位置参数
    # - `w` 作为关键字参数 `w`
    # - `modified` 参数设为 `True`
    def modified_coincidence(self, alpha, beta, w):
        r"""
        Parameters
        ==========

        A coincident pair `\alpha, \beta \in \Omega, w \in Y \cup Y^{-1}`

        See Also
        ========

        coincidence

        """
        self.coincidence(alpha, beta, w=w, modified=True)
# relator-based method
def coset_enumeration_r(fp_grp, Y, max_cosets=None, draft=None,
                        incomplete=False, modified=False):
    """
    This is easier of the two implemented methods of coset enumeration.
    and is often called the HLT method, after Hazelgrove, Leech, Trotter
    The idea is that we make use of ``scan_and_fill`` makes new definitions
    whenever the scan is incomplete to enable the scan to complete; this way
    we fill in the gaps in the scan of the relator or subgroup generator,
    that's why the name relator-based method.

    An instance of `CosetTable` for `fp_grp` can be passed as the keyword
    argument `draft` in which case the coset enumeration will start with
    that instance and attempt to complete it.

    When `incomplete` is `True` and the function is unable to complete for
    some reason, the partially complete table will be returned.

    # TODO: complete the docstring

    See Also
    ========

    scan_and_fill,

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
    >>> F, x, y = free_group("x, y")

    # Example 5.1 from [1]
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = coset_enumeration_r(f, [x])
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [0, 0, 1, 2]
    [1, 1, 2, 0]
    [2, 2, 0, 1]
    >>> C.p
    [0, 1, 2, 1, 1]

    # Example from exercises Q2 [1]
    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    >>> C = coset_enumeration_r(f, [])
    >>> C.compress(); C.standardize()
    >>> C.table
    [[1, 2, 3, 4],
    [5, 0, 6, 7],
    [0, 5, 7, 6],
    [7, 6, 5, 0],
    [6, 7, 0, 5],
    [2, 1, 4, 3],
    [3, 4, 2, 1],
    [4, 3, 1, 2]]

    # Example 5.2
    >>> f = FpGroup(F, [x**2, y**3, (x*y)**3])
    >>> Y = [x*y]
    >>> C = coset_enumeration_r(f, Y)
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [1, 1, 2, 1]
    [0, 0, 0, 2]
    [3, 3, 1, 0]
    [2, 2, 3, 3]

    # Example 5.3
    >>> f = FpGroup(F, [x**2*y**2, x**3*y**5])
    >>> Y = []
    >>> C = coset_enumeration_r(f, Y)
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [1, 3, 1, 3]
    [2, 0, 2, 0]
    [3, 1, 3, 1]
    [0, 2, 0, 2]

    # Example 5.4
    >>> F, a, b, c, d, e = free_group("a, b, c, d, e")
    >>> f = FpGroup(F, [a*b*c**-1, b*c*d**-1, c*d*e**-1, d*e*a**-1, e*a*b**-1])
    >>> Y = [a]
    >>> C = coset_enumeration_r(f, Y)
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    # 实现基于关系的余类枚举方法，通常称为 HLT 方法，即 Hazelgrove, Leech, Trotter 方法
    # 如果给定 draft 参数作为 `CosetTable` 的实例，则从该实例开始尝试完成余类枚举
    # 当 incomplete=True 且函数无法完成时，将返回部分完成的表格
    pass  # TODO: complete the function implementation
    # 1. Initialize a coset table C for < X|R >
    C = CosetTable(fp_grp, Y, max_cosets=max_cosets)
    # Define coset table methods based on whether the table is to be modified or not.
    if modified:
        # Use modified scan and fill method for coset enumeration
        _scan_and_fill = C.modified_scan_and_fill
        # Use modified define method for coset enumeration
        _define = C.modified_define
    else:
        # Use standard scan and fill method for coset enumeration
        _scan_and_fill = C.scan_and_fill
        # Use standard define method for coset enumeration
        _define = C.define
    if draft:
        # Copy table and permutation arrays from draft table to C
        C.table = draft.table[:]
        C.p = draft.p[:]
    # Retrieve relators from the given group
    R = fp_grp.relators
    # Access A_dict from coset table C
    A_dict = C.A_dict
    # Access permutation array p from coset table C
    p = C.p
    # Iterate over Y to perform initial filling of coset table
    for i in range(len(Y)):
        if modified:
            # Perform modified scan and fill for coset enumeration
            _scan_and_fill(0, Y[i], C._grp.generators[i])
        else:
            # Perform standard scan and fill for coset enumeration
            _scan_and_fill(0, Y[i])
    # Initialize alpha to 0 for further computations
    alpha = 0
    # 当 alpha 小于 C.n 时执行循环
    while alpha < C.n:
        # 如果 p[alpha] 等于 alpha，则执行以下操作
        if p[alpha] == alpha:
            try:
                # 遍历 R 中的每个元素 w
                for w in R:
                    # 如果标记 modified 为 True，则调用 _scan_and_fill 函数并传入 alpha, w, C._grp.identity 参数
                    if modified:
                        _scan_and_fill(alpha, w, C._grp.identity)
                    else:
                        # 否则调用 _scan_and_fill 函数并传入 alpha, w 参数
                        _scan_and_fill(alpha, w)
                    # 如果在扫描过程中 alpha 被消除，则中断循环
                    if p[alpha] < alpha:
                        break
                # 如果 alpha 仍然等于 p[alpha]，则执行以下操作
                if p[alpha] == alpha:
                    # 遍历 A_dict 中的每个键 x
                    for x in A_dict:
                        # 如果 C.table[alpha][A_dict[x]] 为 None，则调用 _define 函数并传入 alpha, x 参数
                        if C.table[alpha][A_dict[x]] is None:
                            _define(alpha, x)
            except ValueError as e:
                # 如果发生 ValueError 异常，并且 incomplete 为 True，则返回 C
                if incomplete:
                    return C
                # 否则抛出该异常
                raise e
        # alpha 自增 1
        alpha += 1
    # 循环结束后返回 C
    return C
# 定义一个函数 modified_coset_enumeration_r，用于修改余类枚举方法
def modified_coset_enumeration_r(fp_grp, Y, max_cosets=None, draft=None,
                                    incomplete=False):
    r"""
    Introduce a new set of symbols y \in Y that correspond to the
    generators of the subgroup. Store the elements of Y as a
    word P[\alpha, x] and compute the coset table similar to that of
    the regular coset enumeration methods.

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup
    >>> from sympy.combinatorics.coset_table import modified_coset_enumeration_r
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = modified_coset_enumeration_r(f, [x])
    >>> C.table
    [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1], [None, 1, None, None], [1, 3, None, None]]

    See Also
    ========

    coset_enumertation_r

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.,
           "Handbook of Computational Group Theory",
           Section 5.3.2
    """
    # 调用 coset_enumeration_r 函数，修改余类枚举方法
    return coset_enumeration_r(fp_grp, Y, max_cosets=max_cosets, draft=draft,
                             incomplete=incomplete, modified=True)

# Pg. 166
# 基于余类表的方法
def coset_enumeration_c(fp_grp, Y, max_cosets=None, draft=None,
                                                incomplete=False):
    """
    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_c
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = coset_enumeration_c(f, [x])
    >>> C.table
    [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]

    """
    # 初始化一个 < X|R > 的余类表 C
    X = fp_grp.generators
    R = fp_grp.relators
    C = CosetTable(fp_grp, Y, max_cosets=max_cosets)
    # 如果有草稿（draft），则使用草稿初始化余类表 C
    if draft:
        C.table = draft.table[:]
        C.p = draft.p[:]
        C.deduction_stack = draft.deduction_stack
        # 遍历余类表 C 的所有可能组合，将不为空的填充到推导栈中
        for alpha, x in product(range(len(C.table)), X):
            if C.table[alpha][C.A_dict[x]] is not None:
                C.deduction_stack.append((alpha, x))
    A = C.A
    # 对所有关系元素进行循环规约
    R_cyc_red = [rel.identity_cyclic_reduction() for rel in R]
    # 将所有循环共轭类元素连接成一个列表
    R_c = list(chain.from_iterable((rel.cyclic_conjugates(), (rel**-1).cyclic_conjugates()) \
            for rel in R_cyc_red))
    R_set = set()
    # 将所有循环共轭类元素合并到集合 R_set 中
    for conjugate in R_c:
        R_set = R_set.union(conjugate)
    # 将以每个元素 x 开头的 R_c 元素的子集列表化
    R_c_list = []
    for x in C.A:
        r = {word for word in R_set if word[0] == x}
        R_c_list.append(r)
        R_set.difference_update(r)
    # 扫描并填充余类表 C 中的元素
    for w in Y:
        C.scan_and_fill_c(0, w)
    # 处理推导，用 R_c_list[C.A_dict[x]] 和 R_c_list[C.A_dict_inv[x]] 做参数
    for x in A:
        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
    alpha = 0
    # 当 alpha 小于 C.table 的长度时，执行循环
    while alpha < len(C.table):
        # 如果 C.p[alpha] 等于 alpha，进入条件判断
        if C.p[alpha] == alpha:
            try:
                # 遍历 C.A 中的每个元素 x
                for x in C.A:
                    # 如果 C.p[alpha] 不等于 alpha，则跳出循环
                    if C.p[alpha] != alpha:
                        break
                    # 如果 C.table[alpha][C.A_dict[x]] 为 None
                    if C.table[alpha][C.A_dict[x]] is None:
                        # 调用 define_c 方法定义 C.table[alpha][C.A_dict[x]] 的值
                        C.define_c(alpha, x)
                        # 处理推断，传递 R_c_list[C.A_dict[x]] 和 R_c_list[C.A_dict_inv[x]]
                        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
            # 捕获 ValueError 异常
            except ValueError as e:
                # 如果 incomplete 为真，则返回 C
                if incomplete:
                    return C
                # 否则抛出异常 e
                raise e
        # alpha 自增
        alpha += 1
    # 返回 C
    return C
```