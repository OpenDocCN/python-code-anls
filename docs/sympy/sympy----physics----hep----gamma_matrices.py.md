# `D:\src\scipysrc\sympy\sympy\physics\hep\gamma_matrices.py`

```
# 导入所需模块和类
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.dense import eye
from sympy.matrices.expressions.trace import trace
from sympy.tensor.tensor import TensorIndexType, TensorIndex,\
    TensMul, TensAdd, tensor_mul, Tensor, TensorHead, TensorSymmetry

# 定义洛伦兹指标类型
LorentzIndex = TensorIndexType('LorentzIndex', dim=4, dummy_name="L")

# 定义Gamma矩阵作为张量头
GammaMatrix = TensorHead("GammaMatrix", [LorentzIndex],
                         TensorSymmetry.no_symmetry(1), comm=None)

def extract_type_tens(expression, component):
    """
    从 ``TensExpr`` 中提取所有具有 `component` 的张量。

    返回两个张量表达式：

    * 第一个包含所有具有 `component` 的 ``Tensor``。
    * 第二个包含剩余的所有张量。

    """
    # 如果表达式是单个张量，则转为列表
    if isinstance(expression, Tensor):
        sp = [expression]
    # 如果是张量乘积，则获取其参数列表
    elif isinstance(expression, TensMul):
        sp = expression.args
    else:
        raise ValueError('wrong type')

    # 初始化两个新的表达式
    new_expr = S.One
    residual_expr = S.One

    # 遍历参数列表，将具有指定组分的张量添加到新表达式中，其余添加到剩余表达式中
    for i in sp:
        if isinstance(i, Tensor) and i.component == component:
            new_expr *= i
        else:
            residual_expr *= i

    return new_expr, residual_expr


def simplify_gamma_expression(expression):
    """
    简化表达式中的Gamma矩阵部分。

    提取和简化所有包含Gamma矩阵的张量，并与剩余部分合并。

    """
    # 提取所有包含Gamma矩阵的张量部分和剩余部分
    extracted_expr, residual_expr = extract_type_tens(expression, GammaMatrix)
    # 对提取的部分进行简化
    res_expr = _simplify_single_line(extracted_expr)
    # 返回简化后的结果乘以剩余部分
    return res_expr * residual_expr


def simplify_gpgp(ex, sort=True):
    """
    简化形如 ``G(i)*p(-i)*G(j)*p(-j) -> p(i)*p(-i)`` 的张量乘积。

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        LorentzIndex, simplify_gpgp
    >>> from sympy.tensor.tensor import tensor_indices, tensor_heads
    >>> p, q = tensor_heads('p, q', [LorentzIndex])
    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)
    >>> ps = p(i0)*G(-i0)
    >>> qs = q(i0)*G(-i0)
    >>> simplify_gpgp(ps*qs*qs)
    GammaMatrix(-L_0)*p(L_0)*q(L_1)*q(-L_1)
    """
    # 定义一个函数 `_simplify_gpgp`，用于简化给定的表达式 `ex`
    def _simplify_gpgp(ex):
        # 从表达式 `ex` 中获取其组件列表
        components = ex.components
        # 初始化空列表 `a` 用于存储符合条件的元组
        a = []
        # 初始化空列表 `comp_map`，用于映射每个组件的索引
        comp_map = []
        # 遍历组件列表 `components`
        for i, comp in enumerate(components):
            # 将当前组件的索引重复添加 `comp.rank` 次到 `comp_map` 中
            comp_map.extend([i]*comp.rank)
        # 从表达式 `ex` 中获取哑指标列表，并映射到对应的组件索引
        dum = [(i[0], i[1], comp_map[i[0]], comp_map[i[1]]) for i in ex.dum]
        
        # 遍历组件列表 `components`
        for i in range(len(components)):
            # 如果当前组件不是 `GammaMatrix`，则跳过本次循环
            if components[i] != GammaMatrix:
                continue
            # 遍历哑指标 `dum`
            for dx in dum:
                # 如果哑指标 `dx` 的第三个元素等于当前组件索引 `i`
                if dx[2] == i:
                    p_pos1 = dx[3]
                # 如果哑指标 `dx` 的第四个元素等于当前组件索引 `i`
                elif dx[3] == i:
                    p_pos1 = dx[2]
                else:
                    continue
                # 获取 `p_pos1` 对应的组件
                comp1 = components[p_pos1]
                # 如果组件 `comp1` 的交换性为 0 且秩为 1
                if comp1.comm == 0 and comp1.rank == 1:
                    # 将元组 `(i, p_pos1)` 添加到列表 `a` 中
                    a.append((i, p_pos1))
        
        # 如果列表 `a` 为空，则返回原始表达式 `ex`
        if not a:
            return ex
        
        # 初始化一个空集合 `elim`，用于存储需要消除的索引
        elim = set()
        # 初始化空列表 `tv`，用于存储处理后的组件
        tv = []
        # 初始化标志 `hit` 为 `True`
        hit = True
        # 初始化 `coeff` 为 `S.One`，一个特殊的符号
        coeff = S.One
        # 初始化 `ta` 为 `None`
        ta = None
        
        # 当标志 `hit` 为 `True` 时循环
        while hit:
            # 将标志 `hit` 设为 `False`
            hit = False
            # 遍历列表 `a` 中的元素，但不包括最后一个元素
            for i, ai in enumerate(a[:-1]):
                # 如果 `ai[0]` 已经在 `elim` 中，则跳过本次循环
                if ai[0] in elim:
                    continue
                # 如果 `ai[0]` 不等于下一个元素的第一个值减去 1，则跳过本次循环
                if ai[0] != a[i + 1][0] - 1:
                    continue
                # 如果 `ai[1]` 对应的组件不等于下一个元素的第二个值对应的组件，则跳过本次循环
                if components[ai[1]] != components[a[i + 1][1]]:
                    continue
                # 将 `ai[0]`、`ai[1]`、`a[i + 1][0]` 和 `a[i + 1][1]` 添加到 `elim` 中
                elim.add(ai[0])
                elim.add(ai[1])
                elim.add(a[i + 1][0])
                elim.add(a[i + 1][1])
                # 如果 `ta` 为 `None`
                if not ta:
                    # 将 `ex` 拆分为 `ta`
                    ta = ex.split()
                    # 创建一个 `mu` 张量索引对象
                    mu = TensorIndex('mu', LorentzIndex)
                # 将标志 `hit` 设为 `True`
                hit = True
                # 如果 `i` 等于 0
                if i == 0:
                    # 将 `coeff` 设置为 `ex.coeff`
                    coeff = ex.coeff
                # 创建一个新的张量 `tx`
                tx = components[ai[1]](mu)*components[ai[1]](-mu)
                # 如果 `a` 的长度为 2，则将 `tx` 乘以 4
                if len(a) == 2:
                    tx *= 4  # eye(4)
                # 将 `tx` 添加到列表 `tv` 中
                tv.append(tx)
                # 退出循环
                break
        
        # 如果 `tv` 不为空
        if tv:
            # 从 `ta` 中选择不在 `elim` 中的元素，将其添加到列表 `a` 中
            a = [x for j, x in enumerate(ta) if j not in elim]
            # 将 `tv` 中的元素添加到列表 `a` 中
            a.extend(tv)
            # 将列表 `a` 中的元素进行张量乘法，并乘以 `coeff`
            t = tensor_mul(*a)*coeff
            # 返回处理后的张量 `t`
            return t
        else:
            # 如果 `tv` 为空，则返回原始表达式 `ex`
            return ex
    
    # 如果 `sort` 为真，则对表达式 `ex` 进行按组件排序
    if sort:
        ex = ex.sorted_components()
    # 使用 `_simplify_gpgp` 函数对表达式 `ex` 进行简化，直到简化完成为止
    # 如果 `t` 不等于 `ex`，则更新 `ex` 为 `t`，否则返回 `t`
    while 1:
        t = _simplify_gpgp(ex)
        if t != ex:
            ex = t
        else:
            return t
# 计算给定张量表达式的 gamma 矩阵线路的迹
def gamma_trace(t):
    """
    trace of a single line of gamma matrices

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        gamma_trace, LorentzIndex
    >>> from sympy.tensor.tensor import tensor_indices, tensor_heads
    >>> p, q = tensor_heads('p, q', [LorentzIndex])
    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)
    >>> ps = p(i0)*G(-i0)
    >>> qs = q(i0)*G(-i0)
    >>> gamma_trace(G(i0)*G(i1))
    4*metric(i0, i1)
    >>> gamma_trace(ps*ps) - 4*p(i0)*p(-i0)
    0
    >>> gamma_trace(ps*qs + ps*ps) - 4*p(i0)*p(-i0) - 4*p(i0)*q(-i0)
    0

    """
    # 如果输入张量 t 是 TensAdd 类型，递归地对每个元素调用 gamma_trace，并返回结果
    if isinstance(t, TensAdd):
        res = TensAdd(*[gamma_trace(x) for x in t.args])
        return res
    # 简化单行 gamma 矩阵表达式
    t = _simplify_single_line(t)
    # 计算单行 gamma 矩阵表达式的迹
    res = _trace_single_line(t)
    return res


# 简化单行 gamma 矩阵表达式
def _simplify_single_line(expression):
    """
    Simplify single-line product of gamma matrices.

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        LorentzIndex, _simplify_single_line
    >>> from sympy.tensor.tensor import tensor_indices, TensorHead
    >>> p = TensorHead('p', [LorentzIndex])
    >>> i0,i1 = tensor_indices('i0:2', LorentzIndex)
    >>> _simplify_single_line(G(i0)*G(i1)*p(-i1)*G(-i0)) + 2*G(i0)*p(-i0)
    0

    """
    # 提取出类型为 GammaMatrix 的部分
    t1, t2 = extract_type_tens(expression, GammaMatrix)
    # 如果 t1 不等于 1，使用 kahane_simplify 对其进行简化
    if t1 != 1:
        t1 = kahane_simplify(t1)
    # 返回简化后的结果
    res = t1 * t2
    return res


# 计算单行 gamma 矩阵表达式的迹
def _trace_single_line(t):
    """
    Evaluate the trace of a single gamma matrix line inside a ``TensExpr``.

    Notes
    =====

    If there are ``DiracSpinorIndex.auto_left`` and ``DiracSpinorIndex.auto_right``
    indices trace over them; otherwise traces are not implied (explain)


    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        LorentzIndex, _trace_single_line
    >>> from sympy.tensor.tensor import tensor_indices, TensorHead
    >>> p = TensorHead('p', [LorentzIndex])
    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)
    >>> _trace_single_line(G(i0)*G(i1))
    4*metric(i0, i1)
    >>> _trace_single_line(G(i0)*p(-i0)*G(i1)*p(-i1)) - 4*p(i0)*p(-i0)
    0

    """
    # 定义内部函数 _trace_single_line1，接收参数 t
    def _trace_single_line1(t):
        # 对参数 t 执行 sorted_components() 方法，返回排序后的组件
        t = t.sorted_components()
        # 获取排序后的组件列表赋值给 components
        components = t.components
        # 获取组件列表的长度赋值给 ncomps
        ncomps = len(components)
        # 获取 LorentzIndex 类的 metric 属性赋值给 g
        g = LorentzIndex.metric
        # 初始化变量 hit 为 0
        hit = 0
        # 遍历 components 列表的索引范围
        for i in range(ncomps):
            # 如果当前组件等于 GammaMatrix 类型
            if components[i] == GammaMatrix:
                # 设置 hit 为 1 并退出循环
                hit = 1
                break

        # 从 i + hit 开始遍历 components 列表的索引范围
        for j in range(i + hit, ncomps):
            # 如果当前组件不等于 GammaMatrix 类型
            if components[j] != GammaMatrix:
                # 跳出循环
                break
        else:
            # 如果未跳出循环，将 j 设置为 ncomps
            j = ncomps
        
        # 计算 GammaMatrix 出现的次数
        numG = j - i
        # 如果 numG 等于 0
        if numG == 0:
            # 如果 t.coeff 不为零则返回 t.nocoeff，否则返回 t
            tcoeff = t.coeff
            return t.nocoeff if tcoeff else t
        # 如果 numG 是奇数
        if numG % 2 == 1:
            # 返回一组空的 TensMul 对象
            return TensMul.from_data(S.Zero, [], [], [])
        # 如果 numG 大于 4
        elif numG > 4:
            # 将 t 拆分为子项，赋值给 a
            a = t.split()
            # 获取第 i 项的第一个索引，赋值给 ind1
            ind1 = a[i].get_indices()[0]
            # 获取第 i+1 项的第一个索引，赋值给 ind2
            ind2 = a[i + 1].get_indices()[0]
            # 移除 a 中第 i 和第 i+1 项，赋值给 aa
            aa = a[:i] + a[i + 2:]
            # 构建第一部分 tensor 乘积，乘以 g(ind1, ind2)
            t1 = tensor_mul(*aa)*g(ind1, ind2)
            # 将 t1 和 g 进行度规约简
            t1 = t1.contract_metric(g)
            # 初始化参数列表 args 为 [t1]
            args = [t1]
            # 初始化符号为 1
            sign = 1
            # 从 i+2 到 j 遍历
            for k in range(i + 2, j):
                # 符号取反
                sign = -sign
                # 获取第 k 项的第一个索引，赋值给 ind2
                ind2 = a[k].get_indices()[0]
                # 移除 a 中第 i, i+1 和 k 项，赋值给 aa
                aa = a[:i] + a[i + 1:k] + a[k + 1:]
                # 构建第 k 部分 tensor 乘积，乘以 g(ind1, ind2)
                t2 = sign*tensor_mul(*aa)*g(ind1, ind2)
                # 将 t2 和 g 进行度规约简
                t2 = t2.contract_metric(g)
                # 对 t2 进行 simplify_gpgp 处理，不保留梯度
                t2 = simplify_gpgp(t2, False)
                # 将 t2 添加到 args 列表中
                args.append(t2)
            # 构建 TensAdd 对象，合并 args 列表中的项
            t3 = TensAdd(*args)
            # 对 t3 继续调用 _trace_single_line 函数
            t3 = _trace_single_line(t3)
            # 返回 t3
            return t3
        else:
            # 将 t 拆分为子项，赋值给 a
            a = t.split()
            # 调用 _gamma_trace1 处理 a[i:j] 部分
            t1 = _gamma_trace1(*a[i:j])
            # 移除 a 中第 i 到 j 项，赋值给 a2
            a2 = a[:i] + a[j:]
            # 构建 tensor 乘积 t2
            t2 = tensor_mul(*a2)
            # 计算 t3 为 t1*t2
            t3 = t1*t2
            # 如果 t3 为假值，则返回 t3
            if not t3:
                return t3
            # 对 t3 进行度规约简
            t3 = t3.contract_metric(g)
            # 返回 t3
            return t3

    # 对参数 t 进行展开处理
    t = t.expand()
    # 如果 t 是 TensAdd 类型
    if isinstance(t, TensAdd):
        # 对 t 中的每一项调用 _trace_single_line1 并乘以其系数，构建列表 a
        a = [_trace_single_line1(x)*x.coeff for x in t.args]
        # 返回 TensAdd 对象，合并列表 a 中的项
        return TensAdd(*a)
    # 如果 t 是 Tensor 或 TensMul 类型
    elif isinstance(t, (Tensor, TensMul)):
        # 计算 r 为 t 的系数乘以 _trace_single_line1 处理后的结果
        r = t.coeff*_trace_single_line1(t)
        # 返回 r
        return r
    else:
        # 返回 trace(t) 的结果
        return trace(t)
# 定义函数 `_gamma_trace1`，用于计算四维伽玛矩阵的轨迹
def _gamma_trace1(*a):
    gctr = 4  # FIXME specific for d=4，设定固定值 4，适用于 d=4 的情况
    g = LorentzIndex.metric  # 获取 Lorentz 指标的度量张量
    if not a:
        return gctr  # 若参数列表为空，则直接返回 gctr
    n = len(a)
    if n % 2 == 1:
        # 若参数个数为奇数，返回零张量 S.Zero
        return S.Zero
    if n == 2:
        ind0 = a[0].get_indices()[0]  # 获取第一个参数的第一个指标
        ind1 = a[1].get_indices()[0]  # 获取第二个参数的第一个指标
        return gctr * g(ind0, ind1)  # 返回 gctr 乘以 g 的两个指标的度量值
    if n == 4:
        ind0 = a[0].get_indices()[0]  # 获取第一个参数的第一个指标
        ind1 = a[1].get_indices()[0]  # 获取第二个参数的第一个指标
        ind2 = a[2].get_indices()[0]  # 获取第三个参数的第一个指标
        ind3 = a[3].get_indices()[0]  # 获取第四个参数的第一个指标

        # 返回 gctr 乘以给定四维伽玛矩阵的轨迹
        return gctr * (g(ind0, ind1) * g(ind2, ind3) - \
           g(ind0, ind2) * g(ind1, ind3) + g(ind0, ind3) * g(ind1, ind2))


# 定义函数 `kahane_simplify`，用于简化四维伽玛矩阵的张量表达式
def kahane_simplify(expression):
    r"""
    This function cancels contracted elements in a product of four
    dimensional gamma matrices, resulting in an expression equal to the given
    one, without the contracted gamma matrices.

    Parameters
    ==========

    `expression`    the tensor expression containing the gamma matrices to simplify.

    Notes
    =====

    If spinor indices are given, the matrices must be given in
    the order given in the product.

    Algorithm
    =========

    The idea behind the algorithm is to use some well-known identities,
    i.e., for contractions enclosing an even number of `\gamma` matrices

    `\gamma^\mu \gamma_{a_1} \cdots \gamma_{a_{2N}} \gamma_\mu = 2 (\gamma_{a_{2N}} \gamma_{a_1} \cdots \gamma_{a_{2N-1}} + \gamma_{a_{2N-1}} \cdots \gamma_{a_1} \gamma_{a_{2N}} )`

    for an odd number of `\gamma` matrices

    `\gamma^\mu \gamma_{a_1} \cdots \gamma_{a_{2N+1}} \gamma_\mu = -2 \gamma_{a_{2N+1}} \gamma_{a_{2N}} \cdots \gamma_{a_{1}}`

    Instead of repeatedly applying these identities to cancel out all contracted indices,
    it is possible to recognize the links that would result from such an operation,
    the problem is thus reduced to a simple rearrangement of free gamma matrices.

    Examples
    ========

    When using, always remember that the original expression coefficient
    has to be handled separately

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
    >>> from sympy.physics.hep.gamma_matrices import kahane_simplify
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i0, i1, i2 = tensor_indices('i0:3', LorentzIndex)
    >>> ta = G(i0)*G(-i0)
    >>> kahane_simplify(ta)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])
    >>> tb = G(i0)*G(i1)*G(-i0)
    >>> kahane_simplify(tb)
    -2*GammaMatrix(i1)
    >>> t = G(i0)*G(-i0)
    >>> kahane_simplify(t)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])
    >>> t = G(i0)*G(-i0)
    >>> kahane_simplify(t)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])

    If there are no contractions, the same expression is returned

    >>> tc = G(i0)*G(i1)
    >>> kahane_simplify(tc)
    GammaMatrix(i0)*GammaMatrix(i1)

    References
    ==========

    """
    """
    [1] Algorithm for Reducing Contracted Products of gamma Matrices,
    Joseph Kahane, Journal of Mathematical Physics, Vol. 9, No. 10, October 1968.
    """

    # 检查表达式的类型，如果是乘法表达式（Mul），则直接返回表达式
    if isinstance(expression, Mul):
        return expression
    # 如果是张量加法表达式（TensAdd），则对每个参数递归应用 Kahane 简化，并返回结果的加法张量
    if isinstance(expression, TensAdd):
        return TensAdd(*[kahane_simplify(arg) for arg in expression.args])

    # 如果表达式是张量（Tensor），则直接返回表达式
    if isinstance(expression, Tensor):
        return expression

    # 确保表达式是张量乘法表达式（TensMul）
    assert isinstance(expression, TensMul)

    # 获取表达式中的所有 gamma 矩阵
    gammas = expression.args

    # 确保所有的 gamma 矩阵都是 GammaMatrix 类型
    for gamma in gammas:
        assert gamma.component == GammaMatrix

    # 获取自由指标
    free = expression.free

    # 按 LorentzIndex 类型筛选出哑指标对
    dum = []
    for dum_pair in expression.dum:
        if expression.index_types[dum_pair[0]] == LorentzIndex:
            dum.append((dum_pair[0], dum_pair[1]))

    # 对哑指标对按照位置进行排序
    dum = sorted(dum)

    # 如果没有哑指标对，直接返回表达式
    if len(dum) == 0:
        return expression

    # 找到第一个哑指标的位置
    first_dum_pos = min(map(min, dum))

    # 计算总的指标数量和收缩的指标数量
    total_number = len(free) + len(dum)*2
    number_of_contractions = len(dum)

    # 初始化自由指标位置列表
    free_pos = [None]*total_number
    for i in free:
        free_pos[i[1]] = i[0]

    # 初始化指标是否为自由指标的布尔列表
    index_is_free = [False]*total_number
    for i, indx in enumerate(free):
        index_is_free[indx[1]] = True

    # 初始化链接字典，用于描述 Kahane 论文中的指标连接关系
    links = {i: [] for i in range(first_dum_pos, total_number)}

    # 初始化累积符号变量，标记每个指标的符号
    cum_sign = -1
    # 初始化累积符号列表
    cum_sign_list = [None]*total_number
    block_free_count = 0

    # 将结果系数乘以系数参数，其余部分...
    # 算法结果的标量系数
    resulting_coeff = S.One

    # 初始化索引列表的列表。外部列表包含所有加法张量表达式，内部列表包含自由索引（根据算法重新排列）。
    resulting_indices = [[]]

    # 开始计算 `connected_components`，它与缩并数量一起确定要乘以的 -1 或 +1 因子。
    connected_components = 1

    # 第一个循环：在此处填充 `cum_sign_list` 并绘制连续索引之间的连接（它们存储在 `links` 中）。
    # 非连续索引之间的连接稍后绘制。
    for i, is_free in enumerate(index_is_free):
        # 如果 `expression` 以自由索引开头，则在此处忽略它们；它们稍后将按原样添加到所有 `resulting_indices` 的列表中。
        if i < first_dum_pos:
            continue

        if is_free:
            block_free_count += 1
            # 如果前一个索引也是自由的，则在 `links` 中绘制一条弧线。
            if block_free_count > 1:
                links[i - 1].append(i)
                links[i].append(i - 1)
        else:
            # 如果前面的自由索引数量为偶数，则更改索引的符号 (`cum_sign`)。
            cum_sign *= 1 if (block_free_count % 2) else -1
            if block_free_count == 0 and i != first_dum_pos:
                # 检查是否有两个连续的虚拟指数：
                # 在这种情况下，创建负位置的虚拟指数，
                # 这些“虚拟”指数表示插入两个 gamma^0 矩阵来分隔连续的虚拟指数，因为
                # Kahane 的算法要求虚拟指数由自由指数分隔。两个 gamma^0 矩阵的乘积为单位矩阵，
                # 因此正在检查的新表达式与原始表达式相同。
                if cum_sign == -1:
                    links[-1-i] = [-1-i+1]
                    links[-1-i+1] = [-1-i]
            if (i - cum_sign) in links:
                if i != first_dum_pos:
                    links[i].append(i - cum_sign)
                if block_free_count != 0:
                    if i - cum_sign < len(index_is_free):
                        if index_is_free[i - cum_sign]:
                            links[i - cum_sign].append(i)
            block_free_count = 0

        cum_sign_list[i] = cum_sign

    # 前一个循环只创建了连续自由索引之间的连接，根据 Kahane 论文中描述的规则，
    # 需要适当地创建虚拟指数之间的连接（收缩索引）。
    # Kahane 的规则只有一个例外：处理某些连续自由索引的负索引，
    # （Kahane 的论文只描述了虚拟指数
    # 分离的自由索引，暗示可以添加自由索引而不改变表达式结果。
    for i in dum:
        # 获取两个收缩索引的位置：
        pos1 = i[0]
        pos2 = i[1]

        # 创建Kahane的上链接，即在虚拟（即收缩）索引之间的上弧：
        links[pos1].append(pos2)
        links[pos2].append(pos1)

        # 创建Kahane的下链接，这对应于论文中描述的线下的弧：

        # 根据索引的符号移动`pos1`和`pos2`：
        linkpos1 = pos1 + cum_sign_list[pos1]
        linkpos2 = pos2 + cum_sign_list[pos2]

        # 在创建下弧之前执行一些检查：

        # 确保不超过索引的总数：
        if linkpos1 >= total_number:
            continue
        if linkpos2 >= total_number:
            continue

        # 确保不低于表达式中第一个虚拟索引的位置：
        if linkpos1 < first_dum_pos:
            continue
        if linkpos2 < first_dum_pos:
            continue

        # 检查前面的循环是否创建了在虚拟索引之间的“虚拟”索引，如果是这样，重新链接`linkpos1`和`linkpos2`：
        if (-1 - linkpos1) in links:
            linkpos1 = -1 - linkpos1
        if (-1 - linkpos2) in links:
            linkpos2 = -1 - linkpos2

        # 只有在不靠近自由索引时才移动：
        if linkpos1 >= 0 and not index_is_free[linkpos1]:
            linkpos1 = pos1

        if linkpos2 >= 0 and not index_is_free[linkpos2]:
            linkpos2 = pos2

        # 创建下弧：
        if linkpos2 not in links[linkpos1]:
            links[linkpos1].append(linkpos2)
        if linkpos1 not in links[linkpos2]:
            links[linkpos2].append(linkpos1)

    # 此循环从`first_dum_pos`索引（第一个虚拟索引）开始，遍历图形并从`links`中删除访问过的索引，
    # 每遇到一个自由索引，向结果索引中添加一个gamma矩阵，完全忽略虚拟索引和虚拟索引。
    pointer = first_dum_pos
    previous_pointer = 0
    while True:
        if pointer in links:
            next_ones = links.pop(pointer)
        else:
            break

        if previous_pointer in next_ones:
            next_ones.remove(previous_pointer)

        previous_pointer = pointer

        if next_ones:
            pointer = next_ones[0]
        else:
            break

        if pointer == previous_pointer:
            break
        if pointer >= 0 and free_pos[pointer] is not None:
            for ri in resulting_indices:
                ri.append(free_pos[pointer])

    # 以下循环移除`links`中剩余的连接组件。如果连接组件内有自由索引，它对结果表达式的贡献为
    # ultimate因子。
    # 当前代码段描述了一个算法或者数学运算中的一部分，其详细功能和背景如下：

    while links:
        # 每遇到一个连接的组件，增加 `connected_components` 的计数
        connected_components += 1
        
        # 选择最小的键作为指针，开始追踪连接的路径
        pointer = min(links.keys())
        previous_pointer = pointer
        
        # 从 links 中移除已访问的索引，并将所有自由索引添加到 prepend_indices 列表中
        # 虚拟索引将被忽略
        prepend_indices = []
        while True:
            if pointer in links:
                next_ones = links.pop(pointer)
            else:
                break
            
            # 如果上一个指针在 next_ones 中存在，则移除它（保证不重复访问）
            if previous_pointer in next_ones:
                if len(next_ones) > 1:
                    next_ones.remove(previous_pointer)
            
            previous_pointer = pointer
            
            # 更新指针到下一个连接的索引
            if next_ones:
                pointer = next_ones[0]
            
            # 如果指针指向的位置在第一个虚拟位置之后，并且有一个自由位置存在
            # 将该自由位置添加到 prepend_indices 的开头
            if pointer >= first_dum_pos and free_pos[pointer] is not None:
                prepend_indices.insert(0, free_pos[pointer])
        
        # 如果 prepend_indices 是空的，说明循环中没有自由索引
        # 根据 Kahane 的论文，只含有虚拟索引的循环贡献因子为 2
        if len(prepend_indices) == 0:
            resulting_coeff *= 2
        # 否则，将 prepend_indices 中的自由索引添加到 resulting_indices 中
        else:
            expr1 = prepend_indices
            expr2 = list(reversed(prepend_indices))
            resulting_indices = [expri + ri for ri in resulting_indices for expri in (expr1, expr2)]
    
    # 符号修正，根据 Kahane 的论文描述
    resulting_coeff *= -1 if (number_of_contractions - connected_components + 1) % 2 else 1
    
    # 二的幂因子，根据 Kahane 的论文描述
    resulting_coeff *= 2**(number_of_contractions)
    
    # 如果 first_dum_pos 不是零，表示在 expression 前面有一些尾随的自由 gamma 矩阵，因此乘以它们
    resulting_indices = [ free_pos[0:first_dum_pos] + ri for ri in resulting_indices ]
    
    # 初始化结果表达式为零
    resulting_expr = S.Zero
    
    # 遍历 resulting_indices 中的每一个列表
    for i in resulting_indices:
        # 初始化临时表达式为 1
        temp_expr = S.One
        # 对于每一个索引 j，在 temp_expr 上乘以 GammaMatrix(j)
        for j in i:
            temp_expr *= GammaMatrix(j)
        # 将 temp_expr 添加到 resulting_expr 中
        resulting_expr += temp_expr
    
    # 计算最终结果 t
    t = resulting_coeff * resulting_expr
    t1 = None
    
    # 如果 t 是 TensAdd 类型，取其第一个参数作为 t1
    if isinstance(t, TensAdd):
        t1 = t.args[0]
    # 如果 t 是 TensMul 类型，直接将 t 赋给 t1
    elif isinstance(t, TensMul):
        t1 = t
    
    # 对 t1 进行必要的处理（这里是占位符，未指定具体操作）
    if t1:
        pass
    # 如果条件不满足，执行以下操作：
    else:
        # 创建一个4x4的单位矩阵，并与 t 相乘，返回结果赋给 t
        t = eye(4)*t
    # 返回变量 t 的值
    return t
```