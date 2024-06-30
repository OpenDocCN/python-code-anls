# `D:\src\scipysrc\sympy\sympy\polys\matrices\rref.py`

```
# 导入整数环 ZZ 用于矩阵运算
from sympy.polys.domains import ZZ

# 导入 SDM 类及其相关函数，用于稀疏矩阵的操作
from sympy.polys.matrices.sdm import SDM, sdm_irref, sdm_rref_den
# 导入 DDM 类，用于密集矩阵的操作
from sympy.polys.matrices.ddm import DDM
# 导入密集矩阵的消元算法函数
from sympy.polys.matrices.dense import ddm_irref, ddm_irref_den


def _dm_rref(M, *, method='auto'):
    """
    计算域矩阵的行简化阶梯形式。

    此函数是 DomainMatrix.rref 方法的实现。

    根据矩阵的域、形状、稀疏性以及例如 ZZ 或 QQ 的位数选择最佳算法。结果返回
    在与矩阵域关联的字段上。

    参见
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        调用此函数的 DomainMatrix 方法。
    sympy.polys.matrices.rref._dm_rref_den
        用于计算带分母的 RREF 的替代函数。
    """
    # 根据给定参数选择最佳的算法方法和格式
    method, use_fmt = _dm_rref_choose_method(M, method, denominator=False)

    # 将矩阵 M 转换为指定的格式 use_fmt，并记录转换前的格式 old_fmt
    M, old_fmt = _dm_to_fmt(M, use_fmt)

    if method == 'GJ':
        # 使用在关联字段上的带除法的 Gauss-Jordan 消元法
        Mf = _to_field(M)
        M_rref, pivots = _dm_rref_GJ(Mf)

    elif method == 'FF':
        # 使用当前域上的无分数 Gauss-Jordan 消元法
        M_rref_f, den, pivots = _dm_rref_den_FF(M)
        # 将结果矩阵转换为关联域上的结果
        M_rref = _to_field(M_rref_f) / den
    # 如果方法为'CD'，则执行以下操作：
    #   1. 在相关环域中清除分母并使用无分数的高斯约当消元法。
    _, Mr = M.clear_denoms_rowwise(convert=True)
    # 对清除分母后的矩阵进行分数自由高斯约当消元法，得到分数自由高斯约当消元后的结果M_rref_f，通分的分母den和主元信息pivots。
    M_rref_f, den, pivots = _dm_rref_den_FF(Mr)
    # 将M_rref_f转换为所属域中的值，并将任何分母除掉（因此现在隐含为1）。
    M_rref = _to_field(M_rref_f) / den

    # 如果方法不是'CD'，则引发值错误异常，显示未知的rref方法。
    else:
        raise ValueError(f"Unknown method for rref: {method}")

    # 将结果矩阵M_rref转换为原始格式（稀疏或密集）。
    M_rref, _ = _dm_to_fmt(M_rref, old_fmt)

    # 不变量：
    #   - M_rref与输入矩阵具有相同的格式（稀疏或密集）。
    #   - M_rref在相关的域中，并且任何分母已经除掉（因此现在隐含为1）。
    return M_rref, pivots
# 计算带分母的 DomainMatrix 的行最简形式。
# 这个函数是 DomainMatrix.rref_den 方法的实现。

# 根据输入的 DomainMatrix M 和参数选择最佳算法，包括使用的方法和是否保留域。
# 返回的结果在与输入矩阵相同的域上，除非 keep_domain=False，此时结果可能在相关的环或域域上。
def _dm_rref_den(M, *, keep_domain=True, method='auto'):
    method, use_fmt = _dm_rref_choose_method(M, method, denominator=True)

    # 将 M 转换为指定的格式 use_fmt，并记录原始格式 old_fmt
    M, old_fmt = _dm_to_fmt(M, use_fmt)

    if method == 'FF':
        # 使用无分数 GJ 方法在当前域上进行计算。
        M_rref, den, pivots = _dm_rref_den_FF(M)

    elif method == 'GJ':
        # 使用高斯-约当消元法在相关的域上进行计算。
        M_rref_f, pivots = _dm_rref_GJ(_to_field(M))

        # 如果 keep_domain=True 并且结果域与 M 的域不同，则清除分母并转换。
        if keep_domain and M_rref_f.domain != M.domain:
            _, M_rref = M_rref_f.clear_denoms(convert=True)

            # 如果存在主元，则使用对应的元素作为分母，否则为单位元。
            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
        else:
            # 可能是相关的域
            M_rref = M_rref_f
            den = M_rref.domain.one

    elif method == 'CD':
        # 清除分母并在相关的环中使用无分数 GJ 方法。
        _, Mr = M.clear_denoms_rowwise(convert=True)

        M_rref_r, den, pivots = _dm_rref_den_FF(Mr)

        # 如果 keep_domain=True 并且结果域与 M 的域不同，则转换回到域中。
        if keep_domain and M_rref_r.domain != M.domain:
            M_rref = _to_field(M_rref_r) / den
            den = M.domain.one
        else:
            # 可能是相关的环
            M_rref = M_rref_r

            # 如果存在主元，则使用对应的元素作为分母，否则为单位元。
            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
    else:
        raise ValueError(f"Unknown method for rref: {method}")

    # 将 M_rref 转换回原来的格式 old_fmt
    M_rref, _ = _dm_to_fmt(M_rref, old_fmt)

    # 不变性：
    #   - M_rref 的格式（稀疏或密集）与输入矩阵相同。
    #   - 如果 keep_domain=True，则 M_rref 和 den 与输入矩阵在相同的域上。
    #   - 如果 keep_domain=False，则 M_rref 可能在相关的环或域上，但 den 总是与 M_rref 的域相同。

    return M_rref, den, pivots


def _dm_to_fmt(M, fmt):
    """将矩阵转换为给定的格式，并返回原始格式。"""
    old_fmt = M.rep.fmt
    if old_fmt == fmt:
        pass
    elif fmt == 'dense':
        # 将矩阵转换为密集格式
        M = M.to_dense()
    # 如果格式为 'sparse'，将稠密矩阵 M 转换为稀疏格式
    elif fmt == 'sparse':
        M = M.to_sparse()
    # 如果格式不是 'dense' 也不是 'sparse'，抛出值错误异常并指定格式信息（用于测试覆盖率）
    else:
        raise ValueError(f'Unknown format: {fmt}') # pragma: no cover
    # 返回转换后的矩阵 M 和原始格式 old_fmt
    return M, old_fmt
# 这是四种基本实现方法，我们希望在它们之间进行选择：

# 使用 Gauss-Jordan 消元法和除法计算行简化阶梯形矩阵（RREF）
def _dm_rref_GJ(M):
    if M.rep.fmt == 'sparse':
        return _dm_rref_GJ_sparse(M)  # 如果 M 的表示格式为稀疏格式，则调用稀疏矩阵的 Gauss-Jordan 消元法
    else:
        return _dm_rref_GJ_dense(M)   # 否则调用密集矩阵的 Gauss-Jordan 消元法

# 使用无分数的 Gauss-Jordan 消元法计算行简化阶梯形矩阵（RREF）
def _dm_rref_den_FF(M):
    if M.rep.fmt == 'sparse':
        return _dm_rref_den_FF_sparse(M)  # 如果 M 的表示格式为稀疏格式，则调用稀疏矩阵的无分数 Gauss-Jordan 消元法
    else:
        return _dm_rref_den_FF_dense(M)   # 否则调用密集矩阵的无分数 Gauss-Jordan 消元法

# 使用稀疏矩阵的 Gauss-Jordan 消元法计算行简化阶梯形矩阵（RREF）
def _dm_rref_GJ_sparse(M):
    M_rref_d, pivots, _ = sdm_irref(M.rep)
    M_rref_sdm = SDM(M_rref_d, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_sdm), pivots

# 使用密集矩阵的 Gauss-Jordan 消元法计算行简化阶梯形矩阵（RREF）
def _dm_rref_GJ_dense(M):
    partial_pivot = M.domain.is_RR or M.domain.is_CC
    ddm = M.rep.to_ddm().copy()
    pivots = ddm_irref(ddm, _partial_pivot=partial_pivot)
    M_rref_ddm = DDM(ddm, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_ddm.to_dfm_or_ddm()), pivots

# 使用稀疏矩阵的无分数 Gauss-Jordan 消元法计算行简化阶梯形矩阵（RREF）
def _dm_rref_den_FF_sparse(M):
    M_rref_d, den, pivots = sdm_rref_den(M.rep, M.domain)
    M_rref_sdm = SDM(M_rref_d, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_sdm), den, pivots

# 使用密集矩阵的无分数 Gauss-Jordan 消元法计算行简化阶梯形矩阵（RREF）
def _dm_rref_den_FF_dense(M):
    ddm = M.rep.to_ddm().copy()
    den, pivots = ddm_irref_den(ddm, M.domain)
    M_rref_ddm = DDM(ddm, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_ddm.to_dfm_or_ddm()), den, pivots

# 根据指定的方法选择计算矩阵 M 的行简化阶梯形矩阵（RREF）的最快方法
def _dm_rref_choose_method(M, method, *, denominator=False):
    if method != 'auto':
        if method.endswith('_dense'):
            method = method[:-len('_dense')]  # 如果方法名以 '_dense' 结尾，则去掉 '_dense' 后缀
            use_fmt = 'dense'  # 使用密集表示格式
        else:
            use_fmt = 'sparse'  # 否则使用稀疏表示格式
    else:
        # 如果不满足以上任何条件，则选择稠密格式 (dense)，因为稀疏实现总是更快的
        use_fmt = 'sparse'

        # 获取矩阵 M 的定义域
        K = M.domain

        # 如果定义域是整数环 ZZ
        if K.is_ZZ:
            # 根据整数环 ZZ 选择最佳的行简化方法
            method = _dm_rref_choose_method_ZZ(M, denominator=denominator)
        # 如果定义域是有理数域 QQ
        elif K.is_QQ:
            # 根据有理数域 QQ 选择最佳的行简化方法
            method = _dm_rref_choose_method_QQ(M, denominator=denominator)
        # 如果定义域是实数域 RR 或复数域 CC
        elif K.is_RR or K.is_CC:
            # TODO: 添加对稀疏实现的部分主元支持
            method = 'GJ'
            use_fmt = 'dense'
        # 如果定义域是扩展域 EX，并且矩阵的表示格式是稠密 dense，且没有分母
        elif K.is_EX and M.rep.fmt == 'dense' and not denominator:
            # 对于扩展域 EX，不要切换到稀疏实现，因为其域没有适当的规范化，而稀疏实现会导致在不同顺序进行算术运算时得到等效但不相同的结果。
            # 具体来说，当使用稀疏实现时，test_issue_23718 会得到更复杂的表达式。
            # 目前，如果矩阵已经是稠密的，我们在扩展域 EX 中仍然使用稠密实现。
            method = 'GJ'
            use_fmt = 'dense'
        else:
            # 这显然是次优的。需要更多工作来确定在不同定义域上计算 RREF 的最佳方法。
            # 如果有分母，则选择有限域 (FF) 方法，否则选择高斯-约当 (GJ) 方法
            if denominator:
                method = 'FF'
            else:
                method = 'GJ'

    # 返回选择的方法和使用的格式
    return method, use_fmt
# 选择在 QQ 域上计算 RREF 的最快方法
def _dm_rref_choose_method_QQ(M, *, denominator=False):
    """Choose the fastest method for computing RREF over QQ."""
    # 计算矩阵 M 的行密度和列数
    density, _, ncols = _dm_row_density(M)

    # 对于稀疏矩阵，无论如何都使用 QQ 域上的高斯-约当消元
    if density < min(5, ncols/2):
        return 'GJ'

    # 比较分母的最小公倍数的比特长度和分子的最大比特长度
    #
    # 阈值是经验性的：如果清除分母导致分子矩阵的比特长度是当前分子的5倍，
    # 则我们倾向于使用 QQ 域上的 RREF。
    numers, denoms = _dm_QQ_numers_denoms(M)
    numer_bits = max([n.bit_length() for n in numers], default=1)

    denom_lcm = ZZ.one
    for d in denoms:
        # 计算分母的最小公倍数
        denom_lcm = ZZ.lcm(denom_lcm, d)
        if denom_lcm.bit_length() > 5*numer_bits:
            return 'GJ'

    # 如果到达此处，表示矩阵是稠密的，并且分母的最小公倍数相对于分子并不太大。
    # 对于特别小的分母，清除它们并使用 QQ 域上的无分数高斯-约当消元最快。
    # 特别小的分母情况很常见，因为用户输入中经常会出现像 1/2 或 1/3 这样的小分数。
    if denom_lcm.bit_length() < 50:
        return 'CD'
    else:
        return 'FF'


# 选择在 ZZ 域上计算 RREF 的最快方法
def _dm_rref_choose_method_ZZ(M, *, denominator=False):
    """Choose the fastest method for computing RREF over ZZ."""
    # 对于非常稀疏的矩阵和低比特数，使用 QQ 域上的高斯-约当消元比 ZZ 域上的无分数高斯-约当消元更快。
    # 对于非常密集的矩阵和高比特数，使用 ZZ 域上的无分数高斯-约当消元比 QQ 域上的高斯-约当消元更快。
    # 这两种极端情况需要不同的处理，因为它们导致不同的渐近复杂度。
    # 在这两个极端情况之间，我们需要一个阈值来决定使用哪种方法。这个阈值是通过对随机矩阵进行方法定时来确定的。

    # 使用经验定时的缺点是未来的优化可能会改变相对速度，因此这可能很快过时。
    # 主要的目标是确保极端情况下的渐近复杂度正确，因此阈值的精确值希望不是太重要。
    # 经验确定的参数。
    PARAM = 10000

    # 首先计算密度。这是每行平均非零条目的数量，
    # 但仅计算至少有一个非零条目的行，因为RREF可以忽略完全为零的行。
    density, nrows_nz, ncols = _dm_row_density(M)

    # 对于小矩阵，如果超过一半的条目为零，则使用QQ。
    if nrows_nz < 10:
        if density < ncols/2:
            return 'GJ'
        else:
            return 'FF'

    # 下面的条件语句是公式的简写。
    if density < 5:
        return 'GJ'
    elif density > 5 + PARAM/nrows_nz:
        return 'FF'  # 用于覆盖测试，不计入测试覆盖率统计。

    # 计算矩阵元素的最大位数。
    elements = _dm_elements(M)
    bits = max([e.bit_length() for e in elements], default=1)

    # 宽度参数。对于方阵或长矩阵，此参数为1，但对于宽矩阵大于1。
    wideness = max(1, 2/3*ncols/nrows_nz)

    # 计算最大密度阈值。
    max_density = (5 + PARAM/(nrows_nz*bits**2)) * wideness

    if density < max_density:
        return 'GJ'
    else:
        return 'FF'
# 计算稀疏矩阵的密度指标。
# 定义“密度”为每行的非零条目的平均数，忽略全零行。RREF（行最简形式）可以忽略全零行，因此它们被排除在外。
# 根据定义，“密度 d >= 1”，但我们将零矩阵定义为“密度 d = 0”。

def _dm_row_density(M):
    """Density measure for sparse matrices.

    Defines the "density", ``d`` as the average number of non-zero entries per
    row except ignoring rows that are fully zero. RREF can ignore fully zero
    rows so they are excluded. By definition ``d >= 1`` except that we define
    ``d = 0`` for the zero matrix.

    Returns ``(density, nrows_nz, ncols)`` where ``nrows_nz`` counts the number
    of nonzero rows and ``ncols`` is the number of columns.
    """
    
    # 获取矩阵 M 的列数
    ncols = M.shape[1]
    
    # 将矩阵 M 转换为 SDM 字典-字典表示形式，并获取非零行的列表
    rows_nz = M.rep.to_sdm().values()
    
    # 如果没有非零行，则返回密度为 0，非零行数为 0，列数为 ncols
    if not rows_nz:
        return 0, 0, ncols
    else:
        # 计算非零行的数量
        nrows_nz = len(rows_nz)
        
        # 计算密度，即每个非零行的平均条目数除以非零行的总数
        density = sum(map(len, rows_nz)) / nrows_nz
        
        # 返回密度、非零行数、列数的元组
        return density, nrows_nz, ncols


# 返回 DomainMatrix 的非零元素列表
def _dm_elements(M):
    """Return nonzero elements of a DomainMatrix."""
    elements, _ = M.to_flat_nz()
    return elements


# 返回 DomainMatrix 在 QQ（有理数域）上的分子和分母列表
def _dm_QQ_numers_denoms(Mq):
    """Returns the numerators and denominators of a DomainMatrix over QQ."""
    # 获取 DomainMatrix 的非零元素列表
    elements = _dm_elements(Mq)
    
    # 提取每个元素的分子，构成分子列表
    numers = [e.numerator for e in elements]
    
    # 提取每个元素的分母，构成分母列表
    denoms = [e.denominator for e in elements]
    
    # 返回分子和分母列表
    return numers, denoms


# 尝试将 DomainMatrix 转换为域（field），如果可能的话
def _to_field(M):
    """Convert a DomainMatrix to a field if possible."""
    # 获取 DomainMatrix 的定义域
    K = M.domain
    
    # 如果定义域 K 可以关联到一个域（field），则将 DomainMatrix 转换为域并返回
    if K.has_assoc_Field:
        return M.to_field()
    else:
        # 否则，直接返回 DomainMatrix 本身
        return M
```