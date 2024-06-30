# `D:\src\scipysrc\scipy\scipy\optimize\_remove_redundancy.py`

```
    """
    Routines for removing redundant (linearly dependent) equations from linear
    programming equality constraints.
    """
    # 导入必要的库和模块
    # 作者：Matt Haberland
    import numpy as np
    from scipy.linalg import svd
    from scipy.linalg.interpolative import interp_decomp
    import scipy
    from scipy.linalg.blas import dtrsm


    def _row_count(A):
        """
        Counts the number of nonzeros in each row of input array A.
        Nonzeros are defined as any element with absolute value greater than
        tol = 1e-13. This value should probably be an input to the function.

        Parameters
        ----------
        A : 2-D array
            An array representing a matrix

        Returns
        -------
        rowcount : 1-D array
            Number of nonzeros in each row of A

        """
        # 定义非零的阈值
        tol = 1e-13
        # 计算每行非零元素的数量
        return np.array((abs(A) > tol).sum(axis=1)).flatten()


    def _get_densest(A, eligibleRows):
        """
        Returns the index of the densest row of A. Ignores rows that are not
        eligible for consideration.

        Parameters
        ----------
        A : 2-D array
            An array representing a matrix
        eligibleRows : 1-D logical array
            Values indicate whether the corresponding row of A is eligible
            to be considered

        Returns
        -------
        i_densest : int
            Index of the densest row in A eligible for consideration

        """
        # 计算每行的非零元素数量
        rowCounts = _row_count(A)
        # 找出非零行中非零元素最多的行的索引
        return np.argmax(rowCounts * eligibleRows)


    def _remove_zero_rows(A, b):
        """
        Eliminates trivial equations from system of equations defined by Ax = b
        and identifies trivial infeasibilities

        Parameters
        ----------
        A : 2-D array
            An array representing the left-hand side of a system of equations
        b : 1-D array
            An array representing the right-hand side of a system of equations

        Returns
        -------
        A : 2-D array
            An array representing the left-hand side of a system of equations
        b : 1-D array
            An array representing the right-hand side of a system of equations
        status: int
            An integer indicating the status of the removal operation
            0: No infeasibility identified
            2: Trivially infeasible
        message : str
            A string descriptor of the exit status of the optimization.

        """
        # 初始状态为未发现不可行性
        status = 0
        message = ""
        # 找出所有零行的索引
        i_zero = _row_count(A) == 0
        # 移除 A 和 b 中的零行
        A = A[np.logical_not(i_zero), :]
        # 检查是否 b 中对应零行的元素不为零，表示问题不可行
        if not np.allclose(b[i_zero], 0):
            status = 2
            message = "There is a zero row in A_eq with a nonzero corresponding " \
                      "entry in b_eq. The problem is infeasible."
        b = b[np.logical_not(i_zero)]
        return A, b, status, message


    def bg_update_dense(plu, perm_r, v, j):
        LU, p = plu

        # 对置换后的 v 进行操作
        vperm = v[perm_r]
        # 利用 BLAS 中的矩阵求解方法解 LUx = vperm
        u = dtrsm(1, LU, vperm, lower=1, diag=1)
        # 更新 LU 分解的矩阵
        LU[:j+1, j] = u[:j+1]
        l = u[j+1:]
        piv = LU[j, j]
        LU[j+1:, j] += (l/piv)
        return LU, p


    def _remove_redundancy_pivot_dense(A, rhs, true_rank=None):
        """
        Eliminates redundant equations from system of equations defined by Ax = b
        and identifies infeasibilities.

        Parameters
        ----------
        A : 2-D array
            An array representing the left-hand side of a system of equations
        rhs : 1-D array
            An array representing the right-hand side of a system of equations
        true_rank : int or None, optional
            The true rank of matrix A. If None, it will be computed internally.

        Returns
        -------
        A : 2-D array
            An array representing the left-hand side of a system of equations
        rhs : 1-D array
            An array representing the right-hand side of a system of equations
        status : int
            An integer indicating the status of the removal operation
            0: No infeasibility identified
            2: Trivially infeasible
        message : str
            A string descriptor of the exit status of the optimization.
        """
        # To be continued...
    # 设置误差容限，用于判断数值是否为零
    tolapiv = 1e-8
    # 设置主元的容限，用于判断数值是否为零
    tolprimal = 1e-8
    # 初始化系统状态为0（表示没有发现不可行情况）
    status = 0
    # 初始化消息为空字符串
    message = ""
    # 描述不一致情况的文本，指示出现冗余约束或不一致约束的可能性
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")
    
    # 调用函数 _remove_zero_rows，处理 A 和 rhs，返回处理后的 A、rhs、status 和 message
    A, rhs, status, message = _remove_zero_rows(A, rhs)
    
    # 如果处理后的状态不为0（即发现了不可行情况），直接返回处理后的结果
    if status != 0:
        return A, rhs, status, message
    
    # 获取矩阵 A 的行数 m 和列数 n
    m, n = A.shape
    
    # v 列表存储人工变量的列索引
    v = list(range(m))
    # b 列表存储基变量的列索引，与 v 相同初始值
    b = list(v)
    # d 列表存储依赖行的索引，初始为空列表
    d = []
    # 置换行的索引，初始值为 None
    perm_r = None
    
    # 将原始矩阵 A 备份为 A_orig
    A_orig = A
    # 创建一个 m 行 (m + n) 列的全零矩阵 A，以 Fortran 风格存储
    A = np.zeros((m, m + n), order='F')
    # 将 A 对角线元素设为 1
    np.fill_diagonal(A, 1)
    # 将 A 的右侧部分赋值为 A_orig
    A[:, m:] = A_orig
    # 创建长度为 m 的全零数组 e
    e = np.zeros(m)
    
    # js_candidates 是长度为 n 的候选基变量列索引数组，从 m 开始递增
    js_candidates = np.arange(m, m+n, dtype=int)
    # js_mask 是长度为 n 的布尔数组，全部为 True，用于标记候选基变量列的有效性
    js_mask = np.ones(js_candidates.shape, dtype=bool)
    
    # 实现文献 [2] 中基本算法的改进版本，包括去除零行和 Bartels-Golub 更新思想
    # 单独去除列单元素不是很重要，因为该过程仅在原始问题的等式约束矩阵上执行，
    # 而不是在标准形式矩阵上执行，后者由于不等式约束的松弛变量可能会有更多列单元素
    # 如果矩阵是稀疏的，则“崩溃”初始基的想法只有在特定情况下才有用
    # 初始化 LU 分解为单位矩阵和顺序数组
    lu = np.eye(m, order='F'), np.arange(m)
    # 初始化行置换为 lu 分解后的行索引数组
    perm_r = lu[1]
    for i in v:
        # 设置 e[i] 为 1
        e[i] = 1
        # 如果 i > 0，设置 e[i-1] 为 0
        if i > 0:
            e[i-1] = 0

        try:  # 处理当 i==0 或出现条件不良的情况时的异常
            # 尝试从 b[i-1] 获取 j，并更新 LU 分解
            j = b[i-1]
            lu = bg_update_dense(lu, perm_r, A[:, j], i-1)
        except Exception:
            # 如果异常，重新进行 LU 分解
            lu = scipy.linalg.lu_factor(A[:, b])
            LU, p = lu
            perm_r = list(range(m))
            # 重新排列置换向量 perm_r
            for i1, i2 in enumerate(p):
                perm_r[i1], perm_r[i2] = perm_r[i2], perm_r[i1]

        # 解方程 lu * pi = e，返回 pi
        pi = scipy.linalg.lu_solve(lu, e, trans=1)

        # 选择候选列 js，每次处理 batch 个列
        js = js_candidates[js_mask]
        batch = 50

        # 对列进行批处理，计算与 pi 的投影，并比较阈值 tolapiv
        # 这比逐个遍历列更快一点
        for j_index in range(0, len(js), batch):
            j_indices = js[j_index: min(j_index+batch, len(js))]

            # 计算投影并检查是否超过阈值 tolapiv
            c = abs(A[:, j_indices].transpose().dot(pi))
            if (c > tolapiv).any():
                # 选择具有最大投影值的列 j
                j = js[j_index + np.argmax(c)]  # 非常独立的列
                b[i] = j
                # 更新 js_mask 中对应列的值为 False
                js_mask[j-m] = False
                break
        else:
            # 计算 bibar，检查是否不一致
            bibar = pi.T.dot(rhs.reshape(-1, 1))
            bnorm = np.linalg.norm(rhs)
            if abs(bibar)/(1+bnorm) > tolprimal:  # 不一致情况
                status = 2
                message = "inconsistent"
                return A_orig, rhs, status, message
            else:  # 依赖情况
                # 将当前 i 加入列表 d 中
                d.append(i)
                # 如果指定了 true_rank 且 d 的长度等于 m - true_rank，则找到所有冗余
                if true_rank is not None and len(d) == m - true_rank:
                    break   # 找到所有冗余

    # 保留集合中的元素，减去列表 d 中的元素
    keep = set(range(m))
    keep = list(keep - set(d))
    # 返回筛选后的 A_orig 和 rhs，以及状态和消息
    return A_orig[keep, :], rhs[keep], status, message
    # 设置用于判断冗余的阈值
    tolapiv = 1e-8
    tolprimal = 1e-8
    status = 0
    message = ""
    # 定义关于不一致性的信息
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")
    
    # 调用函数_remove_zero_rows来删除A中的零行，并更新状态和消息
    A, rhs, status, message = _remove_zero_rows(A, rhs)

    # 如果状态不为0，表示发现了冗余约束或者问题不可行，直接返回当前的A、rhs、状态和消息
    if status != 0:
        return A, rhs, status, message

    m, n = A.shape

    # v是基变量的列索引，用列表表示
    v = list(range(m))
    # b是基变量的列索引的副本，也用列表表示
    b = list(v)
    # k是结构变量的列索引，用集合表示，范围从m到m+n
    k = set(range(m, m+n))
    # d是依赖行的索引，初始化为空列表
    d = []

    # 将A_orig设为A的原始副本
    A_orig = A
    # 将A扩展为[m x (m+n)]的稀疏矩阵，左侧是单位矩阵的扩展
    A = scipy.sparse.hstack((scipy.sparse.eye(m), A)).tocsc()
    # 初始化e为长度为m的零向量
    e = np.zeros(m)

    # 实现文献[2]中的基本算法，使用了文献建议的一种改进（删除零行）
    # 删除列单例会更容易，但不如重要，因为此过程仅在原始问题的等式约束矩阵上执行
    # 并非在规范形式矩阵上，后者由于不等式约束的松弛变量会有更多的列单例
    # 对于“崩溃”初始基础的想法听起来很有用，但过程描述似乎假设了对主题的很多熟悉
    # 不太明确。我已经经历了足够多的麻烦才能使基本算法工作，所以我对尝试解释这个不感兴趣
    # （总的来说，这篇文章充满了错误和歧义 - 这很奇怪，因为安德森的其他论文都相当不错）
    # 对于给定的向量 v 中的每一个元素 i，执行以下操作
    for i in v:
        # 从矩阵 A 中选取列索引集合 b 对应的子矩阵 B
        B = A[:, b]

        # 构造一个长度为 v 的零向量 e，并将第 i 个元素设为 1
        e[i] = 1
        # 如果 i 大于 0，则将 e 中的第 i-1 个元素设为 0
        if i > 0:
            e[i-1] = 0

        # 使用 B 的转置求解线性系统 B^T * x = e，其中 x 是列向量 pi
        pi = scipy.sparse.linalg.spsolve(B.transpose(), e).reshape(-1, 1)

        # 计算不在 b 集合中的索引集合 k 与 b 的差集，并转换为列表 js
        js = list(k-set(b))  # 这不是效率最高的，但这不是时间消耗的关键...

        # 计算 A 的子矩阵 A[:, js] 与 pi 的矩阵-向量乘积的绝对值是否大于 tolapiv
        c = (np.abs(A[:, js].transpose().dot(pi)) > tolapiv).nonzero()[0]
        # 如果找到了一个非零的索引 c
        if len(c) > 0:  # 独立的情况
            # 选择第一个非零索引对应的列索引 j
            j = js[c[0]]
            # 替换 b 中的第 i 个元素为 j，用以代替人工列
            b[i] = j
        else:
            # 计算 bibar 和 rhs 的点积的绝对值，以及 rhs 的范数
            bibar = pi.T.dot(rhs.reshape(-1, 1))
            bnorm = np.linalg.norm(rhs)
            # 如果 bibar 的绝对值除以 (1 + bnorm) 大于 tolprimal
            if abs(bibar)/(1 + bnorm) > tolprimal:
                # 设置状态为 2，并返回原始矩阵 A_orig、右手边向量 rhs、状态和消息
                status = 2
                message = inconsistent
                return A_orig, rhs, status, message
            else:  # 依赖的情况
                # 将 i 添加到列表 d 中
                d.append(i)

    # 初始化保留索引集合 keep，包含所有行索引的集合
    keep = set(range(m))
    # 从 keep 中移除列表 d 中的索引，并转换为列表
    keep = list(keep - set(d))
    # 返回 A_orig 中 keep 行的子矩阵、rhs 中 keep 行的子向量、状态和消息
    return A_orig[keep, :], rhs[keep], status, message
# 从线性方程组 Ax = b 中消除冗余的方程，并识别不可行性。

def _remove_redundancy_svd(A, b):
    # 调用辅助函数 _remove_zero_rows 去除 A 和 b 中的零行，并返回处理后的 A, b, 状态和消息
    A, b, status, message = _remove_zero_rows(A, b)

    # 如果存在不可行性，直接返回处理后的 A, b, 状态和消息
    if status != 0:
        return A, b, status, message

    # 对 A 进行奇异值分解
    U, s, Vh = svd(A)

    # 计算机器精度
    eps = np.finfo(float).eps
    # 容差值取决于奇异值中最大值的乘积和 A 的形状
    tol = s.max() * max(A.shape) * eps

    # 获取 A 的行数 m 和列数 n
    m, n = A.shape
    # 如果 m <= n，则选择奇异值 s 的最小非零值；否则设为 0
    s_min = s[-1] if m <= n else 0

    # 这种算法在 nullspace 较小的情况下比 [2] 更快
    # 但是可以通过随机算法和稀疏实现进一步改进。
    # 它依赖于重复的奇异值分解来找到线性相关的行（由对应于零奇异值的 U 的列表示）。
    # 不幸的是，每次分解只能移除一行（尝试其他方式可能会导致问题）。
    # 如果我们可以像 sp.sparse.linalg.svds 那样做截断奇异值分解就好了，
    # 但是那个函数在找到接近零的奇异值时并不可靠。
    # 找到 A A^T 的最大特征值 L，然后通过幂迭代找到 -A A^T + L I 的最大特征值（及其关联的特征向量）
    # 在理论上也可以工作，但只有在 A A^T 的最小非零特征值接近最大非零特征值时才有效率。
    # 当最小的奇异值的绝对值小于容差(tol)时循环执行以下操作
    while abs(s_min) < tol:
        # 取矩阵U的最后一列向量，用于什么用途？是否要将其返回给用户以便问题可以从中排除？
        v = U[:, -1]  # TODO: return these so user can eliminate from problem?
        
        # 找出向量v中绝对值大于tol乘以10的6次方的元素所对应的行
        eligibleRows = np.abs(v) > tol * 10e6
        
        # 如果没有符合条件的行，或者存在v.dot(A)的绝对值大于容差(tol)的元素
        if not np.any(eligibleRows) or np.any(np.abs(v.dot(A)) > tol):
            status = 4
            # 设置消息，指出由于数值问题无法自动移除冗余的等式约束
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            break
        
        # 如果存在v.dot(b)的绝对值大于tol乘以100的元素
        if np.any(np.abs(v.dot(b)) > tol * 100):  # factor of 100 to fix 10038 and 10349
            status = 2
            # 设置消息，指出A_eq的行存在线性组合为零但b_eq相应的线性组合非零的情况，表明约束冲突导致问题不可行
            message = ("There is a linear combination of rows of A_eq that "
                       "results in zero, suggesting a redundant constraint. "
                       "However the same linear combination of b_eq is "
                       "nonzero, suggesting that the constraints conflict "
                       "and the problem is infeasible.")
            break
        
        # 找出稠密度最高的行索引
        i_remove = _get_densest(A, eligibleRows)
        
        # 删除数组A中指定行索引对应的行
        A = np.delete(A, i_remove, axis=0)
        
        # 删除数组b中指定索引对应的元素
        b = np.delete(b, i_remove)
        
        # 对更新后的A进行奇异值分解
        U, s, Vh = svd(A)
        
        # 获取更新后的A的行数m和列数n
        m, n = A.shape
        
        # 计算更新后的A的最小奇异值s_min
        s_min = s[-1] if m <= n else 0

    # 返回更新后的A、b、状态和消息
    return A, b, status, message
# 定义函数 _remove_redundancy_id，用于从方程组中消除冗余方程和识别不可行情况
def _remove_redundancy_id(A, rhs, rank=None, randomized=True):
    """Eliminates redundant equations from a system of equations.

    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    rank : int, optional
        The rank of A
    randomized: bool, optional
        True for randomized interpolative decomposition

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    """

    # 初始状态设为正常
    status = 0
    message = ""

    # 定义描述不一致情况的信息
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")

    # 调用函数 _remove_zero_rows，用于去除 A 中的零行及相应的 rhs 元素
    A, rhs, status, message = _remove_zero_rows(A, rhs)

    # 如果存在不一致情况，则直接返回当前状态和消息
    if status != 0:
        return A, rhs, status, message

    # 获取 A 的行数 m 和列数 n
    m, n = A.shape

    # 若未提供 rank 参数，则计算 A 的秩
    k = rank
    if rank is None:
        k = np.linalg.matrix_rank(A)

    # 对 A 的转置进行插值分解，得到独立行的索引 idx 和对应的投影 proj
    idx, proj = interp_decomp(A.T, k, rand=randomized)

    # idx 的前 k 个索引是独立行的索引，剩余的是 m-k 个依赖行的索引
    # proj 提供了 A2 剩余的 m-k 行的线性组合。如果 rhs 中相同的线性组合不能给出相应的 m-k 条目，则系统不一致，问题不可行。
    if not np.allclose(rhs[idx[:k]] @ proj, rhs[idx[k:]]):
        status = 2  # 将状态设置为不可行
        message = inconsistent  # 设置消息为不一致情况的描述

    # 对索引进行排序，因为其他冗余移除程序保留了原始顺序，测试代码也是基于此编写的
    idx = sorted(idx[:k])
    A2 = A[idx, :]  # 提取出 A 中独立行的子集 A2
    rhs2 = rhs[idx]  # 提取出 rhs 中对应的部分 rhs2
    return A2, rhs2, status, message
```