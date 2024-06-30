# `D:\src\scipysrc\scipy\scipy\optimize\_lsq\dogbox.py`

```
    """Compute LinearOperator to use in LSMR by dogbox algorithm.

    `active_set` mask is used to excluded active variables from computations
    of matrix-vector products.
    """
    # 将 `Jop` 和 `d` 转换为线性操作符，以便在 LSMR 方法中使用
    return LinearOperator(
        (Jop.shape[0], Jop.shape[1]),
        matvec=lambda v: Jop.matvec(v) if np.any(active_set) else Jop.matvec(v) + d * v,
        rmatvec=lambda v: Jop.rmatvec(v) if np.any(active_set) else Jop.rmatvec(v) + d * v
    )
    """
    创建一个线性算子对象并返回
    
    m, n = Jop.shape
    获取矩阵 Jop 的行数 m 和列数 n
    
    def matvec(x):
        定义矩阵向量乘法函数 matvec(x)，其中：
        x_free = x.ravel().copy()
        将输入向量 x 展平并复制一份，存储在 x_free 中
        x_free[active_set] = 0
        将活跃集 active_set 中对应的位置置为 0
        return Jop.matvec(x * d)
        返回 Jop 对 x 乘以 d 的结果
    
    def rmatvec(x):
        定义反向矩阵向量乘法函数 rmatvec(x)，其中：
        r = d * Jop.rmatvec(x)
        计算 Jop 对 x 的反向乘积，乘以常数 d
        r[active_set] = 0
        将活跃集 active_set 中对应的位置置为 0
        return r
        返回处理后的结果向量 r
    
    返回一个 LinearOperator 对象，其尺寸为 (m, n)，使用上述定义的 matvec 和 rmatvec 函数，数据类型为 float
    """
# 初始化变量 f 和 f_true，分别用于保存原始的函数值和拷贝的函数值
f = f0
f_true = f.copy()

# 初始化变量 nfev 为 1，表示已经进行了一次函数评估
nfev = 1

# 将传入的 Jacobian 矩阵 J0 赋值给变量 J，并初始化 njev 为 1，表示已经进行了一次 Jacobian 矩阵的评估
J = J0
njev = 1

# 如果有损失函数 loss_function，则计算损失函数的值 rho，并基于 rho 对 J 和 f 进行缩放处理
if loss_function is not None:
    rho = loss_function(f)
    # 计算损失函数的成本，这里假设 rho 是一个数组，rho[0] 是损失函数的值
    cost = 0.5 * np.sum(rho[0])
    # 根据损失函数的值 rho 缩放 J 和 f
    J, f = scale_for_robust_loss_function(J, f, rho)
    # 如果没有特定的终止状态，则将 cost 设置为函数值 f 的二范数的一半
    else:
        cost = 0.5 * np.dot(f, f)

    # 计算梯度 g = ∇J(x)，其中 J 是目标函数，f 是当前点 x 处的函数值
    g = compute_grad(J, f)

    # 判断是否需要对 Jacobi 矩阵进行缩放
    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        # 计算 Jacobi 矩阵的缩放因子和其逆
        scale, scale_inv = compute_jac_scale(J)
    else:
        # 否则直接使用给定的缩放因子 x_scale，并计算其倒数
        scale, scale_inv = x_scale, 1 / x_scale

    # 计算 x0 经过缩放后的无穷范数
    Delta = norm(x0 * scale_inv, ord=np.inf)
    if Delta == 0:
        Delta = 1.0  # 如果 Delta 为零，将其设置为 1.0，避免除以零错误

    # 初始化一个与 x0 形状相同的数组，用于记录每个变量是否处于边界上
    on_bound = np.zeros_like(x0, dtype=int)
    on_bound[np.equal(x0, lb)] = -1  # 标记处于下界 lb 的变量为 -1
    on_bound[np.equal(x0, ub)] = 1   # 标记处于上界 ub 的变量为 1

    # 初始化迭代的起始点 x，并准备存储每步的步长信息
    x = x0
    step = np.empty_like(x0)

    # 如果未指定最大函数调用次数 max_nfev，则设定为 x0 的长度乘以 100
    if max_nfev is None:
        max_nfev = x0.size * 100

    # 初始化终止状态、迭代次数、步长范数和实际减少量
    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    # 如果设置为详细输出模式 verbose == 2，则打印非线性优化的标题头信息
    if verbose == 2:
        print_header_nonlinear()

    # 如果终止状态仍然为 None，则将其设置为 0
    if termination_status is None:
        termination_status = 0

    # 返回优化结果对象 OptimizeResult，包括最优解 x、损失值 cost、真实的目标函数值 f_true、
    # Jacobi 矩阵 J、完整的梯度 g_full、最优性 g_norm、活跃变量掩码 on_bound、
    # 函数调用次数 nfev、Jacobi 矩阵计算次数 njev、以及终止状态 termination_status
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g_full, optimality=g_norm,
        active_mask=on_bound, nfev=nfev, njev=njev, status=termination_status)
```