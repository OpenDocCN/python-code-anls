# `D:\src\scipysrc\scipy\scipy\spatial\transform\_rotation_spline.py`

```
    # 计算角速度到旋转向量导数的变换矩阵

    norm = np.linalg.norm(rotvecs, axis=1)
    # 计算旋转向量的模长

    k = np.empty_like(norm)
    # 创建一个与模长相同形状的空数组

    mask = norm > 1e-4
    # 创建一个布尔掩码，用于标识模长大于1e-4的情况

    nm = norm[mask]
    # 提取符合条件的模长值

    k[mask] = (1 - 0.5 * nm / np.tan(0.5 * nm)) / nm**2
    # 对于符合条件的模长值，计算对应的系数k

    mask = ~mask
    # 取反操作，获取模长小于等于1e-4的情况的掩码

    nm = norm[mask]
    # 提取符合新条件的模长值

    k[mask] = 1/12 + 1/720 * nm**2
    # 对于符合新条件的模长值，计算对应的系数k

    skew = _create_skew_matrix(rotvecs)
    # 使用给定的旋转向量创建其对应的斜对称矩阵

    result = np.empty((len(rotvecs), 3, 3))
    # 创建一个空的三维数组，用于存储结果矩阵

    result[:] = np.identity(3)
    # 将结果数组初始化为3x3单位矩阵的副本

    result[:] += 0.5 * skew
    # 将斜对称矩阵的0.5倍加到结果矩阵上

    result[:] += k[:, None, None] * np.matmul(skew, skew)
    # 将计算得到的系数k与斜对称矩阵乘积的乘积添加到结果矩阵上

    return result
    # 返回计算得到的结果矩阵
    # 计算旋转向量数组 `rotvecs` 的欧几里得范数，得到长度为 n 的向量
    norm = np.linalg.norm(rotvecs, axis=1)
    
    # 计算 `rotvecs` 和 `rotvecs_dot` 逐元素相乘后的和，形成长度为 n 的向量
    dp = np.sum(rotvecs * rotvecs_dot, axis=1)
    
    # 计算 `rotvecs` 和 `rotvecs_dot` 的叉乘，形成形状为 (n, 3) 的数组
    cp = np.cross(rotvecs, rotvecs_dot)
    
    # 计算 `rotvecs` 和 `cp` 的叉乘，形成形状为 (n, 3) 的数组
    ccp = np.cross(rotvecs, cp)
    
    # 计算 `rotvecs_dot` 和 `cp` 的叉乘，形成形状为 (n, 3) 的数组
    dccp = np.cross(rotvecs_dot, cp)

    # 创建一个与 `norm` 形状相同的空数组 `k1`
    k1 = np.empty_like(norm)
    
    # 创建一个与 `norm` 形状相同的空数组 `k2`
    k2 = np.empty_like(norm)
    
    # 创建一个与 `norm` 形状相同的空数组 `k3`
    k3 = np.empty_like(norm)

    # 使用条件 `norm > 1e-4` 创建一个布尔掩码 `mask`
    mask = norm > 1e-4
    
    # 根据掩码选择 `norm` 中大于 `1e-4` 的部分赋值给 `nm`
    nm = norm[mask]
    
    # 根据给定的公式计算 `k1` 数组中的值
    k1[mask] = (-nm * np.sin(nm) - 2 * (np.cos(nm) - 1)) / nm ** 4
    
    # 根据给定的公式计算 `k2` 数组中的值
    k2[mask] = (-2 * nm + 3 * np.sin(nm) - nm * np.cos(nm)) / nm ** 5
    
    # 根据给定的公式计算 `k3` 数组中的值
    k3[mask] = (nm - np.sin(nm)) / nm ** 3

    # 使用条件 `~mask` 创建一个布尔掩码
    mask = ~mask
    
    # 根据掩码选择 `norm` 中小于等于 `1e-4` 的部分赋值给 `nm`
    nm = norm[mask]
    
    # 根据给定的公式计算 `k1` 数组中的值
    k1[mask] = 1/12 - nm ** 2 / 180
    
    # 根据给定的公式计算 `k2` 数组中的值
    k2[mask] = -1/60 + nm ** 2 / 12604
    
    # 根据给定的公式计算 `k3` 数组中的值
    k3[mask] = 1/6 - nm ** 2 / 120

    # 将 `dp` 变形为列向量
    dp = dp[:, None]
    
    # 将 `k1` 变形为列向量
    k1 = k1[:, None]
    
    # 将 `k2` 变形为列向量
    k2 = k2[:, None]
    
    # 将 `k3` 变形为列向量
    k3 = k3[:, None]

    # 计算最终结果并返回
    return dp * (k1 * cp + k2 * ccp) + k3 * dccp
def _compute_angular_rate(rotvecs, rotvecs_dot):
    """计算给定旋转向量及其导数的角速率。

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        一组旋转向量。
    rotvecs_dot : ndarray, shape (n, 3)
        旋转向量的导数集合。

    Returns
    -------
    ndarray, shape (n, 3)
        角速率数组。
    """
    return _matrix_vector_product_of_stacks(
        _rotvec_dot_to_angular_rate_matrix(rotvecs), rotvecs_dot)


def _compute_angular_acceleration(rotvecs, rotvecs_dot, rotvecs_dot_dot):
    """计算给定旋转向量及其导数的角加速度。

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        一组旋转向量。
    rotvecs_dot : ndarray, shape (n, 3)
        旋转向量的导数集合。
    rotvecs_dot_dot : ndarray, shape (n, 3)
        旋转向量的二阶导数集合。

    Returns
    -------
    ndarray, shape (n, 3)
        角加速度数组。
    """
    return (_compute_angular_rate(rotvecs, rotvecs_dot_dot) +
            _angular_acceleration_nonlinear_term(rotvecs, rotvecs_dot))


def _create_block_3_diagonal_matrix(A, B, d):
    """创建一个三对角块矩阵作为带状矩阵。

    矩阵具有以下结构：

        DB...
        ADB..
        .ADB.
        ..ADB
        ...AD

    其中 A、B 和 D 均为 3x3 的矩阵，而 D 矩阵为 d * I。

    Parameters
    ----------
    A : ndarray, shape (n, 3, 3)
        A 块的堆叠。
    B : ndarray, shape (n, 3, 3)
        B 块的堆叠。
    d : ndarray, shape (n + 1,)
        对角块的值。

    Returns
    -------
    ndarray, shape (11, 3 * (n + 1))
        作为 `scipy.linalg.solve_banded` 使用的带状形式的矩阵。
    """
    ind = np.arange(3)
    ind_blocks = np.arange(len(A))

    A_i = np.empty_like(A, dtype=int)
    A_i[:] = ind[:, None]
    A_i += 3 * (1 + ind_blocks[:, None, None])

    A_j = np.empty_like(A, dtype=int)
    A_j[:] = ind
    A_j += 3 * ind_blocks[:, None, None]

    B_i = np.empty_like(B, dtype=int)
    B_i[:] = ind[:, None]
    B_i += 3 * ind_blocks[:, None, None]

    B_j = np.empty_like(B, dtype=int)
    B_j[:] = ind
    B_j += 3 * (1 + ind_blocks[:, None, None])

    diag_i = diag_j = np.arange(3 * len(d))
    i = np.hstack((A_i.ravel(), B_i.ravel(), diag_i))
    j = np.hstack((A_j.ravel(), B_j.ravel(), diag_j))
    values = np.hstack((A.ravel(), B.ravel(), np.repeat(d, 3)))

    u = 5
    l = 5
    result = np.zeros((u + l + 1, 3 * len(d)))
    result[u + i - j, j] = values
    return result


这些代码片段为给定的函数和类提供了详细的注释，包括每个函数的参数说明、返回值说明以及类的简要描述。
    # 角速度求解器的参数设置：最大迭代次数
    MAX_ITER = 10
    # 角速度求解器的参数设置：迭代收敛容差
    TOL = 1e-9
    # 解决角速率问题的方法，计算角速率的更新过程
    def _solve_for_angular_rates(self, dt, angular_rates, rotvecs):
        # 复制初始角速率作为第一个角速率
        angular_rate_first = angular_rates[0].copy()

        # 计算角速率到旋转向量导数的转换矩阵
        A = _angular_rate_to_rotvec_dot_matrix(rotvecs)
        # 计算旋转向量导数到角速率的转换矩阵的逆
        A_inv = _rotvec_dot_to_angular_rate_matrix(rotvecs)
        # 创建一个块对角线矩阵 M
        M = _create_block_3_diagonal_matrix(
            2 * A_inv[1:-1] / dt[1:-1, None, None],  # 中间部分的逆转换矩阵乘以2除以时间步长
            2 * A[1:-1] / dt[1:-1, None, None],      # 中间部分的转换矩阵乘以2除以时间步长
            4 * (1 / dt[:-1] + 1 / dt[1:]))           # 对角线部分乘以4，是时间步长的倒数和的两倍

        # 计算初始的 b0 向量
        b0 = 6 * (rotvecs[:-1] * dt[:-1, None] ** -2 +
                  rotvecs[1:] * dt[1:, None] ** -2)
        # 调整 b0 的第一个和最后一个元素
        b0[0] -= 2 / dt[0] * A_inv[0].dot(angular_rate_first)
        b0[-1] -= 2 / dt[-1] * A[-1].dot(angular_rates[-1])

        # 迭代求解角速率
        for iteration in range(self.MAX_ITER):
            # 计算当前角速率对应的旋转向量导数
            rotvecs_dot = _matrix_vector_product_of_stacks(A, angular_rates)
            # 计算非线性项的角加速度
            delta_beta = _angular_acceleration_nonlinear_term(
                rotvecs[:-1], rotvecs_dot[:-1])
            # 更新 b 向量
            b = b0 - delta_beta
            # 使用带状矩阵求解线性方程组，得到新的角速率
            angular_rates_new = solve_banded((5, 5), M, b.ravel())
            angular_rates_new = angular_rates_new.reshape((-1, 3))

            # 计算当前解的误差
            delta = np.abs(angular_rates_new - angular_rates[:-1])
            # 更新角速率
            angular_rates[:-1] = angular_rates_new
            # 如果误差满足收敛条件，则跳出循环
            if np.all(delta < self.TOL * (1 + np.abs(angular_rates_new))):
                break

        # 最终计算旋转向量导数，并将初始角速率放在最前面
        rotvecs_dot = _matrix_vector_product_of_stacks(A, angular_rates)
        angular_rates = np.vstack((angular_rate_first, angular_rates[:-1]))

        # 返回更新后的角速率和最终的旋转向量导数
        return angular_rates, rotvecs_dot
    def __init__(self, times, rotations):
        # 导入 PPoly 类从 scipy.interpolate 模块
        from scipy.interpolate import PPoly
        
        # 检查 rotations 是否为单一旋转，若是则抛出数值错误
        if rotations.single:
            raise ValueError("`rotations` must be a sequence of rotations.")
        
        # 检查 rotations 的长度是否至少为 2，若不是则抛出数值错误
        if len(rotations) == 1:
            raise ValueError("`rotations` must contain at least 2 rotations.")
        
        # 将 times 转换为浮点数的 NumPy 数组
        times = np.asarray(times, dtype=float)
        # 检查 times 是否为一维数组，若不是则抛出数值错误
        if times.ndim != 1:
            raise ValueError("`times` must be 1-dimensional.")
        
        # 检查 rotations 和 times 的长度是否相等，若不相等则抛出数值错误
        if len(times) != len(rotations):
            raise ValueError("Expected number of rotations to be equal to "
                             "number of timestamps given, got {} rotations "
                             "and {} timestamps."
                             .format(len(rotations), len(times)))
        
        # 计算时间间隔 dt
        dt = np.diff(times)
        # 检查 dt 是否有非正值，若有则抛出数值错误
        if np.any(dt <= 0):
            raise ValueError("Values in `times` must be in a strictly "
                             "increasing order.")
        
        # 计算相邻旋转之间的旋转向量和角速率
        rotvecs = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
        angular_rates = rotvecs / dt[:, None]
        
        # 若 rotations 的长度为 2，则直接将角速率赋值给 rotvecs_dot
        if len(rotations) == 2:
            rotvecs_dot = angular_rates
        else:
            # 否则调用 _solve_for_angular_rates 方法计算角速率和 rotvecs_dot
            angular_rates, rotvecs_dot = self._solve_for_angular_rates(
                dt, angular_rates, rotvecs)
        
        # 将 dt 转换为列向量
        dt = dt[:, None]
        # 计算系数 coeff
        coeff = np.empty((4, len(times) - 1, 3))
        coeff[0] = (-2 * rotvecs + dt * angular_rates
                    + dt * rotvecs_dot) / dt ** 3
        coeff[1] = (3 * rotvecs - 2 * dt * angular_rates
                    - dt * rotvecs_dot) / dt ** 2
        coeff[2] = angular_rates
        coeff[3] = 0
        
        # 将计算得到的 times、rotations 和 PPoly 对象赋值给类的属性
        self.times = times
        self.rotations = rotations
        self.interpolator = PPoly(coeff, times)
    def __call__(self, times, order=0):
        """计算插值值。

        Parameters
        ----------
        times : float or array_like
            兴趣时刻。
        order : {0, 1, 2}, optional
            微分的阶数：

                * 0（默认）：返回旋转
                * 1：返回弧度/秒的角速率
                * 2：返回弧度/秒²的角加速度

        Returns
        -------
        插值的旋转、角速率或角加速度。
        """
        if order not in [0, 1, 2]:
            raise ValueError("`order` 必须为 0、1 或 2。")

        times = np.asarray(times, dtype=float)
        if times.ndim > 1:
            raise ValueError("`times` 必须至多为 1 维。")

        singe_time = times.ndim == 0
        times = np.atleast_1d(times)

        # 使用插值器计算旋转向量
        rotvecs = self.interpolator(times)
        if order == 0:
            # 查找 `times` 在 `self.times` 中的位置
            index = np.searchsorted(self.times, times, side='right')
            index -= 1
            index[index < 0] = 0
            n_segments = len(self.times) - 1
            index[index > n_segments - 1] = n_segments - 1
            # 计算插值的旋转
            result = self.rotations[index] * Rotation.from_rotvec(rotvecs)
        elif order == 1:
            # 计算角速率
            rotvecs_dot = self.interpolator(times, 1)
            result = _compute_angular_rate(rotvecs, rotvecs_dot)
        elif order == 2:
            # 计算角加速度
            rotvecs_dot = self.interpolator(times, 1)
            rotvecs_dot_dot = self.interpolator(times, 2)
            result = _compute_angular_acceleration(rotvecs, rotvecs_dot,
                                                   rotvecs_dot_dot)
        else:
            assert False

        # 如果 `times` 是单个值，返回结果中的第一个元素
        if singe_time:
            result = result[0]

        return result
```