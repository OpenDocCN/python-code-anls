# `D:\src\scipysrc\scipy\scipy\interpolate\_rbfinterp_pythran.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算


def linear(r):
    return -r  # 返回线性函数 -r


def thin_plate_spline(r):
    if r == 0:
        return 0.0  # 如果 r 等于 0，则返回 0.0
    else:
        return r**2 * np.log(r)  # 否则返回 thin plate spline 函数的计算结果


def cubic(r):
    return r**3  # 返回立方函数 r^3


def quintic(r):
    return -r**5  # 返回五次函数 -r^5


def multiquadric(r):
    return -np.sqrt(r**2 + 1)  # 返回多项式型函数 -sqrt(r^2 + 1)


def inverse_multiquadric(r):
    return 1 / np.sqrt(r**2 + 1)  # 返回反多项式型函数 1 / sqrt(r^2 + 1)


def inverse_quadratic(r):
    return 1 / (r**2 + 1)  # 返回反二次型函数 1 / (r^2 + 1)


def gaussian(r):
    return np.exp(-r**2)  # 返回高斯函数 exp(-r^2)


NAME_TO_FUNC = {  # 函数名称到函数对象的映射字典
   "linear": linear,
   "thin_plate_spline": thin_plate_spline,
   "cubic": cubic,
   "quintic": quintic,
   "multiquadric": multiquadric,
   "inverse_multiquadric": inverse_multiquadric,
   "inverse_quadratic": inverse_quadratic,
   "gaussian": gaussian
   }


def kernel_vector(x, y, kernel_func, out):
    """Evaluate RBFs, with centers at `y`, at the point `x`."""
    for i in range(y.shape[0]):
        out[i] = kernel_func(np.linalg.norm(x - y[i]))  # 计算在点 x 处以 y 为中心的径向基函数值


def polynomial_vector(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at the point `x`."""
    for i in range(powers.shape[0]):
        out[i] = np.prod(x**powers[i])  # 计算在点 x 处指数由 powers 给出的各个单项式的乘积值


def kernel_matrix(x, kernel_func, out):
    """Evaluate RBFs, with centers at `x`, at `x`."""
    for i in range(x.shape[0]):
        for j in range(i+1):
            out[i, j] = kernel_func(np.linalg.norm(x[i] - x[j]))  # 计算在点 x[i] 处以 x[j] 为中心的径向基函数值
            out[j, i] = out[i, j]  # 对称性：填充矩阵的对角线以下部分


def polynomial_matrix(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at `x`."""
    for i in range(x.shape[0]):
        for j in range(powers.shape[0]):
            out[i, j] = np.prod(x[i]**powers[j])  # 计算在点 x[i] 处指数由 powers[j] 给出的各个单项式的乘积值


# pythran export _kernel_matrix(float[:, :], str)
def _kernel_matrix(x, kernel):
    """Return RBFs, with centers at `x`, evaluated at `x`."""
    out = np.empty((x.shape[0], x.shape[0]), dtype=float)  # 创建一个空的矩阵，用于存储径向基函数的计算结果
    kernel_func = NAME_TO_FUNC[kernel]  # 根据 kernel 名称获取对应的函数对象
    kernel_matrix(x, kernel_func, out)  # 调用 kernel_matrix 函数计算径向基函数的值
    return out  # 返回计算结果矩阵


# pythran export _polynomial_matrix(float[:, :], int[:, :])
def _polynomial_matrix(x, powers):
    """Return monomials, with exponents from `powers`, evaluated at `x`."""
    out = np.empty((x.shape[0], powers.shape[0]), dtype=float)  # 创建一个空的矩阵，用于存储单项式的计算结果
    polynomial_matrix(x, powers, out)  # 调用 polynomial_matrix 函数计算单项式的值
    return out  # 返回计算结果矩阵


# pythran export _build_system(float[:, :],
#                              float[:, :],
#                              float[:],
#                              str,
#                              float,
#                              int[:, :])
def _build_system(y, d, smoothing, kernel, epsilon, powers):
    """Build the system used to solve for the RBF interpolant coefficients.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.

    Returns
    -------
    ```
    """
    # 左手边矩阵，形状为 (P + R, P + R)，存储左侧计算结果
    lhs : (P + R, P + R) float ndarray
    # 右手边矩阵，形状为 (P + R, S)，存储右侧计算结果
    rhs : (P + R, S) float ndarray
    # 域偏移量，用于创建多项式矩阵
    shift : (N,) float ndarray
    # 域缩放量，用于创建多项式矩阵

    """
    # 计算矩阵 d 的行数
    p = d.shape[0]
    # 计算矩阵 d 的列数
    s = d.shape[1]
    # 计算多项式矩阵 powers 的行数
    r = powers.shape[0]
    # 从名称到函数的映射中获取指定 kernel 名称对应的函数
    kernel_func = NAME_TO_FUNC[kernel]

    # 计算 y 的每一列的最小值和最大值
    mins = np.min(y, axis=0)
    maxs = np.max(y, axis=0)
    # 计算域的平移量，将 y 映射到 [-1, 1] 的区间
    shift = (maxs + mins)/2
    # 计算域的缩放量，将 y 映射到 [-1, 1] 的区间
    scale = (maxs - mins)/2
    # 如果缩放量中存在零，说明可能所有点在某一维度上相同，避免除以零，将零替换为一
    scale[scale == 0.0] = 1.0

    # 将 y 缩放到 [-1, 1] 区间并乘以 epsilon
    yeps = y * epsilon
    yhat = (y - shift) / scale

    # 创建一个空的 p + r × p + r 的左手边矩阵，用于存储中间计算结果
    lhs = np.empty((p + r, p + r), dtype=float).T
    # 计算核矩阵 yeps 的贡献部分
    kernel_matrix(yeps, kernel_func, lhs[:p, :p])
    # 计算多项式矩阵 yhat 的贡献部分
    polynomial_matrix(yhat, powers, lhs[:p, p:])
    # 将左下角的矩阵转置到右上角
    lhs[p:, :p] = lhs[:p, p:].T
    # 右下角置零
    lhs[p:, p:] = 0.0
    # 添加平滑项到对角线上
    for i in range(p):
        lhs[i, i] += smoothing[i]

    # 创建一个空的 p + r × s 的右手边矩阵，用于存储中间计算结果
    rhs = np.empty((s, p + r), dtype=float).T
    # 将原始数据矩阵 d 存入右手边矩阵的前 p 行
    rhs[:p] = d
    # 右下角置零
    rhs[p:] = 0.0

    # 返回左手边矩阵 lhs，右手边矩阵 rhs，域的平移量 shift，域的缩放量 scale
    return lhs, rhs, shift, scale
# pythran export _build_evaluation_coefficients(float[:, :],
#                          float[:, :],
#                          str,
#                          float,
#                          int[:, :],
#                          float[:],
#                          float[:])
def _build_evaluation_coefficients(x, y, kernel, epsilon, powers,
                                   shift, scale):
    """Construct the coefficients needed to evaluate
    the RBF.

    Parameters
    ----------
    x : (Q, N) float ndarray
        Evaluation point coordinates.
    y : (P, N) float ndarray
        Data point coordinates.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.
    shift : (N,) float ndarray
        Shifts the polynomial domain for numerical stability.
    scale : (N,) float ndarray
        Scales the polynomial domain for numerical stability.

    Returns
    -------
    (Q, P + R) float ndarray
        Coefficients matrix for evaluating the RBF.

    """
    q = x.shape[0]  # 获取评估点的数量 Q
    p = y.shape[0]  # 获取数据点的数量 P
    r = powers.shape[0]  # 获取多项式中每个单项的数量 R
    kernel_func = NAME_TO_FUNC[kernel]  # 根据给定的 RBF 名称获取对应的核函数

    yeps = y * epsilon  # 缩放数据点坐标以便计算
    xeps = x * epsilon  # 缩放评估点坐标以便计算
    xhat = (x - shift) / scale  # 对评估点坐标进行平移和缩放以保证数值稳定性

    vec = np.empty((q, p + r), dtype=float)  # 创建一个空的系数矩阵，用于存储计算结果
    for i in range(q):
        kernel_vector(xeps[i], yeps, kernel_func, vec[i, :p])  # 计算核函数部分的向量
        polynomial_vector(xhat[i], powers, vec[i, p:])  # 计算多项式部分的向量

    return vec  # 返回计算得到的系数矩阵
```