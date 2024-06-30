# `D:\src\scipysrc\scipy\scipy\spatial\transform\_rotation.pyx`

```
# cython: cpow=True

# 导入必要的模块和函数
import re  # 导入正则表达式模块
import warnings  # 导入警告模块
import numpy as np  # 导入NumPy库
from scipy._lib._util import check_random_state  # 从SciPy库中导入随机状态检查函数
from ._rotation_groups import create_group  # 从当前包中导入旋转组创建函数

cimport numpy as np  # 在Cython中导入NumPy库
cimport cython  # 在Cython中导入Cython扩展
from cython.view cimport array  # 在Cython中导入数组视图
from libc.math cimport sqrt, sin, cos, atan2, acos, hypot, isnan, NAN, pi  # 在Cython中导入数学函数和常数

np.import_array()  # 导入NumPy的C语言API

# utilities for empty array initialization
# 定义用于创建空数组的内联函数
cdef inline double[:] _empty1(int n) noexcept:
    return array(shape=(n,), itemsize=sizeof(double), format=b"d")

cdef inline double[:, :] _empty2(int n1, int n2) noexcept :
    return array(shape=(n1, n2), itemsize=sizeof(double), format=b"d")

cdef inline double[:, :, :] _empty3(int n1, int n2, int n3) noexcept:
    return array(shape=(n1, n2, n3), itemsize=sizeof(double), format=b"d")

cdef inline double[:, :] _zeros2(int n1, int n2) noexcept:
    cdef double[:, :] arr = array(shape=(n1, n2),
        itemsize=sizeof(double), format=b"d")
    arr[:, :] = 0
    return arr

# flat implementations of numpy functions
# 实现NumPy函数的扁平化版本
@cython.boundscheck(False)  # 禁用边界检查
@cython.wraparound(False)  # 禁用索引包装
cdef inline double[:] _cross3(const double[:] a, const double[:] b) noexcept:
    cdef double[:] result = _empty1(3)
    result[0] = a[1]*b[2] - a[2]*b[1]
    result[1] = a[2]*b[0] - a[0]*b[2]
    result[2] = a[0]*b[1] - a[1]*b[0]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _dot3(const double[:] a, const double[:] b) noexcept nogil:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _norm3(const double[:] elems) noexcept nogil:
    return sqrt(_dot3(elems, elems))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _normalize4(double[:] elems) noexcept nogil:
    cdef double norm = sqrt(_dot3(elems, elems) + elems[3]*elems[3])

    if norm == 0:
        return NAN

    elems[0] /= norm
    elems[1] /= norm
    elems[2] /= norm
    elems[3] /= norm

    return norm

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _argmax4(const double[:] a) noexcept nogil:
    cdef int imax = 0
    cdef double vmax = a[0]

    for i in range(1, 4):
        if a[i] > vmax:
            imax = i
            vmax = a[i]

    return imax

ctypedef unsigned char uchar  # 定义无符号字符类型

cdef double[3] _ex = [1, 0, 0]  # 定义_x轴的基本单位向量
cdef double[3] _ey = [0, 1, 0]  # 定义_y轴的基本单位向量
cdef double[3] _ez = [0, 0, 1]  # 定义_z轴的基本单位向量

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline const double[:] _elementary_basis_vector(uchar axis) noexcept:
    if axis == b'x': return _ex
    elif axis == b'y': return _ey
    elif axis == b'z': return _ez

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _elementary_basis_index(uchar axis) noexcept:
    if axis == b'x': return 0
    elif axis == b'y': return 1
    elif axis == b'z': return 2

# Reduce the quaternion double coverage of the rotation group to a unique
# canonical "positive" single cover
# 将四元数旋转组的双重覆盖减少为唯一的正定单覆盖
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个内联函数 _quat_canonical_single，用于规范化四元数的单个元素
cdef inline void _quat_canonical_single(double[:] q) noexcept nogil:
    # 检查四元数的符号是否需要调整，确保其第四个分量非负，按照标准形式排列
    if ((q[3] < 0)
        or (q[3] == 0 and q[0] < 0)
        or (q[3] == 0 and q[0] == 0 and q[1] < 0)
        or (q[3] == 0 and q[0] == 0 and q[1] == 0 and q[2] < 0)):
        q[0] *= -1.0
        q[1] *= -1.0
        q[2] *= -1.0
        q[3] *= -1.0

# 使用 Cython 的装饰器定义内联函数 _quat_canonical，对多个四元数进行规范化
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _quat_canonical(double[:, :] q) noexcept:
    # 获取数组的行数
    cdef Py_ssize_t n = q.shape[0]
    # 遍历每个四元数，调用 _quat_canonical_single 进行规范化
    for ind in range(n):
        _quat_canonical_single(q[ind])

# 使用 Cython 的装饰器定义内联函数 _get_angles，计算欧拉角
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _get_angles(
    double[:] angles, bint extrinsic, bint symmetric, bint sign,
    double lamb, double a, double b, double c, double d):
    # 内外部转换标志的辅助变量
    cdef int angle_first, angle_third
    if extrinsic:
        angle_first = 0
        angle_third = 2
    else:
        angle_first = 2
        angle_third = 0

    cdef double half_sum, half_diff
    cdef int case

    # 步骤 2
    # 计算第二个角度...
    angles[1] = 2 * atan2(hypot(c, d), hypot(a, b))

    # ... 并检查是否等于 0 或 pi，以避免奇点
    if abs(angles[1]) <= 1e-7:
        case = 1
    elif abs(angles[1] - <double>pi) <= 1e-7:
        case = 2
    else:
        case = 0  # 正常情况

    # 步骤 3
    # 根据情况计算第一个和第三个角度
    half_sum = atan2(b, a)
    half_diff = atan2(d, c)

    if case == 0:  # 无奇点
        angles[angle_first] = half_sum - half_diff
        angles[angle_third] = half_sum + half_diff
    else:  # 任何退化情况
        angles[2] = 0
        if case == 1:
            angles[0] = 2 * half_sum
        else:
            angles[0] = 2 * half_diff * (-1 if extrinsic else 1)

    # 对于 Tait-Bryan/asymmetric 序列
    if not symmetric:
        angles[angle_third] *= sign
        angles[1] -= lamb

    # 角度限制在 [-pi, pi] 区间内
    for idx in range(3):
        if angles[idx] < -pi:
            angles[idx] += 2 * pi
        elif angles[idx] > pi:
            angles[idx] -= 2 * pi

    if case != 0:
        # 如果存在奇点，发出警告
        warnings.warn("Gimbal lock detected. Setting third angle to zero "
                      "since it is not possible to uniquely determine "
                      "all angles.", stacklevel=3)

# 使用 Cython 的装饰器定义内联函数 _compute_euler_from_matrix，从旋转矩阵计算欧拉角
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :] _compute_euler_from_matrix(
    np.ndarray[double, ndim=3] matrix, const uchar[:] seq, bint extrinsic
) noexcept:
    # 此函数正在被新函数 _compute_euler_from_quat 替代
    #
    # 算法假设内部帧变换。本文中的算法是针对旋转矩阵的，它们是在 Rotation 中使用的转置旋转矩阵。
    # 通过使用论文中定义的 O 矩阵的转置来调整算法，注意交换索引
    # 2. Reversing both axis sequence and angles for extrinsic rotations
    #
    # Based on Malcolm D. Shuster, F. Landis Markley, "General formula for
    # extraction the Euler angles", Journal of guidance, control, and
    # dynamics, vol. 29.1, pp. 215-221. 2006

    if extrinsic:
        # 如果进行外在旋转，反转旋转顺序
        seq = seq[::-1]

    cdef Py_ssize_t num_rotations = matrix.shape[0]

    # Step 0
    # 算法假设轴是列向量，这里我们使用一维向量
    cdef const double[:] n1 = _elementary_basis_vector(seq[0])
    cdef const double[:] n2 = _elementary_basis_vector(seq[1])
    cdef const double[:] n3 = _elementary_basis_vector(seq[2])

    # Step 2
    # 计算 sl 和 cl，来自于参考文献中的公式
    cdef double sl = _dot3(_cross3(n1, n2), n3)
    cdef double cl = _dot3(n1, n3)

    # angle offset is lambda from the paper referenced in [2] from docstring of
    # `as_euler` function
    # 角度偏移量是根据[2]中的文献和`as_euler`函数的文档字符串计算得到的
    cdef double offset = atan2(sl, cl)
    cdef double[:, :] c_ = _empty2(3, 3)
    c_[0, :] = n2
    c_[1, :] = _cross3(n1, n2)
    c_[2, :] = n1
    cdef np.ndarray[double, ndim=2] c = np.asarray(c_)

    rot = np.array([
        [1, 0, 0],
        [0, cl, sl],
        [0, -sl, cl],
    ])

    # some forward definitions
    # 一些预定义
    cdef double[:, :] angles = _empty2(num_rotations, 3)
    cdef double[:, :] matrix_trans # transformed matrix
    cdef double[:] _angles # accessor for each rotation
    cdef np.ndarray[double, ndim=2] res
    cdef double eps = 1e-7
    cdef bint safe1, safe2, safe, adjust
    # 对于每一个旋转操作
    for ind in range(num_rotations):
        _angles = angles[ind, :]

        # Step 3: 矩阵乘法，将旋转操作应用于当前坐标系
        res = np.dot(c, matrix[ind, :, :])
        matrix_trans = np.dot(res, c.T.dot(rot))

        # Step 4: 确保第三行第三列元素在单位范围内
        matrix_trans[2, 2] = min(matrix_trans[2, 2], 1)
        matrix_trans[2, 2] = max(matrix_trans[2, 2], -1)
        _angles[1] = acos(matrix_trans[2, 2])

        # Steps 5, 6: 检查角度安全性
        safe1 = abs(_angles[1]) >= eps
        safe2 = abs(_angles[1] - np.pi) >= eps
        safe = safe1 and safe2

        # Step 4 (Completion): 调整第二角度
        _angles[1] += offset

        # 5b: 如果安全，计算第一和第三角度
        if safe:
            _angles[0] = atan2(matrix_trans[0, 2], -matrix_trans[1, 2])
            _angles[2] = atan2(matrix_trans[2, 0], matrix_trans[2, 1])

        if extrinsic:
            # 6a: 对于外部旋转，如果不安全，将第一角度设为零
            if not safe:
                _angles[0] = 0
            # 6b, 6c: 根据安全性计算第三角度
            if not safe1:
                _angles[2] = atan2(matrix_trans[1, 0] - matrix_trans[0, 1],
                                   matrix_trans[0, 0] + matrix_trans[1, 1])
            if not safe2:
                _angles[2] = -atan2(matrix_trans[1, 0] + matrix_trans[0, 1],
                                    matrix_trans[0, 0] - matrix_trans[1, 1])
        else:
            # 6a: 对于内部旋转，如果不安全，将第三角度设为零
            if not safe:
                _angles[2] = 0
            # 6b, 6c: 根据安全性计算第一角度
            if not safe1:
                _angles[0] = atan2(matrix_trans[1, 0] - matrix_trans[0, 1],
                                   matrix_trans[0, 0] + matrix_trans[1, 1])
            if not safe2:
                _angles[0] = atan2(matrix_trans[1, 0] + matrix_trans[0, 1],
                                   matrix_trans[0, 0] - matrix_trans[1, 1])

        # Step 7: 根据序列的第一个和第三个元素，调整第二角度确保在特定范围内
        if seq[0] == seq[2]:
            adjust = _angles[1] < 0 or _angles[1] > np.pi
        else:
            adjust = _angles[1] < -np.pi / 2 or _angles[1] > np.pi / 2

        # 不调整万向节锁定角度序列
        if adjust and safe:
            # 调整角度确保唯一性
            _angles[0] += np.pi
            _angles[1] = 2 * offset - _angles[1]
            _angles[2] -= np.pi

        # 角度限制在 [-pi, pi] 范围内
        for i in range(3):
            if _angles[i] < -np.pi:
                _angles[i] += 2 * np.pi
            elif _angles[i] > np.pi:
                _angles[i] -= 2 * np.pi

        if extrinsic:
            # 旋转操作逆序
            _angles[0], _angles[2] = _angles[2], _angles[0]

        # Step 8: 如果不安全，发出警告
        if not safe:
            warnings.warn("Gimbal lock detected. Setting third angle to zero "
                          "since it is not possible to uniquely determine "
                          "all angles.")

    return angles
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 Cython 函数，计算从四元数到欧拉角的转换
cdef double[:, :] _compute_euler_from_quat(
    np.ndarray[double, ndim=2] quat, const uchar[:] seq, bint extrinsic
) noexcept:
    # 算法假设外部帧转换。文章中的算法适用于旋转四元数，由 Rotation 直接存储。
    # 根据需要为内部旋转逆序两个轴序列和角度来调整算法

    if not extrinsic:
        # 如果不是外部转换，反转轴序列
        seq = seq[::-1]

    # 获取轴序列对应的索引
    cdef int i = _elementary_basis_index(seq[0])
    cdef int j = _elementary_basis_index(seq[1])
    cdef int k = _elementary_basis_index(seq[2])

    # 检查是否对称
    cdef bint symmetric = i == k
    if symmetric:
        k = 3 - i - j  # 获取第三个轴

    # Step 0
    # 检查排列是偶数 (+1) 还是奇数 (-1)
    cdef int sign = (i - j) * (j - k) * (k - i) // 2

    # 获取四元数数组的大小
    cdef Py_ssize_t num_rotations = quat.shape[0]

    # 一些预定义
    cdef double a, b, c, d
    cdef double[:, :] angles = _empty2(num_rotations, 3)

    for ind in range(num_rotations):

        # Step 1
        # 对四元数元素进行排列
        if symmetric:
            a = quat[ind, 3]
            b = quat[ind, i]
            c = quat[ind, j]
            d = quat[ind, k] * sign
        else:
            a = quat[ind, 3] - quat[ind, j]
            b = quat[ind, i] + quat[ind, k] * sign
            c = quat[ind, j] + quat[ind, 3]
            d = quat[ind, k] * sign - quat[ind, i]

        # 调用函数计算角度
        _get_angles(angles[ind], extrinsic, symmetric, sign, pi / 2, a, b, c, d)

    return angles


@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个 Cython 函数，计算从四元数到达文波特角的转换
cdef double[:, :] _compute_davenport_from_quat(
    np.ndarray[double, ndim=2] quat, np.ndarray[double, ndim=1] n1,
    np.ndarray[double, ndim=1] n2, np.ndarray[double, ndim=1] n3,
    bint extrinsic
):
    # 算法假设外部帧转换。文章中的算法适用于旋转四元数，由 Rotation 直接存储。
    # 根据需要为内部旋转逆序两个轴序列和角度来调整算法

    if not extrinsic:
        # 如果不是外部转换，交换 n1 和 n3
        n1, n3 = n3, n1

    # 计算 n1 和 n2 的叉乘
    cdef double[:] n_cross = _cross3(n1, n2)
    # 计算 lamb
    cdef double lamb = atan2(_dot3(n3, n_cross), _dot3(n3, n1))

    cdef int correct_set = False
    if lamb < 0:
        # 与 as_euler 实现兼容的备选角度集合
        n2 = -n2
        lamb = -lamb
        n_cross[0] = -n_cross[0]
        n_cross[1] = -n_cross[1]
        n_cross[2] = -n_cross[2]
        correct_set = True

    # 计算变换后的四元数
    cdef double[:] quat_lamb = np.array([
            sin(lamb / 2) * n2[0],
            sin(lamb / 2) * n2[1],
            sin(lamb / 2) * n2[2],
            cos(lamb / 2)]
    )

    # 获取四元数数组的大小
    cdef Py_ssize_t num_rotations = quat.shape[0]

    # 一些预定义
    cdef double[:, :] angles = _empty2(num_rotations, 3)
    cdef double[:] quat_transformed = _empty1(4)
    cdef double a, b, c, d
    # 定义四个双精度浮点数变量 a, b, c, d，用于存储计算后的值

    for ind in range(num_rotations):
        # 循环执行 num_rotations 次，ind 为循环变量

        _compose_quat_single(quat_lamb, quat[ind], quat_transformed)
        # 调用函数 _compose_quat_single 对 quat_lamb 和 quat[ind] 进行组合运算，并将结果存入 quat_transformed

        # Step 1
        # 步骤1：重新排列四元数元素
        a = quat_transformed[3]
        # 将 quat_transformed 的第4个元素赋值给变量 a
        b = _dot3(quat_transformed[:3], n1)
        # 调用函数 _dot3 计算 quat_transformed 的前三个元素与 n1 的点积，并将结果赋值给变量 b
        c = _dot3(quat_transformed[:3], n2)
        # 调用函数 _dot3 计算 quat_transformed 的前三个元素与 n2 的点积，并将结果赋值给变量 c
        d = _dot3(quat_transformed[:3], n_cross)
        # 调用函数 _dot3 计算 quat_transformed 的前三个元素与 n_cross 的点积，并将结果赋值给变量 d

        _get_angles(angles[ind], extrinsic, False, 1, lamb, a, b, c, d)
        # 调用函数 _get_angles 计算角度，并将结果存入 angles[ind]

        if correct_set:
            # 如果 correct_set 为真
            angles[ind, 1] = -angles[ind, 1]
            # 将 angles[ind, 1] 的值取负数

    return angles
    # 返回 angles 变量
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个内联函数，用于计算两个四元数的乘积并存储在结果中
cdef inline void _compose_quat_single(
    const double[:] p, const double[:] q, double[:] r
) noexcept:
    # 计算 p 和 q 的前三个元素的叉积
    cdef double[:] cross = _cross3(p[:3], q[:3])

    # 计算四元数乘积的各个元素
    r[0] = p[3]*q[0] + q[3]*p[0] + cross[0]
    r[1] = p[3]*q[1] + q[3]*p[1] + cross[1]
    r[2] = p[3]*q[2] + q[3]*p[2] + cross[2]
    r[3] = p[3]*q[3] - p[0]*q[0] - p[1]*q[1] - p[2]*q[2]

@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个内联函数，用于计算多个四元数的乘积并返回结果
cdef inline double[:, :] _compose_quat(
    const double[:, :] p, const double[:, :] q
) noexcept:
    # 确定 p 和 q 的行数中的较大值作为迭代次数
    cdef Py_ssize_t n = max(p.shape[0], q.shape[0])
    # 创建一个 n x 4 的二维数组，用于存储乘积结果
    cdef double[:, :] product = _empty2(n, 4)

    # 处理广播情况
    if p.shape[0] == 1:
        # 如果 p 只有一行，则对每个 q 中的行进行单个乘积计算
        for ind in range(n):
            _compose_quat_single(p[0], q[ind], product[ind])
    elif q.shape[0] == 1:
        # 如果 q 只有一行，则对每个 p 中的行进行单个乘积计算
        for ind in range(n):
            _compose_quat_single(p[ind], q[0], product[ind])
    else:
        # 否则，对应 p 和 q 的每对行进行乘积计算
        for ind in range(n):
            _compose_quat_single(p[ind], q[ind], product[ind])

    # 返回计算得到的四元数乘积结果
    return product

@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个内联函数，用于创建基础的单轴旋转四元数
cdef inline double[:, :] _make_elementary_quat(
    uchar axis, const double[:] angles
) noexcept:
    # 确定角度数组的长度
    cdef Py_ssize_t n = angles.shape[0]
    # 创建一个 n x 4 的二维数组，用于存储四元数结果
    cdef double[:, :] quat = _zeros2(n, 4)

    cdef int axis_ind
    # 根据轴向量设置对应的索引
    if axis == b'x':   axis_ind = 0
    elif axis == b'y': axis_ind = 1
    elif axis == b'z': axis_ind = 2

    # 对每个角度计算相应的四元数
    for ind in range(n):
        quat[ind, 3] = cos(angles[ind] / 2)
        quat[ind, axis_ind] = sin(angles[ind] / 2)
    
    # 返回计算得到的四元数数组
    return quat

@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个内联函数，用于组合一系列单轴旋转四元数
cdef double[:, :] _elementary_quat_compose(
    const uchar[:] seq, const double[:, :] angles, bint intrinsic=False
) noexcept:
    # 使用第一个角度数组创建初始的四元数数组
    cdef double[:, :] result = _make_elementary_quat(seq[0], angles[:, 0])
    cdef Py_ssize_t seq_len = seq.shape[0]

    # 遍历序列中的每个轴和角度数组，依次组合四元数
    for idx in range(1, seq_len):
        if intrinsic:
            # 如果使用内部组合顺序，则将当前结果与下一个四元数数组组合
            result = _compose_quat(
                result,
                _make_elementary_quat(seq[idx], angles[:, idx]))
        else:
            # 否则，将下一个四元数数组与当前结果组合
            result = _compose_quat(
                _make_elementary_quat(seq[idx], angles[:, idx]),
                result)
    
    # 返回最终组合的四元数数组
    return result

def _format_angles(angles, degrees, num_axes):
    # 将角度数组转换为浮点数类型
    angles = np.asarray(angles, dtype=float)
    if degrees:
        # 如果需要将角度转换为弧度，则进行转换
        angles = np.deg2rad(angles)

    is_single = False
    # 准备角度数组，使其具有形状 (num_rot, num_axes)
    # 如果旋转轴数为1
    if num_axes == 1:
        # 如果角度数组的维度是0
        if angles.ndim == 0:
            # 将其reshape成 (1, 1) 形状
            angles = angles.reshape((1, 1))
            # 标记为单个角度
            is_single = True
        # 如果角度数组的维度是1
        elif angles.ndim == 1:
            # 将其扩展为 (N, 1) 形状
            angles = angles[:, None]
        # 如果角度数组的维度是2且最后一维不是1
        elif angles.ndim == 2 and angles.shape[-1] != 1:
            # 抛出值错误异常，期望 `angles` 参数的形状是 (N, 1)
            raise ValueError("Expected `angles` parameter to have shape "
                             "(N, 1), got {}.".format(angles.shape))
        # 如果角度数组的维度大于2
        elif angles.ndim > 2:
            # 抛出值错误异常，期望浮点数、1D数组或2D数组作为 `angles` 参数
            # 对应于 `seq`，得到形状为 {}
            raise ValueError("Expected float, 1D array, or 2D array for "
                             "parameter `angles` corresponding to `seq`, "
                             "got shape {}.".format(angles.shape))
    else:  # 如果旋转轴数为2或3
        # 如果角度数组的维度不是1或2，或者最后一维不等于旋转轴数
        if angles.ndim not in [1, 2] or angles.shape[-1] != num_axes:
            # 抛出值错误异常，期望 `angles` 是至多二维数组，宽度等于指定的轴数
            # 得到 {} 的形状
            raise ValueError("Expected `angles` to be at most "
                             "2-dimensional with width equal to number "
                             "of axes specified, got "
                             "{} for shape".format(angles.shape))

        # 如果角度数组的维度是1
        if angles.ndim == 1:
            # 将其扩展为 (1, num_axes) 形状
            angles = angles[None, :]
            # 标记为单个角度
            is_single = True

    # 到这一步，角度数组应该具有形状 (num_rot, num_axes)
    # 进行合理性检查
    if angles.ndim != 2 or angles.shape[-1] != num_axes:
        # 抛出值错误异常，期望角度数组的形状是 (num_rotations, num_axes)
        # 得到 {}
        raise ValueError("Expected angles to have shape (num_rotations, "
                         "num_axes), got {}.".format(angles.shape))

    # 返回角度数组和单个角度标志
    return angles, is_single
# 定义一个 Cython 类 Rotation，表示三维空间中的旋转
cdef class Rotation:
    """Rotation in 3 dimensions.

    This class provides an interface to initialize from and represent rotations
    with:

    - Quaternions
    - Rotation Matrices
    - Rotation Vectors
    - Modified Rodrigues Parameters
    - Euler Angles

    The following operations on rotations are supported:

    - Application on vectors
    - Rotation Composition
    - Rotation Inversion
    - Rotation Indexing

    Indexing within a rotation is supported since multiple rotation transforms
    can be stored within a single `Rotation` instance.

    To create `Rotation` objects use ``from_...`` methods (see examples below).
    ``Rotation(...)`` is not supposed to be instantiated directly.

    Attributes
    ----------
    single

    Methods
    -------
    __len__
        # 返回 Rotation 实例中存储的旋转数目

    from_quat
        # 从四元数初始化 Rotation 实例

    from_matrix
        # 从旋转矩阵初始化 Rotation 实例

    from_rotvec
        # 从旋转向量初始化 Rotation 实例

    from_mrp
        # 从修改的 Rodrigues 参数初始化 Rotation 实例

    from_euler
        # 从欧拉角初始化 Rotation 实例

    from_davenport
        # 从 Davenport 参数初始化 Rotation 实例

    as_quat
        # 将 Rotation 实例转换为四元数表示

    as_matrix
        # 将 Rotation 实例转换为旋转矩阵表示

    as_rotvec
        # 将 Rotation 实例转换为旋转向量表示

    as_mrp
        # 将 Rotation 实例转换为修改的 Rodrigues 参数表示

    as_euler
        # 将 Rotation 实例转换为欧拉角表示

    as_davenport
        # 将 Rotation 实例转换为 Davenport 参数表示

    concatenate
        # 合并两个 Rotation 实例

    apply
        # 将 Rotation 实例应用于向量

    __mul__
        # 定义 Rotation 实例之间的乘法操作

    __pow__
        # 定义 Rotation 实例的乘方操作

    inv
        # 返回 Rotation 实例的逆

    magnitude
        # 返回 Rotation 实例的大小

    approx_equal
        # 检查 Rotation 实例是否近似相等

    mean
        # 返回多个 Rotation 实例的平均值

    reduce
        # 简化 Rotation 实例的表示形式

    create_group
        # 创建旋转组

    __getitem__
        # 获取 Rotation 实例中的特定旋转

    identity
        # 返回单位旋转

    random
        # 返回随机旋转

    align_vectors
        # 将 Rotation 实例应用于对齐向量

    See Also
    --------
    Slerp

    Notes
    -----
    .. versionadded:: 1.2.0

    Examples
    --------
    >>> from scipy.spatial.transform import Rotation as R
    >>> import numpy as np

    A `Rotation` instance can be initialized in any of the above formats and
    converted to any of the others. The underlying object is independent of the
    representation used for initialization.

    Consider a counter-clockwise rotation of 90 degrees about the z-axis. This
    corresponds to the following quaternion (in scalar-last format):

    >>> r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])

    The rotation can be expressed in any of the other formats:

    >>> r.as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    >>> r.as_rotvec()
    array([0.        , 0.        , 1.57079633])
    >>> r.as_euler('zyx', degrees=True)
    array([90.,  0.,  0.])

    The same rotation can be initialized using a rotation matrix:

    >>> r = R.from_matrix([[0, -1, 0],
                          [1,  0, 0],
                          [0,  0, 1]])

    Representation in other formats:

    >>> r.as_quat()
    array([0.        , 0.        , 0.70710678, 0.70710678])
    >>> r.as_rotvec()
    array([0.        , 0.        , 1.57079633])
    >>> r.as_euler('zyx', degrees=True)
    array([90.,  0.,  0.])

    The rotation vector corresponding to this rotation is given by:

    >>> r = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))

    Representation in other formats:

    >>> r.as_quat()
    array([0.        , 0.        , 0.70710678, 0.70710678])
    >>> r.as_matrix()
    """
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# 创建一个 3x3 的numpy数组，表示一个单位矩阵，用于描述没有任何旋转的初始姿态。

    >>> r.as_euler('zyx', degrees=True)

    array([90.,  0.,  0.])

# 将当前旋转矩阵 `r` 转换为欧拉角（Z-Y-X顺序），返回角度制的旋转角度数组。

    The ``from_euler`` method is quite flexible in the range of input formats
    it supports. Here we initialize a single rotation about a single axis:

    >>> r = R.from_euler('z', 90, degrees=True)

# 使用欧拉角初始化旋转对象 `r`，绕Z轴旋转90度，角度单位为度数。

    Again, the object is representation independent and can be converted to any
    other format:

    >>> r.as_quat()

    array([0.        , 0.        , 0.70710678, 0.70710678])

# 将旋转对象 `r` 转换为四元数表示，返回四元数数组。

    >>> r.as_matrix()

    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# 将旋转对象 `r` 转换为旋转矩阵表示，返回一个3x3的numpy数组。

    >>> r.as_rotvec()

    array([0.        , 0.        , 1.57079633])

# 将旋转对象 `r` 转换为旋转向量表示，返回一个包含三个分量的numpy数组。

    It is also possible to initialize multiple rotations in a single instance
    using any of the ``from_...`` functions. Here we initialize a stack of 3
    rotations using the ``from_euler`` method:

    >>> r = R.from_euler('zyx', [
    ... [90, 0, 0],
    ... [0, 45, 0],
    ... [45, 60, 30]], degrees=True)

# 使用欧拉角初始化多个旋转对象 `r`，使用Z-Y-X顺序，并分别用度数表示三个旋转角度。

    The other representations also now return a stack of 3 rotations. For
    example:

    >>> r.as_quat()

    array([[0.        , 0.        , 0.70710678, 0.70710678],
           [0.        , 0.38268343, 0.        , 0.92387953],
           [0.39190384, 0.36042341, 0.43967974, 0.72331741]])

# 将多个旋转对象 `r` 分别转换为四元数表示，返回一个包含三个旋转的四元数数组。

    Applying the above rotations onto a vector:

    >>> v = [1, 2, 3]
    >>> r.apply(v)

    array([[-2.        ,  1.        ,  3.        ],
           [ 2.82842712,  2.        ,  1.41421356],
           [ 2.24452282,  0.78093109,  2.89002836]])

# 将向量 `v` 应用于多个旋转对象 `r`，返回一个包含三个旋转后向量的数组。

    A `Rotation` instance can be indexed and sliced as if it were a single
    1D array or list:

    >>> r.as_quat()

    array([[0.        , 0.        , 0.70710678, 0.70710678],
           [0.        , 0.38268343, 0.        , 0.92387953],
           [0.39190384, 0.36042341, 0.43967974, 0.72331741]])

# 将多个旋转对象 `r` 分别转换为四元数表示，返回一个包含三个旋转的四元数数组。

    >>> p = r[0]
    >>> p.as_matrix()

    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# 从多个旋转对象 `r` 中提取第一个对象 `p`，并将其转换为旋转矩阵表示。

    >>> q = r[1:3]
    >>> q.as_quat()

    array([[0.        , 0.38268343, 0.        , 0.92387953],
           [0.39190384, 0.36042341, 0.43967974, 0.72331741]])

# 从多个旋转对象 `r` 中提取第二个和第三个对象 `q`，并将其转换为四元数表示。

    In fact it can be converted to numpy.array:

    >>> r_array = np.asarray(r)
    >>> r_array.shape

    (3,)

# 将多个旋转对象 `r` 转换为numpy数组 `r_array`，并输出其形状。

    >>> r_array[0].as_matrix()

    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# 从numpy数组 `r_array` 中提取第一个旋转对象，并将其转换为旋转矩阵表示。

    Multiple rotations can be composed using the ``*`` operator:

    >>> r1 = R.from_euler('z', 90, degrees=True)
    >>> r2 = R.from_rotvec([np.pi/4, 0, 0])
    >>> v = [1, 2, 3]
    # 使用 r1 对向量 v 进行旋转，然后再使用 r2 对结果进行旋转
    >>> r2.apply(r1.apply(v))
    array([-2.        , -1.41421356,  2.82842712])
    
    # 创建一个新的旋转对象 r3，它是 r2 和 r1 的乘积，注意顺序
    >>> r3 = r2 * r1 # 注意顺序
    # 对向量 v 应用 r3 进行旋转
    >>> r3.apply(v)
    array([-2.        , -1.41421356,  2.82842712])
    
    # 使用 ** 运算符可以将一个旋转与自身组合
    >>> p = R.from_rotvec([1, 0, 0])
    >>> q = p ** 2
    # 将 q 转换为旋转向量形式
    >>> q.as_rotvec()
    array([2., 0., 0.])
    
    # 可以通过 inv() 方法反转旋转
    >>> r1 = R.from_euler('z', [90, 45], degrees=True)
    >>> r2 = r1.inv()
    # 将 r2 表示为欧拉角（'zyx'）形式的旋转
    >>> r2.as_euler('zyx', degrees=True)
    array([[-90.,   0.,   0.],
           [-45.,   0.,   0.]])
    
    # 下面的函数可以用来使用 Matplotlib 绘制旋转后的坐标轴
    >>> import matplotlib.pyplot as plt
    
    # 定义一个函数 plot_rotated_axes，用于绘制旋转后的坐标轴
    >>> def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    ...     colors = ("#FF6666", "#005533", "#1199EE")  # 色盲安全的 RGB 颜色
    ...     loc = np.array([offset, offset])
    ...     for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
    ...                                       colors)):
    ...         axlabel = axis.axis_name
    ...         axis.set_label_text(axlabel)
    ...         axis.label.set_color(c)
    ...         axis.line.set_color(c)
    ...         axis.set_tick_params(colors=c)
    ...         line = np.zeros((2, 3))
    ...         line[1, i] = scale
    ...         line_rot = r.apply(line)
    ...         line_plot = line_rot + loc
    ...         ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
    ...         text_loc = line[1]*1.2
    ...         text_loc_rot = r.apply(text_loc)
    ...         text_plot = text_loc_rot + loc[0]
    ...         ax.text(*text_plot, axlabel.upper(), color=c,
    ...                 va="center", ha="center")
    ...     ax.text(*offset, name, color="k", va="center", ha="center",
    ...             bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})
    
    # 创建三个旋转：单位旋转和使用内部和外部约定的两个欧拉旋转
    >>> r0 = R.identity()
    >>> r1 = R.from_euler("ZYX", [90, -30, 0], degrees=True)  # 内部约定
    >>> r2 = R.from_euler("zyx", [90, -30, 0], degrees=True)  # 外部约定
    
    # 将三个旋转添加到单个图中
    >>> ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
    >>> plot_rotated_axes(ax, r0, name="r0", offset=(0, 0, 0))
    >>> plot_rotated_axes(ax, r1, name="r1", offset=(3, 0, 0))
    >>> plot_rotated_axes(ax, r2, name="r2", offset=(6, 0, 0))
    >>> _ = ax.annotate(
    ...     "r0: 单位旋转\\n"
    ...     "r1: 内部欧拉角旋转 (ZYX)\\n"
    ...     "r2: 外部欧拉角旋转 (zyx)",
    ...     xy=(0.6, 0.7), xycoords="axes fraction", ha="left"
    ... )
    >>> ax.set(xlim=(-1.25, 7.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
    >>> ax.set(xticks=range(-1, 8), yticks=[-1, 0, 1], zticks=[-1, 0, 1])
    >>> ax.set_aspect("equal", adjustable="box")
    # 设置图形的纵横比为“equal”，调整方式为“box”

    >>> ax.figure.set_size_inches(6, 5)
    # 设置图形的尺寸为 6 英寸宽，5 英寸高

    >>> plt.tight_layout()
    # 调整子图的布局以确保紧凑显示

    Show the plot:
    # 显示绘图结果

    >>> plt.show()
    # 显示绘图结果到屏幕

    These examples serve as an overview into the `Rotation` class and highlight
    major functionalities. For more thorough examples of the range of input and
    output formats supported, consult the individual method's examples.
    # 这些示例概述了 `Rotation` 类的主要功能。要查看支持的输入和输出格式的更详细示例，请参考各个方法的示例。

    """
    cdef double[:, :] _quat
    # 声明私有成员变量 `_quat`，类型为 double 类型的二维数组

    cdef bint _single
    # 声明私有成员变量 `_single`，类型为布尔值

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # Cython 编译器指令：禁止边界检查和负索引处理

    def __init__(self, quat, normalize=True, copy=True, scalar_first=False):
        # 定义初始化方法，接受四元数 `quat`，默认进行归一化、复制输入和标量优先处理

        self._single = False
        # 初始化 `_single` 为 False
        quat = np.asarray(quat, dtype=float)
        # 将 `quat` 转换为 NumPy 数组，数据类型为 float

        if quat.ndim not in [1, 2] or quat.shape[len(quat.shape) - 1] != 4:
            # 如果 `quat` 的维度不是 1 或 2，或者最后一个维度不是 4
            raise ValueError("Expected `quat` to have shape (4,) or (N, 4), "
                             f"got {quat.shape}.")
            # 抛出值错误异常，指示预期形状为 (4,) 或 (N, 4)，但实际形状为 `quat.shape`

        # 如果给定单个四元数，将其转换为 2D 的 1x4 矩阵，并设置 self._single 为 True，
        # 这样我们可以在 `to_...` 方法中返回适当的对象
        if quat.shape == (4,):
            quat = quat[None, :]
            self._single = True

        cdef Py_ssize_t num_rotations = quat.shape[0]
        # 声明局部变量 `num_rotations`，其值为 `quat` 的行数

        if scalar_first:
            quat = np.roll(quat, -1, axis=1)
        elif normalize or copy:
            quat = quat.copy()

        if normalize:
            # 如果进行归一化
            for ind in range(num_rotations):
                if isnan(_normalize4(quat[ind, :])):
                    # 如果归一化后的四元数的范数为零
                    raise ValueError("Found zero norm quaternions in `quat`.")
                    # 抛出值错误异常，指示在 `quat` 中发现零范数的四元数

        self._quat = quat
        # 将归一化后的四元数赋值给私有成员变量 `_quat`

    def __getstate__(self):
        # 定义 `__getstate__` 方法，返回对象的状态以便于序列化
        return np.asarray(self._quat, dtype=float), self._single
        # 返回 `_quat` 的 NumPy 数组表示和 `_single` 的当前状态

    def __setstate__(self, state):
        # 定义 `__setstate__` 方法，用于从序列化状态中恢复对象
        quat, single = state
        self._quat = quat.copy()
        # 将传入的四元数状态复制给 `_quat`
        self._single = single
        # 设置 `_single` 的值为传入状态中的值

    @property
    def single(self):
        """Whether this instance represents a single rotation."""
        # 返回当前实例是否表示单个旋转的属性方法文档字符串
        return self._single

    def __bool__(self):
        """Comply with Python convention for objects to be True.

        Required because `Rotation.__len__()` is defined and not always truthy.
        """
        # 返回对象是否为真的方法文档字符串
        return True

    @cython.embedsignature(True)
    def __len__(self):
        """Number of rotations contained in this object.

        Multiple rotations can be stored in a single instance.

        Returns
        -------
        length : int
            Number of rotations stored in object.

        Raises
        ------
        TypeError if the instance was created as a single rotation.
        """
        # 返回对象中包含的旋转数量的方法文档字符串
        if self._single:
            # 如果实例表示单个旋转
            raise TypeError("Single rotation has no len().")
            # 抛出类型错误异常，指示单个旋转没有长度

        return self._quat.shape[0]
        # 返回 `_quat` 的行数作为对象中存储的旋转数量

    @cython.embedsignature(True)
    @classmethod
    @cython.embedsignature(True)
    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.embedsignature(True)
    @classmethod
    @cython.embedsignature(True)
    # Cython 类方法装饰器和指令
    # 声明一个类方法装饰器，允许在编译时嵌入函数签名
    @classmethod
    # 嵌入函数签名到生成的 Cython 代码中
    @cython.embedsignature(True)
    # 声明一个类方法装饰器，禁用边界检查
    @classmethod
    @cython.boundscheck(False)
    # 声明一个类方法装饰器，禁用负索引的边界检查
    @cython.wraparound(False)
    # 嵌入函数签名到生成的 Cython 代码中
    @cython.embedsignature(True)
    # 嵌入函数签名到生成的 Cython 代码中（重复）
    @cython.embedsignature(True)
    # 声明一个类方法装饰器，禁用边界检查（重复）
    @cython.boundscheck(False)
    # 声明一个类方法装饰器，禁用负索引的边界检查（重复）
    @cython.wraparound(False)
    # 嵌入函数签名到生成的 Cython 代码中（重复）
    @cython.embedsignature(True)
    # 声明一个类方法装饰器，禁用边界检查（重复）
    @cython.boundscheck(False)
    # 声明一个类方法装饰器，禁用负索引的边界检查（重复）
    @cython.wraparound(False)
    def as_rotvec(self, degrees=False):
        """Represent as rotation vectors.

        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation [1]_.

        Parameters
        ----------
        degrees : boolean, optional
            Returned magnitudes are in degrees if this flag is True, else they are
            in radians. Default is False.

            .. versionadded:: 1.7.0

        Returns
        -------
        rotvec : ndarray, shape (3,) or (N, 3)
            Shape depends on shape of inputs used for initialization.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Represent a single rotation:

        >>> r = R.from_euler('z', 90, degrees=True)
        >>> r.as_rotvec()
        array([0.        , 0.        , 1.57079633])
        >>> r.as_rotvec().shape
        (3,)

        Represent a rotation in degrees:

        >>> r = R.from_euler('YX', (-90, -90), degrees=True)
        >>> s = r.as_rotvec(degrees=True)
        >>> s
        array([-69.2820323, -69.2820323, -69.2820323])
        >>> np.linalg.norm(s)
        120.00000000000001

        Represent a stack with a single rotation:

        >>> r = R.from_quat([[0, 0, 1, 1]])
        >>> r.as_rotvec()
        array([[0.        , 0.        , 1.57079633]])
        >>> r.as_rotvec().shape
        (1, 3)

        Represent multiple rotations in a single object:

        >>> r = R.from_quat([[0, 0, 1, 1], [1, 1, 0, 1]])
        >>> r.as_rotvec()
        array([[0.        , 0.        , 1.57079633],
               [1.35102172, 1.35102172, 0.        ]])
        >>> r.as_rotvec().shape
        (2, 3)

        """

        cdef Py_ssize_t num_rotations = len(self._quat)
        cdef double angle, scale, angle2
        cdef double[:, :] rotvec = _empty2(num_rotations, 3)
        cdef double[:] quat

        # Iterate over each quaternion to compute the rotation vector
        for ind in range(num_rotations):
            quat = self._quat[ind, :].copy()
            _quat_canonical_single(quat)  # Ensure quaternion is canonicalized

            # Calculate the angle of rotation using the quaternion
            angle = 2 * atan2(_norm3(quat), quat[3])

            # Determine scaling factor based on angle size
            if angle <= 1e-3:  # Small angle Taylor series expansion
                angle2 = angle * angle
                scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
            else:  # Large angle
                scale = angle / sin(angle / 2)

            # Compute the components of the rotation vector
            rotvec[ind, 0] = scale * quat[0]
            rotvec[ind, 1] = scale * quat[1]
            rotvec[ind, 2] = scale * quat[2]

        # Convert rotation vector to degrees if specified
        if degrees:
            rotvec = np.rad2deg(rotvec)

        # Return the rotation vector(s) as ndarray
        if self._single:
            return np.asarray(rotvec[0])
        else:
            return np.asarray(rotvec)

    @cython.embedsignature(True)
    # 定义一个方法用于计算欧拉角，该方法接受序列、角度单位、算法作为参数

    # 检查序列长度是否为3，如果不是则抛出值错误异常
    if len(seq) != 3:
        raise ValueError("Expected 3 axes, got {}.".format(seq))

    # 检查序列是否为内在轴或外在轴表示法（大小写XYZ或xyz），如果不是则抛出值错误异常
    intrinsic = (re.match(r'^[XYZ]{1,3}$', seq) is not None)
    extrinsic = (re.match(r'^[xyz]{1,3}$', seq) is not None)
    if not (intrinsic or extrinsic):
        raise ValueError("Expected axes from `seq` to be from "
                         "['x', 'y', 'z'] or ['X', 'Y', 'Z'], "
                         "got {}".format(seq))

    # 检查连续的轴是否相同，如果相同则抛出值错误异常
    if any(seq[i] == seq[i+1] for i in range(2)):
        raise ValueError("Expected consecutive axes to be different, "
                         "got {}".format(seq))

    # 将序列转换为小写形式
    seq = seq.lower()

    # 根据指定的算法调用相应的欧拉角计算函数（从矩阵或四元数）
    if algorithm == 'from_matrix':
        # 获取当前对象的旋转矩阵
        matrix = self.as_matrix()
        # 如果矩阵维度为2，则扩展为3维
        if matrix.ndim == 2:
            matrix = matrix[None, :, :]
        # 调用计算欧拉角从矩阵的函数，并转换为NumPy数组
        angles = np.asarray(_compute_euler_from_matrix(
            matrix, seq.encode(), extrinsic))
    elif algorithm == 'from_quat':
        # 获取当前对象的四元数
        quat = self.as_quat()
        # 如果四元数维度为1，则扩展为2维
        if quat.ndim == 1:
            quat = quat[None, :]
        # 调用计算欧拉角从四元数的函数，并转换为NumPy数组
        angles = np.asarray(_compute_euler_from_quat(
                quat, seq.encode(), extrinsic))
    else:
        # 如果算法不是'from_quat'或'from_matrix'，则断言失败
        assert False

    # 如果指定以度为单位，则将角度转换为弧度
    if degrees:
        angles = np.rad2deg(angles)

    # 如果对象是单个旋转，则返回第一个角度；否则返回所有角度
    return angles[0] if self._single else angles
    def _as_euler_from_matrix(self, seq, degrees=False):
        """
        Represent as Euler angles.

        Any orientation can be expressed as a composition of 3 elementary
        rotations. Once the axis sequence has been chosen, Euler angles define
        the angle of rotation around each respective axis [1]_.

        The algorithm from [2]_ has been used to calculate Euler angles for the
        rotation about a given sequence of axes.

        Euler angles suffer from the problem of gimbal lock [3]_, where the
        representation loses a degree of freedom and it is not possible to
        determine the first and third angles uniquely. In this case,
        a warning is raised, and the third angle is set to zero. Note however
        that the returned angles still represent the correct rotation.

        Parameters
        ----------
        seq : string, length 3
            3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
            rotations, or {'x', 'y', 'z'} for extrinsic rotations [1]_.
            Adjacent axes cannot be the same.
            Extrinsic and intrinsic rotations cannot be mixed in one function
            call.
        degrees : boolean, optional
            Returned angles are in degrees if this flag is True, else they are
            in radians. Default is False.

        Returns
        -------
        angles : ndarray, shape (3,) or (N, 3)
            Shape depends on shape of inputs used to initialize object.
            The returned angles are in the range:

            - First angle belongs to [-180, 180] degrees (both inclusive)
            - Third angle belongs to [-180, 180] degrees (both inclusive)
            - Second angle belongs to:

                - [-90, 90] degrees if all axes are different (like xyz)
                - [0, 180] degrees if first and third axes are the same
                  (like zxz)

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        .. [2] Malcolm D. Shuster, F. Landis Markley, "General formula for
               extraction the Euler angles", Journal of guidance, control, and
               dynamics, vol. 29.1, pp. 215-221. 2006
        .. [3] https://en.wikipedia.org/wiki/Gimbal_lock#In_applied_mathematics

        """
        # 调用内部方法 _compute_euler，计算从矩阵到欧拉角的转换
        return self._compute_euler(seq, degrees, 'from_matrix')

    @cython.embedsignature(True)
    @cython.embedsignature(True)
    @cython.embedsignature(True)
    def as_mrp(self):
        """Represent as Modified Rodrigues Parameters (MRPs).

        MRPs are a 3 dimensional vector co-directional to the axis of rotation and whose
        magnitude is equal to ``tan(theta / 4)``, where ``theta`` is the angle of rotation
        (in radians) [1]_.

        MRPs have a singularity at 360 degrees which can be avoided by ensuring the angle of
        rotation does not exceed 180 degrees, i.e. switching the direction of the rotation when
        it is past 180 degrees. This function will always return MRPs corresponding to a rotation
        of less than or equal to 180 degrees.

        Returns
        -------
        mrps : ndarray, shape (3,) or (N, 3)
            Shape depends on shape of inputs used for initialization.

        References
        ----------
        .. [1] Shuster, M. D. "A Survey of Attitude Representations",
               The Journal of Astronautical Sciences, Vol. 41, No.4, 1993,
               pp. 475-476

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Represent a single rotation:

        >>> r = R.from_rotvec([0, 0, np.pi])
        >>> r.as_mrp()
        array([0.        , 0.        , 1.         ])
        >>> r.as_mrp().shape
        (3,)

        Represent a stack with a single rotation:

        >>> r = R.from_euler('xyz', [[180, 0, 0]], degrees=True)
        >>> r.as_mrp()
        array([[1.       , 0.        , 0.         ]])
        >>> r.as_mrp().shape
        (1, 3)

        Represent multiple rotations:

        >>> r = R.from_rotvec([[np.pi/2, 0, 0], [0, 0, np.pi/2]])
        >>> r.as_mrp()
        array([[0.41421356, 0.        , 0.        ],
               [0.        , 0.        , 0.41421356]])
        >>> r.as_mrp().shape
        (2, 3)

        Notes
        -----

        .. versionadded:: 1.6.0
        """
        # 获取旋转数量
        cdef Py_ssize_t num_rotations = len(self._quat)
        # 创建一个大小为 (num_rotations, 3) 的双精度浮点型数组
        cdef double[:, :] mrps = _empty2(num_rotations, 3)
        # 定义符号变量
        cdef int sign
        # 定义分母变量
        cdef double denominator

        # 遍历每个旋转
        for ind in range(num_rotations):

            # 确保计算的 MRPs 对应的旋转角度 <= 180
            sign = -1 if self._quat[ind, 3] < 0 else 1

            denominator = 1 + sign * self._quat[ind, 3]
            for i in range(3):
                # 计算 MRPs 的每个分量
                mrps[ind, i] = sign * self._quat[ind, i] / denominator

        # 如果是单个旋转，返回第一个 MRPs 数组；否则返回整个 MRPs 数组
        if self._single:
            return np.asarray(mrps[0])
        else:
            return np.asarray(mrps)
    def concatenate(cls, rotations):
        """
        Concatenate a sequence of `Rotation` objects into a single object.

        This is useful if you want to, for example, take the mean of a set of
        rotations and need to pack them into a single object to do so.

        Parameters
        ----------
        rotations : sequence of `Rotation` objects
            The rotations to concatenate.

        Returns
        -------
        concatenated : `Rotation` instance
            The concatenated rotations.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> r1 = R.from_rotvec([0, 0, 1])
        >>> r2 = R.from_rotvec([0, 0, 2])
        >>> rc = R.concatenate([r1, r2])
        >>> rc.as_rotvec()
        array([[0., 0., 1.],
               [0., 0., 2.]])
        >>> rc.mean().as_rotvec()
        array([0., 0., 1.5])

        Note that it may be simpler to create the desired rotations by passing
        in a single list of the data during initialization, rather then by
        concatenating:

        >>> R.from_rotvec([[0, 0, 1], [0, 0, 2]]).as_rotvec()
        array([[0., 0., 1.],
               [0., 0., 2.]])

        Notes
        -----
        .. versionadded:: 1.8.0
        """
        if not all(isinstance(x, Rotation) for x in rotations):
            raise TypeError("input must contain Rotation objects only")

        # Concatenate quaternion arrays from each Rotation object
        quats = np.concatenate([np.atleast_2d(x.as_quat()) for x in rotations])
        return cls(quats, normalize=False)

    @cython.embedsignature(True)
    @cython.embedsignature(True)
    @cython.embedsignature(True)
    @cython.embedsignature(True)
    def inv(self):
        """
        Invert this rotation.

        Composition of a rotation with its inverse results in an identity
        transformation.

        Returns
        -------
        inverse : `Rotation` instance
            Object containing inverse of the rotations in the current instance.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Inverting a single rotation:

        >>> p = R.from_euler('z', 45, degrees=True)
        >>> q = p.inv()
        >>> q.as_euler('zyx', degrees=True)
        array([-45.,   0.,   0.])

        Inverting multiple rotations:

        >>> p = R.from_rotvec([[0, 0, np.pi/3], [-np.pi/4, 0, 0]])
        >>> q = p.inv()
        >>> q.as_rotvec()
        array([[-0.        , -0.        , -1.04719755],
               [ 0.78539816, -0.        , -0.        ]])

        """
        # Create a copy of the quaternion array and invert each component
        cdef np.ndarray quat = np.array(self._quat, copy=True)
        quat[:, 0] *= -1
        quat[:, 1] *= -1
        quat[:, 2] *= -1

        # If _single flag is set, convert quat to a single-row array
        if self._single:
            quat = quat[0]

        # Return a new instance of the class with the inverted quaternion
        return self.__class__(quat, normalize=False, copy=False)

    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    # 定义一个方法，计算旋转的幅度（角度）。

    def magnitude(self):
        """Get the magnitude(s) of the rotation(s).

        Returns
        -------
        magnitude : ndarray or float
            Angle(s) in radians, float if object contains a single rotation
            and ndarray if object contains multiple rotations. The magnitude
            will always be in the range [0, pi].

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np
        >>> r = R.from_quat(np.eye(4))
        >>> r.as_quat()
        array([[ 1., 0., 0., 0.],
               [ 0., 1., 0., 0.],
               [ 0., 0., 1., 0.],
               [ 0., 0., 0., 1.]])
        >>> r.magnitude()
        array([3.14159265, 3.14159265, 3.14159265, 0.        ])

        Magnitude of a single rotation:

        >>> r[0].magnitude()
        3.141592653589793
        """
        # 使用 Cython 的 cdef 声明定义变量 quat，表示旋转的四元数
        cdef double[:, :] quat = self._quat
        # 使用 Cython 的 cdef 声明定义变量 num_rotations，表示旋转的数量
        cdef Py_ssize_t num_rotations = quat.shape[0]
        # 使用 Cython 的 cdef 声明定义变量 angles，用于存储计算后的角度
        cdef double[:] angles = _empty1(num_rotations)

        # 循环计算每个旋转的幅度（角度）
        for ind in range(num_rotations):
            # 计算角度，使用 atan2 函数和 _norm3 函数
            angles[ind] = 2 * atan2(_norm3(quat[ind, :3]), abs(quat[ind, 3]))

        # 如果只包含单个旋转，返回单个角度值；否则返回角度数组
        if self._single:
            return angles[0]
        else:
            return np.asarray(angles)


    # 使用 Cython 的 embedsignature 进行函数签名嵌入
    @cython.embedsignature(True)
    # 使用 Cython 的 boundscheck 关闭边界检查
    @cython.boundscheck(False)
    # 使用 Cython 的 wraparound 关闭负数索引的循环访问检查
    @cython.wraparound(False)
    # 判断另一个旋转对象是否与当前旋转对象近似相等
    def approx_equal(Rotation self, Rotation other, atol=None, degrees=False):
        """Determine if another rotation is approximately equal to this one.

        Equality is measured by calculating the smallest angle between the
        rotations, and checking to see if it is smaller than `atol`.

        Parameters
        ----------
        other : `Rotation` instance
            Object containing the rotations to measure against this one.
        atol : float, optional
            The absolute angular tolerance, below which the rotations are
            considered equal. If not given, then set to 1e-8 radians by
            default.
        degrees : bool, optional
            If True and `atol` is given, then `atol` is measured in degrees. If
            False (default), then atol is measured in radians.

        Returns
        -------
        approx_equal : ndarray or bool
            Whether the rotations are approximately equal, bool if object
            contains a single rotation and ndarray if object contains multiple
            rotations.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np
        >>> p = R.from_quat([0, 0, 0, 1])
        >>> q = R.from_quat(np.eye(4))
        >>> p.approx_equal(q)
        array([False, False, False, True])

        Approximate equality for a single rotation:

        >>> p.approx_equal(q[0])
        False
        """
        # 如果未提供 `atol` 参数，则根据 `degrees` 标志设定默认的角度容差
        if atol is None:
            if degrees:
                # 如果 `degrees` 为 True，警告用户必须设置 `atol` 以使用角度标志，默认设置为 1e-8 弧度。
                warnings.warn("atol must be set to use the degrees flag, "
                              "defaulting to 1e-8 radians.")
            # 默认情况下，角度容差设置为 1e-8 弧度
            atol = 1e-8  # radians
        # 如果 `degrees` 为 True 并且提供了 `atol` 参数，则将角度转换为弧度
        elif degrees:
            atol = np.deg2rad(atol)

        # 计算另一个旋转对象与当前旋转对象逆的乘积的角度大小
        angles = (other * self.inv()).magnitude()
        # 返回判断两个旋转对象角度是否小于 `atol` 的结果
        return angles < atol

    # 嵌入函数的签名到 Cython 中
    @cython.embedsignature(True)
    def mean(self, weights=None):
        """Get the mean of the rotations.

        The mean used is the chordal L2 mean (also called the projected or
        induced arithmetic mean) [1]_. If ``A`` is a set of rotation matrices,
        then the mean ``M`` is the rotation matrix that minimizes the
        following loss function:

        .. math::

            L(M) = \\sum_{i = 1}^{n} w_i \\lVert \\mathbf{A}_i -
            \\mathbf{M} \\rVert^2 ,

        where :math:`w_i`'s are the `weights` corresponding to each matrix.

        Parameters
        ----------
        weights : array_like shape (N,), optional
            Weights describing the relative importance of the rotations. If
            None (default), then all values in `weights` are assumed to be
            equal.

        Returns
        -------
        mean : `Rotation` instance
            Object containing the mean of the rotations in the current
            instance.

        References
        ----------
        .. [1] Hartley, Richard, et al.,
                "Rotation Averaging", International Journal of Computer Vision
                103, 2013, pp. 267-305.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> r = R.from_euler('zyx', [[0, 0, 0],
        ...                          [1, 0, 0],
        ...                          [0, 1, 0],
        ...                          [0, 0, 1]], degrees=True)
        >>> r.mean().as_euler('zyx', degrees=True)
        array([0.24945696, 0.25054542, 0.24945696])
        """
        # Check if weights are provided; if not, use equal weights for all rotations
        if weights is None:
            weights = np.ones(len(self))
        else:
            # Convert weights to numpy array and validate shape
            weights = np.asarray(weights)
            if weights.ndim != 1:
                raise ValueError("Expected `weights` to be 1 dimensional, got "
                                 "shape {}.".format(weights.shape))
            # Ensure the number of weights matches the number of rotations
            if weights.shape[0] != len(self):
                raise ValueError("Expected `weights` to have number of values "
                                 "equal to number of rotations, got "
                                 "{} values and {} rotations.".format(
                                    weights.shape[0], len(self)))
            # Check if all weights are non-negative
            if np.any(weights < 0):
                raise ValueError("`weights` must be non-negative.")

        # Convert rotation object to quaternion array
        quat = np.asarray(self._quat)
        # Compute the matrix K using weighted quaternions
        K = np.dot(weights * quat.T, quat)
        # Compute eigenvalues and eigenvectors of K
        _, v = np.linalg.eigh(K)
        # Return a new Rotation object with the eigenvector corresponding to the largest eigenvalue
        return self.__class__(v[:, -1], normalize=False)
    # 定义一个类方法，用于创建指定的三维旋转群。
    def create_group(cls, group, axis='Z'):
        """Create a 3D rotation group.

        Parameters
        ----------
        group : string
            The name of the group. Must be one of 'I', 'O', 'T', 'Dn', 'Cn',
            where `n` is a positive integer. The groups are:

                * I: Icosahedral group
                * O: Octahedral group
                * T: Tetrahedral group
                * D: Dicyclic group
                * C: Cyclic group

        axis : integer
            The cyclic rotation axis. Must be one of ['X', 'Y', 'Z'] (or
            lowercase). Default is 'Z'. Ignored for groups 'I', 'O', and 'T'.

        Returns
        -------
        rotation : `Rotation` instance
            Object containing the elements of the rotation group.

        Notes
        -----
        This method generates rotation groups only. The full 3-dimensional
        point groups [PointGroups]_ also contain reflections.

        References
        ----------
        .. [PointGroups] `Point groups
           <https://en.wikipedia.org/wiki/Point_groups_in_three_dimensions>`_
           on Wikipedia.
        """
        # 调用另一个函数 `create_group`，用给定的参数创建指定的旋转群
        return create_group(cls, group, axis=axis)

    # 启用Cython的embedsignature功能，以便在编译时将函数签名嵌入到生成的代码中
    @cython.embedsignature(True)
    def __getitem__(self, indexer):
        """Extract rotation(s) at given index(es) from object.

        Create a new `Rotation` instance containing a subset of rotations
        stored in this object.

        Parameters
        ----------
        indexer : index, slice, or index array
            Specifies which rotation(s) to extract. A single indexer must be
            specified, i.e. as if indexing a 1 dimensional array or list.

        Returns
        -------
        rotation : `Rotation` instance
            Contains
                - a single rotation, if `indexer` is a single index
                - a stack of rotation(s), if `indexer` is a slice, or and index
                  array.

        Raises
        ------
        TypeError if the instance was created as a single rotation.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> r = R.from_quat([
        ... [1, 1, 0, 0],
        ... [0, 1, 0, 1],
        ... [1, 1, -1, 0]])
        >>> r.as_quat()
        array([[ 0.70710678,  0.70710678,  0.        ,  0.        ],
               [ 0.        ,  0.70710678,  0.        ,  0.70710678],
               [ 0.57735027,  0.57735027, -0.57735027,  0.        ]])

        Indexing using a single index:

        >>> p = r[0]
        >>> p.as_quat()
        array([0.70710678, 0.70710678, 0.        , 0.        ])

        Array slicing:

        >>> q = r[1:3]
        >>> q.as_quat()
        array([[ 0.        ,  0.70710678,  0.        ,  0.70710678],
               [ 0.57735027,  0.57735027, -0.57735027,  0.        ]])

        """
        if self._single:
            raise TypeError("Single rotation is not subscriptable.")

        # 返回一个新的 Rotation 实例，包含根据 indexer 提取的旋转子集
        return self.__class__(np.asarray(self._quat)[indexer], normalize=False)

    def __setitem__(self, indexer, value):
        """Set rotation(s) at given index(es) from object.

        Parameters
        ----------
        indexer : index, slice, or index array
            Specifies which rotation(s) to replace. A single indexer must be
            specified, i.e. as if indexing a 1 dimensional array or list.

        value : `Rotation` instance
            The rotations to set.

        Raises
        ------
        TypeError if the instance was created as a single rotation.
        TypeError if `value` is not a Rotation object.

        Notes
        -----
        .. versionadded:: 1.8.0
        """
        if self._single:
            raise TypeError("Single rotation is not subscriptable.")

        if not isinstance(value, Rotation):
            raise TypeError("value must be a Rotation object")

        # 将 value 转换为四元数数组，并替换 self._quat 中的对应索引或切片
        quat = np.asarray(self._quat)
        quat[indexer] = value.as_quat()
        self._quat = quat

    @cython.embedsignature(True)
    @classmethod
    def identity(cls, num=None):
        """Get identity rotation(s).

        Composition with the identity rotation has no effect.

        Parameters
        ----------
        num : int or None, optional
            Number of identity rotations to generate. If None (default), then a
            single rotation is generated.

        Returns
        -------
        identity : Rotation object
            The identity rotation.
        """
        # 如果 num 为 None，则生成单个四元数表示的单位旋转
        if num is None:
            q = [0, 0, 0, 1]
        else:
            # 否则生成 num 个单位旋转，每个都是四元数表示的 [0, 0, 0, 1]
            q = np.zeros((num, 4))
            q[:, 3] = 1
        # 使用生成的四元数数组创建 Rotation 对象，不进行归一化处理
        return cls(q, normalize=False)

    @cython.embedsignature(True)
    @classmethod
    def random(cls, num=None, random_state=None):
        """Generate uniformly distributed rotations.

        Parameters
        ----------
        num : int or None, optional
            Number of random rotations to generate. If None (default), then a
            single rotation is generated.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Returns
        -------
        random_rotation : `Rotation` instance
            Contains a single rotation if `num` is None. Otherwise contains a
            stack of `num` rotations.

        Notes
        -----
        This function is optimized for efficiently sampling random rotation
        matrices in three dimensions. For generating random rotation matrices
        in higher dimensions, see `scipy.stats.special_ortho_group`.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R

        Sample a single rotation:

        >>> R.random().as_euler('zxy', degrees=True)
        array([-110.5976185 ,   55.32758512,   76.3289269 ])  # random

        Sample a stack of rotations:

        >>> R.random(5).as_euler('zxy', degrees=True)
        array([[-110.5976185 ,   55.32758512,   76.3289269 ],  # random
               [ -91.59132005,  -14.3629884 ,  -93.91933182],
               [  25.23835501,   45.02035145, -121.67867086],
               [ -51.51414184,  -15.29022692, -172.46870023],
               [ -81.63376847,  -27.39521579,    2.60408416]])

        See Also
        --------
        scipy.stats.special_ortho_group

       """
        # 使用给定的随机状态对象进行随机数生成
        random_state = check_random_state(random_state)

        # 如果 num 为 None，则生成单个四元数表示的随机旋转
        if num is None:
            sample = random_state.normal(size=4)
        else:
            # 否则生成 num 个随机旋转，每个都是四元数表示的随机向量
            sample = random_state.normal(size=(num, 4))

        # 使用生成的四元数数组创建 Rotation 对象
        return cls(sample)

    @cython.embedsignature(True)
    @classmethod
class Slerp:
    """Spherical Linear Interpolation of Rotations.

    The interpolation between consecutive rotations is performed as a rotation
    around a fixed axis with a constant angular velocity [1]_. This ensures
    that the interpolated rotations follow the shortest path between initial
    and final orientations.

    Parameters
    ----------
    times : array_like, shape (N,)
        Times of the known rotations. At least 2 times must be specified.
    rotations : `Rotation` instance
        Rotations to perform the interpolation between. Must contain N
        rotations.

    Methods
    -------
    __call__
        Perform spherical linear interpolation at specified times.

    See Also
    --------
    Rotation

    Notes
    -----
    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp

    Examples
    --------
    >>> from scipy.spatial.transform import Rotation as R
    >>> from scipy.spatial.transform import Slerp

    Setup the fixed keyframe rotations and times:

    >>> key_rots = R.random(5, random_state=2342345)
    >>> key_times = [0, 1, 2, 3, 4]

    Create the interpolator object:

    >>> slerp = Slerp(key_times, key_rots)

    Interpolate the rotations at the given times:

    >>> times = [0, 0.5, 0.25, 1, 1.5, 2, 2.75, 3, 3.25, 3.60, 4]
    >>> interp_rots = slerp(times)

    The keyframe rotations expressed as Euler angles:

    >>> key_rots.as_euler('xyz', degrees=True)
    array([[ 14.31443779, -27.50095894,  -3.7275787 ],
           [ -1.79924227, -24.69421529, 164.57701743],
           [146.15020772,  43.22849451, -31.34891088],
           [ 46.39959442,  11.62126073, -45.99719267],
           [-88.94647804, -49.64400082, -65.80546984]])

    The interpolated rotations expressed as Euler angles. These agree with the
    keyframe rotations at both endpoints of the range of keyframe times.

    >>> interp_rots.as_euler('xyz', degrees=True)
    array([[  14.31443779,  -27.50095894,   -3.7275787 ],
           [   4.74588574,  -32.44683966,   81.25139984],
           [  10.71094749,  -31.56690154,   38.06896408],
           [  -1.79924227,  -24.69421529,  164.57701743],
           [  11.72796022,   51.64207311, -171.7374683 ],
           [ 146.15020772,   43.22849451,  -31.34891088],
           [  68.10921869,   20.67625074,  -48.74886034],
           [  46.39959442,   11.62126073,  -45.99719267],
           [  12.35552615,    4.21525086,  -64.89288124],
           [ -30.08117143,  -19.90769513,  -78.98121326],
           [ -88.94647804,  -49.64400082,  -65.80546984]])

    """
    def __init__(self, times, rotations):
        # 检查 rotations 是否为 Rotation 的实例，如果不是则抛出类型错误异常
        if not isinstance(rotations, Rotation):
            raise TypeError("`rotations` must be a `Rotation` instance.")

        # 检查 rotations 是否包含至少两个旋转操作，如果不是则抛出数值错误异常
        if rotations.single or len(rotations) == 1:
            raise ValueError("`rotations` must be a sequence of at least 2 rotations.")

        # 将 times 转换为 NumPy 数组
        times = np.asarray(times)
        # 检查 times 是否为一维数组，如果不是则抛出数值错误异常
        if times.ndim != 1:
            raise ValueError("Expected times to be specified in a 1 "
                             "dimensional array, got {} "
                             "dimensions.".format(times.ndim))

        # 检查 times 数组的长度是否与 rotations 的长度相同，如果不同则抛出数值错误异常
        if times.shape[0] != len(rotations):
            raise ValueError("Expected number of rotations to be equal to "
                             "number of timestamps given, got {} rotations "
                             "and {} timestamps.".format(
                                len(rotations), times.shape[0]))
        
        # 将 times 和 rotations 分别作为对象的属性存储
        self.times = times
        # 计算时间间隔数组，并存储为对象的属性
        self.timedelta = np.diff(times)

        # 检查时间间隔数组中是否存在非严格递增的情况，如果有则抛出数值错误异常
        if np.any(self.timedelta <= 0):
            raise ValueError("Times must be in strictly increasing order.")

        # 设置对象的属性来存储 rotations 的前 N-1 个元素
        self.rotations = rotations[:-1]
        # 计算旋转向量，并存储为对象的属性
        self.rotvecs = (self.rotations.inv() * rotations[1:]).as_rotvec()

    def __call__(self, times):
        """Interpolate rotations.

        Compute the interpolated rotations at the given `times`.

        Parameters
        ----------
        times : array_like
            Times to compute the interpolations at. Can be a scalar or
            1-dimensional.

        Returns
        -------
        interpolated_rotation : `Rotation` instance
            Object containing the rotations computed at given `times`.

        """
        # 将输入的 times 转换为 NumPy 数组，确保为最多一维的数组
        compute_times = np.asarray(times)
        if compute_times.ndim > 1:
            raise ValueError("`times` must be at most 1-dimensional.")

        # 判断输入的 times 是否为单个时间点（标量），如果是则进行处理
        single_time = compute_times.ndim == 0
        compute_times = np.atleast_1d(compute_times)

        # 使用二分查找确定插值所需的索引位置
        # 注意：side = 'left' 表示排除左边界（不包含最小时间点）
        ind = np.searchsorted(self.times, compute_times) - 1
        # 对于与 self.times 中第一个时间点相等的 compute_times，将其索引设置为 0
        ind[compute_times == self.times[0]] = 0
        # 检查索引是否超出有效范围，如果超出则抛出数值错误异常
        if np.any(np.logical_or(ind < 0, ind > len(self.rotations) - 1)):
            raise ValueError("Interpolation times must be within the range "
                             "[{}, {}], both inclusive.".format(
                                self.times[0], self.times[-1]))

        # 计算插值系数 alpha
        alpha = (compute_times - self.times[ind]) / self.timedelta[ind]

        # 根据插值系数计算插值后的旋转对象
        result = (self.rotations[ind] *
                  Rotation.from_rotvec(self.rotvecs[ind] * alpha[:, None]))

        # 如果输入的 times 是单个时间点（标量），则返回单个旋转对象而不是数组
        if single_time:
            result = result[0]

        return result
```