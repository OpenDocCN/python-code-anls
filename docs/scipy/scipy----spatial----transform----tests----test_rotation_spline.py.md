# `D:\src\scipysrc\scipy\scipy\spatial\transform\tests\test_rotation_spline.py`

```
from itertools import product  # 导入 itertools 模块中的 product 函数，用于生成迭代器的笛卡尔积
import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy.testing import assert_allclose  # 导入 NumPy 提供的测试工具函数，用于检查数组是否接近
from pytest import raises  # 导入 pytest 库中的 raises 函数，用于检查是否抛出异常
from scipy.spatial.transform import Rotation, RotationSpline  # 导入 SciPy 库中的旋转和旋转样条类
from scipy.spatial.transform._rotation_spline import (  # 导入 SciPy 库中的旋转样条私有函数
    _angular_rate_to_rotvec_dot_matrix,
    _rotvec_dot_to_angular_rate_matrix,
    _matrix_vector_product_of_stacks,
    _angular_acceleration_nonlinear_term,
    _create_block_3_diagonal_matrix)


def test_angular_rate_to_rotvec_conversions():
    np.random.seed(0)  # 设置随机种子，确保可重复性
    rv = np.random.randn(4, 3)  # 生成一个服从标准正态分布的 4x3 数组
    A = _angular_rate_to_rotvec_dot_matrix(rv)  # 调用函数计算旋转向量速率的矩阵表示
    A_inv = _rotvec_dot_to_angular_rate_matrix(rv)  # 调用函数计算旋转向量速率的逆矩阵表示

    # 当旋转向量与角速率对齐时，旋转向量速率和角速率相同
    assert_allclose(_matrix_vector_product_of_stacks(A, rv), rv)
    assert_allclose(_matrix_vector_product_of_stacks(A_inv, rv), rv)

    # A 和 A_inv 应互为逆矩阵
    I_stack = np.empty((4, 3, 3))
    I_stack[:] = np.eye(3)
    assert_allclose(np.matmul(A, A_inv), I_stack, atol=1e-15)


def test_angular_rate_nonlinear_term():
    # 简单测试以确保当旋转向量与自身时，非线性项为零
    np.random.seed(0)  # 设置随机种子，确保可重复性
    rv = np.random.rand(4, 3)  # 生成一个服从均匀分布的 4x3 数组
    assert_allclose(_angular_acceleration_nonlinear_term(rv, rv), 0,
                    atol=1e-19)


def test_create_block_3_diagonal_matrix():
    np.random.seed(0)  # 设置随机种子，确保可重复性
    A = np.empty((4, 3, 3))  # 创建一个空的 4x3x3 数组
    A[:] = np.arange(1, 5)[:, None, None]  # 将数组 A 的每个元素设为不同的值

    B = np.empty((4, 3, 3))  # 创建一个空的 4x3x3 数组
    B[:] = -np.arange(1, 5)[:, None, None]  # 将数组 B 的每个元素设为负的不同的值
    d = 10 * np.arange(10, 15)  # 创建一个包含一系列数值的数组

    banded = _create_block_3_diagonal_matrix(A, B, d)  # 调用函数生成一个带状三对角矩阵

    # 将带状矩阵转换为完整矩阵
    k, l = list(zip(*product(np.arange(banded.shape[0]),
                             np.arange(banded.shape[1]))))  # 生成笛卡尔积的索引列表
    k = np.asarray(k)
    l = np.asarray(l)

    i = k - 5 + l  # 计算行索引
    j = l  # 列索引
    values = banded.ravel()  # 将带状矩阵展平为一维数组
    mask = (i >= 0) & (i < 15)  # 创建一个布尔掩码
    i = i[mask]  # 应用掩码得到有效的行索引
    j = j[mask]  # 应用掩码得到有效的列索引
    values = values[mask]  # 应用掩码得到有效的数值
    full = np.zeros((15, 15))  # 创建一个全零的 15x15 数组
    full[i, j] = values  # 根据索引将数值填入完整矩阵中

    zero = np.zeros((3, 3))  # 创建一个全零的 3x3 数组
    eye = np.eye(3)  # 创建一个 3x3 的单位矩阵

    # 以最直接的方式创建参考的完整矩阵
    ref = np.block([
        [d[0] * eye, B[0], zero, zero, zero],
        [A[0], d[1] * eye, B[1], zero, zero],
        [zero, A[1], d[2] * eye, B[2], zero],
        [zero, zero, A[2], d[3] * eye, B[3]],
        [zero, zero, zero, A[3], d[4] * eye],
    ])

    assert_allclose(full, ref, atol=1e-19)


def test_spline_2_rotations():
    times = [0, 10]  # 定义时间点列表
    rotations = Rotation.from_euler('xyz', [[0, 0, 0], [10, -20, 30]],  # 创建旋转对象
                                    degrees=True)
    spline = RotationSpline(times, rotations)  # 创建旋转样条对象

    rv = (rotations[0].inv() * rotations[1]).as_rotvec()  # 计算旋转向量
    rate = rv / (times[1] - times[0])  # 计算旋转速率
    times_check = np.array([-1, 5, 12])  # 创建用于检查的时间点数组
    dt = times_check - times[0]  # 计算时间差
    rv_ref = rate * dt[:, None]  # 计算参考的旋转向量

    assert_allclose(spline(times_check).as_rotvec(), rv_ref)  # 检查旋转样条的输出是否符合预期
    # 使用 assert_allclose 函数验证 spline(times_check, 1) 的返回值与 np.resize(rate, (3, 3)) 是否在数值上接近
    assert_allclose(spline(times_check, 1), np.resize(rate, (3, 3)))
    
    # 使用 assert_allclose 函数验证 spline(times_check, 2) 的返回值是否数值上接近 0，允许的绝对误差为 1e-16
    assert_allclose(spline(times_check, 2), 0, atol=1e-16)
# 定义一个测试函数，用于验证 RotationSpline 类的常态行为
def test_constant_attitude():
    # 创建一个包含 [0, 1, ..., 9] 的 NumPy 数组
    times = np.arange(10)
    # 创建一个表示旋转向量的 Rotation 对象数组，每个向量都是 [1, 1, 1]
    rotations = Rotation.from_rotvec(np.ones((10, 3)))
    # 使用给定的时间点和旋转向量创建 RotationSpline 对象
    spline = RotationSpline(times, rotations)

    # 创建一个从 -1 到 11 的均匀分布的时间点数组
    times_check = np.linspace(-1, 11)
    # 验证在指定时间点处的旋转向量是否接近 [1, 1, 1]，相对误差不超过 1e-15
    assert_allclose(spline(times_check).as_rotvec(), 1, rtol=1e-15)
    # 验证在指定时间点处的旋转速度是否接近 0，绝对误差不超过 1e-17
    assert_allclose(spline(times_check, 1), 0, atol=1e-17)
    # 验证在指定时间点处的旋转加速度是否接近 0，绝对误差不超过 1e-17
    assert_allclose(spline(times_check, 2), 0, atol=1e-17)

    # 验证在时间点 5.5 处的旋转向量是否接近 [1, 1, 1]，相对误差不超过 1e-15
    assert_allclose(spline(5.5).as_rotvec(), 1, rtol=1e-15)
    # 验证在时间点 5.5 处的旋转速度是否接近 0，绝对误差不超过 1e-17
    assert_allclose(spline(5.5, 1), 0, atol=1e-17)
    # 验证在时间点 5.5 处的旋转加速度是否接近 0，绝对误差不超过 1e-17
    assert_allclose(spline(5.5, 2), 0, atol=1e-17)


# 定义一个测试函数，用于验证 RotationSpline 类的属性
def test_spline_properties():
    # 创建一个包含 [0, 5, 15, 27] 的 NumPy 数组，作为时间点
    times = np.array([0, 5, 15, 27])
    # 创建一个包含四个旋转向量列表的二维数组，表示每个时间点的旋转向量
    angles = [[-5, 10, 27], [3, 5, 38], [-12, 10, 25], [-15, 20, 11]]

    # 使用欧拉角创建 Rotation 对象数组
    rotations = Rotation.from_euler('xyz', angles, degrees=True)
    # 使用给定的时间点和旋转向量创建 RotationSpline 对象
    spline = RotationSpline(times, rotations)

    # 验证在给定时间点处的旋转向量是否接近给定的欧拉角，相对误差不超过默认的误差容限
    assert_allclose(spline(times).as_euler('xyz', degrees=True), angles)
    # 验证在时间点 0 处的旋转向量是否接近 [-5, 10, 27]，相对误差不超过默认的误差容限
    assert_allclose(spline(0).as_euler('xyz', degrees=True), angles[0])

    h = 1e-8
    # 计算在给定时间点处的旋转向量的数值梯度
    rv0 = spline(times).as_rotvec()
    rvm = spline(times - h).as_rotvec()
    rvp = spline(times + h).as_rotvec()
    # 验证数值梯度是否满足近似条件，使用稍大的相对误差容限
    assert_allclose(rv0, 0.5 * (rvp + rvm), rtol=1.5e-15)

    # 计算在给定时间点处的旋转速度的数值梯度
    r0 = spline(times, 1)
    rm = spline(times - h, 1)
    rp = spline(times + h, 1)
    # 验证数值梯度是否满足近似条件，使用稍大的相对误差容限
    assert_allclose(r0, 0.5 * (rm + rp), rtol=1e-14)

    # 计算在给定时间点处的旋转加速度的数值梯度
    a0 = spline(times, 2)
    am = spline(times - h, 2)
    ap = spline(times + h, 2)
    # 验证数值梯度是否满足近似条件，使用较大的相对误差容限
    assert_allclose(a0, am, rtol=1e-7)
    assert_allclose(a0, ap, rtol=1e-7)


# 定义一个测试函数，用于验证 RotationSpline 类的错误处理能力
def test_error_handling():
    # 验证 RotationSpline 初始化时对时间点和旋转向量的错误处理能力
    raises(ValueError, RotationSpline, [1.0], Rotation.random())

    # 创建包含 10 个随机 Rotation 对象的数组
    r = Rotation.random(10)
    # 创建一个不符合要求的时间点数组，并验证其是否能触发 ValueError
    t = np.arange(10).reshape(5, 2)
    raises(ValueError, RotationSpline, t, r)

    # 创建一个长度不符合要求的时间点数组，并验证其是否能触发 ValueError
    t = np.arange(9)
    raises(ValueError, RotationSpline, t, r)

    # 创建一个包含重复时间点的数组，并验证其是否能触发 ValueError
    t = np.arange(10)
    t[5] = 0
    raises(ValueError, RotationSpline, t, r)

    t = np.arange(10)

    # 创建 RotationSpline 对象
    s = RotationSpline(t, r)
    # 验证在不符合要求的参数输入时是否能触发 ValueError
    raises(ValueError, s, 10, -1)

    # 验证在不符合要求的时间点数组输入时是否能触发 ValueError
    raises(ValueError, s, np.arange(10).reshape(5, 2))
```