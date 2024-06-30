# `D:\src\scipysrc\scipy\scipy\spatial\transform\tests\test_rotation.py`

```
# 导入 pytest 库，用于单元测试
import pytest

# 导入 numpy 库，并从中导入测试函数 assert_equal, assert_array_almost_equal, assert_allclose
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose

# 导入 scipy 库中的 spatial.transform 模块中的 Rotation 类和 Slerp 类
from scipy.spatial.transform import Rotation, Slerp

# 导入 scipy 库中的 stats 模块，从中导入 special_ortho_group 函数
from scipy.stats import special_ortho_group

# 导入 itertools 库中的 permutations 函数
from itertools import permutations

# 导入 pickle 库
import pickle

# 导入 copy 库
import copy


# 定义函数 basis_vec，根据给定的轴向返回对应的基向量
def basis_vec(axis):
    if axis == 'x':
        return [1, 0, 0]
    elif axis == 'y':
        return [0, 1, 0]
    elif axis == 'z':
        return [0, 0, 1]

# 定义测试函数 test_generic_quat_matrix
def test_generic_quat_matrix():
    # 定义输入矩阵 x
    x = np.array([[3, 4, 0, 0], [5, 12, 0, 0]])
    # 使用 Rotation 类的 from_quat 方法创建旋转对象 r
    r = Rotation.from_quat(x)
    # 计算期望的四元数值 expected_quat
    expected_quat = x / np.array([[5], [13]])
    # 断言旋转对象 r 的四元数表示与期望值 expected_quat 几乎相等
    assert_array_almost_equal(r.as_quat(), expected_quat)


# 定义测试函数 test_from_single_1d_quaternion
def test_from_single_1d_quaternion():
    # 定义输入的单个一维四元数 x
    x = np.array([3, 4, 0, 0])
    # 使用 Rotation 类的 from_quat 方法创建旋转对象 r
    r = Rotation.from_quat(x)
    # 计算期望的四元数值 expected_quat
    expected_quat = x / 5
    # 断言旋转对象 r 的四元数表示与期望值 expected_quat 几乎相等
    assert_array_almost_equal(r.as_quat(), expected_quat)


# 定义测试函数 test_from_single_2d_quaternion
def test_from_single_2d_quaternion():
    # 定义输入的单个二维四元数 x
    x = np.array([[3, 4, 0, 0]])
    # 使用 Rotation 类的 from_quat 方法创建旋转对象 r
    r = Rotation.from_quat(x)
    # 计算期望的四元数值 expected_quat
    expected_quat = x / 5
    # 断言旋转对象 r 的四元数表示与期望值 expected_quat 几乎相等
    assert_array_almost_equal(r.as_quat(), expected_quat)


# 定义测试函数 test_from_quat_scalar_first
def test_from_quat_scalar_first():
    # 创建随机数生成器 rng
    rng = np.random.RandomState(0)

    # 使用 Rotation 类的 from_quat 方法创建旋转对象 r，使用 scalar_first 参数设为 True
    r = Rotation.from_quat([1, 0, 0, 0], scalar_first=True)
    # 断言旋转对象 r 的旋转矩阵表示与单位矩阵几乎相等
    assert_allclose(r.as_matrix(), np.eye(3), rtol=1e-15, atol=1e-16)

    # 使用 Rotation 类的 from_quat 方法创建旋转对象 r，输入是 10 个相同的四元数，使用 scalar_first 参数设为 True
    r = Rotation.from_quat(np.tile([1, 0, 0, 0], (10, 1)), scalar_first=True)
    # 断言旋转对象 r 的旋转矩阵表示与 10 个单位矩阵的堆叠几乎相等
    assert_allclose(r.as_matrix(), np.tile(np.eye(3), (10, 1, 1)),
                    rtol=1e-15, atol=1e-16)

    # 生成随机的四元数数组 q
    q = rng.randn(100, 4)
    q /= np.linalg.norm(q, axis=1)[:, None]
    # 对每个四元数 qi 进行测试
    for qi in q:
        # 使用 Rotation 类的 from_quat 方法创建旋转对象 r，使用 scalar_first 参数设为 True
        r = Rotation.from_quat(qi, scalar_first=True)
        # 断言旋转对象 r 的四元数表示向右循环移位 1 位后与输入的四元数 qi 几乎相等
        assert_allclose(np.roll(r.as_quat(), 1), qi, rtol=1e-15)

    # 使用 Rotation 类的 from_quat 方法创建旋转对象 r，输入是四元数数组 q，使用 scalar_first 参数设为 True
    r = Rotation.from_quat(q, scalar_first=True)
    # 断言旋转对象 r 的四元数表示向右循环移位 1 位后与输入的四元数数组 q 几乎相等
    assert_allclose(np.roll(r.as_quat(), 1, axis=1), q, rtol=1e-15)


# 定义测试函数 test_as_quat_scalar_first
def test_as_quat_scalar_first():
    # 创建随机数生成器 rng
    rng = np.random.RandomState(0)

    # 使用 Rotation 类的 from_euler 方法创建旋转对象 r，欧拉角为零向量，然后调用 as_quat 方法，使用 scalar_first 参数设为 True
    r = Rotation.from_euler('xyz', np.zeros(3))
    # 断言旋转对象 r 的四元数表示与 [1, 0, 0, 0] 几乎相等
    assert_allclose(r.as_quat(scalar_first=True), [1, 0, 0, 0],
                    rtol=1e-15, atol=1e-16)

    # 使用 Rotation 类的 from_euler 方法创建旋转对象 r，欧拉角为零向量的数组，然后调用 as_quat 方法，使用 scalar_first 参数设为 True
    r = Rotation.from_euler('xyz', np.zeros((10, 3)))
    # 断言旋转对象 r 的四元数表示与 10 个 [1, 0, 0, 0] 的堆叠几乎相等
    assert_allclose(r.as_quat(scalar_first=True),
                    np.tile([1, 0, 0, 0], (10, 1)), rtol=1e-15, atol=1e-16)

    # 生成随机的四元数数组 q
    q = rng.randn(100, 4)
    q /= np.linalg.norm(q, axis=1)[:, None]
    # 对每个四元数 qi 进行测试
    for qi in q:
        # 使用 Rotation 类的 from_quat 方法创建旋转对象 r
        r = Rotation.from_quat(qi)
        # 断言旋转对象 r 的四元数表示向右循环移位 1 位后与输入的四元数 qi 几乎相等
        assert_allclose(r.as_quat(scalar_first=True), np.roll(qi, 1),
                        rtol=1e-15)

        # 断言旋转对象 r 的规范化后的四元数表示向右循环移位 1 位后与其规范化的四元数表示几乎相等
        assert_allclose(r.as_quat(canonical=True, scalar_first=True),
                        np.roll(r.as_quat(canonical=True), 1),
                        rtol=1e-15)

    # 使用 Rotation 类的 from_quat 方法创建旋转对象 r，输入是四元数数组 q
    r = Rotation.from_quat(q)
    # 断言旋转对象 r 的四元数表示向右循环移位 1 位后与输入的四元数数组 q 几乎相等
    assert_allclose(r.as_quat(scalar_first=True), np.roll(q, 1, axis=1),
                    rtol=1e-15)

    # 断言旋转对象 r 的规范化后的四元数表示向右循环移位 1 位
    # 创建一个 NumPy 数组，表示一组四元数
    x = np.array([
        [3, 0, 0, 4],
        [5, 0, 12, 0],
        [0, 0, 0, 1],
        [-1, -1, -1, 1],
        [0, 0, 0, -1],  # 检查双覆盖
        [-1, -1, -1, -1]  # 检查双覆盖
        ])
    # 使用 Rotation 类中的 from_quat 方法，将四元数数组转换为旋转对象
    r = Rotation.from_quat(x)
    # 创建一个期望的四元数数组，与给定的 x 数组进行除法操作
    expected_quat = x / np.array([[5], [13], [1], [2], [1], [2]])
    # 断言旋转对象的四元数与期望的四元数数组近似相等
    assert_array_almost_equal(r.as_quat(), expected_quat)
def test_quat_double_to_canonical_single_cover():
    # 创建一个包含特定四元数的 NumPy 数组
    x = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [-1, -1, -1, -1]
        ])
    # 使用 Rotation 类的 from_quat 方法创建 Rotation 对象
    r = Rotation.from_quat(x)
    # 计算期望的规范化四元数
    expected_quat = np.abs(x) / np.linalg.norm(x, axis=1)[:, None]
    # 断言 Rotation 对象的 as_quat 方法返回的结果与期望的规范化四元数相近
    assert_allclose(r.as_quat(canonical=True), expected_quat)


def test_quat_double_cover():
    # 查看 Rotation.from_quat() 的文档字符串，了解四元数双覆盖属性的范围
    # 检查 from_quat 方法和 as_quat(canonical=False) 方法
    q = np.array([0, 0, 0, -1])
    r = Rotation.from_quat(q)
    # 断言 Rotation 对象的 as_quat 方法以 canonical=False 参数返回的结果与输入的四元数 q 相等
    assert_equal(q, r.as_quat(canonical=False))

    # 检查组合和逆运算
    q = np.array([1, 0, 0, 1])/np.sqrt(2)  # 绕 x 轴旋转 90 度
    r = Rotation.from_quat(q)
    r3 = r*r*r
    # 断言旋转后的四元数乘以 sqrt(2) 后与预期值相近
    assert_allclose(r.as_quat(canonical=False)*np.sqrt(2),
                    [1, 0, 0, 1])
    assert_allclose(r.inv().as_quat(canonical=False)*np.sqrt(2),
                    [-1, 0, 0, 1])
    assert_allclose(r3.as_quat(canonical=False)*np.sqrt(2),
                    [1, 0, 0, -1])
    assert_allclose(r3.inv().as_quat(canonical=False)*np.sqrt(2),
                    [-1, 0, 0, -1])

    # 更多的合理性检查
    assert_allclose((r*r.inv()).as_quat(canonical=False),
                    [0, 0, 0, 1], atol=2e-16)
    assert_allclose((r3*r3.inv()).as_quat(canonical=False),
                    [0, 0, 0, 1], atol=2e-16)
    assert_allclose((r*r3).as_quat(canonical=False),
                    [0, 0, 0, -1], atol=2e-16)
    assert_allclose((r.inv()*r3.inv()).as_quat(canonical=False),
                    [0, 0, 0, -1], atol=2e-16)


def test_malformed_1d_from_quat():
    # 使用 pytest 的 raises 方法检测 ValueError 异常
    with pytest.raises(ValueError):
        Rotation.from_quat(np.array([1, 2, 3]))


def test_malformed_2d_from_quat():
    # 使用 pytest 的 raises 方法检测 ValueError 异常
    with pytest.raises(ValueError):
        Rotation.from_quat(np.array([
            [1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8]
            ]))


def test_zero_norms_from_quat():
    # 使用 pytest 的 raises 方法检测 ValueError 异常
    x = np.array([
            [3, 4, 0, 0],
            [0, 0, 0, 0],
            [5, 0, 12, 0]
            ])
    with pytest.raises(ValueError):
        Rotation.from_quat(x)


def test_as_matrix_single_1d_quaternion():
    # 给定一个一维的四元数，创建对应的旋转矩阵
    quat = [0, 0, 0, 1]
    mat = Rotation.from_quat(quat).as_matrix()
    # 断言旋转矩阵的形状为 (3,3)
    assert_array_almost_equal(mat, np.eye(3))


def test_as_matrix_single_2d_quaternion():
    # 给定一个二维数组形式的四元数，创建对应的旋转矩阵
    quat = [[0, 0, 1, 1]]
    mat = Rotation.from_quat(quat).as_matrix()
    # 断言旋转矩阵的形状为 (1,3,3)
    assert_equal(mat.shape, (1, 3, 3))
    expected_mat = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
        ])
    assert_array_almost_equal(mat[0], expected_mat)


def test_as_matrix_from_square_input():
    # 给定一个多个四元数的列表，创建对应的旋转矩阵数组
    quats = [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, -1]
            ]
    mat = Rotation.from_quat(quats).as_matrix()
    # 断言旋转矩阵数组的形状为 (4,3,3)
    assert_equal(mat.shape, (4, 3, 3))
    # 定义预期的第一个旋转矩阵，用 numpy 数组表示
    expected0 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
        ])
    
    # 断言 mat 数组的第一个元素与预期的第一个旋转矩阵相近
    assert_array_almost_equal(mat[0], expected0)
    
    # 定义预期的第二个旋转矩阵，用 numpy 数组表示
    expected1 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
        ])
    
    # 断言 mat 数组的第二个元素与预期的第二个旋转矩阵相近
    assert_array_almost_equal(mat[1], expected1)
    
    # 断言 mat 数组的第三个元素与单位矩阵相近
    assert_array_almost_equal(mat[2], np.eye(3))
    
    # 断言 mat 数组的第四个元素与单位矩阵相近
    assert_array_almost_equal(mat[3], np.eye(3))
# 定义一个测试函数，用于测试从通用输入转换为矩阵的功能
def test_as_matrix_from_generic_input():
    # 定义四元数列表
    quats = [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 2, 3, 4]
            ]
    # 使用四元数列表创建旋转对象，并将其转换为矩阵形式
    mat = Rotation.from_quat(quats).as_matrix()
    # 断言矩阵的形状为 (3, 3, 3)
    assert_equal(mat.shape, (3, 3, 3))

    # 定义期望的第一个子矩阵
    expected0 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
        ])
    # 断言第一个子矩阵与期望值的近似相等
    assert_array_almost_equal(mat[0], expected0)

    # 定义期望的第二个子矩阵
    expected1 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
        ])
    # 断言第二个子矩阵与期望值的近似相等
    assert_array_almost_equal(mat[1], expected1)

    # 定义期望的第三个子矩阵
    expected2 = np.array([
        [0.4, -2, 2.2],
        [2.8, 1, 0.4],
        [-1, 2, 2]
        ]) / 3
    # 断言第三个子矩阵与期望值的近似相等
    assert_array_almost_equal(mat[2], expected2)


# 定义测试函数，测试从单个二维矩阵转换的功能
def test_from_single_2d_matrix():
    # 定义一个二维矩阵
    mat = [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
            ]
    # 定义期望的四元数
    expected_quat = [0.5, 0.5, 0.5, 0.5]
    # 断言从矩阵转换得到的四元数与期望值的近似相等
    assert_array_almost_equal(
            Rotation.from_matrix(mat).as_quat(),
            expected_quat)


# 定义测试函数，测试从单个三维矩阵转换的功能
def test_from_single_3d_matrix():
    # 定义一个三维矩阵
    mat = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
        ]).reshape((1, 3, 3))
    # 定义期望的四元数
    expected_quat = np.array([0.5, 0.5, 0.5, 0.5]).reshape((1, 4))
    # 断言从矩阵转换得到的四元数与期望值的近似相等
    assert_array_almost_equal(
            Rotation.from_matrix(mat).as_quat(),
            expected_quat)


# 定义测试函数，测试从矩阵计算得到四元数的功能
def test_from_matrix_calculation():
    # 定义期望的四元数
    expected_quat = np.array([1, 1, 6, 1]) / np.sqrt(39)
    # 定义一个旋转矩阵
    mat = np.array([
            [-0.8974359, -0.2564103, 0.3589744],
            [0.3589744, -0.8974359, 0.2564103],
            [0.2564103, 0.3589744, 0.8974359]
            ])
    # 断言从矩阵转换得到的四元数与期望值的近似相等
    assert_array_almost_equal(
            Rotation.from_matrix(mat).as_quat(),
            expected_quat)
    # 断言从重塑后的矩阵转换得到的四元数与重塑后的期望值的近似相等
    assert_array_almost_equal(
            Rotation.from_matrix(mat.reshape((1, 3, 3))).as_quat(),
            expected_quat.reshape((1, 4)))


# 定义测试函数，测试矩阵计算管道功能
def test_matrix_calculation_pipeline():
    # 生成一个特殊正交矩阵
    mat = special_ortho_group.rvs(3, size=10, random_state=0)
    # 断言从矩阵转换得到的矩阵与原始矩阵的近似相等
    assert_array_almost_equal(Rotation.from_matrix(mat).as_matrix(), mat)


# 定义测试函数，测试从矩阵转换得到的正交矩阵的功能
def test_from_matrix_ortho_output():
    # 创建一个随机状态
    rnd = np.random.RandomState(0)
    # 生成随机矩阵
    mat = rnd.random_sample((100, 3, 3))
    # 使用矩阵创建旋转对象，并将其转换为矩阵形式
    ortho_mat = Rotation.from_matrix(mat).as_matrix()

    # 计算正交矩阵的乘积结果
    mult_result = np.einsum('...ij,...jk->...ik', ortho_mat,
                            ortho_mat.transpose((0, 2, 1)))

    # 创建一个单位三维矩阵
    eye3d = np.zeros((100, 3, 3))
    for i in range(3):
        eye3d[:, i, i] = 1.0

    # 断言乘积结果与单位三维矩阵的近似相等
    assert_array_almost_equal(mult_result, eye3d)


# 定义测试函数，测试从单个一维旋转向量转换的功能
def test_from_1d_single_rotvec():
    # 定义一个旋转向量
    rotvec = [1, 0, 0]
    # 定义期望的四元数
    expected_quat = np.array([0.4794255, 0, 0, 0.8775826])
    # 从旋转向量创建旋转对象，然后将其转换为四元数
    result = Rotation.from_rotvec(rotvec)
    # 断言从旋转对象得到的四元数与期望值的近似相等
    assert_array_almost_equal(result.as_quat(), expected_quat)


# 定义测试函数，测试从单个二维旋转向量转换的功能
def test_from_2d_single_rotvec():
    # 定义一个二维旋转向量
    rotvec = [[1, 0, 0]]
    # 定义期望的四元数
    expected_quat = np.array([[0.4794255, 0, 0, 0.8775826]])
    # 从旋转向量创建旋转对象，然后将其转换为四元数
    result = Rotation.from_rotvec(rotvec)
    # 断言从旋转对象得到的四元数与期望值的近似相等
    assert_array_almost_equal(result.as_quat(), expected_quat)


# 定义测试函数，测试从通用旋转向量转换的功能
def test_from_generic_rotvec():
    # 定义一个通用旋转向量列表
    rotvec = [
            [1, 2, 2],
            [1, -1, 0.5],
            [0, 0, 0]
            ]
    # 定义预期的四元数数组，表示三个旋转向量对应的四元数表示
    expected_quat = np.array([
        [0.3324983, 0.6649967, 0.6649967, 0.0707372],  # 第一个四元数
        [0.4544258, -0.4544258, 0.2272129, 0.7316889], # 第二个四元数
        [0, 0, 0, 1]                                    # 第三个四元数
        ])
    # 使用 assert_array_almost_equal 函数验证通过旋转向量得到的四元数与预期的四元数数组近似相等
    assert_array_almost_equal(
            Rotation.from_rotvec(rotvec).as_quat(),  # 将旋转向量转换为四元数
            expected_quat)                          # 预期的四元数数组作为对比目标
def test_from_rotvec_small_angle():
    # 定义一个旋转向量数组，包含三个向量
    rotvec = np.array([
        [5e-4 / np.sqrt(3), -5e-4 / np.sqrt(3), 5e-4 / np.sqrt(3)],  # 第一个旋转向量
        [0.2, 0.3, 0.4],  # 第二个旋转向量
        [0, 0, 0]  # 第三个旋转向量
        ])

    # 将旋转向量数组转换为四元数表示
    quat = Rotation.from_rotvec(rotvec).as_quat()
    # 对于小角度情况，cos(theta/2)约等于1
    assert_allclose(quat[0, 3], 1)
    # 对于小角度情况，sin(theta/2) / theta约等于0.5
    assert_allclose(quat[0, :3], rotvec[0] * 0.5)

    # 验证第二个旋转向量转换后的四元数
    assert_allclose(quat[1, 3], 0.9639685)
    assert_allclose(
            quat[1, :3],
            np.array([
                0.09879603932153465,
                0.14819405898230198,
                0.19759207864306931
                ]))

    # 验证第三个旋转向量转换后的四元数，应为单位四元数
    assert_equal(quat[2], np.array([0, 0, 0, 1]))


def test_degrees_from_rotvec():
    # 定义一个旋转向量数组，包含三个元素
    rotvec1 = [1.0 / np.cbrt(3), 1.0 / np.cbrt(3), 1.0 / np.cbrt(3)]
    # 创建一个以角度为单位的旋转对象
    rot1 = Rotation.from_rotvec(rotvec1, degrees=True)
    # 将角度旋转对象转换为四元数表示
    quat1 = rot1.as_quat()

    # 将角度旋转向量数组转换为弧度表示
    rotvec2 = np.deg2rad(rotvec1)
    # 创建一个以弧度为单位的旋转对象
    rot2 = Rotation.from_rotvec(rotvec2)
    # 将弧度旋转对象转换为四元数表示
    quat2 = rot2.as_quat()

    # 验证两种转换方法得到的四元数是否接近
    assert_allclose(quat1, quat2)


def test_malformed_1d_from_rotvec():
    # 测试输入为一维列表的情况，预期应抛出形状错误的异常
    with pytest.raises(ValueError, match='Expected `rot_vec` to have shape'):
        Rotation.from_rotvec([1, 2])


def test_malformed_2d_from_rotvec():
    # 测试输入为二维列表的情况，预期应抛出形状错误的异常
    with pytest.raises(ValueError, match='Expected `rot_vec` to have shape'):
        Rotation.from_rotvec([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
            ])


def test_as_generic_rotvec():
    # 定义一个四元数数组
    quat = np.array([
            [1, 2, -1, 0.5],
            [1, -1, 1, 0.0003],
            [0, 0, 0, 1]
            ])
    # 将四元数数组进行单位化
    quat /= np.linalg.norm(quat, axis=1)[:, None]

    # 将单位化后的四元数数组转换为旋转向量表示
    rotvec = Rotation.from_quat(quat).as_rotvec()
    # 计算旋转向量的角度
    angle = np.linalg.norm(rotvec, axis=1)

    # 验证四元数到旋转向量转换是否精确
    assert_allclose(quat[:, 3], np.cos(angle/2))
    assert_allclose(np.cross(rotvec, quat[:, :3]), np.zeros((3, 3)))


def test_as_rotvec_single_1d_input():
    # 定义一个四元数数组
    quat = np.array([1, 2, -3, 2])
    # 预期的旋转向量结果
    expected_rotvec = np.array([0.5772381, 1.1544763, -1.7317144])

    # 将单个四元数转换为旋转向量表示
    actual_rotvec = Rotation.from_quat(quat).as_rotvec()

    # 验证旋转向量的形状和数值是否符合预期
    assert_equal(actual_rotvec.shape, (3,))
    assert_allclose(actual_rotvec, expected_rotvec)


def test_as_rotvec_single_2d_input():
    # 定义一个四元数数组
    quat = np.array([[1, 2, -3, 2]])
    # 预期的旋转向量结果
    expected_rotvec = np.array([[0.5772381, 1.1544763, -1.7317144]])

    # 将单个二维四元数转换为旋转向量表示
    actual_rotvec = Rotation.from_quat(quat).as_rotvec()

    # 验证旋转向量的形状和数值是否符合预期
    assert_equal(actual_rotvec.shape, (1, 3))
    assert_allclose(actual_rotvec, expected_rotvec)


def test_as_rotvec_degrees():
    # 定义一个旋转矩阵
    mat = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    # 创建一个旋转对象
    rot = Rotation.from_matrix(mat)
    # 将旋转对象转换为角度制的旋转向量表示
    rotvec = rot.as_rotvec(degrees=True)
    # 计算旋转向量的角度
    angle = np.linalg.norm(rotvec)
    # 验证角度制旋转向量的角度和各分量是否符合预期
    assert_allclose(angle, 120.0)
    assert_allclose(rotvec[0], rotvec[1])
    assert_allclose(rotvec[1], rotvec[2])


def test_rotvec_calc_pipeline():
    # 定义一个旋转向量数组，包含三个向量
    rotvec = np.array([
        [0, 0, 0],  # 第一个旋转向量
        [1, -1, 2],  # 第二个旋转向量
        [-3e-4, 3.5e-4, 7.5e-5]  # 第三个旋转向量
        ])
    # 验证旋转向量到旋转对象再到旋转向量的计算流水线结果
    assert_allclose(Rotation.from_rotvec(rotvec).as_rotvec(), rotvec)
    # 使用 assert_allclose 函数进行断言，比较两个旋转向量的角度表示是否相等
    assert_allclose(Rotation.from_rotvec(rotvec, degrees=True).as_rotvec(degrees=True),
                    rotvec)
def test_from_1d_single_mrp():
    mrp = [0, 0, 1.0]  # 定义一个长度为3的MRP列表
    expected_quat = np.array([0, 0, 1, 0])  # 定义期望的四元数
    result = Rotation.from_mrp(mrp)  # 调用函数计算MRP对应的旋转
    assert_array_almost_equal(result.as_quat(), expected_quat)  # 断言结果与期望的四元数接近


def test_from_2d_single_mrp():
    mrp = [[0, 0, 1.0]]  # 定义一个包含单个长度为3的MRP列表的二维数组
    expected_quat = np.array([[0, 0, 1, 0]])  # 定义期望的四元数的二维数组形式
    result = Rotation.from_mrp(mrp)  # 调用函数计算MRP对应的旋转
    assert_array_almost_equal(result.as_quat(), expected_quat)  # 断言结果与期望的四元数接近


def test_from_generic_mrp():
    mrp = np.array([  # 定义一个包含多个MRP的二维NumPy数组
        [1, 2, 2],
        [1, -1, 0.5],
        [0, 0, 0]])
    expected_quat = np.array([  # 定义期望的四元数的二维NumPy数组形式
        [0.2, 0.4, 0.4, -0.8],
        [0.61538462, -0.61538462, 0.30769231, -0.38461538],
        [0, 0, 0, 1]])
    assert_array_almost_equal(Rotation.from_mrp(mrp).as_quat(), expected_quat)  # 断言结果与期望的四元数接近


def test_malformed_1d_from_mrp():
    with pytest.raises(ValueError, match='Expected `mrp` to have shape'):  # 使用pytest断言捕获值错误异常
        Rotation.from_mrp([1, 2])  # 调用函数，传入错误形状的MRP数组


def test_malformed_2d_from_mrp():
    with pytest.raises(ValueError, match='Expected `mrp` to have shape'):  # 使用pytest断言捕获值错误异常
        Rotation.from_mrp([  # 调用函数，传入错误形状的MRP数组
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])


def test_as_generic_mrp():
    quat = np.array([  # 定义一个包含多个四元数的二维NumPy数组
        [1, 2, -1, 0.5],
        [1, -1, 1, 0.0003],
        [0, 0, 0, 1]])
    quat /= np.linalg.norm(quat, axis=1)[:, None]  # 对四元数数组进行单位化处理

    expected_mrp = np.array([  # 定义期望的MRP数组的二维NumPy数组形式
        [0.33333333, 0.66666667, -0.33333333],
        [0.57725028, -0.57725028, 0.57725028],
        [0, 0, 0]])
    assert_array_almost_equal(Rotation.from_quat(quat).as_mrp(), expected_mrp)  # 断言结果与期望的MRP数组接近


def test_past_180_degree_rotation():
    # 确保大于180度的旋转以小于180度的MRP形式返回
    # 在这种情况下，270度应该返回为-90度的MRP
    expected_mrp = np.array([-np.tan(np.pi/2/4), 0.0, 0])  # 定义期望的MRP
    assert_array_almost_equal(
        Rotation.from_euler('xyz', [270, 0, 0], degrees=True).as_mrp(),  # 调用函数计算欧拉角到MRP的转换
        expected_mrp
    )


def test_as_mrp_single_1d_input():
    quat = np.array([1, 2, -3, 2])  # 定义一个四元数
    expected_mrp = np.array([0.16018862, 0.32037724, -0.48056586])  # 定义期望的MRP
    actual_mrp = Rotation.from_quat(quat).as_mrp()  # 调用函数计算四元数到MRP的转换

    assert_equal(actual_mrp.shape, (3,))  # 断言结果的形状为(3,)
    assert_allclose(actual_mrp, expected_mrp)  # 断言结果与期望的MRP接近


def test_as_mrp_single_2d_input():
    quat = np.array([[1, 2, -3, 2]])  # 定义一个包含单个四元数的二维NumPy数组
    expected_mrp = np.array([[0.16018862, 0.32037724, -0.48056586]])  # 定义期望的MRP的二维NumPy数组形式
    actual_mrp = Rotation.from_quat(quat).as_mrp()  # 调用函数计算四元数到MRP的转换

    assert_equal(actual_mrp.shape, (1, 3))  # 断言结果的形状为(1, 3)
    assert_allclose(actual_mrp, expected_mrp)  # 断言结果与期望的MRP接近


def test_mrp_calc_pipeline():
    actual_mrp = np.array([  # 定义一个包含多个MRP的二维NumPy数组
        [0, 0, 0],
        [1, -1, 2],
        [0.41421356, 0, 0],
        [0.1, 0.2, 0.1]])
    expected_mrp = np.array([  # 定义期望的MRP的二维NumPy数组形式
        [0, 0, 0],
        [-0.16666667, 0.16666667, -0.33333333],
        [0.41421356, 0, 0],
        [0.1, 0.2, 0.1]])
    assert_allclose(Rotation.from_mrp(actual_mrp).as_mrp(), expected_mrp)  # 断言结果与期望的MRP接近


def test_from_euler_single_rotation():
    quat = Rotation.from_euler('z', 90, degrees=True).as_quat()  # 计算绕z轴旋转90度的欧拉角对应的四元数
    expected_quat = np.array([0, 0, 1, 1]) / np.sqrt(2)  # 定义期望的四元数
    assert_allclose(quat, expected_quat)  # 断言计算得到的四元数与期望的四元数接近
def test_single_intrinsic_extrinsic_rotation():
    # 创建一个旋转矩阵，表示绕z轴逆时针旋转90度的外旋转
    extrinsic = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    # 创建一个旋转矩阵，表示绕Z轴逆时针旋转90度的内旋转
    intrinsic = Rotation.from_euler('Z', 90, degrees=True).as_matrix()
    # 断言两个矩阵近似相等
    assert_allclose(extrinsic, intrinsic)


def test_from_euler_rotation_order():
    # 内在旋转顺序与外在旋转顺序相反的简单测试
    rnd = np.random.RandomState(0)
    a = rnd.randint(low=0, high=180, size=(6, 3))
    b = a[:, ::-1]
    # 使用给定顺序a创建四元数表示的旋转
    x = Rotation.from_euler('xyz', a, degrees=True).as_quat()
    # 使用相反顺序b创建四元数表示的旋转
    y = Rotation.from_euler('ZYX', b, degrees=True).as_quat()
    # 断言两个四元数近似相等
    assert_allclose(x, y)


def test_from_euler_elementary_extrinsic_rotation():
    # 简单测试以检查外在旋转是否正确实现
    mat = Rotation.from_euler('zx', [90, 90], degrees=True).as_matrix()
    expected_mat = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])
    # 断言计算出的旋转矩阵与预期矩阵几乎相等
    assert_array_almost_equal(mat, expected_mat)


def test_from_euler_intrinsic_rotation_312():
    angles = [
        [30, 60, 45],
        [30, 60, 30],
        [45, 30, 60]
        ]
    mat = Rotation.from_euler('ZXY', angles, degrees=True).as_matrix()

    assert_array_almost_equal(mat[0], np.array([
        [0.3061862, -0.2500000, 0.9185587],
        [0.8838835, 0.4330127, -0.1767767],
        [-0.3535534, 0.8660254, 0.3535534]
    ]))

    assert_array_almost_equal(mat[1], np.array([
        [0.5334936, -0.2500000, 0.8080127],
        [0.8080127, 0.4330127, -0.3995191],
        [-0.2500000, 0.8660254, 0.4330127]
    ]))

    assert_array_almost_equal(mat[2], np.array([
        [0.0473672, -0.6123725, 0.7891491],
        [0.6597396, 0.6123725, 0.4355958],
        [-0.7500000, 0.5000000, 0.4330127]
    ]))


def test_from_euler_intrinsic_rotation_313():
    angles = [
        [30, 60, 45],
        [30, 60, 30],
        [45, 30, 60]
        ]
    mat = Rotation.from_euler('ZXZ', angles, degrees=True).as_matrix()

    assert_array_almost_equal(mat[0], np.array([
        [0.43559574, -0.78914913, 0.4330127],
        [0.65973961, -0.04736717, -0.750000],
        [0.61237244, 0.61237244, 0.500000]
    ]))

    assert_array_almost_equal(mat[1], np.array([
        [0.6250000, -0.64951905, 0.4330127],
        [0.64951905, 0.1250000, -0.750000],
        [0.4330127, 0.750000, 0.500000]
    ]))

    assert_array_almost_equal(mat[2], np.array([
        [-0.1767767, -0.91855865, 0.35355339],
        [0.88388348, -0.30618622, -0.35355339],
        [0.4330127, 0.25000000, 0.8660254]
    ]))


def test_from_euler_extrinsic_rotation_312():
    angles = [
        [30, 60, 45],
        [30, 60, 30],
        [45, 30, 60]
        ]
    # 使用给定顺序angles创建外在旋转顺序为'zxy'的旋转矩阵
    mat = Rotation.from_euler('zxy', angles, degrees=True).as_matrix()

    assert_array_almost_equal(mat[0], np.array([
        [0.91855865, 0.1767767, 0.35355339],
        [0.25000000, 0.4330127, -0.8660254],
        [-0.30618622, 0.88388348, 0.35355339]
    ]))
    # 断言：验证 mat 数组中索引为 1 的元素与给定的 numpy 数组几乎相等
    assert_array_almost_equal(mat[1], np.array([
        [0.96650635, -0.0580127, 0.25],
        [0.25, 0.4330127, -0.8660254],
        [-0.0580127, 0.89951905, 0.4330127]
    ]))
    
    # 断言：验证 mat 数组中索引为 2 的元素与给定的 numpy 数组几乎相等
    assert_array_almost_equal(mat[2], np.array([
        [0.65973961, -0.04736717, 0.75],
        [0.61237244, 0.61237244, -0.5],
        [-0.43559574, 0.78914913, 0.4330127]
    ]))
# 定义一个测试函数，用于测试从欧拉角到外部旋转矩阵（313）的转换
def test_from_euler_extrinsic_rotation_313():
    # 定义三个欧拉角度数序列的列表
    angles = [
        [30, 60, 45],
        [30, 60, 30],
        [45, 30, 60]
    ]
    # 使用欧拉角序列创建旋转对象，并将其转换为旋转矩阵
    mat = Rotation.from_euler('zxz', angles, degrees=True).as_matrix()

    # 断言第一个旋转矩阵近似等于给定的数值矩阵
    assert_array_almost_equal(mat[0], np.array([
        [0.43559574, -0.65973961, 0.61237244],
        [0.78914913, -0.04736717, -0.61237244],
        [0.4330127, 0.75000000, 0.500000]
    ]))

    # 断言第二个旋转矩阵近似等于给定的数值矩阵
    assert_array_almost_equal(mat[1], np.array([
        [0.62500000, -0.64951905, 0.4330127],
        [0.64951905, 0.12500000, -0.750000],
        [0.4330127, 0.75000000, 0.500000]
    ]))

    # 断言第三个旋转矩阵近似等于给定的数值矩阵
    assert_array_almost_equal(mat[2], np.array([
        [-0.1767767, -0.88388348, 0.4330127],
        [0.91855865, -0.30618622, -0.250000],
        [0.35355339, 0.35355339, 0.8660254]
    ]))


# 参数化测试函数，用于测试欧拉角到轴对称旋转的转换
@pytest.mark.parametrize("seq_tuple", permutations("xyz"))
@pytest.mark.parametrize("intrinsic", (False, True))
def test_as_euler_asymmetric_axes(seq_tuple, intrinsic):
    # 辅助函数，用于执行均值误差测试
    def test_stats(error, mean_max, rms_max):
        # 计算误差的均值和标准差
        mean = np.mean(error, axis=0)
        std = np.std(error, axis=0)
        rms = np.hypot(mean, std)
        # 断言均值误差小于指定的最大均值误差
        assert np.all(np.abs(mean) < mean_max)
        # 断言RMS误差小于指定的最大RMS误差
        assert np.all(rms < rms_max)

    # 使用随机数种子0创建随机状态对象
    rnd = np.random.RandomState(0)
    n = 1000
    # 创建一个形状为(n, 3)的空数组，用于存储随机生成的欧拉角
    angles = np.empty((n, 3))
    # 随机生成欧拉角的三个分量
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles[:, 1] = rnd.uniform(low=-np.pi / 2, high=np.pi / 2, size=(n,))
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))

    # 将欧拉角序列转换为字符串形式
    seq = "".join(seq_tuple)
    if intrinsic:
        # 如果是内禀旋转，将序列转换为大写形式
        seq = seq.upper()
    
    # 使用欧拉角序列创建旋转对象
    rotation = Rotation.from_euler(seq, angles)
    # 将旋转对象转换为欧拉角
    angles_quat = rotation.as_euler(seq)
    # 从旋转矩阵计算欧拉角
    angles_mat = rotation._as_euler_from_matrix(seq)
    # 断言欧拉角与原始欧拉角数组的近似性
    assert_allclose(angles, angles_quat, atol=0, rtol=1e-12)
    assert_allclose(angles, angles_mat, atol=0, rtol=1e-12)
    # 执行欧拉角误差统计测试
    test_stats(angles_quat - angles, 1e-15, 1e-14)
    test_stats(angles_mat - angles, 1e-15, 1e-14)


# 参数化测试函数，用于测试欧拉角到轴对称旋转的转换
@pytest.mark.parametrize("seq_tuple", permutations("xyz"))
@pytest.mark.parametrize("intrinsic", (False, True))
def test_as_euler_symmetric_axes(seq_tuple, intrinsic):
    # 辅助函数，用于执行均值误差测试
    def test_stats(error, mean_max, rms_max):
        # 计算误差的均值和标准差
        mean = np.mean(error, axis=0)
        std = np.std(error, axis=0)
        rms = np.hypot(mean, std)
        # 断言均值误差小于指定的最大均值误差
        assert np.all(np.abs(mean) < mean_max)
        # 断言RMS误差小于指定的最大RMS误差
        assert np.all(rms < rms_max)

    # 使用随机数种子0创建随机状态对象
    rnd = np.random.RandomState(0)
    n = 1000
    # 创建一个形状为(n, 3)的空数组，用于存储随机生成的欧拉角
    angles = np.empty((n, 3))
    # 随机生成欧拉角的三个分量
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles[:, 1] = rnd.uniform(low=0, high=np.pi, size=(n,))
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))

    # 将欧拉角序列转换为字符串形式
    seq = "".join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
    if intrinsic:
        # 如果是内禀旋转，将序列转换为大写形式
        seq = seq.upper()
    # 使用给定的欧拉角序列创建旋转对象
    rotation = Rotation.from_euler(seq, angles)
    # 将旋转对象转换为欧拉角表示（四元数）
    angles_quat = rotation.as_euler(seq)
    # 将旋转对象转换为欧拉角表示（旋转矩阵）
    angles_mat = rotation._as_euler_from_matrix(seq)
    # 断言：检查欧拉角与四元数表示之间的近似相等性
    assert_allclose(angles, angles_quat, atol=0, rtol=1e-13)
    # 断言：检查欧拉角与旋转矩阵表示之间的近似相等性
    assert_allclose(angles, angles_mat, atol=0, rtol=1e-9)
    # 测试：验证欧拉角与四元数表示之间的统计信息
    test_stats(angles_quat - angles, 1e-16, 1e-14)
    # 测试：验证欧拉角与旋转矩阵表示之间的统计信息
    test_stats(angles_mat - angles, 1e-15, 1e-13)
@pytest.mark.parametrize("seq_tuple", permutations("xyz"))
# 对序列元组进行参数化，生成所有可能的排列组合
@pytest.mark.parametrize("intrinsic", (False, True))
# 对 intrinsic 参数进行参数化，测试 False 和 True 两种情况
def test_as_euler_degenerate_asymmetric_axes(seq_tuple, intrinsic):
    # 由于无法检查角度的完全相等性，因此我们检查旋转矩阵的相等性
    # 定义包含角度的 NumPy 数组，以测试非对称轴情况
    angles = np.array([
        [45, 90, 35],
        [35, -90, 20],
        [35, 90, 25],
        [25, -90, 15]])

    seq = "".join(seq_tuple)
    if intrinsic:
        # 内禀旋转（对象自身的旋转）使用大写字母表示
        seq = seq.upper()

    # 使用给定的序列和角度创建 Rotation 对象
    rotation = Rotation.from_euler(seq, angles, degrees=True)
    # 获取预期的旋转矩阵
    mat_expected = rotation.as_matrix()

    # 检查在旋转角度估计时是否发出 UserWarning 提示
    with pytest.warns(UserWarning, match="Gimbal lock"):
        angle_estimates = rotation.as_euler(seq, degrees=True)
    # 使用估计的角度创建 Rotation 对象，获取其旋转矩阵
    mat_estimated = Rotation.from_euler(seq, angle_estimates, degrees=True).as_matrix()

    # 断言预期的旋转矩阵与估计的旋转矩阵几乎相等
    assert_array_almost_equal(mat_expected, mat_estimated)


@pytest.mark.parametrize("seq_tuple", permutations("xyz"))
# 对序列元组进行参数化，生成所有可能的排列组合
@pytest.mark.parametrize("intrinsic", (False, True))
# 对 intrinsic 参数进行参数化，测试 False 和 True 两种情况
def test_as_euler_degenerate_symmetric_axes(seq_tuple, intrinsic):
    # 由于无法检查角度的完全相等性，因此我们检查旋转矩阵的相等性
    # 定义包含角度的 NumPy 数组，以测试对称轴情况
    angles = np.array([
        [15, 0, 60],
        [35, 0, 75],
        [60, 180, 35],
        [15, -180, 25]])

    # 旋转形式为 A/B/A 表示围绕对称轴的旋转
    seq = "".join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
    if intrinsic:
        # 内禀旋转（对象自身的旋转）使用大写字母表示
        seq = seq.upper()

    # 使用给定的序列和角度创建 Rotation 对象
    rotation = Rotation.from_euler(seq, angles, degrees=True)
    # 获取预期的旋转矩阵
    mat_expected = rotation.as_matrix()

    # 检查在旋转角度估计时是否发出 UserWarning 提示
    with pytest.warns(UserWarning, match="Gimbal lock"):
        angle_estimates = rotation.as_euler(seq, degrees=True)
    # 使用估计的角度创建 Rotation 对象，获取其旋转矩阵
    mat_estimated = Rotation.from_euler(seq, angle_estimates, degrees=True).as_matrix()

    # 断言预期的旋转矩阵与估计的旋转矩阵几乎相等
    assert_array_almost_equal(mat_expected, mat_estimated)


@pytest.mark.parametrize("seq_tuple", permutations("xyz"))
# 对序列元组进行参数化，生成所有可能的排列组合
@pytest.mark.parametrize("intrinsic", (False, True))
# 对 intrinsic 参数进行参数化，测试 False 和 True 两种情况
def test_as_euler_degenerate_compare_algorithms(seq_tuple, intrinsic):
    # 此测试确保两种算法在退化情况下做出相同的选择

    # 非对称轴
    angles = np.array([
        [45, 90, 35],
        [35, -90, 20],
        [35, 90, 25],
        [25, -90, 15]])

    seq = "".join(seq_tuple)
    if intrinsic:
        # 内禀旋转（对象自身的旋转）使用大写字母表示
        seq = seq.upper()

    # 使用给定的序列和角度创建 Rotation 对象
    rot = Rotation.from_euler(seq, angles, degrees=True)
    # 检查在旋转角度估计时是否发出 UserWarning 提示
    with pytest.warns(UserWarning, match="Gimbal lock"):
        estimates_matrix = rot._as_euler_from_matrix(seq, degrees=True)
    # 检查在旋转角度估计时是否发出 UserWarning 提示
    with pytest.warns(UserWarning, match="Gimbal lock"):
        estimates_quat = rot.as_euler(seq, degrees=True)
    # 使用 assert_allclose 函数比较矩阵的部分列，确保它们非常接近
    assert_allclose(
        estimates_matrix[:, [0, 2]], estimates_quat[:, [0, 2]], atol=0, rtol=1e-12
    )

    # 再次使用 assert_allclose 函数比较矩阵的另一列，设定宽松的绝对误差容限
    assert_allclose(estimates_matrix[:, 1], estimates_quat[:, 1], atol=0, rtol=1e-7)

    # symmetric axes
    # 绝对误差容限必须更宽松，以直接比较两种算法的结果，因为在接近零角度值附近，
    # _as_euler_from_matrix 方法由于数值精度损失而存在问题

    # 定义一个角度的数组，包含四个角度三元组
    angles = np.array([
        [15, 0, 60],
        [35, 0, 75],
        [60, 180, 35],
        [15, -180, 25]])

    # 找到角度数组中第二列为零的索引，这些角度可能会导致问题
    idx = angles[:, 1] == 0  # find problematic angles indices

    # 根据序列元组构建旋转序列字符串，形式为 A/B/A 表示绕对称轴旋转
    seq = "".join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])

    # 如果是内在旋转（相对于物体自身），则将序列字符串转换为大写
    if intrinsic:
        # Extrinsinc rotation (wrt to global world) at lower case
        # Intrinsic (WRT the object itself) upper case.
        seq = seq.upper()

    # 使用给定的旋转序列和角度创建 Rotation 对象
    rot = Rotation.from_euler(seq, angles, degrees=True)

    # 使用 pytest.warns 检查是否发出 "Gimbal lock" 警告，并计算基于矩阵的欧拉角估计
    with pytest.warns(UserWarning, match="Gimbal lock"):
        estimates_matrix = rot._as_euler_from_matrix(seq, degrees=True)

    # 使用 pytest.warns 检查是否发出 "Gimbal lock" 警告，并计算基于四元数的欧拉角估计
    with pytest.warns(UserWarning, match="Gimbal lock"):
        estimates_quat = rot.as_euler(seq, degrees=True)

    # 再次使用 assert_allclose 函数比较矩阵的部分列，确保它们非常接近
    assert_allclose(
        estimates_matrix[:, [0, 2]], estimates_quat[:, [0, 2]], atol=0, rtol=1e-12
    )

    # 使用 assert_allclose 函数比较矩阵的部分列，排除掉角度为零的问题索引行
    assert_allclose(
        estimates_matrix[~idx, 1], estimates_quat[~idx, 1], atol=0, rtol=1e-7
    )

    # 使用 assert_allclose 函数比较矩阵的部分列，仅对角度为零的问题索引行应用更宽松的绝对误差容限
    assert_allclose(
        estimates_matrix[idx, 1], estimates_quat[idx, 1], atol=1e-6
    )  # problematic, angles[1] = 0
def test_inv():
    # 设置随机种子为0，以确保每次生成的随机数相同
    rnd = np.random.RandomState(0)
    # 设置旋转数量为10，生成随机旋转矩阵 p
    n = 10
    p = Rotation.random(num=n, random_state=rnd)
    # 计算 p 的逆矩阵 q
    q = p.inv()

    # 将 p 和 q 转换为矩阵形式
    p_mat = p.as_matrix()
    q_mat = q.as_matrix()
    # 计算 result1 和 result2，使用 Einstein 求和约定计算矩阵乘积
    result1 = np.einsum('...ij,...jk->...ik', p_mat, q_mat)
    result2 = np.einsum('...ij,...jk->...ik', q_mat, p_mat)

    # 创建一个形状为 (n, 3, 3) 的空数组，并初始化为单位矩阵
    eye3d = np.empty((n, 3, 3))
    eye3d[:] = np.eye(3)

    # 断言 result1 和 result2 等于单位矩阵 eye3d
    assert_array_almost_equal(result1, eye3d)
    assert_array_almost_equal(result2, eye3d)


def test_inv_single_rotation():
    # 设置随机种子为0，以确保每次生成的随机数相同
    rnd = np.random.RandomState(0)
    # 生成单个随机旋转矩阵 p
    p = Rotation.random(random_state=rnd)
    # 计算 p 的逆矩阵 q
    q = p.inv()

    # 将 p 和 q 转换为矩阵形式
    p_mat = p.as_matrix()
    q_mat = q.as_matrix()
    # 计算 res1 和 res2，使用矩阵乘法计算
    res1 = np.dot(p_mat, q_mat)
    res2 = np.dot(q_mat, p_mat)

    # 创建一个 3x3 的单位矩阵 eye
    eye = np.eye(3)

    # 断言 res1 和 res2 等于单位矩阵 eye
    assert_array_almost_equal(res1, eye)
    assert_array_almost_equal(res2, eye)

    # 生成单个随机旋转矩阵 x，并计算其逆矩阵 y
    x = Rotation.random(num=1, random_state=rnd)
    y = x.inv()

    # 将 x 和 y 转换为矩阵形式
    x_matrix = x.as_matrix()
    y_matrix = y.as_matrix()
    # 计算 result1 和 result2，使用 Einstein 求和约定计算矩阵乘积
    result1 = np.einsum('...ij,...jk->...ik', x_matrix, y_matrix)
    result2 = np.einsum('...ij,...jk->...ik', y_matrix, x_matrix)

    # 创建一个形状为 (1, 3, 3) 的空数组，并初始化为单位矩阵
    eye3d = np.empty((1, 3, 3))
    eye3d[:] = np.eye(3)

    # 断言 result1 和 result2 等于单位矩阵 eye3d
    assert_array_almost_equal(result1, eye3d)
    assert_array_almost_equal(result2, eye3d)


def test_identity_magnitude():
    # 设置旋转数量为10，断言单位旋转矩阵的幅值为 0
    n = 10
    assert_allclose(Rotation.identity(n).magnitude(), 0)
    assert_allclose(Rotation.identity(n).inv().magnitude(), 0)


def test_single_identity_magnitude():
    # 断言单个单位旋转矩阵的幅值为 0
    assert Rotation.identity().magnitude() == 0
    assert Rotation.identity().inv().magnitude() == 0


def test_identity_invariance():
    # 设置旋转数量为10，生成随机旋转矩阵 p
    n = 10
    p = Rotation.random(n, random_state=0)

    # 计算 p 与单位旋转矩阵相乘的结果，并断言其四元数表示相等
    result = p * Rotation.identity(n)
    assert_array_almost_equal(p.as_quat(), result.as_quat())

    # 将 result 与 p 的逆矩阵相乘的结果，并断言其幅值为全零数组
    result = result * p.inv()
    assert_array_almost_equal(result.magnitude(), np.zeros(n))


def test_single_identity_invariance():
    # 生成单个随机旋转矩阵 p
    n = 10
    p = Rotation.random(n, random_state=0)

    # 计算 p 与单位旋转矩阵相乘的结果，并断言其四元数表示相等
    result = p * Rotation.identity()
    assert_array_almost_equal(p.as_quat(), result.as_quat())

    # 将 result 与 p 的逆矩阵相乘的结果，并断言其幅值为全零数组
    result = result * p.inv()
    assert_array_almost_equal(result.magnitude(), np.zeros(n))


def test_magnitude():
    # 根据单位四元数创建旋转矩阵 r
    r = Rotation.from_quat(np.eye(4))
    # 计算 r 的幅值，与预期值进行断言
    result = r.magnitude()
    assert_array_almost_equal(result, [np.pi, np.pi, np.pi, 0])

    # 根据负单位四元数创建旋转矩阵 r
    r = Rotation.from_quat(-np.eye(4))
    # 计算 r 的幅值，与预期值进行断言
    result = r.magnitude()
    assert_array_almost_equal(result, [np.pi, np.pi, np.pi, 0])


def test_magnitude_single_rotation():
    # 根据单位四元数创建旋转矩阵 r
    r = Rotation.from_quat(np.eye(4))
    # 计算 r 的第一个分量的幅值，与预期值进行断言
    result1 = r[0].magnitude()
    assert_allclose(result1, np.pi)

    # 计算 r 的最后一个分量的幅值，与预期值进行断言
    result2 = r[3].magnitude()
    assert_allclose(result2, 0)


def test_approx_equal():
    # 设置随机种子为0，生成两组随机旋转矩阵 p 和 q
    rng = np.random.RandomState(0)
    p = Rotation.random(10, random_state=rng)
    q = Rotation.random(10, random_state=rng)
    # 计算 p 和 q 的逆矩阵的乘积，并计算其幅值
    r = p * q.inv()
    r_mag = r.magnitude()
    atol = np.median(r_mag)  # 确保得到混合的 True 和 False
    # 断言 p 是否与 q 近似相等，容差为 atol
    assert_equal(p.approx_equal(q, atol), (r_mag < atol))


def test_approx_equal_single_rotation():
    # 创建一个 Rotation 对象 p，使用 from_rotvec 方法，传入旋转向量 [0, 0, 1e-9]
    # 这个向量小于默认的容差值 atol=1e-8
    p = Rotation.from_rotvec([0, 0, 1e-9])  # less than default atol of 1e-8
    
    # 创建一个 Rotation 对象 q，使用 from_quat 方法，传入单位四元数 np.eye(4)
    q = Rotation.from_quat(np.eye(4))
    
    # 断言 p 是否大致等于 q[3]，即 Rotation 对象 q 的第四个元素
    assert p.approx_equal(q[3])
    
    # 断言 p 是否不等于 q[0]
    assert not p.approx_equal(q[0])
    
    # 使用自定义的容差值 atol=1e-10 来检查 p 是否不等于 q[3]
    assert not p.approx_equal(q[3], atol=1e-10)
    
    # 使用 degrees=True，检查 p 是否不等于 q[3]，且容差值以角度计算（而非弧度）
    assert not p.approx_equal(q[3], atol=1e-8, degrees=True)
    
    # 使用 degrees=True 且未设置容差值来检查 p 是否与 q[3] 大致相等，预期会产生 UserWarning 警告
    with pytest.warns(UserWarning, match="atol must be set"):
        assert p.approx_equal(q[3], degrees=True)
def test_mean():
    # 创建包含单位矩阵和其负对角线的轴数组
    axes = np.concatenate((-np.eye(3), np.eye(3)))
    # 在 [0, π/2] 范围内生成 100 个均匀分布的角度
    thetas = np.linspace(0, np.pi / 2, 100)
    # 对每个角度 t 执行旋转向量操作并计算平均值
    for t in thetas:
        r = Rotation.from_rotvec(t * axes)
        # 断言旋转的平均值的幅度接近于 0，允许的误差为 1E-10
        assert_allclose(r.mean().magnitude(), 0, atol=1E-10)


def test_weighted_mean():
    # 测试加权平均是否等效于多次包含同一旋转
    axes = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    # 在 [0, π/2] 范围内生成 100 个均匀分布的角度
    thetas = np.linspace(0, np.pi / 2, 100)
    # 对每个角度 t 执行旋转向量操作并计算加权平均值
    for t in thetas:
        # 使用前两个轴创建旋转对象 rw
        rw = Rotation.from_rotvec(t * axes[:2])
        # 计算加权平均值 mw
        mw = rw.mean(weights=[1, 2])

        # 使用所有三个轴创建旋转对象 r
        r = Rotation.from_rotvec(t * axes)
        # 计算普通平均值 m
        m = r.mean()
        # 断言 m 乘以 mw 的逆的幅度接近于 0，允许的误差为 1E-10
        assert_allclose((m * mw.inv()).magnitude(), 0, atol=1E-10)


def test_mean_invalid_weights():
    # 测试非法权重是否引发 ValueError 异常，异常信息包含 "non-negative"
    with pytest.raises(ValueError, match="non-negative"):
        r = Rotation.from_quat(np.eye(4))
        r.mean(weights=-np.ones(4))


def test_reduction_no_indices():
    # 测试旋转对象的简化操作，返回的结果应为 Rotation 类的实例
    result = Rotation.identity().reduce(return_indices=False)
    assert isinstance(result, Rotation)


def test_reduction_none_indices():
    # 测试旋转对象的简化操作，返回的结果应为包含三个元素的元组
    result = Rotation.identity().reduce(return_indices=True)
    assert type(result) == tuple
    assert len(result) == 3

    reduced, left_best, right_best = result
    # 确保 left_best 和 right_best 均为 None
    assert left_best is None
    assert right_best is None


def test_reduction_scalar_calculation():
    # 测试旋转对象的简化操作与标量计算的等效性
    rng = np.random.RandomState(0)
    l = Rotation.random(5, random_state=rng)
    r = Rotation.random(10, random_state=rng)
    p = Rotation.random(7, random_state=rng)
    reduced, left_best, right_best = p.reduce(l, r, return_indices=True)

    # 使用循环实现 Rotation.reduce 方法中的向量化计算
    scalars = np.zeros((len(l), len(p), len(r)))
    for i, li in enumerate(l):
        for j, pj in enumerate(p):
            for k, rk in enumerate(r):
                scalars[i, j, k] = np.abs((li * pj * rk).as_quat()[3])
    scalars = np.reshape(np.moveaxis(scalars, 1, 0), (scalars.shape[1], -1))

    max_ind = np.argmax(np.reshape(scalars, (len(p), -1)), axis=1)
    left_best_check = max_ind // len(r)
    right_best_check = max_ind % len(r)
    # 断言简化后的结果与计算的结果在幅度上几乎相等
    assert (left_best == left_best_check).all()
    assert (right_best == right_best_check).all()

    reduced_check = l[left_best_check] * p * r[right_best_check]
    mag = (reduced.inv() * reduced_check).magnitude()
    # 断言 mag 数组的值几乎为零
    assert_array_almost_equal(mag, np.zeros(len(p)))


def test_apply_single_rotation_single_point():
    # 测试单个旋转对单个点的应用
    mat = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    r_1d = Rotation.from_matrix(mat)
    r_2d = Rotation.from_matrix(np.expand_dims(mat, axis=0))

    v_1d = np.array([1, 2, 3])
    v_2d = np.expand_dims(v_1d, axis=0)
    v1d_rotated = np.array([-2, 1, 3])
    v2d_rotated = np.expand_dims(v1d_rotated, axis=0)

    # 断言旋转应用后点的位置几乎相等
    assert_allclose(r_1d.apply(v_1d), v1d_rotated)
    assert_allclose(r_1d.apply(v_2d), v2d_rotated)
    assert_allclose(r_2d.apply(v_1d), v2d_rotated)
    assert_allclose(r_2d.apply(v_2d), v2d_rotated)

    v1d_inverse = np.array([2, -1, 3])
    # 将一维数组 v1d_inverse 扩展为二维数组 v2d_inverse
    v2d_inverse = np.expand_dims(v1d_inverse, axis=0)
    
    # 断言：验证 r_1d 对 v_1d 应用逆转换后的结果是否与 v1d_inverse 接近
    assert_allclose(r_1d.apply(v_1d, inverse=True), v1d_inverse)
    
    # 断言：验证 r_1d 对 v_2d 应用逆转换后的结果是否与 v2d_inverse 接近
    assert_allclose(r_1d.apply(v_2d, inverse=True), v2d_inverse)
    
    # 断言：验证 r_2d 对 v_1d 应用逆转换后的结果是否与 v2d_inverse 接近
    assert_allclose(r_2d.apply(v_1d, inverse=True), v2d_inverse)
    
    # 断言：验证 r_2d 对 v_2d 应用逆转换后的结果是否与 v2d_inverse 接近
    assert_allclose(r_2d.apply(v_2d, inverse=True), v2d_inverse)
def test_apply_single_rotation_multiple_points():
    # 创建一个旋转矩阵
    mat = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    # 使用旋转矩阵创建旋转对象 r1
    r1 = Rotation.from_matrix(mat)
    # 在矩阵 mat 增加一个维度后创建旋转对象 r2
    r2 = Rotation.from_matrix(np.expand_dims(mat, axis=0))

    # 创建一个包含多个点的向量 v
    v = np.array([[1, 2, 3], [4, 5, 6]])
    # 预期的旋转后的向量 v_rotated
    v_rotated = np.array([[-2, 1, 3], [-5, 4, 6]])

    # 断言应用旋转 r1 后得到 v_rotated
    assert_allclose(r1.apply(v), v_rotated)
    # 断言应用旋转 r2 后得到 v_rotated
    assert_allclose(r2.apply(v), v_rotated)

    # 预期的逆向旋转后的向量 v_inverse
    v_inverse = np.array([[2, -1, 3], [5, -4, 6]])

    # 断言应用逆向旋转 r1 后得到 v_inverse
    assert_allclose(r1.apply(v, inverse=True), v_inverse)
    # 断言应用逆向旋转 r2 后得到 v_inverse
    assert_allclose(r2.apply(v, inverse=True), v_inverse)


def test_apply_multiple_rotations_single_point():
    # 创建包含多个旋转矩阵的数组 mat
    mat = np.empty((2, 3, 3))
    mat[0] = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    mat[1] = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    # 使用多个旋转矩阵创建旋转对象 r
    r = Rotation.from_matrix(mat)

    # 创建单个点的向量 v1 和增加一个维度后的向量 v2
    v1 = np.array([1, 2, 3])
    v2 = np.expand_dims(v1, axis=0)

    # 预期的旋转后的向量 v_rotated
    v_rotated = np.array([[-2, 1, 3], [1, -3, 2]])

    # 断言应用旋转 r 后得到 v_rotated
    assert_allclose(r.apply(v1), v_rotated)
    # 断言应用旋转 r 后得到 v_rotated
    assert_allclose(r.apply(v2), v_rotated)

    # 预期的逆向旋转后的向量 v_inverse
    v_inverse = np.array([[2, -1, 3], [1, 3, -2]])

    # 断言应用逆向旋转 r 后得到 v_inverse
    assert_allclose(r.apply(v1, inverse=True), v_inverse)
    # 断言应用逆向旋转 r 后得到 v_inverse
    assert_allclose(r.apply(v2, inverse=True), v_inverse)


def test_apply_multiple_rotations_multiple_points():
    # 创建包含多个旋转矩阵的数组 mat
    mat = np.empty((2, 3, 3))
    mat[0] = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    mat[1] = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    # 使用多个旋转矩阵创建旋转对象 r
    r = Rotation.from_matrix(mat)

    # 创建包含多个点的向量 v
    v = np.array([[1, 2, 3], [4, 5, 6]])
    # 预期的旋转后的向量 v_rotated
    v_rotated = np.array([[-2, 1, 3], [4, -6, 5]])
    # 断言应用旋转 r 后得到 v_rotated
    assert_allclose(r.apply(v), v_rotated)

    # 预期的逆向旋转后的向量 v_inverse
    v_inverse = np.array([[2, -1, 3], [4, 6, -5]])
    # 断言应用逆向旋转 r 后得到 v_inverse
    assert_allclose(r.apply(v, inverse=True), v_inverse)


def test_getitem():
    # 创建包含多个旋转矩阵的数组 mat
    mat = np.empty((2, 3, 3))
    mat[0] = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    mat[1] = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    # 使用多个旋转矩阵创建旋转对象 r
    r = Rotation.from_matrix(mat)

    # 断言获取索引为 0 的旋转对象的矩阵表示与 mat[0] 接近
    assert_allclose(r[0].as_matrix(), mat[0], atol=1e-15)
    # 断言获取索引为 1 的旋转对象的矩阵表示与 mat[1] 接近
    assert_allclose(r[1].as_matrix(), mat[1], atol=1e-15)
    # 断言获取索引为 :-1 的旋转对象的矩阵表示与 mat[0] 在增加一个维度后接近
    assert_allclose(r[:-1].as_matrix(), np.expand_dims(mat[0], axis=0), atol=1e-15)


def test_getitem_single():
    # 断言尝试获取 Rotation.identity() 的索引 0 会引发 TypeError，匹配 'not subscriptable'
    with pytest.raises(TypeError, match='not subscriptable'):
        Rotation.identity()[0]


def test_setitem_single():
    # 创建单位旋转对象 r
    r = Rotation.identity()
    # 断言尝试给 r 的索引 0 赋值 Rotation.identity() 会引发 TypeError，匹配 'not subscriptable'
    with pytest.raises(TypeError, match='not subscriptable'):
        r[0] = Rotation.identity()


def test_setitem_slice():
    # 创建随机数生成器 rng
    rng = np.random.RandomState(seed=0)
    # 创建包含 10 个随机旋转对象的数组 r1 和 5 个随机旋转对象的数组 r2
    r1 = Rotation.random(10, random_state=rng)
    r2 = Rotation.random(5, random_state=rng)
    # 将 r2 赋值给 r1 的索引范围为 1 到 6 的元素
    r1[1:6] = r2
    # 断言 r1 的索引范围为 1 到 6 的旋转对象的四元数表示与 r2 相等
    assert_equal(r1[1:6].as_quat(), r2.as_quat())


def test_setitem_integer():
    # 创建随机数生成器 rng
    rng = np.random.RandomState(seed=0)
    # 创建包含 10 个随机旋转对象的数组 r1 和 一个随机旋转对象 r2
    r1 = Rotation.random(10, random_state=rng)
    r2 = Rotation.random(random_state=rng)
    # 将 r2 赋值给 r1 的索引 1 的元素
    r1[1] = r2
    # 断言 r1 的索引 1 的旋转对象的四元数表示与 r2 相等
    assert_equal(r1[1].as_quat(), r2.as_quat())
# 测试设置错误类型的索引时是否会引发TypeError异常
def test_setitem_wrong_type():
    # 创建一个大小为10的随机旋转对象
    r = Rotation.random(10, random_state=0)
    # 使用pytest断言检查是否会抛出TypeError异常，异常信息需要包含'Rotation object'
    with pytest.raises(TypeError, match='Rotation object'):
        # 尝试将旋转对象的索引0设置为整数1，预期会引发异常
        r[0] = 1


# 测试旋转矩阵对象的数量
def test_n_rotations():
    # 创建一个形状为(2, 3, 3)的空numpy数组
    mat = np.empty((2, 3, 3))
    # 填充数组的第一个元素为特定的旋转矩阵
    mat[0] = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    # 填充数组的第二个元素为另一个特定的旋转矩阵
    mat[1] = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    # 使用给定的旋转矩阵创建Rotation对象
    r = Rotation.from_matrix(mat)

    # 使用断言确保Rotation对象的长度为2
    assert_equal(len(r), 2)
    # 使用断言确保选择除最后一个元素外的所有元素后，Rotation对象的长度为1
    assert_equal(len(r[:-1]), 1)


# 测试随机旋转的形状
def test_random_rotation_shape():
    # 创建一个指定随机种子的numpy随机状态对象
    rnd = np.random.RandomState(0)
    # 使用随机状态对象生成一个随机旋转对象，并检查其四元数表示的形状是否为(4,)
    assert_equal(Rotation.random(random_state=rnd).as_quat().shape, (4,))
    # 使用随机状态对象生成一个未指定数量的随机旋转对象，并检查其四元数表示的形状是否为(4,)
    assert_equal(Rotation.random(None, random_state=rnd).as_quat().shape, (4,))

    # 使用随机状态对象生成一个包含1个随机旋转的对象，并检查其四元数表示的形状是否为(1, 4)
    assert_equal(Rotation.random(1, random_state=rnd).as_quat().shape, (1, 4))
    # 使用随机状态对象生成一个包含5个随机旋转的对象，并检查其四元数表示的形状是否为(5, 4)
    assert_equal(Rotation.random(5, random_state=rnd).as_quat().shape, (5, 4))


# 测试对齐向量时不进行任何旋转的情况
def test_align_vectors_no_rotation():
    # 创建两个相同的numpy数组作为输入向量
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = x.copy()

    # 调用对齐向量的方法，返回旋转对象和旋转后的RSSD值
    r, rssd = Rotation.align_vectors(x, y)
    # 使用断言检查旋转矩阵是否近似为单位矩阵
    assert_array_almost_equal(r.as_matrix(), np.eye(3))
    # 使用断言检查RSSD是否接近于0，允许的误差为1e-6
    assert_allclose(rssd, 0, atol=1e-6)


# 测试对齐向量时不存在噪声的情况
def test_align_vectors_no_noise():
    # 创建指定随机种子的numpy随机状态对象
    rnd = np.random.RandomState(0)
    # 创建一个随机旋转对象c
    c = Rotation.random(random_state=rnd)
    # 创建一个形状为(5, 3)的正态分布随机数组b
    b = rnd.normal(size=(5, 3))
    # 对b应用旋转c得到a
    a = c.apply(b)

    # 调用对齐向量的方法，返回估计的旋转对象和RSSD值
    est, rssd = Rotation.align_vectors(a, b)
    # 使用断言检查估计的旋转四元数是否接近于真实的旋转四元数c.as_quat()
    assert_allclose(c.as_quat(), est.as_quat())
    # 使用断言检查RSSD是否接近于0，允许的误差为1e-7
    assert_allclose(rssd, 0, atol=1e-7)


# 测试对齐向量时使用不恰当的旋转情况
def test_align_vectors_improper_rotation():
    # 创建两个示例向量x和y
    x = np.array([[0.89299824, -0.44372674, 0.0752378],
                  [0.60221789, -0.47564102, -0.6411702]])
    y = np.array([[0.02386536, -0.82176463, 0.5693271],
                  [-0.27654929, -0.95191427, -0.1318321]])

    # 调用对齐向量的方法，返回估计的旋转对象和RSSD值
    est, rssd = Rotation.align_vectors(x, y)
    # 使用断言检查x经过估计的旋转应用到y后是否近似于x本身，允许的误差为1e-6
    assert_allclose(x, est.apply(y), atol=1e-6)
    # 使用断言检查RSSD是否接近于0，允许的误差为1e-7
    assert_allclose(rssd, 0, atol=1e-7)


# 测试对齐向量时RSSD敏感性
def test_align_vectors_rssd_sensitivity():
    # 预期的RSSD值
    rssd_expected = 0.141421356237308
    # 预期的敏感性矩阵
    sens_expected = np.array([[0.2, 0. , 0.],
                              [0. , 1.5, 1.],
                              [0. , 1. , 1.]])
    # 允许的误差
    atol = 1e-6
    # 创建两个输入向量a和b
    a = [[0, 1, 0], [0, 1, 1], [0, 1, 1]]
    b = [[1, 0, 0], [1, 1.1, 0], [1, 0.9, 0]]
    # 调用对齐向量的方法，返回估计的旋转对象、RSSD值和敏感性矩阵
    rot, rssd, sens = Rotation.align_vectors(a, b, return_sensitivity=True)
    # 使用断言检查估计的RSSD值是否接近于预期的RSSD值，允许的误差为atol
    assert np.isclose(rssd, rssd_expected, atol=atol)
    # 使用断言检查估计的敏感性矩阵是否接近于预期的敏感性矩阵，允许的误差为atol
    assert np.allclose(sens, sens_expected, atol=atol)


# 测试对齐向量时使用缩放权重
def test_align_vectors_scaled_weights():
    # 创建大小为10的随机旋转对象a和b
    n = 10
    a = Rotation.random(n, random_state=0).apply([1, 0, 0])
    b = Rotation.random(n, random_state=1).apply([1, 0, 0])
    # 设置缩放因子为2
    scale = 2

    # 调用对齐向量的方法，返回估计的旋转对象、RSSD值和协方差矩阵
    est1, rssd1, cov1 = Rotation.align_vectors(a, b, np.ones(n), True)
    est
    # 生成具有正态分布随机数的三维向量数组，表示为 (n_vectors, 3)
    vectors = rnd.normal(size=(n_vectors, 3))
    # 应用给定的旋转到向量数组上，得到旋转后的结果
    result = rot.apply(vectors)

    # 论文中将独立分布的角度误差添加为噪声
    sigma = np.deg2rad(1)
    tolerance = 1.5 * sigma
    # 创建一个旋转对象，其旋转向量从正态分布生成，表示为 (n_vectors, 3)，标准差为 sigma
    noise = Rotation.from_rotvec(
        rnd.normal(
            size=(n_vectors, 3),
            scale=sigma
        )
    )

    # 对结果中的每个向量应用随机的旋转，以模拟姿态误差
    noisy_result = noise.apply(result)

    # 使用 Rotation.align_vectors 函数对噪声后的结果向量和原始向量进行对齐，返回估计的旋转、残差平方和、协方差矩阵
    est, rssd, cov = Rotation.align_vectors(noisy_result, vectors,
                                            return_sensitivity=True)

    # 使用旋转合成计算误差向量，即 rot 与 est 的逆旋转的旋转向量
    error_vector = (rot * est.inv()).as_rotvec()
    # 使用 assert_allclose 检查误差向量的每个分量是否接近于 0，允许的绝对误差为 tolerance
    assert_allclose(error_vector[0], 0, atol=tolerance)
    assert_allclose(error_vector[1], 0, atol=tolerance)
    assert_allclose(error_vector[2], 0, atol=tolerance)

    # 使用协方差矩阵来检查误差边界，每个主对角线元素乘以 sigma
    cov *= sigma
    assert_allclose(cov[0, 0], 0, atol=tolerance)
    assert_allclose(cov[1, 1], 0, atol=tolerance)
    assert_allclose(cov[2, 2], 0, atol=tolerance)

    # 使用 assert_allclose 检查残差平方和（rssd），与 noisy_result 与 est 应用到原始向量之间的差异的平方和的平方根是否接近
    assert_allclose(rssd, np.sum((noisy_result - est.apply(vectors))**2)**0.5)
def test_align_vectors_invalid_input():
    # 测试当输入向量 `a` 的长度不正确时是否引发 ValueError 异常
    with pytest.raises(ValueError, match="Expected input `a` to have shape"):
        Rotation.align_vectors([1, 2, 3, 4], [1, 2, 3])

    # 测试当输入向量 `b` 的长度不正确时是否引发 ValueError 异常
    with pytest.raises(ValueError, match="Expected input `b` to have shape"):
        Rotation.align_vectors([1, 2, 3], [1, 2, 3, 4])

    # 测试当输入向量 `a` 和 `b` 的形状不一致时是否引发 ValueError 异常
    with pytest.raises(ValueError, match="Expected inputs `a` and `b` "
                                         "to have same shapes"):
        Rotation.align_vectors([[1, 2, 3],[4, 5, 6]], [[1, 2, 3]])

    # 测试当权重向量 `weights` 不是一维时是否引发 ValueError 异常
    with pytest.raises(ValueError,
                       match="Expected `weights` to be 1 dimensional"):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], weights=[[1]])

    # 测试当权重向量 `weights` 的长度不正确时是否引发 ValueError 异常
    with pytest.raises(ValueError,
                       match="Expected `weights` to have number of values"):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                               weights=[1, 2, 3])

    # 测试当权重向量 `weights` 中包含负值时是否引发 ValueError 异常
    with pytest.raises(ValueError,
                       match="`weights` may not contain negative values"):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], weights=[-1])

    # 测试当权重向量 `weights` 中包含多个无穷大值时是否引发 ValueError 异常
    with pytest.raises(ValueError,
                       match="Only one infinite weight is allowed"):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                               weights=[np.inf, np.inf])

    # 测试当主向量长度为零时是否引发 ValueError 异常
    with pytest.raises(ValueError,
                       match="Cannot align zero length primary vectors"):
        Rotation.align_vectors([[0, 0, 0]], [[1, 2, 3]])

    # 测试当请求返回敏感度矩阵时是否引发 ValueError 异常
    with pytest.raises(ValueError,
                       match="Cannot return sensitivity matrix"):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                               return_sensitivity=True, weights=[np.inf, 1])

    # 测试当请求返回敏感度矩阵时是否引发 ValueError 异常
    with pytest.raises(ValueError,
                       match="Cannot return sensitivity matrix"):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]],
                               return_sensitivity=True)


def test_align_vectors_align_constrain():
    # 通过将主轴 +X B 与主轴 +Y A 对齐，并围绕此轴旋转，使得次轴 +Y B
    # (次要 b 向量的 [1, 1, 0] 余量) 与 +Z A 轴对齐（次要 a 向量的 [0, 1, 1] 余量）
    atol = 1e-12
    b = [[1, 0, 0], [1, 1, 0]]
    a = [[0, 1, 0], [0, 1, 1]]
    m_expected = np.array([[0, 0, 1],
                           [1, 0, 0],
                           [0, 1, 0]])
    # 测试 Rotation.align_vectors 方法对向量进行对齐和旋转操作
    R, rssd = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    assert_allclose(R.apply(b), a, atol=atol)  # 主向量和次向量完全对齐
    assert np.isclose(rssd, 0, atol=atol)

    # 通过将主轴 +X B 与主轴 +Y A 对齐，并围绕此轴旋转，使得次轴 +Y B
    # (次要 b 向量的 [1, 2, 0] 余量) 与 +Z A 轴对齐（次要 a 向量的 [0, 1, 1] 余量）
    # 测试 Rotation.align_vectors 方法对向量进行对齐和旋转操作
    b = [[1, 0, 0], [1, 2, 0]]
    rssd_expected = 1.0
    R, rssd = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    # 断言检查第一个返回的结果是否与预期结果 a[0] 在给定的误差范围内完全对齐
    assert_allclose(R.apply(b)[0], a[0], atol=atol)  # Only pri aligns exactly
    
    # 断言检查计算得到的 rssd 是否与预期值 rssd_expected 在给定的误差范围内非常接近
    assert np.isclose(rssd, rssd_expected, atol=atol)
    
    # 预期的结果 a_expected 是一个包含两个子列表的列表，用于与 R.apply(b) 的结果进行比较
    a_expected = [[0, 1, 0], [0, 1, 2]]
    
    # 断言检查 R.apply(b) 的结果是否与预期的 a_expected 在给定的误差范围内非常接近
    assert_allclose(R.apply(b), a_expected, atol=atol)

    # 检查使用随机向量 b 和 a 进行旋转后的结果
    b = [[1, 2, 3], [-2, 3, -1]]
    a = [[-1, 3, 2], [1, -1, 2]]
    
    # 预期的 rssd_expected 是旋转后的向量 a 的标准差的预期值
    rssd_expected = 1.3101595297515016
    
    # 使用 Rotation 类的 align_vectors 方法计算旋转矩阵 R 和标准差 rssd
    R, rssd = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    
    # 断言检查第一个返回的结果是否与预期结果 a[0] 在给定的误差范围内完全对齐
    assert_allclose(R.apply(b)[0], a[0], atol=atol)  # Only pri aligns exactly
    
    # 断言检查计算得到的 rssd 是否与预期值 rssd_expected 在给定的误差范围内非常接近
    assert np.isclose(rssd, rssd_expected, atol=atol)
def test_align_vectors_near_inf():
    # align_vectors should return near the same result for high weights as for
    # infinite weights. rssd will be different with floating point error on the
    # exactly aligned vector being multiplied by a large non-infinite weight
    n = 100
    mats = []
    for i in range(6):
        mats.append(Rotation.random(n, random_state=10 + i).as_matrix())

    for i in range(n):
        # Get random pairs of 3-element vectors
        a = [1*mats[0][i][0], 2*mats[1][i][0]]
        b = [3*mats[2][i][0], 4*mats[3][i][0]]

        # Align vectors 'a' and 'b' using 'Rotation.align_vectors' method with high and infinite weights
        R, _ = Rotation.align_vectors(a, b, weights=[1e10, 1])
        R2, _ = Rotation.align_vectors(a, b, weights=[np.inf, 1])
        # Assert that the matrices representing the rotations 'R' and 'R2' are close within a tolerance
        assert_allclose(R.as_matrix(), R2.as_matrix(), atol=1e-4)

    for i in range(n):
        # Get random triplets of 3-element vectors
        a = [1*mats[0][i][0], 2*mats[1][i][0], 3*mats[2][i][0]]
        b = [4*mats[3][i][0], 5*mats[4][i][0], 6*mats[5][i][0]]

        # Align vectors 'a' and 'b' using 'Rotation.align_vectors' method with high and infinite weights
        R, _ = Rotation.align_vectors(a, b, weights=[1e10, 2, 1])
        R2, _ = Rotation.align_vectors(a, b, weights=[np.inf, 2, 1])
        # Assert that the matrices representing the rotations 'R' and 'R2' are close within a tolerance
        assert_allclose(R.as_matrix(), R2.as_matrix(), atol=1e-4)


def test_align_vectors_parallel():
    atol = 1e-12
    a = [[1, 0, 0], [0, 1, 0]]
    b = [[0, 1, 0], [0, 1, 0]]
    m_expected = np.array([[0, 1, 0],
                           [-1, 0, 0],
                           [0, 0, 1]])
    # Align vectors 'a' and 'b' using 'Rotation.align_vectors' method with one infinite weight
    R, _ = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    # Assert that the resulting rotation matrix 'R' matches the expected matrix 'm_expected'
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    # Align vectors 'a[0]' and 'b[0]' using 'Rotation.align_vectors' method without explicit weights
    R, _ = Rotation.align_vectors(a[0], b[0])
    # Assert that the resulting rotation matrix 'R' matches the expected matrix 'm_expected'
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    # Assert that applying rotation 'R' to vector 'b[0]' gives vector 'a[0]' within a tolerance
    assert_allclose(R.apply(b[0]), a[0], atol=atol)

    b = [[1, 0, 0], [1, 0, 0]]
    m_expected = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    # Align vectors 'a' and 'b' using 'Rotation.align_vectors' method with one infinite weight
    R, _ = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    # Assert that the resulting rotation matrix 'R' matches the expected matrix 'm_expected'
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    # Align vectors 'a[0]' and 'b[0]' using 'Rotation.align_vectors' method without explicit weights
    R, _ = Rotation.align_vectors(a[0], b[0])
    # Assert that the resulting rotation matrix 'R' matches the expected matrix 'm_expected'
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    # Assert that applying rotation 'R' to vector 'b[0]' gives vector 'a[0]' within a tolerance
    assert_allclose(R.apply(b[0]), a[0], atol=atol)


def test_align_vectors_antiparallel():
    # Test exact 180 deg rotation
    atol = 1e-12
    as_to_test = np.array([[[1, 0, 0], [0, 1, 0]],
                           [[0, 1, 0], [1, 0, 0]],
                           [[0, 0, 1], [0, 1, 0]]])
    bs_to_test = [[-a[0], a[1]] for a in as_to_test]
    for a, b in zip(as_to_test, bs_to_test):
        # Align vectors 'a' and 'b' using 'Rotation.align_vectors' method with one infinite weight
        R, _ = Rotation.align_vectors(a, b, weights=[np.inf, 1])
        # Assert that the rotation angle of 'R' is approximately pi (180 degrees)
        assert_allclose(R.magnitude(), np.pi, atol=atol)
        # Assert that applying rotation 'R' to vector 'b[0]' gives vector 'a[0]' within a tolerance
        assert_allclose(R.apply(b[0]), a[0], atol=atol)

    # Test exact rotations near 180 deg
    Rs = Rotation.random(100, random_state=0)
    dRs = Rotation.from_rotvec(Rs.as_rotvec()*1e-4)  # scale down to small angle
    a = [[ 1, 0, 0], [0, 1, 0]]
    b = [[-1, 0, 0], [0, 1, 0]]
    as_to_test = []
    for dR in dRs:
        as_to_test.append([dR.apply(a[0]), a[1]])
    # 对于每个向量 a 在 as_to_test 列表中进行以下操作：
    R, _ = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    # 使用 Rotation 类的 align_vectors 方法，对向量 a 和 b 进行对齐，使用权重 [无穷大, 1]。
    # 返回结果中 R 是对齐后的旋转矩阵，_ 是忽略的额外信息。
    
    R2, _ = Rotation.align_vectors(a, b, weights=[1e10, 1])
    # 使用 Rotation 类的 align_vectors 方法，对向量 a 和 b 进行对齐，使用权重 [1e10, 1]。
    # 返回结果中 R2 是对齐后的旋转矩阵，_ 是忽略的额外信息。
    
    assert_allclose(R.as_matrix(), R2.as_matrix(), atol=atol)
    # 断言函数 assert_allclose 检查 R 和 R2 的矩阵表示，在指定的绝对误差阈值 atol 内应该非常接近。
# 定义测试函数，用于测试只对主要向量进行向量对齐的功能
def test_align_vectors_primary_only():
    # 设置比较的数值容差
    atol = 1e-12
    # 生成随机的旋转矩阵作为第一个集合
    mats_a = Rotation.random(100, random_state=0).as_matrix()
    # 生成随机的旋转矩阵作为第二个集合
    mats_b = Rotation.random(100, random_state=1).as_matrix()
    # 遍历两个矩阵集合中的每一对旋转矩阵
    for mat_a, mat_b in zip(mats_a, mats_b):
        # 从每个旋转矩阵中获取第一个随机的三元单位向量
        a = mat_a[0]
        b = mat_b[0]

        # 使用只考虑主要向量的方法来比较向量的对齐
        R, rssd = Rotation.align_vectors(a, b)
        # 断言对变换后的向量应用 R 后结果与 a 在指定容差内相等
        assert_allclose(R.apply(b), a, atol=atol)
        # 断言 RSSD 值应在指定容差内接近于 0
        assert np.isclose(rssd, 0, atol=atol)


# 定义测试函数，用于测试 Slerp 插值器的功能
def test_slerp():
    # 使用固定的随机种子生成随机数
    rnd = np.random.RandomState(0)

    # 生成随机四元数并转换为旋转矩阵作为关键帧的旋转
    key_rots = Rotation.from_quat(rnd.uniform(size=(5, 4)))
    key_quats = key_rots.as_quat()

    # 设定关键帧的时间
    key_times = [0, 1, 2, 3, 4]
    # 使用 Slerp 插值器初始化
    interpolator = Slerp(key_times, key_rots)

    # 设定需要插值的时间点
    times = [0, 0.5, 0.25, 1, 1.5, 2, 2.75, 3, 3.25, 3.60, 4]
    # 对给定时间点进行插值得到旋转矩阵
    interp_rots = interpolator(times)
    interp_quats = interp_rots.as_quat()

    # 调整四元数的符号以匹配键帧的四元数的符号
    interp_quats[interp_quats[:, -1] < 0] *= -1
    key_quats[key_quats[:, -1] < 0] *= -1

    # 在关键帧上进行四元数的相等性断言，包括两个端点
    assert_allclose(interp_quats[0], key_quats[0])
    assert_allclose(interp_quats[3], key_quats[1])
    assert_allclose(interp_quats[5], key_quats[2])
    assert_allclose(interp_quats[7], key_quats[3])
    assert_allclose(interp_quats[10], key_quats[4])

    # 在关键帧之间的角速度恒定性断言，通过等时差的四元数对之间的余弦值
    cos_theta1 = np.sum(interp_quats[0] * interp_quats[2])
    cos_theta2 = np.sum(interp_quats[2] * interp_quats[1])
    assert_allclose(cos_theta1, cos_theta2)

    cos_theta4 = np.sum(interp_quats[3] * interp_quats[4])
    cos_theta5 = np.sum(interp_quats[4] * interp_quats[5])
    assert_allclose(cos_theta4, cos_theta5)

    # 对于两个关键帧之间的角速度，使用双角公式验证等时差的四元数对
    cos_theta3 = np.sum(interp_quats[1] * interp_quats[3])
    assert_allclose(cos_theta3, 2 * (cos_theta1**2) - 1)

    # 验证插值后的旋转矩阵与时间点的长度相等
    assert_equal(len(interp_rots), len(times))


# 定义测试函数，用于测试 Slerp 插值器在输入错误类型时的行为
def test_slerp_rot_is_rotation():
    # 使用 pytest 来断言输入类型错误时引发异常
    with pytest.raises(TypeError, match="must be a `Rotation` instance"):
        r = np.array([[1,2,3,4],
                      [0,0,0,1]])
        t = np.array([0, 1])
        Slerp(t, r)


# 定义测试函数，用于测试 Slerp 插值器在单个旋转矩阵输入时的行为
def test_slerp_single_rot():
    # 使用 pytest 来断言输入长度不足时引发异常
    msg = "must be a sequence of at least 2 rotations"
    with pytest.raises(ValueError, match=msg):
        r = Rotation.from_quat([1, 2, 3, 4])
        Slerp([1], r)


# 定义测试函数，用于测试 Slerp 插值器在旋转矩阵长度为 1 时的行为
def test_slerp_rot_len1():
    # 使用 pytest 来断言输入长度不足时引发异常
    msg = "must be a sequence of at least 2 rotations"
    with pytest.raises(ValueError, match=msg):
        r = Rotation.from_quat([[1, 2, 3, 4]])
        Slerp([1], r)


# 定义测试函数，用于测试 Slerp 插值器在时间维度不匹配时的行为
def test_slerp_time_dim_mismatch():
    # 使用 pytest 来测试代码中的异常情况，预期会抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError,
                       match="times to be specified in a 1 dimensional array"):
        # 创建一个随机数生成器 rnd，并设定种子为 0
        rnd = np.random.RandomState(0)
        # 使用随机数生成器创建一个四元数数组，形状为 (2, 4)，表示两个四元数
        r = Rotation.from_quat(rnd.uniform(size=(2, 4)))
        # 创建一个二维 NumPy 数组 t，表示时间点，其中有两个时间点分别为 1 和 2
        t = np.array([[1],
                      [2]])
        # 使用 Slerp 类来执行 Spherical Linear Interpolation，传入时间点 t 和旋转数组 r
        Slerp(t, r)
def test_slerp_num_rotations_mismatch():
    # 使用 pytest 检查是否会引发 ValueError，并验证错误消息中包含特定文本
    with pytest.raises(ValueError, match="number of rotations to be equal to "
                                         "number of timestamps"):
        # 使用随机数种子生成器创建随机状态对象
        rnd = np.random.RandomState(0)
        # 生成一个 5x4 大小的随机四元数数组，并转换为 Rotation 对象
        r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
        # 创建一个包含 7 个时间戳的数组
        t = np.arange(7)
        # 创建 Slerp 对象，预期会引发异常
        Slerp(t, r)


def test_slerp_equal_times():
    # 使用 pytest 检查是否会引发 ValueError，并验证错误消息中包含特定文本
    with pytest.raises(ValueError, match="strictly increasing order"):
        # 使用随机数种子生成器创建随机状态对象
        rnd = np.random.RandomState(0)
        # 生成一个 5x4 大小的随机四元数数组，并转换为 Rotation 对象
        r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
        # 创建一个包含时间戳的数组，其中有重复的时间戳
        t = [0, 1, 2, 2, 4]
        # 创建 Slerp 对象，预期会引发异常
        Slerp(t, r)


def test_slerp_decreasing_times():
    # 使用 pytest 检查是否会引发 ValueError，并验证错误消息中包含特定文本
    with pytest.raises(ValueError, match="strictly increasing order"):
        # 使用随机数种子生成器创建随机状态对象
        rnd = np.random.RandomState(0)
        # 生成一个 5x4 大小的随机四元数数组，并转换为 Rotation 对象
        r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
        # 创建一个包含时间戳的数组，其中时间戳无序
        t = [0, 1, 3, 2, 4]
        # 创建 Slerp 对象，预期会引发异常
        Slerp(t, r)


def test_slerp_call_time_dim_mismatch():
    # 使用随机数种子生成器创建随机状态对象
    rnd = np.random.RandomState(0)
    # 生成一个 5x4 大小的随机四元数数组，并转换为 Rotation 对象
    r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
    # 创建一个包含 5 个时间戳的数组
    t = np.arange(5)
    # 创建 Slerp 对象
    s = Slerp(t, r)

    # 使用 pytest 检查是否会引发 ValueError，并验证错误消息中包含特定文本
    with pytest.raises(ValueError,
                       match="`times` must be at most 1-dimensional."):
        # 创建一个 2x1 大小的二维数组作为插值时间，预期会引发异常
        interp_times = np.array([[3.5],
                                 [4.2]])
        # 调用 Slerp 对象进行插值，预期会引发异常
        s(interp_times)


def test_slerp_call_time_out_of_range():
    # 使用随机数种子生成器创建随机状态对象
    rnd = np.random.RandomState(0)
    # 生成一个 5x4 大小的随机四元数数组，并转换为 Rotation 对象
    r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
    # 创建一个包含时间戳的数组，每个时间戳增加 1
    t = np.arange(5) + 1
    # 创建 Slerp 对象
    s = Slerp(t, r)

    # 使用 pytest 检查是否会引发 ValueError，并验证错误消息中包含特定文本
    with pytest.raises(ValueError, match="times must be within the range"):
        # 调用 Slerp 对象时传递超出范围的时间戳，预期会引发异常
        s([0, 1, 2])
    with pytest.raises(ValueError, match="times must be within the range"):
        # 调用 Slerp 对象时传递超出范围的时间戳，预期会引发异常
        s([1, 2, 6])


def test_slerp_call_scalar_time():
    # 使用欧拉角创建一个 Rotation 对象，包含两个时间戳
    r = Rotation.from_euler('X', [0, 80], degrees=True)
    # 创建 Slerp 对象
    s = Slerp([0, 1], r)

    # 使用 Slerp 对象进行插值，得到插值后的 Rotation 对象
    r_interpolated = s(0.25)
    # 创建预期的插值后的 Rotation 对象
    r_interpolated_expected = Rotation.from_euler('X', 20, degrees=True)

    # 计算两个 Rotation 对象之间的差异
    delta = r_interpolated * r_interpolated_expected.inv()

    # 使用 assert_allclose 断言两个 Rotation 对象的差异非常小
    assert_allclose(delta.magnitude(), 0, atol=1e-16)


def test_multiplication_stability():
    # 生成包含 50 个随机四元数的 Rotation 对象数组
    qs = Rotation.random(50, random_state=0)
    # 生成包含 1000 个随机四元数的 Rotation 对象数组
    rs = Rotation.random(1000, random_state=1)
    # 对每个 qs 中的 Rotation 对象进行循环处理
    for q in qs:
        # 对 rs 中的 Rotation 对象进行连续乘法操作
        rs *= q * rs
        # 使用 assert_allclose 断言乘法后的四元数数组的模长为 1
        assert_allclose(np.linalg.norm(rs.as_quat(), axis=1), 1)


def test_pow():
    # 设置误差容限
    atol = 1e-14
    # 生成包含 10 个随机四元数的 Rotation 对象数组
    p = Rotation.random(10, random_state=0)
    # 计算 p 的逆
    p_inv = p.inv()
    
    # 循环测试 Rotation 对象的乘幂运算，包括负数、零和正数
    for n in [-5, -2, -1, 0, 1, 2, 5]:
        # 测试精度
        q = p ** n
        r = Rotation.identity(10)
        for _ in range(abs(n)):
            if n > 0:
                r = r * p
            else:
                r = r * p_inv
        ang = (q * r.inv()).magnitude()
        # 使用 np.all 和 assert 断言角度变化的精度
        assert np.all(ang < atol)

        # 测试形状保持
        r = Rotation.from_quat([0, 0, 0, 1])
        # 使用 assert 断言 Rotation 对象乘幂后的形状为 (4,)
        assert (r**n).as_quat().shape == (4,)
        r = Rotation.from_quat([[0, 0, 0, 1]])
        # 使用 assert 断言 Rotation 对象乘幂后的形状为 (1, 4)
        assert (r**n).as_quat().shape == (1, 4)

    # 大角度分数幂
    # 对于给定的每个旋转角度n，执行以下操作
    for n in [-1.5, -0.5, -0.0, 0.0, 0.5, 1.5]:
        # 计算旋转p的n次幂
        q = p ** n
        # 使用旋转向量n * p的旋转来创建旋转对象r
        r = Rotation.from_rotvec(n * p.as_rotvec())
        # 断言两个旋转对象的四元数表示在给定的数值误差范围atol内相等
        assert_allclose(q.as_quat(), r.as_quat(), atol=atol)

    # 对于一个小角度的旋转p
    p = Rotation.from_rotvec([1e-12, 0, 0])
    # 设定一个整数n
    n = 3
    # 计算旋转p的n次幂
    q = p ** n
    # 使用旋转向量n * p的旋转来创建旋转对象r
    r = Rotation.from_rotvec(n * p.as_rotvec())
    # 断言两个旋转对象的四元数表示在给定的数值误差范围atol内相等
    assert_allclose(q.as_quat(), r.as_quat(), atol=atol)
# 测试函数：测试幂运算抛出错误
def test_pow_errors():
    # 创建一个随机旋转对象 p
    p = Rotation.random(random_state=0)
    # 使用 pytest 检查 pow 函数对 p 进行幂运算时是否抛出 NotImplementedError，并检查错误信息
    with pytest.raises(NotImplementedError, match='modulus not supported'):
        pow(p, 1, 1)


# 测试函数：测试将旋转对象转换为 NumPy 数组
def test_rotation_within_numpy_array():
    # 创建单个随机旋转对象 single 和包含 2 个随机旋转对象的 multiple
    single = Rotation.random(random_state=0)
    multiple = Rotation.random(2, random_state=1)

    # 将单个旋转对象转换为 NumPy 数组，检查其形状是否为 ()
    array = np.array(single)
    assert_equal(array.shape, ())

    # 将多个旋转对象转换为 NumPy 数组，检查其形状是否为 (2,)
    array = np.array(multiple)
    assert_equal(array.shape, (2,))
    # 检查数组中每个元素与对应旋转对象的矩阵表示是否接近
    assert_allclose(array[0].as_matrix(), multiple[0].as_matrix())
    assert_allclose(array[1].as_matrix(), multiple[1].as_matrix())

    # 将单个旋转对象作为数组的元素，检查其形状是否为 (1,)
    array = np.array([single])
    assert_equal(array.shape, (1,))
    assert_equal(array[0], single)

    # 将多个旋转对象作为数组的元素，检查其形状是否为 (1, 2)
    array = np.array([multiple])
    assert_equal(array.shape, (1, 2))
    # 检查数组中每个元素与对应旋转对象的矩阵表示是否接近
    assert_allclose(array[0, 0].as_matrix(), multiple[0].as_matrix())
    assert_allclose(array[0, 1].as_matrix(), multiple[1].as_matrix())

    # 将多个旋转对象作为对象类型的数组元素，检查其形状是否为 (2,)
    array = np.array([single, multiple], dtype=object)
    assert_equal(array.shape, (2,))
    assert_equal(array[0], single)
    assert_equal(array[1], multiple)

    # 创建包含多个相同旋转对象的数组，检查其形状是否为 (3, 2)
    array = np.array([multiple, multiple, multiple])
    assert_equal(array.shape, (3, 2))


# 测试函数：测试旋转对象的序列化和反序列化
def test_pickling():
    # 创建一个旋转对象 r，并将其序列化为 pkl 字符串
    r = Rotation.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    pkl = pickle.dumps(r)
    # 反序列化 pkl 字符串，得到 unpickled 对象，并比较其矩阵表示是否接近原始对象 r
    unpickled = pickle.loads(pkl)
    assert_allclose(r.as_matrix(), unpickled.as_matrix(), atol=1e-15)


# 测试函数：测试旋转对象的深拷贝
def test_deepcopy():
    # 创建一个旋转对象 r，并对其进行深拷贝，得到 r1
    r = Rotation.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    r1 = copy.deepcopy(r)
    # 检查 r 和 r1 的矩阵表示是否接近
    assert_allclose(r.as_matrix(), r1.as_matrix(), atol=1e-15)


# 测试函数：测试将旋转对象转换为欧拉角时的内存布局是否连续
def test_as_euler_contiguous():
    # 创建一个旋转对象 r
    r = Rotation.from_quat([0, 0, 0, 1])
    # 获取 r 转换为 'xyz' 和 'XYZ' 欧拉角时的结果 e1 和 e2
    e1 = r.as_euler('xyz')  # 外围欧拉角
    e2 = r.as_euler('XYZ')  # 内围欧拉角
    # 断言 e1 和 e2 的内存布局是否连续
    assert e1.flags['C_CONTIGUOUS'] is True
    assert e2.flags['C_CONTIGUOUS'] is True
    # 断言 e1 和 e2 的步长是否都大于等于 0
    assert all(i >= 0 for i in e1.strides)
    assert all(i >= 0 for i in e2.strides)


# 测试函数：测试多个旋转对象的拼接
def test_concatenate():
    # 创建包含 10 个随机旋转对象的 rotation
    rotation = Rotation.random(10, random_state=0)
    sizes = [1, 2, 3, 1, 3]
    starts = [0] + list(np.cumsum(sizes))
    # 根据 sizes 和 starts 拆分 rotation，得到 split 列表
    split = [rotation[i:i + n] for i, n in zip(starts, sizes)]
    # 将 split 中的旋转对象拼接成一个 result 旋转对象
    result = Rotation.concatenate(split)
    # 断言 rotation 和 result 的四元数表示是否接近
    assert_equal(rotation.as_quat(), result.as_quat())


# 测试函数：测试拼接包含不支持类型的旋转对象时是否抛出异常
def test_concatenate_wrong_type():
    # 使用 pytest 检查 Rotation.concatenate 函数在包含不支持类型的对象时是否抛出 TypeError 异常
    with pytest.raises(TypeError, match='Rotation objects only'):
        Rotation.concatenate([Rotation.identity(), 1, None])


# 回归测试：测试旋转对象的长度和布尔值
def test_len_and_bool():
    # 创建不同类型的旋转对象
    rotation_multi_empty = Rotation(np.empty((0, 4)))
    rotation_multi_one = Rotation([[0, 0, 0, 1]])
    rotation_multi = Rotation([[0, 0, 0, 1], [0, 0, 0, 1]])
    rotation_single = Rotation([0, 0, 0, 1])

    # 断言不同类型旋转对象的长度是否符合预期
    assert len(rotation_multi_empty) == 0
    assert len(rotation_multi_one) == 1
    assert len(rotation_multi) == 2
    # 使用 pytest 检查单个旋转对象是否抛出 TypeError 异常
    with pytest.raises(TypeError, match="Single rotation has no len()."):
        len(rotation_single)

    # 断言旋转对象在布尔上下文中是否始终为真
    assert rotation_multi_empty
    assert rotation_multi_one
    assert rotation_multi
    # 断言确保 rotation_single 变量的值为真
    assert rotation_single
# 定义用于测试 Davenport 方法的单旋转案例
def test_from_davenport_single_rotation():
    # 定义旋转轴为 z 轴
    axis = [0, 0, 1]
    # 使用 Davenport 方法创建绕 z 轴的旋转，角度为 90 度，返回四元数表示的旋转
    quat = Rotation.from_davenport(axis, 'extrinsic', 90, 
                                   degrees=True).as_quat()
    # 预期的四元数表示，对应绕 z 轴旋转 90 度
    expected_quat = np.array([0, 0, 1, 1]) / np.sqrt(2)
    # 使用 assert_allclose 检查计算得到的四元数与预期值的接近程度
    assert_allclose(quat, expected_quat)


# 定义用于测试 Davenport 方法的单轴或双轴旋转案例
def test_from_davenport_one_or_two_axes():
    ez = [0, 0, 1]
    ey = [0, 1, 0]

    # 单旋转，单轴，轴的形状为 (3, )
    rot = Rotation.from_rotvec(np.array(ez) * np.pi/4)
    rot_dav = Rotation.from_davenport(ez, 'e', np.pi/4)
    # 使用 assert_allclose 检查两种旋转方法计算得到的四元数的接近程度
    assert_allclose(rot.as_quat(canonical=True),
                    rot_dav.as_quat(canonical=True))

    # 单旋转，单轴，轴的形状为 (1, 3)
    rot = Rotation.from_rotvec([np.array(ez) * np.pi/4])
    rot_dav = Rotation.from_davenport([ez], 'e', [np.pi/4])
    # 使用 assert_allclose 检查两种旋转方法计算得到的四元数的接近程度
    assert_allclose(rot.as_quat(canonical=True),
                    rot_dav.as_quat(canonical=True))
    
    # 单旋转，双轴，轴的形状为 (2, 3)
    rot = Rotation.from_rotvec([np.array(ez) * np.pi/4, 
                                np.array(ey) * np.pi/6])
    rot = rot[0] * rot[1]
    rot_dav = Rotation.from_davenport([ey, ez], 'e', [np.pi/6, np.pi/4])
    # 使用 assert_allclose 检查两种旋转方法计算得到的四元数的接近程度
    assert_allclose(rot.as_quat(canonical=True),
                    rot_dav.as_quat(canonical=True))
    
    # 双旋转，单轴，轴的形状为 (3, )
    rot = Rotation.from_rotvec([np.array(ez) * np.pi/6, 
                                np.array(ez) * np.pi/4])
    rot_dav = Rotation.from_davenport([ez], 'e', [np.pi/6, np.pi/4])
    # 使用 assert_allclose 检查两种旋转方法计算得到的四元数的接近程度
    assert_allclose(rot.as_quat(canonical=True),
                    rot_dav.as_quat(canonical=True))


# 定义用于测试 Davenport 方法的无效输入案例
def test_from_davenport_invalid_input():
    ez = [0, 0, 1]
    ey = [0, 1, 0]
    ezy = [0, 1, 1]
    # 使用 pytest 检查当轴不正交时抛出 ValueError 异常
    with pytest.raises(ValueError, match="must be orthogonal"):
        Rotation.from_davenport([ez, ezy], 'e', [0, 0])
    # 使用 pytest 检查当轴不正交时抛出 ValueError 异常
    with pytest.raises(ValueError, match="must be orthogonal"):
        Rotation.from_davenport([ez, ey, ezy], 'e', [0, 0, 0])
    # 使用 pytest 检查当指定的旋转顺序无效时抛出 ValueError 异常
    with pytest.raises(ValueError, match="order should be"):
        Rotation.from_davenport([ez], 'xyz', [0])
    # 使用 pytest 检查当角度数组不符合预期时抛出 ValueError 异常
    with pytest.raises(ValueError, match="Expected `angles`"):
        Rotation.from_davenport([ez, ey, ez], 'e', [0, 1, 2, 3])


# 定义用于测试 Davenport 方法的旋转输出为 Davenport 格式的案例
def test_as_davenport():
    rnd = np.random.RandomState(0)
    n = 100
    angles = np.empty((n, 3))
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles_middle = rnd.uniform(low=0, high=np.pi, size=(n,))
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    lambdas = rnd.uniform(low=0, high=np.pi, size=(20,))

    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    # 遍历 lambdas 列表中的每个 lambda 值
    for lamb in lambdas:
        # 构建旋转轴 ax_lamb，包括 e1, e2 和 lamb*e2 经旋转后的 e1
        ax_lamb = [e1, e2, Rotation.from_rotvec(lamb*e2).apply(e1)]
        
        # 更新 angles 列表的第二列值为 angles_middle 减去 lamb
        angles[:, 1] = angles_middle - lamb
        
        # 遍历旋转顺序 'extrinsic' 和 'intrinsic'
        for order in ['extrinsic', 'intrinsic']:
            # 如果顺序为 'intrinsic'，则使用 ax_lamb；否则使用 ax_lamb 的反向顺序
            ax = ax_lamb if order == 'intrinsic' else ax_lamb[::-1]
            
            # 使用 Davenport 方法进行旋转
            rot = Rotation.from_davenport(ax, order, angles)
            
            # 将旋转后的角度转换为 Davenport 格式
            angles_dav = rot.as_davenport(ax, order)
            
            # 断言旋转后的角度与原始角度非常接近
            assert_allclose(angles_dav, angles)
def test_as_davenport_degenerate():
    # 由于无法检查角度的相等性，我们检查旋转矩阵的相等性
    rnd = np.random.RandomState(0)
    n = 5
    angles = np.empty((n, 3))

    # 生成对称序列的角度
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles_middle = [rnd.choice([0, np.pi]) for i in range(n)]
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    lambdas = rnd.uniform(low=0, high=np.pi, size=(5,))

    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])

    # 遍历每个 lambda 值
    for lamb in lambdas:
        # 计算旋转轴 ax_lamb
        ax_lamb = [e1, e2, Rotation.from_rotvec(lamb*e2).apply(e1)]
        angles[:, 1] = angles_middle - lamb
        # 遍历旋转顺序 'extrinsic' 和 'intrinsic'
        for order in ['extrinsic', 'intrinsic']:
            # 根据旋转顺序选择轴向 ax
            ax = ax_lamb if order == 'intrinsic' else ax_lamb[::-1]
            # 创建 Davenport 参数化的旋转对象 rot
            rot = Rotation.from_davenport(ax, order, angles)
            # 断言预期的矩阵与估算的矩阵几乎相等
            with pytest.warns(UserWarning, match="Gimbal lock"):
                angles_dav = rot.as_davenport(ax, order)
            mat_expected = rot.as_matrix()
            mat_estimated = Rotation.from_davenport(ax, order, angles_dav).as_matrix()
            assert_array_almost_equal(mat_expected, mat_estimated)


def test_compare_from_davenport_from_euler():
    rnd = np.random.RandomState(0)
    n = 100
    angles = np.empty((n, 3))

    # 生成对称序列的角度
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles[:, 1] = rnd.uniform(low=0, high=np.pi, size=(n,))
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    
    # 遍历旋转顺序 'extrinsic' 和 'intrinsic'
    for order in ['extrinsic', 'intrinsic']:
        # 遍历 Euler 序列的全排列
        for seq_tuple in permutations('xyz'):
            seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
            ax = [basis_vec(i) for i in seq]
            if order == 'intrinsic':
                seq = seq.upper()
            # 使用 Euler 角创建旋转对象 eul 和 Davenport 参数化的旋转对象 dav
            eul = Rotation.from_euler(seq, angles)
            dav = Rotation.from_davenport(ax, order, angles)
            # 断言两种参数化的四元数表示几乎相等
            assert_allclose(eul.as_quat(canonical=True), dav.as_quat(canonical=True), 
                            rtol=1e-12)

    # 生成非对称序列的角度
    angles[:, 1] -= np.pi / 2
    # 遍历旋转顺序 'extrinsic' 和 'intrinsic'
    for order in ['extrinsic', 'intrinsic']:
        # 遍历 Euler 序列的全排列
        for seq_tuple in permutations('xyz'):
            seq = ''.join(seq_tuple)
            ax = [basis_vec(i) for i in seq]
            if order == 'intrinsic':
                seq = seq.upper()
            # 使用 Euler 角创建旋转对象 eul 和 Davenport 参数化的旋转对象 dav
            eul = Rotation.from_euler(seq, angles)
            dav = Rotation.from_davenport(ax, order, angles)
            # 断言两种参数化的四元数表示几乎相等
            assert_allclose(eul.as_quat(), dav.as_quat(), rtol=1e-12)


def test_compare_as_davenport_as_euler():
    rnd = np.random.RandomState(0)
    n = 100
    angles = np.empty((n, 3))

    # 生成对称序列的角度
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles[:, 1] = rnd.uniform(low=0, high=np.pi, size=(n,))
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))

    # 遍历旋转顺序 'extrinsic' 和 'intrinsic'
    for order in ['extrinsic', 'intrinsic']:
        # 遍历 Euler 序列的全排列
        for seq_tuple in permutations('xyz'):
            seq = ''.join(seq_tuple)
            ax = [basis_vec(i) for i in seq]
            if order == 'intrinsic':
                seq = seq.upper()
            # 使用 Euler 角创建旋转对象 eul 和 Davenport 参数化的旋转对象 dav
            eul = Rotation.from_euler(seq, angles)
            dav = Rotation.from_davenport(ax, order, angles)
            # 这里可能还有更多的断言，以比较两种参数化方式的角度表示
    `
    # 对每种旋转顺序进行循环：外禀顺序和内禀顺序
    for order in ['extrinsic', 'intrinsic']:
        # 对 'xyz' 的全排列进行循环
        for seq_tuple in permutations('xyz'):
            # 将元组转换为字符串，并且重复第一个元素以构成特定的序列
            seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
            # 根据序列创建基向量列表
            ax = [basis_vec(i) for i in seq]
            # 如果顺序是内禀的，则将序列转换为大写
            if order == 'intrinsic':
                seq = seq.upper()
            # 根据欧拉角序列创建旋转对象
            rot = Rotation.from_euler(seq, angles)
            # 将旋转对象转换为欧拉角
            eul = rot.as_euler(seq)
            # 将旋转对象转换为 Davenport 参数化形式
            dav = rot.as_davenport(ax, order)
            # 使用数值容差进行近似断言
            assert_allclose(eul, dav, rtol=1e-12)
    
    # 对非对称序列进行处理
    angles[:, 1] -= np.pi / 2
    # 对每种旋转顺序进行循环：外禀顺序和内禀顺序
    for order in ['extrinsic', 'intrinsic']:
        # 对 'xyz' 的全排列进行循环
        for seq_tuple in permutations('xyz'):
            # 将元组转换为字符串
            seq = ''.join(seq_tuple)
            # 根据序列创建基向量列表
            ax = [basis_vec(i) for i in seq]
            # 如果顺序是内禀的，则将序列转换为大写
            if order == 'intrinsic':
                seq = seq.upper()
            # 根据欧拉角序列创建旋转对象
            rot = Rotation.from_euler(seq, angles)
            # 将旋转对象转换为欧拉角
            eul = rot.as_euler(seq)
            # 将旋转对象转换为 Davenport 参数化形式
            dav = rot.as_davenport(ax, order)
            # 使用数值容差进行近似断言
            assert_allclose(eul, dav, rtol=1e-12)
```