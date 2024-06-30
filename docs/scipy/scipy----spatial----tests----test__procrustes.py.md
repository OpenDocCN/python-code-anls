# `D:\src\scipysrc\scipy\scipy\spatial\tests\test__procrustes.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal  # 导入 NumPy 测试模块中的断言函数
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 函数并重命名为 assert_raises
from scipy.spatial import procrustes  # 导入 scipy 库中的 procrustes 函数


class TestProcrustes:
    def setup_method(self):
        """设置测试数据"""
        # 原始 L 形状的数据
        self.data1 = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')

        # 放大、平移、镜像后的 L 形状的数据
        self.data2 = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')

        # 上移、右移并使第四个点额外右移 0.5 后的 L 形状的数据
        # 与 data1 存在点间距的差异：3*(2) + (1 + 1.5^2)
        self.data3 = np.array([[2, 4], [2, 3], [2, 2], [3, 2.5]], 'd')

        # data4 和 data5 是经过标准化处理的数据（trace(A*A') = 1）
        # 如果作为第一个矩阵参数，procrustes 应返回一个完全相同的副本。
        shiftangle = np.pi / 8
        self.data4 = np.array([[1, 0], [0, 1], [-1, 0],
                              [0, -1]], 'd') / np.sqrt(4)
        self.data5 = np.array([[np.cos(shiftangle), np.sin(shiftangle)],
                              [np.cos(np.pi / 2 - shiftangle),
                               np.sin(np.pi / 2 - shiftangle)],
                              [-np.cos(shiftangle),
                               -np.sin(shiftangle)],
                              [-np.cos(np.pi / 2 - shiftangle),
                               -np.sin(np.pi / 2 - shiftangle)]],
                              'd') / np.sqrt(4)

    def test_procrustes(self):
        # 测试 procrustes 函数匹配两个矩阵的能力。
        #
        # 第二个矩阵是第一个矩阵的旋转、平移、缩放和镜像版本，仅限于二维空间。
        #
        # 能够将一个 'L' 形状进行平移、镜像和缩放吗？
        a, b, disparity = procrustes(self.data1, self.data2)
        assert_allclose(b, a)  # 断言：b 与 a 是几乎完全相等的
        assert_almost_equal(disparity, 0.)  # 断言：disparity 接近于 0

        # 如果第一个矩阵是标准化的，那么将不会改变第一个矩阵？
        m4, m5, disp45 = procrustes(self.data4, self.data5)
        assert_equal(m4, self.data4)  # 断言：m4 等于 self.data4

        # data3 最差情况下是一个 'L' 形状，其中一个点偏移了 0.5
        m1, m3, disp13 = procrustes(self.data1, self.data3)
        #assert_(disp13 < 0.5 ** 2)  # 断言：disp13 小于 0.5 的平方
    def test_procrustes2(self):
        # 使用 procrustes 函数计算 m1, m3 之间的最优映射以及其误差 disp13
        m1, m3, disp13 = procrustes(self.data1, self.data3)
        # 使用 procrustes 函数计算 m3, m1 之间的最优映射以及其误差 disp31
        m3_2, m1_2, disp31 = procrustes(self.data3, self.data1)
        # 断言 disp13 和 disp31 的值近似相等
        assert_almost_equal(disp13, disp31)

        # 使用 3D 数据进行测试，每个数据集包含 8 个点
        rand1 = np.array([[2.61955202, 0.30522265, 0.55515826],
                         [0.41124708, -0.03966978, -0.31854548],
                         [0.91910318, 1.39451809, -0.15295084],
                         [2.00452023, 0.50150048, 0.29485268],
                         [0.09453595, 0.67528885, 0.03283872],
                         [0.07015232, 2.18892599, -1.67266852],
                         [0.65029688, 1.60551637, 0.80013549],
                         [-0.6607528, 0.53644208, 0.17033891]])

        rand3 = np.array([[0.0809969, 0.09731461, -0.173442],
                         [-1.84888465, -0.92589646, -1.29335743],
                         [0.67031855, -1.35957463, 0.41938621],
                         [0.73967209, -0.20230757, 0.52418027],
                         [0.17752796, 0.09065607, 0.29827466],
                         [0.47999368, -0.88455717, -0.57547934],
                         [-0.11486344, -0.12608506, -0.3395779],
                         [-0.86106154, -0.28687488, 0.9644429]])
        # 使用 procrustes 函数计算 rand1 和 rand3 之间的最优映射以及其误差 disp13
        res1, res3, disp13 = procrustes(rand1, rand3)
        # 使用 procrustes 函数计算 rand3 和 rand1 之间的最优映射以及其误差 disp31
        res3_2, res1_2, disp31 = procrustes(rand3, rand1)
        # 断言 disp13 和 disp31 的值近似相等
        assert_almost_equal(disp13, disp31)

    def test_procrustes_shape_mismatch(self):
        # 断言当输入的数据维度不匹配时，procrustes 函数会引发 ValueError 异常
        assert_raises(ValueError, procrustes,
                      np.array([[1, 2], [3, 4]]),
                      np.array([[5, 6, 7], [8, 9, 10]]))

    def test_procrustes_empty_rows_or_cols(self):
        # 断言当输入的数据为空时，procrustes 函数会引发 ValueError 异常
        empty = np.array([[]])
        assert_raises(ValueError, procrustes, empty, empty)

    def test_procrustes_no_variation(self):
        # 断言当输入的数据没有变化时，procrustes 函数会引发 ValueError 异常
        assert_raises(ValueError, procrustes,
                      np.array([[42, 42], [42, 42]]),
                      np.array([[45, 45], [45, 45]]))

    def test_procrustes_bad_number_of_dimensions(self):
        # 断言当输入的数据维度不符合要求时，procrustes 函数会引发 ValueError 异常
        # 一个数据集的维度较少
        assert_raises(ValueError, procrustes,
                      np.array([1, 1, 2, 3, 5, 8]),
                      np.array([[1, 2], [3, 4]]))

        # 两个数据集的维度均较少
        assert_raises(ValueError, procrustes,
                      np.array([1, 1, 2, 3, 5, 8]),
                      np.array([1, 1, 2, 3, 5, 8]))

        # 数据集的维度为零
        assert_raises(ValueError, procrustes, np.array(7), np.array(11))

        # 数据集的维度过多
        assert_raises(ValueError, procrustes,
                      np.array([[[11], [7]]]),
                      np.array([[[5, 13]]]))
```