# `D:\src\scipysrc\scipy\scipy\signal\tests\test_ltisys.py`

```
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
                           assert_, suppress_warnings)  # 导入 NumPy 测试相关函数
from pytest import raises as assert_raises  # 导入 pytest 的 raises 函数别名为 assert_raises

from scipy.signal import (ss2tf, tf2ss, lti,
                          dlti, bode, freqresp, lsim, impulse, step,  # 导入信号处理相关函数
                          abcd_normalize, place_poles,
                          TransferFunction, StateSpace, ZerosPolesGain)
from scipy.signal._filter_design import BadCoefficients  # 导入信号处理模块中的异常类
import scipy.linalg as linalg  # 导入 SciPy 线性代数模块


def _assert_poles_close(P1,P2, rtol=1e-8, atol=1e-8):
    """
    检查 P1 中的每个极点是否与 P2 中的某个极点非常接近，使用 1e-8 的相对容差或 1e-8 的绝对容差（对于零极点很有用）。
    这些容差非常严格，但被测试的系统已知可以接受这些极点，因此我们应该接近所需的结果。
    """
    P2 = P2.copy()  # 复制 P2，以便在迭代中修改而不影响原始对象
    for p1 in P1:
        found = False
        for p2_idx in range(P2.shape[0]):
            if np.allclose([np.real(p1), np.imag(p1)],
                           [np.real(P2[p2_idx]), np.imag(P2[p2_idx])],
                           rtol, atol):  # 使用 np.allclose 判断两个极点是否非常接近
                found = True
                np.delete(P2, p2_idx)  # 如果找到接近的极点，从 P2 中删除该极点
                break
        if not found:
            raise ValueError("Can't find pole " + str(p1) + " in " + str(P2))  # 如果找不到匹配的极点，抛出 ValueError


class TestPlacePoles:

    def _check(self, A, B, P, **kwargs):
        """
        对 place_poles 计算的极点执行最常见的测试，并返回 Bunch 对象以进行进一步的特定测试
        """
        fsf = place_poles(A, B, P, **kwargs)  # 调用 place_poles 函数计算控制增益矩阵
        expected, _ = np.linalg.eig(A - np.dot(B, fsf.gain_matrix))  # 计算期望的闭环极点
        _assert_poles_close(expected, fsf.requested_poles)  # 检查期望的极点与请求的极点是否非常接近
        _assert_poles_close(expected, fsf.computed_poles)  # 检查期望的极点与计算得到的极点是否非常接近
        _assert_poles_close(P, fsf.requested_poles)  # 检查请求的极点与期望的极点是否非常接近
        return fsf  # 返回计算得到的 fsf 对象，供进一步测试使用

    def test_real(self):
        # 测试使用 KNV 和 YT0 算法进行实部极点放置，并参照参考文献中第 4 节示例 1 进行验证
        A = np.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
                      0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
                      1.343, -2.104]).reshape(4, 4)  # 定义状态矩阵 A
        B = np.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0]).reshape(4, 2)  # 定义输入矩阵 B
        P = np.array([-0.2, -0.5, -5.0566, -8.6659])  # 定义期望的极点数组 P

        # 检查 KNV 和 YT 算法是否计算正确的 K 矩阵
        self._check(A, B, P, method='KNV0')
        self._check(A, B, P, method='YT')

        # 尝试达到 _YT_real 中的特定情况，其中两个奇异值几乎相等。这是为了提高代码覆盖率，但我
        # 无法确定是否真的达到了这段代码

        # 在某些架构上，这可能导致 RuntimeWarning: invalid value in divide (见 gh-7590)，因此暂时将其禁止
        with np.errstate(invalid='ignore'):
            self._check(A, B, (2, 2, 3, 3))  # 使用禁止无效状态测试函数进行检查
    def test_tricky_B(self):
        # 检查我们如何处理只有一列B矩阵和n列B矩阵（其中n是满足shape(A)=(n, n)的情况）
        A = np.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
                      0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
                      1.343, -2.104]).reshape(4, 4)
        B = np.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0, 1, 2, 3, 4,
                      5, 6, 7, 8]).reshape(4, 4)

        # 对于此情况，不会调用KNV或YT，这是一个特殊情况，只有一个唯一解
        P = np.array([-0.2, -0.5, -5.0566, -8.6659])
        # 调用_check方法，返回一个特定结果fsf
        fsf = self._check(A, B, P)
        # 当单位矩阵可以作为传递矩阵时，rtol和nb_iter应设置为np.nan
        assert_equal(fsf.rtol, np.nan)
        assert_equal(fsf.nb_iter, np.nan)

        # 当存在复数极点时，它们会触发一个特殊情况的处理
        P = np.array((-2+1j, -2-1j, -3, -2))
        fsf = self._check(A, B, P)
        assert_equal(fsf.rtol, np.nan)
        assert_equal(fsf.nb_iter, np.nan)

        # 现在测试一个只有一列B矩阵的情况（无法进行优化）
        B = B[:,0].reshape(4,1)
        P = np.array((-2+1j, -2-1j, -3, -2))
        fsf = self._check(A, B, P)

        # 我们无法优化任何内容，检查它们是否按预期设置为0
        assert_equal(fsf.rtol, 0)
        assert_equal(fsf.nb_iter, 0)
    # 定义一个测试方法，用于测试错误情况
    def test_errors(self):
        # 创建一个4x4的numpy数组A，表示系统的状态空间矩阵
        A = np.array([0,7,0,0,0,0,0,7/3.,0,0,0,0,0,0,0,0]).reshape(4,4)
        # 创建一个4x2的numpy数组B，表示输入矩阵
        B = np.array([0,0,0,0,1,0,0,1]).reshape(4,2)

        # 应当因为method关键字无效而失败
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4),
                      method="foo")

        # 应当因为极点不是1维数组而失败
        assert_raises(ValueError, place_poles, A, B,
                      np.array((-2.1,-2.2,-2.3,-2.4)).reshape(4,1))

        # 应当因为A不是2D数组而失败
        assert_raises(ValueError, place_poles, A[:,:,np.newaxis], B,
                      (-2.1,-2.2,-2.3,-2.4))

        # 应当因为B不是2D数组而失败
        assert_raises(ValueError, place_poles, A, B[:,:,np.newaxis],
                      (-2.1,-2.2,-2.3,-2.4))

        # 应当因为极点数量过多而失败
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4,-3))

        # 应当因为极点数量不足而失败
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3))

        # 应当因为rtol大于1而失败
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4),
                      rtol=42)

        # 应当因为maxiter小于1而失败
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4),
                      maxiter=-42)

        # 应当因为B的维度不是二维而失败
        assert_raises(ValueError, place_poles, A, B, (-2,-2,-2,-2))

        # 不可控系统，预期会失败
        assert_raises(ValueError, place_poles, np.ones((4,4)),
                      np.ones((4,2)), (1,2,3,4))

        # 极点可以被放置，但预期会发出警告，因为未达到收敛
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 在A和B上放置极点(-1,-2,-3,-4)，设置收敛容差和最大迭代次数
            fsf = place_poles(A, B, (-1,-2,-3,-4), rtol=1e-16, maxiter=42)
            # 预期有一条警告信息被记录
            assert_(len(w) == 1)
            # 预期警告类型是UserWarning
            assert_(issubclass(w[-1].category, UserWarning))
            # 预期警告信息包含未达到最大迭代次数的提示
            assert_("Convergence was not reached after maxiter iterations"
                    in str(w[-1].message))
            # 确保返回的对象包含迭代次数属性nb_iter为42
            assert_equal(fsf.nb_iter, 42)

        # 应当因为复数极点没有其共轭而失败
        assert_raises(ValueError, place_poles, A, B, (-2+1j,-2-1j,-2+3j,-2))

        # 应当因为A不是方阵而失败
        assert_raises(ValueError, place_poles, A[:,:3], B, (-2,-3,-4,-5))

        # 应当因为B的行数与A不匹配而失败
        assert_raises(ValueError, place_poles, A, B[:3,:], (-2,-3,-4,-5))

        # 应当因为KNV0方法不支持复数极点而失败
        assert_raises(ValueError, place_poles, A, B,
                      (-2+1j,-2-1j,-2+3j,-2-3j), method="KNV0")
class TestSS2TF:

    def check_matrix_shapes(self, p, q, r):
        # 调用 ss2tf 函数进行状态空间到传递函数的转换，传入零矩阵和零作为其他参数
        ss2tf(np.zeros((p, p)),  # 创建一个 p x p 的零矩阵作为传递函数的分子
              np.zeros((p, q)),  # 创建一个 p x q 的零矩阵作为传递函数的分母
              np.zeros((r, p)),  # 创建一个 r x p 的零矩阵作为状态空间转换矩阵 A
              np.zeros((r, q)),  # 创建一个 r x q 的零矩阵作为状态空间转换矩阵 B
              0)                # 传递函数的零次项

    def test_shapes(self):
        # 对每组状态空间的维度进行测试
        for p, q, r in [(3, 3, 3), (1, 3, 3), (1, 1, 1)]:
            self.check_matrix_shapes(p, q, r)

    def test_basic(self):
        # 测试通过 tf2ss 和 ss2tf 的往返转换
        b = np.array([1.0, 3.0, 5.0])
        a = np.array([1.0, 2.0, 3.0])

        A, B, C, D = tf2ss(b, a)  # 将传递函数系数 b 和 a 转换为状态空间表示 A, B, C, D
        assert_allclose(A, [[-2, -3], [1, 0]], rtol=1e-13)  # 断言 A 的值符合预期
        assert_allclose(B, [[1], [0]], rtol=1e-13)           # 断言 B 的值符合预期
        assert_allclose(C, [[1, 2]], rtol=1e-13)             # 断言 C 的值符合预期
        assert_allclose(D, [[1]], rtol=1e-14)                # 断言 D 的值符合预期

        bb, aa = ss2tf(A, B, C, D)  # 将状态空间表示 A, B, C, D 转换回传递函数的系数 bb, aa
        assert_allclose(bb[0], b, rtol=1e-13)  # 断言转换后的传递函数分子 bb 与原始 b 相符
        assert_allclose(aa, a, rtol=1e-13)     # 断言转换后的传递函数分母 aa 与原始 a 相符

    def test_zero_order_round_trip(self):
        # 测试零阶系统的往返转换
        # 见问题 gh-5760
        tf = (2, 1)
        A, B, C, D = tf2ss(*tf)  # 将传递函数 (2, 1) 转换为状态空间表示 A, B, C, D
        assert_allclose(A, [[0]], rtol=1e-13)    # 断言 A 的值符合预期
        assert_allclose(B, [[0]], rtol=1e-13)    # 断言 B 的值符合预期
        assert_allclose(C, [[0]], rtol=1e-13)    # 断言 C 的值符合预期
        assert_allclose(D, [[2]], rtol=1e-13)    # 断言 D 的值符合预期

        num, den = ss2tf(A, B, C, D)  # 将状态空间表示 A, B, C, D 转换回传递函数的系数 num, den
        assert_allclose(num, [[2, 0]], rtol=1e-13)  # 断言转换后的传递函数分子 num 符合预期
        assert_allclose(den, [1, 0], rtol=1e-13)    # 断言转换后的传递函数分母 den 符合预期

        tf = ([[5], [2]], 1)
        A, B, C, D = tf2ss(*tf)  # 将传递函数 ([[5], [2]], 1) 转换为状态空间表示 A, B, C, D
        assert_allclose(A, [[0]], rtol=1e-13)    # 断言 A 的值符合预期
        assert_allclose(B, [[0]], rtol=1e-13)    # 断言 B 的值符合预期
        assert_allclose(C, [[0], [0]], rtol=1e-13)  # 断言 C 的值符合预期
        assert_allclose(D, [[5], [2]], rtol=1e-13)  # 断言 D 的值符合预期

        num, den = ss2tf(A, B, C, D)  # 将状态空间表示 A, B, C, D 转换回传递函数的系数 num, den
        assert_allclose(num, [[5, 0], [2, 0]], rtol=1e-13)  # 断言转换后的传递函数分子 num 符合预期
        assert_allclose(den, [1, 0], rtol=1e-13)    # 断言转换后的传递函数分母 den 符合预期
    # 定义一个测试函数，用于测试状态空间到传递函数的转换
    def test_simo_round_trip(self):
        # 使用特定的转换函数将给定的传递函数（TF）转换为状态空间（SS），用于测试目的
        tf = ([[1, 2], [1, 1]], [1, 2])
        A, B, C, D = tf2ss(*tf)
        # 断言状态空间矩阵 A 的近似值
        assert_allclose(A, [[-2]], rtol=1e-13)
        # 断言状态空间矩阵 B 的近似值
        assert_allclose(B, [[1]], rtol=1e-13)
        # 断言状态空间矩阵 C 的近似值
        assert_allclose(C, [[0], [-1]], rtol=1e-13)
        # 断言状态空间矩阵 D 的近似值
        assert_allclose(D, [[1], [1]], rtol=1e-13)

        # 使用状态空间矩阵 A、B、C、D 进行反向转换，得到传递函数的分子和分母
        num, den = ss2tf(A, B, C, D)
        # 断言传递函数分子的近似值
        assert_allclose(num, [[1, 2], [1, 1]], rtol=1e-13)
        # 断言传递函数分母的近似值
        assert_allclose(den, [1, 2], rtol=1e-13)

        # 重复上述步骤，使用不同的传递函数 tf
        tf = ([[1, 0, 1], [1, 1, 1]], [1, 1, 1])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-1, -1], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[-1, 0], [0, 0]], rtol=1e-13)
        assert_allclose(D, [[1], [1]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[1, 0, 1], [1, 1, 1]], rtol=1e-13)
        assert_allclose(den, [1, 1, 1], rtol=1e-13)

        tf = ([[1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-2, -3, -4], [1, 0, 0], [0, 1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0], [0]], rtol=1e-13)
        assert_allclose(C, [[1, 2, 3], [1, 2, 3]], rtol=1e-13)
        assert_allclose(D, [[0], [0]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1, 2, 3], [0, 1, 2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 2, 3, 4], rtol=1e-13)

        tf = (np.array([1, [2, 3]], dtype=object), [1, 6])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-6]], rtol=1e-31)
        assert_allclose(B, [[1]], rtol=1e-31)
        assert_allclose(C, [[1], [-9]], rtol=1e-31)
        assert_allclose(D, [[0], [2]], rtol=1e-31)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1], [2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 6], rtol=1e-13)

        tf = (np.array([[1, -3], [1, 2, 3]], dtype=object), [1, 6, 5])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-6, -5], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[1, -3], [-4, -2]], rtol=1e-13)
        assert_allclose(D, [[0], [1]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1, -3], [1, 2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 6, 5], rtol=1e-13)

    # 定义另一个测试函数，用于测试所有整数数组的状态空间到传递函数的转换
    def test_all_int_arrays(self):
        # 定义状态空间的系数矩阵 A、B、C、D
        A = [[0, 1, 0], [0, 0, 1], [-3, -4, -2]]
        B = [[0], [0], [1]]
        C = [[5, 1, 0]]
        D = [[0]]
        # 使用状态空间矩阵 A、B、C、D 进行状态空间到传递函数的转换
        num, den = ss2tf(A, B, C, D)
        # 断言传递函数分子的近似值
        assert_allclose(num, [[0.0, 0.0, 1.0, 5.0]], rtol=1e-13, atol=1e-14)
        # 断言传递函数分母的近似值
        assert_allclose(den, [1.0, 2.0, 4.0, 3.0], rtol=1e-13)
    def test_multioutput(self):
        # Regression test for gh-2669.
        # 对 gh-2669 的回归测试

        # 4 states
        A = np.array([[-1.0, 0.0, 1.0, 0.0],
                      [-1.0, 0.0, 2.0, 0.0],
                      [-4.0, 0.0, 3.0, 0.0],
                      [-8.0, 8.0, 0.0, 4.0]])
        # 定义系统的状态矩阵 A

        # 1 input
        B = np.array([[0.3],
                      [0.0],
                      [7.0],
                      [0.0]])
        # 定义系统的输入矩阵 B

        # 3 outputs
        C = np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [8.0, 8.0, 0.0, 0.0]])
        # 定义系统的输出矩阵 C

        D = np.array([[0.0],
                      [0.0],
                      [1.0]])
        # 定义系统的传递矩阵 D

        # Get the transfer functions for all the outputs in one call.
        # 一次调用获取所有输出的传递函数
        b_all, a = ss2tf(A, B, C, D)

        # Get the transfer functions for each output separately.
        # 分别获取每个输出的传递函数
        b0, a0 = ss2tf(A, B, C[0], D[0])
        b1, a1 = ss2tf(A, B, C[1], D[1])
        b2, a2 = ss2tf(A, B, C[2], D[2])

        # Check that we got the same results.
        # 检查是否得到相同的结果
        assert_allclose(a0, a, rtol=1e-13)
        assert_allclose(a1, a, rtol=1e-13)
        assert_allclose(a2, a, rtol=1e-13)
        assert_allclose(b_all, np.vstack((b0, b1, b2)), rtol=1e-13, atol=1e-14)
class TestLsim:
    digits_accuracy = 7  # 设置精度，用于断言的小数位数

    def lti_nowarn(self, *args):
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            system = lti(*args)  # 创建线性时不变系统对象
        return system

    def test_first_order(self):
        # y' = -y
        # 精确解为 y(t) = exp(-t)
        system = self.lti_nowarn(-1., 1., 1., 0.)  # 创建一阶系统对象
        t = np.linspace(0, 5)
        u = np.zeros_like(t)
        tout, y, x = lsim(system, u, t, X0=[1.0])  # 模拟系统响应
        expected_x = np.exp(-tout)
        assert_almost_equal(x, expected_x)  # 断言结果 x 等于预期结果
        assert_almost_equal(y, expected_x)  # 断言结果 y 等于预期结果

    def test_second_order(self):
        t = np.linspace(0, 10, 1001)
        u = np.zeros_like(t)
        # 二阶系统，具有重复根：x''(t) + 2*x(t) + x(t) = 0.
        # 初始条件为 x(0)=1.0 和 x'(t)=0.0，精确解为 (1-t)*exp(-t).
        system = self.lti_nowarn([1.0], [1.0, 2.0, 1.0])  # 创建二阶系统对象
        tout, y, x = lsim(system, u, t, X0=[1.0, 0.0])  # 模拟系统响应
        expected_x = (1.0 - tout) * np.exp(-tout)
        assert_almost_equal(x[:, 0], expected_x)  # 断言结果 x 的第一列等于预期结果

    def test_integrator(self):
        # 积分器: y' = u
        system = self.lti_nowarn(0., 1., 1., 0.)  # 创建积分器系统对象
        t = np.linspace(0, 5)
        u = t
        tout, y, x = lsim(system, u, t)  # 模拟系统响应
        expected_x = 0.5 * tout**2
        assert_almost_equal(x, expected_x, decimal=self.digits_accuracy)  # 断言结果 x 等于预期结果
        assert_almost_equal(y, expected_x, decimal=self.digits_accuracy)  # 断言结果 y 等于预期结果

    def test_two_states(self):
        # 一个具有两个状态变量、两个输入和一个输出的系统。
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = np.array([1.0, 0.0])
        D = np.zeros((1, 2))

        system = self.lti_nowarn(A, B, C, D)  # 创建具有两个状态变量的系统对象

        t = np.linspace(0, 10.0, 21)
        u = np.zeros((len(t), 2))
        tout, y, x = lsim(system, U=u, T=t, X0=[1.0, 1.0])  # 模拟系统响应
        expected_y = np.exp(-tout)
        expected_x0 = np.exp(-tout)
        expected_x1 = np.exp(-2.0 * tout)
        assert_almost_equal(y, expected_y)  # 断言结果 y 等于预期结果
        assert_almost_equal(x[:, 0], expected_x0)  # 断言结果 x 的第一列等于预期结果
        assert_almost_equal(x[:, 1], expected_x1)  # 断言结果 x 的第二列等于预期结果

    def test_double_integrator(self):
        # 双积分器: y'' = 2u
        A = np.array([[0., 1.], [0., 0.]])
        B = np.array([[0.], [1.]])
        C = np.array([[2., 0.]])
        system = self.lti_nowarn(A, B, C, 0.)  # 创建双积分器系统对象
        t = np.linspace(0, 5)
        u = np.ones_like(t)
        tout, y, x = lsim(system, u, t)  # 模拟系统响应
        expected_x = np.transpose(np.array([0.5 * tout**2, tout]))
        expected_y = tout**2
        assert_almost_equal(x, expected_x, decimal=self.digits_accuracy)  # 断言结果 x 等于预期结果
        assert_almost_equal(y, expected_y, decimal=self.digits_accuracy)  # 断言结果 y 等于预期结果
    def test_jordan_block(self):
        # 非对角化的 A 矩阵
        #   x1' + x1 = x2
        #   x2' + x2 = u
        #   y = x1
        # 当 u = 0 时的精确解是 y(t) = t * exp(-t)
        A = np.array([[-1., 1.], [0., -1.]])  # 定义矩阵 A
        B = np.array([[0.], [1.]])  # 定义向量 B
        C = np.array([[1., 0.]])  # 定义矩阵 C
        system = self.lti_nowarn(A, B, C, 0.)  # 创建线性时不发出警告的系统
        t = np.linspace(0, 5)  # 在 [0, 5] 范围内生成时间向量 t
        u = np.zeros_like(t)  # 创建与 t 相同大小的零向量 u
        tout, y, x = lsim(system, u, t, X0=[0.0, 1.0])  # 使用 lsim 模拟系统响应
        expected_y = tout * np.exp(-tout)  # 计算预期的输出 y(t)
        assert_almost_equal(y, expected_y)  # 断言输出 y 与预期值 expected_y 几乎相等

    def test_miso(self):
        # 一个有两个状态变量、两个输入和一个输出的系统
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])  # 定义矩阵 A
        B = np.array([[1.0, 0.0], [0.0, 1.0]])  # 定义矩阵 B
        C = np.array([1.0, 0.0])  # 定义矩阵 C
        D = np.zeros((1,2))  # 创建零矩阵 D
        system = self.lti_nowarn(A, B, C, D)  # 创建线性时不发出警告的系统

        t = np.linspace(0, 5.0, 101)  # 在 [0, 5] 范围内生成时间向量 t
        u = np.zeros((len(t), 2))  # 创建与 t 大小相同的零输入向量 u
        tout, y, x = lsim(system, u, t, X0=[1.0, 1.0])  # 使用 lsim 模拟系统响应
        expected_y = np.exp(-tout)  # 计算预期的输出 y(t)
        expected_x0 = np.exp(-tout)  # 计算预期的状态变量 x0(t)
        expected_x1 = np.exp(-2.0*tout)  # 计算预期的状态变量 x1(t)
        assert_almost_equal(y, expected_y)  # 断言输出 y 与预期值 expected_y 几乎相等
        assert_almost_equal(x[:,0], expected_x0)  # 断言状态变量 x0 与预期值 expected_x0 几乎相等
        assert_almost_equal(x[:,1], expected_x1)  # 断言状态变量 x1 与预期值 expected_x1 几乎相等

    def test_nonzero_initial_time(self):
        system = self.lti_nowarn(-1., 1., 1., 0.)  # 创建线性时不发出警告的系统
        t = np.linspace(1, 2)  # 在 [1, 2] 范围内生成时间向量 t
        u = np.zeros_like(t)  # 创建与 t 相同大小的零向量 u
        tout, y, x = lsim(system, u, t, X0=[1.0])  # 使用 lsim 模拟系统响应
        expected_y = np.exp(-tout)  # 计算预期的输出 y(t)
        assert_almost_equal(y, expected_y)  # 断言输出 y 与预期值 expected_y 几乎相等

    def test_nonequal_timesteps(self):
        t = np.array([0.0, 1.0, 1.0, 3.0])  # 定义不等间隔的时间步长
        u = np.array([0.0, 0.0, 1.0, 1.0])  # 定义输入向量 u
        # 简单积分器: x'(t) = u(t)
        system = ([1.0], [1.0, 0.0])  # 定义输入输出系统
        with assert_raises(ValueError,
                           match="Time steps are not equally spaced."):  # 断言引发 ValueError 异常，且异常消息包含指定文本
            tout, y, x = lsim(system, u, t, X0=[1.0])  # 使用 lsim 模拟系统响应
class TestImpulse:
    def test_first_order(self):
        # First order system: x'(t) + x(t) = u(t)
        # Exact impulse response is x(t) = exp(-t).
        # 定义一个一阶系统，其微分方程为 x'(t) + x(t) = u(t)，其精确冲激响应为 x(t) = exp(-t)。
        system = ([1.0], [1.0,1.0])
        # 调用 impulse 函数计算系统的冲激响应
        tout, y = impulse(system)
        # 期望输出 y 应当接近于 exp(-tout)
        expected_y = np.exp(-tout)
        # 使用 assert_almost_equal 断言确认计算值与期望值的接近程度
        assert_almost_equal(y, expected_y)

    def test_first_order_fixed_time(self):
        # Specify the desired time values for the output.
        # 指定输出的时间值。

        # First order system: x'(t) + x(t) = u(t)
        # Exact impulse response is x(t) = exp(-t).
        # 定义一个一阶系统，其微分方程为 x'(t) + x(t) = u(t)，其精确冲激响应为 x(t) = exp(-t)。
        system = ([1.0], [1.0,1.0])
        # 创建包含 21 个时间点的线性空间，范围从 0 到 2.0
        n = 21
        t = np.linspace(0, 2.0, n)
        # 使用指定的时间值调用 impulse 函数计算系统的冲激响应
        tout, y = impulse(system, T=t)
        # 断言确保输出的时间向量 tout 的形状为 (n,)
        assert_equal(tout.shape, (n,))
        # 断言确保计算得到的时间向量 tout 与预期的时间向量 t 接近
        assert_almost_equal(tout, t)
        # 计算期望的响应值，即 x(t) = exp(-t)
        expected_y = np.exp(-t)
        # 使用 assert_almost_equal 断言确认计算得到的响应 y 与期望的响应 expected_y 接近
        assert_almost_equal(y, expected_y)

    def test_first_order_initial(self):
        # Specify an initial condition as a scalar.
        # 将初始条件指定为标量。

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact impulse response is x(t) = 4*exp(-t).
        # 定义一个一阶系统，其微分方程为 x'(t) + x(t) = u(t)，初始条件为 x(0)=3.0，其精确冲激响应为 x(t) = 4*exp(-t)。
        system = ([1.0], [1.0,1.0])
        # 调用 impulse 函数计算系统的冲激响应，指定初始条件 X0=3.0
        tout, y = impulse(system, X0=3.0)
        # 计算期望的响应值，即 x(t) = 4*exp(-t)
        expected_y = 4.0 * np.exp(-tout)
        # 使用 assert_almost_equal 断言确认计算得到的响应 y 与期望的响应 expected_y 接近
        assert_almost_equal(y, expected_y)

    def test_first_order_initial_list(self):
        # Specify an initial condition as a list.
        # 将初始条件指定为列表。

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact impulse response is x(t) = 4*exp(-t).
        # 定义一个一阶系统，其微分方程为 x'(t) + x(t) = u(t)，初始条件为 x(0)=3.0，其精确冲激响应为 x(t) = 4*exp(-t)。
        system = ([1.0], [1.0,1.0])
        # 调用 impulse 函数计算系统的冲激响应，初始条件 X0=[3.0]
        tout, y = impulse(system, X0=[3.0])
        # 计算期望的响应值，即 x(t) = 4*exp(-t)
        expected_y = 4.0 * np.exp(-tout)
        # 使用 assert_almost_equal 断言确认计算得到的响应 y 与期望的响应 expected_y 接近
        assert_almost_equal(y, expected_y)

    def test_integrator(self):
        # Simple integrator: x'(t) = u(t)
        # 简单积分器：x'(t) = u(t)
        system = ([1.0], [1.0,0.0])
        # 调用 impulse 函数计算系统的冲激响应
        tout, y = impulse(system)
        # 计算期望的响应值，即 x(t) = 1.0 * t
        expected_y = np.ones_like(tout)
        # 使用 assert_almost_equal 断言确认计算得到的响应 y 与期望的响应 expected_y 接近
        assert_almost_equal(y, expected_y)

    def test_second_order(self):
        # Second order system with a repeated root:
        #     x''(t) + 2*x'(t) + x(t) = u(t)
        # The exact impulse response is t*exp(-t).
        # 具有重复根的二阶系统：
        #     x''(t) + 2*x'(t) + x(t) = u(t)
        # 其精确冲激响应为 t*exp(-t)。
        system = ([1.0], [1.0, 2.0, 1.0])
        # 调用 impulse 函数计算系统的冲激响应
        tout, y = impulse(system)
        # 计算期望的响应值，即 x(t) = t * exp(-t)
        expected_y = tout * np.exp(-tout)
        # 使用 assert_almost_equal 断言确认计算得到的响应 y 与期望的响应 expected_y 接近
        assert_almost_equal(y, expected_y)

    def test_array_like(self):
        # Test that function can accept sequences, scalars.
        # 测试函数能够接受序列和标量。

        system = ([1.0], [1.0, 2.0, 1.0])
        # TODO: add meaningful test where X0 is a list
        # TODO: 添加一个有意义的测试，其中 X0 是一个列表
        tout, y = impulse(system, X0=[3], T=[5, 6])
        tout, y = impulse(system, X0=[3], T=[5])

    def test_array_like2(self):
        system = ([1.0], [1.0, 2.0, 1.0])
        # 调用 impulse 函数计算系统的冲激响应，指定初始条件 X0=3，时间值 T=5
        tout, y = impulse(system, X0=3, T=5)


class TestStep:
    def test_first_order(self):
        # First order system: x'(t) + x(t) = u(t)
        # Exact step response is x(t) = 1 - exp(-t).
        # 定义一个一阶系统，其微分方程为 x'(t) + x(t) = u(t)，其精确阶跃响应为 x(t) = 1 - exp(-t)。
        system = ([1.0], [1.0,1.0])
        # 调用 step 函数计算系统的阶跃响应
        tout, y = step(system)
        # 计算期望的响应值，即 x(t) = 1.0 - exp(-t)
        expected_y = 1.0 - np.exp(-tout)
        # 使用 assert_almost_equal 断言确认计算得到的响应 y 与期望的响应 expected_y 接近
        assert_almost_equal(y, expected_y)
    def test_first_order_fixed_time(self):
        # Specify the desired time values for the output.
        # 指定输出的时间值

        # First order system: x'(t) + x(t) = u(t)
        # Exact step response is x(t) = 1 - exp(-t).
        # 一阶系统: x'(t) + x(t) = u(t)
        # 精确的阶跃响应是 x(t) = 1 - exp(-t).
        system = ([1.0], [1.0,1.0])
        # 定义系统的传递函数系数 ([分子多项式], [分母多项式])
        n = 21
        # 设定时间点数量
        t = np.linspace(0, 2.0, n)
        # 生成从 0 到 2.0 的 n 个等间隔时间点
        tout, y = step(system, T=t)
        # 调用 step 函数计算系统的阶跃响应
        assert_equal(tout.shape, (n,))
        # 断言输出时间点数组的形状为 (n,)
        assert_almost_equal(tout, t)
        # 断言计算得到的时间点数组与预期的时间点数组几乎相等
        expected_y = 1 - np.exp(-t)
        # 计算预期的系统响应值
        assert_almost_equal(y, expected_y)
        # 断言计算得到的系统响应值与预期的系统响应值几乎相等

    def test_first_order_initial(self):
        # Specify an initial condition as a scalar.
        # 将初始条件指定为标量

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact step response is x(t) = 1 + 2*exp(-t).
        # 一阶系统: x'(t) + x(t) = u(t), x(0)=3.0
        # 精确的阶跃响应是 x(t) = 1 + 2*exp(-t).
        system = ([1.0], [1.0,1.0])
        # 定义系统的传递函数系数 ([分子多项式], [分母多项式])
        tout, y = step(system, X0=3.0)
        # 调用 step 函数计算系统的阶跃响应，设置初始条件为 3.0
        expected_y = 1 + 2.0*np.exp(-tout)
        # 计算预期的系统响应值
        assert_almost_equal(y, expected_y)
        # 断言计算得到的系统响应值与预期的系统响应值几乎相等

    def test_first_order_initial_list(self):
        # Specify an initial condition as a list.
        # 将初始条件指定为列表

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact step response is x(t) = 1 + 2*exp(-t).
        # 一阶系统: x'(t) + x(t) = u(t), x(0)=3.0
        # 精确的阶跃响应是 x(t) = 1 + 2*exp(-t).
        system = ([1.0], [1.0,1.0])
        # 定义系统的传递函数系数 ([分子多项式], [分母多项式])
        tout, y = step(system, X0=[3.0])
        # 调用 step 函数计算系统的阶跃响应，设置初始条件为 [3.0]
        expected_y = 1 + 2.0*np.exp(-tout)
        # 计算预期的系统响应值
        assert_almost_equal(y, expected_y)
        # 断言计算得到的系统响应值与预期的系统响应值几乎相等

    def test_integrator(self):
        # Simple integrator: x'(t) = u(t)
        # Exact step response is x(t) = t.
        # 简单积分器: x'(t) = u(t)
        # 精确的阶跃响应是 x(t) = t.
        system = ([1.0],[1.0,0.0])
        # 定义系统的传递函数系数 ([分子多项式], [分母多项式])
        tout, y = step(system)
        # 调用 step 函数计算系统的阶跃响应
        expected_y = tout
        # 计算预期的系统响应值
        assert_almost_equal(y, expected_y)
        # 断言计算得到的系统响应值与预期的系统响应值几乎相等

    def test_second_order(self):
        # Second order system with a repeated root:
        #     x''(t) + 2*x'(t) + x(t) = u(t)
        # The exact step response is 1 - (1 + t)*exp(-t).
        # 二阶系统，具有重复的根:
        #     x''(t) + 2*x'(t) + x(t) = u(t)
        # 精确的阶跃响应是 1 - (1 + t)*exp(-t).
        system = ([1.0], [1.0, 2.0, 1.0])
        # 定义系统的传递函数系数 ([分子多项式], [分母多项式])
        tout, y = step(system)
        # 调用 step 函数计算系统的阶跃响应
        expected_y = 1 - (1 + tout) * np.exp(-tout)
        # 计算预期的系统响应值
        assert_almost_equal(y, expected_y)
        # 断言计算得到的系统响应值与预期的系统响应值几乎相等

    def test_array_like(self):
        # Test that function can accept sequences, scalars.
        # 测试函数能够接受序列、标量

        system = ([1.0], [1.0, 2.0, 1.0])
        # 定义系统的传递函数系数 ([分子多项式], [分母多项式])
        # TODO: add meaningful test where X0 is a list
        # 待办事项: 添加 X0 是列表时的有意义测试

        tout, y = step(system, T=[5, 6])
        # 调用 step 函数计算系统的阶跃响应，指定时间点为 [5, 6]

    def test_complex_input(self):
        # Test that complex input doesn't raise an error.
        # `step` doesn't seem to have been designed for complex input, but this
        # works and may be used, so add regression test.  See gh-2654.
        # 测试复杂输入不会引发错误
        # `step` 函数似乎并没有为复杂输入而设计，但这个用法有效且可能被使用，因此添加回归测试。参见 gh-2654.
        step(([], [-1], 1+0j))
        # 调用 step 函数，传入复杂输入
class TestLti:
    def test_lti_instantiation(self):
        # Test that lti can be instantiated with sequences, scalars.
        # See PR-225.

        # TransferFunction
        # 使用给定的序列和标量实例化 lti 对象，并验证其类型
        s = lti([1], [-1])
        assert_(isinstance(s, TransferFunction))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)

        # ZerosPolesGain
        # 使用给定的空数组、极点和增益实例化 lti 对象，并验证其类型
        s = lti(np.array([]), np.array([-1]), 1)
        assert_(isinstance(s, ZerosPolesGain))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)

        # StateSpace
        # 使用给定的空列表、极点和增益实例化 lti 对象，并验证其类型
        s = lti([], [-1], 1)
        # 使用给定的序列、极点、增益和采样周期实例化 lti 对象，并验证其类型
        s = lti([1], [-1], 1, 3)
        assert_(isinstance(s, StateSpace))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)


class TestStateSpace:
    def test_initialization(self):
        # Check that all initializations work
        # 验证所有初始化方式是否正常工作
        StateSpace(1, 1, 1, 1)
        StateSpace([1], [2], [3], [4])
        StateSpace(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]),
                   np.array([[1, 0]]), np.array([[0]]))

    def test_conversion(self):
        # Check the conversion functions
        # 验证转换函数是否正确
        s = StateSpace(1, 2, 3, 4)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        # 确保复制功能正常工作
        assert_(StateSpace(s) is not s)
        assert_(s.to_ss() is not s)

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_tf() and to_zpk()
        # 测试跨类属性的设置器/获取器，隐含地测试 to_tf() 和 to_zpk() 方法

        # Getters
        # 获取器验证
        s = StateSpace(1, 1, 1, 1)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])
        assert_(s.dt is None)


class TestTransferFunction:
    def test_initialization(self):
        # Check that all initializations work
        # 验证所有初始化方式是否正常工作
        TransferFunction(1, 1)
        TransferFunction([1], [2])
        TransferFunction(np.array([1]), np.array([2]))

    def test_conversion(self):
        # Check the conversion functions
        # 验证转换函数是否正确
        s = TransferFunction([1, 0], [1, -1])
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        # 确保复制功能正常工作
        assert_(TransferFunction(s) is not s)
        assert_(s.to_tf() is not s)

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_ss() and to_zpk()
        # 测试跨类属性的设置器/获取器，隐含地测试 to_ss() 和 to_zpk() 方法

        # Getters
        # 获取器验证
        s = TransferFunction([1, 0], [1, -1])
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])


class TestZerosPolesGain:
    def test_initialization(self):
        # Check that all initializations work
        # 验证所有初始化方式是否正常工作
        ZerosPolesGain(1, 1, 1)
        ZerosPolesGain([1], [2], 1)
        ZerosPolesGain(np.array([1]), np.array([2]), 1)
    def test_conversion(self):
        # 检查转换函数的正确性

        # 创建一个 ZerosPolesGain 对象
        s = ZerosPolesGain(1, 2, 3)
        
        # 断言转换为 StateSpace 类型
        assert_(isinstance(s.to_ss(), StateSpace))
        
        # 断言转换为 TransferFunction 类型
        assert_(isinstance(s.to_tf(), TransferFunction))
        
        # 断言转换为 ZerosPolesGain 类型
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # 确保复制操作有效
        assert_(ZerosPolesGain(s) is not s)
        
        # 确保转换后的对象不是原始对象本身
        assert_(s.to_zpk() is not s)
class Test_abcd_normalize:
    # 设置测试方法的初始化操作
    def setup_method(self):
        # 初始化测试数据 A, B, C, D
        self.A = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.B = np.array([[-1.0], [5.0]])
        self.C = np.array([[4.0, 5.0]])
        self.D = np.array([[2.5]])

    # 测试当没有矩阵输入时抛出 ValueError 异常
    def test_no_matrix_fails(self):
        assert_raises(ValueError, abcd_normalize)

    # 测试当输入的 A 矩阵不是方阵时抛出 ValueError 异常
    def test_A_nosquare_fails(self):
        assert_raises(ValueError, abcd_normalize, [1, -1],
                      self.B, self.C, self.D)

    # 测试当输入的 A 和 B 矩阵维度不匹配时抛出 ValueError 异常
    def test_AB_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    # 测试当输入的 A 和 C 矩阵维度不匹配时抛出 ValueError 异常
    def test_AC_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      [[4.0], [5.0]], self.D)

    # 测试当输入的 C 和 D 矩阵维度不匹配时抛出 ValueError 异常
    def test_CD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      self.C, [2.5, 0])

    # 测试当输入的 B 和 D 矩阵维度不匹配时抛出 ValueError 异常
    def test_BD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    # 测试正常情况下 abcd_normalize 函数返回的 A, B, C, D 矩阵与预期相同
    def test_normalized_matrices_unchanged(self):
        A, B, C, D = abcd_normalize(self.A, self.B, self.C, self.D)
        assert_equal(A, self.A)
        assert_equal(B, self.B)
        assert_equal(C, self.C)
        assert_equal(D, self.D)

    # 测试返回的 A, B, C, D 矩阵的形状符合预期
    def test_shapes(self):
        A, B, C, D = abcd_normalize(self.A, self.B, [1, 0], 0)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape[0], C.shape[1])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(B.shape[1], D.shape[1])

    # 测试输入 B 和 D 矩阵维度为零时，返回的 B 和 D 矩阵保持不变
    def test_zero_dimension_is_not_none1(self):
        B_ = np.zeros((2, 0))
        D_ = np.zeros((0, 0))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, D=D_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(D, D_)
        assert_equal(C.shape[0], D_.shape[0])
        assert_equal(C.shape[1], self.A.shape[0])

    # 测试输入 B 和 C 矩阵维度为零时，返回的 B 和 C 矩阵保持不变
    def test_zero_dimension_is_not_none2(self):
        B_ = np.zeros((2, 0))
        C_ = np.zeros((0, 2))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, C=C_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(C, C_)
        assert_equal(D.shape[0], C_.shape[0])
        assert_equal(D.shape[1], B_.shape[1])

    # 测试缺少 A 矩阵输入时，返回的 A 矩阵形状符合预期
    def test_missing_A(self):
        A, B, C, D = abcd_normalize(B=self.B, C=self.C, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))

    # 测试缺少 B 矩阵输入时，返回的 B 矩阵形状符合预期
    def test_missing_B(self):
        A, B, C, D = abcd_normalize(A=self.A, C=self.C, D=self.D)
        assert_equal(B.shape[0], A.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))
    # 测试函数，用于检查在不同输入参数条件下的函数 abcd_normalize 的行为
    
    def test_missing_C(self):
        # 调用 abcd_normalize 函数，处理参数 A, B, D，并返回处理后的结果 A, B, C, D
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, D=self.D)
        # 断言：C 的行数应与 D 的行数相等
        assert_equal(C.shape[0], D.shape[0])
        # 断言：C 的列数应与 A 的行数相等
        assert_equal(C.shape[1], A.shape[0])
        # 断言：C 的形状应为 (D 的行数, A 的行数)
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))
    
    def test_missing_D(self):
        # 调用 abcd_normalize 函数，处理参数 A, B, C，并返回处理后的结果 A, B, C, D
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, C=self.C)
        # 断言：D 的行数应与 C 的行数相等
        assert_equal(D.shape[0], C.shape[0])
        # 断言：D 的列数应与 B 的列数相等
        assert_equal(D.shape[1], B.shape[1])
        # 断言：D 的形状应为 (C 的行数, B 的列数)
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))
    
    def test_missing_AB(self):
        # 调用 abcd_normalize 函数，处理参数 C, D，并返回处理后的结果 A, B, C, D
        A, B, C, D = abcd_normalize(C=self.C, D=self.D)
        # 断言：A 的行数应等于 A 的列数，即 A 是方阵
        assert_equal(A.shape[0], A.shape[1])
        # 断言：A 的行数应与 B 的行数相等
        assert_equal(A.shape[0], B.shape[0])
        # 断言：B 的列数应与 D 的列数相等
        assert_equal(B.shape[1], D.shape[1])
        # 断言：A 的形状应为 (C 的列数, C 的列数)
        assert_equal(A.shape, (self.C.shape[1], self.C.shape[1]))
        # 断言：B 的形状应为 (C 的列数, D 的列数)
        assert_equal(B.shape, (self.C.shape[1], self.D.shape[1]))
    
    def test_missing_AC(self):
        # 调用 abcd_normalize 函数，处理参数 B, D，并返回处理后的结果 A, B, C, D
        A, B, C, D = abcd_normalize(B=self.B, D=self.D)
        # 断言：A 的行数应等于 A 的列数，即 A 是方阵
        assert_equal(A.shape[0], A.shape[1])
        # 断言：A 的行数应与 B 的行数相等
        assert_equal(A.shape[0], B.shape[0])
        # 断言：C 的行数应与 D 的行数相等
        assert_equal(C.shape[0], D.shape[0])
        # 断言：C 的列数应与 A 的行数相等
        assert_equal(C.shape[1], A.shape[0])
        # 断言：A 的形状应为 (B 的行数, B 的行数)
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        # 断言：C 的形状应为 (D 的行数, B 的行数)
        assert_equal(C.shape, (self.D.shape[0], self.B.shape[0]))
    
    def test_missing_AD(self):
        # 调用 abcd_normalize 函数，处理参数 B, C，并返回处理后的结果 A, B, C, D
        A, B, C, D = abcd_normalize(B=self.B, C=self.C)
        # 断言：A 的行数应等于 A 的列数，即 A 是方阵
        assert_equal(A.shape[0], A.shape[1])
        # 断言：A 的行数应与 B 的行数相等
        assert_equal(A.shape[0], B.shape[0])
        # 断言：D 的行数应与 C 的行数相等
        assert_equal(D.shape[0], C.shape[0])
        # 断言：D 的列数应与 B 的列数相等
        assert_equal(D.shape[1], B.shape[1])
        # 断言：A 的形状应为 (B 的行数, B 的行数)
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        # 断言：D 的形状应为 (C 的行数, B 的列数)
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))
    
    def test_missing_BC(self):
        # 调用 abcd_normalize 函数，处理参数 A, D，并返回处理后的结果 A, B, C, D
        A, B, C, D = abcd_normalize(A=self.A, D=self.D)
        # 断言：B 的行数应与 A 的行数相等
        assert_equal(B.shape[0], A.shape[0])
        # 断言：B 的列数应与 D 的列数相等
        assert_equal(B.shape[1], D.shape[1])
        # 断言：C 的行数应与 D 的行数相等
        assert_equal(C.shape[0], D.shape[0])
        # 断言：C 的列数应与 A 的行数相等
        assert_equal(C.shape[1], A.shape[0])
        # 断言：B 的形状应为 (A 的行数, D 的列数)
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))
        # 断言：C 的形状应为 (D 的行数, A 的行数)
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))
    
    def test_missing_ABC_fails(self):
        # 断言：调用 abcd_normalize 函数，缺少参数 ABC 会引发 ValueError 异常
        assert_raises(ValueError, abcd_normalize, D=self.D)
    
    def test_missing_BD_fails(self):
        # 断言：调用 abcd_normalize 函数，缺少参数 BD 会引发 ValueError 异常
        assert_raises(ValueError, abcd_normalize, A=self.A, C=self.C)
    
    def test_missing_CD_fails(self):
        # 断言：调用 abcd_normalize 函数，缺少参数 CD 会引发 ValueError 异常
        assert_raises(ValueError, abcd_normalize, A=self.A, B=self.B)
class Test_bode:

    def test_01(self):
        # Test bode() magnitude calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        # cutoff: 1 rad/s, slope: -20 dB/decade
        #   H(s=0.1) ~= 0 dB
        #   H(s=1) ~= -3 dB
        #   H(s=10) ~= -20 dB
        #   H(s=100) ~= -40 dB
        # Create a transfer function system representing the low-pass filter
        system = lti([1], [1, 1])
        # Frequencies at which to evaluate the bode plot
        w = [0.1, 1, 10, 100]
        # Compute bode plot (magnitude and phase response)
        w, mag, phase = bode(system, w=w)
        # Expected magnitude values for the given frequencies
        expected_mag = [0, -3, -20, -40]
        # Check if computed magnitude matches expected values within tolerance
        assert_almost_equal(mag, expected_mag, decimal=1)

    def test_02(self):
        # Test bode() phase calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        #   angle(H(s=0.1)) ~= -5.7 deg
        #   angle(H(s=1)) ~= -45 deg
        #   angle(H(s=10)) ~= -84.3 deg
        # Create a transfer function system representing the low-pass filter
        system = lti([1], [1, 1])
        # Frequencies at which to evaluate the bode plot
        w = [0.1, 1, 10]
        # Compute bode plot (magnitude and phase response)
        w, mag, phase = bode(system, w=w)
        # Expected phase values for the given frequencies
        expected_phase = [-5.7, -45, -84.3]
        # Check if computed phase matches expected values within tolerance
        assert_almost_equal(phase, expected_phase, decimal=1)

    def test_03(self):
        # Test bode() magnitude calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # Create a transfer function system representing the low-pass filter
        system = lti([1], [1, 1])
        # Frequencies at which to evaluate the bode plot
        w = [0.1, 1, 10, 100]
        # Compute bode plot (magnitude and phase response)
        w, mag, phase = bode(system, w=w)
        # Calculate magnitude in dB using polynomial evaluation
        jw = w * 1j
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
        expected_mag = 20.0 * np.log10(abs(y))
        # Check if computed magnitude matches expected values
        assert_almost_equal(mag, expected_mag)

    def test_04(self):
        # Test bode() phase calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # Create a transfer function system representing the low-pass filter
        system = lti([1], [1, 1])
        # Frequencies at which to evaluate the bode plot
        w = [0.1, 1, 10, 100]
        # Compute bode plot (magnitude and phase response)
        w, mag, phase = bode(system, w=w)
        # Calculate phase using arctangent function
        jw = w * 1j
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
        expected_phase = np.arctan2(y.imag, y.real) * 180.0 / np.pi
        # Check if computed phase matches expected values
        assert_almost_equal(phase, expected_phase)

    def test_05(self):
        # Test that bode() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # Create a transfer function system representing the low-pass filter
        system = lti([1], [1, 1])
        # Number of points to generate in logarithmic scale
        n = 10
        # Expected frequency range is from 0.01 to 10
        expected_w = np.logspace(-2, 1, n)
        # Compute bode plot (magnitude and phase response)
        w, mag, phase = bode(system, n=n)
        # Check if computed frequency range matches expected values
        assert_almost_equal(w, expected_w)

    def test_06(self):
        # Test that bode() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        # Create a transfer function system representing the integrator
        system = lti([1], [1, 0])
        # Compute bode plot (magnitude and phase response)
        w, mag, phase = bode(system, n=2)
        # Check if the lowest frequency is correctly computed as 0.01
        assert_equal(w[0], 0.01)  # a fail would give not-a-number

    def test_07(self):
        # bode() should not fail on a system with pure imaginary poles.
        # The test passes if bode doesn't raise an exception.
        # Create a transfer function system with pure imaginary poles
        system = lti([1], [1, 0, 100])
        # Compute bode plot (magnitude and phase response)
        w, mag, phase = bode(system, n=2)
    # 定义单元测试函数 test_08，测试 bode() 方法是否返回连续相位，在 issues/2331 中有说明。
    def test_08(self):
        # 创建一个零输入传递函数系统，其特征方程为 s^5 + 10s^4 + 30s^3 + 40s^2 + 60s + 70 = 0
        system = lti([], [-10, -30, -40, -60, -70], 1)
        # 调用 system 的 bode 方法计算频率响应，返回频率数组 w、幅度 mag、相位 phase
        w, mag, phase = system.bode(w=np.logspace(-3, 40, 100))
        # 断言最小相位是否接近 -450，精度为小数点后 15 位
        assert_almost_equal(min(phase), -450, decimal=15)

    # 定义单元测试函数 test_from_state_space，验证从状态空间表示法矩阵 A、B、C、D 创建的系统是否能正常使用 bode 方法。
    def test_from_state_space(self):
        # 定义一个 Butterworth 低通滤波器的系数数组 a
        a = np.array([1.0, 2.0, 2.0, 1.0])
        # 使用 linalg.companion(a).T 创建状态空间矩阵 A
        A = linalg.companion(a).T
        # 定义输入矩阵 B
        B = np.array([[0.0], [0.0], [1.0]])
        # 定义输出矩阵 C
        C = np.array([[1.0, 0.0, 0.0]])
        # 定义传递矩阵 D
        D = np.array([[0.0]])
        
        # 使用 suppress_warnings() 上下文管理器捕获 BadCoefficients 警告
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            # 使用 A、B、C、D 创建 lti 系统对象
            system = lti(A, B, C, D)
            # 调用 bode 方法计算系统的频率响应，返回频率数组 w、幅度 mag、相位 phase
            w, mag, phase = bode(system, n=100)

        # 计算预期的幅度响应，对于 Butterworth 低通滤波器，其精确的频率响应已知
        expected_magnitude = 20 * np.log10(np.sqrt(1.0 / (1.0 + w**6)))
        # 断言计算得到的幅度与预期的幅度响应是否接近
        assert_almost_equal(mag, expected_magnitude)


这段代码是两个 Python 单元测试函数，用于测试控制系统的频率响应函数 `bode()` 的功能。第一个函数 `test_08` 测试了一个零输入传递函数系统的相位响应是否满足特定条件，第二个函数 `test_from_state_space` 则测试了使用状态空间表示法创建的系统的频率响应是否正确。
class Test_freqresp:

    def test_output_manual(self):
        # Test freqresp() output calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        #   re(H(s=0.1)) ~= 0.99
        #   re(H(s=1)) ~= 0.5
        #   re(H(s=10)) ~= 0.0099
        # 创建一个一阶低通滤波器系统 H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        # 设定频率点
        w = [0.1, 1, 10]
        # 调用 freqresp 函数计算频率响应
        w, H = freqresp(system, w=w)
        # 期望的实部
        expected_re = [0.99, 0.5, 0.0099]
        # 期望的虚部
        expected_im = [-0.099, -0.5, -0.099]
        # 检查实部是否接近期望值
        assert_almost_equal(H.real, expected_re, decimal=1)
        # 检查虚部是否接近期望值
        assert_almost_equal(H.imag, expected_im, decimal=1)

    def test_output(self):
        # Test freqresp() output calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # 创建一个一阶低通滤波器系统 H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        # 设定频率点
        w = [0.1, 1, 10, 100]
        # 调用 freqresp 函数计算频率响应
        w, H = freqresp(system, w=w)
        # 计算 s = jω
        s = w * 1j
        # 计算期望的频率响应
        expected = np.polyval(system.num, s) / np.polyval(system.den, s)
        # 检查实部是否接近期望值
        assert_almost_equal(H.real, expected.real)
        # 检查虚部是否接近期望值
        assert_almost_equal(H.imag, expected.imag)

    def test_freq_range(self):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # Expected range is from 0.01 to 10.
        # 创建一个一阶低通滤波器系统 H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        # 设定频率点数量
        n = 10
        # 期望的频率范围
        expected_w = np.logspace(-2, 1, n)
        # 调用 freqresp 函数计算频率响应
        w, H = freqresp(system, n=n)
        # 检查频率点是否接近期望的频率范围
        assert_almost_equal(w, expected_w)

    def test_pole_zero(self):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        # 创建一个具有在 s=0 处的极点的系统，积分器：H(s) = 1 / s
        system = lti([1], [1, 0])
        # 调用 freqresp 函数计算频率响应
        w, H = freqresp(system, n=2)
        # 检查第一个频率点是否接近期望值 0.01
        assert_equal(w[0], 0.01)  # a fail would give not-a-number

    def test_from_state_space(self):
        # Ensure that freqresp works with a system that was created from the
        # state space representation matrices A, B, C, D.  In this case,
        # system.num will be a 2-D array with shape (1, n+1), where (n,n) is
        # the shape of A.
        # A Butterworth lowpass filter is used, so we know the exact
        # frequency response.
        # 确保 freqresp 能够处理从状态空间表示矩阵 A、B、C、D 创建的系统。在这种情况下，
        # system.num 将是一个形状为 (1, n+1) 的二维数组，其中 (n,n) 是 A 的形状。
        # 使用 Butterworth 低通滤波器，因此我们知道确切的频率响应。
        a = np.array([1.0, 2.0, 2.0, 1.0])
        A = linalg.companion(a).T
        B = np.array([[0.0],[0.0],[1.0]])
        C = np.array([[1.0, 0.0, 0.0]])
        D = np.array([[0.0]])
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            # 创建一个由 A、B、C、D 创建的系统
            system = lti(A, B, C, D)
            # 调用 freqresp 函数计算频率响应
            w, H = freqresp(system, n=100)
        # 计算 s = jω
        s = w * 1j
        # 计算期望的频率响应
        expected = (1.0 / (1.0 + 2*s + 2*s**2 + s**3))
        # 检查实部是否接近期望值
        assert_almost_equal(H.real, expected.real)
        # 检查虚部是否接近期望值
        assert_almost_equal(H.imag, expected.imag)

    def test_from_zpk(self):
        # 4th order low-pass filter: H(s) = 1 / (s + 1)
        # 创建一个四阶低通滤波器系统 H(s) = 1 / (s + 1)
        system = lti([], [-1]*4, [1])
        # 设定频率点
        w = [0.1, 1, 10, 100]
        # 调用 freqresp 函数计算频率响应
        w, H = freqresp(system, w=w)
        # 计算 s = jω
        s = w * 1j
        # 计算期望的频率响应
        expected = 1 / (s + 1)**4
        # 检查实部是否接近期望值
        assert_almost_equal(H.real, expected.real)
        # 检查虚部是否接近期望值
        assert_almost_equal(H.imag, expected.imag)
```